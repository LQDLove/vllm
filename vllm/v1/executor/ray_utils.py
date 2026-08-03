# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/ray_utils.py
# 本文件提供 Ray 后端的通用工具：
#   - Ray 集群初始化、Placement Group 创建/校验/等待（initialize_ray_cluster 等）。
#   - RayWorkerWrapper：运行在 Ray actor 内的 worker 包装（含 DAG 执行逻辑）。
#   - FutureWrapper：包装 Ray ObjectRef 以兼容 .result() 接口。
#   - gate_zero_copy / detach_zero_copy：处理 Ray SHM 零拷贝缓冲。
#   - bundle 解析工具：get_bundles_for_indices / get_bundles_sorted_by_node。
# =============================================================================
import os
# 导入 os：读取/设置环境变量（RAY_USAGE_STATS_ENABLED、VLLM_HOST_IP 等）。
import time
# 导入 time：等待 placement group 就绪时的轮询计时。
from collections import defaultdict
# 导入 defaultdict：按节点静态统计 bundle（_verify_bundles 等）。
from concurrent.futures import Future
# 导入 Future：FutureWrapper 的父类。
from typing import TYPE_CHECKING, Union
# 导入类型工具：TYPE_CHECKING（延迟导入）、Union（联合类型标注）。

import numpy as np
# 导入 numpy：检测 ModelRunnerOutput.logprobs 中的 SHM 零拷贝数组并拷贝。

import vllm.platforms
# 导入 vllm.platforms 包（用于访问 current_platform，避免顶层循环导入）。
from vllm.config import ParallelConfig
# 导入 ParallelConfig：ray 集群初始化的参数类型。
from vllm.distributed import get_pp_group
# 导入 PP 通信组（RayWorkerWrapper 中判断 PP rank 时使用）。
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
# 导入 KV 输出聚合器（FutureWrapper 聚合所有 worker 输出时使用）。
from vllm.logger import init_logger
# 导入日志初始化函数。
from vllm.platforms import current_platform
# 导入当前平台抽象。
from vllm.sequence import IntermediateTensors
# 导入 IntermediateTensors：PP 中间张量集合（DAG 节点间传递）。
from vllm.utils.network_utils import get_ip
# 导入获取本机 IP 的工具。
from vllm.v1.outputs import AsyncModelRunnerOutput
# 导入异步模型输出（execute_model_ray 中等待 CUDA event）。
from vllm.v1.serial_utils import run_method
# 导入 run_method：在对象上按方法名/可调用对象执行。
from vllm.v1.worker.worker_base import WorkerWrapperBase
# 导入 worker 包装基类（RayWorkerWrapper 的父类）。

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput
# 仅类型检查时导入调度输出与模型输出类型。

logger = init_logger(__name__)
# 初始化本模块日志。
PG_WAIT_TIMEOUT = 1800
# Placement Group 就绪最长等待时间（秒）= 30 分钟。

# Env vars that are worker-specific and must NOT be copied from the
# driver to Ray workers — they are set per-worker after GPU discovery.
# 注释：这些是 worker 专属环境变量，绝不能从 driver 复制给 Ray worker；
# 它们在 GPU 发现后按 worker 单独设置。
WORKER_SPECIFIC_ENV_VARS: set[str] = {
    "VLLM_HOST_IP",
    # 主机 IP（每 worker 节点不同）。
    "VLLM_HOST_PORT",
    # 主机端口。
    "VLLM_NIXL_SIDE_CHANNEL_HOST",
    # NIXL（KV 迁移库）侧信道主机。
    "LOCAL_RANK",
    # 本地 rank（由 worker 内部决定）。
    "CUDA_VISIBLE_DEVICES",
    # CUDA 可见设备（由 Ray/vLLM 管理）。
    "HIP_VISIBLE_DEVICES",
    # AMD HIP 可见设备。
    "ROCR_VISIBLE_DEVICES",
    # ROCm 可见设备。
}
# worker 专属环境变量集合。

try:
    import ray
    # 尝试导入 Ray。
    from ray.util import placement_group_table
    # 导入 placement group 状态表函数。
    from ray.util.placement_group import PlacementGroup
    # 导入 PlacementGroup 类型。

    try:
        from ray._private.state import available_resources_per_node
    except ImportError:
        # Ray 2.9.x doesn't expose `available_resources_per_node`
        # 注释：Ray 2.9.x 不暴露 available_resources_per_node。
        from ray._private.state import state as _state

        available_resources_per_node = _state._available_resources_per_node
    # 兼容不同 Ray 版本的资源查询 API。

    class RayWorkerWrapper(WorkerWrapperBase):
        # =====================================================================
        # RayWorkerWrapper：Ray actor 内的 worker 包装。
        # 允许在 Ray 设置好 CUDA_VISIBLE_DEVICES 之后惰性初始化底层 Worker。
        # =====================================================================
        """Ray wrapper for vllm.worker.Worker, allowing Worker to be
        lazily initialized after Ray sets CUDA_VISIBLE_DEVICES."""
        # 文档字符串：Ray 包装器，允许在 Ray 设置 CUDA_VISIBLE_DEVICES 后
        # 惰性初始化 vllm.worker.Worker。

        def __init__(self, *args, **kwargs) -> None:
            # 构造函数。
            super().__init__(*args, **kwargs)
            # 调用父类构造。
            # Since the compiled DAG runs a main execution
            # in a different thread that calls cuda.set_device.
            # The flag indicates is set_device is called on
            # that thread.
            # 注释：由于编译 DAG 在另一线程中执行主流程并调用 cuda.set_device，
            # 此标志表示是否已在该线程调用 set_device。
            self.compiled_dag_cuda_device_set = False
            # 编译 DAG 线程是否已设置 CUDA 设备。

        rpc_rank: int
        # 类型标注：RPC rank。

        def adjust_rank(self, rank_mapping: dict[int, int]) -> None:
            # -----------------------------------------------------------------
            # 根据映射调整 rpc_rank（executor 初始化期间调用）。
            # -----------------------------------------------------------------
            """
            Adjust the rpc_rank based on the given mapping.
            It is only used during the initialization of the executor,
            to adjust the rpc_rank of workers after we create all workers.
            """
            # 文档字符串：按给定映射调整 rpc_rank；仅 executor 初始化时使用，
            # 用于创建完所有 worker 后重排 rank。
            if self.rpc_rank in rank_mapping:
                # 若本 worker 的 rank 在映射中。
                self.rpc_rank = rank_mapping[self.rpc_rank]
                # 更新为调整后的 rank。

        def execute_method(self, method: str | bytes, *args, **kwargs):
            # -----------------------------------------------------------------
            # 执行 worker 方法（Ray actor 远程调用入口）。
            # -----------------------------------------------------------------
            try:
                return run_method(self, method, args, kwargs)
                # 用 run_method 执行（字符串名 getattr / bytes 反序列化）。
            except Exception as e:
                # if the driver worker also execute methods,
                # exceptions in the rest worker may cause deadlock in rpc
                # see https://github.com/vllm-project/vllm/issues/3455
                # 注释：若 driver worker 也在执行方法，其他 worker 的异常可能
                # 导致 RPC 死锁（见 issue #3455）。
                msg = (
                    f"Error executing method {method!r}. "
                    "This might cause deadlock in distributed execution."
                )
                # 错误消息。
                logger.exception(msg)
                # 记录日志。
                raise e
                # 重新抛出。

        def get_node_ip(self) -> str:
            # -----------------------------------------------------------------
            # 返回 worker 所在节点 IP。
            # -----------------------------------------------------------------
            return get_ip()
            # 用本机 IP 工具。

        def get_node_and_physical_gpu_ids(self) -> tuple[str, list[int]]:
            # -----------------------------------------------------------------
            # 返回 (node_id, 物理 GPU ID 列表)。
            # -----------------------------------------------------------------
            node_id = ray.get_runtime_context().get_node_id()
            # 取节点 ID。
            device_key = vllm.platforms.current_platform.ray_device_key
            # 平台设备资源键。
            if not device_key:
                # 平台不支持。
                raise RuntimeError(
                    "current platform %s does not support ray.",
                    vllm.platforms.current_platform.device_name,
                )
                # 报错。
            physical_gpu_ids = ray.get_runtime_context().get_accelerator_ids()[
                device_key
            ]
            # 取加速器 ID。
            return node_id, physical_gpu_ids
            # 返回。

        def setup_device_if_necessary(self):
            # -----------------------------------------------------------------
            # 若必要则设置当前 CUDA 设备（编译 DAG 后台线程需要）。
            # -----------------------------------------------------------------
            # TODO(swang): This is needed right now because Ray CG executes
            # on a background thread, so we need to reset torch's current
            # device.
            # We can remove this API after it is fixed in compiled graph.
            # 注释（TODO swang）：Ray CG 在后台线程执行，因此需重置 torch 当前
            # 设备；编译图修复后可移除该 API。
            assert self.worker is not None, "Worker is not initialized"
            # 断言 worker 已初始化。
            if not self.compiled_dag_cuda_device_set:
                # 若尚未设置。
                if current_platform.is_tpu():
                    # TPU 无需设置。
                    pass
                    # 跳过。
                else:
                    assert self.worker.device is not None
                    # 断言设备存在。
                    current_platform.set_device(self.worker.device)
                    # 设置为 worker 设备。
                self.compiled_dag_cuda_device_set = True
                # 置位标志。

        def execute_model_ray(
            self,
            execute_model_input: tuple["SchedulerOutput", "GrammarOutput"]
            | tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"],
        ) -> Union[
            "ModelRunnerOutput",
            tuple["SchedulerOutput", "GrammarOutput", "IntermediateTensors"],
        ]:
            # -----------------------------------------------------------------
            # Ray 编译图调用的模型执行方法（定义 DAG 节点的执行体）。
            # 输入可能是 (调度输出, grammar) 或带中间张量的三元组。
            # -----------------------------------------------------------------
            # This method is used by Ray Compiled Graph to execute the model,
            # and it needs a special logic of self.setup_device_if_necessary()
            # 注释：本方法被 Ray 编译图调用，需要先处理设备设置。
            self.setup_device_if_necessary()
            # 确保当前线程 CUDA 设备正确。
            assert self.worker is not None, "Worker is not initialized"
            # 断言 worker 就绪。
            if len(execute_model_input) == 3:
                # 若带中间张量（非首个 PP stage）。
                scheduler_output, grammar_output, intermediate_tensors = (
                    execute_model_input
                )
                # 解包三元组。
            else:
                scheduler_output, grammar_output = execute_model_input
                # 解包二元组。
                intermediate_tensors = None
                # 无中间张量。
            assert self.worker.model_runner is not None
            # 断言模型运行器存在。
            output = self.worker.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            # 执行模型（传入调度输出与（可选）中间张量）。
            if self._is_intermediate_tensors(output):
                # 若输出是中间张量（PP 中段 stage）。
                if (
                    self.worker.model_runner.supports_mm_inputs
                    and get_pp_group().is_first_rank
                ):
                    # Strip mm_features before Ray forwards it to the next PP Stage.
                    # PP Stage>0 only needs the intermediate tensors,
                    # not preprocessed multimodal data.
                    # 注释：在 Ray 转发给下一 PP stage 之前剥离 mm_features。
                    # PP stage>0 只需要中间张量，不需要预处理的多模态数据。
                    # scheduled_new_reqs is a required field of SchedulerOutput,
                    # so accessing it directly will raise AttributeError if missing.
                    # 注释：scheduled_new_reqs 是 SchedulerOutput 的必填字段，
                    # 缺失时直接访问会抛 AttributeError。
                    for req in scheduler_output.scheduled_new_reqs:
                        # 遍历新调度的请求。
                        req.mm_features = []
                        # 清空多模态特征（减小传输开销）。
                return scheduler_output, grammar_output, output
                # 返回 (调度输出, grammar, 中间张量) 供下一 stage 消费。

            if isinstance(output, AsyncModelRunnerOutput):
                # 若为异步输出。
                output = output.get_output()
                # 等待并取出最终输出（同步 CUDA event）。
            if not self._is_last_rank():
                # 非最后 PP rank。
                # Case where there are no scheduled requests
                # but may still be finished requests.
                # 注释：当无调度请求但可能仍有完成请求时的情形。
                assert not output or not output.req_ids
                # 断言无调度输出或输出中无请求 ID。
                output = scheduler_output, grammar_output, None
                # 中间节点无实际输出，传递调度信息。
            elif output is None:
                # 最后 rank 且输出为 None（需要采样）。
                output = self.worker.model_runner.sample_tokens(grammar_output)
                # 执行采样生成 token。
                # Ensure outputs crossing Ray compiled DAG are serializable.
                # AsyncModelRunnerOutput holds CUDA events and cannot be
                # pickled.
                # 注释：确保跨 Ray 编译 DAG 的输出可序列化。
                # AsyncModelRunnerOutput 持有 CUDA event，无法被 pickle。
                if isinstance(output, AsyncModelRunnerOutput):
                    # 若采样结果仍是异步输出。
                    output = output.get_output()
                    # 转为最终输出。
            return output
            # 返回最终模型输出。

        def override_env_vars(self, vars: dict[str, str]):
            # -----------------------------------------------------------------
            # 覆盖 worker 环境变量。
            # -----------------------------------------------------------------
            os.environ.update(vars)
            # 批量更新环境变量。

        def _is_intermediate_tensors(self, output) -> bool:
            # -----------------------------------------------------------------
            # 判断输出是否为 PP 中间张量。
            # -----------------------------------------------------------------
            return isinstance(output, IntermediateTensors)
            # 类型判断。

        def _is_last_rank(self) -> bool:
            # -----------------------------------------------------------------
            # 是否最后 PP rank。
            # -----------------------------------------------------------------
            return get_pp_group().is_last_rank
            # 查询 PP 组属性。

    ray_import_err = None
    # ray 导入成功，错误为 None。

except ImportError as e:
    # 若导入 ray 失败。
    ray = None  # type: ignore
    # ray 置 None。
    # only capture string to avoid variable references in the traceback that can
    # prevent garbage collection in some cases
    # 注释：仅捕获字符串，避免 traceback 中的变量引用阻碍垃圾回收。
    ray_import_err = str(e)
    # 保存导入错误字符串。
    RayWorkerWrapper = None  # type: ignore
    # 包装类置 None。


def detach_zero_copy_from_model_runner_output(output: "ModelRunnerOutput") -> None:
    # =========================================================================
    # 从 ModelRunnerOutput 中原位分离 Ray SHM 零拷贝缓冲。
    # Ray 编译 DAG 的 SHM 通道可能返回零拷贝对象（如 np.ndarray），
    # 若跨调度迭代持续持有会阻塞通道读取。
    # =========================================================================
    """Detach Ray SHM-channel zero-copy buffers from a ModelRunnerOutput in-place.

    Ray compiled DAG SHM channels may return zero-copy objects (e.g. `np.ndarray`)
    backed by Ray's shared-memory object store. Ray's channel docs explicitly
    warn that subsequent reads may block if such an object is still in scope.

    vLLM can return numpy-backed logprobs and routed experts in
    `ModelRunnerOutput`. If those arrays are backed by Ray SHM (commonly
    read-only), retaining them in scope across scheduler iterations can stall
    the channel and eventually hit `RAY_CGRAPH_get_timeout`.

    Copy read-only numpy arrays so the returned output no longer retains
    references to Ray's shared-memory buffers.

    We intentionally do not touch `prompt_logprobs_dict`: those entries are
    `LogprobsTensors` backed by PyTorch-owned CPU tensors (`to_cpu_nonblocking`
    or `empty_cpu`), not NumPy views decoded from Ray channels.
    """
    # 文档字符串：说明为什么需要分离零拷贝缓冲——Ray SHM 通道返回的 numpy 数组
    # 可能仍引用 Ray 共享内存对象存储；若跨调度迭代持有会阻塞通道读取并触发
    # RAY_CGRAPH_get_timeout。只读 numpy 数组会被拷贝；故意不动 prompt_logprobs_dict：
    # 它由 PyTorch 自有 CPU 张量支撑，不是 Ray 通道的解码视图。
    if output.logprobs is None:
        # 若无 logprobs。
        return
        # 直接返回。

    token_ids, logprobs, ranks, cu_num_generated_tokens = output.logprobs
    # 解包 logprobs 元组。

    def _copy_if_readonly(arr):
        # 辅助函数：只读 numpy 数组则拷贝。
        if isinstance(arr, np.ndarray) and not arr.flags.writeable:
            # 只读数组。
            return arr.copy()
            # 拷贝。
        return arr
        # 否则原样返回。

    # `cu_num_generated_tokens` is already a plain Python list (or None), so it
    # never aliases Ray SHM buffers and can be reused as-is.
    # 注释：cu_num_generated_tokens 已是纯 Python 列表（或 None），不会别名
    # Ray SHM 缓冲，可直接复用。
    token_ids_c = _copy_if_readonly(token_ids)
    # 拷贝（如需）token_ids。
    logprobs_c = _copy_if_readonly(logprobs)
    # 拷贝（如需）logprobs。
    ranks_c = _copy_if_readonly(ranks)
    # 拷贝（如需）ranks。
    if token_ids_c is token_ids and logprobs_c is logprobs and ranks_c is ranks:
        # 若无任何拷贝发生。
        return
        # 直接返回。

    output.logprobs = type(output.logprobs)(
        token_ids_c, logprobs_c, ranks_c, cu_num_generated_tokens
    )
    # 用拷贝后的数组重建同名 logprobs 对象。


class FutureWrapper(Future):
    # =========================================================================
    # FutureWrapper：包装 Ray 输出引用，满足 .execute_model() 的接口要求。
    # 顶层（core busy loop）期望 .result() 阻塞并返回单个输出。
    # =========================================================================
    """A wrapper around Ray output reference to meet the interface
    of .execute_model(): The top level (core busy loop) expects .result() api
    to block and return a single output.

    If aggregator is provided, the outputs from all workers are aggregated upon
    the result() call. If not only the first worker's output is returned.
    """
    # 文档字符串：包装 Ray 输出引用以满足 .execute_model() 的接口；
    # 顶层期望 .result() 阻塞并返回单个输出。提供聚合器时 result() 时聚合
    # 所有 worker 输出，否则只返回第一个 worker 的输出。

    def __init__(self, ref_or_refs, aggregator: KVOutputAggregator | None = None):
        # 构造函数。
        super().__init__()
        # 父类构造。
        self.ref_or_refs = ref_or_refs
        # 保存 ObjectRef 或 ObjectRef 列表。
        self.aggregator = aggregator
        # 保存聚合器。

    def result(self, timeout=None):
        # -------------------------------------------------------------------
        # 阻塞获取结果；有聚合器时聚合所有输出。
        # -------------------------------------------------------------------
        outputs = ray.get(self.ref_or_refs, timeout=timeout)
        # 获取输出值（支持超时）。
        if self.aggregator is None:
            # 无聚合器。
            detach_zero_copy_from_model_runner_output(outputs)
            # 分离零拷贝缓冲。
            return outputs
            # 返回单个输出。

        for output in outputs:
            # 遍历。
            detach_zero_copy_from_model_runner_output(output)
            # 分离缓冲。
        return self.aggregator.aggregate(outputs, output_rank=0)
        # 聚合所有 worker 输出。


def ray_is_available() -> bool:
    # =========================================================================
    # 判断 Ray 是否可用。
    # =========================================================================
    """Returns True if Ray is available."""
    # 文档字符串：若 Ray 可用返回 True。
    return ray is not None
    # 判断模块非 None。


def assert_ray_available():
    # =========================================================================
    # 断言 Ray 可用，否则报错并提示安装。
    # =========================================================================
    """Raise an exception if Ray is not available."""
    # 文档字符串：若 Ray 不可用则抛异常。
    if ray is None:
        # 若未安装。
        raise ValueError(
            f"Failed to import Ray: {ray_import_err}."
            "Please install Ray with `pip install ray`."
        )
        # 报错。


def _verify_bundles(
    placement_group: "PlacementGroup",
    # placement group。
    parallel_config: ParallelConfig,
    # 并行配置。
    device_str: str,
    # 设备资源键字符串。
    require_gpu_on_driver: bool = True,
    # 是否要求 driver 节点有 GPU。
):
    # =========================================================================
    # 校验 placement group 的 bundle 位置是否满足约束。
    # 规则：① TP worker 无法放入单节点时告警；② driver 节点不在 PG 内时报错。
    # =========================================================================
    """Verify a given placement group has bundles located in the right place.

    There are 2 rules.
    - Warn if all tensor parallel workers cannot fit in a single node.
    - Fail if driver node is not included in a placement group
      (only when require_gpu_on_driver is True).
    """
    # 文档字符串：校验 PG 的 bundle 位置。两条规则——
    # ①全部 TP worker 装不进单节点时告警；②driver 节点不在 PG 内时报错。
    assert ray.is_initialized(), (
        "Ray is not initialized although distributed-executor-backend is ray."
    )
    # 断言 Ray 已初始化。
    pg_data = placement_group_table(placement_group)
    # 取 PG 状态表。
    # bundle_idx -> node_id
    # 注释：bundle 索引 → 节点 ID。
    bundle_to_node_ids = pg_data["bundles_to_node_id"]
    # 取映射。
    # bundle_idx -> bundle (e.g., {"GPU": 1})
    # 注释：bundle 索引 → 资源描述（如 {"GPU": 1}）。
    bundles = pg_data["bundles"]
    # 取 bundle 列表。
    # node_id -> List of bundle (e.g., {"GPU": 1})
    # 注释：节点 ID → bundle 列表。
    node_id_to_bundle: dict[str, list[dict[str, float]]] = defaultdict(list)
    # 初始化归集。

    for bundle_idx, node_id in bundle_to_node_ids.items():
        # 遍历映射。
        node_id_to_bundle[node_id].append(bundles[bundle_idx])
        # 归集到节点。
    driver_node_id = ray.get_runtime_context().get_node_id()
    # driver 节点 ID。

    if require_gpu_on_driver and driver_node_id not in node_id_to_bundle:
        # 若要求 driver 有 GPU 但不在 PG 内。
        raise RuntimeError(
            f"driver node id {driver_node_id} is not included in a placement "
            f"group {placement_group.id}. Node id -> bundles "
            f"{node_id_to_bundle}. "
            "You don't have enough GPUs available in a current node. Check "
            "`ray status` and `ray list nodes` to see if you have available "
            "GPUs in a node `{driver_node_id}` before starting an vLLM engine."
        )
        # 报错并提示检查资源。

    for node_id, bundles in node_id_to_bundle.items():
        # 遍历节点。
        if len(bundles) < parallel_config.tensor_parallel_size:
            # 若该节点 bundle 数 < TP 大小。
            logger.warning(
                "tensor_parallel_size=%d "
                "is bigger than a reserved number of %ss (%d "
                "%ss) in a node %s. Tensor parallel workers can be "
                "spread out to 2+ nodes which can degrade the performance "
                "unless you have fast interconnect across nodes, like "
                "Infiniband. To resolve this issue, make sure you have more "
                "than %d GPUs available at each node.",
                parallel_config.tensor_parallel_size,
                # TP 大小。
                device_str,
                # 设备名。
                len(bundles),
                # 节点内数量。
                device_str,
                # 设备名。
                node_id,
                # 节点。
                parallel_config.tensor_parallel_size,
                # 建议的最小 GPU 数。
            )
            # 告警：TP worker 跨节点可能性能退化。


def build_actor_name(
    instance_id: str,
    # 实例 ID。
    rank: int,
    # 全局 rank。
    tp_size: int,
    # TP 大小。
    pp_size: int,
    # PP 大小。
    pcp_size: int,
    # PCP 大小。
) -> str:
    # =========================================================================
    # 构建描述性 Ray actor 名（dashboard 可见）。
    # =========================================================================
    """Build a descriptive Ray actor name for dashboard visibility."""
    # 文档字符串：为 dashboard 可见性构建描述性 actor 名。
    name = f"vllm_Worker_{instance_id}"
    # 基础名。
    if tp_size > 1:
        # 若 TP>1。
        name += f"_TP{rank % tp_size}"
        # 追加 TP rank。
    if pp_size > 1:
        # 若 PP>1。
        name += f"_PP{(rank // tp_size) % pp_size}"
        # 追加 PP rank。
    if pcp_size > 1:
        # 若 PCP>1。
        name += f"_PCP{rank // (tp_size * pp_size)}"
        # 追加 PCP rank。
    return name
    # 返回。


def get_bundles_for_indices(
    placement_group: "PlacementGroup",
    # PG。
    bundle_indices: list[int],
    # 显式 bundle 索引。
    world_size: int,
    # world_size。
) -> list[tuple[int, str, str]]:
    # =========================================================================
    # 按显式 bundle 索引返回 (bundle_id, node_id, node_ip) 列表。
    # =========================================================================
    """
    Return GPU bundle indices paired with node IDs and node IPs for
    explicit bundle indices specified via VLLM_RAY_BUNDLE_INDICES.
    """
    # 文档字符串：为 VLLM_RAY_BUNDLE_INDICES 指定的索引返回 (索引, 节点ID, 节点IP)。
    assert len(bundle_indices) == world_size, (
        "VLLM_RAY_BUNDLE_INDICES must have the same size"
        f" as the world size, but got {bundle_indices=} "
        f"and {world_size=}"
    )
    # 断言索引数量与 world_size 一致。
    assert len(set(bundle_indices)) == len(bundle_indices), (
        "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
        f" but got {bundle_indices=}"
    )
    # 断言无重复。

    pg_data = placement_group_table(placement_group)
    # 取 PG 状态表。
    pg_bundle_to_node = pg_data["bundles_to_node_id"]
    # bundle → 节点映射。
    node_id_to_ip = {
        n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]
    }
    # 构建 节点ID → 节点IP 映射。
    return [
        (bid, pg_bundle_to_node[bid], node_id_to_ip[pg_bundle_to_node[bid]])
        # 组装三元组。
        for bid in bundle_indices
        # 遍历索引。
    ]
    # 返回。


def get_bundles_sorted_by_node(
    placement_group: "PlacementGroup",
) -> list[tuple[int, str, str]]:
    # =========================================================================
    # 返回与节点配对的 GPU bundle 索引（按 driver 优先排序）。
    # 必须从 driver 节点调用。
    # =========================================================================
    """
    Return GPU bundle indices paired with node IDs and node IPs,
    sorted driver-first.

    This utility has to be invoked from the driver node.

    Example: 3-node cluster, driver on node-A, PG bundles spread
    across nodes:

      Input: [
          (0, node-C),
          (1, node-A),
          (2, node-B),
          (3, node-C),
          (4, node-A),
          (5, node-B),
      ]
      Output: [
          (1, node-A),
          (4, node-A),
          (2, node-B),
          (5, node-B),
          (0, node-C),
          (3, node-C),
      ]
    """
    # 文档字符串：返回按 driver 优先排序的 (bundle, node_id, node_ip)。
    # 示例：3 节点集群、driver 在 node-A 时，输出先排 driver 节点（A）、
    # 再排 B、C，且各节点内 bundle 连续。
    pg_data = placement_group_table(placement_group)
    # 取 PG 状态表。
    bundle_to_node = pg_data["bundles_to_node_id"]
    # bundle → 节点映射。

    ray_device_key = current_platform.ray_device_key
    # 平台设备资源键。
    if not ray_device_key:
        # 平台不支持。
        raise ValueError(
            f"current platform {current_platform.device_name} does not support ray."
        )
        # 报错。

    node_id_to_ip = {
        n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]
    }
    # 节点ID → IP 映射。

    bundle_specs = placement_group.bundle_specs
    # 取 bundle 规格。
    assert bundle_specs is not None
    # 断言非空。
    bundle_to_node_id: list[tuple[int, str, str]] = []
    # 初始化结果。
    for bundle_idx, bundle in enumerate(bundle_specs):
        # 遍历。
        if bundle.get(ray_device_key):
            # 若含设备资源。
            node_id = bundle_to_node.get(bundle_idx)
            # 取节点。
            bundle_to_node_id.append((bundle_idx, node_id, node_id_to_ip[node_id]))
            # 记录。

    driver_node = ray.get_runtime_context().get_node_id()
    # driver 节点。

    def _sort_key(item):
        # 排序键。
        _, node_id, _ = item
        # 取节点。
        return (0 if node_id == driver_node else 1, node_id)
        # driver 优先，再按节点 ID。

    bundle_to_node_id.sort(key=_sort_key)
    # 排序。

    return bundle_to_node_id
    # 返回。


def _wait_until_pg_ready(current_placement_group: "PlacementGroup"):
    # =========================================================================
    # 等待 placement group 就绪；超时前打印提示信息。
    # =========================================================================
    """Wait until a placement group is ready.

    It prints the informative log messages if the placement group is
    not created within time.

    """
    # 文档字符串：等待 PG 就绪；超时未创建则打印信息日志。
    # Wait until PG is ready - this will block until all
    # requested resources are available, and will time out
    # if they cannot be provisioned.
    # 注释：阻塞直到所需资源可用；无法供应时超时。
    placement_group_specs = current_placement_group.bundle_specs
    # 取 bundle 规格。

    s = time.time()
    # 记录开始时间。
    pg_ready_ref = current_placement_group.ready()
    # 获取就绪引用。
    wait_interval = 10
    # 初始等待间隔。
    while time.time() - s < PG_WAIT_TIMEOUT:
        # 在超时窗口内。
        ready, _ = ray.wait([pg_ready_ref], timeout=wait_interval)
        # 等待就绪（最多 wait_interval 秒）。
        if len(ready) > 0:
            # 就绪。
            break
            # 退出循环。

        # Exponential backoff for warning print.
        # 注释：告警打印指数退避。
        wait_interval *= 2
        # 间隔翻倍。
        logger.info(
            "Waiting for creating a placement group of specs for "
            "%d seconds. specs=%s. Check `ray status` and "
            "`ray list nodes` to see if you have enough resources,"
            " and make sure the IP addresses used by ray cluster"
            " are the same as VLLM_HOST_IP environment variable"
            " specified in each node if you are running on a multi-node.",
            int(time.time() - s),
            # 已等待秒数。
            placement_group_specs,
            # 规格。
        )
        # 提示等待。

    try:
        ray.get(pg_ready_ref, timeout=0)
        # 立即取结果（若未就绪会抛 GetTimeoutError）。
    except ray.exceptions.GetTimeoutError:
        # Provide more helpful error message when GPU count is exceeded
        # 注释：GPU 数量超出时提供更有帮助的错误信息。
        total_gpu_required = sum(spec.get("GPU", 0) for spec in placement_group_specs)
        # 计算所需 GPU 总数。
        # If more than one GPU is required for the placement group, provide a
        # more specific error message.
        # We use >1 here because multi-GPU (tensor parallel) jobs are more
        # likely to fail due to insufficient cluster resources, and users may
        # need to adjust tensor_parallel_size to fit available GPUs.
        # 注释：若 PG 需要多 GPU 则提供更具体的错误；用 >1 是因为多 GPU
        # （张量并行）任务更可能因资源不足失败，用户可能需要调整 TP 大小。
        if total_gpu_required > 1:
            # 多 GPU。
            raise ValueError(
                f"Cannot provide a placement group requiring "
                f"{total_gpu_required} GPUs "
                f"(placement_group_specs={placement_group_specs}) within "
                f"{PG_WAIT_TIMEOUT} seconds.\n"
                f"Tensor parallel size may exceed available GPUs in your "
                f"cluster. Check resources with `ray status` and "
                f"`ray list nodes`.\n"
                f"If running on K8s with limited GPUs, consider reducing "
                f"--tensor-parallel-size to match available GPU resources."
            ) from None
            # 报错：TP 大小可能超过集群 GPU 数量。
        else:
            raise ValueError(
                "Cannot provide a placement group of "
                f"{placement_group_specs=} within "
                f"{PG_WAIT_TIMEOUT} seconds. See "
                "`ray status` and `ray list nodes` to make sure the cluster "
                "has enough resources."
            ) from None
            # 通用资源不足错误。


def _wait_until_pg_removed(current_placement_group: "PlacementGroup"):
    # =========================================================================
    # 移除 placement group 并等待完全移除。
    # =========================================================================
    ray.util.remove_placement_group(current_placement_group)
    # 请求移除 PG。
    s = time.time()
    # 开始时间。
    wait_interval = 10
    # 等待间隔。
    while time.time() - s < PG_WAIT_TIMEOUT:
        # 超时窗口内。
        pg = ray.util.get_current_placement_group()
        # 查当前 PG。
        if pg is None:
            # 已移除。
            break
            # 退出。

        # Exponential backoff for warning print.
        # 注释：告警指数退避。
        wait_interval *= 2
        # 间隔翻倍。
        logger.info(
            "Waiting for removing a placement group of specs for %d seconds.",
            int(time.time() - s),
        )
        # 提示。
        time.sleep(wait_interval)
        # 休眠。


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    # 并行配置。
    ray_address: str | None = None,
    # Ray 地址（默认 None 使用默认）。
    require_gpu_on_driver: bool = True,
    # 是否要求 driver 节点有 GPU（RayExecutorV2 设为 False）。
):
    # =========================================================================
    # 初始化分布式集群（Ray）：
    # 连接/创建 Ray 集群，创建或复用 placement group。
    # =========================================================================
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
        require_gpu_on_driver: If True (default), require at least one GPU
            on the current (driver) node and pin the first PG bundle to it.
            Set to False for executors like RayExecutorV2 where all GPU work
            is delegated to remote Ray actors.
    """
    # 文档字符串：用 Ray 初始化分布式集群——连接/创建 Ray 并创建 PG；
    # require_gpu_on_driver=True 时要求 driver 节点至少 1 张 GPU 并把首个
    # PG bundle 固定到它；如 RayExecutorV2 等所有计算交给远程 actor 的执行器
    # 应设为 False。
    assert_ray_available()
    # 断言 Ray 可用。
    from vllm.platforms import current_platform
    # 延迟导入平台。

    # Disable Ray usage stats collection
    # 注释：禁用 Ray 使用统计收集。
    if os.environ.get("RAY_USAGE_STATS_ENABLED", "0") != "1":
        # 未显式开启。
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
        # 置 0。

    # Prevalidate GPU requirements before Ray processing
    # 注释：在 Ray 处理前预校验 GPU 需求。
    if current_platform.is_cuda() and parallel_config.world_size > 1:
        # CUDA 且多卡。
        available_gpus = current_platform.device_count()
        # 本机 GPU 数。
        if parallel_config.world_size > available_gpus:
            # 需求超本地数量（可能靠集群其他节点满足）。
            logger.warning(
                "Tensor parallel size (%d) exceeds available GPUs (%d). "
                "This may result in Ray placement group allocation failures. "
                "Consider reducing tensor_parallel_size to %d or less, "
                "or ensure your Ray cluster has %d GPUs available.",
                parallel_config.world_size,
                # TP 大小。
                available_gpus,
                # 可用数。
                available_gpus,
                # 建议值。
                parallel_config.world_size,
                # 集群所需。
            )
            # 告警。

    if ray.is_initialized():
        # 已初始化。
        logger.info("Ray is already initialized. Skipping Ray initialization.")
        # 跳过初始化。
    elif current_platform.is_rocm() or current_platform.is_xpu():
        # ROCm/XPU 平台。
        # Try to connect existing ray instance and create a new one if not found
        # 注释：尝试连接已有 Ray 实例，找不到则新建。
        try:
            ray.init("auto")
            # 连接自动发现。
        except ConnectionError:
            logger.warning(
                "No existing RAY instance detected. "
                "A new instance will be launched with current node resources."
            )
            # 提示新建。
            ray.init(
                address=ray_address,
                # 地址。
                num_gpus=parallel_config.world_size,
                # GPU 数。
                runtime_env=parallel_config.ray_runtime_env,
                # runtime_env。
            )
            # 新建集群。
    else:
        ray.init(address=ray_address, runtime_env=parallel_config.ray_runtime_env)
        # 常规初始化。

    device_str = current_platform.ray_device_key
    # 设备资源键。
    if not device_str:
        # 不支持。
        raise ValueError(
            f"current platform {current_platform.device_name} does not support ray."
        )
        # 报错。

    # Create or get the placement group for worker processes
    # 注释：创建或获取 worker 进程的 placement group。
    if parallel_config.placement_group:
        # 已有 PG。
        current_placement_group = parallel_config.placement_group
        # 直接复用。
    else:
        current_placement_group = ray.util.get_current_placement_group()
        # 检查当前上下文 PG。

    if current_placement_group:
        # 存在可用 PG。
        logger.info("Using the existing placement group")
        # 记录。

        # We are in a placement group
        # 注释：已在 PG 中。
        bundles = current_placement_group.bundle_specs
        # 取规格。
        # Verify that we can use the placement group.
        # 注释：校验可用性。
        device_bundles = 0
        # 计数设备 bundle。
        for bundle in bundles:
            # 遍历。
            bundle_devices = bundle.get(device_str, 0)
            # 该 bundle 设备数。
            if bundle_devices > 1:
                # 超过 1。
                raise ValueError(
                    f"Placement group bundle cannot have more than 1 {device_str}."
                )
                # 报错。
            if bundle_devices:
                # 有设备。
                device_bundles += 1
                # 计数。
        if parallel_config.world_size > device_bundles:
            # 需要设备数超过可用。
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group. "
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}."
            )
            # 报错。
    else:
        logger.info("No current placement group found. Creating a new placement group.")
        # 无 PG，新建。
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # 集群总设备数。
        # Log a warning message and delay resource allocation failure response.
        # Avoid immediate rejection to allow user-initiated placement group
        # created and wait cluster to be ready
        # 注释：记录告警并延迟资源分配失败的响应；避免立即拒绝，给用户创建的
        # placement group 与集群就绪留出时间。
        if parallel_config.world_size > num_devices_in_cluster:
            # 需求超集群资源。
            logger.warning(
                "The number of required %ss exceeds the total "
                "number of available %ss in the placement group.",
                device_str,
                device_str,
            )
            # 告警。
        # Create a new placement group
        # 注释：创建新 PG。
        placement_group_specs: list[dict[str, float]] = [
            {device_str: 1.0} for _ in range(parallel_config.world_size)
        ]
        # 每 worker 一个设备 bundle。

        # vLLM engine is also a worker to execute model with an accelerator,
        # so it requires to have the device in a current node. Check if
        # the current node has at least one device.
        # 注释：vLLM 引擎自身也是带加速器执行模型的 worker，因此当前节点需要
        # 至少一个设备。
        current_ip = get_ip()
        # 当前 IP。
        current_node_id = ray.get_runtime_context().get_node_id()
        # 当前节点。
        current_node_resource = available_resources_per_node()[current_node_id]
        # 当前节点可用资源。
        # TODO (jeffreywang): require_gpu_on_driver should be always False
        # after deprecating RayDistributedExecutor.
        # TODO 注释：弃用 RayDistributedExecutor 后 require_gpu_on_driver 应恒为 False。
        if require_gpu_on_driver:
            # 要求 driver 有 GPU。
            if current_node_resource.get(device_str, 0) < 1:
                # 当前节点无设备。
                raise ValueError(
                    f"Current node has no {device_str} available. "
                    f"{current_node_resource=}. vLLM engine cannot start "
                    f"without {device_str}. Make sure you have at least 1 "
                    f"{device_str} available in a node "
                    f"{current_node_id=} {current_ip=}."
                )
                # 报错。
            # This way, at least bundle is required to be created in a
            # current node.
            # 注释：这样至少一个 bundle 会被创建在当前节点。
            placement_group_specs[0][f"node:{current_ip}"] = 0.001
            # 用 node affinity 约束第一个 bundle 落在当前节点。

        # By default, Ray packs resources as much as possible.
        # 注释：默认 Ray 尽可能打包资源。
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK"
        )
        # 创建 PG（PACK 策略）。
        _wait_until_pg_ready(current_placement_group)
        # 等待就绪。

    assert current_placement_group is not None
    # 断言 PG 非空。
    _verify_bundles(
        current_placement_group, parallel_config, device_str, require_gpu_on_driver
    )
    # 校验 bundle 布局。
    # Set the placement group in the parallel config
    # 注释：把 PG 设置到并行配置。
    parallel_config.placement_group = current_placement_group
    # 保存引用。


def get_num_tpu_nodes() -> int:
    # =========================================================================
    # 获取 Ray 集群中 TPU 节点数。
    # =========================================================================
    from ray._private.accelerators import TPUAcceleratorManager
    # 延迟导入 TPU 加速器管理器。

    cluster_resources = ray.cluster_resources()
    # 集群资源。
    total_tpus = int(cluster_resources["TPU"])
    # TPU 总数。
    tpus_per_node = TPUAcceleratorManager.get_current_node_num_accelerators()
    # 每节点 TPU 数。
    assert total_tpus % tpus_per_node == 0
    # 断言整除。
    return total_tpus // tpus_per_node
    # 返回节点数。


def get_num_nodes_in_placement_group() -> int:
    # =========================================================================
    # 获取当前 PG 覆盖的节点数。
    # =========================================================================
    pg_table = ray.util.placement_group_table()
    # PG 状态表。
    current_pg = ray.util.get_current_placement_group()
    # 当前 PG。
    num_nodes = 0
    # 初始化。
    if current_pg:
        # 有 PG。
        nodes_in_pg = set()
        # 节点集合。
        for pg_key, pg in pg_table.items():
            # 遍历。
            if pg_key == current_pg.id.hex():
                # 匹配当前 PG。
                for _, node in pg["bundles_to_node_id"].items():
                    # 遍历 bundle。
                    nodes_in_pg.add(node)
                    # 收集节点。
        num_nodes = len(nodes_in_pg)
        # 节点数。

    return num_nodes
    # 返回。