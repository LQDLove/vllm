# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/ray_executor.py
# 本文件实现「旧版 Ray 执行器」RayDistributedExecutor：
#   - 每个 Worker 是 Ray actor（RayWorkerWrapper），由 Placement Group 调度。
#   - 控制平面：collective_rpc 经 Ray actor 方法调用（execute_method）。
#   - 数据平面：Ray Compiled DAG（编译图）做 PP 各 stage 的张量传输。
#   - 由于采用 Compile DAG，execute_model 与 sample_tokens 分离延迟执行。
# 注意：新版 RayExecutorV2（ray_executor_v2.py）基于 MQ 通信，正在取代本实现。
# =============================================================================
import os
# 导入 os：设置环境变量（VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE、RAY_CGRAPH 等）。
from collections import defaultdict
# 导入 defaultdict：按节点归集 worker 与物理 GPU（node -> [ranks/GPU ids]）。
from collections.abc import Callable
# 导入 Callable：类型标注 collective_rpc 的 method 参数。
from concurrent.futures import Future
# 导入 Future：非阻塞模式返回异步结果句柄。
from dataclasses import dataclass
# 导入 dataclass：定义 RayWorkerMetaData（worker actor 元数据）。
from typing import TYPE_CHECKING, Any
# 导入类型工具：TYPE_CHECKING（延迟导入）、Any（宽松标注）。

import cloudpickle
# 导入 cloudpickle：序列化可调用对象（RPC 传给 Ray actor）。

import vllm.envs as envs
# 导入 vllm 环境变量模块。
from vllm.logger import init_logger
# 导入日志初始化函数。
from vllm.platforms import current_platform
# 导入当前平台抽象（TPU/XPU 场景需切换 DAG 通道类型）。
from vllm.ray.ray_env import get_env_vars_to_copy
# 导入环境变量复制工具：把 driver 的环境变量同步给 Ray worker。
from vllm.utils.network_utils import (
    get_distributed_init_method,
    # 生成 torch.distributed 初始化地址。
    get_ip,
    # 获取本机 IP。
    get_open_port,
    # 获取空闲端口。
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
# 导入调度器输出类型。
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
# 导入分布式重配置请求与重配置秩类型（DP 弹性扩缩容）。
from vllm.v1.executor.abstract import Executor
# 导入抽象基类 Executor。
from vllm.v1.executor.ray_utils import (
    WORKER_SPECIFIC_ENV_VARS,
    # 不应从 driver 复制给 worker 的专属环境变量集合。
    FutureWrapper,
    # Ray 输出引用包装（满足 .result() 接口）。
    RayWorkerWrapper,
    # Ray actor 内的 worker 包装类。
    detach_zero_copy_from_model_runner_output,
    # 分离 Ray SHM 零拷贝缓冲（防止通道阻塞）。
    initialize_ray_cluster,
    # 初始化 Ray 集群与创建/复用 Placement Group。
    ray,
    # Ray 模块（未安装时为 None）。
)
from vllm.v1.outputs import ModelRunnerOutput
# 导入模型运行器输出类型。

if ray is not None:
    # 仅当 Ray 可用时。
    from ray.actor import ActorHandle
    # 导入 actor 句柄类型。
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    # 导入调度策略（把 actor 调度到指定 PG bundle）。
else:
    ActorHandle = None
    # Ray 不可用时置 None。

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup
# 仅类型检查时导入 PlacementGroup（避免运行时依赖 ray）。

logger = init_logger(__name__)
# 初始化本模块日志。

COMPLETED_NONE_FUTURE: Future[ModelRunnerOutput | None] = Future()
# 预创建的「已完成且结果为 None」的 Future 常量。
COMPLETED_NONE_FUTURE.set_result(None)
# 直接设为完成（结果 None），供 execute_model 返回空结果的场景复用。


@dataclass
class RayWorkerMetaData:
    # =========================================================================
    # RayWorkerMetaData：Ray worker 的元数据。
    # 因为 Ray actor 的创建顺序随机，需要创建完后统一重排 rank。
    # =========================================================================
    """
    Metadata for a Ray worker.
    The order of ray worker creation can be random,
    and we need to reset the rank after creating all workers.
    """
    # 文档字符串：Ray worker 元数据；创建顺序可能随机，创建后需重置 rank。
    worker: ActorHandle
    # Ray actor 句柄。
    created_rank: int
    # 创建时的临时 rank。
    adjusted_rank: int = -1
    # 重排后的最终 rank（初始 -1 表示未调整）。
    ip: str = ""
    # worker 所在节点 IP。


class RayDistributedExecutor(Executor):
    # =========================================================================
    # RayDistributedExecutor：基于 Ray 的分布式执行器（旧版）。
    # =========================================================================
    """Ray-based distributed executor"""
    # 文档字符串：基于 Ray 的分布式执行器。
    uses_ray: bool = True
    # 覆盖父类：使用 Ray 编排。
    supports_pp: bool = True
    # 覆盖父类：支持流水线并行。

    def _init_executor(self) -> None:
        # -------------------------------------------------------------------
        # 初始化 executor：初始化 Ray、创建 placement group、拉起 worker。
        # -------------------------------------------------------------------
        self.forward_dag: ray.dag.CompiledDAG | None = None
        # 保存「已编译的 Ray DAG」（数据平面计算图），初始为 None。

        # For TPU or XPU, avoid compiling NVIDIA's NCCL
        # 注释：TPU/XPU 平台避免编译 NVIDIA NCCL。
        if current_platform.is_tpu() or current_platform.is_xpu():
            # 若为 TPU 或 XPU。
            os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"
            # 强制 DAG 通道类型为共享内存（不使用 NCCL）。

        assert self.uses_ray
        # 断言使用 Ray。
        initialize_ray_cluster(self.parallel_config)
        # 初始化 Ray 集群 & 创建/复用 placement group。
        placement_group = self.parallel_config.placement_group
        # 获取 placement group（worker 的放置约束）。

        # Disable Ray usage stats collection.
        # 注释：禁用 Ray 使用统计收集。
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        # 读取当前设置。
        if ray_usage != "1":
            # 若非显式开启。
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
            # 显式置 0。

        # Create the parallel GPU workers.
        # 注释：创建并行 GPU workers。
        self._init_workers_ray(placement_group)
        # 在 placement group 中创建并初始化所有 worker。

        # KV connector setup
        # 注释：KV 连接器设置。
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        # 判断是否配置了 KV 迁移（disaggregated 场景），决定是否聚合所有 worker 输出。

        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            # 未配置编码器转移。
            or self.vllm_config.ec_transfer_config.is_ec_consumer
            # 或本身是编码器消费者。
        )
        # 判断是否需要执行采样：pooling 任务/纯编码器生产者不需要采样。
        # 若无需采样，execute_model 可直接返回输出而不必等 sample_tokens。

        self.scheduler_output: SchedulerOutput | None = None
        # 暂存最近的调度输出（execute_model → sample_tokens 延迟执行用）。

    def shutdown(self) -> None:
        # -------------------------------------------------------------------
        # 关闭执行器：teardown DAG，kill 所有 worker。
        # -------------------------------------------------------------------
        if logger:
            # 防御性检查（偶发 logger 为 None）。
            # Somehow logger can be None here.
            # 注释：此处 logger 可能为 None。
            logger.info(
                "Shutting down Ray distributed executor. If you see error log "
                "from logging.cc regarding SIGTERM received, please ignore "
                "because this is the expected termination process in Ray."
            )
            # 提示：Ray 关闭时 logging.cc 的 SIGTERM 报错属正常现象。
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            # 若 DAG 已构建。
            self.forward_dag.teardown()
            # 拆解编译 DAG（释放图资源）。
            import ray
            # 延迟导入 ray。

            for worker in self.workers:
                # 遍历所有 worker。
                ray.kill(worker)
                # 强杀 worker actor。
            self.forward_dag = None
            # 置空 DAG。

    def _configure_ray_workers_use_nsight(self, ray_remote_kwargs) -> dict[str, Any]:
        # -------------------------------------------------------------------
        # 为 Ray worker 配置 nsight 性能剖析（通过 runtime_env）。
        # -------------------------------------------------------------------
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        # 注释：若启用 nsight 剖析，需要把剖析配置通过 runtime_env 传给 worker。
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        # 取（或创建）runtime_env 字典。
        runtime_env.update(
            {
                "nsight": {
                    # nsight 配置。
                    "t": "cuda,cudnn,cublas",
                    # 追踪的库。
                    "o": "'worker_process_%p'",
                    # 输出文件按进程号命名。
                    "cuda-graph-trace": "node",
                    # 追踪 CUDA graph 节点。
                }
            }
        )
        # 更新 nsight 配置。
        return ray_remote_kwargs
        # 返回更新后的 kwargs。

    def _update_noset_device_env_vars(self, ray_remote_kwargs):
        # -------------------------------------------------------------------
        # 设置「不自动设置设备」的环境变量（由 vLLM 自行管理设备）。
        # -------------------------------------------------------------------
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        # 取 runtime_env。
        env_vars = runtime_env.setdefault("env_vars", {})
        # 取 env_vars 段。
        env_vars.update(
            {env_var: "1" for env_var in current_platform.ray_noset_device_env_vars}
        )
        # 把平台的禁用自动设备变量置 1（如 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES）。
        return ray_remote_kwargs
        # 返回 kwargs。

    # child class could overwrite this to return actual env vars.
    # 注释：子类可覆盖此方法返回实际环境变量。
    def _get_env_vars_to_be_updated(self):
        # -------------------------------------------------------------------
        # 返回要同步给 worker 的环境变量列表（默认返回全部 workers 的环境）。
        # -------------------------------------------------------------------
        return self._env_vars_for_all_workers
        # 返回按 worker 排列的环境变量列表。

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        # -------------------------------------------------------------------
        # 在 placement group 中创建全部 Ray worker 并完成初始化。
        # -------------------------------------------------------------------
        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS
        # 每个 worker 申请的 GPU 数（默认 1）。

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        # 注释：driver 空 worker 不实际使用资源，但保留 driver worker 的资源位。
        self.driver_dummy_worker: RayWorkerWrapper | None = None
        # driver 空 worker（本实现未使用，兼容字段）。
        # The remaining workers are the actual ray actors.
        # 注释：其余 worker 是真正的 Ray actor。
        self.workers: list[RayWorkerWrapper] = []
        # 保存所有 worker actor 句柄。

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        # 注释：用于 Ray 编译 DAG；先按 PP rank、再按 TP rank 索引。
        # 内层列表即某 PP stage 的 TP worker 组。
        self.pp_tp_workers: list[list[RayWorkerWrapper]] = []
        # 保存 PP×TP 二维 worker 列表。

        if self.parallel_config.ray_workers_use_nsight:
            # 若配置了 nsight。
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs
            )
            # 注入 nsight 配置。

        # The way ray actors are setup in vllm is that the visible devices are
        # not set by actors, they are left unset by ray. Internally we index
        # the right gpu with local_rank. This is similar to how mp mode works.
        # 注释：vLLM 的 Ray actor 不设置 CUDA_VISIBLE_DEVICES，保持由 Ray 分配；
        # 内部用 local_rank 索引正确 GPU，与多进程模式类似。
        self._update_noset_device_env_vars(ray_remote_kwargs)
        # 注入禁用自动设备的环境变量。

        # Create the workers.
        # 注释：创建 workers。
        bundle_indices: list[int]
        # 初始化 bundle 索引列表。
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # 若用户显式指定 bundle 索引。
            # Use the bundle indices specified by the user.
            # 注释：使用用户指定的 bundle 索引。
            bundle_indices = list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            # 解析环境变量为整数列表。
            assert len(bundle_indices) == self.parallel_config.world_size, (
                "VLLM_RAY_BUNDLE_INDICES must have the same size"
                f" as the world size, but got {bundle_indices=} "
                f"and {self.parallel_config.world_size=}"
            )
            # 断言数量与 world_size 一致。
            assert len(set(bundle_indices)) == len(bundle_indices), (
                "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
                f" but got {bundle_indices=}"
            )
            # 断言无重复。
        else:
            # use the first N bundles that have GPU resources.
            # 注释：使用前 N 个含有 GPU 资源的 bundle。
            bundle_indices = []
            # 初始化列表。
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                # 遍历所有 bundle。
                if bundle.get(current_platform.ray_device_key, 0):
                    # 若该 bundle 有设备资源。
                    bundle_indices.append(bundle_id)
                    # 记录 bundle id。
            bundle_indices = bundle_indices[: self.parallel_config.world_size]
            # 截取前 world_size 个。

        worker_metadata: list[RayWorkerMetaData] = []
        # 保存 worker 元数据列表。
        driver_ip = get_ip()
        # 获取 driver（本机）IP。
        for rank, bundle_id in enumerate(bundle_indices):
            # 遍历每个 bundle。
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                # PG。
                placement_group_capture_child_tasks=True,
                # 捕获子任务。
                placement_group_bundle_index=bundle_id,
                # 指定 bundle。
            )
            # 构造调度策略：把 actor 固定到对应 bundle（即对应 GPU）。
            if current_platform.ray_device_key == "GPU":
                # NV+AMD GPUs, and Intel XPUs
                # 注释：NV+AMD GPU 与 Intel XPU 走此分支。
                worker = ray.remote(
                    num_cpus=0,
                    # 不占 CPU 资源。
                    num_gpus=num_gpus,
                    # 申请 GPU。
                    scheduling_strategy=scheduling_strategy,
                    # 调度策略。
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)
                # 创建 Ray actor（RayWorkerWrapper），rpc_rank 为临时 rank。
            else:
                worker = ray.remote(
                    num_cpus=0,
                    # 不占 CPU。
                    num_gpus=0,
                    # 不占标准 GPU 字段。
                    resources={current_platform.ray_device_key: num_gpus},
                    # 使用平台自定义设备资源键（如 TPU）。
                    scheduling_strategy=scheduling_strategy,
                    # 调度策略。
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(rpc_rank=rank)
                # 创建带自定义资源的 actor。

            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))
            # 记录元数据。

        worker_ips = ray.get(
            [
                each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
                for each in worker_metadata
            ]
        )
        # 并行获取每个 worker 所在节点 IP。

        for each, ip in zip(worker_metadata, worker_ips):
            # 遍历填充 IP。
            each.ip = ip
            # 设置元数据的 IP。

        logger.debug("workers: %s", worker_metadata)
        # 调试日志：worker 元数据。
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)
        # 调试日志：driver 空 worker。

        ip_counts: dict[str, int] = {}
        # 统计每个 IP 上的 worker 数量。
        for ip in worker_ips:
            # 遍历 IP。
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
            # 计数。

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            # 排序键函数：按 (是否 driver 节点, 节点 worker 数, IP) 排序。
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            # 文档字符串：按 3 个属性排序——①driver 同节点优先；
            # ②节点 worker 数少者优先；③IP 小者优先。
            ip = item.ip
            # 取 worker IP。
            return 0 if ip == driver_ip else 1, ip_counts[ip], ip
            # 返回排序元组。

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        # 注释：排序后同节点 worker 相邻，driver 节点 worker 排最前。
        sorted_worker_metadata = sorted(
            worker_metadata, key=sort_by_driver_then_worker_ip
        )
        # 排序。
        for i, item in enumerate(sorted_worker_metadata):
            # 遍历。
            item.adjusted_rank = i
            # 设置调整后的 rank。
        self.workers = [item.worker for item in sorted_worker_metadata]
        # 按新 rank 顺序保存 actor 句柄。
        rerank_mapping = {
            item.created_rank: item.adjusted_rank for item in sorted_worker_metadata
        }
        # 构建 旧rank → 新rank 映射。
        self.collective_rpc("adjust_rank", args=(rerank_mapping,))
        # 广播给所有 worker 调整内部 rpc_rank。

        # Get the set of physical GPU IDs used on each node.
        # 注释：获取每个节点上使用的物理 GPU ID 集合。
        worker_node_and_physical_gpu_ids = []
        # 初始化列表。
        for worker in [self.driver_dummy_worker] + self.workers:
            # 遍历（含空 worker）。
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                # 注释：使用 ray SPMD worker 时 driver_dummy_worker 可能为 None。
                continue
                # 跳过。
            worker_node_and_physical_gpu_ids.append(
                ray.get(worker.get_node_and_physical_gpu_ids.remote())  # type: ignore[attr-defined]
            )
            # 获取 (node_id, physical_gpu_ids)。

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        # 节点 → worker rank 列表。
        node_physical_gpu_ids = defaultdict(list)  # node id -> physical GPU IDs
        # 节点 → 物理 GPU ID 列表。

        for i, (node_id, physical_gpu_ids) in enumerate(
            worker_node_and_physical_gpu_ids
        ):
            # 遍历。
            node_workers[node_id].append(i)
            # 记录该节点上的 worker rank。
            # `physical_gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: physical GPU IDs can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            # 注释：physical_gpu_ids 可能是字符串或整数列表，统一转成整数；
            # 注意物理 GPU ID 可大于 9（如 16 卡），字符串排序不够。
            physical_gpu_ids = [
                current_platform.device_control_id_to_physical_device_id(str(x))
                # 控制 ID 转物理设备 ID。
                for x in physical_gpu_ids
                # 遍历。
            ]
            # 统一转换为物理设备 ID。
            node_physical_gpu_ids[node_id].extend(physical_gpu_ids)
            # 累积到该节点的物理 GPU 集合。
        for node_id, physical_gpu_ids in node_physical_gpu_ids.items():
            # 遍历各节点。
            node_physical_gpu_ids[node_id] = sorted(physical_gpu_ids)
            # 排序（保证 local_rank 映射稳定）。

        all_ips = set(worker_ips + [driver_ip])
        # 全集 IP。
        n_ips = len(all_ips)
        # IP 数量。
        n_nodes = len(node_workers)
        # 节点数量。

        if n_nodes != n_ips:
            # 若节点数与 IP 数不一致。
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP`"
                " environment variable, make sure it is unique for"
                " each node."
            )
            # 报错：每节点应唯一 IP。检查 VLLM_HOST_IP 配置。

        all_args_to_update_environment_variables: list[dict[str, str]] = [
            {} for _ in worker_node_and_physical_gpu_ids
        ]
        # 初始化每个 worker 的环境变量字典。

        # Environment variables to copy from driver to workers
        # 注释：需要从 driver 复制到 worker 的环境变量。
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
            # 排除 worker 专属变量。
            additional_vars=set(current_platform.additional_env_vars),
            # 添加平台附加变量。
            destination="workers",
            # 目标为 worker。
        )
        # 获取要复制的环境变量集合。

        # Copy existing env vars to each worker's args
        # 注释：把现有环境变量复制到每个 worker 参数。
        for args in all_args_to_update_environment_variables:
            # 遍历每 worker 的 env 字典。
            # TODO: refactor platform-specific env vars
            # TODO 注释：重构平台专属环境变量。
            for name in env_vars_to_copy:
                # 遍历要复制的变量名。
                if name in os.environ:
                    # 若 driver 中存在。
                    args[name] = os.environ[name]
                    # 复制值。

        self._env_vars_for_all_workers = all_args_to_update_environment_variables
        # 保存到实例（子类可覆盖获取逻辑）。

        self.collective_rpc(
            "update_environment_variables", args=(self._get_env_vars_to_be_updated(),)
        )
        # 广播让所有 worker 更新环境变量。

        if len(node_physical_gpu_ids) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            # 注释：单节点时无需外部 IP，用回环地址足矣。节点可能有多个网卡 IP，
            # get_ip() 可能返回任一个，复杂的网络配置下可能不适用于节点内通信；
            # 回环地址总是适用。
            driver_ip = "127.0.0.1"
            # 用回环地址。
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port()
        )
        # 生成 torch.distributed 初始化地址。

        # Initialize the actual workers inside worker wrapper.
        # 注释：在 worker 包装器内初始化真正的 worker。
        all_kwargs = []
        # 初始化参数列表。
        for rank, (node_id, _) in enumerate(worker_node_and_physical_gpu_ids):
            # 遍历每个 worker。
            local_rank = node_workers[node_id].index(rank)
            # 该 worker 在节点内的 local_rank。
            kwargs = dict(
                vllm_config=self.vllm_config,
                # 配置。
                assigned_physical_gpu_ids=sorted(node_physical_gpu_ids[node_id]),
                # 该节点物理 GPU 映射。
                local_rank=local_rank,
                # 本地 rank。
                rank=rank,
                # 全局 rank。
                distributed_init_method=distributed_init_method,
                # 分布式初始化地址。
                is_driver_worker=(not self.parallel_config)
                # 无并行配置时恒为 driver。
                or (rank % self.parallel_config.tensor_parallel_size == 0),
                # 否则 TP rank 0 为 driver。
            )
            # 组装 worker 初始化参数。
            all_kwargs.append(kwargs)
            # 追加。
        self.collective_rpc("init_worker", args=(all_kwargs,))
        # 广播初始化所有 worker。

        self.collective_rpc("init_device")
        # 广播初始化设备。
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            # 若启用弹性 EP 扩容。
            self.collective_rpc("elastic_ep_execute", args=("load_model",))
            # 走弹性 EP 加载。
        else:
            self.collective_rpc("load_model")
            # 常规加载模型。

        def _update_block_size(worker):
            # 辅助函数：更新 block size。
            current_platform.update_block_size_for_backend(worker.vllm_config)
            # 按平台更新。

        self.collective_rpc(_update_block_size)
        # 广播调用（可调用对象形式）更新 block size。

        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            # 遍历 PP stage。
            self.pp_tp_workers.append([])
            # 为每个 PP stage 建空 TP 组。
            for tp_rank in range(self.parallel_config.tensor_parallel_size):
                # 遍历 TP rank。
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                # 注释：PP=2、TP=4 时 pp_tp_workers = [[0,1,2,3], [4,5,6,7]]。
                rank = (pp_rank * self.parallel_config.tensor_parallel_size) + tp_rank
                # 计算全局 rank。
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                # 断言构建一致性。
                assert pp_rank < len(self.pp_tp_workers)
                # 断言索引合法。
                self.pp_tp_workers[pp_rank].append(self.workers[rank])
                # 填入 worker 句柄。

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        # -------------------------------------------------------------------
        # DP 弹性扩缩容：广播重配置请求；若当前 rank 被要求关闭则自杀。
        # -------------------------------------------------------------------
        self.collective_rpc("reinitialize_distributed", args=(reconfig_request,))
        # 广播重配置。
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            # 若该 rank 被要求关闭。
            self.shutdown()
            # 关闭自己。

    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        # 调度输出。
        non_block: bool = False,
        # 非阻塞标志。
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # 执行模型（旧版 Ray 的延迟执行入口）。
        # 若需要采样，则先暂存 scheduler_output 并立即返回 None/Future，
        # 等后续 sample_tokens() 才真正驱动 DAG 执行。
        # -------------------------------------------------------------------
        if self.scheduler_output is not None:
            # 状态校验：上一个输出尚未被 sample_tokens 消费。
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )
            # 抛状态错误。

        if not self.uses_sampler or not scheduler_output.total_num_scheduled_tokens:
            # Model will not execute, call model runner immediately.
            # 注释：模型不执行（无采样需求或本次无调度 token），立即调用模型。
            return self._execute_dag(scheduler_output, None, non_block)
            # 直接驱动 DAG 执行。

        # Model will execute, defer to sample_tokens() call.
        # 注释：模型需要执行，延迟到 sample_tokens() 调用。
        self.scheduler_output = scheduler_output
        # 暂存调度输出。
        return COMPLETED_NONE_FUTURE if non_block else None
        # 非阻塞返回「已完成的 None Future」，同步返回 None。

    def sample_tokens(  # type: ignore[override]
        self,
        grammar_output: "GrammarOutput | None",
        # grammar 输出约束。
        non_block: bool = False,
        # 非阻塞标志。
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # 采样方法（旧版 Ray 延迟执行的核心）：真正驱动 DAG 执行模型并采样。
        # -------------------------------------------------------------------
        """Execute the model on the Ray workers.

        The scheduler output to use should have been provided in
        a prior call to execute_model().

        Args:
            grammar_output: The structured outputs grammar bitmask, if applicable.
            non_block: If True, the method will return a Future.

        Returns:
            The model runner output.
        """
        # 文档字符串：调度输出应在之前 execute_model() 提供；
        # grammar_output 为结构化输出位掩码；non_block=True 返回 Future。
        scheduler_output = self.scheduler_output
        # 取暂存的调度输出。
        if scheduler_output is None:
            # 无暂存输出（没有实际执行）。
            return COMPLETED_NONE_FUTURE if non_block else None
            # 返回空结果。

        self.scheduler_output = None
        # 清空暂存。

        return self._execute_dag(scheduler_output, grammar_output, non_block)
        # 驱动 DAG 执行。

    def _execute_dag(
        self,
        scheduler_output: SchedulerOutput,
        # 调度输出。
        grammar_output: "GrammarOutput | None",
        # grammar。
        non_block: bool = False,
        # 非阻塞。
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # 实际执行编译 DAG：首次调用时构建 DAG，然后执行并取结果。
        # -------------------------------------------------------------------
        # Build the compiled DAG for the first time.
        # 注释：首次执行时构建编译 DAG。
        if self.forward_dag is None:  # type: ignore
            # 若尚未构建。
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)
            # 构建编译 DAG（PP 数据流图）。

        refs = self.forward_dag.execute((scheduler_output, grammar_output))  # type: ignore
        # 执行 DAG，输入为 (调度输出, grammar)，返回各输出的 ObjectRef。

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            # 注释：无 KV 连接器时只取单个 worker 输出；无 PP 时在此阻塞等结果。
            if not non_block:
                # 同步模式。
                output = refs[0].get()
                # 阻塞取第一个输出的值。
                detach_zero_copy_from_model_runner_output(output)
                # 分离 SHM 零拷贝缓冲（防阻塞）。
                return output
                # 返回输出。

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            # 注释：使用 PP 时立即返回 FutureWrapper，让调度器让位给下一批。
            return FutureWrapper(refs[0])
            # 异步返回包装器。

        # Get output from all workers when connector is present
        # 注释：有连接器时需从所有 worker 取输出。
        assert self.kv_output_aggregator is not None
        # 断言聚合器已初始化。
        if not non_block:
            # 同步模式。
            # Block and get results from all workers
            # 注释：阻塞获取所有 worker 结果。
            outputs = ray.get(refs)
            # 取全部输出。
            for output in outputs:
                # 遍历。
                detach_zero_copy_from_model_runner_output(output)
                # 分离零拷贝缓冲。
            return self.kv_output_aggregator.aggregate(outputs)
            # 聚合返回。

        # Return a future that will aggregate outputs from all workers
        # 注释：返回会把所有 worker 输出聚合的 Future。
        return FutureWrapper(refs, self.kv_output_aggregator)
        # 异步返回带聚合的包装器。

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        # 方法名或可调用对象。
        timeout: float | None = None,
        # 超时。
        args: tuple = (),
        # 位置参数。
        kwargs: dict[str, Any] | None = None,
        # 关键字参数。
        non_block: bool = False,
        # 非阻塞。
    ) -> list[Any] | Future[list[Any]]:
        # -------------------------------------------------------------------
        # 在全部 worker 上执行方法（Ray actor 远程调用）。
        # -------------------------------------------------------------------
        """Runs the given method on all workers."""
        # 文档字符串：在所有 worker 上运行指定方法。
        sent_method = method if isinstance(method, str) else cloudpickle.dumps(method)
        # 字符串原样传递；可调用对象用 cloudpickle 序列化。
        del method
        # 删除原引用（避免闭包扣住大对象）。
        if kwargs is None:
            kwargs = {}
            # None 归一化。
        ray_worker_outputs = [
            worker.execute_method.remote(  # type: ignore[attr-defined]
                sent_method, *args, **kwargs
            )
            # 远程调用每个 worker 的 execute_method。
            for worker in self.workers
            # 遍历所有 worker。
        ]
        # 发起全部远程调用。

        # Get the results of the ray workers.
        # 注释：获取 Ray worker 的结果。
        if non_block:
            # 非阻塞。
            return FutureWrapper(ray_worker_outputs)
            # 返回包装器（result() 时阻塞取）。

        return ray.get(ray_worker_outputs, timeout=timeout)
        # 阻塞获取（带超时）。

    def _check_ray_cgraph_installation(self):
        # -------------------------------------------------------------------
        # 校验 Ray CGraph（编译图）及其依赖（cupy）是否安装。
        # -------------------------------------------------------------------
        import importlib.metadata
        # 延迟导入（读取版本）。

        from packaging import version
        # 导入版本解析。

        required_version = version.parse("2.43.0")
        # 最低 Ray 版本。
        current_version = version.parse(importlib.metadata.version("ray"))
        # 当前 Ray 版本。
        if current_version < required_version:
            # 版本过低。
            raise ValueError(
                f"Ray version {required_version} is "
                f"required, but found {current_version}"
            )
            # 报错。

        import importlib.util
        # 延迟导入。

        cgraph_spec = importlib.util.find_spec("ray.experimental.compiled_dag_ref")
        # 查找 CGraph 模块。
        if cgraph_spec is None:
            # 未安装。
            raise ValueError(
                "Ray Compiled Graph is not installed. "
                "Run `pip install ray[cgraph]` to install it."
            )
            # 提示安装。

        cupy_spec = importlib.util.find_spec("cupy")
        # 查找 cupy。
        if cupy_spec is None and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE == "nccl":
            # NCCL 通道需要 cupy 但未安装。
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE is set to 'nccl'. "
                "Run `pip install ray[cgraph]` and check cupy installation."
            )
            # 报错。

    def _compiled_ray_dag(self, enable_asyncio: bool):
        # -------------------------------------------------------------------
        # 构建 Ray 编译 DAG：定义 PP 各 stage 间的数据流（TP 组内 SPMD 执行）。
        # -------------------------------------------------------------------
        assert self.parallel_config.use_ray
        # 断言使用 Ray。
        self._check_ray_cgraph_installation()
        # 校验依赖。
        # Enlarge the default value of "RAY_CGRAPH_get_timeout" to 300 seconds
        # (it is 10 seconds by default). This is a Ray environment variable to
        # control the timeout of getting result from a compiled graph execution,
        # i.e., the distributed execution that includes model forward runs and
        # intermediate tensor communications, in the case of vllm.
        # Note: we should set this env var before importing
        # ray.dag, otherwise it will not take effect.
        # 注释：把 RAY_CGRAPH_get_timeout 从默认 10 秒放大到 300 秒。
        # 这是 Ray 环境变量，控制编译图执行（含模型 forward 与中间张量通信）
        # 的结果获取超时。注意：必须在导入 ray.dag 之前设置才生效。
        os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")  # noqa: SIM112
        # 设置 get 超时为 300 秒（仅当未设置时）。
        from ray.dag import InputNode, MultiOutputNode
        # 延迟导入 DAG 构建 API。

        logger.info(
            "RAY_CGRAPH_get_timeout is set to %s",
            os.environ["RAY_CGRAPH_get_timeout"],  # noqa: SIM112
        )
        # 记录超时设置。
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE,
        )
        # 记录通道类型。
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )
        # 记录通信重叠开关。

        channel_type = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
        # 取通道类型。
        if channel_type not in ("auto", "nccl", "shm"):
            # 非法值。
            raise ValueError(
                "Invalid value for VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: "
                f"{channel_type}. Valid values are: 'auto', 'nccl', or 'shm'."
            )
            # 报错。

        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            #
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501
            # 注释：示例 DAG（PP=2、TP=4）：
            #   SchedulerOutput → 0..3（PP0，TP 组 SPMD）→ 带 IntermediateTensors 的中间输出
            #   → 4..7（PP1，TP 组 SPMD）→ ModelRunnerOutput。
            # All workers in the first TP group will take in the
            # ExecuteModelRequest as input.
            # 注释：第一 TP 组的所有 worker 接收 SchedulerOutput 作为输入。
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            # 初始输入：给第一个 PP stage 的每个 TP worker 一份 input_data。
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker,
                # and the TP group executes in SPMD fashion.
                # 注释：每个 PP worker 接收上一 PP worker 的输出，TP 组以 SPMD 方式执行。
                outputs = [
                    worker.execute_model_ray.bind(outputs[i])  # type: ignore[attr-defined]
                    # 绑定每个 TP worker 的 DAG 节点，输入为上一 stage 对应 TP rank 的输出。
                    for i, worker in enumerate(tp_group)
                    # 遍历 TP 组。
                ]
                # 当前 PP stage 的输出。
                last_pp_rank = len(self.pp_tp_workers) - 1
                # 最后一个 PP stage。
                if (
                    pp_rank < last_pp_rank
                    # 非最后 stage。
                    and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE != "shm"
                    # 且非 SHM 通道。
                ):
                    # Specify how intermediate tensors should be passed
                    # between pp stages, no need to specify for the last
                    # pp stage or when using shared memory (the default).
                    # 注释：指定 PP stage 间中间张量的传输方式；
                    # 最后 stage 或使用共享内存（默认）时无需指定。
                    transport = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
                    # 取传输类型。
                    outputs = [
                        output.with_tensor_transport(transport=transport)
                        # 为节点指定张量传输协议。
                        for output in outputs
                        # 遍历。
                    ]
                    # 设置传输协议。

            forward_dag = MultiOutputNode(outputs)
            # 把最终输出聚合为多输出节点。

        if envs.VLLM_USE_RAY_WRAPPED_PP_COMM:
            # 若启用 Ray 包装的 PP 通信器。
            from ray.experimental.channel.accelerator_context import (
                register_accelerator_context,
            )
            # 导入加速器上下文注册 API。

            from vllm.distributed.device_communicators.ray_communicator import (
                RayPPCommunicator,
            )
            # 导入 Ray PP 通信器。

            register_accelerator_context(
                torch_module_name="cuda",
                # 模块名。
                communicator_cls=RayPPCommunicator,
                # 通信器类。
            )
            # 注册加速器上下文（让 Ray 通道调用 vLLM 的 PP 通信器）。
            logger.info(
                "Using RayPPCommunicator "
                "(which wraps vLLM _PP GroupCoordinator) "
                "for Ray Compiled Graph communication."
            )
            # 记录。
        else:
            logger.info(
                "Using Ray's NCCL communicator for Ray Compiled Graph communication."
            )
            # 记录默认使用 Ray NCCL 通信器。

        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            # 是否异步。
            _overlap_gpu_communication=envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
            # 是否重叠 GPU 通信。
        )
        # 编译 DAG 并返回。

    def __del__(self):
        # -------------------------------------------------------------------
        # 析构函数：关闭执行器（防御性清理）。
        # -------------------------------------------------------------------
        self.shutdown()
        # 调用关闭逻辑。

    def check_health(self) -> None:
        # -------------------------------------------------------------------
        # 健康检查：旧版 Ray 执行器假定 worker 健康（TODO 待实现）。
        # -------------------------------------------------------------------
        # Assume that the Ray workers are healthy.
        # TODO: check the health of the Ray workers
        # 注释：假定 Ray worker 健康；TODO 尚未实现实际健康检查。
        return
        # 直接返回。