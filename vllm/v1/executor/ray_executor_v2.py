# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/ray_executor_v2.py
# 本文件实现「新版 Ray 执行器」RayExecutorV2：
#   - 继承 MultiprocExecutor，复用其 MessageQueue 控制平面与 FutureWrapper。
#   - 但 worker 不再是 OS 子进程，而是 Ray actor（RayWorkerProc）。
#   - GPU 分配采用「延迟初始化」：先创建轻量 actor，等 Ray 完成 placement 后
#     发现物理 GPU ID，再调用 initialize_worker 完成完整初始化。
#   - 该机制允许多个 vLLM 实例共存于同一节点且互不冲突。
# =============================================================================
import copy
# 导入 copy：深拷贝 ray runtime_env 配置（避免共享字典被意外修改）。
import os
# 导入 os：在 worker 内设置环境变量（driver_env_vars/env_vars）。
import threading
# 导入 threading：监控线程、关闭锁。
import weakref
# 导入 weakref：弱引用 self 供监控线程访问 executor。
from collections import defaultdict, deque
# 导入 defaultdict（按节点分组）、deque（FutureWrapper FIFO 队列）。
from dataclasses import dataclass
# 导入 dataclass：定义 RayWorkerHandle。
from typing import Any
# 导入 Any 类型。

import vllm.envs as envs
# 导入 vllm 环境变量模块。
from vllm.config import VllmConfig
# 导入 VllmConfig（worker 初始化参数）。
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle,
    MessageQueue,
)
# 导入共享内存消息队列与句柄（控制平面通道）。
from vllm.logger import init_logger
# 导入日志初始化函数。
from vllm.platforms import current_platform
# 导入当前平台抽象（ray_device_key 等）。
from vllm.utils.network_utils import (
    _get_open_port,
    # 从指定起始端口开始找空闲端口。
    get_distributed_init_method,
    # 生成 torch.distributed 初始化地址。
    get_open_port,
    # 获取随机空闲端口。
)
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    # FIFO 异步结果包装（复用）。
    MultiprocExecutor,
    # 父类执行器（复用 MQ 控制平面与集体 RPC）。
    WorkerProc,
    # worker 进程封装（作为 RayWorkerProc 的父类）。
)
from vllm.v1.executor.ray_env_utils import get_driver_env_vars
# 导入 driver 环境变量获取工具。
from vllm.v1.executor.ray_utils import (
    WORKER_SPECIFIC_ENV_VARS,
    # worker 专属环境变量集合。
    build_actor_name,
    # 构建带 TP/PP/PCP 信息的 actor 名（Ray dashboard 可见）。
    get_bundles_for_indices,
    # 按显式 bundle 索引获取 (bundle_id, node_id, node_ip)。
    get_bundles_sorted_by_node,
    # 按节点排序获取 bundle 绑定。
    initialize_ray_cluster,
    # 初始化 Ray 集群与 placement group。
    ray,
    # Ray 模块（未安装时为 None）。
)

if ray is not None:
    # 仅当 Ray 可用时。
    from ray.actor import ActorHandle
    # actor 句柄类型。
    from ray.types import ObjectRef
    # ObjectRef 类型（用于跟踪 run() 的存活）。
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    # 调度策略（绑定 bundle）。
else:
    ActorHandle = None
    # Ray 不可用时置 None。

logger = init_logger(__name__)
# 初始化本模块日志。


@dataclass
class RayWorkerHandle:
    # =========================================================================
    # RayWorkerHandle：Ray worker actor 的句柄，与 MultiprocExecutor 兼容。
    # =========================================================================
    """Handle for a Ray worker actor, compatible with MultiprocExecutor."""
    # 文档字符串：Ray worker actor 句柄，与 MultiprocExecutor 兼容。
    actor: ActorHandle
    """Ray worker actor"""
    # 字段注释：Ray worker actor。
    rank: int
    """Rank of the worker"""
    # 字段注释：worker 的全局 rank。
    local_rank: int
    """Local rank of the worker"""
    # 字段注释：worker 的本地 rank。
    node_id: str
    """Node ID of the worker"""
    # 字段注释：worker 所在节点 ID。
    bundle_id_idx: int = -1
    """Placement group bundle index for the worker"""
    # 字段注释：worker 对应的 placement group bundle 索引。
    run_ref: ObjectRef | None = None
    """run() ObjectRef used as a sentinel for health monitoring"""
    # 字段注释：run() 的 ObjectRef，作为存活监控哨兵。

    def run(self):
        # -------------------------------------------------------------------
        # 启动 worker 的忙循环（远程调用 actor.run()）。
        # -------------------------------------------------------------------
        """Start the worker's busy loop"""
        # 文档字符串：启动 worker 的忙循环。
        self.run_ref = self.actor.run.remote()
        # 远程调用 run() 并保存 ObjectRef（用于监控存活）。


class RayWorkerProc(WorkerProc):
    # =========================================================================
    # RayWorkerProc：运行在 Ray actor 内部的 worker 进程。
    # 初始化拆分为两个阶段：
    #   1. __init__：轻量创建，仅保存初始化参数（不做设备/模型初始化）。
    #   2. initialize_worker：GPU ID 被发现后调用，完成完整初始化。
    # =========================================================================
    """Worker process that runs inside a Ray actor.

    Initialization is split into two phases:
    1. __init__: lightweight setup, stores init args (no device/model init)
    2. initialize_worker: called after GPU IDs are discovered, completes
       the full WorkerProc initialization with the correct local_rank and
       logical-to-physical GPU mapping.

    GPU assignment flow:

    1. RayExecutorV2 enables RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES so Ray does
       not set CUDA_VISIBLE_DEVICES on RayWorkerProc actors at creation time.
    2. Each actor is scheduled with a placement group and bundle index; Ray resolves
       the physical GPU ID for that bundle at placement time.
    3. After placement, the executor discovers each worker's GPU ID and passes the
       node's logical-to-physical mapping (assigned_physical_gpu_ids) to
       initialize_worker(); CUDA_VISIBLE_DEVICES is never modified.

    Scheduling must complete before the mapping is known when the placement
    group is externally managed: only then is the GPU tied to the worker's
    bundle resolved.

    This sequence allows multiple vLLM instances to coexist on the same node:
    each instance is unaware which physical devices others hold, and the
    externally managed placement group avoids device assignment conflicts
    by binding workers to specific placement group bundles.
    """
    # 类文档字符串：详细描述两阶段初始化与 GPU 分配流程——
    # ①启用 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES，Ray 创建 actor 时不设
    # CUDA_VISIBLE_DEVICES；②每个 actor 按 PG bundle 调度，Ray 在 placement 时
    # 解析该 bundle 的物理 GPU；③placement 后 executor 发现每个 worker 的 GPU ID，
    # 把节点逻辑→物理映射传入 initialize_worker；CUDA_VISIBLE_DEVICES 从不被修改。
    # 该机制让多个 vLLM 实例可共存于同一节点，互不感知对方持有的设备。

    def __init__(
        self,
        vllm_config: VllmConfig,
        # 配置。
        rank: int,
        # 全局 rank。
        distributed_init_method: str,
        # 分布式初始化地址。
        input_shm_handle: Handle,
        # 广播 MQ 句柄。
        is_driver_worker: bool,
        # 是否 driver。
        is_driver_node: bool = False,
        # 是否位于 driver（本 executor）节点。
    ):
        # Defer WorkerProc.__init__ until GPU IDs are known.
        # 注释：延迟 WorkerProc.__init__ 直到 GPU ID 已知。
        self._is_driver_node = is_driver_node
        # 保存是否位于 driver 节点（决定 MQ 是否用共享内存）。
        self._init_kwargs = dict(
            vllm_config=vllm_config,
            # 配置。
            rank=rank,
            # 全局 rank。
            distributed_init_method=distributed_init_method,
            # 分布式地址。
            input_shm_handle=input_shm_handle,
            # 广播 MQ 句柄。
            shared_worker_lock=None,
            # 共享锁延迟到 initialize_worker 处理。
            is_driver_worker=is_driver_worker,
            # 是否 driver。
        )
        # 只保存初始化参数，暂不调用父类构造（避免提前初始化设备）。

    def get_node_and_physical_gpu_ids(self) -> tuple[str, list[int]]:
        # -------------------------------------------------------------------
        # 返回 (node_id, physical_gpu_ids)，即 Ray 分配给该 actor 的设备。
        # -------------------------------------------------------------------
        """Return (node_id, physical_gpu_ids) assigned to this actor by Ray."""
        # 文档字符串：返回 Ray 分配给该 actor 的 (node_id, 物理 GPU IDs)。
        node_id = ray.get_runtime_context().get_node_id()
        # 获取 actor 所在节点 ID。
        device_key = current_platform.ray_device_key
        # 平台设备资源键。
        if not device_key:
            # 若平台不支持 ray。
            raise RuntimeError(
                f"current platform {current_platform.device_name} does not support ray."
            )
            # 报错。
        physical_gpu_ids = ray.get_runtime_context().get_accelerator_ids()[device_key]
        # 从 Ray 运行时上下文取该 actor 加速器 ID。
        return node_id, [
            current_platform.device_control_id_to_physical_device_id(str(x))
            # 控制 ID → 物理设备 ID。
            for x in physical_gpu_ids
            # 遍历。
        ]
        # 返回节点与物理 GPU 列表。

    def initialize_worker(
        self,
        local_rank: int,
        # 本地 rank。
        env_vars: dict[str, str],
        # 覆盖式环境变量（总是覆盖）。
        driver_env_vars: dict[str, str] | None = None,
        # driver 环境变量（setdefault 语义：仅补缺不覆盖）。
        assigned_physical_gpu_ids: list[int] | None = None,
        # 逻辑→物理 GPU 映射。
    ) -> None:
        # -------------------------------------------------------------------
        # GPU 分配已知后完成完整初始化（第二阶段）。
        # -------------------------------------------------------------------
        """Complete initialization after GPU assignment is known.

        *driver_env_vars* are applied with ``setdefault`` — they fill
        in missing vars but never overwrite node-local values.
        *env_vars* always overwrite.
        *assigned_physical_gpu_ids* maps local_rank to physical CUDA device ID.
        """
        # 文档字符串：GPU 分配已知后完成初始化。
        # driver_env_vars 用 setdefault 应用（只补缺不覆盖节点本地值）；
        # env_vars 总是覆盖；assigned_physical_gpu_ids 映射 local_rank→物理设备。
        if driver_env_vars:
            # 若提供 driver 环境变量。
            for key, value in driver_env_vars.items():
                # 遍历。
                os.environ.setdefault(key, value)
                # 只补充缺失项。
        for key, value in env_vars.items():
            # 遍历覆盖式变量。
            os.environ[key] = value
            # 总是覆盖。

        if assigned_physical_gpu_ids is not None:
            # 若提供 GPU 映射。
            vllm_config = self._init_kwargs["vllm_config"]
            # 取配置。
            assert isinstance(vllm_config, VllmConfig)
            # 断言类型。
            vllm_config.parallel_config.assigned_physical_gpu_ids = (
                assigned_physical_gpu_ids
            )
            # 写入逻辑→物理映射。

        self.local_rank = local_rank
        # 保存本地 rank。
        super().__init__(
            local_rank=local_rank,
            # 本地 rank。
            **self._init_kwargs,
            # 其余初始化参数。
        )
        # 现在才调用父类（WorkerProc）完整初始化：加载模型、建立 MQ 等。

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        # -------------------------------------------------------------------
        # 初始化消息队列：
        # 与 executor 同节点的 worker 用共享内存 MQ；跨节点用 TCP。
        # -------------------------------------------------------------------
        """
        Workers on the same node as the executor use shared memory for
        both the broadcast (input) MQ and the response MQ. Workers on
        different nodes use TCP (n_local_reader=0).
        """
        # 文档字符串：与 executor 同节点的 worker 对广播/响应 MQ 都用共享内存；
        # 不同节点用 TCP（n_local_reader=0）。
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank
        )
        # 从句柄创建广播 MQ 连接。

        n_local = 1 if self._is_driver_node else 0
        # 在 driver 节点上本地读者数为 1，否则为 0（跨节点走 TCP）。
        # Use ray.util.get_node_ip_address() to get Ray's internal IP.
        # get_ip() returns host's external IP which is typically not
        # routable between nodes within the cluster.
        # 注释：用 ray.util.get_node_ip_address() 获取 Ray 内部 IP；
        # get_ip() 返回主机外部 IP，通常不能在集群节点间路由。
        self.worker_response_mq = MessageQueue(
            n_reader=1,
            # 读者数 1（executor）。
            n_local_reader=n_local,
            # 本地读者数。
            connect_ip=ray.util.get_node_ip_address(),
            # Ray 内部 IP（跨节点可路由）。
        )
        # 创建响应 MQ。
        self.peer_response_handles: list[dict] = []
        # 无对端（本实现不需要），置空。

    def wait_for_init(self) -> dict:
        # -------------------------------------------------------------------
        # 向 driver 的 wait_until_ready() 屏障应答就绪。
        # -------------------------------------------------------------------
        """Respond to the driver's wait_until_ready() barrier."""
        # 文档字符串：应答 driver 的 wait_until_ready() 屏障。
        assert self.worker_response_mq is not None
        # 断言响应 MQ 已建。
        return {
            "status": self.READY_STR,
            # 就绪状态。
            "handle": self.worker_response_mq.export_handle(),
            # 导出响应 MQ 句柄（driver 据此创建连接）。
        }
        # 返回就绪消息。

    def run(self) -> None:
        # -------------------------------------------------------------------
        # actor 主入口（经 actor.run.remote() 调用）：进入 worker 忙循环。
        # -------------------------------------------------------------------
        """Main entry point called via actor.run.remote()."""
        # 文档字符串：经 actor.run.remote() 调用的主入口。
        try:
            assert self.rpc_broadcast_mq is not None
            # 断言广播 MQ 已建。
            self.rpc_broadcast_mq.wait_until_ready()
            # 等待广播 MQ 就绪。
            assert self.worker_response_mq is not None
            # 断言响应 MQ 已建。
            self.worker_response_mq.wait_until_ready()
            # 等待响应 MQ 就绪。

            self.worker_busy_loop()
            # 进入忙循环（处理 RPC）。
        except Exception as e:
            logger.exception("RayWorkerProc failed: %s", e)
            # 记录失败。
            raise
            # 重新抛出（driver 的监控会感知）。
        finally:
            self.shutdown()
            # 退出时清理。


class RayExecutorV2(MultiprocExecutor):
    # =========================================================================
    # RayExecutorV2：基于 Ray actor + MessageQueue 的分布式执行器。
    # 继承 MultiprocExecutor 复用 MQ 控制平面与 NCCL 数据平面；
    # 唯一差异是 worker 为 Ray actor。
    # =========================================================================
    """Ray-based distributed executor using MessageQueue communication.

    Inherits from MultiprocExecutor to reuse the MQ-based control plane
    and NCCL data plane. Workers are Ray actors.

    Async scheduling is enabled, inherited from MultiprocExecutor.
    This is cricitcal for RayExecutorV2 to be performant.
    """
    # 类文档字符串：基于 MQ 通信的 Ray 分布式执行器；
    # 继承 MultiprocExecutor 以复用 MQ 控制平面与 NCCL 数据平面，worker 是 Ray actor；
    # 异步调度继承自 MultiprocExecutor，对性能至关重要。
    uses_ray: bool = True
    # 覆盖父类：使用 Ray。
    supports_pp: bool = True
    # 覆盖父类：支持 PP。

    def __init__(self, vllm_config: VllmConfig):
        # 构造函数。
        super().__init__(vllm_config)
        # 调用 MultiprocExecutor 构造（内部调 _init_executor）。

    def _build_runtime_env(self) -> dict:
        # -------------------------------------------------------------------
        # 构建 Ray actor 的 runtime_env（环境变量、nsight 等）。
        # driver 环境变量后续经 initialize_worker 用 setdefault 单独应用。
        # -------------------------------------------------------------------
        """Build a runtime_env dict for RayWorkerProc actors.

        Driver env vars are applied separately via initialize_worker
        with setdefault semantics.
        """
        # 文档字符串：为 RayWorkerProc actor 构建 runtime_env。
        # driver 环境变量经 initialize_worker 用 setdefault 单独应用。
        base = self.parallel_config.ray_runtime_env
        # 用户配置的 runtime_env。
        runtime_env: dict = copy.deepcopy(dict(base)) if base else {}
        # 深拷贝（避免修改用户配置）。

        env_vars = runtime_env.setdefault("env_vars", {})
        # 取 env_vars 段。
        env_vars.update({v: "1" for v in current_platform.ray_noset_device_env_vars})
        # 注入禁用自动设备的环境变量。
        if self.parallel_config.ray_workers_use_nsight:
            # 若启用 nsight。
            runtime_env["nsight"] = {
                "t": "cuda,cudnn,cublas",
                # 追踪库。
                "o": "'worker_process_%p'",
                # 输出名。
                "cuda-graph-trace": "node",
                # CUDA graph 追踪。
            }
            # 注入 nsight 配置。
        return runtime_env
        # 返回。

    @staticmethod
    def _get_actor_resource_kwargs() -> dict[str, Any]:
        # -------------------------------------------------------------------
        # 返回当前平台的 actor 资源参数。
        # -------------------------------------------------------------------
        """Return Ray actor resource kwargs for the current platform."""
        # 文档字符串：返回当前平台的 Ray actor 资源参数。
        num_devices = envs.VLLM_RAY_PER_WORKER_GPUS
        # 每个 worker 设备数。
        device_key = current_platform.ray_device_key
        # 平台设备资源键。
        if device_key == "GPU":
            # 标准 GPU。
            return {"num_gpus": num_devices}
            # 用 num_gpus。
        return {"num_gpus": 0, "resources": {device_key: num_devices}}
        # 否则用自定义资源键。

    @staticmethod
    def _select_tcpstore_port(local_dp_rank: int | None, master_port: int) -> int:
        # -------------------------------------------------------------------
        # 为引擎选择 torch.distributed TCPStore 端口。
        # 同节点的多个 DP 引擎若随机选端口会偶发冲突；
        # 用「本地 DP rank 做种子」为每个 rank 分到互不重叠的端口窗口。
        # -------------------------------------------------------------------
        """Pick the torch.distributed TCPStore port for this engine.

        Co-located DP engines choosing this port with a shared random search
        collide intermittently. Seeding by node-local DP rank gives each a
        disjoint window. Non-DP engines and full windows fall back to a
        random port.
        """
        # 文档字符串：为引擎选择 TCPStore 端口。同节点 DP 引擎共用随机搜索
        # 会间歇冲突；按节点本地 DP rank 播种，给每个引擎互不重叠的窗口。
        # 非 DP 引擎或窗口耗尽时回退随机端口。
        if local_dp_rank is None:
            # 非 DP 引擎。
            return get_open_port()
            # 直接随机端口。
        # Offset past the DP master port reserved range, one window per rank.
        # 注释：从 DP 主端口保留区之后偏移，每个 rank 一个窗口。
        window = 32
        # 窗口大小。
        start_port = master_port + 100 + local_dp_rank * window
        # 起始端口 = 主端口+100 + rank×窗口。
        try:
            return _get_open_port(start_port=start_port, max_attempts=window)
            # 在该窗口内找空闲端口。
        except RuntimeError:
            return get_open_port()
            # 窗口耗尽则回退随机。

    def _init_executor(self) -> None:
        # -------------------------------------------------------------------
        # 初始化 RayExecutorV2：
        # 初始化 Ray → 分配 bundle → 创建轻量 actor → 发现 GPU → 完整初始化
        # → 建立响应 MQ → 启动忙循环 → 等待就绪 → 启动监控。
        # -------------------------------------------------------------------
        """Initialize the RayExecutorV2 executor."""
        # 文档字符串：初始化 RayExecutorV2。
        self._finalizer = weakref.finalize(self, self.shutdown)
        # 注册退出清理。
        self.is_failed = False
        # 失败标志。
        self.failure_callback = None
        # 失败回调。
        self.shutting_down = False
        # 关闭标志。
        self.shutdown_lock = threading.Lock()
        # 关闭锁（防止并发 shutdown）。

        # Step 1: Initialize Ray cluster and retrieve placement group
        # 注释：第 1 步——初始化 Ray 集群并获取 placement group。
        if ray is None:
            # 若 Ray 不可用。
            raise ImportError("Using Ray backend requires installation of ray.")
            # 报错。
        initialize_ray_cluster(self.parallel_config, require_gpu_on_driver=False)
        # 初始化 Ray；不要 driver 上必须有 GPU（RayExecutorV2 所有计算都在远程 actor）。
        placement_group = self.parallel_config.placement_group
        # 获取 placement group。

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        # 取并行规模。
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )
        # 断言 world_size = TP×PP×PCP。

        # Step 2: Build bundle assignments for worker rank placement
        # while respecting VLLM_RAY_BUNDLE_INDICES.
        # 注释：第 2 步——构建 bundle 分配，考虑 VLLM_RAY_BUNDLE_INDICES。
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # 若显式指定 bundle。
            bundle_to_node_id = get_bundles_for_indices(
                placement_group,
                # PG。
                list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(","))),
                # 解析索引。
                self.world_size,
                # world_size。
            )
            # 按指定索引获取。
        else:
            bundle_to_node_id = get_bundles_sorted_by_node(placement_group)
            # 否则按节点排序获取（driver 优先）。
        driver_node = ray.get_runtime_context().get_node_id()
        # driver 节点 ID。

        bundle_assignments: list[dict[str, Any]] = []
        # 初始化 bundle 分配列表。
        for rank, (bundle_id_idx, node_id, node_ip) in enumerate(bundle_to_node_id):
            # 遍历元组。
            bundle_assignments.append(
                {
                    "rank": rank,
                    # rank。
                    "bundle_id_idx": bundle_id_idx,
                    # bundle 索引。
                    "node_id": node_id,
                    # 节点 ID。
                    "node_ip": node_ip,
                    # 节点 IP。
                }
            )
            # 记录每个 rank 的 bundle 信息。

        # Step 3: Resolve the IP for torch.distributed TCPStore.
        # The TCPStore server runs on rank 0's node, so all workers
        # must be able to reach this address.
        # 注释：第 3 步——解析 TCPStore 的 IP。TCPStore server 运行在 rank 0 节点，
        # 所有 worker 必须能到达该地址。
        dist_ip = bundle_assignments[0]["node_ip"]
        # rank 0 所在节点 IP。
        parallel_config = self.vllm_config.parallel_config
        # 取并行配置。
        port = self._select_tcpstore_port(
            parallel_config.data_parallel_rank_local,
            # 本地 DP rank。
            parallel_config.data_parallel_master_port,
            # DP 主端口。
        )
        # 选择 TCPStore 端口。
        distributed_init_method = get_distributed_init_method(dist_ip, port)
        # 生成分布式初始化地址。

        # Step 4: Create broadcast MessageQueue.
        # Workers on the driver node use shared memory; the rest use TCP.
        # 注释：第 4 步——创建广播 MQ。driver 节点 worker 用共享内存，其余用 TCP。
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        # 单块最大字节数。
        n_local = sum(1 for a in bundle_assignments if a["node_id"] == driver_node)
        # 统计 driver 节点上的本地 worker 数。
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            # 读者总数。
            n_local,
            # 本地读者数。
            max_chunk_bytes=max_chunk_bytes,
            # 单块上限。
            connect_ip=ray.util.get_node_ip_address(),
            # Ray 内部 IP。
        )
        # 创建广播 MQ。
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # 导出句柄。

        # Step 5: Spawn RayWorkerProc actors into PG bundles (deferred init).
        # Workers are created lightweight here; full initialization happens
        # in Step 7 after GPU IDs are discovered.
        # 注释：第 5 步——向 PG bundle 生成 RayWorkerProc actor（延迟初始化）。
        # 此处只轻量创建，完整初始化在第 7 步 GPU ID 发现后进行。
        self.ray_worker_handles: list[RayWorkerHandle] = []
        # 保存 actor 句柄。
        instance_id = self.vllm_config.instance_id
        # 实例 ID（用于 actor 命名唯一性）。

        # Collect driver env vars and apply but don't overwrite node-local values.
        # 注释：收集 driver 环境变量；应用时用 setdefault 不覆盖节点本地值。
        self.driver_env_vars = get_driver_env_vars(
            worker_specific_vars=WORKER_SPECIFIC_ENV_VARS,
            # 排除 worker 专属变量。
        )
        # 获取 driver 环境变量。

        runtime_env = self._build_runtime_env()
        # 构建 runtime_env。
        resource_kwargs = self._get_actor_resource_kwargs()
        # 取资源参数。

        for bundle_idx in range(self.world_size):
            # 遍历每个 bundle。
            bundle = bundle_assignments[bundle_idx]
            # 取分配信息。
            is_driver_worker = self._is_driver_worker(bundle["rank"])
            # 是否 driver worker。
            is_driver_node = bundle["node_id"] == driver_node
            # 是否在 driver 节点。

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                # PG。
                placement_group_bundle_index=bundle["bundle_id_idx"],
                # 绑定 bundle。
            )
            # 构造调度策略。

            actor_name = build_actor_name(
                instance_id, bundle["rank"], tp_size, pp_size, pcp_size
            )
            # 构建描述性 actor 名（dashboard 可见）。

            actor = (
                ray.remote(RayWorkerProc)
                # 创建 actor 类。
                .options(
                    name=actor_name,
                    # actor 名。
                    num_cpus=0,
                    # 不占 CPU。
                    **resource_kwargs,
                    # 平台资源。
                    scheduling_strategy=scheduling_strategy,
                    # 调度策略。
                    runtime_env=runtime_env,
                    # 运行环境。
                )
                # 配置选项。
                .remote(
                    vllm_config=self.vllm_config,
                    # 配置。
                    rank=bundle["rank"],
                    # rank。
                    distributed_init_method=distributed_init_method,
                    # 分布式地址。
                    input_shm_handle=scheduler_output_handle,
                    # 广播 MQ 句柄。
                    is_driver_worker=is_driver_worker,
                    # 是否 driver。
                    is_driver_node=is_driver_node,
                    # 是否 driver 节点。
                )
                # 创建 actor 实例。
            )
            # 创建 actor。

            handle = RayWorkerHandle(
                actor=actor,
                # actor。
                rank=bundle["rank"],
                # rank。
                local_rank=-1,  # Set in Step 7 after GPU ID discovery
                # local_rank 初始 -1，第 7 步 GPU ID 发现后设置。
                node_id=bundle["node_id"],
                # 节点 ID。
                bundle_id_idx=bundle["bundle_id_idx"],
                # bundle 索引。
            )
            # 创建句柄。
            self.ray_worker_handles.append(handle)
            # 保存。

        # Step 6: Discover physical GPU IDs assigned to each worker via Ray
        # runtime context.
        # 注释：第 6 步——经 Ray 运行时上下文发现各 worker 的物理 GPU ID。
        worker_node_and_physical_gpu_ids = ray.get(
            [
                h.actor.get_node_and_physical_gpu_ids.remote()
                # 远程获取 GPU ID。
                for h in self.ray_worker_handles
                # 遍历。
            ]
        )
        # 获取所有 worker 的 (node_id, 物理 GPU)。

        node_workers: dict[str, list[int]] = defaultdict(list)
        # 节点 → worker 列表。
        node_physical_gpu_ids: dict[str, list[int]] = defaultdict(list)
        # 节点 → 物理 GPU 列表。

        for i, (node_id, physical_gpu_ids) in enumerate(
            worker_node_and_physical_gpu_ids
        ):
            # 遍历。
            node_workers[node_id].append(i)
            # 记录 worker。
            node_physical_gpu_ids[node_id].extend(physical_gpu_ids)
            # 记录 GPU。
        for node_id, physical_gpu_ids in node_physical_gpu_ids.items():
            # 遍历节点。
            node_physical_gpu_ids[node_id] = sorted(physical_gpu_ids)
            # 排序。

        # Step 7: Initialize workers with local logical ranks and the
        # logical-to-physical GPU mapping discovered from Ray placement.
        # 注释：第 7 步——用本地逻辑 rank 与从 Ray placement 发现的映射初始化 worker。
        init_worker_refs = []
        # 初始化引用列表。
        for i, (node_id, _) in enumerate(worker_node_and_physical_gpu_ids):
            # 遍历 worker。
            local_rank = node_workers[node_id].index(i)
            # 节点内本地 rank。
            assigned_physical_gpu_ids = sorted(node_physical_gpu_ids[node_id])
            # 该节点物理 GPU 映射。
            worker_env_vars: dict[str, str] = {}
            # （本实现未使用覆盖式环境变量）。
            self.ray_worker_handles[i].local_rank = local_rank
            # 更新句柄的 local_rank。
            init_worker_refs.append(
                self.ray_worker_handles[i].actor.initialize_worker.remote(
                    local_rank,
                    # 本地 rank。
                    worker_env_vars,
                    # 环境变量。
                    self.driver_env_vars,
                    # driver 环境变量。
                    assigned_physical_gpu_ids=assigned_physical_gpu_ids,
                    # GPU 映射。
                )
            )
            # 远程完成完整初始化。
        # Also set on the executor-side config for consistency. The mapping
        # is per-node, so only do this when all workers share one node.
        # 注释：同时设置 executor 侧配置保持一致。映射按节点，因此仅当
        # 所有 worker 共享一个节点时才设置。
        if len(node_physical_gpu_ids) == 1:
            # 单节点。
            node_id_0 = worker_node_and_physical_gpu_ids[0][0]
            # 取节点。
            self.vllm_config.parallel_config.assigned_physical_gpu_ids = sorted(
                node_physical_gpu_ids[node_id_0]
            )
            # 设置逻辑→物理映射。
        ray.get(init_worker_refs)
        # 等待所有 worker 初始化完成。

        # Step 8: Collect response MQ handles
        # 注释：第 8 步——收集响应 MQ 句柄。
        init_results = ray.get(
            [h.actor.wait_for_init.remote() for h in self.ray_worker_handles]
        )
        # 获取就绪句柄。

        self.response_mqs: list[MessageQueue] = []
        # 初始化响应 MQ 列表。
        for i, result in enumerate(init_results):
            # 遍历。
            if result["status"] != RayWorkerProc.READY_STR:
                # 若不就绪。
                raise RuntimeError(f"Worker {i} failed to initialize: {result}")
                # 报错。
            self.response_mqs.append(
                MessageQueue.create_from_handle(result["handle"], 0)
            )
            # 创建响应 MQ 连接。

        # Step 9: Start run() before wait_until_ready() to avoid
        # deadlock — workers send subscriptions inside run().
        # 注释：第 9 步——在 wait_until_ready() 之前启动 run() 避免死锁——
        # worker 在 run() 内部发送订阅。
        for handle in self.ray_worker_handles:
            # 遍历。
            handle.run()
            # 启动忙循环。

        # Step 10: wait_until_ready() barrier
        # 注释：第 10 步——wait_until_ready() 屏障。
        self.rpc_broadcast_mq.wait_until_ready()
        # 等待广播 MQ 就绪。
        for response_mq in self.response_mqs:
            # 遍历。
            response_mq.wait_until_ready()
            # 等待响应 MQ 就绪。

        self.futures_queue = deque[FutureWrapper]()
        # 创建 Future FIFO 队列。
        self._post_init_executor()
        # 后置钩子。

        self.start_worker_monitor()
        # 启动监控线程。
        self.output_rank = self._get_output_rank()
        # 计算输出 rank。

    def start_worker_monitor(self, inline=False) -> None:
        # -------------------------------------------------------------------
        # 用 ray.wait() 轮询 run() 的 ObjectRef 监控 worker 存活。
        # 轮询带超时：阻塞调用在 Ray teardown 时可能 segfault。
        # -------------------------------------------------------------------
        """Monitor worker liveness via ray.wait() on run() ObjectRefs."""
        # 文档字符串：通过 ray.wait() 轮询 run() 的 ObjectRef 监控 worker 存活。
        run_refs = [h.run_ref for h in self.ray_worker_handles if h.run_ref is not None]
        # 收集所有 run ObjectRef。
        if not run_refs:
            # 若无引用。
            raise RuntimeError("Ray workers have not started successfully.")
            # 报错。

        self_ref = weakref.ref(self)
        # 弱引用 self。
        ref_to_rank = {
            h.run_ref: h.rank for h in self.ray_worker_handles if h.run_ref is not None
        }
        # ObjectRef → rank 映射。

        def _should_stop() -> bool:
            # 是否应停止监控。
            executor = self_ref()
            # 取 executor。
            return not executor or executor.shutting_down
            # 已回收或已关闭。

        def monitor_workers():
            # 监控线程主体。
            # Poll with a timeout rather than blocking on ray.wait()
            # because a blocking call would segfault if Ray is torn down
            # while this thread is inside it.
            # 注释：用带超时的轮询而非阻塞 ray.wait()；阻塞调用在 Ray 正被
            # teardown 时可能 segfault。
            while not _should_stop() and ray.is_initialized():
                # 持续轮询。
                try:
                    done, _ = ray.wait(run_refs, num_returns=1, timeout=5.0)
                    # 阻塞最多 5 秒等任一完成。
                except Exception:
                    logger.exception(
                        "RayWorkerMonitor: unexpected error, exiting monitor thread"
                    )
                    # 记录异常。
                    return
                    # 退出线程。

                if not done or _should_stop():
                    # 无完成或应停止。
                    continue
                    # 继续轮询。

                dead_ranks = [ref_to_rank[r] for r in done]
                # 解析死亡 worker 的 ranks。
                executor = self_ref()
                # 取 executor。
                if not executor:
                    # 已回收。
                    return
                    # 退出。
                executor.is_failed = True
                # 置位失败。
                logger.error(
                    "RayWorkerProc rank=%s died unexpectedly, shutting down executor.",
                    dead_ranks,
                )
                # 记录错误。
                executor.shutdown()
                # 关闭 executor。
                if executor.failure_callback is not None:
                    # 若注册了回调。
                    callback = executor.failure_callback
                    # 取回调。
                    executor.failure_callback = None
                    # 清空。
                    callback()
                    # 触发回调。
                return
                # 退出线程。

        t = threading.Thread(
            target=monitor_workers, daemon=True, name="RayWorkerMonitor"
        )
        # 创建监控线程。
        t.start()
        # 启动。
        self._monitor_thread = t
        # 保存线程引用。

    def _join_monitor_thread(self) -> None:
        # -------------------------------------------------------------------
        # 等待监控线程退出。必须在 teardown Ray 资源前调用——
        # 监控线程可能正处于 ray.wait() 内，若 Ray 被关闭会 segfault。
        # -------------------------------------------------------------------
        """Wait for the monitor thread to exit.

        Must be called before tearing down Ray resources — the monitor
        may be inside ray.wait() which would segfault if Ray is shut
        down underneath it. When the monitor itself calls shutdown()
        on worker death, we skip the join because the thread is about
        to return anyway.
        """
        # 文档字符串：等待监控线程退出。必须在 Ray 资源 teardown 前调用；
        # 监控线程可能正在 ray.wait() 内。当监控线程自身在 worker 死亡时调用
        # shutdown()，则跳过 join（线程马上要返回）。
        monitor = getattr(self, "_monitor_thread", None)
        # 取线程。
        if (
            monitor is not None
            # 存在。
            and monitor.is_alive()
            # 且存活。
            and threading.current_thread() is not monitor
            # 且不是自身（避免死锁）。
        ):
            monitor.join(timeout=10)
            # 最多等 10 秒。

    def shutdown(self) -> None:
        # -------------------------------------------------------------------
        # 关闭 executor：kill 所有 actor、关闭所有 MQ。
        # -------------------------------------------------------------------
        """Properly shut down the executor and its workers."""
        # 文档字符串：正确关闭 executor 与 worker。
        lock = getattr(self, "shutdown_lock", None)
        # 取关闭锁。
        if lock is None:
            # 若尚未初始化。
            return
            # 直接返回。
        with lock:
            # 加锁。
            if getattr(self, "shutting_down", False):
                # 已在关闭。
                return
                # 返回。
            self.shutting_down = True
            # 置位关闭标志。

        self._join_monitor_thread()
        # 先停监控线程（避免 ray.wait segfault）。

        for handle in getattr(self, "ray_worker_handles", []):
            # 遍历 actor。
            try:
                ray.kill(handle.actor)
                # 强杀。
                logger.debug("Killed actor rank=%d", handle.rank)
                # 记录。
            except Exception:
                logger.exception("Failed to kill actor rank=%d", handle.rank)
                # 失败仅记录。

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            # 若广播 MQ 存在。
            rpc_broadcast_mq.shutdown()
            # 关闭。
            self.rpc_broadcast_mq = None
            # 置空。

        for mq in getattr(self, "response_mqs", []):
            # 遍历响应 MQ。
            mq.shutdown()
            # 关闭。
        self.response_mqs = []
        # 清空。