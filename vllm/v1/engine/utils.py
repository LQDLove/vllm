# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import contextlib  # contextlib：上下文管理器工具（@contextmanager 装饰器）
import os  # os：操作系统接口（环境变量读取等）
import threading  # threading：线程模块（SignalCallback 专用线程）
import weakref  # weakref：弱引用（防止循环引用导致无法 GC）
from collections.abc import Callable, Iterator  # 类型标注：可调用对象、迭代器
from dataclasses import dataclass  # dataclasses：数据类装饰器
from enum import Enum, auto  # Enum：枚举；auto：自动赋值
from multiprocessing import Process, connection
# multiprocessing.Process：多进程；connection：进程间连接（sentinel 监控用）
from multiprocessing.process import BaseProcess  # BaseProcess：进程基类类型
from multiprocessing.queues import Queue  # Queue：多进程队列（tensor IPC 用）
from typing import TYPE_CHECKING, cast  # TYPE_CHECKING：条件导入；cast：类型转换标注

import msgspec  # msgspec：高性能 msgpack 序列化库
import zmq  # zmq：ZeroMQ 消息队列库（跨进程 IPC）

from vllm import envs  # vllm.envs：vLLM 环境变量设置
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
# 配置类：缓存配置、并行配置、vLLM 全局配置
from vllm.logger import init_logger  # 初始化 vLLM 日志记录器
from vllm.platforms import current_platform  # 当前平台抽象（GPU 设备管理）
from vllm.ray.ray_env import get_env_vars_to_copy  # 获取需要复制给 Ray actor 的环境变量
from vllm.utils import numa_utils  # NUMA 工具（CPU 亲和性配置）
from vllm.utils.network_utils import (
    get_open_port,  # 获取开放端口
    get_open_zmq_ipc_path,  # 获取开放 ZMQ IPC 路径（进程内通信地址）
    get_tcp_uri,  # 获取 TCP URI
    zmq_socket_ctx,  # ZMQ socket 上下文管理器
)
from vllm.utils.system_utils import get_mp_context  # 获取多进程上下文（spawn/fork）
from vllm.v1.engine.coordinator import DPCoordinator  # DP 协调器（数据并行协调进程）
from vllm.v1.executor import Executor  # 执行器抽象类
from vllm.v1.executor.ray_utils import WORKER_SPECIFIC_ENV_VARS  # worker 专属环境变量
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown
# 获取引擎客户端 ZMQ 地址；进程关闭工具函数

if TYPE_CHECKING:
    # TYPE_CHECKING 块：仅类型检查时导入，避免运行时导入开销
    from ray.util.placement_group import PlacementGroup  # Ray 放置组（DP 资源调度）

logger = init_logger(__name__)  # 模块级日志记录器

STARTUP_POLL_PERIOD_MS = 10000  # 引擎启动轮询周期（10 秒）


class CoreEngineState(Enum):
    # 核心引擎握手状态枚举
    NEW = auto()  # 新建：引擎尚未发送任何消息
    CONNECTED = auto()  # 已连接：引擎已发送 HELLO，收到初始化消息
    READY = auto()  # 已就绪：引擎完成初始化，发送 READY 消息


class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""
    # 每个数据并行 rank 一个实例，用于跟踪握手过程中的状态

    def __init__(self, index: int = 0, local: bool = True):
        # 构造函数
        self.local = local  # 是否本地引擎（与前端同机）
        self.identity = index.to_bytes(2, "little")
        # 引擎标识：DP rank 转为 2 字节小端字节数组（ZMQ 身份标识）
        self.state = CoreEngineState.NEW  # 初始状态：NEW


@dataclass
class EngineZmqAddresses:
    # ZMQ 地址集合数据类：描述所有 socket 连接地址
    # ZMQ input socket addresses for each front-end client (requests)
    # 每个前端客户端的 ZMQ input socket 地址（请求方向）
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    # 每个前端客户端的 ZMQ output socket 地址（响应方向）
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    # DP 协调器的 input socket 地址（如适用）
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    # DP 协调器的 output socket 地址（如适用）
    coordinator_output: str | None = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # 前端连接 DP 协调器的 ZMQ socket 地址
    # Not used by engine, just relayed to front-end in handshake response.
    # 引擎不使用，仅通过握手响应转发给前端
    # Only required for external DP LB case.
    # 仅在外部 DP 负载均衡模式下需要
    frontend_stats_publish_address: str | None = None


@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """
    # 启动握手时发送给每个引擎进程的元数据，
    # 包含前端 ZMQ 队列的地址（引擎需要连接这些队列）

    addresses: EngineZmqAddresses  # ZMQ 地址集合
    parallel_config: dict[str, int | str | list[int]]  # DP 并行配置参数（以字典传输）


def _make_control_bundle(node_ip: str) -> dict[str, float]:
    # 创建控制 bundle（Ray 放置组中的 CPU-only bundle）
    # The engine actor is scheduled on the final CPU-only bundle. Keep that
    # bundle colocated with the group's first GPU bundle so the actor does not
    # float to an unrelated node and reorder worker ranks away from the
    # advertised DP bootstrap host.
    # 引擎 actor 调度在最后的 CPU-only bundle 上。让该 bundle 与组的第一个
    # GPU bundle 同节点，避免 actor 漂移到无关节点导致 worker rank 与
    # 通告的 DP 引导主机错位。
    return {"CPU": 1.0, "node:" + node_ip: 0.001}
    # 返回 bundle 资源需求：1 个 CPU + 强绑定指定节点（0.001 权重为亲和性约束）


def _get_bundle_node_ip(bundle: dict[str, float]) -> str:
    # 从 bundle 资源字典中提取节点 IP
    for key in bundle:
        # 遍历资源键
        if key.startswith("node:"):
            # 找到 "node:" 前缀的键
            return key.split(":", 1)[1]
            # 分割取出 IP 部分
    raise ValueError(f"Missing node affinity in placement bundle: {bundle}")
    # 未找到节点亲和性键则抛异常


def _node_ip_from_resources(node_resources: dict) -> str | None:
    """Return the node IP encoded in a Ray per-node resource dict, or None.

    Ray advertises each node's IP as a ``node:<ip>`` resource key. The head node
    also carries ``node:__internal_head__``, and placement groups add
    ``..._group_...`` keys; both are ignored.
    """
    # 从 Ray 每节点资源字典中提取节点 IP；找不到返回 None。
    # Ray 将节点 IP 作为 ``node:<ip>`` 资源键通告。head 节点还带有
    # ``node:__internal_head__``，放置组会添加 ``..._group_...`` 键；均被忽略。
    for key in node_resources:
        # 遍历资源键
        if (
            key.startswith("node:")  # 是节点亲和性键
            and key != "node:__internal_head__"  # 排除 head 节点内部键
            and "_group_" not in key  # 排除放置组生成键
        ):
            return key.split(":", 1)[1]
            # 返回节点 IP
    return None  # 未找到返回 None


class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """
    # 工具类：管理 AsyncLLM 和 LLMEngine 使用的后台核心引擎进程的
    # 创建、就绪检测和关闭

    def __init__(
        self,
        local_engine_count: int,  # 本地引擎数量
        start_index: int,  # 全局起始 DP rank
        local_start_index: int,  # 本地起始 DP rank
        vllm_config: VllmConfig,  # vLLM 全局配置
        local_client: bool,  # 是否为本地客户端模式
        handshake_address: str,  # 握手地址
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        client_handshake_address: str | None = None,  # 客户端握手地址（可选）
        tensor_queue: Queue | None = None,  # 张量 IPC 队列（可选）
    ):
        context = get_mp_context()  # 获取多进程上下文（确定 spawn/fork）
        common_kwargs = {
            # 传递给所有引擎进程的公共参数
            "vllm_config": vllm_config,  # 配置
            "local_client": local_client,  # 本地客户端模式
            "handshake_address": handshake_address,  # 握手地址
            "executor_class": executor_class,  # 执行器类
            "log_stats": log_stats,  # 日志统计
            "tensor_queue": tensor_queue,  # 张量队列
        }

        if client_handshake_address:
            # 如果提供了客户端握手地址
            common_kwargs["client_handshake_address"] = client_handshake_address
            # 加入公共参数

        is_dp = vllm_config.parallel_config.data_parallel_size > 1
        # 是否为数据并行模式（DP>1）

        from vllm.v1.engine.core import EngineCoreProc
        # 延迟导入 EngineCoreProc（避免循环依赖）

        self.processes: list[BaseProcess] = []  # 引擎进程列表
        local_dp_ranks = []  # 本地 DP rank 列表
        for index in range(local_engine_count):
            # 为每个本地引擎创建进程
            local_index = local_start_index + index  # 本地索引
            global_index = start_index + index  # 全局索引

            # Start EngineCore in background process.
            # 在后台进程中启动 EngineCore
            local_dp_ranks.append(local_index)  # 记录本地 rank
            self.processes.append(
                context.Process(
                    # 创建多进程
                    target=EngineCoreProc.run_engine_core,
                    # 目标函数：引擎核心主入口
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    # 进程名：DP 模式带全局 rank
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                    # 传入公共参数 + 各引擎的 DP rank 信息
                )
            )

        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        # 弱引用终结器：对象被 GC 时自动关闭所有进程（防资源泄漏）
        self.manager_stopped = threading.Event()  # 管理器停止事件
        self.failed_proc_name: str | None = None  # 记录失败的进程名

        # All ranks share this config object: capture the user-provided
        # --device-ids list before the per-rank shard overwrites it. Mutating
        # the config before each proc.start() works because the spawn method
        # pickles process args at start() time, sequentially per rank.
        # 所有 rank 共享此配置对象：在每 rank 分片覆盖前捕获用户提供的
        # --device-ids 列表。由于 spawn 模式在 start() 时按 rank 顺序
        # 序列化进程参数，因此在每个 proc.start() 前修改配置是安全的。
        user_assigned_gpu_ids = vllm_config.parallel_config.assigned_physical_gpu_ids
        # 保存用户分配的物理 GPU ID（完整列表）
        try:
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # 遍历进程和本地 rank
                # Populate the logical-to-physical GPU mapping in DP for
                # platforms that cannot rely on
                # torch.accelerator.set_device_index(), and for Ray.
                # 为无法依赖 torch.accelerator.set_device_index() 的平台
                # 以及 Ray 填充 DP 的逻辑到物理 GPU 映射。
                needs_device_env_isolation = not (
                    current_platform.is_cuda_alike() or current_platform.is_xpu()
                )
                # 是否需要通过环境变量隔离设备（非 CUDA/XPU 平台）
                if is_dp and (
                    needs_device_env_isolation or vllm_config.parallel_config.use_ray
                ):
                    # DP 模式且需要隔离或使用 Ray 时
                    set_assigned_physical_gpu_ids_for_dp_rank(
                        # 为每个 DP rank 设置物理 GPU ID
                        vllm_config, local_dp_rank, user_assigned_gpu_ids
                    )

                with numa_utils.configure_subprocess(
                    # EngineCore itself does not have a TP/PP-local rank.
                    # When DP is enabled, set_assigned_physical_gpu_ids_for_dp_rank()
                    # populates the logical-to-physical mapping for this DP
                    # shard, so local_rank=0 means "the first local GPU in
                    # this shard". The actual TP/PP worker processes spawned
                    # by the executor are bound separately with their own
                    # local_rank values.
                    # EngineCore 本身没有 TP/PP 本地 rank。
                    # 启用 DP 时，set_assigned_physical_gpu_ids_for_dp_rank()
                    # 为这个 DP 分片填充逻辑到物理映射，因此 local_rank=0
                    # 表示"该分片中的第一块本地 GPU"。执行器实际派生的
                    # TP/PP worker 进程使用各自的 local_rank 单独绑定。
                    vllm_config,  # 配置
                    local_rank=0,  # 引擎核心本地 rank 固定为 0
                    dp_local_rank=local_dp_rank,  # DP 本地 rank
                    process_kind="EngineCore",  # 进程类型
                ):
                    proc.start()  # 在 NUMA 配置上下文中启动进程
        finally:
            # Kill other procs if not all are running.
            # 如果并非全部进程都启动成功，则杀死已启动的进程
            if self.finished_procs():
                # 如果有进程已结束（异常退出）
                self.shutdown()  # 关闭所有进程

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine core processes with configurable timeout."""
        # 关闭引擎核心进程，支持可配置超时
        self.manager_stopped.set()  # 标记管理器已停止
        if self._finalizer.detach() is not None:
            # 如果终结器尚未执行（detach 返回非 None 表示本次接管）
            shutdown(self.processes, timeout=timeout)  # 关闭所有进程

    def monitor_engine_liveness(self) -> None:
        """Monitor engine core process liveness."""
        # 监控引擎核心进程的存活状态

        sentinel_to_proc = {proc.sentinel: proc for proc in self.processes}
        # 构建 sentinel（进程退出信号）→ 进程的映射
        sentinels = set(sentinel_to_proc.keys())  # 所有 sentinel 集合

        while sentinels and not self.manager_stopped.is_set():
            # 循环直到所有进程退出或管理器停止
            died_sentinels = connection.wait(sentinels, timeout=1)
            # 等待任何进程退出信号（1 秒超时）

            for sentinel in died_sentinels:
                # 遍历已退出的进程
                proc = sentinel_to_proc.pop(cast(int, sentinel))  # 取出对应进程
                exitcode = proc.exitcode  # 获取退出码
                if exitcode != 0 and not self.manager_stopped.is_set():
                    # 如果非正常退出（退出码非 0）且管理器未停止
                    self.failed_proc_name = proc.name  # 记录失败进程名
            if died_sentinels:
                # 如果有进程退出
                break  # 退出循环

        self.shutdown()  # 关闭所有进程

    def sentinels(self) -> list:
        # 返回所有进程的 sentinel（供外部 poller 监控）
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        # 返回已结束进程的 {进程名: 退出码} 字典
        return {
            proc.name: proc.exitcode  # 进程名 → 退出码
            for proc in self.processes  # 遍历所有进程
            if proc.exitcode is not None  # 仅包含已退出（exitcode 非 None）的进程
        }


class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""
    # 通过专用线程在信号处理器上下文中安全触发回调
    # （信号处理器中不能安全调用非重入函数，如 queue.put）

    def __init__(self, callback: Callable[[], None]):
        # 构造函数
        self._callback = callback  # 保存回调函数
        self._event = threading.Event()  # 触发事件
        self._stopped = False  # 停止标志
        self._thread = threading.Thread(
            # 创建专用后台线程
            target=self._run,  # 线程目标
            daemon=True,  # 守护线程（不阻止进程退出）
            name="signal-callback",  # 线程名
        )
        self._thread.start()  # 启动线程

    def _run(self):
        # 线程主循环
        self._event.wait()  # 阻塞等待触发事件
        if not self._stopped:  # 如果未被停止
            self._callback()  # 执行回调（此时在安全线程上下文中）

    def trigger(self):
        # 触发回调（由信号处理器调用，仅设置事件，线程安全）
        self._event.set()

    def stop(self):
        # 停止回调线程
        self._stopped = True  # 标记停止
        self._event.set()  # 唤醒线程使其退出


def set_assigned_physical_gpu_ids_for_dp_rank(
    vllm_config: VllmConfig,  # vLLM 配置
    local_dp_rank: int,  # 本地 DP rank
    user_assigned_gpu_ids: list[int] | None = None,  # 用户分配的 GPU ID（可选）
) -> None:
    """
    Populate assigned_physical_gpu_ids on the config for the given DP rank.

    user_assigned_gpu_ids is the full (un-sharded) --device-ids list, if the
    user provided one; this DP rank's shard is sliced from it. It is passed
    explicitly rather than read from the config because callers may reuse
    one config object across DP ranks, overwriting the field each time.
    """
    # 为指定的 DP rank 在配置上填充 assigned_physical_gpu_ids。
    # user_assigned_gpu_ids 是完整的（未分片的）--device-ids 列表（如果用户提供）；
    # 该 DP rank 的分片从中切片。显式传入而不是从配置读取，
    # 因为调用方可能跨 DP rank 复用同一配置对象，每次覆盖该字段。
    world_size = vllm_config.parallel_config.world_size
    # 全局世界大小（TP×PP 总数）
    local_world_size = vllm_config.parallel_config.local_world_size
    # 本地世界大小
    evar = current_platform.device_control_env_var
    # 平台设备控制环境变量名（如 CUDA_VISIBLE_DEVICES）

    physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
        # 计算该 DP rank 对应的物理 GPU ID
        evar,  # 设备控制环境变量
        local_dp_rank,  # 本地 DP rank
        world_size,  # 世界大小
        local_world_size,  # 本地世界大小
        user_assigned_gpu_ids=user_assigned_gpu_ids,  # 用户分配的 GPU ID
    )
    vllm_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids
    # 将计算出的物理 GPU ID 写回配置


def get_physical_gpu_ids_for_local_dp_rank(
    device_control_env_var: str,  # 设备控制环境变量
    local_dp_rank: int,  # 本地 DP rank
    world_size: int,  # 世界大小
    local_world_size: int | None = None,  # 本地世界大小（可选）
    user_assigned_gpu_ids: list[int] | None = None,  # 用户分配的 GPU ID（可选）
) -> list[int]:
    """
    Returns list of physical GPU IDs for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will return [2, 3] for local_dp_rank=1.

    If user_assigned_gpu_ids is provided (e.g. from --device-ids), this DP
    rank's shard is sliced from it instead of being derived from the
    device-control env var.
    """
    # 返回指定数据并行 rank 的物理 GPU ID 列表。
    # 例如：world_size=2、local_dp_rank=1、共 4 个设备时，返回 [2, 3]。
    # 如果提供了 user_assigned_gpu_ids（如 --device-ids），则从该列表切片
    # 该 DP rank 的分片，而不是从设备控制环境变量推导。
    if local_world_size is None:
        local_world_size = world_size  # 未提供时默认等于世界大小
    if user_assigned_gpu_ids is not None:
        # 用户显式指定了 GPU 列表
        start = local_dp_rank * world_size  # 分片起始索引
        stop = start + local_world_size  # 分片结束索引
        if stop > len(user_assigned_gpu_ids):
            # 如果分片超出列表长度
            raise ValueError(
                # 抛出错误
                f"--device-ids provides {len(user_assigned_gpu_ids)} devices, "
                f"but DP rank {local_dp_rank} needs devices [{start}, {stop})"
            )
        return user_assigned_gpu_ids[start:stop]
        # 从用户列表切片返回
    try:
        return [
            # 尝试从环境变量推导物理 GPU ID
            current_platform.device_id_to_physical_device_id(i)
            # 将逻辑设备 ID 转为物理设备 ID
            for i in range(
                local_dp_rank * world_size,  # 起始逻辑 ID
                local_dp_rank * world_size + local_world_size,  # 结束逻辑 ID
            )
        ]
    except IndexError as e:
        # 捕获索引越界（设备不足）
        raise Exception(
            # 抛出包含详细信息的异常
            f"Error computing device indices for "
            f"{device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e


def _apply_dp_identity_suffix(dp_vllm_config, dp_rank: int) -> None:
    # 为 DP 配置添加身份后缀（保证跨引擎唯一）
    # Ray actor names (RayExecutorV2) and KV-connector engine_ids must
    # be unique across sibling DP engines or registration collides.
    # Use the global DP rank, not a node-local rank, since sibling DP
    # engines can span multiple nodes.
    # Ray actor 名称（RayExecutorV2）和 KV-connector engine_id 必须在
    # 兄弟 DP 引擎间唯一，否则注册会冲突。
    # 使用全局 DP rank 而非节点本地 rank，因为兄弟 DP 引擎可能跨节点。
    dp_vllm_config.instance_id = f"{dp_vllm_config.instance_id}_dp{dp_rank}"
    # 为 instance_id 追加 "_dp{rank}" 后缀
    if dp_vllm_config.kv_transfer_config is not None:
        # 如果有 KV 传输配置
        dp_vllm_config.kv_transfer_config.engine_id = (
            f"{dp_vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
        )
        # 同样为 KV 传输 engine_id 追加后缀


class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """
    # 工具类：管理 AsyncLLM 和 LLMEngine 使用的核心引擎 Ray actor 的
    # 创建、就绪检测和关闭。
    # 与 CoreEngineProcManager 不同，本类同时管理本地和远程节点的核心引擎。

    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 配置
        addresses: EngineZmqAddresses,  # ZMQ 地址
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        placement_groups: list["PlacementGroup"] | None = None,  # 放置组列表（可选）
        local_dp_ranks: list[int] | None = None,  # 本地 DP rank 列表（可选）
    ):
        import copy  # copy：深拷贝配置

        import ray  # ray：Ray 分布式框架
        from ray.runtime_env import RuntimeEnv  # Ray 运行时环境
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        # 放置组调度策略

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor
        # 延迟导入引擎核心 actor 类

        dp_size = vllm_config.parallel_config.data_parallel_size
        # 数据并行大小
        actor_class = (
            DPMoEEngineCoreActor
            # MoE 模型且 DP>1 时使用专用 actor
            if dp_size > 1 and vllm_config.model_config.is_moe
            else EngineCoreActor
            # 否则使用普通 actor
        )

        self.local_engine_actors: list[ray.ActorHandle] = []  # 本地引擎 actor
        self.remote_engine_actors: list[ray.ActorHandle] = []  # 远程引擎 actor

        env_vars_list = get_env_vars_to_copy(
            # 获取需要复制给 actor 的环境变量
            destination=actor_class.__name__,  # 目标 actor 类名
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,  # 排除 worker 专属变量
        )
        self.env_vars_dict = {
            name: os.environ[name] for name in env_vars_list if name in os.environ
        }
        # 构建环境变量字典（仅保留存在的变量）
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)
        # 创建 Ray 运行时环境

        self.addresses = addresses  # 保存地址
        self.executor_class = executor_class  # 保存执行器类
        self.log_stats = log_stats  # 保存日志统计标志
        local_engine_count = vllm_config.parallel_config.data_parallel_size_local
        # 本地引擎数量
        world_size = vllm_config.parallel_config.world_size  # 世界大小
        self.manager_stopped = threading.Event()  # 管理器停止事件
        self.failed_proc_name: str | None = None  # 失败 actor 名称

        if ray.is_initialized():
            # 如果 Ray 已初始化
            logger.info("Ray is already initialized. Skipping Ray initialization.")
            # 记录日志并跳过
        else:
            ray.init()  # 否则初始化 Ray

        parallel_config = vllm_config.parallel_config  # 简化引用
        if parallel_config.enable_elastic_ep:
            # 如果启用弹性专家并行
            from vllm.distributed.utils import create_tcp_store  # TCP store 工具

            ip = parallel_config.data_parallel_master_ip  # 主节点 IP
            store = create_tcp_store(
                # 创建 TCP 协调存储（用于分布式配置同步）
                ip,  # IP
                0,  # 端口 0 = 自动分配
                is_master=True,  # 本节点为主节点
                world_size=-1,  # 动态世界大小
                wait_for_workers=False,  # 不等待 worker
            )
            parallel_config._coord_store_port = store.port  # 记录端口
            self._coord_store = store  # 保存 store

        if placement_groups is not None:
            # 如果提供了放置组
            assert local_dp_ranks is not None, (
                # 断言必须同时提供本地 DP rank
                "local_dp_ranks must be provided if placement_groups is provided"
            )
            assert len(placement_groups) == len(local_dp_ranks), (
                # 断言两者长度一致
                "placement_groups and local_dp_ranks must have the same length"
            )
            logger.info("Using provided placement groups")  # 记录日志
            # TODO(rui): validate passed-in placement groups
            # TODO(rui)：验证传入的放置组
            self.created_placement_groups = []  # 不管理传入的放置组
        else:
            placement_groups, local_dp_ranks = (
                CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            )
            # 自动创建 DP 放置组
            self.created_placement_groups = placement_groups  # 记录创建的放置组
        assert len(placement_groups) == dp_size, (
            # 断言放置组数量与 DP 大小一致
            "Number of placement groups must match data parallel size"
        )

        self.placement_group_is_local = []  # 记录每个放置组是否本地
        refs = []  # actor 初始化引用列表
        for index, local_index, pg in zip(
            # 并行遍历 DP rank、本地 rank、放置组
            range(dp_size), local_dp_ranks, placement_groups
        ):
            dp_vllm_config = copy.deepcopy(vllm_config)  # 深拷贝配置（每 rank 独立）
            if dp_size > 1:
                # 如果 DP>1
                _apply_dp_identity_suffix(dp_vllm_config, index)
                # 应用 DP 身份后缀
            dp_vllm_config.parallel_config.placement_group = pg
            # 设置放置组
            local_client = index < local_engine_count
            # 该 rank 是否本地客户端

            # Ray XPU known issue: dpctl initializes the GPU runtime early, so
            # setting device env vars in Ray actor's initialization method
            # will not affect device selection. See:
            # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
            # Ray XPU 已知问题：dpctl 提前初始化 GPU 运行时，
            # 因此在 Ray actor 的初始化方法中设置设备环境变量不影响设备选择。
            if current_platform.is_xpu():
                # 如果是 XPU 平台
                device_evar = current_platform.device_control_env_var
                # 设备控制环境变量
                physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
                    # 获取物理 GPU ID
                    device_evar, local_index, world_size
                )
                actor_env_vars = self.env_vars_dict.copy()  # 复制环境变量
                actor_env_vars[device_evar] = ",".join(str(d) for d in physical_gpu_ids)
                # 设置设备环境变量
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)
                # 创建专用运行时环境

            actor = (
                ray.remote(actor_class)  # 创建远程 actor 类
                .options(
                    # 配置调度选项
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,  # 放置组
                        placement_group_bundle_index=world_size,
                        # bundle 索引 = 世界大小（控制 bundle 位置）
                    ),
                    runtime_env=runtime_env,  # 运行时环境
                )
                .remote(
                    # 远程初始化 actor
                    vllm_config=dp_vllm_config,  # 配置
                    executor_class=executor_class,  # 执行器类
                    log_stats=log_stats,  # 日志统计
                    local_client=local_client,  # 本地客户端标志
                    addresses=addresses,  # ZMQ 地址
                    dp_rank=index,  # DP rank
                    local_dp_rank=local_index,  # 本地 DP rank
                )
            )
            if local_client:
                # 如果是本地引擎
                self.local_engine_actors.append(actor)  # 加入本地列表
            else:
                # 否则
                self.remote_engine_actors.append(actor)  # 加入远程列表
            self.placement_group_is_local.append(local_client)
            # 记录放置组是否本地
            refs.append(actor.wait_for_init.remote())
            # 提交初始化等待任务

        ray.get(refs)  # 等待所有 actor 初始化完成
        self.run_refs = []  # 运行引用列表
        self.actor_run_ref_dict = dict()  # actor → run ref 映射
        for actor in self.local_engine_actors + self.remote_engine_actors:
            # 遍历所有 actor
            ref = actor.run.remote()  # 启动忙碌循环
            self.run_refs.append(ref)  # 记录
            self.actor_run_ref_dict[actor] = ref  # 建立映射

    @staticmethod
    def create_dp_placement_groups(
        vllm_config: VllmConfig,  # vLLM 配置
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Create placement groups for data parallel.
        """
        # 为数据并行创建放置组。

        import ray  # ray 库
        from ray._private.state import available_resources_per_node
        # 获取每节点可用资源

        logger.info("Creating placement groups for data parallel")  # 记录日志
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        # DP 主节点 IP
        dp_size = vllm_config.parallel_config.data_parallel_size  # DP 大小
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local
        # 本地 DP 大小

        available_resources = available_resources_per_node()  # 可用资源
        world_size = vllm_config.parallel_config.world_size  # 世界大小
        placement_groups: list[PlacementGroup] = []  # 放置组列表
        local_dp_ranks: list[int] = []  # 本地 DP rank 列表

        dp_master_ip_key = f"node:{dp_master_ip}"  # 主节点资源键
        nodes = sorted(
            # 排序节点：主节点优先
            available_resources.values(), key=lambda x: dp_master_ip_key not in x
        )
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        # 断言有可用节点
        assert dp_master_ip_key in nodes[0], (
            # 断言第一个节点是主节点
            f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        )

        # optionally restrict DP placement to a caller-provided node set.
        # 可选：将 DP 放置限制在调用者提供的节点集合
        requested_node_ips = {
            # 从环境变量读取请求的节点 IP 集合
            ip.strip()
            for ip in envs.VLLM_RAY_DP_PLACEMENT_NODE_IPS.split(",")
            if ip.strip()
        }
        if requested_node_ips:
            # 如果指定了节点集合
            allowed_node_ips = set(requested_node_ips)  # 允许的节点
            # The master node must host the local ranks, so it has to be allowed.
            # 主节点必须承载本地 rank，因此必须允许
            if dp_master_ip not in allowed_node_ips:
                # 如果主节点不在允许列表
                allowed_node_ips.add(dp_master_ip)  # 补充主节点
            filtered_nodes = [
                # 过滤节点
                node_resources
                for node_resources in nodes
                if _node_ip_from_resources(node_resources) in allowed_node_ips
                # 仅保留允许的节点
            ]
            logger.info(
                # 记录过滤信息
                "VLLM_RAY_DP_PLACEMENT_NODE_IPS set; restricting DP placement "
                "from %d to %d node(s): %s",
                len(nodes),  # 原节点数
                len(filtered_nodes),  # 过滤后节点数
                sorted(allowed_node_ips),  # 允许的节点 IP
            )
            nodes = filtered_nodes  # 更新节点列表

        device_str = current_platform.ray_device_key  # Ray 设备资源键
        n_node_devices: list[int] = [
            # 每节点设备数
            int(node_resources[device_str])
            for node_resources in nodes
            if device_str in node_resources
        ]
        assert n_node_devices, f"No {device_str} found in Ray cluster."
        # 断言节点有设备资源
        max_device_per_node = max(n_node_devices)  # 每节点最大设备数

        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY  # 打包策略环境变量
        _supported_pack_strategies = ("strict", "fill", "span")
        # 支持的策略：严格打包、填满、跨节点
        if pack_strategy not in _supported_pack_strategies:
            # 如果策略不支持
            raise ValueError(
                # 抛出错误
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                f"to one of {_supported_pack_strategies}"
            )

        all2all_backend = vllm_config.parallel_config.all2all_backend
        # all2all 后端配置
        if pack_strategy == "fill" and (
            # fill 策略与 DeepEP 不兼容
            all2all_backend == "deepep_high_throughput"
            or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                # 抛出错误
                "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                "does not guarantee that. "
                "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        if pack_strategy in ("strict", "fill"):
            # strict/fill 策略
            placement_strategy = "STRICT_PACK"  # 严格打包（所有 bundle 同节点）
        else:
            # span 策略
            placement_strategy = "PACK"  # 宽松打包（可跨节点）
            assert world_size > max_device_per_node, (
                # 断言世界大小超过每节点设备数（否则不需要 span）
                f"World size {world_size} is smaller than the "
                "maximum number of devices per node "
                f"{max_device_per_node}. Make sure to set "
                "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
            )

            # if we need multiple nodes per dp group, we require for now that
            # available nodes are homogeneous
            # 如果需要每 DP 组跨节点，暂时要求可用节点同构
            assert set(n_node_devices) == {max_device_per_node}, (
                # 断言节点设备数一致
                f"Nodes are not homogeneous, {nodes}"
            )
            assert world_size % max_device_per_node == 0, (
                # 断言世界大小是每节点设备数的倍数
                f"For multi-node data parallel groups, world_size ({world_size}) must "
                f"be a multiple of number of devices per node ({max_device_per_node})."
            )
            assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (
                # 断言总可用设备足够
                f"Not enough total available nodes ({len(n_node_devices)}) "
                f"and devices per node ({max_device_per_node}) "
                f"to satisfy required world size {world_size} and data parallel size "
                f"{dp_size}"
            )
            assert dp_size_local == 1, (
                # 断言 span 模式下本地 DP 大小为 1
                f"data-parallel-size-local {dp_size_local} should be set as the "
                "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                "The actual data-parallel-size-local will be auto determined."
            )

        # bundles collected for a single DP rank from multiple nodes,
        # for "span" pack strategy
        # 为单个 DP rank 从多个节点收集的 bundles（用于 span 策略）
        collected_bundles = []  # 收集的跨节点 bundle
        for node_resources in nodes:
            # 遍历节点
            node_ip = _node_ip_from_resources(node_resources)  # 节点 IP
            assert node_ip is not None, (
                # 断言节点有 IP
                f"No node IP key found in node resources: {node_resources}"
            )

            n_device_on_node = int(node_resources.get(device_str, 0))
            # 节点上的设备数
            if pack_strategy == "span" and n_device_on_node != 0:
                # span 策略且节点有设备
                # Strictly speaking,
                # dp_size_available = n_device_on_node / world_size
                # and is a fraction, but we use 1 for easier processing
                # 严格来说 dp_size_available = 节点设备数 / 世界大小，
                # 可能是分数，此处为简化处理使用 1
                dp_size_available = 1
            else:
                # 其他策略
                dp_size_available = n_device_on_node // world_size
                # 可容纳的 DP rank 数 = 设备数整除世界大小

            if node_ip == dp_master_ip:
                # 如果是主节点
                if dp_size_available < dp_size_local:
                    # 如果可用容量不足
                    raise ValueError(
                        # 抛出错误
                        f"Not enough resources to allocate {dp_size_local} DP ranks "
                        f"on DP master node {dp_master_ip}, possible to fit "
                        f"{dp_size_available} DP ranks."
                    )
                dp_size_to_allocate = dp_size_local
                # 主节点分配本地 DP 大小
            elif pack_strategy == "strict":
                # 非主节点且 strict 策略
                if dp_size_available < dp_size_local:
                    # 如果容量不足
                    logger.info(
                        # 记录日志并跳过该节点
                        "Skipping node %s as %s DP ranks could not fit, "
                        "possible to fit %s DP ranks",
                        node_ip, dp_size_local, dp_size_available,
                    )
                    continue  # 跳过该节点
                dp_size_to_allocate = dp_size_local  # 分配本地 DP 大小
            else:
                # for "pack_strategy" in "fill" and "span"
                # we always take everything that's available
                # 对于 "fill" 和 "span" 策略，总是占用所有可用容量
                dp_size_to_allocate = dp_size_available

            for i in range(dp_size_to_allocate):
                # 为每个 DP rank 创建 bundle
                device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]
                # 设备 bundle：1 个设备 + 节点亲和性约束
                if pack_strategy == "span":
                    # span 策略
                    collected_bundles += device_bundle * n_device_on_node
                    # 收集该节点所有设备 bundle
                    assert len(collected_bundles) <= world_size, (
                        # 断言不超过世界大小
                        "collected_bundles should be <= world_size, "
                        f"but got {len(collected_bundles)=} and {world_size=}"
                    )

                    # we only create a placement group if we collected enough devices
                    # 仅当收集到足够设备时才创建放置组
                    if len(collected_bundles) < world_size:
                        continue  # 未收集够，继续收集

                    control_node_ip = _get_bundle_node_ip(collected_bundles[0])
                    # 控制节点 IP = 第一个 bundle 所在节点
                    bundles = collected_bundles + [
                        _make_control_bundle(control_node_ip)
                    ]
                    # bundles = 设备 bundles + 控制 bundle
                    collected_bundles = []  # 重置收集
                else:
                    # STRICT_PACK already keeps every bundle in the placement
                    # group on one node, so the explicit node affinity on the
                    # control bundle is redundant for correctness here. Keep it
                    # anyway for consistency with the span path and to preserve
                    # intent if this scheduling strategy changes later.
                    # STRICT_PACK 已将放置组所有 bundle 保持在单节点上，
                    # 因此控制 bundle 上的显式节点亲和性在正确性上是冗余的。
                    # 此处仍保留，与 span 路径保持一致，并在未来调度策略
                    # 变化时保留意图。
                    bundles = device_bundle * world_size + [
                        _make_control_bundle(node_ip)
                    ]
                    # bundles = 世界大小个设备 bundle + 1 个控制 bundle

                pg = ray.util.placement_group(
                    # 创建放置组
                    name=f"dp_rank_{len(placement_groups)}",
                    # 名称：dp_rank_{序号}
                    strategy=placement_strategy,  # 调度策略
                    bundles=bundles,  # bundle 列表
                )
                placement_groups.append(pg)  # 加入列表
                local_dp_ranks.append(i)  # 记录本地 rank
                if len(placement_groups) == dp_size:
                    # 如果已创建足够放置组
                    break  # 跳出循环

            if len(placement_groups) == dp_size:
                # 如果已创建足够放置组
                break  # 跳出外层循环

        if len(placement_groups) < dp_size:
            # 如果创建数量不足
            raise ValueError(
                # 抛出错误
                f"Not enough resources to allocate {dp_size} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            # 断言数量一致
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            # 断言 rank 数量一致
            f"local_dp_ranks length {len(local_dp_ranks)} does not match "
            f"expected {dp_size}"
        )
        return placement_groups, local_dp_ranks  # 返回放置组和本地 rank

    @staticmethod
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig,  # 旧配置
        new_data_parallel_size: int,  # 新的 DP 大小
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Add placement groups for new data parallel size.
        """
        # 为新的 DP 大小添加放置组（弹性扩容用）
        import ray  # ray 库
        from ray._private.state import (
            available_resources_per_node,  # 可用资源
            total_resources_per_node,  # 总资源
        )
        from ray.util.state import list_nodes  # 节点列表

        old_dp_size = old_vllm_config.parallel_config.data_parallel_size
        # 旧 DP 大小
        num_pg_to_create = new_data_parallel_size - old_dp_size
        # 需要创建的放置组数量

        if num_pg_to_create <= 0:
            # 如果不需要创建
            return [], []  # 返回空

        dp_master_ip = old_vllm_config.parallel_config.data_parallel_master_ip
        # DP 主节点 IP
        world_size = old_vllm_config.parallel_config.world_size  # 世界大小

        nodes = list_nodes()  # 获取节点列表
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)
        # 排序：主节点优先
        assert nodes[0].node_ip == dp_master_ip, "The first node must be the head node"
        # 断言第一个是主节点
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            # 断言只有一个主节点
            "There can only be one head node"
        )

        available_resources = available_resources_per_node()  # 可用资源
        total_resources = total_resources_per_node()  # 总资源

        placement_groups = []  # 放置组列表
        local_dp_ranks = []  # 本地 rank 列表
        num_pg_created = 0  # 已创建数量

        device_str = current_platform.ray_device_key  # 设备资源键
        for node in nodes:
            # 遍历节点
            if num_pg_created >= num_pg_to_create:
                # 如果已创建足够
                break  # 退出

            node_ip = node.node_ip  # 节点 IP
            node_id = node.node_id  # 节点 ID
            if device_str not in available_resources[node_id]:
                # 如果该节点无可用设备
                continue  # 跳过
            available_gpus = int(available_resources[node_id][device_str])
            # 可用 GPU 数

            # Get total GPUs on this node from the node's resources
            # Ray stores node resources with node ID as key
            # 从节点资源获取该节点的总 GPU 数（按节点 ID 为键）
            total_gpus = int(total_resources[node_id][device_str])

            # Calculate used GPUs and used engines on this node
            # 计算该节点已用的 GPU 和引擎数
            used_gpus = max(0, total_gpus - available_gpus)  # 已用 GPU
            used_engines_on_node = used_gpus // world_size  # 已用引擎数

            # Calculate how many new engines this node can accommodate
            # 计算该节点可容纳的新引擎数
            available_engine_count = available_gpus // world_size

            # Create placement groups for new engines on this node
            # 为该节点的新引擎创建放置组
            for i in range(available_engine_count):
                # 遍历可用引擎槽位
                if num_pg_created >= num_pg_to_create:
                    # 如果已创建足够
                    break  # 退出

                rank = old_dp_size + num_pg_created  # 新 rank 号

                # Create bundles with node constraint for master node
                # 为主节点创建带节点约束的 bundles
                if node_ip == dp_master_ip:
                    # 主节点
                    bundles = [
                        {device_str: 1.0, "node:" + dp_master_ip: 0.001}
                    ] * world_size + [{"CPU": 1.0}]
                    # 设备 bundles + CPU 控制 bundle（带节点约束）
                else:
                    # 非主节点
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]
                    # 设备 bundles + CPU bundle

                pg = ray.util.placement_group(
                    # 创建放置组
                    name=f"dp_rank_{rank}",  # 名称
                    strategy="STRICT_PACK",  # 严格打包
                    bundles=bundles,  # bundles
                )
                placement_groups.append(pg)  # 加入列表

                # Local rank starts from the number of engines already used
                # on this node
                # 本地 rank 从该节点已用引擎数开始计数
                local_rank = used_engines_on_node + i
                # 本地 rank
                local_dp_ranks.append(local_rank)  # 加入列表
                num_pg_created += 1  # 计数 +1

        return placement_groups, local_dp_ranks  # 返回

    def scale_up_elastic_ep(
        self,
        cur_vllm_config: VllmConfig,
        new_data_parallel_size: int,
        num_redundant_experts: int,
    ) -> None:
        # 弹性专家并行扩容：启动新的引擎 actor
        import copy  # copy 模块

        import ray  # ray 库
        from ray.runtime_env import RuntimeEnv  # 运行时环境
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        # 调度策略

        from vllm.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor
        # 延迟导入 actor 类

        actor_class = (
            DPMoEEngineCoreActor  # MoE 模型用专用 actor
            if cur_vllm_config.model_config.is_moe
            else EngineCoreActor  # 否则普通 actor
        )

        cur_data_parallel_size = len(self.local_engine_actors) + len(
            self.remote_engine_actors
        )
        # 当前 DP 大小 = 本地 + 远程引擎数

        assert new_data_parallel_size > cur_data_parallel_size, (
            # 断言新大小更大
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up"
        )

        placement_groups, local_dp_ranks = self.add_dp_placement_groups(
            # 添加新放置组
            cur_vllm_config, new_data_parallel_size
        )

        world_size = cur_vllm_config.parallel_config.world_size  # 世界大小
        dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip
        # 主节点 IP
        new_local_engines = 0  # 新本地引擎计数

        runtime_env = RuntimeEnv(
            # 创建运行时环境（标记弹性启动）
            env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}
        )
        for i, (pg, local_rank) in enumerate(zip(placement_groups, local_dp_ranks)):
            # 遍历新放置组
            rank = cur_data_parallel_size + i  # 新 rank
            dp_vllm_config = copy.deepcopy(cur_vllm_config)  # 深拷贝配置
            if new_data_parallel_size > 1:
                # 如果新 DP>1
                _apply_dp_identity_suffix(dp_vllm_config, rank)  # 加身份后缀
            dp_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            # 更新 DP 大小
            dp_vllm_config.parallel_config.placement_group = pg
            # 设置放置组

            # Check if this placement group is on the head node
            # 检查放置组是否在主节点
            local_client = any(
                bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs
            )

            if local_client:
                # 如果是本地引擎
                new_local_engines += 1  # 计数 +1
                # Update data_parallel_size_local
                # 更新本地 DP 大小
                dp_vllm_config.parallel_config.data_parallel_size_local = (
                    cur_vllm_config.parallel_config.data_parallel_size_local
                    + new_local_engines
                )

            actor = (
                ray.remote(actor_class)  # 创建 actor 类
                .options(
                    # 配置
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,  # 放置组
                        placement_group_bundle_index=world_size,  # bundle 索引
                    ),
                    runtime_env=runtime_env,  # 运行时环境
                )
                .remote(
                    # 远程初始化
                    vllm_config=dp_vllm_config,  # 配置
                    executor_class=self.executor_class,  # 执行器
                    log_stats=self.log_stats,  # 日志
                    local_client=local_client,  # 本地标志
                    addresses=self.addresses,  # 地址
                    dp_rank=rank,  # DP rank
                    local_dp_rank=local_rank,  # 本地 rank
                )
            )

            if local_client:
                # 本地引擎
                self.local_engine_actors.append(actor)  # 加入本地
            else:
                # 远程引擎
                self.remote_engine_actors.append(actor)  # 加入远程
            self.created_placement_groups.append(pg)  # 记录放置组
            self.placement_group_is_local.append(local_client)  # 记录本地标志

        ray.get(
            # 等待新 actor 初始化完成
            [
                actor.wait_for_init.remote()
                for actor in (
                    self.local_engine_actors[-new_local_engines:]
                    if new_local_engines > 0
                    else []
                )
                + self.remote_engine_actors[
                    -(len(placement_groups) - new_local_engines):
                ]
            ]
        )

        actors = (
            self.local_engine_actors[-new_local_engines:]
            if new_local_engines > 0
            else []
        ) + self.remote_engine_actors[-(len(placement_groups) - new_local_engines):]
        # 新 actor 列表

        ray.get([actor.wait_for_init.remote() for actor in actors])
        for actor in actors:
            # 遍历新 actor
            ref = actor.run.remote()  # 启动忙碌循环
            self.run_refs.append(ref)  # 记录
            self.actor_run_ref_dict[actor] = ref  # 建立映射

        cur_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # 更新配置中的 DP 大小
        # Update old_vllm_config with new data_parallel_size_local if any new
        # local engines were added
        # 如果添加了新本地引擎，更新配置中的本地 DP 大小
        if new_local_engines > 0:
            cur_vllm_config.parallel_config.data_parallel_size_local += (
                new_local_engines
            )

    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        # 弹性专家并行缩容：移除引擎 actor
        import ray  # ray 库

        assert cur_data_parallel_size > new_data_parallel_size, (
            # 断言新大小更小
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down"
        )
        for _ in range(cur_data_parallel_size - new_data_parallel_size):
            # 循环移除多余引擎
            pg = self.created_placement_groups.pop()  # 弹出放置组
            is_local = self.placement_group_is_local.pop()  # 弹出本地标志
            if is_local:
                # 如果是本地
                self.local_engine_actors.pop()  # 移除本地 actor
            else:
                # 否则
                self.remote_engine_actors.pop()  # 移除远程 actor
            ray.util.remove_placement_group(pg)  # 移除放置组

    def remove_run_refs_for_scale_down(self, removed_dp_size: int) -> None:
        # 缩容后清理运行引用
        if removed_dp_size <= 0:
            # 如果移除数为 0
            return  # 直接返回
        flags = self.placement_group_is_local[-removed_dp_size:]
        # 最后 removed_dp_size 个本地标志
        li = len(self.local_engine_actors) - 1  # 本地 actor 索引
        ri = len(self.remote_engine_actors) - 1  # 远程 actor 索引
        for is_local in reversed(flags):
            # 逆序遍历标志
            if is_local:
                # 本地
                actor = self.local_engine_actors[li]  # 取本地 actor
                li -= 1  # 索引减一
            else:
                # 远程
                actor = self.remote_engine_actors[ri]  # 取远程 actor
                ri -= 1  # 索引减一
            ref = self.actor_run_ref_dict.pop(actor)  # 移除映射
            self.run_refs.remove(ref)  # 移除引用

    def get_run_refs(self):
        # 返回运行引用列表
        return self.run_refs

    def monitor_engine_liveness(self) -> None:
        # 监控引擎 actor 存活状态
        import ray  # ray 库

        while not self.manager_stopped.is_set():
            # 循环直到管理器停止
            actor_run_refs = list(self.get_run_refs())  # 获取运行引用
            if not actor_run_refs:
                # 如果没有 actor
                logger.info(
                    # 记录日志并退出
                    "There are no actors to monitor currently. "
                    "The monitoring function is about to terminate."
                )
                break  # 退出
            actor_done_refs, _ = ray.wait(actor_run_refs, timeout=5)
            # 等待任何 actor 完成（5 秒超时）
            unexpected_failure = False  # 是否意外失败
            for actor_ref in actor_done_refs:
                # 遍历完成的 actor
                if self.manager_stopped.is_set():
                    # 如果管理器已停止
                    break  # 退出
                if actor_ref not in self.get_run_refs():
                    # The run refs may have been updated by elastic scale-down.
                    # 运行引用可能已被弹性缩容更新。
                    continue  # 跳过
                try:
                    ray.get(actor_ref)  # 获取结果（若 actor 崩溃则抛异常）
                except ray.exceptions.RayActorError:
                    # 捕获 actor 错误
                    self.failed_proc_name = f"Actor {actor_ref}"  # 记录失败
                    unexpected_failure = True  # 标记意外失败

            if unexpected_failure:
                # 如果意外失败
                break  # 退出

        self.shutdown()  # 关闭所有 actor

    def shutdown(self, timeout: float | None = None) -> None:
        # 关闭所有引擎 actor
        import ray  # ray 库

        self.manager_stopped.set()  # 标记停止
        for actor in self.local_engine_actors + self.remote_engine_actors:
            # 遍历所有 actor
            ray.kill(actor)  # 杀死 actor
        for pg in self.created_placement_groups:
            # 遍历放置组
            ray.util.remove_placement_group(pg)  # 移除放置组


def get_engine_zmq_addresses(
    vllm_config: VllmConfig,  # vLLM 配置
    num_api_servers: int = 1,  # API 服务器数量
    *,
    defer_api_server_ports: bool = True,  # 是否延迟确定 API 服务器端口
) -> EngineZmqAddresses:
    """Allocate ZMQ addresses for engine-client communication.

    By default each TCP address is a ``tcp://host:0`` placeholder; the
    consumer (API-server child or single-process ``MPClient``) binds, then
    recovers the kernel-assigned port via ``getsockopt(zmq.LAST_ENDPOINT)``
    and writes it back into ``addresses`` before the engine handshake.

    Set ``defer_api_server_ports=False`` only when the consumer cannot
    report a bound port back (e.g. the Rust front-end). IPC paths are
    unaffected."""
    # 为引擎-客户端通信分配 ZMQ 地址。
    # 默认每个 TCP 地址是 ``tcp://host:0`` 占位符；消费者（API 服务器子进程
    # 或单进程 MPClient）绑定后，通过 ``getsockopt(zmq.LAST_ENDPOINT)`` 恢复
    # 内核分配的端口，并在引擎握手前写回 ``addresses``。
    # 仅当消费者无法回报绑定端口时（如 Rust 前端）设置
    # ``defer_api_server_ports=False``。IPC 路径不受影响。
    parallel_config = vllm_config.parallel_config  # 并行配置
    local_engine_count = parallel_config.data_parallel_size_local
    # 本地引擎数量
    local_start_index = parallel_config.data_parallel_rank_local
    # 本地起始 DP rank
    dp_size = parallel_config.data_parallel_size  # DP 大小
    host = parallel_config.data_parallel_master_ip  # 主节点 IP
    local_engines_only = parallel_config.local_engines_only  # 仅本地引擎

    # In offline mode there is an LLM instance per DP rank and
    # one core engine per LLM, see
    # examples/features/data_parallel/data_parallel_offline.py.
    # 离线模式下每个 DP rank 一个 LLM 实例、每个 LLM 一个核心引擎
    offline_mode = local_start_index is not None

    # client_local_only = True for cases where this front-end
    # sends requests only to colocated engines.
    # client_local_only = True 表示该前端仅向同机引擎发送请求
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )
    # NOTE(yongji): handling scaling from intra-node to inter-node
    # 注意：处理从节点内扩展到节点间扩展
    if parallel_config.enable_elastic_ep:
        # 弹性 EP 时
        client_local_only = False  # 不能仅限本地

    def _addr() -> str:
        # 生成地址的内部函数
        if client_local_only:
            # 如果仅本地
            return get_open_zmq_ipc_path()  # 使用 IPC 路径（节点内高效通信）
        return get_tcp_uri(host, 0 if defer_api_server_ports else get_open_port())
        # 否则使用 TCP 地址（端口 0 或显式开放端口）

    return EngineZmqAddresses(
        # 返回地址集合
        inputs=[_addr() for _ in range(num_api_servers)],
        # 每个 API 服务器一个 input 地址
        outputs=[_addr() for _ in range(num_api_servers)],
        # 每个 API 服务器一个 output 地址
    )


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,  # vLLM 配置
    executor_class: type[Executor],  # 执行器类
    log_stats: bool,  # 是否记录统计
    addresses: EngineZmqAddresses,  # ZMQ 地址
    num_api_servers: int = 1,  # API 服务器数量
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
        Queue | None,
    ]
]:
    """Launch engine and DP coordinator processes as needed."""
    # 按需启动引擎进程和 DP 协调器进程。

    parallel_config = vllm_config.parallel_config  # 并行配置
    dp_size = parallel_config.data_parallel_size  # DP 大小
    local_engine_count = parallel_config.data_parallel_size_local
    # 本地引擎数量
    local_start_index = parallel_config.data_parallel_rank_local
    # 本地起始 rank
    dp_rank = parallel_config.data_parallel_rank  # DP rank
    host = parallel_config.data_parallel_master_ip  # 主节点 IP
    local_engines_only = parallel_config.local_engines_only  # 仅本地引擎

    offline_mode = local_start_index is not None  # 是否离线模式

    # Create a single tensor IPC queue for sharing multimodal tensors between
    # API servers and engine core. Returns a single queue since we only support
    # DP=1 for this data flow.
    # 创建单条张量 IPC 队列，用于 API 服务器与引擎核心间共享多模态张量。
    # 返回单队列，因为此数据流仅支持 DP=1。
    tensor_queue: Queue | None = None  # 张量队列初始化为 None
    multimodal_config = vllm_config.model_config.multimodal_config
    # 多模态配置
    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        # 如果启用了 torch 共享内存张量 IPC
        tensor_queue = get_mp_context().Queue()  # 创建多进程队列

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # 在线 DP 模式下，rank 0 运行 DP 协调器进程。
    # The coordinator is needed for:
    # 协调器用于：
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    # 1. 内部/混合负载均衡：收集和发布队列统计用于负载均衡
    # 2. MoE 模型：除统计外还进行 wave 协调
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )
    # 仅在需要协调器、非离线模式、rank 0 时运行

    if run_coordinator:
        # 如果需要运行协调器
        coordinator = DPCoordinator(  # 创建协调器
            parallel_config,  # 并行配置
            enable_wave_coordination=vllm_config.model_config.is_moe,
            # MoE 模型启用 wave 协调
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        # 获取引擎 socket 地址
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )
        # 获取统计发布地址

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
        # 记录协调器 PID
    else:
        coordinator = None  # 否则不创建协调器

    if parallel_config.data_parallel_backend == "ray":
        # 如果使用 Ray 后端
        logger.info("Starting ray-based data parallel backend")  # 记录日志

        engine_actor_manager = CoreEngineActorManager(  # 创建 actor 管理器
            vllm_config=vllm_config,  # 配置
            addresses=addresses,  # 地址
            executor_class=executor_class,  # 执行器
            log_stats=log_stats,  # 日志
        )

        yield engine_actor_manager, coordinator, addresses, tensor_queue
        # 产出（生成器返回值）
        return  # 结束

    if offline_mode:
        # 离线模式
        assert local_engine_count == 1  # 断言只有一个本地引擎
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
        # 只需与本地引擎握手
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        # rank 0 持有协调器，因此在外部和内部 dplb 模式下都与所有核心握手。
        # 注意这也覆盖了零本地引擎且 rank 0 无头的情况。
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
        # 与所有引擎握手
    else:
        # Rank > 0 handshakes with just the local cores it is managing.
        # rank > 0 只与其管理的本地核心握手。
        assert local_engines_only, (
            # 断言仅本地引擎模式
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]
        # 仅与本地引擎握手

    # Whether the started engines will handshake only with co-located
    # front-end processes. In external_dp_lb mode, ranks > 0 handshake with
    # their co-located frontend and also the rank 0 front-end, and hence this
    # will be False.
    # 已启动的引擎是否仅与同机前端进程握手。
    # 外部 dp_lb 模式下，rank > 0 与同机前端和 rank 0 前端都握手，因此为 False。
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # NOTE(yongji): handling scaling from intra-node to inter-node
    # 注意：处理从节点内扩展到节点间扩展
    if parallel_config.enable_elastic_ep:
        # 弹性 EP 时
        handshake_local_only = False  # 不能仅限本地

    # Preserve "port=0 means auto-pick" for the handshake address, which
    # is consumed by engines spawned in this process and so cannot defer
    # port resolution to bind time.
    # 为握手地址保留"port=0 表示自动选择"语义，该地址由本进程派生的
    # 引擎消费，因此不能在绑定时才解析端口。
    rpc_port = parallel_config.data_parallel_rpc_port or get_open_port()
    # RPC 端口
    handshake_address = get_engine_client_zmq_addr(handshake_local_only, host, rpc_port)
    # 获取握手地址

    if local_engines_only and dp_rank > 0:
        # 仅本地引擎且 rank>0
        assert not handshake_local_only  # 断言非仅本地握手
        local_handshake_address = get_open_zmq_ipc_path()  # 本地握手地址（IPC）
        client_handshake_address = local_handshake_address  # 客户端握手地址
    else:
        # 其他情况
        local_handshake_address = handshake_address  # 使用同一地址
        client_handshake_address = None  # 无客户端握手地址

    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        # 创建握手 socket（绑定本地地址）
        # Start local engines.
        # 启动本地引擎
        if local_engine_count:
            # 如果有本地引擎
            local_engine_manager = CoreEngineProcManager(
                # 创建进程管理器
                vllm_config=vllm_config,  # 配置
                executor_class=executor_class,  # 执行器
                log_stats=log_stats,  # 日志
                handshake_address=handshake_address,  # 握手地址
                client_handshake_address=client_handshake_address,
                # 客户端握手地址
                local_client=True,  # 本地客户端
                local_engine_count=local_engine_count,  # 本地引擎数
                start_index=dp_rank,  # 起始索引
                local_start_index=local_start_index or 0,  # 本地起始索引
                tensor_queue=tensor_queue,  # 张量队列
            )
        else:
            # 无本地引擎
            local_engine_manager = None  # 管理器为 None

        yield local_engine_manager, coordinator, addresses, tensor_queue
        # 生成器产出（由 MPClient 消费后启动引擎）

        # Now wait for engines to start.
        # 现在等待引擎启动
        wait_for_engine_startup(
            # 等待引擎启动完成
            handshake_socket,  # 握手 socket
            addresses,  # 地址
            engines_to_handshake,  # 待握手引擎
            parallel_config,  # 并行配置
            dp_size > 1 and vllm_config.model_config.is_moe,
            # 协调 DP（MoE 且 DP>1）
            vllm_config.cache_config,  # 缓存配置
            local_engine_manager,  # 进程管理器
            coordinator.proc if coordinator else None,
            # 协调器进程（如存在）
        )


def wait_for_engine_startup(
    handshake_socket: zmq.Socket,  # 握手 socket
    addresses: EngineZmqAddresses,  # ZMQ 地址
    core_engines: list[CoreEngine],  # 待握手引擎列表
    parallel_config: ParallelConfig,  # 并行配置
    coordinated_dp: bool,  # 是否协调 DP
    cache_config: CacheConfig,  # 缓存配置
    proc_manager: CoreEngineProcManager | None,  # 进程管理器
    coord_process: Process | None,  # 协调器进程
):
    # Wait for engine core process(es) to send ready messages.
    # 等待引擎核心进程发送就绪消息。
    local_count = parallel_config.data_parallel_size_local  # 本地引擎数
    remote_count = len(core_engines) - local_count  # 远程引擎数
    # [local, remote] counts
    # [本地, 远程] 计数
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    # 待连接数、待启动数
    poller = zmq.Poller()  # 创建 ZMQ 轮询器
    poller.register(handshake_socket, zmq.POLLIN)  # 注册握手 socket

    remote_should_be_headless = (
        # 远程引擎是否应为无头模式
        not parallel_config.data_parallel_hybrid_lb
        and not parallel_config.data_parallel_external_lb
    )
    # 非混合/外部负载均衡模式时远程引擎应为无头

    if proc_manager is not None:
        # 如果进程管理器存在
        for sentinel in proc_manager.sentinels():
            # 注册进程退出哨兵
            poller.register(sentinel, zmq.POLLIN)
    if coord_process is not None:
        # 如果协调器进程存在
        poller.register(coord_process.sentinel, zmq.POLLIN)  # 注册退出哨兵
    while any(conn_pending) or any(start_pending):
        # 循环直到所有引擎连接且启动
        events = poller.poll(STARTUP_POLL_PERIOD_MS)  # 轮询（10 秒超时）
        if not events:
            # 如果没有事件
            if any(conn_pending):
                # 有待连接引擎
                logger.debug(
                    # 记录调试日志
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                # 有待启动引擎
                logger.debug(
                    # 记录调试日志
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue  # 继续轮询
        if len(events) > 1 or events[0][0] != handshake_socket:
            # One of the local core processes exited.
            # 某个本地核心进程退出了。
            finished = proc_manager.finished_procs() if proc_manager else {}
            # 获取已结束进程
            if coord_process is not None and coord_process.exitcode is not None:
                # 如果协调器也已退出
                finished[coord_process.name] = coord_process.exitcode
                # 加入已结束列表
            raise RuntimeError(
                # 抛出初始化失败错误
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"
            )

        # Receive HELLO and READY messages from the input socket.
        # 从输入 socket 接收 HELLO 和 READY 消息。
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        # 接收多部分消息（引擎身份 + 消息体）
        eng_index = int.from_bytes(eng_identity, "little")  # 解析引擎索引
        engine = next((e for e in core_engines if e.identity == eng_identity), None)
        # 查找对应的引擎对象
        if engine is None:
            # 如果找不到
            raise RuntimeError(
                # 抛出错误
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        msg = msgspec.msgpack.decode(ready_msg_bytes)  # 解码消息
        status, local, headless = msg["status"], msg["local"], msg["headless"]
        # 解析状态、本地标志、无头标志
        if local != engine.local:
            # 验证本地标志一致
            raise RuntimeError(
                # 抛出错误
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # Remote engines must be headless iff we aren't in hybrid dp lb mode.
        # 远程引擎必须是无头的，当且仅当我们不在混合 dp lb 模式。
        if not local and headless != remote_should_be_headless:
            # 验证远程引擎的无头标志
            if headless:
                # 应为有头但实际无头
                raise RuntimeError(
                    # 抛出错误
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:
                # 应为无头但实际有头
                raise RuntimeError(
                    # 抛出错误
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # Send init message with DP config info.
            # 引擎发送 HELLO 且状态为 NEW：发送带 DP 配置信息的初始化消息
            init_message = msgspec.msgpack.encode(
                # 编码初始化消息
                EngineHandshakeMetadata(
                    addresses=addresses,  # ZMQ 地址
                    parallel_config={
                        # DP 配置字典
                        k: getattr(parallel_config, k)  # 动态获取属性
                        for k in (
                            "data_parallel_master_ip",  # 主节点 IP
                            "data_parallel_master_port",  # 主节点端口
                            "_data_parallel_master_port_list",  # 端口列表
                            "data_parallel_size",  # DP 大小
                        )
                    }
                    if coordinated_dp  # 仅协调 DP 时发送
                    else {},  # 否则空字典
                )
            )
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            # 发送初始化消息（零拷贝）
            conn_pending[0 if local else 1] -= 1  # 减少待连接数
            start_pending[0 if local else 1] += 1  # 增加待启动数
            engine.state = CoreEngineState.CONNECTED  # 更新状态为已连接
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # 引擎发送 READY 且状态为已连接：启动完成
            # Validate config hash consistency across DP workers for MoE models.
            # 验证 MoE 模型各 DP worker 的配置哈希一致性。
            if coordinated_dp:
                # 协调 DP 时
                worker_config_hash = msg.get("parallel_config_hash")
                # 获取 worker 配置哈希
                expected_hash = parallel_config.compute_hash()  # 计算期望哈希
                if worker_config_hash != expected_hash:
                    # 如果不一致
                    raise RuntimeError(
                        # 抛出错误
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            start_pending[0 if local else 1] -= 1  # 减少待启动数
            engine.state = CoreEngineState.READY  # 更新状态为已就绪
        else:
            # 其他情况：非法状态转换
            raise RuntimeError(
                # 抛出错误
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        logger.debug(
            # 记录调试日志
            "%s from %s core engine process %s.",
            status,  # 状态
            "local" if local else "remote",  # 本地/远程
            eng_index,  # 引擎索引
        )