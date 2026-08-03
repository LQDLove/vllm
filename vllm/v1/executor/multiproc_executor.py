# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/multiproc_executor.py
# 本文件实现「多进程执行器」MultiprocExecutor —— vLLM v1 默认的多 GPU 后端。
# 核心设计：
#   1. 每个 Worker 运行在一个独立 OS 进程中（WorkerProc / worker_main）。
#   2. 控制平面：共享内存 MessageQueue（shm_broadcast）广播 SchedulerOutput / RPC。
#   3. 数据平面：NCCL（torch.distributed）做模型张量通信。
#   4. 异步调度：WorkerProc 内专用线程搬运模型输出，实现 CPU 调度与 GPU 推理重叠。
#   5. 容错：后台监控线程监听 worker 进程存活，异常死亡时回调引擎触发关闭。
# =============================================================================
import multiprocessing
# 导入 multiprocessing：用于 process.sentinel 监控子进程存活（connection.wait）。
import os
# 导入 os：设置环境变量（如 OMP_NUM_THREADS）、关闭继承的文件描述符。
import pickle
# 导入 pickle：序列化可调用对象（与 cloudpickle 配合），经 MQ 传给 worker。
import queue
# 导入 queue：worker 内异步输出队列（async_output_queue）。
import signal
# 导入 signal：在 worker 子进程中注册 SIGTERM/SIGINT 信号处理函数（优雅退出）。
import threading
# 导入 threading：启动 worker 响应搬运线程、死亡管道监控线程、worker 存活监控线程。
import time
# 导入 time：RPC 超时 deadline 计算、退出等待循环。
import traceback
# 导入 traceback：worker 异常时把完整堆栈附加到异常信息（add_note）。
import weakref
# 导入 weakref：弱引用 self 供后台监控线程访问 executor（避免引用环/泄漏）。
from collections import deque
# 导入 deque：FIFO 队列保存 FutureWrapper，保证多个 RPC 结果的消费顺序。
from collections.abc import Callable, Sequence
# 导入类型：
#   Callable —— 可调用对象（RPC method）；
#   Sequence —— 序列类型（response_mqs 切片）。
from concurrent.futures import Future, InvalidStateError
# 导入 Future（异步结果）与 InvalidStateError（设置已完成 Future 时的抑制异常）。
from contextlib import suppress
# 导入 suppress 上下文管理器：静默忽略 InvalidStateError 等预期异常。
from dataclasses import dataclass
# 导入 dataclass：定义 UnreadyWorkerProcHandle / WorkerProcHandle 数据结构。
from enum import Enum, auto
# 导入 Enum/auto：定义 RPC 响应状态枚举 ResponseStatus（SUCCESS/FAILURE）。
from functools import partial
# 导入 partial：构建 KV 输出聚合的偏函数（绑定 aggregator 与输出 rank）。
from multiprocessing.connection import Connection
# 导入 Connection：类型标注 ready/death 管道。
from multiprocessing.process import BaseProcess
# 导入 BaseProcess：类型标注 worker 进程对象。
from multiprocessing.synchronize import Lock as LockType
# 导入 Lock 类型：类型标注进程共享锁。
from threading import Thread
# 导入 Thread：后台监控线程（WorkerProc 健康监控）。
from typing import Any, cast
# 导入类型：Any（宽松标注）、cast（wait_for_ready 中把 list[None] 强转）。

import cloudpickle
# 导入 cloudpickle：可序列化闭包/局部函数（RPC 传递可调用对象时用）。
import torch
# 导入 torch（模块顶部引入副作用，供 worker 环境使用）。

import vllm.envs as envs
# 导入 vllm 环境变量模块（MQ 大小、超时、关闭超时等配置）。
from vllm.config import VllmConfig
# 导入 VllmConfig（worker 初始化参数之一）。
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
# 导入分布式销毁函数：
#   destroy_distributed_environment —— 关闭 TCPStore 等分布式环境；
#   destroy_model_parallel —— 销毁模型并行通信组。
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
# 导入共享内存广播队列：
#   MessageQueue —— 多读多写共享内存消息队列（executor 与 worker 的控制/结果通道）；
#   Handle —— MessageQueue 的导出句柄（跨进程传递 MQ 引用）。
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
# 导入 KV 输出聚合器（disaggregated serving 场景聚合所有 worker 输出）。
from vllm.distributed.parallel_state import (
    get_dcp_group,
    # 获取 DCP（数据上下文并行）通信组，用于 worker 进程命名。
    get_dp_group,
    # 获取 DP 通信组（进程命名）。
    get_ep_group,
    # 获取 EP（专家并行）通信组（进程命名）。
    get_inner_dp_world_group,
    # 获取 DP 组内的世界通信组：多节点时用于跨节点建立 MQ 广播。
    get_pcp_group,
    # 获取 PCP（prefill 上下文并行）通信组（进程命名）。
    get_pp_group,
    # 获取 PP 通信组（进程命名）。
    get_tp_group,
    # 获取 TP 通信组（进程命名）。
    model_parallel_is_initialized,
    # 判断模型并行组是否已初始化（决定进程命名的时机）。
)
from vllm.envs import enable_envs_cache
# 导入环境变量缓存开关：worker 初始化完成后锁定 env 缓存以提升 env 读取性能。
from vllm.logger import init_logger
# 导入日志初始化函数。
from vllm.platforms import current_platform
# 导入当前平台抽象。
from vllm.tracing import instrument, maybe_init_worker_tracer
# 导入追踪工具：
#   instrument —— worker 初始化 span 埋点；
#   maybe_init_worker_tracer —— 在 worker 子进程中初始化 OpenTelemetry tracer。
from vllm.utils import numa_utils
# 导入 NUMA 工具：为 worker 子进程做 NUMA 绑定 / 配置 OpenMP 亲和性。
from vllm.utils.network_utils import (
    get_distributed_init_method,
    # 生成 torch.distributed 初始化地址。
    get_ip,
    # 获取本机 IP。
    get_loopback_ip,
    # 获取回环 IP（127.0.0.1）：单机通信用它避免多网卡歧义。
    get_open_port,
    # 获取空闲端口。
)
from vllm.utils.ompmultiprocessing import OMPProcessManager
# 导入 OpenMP 进程管理器：CPU 后端下为每个 worker 配置 OpenMP 线程亲和性。
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    # 根据平台/环境决定是否强制 spawn 启动方式。
    decorate_logs,
    # 给 worker 日志加前缀（如 "Worker_TP0"），便于区分各进程日志。
    get_mp_context,
    # 获取 multiprocessing 上下文（spawn/fork）。
    set_process_title,
    # 设置进程名（便于 ps/top 观察）。
)
from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
# 导入调度器输出类型。
from vllm.v1.executor.abstract import Executor, FailureCallback
# 导入抽象基类 Executor 与失败回调类型。
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
# 导入 GPU→NIC 映射工具（RDMA 网络设备设置）。
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
# 导入模型输出类型。
from vllm.v1.worker.worker_base import WorkerWrapperBase
# 导入 worker 包装基类（WorkerProc 内持有 wrapper，rpc_rank 为 local_rank）。

logger = init_logger(__name__)
# 初始化本模块日志。


class FutureWrapper(Future):
    # =========================================================================
    # FutureWrapper：多进程 RPC 的异步结果包装。
    # 关键点：维护一个全局 FIFO 队列 futures_queue。
    # 当调用 .result() 时，会按入队顺序「先服务排在自己前面的 Future」，
    # 从而严格保证多轮 RPC 的完成/消费顺序与提交顺序一致（FIFO 语义）。
    # =========================================================================
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        # futures_queue：全局 FIFO 队列引用（executor 上的共享 deque）。
        get_response: Callable[[], Any],
        # get_response：实际从 MQ 读取响应的回调函数。
        aggregate: Callable = lambda x: x,
        # aggregate：可选的输出聚合函数（默认恒等）。
    ):
        self.futures_queue = futures_queue
        # 保存全局队列引用。
        self.get_response = get_response
        # 保存响应读取回调。
        self.aggregate = aggregate
        # 保存聚合函数。
        super().__init__()
        # 调用父类 Future 构造（pending 状态）。
        self.futures_queue.appendleft(self)
        # appendleft 入队：队尾 pop，队头 appendleft 构成 FIFO（先入先出）。
        # 注意 deque 用 appendleft + pop 实现，append 在左侧、pop 从右侧弹出最早的元素。

    def result(self, timeout=None):
        # -------------------------------------------------------------------
        # 重写 result()：先排空前面所有 Future，再返回自身结果。
        # -------------------------------------------------------------------
        if timeout is not None:
            # 本实现不支持超时。
            raise RuntimeError("timeout not implemented")
            # 抛错。
        # Drain any futures ahead of us in the queue.
        # 注释：排空队列中排在自己前面的所有 Future。
        while not self.done():
            # 只要自身尚未完成。
            future = self.futures_queue.pop()
            # 从队列右侧弹出最先入队的 Future。
            future._wait_for_response()
            # 调用其 _wait_for_response() 读取响应并填充自身结果。
            # 由于后入队的 Future 在前面的 Future 完成后才能轮到，
            # 循环保证「前面所有 Future 都完成」后才退出。
        return super().result()
        # 此时自身必已完成，返回结果（异常也会在此抛出）。

    def _wait_for_response(self):
        # -------------------------------------------------------------------
        # 实际读取响应并填充本 Future 的结果/异常。
        # -------------------------------------------------------------------
        try:
            response = self.aggregate(self.get_response())
            # 调用 get_response 从 MQ 读取原始响应，再经 aggregate 聚合。
            with suppress(InvalidStateError):
                # 若 Future 已被设置（如超时路径），静默忽略重复设置——
                # 使用 suppress 捕获 InvalidStateError。
                self.set_result(response)
                # 设置结果为聚合后的响应。
        except Exception as e:
            with suppress(InvalidStateError):
                # 同上，异常路径也忽略重复设置。
                self.set_exception(e)
                # 将异常存入 Future。


class MultiprocExecutor(Executor):
    # =========================================================================
    # MultiprocExecutor：多进程分布式执行器（默认多 GPU 后端）。
    # 特性：
    #   - 支持 PP（流水线并行）与 PCP（prefill 上下文并行）。
    #   - 每个 worker 是独立 OS 进程，通过共享内存 MQ 与控制端通信。
    #   - 支持多节点（nnodes_within_dp > 1）部署（跨节点走 NCCL MQ）。
    #   - 支持异步调度（supports_async_scheduling() == True）。
    # =========================================================================
    supports_pp: bool = True
    # 覆盖父类：支持流水线并行。

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        # 构造函数。
        self.monitor_workers = monitor_workers
        # 是否启动 worker 存活监控线程（某些场景（如测试）可关闭）。
        super().__init__(vllm_config)
        # 调用父类构造（内部会调用 _init_executor()）。

    def _init_executor(self) -> None:
        # -------------------------------------------------------------------
        # 初始化 executor：拉起所有 worker 进程、建立 MQ、等待就绪。
        # -------------------------------------------------------------------
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        # 注释：注册退出时的清理回调，确保 worker 进程被终止。
        self._finalizer = weakref.finalize(self, self.shutdown)
        # 用 weakref.finalize 注册 self.shutdown：对象被 GC 或进程退出时自动清理。
        self.is_failed = False
        # 初始化失败标志。
        self.failure_callback: FailureCallback | None = None
        # 初始化失败回调为 None（后续 register_failure_callback 设置）。

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        # 获取 TP / PP / PCP 大小。
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )
        # 断言：全局 world_size 必须恰好等于 TP×PP×PCP。

        set_multiprocessing_worker_envs()
        # 设置子进程环境（强制 spawn、OMP_NUM_THREADS 收敛等）。

        # use the loopback address get_loopback_ip() for communication.
        # 注释：单机场景使用回环地址进行通信（避免多网卡歧义）。
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        # 为 torch.distributed 生成 tcp://127.0.0.1:port 初始化地址。
        self.rpc_broadcast_mq: MessageQueue | None = None
        # 广播 MQ（executor 发往所有 worker 的 SchedulerOutput/RPC），初始为 None。
        scheduler_output_handle: Handle | None = None
        # 广播 MQ 的导出句柄（会传给所有 worker，它们据此连接），初始为 None。

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        # 注释：初始化 worker 并建立 SchedulerOutput / ModelRunnerOutput 的 MQ。
        if self.parallel_config.node_rank_within_dp == 0:
            # 仅 DP 组内的「领导节点」创建广播 MQ。
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            # 注释：每个 DP 组有各自的领导 executor，负责本 DP 组的 MQ。
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            # 计算 MQ 单块消息最大字节数（MB → B）。
            mq_connect_ip = get_ip()
            # 获取本机对外 IP 作为 MQ 连接地址（多节点场景节点间需要可达）。
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                # 节点序号。
                self.parallel_config.node_rank_within_dp,
                # DP 组内节点序号。
                self.parallel_config.master_addr,
                # 主节点地址。
                mq_connect_ip,
                # MQ 连接 IP。
                self.world_size,
                # 全局 world_size。
                self.local_world_size,
                # 本节点 worker 数。
            )
            # 打印 DP 领导节点的拓扑信息，便于排障。
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            # 创建共享内存广播 MQ：world_size 个读者（每 worker 一个），
            # local_world_size 个本地读者（本节点 worker 走共享内存）。
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
            # 导出 MQ 句柄，稍后传给每个 worker 进程创建各自的连接。
        # Create workers
        # 注释：创建 worker 进程。
        context = get_mp_context()
        # 获取 multiprocessing 上下文（spawn/fork）。
        shared_worker_lock = context.Lock()
        # 创建进程共享锁（跨所有 worker 保证 CUDA 初始化等互斥）。
        unready_workers: list[UnreadyWorkerProcHandle] = []
        # 记录尚未就绪的 worker 句柄列表。
        success = False
        # 初始化成功标志。
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # 计算本节点内 worker 的起始全局 rank：本节点在 DP 组内的序号 × 每节点 worker 数。
            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            # 注释：fork 模式下跟踪被 worker 继承的 socket fd，
            # 以便在后续 worker 中显式关闭，避免 fd 泄漏。
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )
            # 若启动方式为 fork，则维护继承 fd 列表；spawn 无需（不继承）。

            # For CPU backend only, to setup OpenMP threads affinity
            # 注释：仅 CPU 后端需要设置 OpenMP 线程亲和性。
            cpu_omp_manager = OMPProcessManager(self.vllm_config)
            # 创建 OpenMP 进程管理器（CPU 后端绑定线程亲和性）。
            for local_rank in range(self.local_world_size):
                # 遍历本节点所有本地 rank。
                global_rank = global_start_rank + local_rank
                # 计算本地 rank 对应的全局 rank。
                is_driver_worker = self._is_driver_worker(global_rank)
                # 判断该 worker 是否为 driver（TP rank 0 的是 driver）。
                with cpu_omp_manager.configure_omp_envs(
                    rank=global_rank, local_rank=local_rank
                ):
                    # 为 worker 配置 OpenMP 环境（只影响子进程）。
                    unready_worker_handle = WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        # 传入配置。
                        local_rank=local_rank,
                        # 本地 rank。
                        rank=global_rank,
                        # 全局 rank。
                        distributed_init_method=distributed_init_method,
                        # 分布式初始化地址。
                        input_shm_handle=scheduler_output_handle,
                        # 输入（广播）MQ 句柄。
                        shared_worker_lock=shared_worker_lock,
                        # 进程共享锁。
                        is_driver_worker=is_driver_worker,
                        # 是否 driver。
                        inherited_fds=inherited_fds,
                        # 待关闭的继承 fd 列表（fork 用）。
                    )
                    # 创建 worker 子进程。
                unready_workers.append(unready_worker_handle)
                # 记录未就绪句柄。
                if inherited_fds is not None:
                    # fork 模式下。
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    # 记录死亡管道写端 fd（后续 worker 需关闭）。
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())
                    # 记录就绪管道读端 fd。

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            # 注释：必须先创建全部 worker 再等待就绪，避免死锁——
            # 因为 worker 的 init_device() 会做设备同步，若逐个等待可能相互阻塞。

            # Wait for all local workers to be ready.
            # 注释：等待所有本地 worker 就绪。
            self.workers = WorkerProc.wait_for_ready(unready_workers)
            # 阻塞等待 worker 通过 ready 管道报告初始化完成（返回 WorkerProcHandle 列表）。

            # The workers have inherited their thread count (see
            # set_multiprocessing_worker_envs); this process only schedules, so
            # it gets no benefit from torch intra-op parallelism, just CPU
            # contention with them.
            set_torch_threads_for_runtime()

            # Start background thread to monitor worker health if not in headless mode.
            # 注释：非无头模式下启动后台线程监控 worker 健康。
            if self.monitor_workers:
                # 若启用了监控。
                self.start_worker_monitor()
                # 启动 worker 存活监控线程。

            self.response_mqs = []
            # 初始化响应 MQ 列表。
            # Only leader node have remote response mqs
            # 注释：仅领导节点持有（含远程）响应 MQ。
            if self.parallel_config.node_rank_within_dp == 0:
                # DP 领导节点。
                for rank in range(self.world_size):
                    # 遍历所有全局 rank。
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        # 本节点 worker 的响应 MQ（本进程直接持有）。
                        assert local_message_queue is not None
                        # 断言非空。
                        self.response_mqs.append(local_message_queue)
                        # 加入响应 MQ 列表。
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        # 远程节点 worker 的响应 MQ（经 worker 0 的隧道转接）。
                        assert remote_message_queue is not None
                        # 断言非空。
                        self.response_mqs.append(remote_message_queue)
                        # 加入列表。

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            # 注释：确保 MQ 就绪。顺序不能调整，否则会死锁；
            # 必须与 WorkerProc 中的顺序保持一致。

            # Wait for all input mqs to be ready.
            # 注释：等待所有输入（广播）MQ 就绪。
            if self.rpc_broadcast_mq is not None:
                # 若创建了广播 MQ。
                self.rpc_broadcast_mq.wait_until_ready()
                # 阻塞等待所有 worker 完成订阅。
            # Wait for all remote response mqs to be ready.
            # 注释：等待所有远程响应 MQ 就绪。
            for response_mq in self.response_mqs:
                # 遍历每个响应 MQ。
                response_mq.wait_until_ready()
                # 等待其就绪。

            self.futures_queue = deque[FutureWrapper]()
            # 创建全局 Future FIFO 队列（保证异步 RPC 结果消费顺序）。
            self._post_init_executor()
            # 调用后置初始化钩子（子类可覆盖；默认空）。
            success = True
            # 标记初始化成功。
        finally:
            if not success:
                # 若初始化失败，需要清理已创建的 worker。
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                # 注释：失败时清理 worker 进程，先关闭死亡写端以通知 worker 退出。
                for uw in unready_workers:
                    # 遍历所有未就绪 worker。
                    if uw.death_writer is not None:
                        # 若死亡写端存在。
                        uw.death_writer.close()
                        # 关闭它（子进程读到 EOF 即知父进程放弃）。
                        uw.death_writer = None
                        # 置空避免重复关闭。
                self._ensure_worker_termination([uw.proc for uw in unready_workers])
                # 强制终止所有 worker 进程（等待→SIGTERM→SIGKILL）。

        self.output_rank = self._get_output_rank()
        # 计算输出 rank（TP rank 0 + 最后 PP stage 的 worker），后续只从它取结果。

    def get_response_mqs(self, unique_reply_rank: int = -1) -> list[MessageQueue]:
        # -------------------------------------------------------------------
        # 获取响应 MQ 列表；unique_reply_rank=-1 返回全部，否则只返回该 rank 的。
        # -------------------------------------------------------------------
        assert unique_reply_rank >= -1 and unique_reply_rank < self.world_size, (
            f"unique_reply_rank must be -1 or < world_size,"
            f"unique_reply_rank = {unique_reply_rank}, "
            f"world_size={self.world_size}"
        )
        # 断言参数范围合法。
        ranks = (
            [unique_reply_rank] if unique_reply_rank != -1 else range(self.world_size)
        )
        # 计算需要返回的 rank 列表。
        return [self.workers[rank].worker_response_mq for rank in ranks]
        # 返回对应 worker 的响应 MQ 列表。

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        # -------------------------------------------------------------------
        # 计算并校验并行规模，返回 (tp_size, pp_size, pcp_size)。
        # -------------------------------------------------------------------
        self.world_size = self.parallel_config.world_size
        # 从配置取全局 world_size。
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        # 断言 world_size 能被 DP 组内节点数整除。
        self.local_world_size = self.parallel_config.local_world_size
        # 取本节点 worker 数。
        tp_size = self.parallel_config.tensor_parallel_size
        # TP 大小。
        pp_size = self.parallel_config.pipeline_parallel_size
        # PP 大小。
        pcp_size = self.parallel_config.prefill_context_parallel_size
        # PCP 大小。
        return tp_size, pp_size, pcp_size
        # 返回三元组。

    def _post_init_executor(self) -> None:
        # -------------------------------------------------------------------
        # 后置初始化钩子（模板方法），供子类（如 RayExecutorV2）扩展。
        # -------------------------------------------------------------------
        pass
        # 默认空实现。

    def _is_driver_worker(self, rank: int) -> bool:
        # -------------------------------------------------------------------
        # 判断某全局 rank 是否为 driver worker（TP rank 0）。
        # -------------------------------------------------------------------
        return rank % self.parallel_config.tensor_parallel_size == 0
        # 每个 TP 组内第一个 worker 是 driver。

    def start_worker_monitor(self, inline=False) -> None:
        # -------------------------------------------------------------------
        # 启动后台线程监控 worker 进程存活；有 worker 意外死亡时：
        # 记录错误 → shutdown executor → 触发失败回调通知引擎。
        # -------------------------------------------------------------------
        workers = self.workers
        # 保存 worker 列表引用。
        self_ref = weakref.ref(self)
        # 弱引用 self（避免线程持有强引用导致无法回收）。

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        # 注释：监控 worker 进程存活；某 worker 意外死亡时记录错误、
        # 关闭 executor 并调用失败回调通知引擎。
        def monitor_workers():
            # 监控线程主函数。
            sentinels = [h.proc.sentinel for h in workers]
            # 收集所有 worker 进程的哨兵（进程退出时 sentinel 变为可用）。
            died = multiprocessing.connection.wait(sentinels)
            # 阻塞等待任一 worker 进程退出（sentinel 触发）。
            _self = self_ref()
            # 通过弱引用取回 executor 对象。
            if not _self or getattr(_self, "shutting_down", False):
                # 若 executor 已被回收或已在关闭中。
                logger.debug("MultiprocWorkerMonitor: shutdown already initiated")
                # 记录调试日志并直接返回。
                return
            _self.is_failed = True
            # 置位失败标志。
            proc = next(h.proc for h in workers if h.proc.sentinel == died[0])
            # 找出发生死亡的 worker 进程。
            logger.error(
                "Worker proc %s died unexpectedly (exit code: %s), "
                "shutting down executor.",
                proc.name,
                # 进程名。
                proc.exitcode,
                # 退出码。
            )
            # 记录错误日志。
            _self.shutdown()
            # 触发 executor 关闭。
            callback = _self.failure_callback
            # 取出失败回调。
            if callback is not None:
                # 若已注册回调。
                _self.failure_callback = None
                # 先清空回调（避免重复触发）。
                callback()
                # 调用回调通知引擎执行失败流程。

        if not inline:
            # 非内联模式。
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            # 启动守护监控线程。
            return
            # 返回。

        monitor_workers()
        # 内联模式：直接同步执行监控逻辑（测试用）。

    def register_failure_callback(self, callback: FailureCallback):
        # -------------------------------------------------------------------
        # 注册失败回调；若 executor 已处于失败态则立即调用。
        # -------------------------------------------------------------------
        if self.is_failed:
            # 若已失败。
            callback()
            # 立即回调。
        else:
            self.failure_callback = callback
            # 否则保存回调，供监控线程触发。

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # 执行一轮模型推理（多进程版）。
        # type: ignore[override]：本实现额外使用 unique_reply_rank / 超时 / KV 聚合。
        # -------------------------------------------------------------------
        return self.collective_rpc(
            "execute_model",
            # 调用每个 worker 的 execute_model。
            args=(scheduler_output,),
            # 入参调度输出。
            unique_reply_rank=self.output_rank,
            # 只取输出 rank 的结果（避免收所有 worker 的重复输出）。
            non_block=non_block,
            # 透传非阻塞。
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            # RPC 超时（防止慢 worker 挂死调度循环）。
            kv_output_aggregator=self.kv_output_aggregator,
            # KV 输出聚合器（disaggregated 场景用，否则 None）。
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        # -------------------------------------------------------------------
        # 采样方法（多进程版），参数含义与 execute_model 相同。
        # -------------------------------------------------------------------
        return self.collective_rpc(
            "sample_tokens",
            # 调用 worker 的 sample_tokens。
            args=(grammar_output,),
            # 入参 grammar。
            unique_reply_rank=self.output_rank,
            # 只取输出 rank。
            non_block=non_block,
            # 透传非阻塞。
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            # 超时。
            kv_output_aggregator=self.kv_output_aggregator,
            # KV 聚合器。
        )

    def execute_dummy_batch(self) -> None:
        # -------------------------------------------------------------------
        # 执行空批（预热）。
        # -------------------------------------------------------------------
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)
        # 广播且仅等输出 rank 的结果。

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # -------------------------------------------------------------------
        # 取投机解码草稿 token；只从输出 rank 取。
        # -------------------------------------------------------------------
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        # 注释：优化——只从单个 worker（output_rank）取输出。
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )
        # 广播且只等输出 rank。

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        # method：worker 方法名或可调用对象。
        timeout: float | None = None,
        # timeout：等待响应的最大秒数。
        args: tuple = (),
        # args：位置参数。
        kwargs: dict | None = None,
        # kwargs：关键字参数。
        non_block: bool = False,
        # non_block：True 返回 Future。
        unique_reply_rank: int | None = None,
        # unique_reply_rank：只等待某单个 rank 的响应（而非全部）。
        kv_output_aggregator: KVOutputAggregator | None = None,
        # kv_output_aggregator：聚合所有 worker 输出的聚合器。
    ) -> Any:
        # -------------------------------------------------------------------
        # 核心控制平面：向广播 MQ 入队一条 RPC 消息，等待（或异步）响应。
        # 返回单个结果（提供 unique_reply_rank 或 kv_output_aggregator）或列表。
        # -------------------------------------------------------------------
        """Returns single result if unique_reply_rank and/or kv_output_aggregator
        is provided, otherwise list."""
        # 文档字符串：提供了 unique_reply_rank 或 kv_output_aggregator 时返回单值，否则返回列表。
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        # 断言：仅领导节点可调用（跟随节点没有广播 MQ）。
        if self.is_failed:
            # 若 executor 已失败。
            raise RuntimeError("Executor failed.")
            # 直接抛错。

        deadline = None if timeout is None else time.monotonic() + timeout
        # 计算超时绝对截止时间（基于单调时钟）。
        kwargs = kwargs or {}
        # None 归一化为空字典。

        if kv_output_aggregator is not None:
            # 若提供聚合器。
            output_rank = None
            # 需要等所有 worker 的输出（聚合需要全部）。
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
            # 构造聚合偏函数：把各 worker 输出聚合为一个。
        else:
            output_rank = unique_reply_rank
            # 否则只需等指定 rank。
            aggregate = lambda x: x
            # 无需聚合。

        if isinstance(method, str):
            # 若方法是字符串。
            send_method = method
            # 直接发送方法名。
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
            # 否则用 cloudpickle 序列化可调用对象（worker 端反序列化执行）。
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
        # 向广播 MQ 入队 (方法, 参数, 输出 rank)；所有 worker 都会收到。
        # 每个 worker 根据 output_rank 决定是否回写响应。

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        # 取全部响应 MQ。
        if output_rank is not None:
            # 若指定了输出 rank。
            response_mqs = (response_mqs[output_rank],)
            # 只订阅该 rank 的响应 MQ。

        def get_response():
            # 读取响应的闭包。
            responses = []
            # 结果列表。
            for mq in response_mqs:
                # 遍历订阅的 MQ。
                dequeue_timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                # 计算剩余超时时间。
                try:
                    status, result = mq.dequeue(timeout=dequeue_timeout)
                    # 从 MQ 取一条响应 (状态, 结果)。
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                    # 超时则抛 TimeoutError（带触发源）。
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    # 若 worker 返回失败状态。
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                    # 抛 RuntimeError，提示查看上方堆栈。
                responses.append(result)
                # 累积成功结果。
            return responses[0] if output_rank is not None else responses
            # 单 rank 返回单个值，否则返回列表。

        future = FutureWrapper(
            self.futures_queue, get_response=get_response, aggregate=aggregate
        )
        # 创建 FutureWrapper（自动入队 global FIFO，保证消费顺序）。

        return future if non_block else future.result()
        # 非阻塞返回 Future；阻塞调用 result() 获取最终值。

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        # -------------------------------------------------------------------
        # 确保所有 worker 进程终止：先等待优雅退出，
        # 超时后逐级升级 SIGTERM → SIGKILL。
        # -------------------------------------------------------------------
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""
        # 文档字符串：确保 worker 进程终止；假定已收到终止请求，先等待，
        # 必要时再发终止/强杀信号。

        def wait_for_termination(procs, timeout):
            # 等待所有进程退出。
            if not time:
                # 若解释器关闭期 time 被置 None。
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                # 注释：若处于关闭后期，解释器可能把 time 置为 None。
                return all(not proc.is_alive() for proc in procs)
                # 直接返回是否全部退出。
            start_time = time.time()
            # 记录开始时间。
            while time.time() - start_time < timeout:
                # 在超时窗口内轮询。
                if all(not proc.is_alive() for proc in procs):
                    # 全部退出。
                    return True
                    # 返回成功。
                time.sleep(0.1)
                # 每 100ms 轮询一次。
            return False
            # 超时仍未全部退出。

        active_procs = lambda: [proc for proc in worker_procs if proc.is_alive()]
        # 辅助函数：返回仍在运行的进程列表。
        initial_count = len(active_procs())
        # 统计初始存活进程数。

        # Give processes time to clean themselves up properly first
        # 注释：先给进程时间做优雅清理。
        logger.info(
            "[shutdown] Executor: waiting for worker exit count=%d",
            initial_count,
        )
        # 记录等待信息。
        if wait_for_termination(
            active_procs(), timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
        ):
            # 在宽限期内全部退出。
            logger.info_once("[shutdown] Executor: all workers exited gracefully")
            # 记录一次性日志。
            return
            # 完成。

        # Send SIGTERM if still running
        # 注释：仍存活则发送 SIGTERM。
        remaining = active_procs()
        # 取剩余进程。
        logger.warning(
            "[shutdown] Executor: workers still running after grace period; "
            "sending SIGTERM count=%d",
            len(remaining),
        )
        # 告警。
        for p in remaining:
            p.terminate()
            # 发送 SIGTERM。
        if not wait_for_termination(active_procs(), 4):
            # 再等 4 秒。
            # Send SIGKILL if still running
            # 注释：仍存活则发送 SIGKILL。
            remaining = active_procs()
            # 再次取剩余。
            logger.warning(
                "[shutdown] Executor: workers still running after SIGTERM; "
                "sending SIGKILL count=%d",
                len(remaining),
            )
            # 告警。
            for p in remaining:
                p.kill()
                # 强杀。

    def shutdown(self):
        # -------------------------------------------------------------------
        # 优雅关闭 executor 及其所有 worker。
        # 流程：关闭死亡写端 → 等待 worker 退出 → 关闭所有响应 MQ。
        # -------------------------------------------------------------------
        """Properly shut down the executor and its workers"""
        # 文档字符串：正确关闭 executor 与 worker。
        if not getattr(self, "shutting_down", False):
            # 若尚未进入关闭流程。
            worker_count = len(getattr(self, "workers", None) or [])
            # 统计 worker 数量。
            logger.debug(
                "[shutdown] Executor: start worker_count=%d",
                worker_count,
            )
            # 记录开始关闭。
            self.shutting_down = True
            # 置位关闭标志（防止重复进入）。

            # Make sure all the worker processes are terminated first.
            # 注释：首先确保所有 worker 进程被终止。
            if workers := getattr(self, "workers", None):
                # 若存在 worker。
                for w in workers:
                    # 遍历。
                    # Close death_writer to signal child processes to exit
                    # 注释：关闭死亡写端以通知子进程退出。
                    if w.death_writer is not None:
                        # 若写端存在。
                        w.death_writer.close()
                        # 关闭（子进程读到 EOF 触发退出）。
                        w.death_writer = None
                        # 置空。
                self._ensure_worker_termination([w.proc for w in workers])
                # 等待/强制终止所有 worker 进程。

                for w in workers:
                    # 遍历。
                    # Shutdown response queues
                    # 注释：关闭响应队列。
                    if w.worker_response_mq is not None:
                        # 若响应 MQ 存在。
                        w.worker_response_mq.shutdown()
                        # 关闭该 MQ。
                        w.worker_response_mq = None
                        # 置空。

        if rpc_broadcast_mq := getattr(self, "rpc_broadcast_mq", None):
            # 若广播 MQ 存在。
            rpc_broadcast_mq.shutdown()
            # 关闭广播 MQ。
            self.rpc_broadcast_mq = None
            # 置空。
        if response_mqs := getattr(self, "response_mqs", None):
            # 若响应 MQ 列表存在。
            for mq in response_mqs:
                # 遍历。
                mq.shutdown()
                # 逐个关闭。
            self.response_mqs = []
            # 清空列表。

        logger.debug_once("[shutdown] Executor: complete")
        # 记录关闭完成（一次性）。

    def check_health(self) -> None:
        # -------------------------------------------------------------------
        # 健康检查：向所有 worker 广播 check_health（10 秒超时）。
        # -------------------------------------------------------------------
        self.collective_rpc("check_health", timeout=10)
        # 广播健康检查。
        return
        # 正常返回表示健康。

    def _get_output_rank(self) -> int:
        # -------------------------------------------------------------------
        # 计算输出 rank：TP rank 0 + 最后一个 PP stage 的 worker。
        # -------------------------------------------------------------------
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        # 注释：只从 TP rank 0 且 PP 末级 stage 取输出。
        # 例：TP=8、PP=4 时 world_size=32：
        #   0-7 → PP0；8-15 → PP1；16-23 → PP2；24-31 → PP3；
        # 因此 world_size - tp_size = 32 - 8 = 24 是 PP 末级（rank 3）的第一个 TP worker。
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )
        # 返回 world_size - TP×PCP，即最后一个 PP stage 的第一个 TP worker 的全局 rank。

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # -------------------------------------------------------------------
        # 多进程执行器支持异步调度。
        # -------------------------------------------------------------------
        return True
        # 返回 True。


@dataclass
class UnreadyWorkerProcHandle:
    # =========================================================================
    # UnreadyWorkerProcHandle：worker 进程「就绪前」的句柄。
    # 在 wait_for_ready() 之前由 make_worker_process() 返回。
    # =========================================================================
    """WorkerProcess handle before READY."""
    # 文档字符串：worker 进程在 READY 之前的句柄。
    proc: BaseProcess
    # 底层 worker 进程对象。
    rank: int
    # worker 的全局 rank。
    ready_pipe: Connection
    # 就绪管道：子进程初始化完成后写入 READY 消息。
    death_writer: Connection | None = None
    # 死亡管道写端：父进程持有，关闭时子进程读到 EOF 从而感知父进程退出。


@dataclass
class WorkerProcHandle:
    # =========================================================================
    # WorkerProcHandle：worker 进程「就绪后」的句柄，含所有通信队列。
    # =========================================================================
    proc: BaseProcess
    # 底层 worker 进程。
    rank: int
    # 全局 rank。
    # The worker process writes to this MQ in single-node mode
    # 注释：单节点模式下 worker 把结果写入此 MQ。
    worker_response_mq: MessageQueue | None
    # 本 worker 的响应 MQ（单节点时非空）。
    # This is only non empty on driver node,
    # the peer worker process i writes to MQ
    # `peer_worker_response_mqs[i]`
    # 注释：该列表仅在 driver 节点非空，第 i 个对端 worker 进程把输出写入
    # peer_worker_response_mqs[i]。
    peer_worker_response_mqs: list[MessageQueue | None]
    # 对端 worker 的响应 MQ 列表（多节点时经隧道转发）。
    death_writer: Connection | None = None
    # 死亡管道写端。

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        # 未就绪句柄。
        worker_response_mq: MessageQueue | None,
        # 本 worker 响应 MQ。
        peer_worker_response_mqs: list[MessageQueue | None],
        # 对端响应 MQ 列表。
    ) -> "WorkerProcHandle":
        # -------------------------------------------------------------------
        # 由未就绪句柄 + 响应 MQ 组装为就绪句柄。
        # -------------------------------------------------------------------
        return cls(
            proc=unready_handle.proc,
            # 进程。
            rank=unready_handle.rank,
            # rank。
            worker_response_mq=worker_response_mq,
            # 响应 MQ。
            peer_worker_response_mqs=peer_worker_response_mqs,
            # 对端 MQ 列表。
            death_writer=unready_handle.death_writer,
            # 死亡写端。
        )


class WorkerProc:
    # =========================================================================
    # WorkerProc：在独立进程中运行一个 Worker 的封装。
    # 职责：初始化消息队列、加载模型、进入主忙循环（busy_loop）处理 RPC。
    # =========================================================================
    """Wrapper that runs one Worker in a separate process."""
    # 文档字符串：在独立进程中运行一个 Worker 的包装器。
    READY_STR = "READY"
    # 就绪消息：子进程初始化完成后通过管道发送给父进程。
    rpc_broadcast_mq: MessageQueue | None
    # 类级类型标注：广播 MQ。
    worker_response_mq: MessageQueue | None
    # 类级类型标注：响应 MQ。

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        # -------------------------------------------------------------------
        # 初始化消息队列：单个 DP 组内节点数与跨节点逻辑不同。
        # -------------------------------------------------------------------
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # 单节点（无跨节点）。
            # Initialize MessageQueue for receiving SchedulerOutput
            # 注释：初始化用于接收 SchedulerOutput 的 MessageQueue。
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )
            # 从导出句柄创建本 worker 的广播 MQ 连接（rank 作为订阅者 id）。

            # Initializes a message queue for sending the model output
            # 注释：初始化用于发送模型输出的消息队列。
            self.worker_response_mq = MessageQueue(1, 1)
            # 创建响应 MQ（n_reader=1，n_local_reader=1），只有一个读者（executor）。
            self.peer_response_handles = []
            # 单节点无对端，置空。
        else:
            # 多节点（跨 DP 组内节点）。
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            # 注释：初始化跨节点接收 SchedulerOutput 的远程 MQ。
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # 传入外部（executor 进程）的写端句柄。
                # Since there is external_writer_handle from executor proc,
                # where the ready signal from actual writer is sent out of the
                # create_mq_broadcaster method and after this setup, we make it
                # non blocking. The handshake will be triggered when
                # worker.rpc_broadcast_mq.wait_until_ready() is called
                # 注释：由于 executor 进程提供外部写端句柄，实际写方的就绪信号
                # 在 create_mq_broadcaster 方法之外发送，因此这里设为非阻塞；
                # 握手将在调用 wait_until_ready() 时触发。
                blocking=False,
            )
            # 经 DP 世界组创建跨节点 MQ 广播器（基于 NCCL）。
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            # that include handles for all ranks
            # 注释：初始化向 driver worker 发送模型输出的远程 MQ，
            # 暴露包含所有 rank 句柄的 peer_response_handles 供 driver 使用。
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )
            # 创建单读者（rank 0 driver）的 MQ，返回本 worker 响应 MQ 与对端句柄列表。

    @instrument(span_name="Worker init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        # 配置。
        local_rank: int,
        # 本地 rank。
        rank: int,
        # 全局 rank。
        distributed_init_method: str,
        # 分布式初始化地址。
        input_shm_handle: Handle,
        # 输入（广播）MQ 句柄。
        shared_worker_lock: LockType,
        # 进程共享锁。
        is_driver_worker: bool,
        # 是否 driver。
    ):
        # -------------------------------------------------------------------
        # WorkerProc 构造函数（在子进程内执行）：初始化 worker、加载模型、
        # 建立消息队列、启动异步输出线程。
        # 被 @instrument 装饰：创建 "Worker init" 追踪 span。
        # -------------------------------------------------------------------
        self.rank = rank
        # 保存全局 rank。
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        # 创建 worker 包装器（rpc_rank=local_rank，global_rank=rank）。
        # TODO: move `init_worker` to executor level as a collective rpc call
        # TODO 注释：未来把 init_worker 提升为 executor 层的集体 RPC 调用。
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        # 初始化 world_size 个空参数字典（每个全局 rank 一份）。
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            # 配置。
            "local_rank": local_rank,
            # 本地 rank。
            "rank": rank,
            # 全局 rank。
            "distributed_init_method": distributed_init_method,
            # 分布式初始化地址。
            "is_driver_worker": is_driver_worker,
            # 是否 driver。
            "shared_worker_lock": shared_worker_lock,
            # 共享锁。
        }
        # 在 local_rank 位置填入本 worker 的参数。
        wrapper.init_worker(all_kwargs)
        # 初始化 worker（内部建立并行通信组等）。
        self.worker = wrapper
        # 保存 worker 引用。

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        # 设置进程名与日志前缀（此时并行组未初始化，会走默认分支）。

        # Load model
        # 注释：加载模型。
        self.worker.init_device()
        # 初始化设备。
        # Update process title now that parallel groups are initialized
        # 注释：并行组已初始化，更新进程名与日志前缀。
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        # 用完整的并行拓扑信息命名进程（如 Worker_TP0_PP1）。
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            # 若启用弹性 EP 扩容。
            self.worker.elastic_ep_execute("load_model")
            # 走弹性 EP 加载路径。
        else:
            self.worker.load_model()
            # 常规加载模型。

        scheduler_config = vllm_config.scheduler_config
        # 取调度器配置。
        self.use_async_scheduling = scheduler_config.async_scheduling
        # 判断是否启用异步调度。
        if self.use_async_scheduling:
            # 若启用异步调度。
            self.async_output_queue: queue.Queue = queue.Queue()
            # 创建异步输出队列（模型输出先入队，由专用线程搬运）。
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                # 线程目标。
                daemon=True,
                # 守护线程。
                name="WorkerAsyncOutputCopy",
                # 线程名。
            )
            # 创建异步输出搬运线程。
            self.async_output_copy_thread.start()
            # 启动线程。

        # Set block size based on the attention backends
        # 注释：根据注意力后端设置 block size。
        current_platform.update_block_size_for_backend(vllm_config)
        # 更新 block size 配置。

        # Initialize message queues after init_device() since multi-node setups
        # (nnodes_within_dp > 1) require distributed groups to be initialized
        # 注释：在 init_device() 之后初始化消息队列，因为多节点设置需要
        # 先初始化分布式通信组。
        self._init_message_queues(input_shm_handle, vllm_config)
        # 初始化消息队列。

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        # 注释：开启环境变量缓存（此后假定不再有环境变量覆盖）。
        enable_envs_cache()
        # 锁定 env 缓存。

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        # 配置。
        local_rank: int,
        # 本地 rank。
        rank: int,
        # 全局 rank。
        distributed_init_method: str,
        # 分布式初始化地址。
        input_shm_handle,  # Receive SchedulerOutput
        # 输入（广播）MQ 句柄——注释：用于接收 SchedulerOutput。
        shared_worker_lock: LockType,
        # 进程共享锁。
        is_driver_worker: bool,
        # 是否 driver。
        inherited_fds: list[int] | None = None,
        # 待关闭的继承 fd（fork 模式）。
    ) -> UnreadyWorkerProcHandle:
        # -------------------------------------------------------------------
        # 创建 worker 子进程，返回「未就绪」句柄。
        # 在父进程（executor）中调用。
        # -------------------------------------------------------------------
        context = get_mp_context()
        # 获取 mp 上下文。
        # Ready pipe to communicate readiness from child to parent
        # 注释：就绪管道用于子进程向父进程通信就绪状态。
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # 创建单向就绪管道（父读、子写）。
        # Death pipe to let child detect parent process exit
        # 注释：死亡管道让子进程检测父进程退出。
        death_reader, death_writer = context.Pipe(duplex=False)
        # 创建单向死亡管道（父写、子读）。
        if inherited_fds is not None:
            # fork 模式。
            inherited_fds = inherited_fds.copy()
            # 复制列表（避免修改调用方）。
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
            # 把本进程新创建的管道 fd 也加入待关闭列表，供后续 worker 关闭。
        process_kwargs = {
            "vllm_config": vllm_config,
            # 配置。
            "local_rank": local_rank,
            # 本地 rank。
            "rank": rank,
            # 全局 rank。
            "distributed_init_method": distributed_init_method,
            # 分布式初始化地址。
            "input_shm_handle": input_shm_handle,
            # 广播 MQ 句柄。
            "ready_pipe": ready_writer,
            # 就绪管道写端（子进程用）。
            "death_pipe": death_reader,
            # 死亡管道读端（子进程用）。
            "shared_worker_lock": shared_worker_lock,
            # 共享锁。
            "is_driver_worker": is_driver_worker,
            # 是否 driver。
            # Have the worker close parent end of this worker's pipes too
            # 注释：让 worker 同时关闭本 worker 管道的父端。
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
            # 需要关闭的继承 fd（空列表兜底）。
        }
        # 组装子进程参数。
        # Run EngineCore busy loop in background process.
        # 注释：在后台进程中运行引擎核心忙循环。
        proc = context.Process(
            target=WorkerProc.worker_main,
            # 子进程入口。
            kwargs=process_kwargs,
            # 传入参数。
            name=f"VllmWorker-{rank}",
            # 进程名。
            daemon=True,
            # 守护进程（父进程退出则子进程被终止）。
        )
        # 创建子进程对象。

        # Apply NUMA binding if configured
        # 注释：若配置了 NUMA 绑定则应用。
        with numa_utils.configure_subprocess(
            vllm_config, local_rank, process_kind="worker"
        ):
            # 按配置为子进程准备 NUMA 环境。
            proc.start()
            # 启动子进程。

        # Close child ends of pipes here in the parent
        # 注释：在父进程关闭管道的子端。
        ready_writer.close()
        # 关闭就绪写端（父进程不需要）。
        death_reader.close()
        # 关闭死亡读端（父进程不需要）。
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        # 注释：父进程保持死亡写端打开——父进程退出时，子进程的死亡读端
        # 会收到 EOFError，从而感知父进程退出。
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)
        # 返回未就绪句柄（父进程持有就绪读端与死亡写端）。

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        # -------------------------------------------------------------------
        # 根据子进程上报的 MQ 句柄构建 WorkerProcHandle。
        # -------------------------------------------------------------------
        response_handle = handles["handle"]
        # 取响应 MQ 句柄。
        worker_response_mq: MessageQueue | None = None
        # 初始化响应 MQ。
        if len(response_handle.local_reader_ranks) > 0:
            # 若有本地读者。
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
            # 创建本进程侧响应 MQ。
        peer_response_handles = handles["peer_response_handles"]
        # 取对端句柄列表。
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            # 为每个对端句柄创建 MQ。
            if handle.remote_subscribe_addr is not None
            # 仅当句柄有远程订阅地址（跨节点）。
            else None
            # 否则置 None。
            for handle in peer_response_handles
            # 遍历所有对端句柄。
        ]
        # 构建对端响应 MQ 列表。
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            # 未就绪句柄。
            worker_response_mq,
            # 本 worker 响应 MQ。
            peer_worker_response_mqs=peer_worker_response_mqs,
            # 对端 MQ 列表。
        )
        # 组装并返回。

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        # -------------------------------------------------------------------
        # 统一等待所有 worker 就绪（读取 ready 管道）。
        # -------------------------------------------------------------------
        e = Exception(
            "WorkerProc initialization failed due to an exception in a "
            "background process. See stack trace for root cause."
        )
        # 预构造异常（当某 worker 上报失败时抛出）。
        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        # 建立 ready 管道 → 句柄的映射。
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        # 初始化结果列表。
        while pipes:
            # 只要还有未就绪的管道。
            ready = multiprocessing.connection.wait(pipes.keys())
            # 阻塞等待任一管道可读（worker 发送就绪或失败）。
            for pipe in ready:
                # 遍历已就绪的管道。
                assert isinstance(pipe, Connection)
                # 断言管道类型。
                try:
                    # Wait until the WorkerProc is ready.
                    # 注释：等待 WorkerProc 就绪。
                    unready_proc_handle = pipes.pop(pipe)
                    # 弹出该管道对应的句柄。
                    response: dict[str, Any] = pipe.recv()
                    # 读取子进程发送的就绪消息。
                    if response["status"] != "READY":
                        # 若状态不是 READY。
                        raise e
                        # 抛初始化失败异常。

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    # 计算结果列表下标（rank 对长度取模，处理多节点 rank 对齐）。
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                    # 构建并填入就绪句柄。
                except EOFError:
                    # 若管道提前关闭（子进程崩溃）。
                    e.__suppress_context__ = True
                    # 抑制上下文链。
                    raise e from None
                    # 直接抛初始化失败异常。

                finally:
                    # Close connection.
                    # 注释：关闭连接。
                    pipe.close()
                    # 关闭管道。

        return cast(list[WorkerProcHandle], ready_proc_handles)
        # 类型强转后返回所有 worker 的就绪句柄。

    def shutdown(self):
        # -------------------------------------------------------------------
        # 关闭 WorkerProc：关闭 MQ、关闭 worker、销毁分布式环境。
        # -------------------------------------------------------------------
        if self.rpc_broadcast_mq is not None:
            # 若广播 MQ 存在。
            self.rpc_broadcast_mq.shutdown()
            # 关闭。
        if self.worker_response_mq is not None:
            # 若响应 MQ 存在。
            self.worker_response_mq.shutdown()
            # 关闭。
        self.worker.shutdown()
        # 关闭底层 worker（释放模型、KV cache）。
        self.rpc_broadcast_mq = None
        # 置空。
        self.worker_response_mq = None
        # 置空。
        destroy_model_parallel()
        # 销毁模型并行组。
        destroy_distributed_environment()
        # 销毁分布式环境（TCPStore 等）。

    def monitor_death_pipe(self, death_pipe, shutdown_requested: threading.Event):
        # -------------------------------------------------------------------
        # 启动线程监控死亡管道：父进程退出（管道 EOF）时，通知 worker 关闭 MQ。
        # -------------------------------------------------------------------
        if death_pipe is None:
            # 无死亡管道。
            return
            # 直接返回。

        def death_pipe_monitor(queues_to_shutdown: list[MessageQueue]):
            # 监控线程主体。
            try:
                # This will block until parent process exits (pipe closes)
                # 注释：阻塞直到父进程退出（管道关闭）。
                death_pipe.recv()
                # 阻塞读取；父进程退出时抛 EOFError。
            except EOFError:
                logger.info_once("Parent process exited, terminating worker queues")
                # 记录父进程退出日志。
                shutdown_requested.set()
                # 置位关闭请求事件。
                for mq in queues_to_shutdown:
                    # 遍历要关闭的 MQ。
                    if mq is not None:
                        # 非空则关闭。
                        mq.shutdown()
                        # 关闭 MQ（唤醒正在阻塞的 dequeue）。
            except Exception as e:
                logger.warning("Death monitoring error: %s", e)
                # 其他异常仅告警。

        # Pass queue references directly to avoid gc issues if passing self
        # 注释：直接传队列引用，避免传 self 引发 GC 问题。
        Thread(
            target=death_pipe_monitor,
            # 线程目标。
            args=([self.rpc_broadcast_mq, self.worker_response_mq],),
            # 传入要关闭的 MQ 列表。
            daemon=True,
            # 守护线程。
            name="DeathPipeMonitor",
            # 线程名。
        ).start()
        # 启动监控线程。

    @staticmethod
    def worker_main(*args, **kwargs):
        # -------------------------------------------------------------------
        # 子进程入口：完成 worker 全部生命周期（初始化 → 忙循环 → 清理）。
        # -------------------------------------------------------------------
        """Worker initialization and execution loops.
        This runs a background process"""
        # 文档字符串：worker 初始化与执行循环，运行在后台进程中。

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        # 注释：信号处理器用于优雅终止；SystemExit 只抛一次，
        # 让本 worker 与其它 worker 进程无错退出。
        shutdown_requested = threading.Event()
        # 创建关闭请求事件。

        def signal_handler(signum, frame):
            # 信号处理函数（SIGTERM/SIGINT）。
            nonlocal shutdown_requested
            # 声明引用外层事件。
            if not shutdown_requested.is_set():
                # 若尚未请求关闭。
                shutdown_requested.set()
                # 置位关闭事件。
                logger.debug(
                    "WorkerProc handling signal %d, raising SystemExit", signum
                )
                # 记录信号。
                raise SystemExit()
                # 抛 SystemExit 优雅退出。

        # Either SIGTERM or SIGINT will terminate the worker
        # 注释：SIGTERM 或 SIGINT 都会终止 worker。
        signal.signal(signal.SIGTERM, signal_handler)
        # 注册 SIGTERM 处理。
        signal.signal(signal.SIGINT, signal_handler)
        # 注册 SIGINT 处理。

        # Publish the logical-to-physical mapping early so topology helpers
        # work before init_device (needed by set_worker_net_device below).
        # 注释：尽早发布逻辑→物理 GPU 映射，使拓扑工具在 init_device 之前可用
        #（set_worker_net_device 需要它）。
        assigned_physical_gpu_ids = kwargs[
            "vllm_config"
        ].parallel_config.assigned_physical_gpu_ids
        # 取出配置的物理 GPU 映射。
        if assigned_physical_gpu_ids is not None:
            # 若存在映射。
            from vllm.platforms.interface import set_assigned_physical_gpu_ids
            # 延迟导入设置工具。

            set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
            # 设置逻辑→物理映射。

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        # 注释：若设置 VLLM_GPU_NIC_PCIE_MAPPING，则为 worker 设置网卡环境变量。
        set_worker_net_device(kwargs.get("local_rank", 0), kwargs["vllm_config"])
        # 设置 GPU→NIC 映射。

        worker = None
        # 初始化 worker 变量（供 finally 清理）。
        ready_writer = kwargs.pop("ready_pipe")
        # 弹出并取出就绪管道写端。
        death_pipe = kwargs.pop("death_pipe", None)
        # 弹出死亡管道读端。

        # Close inherited pipes from parent (incl. other worker pipes)
        # Explicitly passing in existing pipes and closing them makes the pipe
        # behave when using fork. Otherwise, a hidden reference to the pipes
        # exist in the child process and prevents EOF closure.
        # 注释：关闭从父进程继承的管道（包括其他 worker 的管道）。显式传入并
        # 关闭现有管道使 fork 模式下管道行为正确；否则子进程持有隐藏引用，
        # 会阻止 EOF 触发。
        for fd in kwargs.pop("inherited_fds", []):
            # 遍历要关闭的 fd。
            try:
                os.close(fd)
                # 关闭 fd。
            except Exception as e:
                logger.warning("Error closing inherited connection: %s: %s", type(e), e)
                # 关闭失败仅告警。

        try:
            # Initialize tracer
            # 注释：初始化追踪器。
            rank = kwargs.get("rank", 0)
            # 取 rank。
            maybe_init_worker_tracer(
                instrumenting_module_name="vllm.worker",
                # 追踪模块名。
                process_kind="worker",
                # 进程类型。
                process_name=f"Worker_{rank}",
                # 进程名。
            )
            # 在子进程初始化 OpenTelemetry tracer。

            worker = WorkerProc(*args, **kwargs)
            # 构造 WorkerProc（完整初始化：模型加载、MQ 建立等）。
            assert worker.worker_response_mq is not None
            # 断言响应 MQ 已创建。
            if kwargs["vllm_config"].parallel_config.numa_bind:
                # 若启用 NUMA 绑定。
                numa_utils.log_current_affinity_state(f"Worker_{worker.rank}")
                # 记录当前 CPU 亲和状态。

            worker.monitor_death_pipe(death_pipe, shutdown_requested)
            # 启动死亡管道监控线程。

            # Send READY once we know everything is loaded
            # 注释：确认一切加载完成后发送 READY。
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    # 状态。
                    "handle": worker.worker_response_mq.export_handle(),
                    # 导出响应 MQ 句柄。
                    "peer_response_handles": worker.peer_response_handles,
                    # 对端句柄列表。
                }
            )
            # 向父进程发送就绪消息。

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            # 注释：确保 MQ 就绪。顺序不可调换，否则死锁；
            # 必须与 Executor 保持一致。
            if worker.rpc_broadcast_mq is not None:
                # 若广播 MQ 存在。
                worker.rpc_broadcast_mq.wait_until_ready()
                # 等待广播 MQ 就绪。
            worker.worker_response_mq.wait_until_ready()
            # 等待响应 MQ 就绪。
            ready_writer.close()
            # 关闭就绪管道写端。
            ready_writer = None
            # 置空。

            worker.worker_busy_loop()
            # 进入主忙循环（阻塞处理 RPC，直到退出）。

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.
            # 注释：若忙循环中抛异常，通过 MQ RPC 发送 FAILURE 消息通知
            # Executor，触发系统关闭；TODO：处理 MQ 自身损坏的情况。

            if ready_writer is not None:
                # 若还没发送过就绪（初始化阶段失败）。
                logger.exception("WorkerProc failed to start.")
                # 记录启动失败（含堆栈）。
            elif shutdown_requested.is_set():
                # 若是关闭请求引发的退出。
                logger.debug_once(
                    "[shutdown] WorkerProc: exiting after shutdown request"
                )
                # 记录一次性调试日志。
            else:
                logger.exception("WorkerProc failed.")
                # 否则记录运行失败。

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            # 注释：父进程会在任一 worker 死亡时向所有 worker 发 SIGTERM。
            # 置位此值避免再次抛出 SystemExit，从而避免 __del__ 中 zmq 异常。
            shutdown_requested.set()
            # 置位关闭事件。

        except SystemExit as e:
            # SystemExit is raised on SIGTERM or SIGKILL, which usually indicates that
            # the graceful shutdown process did not succeed
            # 注释：SystemExit 由 SIGTERM/SIGKILL 触发，通常意味着优雅关闭未成功。
            if shutdown_requested.is_set():
                # 若是已请求关闭。
                logger.debug_once(
                    "[shutdown] WorkerProc: terminated by shutdown signal"
                )
                # 记录。
            else:
                logger.warning("WorkerProc was terminated")
                # 否则告警。
            # SystemExit must never be ignored
            # 注释：SystemExit 不能被吞掉。
            raise e
            # 重新抛出。

        finally:
            if ready_writer is not None:
                # 若就绪写端仍存在。
                ready_writer.close()
                # 关闭。
            if death_pipe is not None:
                # 若死亡管道存在。
                death_pipe.close()
                # 关闭。
            # Clean up once worker exits busy loop
            # 注释：worker 退出忙循环后清理。
            if worker is not None:
                # 若 worker 已创建。
                worker.shutdown()
                # 调用 WorkerProc.shutdown() 清理。

    class ResponseStatus(Enum):
        # 枚举：RPC 响应状态。
        SUCCESS = auto()
        # 成功。
        FAILURE = auto()
        # 失败。

    def enqueue_output(self, output: Any):
        # -------------------------------------------------------------------
        # 把 worker 输出准备后入队到响应 MQ；异常转为 FAILURE 状态。
        # -------------------------------------------------------------------
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response."""
        # 文档字符串：准备 worker 输出并入队；异常输出转为 FAILURE。
        if isinstance(output, AsyncModelRunnerOutput):
            # 若为异步输出。
            try:
                output = output.get_output()
                # 尝试取出最终输出。
            except Exception as e:
                logger.exception("Error getting async model runner output")
                # 记录错误。
                output = e
                # 把异常作为输出传递。

        if isinstance(output, Exception):
            # 若输出是异常。
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
            # 打包为 (FAILURE, 异常字符串)。
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
            # 打包为 (SUCCESS, 输出)。
        if (response_mq := self.worker_response_mq) is not None:
            # 若响应 MQ 存在。
            response_mq.enqueue(result)
            # 入队结果。

    def handle_output(self, output: Any):
        # -------------------------------------------------------------------
        # 处理输出：异步调度走队列由搬运线程发送；否则直接发送。
        # -------------------------------------------------------------------
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq."""
        # 文档字符串：处理 worker 输出；异步调度时交给搬运线程，否则直接入队。
        if self.use_async_scheduling:
            # 若启用异步调度。
            self.async_output_queue.put(output)
            # 放入异步输出队列（由搬运线程取出并发送）。
        else:
            self.enqueue_output(output)
            # 直接入队发送。

    def async_output_busy_loop(self):
        # -------------------------------------------------------------------
        # 异步输出搬运线程主体：从队列取输出并转发到响应 MQ。
        # -------------------------------------------------------------------
        """Entrypoint for the thread which handles outputs asynchronously."""
        # 文档字符串：异步处理输出的线程入口。

        # set device to the worker device for the thread.
        # a thread will not inherit the context of the main thread.
        # when calling any cuda runtime functions, it will implicitly
        # create a new cuda context on device 0, consuming extra memory.
        # here we set the device to the worker device for the thread,
        # enforcing the context to be the same as the main thread.
        # 注释：为线程设置 worker 设备。线程不会继承主线程上下文；
        # 调用 CUDA 运行时函数时会在设备 0 隐式创建新 context，消耗额外显存。
        # 此处把设备设为 worker 设备，使线程与主线程共享同一 context。
        from vllm.platforms import current_platform
        # 延迟导入（避免循环）。

        if hasattr(self.worker, "device"):
            # 若 worker 有 device 属性。
            current_platform.set_device(self.worker.device)
            # 设置线程当前设备。

        while True:
            # 无限循环。
            output = self.async_output_queue.get()
            # 阻塞取输出。
            self.enqueue_output(output)
            # 转发到响应 MQ。

    def worker_busy_loop(self):
        # -------------------------------------------------------------------
        # 主忙循环：从广播 MQ 取 RPC 请求，执行并把输出回写到响应 MQ。
        # 这是多进程 worker 的事件主循环。
        # -------------------------------------------------------------------
        """Main busy loop for Multiprocessing Workers"""
        # 文档字符串：多进程 worker 的主忙循环。
        assert self.rpc_broadcast_mq is not None
        # 断言广播 MQ 已建立。
        while True:
            # 无限循环。
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                indefinite=True
            )
            # 阻塞取一条 RPC 请求（indefinite=True 表示无限等待）。
            # 消息结构：(方法, 位置参数, 关键字参数, 输出 rank)。
            try:
                if isinstance(method, str):
                    # 若方法为字符串。
                    func = getattr(self.worker, method)
                    # 按名取 worker 方法。
                elif isinstance(method, bytes):
                    # 若方法为序列化对象。
                    func = partial(cloudpickle.loads(method), self.worker)
                    # 反序列化并把 worker 绑定为 self。

                output = func(*args, **kwargs)
                # 执行方法。

                if output_rank is None or self.rank == output_rank:
                    # 若输出 rank 为 None（所有 worker 都要回写）或本 rank 匹配。
                    self.handle_output(output)
                    # 处理输出（回写响应 MQ）。
            except Exception as e:
                # Notes have been introduced in python 3.11
                # 注释：python 3.11 引入 add_note。
                if hasattr(e, "add_note"):
                    # 若支持 add_note。
                    e.add_note(traceback.format_exc())
                    # 把完整堆栈附加到异常（供主进程打印诊断）。
                logger.exception("WorkerProc hit an exception.")
                # 记录异常日志。
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                # 注释：异常可能不可序列化，转换为字符串仅供日志使用。
                if output_rank is None or self.rank == output_rank:
                    # 若本 worker 需要回写。
                    self.handle_output(e)
                    # 把异常作为结果回写（会转为 FAILURE 状态）。

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        # -------------------------------------------------------------------
        # 根据并行拓扑设置进程名与日志前缀（如 Worker_DP0_PP1_TP2）。
        # -------------------------------------------------------------------
        # Check if parallel groups are initialized first
        # 注释：先检查并行组是否已初始化。
        if not model_parallel_is_initialized():
            # 未初始化。
            # Parallel groups not yet initialized, use default process name
            # 注释：并行组尚未初始化，使用默认进程名。
            set_process_title(name="Worker")
            # 设置默认进程名。
            decorate_logs("Worker")
            # 设置默认日志前缀。
            return
            # 返回。

        dp_size = get_dp_group().world_size
        # DP 组大小。
        dp_rank = get_dp_group().rank_in_group
        # DP 组内 rank。
        pp_size = get_pp_group().world_size
        # PP 组大小。
        pp_rank = get_pp_group().rank_in_group
        # PP 组内 rank。
        pcp_size = get_pcp_group().world_size
        # PCP 组大小。
        pcp_rank = get_pcp_group().rank_in_group
        # PCP 组内 rank。
        tp_size = get_tp_group().world_size
        # TP 组大小。
        tp_rank = get_tp_group().rank_in_group
        # TP 组内 rank。
        dcp_size = get_dcp_group().world_size
        # DCP 组大小。
        dcp_rank = get_dcp_group().rank_in_group
        # DCP 组内 rank。
        process_name = "Worker"
        # 基础进程名。
        if dp_size > 1:
            # 若 DP>1。
            process_name += f"_DP{dp_rank}"
            # 追加 DP rank。
        if pp_size > 1:
            # 若 PP>1。
            process_name += f"_PP{pp_rank}"
            # 追加 PP rank。
        if pcp_size > 1:
            # 若 PCP>1。
            process_name += f"_PCP{pcp_rank}"
            # 追加 PCP rank。
        if tp_size > 1:
            # 若 TP>1。
            process_name += f"_TP{tp_rank}"
            # 追加 TP rank。
        if dcp_size > 1:
            # 若 DCP>1。
            process_name += f"_DCP{dcp_rank}"
            # 追加 DCP rank。
        if enable_ep:
            # 若启用专家并行。
            ep_rank = get_ep_group().rank_in_group
            # EP 组内 rank。
            process_name += f"_EP{ep_rank}"
            # 追加 EP rank。
        set_process_title(name=process_name)
        # 设置完整进程名。
        decorate_logs(process_name)
        # 设置日志前缀。


def set_multiprocessing_worker_envs():
    # =========================================================================
    # 设置多进程环境变量：父进程创建 worker 前调用。
    # 主要目的：收敛 CPU 线程数、必要时强制 spawn。
    # =========================================================================
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""
    # 文档字符串：设置多进程环境下的环境变量，由父进程在创建 worker 前调用。

    _maybe_force_spawn()
    # 根据条件决定是否强制 spawn 启动方式。

    if not current_platform.is_cpu():
        # 非 CPU 平台（GPU）。
        # Configure thread parallelism if OMP_NUM_THREADS isn't set
        #
        # Helps to avoid CPU contention. The default of spawning a thread per
        # core combined with multiprocessing for each GPU can have a negative
        # impact on performance. The contention is amplified when running in a
        # container where CPU limits can cause throttling.
        # 注释：若未设置 OMP_NUM_THREADS 则配置线程并行度。
        # 有助于避免 CPU 竞争：每个核心一个线程的默认设置结合每 GPU 多进程
        # 会对性能产生负面影响；容器内 CPU 限制导致的节流会放大竞争。
        default_omp_num_threads = 1
        # 默认 OpenMP 线程数为 1。
        if (
            "OMP_NUM_THREADS" not in os.environ
            # 未显式设置 OMP_NUM_THREADS。
            and (current_parallelism := torch.get_num_threads())
            # 当前 torch 线程数。
            > default_omp_num_threads
            # 且大于默认值。
        ):
            logger.warning_once(
                "Reducing Torch parallelism from %d threads to %d to avoid "
                "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
                "external environment to tune this value as needed.",
                current_parallelism,
                # 原线程数。
                default_omp_num_threads,
                # 目标线程数。
            )
            # 一次性告警。
            os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
            # 设置环境变量。
            torch.set_num_threads(default_omp_num_threads)
            # 同步设置 torch 线程数。