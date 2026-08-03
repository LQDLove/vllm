# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import copy  # copy：浅拷贝（引擎负载统计复制）
import multiprocessing  # multiprocessing：多进程模块
import multiprocessing.connection  # connection：进程间连接（等待 sentinel）
import time  # time：时间模块（统计发布间隔计算）
import weakref  # weakref：弱引用（进程终结器）
from typing import Any  # Any：通用类型标注

import msgspec.msgpack  # msgspec：高性能 msgpack 序列化库
import zmq  # zmq：ZeroMQ 消息队列库（跨进程通信）

from vllm.config import ParallelConfig  # 并行配置
from vllm.logger import init_logger  # 初始化 vLLM 日志记录器
from vllm.utils.network_utils import make_zmq_socket  # 创建 ZMQ socket
from vllm.utils.system_utils import get_mp_context, set_process_title
# 获取多进程上下文；设置进程标题
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
# 引擎核心输出容器；引擎核心请求类型枚举（START_DP_WAVE 等）
from vllm.v1.serial_utils import MsgpackDecoder  # msgpack 解码器
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown
# 获取引擎客户端 ZMQ 地址；进程关闭工具函数

logger = init_logger(__name__)  # 模块级日志记录器


class DPCoordinator:
    """Coordinator process used for data-parallel deployments (DP>1).

    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * Collects stats from each DP engine (currently just waiting and running
      queue lengths), and publishes these to all front-ends for use in
      load-balancing decisions.

    * Keeps track of the current DP "request wave" number and running state
      of the engines. This is received from the DP rank 0 engine and published
      to the front-end processes along with the current load stats.

      The engines alternate between a global running/paused state. The global
      "request wave" number is a count of the number of times that the workers
      collectively move from a running state to a paused state. This transition
      is synchronized via the all-reduce operation performed in the
      DPEngineCoreProc._has_global_unfinished_reqs method.

    * Broadcasts the START_DP_WAVE message to engines to move them from paused
      to running state when one engine receives a new request. This can happen
      in two cases:
      1) A front-end sending a new request while the engines are paused will
         concurrently notify the coordinator.
      2) An engine receiving a request for a stale request wave while in paused
         state will notify the coordinator.

    Engines will move into running state when receiving a new request or
    START_DP_WAVE message.

    Note that when deployed in External LB mode, no stats will be published by
    the engines and thus updates will only be sent to front-ends when the
    request wave / running state changes.
    """
    # 用于数据并行部署（DP>1）的协调器进程。
    # 在多个 DP 引擎 rank 进程和一个或多个前端 API 服务器进程间进行中介。
    #
    # * 收集每个 DP 引擎的状态（目前只是等待和运行队列长度），
    #   并发布给所有前端用于负载均衡决策。
    #
    # * 跟踪当前 DP "请求 wave"编号和引擎运行状态。
    #   此信息从 DP rank 0 引擎接收，并随当前负载统计一起发布给前端进程。
    #   引擎在全局运行/暂停状态间交替。全局 "请求 wave"编号是对
    #   worker 从运行到暂停状态集体转换次数的计数。此转换通过
    #   DPEngineCoreProc._has_global_unfinished_reqs 方法中的 all-reduce 同步。
    #
    # * 将一个引擎收到新请求时，向引擎广播 START_DP_WAVE 消息，
    #   使其从暂停状态转为运行状态。有两种情况：
    #   1) 前端在引擎暂停时发送新请求，同时通知协调器。
    #   2) 引擎在暂停状态收到旧请求 wave 的请求时通知协调器。
    #
    # 引擎在收到新请求或 START_DP_WAVE 消息时进入运行状态。
    #
    # 注意：在外部负载均衡模式下，引擎不发布统计，
    # 因此只有请求 wave / 运行状态变化时才会向前端发送更新。

    def _wait_for_zmq_addrs(self, zmq_addr_pipe) -> tuple[str, str, str]:
        # 等待协调器进程报告其 ZMQ 地址
        try:
            timeout = 120  # 超时时间：120 秒
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=timeout
            )
            # 等待地址管道有数据或进程退出
            if not ready:
                # 如果超时无响应
                raise RuntimeError(
                    # 抛出启动失败错误
                    "DP Coordinator process failed to report ZMQ addresses "
                    f"within timeout={timeout} seconds during startup."
                )
            try:
                return zmq_addr_pipe.recv()  # 接收地址元组
            except EOFError:
                # 如果管道关闭（进程崩溃）
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()  # 关闭地址管道

    def __init__(
        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True
    ):
        # 构造函数：启动协调器进程
        dp_size = parallel_config.data_parallel_size  # DP 大小
        assert dp_size > 1, "Coordinator only used for data parallel"
        # 断言 DP>1（协调器仅用于数据并行）

        host = parallel_config.data_parallel_master_ip  # 主节点 IP

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        # 假设不在外部或混合 DP 负载均衡模式时，协调器与前端进程同机。
        local_only = not parallel_config.local_engines_only
        # 是否仅本地地址（非本地引擎模式时用 IPC）
        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        # 引擎是否全部本地（DP 大小等于本地 DP 大小）
        # NOTE(yongji): handling scaling from intra-node to inter-node
        # 注意：处理从节点内扩展到节点间扩展
        if parallel_config.enable_elastic_ep:
            # 如果启用弹性 EP
            local_only_eng = False  # 引擎可能跨节点，不能用 IPC

        front_publish_address = get_engine_client_zmq_addr(local_only, host=host)
        # 前端发布地址（向 API 服务器发布负载统计）
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host=host)
        # 后端发布地址（向引擎发布 wave 同步消息）
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host=host)
        # 后端输出地址（接收引擎的统计和 wave 通知）

        context = get_mp_context()  # 获取多进程上下文
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        # 创建父子进程通信管道（子进程报告 ZMQ 地址）
        self.proc: multiprocessing.Process = context.Process(
            # 创建协调器进程
            target=DPCoordinatorProc.run_coordinator,  # 进程入口
            name="VLLM_DP_Coordinator",  # 进程名
            kwargs={
                # 传递给协调器进程的参数
                "engine_count": parallel_config.data_parallel_size,  # 引擎数
                "front_publish_address": front_publish_address,  # 前端发布地址
                "back_output_address": back_output_address,  # 后端输出地址
                "back_publish_address": back_publish_address,  # 后端发布地址
                "zmq_addr_pipe": child_zmq_addr_pipe,  # 地址报告管道
                "enable_wave_coordination": enable_wave_coordination,  # wave 协调
            },
            daemon=True,  # 守护进程
        )
        self.proc.start()  # 启动进程
        child_zmq_addr_pipe.close()  # 父进程关闭子端管道
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)
        # 等待并获取实际绑定的地址（可能随机分配）

        self.stats_publish_address = front_publish_address  # 统计发布地址
        self.coord_in_address = back_publish_address  # 引擎输入地址
        self.coord_out_address = back_output_address  # 引擎输出地址
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])
        # 弱引用终结器：对象被 GC 时关闭进程

    def get_stats_publish_address(self) -> str:
        # 获取统计发布地址（前端订阅用）
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        """Returns tuple of ZMQ input address, output address."""
        # 返回引擎输入、输出 ZMQ 地址元组
        return self.coord_in_address, self.coord_out_address

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown coordinator process with configurable timeout."""
        # 关闭协调器进程，支持可配置超时
        if self._finalizer.detach() is not None:
            # 如果终结器尚未执行
            shutdown([self.proc], timeout=timeout)  # 关闭进程


class EngineState:
    # 单个引擎的状态跟踪
    def __init__(self):
        # [waiting, running, kv_cache_usage]
        # [等待数, 运行数, KV 缓存使用率]
        self.request_counts: list[int | float] = [0, 0, 0.0]
        # 请求计数列表，初始全为 0


class DPCoordinatorProc:
    # 数据并行协调器进程主体

    def __init__(
        self,
        engine_count: int,  # 引擎数量
        min_stats_update_interval_ms: int = 100,  # 最小统计发布间隔（毫秒）
        enable_wave_coordination: bool = True,  # 是否启用 wave 协调
    ):
        set_process_title("DPCoordinator")  # 设置进程标题
        self.ctx = zmq.Context()  # 创建 ZMQ 上下文

        self.engines = [EngineState() for _ in range(engine_count)]
        # 为每个引擎创建状态跟踪对象

        self.stats_update_interval_ms = min_stats_update_interval_ms
        # 保存统计更新间隔
        self.enable_wave_coordination = enable_wave_coordination
        # 保存 wave 协调标志

    @staticmethod
    def run_coordinator(
        # 协调器进程入口（静态方法）
        engine_count: int,  # 引擎数量
        front_publish_address: str,  # 前端发布地址
        back_output_address: str,  # 后端输出地址
        back_publish_address: str,  # 后端发布地址
        zmq_addr_pipe=None,  # 地址报告管道（可选）
        min_stats_update_interval_ms: int = 100,  # 统计发布间隔
        enable_wave_coordination: bool = True,  # wave 协调标志
    ):
        coordinator = DPCoordinatorProc(  # 创建协调器实例
            engine_count=engine_count,  # 引擎数
            min_stats_update_interval_ms=min_stats_update_interval_ms,  # 间隔
            enable_wave_coordination=enable_wave_coordination,  # wave 协调
        )
        try:
            coordinator.process_input_socket(
                # 进入主循环
                front_publish_address,  # 前端发布
                back_output_address,  # 后端输出
                back_publish_address,  # 后端发布
                zmq_addr_pipe,  # 地址管道
            )
        except KeyboardInterrupt:
            # 捕获 Ctrl+C
            logger.info("DP Coordinator process exiting")  # 记录退出日志
        finally:
            if zmq_addr_pipe is not None:
                # 如果管道存在
                zmq_addr_pipe.close()  # 关闭管道

    def process_input_socket(
        self,
        front_publish_address: str,  # 前端发布地址（XPUB）
        back_output_address: str,  # 后端输出地址（PULL）
        back_publish_address: str,  # 后端发布地址（XPUB）
        zmq_addr_pipe=None,  # 地址管道
    ):
        # 协调器主循环
        decoder = MsgpackDecoder(EngineCoreOutputs)  # 创建解码器

        # For tracking request wave progression.
        # 用于跟踪请求 wave 进展
        current_wave = 0  # 当前 wave 编号
        engines_running = False  # 引擎是否运行

        # For tracking request counts for internal load-balancing.
        # 用于跟踪内部负载均衡的请求计数
        stats_changed = False  # 统计是否变化
        last_stats_step = -1  # 上次统计的 step 编号
        last_stats_wave = -1  # 上次统计的 wave 编号
        last_step_counts: list[list[int | float]] | None = None
        # 上次 step 的完整计数快照

        with (
            make_zmq_socket(
                # 创建前端发布 socket（向 API 服务器发布统计）
                path=front_publish_address,  # IPC
                ctx=self.ctx,  # 上下文
                socket_type=zmq.XPUB,  # XPUB：发布/订阅（可跟踪订阅）
                bind=True,  # 绑定
            ) as publish_front,
            make_zmq_socket(
                # 创建后端输出 socket（接收引擎的统计）
                path=back_output_address,  # IPC or TCP
                ctx=self.ctx,  # 上下文
                socket_type=zmq.PULL,  # PULL：拉取
                bind=True,  # 绑定
            ) as output_back,
            make_zmq_socket(
                # 创建后端发布 socket（向引擎发布 wave 消息）
                path=back_publish_address,  # IPC or TCP
                ctx=self.ctx,  # 上下文
                socket_type=zmq.XPUB,  # XPUB
                bind=True,  # 绑定
            ) as publish_back,
        ):
            if zmq_addr_pipe is not None:
                # 如果提供了地址管道
                try:
                    zmq_addr_pipe.send(
                        # 向父进程报告实际绑定的地址
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()  # 关闭管道
            # Wait until all engines subscribe.
            # 等待所有引擎订阅
            for _ in self.engines:
                # 遍历每个引擎
                if publish_back.recv() != b"\x01":
                    # 等待订阅消息（0x01 = 订阅）
                    logger.error(
                        # 记录错误
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return  # 退出
            # Send ready message to engines.
            # 向引擎发送就绪消息
            publish_back.send(b"READY")  # 广播 READY

            logger.info("All engine subscriptions received by DP coordinator")
            # 记录所有引擎订阅完成

            poller = zmq.Poller()  # 创建轮询器
            poller.register(publish_front, zmq.POLLIN)  # 注册前端发布
            poller.register(publish_back, zmq.POLLIN)  # 注册后端发布
            poller.register(output_back, zmq.POLLIN)  # 注册后端输出
            last_publish_time = 0  # 上次发布时间
            while True:
                # 主循环
                elapsed = int(time.time() * 1000) - last_publish_time
                # 距上次发布时间（毫秒）
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                # 若统计有变化则按 stats_update_interval_ms 间隔发送，
                # 否则每 5 秒发送一次。
                wait_for = self.stats_update_interval_ms if stats_changed else 5000
                # 等待时间（毫秒）

                # Wait at least 50ms to ensure we've received all stats for
                # the current step. Only applicable to lockstep (MoE) DP;
                # non-lockstep engines have no synchronized step boundaries.
                # 至少等待 50ms，确保收到当前 step 的所有统计。
                # 仅适用于锁步（MoE）DP；非锁步引擎没有同步 step 边界。
                if self.enable_wave_coordination and last_step_counts is None:
                    # 如果启用 wave 协调且无已发布的计数
                    min_timeout = 50  # 最小超时 50ms
                else:
                    min_timeout = 0  # 否则 0

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                # 轮询事件（等待剩余时间）
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    # 轮询超时：向前端发布当前统计
                    if last_step_counts is not None:
                        # 如果有缓存的 step 计数
                        engine_req_counts_list = last_step_counts  # 使用缓存
                        last_step_counts = None  # 清除缓存
                    else:
                        # 否则获取当前计数
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False  # 重置统计变化标志

                    to_publish = (engine_req_counts_list, current_wave, engines_running)
                    # 构建发布内容：(各引擎计数, wave, 运行状态)
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    # 编码并发布到前端
                    last_publish_time = int(time.time() * 1000)  # 更新时间
                    continue  # 继续循环

                events = dict(events)  # 转为字典
                wave_state_changed = False  # wave 状态变化标志

                if publish_back in events:
                    # 后端发布 socket 有消息（引擎的订阅/心跳）
                    buffer = publish_back.recv()  # 接收消息
                    if buffer == b"\x01":
                        # NOTE(yongji): newly started engine subscribed
                        # We need to send READY message here instead of receiving
                        # SCALE_ELASTIC_EP notification from engine core client
                        # as SCALE_ELASTIC_EP is only sent when
                        # new engines finished initialization.
                        # Subscription message, on the other hand, is sent
                        # by each engine during initialization
                        # 注意：新启动的引擎订阅了。
                        # 我们需要在此发送 READY 消息，而不是等待从引擎核心
                        # 客户端接收 SCALE_ELASTIC_EP 通知，因为该通知仅在
                        # 新引擎完成初始化时才发送。而订阅消息是每个引擎
                        # 在初始化期间发送的。
                        publish_back.send(b"READY")  # 回复 READY
                    elif buffer != b"\x00":
                        # 0x00 = 退订；其他为未知消息
                        logger.error(
                            # 记录错误
                            "DP Coordinator received unexpected message from engines"
                        )

                if publish_front in events:
                    # 前端发布 socket 有消息（API 服务器的请求或通知）
                    buffer = publish_front.recv()  # 接收消息
                    if buffer in (b"\x01", b"\x00"):
                        # Ignore subscription messages.
                        # 忽略订阅/退订消息（0x01/0x00）
                        continue

                    decoded = msgspec.msgpack.decode(buffer)  # 解码消息
                    if (
                        isinstance(decoded, (list, tuple))  # 是列表/元组
                        and len(decoded) == 2  # 长度 2
                        and decoded[0] == "SCALE_ELASTIC_EP"  # 弹性扩展通知
                    ):
                        # Handle scale up notification
                        # 处理弹性扩展通知
                        new_engine_count = decoded[1]  # 新引擎数量
                        current_count = len(self.engines)  # 当前引擎数量
                        if new_engine_count > current_count:
                            # 扩容
                            for _ in range(new_engine_count - current_count):
                                # 为每个新引擎创建状态
                                self.engines.append(EngineState())
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            # 注意：处理新启动引擎 current_wave=0 的情况。
                            # 如果现有引擎刚完成一个 wave，且 CoordinatorProc
                            # 中 engines_running 尚未更新，则路由到新启动引擎
                            # 的请求可能无法唤醒现有引擎，
                            # 只要 0 < 请求.wave < 现有引擎的 current_wave。
                            # 注意：0 是新引擎的 wave 编号。
                            logger.info(
                                # 记录扩容日志
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,  # 当前数量
                                new_engine_count,  # 新数量
                            )
                        else:
                            # 缩容
                            self.engines = self.engines[:new_engine_count]
                            # 截断引擎列表
                            logger.info(
                                # 记录缩容日志
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,  # 当前数量
                                new_engine_count,  # 新数量
                            )
                        continue  # Skip normal engine notification processing
                        # 跳过正常的引擎通知处理

                    # Wave coordination: handle new-request messages from front-end.
                    # Only process these when wave coordination is enabled
                    # Wave 协调：处理来自前端的新请求消息。
                    # 仅在启用 wave 协调时处理
                    if self.enable_wave_coordination:
                        # We received a message on the front-end XPUB socket,
                        # from an API server sending a new request while the
                        # engines are paused, so that we can wake the other
                        # engines.
                        # 我们在前端 XPUB socket 上收到消息，来自暂停状态下
                        # 发送新请求的 API 服务器，以便唤醒其他引擎。
                        engine_to_exclude, wave = decoded  # 解包 (排除引擎, wave)
                        if not engines_running:
                            # 如果引擎当前暂停
                            if wave < current_wave:
                                # If the wave number is stale, ensure the message
                                # is handled by all the engines.
                                # 如果 wave 编号已过期，确保消息由所有引擎处理。
                                engine_to_exclude = None  # 不排除任何引擎

                            engines_running = True  # 标记引擎运行
                            wave_state_changed = True  # wave 状态已变化
                            self._send_start_wave(  # 广播启动 wave
                                publish_back,  # 后端发布 socket
                                current_wave,  # 当前 wave
                                engine_to_exclude,  # 排除的引擎
                            )

                if output_back in events:
                    # We received a message from one of the engines.
                    # 收到来自某个引擎的消息
                    buffer = output_back.recv()  # 接收消息
                    outputs: EngineCoreOutputs = decoder.decode(buffer)
                    # 解码为 EngineCoreOutputs

                    assert not outputs.outputs  # 断言无推理输出
                    assert outputs.utility_output is None  # 断言无工具输出

                    eng_index = outputs.engine_index  # 引擎索引
                    scheduler_stats = outputs.scheduler_stats  # 调度统计
                    if scheduler_stats:
                        # Elastic EP stats may arrive while the engine list changes.
                        if eng_index >= len(self.engines):
                            continue
                        # 1. Updated request load stats - update our local
                        # state with these.
                        # 1. 更新请求负载统计 - 用这些数据更新本地状态
                        stats = self.engines[eng_index].request_counts
                        # 获取该引擎的请求计数
                        if self.enable_wave_coordination:
                            # Steps are synchronized across lockstep (MoE) DP
                            # ranks; snapshot counts at step boundaries.
                            # 锁步（MoE）DP rank 间 step 同步；
                            # 在 step 边界拍摄计数快照。
                            stats_step = scheduler_stats.step_counter  # step 编号
                            stats_wave = scheduler_stats.current_wave  # wave 编号
                            if (
                                stats_wave > last_stats_wave  # 更新的 wave
                                or stats_wave == last_stats_wave  # 同一 wave
                                and stats_step > last_stats_step  # 更新的 step
                            ):
                                if stats_changed:
                                    # 如果已有变化
                                    last_step_counts = self._get_engine_counts(
                                        do_copy=True
                                    )
                                    # 保存计数快照副本
                                last_stats_step = stats_step  # 更新 step
                                last_stats_wave = stats_wave  # 更新 wave
                            elif stats_wave != last_stats_wave or (
                                stats_step != last_stats_step
                            ):
                                # 乱序统计
                                logger.warning(
                                    # 记录警告
                                    "Received stats for out-of-order "
                                    "step (%d, %d) from engine %d (expected "
                                    "> (%d, %d))",
                                    stats_wave,  # 收到的 wave
                                    stats_step,  # 收到的 step
                                    eng_index,  # 引擎索引
                                    last_stats_wave,  # 期望的 wave
                                    last_stats_step,  # 期望的 step
                                )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        # 更新等待请求数
                        stats[1] = scheduler_stats.num_running_reqs
                        # 更新运行请求数
                        stats[2] = scheduler_stats.kv_cache_usage
                        # 更新 KV 缓存使用率
                        stats_changed = True  # 标记统计已变化

                    # Wave coordination: handle wave completion and start notifications
                    # Only process these when wave coordination is enabled
                    # Wave 协调：处理 wave 完成和启动通知。
                    # 仅在启用 wave 协调时处理
                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            # 2. Notification from rank 0 engine that we've
                            # moved into the global paused state
                            # (engines_running==False).
                            # 2. rank 0 引擎通知我们已进入全局暂停状态。
                            if current_wave <= wave:
                                # 如果 wave 有效
                                new_wave = wave + 1  # 下一 wave = 完成 wave + 1
                                logger.debug(
                                    # 记录调试日志
                                    "Moving DP wave from %d to %d.",
                                    current_wave,  # 旧 wave
                                    new_wave,  # 新 wave
                                )
                                current_wave = new_wave  # 更新 wave
                                engines_running = False  # 标记暂停
                                wave_state_changed = True  # wave 状态已变化
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave  # 更新的 wave
                            or (wave == current_wave and not engines_running)
                            # 同 wave 但引擎暂停
                        ):
                            # 3. The engine received request for a non-current wave
                            # so we must ensure that other engines progress to the
                            # next wave (race condition handling).
                            # 3. 引擎收到非当前 wave 的请求，
                            # 因此我们必须确保其他引擎推进到下一 wave（竞态处理）。
                            logger.debug(
                                # 记录调试日志
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,  # wave 编号
                            )
                            current_wave = wave  # 更新 wave
                            engines_running = True  # 标记运行
                            wave_state_changed = True  # wave 状态已变化
                            self._send_start_wave(  # 广播启动 wave
                                publish_back,  # 后端发布 socket
                                wave,  # wave 编号
                                eng_index,  # 排除通知发来的引擎
                            )

                if wave_state_changed:
                    # 如果 wave 状态变化
                    message = (None, current_wave, engines_running)
                    # 构建消息：(无统计, wave, 运行状态)
                    publish_front.send(msgspec.msgpack.encode(message))
                    # 向前端发布 wave 状态更新

    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket,  # 后端发布 socket
        wave: int,  # wave 编号
        exclude_engine_index: int | None,  # 需要排除的引擎索引
    ):
        """Broadcast the START_DP_WAVE message to all the engines.
        It includes the current wave number and index of engine which
        has already received a request with this wave number and so doesn't
        require additional notification.
        """
        # 向所有引擎广播 START_DP_WAVE 消息。
        # 包含当前 wave 编号和已收到该 wave 请求的引擎索引
        # （该引擎不需要额外的通知）。
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        # 编码 (wave, 排除引擎)
        socket.send_multipart(
            (EngineCoreRequestType.START_DP_WAVE.value, wave_encoded)
        )
        # 发送多部分消息：请求类型 + wave 数据

    def _get_engine_counts(self, do_copy=False) -> list[list[int | float]]:
        """Return list of [waiting, running] count lists for each engine."""
        # 返回每个引擎的 [等待, 运行] 计数列表
        if do_copy:
            # 如果需要副本
            return [copy.copy(e.request_counts) for e in self.engines]
            # 返回每个引擎计数的浅拷贝
        return [e.request_counts for e in self.engines]
        # 否则返回直接引用