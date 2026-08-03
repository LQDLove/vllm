# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import asyncio  # asyncio：异步 I/O 框架
import contextlib  # contextlib：上下文管理器工具（suppress 异常抑制）
import queue  # queue：同步队列（SyncMPClient 用）
import sys  # sys：系统模块（maxsize 用于负载均衡评分）
import uuid  # uuid：生成唯一 ID（call_id 用）
import weakref  # weakref：弱引用（防止循环引用）
from abc import ABC, abstractmethod  # ABC：抽象基类；abstractmethod：抽象方法
from collections import Counter, defaultdict, deque
# Counter：计数器；defaultdict：带默认值字典；deque：双端队列
from collections.abc import Awaitable, Callable, Sequence
# Awaitable：可等待对象；Callable：可调用对象；Sequence：序列类型
from concurrent.futures import Future  # Future：并发未来对象
from dataclasses import dataclass  # dataclass：数据类装饰器
from multiprocessing.connection import Connection  # 进程间管道连接
from multiprocessing.queues import Queue  # 多进程队列（tensor IPC）
from threading import Thread  # 线程
from typing import Any, TypeAlias, TypeVar  # 类型标注工具

import msgspec  # msgspec：高性能 msgpack 序列化
import msgspec.msgpack  # msgpack 编码解码
import zmq  # zmq：ZeroMQ 消息队列
import zmq.asyncio  # zmq 异步支持

from vllm import envs  # vllm 环境变量
from vllm.config import VllmConfig  # vLLM 全局配置
from vllm.envs import VLLM_ENGINE_READY_TIMEOUT_S  # 引擎就绪超时
from vllm.logger import init_logger  # 初始化日志记录器
from vllm.lora.request import LoRARequest  # LoRA 请求
from vllm.tasks import SupportedTask  # 支持的任务类型
from vllm.tracing import instrument  # 追踪装饰器
from vllm.utils.async_utils import in_loop  # 检查是否在事件循环中
from vllm.utils.network_utils import (
    close_sockets,  # 关闭 sockets
    get_open_zmq_inproc_path,  # 获取开放 ZMQ IPC 路径
    make_zmq_socket,  # 创建 ZMQ socket
)
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,  # 弹性 EP 通知 call_id
    FT_STATUS_CALL_ID,  # 容错状态 call_id
    EEPNotificationType,  # 弹性 EP 通知类型
    EngineCoreOutputs,  # 引擎核心输出容器
    EngineCoreReadyResponse,  # 引擎就绪响应
    EngineCoreRequest,  # 引擎核心请求
    EngineCoreRequestType,  # 引擎核心请求类型
    PauseMode,  # 暂停模式
    ReconfigureDistributedRequest,  # 分布式重配置请求
    ReconfigureRankType,  # 重配置 rank 类型
    UtilityOutput,  # 工具输出
)
from vllm.v1.engine.coordinator import DPCoordinator  # DP 协调器
from vllm.v1.engine.core import EngineCore, EngineCoreProc
# 引擎核心；引擎核心进程
from vllm.v1.engine.exceptions import EngineDeadError  # 引擎死亡错误
from vllm.v1.engine.tensor_ipc import TensorIpcSender  # 张量 IPC 发送器
from vllm.v1.engine.utils import (
    CoreEngineActorManager,  # Ray actor 引擎管理器
    CoreEngineProcManager,  # 进程引擎管理器
    get_engine_zmq_addresses,  # 获取引擎 ZMQ 地址
    launch_core_engines,  # 启动核心引擎
)
from vllm.v1.executor import Executor  # 执行器抽象类
from vllm.v1.fault_tolerance.engine_core_sentinel import FT_UTILITY_METHOD
# 容错工具方法名
from vllm.v1.fault_tolerance.utils import (
    FaultToleranceRequest,  # 容错请求
    FaultToleranceResult,  # 容错结果
)
from vllm.v1.pool.late_interaction import get_late_interaction_engine_index
# 获取晚期交互引擎索引（pooling 模型）
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
# msgpack 解码器、编码器、字节字符串类型

logger = init_logger(__name__)  # 模块级日志记录器

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]
# 类型别名：异步或同步 Future

_R = TypeVar("_R")  # 泛型返回类型变量
EngineIdentity = bytes  # 引擎身份类型（字节标识）


class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """
    # 引擎核心客户端抽象基类。
    # 子类处理不同方式的前端与核心引擎通信：
    # * InprocClient：同进程 EngineCore（V0 风格 LLMEngine 使用）
    # * SyncMPClient：ZMQ + 后台进程（LLM 使用，同步）
    # * AsyncMPClient：ZMQ + 后台进程（AsyncLLM 使用，异步）

    @staticmethod
    def make_client(
        multiprocess_mode: bool,  # 是否多进程模式
        asyncio_mode: bool,  # 是否异步模式
        vllm_config: VllmConfig,  # vLLM 配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
    ) -> "EngineCoreClient":
        # 静态工厂：根据模式创建合适的客户端
        # TODO: support this for debugging purposes.
        # TODO：为调试目的支持此功能
        if asyncio_mode and not multiprocess_mode:
            # 异步但非多进程（目前不支持）
            raise NotImplementedError(
                # 抛出未实现错误
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            # 多进程 + 异步
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )
            # 创建异步多进程客户端

        if multiprocess_mode and not asyncio_mode:
            # 多进程 + 同步
            return SyncMPClient(vllm_config, executor_class, log_stats)
            # 创建同步多进程客户端

        return InprocClient(vllm_config, executor_class, log_stats)
        # 同进程客户端

    @staticmethod
    @instrument(span_name="Overall Loading")
    def make_async_mp_client(
        vllm_config: VllmConfig,  # vLLM 配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, Any] | None = None,  # 客户端地址
        client_count: int = 1,  # 客户端数量
        client_index: int = 0,  # 客户端索引
    ) -> "AsyncMPClient":
        # 静态工厂：创建异步多进程客户端
        parallel_config = vllm_config.parallel_config  # 并行配置
        client_args = (  # 公共参数
            vllm_config,  # 配置
            executor_class,  # 执行器
            log_stats,  # 日志统计
            client_addresses,  # 地址
            client_count,  # 客户端数
            client_index,  # 客户端索引
        )
        if parallel_config.data_parallel_size > 1:
            # 如果 DP>1
            if parallel_config.data_parallel_external_lb:
                # 外部负载均衡
                return DPAsyncMPClient(*client_args)
                # 返回 DP 客户端（外部 LB）
            # Internal load balancer - client balances to all DP ranks.
            # 内部负载均衡 - 客户端在所有 DP rank 间负载均衡
            return DPLBAsyncMPClient(*client_args)
            # 返回 DP 负载均衡客户端
        return AsyncMPClient(*client_args)
        # 非 DP：返回普通异步客户端

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None: ...
    # 抽象方法：关闭客户端

    def get_output(self) -> EngineCoreOutputs:
        # 获取输出（默认未实现）
        raise NotImplementedError

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 获取支持的任务（默认未实现）
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        # 添加请求（默认未实现）
        raise NotImplementedError

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        # 启停性能分析（默认未实现）
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        # 重置多模态缓存（默认未实现）
        raise NotImplementedError

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 重置前缀缓存（默认未实现）
        raise NotImplementedError

    def reset_encoder_cache(self) -> None:
        # 重置编码器缓存（默认未实现）
        raise NotImplementedError

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 引擎休眠（默认未实现）
        raise NotImplementedError

    def wake_up(self, tags: list[str] | None = None) -> None:
        # 引擎唤醒（默认未实现）
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        # 检查休眠（默认未实现）
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        # 执行空 batch（默认未实现）
        raise NotImplementedError

    def set_weight_version(self, weight_version: str) -> None:
        raise NotImplementedError

    def get_weight_version(self) -> str:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        # 异步执行空 batch（默认未实现）
        raise NotImplementedError

    async def set_weight_version_async(self, weight_version: str) -> None:
        raise NotImplementedError

    async def get_weight_version_async(self) -> str:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        # 中止请求（默认未实现）
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 添加 LoRA（默认未实现）
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        # 移除 LoRA（默认未实现）
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        # 列出 LoRA（默认未实现）
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        # 固定 LoRA（默认未实现）
        raise NotImplementedError

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 保存分片状态（默认未实现）
        raise NotImplementedError

    def collective_rpc(
        self,
        method: str | Callable[..., _R],  # 方法名或函数
        timeout: float | None = None,  # 超时
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:
        # 集体 RPC（默认未实现）
        raise NotImplementedError

    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        # 返回 DP 引擎是否集体处于运行状态（默认未实现）
        raise NotImplementedError

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        # 弹性 EP 缩放（默认未实现）
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        # 异步获取输出（默认未实现）
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        # 异步获取支持任务（默认未实现）
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # 异步添加请求（默认未实现）
        raise NotImplementedError

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        # 异步启停 profiler（默认未实现）
        raise NotImplementedError

    async def reset_mm_cache_async(self) -> None:
        # 异步重置多模态缓存（默认未实现）
        raise NotImplementedError

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 异步重置前缀缓存（默认未实现）
        raise NotImplementedError

    async def reset_encoder_cache_async(self) -> None:
        # 异步重置编码器缓存（默认未实现）
        raise NotImplementedError

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 异步休眠（默认未实现）
        raise NotImplementedError

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        # 异步唤醒（默认未实现）
        raise NotImplementedError

    async def is_sleeping_async(self) -> bool:
        # 异步检查休眠（默认未实现）
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # 异步中止请求（默认未实现）
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        # 异步添加 LoRA（默认未实现）
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        # 异步移除 LoRA（默认未实现）
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        # 异步列出 LoRA（默认未实现）
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        # 异步固定 LoRA（默认未实现）
        raise NotImplementedError

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 异步保存分片状态（默认未实现）
        raise NotImplementedError

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],  # 方法名或函数
        timeout: float | None = None,  # 超时
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:
        # 异步集体 RPC（默认未实现）
        raise NotImplementedError

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        # 处理故障（默认未实现）
        raise NotImplementedError

    async def get_status(self):
        # 获取引擎状态（默认未实现）
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """
    # 同进程客户端：用于 LLMEngine 的 V0 风格 add_request() 和 step()。
    # 核心引擎与客户端在同一进程（无 busy loop）。
    # * 将 EngineCoreRequest 直接推入 EngineCore
    # * 通过步进 EngineCore 拉取输出

    def __init__(self, *args, **kwargs):
        # 构造函数
        self.engine_core = EngineCore(*args, **kwargs)
        # 在本地创建 EngineCore 实例

    def get_output(self) -> EngineCoreOutputs:
        # 获取输出：步进引擎
        outputs, model_executed = self.engine_core.step_fn()
        # 调用引擎核心的步进方法
        self.engine_core.post_step(model_executed=model_executed)
        # 步进后处理（如草稿 token 更新）
        return outputs and outputs.get(0) or EngineCoreOutputs()
        # 返回引擎 0 的输出（无则返回空容器）

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 获取支持的任务
        return self.engine_core.get_supported_tasks()
        # 直接从引擎核心获取

    def add_request(self, request: EngineCoreRequest) -> None:
        # 添加请求：直接推入同进程引擎
        req, request_wave = self.engine_core.preprocess_add_request(request)
        # 预处理请求（多模态缓存、结构化输出初始化）
        self.engine_core.add_request(req, request_wave)
        # 添加到调度器

    def abort_requests(self, request_ids: list[str]) -> None:
        # 中止请求
        if len(request_ids) > 0:
            # 如果有请求需要中止
            self.engine_core.abort_requests(request_ids)
            # 直接调用引擎核心

    def shutdown(self, timeout: float | None = None) -> None:
        # 关闭：直接关闭引擎核心
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        # 启停 profiler
        self.engine_core.profile(is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        # 重置多模态缓存
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 重置前缀缓存
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        # 重置编码器缓存
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 引擎休眠
        if mode == "wait":
            # wait 模式不支持
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
        result = self.engine_core.sleep(level, mode)  # 休眠
        assert result is None  # 断言同步完成

    def wake_up(self, tags: list[str] | None = None) -> None:
        # 引擎唤醒
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        # 检查休眠
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self) -> None:
        # 执行空 batch
        self.engine_core.execute_dummy_batch()

    def set_weight_version(self, weight_version: str) -> None:
        self.engine_core.set_weight_version(weight_version)

    def get_weight_version(self) -> str:
        return self.engine_core.get_weight_version()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 添加 LoRA
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        # 移除 LoRA
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        # 列出 LoRA
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        # 固定 LoRA
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 保存分片状态
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        # 集体 RPC
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def dp_engines_running(self) -> bool:
        # DP 引擎运行状态（同进程无 DP）
        return False


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""
    # 后台资源：用于干净关闭的终结器，避免循环引用回客户端对象

    ctx: zmq.Context  # ZMQ 上下文
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    # 如果是 CoreEngineProcManager，管理本地引擎；
    # 如果是 CoreEngineActorManager，管理所有引擎。
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None
    # 引擎管理器
    coordinator: DPCoordinator | None = None  # DP 协调器
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    # 输出 socket
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    # 输入 socket
    first_req_send_socket: zmq.asyncio.Socket | None = None
    # 首请求发送 socket（DP 唤醒通知）
    first_req_rcv_socket: zmq.asyncio.Socket | None = None
    # 首请求接收 socket
    stats_update_socket: zmq.asyncio.Socket | None = None
    # 统计更新 socket（订阅协调器）
    output_queue_task: asyncio.Task | None = None  # 输出队列任务
    stats_update_task: asyncio.Task | None = None  # 统计更新任务
    shutdown_path: str | None = None  # 关闭路径

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    # 标记任何引擎是否死亡。放在这里以便输出处理线程无需持有客户端引用
    # 就能访问。
    engine_dead: bool = False  # 引擎死亡标志

    def __call__(self):
        """Clean up background resources."""
        # 清理后台资源

        logger.debug_once("[shutdown] MPClient: background resource cleanup start")
        # 记录清理开始
        self.engine_dead = True  # 标记引擎死亡
        if self.engine_manager is not None:
            # 如果有引擎管理器
            self.engine_manager.shutdown(
                timeout=envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS
            )
            # 关闭引擎管理器
        if self.coordinator is not None:
            # 如果有协调器
            self.coordinator.shutdown()  # 关闭协调器

        if isinstance(self.output_socket, zmq.asyncio.Socket):
            # 异步情况
            loop = self.output_queue_task._loop if self.output_queue_task else None
            # 获取事件循环
            sockets = (  # 需要关闭的所有 sockets
                self.output_socket,
                self.input_socket,
                self.first_req_send_socket,
                self.first_req_rcv_socket,
                self.stats_update_socket,
            )
            tasks = (self.output_queue_task, self.stats_update_task)
            # 需要取消的任务

            def close_sockets_and_tasks():
                # 关闭 sockets 和任务
                close_sockets(sockets)  # 关闭 sockets
                for task in tasks:
                    # 遍历任务
                    if task is not None and not task.done():
                        # 如果任务未完成
                        with contextlib.suppress(Exception):
                            # 抑制异常
                            task.cancel()  # 取消任务

            if loop is not None:
                # 如果事件循环存在
                if in_loop(loop):
                    # 如果在当前循环中
                    close_sockets_and_tasks()  # 直接执行
                elif not loop.is_closed():
                    # 如果循环未关闭
                    loop.call_soon_threadsafe(close_sockets_and_tasks)
                    # 线程安全调度
            else:
                # Loop has been closed, try to clean up directly.
                # 循环已关闭，直接清理
                del tasks  # 删除任务引用
                del close_sockets_and_tasks  # 删除函数引用
                close_sockets(sockets)  # 关闭 sockets
                del self.output_queue_task  # 删除任务引用
                del self.stats_update_task  # 删除任务引用
        else:
            # Sync case.
            # 同步情况
            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            # 如果 sockets 未显式关闭，ZMQ 上下文终止可能挂起。
            close_sockets((self.output_socket, self.input_socket))
            # 关闭输入输出 sockets

            if self.shutdown_path is not None:
                # 如果有关闭路径
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                # 必须确保同步输出 socket 在自己的线程中干净关闭
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    # 创建关闭通知 socket
                    shutdown_sender.connect(self.shutdown_path)
                    # 连接关闭路径
                    shutdown_sender.send(b"")  # 发送关闭信号

        logger.debug_once("[shutdown] MPClient: background resource cleanup complete")
        # 记录清理完成

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        # 验证引擎存活（检查死亡信号）
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            # 如果收到引擎死亡信号
            self.engine_dead = True  # 标记死亡
            raise EngineDeadError()  # 抛出引擎死亡错误


@dataclass
class ElasticScalingCache:
    # 弹性扩展缓存
    existing_core_engines: list[EngineIdentity]
    # 现有引擎列表
    num_new_core_engines: int
    # 新引擎数量（负数表示缩容）
    pending_notifications: dict[EEPNotificationType, set[int]]
    # 待处理通知


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """
    # 多进程客户端基类。核心引擎在后台进程忙循环中运行。
    # * 通过 input_socket 推送请求
    # * 通过 output_socket 拉取输出
    # * AsyncMPClient 子类用于 AsyncLLM
    # * SyncMPClient 子类用于 LLM

    def __init__(
        self,
        asyncio_mode: bool,  # 是否异步模式
        vllm_config: VllmConfig,  # vLLM 配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, Any] | None = None,  # 客户端地址
    ):
        self.vllm_config = vllm_config  # 保存配置

        # ZMQ setup.
        # ZMQ 初始化
        sync_ctx = zmq.Context(io_threads=2)  # 创建同步上下文（2 个 IO 线程）
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx
        # 异步模式用异步上下文，否则用同步上下文

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        # 这确保客户端被 GC 时创建的资源被关闭，即使构造中途抛出异常。
        self.resources = BackgroundResources(ctx=sync_ctx)  # 后台资源
        self._finalizer = weakref.finalize(self, self.resources)
        # 弱引用终结器
        success = False  # 构造成功标志
        try:
            # State used for data parallel.
            # 用于数据并行的状态
            self.engines_running = False  # 引擎运行标志
            parallel_config = vllm_config.parallel_config  # 并行配置
            # Elastic EP can remove a rank and later add it back with the same
            # identity. The client input ROUTER needs handover to allow the new
            # engine to replace the dead connection.
            # 弹性 EP 可以移除 rank 后用相同身份重新添加。
            # 客户端 input ROUTER 需要交接（handover）以允许新引擎
            # 替换死亡连接。
            enable_input_socket_handover = parallel_config.enable_elastic_ep
            # 是否启用输入 socket 交接

            self.stats_update_address: str | None = None  # 统计更新地址
            tensor_queue: Queue | None = None  # 张量队列
            if client_addresses:
                # Engines are managed externally to this client.
                # 引擎由客户端外部管理（多 API 服务器场景）
                input_address = client_addresses["input_address"]  # 输入地址
                output_address = client_addresses["output_address"]  # 输出地址
                self.stats_update_address = client_addresses.get("stats_update_address")
                # 统计更新地址
                # Tensor queues passed via client_addresses for multi-API-server case
                # 多 API 服务器场景通过 client_addresses 传递张量队列
                tensor_queue = client_addresses.get("tensor_queue")  # 张量队列
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,  # 上下文
                    input_address,  # 输入地址
                    zmq.ROUTER,  # 路由器模式
                    bind=True,  # 绑定
                    router_handover=enable_input_socket_handover,  # 交接支持
                )
                # 创建输入 ROUTER socket
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )
                # 创建输出 PULL socket

                # Report bound endpoints back so the parent can forward
                # them to engines (mirrors the DPCoordinator pattern).
                # 报告绑定端点以便父进程转发给引擎（镜像 DPCoordinator 模式）。
                actual_address_pipe: Connection | None = client_addresses.get(
                    "actual_address_pipe"
                )
                # 获取地址管道
                if actual_address_pipe is not None:
                    # 如果管道存在
                    try:
                        actual_input = self.input_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        # 获取实际输入地址
                        actual_output = self.resources.output_socket.getsockopt(
                            zmq.LAST_ENDPOINT
                        ).decode()
                        # 获取实际输出地址
                        actual_address_pipe.send(
                            # 报告地址
                            {
                                "input_address": actual_input,  # 输入
                                "output_address": actual_output,  # 输出
                            }
                        )
                    finally:
                        actual_address_pipe.close()  # 关闭管道
            else:
                # Engines are managed by this client.
                # 引擎由本客户端管理
                addresses = get_engine_zmq_addresses(vllm_config)
                # 获取引擎 ZMQ 地址
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,  # 上下文
                    addresses.inputs[0],  # 输入地址
                    zmq.ROUTER,  # 路由器模式
                    bind=True,  # 绑定
                    router_handover=enable_input_socket_handover,  # 交接
                )
                # 创建输入 ROUTER socket
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, addresses.outputs[0], zmq.PULL
                )
                # 创建输出 PULL socket

                # Resolve ``tcp://host:0`` placeholders to bound endpoints
                # before engines DEALER-connect. No-op for IPC.
                # 在引擎 DEALER 连接前解析 ``tcp://host:0`` 占位符为绑定端点。
                # IPC 路径无操作。
                addresses.inputs[0] = self.input_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                # 解析输入地址
                addresses.outputs[0] = self.resources.output_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                # 解析输出地址

                with launch_core_engines(
                    vllm_config, executor_class, log_stats, addresses
                ) as (engine_manager, coordinator, addresses, tensor_queue):
                    # 启动核心引擎进程
                    self.resources.coordinator = coordinator  # 保存协调器
                    self.resources.engine_manager = engine_manager  # 保存管理器

                self.stats_update_address = addresses.frontend_stats_publish_address
                # 统计更新地址
                if coordinator is not None:
                    # 如果有协调器
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address()
                    )
                    # 断言地址一致

            # Serialization setup with tensor queues for multimodal tensor IPC.
            # 为多模态张量 IPC 设置序列化（带张量队列）
            tensor_ipc_sender: TensorIpcSender | None = None  # 张量发送器
            model_config = getattr(vllm_config, "model_config", None)  # 模型配置
            if model_config is not None and model_config.multimodal_config is not None:
                # 如果有多模态配置
                mm_tensor_ipc = model_config.multimodal_config.mm_tensor_ipc
                # 张量 IPC 模式
                if mm_tensor_ipc == "torch_shm" and tensor_queue is not None:
                    # 如果启用 torch 共享内存
                    tensor_ipc_sender = TensorIpcSender(tensor_queue)
                    # 创建张量发送器

            self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
            # 创建编码器（带张量旁路）
            self.decoder = MsgpackDecoder(EngineCoreOutputs)
            # 创建解码器

            dp_size = parallel_config.data_parallel_size  # DP 大小
            dp_rank = parallel_config.data_parallel_index  # DP rank
            dp_local_size = parallel_config.data_parallel_size_local  # 本地 DP
            offline_mode = parallel_config.data_parallel_rank_local is not None
            # 是否离线模式
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            # 纯内部 LB 模式：客户端管理本地+远程引擎
            # 混合和外部 LB 模式：客户端只管理本地引擎
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
            # rank 数量
            self.engine_ranks_managed = (
                # 管理的引擎 rank 列表
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
                # 离线模式只管理本地 rank
            )
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )
            # 断言本地 DP 大小不超过管理数量

            # ZMQ identity of each engine that this client will talk to.
            # 本客户端将与之通信的每个引擎的 ZMQ 身份
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]
            # 将 rank 转为 2 字节小端标识

            # Wait for ready messages from each engine on the input socket.
            # 在输入 socket 上等待每个引擎的就绪消息
            identities = set(self.core_engines)  # 待就绪的引擎集合
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            # 从异步 socket 创建同步影子（用于阻塞等待）
            while identities:
                # 循环直到所有引擎就绪
                if not sync_input_socket.poll(
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
                ):
                    # 如果超时
                    raise TimeoutError(
                        # 抛出超时错误
                        f"Timed out waiting for engine core processes to "
                        f"start. This is often caused by slow weight loading "
                        f"for large models. Waited "
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                        f"timeout, set the environment variable: "
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                    )
                identity, payload = sync_input_socket.recv_multipart()
                # 接收身份和负载
                identities.remove(identity)  # 移除已就绪的引擎
                self._apply_ready_response(payload)  # 应用就绪响应

            self.core_engine: EngineIdentity = self.core_engines[0]
            # 默认使用第一个引擎
            self.utility_results: dict[int, AnyFuture] = {}
            # 工具调用结果映射（call_id → Future）

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            # 请求对象可能包含 PyTorch 分配的 tensor，
            # 需要保持引用直到 zmq 完成底层数据处理。
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()
            # 待处理消息队列

            # Start monitoring engine core processes for unexpected failures
            # 启动引擎核心进程监控（检测意外故障）
            self.start_engine_core_monitor()

            success = True  # 标记构造成功
        finally:
            if not success:
                # 如果构造失败
                self._finalizer()  # 触发清理

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        # 在超时内关闭引擎管理器并清理资源
        if self._finalizer.detach() is not None:
            # 如果终结器尚未执行
            timeout_str = "default" if timeout is None else f"{timeout}s"
            # 格式化超时字符串
            logger.info("[shutdown] MPClient: start timeout=%s", timeout_str)
            # 记录关闭开始
            if self.resources.engine_manager is not None:
                # 如果有引擎管理器
                logger.info_once("[shutdown] MPClient: stopping engine manager")
                # 记录日志
                self.resources.engine_manager.shutdown(timeout=timeout)
                # 关闭引擎管理器
                logger.info_once("[shutdown] MPClient: engine manager stopped")
                # 记录日志
            logger.info_once("[shutdown] MPClient: cleaning up background resources")
            # 记录日志
            self.resources()  # 清理后台资源
            logger.info_once("[shutdown] MPClient: complete")  # 记录完成

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        # 如果出错，使用 EngineDeadError 使根因清晰
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )
        # 引擎死亡时包装错误，否则返回原错误

    def ensure_alive(self):
        # 确保引擎存活
        if self.resources.engine_dead:
            # 如果引擎死亡
            raise EngineDeadError()  # 抛出错误

    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        # 添加待处理消息（保持 tensor 引用）
        if not tracker.done:
            # 如果跟踪器未完成
            self.pending_messages.appendleft((tracker, msg))
            # 加入队列

    def free_pending_messages(self):
        # 释放已完成的待处理消息
        while self.pending_messages and self.pending_messages[-1][0].done:
            # 循环直到队尾未完成
            self.pending_messages.pop()  # 弹出已完成的

    def dp_engines_running(self) -> bool:
        # DP 引擎是否运行
        return self.engines_running  # 返回标志

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        # 启动引擎核心进程监控线程
        engine_manager = self.resources.engine_manager  # 引擎管理器
        if engine_manager is None:
            # 如果没有引擎进程可监控
            return  # 直接返回

        self_ref = weakref.ref(self)  # 弱引用 self（避免循环引用）

        # Monitor engine core process liveness. If any die unexpectedly,
        # marks the engine as dead, and shuts down the client.
        # 监控引擎核心进程存活。如有意外死亡，标记引擎死亡并关闭客户端。
        def monitor_engine_cores():
            # 监控线程目标
            engine_manager.monitor_engine_liveness()  # 监控引擎存活
            _self = self_ref()  # 获取 self（可能为 None）
            if not _self or not _self._finalizer.alive or _self.resources.engine_dead:
                # 如果 self 已回收或已死亡
                return  # 返回
            _self.resources.engine_dead = True  # 标记引擎死亡
            logger.warning_once(
                # 记录警告
                "[shutdown] MPClient: engine core exited unexpectedly; starting cleanup"
            )
            _self.shutdown()  # 关闭客户端
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError
            # 注意：MPClient 没有像 MultiprocExecutor 那样的失败回调机制，
            # 但我们设置了 engine_dead 标志，将导致后续操作抛出 EngineDeadError

        Thread(
            # 创建监控线程
            target=monitor_engine_cores,  # 目标函数
            daemon=True,  # 守护线程
            name="MPClientEngineMonitor",  # 线程名
        ).start()  # 启动

    def _apply_ready_response(self, payload: bytes) -> None:
        """Decode an EngineCoreReadyResponse and sync any post-initialization
        config changes (e.g. auto-fitted max_model_len) back to the frontend."""
        # 解码 EngineCoreReadyResponse 并同步初始化后的配置更改
        # （如自动适配的 max_model_len）回前端
        if not payload:
            # 如果负载为空
            return  # 直接返回
        vllm_config = self.vllm_config  # 获取配置
        response = msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
        # 解码就绪响应
        vllm_config.model_config.max_model_len = min(
            vllm_config.model_config.max_model_len, response.max_model_len
        )
        # 取较小的 max_model_len

        # Setup KV cache config with initialization state from
        # engine core process. Sum num_gpu_blocks from all engines in DP case.
        # 使用引擎核心进程的初始化状态设置 KV 缓存配置。
        # DP 情况下将所有引擎的 num_gpu_blocks 相加。
        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        # 现有 GPU blocks
        num_gpu_blocks += response.num_gpu_blocks  # 累加新引擎的 blocks
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks  # 更新配置

        # Sync block_size: may be enlarged by _align_hybrid_block_size in the
        # worker for hybrid Mamba models.
        # 同步 block_size：可能被混合 Mamba 模型的 worker 中的
        # _align_hybrid_block_size 放大。
        cache_config = vllm_config.cache_config  # 缓存配置
        cache_config.block_size = response.block_size  # 更新 block_size
        # Keep these as per-engine cache_config_info values; do not sum across DP.
        # 保持这些为每引擎缓存配置值；不跨 DP 相加。
        cache_config.kv_cache_size_tokens = (
            getattr(cache_config, "kv_cache_size_tokens", None)
            if getattr(cache_config, "kv_cache_size_tokens", None) is not None
            else response.kv_cache_size_tokens
        )
        # 更新 KV 缓存大小（如果之前未设置）
        cache_config.kv_cache_max_concurrency = (
            getattr(cache_config, "kv_cache_max_concurrency", None)
            if getattr(cache_config, "kv_cache_max_concurrency", None) is not None
            else response.kv_cache_max_concurrency
        )
        # 更新 KV 缓存最大并发度

        # In external DP LB mode, the coordinator address that the
        # front-end procs connect to is obtained by each engine via it's
        # initial handshake with the rank 0 front-end.
        # 在外部 DP LB 模式下，前端进程连接的协调器地址由每个引擎
        # 通过与 rank 0 前端的初始握手获取。
        if response.dp_stats_address is not None:
            # 如果有 DP 统计地址
            if self.stats_update_address is None:
                # 如果尚未设置
                self.stats_update_address = response.dp_stats_address
                # 设置地址
            else:
                assert response.dp_stats_address == self.stats_update_address
                # 断言地址一致


def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    # 将工具方法的执行结果设置到等待的 Future 中
    future = utility_results.pop(output.call_id)  # 取出对应的 Future
    failure_message = output.failure_message  # 失败消息
    try:
        if failure_message is not None:
            # 如果调用失败
            future.set_exception(Exception(failure_message))  # 设置异常
        else:
            assert output.result is not None  # 断言结果存在
            future.set_result(output.result.result)  # 设置结果
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        # 如果原始调用任务被取消，Future 可能已取消
        if failure_message is not None:
            # 如果有失败消息
            logger.error(
                # 记录错误
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""
    # 多进程核心引擎的同步客户端

    @instrument(span_name="SyncMPClient init")
    def __init__(
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool
    ):
        # 构造函数
        super().__init__(
            asyncio_mode=False,  # 同步模式
            vllm_config=vllm_config,  # 配置
            executor_class=executor_class,  # 执行器
            log_stats=log_stats,  # 日志统计
        )

        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1
        # 是否为 DP 模式
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()
        # 同步输出队列

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        # 确保输出 socket 处理线程没有阻止 GC 的客户端引用。
        ctx = self.ctx  # 上下文
        out_socket = self.resources.output_socket  # 输出 socket
        decoder = self.decoder  # 解码器
        utility_results = self.utility_results  # 工具结果
        outputs_queue = self.outputs_queue  # 输出队列
        # 修复：这一行在原始代码中缩进有问题，但保留原样

        shutdown_path = get_open_zmq_inproc_path()  # 关闭路径
        resources = self.resources  # 后台资源
        resources.shutdown_path = shutdown_path  # 设置关闭路径

        def process_outputs_socket():
            # 输出 socket 处理线程
            assert isinstance(out_socket, zmq.Socket)  # 断言是同步 socket
            shutdown_socket = ctx.socket(zmq.PAIR)  # 创建关闭通知 socket
            try:
                shutdown_socket.bind(shutdown_path)  # 绑定关闭路径
                poller = zmq.Poller()  # 创建轮询器
                poller.register(shutdown_socket, zmq.POLLIN)  # 注册关闭
                poller.register(out_socket, zmq.POLLIN)  # 注册输出
                while True:
                    socks = poller.poll()  # 轮询
                    if not socks:
                        # 如果没有事件
                        continue  # 继续
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        # 收到关闭信号，退出线程
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    # 接收帧
                    resources.validate_alive(frames)  # 验证存活
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    # 解码
                    if outputs.utility_output:
                        # 如果是工具输出
                        _process_utility_output(outputs.utility_output, utility_results)
                        # 处理工具结果
                    else:
                        outputs_queue.put_nowait(outputs)  # 推入输出队列
            except Exception as e:
                outputs_queue.put_nowait(e)  # 推送异常
            finally:
                # Close sockets.
                # 关闭 sockets
                shutdown_socket.close(linger=0)  # 关闭关闭 socket
                out_socket.close(linger=0)  # 关闭输出 socket

        # Process outputs from engine in separate thread.
        # 在独立线程中处理引擎输出
        self.output_queue_thread = Thread(
            target=process_outputs_socket,  # 线程目标
            name="EngineCoreOutputQueueThread",  # 线程名
            daemon=True,  # 守护线程
        )
        self.output_queue_thread.start()  # 启动线程

        # The thread takes on responsibility for closing the socket.
        # 线程负责关闭 socket。
        self.resources.output_socket = None  # 清空客户端引用

    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        # 如果 process_outputs_socket 任务中出现异常，
        # 它会转发到 outputs_queue，我们可以从这里（run_output_handler）
        # 抛出以关闭服务器。
        outputs = self.outputs_queue.get()  # 阻塞获取输出

        if isinstance(outputs, Exception):
            # 如果输出是异常
            raise self._format_exception(outputs) from None
            # 格式化并抛出
        if outputs.wave_complete is not None:
            # 如果 wave 完成
            self.engines_running = False  # 标记引擎暂停
        return outputs  # 返回输出

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        # 发送输入到核心引擎
        self.ensure_alive()  # 确保存活
        self.free_pending_messages()  # 释放已完成消息
        # (Identity, RequestType, SerializedRequest)
        # (身份, 请求类型, 序列化请求)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
        # 构建消息

        if len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            # 无辅助缓冲区 => 请求中无 tensor 底层缓冲区
            self.input_socket.send_multipart(msg, copy=False)  # 直接发送
            return

        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
        # 发送并跟踪（保持 tensor 引用）
        self.add_pending_message(tracker, request)  # 添加待处理消息

    def call_utility(self, method: str, *args) -> Any:
        # 同步工具调用
        call_id = uuid.uuid1().int >> 64  # 生成唯一 call_id
        future: Future[Any] = Future()  # 创建同步 Future
        self.utility_results[call_id] = future  # 注册等待
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))
        # 发送工具调用
        return future.result()  # 阻塞等待结果

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 获取支持的任务
        return self.call_utility("get_supported_tasks")  # 工具调用

    def add_request(self, request: EngineCoreRequest) -> None:
        # 添加请求
        if self.is_dp:
            # 如果 DP 模式
            self.engines_running = True  # 标记引擎运行
        self._send_input(EngineCoreRequestType.ADD, request)  # 发送请求

    def abort_requests(self, request_ids: list[str]) -> None:
        # 中止请求
        if request_ids and not self.resources.engine_dead:
            # 如果有请求且引擎存活
            self._send_input(EngineCoreRequestType.ABORT, request_ids)
            # 发送中止请求

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        # 启停 profiler
        self.call_utility("profile", is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        # 重置多模态缓存
        self.call_utility("reset_mm_cache")

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 重置前缀缓存
        return self.call_utility(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        # 重置编码器缓存
        self.call_utility("reset_encoder_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 添加 LoRA
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        # 移除 LoRA
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        # 列出 LoRA
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        # 固定 LoRA
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 引擎休眠
        self.call_utility("sleep", level, mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        # 引擎唤醒
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        # 检查休眠
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self) -> None:
        # 执行空 batch
        self.call_utility("execute_dummy_batch")

    def set_weight_version(self, weight_version: str) -> None:
        self.call_utility("set_weight_version", weight_version)

    def get_weight_version(self) -> str:
        return self.call_utility("get_weight_version")

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        # 集体 RPC
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 保存分片状态
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""
    # 多进程核心引擎的异步兼容客户端

    @instrument(span_name="AsyncMPClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, Any] | None = None,  # 客户端地址
        client_count: int = 1,  # 客户端数量
        client_index: int = 0,  # 客户端索引
    ):
        super().__init__(
            asyncio_mode=True,  # 异步模式
            vllm_config=vllm_config,  # 配置
            executor_class=executor_class,  # 执行器
            log_stats=log_stats,  # 日志统计
            client_addresses=client_addresses,  # 地址
        )

        self.client_count = client_count  # 客户端数量
        self.client_index = client_index  # 客户端索引
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
        # 异步输出队列

        # locally-cached engine status
        # 本地缓存的引擎状态
        self._engine_status: dict[int, dict] = {}  # 引擎状态字典
        if self.vllm_config.parallel_config.enable_fault_tolerance:
            # 如果启用故障容错
            self._engine_status = {
                rank: {"id": rank, "status": "healthy"}  # 初始化为健康
                for rank in self.engine_ranks_managed  # 遍历管理的 rank
            }
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            # 如果在 asyncio 事件循环中，启动队列任务。
            # 否则将懒启动。如果此处未启动，可能错过引擎核心的
            # EXECUTOR_FAILED 消息（如果在发送任何请求前发生）。
            asyncio.get_running_loop()  # 获取事件循环
            self._ensure_output_queue_task()  # 启动输出队列任务
        except RuntimeError:
            pass  # 不在事件循环中，稍后懒启动

    def _ensure_output_queue_task(self):
        # 确保输出队列任务在运行（懒启动）
        resources = self.resources  # 后台资源
        if resources.output_queue_task is not None:
            # 如果任务已在运行
            return  # 直接返回

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        # 在独立任务中执行 IO 以尽可能并行化。
        # 避免任务直接引用客户端（防止循环引用）。
        decoder = self.decoder  # 解码器
        utility_results = self.utility_results  # 工具结果
        outputs_queue = self.outputs_queue  # 输出队列
        output_handler: (
            # 输出处理回调（子类可覆盖）
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)  # 获取类方法
        _self_ref = weakref.ref(self)  # 弱引用 self（避免循环引用）
        output_socket = resources.output_socket  # 输出 socket
        assert output_socket is not None  # 断言 socket 存在

        notification_callback_handler: (
            # 弹性 EP 通知处理回调
            Callable[[AsyncMPClient, Sequence[Any]], Any] | None
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)
        # 获取类方法

        async def process_outputs_socket():
            # 内部协程：处理输出 socket
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    # 异步接收帧
                    resources.validate_alive(frames)  # 验证引擎存活
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    # 解码输出
                    if outputs.utility_output:
                        # 如果是工具输出
                        if (
                            outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID
                            # 弹性 EP 通知
                            and notification_callback_handler is not None
                            # 有回调处理器
                        ):
                            assert _self_ref is not None  # 断言引用存在
                            _self = _self_ref()  # 获取 self（可能为 None）
                            if not _self:
                                # 如果客户端已回收
                                return  # 退出
                            if outputs.utility_output.result is None:
                                # 如果结果为空
                                continue  # 跳过
                            notification_data = outputs.utility_output.result.result
                            # 获取通知数据
                            assert isinstance(notification_data, Sequence)
                            # 断言是序列
                            assert len(notification_data) == 2  # 断言长度 2
                            asyncio.create_task(  # 创建通知处理任务
                                notification_callback_handler(_self, notification_data)
                            )
                        elif outputs.utility_output.call_id == FT_STATUS_CALL_ID:
                            # 容错状态通知
                            _self = _self_ref()  # 获取 self
                            if not _self:
                                # 如果客户端已回收
                                return  # 退出
                            if outputs.utility_output.result is not None:
                                # 如果有结果
                                _self._engine_status[outputs.engine_index] = (
                                    # 更新引擎状态缓存
                                    outputs.utility_output.result.result
                                )
                        else:
                            _process_utility_output(
                                # 普通工具输出：处理结果
                                outputs.utility_output, utility_results
                            )
                        continue  # 继续循环

                    if output_handler is not None:
                        # 如果有输出处理回调（DPLB 客户端）
                        assert _self_ref is not None  # 断言引用存在
                        _self = _self_ref()  # 获取 self
                        if not _self:
                            # Client has been garbage collected, abort.
                            # 客户端已被垃圾回收，中止
                            return
                        await output_handler(_self, outputs)  # 调用回调

                    if outputs.outputs or outputs.scheduler_stats:
                        # 如果有推理输出或调度统计
                        outputs_queue.put_nowait(outputs)  # 推入输出队列
            except Exception as e:
                outputs_queue.put_nowait(e)  # 推送异常
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())  # 推送引擎死亡错误

        resources.output_queue_task = asyncio.create_task(
            # 创建输出队列任务
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> EngineCoreOutputs:
        # 异步获取输出
        self._ensure_output_queue_task()  # 确保任务在运行
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        # 如果 process_outputs_socket 任务中出现异常，
        # 它会转发到 outputs_queue，我们可以从这里（run_output_handler）
        # 抛出以关闭服务器。
        assert self.outputs_queue is not None  # 断言队列存在
        outputs = await self.outputs_queue.get()  # 异步获取输出
        if isinstance(outputs, Exception):
            # 如果输出是异常
            raise self._format_exception(outputs) from None
            # 格式化并抛出
        return outputs  # 返回输出

    def _send_input(
        self,
        request_type: EngineCoreRequestType,  # 请求类型
        request: Any,  # 请求数据
        engine: EngineIdentity | None = None,  # 目标引擎（可选）
    ) -> Awaitable[Any]:
        # 发送输入到核心引擎
        if engine is None:
            # 如果未指定引擎
            engine = self.core_engine  # 使用默认引擎

        message = (request_type.value, *self.encoder.encode(request))
        # 构建消息（类型 + 序列化数据）
        return self._send_input_message(message, engine, request)
        # 发送消息

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        # objects 是需要保持的引用，直到 zmq 完成底层缓冲区的处理，
        # 以防它们是从请求中的张量提取的。
        self.ensure_alive()  # 确保引擎存活
        self.free_pending_messages()  # 释放已完成消息

        msg = (engine,) + message  # 拼接消息（引擎身份 + 消息）
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            # 无辅助缓冲区 => 请求中无 tensor 底层缓冲区
            return self.input_socket.send_multipart(msg, copy=False)
            # 直接发送（零拷贝）

        future: asyncio.Future[zmq.MessageTracker]
        # 异步 Future
        future = self.input_socket.send_multipart(msg, copy=False, track=True)
        # 发送并跟踪（保持张量引用）

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            # 完成后添加待处理消息（保持引用）
            with contextlib.suppress(BaseException):
                # 抑制异常
                self.add_pending_message(f.result(), objects)
                # 添加消息

        future.add_done_callback(add_pending)  # 注册完成回调
        return future  # 返回 Future

    async def call_utility_async(self, method: str, *args) -> Any:
        # 异步工具调用
        return await self._call_utility_async(method, *args, engine=self.core_engine)
        # 委托给私有方法（默认引擎）

    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity  # 目标引擎
    ) -> Any:
        # 私有方法：异步工具调用（指定引擎）
        call_id = uuid.uuid1().int >> 64  # 生成唯一 call_id
        future = asyncio.get_running_loop().create_future()  # 创建异步 Future
        self.utility_results[call_id] = future  # 注册等待
        message = (
            EngineCoreRequestType.UTILITY.value,  # 工具调用类型
            *self.encoder.encode((self.client_index, call_id, method, args)),
            # 编码 (客户端索引, call_id, 方法名, 参数)
        )
        await self._send_input_message(message, engine, args)  # 发送消息
        self._ensure_output_queue_task()  # 确保输出队列任务运行
        return await future  # 等待结果

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        # 异步获取支持的任务
        return await self.call_utility_async("get_supported_tasks")  # 工具调用

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # 异步添加请求
        request.client_index = self.client_index  # 设置客户端索引
        await self._send_input(EngineCoreRequestType.ADD, request)  # 发送请求
        self._ensure_output_queue_task()  # 确保输出队列任务运行

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # 异步中止请求
        if request_ids and not self.resources.engine_dead:
            # 如果有请求且引擎存活
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)
            # 发送中止请求

    async def pause_scheduler_async(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> None:
        # 异步暂停调度器
        await self.call_utility_async("pause_scheduler", mode, clear_cache)
        # 工具调用

    async def resume_scheduler_async(self) -> None:
        # 异步恢复调度器
        await self.call_utility_async("resume_scheduler")  # 工具调用

    async def is_scheduler_paused_async(self) -> bool:
        # 异步检查调度器暂停状态
        return await self.call_utility_async("is_scheduler_paused")  # 工具调用

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        # 异步启停 profiler
        await self.call_utility_async("profile", is_start, profile_prefix)

    async def reset_mm_cache_async(self) -> None:
        # 异步重置多模态缓存
        await self.call_utility_async("reset_mm_cache")  # 工具调用

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 异步重置前缀缓存
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )
        # 工具调用

    async def reset_encoder_cache_async(self) -> None:
        # 异步重置编码器缓存
        await self.call_utility_async("reset_encoder_cache")  # 工具调用

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 异步引擎休眠
        await self.call_utility_async("sleep", level, mode)  # 工具调用

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        # 异步引擎唤醒
        await self.call_utility_async("wake_up", tags)  # 工具调用

    async def is_sleeping_async(self) -> bool:
        # 异步检查休眠
        return await self.call_utility_async("is_sleeping")  # 工具调用

    async def execute_dummy_batch_async(self) -> None:
        # 异步执行空 batch
        await self.call_utility_async("execute_dummy_batch")  # 工具调用

    async def set_weight_version_async(self, weight_version: str) -> None:
        await self.call_utility_async("set_weight_version", weight_version)

    async def get_weight_version_async(self) -> str:
        return await self.call_utility_async("get_weight_version")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        # 异步添加 LoRA
        return await self.call_utility_async("add_lora", lora_request)  # 工具调用

    async def remove_lora_async(self, lora_id: int) -> bool:
        # 异步移除 LoRA
        return await self.call_utility_async("remove_lora", lora_id)  # 工具调用

    async def list_loras_async(self) -> set[int]:
        # 异步列出 LoRA
        return await self.call_utility_async("list_loras")  # 工具调用

    async def pin_lora_async(self, lora_id: int) -> bool:
        # 异步固定 LoRA
        return await self.call_utility_async("pin_lora", lora_id)  # 工具调用

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 异步保存分片状态
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)
        # 工具调用

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],  # 方法名或函数
        timeout: float | None = None,  # 超时
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:
        # 异步集体 RPC
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )
        # 工具调用

    async def handle_fault(
        self, ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        # 处理故障（容错）
        res = await self.call_utility_async(FT_UTILITY_METHOD, ft_request)
        # 调用容错工具方法
        result = msgspec.convert(res, FaultToleranceResult)
        # 转换结果类型
        if not result.success:
            # 如果失败
            status = self._engine_status.get(self.engine_ranks_managed[0])
            # 获取引擎状态
            if status is not None:
                # 如果有状态
                status["last_ft_request_id"] = result.request_id
                # 记录最近故障请求 ID
                status["ft_error"] = result.reason  # 记录错误原因
        return result  # 返回结果

    async def get_status(self):
        # 获取引擎状态
        return {
            "schema_version": 1,  # 模式版本
            "total_engines": len(self.engine_ranks_managed),  # 总引擎数
            "engines": list(self._engine_status.values()),  # 引擎状态列表
        }


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""
    # 多进程、多引擎（数据并行）核心引擎的异步兼容客户端。
    # 默认假设外部负载均衡。

    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, Any] | None = None,  # 客户端地址
        client_count: int = 1,  # 客户端数量
        client_index: int = 0,  # 客户端索引
    ):
        self.current_wave = 0  # 当前 wave 编号

        super().__init__(  # 调用父类初始化
            vllm_config,  # 配置
            executor_class,  # 执行器
            log_stats,  # 日志统计
            client_addresses,  # 地址
            client_count,  # 客户端数
            client_index,  # 客户端索引
        )

        # List of [waiting, running, kv_cache_usage] per engine.
        # Used only by DPLBAsyncMPClient subclass.
        # 每个引擎的 [等待数, 运行数, KV 缓存使用率] 列表。
        # 仅由 DPLBAsyncMPClient 子类使用。
        self.lb_engines: list[list[int | float]] = [
            [0, 0, 0.0] for _ in self.core_engines
        ]
        # 初始化负载均衡引擎状态

        self.eep_scaling_cache: ElasticScalingCache | None = None
        # 弹性 EP 缩放缓存

        self.first_req_sock_addr = get_open_zmq_inproc_path()
        # 首请求 socket 地址（IPC）
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        # 创建首请求发送 socket（用于唤醒暂停的引擎）
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            # 如果在 asyncio 事件循环中，启动统计任务。
            # 否则将懒启动。
            asyncio.get_running_loop()  # 获取事件循环
            self._ensure_stats_update_task()  # 启动统计更新任务
        except RuntimeError:
            pass  # 不在事件循环中，稍后懒启动

    def _ensure_stats_update_task(self):
        # 确保统计更新任务在运行
        resources = self.resources  # 后台资源
        if resources.stats_update_task is not None:
            # 如果任务已在运行
            return  # 直接返回

        assert self.stats_update_address is not None  # 断言统计地址存在
        stats_addr: str = self.stats_update_address  # 统计地址
        assert len(self.engine_ranks_managed) > 0  # 断言有管理的引擎

        async def run_engine_stats_update_task():
            # 内部协程：运行引擎统计更新
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                # 创建订阅协调器统计的 socket
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
                # 创建首请求接收 socket
            ):
                assert isinstance(socket, zmq.asyncio.Socket)  # 断言是异步 socket
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket  # 保存 socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                # 发送订阅消息。
                await socket.send(b"\x01")  # 订阅

                poller = zmq.asyncio.Poller()  # 创建异步轮询器
                poller.register(socket, zmq.POLLIN)  # 注册统计
                poller.register(first_req_rcv_socket, zmq.POLLIN)  # 注册首请求

                while True:
                    events = await poller.poll()  # 轮询事件
                    if (
                        not self.engines_running  # 引擎暂停
                        and len(events) == 2  # 两个 socket 都有事件
                        or (events[0][0] == first_req_rcv_socket)  # 或首请求事件
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        # 检查是常规请求通知还是扩缩容通知
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()
                        # 非阻塞接收

                        decoded = msgspec.msgpack.decode(buf)  # 解码消息
                        if (
                            isinstance(decoded, (list, tuple))  # 是列表/元组
                            and len(decoded) == 2  # 长度 2
                            and decoded[0] == "SCALE_ELASTIC_EP"  # 扩缩容通知
                        ):
                            # Extract new engine count from the decoded message
                            # 从解码消息中提取新引擎数量
                            new_engine_count = decoded[1]  # 新引擎数量
                            # Update engine_ranks_managed and count_slice
                            # 更新管理的引擎 rank 和计数切片
                            parallel_config = self.vllm_config.parallel_config
                            # 并行配置
                            dp_size = parallel_config.data_parallel_size  # DP 大小
                            dp_rank = parallel_config.data_parallel_rank  # DP rank
                            assert dp_rank == 0  # 断言 rank 0
                            assert dp_size == new_engine_count  # 断言大小一致
                            assert not (  # 断言非混合/外部负载均衡
                                parallel_config.data_parallel_hybrid_lb
                                or parallel_config.data_parallel_external_lb
                            )
                            num_ranks = dp_size  # rank 数量
                            self.engine_ranks_managed = list(
                                range(dp_rank, dp_rank + num_ranks)
                            )
                            # 更新管理的引擎
                            if len(self.lb_engines) < new_engine_count:
                                # 如果负载均衡列表过短（扩容）
                                self.lb_engines = self.lb_engines + [
                                    # 追加新的引擎状态
                                    [0, 0, 0.0]
                                    for _ in range(
                                        new_engine_count - len(self.lb_engines)
                                    )
                                ]
                            else:
                                # 缩容：截断
                                self.lb_engines = self.lb_engines[:new_engine_count]
                            # Send scale up notification to coordinator
                            # 向协调器发送扩缩容通知
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)  # 发送通知
                            continue  # 继续循环

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        # 引擎暂停时发送请求，以便唤醒其他引擎
                        # （运行空 EP 循环）。
                        assert decoded[0] == "FIRST_REQ"  # 断言是首请求
                        target_eng_index = decoded[1]  # 目标引擎索引
                        self.engines_running = True  # 标记引擎运行
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        # 编码 (目标引擎, 当前 wave)
                        await socket.send(msg)  # 发送唤醒消息

                    buf = None  # 缓冲区初始化为 None
                    while True:
                        # Drain all stats events (we only care about latest).
                        # 排空所有统计事件（只关心最新的）
                        future: asyncio.Future[bytes] = socket.recv(
                            flags=zmq.NOBLOCK
                        )
                        # 非阻塞接收
                        if isinstance(future.exception(), zmq.Again):
                            # 如果没有更多消息
                            break  # 退出排空
                        buf = future.result()  # 获取最新消息
                    if buf is None:
                        # 如果没有统计数据
                        continue  # 继续循环

                    # Update local load-balancing state.
                    # 更新本地负载均衡状态
                    counts, wave, running = msgspec.msgpack.decode(buf)
                    # 解码 (计数, wave, 运行状态)
                    self.current_wave = wave  # 更新当前 wave
                    self.engines_running = running  # 更新运行状态
                    if counts is not None:
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        # 运行和等待计数来自协调器的全局统计（含所有引擎核）。
                        # 切片获取本客户端管理的核心部分。
                        ranks = self.engine_ranks_managed  # 管理的 rank
                        count_slice = slice(ranks[0], ranks[-1] + 1)  # 切片
                        sliced_counts = counts[count_slice]  # 切片计数
                        self.lb_engines = sliced_counts  # 更新负载均衡状态
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )
                        # 记录调试日志

        resources.stats_update_task = asyncio.create_task(
            # 创建统计更新任务
            run_engine_stats_update_task()
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # 异步添加请求（DP 版本）
        self._ensure_stats_update_task()  # 确保统计任务运行

        request.current_wave = self.current_wave  # 设置当前 wave
        request.client_index = self.client_index  # 设置客户端索引

        chosen_engine = self.get_core_engine_for_request(request)
        # 选择目标引擎
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        # 发送请求到目标引擎
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            # 通知协调器正在发送请求（唤醒其他引擎）
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            # 编码首请求通知
            await self.first_req_send_socket.send(req_msg)
            # 发送通知

        await to_await  # 等待发送完成

        self._ensure_output_queue_task()  # 确保输出队列任务运行

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        # 为请求选择目标引擎（默认返回核心引擎）
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""
    # 多进程、多引擎（数据并行）核心引擎的异步兼容客户端。
    # 在多个引擎进程之间进行负载均衡。

    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        client_addresses: dict[str, Any] | None = None,  # 客户端地址
        client_count: int = 1,  # 客户端数量
        client_index: int = 0,  # 客户端索引
    ):
        self.client_count = client_count  # 客户端数量

        # To route aborts to the correct engine.
        # 用于将中止请求路由到正确的引擎。
        self.reqs_in_flight: dict[str, EngineIdentity] = {}
        # 进行中的请求 → 引擎映射

        # Exact per-engine count of this client's unfinished requests.
        # 本客户端在每个引擎上的未完成请求精确计数。
        self.engine_inflight: Counter[EngineIdentity] = Counter()
        # 引擎进行中计数器

        super().__init__(  # 调用父类初始化
            vllm_config,  # 配置
            executor_class,  # 执行器
            log_stats,  # 日志统计
            client_addresses,  # 地址
            client_count,  # 客户端数
            client_index,  # 客户端索引
        )

        assert len(self.core_engines) > 1  # 断言有多个引擎（否则无需负载均衡）

        self.eng_start_index = (
            len(self.core_engines) * self.client_index  # 起始索引
        ) // client_count
        # 计算本客户端的引擎扫描起始索引（支持多客户端负载均衡）

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        # 引擎按 rank 顺序排列
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            # 未指定数据并行 rank，且非晚期交互模型
            current_counts = self.lb_engines  # 当前引擎负载
            # TODO use P2C alg for larger DP sizes
            # TODO：较大 DP 规模时使用 P2C 算法
            num_engines = len(current_counts)  # 引擎数量
            min_score: float = sys.maxsize  # 最小评分初始化为最大值
            eng_index = 0  # 默认引擎 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                # 从 client_index 开始以帮助平衡（引擎为空时）。
                idx = (self.eng_start_index + i) % num_engines  # 轮询索引
                waiting, running, kv_cache_usage = current_counts[idx]
                # 获取引擎负载
                # Estimate engine load as the greater of the coordinator's
                # latest (waiting + running) snapshot and this client's own
                # in-flight count (scaled by the number of clients). The
                # in-flight floor is exact and can't be erased by a snapshot
                # rebind, so a burst spreads round-robin even when snapshots
                # race with routing decisions; the snapshot raises the score
                # when other clients or stale requests load the engine.
                # 估计引擎负载为协调器最新（等待+运行）快照与本客户端
                # 进行中计数（按客户端数量缩放）的较大值。
                # 进行中计数是精确的，不会被快照重绑定抹除，
                # 因此即使快照与路由决策竞争，突发流量也会轮询分布；
                # 当其他客户端或过期请求给引擎加载时，快照会提高评分。
                inflight = self.engine_inflight[self.core_engines[idx]]
                # 进行中的请求数
                score: float = max(self.client_count * inflight, waiting + running)
                # 评分 = 客户端数×进行中 与 等待+运行 的较大值
                if waiting:
                    # Waiting requests are penalized in proportion to KV cache
                    # pressure: a queue on a KV-bound engine drains slowly, so
                    # new requests should strongly prefer other engines. With
                    # low KV usage the queue is transient (e.g. mid-burst) and
                    # the penalty stays off, preserving exact round-robin.
                    # Ramps from 0 at <=50% usage to 3x waiting at 100%.
                    # 等待请求按 KV 缓存压力比例受罚：KV 受限引擎上的队列
                    # 排空缓慢，因此新请求应强烈偏好其他引擎。
                    # 低 KV 使用率时队列是暂时的（如突发中），
                    # 罚项保持关闭，保留精确轮询。
                    # 从 <=50% 使用率的 0 线性增加到 100% 的 3 倍等待。
                    score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
                    # 增加 KV 压力罚项
                if score < min_score:
                    # 如果评分更低
                    min_score = score  # 更新最小评分
                    eng_index = idx  # 选择该引擎
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            # 增加本地等待计数，以便在协调器统计更新（每 100ms）之间
            # 更好地平衡。
            current_counts[eng_index][0] += self.client_count
            # 增加本地计数
            # Rotate the scan start so that ties (equal scores, e.g. right
            # after a coordinator stats reset when engines look equally loaded)
            # don't systematically favor the same engine. This removes the
            # fixed tie-break bias without affecting load-aware decisions when
            # scores actually differ.
            # 轮换扫描起始位置，使平局（相同评分，如协调器统计重置后引擎
            # 看起来负载相同）不会系统性地偏向同一引擎。
            # 这消除了固定决胜偏见，而不影响评分实际不同时的负载感知决策。
            self.eng_start_index = (self.eng_start_index + 1) % num_engines
            # 轮换起始索引

        chosen_engine = self.core_engines[eng_index]  # 选择的引擎
        # Record which engine is chosen for this request, to handle aborts.
        # 记录请求选择的引擎，用于处理中止。
        self.reqs_in_flight[request.request_id] = chosen_engine  # 记录映射
        self.engine_inflight[chosen_engine] += 1  # 增加进行中计数
        return chosen_engine  # 返回选择的引擎

    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        # 只返回第一个引擎的结果。
        return (
            await asyncio.gather(
                # 并发调用所有引擎
                *[
                    self._call_utility_async(method, *args, engine=engine)  # 调用
                    for engine in self.core_engines  # 遍历所有引擎
                ]
            )
        )[0]  # 取第一个结果

    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        # 静态方法：处理引擎输出（DPLB 专用）
        if outputs.finished_requests and self.reqs_in_flight:
            # 如果有完成的请求且在途请求
            for req_id in outputs.finished_requests:
                # 遍历完成的请求
                if (engine := self.reqs_in_flight.pop(req_id, None)) is not None:
                    # 如果找到对应引擎
                    self.engine_inflight[engine] -= 1  # 减少进行中计数

    @staticmethod
    async def eep_process_engine_core_notification(
        self: "DPLBAsyncMPClient", notification_data: tuple[str, int]
    ):
        # 静态方法：处理弹性 EP 引擎核心通知（DPLB 专用）
        cache = self.eep_scaling_cache  # 弹性 EP 缓存
        notification_type_str, dp_rank = notification_data  # 解包通知
        try:
            notification_type = EEPNotificationType(notification_type_str)
            # 转换通知类型
        except ValueError as e:
            # 如果类型无效
            raise ValueError(
                f"Unknown EEP notification type: {notification_type_str}"
            ) from e

        if notification_type == EEPNotificationType.RECONFIGURE_FINISHED:
            # 如果是重配置完成通知
            from vllm.v1.engine import UtilityResult  # 延迟导入

            # NOTE(yongji): process a dummy UtilityOutput to resolve the future
            # awaited in _eep_wait_for_setup_switch_complete(), signaling that
            # all engine cores have completed reconfiguration.
            # 注意：处理虚拟 UtilityOutput 以解析 _eep_wait_for_setup_switch_complete()
            # 中等待的 Future，表示所有核心引擎已完成重配置。
            dummy_output = UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID, result=UtilityResult(None)
            )
            # 创建虚拟输出
            _process_utility_output(dummy_output, self.utility_results)
            # 处理虚拟输出（解析等待的 Future）
            return  # 返回
        assert cache is not None  # 断言缓存存在
        if notification_type not in cache.pending_notifications:
            # 如果该类型尚未有待处理通知
            cache.pending_notifications[notification_type] = set()
            # 创建集合
        if dp_rank in cache.pending_notifications[notification_type]:
            # 如果重复通知
            raise ValueError(
                f"Duplicate notification {notification_type} from dp_rank {dp_rank}"
            )
            # 抛出错误
        cache.pending_notifications[notification_type].add(dp_rank)
        # 添加通知
        if len(cache.pending_notifications[notification_type]) >= abs(
            cache.num_new_core_engines  # 新引擎数量绝对值
        ):
            # 如果收到所有必需的通知
            if notification_type == EEPNotificationType.SHUTDOWN_COMPLETE:
                # 如果是关闭完成通知（缩容场景）
                assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
                # 断言引擎管理器类型
                assert cache.num_new_core_engines < 0  # 断言是缩容
                old_dp_size = len(cache.existing_core_engines)  # 旧 DP 大小
                new_dp_size = old_dp_size + cache.num_new_core_engines  # 新 DP 大小
                self.resources.engine_manager.scale_down_elastic_ep(
                    old_dp_size, new_dp_size
                )
                # 执行弹性缩容
            else:
                await asyncio.gather(
                    # 并发通知所有现有引擎
                    *[
                        self._call_utility_async(
                            "eep_handle_engine_core_notification",  # 方法名
                            notification_type,  # 通知类型
                            engine=engine,  # 目标引擎
                        )
                        for engine in cache.existing_core_engines  # 遍历现有引擎
                    ]
                )
            cache.pending_notifications[notification_type] = set()
            # 重置待处理通知
            if notification_type in [
                EEPNotificationType.SHUTDOWN_COMPLETE,  # 关闭完成
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,
                # 新引擎权重初始化完成
            ]:
                self.eep_scaling_cache = None  # 清除缩放缓存

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # 异步中止请求（DPLB 版本：路由到正确引擎）
        if not request_ids or self.resources.engine_dead:
            # 如果没有请求或引擎已死亡
            return  # 直接返回

        if len(request_ids) == 1:
            # Fast-path common case.
            # 快路径：常见单请求情况
            if engine := self.reqs_in_flight.get(request_ids[0]):
                # 如果找到请求对应的引擎
                await self._abort_requests(request_ids, engine)
                # 发送给该引擎
            return  # 返回

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        # 按引擎分组的请求 ID
        for req_id in request_ids:
            # 遍历请求
            if engine := self.reqs_in_flight.get(req_id):
                # 如果找到引擎
                by_engine[engine].append(req_id)  # 分组
        for engine, req_ids in by_engine.items():
            # 遍历每组
            await self._abort_requests(req_ids, engine)  # 逐组发送

    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity  # 目标引擎
    ) -> None:
        # 向指定引擎发送中止请求
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)
        # 发送

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale elastic EP data parallel size"""
        # 弹性 EP 数据并行大小缩放
        cur_data_parallel_size = len(self.core_engines)  # 当前 DP 大小

        assert new_data_parallel_size != cur_data_parallel_size, (
            # 断言大小不同
            f"new_data_parallel_size {new_data_parallel_size} must be "
            f"different from cur_data_parallel_size {cur_data_parallel_size}"
        )
        self._prepared_elastic_ep = None

    async def prepare_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Prepare elastic EP scaling without routing requests to new engines."""
        if (prepared := self._prepared_elastic_ep) is not None:
            if prepared[0] == new_data_parallel_size:
                return
            raise RuntimeError("Elastic EP scaling is already prepared")
        cur_data_parallel_size = len(self.core_engines)
        assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (
            # 断言使用 Ray 后端
            "Only ray DP backend supports scaling elastic EP"
        )

        scale_up = new_data_parallel_size > cur_data_parallel_size  # 是否扩容

        if scale_up:
            # 扩容
            await self._scale_up_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )
            # 调用扩容方法
        else:
            # 缩容
            await self._scale_down_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )
            # 调用缩容方法

    def _eep_wait_for_setup_switch_complete(self) -> asyncio.Future:
        """
        Wait for core engines to switch to the new setup.

        In eep_process_engine_core_notification(), a dummy UtilityOutput with
        EEP_NOTIFICATION_CALL_ID will be set when RECONFIGURE_FINISHED
        notification is received from engine 0. We create a future with
        that call_id and wait for it to be resolved.
        """
        # 等待核心引擎切换到新配置。
        # 在 eep_process_engine_core_notification() 中，当从引擎 0 收到
        # RECONFIGURE_FINISHED 通知时，将设置带 EEP_NOTIFICATION_CALL_ID 的
        # 虚拟 UtilityOutput。我们创建该 call_id 的 Future 并等待它被解析。
        future = asyncio.get_running_loop().create_future()  # 创建 Future
        self.utility_results[EEP_NOTIFICATION_CALL_ID] = future  # 注册等待
        self._ensure_output_queue_task()  # 确保输出队列任务运行
        await future  # 等待完成

    def _setup_elastic_ep_reconfig_bootstrap(self) -> tuple[str, int]:
        # 设置弹性 EP 重配置引导
        from vllm.distributed.utils import create_tcp_store  # 延迟导入
        from vllm.utils.network_utils import get_open_ports_list  # 延迟导入

        parallel_config = self.vllm_config.parallel_config  # 并行配置
        parallel_config._data_parallel_master_port_list = get_open_ports_list(5)
        # 获取 5 个开放端口
        parallel_config.data_parallel_master_port = (
            parallel_config._data_parallel_master_port_list.pop()
        )
        # 弹出主端口

        ip = parallel_config.data_parallel_master_ip  # 主节点 IP
        store = create_tcp_store(
            # 创建 TCP 协调存储
            ip,  # IP
            0,  # 端口 0 = 自动
            is_master=True,  # 主节点
            world_size=-1,  # 动态
            wait_for_workers=False,  # 不等待
        )
        parallel_config._coord_store_port = store.port  # 记录端口
        self._coord_store = store  # 保存 store
        return ip, store.port  # 返回 (IP, 端口)

    async def _scale_up_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale up the data parallel size by creating new engine cores
        and reconfiguring existing ones."""
        # 通过创建新引擎核心并重配置现有引擎来扩展数据并行大小
        cur_data_parallel_size = len(self.core_engines)  # 当前 DP 大小

        self.eep_scaling_cache = ElasticScalingCache(  # 创建缩放缓存
            existing_core_engines=self.core_engines.copy(),  # 现有引擎
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            # 新引擎数
            pending_notifications=dict(),  # 待处理通知为空
        )

        parallel_config = self.vllm_config.parallel_config  # 并行配置
        ip, coord_store_port = self._setup_elastic_ep_reconfig_bootstrap()
        # 设置引导

        # Phase 1: Send reconfig messages to existing engines
        # 阶段 1：向现有引擎发送重配置消息
        reconfig_futures = []  # 重配置 Future 列表
        for engine in self.core_engines:
            # 遍历现有引擎
            reconfig_request = ReconfigureDistributedRequest(
                # 创建重配置请求
                new_data_parallel_size=new_data_parallel_size,  # 新 DP 大小
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                # 保持当前 rank
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                # 保持本地 rank
                new_data_parallel_master_ip=ip,  # 主节点 IP
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                # 主端口
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                # 端口列表
                coord_store_port=coord_store_port,  # 协调存储端口
            )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            # 调用引擎重配置方法
            reconfig_futures.append(asyncio.create_task(coro))  # 创建任务

        # Phase 2: Create new engines
        # 阶段 2：创建新引擎
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        # 断言是 actor 管理器
        parallel_config.eplb_config.num_redundant_experts = 0  # 重置冗余专家
        start_new_worker_future = asyncio.to_thread(
            # 在线程中启动新引擎（阻塞操作）
            self.resources.engine_manager.scale_up_elastic_ep,  # 扩容方法
            self.vllm_config,  # 配置
            new_data_parallel_size,  # 新大小
        )
        wait_future = self._eep_wait_for_setup_switch_complete()  # 等待切换完成

        # Phase 3: Wait for new engines to be created
        # and reconfig messages to be received
        # 阶段 3：等待新引擎创建和重配置消息接收
        await asyncio.gather(start_new_worker_future, *reconfig_futures)
        # 等待所有任务完成
        logger.info("[Elastic EP] Successfully started new engines")
        # 记录日志

        # Create new CoreEngine objects for the new engines
        # 为新引擎创建新的 CoreEngine 对象
        new_engine_identities = set()  # 新引擎身份集合
        for i in range(cur_data_parallel_size, new_data_parallel_size):
            # 遍历新引擎索引
            new_engine = i.to_bytes(2, "little")  # 转换为字节身份
            self.core_engines.append(new_engine)  # 添加到列表
            # NOTE(yongji): we don't update lb_engines here,
            # we let run_engine_stats_update_task to update it.
            # 注意：这里不更新 lb_engines，
            # 让 run_engine_stats_update_task 更新它。
            new_engine_identities.add(new_engine)  # 添加到集合

        # Wait for ready messages from new engines on the input socket
        # 在输入 socket 上等待新引擎的就绪消息
        sync_input_socket = zmq.Socket.shadow(self.input_socket)
        # 创建同步影子 socket
        while new_engine_identities:
            # 循环直到所有新引擎就绪
            if not sync_input_socket.poll(
                timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
            ):
                # 如果超时
                raise TimeoutError(
                    # 抛出超时错误
                    f"Timed out waiting for new engine core processes to "
                    f"start. Waited "
                    f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                    f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                    f"timeout, set the environment variable: "
                    f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                )
            identity, payload = sync_input_socket.recv_multipart()
            # 接收身份和负载
            new_engine_identities.discard(identity)  # 移除已就绪的引擎
            self._apply_ready_response(payload)  # 应用就绪响应

        # NOTE(yongji): Before we schedule any requests on the new workers,
        # we should wait for them to switch to the new setup.
        # 注意：在新 worker 上调度任何请求前，
        # 应等待它们切换到新配置。
        await wait_future  # 等待切换完成
        # Update the parallel config
        # 更新并行配置
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # 更新 DP 大小
        # Notify coordinator about scale up through existing
        # stats_update_task connection
        # 通过现有 stats_update_task 连接通知协调器扩缩容
        self._ensure_stats_update_task()  # 确保统计任务运行
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        # 编码扩缩容标记
        await self.first_req_send_socket.send(scale_up_marker)  # 发送通知

        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )
        # 记录扩容完成日志

    async def _prepare_scale_down_elastic_ep(self, new_data_parallel_size: int) -> None:
        self._setup_elastic_ep_reconfig_bootstrap()

        reconfig_futures = []
        for engine in self.core_engines[:new_data_parallel_size]:
            reconfig_request = self._make_reconfig_request(new_data_parallel_size)
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        ready_keys = await asyncio.gather(*reconfig_futures)
        await asyncio.to_thread(self._coord_store.wait, ready_keys)

    async def _commit_scale_down_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        # 通过关闭和重配置现有引擎核心来缩减数据并行大小
        cur_data_parallel_size = len(self.core_engines)  # 当前 DP 大小

        self.eep_scaling_cache = ElasticScalingCache(  # 创建缩放缓存
            existing_core_engines=self.core_engines.copy(),  # 现有引擎
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            # 新引擎数（负数 = 缩容）
            pending_notifications=dict(),  # 待处理通知
        )

        parallel_config = self.vllm_config.parallel_config  # 并行配置
        ip, coord_store_port = self._setup_elastic_ep_reconfig_bootstrap()
        # 设置引导

        removed_dp_size = cur_data_parallel_size - new_data_parallel_size
        # 移除的 DP 数量
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        # 断言是 actor 管理器
        self.resources.engine_manager.remove_run_refs_for_scale_down(removed_dp_size)
        # 移除被移除引擎的运行引用
        reconfig_futures = []  # 重配置 Future 列表
        for cur_dp_rank, engine in enumerate(self.core_engines):
            # 遍历所有引擎
            reconfig_request = ReconfigureDistributedRequest(
                # 创建重配置请求
                new_data_parallel_size=new_data_parallel_size,  # 新 DP 大小
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                # 默认保持 rank
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                # 保持本地 rank
                new_data_parallel_master_ip=ip,  # 主节点 IP
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                # 主端口
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                # 端口列表
                coord_store_port=coord_store_port,  # 协调存储端口
            )
            if cur_dp_rank >= new_data_parallel_size:
                # 如果当前 rank 超出新大小（需要关闭）
                reconfig_request.new_data_parallel_rank = (
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK
                )
                # 标记关闭当前 rank
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            # 调用引擎重配置
            reconfig_futures.append(asyncio.create_task(coro))  # 创建任务

        # NOTE(yongji): Immediately stop sending requests to the removing engines.
        # 注意：立即停止向被移除的引擎发送请求。
        self.core_engines = self.core_engines[:new_data_parallel_size]
        # 截断引擎列表
        self.lb_engines = self.lb_engines[:new_data_parallel_size]
        # 截断负载均衡列表
        wait_future = self._eep_wait_for_setup_switch_complete()  # 等待切换完成

        await asyncio.gather(*reconfig_futures)  # 等待所有重配置完成

        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # 更新 DP 大小
        self._ensure_stats_update_task()  # 确保统计任务运行
        scale_down_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        # 编码缩容标记
        await self.first_req_send_socket.send(scale_down_marker)  # 发送通知

        # NOTE(yongji): Unlike scaling up,
        # here we don't actually need to wait for the setup switch to complete.
        # We may want to remove it in the future.
        # 注意：与扩容不同，
        # 这里实际上不需要等待配置切换完成。
        # 未来可能需要移除这段等待。
        await wait_future  # 等待切换完成
        logger.info(
            "[Elastic EP
