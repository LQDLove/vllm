# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# DBO(Double-Buffering Overlap)微批调度上下文。
# 在 micro-batch 粒度上重叠通信与计算:通过线程事件与 CUDA 事件同步,
# 切换计算/通信 CUDA 流,并配合前向上下文管理,隐藏集体通信延迟。

# 导入 threading,用于线程事件/屏障同步微批线程。
import threading

# 导入 PyTorch,用于 CUDA 流与事件。
import torch

# 导入 vllm.forward_context 模块(全局前向上下文)。
from vllm import forward_context
# 导入 ForwardContext 类型,微批上下文保存各自的前向上下文。
from vllm.forward_context import ForwardContext
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 current_stream,用于查询当前 CUDA 流。
from vllm.utils.torch_utils import current_stream

# 创建本模块的日志记录器。
logger = init_logger(__name__)

# 线程 id -> 当前微批 id 的映射(仅 DBO 活跃线程有记录)。
_THREAD_ID_TO_CONTEXT: dict = {}
# Here we hardcode the number of microbatches to 2 for default.
# 此处默认把微批数硬编码为 2(可在 make_ubatch_contexts 中覆盖)。
_NUM_UBATCHES: int = 2
# 当前活跃的微批上下文列表(下标为微批 id,元素可为 None)。
_CURRENT_CONTEXTS: list["UBatchContext | None"] = []


class UBatchContext:
    """
    Context manager for micro-batching synchronization using threading events.
    """
    # 微批同步上下文管理器:使用线程事件与 CUDA 流/事件实现微批调度。

    def __init__(
        self,
        id: int,
        comm_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        gpu_comm_done_event: torch.Event,
        gpu_compute_done_event: torch.Event,
        schedule: str = "default",
    ):
        # 初始化微批上下文。
        # 参数:
        #   id: 微批编号。
        #   comm_stream: 通信流。
        #   compute_stream: 计算流。
        #   forward_context: 该微批的前向上下文。
        #   ready_barrier: 线程就绪屏障(所有微批线程到齐后同时开始)。
        #   cpu_wait_event: 本微批等待的事件(CPU 同步)。
        #   cpu_signal_event: 本微批通知其它微批的事件(CPU 同步)。
        #   gpu_comm_done_event: 通信完成 CUDA 事件。
        #   gpu_compute_done_event: 计算完成 CUDA 事件。
        #   schedule: 调度策略名(默认 "default")。
        # 记录微批 id。
        self.id = id
        # 记录通信流。
        self.comm_stream = comm_stream
        # 记录计算流。
        self.compute_stream = compute_stream
        # 记录本微批的前向上下文。
        self.forward_context = forward_context
        # 记录就绪屏障。
        self.ready_barrier = ready_barrier
        # 记录 CPU 等待事件。
        self.cpu_wait_event = cpu_wait_event
        # 记录 CPU 信号事件。
        self.cpu_signal_event = cpu_signal_event
        # 当前流初始为计算流。
        self.current_stream = compute_stream
        # 记录通信完成事件。
        self.gpu_comm_done_event = gpu_comm_done_event
        # 记录计算完成事件。
        self.gpu_compute_done_event = gpu_compute_done_event
        # 记录调度策略。
        self.schedule = schedule
        # 接收钩子(用于跨 rank 数据就绪通知),初始为 None。
        self.recv_hook = None

    def __enter__(self):
        # 进入微批上下文:注册线程、等待就绪屏障,然后切换初始流。
        # 引用全局的当前上下文列表与线程映射。
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        # 注册当前线程 -> 本微批 id。
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        # 把本微批上下文放入当前上下文列表。
        _CURRENT_CONTEXTS[self.id] = self
        # 等待所有微批线程到达就绪屏障(保证同步启动)。
        self.ready_barrier.wait()

        # 等待 CPU 等待事件(前一个微批放行)。
        self.cpu_wait_event.wait()
        # 清除该事件。
        self.cpu_wait_event.clear()
        # 恢复本微批的前向上下文。
        self._restore_context()
        # Assume we want to start on the compute stream
        # 假定从计算流开始执行。
        self.update_stream(self.compute_stream)
        # 返回自身支持 with 语句。
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出微批上下文:清理注册,运行接收钩子,通知下一个微批。
        # 引用全局变量。
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        # 把本微批从当前上下文列表清空。
        _CURRENT_CONTEXTS[self.id] = None
        # 移除当前线程的微批映射。
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        # 若注册了接收钩子,运行之。
        self.maybe_run_recv_hook()
        # 置位 CPU 信号事件,通知下一个微批可以继续。
        self.cpu_signal_event.set()
        # 清空自己的等待事件(避免残留)。
        self.cpu_wait_event.clear()
        # 返回 False:不吞掉异常。
        return False

    def _restore_context(self):
        # 把全局前向上下文恢复为本微批的上下文。
        forward_context._forward_context = self.forward_context

    def update_stream(self, stream):
        # 更新当前流:记录并切到新流(若与当前流不同)。
        self.current_stream = stream
        # 若 CUDA 当前流不是目标流:
        if current_stream() != self.current_stream:
            # 切换 CUDA 当前流。
            torch.cuda.set_stream(self.current_stream)

    def _signal_comm_done(self):
        # 在通信流上记录通信完成事件。
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        # 在计算流上记录计算完成事件。
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        # 让通信流等待计算完成事件。
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        # 让计算流等待通信完成事件。
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self):
        # CPU 让出:通知下一个微批线程运行,然后本线程挂起等待。
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        # 说明:一次只能有一个线程运行,这对正确性至关重要。以下断言
        # 确保在唤醒其它线程并挂起前,当前是唯一运行的线程。
        # 断言全局前向上下文仍是本微批的(未被其它线程占用)。
        assert forward_context._forward_context == self.forward_context
        # 断言当前 CUDA 流仍是本微批记录的流。
        assert current_stream() == self.current_stream
        # 断言等待事件未被设置(避免状态错乱)。
        assert not self.cpu_wait_event.is_set()

        # 置位信号事件,允许下一个微批线程运行。
        self.cpu_signal_event.set()
        # 挂起并等待被唤回。
        self.cpu_wait_event.wait()
        # 清除等待事件。
        self.cpu_wait_event.clear()
        # 恢复本微批的前向上下文。
        self._restore_context()

    def switch_to_comm(self):
        # 切换到通信流(异步)。
        self.update_stream(self.comm_stream)

    def switch_to_compute(self):
        # 切换到计算流(异步)。
        self.update_stream(self.compute_stream)

    def switch_to_comm_sync(self):
        # 同步切换到通信流:先记录计算完成事件,切换流,再等待计算完成。
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self):
        # 同步切换到计算流:先记录通信完成事件,切换流,再等待通信完成。
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def maybe_run_recv_hook(self):
        # 若注册了接收钩子,运行一次并清除。
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    def yield_(self):
        # 记录当前流,执行 CPU 让出,恢复原流。
        self.current_stream = current_stream()
        self._cpu_yield()
        self.update_stream(self.current_stream)

    def yield_and_switch_from_compute_to_comm(self):
        # 从计算阶段让出并切到通信阶段(同步)。
        # 断言当前在计算流上。
        assert current_stream() == self.compute_stream
        # 记录计算完成事件。
        self._signal_compute_done()
        # CPU 让出。
        self._cpu_yield()
        # 断言让出后仍记录为计算流。
        assert self.current_stream == self.compute_stream
        # 切换到通信流。
        self.update_stream(self.comm_stream)
        # 等待计算完成。
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        # 从通信阶段让出并切到计算阶段(同步)。
        # 断言当前在通信流上。
        assert current_stream() == self.comm_stream
        # 记录通信完成事件。
        self._signal_comm_done()
        # CPU 让出。
        self._cpu_yield()
        # 断言让出后仍记录为通信流。
        assert self.current_stream == self.comm_stream
        # 切换到计算流。
        self.update_stream(self.compute_stream)
        # 等待通信完成。
        self._wait_comm_done()


def dbo_enabled() -> bool:
    # 查询 DBO 是否启用:当前线程是否在微批上下文内。
    return len(_THREAD_ID_TO_CONTEXT) > 0


def dbo_current_ubatch_id() -> int:
    # 获取当前线程所属的微批 id。
    # 若当前不在微批上下文内,返回 0。
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    # 返回当前线程对应的微批 id。
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


def _register_ubatch_function(func):
    # 把 UBatchContext 的方法包装为 DBO 感知的模块级函数:
    # 调用时若当前线程在微批上下文内,则用当前上下文调用原方法。
    def wrapper(*args, **kwargs):
        # 若当前线程在微批上下文内:
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            # 取当前线程的微批 id。
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            # 取对应的微批上下文。
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            # 以该上下文为 self 调用原方法。
            func(ctx, *args, **kwargs)

    # 返回包装函数。
    return wrapper


# 注册模块级 DBO 辅助函数(在微批上下文内运行对应方法):
# 运行接收钩子。
dbo_maybe_run_recv_hook = _register_ubatch_function(UBatchContext.maybe_run_recv_hook)
# 执行让出。
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
# 从计算让出并切到通信。
dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_compute_to_comm
)
# 从通信让出并切到计算。
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_comm_to_compute
)
# 切到通信流。
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
# 切到计算流。
dbo_switch_to_compute = _register_ubatch_function(UBatchContext.switch_to_compute)
# 同步切到通信流。
dbo_switch_to_comm_sync = _register_ubatch_function(UBatchContext.switch_to_comm_sync)
# 同步切到计算流。
dbo_switch_to_compute_sync = _register_ubatch_function(
    UBatchContext.switch_to_compute_sync
)


def dbo_register_recv_hook(recv_hook):
    # 在下一个微批上注册接收钩子(跨 rank 数据就绪时触发)。
    # 若当前线程在微批上下文内:
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        # 取当前线程的微批 id。
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        # 取下一个微批上下文(循环取模)。
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
        # 把钩子注册到下一个微批。
        next_ctx.recv_hook = recv_hook


def dbo_get_previous_event(func, *args, **kwargs):
    # 在当前微批的计算流上执行 func(用于记录/等待事件),并返回结果。
    # 若当前线程在微批上下文内:
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        # 取当前线程的微批 id。
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        # 取对应的微批上下文。
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        # execute callable on the ubatch compute stream to record/wait events there
        # 在微批计算流上执行可调用对象,以在该流上记录/等待事件。
        with torch.cuda.stream(ctx.compute_stream):
            # 在计算流上下文内调用 func。
            return func(*args, **kwargs)


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[UBatchContext]:
    # 创建全部微批的 UBatchContext 列表。
    # 参数:
    #   num_micro_batches: 微批数量(必须 > 1)。
    #   compute_stream: 计算流。
    #   comm_stream: 通信流。
    #   forward_contexts: 各微批的前向上下文列表。
    #   ready_barrier: 线程就绪屏障。
    #   schedule: 调度策略。
    # 引用全局微批数与当前上下文列表。
    global _NUM_UBATCHES, _CURRENT_CONTEXTS
    # 断言微批数 > 1(DBO 需要至少双缓冲)。
    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"

    # 更新全局微批数。
    _NUM_UBATCHES = num_micro_batches
    # Ensure the global context list is large enough
    # 确保全局上下文列表长度足够:
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        # 用 None 扩展列表至所需长度。
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    # 为每个微批创建 CPU 线程事件(用于线程间同步)。
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    # 为每个微批创建通信完成 CUDA 事件。
    gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
    # 为每个微批创建计算完成 CUDA 事件。
    gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]

    # 初始化微批上下文列表。
    ctxs = []
    # 遍历每个微批:
    for i in range(num_micro_batches):
        # 创建微批上下文:
        ctx = UBatchContext(
            # 微批编号。
            id=i,
            # 计算流。
            compute_stream=compute_stream,
            # 通信流。
            comm_stream=comm_stream,
            # 该微批的前向上下文。
            forward_context=forward_contexts[i],
            # 就绪屏障。
            ready_barrier=ready_barrier,
            # 本微批等待的事件。
            cpu_wait_event=cpu_events[i],
            # 信号事件指向下一个微批(循环),用于接力放行。
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            # 通信完成事件。
            gpu_comm_done_event=gpu_comm_done_events[i],
            # 计算完成事件。
            gpu_compute_done_event=gpu_compute_done_events[i],
            # 调度策略。
            schedule=schedule,
        )
        # 加入列表。
        ctxs.append(ctx)

    # 返回全部微批上下文。
    return ctxs