# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/uniproc_executor.py
# 本文件实现「单进程执行器」UniProcExecutor：
#   - 不创建子进程 / Ray actor，而是在主进程内直接驱动一个 Worker。
#   - 适用于单 GPU（TP=1）或无需跨进程并行的场景，是开销最小的后端。
# 文件同时定义 ExecutorWithExternalLauncher：为 torchrun 等外部启动器设计，
# 允许启动多个独立 vLLM 引擎协同处理同一批请求（外部 TP 场景）。
# =============================================================================
import os
# 导入 os 模块：读取 RANK / LOCAL_RANK 等环境变量（外部 launcher 场景）。
from collections.abc import Callable
# 导入 Callable 类型，用于类型标注 collective_rpc 的 method 参数。
from concurrent.futures import Future
# 导入 Future：non_block 模式下返回异步结果句柄。
from multiprocessing import Lock
# 导入 multiprocessing.Lock：创建进程间共享锁，保护 worker 内共享资源
# （如模型加载、CUDA context 初始化避免竟态）。
from typing import Any
# 导入 Any 类型，用于宽松标注 RPC 返回值的动态类型。

import torch
# 导入 torch：外部 launcher 场景下做跨 rank 显存最小值 all_reduce。
import torch.distributed as dist
# 导入 torch.distributed：使用 CPU 组做 all_reduce（determine_available_memory）。

import vllm.envs as envs
# 导入 vllm 环境变量模块：读取 VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 等开关。
from vllm.logger import init_logger
# 导入日志初始化函数，创建本模块 logger。
from vllm.platforms import current_platform
# 导入当前平台抽象：根据后端（CUDA/CPU 等）更新 block size 等。
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
# 导入网络工具：
#   get_distributed_init_method —— 生成 torch.distributed 初始化地址（tcp://ip:port）；
#   get_ip —— 获取本机 IP；
#   get_open_port —— 获取空闲端口，用于建立分布式通信。
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
# 导入调度器相关输出类型（与 abstract.py 中一致）。
from vllm.v1.executor.abstract import Executor
# 导入抽象基类 Executor，UniProcExecutor 继承它。
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
# 导入 GPU→NIC 映射工具：为 worker 设置 RDMA 网卡相关环境变量。
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
# 导入输出类型：
#   ModelRunnerOutput —— 常规模型执行输出；
#   AsyncModelRunnerOutput —— 异步调度下的模型输出（内部持有 CUDA event，需 get_output() 取结果）；
#   DraftTokenIds —— 投机解码的草稿 token。
from vllm.v1.serial_utils import run_method
# 导入 run_method：在对象上执行"按字符串方法名或可调用对象"的统一调用工具，
# 是 UniProc 下 collective_rpc 的核心执行器。
from vllm.v1.worker.worker_base import WorkerWrapperBase
# 导入 WorkerWrapperBase：worker 的包装基类（单进程下即 driver_worker）。

logger = init_logger(__name__)
# 初始化本模块日志。


class AsyncOutputFuture(Future):
    # =========================================================================
    # AsyncOutputFuture：包装 AsyncModelRunnerOutput 的 Future 子类。
    # 当 worker 返回异步输出时，把「取结果」延迟到用户调用 .result() 时才执行，
    # 使得调用方可以用统一的 Future 接口消费异步调度结果。
    # =========================================================================
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        # 构造函数。
        self.async_output = async_output
        # 保存异步模型输出对象（内部持有 CUDA event / 尚未同步的输出数据）。
        self.single_value = single_value
        # 标记 result() 返回时是返回单个值还是包装成列表
        #（对应 collective_rpc 的 single_value 参数）。
        super().__init__()
        # 调用父类 Future 的构造（初始状态为 pending）。

    def result(self, timeout=None):
        # -------------------------------------------------------------------
        # 重写 Future.result()：首次调用时真正从 async_output 中取出结果。
        # -------------------------------------------------------------------
        if timeout is not None:
            # 本实现不支持超时。
            raise RuntimeError("timeout not implemented")
            # 传入 timeout 直接抛错。

        if not super().done():
            # 若 Future 尚未完成（从未取过结果）。
            try:
                output = self.async_output.get_output()
                # 调用异步输出的 get_output()：此时会同步 CUDA 事件、拷贝输出。
                self.set_result(output if self.single_value else [output])
                # 根据 single_value 决定把结果包装为单个值还是列表，并设置到 Future。
            except Exception as e:
                self.set_exception(e)
                # 获取输出过程中的异常也存入 Future，供调用方通过 result() 抛出。
        return super().result()
        # 返回父类 Future 的结果（已完成，直接取缓存值）。


class UniProcExecutor(Executor):
    # =========================================================================
    # UniProcExecutor：单进程执行器。
    # 所有 worker 调用都在主进程内完成，无进程创建开销。
    # =========================================================================
    def _init_executor(self) -> None:
        # -------------------------------------------------------------------
        # 初始化 executor：创建 worker、初始化设备、加载模型。
        # 这是父类抽象方法 _init_executor 的实现。
        # -------------------------------------------------------------------
        """Initialize the worker and load the model."""
        # 文档字符串：初始化 worker 并加载模型。
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        # 创建唯一的 driver worker（rpc_rank=0，即全局第一个 worker）。
        distributed_init_method, rank, local_rank = self._distributed_args()
        # 计算分布式初始化地址、全局 rank、本地 rank。
        kwargs = dict(
            vllm_config=self.vllm_config,
            # 传入完整配置。
            local_rank=local_rank,
            # 本地 rank（设备索引）。
            rank=rank,
            # 全局 rank。
            distributed_init_method=distributed_init_method,
            # 分布式初始化方法（tcp://...）。
            is_driver_worker=True,
            # 单进程下此 worker 必然是 driver worker。
            shared_worker_lock=Lock(),
            # 创建进程级共享锁（保证 CUDA 初始化等操作的互斥）。
        )
        # 组装 worker 初始化所需的全部关键字参数。

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        # 若设置了 VLLM_GPU_NIC_PCIE_MAPPING，则为 worker 设置正确的网卡环境变量。
        set_worker_net_device(local_rank, self.vllm_config)
        # 调用 GPU→NIC 映射工具（RDMA 场景需要）。

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        # 初始化 worker：建立统一参数列表（单进程只有一个元素），
        # 内部会据此初始化模型并行组等。
        self.driver_worker.init_device()
        # 初始化设备（设置当前 CUDA 设备、分配显存池等）。

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            # 若启用弹性专家并行（EP）扩容启动模式。
            self.driver_worker.elastic_ep_execute("load_model")
            # 走弹性 EP 专用的模型加载路径（需要跨 worker 协调专家放置）。
        else:
            self.driver_worker.load_model()
            # 常规路径：直接加载模型权重。
        current_platform.update_block_size_for_backend(self.vllm_config)
        # 根据当前平台与注意力后端，更新 vLLM 的 block size 配置。

    def _distributed_args(self) -> tuple[str, int, int]:
        # -------------------------------------------------------------------
        # 计算分布式参数：(distributed_init_method, rank, local_rank)。
        # 单进程模式：rank 固定为 0；local_rank 取设备配置中的索引。
        # -------------------------------------------------------------------
        """Return (distributed_init_method, rank, local_rank)."""
        # 文档字符串：返回 (分布式初始化方法, rank, local_rank)。
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # 用本机 IP + 空闲端口生成 tcp:// 初始化地址。
        # set local rank as the device index if specified
        # 如果指定了设备索引，则 local_rank 取该索引。
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        # 设备字符串形如 "cuda:0"，按冒号拆分。
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        # 有冒号则取数字部分为 local_rank，否则默认为 0。
        return distributed_init_method, 0, local_rank
        # 返回：单进程全局 rank 恒为 0。

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        # method：worker 方法名，或可调用对象。
        timeout: float | None = None,
        # timeout：支持超时（本实现未实际使用，仅保持接口兼容）。
        args: tuple = (),
        # args：传给方法的位置参数。
        kwargs: dict | None = None,
        # kwargs：传给方法的关键字参数。
        non_block: bool = False,
        # non_block：True 时返回 Future。
        single_value: bool = False,
        # single_value：True 时返回单个结果（而非 list），本类新增的扩展参数。
    ) -> Any:
        # -------------------------------------------------------------------
        # 实现集体 RPC：单进程下直接调用 driver_worker 上对应方法。
        # type: ignore[override]：参数比父类多了 single_value，覆盖类型检查告警。
        # -------------------------------------------------------------------
        if kwargs is None:
            kwargs = {}
            # 将 None 归一化为空字典。

        if not non_block:
            # 同步模式。
            result = run_method(self.driver_worker, method, args, kwargs)
            # 在 driver_worker 上执行方法（字符串名则 getattr，可调用则调用）。
            if isinstance(result, AsyncModelRunnerOutput):
                # 若返回的是异步模型输出对象。
                result = result.get_output()
                # 立即同步取出真正输出（同步模式下调用方需要最终值）。
            return result if single_value else [result]
            # 按 single_value 决定返回单个结果还是列表包装。

        try:
            # 异步模式。
            result = run_method(self.driver_worker, method, args, kwargs)
            # 同样先执行方法（UniProc 是同步执行，但包装成 Future 返回）。
            if isinstance(result, AsyncModelRunnerOutput):
                # 若结果为异步输出对象。
                return AsyncOutputFuture(result, single_value)
                # 包装为 AsyncOutputFuture，延迟到 .result() 时再取最终值。
            future = Future[Any]()
            # 否则创建普通 Future。
            future.set_result(result if single_value else [result])
            # 立即设置结果（已执行完）。
        except Exception as e:
            future = Future[Any]()
            # 捕获执行异常。
            future.set_exception(e)
            # 将异常存入 Future，让调用方在 result() 时收到。
        return future
        # 返回 Future（成功或异常均已设置）。

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # 执行一轮模型推理（数据平面核心入口）。
        # type: ignore[override]：本实现额外使用 single_value=True 语义。
        # -------------------------------------------------------------------
        output = self.collective_rpc(
            "execute_model",
            # 调用 driver_worker 的 execute_model。
            args=(scheduler_output,),
            # 入参为调度输出。
            non_block=non_block,
            # 透传非阻塞标志。
            single_value=True,
            # 单进程只返回单个结果。
        )
        # In non-blocking mode, surface any exception as early as possible.
        # 非阻塞模式下尽早暴露异常：若任务已经完成，立刻检查并抛出失败。
        if non_block and output.done():
            # 异步模式且 Future 已完成。
            # Raise the exception in-line if the task failed.
            # 若任务失败则立即抛异常（而不是等调用方）。
            output.result()
            # 触发内部异常抛出。
        return output
        # 返回输出或 Future。

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # 基于模型 logits 做采样（配 execute_model 使用）。
        # -------------------------------------------------------------------
        return self.collective_rpc(
            "sample_tokens",
            # 调用 driver_worker 的 sample_tokens。
            args=(grammar_output,),
            # 入参为 grammar（结构化输出约束）。
            non_block=non_block,
            # 透传非阻塞标志。
            single_value=True,
            # 返回单个结果。
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # -------------------------------------------------------------------
        # 获取投机解码草稿 token。
        # -------------------------------------------------------------------
        return self.collective_rpc("take_draft_token_ids", single_value=True)
        # 调用 driver_worker，返回单个草稿 token 结果。

    def check_health(self) -> None:
        # -------------------------------------------------------------------
        # 健康检查：单进程下进程活着即健康，恒返回。
        # -------------------------------------------------------------------
        # UniProcExecutor will always be healthy as long as
        # it's running.
        # 注释：只要进程在运行，UniProcExecutor 就永远是健康的。
        return
        # 直接返回，不做额外检查。

    def shutdown(self) -> None:
        # -------------------------------------------------------------------
        # 关闭 executor：优雅关闭 driver_worker。
        # -------------------------------------------------------------------
        if worker := self.driver_worker:
            # 海象运算符：若 worker 非空。
            worker.shutdown()
            # 调用 worker 的关闭逻辑。

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # -------------------------------------------------------------------
        # 本类支持异步调度（配合 AsyncOutputFuture 实现）。
        # -------------------------------------------------------------------
        return True
        # 返回 True。


class ExecutorWithExternalLauncher(UniProcExecutor):
    # =========================================================================
    # ExecutorWithExternalLauncher：使用外部 launcher 的执行器。
    # 专为 torchrun 兼容 launcher 设计，用于离线 TP 推理：
    #   每个 vLLM 引擎只创建一个 worker，但用户会启动多个引擎进程协同处理
    #   相同 prompt；在确定性调度下各引擎产生相同输出，无需互相同步状态。
    # =========================================================================
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/features/torchrun/torchrun_example_offline.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """
    # 类文档字符串：解释设计动机与用法——
    # 虽然是 TP 推理，但每个 executor 只创建一个 worker；用户用 torchrun
    # 启动多个引擎，它们共同处理相同的 prompt。调度确定性时各引擎输出一致，
    # 因此无需互相同步状态。

    def _init_executor(self) -> None:
        # -------------------------------------------------------------------
        # 初始化 executor：要求关闭 v1 多进程模式以保证确定性。
        # -------------------------------------------------------------------
        """Initialize the worker and load the model."""
        # 文档字符串：初始化 worker 并加载模型。
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        # 断言未启用 v1 多进程模式，否则无法保证多引擎输出确定性。
        super()._init_executor()
        # 复用父类（UniProcExecutor）的初始化逻辑。

    def _distributed_args(self) -> tuple[str, int, int]:
        # -------------------------------------------------------------------
        # 分布式参数：从 torchrun 注入的环境变量中读取 rank 与地址。
        # -------------------------------------------------------------------
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        # 注释：引擎由 torchrun 兼容 launcher 启动，因此可用 env:// 方法；
        # 需要环境变量 RANK / LOCAL_RANK / MASTER_ADDR / MASTER_PORT。
        distributed_init_method = "env://"
        # 使用 torchrun 的标准环境变量初始化方式。
        rank = int(os.environ["RANK"])
        # 从环境变量读取全局 rank。
        local_rank = int(os.environ["LOCAL_RANK"])
        # 从环境变量读取本地 rank。
        return distributed_init_method, rank, local_rank
        # 返回三者。

    def determine_available_memory(self) -> list[int]:  # in bytes
        # -------------------------------------------------------------------
        # 可用显存：多引擎协同需要取所有 rank 的最小值，
        # 保证各引擎的 KV cache 分配一致。
        # -------------------------------------------------------------------
        # we need to get the min across all ranks.
        # 注释：需要取所有 rank 的最小值。
        memory = super().determine_available_memory()
        # 先取本 rank 的可用显存（父类实现）。
        from vllm.distributed.parallel_state import get_world_group
        # 延迟导入全局并行组访问器。

        cpu_group = get_world_group().cpu_group
        # 获取 CPU 通信组（跨所有外部引擎进程）。
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        # 把本 rank 的显存封装为 CPU 上的 int64 张量（all_reduce 输入）。
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        # 在 CPU 组内做 MIN 归约，得到所有 rank 的最小显存。
        return [memory_tensor.item()]
        # 返回最小值（列表形式，与父类接口一致）。