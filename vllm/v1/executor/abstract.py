# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/abstract.py
# 本文件定义 vLLM v1 引擎中 Executor（执行器）的抽象基类。
# Executor 位于 Engine（调度侧）与 Worker（设备侧）之间，负责：
#   1. 统一向底层一个或多个 worker 广播 RPC 调用（控制平面）。
#   2. 统一执行模型 forward / 采样（数据平面）。
#   3. 根据分布式后端（ray / mp / uni / external_launcher）做工厂分发。
# =============================================================================
import time
# 导入 time 模块，用于 sleep()/wake_up() 中测量耗时（性能日志）。
from abc import ABC, abstractmethod
# 从 abc 导入 ABC 抽象基类与 abstractmethod 抽象方法装饰器，用于定义接口骨架。
from collections.abc import Callable
# 导入 Callable 类型，标注回调函数（如 FailureCallback）与可调用方法参数。
from concurrent.futures import Future
# 导入 Future 类型，用于非阻塞（non_block）模式下返回异步结果句柄。
from functools import cached_property
# 导入 cached_property 装饰器，将只读属性缓存以避免重复 RPC 调用。
from typing import TYPE_CHECKING, Literal, TypeVar, overload
# 导入类型工具：
#   TYPE_CHECKING —— 仅类型检查阶段的导入保护，避免循环导入；
#   Literal —— 精确字面量类型（区分 non_block 的 True/False 重载）；
#   TypeVar —— 泛型类型变量（collective_rpc 的返回值泛型化）；
#   overload —— 函数重载声明，仅提供类型信息，不改变运行时行为。

import vllm.envs as envs
# 导入 vllm 的环境变量模块。用于读取 VLLM_USE_RAY_V2_EXECUTOR_BACKEND 等开关。
from vllm.config import VllmConfig
# 导入 VllmConfig（聚合所有子配置的顶层配置对象），Executor 构造函数的唯一入参。
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
# 导入 KVOutputAggregator：KV 迁移（disaggregated serving）场景下，
# 聚合所有 worker 的模型输出结果的工具类。
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
# 导入 KVConnectorHandshakeMetadata：KV 连接器握手元数据，用于跨引擎 KV 传输前的协商。
from vllm.logger import init_logger
# 导入日志初始化函数，创建本模块的 logger 实例。
from vllm.lora.request import LoRARequest
# 导入 LoRARequest：LoRA 适配器的请求数据结构（add/remove lora 时传递）。
from vllm.tasks import SupportedTask
# 导入 SupportedTask：任务类型枚举（generate / embed / classify / reward 等），
# 用于查询 executor 支持的推理任务集合。
from vllm.tracing import instrument
# 导入 instrument 装饰器，用于 OpenTelemetry 链路追踪（span 埋点）。
from vllm.utils.import_utils import resolve_obj_by_qualname
# 导入按限定名（字符串）解析对象的工具，
# 用于支持用户自定义 executor 后端（distributed_executor_backend 传类路径字符串）。
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
# 导入调度器输出类型：
#   SchedulerOutput —— 一轮调度决策的完整输出（哪些请求被调度、KV 分配等）；
#   GrammarOutput —— 结构化输出（grammar 约束）的位掩码输出。
from vllm.v1.engine import ReconfigureDistributedRequest
# 导入 ReconfigureDistributedRequest：分布式重配置请求，
# 用于 DP（数据并行）弹性扩缩容时通知 worker 调整并行秩。
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
# 导入 KV Cache 相关类型：
#   KVCacheConfig —— KV cache 配置（各层/张量并行分片的分配参数）；
#   KVCacheSpec —— 每一层 KV cache 的规格描述。
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
# 导入模型执行输出类型：
#   ModelRunnerOutput —— ModelRunner 一轮执行的输出（hidden states 采样结果）；
#   DraftTokenIds —— 投机解码中草稿模型（draft model）产生的候选 token。
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
# 导入 Worker 侧类型：
#   WorkerBase —— 底层 worker 的抽象基类（executor 的 RPC 目标）；
#   CompilationTimes —— 模型编译耗时记录（语言模型与编码器分别统计），
#                       供主进程汇总传播编译性能数据。

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
# 仅在进行类型检查时导入 KVConnectorBase，避免运行时循环导入；
# 该类型在 init_kv_output_aggregator() 中用作类型标注。

logger = init_logger(__name__)
# 初始化本模块（vllm.v1.executor.abstract）的日志记录器。

_R = TypeVar("_R")
# 定义泛型类型变量 _R，使 collective_rpc 的返回值可泛化为任意 worker 返回类型。

FailureCallback = Callable[[], None]
# 类型别名：失败回调函数。当 worker 异常退出导致 executor 进入永久失败态时被调用，
# 无参数、无返回值，通常用于通知上层引擎主动触发失败关闭流程。


class Executor(ABC):
    # =========================================================================
    # 抽象基类 Executor：所有执行器（单进程/多进程/Ray）的统一接口。
    # =========================================================================
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """
    # 类文档字符串：说明 executor 的职责——在单个设备上执行模型，
    # 或是可跨多设备执行的分布式执行器。

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    # 类属性：标记该 executor 是否使用 Ray 做分布式编排（RayDistributedExecutor=True）。
    supports_pp: bool = False  # whether the executor supports PP
    # 类属性：标记该 executor 是否支持流水线并行（Pipeline Parallelism）。

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        # -------------------------------------------------------------------
        # 静态工厂方法：根据配置中的分布式后端字符串，解析出对应的 Executor 子类。
        # 这是 vLLM 支持可插拔分布式后端的关键入口。
        # -------------------------------------------------------------------
        executor_class: type[Executor]
        # 声明局部变量：最终选中的 executor 类。
        parallel_config = vllm_config.parallel_config
        # 取出并行配置对象（含 distributed_executor_backend 字段）。
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # 取出分布式后端标识：可以是类型对象 / 字符串（"ray"/"mp"/"uni"/...）。
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        # （VllmConfig.__post_init__ 中保证该字段一定有值。）
        if isinstance(distributed_executor_backend, type):
            # 情形 1：用户直接传入一个类型（自定义 executor 类）。
            if not issubclass(distributed_executor_backend, Executor):
                # 校验该类型必须是 Executor 的子类。
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
                # 否则抛出 TypeError，明确提示必须是 Executor 子类。
            executor_class = distributed_executor_backend
            # 校验通过后直接使用该自定义 executor 类。
        elif distributed_executor_backend == "ray":
            # 情形 2a：后端为 "ray"。
            if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                # 若设置了环境变量 VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1，
                # 则选用新版 Ray 执行器。
                from vllm.v1.executor.ray_executor_v2 import RayExecutorV2
                # 延迟导入 RayExecutorV2（基于 MessageQueue 通信的新版 Ray 执行器）。

                executor_class = RayExecutorV2
                # 使用 RayExecutorV2。
            else:
                from vllm.v1.executor.ray_executor import RayDistributedExecutor
                # 否则使用旧版 Ray 执行器（基于 Compiled DAG）。

                executor_class = RayDistributedExecutor
                # 使用 RayDistributedExecutor。
        elif distributed_executor_backend == "mp":
            # 情形 2b：后端为 "mp"（多进程）。
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
            # 延迟导入 MultiprocExecutor（默认的多 GPU 后端）。

            executor_class = MultiprocExecutor
            # 使用 MultiprocExecutor。
        elif distributed_executor_backend == "uni":
            # 情形 2c：后端为 "uni"（单进程）。
            from vllm.v1.executor.uniproc_executor import UniProcExecutor
            # 延迟导入 UniProcExecutor。

            executor_class = UniProcExecutor
            # 使用 UniProcExecutor。
        elif distributed_executor_backend == "external_launcher":
            # 情形 2d：后端为 "external_launcher"（外部 launcher，如 torchrun）。
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            # TODO 注释：未来需要让 v1 调度确定性化，以完整支持外部 launcher 场景。
            executor_class = ExecutorWithExternalLauncher
            # 使用 ExecutorWithExternalLauncher（定义于本文件底部导入的 uniproc 模块）。
        elif isinstance(distributed_executor_backend, str):
            # 情形 3：后端为任意字符串（用户自定义类的限定名，如 "pkg.mod.MyExecutor"）。
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            # 按限定名解析出 executor 类对象。
            if not issubclass(executor_class, Executor):
                # 校验解析出的类必须是 Executor 子类。
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
                # 否则抛 TypeError。
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
            # 兜底：无法识别的后端值直接抛 ValueError。
        return executor_class
        # 返回最终确定的 executor 类（供引擎实例化）。

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        # 构造参数：VllmConfig 顶层配置（聚合模型/并行/缓存/设备等全部子配置）。
    ) -> None:
        # -------------------------------------------------------------------
        # 构造函数：保存各类配置引用，并调用子类实现的 _init_executor()。
        # 被 @instrument 装饰：在链路追踪中创建名为 "Executor init" 的 span。
        # -------------------------------------------------------------------
        self.vllm_config = vllm_config
        # 保存顶层配置对象引用。
        self.model_config = vllm_config.model_config
        # 保存模型配置（模型名、架构、max_model_len 等）。
        self.cache_config = vllm_config.cache_config
        # 保存 KV cache 配置（块大小、GPU 内存预算等）。
        self.lora_config = vllm_config.lora_config
        # 保存 LoRA 配置（lora 模块、最大适配器等）。
        self.load_config = vllm_config.load_config
        # 保存权重加载配置（加载方式、dtype、模型目录等）。
        self.parallel_config = vllm_config.parallel_config
        # 保存并行配置（TP/PP/DP/PCP 大小、分布式后端等）。
        self.scheduler_config = vllm_config.scheduler_config
        # 保存调度器配置（调度策略、批大小上限等）。
        self.device_config = vllm_config.device_config
        # 保存设备配置（cuda/cpu 等设备类型与设备 id）。
        self.speculative_config = vllm_config.speculative_config
        # 保存投机解码配置（草稿模型、投机长度等）。
        self.observability_config = vllm_config.observability_config
        # 保存可观测性配置（结构化 output 日志、统计标志等）。
        self._init_executor()
        # 调用子类专属的初始化逻辑（如创建 worker、建立通信）——模板方法模式。
        self.is_sleeping = False
        # 初始化休眠状态标志：executor 当前是否处于低功耗「睡眠」态。
        self.sleeping_tags: set[str] = set()
        # 记录已进入睡眠的组件标签集合（如 "weights"、"kv_cache"），用于按需唤醒。
        self.kv_output_aggregator: KVOutputAggregator | None = None
        # KV 输出聚合器初始为 None，稍后通过 init_kv_output_aggregator() 创建。

    @abstractmethod
    def _init_executor(self) -> None:
        # -------------------------------------------------------------------
        # 抽象方法：子类必须实现的具体初始化逻辑。
        # 例如 UniProcExecutor 创建 driver_worker；MultiprocExecutor 拉起 worker 进程。
        # -------------------------------------------------------------------
        raise NotImplementedError
        # 基类不实现，直接抛未实现异常。

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        # -------------------------------------------------------------------
        # 根据配置初始化 KV cache，并触发底层 worker 启动模型执行循环前的
        # 编译（torch.compile）与预热（warm up）。
        # -------------------------------------------------------------------
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        # 文档字符串：初始化 KV cache 并启动底层 worker 的模型执行循环。
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
        # 向所有 worker 广播调用 initialize_from_config，
        # 每个 worker 会根据各自的 KVCacheConfig 分配 GPU 上的 KV cache 空间。
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        # 广播调用 compile_or_warm_up_model，收集各 worker 的编译耗时。
        # Propagate compilation time from workers back to the main process.
        # With TP>1, compilation happens in worker processes, so the main
        # process config is never updated. Use max across workers since they
        # compile in parallel.
        # 注释：将编译耗时从 worker 传播回主进程。TP>1 时编译发生在 worker 进程中，
        # 主进程的配置从未被更新，因此取所有 worker 的最大值（它们并行编译）。
        if compilation_times:
            # 只有当 collection 非空时（单进程 worker 也会返回列表）才更新。
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            # 语言模型（主模型）编译耗时取所有 worker 的最大值，写入主进程配置。
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )
            # 编码器模型（多模态场景）编译耗时同样取最大值。

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        # -------------------------------------------------------------------
        # 注册失败回调：executor 进入永久失败态时调用。
        # noqa: B027 表示忽略 flake8 关于「空抽象方法体」的告警——
        # 基类的默认实现是 no-op，子类可按需覆盖（如 Multiproc/RayV2 实现）。
        # -------------------------------------------------------------------
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        # 文档字符串：注册一个函数，当 executor 进入永久失败状态时被调用。
        pass
        # 基类默认什么都不做（UniProcExecutor 无需监控失败）。

    def determine_available_memory(self) -> list[int]:  # in bytes
        # -------------------------------------------------------------------
        # 让各 worker 计算各自可用的 GPU 显存（字节数），供 KV cache 分配预算。
        # -------------------------------------------------------------------
        return self.collective_rpc("determine_available_memory")
        # 广播调用 worker 的 determine_available_memory，返回各 worker 的可用显存列表。

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        # -------------------------------------------------------------------
        # 获取各 worker 返回的 KV cache 规格（每层张量形状、dtype 等）。
        # -------------------------------------------------------------------
        return self.collective_rpc("get_kv_cache_spec")
        # 广播调用 worker 的 get_kv_cache_spec，返回规格列表。

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        # method 参数：worker 方法名字符串，或一个以 WorkerBase 为 self 的可调用对象。
        timeout: float | None = None,
        # timeout：等待结果的最长秒数；None 表示无限等待。
        args: tuple = (),
        # args：传给 worker 方法的位置参数。
        kwargs: dict | None = None,
        # kwargs：传给 worker 方法的关键字参数。
        non_block: Literal[False] = False,
        # non_block：False（同步模式）时返回 list[_R]。
    ) -> list[_R]:
        # -------------------------------------------------------------------
        # collective_rpc 的重载声明 ①（同步模式）：
        # 仅用于类型检查，运行时由下方抽象定义实际执行。
        # -------------------------------------------------------------------
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        # 文档字符串：说明集体 RPC 的语义与参数：
        #   method 可以是方法名或可调用对象（被序列化后发给所有 worker 执行）；
        #   non_block=True 时返回 Future 列表而非阻塞等待；
        #   建议本 API 只传递控制消息，数据平面另行建立通道。
        pass
        # 重载声明体为空，仅提供类型签名。

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        # 与重载①相同的 method 参数。
        timeout: float | None = None,
        # 与重载①相同的 timeout。
        args: tuple = (),
        # 与重载①相同的 args。
        kwargs: dict | None = None,
        # 与重载①相同的 kwargs。
        non_block: Literal[True] = True,
        # non_block：True（异步模式）时返回 Future[list[_R]]。
    ) -> Future[list[_R]]:
        # -------------------------------------------------------------------
        # collective_rpc 的重载声明 ②（异步模式）：
        # 返回 Future[list[_R]] 而非直接结果。
        # -------------------------------------------------------------------
        pass
        # 重载声明体为空，仅提供类型签名。

    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        # -------------------------------------------------------------------
        # collective_rpc 的实际抽象定义（运行时生效）：
        # 各子类（UniProc/Multiproc/Ray）分别实现自己的广播机制。
        # -------------------------------------------------------------------
        raise NotImplementedError
        # 基类不实现，抛未实现异常。

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[tuple[int, int], KVConnectorHandshakeMetadata]]:
        # -------------------------------------------------------------------
        # 收集各 worker 的 KV 连接器握手元数据。
        # 用于解耦式部署（disaggregated serving）中 prefill 实例与 decode 实例
        # 在建立 KV 传输连接前的参数协商。
        # 返回结构：list（按 worker）-> dict（key=(连出 rank, 连入 rank) -> 元数据）。
        # -------------------------------------------------------------------
        return self.collective_rpc("get_kv_connector_handshake_metadata")
        # 广播调用 worker 的该方法，汇总握手元数据列表。

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        # -------------------------------------------------------------------
        # execute_model 重载声明 ①（同步）：
        # 执行一轮调度输出；返回 ModelRunnerOutput（或 None 表示本次无输出）。
        # -------------------------------------------------------------------
        pass
        # 仅类型声明。

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # execute_model 重载声明 ②（异步）：返回 Future。
        # -------------------------------------------------------------------
        pass
        # 仅类型声明。

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # -------------------------------------------------------------------
        # execute_model 实际实现（数据平面核心路径）：
        # 把调度输出作为参数，经 collective_rpc 广播到所有 worker，
        # 只取第一个 worker 的输出（TP/PP 场景下只有 rank0+last-pp 有完整输出）。
        # -------------------------------------------------------------------
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        # 广播执行 worker 的 execute_model，arg 为调度输出；忽略重载类型提示。
        return output[0]
        # 返回第一个 worker 的结果（同步返回输出；异步返回 Future，接口一致）。

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        # -------------------------------------------------------------------
        # sample_tokens 重载声明 ①（同步）：
        # 基于模型 logits 采样生成 token；grammar_output 提供结构化输出约束。
        # -------------------------------------------------------------------
        pass
        # 仅类型声明。

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        # -------------------------------------------------------------------
        # sample_tokens 重载声明 ②（异步）：返回 Future。
        # -------------------------------------------------------------------
        pass
        # 仅类型声明。

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        # -------------------------------------------------------------------
        # sample_tokens 实际实现：广播采样调用，取第一个 worker 的输出。
        # 与 execute_model 配对使用：execute_model 完成后由本方法做采样。
        # -------------------------------------------------------------------
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        # 广播执行 worker 的 sample_tokens；忽略重载类型提示。
        return output[0]
        # 返回第一个 worker 的采样结果。

    def execute_dummy_batch(self) -> None:
        # -------------------------------------------------------------------
        # 执行一个空批（dummy batch），用于预热 CUDA graph / 触发 kernel 编译，
        # 通常在模型加载后、正式服务前调用一次以消除首个请求的延迟尖峰。
        # -------------------------------------------------------------------
        self.collective_rpc("execute_dummy_batch")
        # 广播到所有 worker 执行 dummy 批。

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # -------------------------------------------------------------------
        # 获取投机解码中草稿模型产出的候选 token（draft token）。
        # 仅在启用 speculative decoding 时返回非 None。
        # -------------------------------------------------------------------
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        # 广播调用 worker 的 take_draft_token_ids。
        return output[0]
        # 只取第一个 worker 的结果（草稿 token 在所有 worker 间是一致的）。

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # -------------------------------------------------------------------
        # 启动/停止 nsight 等性能剖析（profiling）。
        # is_start=True 开始剖析；is_start=False 结束。
        # -------------------------------------------------------------------
        self.collective_rpc("profile", args=(is_start, profile_prefix))
        # 广播调用 worker 的 profile 方法。

    def save_sharded_state(
        self,
        path: str,
        # path：保存分片权重/状态的目标目录。
        pattern: str | None = None,
        # pattern：可选的权重名匹配模式（如 "*.bin"）。
        max_size: int | None = None,
        # max_size：单个分片文件的最大字节数（超出则自动切分）。
    ) -> None:
        # -------------------------------------------------------------------
        # 将模型状态（权重）按分片保存到磁盘，用于断点保存/热重载。
        # -------------------------------------------------------------------
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )
        # 广播调用所有 worker 的 save_sharded_state（每个 worker 保存自己的分片）。

    @abstractmethod
    def check_health(self) -> None:
        # -------------------------------------------------------------------
        # 抽象方法：健康检查。executor 不健康时应抛出异常。
        # 子类实现差异：UniProc 恒健康；Multiproc 广播 check_health；
        # Ray 旧版假定健康（TODO 待实现）。
        # -------------------------------------------------------------------
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        # 文档字符串：检查 executor 是否健康，不健康时抛异常。
        raise NotImplementedError
        # 基类不实现。

    def shutdown(self) -> None:
        # -------------------------------------------------------------------
        # 优雅关闭 executor：广播 shutdown 到所有 worker，
        # 释放 KV cache、销毁分布式环境等。
        # -------------------------------------------------------------------
        """Shutdown the executor."""
        # 文档字符串：关闭 executor。
        self.collective_rpc("shutdown")
        # 广播调用所有 worker 的 shutdown。

    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        # -------------------------------------------------------------------
        # 根据 KV 连接器初始化 KV 输出聚合器（仅在 KV 迁移部署下使用）。
        # -------------------------------------------------------------------
        """Init KVOutputAggregator"""
        # 文档字符串：初始化 KVOutputAggregator。
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )
        # 用连接器与全局 world_size 构建聚合器，后续 execute_model 结果会经它聚合。

    @cached_property  # Avoid unnecessary RPC calls
    # cached_property：结果缓存到实例属性，避免多次 RPC 调用（加了注释说明）。
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        # -------------------------------------------------------------------
        # 查询当前 worker 支持的推理任务类型（generate/embed 等）。
        # 由于结果不变，用 cached_property 缓存。
        # -------------------------------------------------------------------
        output: list[tuple[SupportedTask, ...]]
        # 类型标注：collective_rpc 返回的元素是任务元组。
        output = self.collective_rpc("get_supported_tasks")
        # 广播调用 worker 的 get_supported_tasks。
        return output[0]
        # 只取第一个 worker 的结果（所有 worker 支持的任务一致）。

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # -------------------------------------------------------------------
        # 向所有 worker 动态加载一个 LoRA 适配器。
        # -------------------------------------------------------------------
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        # 断言 LoRA 的整数 id 必须大于 0（id 唯一标识一个 LoRA）。
        return all(self.collective_rpc("add_lora", args=(lora_request,)))
        # 广播 add_lora；只有所有 worker 都成功加载才返回 True（all() 逻辑与）。

    def remove_lora(self, lora_id: int) -> bool:
        # -------------------------------------------------------------------
        # 从所有 worker 移除一个 LoRA 适配器。
        # -------------------------------------------------------------------
        assert lora_id > 0, "lora_id must be greater than 0."
        # 断言 lora id 必须大于 0。
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))
        # 广播 remove_lora；全部成功才返回 True。

    def pin_lora(self, lora_id: int) -> bool:
        # -------------------------------------------------------------------
        # 「固定」一个 LoRA：将某个热门适配器常驻显存，避免频繁换入换出。
        # -------------------------------------------------------------------
        assert lora_id > 0, "lora_id must be greater than 0."
        # 断言 lora id 必须大于 0。
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))
        # 广播 pin_lora；全部成功才返回 True。

    def list_loras(self) -> set[int]:
        # -------------------------------------------------------------------
        # 获取当前所有 worker 上已加载的 LoRA id 集合。
        # -------------------------------------------------------------------
        sets: list[set[int]] = self.collective_rpc("list_loras")
        # 广播 list_loras，得到各 worker 返回的 LoRA id 集合。
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
            # 断言所有 worker 的 LoRA 集合一致（分布式一致性约束）。
        return sets[0]
        # 返回第一个 worker 的集合即可代表全局。

    def reset_mm_cache(self) -> None:
        # -------------------------------------------------------------------
        # 重置每个 worker 中的多模态（multi-modal）特征缓存。
        # 用于像 encoder 特征这类跨请求可复用数据的显式清理。
        # -------------------------------------------------------------------
        """Reset the multi-modal cache in each worker."""
        # 文档字符串：重置每个 worker 中的多模态缓存。
        self.collective_rpc("reset_mm_cache")
        # 广播 reset_mm_cache。

    def reset_encoder_cache(self) -> None:
        # -------------------------------------------------------------------
        # 重置 encoder 缓存，清除缓存的 encoder 输出。
        # 多模态模型中视觉/语音编码器输出按请求缓存，需要时手动清空。
        # -------------------------------------------------------------------
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        # 文档字符串：重置 encoder 缓存以清除已缓存的 encoder 输出。
        self.collective_rpc("reset_encoder_cache")
        # 广播 reset_encoder_cache。

    def sleep(self, level: int = 1):
        # -------------------------------------------------------------------
        # 让 executor 进入低功耗「睡眠」状态（如 model sleep 时 GPU 降频省电）。
        # -------------------------------------------------------------------
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            # 若已在睡眠态，则告警并直接返回（幂等保护）。
            return
        time_before_sleep = time.perf_counter()
        # 记录入睡前的高精度时间戳，用于统计入睡耗时。
        self.collective_rpc("sleep", kwargs=dict(level=level))
        # 广播调用所有 worker 的 sleep（level 表示睡眠深度）。
        time_after_sleep = time.perf_counter()
        # 记录入睡后的时间戳。
        self.sleeping_tags = {"weights", "kv_cache"}
        # 标记已「睡」的组件标签：权重与 KV cache 都进入休眠。
        self.is_sleeping = True
        # 置位休眠标志。
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )
        # 记录入睡耗时日志（便于观测睡眠开销）。

    def wake_up(self, tags: list[str] | None = None):
        # -------------------------------------------------------------------
        # 按标签唤醒 executor 的若干组件（weights / kv_cache）。
        # tags=None 表示全部唤醒。
        # -------------------------------------------------------------------
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            # 未在睡眠态则告警并返回（幂等保护）。
            return
        if tags:
            # 若指定了 tags，先做校验。
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    # 若请求唤醒的组件并未处于睡眠态，则告警并中止整体唤醒。
                    return
        time_before_wakeup = time.perf_counter()
        # 记录唤醒前时间戳。
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        # 广播调用所有 worker 的 wake_up。
        time_after_wakeup = time.perf_counter()
        # 记录唤醒后时间戳。
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        # 记录唤醒耗时日志。
        if tags:
            # 若指定了 tags，则只移除对应标签。
            for tag in tags:
                self.sleeping_tags.remove(tag)
                # 从睡眠标签集合中逐个移除已唤醒的组件。
        else:
            self.sleeping_tags.clear()
            # 未指定 tags 则清空全部睡眠标签。
        if not self.sleeping_tags:
            self.is_sleeping = False
            # 当没有任何组件处于睡眠时，整体退出睡眠态。

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        # -------------------------------------------------------------------
        # 重新初始化分布式环境（DP 弹性扩缩容时调用）。
        # 基类默认不支持；Ray 执行器会覆盖此方法。
        # -------------------------------------------------------------------
        raise NotImplementedError
        # 基类抛未实现异常，需要弹性扩缩容的后端自行实现。

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # -------------------------------------------------------------------
        # 类方法：查询该 executor 是否支持异步调度（CPU 调度与 GPU 推理重叠）。
        # 基类默认返回 False；UniProc/Multiproc/RayV2 覆盖为 True。
        # -------------------------------------------------------------------
        """
        Whether the executor supports async scheduling.
        """
        # 文档字符串：该 executor 是否支持异步调度。
        return False
        # 默认不支持。


from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
# 文件底部导入 ExecutorWithExternalLauncher（noqa: E402 允许 import 不在文件头部）。
# 这样 abstract.py 顶部的 get_class() 可直接引用而不会循环导入。
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)
# 同样在底部导入 UniProcExecutor（别名 _UniProcExecutor）。

# For backwards compatibility.
# 注释：以下两行用于向后兼容（外部代码可能从本模块导入这两个类）。
UniProcExecutor = _UniProcExecutor
# 将底部导入的类重新赋值为原名字供外部使用。
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
# 同上，暴露 ExecutorWithExternalLauncher 原名。