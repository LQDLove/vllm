# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import time  # time：时间模块（统计日志间隔用）
import weakref  # weakref：弱引用（防止循环引用导致模型无法回收）
from collections.abc import Callable, Mapping
# Callable：可调用对象类型；Mapping：映射类型（trace_headers 用）
from copy import copy  # copy：浅拷贝（n>1 时复制子请求）
from typing import Any  # Any：通用类型标注

import torch.nn as nn  # nn：PyTorch 神经网络模块（apply_model 用）
from typing_extensions import TypeVar  # TypeVar：泛型类型变量（支持 default 参数）

import vllm.envs as envs  # vllm.envs：vLLM 环境变量配置
from vllm.config import ParallelConfig, VllmConfig
# 配置类：并行配置、vLLM 全局配置
from vllm.distributed import stateless_destroy_torch_distributed_process_group
# 销毁无状态 torch 分布式进程组（DP group 清理）
from vllm.distributed.parallel_state import get_dp_group  # 获取 DP 进程组
from vllm.engine.arg_utils import EngineArgs  # 引擎命令行参数
from vllm.inputs import EngineInput, PromptType  # 输入类型
from vllm.logger import init_logger  # 初始化 vLLM 日志记录器
from vllm.lora.request import LoRARequest  # LoRA 请求
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
# 多模态注册表
from vllm.outputs import PoolingRequestOutput, RequestOutput  # 输出类型
from vllm.pooling_params import PoolingParams  # 池化参数
from vllm.renderers import renderer_from_config  # 从配置创建 renderer
from vllm.renderers.inputs.preprocess import extract_prompt_components
# 提取 prompt 组件（文本等）
from vllm.sampling_params import SamplingParams  # 采样参数
from vllm.tasks import SupportedTask  # 支持的任务类型
from vllm.tokenizers import TokenizerLike  # tokenizer 接口类型
from vllm.tracing import init_tracer  # 初始化 OpenTelemetry 追踪
from vllm.usage.usage_lib import UsageContext  # 使用场景上下文
from vllm.v1.engine import EngineCoreRequest, PauseMode
# 引擎核心请求、暂停模式
from vllm.v1.engine.core_client import EngineCoreClient  # 核心引擎客户端
from vllm.v1.engine.input_processor import InputProcessor  # 输入处理器
from vllm.v1.engine.output_processor import OutputProcessor  # 输出处理器
from vllm.v1.engine.parallel_sampling import ParentRequest  # 并行采样父请求
from vllm.v1.executor import Executor  # 执行器抽象类
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
# 统计日志器工厂、管理器
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
# 指标读取：Metric 类型、获取指标快照
from vllm.v1.metrics.stats import IterationStats  # 迭代统计
from vllm.v1.utils import record_function_or_nullcontext
# 性能分析上下文（可空 context，用于 torch profiler 标记）
from vllm.v1.worker.worker_base import WorkerBase  # worker 基类（collective_rpc 用）

logger = init_logger(__name__)  # 模块级日志记录器

_R = TypeVar("_R", default=Any)  # 泛型返回类型变量（collective_rpc/apply_model 用）


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""
    # 遗留版 LLMEngine：用于向后兼容（V0 风格的同步 API）

    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        aggregate_engine_logging: bool = False,  # 是否聚合引擎日志
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用场景
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义统计日志器
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表
        multiprocess_mode: bool = False,  # 是否使用多进程模式
    ) -> None:
        self.vllm_config = vllm_config  # 保存全局配置
        self.model_config = vllm_config.model_config  # 保存模型配置
        self.observability_config = vllm_config.observability_config  # 可观测配置

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        # 获取 OpenTelemetry 追踪端点
        if tracing_endpoint is not None:
            # 如果配置了追踪端点
            init_tracer("vllm.llm_engine", tracing_endpoint)
            # 初始化追踪器

        self.log_stats = log_stats  # 保存日志统计标志

        parallel_config = vllm_config.parallel_config  # 获取并行配置
        executor_backend = parallel_config.distributed_executor_backend
        # 执行器后端类型

        self.external_launcher_dp = (
            # 是否为外部启动器 DP 模式
            parallel_config.data_parallel_size > 1  # DP>1
            and executor_backend == "external_launcher"  # 外部启动器后端
        )
        # important: init dp group before init the engine_core
        # 重要：在初始化 engine_core 之前初始化 DP 进程组
        # In the decoupled engine case this is handled in EngineCoreProc.
        # 在解耦引擎场景中，此操作由 EngineCoreProc 处理。
        if (
            not multiprocess_mode  # 非多进程模式
            and parallel_config.data_parallel_size > 1  # DP>1
            and not self.external_launcher_dp  # 非外部启动器
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
            # 初始化无状态 DP 进程组
        else:
            self.dp_group = None  # 其他情况无 DP group
        self.should_execute_dummy_batch = False
        # 是否应执行空 batch（DP 同步用）

        self.renderer = renderer = renderer_from_config(self.vllm_config)
        # 创建 renderer（tokenizer + 多模态处理）

        # Convert EngineInput --> EngineCoreRequest.
        # 将 EngineInput 转换为 EngineCoreRequest
        self.input_processor = InputProcessor(self.vllm_config, renderer)
        # 创建输入处理器

        # Converts EngineCoreOutputs --> RequestOutput.
        # 将 EngineCoreOutputs 转换为 RequestOutput
        self.output_processor = OutputProcessor(
            renderer.tokenizer,  # tokenizer
            log_stats=self.log_stats,  # 日志统计
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            # 流式输出间隔
            tracing_enabled=tracing_endpoint is not None,  # 是否启用追踪
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        # 创建核心引擎客户端（接收请求，返回输出）
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,  # 多进程模式
            asyncio_mode=False,  # 同步模式（非异步）
            vllm_config=vllm_config,  # 配置
            executor_class=executor_class,  # 执行器
            log_stats=self.log_stats,  # 日志统计
        )

        self.logger_manager: StatLoggerManager | None = None  # 日志管理器
        if self.log_stats:
            # 如果启用了日志统计
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,  # 配置
                custom_stat_loggers=stat_loggers,  # 自定义日志器
                enable_default_loggers=log_stats,  # 启用默认日志器
                aggregate_engine_logging=aggregate_engine_logging,  # 聚合日志
            )
            self.logger_manager.log_engine_initialized()
            # 记录引擎初始化事件

        if not multiprocess_mode:
            # for v0 compatibility
            # 非多进程模式：为了 V0 兼容性
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore
            # 直接从同进程引擎核心获取模型执行器

            # Capture the model while reachable so the finalizer can drop the
            # bytecode hooks pinning it (frees GPU memory on engine deletion).
            # 在模型可达时捕获它，以便终结器可以移除固定它的字节码钩子
            # （引擎删除时释放 GPU 内存）
            model = self._get_driver_model_for_cleanup()  # 获取 driver 模型
            if model is not None:
                # 如果模型存在
                self._finalizer = weakref.finalize(
                    # 创建弱引用终结器
                    self, LLMEngine._cleanup_instance_caches, model
                    # 对象回收时清理编译缓存
                )

        if self.external_launcher_dp:
            # 如果使用外部启动器 DP 模式
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            # 若在外部启动器模式中使用 DP，复用已有的用于数据通信的 DP group。
            self.dp_group = get_dp_group().cpu_group  # 获取 CPU 进程组

        # Don't keep the dummy data in memory
        # 不要将占位数据保留在内存中
        self.reset_mm_cache()  # 重置多模态缓存

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,  # vLLM 配置
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用场景
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义日志器
        disable_log_stats: bool = False,  # 是否禁用日志统计
    ) -> "LLMEngine":
        # 工厂方法：从 VllmConfig 创建 LLMEngine
        return cls(
            vllm_config=vllm_config,  # 配置
            executor_class=Executor.get_class(vllm_config),
            # 根据配置自动选择执行器类
            log_stats=(not disable_log_stats),  # 日志统计标志取反
            usage_context=usage_context,  # 使用场景
            stat_loggers=stat_loggers,  # 自定义日志器
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
            # 从环境变量读取多进程模式
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,  # 引擎命令行参数
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用场景
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义日志器
        enable_multiprocessing: bool = False,  # 是否启用多进程
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # 从引擎命令行参数创建 LLMEngine

        # Create the engine configs.
        # 创建引擎配置
        vllm_config = engine_args.create_engine_config(usage_context)
        # 从参数创建配置
        executor_class = Executor.get_class(vllm_config)
        # 选择执行器类

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            # 如果环境变量启用了多进程
            logger.debug("Enabling multiprocessing for LLMEngine.")
            # 记录调试日志
            enable_multiprocessing = True  # 强制启用多进程

        # Create the LLMEngine.
        # 创建 LLMEngine
        return cls(
            vllm_config=vllm_config,  # 配置
            executor_class=executor_class,  # 执行器
            log_stats=not engine_args.disable_log_stats,  # 日志统计
            usage_context=usage_context,  # 使用场景
            stat_loggers=stat_loggers,  # 自定义日志器
            multiprocess_mode=enable_multiprocessing,  # 多进程模式
        )

    def get_num_unfinished_requests(self) -> int:
        # 获取未完成请求数
        return self.output_processor.get_num_unfinished_requests()
        # 委托给输出处理器

    def has_unfinished_requests(self) -> bool:
        # 是否有未完成的请求
        has_unfinished = self.output_processor.has_unfinished_requests()
        # 查询本地是否还有未完成请求
        if self.dp_group is None:
            # 如果没有 DP group
            return has_unfinished or self.engine_core.dp_engines_running()
            # 本地未完成 或 引擎运行中
        return self.has_unfinished_requests_dp(has_unfinished)
        # 否则使用 DP 聚合判断

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        # DP 模式下聚合判断是否有未完成请求
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        # 通过 all-reduce 聚合所有 DP rank 的状态
        if not has_unfinished and aggregated_has_unfinished:
            # 如果本地无请求但全局有请求（其他 rank 有）
            self.should_execute_dummy_batch = True
            # 标记需要执行空 batch（让所有 rank 保持同步调度）
        return aggregated_has_unfinished  # 返回聚合结果

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 获取支持的任务类型
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            # 缓存结果（避免重复查询）
            self._supported_tasks = self.engine_core.get_supported_tasks()
            # 从核心引擎查询

        return self._supported_tasks  # 返回缓存结果

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""
        # 从 EngineCore 和 Detokenizer 移除请求

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        # 从前端输出处理器中止（返回需要发给核心的 ID 列表）
        self.engine_core.abort_requests(request_ids)
        # 从核心引擎中止

    def add_request(
        self,
        request_id: str,  # 用户提供的请求 ID
        prompt: EngineCoreRequest | PromptType | EngineInput,  # 输入
        params: SamplingParams | PoolingParams,  # 采样/池化参数
        arrival_time: float | None = None,  # 到达时间（可选）
        lora_request: LoRARequest | None = None,  # LoRA 请求（可选）
        tokenization_kwargs: dict[str, Any] | None = None,  # tokenize 参数（可选）
        trace_headers: Mapping[str, str] | None = None,  # 追踪头（可选）
        priority: int = 0,  # 优先级
        prompt_text: str | None = None,  # prompt 文本（可选）
    ) -> str:
        # 添加请求到引擎，返回内部请求 ID
        # Validate the request_id type.
        # 验证 request_id 类型
        if not isinstance(request_id, str):
            # 如果不是字符串
            raise TypeError(f"request_id must be a string, got {type(request_id)}")
            # 抛出类型错误

        # Process raw inputs into the request.
        # 将原始输入处理为请求
        if isinstance(prompt, EngineCoreRequest):
            # 如果输入已是 EngineCoreRequest
            logger.warning_once(
                # 记录一次性警告（弃用提示）
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt  # 直接使用（已预处理）
            if request_id != request.request_id:
                # 如果传入的 request_id 与请求内不一致
                logger.warning_once(
                    # 记录一次性警告
                    "LLMEngine.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            request = self.input_processor.process_inputs(
                # 通过输入处理器处理
                request_id,  # 请求 ID
                prompt,  # 原始输入
                params,  # 参数
                supported_tasks=self.get_supported_tasks(),  # 支持的任务
                arrival_time=arrival_time,  # 到达时间
                lora_request=lora_request,  # LoRA
                tokenization_kwargs=tokenization_kwargs,  # tokenize 参数
                trace_headers=trace_headers,  # 追踪头
                priority=priority,  # 优先级
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)
            # 提取 prompt 组件（文本部分）

        self.input_processor.assign_request_id(request)
        # 分配内部唯一请求 ID（保存外部 ID 到 external_req_id）

        req_id = request.request_id  # 获取内部请求 ID

        # Use cloned params that may have been updated in process_inputs()
        # 使用可能已在 process_inputs() 中更新的克隆参数
        params = request.params  # 获取处理后的参数

        n = params.n if isinstance(params, SamplingParams) else 1
        # 并行采样数：仅 SamplingParams 有 n，pooling 视为 1

        if n == 1:
            # Make a new RequestState and queue.
            # 创建新的 RequestState 和队列
            self.output_processor.add_request(request, prompt_text, None, 0)
            # 在前端注册请求状态
            # Add the request to EngineCore.
            # 将请求添加到核心引擎
            self.engine_core.add_request(request)
            return req_id  # 返回内部请求 ID

        # Fan out child requests (for n>1).
        # 并行采样（n>1）：将请求拆分为多个子请求
        parent_req = ParentRequest(request)  # 创建父请求管理器
        for idx in range(n):
            # 遍历每个子请求索引
            request_id, child_params = parent_req.get_child_info(idx)
            # 获取子请求 ID 和参数
            child_request = request if idx == n - 1 else copy(request)
            # 最后一个子请求复用原对象（避免多余拷贝）
            child_request.request_id = request_id  # 设置子请求 ID
            child_request.sampling_params = child_params  # 设置子请求参数

            # Make a new RequestState and queue.
            # 创建新的 RequestState
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # 注册子请求状态（关联父请求）
            # Add the request to EngineCore.
            # 将子请求添加到核心引擎
            self.engine_core.add_request(child_request)

        return req_id  # 返回父请求 ID

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        # 单步执行引擎：拉取输出、处理、返回结果
        if self.should_execute_dummy_batch:
            # 如果需要执行空 batch
            self.should_execute_dummy_batch = False  # 重置标志
            self.engine_core.execute_dummy_batch()  # 执行空 batch
            return []  # 返回空（无输出）

        # 1) Get EngineCoreOutput from the EngineCore.
        # 1) 从核心引擎获取输出
        with record_function_or_nullcontext("llm_engine step: get_output"):
            # 使用性能分析上下文标记
            outputs = self.engine_core.get_output()
            # 获取 EngineCoreOutputs（同步阻塞读取）

        # 2) Process EngineCoreOutputs.
        # 2) 处理 EngineCoreOutputs
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            # 性能分析上下文
            iteration_stats = IterationStats() if self.log_stats else None
            # 创建迭代统计（如启用日志）
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,  # 核心输出列表
                engine_core_timestamp=outputs.timestamp,  # 输出时间戳
                iteration_stats=iteration_stats,  # 迭代统计
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)
            # 更新调度器统计

        # 3) Abort any reqs that finished due to stop strings.
        # 3) 中止因 stop string 结束的请求
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            # 性能分析上下文
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
            # 向核心引擎发送中止请求

        # 4) Record stats
        # 4) 记录统计
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            # 性能分析上下文
            if (
                self.logger_manager is not None  # 有日志管理器
                and outputs.scheduler_stats is not None  # 有调度统计
                and len(outputs.outputs) > 0  # 有输出
            ):
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,  # 调度统计
                    iteration_stats=iteration_stats,  # 迭代统计
                    mm_cache_stats=self.renderer.stat_mm_cache(),  # 多模态缓存统计
                )
                self.do_log_stats_with_interval()  # 按间隔记录日志

        return processed_outputs.request_outputs  # 返回处理后的输出列表

    def start_profile(self, profile_prefix: str | None = None):
        # 开始性能分析
        self.engine_core.profile(True, profile_prefix)  # 通知核心引擎启动 profiler

    def stop_profile(self):
        # 停止性能分析
        self.engine_core.profile(False)  # 通知核心引擎停止 profiler

    def reset_mm_cache(self):
        # 重置多模态缓存
        self.renderer.clear_mm_cache()  # 清空前端 renderer 缓存
        self.engine_core.reset_mm_cache()  # 通知核心引擎清空

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 重置前缀缓存
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )
        # 委托给核心引擎

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        # 重置编码器缓存以失效所有缓存编码器输出。
        # 模型权重更新时应调用此方法，确保不会复用旧权重计算的过期视觉嵌入。
        self.engine_core.reset_encoder_cache()  # 委托给核心引擎

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        # 引擎休眠（释放 GPU 内存）
        if level >= 1:
            # 如果休眠级别 >=1
            self.renderer.clear_mm_cache()  # 清空多模态缓存
        self.engine_core.sleep(level, mode)  # 通知核心引擎休眠

        if self.logger_manager is not None:
            # 如果有日志管理器
            self.logger_manager.record_sleep_state(1, level)
            # 记录休眠状态

    def wake_up(self, tags: list[str] | None = None):
        # 引擎唤醒
        self.engine_core.wake_up(tags)  # 通知核心引擎唤醒

        if self.logger_manager is not None:
            # 如果有日志管理器
            self.logger_manager.record_sleep_state(0, 0)
            # 记录唤醒状态

    def is_sleeping(self) -> bool:
        # 检查引擎是否在休眠
        return self.engine_core.is_sleeping()  # 委托给核心引擎

    def get_metrics(self) -> list[Metric]:
        # 获取指标快照
        assert self.log_stats, "Stat logging disabled"
        # 断言已启用统计日志
        return get_metrics_snapshot()  # 返回全局指标快照

    @property
    def tokenizer(self) -> TokenizerLike | None:
        # 属性：获取 tokenizer（可能为 None）
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        # 获取 tokenizer 实例（保证非 None）
        return self.renderer.get_tokenizer()

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        # 如果启用了日志则记录统计
        if self.logger_manager:
            # 如果有日志管理器
            self.logger_manager.log()  # 立即记录

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        # 当时间间隔经过时记录统计
        now = time.time()  # 当前时间
        if not hasattr(self, "_last_log_time"):
            # 如果没有上次记录时间
            self._last_log_time = now  # 初始化为当前时间
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            # 如果距离上次记录已超过日志间隔
            self.do_log_stats()  # 记录日志
            self._last_log_time = now  # 更新上次记录时间

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        # 加载新的 LoRA 适配器供后续请求使用
        return self.engine_core.add_lora(lora_request)  # 委托给核心引擎

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        # 移除已加载的 LoRA 适配器
        return self.engine_core.remove_lora(lora_id)  # 委托给核心引擎

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        # 列出所有已注册的适配器
        return self.engine_core.list_loras()  # 委托给核心引擎

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        # 防止适配器被逐出（固定）
        return self.engine_core.pin_lora(lora_id)  # 委托给核心引擎

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],  # 调用的方法名或函数
        timeout: float | None = None,  # 超时（可选）
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:
        # 集体 RPC：在所有 worker 上调用同一方法
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)
        # 委托给核心引擎

    def set_weight_version(self, weight_version: str) -> None:
        self.engine_core.set_weight_version(weight_version)

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self.engine_core.get_weight_version()

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        # 在所有 worker 上应用模型函数
        return self.collective_rpc("apply_model", args=(func,))
        # 通过 collective_rpc 调用 apply_model

    def _get_driver_model_for_cleanup(self) -> nn.Module | None:
        # 获取 driver worker 的模型（用于清理）
        driver_worker = getattr(self.model_executor, "driver_worker", None)
        # 获取 driver worker
        model_runner = getattr(driver_worker, "model_runner", None)
        # 获取模型运行器
        return getattr(model_runner, "model", None)  # 返回模型

    @staticmethod
    def _cleanup_instance_caches(model) -> None:
        """Remove the bytecode hooks that pin the compiled model."""
        # 移除固定编译模型的字节码钩子
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
        # 延迟导入编译包装器

        for module in model.modules():
            # 遍历模型所有模块
            if isinstance(module, TorchCompileWithNoGuardsWrapper):
                # 如果是编译包装器
                module.cleanup()  # 清理钩子

    def __del__(self):
        # 析构函数：销毁 DP 进程组
        dp_group = getattr(self, "dp_group", None)  # 获取 DP group
        if dp_group is not None and not self.external_launcher_dp:
            # 如果有 DP group 且非外部启动器模式（外部模式由外部管理）
            stateless_destroy_torch_distributed_process_group(dp_group)
            # 销毁无状态 torch 分布式进程组