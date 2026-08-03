# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import asyncio  # asyncio：异步 I/O 框架（async/await 支持）
import os  # os：操作系统接口（获取 PID、主机名等）
import socket  # socket：网络套接字（获取主机名用于 profiler）
import time  # time：时间模块（到达时间、等待间隔）
import warnings  # warnings：警告模块（DeprecationWarning）
from collections.abc import AsyncGenerator, Iterable, Mapping
# AsyncGenerator：异步生成器类型；Iterable：可迭代对象；Mapping：映射类型
from copy import copy  # copy：浅拷贝（n>1 时复制子请求）
from typing import Any  # Any：通用类型标注

import torch  # torch：PyTorch
import vllm.envs as envs  # vllm.envs：vLLM 环境变量配置
from vllm import TokensPrompt  # TokensPrompt：基于 token IDs 的 prompt 类型
from vllm.config import VllmConfig  # vLLM 全局配置
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,  # 权重传输初始化请求（RL 训练）
    WeightTransferUpdateRequest,  # 权重更新请求（RL 训练）
)
from vllm.engine.arg_utils import AsyncEngineArgs  # 异步引擎命令行参数
from vllm.engine.protocol import EngineClient, StreamingInput
# EngineClient：引擎客户端协议接口；StreamingInput：流式输入类型
from vllm.entrypoints.serve.elastic_ep.middleware import set_scaling_elastic_ep
# 设置弹性 EP 缩放中标志（中间件）
from vllm.inputs import EngineInput, PromptType  # 输入类型
from vllm.logger import init_logger  # 初始化 vLLM 日志记录器
from vllm.lora.request import LoRARequest  # LoRA 请求
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
# 多模态注册表
from vllm.outputs import STREAM_FINISHED, PoolingRequestOutput, RequestOutput
# 输出类型：流完成哨兵、池化输出、请求输出
from vllm.pooling_params import PoolingParams  # 池化参数
from vllm.renderers import renderer_from_config  # 从配置创建 renderer
from vllm.renderers.inputs.preprocess import extract_prompt_components
# 提取 prompt 组件（文本部分）
from vllm.sampling_params import RequestOutputKind, SamplingParams
# 采样参数；输出类型枚举（FINAL_ONLY/DELTA 等）
from vllm.tasks import SupportedTask  # 支持的任务类型
from vllm.tokenizers import TokenizerLike  # tokenizer 接口类型
from vllm.tracing import init_tracer  # 初始化 OpenTelemetry 追踪
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
# 注册配置按值序列化（跨进程传输需要）
from vllm.usage.usage_lib import UsageContext  # 使用场景上下文
from vllm.utils.async_utils import cancel_task_threadsafe  # 线程安全取消任务
from vllm.utils.collection_utils import as_list  # 将值转为列表
from vllm.v1.engine import EngineCoreRequest, PauseMode
# 引擎核心请求；暂停模式（abort/wait/keep）
from vllm.v1.engine.core_client import EngineCoreClient  # 核心引擎客户端
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
# 引擎死亡错误；引擎生成错误
from vllm.v1.engine.input_processor import InputProcessor  # 输入处理器
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
# 输出处理器；请求输出收集器（per-request 队列）
from vllm.v1.engine.parallel_sampling import ParentRequest  # 并行采样父请求
from vllm.v1.executor import Executor  # 执行器抽象类
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult
# 故障容错请求与结果类型
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,  # 统计日志器工厂
    StatLoggerManager,  # 统计日志器管理器
    load_stat_logger_plugin_factories,  # 加载统计日志器插件工厂
)
from vllm.v1.metrics.prometheus import shutdown_prometheus  # 关闭 Prometheus
from vllm.v1.metrics.stats import IterationStats  # 迭代统计

logger = init_logger(__name__)  # 模块级日志记录器


class InputStreamError(Exception):
    """Wrapper for errors from the input stream generator.

    This is used to propagate errors from the user's input generator
    without wrapping them in EngineGenerateError.
    """
    # 输入流生成器错误的包装器。
    # 用于传播用户输入生成器的错误，而不包装在 EngineGenerateError 中。

    def __init__(self, cause: Exception):
        # 构造函数
        self.cause = cause  # 保存原始异常
        super().__init__(str(cause))  # 调用父类初始化


class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""
    # vLLM 引擎的异步包装器（前端引擎）

    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用场景
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表
        log_requests: bool = True,  # 是否记录请求日志
        start_engine_loop: bool = True,  # 是否启动引擎循环
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义统计日志器
        aggregate_engine_logging: bool = False,  # 是否聚合引擎日志
        client_addresses: dict[str, Any] | None = None,  # 客户端地址（多 API 服务器）
        client_count: int = 1,  # 客户端数量
        client_index: int = 0,  # 当前客户端索引
    ) -> None:
        """
        Create an AsyncLLM.
        # 创建 AsyncLLM

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # 参数说明：
        # vllm_config：全局配置
        # executor_class：执行器实现，如 MultiprocExecutor
        # log_stats：是否记录统计
        # usage_context：LLM 的使用场景
        # mm_registry：多模态注册表
        # log_requests：是否记录请求日志
        # start_engine_loop：是否启动引擎循环
        # stat_loggers：自定义统计日志器。如未提供，使用默认日志器。
        #   请注意 V1 中统计日志器不稳定，其基类接口可能变化。
        # 返回：None

        # Ensure we can serialize custom transformer configs
        # 确保可以序列化自定义 transformer 配置
        maybe_register_config_serialize_by_value()

        self.vllm_config = vllm_config  # 保存全局配置
        self.model_config = vllm_config.model_config  # 保存模型配置
        self.observability_config = vllm_config.observability_config  # 可观测配置

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        # 获取 OpenTelemetry 追踪端点
        if tracing_endpoint is not None:
            # 如果配置了追踪端点
            init_tracer("vllm.llm_engine", tracing_endpoint)
            # 初始化追踪器

        self.log_requests = log_requests  # 保存请求日志标志

        custom_stat_loggers = list(stat_loggers or [])
        # 复制自定义统计日志器列表
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())
        # 加载并追加插件提供的统计日志器

        has_custom_loggers = bool(custom_stat_loggers)  # 是否有自定义日志器
        self.log_stats = log_stats or has_custom_loggers
        # 记录统计：用户设置或存在自定义日志器
        if not log_stats and has_custom_loggers:
            # 如果用户未启用但存在自定义日志器
            logger.info(
                # 记录信息
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

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

        # EngineCore (starts the engine in background process).
        # 创建核心引擎（在后台进程中启动引擎）
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,  # 配置
            executor_class=executor_class,  # 执行器
            log_stats=self.log_stats,  # 日志统计
            client_addresses=client_addresses,  # 客户端地址
            client_count=client_count,  # 客户端数量
            client_index=client_index,  # 客户端索引
        )

        # Loggers.
        # 日志器
        self.logger_manager: StatLoggerManager | None = None  # 日志管理器
        if self.log_stats:
            # 如果启用了日志统计
            self.logger_manager = StatLoggerManager(
                # 创建统计日志管理器
                vllm_config=vllm_config,  # 配置
                engine_idxs=self.engine_core.engine_ranks_managed,  # 引擎索引
                custom_stat_loggers=custom_stat_loggers,  # 自定义日志器
                enable_default_loggers=log_stats,  # 默认日志器
                client_count=client_count,  # 客户端数量
                aggregate_engine_logging=aggregate_engine_logging,  # 聚合日志
            )
            self.logger_manager.log_engine_initialized()  # 记录引擎初始化

        self._client_count = client_count  # 保存客户端数量

        self.output_handler: asyncio.Task | None = None  # 输出处理任务
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            # 如果已在 asyncio 事件循环中，则立即启动输出处理器
            asyncio.get_running_loop()  # 获取当前运行的事件循环
            self._run_output_handler()  # 启动输出处理器
        except RuntimeError:
            # 如果不在事件循环中（初始化时）则跳过，稍后懒启动
            pass

        if (
            vllm_config.profiler_config.profiler == "torch"  # 使用 torch profiler
            and not vllm_config.profiler_config.ignore_frontend  # 不忽略前端
        ):
            # 启用 torch profiler 收集前端 CPU 追踪
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            # 获取 profiler 输出目录
            logger.info(
                # 记录 info 日志
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            # 生成 worker 名称：主机名 + PID + 标识
            self.profiler = torch.profiler.profile(
                # 创建 torch profiler
                activities=[  # 只收集 CPU 活动
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                # 是否包含栈信息
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    # trace 完成时的处理器
                    profiler_dir,  # 输出目录
                    worker_name=worker_name,  # worker 名称
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                    # 是否使用 gzip 压缩
                ),
            )
        else:
            self.profiler = None  # 否则不创建 profiler

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,  # vLLM 配置
        start_engine_loop: bool = True,  # 是否启动引擎循环
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用场景
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义日志器
        enable_log_requests: bool = False,  # 是否启用请求日志
        aggregate_engine_logging: bool = False,  # 是否聚合引擎日志
        disable_log_stats: bool = False,  # 是否禁用统计日志
        client_addresses: dict[str, Any] | None = None,  # 客户端地址
        client_count: int = 1,  # 客户端数量
        client_index: int = 0,  # 客户端索引
    ) -> "AsyncLLM":
        # 工厂方法：从 VllmConfig 创建
        # Create the LLMEngine.
        # 创建 LLMEngine
        return cls(
            vllm_config=vllm_config,  # 配置
            executor_class=Executor.get_class(vllm_config),  # 选择执行器
            start_engine_loop=start_engine_loop,  # 启动引擎循环
            stat_loggers=stat_loggers,  # 自定义日志器
            log_requests=enable_log_requests,  # 请求日志
            log_stats=not disable_log_stats,  # 日志统计（取反）
            aggregate_engine_logging=aggregate_engine_logging,  # 聚合日志
            usage_context=usage_context,  # 使用场景
            client_addresses=client_addresses,  # 客户端地址
            client_count=client_count,  # 客户端数量
            client_index=client_index,  # 客户端索引
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,  # 异步引擎命令行参数
        start_engine_loop: bool = True,  # 是否启动引擎循环
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,  # 使用场景
        stat_loggers: list[StatLoggerFactory] | None = None,  # 自定义日志器
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # 从 EngineArgs 创建 AsyncLLM

        # Create the engine configs.
        # 创建引擎配置
        vllm_config = engine_args.create_engine_config(usage_context)
        # 从参数创建配置
        executor_class = Executor.get_class(vllm_config)  # 选择执行器

        # Create the AsyncLLM.
        # 创建 AsyncLLM
        return cls(
            vllm_config=vllm_config,  # 配置
            executor_class=executor_class,  # 执行器
            log_requests=engine_args.enable_log_requests,  # 请求日志
            log_stats=not engine_args.disable_log_stats,  # 日志统计
            start_engine_loop=start_engine_loop,  # 启动引擎循环
            usage_context=usage_context,  # 使用场景
            stat_loggers=stat_loggers,  # 自定义日志器
        )

    def __del__(self):
        # 析构函数：对象被 GC 时自动关闭引擎
        self.shutdown()  # 关闭所有资源

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown, cleaning up the background proc and IPC."""
        # 关闭：清理后台进程和 IPC
        shutdown_prometheus()  # 1. 关闭 Prometheus 指标

        if renderer := getattr(self, "renderer", None):
            # 2. 关闭 renderer（防御性访问）
            renderer.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            # 3. 关闭核心引擎进程 + ZMQ（防御性访问）
            engine_core.shutdown(timeout=timeout)

        handler = getattr(self, "output_handler", None)
        # 4. 获取输出处理任务（防御性访问）
        if handler is not None:
            # 4. 取消后台输出处理任务
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 异步获取支持的任务类型
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            # 缓存结果（避免重复查询）
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()
            # 从核心引擎查询

        return self._supported_tasks  # 返回缓存结果

    async def add_request(
        self,
        request_id: str,  # 用户提供的请求 ID
        prompt: EngineCoreRequest  # 输入类型之一：
        | PromptType  # 原始 prompt
        | EngineInput  # 预处理输入
        | AsyncGenerator[StreamingInput, None],  # 流式输入
        params: SamplingParams | PoolingParams,  # 采样/池化参数
        arrival_time: float | None = None,  # 到达时间（可选）
        lora_request: LoRARequest | None = None,  # LoRA（可选）
        tokenization_kwargs: dict[str, Any] | None = None,  # tokenize 参数（可选）
        trace_headers: Mapping[str, str] | None = None,  # 追踪头（可选）
        priority: int = 0,  # 优先级
        data_parallel_rank: int | None = None,  # 目标 DP rank（可选）
        prompt_text: str | None = None,  # prompt 文本（可选）
        reasoning_ended: bool | None = None,  # 推理阶段是否结束（可选）
        reasoning_parser_kwargs: dict[str, Any] | None = None,  # 推理解析参数（可选）
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""
        # 向 AsyncLLM 添加新请求

        if self.errored:
            # 如果引擎已出错
            raise EngineDeadError()  # 抛出引擎死亡错误

        is_pooling = isinstance(params, PoolingParams)  # 是否为池化请求

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            # 启用了 KV 共享快速 prefill
            and not is_pooling  # 非池化
            and params.prompt_logprobs  # 请求了 prompt logprobs
        ):
            # KV 共享快速 prefill 与 prompt logprobs 不兼容
            raise ValueError(
                # 抛出错误
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        if isinstance(prompt, AsyncGenerator):
            # 如果是流式输入
            if reasoning_ended is not None or reasoning_parser_kwargs is not None:
                # 流式输入不支持推理阶段控制
                raise NotImplementedError  # 抛出未实现错误

            # Streaming input case.
            # 流式输入情况
            return await self._add_streaming_input_request(
                # 调用流式输入处理方法
                request_id,  # 请求 ID
                prompt,  # 输入流
                params,  # 参数
                arrival_time,  # 到达时间
                lora_request,  # LoRA
                tokenization_kwargs,  # tokenize 参数
                trace_headers,  # 追踪头
                priority,  # 优先级
                data_parallel_rank,  # DP rank
            )

        # Convert Input --> Request.
        # 将输入转换为请求
        if isinstance(prompt, EngineCoreRequest):
            # 如果输入已是 EngineCoreRequest（已预处理）
            logger.warning_once(
                # 记录一次性废弃警告
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt  # 直接使用
            if request_id != request.request_id:
                # 如果传入 ID 不一致
                logger.warning_once(
                    # 记录一次性警告
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            request = self.input_processor.process_inputs(
                # 通过输入处理器处理
                request_id,  # 请求 ID
                prompt,  # 原始输入
                params,  # 参数
                supported_tasks=await self.get_supported_tasks(),  # 支持的任务
                arrival_time=arrival_time,  # 到达时间
                lora_request=lora_request,  # LoRA
                tokenization_kwargs=tokenization_kwargs,  # tokenize 参数
                trace_headers=trace_headers,  # 追踪头
                priority=priority,  # 优先级
                data_parallel_rank=data_parallel_rank,  # DP rank
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)
            # 提取 prompt 组件（文本部分）

        if reasoning_ended is not None:
            # 如果设置了推理结束标志
            request.reasoning_ended = reasoning_ended  # 设置到请求
        if reasoning_parser_kwargs is not None:
            # 如果设置了推理解析参数
            request.reasoning_parser_kwargs = reasoning_parser_kwargs  # 设置到请求

        self.input_processor.assign_request_id(request)
        # 分配内部唯一请求 ID

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        # 我们在第一次调用 add_request() 时启动 output_handler，
        # 因此可以在事件循环之前调用 __init__，这使得我们能够在
        # OpenAI 服务器中优雅地处理启动失败。
        self._run_output_handler()  # 确保输出处理器在运行

        # Create a new output collector for the request.
        # 为请求创建新的输出收集器
        queue = RequestOutputCollector(params.output_kind, request.request_id)
        # 创建 per-request 输出队列

        # Use cloned params that may have been updated in process_inputs()
        # 使用可能在 process_inputs() 中更新过的克隆参数
        params = request.params  # 获取处理后的参数

        if is_pooling or params.n == 1:
            # 如果是池化请求或 n=1
            await self._add_request(request, prompt_text, None, 0, queue)
            # 单个请求直接发送
            return queue  # 返回输出收集器

        parent_params = params  # 父参数
        assert isinstance(parent_params, SamplingParams)  # 断言是采样参数

        # Fan out child requests (for n>1).
        # 并行采样（n>1）：拆分为多个子请求
        parent_request = ParentRequest(request)  # 创建父请求管理器
        for idx in range(parent_params.n):
            # 遍历每个子请求索引
            request_id, child_params = parent_request.get_child_info(idx)
            # 获取子请求 ID 和参数
            child_request = request if idx == parent_params.n - 1 else copy(request)
            # 最后一个子请求复用原对象（避免多余拷贝）
            child_request.request_id = request_id  # 设置子请求 ID
            child_request.sampling_params = child_params  # 设置子请求参数
            await self._add_request(  # 发送子请求
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue  # 返回输出收集器

    async def _add_request(
        self,
        request: EngineCoreRequest,  # 引擎核心请求
        prompt: str | None,  # prompt 文本（可选）
        parent_req: ParentRequest | None,  # 父请求（可选）
        index: int,  # 子请求索引
        queue: RequestOutputCollector,  # 输出收集器
    ):
        # 添加请求到前端处理器和核心引擎
        # Add the request to OutputProcessor (this process).
        # 在 OutputProcessor（本进程）中注册请求
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        # 将 EngineCoreRequest 添加到 EngineCore（独立进程）
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            # 如果启用了请求日志
            logger.info("Added request %s.", request.request_id)  # 记录日志

    async def _add_streaming_input_request(
        self,
        request_id: str,  # 请求 ID
        input_stream: AsyncGenerator[StreamingInput, None],  # 输入生成器
        sampling_params: SamplingParams | PoolingParams,  # 参数
        arrival_time: float | None = None,  # 到达时间
        lora_request: LoRARequest | None = None,  # LoRA
        tokenization_kwargs: dict[str, Any] | None = None,  # tokenize 参数
        trace_headers: Mapping[str, str] | None = None,  # 追踪头
        priority: int = 0,  # 优先级
        data_parallel_rank: int | None = None,  # DP rank
    ) -> RequestOutputCollector:
        # 处理流式输入请求
        self._validate_streaming_input_sampling_params(sampling_params)
        # 验证流式输入的采样参数

        inputs = dict(
            # 构建公共输入参数字典
            supported_tasks=await self.get_supported_tasks(),  # 支持的任务
            arrival_time=arrival_time,  # 到达时间
            lora_request=lora_request,  # LoRA
            tokenization_kwargs=tokenization_kwargs,  # tokenize 参数
            trace_headers=trace_headers,  # 追踪头
            priority=priority,  # 优先级
            data_parallel_rank=data_parallel_rank,  # DP rank
        )

        if not sampling_params.skip_clone:
            # 如果尚未跳过克隆
            sampling_params = sampling_params.clone()  # 克隆参数
            sampling_params.skip_clone = True  # 标记跳过克隆

        # Create request for validation, also used as the finished signal
        # once the input stream is closed.
        # 创建验证用请求，也用作输入流关闭时的完成信号
        final_req = self.input_processor.process_inputs(
            request_id=request_id,  # 请求 ID
            prompt=TokensPrompt(prompt_token_ids=[0]),  # 占位 prompt
            params=sampling_params,  # 参数
            **inputs,  # type: ignore[arg-type]
            # 解包公共参数
        )
        self.input_processor.assign_request_id(final_req)  # 分配内部 ID
        internal_req_id = final_req.request_id  # 获取内部请求 ID

        queue = RequestOutputCollector(sampling_params.output_kind, internal_req_id)
        # 创建输出收集器

        async def handle_inputs():
            # 内部协程：处理输入流
            cancelled = False  # 是否被取消
            try:
                async for input_chunk in input_stream:
                    # 遍历输入流中的每个块
                    sp = input_chunk.sampling_params  # 获取块的采样参数
                    if sp:
                        # 如果块有自己的参数
                        self._validate_streaming_input_sampling_params(sp)
                        # 验证参数
                    else:
                        sp = sampling_params  # 否则使用默认参数
                    # TODO(nick): Avoid re-validating reused sampling parameters
                    # TODO(nick)：避免重新验证复用的采样参数
                    req = self.input_processor.process_inputs(
                        # 处理每个输入块
                        request_id=internal_req_id,  # 请求 ID
                        prompt=input_chunk.prompt,  # 块的 prompt
                        params=sp,  # 块的参数
                        resumable=True,  # 标记可续传
                        **inputs,  # type: ignore[arg-type]
                        # 解包公共参数
                    )
                    req.external_req_id = request_id  # 设置外部 ID
                    if req.prompt_embeds is not None:
                        # 流式输入不支持 embedding
                        raise ValueError(
                            "prompt_embeds not supported for streaming inputs"
                        )
                    prompt_text, _, _ = extract_prompt_components(
                        self.model_config, input_chunk.prompt
                    )
                    # 提取 prompt 文本
                    await self._add_request(req, prompt_text, None, 0, queue)
                    # 发送每个块作为子请求
            except (asyncio.CancelledError, GeneratorExit):
                # 如果被取消
                cancelled = True  # 标记取消
            except Exception as error:
                # Wrap in InputStreamError so generate() can propagate it
                # without wrapping in EngineGenerateError.
                # 包装在 InputStreamError 中，使 generate() 可以不包装在
                # EngineGenerateError 中直接传播。
                queue.put(InputStreamError(error))  # 推送错误到队列
            finally:
                queue._input_stream_task = None  # 清除任务引用
                if not cancelled:
                    # 如果未被取消
                    # Send empty final request to indicate that inputs have
                    # finished. Don't send if cancelled (session was aborted).
                    # 发送空的最终请求表示输入已完成。
                    # 如果已取消（会话中止）则不发送。
                    await self._add_request(final_req, None, None, 0, queue)
                    # 发送最终请求

        # Ensure output handler is running.
        # 确保输出处理器在运行
        self._run_output_handler()

        queue._input_stream_task = asyncio.create_task(handle_inputs())
        # 启动输入流处理任务
        return queue  # 返回输出收集器

    @staticmethod
    def _validate_streaming_input_sampling_params(
        params: SamplingParams | PoolingParams,  # 采样/池化参数
    ):
        # 验证流式输入的采样参数
        if (
            not isinstance(params, SamplingParams)  # 必须是采样参数
            or params.n > 1  # 不支持 n>1
            or params.output_kind == RequestOutputKind.FINAL_ONLY  # 不支持最终输出
            or params.stop  # 不支持 stop strings
        ):
            # 流式输入不支持这些参数组合
            raise ValueError(
                # 抛出错误
                "Input streaming not currently supported "
                "for pooling models, n > 1, request_kind = FINAL_ONLY "
                "or with stop strings."
            )

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    # TODO：我们应该支持一次调用多个 prompt，如 LLM.generate。
    # 这样多 prompt 完成请求无需向核心进程发送多条消息，
    # 也无需多个流再在 API 服务器中重新多路复用。
    async def generate(
        self,
        prompt: EngineCoreRequest  # 输入类型之一：
        | PromptType  # 原始 prompt
        | EngineInput  # 预处理输入
        | AsyncGenerator[StreamingInput, None],  # 流式输入
        sampling_params: SamplingParams,  # 采样参数
        request_id: str,  # 请求 ID
        *,
        prompt_text: str | None = None,  # prompt 文本（可选）
        lora_request: LoRARequest | None = None,  # LoRA（可选）
        tokenization_kwargs: dict[str, Any] | None = None,  # tokenize 参数（可选）
        trace_headers: Mapping[str, str] | None = None,  # 追踪头（可选）
        priority: int = 0,  # 优先级
        data_parallel_rank: int | None = None,  # DP rank（可选）
        reasoning_ended: bool | None = None,  # 推理结束标志（可选）
        reasoning_parser_kwargs: dict[str, Any] | None = None,  # 推理解析参数（可选）
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """
        # API 服务器调用的主要函数，用于启动请求：
        # 1) 创建与请求对应的 AsyncStream
        # 2) 处理输入
        # 3) 将请求添加到 Detokenizer
        # 4) 将请求添加到 EngineCore（独立进程）
        # 一个独立的 output_handler 循环在后台 AsyncIO 任务中运行，
        # 从 EngineCore 拉取输出并放入 per-request AsyncStream。
        # generate() 的调用者迭代返回的 AsyncGenerator，
        # 将 RequestOutput 返回给调用者。

        q: RequestOutputCollector | None = None  # 输出收集器
        try:
            q = await self.add_request(  # 添加请求
                request_id,  # 请求 ID
                prompt,  # 输入
                sampling_params,  # 参数
                lora_request=lora_request,  # LoRA
                tokenization_kwargs=tokenization_kwargs,  # tokenize 参数
                trace_headers=trace_headers,  # 追踪头
                priority=priority,  # 优先级
                data_parallel_rank=data_parallel_rank,  # DP rank
                prompt_text=prompt_text,  # prompt 文本
                reasoning_ended=reasoning_ended,  # 推理结束
                reasoning_parser_kwargs=reasoning_parser_kwargs,  # 推理解析
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            # output_handler 任务将数据推入队列。
            # 本任务从队列拉取并 yield 给调用者。
            finished = False  # 完成标志
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                # 注意：尽可能不用 await 排空队列（避免高负载下任务切换，有助于性能）
                out = q.get_nowait() or await q.get()
                # 先尝试非阻塞获取，为空则等待

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                # 注意：OutputProcessor 和 EngineCore 都会根据完成状态
                # 自行清理请求。
                assert isinstance(out, RequestOutput)  # 断言输出类型
                finished = out.finished  # 获取完成状态
                if out is not STREAM_FINISHED:
                    # 如果不是流结束哨兵
                    yield out  # 产出输出

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        # 如果客户端断开连接、generate() 被取消或生成器被垃圾回收，
        # 则中止请求。
        except (asyncio.CancelledError, GeneratorExit):
            # 捕获取消和生成器退出
            if q is not None:
                # 如果有输出收集器
                await self.abort(q.request_id, internal=True)  # 内部中止
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s aborted.", request_id)  # 记录日志
            raise  # 重新抛出

        # Engine is dead. Do not abort since we shut down.
        # 引擎已死亡。不中止请求，因为正在关闭。
        except EngineDeadError:
            # 捕获引擎死亡错误
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s failed (engine dead).", request_id)  # 记录
            raise  # 重新抛出

        # Request validation error.
        # 请求验证错误
        except ValueError as e:
            # 捕获值错误
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s failed (bad request): %s.", request_id, e)
                # 记录日志
            raise  # 重新抛出

        # Error from input stream generator - propagate directly.
        # 来自输入流生成器的错误 - 直接传播
        except InputStreamError as e:
            # 捕获输入流错误
            if q is not None:
                # 如果有输出收集器
                await self.abort(q.request_id, internal=True)  # 内部中止
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s failed (input error): %s.", request_id, e)
                # 记录日志
            raise e.cause from e  # 抛出原始异常

        # Unexpected error in the generate() task (possibly recoverable).
        # generate() 任务中的意外错误（可能可恢复）
        except Exception as e:
            # 捕获所有异常
            if q is not None:
                # 如果有输出收集器
                await self.abort(q.request_id, internal=True)  # 内部中止
            if self.log_requests:
                # 如果启用了请求日志
                try:
                    s = f"{e.__class__.__name__}: {e}"  # 格式化错误
                except Exception as e2:
                    # 如果格式化也失败
                    s = (
                        f"{e.__class__.__name__}: "
                        "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                    # 备用格式化
                logger.info("Request %s failed due to %s.", request_id, s)  # 记录
            raise EngineGenerateError() from e  # 包装为引擎生成错误
        finally:
            if q is not None:
                # 如果有输出收集器
                q.close()  # 关闭收集器

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""
        # 后台循环：从 EngineCore 拉取输出并推送到 AsyncStreams

        if self.output_handler is not None:
            # 如果输出处理器已在运行
            return  # 直接返回（幂等）

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        # 确保任务不会循环引用回 AsyncLLM 对象，
        # 否则它不会被垃圾回收和正确清理。
        engine_core = self.engine_core  # 本地引用核心引擎
        output_processor = self.output_processor  # 本地引用输出处理器
        log_stats = self.log_stats  # 本地引用日志统计标志
        # We use a mutable list for logger_manager so that it can be updated
        # during elastic EP scaling (see scale_elastic_ep) without creating
        # a circular reference via self.
        # 对 logger_manager 使用可变列表，以便在弹性 EP 缩放期间更新
        # （见 scale_elastic_ep），而不会通过 self 创建循环引用。
        self._logger_ref = [self.logger_manager]  # 日志管理器引用（可变包装）
        logger_ref = self._logger_ref  # 本地引用
        renderer = self.renderer  # 本地引用 renderer
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE  # 分块大小

        async def output_handler():
            # 内部协程：输出处理循环
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    # 1) 从核心引擎拉取输出
                    outputs = await engine_core.get_output_async()
                    # 获取 EngineCoreOutputs
                    num_outputs = len(outputs.outputs)  # 输出数量

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )
                    # 创建迭代统计（如启用日志且有输出）

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    # 将输出分成最多 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE 的块，
                    # 避免长时间阻塞事件循环。
                    engine_core_outputs = outputs.outputs  # 输出列表
                    for start in range(0, num_outputs, chunk_size):
                        # 分块遍历
                        end = start + chunk_size  # 块结束
                        outputs_slice = engine_core_outputs[start:end]  # 切片
                        # 2) Process EngineCoreOutputs.
                        # 2) 处理 EngineCoreOutputs
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # 处理输出并分发到各请求队列
                        # NOTE: RequestOutputs are pushed to their queues.
                        # 注意：RequestOutput 已推送到各自的队列
                        assert not processed_outputs.request_outputs  # 断言为空

                        # Allow other asyncio tasks to run between chunks
                        # 允许其他 asyncio 任务在块之间运行
                        if end < num_outputs:
                            await asyncio.sleep(0)  # 让出控制权

                        # 3) Abort any reqs that finished due to stop strings.
                        # 3) 中止因 stop strings 结束的请求
                        if processed_outputs.reqs_to_abort:
                            # 如果有需要中止的请求
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )
                            # 向核心引擎发送中止

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)
                    # 更新调度器统计

                    # 4) Logging.
                    # 4) 记录日志
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    # TODO(rob)：在 Prometheus 开销不可忽略时改为协程并在后台线程中运行
                    if logger_ref[0]:
                        # 如果有日志管理器
                        logger_ref[0].record(
                            engine_idx=outputs.engine_index,  # 引擎索引
                            scheduler_stats=outputs.scheduler_stats,  # 调度统计
                            iteration_stats=iteration_stats,  # 迭代统计
                            mm_cache_stats=renderer.stat_mm_cache(),  # 多模态缓存
                        )
            except Exception as e:
                # 捕获所有异常
                logger.exception("AsyncLLM output_handler failed.")  # 记录异常
                output_processor.propagate_error(e)  # 通知所有等待的请求出错

        self.output_handler = asyncio.create_task(output_handler())
        # 启动后台任务

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""
        # 在 OutputProcessor 和 EngineCore 中中止请求

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        # 规范化输入为元组/列表
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        # 在前端输出处理器中止，返回需要发给核心的 ID
        await self.engine_core.abort_requests_async(all_request_ids)
        # 在核心引擎中止

        if self.log_requests:
            # 如果启用了请求日志
            logger.info("Aborted request(s) %s.", ",".join(request_ids))  # 记录日志

    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,  # 请求 ID
        kv_transfer_params: dict[str, Any],  # KV 传输参数
        *,
        data_parallel_rank: int | None = None,  # DP rank（可选）
    ) -> None:
        """Submit a pre-aborted request so the connector's request_finished
        hook runs to free any pre-admission KV-transfer resources (e.g. NIXL
        prefill blocks pinned on the P node)."""
        # 提交一个预中止的请求，使 connector 的 request_finished 钩子运行，
        # 释放任何预准入的 KV 传输资源（如 P 节点上固定的 NIXL prefill 块）。
        request = EngineCoreRequest(  # 构建特殊请求
            request_id=request_id,  # 请求 ID
            prompt_token_ids=[0],  # 占位 token
            mm_features=None,  # 无多模态
            sampling_params=SamplingParams(  # 采样参数
                max_tokens=1,  # 仅 1 个 token
                extra_args={"kv_transfer_params": dict(kv_transfer_params)},
                # 传递 KV 传输参数
            ),
            pooling_params=None,  # 无池化参数
            arrival_time=time.time(),  # 到达时间
            lora_request=None,  # 无 LoRA
            cache_salt=None,  # 无缓存 salt
            data_parallel_rank=data_parallel_rank,  # DP rank
            abort_immediately=True,  # 添加后立即中止
        )
        await self.engine_core.add_request_async(request)
        # 发送到核心引擎

    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",  # 暂停模式（默认 abort）
        wait_for_inflight_requests: bool | None = None,  # 已废弃参数
        clear_cache: bool = True,  # 是否清空缓存
    ) -> None:
        """
        Pause generation to allow model weight updates.

        All mode handling (abort / wait / keep) and cache clearing is done
        in the engine. New generation/encoding requests will not be scheduled
        until resume is called.

        Args:
            mode: How to handle in-flight requests:
                - ``"abort"``: Abort all in-flight requests immediately
                  (default).
                - ``"wait"``: Wait for in-flight requests to complete.
                - ``"keep"``: Freeze requests in queue; they resume on
                  :meth:`resume_generation`.
            wait_for_inflight_requests: DEPRECATED: use mode argument.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
        """
        # 暂停生成以允许模型权重更新。
        # 所有模式处理（abort / wait / keep）和缓存清理都在引擎中完成。
        # 在调用 resume 前，新的生成/编码请求不会被调度。
        # 参数说明：
        # mode：如何处理进行中的请求：
        #   - "abort"：立即中止所有进行中请求（默认）
        #   - "wait"：等待进行中请求完成
        #   - "keep"：冻结队列中的请求；恢复时继续
        # wait_for_inflight_requests：已废弃，请使用 mode 参数
        # clear_cache：排空后是否清空 KV 缓存和前缀缓存。
        #   设为 False 保留缓存以加速恢复。
        if wait_for_inflight_requests:
            # 如果使用了已废弃参数
            warnings.warn(
                # 发出废弃警告
                "The `wait_for_inflight_requests` parameter in "
                "`AsyncLLM.pause_generation()` is deprecated. "
                "Please use `mode` argument instead.",
                DeprecationWarning,  # 废弃警告类型
                stacklevel=2,  # 栈级别
            )
            mode = "wait"  # 映射到 wait 模式
        if clear_cache:
            # 如果需要清空缓存
            await self.renderer.clear_mm_cache_async()  # 清空前端多模态缓存
        await self.engine_core.pause_scheduler_async(mode=mode, clear_cache=clear_cache)
        # 暂停核心引擎调度器
        # Small sleep to help ensure that final outputs from any in-flight requests are
        # returned prior to this method returning. These outputs come out of the engine
        # prior to the wait-for-idle completion event, but involve additional async
        # tasks in output processing.
        # Note that this is not required for correctness, just more intuitive ordering
        # of events from caller's pov.
        # 短暂休眠确保任何进行中请求的最终输出在方法返回前返回。
        # 这些输出在等待空闲完成事件之前从引擎中出来，
        # 但涉及输出处理中的额外异步任务。
        # 注意：这对正确性不是必需的，只是从调用者角度看更直观的事件排序。
        await asyncio.sleep(0.02)  # 等待 20ms

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""
        # 在 pause_generation() 之后恢复生成
        await self.engine_core.resume_scheduler_async()  # 恢复调度器

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""
        # 返回引擎当前是否暂停
        return await self.engine_core.is_scheduler_paused_async()
        # 查询核心引擎

    async def encode(
        self,
        prompt: PromptType | EngineInput,  # 输入类型
        pooling_params: PoolingParams,  # 池化参数
        request_id: str,  # 请求 ID
        lora_request: LoRARequest | None = None,  # LoRA（可选）
        trace_headers: Mapping[str, str] | None = None,  # 追踪头（可选）
        priority: int = 0,  # 优先级
        tokenization_kwargs: dict[str, Any] | None = None,  # tokenize 参数（可选）
        reasoning_ended: bool | None = None,  # 推理结束标志（可选）
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """
        # API 服务器调用的主要函数（池化任务）：
        # 1) 创建与请求对应的 AsyncStream
        # 2) 处理输入
        # 3) 将请求添加到 EngineCore（独立进程）
        # 一个独立的 output_handler 循环在后台 AsyncIO 任务中运行，
        # 从 EngineCore 拉取输出并放入 per-request AsyncStream。
        # generate() 的调用者迭代返回的 AsyncGenerator，
        # 将 RequestOutput 返回给调用者。

        q: RequestOutputCollector | None = None  # 输出收集器
        try:
            q = await self.add_request(  # 添加请求
                request_id,  # 请求 ID
                prompt,  # 输入
                pooling_params,  # 池化参数
                lora_request=lora_request,  # LoRA
                tokenization_kwargs=tokenization_kwargs,  # tokenize 参数
                trace_headers=trace_headers,  # 追踪头
                priority=priority,  # 优先级
                reasoning_ended=reasoning_ended,  # 推理结束
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            # output_handler 任务将数据推入队列。
            # 本任务从队列拉取并 yield 给调用者。
            finished = False  # 完成标志
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                # 注意：尽可能不用 await 排空队列（避免高负载下任务切换）
                out = q.get_nowait() or await q.get()
                # 先尝试非阻塞获取，为空则等待
                assert isinstance(out, PoolingRequestOutput)  # 断言输出类型
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                # 注意：OutputProcessor 和 EngineCore 都会根据完成状态
                # 自行清理请求。
                finished = out.finished  # 获取完成状态
                yield out  # 产出输出

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        # 如果客户端断开连接，generate() 被取消。因此中止请求。
        except asyncio.CancelledError:
            # 捕获取消
            if q is not None:
                # 如果有输出收集器
                await self.abort(q.request_id, internal=True)  # 内部中止
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s aborted.", request_id)  # 记录日志
            raise  # 重新抛出

        # Engine is dead. Do not abort since we shut down.
        # 引擎已死亡。不中止请求，因为正在关闭。
        except EngineDeadError:
            # 捕获引擎死亡错误
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s failed (engine dead).", request_id)  # 记录
            raise  # 重新抛出

        # Request validation error.
        # 请求验证错误
        except ValueError:
            # 捕获值错误
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s failed (bad request).", request_id)  # 记录
            raise  # 重新抛出

        # Unexpected error in the generate() task (possibly recoverable).
        # generate() 任务中的意外错误（可能可恢复）
        except Exception as e:
            # 捕获所有异常
            if q is not None:
                # 如果有输出收集器
                await self.abort(q.request_id, internal=True)  # 内部中止
            if self.log_requests:
                # 如果启用了请求日志
                logger.info("Request %s failed.", request_id)  # 记录日志
            raise EngineGenerateError() from e  # 包装为引擎生成错误
        finally:
            if q is not None:
                # 如果有输出收集器
                q.close()  # 关闭收集器

    @property
    def tokenizer(self) -> TokenizerLike | None:
        # 属性：获取 tokenizer（可能为 None）
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        # 获取 tokenizer 实例（保证非 None）
        return self.renderer.get_tokenizer()

    async def is_tracing_enabled(self) -> bool:
        # 检查是否启用追踪
        return self.observability_config.otlp_traces_endpoint is not None
        # 是否配置了追踪端点

    async def do_log_stats(self) -> None:
        # 手动触发日志记录
        if self.logger_manager:
            # 如果有日志管理器
            self.logger_manager.log()  # 记录日志

    async def check_health(self) -> None:
        # 健康检查
        logger.debug("Called check_health.")  # 记录调试日志
        if self.errored:
            # 如果引擎出错
            raise self.dead_error  # 抛出死亡错误

    async def start_profile(self, profile_prefix: str | None = None) -> None:
        # 开始性能分析
        coros = [self.engine_core.profile_async(True, profile_prefix)]  # 核心 profiler
        if self.profiler is not None:
            # 如果前端也有 profiler
            coros.append(asyncio.to_thread(self.profiler.start))  # 线程中启动
        await asyncio.gather(*coros)  # 并发执行

    async def stop_profile(self) -> None:
        # 停止性能分析
        coros = [self.engine_core.profile_async(False)]  # 核心 profiler
        if self.profiler is not None:
            # 如果前端也有 profiler
            coros.append(asyncio.to_thread(self.profiler.stop))  # 线程中停止
        await asyncio.gather(*coros)  # 并发执行

    async def reset_mm_cache(self) -> None:
        # 重置多模态缓存
        await self.renderer.clear_mm_cache_async()  # 清空前端缓存
        await self.engine_core.reset_mm_cache_async()  # 清空核心缓存

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 重置前缀缓存
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )
        # 委托给核心引擎

    async def reset_encoder_cache(self) -> None:
        # 重置编码器缓存
        await self.engine_core.reset_encoder_cache_async()  # 委托给核心引擎

    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 引擎休眠（释放 GPU 内存）
        if level >= 1:
            # 如果休眠级别 >=1
            await self.renderer.clear_mm_cache_async()  # 清空多模态缓存
        await self.engine_core.sleep_async(level, mode)  # 通知核心引擎

        if self.logger_manager is not None:
            # 如果有日志管理器
            self.logger_manager.record_sleep_state(1, level)  # 记录休眠状态

    async def wake_up(self, tags: list[str] | None = None) -> None:
        # 引擎唤醒
        await self.engine_core.wake_up_async(tags)  # 通知核心引擎

        if self.logger_manager is not None:
            # 如果有日志管理器
            self.logger_manager.record_sleep_state(0, 0)  # 记录唤醒状态

    async def checkpoint_prepare(self) -> None:
        # 检查点准备（容错）
        await self.collective_rpc("checkpoint_prepare")  # 集体 RPC

    async def checkpoint_restore(self) -> None:
        # 检查点恢复（容错）
        await self.collective_rpc("checkpoint_restore")  # 集体 RPC

    async def is_sleeping(self) -> bool:
        # 检查引擎是否在休眠
        return await self.engine_core.is_sleeping_async()  # 委托给核心引擎

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        # 加载新的 LoRA 适配器供后续请求使用
        return await self.engine_core.add_lora_async(lora_request)  # 委托核心

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        # 移除已加载的 LoRA 适配器
        return await self.engine_core.remove_lora_async(lora_id)  # 委托核心

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        # 列出所有已注册的适配器
        return await self.engine_core.list_loras_async()  # 委托核心

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        # 防止适配器被逐出（固定）
        return await self.engine_core.pin_lora_async(lora_id)  # 委托核心

    async def collective_rpc(
        self,
        method: str,  # 调用的方法名
        timeout: float | None = None,  # 超时（可选）
        args: tuple = (),  # 位置参数
        kwargs: dict | None = None,  # 关键字参数
    ):
        """
        Perform a collective RPC call to the given path.
        """
        # 对给定路径执行集体 RPC 调用
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )
        # 委托给核心引擎

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        # 等待所有请求排空
        start_time = time.time()  # 起始时间
        while time.time() - start_time < drain_timeout:
            # 循环直到超时
            if not self.engine_core.dp_engines_running():
                # 如果引擎已空闲
                logger.info("Engines are idle, requests have been drained")
                # 记录日志
                return  # 返回

            logger.info("Engines are still running, waiting for requests to drain...")
            # 记录等待日志
            await asyncio.sleep(1)  # 等待 1 秒再检查

        raise TimeoutError(  # 超时则抛出错误
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def _drain_requests_for_elastic_ep(self, drain_timeout: int) -> None:
        try:
            logger.info(
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "
                "waiting for requests to drain before scaling"
            )
            await self.wait_for_requests_to_drain(drain_timeout)
        except BaseException:
            set_scaling_elastic_ep(False)
            raise

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """
        Scale up or down the data parallel size by adding or removing
        engine cores.
        Args:
            new_data_parallel_size: The new number of data parallel workers
            drain_timeout:
                Maximum time to wait for requests to drain (seconds)
        """
        # 通过添加或移除引擎核心来扩展或缩减数据并行大小。
        # 参数：new_data_parallel_size 新的 DP worker 数量；
        # drain_timeout 等待请求排空的最大时间（秒）。
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        # 获取当前 DP 大小
        if old_data_parallel_size == new_data_parallel_size:
            # 如果大小相同
            logger.info(
                # 记录日志并跳过
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return  # 直接返回

        if envs.VLLM_ELASTIC_EP_DRAIN_REQUESTS:
            # 如果配置了先排空请求
            logger.info(
                # 记录日志
                "VLLM_ELASTIC_EP_DRAIN_REQUESTS is set, "
                "waiting for requests to drain before scaling"
            )
            await self.wait_for_requests_to_drain(drain_timeout)
            # 等待请求排空

        # recreate stat loggers
        # 重建统计日志器
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # 扩容且启用日志统计
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            # TODO(rob)：与 Ray 团队讨论后修复。
            # 这会重置所有 Prometheus 指标，因为我们在初始化时注销。
            # 需要更好地理解这里的预期行为。
            self.logger_manager = StatLoggerManager(
                # 重新创建统计日志管理器
                vllm_config=self.vllm_config,  # 配置
                engine_idxs=list(range(new_data_parallel_size)),  # 新引擎索引
                custom_stat_loggers=None,  # 无自定义日志器
            )
            # Update the mutable ref so output_handler picks up the
            # new logger without creating a circular reference via self.
            # 更新可变引用，使 output_handler 拾取新日志器，而不通过 self
            # 创建循环引用。
            if hasattr(self, "_logger_ref"):
                # 如果有引用列表
                self._logger_ref[0] = self.logger_manager  # 更新引用
            self.logger_manager.log_engine_initialized()  # 记录初始化

        set_scaling_elastic_ep(True)  # 标记缩放中
        try:
            await self.engine_core.scale_elastic_ep(new_data_parallel_size)
            # 通知核心引擎缩放
            self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            # 更新配置中的 DP 大小
        finally:
            set_scaling_elastic_ep(False)  # 清除缩放标记

    async def handle_fault(
        self, fault_tolerance_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """send fault tolerance instruction to the engine"""
        # 向引擎发送故障容错指令
        return await self.engine_core.handle_fault(fault_tolerance_request)
        # 委托给核心引擎

    async def get_status(self):
        # 获取引擎状态（容错）
        return await self.engine_core.get_status()  # 委托给核心引擎

    @property
    def is_running(self) -> bool:
        # 属性：引擎是否在运行
        # Is None before the loop is started.
        # 循环启动前为 None
        return self.output_handler is None or not self.output_handler.done()
        # 输出处理器未启动或未完成

    @property
    def is_stopped(self) -> bool:
        # 属性：引擎是否已停止
        return self.errored  # 引擎出错即停止

    @property
    def errored(self) -> bool:
        # 属性：引擎是否出错
        return self.engine_core.resources.engine_dead or not self.is_running
        # 核心引擎死亡或不在运行

    @property
    def dead_error(self) -> BaseException:
        # 属性：引擎死亡错误
        return EngineDeadError()  # 返回引擎死亡错误实例

    async def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest
    ) -> None:
        """
        Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info
        """
        # 为 RL 训练初始化权重传输。
        # 参数：request 带后端特定信息的权重传输初始化请求
        await self.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": request.init_info}
        )
        # 集体 RPC 初始化

    async def start_weight_update(self) -> None:
        """Start a new weight update."""
        # 开始新的权重更新
        await self.collective_rpc("start_weight_update")  # 集体 RPC

    async def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        # 开始针对投机草稿模型的新权重更新
        await self.collective_rpc("start_draft_weight_update")  # 集体 RPC

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """
        Batched weight update for RL training.

        Args:
            request: Weight update request with backend-specific update info
        """
        # 用于 RL 训练的批量权重更新。
        # 参数：request 带后端特定更新信息的权重更新请求
        await self.collective_rpc(
            "update_weights", kwargs={"update_info": request.update_info}
        )
        # 集体 RPC 更新权重

    async def finish_weight_update(self) -> None:
        """Finish the current weight update."""
        # 完成当前权重更新
        await self.collective_rpc("finish_weight_update")  # 集体 RPC