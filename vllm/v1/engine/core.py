# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import gc  # gc：垃圾回收模块（gc.freeze/unfreeze 管理 GC 堆）
import os  # os：操作系统接口（环境变量读取）
import queue  # queue：同步队列（aborts_queue 用）
import signal  # signal：信号处理（SIGTERM/SIGINT 优雅关闭）
import threading  # threading：线程模块（IO 线程、事件）
import time  # time：时间模块（性能计时、时间戳）
from collections import defaultdict, deque
# defaultdict：带默认值字典；deque：双端队列
from collections.abc import Callable, Generator
# Callable：可调用对象；Generator：生成器类型
from concurrent.futures import Future  # Future：并发未来对象
from contextlib import ExitStack, contextmanager
# ExitStack：上下文管理器堆栈；contextmanager：上下文管理器装饰器
from enum import IntEnum  # IntEnum：整数枚举
from functools import partial  # partial：偏函数（绑定参数）
from inspect import isclass, signature  # 检查类；获取函数签名
from logging import DEBUG  # DEBUG 日志级别
from multiprocessing.queues import Queue  # 多进程队列（tensor IPC）
from typing import Any, TypeVar, cast  # 类型标注工具
import msgspec  # msgspec：高性能 msgpack 序列化
import zmq  # zmq：ZeroMQ 消息队列

import vllm.envs as envs  # vllm 环境变量
from vllm.config import ParallelConfig, VllmConfig  # 并行配置；全局配置
from vllm.config.pooler import POOLER_CONFIG_LOG_FIELDS  # 池化配置日志字段
from vllm.distributed import (
    cleanup_dist_env_and_memory,  # 清理分布式环境和内存
    stateless_destroy_torch_distributed_process_group,  # 销毁无状态分布式进程组
)
from vllm.envs import enable_envs_cache  # 启用环境变量缓存
from vllm.logger import init_logger  # 初始化日志记录器
from vllm.logging_utils.dump_input import dump_engine_exception
# 转储引擎异常信息（调试）
from vllm.lora.request import LoRARequest  # LoRA 请求
from vllm.multimodal import MULTIMODAL_REGISTRY  # 多模态注册表
from vllm.tasks import POOLING_TASKS, SupportedTask  # 池化任务；支持任务
from vllm.tracing import instrument, maybe_init_worker_tracer  # 追踪工具
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
# 配置按值序列化注册
from vllm.utils import numa_utils  # NUMA 工具
from vllm.utils.gc_utils import (
    freeze_gc_heap,  # 冻结 GC 堆（减少 GC 暂停）
    maybe_attach_gc_debug_callback,  # 附加 GC 调试回调（可选）
)
from vllm.utils.hashing import get_hash_fn_by_name  # 获取哈希函数
from vllm.utils.network_utils import make_zmq_socket  # 创建 ZMQ socket
from vllm.utils.system_utils import decorate_logs, set_process_title
# 装饰日志；设置进程标题
from vllm.v1.core.kv_cache_utils import (
    BlockHash,  # 块哈希
    generate_scheduler_kv_cache_config,  # 生成调度器 KV 缓存配置
    get_kv_cache_capacity,  # 获取 KV 缓存容量
    get_kv_cache_configs,  # 获取 KV 缓存配置列表
    get_request_block_hasher,  # 获取请求块哈希器
    init_none_hash,  # 初始化空哈希
    resolve_kv_cache_block_sizes,  # 解析 KV 缓存块大小
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
# 暂停状态；调度器接口
from vllm.v1.core.sched.output import SchedulerOutput  # 调度器输出
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
# 注册所有 KV 缓存规格
from vllm.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,  # 弹性 EP 通知 call_id
    EEPNotificationType,  # 弹性 EP 通知类型
    EngineCoreOutput,  # 引擎核心输出（单请求）
    EngineCoreOutputs,  # 引擎核心输出容器（批次）
    EngineCoreReadyResponse,  # 引擎就绪响应
    EngineCoreRequest,  # 引擎核心请求
    EngineCoreRequestType,  # 引擎核心请求类型
    FinishReason,  # 完成原因
    PauseMode,  # 暂停模式
    ReconfigureDistributedRequest,  # 分布式重配置请求
    ReconfigureRankType,  # 重配置 rank 类型
    UtilityOutput,  # 工具输出
    UtilityResult,  # 工具结果
)
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver  # 张量 IPC 接收器
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,  # 引擎握手元数据
    EngineZmqAddresses,  # 引擎 ZMQ 地址
    SignalCallback,  # 信号回调
    get_physical_gpu_ids_for_local_dp_rank,  # 获取本地 DP rank 的物理 GPU ID
)
from vllm.v1.executor import Executor  # 执行器抽象类
from vllm.v1.fault_tolerance.engine_core_sentinel import (
    FT_UTILITY_METHOD,  # 容错工具方法名
    EngineCoreSentinel,  # 引擎核心哨兵（容错）
    fault_tolerant_wrapper,  # 容错包装器
)
from vllm.v1.kv_cache_interface import KVCacheConfig, get_kv_cache_spec_kind
# KV 缓存配置；获取 KV 缓存规格类型
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats
# 调度器迭代详情；调度器统计
from vllm.v1.outputs import ModelRunnerOutput  # 模型运行器输出
from vllm.v1.request import Request, RequestStatus  # 请求；请求状态
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
# msgpack 解码器、编码器
from vllm.v1.structured_output import StructuredOutputManager
# 结构化输出管理器
from vllm.v1.utils import compute_iteration_details  # 计算迭代详情
from vllm.version import __version__ as VLLM_VERSION  # vLLM 版本号

logger = init_logger(__name__)  # 模块级日志记录器

HANDSHAKE_TIMEOUT_MINS = 5  # 握手超时时间（5 分钟）

_R = TypeVar("_R")  # 泛型返回类型变量（collective_rpc 用）


class EngineCore:
    """Inner loop of vLLM's Engine."""
    # vLLM 引擎的内层循环（核心引擎主体）

    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        executor_fail_callback: Callable | None = None,  # 执行器失败回调（可选）
        include_finished_set: bool = False,  # 是否包含已完成集合
    ):
        # plugins need to be loaded at the engine/scheduler level too
        # 插件需要在引擎/调度器层面也加载
        from vllm.plugins import load_general_plugins  # 延迟导入

        load_general_plugins()  # 加载通用插件

        self.vllm_config = vllm_config  # 保存全局配置
        if not vllm_config.parallel_config.data_parallel_rank_local:
            # 如果不是 DP 本地 rank（只记录一次）
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,  # 版本号
                vllm_config,  # 配置
            )

        self.log_stats = log_stats  # 保存日志统计标志

        # Setup Model.
        # 设置模型
        self.model_executor = executor_class(vllm_config)
        # 创建模型执行器（管理 GPU worker）
        self._pooler_config_logged = False  # 池化配置日志标志
        if executor_fail_callback is not None:
            # 如果提供了失败回调
            self.model_executor.register_failure_callback(executor_fail_callback)
            # 注册失败回调

        self.available_gpu_memory_for_kv_cache = -1
        # KV 缓存可用 GPU 内存（-1 = 未确定）

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            # 如果是弹性 EP 扩容启动
            self._eep_scale_up_before_kv_init()
            # 在 KV 初始化前执行弹性扩展准备

        # Setup KV Caches and update CacheConfig after profiling.
        # 配置 KV 缓存并在 profiling 后更新 CacheConfig。
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        # 初始化 KV 缓存（内存 profiling + 分配）
        self.structured_output_manager = StructuredOutputManager(vllm_config)
        # 创建结构化输出管理器

        # Setup scheduler.
        # 设置调度器
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        # 获取调度器类

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            # 无 KV 缓存的编码器模型不支持分块 prefill。SSM 模型呢？
            if vllm_config.scheduler_config.enable_chunked_prefill:
                # 如果启用了分块 prefill
                logger.warning("Disabling chunked prefill for model without KVCache")
                # 记录警告
                vllm_config.scheduler_config.enable_chunked_prefill = False
                # 禁用分块 prefill

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        # 解析调度器块大小和哈希块大小

        self.scheduler: SchedulerInterface = Scheduler(
            # 创建调度器
            vllm_config=vllm_config,  # 配置
            kv_cache_config=kv_cache_config,  # KV 缓存配置
            structured_output_manager=self.structured_output_manager,
            # 结构化输出管理器
            include_finished_set=include_finished_set,  # 包含已完成集合
            log_stats=self.log_stats,  # 日志统计
            block_size=scheduler_block_size,  # 块大小
            hash_block_size=hash_block_size,  # 哈希块大小
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        # 是否使用投机解码
        self.check_for_draft_tokens = (
            # 是否检查草稿 token
            self.use_spec_decode  # 使用投机解码
            or vllm_config.model_config.is_diffusion  # 或是扩散模型
        )
        if self.scheduler.connector is not None:  # type: ignore
            # 如果调度器有连接器（KV 传输）
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore
            # 初始化 KV 输出聚合器

        mm_registry = MULTIMODAL_REGISTRY  # 多模态注册表
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )
        # 创建多模态接收器缓存

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        # 如果为调度器初始化了 KV 连接器，需要从所有 worker 收集握手元数据，
        # 使调度器中的连接器具有完整上下文
        kv_connector = self.scheduler.get_kv_connector()  # 获取 KV 连接器
        if kv_connector is not None:
            # 如果有 KV 连接器
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            # 从 worker 收集并存储 KV 连接器传输元数据（KV 缓存注册后）
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )
            # 获取传输握手元数据

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {(pp_rank, tp_rank): metadata}
                # Merge all worker dicts into a single dict
                # xfer_handshake_metadata 是来自 worker 的字典列表
                # 每个字典已具有结构 {(pp_rank, tp_rank): metadata}
                # 将所有 worker 字典合并为单个字典
                content: dict[tuple[int, int], Any] = {}  # 合并字典
                for worker_dict in xfer_handshake_metadata:
                    # 遍历 worker 字典
                    if worker_dict is not None:
                        # 如果非空
                        content.update(worker_dict)  # 合并
                kv_connector.set_xfer_handshake_metadata_pp_aware(content)
                # 设置传输握手元数据（PP 感知）

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        # 为流水线并行设置批次队列。
        # 批次队列用于已调度的批次。这使我们能异步调度和执行批次，
        # 流水线并行需要它消除流水线气泡。
        self.batch_queue_size = vllm_config.max_concurrent_batches
        # 批次队列大小（最大并发批次数）
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        # 批次队列（保存未来结果、调度输出、执行未来）
        if self.batch_queue_size > 1:
            # 如果批次队列大小 > 1
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            # 记录调试日志
            self.batch_queue = deque(maxlen=self.batch_queue_size)
            # 创建批次队列

        self.is_ec_consumer = (
            # 是否是专家缓存消费者
            vllm_config.ec_transfer_config is None
            # 无专家缓存传输配置
            or vllm_config.ec_transfer_config.is_ec_consumer
            # 或是专家缓存消费者
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"
        # 是否是池化模型

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        # 请求块哈希器（前缀缓存用）
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            # 如果启用前缀缓存或有 KV 连接器
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            # 获取缓存哈希函数
            init_none_hash(caching_hash_fn)  # 初始化空哈希

            self.request_block_hasher = get_request_block_hasher(
                hash_block_size, caching_hash_fn
            )
            # 创建请求块哈希器

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        # 选择步进函数（有批次队列时用带队列版本）
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling
        # 是否异步调度

        self.aborts_queue = queue.Queue[list[str]]()  # 中止请求队列
        self._idle_state_callbacks: list[Callable] = []  # 空闲状态回调列表

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        # 将启动堆标记为静态，使 GC 忽略它。
        # 减少最老代集合的暂停时间。
        freeze_gc_heap()  # 冻结 GC 堆
        # If enable, attach GC debugger after static variable freeze.
        # 如果启用，在静态变量冻结后附加 GC 调试器。
        maybe_attach_gc_debug_callback()  # 附加 GC 调试回调（可选）
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        # 启用环境变量缓存（假设此后不再有环境变量覆盖）
        enable_envs_cache()  # 启用环境变量缓存

    @instrument(span_name="Prepare model")
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
        # 初始化 KV 缓存（内存 profiling + 分配）
        start = time.time()  # 记录开始时间

        # register all kvcache specs in enginecore process.
        # 在引擎核心进程中注册所有 KV 缓存规格。
        register_all_kvcache_specs(vllm_config)  # 注册规格

        # Get all kv cache needed by the model
        # 获取模型所需的所有 KV 缓存
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        # 获取 KV 缓存规格（每个 worker 一组）

        # Some layers (e.g. Prefix LM attention) run non-causally and tag their
        # KV cache spec with ``non_causal=True``. The specs are collected here in
        # the engine-core process (the same process that builds the scheduler),
        # so this is the multiproc-safe place to translate that layer-level
        # signal into a scheduling policy: chunked prefill and prefix caching
        # both assume causal attention and would corrupt non-causal prefill.
        # 某些层（如前缀 LM 注意力）非因果运行，并将其 KV 缓存规格标记为
        # ``non_causal=True``。规格在此引擎核心进程（构建调度器同一进程）
        # 收集，因此这是将层级信号转换为调度策略的多进程安全位置：
        # 分块 prefill 和前缀缓存都假设因果关系，会破坏非因果 prefill。
        if any(
            getattr(spec, "non_causal", False)  # 是否非因果
            for worker_specs in kv_cache_specs  # 遍历 worker 规格
            for spec in worker_specs.values()  # 遍历规格值
        ):
            # 如果有非因果层
            if vllm_config.scheduler_config.enable_chunked_prefill:
                # 如果启用了分块 prefill
                logger.info(
                    "Disabling chunked prefill: model has non-causal attention layers."
                )
                # 记录日志
                vllm_config.scheduler_config.enable_chunked_prefill = False
                # 禁用分块 prefill
            if vllm_config.cache_config.enable_prefix_caching:
                # 如果启用了前缀缓存
                logger.info(
                    "Disabling prefix caching: model has non-causal attention layers."
                )
                # 记录日志
                vllm_config.cache_config.enable_prefix_caching = False
                # 禁用前缀缓存

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        # 是否使用 KV 缓存
        if has_kv_cache:
            # 如果使用 KV 缓存
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # 如果是弹性 EP 扩容启动
                # NOTE(yongji): should already be set
                # during _eep_scale_up_before_kv_init
                # 注意：应在 _eep_scale_up_before_kv_init 中已设置
                assert self.available_gpu_memory_for_kv_cache > 0
                # 断言内存已确定
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
                # 使用预设内内存
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                # 分析模型的峰值内存使用，确定可为 KV 缓存分配多少内存。
                available_gpu_memory = self.model_executor.determine_available_memory()
                # 计算可用 GPU 内存
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
                # 保存第一个值
        else:
            # Attention free models don't need memory for kv cache
            # 无注意力模型不需要 KV 缓存内存
            available_gpu_memory = [0] * len(kv_cache_specs)
            # 全部设为 0

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # 断言规格数和内存数一致

        # Track max_model_len before KV cache config to detect auto-fit changes
        # 在 KV 缓存配置前跟踪 max_model_len，检测自动适配变化
        max_model_len_before = vllm_config.model_config.max_model_len
        # 保存调整前的 max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        # 生成 KV 缓存配置（可能自动适配 max_model_len）

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        # 如果自动适配减少了 max_model_len，将新值同步给 worker。
        # 因为 worker 在内存 profiling 前已启动，缓存了原始（较大）值。
        max_model_len_after = vllm_config.model_config.max_model_len
        # 获取调整后的 max_model_len
        if max_model_len_after != max_model_len_before:
            # 如果发生了变化
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))
            # 同步给所有 worker

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        # 生成调度器 KV 缓存配置
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        # 更新 GPU 块数
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        # KV 缓存组
        if kv_cache_groups:
            # 如果有 KV 缓存组
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )
            # 取最小块大小
            num_tokens, max_concurrency = get_kv_cache_capacity(
                vllm_config, scheduler_kv_cache_config
            )
            # 计算 KV 缓存容量
            vllm_config.cache_config.kv_cache_size_tokens = num_tokens
            # 保存容量（token 数）
            vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency
            # 保存最大并发度

        vllm_config.validate_block_size()  # 验证块大小

        # Initialize kv cache and warmup the execution
        # 初始化 KV 缓存并预热执行
        self.model_executor.initialize_from_config(kv_cache_configs)
        # 从配置初始化 KV 缓存并预热

        elapsed = time.time() - start  # 计算耗时
        compile_time = vllm_config.compilation_config.compilation_time
        # 编译耗时
        encoder_compile_time = vllm_config.compilation_config.encoder_compile_time
        # 编码器编译耗时
        if encoder_compile_time > 0:
            # 如果编码器有编译耗时
            logger.info_once(
                # 记录一次性日志
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s — language_model: %.2f s, "
                "encoder: %.2f s)",
                elapsed,  # 总耗时
                compile_time + encoder_compile_time,  # 总编译
                compile_time,  # 语言模型编译
                encoder_compile_time,  # 编码器编译
            )
        elif compile_time > 0:
            # 如果只有语言模型编译
            logger.info_once(
                # 记录一次性日志
                "init engine (profile, create kv cache, warmup model) took "
                "%.2f s (compilation: %.2f s)",
                elapsed,  # 总耗时
                compile_time,  # 编译耗时
            )
        else:
            # 无编译耗时
            logger.info_once(
                # 记录一次性日志
                "init engine (profile, create kv cache, warmup model) took %.2f s",
                elapsed,  # 总耗时
            )
        return scheduler_kv_cache_config  # 返回调度器 KV 缓存配置

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 获取支持的任务类型
        supported_tasks = self.model_executor.supported_tasks  # 从执行器获取
        self._log_pooler_config(supported_tasks)  # 记录池化配置（如适用）
        return supported_tasks  # 返回

    def _log_pooler_config(self, supported_tasks: tuple[SupportedTask, ...]) -> None:
        # 记录池化配置（仅一次）
        if self._pooler_config_logged:
            # 如果已记录过
            return  # 直接返回

        model_config = self.vllm_config.model_config  # 模型配置
        pooler_config = model_config.pooler_config  # 池化配置
        if (
            self.vllm_config.parallel_config.data_parallel_rank_local
            # 是 DP 本地 rank
            or model_config.runner_type != "pooling"  # 非池化模型
            or pooler_config is None  # 无池化配置
        ):
            return  # 不记录

        supported_pooling_tasks = tuple(
            sorted(set(supported_tasks) & set(POOLING_TASKS))
        )
        # 计算支持的池化任务
        if not supported_pooling_tasks:
            # 如果没有池化任务
            return  # 不记录

        self._pooler_config_logged = True  # 标记已记录
        task_set = set(supported_pooling_tasks)  # 任务集合
        use_activation = pooler_config.use_activation  # 是否使用激活
        if use_activation is None:
            # 如果未指定
            use_activation = True  # 默认为 True
        sources = getattr(model_config, "_pooler_config_sources", {})
        # 池化配置来源
        pooling_type_field = (
            # 池化类型字段
            "seq_pooling_type"
            if task_set & {"embed", "classify"}  # 嵌入/分类任务
            else "tok_pooling_type"
            # 否则 token 池化
        )

        def log_field(name: str, field: str) -> str:
            # 格式化单个字段
            value = (
                use_activation  # 激活值
                if field == "use_activation"
                else getattr(pooler_config, field)  # 否则获取字段
            )
            source = sources.get(field, "unknown")  # 获取来源
            return f"{name}={value}(source={source})"
            # 格式化

        log_items = [("pooling_type", pooling_type_field)]  # 池化类型
        log_items.extend(  # 追加其他字段
            (field, field)  # (名称, 字段)
            for field in POOLER_CONFIG_LOG_FIELDS  # 遍历日志字段
            if field != pooling_type_field  # 排除池化类型
        )
        config_fields = ", ".join(log_field(name, field) for name, field in log_items)
        # 拼接所有字段

        logger.info_once(
            "Resolved pooling config: %s, supported_tasks=%s",
            config_fields,  # 配置字段
            supported_pooling_tasks,  # 支持的任务
        )
        # 记录日志

    def get_kv_cache_group_metadata(self) -> list[dict[str, int | str | None]]:
        """Return msgspec-serializable metadata for scheduler KV cache groups."""
        # 返回调度器 KV 缓存组的可序列化元数据
        kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
        # 获取 KV 缓存配置
        if kv_cache_config is None:
            # 如果没有
            return []  # 返回空列表

        metadata: list[dict[str, int | str | None]] = []  # 元数据列表
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            # 遍历 KV 缓存组
            spec = group.kv_cache_spec  # 获取规格
            metadata.append(
                {
                    "group_idx": group_idx,  # 组索引
                    "kind": get_kv_cache_spec_kind(spec).value,  # 类型
                    "block_size": spec.block_size,  # 块大小
                    "sliding_window": getattr(spec, "sliding_window", None),
                    # 滑动窗口
                }
            )
        return metadata  # 返回元数据

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # 将请求添加到调度器。
        # request_wave：指示请求预计属于 DP 情况下的哪个 wave。
        # Validate the request_id type.
        # 验证请求 ID 类型。
        if not isinstance(request.request_id, str):
            # 如果不是字符串
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )
            # 抛出类型错误

        if pooling_params := request.pooling_params:
            # 如果有池化参数
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]
            # 筛选支持的池化任务

            if pooling_params.task not in supported_pooling_tasks:
                # 如果任务不支持
                raise ValueError(
                    # 抛出错误
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            # 如果有 KV 传输参数但无连接器
            logger.warning(
                # 记录警告
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        if (
            request.ec_transfer_params is not None
            and self.scheduler.get_ec_connector() is None
        ):
            # 如果有专家缓存传输参数但无连接器
            logger.warning(
                # 记录警告
                "Got ec_transfer_params, but no ECConnector found. "
                "Disabling ECTransfer for this request."
            )

        self.scheduler.add_request(request)  # 添加到调度器
        if request.abort_immediately:
            # 如果需要立即中止
            # Immediately abort so the connector's request_finished hook runs
            # to free any pre-admission KV-transfer resources.
            # 立即中止，使连接器的 request_finished 钩子运行以释放
            # 任何预准入的 KV 传输资源。
            self.abort_requests([request.request_id])  # 立即中止

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
        # 从调度器中止请求

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        # TODO：调度器不需要知道具体完成原因，
        # 是否传播该信息待定（即客户端中止 vs 满足停止标准）。
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
        # 以"已中止"状态结束请求

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        # 执行模型并在失败时记录详细信息
        try:
            yield  # 让出执行权
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.
            # 此处不捕获 BaseException，因为只对 execute_model 自身
            # 错误导致的异常感兴趣，需要转储信息。
            # NOTE: This method is exception-free
            # 注意：此方法无异常
            dump_engine_exception(
                # 转储引擎异常
                self.vllm_config,  # 配置
                scheduler_output,  # 调度器输出
                self.scheduler.make_stats(),  # 调度器统计
            )
            raise err  # 重新抛出

    @contextmanager
    def capture_iteration_details(
        self, scheduler_output: SchedulerOutput | None
    ) -> Generator[SchedulerIterationDetails | None, None, None]:
        # 捕获迭代详情（用于日志）
        enable_details = (
            self.vllm_config.observability_config.enable_logging_iteration_details
        )
        # 是否启用迭代详情日志
        if not self.log_stats or not enable_details:
            # 如果未启用
            yield None  # 产出 None
            return  # 返回
        # 0-token step: let the dummy_batch wrapper log it (avoids double-log).
        # 0 token step：让 dummy_batch 包装器记录（避免重复记录）。
        if (
            scheduler_output is not None
            and scheduler_output.total_num_scheduled_tokens == 0
        ):
            # 如果调度了 0 个 token
            yield None  # 产出 None
            return  # 返回

        iteration_index = getattr(self, "_iteration_index", 0)
        # 迭代索引
        # scheduler_output=None marks a DP dummy iteration.
        # scheduler_output=None 表示 DP 空迭代。
        if scheduler_output is None:
            # 如果是空迭代
            iteration_details = SchedulerIterationDetails(
                # 创建空迭代详情
                iteration_index=iteration_index,  # 迭代索引
                num_ctx_requests=0,  # 上下文请求数
                num_ctx_tokens=0,  # 上下文 token 数
                num_generation_requests=0,  # 生成请求数
                num_generation_tokens=0,  # 生成 token 数
                elapsed_ms=0.0,  # 耗时（毫秒）
                is_dummy=True,  # 标记空迭代
            )
        else:
            # 正常迭代
            details = compute_iteration_details(scheduler_output)
            # 计算迭代详情
            iteration_details = SchedulerIterationDetails(
                # 创建迭代详情
                iteration_index=iteration_index,  # 迭代索引
                num_ctx_requests=details.num_ctx_requests,  # 上下文请求数
                num_ctx_tokens=details.num_ctx_tokens,  # 上下文 token 数
                num_generation_requests=details.num_generation_requests,
                # 生成请求数
                num_generation_tokens=details.num_generation_tokens,
                # 生成 token 数
                elapsed_ms=0.0,  # 耗时（稍后填充）
                num_encoder_inputs=details.num_encoder_inputs,  # 编码器输入数
                num_encoder_output_tokens=details.num_encoder_output_tokens,
                # 编码器输出 token 数
            )

        start_time = time.monotonic()  # 记录开始时间（单调时钟）
        yield iteration_details  # 产出详情
        iteration_details.elapsed_ms = (time.monotonic() - start_time) * 1000
        # 计算耗时（毫秒）
        self._iteration_index = iteration_index + 1  # 更新迭代索引

    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        # 创建带迭代详情的调度统计
        stats = self.scheduler.make_stats() or SchedulerStats()
        # 获取调度器统计（或创建空）
        stats.iteration_details = iteration_details  # 设置迭代详情
        return stats  # 返回

    def _attach_iteration_details(
        self,
        outputs: dict[int, EngineCoreOutputs],  # 输出字典
        iteration_details: SchedulerIterationDetails | None,  # 迭代详情
    ) -> None:
        # 附加迭代详情到输出
        if iteration_details is None:
            # 如果无详情
            return  # 返回

        if (eco := next(iter(outputs.values()), None)) is None:
            # 如果没有输出
            outputs[0] = eco = EngineCoreOutputs()  # 创建空输出
        if eco.scheduler_stats is None:
            # 如果没有调度统计
            eco.scheduler_stats = self._make_iteration_details_stats(iteration_details)
            # 创建带详情的统计
        else:
            eco.scheduler_stats.iteration_details = iteration_details
            # 设置迭代详情

    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        # 是否在本 step 推迟新 prefill（DP prefill 平衡）。
        # 由 DP 引擎核心覆盖；否则永不限流。
        return False  # 默认不限流

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """
        # 调度、执行并生成输出。
        # 返回 (输出字典, 模型是否执行) 元组。

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        # 检查调度器中是否还有请求 - 未完成的，
        # 或已完成的但尚未从批次中移除的。
        if not self.scheduler.has_requests():
            # 如果没有请求
            return {}, False  # 返回空输出、未执行

        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        # 调度请求
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        # 非阻塞执行模型
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        # 获取语法位掩码（结构化输出）
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            # 捕获迭代详情
            self.log_error_detail(scheduler_output),  # 错误详情
        ):
            model_output = future.result()  # 等待模型输出
            if model_output is None:
                # 如果无模型输出（需要采样）
                model_output = self.model_executor.sample_tokens(grammar_output)
                # 采样 token

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        # 在处理模型输出前，处理模型执行期间发生的任何中止。
        self._process_aborts_queue()  # 处理中止队列
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        # 从输出更新调度器状态
        self._attach_iteration_details(engine_core_outputs, iteration_details)
        # 附加迭代详情

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
        # 返回输出和是否执行了模型

    def post_step(self, model_executed: bool) -> None:
        # 步进后处理
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        # 使用异步调度时无法提前获取草稿 token ID，
        # 因此在 worker 进程中更新草稿 token ID，无需在此更新。
        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            # 如果检查草稿 token、非异步调度、且模型已执行
            draft_token_ids = self.model_executor.take_draft_token_ids()
            # 获取草稿 token ID
            if draft_token_ids is not None:
                # 如果有草稿 token
                self.scheduler.update_draft_token_ids(draft_token_ids)
                # 更新草稿 token ID

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        # 使用批次队列调度和执行批次。
        # 注意：如果本 step 无输出，返回 None。
        # 执行流程：
        # 1. 如果批次队列未满，尝试调度新批次。
        #    如果调度了新批次，直接返回空引擎核心输出。
        #    换句话说，填满批次队列的优先级高于获取模型输出。
        # 2. 如果没有新调度批次（队列满或无可调度请求），
        #    阻塞直到作业队列中第一个批次完成。
        # 3. 从输出更新调度器。

        batch_queue = self.batch_queue  # 批次队列
        assert batch_queue is not None  # 断言非空

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        # 如果批次队列未满，尝试调度新批次，
        # 但如果所有请求都已调度，调度器可能返回空批次。
        # 注意：此操作非阻塞。
        assert len(batch_queue) < self.batch_queue_size  # 断言队列未满

        model_executed = False  # 模型是否执行
        deferred_scheduler_output = None  # 延迟的调度器输出
        if self.scheduler.has_requests():
            # 如果有请求
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            # 调度请求
            with self.log_error_detail(scheduler_output):  # 错误详情
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
                # 非阻塞执行模型
            if self.is_ec_consumer:
                # 如果是专家缓存消费者
                model_executed = scheduler_output.total_num_scheduled_tokens > 0
                # 模型是否执行

            if self.is_pooling_model or not model_executed:
                # 如果是池化模型或未执行
                # No sampling required (no requests scheduled).
                # 无需采样（未调度请求）
                future = cast(Future[ModelRunnerOutput], exec_future)
                # 直接使用执行未来
            else:
                # 需要采样
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    # 未等待任何 token，获取语法输出并立即采样。
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    # 获取语法位掩码
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                    # 非阻塞采样
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    # 需延迟采样，直到处理完上一步的模型输出。
                    deferred_scheduler_output = scheduler_output
                    # 保存延迟输出

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                # 将该 step 的未来加入队列。
                batch_queue.appendleft((future, scheduler_output, exec_future))
                # 推入队列
                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # If the queue is not full and there is more work, don't
                    # block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    # 如果队列未满且有更多工作，除非队列已满或无更多请求可调度，
                    # 否则不阻塞等待下一个 worker 响应。
                    return None, model_executed  # 返回空输出

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            # 队列为空。不应到达此处，因为此方法只应在
            # 调度器包含请求或队列非空时调用。
            return None, False  # 返回空输出

        # Block until the next result is available.
        # 阻塞直到下一个结果可用。
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        # 弹出最早批次
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            # 捕获迭代详情
            self.log_error_detail(scheduler_output),  # 错误详情
        ):
            model_output = future.result()  # 等待模型输出
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                # sample_tokens() 返回 None 表示原始 execute_model() 失败。
                exec_model_fut.result()  # 抛出异常
                raise RuntimeError("unexpected error")  # 抛出错误

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        # 在处理模型输出前，处理模型执行期间发生的任何中止。
        self._process_aborts_queue()  # 处理中止队列
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        # 从输出更新调度器
        self._attach_iteration_details(engine_core_outputs, iteration_details)
        # 附加迭代详情

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        # 注意：我们可以在此处理延迟任务或保存在字段中，在下次调用
        # step_with_batch_queue 时立即处理。后者略偏向 TTFT 而非 TPOT/吞吐量。
        if deferred_scheduler_output:
            # 如果有延迟任务
            # When draft tokens are used with structured output, validate them
            # before computing the grammar bitmask for the deferred request.
            # 结构化输出与草稿 token 一起使用时，在计算延迟请求的
            # 语法位掩码前验证草稿 token。
            if self.check_for_draft_tokens:
                # 如果检查草稿 token
                draft_token_ids = self.model_executor.take_draft_token_ids()
                # 获取草稿 token
                if draft_token_ids is not None:
                    # Update the draft token ids in the scheduler output to
                    # filter out the invalid spec tokens, which will be padded
                    # with -1 and skipped by the grammar bitmask computation.
                    # 更新调度器输出中的草稿 token ID，过滤无效投机 token，
                    # 这些 token 将用 -1 填充并被语法位掩码计算跳过。
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
                    # 更新草稿 token
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            # 现在有计算延迟请求位掩码所需的 token。获取位掩码并采样。
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            # 获取语法位掩码
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            # 非阻塞采样
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))
            # 推入队列

        return engine_core_outputs, model_executed  # 返回输出和执行标志

    def _process_aborts_queue(self):
        # 处理中止队列
        if not self.aborts_queue.empty():
            # 如果队列非空
            request_ids = []  # 请求 ID 列表
            while not self.aborts_queue.empty():
                # 排空队列
                ids = self.aborts_queue.get_nowait()  # 取出
                # Should be a list here, but also handle string just in case.
                # 此处应为列表，但以防万一也处理字符串。
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
                # 扩展列表
            # More efficient to abort all as a single batch.
            # 一次性批量中止更高效。
            self.abort_requests(request_ids)  # 批量中止

    def shutdown(self):
        # 关闭引擎核心
        logger.debug_once("[shutdown] EngineCore: tearing down local resources")
        # 记录日志
        self.structured_output_manager.clear_backend()  # 清理结构化输出后端
        if self.model_executor:
            # 如果有执行器
            self.model_executor.shutdown()  # 关闭执行器
        if self.scheduler:
            # 如果有调度器
            self.scheduler.shutdown()  # 关闭调度器

        # Undo the gc.freeze() from __init__ so that the objects allocated
        # during engine startup (model weights, KV caches, etc.) become
        # visible to the garbage collector again. Without this, deleting
        # the engine in-process (e.g. unit tests) leaks GPU memory.
        # 撤销 __init__ 中的 gc.freeze()，使引擎启动期间分配的对象
        # （模型权重、KV 缓存等）对垃圾回收器重新可见。
        # 否则，在进程内删除引擎（如单元测试）会泄漏 GPU 内存。
        gc.unfreeze()  # 解冻 GC 堆
        # Tear down distributed state initialized in this EngineCore process
        # before it exits and release cached memory.
        # 在此 EngineCore 进程退出前拆除其初始化的分布式状态并释放缓存内存。
        cleanup_dist_env_and_memory()  # 清理分布式环境和内存
        logger.debug_once("[shutdown] EngineCore: local resource teardown complete")
        # 记录日志

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # 启停性能分析
        self.model_executor.profile(is_start, profile_prefix)  # 委托给执行器

    def reset_mm_cache(self):
        # 重置多模态缓存
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        # 注意：这主要用于调试，不尝试重新同步内部缓存（P0 发送、P1 接收）
        if self.scheduler.has_unfinished_requests():
            # 如果有未完成请求
            logger.warning(
                # 记录警告
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        # 缓存存在于 EngineCore 或 WorkerWrapperBase 中
        if self.mm_receiver_cache is not None:
            # 如果有接收器缓存
            self.mm_receiver_cache.clear_cache()  # 清空缓存

        self.model_executor.reset_mm_cache()  # 委托给执行器

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 重置前缀缓存
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )
        # 委托给调度器

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        Clears both the scheduler's cache manager and the GPU model runner's cache.
        """
        # 重置编码器缓存以失效所有缓存的编码器输出。
        # 模型权重更新时应调用此方法，确保不会复用旧权重计算的过期视觉嵌入。
        # 同时清空调度器的缓存管理器和 GPU 模型运行器的缓存。
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        # 注意：这主要用于调试，不尝试重新同步内部缓存。
        if self.scheduler.has_unfinished_requests():
            # 如果有未完成请求
            logger.warning(
                # 记录警告
                "Resetting the encoder cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # Reset the scheduler's encoder cache manager (logical state)
        # 重置调度器的编码器缓存管理器（逻辑状态）
        self.scheduler.reset_encoder_cache()
        # Reset the GPU model runner's encoder cache (physical storage)
        # 重置 GPU 模型运行器的编码器缓存（物理存储）
        self.model_executor.reset_encoder_cache()

    def _reset_caches(
        self,
        reset_running_requests: bool = True,
        reset_connector: bool = True,
    ) -> None:
        # 重置所有缓存
        # reset_connector=True so external connectors clear alongside
        # local caches, matching the pause_generation(clear_cache=True)
        # contract. No-op when no connector is configured.
        # reset_connector=True 使外部连接器与本地缓存一起清空，
        # 匹配 pause_generation(clear_cache=True) 契约。
        # 未配置连接器时为无操作。
        self.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        )
        # 重置前缀缓存
        self.reset_mm_cache()  # 重置多模态缓存
        self.reset_encoder_cache()  # 重置编码器缓存

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        # 暂停生成；行为取决于模式。
        # 所有暂停模式都会排队新的添加请求 -- "abort" 和 "keep" 跳过 step()；
        # "wait" 允许 step() 使进行中请求排空。
        # - ``abort``：设置 PAUSED_NEW，中止所有请求，等待中止输出发送
        #   （使用 output_queue 运行时），可选清空缓存，然后完成返回的 Future。
        # - ``wait``：设置 PAUSED_NEW（排队添加，继续 step）；排空后
        #   可选清空缓存，然后完成返回的 Future。
        # - ``keep``：设置 PAUSED_ALL；返回输出队列为空时完成的 Future。
        if mode not in ("keep", "abort", "wait"):
            # 如果模式无效
            raise ValueError(f"Invalid pause mode: {mode}")  # 抛出错误
        if mode == "wait":
            # if 'wait' 模式在 inproc 引擎模式不可用
            raise ValueError("'wait' mode can't be used in inproc-engine mode")

        if mode == "abort":
            # 如果是 abort 模式
            self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
            # 中止所有请求

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        # 设置暂停状态
        self.scheduler.set_pause_state(pause_state)  # 设置暂停状态
        if clear_cache:
            # 如果需要清空缓存
            self._reset_caches()  # 重置所有缓存

        return None  # 同步完成

    def resume_scheduler(self) -> None:
        """Resume the scheduler and flush any requests queued while paused."""
        # 恢复调度器并刷新暂停期间排队的请求
        self.scheduler.set_pause_state(PauseState.UNPAUSED)  # 设置为未暂停

    def is_scheduler_paused(self) -> bool:
        """Return whether the scheduler is in any pause state."""
        # 返回调度器是否处于任何暂停状态
        return self.scheduler.pause_state != PauseState.UNPAUSED
        # 检查暂停状态

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
        """Put the engine to sleep at the specified level.

        Args:
            level: Sleep level.
                - Level 0: Pause scheduling only. Requests are still accepted
                           but not processed. No GPU memory changes.
                - Level 1: Offload model weights to CPU, discard KV cache.
                - Level 2: Discard all GPU memory.
            mode: Pause mode - how to deal with any existing requests, see
                documentation of pause_scheduler method.
        """
        # 将引擎置于指定级别的休眠。
        # 参数说明：
        # level：休眠级别。
        #   - 级别 0：仅暂停调度。请求仍被接受但不处理。无 GPU 内存变化。
        #   - 级别 1：将模型权重卸载到 CPU，丢弃 KV 缓存。
        #   - 级别 2：丢弃所有 GPU 内存。
        # mode：暂停模式 - 如何处理现有请求，见 pause_scheduler 方法文档。

        # Pause scheduler before sleeping.
        # 休眠前暂停调度器。
        clear_prefix_cache = level >= 1  # 级别 >=1 时清空前缀缓存
        pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
        # 暂停调度器
        if level < 1:
            # 如果是级别 0
            return pause_future  # 返回暂停未来

        # Level 1+: Delegate to executor for GPU memory management
        # 级别 1+：委托给执行器进行 GPU 内存管理
        model_executor = self.model_executor  # 模型执行器
        if pause_future is None:
            # 如果暂停已同步完成
            model_executor.sleep(level)  # 执行器休眠
            return None  # 返回 None

        future = Future[Any]()  # 创建未来

        def pause_complete(f: Future):
            # 暂停完成回调
            try:
                f.result()  # propagate any exception
                # 传播任何异常
                future.set_result(model_executor.sleep(level))  # 休眠
            except Exception as e:
                future.set_exception(e)  # 设置异常

        logger.info("Waiting for in-flight requests to complete before sleeping...")
        # 记录日志
        pause_future.add_done_callback(pause_complete)  # 注册回调
        return future  # 返回未来

    def wake_up(self, tags: list[str] | None = None):
        """Wake up the engine from sleep.

        Args:
            tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.
        """
        # 从休眠中唤醒引擎。
        # 参数：tags 要唤醒的标签。级别 0 唤醒使用 ["scheduling"]。
        if tags is not None and "scheduling" in tags:
            # Remove "scheduling" from tags if there are other tags to process.
            # 如果有其他标签要处理，从 tags 中移除 "scheduling"。
            tags = [t for t in tags if t != "scheduling"]  # 过滤标签

        if tags is None or tags:
            # 如果无标签或有标签
            self.model_executor.wake_up(tags)  # 唤醒执行器

        # Partial wakes intentionally keep the remaining allocations asleep.
        # Resume scheduling only once all executor memory is resident again.
        # 部分唤醒有意保持剩余分配休眠。
        # 仅在所有执行器内存重新驻留后才恢复调度。
        if not self.model_executor.is_sleeping:
            # 如果执行器不再休眠
            self.resume_scheduler()  # 恢复调度器

    def is_sleeping(self) -> bool:
        """Check if engine is sleeping at any level."""
        # 检查引擎是否处于任何级别的休眠
        return self.is_scheduler_paused() or self.model_executor.is_sleeping
        # 调度器暂停或执行器休眠

    def execute_dummy_batch(self):
        # 执行空批次（DP 同步用）
        self.model_executor.execute_dummy_batch()  # 委托给执行器

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 添加 LoRA 适配器
        return self.model_executor.add_lora(lora_request)  # 委托给执行器

    def remove_lora(self, lora_id: int) -> bool:
        # 移除 LoRA 适配器
        return self.model_executor.remove_lora(lora_id)  # 委托给执行器

    def list_loras(self) -> set[int]:
        # 列出已注册的 LoRA
        return self.model_executor.list_loras()  # 委托给执行器

    def pin_lora(self, lora_id: int) -> bool:
        # 固定 LoRA 适配器（防止被逐出）
        return self.model_executor.pin_lora(lora_id)  # 委托给执行器

    def save_sharded_state(
        self,
        path: str,  # 保存路径
        pattern: str | None = None,  # 模式（可选）
        max_size: int | None = None,  # 最大大小（可选）
    ) -> None:
        # 保存分片状态
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )
        # 委托给执行器

    def collective_rpc(
        self,
        method: str | Callable[..., _R],  # 方法名或函数
        timeout: float | None = None,  # 超时
        args: tuple = (),  # 位置参数
        kwargs: dict[str, Any] | None = None,  # 关键字参数
    ) -> list[_R]:
        # 集体 RPC：在所有 worker 上调用
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)
        # 委托给执行器

    def set_weight_version(self, weight_version: str) -> None:
        self._weight_version = weight_version

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self._weight_version

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # 预处理请求。
        # 此函数可直接用于输入处理线程，使请求初始化与模型前向并行。
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        # 线程安全说明：无竞态条件。
        # `mm_receiver_cache` 在 LLMEngine 初始化结束时重置，
        # 此后仅由输入处理线程访问。
        if self.mm_receiver_cache is not None and request.mm_features:
            # 如果有多模态接收器缓存和特性
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )
            # 更新多模态特性

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        # 将引擎核心请求转换为内部请求（含块哈希）
        if req.use_structured_output:
            # 如果使用结构化输出
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            # 线程安全说明：无竞态条件。
            # `grammar_init` 仅由输入处理线程调用。对于
            # `structured_output_manager`，每个请求独立且语法编译异步。
            # 调度器总在调度请求前检查语法编译状态。
            self.structured_output_manager.grammar_init(req)  # 初始化语法
        return req, request.current_wave  # 返回请求和 wave

    def _eep_scale_up_before_kv_init(self):
        # 弹性 EP 扩容前的 KV 初始化准备（基类未实现）
        raise NotImplementedError

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,  # 通知类型
        vllm_config: VllmConfig | None = None,  # 配置（可选）
    ):
        # 发送引擎核心通知（基类未实现）
        raise NotImplementedError


class EngineShutdownState(IntEnum):
    # 引擎关闭状态枚举
    RUNNING = 0  # 运行中
    REQUESTED = 1  # 已请求关闭
    SHUTTING_DOWN = 2  # 正在关闭


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
    # 在后台进程中运行 EngineCore 的 ZMQ 包装器（引擎核心进程）

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"
    # 引擎核心死亡信号（发送给客户端）
    addresses: EngineZmqAddresses  # ZMQ 地址

    @instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 配置
        local_client: bool,  # 是否本地客户端
        handshake_address: str,  # 握手地址
        executor_class: type[Executor],  # 执行器类
        log_stats: bool,  # 是否记录统计
        client_handshake_address: str | None = None,  # 客户端握手地址
        tensor_queue: Queue | None = None,  # 张量队列
        *,
        engine_index: int = 0,  # 引擎索引
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        # 输入队列（请求类型 + 数据）
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        # 输出队列（客户端索引 + 输出 或 字节信号）
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )
        # 执行器失败回调（向输入队列发送失败信号）

        self.engine_index = engine_index  # 引擎索引
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        # ZMQ 身份（2 字节小端）
        self.engines_running = False  # 引擎运行标志
        self.shutdown_state = EngineShutdownState.RUNNING  # 关闭状态

        # Receiver for tensor IPC
        # 张量 IPC 接收器
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None  # 接收器
        if tensor_queue is not None:
            # 如果有张量队列
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            # 创建接收器
            logger.info("Using tensor IPC queue for multimodal tensor sharing")
            # 记录日志

        with self._perform_handshakes(
            handshake_address,  # 握手地址
            identity,  # 引擎身份
            local_client,  # 本地客户端标志
            vllm_config,  # 配置
            client_handshake_address,  # 客户端握手地址
        ) as addresses:
            # Set up data parallel environment.
            # 设置数据并行环境。
            self.has_coordinator = addresses.coordinator_output is not None
            # 是否有协调器
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            # 前端统计发布地址
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            # 记录调试日志
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # 是否内部 DP 负载均衡
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            # 仅为"内部"和"混合"负载均衡模式发布请求队列统计到协调器。
            self.publish_dp_lb_stats = internal_dp_balancing
            # 是否发布 DP 负载均衡统计
            self.last_counts = (0, 0)  # 上次计数

            self.addresses = addresses  # 保存地址
            self.process_input_queue_block = True  # 输入队列阻塞标志
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # 如果是弹性 EP 扩容启动
                self._eep_send_engine_core_notification(
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,  # 通知类型
                    vllm_config=vllm_config,  # 配置
                )
                # 发送新引擎初始化就绪通知
            self._init_data_parallel(vllm_config)  # 初始化数据并行

            super().__init__(
                vllm_config,  # 配置
                executor_class,  # 执行器
                log_stats,  # 日志统计
                executor_fail_callback,  # 失败回调
                internal_dp_balancing,  # 内部 DP 平衡
            )
            # 调用基类初始化

            # Initialize fault tolerance settings.
            # 初始化容错设置。
            self.enable_fault_tolerance = (
                vllm_config.parallel_config.enable_fault_tolerance
            )
            # 是否启用容错
            if self.enable_fault_tolerance:
                # 如果启用容错
                self.ft_sentinel = EngineCoreSentinel(
                    # 创建引擎核心哨兵
                    engine=self,  # 引擎引用
                    parallel_config=vllm_config.parallel_config,  # 并行配置
                )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            # 后台线程和队列用于 IO。由于它们释放 GIL，
            # 可将 ZMQ socket IO 与 GPU 重叠，并将部分序列化/反序列化
            # 与模型前向重叠。
            # 线程处理 Socket 与队列之间，核心忙循环使用队列。
            ready_event = threading.Event()  # 就绪事件
            input_thread = threading.Thread(
                target=self.process_input_sockets,  # 输入 socket 处理
                args=(
                    addresses.inputs,  # 输入地址
                    addresses.coordinator_input,  # 协调器输入地址
                    identity,  # 引擎身份
                    ready_event,  # 就绪事件
                ),
                daemon=True,  # 守护线程
            )
            input_thread.start()  # 启动输入线程

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,  # 输出 socket 处理
                args=(
                    addresses.outputs,  # 输出地址
                    addresses.coordinator_output,  # 协调器输出地址
                    self.engine_index,  # 引擎索引
                ),
                daemon=True,  # 守护线程
            )
            self.output_thread.start()  # 启动输出线程

            # Don't complete handshake until DP coordinator ready message is
            # received.
            # 在收到 DP 协调器就绪消息前不完成握手。
            while not ready_event.wait(timeout=10):
                # 循环等待就绪事件（10 秒超时）
                if not input_thread.is_alive():
                    # 如果输入线程已死亡
                    raise RuntimeError("Input socket thread died during startup")
                    # 抛出错误
                assert addresses.coordinator_input is not None  # 断言有协调器
                logger.info("Waiting for READY message from DP Coordinator...")
                # 记录日志

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,  # 握手地址
        identity: bytes,  # 引擎身份
        local_client: bool,  # 本地客户端标志
        vllm_config: VllmConfig,  # 配置
        client_handshake_address: str | None,  # 客户端握手地址
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        # 执行启动握手。
        # 对于 DP=1 或离线模式，与同机前端进程握手。
        # 对于内部负载均衡的 DP>1，与可能位于不同节点的共享前端进程握手。
        # 对于外部或混合负载均衡的 DP>1，执行两次握手：
        #   - 与 rank 0 前端进程握手，获取 DP 协调器 ZMQ 地址和 DP 进程组地址。
        #   - 与同机前端进程握手，获取客户端输入/输出 socket 地址。
        # 例外：rank 0 和同机引擎本身不需要第二次握手。
        # 此处"前端"进程可指包含引擎核心客户端的进程
        # （API 服务器未扩展时即 API 服务器进程），
        # 或运行 serve.py 中 run_multi_api_server() 的启动器进程。
        input_ctx = zmq.Context()  # 创建 ZMQ 上下文
        is_local = local_client and client_handshake_address is None
        # 是否仅本地握手
        headless = not local_client  # 是否无头模式
        handshake = self._perform_handshake(
            input_ctx,  # 上下文
            handshake_address,  # 握手地址
            identity,  # 身份
            is_local,  # 本地标志
            headless,  # 无头标志
            vllm_config,  # 配置
            vllm_config.parallel_config,  # 并行配置
        )
        if client_handshake_address is None:
            # We only need to handshake with one party.
            # 只需与一方握手。
            with handshake as addresses:
                yield addresses  # 产出地址
        else:
            # We need to handshake with rank 0 front-end and our colocated frontend.
            # 需要与 rank 0 前端和同机前端握手。
            assert local_client  # 断言本地客户端
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            # 执行本地握手
            with handshake as addresses, local_handshake as client_addresses:
                # 1. Obtain DP Coordinator zmq address and DP process group address
                #    (addresses).
                # 2. Add front-end input/output addresses from colocated front-end
                #    (client_addresses).
                # 1. 获取 DP 协调器 ZMQ 地址和 DP 进程组地址（addresses）。
                # 2. 从同机前端添加前后端输入/输出地址（client_addresses）。
                addresses.inputs = client_addresses.inputs  # 输入地址
                addresses.outputs = client_addresses.outputs  # 输出地址
                yield addresses  # 产出地址

        # Update config which may have changed from the handshake
        # 更新握手后可能变化的配置
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,  # ZMQ 上下文
        handshake_address: str,  # 握手地址
        identity: bytes,  # 引擎身份
        local_client: bool,  # 本地客户端标志
        headless: bool,  # 无头标志
        vllm_config: VllmConfig,  # 配置
        parallel_config_to_update: ParallelConfig | None = None,  # 待更新配置
    ) -> Generator[EngineZmqAddresses, None, None]:
        # 执行单次握手
        with make_zmq_socket(
            ctx,  # 上下文
            handshake_address,  # 地址
            zmq.DEALER,  # DEALER 模式
            identity=identity,  # 身份
            linger=5000,  # 存活时间
            bind=False,  # 不绑定（连接）
        ) as handshake_socket:
            # Register engine with front-end.
            # 向前端注册引擎。
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            # 启动握手
            yield addresses  # 产出地址

            # Send ready message.
            # 发送就绪消息。
            ready_msg = {
                "status": "READY",  # 状态
                "local": local_client,  # 本地标志
                "headless": headless,  # 无头标志
            }
            # Include config hash for DP configuration validation
            # 包含配置哈希用于 DP 配置验证
            if vllm_config.parallel_config.data_parallel_size > 1:
                # 如果 DP>1
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )
                # 计算配置哈希

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))
            # 发送就绪消息

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,  # 握手 socket
        local_client: bool,  # 本地客户端标志
        headless: bool,  # 无头标志
        parallel_config: ParallelConfig | None = None,  # 并行配置（可选）
    ) -> EngineZmqAddresses:
        # 执行启动握手（注册 + 获取地址）
        # Send registration message.
        # 发送注册消息。
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",  # 状态
                    "local": local_client,  # 本地标志
                    "headless": headless,  # 无头标志
                }
            )
        )
        # 发送 HELLO

        # Receive initialization message.
        # 接收初始化消息。
        logger.debug("Waiting for init message from front-end.")  # 记录日志
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            # 如果握手超时（5 分钟）
            raise RuntimeError(
                # 抛出超时错误
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()  # 接收初始化消息
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        # 解码初始化消息
        logger.debug("Received init message: %s", init_message)  # 记录日志

        if parallel_config is not None:
            # 如果有并行配置要更新
            for key, value in init_message.parallel_config.items():
                # 遍历配置项
                setattr(parallel_config, key, value)  # 更新配置

        return init_message.addresses  # 返回地址

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""
        # 在后台进程中启动引擎核心忙循环。

        # Ensure we can serialize transformer config after spawning
        # 确保启动后可序列化 transformer 配置
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None  # 引擎核心
        signal_callback: SignalCallback | None = None  # 信号回调
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]  # 配置
            parallel_config: ParallelConfig = vllm_config.parallel_config
            # 并行配置
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            # 是否数据并行
            if data_parallel:
                # 如果数据并行
                parallel_config.data_parallel_rank_local = local_dp_rank
                # 设置本地 DP rank
                process_title = f"EngineCore_DP{dp_rank}"  # 进程标题
            else:
                process_title = "EngineCore"  # 进程标题
            set_process_title(process_title)  # 设置进程标题
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            # 初始化 worker 追踪器
            decorate_logs()  # 装饰日志
            if parallel_config.numa_bind:
                # 如果启用 NUMA 绑定
                numa_utils.log_current_affinity_state(process_title)
                # 记录当前亲和性状态

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # 如果数据并行且配置了 KV 传输
                # modify the engine_id and append the dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                # 修改 engine_id 并追加 dp_rank，确保每个 DP rank 的
                # kv_transfer_config 唯一。
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
                )
                # 追加 DP rank 后缀
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )
                # 记录日志

            parallel_config.data_parallel_index = dp_rank  # 设置 DP 索引
            if data_parallel and vllm_config.model_config.is_moe:
                # 如果数据并行且是 MoE 模型
                # Set data parallel rank for this engine process.
                # 为此引擎进程设置数据并行 rank。
                parallel_config.data_parallel_rank = dp_rank  # 设置 DP rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
                # 创建 DP 引擎核心进程
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                # 非 MoE DP rank 完全独立，因此视作 DP=1。
                # 注意 data_parallel_index 仍反映原始 DP rank。
                parallel_config.data_parallel_size = 1  # DP 大小设为 1
                parallel_config.data_parallel_size_local = 1  # 本地 DP 设为 1
                parallel_config.data_parallel_rank = 0  # DP rank 设为 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
                # 创建普通引擎核心进程

            assert engine_core is not None  # 断言创建成功

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                # 请求关闭时通过 input_queue 唤醒空闲引擎
                # 在信号处理器中不安全 - 可能在主线程持有非可重入的
                # input_queue.mutex 时中断它
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
                # 向输入队列发送唤醒信号

            signal_callback = SignalCallback(wakeup_engine)  # 创建信号回调

            def signal_handler(signum, frame):
                # 信号处理器
                signal_name = signal.Signals(signum).name  # 信号名
                logger.info(
                    "[shutdown] EngineCore: trigger received signal=%s",
                    signal_name,
                )
                # 记录日志
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                # 标记请求关闭
                signal_callback.trigger()  # 触发回调

            signal.signal(signal.SIGTERM, signal_handler)  # 注册 SIGTERM
            signal.signal(signal.SIGINT, signal_handler)  # 注册 SIGINT

            engine_core.run_busy_loop()  # 运行忙循环

        except SystemExit:
            # 捕获退出
            logger.info_once("[shutdown] EngineCore: exiting busy loop")  # 记录
            raise  # 重新抛出
        except Exception as e:
            # 捕获异常
            if engine_core is None:
                # 如果引擎核心未创建
                logger.exception("EngineCore failed to start.")  # 记录
            else:
                logger.exception("EngineCore encountered a fatal error.")  # 记录
                engine_core._send_engine_dead()  # 发送引擎死亡信号
            raise e  # 重新抛出
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)  # 恢复默认信号
            signal.signal(signal.SIGINT, signal.SIG_DFL)  # 恢复默认信号
            if signal_callback is not None:
                # 如果有信号回调
                signal_callback.stop()  # 停止回调
            if engine_core is not None:
                # 如果有引擎核心
                engine_core.shutdown()  # 关闭引擎

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # 初始化数据并行（基类空实现）
        pass

    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        # 返回引擎是否应步进
        return (
            self.engines_running  # 引擎在运行
            or self.scheduler.has_requests()  # 调度器有请求
            or bool(self.batch_queue)  # 批次队列非空
        )

    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        # 返回是否尚未请求关闭
        return self.shutdown_state == EngineShutdownState.RUNNING
        # 检查关闭状态

    @fault_tolerant_wrapper
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        # 引擎核心的忙循环。
        while self._handle_shutdown():
            # 循环直到关闭
            # 1) Poll the input queue until there is work to do.
            # 1) 轮询输入队列直到有工作要做。
            self._process_input_queue()  # 处理输入队列
            # Publish request counts before and after GPU step to ensure freshness.
            # 在 GPU step 前后发布请求计数以确保新鲜度。
            self._maybe_publish_request_counts()  # 发布请求计数
            # 2) Step the engine core and return the outputs.
            # 2) 步进引擎核心并返回输出。
            self._process_engine_step()  # 步进引擎
            self._maybe_publish_request_counts()  # 发布请求计数

        raise SystemExit  # 退出进程

    def _maybe_publish_request_counts(self):
        # 可能发布请求计数（DP 负载均衡）
        if not self.publish_dp_lb_stats:
            # 如果不需要发布
            return  # 返回

        # Publish our request counts (if they've changed).
        # 如果我们的请求计数有变化则发布。
        counts = self.scheduler.get_request_counts()  # 获取请求计数
        if counts != self.last_counts:
            # 如果计数有变化
            self.last_counts = counts  # 更新上次计数
            stats = SchedulerStats(
                *counts,  # 展开计数
                kv_cache_usage=self.scheduler.get_kv_cache_usage(),  # KV 缓存使用率
            )
            # 创建统计
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))
            # 发布给协调器（-1 = 协调器）

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""
        # 当需要执行引擎步进时退出。

        waited = False  # 是否等待过
        while not self.has_work() and self.is_running():
            # 循环直到有工作或停止运行
            # Notify callbacks waiting for engine to become idle.
            # 通知等待引擎空闲的回调。
            self._notify_idle_state_callbacks()  # 通知空闲回调
            if self.input_queue.empty():
                # 如果输入队列为空
                # Drain aborts queue; all aborts are also processed via input_queue.
                # 排空中止队列；所有中止也通过输入队列处理。
                with self.aborts_queue.mutex:
                    # 获取锁
                    self.aborts_queue.queue.clear()  # 清空中止队列
                if logger.isEnabledFor(DEBUG):
                    # 如果启用了 DEBUG 日志
                    logger.debug("EngineCore waiting for work.")  # 记录日志
                    waited = True  # 标记等待过
            block = self.process_input_queue_block  # 是否阻塞
            try:
                req = self.input_queue.get(block=block)  # 获取请求
                self._handle_client_request(*req)  # 处理客户端请求
            except queue.Empty:
                # 如果队列空
                break  # 退出
            if not block:
                # 如果非阻塞
                break  # 退出

        if waited:
            # 如果等待过
            logger.debug("EngineCore loop active.")  # 记录日志

        # Handle any more client requests.
        # 处理更多客户端请求。
        while not self.input_queue.empty():
            # 排空输入队列
            req = self.input_queue.get_nowait()  # 非阻塞获取
            self._handle_client_request(*req)  # 处理请求

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""
        # 仅在存在未完成的本地请求时调用。

        # Step the engine core.
        # 步进引擎核心。
        outputs, model_executed = self.step_fn()  # 步进
        # Put EngineCoreOutputs into the output queue.
        # 将 EngineCoreOutputs 放入输出队列。
        for output in outputs.items() if outputs else ():
            # 遍历输出
            self.output_queue.put_nowait(output)  # 入队
        # Post-step hook.
        # 步进后钩子。
        self.post_step(model_executed)  # 步进后处理

        # If no model execution happened but there is still scheduler work
        # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
        # the GIL briefly to allow background transfer threads to make progress.
        # 如果没有模型执行但仍有调度器工作（如等待远程 KV 或延迟的 KV 连接器释放），
        # 短暂让出 GIL 允许后台传输线程推进。
        if not model_executed and self.scheduler.has_requests():
            # 如果未执行且有请求
            time.sleep(0.001)  # 短暂休眠

        return model_executed  # 返回是否执行

    def _notify_idle_state_callbacks(self) -> None:
        # 通知空闲状态回调
        while self._idle_state_callbacks:
            # 遍历回调
            callback = self._idle_state_callbacks.pop()  # 弹出回调
            callback(self)  # 调用回调

    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        # 检查是否请求了关闭并处理
        if self.shutdown_state == EngineShutdownState.RUNNING:
            # 如果仍在运行
            return True  # 继续循环

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            # 如果已请求关闭
            shutdown_timeout = self.vllm_config.shutdown_timeout  # 关闭超时
            mode = "abort" if shutdown_timeout == 0 else "drain"
            # 超时为 0 则中止，否则排空

            logger.info(
                "[shutdown] EngineCore: start mode=%s timeout=%ds",
                mode,  # 模式
                shutdown_timeout,  # 超时
            )
            # 记录日志

            if shutdown_timeout == 0:
                # 如果是中止模式
                num_requests = self.scheduler.get_num_unfinished_requests()
                # 获取未完成请求数
                if num_requests > 0:
                    # 如果有请求
                    logger.info(
                        "[shutdown] EngineCore: aborting in-flight requests count=%d",
                        num_requests,
                    )
                    # 记录日志
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                # 中止所有请求
                self._send_abort_outputs(aborted_reqs)  # 发送中止输出
            else:
                # 如果是排空模式
                num_requests = self.scheduler.get_num_unfinished_requests()
                # 获取未完成请求数
                if num_requests > 0:
                    # 如果有请求
                    logger.info(
                        "[shutdown] EngineCore: draining in-flight requests "
                        "count=%d timeout=%ds",
                        num_requests,
                        shutdown_timeout,
                    )
                    # 记录日志

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN
            # 标记关闭中

        # Exit when no work remaining
        # 无剩余工作时退出
        if not self.has_work():
            # 如果没有工作
            logger.info(
                "[shutdown] EngineCore: request processing complete; "
                "starting resource teardown"
            )
            # 记录日志
            return False  # 退出循环

        return True  # 继续循环

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""
        # 分发来自客户端的请求。

        if request_type == EngineCoreRequestType.WAKEUP:
            # 如果是唤醒消息
            return  # 忽略
        elif request_type == EngineCoreRequestType.ADD:
            # 如果是添加请求
            req, request_wave = request  # 解包
            if self._reject_add_in_shutdown(req):
                # 如果关闭中拒绝
                return  # 返回
            self.add_request(req, request_wave)  # 添加请求
        elif request_type == EngineCoreRequestType.ABORT:
            # 如果是中止请求
            self.abort_requests(request)  # 中止请求
        elif request_type == EngineCoreRequestType.UTILITY:
            # 如果是工具调用
            client_idx, call_id, method_name, args = request  # 解包
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                # 如果关闭中拒绝
                return  # 返回
            output = UtilityOutput(call_id)  # 创建工具输出
            # Lazily look-up utility method so that failure will be handled/returned.
            # 延迟查找工具方法，以便处理/返回失败。
            get_result = lambda: (
                (method := getattr(self, method_name))  # 查找方法
                and method(*self._convert_msgspec_args(method, args))
                # 转换参数并调用
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            # 入队输出
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
            # 调用工具方法
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            # 如果是执行器失败
            raise RuntimeError("Executor failed.")  # 抛出错误
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )
            # 记录错误

    def _reject_add_in_shutdown(self, request: Request) -> bool:
        # 关闭中是否拒绝添加请求
        if self.shutdown_state == EngineShutdownState.RUNNING:
            # 如果仍在运行
            return False  # 不拒绝

        logger.debug(
            "[shutdown] EngineCore: rejecting new request request_id=%s",
            request.request_id,
        )
        # 记录日志
        self._send_abort_outputs_to_client([request.request_id], request.client_index)
        # 发送中止输出
        return True  # 拒绝

    def _reject_utility_in_shutdown(
        self, client_idx: int, call_id: int, method_name: str
    ) -> bool:
        # 关闭中是否拒绝工具调用
        if self.shutdown_state == EngineShutdownState.RUNNING:
            # 如果仍在运行
            return False  # 不拒绝

        logger.warning(
            "[shutdown] EngineCore: rejecting utility call method=%s",
            method_name,
        )
        # 记录警告
        output = UtilityOutput(call_id, failure_message="Server shutting down")
        # 创建失败输出
        self.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=output))
        )
        # 入队输出
        return True  # 拒绝

    @staticmethod
    def _invoke_utility_method(
        name: str,  # 方法名
        get_result: Callable,  # 获取结果函数
        output: UtilityOutput,  # 工具输出
        enqueue_output: Callable,  # 入队输出函数
    ):
        # 调用工具方法
        try:
            result = get_result()  # 获取结果
            if isinstance(result, Future):
                # Defer utility output handling until future completion.
                # 延迟工具输出处理直到未来完成。
                callback = lambda future: EngineCoreProc._invoke_utility_method(
                    name, future.result, output, enqueue_output
                )
                # 创建回调
                result.add_done_callback(callback)  # 注册回调
                return  # 返回
            output.result = UtilityResult(result)  # 设置结果
        except Exception as e:
            # 捕获异常
            logger.exception("Invocation of %s method failed", name)  # 记录
            output.failure_message = f"Call to {name} method failed: {str(e)}"
            # 设置失败消息
        enqueue_output(output)  # 入队输出

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        # 如果提供的参数类型与目标方法参数类型不匹配，尝试转换为 msgspec 对象。
        if not args:
            # 如果没有参数
            return args  # 返回
        arg_types = signature(method).parameters.values()  # 参数类型
        assert len(args) <= len(arg_types)  # 断言长度
        return tuple(
            msgspec.convert(v, type=p.annotation)  # 转换参数
            if isclass(p.annotation)  # 是类
            and issubclass(p.annotation, msgspec.Struct)  # 是 msgspec 结构
            and not isinstance(v, p.annotation)  # 不是该类型
            else v  # 否则保持原样
            for v, p in zip(args, arg_types)  # 遍历
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""
        # 向 EngineCoreClient 发送引擎死亡状态。

        # Put ENGINE_CORE_DEAD in the queue.
        # 将 ENGINE_CORE_DEAD 放入队列。
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)  # 入队

        # Wait until msg sent by the daemon before shutdown.
        # 关闭前等待守护线程发送消息。
        self.output_thread.join(timeout=5.0)  # 等待输出线程（5 秒超时）
        if self.output_thread.is_alive():
            # 如果线程仍存活
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )
            # 记录致命错误

    def _make_ready_response(self) -> EngineCoreReadyResponse:
        parallel_config = self.vllm_config.parallel_config
        scheduler_config = self.vllm_config.scheduler_config
        return EngineCoreReadyResponse(
            max_model_len=self.vllm_config.model_config.max_model_len,
            num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
            block_size=self.vllm_config.cache_config.block_size,
            dp_stats_address=self.frontend_stats_publish_address,
            dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
            vllm_version=VLLM_VERSION,
            world_size=self.vllm_config.parallel_config.world_size,
            data_parallel_size=parallel_config.data_parallel_size,
            kv_cache_size_tokens=self.vllm_config.cache_config.kv_cache_size_tokens,
            kv_cache_max_concurrency=(
                self.vllm_config.cache_config.kv_cache_max_concurrency
            ),
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            data_parallel_rank=self.engine_index,
            max_num_seqs=scheduler_config.max_num_seqs,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            instance_id=self.vllm_config.instance_id,
            kv_events_config=self.scheduler.get_kv_event_publisher_config(),
        )

    def process_input_sockets(
        self,
        input_addresses: list[str],  # 输入地址列表
        coord_input_address: str | None,  # 协调器输入地址（可选）
        identity: bytes,  # 引擎身份
        ready_event: threading.Event,  # 就绪事件
    ):
        """Input socket IO thread."""
        # 输入 socket IO 线程。

        # Msgpack serialization decoding with optional tensor IPC receiver.
        # msgpack 序列化解码（带可选张量 IPC 接收器）。
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest,  # 解码类型
            oob_tensor_provider=self.tensor_ipc_receiver  # 张量提供器
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)
        # 通用解码器

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                # 创建输入 sockets
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses  # 遍历地址
            ]
            if coord_input_address is None:
                # 如果没有协调器输入
                coord_socket = None  # 无协调器 socket
            else:
                # 否则创建协调器 socket
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,  # 上下文
                        coord_input_address,  # 地址
                        zmq.XSUB,  # 订阅模式
                        identity=identity,  # 身份
                        bind=False,  # 连接
                    )
                )
                # Send subscription message to coordinator.
                # 向协调器发送订阅消息。
                coord_socket.send(b"\x01")  # 订阅

            # Register sockets with poller.
            # 用轮询器注册 sockets。
            poller = zmq.Poller()  # 创建轮询器
            ready_response = EngineCoreReadyResponse(
                # 创建就绪响应
                max_model_len=self.vllm_config.model_config.max_model_len,
                # 最大模型长度
                num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
                # GPU 块数
                block_size=self.vllm_config.cache_config.block_size,  # 块大小
                dp_stats_address=self.frontend_stats_publish_address,  # DP 统计地址
                dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
                # 数据类型
                vllm_version=VLLM_VERSION,  # 版本号
                world_size=self.vllm_config.parallel_config.world_size,  # 世界大小
                data_parallel_size=self.vllm_config.parallel_config.data_parallel_size,
                # DP 大小
                kv_cache_size_tokens=(
                    self.vllm_config.cache_config.kv_cache_size_tokens
                ),
                # KV 缓存容量
                kv_cache_max_concurrency=(
                    self.vllm_config.cache_config.kv_cache_max_concurrency
                ),
                # 最大并发度
            )
            ready_payload = msgspec.msgpack.encode(ready_response)  # 编码
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                # 向每个输入 socket 发送初始消息 - 前端 ROUTER socket
                # 必须先收到此消息才能向我们发送输入消息。
                input_socket.send(ready_payload)  # 发送就绪
                poller.register(input_socket, zmq.POLLIN)  # 注册轮询

            if coord_socket is not None:
                # 如果有协调器 socket
                # Wait for ready message from coordinator.
                # 等待协调器就绪消息。
                assert coord_socket.recv() == b"READY"  # 断言 READY
                poller.register(coord_socket, zmq.POLLIN)  # 注册轮询

            ready_event.set()  # 设置就绪事件
            del ready_event  # 删除引用
            while True:
                # 主循环
                for input_socket, _ in poller.poll():
                    # 轮询事件
                    # (RequestType, RequestData)
                    # (请求类型, 请求数据)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # 接收多部分消息
                    # NOTE(yongji): ignore READY message sent by DP coordinator
                    # that is used to notify newly started engines
                    # 注意：忽略 DP 协调器发送的 READY 消息（用于通知新启动引擎）
                    if type_frame.buffer == b"READY":
                        # 如果是 READY
                        assert input_socket == coord_socket  # 断言来自协调器
                        continue  # 继续
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))
                    # 解析请求类型

                    # Deserialize the request data.
                    # 反序列化请求数据。
                    request: Any  # 请求
                    if request_type == EngineCoreRequestType.ADD:
                        # 如果是添加请求
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        # 解码请求
                        try:
                            request = self.preprocess_add_request(req)  # 预处理
                        except Exception:
                            self._handle_request_preproc_error(req)  # 处理错误
                            continue  # 继续
                    elif request_type == EngineCoreRequestType.UTILITY:
                        # 如果是工具调用
                        request = generic_decoder.decode(data_frames)  # 解码
                        client_idx, call_id, method, args = request  # 解包
                        if method == FT_UTILITY_METHOD:
                            # 如果是容错方法
                            self.ft_sentinel.handle_command(
                                client_idx, call_id, args[0]
                            )
                            # 处理命令
                            continue  # 继续
                    else:
                        request = generic_decoder.decode(data_frames)  # 解码

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            # 中止请求同时加入两个队列，允许我们急切处理中止，
                            # 同时确保输入队列中的顺序以避免泄漏请求。
                            # 这没问题，因为在调度器中中止是幂等的。
                            self.aborts_queue.put_nowait(request)  # 加入中止队列

                    # Push to input queue for core busy loop.
                    # 推入输入队列供核心忙循环处理。
                    self.input_queue.put_nowait((request_type, request))  # 入队

    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""
        # 输出 socket IO 线程。

        # Msgpack serialization encoding.
        # msgpack 序列化编码。
        encoder = MsgpackEncoder()  # 编码器
        # Send buffers to reuse.
        # 可复用的发送缓冲区。
        reuse_buffers: list[bytearray] = []  # 复用缓冲区列表
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        # 保持输出和缓冲区引用直到 zmq 用完它们（输出可能包含
        # 张量/np 数组，其底层缓冲区被提取用于零拷贝发送）。
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()  # 待处理队列

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        # 必须设置 linger 确保 ENGINE_CORE_DEAD 消息在关闭 socket 前发送。
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                # 创建输出 sockets
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths  # 遍历输出路径
            ]
            coord_socket = (
                # 创建协调器输出 socket
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None  # 如果有协调器输出
                else None
            )
            max_reuse_bufs = len(sockets) + 1  # 最大复用缓冲区数

            while True:
                # 主循环
                output = self.output_queue.get()  # 获取输出
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    # 如果是死亡信号
                    for socket in sockets:
                        # 向所有 socket 发送
                        socket.send(output)  # 发送死亡信号
                    break  # 退出
                assert not isinstance(output, bytes)  # 断言非字节
                client_index, outputs = output  # 解包
                outputs.engine_index = engine_index  # 设置引擎索引

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    # 不为协调器消息复用缓冲区（消息很小）。
                    assert coord_socket is not None  # 断言协调器存在
                    coord_socket.send_multipart(encoder.encode(outputs))
                    # 发送到协调器
                    continue  # 继续

                # Reclaim buffers that zmq is finished with.
                # 回收 zmq 已用完的缓冲区。
                while pending and pending[-1][0].done:
                    # 回收完成的缓冲区
                    reuse_buffers.append(pending.pop()[2])  # 取出缓冲区

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                # 获取或创建缓冲区
                buffers = encoder.encode_into(outputs, buffer)  # 编码到缓冲区
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                # 发送（零拷贝 + 跟踪）
                if not tracker.done:
                    # 如果未完成
                    ref = outputs if len(buffers) > 1 else None  # 引用输出
                    pending.appendleft((tracker, ref, buffer))  # 加入待处理
                elif len(reuse_buffers) < max_reuse_bufs:
                    # 如果可复用
                    # Limit the number of buffers to reuse.
                    # 限制复用缓冲区数量。
                    reuse_buffers.append(buffer)  # 加入复用列表

    @staticmethod
    def _send_msg_tracking_payload(
        socket: zmq.Socket, buffers: Sequence[bytestr]
    ) -> zmq.MessageTracker:
        """Send `buffers` as a zero-copy multipart message, returning a tracker
        for the *first* frame.

        Used instead of `Socket.send_multipart()` because we reuse the buffer
        passed to `MsgpackEncoder.encode_into()`: `send_multipart()` returns a
        tracker for the last frame only.
        """
        more_flag = zmq.SNDMORE if len(buffers) > 1 else 0
        tracker = socket.send(buffers[0], more_flag, copy=False, track=True)
        if more_flag:
            socket.send_multipart(buffers[1:], copy=False)
        return tracker

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        # 记录输入 socket 处理线程中添加请求预处理引发的异常，
        # 并返回请求范围的错误响应。
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        # 记录日志
        self._send_error_outputs_to_client([request.request_id], request.client_index)
        # 发送错误输出

    def pause_scheduler(
        self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> Future | None:
        """Pause generation; behavior depends on mode.

        All pause modes queue new adds -- "abort" and "keep" skip step();
        "wait" allows step() so in-flight requests can drain.

        - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
          outputs to be sent (when running with output_queue), optionally
          clear caches, then complete the returned Future.
        - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
          optionally clear caches, then complete the returned Future.
        - ``keep``: Set PAUSED_ALL; return a Future that completes when the
          output queue is empty.
        """
        # 暂停生成；行为取决于模式。
        # 所有暂停模式都会排队新的添加请求。
        # - ``abort``：设置 PAUSED_NEW，中止所有请求，等待中止输出发送，
        #   可选清空缓存，然后完成返回的 Future。
        # - ``wait``：设置 PAUSED_NEW（排队添加，继续步进）；排空后
        #   可选清空缓存，然后完成返回的 Future。
        # - ``keep``：设置 PAUSED_ALL；返回输出队列为空时完成的 Future。
        if mode not in ("keep", "abort", "wait"):
            # 如果模式无效
            raise ValueError(f"Invalid pause mode: {mode}")  # 抛出错误

        def engine_idle_callback(engine: "EngineCoreProc", future: Future[Any]) -> None:
            # 引擎空闲回调
            if clear_cache:
                # 如果需要清空缓存
                engine._reset_caches()  # 重置缓存
            future.set_result(None)  # 完成未来

        if mode == "abort":
            # 如果是中止模式
            aborted_reqs = self.scheduler.finish_requests(
                None, RequestStatus.FINISHED_ABORTED
            )
            # 中止所有请求
            self._send_abort_outputs(aborted_reqs)  # 发送中止输出

        pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
        # 设置暂停状态
        self.scheduler.set_pause_state(pause_state)  # 设置暂停状态

        if self._pause_complete():
            # 如果暂停已完全完成
            if clear_cache:
                # 如果需要清空缓存
                self._reset_caches()  # 重置缓存
            return None  # 同步完成

        future = Future[Any]()  # 创建未来
        self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
        # 注册空闲回调
        return future  # 返回未来

    def _pause_complete(self) -> bool:
        """Returns True if the pause has fully completed and the caller can
        return ``None`` synchronously; False if the pause is still pending
        and the caller should register an idle-state callback to finish it.
        """
        # 返回暂停是否已完全完成（可同步返回 None）；
        # False 表示暂停仍待完成，调用者应注册空闲状态回调。
        return not self.has_work()  # 无工作即完成

    def _send_finish_outputs_to_client(
        self, req_ids: list[str], client_index: int, finish_reason: FinishReason
    ) -> None:
        # 向客户端发送完成输出
        outputs = [
            EngineCoreOutput(req_id, [], finish_reason=finish_reason)
            # 创建输出
            for req_id in req_ids  # 遍历请求
        ]
        eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
        # 创建输出容器
        self.output_queue.put_nowait((client_index, eco))  # 入队

    def _send_abort_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        # 向客户端发送中止输出
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)
        # 以中止原因发送

    def _send_error_outputs_to_client(
        self, req_ids: list[str], client_index: int
    ) -> None:
        # 向客户端发送错误输出
        self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)
        # 以错误原因发送

    def _send_abort_outputs(self, aborted_reqs: list[Request]) -> None:
        # 发送中止输出（按客户端分组）
        # TODO(nick) this will be moved inside the scheduler
        # TODO(nick)：这将移至调度器内部
        if aborted_reqs:
            # 如果有中止的请求
            # Map client_index to list of request_ids that belong to that client.
            # 将 client_index 映射到属于该客户端的请求 ID 列表。
            by_client = defaultdict[int, set[str]](set)  # 按客户端分组
            for request in aborted_reqs:
                # 遍历请求
                by_client[request.client_index].add(request.request_id)  # 分组
            for client_index, req_ids in by_client.items():
                # 遍历分组
                self._send_abort_outputs_to_client(list(req_ids), client_index)
                # 发送中止输出


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""
    # 在数据并行上下文中运行 EngineCore 的 ZMQ 包装器

    def __init__(
        self,
        vllm_config: VllmConfig,  # 配置
        local_client: bool,  # 本地客户端标志
        handshake_address: str,  # 握手地址
        executor_class: type[Executor],  # 执行器
        log_stats: bool,  # 是否记录统计
        client_handshake_address: str | None = None,  # 客户端握手地址
        tensor_queue: Queue | None = None,  # 张量队列
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )
        # 断言是 MoE 模型（仅 MoE 使用）

        scheduler_config = vllm_config.scheduler_config  # 调度器配置
        self.prefill_schedule_interval = scheduler_config.prefill_schedule_interval
        # prefill 调度间隔

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        # 计数模型前向次数，以便每 N 步与 DP 对等体同步完成状态。
        self.step_counter = 0  # 步进计数器
        self.current_wave = 0  # 当前 wave

        # Two-phase pause protocol state. When pending_pause is True, the
        # engine keeps stepping (dummy batches) while waiting for all DP
        # ranks to also set pending_pause. Once all ranks agree via
        # all-reduce, ignore_start_dp_wave is set so that stale
        # START_DP_WAVE messages cannot re-wake the engines.
        # 两阶段暂停协议状态。当 pending_pause 为 True 时，引擎继续步进
        # （空批次）等待所有 DP rank 也设置 pending_pause。
        # 一旦所有 rank 通过 all-reduce 同意，设置 ignore_start_dp_wave，
        # 使过期的 START_DP_WAVE 消息无法重新唤醒引擎。
        self.pending_pause = False  # 待定暂停标志
        self.ignore_start_dp_wave = False  # 忽略启动 wave 标志

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState
        # 延迟导入弹性状态

        self.eep_scaling_state: ElasticEPScalingState | None = None
        # 弹性 EP 缩放状态

        # Initialize the engine.
        # 初始化引擎。
        dp_rank = vllm_config.parallel_config.data_parallel_rank  # DP rank
        super().__init__(
            vllm_config,  # 配置
            local_client,  # 本地客户端
            handshake_address,  # 握手地址
            executor_class,  # 执行器
            log_stats,  # 日志统计
            client_handshake_address,  # 客户端握手地址
            engine_index=dp_rank,  # 引擎索引
            tensor_queue=tensor_queue,  # 张量队列
        )
        # 调用父类初始化

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        # 为数据并行配置 GPU 和无状态进程组。
        parallel_config = vllm_config.parallel_config  # 并行配置
        dp_rank = parallel_config.data_parallel_rank  # DP rank
        dp_size = parallel_config.data_parallel_size  # DP 大小
        local_dp_rank = parallel_config.data_parallel_rank_local  # 本地 DP rank

        assert dp_size > 1  # 断言 DP>1
        assert local_dp_rank is not None  # 断言本地 rank 存在
        assert 0 <= local_dp_rank <= dp_rank < dp_size  # 断言范围

        self.dp_rank = dp_rank  # 保存 DP rank
        self.dp_size = dp_size  # 保存 DP 大小
        dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
        # 初始化无状态 DP 进程组
        self.dp_group, self.dp_store = dp_group, dp_store  # 保存

    def shutdown(self):
        # 关闭引擎
        super().shutdown()  # 调用父类关闭
        if dp_group := getattr(self, "dp_group", None):
            # 如果有 DP 进程组
            stateless_destroy_torch_distributed_process_group(dp_group)
            # 销毁进程组

    def _pause_complete(self) -> bool:
        """Two-phase DP-aware pause.

        Phase 1: Set local pause state and ``pending_pause`` flag. If the
        engines are idle, kick-start them by setting ``engines_running`` to
        True so ranks enter the stepping loop and reach the all-reduce
        consensus checkpoint in ``_has_global_unfinished_reqs``.

        Phase 2 (in ``_has_global_unfinished_reqs``): Once the all-reduce
        confirms that **all** ranks have ``pending_pause`` set, collectively
        stop stepping and set ``ignore_start_dp_wave`` so that stale
        ``START_DP_WAVE`` messages cannot re-wake any engine.
        """
        # 两阶段 DP 感知暂停。
        # 阶段 1：设置本地暂停状态和 pending_pause 标志。如果引擎空闲，
        # 通过设置 engines_running 为 True 启动它们，使 rank 进入步进循环
        # 并在 _has_global_unfinished_reqs 中到达 all-reduce 共识检查点。
        # 阶段 2（在 _has_global_unfinished_reqs 中）：一旦 all-reduce 确认
        # **所有** rank 都设置了 pending_pause，集体停止步进并设置
        # ignore_start_dp_wave，使过期的 START_DP_WAVE 消息无法重新唤醒任何引擎。
        self.pending_pause = True  # 设置待定暂停
        self.engines_running = True  # 启动引擎（进入步进循环）

        return False  # 暂停未完成

    def add_request(self, request: Request, request_wave: int = 0):
        # 添加请求（DP 版本：wave 跟踪）
        super().add_request(request, request_wave)  # 调用父类
        if self.has_coordinator and request_wave != self.current_wave:
            # 如果有协调器且 wave 不匹配
            if request_wave > self.current_wave:
                # 如果新 wave 更大
                self.current_wave = request_wave  # 更新 wave
            elif (
                not self.engines_running  # 引擎未运行
                and self.scheduler.pause_state == PauseState.UNPAUSED  # 未暂停
            ):
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                # 收到已完成 wave 的请求，通知前端需要启动下一 wave。
                self.engines_running = True  # 启动引擎
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )
                # 发送启动 wave 通知

    def resume_scheduler(self):
        # 恢复调度器（DP 版本）
        if self.pending_pause or (self.engines_running and self.ignore_start_dp_wave):
            # 如果暂停仍在进行或忽略 wave 但引擎运行
            raise RuntimeError(
                # 抛出错误
                "resume_scheduler called while pause is still in "
                "flight. Wait for the pause future to resolve before "
                "resuming."
            )
        if self.engines_running:
            # 如果引擎已在运行
            logger.debug("Resume called while engines are not paused, ignoring.")
            # 记录日志
            return

        super().resume_scheduler()  # 调用父类
        self.ignore_start_dp_wave = False  # 清除忽略标志

        # Barrier: wait for all DP ranks to have resumed (and cleared
        # ignore_start_dp_wave) before any rank starts stepping. Uses
        # the existing all-reduce which is safe because engines are
        # stopped.
        # 屏障：在任一条令开始步进前等待所有 DP rank 已恢复
        # （并清除 ignore_start_dp_wave）。使用现有 all-reduce，
        # 因为引擎已停止所以是安全的。
        has_global_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, self.scheduler.has_unfinished_requests()
        )
        # 检查全局未完成请求

        if has_global_unfinished:
            # 如果全局有未完成
            self.engines_running = True  # 启动引擎

    def barrier(self):
        """Blocking barrier on the DP process group (test-only utility)."""
        # DP 进程组上的阻塞屏障（仅测试用）
        import torch.distributed as dist  # 延迟导入

        dist.barrier(group=self.dp_group)  # 执行屏障

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        # 处理客户端请求（DP 版本）
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            # 如果是启动 DP wave
            if self.ignore_start_dp_wave:
                # 如果忽略启动 wave
                return  # 忽略
            new_wave, exclude_eng_index = request  # 解包
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                # 如果排除引擎不是自己且 wave 有效
                self.current_wave = new_wave  # 更新 wave
                if not self.engines_running:
                    # 如果引擎未运行
                    logger.debug(
                        "EngineCore starting idle loop for wave %d.",
                        new_wave,
                    )
                    # 记录日志
                    self.engines_running = True  # 启动引擎
        else:
            super()._handle_client_request(request_type, request)  # 调用父类

    def _maybe_publish_request_counts(self):
        # 可能发布请求计数（DP 版本）
        if not self.publish_dp_lb_stats:
            # 如果不需要发布
            return  # 返回

        # Publish our request counts (if they've changed), stamped with the
        # lockstep-synchronized step counter and wave number.
        # 发布请求计数（如有变化），并盖上锁步同步的步进计数器和 wave 编号。
        counts = self.scheduler.get_request_counts()  # 获取计数
        if counts != self.last_counts:
            # 如果有变化
            self.last_counts = counts  # 更新
            stats = SchedulerStats(
                *counts,  # 展开
                kv_cache_usage=self.scheduler.get_kv_cache_usage(),  # KV 使用率
                step_counter=self.step_counter,  # 步进计数
                current_wave=self.current_wave,  # 当前 wave
            )
            # 创建统计
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))
            # 发布

    def _should_throttle_prefills(self) -> bool:
        # Throttle new prefills to cadence-aligned steps for DP balancing.
        # step_counter is identical across DP ranks. On a fresh wave the
        # counter is 0, so prefills are admitted immediately after idle.
        # 为 DP 平衡将新 prefill 限制到节奏对齐的步进。
        # step_counter 在各 DP rank 间相同。新 wave 上计数器为 0，
        # 因此空闲后立即允许 prefill。
        return (
            self.prefill_schedule_interval > 1  # 间隔大于 1
            and self.step_counter % self.prefill_schedule_interval != 0
            # 非对齐步
        )

    @fault_tolerant_wrapper
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""
        # 数据并行情况下的引擎核心忙循环。

        # Loop until process is sent a SIGINT or SIGTERM
        # 循环直到进程收到 SIGINT 或 SIGTERM
        while self._handle_shutdown():
            # 循环直到关闭
            # 1) Poll the input queue until there is work to do.
            # 1) 轮询输入队列直到有工作要做。
            self._process_input_queue()  # 处理输入队列
            # Publish request counts before and after GPU step to ensure freshness.
            # 在 GPU 步进前后发布请求计数以确保新鲜度。
            self._maybe_publish_request_counts()  # 发布计数

            if self.eep_scaling_state is not None:
                # 如果有弹性缩放状态
                _ = self.eep_scaling_state.progress()  # 推进状态机
                if self.eep_scaling_state.is_complete():
                    # 如果缩放完成
                    if self.eep_scaling_state.worker_type == "removing":
                        # 如果是被移除的 worker
                        raise SystemExit  # 退出
                    self.process_input_queue_block = True  # 恢复阻塞
                    self.eep_scaling_state = None  # 清除状态

            executed = self._process_engine_step()  # 步进引擎
            self._maybe_publish_request_counts()  # 发布计数

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            # 本地未完成请求
            if not executed:
                # 如果未执行
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    # 所有引擎空闲。
                    continue  # 继续

                # Execute a dummy pass when no ready requests ran, unless the
                # engine is sleeping.
                # 无就绪请求运行时执行空 pass，除非引擎在休眠。
                elif not self.model_executor.is_sleeping:
                    # 如果引擎未休眠
                    with self.capture_iteration_details(None) as iteration_details:
                        # 捕获空迭代详情
                        self.execute_dummy_batch()  # 执行空批次
                    if iteration_details is not None and not self.has_coordinator:
                        # 如果有详情且无协调器
                        stats = self._make_iteration_details_stats(iteration_details)
                        # 创建统计
                        self.output_queue.put_nowait(
                            (0, EngineCoreOutputs(scheduler_stats=stats))
                        )
                        # 发布统计

            # 3) All-reduce operation to determine global unfinished reqs.
            # 3) All-reduce 操作确定全局未完成请求。
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )
            # 计算全局未完成状态

            if not self.engines_running:
                # 如果所有引擎空闲
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    # 通知客户端正在暂停循环。
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # 记录日志
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    # 有协调器时，dp rank 0 发送更新给协调器。
                    # 否则（离线 spmd 情况），每个 rank 发送更新给同机前端进程。
                    client_index = -1 if self.has_coordinator else 0
                    # 客户端索引
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                    # 发送 wave 完成通知
                # Increment wave count and reset step counter.
                # 增加 wave 计数并重置步进计数器。
                self.current_wave += 1  # wave +1
                self.step_counter = 0  # 重置步进计数

        raise SystemExit  # 退出进程

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # 检查全局未完成请求
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        # 优化 - 仅每 32 步执行完成同步 all-reduce。
        self.step_counter += 1  # 步进计数 +1
        if self.step_counter % 32 != 0:
            # 如果未到 32 步
            return True  # 假设有未完成

        has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
            self.dp_group,  # DP 进程组
            has_unfinished=local_unfinished,  # 本地未完成
            pending_pause=self.pending_pause,  # 待定暂停
        )
        # 同步 DP 状态

        if pause_consensus:
            # 如果暂停共识达成
            self.ignore_start_dp_wave = True  # 忽略启动 wave
            self.pending_pause = False  # 清除待定暂停
            logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")
            # 记录日志

        return has_unfinished  # 返回全局未完成状态

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        # 重新初始化分布式（弹性 EP 缩放时调用）
        from copy import deepcopy  # 延迟导入

        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState
        # 延迟导入弹性状态

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        # 深拷贝并行配置
        old_dp_size = new_parallel_config.data_parallel_size  # 旧 DP 大小
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        # 设置新 DP 大小
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            # 如果指定了新 rank
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
            # 设置新 rank
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        # 设置主节点 IP
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        # 设置主端口
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        # 设置端口列表
        new_parallel_config._coord_store_port = reconfig_request.coord_store_port
        # 设置协调存储端口

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        # 是否缩容
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )
        # 是否关闭当前 rank

        self.eep_scaling_state = ElasticEPScalingState(
            # 创建弹性缩放状态机
            model_executor=self.model_executor,  # 执行器
            engine_core=self,  # 引擎核心
            vllm_config=self.vllm_config,  # 配置
            new_parallel_config=new_parallel_config,  # 新配置
            worker_type="removing" if is_shutdown else "existing",
            # worker 类型
            scale_type="scale_down" if is_scale_down else "scale_up",
            # 缩放类型
            reconfig_request=reconfig_request,  # 重配置请求
        )
        self.process_input_queue_block = False  # 非阻塞输入（状态机处理）
        logger.info(
            "[Elastic EP] Received reconfiguration request and starting scaling up/down"
        )
        # 记录日志

    def _eep_send_engine_core_notification(
        self,
        notification_type: EEPNotificationType,  # 通知类型
        vllm_config: VllmConfig | None = None,  # 配置（可选）
    ):
        """
        Send notifications to EngineCoreClient, which can then forward
        the notifications to other engine core processes. It is used for:
        1) In scale down: removing core engines to notify EngineCoreClient
           so EngineCoreClient can release their ray placement groups;
        2) Both scale up/down: to notify EngineCoreClient that existing
           core engines have already switched to the new parallel setup.
        """
        # 向 EngineCoreClient 发送通知，Client 可转发给其他引擎核心进程。
        # 用于：
        # 1) 扩容：新核心引擎通知现有核心引擎它们已就绪；
        # 2) 缩容：被移除的核心引擎通知 EngineCoreClient 释放它们的 Ray 放置组；
        # 3) 扩缩容：通知 EngineCoreClient 现有核心引擎已切换到新并行配置。
        if vllm_config is None:
            # 如果未提供配置
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            # 使用当前配置
        else:
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            # 使用新配置
        notification_data = (notification_type.value, dp_rank)
        # 通知数据（类型 + rank）
        outputs = EngineCoreOutputs(
            # 创建输出容器
            utility_output=UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID,  # 通知 call_id
                result=UtilityResult(notification_data),  # 通知数据
            )
        )
        outputs.engine_index = self.engine_index  # 设置引擎索引

        if hasattr(self, "output_thread") and self.output_thread.is_alive():
            # 如果输出线程存活
            self.output_queue.put_nowait((0, outputs))  # 入队
        else:
            # 否则直接发送
            encoder = MsgpackEncoder()  # 编码器
            with (
                zmq.Context() as ctx,  # 上下文
                make_zmq_socket(
                    ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                ) as socket,  # 输出 socket
            ):
                socket.send_multipart(encoder.encode(outputs))  # 发送

    def eep_handle_engine_core_notification(
        self, notification_type: str | EEPNotificationType
    ):
        """
        Handle notification received from EngineCoreClient
        (forwarded from new core engines).
        """
        # 处理从 EngineCoreClient 收到的通知（从新核心引擎转发）。
        assert self.eep_scaling_state is not None  # 断言缩放状态存在
        if isinstance(notification_type, str):
            # 如果是字符串
            notification_type = EEPNotificationType(notification_type)
            # 转换为枚举
        self.eep_scaling_state.handle_notification(notification_type)
        # 处理通知

    def _eep_scale_up_before_kv_init(self):
        # 弹性 EP 扩容前的 KV 初始化准备
        from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState
        # 延迟导入

        self.eep_scaling_state = ElasticEPScalingState(
            # 创建弹性缩放状态机
            model_executor=self.model_executor,  # 执行器
            engine_core=self,  # 引擎核心
            vllm_config=self.vllm_config,  # 配置
            new_parallel_config=self.vllm_config.parallel_config,  # 并行配置
            worker_type="new",  # 新 worker
            scale_type="scale_up",  # 扩容
            reconfig_request=None,  # 无重配置请求
        )
        self.eep_scaling_state.run_pre_kv_init_states()  # 运行 KV 初始化前状态
        self.process_input_queue_block = False  # 非阻塞输入


class EngineCoreActorMixin:
    """
    Ray actor for running EngineCore in a data parallel context
    """
    # 在数据并行上下文中运行 EngineCore 的 Ray actor

    def __init__(
        self,
        vllm_config: VllmConfig,  # 配置
        addresses: EngineZmqAddresses,  # ZMQ 地址
        dp_rank: int = 0,  # DP rank
        local_dp_rank: int = 0,  # 本地 DP rank
    ):
        # Initialize tracer for distributed tracing if configured.
        # 如果配置了分布式追踪则初始化追踪器。
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.engine_core",  # 模块名
            process_kind="engine_core",  # 进程类型
            process_name=f"DPEngineCoreActor_DP{dp_rank}",  # 进程名
        )

        self.addresses = addresses  # 保存地址
        vllm_config.parallel_config.data_parallel_index = dp_rank  # DP 索引
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank
        # 本地 DP rank

        self._set_nixl_side_channel_host()  # 设置 NIXL 侧通道主机

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error.
        # 在 actor 生命周期中尽早设置 CUDA_VISIBLE_DEVICES
        # 注意：MP 模式在进程创建时设置，Ray 无法用同样方式，因为：
        # 1) Ray 管理所有 Ray worker（包括 DPEngineCoreActor）的生命周期
        # 2) Ray 基于 num_gpus 配置设置 CUDA_VISIBLE_DEVICES
        # 绕过 2 需设置 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES，
        # 但此后创建的 vLLM worker 会带有粘性的 CUDA_VISIBLE_DEVICES：
        # 当 vLLM worker 执行任务时索引粘性设备列表而非直接使用 GPU ID，
        # 可能导致索引越界错误。
        self._set_visible_devices(vllm_config, local_dp_rank)  # 设置可见设备

    @staticmethod
    def _set_nixl_side_channel_host():
        # 设置 NIXL 侧通道主机
        import ray  # 延迟导入

        # The driver-side value is excluded from Ray actor env propagation.
        # Fill in an actor-local default while preserving explicit overrides.
        # 驱动端值不包含在 Ray actor 环境传播中。
        # 填充 actor 本地默认值，同时保留显式覆盖。
        os.environ.setdefault(
            "VLLM_NIXL_SIDE_CHANNEL_HOST", ray.util.get_node_ip_address()
        )
        # 设置环境变量

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        # 设置可见设备
        from vllm.platforms import current_platform  # 延迟导入

        if current_platform.is_xpu():
            # 如果是 XPU
            pass  # 跳过
        else:
            device_control_env_var = current_platform.device_control_env_var
            # 设备控制环境变量
            self._set_assigned_physical_gpu_ids(
                vllm_config, local_dp_rank, device_control_env_var
            )
            # 设置分配的物理 GPU ID

    def _set_assigned_physical_gpu_ids(
        self,
        vllm_config: VllmConfig,  # 配置
        local_dp_rank: int,  # 本地 DP rank
        device_control_env_var: str,  # 设备控制环境变量
    ):
        # 设置分配的物理 GPU ID
        world_size = vllm_config.parallel_config.world_size  # 世界大小
        try:
            physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
                device_control_env_var,  # 环境变量
                local_dp_rank,  # 本地 rank
                world_size,  # 世界大小
                user_assigned_gpu_ids=(  # 用户分配 GPU
                    vllm_config.parallel_config.assigned_physical_gpu_ids
                ),
            )
            vllm_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids
            # 设置 GPU ID
        except IndexError as e:
            # 捕获索引错误
            raise Exception(
                f"Error computing assigned_physical_gpu_ids: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e
            # 抛出错误

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,  # 握手地址
        identity: bytes,  # 身份
        local_client: bool,  # 本地客户端
        vllm_config: VllmConfig,  # 配置
        client_handshake_address: str | None,  # 客户端握手地址
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        # 对于 Ray，无需实际执行握手。
        # 所有地址信息在 actor 创建前已知。
        # 因此，直接产出这些地址。
        yield self.addresses  # 产出地址

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        # 等待引擎核心初始化完成。
        # 这是一个空方法。当对该方法（或 actor 的任何方法）执行
        # ray.get() 返回时，保证 actor 创建（即 __init__）已完成。
        pass  # 空实现

    def run(self):
        """
        Run the engine core busy loop.
        """
        # 运行引擎核心忙循环。
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]  # 运行忙循环
        except SystemExit:
            # 捕获退出
            logger.debug("EngineCore exiting.")  # 记录日志
            raise  # 重新抛出
        except Exception:
            # 捕获异常
            logger.exception("EngineCore encountered a fatal error.")  # 记录
            raise  # 重新抛出
        finally:
            self.shutdown()  # type: ignore[attr-defined]  # 关闭


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""
    # 用于 MoE 模型数据并行情况

    def __init__(
        self,
        vllm_config: VllmConfig,  # 配置
        local_client: bool,  # 本地客户端
        addresses: EngineZmqAddresses,  # 地址
        executor_class: type[Executor],  # 执行器
        log_stats: bool,  # 日志统计
        dp_rank: int = 0,  # DP rank
        local_dp_rank: int = 0,  # 本地 DP rank
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank  # 设置 DP rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        # 初始化 actor 混入
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )
        # 初始化 DP 引擎核心进程


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""
    # 用于非 MoE 和非 DP 情况

    def __init__(
        self,
        vllm_config: VllmConfig,  # 配置
        local_client: bool,  # 本地客户端
        addresses: EngineZmqAddresses,  # 地址
        executor_class: type[Executor],  # 执行器
        log_stats: bool,  # 日志统计
        dp_rank: int = 0,  # DP rank
        local_dp_rank: int = 0,  # 本地 DP rank
    ):
        vllm_config.parallel_config.data_parallel_size = 1  # DP 大小设为 1
        vllm_config.parallel_config.data_parallel_size_local = 1  # 本地 DP 设为 1
        vllm_config.parallel_config.data_parallel_rank = 0  # DP rank 设为 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        # 初始化 actor 混入
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
        # 初始化引擎核心进程