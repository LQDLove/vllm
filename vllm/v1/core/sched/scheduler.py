# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX 文件版权声明：vLLM 项目贡献者
import itertools  # itertools：迭代器工具（用于串联 running 与 resumed 请求）
import time  # time：时间戳（用于记录调度/抢占时间）
from collections import defaultdict, deque  # defaultdict 与 deque（默认字典、双端队列）
from collections.abc import Iterable  # Iterable：可迭代类型标注
from dataclasses import replace  # dataclasses.replace：不可变地替换 dataclass 字段
from typing import Any  # Any 类型标注

from vllm.compilation.cuda_graph import CUDAGraphStat  # CUDA 图统计
from vllm.config import KVEventsConfig, VllmConfig  # KV 事件配置与全局配置
from vllm.distributed.ec_transfer.ec_connector.base import (
    # EC（encoder cache）连接器基类相关
    ECConnectorBase,  # EC 连接器基类
    ECConnectorMetadata,  # EC 连接器元数据
    ECConnectorRole,  # EC 连接器角色
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory  # EC 连接器工厂
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch  # KV 事件发布器与批次
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory  # KV 连接器工厂
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    # KV 连接器 v1 接口
    KVConnectorBase_V1,  # KV 连接器基类 v1
    KVConnectorRole,  # KV 连接器角色
    SupportsHMA,  # 是否支持混合内存分配（HMA）
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata  # KV 连接器元数据
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats  # KV 连接器统计
from vllm.logger import init_logger  # 日志初始化
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    # MoE 路由专家捕获器
    RoutedExpertsManager,  # 路由专家管理器
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry  # 多模态注册表
from vllm.multimodal.encoder_budget import MultiModalBudget  # 多模态预算
from vllm.multimodal.utils import get_mm_features_in_window  # 获取窗口内多模态特征
from vllm.v1.core.encoder_cache_manager import (
    # 编码器缓存管理器
    EncoderCacheManager,  # 编码器缓存管理器
    EncoderDecoderCacheManager,  # 编码器-解码器缓存管理器
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager  # KV 缓存管理器
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector  # KV 缓存指标收集器
from vllm.v1.core.kv_cache_utils import KVCacheBlock  # KV 缓存块
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface  # 调度器接口与暂停状态
from vllm.v1.core.sched.output import (
    # 调度输出数据结构
    CachedRequestData,  # 缓存请求数据
    GrammarOutput,  # 语法输出
    NewRequestData,  # 新请求数据
    ScheduledEncoderInputStats,  # 编码器输入调度统计
    SchedulerOutput,  # 调度器输出
)
from vllm.v1.core.sched.request_queue import (
    # 请求队列
    RequestQueue,  # 请求队列基类
    SchedulingPolicy,  # 调度策略枚举
    create_request_queue,  # 创建请求队列
)
from vllm.v1.core.sched.utils import check_stop, remove_all  # 停止检查与批量移除工具
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs  # 引擎核心事件与输出
from vllm.v1.kv_cache_interface import KVCacheConfig  # KV 缓存配置
from vllm.v1.metrics.perf import ModelMetrics, PerfStats  # 性能指标
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats  # 前缀缓存与调度统计
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput  # 模型运行器输出
from vllm.v1.request import Request, RequestStatus, StreamingUpdate  # 请求、状态、流式更新
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup  # 动态投机解码查找表
from vllm.v1.spec_decode.metrics import SpecDecodingStats  # 投机解码统计
from vllm.v1.structured_output import StructuredOutputGrammar, StructuredOutputManager  # 结构化输出
from vllm.v1.utils import record_function_or_nullcontext  # 函数记录上下文（性能分析）

logger = init_logger(__name__)  # 初始化本模块日志器


class Scheduler(SchedulerInterface):  # 调度器：vLLM v1 核心调度逻辑
    def __init__(
        self,
        vllm_config: VllmConfig,  # 全局配置
        kv_cache_config: KVCacheConfig,  # KV 缓存配置
        structured_output_manager: StructuredOutputManager,  # 结构化输出管理器
        block_size: int,  # KV 缓存块大小
        hash_block_size: int | None = None,  # 哈希块大小（前缀缓存用，可选）
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表
        include_finished_set: bool = False,  # 是否包含已完成请求集合
        log_stats: bool = False,  # 是否记录统计
    ) -> None:
        self.vllm_config = vllm_config  # 保存全局配置
        self.scheduler_config = vllm_config.scheduler_config  # 调度器配置
        self.cache_config = vllm_config.cache_config  # 缓存配置
        self.lora_config = vllm_config.lora_config  # LoRA 配置
        self.kv_cache_config = kv_cache_config  # KV 缓存配置
        self.kv_events_config = vllm_config.kv_events_config  # KV 事件配置
        self.parallel_config = vllm_config.parallel_config  # 并行配置
        self.log_stats = log_stats  # 统计开关
        self.observability_config = vllm_config.observability_config  # 可观测性配置
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None  # KV 指标收集器
        if self.observability_config.kv_cache_metrics:
            # 启用 KV 缓存指标时创建收集器
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,  # 采样频率
            )
        self.structured_output_manager = structured_output_manager  # 结构化输出管理器
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder  # 是否编码器-解码器模型
        self.is_encoder_only = vllm_config.is_encoder_only  # 是否纯编码器模型

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        # include_finished_set 控制 update_from_outputs 返回的 EngineCoreOutputs
        # 是否包含独立的已完成请求 id 集合。当前用于多引擎场景高效跟踪请求生命周期
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # 按 client_index 维护已完成请求 id（启用时），否则为 None
        # Track requests scheduled in prior step (MRV1-only).
        # 跟踪上一步调度的请求（仅 MRV1）
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        # 调度约束
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs  # 最大运行请求数
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        # 单步最大调度 token 数：优先用显式配置，否则回退到批处理 token 上限
        self.max_model_len = vllm_config.model_config.max_model_len  # 模型最大序列长度
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # 是否启用 KV 缓存事件
        # Diffusion models may not sample any tokens for a denoising step.
        # 扩散模型在去噪步可能不采样任何 token
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )
        # 每步采样 token 数：非扩散模型为 1，扩散模型为 0

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        # 为调度器创建 KVConnector。注意每个 Worker 会有对应的 Role=WORKER 连接器
        # KV Connector 用于 P/D 和 offloading 的远程 KV 推送/拉取
        self.connector = None  # KV 连接器（默认无）
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None  # 连接器前缀缓存统计
        self.recompute_kv_load_failures = True  # KV 加载失败时是否重算（默认重算）
        self.defer_block_free = False  # 是否延迟释放块（默认否）
        # Whether a preempted request's in-flight output must be dropped; see
        # KVConnectorBase_V1.requires_kv_delivery.
        # 被抢占请求的进行中输出是否必须丢弃；见 KVConnectorBase_V1.requires_kv_delivery
        self.requires_kv_delivery = False  # 是否要求 KV 交付
        kv_transfer_config = self.vllm_config.kv_transfer_config  # KV 传输配置
        if kv_transfer_config is not None:
            # 配置了 KV 传输（P/D 分离或 offloading）
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            # 断言：编码器-解码器模型暂不支持 KV 连接器
            self.connector = KVConnectorFactory.create_connector(
                # 通过工厂创建调度器角色的 KV 连接器
                config=self.vllm_config,  # 全局配置
                role=KVConnectorRole.SCHEDULER,  # 调度器角色
                kv_cache_config=self.kv_cache_config,  # KV 缓存配置
            )
            if self.log_stats:
                # 记录统计时创建连接器前缀缓存统计
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy  # KV 加载失败策略
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"  # 是否为重算策略

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            # 在批次重叠（异步调度或 PP）时，某步可能仍在写已释放请求的 KV 块。
            # 消费者 KV 连接器可能通过未与该写排序的 load 重新分配并填充这些块，
            # 因此延迟释放它们
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1  # 是否有多个在途批次
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                # 多在途批次且为 KV 消费者时启用延迟释放
                self.defer_block_free = True

            self.requires_kv_delivery = self.connector.requires_kv_delivery  # 记录是否要求 KV 交付

        self.kv_event_publisher = EventPublisherFactory.create(
            # 创建 KV 事件发布器
            self.kv_events_config,  # 事件配置
            self.parallel_config.data_parallel_index,  # DP 索引
        )
        self.ec_connector = None  # EC 连接器（默认无）
        if self.vllm_config.ec_transfer_config is not None:
            # 配置了 EC 传输时创建 EC 连接器
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER  # 调度器角色
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks  # GPU 块总数
        assert num_gpu_blocks is not None and num_gpu_blocks > 0  # 断言有有效 GPU 块

        self.block_size = block_size  # 保存块大小
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size  # 解码上下文并行大小
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size  # prefill 上下文并行大小

        # req_id -> Request
        # 请求 id → Request 的映射
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        # 调度策略
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)  # 解析策略枚举
        except ValueError as e:
            raise ValueError(
                # 未知策略抛出错误
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        # 请求优先级队列
        self.waiting = create_request_queue(self.policy)  # 等待队列（按策略创建）
        # requests skipped in waiting flow due async deps or constraints.
        # 因异步依赖或约束在等待流程中被跳过的请求
        self.skipped_waiting = create_request_queue(self.policy)  # 被跳过的等待队列
        self.running: list[Request] = []  # 运行队列（列表）

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        # 上一步和当前步之间完成的请求 id。用于通知 worker 释放这些请求的缓存状态。
        # 每个调度步结束时清空
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        # 自上次 schedule() 调用以来被抢占的请求 id
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        # 等待流式输入的请求计数。用于计算未完成请求数
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        # KV 连接器：正在异步加载或接收 KV 的请求
        self.finished_recving_kv_req_ids: set[str] = set()  # 完成接收 KV 的请求 id
        self.failed_recving_kv_req_ids: set[str] = set()  # 接收 KV 失败的请求 id

        # Grammar compilation failures to finish as per-request errors in
        # update_from_output.
        # 语法编译失败的请求，将在 update_from_output 中作为错误结束
        self.grammar_compile_error_reqs: set[str] = set()

        # Encoder-related.
        # 编码器相关
        # Calculate encoder cache size if applicable
        # 若适用则计算编码器缓存大小
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config  # 检查模型是否支持多模态输入
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )
        # 多模态预算（支持多模态时创建），否则 None

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        # 注意：纯文本编码器-解码器模型为方便起见也实现为多模态模型
        if self.is_encoder_decoder:
            # 编码器-解码器模型
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )
            # 断言：编码器-解码器模型至多实现一种模态

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        # 最大编码器输入 token 数（编码器计算预算）
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0  # 编码器缓存大小
        manager_cls_obj = vllm_config.ec_manager_config.get_encoder_cache_manager_obj()  # 自定义编码器缓存管理器类
        if manager_cls_obj is not None:
            # 提供了自定义管理器类
            self.encoder_cache_manager = manager_cls_obj(cache_size=encoder_cache_size)
        else:
            self.encoder_cache_manager = (
                EncoderDecoderCacheManager(cache_size=encoder_cache_size)
                if self.is_encoder_decoder
                else EncoderCacheManager(cache_size=encoder_cache_size)
            )
            # 默认：编码器-解码器用 EncoderDecoderCacheManager，否则 EncoderCacheManager
        speculative_config = vllm_config.speculative_config  # 投机解码配置
        self.use_eagle = False  # 是否使用 EAGLE
        self.num_spec_tokens = vllm_config.num_speculative_tokens  # 投机 token 数
        self.num_lookahead_tokens = 0  # 前瞻 token 数（投机解码预留槽位）
        self.dynamic_sd_lookup: list[int] | None = None  # 动态投机解码查找表
        if speculative_config is not None:
            # 配置了投机解码
            if speculative_config.num_speculative_tokens_per_batch_size:
                # 按批大小动态调整投机 token 数
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,  # 映射表
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,  # 最大批大小
                    vllm_num_speculative_tokens=self.num_spec_tokens,  # 投机 token 数
                )
            if speculative_config.use_eagle():
                # 使用 EAGLE
                self.use_eagle = True  # 标记
                self.num_lookahead_tokens = self.num_spec_tokens  # 前瞻 = 投机 token 数
            if speculative_config.uses_draft_model():
                # 使用草稿模型
                self.num_lookahead_tokens = self.num_spec_tokens  # 前瞻 = 投机 token 数
            if speculative_config.use_dflash():
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                # DFlash 需额外一个前瞻槽位：它用 in-fill 风格解码而非标准下一 token 采样，
                # 因此对最后采样 token 有一个查询，加上每个草稿 token 的查询
                self.num_lookahead_tokens = self.num_spec_tokens + 1
            if speculative_config.use_dspark():
                # DSpark drafts a block of num_spec_tokens query tokens in which the
                # anchor itself is the first prediction position (no separate bonus
                # query), so it needs exactly num_spec_tokens lookahead slots.
                # DSpark 起草 num_spec_tokens 个查询 token 的块，锚点本身是第一个预测
                # 位置（无单独奖励查询），因此恰好需要 num_spec_tokens 个前瞻槽位
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        # 创建 KV 缓存管理器
        if hash_block_size is None:
            hash_block_size = block_size  # 未指定哈希块大小则等于块大小
        self.hash_block_size = hash_block_size  # 保存哈希块大小
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,  # KV 缓存配置
            max_model_len=self.max_model_len,  # 模型最大长度
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,  # 最大在途 token
            enable_caching=self.cache_config.enable_prefix_caching,  # 是否启用前缀缓存
            use_eagle=self.use_eagle,  # 是否 EAGLE
            log_stats=self.log_stats,  # 统计开关
            enable_kv_cache_events=self.enable_kv_cache_events,  # KV 事件开关
            dcp_world_size=self.dcp_world_size,  # 解码上下文并行大小
            pcp_world_size=1,  # prefill 上下文并行大小（此处固定 1）
            scheduler_block_size=self.block_size,  # 调度器块大小
            hash_block_size=hash_block_size,  # 哈希块大小
            metrics_collector=self.kv_metrics_collector,  # 指标收集器
            watermark=self.scheduler_config.watermark,  # 水位线
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        # 将 GPU 块池绑定到 KV 连接器。必须在 kv_cache_manager 构造后进行，
        # 这样 block_pool 才可用
        if self.connector is not None:
            # 有连接器时绑定块池
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1  # 是否流水线并行
        self.use_v2_model_runner = vllm_config.use_v2_model_runner  # 是否 v2 模型运行器
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        # 调度器迭代计数器。驱动 V2+PP+async 的解码节流节奏（next_decode_eligible_step）
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        # DP prefill 均衡：跟踪上一个节奏对齐的 prefill 批次是否完全清空等待队列。
        # 此情况下禁用 prefill 节流
        self.prefill_capacity_bound = False  # 是否容量受限
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl  # 是否为完整 ISL 预留
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers  # 是否有 Mamba 层
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing  # 是否需要 KV 缓存清零
        # Blocks that async KV loads will overwrite this step, skipped from
        # zeroing since the zeroing could race the out-of-band write.
        # 异步 KV 加载本步将覆盖的块，跳过清零（清零可能与带外写竞争）
        self._skip_zero_block_ids: set[int] = set()
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # 是否需要 Mamba 块对齐分块（align 缓存模式）
        # A finer prefix_match_unit is configured: a mamba partial tail entry
        # can only be registered by a step ending exactly at the prompt's last
        # hash boundary, so the split adds that stop.
        # 配置了更细的 prefix_match_unit：mamba 部分尾部条目只能由恰好在提示词最后
        # 哈希边界结束的步注册，因此分块会添加该停止点
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
        )
        # 是否存在 Mamba 部分缓存命中

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        # 已调度/已处理的非空步计数。update_from_output 按 FIFO 对每个调度步调用一次，
        # 因此两者保持同步
        self.sched_step_seq = 0  # 已调度步序号
        self.processed_step_seq = 0  # 已处理步序号
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        # (fence_seq, blocks) 的 FIFO：当 processed_step_seq >= fence_seq 时块可安全释放
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None  # 性能指标
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            # 启用统计且启用 MFU 指标时创建
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts  # 是否返回路由专家
        )

        if self.enable_return_routed_experts:
            # 启用返回路由专家
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )
            # 断言：不支持上下文并行

            self.routed_experts_mgr = RoutedExpertsManager(
                # 创建路由专家管理器
                vllm_config=vllm_config,  # 全局配置
                kv_cache_config=kv_cache_config,  # KV 缓存配置
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            # 调度时（前向前）获取的块 ID 快照，即使后续 schedule() 释放了块，
            # update_from_output 仍能读取槽数据（避免异步调度竞争）
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED  # 暂停状态（初始未暂停）

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        # 仍在 prefill 的在途请求（prefill 分块 + 进行中的异步 KV 加载）。
        # 它们的剩余块预留是异步加载的门控
        self._inflight_prefills: set[Request] = set()

    def _mamba_block_aligned_split(
        self,
        request: Request,  # 请求
        num_new_tokens: int,  # 新 token 数
        num_new_local_computed_tokens: int = 0,  # 本地新计算 token 数
        num_external_computed_tokens: int = 0,  # 外部计算 token 数
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" cache mode reusable SSM states are materialized at block
        boundaries, plus mandatory early stops (the prompt's partial-tail hash
        boundary, a detected shared-prefix junction). If a block is larger
        than the configured prefill chunk limit, intermediate chunks keep
        private running state until they reach the next cacheable position.
        """
        # 裁剪 prefill 分块使其在 Mamba 状态必须缓存处结束。
        # "align" 缓存模式下，可复用的 SSM 状态在块边界实体化，
        # 加上强制早停（提示词部分尾部哈希边界、检测到的共享前缀汇合点）。
        # 若块大于配置的 prefill 分块上限，中间分块保持私有运行状态，
        # 直到达到下一个可缓存位置
        start = (
            request.num_computed_tokens  # 已计算 token 数
            + num_new_local_computed_tokens  # 本地新计算
            + num_external_computed_tokens  # 外部计算
        )
        # 分块起点 = 已计算 + 本地新计算 + 外部计算
        # Split only during prefill: `request.num_tokens - 1` extends this to
        # resumed requests replaying their output tokens.
        # 仅 prefill 期间分块：num_tokens - 1 将其扩展到重放输出 token 的恢复请求
        if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
            return num_new_tokens  # 已过 prefill 阶段，不裁剪

        block_size = self.cache_config.block_size  # 块大小
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        # 最后一个状态可缓存的块对齐位置。EAGLE 时 FullAttn 会裁剪最后匹配块，
        # 因此退后一块以避免 Mamba 缓存未命中
        last_cache_position = request.num_tokens - request.num_tokens % block_size  # 向下对齐块边界
        if self.use_eagle:
            # EAGLE 时退后一块
            last_cache_position = max(last_cache_position - block_size, 0)

        end = start + num_new_tokens  # 分块终点
        # Until `last_cache_position`, prefer chunks ending on block
        # boundaries. When a block cannot fit in any configured prefill chunk,
        # allow sub-block progress and re-align at the next reachable boundary.
        # 在到达 last_cache_position 前，优先以块边界结束分块。当块无法放入
        # 任何配置的 prefill 分块时，允许子块进度并在下一个可达边界重新对齐
        if end < last_cache_position:
            max_prefill_tokens = self.max_num_scheduled_tokens  # 最大调度 token 数
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold  # 长 prefill 阈值
            if long_prefill_threshold > 0:
                # 有正阈值时取更小者
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size  # 向下对齐到块边界
            if aligned_end > start or block_size <= max_prefill_tokens:
                # 对齐终点有效或块能放下时，采用对齐终点
                end = aligned_end

        next_block_boundary = (start // block_size + 1) * block_size  # 下一个块边界
        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit
            else 0
        )
        # 尾部边界：mamba 部分缓存命中时，提示词按哈希块大小对齐的位置
        stops = (
            # Resumed mid-block (fine-grained partial hash hit): re-align to
            # the block grid before running on, so the crossed boundary's
            # state is materialized (unless it is past the cacheable range).
            # 块中间恢复（细粒度部分哈希命中）：继续运行前重新对齐到块网格，
            # 使跨越边界的状态实体化（除非超出可缓存范围）
            next_block_boundary
            if start % block_size != 0 and next_block_boundary <= last_cache_position
            else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            # 分块中途绝不越过最后可缓存块边界
            last_cache_position,
            # Fine-grained hits: the prompt's partial-tail entry can only be
            # registered by a chunk ending exactly at its last hash boundary.
            # 细粒度命中：提示词部分尾部条目只能由恰好在其最后哈希边界结束的分块注册
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            # Marconi shared-prefix junction, block-floored (a sub-block
            # junction's state is not separately cacheable): cache its state
            # so sibling requests sharing the prefix can reuse it.
            # Marconi 共享前缀汇合点，向下取块边界（子块汇合点状态不可单独缓存）：
            # 缓存其状态使共享前缀的兄弟请求可复用
            start + (request.shared_prefix_boundary - start) // block_size * block_size
            if start < request.shared_prefix_boundary < end
            else 0,
        )
        # Stop at the earliest mandatory position strictly inside the chunk.
        # 在分块内部最早的强制位置停止
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)  # 返回裁剪后的新 token 数

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:  # 核心调度方法
        self.current_step += 1  # 调度步计数自增
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
        # 调度算法说明：调度器没有"解码阶段"或"prefill 阶段"之分。
        # 每个请求只有 num_computed_tokens 和 num_tokens_with_spec。
        # 每步调度器为请求分配 token，使 num_computed_tokens 追上 num_tokens_with_spec。
        # 这足够通用，覆盖分块 prefill、前缀缓存、投机解码和未来的"跳跃解码"优化

        scheduled_new_reqs: list[Request] = []  # 本步新调度的请求
        scheduled_resumed_reqs: list[Request] = []  # 本步恢复的（被抢占过的）请求
        scheduled_running_reqs: list[Request] = []  # 本步继续运行的请求
        preempted_reqs: list[Request] = []  # 本步被抢占的请求

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}  # 请求 → 新分配块
        num_scheduled_tokens: dict[str, int] = {}  # 请求 → 本步调度 token 数
        token_budget = self.max_num_scheduled_tokens  # 本步 token 预算
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            # 完全暂停时不调度任何请求
            token_budget = 0

        # Encoder-related.
        # 编码器相关
        scheduled_encoder_inputs: dict[str, list[int]] = {}  # 请求 → 调度的编码器输入索引
        encoder_compute_budget = self.max_num_encoder_input_tokens  # 编码器计算预算
        # Spec decode-related.
        # 投机解码相关
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}  # 请求 → 投机解码 token
        # Whether the running batch contains any prefill requests.
        # 运行批次是否包含任何 prefill 请求
        prefill_scheduled = False

        # For logging.
        # 用于日志
        scheduled_timestamp = time.monotonic()  # 调度时间戳

        self.kv_cache_manager.new_step_starts()  # 通知 KV 缓存管理器新步开始

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        # DP prefill 均衡：在节流（非节奏对齐）步上，除非饱和否则推迟所有 prefill 计算
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound  # 节流且非容量受限
        ) and any(not r.is_prefill_chunk for r in self.running)  # 且运行队列有非 prefill 请求

        # First, schedule the RUNNING requests.
        # 首先调度 RUNNING 请求
        req_index = 0  # 请求索引
        while req_index < len(self.running) and token_budget > 0:
            # 遍历运行队列且预算未耗尽
            request = self.running[req_index]  # 取当前请求

            if (
                request.num_output_placeholders > 0  # 有输出占位符（异步调度）
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                # 即 (num_computed_tokens + 1) - (num_output_placeholders - 1)。
                # 由于输出占位符也计入已计算 token，减去 (num_output_placeholders - 1)
                # 以移除草稿 token，即使全部被拒也能确保无需更多步
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                # 异步调度：确定上一步已达 max_tokens 时避免调度额外一步。
                # 不调度部分草稿 token，因为这会阻碍统一解码优化
                req_index += 1  # 跳过
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                # V2+PP+async：强制同一请求两次解码间隔 pp_size 步，
                # 以匹配 worker 侧采样 token 广播槽环节奏
                req_index += 1  # 尚未到可解码步，跳过
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                # DP prefill 均衡：将此进行中的 prefill 分块推迟到节奏对齐步；
                # 解码仍运行以填满本步
                req_index += 1  # 推迟 prefill 分块
                continue

            num_new_tokens = (
                request.num_tokens_with_spec  # 含投机 token 的总 token 数
                + request.num_output_placeholders  # 加输出占位符
                - request.num_computed_tokens  # 减已计算
            )
            # 本步需新调度的 token 数
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                # 长 prefill 阈值限制
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)  # 不超过剩余预算

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            # 确保输入位置不超过模型最大长度（投机解码时必需）
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len  # 模型最大长度
                - request.num_computed_tokens  # 减已计算
                - self.num_sampled_tokens_per_step,  # 减每步采样数（留采样空间）
            )

            # Schedule encoder inputs.
            # 调度编码器输入
            encoder_inputs_to_schedule = None  # 待调度编码器输入
            external_load_encoder_input: list[int] = []  # 外部加载的编码器输入
            new_encoder_compute_budget = encoder_compute_budget  # 新编码器计算预算
            if request.has_encoder_inputs:
                # 请求有编码器输入（多模态）
                (
                    encoder_inputs_to_schedule,  # 待调度编码器输入
                    num_new_tokens,  # 可能被调整的新 token 数
                    new_encoder_compute_budget,  # 更新后预算
                    external_load_encoder_input,  # 外部加载编码器输入
                ) = self._try_schedule_encoder_inputs(
                    request,  # 请求
                    request.num_computed_tokens,  # 已计算 token 数
                    num_new_tokens,  # 新 token 数
                    encoder_compute_budget,  # 编码器计算预算
                    shift_computed_tokens=1 if self.use_eagle else 0,  # EAGLE 偏移
                )

            if self.need_mamba_block_aligned_split:
                # 需要 Mamba 块对齐分块
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens  # 裁剪 token 数
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # 请求无法调度的原因：1. 无新 token（PP>1 未结束/异步已达上限）
                # 2. 编码器预算耗尽 3. 编码器缓存耗尽 4. Mamba align 模式块对齐分块预算不足
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                # 用 continue 而非 break，不严格遵循 FCFS，允许低优先级请求被调度
                req_index += 1  # 跳过该请求
                continue

            # Schedule newly needed KV blocks for the request.
            # 为请求调度新需要的 KV 块
            with record_function_or_nullcontext("schedule: allocate_slots"):
                # 性能记录上下文：分配槽位
                while True:
                    # 循环直到分配成功或无可抢占
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,  # 请求
                        num_new_tokens,  # 新 token 数
                        num_lookahead_tokens=self.num_lookahead_tokens,  # 前瞻 token 数
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        # 分配成功，请求可调度
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    # 分配失败，抢占最低优先级请求
                    if self.policy == SchedulingPolicy.PRIORITY:
                        # 优先级策略：抢占优先级最低者
                        preempted_req = max(
                            self.running,  # 运行队列
                            key=lambda r: (r.priority, r.arrival_time),  # 优先级低且到达晚者
                        )
                        self.running.remove(preempted_req)  # 从运行队列移除
                        if preempted_req in scheduled_running_reqs:
                            # 若该请求本步已被调度，需回滚其调度信息
                            preempted_req_id = preempted_req.request_id  # 请求 id
                            scheduled_running_reqs.remove(preempted_req)  # 移出已调度列表
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)  # 归还 token 预算
                            req_to_new_blocks.pop(preempted_req_id)  # 移除块记录
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)  # 移除投机 token
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None  # 移除编码器输入
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                # 被抢占请求本步有编码器输入时恢复编码器计算预算
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)  # 各输入嵌入数
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore  # 恢复预算
                            req_index -= 1  # 回退索引（重新审视当前位置）
                    else:
                        # FCFS 策略：抢占运行队列末尾（最新）请求
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        # 执行抢占
                        preempted_req,  # 被抢占请求
                        scheduled_timestamp,  # 时间戳
                        drop_stale_output=self.requires_kv_delivery,  # 是否丢弃陈旧输出
                    )
                    preempted_reqs.append(preempted_req)  # 记录被抢占请求
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        # 抢占了自身，无可抢占对象，本请求无法调度
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                # 仍无法分配，停止调度运行队列
                break

            # Schedule the request.
            # 调度该请求
            scheduled_running_reqs.append(request)  # 加入已调度运行列表
            prefill_scheduled |= request.is_prefill_chunk  # 更新 prefill 标志
            request_id = request.request_id  # 请求 id
            req_to_new_blocks[request_id] = new_blocks  # 记录新分配块
            num_scheduled_tokens[request_id] = num_new_tokens  # 记录调度 token 数
            token_budget -= num_new_tokens  # 扣减预算
            req_index += 1  # 索引前进

            # Speculative decode related.
            # 投机解码相关
            if request.spec_token_ids:
                # 请求有投机 token
                num_scheduled_spec_tokens = (
                    num_new_tokens  # 新调度 token 数
                    + request.num_computed_tokens  # 加已计算
                    - request.num_tokens  # 减真实 token 数
                    - request.num_output_placeholders  # 减输出占位符
                )
                # 计算本步调度的投机 token 数
                if num_scheduled_spec_tokens > 0:
                    # 有投机 token 被调度
                    spec_token_ids = request.spec_token_ids  # 取投机 token id
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        # 超出调度数量则截断
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids  # 记录

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                # 新的投机 token 将在下一步前的 update_draft_token_ids 中设置
                request.spec_token_ids = []  # 清空已消费的投机 token

            # Encoder-related.
            # 编码器相关
            if encoder_inputs_to_schedule:
                # 有编码器输入需调度
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule  # 记录
                # Allocate the encoder cache.
                # 分配编码器缓存
                for i in encoder_inputs_to_schedule:
                    # 遍历各编码器输入
                    self.encoder_cache_manager.allocate(request, i)  # 分配缓存
                    if self.ec_connector is not None:
                        # 有 EC 连接器时更新状态
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget  # 更新预算
            if external_load_encoder_input:
                # 有外部加载的编码器输入
                for i in external_load_encoder_input:
                    # 遍历并分配
                    self.encoder_cache_manager.allocate(request, i)  # 分配缓存
                    if self.ec_connector is not None:
                        # 有 EC 连接器时更新状态
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        # 记录已调度运行请求中的 LoRA
        scheduled_loras: set[int] = set()  # 已调度 LoRA 集合
        if self.lora_config:
            # 启用 LoRA
            scheduled_loras = set(
                req.lora_request.lora_int_id  # LoRA 整数 id
                for req in scheduled_running_reqs  # 遍历已调度运行请求
                if req.lora_request and req.lora_request.lora_int_id > 0  # 有有效 LoRA
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras  # 断言不超过最大 LoRA 数

        # Next, schedule the WAITING requests.
        # 接下来调度 WAITING 请求
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # 本步无抢占且未暂停时才调度等待队列
            step_skipped_waiting = create_request_queue(self.policy)  # 本步跳过的等待队列

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # 等待队列非空且预算未耗尽
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                # 暂停的流式会话不在 running 中但仍占用模型运行器请求槽
                num_running = len(self.running) + self.num_waiting_for_streaming_input  # 有效运行数
                if num_running >= self.max_num_running_reqs:
                    # 达到最大运行请求数
                    break

                request_queue = self._select_waiting_queue_for_scheduling()  # 选择等待队列
                assert request_queue is not None  # 断言队列非空

                request = request_queue.peek_request()  # 查看队首请求
                request_id = request.request_id  # 请求 id

                # try to promote blocked statuses while traversing skipped queue.
                # 遍历跳过队列时尝试提升阻塞状态
                if self._is_blocked_waiting_status(
                    request.status  # 是阻塞等待状态
                ) and not self._try_promote_blocked_waiting_request(request):
                    # 且无法提升（仍未就绪）
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        # 仍在等待远程 KV
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",  # 调试日志
                            request_id,
                        )
                    request_queue.pop_request()  # 从队列弹出
                    step_skipped_waiting.prepend_request(request)  # 放入本步跳过队列
                    continue

                if (
                    request.num_stale_output_tokens > 0  # 有陈旧输出 token
                    and not request.drop_stale_output  # 且不丢弃
                ):
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    # 可交付的陈旧输出仍在途：现在恢复可能对之后交付的输出位置重采样。
                    # 它会在流水线深度内排空
                    request_queue.pop_request()  # 弹出
                    step_skipped_waiting.prepend_request(request)  # 放入跳过队列
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                # 检查添加请求仍满足 max_loras 约束
                if (
                    self.lora_config  # 启用 LoRA
                    and request.lora_request  # 请求有 LoRA
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras  # 已达最大 LoRA 数
                        and request.lora_request.lora_int_id not in scheduled_loras  # 且是新 LoRA
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    # 调度会超出 max_loras，跳过
                    request_queue.pop_request()  # 弹出
                    step_skipped_waiting.prepend_request(request)  # 放入跳过队列
                    continue

                num_external_computed_tokens = 0  # 外部计算 token 数
                load_kv_async = False  # 是否异步加载 KV
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0  # 连接器前缀缓存查询/命中
                did_prefix_cache_lookup = False  # 是否做了前缀缓存查找

                # Get already-cached tokens.
                # 获取已缓存的 token
                if request.num_computed_tokens == 0:
                    # 首次调度（无已计算 token）
                    did_prefix_cache_lookup = True  # 标记做了前缀缓存查找
                    hit_diverged = False  # 命中是否分歧
                    # Get locally-cached tokens.
                    # 获取本地缓存的 token
                    if self.connector is not None:
                        # 有 KV 连接器：混合感知查找，可能跨组分歧
                        (
                            new_computed_blocks,  # 新计算的块
                            num_new_local_computed_tokens,  # 本地新计算 token 数
                            request.shared_prefix_boundary,  # 共享前缀边界
                            hit_diverged,  # 命中是否分歧
                        ) = self.kv_cache_manager.get_computed_blocks_for_connector(
                            request  # 请求
                        )
                    else:
                        # 无连接器：普通前缀缓存查找
                        (
                            new_computed_blocks,  # 新计算的块
                            num_new_local_computed_tokens,  # 本地新计算 token 数
                            # Marconi shared-prefix junction to pin; 0 if none.
                            # 要固定的 Marconi 共享前缀汇合点；无则为 0
                            request.shared_prefix_boundary,
                        ) = self.kv_cache_manager.get_computed_blocks(request)

                    # Get externally-cached tokens if using a KVConnector.
                    # 使用 KV 连接器时获取外部缓存的 token
                    if self.connector is not None:
                        # 有连接器
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        # 向连接器呈现块对齐的本地命中，使严格更长的远程命中能取代
                        # 本地子块尾部而不与其写时复制竞争
                        partial_tail = num_new_local_computed_tokens % self.block_size  # 部分尾部
                        block_aligned_local = (
                            num_new_local_computed_tokens - partial_tail  # 块对齐的本地命中
                        )
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, block_aligned_local  # 查询远程匹配 token 数
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            # 连接器无法确定匹配 token 数，请求不可调度
                            request_queue.pop_request()  # 弹出
                            step_skipped_waiting.prepend_request(request)  # 放入跳过队列
                            continue

                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            # 远程严格超过完整本地命中：丢弃子块尾部以免需 CoW，让加载覆盖它。
                            # 从本地计算块中修剪部分块，使其不从缓存采纳
                            new_computed_blocks = (
                                self.kv_cache_manager.truncate_computed_blocks(
                                    new_computed_blocks, block_aligned_local  # 截断到块对齐
                                )
                            )
                            num_new_local_computed_tokens = block_aligned_local  # 用块对齐值
                            num_external_computed_tokens = ext_tokens  # 外部计算 token 数
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            # 远程未超过完整本地命中：保留本地子块尾部，不加载外部
                            num_external_computed_tokens = 0  # 无外部 token
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            # 无远程加载 -> 非异步加载步；清除避免下方断言
                            load_kv_async = False
                        else:
                            # 无部分尾部
                            num_external_computed_tokens = ext_tokens  # 外部计算 token 数

                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            # 无外部 token 支持更深的本地命中，其恢复边界将无有效 Mamba 状态。
                            # 调和到所有组一致的边界
                            (
                                new_computed_blocks,  # 重新获取块
                                num_new_local_computed_tokens,  # 本地计算数
                                request.shared_prefix_boundary,  # 共享前缀边界
                            ) = self.kv_cache_manager.get_computed_blocks(request)

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens  # 查询数
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens  # 命中数

                    # Total computed tokens (local + external).
                    # 总计算 token 数（本地 + 外部）
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens  # 断言不超过总 token

                    # Skip request with pending mm encoding prefetches
                    # 跳过有未完成多模态编码预取的请求
                    if (
                        self.ec_connector is not None  # 有 EC 连接器
                        and request.mm_features  # 有多模态特征
                        and not self.ec_connector.ensure_cache_available(
                            request, num_computed_tokens  # 确保缓存可用失败
                        )
                    ):
                        request_queue.pop_request()  # 弹出
                        step_skipped_waiting.prepend_request(request)  # 放入跳过队列
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    # 跟踪首次调度的 prefill，而非抢占后的重复 prefill
                    if request.prefill_stats and request.num_preemptions <= 0:
                        # 有 prefill 统计且未被抢占过
                        assert num_computed_tokens <= request.num_prompt_tokens  # 断言不超提示词
                        request.prefill_stats.set(
                            # 设置 prefill 统计
                            num_prompt_tokens=request.num_prompt_tokens,  # 提示词 token 数
                            num_local_cached_tokens=num_new_local_computed_tokens,  # 本地缓存
                            num_external_cached_tokens=num_external_computed_tokens,  # 外部缓存
                        )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    # KV 传输：异步 KV 接收完成后，WAITING 请求的 num_computed_tokens > 0
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks  # 空块
                    num_new_local_computed_tokens = 0  # 本地新计算为 0
                    num_computed_tokens = request.num_computed_tokens  # 用已有计算数

                encoder_inputs_to_schedule = None  # 待调度编码器输入
                external_load_encoder_input = []  # 外部加载编码器输入
                new_encoder_compute_budget = encoder_compute_budget  # 新编码器预算
                pad_spec_decode = False  # 是否填充投机解码

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    # 加载远程 KV，不为新工作分配
                    assert num_external_computed_tokens > 0  # 断言有外部 token
                    num_new_tokens = 0  # 无新 token
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    # DP prefill 均衡：将本步本地 prefill 计算推迟到节奏对齐步
                    break  # 停止调度等待队列
                else:
                    # Number of tokens to be scheduled.
                    # 要调度的 token 数
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    # 用 num_tokens 而非 num_prompt_tokens，以考虑有输出 token 的恢复请求
                    num_new_tokens = request.num_tokens - num_computed_tokens  # 新 token 数

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    # Not for diffusion where draft tokens can't be padded.
                    # 将新解码请求填充到统一投机解码大小以保留本步完整 cudagraph。
                    # 扩散模型不适用（草稿 token 无法填充）
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)  # 静态投机解码
                        and self.num_sampled_tokens_per_step > 0  # 有采样
                        and num_new_tokens == 1  # 单 token（纯解码）
                        and (scheduled_running_reqs and not prefill_scheduled)  # 有运行请求且无 prefill
                    ):
                        num_new_tokens = 1 + self.num_spec_tokens  # 填充到含投机 token
                        if (
                            num_new_tokens > token_budget  # 超预算
                            or num_computed_tokens + num_new_tokens > self.max_model_len  # 超模型长度
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            # 宁可调度过也不调度未填充的
                            break
                        pad_spec_decode = True  # 标记已填充

                    threshold = self.scheduler_config.long_prefill_token_threshold  # 长 prefill 阈值
                    if 0 < threshold < num_new_tokens:
                        # 阈值限制
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    # 必须显式启用分块 prefill 才允许池化请求分块
                    if (
                        not self.scheduler_config.enable_chunked_prefill  # 未启用分块 prefill
                        and num_new_tokens > token_budget  # 且超预算
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        # 未启用分块 prefill 时可在此停止调度
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)  # 不超过预算
                    assert num_new_tokens > 0  # 断言为正

                    # Schedule encoder inputs.
                    # 调度编码器输入
                    if request.has_encoder_inputs:
                        # 请求有编码器输入
                        (
                            encoder_inputs_to_schedule,  # 待调度编码器输入
                            num_new_tokens,  # 调整后的新 token 数
                            new_encoder_compute_budget,  # 新编码器预算
                            external_load_encoder_input,  # 外部加载编码器输入
                        ) = self._try_schedule_encoder_inputs(
                            request,  # 请求
                            num_computed_tokens,  # 已计算 token 数
                            num_new_tokens,  # 新 token 数
                            encoder_compute_budget,  # 编码器计算预算
                            shift_computed_tokens=1 if self.use_eagle else 0,  # EAGLE 偏移
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            # 请求不可调度
                            break

                # Skip block alignment when setting up async receive (no local work).
                # 设置异步接收时跳过块对齐（无本地工作）
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    # 需 Mamba 块对齐且非异步加载
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,  # 请求
                        num_new_tokens,  # 新 token 数
                        num_new_local_computed_tokens,  # 本地计算数
                        num_external_computed_tokens,  # 外部计算数
                    )
                    if num_new_tokens == 0:
                        # 裁剪后无 token
                        break

                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                # 异步 KV 加载期间尚未运行前向。稍后分配投机前瞻槽，
                # 避免本地与远程块数不匹配
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0  # 是否限制前瞻
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens  # 有效前瞻数
                )

                # Determine if we need to allocate cross-attention blocks.
                # 确定是否需要分配交叉注意力块
                num_encoder_tokens = 0  # 编码器 token 数
                if (
                    self.is_encoder_decoder  # 编码器-解码器模型
                    and request.has_encoder_inputs  # 有编码器输入
                    and encoder_inputs_to_schedule  # 有编码器输入被调度
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)  # 各输入嵌入数
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0  # 预留块数
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    # 异步加载在整个传输期间持有其块且无前向进展，此处不可抢占。
                    # 仅当能放入（空闲 - 其他在途预留）时才接纳，避免死锁和可预测的抢占
                    reserved_blocks = self._inflight_prefill_reserved_blocks()  # 在途 prefill 预留块

                new_blocks = self.kv_cache_manager.allocate_slots(
                    # 分配槽位
                    request,  # 请求
                    num_new_tokens,  # 新 token 数
                    num_new_computed_tokens=num_new_local_computed_tokens,  # 本地新计算数
                    new_computed_blocks=new_computed_blocks,  # 新计算块
                    num_lookahead_tokens=effective_lookahead_tokens,  # 有效前瞻数
                    num_external_computed_tokens=num_external_computed_tokens,  # 外部计算数
                    delay_cache_blocks=load_kv_async,  # 异步加载时延迟缓存块
                    num_encoder_tokens=num_encoder_tokens,  # 编码器 token 数
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,  # 完整序列须放入
                    reserved_blocks=reserved_blocks,  # 预留块
                    has_scheduled_reqs=bool(self.running),  # 是否有已调度请求
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    # 请求无法调度

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    # 需从编码器缓存管理器撤销该请求
                    if request.has_encoder_inputs:
                        # 有编码器输入则释放
                        self.encoder_cache_manager.free(request)
                    break  # 停止调度等待队列

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                # KV 传输：连接器用此信息判断是否需要加载
                if self.connector is not None:
                    # 有连接器
                    self.connector.update_state_after_alloc(
                        # 分配后更新连接器状态
                        request,  # 请求
                        self.kv_cache_manager.get_blocks(request_id),  # 请求的块
                        num_external_computed_tokens,  # 外部计算数
                    )
                    if (
                        self.connector_prefix_cache_stats is not None  # 有连接器前缀缓存统计
                        and connector_prefix_cache_queries != 0  # 且有查询
                    ):
                        self.connector_prefix_cache_stats.record(
                            # 记录统计
                            num_tokens=connector_prefix_cache_queries,  # 查询 token 数
                            num_hits=connector_prefix_cache_hits,  # 命中数
                            preempted=request.num_preemptions > 0,  # 是否被抢占过
                        )

                # Record at admission so unscheduled lookups are not counted.
                # 在接纳时记录，使未调度的查找不被计数
                if did_prefix_cache_lookup:
                    # 做了前缀缓存查找
                    self.kv_cache_manager.record_prefix_cache_stats(
                        request, num_new_local_computed_tokens  # 记录前缀缓存统计
                    )

                request = request_queue.pop_request()  # 从队列弹出请求
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    # 异步加载时，分配内存并将请求放入 WAITING_FOR_REMOTE_KVS 状态
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS  # 设为等待远程 KV
                    step_skipped_waiting.prepend_request(request)  # 放入跳过队列
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    # 即使 KV 尚未加载也设置 num_computed_tokens。
                    # 在请求完成 KV 传输前不会在任何地方使用它
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    # 若连接器报告传输错误，num_computed_tokens 将在
                    # _update_requests_with_invalid_blocks 中相应重设
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    # 传输结束（无论成功与否）时，num_computed_tokens 将正确反映
                    # 计算 token 数。_update_waiting_for_remote_kv 随后只缓存成功加载的 token
                    request.num_computed_tokens = num_computed_tokens  # 设置已计算数
                    self._inflight_prefills.add(request)  # 加入在途 prefill 集合
                    if self.needs_kv_cache_zeroing:
                        # Skip zeroing of the blocks the async load will
                        # overwrite; the zeroing could race the write.
                        # 跳过对异步加载将覆盖块的清零；清零可能与写竞争
                        self._skip_zero_block_ids.update(
                            # 记录跳过清零的块 id
                            self.kv_cache_manager.get_zeroing_block_ids_in_range(
                                request.request_id,  # 请求 id
                                num_new_local_computed_tokens,  # 本地计算起点
                                num_computed_tokens,  # 计算终点
                            )
                        )
                    continue

                self.running.append(request)  # 加入运行队列
                if self.log_stats:
                    # 记录调度事件
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp  # 已调度事件
                    )
                if request.status == RequestStatus.WAITING:
                    # 首次等待
                    scheduled_new_reqs.append(request)  # 加入新请求列表
                elif request.status == RequestStatus.PREEMPTED:
                    # 被抢占后恢复
                    scheduled_resumed_reqs.append(request)  # 加入恢复请求列表
                else:
                    # 非法状态
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    # 记录 LoRA
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id  # 记录新分配块
                )
                num_scheduled_tokens[request_id] = num_new_tokens  # 记录调度 token 数
                token_budget -= num_new_tokens  # 扣减预算
                request.status = RequestStatus.RUNNING  # 设为运行状态
                request.num_computed_tokens = num_computed_tokens  # 设置已计算数
                if pad_spec_decode:
                    # 填充投机解码
                    scheduled_spec_decode_tokens[request_id] = [
                        -1  # 占位符
                    ] * self.num_spec_tokens
                # Only track requests that will still be prefilling after this chunk.
                # 仅跟踪本分块后仍在 prefill 的请求
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    # 本分块后仍未完成
                    self._inflight_prefills.add(request)  # 加入在途 prefill
                # Encoder-related.
                # 编码器相关
                if encoder_inputs_to_schedule:
                    # 有编码器输入被调度
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule  # 记录
                    # Allocate the encoder cache.
                    # 分配编码器缓存
                    for i in encoder_inputs_to_schedule:
                        # 遍历分配
                        self.encoder_cache_manager.allocate(request, i)  # 分配缓存
                        if self.ec_connector is not None:
                            # 有 EC 连接器时更新状态
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget  # 更新预算
                # Allocate for external load encoder cache
                # 为外部加载的编码器缓存分配
                if external_load_encoder_input:
                    # 有外部加载编码器输入
                    for i in external_load_encoder_input:
                        # 遍历分配
                        self.encoder_cache_manager.allocate(request, i)  # 分配缓存
                        if self.ec_connector is not None:
                            # 有 EC 连接器时更新状态
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            # 将本遍跳过的请求重新入队，置于更旧的跳过项之前
            if step_skipped_waiting:
                # 本步有跳过请求
                self.skipped_waiting.prepend_requests(step_skipped_waiting)  # 前置合并

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            # DP prefill 均衡：在接纳了 prefill 的步上记录是否容量受限
            if not defer_prefills:
                # 未推迟 prefill
                self.prefill_capacity_bound = bool(self.waiting)  # 等待队列非空则容量受限

        # Check if the scheduling constraints are satisfied.
        # 检查调度约束是否满足
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())  # 总调度 token 数
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens  # 断言不超上限

        assert token_budget >= 0  # 断言预算非负
        assert len(self.running) <= self.max_num_running_reqs  # 断言运行数不超上限
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        # 由于运行队列中某些请求本步可能未被调度，已调度请求总数可小于 len(running)
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs  # 三类已调度请求总数
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        # 获取运行队列所有请求的最长公共前缀。可用于级联注意力
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)  # 各 KV 组公共前缀块数
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            # 性能记录上下文
            if self.running:
                # 运行队列非空
                any_request_id = self.running[0].request_id  # 任取一请求 id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)  # 计算公共前缀块数
                )

        # Construct the scheduler output.
        # 构造调度器输出
        if self.use_v2_model_runner:
            # v2 模型运行器：恢复请求并入新请求
            scheduled_new_reqs.extend(scheduled_resumed_reqs)  # 合并
            scheduled_resumed_reqs.clear()  # 清空恢复列表
            new_reqs_data = [
                NewRequestData.from_request(
                    req,  # 请求
                    req_to_new_blocks[req.request_id].get_block_ids(),  # 块 id
                    req._all_token_ids,  # 全部 token id（v2 需显式传入）
                )
                for req in scheduled_new_reqs  # 遍历新请求
            ]
        else:
            # 非 v2 模型运行器
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()  # 请求与块 id
                )
                for req in scheduled_new_reqs  # 遍历新请求
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            # 性能记录上下文：构造缓存请求数据
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,  # 运行请求
                scheduled_resumed_reqs,  # 恢复请求
                num_scheduled_tokens,  # 调度 token 数
                scheduled_spec_decode_tokens,  # 投机解码 token
                req_to_new_blocks,  # 新分配块
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        # 记录本步调度的请求 id（仅 MRV1）
        if not self.use_v2_model_runner:
            # 非 v2 运行器
            self.prev_step_scheduled_req_ids.clear()  # 清空上一步记录
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())  # 更新为本步

        # Producer partial-tail hand-off for external KV connectors. Drained
        # before the CoW retentions are released below, so the pin lands while
        # the cow block still holds a retention ref. Without a producer-side
        # connector nothing consumes the hand-off, so skip the drain (and its
        # pin); the manager drops stale entries when the request's blocks are
        # popped for free.
        # 外部 KV 连接器的生产者部分尾部交接。在下方 CoW 保留释放前排空，
        # 使 pin 在 cow 块仍持有保留引用时落位。无生产者侧连接器则无人消费交接，
        # 跳过排空（及其 pin）；管理器在请求块弹出释放时丢弃陈旧条目
        pending_partial_tail_offloads = None  # 待处理部分尾部卸载
        if (
            self.connector is not None  # 有连接器
            and self.vllm_config.kv_transfer_config is not None  # 有 KV 传输配置
            and self.vllm_config.kv_transfer_config.is_kv_producer  # 且为 KV 生产者
        ):
            pending_partial_tail_offloads = (
                self.kv_cache_manager.take_partial_tail_offloads() or None  # 取部分尾部卸载
            )

        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()  # 取 KV 缓存块拷贝与 CoW 保留块
        )
        if kv_cache_block_copies:
            # 有块拷贝
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            # 拷贝随本步执行；其后第一个非空步获得序号 sched_step_seq + 1
            # （0 token 步不推进序号），其完成意味着拷贝已运行
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)  # 延迟释放 CoW 保留块
        pending_kv_cache_block_copies = kv_cache_block_copies or None  # 待处理块拷贝

        # Dynamic speculative decoding: compute optimal K
        # 动态投机解码：计算最优 K
        num_spec_tokens_to_schedule = self.num_spec_tokens  # 默认投机 token 数
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            # 有动态查找表且有调度请求
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
                len(num_scheduled_tokens)  # 按批大小查表
            ]

        scheduled_encoder_input_stats = None  # 编码器输入调度统计
        if (
            self.log_stats  # 记录统计
            and self.observability_config.enable_logging_iteration_details  # 启用迭代详情日志
        ):
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
                scheduled_encoder_inputs  # 构造编码器输入统计
            )

        scheduler_output = SchedulerOutput(
            # 构造调度器输出对象
            scheduled_new_reqs=new_reqs_data,  # 新请求数据
            scheduled_cached_reqs=cached_reqs_data,  # 缓存请求数据
            num_scheduled_tokens=num_scheduled_tokens,  # 调度 token 数
            total_num_scheduled_tokens=total_num_scheduled_tokens,  # 总调度 token 数
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,  # 投机解码 token
            scheduled_encoder_inputs=scheduled_encoder_inputs,  # 编码器输入
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,  # 编码器输入统计
            num_common_prefix_blocks=num_common_prefix_blocks,  # 公共前缀块数
            preempted_req_ids=self.reset_preempted_req_ids,  # 被抢占请求 id
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            # finished_req_ids 是调度器中的既有状态，而非本步新调度。
            # 它包含上一步与当前步之间完成的请求 id
            finished_req_ids=self.finished_req_ids,  # 已完成请求 id
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),  # 已释放多模态哈希
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),  # 需清零的新块 id
            kv_cache_block_copies=pending_kv_cache_block_copies,  # KV 缓存块拷贝
            partial_tail_offloads=pending_partial_tail_offloads,  # 部分尾部卸载
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,  # 投机 token 数
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),  # EC 管理器元数据
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        # 该函数多用途：1. 规划 KV 缓存存储 2. 将所有 KV 加载/保存操作
        # 包装为不透明对象 3. 清除连接器内部状态
        if self.connector is not None:
            # 有 KV 连接器
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)  # 构造连接器元数据
            scheduler_output.kv_connector_metadata = meta  # 设置到输出

        # Build the connector meta for ECConnector
        # 为 EC 连接器构造元数据
        if self.ec_connector is not None:
            # 有 EC 连接器
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output  # 构造 EC 元数据
            )
            scheduler_output.ec_connector_metadata = ec_meta  # 设置到输出

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        # 仅为非空步推进围栏（实际写 KV 且其输出稍后在 update_from_output 处理的步）
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            # 延迟释放且有调度 token
            self.sched_step_seq += 1  # 调度步序号自增

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            # 性能记录上下文：调度后更新
            self._update_after_schedule(scheduler_output)  # 执行调度后更新
        return scheduler_output  # 返回调度器输出

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput  # 连接器与调度输出
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)  # 委托连接器构造元数据

    def _get_new_block_ids_to_zero(self) -> list[int] | None:  # 获取需清零的新块 id
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        # 每步排空新的注意力块 id，使管理器侧列表不无限增长；仅 KV 缓存清零消费它们
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()  # 取新块 id
        if not self.needs_kv_cache_zeroing:
            # 不需要 KV 缓存清零
            return None

        if self._skip_zero_block_ids:
            # 有跳过清零的块
            skip = self._skip_zero_block_ids  # 跳过集合
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]  # 过滤
            skip.clear()  # 清空跳过集合

        return new_block_ids_to_zero or None  # 返回需清零块 id（空则 None）

    def _preempt_request(
        self, request: Request, timestamp: float, drop_stale_output: bool = False  # 请求、时间戳、是否丢弃陈旧输出
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        # 抢占请求并将其放回等待队列。
        # 注意：请求应在本方法之外从运行队列弹出。
        # drop_stale_output：丢弃（而非交付）任何在途输出；reset_prefix_cache 使用，
        # 其同步恢复否则会乱序交付 token；也用于有待处理 KV 交接的连接器，
        # 抢占的块释放会使其失去有效 KV
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"  # 仅运行请求可被抢占
        )
        self._free_request_blocks(request)  # 释放请求的 KV 块
        self.encoder_cache_manager.free(request)  # 释放编码器缓存
        self._inflight_prefills.discard(request)  # 从在途 prefill 移除
        request.status = RequestStatus.PREEMPTED  # 设为已抢占
        request.num_computed_tokens = 0  # 已计算 token 清零
        if request.spec_token_ids:
            # 有投机 token
            request.spec_token_ids = []  # 清空
        # Async scheduling: mark all in-flight output as stale. Its tokens are
        # still delivered on return (dropping them would perturb spec-decode
        # acceptance) but must not mutate the reset counters; each step drains
        # its share in update_from_output. num_in_flight_tokens already
        # includes any undrained stale share, so assign rather than accumulate.
        # An undrained drop-mode share stays dropped: its positions have
        # already been resampled.
        # 异步调度：将所有在途输出标记为陈旧。其 token 仍在返回时交付
        # （丢弃会干扰投机解码接受），但不得变更重置计数器；每步在
        # update_from_output 中排空其份额。num_in_flight_tokens 已含未排空的陈旧份额，
        # 因此赋值而非累加。未排空的丢弃模式份额保持丢弃：其位置已被重采样
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0  # 合并丢弃标志
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens  # 陈旧输出 token 数 = 在途数
        request.num_output_placeholders = 0  # 输出占位符清零
        request.num_preemptions += 1  # 抢占计数自增
        if self.log_stats:
            # 记录抢占事件
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        # 将请求放回等待队列
        self.waiting.prepend_request(request)  # 前置加入等待队列
        self.reset_preempted_req_ids.add(request.request_id)  # 记录被抢占 id

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:  # 调度后更新
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        # 在请求被调度后推进其已计算 token 数。
        # 1. 当前步的 scheduler_output 须含原始调度 token 数以确定输入 ID。
        # 2. 此处推进已计算数，使我们能在下一调度步立即再次调度 prefill 请求。
        # 3. 若某些 token（如投机 token）稍后被拒，已计算数将在 update_from_output 调整
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens  # 调度 token 数
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            # 遍历已调度请求
            request = self.requests[req_id]  # 取请求
            request.num_computed_tokens += num_scheduled_token  # 推进已计算数
            request.num_in_flight_tokens += num_scheduled_token  # 推进在途 token 数
            if self.defer_block_free:
                # 延迟释放块时
                # Record the in-flight step, to fence deferred block freeing.
                # 记录在途步，作为延迟块释放的围栏
                request.last_sched_seq = self.sched_step_seq  # 记录最后调度序号
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders  # 是否仍在 prefill 分块
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk  # 有结构化输出请求
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            # 一旦不再 prefill 就从在途 prefill 集合移除
            if not request.is_prefill_chunk:
                # 已完成 prefill
                self._inflight_prefills.discard(request)  # 移除

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        # 在前向开始前为路由专家快照块 ID。并发的 schedule() 可能在 update_from_output
        # 运行前抢占请求并释放块；快照能在其后存活。用 update() 保留上一步
        # 尚未被 update_from_output 消费的条目（异步调度可能在前一个
        # update_from_output 运行前再次调用 _update_after_schedule）
        if self.enable_return_routed_experts:
            # 启用返回路由专家
            gid = self.routed_experts_mgr.attn_gid  # 注意力组 id
            self._re_block_ids.update(
                # 更新块 id 快照
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]  # 取该组块 id
                    for rid in num_scheduled_tokens  # 遍历已调度请求
                }
            )

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        # 清空已完成和被抢占的请求 id。
        # 注意：不能只 clear()，因为也会影响调度器输出
        self.finished_req_ids = set()  # 重新赋值为空集合
        self.reset_preempted_req_ids = set()  # 重新赋值为空集合

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate  # 会话请求与流式更新
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """
        # 用下一个流式更新更新等待会话。
        # 丢弃上一个输入分块的最后采样输出 token

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        # 当前流式输入行为：仅保留已计算的输出 token（丢弃最后采样的输出 token）
        num_computed_tokens = session.num_computed_tokens  # 已计算 token 数
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens  # 保留的输出 token 切片
        ]
        del session._all_token_ids[num_computed_tokens:]  # 删除已计算之后的 token
        session._output_token_ids.clear()  # 清空输出 token id
        assert session.prompt_token_ids is not None  # 断言提示词非 None
        # Extend prompt with kept output tokens.
        # 用保留的输出 token 扩展提示词
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            # 更新含多模态特征
            base = session.num_tokens  # 当前 token 总数作为偏移基准
            for mm_feature in update.mm_features:
                # 遍历多模态特征
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base  # 偏移加基准
                )
            session.mm_features.extend(update.mm_features)  # 扩展多模态特征

        session._all_token_ids.extend(update.prompt_token_ids or ())  # 扩展全部 token id
        session.prompt_token_ids.extend(update.prompt_token_ids or ())  # 扩展提示词 token id
        # Update block hashes for the new tokens.
        # 为新 token 更新块哈希
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)  # 更新提示词 token 数
        session.arrival_time = update.arrival_time  # 更新到达时间
        session.sampling_params = update.sampling_params  # 更新采样参数
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            # 之前在等待流式输入
            self.num_waiting_for_streaming_input -= 1  # 计数减一
        session.status = RequestStatus.WAITING  # 设为等待状态

        if self.log_stats:
            # 记录入队事件
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],  # 运行请求
        resumed_reqs: list[Request],  # 恢复请求
        num_scheduled_tokens: dict[str, int],  # 调度 token 数
        spec_decode_tokens: dict[str, list[int]],  # 投机解码 token
        req_to_new_blocks: dict[str, KVCacheBlocks],  # 请求 → 新块
    ) -> CachedRequestData:
        req_ids: list[str] = []  # 请求 id 列表
        new_token_ids: list[list[int]] = []  # 新 token id 列表
        new_block_ids: list[tuple[list[int], ...] | None] = []  # 新块 id 列表
        all_token_ids: dict[str, list[int]] = {}  # 全部 token id 映射
        num_computed_tokens: list[int] = []  # 已计算 token 数列表
        num_output_tokens: list[int] = []  # 输出 token 数列表
        resumed_req_ids = set()  # 恢复请求 id 集合

        num_running_reqs = len(running_reqs)  # 运行请求数
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            # 串联运行与恢复请求并遍历
            req_id = req.request_id  # 请求 id
            req_ids.append(req_id)  # 加入 id 列表
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            # 在 PP+异步调度中，通过直接 GPU 广播路径消费 token id，可省略此负载
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # PP 且非异步调度
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                # 使用 PP 时调度器回传采样 token，因为首末 stage worker 间无直接通信。
                # 否则无需回传，模型运行器会缓存它们
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())  # 调度 token 数减投机 token 数
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens  # 取新 token 切片
                ]
                new_token_ids.append(token_ids)  # 加入新 token id
            if idx >= num_running_reqs:
                # 索引超出运行请求数，属恢复请求
                resumed_req_ids.add(req_id)  # 加入恢复 id 集合
            if not self.use_v2_model_runner:  # noqa: SIM102
                # 非 v2 运行器
                if req_id not in self.prev_step_scheduled_req_ids:
                    # 上一步未调度（新进入缓存批次）
                    all_token_ids[req_id] = req.all_token_ids.copy()  # 复制全部 token id
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)  # 新块 id（允许 None）
            )
            num_computed_tokens.append(req.num_computed_tokens)  # 已计算 token 数
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders  # 输出 token 数加占位符
            )

        return CachedRequestData(
            # 构造缓存请求数据
            req_ids=req_ids,  # 请求 id
            resumed_req_ids=resumed_req_ids,  # 恢复请求 id
            new_token_ids=new_token_ids,  # 新 token id
            all_token_ids=all_token_ids,  # 全部 token id
            new_block_ids=new_block_ids,  # 新块 id
            num_computed_tokens=num_computed_tokens,  # 已计算 token 数
            num_output_tokens=num_output_tokens,  # 输出 token 数
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,  # 请求
        num_computed_tokens: int,  # 已计算 token 数
        num_new_tokens: int,  # 新 token 数
        encoder_compute_budget: int,  # 编码器计算预算
        shift_computed_tokens: int = 0,  # 已计算 token 偏移（EAGLE）
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        # 确定当前步需调度哪些编码器输入，并相应更新 num_new_tokens 与编码器 token 预算。
        # 编码器输入在以下条件满足时被调度：
        # - 其输出 token 与本步计算的 token 范围重叠；
        # - 尚未计算并存入编码器缓存；
        # - 不存在于远程编码器缓存（经 ECConnector）；
        # - 有足够编码器 token 预算处理它；
        # - 编码器缓存有空间存储它。
        # 若因缓存或预算限制无法调度，方法调整 num_new_tokens，仅调度到
        # 不可调度编码器输入之前的解码 token。
        # 注意 num_computed_tokens 含本地缓存块和外部缓存块（经 KVConnector）
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            # 无新 token 或无编码器输入
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []  # 待调度编码器输入
        mm_features = request.mm_features  # 多模态特征
        assert mm_features is not None  # 断言非 None
        assert len(mm_features) > 0  # 断言非空
        external_load_encoder_input = []  # 外部加载编码器输入

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        # 调度器在请求级操作（每请求可能多个编码器输入），需在编码器输入级
        # 创建临时跟踪器进行核算
        mm_hashes_to_schedule = set()  # 待调度多模态哈希集合
        num_embeds_to_schedule = 0  # 待调度嵌入数

        lo, hi = get_mm_features_in_window(
            # 获取窗口 [start, end) 内的多模态特征范围
            mm_features,
            start=num_computed_tokens,  # 窗口起点
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,  # 窗口终点
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        # 编码器-解码器时所有输入位于 start_pos=0，因此 lo 恒为 0
        if self.is_encoder_decoder:
            # 编码器-解码器模型
            lo = 0

        for i in range(lo, hi):
            # 遍历窗口内的编码器输入
            mm_feature = mm_features[i]  # 多模态特征
            start_pos = mm_feature.mm_position.offset  # 起始位置
            num_encoder_tokens = mm_feature.mm_position.length  # 编码器 token 数
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()  # 编码器嵌入数
            item_identifier = mm_feature.identifier  # 项标识符

            if self.is_encoder_decoder and num_computed_tokens > 0:
                # 编码器-解码器且已有计算 token
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # 断言：编码器-解码器时编码器输入应在序列开头处理
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                # 编码器输入已计算。此处计算略有不同：不把编码器输出变成解码器处理
                # 并反映在 num_computed_tokens 中的 token。start_pos 反映需确保计算
                # 编码器输入的位置，应恒为 0 以确保在运行解码器前计算编码器输入。
                # 一旦计算了一些解码 token（num_computed_tokens > 0），即知已计算
                # 编码器输入，可在此跳过
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                # 编码器-解码器模型暂不使用编码器缓存
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    # 相同编码器输入已在本步调度
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    # 编码器输入已在之前步计算并缓存
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            # 若不允许编码器输入分块，不想部分调度多模态项。若调度范围只覆盖
            # 多模态输入的一部分，回滚到该多模态项之前
            if (
                self.scheduler_config.disable_chunked_mm_input  # 禁用分块多模态输入
                and num_computed_tokens < start_pos  # 已计算在起始位置前
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)  # 调度范围未覆盖完整项
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                # 回滚时考虑 EAGLE 偏移以避免编码器缓存未命中。
                # 确保即使有偏移调度范围也停在 start_pos 之前
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)  # 回滚新 token 数
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule  # 能否分配
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # 编码器缓存满或编码器预算耗尽
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                # 假设编码器输入 token 应一并处理，因编码器通常用双向注意力
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # 已计算在起始位置前
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    # 仅调度编码器输入之前的解码 token
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens  # 到起始位置的距离
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    # 由于前缀缓存，即使编码器输入不可用 num_computed_tokens 也大于
                    # start_pos。此时本步无法为该请求调度任何 token
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            # 计算当前调度的编码器占位 token 范围内要调度的嵌入数
            start_idx_rel = max(0, num_computed_tokens - start_pos)  # 相对起始索引
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos  # 相对结束索引
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel  # 范围内嵌入索引
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            # 当前编码器占位 token 范围内无嵌入，可跳过该编码器输入
            if curr_embeds_end - curr_embeds_start == 0:
                # 无嵌入
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier  # EC 连接器有该缓存项
            ):
                # 远程编码器缓存已有该项
                mm_hashes_to_schedule.add(item_identifier)  # 加入待调度哈希
                external_load_encoder_input.append(i)  # 加入外部加载列表
                num_embeds_to_schedule += num_encoder_embeds  # 累加嵌入数
                continue

            num_embeds_to_schedule += num_encoder_embeds  # 累加待调度嵌入数
            encoder_compute_budget -= num_encoder_embeds  # 扣减编码器预算
            mm_hashes_to_schedule.add(item_identifier)  # 加入待调度哈希
            encoder_inputs_to_schedule.append(i)  # 加入待调度编码器输入

        return (
            encoder_inputs_to_schedule,  # 待调度编码器输入
            num_new_tokens,  # 调整后的新 token 数
            encoder_compute_budget,  # 更新后的编码器预算
            external_load_encoder_input,  # 外部加载编码器输入
        )

    def _make_scheduled_encoder_input_stats(
        self, scheduled_encoder_inputs: dict[str, list[int]]  # 已调度编码器输入
    ) -> ScheduledEncoderInputStats | None:
        stats = ScheduledEncoderInputStats()  # 创建统计对象

        for req_id, input_ids in scheduled_encoder_inputs.items():
            # 遍历已调度编码器输入
            request = self.requests.get(req_id)  # 取请求
            if request is None:
                # 请求不存在
                continue

            for input_id in input_ids:
                # 遍历各编码器输入
                mm_feature = request.mm_features[input_id]  # 多模态特征
                stats.num_inputs += 1  # 输入数自增
                stats.output_tokens += mm_feature.mm_position.get_num_embeds()  # 累加嵌入数

        return stats if stats.num_inputs else None  # 有输入则返回统计，否则 None

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput  # 调度输出
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        # 收集使用结构化输出的已调度请求 id 列表。
        # 位掩码的对应行将按此顺序排列
        if not scheduler_output.has_structured_output_requests:
            # 无结构化输出请求
            return None

        structured_output_request_ids = [
            req_id  # 请求 id
            for req_id in scheduler_output.num_scheduled_tokens  # 遍历已调度请求
            if (req := self.requests.get(req_id))  # 请求存在
            and (req.use_structured_output and not req.is_prefill_chunk)  # 用结构化输出且非 prefill
        ]
        if not structured_output_request_ids:
            # 无符合条件的请求
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            # 构造语法位掩码
            self.requests,  # 请求映射
            structured_output_request_ids,  # 结构化输出请求 id
            scheduler_output.scheduled_spec_decode_tokens,  # 投机解码 token
        )
        return GrammarOutput(structured_output_request_ids, bitmask)  # 返回语法输出

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,  # 调度输出
        model_runner_output: ModelRunnerOutput,  # 模型运行器输出
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids  # 采样 token id
        logprobs = model_runner_output.logprobs  # 对数概率
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict  # 提示词对数概率
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens  # 调度 token 数
        pooler_outputs = model_runner_output.pooler_output  # 池化输出
        num_nans_in_logits = model_runner_output.num_nans_in_logits  # logits 中 NaN 数
        kv_connector_output = model_runner_output.kv_connector_output  # KV 连接器输出
        cudagraph_stats = model_runner_output.cudagraph_stats  # CUDA 图统计

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        # 本步及更早步排队的所有 GPU 写已完成，可安全将延迟释放块归还块池
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            # 延迟释放且有调度 token
            self.processed_step_seq += 1  # 已处理步序号自增
            self._drain_deferred_frees()  # 排空延迟释放

        perf_stats: PerfStats | None = None  # 性能统计
        if self.perf_metrics and self.perf_metrics.is_enabled():
            # 性能指标启用
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)  # 获取每 GPU 步性能统计

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)  # 客户端 → 输出列表
        spec_decoding_stats: SpecDecodingStats | None = None  # 投机解码统计

        failed_kv_load_req_ids = None  # KV 加载失败的请求 id
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # 有 KV 连接器输出且有无效块
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            # 这些块含加载失败的外部计算 token。识别受影响请求并调整其计算
            # token 数以触发无效块重算
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,  # 无效块 id
                num_scheduled_tokens,  # 调度 token 数
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        # 将每步路由专家持久化到调度器侧槽缓冲（CPU->CPU 花式索引赋值；每步约几 MB）。
        # 必须先于下方的逐请求路由读取：停止的请求可能终止于本步生成的 token，
        # 其路由刚 D2H 到 model_runner_output
        routing_data = None  # 路由数据
        routing_offsets: dict[str, int] = {}  # 请求 → 路由偏移
        if model_runner_output.routed_experts is not None:
            # 有路由专家输出
            re = model_runner_output.routed_experts  # 路由专家对象
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)  # 存储批次
            routing_data = re.routing_data.astype(
                # 转换为槽缓冲 dtype
                self.routed_experts_mgr.routed_experts_by_slot.dtype,  # 目标 dtype
                copy=False,  # 不拷贝
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            # 用模型运行器的请求顺序（input_batch 顺序）构建偏移映射，
            # 而非调度器字典顺序
            offset = 0  # 偏移起点
            for rid in model_runner_output.req_ids:
                # 遍历模型运行器请求顺序
                routing_offsets[rid] = offset  # 记录偏移
                offset += num_scheduled_tokens[rid]  # 累加该请求调度 token 数

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        # 由于 len(num_scheduled_tokens) 可达 1K 以上，下方循环可能是性能瓶颈。
        # 应尽量避免循环内的昂贵操作
        stopped_running_reqs: set[Request] = set()  # 停止的运行请求集合
        stopped_preempted_reqs: set[Request] = set()  # 停止的被抢占请求集合
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            # 遍历已调度请求
            assert num_tokens_scheduled > 0  # 断言调度 token 数为正
            request = self.requests.get(req_id)  # 取请求
            output_is_stale = False  # 输出是否陈旧
            if request is not None:
                # 请求存在
                request.num_in_flight_tokens -= num_tokens_scheduled  # 扣减在途 token 数
                # Drain any stale share (see _preempt_request) in lockstep.
                # 同步排空任何陈旧份额（见 _preempt_request）
                if request.num_stale_output_tokens > 0:
                    # 有陈旧输出 token
                    output_is_stale = True  # 标记陈旧
                    request.num_stale_output_tokens -= num_tokens_scheduled  # 扣减陈旧份额
                    assert request.num_stale_output_tokens >= 0  # 断言非负
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                # 跳过 KV 加载失败或重新调度的请求
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # 请求已完成。可能发生在模型执行时被中止（如 PP 或异步调度）
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                # delay_free_blocks=True 时（KV 连接器异步 KV 缓存传输），中止请求
                # 不会置 None（为完成异步 KV 传输）。此时用 is_finished() 检查
                continue

            # Drop-mode stale output (same-step resume) is discarded entirely.
            # 丢弃模式的陈旧输出（同步恢复）被完全丢弃
            if output_is_stale and request.drop_stale_output:
                # 陈旧且丢弃
                continue

            req_index = model_runner_output.req_id_to_index[req_id]  # 请求在输出中的索引
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []  # 生成的 token id
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)  # 调度的投机 token id
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0  # 有生成或扩散模型
            ):
                # 有投机 token 且（有生成或扩散）
                num_draft_tokens = len(scheduled_spec_token_ids)  # 草稿 token 数
                num_sampled = self.num_sampled_tokens_per_step  # 采样数
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)  # 接受数
                num_rejected = num_draft_tokens - num_accepted  # 拒绝数
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                # 拒绝会回滚 num_computed_tokens（异步调度下还有覆盖投机 token 的
                # num_output_placeholders）。陈旧的拒绝计数早于抢占回滚，不得应用
                if not output_is_stale:
                    # 非陈旧输出
                    if request.num_computed_tokens > 0:
                        # 有已计算 token
                        request.num_computed_tokens -= num_rejected  # 回滚拒绝数
                    if request.num_output_placeholders > 0:
                        # 有输出占位符
                        request.num_output_placeholders -= num_rejected  # 回滚占位符
                spec_decoding_stats = self.make_spec_decoding_stats(
                    # 累计投机解码统计
                    spec_decoding_stats,  # 现有统计
                    num_draft_tokens=num_draft_tokens,  # 草稿数
                    num_accepted_tokens=num_accepted,  # 接受数
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,  # 无效投机数
                    request_id=req_id,  # 请求 id
                )

            # Free encoder inputs only after the step has actually executed.
            # 仅在步实际执行后释放编码器输入
            if request.has_encoder_inputs:
                # 有编码器输入
                self._free_encoder_inputs(request)  # 释放编码器输入

            stopped = False  # 是否停止
            new_logprobs = None  # 新对数概率
            new_token_ids = generated_token_ids  # 新 token id
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None  # 池化输出
            kv_transfer_params = None  # KV 传输参数
            ec_transfer_params = None  # EC 传输参数
            prefill_stats = None  # prefill 统计
            status_before_stop = request.status  # 停止前状态
            num_output_tokens_before = len(request._output_token_ids)  # 更新前输出 token 数

            # Check for stop and update request status.
            # 检查停止并更新请求状态
            if new_token_ids:
                # 有新 token
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale  # 用输出更新请求
                )
            elif request.pooling_params and pooler_output is not None:
                # 池化请求且有池化输出
                # Pooling stops as soon as there is output.
                # 池化一旦有输出即停止
                request.status = RequestStatus.FINISHED_STOPPED  # 设为完成停止
                stopped = True
            elif (
                self.is_encoder_only  # 纯编码器模型
                and request.num_computed_tokens >= request.num_prompt_tokens  # 提示词已消费完
            ):
                # An encoder instance runs the encoder and publishes the
                # embeddings instead of sampling, so it stops as soon as the
                # whole prompt is consumed. Encoder inputs are never scheduled
                # past a multi-modal item the encoder cache could not admit, so
                # a consumed prompt also means every item in it was encoded.
                # 编码器实例运行编码器并发布嵌入而非采样，因此整个提示词消费完即停止。
                # 编码器输入从不调度到编码器缓存无法接纳的多模态项之后，
                # 故消费完提示词也意味着其中每项都已编码
                request.status = RequestStatus.FINISHED_STOPPED  # 设为完成停止
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids  # 是否应推进结构化输出
            ):
                # 有新 token 且应推进
                struct_output_request = request.structured_output_request  # 结构化输出请求
                assert struct_output_request is not None  # 断言非 None
                grammar = struct_output_request.grammar  # 语法对象
                assert isinstance(grammar, StructuredOutputGrammar)  # 断言类型
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                # new_token_ids 可能是混合块：推理内容、推理结束标记、语法内容开头。
                # 修剪推理内容，使语法只看到语法内容
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids  # 修剪推理以推进
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids  # 语法不接受 token
                ):
                    # 语法拒绝 token
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR  # 设为错误完成
                    request.resumable = False  # 不可恢复
                    stopped = True

            routed_experts = None  # 路由专家
            if (
                self.enable_return_routed_experts  # 启用返回路由专家
                and routing_data is not None  # 有路由数据
                and new_token_ids  # 有新 token
            ):
                req_offset = routing_offsets[req_id]  # 请求路由偏移
                end = req_offset + num_tokens_scheduled  # 路由终点
                block_ids = self._re_block_ids.pop(req_id, [])  # 弹出块 id 快照
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    # prefill 完成：用调度时获取的块 ID 快照从槽缓冲读取完整提示词
                    # 路由（免疫异步抢占）
                    if (
                        request.sampling_params is not None  # 有采样参数
                        and request.sampling_params.routed_experts_prompt_start
                        is not None  # 有路由专家提示词起点
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start  # 用指定起点
                        )
                        assert prompt_start < request.num_prompt_tokens  # 断言有效
                    else:
                        prompt_start = 0  # 默认起点 0
                    routed_experts = self.routed_experts_mgr.get(
                        # 获取路由专家
                        block_ids,  # 块 id
                        request.num_prompt_tokens,  # 提示词 token 数
                        token_start=prompt_start,  # token 起点
                    )
                else:
                    # 已有输出 token（decode）
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        # 投机解码：接受的 token 在调度范围开头，拒绝的在末尾
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)  # 取开头部分
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        # 普通解码/重新 prefill：token 在末尾
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            should_emit_output = bool(
                new_token_ids or pooler_output is not None or stopped  # 是否发出输出
            )
            if should_emit_output:
                # 应发出输出
                prefill_stats = request.take_prefill_stats()  # 取 prefill 统计
                if prefill_stats is not None:
                    # 有 prefill 统计
                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)  # 最终化：估计缓存 token
                    )

            finish_reason = None  # 完成原因
            if stopped:
                # 已停止
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                # 在 _handle_stopped_request 前捕获 finish_reason，后者可能将
                # 继续的流式请求状态重置为 WAITING
                finish_reason = request.get_finished_reason()  # 获取完成原因
                finished = self._handle_stopped_request(request)  # 处理停止请求
                if finished:
                    # 真正完成
                    kv_transfer_params, ec_transfer_params = self._free_request(request)  # 释放请求

                if status_before_stop == RequestStatus.RUNNING:
                    # 停止前是运行状态
                    stopped_running_reqs.add(request)  # 加入停止运行集合
                else:
                    stopped_preempted_reqs.add(request)  # 加入停止被抢占集合

            # Extract sample logprobs if needed.
            # 需要时提取采样对数概率
            if (
                request.sampling_params is not None  # 有采样参数
                and request.sampling_params.num_logprobs is not None  # 请求对数概率
                and logprobs  # 有对数概率
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))  # 切片请求

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                # 有 NaN 统计且含该请求
                request.num_nans_in_logits = num_nans_in_logits[req_id]  # 记录 NaN 数

            # Get prompt logprobs for this request.
            # 获取该请求的提示词对数概率
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                # 为该请求添加 EngineCoreOutput
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,  # 请求 id
                        new_token_ids=new_token_ids,  # 新 token id
                        finish_reason=finish_reason,  # 完成原因
                        new_logprobs=new_logprobs,  # 新对数概率
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,  # 提示词对数概率
                        pooling_output=pooler_output,  # 池化输出
                        stop_reason=request.stop_reason,  # 停止原因
                        events=request.take_events(),  # 事件
                        prefill_stats=prefill_stats,  # prefill 统计
                        kv_transfer_params=kv_transfer_params,  # KV 传输参数
                        ec_transfer_params=ec_transfer_params,  # EC 传输参数
                        trace_headers=request.trace_headers,  # 跟踪头
                        routed_experts=routed_experts,  # 路由专家
                        num_nans_in_logits=request.num_nans_in_logits,  # NaN 数
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                # 不变量：EngineCore 不返回部分 prefill 输出
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        # 从运行和等待队列移除停止的请求
        if stopped_running_reqs:
            # 有停止的运行请求
            self.running = remove_all(self.running, stopped_running_reqs)  # 从运行队列移除
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            # 罕见情况，不太可能影响性能
            self.waiting.remove_requests(stopped_preempted_reqs)  # 从等待队列移除
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)  # 从跳过队列移除

        error_req_ids = set(self.grammar_compile_error_reqs)  # 语法编译错误请求 id
        self.grammar_compile_error_reqs.clear()  # 清空
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            # KV 加载失败且不重算（fail 策略）
            error_req_ids.update(failed_kv_load_req_ids)  # 加入错误请求

        if error_req_ids:
            # 有错误请求
            error_reqs = self.finish_requests(
                error_req_ids, RequestStatus.FINISHED_ERROR  # 以错误完成
            )
            for request in error_reqs:
                # 遍历错误请求发出输出
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,  # 请求 id
                        new_token_ids=[],  # 无新 token
                        finish_reason=request.get_finished_reason(),  # 完成原因
                        events=request.take_events(),  # 事件
                        trace_headers=request.trace_headers,  # 跟踪头
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        # KV 连接器：更新已完成 KV 传输的状态
        if kv_connector_output:
            # 有 KV 连接器输出
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        # 从模型运行器输出获取 worker 侧 KV 连接器统计
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            # 连接器更新后收集的调度器侧 KV 连接器统计
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if (
                scheduler_kv_connector_stats is not None  # 有调度器侧统计
                and not scheduler_kv_connector_stats.is_empty()  # 且非空
            ):
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)  # 聚合
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats  # 否则直接用调度器侧
                )

        # collect KV cache events from KV cache manager
        # 从 KV 缓存管理器收集 KV 缓存事件
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        # 从连接器收集 KV 缓存事件
        if self.connector is not None:
            # 有连接器
            connector_events = self.connector.take_events()  # 取连接器事件
            if connector_events:
                # 有事件
                if events is None:
                    # 尚无事件
                    events = list(connector_events)  # 用连接器事件
                else:
                    events.extend(connector_events)  # 扩展

        # publish collected KV cache events
        # 发布收集的 KV 缓存事件
        if events:
            # 有事件
            batch = KVEventBatch(ts=time.time(), events=events)  # 构造事件批次
            self.kv_event_publisher.publish(batch)  # 发布

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        # 为本步有请求输出的所有客户端创建 EngineCoreOutputs
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)  # 客户端 → 输出
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict  # 已完成请求 id 字典
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            # 包含自上次发送输出以来完成的请求 id
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                # 为该客户端的 EngineCoreOutputs 设置已完成请求集合
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    # 已有该客户端输出
                    eco.finished_requests = finished_set  # 设置已完成集合
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set  # 新建仅含已完成集合
                    )
            finished_req_ids.clear()  # 清空已完成字典

        if (
            stats := self.make_stats(
                # 构造统计
                spec_decoding_stats,  # 投机解码统计
                kv_connector_stats,  # KV 连接器统计
                cudagraph_stats,  # CUDA 图统计
                perf_stats,  # 性能统计
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            # 仅向其中一个前端返回统计
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                # 即使本步无请求输出也必须返回统计
                engine_core_outputs[0] = eco = EngineCoreOutputs()  # 创建空输出容器
            eco.scheduler_stats = stats  # 设置调度器统计

        return engine_core_outputs  # 返回引擎核心输出

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:  # 是否阻塞等待状态
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,  # 等待结构化输出语法
            RequestStatus.WAITING_FOR_REMOTE_KVS,  # 等待远程 KV
            RequestStatus.WAITING_FOR_STREAMING_REQ,  # 等待流式请求
        )

    def _enqueue_waiting_request(self, request: Request) -> None:  # 入队等待请求
        if self._is_blocked_waiting_status(request.status):
            # 阻塞等待状态入跳过队列
            self.skipped_waiting.add_request(request)
        else:
            # 普通等待状态入等待队列
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:  # 选择调度用等待队列
        if self.policy == SchedulingPolicy.FCFS:
            # FCFS：优先跳过队列，其次等待队列
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        # 优先级模式：两队列都非空时比较队首
        if self.waiting and self.skipped_waiting:
            # 两队列都非空
            waiting_req = self.waiting.peek_request()  # 等待队列队首
            skipped_req = self.skipped_waiting.peek_request()  # 跳过队列队首
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting  # 返回优先级高者

        return self.waiting or self.skipped_waiting or None  # 返回非空队列

    def _handle_stopped_request(self, request: Request) -> bool:  # 处理停止请求
        """Return True if finished (can be False for resumable requests)."""
        # 若完成返回 True（可恢复请求可能为 False）
        if not request.resumable:
            # 不可恢复
            return True

        if request.streaming_queue:
            # 有流式队列
            update = request.streaming_queue.popleft()  # 弹出下一更新
            if update is None:
                # Streaming request finished.
                # 流式请求完成
                return True
            self._update_request_as_session(request, update)  # 用更新更新会话
        else:
            # 无流式更新
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ  # 设为等待流式输入
            self.num_waiting_for_streaming_input += 1  # 计数自增

        self._enqueue_waiting_request(request)  # 重新入队
        return False  # 未完成

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False  # 请求、新 token、是否陈旧
    ) -> tuple[list[int], bool]:
        # is_stale is only used by the AsyncScheduler override.
        # is_stale 仅由 AsyncScheduler 重写使用
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        # 追加生成的 token 并检查停止。注意若请求仍在 prefill，
        # 期望模型运行器为该请求返回空 token id
        stopped = False  # 是否停止
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            # 逐个追加新 token
            request.append_output_token_ids(output_token_id)  # 追加输出 token

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            # 检查停止并更新请求状态。必须在创建 EngineCoreOutput 前调用
            stopped = check_stop(request, self.max_model_len)  # 检查停止
            if stopped:
                # 已停止
                del new_token_ids[num_new:]  # Trim new tokens if needed. 按需修剪新 token
                break
        return new_token_ids, stopped  # 返回新 token 与是否停止

    def _free_encoder_inputs(self, request: Request) -> None:  # 释放编码器输入
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request  # 获取已缓存的编码器输入 id
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        # 优化：集合为空时避免 list(set)
        if not cached_encoder_input_ids:
            # 无缓存输入
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's +1 read has also passed it, mirroring
        # the shift the encoder scheduling path applies.
        # 按草稿器前瞻延迟释放，使条目保持被引用直到草稿器的 +1 读取也越过它，
        # 镜像编码器调度路径应用的偏移
        spec_lookahead = 1 if self.use_eagle else 0  # 投机前瞻（EAGLE 为 1）

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        # 用 list(set) 避免遍历时修改集合
        for input_id in list(cached_encoder_input_ids):
            # 遍历已缓存输入
            mm_feature = request.mm_features[input_id]  # 多模态特征
            start_pos = mm_feature.mm_position.offset  # 起始位置
            num_tokens = mm_feature.mm_position.length  # token 数
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                # Whisper 一旦生成单个 token 即知编码器输入完成。
                # 交叉注意力 KV 已计算并缓存
                self.encoder_cache_manager.free_encoder_input(request, input_id)  # 释放
            elif (
                start_pos + num_tokens + spec_lookahead  # 项终点加前瞻
                <= request.num_computed_tokens - request.num_output_placeholders  # 已处理
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                # 已处理、已存入解码器 KV 缓存，且足够远越过占位符范围（加草稿器前瞻），
                # 使拒绝或草稿器收集都无法引用它
                self.encoder_cache_manager.free_encoder_input(request, input_id)  # 释放

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:  # 更新草稿 token id
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,  # 请求 id
            draft_token_ids.draft_token_ids,  # 草稿 token id
        ):
            request = self.requests.get(req_id)  # 取请求
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                # 请求可能已完成，跳过
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                # 忽略 prefill 分块的草稿 token
                if request.spec_token_ids:
                    # 有投机 token
                    request.spec_token_ids = []  # 清空
                continue

            # Add newly generated spec token ids to the request.
            # 将新生成的投机 token id 加入请求
            if self.structured_output_manager.should_advance(request):
                # 应推进结构化输出
                metadata = request.structured_output_request  # 结构化输出元数据
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]  # 语法校验
            request.spec_token_ids = spec_token_ids  # 设置投机 token

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput  # 草稿 token 与调度输出
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}  # 无效投机 token 数

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens  # 调度的投机 token
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,  # 请求 id
            draft_token_ids.draft_token_ids,  # 草稿 token id
        ):
            request = self.requests.get(req_id)  # 取请求
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                # 请求可能已完成，跳过
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)  # 占位投机 token
            if not placeholder_spec_tokens:
                # 无占位
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)  # 原始投机 token 数
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            # 将草稿修剪到调度的投机 token 数（如分块 prefill 场景需要）
            del spec_token_ids[orig_num_spec_tokens:]  # 删除多余草稿
            # Filter out spec tokens which do not adhere to the grammar.
            # 过滤不符合语法的投机 token
            if self.structured_output_manager.should_advance(request):
                # 应推进结构化输出
                metadata = request.structured_output_request  # 元数据
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]  # 校验
            # Pad to original number of spec tokens.
            # 填充回原始投机 token 数
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)  # 无效 token 数
            if num_invalid_tokens:
                # 有无效 token
                spec_token_ids.extend([-1] * num_invalid_tokens)  # 用 -1 填充
                num_invalid_spec_tokens[req_id] = num_invalid_tokens  # 记录无效数

            sched_spec_tokens[req_id] = spec_token_ids  # 写回

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens  # 设置无效投机数

    def get_request_counts(self) -> tuple[int, int]:  # 获取请求计数
        """Returns (num_running_reqs, num_waiting_reqs)."""
        # 返回（运行请求数、等待请求数）
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def get_kv_cache_usage(self) -> float:  # 获取 KV 缓存使用率
        """Returns the fraction of the KV cache currently in use (0.0-1.0)."""
        # 返回当前使用的 KV 缓存比例（0.0-1.0）
        return self.kv_cache_manager.usage

    def add_request(self, request: Request) -> None:  # 添加请求
        existing = self.requests.get(request.request_id)  # 查询已有请求
        if existing is not None:
            # 已有同 id 请求（流式输入续传）
            update = StreamingUpdate.from_request(request)  # 从请求构造流式更新
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                # 不在等待流式输入状态
                assert existing.streaming_queue is not None, "duplicate request id"  # 断言有流式队列
                # Queue next input chunk (or finished sentinel).
                # 排队下一输入分块（或完成哨兵）
                existing.streaming_queue.append(update)
            elif update is not None:
                # 在等待流式输入且有更新
                # Commence next input chunk.
                # 开始下一输入分块
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                # 流式输入会话完成
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)  # 中止完成
        else:
            # 新请求
            if request.resumable:
                # 可恢复请求
                request.streaming_queue = deque()  # 创建流式队列
            self._enqueue_waiting_request(request)  # 入队等待
            self.requests[request.request_id] = request  # 加入请求映射
            if self.connector is not None:
                # 有连接器
                self.connector.on_new_request(request)  # 通知连接器新请求
            if self.log_stats:
                # 记录入队事件
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus  # 请求 id 与完成状态
    ) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        # 处理来自调度器外部的完成信号。例如 API 服务器可在客户端断开时中止请求。
        # 若 request_ids 为 None，完成所有请求。
        # 返回被中止的请求列表，不包含已完成的
        assert RequestStatus.is_finished(finished_status)  # 断言是完成状态
        if isinstance(request_ids, str):
            # 单个字符串
            request_ids = (request_ids,)  # 转为元组
        elif request_ids is not None:
            # 可迭代
            request_ids = set(request_ids)  # 转为集合
        else:
            # None：所有请求
            request_ids = self.requests.keys()

        running_requests_to_remove = set()  # 待从运行队列移除的请求
        waiting_requests_to_remove = []  # 待从等待队列移除的请求
        valid_requests = []  # 有效请求列表

        # First pass: collect requests to remove from queues
        # 第一遍：收集要从队列移除的请求
        for req_id in request_ids:
            # 遍历请求 id
            request = self.requests.get(req_id)  # 取请求
            if request is None or request.is_finished():
                # Invalid request ID.
                # 无效请求 id
                continue

            valid_requests.append(request)  # 加入有效列表
            if request.status == RequestStatus.RUNNING:
                # 运行状态
                running_requests_to_remove.add(request)  # 加入运行移除集合
            else:
                # 非运行状态
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    # 等待流式输入
                    self.num_waiting_for_streaming_input -= 1  # 计数减一
                waiting_requests_to_remove.append(request)  # 加入等待移除列表

        # Remove all requests from queues at once for better efficiency
        # 一次性从队列移除所有请求以提高效率
        if running_requests_to_remove:
            # 有运行请求需移除
            self.running = remove_all(self.running, running_requests_to_remove)  # 批量移除
        if waiting_requests_to_remove:
            # 有等待请求需移除
            self.waiting.remove_requests(waiting_requests_to_remove)  # 从等待队列移除
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)  # 从跳过队列移除

        # Second pass: set status and free requests
        # 第二遍：设置状态并释放请求
        for request in valid_requests:
            # 遍历有效请求
            delay_free_blocks = False  # 是否延迟释放块
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # 等待远程 KV
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids  # 未完成接收则延迟
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)  # 移除完成接收记录
                self.failed_recving_kv_req_ids.discard(request.request_id)  # 移除失败接收记录

            request.status = finished_status  # 设置完成状态
            self._free_request(request, delay_free_blocks=delay_free_blocks)  # 释放请求

        return valid_requests  # 返回有效请求

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False  # 请求、是否延迟释放块
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()  # 断言已完成

        self._inflight_prefills.discard(request)  # 从在途 prefill 移除
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)  # 连接器完成处理

        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        # EC 连接器：镜像 KV 钩子。约定要求在编码器缓存释放前触发，使连接器能
        # 检查每请求状态（如 save_caches 期间记录的 mm_hashes）并为响应体
        # 发出 ec_transfer_params
        ec_xfer_params: dict[str, Any] | None = None  # EC 传输参数
        if self.ec_connector is not None:
            # 有 EC 连接器
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)  # EC 请求完成
            connector_delay_free_blocks |= ec_delay_free  # 合并延迟释放标志

        self.encoder_cache_manager.free(request)  # 释放编码器缓存
        request_id = request.request_id  # 请求 id
        self.finished_req_ids.add(request_id)  # 加入已完成 id 集合
        if self.finished_req_ids_dict is not None:
            # 有按客户端的已完成字典
            self.finished_req_ids_dict[request.client_index].add(request_id)  # 加入对应客户端

        delay_free_blocks |= connector_delay_free_blocks  # 合并延迟释放标志
        if not delay_free_blocks:
            # 不延迟释放
            self._free_blocks(request)  # 立即释放块

        return kv_xfer_params, ec_xfer_params  # 返回 KV/EC 传输参数

    def _free_blocks(self, request: Request):  # 释放块
        assert request.is_finished()  # 断言已完成
        self._free_request_blocks(request)  # 释放请求 KV 块
        del self.requests[request.request_id]  # 从请求映射删除

    @property
    def pause_state(self) -> PauseState:  # 暂停状态属性
        return self._pause_state  # 返回暂停状态

    def set_pause_state(self, pause_state: PauseState) -> None:  # 设置暂停状态
        self._pause_state = pause_state  # 赋值

    def _free_request_blocks(self, request: Request):  # 释放请求 KV 块
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        # 释放请求的 KV 块；当在途 GPU 步可能仍在写它们时延迟归还块池
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            # 最后调度步已处理：无在途写残留（正常完成总是如此），立即释放
            request.last_sched_seq <= self.processed_step_seq  # 最后调度序号已处理
        ):
            self.kv_cache_manager.free(request)  # 直接释放
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)  # 弹出待释放块
        if blocks:
            # 有块
            self.deferred_frees.append((self.sched_step_seq, blocks))  # 加入延迟释放队列

    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int  # 块与围栏序号
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        # 释放 CoW 拷贝保留；当运行拷贝的步可能仍在途时延迟归还块池
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            # 不延迟或围栏已处理
            self.kv_cache_manager.block_pool.free_blocks(blocks)  # 直接释放
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))  # 反转后加入延迟释放

    def _drain_deferred_frees(self):  # 排空延迟释放
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        # 归还围栏步已完成的延迟块。围栏按近单调顺序追加（CoW 保留围栏
        # 可能领先请求释放围栏一步），因此在第一个待处理处停止；
        # 其后任何已满足条目只是稍后释放
        while self.deferred_frees:
            # 有待处理延迟释放
            fence, _ = self.deferred_frees[0]  # 查看队首围栏
            if fence > self.processed_step_seq:
                # 围栏尚未处理
                break
            _, blocks = self.deferred_frees.popleft()  # 弹出
            # Free in reverse order so that the tail blocks are evicted first.
            # 反序释放使尾部块先被驱逐
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:  # 获取未完成请求数
        if self._pause_state == PauseState.PAUSED_ALL:
            # 完全暂停
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            # 暂停新请求
            return len(self.running)  # 仅运行请求
        num_waiting = (
            len(self.waiting)  # 等待队列
            + len(self.skipped_waiting)  # 跳过队列
            - self.num_waiting_for_streaming_input  # 减等待流式输入
        )
        return num_waiting + len(self.running)  # 等待加运行

    def has_finished_requests(self) -> bool:  # 是否有已完成请求
        if self.finished_req_ids:
            # 有已完成 id
            return True
        if self.connector is None:
            # 无连接器
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        # 等待延迟连接器清理的已完成请求在从调度队列移除后仍留在 self.requests
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)  # 队列中请求总数
        )
        return len(self.requests) > num_in_queues  # 请求映射超出队列数则有已完成请求

    def has_requests(self) -> bool:  # 是否有请求
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        # 重写接口默认值，使连接器仍有待处理推送工作时引擎保持存活
        # （如所有"活跃"请求完成后仍在途的推送模式 WRITE 传输）。
        # 无此钩子引擎会在连接器排空完成前静默。TODO：用更通用机制替代
        return (
            self.has_unfinished_requests()  # 有未完成请求
            or self.has_finished_requests()  # 或有已完成请求
            or (self.connector is not None and self.connector.has_pending_push_work())  # 或连接器有待推送工作
            or (
                self.ec_connector is not None  # 或 EC 连接器有待推送工作
                and self.ec_connector.has_pending_push_work()
            )
        )

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False  # 是否重置运行请求、连接器
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        # 重置 KV 前缀缓存。若 reset_running_requests 为 True，所有运行请求
        # 将被抢占并移入等待队列。否则仅在无运行请求占用 KV 缓存时重置
        if reset_running_requests:
            # 重置运行请求
            # For logging.
            # 用于日志
            timestamp = time.monotonic()  # 时间戳
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            # 通过将当前所有运行请求推入等待队列来使其 KV 失效。这样可将所有
            # KV 块引用计数降为 0，确保重置成功。反序抢占使请求按 FIFO 顺序加回运行队列
            while self.running:
                # 运行队列非空
                request = self.running.pop()  # 弹出末尾请求
                self._preempt_request(request, timestamp, drop_stale_output=True)  # 抢占并丢弃陈旧输出

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            # 清空已调度请求 id 缓存。由于强制同步抢占+恢复，必须表现得像这些请求
            # 上一步未被调度。它们将从模型运行器的持久批次中刷新
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()  # 重置前缀缓存
        if reset_running_requests and not reset_successful:
            # 重置运行请求但失败
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )
            # 抛出运行时错误：即使抢占所有运行请求仍无法重置 KV 缓存，
            # 可能因存在等待远程 KV 传输的运行请求（尚不支持）

        if reset_connector:
            # 重置连接器
            reset_successful = self.reset_connector_cache() and reset_successful  # 合并结果

        return reset_successful  # 返回是否成功

    def reset_connector_cache(self) -> bool:  # 重置连接器缓存
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            # 无连接器 -> 无需重置，视为成功。使无条件请求连接器重置的调用方
            # （如权重更新后缓存清除级联的一部分）不会仅因未配置连接器
            # 就看到 reset_prefix_cache() 变为 False
            logger.debug(
                "reset_connector requested but no KV connector is configured; "
                "treating as no-op success."
            )
            return True

        if self.connector.reset_cache() is False:
            # 连接器重置失败
            return False

        if self.log_stats:
            # 记录统计
            assert self.connector_prefix_cache_stats is not None  # 断言有统计
            self.connector_prefix_cache_stats.reset = True  # 标记重置

        return True  # 成功

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        # 重置编码器缓存以使所有缓存的编码器输出失效。
        # 模型权重更新时应调用，确保陈旧视觉嵌入不被复用
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,  # 投机解码统计
        kv_connector_stats: KVConnectorStats | None = None,  # KV 连接器统计
        cudagraph_stats: CUDAGraphStat | None = None,  # CUDA 图统计
        perf_stats: PerfStats | None = None,  # 性能统计
    ) -> SchedulerStats | None:
        if not self.log_stats:
            # 不记录统计
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()  # 前缀缓存统计
        assert prefix_cache_stats is not None  # 断言非 None
        connector_prefix_cache_stats: PrefixCacheStats | None = None  # 连接器前缀缓存统计
        if self.connector_prefix_cache_stats is not None:
            # 有连接器前缀缓存统计
            connector_prefix_cache_stats = self.connector_prefix_cache_stats  # 取用
            self.connector_prefix_cache_stats = PrefixCacheStats()  # 重置为新对象
        eviction_events = (
            self.kv_metrics_collector.drain_events()  # 排空驱逐事件
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats  # 投机解码统计
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None  # 连接器统计数据
        )
        return SchedulerStats(
            # 构造调度器统计
            num_running_reqs=len(self.running),  # 运行请求数
            num_waiting_reqs=len(self.waiting),  # 等待请求数
            num_skipped_waiting_reqs=len(self.skipped_waiting),  # 跳过等待请求数
            kv_cache_usage=self.kv_cache_manager.usage,  # KV 缓存使用率
            prefix_cache_stats=prefix_cache_stats,  # 前缀缓存统计
            connector_prefix_cache_stats=connector_prefix_cache_stats,  # 连接器前缀缓存统计
            kv_cache_eviction_events=eviction_events,  # KV 缓存驱逐事件
            spec_decoding_stats=spec_stats,  # 投机解码统计
            kv_connector_stats=connector_stats_payload,  # KV 连接器统计
            cudagraph_stats=cudagraph_stats,  # CUDA 图统计
            perf_stats=perf_stats,  # 性能统计
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,  # 现有统计
        num_draft_tokens: int,  # 草稿 token 数
        num_accepted_tokens: int,  # 接受 token 数
        num_invalid_spec_tokens: dict[str, int] | None,  # 无效投机 token 数
        request_id: str,  # 请求 id
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            # 不记录统计或无草稿 token
            return None
        if spec_decoding_stats is None:
            # 尚无统计
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)  # 新建
        if num_invalid_spec_tokens:
            # 有无效投机 token
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)  # 扣减无效数
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens  # 观察草稿
        )
        return spec_decoding_stats  # 返回统计

    def shutdown(self) -> None:  # 关闭调度器
        logger.debug_once("[shutdown] Scheduler: start")  # 调试日志
        if self.kv_event_publisher:
            # 有 KV 事件发布器
            self.kv_event_publisher.shutdown()  # 关闭
        if self.connector is not None:
            # 有 KV 连接器
            self.connector.shutdown()  # 关闭

        if self.ec_connector is not None:
            # 有 EC 连接器
            self.ec_connector.shutdown()  # 关闭

        logger.debug_once("[shutdown] Scheduler: complete")  # 调试日志

    ########################################################################
    # KV Connector Related Methods
    # KV 连接器相关方法
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:  # 获取 KV 连接器
        return self.connector  # 返回连接器

    def get_ec_connector(self) -> ECConnectorBase | None:  # 获取 EC 连接器
        return self.ec_connector  # 返回 EC 连接器

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:  # 获取 KV 事件发布器配置
        return self.kv_event_publisher.get_publisher_config()  # 返回发布器配置

    def _connector_finished(
        self, request: Request  # 请求
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        # 适用时调用 KV 连接器的 request_finished() 方法。
        # 返回可选的 KV 传输参数以并入请求输出
        if self.connector is None:
            # 无连接器
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector, on the processed-token basis (see `allocate_slots`).
        # 在将块表交给连接器前，基于已处理 token 数释放窗口外前缀块
        # （见 `allocate_slots`）
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,  # 请求 id
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens  # 已处理已计算 token
            ),
            num_prompt_tokens=request.num_prompt_tokens,  # prompt token 数
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,  # 请求 id
            num_computed_tokens=request.num_computed_tokens,  # 已计算 token 数
        )
        # 获取已计算 token 对应的块 id

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            # 注：强制所有连接器支持 HMA 后应弃用此路径。
            # 混合内存分配器在此路径应已关闭，但这里再确认
            assert len(self.kv_cache_config.kv_cache_groups) == 1  # 断言仅单 KV 缓存组
            return self.connector.request_finished(request, block_ids[0])  # 单组完成回调

        return self.connector.request_finished_all_groups(request, block_ids)  # 全组完成回调

    def _request_remaining_blocks(self, request: Request) -> int:  # 请求剩余所需块数
        """Blocks `request` still needs to allocate to hold its full sequence."""
        # 请求容纳完整序列仍需分配的块数
        full_num_tokens = min(request.num_tokens, self.max_model_len)  # 完整 token 数（限最大长度）
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,  # 请求 id
            num_tokens=full_num_tokens,  # token 数
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,  # 空块占位
            num_encoder_tokens=0,  # 编码器 token 数
            total_computed_tokens=request.num_computed_tokens,  # 总已计算 token
            num_local_computed_tokens=request.num_computed_tokens,  # 本地已计算 token
            num_tokens_main_model=full_num_tokens,  # 主模型 token 数
            apply_admission_cap=True,  # 应用准入上限
        )

    def _inflight_prefill_reserved_blocks(self) -> int:  # 在途 prefill 保留块数
        """Num blocks in-flight prefills still need to finish (their reservation)."""
        # 在途 prefill 完成所需的块数（其预留量）
        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills  # 累计各在途请求剩余块
        )

    def _update_waiting_for_remote_kv(self, request: Request) -> None:  # 更新等待远程 KV 的请求
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        # KV 连接器：异步接收完成后更新请求状态。
        # KV 传输就绪时缓存块，请求状态从 WAITING_FOR_REMOTE_KV 移回 WAITING
        assert self.connector is not None  # 断言有连接器

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            # 请求有 KV 加载失败；num_computed_tokens 已在
            # _update_requests_with_invalid_blocks 中更新
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                # 缓存任何有效的已计算 token
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)  # 缓存块
                if self.needs_kv_cache_zeroing:
                    # The failed load left the blocks beyond the valid
                    # prefix unwritten and their zeroing was skipped; zero
                    # them before they are recomputed locally.
                    # 失败加载使有效前缀之后的块未写入且跳过了清零；
                    # 在本地重算前先清零
                    self.kv_cache_manager.record_blocks_for_zeroing(
                        request.request_id, request.num_computed_tokens  # 记录待清零块
                    )
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                # (Freed blocks are re-recorded for zeroing when
                # reallocated, so the skipped blocks need no handling.)
                # 无有效已计算 token，释放已分配块。重试时可能本地缓存命中。
                # （释放的块重分配时会重新记录清零，跳过的块无需处理）
                self.kv_cache_manager.free(request)  # 释放

            self.failed_recving_kv_req_ids.remove(request.request_id)  # 移出失败接收集合
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            # 块已就绪，实际缓存它们。仅当缓存启用时才缓存
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)  # 缓存块

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            # 整个 prompt 全部命中时，需重算最后一个 token 才能采样下一 token
            if request.num_computed_tokens == request.num_tokens:
                # 已计算 token 等于总 token 数
                request.num_computed_tokens = request.num_tokens - 1  # 回退一个

        self.finished_recving_kv_req_ids.remove(request.request_id)  # 移出完成接收集合

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:  # 尝试提升被阻塞的等待请求
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        # 尝试将被阻塞的等待请求提升回可调度状态
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            # finished_recving_kv_req_ids 在 update_from_output() 中填充，
            # 依据 worker 侧连接器信号 KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                # 未完成接收
                return False
            self._update_waiting_for_remote_kv(request)  # 更新等待远程 KV 状态
            if request.num_preemptions:
                # 有抢占历史
                request.status = RequestStatus.PREEMPTED  # 设为被抢占
            else:
                # 无抢占历史
                request.status = RequestStatus.WAITING  # 设为等待
            return True  # 提升成功

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            # 等待结构化输出语法
            structured_output_req = request.structured_output_request  # 结构化输出请求
            if not structured_output_req or structured_output_req.grammar is None:
                # 无请求或语法未就绪
                return False
            if isinstance(structured_output_req.grammar, Exception):
                # 语法编译异常
                self.grammar_compile_error_reqs.add(request.request_id)  # 记入编译错误集合
                return False
            request.status = RequestStatus.WAITING  # 语法就绪，设为等待
            return True  # 提升成功

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            # 等待流式输入
            assert not request.streaming_queue  # 断言流式队列为空
            return False  # 无法提升

        raise AssertionError(
            # 抛出断言：提升时出现意外的阻塞等待状态
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):  # 依据 KV 传输完成更新
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """
        # KV 连接器：根据输出更新调度器状态。
        # worker 侧连接器将 finished_recving 与 finished_sending 请求加入输出。
        # finished_sending：释放块；finished_recving：加入状态以便下一步调度

        if self.connector is not None:
            # 有连接器
            self.connector.update_connector_output(kv_connector_output)  # 更新连接器输出

        # KV Connector:: update recv and send status from last step.
        # KV 连接器：更新上一步的接收与发送状态
        for req_id in kv_connector_output.finished_recving or ():
            # 遍历完成接收的请求
            logger.debug("Finished recving KV transfer for request %s", req_id)  # 调试日志
            assert req_id in self.requests  # 断言请求存在
            req = self.requests[req_id]  # 取请求
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # 等待远程 KV
                self.finished_recving_kv_req_ids.add(req_id)  # 加入完成接收集合
            else:
                # 其他状态（已完成）
                assert RequestStatus.is_finished(req.status)  # 断言已完成
                self._free_blocks(self.requests[req_id])  # 释放块
        for req_id in kv_connector_output.finished_sending or ():
            # 遍历完成发送的请求
            logger.debug("Finished sending KV transfer for request %s", req_id)  # 调试日志
            assert req_id in self.requests  # 断言请求存在
            self._free_blocks(self.requests[req_id])  # 释放块

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],  # 待扫描请求
        invalid_block_ids: set[int],  # 无效块 id 集合
        num_scheduled_tokens: dict[str, int],  # 各请求调度 token 数
        evict_blocks: bool = True,  # 是否收集驱逐块
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        # 识别并更新受无效 KV 缓存块影响的请求。
        # 扫描给定请求，检测含无效块者并将其 num_computed_tokens
        # 调整为最长有效前缀。为可观测性，还累计所有受影响请求需重算的总 token 数。
        # 参数：requests 待扫描请求；invalid_block_ids 无效块 id；
        # num_scheduled_tokens 请求 id -> 调度 token 数；
        # evict_blocks 是否收集驱逐块（未缓存的异步请求为 False）。
        # 返回：受影响请求 id 集合、需重算 token 总数、待驱逐块 id 集合
        affected_req_ids: set[str] = set()  # 受影响请求 id 集合
        total_affected_tokens = 0  # 受影响 token 总数
        blocks_to_evict: set[int] = set()  # 待驱逐块集合
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        # 无效块被批内多个请求共享时，这些请求都必须重调度，
        # 但只有第一个重算它。此集合追踪已标记重算的块
        marked_invalid_block_ids: set[int] = set()  # 已标记的无效块集合
        for request in requests:
            # 遍历请求
            is_affected = False  # 是否受影响
            marked_invalid_block = False  # 是否已标记过无效块
            req_id = request.request_id  # 请求 id
            # TODO (davidb): add support for hybrid memory allocator
            # TODO：支持混合内存分配器
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)  # 取请求块 id
            # We iterate only over blocks that may contain externally computed
            # tokens
            # 只遍历可能包含外部计算 token 的块
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)  # 扣除本步调度 token
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1  # 向上取整
            ) // self.block_size  # 已计算块数
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                # 遍历已计算块
                if block_id not in invalid_block_ids:
                    # 块有效
                    continue

                is_affected = True  # 受影响

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    # 该无效块与先前请求共享且已标记重算。
                    # 重调度时本请求仍可将其视为已计算。
                    # 目前仅适用于同步加载；异步加载尚不支持块共享
                    continue

                marked_invalid_block_ids.add(block_id)  # 标记该块已重算

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    # 本请求已标记过无效块并更新了 num_computed_tokens
                    continue

                marked_invalid_block = True  # 标记本请求已处理
                # Truncate the computed tokens at the first failed block
                # 在首个失败块处截断已计算 token
                request.num_computed_tokens = idx * self.block_size  # 截断到块边界
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens  # 受影响 token 数
                )
                total_affected_tokens += num_affected_tokens  # 累计

                # collect invalid block and all downstream dependent blocks
                # 收集无效块及所有下游依赖块
                if evict_blocks:
                    # 需驱逐
                    blocks_to_evict.update(req_block_ids[idx:])  # 加入从该块起的后续块

            if is_affected:
                # 受影响
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    # 本请求所有无效块均与先前请求共享并将由其重算。
                    # 回退为仅将已缓存 token 视为已计算。
                    # 目前仅适用于同步加载；异步加载尚不支持块共享
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens  # 补回差值
                    )
                    request.num_computed_tokens = req_num_computed_tokens  # 恢复已计算数

                affected_req_ids.add(request.request_id)  # 加入受影响集合

        return affected_req_ids, total_affected_tokens, blocks_to_evict  # 返回三元组

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]  # 无效块 id 与调度 token 数
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        # 处理受无效 KV 缓存块影响的请求。
        # 返回在 update_from_output 主循环中需跳过的受影响请求 id 集合
        should_fail = not self.recompute_kv_load_failures  # 是否应失败（不重算则失败）

        # handle async KV loads (not cached yet, evict_blocks=False)
        # 处理异步 KV 加载（尚未缓存，evict_blocks=False）
        async_load_reqs = (
            req  # 异步加载请求生成器
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS  # 等待远程 KV 的跳过队列请求
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,  # 异步加载请求
                invalid_block_ids,  # 无效块 id
                num_scheduled_tokens,  # 调度 token 数
                evict_blocks=False,  # 不驱逐
            )
        )

        total_failed_requests = len(async_failed_req_ids)  # 失败请求总数（异步部分）
        total_failed_tokens = num_failed_tokens  # 失败 token 总数（异步部分）

        # handle sync loads (may be cached, collect blocks for eviction)
        # 处理同步加载（可能已缓存，收集待驱逐块）
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True  # 运行队列、驱逐
            )
        )

        total_failed_requests += len(sync_failed_req_ids)  # 累加同步失败请求数
        total_failed_tokens += num_failed_tokens  # 累加同步失败 token 数

        if not total_failed_requests:
            # 无失败请求
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        # 仅在非重算策略时从缓存驱逐无效块及下游依赖块
        # （重算策略下块将被重算并被共享它们的其他请求复用）
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            # 有待驱逐块且非重算策略
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)  # 驱逐块

        if should_fail:
            # 应失败策略
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids  # 合并所有失败 id
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            # 错误日志：因 KV 加载失败而使 %d 个请求失败
            # （failure_policy=fail，%d 个 token 受影响）。请求 ID：%s
            return all_failed_req_ids  # 返回所有失败 id

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )
        # 警告日志：从 KV 加载失败恢复：%d 个请求被重调度（%d 个 token 受影响）

        # Mark async requests with KV load failures for retry once loading completes
        # 标记有 KV 加载失败的异步请求，待加载完成后重试
        self.failed_recving_kv_req_ids |= async_failed_req_ids  # 并入失败接收集合
        # Return sync affected IDs to skip in update_from_output
        # 返回同步受影响 id 以在 update_from_output 中跳过
        return sync_failed_req_ids
