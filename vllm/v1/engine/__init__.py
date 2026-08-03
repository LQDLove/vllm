# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import enum  # enum：枚举模块（IntEnum/Enum 基类）
import time  # time：时间模块（monotonic 单调时钟时间戳）
from collections.abc import Mapping  # Mapping：映射类型（trace_headers）
from dataclasses import dataclass  # dataclass：数据类装饰器
from typing import Any, Literal  # Any：通用类型；Literal：字面量类型标注

import msgspec  # msgspec：高性能 msgpack 序列化库（跨进程传输）
import numpy as np  # numpy：科学计算库（routed_experts 数据类型）
import torch  # torch：PyTorch（prompt_embeds 数据类型）

from vllm.lora.request import LoRARequest  # LoRA 请求类型
from vllm.multimodal.inputs import MultiModalFeatureSpec  # 多模态特征规格
from vllm.pooling_params import PoolingParams  # 池化参数（embedding 任务）
from vllm.sampling_params import SamplingParams  # 采样参数（生成任务）
from vllm.v1.metrics.stats import PrefillStats, SchedulerStats
# prefill 统计；调度器统计
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
# logprobs 列表（采样阶段）；logprobs 张量（prompt 阶段）
from vllm.v1.serial_utils import UtilityResult  # 工具调用结果

# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
# 暂停生成模式的参数类型。
# - "abort"：立即中止所有进行中请求（默认）。
# - "wait"：等待进行中请求完成后再暂停。
# - "keep"：冻结队列中的请求；恢复时继续（resume_generation()）。
PauseMode = Literal["abort", "wait", "keep"]
# 暂停模式字面量类型

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
# 这些是 RequestOutput.finish_reason 的可能值，构成外部 API 的一部分。
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")
# 完成原因字符串元组（与 FinishReason 枚举值一一对应）

EEP_NOTIFICATION_CALL_ID = -1
# 弹性 EP 通知的专用 call_id（-1 表示此输出是弹性扩展通知而非工具调用结果）

FT_STATUS_CALL_ID = -2
# 容错状态查询的专用 call_id（-2 表示此输出是引擎健康状态上报）


class EEPNotificationType(enum.Enum):
    # 弹性专家并行（Elastic Expert Parallelism）通知类型枚举
    NEW_CORE_ENGINES_INIT_READY = "NEW_CORE_ENGINES_INIT_READY"
    # 新核心引擎初始化就绪（扩容时新引擎通知现有引擎）
    NEW_CORE_ENGINES_WEIGHTS_INIT_READY = "NEW_CORE_ENGINES_WEIGHTS_INIT_READY"
    # 新核心引擎权重加载就绪（现有引擎可开始复制权重）
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    # 重配置完成（所有引擎已切换到新分布式配置）
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"
    # 被移除引擎关闭完成（可释放资源）


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """
    # 请求完成原因枚举。
    # 使用整数而非字符串，以便更紧凑的序列化。
    # stop：发出停止字符串
    # length：消耗了 max_tokens，或达到 max_model_len
    # abort：客户端中止
    # error：可重试的请求级内部错误（如 KV 加载失败）。
    #         不变式：始终转换为 500 内部服务器错误。
    # repetition：检测到重复 token 模式（幻觉）

    STOP = 0  # 停止字符串触发
    LENGTH = 1  # 达到长度上限
    ABORT = 2  # 客户端中止
    ERROR = 3  # 内部错误
    REPETITION = 4  # 重复模式检测

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]
        # 返回对应的字符串形式（如 STOP → "stop"）


@dataclass
class EngineCoreReadyResponse:
    """Sent from EngineCore to each frontend at the end of engine startup.

    Contains post-initialization config that may differ from the original
    values (e.g. max_model_len after KV cache auto-fitting).
    """
    # 引擎启动完成时从 EngineCore 发送给每个前端的就绪响应。
    # 包含初始化后可能与原始配置不同的值
    # （如 KV 缓存自动适配后 max_model_len）。

    max_model_len: int  # 最大模型长度（可能被 auto-fit 调整）
    num_gpu_blocks: int  # GPU KV 缓存块数
    block_size: int  # 块大小（可能被模型对齐逻辑放大）
    dp_stats_address: str | None  # DP 统计发布地址（用于负载均衡订阅）
    dtype: str  # 模型数据类型（如 "float16"）
    vllm_version: str  # vLLM 版本号
    world_size: int  # 全局进程数（PP×TP）
    data_parallel_size: int  # DP 并行数
    # KV cache capacity (None for encoder-only/attention-free models).
    # KV 缓存容量（编码器/无注意力模型为 None）。
    kv_cache_size_tokens: int | None = None  # KV 缓存容量（token 数）
    kv_cache_max_concurrency: float | None = None  # KV 缓存最大并发度


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    # 使用紧凑数组格式序列化（比字典格式省 ~40%）
    omit_defaults=True,  # type: ignore[call-arg]
    # 跳过默认值字段（减少传输数据量）
    gc=False,  # 禁用 GC 跟踪（高频创建的小对象，减少 GC 压力）
):  # type: ignore[call-arg]
    # 引擎核心请求：前端 → 核心引擎的跨进程请求载体
    request_id: str  # 内部唯一请求 ID（外部 ID + 随机后缀）
    prompt_token_ids: list[int] | None  # prompt token IDs
    mm_features: list[MultiModalFeatureSpec] | None  # 多模态特征列表
    sampling_params: SamplingParams | None  # 采样参数（生成任务）
    pooling_params: PoolingParams | None  # 池化参数（embedding 任务）
    arrival_time: float  # 请求到达时间戳
    lora_request: LoRARequest | None  # LoRA 适配器请求
    cache_salt: str | None  # 前缀缓存的安全隔离 salt
    data_parallel_rank: int | None  # 目标 DP rank（DP 路由用）
    prompt_embeds: torch.Tensor | None = None  # 预计算 embedding（替代 token）

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    # 混合模式输入的逐位置掩码（如带 prompt_embeds 内容部分的聊天补全）。
    # `True` 表示该位置是真实 token ID；`False` 表示该位置使用来自
    # `prompt_embeds` 的预计算条目。纯 token 和纯 embedding 请求为 `None`。
    prompt_is_token_ids: list[bool] | None = None  # 混合模式掩码

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    # 客户端索引，多 API 服务器扩展时确保输出回传同一客户端。
    client_index: int = 0  # 客户端索引

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    # DP 情况下指示请求预期属于哪个 wave，
    # 覆盖请求在收到 wave 完成通知前被发送的竞态条件。
    current_wave: int = 0  # 当前 wave 编号
    priority: int = 0  # 请求优先级（越大越优先调度）

    trace_headers: Mapping[str, str] | None = None  # OpenTelemetry 追踪上下文
    resumable: bool = False  # 是否可续传（流式输入）

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    # 用户提供的原始请求 ID。此字段在内部设置，
    # 从最初分配给 request_id 字段的用户请求 ID 复制而来，
    # 见 InputProcessor.assign_request_id()。
    # 用于输出和 abort(req_id, internal=False) 支持。
    external_req_id: str | None = None  # 外部请求 ID（用户可识别）

    reasoning_ended: bool | None = None  # 推理阶段是否已结束（推理模型）
    reasoning_parser_kwargs: dict[str, Any] | None = None  # 推理解析器参数

    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    # 如果为 True，请求将添加到调度器等待队列后立即中止，
    # 使连接器端清理通过标准 request_finished 钩子运行。
    # 用于在 D 节点拒绝 KV 传输请求（引擎准入前）时释放 P 侧 prefill 块。
    abort_immediately: bool = False  # 添加后立即中止标志

    session_id: str | None = None

    @property
    def params(self) -> SamplingParams | PoolingParams:
        """Return the processed params (sampling or pooling)."""
        # 返回当前请求的活动参数（采样或池化）
        if self.sampling_params is not None:
            return self.sampling_params  # 采样参数
        assert self.pooling_params is not None  # 断言池化参数存在
        return self.pooling_params  # 池化参数


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""
    # 引擎核心请求事件的类型枚举

    QUEUED = 1  # 请求进入调度队列
    SCHEDULED = 2  # 请求被调度执行
    PREEMPTED = 3  # 请求被抢占


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamp and is used by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """
    # 与请求关联的带时间戳的引擎核心事件。
    # 时间戳是单调时钟时间戳，由引擎前端用于计算事件间隔。
    # 这些时间戳不应与来自其他进程的时间戳比较。

    type: EngineCoreEventType  # 事件类型
    timestamp: float  # 时间戳（time.monotonic()）

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        # 创建新事件（时间戳默认为单调时钟）
        timestamp = time.monotonic() if timestamp is None else timestamp
        # 使用单调时钟（不受系统时间跳变影响）
        return cls(event_type, timestamp)  # 创建事件


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    # 紧凑数组格式
    omit_defaults=True,  # type: ignore[call-arg]
    # 跳过默认值
    gc=False,  # 禁用 GC 跟踪
):  # type: ignore[call-arg]
    # 引擎核心输出：核心引擎 → 前端的单请求输出载体
    request_id: str  # 请求 ID（输出路由用）
    new_token_ids: list[int]  # 新生成的 token ID 列表

    new_logprobs: LogprobsLists | None = None  # 新 token 的 logprobs（采样阶段）
    new_prompt_logprobs_tensors: LogprobsTensors | None = None
    # prompt 阶段的 logprobs 张量

    pooling_output: torch.Tensor | None = None  # 池化模型输出（embedding）

    finish_reason: FinishReason | None = None  # 完成原因（None = 未完成）
    stop_reason: int | str | None = None  # 停止原因（token ID 或字符串）
    events: list[EngineCoreEvent] | None = None  # 请求生命周期事件
    kv_transfer_params: dict[str, Any] | None = None  # KV 传输参数（P-D 分离）
    ec_transfer_params: dict[str, Any] | None = None  # 专家缓存传输参数（MoE）

    trace_headers: Mapping[str, str] | None = None  # 分布式追踪上下文

    prefill_stats: PrefillStats | None = None  # prefill 阶段统计

    routed_experts: np.ndarray | None = None  # MoE 路由的专家列表
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    # logits 中的 NaN 数量。大于 0 表示输出已损坏。
    num_nans_in_logits: int = 0  # NaN 计数（健康监控）

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None
        # 是否完成（finish_reason 非 None）


class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    # 紧凑数组格式
    gc=False,  # 禁用 GC 跟踪
):  # type: ignore[call-arg]
    # 工具输出：核心引擎对前端工具调用的异步响应
    call_id: int  # 调用 ID（与前端等待的 Future 对应）

    # Non-None implies the call failed, result should be None.
    # 非 None 表示调用失败，result 应为 None。
    failure_message: str | None = None  # 失败消息（失败时非 None）
    result: UtilityResult | None = None  # 工具调用结果（成功时非 None）


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    # 紧凑数组格式
    omit_defaults=True,  # type: ignore[call-arg]
    # 跳过默认值
    gc=False,  # 禁用 GC 跟踪
):  # type: ignore[call-arg]
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout
    # 注意：可考虑更紧凑的方式，如按列布局
    # 引擎核心输出容器：ZMQ 传输的最小单元

    engine_index: int = 0  # 产出引擎的索引（DP 多引擎场景）

    # [num_reqs]
    # 请求数量
    outputs: list[EngineCoreOutput] = []  # 推理输出列表（单请求输出）
    scheduler_stats: SchedulerStats | None = None  # 调度器统计
    timestamp: float = 0.0  # 输出时间戳（默认 monotonic）

    utility_output: UtilityOutput | None = None  # 工具调用输出（特殊）
    finished_requests: set[str] | None = None  # 已完成的请求 ID 集合

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    # DP 情况下，标记当前请求 wave 已完成且引擎已暂停。
    wave_complete: int | None = None  # wave 完成信号
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    # DP 情况下，标记收到"旧"wave 的请求，
    # 需要在其他引擎启动下一 wave。
    start_wave: int | None = None  # 启动 wave 信号

    def __post_init__(self):
        # 初始化后处理：设置默认时间戳
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()
            # 使用单调时钟（默认）


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    # 请求类型定义为十六进制字节字符串，
    # 可直接在 socket 上发送而无需单独编码步骤。

    ADD = b"\x00"  # 添加新推理请求
    ABORT = b"\x01"  # 中止请求
    START_DP_WAVE = b"\x02"  # DP 模式：启动新 wave（同步引擎）
    UTILITY = b"\x03"  # 工具调用（reset_cache、add_lora 等）
    # Sentinel used within EngineCoreProc.
    # EngineCoreProc 内部使用的哨兵。
    EXECUTOR_FAILED = b"\x04"  # 执行器失败通知（GPU worker 崩溃）
    # Sentinel to wake up input_queue.get() during shutdown.
    # 关闭期间唤醒 input_queue.get() 的哨兵。
    WAKEUP = b"\x05"  # 唤醒空闲引擎（shutdown 信号处理）


class ReconfigureDistributedRequest(msgspec.Struct):
    # 分布式重配置请求：弹性 EP 扩缩容时发给引擎的配置切换指令
    new_data_parallel_size: int  # 新的 DP 总数
    new_data_parallel_rank: int  # 该引擎的新 DP rank（或 KEEP/SHUTDOWN）
    new_data_parallel_rank_local: int  # 该引擎的本地 DP rank
    new_data_parallel_master_ip: str  # DP 主节点 IP
    new_data_parallel_master_port: int  # DP 主节点端口
    new_data_parallel_master_port_list: list[int]  # 所有引擎端口列表
    coord_store_port: int  # 协调存储端口


class ReconfigureRankType(enum.IntEnum):
    """
    Rank type for reconfiguring distributed request.
    """
    # 重配置分布式请求的 rank 类型

    KEEP_CURRENT_RANK = -1  # 保持当前 rank（不改变）
    SHUTDOWN_CURRENT_RANK = -2  # 关闭当前 rank（缩容移除）


class EngineStatusType(enum.IntEnum):
    # 引擎健康状态类型（容错系统）
    HEALTHY = 0  # 引擎正常运行
    DEAD = 1  # 引擎已死（不可恢复的 GPU 进程崩溃）
    UNHEALTHY = 2  # 引擎不健康（可恢复的临时故障）