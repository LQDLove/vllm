# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 定义 GPU 输入批次(InputBatch)的数据结构。

# 从 dataclasses 导入 dataclass 装饰器,用于定义缓存请求状态数据类。
from dataclasses import dataclass
# 从 typing 导入 cast,用于类型断言转换。
from typing import cast

# 导入 numpy,用于 CPU 侧数组操作。
import numpy as np
# 导入 PyTorch,用于张量操作。
import torch

# 导入推理(思考预算)配置。
from vllm.config.reasoning import ReasoningConfig
# 导入 LoRA 请求数据结构。
from vllm.lora.request import LoRARequest
# 导入多模态特征规格。
from vllm.multimodal.inputs import MultiModalFeatureSpec
# 导入池化参数。
from vllm.pooling_params import PoolingParams
# 导入采样参数与采样类型枚举。
from vllm.sampling_params import SamplingParams, SamplingType
# 导入从 prompt token ids 或 embeds 计算长度的工具函数。
from vllm.utils import length_from_prompt_token_ids_or_embeds
# 导入按索引交换字典值的工具。
from vllm.utils.collection_utils import swap_dict_values
# 导入锁页内存常量(决定张量是否固定内存以加速 H2D 拷贝)。
from vllm.utils.torch_utils import PIN_MEMORY
# 导入 logprobs 张量结构。
from vllm.v1.outputs import LogprobsTensors
# 导入池化元数据与池化状态。
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
# 导入 logits 处理器相关结构:批次更新构建器、处理器集合、移动方向枚举。
from vllm.v1.sample.logits_processor import (
    BatchUpdateBuilder,
    LogitsProcessors,
    MoveDirectionality,
)
# 导入采样元数据。
from vllm.v1.sample.metadata import SamplingMetadata
# 导入思考预算状态持有者的创建函数。
from vllm.v1.sample.thinking_budget_state import (
    maybe_create_thinking_budget_state_holder,
)
# 导入 CPU->GPU 切片拷贝工具。
from vllm.v1.utils import copy_slice
# 导入多组块表与 slot 映射模式枚举。
from vllm.v1.worker.block_table import MultiGroupBlockTable, SlotMappingMode


@dataclass
class CachedRequestState:
    # 缓存的请求状态:GPU model runner 为每个活动请求维护的持久状态。

    # 请求唯一 ID。
    req_id: str
    # prompt 的 token id 列表(使用 prompt_embeds 输入时为 None)。
    prompt_token_ids: list[int] | None
    # 该请求的多模态特征规格列表。
    mm_features: list[MultiModalFeatureSpec]
    # 采样参数(池化请求为 None)。
    sampling_params: SamplingParams | None
    # 该请求专属的随机数生成器(未指定时为 None)。
    generator: torch.Generator | None

    # 每个 KV cache 组的物理块 id 列表组成的元组。
    block_ids: tuple[list[int], ...]
    # 已计算(注意力已处理)的 token 数。
    num_computed_tokens: int
    # 已生成(采样得到)的输出 token id 列表。
    output_token_ids: list[int]

    # mrope 位置张量(多模态 RoPE 位置编码),未使用时为 None。
    mrope_positions: torch.Tensor | None = None
    # mrope 位置增量(用于连续生成时更新位置),未使用时为 None。
    mrope_position_delta: int | None = None

    # xdrope 位置张量(扩展维度 RoPE),未使用时为 None。
    xdrope_positions: torch.Tensor | None = None

    # 该请求关联的 LoRA 请求(不使用 LoRA 时为 None)。
    lora_request: LoRARequest | None = None
    # prompt 嵌入张量(prompt_embeds 输入方式时非空)。
    prompt_embeds: torch.Tensor | None = None
    # 跨多个 prefill 步骤累积的 prompt logprobs 张量分块。
    in_progress_prompt_logprobs_cpu: LogprobsTensors | None = None

    # 混合模式输入(如带 prompt_embeds 内容部分)的按位置掩码,
    # 标记哪些位置本身已是 token,见 `Request.prompt_is_token_ids`。
    prompt_is_token_ids: list[bool] | None = None

    # 同时启用 async_scheduling 与 spec_decode 时记录上一步草稿长度。
    prev_num_draft_len: int = 0

    # 池化模型专用字段。
    # 池化参数(池化请求非空)。
    pooling_params: PoolingParams | None = None
    # 池化状态(池化请求非空)。
    pooling_states: PoolingStates | None = None

    def __post_init__(self):
        # 数据类构造后处理:由 token ids 或 embeds 计算 prompt token 总数。
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )

        # 若为池化请求,初始化空的池化状态对象。
        if self.pooling_params is not None:
            self.pooling_states = PoolingStates()

    @property
    def num_tokens(self) -> int:
        # 返回总 token 数 = prompt token 数 + 已生成输出 token 数。
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        # 返回全局位置 idx 处的 token id;无法得知时返回 -1。
        # 位置落在 prompt 范围内。
        if idx < self.num_prompt_tokens:
            # prompt 由 embeds 提供时无法得知对应 token id。
            if self.prompt_token_ids is None:
                raise ValueError(
                    f"Tried to access token index {idx}, but that token was "
                    "provided via prompt_embeds, and its ID is unknown."
                )
            # 返回 prompt 中对应位置的 token id。
            return self.prompt_token_ids[idx]
        # 位置落在输出 token 范围内,返回对应输出 token id。
        if idx - self.num_prompt_tokens < len(self.output_token_ids):
            return self.output_token_ids[idx - self.num_prompt_tokens]
        # 位置越界,返回 -1 表示未知。
        return -1


class InputBatch:
    # GPU 上的持久批次:以 req_index 为主键维护所有请求的批次状态,
    # 包括 token ids、块表、采样参数、LoRA、logprobs 等的 CPU/GPU 缓冲,
    # 并支持增删请求、交换/压缩行与生成采样元数据。

    def __init__(
        self,
        # 最大并发请求数。
        max_num_reqs: int,
        # 模型最大上下文长度。
        max_model_len: int,
        # 单批最大调度 token 数。
        max_num_batched_tokens: int,
        # 目标 GPU 设备。
        device: torch.device,
        # 词表大小。
        vocab_size: int,
        # 每个 KV cache 组的分配块大小列表。
        block_sizes: list[int],  # The block_size of each kv cache group
        # 每个 KV cache 组的注意力 kernel 块大小列表。
        kernel_block_sizes: list[int],
        # 每请求在每个组的最大块数列表。
        max_num_blocks_per_req: list[int],
        # logits 处理器集合(未提供则为 None)。
        logitsprocs: LogitsProcessors | None = None,
        # logits 处理器是否需要访问输出 token ids。
        logitsprocs_need_output_token_ids: bool = False,
        # 投机解码的草稿 token 数。
        num_spec_tokens: int = 0,
        # 是否为池化(打分)模型。
        is_pooling_model: bool = False,
        # CP 场景下 KV cache 交错大小。
        cp_kv_cache_interleave_size: int = 1,
        # 推理(思考预算)配置,未配置时为 None。
        reasoning_config: ReasoningConfig | None = None,
        # 是否启用 Mamba2 ReplaySSM 特性。
        use_replayssm: bool = False,
        # 每个 KV cache 组的 slot 映射模式列表。
        slot_mapping_modes: list[SlotMappingMode] | None = None,
    ):
        # 按需创建思考预算状态持有者(未配置思考预算时为 None)。
        self.thinking_budget_state_holder = maybe_create_thinking_budget_state_holder(
            reasoning_config,
            max_num_reqs,
            num_spec_tokens,
            device,
            PIN_MEMORY,
        )
        # 当前启用了思考 token 预算的请求 id 集合。
        self.thinking_token_budget_reqs: set[str] = set()
        # 记录是否为池化模型。
        self.is_pooling_model = is_pooling_model
        # 保存最大并发请求数。
        self.max_num_reqs = max_num_reqs
        # 保存模型最大上下文长度。
        self.max_model_len = max_model_len
        # 保存单批最大调度 token 数。
        self.max_num_batched_tokens = max_num_batched_tokens
        # 保存目标设备。
        self.device = device
        # 保存词表大小。
        self.vocab_size = vocab_size

        # 按批次索引存放请求 id;移除后对应位置为 None。
        self._req_ids: list[str | None] = []
        # 请求 id -> 批次索引 的映射。
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): 若 max_model_len 很大此缓冲可能过大,
        # 需要寻找降低 CPU 内存占用的方法。
        # 此缓冲不直接传输到 GPU,因此无需锁页。
        # 分配 [max_num_reqs, max_model_len] 的 int32 全零 CPU 张量,
        # 存放每个请求的完整 token 序列(prompt + 输出)。
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        # 转成 numpy 视图便于高效切片赋值。
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        # 同形状的布尔张量,标记对应位置是否为真实 token id。
        self.is_token_ids_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=bool,
            pin_memory=False,
        )
        # 转成 numpy 视图。
        self.is_token_ids = self.is_token_ids_tensor.numpy()
        # 按请求存放 prompt 嵌入,避免 max_model_len 大时一次性分配 OOM。
        # 映射: req_index -> 形状 (num_prompt_tokens, hidden_size) 的张量
        self.req_prompt_embeds: dict[int, torch.Tensor] = {}
        # CPU 侧记录每请求不含投机 token 的总 token 数(锁页)。
        self.num_tokens_no_spec_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        # 转成 numpy 视图。
        self.num_tokens_no_spec = self.num_tokens_no_spec_cpu_tensor.numpy()
        # CPU 侧记录每请求 prompt token 数(锁页)。
        self.num_prompt_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        # 转成 numpy 视图。
        self.num_prompt_tokens = self.num_prompt_tokens_cpu_tensor.numpy()
        # CPU 侧记录每请求已计算 token 数(锁页)。
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        # 转成 numpy 视图。
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # Mamba2 ReplaySSM 解码环起点(每请求最后一次完整状态写入时的
        # num_computed 值);仅在特性开启时填充。
        self.use_replayssm = use_replayssm
        # CPU 侧记录 ReplaySSM 解码基址(锁页)。
        self.replayssm_decode_base_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        # 转成 numpy 视图。
        self.replayssm_decode_base = self.replayssm_decode_base_cpu_tensor.numpy()

        # 块表:构建多组块表(支持 hybrid 多 KV cache 组)。
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=PIN_MEMORY,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks=max_num_blocks_per_req,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            slot_mapping_modes=slot_mapping_modes,
        )

        # 采样相关缓冲。
        # GPU 侧温度张量(采样时使用)。
        self.temperature = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device=device
        )
        # CPU 侧温度张量(锁页,用于 H2D 传输)。
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=PIN_MEMORY
        )
        # 转成 numpy 视图。
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        # 贪心采样(temperature=0)请求 id 集合。
        self.greedy_reqs: set[str] = set()
        # 随机采样请求 id 集合。
        self.random_reqs: set[str] = set()

        # GPU 侧 top_p 张量。
        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)
        # CPU 侧 top_p 张量(锁页)。
        self.top_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=PIN_MEMORY
        )
        # 转成 numpy 视图。
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        # 启用 top_p(<1)的请求 id 集合。
        self.top_p_reqs: set[str] = set()

        # GPU 侧 top_k 张量。
        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        # CPU 侧 top_k 张量(锁页)。
        self.top_k_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        # 转成 numpy 视图。
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        # 启用 top_k 的请求 id 集合。
        self.top_k_reqs: set[str] = set()

        # 频率惩罚相关数据结构。
        # GPU 侧频率惩罚张量。
        self.frequency_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        # CPU 侧频率惩罚张量(锁页)。
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=PIN_MEMORY
        )
        # 转成 numpy 视图。
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        # 非零频率惩罚的请求 id 集合。
        self.frequency_penalties_reqs: set[str] = set()

        # 存在惩罚(presence penalty)相关数据结构。
        # GPU 侧存在惩罚张量。
        self.presence_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        # CPU 侧存在惩罚张量(锁页)。
        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=PIN_MEMORY
        )
        # 转成 numpy 视图。
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        # 非零存在惩罚的请求 id 集合。
        self.presence_penalties_reqs: set[str] = set()

        # 重复惩罚相关数据结构。
        # GPU 侧重复惩罚张量。
        self.repetition_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        # CPU 侧重复惩罚张量(锁页)。
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=PIN_MEMORY
        )
        # 转成 numpy 视图。
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        # 非默认(≠1.0)重复惩罚的请求 id 集合。
        self.repetition_penalties_reqs: set[str] = set()

        # 投机解码相关。
        # CPU 侧每请求接受的 token 数(初始为 1,锁页)。
        self.num_accepted_tokens_cpu_tensor = torch.ones(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        # 转成 numpy 视图。
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()

        # LoRA 相关。
        # 每请求的 LoRA 整数 id(0 表示无 LoRA)。
        self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
        # LoRA id -> 使用该 LoRA 的请求 id 集合。
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        # LoRA id -> LoRARequest 对象。
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # req_index -> 请求专属生成器
        # 注意(woosuk): 没有专属生成器的请求索引不应出现在该字典中。
        self.generators: dict[int, torch.Generator] = {}

        # req_id -> 需要计算 logprobs 的 top-k 数量。
        self.num_logprobs: dict[str, int] = {}

        # req_id -> 指定要计算 logprobs 的具体 token id 列表,
        # 仅需少量 token 时比 num_logprobs=-1 更高效。
        self.logprob_token_ids: dict[str, list[int]] = {}

        # 每步批次状态变化的内部表示,用于重排持久批次与生成
        # logits 处理器的批次状态更新,每步应重置。
        self.batch_update_builder = BatchUpdateBuilder()

        # TODO 转换为 LogitsProcessor 实现。
        # 使用 allowed_token_ids(允许 token 白名单)的请求 id 集合。
        self.has_allowed_token_ids: set[str] = set()
        # 注意(lufang): 掩码张量中对应 token 被允许的值为 False,
        # 因为后面用 masked_fill_ 填 -inf。
        # GPU 侧允许 token 掩码(懒分配)。
        self.allowed_token_ids_mask: torch.Tensor | None = None
        # CPU 侧允许 token 掩码(懒分配)。
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None

        # req_index -> 禁用词(bad words)token id 列表。
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        # 每请求是否需要输出 token ids 做 logits 处理(布尔数组)。
        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs, dtype=bool)

        # 按批次索引存放各请求的输出 token id 列表。
        self.req_output_token_ids: list[list[int] | None] = []

        # 保存传入的 logits 处理器;未提供时初始化空结构。
        self.logitsprocs = logitsprocs or LogitsProcessors()
        # 记录 logits 处理器是否需要输出 token ids。
        self.logitsprocs_need_output_token_ids = logitsprocs_need_output_token_ids

        # 为采样器保存上一步的投机(draft)token,按请求索引初始化为空表。
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]

        # 每当批次成员变化时更新采样元数据。
        self.sampling_metadata = self._make_sampling_metadata()

        # 池化模型专用。
        # req_id -> 池化参数。
        self.pooling_params: dict[str, PoolingParams] = {}
        # req_id -> 池化状态。
        self.pooling_states: dict[str, PoolingStates] = {}

        # 缓存上一步采样 token 的 GPU 张量引用。
        self.prev_sampled_token_ids: torch.Tensor | None = None
        # 上一步 req_id -> 批次索引 映射。
        self.prev_req_id_to_index: dict[str, int] | None = None
        # 用于在需要时(如惩罚)用上一步真实采样 id 更新 output_token_ids。
        self.sampled_token_ids_cpu: torch.Tensor | None = None
        # 异步拷贝完成事件(异步调度时使用)。
        self.async_copy_ready_event: torch.Event | None = None

    @property
    def req_ids(self) -> list[str]:
        # None 元素只应在对批次做状态更新期间短暂存在。
        # 将 _req_ids 断言转换为 list[str] 返回。
        return cast(list[str], self._req_ids)

    def _register_add_request(self, request: "CachedRequestState") -> int:
        """为 logits 处理器登记添加请求操作。对池化模型不适用。
        """

        # 若存在之前移除请求腾出的空位,优先填充下一个空索引。
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # 否则追加到批次末尾。
            new_req_index = self.num_reqs

        # 断言新索引在容量范围内。
        assert new_req_index < self.max_num_reqs
        # 标记批次已发生变化。
        self.batch_update_builder.batch_changed = True
        if request.sampling_params:
            # 只有非池化模型需要详细的添加请求元数据,以支持 logits 处理器。
            # 记录新增请求的索引、采样参数与 prompt/输出 token ids。
            self.batch_update_builder.added.append(
                (
                    new_req_index,
                    request.sampling_params,
                    request.prompt_token_ids,
                    request.output_token_ids,
                )
            )

        # 返回该请求被分配的批次索引。
        return new_req_index

    def add_request(
        self,
        # 要加入批次的缓存请求状态。
        request: "CachedRequestState",
    ) -> int:
        # 先登记添加操作,拿到批次索引。
        req_index = self._register_add_request(request)

        # 取出请求 id。
        req_id = request.req_id
        # 若索引等于列表当前长度,说明是追加到末尾。
        if req_index == len(self._req_ids):
            # 在各列表末尾追加对应条目。
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
            self.spec_token_ids.append([])
        else:
            # 否则是复用空位,直接覆盖对应索引的条目。
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids
            self.spec_token_ids[req_index].clear()

        # 建立 req_id -> 索引 映射。
        self.req_id_to_index[req_id] = req_index

        # 拷贝 prompt token ids 与输出 token ids。
        # 由 token ids 或 embeds 计算 prompt 长度。
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        # 写入该请求的 prompt token 数。
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        # 输出 token 在行内的起始偏移。
        start_idx = num_prompt_tokens
        # 输出 token 在行内的结束偏移。
        end_idx = start_idx + len(request.output_token_ids)
        # 若 prompt 以 token ids 形式提供。
        if request.prompt_token_ids is not None:
            # 将 prompt token ids 拷入块表行首段。
            self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
            if request.prompt_is_token_ids is not None:
                # 提供了按位置掩码则拷入。
                self.is_token_ids[req_index, :num_prompt_tokens] = (
                    request.prompt_is_token_ids
                )
            else:
                # 否则整段 prompt 都标记为真实 token id。
                self.is_token_ids[req_index, :num_prompt_tokens] = True
        else:
            # prompt 由 embeds 提供,无 token id,整段标记 False。
            self.is_token_ids[req_index, :num_prompt_tokens] = False
        if request.prompt_embeds is not None:
            # 保存该请求的 prompt 嵌入张量。
            self.req_prompt_embeds[req_index] = request.prompt_embeds
        # 将输出 token ids 拷入行内输出段。
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        # 输出段全部标记为真实 token id。
        self.is_token_ids[req_index, start_idx:end_idx] = True
        # 写入不含投机 token 的总 token 数。
        self.num_tokens_no_spec[req_index] = request.num_tokens

        if self.use_replayssm:
            # (重)加入时环起点 = 完整上下文(prompt + 恢复的输出),
            # 使恢复的请求越过 prompt 重新锚定。
            self.replayssm_decode_base[req_index] = request.num_tokens

        # 写入已计算 token 数。
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        # 将该请求的块 id 写入块表对应行。
        self.block_table.add_row(request.block_ids, req_index)

        # 处理采样参数(池化请求没有)。
        if sampling_params := request.sampling_params:
            # 贪心采样类型。
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # 避免后续 apply_temperature 除零,温度置 0。
                self.temperature_cpu[req_index] = 0.0
                # 加入贪心请求集合。
                self.greedy_reqs.add(req_id)
            else:
                # 写入实际温度并加入随机请求集合。
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)

            # 写入 top_p 值。
            self.top_p_cpu[req_index] = sampling_params.top_p
            # top_p < 1 表示启用了 nucleus 采样。
            if sampling_params.top_p < 1:
                self.top_p_reqs.add(req_id)
            # 取出 top_k 值。
            top_k = sampling_params.top_k
            # 0 < top_k < vocab_size 表示启用了 top-k 采样。
            if 0 < top_k < self.vocab_size:
                self.top_k_reqs.add(req_id)
            else:
                # 未启用时等价于使用整个词表。
                top_k = self.vocab_size
            # 写入 top_k 值。
            self.top_k_cpu[req_index] = top_k
            # 写入频率惩罚值。
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
            # 非零则加入频率惩罚集合。
            if sampling_params.frequency_penalty != 0.0:
                self.frequency_penalties_reqs.add(req_id)
            # 写入存在惩罚值。
            self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
            # 非零则加入存在惩罚集合。
            if sampling_params.presence_penalty != 0.0:
                self.presence_penalties_reqs.add(req_id)
            # 写入重复惩罚值。
            self.repetition_penalties_cpu[req_index] = (
                sampling_params.repetition_penalty
            )
            # 不等于默认值 1.0 则加入重复惩罚集合。
            if sampling_params.repetition_penalty != 1.0:
                self.repetition_penalties_reqs.add(req_id)

            # 注意(woosuk): self.generators 不应包含没有专属生成器的请求。
            if request.generator is not None:
                # 登记请求专属生成器。
                self.generators[req_index] = request.generator

            if sampling_params.logprobs is not None:
                # -1 表示返回全部词表 logprobs,否则记录指定数量。
                self.num_logprobs[req_id] = (
                    self.vocab_size
                    if sampling_params.logprobs == -1
                    else sampling_params.logprobs
                )

            # 保存要计算 logprobs 的具体 token ids(更高效)。
            if sampling_params.logprob_token_ids is not None:
                self.logprob_token_ids[req_id] = sampling_params.logprob_token_ids

            if sampling_params.allowed_token_ids:
                # 加入允许 token 白名单请求集合。
                self.has_allowed_token_ids.add(req_id)
                if self.allowed_token_ids_mask_cpu_tensor is None:
                    # 惰性分配该大张量(形状 [max_num_reqs, vocab_size])。
                    # False 表示不填 -inf。
                    self.allowed_token_ids_mask = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    # CPU 侧同步分配一份。
                    self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device="cpu",
                    )
                # 先把整行初始化为 True(即全部禁止)。
                self.allowed_token_ids_mask_cpu_tensor[req_index] = True
                # 把白名单中的 token 位置设为 False(允许,不填 -inf)。
                self.allowed_token_ids_mask_cpu_tensor[req_index][
                    sampling_params.allowed_token_ids
                ] = False

            if sampling_params.bad_words_token_ids:
                # 登记该请求的禁用词 token ids。
                self.bad_words_token_ids[req_index] = (
                    sampling_params.bad_words_token_ids
                )
        elif pooling_params := request.pooling_params:
            # 池化请求分支:取出池化状态。
            pooling_states = request.pooling_states
            # 断言池化状态必须存在。
            assert pooling_states is not None

            # 登记 req_id -> 池化参数。
            self.pooling_params[req_id] = pooling_params
            # 登记 req_id -> 池化状态。
            self.pooling_states[req_id] = pooling_states
            # 记录该池化请求是否需要 token ids。
            self.logits_processing_needs_token_ids[req_index] = (
                pooling_params.requires_token_ids
            )
        else:
            # 既无采样参数也无池化参数,未知请求类型。
            raise NotImplementedError("Unrecognized request type")

        # 投机解码:默认每步接受 1 个 token。
        self.num_accepted_tokens_cpu[req_index] = 1

        # 添加请求的 LoRA 映射。
        if request.lora_request:
            # 取出 LoRA 整数 id。
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                # 该 LoRA 首次出现,初始化其请求集合。
                self.lora_id_to_request_ids[lora_id] = set()

            # 写入该请求索引的 LoRA id。
            self.request_lora_mapping[req_index] = lora_id
            # 把请求 id 加入该 LoRA 的使用集合。
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            # 登记 LoRA id -> LoRARequest。
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # 不使用 LoRA,映射置 0。
            self.request_lora_mapping[req_index] = 0

        # 返回该请求的批次索引。
        return req_index

    def update_req_spec_token_ids(
        self, request: CachedRequestState, scheduled_spec_tokens: dict[str, list[int]]
    ) -> None:
        # 更新该请求本轮调度到的投机(draft)token ids。
        # 取出请求 id。
        req_id = request.req_id
        # 查询其批次索引。
        req_index = self.req_id_to_index[req_id]
        # 取出当前保存的投机 token 列表。
        cur_spec_token_ids = self.spec_token_ids[req_index]
        # 投机解码与结构化输出同用时,调度器可能丢弃不符合
        # schema 的草稿 token,导致 scheduled_spec_decode_tokens
        # 在投机解码开启时也可能为空。
        # 先清空当前投机 token 列表。
        cur_spec_token_ids.clear()
        # 取出本轮调度给该请求的草稿 token(可能为空)。
        spec_token_ids = scheduled_spec_tokens.get(req_id, ())
        # 计算草稿 token 数。
        num_spec_tokens = len(spec_token_ids)
        # 记录上一步草稿长度(供异步调度使用)。
        request.prev_num_draft_len = num_spec_tokens
        # 没有草稿 token 则直接返回。
        if not spec_token_ids:
            return

        # 异步调度下,这里写入 token_ids_cpu 的投机 token 是占位,
        # 之后会在 _prepare_input_ids 中被覆盖。
        # 计算草稿 token 在行内的写入起点。
        start_index = self.num_tokens_no_spec[req_index]
        # 计算写入终点。
        end_token_index = start_index + num_spec_tokens
        # 把草稿 token 占位写入 token 行。
        self.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
        # 对应位置标记为真实 token id。
        self.is_token_ids[req_index, start_index:end_token_index] = True
        # 保存到投机 token 列表。
        cur_spec_token_ids.extend(spec_token_ids)

    def remove_request(self, req_id: str) -> int | None:
        """移除请求;之后必须调用 condense()。

        Args:
          req_id: 要移除的请求 id

        Returns:
          被移除请求的索引;若 `req_id` 未识别则返回 None
        """

        # 从映射中弹出该请求的索引。
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            # 请求不在批次中,返回 None。
            return None

        # 登记该索引被移除(供 condense 与 logits 处理器使用)。
        self.batch_update_builder.removed_append(req_index)
        # 请求 id 槽位置空。
        self._req_ids[req_index] = None
        # 输出 token 列表置空。
        self.req_output_token_ids[req_index] = None
        # 清空投机 token 列表。
        self.spec_token_ids[req_index].clear()
        # 清空块表中该行。
        self.block_table.clear_row(req_index)

        # LoRA 清理。
        # 取出该请求的 LoRA id。
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            # 取出使用该 LoRA 的请求 id 集合。
            lora_req_ids = self.lora_id_to_request_ids[lora_id]
            # 从集合中移除该请求。
            lora_req_ids.discard(req_id)
            if not lora_req_ids:
                # 该 LoRA 已无使用者,删除相关登记。
                del self.lora_id_to_request_ids[lora_id]
                del self.lora_id_to_lora_request[lora_id]
            # 该请求索引的 LoRA 映射归零。
            self.request_lora_mapping[req_index] = 0

        if self.is_pooling_model:
            # 池化模型:移除池化参数与状态后即可返回。
            self.pooling_params.pop(req_id, None)
            self.pooling_states.pop(req_id, None)
            return req_index

        # 从各采样相关集合中移除该请求。
        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        # 移除其专属生成器。
        self.generators.pop(req_index, None)
        # 移除 logprobs 相关登记。
        self.num_logprobs.pop(req_id, None)
        self.logprob_token_ids.pop(req_id, None)
        if self.prev_req_id_to_index is not None:
            # 从上一步索引映射中移除。
            self.prev_req_id_to_index.pop(req_id, None)

        # 移除允许 token 白名单登记。
        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # 将掩码行重置为 False(False 表示不填 -inf)。
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        # 移除禁用词登记。
        self.bad_words_token_ids.pop(req_index, None)
        # 移除思考预算登记。
        self.thinking_token_budget_reqs.discard(req_id)
        # 返回被移除请求的索引。
        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        # 交换批次中两个索引位置的全部请求状态。
        # 暂存两个位置的请求 id。
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        # 只交换每个请求的活动 token 前缀;重排时整行拷贝
        # max_model_len 既昂贵又无必要。
        # 取两个请求的活动 token 数。
        i1_active_token_count = self._get_active_token_count(i1)
        i2_active_token_count = self._get_active_token_count(i2)
        # 活动数取较大值作为拷贝范围。
        max_active_token_count = max(i1_active_token_count, i2_active_token_count)

        # 交换请求 id 列表中的两个槽位。
        self._req_ids[i1], self._req_ids[i2] = self._req_ids[i2], self._req_ids[i1]  # noqa
        # 交换输出 token id 列表。
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )
        # 交换投机 token 列表。
        self.spec_token_ids[i1], self.spec_token_ids[i2] = (
            self.spec_token_ids[i2],
            self.spec_token_ids[i1],
        )
        # 断言两个位置均有有效请求 id。
        assert old_id_i1 is not None and old_id_i2 is not None
        # 交换 req_id -> 索引 映射中的两个条目。
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = (
            self.req_id_to_index[old_id_i2],
            self.req_id_to_index[old_id_i1],
        )
        # 交换不含投机 token 的 token 数。
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] = (
            self.num_tokens_no_spec[i2],
            self.num_tokens_no_spec[i1],
        )
        # 交换 prompt token 数。
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] = (
            self.num_prompt_tokens[i2],
            self.num_prompt_tokens[i1],
        )
        if self.use_replayssm:
            # 交换 ReplaySSM 解码基址。
            self.replayssm_decode_base[i1], self.replayssm_decode_base[i2] = (
                self.replayssm_decode_base[i2],
                self.replayssm_decode_base[i1],
            )
        # 交换已计算 token 数。
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] = (
            self.num_computed_tokens_cpu[i2],
            self.num_computed_tokens_cpu[i1],
        )

        # 注意:下面这种写法是不安全的(别名问题):
        # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
        #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
        # 因此需要先临时拷贝其中一个索引的数据。
        # 拷贝 i1 的活动 token 段。
        tmp_token_ids = self.token_ids_cpu[i1, :max_active_token_count].copy()
        # 将 i2 的活动段拷给 i1。
        self.token_ids_cpu[i1, :max_active_token_count] = self.token_ids_cpu[
            i2, :max_active_token_count
        ]
        # 将暂存的原 i1 数据拷给 i2。
        self.token_ids_cpu[i2, :max_active_token_count] = tmp_token_ids

        # 用花式索引一次性交换 is_token_ids 的活动段。
        self.is_token_ids[[i1, i2], :max_active_token_count] = self.is_token_ids[
            [i2, i1], :max_active_token_count
        ]

        # 若存在 prompt 嵌入则一并交换。
        # 取出两请求的嵌入。
        embeds_i1 = self.req_prompt_embeds.get(i1)
        embeds_i2 = self.req_prompt_embeds.get(i2)
        if embeds_i1 is not None:
            # i1 的嵌入挪给 i2。
            self.req_prompt_embeds[i2] = embeds_i1
        else:
            # i1 无嵌入,i2 处删除。
            self.req_prompt_embeds.pop(i2, None)
        if embeds_i2 is not None:
            # i2 的嵌入挪给 i1。
            self.req_prompt_embeds[i1] = embeds_i2
        else:
            # i2 无嵌入,i1 处删除。
            self.req_prompt_embeds.pop(i1, None)

        # 交换块表中的两行。
        self.block_table.swap_row(i1, i2)

        # 交换 LoRA 映射中的两个条目。
        self.request_lora_mapping[i1], self.request_lora_mapping[i2] = (
            self.request_lora_mapping[i2],
            self.request_lora_mapping[i1],
        )

        if self.is_pooling_model:
            # 池化模型不使用采样与 logits 参数,直接返回。
            return

        # 自回归模型需要记录详细的请求重排信息以支持 logits 处理器。
        self.batch_update_builder.moved.append((i1, i2, MoveDirectionality.SWAP))

        # 交换 CPU 侧温度值。
        self.temperature_cpu[i1], self.temperature_cpu[i2] = (
            self.temperature_cpu[i2],
            self.temperature_cpu[i1],
        )
        # 交换 top_p 值。
        self.top_p_cpu[i1], self.top_p_cpu[i2] = self.top_p_cpu[i2], self.top_p_cpu[i1]
        # 交换 top_k 值。
        self.top_k_cpu[i1], self.top_k_cpu[i2] = self.top_k_cpu[i2], self.top_k_cpu[i1]
        # 交换频率惩罚值。
        self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2] = (
            self.frequency_penalties_cpu[i2],
            self.frequency_penalties_cpu[i1],
        )
        # 交换存在惩罚值。
        self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2] = (
            self.presence_penalties_cpu[i2],
            self.presence_penalties_cpu[i1],
        )
        # 交换重复惩罚值。
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] = (
            self.repetition_penalties_cpu[i2],
            self.repetition_penalties_cpu[i1],
        )
        # 交换接受 token 数。
        self.num_accepted_tokens_cpu[i1], self.num_accepted_tokens_cpu[i2] = (
            self.num_accepted_tokens_cpu[i2],
            self.num_accepted_tokens_cpu[i1],
        )

        # 交换生成器与禁用词字典中的条目。
        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # 交换允许 token 掩码的两行。
            (
                self.allowed_token_ids_mask_cpu_tensor[i1],
                self.allowed_token_ids_mask_cpu_tensor[i2],
            ) = (
                self.allowed_token_ids_mask_cpu_tensor[i2],
                self.allowed_token_ids_mask_cpu_tensor[i1],
            )

    def _get_active_token_count(self, req_index: int) -> int:
        # 返回该请求的活动 token 数 = 不含投机 token 的数量 + 投机 token 数。
        return int(self.num_tokens_no_spec[req_index]) + len(
            self.spec_token_ids[req_index]
        )

    def condense(self) -> None:
        """把非空请求向下滑动,填入更低位置的空索引。

        列表末尾连续的空索引不会被填充。

        Returns:
          swaps: 被移动请求的 (from, to) 交换元组列表
          empty_req_indices: 压缩后仍未填充的索引
        """
        # 取当前批次请求数。
        num_reqs = self.num_reqs

        # 取出被移除请求留下的空索引集合。
        if not (empty_req_indices := self.batch_update_builder.removed):
            # 所有被移除请求都已被新请求填充,或根本没有请求被移除,
            # 无需压缩。
            return
        if num_reqs == 0:
            # 批次状态为空,直接清空各列表。
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            self.spec_token_ids.clear()
            return

        # 注意(woosuk): 此函数假设 empty_req_indices 按降序排序。
        # 初始候选:最后一个非空位置(含空位总数)。
        last_req_index = num_reqs + len(empty_req_indices) - 1
        # 循环直到没有空索引。
        while empty_req_indices:
            # 跳过末尾的空索引,找到最大的非空索引。
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # 取最小的空索引(peek 不弹出)。
            empty_index = self.batch_update_builder.peek_removed()
            # 断言存在空索引。
            assert empty_index is not None
            if empty_index >= last_req_index:
                # 空索引已不小于非空索引,压缩完成。
                break

            # 把活动请求向下移入空索引。
            # 弹出该空索引。
            self.batch_update_builder.pop_removed()
            # 取出待移动请求的 id。
            req_id = self._req_ids[last_req_index]
            # 取出其输出 token 列表。
            output_token_ids = self.req_output_token_ids[last_req_index]
            # 断言请求 id 有效。
            assert req_id is not None
            # 请求 id 写入空位,原位置置空。
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            # 输出 token 列表同样搬移。
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            # 更新 req_id -> 索引 映射。
            self.req_id_to_index[req_id] = empty_index

            # 计算该请求的活动 token 数。
            num_tokens = self._get_active_token_count(last_req_index)

            # 交换投机 token 列表(借用交换实现搬移)。
            (self.spec_token_ids[last_req_index], self.spec_token_ids[empty_index]) = (
                self.spec_token_ids[empty_index],
                self.spec_token_ids[last_req_index],
            )
            # 原位置的投机列表清空。
            self.spec_token_ids[last_req_index].clear()

            # 拷贝活动 token 段。
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            # 拷贝 is_token_ids 段。
            self.is_token_ids[empty_index, :num_tokens] = self.is_token_ids[
                last_req_index, :num_tokens
            ]
            if last_req_index in self.req_prompt_embeds:
                # 搬移 prompt 嵌入(若有)。
                self.req_prompt_embeds[empty_index] = self.req_prompt_embeds.pop(
                    last_req_index
                )
            # 拷贝不含投机 token 的数量。
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index
            ]
            # 拷贝 prompt token 数。
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            if self.use_replayssm:
                # 拷贝 ReplaySSM 解码基址。
                self.replayssm_decode_base[empty_index] = self.replayssm_decode_base[
                    last_req_index
                ]
            # 拷贝已计算 token 数。
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]
            # 块表行搬移。
            self.block_table.move_row(last_req_index, empty_index)

            # 拷贝 LoRA 映射。
            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index
            ]

            if self.is_pooling_model:
                # 池化模型不使用采样状态,处理下一对索引。
                last_req_index -= 1
                continue

            # 自回归模型需要详细记录压缩操作以支持 logits 处理器。
            self.batch_update_builder.moved.append(
                (last_req_index, empty_index, MoveDirectionality.UNIDIRECTIONAL)
            )

            # 依次拷贝各采样相关标量。
            self.temperature_cpu[empty_index] = self.temperature_cpu[last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[empty_index] = self.frequency_penalties_cpu[
                last_req_index
            ]
            self.presence_penalties_cpu[empty_index] = self.presence_penalties_cpu[
                last_req_index
            ]
            self.repetition_penalties_cpu[empty_index] = self.repetition_penalties_cpu[
                last_req_index
            ]
            self.num_accepted_tokens_cpu[empty_index] = self.num_accepted_tokens_cpu[
                last_req_index
            ]
            # 弹出原位置的生成器。
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                # 有生成器则登记到新位置。
                self.generators[empty_index] = generator

            # TODO 将这些转换为 LogitsProcessor。
            if self.allowed_token_ids_mask_cpu_tensor is not None:
                # 拷贝允许 token 掩码行(若已分配)。
                self.allowed_token_ids_mask_cpu_tensor[empty_index] = (
                    self.allowed_token_ids_mask_cpu_tensor[last_req_index]
                )

            # 弹出原位置的禁用词。
            bad_words_token_ids = self.bad_words_token_ids.pop(last_req_index, None)
            if bad_words_token_ids is not None:
                # 有禁用词则登记到新位置。
                self.bad_words_token_ids[empty_index] = bad_words_token_ids

            # 原位置已空,向前递减。
            last_req_index -= 1

        # 把各列表裁剪到当前批次大小。
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]
        del self.spec_token_ids[num_reqs:]

    def refresh_metadata(self):
        """将累积的批次更新应用到采样元数据。"""

        if self.is_pooling_model:
            # 池化模型:重置批次更新构建器。
            batch_changed = self.batch_update_builder.reset()
            if batch_changed:
                # 批次变化时重建采样元数据。
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # 非池化模型:生成并应用 logits 处理器更新;
        # 重置批次更新跟踪;批次状态变化时更新采样元数据。
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        if self.thinking_budget_state_holder is not None and batch_update:
            # 同步思考预算状态持有者。
            self.thinking_budget_state_holder.sync_batch(batch_update)
        # 依次让每个 logits 处理器更新其内部状态。
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            # 批次有变化则重建采样元数据。
            self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        # 构建当前批次的采样元数据(每次批次变化后调用)。
        # 取当前请求数。
        num_reqs = self.num_reqs
        if not self.all_greedy:
            # 存在随机采样请求,同步温度到 GPU。
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            # 全部贪心,无需温度。
            temperature = None
        if not self.no_top_p:
            # 存在 top_p 请求,同步 top_p。
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            # 存在 top_k 请求,同步 top_k。
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
            # 同步这些张量开销较大,仅在必要时拷贝,
            # 即有请求需要在采样时应用惩罚。
            # 同步频率惩罚。
            copy_slice(
                self.frequency_penalties_cpu_tensor, self.frequency_penalties, num_reqs
            )
            # 同步存在惩罚。
            copy_slice(
                self.presence_penalties_cpu_tensor, self.presence_penalties, num_reqs
            )
            # 同步重复惩罚。
            copy_slice(
                self.repetition_penalties_cpu_tensor,
                self.repetition_penalties,
                num_reqs,
            )

        # 是否需要 prompt token ids:有惩罚请求或 logits 处理需要。
        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any()
        )
        # prompt token 仅在应用惩罚或 step pooling 时使用,
        # 因此只有存在需要惩罚/step_pooler 的请求时才拷贝。
        # 构建 CPU 侧 prompt token 张量(或 None)。
        prompt_token_ids_cpu = (
            self._make_prompt_token_ids_cpu_tensor() if needs_prompt_token_ids else None
        )
        # 异步拷贝到 GPU(或 None)。
        prompt_token_ids = (
            prompt_token_ids_cpu.to(device=self.device, non_blocking=True)
            if prompt_token_ids_cpu is not None
            else None
        )

        # 仅当当前请求的采样参数需要时才设置 output_token_ids。
        # 取思考预算状态持有者。
        holder = self.thinking_budget_state_holder
        # 是否有请求正被思考预算跟踪。
        thinking_budget_tracks_reqs = (
            holder is not None and holder.has_tracked_requests()
        )
        # 判断是否需要输出 token ids。
        needs_output_token_ids = (
            not self.no_penalties
            or bool(self.bad_words_token_ids)
            or self.logitsprocs_need_output_token_ids
            or thinking_budget_tracks_reqs
        )
        # 需要则传入输出 token 列表,否则传空列表。
        output_token_ids = (
            cast(list[list[int]], self.req_output_token_ids)
            if needs_output_token_ids
            else []
        )

        # 允许 token 掩码变量初始化为 None。
        allowed_token_ids_mask: torch.Tensor | None = None
        if not self.no_allowed_token_ids:
            # 存在使用白名单的请求。
            # 断言 GPU 侧掩码已分配。
            assert self.allowed_token_ids_mask is not None
            # 同步掩码到 GPU。
            copy_slice(
                self.allowed_token_ids_mask_cpu_tensor,
                self.allowed_token_ids_mask,
                num_reqs,
            )
            # 截取当前批次的掩码切片。
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        # 构建 req_index -> token_ids 的 logprob 映射。
        logprob_token_ids_by_index: dict[int, list[int]] | None = None
        if self.logprob_token_ids:
            # 存在指定 token logprobs 请求。
            # 初始化映射字典。
            logprob_token_ids_by_index = {}
            for req_id, token_ids in self.logprob_token_ids.items():
                if req_id in self.req_id_to_index:
                    # 请求仍在批次中,转换为索引键。
                    req_index = self.req_id_to_index[req_id]
                    logprob_token_ids_by_index[req_index] = token_ids

        # 组装并返回采样元数据。
        return SamplingMetadata(
            # 温度张量(全贪心时为 None)。
            temperature=temperature,
            # 是否全部贪心。
            all_greedy=self.all_greedy,
            # 是否全部随机。
            all_random=self.all_random,
            # top_p(未启用时为 None)。
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            # top_k(未启用时为 None)。
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            # 请求专属生成器。
            generators=self.generators,
            # 各请求最大 logprobs 数。
            max_num_logprobs=self.max_num_logprobs,
            # req_index -> 指定 logprob token ids。
            logprob_token_ids=logprob_token_ids_by_index,
            # GPU 侧 prompt token ids。
            prompt_token_ids=prompt_token_ids,
            # 频率惩罚切片。
            frequency_penalties=self.frequency_penalties[:num_reqs],
            # 存在惩罚切片。
            presence_penalties=self.presence_penalties[:num_reqs],
            # 重复惩罚切片。
            repetition_penalties=self.repetition_penalties[:num_reqs],
            # 输出 token 列表。
            output_token_ids=output_token_ids,
            # 投机 token 列表。
            spec_token_ids=self.spec_token_ids,
            # 是否无惩罚请求。
            no_penalties=self.no_penalties,
            # 允许 token 掩码。
            allowed_token_ids_mask=allowed_token_ids_mask,
            # 禁用词映射。
            bad_words_token_ids=self.bad_words_token_ids,
            # logits 处理器集合。
            logitsprocs=self.logitsprocs,
            # 思考预算状态持有者。
            thinking_budget_state_holder=self.thinking_budget_state_holder,
        )

    def get_pooling_params(self) -> list[PoolingParams]:
        # 断言请求数与池化参数数一致。
        assert len(self.req_ids) == len(self.pooling_params)
        # 按 req_ids 顺序返回各请求的池化参数。
        return [self.pooling_params[req_id] for req_id in self.req_ids]

    def get_pooling_states(self) -> list[PoolingStates]:
        # 断言请求数与池化状态数一致。
        assert len(self.req_ids) == len(self.pooling_states)
        # 按 req_ids 顺序返回各请求的池化状态。
        return [self.pooling_states[req_id] for req_id in self.req_ids]

    def get_pooling_metadata(self) -> PoolingMetadata:
        # 取当前批次的池化参数列表。
        pooling_params = self.get_pooling_params()
        # 取当前批次的池化状态列表。
        pooling_states = self.get_pooling_states()
        # prompt token ids CPU 张量默认 None。
        prompt_token_ids_cpu = None
        if any(p.requires_token_ids for p in pooling_params):
            # 有池化请求需要 token ids 时才构建。
            prompt_token_ids_cpu = self._make_prompt_token_ids_cpu_tensor()

        # 组装并返回池化元数据。
        return PoolingMetadata(
            # prompt 长度切片(克隆)。
            prompt_lens=self.num_prompt_tokens_cpu_tensor[: self.num_reqs].clone(),
            # prompt token ids(复用采样元数据中的)。
            prompt_token_ids=self.sampling_metadata.prompt_token_ids,
            # CPU 侧 prompt token ids。
            prompt_token_ids_cpu=prompt_token_ids_cpu,
            # 池化参数列表。
            pooling_params=pooling_params,
            # 池化状态列表。
            pooling_states=pooling_states,
        )

    def _make_prompt_token_ids_cpu_tensor(self) -> torch.Tensor:
        # 构建 CPU 侧 prompt token ids 张量(右侧填充)。
        # 取当前请求数。
        num_reqs = self.num_reqs
        # 计算批次内最大 prompt 长度。
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        # 分配 [num_reqs, max_prompt_len] 的 int64 锁页张量。
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=PIN_MEMORY,
        )
        # 转 numpy 视图。
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        # 用 token 行的 prompt 段填充(短行自动截断)。
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        # 用 vocab_size 作为填充值(词表中不存在该 token id)。
        for i in range(num_reqs):
            # 把各行超出实际 prompt 长度的尾部填为 vocab_size。
            prompt_token_ids[i, self.num_prompt_tokens[i] :] = self.vocab_size
        # 返回构建好的张量。
        return prompt_token_ids_cpu_tensor

    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        给定批次中每个请求的 num_scheduled_tokens,返回激活当前
        LoRA 所需的数据结构。
        Returns:
            1. prompt_lora_mapping: 长度为 np.sum(num_sampled_tokens) 的元组,
               prompt_lora_mapping[i] 为第 i 个采样 token 应使用的 LoRA id。
            2. token_lora_mapping: 长度为 np.sum(num_scheduled_tokens) 的元组,
               token_lora_mapping[i] 为第 i 个 token 应使用的 LoRA id。
            3. lora_requests: 相关 LoRA 请求集合。
        """

        # 取当前批次的请求 LoRA 映射切片。
        req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
        # 按每请求采样 token 数重复,生成 prompt 级 LoRA 映射。
        prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
        # 按每请求调度 token 数重复,生成 token 级 LoRA 映射。
        token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))

        # 收集所有活跃的 LoRA 请求。
        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values()
        )

        # 返回映射与请求集合。
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    def set_async_sampled_token_ids(
        self,
        sampled_token_ids_cpu: torch.Tensor,
        async_copy_ready_event: torch.Event,
    ) -> None:
        """
        异步调度场景:保存 sampled_token_ids_cpu 张量引用及对应的
        拷贝就绪事件。用于在 logits 处理器需要时,在采样前修复
        output_token_ids。
        """
        if self.sampling_metadata.output_token_ids:
            # 采样元数据需要输出 token ids,保存引用与事件。
            self.sampled_token_ids_cpu = sampled_token_ids_cpu
            self.async_copy_ready_event = async_copy_ready_event
        else:
            # 不需要则清空引用。
            self.sampled_token_ids_cpu = None
            self.async_copy_ready_event = None

    def update_async_output_token_ids(self) -> None:
        """
        异步调度场景:当上一步采样 token 拷贝到 CPU 完成后,用它更新
        采样元数据中的 output_token_ids。在 logits 处理器需要之前调用。
        """
        # 取当前采样元数据中的输出 token 列表。
        output_token_ids = self.sampling_metadata.output_token_ids
        if self.sampled_token_ids_cpu is None or not output_token_ids:
            # 输出 token ids 不需要或未启用异步调度。
            return

        # 断言上一步索引映射存在。
        assert self.prev_req_id_to_index is not None
        # 上一步采样结果列表,首次需要时才同步并转换。
        sampled_token_ids = None
        # 遍历当前批次的每个请求。
        for index, req_id in enumerate(self.req_ids):
            # 查询该请求上一步的批次索引。
            prev_index = self.prev_req_id_to_index.get(req_id)
            if prev_index is None:
                # 上一步不在批次中,跳过。
                continue
            # 取该请求当前输出 token 列表。
            req_output_token_ids = output_token_ids[index]
            if not req_output_token_ids or req_output_token_ids[-1] != -1:
                # 末尾不是占位符 -1,说明 kv-load 失败后有 token 被丢弃,
                # 无需替换,跳过。
                continue
            if sampled_token_ids is None:
                # 首次需要:断言事件存在并等待异步拷贝完成。
                assert self.async_copy_ready_event is not None
                self.async_copy_ready_event.synchronize()
                # 转成 Python 列表。
                sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            # 取该请求上一步的实际采样 id,替换占位符。
            new_ids: list[int] = sampled_token_ids[prev_index]
            if not new_ids:
                # 无新采样 id,跳过。
                continue
            # -1 之后的都是无效值,只取有效数量。
            num_sampled_ids = len(new_ids) if new_ids[-1] != -1 else new_ids.index(-1)
            # 兼容占位符偏少(kv-load 失败丢弃 token)或偏多
            # (异步投机解码的乐观占位可能超过实际接受数)两种情况。
            # 定位第一个占位符位置。
            first_placeholder = len(req_output_token_ids)
            while (
                first_placeholder > 0
                and req_output_token_ids[first_placeholder - 1] == -1
            ):
                # 向前扫描连续的 -1 占位符。
                first_placeholder -= 1
            # 统计占位符数量。
            num_placeholders = len(req_output_token_ids) - first_placeholder
            # 实际可替换数量取两者较小值。
            num_to_replace = min(num_sampled_ids, num_placeholders)
            # 截断新 id 列表到可替换数量。
            del new_ids[num_to_replace:]
            # 用真实采样 id 替换占位符段。
            req_output_token_ids[first_placeholder:] = new_ids
            # ^ 该赋值隐式把列表调整为 (first_placeholder + num_to_replace) 长度

    def update_async_spec_token_ids(self, draft_token_ids: list[list[int]]) -> None:
        """
        异步调度场景:用上一步真实的草稿 token ids 更新采样元数据中的
        spec_token_ids。在拒绝采样器做惩罚/禁用词计算需要它们之前调用。
        """
        if not draft_token_ids or not self.prev_req_id_to_index:
            # 无草稿 token 或无上一步映射,直接返回。
            return

        # 取采样元数据中的投机 token 列表。
        if (spec_token_ids := self.sampling_metadata.spec_token_ids) is not None:
            # 遍历当前批次每个请求的投机列表。
            for req_id, spec_ids in zip(self.req_ids, spec_token_ids):
                if spec_ids:
                    # 该请求有投机 token,查询上一步索引。
                    prev_index = self.prev_req_id_to_index.get(req_id)
                    if prev_index is not None:
                        # 取上一步该请求的草稿 ids。
                        draft_ids = draft_token_ids[prev_index]
                        if draft_ids:
                            # 截断草稿 ids 至投机列表长度(丢弃多余部分)。
                            del draft_ids[len(spec_ids) :]
                            # 清空后写入真实草稿 ids。
                            spec_ids.clear()
                            spec_ids.extend(draft_ids)

    @property
    def num_reqs(self) -> int:
        # 返回当前批次的请求总数。
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        # 无随机采样请求即全部贪心。
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        # 无贪心请求即全部随机。
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        # 无 top_p 请求时为 True。
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        # 无 top_k 请求时为 True。
        return len(self.top_k_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        # 三类惩罚均无请求启用时为 True。
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def no_thinking_budget(self) -> bool:
        # 思考预算未启用或无请求跟踪时为 True。
        return (
            self.thinking_budget_state_holder is None
            or len(self.thinking_token_budget_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:
        # 返回各请求所需最大 logprobs 数;无请求需要时为 None。
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_allowed_token_ids(self) -> bool:
        # 无白名单请求时为 True。
        return len(self.has_allowed_token_ids) == 0
