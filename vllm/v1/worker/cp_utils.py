# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 上下文并行(CP / DCP)工具。
# 检查注意力后端与 CP 的兼容性、获取 KV cache 分片数、为 DCP CUDA Graph 热身
# 准备 dummy 上下文元数据、判定/切分 DCP 上下文注意力,并支持将 Fa2 的
# DCP 上下文注意力按 decode/extend 区域分离执行。

# 导入 SimpleNamespace,用于构造轻量的命名空间对象(传给元数据拆分工具)。
from types import SimpleNamespace
# 导入类型工具:TYPE_CHECKING 条件导入、Any 任意类型、cast 类型转换。
from typing import TYPE_CHECKING, Any, cast

# 导入 PyTorch,用于张量操作。
import torch

# 导入 VllmConfig(配置)与 get_layers_from_vllm_config(按层类型取层配置)。
from vllm.config import VllmConfig, get_layers_from_vllm_config
# 导入 get_dcp_group,用于获取解码上下文并行(DCP)进程组。
from vllm.distributed import get_dcp_group
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入通用注意力元数据,用于构造 DCP 查询切分所需的元数据对象。
from vllm.v1.attention.backend import CommonAttentionMetadata
# 导入请求拆分工具:把请求划分为 decode / prefill / extend 三组。
from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends

# 仅类型检查时导入 AttentionLayerBase(避免运行时循环依赖)。
if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    # 运行时用 object 占位,保持类型签名稳定。
    AttentionLayerBase = object

# 创建本模块的日志记录器。
logger = init_logger(__name__)


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    # 检查注意力后端与上下文并行配置(PCP/DCP/interleave)的兼容性,不兼容则断言失败。
    # 参数: vllm_config: 完整 vLLM 配置。
    # 读取预填充上下文并行大小(PCP)。
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    # 读取解码上下文并行大小(DCP)。
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    # 读取 CP 的 KV 交错粒度。
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    # 只要启用了 PCP 或 DCP(乘积 > 1)就需要逐层检查:
    if pcp_size * dcp_size > 1:
        # 将占位类型转换为 AttentionLayerBase 类型,用于按类型取层。
        layer_type = cast(type[Any], AttentionLayerBase)
        # 从配置中获取所有注意力层(按层名 -> 层实例/配置)。
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        # 遍历每一层:
        for layer in layers.values():
            # 尝试获取层的 get_attn_backend 方法(可能不存在)。
            get_attn_backend = getattr(layer, "get_attn_backend", None)
            # 若启用了 PCP 且层能提供后端:
            if pcp_size > 1 and get_attn_backend is not None:
                # 获取该层的注意力后端。
                backend = get_attn_backend()
                # 断言后端支持 PCP。
                assert backend.supports_pcp(), (
                    "PCP requires attention backend support, "
                    f"but {backend.get_name()} does not support PCP."
                )
            # 尝试获取层的实现对象(impl),用于检查具体实现能力。
            layer_impl = getattr(layer, "impl", None)
            # 无实现对象则跳过该层。
            if layer_impl is None:
                continue
            # 若配置了规范化解码,且 KV 交错粒度 > 1:
            if vllm_config.speculative_config is not None and interleave_size > 1:
                # 断言实现支持非平凡交错粒度下的 MTP(多 token 预测)。
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            # 若启用了 DCP:
            if dcp_size > 1:
                # 断言实现需要在 decode 时返回 softmax LSE(用于跨 rank 合并)。
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )


def get_kv_cache_shard_count() -> int:
    # 返回 KV cache 的分片数(即 DCP 世界大小)。
    try:
        # 尝试从 DCP 组获取世界大小。
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        # 在测试环境中 DCP 可能尚未初始化,回退为 1。
        dcp_world_size = 1
    # 返回 DCP 世界大小作为 KV cache 分片数。
    return dcp_world_size


def get_dcp_dummy_context_len(
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int,
    has_kv_cache_config: bool,
    create_mixed_batch: bool,
    is_graph_capturing: bool,
    uniform_decode: bool,
) -> int:
    # 计算 DCP 下用于 CUDA Graph 热身的 dummy 上下文长度。
    # 参数:
    #   dcp_world_size: DCP 大小。
    #   cp_kv_cache_interleave_size: KV 交错粒度。
    #   has_kv_cache_config: 是否有 KV cache 配置。
    #   create_mixed_batch: 是否创建混合 batch(含上下文)。
    #   is_graph_capturing: 是否正在捕获 CUDA Graph。
    #   uniform_decode: 批中是否全为均匀解码。
    # 若不满足需要 dummy 上下文的条件(无 DCP、无 KV 缓存、
    # 且既非混合 batch 也非均匀解码图捕获):
    if (
        dcp_world_size <= 1
        or not has_kv_cache_config
        or not (create_mixed_batch or (is_graph_capturing and uniform_decode))
    ):
        # 返回 0(不需要 dummy 上下文)。
        return 0
    # 否则返回 dummy 上下文长度 = DCP 大小 × 交错粒度。
    return dcp_world_size * cp_kv_cache_interleave_size


def prepare_dcp_dummy_context_metadata(
    *,
    input_batch: Any,
    kv_cache_config: Any,
    query_pos: Any,
    positions: torch.Tensor,
    query_start_loc: Any,
    num_reqs: int,
    num_tokens_unpadded: int,
    dcp_dummy_context_len: int,
) -> None:
    """Populate valid fake KV metadata for DCP CUDA graph warmup/capture."""
    # 为 DCP CUDA Graph 热身/捕获填充有效的伪 KV 元数据。
    # 参数均为关键字参数:
    #   input_batch: 输入批处理(含块表)。
    #   kv_cache_config: KV cache 配置(提供块数)。
    #   query_pos: 查询位置缓冲(CPU/GPU)。
    #   positions: 位置张量。
    #   query_start_loc: 查询起始位置。
    #   num_reqs: 请求数。
    #   num_tokens_unpadded: 未填充的 token 数。
    #   dcp_dummy_context_len: dummy 上下文长度。
    # 若 dummy 上下文长度为 0,无需准备,直接返回。
    if dcp_dummy_context_len == 0:
        return

    # DCP graph warmup may exercise context attention, so block-table entries
    # must point at allocated KV blocks.
    # 说明:DCP 图热身可能执行上下文注意力,因此块表条目必须指向已分配的 KV 块。
    # 断言 KV cache 配置存在。
    assert kv_cache_config is not None
    # 最大有效块 id = 总块数 - 1。
    max_valid_block_id = kv_cache_config.num_blocks - 1
    # 断言至少存在一个有效块。
    assert max_valid_block_id > 0
    # 遍历输入批处理的每个块表:
    for blk_table in input_batch.block_table.block_tables:
        # 每请求最大内存块数(按 kernel 块换算)。
        max_row_blocks = (
            blk_table.max_num_blocks_per_req // blk_table.blocks_per_kv_block
        )
        # 生成一组有效块 id(循环取模,保证落在已分配块内)。
        block_ids = [
            (block_idx % max_valid_block_id) + 1 for block_idx in range(max_row_blocks)
        ]
        # 为每个请求写入相同的块 id 行(dummy 数据)。
        for req_idx in range(num_reqs):
            blk_table.add_row(block_ids, req_idx)
        # 提交所有请求的块表更新。
        blk_table.commit_block_table(num_reqs)

    # 把未填充 token 数的查询位置拷贝到 GPU。
    query_pos.copy_to_gpu(num_tokens_unpadded)
    # 把位置张量前移 dummy 上下文长度,模拟带上下文的伪解码。
    positions[:num_tokens_unpadded] = (
        query_pos.gpu[:num_tokens_unpadded] + dcp_dummy_context_len
    )
    # 根据伪位置计算每个 token 的 slot mapping(指向有效 KV 块)。
    input_batch.block_table.compute_slot_mapping(
        num_reqs,
        query_start_loc.gpu[: num_reqs + 1],
        positions[:num_tokens_unpadded],
    )


def should_skip_dcp_context_attention(context_kv_lens_cpu: torch.Tensor) -> bool:
    """Whether DCP context attention can be skipped for this batch.

    Must be computed from rank-invariant inputs only (the global context
    lengths, NOT this rank's local share from get_dcp_local_seq_lens): the
    non-skip path in _forward_with_dcp issues DCP collectives (query
    all-gather + LSE combine), so every DCP rank must take the same branch.
    A rank can hold zero local context tokens while other ranks still hold
    context for the same batch.
    """
    # 判断本批是否可以跳过 DCP 上下文注意力。
    # 必须只用与 rank 无关的输入(全局上下文长度,而非本 rank 的本地份额)
    # 计算:非跳过路径会发起 DCP 集合通信(查询 all-gather + LSE 合并),
    # 因此每个 DCP rank 必须走相同分支。
    # 某个 rank 可能本地上下文 token 为 0,而其它 rank 仍持有同一批的上下文。
    # 取全局上下文长度张量的最大值并转为 int,等于 0 说明没有任何上下文。
    return int(context_kv_lens_cpu.max().item()) == 0


def split_dcp_context_queries(
    query_start_loc: torch.Tensor,
    seq_lens_cpu_upper_bound: torch.Tensor | None,
    max_query_len: int,
    num_actual_tokens: int,
) -> tuple[int, int, int, int]:
    """Split reordered DCP context queries into decode and extend regions."""
    # 把重排后的 DCP 上下文查询切分为 decode 与 extend 两个区域。
    # 返回 (num_decodes, num_extends, num_decode_tokens, num_extend_tokens)。
    # 请求数 = 查询起始位置长度 - 1。
    num_reqs = query_start_loc.shape[0] - 1
    # 若最大查询长度为 1(全部是单 token 解码):
    if max_query_len <= 1:
        # 全部是 decode:返回 (num_reqs, 0, num_actual_tokens, 0)。
        return num_reqs, 0, num_actual_tokens, 0
    # 若无 CPU 侧序列长度上界(无法区分 extend):
    if seq_lens_cpu_upper_bound is None:
        # 全部视为 extend:返回 (0, num_reqs, 0, num_actual_tokens)。
        return 0, num_reqs, 0, num_actual_tokens

    # 构造通用注意力元数据命名空间,供请求拆分工具使用。
    common_attn_metadata = cast(
        CommonAttentionMetadata,
        SimpleNamespace(
            max_query_len=max_query_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            query_start_loc_cpu=query_start_loc,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            is_prefilling=None,
        ),
    )
    # 调用拆分工具,把请求分为 decode/extend/prefill 三组并统计 token 数。
    (
        num_decodes,
        num_extends,
        _num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        _num_prefill_tokens,
    ) = split_decodes_prefills_and_extends(common_attn_metadata)
    # 返回 (decode 请求数, extend 请求数, decode token 数, extend token 数)。
    return num_decodes, num_extends, num_decode_tokens, num_extend_tokens


def should_split_fa2_dcp_context_attention(
    fa_version: int | None,
    max_query_len: int,
    num_reqs: int,
    num_decode_reqs: int,
    num_context_prefill_reqs: int,
) -> bool:
    # 判断是否需要对 Fa2 后端执行 DCP 上下文注意力切分。
    # 参数:
    #   fa_version: FlashAttention 版本(2 为 Fa2)。
    #   max_query_len: 最大查询长度。
    #   num_reqs: 请求总数。
    #   num_decode_reqs: decode 请求数。
    #   num_context_prefill_reqs: 带上下文的 prefill 请求数。
    # 总 prefill 请求数 = 总请求数 - decode 请求数。
    num_prefills = num_reqs - num_decode_reqs
    # TODO: Remove this FA2-only DCP compatibility path once FA4 supports
    # the Qwen3.5 head_size=256 shape on Blackwell and can be used here.
    # FA2 paged-varlen context attention can fail for DCP mixed batches when
    # decode rows, context-bearing extend rows, and zero-context pure prefill
    # rows are submitted together.
    # 说明:一旦 FA4 在 Blackwell 上支持 Qwen3.5 head_size=256 形状,即可移除
    # 此 FA2 专用的 DCP 兼容路径。FA2 的 paged-varlen 上下文注意力在
    # decode 行、带上下文的 extend 行与零上下文纯 prefill 行混合提交时可能失败。
    # 返回是否需要切分:Fa2 且有多 token 查询,存在 prefill,
    # 且存在 decode 或部分 prefill 无上下文时,需要切分。
    return (
        fa_version == 2
        and max_query_len > 1
        and num_prefills > 0
        and (num_decode_reqs > 0 or num_context_prefill_reqs < num_prefills)
    )


def run_split_fa2_dcp_context_attention(
    flash_attn_varlen_func: Any,
    query_across_dcp: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    dcp_context_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    dcp_context_kv_lens: torch.Tensor,
    max_dcp_context_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window_size: list[int] | None,
    block_table: torch.Tensor,
    softcap: float,
    fa_version: int,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    max_num_splits: int,
    num_heads: int,
    dcp_world_size: int,
    num_decode_reqs: int,
    num_context_prefill_reqs: int,
    num_decode_tokens: int,
    num_context_prefill_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 执行切分后的 Fa2 DCP 上下文注意力:分别对 decode 段与 context-prefill 段
    # 调用 flash_attn_varlen_func,并把两段写回 dcp_context_out。
    # 返回 (dcp_context_out, context_lse)。
    # 先把输出缓冲清零。
    dcp_context_out.zero_()
    # 初始化上下文 LSE 张量:形状 (num_heads*dcp_world_size, 查询行数),
    # 填充 -inf(Fa2 返回 softmax LSE,初始为负无穷表示无有效行)。
    context_lse = torch.full(
        (num_heads * dcp_world_size, query_across_dcp.shape[0]),
        -torch.inf,
        dtype=torch.float32,
        device=query_across_dcp.device,
    )

    # 若存在 decode token 段:
    if num_decode_tokens > 0:
        # 对 decode 段调用 Fa2 varlen 注意力(decode 段在查询张量最前面)。
        _, decode_context_lse = flash_attn_varlen_func(
            # 查询:前 num_decode_tokens 行。
            q=query_across_dcp[:num_decode_tokens],
            # 键:KV cache 中所有(已按上下文分片)的键。
            k=key_cache,
            # 值:同上。
            v=value_cache,
            # 输出:写到 dcp_context_out 的前 num_decode_tokens 行。
            out=dcp_context_out[:num_decode_tokens],
            # 查询段起始位置:cumsum 前缀,取 decode 请求数 + 1 个元素。
            cu_seqlens_q=cu_seqlens_q[: num_decode_reqs + 1],
            # decode 每行查询长度为 1。
            max_seqlen_q=1,
            # 每请求实际用到的键长度(decode 的上下文 KV 长度)。
            seqused_k=dcp_context_kv_lens[:num_decode_reqs],
            # 最大键长度。
            max_seqlen_k=max_dcp_context_kv_len,
            # softmax 缩放因子。
            softmax_scale=softmax_scale,
            # 上下文注意力为非因果(允许看全部上下文)。
            causal=False,
            # 位置偏置(alibi),可为 None。
            alibi_slopes=alibi_slopes,
            # 滑动窗口大小,可为 None。
            window_size=sliding_window_size,
            # 块表:取前 num_decode_reqs 行。
            block_table=block_table[:num_decode_reqs],
            # softmax 上限(softcap)。
            softcap=softcap,
            # 需要返回 softmax LSE(供 DCP 合并)。
            return_softmax_lse=True,
            # 不使用调度器元数据。
            scheduler_metadata=None,
            # FlashAttention 版本。
            fa_version=fa_version,
            # 量化反缩放(按需切片),可为 None。
            q_descale=q_descale[:num_decode_reqs] if q_descale is not None else None,
            k_descale=k_descale[:num_decode_reqs] if k_descale is not None else None,
            v_descale=v_descale[:num_decode_reqs] if v_descale is not None else None,
            # 最大切分数(用于长上下文 split-k)。
            num_splits=max_num_splits,
        )
        # 把 decode 段的 LSE 写入 context_lse 的前 num_decode_tokens 列。
        context_lse[:, :num_decode_tokens] = decode_context_lse

    # 若存在带上下文的 prefill token 段:
    if num_context_prefill_tokens > 0:
        # prefill 段起点 = decode token 数之后。
        prefill_start = num_decode_tokens
        # prefill 段终点 = 起点 + prefill token 数。
        prefill_end = prefill_start + num_context_prefill_tokens
        # 计算 prefill 请求在查询 cumsum 中的相对起始位置(decode 段后的切片,
        # 并减去 decode token 数,使其从 0 开始)。
        prefill_query_start_loc = (
            cu_seqlens_q[
                num_decode_reqs : num_decode_reqs + num_context_prefill_reqs + 1
            ]
            - num_decode_tokens
        )
        # 构造 prefill 请求的切片(索引范围)。
        prefill_req_slice = slice(
            num_decode_reqs, num_decode_reqs + num_context_prefill_reqs
        )
        # 对 prefill 段调用 Fa2 varlen 注意力。
        _, prefill_context_lse = flash_attn_varlen_func(
            # 查询:prefill 段区间。
            q=query_across_dcp[prefill_start:prefill_end],
            # 键表:全部上下文键。
            k=key_cache,
            # 值表:同上。
            v=value_cache,
            # 输出:prefill 段区间。
            out=dcp_context_out[prefill_start:prefill_end],
            # 查询段起始位置(prefill 请求相对位置)。
            cu_seqlens_q=prefill_query_start_loc,
            # 最大查询段长度。
            max_seqlen_q=max_seqlen_q,
            # 该请求的键长度(上下文长度)。
            seqused_k=dcp_context_kv_lens[prefill_req_slice],
            # 最大键长度。
            max_seqlen_k=max_dcp_context_kv_len,
            # softmax 缩放。
            softmax_scale=softmax_scale,
            # 非因果。
            causal=False,
            # alibi 斜率(可为 None)。
            alibi_slopes=alibi_slopes,
            # 滑动窗口(可为 None)。
            window_size=sliding_window_size,
            # 块表:取 prefill 请求行。
            block_table=block_table[prefill_req_slice],
            # softmax 上限。
            softcap=softcap,
            # 返回 softmax LSE。
            return_softmax_lse=True,
            # 无调度器元数据。
            scheduler_metadata=None,
            # FA 版本。
            fa_version=fa_version,
            # 量化反缩放(按 prefill 请求切片)。
            q_descale=q_descale[prefill_req_slice] if q_descale is not None else None,
            k_descale=k_descale[prefill_req_slice] if k_descale is not None else None,
            v_descale=v_descale[prefill_req_slice] if v_descale is not None else None,
            # 最大切分数。
            num_splits=max_num_splits,
        )
        # 把 prefill 段的 LSE 写入 context_lse 的对应列区间。
        context_lse[:, prefill_start:prefill_end] = prefill_context_lse

    # 返回 (上下文注意力输出, 合并后的上下文 LSE)。
    return dcp_context_out, context_lse