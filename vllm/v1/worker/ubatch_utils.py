# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 微批(ubatch)切片工具。
# 根据阈值判断是否启用微批,生成各微批的 token/请求切片,划分注意力元数据,
# 以及处理最后一个空微批等场景。

# 导入 dataclass 装饰器,用于定义 UBatchSlice 数据类。
from dataclasses import dataclass
# 导入 TypeAlias,用于定义类型别名。
from typing import TypeAlias

# 导入 numpy,用于累加与二分查找。
import numpy as np
# 导入 PyTorch,用于张量操作。
import torch

# 导入 ParallelConfig,用于读取微批阈值配置。
from vllm.config import ParallelConfig
# 导入通用注意力元数据,用于划分注意力元数据。
from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass
class UBatchSlice:
    # 单个微批次的数据切片描述。
    # 请求切片:该微批包含的请求索引区间(不含 stop)。
    request_slice: slice
    # token 切片:该微批包含的 token 索引区间(不含 stop)。
    token_slice: slice

    def is_empty(self) -> bool:
        # 判断该切片是否为空(无请求或无 token)。
        return (
            # 请求切片为空(起点等于终点)。
            self.request_slice.start == self.request_slice.stop
            # 或 token 切片为空。
            or self.token_slice.start == self.token_slice.stop
        )

    @property
    def num_tokens(self) -> int:
        # (属性)该微批的 token 数量 = token 切片长度。
        return self.token_slice.stop - self.token_slice.start


# 类型别名:微批切片列表。
UBatchSlices: TypeAlias = list[UBatchSlice]


def is_last_ubatch_empty(
    orig_num_tokens: int, padded_num_tokens: int, num_ubatches: int
) -> bool:
    # 判断最后一个微批是否为空(仅存在于 padding 中)。
    # 参数:
    #   orig_num_tokens: 未填充的 token 总数。
    #   padded_num_tokens: 填充后的 token 总数。
    #   num_ubatches: 微批数量。
    # 若前 (num_ubatches-1) 个微批的总容量(每批 padded_num_tokens//num_ubatches)
    # 已覆盖所有真实 token,则最后一个微批中无真实 token,视为空。
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens


def check_ubatch_thresholds(
    config: ParallelConfig, num_tokens: int, uniform_decode: bool
) -> bool:
    # 检查是否满足启用微批的阈值条件。
    # 参数:
    #   config: 并行配置(含微批开关与阈值)。
    #   num_tokens: 当前批的 token 数。
    #   uniform_decode: 批中是否全为均匀解码。
    # 若未启用微批,返回 False。
    if not config.use_ubatching:
        return False
    # 若是均匀解码,使用 decode 阈值判断。
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    # 否则(含 prefill),使用 prefill 阈值判断。
    else:
        return num_tokens >= config.dbo_prefill_token_threshold


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
# 说明:此函数把最后一个微批切片扩展到总 token 数(token 数 + padding),
# 因为我们在应用 DP padding 之前就创建了微批切片。
def _pad_out_ubatch_slices(
    ubatch_slices: UBatchSlices, num_total_tokens: int, num_reqs_padded: int
) -> UBatchSlices:
    # 把最后一个微批切片扩展到填充后的总 token/请求数。
    # 参数:
    #   ubatch_slices: 原始微批切片列表。
    #   num_total_tokens: 填充后的总 token 数。
    #   num_reqs_padded: 填充后的总请求数。
    # 取最后一个切片的引用。
    last_slice = ubatch_slices[-1]
    # 构造扩展后的请求切片:从原请求起点到填充后的请求总数。
    padded_last_request_slice = slice(last_slice.request_slice.start, num_reqs_padded)
    # 构造扩展后的 token 切片:从原 token 起点到填充后的总 token 数。
    padded_last_token_slice = slice(last_slice.token_slice.start, num_total_tokens)

    # 返回前面切片不变、末尾替换为扩展后切片的新列表。
    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


def maybe_create_ubatch_slices(
    should_ubatch: bool,
    num_scheduled_tokens: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    num_ubatches: int,
    split_point: list[int] | int | None = None,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    # 视条件创建微批切片(含 padding 处理)。
    # 参数:
    #   should_ubatch: 是否启用微批。
    #   num_scheduled_tokens: 各请求调度 token 数数组。
    #   num_tokens_padded: 填充后的 token 总数。
    #   num_reqs_padded: 填充后的请求总数。
    #   num_ubatches: 微批数量。
    #   split_point: 可选,自定义拆分点(单个 int 或每批 token 数列表)。
    # 返回 (未填充切片的微批列表, 含 padding 的微批列表),不启用时 (None, None)。
    # 若不启用微批,返回 (None, None)。
    if not should_ubatch:
        return None, None

    # 若未指定拆分点,按填充后总 token 数均分。
    if split_point is None:
        split_point = int(num_tokens_padded) // num_ubatches

    # 生成各微批的 token 拆分点(第 i 个微批的结束位置 = split_point * i)。
    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    # TODO(lucas): Refactor the gpu_model_runner.py so we can pass
    # in cu_num_tokens directly (i.e. query_start_loc)
    # 注:计划重构 gpu_model_runner.py,使能直接传入 cu_num_tokens(sql 起始位置)。
    # 构造请求 token 累积和数组(长度 = 请求数 + 1)。
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    # 计算累积和。
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    # 初始化微批切片列表。
    ubatch_slices = []
    # 记录当前微批的起始 token。
    start_token = 0

    # Add the end point to the split points to make iteration easier
    # 把总 token 数作为最后一个拆分点,使遍历更简单。
    all_points = token_split_points + [cu_num_tokens[-1]]

    # 遍历每个拆分点作为微批结束位置:
    for end_token in all_points:
        # 构造 token 切片 [start_token, end_token)。
        token_slice = slice(start_token, end_token)

        # Determine request slices using exclusive stop semantics
        # Ubatch includes requests whose tokens overlap [start_token, end_token)
        # 使用排他停止语义确定请求切片:微批包含 token 位于 [start, end) 的请求。
        # Start at the request that contains the start_token
        # or the request starting exactly at start_token (if on boundary)
        # 起点:包含 start_token 的请求(若恰在边界,则请求恰从 start_token 开始)。
        req_start = int(np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

        # Stop at the request that starts at or after end_token
        # 终点:起始位置 >= end_token 的第一个请求。
        req_stop = int(np.searchsorted(cu_num_tokens, end_token, side="left"))

        # 构造请求切片。
        req_slice = slice(req_start, req_stop)
        # 把该微批切片加入列表。
        ubatch_slices.append(UBatchSlice(req_slice, token_slice))

        # 下个微批从当前结束位置开始。
        start_token = end_token

    # 把最后一个微批扩展到包含 padding 的总范围。
    ubatch_slices_padded = _pad_out_ubatch_slices(
        ubatch_slices, num_tokens_padded, num_reqs_padded
    )

    # 断言所有含 padding 微批的 token 数之和等于填充后的总 token 数。
    assert sum(s.num_tokens for s in ubatch_slices_padded) == num_tokens_padded

    # 返回 (未填充切片列表, 含 padding 切片列表)。
    return ubatch_slices, ubatch_slices_padded


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    # 创建与 request_slice 中请求对应的新 query_start_loc。
    # 注意:此函数创建新张量以容纳新的 query_start_locs,这会破坏
    # CUDA Graph 兼容性(因为地址不再固定)。
    return (
        # 取请求区间 [start, stop+1) 的起始位置(含 stop 作为末尾边界)。
        query_start_loc[request_slice.start : request_slice.stop + 1]
        # 减去第一个请求的起始位置,使其相对化(从 0 开始)。
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    # 创建与该微批切片中请求对应的新 CommonAttentionMetadata。
    # 参数:
    #   ubatch_slice: 微批切片。
    #   attn_metadata: 完整批的注意力元数据。
    # 断言微批切片非空。
    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    # 解包请求切片与 token 切片。
    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    # 取完整批的各请求起始位置(CPU)。
    start_locs = attn_metadata.query_start_loc_cpu
    # 该微批第一个请求的索引。
    first_req = request_slice.start
    # 该微批第一个 token 的索引。
    first_tok = token_slice.start
    # 该微批最后一个请求的索引。
    last_req = request_slice.stop - 1
    # 该微批最后一个 token 的索引。
    last_tok = token_slice.stop - 1

    # 断言 token 切片起点落在第一个请求的 token 区间内。
    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # NOTE: last token can be outside of the last request if we have CG padding.
    # 注:若存在 CUDA Graph padding,最后一个 token 可能超出最后一个请求。

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    # 若请求跨微批拆分,则需调整元数据:
    # splits_first_request: 本切片第一个请求是前一切片中某请求的延续。
    # splits_last_request:  本切片最后一个请求延续到下一片。
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    # 计算微批的 CPU 查询起始位置(相对化)。
    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    # 计算微批的设备端查询起始位置(相对化)。
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    # 断言查询起始位置至少 2 个元素(至少一个请求)。
    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    # 若第一个请求被拆分(本微批从其中间开始):
    if splits_first_request:
        # 计算被跳过的 token 数(第一个 token 相对请求起点的偏移)。
        tokens_skipped = first_tok - start_locs[first_req]
        # 设备端查询起始位置减去被跳过的 token(从第 2 个元素起)。
        query_start_loc[1:] -= tokens_skipped
        # CPU 端同样处理。
        query_start_loc_cpu[1:] -= tokens_skipped
    # 取该微批的序列长度张量(设备端)。
    seq_lens = attn_metadata.seq_lens[request_slice]
    # Read raw fields to avoid triggering the deprecated D2H-syncing properties.
    # 直接读原始字段,避免触发已弃用的设备到主机同步属性。
    seq_lens_cpu = (
        # 取 CPU 端序列长度(若存在)。
        attn_metadata._seq_lens_cpu[request_slice]
        if attn_metadata._seq_lens_cpu is not None
        else None
    )
    # 取 CPU 端序列长度上界(若存在)。
    seq_lens_cpu_upper_bound = (
        attn_metadata.seq_lens_cpu_upper_bound[request_slice]
        if attn_metadata.seq_lens_cpu_upper_bound is not None
        else None
    )
    # 取 CPU 端已计算 token 数(若存在)。
    num_computed_tokens_cpu = (
        attn_metadata._num_computed_tokens_cpu[request_slice]
        if attn_metadata._num_computed_tokens_cpu is not None
        else None
    )

    # 若最后一个请求被拆分(本微批在其中间结束):
    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        # 注:使用原始 start_locs 计算被跳过 token,因为 query_start_loc_cpu
        # 可能在 splits_first_request 时已被修改。
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        # 设备端查询起始位置最后一个元素减去被跳过 token。
        query_start_loc[-1] -= tokens_skipped
        # CPU 端同样处理。
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        # (not cudagraph compatible)
        # 确保不修改原始 seq_lens 张量(否则不兼容 CUDA Graph)。
        # 克隆 seq_lens 再调整。
        seq_lens = seq_lens.clone()
        # 最后一个请求的序列长度减去被跳过 token。
        seq_lens[-1] -= tokens_skipped
        # 若存在 CPU 端序列长度,克隆并调整。
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu.clone()
            seq_lens_cpu[-1] -= tokens_skipped
        # 若存在序列长度上界,克隆并调整。
        if seq_lens_cpu_upper_bound is not None:
            seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound.clone()
            seq_lens_cpu_upper_bound[-1] -= tokens_skipped

    # 断言序列长度上界存在(后续取最大值需要)。
    assert seq_lens_cpu_upper_bound is not None
    # Preserve the max_seq_len override set during CUDA-graph capture so
    # the attention backend selects the correct kernel for SWA layers.
    # 保留 CUDA Graph 捕获期设置的 max_seq_len 覆盖,使注意力后端
    # 为滑动窗口注意力(SWA)层选择正确 kernel。
    max_seq_len = max(int(seq_lens_cpu_upper_bound.max()), attn_metadata.max_seq_len)

    # 该微批的请求数。
    num_requests = request_slice.stop - request_slice.start
    # 该微批的实际 token 数。
    num_actual_tokens = token_slice.stop - token_slice.start
    # 计算最大查询长度:查询起始位置相邻差的最大绝对值。
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    # 处理 dummy 运行(查询起始位置全 0)导致最大查询长度为 0 的情况。
    if max_query_len == 0:
        # 回退使用完整批的 max_query_len。
        max_query_len = attn_metadata.max_query_len

    # 取该微批的块表张量(按请求切片)。
    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    # 取该微批的 slot mapping(按 token 切片)。
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    # 构造并返回该微批的注意力元数据。
    return CommonAttentionMetadata(
        # 设备端查询起始位置。
        query_start_loc=query_start_loc,
        # CPU 端查询起始位置。
        query_start_loc_cpu=query_start_loc_cpu,
        # 设备端序列长度。
        seq_lens=seq_lens,
        # 请求数。
        num_reqs=num_requests,
        # 实际 token 数。
        num_actual_tokens=num_actual_tokens,
        # 最大查询长度。
        max_query_len=max_query_len,
        # 最大序列长度。
        max_seq_len=max_seq_len,
        # 块表张量。
        block_table_tensor=block_table_tensor,
        # slot mapping。
        slot_mapping=slot_mapping,
        # CPU 端序列长度上界。
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        # CPU 端序列长度(原始字段)。
        _seq_lens_cpu=seq_lens_cpu,
        # CPU 端已计算 token 数(原始字段)。
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    # 创建与每个微批切片中请求对应的 CommonAttentionMetadata 实例列表。
    # 注意:此函数不会修改 common_attn_metadata。
    # 初始化结果列表。
    results = []
    # 遍历每个微批切片:
    for ubatch_slice in ubatch_slices:
        # 为该切片构造子元数据并加入结果。
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    # 返回各微批的元数据列表。
    return results