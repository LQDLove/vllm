# SPDX-License-Identifier: Apache-2.0  # 许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明
import dataclasses  # 数据类装饰器（MambaCopyBuffers / MambaBuffers 等）
import itertools  # 迭代工具（chain 合并多个请求 id 集合）
from collections.abc import Callable  # 可调用对象类型注解（make_buffer 工厂）
from typing import Any, NamedTuple  # 类型注解工具（_FusedPrecopy 等）

import torch  # PyTorch 主库（张量与设备操作）

from vllm.config import CacheConfig  # 缓存配置（如 prefix_caching 开关）
from vllm.model_executor.layers.mamba.mamba_utils import (  # Mamba 层通用工具
    MambaStateCopyFunc,  # 状态拷贝函数类型（区分 conv / temporal 语义）
    get_conv_copy_spec,  # conv 状态拷贝规格生成器（源地址/元素数）
    get_temporal_copy_spec,  # temporal 状态拷贝规格生成器
    is_conv_state_dim_first,  # 判断 conv 状态是否为 DS（dim-first）内存布局
)
from vllm.triton_utils import tl, triton  # Triton JIT 与语言原语（GPU kernel 编写）
from vllm.utils.math_utils import cdiv  # 向上取整除法（块数计算）
from vllm.v1.core.sched.output import SchedulerOutput  # 调度器输出（每步调度计划）
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec  # KV cache 配置与 Mamba 规格说明
from vllm.v1.utils import CpuGpuBuffer  # CPU/GPU 镜像缓冲（固定内存 + 非阻塞 H2D 拷贝）
from vllm.v1.worker.gpu_input_batch import CachedRequestState  # 请求持久状态（block_ids 等）
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch  # GPU 输入批（req_ids 等运行期状态）


@triton.jit  # Triton JIT 编译装饰器（GPU 设备函数）
def _copy_mamba_state_block(  # 在块列之间拷贝单个 (层, 状态类型) 的 mamba 状态块
    state_idx,  # 状态索引：展平的 (层 * 状态类型) 元数据槽位
    bt_row_idx,  # 块表行索引（batch 行或 req 槽位）
    src_col,  # 源块列（源 block 在块表中的列号）
    dst_col,  # 目标块列
    token_bias,  # token 偏移：conv 为窗口平移量；temporal 为被接受的 speculative 列偏移
    block_table_ptrs_ptr,  # 各 mamba 组块表的基地址数组（int64 指针）
    block_table_stride_req,  # 块表中请求行之间的步长（元素数）
    state_base_addrs_ptr,  # 各状态张量的基地址数组
    state_block_strides_ptr,  # 各状态每块页步长（字节）
    state_elem_sizes_ptr,  # 各状态元素字节大小
    state_inner_sizes_ptr,  # 各状态内层维度元素数
    state_conv_widths_ptr,  # conv 窗口宽度（temporal 为 0）
    state_group_indices_ptr,  # state_idx -> 块表组索引映射
    # DS conv row metadata. Zero keeps the single-region copy path.
    # DS conv 行元数据；为 0 时保持单区域拷贝路径
    state_dim_row_count_ptr,  # DS conv 每块的 dim 行数
    state_dim_row_stride_ptr,  # DS conv 行间字节数
    COPY_BLOCK_SIZE: tl.constexpr,  # 拷贝循环分块大小（编译期常量，调优参数）
    CONV_STATE_DIM_FIRST: tl.constexpr,  # conv 状态是否为 DS（dim-first）布局
):
    """Copy one (layer, state-type) mamba state block between block columns.

    Shared copy body of ``postprocess_mamba_fused_kernel`` and
    ``precopy_mamba_align_fused_kernel``, mirroring the V1 copy specs
    (``get_conv_copy_spec`` / ``get_temporal_copy_spec``):
    - conv state (conv_width > 0): shift the window by ``token_bias`` tokens,
      ``state[bt[src_col], token_bias:] ->
      state[bt[dst_col], :conv_width - token_bias]``
    - temporal state: ``token_bias`` selects the accepted speculative column,
      ``state[bt[src_col + token_bias]] -> state[bt[dst_col]]``

    The caller owns the decision logic (which columns, whether to copy); this
    device function only performs the byte copy for the given metadata slot.

    在块列之间拷贝一个 (层, 状态类型) 的 mamba 状态块。
    为 postprocess / precopy 两个融合 kernel 共享的拷贝体，语义与 V1 拷贝规格一致：
    - conv 状态（conv_width > 0）：滑窗按 token_bias 平移后拷贝；
    - temporal 状态：token_bias 选择被接受的 speculative 列，整块拷贝。
    决策逻辑（选哪些列、是否拷贝）由调用方负责，本函数仅执行字节拷贝。
    """
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)  # 该状态张量的基地址（字节）
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)  # 每块页步长（字节）
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)  # 元素大小（字节）
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)  # 内层维度元素数
    conv_width = tl.load(state_conv_widths_ptr + state_idx)  # conv 窗口宽度（0 表示 temporal）

    # Load the group index for this state, then index into the correct
    # group's block table. Each mamba group has independently allocated
    # physical blocks. Reinterpret as int32* since block ids are int32.
    # 加载该状态所属组索引，再索引到对应组的块表；各组物理块独立分配。
    # 块 id 为 int32，故把组块表基地址重解释为 int32 指针
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)  # 组索引（扩宽为 int64 用于寻址）
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)  # 该组块表的基地址
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))  # 转为 int32 指针
    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req  # 定位到该请求的块表行首

    # Widen block ids to int64 before they reach `block_id * state_block_stride`
    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,
    # and Triton would otherwise do the multiply in int32 and wrap.
    # 块 id 需先扩宽为 int64 再参与 block_id * 页步长 乘法：
    # 大缓存下页步长可能超过 2**31 字节，int32 乘法会溢出回绕
    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)  # 目标块 id（int64 扩宽）
    dst_addr = state_base_addr + dest_block_id * state_block_stride  # 目标块的字节地址

    is_conv_state = conv_width > 0  # 是否为 conv（滑窗）状态

    if CONV_STATE_DIM_FIRST and is_conv_state:  # DS（dim-first）布局的 conv 状态
        # DS conv layout: state_len is the slide axis; copy per dim row.
        # DS 布局：state_len 是滑动轴，需按 dim 行逐行拷贝
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)  # 源块 id
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)  # dim 行数（每块）
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)  # 行间字节数
        per_row_bytes = (conv_width - token_bias).to(tl.int64) * state_elem_size  # 每行需拷贝的字节数
        bias_bytes = token_bias.to(tl.int64) * state_elem_size  # 源端跳过的偏移字节数
        src_block_addr = state_base_addr + src_block_id * state_block_stride  # 源块起始地址
        offsets = tl.arange(0, COPY_BLOCK_SIZE)  # 拷贝循环内的偏移向量
        for d in range(0, dim_rows):  # 逐 dim 行拷贝
            row_src = src_block_addr + d * row_stride + bias_bytes  # 源行地址（跳过偏移）
            row_dst = dst_addr + d * row_stride  # 目标行地址（从行首写入）
            for i in range(0, per_row_bytes, COPY_BLOCK_SIZE):  # 行内分块拷贝
                mask = (i + offsets) < per_row_bytes  # 尾块边界掩码
                curr_src = (row_src + i + offsets).to(tl.pointer_type(tl.uint8))  # 源字节指针
                curr_dst = (row_dst + i + offsets).to(tl.pointer_type(tl.uint8))  # 目标字节指针
                data = tl.load(curr_src, mask=mask)  # 按掩码读取源数据
                tl.store(curr_dst, data, mask=mask)  # 按掩码写入目标
        return  # DS conv 拷贝完成，提前返回

    if is_conv_state:  # SD（dim 后置）布局的 conv 状态：单段连续拷贝
        # SD conv: copy
        #   state[bt[src_col], token_bias:] ->
        #   state[bt[dst_col], :conv_width - token_bias]
        # SD conv：源端跳过 token_bias 个滑窗位置，拷到目标块开头
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)  # 源块 id
        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size  # 源端偏移字节
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset  # 源地址
        num_elems_to_copy = (conv_width - token_bias).to(tl.int64) * state_inner_size  # 需拷贝元素数
        copy_size = num_elems_to_copy * state_elem_size  # 需拷贝字节数
        offsets = tl.arange(0, COPY_BLOCK_SIZE)  # 偏移向量
        for i in range(0, copy_size, COPY_BLOCK_SIZE):  # 分块拷贝
            mask = (i + offsets) < copy_size  # 尾块边界掩码
            curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))  # 源字节指针
            curr_dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))  # 目标字节指针
            data = tl.load(curr_src, mask=mask)  # 读源数据
            tl.store(curr_dst, data, mask=mask)  # 写目标
        return  # SD conv 拷贝完成，提前返回

    # Temporal state: copy state[bt[src_col + token_bias]] -> state[bt[dst_col]]
    # temporal 状态：token_bias 在源列上选中被接受的 speculative 列，整块拷贝
    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)  # 实际源块 id（含偏移）
    src_addr = state_base_addr + actual_src_block_id * state_block_stride  # 源地址
    # Use natural block data size (inner_size * elem_size), NOT
    # state_block_stride which is the page stride and can exceed the
    # actual data when the state tensor uses as_strided page padding.
    # 使用自然块数据大小（inner_size * elem_size）而非页步长：
    # 状态张量经 as_strided 页填充时，页步长可能大于实际数据
    copy_size = state_inner_size * state_elem_size  # 实际需拷贝的字节数

    # Vectorize via uint64 (8B per thread → LDG.64/STG.64): both temporal
    # and SD conv produce src/dst addresses aligned to a full token slice
    # (inner_size * elem_size) and a copy_size that's a multiple of it,
    # which is 8B-aligned for all state dtypes in use. A masked byte tail
    # covers any remaining 0-7 bytes (only reachable for sub-8B slices).
    # 用 uint64 向量化拷贝（每线程 8 字节 → LDG.64/STG.64）：
    # temporal 与 SD conv 的源/目标地址均按完整 token 切片对齐，
    # copy_size 为其倍数，对现用 dtype 均满足 8 字节对齐；
    # 剩余 0-7 字节由掩码字节尾部循环兜底（仅切片小于 8B 时触发）
    copy_size_u64 = copy_size // 8  # 8 字节单元数
    src_u64 = src_addr.to(tl.pointer_type(tl.uint64))  # 源指针转为 uint64 类型
    dst_u64 = dst_addr.to(tl.pointer_type(tl.uint64))  # 目标指针转为 uint64 类型
    offsets = tl.arange(0, COPY_BLOCK_SIZE)  # 偏移向量
    for i in range(0, copy_size_u64, COPY_BLOCK_SIZE):  # 以 8 字节粒度分块拷贝
        mask = (i + offsets) < copy_size_u64  # 尾块边界掩码
        data = tl.load(src_u64 + i + offsets, mask=mask)  # 读 8 字节单元
        tl.store(dst_u64 + i + offsets, data, mask=mask)  # 写 8 字节单元

    tail_start = copy_size_u64 * 8  # 尾部起始字节偏移
    tail_bytes = copy_size - tail_start  # 尾部字节数（0-7）
    tail_off = tl.arange(0, 8)  # 尾部偏移向量（最多 8 字节）
    tail_src = (src_addr + tail_start).to(tl.pointer_type(tl.uint8))  # 尾部源字节指针
    tail_dst = (dst_addr + tail_start).to(tl.pointer_type(tl.uint8))  # 尾部目标字节指针
    tail_mask = tail_off < tail_bytes  # 尾部有效掩码
    tail_data = tl.load(tail_src + tail_off, mask=tail_mask)  # 读尾部字节
    tl.store(tail_dst + tail_off, tail_data, mask=tail_mask)  # 写尾部字节


@triton.jit(do_not_specialize=["num_reqs"])  # num_reqs 不参与特化，避免按批大小重复编译
def postprocess_mamba_fused_kernel(  # 融合版 mamba 后处理 kernel：决策 + 状态拷贝全在 GPU 完成
    # Decision inputs (per-request)
    # 决策输入（每请求）
    num_accepted_tokens_ptr,  # 每请求被接受的 speculative token 数
    mamba_state_idx_ptr,  # 每请求的 mamba 状态块列（源块索引）
    num_scheduled_tokens_ptr,  # 每请求本轮调度的 token 数
    num_computed_tokens_ptr,  # 每请求已计算的 token 数
    num_draft_tokens_ptr,  # 每请求的草稿 token 数
    # Per-group block table base addresses: int64[num_groups]. Each entry is
    # the data_ptr of that group's persistent [max_reqs, max_blocks] int32
    # block table.
    # 各组块表基地址：int64[组数]；每项是该组持久化 [max_reqs, max_blocks]
    # int32 块表的 data_ptr
    block_table_ptrs_ptr,  # 组块表基地址数组
    block_table_stride_req: tl.int64,  # 块表请求行步长（元素数）
    # Mamba state metadata (per-layer, per-state-type)
    # These are 1D arrays indexed by (layer_idx * num_state_types + state_type_idx)
    # mamba 状态元数据（每层 × 每状态类型），按 (层 * 状态类型数 + 状态类型) 索引
    state_base_addrs_ptr,  # 各状态张量的基地址
    state_block_strides_ptr,  # 各状态每块字节数（页步长）
    state_elem_sizes_ptr,  # 各状态元素字节大小
    state_inner_sizes_ptr,  # 各状态内层维度元素数
    state_conv_widths_ptr,  # conv 宽度（temporal 为 0）
    state_group_indices_ptr,  # state_idx -> 块表组索引映射
    # DS conv row metadata. Zero keeps the single-region copy path.
    # DS conv 行元数据；为 0 时保持单区域拷贝路径
    state_dim_row_count_ptr,  # int32：DS conv 每块 dim 行数
    state_dim_row_stride_ptr,  # int64：DS conv 行间字节数
    # Output: num_accepted_tokens update (for src==dst case)
    # 输出：src==dst 情况下 num_accepted_tokens 的更新值
    num_accepted_tokens_out_ptr,  # 输出缓冲（同块内拷贝时重置为 1）
    # Optional: batch_idx -> req_idx mapping (V2 model runner / PP). The
    # per-request decision arrays are in req-state-slot order; the block table
    # is in batch order, so HAS_IDX_MAPPING splits the two indexings.
    # 可选：batch_idx -> req_idx 映射（V2 runner / PP）。决策数组按 req 槽位
    # 排列而块表按 batch 排列，HAS_IDX_MAPPING 区分两种索引方式
    idx_mapping_ptr,  # batch_idx -> req 槽位映射（-1 表示跳过）
    # Runtime parameter (varies per batch - NOT constexpr to avoid recompilation)
    # 运行期参数（随批变化；非 constexpr 以避免重复编译）
    num_reqs,  # 活跃请求数
    # Compile-time constants (fixed after model initialization)
    # 编译期常量（模型初始化后固定）
    # block_size: determined by model config, constant for all invocations
    # block_size：由模型配置决定，所有调用一致
    block_size: tl.constexpr,  # mamba 块大小（编译期常量）
    # COPY_BLOCK_SIZE: fixed tuning parameter for memory copy loop
    # COPY_BLOCK_SIZE：内存拷贝循环的固定调优参数
    COPY_BLOCK_SIZE: tl.constexpr,  # 拷贝分块大小
    CONV_STATE_DIM_FIRST: tl.constexpr,  # conv 状态是否为 DS（dim-first）布局
    # HAS_IDX_MAPPING: when True, program_id(0) is a batch index resolved to a
    # req-state slot via idx_mapping_ptr (V2). When False, it is the req index.
    # HAS_IDX_MAPPING：为 True 时 program_id(0) 是 batch 索引，需经
    # idx_mapping 解析为 req 槽位（V2）；为 False 时即 req 索引
    HAS_IDX_MAPPING: tl.constexpr = False,  # 是否需要 idx_mapping（默认否）
    # PRECOMPUTED_NEW_COMPUTED: when True, num_computed_tokens_ptr already holds
    # the post-step new_num_computed value (V2 supplies the advanced count).
    # PRECOMPUTED_NEW_COMPUTED：为 True 时 num_computed_tokens_ptr 已存有
    # 本步之后的新计算数（V2 提供前移后的值）
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,  # 是否已预计算 new_computed
):
    """
    Fused GPU kernel for postprocess_mamba that computes decisions AND performs
    mamba state copies without any CPU-GPU synchronization.

    Grid: (num_reqs, num_layers * num_state_types)
    - program_id(0) = request/batch index
    - program_id(1) = state_idx (flattened index into layer/state_type metadata)

    Note: num_layers and num_state_types are not passed as kernel parameters
    because the kernel indexes directly into pre-flattened metadata arrays
    using program_id(1). The grid dimensions encode the total state count.

    融合 GPU kernel：计算后处理决策并完成 mamba 状态拷贝，全程无 CPU-GPU 同步。
    Grid 为 (num_reqs, num_layers * num_state_types)：
    program_id(0) 为请求/batch 索引，program_id(1) 为状态索引。
    层数与状态类型数不作为参数传入——kernel 直接用 program_id(1)
    索引预展平的元数据数组，网格维度即编码了状态总数。
    """
    batch_idx = tl.program_id(0)  # 当前程序处理的请求/batch 索引
    state_idx = tl.program_id(1)  # 当前程序处理的状态索引（展平）

    # Bounds check
    # 边界检查：网格可能大于实际批大小
    if batch_idx >= num_reqs:  # 超出批大小
        return  # 直接退出

    if HAS_IDX_MAPPING:  # V2 / PP 场景：batch 行 -> req 槽位
        req_idx = tl.load(idx_mapping_ptr + batch_idx)  # 解析请求槽位
        if req_idx < 0:  # 无效槽位（该 batch 行无请求）
            return  # 跳过
    else:  # V1：batch 顺序即请求顺序
        req_idx = batch_idx  # 直接使用 batch 索引

    # Compute decision logic (mirrors postprocess_mamba Python reference)
    # 计算决策逻辑（与 postprocess_mamba 的 Python 参考实现一致）
    num_accepted = tl.load(num_accepted_tokens_ptr + req_idx)  # 本步被接受的 token 数
    src_block_idx = tl.load(mamba_state_idx_ptr + req_idx)  # 上一步运行状态所在块列

    if PRECOMPUTED_NEW_COMPUTED:  # V2：已预计算新计算数
        new_num_computed = tl.load(num_computed_tokens_ptr + req_idx)  # 直接读取
        num_tokens_running_state = new_num_computed - num_accepted + 1  # 运行状态对应的 token 数
    else:  # V1：由调度信息推导
        num_scheduled = tl.load(num_scheduled_tokens_ptr + req_idx)  # 本步调度 token 数
        num_computed = tl.load(num_computed_tokens_ptr + req_idx)  # 已计算 token 数
        num_draft = tl.load(num_draft_tokens_ptr + req_idx)  # 草稿 token 数
        num_tokens_running_state = num_computed + num_scheduled - num_draft  # 运行状态 token 数
        new_num_computed = num_tokens_running_state + num_accepted - 1  # 接受后的新计算数

    aligned_new_computed = (new_num_computed // block_size) * block_size  # 向下对齐到块边界

    needs_copy = aligned_new_computed >= num_tokens_running_state  # 是否需要状态拷贝（对齐后越界）

    if not needs_copy:  # 无需拷贝
        return  # 提前退出

    # Compute copy parameters
    # 计算拷贝参数
    accept_token_bias = aligned_new_computed - num_tokens_running_state  # 源块内偏移（接受量）
    dest_block_idx = aligned_new_computed // block_size - 1  # 目标块列（对齐块的前一块）

    # Update accepted-token count before early exits (per-request, so only
    # state_idx == 0 writes).
    # 在提前退出前更新接受计数（每请求只需写一次，故仅 state_idx==0 写）
    if src_block_idx == dest_block_idx and state_idx == 0:  # 源/目标同块
        tl.store(num_accepted_tokens_out_ptr + req_idx, 1)  # 重置接受计数为 1

    # Skip no-op self-copy.
    # 跳过无实际效果的自拷贝
    if src_block_idx == dest_block_idx and accept_token_bias == 0:  # 同块且无偏移
        return  # 无需拷贝

    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx  # 块表行索引按模式选择
    _copy_mamba_state_block(  # 调用共享拷贝体执行实际拷贝
        state_idx,  # 状态索引
        bt_row_idx,  # 块表行索引
        src_block_idx,  # 源块列
        dest_block_idx,  # 目标块列
        accept_token_bias,  # token 偏移
        block_table_ptrs_ptr,  # 组块表基地址数组
        block_table_stride_req,  # 块表行步长
        state_base_addrs_ptr,  # 状态基地址数组
        state_block_strides_ptr,  # 每块字节数数组
        state_elem_sizes_ptr,  # 元素大小数组
        state_inner_sizes_ptr,  # 内层元素数数组
        state_conv_widths_ptr,  # conv 宽度数组
        state_group_indices_ptr,  # 组索引映射数组
        state_dim_row_count_ptr,  # DS conv 行数数组
        state_dim_row_stride_ptr,  # DS conv 行步长数组
        COPY_BLOCK_SIZE,  # 拷贝分块大小
        CONV_STATE_DIM_FIRST,  # conv 布局标志
    )


@triton.jit(do_not_specialize=["num_reqs"])  # num_reqs 不参与特化，避免按批大小重复编译
def preprocess_mamba_align_fused_kernel(  # 融合版 align 预处理：一次启动同时输出预拷贝源列/偏移并前移状态索引（V2 align）
    idx_mapping_ptr,  # batch_idx -> req 槽位映射数组（决策数组按 req 槽位排列）
    state_idx_ptr,  # 每请求当前状态块列（读旧值、写新值，原地前移）
    num_computed_tokens_ptr,  # 每请求已计算 token 数
    query_start_loc_ptr,  # 本批 token 的前缀和数组（长度 num_reqs+1），用于推导各请求本步调度 token 数
    num_accepted_tokens_ptr,  # 每请求被接受的 speculative token 数（读旧值，块跨界时重置）
    src_col_ptr,  # 输出：预拷贝源块列（= 前移前的 state_idx）
    src_off_ptr,  # 输出：预拷贝的接受 token 偏移（= num_accepted - 1）
    num_reqs,  # 活跃请求数（运行期参数，非 constexpr 以避免重复编译）
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的请求向量宽度（编译期常量）
    MAMBA_BLOCK_SIZE: tl.constexpr,  # mamba 状态块大小（编译期常量，模型配置决定）
):
    """Fused align preprocess: emit the pre-copy src column/offset AND advance
    state_idx (with accepted-token reset) in a single launch (V2 align).

    Per batch_idx (0..num_reqs-1), resolving req slot via idx_mapping:
      1. Read pre-advance state_idx and num_accepted (last step's values).
      2. Store the pre-copy src columns for ``precopy_mamba_align_fused_kernel``:
         - src_col = state_idx (the previous running block column)
         - src_off = max(num_accepted - 1, 0) (the accepted-token bias)
      3. Advance state_idx to the new running block, and reset num_accepted to 1
         when a block boundary is crossed (so the migrated state, now at the
         start of the new block, is read with the neutral bias).

    融合 align 预处理 kernel：单次启动同时完成
      (a) 输出预拷贝所需的源块列/偏移（供 precopy kernel 使用）；
      (b) 前移 state_idx 到新的运行块，并在跨越块边界时重置接受计数。
    对每个 batch_idx（0..num_reqs-1），经 idx_mapping 解析 req 槽位后：
      1. 读取前移前的 state_idx 与 num_accepted（上一步的值）；
      2. 写出预拷贝源列与偏移：src_col = state_idx（旧运行块列），
         src_off = max(num_accepted - 1, 0)（接受 token 偏移）；
      3. 前移 state_idx 到新运行块；若跨越块边界则将 num_accepted 重置为 1，
         使已迁移到新块开头的状态以中性偏移被读取。
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 当前 program 负责的请求偏移向量
    mask = offsets < num_reqs  # 向量越界掩码（最后一个 program 可能不满）
    req_indices = tl.load(idx_mapping_ptr + offsets, mask=mask, other=0)  # batch -> req 槽位（无效时填 0，由 mask 屏蔽）

    state_idx = tl.load(state_idx_ptr + req_indices, mask=mask, other=-1)  # 读取前移前状态块列（-1 表示新请求/无状态）
    num_accepted = tl.load(num_accepted_tokens_ptr + req_indices, mask=mask, other=1)  # 读取前移前接受计数

    src_off = tl.maximum(num_accepted - 1, 0)  # 接受偏移：接受 n 个 token 时状态偏移 n-1；下限 0 防止负偏移
    tl.store(src_col_ptr + req_indices, state_idx, mask=mask)  # 输出源块列（旧运行块）
    tl.store(src_off_ptr + req_indices, src_off, mask=mask)  # 输出接受偏移

    num_computed = tl.load(num_computed_tokens_ptr + req_indices, mask=mask, other=0)  # 已计算 token 数
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)  # 本请求在本批 token 中的起始位置
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)  # 本请求在本批 token 中的结束位置
    computed_after = num_computed + query_end - query_start  # 本步计算后的总 token 数（调度数 = end - start）
    new_state_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1  # 前移后的运行块列（向上取整减一）
    tl.store(state_idx_ptr + req_indices, new_state_idx, mask=mask)  # 写回新状态块列（原地前移）
    should_reset = (state_idx >= 0) & (state_idx != new_state_idx)  # 块边界跨越判定：旧块列有效且与新块列不同
    tl.store(num_accepted_tokens_ptr + req_indices, 1, mask=mask & should_reset)  # 跨界时重置接受计数为 1（状态已迁移至新块开头）


@triton.jit(do_not_specialize=["num_reqs"])  # num_reqs 不参与特化，避免按批大小重复编译
def precopy_mamba_align_fused_kernel(
    # Per-request-slot inputs (indexed by req_idx via idx_mapping), produced by
    # the V2 fused align preprocess kernel for the current step:
    # 以下三组每请求数组由本轮 V2 融合 align 预处理 kernel 产出，
    # 按 req 槽位（经 idx_mapping 解析）索引：
    mamba_state_idx_ptr,  # 前移后的目标块列（dst）
    src_col_ptr,  # 前移前的源块列（-1 = 全新状态，无需拷贝）
    token_bias_ptr,  # 接受 token 偏移 = num_accepted - 1（重置前的值）
    # Same flattened state-layout metadata as postprocess_mamba_fused_kernel
    # 与 postprocess_mamba_fused_kernel 相同的展平状态布局元数据
    block_table_ptrs_ptr,  # 各 mamba 组块表基地址数组（int64 指针）
    block_table_stride_req: tl.int64,  # 块表请求行步长（元素数）
    state_base_addrs_ptr,  # 各状态张量基地址数组
    state_block_strides_ptr,  # 各状态每块页步长（字节）
    state_elem_sizes_ptr,  # 各状态元素字节大小
    state_inner_sizes_ptr,  # 各状态内层维度元素数
    state_conv_widths_ptr,  # conv 窗口宽度（temporal 为 0）
    state_group_indices_ptr,  # state_idx -> 块表组索引映射
    state_dim_row_count_ptr,  # DS conv 每块 dim 行数
    state_dim_row_stride_ptr,  # DS conv 行间字节数
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)（batch -> req 槽位映射，-1 跳过）
    num_reqs,  # 活跃请求数（运行期参数，非 constexpr 以避免重复编译）
    COPY_BLOCK_SIZE: tl.constexpr,  # 拷贝循环分块大小（编译期调优常量）
    CONV_STATE_DIM_FIRST: tl.constexpr,  # conv 状态是否为 DS（dim-first）布局
    HAS_IDX_MAPPING: tl.constexpr = True,  # 是否需要 idx_mapping（V2 为真；V1 数组本身按 batch 排列时为假）
):
    """Pre-copy mamba "align" state across block boundaries.

    Before the forward pass, copy each request's last SSM/conv state from its
    previous block column into the new window block column, so the kernels read
    the initial state from the write-side block as usual (V1 align semantics).
    Same per-(layer, state) copy semantics as ``postprocess_mamba_fused_kernel``
    (shared ``_copy_mamba_state_block`` body, i.e. the V1 ``preprocess_mamba``
    copy specs), but driven by the GPU-resident src columns so it needs no
    CPU-GPU sync (async-scheduling safe).

    Grid: (num_reqs, num_layers * num_state_types). V2 passes a batch-to-state
    idx_mapping; V1 already stores the staged arrays in batch order and uses
    HAS_IDX_MAPPING=False.

    前向推理前，将各请求上一个块列中的最新 SSM/conv 状态预拷贝到新窗口块列，
    使 kernel 照常从写侧块读取初始状态（与 V1 align 语义一致）。
    每 (层, 状态) 的拷贝语义与 postprocess kernel 相同（共享
    _copy_mamba_state_block 拷贝体）；区别在于源列由 GPU 端驻留数据驱动，
    无需 CPU-GPU 同步（对异步调度安全）。
    Grid 为 (num_reqs, num_layers * num_state_types)；V2 传入 batch->req
    映射，V1 的暂存数组本身按 batch 排列、HAS_IDX_MAPPING=False。
    """
    batch_idx = tl.program_id(0)  # 当前程序处理的请求/batch 索引
    state_idx = tl.program_id(1)  # 当前程序处理的状态索引（展平的 层×状态类型）
    if batch_idx >= num_reqs:  # 边界检查：网格可能大于实际批大小
        return  # 直接退出
    if HAS_IDX_MAPPING:  # V2：batch 行 -> req 槽位
        req_idx = tl.load(idx_mapping_ptr + batch_idx)  # 解析请求槽位
        if req_idx < 0:  # 无效槽位（该 batch 行无请求）
            return  # 跳过
    else:  # V1：batch 顺序即请求顺序
        req_idx = batch_idx  # 直接使用 batch 索引

    src_col = tl.load(src_col_ptr + req_idx)  # 源块列（前移前的运行块）
    dst_col = tl.load(mamba_state_idx_ptr + req_idx)  # 目标块列（前移后的运行块）
    # Fresh state, or still writing the same block: kernels locate the initial
    # state in-block via num_accepted (preserved when no boundary is crossed),
    # so there is nothing to copy.
    # 全新状态或仍在同一块内写入时：kernel 通过 num_accepted 在块内定位
    # 初始状态（未跨界时该值被保留），因此无需拷贝
    if src_col < 0 or src_col == dst_col:  # 无效源列或源/目标同块
        return  # 无需拷贝，直接退出

    token_bias = tl.load(token_bias_ptr + req_idx)  # 接受偏移（重置前的 num_accepted - 1）
    _copy_mamba_state_block(  # 调用共享拷贝体执行实际拷贝
        state_idx,  # 状态索引
        batch_idx,  # 块表行索引（块表按 batch 排列）
        src_col,  # 源块列
        dst_col,  # 目标块列
        token_bias,  # token 偏移
        block_table_ptrs_ptr,  # 组块表基地址数组
        block_table_stride_req,  # 块表行步长
        state_base_addrs_ptr,  # 状态基地址数组
        state_block_strides_ptr,  # 每块字节数数组
        state_elem_sizes_ptr,  # 元素大小数组
        state_inner_sizes_ptr,  # 内层元素数数组
        state_conv_widths_ptr,  # conv 宽度数组
        state_group_indices_ptr,  # 组索引映射数组
        state_dim_row_count_ptr,  # DS conv 行数数组
        state_dim_row_stride_ptr,  # DS conv 行步长数组
        COPY_BLOCK_SIZE,  # 拷贝分块大小
        CONV_STATE_DIM_FIRST,  # conv 布局标志
    )


@triton.jit  # Triton JIT 编译装饰器（GPU 设备函数）
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):  # 批量异步内存拷贝 kernel：每个 program 处理一段拷贝任务
    pid = tl.program_id(0)  # 当前 program 的拷贝任务编号（对应 grid 中的一项）

    src_ptr = tl.load(src_ptrs + pid)  # 本任务的源地址（int64 字节地址，由 CPU 端预填充）
    dst_ptr = tl.load(dst_ptrs + pid)  # 本任务的目标地址
    size = tl.load(sizes + pid)  # 本任务需拷贝的字节数

    offsets = tl.arange(0, BLOCK_SIZE)  # 拷贝循环内的偏移向量（BLOCK_SIZE 为编译期常量）
    for i in range(0, size, BLOCK_SIZE):  # 按块循环，覆盖整个拷贝区间
        mask = (i + offsets) < size  # 尾块边界掩码（最后一块可能不满）

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))  # 源字节指针（逐字节寻址）
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))  # 目标字节指针

        data = tl.load(curr_src_ptr, mask=mask)  # 按掩码读取源数据
        tl.store(curr_dst_ptr, data, mask=mask)  # 按掩码写入目标


def batch_memcpy(src_ptrs, dst_ptrs, sizes):  # 批量拷贝的 Python 启动封装（设备端版本）
    batch = src_ptrs.shape[0]  # 拷贝任务数量
    assert dst_ptrs.shape[0] == batch  # 三个数组长度必须一致
    assert sizes.shape[0] == batch  # 否则任务-地址-大小无法一一对应

    grid = (batch,)  # 每个任务一个 program
    BLOCK_SIZE = 1024  # 每次迭代拷贝 1024 字节（经验调优值）
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)  # 启动 kernel；注意 size 为运行期值，非 constexpr


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:  # 从 KV 缓存配置中筛选出所有 mamba 组
    mamba_group_ids: list[int] = []  # mamba 组在 kv_cache_groups 中的索引列表
    mamba_specs: list[MambaSpec] = []  # 对应组的 MambaSpec（块大小、层名等）
    for i in range(len(kv_cache_config.kv_cache_groups)):  # 遍历所有 KV 缓存组
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec  # 该组的缓存规格
        if isinstance(kv_cache_spec, MambaSpec):  # 仅收集 mamba 类型的组
            mamba_group_ids.append(i)  # 记录组索引
            mamba_specs.append(kv_cache_spec)  # 记录组规格
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"  # 模型必须包含至少一个 mamba 层，否则不应调用本函数
    assert all(mamba_specs[0] == spec for spec in mamba_specs)  # 所有多 mamba 组规格必须一致（同块大小/布局），否则拷贝元数据无法共享
    return mamba_group_ids, mamba_specs[0]  # 返回组索引列表与第一组的规格（各组规格相同）


@dataclasses.dataclass  # 数据类装饰器：自动生成 __init__/__repr__ 等
class MambaCopyBuffers:  # 非融合路径的批量拷贝缓冲：CPU 预填充指针/大小数组，GPU 端 kernel 消费
    src_ptrs: CpuGpuBuffer  # 源地址数组（uint64，CPU+GPU 镜像）
    dst_ptrs: CpuGpuBuffer  # 目标地址数组（uint64）
    sizes: CpuGpuBuffer  # 每项拷贝字节数数组（int32）
    mamba_group_ids: list[int]  # 拷贝涉及的 mamba 组索引
    mamba_spec: MambaSpec  # mamba 规格（决定每请求条目数）
    offset: int = 0  # 当前已填充的条目数游标（每次构建批后归零）

    @classmethod
    def create(  # 工厂方法：按最大请求数预分配固定容量缓冲
        cls,
        max_num_reqs: int,  # 最大并发请求数（决定缓冲容量）
        kv_cache_config: KVCacheConfig,  # KV 缓存配置（用于解析 mamba 组）
        copy_funcs: tuple[MambaStateCopyFunc, ...],  # 每请求需执行的拷贝函数集合（每个函数生成一组 src/dst/size）
        make_buffer: Callable[..., CpuGpuBuffer],  # 缓冲工厂（统一走 CPU+GPU 镜像分配）
    ) -> "MambaCopyBuffers":
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)  # 解析 mamba 组索引与规格
        entries_per_req = sum(  # 每请求条目数 = Σ(每组层数) × 拷贝函数数
            len(kv_cache_config.kv_cache_groups[gid].layer_names)  # 该组的 mamba 层数
            for gid in mamba_group_ids  # 遍历各 mamba 组
        ) * len(copy_funcs)  # 再乘拷贝函数个数
        n = max_num_reqs * entries_per_req  # 总容量上限（按最坏情况全批请求预分配）

        return cls(  # 实例化：指针用 uint64（可容纳 64 位地址），大小用 int32
            src_ptrs=make_buffer(n, dtype=torch.uint64),  # 源地址缓冲
            dst_ptrs=make_buffer(n, dtype=torch.uint64),  # 目标地址缓冲
            sizes=make_buffer(n, dtype=torch.int32),  # 拷贝大小缓冲（单次拷贝不超过 2GB）
            mamba_group_ids=mamba_group_ids,  # mamba 组索引
            mamba_spec=mamba_spec,  # mamba 规格
        )


@dataclasses.dataclass  # 数据类装饰器：自动生成 __init__/__repr__ 等
class MambaSpecDecodeGPUContext:  # 融合后处理路径的 GPU 端 mamba 状态拷贝上下文（spec decode 专用）
    """
    Context for GPU-side Mamba state copy operations during the
    fused postprocess path.

    Only used when speculative decoding is enabled on a hybrid model
    (and the mamba_cache_config is in align mode).

    Precomputes memory layout metadata (base addresses, strides, element sizes)
    so the GPU kernel can perform state copies without CPU-GPU sync.

    State types are distinguished by conv_width: >0 for conv states (sliding
    window with offset-based copies), 0 for temporal states (full block copies).

    融合后处理路径中 GPU 端 mamba 状态拷贝的上下文。
    仅在混合模型启用 speculative decoding（且 mamba 缓存为 align 模式）时使用。
    预先计算内存布局元数据（基地址/步长/元素大小），使 GPU kernel
    能在无 CPU-GPU 同步的情况下完成状态拷贝。
    状态类型以 conv_width 区分：>0 为 conv 状态（滑窗、按偏移拷贝），
    0 为 temporal 状态（整块拷贝）。
    """

    # Per-state metadata tensors (shape: [num_layers * num_state_types])
    # These are populated from forward_context during the first forward pass
    # 每状态元数据张量（形状 [层数 × 状态类型数]），
    # 首次前向时从 forward_context 填充
    state_base_addrs: torch.Tensor  # int64: base address of each state tensor（各状态张量的基地址）
    state_block_strides: torch.Tensor  # int64: bytes per block（各状态每块字节数，即页步长）
    state_elem_sizes: torch.Tensor  # int32: element size in bytes（元素字节大小，如 fp16 为 2）
    state_inner_sizes: torch.Tensor  # int64: elements in inner dimensions（内层维度元素数，conv 每滑窗位置的元素数）
    state_conv_widths: torch.Tensor  # int32: conv width (0 for temporal states)（conv 窗口宽度；temporal 为 0）
    state_group_indices: torch.Tensor  # int32: maps state_idx to group index（state_idx -> 组索引映射）
    # DS conv row metadata. Zero keeps the single-region copy path.
    # DS conv 行元数据；为 0 时 kernel 走单区域拷贝路径
    state_dim_row_count: torch.Tensor  # int32: per-block dim row count（DS conv 每块 dim 行数）
    state_dim_row_stride: torch.Tensor  # int64: bytes between rows（DS conv 行间字节数）

    # Configuration
    # 配置项（模型初始化后固定）
    block_size: int  # mamba 块大小（与 spec decode 的 token 对齐相关）
    num_layers: int  # 所有 mamba 组的总层数
    num_state_types: int  # 每层的状态类型数（如 conv + temporal = 2）
    mamba_group_ids: list[int]  # mamba 组在 kv_cache_groups 中的索引列表
    num_groups: int  # mamba 组数量

    # Output buffer for num_accepted_tokens updates
    # 输出缓冲：kernel 在 src==dst 时重置接受计数（避免读改写竞争）
    num_accepted_tokens_out: torch.Tensor

    # Per-group block-table base addresses: int64[num_groups]. Populated in
    # initialize_from_forward_context from the persistent per-group block
    # table tensors (whose data_ptr is stable across steps).
    # 各组块表基地址：int64[组数]。在 initialize_from_forward_context 中从
    # 持久化组块表张量填充（其 data_ptr 跨步骤稳定，可安全缓存）
    block_table_ptrs: torch.Tensor
    block_table_stride_req: int = 0  # 块表请求行步长（元素数）

    # Per-request staging buffers (CPU+GPU mirrors). The runner stages
    # values into the CPU view in ``_prepare_inputs`` and the fused kernel
    # reads the GPU side. These only exist when the postprocess kernel is
    # enabled (spec decode + hybrid + align mode).
    # 每请求暂存缓冲（CPU+GPU 镜像）。runner 在 _prepare_inputs 中写入
    # CPU 视图，融合 kernel 读取 GPU 侧。仅在启用后处理 kernel
    # （spec decode + 混合模型 + align 模式）时存在。
    mamba_state_idx_buf: CpuGpuBuffer | None = None  # 每请求当前状态块列
    num_scheduled_tokens_buf: CpuGpuBuffer | None = None  # 每请求本轮调度 token 数（V1 决策用）
    num_computed_tokens_buf: CpuGpuBuffer | None = None  # 每请求已计算 token 数
    num_draft_tokens_buf: CpuGpuBuffer | None = None  # 每请求草稿 token 数（V1 决策用）
    precopy_src_col_buf: CpuGpuBuffer | None = None  # 预拷贝源块列（align precopy 用）
    precopy_token_bias_buf: CpuGpuBuffer | None = None  # 预拷贝接受偏移（align precopy 用）

    # Flag to track if metadata has been populated
    # 元数据是否已填充的标志（保证 initialize 只执行一次）
    is_initialized: bool = False

    @classmethod
    def create(  # 工厂方法：分配全部缓冲，元数据留待首次前向填充
        cls,
        max_num_reqs: int,  # 最大并发请求数（决定每请求缓冲容量）
        kv_cache_config: KVCacheConfig,  # KV 缓存配置（用于解析 mamba 组与层数）
        num_state_types: int,  # 每层状态类型数
        device: torch.device,  # 元数据张量所在设备（须与 kernel 同设备）
        make_buffer: Callable[..., CpuGpuBuffer],  # 缓冲工厂（CPU+GPU 镜像分配）
    ) -> "MambaSpecDecodeGPUContext":
        """Create context with allocated buffers (metadata populated later)."""
        # 创建上下文并分配缓冲；元数据延迟到首次前向时填充
        mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)  # 解析 mamba 组索引与规格

        # Count total layers across all mamba groups
        # 统计所有 mamba 组的总层数
        num_layers = sum(  # Σ 每组层数
            len(kv_cache_config.kv_cache_groups[gid].layer_names)  # 该组的 mamba 层数
            for gid in mamba_group_ids  # 遍历各 mamba 组
        )
        total_states = num_layers * num_state_types  # 展平状态槽位总数（kernel grid 第二维的大小）

        return cls(  # 全部元数据先置零、首前向时覆盖；维度/dtype 与 kernel 读取约定一致
            state_base_addrs=torch.zeros(  # 各状态基地址（int64 足以容纳 64 位地址）
                total_states, dtype=torch.int64, device=device
            ),
            state_block_strides=torch.zeros(  # 各状态每块页步长（字节，可能超 int32 上限）
                total_states, dtype=torch.int64, device=device
            ),
            state_elem_sizes=torch.zeros(  # 元素字节大小（小整数，int32 足够）
                total_states, dtype=torch.int32, device=device
            ),
            state_inner_sizes=torch.zeros(  # 内层维度元素数（可能较大，int64）
                total_states, dtype=torch.int64, device=device
            ),
            state_conv_widths=torch.zeros(  # conv 宽度（0 表示 temporal）
                total_states, dtype=torch.int32, device=device
            ),
            state_group_indices=torch.zeros(  # state_idx -> 组索引映射
                total_states, dtype=torch.int32, device=device
            ),
            state_dim_row_count=torch.zeros(  # DS conv 每块 dim 行数（0 = 非 DS 布局）
                total_states, dtype=torch.int32, device=device
            ),
            state_dim_row_stride=torch.zeros(  # DS conv 行间字节数
                total_states, dtype=torch.int64, device=device
            ),
            block_size=mamba_spec.block_size,  # mamba 块大小（来自规格）
            num_layers=num_layers,  # 总层数
            num_state_types=num_state_types,  # 状态类型数
            mamba_group_ids=mamba_group_ids,  # mamba 组索引列表
            num_groups=len(mamba_group_ids),  # 组数量
            num_accepted_tokens_out=torch.zeros(  # 接受计数输出缓冲（每请求一个槽位）
                max_num_reqs, dtype=torch.int32, device=device
            ),
            block_table_ptrs=torch.zeros(  # 各组块表基地址（首前向时从持久化块表填充）
                len(mamba_group_ids), dtype=torch.int64, device=device
            ),
            mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),  # 暂存：状态块列
            num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),  # 暂存：调度 token 数
            num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),  # 暂存：已计算 token 数
            num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),  # 暂存：草稿 token 数
            precopy_src_col_buf=make_buffer(max_num_reqs, dtype=torch.int32),  # 暂存：预拷贝源列
            precopy_token_bias_buf=make_buffer(max_num_reqs, dtype=torch.int32),  # 暂存：预拷贝偏移
            is_initialized=False,  # 元数据尚未填充
        )

    def initialize_from_forward_context(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],  # 层名 -> attention 对象的映射（模型加载后填充）
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],  # 拷贝函数元组（每状态类型一个），用于判别 conv/temporal
        block_tables: list[torch.Tensor],  # 各 mamba 组的持久化块表张量（顺序与 mamba_group_ids 一致）
    ) -> None:
        """
        Extract and cache memory layout metadata from Mamba state tensors.

        This method populates the pre-allocated metadata tensors with information
        needed by `postprocess_mamba_fused_kernel` to perform state copies entirely
        on the GPU without CPU-GPU synchronization.

        For each Mamba layer and state type, the following metadata is extracted:
        - state_base_addrs: GPU memory address (data_ptr) of the state tensor
        - state_block_strides: Bytes between consecutive blocks (stride * elem_size)
        - state_elem_sizes: Element size in bytes (e.g., 2 for float16)
        - state_inner_sizes: For conv states, elements per conv position (stride(1)),
          used to compute offset when slicing state[block, offset:]. For temporal
          states, this field is unused (set to 1).
        - state_conv_widths: Conv dimension size for conv states, 0 for temporal states

        The conv vs temporal state type is detected by inspecting the copy function
        name: functions containing "conv" are treated as conv states.

        This method is idempotent - it only executes once (guarded by is_initialized
        flag) since the metadata is static after model loading.

        Args:
            kv_cache_config: Configuration containing KV cache group info and
                layer name mappings.
            forward_context: Dictionary mapping layer names to attention objects,
                populated after the model is loaded. Each attention object must
                have a `kv_cache` attribute containing the list of state tensors.
            mamba_state_copy_funcs: Tuple of copy functions (one per state type)
                used to determine whether each state is a conv or temporal state.
            block_tables: per-mamba-group persistent block-table tensors, in
                the same order as `mamba_group_ids`. Their `data_ptr()` /
                `stride(0)` are captured once for the kernel to index into.

        从 mamba 状态张量中提取并缓存内存布局元数据。
        填充 create() 预分配的元数据张量，供融合 kernel 在 GPU 端
        无 CPU-GPU 同步地完成状态拷贝。每 (层, 状态类型) 提取：
        基地址（data_ptr）、块步长（stride × 元素大小）、元素字节大小、
        内层元素数（conv 为每滑窗位置元素数；temporal 为 1/自然块大小）、
        conv 宽度（temporal 为 0）。
        conv/temporal 的判别依据拷贝函数是否为 get_conv_copy_spec。
        本方法幂等（is_initialized 保护）：模型加载后元数据静态不变，仅执行一次。
        """
        if self.is_initialized:  # 已初始化：元数据静态，直接返回（幂等保护）
            return

        idx = 0  # 展平状态槽位游标：按 (组, 层, 状态类型) 顺序递增
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):  # 遍历各 mamba 组
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names  # 该组包含的 mamba 层名
            for layer_name in layer_names:  # 遍历组内各层
                attention = forward_context[layer_name]  # 该层的 attention/前向上下文对象
                kv_caches: list[torch.Tensor] = attention.kv_cache  # 该层的状态张量列表（conv、temporal 等）

                for state_type_idx, state in enumerate(kv_caches):  # 遍历该层各状态类型
                    # Base address
                    # 基地址：状态张量的 GPU 起始地址（生命周期须覆盖引擎全程）
                    self.state_base_addrs[idx] = state.data_ptr()

                    # Block stride (bytes between consecutive blocks)
                    # state shape: [num_blocks, ...], stride(0) = elements per block
                    # 块步长（相邻块之间的字节数）：state 形状为 [num_blocks, ...]，
                    # stride(0) 即每块元素数
                    if state.dim() > 1:  # 多维张量：直接取 stride(0)
                        block_stride_elems = state.stride(0)
                    else:  # 一维退化情况：整张量即一个块
                        block_stride_elems = state.numel()
                    self.state_block_strides[idx] = (
                        block_stride_elems * state.element_size()  # 元素步长 × 元素字节 = 字节步长
                    )

                    # Element size
                    # 元素字节大小（如 fp16 为 2）
                    self.state_elem_sizes[idx] = state.element_size()

                    copy_func = mamba_state_copy_funcs[state_type_idx]  # 该状态类型对应的拷贝函数
                    assert (
                        copy_func is get_conv_copy_spec
                        or copy_func is get_temporal_copy_spec
                    ), f"unexpected copy func: {copy_func}"  # 仅支持这两种拷贝规格，其余拒绝
                    if copy_func is get_conv_copy_spec:  # conv（滑窗）状态
                        if state.dim() != 3:  # conv 状态必须是 3 维 [块, ..., 滑窗]
                            raise ValueError(
                                "Expected 3D conv state cache, got "
                                f"shape {tuple(state.shape)}"
                            )
                        if is_conv_state_dim_first():  # DS（dim-first）布局：滑动轴在最后一维
                            # DS layout: state_len is the slide axis.
                            self.state_conv_widths[idx] = state.size(2)  # 滑窗宽度 = 最后一维大小
                            self.state_inner_sizes[idx] = 1  # DS 布局不用 inner_size，置 1
                            self.state_dim_row_count[idx] = state.size(1)  # dim 行数 = 第二维大小
                            self.state_dim_row_stride[idx] = (
                                state.stride(1) * state.element_size()  # 行间字节数
                            )
                        else:  # SD 布局：dim 维连续、滑动维在 dim 之后
                            # SD layout: dim is contiguous.
                            self.state_conv_widths[idx] = state.size(1)  # 滑窗宽度 = 第二维大小
                            self.state_inner_sizes[idx] = state.stride(1)  # 每滑窗位置元素数 = stride(1)
                    else:  # temporal 状态
                        # Temporal state: inner_size = natural elements per
                        # block (prod of inner dims).  The kernel uses this
                        # to compute copy_size = inner_size * elem_size,
                        # which gives the correct byte count even when the
                        # state tensor is as_strided with padded page strides
                        # (state_block_stride would be the page size, too big).
                        # temporal：inner_size = 每块自然元素数（内层各维之积）。
                        # kernel 用 copy_size = inner_size × elem_size 计算
                        # 实际拷贝字节数——即使状态张量经 as_strided 页填充
                        # （页步长偏大），也能得到正确的大小
                        self.state_conv_widths[idx] = 0  # 宽度 0 标记 temporal
                        self.state_inner_sizes[idx] = (
                            state[0].numel() if state.dim() > 1 else 1  # 单块自然元素数
                        )
                        # Temporal copies are vectorized with uint64
                        # loads/stores; base pointer and block stride must
                        # be 8B-aligned (tail loop handles copy_size % 8).
                        # temporal 拷贝用 uint64 向量化读写：基地址与块步长
                        # 必须 8 字节对齐（copy_size 非 8 倍数由尾部循环兜底）
                        base_addr = state.data_ptr()  # 状态基地址
                        block_stride_bytes = block_stride_elems * state.element_size()  # 块步长字节数
                        assert base_addr % 8 == 0, (  # 对齐校验：失败说明分配器行为异常
                            f"layer {layer_name}: state.data_ptr() = "
                            f"{base_addr:#x} is not 8B-aligned; "
                            f"_copy_mamba_state_block uint64 "
                            f"vectorization requires it"
                        )
                        assert block_stride_bytes % 8 == 0, (  # 块步长也须 8 字节对齐
                            f"layer {layer_name}: block stride = "
                            f"{block_stride_bytes}B is not 8B-aligned; "
                            f"_copy_mamba_state_block uint64 "
                            f"vectorization requires it"
                        )

                    self.state_group_indices[idx] = group_local_idx  # 记录该状态所属组（局部索引）
                    idx += 1  # 前进到下一个状态槽位

        # Cache per-group block-table base addresses and per-request stride.
        # `block_tables[i]` is the persistent 2D int32 block-table tensor for
        # `mamba_group_ids[i]`; `data_ptr()` / `stride(0)` are stable for the
        # engine's lifetime, so we capture them once here.
        # 缓存各组块表基地址与请求行步长。block_tables[i] 是第 i 组的
        # 持久化 2D int32 块表张量；其 data_ptr()/stride(0) 在引擎生命周期
        # 内稳定，故只需在此捕获一次
        assert len(block_tables) == self.num_groups, (  # 块表数量须与 mamba 组数一致
            f"expected {self.num_groups} block tables, got {len(block_tables)}"
        )
        strides = {bt.stride(0) for bt in block_tables}  # 收集各组块表的行步长
        assert len(strides) == 1, (  # 所有组必须共享同一行步长（kernel 只传一个标量）
            f"all mamba block tables must share stride(0), got {strides}"
        )
        self.block_table_stride_req = int(next(iter(strides)))  # 记录公共行步长
        for i, bt in enumerate(block_tables):  # 逐组捕获块表基地址
            self.block_table_ptrs[i] = bt.data_ptr()  # 组 i 的块表起始地址

        self.is_initialized = True  # 标记初始化完成（此后元数据不再变化）

    def run_fused_postprocess(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,  # [num_reqs] 每请求被接受的 token 数（GPU 决策数组）
        mamba_state_idx_gpu: torch.Tensor,  # [num_reqs] 每请求源状态块列（上一步运行块）
        num_scheduled_tokens_gpu: torch.Tensor,  # [num_reqs] 本轮调度 token 数（V1 推导用）
        num_computed_tokens_gpu: torch.Tensor,  # [num_reqs] 已计算 token 数（V1 推导用）
        num_draft_tokens_gpu: torch.Tensor,  # [num_reqs] 草稿 token 数（V1 推导用）
    ) -> None:
        """
        Run the fused postprocess_mamba kernel on GPU.

        This computes decisions and performs mamba state copies entirely on GPU,
        eliminating the CPU-GPU sync that was previously needed.

        Args:
            num_reqs: Number of active requests
            num_accepted_tokens_gpu: [num_reqs] accepted token counts
            mamba_state_idx_gpu: [num_reqs] source block indices
            num_scheduled_tokens_gpu: [num_reqs] scheduled token counts
            num_computed_tokens_gpu: [num_reqs] computed token counts
            num_draft_tokens_gpu: [num_reqs] draft token counts

        在 GPU 上运行融合后处理 kernel。
        决策计算与状态拷贝全部在 GPU 完成，消除原实现所需的 CPU-GPU 同步。
        输入为 V1 风格的每请求数组（按 req 顺序排列，HAS_IDX_MAPPING=False）。
        """
        if num_reqs == 0 or not self.is_initialized:  # 空批或元数据未就绪：直接返回
            return

        # Initialize output to current values (unchanged unless src==dst)
        # 输出缓冲初始化为当前值：仅 src==dst 时 kernel 才会改写（重置为 1）
        self.num_accepted_tokens_out[:num_reqs].copy_(
            num_accepted_tokens_gpu[:num_reqs]
        )

        total_states = self.num_layers * self.num_state_types  # 展平状态槽位总数（grid 第二维）
        grid = (num_reqs, total_states)  # 网格：每 (请求, 状态) 一个 program

        postprocess_mamba_fused_kernel[grid](  # 启动融合后处理 kernel（V1 路径）
            num_accepted_tokens_gpu,  # 决策：接受 token 数
            mamba_state_idx_gpu,  # 决策：源块列
            num_scheduled_tokens_gpu,  # 决策：调度 token 数（V1）
            num_computed_tokens_gpu,  # 决策：已计算 token 数（V1）
            num_draft_tokens_gpu,  # 决策：草稿 token 数（V1）
            self.block_table_ptrs,  # 组块表基地址数组
            self.block_table_stride_req,  # 块表行步长
            self.state_base_addrs,  # 状态基地址元数据
            self.state_block_strides,  # 块步长元数据
            self.state_elem_sizes,  # 元素大小元数据
            self.state_inner_sizes,  # 内层元素数元数据
            self.state_conv_widths,  # conv 宽度元数据
            self.state_group_indices,  # 组索引映射元数据
            self.state_dim_row_count,  # DS conv 行数元数据
            self.state_dim_row_stride,  # DS conv 行步长元数据
            self.num_accepted_tokens_out,  # 接受计数输出缓冲
            None,  # idx_mapping: V1 decision arrays are already in req order（V1 无需映射）
            num_reqs,  # 活跃请求数
            block_size=self.block_size,  # mamba 块大小（编译期常量）
            COPY_BLOCK_SIZE=1024,  # 拷贝分块大小（调优参数）
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),  # conv 布局标志（编译期常量）
        )

    def run_fused_precopy(
        self,
        num_reqs: int,  # 活跃请求数（batch 顺序）
        state_idx_gpu: torch.Tensor,  # [max_reqs] 前移后的目标块列（按 req 槽位）
        src_col_gpu: torch.Tensor,  # [max_reqs] 前移前的源块列（-1 = 全新状态）
        token_bias_gpu: torch.Tensor,  # [max_reqs] 接受 token 偏移（num_accepted - 1）
        idx_mapping: torch.Tensor | None,  # 可选 [num_reqs] batch -> req 槽位映射（None 表示 V1 顺序一致）
    ) -> None:
        """Pre-copy each request's previous running block into its new window
        block before the forward pass (align boundary migration).

        Args:
            num_reqs: Number of active requests (batch order).
            state_idx_gpu: [max_reqs] post-advance dst block column per req slot.
            src_col_gpu: [max_reqs] pre-advance src block column (-1 = fresh).
            token_bias_gpu: [max_reqs] accepted-token bias (num_accepted - 1).
            idx_mapping: optional [num_reqs] batch_idx -> req_state_idx.
                None means V1 batch order already equals request state order.

        前向推理前，将各请求上一个运行块的状态预拷贝到新窗口块
        （align 跨界迁移）。输入数组由 GPU 端 align 预处理 kernel 产出，
        无需 CPU-GPU 同步。
        """
        if num_reqs == 0 or not self.is_initialized:  # 空批或未初始化：直接返回
            return
        total_states = self.num_layers * self.num_state_types  # 展平状态槽位总数
        grid = (num_reqs, total_states)  # 网格：每 (请求, 状态) 一个 program
        precopy_mamba_align_fused_kernel[grid](  # 启动 align 预拷贝 kernel
            state_idx_gpu,  # 目标块列（前移后）
            src_col_gpu,  # 源块列（前移前，-1 = 新请求）
            token_bias_gpu,  # 接受偏移
            self.block_table_ptrs,  # 组块表基地址数组
            self.block_table_stride_req,  # 块表行步长
            self.state_base_addrs,  # 状态基地址元数据
            self.state_block_strides,  # 块步长元数据
            self.state_elem_sizes,  # 元素大小元数据
            self.state_inner_sizes,  # 内层元素数元数据
            self.state_conv_widths,  # conv 宽度元数据
            self.state_group_indices,  # 组索引映射元数据
            self.state_dim_row_count,  # DS conv 行数元数据
            self.state_dim_row_stride,  # DS conv 行步长元数据
            idx_mapping,  # batch -> req 槽位映射（可为 None）
            num_reqs,  # 活跃请求数
            COPY_BLOCK_SIZE=1024,  # 拷贝分块大小（调优参数）
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),  # conv 布局标志
            HAS_IDX_MAPPING=idx_mapping is not None,  # 有映射则为 V2 模式
        )

    def run_fused_postprocess_align(
        self,
        num_reqs: int,  # 活跃请求数（batch 顺序）
        num_accepted_tokens_gpu: torch.Tensor,  # [num_reqs] 接受 token 数（kernel 内原地把跨界者重置为 1）
        state_idx_gpu: torch.Tensor,  # [num_reqs] 源状态块列（req 槽位序，经 idx_mapping 解析）
        new_num_computed_tokens_gpu: torch.Tensor,  # [num_reqs] 本步之后的新计算数（V2 已前移）
        idx_mapping: torch.Tensor,  # [num_reqs] batch 行 -> req 状态槽位映射（V2/PP）
    ) -> None:
        """V2 align postprocess: save the running state to the block-aligned
        position after spec-decode acceptance leaves the sequence non-aligned.

        ``num_accepted_tokens_gpu`` is updated in place while the kernel reads
        from a snapshot to avoid cross-program races when the accepted position
        stays in the running block and the count is reset to 1.
        ``new_num_computed_tokens`` already holds the post-step computed count
        (PRECOMPUTED_NEW_COMPUTED).
        ``idx_mapping`` maps batch row -> req-state slot (HAS_IDX_MAPPING).

        V2 align 后处理：spec-decode 接受后序列不再块对齐，
        将运行状态保存到块对齐位置。
        num_accepted_tokens_gpu 在 kernel 内原地更新；kernel 从快照读取
        以避免跨 program 竞争（接受位置留在运行块内且计数被重置为 1 时）。
        new_num_computed_tokens 已是本步后的计算数（PRECOMPUTED_NEW_COMPUTED=True）。
        idx_mapping 将 batch 行映射到 req 状态槽位（HAS_IDX_MAPPING=True）。
        """
        if num_reqs == 0 or not self.is_initialized:  # 空批或未初始化：直接返回
            return

        # V2 reads non-contiguous idx_mapping positions, so snapshot the whole
        # decision buffer rather than only [:num_reqs].
        # V2 经 idx_mapping 非连续读取，故对整个决策缓冲做快照
        # （而非仅前 num_reqs 项），保证 kernel 读到完整数据
        num_accepted_tokens_snapshot = self.num_accepted_tokens_out  # 复用输出缓冲作为快照（尺寸为 max_reqs）
        num_accepted_tokens_snapshot.copy_(num_accepted_tokens_gpu)  # 快照当前接受计数（GPU-GPU 拷贝）

        total_states = self.num_layers * self.num_state_types  # 展平状态槽位总数
        grid = (num_reqs, total_states)  # 网格：每 (请求, 状态) 一个 program
        postprocess_mamba_fused_kernel[grid](  # 启动融合后处理 kernel（V2 align 变体）
            num_accepted_tokens_snapshot,  # 决策输入：接受计数快照（只读）
            state_idx_gpu,  # 决策输入：源块列
            None,  # num_scheduled: unused under PRECOMPUTED_NEW_COMPUTED（V2 不用）
            new_num_computed_tokens_gpu,  # 决策输入：新计算数（已前移）
            None,  # num_draft: unused under PRECOMPUTED_NEW_COMPUTED（V2 不用）
            self.block_table_ptrs,  # 组块表基地址数组
            self.block_table_stride_req,  # 块表行步长
            self.state_base_addrs,  # 状态基地址元数据
            self.state_block_strides,  # 块步长元数据
            self.state_elem_sizes,  # 元素大小元数据
            self.state_inner_sizes,  # 内层元素数元数据
            self.state_conv_widths,  # conv 宽度元数据
            self.state_group_indices,  # 组索引映射元数据
            self.state_dim_row_count,  # DS conv 行数元数据
            self.state_dim_row_stride,  # DS conv 行步长元数据
            num_accepted_tokens_gpu,  # 输出：原接受计数缓冲（src==dst 时被重置为 1）
            idx_mapping,  # batch -> req 槽位映射
            num_reqs,  # 活跃请求数
            block_size=self.block_size,  # mamba 块大小（编译期常量）
            COPY_BLOCK_SIZE=1024,  # 拷贝分块大小（调优参数）
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),  # conv 布局标志
            HAS_IDX_MAPPING=True,  # V2：需要 idx_mapping
            PRECOMPUTED_NEW_COMPUTED=True,  # V2：已预计算新计算数
        )


@dataclasses.dataclass  # 数据类装饰器：自动生成 __init__/__repr__ 等
class MambaBuffers:  # runner 所有 mamba 专用缓冲的唯一持有者
    """Single owner for all mamba-specific runner buffers.

    The two sub-objects have different gates:
    ``preprocess`` is needed whenever ``mamba_cache_mode == "align"``;
    ``postprocess_align`` is needed only when align is combined with
    speculative decoding on a hybrid model, and is ``None`` otherwise.

    runner 所有 mamba 专用缓冲的唯一持有者。
    两个子对象的启用条件不同：
    ``preprocess`` 在 mamba_cache_mode == "align" 时始终需要；
    ``postprocess_align`` 仅在 align 与混合模型 spec decode 叠加时需要，
    否则为 None。
    """

    preprocess: MambaCopyBuffers  # align 预处理拷贝缓冲（非融合路径）
    postprocess_align: MambaSpecDecodeGPUContext | None  # spec decode 融合后处理上下文（可选）

    @classmethod
    def create(  # 工厂方法：按需构建两套子缓冲
        cls,
        max_num_reqs: int,  # 最大并发请求数（缓冲容量依据）
        kv_cache_config: KVCacheConfig,  # KV 缓存配置（解析 mamba 组）
        copy_funcs: tuple[MambaStateCopyFunc, ...],  # 拷贝函数集合（决定每请求条目数与状态类型数）
        make_buffer: Callable[..., CpuGpuBuffer],  # CPU+GPU 镜像缓冲工厂
        device: torch.device,  # 融合上下文元数据的设备
        with_postprocess_align: bool,  # 是否启用融合后处理（align + spec decode）
    ) -> "MambaBuffers":
        return cls(  # 组装：预处理缓冲总是创建；融合上下文按需创建
            preprocess=MambaCopyBuffers.create(  # 预处理拷贝缓冲
                max_num_reqs, kv_cache_config, copy_funcs, make_buffer
            ),
            postprocess_align=(  # 融合后处理上下文（可选）
                MambaSpecDecodeGPUContext.create(  # 仅 align + spec decode 时创建
                    max_num_reqs=max_num_reqs,  # 最大请求数
                    kv_cache_config=kv_cache_config,  # KV 缓存配置
                    num_state_types=len(copy_funcs),  # 状态类型数 = 拷贝函数个数
                    device=device,  # 元数据设备
                    make_buffer=make_buffer,  # 缓冲工厂
                )
                if with_postprocess_align
                else None  # 未启用则置空
            ),
        )


def collect_mamba_copy_meta(  # 非融合路径：为单个请求收集 mamba 状态拷贝元数据（CPU 端填充）
    copy_bufs: MambaCopyBuffers,  # 拷贝缓冲（CPU 侧 numpy 视图写入）
    kv_cache_config: KVCacheConfig,  # KV 缓存配置（层名解析）
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],  # 拷贝规格函数（每状态类型一个）
    mamba_group_ids: list[int],  # mamba 组索引列表
    src_block_idx: int,  # 源块列
    dest_block_idx: int,  # 目标块列
    accept_token_bias: int,  # 接受 token 偏移（conv 滑窗平移量 - 1）
    req_state: CachedRequestState,  # 请求状态（含各组的物理块 id）
    forward_context: dict[str, Any],  # 层名 -> attention 对象映射
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:  # 源/目标同块且无偏移：无需拷贝
        return

    src_ptrs_np = copy_bufs.src_ptrs.np  # CPU 侧 numpy 视图：源地址数组
    dst_ptrs_np = copy_bufs.dst_ptrs.np  # 目标地址数组
    sizes_np = copy_bufs.sizes.np  # 拷贝字节数数组
    offset = copy_bufs.offset  # 当前填充游标（跨请求累计）

    for mamba_group_id in mamba_group_ids:  # 遍历各 mamba 组
        block_ids = req_state.block_ids[mamba_group_id]  # 该组为本请求分配的物理块 id 列表
        dest_block_id = block_ids[dest_block_idx]  # 目标列对应的物理块 id
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names  # 该组的 mamba 层名
        for layer_name in layer_names:  # 遍历组内各层
            attention = forward_context[layer_name]  # 该层的前向上下文对象
            kv_caches: list[torch.Tensor] = attention.kv_cache  # 该层的状态张量列表
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):  # 逐状态类型
                copy_spec = state_copy_func(  # 由拷贝规格函数计算源地址/元素数（conv 按偏移切片）
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                src_ptrs_np[offset] = copy_spec.start_addr  # 源起始地址（含偏移）
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()  # 目标块起始地址
                sizes_np[offset] = copy_spec.num_elements * state.element_size()  # 拷贝字节数
                offset += 1  # 前进到下一条目

    copy_bufs.offset = offset  # 写回游标（供 do_mamba_copy_block 使用）


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):  # 执行已收集的批量拷贝（非融合路径的收尾）
    n = copy_bufs.offset  # 本次批的拷贝条目数
    if n == 0:  # 无条目：跳过 kernel 启动
        return
    batch_memcpy(  # 启动批量拷贝 kernel（仅上传前 n 项到 GPU）
        copy_bufs.src_ptrs.copy_to_gpu(n),  # 源地址（CPU -> GPU）
        copy_bufs.dst_ptrs.copy_to_gpu(n),  # 目标地址
        copy_bufs.sizes.copy_to_gpu(n),  # 拷贝大小
    )


def cleanup_mamba_state_idx(  # 清理已结束/被抢占/被恢复请求的陈旧状态块列记录
    scheduler_output: SchedulerOutput,  # 调度器输出（含各类请求 id 集合）
    mamba_state_idx: dict[str, int],  # req_id -> 状态块列 的持久化映射
) -> None:
    """Pop stale `mamba_state_idx` entries for finished/preempted/resumed reqs.

    Force-preempted requests (e.g., during reset_prefix_cache / KV cache
    flush) appear in resumed_req_ids without a corresponding entry in
    preempted_req_ids, leaving stale entries that can point to block
    indices beyond the new (smaller) block allocation.

    清理已结束/被抢占/被恢复请求的 mamba_state_idx 条目。
    强制抢占的请求（如 reset_prefix_cache / KV 缓存清空期间）只会出现在
    resumed_req_ids 而不在 preempted_req_ids 中，残留的陈旧条目可能指向
    超出新（更小）块分配范围的块索引，必须移除。
    """
    finished_req_ids = scheduler_output.finished_req_ids  # 本步已完成的请求
    preempted_req_ids = scheduler_output.preempted_req_ids or set()  # 被抢占的请求（可能为 None）
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids  # 被恢复的请求（含强制抢占者）
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):  # 三类请求统一遍历
        mamba_state_idx.pop(req_id, None)  # 移除记录；不存在也不报错


class _FusedPrecopy(NamedTuple):  # 融合预拷贝资源的不可变捆绑（ NamedTuple 轻量结构）
    """Resolved fused align pre-copy resources (all non-None once resolved)."""

    # 已解析的融合 align 预拷贝资源（解析后各成员均非 None）
    ctx: "MambaSpecDecodeGPUContext"  # 融合上下文（含块表/元数据/初始化标志）
    state_idx: CpuGpuBuffer  # 目标块列暂存缓冲
    src_col: CpuGpuBuffer  # 源块列暂存缓冲
    token_bias: CpuGpuBuffer  # 接受偏移暂存缓冲


def _resolve_fused_precopy(  # 将融合路径缓冲打包，标量路径返回 None
    align_ctx: "MambaSpecDecodeGPUContext | None",  # 融合上下文（可能为 None = 标量路径）
) -> _FusedPrecopy | None:
    """Bundle the fused-path buffers, or None for the scalar path.

    Returning one non-None bundle lets callers narrow all four members with a
    single ``is not None`` check instead of re-asserting each buffer per use.

    打包融合路径缓冲；标量路径返回 None。
    返回单一非 None 捆绑后，调用方只需一次 ``is not None`` 判断
    即可收窄全部四个成员，无需逐个断言。
    """
    if align_ctx is None:  # 未启用融合路径
        return None
    assert align_ctx.mamba_state_idx_buf is not None  # 三个暂存缓冲必须已由 create() 分配
    assert align_ctx.precopy_src_col_buf is not None
    assert align_ctx.precopy_token_bias_buf is not None
    return _FusedPrecopy(  # 组装捆绑
        align_ctx,  # 上下文
        align_ctx.mamba_state_idx_buf,  # 目标块列缓冲
        align_ctx.precopy_src_col_buf,  # 源块列缓冲
        align_ctx.precopy_token_bias_buf,  # 接受偏移缓冲
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],  # req_id -> 请求状态映射
    forward_context: dict[str, Any],  # 层名 -> attention 对象映射（元数据初始化用）
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],  # 拷贝规格函数集合
    copy_bufs: MambaCopyBuffers,  # 非融合路径的拷贝缓冲
    align_ctx: MambaSpecDecodeGPUContext | None = None,  # 融合路径上下文（None = 走 CPU 收集路径）
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.

    预处理：将上一步的 mamba 状态拷贝到最后 (1 + num_speculative_blocks)
    个块处（即 curr_state_idx 块列）。存在两条路径：
    融合路径（align_ctx 非 None，GPU 端暂存 + run_fused_precopy）
    与标量路径（CPU 端收集元数据 + batch_memcpy）。
    """
    fused = _resolve_fused_precopy(align_ctx)  # 解析融合路径资源（None 则走标量路径）
    mamba_group_ids = copy_bufs.mamba_group_ids  # mamba 组索引列表
    mamba_spec = copy_bufs.mamba_spec  # mamba 规格
    num_speculative_blocks = mamba_spec.num_speculative_blocks  # speculative 预留块数（决定运行状态存放偏移）
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching  # align 模式依赖前缀缓存（块复用语义）
    block_size = mamba_spec.block_size  # mamba 块大小
    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)  # 先清理陈旧请求的状态记录

    copy_bufs.offset = 0  # 重置标量路径的条目游标（每步重建）
    num_reqs = len(input_batch.req_ids)  # 本批请求数

    if fused is not None:  # 融合路径准备
        if num_reqs == 0:  # 空批：无需预拷贝
            return
        if not fused.ctx.is_initialized:  # 元数据未填充（首次前向）：惰性初始化
            fused.ctx.initialize_from_forward_context(  # 提取状态张量布局元数据并缓存块表指针
                kv_cache_config,  # KV 缓存配置
                forward_context,  # 前向上下文（状态张量来源）
                mamba_state_copy_funcs,  # 拷贝函数（判别 conv/temporal）
                [  # 各 mamba 组的持久化块表设备张量
                    input_batch.block_table[gid].get_device_tensor(num_reqs)
                    for gid in fused.ctx.mamba_group_ids  # 按组索引顺序收集
                ],
            )

        fused.src_col.np[:num_reqs] = -1  # 默认源列置 -1（新请求/同块：无需拷贝）
        fused.token_bias.np[:num_reqs] = 0  # 默认接受偏移置 0

    for i, req_id in enumerate(input_batch.req_ids):  # 逐请求计算状态块列
        req_state = requests[req_id]  # 该请求的缓存状态
        prev_state_idx = mamba_state_idx.get(req_id)  # 上一步的状态块列（可能缺失）
        if prev_state_idx is None:  # 新请求/被恢复的请求：由已计算 token 数推导
            # New / resumed request; num_computed_tokens == 0 gives -1.
            # num_computed_tokens == 0 时得到 -1（表示尚无状态可拷贝）
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]  # 本步调度的 token 数
        num_blocks = (  # 本步之后所需的总块数 = 已覆盖块数 + speculative 预留块数
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )
        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        # 运行状态总是保存在倒数第 (1 + num_speculative_blocks) 个块。
        # 边界示例（block_size=4, num_speculative_tokens=2，请求 [A,B,C] +
        # 2 个草稿 token [draft1, draft2]）：
        #   块 0: [A, B, C, draft1]
        #   块 1: [draft2, TOFILL, TOFILL, TOFILL]
        #   块 2/3: speculative 块
        # 此时用块 1 保存运行状态。
        curr_state_idx = num_blocks - 1 - num_speculative_blocks  # 当前状态块列 = 总块数 - 预留块数 - 1
        mamba_state_idx[req_id] = curr_state_idx  # 记录（供本步后处理与下一步预处理使用）
        if fused is not None:  # 融合路径：写入 CPU 视图
            fused.state_idx.np[i] = curr_state_idx  # 目标块列

        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:  # 有旧状态且块列发生变化：需迁移
            accept_token_bias = int(input_batch.num_accepted_tokens_cpu[i]) - 1  # 接受偏移 = 接受数 - 1
            if fused is not None:  # 融合路径：仅记录源列与偏移，拷贝延后到 GPU
                assert accept_token_bias >= 0  # 仅 spec decode 场景，接受数至少为 1
                fused.src_col.np[i] = prev_state_idx  # 源块列
                fused.token_bias.np[i] = accept_token_bias  # 接受偏移
            else:  # 标量路径：立即在 CPU 收集拷贝元数据
                collect_mamba_copy_meta(  # 为该请求各层各状态填充 src/dst/size 条目
                    copy_bufs,  # 拷贝缓冲
                    kv_cache_config,  # KV 缓存配置
                    mamba_state_copy_funcs,  # 拷贝函数集合
                    mamba_group_ids,  # mamba 组
                    prev_state_idx,  # 源块列
                    curr_state_idx,  # 目标块列
                    accept_token_bias,  # 接受偏移
                    req_state,  # 请求状态（物理块 id）
                    forward_context,  # 前向上下文（状态张量）
                )
            input_batch.num_accepted_tokens_cpu[i] = 1  # 预拷贝后重置接受计数为 1（状态已迁移至新块开头）

    if fused is not None:  # 融合路径收尾：上传暂存数组并启动 GPU 预拷贝
        fused.state_idx.copy_to_gpu(num_reqs)  # H→D：目标块列
        fused.src_col.copy_to_gpu(num_reqs)  # H→D：源块列
        fused.token_bias.copy_to_gpu(num_reqs)  # H→D：接受偏移
        fused.ctx.run_fused_precopy(  # 启动融合预拷贝 kernel
            num_reqs=num_reqs,  # 请求数
            state_idx_gpu=fused.state_idx.gpu,  # GPU 侧目标块列
            src_col_gpu=fused.src_col.gpu,  # GPU 侧源块列
            token_bias_gpu=fused.token_bias.gpu,  # GPU 侧接受偏移
            idx_mapping=None,  # CPU 循环已按 batch 顺序填充，无需映射
        )
    else:  # 标量路径收尾：一次性执行批量拷贝
        do_mamba_copy_block(copy_bufs)


def postprocess_mamba_all(  # all 模式后处理：记录本步最后一个被调度 token 所在块列
    scheduler_output: SchedulerOutput,  # 调度器输出（调度 token 数）
    kv_cache_config: KVCacheConfig,  # KV 缓存配置（取 mamba 规格）
    input_batch: GPUInputBatch,  # 输入批（请求 id 顺序）
    requests: dict[str, CachedRequestState],  # req_id -> 请求状态映射
    mamba_state_idx: dict[str, int],  # req_id -> 状态块列 的持久化映射（本函数更新）
    num_spec_tokens: int,  # speculative token 数（all 模式必须 > 0）
    num_reqs: int,  # 活跃请求数
):
    """All-mode postprocess (only meaningful with num_spec_tokens > 0):
    record per-request the block index of the last token scheduled this
    step, so the next step can anchor its in-place writes when accepted
    drafts leave the sequence at a non-block-aligned position.

    all 模式后处理（仅在 num_spec_tokens > 0 时有意义）：
    记录每请求本步最后一个被调度 token 所在的块列，
    使下一步在草稿被接受导致序列非块对齐时，仍能锚定就地写入的位置。
    """
    if num_spec_tokens <= 0:  # 无 speculative token：all 模式不适用
        return
    _, mamba_spec = get_mamba_groups(kv_cache_config)  # 取 mamba 规格（组索引不需要）
    block_size = mamba_spec.block_size  # mamba 块大小
    full_decode_len = 1 + num_spec_tokens  # 完整一次 speculative decode 的 token 数（1 目标 + N 草稿）
    scheduled = scheduler_output.num_scheduled_tokens  # 每请求调度 token 数
    for req_id in input_batch.req_ids[:num_reqs]:  # 遍历活跃请求
        num_query = scheduled.get(req_id, 0)  # 该请求本步调度的 token 数
        if num_query == full_decode_len:  # 完整 spec decode 步：更新状态块列
            req = requests[req_id]  # 请求状态
            seq_len = req.num_computed_tokens + num_query  # 本步之后的序列长度
            mamba_state_idx[req_id] = max(0, (seq_len - 1) // block_size)  # 最后一个 token 所在块列（下限 0 防负）
        else:  # 非完整步（如仅目标 token 或 prefill）：移除记录，下一步按新请求处理
            mamba_state_idx.pop(req_id, None)


def preprocess_mamba_all_specdec(  # all 模式预处理：暂存上一步的"最后调度块列"供 kernel 读取
    scheduler_output: SchedulerOutput,  # 调度器输出（清理陈旧记录用）
    input_batch: GPUInputBatch,  # 输入批（请求 id 顺序）
    mamba_state_idx: dict[str, int],  # req_id -> 状态块列映射
    num_reqs: int,  # 活跃请求数
    prev_last_scheduled_idx_buf: CpuGpuBuffer,  # 暂存缓冲：上一步最后调度块列
) -> None:
    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)  # 先清理陈旧请求的记录
    np_view = prev_last_scheduled_idx_buf.np  # CPU 侧 numpy 视图
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):  # 逐请求填入记录（缺失则 -1）
        np_view[i] = mamba_state_idx.get(req_id, -1)  # -1 表示无上一步状态（新请求）
    np_view[num_reqs:].fill(-1)  # 尾部无效槽位也置 -1（kernel 可能整缓冲读取）
    prev_last_scheduled_idx_buf.copy_to_gpu()  # H→D 拷贝整个缓冲（供 kernel 使用）


def postprocess_mamba_align_gpu(  # GPU 端 align 后处理（spec decode + 混合模型 + align 模式专用入口）
    *,
    bufs: "MambaBuffers",  # mamba 缓冲集合（提供融合上下文）
    num_reqs: int,  # 活跃请求数
    num_accepted_tokens_gpu: torch.Tensor,  # GPU 上的每请求接受 token 数（runner 前向产出）
    num_accepted_tokens_cpu_tensor: torch.Tensor,  # CPU 侧接受计数张量（下一轮预处理读取）
    input_batch: GPUInputBatch,  # 输入批（块表来源）
    kv_cache_config: KVCacheConfig,  # KV 缓存配置（元数据初始化用）
    forward_context: dict[str, Any],  # 前向上下文（状态张量来源）
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],  # 拷贝函数集合（判别 conv/temporal）
) -> None:
    """GPU-side mamba postprocess for spec decode + hybrid + align mode.

    Lazily binds the fused-kernel context to the persistent block tables and
    forward-context state pointers on the first call, runs the fused kernel,
    and async-copies the per-request accepted-token counts back to the input
    batch's CPU tensor for the next iteration's preprocess.

    GPU 端 mamba 后处理（spec decode + 混合模型 + align 模式）。
    首次调用时惰性绑定融合上下文与持久化块表/前向上下文指针，
    运行融合 kernel，并将每请求接受计数异步拷回输入批的 CPU 张量，
    供下一轮预处理使用。
    """
    ctx = bufs.postprocess_align  # 取融合后处理上下文
    # Caller is responsible for gating on spec decode + hybrid; this assert is
    # a tripwire if those gates ever drift apart.
    # 调用方负责按 spec decode + 混合模型门控；此断言是门控失效时的绊线
    assert ctx is not None  # 上下文必须存在
    assert ctx.mamba_state_idx_buf is not None  # 四个暂存缓冲必须已分配
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    if not ctx.is_initialized:  # 首次调用：惰性初始化元数据
        ctx.initialize_from_forward_context(  # 提取状态布局元数据并缓存块表指针
            kv_cache_config,  # KV 缓存配置
            forward_context,  # 前向上下文
            mamba_state_copy_funcs,  # 拷贝函数集合
            [  # 各 mamba 组的持久化块表设备张量
                input_batch.block_table[gid].get_device_tensor(num_reqs)
                for gid in ctx.mamba_group_ids  # 按组索引顺序收集
            ],
        )

    ctx.run_fused_postprocess(  # 启动融合后处理 kernel（V1 决策路径）
        num_reqs=num_reqs,  # 请求数
        num_accepted_tokens_gpu=num_accepted_tokens_gpu,  # 接受计数（GPU 决策输入）
        mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,  # 暂存的状态块列
        num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,  # 暂存的调度 token 数
        num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,  # 暂存的已计算 token 数
        num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,  # 暂存的草稿 token 数
    )

    # ``num_accepted_tokens_out`` is pre-initialized from
    # ``num_accepted_tokens_gpu``; the kernel only overwrites entries to 1
    # when src_block_idx == dest_block_idx (copy within the same block), so
    # the original count is preserved for everyone else.
    # num_accepted_tokens_out 已预先用输入值初始化；kernel 仅在
    # src==dst（同块内拷贝）时改写为 1，其余请求保留原计数。
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(  # GPU -> CPU 异步拷贝接受计数
        ctx.num_accepted_tokens_out[:num_reqs], non_blocking=True  # 非阻塞：利用 pinned 内存
    )


def stage_postprocess_inputs_to_gpu(  # 将融合后处理 kernel 所需的每请求输入暂存到 GPU
    ctx: MambaSpecDecodeGPUContext,  # 融合上下文（持有暂存缓冲）
    scheduler_output: SchedulerOutput,  # 调度器输出（调度 token 数/草稿 token）
    req_ids: list[str],  # 本批请求 id（batch 顺序）
    num_reqs: int,  # 活跃请求数
    requests: dict[str, CachedRequestState],  # req_id -> 请求状态映射
    mamba_state_idx: dict[str, int],  # req_id -> 状态块列映射（preprocess 已填充）
) -> None:
    """Stage all per-request inputs the fused mamba postprocess kernel reads.

    Walks ``req_ids[:num_reqs]`` once, writing each request's mamba block
    index and scheduled/computed/draft token counts into the matching pinned
    numpy views, then issues four non-blocking H→D copies. The fused kernel
    indexes the resulting GPU tensors by ``req_idx``. Buffers live on ``ctx``
    and only exist when the postprocess kernel is enabled.

    Invariant: ``preprocess_mamba`` must have run first for the same batch so
    that every ``req_ids[i]`` has an entry in ``mamba_state_idx``.

    暂存融合 mamba 后处理 kernel 读取的全部每请求输入。
    单次遍历 req_ids[:num_reqs]，把各请求的 mamba 块列与调度/已计算/
    草稿 token 数写入对应的 pinned numpy 视图，然后发起四次非阻塞
    H→D 拷贝。融合 kernel 以 req_idx 索引这些 GPU 张量。
    不变式：同一批必须先运行 preprocess_mamba，保证每个 req_ids[i]
    在 mamba_state_idx 中有记录。
    """
    assert ctx.mamba_state_idx_buf is not None  # 四个暂存缓冲必须已分配（仅融合路径存在）
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens  # 每请求本步调度的草稿 token
    num_scheduled = scheduler_output.num_scheduled_tokens  # 每请求调度 token 总数
    state_idx_np = ctx.mamba_state_idx_buf.np  # CPU 视图：状态块列
    scheduled_np = ctx.num_scheduled_tokens_buf.np  # CPU 视图：调度 token 数
    computed_np = ctx.num_computed_tokens_buf.np  # CPU 视图：已计算 token 数
    draft_np = ctx.num_draft_tokens_buf.np  # CPU 视图：草稿 token 数

    for i in range(num_reqs):  # 逐请求填充四个 pinned 视图
        req_id = req_ids[i]  # 第 i 个 batch 行的请求 id
        state_idx = mamba_state_idx.get(req_id)  # 读取状态块列（须已存在）
        assert state_idx is not None, (  # 违反不变式立即失败，避免 kernel 读到垃圾索引
            f"mamba_state_idx missing entry for {req_id!r}; "
            "preprocess_mamba must run before stage_postprocess_inputs_to_gpu"
        )
        state_idx_np[i] = state_idx  # 状态块列
        scheduled_np[i] = num_scheduled[req_id]  # 调度 token 数
        computed_np[i] = requests[req_id].num_computed_tokens  # 已计算 token 数
        draft_np[i] = len(scheduled_spec_tokens.get(req_id, []))  # 草稿 token 数（无则 0）

    ctx.mamba_state_idx_buf.copy_to_gpu(num_reqs)  # H→D：状态块列（仅前 num_reqs 项）
    ctx.num_scheduled_tokens_buf.copy_to_gpu(num_reqs)  # H→D：调度 token 数
    ctx.num_computed_tokens_buf.copy_to_gpu(num_reqs)  # H→D：已计算 token 数
    ctx.num_draft_tokens_buf.copy_to_gpu(num_reqs)  # H→D：草稿 token 数
