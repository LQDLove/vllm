# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# worker 层通用工具模块。
# 包含 KV cache 块清零 Triton kernel(KVBlockZeroer)、注意力分组(AttentionGroup)、
# 内核块大小选择、多模态编码器输出检查、请求内存计算、KV 共享层绑定、
# KV cache 绑定与块拷贝,以及序列并行(SP)残差分散判断等工具。

# 导入 math 模块,用于 request_memory 中的向上取整。
import math
# 导入 defaultdict,用于按层索引聚合 KV cache 层名。
from collections import defaultdict
# 导入 Iterable 与 Sequence 抽象类型,用于类型标注。
from collections.abc import Iterable, Sequence
# 导入 dataclass 与 field,用于定义 AttentionGroup 数据类。
from dataclasses import dataclass, field
# 导入 itertools.product 并命名为 iprod,用于多维外层组合遍历。
from itertools import product as iprod
# 导入 Any 类型,用于任意类型的标注。
from typing import Any

# 导入 numpy,用于 KV 块拷贝数据的构造。
import numpy as np
# 导入 PyTorch,用于张量操作。
import torch

# 导入 CacheConfig 与 VllmConfig 配置类。
from vllm.config import CacheConfig, VllmConfig
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入注意力层类 Attention(用于 bind_kv_cache 类型标注)。
from vllm.model_executor.layers.attention import Attention
# 导入多模态嵌入输出类型 MultiModalEmbeddings。
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
# 导入 extract_layer_index,用于从层名提取层索引。
from vllm.model_executor.models.utils import extract_layer_index
# 导入 current_platform,用于平台判断(在 bind_kv_cache 中)。
from vllm.platforms import current_platform
# 导入 Triton 工具(tl 语言与 triton 装饰器),用于定义 Triton kernel。
from vllm.triton_utils import tl, triton
# 导入 largest_power_of_2_divisor,用于计算 2 的幂的最大除数。
from vllm.utils.math_utils import largest_power_of_2_divisor
# 导入 MemorySnapshot 与 format_gib,用于内存快照与单位格式化。
from vllm.utils.mem_utils import MemorySnapshot, format_gib
# 导入异步主机到设备拷贝函数 async_tensor_h2d。
from vllm.utils.torch_utils import async_tensor_h2d
# 导入注意力后端类型:AttentionBackend、AttentionMetadataBuilder、MultipleOf。
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    MultipleOf,
)
# 导入 KV cache 块拷贝数据结构 KVCacheBlockCopy。
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
# 导入 KV cache 相关规格类型:AttentionSpec、EncoderOnlyAttentionSpec 等。
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.block_table import get_block_table_width

# 创建本模块的日志记录器。
logger = init_logger(__name__)


def raise_if_nan_logits(num_nans_in_logits: Mapping[str, int]) -> None:
    if not any(num_nans_in_logits.values()):
        return

    corrupted_requests = {
        req_id: num_nans
        for req_id, num_nans in num_nans_in_logits.items()
        if num_nans > 0
    }
    raise RuntimeError(f"NaNs detected in logits: {corrupted_requests}")


@triton.jit(do_not_specialize=["n_blocks"])
def _zero_kv_blocks_kernel(
    # Triton kernel:在一次启动中跨所有段清零 KV cache 块。
    # 每个段是某个块数据的连续区域;对块在最外层(block_dim=0)的后端,每个缓冲一个段;
    # 对 K/V 在最外层(block_dim=1)的后端,每个缓冲两个段(K 一个、V 一个)。
    # 段的页大小可能不同(如 MLA + DSA 索引器的多 KV cache 组),每个段的页大小
    # 从 seg_page_sizes_ptr 读取,chunk 索引超界的程序提前退出。
    # seg_addrs_ptr 保存每段的绝对字节地址(int64),使段可跨多个 CUDA 分配。
    # 程序映射为 (block_index, seg_index, chunk_index)。
    seg_addrs_ptr,
    # 段基地址指针(int64 数组)。
    seg_page_sizes_ptr,
    # 段页大小指针(int64 数组)。
    block_ids_ptr,
    # 块 id 指针(待清零的块编号)。
    n_blocks,
    # 块总数。
    N_SEGS: tl.constexpr,
    # 段数(编译期常量)。
    MAX_CHUNKS: tl.constexpr,
    # 每段最大 chunk 数(编译期常量)。
    BLOCK_SIZE: tl.constexpr,
    # 每个 chunk 处理的元素数(编译期常量)。
):
    # 单次启动跨所有段清零 KV cache 块。
    # 每个段是某块数据的连续区域;块最外层后端每缓冲一个段,K/V 最外层后端
    # 每缓冲两个段。段的页大小可不同,超界程序提前退出;地址为绝对字节地址。
    # 获取程序 id(一维网格)。
    pid = tl.program_id(0)
    # 每块的工作量 = 段数 × 每段最大 chunk 数。
    work_per_block = N_SEGS * MAX_CHUNKS
    # 块索引 = pid // 每块工作量。
    block_index = pid // work_per_block
    # 若块索引超出待清零块数,提前退出。
    if block_index >= n_blocks:
        return
    # 计算块内余数,用于解析段与 chunk。
    remainder = pid % work_per_block
    # 段索引 = 余数 // 每段最大 chunk 数。
    seg_index = remainder // MAX_CHUNKS
    # chunk 索引 = 余数 % 每段最大 chunk 数。
    chunk_index = remainder % MAX_CHUNKS
    # 加载该段的页大小(元素数)。
    page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
    # 若 chunk 索引超出该段页大小能容纳的 chunk 数,提前退出。
    if chunk_index >= page_size_el // BLOCK_SIZE:
        return
    # 加载块 id。
    block_id = tl.load(block_ids_ptr + block_index)
    # 加载段基地址(字节)。
    seg_addr = tl.load(seg_addrs_ptr + seg_index)
    # 把地址转为 int32 指针(用于写入)。
    ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
    # 计算偏移:块 id × 页大小 + chunk × BLOCK_SIZE(int64 运算防溢出)。
    offset = (
        block_id.to(tl.int64) * page_size_el.to(tl.int64)
        + chunk_index.to(tl.int64) * BLOCK_SIZE
    )
    # 生成列偏移向量(0..BLOCK_SIZE)并转为 int64。
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    # 把该 chunk 区域写为零(int32 全零)。
    tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


class KVBlockZeroer:
    # KV cache 块清零器:通过 Triton kernel 高效清零 KV cache 块。
    # 在 KV cache 分配后构造一次以预计算段地址,然后每步调用
    # zero_block_ids 清零新分配的块。
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Construct once after KV caches are allocated to precompute segment
    addresses, then call zero_block_ids each step to zero newly-allocated blocks.
    """

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        static_forward_context: dict[str, Any],
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        # 预计算 Triton 清零 kernel 的绝对地址表。
        # 每个条目是段在 GPU 上起始的绝对字节地址,因此跨 CUDA 分配的段可正常工作。
        # 调度器的块 id 引用的逻辑块大小可能不同于 kernel 块大小(虚拟块拆分);
        # 每段的 page_size_el 考虑该比例,使块 id × 页大小落在正确偏移。
        # 只处理 AttentionSpec 层;Mamba 层跳过。
        # 保存目标设备。
        self.device = device
        # 元数据元组(段地址、页大小、最大 chunk 数、块大小、段数),初始为 None。
        self._meta: tuple[torch.Tensor, torch.Tensor, int, int, int] | None = None

        # 若未提供仅 runner 的注意力层集合,初始化为空集。
        if runner_only_attn_layers is None:
            runner_only_attn_layers = set()
        # 用集合记录已处理的 KV 张量数据地址,避免重复。
        seen_ptrs: set[int] = set()
        # 累积段基地址(字节)。
        seg_addrs: list[int] = []
        # 累积段页大小(元素数)。
        seg_page_sizes: list[int] = []

        # 遍历每个注意力组:
        for group in attn_groups_iter:
            # 获取该组的 KV cache 规格。
            spec = group.kv_cache_spec
            # 仅处理全注意力规格(跳过 Mamba 等)。
            if not isinstance(spec, FullAttentionSpec):
                continue
            # 若该组 id 超出 kernel 块大小列表,跳过。
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            # 取该组对应的 kernel 块大小。
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            # 计算分配块大小与 kernel 块大小之比(虚拟块拆分比例)。
            ratio = spec.block_size // kernel_bs
            # 查询后端 KV cache 的块维度(block_dim)。
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_bs,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )

            # 遍历该组的每个层:
            for layer_name in group.layer_names:
                # 跳过仅 runner 的注意力层(无自己的 KV cache)。
                if layer_name in runner_only_attn_layers:
                    continue
                # 从静态前向上下文获取该层的 KV cache 张量。
                kv = static_forward_context[layer_name].kv_cache
                # 若非 torch.Tensor 则跳过。
                if not isinstance(kv, torch.Tensor):
                    continue
                # 取该张量的数据地址。
                dp = kv.data_ptr()
                # 若该地址已处理过,跳过。
                if dp in seen_ptrs:
                    continue
                # 记录已处理地址。
                seen_ptrs.add(dp)

                # 取每元素字节数。
                el = kv.element_size()
                # 计算沿块维度的跨步字节数。
                cur_bytes = kv.stride(block_dim) * el
                # 断言跨步字节数为 4 的倍数(Triton int32 写入要求)。
                assert cur_bytes % 4 == 0
                # kernel 块元素数 = 跨步字节数 / 4。
                kernel_block_el = cur_bytes // 4
                # 页元素数 = kernel 块元素数 × 虚拟块比例。
                cur_page_el = kernel_block_el * ratio

                # 记录块维度的跨步字节。
                block_stride_bytes = cur_bytes
                # 找出块维度之外且跨步字节大于块跨步的外层维度。
                outer_dims = [
                    d
                    for d in range(block_dim)
                    if kv.stride(d) * el > block_stride_bytes
                ]
                # 计算这些外层维度的跨步字节。
                outer_strides = [kv.stride(d) * el for d in outer_dims]
                # 遍历外层维度的所有组合:
                for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                    # 计算外层组合的字节偏移。
                    off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                    # 记录段基地址(数据地址 + 外层偏移)。
                    seg_addrs.append(dp + off_bytes)
                    # 记录对应的页大小。
                    seg_page_sizes.append(cur_page_el)

        # 若没有找到任何段(无注意力层需要清零):
        if not seg_addrs:
            # 元数据置为 None。
            self._meta = None
            return

        # 取所有段页大小的最大值。
        max_page_size_el = max(seg_page_sizes)
        # 选择块大小:所有段页大小的 2 的幂最大除数的最小值,且不超过 1024。
        blk_size = min(
            min(largest_power_of_2_divisor(ps) for ps in seg_page_sizes),
            1024,
        )
        # 保存完整元数据:
        self._meta = (
            # 段地址张量(uint64,设备端)。
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            # 段页大小张量(int64,设备端)。
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            # 最大 chunk 数 = 最大页大小 / 块大小。
            max_page_size_el // blk_size,
            # 块大小。
            blk_size,
            # 段总数。
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        # 清零给定块 id 列表对应的 KV cache 内存。
        # 若块列表为空或无元数据,直接返回。
        if not block_ids or self._meta is None:
            return
        # 解包元数据:段地址、页大小、最大 chunk 数、块大小、段数。
        seg_addrs, seg_page_sizes, max_chunks, blk_size, n_segs = self._meta
        # 块数量。
        n_blocks = len(block_ids)
        # 把块 id 列表异步拷到设备端 int64 张量。
        idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
        # 计算网格大小 = 块数 × 段数 × 最大 chunk 数。
        grid = (n_blocks * n_segs * max_chunks,)
        # 启动清零 kernel。
        _zero_kv_blocks_kernel[grid](
            seg_addrs,
            seg_page_sizes,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            MAX_CHUNKS=max_chunks,
            BLOCK_SIZE=blk_size,
        )

    def warmup(self, num_kv_blocks: int) -> None:
        """JIT-compile the zeroing kernel before the first real request."""
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


@dataclass
class AttentionGroup:
    # 注意力组:将共享同一 attention 后端与 KV cache 布局的层归为一组。
    # 后端类(该组注意力实现)。
    backend: type[AttentionBackend]
    # 组内层名列表。
    layer_names: list[str]
    # 该组的 KV cache 规格。
    kv_cache_spec: KVCacheSpec
    # 该组在 KV cache 组列表中的 id。
    kv_cache_group_id: int
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistent buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    # 说明:启用微批时,为每个微批各建一个元数据构建器,使其使用 CUDA Graph
    # 内部持久缓冲时不会与其它微批冲突。
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=lambda: []
        # 元数据构建器列表(默认空)。
    )

    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ):
        # 创建该注意力组的元数据构建器列表。
        # 参数:
        #   vllm_config: 完整配置。
        #   device: 目标设备。
        #   kernel_block_size: 可选,覆盖内核块大小。
        #   num_metadata_builders: 构建器数量(微批数)。
        # 若指定了 kernel 块大小,生成一个带新块大小的 KV 规格副本;
        kv_cache_spec_builder = (
            self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
            if kernel_block_size is not None
            # 未指定则沿用原规格。
            else self.kv_cache_spec
        )
        # 为每个微批创建一个后端构建器实例。
        self.metadata_builders = [
            builder_cls(
                kv_cache_spec_builder,
                self.layer_names,
                vllm_config,
                device,
                **builder_kwargs,
            )
            for _ in range(num_metadata_builders)
        ]

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        # 按微批 id 获取该组的注意力元数据构建器。
        # 断言微批 id 在构建器数量范围内。
        assert len(self.metadata_builders) > ubatch_id
        # 返回对应微批的构建器。
        return self.metadata_builders[ubatch_id]


def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
) -> int:
    # 选择被所有后端支持且为 kv_manager_block_size 因子的块大小。
    # 若 kv_manager_block_size 被所有后端支持则直接返回,否则返回最大支持值。
    # 若找不到有效块大小则抛 ValueError。

    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
    ) -> bool:
        # 检查块大小是否被所有后端支持。
        # 遍历每个后端:
        for backend in backends:
            # 初始化支持标志为 False。
            is_supported = False
            # 遍历该后端支持的内核块大小列表:
            for supported_size in backend.get_supported_kernel_block_sizes():
                # 若支持大小是整数,则需精确匹配。
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                # 若支持大小是 MultipleOf 类型,则块大小需为其基数的倍数。
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                # 未知支持类型则抛出错误。
                else:
                    raise ValueError(f"Unknown supported size: {supported_size}")
            # 若该后端不支持,整体不支持,返回 False。
            if not is_supported:
                return False
        # 所有后端都支持则返回 True。
        return True

    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    # 情形 1:若 kv cache 管理器的块大小被所有后端支持,直接返回之。
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # Case 2: otherwise, the block_size must be an `int`-format supported size of
    # at least one backend. Iterate over all `int`-format supported sizes in
    # descending order and return the first one that is supported by all backends.
    # 情形 2:否则,块大小必须是至少一个后端的 int 格式支持值。
    # 按降序遍历所有 int 格式支持值,返回第一个被所有后端支持的。
    # 收集所有后端的 int 格式支持大小。
    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    # 按降序遍历候选大小:
    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        # 若大小不是 kv 管理器块大小的因子,跳过。
        if kv_manager_block_size % supported_size != 0:
            continue
        # 若该大小被所有后端支持:
        if block_size_is_supported(backends, supported_size):
            # 返回该大小。
            return supported_size
    # 未找到公共块大小则抛出错误。
    raise ValueError(f"No common block size for {kv_manager_block_size}. ")


def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    # 生成与每个 block_size 匹配的 kernel_block_sizes。
    # 对支持虚拟块拆分的注意力后端,使用后端支持的大小;
    # 对其它后端(如 Mamba),使用相同块大小(不拆分)。
    # 初始化 kernel 块大小列表。
    kernel_block_sizes = []
    # 遍历每个 KV cache 组:
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        # 取该组的 KV cache 规格。
        kv_cache_spec = kv_cache_group.kv_cache_spec
        # 若规格是 UniformTypeKVCacheSpecs(组内所有层同类型):
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            # 取任意一个子规格用于派发(组内层同类型)。
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        # 若是仅编码器注意力规格,跳过(无内核块)。
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            continue
        # 若是注意力规格(支持虚拟块拆分):
        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            # 取 KV 管理器块大小。
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            # 取该组对应的注意力后端类列表。
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            # 选择所有后端支持的公共内核块大小。
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            # 记录选择的内核块大小。
            kernel_block_sizes.append(selected_kernel_size)
        # 若是 Mamba 规格(非注意力缓存,不拆分):
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            # 直接使用规格的块大小。
            kernel_block_sizes.append(kv_cache_spec.block_size)
        # 其它未知规格:
        else:
            # 抛出未实现错误。
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    # 返回所有组的内核块大小列表。
    return kernel_block_sizes


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    # 对 embed_multimodal 的结果做健全性检查。
    # 断言多模态嵌入是 list/tuple(2D 张量序列)或单个 3D 张量。
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    # 断言嵌入数量与输入项数量一致。
    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )

    # 断言每个嵌入都是 2D 张量(序列情况)。
    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `embed_multimodal` method."
    )


def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    # 计算 vLLM 所需内存,并验证当前空闲内存是否足够。
    # 请求内存 = 总内存 × 显存利用率(向上取整)。
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    # 若当前空闲内存小于请求内存:
    if init_snapshot.free_memory < requested_memory:
        # 抛出错误:空闲内存不足以满足目标利用率。
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    # 返回请求内存大小。
    return requested_memory


def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None:
    # 建立 KV cache 共享:为不分配自己 KV cache 的层复用已分配的 KV cache,
    # 依据 shared_kv_cache_layers 映射。把这些层加入对应 KV cache 组,
    # 以确保后续注意力元数据被正确分配。
    # 若无共享层映射,直接返回。
    if not shared_kv_cache_layers:
        return

    # 构建 "层名 -> KV cache 组" 映射。
    layer_to_kv_cache_group: dict[str, KVCacheGroupSpec] = {}
    # 遍历每个 KV cache 组:
    for kv_cache_group in kv_cache_groups:
        # 遍历组内每个层名:
        for layer_name in kv_cache_group.layer_names:
            # 记录层名到组的映射。
            layer_to_kv_cache_group[layer_name] = kv_cache_group

    # 遍历共享映射(源层名 -> 目标层名):
    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        # 找到目标层所在的 KV cache 组。
        tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
        # 把源层名追加到该组(共享目标的 KV cache)。
        tgt_kv_cache_group.layer_names.append(layer_name)

        # 若提供了仅 runner 的注意力层集合:
        if runner_only_attn_layers is not None:
            # 把源层加入其中(说明它不分配自己的 KV cache)。
            runner_only_attn_layers.add(layer_name)


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None:
    # 把分配的 KV cache 绑定到 ModelRunner 与前向上下文,使其可在前向中使用。
    # 1) 用 kv_caches 填充 ModelRunner 的 kv cache 列表;
    # 2) 把 forward_context 中的每个注意力层与其 KV cache 关联。
    # Bind kv_caches to ModelRunner(先把 kv_caches 绑定到 ModelRunner)。
    # 断言 runner 的 kv cache 列表当前为空。
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    # 把 kv_caches 字典按层索引顺序转换为张量列表。
    # 按层索引聚合层名(一个索引下可能多个层,如 encoder-decoder 的 cross/self)。
    index2name = defaultdict(list)
    # 遍历每个层名:
    for layer_name in kv_caches:
        # 提取层索引并追加层名。
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    # 按层索引升序遍历:
    for layer_index in sorted(index2name.keys()):
        # 取该索引下的所有层名。
        layer_names = index2name[layer_index]
        # 若同一索引有多个层:
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            # 典型场景是 encoder-decoder 模型(如 bart):同一 decoder 层中的
            # cross attention 与 self attention 层名不同但层索引相同。
            # TODO - analyze where runner_kv_caches is used and the right
            # way to ensure it properly reflects multiple attention layers
            # in the same decoder block.
            # 注:需分析 runner_kv_caches 的使用位置,以及如何正确反映
            # 同一 decoder 块中的多个注意力层。
            # 若平台是 CUDA 类/XPU/CPU:
            if (
                current_platform.is_cuda_alike()
                or current_platform.is_xpu()
                or current_platform.is_cpu()
            ):
                # We know that the GPU / CPU runner is not impacted by this
                # case. Some test code depends on runner_kv_caches, but
                # not in a way that's impacted by ignoring this.
                # 已知 GPU/CPU runner 不受此情况影响;忽略不影响测试代码。
                pass
            # 其它平台:
            else:
                # 抛出未实现错误。
                raise NotImplementedError
        # 遍历该索引下的每个层名:
        for layer_name in layer_names:
            # 把对应 KV cache 张量追加到 runner 列表。
            runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context. Each layer's bind_kv_cache unpacks
    # its raw allocation into the per-layer view(s) it needs (e.g. Mamba
    # splits conv/ssm), so the kv_caches dict can hold a single tensor per
    # layer for the KV connector to register.
    # 把 kv_caches 绑定到前向上下文。每个层的 bind_kv_cache 会把其原始分配解包
    # 成所需的分层视图(如 Mamba 拆分为 conv/ssm),因此 kv_caches 字典每层
    # 可只持有一个张量供 KV connector 注册。
    # 遍历每个层名与 KV cache:
    for layer_name, kv_cache in kv_caches.items():
        # 调用该层注意力对象的 bind_kv_cache 绑定。
        forward_context[layer_name].bind_kv_cache(kv_cache)


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    # 就地拷贝 KV cache 块(前缀缓存部分命中的 copy-on-write 场景)。
    # kv_caches 提供底层存储,按 (src, dst) 对在存储张量间做块级拷贝。
    # 若无待拷贝块,直接返回。
    if not kv_cache_block_copies:
        return

    # 收集涉及的底层存储张量(去重)。
    storage_tensors: list[torch.Tensor] = []
    # 用集合记录已见的存储地址。
    seen_storage: set[int] = set()
    # 遍历每个 KV cache 条目:
    for entry in kv_caches:
        # Mamba layers hold a list of state tensors; attention layers a single
        # tensor. Both alias the shared block-major backing storage.
        # Mamba 层保存状态张量列表;注意力层保存单个张量;两者都别名共享的
        # 块主序底层存储。
        # 列表/元组则展开为多个张量,否则当作单个张量。
        tensors = entry if isinstance(entry, (list, tuple)) else (entry,)
        # 遍历每个张量:
        for tensor in tensors:
            # 取底层存储的数据地址。
            ptr = tensor.untyped_storage().data_ptr()
            # 若该存储已处理过,跳过。
            if ptr in seen_storage:
                continue
            # 记录已处理的存储地址。
            seen_storage.add(ptr)
            # 把该张量加入存储张量列表。
            storage_tensors.append(tensor)

    # 若无存储张量,直接返回。
    if not storage_tensors:
        return
    # 取第一个存储张量的设备。
    device = storage_tensors[0].device
    # 把拷贝对列表转为 numpy int64 数组。
    indices_np = np.array(kv_cache_block_copies, dtype=np.int64)
    # 异步拷到设备端。
    indices = async_tensor_h2d(indices_np, device=device)
    # 按列拆分为源索引与目标索引。
    src_indices, dst_indices = indices.unbind(dim=1)

    # 遍历每个存储张量:
    for tensor in storage_tensors:
        # 断言张量在目标设备上。
        assert tensor.device == device
        # 创建空 uint8 张量作为存储的视图容器。
        blocks = torch.empty(0, dtype=torch.uint8, device=device)
        # 把该张量指向底层存储(按字节视图)。
        blocks.set_(tensor.untyped_storage())
        # Block-major backing storage: block i owns the contiguous byte range
        # [i * page_size, (i + 1) * page_size).
        # 块主序存储:块 i 拥有连续字节区间 [i*page_size, (i+1)*page_size)。
        # 断言存储字节数为块数的整数倍。
        assert blocks.numel() % num_blocks == 0
        # 把存储视图重塑为 (块数, 每块字节数)。
        blocks = blocks.view(num_blocks, -1)
        # 执行块级拷贝:目标块 = 源块。
        blocks[dst_indices] = blocks[src_indices]


def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool:
    # 判断序列并行(SP)下残差张量是否分散。
    # 启用 SP 与 TP 时残差张量分散在各 TP rank;SP 仅在整图编译模式下支持。
    # 若未启用 SP,返回 False。
    if not vllm_config.compilation_config.pass_config.enable_sp:
        return False

    # 取张量并行大小。
    tp = vllm_config.parallel_config.tensor_parallel_size

    # 若 TP 为 1(无 TP),残差不分散。
    if tp == 1:
        return False

    # 断言处于整图编译模式(使用 inductor 图分区或未拆分算子)。
    assert (
        vllm_config.compilation_config.use_inductor_graph_partition
        or not vllm_config.compilation_config.splitting_ops
    ), "Sequence parallelism requires full-graph compilation"

    # When sequence parallelism is enabled, we always pad num_input_tokens
    # to be a multiple of tensor_parallel_size (tp) earlier.
    # 启用 SP 时,之前总会把 num_input_tokens 填充为 tp 的倍数。
    # 断言输入 token 数为 tp 的倍数。
    assert num_input_tokens % tp == 0

    # 返回 True:残差处于分散布局。
    return True