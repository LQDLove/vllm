# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# KV cache 块表(BlockTable / MultiGroupBlockTable)。
# 维护请求与 KV cache 物理块的映射,计算 slot mapping(供注意力 kernel 使用),
# 支持虚拟块拆分(hybrid)、上下文并行(CP/DCP/PCP)分片与块行操作。

# 导入 Enum,用于定义 slot 映射模式枚举。
from enum import Enum

# 导入 numpy,用于块表数组操作。
import numpy as np
# 导入 PyTorch,用于张量操作。
import torch

# 导入 DCP 组与 PCP 组访问器。
from vllm.distributed import get_dcp_group, get_pcp_group
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 Triton 工具(tl 与 triton),用于定义 slot 映射 kernel。
from vllm.triton_utils import tl, triton
# 导入向上整除工具 cdiv。
from vllm.utils.math_utils import cdiv
# 导入 PAD_SLOT_ID 常量(填充槽位 id)。
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
# 导入 CPU/GPU 双缓冲。
from vllm.v1.utils import CpuGpuBuffer

# 创建本模块的日志记录器。
logger = init_logger(__name__)


def get_block_table_width(
    max_num_blocks: int,
    block_size: int,
    kernel_block_size: int | None = None,
    *,
    token_alignment: int | None = 128,
) -> int:
    """Return the width after optional alignment and virtual block splitting."""
    if kernel_block_size is None:
        kernel_block_size = block_size
    if block_size % kernel_block_size != 0:
        raise ValueError(
            f"kernel_block_size {kernel_block_size} must divide "
            f"block_size {block_size} evenly"
        )
    if token_alignment is not None:
        if token_alignment <= 0:
            raise ValueError("token_alignment must be positive")
        block_alignment = token_alignment // math.gcd(token_alignment, block_size)
        max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment
    return max_num_blocks * block_size // kernel_block_size


class SlotMappingMode(Enum):
    # slot mapping 模式枚举。
    # TOKEN_TO_KV_SLOT: 将调度 token 映射到 KV cache 槽位(常规注意力)。
    TOKEN_TO_KV_SLOT = "token_to_kv_slot"
    # NONE: 不使用 token 槽位映射(Mamba/GDN 等状态缓存)。
    NONE = "none"


class BlockTable:
    # KV cache 块表:维护每个请求与 KV cache 物理块的映射。
    # 支持分配块大小与 kernel 块大小不同(hybrid 拆分)时的映射,
    # 并负责计算 token -> 槽位的 slot mapping 及块的增删移动。

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        slot_mapping_mode: SlotMappingMode = SlotMappingMode.TOKEN_TO_KV_SLOT,
    ):
        # 初始化块表缓冲区。
        # Args:
        #     block_size: KV cache 分配块大小。
        #     max_num_reqs: 最大并发请求数。
        #     max_num_blocks_per_req: 每请求最大块数。
        #     max_num_batched_tokens: 单批最大 token 数。
        #     pin_memory: 是否锁页加速 GPU 传输。
        #     device: 目标设备。
        #     kernel_block_size: 注意力 kernel 的块大小;若 `block_size` 被
        #         kernel 支持则与之相同。
        #     slot_mapping_mode: 本缓存组如何将调度 token 映射到缓存槽位;
        #         Mamba 类状态缓存不使用 token 槽映射,应用 NONE。
        # 记录最大请求数。
        self.max_num_reqs = max_num_reqs
        # 记录单批最大 token 数。
        self.max_num_batched_tokens = max_num_batched_tokens
        # 记录是否锁页。
        self.pin_memory = pin_memory
        # 记录目标设备。
        self.device = device
        # 记录 KV cache 分配块大小。
        self.kv_cache_block_size = block_size

        # 若 kernel 块大小与分配块大小相同:
        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            # 标准情形:分配与计算使用相同块大小,无需拆分,直接映射。
            self.block_size = block_size
            # 每 KV 块对应 1 个 kernel 块。
            self.blocks_per_kv_block = 1
            # 不使用混合块。
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            # 混合情形:分配块大小与 kernel 块大小不同;内存块按 kernel 需求细分。
            # 例:32-token 内存块与 16-token kernel 块 -> 每内存块对应 2 个 kernel 块。
            # 若分配块大小不是 kernel 块大小的整数倍,报错。
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            # 使用 kernel 块大小作为计算块大小。
            self.block_size = kernel_block_size
            # 每 KV 块对应的 kernel 块数。
            self.blocks_per_kv_block = block_size // kernel_block_size
            # 使用混合块。
            self.use_hybrid_blocks = True

        # 每请求最大块数按 kernel 块数换算。
        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        # 创建块表缓冲区(请求数 × 最大块数)。
        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        # 记录每行(请求)的块数。
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        # 创建 slot mapping 缓冲区(批最大 token 数)。
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        # 若使用混合块:
        if self.use_hybrid_blocks:
            # 预计算 kernel 块 arange(用于块 id 拆分)。
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            # 非混合块时无需。
            self._kernel_block_arange = None

        # 尝试获取 PCP 组信息:
        try:
            # PCP 世界大小。
            self.pcp_world_size = get_pcp_group().world_size
            # PCP rank。
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            # 测试中 PCP 可能未初始化。
            self.pcp_world_size = 1
            self.pcp_rank = 0
        # 尝试获取 DCP 组信息:
        try:
            # DCP 世界大小。
            self.dcp_world_size = get_dcp_group().world_size
            # DCP rank。
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            # 测试中 DCP 可能未初始化。
            self.dcp_world_size = 1
            self.dcp_rank = 0
        # 记录 CP 的 KV 交错粒度。
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size
        # 记录 slot 映射模式。
        self.slot_mapping_mode = slot_mapping_mode

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        # 为请求行追加一行新块。
        # 参数:
        #   block_ids: 新块 id 列表。
        #   row_idx: 目标行(请求)索引。
        # 若块列表为空,直接返回。
        if not block_ids:
            return

        # 若使用混合块,先把 KV 块 id 拆分为 kernel 块 id。
        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        # 新块数量。
        num_blocks = len(block_ids)
        # 取当前行已有块数作为写入起点。
        start = self.num_blocks_per_row[row_idx]
        # 更新行块数。
        self.num_blocks_per_row[row_idx] += num_blocks
        # 把块 id 写入块表(从起点开始的区间)。
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        # 覆盖写入一行(先清空再追加)。
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def clear_row(self, row_idx: int) -> None:
        # 清空指定行,归还块槽位。
        # 取该行块数。
        num_blocks = self.num_blocks_per_row[row_idx]
        # 若块数 > 0,将块表对应位置清零。
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        # 重置行块数。
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        # 把 src 行的块内容移动到 tgt 行。
        # 取源行块数。
        num_blocks = self.num_blocks_per_row[src]
        # 取块表 numpy 视图。
        block_table_np = self.block_table.np
        # 拷贝块内容。
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        # 更新目标行块数。
        self.num_blocks_per_row[tgt] = num_blocks
        # Clear the vacated source row: dummy-run batches dereference stale
        # rows as mamba state slots and write state in place there, possibly
        # after the blocks have been freed and reallocated.
        block_table_np[src, :num_blocks] = 0
        self.num_blocks_per_row[src] = 0

    def swap_row(self, src: int, tgt: int) -> None:
        # 交换 src 与 tgt 两行的块。
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        # 交换块数。
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        # 交换块内容。
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # 计算每请求 token 到 KV 槽位的映射。
        # 参数:
        #   num_reqs: 请求数。
        #   query_start_loc: 各请求起始位置。
        #   positions: token 位置。
        # token 总数。
        num_tokens = positions.shape[0]
        # 若映射模式为 NONE(Mamba/GDN 组以块表为循环状态索引,无需逐 token 映射):
        if self.slot_mapping_mode == SlotMappingMode.NONE:
            # Mamba/GDN groups consume the block table as recurrent state
            # indices and do not use per-token slot mappings.
            # 直接返回。
            return
        # 断言映射模式为 TOKEN_TO_KV_SLOT。
        assert self.slot_mapping_mode == SlotMappingMode.TOKEN_TO_KV_SLOT

        # 启动 Triton kernel 计算 slot 映射(每请求一个程序)。
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            KV_CACHE_BLOCK_SIZE=self.kv_cache_block_size,
            BLOCKS_PER_KV_BLOCK=self.blocks_per_kv_block,
            TOTAL_CP_WORLD_SIZE=self.dcp_world_size,
            TOTAL_CP_RANK=self.dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    def commit_block_table(self, num_reqs: int) -> None:
        # 把前 num_reqs 行的块表提交(拷贝)到 GPU。
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        # 清空 GPU 与 CPU 块表。
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        # 把 KV 管理器块 id 转换为 kernel 块 id。
        # 例:kv_manager_block_ids=[0,1,2],每个 KV 块拆为 2 个 kernel 块,
        # 结果 [0,1,2,3,4,5](0→[0,1],1→[2,3],2→[4,5])。
        # 若每 KV 块仅 1 个 kernel 块,直接返回。
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        # 每个 KV 块 id 映射为 blocks_per_kv_block 个 kernel 块 id。
        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        # 展平返回。
        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        # 返回前 num_reqs 行的设备端块表张量。
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        # 返回 CPU 端块表张量。
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        # 返回块表的 numpy 视图。
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        # 创建指定形状/类型的 CPU-GPU 双缓冲。
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


class MultiGroupBlockTable:
    # 各 KV cache 组的块表集合(每个组一个 BlockTable)。
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int],
        cp_kv_cache_interleave_size: int = 1,
        slot_mapping_modes: list[SlotMappingMode] | None = None,
    ) -> None:
        # 检查 kernel 块大小列表与块大小列表长度一致。
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        # 若未提供 slot 映射模式,默认全部为 TOKEN_TO_KV_SLOT。
        if slot_mapping_modes is None:
            slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT] * len(block_sizes)
        # 检查模式列表长度。
        if len(slot_mapping_modes) != len(block_sizes):
            raise ValueError(
                f"slot_mapping_modes length ({len(slot_mapping_modes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # 检查每请求最大块数列表长度。
        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # Align to a multiple of (128 / block_size) as required
        # by some attention backends such as TRTLLM (#39324)
        # 对齐到 (128/block_size) 的倍数(部分后端如 TRTLLM 要求)。
        max_num_blocks = [
            (
                get_block_table_width(n, block_size, token_alignment=None)
                if slot_mapping_mode == SlotMappingMode.NONE
                else get_block_table_width(n, block_size)
            )
            for n, block_size, slot_mapping_mode in zip(
                max_num_blocks, block_sizes, slot_mapping_modes
            )
        ]

        # 为每个 KV cache 组创建一个 BlockTable。
        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
                slot_mapping_mode=slot_mapping_mode,
            )
            for (
                block_size,
                kernel_block_size,
                max_num_blocks_per_req,
                slot_mapping_mode,
            ) in zip(
                block_sizes, kernel_block_sizes, max_num_blocks, slot_mapping_modes
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        # 为所有组追加一行块(块 id 为每组各一个列表)。
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        # 为所有组覆盖写入一行。
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        # 清空所有组的指定行。
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        # 移动所有组的行。
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        # 交换所有组的行。
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        # 为所有组计算 slot 映射。
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        # 提交所有组的块表到 GPU。
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        # 清空所有组的块表。
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        # 返回第 idx 个 KV cache 组对应的 BlockTable。
        return self.block_tables[idx]


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_kernel(
    # Triton kernel:批量计算 token -> KV 槽位映射。
    # 注意:num_tokens 与 max_num_tokens 不做特化(避免过多编译变体)。
    num_tokens,
    # token 总数。
    max_num_tokens,
    # 批最大 token 数。
    query_start_loc_ptr,  # [num_reqs + 1], int32
    # 各请求起始位置(长度 num_reqs+1,int32)。
    positions_ptr,  # [num_tokens], int64
    # token 位置(长度 num_tokens,int64)。
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    # 块表(展平 int32)。
    block_table_stride,  # max_num_blocks_per_req
    # 块表行跨步。
    block_size,
    # 计算块大小。
    slot_mapping_ptr,  # [max_num_tokens], int64
    # slot 映射输出缓冲。
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    # KV cache 分配块大小(编译期)。
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    # 每 KV 块的 kernel 块数(编译期)。
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    # DCP 世界大小(编译期)。
    TOTAL_CP_RANK: tl.constexpr,
    # 本 rank 在 DCP 中的编号(编译期)。
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    # CP 的 KV 交错粒度(编译期)。
    PAD_ID: tl.constexpr,
    # 填充槽位 id(编译期)。
    BLOCK_SIZE: tl.constexpr,
    # kernel 块处理元素数(编译期)。
):
    # 取程序 id 作为请求索引。
    req_idx = tl.program_id(0)

    # 若这是最后一个程序(用于填充):
    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        # 为 CUDA Graph 兼容性填充剩余槽位。
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            # 计算偏移向量。
            offsets = i + tl.arange(0, BLOCK_SIZE)
            # 写入 PAD_ID(超出 token 数的位置)。
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        # 填充完成返回。
        return

    # 取本请求起始 token 索引。
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    # 取本请求结束 token 索引。
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    # 虚拟块大小 = KV 块大小 × DCP 世界大小(虚拟块跨全部 DCP rank)。
    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    # 本请求在块表中的行偏移。
    row_offset = req_idx * block_table_stride
    # 以 BLOCK_SIZE 为步长遍历本请求的 token:
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        # 偏移向量。
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # 有效 mask。
        mask = offsets < end_idx
        # 加载位置。
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        # 计算虚拟块索引。
        virtual_block_indices = pos // virtual_block_size
        # 计算虚拟块内偏移。
        virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
        # 判断该 token 是否属于本 DCP rank(按交错粒度分配)。
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        # 计算本地块内偏移(考虑 DCP 分片与交错)。
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        # 计算 kernel 块索引。
        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        # 加载块编号(仅本 rank 的 token)。
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
        # 计算槽位内偏移。
        slot_offsets = local_block_offsets % block_size
        # 计算槽位 id = 块编号 × 块大小 + 槽位内偏移。
        slot_ids = block_numbers * block_size + slot_offsets
        # 非本 rank 的位置填 PAD_ID。
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        # 写入 slot 映射缓冲。
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)