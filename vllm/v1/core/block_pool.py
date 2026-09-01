# SPDX-License-Identifier: Apache-2.0  # SPDX 许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # SPDX 版权文本：vLLM 项目贡献者版权
from collections.abc import Iterable, Sequence  # 导入通用抽象基类：可迭代对象与序列
from typing import Any  # 导入类型工具：Any 任意类型

from vllm.distributed.kv_events import (  # 从 KV 事件模块导入（多行导入开始）
    MEDIUM_GPU,  # 事件介质：GPU
    AllBlocksCleared,  # 所有块被清除事件
    BlockRemoved,  # 块被移除事件
    BlockStored,  # 块被存储事件
    KVCacheEvent,  # KV 缓存事件基类
)
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector  # 导入 KV 缓存指标收集器
from vllm.v1.core.kv_cache_utils import (  # 从 KV 缓存工具模块导入（多行导入开始）
    BlockHash,  # 块哈希类型
    BlockHashWithGroupId,  # 带组 ID 的块哈希类型
    ExternalBlockHash,  # 外部块哈希类型（对外暴露用）
    FreeKVCacheBlockQueue,  # 空闲 KV 缓存块队列（双向链表）
    KVCacheBlock,  # KV 缓存块元数据类
    generate_block_hash_extra_keys,  # 生成块哈希附加键（多模态特征等）
    get_block_hash,  # 从带组 ID 哈希提取纯块哈希
    get_group_id,  # 从带组 ID 哈希提取组 ID
    make_block_hash_with_group_id,  # 构造带组 ID 的块哈希
    maybe_convert_block_hash,  # 条件转换块哈希为外部哈希
    resolve_block_hashes,  # 解析请求块哈希到目标块大小
)
from vllm.v1.request import Request  # 导入请求类

logger = init_logger(__name__)  # 初始化模块日志器


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """
    # 用于前缀缓存的块缓存：将块哈希直接映射到一个或多个块（{block_hash: KVCacheBlocks}）
    # - 大多数情况 block_hash 映射到单个 KVCacheBlock，此时 KVCacheBlocks 即该块本身
    # - 否则 KVCacheBlocks 是 {block_id: KVCacheBlock} 的字典
    # 缓存块是带块哈希的完整块，可用于前缀缓存。
    # 缓存块可能正被运行中的请求使用，或位于可能被驱逐的空闲块队列中。
    # 注意 #1：当前不去重缓存中的块——块变满并被缓存时不检查是否已有相同块，
    # 因为要保证分配的块 ID 不变，使块表只追加（append-only）。
    # 注意 #2：使用联合类型是为了减少内部 dict 带来的 GC 开销。

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}
        # 缓存字典：带组 ID 的块哈希 → 单个块 或 按 block_id 索引的块字典

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        # 根据块哈希键获取任意一个块
        blocks = self._cache.get(key)  # 从缓存取该键对应的块
        if blocks is not None:
            # 键存在
            if isinstance(blocks, KVCacheBlock):
                return blocks  # 单块：直接返回
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))  # 多块：返回其中任意一个
            self._unexpected_blocks_type(blocks)  # 类型异常则报错
        return None  # 未命中返回 None

    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        """
        Checks whether the key maps to the given block ID.
        """
        # 检查该键是否映射到指定块 ID
        blocks = self._cache.get(key)  # 取该键对应的块
        if blocks is None:
            return False  # 键不存在
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id  # 单块：比较块 ID
        if isinstance(blocks, dict):
            return block_id in blocks  # 多块：检查 block_id 是否在字典中
        self._unexpected_blocks_type(blocks)  # 类型异常则报错
        return False

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        # 将 KVCacheBlock 插入缓存
        blocks = self._cache.get(key)  # 取该键对应的已有块
        if blocks is None:
            # When key is not found, attach a single block to the key
            # 键未命中时，直接挂单个块到该键
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            # 同键已有单块：把原块和新块合并成一个字典
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            # 已是字典：直接插入新块
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)  # 类型异常则报错

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        # 检查块哈希是否存在，并从缓存弹出指定 block_id 的块
        blocks = self._cache.pop(key, None)  # 弹出并移除该键（默认 None）
        if blocks is None:
            # block_hash not found in the cache
            # 块哈希未在缓存中找到
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        # TODO(Jialin)：键找到时 block_id 应当始终存在于块中。
        # 当前为安全保留原行为。后续将添加
        # block_id == blocks.block_id 断言，改用 del blocks[block_id]。
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks  # 单块且 ID 匹配：返回该块
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            # 单块 ID 不匹配时把块放回缓存（此情况很少发生）
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            # 尝试从字典弹出 block_id；若字典仍含块则放回缓存
            block = blocks.pop(block_id, None)  # 弹出指定块（默认 None）
            if len(blocks) > 0:
                self._cache[key] = blocks  # 还有剩余块则放回
            return block
        self._unexpected_blocks_type(blocks)  # 类型异常则报错
        return None

    def __len__(self) -> int:
        return len(self._cache)  # 缓存条目数

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")
        # 抛出断言错误：意外的 KV 缓存块类型


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """
    # 管理 KVCacheBlocks 的块池。
    # 提供分配、释放、缓存 KV 缓存块的方法。
    # free_block_queue 按驱逐顺序存储空闲块，支持分配、释放与缓存驱逐。
    # cached_block_hash_to_block 在块哈希与缓存块之间映射，支持按哈希查找缓存块。
    # 参数：
    #   num_gpu_blocks: 池中块的总数。
    #   enable_caching: 是否启用前缀缓存。
    #   hash_block_size: 计算块哈希所用的块大小。
    #       实际块大小通常等于 hash_block_size，但当不同 KV 缓存组块大小不同时，
    #       实际块大小可以是 hash_block_size 的倍数。
    #   enable_kv_cache_events: 是否启用 KV 缓存事件。
    #   metrics_collector: 可选的指标收集器，用于跟踪块驻留情况。

    def __init__(
        self,
        num_gpu_blocks: int,  # 块池中块的总数
        enable_caching: bool,  # 是否启用前缀缓存
        hash_block_size: int,  # 计算块哈希的块大小
        enable_kv_cache_events: bool = False,  # 是否启用 KV 缓存事件（默认否）
        metrics_collector: KVCacheMetricsCollector | None = None,  # 可选的指标收集器
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0  # 断言块数合法且为正
        self.num_gpu_blocks = num_gpu_blocks  # 保存块总数
        self.enable_caching = enable_caching  # 保存前缀缓存开关
        self.hash_block_size = hash_block_size  # 保存哈希块大小
        # All kv-cache blocks.
        # 全部 KV 缓存块
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)  # 为每个索引创建块
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        # 空闲块队列：构建并操作空闲块的双向链表
        # （启用缓存时包含驱逐候选块）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        # 用于块查找的缓存
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()  # 块哈希 → 缓存块映射
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}
        # 反向映射：块 ID → 指向该块的所有哈希键集合（部分条目用）

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        # 用 block_id=0 的占位块表示空块。
        # null_block 的引用计数不维护，需特殊处理避免被释放
        self.null_block = self.free_block_queue.popleft()  # 从空闲队列取一个块作空块
        self.null_block.is_null = True  # 标记为空块

        self.enable_kv_cache_events = enable_kv_cache_events  # 保存 KV 缓存事件开关
        self.kv_event_queue: list[KVCacheEvent] = []  # KV 缓存事件队列

        self.metrics_collector = metrics_collector  # 保存指标收集器

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]  # 块哈希；KV 缓存组 ID 列表
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        # 按块哈希为每个组取缓存块；任一组未命中则返回 None。
        # 若有重复块，返回缓存中的第一个块。
        cached_blocks = []  # 缓存命中块列表
        for group_id in kv_cache_group_ids:  # 遍历各组
            block_hash_with_group_id = make_block_hash_with_group_id(  # 构造带组 ID 的哈希
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(  # 查该组的缓存块
                block_hash_with_group_id
            )
            if not block:
                return None  # 任一组未命中则整体未命中
            cached_blocks.append(block)  # 收集命中块
        return cached_blocks  # 返回各组的命中块

    def cache_full_blocks(
        self,
        request: Request,  # 待缓存块的请求
        blocks: list[KVCacheBlock],  # 请求的所有块
        num_cached_blocks: int,  # 已缓存的块数
        num_full_blocks: int,  # 本函数后应已满并缓存的块数
        block_size: int,  # 每块 token 数
        kv_cache_group_id: int,  # KV 缓存组 ID
        block_mask: list[bool] | None = None,  # 可选的掩码，与 blocks[num_cached_blocks:num_full_blocks] 对齐
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        # 缓存一批满块以支持前缀缓存。
        # 给定请求，更新每个块的块哈希元数据并存入 cached_block_hash_to_block。
        # 块哈希值在请求创建及追加新 token 时由 Request 对象即时计算。
        # block_mask：可选，与 blocks[num_cached_blocks:num_full_blocks] 对齐。
        # 掩码为 False 的块被跳过（当作空块）。用于 find_longest_cache_hit
        # 只查询部分块（如 SWA 尾窗口）的组，使永远无法命中的块不进前缀缓存哈希表
        if num_cached_blocks >= num_full_blocks:
            return  # 无需缓存（已全部缓存）
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]  # 待新缓存的满块
        assert block_mask is None or len(block_mask) == len(new_full_blocks)  # 断言掩码长度匹配
        block_hashes = resolve_block_hashes(  # 解析请求块哈希到目标块大小
            request.block_hashes, self.hash_block_size, block_size
        )

        new_block_hashes = block_hashes[num_cached_blocks:]  # 新缓存块的哈希（从已缓存数起）
        new_hashes: list[ExternalBlockHash] | None = (  # 新哈希外部表示列表（事件用）
            [] if self.enable_kv_cache_events else None  # 仅在启用事件时收集
        )
        for i, blk in enumerate(new_full_blocks):  # 遍历新满块
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            # 启用稀疏注意力（如滑动窗口注意力）或 Mamba align 模式前缀缓存时，
            # 部分块可能是空块或被掩码排除。这里跳过空块
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue  # 空块或掩码排除：跳过
            block_hash = new_block_hashes[i]  # 取该块哈希
            num_hash_tokens = (num_cached_blocks + i + 1) * block_size  # 该块覆盖的前缀 token 数

            # Update and added the full block to the cache.
            # 更新并把满块加入缓存
            block_hash_with_group_id = make_block_hash_with_group_id(  # 构造带组 ID 哈希
                block_hash, kv_cache_group_id
            )
            if blk.block_hash is not None:
                # The only valid case where a "new full block" already has a
                # hash is partial->full promotion of the same cache block.
                # “新满块”已有哈希的唯一合法场景：同一缓存块从部分条目晋升为满块
                assert (
                    blk.block_hash_num_tokens is not None  # 断言已有 token 数
                    and blk.block_hash_num_tokens < num_hash_tokens  # 且旧值小于新值（晋升）
                )
                removed_hashes = self._remove_cached_block_hashes(blk)  # 移除旧的缓存哈希
                self._emit_block_removed_events(removed_hashes)  # 发出块移除事件
            self._insert_block_hash(  # 插入新哈希键
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,  # 记录覆盖 token 数
            )
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))  # 收集外部哈希

        if self.enable_kv_cache_events:  # 启用事件时构造 BlockStored 事件
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None  # 无父块
            else:
                parent_block_hash = maybe_convert_block_hash(  # 父块哈希 = 前一块哈希
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            # 计算被缓存块的 token 范围
            start_token_idx = num_cached_blocks * block_size  # 起始 token 索引
            end_token_idx = num_full_blocks * block_size  # 结束 token 索引

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            # 为每个块单独生成附加键。
            # 每个块的附加键可能不同（如不同的多模态特征，或仅首块有 cache_salt）。
            # 跳过空块/掩码块，使列表长度与 new_hashes 一致
            extra_keys_list: list[tuple[Any, ...] | None] = []  # 附加键列表
            curr_mm_idx = 0  # 当前多模态索引
            for i in range(num_cached_blocks, num_full_blocks):  # 遍历待缓存块
                if blocks[i].is_null:
                    continue  # 空块跳过
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue  # 掩码排除跳过
                block_start = i * block_size  # 块起始 token 索引
                block_end = block_start + block_size  # 块结束 token 索引
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(  # 生成附加键并推进 mm 索引
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)  # 收集附加键

            self.kv_event_queue.append(  # 事件入队
                self._build_block_stored_event(  # 构造 BlockStored 事件
                    request,
                    block_hashes=new_hashes,  # 新哈希列表
                    parent_block_hash=parent_block_hash,  # 父块哈希
                    start_token_idx=start_token_idx,  # 起始 token 索引
                    end_token_idx=end_token_idx,  # 结束 token 索引
                    block_size=block_size,  # 块大小
                    kv_cache_group_id=kv_cache_group_id,  # 组 ID
                    extra_keys_list=extra_keys_list,  # 附加键列表
                )
            )

    def _build_block_stored_event(
        self,
        request: Request,  # 关联请求
        block_hashes: list[ExternalBlockHash] | None,  # 新块外部哈希（可为 None）
        parent_block_hash: ExternalBlockHash | None,  # 父块外部哈希（可为 None）
        start_token_idx: int,  # 起始 token 索引
        end_token_idx: int,  # 结束 token 索引
        block_size: int,  # 块大小
        kv_cache_group_id: int,  # KV 缓存组 ID
        extra_keys_list: list[tuple[Any, ...] | None],  # 附加键列表
    ) -> BlockStored:
        """Build a ``BlockStored`` KV event for ``request``.

        Shared by ``cache_full_blocks`` (newly cached blocks) and
        ``emit_cached_block_events`` (prefix-cache-reused blocks) so both emit
        identical event shapes for downstream consumers.
        """
        # 为请求构造 BlockStored KV 事件。
        # 供 cache_full_blocks（新缓存块）与 emit_cached_block_events
        # （前缀缓存复用块）共用，保证下游消费者收到一致的事件形状
        return BlockStored(
            block_hashes=block_hashes,  # 块外部哈希
            parent_block_hash=parent_block_hash,  # 父块哈希
            token_ids=request.all_token_ids[start_token_idx:end_token_idx],  # 对应 token 序列
            block_size=block_size,  # 块大小
            lora_id=request.lora_request.adapter_id if request.lora_request else None,  # LoRA 适配器 ID（如有）
            medium=MEDIUM_GPU,  # 介质：GPU
            lora_name=request.lora_request.name if request.lora_request else None,  # LoRA 名称（如有）
            extra_keys=extra_keys_list if extra_keys_list else None,  # 附加键（空则 None）
            group_idx=kv_cache_group_id,  # 组索引
        )

    def emit_cached_block_events(
        self,
        request: Request,  # 复用前缀缓存块的请求
        num_cached_blocks: int,  # 缓存命中块数
        block_size: int,  # 每块 token 数
        kv_cache_group_id: int,  # KV 缓存组 ID
    ) -> None:
        """Generate BlockStored events for blocks reused from prefix cache.

        Unlike cache_full_blocks(), this does NOT modify block state —
        the blocks are already cached. It only generates events so that
        external consumers (e.g. gateway) can learn about reused blocks.

        Args:
            request: The request whose prefix cache blocks were reused.
            num_cached_blocks: Number of blocks that were cache hits.
            block_size: Number of tokens per block.
            kv_cache_group_id: The KV cache group ID.
        """
        # 为从前缀缓存复用的块生成 BlockStored 事件。
        # 与 cache_full_blocks() 不同：本函数不改动块状态（块已缓存），
        # 仅生成事件供外部消费者（如网关）了解被复用的块
        if not self.enable_kv_cache_events or num_cached_blocks == 0:
            return  # 未启用事件或无命中块则返回

        block_hashes = resolve_block_hashes(  # 解析块哈希到目标块大小
            request.block_hashes, self.hash_block_size, block_size
        )

        # Collect external hashes and extra_keys for cached blocks.
        # 收集缓存块的外部哈希与附加键
        cached_hashes: list[ExternalBlockHash] = []  # 外部哈希列表
        extra_keys_list: list[tuple[Any, ...] | None] = []  # 附加键列表
        curr_mm_idx = 0  # 当前多模态索引
        for i in range(num_cached_blocks):  # 遍历命中块
            block_start = i * block_size  # 块起始 token 索引
            block_end = block_start + block_size  # 块结束 token 索引
            cached_hashes.append(maybe_convert_block_hash(block_hashes[i]))  # 转外部哈希
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(  # 生成附加键
                request, block_start, block_end, curr_mm_idx
            )
            extra_keys_list.append(extra_keys)  # 收集附加键

        if not cached_hashes:
            return  # 无命中哈希则返回

        # Prefix-cache hits always form a contiguous prefix starting at block 0,
        # so the first (and thus the whole group's) parent block hash is None.
        # 前缀缓存命中总是构成从块 0 开始的连续前缀，
        # 因此首块（从而整组）的父块哈希为 None
        parent_block_hash: ExternalBlockHash | None = None  # 父块哈希恒为 None
        start_token_idx = 0  # 起始 token 索引
        end_token_idx = num_cached_blocks * block_size  # 结束 token 索引

        logger.debug(  # 调试日志
            "EmitCachedBlock event: block_size=%d, "  # 块大小
            "num_cached_blocks=%d, parent_block_hash=%s, "  # 命中块数、父块哈希
            "token_ids_len=%d, group_idx=%s",  # token 序列长度、组索引
            block_size,
            num_cached_blocks,
            parent_block_hash,
            len(request.all_token_ids[start_token_idx:end_token_idx]),  # token 序列长度
            kv_cache_group_id,
        )

        self.kv_event_queue.append(  # 事件入队
            self._build_block_stored_event(  # 构造 BlockStored 事件
                request,
                block_hashes=cached_hashes,  # 命中块外部哈希
                parent_block_hash=parent_block_hash,  # 父块哈希（None）
                start_token_idx=start_token_idx,  # 起始 token 索引
                end_token_idx=end_token_idx,  # 结束 token 索引
                block_size=block_size,  # 块大小
                kv_cache_group_id=kv_cache_group_id,  # 组 ID
                extra_keys_list=extra_keys_list,  # 附加键列表
            )
        )

    def cache_partial_block(
        self,
        request: Request,  # 请求
        block: KVCacheBlock,  # 现有的缓存块
        num_tokens: int,  # 部分条目代表的前缀长度
        kv_cache_group_id: int,  # 拥有该部分条目的 KV 缓存组
        block_size: int,  # 所属组的缓存块大小
    ) -> BlockHashWithGroupId | None:
        """Register a partial prefix-cache entry for an existing block.

        Prefix-cache keys normally identify full cache blocks. A partial entry
        makes an existing cache block reachable from a fine-grained prefix
        boundary inside that block without allocating or copying a new
        ``KVCacheBlock``.

        The partial entry is lookup metadata owned by ``block``. If ``block``
        has no primary hash, the key becomes its primary hash. If the block
        already has a primary hash, the partial entry is tracked in
        ``cached_block_hashes_by_block`` so eviction, reset, and promotion can
        remove every hash key that points to the block.

        Args:
            request: Request whose token IDs and block hashes define the
                partial entry.
            block: Existing cache block to make reachable from the partial
                prefix boundary.
            num_tokens: Prefix length represented by the partial entry. It
                must be a positive multiple of ``self.hash_block_size`` and
                cannot exceed the request's computed block hashes.
            kv_cache_group_id: KV cache group that owns the partial entry.
            block_size: Cache block size for the owning group. The partial
                entry hash itself is always the prefix-chain hash at
                ``num_tokens``; ``block_size`` is used to assert that the
                entry is partial within the owning cache block.

        Returns:
            The hash key with group ID if a partial entry can be registered;
            otherwise ``None`` for null blocks.
        """
        # 为现有块注册一个部分前缀缓存条目。
        # 前缀缓存键通常标识完整缓存块；部分条目使已有缓存块可从块内
        # 的细粒度前缀边界被查找，无需分配或复制新的 KVCacheBlock。
        # 部分条目是 block 拥有的查找元数据：若块无主哈希则其成为主哈希；
        # 若已有主哈希则记录在 cached_block_hashes_by_block 中，
        # 使驱逐、重置、晋升能移除所有指向该块的哈希键。
        # num_tokens 必须是 hash_block_size 的正整数倍，且不超过请求的已计算块哈希数。
        # 返回：能注册时返回带组 ID 的哈希键；空块返回 None
        if block.is_null:
            return None  # 空块不可注册

        assert block_size > self.hash_block_size  # 断言块大小大于哈希块大小（部分条目不适用等大小）
        assert block_size % self.hash_block_size == 0  # 断言整除关系
        assert num_tokens % block_size != 0  # 断言确实为部分条目（未到整块边界）
        block_hash = self._get_partial_block_hash(request, num_tokens)  # 取部分条目的前缀链哈希
        num_hash_blocks = num_tokens // self.hash_block_size  # 哈希块数量
        block_hash_with_group_id = make_block_hash_with_group_id(  # 构造带组 ID 哈希
            block_hash, kv_cache_group_id
        )
        already_cached = block.block_hash == block_hash_with_group_id or (  # 是否已缓存
            self.cached_block_hash_to_block.contain(  # 或哈希表已含该键与该块
                block_hash_with_group_id, block.block_id
            )
        )
        if (
            not already_cached  # 未缓存
            and block.block_hash is not None  # 且块已有主哈希
            and block.block_hash_num_tokens is not None  # 且已有覆盖 token 数
            and block.block_hash_num_tokens < num_hash_blocks * self.hash_block_size  # 且旧条目较短（覆盖更短前缀）
        ):
            removed_hashes = self._remove_cached_block_hashes(block)  # 移除块的所有旧缓存哈希
            self._emit_block_removed_events(removed_hashes)  # 发出块移除事件
        self._insert_block_hash(  # 插入新部分条目
            block_hash_with_group_id,
            block,
            num_tokens=num_hash_blocks * self.hash_block_size,  # 记录覆盖 token 数
        )
        if self.enable_kv_cache_events and not already_cached:  # 启用事件且新条目
            parent_hash, block_start = self._get_partial_block_parent_hash_and_start(  # 取父哈希与起始索引
                request, num_tokens
            )
            parent_block_hash = (  # 父块外部哈希
                maybe_convert_block_hash(parent_hash)  # 有父哈希则转换
                if parent_hash is not None
                else None
            )
            block_end = num_tokens  # 块结束索引 = 部分前缀长度
            curr_mm_idx = -1 if block_start > 0 else 0  # 多模态索引（非首块时 -1）
            extra_keys, _ = generate_block_hash_extra_keys(  # 生成附加键
                request, block_start, block_end, curr_mm_idx
            )
            self.kv_event_queue.append(  # 事件入队
                BlockStored(
                    block_hashes=[maybe_convert_block_hash(block_hash)],  # 该块外部哈希
                    parent_block_hash=parent_block_hash,  # 父块哈希
                    token_ids=request.all_token_ids[block_start:block_end],  # 对应 token 序列
                    block_size=block_end - block_start,  # 事件块大小 = 部分长度
                    lora_id=request.lora_request.adapter_id  # LoRA 适配器 ID
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,  # 介质：GPU
                    lora_name=request.lora_request.name  # LoRA 名称
                    if request.lora_request
                    else None,
                    extra_keys=[extra_keys],  # 附加键
                    group_idx=kv_cache_group_id,  # 组索引
                )
            )
        return block_hash_with_group_id  # 返回新注册的哈希键

    def _get_partial_block_hash(
        self,
        request: Request,  # 请求
        num_tokens: int,  # 部分前缀长度
    ) -> BlockHash:
        assert num_tokens % self.hash_block_size == 0  # 断言长度是哈希块大小的整数倍
        num_hash_blocks = num_tokens // self.hash_block_size  # 哈希块数量
        assert 0 < num_hash_blocks <= len(request.block_hashes)  # 断言数量合法

        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        # 每个 hash_block_size 哈希链覆盖其完整前缀，
        # 因此任何组块大小的部分条目就是该前缀边界处的哈希
        return request.block_hashes[num_hash_blocks - 1]  # 返回边界处哈希

    def _get_partial_block_parent_hash_and_start(
        self,
        request: Request,  # 请求
        num_tokens: int,  # 部分前缀长度
    ) -> tuple[BlockHash | None, int]:
        num_hash_blocks = num_tokens // self.hash_block_size  # 哈希块数量
        parent_hash = (  # 父哈希 = 前一哈希块边界处哈希
            request.block_hashes[num_hash_blocks - 2] if num_hash_blocks > 1 else None  # 只有一个块则无父
        )
        block_start = (num_hash_blocks - 1) * self.hash_block_size  # 部分条目的起始 token 索引
        return parent_hash, block_start  # 返回（父哈希，起始索引）

    def _remove_cached_block_hashes(  # 移除块的所有缓存哈希键
        self,
        block: KVCacheBlock,  # 目标块
    ) -> list[BlockHashWithGroupId]:
        block_hashes: list[BlockHashWithGroupId] = []  # 收集该块的所有哈希键
        if block.block_hash is not None:
            block_hashes.append(block.block_hash)  # 主哈希
        block_hashes.extend(self.cached_block_hashes_by_block.pop(block.block_id, ()))  # 反向映射中的部分条目哈希
        if not block_hashes:
            return []  # 无哈希可移除
        removed_hashes: list[BlockHashWithGroupId] = []  # 实际被移除的哈希
        for block_hash in block_hashes:  # 遍历各哈希键
            if (  # 从缓存弹出，仅当确有其键时才计为移除
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ):
                removed_hashes.append(block_hash)  # 记录被移除哈希
        block.reset_hash()  # 重置块的哈希元数据
        return removed_hashes  # 返回被移除哈希列表

    def _emit_block_removed_events(  # 发出块移除事件
        self,
        block_hashes: list[BlockHashWithGroupId],  # 被移除的哈希键列表
    ) -> None:
        if not self.enable_kv_cache_events:
            return  # 未启用事件则返回
        for block_hash in block_hashes:  # 遍历被移除哈希
            self.kv_event_queue.append(  # 事件入队
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],  # 外部哈希
                    medium=MEDIUM_GPU,  # 介质：GPU
                    group_idx=get_group_id(block_hash),  # 组索引
                )
            )

    def _insert_block_hash(  # 将块哈希键插入缓存
        self,
        block_hash_with_group_id: BlockHashWithGroupId,  # 带组 ID 的块哈希
        block: KVCacheBlock,  # 目标块
        num_tokens: int | None,  # 覆盖的 token 数（可选）
    ) -> None:
        if block.block_hash == block_hash_with_group_id:
            return  # 主哈希相同则无需操作

        if self.cached_block_hash_to_block.contain(  # 哈希表已含该键与该块
            block_hash_with_group_id, block.block_id
        ):
            return  # 已存在则返回

        if block.block_hash is None:
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)  # 无主哈希：设为新主哈希
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(  # 有主哈希：记录到反向映射
                block_hash_with_group_id
            )
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)  # 插入哈希表

    def move_block_hashes(  # 将 src 块的条目重指向 dst 块
        self,
        src_block: KVCacheBlock,  # 源块
        dst_block: KVCacheBlock,  # 目标块
    ) -> None:
        """Re-point ``src_block``'s prefix-cache entries to ``dst_block``.

        Used when the request owning ``src_block`` keeps writing into it
        : the prefix cache holds a private copy (``dst_block``)
        under the same hashes instead. Entries stay live; no events emitted.
        """
        # 将 src_block 的前缀缓存条目重指向 dst_block。
        # 当拥有 src_block 的请求继续写入它时使用：前缀缓存改持有一个
        # 使用相同哈希的私有副本（dst_block）。条目保持有效，不发出事件
        assert dst_block.block_hash is None  # 断言目标块无主哈希
        assert dst_block.block_id not in self.cached_block_hashes_by_block  # 断言目标块不在反向映射中
        num_tokens = src_block.block_hash_num_tokens  # 源块覆盖 token 数
        for block_hash in self._remove_cached_block_hashes(src_block):  # 移除源块所有哈希并遍历
            # `num_tokens` only applies to the first (primary) insertion.
            # num_tokens 仅适用于首次（主哈希）插入
            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)  # 插入到目标块

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:  # 从空闲池获取新块
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        # 从空闲块池获取新块。注意：本函数不检查块缓存
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
            # 空闲块不足则抛错

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)  # 从空闲队列弹出 N 块

        # In order to only iterate the list once, we duplicated code a bit
        # 为只遍历一次列表，这里略微复制了代码
        if self.enable_caching:  # 启用缓存：取出时可能需驱逐缓存块
            for block in ret:  # 遍历新块
                self._maybe_evict_cached_block(block)  # 若是缓存块则驱逐
                assert block.ref_cnt == 0  # 断言引用计数为 0
                block.ref_cnt += 1  # 引用计数自增
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)  # 记录分配指标
        else:  # 未启用缓存
            for block in ret:  # 遍历新块
                assert block.ref_cnt == 0  # 断言引用计数为 0
                block.ref_cnt += 1  # 引用计数自增
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)  # 记录分配指标
        return ret  # 返回新块列表

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:  # 若块已缓存则驱逐
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # 若块已缓存在 cached_block_hash_to_block 中，重置其哈希元数据并从缓存驱逐
        # Clean up metrics tracking first to prevent leaks
        # 先清理指标跟踪以防泄漏
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)  # 记录驱逐指标

        evicted_hashes = self._remove_cached_block_hashes(block)  # 移除该块所有缓存哈希
        if not evicted_hashes:
            # The block doesn't have hash, eviction is not needed
            # 该块无哈希，无需驱逐
            return False

        self._emit_block_removed_events(evicted_hashes)  # 发出块移除事件
        return True  # 驱逐成功

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:  # 触碰块：引用计数 +1
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        # 触碰块使其引用计数 +1，并可能将其从空闲队列移除。
        # 用于另一请求命中相同前缀时
        for block in blocks:  # 遍历块
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            # ref_cnt=0 表示该块在空闲列表（即驱逐候选），需将其移除
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)  # 从空闲队列移除
            block.ref_cnt += 1  # 引用计数自增
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)  # 记录访问指标

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:  # 释放一批块
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # 释放一批块。块应按驱逐优先级排序，第一个块最先被驱逐
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        # 区分有哈希块（LRU 缓存）与无哈希块（永不匹配前缀缓存）
        blocks_with_hash = []  # 有哈希块列表
        blocks_without_hash = []  # 无哈希块列表
        for block in ordered_blocks:  # 遍历待释放块
            block.ref_cnt -= 1  # 引用计数自减
            if block.ref_cnt == 0 and not block.is_null:  # 归零且非空块：可回空闲池
                # When caching is disabled we always append for better
                # GPU cache locality from reusing recently used blocks
                # 未启用缓存时总是追加，以复用最近使用的块获得更好的 GPU 缓存局部性
                if block.block_hash is None and self.enable_caching:
                    blocks_without_hash.append(block)  # 无哈希（缓存开启）：归入无哈希组
                else:
                    blocks_with_hash.append(block)  # 有哈希或未启用缓存：归入有哈希组

        # Blocks without hash get evicted first - prepend them last to the tail
        # 无哈希块最先被驱逐——prepend 使它们排到队尾最后
        self.free_block_queue.prepend_n(blocks_without_hash)  # 无哈希块前插（最后被取走）
        self.free_block_queue.append_n(blocks_with_hash)  # 有哈希块追加（最先被取走）

    def evict_blocks(self, block_ids: set[int]) -> None:  # 按块 ID 从前缀缓存驱逐
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        # 按块 ID 从前缀缓存驱逐块。
        # 仅驱逐当前已缓存（有哈希）的块；ref_cnt > 0 的块不释放出块池，
        # 仅从前缀缓存哈希表驱逐
        for block_id in block_ids:  # 遍历块 ID
            assert block_id < len(self.blocks), (  # 断言 ID 合法
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "  # 非法块 ID 提示
                f"This indicates a bug in the KV connector - workers should "  # 可能为 KV 连接器缺陷
                f"only report block IDs that were allocated by the scheduler."  # worker 只应上报调度器分配的块
            )
            block = self.blocks[block_id]  # 取块
            self._maybe_evict_cached_block(block)  # 若已缓存则驱逐

    def reset_prefix_cache(self) -> bool:  # 重置前缀缓存
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        # 重置前缀缓存。可用于 RLHF 流程在权重更新后使前缀缓存失效，
        # 或用于基准测试时重置前缀缓存状态
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()  # 已用块数
        if num_used_blocks != 1:  # The null block is always marked as used  # 正常只应剩空块占用
            logger.warning(  # 警告：仍有块未释放
                "Failed to reset prefix cache because some "  # 重置失败原因
                "blocks (%d) are not freed yet",  # 未释放块数
                num_used_blocks - 1,
            )
            return False  # 返回失败

        # Remove all hashes so that no new blocks will hit.
        # 移除所有哈希，使新块不再命中
        self.cached_block_hash_to_block = BlockHashToBlockMap()  # 重建空哈希映射
        self.cached_block_hashes_by_block.clear()  # 清空反向映射

        # Remove all hashes from all blocks.
        # 移除所有块的哈希
        for block in self.blocks:  # 遍历所有块
            block.reset_hash()  # 重置块哈希

        if self.metrics_collector:
            self.metrics_collector.reset()  # 重置指标

        logger.info("Successfully reset prefix cache")  # 成功日志

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())  # 发出全部块已清除事件

        return True  # 返回成功

    def get_num_free_blocks(self) -> int:  # 获取空闲块数
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        # 获取池中空闲块数
        return self.free_block_queue.num_free_blocks  # 返回空闲队列计数

    def get_usage(self) -> float:  # 获取 KV 缓存使用率
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        # 获取 KV 缓存使用率（0.0 ~ 1.0）

        # Subtract 1 to account for null block.
        # 减去 1 以扣除空块
        total_gpu_blocks = self.num_gpu_blocks - 1  # 可用的总块数
        if not total_gpu_blocks:
            return 0  # 无可用块返回 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)  # 使用率 = 1 - 空闲占比

    def take_events(self) -> list[KVCacheEvent]:  # 取出并清空事件队列
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        # 原子地取出全部事件并清空队列
        if not self.enable_kv_cache_events:
            return []  # 未启用事件返回空列表
        events = self.kv_event_queue  # 取当前队列
        self.kv_event_queue = []  # 重置为空队列
        return events  # 返回事件列表
