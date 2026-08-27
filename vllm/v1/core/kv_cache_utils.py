# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX 文件版权声明：vLLM 项目贡献者
"""KV-Cache Utilities."""
# KV 缓存工具模块

import copy  # copy：浅拷贝/深拷贝
import hashlib  # hashlib：哈希算法（sha256 等）
import math  # math：数学函数（lcm/gcd 等）
import os  # os：操作系统接口（环境变量读取）
from collections import defaultdict  # defaultdict：带默认值字典
from collections.abc import Callable, Iterable, Iterator, Sequence
# Callable：可调用对象；Iterable：可迭代；Iterator：迭代器；Sequence：序列
from dataclasses import dataclass, replace  # 数据类；replace：创建修改字段的副本
from functools import partial  # partial：偏函数（绑定参数）
from typing import Any, NamedTuple, NewType, TypeAlias, cast, overload
# 类型工具：任意类型、命名元组、新类型、类型别名、类型转换、重载

from vllm import envs  # vllm 环境变量
from vllm.config import VllmConfig  # vLLM 全局配置
from vllm.logger import init_logger  # 日志初始化
from vllm.utils.hashing import sha256_cbor, xxhash_cbor  # CBOR 哈希函数
from vllm.utils.math_utils import cdiv, round_up  # 向上取整除法；向上取整
from vllm.utils.mem_utils import format_gib  # 字节数格式化为 GiB 字符串
from vllm.utils.torch_utils import get_dtype_size  # 获取 dtype 字节大小
from vllm.v1.kv_cache_interface import (
    AttentionSpec,  # 注意力 KV 缓存规格
    ChunkedLocalAttentionSpec,  # 分块局部注意力规格
    FullAttentionSpec,  # 全注意力规格
    HiddenStateCacheSpec,  # 隐藏状态缓存规格
    KVCacheConfig,  # KV 缓存配置
    KVCacheGroupSpec,  # KV 缓存组规格
    KVCacheSpec,  # KV 缓存规格（基类）
    KVCacheTensor,  # KV 缓存张量
    MambaSpec,  # Mamba/SSM 状态缓存规格
    MLAAttentionSpec,  # MLA（多头潜在注意力）规格
    SlidingWindowMLASpec,  # 滑动窗口 MLA 规格
    SlidingWindowSpec,  # 滑动窗口注意力规格
    UniformTypeKVCacheSpecs,  # 统一类型 KV 缓存规格集合
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry  # 规格注册表
from vllm.v1.request import Request  # 请求对象
from vllm.v1.utils import tensor_data  # 张量数据提取（哈希用）

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
# BlockHash 表示单个 KV 缓存块的哈希，用于前缀缓存。
# 将其作为 `bytes` 的独立类型有助于捕获传递原始字节串时的误用。
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
# `BlockHashWithGroupId` 将 `BlockHash` 与其 KV 缓存组 ID 组合。
# 以原始字节表示以求紧凑高效。下方辅助函数将块哈希与组 ID 打包/解包进该键。
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
# ExternalBlockHash 用于可复现的前缀缓存块哈希。
# 它是 `bytes` 与 `int` 的联合类型，以在默认改用 sha256 字节后保持向后兼容。
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int  # 块哈希；组 ID
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    # 将 `BlockHash` 与组 ID 打包进 `BlockHashWithGroupId`。
    # 组 ID 用 4 字节大端序编码并追加到块哈希字节后。
    # 此表示避免创建元组，同时仍可恢复两个组成部分。
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))
    # 返回 块哈希 + 4 字节组 ID 拼接结果


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    # 从 `BlockHashWithGroupId` 提取 `BlockHash`（去掉尾部 4 字节组 ID）
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    # 从 `BlockHashWithGroupId` 提取组 ID（取尾部 4 字节大端解码）
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    # 可能将块哈希字节转换为整数（KV 事件日志用）
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        # 未启用整数哈希
        return hash_bytes  # 原样返回字节
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)
    # 转大端整数并截断到 64 位


logger = init_logger(__name__)  # 模块级日志器

# The hash seed for the first block of any prefix block sequence.
# 任意前缀块序列首块的哈希种子。
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
# 使用随机值避免哈希碰撞；若设置了 PYTHONHASHSEED 环境变量则按需共享种子。
# 这与 Python 的 hash() 函数行为一致（未设置时也使用随机种子）。
#
# The function `init_none_hash` initializes this variable globally.
# 函数 `init_none_hash` 全局初始化该变量。
NONE_HASH: BlockHash  # 首块哈希种子（全局，运行前初始化）
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})  # CBOR 哈希函数集合


def init_none_hash(hash_fn: Callable[[Any], bytes]):  # 初始化 NONE_HASH
    global NONE_HASH  # 声明全局变量

    hash_seed = os.getenv("PYTHONHASHSEED")  # 读取哈希种子环境变量
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        # 未设置种子且使用 CBOR 哈希函数
        logger.warning(
            # 警告：CBOR 哈希在无固定种子时不具可复现性
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        # 未设置种子：使用随机 32 字节
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        # 已设置种子：用哈希函数处理种子值
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""
    # KV 缓存块元数据（slots 数据类，省内存）

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    # 块 ID，范围 0 到 num_gpu_blocks - 1。
    block_id: int
    # Reference count.
    # 引用计数。
    ref_cnt: int = 0
    # The hash key (block hash + group id) of the block, only available
    # when the block is full and cached.
    # 块的哈希键（块哈希 + 组 ID），仅当块已满并缓存时可用。
    _block_hash: BlockHashWithGroupId | None = None
    # Number of prefix tokens covered by _block_hash. For full blocks this is
    # the full block boundary; partial entries can end inside a cache block.
    # _block_hash 覆盖的前缀 token 数。满块为整块边界；部分条目可止于缓存块内部。
    _block_hash_num_tokens: int | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    # 用于构建空闲块双向链表。
    # 这两个属性只能由 FreeKVCacheBlockQueue 操作。
    prev_free_block: "KVCacheBlock | None" = None  # 前驱空闲块
    next_free_block: "KVCacheBlock | None" = None  # 后继空闲块

    # Whether the block is a null block that should never be cached.
    # 是否为永不缓存的空块。
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        # 只读属性：块哈希键
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        # 只读属性：哈希覆盖 token 数
        return self._block_hash_num_tokens

    def set_block_hash(
        self,
        block_hash: BlockHashWithGroupId,  # 新哈希键
        num_tokens: int | None = None,  # 覆盖 token 数（可选）
    ) -> None:
        assert self.block_hash is None and self._block_hash_num_tokens is None, (
            "The block already has a hash. This should not happen."
        )
        # 断言块尚未设置哈希，防止重复设置
        self._block_hash = block_hash  # 设置哈希键
        self._block_hash_num_tokens = num_tokens  # 设置覆盖 token 数

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        # 块被驱逐时重置哈希
        self._block_hash = None  # 清空哈希键
        self._block_hash_num_tokens = None  # 清空 token 数

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        # 用 block_id 而非 KVCacheBlock 对象，避免 __repr__ 递归调用。
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        # 前驱块 ID
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        # 后继块 ID
        return (
            # 构造可读字符串表示
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"_block_hash_num_tokens={self._block_hash_num_tokens}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


class KVCacheBlockCopy(NamedTuple):
    # KV 缓存块拷贝记录（CoW 用）
    src_block_id: int  # 源块 ID
    dst_block_id: int  # 目标块 ID


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """
    # 将 KVCacheBlock 列表组织为空闲块双向链表。
    # 不用 Python 内建 deque，而用此类，以支持 O(1) 移除队列中间块。
    # 为弥合与 C++ 实现 deque 的性能差距，此类操作链表时不分配任何 Python 对象，
    # 而是直接操作各块的 prev_free_block / next_free_block 属性。
    # 队列初始按块 ID 排序。块被分配再释放后，按驱逐顺序追加回队列：
    # 1. 最近最少使用（LRU）的块在队首。
    # 2. 最后访问时间相同（同一序列分配）时，哈希 token 更多（块链尾部）者在前。
    # 注：释放请求块时通过反转块顺序维护此序，该操作在本类之外。

    def __init__(self, blocks: list[KVCacheBlock]) -> None:  # 构造函数
        self.num_free_blocks = len(blocks)  # 空闲块数量

        # Initialize doubly links of consecutive blocks
        # 初始化相邻块的双向链接
        for i in range(self.num_free_blocks):
            # 遍历所有块
            if i > 0:
                # 非首块
                blocks[i].prev_free_block = blocks[i - 1]  # 前驱指向前一块
            if i < self.num_free_blocks - 1:
                # 非尾块
                blocks[i].next_free_block = blocks[i + 1]  # 后继指向后一块

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        # 创建伪头块与伪尾块，减少代码中的分支判断
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        # 实现保证伪头/伪尾永不被弹出，因此可安全假设队列中每个真实块
        # 都有前驱和后继。
        self.fake_free_list_head = KVCacheBlock(block_id=-1)  # 伪头块（ID=-1）
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)  # 伪尾块（ID=-1）
        if self.num_free_blocks > 0:
            # 有真实块
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            # 将伪头、伪尾分别连到首块和末块。
            self.fake_free_list_head.next_free_block = blocks[0]  # 伪头指向首块
            blocks[0].prev_free_block = self.fake_free_list_head  # 首块前驱指伪头
            self.fake_free_list_tail.prev_free_block = blocks[-1]  # 末块前驱指伪尾
            blocks[-1].next_free_block = self.fake_free_list_tail  # 伪尾后驱
        else:
            # For empty list, simply connect the fake head and tail.
            # 空列表：直接连接伪头与伪尾。
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail  # 伪头指向伪尾
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head  # 伪尾指回伪头

    def popleft(self) -> KVCacheBlock:  # 弹出队首空闲块
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        # 弹出队首空闲块并将 num_free_blocks 减 1。返回队首块。
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail  # 队列已空
            or self.fake_free_list_head.next_free_block is None  # 链表损坏
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            # 断言计数与链表同步
            raise ValueError("No free blocks available")  # 抛出：无空闲块

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block  # 取队首块

        if first_block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            # 若块来自空闲链表，此情况不应发生，表明调用方逻辑有 bug。
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )  # 抛出：无效块

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        # 将伪头连接到队首块的后继（即第二块或伪尾）。
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # Remove the block from the linked list.
        # 将该块移出链表。
        first_block.prev_free_block = first_block.next_free_block = None  # 清空链接

        self.num_free_blocks -= 1  # 空闲块数减一
        return first_block  # 返回队首块

    def popleft_n(self, n: int) -> list[KVCacheBlock]:  # 弹出前 n 个空闲块
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.
        """
        # 弹出前 n 个空闲块并将 num_free_blocks 减 n。返回 n 个空闲块的列表。
        if n == 0:
            # 数量为 0
            return []  # 返回空列表
        assert self.num_free_blocks >= n  # 断言空闲块充足
        self.num_free_blocks -= n  # 数量减 n

        curr_block = self.fake_free_list_head.next_free_block  # 从队首开始
        # Pop n blocks from the head of the list
        # 从队首弹出 n 个块
        ret = []  # 结果列表
        for _ in range(n):
            assert curr_block is not None  # 断言当前块存在
            ret.append(curr_block)  # 加入结果
            last_block = curr_block  # 记录已弹出的最后一块
            curr_block = curr_block.next_free_block  # 移到后继
            # Reset prev_free_block and next_free_block of all popped blocks
            # 清空所有弹出块的双向链接
            last_block.prev_free_block = None  # 清空前驱
            last_block.next_free_block = None  # 清空后继

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            # 队列未空：将伪头连接到新的队首块。
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret  # 返回弹出块列表

    def remove(self, block: KVCacheBlock) -> None:  # 从空闲链表移除指定块
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        # 移除空闲链表中的块并将 num_free_blocks 减 1。
        if block.prev_free_block is None or block.next_free_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            # 若块来自空闲链表，此情况不应发生，表明调用方逻辑有 bug。
            raise RuntimeError(f"remove() called on an invalid block: {block}")  # 抛出

        # Link the previous block to the next block.
        # 将前驱块链接到后继块。
        block.prev_free_block.next_free_block = block.next_free_block
        # Link the next block to the previous block.
        # 将后继块链接到前驱块。
        block.next_free_block.prev_free_block = block.prev_free_block

        # Remove the block from the linked list.
        # 将该块移出链表。
        block.prev_free_block = block.next_free_block = None  # 清空链接
        self.num_free_blocks -= 1  # 空闲块数减一

    def append(self, block: KVCacheBlock) -> None:  # 将块追加到队尾
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        # 将块放回空闲链表并将 num_free_blocks 加 1。
        if self.fake_free_list_tail.prev_free_block is None:
            # 伪尾前驱不应为 None
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )  # 抛出
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block  # 取当前队尾

        # Connect the new block after the last block.
        # 将新块连接到队尾之后。
        last_block.next_free_block = block  # 队尾后驱指向新块
        block.prev_free_block = last_block  # 新块前驱指队尾

        # Connect the fake tail after the new block.
        # 将伪尾连接到新块之后。
        block.next_free_block = self.fake_free_list_tail  # 新块后驱指伪尾
        self.fake_free_list_tail.prev_free_block = block  # 伪尾前驱指新块

        self.num_free_blocks += 1  # 空闲块数加一

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:  # 将块列表放到队首
        """Put a list of blocks at the front of the free list."""
        # 将一批块放到空闲链表队首。
        if len(blocks) == 0:
            # 空列表
            return  # 直接返回

        first_block = self.fake_free_list_head.next_free_block  # 原队首
        assert first_block is not None, (
            "next_free_block of fake_free_list_head should always exist"
        )  # 断言伪头有后继

        prev_block = self.fake_free_list_head  # 从伪头开始
        for block in blocks:  # 逐个插入
            block.prev_free_block = prev_block  # 新块前驱指前一块
            prev_block.next_free_block = block  # 前一块后驱指新块
            prev_block = block  # 前移

        prev_block.next_free_block = first_block  # 最后一块后驱指原队首
        first_block.prev_free_block = prev_block  # 原队首前驱指最后一块

        self.num_free_blocks += len(blocks)  # 空闲块数增加

    def append_n(self, blocks: list[KVCacheBlock]) -> None:  # 将块列表追加到队尾
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.
        """
        # 将一批块追加回空闲链表。
        if len(blocks) == 0:
            # 空列表
            return  # 直接返回

        last_block = self.fake_free_list_tail.prev_free_block  # 当前队尾
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )  # 断言伪尾有前驱
        # Add inter-connections between consecutive blocks
        # 建立块之间相互链接
        for block in blocks:  # 逐个追加
            block.prev_free_block = last_block  # 新块前驱指队尾
            last_block.next_free_block = block  # 队尾后驱指新块
            last_block = block  # 队尾前移

        # Connect the last block of <blocks> to the fake tail
        # 将 <blocks> 的最后一块连接到伪尾
        last_block.next_free_block = self.fake_free_list_tail  # 后驱指伪尾
        self.fake_free_list_tail.prev_free_block = last_block  # 伪尾前驱指最后一块

        self.num_free_blocks += len(blocks)  # 空闲块数增加

    def get_all_free_blocks(self) -> list[KVCacheBlock]:  # 获取全部空闲块（主要测试用）
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        # 获取空闲链表中的全部空闲块，主要用于测试。
        ret = []  # 结果列表
        if self.fake_free_list_head.next_free_block is None:
            # 伪头无后继
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )  # 抛出
        # Start from the first block
        # 从队首开始
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block  # 当前块
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        # 只要 next_free_block 存在，就还没到达伪尾。
        while curr_block.next_free_block is not None:
            # 未到伪尾
            ret.append(curr_block)  # 加入结果
            curr_block = curr_block.next_free_block  # 前移
        return ret  # 返回所有空闲块

    def iter_blocks_after(
        self,
        cursor: KVCacheBlock | None,  # 游标块（None 表示从头开始）
    ) -> Iterator[KVCacheBlock]:
        """Iterate free blocks in eviction order after the cursor."""
        # 从游标之后按驱逐顺序迭代空闲块。
        if cursor is None:
            # 无游标
            curr_block = self.fake_free_list_head.next_free_block  # 从队首开始
        else:
            # 有游标
            curr_block = cursor.next_free_block  # 从游标后继开始

        while curr_block is not None and curr_block is not self.fake_free_list_tail:
            # 未到链表尾
            yield curr_block  # 产出当前块
            curr_block = curr_block.next_free_block  # 前移


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
    # 请求；块起始 token 索引；块结束 token 索引；起始多模态索引
) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    # 为块哈希计算生成多模态请求相关的额外键。
    # 对多模态输入，额外键为 (mm_hash, start_offset)，表示块内含的多模态
    # 输入及其在块 token 中的起始偏移。返回（额外键列表，下一多模态索引）。
    extra_keys: list[Any] = []  # 额外键列表

    mm_features = request.mm_features  # 请求的多模态特征
    if not mm_features:
        # 无多模态特征
        return extra_keys, start_mm_idx  # 直接返回

    # Note that we assume mm_features are sorted by mm_position.offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    # 假设 mm_features 已按 mm_position.offset 排序。
    # 若起始 token 索引超出范围则无需检查所有多模态输入。
    # 这通常发生在 prefill 后期和解码阶段。
    last_pos = mm_features[-1].mm_position  # 最后一个多模态位置
    if last_pos.offset + last_pos.length <= start_token_idx:
        # 最后一个多模态已完全在当前块之前
        return extra_keys, start_mm_idx  # 直接返回

    # Support start_mm_idx == -1 to indicate the last mm input.
    # 支持 start_mm_idx == -1 表示最后一个多模态输入。
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)  # 断言索引合法
        start_mm_idx = len(mm_features) + start_mm_idx  # 转为正索引

    curr_mm_idx = start_mm_idx  # 当前多模态索引
    while mm_features and curr_mm_idx < len(mm_features):
        # 遍历多模态特征
        mm_feature = mm_features[curr_mm_idx]  # 当前特征
        assert mm_feature.identifier is not None  # 断言有标识符
        offset = mm_feature.mm_position.offset  # 偏移
        length = mm_feature.mm_position.length  # 长度
        if end_token_idx > offset:
            # 块结束位置在多模态偏移之后
            if start_token_idx >= offset + length:
                # This block has passed the current mm input.
                # 当前块已越过该多模态输入。
                curr_mm_idx += 1  # 移到下一个
                continue

            # The block contains the current mm input. Include its offset
            # relative to the start of the block so prefix-cache keys stay
            # distinct when the same MM item appears at different positions
            # within otherwise-identical placeholder blocks.
            # 块包含当前多模态输入。加入其相对块起始的偏移，
            # 使同一多模态项在相同占位块的不同位置时前缀缓存键保持可区分。
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                # 若块包含当前多模态输入的末尾，移到下一个多模态输入，
                # 因为块可能还包含下一个多模态输入。
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                # 否则本块的多模态输入处理完毕。
                break
        else:
            # This block has not reached the current mm input.
            # 当前块尚未到达该多模态输入。
            break
    return extra_keys, curr_mm_idx  # 返回额外键与下一多模态索引


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA name of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    # 为块哈希计算生成 LoRA 相关额外键。
    # 若为 LoRA 请求返回 LoRA 名称，否则返回空列表。
    if not request.lora_request:
        # 非 LoRA 请求
        return []  # 返回空列表
    return [request.lora_request.lora_name]  # 返回 LoRA 名称


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
    # 请求；块起始 token 索引；块结束 token 索引
) -> list[bytes]:
    """Generate extra keys related to prompt embeds for block hash computation.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.

    Returns:
        Return a stable hash of the block prompt embeddings if prompt embeds
        are present. Return empty list otherwise.
    """
    # 为块哈希计算生成 prompt 嵌入相关额外键。
    # 若存在 prompt 嵌入，返回其稳定哈希；否则返回空列表。
    if request.prompt_embeds is None:
        # 无 prompt 嵌入
        return []  # 返回空列表
    block_range = (start_token_idx, end_token_idx)  # 块 token 范围
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    # 查询缓存的块嵌入哈希
    if embeds_hash is None:
        # 未缓存
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # 取该块范围的 prompt 嵌入
        # Hash prompt embeds once per block and cache on request
        # 每块只哈希一次并缓存在请求上
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        # 用 sha256 计算嵌入哈希
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
        # 缓存结果
    return [embeds_hash]  # 返回哈希


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
    # 请求；块起始 token 索引；块结束 token 索引；起始多模态索引
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs, request specific metadata (e.g., LoRA names), and
    hashed data from prompt embeddings.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    # 生成块哈希的额外键。额外键可来自多模态输入、请求特定元数据
    # （如 LoRA 名称）以及 prompt 嵌入的哈希数据。
    # 返回（额外键元组，下一多模态索引）。
    mm_extra_keys: list[Any]  # 多模态额外键
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    # 生成多模态额外键
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)  # LoRA 额外键
    cache_salt_keys: list[str] = (
        # 缓存盐额外键（仅首块且设置了盐时）
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )
    # 生成 prompt 嵌入额外键

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )
    # 合并所有额外键

    if not extra_keys:
        # 无额外键
        return None, new_start_mm_idx  # 返回 None

    return tuple(extra_keys), new_start_mm_idx  # 返回元组与下一多模态索引


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],  # 哈希函数
    parent_block_hash: BlockHash | None,  # 父块哈希（首块为 None）
    curr_block_token_ids: Sequence[int],  # 当前块 token id 序列
    extra_keys: tuple[Any, ...] | None = None,  # 额外键（可选）
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.
    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.
    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    # 计算块内容及前驱块内容的哈希值。哈希值用于前缀缓存。
    # 用 LRU 缓存避免对相同块内容重复计算。
    # 参数：hash_function 哈希函数；parent_block_hash 父块哈希（首块为 None）；
    # curr_block_token_ids 当前块 token id 列表（假设为满块）；
    # extra_keys 额外键。返回块哈希值，整个元组作为块哈希键。
    if not parent_block_hash:
        # 无父块哈希（首块）
        parent_block_hash = NONE_HASH  # 使用全局种子哈希

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)  # 转元组
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
    # 对（父哈希、当前块 token、额外键）整体做哈希


def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,  # KV 缓存配置
    vllm_config: VllmConfig,  # 全局配置
) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler (e.g. for ``num_computed_tokens`` rounding). Single group:
      ``cache_config.block_size * dcp``. Multiple groups: LCM of every
      group's effective block size. Attention groups are scaled by DCP;
      Mamba groups keep their full per-rank state and are not scaled.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.prefix_match_unit`` override if set, else the GCD of
      group block sizes; every group's block size must be divisible by it.
      Returns the scheduler block size (i.e. disables finer hashing) if block
      hashing is inactive or a mamba group's block size diverges from the
      cache block size (mamba_cache_mode != "align").
    """
    # 解析（调度器块大小、哈希块大小）。
    # - scheduler_block_size 是调度器使用的 token 对齐不变量（如 num_computed_tokens 取整）。
    #   单组：cache_config.block_size * dcp；多组：各组有效块大小的最小公倍数。
    #   注意力组按 DCP 缩放；Mamba 组保持完整每 rank 状态不缩放。
    # - hash_block_size 是 Request.block_hashes 的计算粒度。单组：等于调度器块大小。
    #   多组：优先 cache_config.prefix_match_unit，否则取各组块大小的最大公约数；
    #   每个组的块大小必须能被其整除。若块哈希未启用，或 Mamba 组块大小偏离
    #   缓存块大小（mamba_cache_mode != "align"），则返回调度器块大小（即禁用更细哈希）。
    cache_config = vllm_config.cache_config  # 缓存配置
    dcp = vllm_config.parallel_config.decode_context_parallel_size  # 解码上下文并行大小
    groups = kv_cache_config.kv_cache_groups  # KV 缓存组列表

    if len(groups) <= 1:
        # 单组
        bs = cache_config.block_size * dcp  # 有效块大小
        return bs, bs  # 调度块大小 = 哈希块大小

    group_block_sizes = [
        # 各组有效块大小
        g.kv_cache_spec.block_size * dcp  # 注意力组按 DCP 缩放
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size  # Mamba 组不缩放
        for g in groups
    ]
    scheduler_block_size = math.lcm(*group_block_sizes)  # 最小公倍数作为调度块大小

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    # 块哈希仅被前缀缓存和 KV 连接器（P/D、卸载）消费；两者都未启用时，
    # 保持 hash_block_size 等于调度器块大小。
    connector_enabled = vllm_config.kv_transfer_config is not None  # 是否启用 KV 连接器
    if not (cache_config.enable_prefix_caching or connector_enabled):
        # 前缀缓存和连接器都未启用
        return scheduler_block_size, scheduler_block_size

    if any(
        # 存在块大小偏离缓存块大小的 Mamba 组
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_cache_mode != "align"
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size  # 禁用更细哈希

    requested = cache_config.prefix_match_unit  # 用户指定前缀匹配单元
    hash_block_size = (
        requested if requested is not None else math.gcd(*group_block_sizes)
    )
    # 哈希块大小：优先用户指定，否则各组块大小最大公约数
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        # 存在组块大小不能被哈希块大小整除
        raise ValueError(
            f"Invalid prefix_match_unit={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by prefix_match_unit. "
            f"Got group block sizes={group_block_sizes}."
        )
        # 抛出：无效的 prefix_match_unit，所有组块大小必须能整除它
    return scheduler_block_size, hash_block_size  # 返回两个块大小


def get_request_block_hasher(
    hash_block_size: int,  # 哈希块大小
    caching_hash_fn: Callable[[Any], bytes],  # 缓存哈希函数
) -> Callable[[Request], list[BlockHash]]:
    """
    Returns a function which computes the list of un-computed block hashes
    of a request.

    Hashes are computed at ``hash_block_size`` granularity and chained over the
    full prefix, so each hash uniquely fingerprints the prefix ending at its
    boundary. Coarser group block sizes and partial-cache boundaries reuse
    these hashes directly (see ``BlockHashListWithBlockSize``).
    """
    # 返回计算请求未计算块哈希列表的函数。
    # 哈希按 hash_block_size 粒度计算并在整个前缀上链式传递，
    # 因此每个哈希唯一标识到其边界为止的前缀。更粗的组块大小和
    # 部分缓存边界直接复用这些哈希（见 BlockHashListWithBlockSize）。

    def request_block_hasher(request: Request) -> list[BlockHash]:  # 内部哈希函数
        start_token_idx = len(request.block_hashes) * hash_block_size
        # 起始 token 索引 = 已有块哈希数 * 哈希块大小
        num_tokens = request.num_tokens  # 请求 token 总数

        if start_token_idx + hash_block_size > num_tokens:
            # Early stop when there no new full blocks created.
            # 没有新的满块可创建时提前停止。
            return []  # 返回空列表

        curr_mm_idx = 0  # 当前多模态索引
        if start_token_idx > 0:
            # Set curr_mm_idx = -1 to indicate the last mm input.
            # Note that since we reach to this branch only when the block is
            # completed with generated tokens, we only need to consider the
            # last mm input.
            # 设 curr_mm_idx = -1 表示最后一个多模态输入。
            # 因仅在块由生成 token 填满时才进入此分支，只需考虑最后一个多模态输入。
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        # 前一块哈希值（无则 None）
        new_block_hashes: list[BlockHash] = []  # 新块哈希列表
        while True:
            # 循环直到不足一个满块
            end_token_idx = start_token_idx + hash_block_size  # 块结束索引
            if end_token_idx > num_tokens:
                # We only hash full blocks
                # 只哈希满块
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            # 多模态和 LoRA 请求需要额外键计算块哈希。
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )
            # 生成额外键

            # Compute the hash of the current block
            # 计算当前块哈希
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            # 取块内 token
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )
            # 计算链式哈希

            new_block_hashes.append(block_hash)  # 加入结果
            start_token_idx += hash_block_size  # 前移
            prev_block_hash_value = block_hash  # 更新父哈希

        return new_block_hashes  # 返回新块哈希列表

    return request_block_hasher  # 返回哈希函数


def _check_enough_kv_cache_memory(
    available_memory: int,  # 可用内存（字节）
    get_needed_memory: Callable[[], int],  # 获取所需内存函数
    max_model_len: int,  # 最大模型长度
    estimate_max_model_len: Callable[[int], int],  # 估算最大模型长度函数
):
    # 校验可用内存是否足够
    if available_memory <= 0:
        # 无可用内存
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )
        # 抛出：无可用内存用于缓存块。建议提高 gpu_memory_utilization 或
        # 参考 conserving_memory 文档

    needed_memory = get_needed_memory()  # 计算所需内存

    if needed_memory > available_memory:
        # 所需内存超过可用
        estimated_max_len = estimate_max_model_len(available_memory)  # 估算可支持长度
        estimated_msg = ""  # 估算消息
        if estimated_max_len > 0:
            # 有有效估算
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )
            # 说明估算出的最大模型长度

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )
        # 抛出：KV 缓存内存不足，建议提高 gpu_memory_utilization 或减小 max_model_len


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]  # 配置；规格集合
) -> int:
    """
    Get the maximum memory usage in bytes for the given KV cache specs.
    """
    # 获取给定 KV 缓存规格的最大内存用量（字节）。
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)
    # 对所有规格的每层最大内存求和


def estimate_max_model_len(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_spec: dict[str, KVCacheSpec],  # 每注意力层的规格
    available_memory: int,  # 可用内存（字节）
) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """
    # 用二分搜索估算可用内存可容纳的最大模型长度。
    # 估算期间临时修改 max_model_len，返回前恢复原值，确保无副作用。
    # 返回可容纳的最大模型长度。
    # Save the original max_model_len to restore after estimation
    # 保存原始 max_model_len 以便估算后恢复
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    # 定义检查给定模型长度是否适合内存的函数
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        # 临时修改 max_model_len 用于计算
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        # 计算给定模型长度所需内存
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory  # 是否满足

    try:
        # Binary search for the maximum model length
        # 二分搜索最大模型长度
        left, right = 1, original_max_model_len  # 搜索区间

        # If even the smallest model length doesn't fit, return 0
        # 即使最小模型长度也放不下则返回 0
        if not fits_in_memory(left):
            return 0  # 返回 0

        # Binary search for the maximum model length that fits
        # 二分搜索可容纳的最大模型长度
        result = 1  # 结果初始为 1
        while left <= right:
            # 二分循环
            mid = (left + right) // 2  # 中点
            if fits_in_memory(mid):
                # 中点可容纳
                result = mid  # 更新结果
                left = mid + 1  # 尝试更大
            else:
                # 中点不可容纳
                right = mid - 1  # 尝试更小
        return result  # 返回结果
    finally:
        # Always restore the original max_model_len to avoid side effects
        # 总是恢复原始 max_model_len 避免副作用
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_spec: dict[str, KVCacheSpec],  # 每注意力层规格
    available_memory: int,  # 可用内存（字节）
):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """
    # 检查 available_memory 是否足够 KV 缓存容纳至少一个 max_model_len 的请求。
    # 若 KV 缓存内存不足则抛出 ValueError。

    # No need to check for available memory if the kv_cache_spec is empty
    # kv_cache_spec 为空时无需检查可用内存
    if kv_cache_spec:
        # 非空规格
        _check_enough_kv_cache_memory(
            available_memory,  # 可用内存
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            # 所需内存 lambda
            vllm_config.model_config.max_model_len,  # 最大模型长度
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
            # 估算最大长度 lambda
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
    # 层名到规格的映射；分组后的层名列表（每组一个列表）
) -> list[KVCacheGroupSpec]:
    """
    Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same
    KVCacheSpec.

    Args:
        kv_cache_spec:
            A mapping from each layer name to its corresponding KVCacheSpec.
        grouped_layer_names:
            A list of kv cache groups, where each element is a list of layer
            names that belong to the same group and should share the same
            KVCacheSpec.
    Returns:
        A list of KVCacheGroupSpec objects, one for each group.
    """
    # 为每个 KV 缓存组创建 KVCacheGroupSpec 对象。
    # 同一组内的层应共享相同 KVCacheSpec。
    # 返回每个组一个 KVCacheGroupSpec 对象的列表。
    kv_cache_groups = []  # 组规格列表
    for layer_names_one_group in grouped_layer_names:
        # 遍历每组层名
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        # 取组内各层的规格
        merged_layer_spec = layer_specs[0].merge(layer_specs)  # 合并组内规格
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
        # 创建组规格并加入列表
    return kv_cache_groups  # 返回组规格列表


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same KV cache spec.
    Note that we regard FullAttentionSpec with and without sliding window as
    the same type.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """
    # 给定 KVCacheSpec 中所有层是否具有相同 KV 缓存规格。
    # 注意：带滑动窗口与不带滑动窗口的 FullAttentionSpec 视为同类型。
    # 所有层同类型返回 True，否则 False。

    if not kv_cache_spec:
        # Encoder-only models do not have KV cache, kv_cache_type can be
        # regarded as uniform.
        # 仅编码器模型没有 KV 缓存，可视为均匀类型。
        return True  # 返回 True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())  # 所有规格值
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)  # 尝试合并
    except AssertionError:
        # 合并失败（规格不兼容）
        return False  # 返回 False
    return True  # 合并成功返回 True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig  # 全局配置；KV 缓存配置
) -> float:
    """
    Get the maximum concurrency for the given KV cache configuration.

    A request at max_model_len consumes whole blocks from each group's block
    table — cdiv(per-request bytes, page bytes) of the group's spec — and all
    groups draw those block ids from one shared pool, so the per-request
    total is the sum over groups. The memory/page ratio is identical whether
    a group carries an aggregated UniformTypeKVCacheSpecs (worker config) or
    a representative per-layer spec (scheduler config), so both capacity
    call sites agree.
    """
    # 获取给定 KV 缓存配置的最大并发度。
    # max_model_len 的请求从每组块表消耗整块——组的规格的
    # cdiv(每请求字节数, 页字节数)——所有组从同一共享池取块 id，
    # 因此每请求总量为各组之和。组携带聚合的 UniformTypeKVCacheSpecs
    #（worker 配置）或代表性每层规格（调度器配置）时，内存/页比例一致，
    # 因此两个容量调用点结果一致。
    num_blocks_per_request = sum(
        # 每请求块数 = 各组之和
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),  # 组最大内存
            group.kv_cache_spec.page_size_bytes,  # 组页字节数
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request  # 并发度
    return max_concurrency  # 返回并发度


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    # 若设置了 `num_gpu_blocks_override` 则覆盖 KV 缓存块数。
    # 覆盖日志在 `get_kv_cache_configs` 调用点仅记录一次。
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        # 设置了覆盖值
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override  # 采用覆盖值
    return num_blocks  # 返回块数


def _pool_bytes_per_block(
    vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]  # 配置；组列表
) -> int:
    """
    Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    # worker 共享 KV 缓存池中一个块消耗的字节数，镜像
    # `get_kv_cache_config_from_groups` 将 `available_memory` 转换为
    # `num_blocks` 所用的除数。用于在应用 `num_gpu_blocks_override` 后
    # 计算有效 KV 缓存容量。
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # 单组 UniformTypeKVCacheSpecs 特例
        return kv_cache_groups[0].kv_cache_spec.page_size_bytes  # 直接返回页字节数
    if _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # 使用打包布局
        block_stride, _ = _get_packed_kv_cache_layout(kv_cache_groups)  # 块步长
        return block_stride  # 返回块步长
    group_size = max(len(g.layer_names) for g in kv_cache_groups)  # 最大组层数
    page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
    # 统一页字节数
    return page_size * group_size  # 页字节数 * 组大小


def get_num_blocks(
    vllm_config: VllmConfig,  # 全局配置
    num_layers: int,  # 层数
    available_memory: int,  # 可用内存（字节）
    page_size: int,  # 页大小
) -> int:
    """
    Get the number of kv cache blocks.

    Args:
        vllm_config: The global VllmConfig
        num_layers: The number of layers
        available_memory: Memory available for KV cache in bytes.
        page_size: The page size of the KV cache.
    """
    # 获取 KV 缓存块数。
    num_blocks = int(available_memory // page_size // num_layers)  # 按层分摊块数
    num_blocks = max(num_blocks, 0)  # 不为负
    return may_override_num_blocks(vllm_config, num_blocks)  # 应用覆盖


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """
    Get the page size of the KV cache.
    """
    # 获取 KV 缓存页大小（要求所有层页大小一致）
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}  # 页大小集合
    assert len(page_sizes) == 1  # 断言只有一个页大小
    return page_sizes.pop()  # 弹出唯一值


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],  # 每层规格
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with the same KV cache
    spec for all layers.

    Args:
        kv_cache_specs: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroupSpecs
    """
    # 为所有层具有相同 KV 缓存规格的模型生成 KV 缓存配置。
    # 返回生成的 KVCacheGroupSpecs。

    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])
    # 所有层放入一个组并创建组规格


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,  # 统一类型规格
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache configuration for a model with one type of KV cache
    but different hidden sizes. All layers are merged into one group.

    Args:
        spec: The UniformTypeKVCacheSpecs of the model

    Returns:
        The generated KVCacheGroupSpecs
    """
    # 为具有一种 KV 缓存类型但隐藏大小不同的模型生成 KV 缓存配置。
    # 所有层合并到一个组。返回生成的 KVCacheGroupSpecs。

    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]
    # 单组：全部层名 + 统一类型规格


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],  # 每层规格
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Two
    cases cannot be unified by block size alone and pad their physical page to
    the maximum instead: Mamba layers, whose page size comes from state shapes
    and is independent of block size; and attention layers whose page does not
    evenly divide the maximum and whose backend opts in via
    ``AttentionSpec.indexes_kv_by_block_stride`` (the padded page is read through
    a strided view, which not every backend handles). Raise NotImplementedError
    if failed to unify the page size.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model

    Returns:
        The updated KVCacheSpec with the same page_size_bytes.
    """
    # 统一给定 KVCacheSpec 的页大小。若所有层页大小相同则返回原规格。
    # 否则通过增大较小页大小层的块大小来统一。两种情况无法仅靠块大小统一，
    # 需将物理页填充到最大值：
    # - Mamba 层：页大小来自状态形状，与块大小无关；
    # - 注意力层：页不能整除最大值且后端通过
    #   ``AttentionSpec.indexes_kv_by_block_stride`` 选择支持（填充页通过
    #   步长视图读取，并非所有后端都支持）。
    # 统一失败则抛 NotImplementedError。返回统一页大小后的规格。
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    # 所有层页大小集合
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        # 所有层页大小相同，无需统一。
        return kv_cache_spec  # 原样返回

    max_page_size = max(page_sizes)  # 最大页大小
    new_kv_cache_spec = {}  # 新规格字典
    for layer_name, layer_spec in kv_cache_spec.items():
        # 遍历每层
        if layer_spec.page_size_bytes == max_page_size:
            # 已等于最大页大小
            new_kv_cache_spec[layer_name] = layer_spec  # 原样保留
        elif isinstance(layer_spec, MambaSpec):
            # MambaSpec's page size is determined by its state shapes and does
            # not scale with block_size, so pad the page instead. This is the
            # same padding mechanism the platform uses to align Mamba pages
            # with the main model's attention page size; it is needed here
            # when another layer (e.g. from a draft model) has a larger page
            # than the already-aligned Mamba page.
            # MambaSpec 的页大小由状态形状决定，不随 block_size 缩放，因此填充页。
            # 这是平台用于将 Mamba 页与主模型注意力页对齐的同一填充机制；
            # 当另一层（如来自草稿模型）的页大于已对齐的 Mamba 页时需要此处理。
            new_spec: KVCacheSpec = replace(layer_spec, page_size_padded=max_page_size)
            # 填充页大小到最大值
            assert new_spec.page_size_bytes == max_page_size  # 断言页大小正确
            new_kv_cache_spec[layer_name] = new_spec  # 存入新规格
        else:
            layer_page_size = layer_spec.page_size_bytes  # 当前层页大小
            if max_page_size % layer_page_size == 0:
                # 最大页能整除当前页
                ratio = max_page_size // layer_page_size  # 放大比例
                new_block_size = layer_spec.block_size * ratio  # 放大块大小
                new_spec = replace(layer_spec, block_size=new_block_size)  # 替换
            elif (
                isinstance(layer_spec, AttentionSpec)
                and layer_spec.indexes_kv_by_block_stride
            ):
                # 注意力层且支持按块步长索引
                new_spec = replace(layer_spec, page_size_padded=max_page_size)  # 填充
            else:
                # 无法统一
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for attention layers whose backend indexes KV "
                    "pages by the block stride (indexes_kv_by_block_stride is "
                    "True)."
                )
                # 抛出：层页大小不可整除且不能填充
            assert new_spec.page_size_bytes == max_page_size  # 断言页大小正确
            new_kv_cache_spec[layer_name] = new_spec  # 存入新规格
    return new_kv_cache_spec  # 返回统一后的规格


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    # kv_cache_spec is an empty dict for attention free models
    # 无注意力模型（attention free）的 kv_cache_spec 为空字典
    return not kv_cache_spec  # 空字典即无注意力模型


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],  # 每层规格
) -> list[KVCacheGroupSpec]:
    """
    Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers. It is already handled by
    `_get_kv_cache_config_uniform_type`.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    attention type. One exception is that, when
    `--disable-hybrid-kv-cache-manager` is true, the single group for full
    attention layers may also include attention layers using sliding window or
    LLaMA 4 local attention. See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        kv_cache_spec: The KVCacheSpec of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs
    """
    # 为混合模型生成 KV 缓存组：多种注意力类型但所有层页大小统一。
    # 混合模型 KV 缓存管理详解：
    # 模型层按某种模式重复。KVCacheManager 为模式中每层分配不同块表，
    # 并各自重复以生成整模型的块表。因此可将层分成多个 kv_cache_groups，
    # 每组含若干层。
    # 实现简化假设：
    # 1. 每块物理内存：所有组必须相同（不同大小块分配有内存碎片顾虑）。
    # 2. 每块 token 数（block_size）：目前所有层直接用 CacheConfig.block_size，
    #    可扩展为按组变化，但组内所有层须共享块大小。
    # 3. 每层每 token 物理内存：由模型配置决定，目前仅支持所有层相同。
    # 4. 每组层数：目前假设所有层相同。
    # 5. 组内注意力类型：组内所有层须同注意力类型。例外：启用
    #    --disable-hybrid-kv-cache-manager 时全注意力组可含滑动窗口层。
    # 6. 多注意力类型支持：大部分组件对任意数量类型通用，但
    #    find_longest_cache_hit 仅支持一种类型或两种全注意力加一种其他类型。
    # Group all layers by kv_cache_spec.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    # 按 kv_cache_spec 对所有层分组。
    # 如 2 全注意力层 + 3 滑动窗口层 → (full.0, full.1), (sw.0, sw.1, sw.2)。
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)  # 同类层映射
    for layer_name, layer_spec in kv_cache_spec.items():
        # 遍历每层
        same_type_layers[layer_spec].append(layer_name)  # 按规格归类

    # Attempt to further merge same-type layers based on whether their KV
    # cache specs can be merged, to minimize the group count. This benefits
    # situations where specs share a block layout and differ only in a
    # property it can reconcile (e.g. full attention layers differing only in
    # sliding window / attention chunk size).
    # 尝试按规格可否合并进一步合并同类层以减少组数。这有利于规格共享块布局
    # 且仅在可协调属性（如仅滑动窗口/注意力分块大小不同）上不同者。
    layer_buckets: list[list[str]] = []  # 层名桶
    spec_buckets: list[list[KVCacheSpec]] = []  # 规格桶
    for layer_spec, layer_names in same_type_layers.items():
        # 遍历每组同类层
        for names, specs in zip(layer_buckets, spec_buckets):
            # 尝试并入已有桶
            try:
                # A raise means that the specs are incompatible.
                # 抛异常表示规格不兼容。
                type(specs[0]).merge([*specs, layer_spec])  # 尝试合并
            except (AssertionError, ValueError):
                # 合并失败
                continue  # 试下一个桶
            names.extend(layer_names)  # 并入层名
            specs.append(layer_spec)  # 并入规格
            break  # 结束尝试
        else:
            # 无法并入任何已有桶
            layer_buckets.append(list(layer_names))  # 新建桶
            spec_buckets.append([layer_spec])  # 新建规格桶

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    # 将每组拆成更小组，使各组层数一致。必要时为每种类型最后组添加填充层。
    # FIXME(Chen)：编写此代码时（2025-06-02）所有开源混合模型遵循 n:1 模式
    #（如 Gemma3 sw:full=5:1，LLaMA4 local:full=3:1），可用 n:1 中的 "1"
    # 作为组大小（即各注意力类型的最小层数）。支持更复杂模式需更好策略。
    min_num_layers = min([len(layers) for layers in layer_buckets])  # 最少层数
    group_size = min_num_layers  # 组大小取最小层数
    max_num_layers = max([len(layers) for layers in layer_buckets])  # 最多层数
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        # 若层数不比最小层数大太多，用最大层数作为组大小以避免过多填充层。
        # 典型例子 gpt-oss-20b + eagle：12 sw + 13 full，填充为 (13 sw, 13 full)
        # 而非 (12 sw, 24 full)。1.5 是启发式，避免过多填充层同时容纳
        # 为某一注意力类型增加额外层的投机解码草稿器。
        group_size = max_num_layers  # 组大小取最大层数
    grouped_layers = []  # 分组后的层名列表
    for layers in layer_buckets:
        # 遍历每个层名桶
        num_padding_layers = group_size - len(layers) % group_size  # 需填充层数
        if num_padding_layers != group_size:
            # 需要填充
            logger.warning(
                # 警告：添加填充层可能浪费 KV 缓存内存
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)  # 组数
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        # PP 情形下须交错分配层（layers[i::num_groups]）而非连续切片，
        # 否则各阶段组会缺失并被填充，造成内存浪费。
        for i in range(num_groups):
            # 生成各组
            grouped_layers.append(layers[i::num_groups])  # 交错取层
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)  # 创建组规格


def _get_packed_kv_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],  # KV 缓存组
) -> tuple[int, dict[int, list[str]]]:
    """Lay out each cache group densely in one shared block slab.

    A block ID is owned by one cache group at a time, so layouts from different
    groups may overlap. Layers within a group remain disjoint.
    """
    # 将每个缓存组密集布局到单个共享块 slab 中。
    # 一个块 ID 同一时刻归属一个缓存组，因此不同组的布局可能重叠。
    # 组内各层保持不相交。
    layers_by_offset: dict[int, list[str]] = defaultdict(list)  # 偏移到层名列表
    block_stride = 0  # 块步长（每块总字节数）
    for group in kv_cache_groups:
        # 遍历每组
        spec = group.kv_cache_spec  # 组规格
        byte_offset = 0  # 当前字节偏移
        for layer_name in group.layer_names:
            # 遍历组内各层
            if isinstance(spec, UniformTypeKVCacheSpecs):
                # 统一类型规格
                page_size = spec.kv_cache_specs[layer_name].page_size_bytes
                # 取该层页大小
            else:
                # 普通规格
                page_size = spec.page_size_bytes  # 取页大小
            layers_by_offset[byte_offset].append(layer_name)  # 记录该偏移层
            byte_offset += page_size  # 前移偏移
        block_stride = max(block_stride, byte_offset)  # 块步长取最大
    assert block_stride > 0  # 断言块步长为正
    return block_stride, layers_by_offset  # 返回步长与偏移映射


def _use_packed_kv_cache_config(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_groups: list[KVCacheGroupSpec],  # KV 缓存组
) -> bool:
    # 是否使用打包 KV 缓存配置
    is_dsv4 = all(
        # 所有组是否都是统一类型规格（DeepSeek V4 情形）
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    )
    kv_transfer_config = vllm_config.kv_transfer_config  # KV 传输配置
    extra_config = (
        kv_transfer_config.kv_connector_extra_config
        if kv_transfer_config is not None
        else {}
    )
    # 连接器额外配置
    # NOTE: enable_cross_layers_blocks is an experimental API and subject to change with
    # https://github.com/vllm-project/vllm/issues/42082
    # 注意：enable_cross_layers_blocks 是实验性 API，可能随问题 #42082 变化
    enable_cross_layers = (
        str(extra_config.get("enable_cross_layers_blocks", "False")).lower() == "true"
    )
    # 是否启用跨层块
    return is_dsv4 or (enable_cross_layers and len(kv_cache_groups) > 1)
    # DeepSeek V4 默认，或多组且启用跨层块


def _get_kv_cache_config_packed(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_groups: list[KVCacheGroupSpec],  # KV 缓存组
    available_memory: int,  # 可用内存（字节）
) -> tuple[int, list[KVCacheTensor]]:
    """Plan a packed per-block KV cache tensor layout.

    Cache groups use dense, overlapping layouts within one block slab. Each
    emitted tensor aliases the same physical backing allocation.
    """
    # 规划打包的每块 KV 缓存张量布局。
    # 缓存组在单个块 slab 内使用密集重叠布局。
    # 每个产出的张量别名同一物理后备分配。
    block_stride, layers_by_offset = _get_packed_kv_cache_layout(kv_cache_groups)
    # 获取布局

    num_blocks = available_memory // block_stride  # 块数 = 内存 // 步长
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)  # 应用覆盖

    total_size = block_stride * num_blocks  # 总大小

    kv_cache_tensors: list[KVCacheTensor] = []  # 张量列表
    for byte_offset in sorted(layers_by_offset):
        # 按偏移遍历
        kv_cache_tensors.append(
            KVCacheTensor(
                size=total_size,  # 总大小（别名同一分配）
                shared_by=layers_by_offset[byte_offset],  # 共享该偏移的层
                offset=byte_offset,  # 字节偏移
                block_stride=block_stride,  # 块步长
            )
        )
        # 每个偏移创建一个张量

    return num_blocks, kv_cache_tensors  # 返回块数与张量


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_groups: list[KVCacheGroupSpec],  # KV 缓存组
    available_memory: int,  # 可用内存（字节）
) -> KVCacheConfig:
    """
    Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig
    """
    # 从 KV 缓存组和每层规格生成 KV 缓存配置。
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        # 无注意力模型没有 KV 缓存。返回 num_blocks=1，因 BlockPool 总需要一个空块。
        return KVCacheConfig(
            num_blocks=1,  # 1 个块
            kv_cache_tensors=[],  # 无张量
            kv_cache_groups=kv_cache_groups,  # 空组
        )

    # Determine how model runners should initialize the KV cache tensors.
    # 决定模型运行器如何初始化 KV 缓存张量。
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # Special case: all layers have the same type of KV cache but with
        # different hidden sizes. Allocate different amount of memory for each
        # layer based on its hidden size.
        # 特例：所有层 KV 缓存类型相同但隐藏大小不同。
        # 根据各层隐藏大小分配不同内存量。
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        # 块数 = 内存 // 页大小
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)  # 应用覆盖
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        # 每层规格
        kv_cache_tensors = [
            # 为每层创建张量
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                # 该层页大小 * 块数
                shared_by=[layer_name],  # 仅该层共享
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    elif _use_packed_kv_cache_config(vllm_config, kv_cache_groups):
        # DeepSeek V4 uses the packed layout by default. Other multi-group
        # layouts can opt in with --enable-cross-layers.
        # DeepSeek V4 默认使用打包布局。其他多组布局可通过
        # --enable-cross-layers 选择加入。
        num_blocks, kv_cache_tensors = _get_kv_cache_config_packed(
            vllm_config, kv_cache_groups, available_memory
        )
        # 生成打包布局
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        # 一般情形：
        # 有 group_size 个内存池，每个池由各组中一个层共享。
        # 因不同组层有不同的块表，它们使用共享 Tensor 的不同部分。
        # 3 组布局示例（group_size=2）：
        # full.0, sw.0, sw.1 共享 size=available_memory//2 的 Tensor；
        # full.1, sw.2 共享另一个 size=available_memory//2 的 Tensor。
        group_size = max(len(group.layer_names) for group in kv_cache_groups)
        # 最大组层数
        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        # 统一页大小
        assert group_size > 0, "group_size must be greater than 0"  # 断言组大小为正
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        # 计算块数
        kv_cache_tensors = []  # 张量列表
        for i in range(group_size):
            # 遍历组内位置
            shared_by = []  # 共享层列表
            for j in range(len(kv_cache_groups)):
                # 遍历所有组
                if i < len(kv_cache_groups[j].layer_names):
                    # 该组第 i 层存在
                    shared_by.append(kv_cache_groups[j].layer_names[i])
                    # 加入共享列表
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )
            # 创建共享张量

    return KVCacheConfig(
        num_blocks=num_blocks,  # 块数
        kv_cache_tensors=kv_cache_tensors,  # 张量列表
        kv_cache_groups=kv_cache_groups,  # 组列表
    )


def _promote_local_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],  # 每层规格
) -> dict[str, KVCacheSpec]:
    """Use full-attention allocation for local-attention cache specs.

    The returned specs affect KV cache management only. Attention modules keep
    their original sliding-window or chunked-local compute behavior.
    """
    # 对局部注意力缓存规格使用全注意力分配。
    # 返回的规格仅影响 KV 缓存管理。注意力模块保持原始滑动窗口或
    # 分块局部计算行为。
    promoted_specs = kv_cache_spec.copy()  # 复制规格

    if is_kv_cache_spec_uniform(
        promoted_specs
    ) or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs):
        # 已统一或已是统一类型
        return promoted_specs  # 直接返回

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in promoted_specs.values()
    )
    # 是否有全注意力层
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in promoted_specs.values()
    )
    # 是否有滑动窗口层
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in promoted_specs.values()
    )
    # 是否有分块局部注意力层
    full_block_sizes = {
        spec.block_size
        for spec in promoted_specs.values()
        if isinstance(spec, FullAttentionSpec)
    }
    # 全注意力层块大小集合
    full_attention_block_size = (
        next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    )
    # 全注意力块大小（唯一时）

    def promoted_page_size_padded(spec: AttentionSpec, block_size: int) -> int | None:
        # 计算提升后规格的填充页大小
        if spec.page_size_padded is None:
            # 无填充
            return None  # 返回 None
        unpadded_page_size = (
            spec.unpadded_page_size_bytes * block_size // spec.block_size
        )
        # 按新块大小换算未填充页大小
        return max(spec.page_size_padded, unpadded_page_size)  # 取较大者

    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        # 同时有全注意力与局部注意力
        for layer_name, spec in kv_cache_spec.items():
            # 遍历每层
            if isinstance(spec, SlidingWindowMLASpec):
                # 滑动窗口 MLA 层 → 提升为 MLA
                block_size = full_attention_block_size or spec.block_size  # 块大小
                promoted_specs[layer_name] = MLAAttentionSpec(
                    block_size=block_size,  # 块大小
                    num_kv_heads=spec.num_kv_heads,  # KV 头数
                    head_size=spec.head_size,  # 头大小
                    dtype=spec.dtype,  # 数据类型
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                    # 填充页大小
                    cache_dtype_str=spec.cache_dtype_str,  # 缓存类型字符串
                    alignment=spec.alignment,  # 对齐
                    compress_ratio=spec.compress_ratio,  # 压缩比
                    model_version=spec.model_version,  # 模型版本
                )
            elif isinstance(spec, SlidingWindowSpec):
                # 滑动窗口层 → 提升为全注意力
                block_size = full_attention_block_size or spec.block_size  # 块大小
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,  # 块大小
                    num_kv_heads=spec.num_kv_heads,  # KV 头数
                    head_size=spec.head_size,  # 头大小
                    head_size_v=spec.head_size_v,  # V 头大小
                    dtype=spec.dtype,  # 数据类型
                    kv_quant_mode=spec.kv_quant_mode,  # KV 量化模式
                    sliding_window=spec.sliding_window,  # 滑动窗口（保留计算）
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                    # 填充页大小
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                # 分块局部注意力层 → 提升为全注意力
                block_size = full_attention_block_size or spec.block_size  # 块大小
                promoted_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,  # 块大小
                    num_kv_heads=spec.num_kv_heads,  # KV 头数
                    head_size=spec.head_size,  # 头大小
                    dtype=spec.dtype,  # 数据类型
                    attention_chunk_size=spec.attention_chunk_size,  # 分块大小
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                    # 填充页大小
                )

    if not (
        is_kv_cache_spec_uniform(promoted_specs)
        or UniformTypeKVCacheSpecs.is_uniform_type(promoted_specs)
    ):
        # 提升后仍未统一
        raise ValueError("Failed to promote local KV cache specs to one unified type.")
        # 抛出：提升失败

    return promoted_specs  # 返回提升后规格


def _try_get_full_allocation_fallback_groups(
    kv_cache_spec: dict[str, KVCacheSpec],  # 每层规格
) -> list[KVCacheGroupSpec] | None:
    """Try a supported full-allocation fallback for local-attention layers."""
    # 为局部注意力层尝试受支持的全分配回退方案。
    if any(isinstance(spec, HiddenStateCacheSpec) for spec in kv_cache_spec.values()):
        # 含隐藏状态缓存层
        return None  # 不支持回退
    if any(
        isinstance(spec, (SlidingWindowMLASpec, ChunkedLocalAttentionSpec))
        for spec in kv_cache_spec.values()
    ):
        # 含滑动窗口 MLA 或分块局部注意力层
        return None  # 不支持回退

    has_mla = any(isinstance(spec, MLAAttentionSpec) for spec in kv_cache_spec.values())
    # 是否有 MLA 层
    has_regular_swa = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    # 是否有普通滑动窗口层
    if not (has_mla and has_regular_swa):
        # 非 MLA + 滑动窗口组合
        return None  # 不支持回退

    try:
        promoted_specs = _promote_local_kv_cache_specs(kv_cache_spec)  # 尝试提升
    except ValueError:
        # 提升失败
        return None  # 不支持回退
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(promoted_specs)  # 转统一类型
    if uniform_spec is None:
        # 无法转换
        return None  # 不支持回退
    logger.warning(
        "KV cache page sizes cannot be unified; treating sliding-window "
        "layers as full attention for cache allocation. Sliding-window "
        "attention compute is unchanged."
    )
    # 警告：页大小无法统一，将滑动窗口层按全注意力分配缓存，计算不变
    return _get_kv_cache_groups_uniform_type(uniform_spec)  # 返回统一类型组


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    This function tries to convert the KV cache specs to one type if the model
    is a hybrid model with multiple type of KV cache. It will convert all
    SlidingWindowSpec to FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """
    # 若模型为多类型 KV 缓存混合模型，尝试将 KV 缓存规格转换为一种类型。
    # 若全注意力与滑动窗口类型同时存在，将所有 SlidingWindowSpec
    # 转换为 FullAttentionSpec。

    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        # 已统一
        return  # 直接返回

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )
    # 警告：混合 KV 缓存管理器已禁用，不启用 KV 缓存内存优化
    #（如丢弃滑动窗口外的 KV 缓存），滑动窗口层计算仍被节省。
    kv_cache_spec.update(_promote_local_kv_cache_specs(kv_cache_spec))
    # 就地更新为提升后的规格


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],  # 每层规格
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    # 对 KV 缓存规格分组，并将每组统一为一个 UniformTypeKVCacheSpecs。
    # 目前仅用于 DeepseekV4。
    if not any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    ):
        # 无滑动窗口 MLA 层
        return None  # 返回 None

    # SlidingWindowMLASpec models with uniform page sizes don't need tuple packing.
    # 页大小统一的 SlidingWindowMLASpec 模型无需元组打包。
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # 页大小统一
        return None  # 返回 None

    mla_specs: dict[str, KVCacheSpec] = {}  # MLA 层规格
    grouped_swa_mla_specs: dict[tuple[int, int], dict[str, KVCacheSpec]] = defaultdict(
        dict
    )
    # 按（块大小, 滑动窗口）分组的 SWA 层规格
    # NOTE: Here we group SWA layers by (block_size, sliding_window), which separates
    # SWA layers, C4I+C4A layers, and C128A layers into three different groups. It can
    # be fragile with only block_size and sliding_window as keys, but fine for now.
    # 注意：这里按（block_size, sliding_window）对 SWA 层分组，将 SWA 层、
    # C4I+C4A 层、C128A 层分成三组。仅用块大小与滑动窗口作键可能脆弱，暂可用。
    for name, spec in kv_cache_spec.items():
        # 遍历每层
        if isinstance(spec, SlidingWindowMLASpec):
            # 滑动窗口 MLA 层
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
            # 按键分组
        elif isinstance(spec, MLAAttentionSpec):
            # 全 MLA 层
            mla_specs[name] = spec  # 加入 MLA 集合

    assert len(mla_specs) > 0  # 断言存在 MLA 层
    mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)  # 统一 MLA 规格
    assert mla_uniform_spec is not None  # 断言转换成功

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []  # SWA 统一规格列表
    for spec_dict in grouped_swa_mla_specs.values():
        # 遍历每组 SWA 层
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)  # 统一
        assert uniform_spec is not None  # 断言转换成功
        swa_uniform_specs.append(uniform_spec)  # 加入列表

    return [mla_uniform_spec, *swa_uniform_specs]  # 返回 MLA 统一规格 + SWA 各统一规格


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    # 选择使总向上填充最小的块大小。
    # 每个 x 向上取整到 d 的倍数。总填充 pad(d) 为各向上取整后减去原值的和。
    # 在 [lower_bound, max(values)] 中暴力搜索 d（适合小列表/小最大值），
    # 返回填充最小的 d。并列时优先较大 d。
    if not values:
        # 空列表
        raise ValueError("values must be non-empty")  # 抛出
    if any(x <= 0 for x in values):
        # 含非正值
        raise ValueError(f"values must be positive, got: {list(values)!r}")  # 抛出

    min_d = max(1, lower_bound if lower_bound is not None else 1)  # 搜索下界
    max_d = max(values)  # 搜索上界
    if min_d > max_d:
        # 下界超过上界
        return min_d  # 返回下界

    best_d = min_d  # 最优 d
    best_pad: int | None = None  # 最优填充
    for d in range(min_d, max_d + 1):
        # 暴力搜索所有候选 d
        pad = sum((d - (x % d)) % d for x in values)  # 计算总填充
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            # 更优或并列但更大
            best_pad = pad  # 更新填充
            best_d = d  # 更新 d

    return best_d  # 返回最优 d


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],  # 分组后的统一规格
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    # 从分组规格生成 KV 缓存组。
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    # 断言分组非空且全为统一类型规格
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    # 目前限制第一个分组规格为仅含 MLAAttentionSpec 的 UniformTypeKVCacheSpecs。
    full_mla_spec = grouped_specs[0]  # 第一个规格（全 MLA）
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    # 断言全为 MLA 规格
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),  # 全部 MLA 层名
        kv_cache_spec=full_mla_spec,  # 规格
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    # 将"层元组"定义为一组具有不同页大小的层，一个 UniformTypeKVCacheSpecs
    # 包含一系列层元组。如 11 个 C4 层 + 10 个 C128 层，可定义层元组为
    # [C4I, C4A, C128]，full_mla_group 将包含 11 个层元组。其他统一规格
    # 同样划分成层元组。如 21 个同页大小的 SWA 层则有 21 个层元组。
    num_layer_tuples_per_group: list[int] = [
        g_spec.get_num_layer_tuples() for g_spec in grouped_specs
    ]
    # 每组层元组数
    # Choose `num_layer_tuples` to minimize total padding across groups.
    # 选择 `num_layer_tuples` 以最小化各组总填充。
    num_layer_tuples = _approximate_gcd(
        num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
    )
    # 近似 GCD 计算层元组数
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    # 向上取整到 `num_layer_tuples` 的倍数（即填充）
    num_layer_tuples_per_group = [
        round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group
    ]
    # 填充后的每组层元组数

    swa_mla_specs = grouped_specs[1:]  # 除全 MLA 外的其余规格
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in swa_mla_specs
        for spec in group.kv_cache_specs.values()
    )
    # 断言其余全为滑动窗口 MLA 规格

    # Split each SWA UniformKV group into smaller groups to align their
    # numbers of layer tuples. The packed block planner overlays groups, so
    # their page sizes do not need to match.
    # 将每个 SWA 统一 KV 组拆分为更小组以对齐层元组数。
    # 打包块规划器重叠各组，因此它们的页大小无需匹配。
    swa_mla_groups = []  # SWA 组列表
    for sm_spec in swa_mla_specs:
        # 遍历每个 SWA 统一规格
        layers_per_size: dict[int, list[str]] = defaultdict(list)  # 按页大小分层的层名

        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            # 遍历组内各层
            layers_per_size[layer_spec.page_size_bytes].append(layer_name)
            # 按页大小归类
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        # 注意：目前统一 KV 组内每个页大小应有相同层数。这也意味着无需
        # 在部分满的层元组内填充层。
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        # 断言各页大小层数一致
        num_layers_per_size = len(next(iter(layers_per_size.values())))
        # 每页大小层数

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        # 拆分每个统一 KV 组内的层以实现层数对齐。详见
        # `_get_kv_cache_groups_uniform_page_size`。
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)  # 元组组数
        layer_tuples = list(zip(*layers_per_size.values()))  # 转置为层元组
        for i in range(num_tuple_groups):
            # 遍历元组组
            group_layer_tuples = layer_tuples[i::num_tuple_groups]  # 交错取元组
            # Flatten tuples and build dict for from_specs
            # 展开元组并为 from_specs 构建字典
            group_layer_names = [
                name for layer_tuple in group_layer_tuples for name in layer_tuple
            ]
            # 展平层名
            group_layer_specs = {
                name: sm_spec.kv_cache_specs[name] for name in group_layer_names
            }
            # 取组内层规格
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            # 统一子规格
            assert sub_sm_spec is not None  # 断言成功
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,  # 层名
                    kv_cache_spec=sub_sm_spec,  # 规格
                )
            )
            # 加入 SWA 组

    return [full_mla_group, *swa_mla_groups]  # 返回全 MLA 组 + 各 SWA 组


def _annotate_eagle_groups_deepseek_v4(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_spec: dict[str, KVCacheSpec],  # 每层规格
    kv_cache_groups: list[KVCacheGroupSpec],  # KV 缓存组
) -> None:
    # 标记 DeepSeek V4 的 eagle 组
    spec_config = vllm_config.speculative_config  # 投机配置
    if spec_config is None or not spec_config.use_eagle():
        # 未使用 eagle 投机
        return  # 返回
    # Detection uses the merged MLA spec's model_version.
    # 检测使用合并后 MLA 规格的 model_version。
    if not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        # 非 DeepSeek V4 模型
        return  # 返回
    # DeepseekV4's MTP attention layer is always the last layer, and we flag whichever
    # group contains it.
    # FIXME(yifan): avoid/generalize this hacky check.
    # DeepSeekV4 的 MTP 注意力层总是最后一层，标记包含它的组。
    # FIXME：避免/泛化此 hacky 检查。
    last_layer = next(reversed(kv_cache_spec))  # 最后一层名
    for group in kv_cache_groups:
        # 遍历组
        if last_layer in group.layer_names:
            # 组包含最后一层
            group.is_eagle_group = True  # 标记为 eagle 组
            break  # 结束


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]  # 配置；每层规格
) -> list[KVCacheGroupSpec]:
    """
    Split the layers in the model into groups with the same KV cache spec.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        The generated KVCacheGroups
    """
    # 将模型层拆分为具有相同 KV 缓存规格的组。返回生成的 KV 缓存组。
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        # 禁用了混合 KV 缓存管理器
        unify_hybrid_kv_cache_specs(kv_cache_spec)  # 统一混合规格

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        # 返回空列表以允许 KVCacheManager 处理无注意力模型。
        return []  # 返回空列表

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        # 所有层 KV 缓存相同（大多数模型如此）。为每层分配相同内存。
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)  # 单组
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        # 所有层需要相同 token 槽位数（如全部全注意力，或全部同窗口滑动窗口）。
        # 将所有层放入一个组。
        return _get_kv_cache_groups_uniform_type(uniform_spec)  # 统一类型单组
    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):
        # DeepseekV4 case: All layers need the same number of token slots,
        # yet some layers are full attention while others are sliding window
        # attention in different sizes. Need to group layers into multiple
        # UniformTypeKVCacheSpecs.
        # DeepSeekV4 情形：所有层需相同 token 槽位数，但部分层全注意力、
        # 部分层不同大小滑动窗口注意力。需将层分成多个统一类型规格。
        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
        # 生成统一组
        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)
        # 标记 eagle 组
        return kv_cache_groups  # 返回组

    # Pull HiddenStateCacheSpec layers out before the general multi-group
    # path so they don't affect page-size unification or grouping.
    # 在一般多组路径前先抽出 HiddenStateCacheSpec 层，
    # 使它们不影响页大小统一或分组。
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    # 隐藏状态缓存层
    filtered_spec = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, HiddenStateCacheSpec)
    }
    # 过滤后的其余层

    # Prefer preserving each layer's cache semantics. If physical pages cannot
    # be unified, try a supported allocation-only fallback before failing.
    # 优先保留每层缓存语义。若物理页无法统一，在失败前尝试受支持的回退。
    try:
        filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)  # 统一页大小
    except NotImplementedError:
        # 无法统一
        fallback_groups = _try_get_full_allocation_fallback_groups(kv_cache_spec)
        # 尝试回退
        if fallback_groups is None:
            # 回退失败
            raise  # 重新抛出
        return fallback_groups  # 返回回退组
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)  # 生成一般组

    # Add hidden-state layers back with page aligned to the common page.
    # 将隐藏状态层加回，页对齐到公共页大小。
    if hidden_specs:
        common_page = get_uniform_page_size([g.kv_cache_spec for g in groups])
        # 公共页大小
        for name, spec in hidden_specs.items():
            # 遍历隐藏层
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            # 每 token 字节数
            new_bs = max(common_page // per_token, 1)  # 新块大小
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            # 对齐页大小
            groups.append(KVCacheGroupSpec([name], aligned))  # 追加组

    return groups  # 返回组列表


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],  # 各 worker 的 KV 缓存配置
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    # 为调度器生成 KV 缓存配置。
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # 断言所有 worker 块数一致
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    # 所有 worker 除层名外 KV 缓存配置相同，用任意一个初始化调度器。
    cfg = copy.deepcopy(kv_cache_configs[0])  # 深拷贝第一个配置
    for group in cfg.kv_cache_groups:
        # 遍历组
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            # 统一类型规格内所有层类型相同，用任意一个初始化调度器。
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
            # 取第一个子规格替换
    return cfg  # 返回调度器配置


def get_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig  # 配置；KV 缓存配置
) -> tuple[int, float]:
    """
    Get the group-aware KV cache token capacity and max concurrency.
    """
    # 获取组感知的 KV 缓存 token 容量与最大并发度。
    max_model_len = vllm_config.model_config.max_model_len  # 最大模型长度
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    # 最大并发度
    return int(max_concurrency * max_model_len), max_concurrency
    # 返回（容量 token 数，最大并发度）


def update_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig  # 配置；KV 缓存配置
) -> None:
    """Store and log the resolved KV cache capacity."""
    # 存储并记录解析出的 KV 缓存容量。
    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
    # 计算容量与并发度
    vllm_config.cache_config.kv_cache_size_tokens = num_tokens  # 写回 token 容量
    vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency  # 写回并发度
    max_model_len = vllm_config.model_config.max_model_len  # 最大模型长度
    logger.info_once(
        # 一次性记录日志
        "GPU KV cache size: %s tokens, "
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{num_tokens:,}",
        f"{max_model_len:,}",
        max_concurrency,
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_groups: list[KVCacheGroupSpec],  # KV 缓存组
) -> int:
    """
    Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.
    """
    # 从 KV 缓存组计算最大内存用量（字节）。
    # 正确考虑混合模型的填充。如模型有 8 全注意力 + 9 滑动窗口层，
    # 将被填充为 9 全 + 9 滑动窗口以保证组大小统一。
    if not kv_cache_groups:
        # 无组
        return 0  # 返回 0

    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # UniformTypeKVCacheSpecs 特例（单组、逐层规格）
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs  # 每层规格
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )
        # 各层最大内存求和
    elif all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        for group in kv_cache_groups
    ):
        # Special case (only DeepseekV4 for now): all groups are
        # UniformTypeKVCacheSpecs.
        # They must already be page_size aligned and share a common padded
        # layer-tuple layout. Even groups with fewer actual tuples still reserve
        # the global number of tuple slots in the shared tensor layout.
        # 特例（目前仅 DeepSeekV4）：所有组都是 UniformTypeKVCacheSpecs。
        # 它们必须已页大小对齐并共享公共的填充层元组布局。
        # 即使实际元组较少的组也在共享张量布局中预留全局元组槽位数。
        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)
        # 首个规格（全 MLA）
        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())  # 层元组字节数
        num_layer_tuples = max(
            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).get_num_layer_tuples()
            for group in kv_cache_groups
        )
        # 最大层元组数

        total_max_mem_usage_bytes = 0  # 总最大内存
        for group in kv_cache_groups:
            # 遍历每组
            group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)  # 组规格
            g_max_mem_usage_pages = group_spec.max_memory_usage_pages(vllm_config)
            # 组最大内存页数
            g_max_mem_usage_page_bytes = (
                num_layer_tuples * g_max_mem_usage_pages * layer_tuple_bytes
            )
            # 组内存字节数（元组数 * 页数 * 元组字节）
            total_max_mem_usage_bytes += g_max_mem_usage_page_bytes  # 累加
        return total_max_mem_usage_bytes  # 返回总内存

    # 一般情形

    # General case: group_size pools, each shared by one layer per group
    # Memory = group_size * page_size * blocks_for_max_len
    # 一般情形：group_size 个池，每个池由每组一个层共享。
    # 内存 = group_size * page_size * blocks_for_max_len
    group_size = max(len(group.layer_names) for group in kv_cache_groups)  # 最大组层数
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    # 统一页大小
    blocks_needed = sum(
        # 所需块数 = 各组最大内存页数之和
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    return group_size * page_size * blocks_needed  # 总内存 = 组大小 * 页 * 块数


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_groups: list[KVCacheGroupSpec],  # KV 缓存组
    available_memory: int,  # 可用内存（字节）
) -> int:
    """
    Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    # 二分搜索可容纳于可用内存的最大模型长度。
    # 即使 1 个 token 也放不下时返回 0。
    original_max = vllm_config.model_config.max_model_len  # 保存原值

    def fits(model_len: int) -> bool:
        # 检查给定长度是否适合
        vllm_config.model_config.max_model_len = model_len  # 临时修改
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )
        # 组内存用量是否不超过可用内存

    try:
        left, right = 1, original_max  # 搜索区间
        if not fits(left):
            # 最小长度都不行
            return 0  # 返回 0
        result = 1  # 结果
        while left <= right:
            # 二分循环
            mid = (left + right) // 2  # 中点
            if fits(mid):
                # 中点适合
                result = mid  # 更新结果
                left = mid + 1  # 尝试更大
            else:
                # 中点不适合
                right = mid - 1  # 尝试更小
        return result  # 返回结果
    finally:
        vllm_config.model_config.max_model_len = original_max  # 恢复原值


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,  # 全局配置
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],  # 各 worker 投影组
    available_memory: list[int],  # 各 worker 可用内存（字节）
) -> None:
    """
    When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.
    """
    # 当 max_model_len 设为 -1 时，此函数估算可用 GPU 内存可支持的最大
    # 上下文长度。用二分搜索找到所有 worker 都能容纳的最大长度。
    # 注意：vllm_config 会被就地修改。
    original_max = vllm_config.model_config.max_model_len  # 原 max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        # 所有 worker 规格为空（无注意力模型）
        logger.info_once(
            # 记录日志
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return  # 返回

    # Find the max_model_len that fits across all workers.
    # 找到所有 worker 都能容纳的 max_model_len。
    auto_fit_max = original_max  # 自动适配结果
    limiting_worker_mem = available_memory[0]  # 限制性 worker 内存
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        # 遍历各 worker
        if not groups:
            # 该 worker 无组
            continue  # 跳过
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        # 该 worker 可容纳长度
        if worker_max < auto_fit_max:
            # 更小则更新
            auto_fit_max = worker_max  # 更新结果
            limiting_worker_mem = avail_mem  # 更新限制内存

    if auto_fit_max <= 0:
        # 无法容纳任何 token
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )
        # 抛出：内存不足以服务单个 token

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        # 模型完整上下文长度可容纳
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        # 需缩小 max_model_len 以适应内存
        vllm_config.model_config.max_model_len = auto_fit_max  # 写入新值
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],  # 全局 KV 缓存组
    worker_spec: dict[str, KVCacheSpec],  # 该 worker 的层规格
) -> list[KVCacheGroupSpec]:
    """
    Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker, adjusting UniformTypeKVCacheSpecs accordingly.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.
    """
    # 将全局 KV 缓存组投影到单个 worker 被分配的层上。
    # 流水线并行中每个 worker 只拥有部分层。此函数过滤全局组，
    # 仅保留该 worker 上存在的层，并相应调整 UniformTypeKVCacheSpecs。
    # 返回仅含该 worker 层的投影组。
    projected_groups: list[KVCacheGroupSpec] = []  # 投影组列表
    for group in global_kv_cache_groups:
        # 遍历全局组
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        # 该 worker 上的层名
        group_spec = group.kv_cache_spec  # 组规格
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            # 有该 worker 的层且为统一类型规格
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,  # 块大小
                kv_cache_specs={
                    # 仅保留该 worker 的层规格
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,  # 投影后的层名
                group_spec,  # 调整后的规格
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
                # eagle 标记仅在仍有层时保留
            )
        )
    return projected_groups  # 返回投影组


def get_kv_cache_configs(
    vllm_config: VllmConfig,  # 全局配置
    kv_cache_specs: list[dict[str, KVCacheSpec]],  # 每 worker 的层规格
    available_memory: list[int],  # 每 worker 可用内存（字节）
) -> list[KVCacheConfig]:
    """
    Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the KV cache specs of all workers to get the KVCacheSpecs for
       the whole model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: List of dict[layer_name, KVCacheSpec] for each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.
    """
    # 为模型生成 KV 缓存配置。
    # 因所有 worker 使用共享集中控制器，需要各 worker 的 kv_cache_config
    # 一致以确保 KV 缓存分配可应用到所有 worker。但不同 worker 可能内存不同、
    # 层类型不同（启用流水线并行时）。处理差异的实现步骤：
    # 1. 合并所有 worker 的 KV 缓存规格得到整模型规格。
    # 2. 基于整模型层比例生成 KV 缓存组（同时处理混合模型规格统一）。
    # 3. 用每 worker 投影组处理自动适配 max_model_len 与内存检查（考虑 PP 分片）。
    # 4. 基于分组策略为每个 worker 生成 KV 缓存配置（不同 PP 阶段层比例相似，合理）。
    # 5. 将各 worker 的 num_blocks 改为所有 worker 最小值，并按比例缩小
    #    张量大小避免分配未用内存。

    # Merge the KV cache specs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same KV cache spec.
    # 合并所有 worker 的 KV 缓存规格。不同 PP 阶段可能层名不同，
    # 同一 PP 阶段的不同 TP rank 应有相同 KV 缓存规格。
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}  # 合并后的规格
    for kv_cache_spec_one_worker in kv_cache_specs:
        # 遍历每 worker 规格
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            # 遍历该 worker 的每层
            if layer_name not in merged_kv_cache_specs:
                # 新层
                merged_kv_cache_specs[layer_name] = layer_spec  # 直接加入
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )
                # 断言同层规格一致，否则不支持

    # Check if the KV cache specs are registered correctly.
    # This is to prevent that some layers are initialized with unregistered specs.
    # 检查 KV 缓存规格注册是否正确，防止某些层使用未注册规格初始化。
    KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)
    # Get global KV cache groups. This also handles spec unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_kv_cache_specs may be modified in-place.
    # 获取全局 KV 缓存组。启用 disable_hybrid_kv_cache_manager 时
    # 也处理混合模型规格统一。此调用后 merged_kv_cache_specs 可能被就地修改。
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    # 若 original_max_model_len 为 -1，自动确定可用 GPU 内存可容纳的
    # 最大模型长度。使用每 worker 投影组以考虑 PP 分片。
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]
    # 各 worker 的投影组

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    # 若设置了 `num_gpu_blocks_override`，实际分配的缓存大小与 profiling 出的
    # `available_memory` 解耦：`may_override_num_blocks` 将 `num_blocks`
    # 钳制到覆盖值。此处将其反映到 `available_memory`，使自动适配、准入
    # 检查与每 worker 配置构建都基于相同的有效容量规划。
    override = vllm_config.cache_config.num_gpu_blocks_override  # 覆盖值
    if override is not None:
        # 有覆盖
        adjusted_memory: list[int] = []  # 调整后的内存
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            # 遍历各 worker
            if not groups:
                # 无组
                adjusted_memory.append(avail_mem)  # 原样保留
                continue
            bytes_per_block = _pool_bytes_per_block(vllm_config, groups)  # 每块字节
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            # 记录覆盖日志
            adjusted_memory.append(override * bytes_per_block)  # 覆盖后内存
        available_memory = adjusted_memory  # 更新可用内存

    if vllm_config.model_config.original_max_model_len == -1:
        # 需要自动适配
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )
        # 自动适配 max_model_len

    # Check if the available memory is enough per worker.
    # 逐 worker 检查可用内存是否足够。
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        # 遍历各 worker
        if not groups:
            # 无组
            continue  # 跳过
        _check_enough_kv_cache_memory(
            avail_mem,  # 可用内存
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            # 所需内存偏函数
            vllm_config.model_config.max_model_len,  # 最大模型长度
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
            # 估算长度偏函数
        )

    kv_cache_configs: list[KVCacheConfig] = []  # 各 worker 配置
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        # 遍历各 worker
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        # 断言所有层都已分配到组
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )
        # 为该 worker 生成配置

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    # 将各 rank 的 num_blocks 改为所有 rank 的最小值。
    # 同时按比例缩小张量大小避免分配未用内存。
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    # 最小块数
    for kv_cache_config in kv_cache_configs:
        # 遍历配置
        num_blocks_old = kv_cache_config.num_blocks  # 旧块数
        kv_cache_config.num_blocks = min_num_blocks  # 设为最小值

        # Shrink tensor size proportionally
        # 按比例缩小张量大小
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0  # 断言整除
            tensor.size = tensor.size // num_blocks_old * min_num_blocks  # 缩小

    return kv_cache_configs  # 返回各 worker 配置


class BlockHashListWithBlockSize:
    """
    Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency. Each `hash_block_size` hash is
    already chained over its entire prefix, so the hash at the last
    `hash_block_size` boundary of a `target_block_size` block uniquely
    fingerprints that block's prefix; we use it directly.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    the second 16-size hash already covers tokens 0-31, so it is the 32-size
    hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | B    | D     |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.
    """
    # 将块哈希粒度从 `hash_block_size` 转换为 `target_block_size`。
    # 用于 KV 缓存组块大小不同时：`hash_block_size` 是计算原始
    # `block_hashes` 所用大小；`target_block_size` 是组的实际块大小。
    # 目前仅支持按整数因子放大（即 `target_block_size` 是 `hash_block_size`
    # 的倍数）。转换在访问时惰性执行以提高效率。每个 `hash_block_size`
    # 哈希已链式覆盖其整个前缀，因此 `target_block_size` 块最后一个
    # `hash_block_size` 边界处的哈希唯一标识该块前缀；直接使用之。
    # 示例（hash_block_size=16，target_block_size=32）：第二个 16 大小哈希
    # 已覆盖 token 0-31，因此即 32 大小哈希。

    def __init__(
        self,
        block_hashes: list[BlockHash],  # 待转换的块哈希
        hash_block_size: int,  # 原哈希粒度
        target_block_size: int,  # 目标块大小
    ):
        self.block_hashes = block_hashes  # 保存块哈希
        assert target_block_size % hash_block_size == 0  # 断言倍数关系
        self.scale_factor = target_block_size // hash_block_size  # 缩放因子

    def __len__(self) -> int:
        # 转换后块哈希数
        return len(self.block_hashes) // self.scale_factor  # 除以缩放因子

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...  # 整数索引重载

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...  # 切片索引重载

    def __getitem__(self, idx):
        # 获取元素
        if isinstance(idx, int):
            # 整数索引
            return self._get_value_at(idx)  # 返回该处哈希

        if isinstance(idx, slice):
            # 切片索引
            start, stop, step = idx.indices(len(self))  # 解析切片
            return [self._get_value_at(i) for i in range(start, stop, step)]
            # 逐元素返回

        raise TypeError(f"Invalid index type: {type(idx)!r}")  # 无效索引类型

    def __iter__(self) -> Iterator[BlockHash]:
        # 迭代器
        for i in range(len(self)):
            # 遍历索引
            yield self._get_value_at(i)  # 产出哈希

    def _get_value_at(self, idx: int) -> BlockHash:
        # The last hash_block_size hash within the target block already chains
        # over the whole prefix, so it is the target block's hash.
        # 目标块内最后一个 hash_block_size 哈希已覆盖整个前缀，
        # 因此即目标块的哈希。
        return self.block_hashes[(idx + 1) * self.scale_factor - 1]
        # 取目标块末尾的哈希


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
# 块哈希列表类型别名（普通列表或带块大小的视图）


def resolve_block_hashes(
    block_hashes: BlockHashList,  # 块哈希列表
    hash_block_size: int,  # 哈希粒度
    block_size: int,  # 目标块大小
    *,
    supports_fine_grained_hash_lookup: bool = False,  # 是否支持细粒度查找
    alignment_tokens: int | None = None,  # 对齐 token 数（可选）
) -> BlockHashList:
    """Resolve the block-hash view at ``block_size``.

    When ``block_size`` equals ``hash_block_size``, reuse the precomputed block
    hashes directly; otherwise view them at ``block_size`` granularity.
    Fine-grained lookup keeps the original hashes for partial cache hits.
    """
    # 解析 `block_size` 下的块哈希视图。
    # 当 `block_size` 等于 `hash_block_size` 时直接复用预计算块哈希；
    # 否则按 `block_size` 粒度查看。细粒度查找保留原始哈希以支持部分缓存命中。
    if block_size == hash_block_size:
        # 块大小等于哈希粒度
        return block_hashes  # 直接返回
    if isinstance(block_hashes, BlockHashListWithBlockSize):
        # Already a block-size view
        # 已是块大小视图
        assert block_hashes.scale_factor == block_size // hash_block_size  # 断言因子
        return block_hashes  # 直接返回
    # Fine-grained partial hits keep the raw hashes. The caller passes
    # alignment_tokens = hash_block_size to enable them, else >= block_size.
    # 细粒度部分命中保留原始哈希。调用方传 alignment_tokens =
    # hash_block_size 以启用，否则应 >= block_size。
    if (
        supports_fine_grained_hash_lookup  # 支持细粒度查找
        and alignment_tokens is not None  # 有对齐 token
        and alignment_tokens < block_size  # 对齐小于块大小
        and block_size % alignment_tokens == 0  # 块大小能整除对齐
    ):
        return block_hashes  # 返回原始哈希
    assert block_size % hash_block_size == 0  # 断言倍数关系
    return BlockHashListWithBlockSize(block_hashes, hash_block_size, block_size)
    # 创建带块大小的哈希视图
