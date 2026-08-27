# SPDX-License-Identifier: Apache-2.0  # 许可证标识：采用 Apache-2.0 开源协议
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：版权归 vLLM 项目全体贡献者所有
import itertools  # 导入 itertools 模块，用于高效迭代（如 islice 限量切片迭代）
from abc import ABC, abstractmethod  # 导入抽象基类 ABC 与抽象方法装饰器，用于定义抽象接口
from collections import defaultdict  # 导入 defaultdict，可为缺失的键自动创建默认值（如空列表）
from collections.abc import Sequence  # 导入 Sequence 序列抽象类型，用于类型注解
from typing import ClassVar  # 导入 ClassVar，用于声明类级别（而非实例级）变量注解

from vllm.utils.math_utils import cdiv  # 导入向上取整除法函数 cdiv（ceiling division）
from vllm.v1.core.block_pool import BlockPool  # 导入块池：统一管理所有 KV cache 物理块的分配、回收与缓存
from vllm.v1.core.kv_cache_utils import (  # 从 kv_cache_utils 导入块哈希与 KV 块相关的基础类型
    BlockHashList,  # 块哈希列表类型（按块粒度组织的前缀哈希序列）
    BlockHashListWithBlockSize,  # 带块大小视图的哈希列表（细粒度哈希到粗粒度块的视图）
    BlockHashWithGroupId,  # 携带 KV cache 分组 ID 的块哈希，作为缓存条目的键
    KVCacheBlock,  # KV cache 物理块对象（含 block_id、引用计数 ref_cnt、block_hash 等）
    resolve_block_hashes,  # 将请求哈希解析为指定块大小/对齐粒度的块哈希序列
)
from vllm.v1.kv_cache_interface import (  # 从 kv_cache_interface 导入各类注意力层的 KV cache 规格（spec）
    ChunkedLocalAttentionSpec,  # 分块局部注意力规格（如 Gemma 系列的分块局部窗口）
    CrossAttentionSpec,  # 交叉注意力规格（编码器-解码器架构使用）
    FullAttentionSpec,  # 全注意力规格（标准因果自注意力）
    HiddenStateCacheSpec,  # 隐藏状态缓存规格（供 EAGLE 等投机解码草稿模型复用）
    KVCacheSpec,  # 所有 KV cache 规格的抽象基类
    MambaSpec,  # Mamba / 状态空间模型层的 KV cache 规格
    MLAAttentionSpec,  # MLA（Multi-head Latent Attention，多头潜在注意力）规格
    RSWASpec,  # R-SWA（Reference Sliding Window Attention）规格
    SinkFullAttentionSpec,  # 带 sink 块的全注意力规格（StreamingLLM 风格）
    SlidingWindowMLASpec,  # 滑动窗口 + MLA 组合规格
    SlidingWindowSpec,  # 滑动窗口注意力规格
    TQFullAttentionSpec,  # TQ（Tri-Query）全注意力规格
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry  # 导入规格注册表：维护 spec 类型到管理器类的映射
from vllm.v1.request import Request  # 导入请求对象类型（携带 prompt token 数等元信息）


class SingleTypeKVCacheManager(ABC):  # 单类型 KV cache 管理器的抽象基类，每种注意力层类型派生一个子类
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    管理单一类型注意力层的 KV cache 管理逻辑的抽象基类。
    """

    # 类变量：是否支持细粒度（块内部分）哈希查找，默认 False，由支持部分命中的子类（全注意力、Mamba）置 True
    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def __init__(  # 构造函数，初始化管理器的所有状态
        self,
        kv_cache_spec: KVCacheSpec,  # 本管理器对应的 KV cache 规格（决定 block_size、窗口等属性）
        block_pool: BlockPool,  # 共享的块池，所有物理块的分配/释放/缓存均通过它完成
        enable_caching: bool,  # 是否启用前缀缓存（prefix caching）
        kv_cache_group_id: int,  # 本管理器所属 KV cache 分组的编号
        scheduler_block_size: int,  # 调度粒度（所有分组块大小的最小公倍数），是本管理器 block_size 的倍数
        dcp_world_size: int = 1,  # 解码上下文并行（Decode Context Parallelism）的并行度，默认 1（不启用）
        pcp_world_size: int = 1,  # 预填充上下文并行（Prefill Context Parallelism）的并行度，默认 1
        needs_kv_cache_zeroing: bool = False,  # worker 端是否需要对 KV cache 清零（需要管理器上报新分配的块 ID）
        max_admission_blocks_per_request: int | None = None,  # 感知回收的每请求块数上限；None 表示不限制
    ) -> None:  # 构造函数返回 None
        """
        Initializes the SingleTypeKVCacheManager.
        初始化 SingleTypeKVCacheManager。
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.  # 本管理器对应的 kv_cache_spec
            block_pool: The block pool.  # 块池
            kv_cache_group_id: The id of the kv cache group of this manager.  # 本管理器所属 kv cache 分组 ID
            scheduler_block_size: The scheduling granularity (LCM of all group
                block sizes); a multiple of this manager's ``block_size``.
                # 调度粒度（所有组块大小的最小公倍数），是本管理器 block_size 的倍数
            needs_kv_cache_zeroing: Whether worker-side KV cache zeroing needs
                newly allocated block IDs from this manager.
                # worker 端 KV cache 清零是否需要本管理器提供新分配的块 ID
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every
                block until the request finishes.
                # 供 get_num_blocks_to_allocate 使用的感知回收的每请求块数上限；
                # 仅对会跨 chunk 回收块的规格（SWA、分块局部）设置；
                # None（默认）表示不限制，这对全注意力风格（持有块直至请求结束）的规格是正确的
        """
        self.scheduler_block_size = scheduler_block_size  # 保存调度粒度，供对齐计算使用
        # The block size for this manager; used for actual block allocation.
        # 本管理器实际用于块分配的块大小
        self.block_size = kv_cache_spec.block_size  # 从规格中读取基础块大小
        self.dcp_world_size = dcp_world_size  # 保存解码上下文并行度
        self.pcp_world_size = pcp_world_size  # 保存预填充上下文并行度
        if dcp_world_size > 1:  # 若启用 DCP，每个块会被切分到多个 rank 上
            self.block_size *= dcp_world_size  # 块大小乘以并行度（哈希视角下按分片后的块大小计）
        self.kv_cache_spec = kv_cache_spec  # 保存 KV cache 规格对象
        self.block_pool = block_pool  # 保存块池引用
        self.enable_caching = enable_caching  # 保存是否启用前缀缓存
        self._max_admission_blocks_per_request = max_admission_blocks_per_request  # 保存每请求准入块数上限
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        # 仅当 worker 端清零会消费块 ID、且本管理器的规格类型属于需要清零的类型时，才记录新分配的块 ID
        self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (  # 计算是否需要记录新块 ID
            FullAttentionSpec,  # 全注意力规格需要清零
            TQFullAttentionSpec,  # TQ 全注意力规格需要清零
            MLAAttentionSpec,  # MLA 规格需要清零
            HiddenStateCacheSpec,  # 隐藏状态缓存规格需要清零
        )
        self.new_block_ids: list[int] = []  # 记录新分配的块 ID 列表（供 worker 端清零使用）

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        # 请求 ID 到块列表的映射，跟踪每个请求已分配的块，便于请求结束时释放
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)  # 每个请求的块列表（默认空列表）

        # {req_id: The number of cached blocks for this given request}
        # {请求 ID：该请求已缓存（已写入前缀缓存哈希）的块数量}
        # This is used to track the number of cached blocks for each request.
        # 用于跟踪每个请求的已缓存块数量
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        # 只跟踪 RUNNING 状态的请求，不跟踪被抢占（preempted）的请求
        self.num_cached_block: dict[str, int] = {}  # 请求 ID -> 已缓存块数的字典

        self.kv_cache_group_id = kv_cache_group_id  # 保存 KV cache 分组 ID
        self._null_block = block_pool.null_block  # 缓存空块（null block）引用，用作跳过块/空槽位的占位符

        # Whether this group's prefix-cache hits drop the EAGLE/MTP lookahead
        # block. Only consulted by managers whose hit logic is sparse within an
        # aligned segment (SWA). Initialized lazily by the coordinator after
        # determining the attention groups.
        # 本分组的缓存命中是否需要丢弃 EAGLE/MTP 前瞻块；仅被段内命中稀疏的管理器（SWA）查询；
        # 由协调器在确定注意力分组之后惰性初始化
        self.use_eagle = False  # 默认不使用 EAGLE，后续由协调器设置

        # Partial-hit copy-on-write bookkeeping. Populated only by fine-grained
        # managers (full attention, mamba "align"); harmlessly empty elsewhere.
        # 部分命中（partial hit）的写时复制（CoW）记账；仅细粒度管理器（全注意力、mamba "align"）填充，其余为空
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}  # 请求 ID -> (块索引, 共享尾块) 的映射
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []  # 待执行的 CoW (源块, 目标块) 队列
        # Partial-tail offload hand-off for external KV connectors: when a
        # producer registers its last-prompt-boundary partial tail and the
        # 部分尾部（partial tail）卸载交接，用于外部 KV connector：当生产者注册了
        # 其最后一个 prompt 边界处的部分尾部，且
        # durable boundary block is not on the append-only request block table
        # (mamba "align" CoW target), record the request, group, block, and
        # exact token boundary so a connector can offload it under the right
        # hash. Populated only by mamba "align".
        # 持久边界块不在追加式的请求块表中（即 mamba "align" 的 CoW 目标块）时，
        # 记录请求、分组、块与精确的 token 边界，以便 connector 用正确的哈希卸载它；
        # 仅由 mamba "align" 模式填充
        self._pending_partial_tail_offloads: list[  # 待处理的部分尾部卸载交接列表
            tuple[str, int, KVCacheBlock, int]  # 每项为 (请求 ID, 分组 ID, 块对象, 边界 token 数)
        ] = []  # 初始为空列表

    @classmethod  # 类方法装饰器
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):  # 统计给定块中可被驱逐的块数
        # 可驱逐 = 引用计数为 0 且不是空块；sum 对生成器布尔值求和即为计数
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def _has_partial_local_hit(  # 判断本地前缀缓存命中是否以“部分命中”收尾
        self,
        new_computed_blocks: Sequence[KVCacheBlock],  # 本次命中前缀缓存得到的新块
        num_local_computed_tokens: int,  # 本地已计算（命中缓存）的 token 数
    ) -> bool:  # 返回布尔值：是否存在部分命中
        # The local prefix-cache hit ends inside one of this manager's
        # blocks: the shared tail block needs CoW.
        # 本地前缀缓存命中结束在本管理器某个块的内部（未填满整块）：
        # 该共享尾块需要写时复制（CoW），避免多请求共享同一块时互相覆盖
        return (  # 同时满足两个条件才算部分命中
            len(new_computed_blocks) > 0  # 条件 1：确实命中了至少一个块
            and num_local_computed_tokens % self.block_size != 0  # 条件 2：命中 token 数不是块大小的整数倍
        )

    def get_num_blocks_to_allocate(  # 计算为某请求还需分配的块数（调度器用于准入判断）
        self,
        request_id: str,  # 请求 ID
        num_tokens: int,  # 需要槽位的总 token 数（含已分配的 token）
        new_computed_blocks: Sequence[KVCacheBlock],  # 刚命中前缀缓存得到的新块
        total_computed_tokens: int,  # 已计算 token 总数（含本地与外部 connector 计算的）
        num_local_computed_tokens: int,  # 本地前缀缓存命中的已计算 token 数
        num_tokens_main_model: int,  # 主模型（目标模型）需要的 token 数；无投机解码时等于 num_tokens
        apply_admission_cap: bool = False,  # 是否应用准入上限（感知回收的规格需要）
    ) -> int:  # 返回还需分配的块数
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, clamp by `num_required_blocks` by
                `_max_admission_blocks_per_request`for recycling-aware specs
                (SWA, chunked-local).

        Returns:
            The number of blocks to allocate.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)  # 容纳 num_tokens 所需的最少块数（向上取整）
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:  # 需要应用准入上限且上限存在时
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            # 感知回收的规格（SWA、分块局部）在此限制每请求的预留块数，使准入
            # 与启动时的池容量估算一致（统一来源：SlidingWindowSpec.max_admission_blocks_per_request
            # 及其分块局部对应方法）。remove_skipped_blocks 会在每个 chunk 的
            # get_num_blocks_to_allocate 之前由 allocate_slots 调用，因此每请求
            # 峰值实际持有块数 <= 该上限，从而保证「预留总和 <= 池容量」
            # 等价于「峰值实际持有总和 <= 池容量」。两者一旦失配，会重新引入
            # issue #39734 的死锁，甚至更糟：prefill 中途 OOM。
            num_required_blocks = min(  # 用上限钳制需求块数
                num_required_blocks, self._max_admission_blocks_per_request  # 取需求与上限的较小值
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))  # 该请求当前已持有的块数（未分配过则为 0）

        if request_id in self.num_cached_block:  # 快速路径：请求已在运行（num_cached_block 仅跟踪 RUNNING 请求）
            # Fast-path: a running request won't have any new prefix-cache hits.
            # 运行中的请求不会再产生新的前缀缓存命中
            assert len(new_computed_blocks) == 0  # 断言：运行中请求不应有新的缓存命中块
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            # 注意：投机解码时可能为草稿 token 多分配了块（后被拒绝），
            # 此时 num_required_blocks 可能小于已持有块数
            return max(num_required_blocks - num_req_blocks, 0)  # 返回差额，且不为负

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)  # 注意力窗口外被跳过的 token 数（默认实现返回 0）
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks  # 本地已计算块数（新命中块 + 已持有块）
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        # 被注意力窗口跳过的完整块数；没有跳过时为 0
        num_skipped_blocks = num_skipped_tokens // self.block_size  # 跳过 token 数整除块大小得到跳过块数
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        # 只需为未跳过的后缀部分分配块：若窗口内仍有本地已计算块，
        # 它们可抵扣所需容量；否则以跳过块数为主
        num_new_blocks = max(  # 新需块数 = 总需求 - 已覆盖（跳过块与本地已计算块取较大者），且不为负
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),  # 覆盖量取两者较大值
            0,  # 结果下限为 0
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        # new_computed_blocks 中前 num_skipped_blocks 个块被跳过；其中 num_req_blocks 个
        # 可能已在 req_to_blocks 中，因此只从 new_computed_blocks 中跳过剩余部分
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)  # 需从新命中块中跳过的数量（不为负）

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        # 若某已计算块是驱逐候选（在空闲队列且 ref_cnt == 0），被请求 touch 后
        # 会从空闲队列移除，因此空闲容量检查时必须把它计入（会占用空闲额度）
        num_evictable_blocks = self._get_num_evictable_blocks(  # 统计未被跳过的新命中块中可驱逐的块数
            new_computed_blocks[num_skipped_new_computed_blocks:]  # 跳过被窗口跳过的部分后再统计
        )
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):  # 若存在部分命中
            # Reserve the extra block that allocate_new_blocks pulls for the
            # partial-hit CoW redirect.
            # 额外预留一个块：allocate_new_blocks 会为部分命中的 CoW 重定向取出一个新块
            num_new_blocks += 1  # 预留 CoW 块
        return num_new_blocks + num_evictable_blocks  # 最终需分配块数 = 新块 + 会占用空闲额度的可驱逐块

    def add_local_computed_blocks(  # 将本地前缀缓存命中的块挂到请求上
        self,
        request_id: str,  # 请求 ID
        new_computed_blocks: Sequence[KVCacheBlock],  # 本次命中前缀缓存的新块
        num_local_computed_tokens: int,  # 本地已计算 token 数
        num_external_computed_tokens: int,  # 外部（KV connector）已计算 token 数
    ) -> None:  # 无返回值
        """
        Add the locally cached (prefix-hit) blocks to the request:
        1. Touch the computed blocks (paired with adding them to `req_blocks`)
           so their ref_cnt exactly tracks the referencing requests.
        1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
        2. Add the remaining computed blocks.
        将本地缓存（前缀命中）的块挂到请求上：
        1. Touch（触碰）已计算块（与把它们加入 req_blocks 配对进行），
           使 ref_cnt 精确反映引用它们的请求。
        1.5.（可选）对滑动窗口，被跳过的块用 null 块填充。
        2. 添加剩余的已计算块。

        Args:
            request_id: The request ID.  # 请求 ID
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.  # 刚命中前缀缓存的新块
            num_local_computed_tokens: The number of local computed tokens.
            # 本地已计算 token 数
            num_external_computed_tokens: The number of external computed tokens.
            # 外部已计算 token 数
        """
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        # 协调器仅对首次分配调用此方法（运行中的请求在那里会被短路跳过），
        # 因此该请求此时还没有任何块
        req_blocks = self.req_to_blocks[request_id]  # 获取该请求的块列表（首次访问时 defaultdict 自动建空表）
        assert len(req_blocks) == 0  # 断言：请求尚未分配过块
        num_total_computed_tokens = (  # 本地 + 外部已计算 token 总数
            num_local_computed_tokens + num_external_computed_tokens  # 两者相加
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)  # 计算被注意力窗口跳过的 token 数
        num_skipped_blocks = num_skipped_tokens // self.block_size  # 换算成被跳过的完整块数
        if num_skipped_blocks > 0:  # 若存在被跳过的块
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            # 当 num_skipped_blocks > len(new_computed_blocks) 时，可能所有新命中块都被跳过
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]  # 丢弃被跳过的头部块

        # Touch the computed blocks to make sure they won't be evicted.
        # Touch 已计算块，确保它们不会被驱逐
        if self.enable_caching:  # 启用前缀缓存时
            self.block_pool.touch(new_computed_blocks)  # 通过块池 touch：增加引用计数并从空闲队列移除
        else:  # 未启用缓存时
            assert not any(new_computed_blocks), (  # 断言：缓存关闭时不应有任何命中块
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        # 被跳过的块位置用 null 块填充
        req_blocks.extend([self._null_block] * num_skipped_blocks)  # 用 num_skipped_blocks 个空块补齐索引对齐
        # Add the remaining computed blocks.
        # 添加剩余的已计算块
        req_blocks.extend(new_computed_blocks)  # 追加实际命中并保留的块
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        # 所有缓存命中（包括跳过的空块）已经缓存过了；在此标记，
        # 使 cache_blocks() 不会尝试重新缓存已有 block_hash 的块
        self.num_cached_block[request_id] = len(req_blocks)  # 已缓存块数 = 当前全部块数（含空块占位）
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):  # 若存在部分命中
            # Record the partial tail for the CoW redirect in
            # allocate_new_blocks; cap the cached count at the full blocks so
            # cache_blocks() re-caches the private copy once full.
            # 记录部分尾部，供 allocate_new_blocks 做 CoW 重定向；
            # 同时把已缓存数限制为完整块数，使 cache_blocks() 在私有副本填满后重新缓存它
            block_idx = num_local_computed_tokens // self.block_size  # 部分命中块在请求块表中的索引
            self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])  # 记录 (块索引, 共享尾块)
            self.num_cached_block[request_id] = block_idx  # 已缓存数回退到完整块数（不含部分命中的尾块）

    def allocate_external_computed_blocks(  # 为外部（KV connector）计算的 token 分配新块
        self,
        request_id: str,  # 请求 ID
        num_local_computed_tokens: int,  # 本地已计算 token 数
        num_external_computed_tokens: int,  # 外部已计算 token 数
    ) -> None:  # 无返回值
        """
        Allocate new blocks for external (KV-connector) computed tokens.
        为外部（KV connector）已计算的 token 分配新块。

        Must run only after every group's local blocks have been touched via
        `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
        evict another group's cache-hit blocks (issue #33775).
        必须在所有分组的本地块都通过 add_local_computed_blocks 完成 touch 之后才能运行，
        这样本分组的 get_new_blocks 才不会驱逐其他分组的缓存命中块（issue #33775）。

        Args:
            request_id: The request ID.  # 请求 ID
            num_local_computed_tokens: The number of local computed tokens.  # 本地已计算 token 数
            num_external_computed_tokens: The number of external computed tokens.  # 外部已计算 token 数
        """
        num_total_computed_tokens = (  # 本地 + 外部已计算 token 总数
            num_local_computed_tokens + num_external_computed_tokens  # 两者相加
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)  # 被窗口跳过的 token 数
        if num_skipped_tokens > 0:  # 若存在被跳过的 token
            # Some external computed tokens may be skipped too.
            # 外部已计算的 token 也可能被跳过
            num_external_computed_tokens = min(  # 把外部 token 数压缩到窗口内剩余部分
                num_total_computed_tokens - num_skipped_tokens,  # 窗口内保留的 token 总量
                num_external_computed_tokens,  # 与外部 token 数取较小者
            )
        if num_external_computed_tokens <= 0:  # 窗口内没有需要块的外部 token
            return  # 直接返回，无需分配

        req_blocks = self.req_to_blocks[request_id]  # 获取请求当前的块列表
        allocated_blocks = self.block_pool.get_new_blocks(  # 从块池取出新块
            cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)  # 需要的总块数减去已有块数
        )
        req_blocks.extend(allocated_blocks)  # 追加新分配的块
        if self._record_new_block_ids:  # 若需要为 worker 端清零记录块 ID
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)  # 记录所有新块的 ID

    def allocate_new_blocks(  # 为请求分配新块，使其至少有 num_tokens 个槽位
        self, request_id: str, num_tokens: int, num_tokens_main_model: int  # 请求 ID、总 token 数、主模型 token 数
    ) -> list[KVCacheBlock]:  # 返回新分配的块列表
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.
        为请求分配新块，使其至少拥有 num_tokens 个 token 槽位。

        Args:
            request_id: The request ID.  # 请求 ID
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).  # 需要槽位的总 token 数（含已分配的）
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
                # 主模型（投机解码中即目标模型）的 token 数；无投机解码时等于 num_tokens，
                # 有投机解码时为 num_tokens - num_lookahead_tokens
        Returns:
            The new allocated blocks.  # 新分配的块
        """
        cow_blocks: list[KVCacheBlock] = []  # CoW（写时复制）产生的块列表
        if request_id in self._partial_hit_reqs:  # 若该请求存在部分命中待 CoW
            # Partial hit: redirect the shared tail to a private CoW block.
            # Replacing in place keeps the length-based allocation below
            # correct; the extra block was reserved by
            # get_num_blocks_to_allocate.
            # 部分命中：把共享尾块重定向到私有的 CoW 块。
            # 原地替换可保证下面基于长度的分配计算仍然正确；
            # 额外的块已由 get_num_blocks_to_allocate 预留过
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)  # 取出并移除 (块索引, 源共享块)
            cow_block = self.block_pool.get_new_blocks(1)[0]  # 从块池取一个新块作为 CoW 目标块
            self._apply_cow(request_id, block_idx, source_block, cow_block)  # 执行 CoW 重定向（块表替换 + 登记拷贝）
            self.new_block_ids.append(cow_block.block_id)  # 记录 CoW 块 ID（worker 清零/拷贝需要）
            cow_blocks.append(cow_block)  # 加入返回的 CoW 块列表

        req_blocks = self.req_to_blocks[request_id]  # 获取请求当前的块列表
        num_required_blocks = cdiv(num_tokens, self.block_size)  # 容纳 num_tokens 所需块数
        num_new_blocks = num_required_blocks - len(req_blocks)  # 还需补充的块数
        if num_new_blocks <= 0:  # 已有块足够，无需再分配
            return cow_blocks  # 仅返回可能的 CoW 块（可能为空）
        else:  # 还需要更多块
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)  # 从块池取出所需数量的新块
            req_blocks.extend(new_blocks)  # 追加到请求块表
            if self._record_new_block_ids:  # 若需要记录块 ID 供 worker 清零
                self.new_block_ids.extend(b.block_id for b in new_blocks)  # 记录所有新块 ID
            return cow_blocks + new_blocks  # 返回 CoW 块 + 普通新块

    @property  # 只读属性
    def records_new_block_ids(self) -> bool:  # 本管理器的新块是否会被 worker 清零
        """Whether this manager's new blocks are zeroed by the worker."""
        # 本管理器的新块是否会被 worker 清零
        return self._record_new_block_ids  # 直接返回内部标志位

    def take_new_block_ids(self) -> list[int]:  # 取出并清空自上次调用以来新分配的块 ID
        """Drain and return block IDs allocated since the last call."""
        # 取出（drain）并返回自上次调用以来分配的块 ID
        ids = self.new_block_ids  # 保存当前列表引用
        self.new_block_ids = []  # 重置为空列表（下次从空开始累积）
        return ids  # 返回旧列表

    def take_pending_cow_copies(  # 取出待执行的 CoW 拷贝对
        self,
    ) -> list[tuple[KVCacheBlock, KVCacheBlock]]:  # 返回 (源块, 目标块) 对列表
        """Drain pending CoW source and destination block pairs."""
        # 取出待执行的 CoW 源块与目标块对
        pending_copies = self._pending_cow_copies  # 保存当前队列引用
        self._pending_cow_copies = []  # 重置队列
        return pending_copies  # 返回旧队列

    def take_pending_partial_tail_offloads(  # 取出生产者注册的部分尾部卸载交接
        self,
    ) -> list[tuple[str, int, KVCacheBlock, int]]:  # 返回 (请求 ID, 分组 ID, 块, 边界 token 数) 列表
        """Drain producer partial-tail hand-offs.
        取出生产者注册的部分尾部交接。

        Entries are ``(req_id, group_id, block, boundary_tokens)``.
        每项为 (req_id, group_id, block, boundary_tokens)。

        Only mamba "align" populates this. The block lives off the request
        block table, so the caller must pin it until the connector has read
        it — nothing else keeps it alive once the CoW retention is released.
        仅 mamba "align" 模式填充此列表。该块不在请求块表上，
        因此调用方必须 pin 住它，直到 connector 读取完毕 ——
        一旦 CoW 保留解除，没有其他东西能保活该块。
        """
        pending = self._pending_partial_tail_offloads  # 保存当前交接列表引用
        self._pending_partial_tail_offloads = []  # 重置列表
        return pending  # 返回旧列表

    def _apply_cow(  # 执行部分命中块的 CoW 重定向
        self,
        request_id: str,  # 请求 ID
        block_idx: int,  # 块在请求块表中的索引
        source_block: KVCacheBlock,  # 源块（共享的缓存命中尾块）
        cow_block: KVCacheBlock,  # CoW 目标块（私有副本）
    ) -> None:  # 无返回值
        """Redirect a partial prefix-cache hit to a private CoW block.
        将部分前缀缓存命中重定向到私有的 CoW 块。

        Both copy endpoints stay retained until the copy has run on the worker,
        so a same-step free cannot recycle them: ``source_block`` keeps its
        hit-ref, ``cow_block`` takes an extra ref beyond the one handed to the
        request.
        拷贝的两个端点都会被保留到 worker 上真正执行拷贝为止，
        因此同一步内的释放不会回收它们：source_block 保留命中引用，
        cow_block 在交给请求的引用之外额外持有一个引用。
        """
        req_blocks = self.req_to_blocks[request_id]  # 获取请求块表
        assert block_idx < len(req_blocks)  # 断言：索引在块表范围内
        assert req_blocks[block_idx] is source_block  # 断言：该位置确实是源块
        assert not source_block.is_null and source_block.ref_cnt > 0  # 断言：源块非空且仍被引用
        req_blocks[block_idx] = cow_block  # 块表中用私有 CoW 块替换共享源块
        self._pending_cow_copies.append((source_block, cow_block))  # 登记待执行的拷贝 (源, 目标)
        cow_block.ref_cnt += 1  # 为 CoW 块额外加一个引用，防止在拷贝完成前被回收

    def cache_blocks(  # 将请求的完整块写入前缀缓存
        self,
        request: Request,  # 请求对象（含 prompt token 数等）
        num_tokens: int,  # 需要缓存的总 token 数（含已缓存的）
        retention_interval: int | None = None,  # 稀疏本地检查点粒度；None 表示密集检查点
    ) -> None:  # 无返回值
        """
        Cache the blocks for the request.
        为请求缓存块。

        Args:
            request: The request.  # 请求对象
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
                # 需要缓存的总 token 数（含已缓存的）
            retention_interval: Sparse local-checkpoint granularity. ``None``
                keeps dense checkpointing; ``0`` keeps only the latest replay
                boundary; a positive multiple of ``scheduler_block_size`` keeps
                a tail once per that-sized segment. Only SWA acts on it.
                # 稀疏本地检查点粒度：None 保持密集检查点；0 仅保留最新回放边界；
                # scheduler_block_size 的正整数倍表示每段保留一个尾部。仅 SWA 生效。
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)  # 该请求已缓存的块数（默认 0）
        num_full_blocks = num_tokens // self.block_size  # num_tokens 覆盖的完整块数

        if num_cached_blocks >= num_full_blocks:  # 已缓存块数不少于完整块数
            return  # 没有新内容可缓存，直接返回

        # Token boundaries whose reachable tail must be retained under sparse
        # retention: the replay boundary (``num_prompt - 1``, capped by
        # ``get_computed_blocks``) and any detected shared-prefix junction.
        # 稀疏保留策略下必须保留其可达尾部的 token 边界：
        # 回放边界（num_prompt - 1，会被 get_computed_blocks 钳制）与检测到的共享前缀交汇点
        reachable_boundaries = [request.num_prompt_tokens - 1]  # 先加入回放边界
        if request.shared_prefix_boundary:  # 若存在共享前缀交汇点
            reachable_boundaries.append(request.shared_prefix_boundary)  # 也加入边界列表

        block_mask = self.reachable_block_mask(  # 计算每个块的“可达”掩码（决定哪些块真正写入缓存）
            start_block=num_cached_blocks,  # 起始块索引（已缓存的边界）
            end_block=num_full_blocks,  # 结束块索引（完整块数）
            alignment_tokens=self.scheduler_block_size,  # 对齐粒度 = 调度块大小
            kv_cache_spec=self.kv_cache_spec,  # 本管理器的 KV cache 规格
            use_eagle=self.use_eagle,  # 是否启用 EAGLE
            retention_interval=retention_interval,  # 稀疏保留粒度
            reachable_boundaries=reachable_boundaries,  # 必须保留的边界列表
        )
        self.block_pool.cache_full_blocks(  # 通过块池把完整块写入前缀缓存（带掩码过滤）
            request=request,  # 请求对象
            blocks=self.req_to_blocks[request.request_id],  # 请求的完整块表
            num_cached_blocks=num_cached_blocks,  # 已缓存块数（本次从这里开始）
            num_full_blocks=num_full_blocks,  # 完整块数（本次到这里结束）
            block_size=self.block_size,  # 块大小
            kv_cache_group_id=self.kv_cache_group_id,  # KV cache 分组 ID
            block_mask=block_mask,  # 可达掩码（None 表示全部缓存）
        )

        self.num_cached_block[request.request_id] = num_full_blocks  # 更新已缓存块数为完整块数

    @classmethod  # 类方法：子类可覆盖以实现稀疏命中语义
    def reachable_block_mask(  # 计算 cache_full_blocks 使用的每块掩码
        cls,
        start_block: int,  # 起始块索引（含）
        end_block: int,  # 结束块索引（不含）
        alignment_tokens: int | None,  # 对齐粒度（token 数），None 表示无对齐约束
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        use_eagle: bool,  # 是否启用 EAGLE
        retention_interval: int | None = None,  # 稀疏保留粒度
        reachable_boundaries: Sequence[int] = (),  # 必须保留可达尾部的 token 边界序列
    ) -> list[bool] | None:  # 返回每块布尔掩码；None 表示全部缓存
        """Per-block mask for ``cache_full_blocks``. ``None`` means cache
        every (non-null) block — the default for full attention.
        cache_full_blocks 使用的每块掩码。None 表示缓存所有（非空）块 ——
        这是全注意力的默认行为。

        Subclasses with sparse hit semantics (SWA / Mamba) override this to skip
        blocks that can never serve a hit at any alignment-aligned prefix length.
        ``reachable_boundaries`` are token positions whose reachable tail must be
        retained; the base (dense) policy ignores them.
        具有稀疏命中语义的子类（SWA / Mamba）覆盖本方法，跳过在任何
        alignment 对齐前缀长度下都不可能产生命中的块。
        reachable_boundaries 是必须保留可达尾部的 token 位置；基础（密集）策略忽略它们。
        """
        return None  # 默认返回 None：密集缓存所有块

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:  # 弹出请求的块（不归还块池）
        """
        Pop the request's bookkeeping and return its blocks without yet
        returning them to the block pool. The caller is responsible for
        eventually passing the returned blocks to `block_pool.free_blocks`,
        freeing them in reverse order (so that tail blocks are evicted first).
        弹出该请求的记账信息并返回其块，但暂不归还块池。调用方负责
        最终把返回的块传给 block_pool.free_blocks，并按逆序释放
        （使尾部块最先被驱逐）。

        Args:
            request_id: The request ID.  # 请求 ID

        Returns:
            The request's blocks in allocation order.  # 按分配顺序返回的该请求块列表
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        # 默认为 []，以防请求在分配前就被释放（中止）
        req_blocks = self.req_to_blocks.pop(request_id, [])  # 从映射中弹出块列表
        self.num_cached_block.pop(request_id, None)  # 清除已缓存块数记录
        self._partial_hit_reqs.pop(request_id, None)  # 清除部分命中记录
        return req_blocks  # 返回块列表

    def free(self, request_id: str) -> None:  # 释放请求的全部块
        """
        Free the blocks for the request.
        释放该请求的块。

        Args:
            request_id: The request ID.  # 请求 ID
        """
        # Free blocks in reverse order so that the tail blocks are freed first.
        # 按逆序释放块，使尾部块先被释放（更可能被优先复用）
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))  # 弹出块后逆序交给块池释放

    @abstractmethod  # 抽象方法：子类必须实现
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:  # 获取公共前缀块数量
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache.
        获取所有已分配 KV cache 的请求共享的公共前缀块数量（用于级联注意力）。

        Args:
            running_request_id: The request ID.  # 运行中的请求 ID

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.  # 所有已分配 KV cache 请求的公共前缀块数
        """

        raise NotImplementedError  # 抽象方法，直接调用会抛未实现异常

    @classmethod  # 类方法：由子类按各自注意力类型定制
    @abstractmethod  # 抽象方法：子类必须实现
    def find_longest_cache_hit(  # 查找最长前缀缓存命中
        cls,
        block_hashes: BlockHashList,  # 请求的块哈希序列
        max_length: int,  # 缓存命中前缀的最大长度（token 数）
        kv_cache_group_ids: list[int],  # 参与查找的 KV cache 分组 ID 列表
        block_pool: BlockPool,  # 块池（用于查缓存块）
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        drop_eagle_block: bool,  # 是否为 EAGLE/MTP 丢弃最后一个匹配块
        alignment_tokens: int,  # 命中长度（token）必须是该值的倍数
        dcp_world_size: int = 1,  # 解码上下文并行度
        pcp_world_size: int = 1,  # 预填充上下文并行度
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:  # 返回 (每分组的命中块列表元组, 命中 token 长度)
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.
        If eagle is enabled, drop the last matched block to force recompute the
        last block to get the required hidden states for eagle drafting head.
        Need to be customized for each attention type.
        获取不超过 max_length 的最长缓存命中前缀。该前缀必须是
        kv_cache_group_ids 中所有 KV cache 分组的公共前缀命中。
        若未找到缓存命中，返回空列表。
        若启用 EAGLE，丢弃最后一个匹配块，强制重算最后一块，
        以获得 EAGLE 草稿头所需的隐藏状态。
        每种注意力类型都需要定制实现。

        Args:
            block_hashes: The block hashes of the request.  # 请求的块哈希
            max_length: The maximum length of the cache hit prefix.  # 缓存命中前缀的最大长度
            kv_cache_group_ids: The ids of the kv cache groups.  # KV cache 分组 ID
            block_pool: The block pool.  # 块池
            kv_cache_spec: The kv cache spec.  # KV cache 规格
            drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
                Always False for non-EAGLE/MTP groups, but can be False for EAGLE/MTP
                groups too if the last block is already dropped (e.g., in a
                convergence loop in `find_longest_cache_hit`).
                # 是否为 EAGLE/MTP 丢弃最后一个匹配块。非 EAGLE/MTP 分组恒为 False，
                # 但若最后一块已被丢弃（如收敛循环中），EAGLE/MTP 分组也可能为 False
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens). By default, it should
                be set to the block_size.
                # 返回的缓存命中长度（token）必须是该值（token）的倍数，默认设为 block_size
            dcp_world_size: The world size of decode context parallelism.  # 解码上下文并行度
            pcp_world_size: The world size of prefill context parallelism.  # 预填充上下文并行度

        Returns:
            A tuple containing cached blocks and the exact cache-hit length in
            tokens. The cached block tuple has skipped blocks replaced by null
            blocks for each kv cache group in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
            返回一个元组：缓存块与精确的缓存命中长度（token）。缓存块元组中，
            每个 KV cache 分组被跳过的块都用空块替换。
            例如滑动窗口管理器在 block size 4、窗口 8、单分组时应返回形如
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) 的列表。
        """

        raise NotImplementedError  # 抽象方法，由子类实现

    def _remove_blocks_in_range(  # 释放指定范围内的块并用空块替换
        self,
        request_id: str,  # 请求 ID
        first_block: int,  # 起始块索引（含）
        last_block: int,  # 结束块索引（不含）
    ) -> None:  # 无返回值
        """Free blocks in ``[first_block, last_block)`` and replace with null_block.
        释放 [first_block, last_block) 内的块并替换为 null_block。

        Iterates backward so newly-evictable tail blocks are reached even after
        earlier blocks in the range were nulled in a prior call.
        逆序遍历，这样即使范围内靠前的块已在先前调用中被置空，
        新变为可驱逐的尾部块也能被处理到。
        """
        if request_id not in self.req_to_blocks:  # 请求无块（如尚未分配就被中止）
            return  # 直接返回
        if first_block >= last_block:  # 空范围
            return  # 无需处理
        blocks = self.req_to_blocks[request_id]  # 获取请求块表
        last_block = min(last_block, len(blocks))  # 上界钳制到实际块表长度

        freed: list[KVCacheBlock] = []  # 收集本次要释放的块
        for i in range(last_block - 1, first_block - 1, -1):  # 从后往前遍历范围内的块
            if blocks[i] == self._null_block:  # 遇到已被置空的块
                break  # 更早的块也已处理过，停止遍历
            freed.append(blocks[i])  # 收集该块待释放
            blocks[i] = self._null_block  # 块表中置为空块占位
        if freed:  # 若确实释放了一些块
            self.block_pool.free_blocks(freed)  # 交给块池统一释放

    def remove_skipped_blocks(  # 移除不再参与注意力计算的块（窗口外的块）
        self,
        request_id: str,  # 请求 ID
        processed_computed_tokens: int,  # 已完整处理并提交的已计算 token 前缀长度（可安全释放）
        num_prompt_tokens: int | None = None,  # 可选的 prompt 长度（R-SWA 等释放中间空隙的类型使用）
    ) -> None:  # 无返回值
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.
        移除并释放注意力计算不再需要的块。被移除的块应以 null_block 替换。

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.
        本函数依赖 get_num_skipped_tokens，各注意力类型的实现各不相同。

        Args:
            request_id: The request ID.  # 请求 ID
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
                # 仅覆盖已完整处理并提交 token 的已计算前缀长度（可安全释放）
            num_prompt_tokens: Optional prompt length for attention types (e.g.
                R-SWA) that evict a middle gap rather than a head prefix. Ignored
                by the default implementation.
                # 可选 prompt 长度，供驱逐中间空隙（而非头部前缀）的注意力类型（如 R-SWA）使用；
                # 默认实现忽略此参数
        """
        del num_prompt_tokens  # 默认实现不使用该参数，显式删除以免误用
        # Remove the blocks that will be skipped during attention computation.
        # 移除注意力计算中将被跳过的块
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)  # 计算可跳过的 token 数
        if num_skipped_tokens <= 0:  # 没有可跳过的 token
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            # 这说明所有 token 都在注意力窗口内，无需释放窗口外的块。
            # 典型例子是全注意力：请求结束前从不释放任何块
            return  # 直接返回
        blocks = self.req_to_blocks[request_id]  # 获取请求块表
        num_skipped_blocks = num_skipped_tokens // self.block_size  # 换算成可跳过的完整块数
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        # num_skipped_tokens 可能包含尚未分配的 token（例如注意力窗口
        # 已移入外部已计算 token 区域），因此必须钳制到该请求当前
        # 实际存在的块数
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))  # 钳制到现有块数
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)  # 释放 [0, num_skipped_blocks) 范围的块

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:  # 计算被跳过的 token 数（默认实现）
        """
        Get the number of tokens that will be skipped for attention computation.
        获取注意力计算中将被跳过的 token 数。

        Args:
            num_computed_tokens: The number of tokens that have been computed.  # 已计算的 token 数

        Returns:
            The number of tokens that will be skipped for attention computation.  # 将被跳过的 token 数
        """
        # The default behavior is to not skip any tokens.
        # 默认行为：不跳过任何 token（全注意力等类型使用）
        return 0  # 返回 0

    def new_step_starts(self) -> None:  # 新调度步开始的钩子（默认空实现）
        return None  # 基类无状态需要重置


class FullAttentionManager(SingleTypeKVCacheManager):  # 全注意力（及分块局部注意力）管理器
    supports_fine_grained_hash_lookup: ClassVar[bool] = True  # 支持细粒度哈希查找（可部分命中）

    @classmethod  # 类方法：实现最长缓存命中查找
    def find_longest_cache_hit(  # 全注意力的最长前缀缓存命中查找
        cls,
        block_hashes: BlockHashList,  # 请求的块哈希序列
        max_length: int,  # 命中前缀的最大长度
        kv_cache_group_ids: list[int],  # KV cache 分组 ID 列表
        block_pool: BlockPool,  # 块池
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        drop_eagle_block: bool,  # 是否为 EAGLE/MTP 丢弃最后一块
        alignment_tokens: int,  # 命中长度的对齐粒度（token）
        dcp_world_size: int = 1,  # 解码上下文并行度
        pcp_world_size: int = 1,  # 预填充上下文并行度
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:  # 返回 (每分组命中块列表元组, 命中长度)
        assert isinstance(  # 断言规格类型匹配
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec  # 仅支持全注意力与分块局部注意力规格
        ), (  # 断言失败时的错误信息
            "FullAttentionManager can only be used for full attention "  # 只能用于全注意力
            "and chunked local attention groups"  # 与分块局部注意力分组
        )
        block_size = kv_cache_spec.block_size  # 取规格中的块大小
        if dcp_world_size > 1:  # 若启用 DCP
            # DCP shards each block's KV across ranks; hashes must be viewed at
            # the sharded block size.
            # DCP 将每个块的 KV 切分到各 rank；哈希必须按分片后的块大小来看
            block_size *= dcp_world_size  # 块大小乘以并行度
        block_hashes = resolve_block_hashes(  # 将请求哈希解析为当前块大小的块哈希
            block_hashes,  # 原始块哈希
            block_pool.hash_block_size,  # 块池的哈希粒度
            block_size,  # 目标块大小
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,  # 是否支持细粒度查找
            alignment_tokens=alignment_tokens,  # 对齐粒度
        )

        # Fine-grained mode (alignment_tokens == hash_block_size <
        # block_size): resolve_block_hashes kept the raw hash-granularity
        # list so interior boundaries can be probed.
        # 细粒度模式（alignment_tokens == hash_block_size < block_size）：
        # resolve_block_hashes 保留了原始哈希粒度的列表，以便探测块内边界
        fine_grained = (  # 判断是否处于细粒度模式
            alignment_tokens < block_size and block_size % alignment_tokens == 0  # 对齐粒度更小且能整除块大小
        )
        if fine_grained:  # 细粒度模式下
            # list or lazy BlobBlockHashes view
            # 块哈希是列表或惰性的 BlobBlockHashes 视图
            assert isinstance(block_hashes, Sequence)  # 断言是可索引的序列
            full_block_hashes: BlockHashList = BlockHashListWithBlockSize(  # 构造哈希粒度到块粒度的视图
                block_hashes, alignment_tokens, block_size  # 原始哈希、哈希粒度、块大小
            )
        else:  # 非细粒度模式
            full_block_hashes = block_hashes  # 直接使用解析后的块哈希

        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(  # 每个分组一个命中块列表，初始为空
            [] for _ in range(len(kv_cache_group_ids))  # 按分组数量创建
        )
        # Phase 1: longest run of cached full blocks from the start. A missing
        # block implies every later block misses too (chained hashes).
        # 阶段 1：从头开始查找最长的连续已缓存完整块。缺失某个块意味着
        # 其后的所有块也都缺失（哈希是链式依赖的）
        for block_hash in itertools.islice(full_block_hashes, max_length // block_size):  # 限量遍历块哈希
            cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)  # 查缓存块
            if not cached_block:  # 未命中
                break  # 链式哈希下后续也不可能命中，提前退出
            for computed, cached in zip(computed_blocks, cached_block):  # 遍历每个分组的命中块
                computed.append(cached)  # 追加到对应分组的列表
        hit_length = len(computed_blocks[0]) * block_size  # 命中长度 = 命中块数 × 块大小

        # Phase 2 (fine-grained only): extend into the first non-full block by
        # probing its interior hash boundaries high-to-low (longest hit first).
        # 阶段 2（仅细粒度）：探测第一个未填满块的内部哈希边界，
        # 从高到低探测（优先最长命中），以延伸命中
        if fine_grained:  # 仅细粒度模式执行
            # list or lazy BlobBlockHashes view
            # 块哈希是列表或惰性视图
            assert isinstance(block_hashes, Sequence)  # 断言可索引
            scale_factor = block_size // alignment_tokens  # 一个块内有多少个哈希粒度单元
            first_partial_idx = len(computed_blocks[0]) * scale_factor  # 第一个部分块的起始哈希索引
            max_partial_idx = min(  # 探测的最大索引（不含）
                first_partial_idx + scale_factor - 1,  # 该块内最后一个哈希单元
                max_length // alignment_tokens,  # 受最大长度约束
                len(block_hashes),  # 受实际哈希数约束
            )
            for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):  # 从高到低探测哈希边界
                cached_tail = block_pool.get_cached_block(  # 查该哈希边界对应的缓存块
                    block_hashes[fine_idx], kv_cache_group_ids  # 哈希与分组
                )
                if not cached_tail:  # 该边界未命中
                    continue  # 尝试更短的边界
                for computed, cached in zip(computed_blocks, cached_tail):  # 命中：追加各分组块
                    computed.append(cached)  # 追加
                hit_length = (fine_idx + 1) * alignment_tokens  # 命中长度 = (边界索引+1) × 哈希粒度
                break  # 找到最长命中，停止探测

        # Eagle needs the tokens right before the generation point recomputed:
        # drop one hash unit when fine-grained (the tail block's KV is
        # append-only, so it still covers the reduced length), else one cache
        # block.
        # EAGLE 需要重算生成点之前的 token：细粒度时丢弃一个哈希单元
        # （尾块的 KV 是追加式的，仍能覆盖缩短后的长度），否则丢弃一个缓存块
        if drop_eagle_block and hit_length > 0:  # 需要丢弃且确实有命中
            hit_length -= min(alignment_tokens, block_size)  # 命中长度回退一个最小单元
        # Round down to the alignment; a no-op when fine-grained (hits land on
        # hash boundaries by construction) and when alignment_tokens ==
        # block_size. Then trim blocks past the new tail.
        # 向下取整到对齐粒度；细粒度时（命中按构造落在哈希边界上）以及
        # alignment_tokens == block_size 时此操作为空。然后裁剪新尾部之后的块
        hit_length -= hit_length % alignment_tokens  # 对齐取整
        num_blocks = cdiv(hit_length, block_size)  # 对齐后的命中长度对应多少块
        for computed in computed_blocks:  # 遍历每个分组的块列表
            del computed[num_blocks:]  # 删除超出部分
        return computed_blocks, hit_length  # 返回 (各分组命中块, 命中长度)

    def cache_blocks(  # 覆盖基类缓存逻辑：额外处理部分尾块缓存
        self,
        request: Request,  # 请求对象
        num_tokens: int,  # 需缓存的总 token 数
        retention_interval: int | None = None,  # 稀疏保留粒度（透传给基类）
    ) -> None:  # 无返回值
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)  # 先走基类密集缓存逻辑
        hash_block_size = self.block_pool.hash_block_size  # 块池的哈希粒度
        if self.block_size == hash_block_size:  # 块大小与哈希粒度一致
            return  # 不存在块内边界，无需处理部分尾块
        self._cache_partial_tail_block(request, num_tokens)  # 缓存 prompt 尾部（若结束在块内部）

    def _cache_partial_tail_block(  # 当 prompt 尾部结束在缓存块内部时缓存该尾部
        self,
        request: Request,  # 请求对象
        num_tokens: int,  # 当前需缓存的 token 数
    ) -> None:  # 无返回值
        """Cache the prompt tail when it ends inside a cache block.
        当 prompt 尾部结束在缓存块内部时缓存该尾部。

        Only the final prompt hash boundary is registered as a partial
        prefix-cache entry; intermediate hash boundaries inside the same cache
        block are intentionally skipped.
        只把最后一个 prompt 哈希边界注册为部分前缀缓存条目；
        同一缓存块内的中间哈希边界被有意跳过。
        """
        hash_block_size = self.block_pool.hash_block_size  # 哈希粒度
        boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size  # 最后一个 prompt 哈希边界的 token 位置
        if boundary_tokens == 0 or boundary_tokens > num_tokens:  # 边界为 0 或超过当前已缓存量
            return  # 无尾部可缓存
        if boundary_tokens % self.block_size == 0:  # 边界恰好落在整块边界上
            return  # 不存在块内部分，无需处理

        blocks = self.req_to_blocks[request.request_id]  # 获取请求块表
        block_idx = boundary_tokens // self.block_size  # 该边界所在块的索引
        if block_idx >= len(blocks):  # 块尚未分配
            return  # 无法缓存
        self.block_pool.cache_partial_block(  # 通过块池注册部分块缓存条目
            request=request,  # 请求对象
            block=blocks[block_idx],  # 目标块
            num_tokens=boundary_tokens,  # 该块内有效的 token 数
            kv_cache_group_id=self.kv_cache_group_id,  # 分组 ID
            block_size=self.block_size,  # 块大小
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:  # 计算公共前缀块数（级联注意力用）
        blocks = self.req_to_blocks[running_request_id]  # 取该请求的块表
        num_common_blocks = 0  # 公共前缀块计数
        for block in blocks:  # 从头遍历块
            if block.ref_cnt == len(self.req_to_blocks):  # 引用计数等于全部请求数：说明所有请求共享此块
                num_common_blocks += 1  # 计入公共前缀
            else:  # 遇到非共享块
                break  # 公共前缀到此为止
        return num_common_blocks  # 返回公共前缀块数


class RSWAManager(FullAttentionManager):  # R-SWA（Reference Sliding Window Attention）管理器，继承全注意力管理器
    """KV cache manager for Reference Sliding Window Attention (R-SWA).
    Reference Sliding Window Attention（R-SWA）的 KV cache 管理器。

    When ``num_prompt_tokens`` is supplied to ``remove_skipped_blocks``, frees
    gap blocks between the prefill tail and the current decode window.  This
    bounds per-request KV memory at O(prefix_len + rswa_window) instead of
    growing linearly with decode length.
    当 remove_skipped_blocks 传入 num_prompt_tokens 时，释放 prefill 尾部
    与当前解码窗口之间的空隙（gap）块。这使每请求的 KV 内存被限制在
    O(prefix_len + rswa_window)，而不是随解码长度线性增长。
    """

    def __init__(self, kv_cache_spec: RSWASpec, **kwargs) -> None:  # 构造函数
        super().__init__(kv_cache_spec, **kwargs)  # 调用基类初始化
        self.rswa_window: int = kv_cache_spec.rswa_window  # 保存 R-SWA 窗口大小

    def remove_skipped_blocks(  # 覆盖：释放 R-SWA 中间空隙块
        self,
        request_id: str,  # 请求 ID
        processed_computed_tokens: int,  # 已完整处理的已计算 token 前缀长度
        num_prompt_tokens: int | None = None,  # prompt 长度（用于定位空隙区间）
    ) -> None:  # 无返回值
        """Free gap blocks that are no longer needed for attention.
        释放注意力不再需要的空隙块。

        Gap = blocks entirely within
            [ceil(prefix_len / block_size) * block_size,
             max(prefix_len, processed_computed_tokens - rswa_window))
        空隙 = 完全位于该区间内的块：
            [ceil(prefix_len / block_size) * block_size,
             max(prefix_len, processed_computed_tokens - rswa_window))

        Freed blocks are replaced with null_block in req_to_blocks so the
        block_table passed to FA4 is valid (null_block KV is all-zero;
        rswa_mask_mod marks gap positions as non-visible so FA4 skips them).
        被释放的块在 req_to_blocks 中替换为 null_block，使传给 FA4 的
        block_table 仍然有效（null_block 的 KV 全为零；
        rswa_mask_mod 将空隙位置标记为不可见，FA4 会跳过它们）。
        """
        if num_prompt_tokens is None:  # 未提供 prompt 长度
            super().remove_skipped_blocks(  # 退化为基类行为（释放头部前缀）
                request_id, processed_computed_tokens, num_prompt_tokens  # 透传参数
            )
            return  # 结束

        bs = self.block_size  # 块大小简写
        # First block fully after the prefill boundary.
        # 第一个完全位于 prefill 边界之后的块
        first_gap_block = cdiv(num_prompt_tokens, bs)  # 空隙起始块索引（向上取整）
        # Decode window start position; blocks before this are evictable.
        # 解码窗口起始位置；此位置之前的块可被驱逐
        window_start = max(  # 窗口起点 = prompt 长度与 (已计算-窗口) 的较大者
            num_prompt_tokens, processed_computed_tokens - self.rswa_window  # 两个候选
        )
        last_gap_block = window_start // bs  # exclusive upper bound  # 空隙结束块索引（不含）
        self._remove_blocks_in_range(request_id, first_gap_block, last_gap_block)  # 释放空隙范围内的块


class SlidingWindowManager(SingleTypeKVCacheManager):  # 滑动窗口注意力管理器
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:  # 构造函数
        super().__init__(kv_cache_spec, **kwargs)  # 调用基类初始化
        self.sliding_window = kv_cache_spec.sliding_window  # 保存滑动窗口大小

    @classmethod  # 类方法：计算命中所需连续块数
    def _contiguous_blocks_for_hit(  # 前缀缓存命中所需的连续块数
        cls, window_size: int, block_size: int, use_eagle: bool  # 窗口大小、块大小、是否启用 EAGLE
    ) -> int:  # 返回所需连续块数
        blocks = cdiv(window_size - 1, block_size)  # 覆盖窗口（除生成点外）所需的块数
        if use_eagle:  # 若启用 EAGLE
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            # 启用 EAGLE 时需丢弃最后一个匹配块。对滑动窗口层，
            # 通过将命中所需连续块数加一、再丢弃最后匹配块来实现
            blocks += 1  # 所需连续块数加一
        return blocks  # 返回所需连续块数

    @classmethod  # 类方法：滑动窗口的最长缓存命中查找
    def find_longest_cache_hit(  # 从右向左查找满足窗口长度的连续命中
        cls,
        block_hashes: BlockHashList,  # 请求的块哈希序列
        max_length: int,  # 命中前缀最大长度
        kv_cache_group_ids: list[int],  # KV cache 分组 ID 列表
        block_pool: BlockPool,  # 块池
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        drop_eagle_block: bool,  # 是否为 EAGLE/MTP 丢弃最后一块
        alignment_tokens: int,  # 对齐粒度
        dcp_world_size: int = 1,  # 解码上下文并行度
        pcp_world_size: int = 1,  # 预填充上下文并行度
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:  # 返回 (各分组命中块, 命中长度)
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (  # 断言规格为滑动窗口
            "SlidingWindowManager can only be used for sliding window groups"  # 仅用于滑动窗口分组
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."  # DCP 暂不支持滑动窗口
        assert pcp_world_size == 1, "PCP not support sliding window attn now."  # PCP 暂不支持滑动窗口
        # Fine-grained partial hits are not supported for sliding window now
        # 滑动窗口暂不支持细粒度部分命中
        assert alignment_tokens % kv_cache_spec.block_size == 0, (  # 对齐粒度必须是块大小的整数倍
            "SlidingWindowManager does not support fine-grained (partial) cache hits"  # 不支持细粒度（部分）缓存命中
        )
        block_hashes = resolve_block_hashes(  # 解析块哈希到滑动窗口的块大小
            block_hashes,  # 原始哈希
            block_pool.hash_block_size,  # 哈希粒度
            kv_cache_spec.block_size,  # 目标块大小
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,  # 细粒度支持（此处为 False）
            alignment_tokens=alignment_tokens,  # 对齐粒度
        )

        # The number of contiguous blocks needed for a prefix cache hit.
        # 前缀缓存命中所需的连续块数
        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(  # 按窗口大小计算
            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block  # 窗口、块大小、EAGLE 标志
        )

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        # TODO: 缓存未命中时让 i 直接减去 sliding_window_contiguous_blocks，
        # 把时间复杂度从 O(max_num_blocks) 优化到
        # O(max_num_blocks / sliding_window_contiguous_blocks + sliding_window_contiguous_blocks)，
        # 对低缓存命中率场景有利
        max_num_blocks = max_length // kv_cache_spec.block_size  # 最大长度内的块数
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(  # 每个分组初始化为全空块的列表
            [block_pool.null_block] * max_num_blocks  # 先用空块占满（滑动窗口前面的块本就不需要）
            for _ in range(len(kv_cache_group_ids))  # 按分组数量创建
        )
        block_size = kv_cache_spec.block_size  # 块大小简写
        num_contiguous_blocks = 0  # 当前连续命中块计数
        match_found = False  # 是否找到满足窗口长度的匹配
        # Search from right to left and early stop when a match is found.
        # 从右向左搜索，找到匹配即提前停止
        for i in range(max_num_blocks - 1, -1, -1):  # 逆序遍历块索引
            if cached_block := block_pool.get_cached_block(  # 查询该块哈希的缓存块（海象赋值）
                block_hashes[i], kv_cache_group_ids  # 块哈希与分组
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                # 若该块与 alignment_tokens 不对齐，跳过前缀匹配检查
                if num_contiguous_blocks == 0 and block_size != alignment_tokens:  # 连续命中刚开始且块大小≠对齐粒度
                    post_pop_blocks = i if drop_eagle_block else i + 1  # EAGLE 丢弃后剩余的块数
                    if (post_pop_blocks * block_size) % alignment_tokens != 0:  # 不对齐
                        continue  # 跳过该块
                # Add the cached block to the computed blocks.
                # 把缓存块加入各分组的命中列表
                for computed, cached in zip(computed_blocks, cached_block):  # 遍历分组
                    computed[i] = cached  # 按索引原位写入（列表已用空块占位）
                num_contiguous_blocks += 1  # 连续命中计数加一
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:  # 连续命中达到窗口所需
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    # 裁剪命中段之后的尾部块。
                    # 例如 sliding_window_contiguous_blocks=2 时，
                    # [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    for computed in computed_blocks:  # 遍历分组
                        del computed[i + num_contiguous_blocks :]  # 删除命中段之后的所有块
                    match_found = True  # 标记找到匹配
                    break  # 提前停止搜索
            else:  # 该块未命中
                num_contiguous_blocks = 0  # 连续命中计数清零
        if not match_found:  # 未找到满足窗口的匹配
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            # 即使 num_contiguous_blocks 不足窗口所需，开头的连续命中仍算命中
            for computed in computed_blocks:  # 遍历分组
                del computed[num_contiguous_blocks:]  # 仅保留开头的连续命中块
            while (  # 对齐修正：命中长度必须是对齐粒度的倍数
                block_size != alignment_tokens  # Faster for common case.  # 常见情况（相等）下快速跳过
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0  # 长度未对齐
            ):
                for computed in computed_blocks:  # 逐块弹出直到对齐
                    computed.pop()  # 弹出末尾块
        if drop_eagle_block and computed_blocks[0]:  # 需要丢弃 EAGLE 块且存在命中
            for computed in computed_blocks:  # 每个分组
                computed.pop()  # 丢弃最后一个匹配块（强制重算以获得隐藏状态）
            # Re-align after eagle pop: the pop may break the alignment
            # when block_size != alignment_tokens (hybrid models with
            # different page sizes, e.g. Gemma4).
            # EAGLE 弹出后重新对齐：块大小≠对齐粒度时（不同页大小的混合模型，
            # 如 Gemma4），弹出可能破坏对齐
            while (  # 循环弹出直到对齐
                block_size != alignment_tokens  # 常见情况快速跳过
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0  # 仍未对齐
            ):
                for computed in computed_blocks:  # 逐块弹出
                    computed.pop()  # 弹出末尾块
        hit_length = len(computed_blocks[0]) * block_size  # 命中长度 = 命中块数 × 块大小
        return computed_blocks, hit_length  # 返回 (各分组命中块, 命中长度)

    @classmethod  # 类方法：SWA 的可达块掩码（稀疏保留策略）
    def reachable_block_mask(  # 计算哪些块值得写入缓存（能在未来命中）
        cls,
        start_block: int,  # 起始块索引
        end_block: int,  # 结束块索引（不含）
        alignment_tokens: int | None,  # 对齐粒度
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        use_eagle: bool,  # 是否启用 EAGLE
        retention_interval: int | None = None,  # 稀疏保留粒度
        reachable_boundaries: Sequence[int] = (),  # 必须保留可达尾部的 token 边界
    ) -> list[bool] | None:  # 返回每块掩码或 None（全缓存）
        assert isinstance(kv_cache_spec, SlidingWindowSpec)  # 断言规格为滑动窗口
        if alignment_tokens is None:  # 无对齐约束
            # Fast path: when the coordinator imposes no alignment constraint.
            # 快速路径：协调器未施加对齐约束
            return None  # 密集缓存所有块
        assert alignment_tokens % kv_cache_spec.block_size == 0  # 对齐粒度必须是块大小的整数倍

        block_size = kv_cache_spec.block_size  # 块大小简写
        # Contiguous blocks a hit needs at a boundary (incl. the EAGLE peek).
        # 边界处一次命中所需的连续块数（含 EAGLE 的额外一块）
        need = cls._contiguous_blocks_for_hit(  # 计算所需连续块数
            window_size=kv_cache_spec.sliding_window,  # 窗口大小
            block_size=block_size,  # 块大小
            use_eagle=use_eagle,  # EAGLE 标志
        )
        # The matched run's right edge sits on the aligned boundary block when
        # EAGLE peeks one block past it (shift=1), otherwise on the last block
        # before the boundary (shift=0).
        # 当 EAGLE 越过边界多看一块时（shift=1），匹配段的右缘落在对齐边界块上；
        # 否则落在边界前的最后一块（shift=0）
        shift = 1 if use_eagle else 0  # 计算偏移

        mask = [False] * (end_block - start_block)  # 初始化全 False 掩码

        # (1) Segment-boundary tails. ``retention_interval``:
        #   None -> dense (a tail at every ``alignment_tokens`` boundary);
        #   0    -> no dense tails (only the replay boundary below);
        #   >0   -> a tail once per ``retention_interval``-sized segment.
        # (1) 段边界尾部。retention_interval：
        #   None -> 密集（每个 alignment_tokens 边界处保留一个尾部）；
        #   0    -> 不保留密集尾部（仅保留下面的回放边界）；
        #   >0   -> 每个 retention_interval 大小的段保留一个尾部
        segment_tokens = (  # 确定段大小（token）
            alignment_tokens  # None 时用对齐粒度（密集）
            if retention_interval is None
            else (None if retention_interval == 0 else retention_interval)  # 0 表示无密集段；否则用给定间隔
        )
        if segment_tokens is not None:  # 需要保留段边界尾部
            per_segment = segment_tokens // block_size  # 每段包含的块数
            if need >= per_segment:  # 所需尾部块数不少于每段块数
                # Every block is reachable; cache them all.
                # 每个块都可达；全部缓存
                return None  # 返回 None 表示密集缓存
            for i in range(start_block, end_block):  # 遍历本次缓存范围内的每个块
                if i >= shift and (i - shift) % per_segment >= per_segment - need:  # 属于段末尾 need 个块之一
                    mask[i - start_block] = True  # 标记为可达（写入缓存）

        # (2) Reachable-boundary tails: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction. Both
        # land before segments would cover them under sparse retention, so keep
        # the ``need``-block tail ending on each boundary explicitly.
        # (2) 可达边界尾部：回放边界（num_prompt - 1，受 get_computed_blocks 钳制）
        # 与共享前缀交汇点。稀疏保留下它们可能落在段覆盖之外，
        # 因此显式保留以每个边界结尾的 need 个块的尾部
        if retention_interval is not None:  # 稀疏保留模式才需要额外保留边界尾部
            for boundary_tokens in reachable_boundaries:  # 遍历每个必须保留的 token 边界
                aligned = boundary_tokens // alignment_tokens * alignment_tokens  # 边界向下对齐到对齐粒度
                end = aligned // block_size + shift  # 尾部结束块索引（含 EAGLE 偏移）
                for j in range(max(start_block, end - need), min(end_block, end)):  # 尾部覆盖的块范围
                    mask[j - start_block] = True  # 标记为可达

        return mask  # 返回最终的每块掩码

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:  # 计算滑动窗口跳过的 token 数
        """
        Get the number of tokens that will be skipped for attention computation.
        获取注意力计算中将被跳过的 token 数。

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.
        对滑动窗口而言，这对应于当前滑动窗口之前的 token。

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.
        当前窗口包含 token 4~7。token 0~3 在滑动窗口之外，
        注意力计算时会被跳过，因此 get_num_skipped_tokens(7) == 4。

        Args:
            num_computed_tokens: The number of tokens that have been computed.
            # 已计算的 token 数

        Returns:
            The number of tokens that will be skipped for attention computation.
            # 将被跳过的 token 数
        """
        # 窗口能容纳 sliding_window 个 token（含下一个待计算 token 的位置），
        # 超出部分即被跳过；结果不为负
        return max(0, num_computed_tokens - self.sliding_window + 1)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:  # 公共前缀块数（SWA 恒为 0）
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        注：滑动窗口层的前缀块是空块（null block），因此不能像
        FullAttentionManager 那样按引用计数统计。为正确性此处返回 0。
        未来需支持级联注意力 + 滑动窗口。
        """
        return 0  # 滑动窗口不支持级联注意力的公共前缀统计


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):  # 分块局部注意力管理器（如 Gemma 系列）
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:  # 构造函数
        super().__init__(kv_cache_spec, **kwargs)  # 调用基类初始化
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size  # 保存局部注意力分块大小

    @classmethod  # 类方法：分块局部注意力的最长缓存命中查找
    def find_longest_cache_hit(  # 窗口外块置空 + 窗口内按缓存查找
        cls,
        block_hashes: BlockHashList,  # 请求的块哈希序列
        max_length: int,  # 命中前缀最大长度
        kv_cache_group_ids: list[int],  # KV cache 分组 ID 列表
        block_pool: BlockPool,  # 块池
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        drop_eagle_block: bool,  # 是否为 EAGLE/MTP 丢弃最后一块
        alignment_tokens: int,  # 对齐粒度
        dcp_world_size: int = 1,  # 解码上下文并行度
        pcp_world_size: int = 1,  # 预填充上下文并行度
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:  # 返回 (各分组命中块, 命中长度)
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local
        window, and set the block as null. Examples:
        对分块局部注意力，需找到不超过 max_length 的最长缓存命中前缀。
        该前缀必须是 kv_cache_group_ids 中所有分组的公共前缀命中。
        若未找到命中，返回空列表。注意：整块位于局部窗口之外的块
        视为已计算并置为空块。示例：

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return
        [null, null, block 3], otherwise, we return [null, null]
        1. 注意力分块大小 8、块大小 4、最大长度 15：
        下一个待计算 token 为第 15 个（从 0 计），第 8~14 个 token 在窗口内
        （需要查缓存），第 0~7 个不在窗口内，已标记为已计算。
        检查完整的 block3（第 8~11 个 token），若 block3 命中则返回
        [null, null, block 3]，否则返回 [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 blocks[null, null, null, null]
        2. 注意力分块大小 8、块大小 4、最大长度 16：
        下一个待计算 token 为第 16 个，第 0~15 个 token 都不在窗口内，
        均已标记为已计算，返回 4 个空块 [null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.  # 请求的块哈希
            max_length: The maximum length of the cache hit prefix.  # 命中前缀最大长度
            kv_cache_group_ids: The ids of the kv cache groups.  # KV cache 分组 ID
            block_pool: The block pool.  # 块池
            kv_cache_spec: The kv cache spec.  # KV cache 规格
            drop_eagle_block: Whether to drop the last matched block for EAGLE/MTP.
            # 是否为 EAGLE/MTP 丢弃最后一个匹配块
            dcp_world_size: The world size of decode context parallelism.  # 解码上下文并行度
            pcp_world_size: The world size of prefill context parallelism.  # 预填充上下文并行度
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens).
                # 返回的命中长度（token）必须是该值的倍数

        Returns:
            A list of cached blocks  # 缓存块列表
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (  # 断言规格为分块局部注意力
            "ChunkedLocalAttentionManager can only be used for "  # 仅可用于分块局部注意力分组
            "chunked local attention groups"
        )
        assert drop_eagle_block is False, (  # 断言不启用 EAGLE 丢块
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )  # 混合 KV cache 不支持 EAGLE + 分块局部注意力
        assert dcp_world_size == 1, "DCP not support chunked local attn now."  # DCP 暂不支持分块局部注意力
        assert pcp_world_size == 1, "PCP not support chunked local attn now."  # PCP 暂不支持分块局部注意力
        assert kv_cache_spec.block_size == alignment_tokens, (  # 块大小必须等于对齐粒度
            "KV cache groups with different block sizes are not compatible with "  # 不同块大小的分组目前不兼容
            "chunked local attention now"
        )
        block_hashes = resolve_block_hashes(  # 解析块哈希到分块局部注意力的块大小
            block_hashes,  # 原始块哈希
            block_pool.hash_block_size,  # 哈希粒度
            kv_cache_spec.block_size,  # 目标块大小
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,  # 细粒度支持（此处为 False）
            alignment_tokens=alignment_tokens,  # 对齐粒度
        )
        max_num_blocks = max_length // kv_cache_spec.block_size  # 最大长度内的块数
        if max_length > 0:  # 有效长度大于 0 时
            local_attention_start_idx = (  # 当前局部窗口的起始 token 索引
                max_length  # 最大长度
                // kv_cache_spec.attention_chunk_size  # 向下对齐到分块大小
                * kv_cache_spec.attention_chunk_size  # 得到分块边界的起点
            )
        else:  # 无有效长度
            local_attention_start_idx = 0  # 窗口从头开始
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        # 窗口外的块用空块标记为已计算，窗口内的块按缓存查找结果填充：
        # [null] [null] ... [null] [命中块 1（包含上一个窗口的首块）] [命中块 2] ... [命中块 x]
        local_attention_start_block_idx = (  # 窗口起点对应的块索引
            local_attention_start_idx // kv_cache_spec.block_size  # token 索引换算为块索引
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(  # 每个分组的命中块列表
            [block_pool.null_block] * local_attention_start_block_idx  # 窗口外用空块占位
            for _ in range(len(kv_cache_group_ids))  # 按分组数量创建
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):  # 从窗口起点遍历块
            block_hash = block_hashes[i]  # 当前块的哈希
            if cached_block := block_pool.get_cached_block(  # 查缓存块（海象赋值）
                block_hash, kv_cache_group_ids  # 块哈希与分组
            ):
                for computed, cached in zip(computed_blocks, cached_block):  # 遍历分组
                    computed.append(cached)  # 追加命中块
            else:  # 未命中
                break  # 链式哈希下后续也不会命中，停止
        hit_length = len(computed_blocks[0]) * kv_cache_spec.block_size  # 命中长度 = 块数 × 块大小
        return computed_blocks, hit_length  # 返回 (各分组命中块, 命中长度)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:  # 计算分块局部注意力跳过的 token 数
        """
        Get the number of tokens that will be skipped for attention computation.
        获取注意力计算中将被跳过的 token 数。

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.
        对分块局部注意力，这对应于当前分块左侧的 token。

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8
        示例 1：分块大小 8、已计算 13 个 token：下一个待计算 token 为第 13 个，
        注意力窗口覆盖第 8~15 个 token，第 0~7 个被跳过，结果为 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8
        示例 2：分块大小 8、已计算 8 个 token：下一个待计算 token 为第 8 个，
        窗口从第 8 个 token 开始，第 0~7 个全部被跳过，结果为 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0
        示例 3：分块大小 8、已计算 7 个 token：窗口仍覆盖开头，
        没有 token 被跳过，结果为 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.
            # 已计算的 token 数

        Returns:
            The number of tokens that will be skipped for attention computation.
            # 将被跳过的 token 数
        """
        num_skipped_tokens = (  # 已计算 token 向下对齐到分块大小
            num_computed_tokens // self.attention_chunk_size  # 整除得分块数
        ) * self.attention_chunk_size  # 乘回分块大小即当前分块的起点（左侧全部被跳过）
        return num_skipped_tokens  # 返回被跳过的 token 数

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:  # 公共前缀块数（分块局部恒为 0）
        """
        cascade attention is not supported by chunked local attention.
        分块局部注意力不支持级联注意力。
        """
        return 0  # 恒返回 0


class MambaManager(SingleTypeKVCacheManager):  # Mamba / 状态空间模型层的管理器
    supports_fine_grained_hash_lookup: ClassVar[bool] = True  # 支持细粒度哈希查找（状态快照可部分命中）

    def __init__(  # 构造函数
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs  # Mamba 规格、块池、其余透传参数
    ) -> None:  # 无返回值
        super().__init__(kv_cache_spec, block_pool, **kwargs)  # 调用基类初始化
        # Mamba layers use TP instead of DCP, so each rank holds the full
        # recurrent state. Undo the DCP/PCP block_size scaling that the base
        # class applies for attention groups whose KV cache is partitioned.
        # Mamba 层使用 TP 而非 DCP，每个 rank 持有完整的循环状态。
        # 撤销基类为 KV cache 被切分的注意力分组所做的 DCP/PCP block_size 放大
        self.block_size = kv_cache_spec.block_size  # 恢复为规格中的原始块大小
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode  # Mamba 缓存模式（如 "align"）
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks  # 投机解码所需块数
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()  # 本步已缓存块的哈希集合（去重用）
        if self.mamba_cache_mode == "align":  # "align" 模式（对齐式状态快照管理）
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            # 请求 ID -> 上一步分配的块索引的映射
            self.last_state_block_idx: dict[str, int] = {}  # 记录每请求上一步的状态块索引
            # The set of the requests that have been allocated blocks
            # 已分配过块的请求集合
            self._allocated_block_reqs: set[str] = set()  # 初始为空集合
            # Requests that registered their own last-prompt-boundary partial
            # tail (producers). On the next step's CoW the boundary state moves
            # into a private cow_block; we record that block for connector
            # offload (see _pending_partial_tail_offloads).
            # 注册了自己最后一个 prompt 边界处部分尾部（生产者）的请求。
            # 下一步 CoW 时边界状态会移入私有 cow_block；
            # 记录该块供 connector 卸载（见 _pending_partial_tail_offloads）
            self._producer_partial_tail_reqs: dict[str, int] = {}  # 请求 ID -> 边界 token 数的映射

    @classmethod  # 类方法：Mamba 的最长缓存命中查找
    def find_longest_cache_hit(  # 状态快照命中：只需最后一个匹配的状态块
        cls,
        block_hashes: BlockHashList,  # 请求的块哈希序列
        max_length: int,  # 命中前缀最大长度
        kv_cache_group_ids: list[int],  # KV cache 分组 ID 列表
        block_pool: BlockPool,  # 块池
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        drop_eagle_block: bool,  # 是否为 EAGLE/MTP 丢弃最后一块
        alignment_tokens: int,  # 对齐粒度
        dcp_world_size: int = 1,  # 解码上下文并行度
        pcp_world_size: int = 1,  # 预填充上下文并行度
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:  # 返回 (各分组命中块, 命中长度)
        assert isinstance(kv_cache_spec, MambaSpec), (  # 断言规格为 Mamba
            "MambaManager can only be used for mamba groups"  # 仅可用于 mamba 分组
        )
        assert dcp_world_size == 1, "DCP not support mamba now."  # DCP 暂不支持 mamba
        assert pcp_world_size == 1, "PCP not support mamba now."  # PCP 暂不支持 mamba
        block_hashes = resolve_block_hashes(  # 解析块哈希到 Mamba 的块大小
            block_hashes,  # 原始块哈希
            block_pool.hash_block_size,  # 哈希粒度
            kv_cache_spec.block_size,  # 目标块大小
            supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,  # 支持细粒度查找
            alignment_tokens=alignment_tokens,  # 对齐粒度
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(  # 每个分组一个命中块列表
            [] for _ in range(len(kv_cache_group_ids))  # 初始为空
        )
        hit_length = 0  # 命中长度初始为 0

        block_size = kv_cache_spec.block_size  # 块大小简写
        if alignment_tokens < block_size and block_size % alignment_tokens == 0:  # 细粒度模式
            # list or lazy BlobBlockHashes view
            # 块哈希是列表或惰性视图
            assert isinstance(block_hashes, Sequence)  # 断言可索引
            hash_block_size = alignment_tokens  # 哈希粒度即对齐粒度
            scale_factor = block_size // hash_block_size  # 一个块内的哈希单元数
            max_num_partial_units = min(  # 可探测的最大哈希单元数
                max_length // hash_block_size, len(block_hashes)  # 受长度与实际哈希数约束
            )
            for fine_idx in range(max_num_partial_units - 1, -1, -1):  # 从高到低探测（优先最长命中）
                num_tokens = (fine_idx + 1) * hash_block_size  # 该边界对应的 token 数
                block_hash = block_hashes[fine_idx]  # 该边界的哈希
                if cached_block := block_pool.get_cached_block(  # 查缓存的状态块
                    block_hash, kv_cache_group_ids  # 哈希与分组
                ):
                    block_idx = fine_idx // scale_factor  # 该状态块在请求块表中的索引
                    for computed, cached in zip(computed_blocks, cached_block):  # 遍历分组
                        computed.extend([block_pool.null_block] * block_idx)  # 前面用空块占位
                        computed.append(cached)  # 追加命中的状态块
                    hit_length = num_tokens  # 记录命中长度
                    break  # 找到最长命中即停止
            return computed_blocks, hit_length  # 细粒度模式直接返回

        max_num_blocks = max_length // block_size  # 最大长度内的块数
        # Search from right to left and early stop when a match is found.
        # 从右向左搜索，找到匹配即提前停止（Mamba 只需最后一个状态快照）
        for i in range(max_num_blocks - 1, -1, -1):  # 逆序遍历块索引
            if cached_block := block_pool.get_cached_block(  # 查该块哈希的缓存状态块
                block_hashes[i], kv_cache_group_ids  # 块哈希与分组
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                # 启用 Mamba 前缀缓存时，block_size 会在全注意力层与 Mamba 层之间
                # 对齐，以保证前缀命中长度按块对齐
                if (
                    block_size != alignment_tokens  # Faster for common case.  # 常见情况快速跳过
                    and (i + 1) * block_size % alignment_tokens != 0  # 命中长度未对齐
                ):
                    continue  # 跳过该块，尝试更短的
                for computed, cached in zip(computed_blocks, cached_block):  # 遍历分组
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    # 后续命中长度逻辑假设块数可直接换算长度，
                    # 因此在开头插入空块占位：
                    computed.extend([block_pool.null_block] * i)  # 前面用空块占位
                    computed.append(cached)  # 追加命中的状态块
                hit_length = (i + 1) * block_size  # 命中长度 = (块索引+1) × 块大小
                break  # we just need the last match - early stopping  # 只需最后一个匹配，提前停止

        return computed_blocks, hit_length  # 返回 (各分组命中块, 命中长度)

    @classmethod  # 类方法：Mamba 的可达块掩码（稀疏状态快照保留）
    def reachable_block_mask(  # 决定哪些状态快照块值得写入缓存
        cls,
        start_block: int,  # 起始块索引
        end_block: int,  # 结束块索引（不含）
        alignment_tokens: int | None,  # 对齐粒度
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        use_eagle: bool,  # 是否启用 EAGLE（Mamba 不使用）
        retention_interval: int | None = None,  # 稀疏保留粒度
        reachable_boundaries: Sequence[int] = (),  # 必须保留的 token 边界
    ) -> list[bool] | None:  # 返回每块掩码或 None（全缓存）
        """Sparse Mamba state-snapshot retention.
        Mamba 状态快照的稀疏保留。

        ``retention_interval``:

          ``None`` -> dense (cache every block; default, unchanged behavior)
          ``0``    -> keep only the ``reachable_boundaries`` states
          ``> 0``  -> keep one state per ``retention_interval``-sized segment
          None -> 密集（缓存所有块；默认，行为不变）
          0    -> 仅保留 reachable_boundaries 处的状态
          > 0  -> 每个 retention_interval 大小的段保留一个状态

        ``reachable_boundaries`` are proven reuse points (the replay boundary and
        any cross-request shared-prefix junction, Marconi-style APC); their
        boundary state is always kept so sparse retention does not defeat reuse.
        reachable_boundaries 是经过验证的复用点（回放边界与跨请求共享前缀
        交汇点，即 Marconi 风格的 APC）；其边界状态始终保留，
        以免稀疏保留破坏复用。
        """
        if retention_interval is None or alignment_tokens is None:  # 密集缓存或无对齐约束
            # Dense caching (default) or no alignment constraint imposed.
            # 密集缓存（默认）或未施加对齐约束
            return None  # 返回 None：全部缓存
        assert isinstance(kv_cache_spec, MambaSpec)  # 断言规格为 Mamba
        block_size = kv_cache_spec.block_size  # 块大小简写
        mask = [False] * (end_block - start_block)  # 初始化全 False 掩码

        # (1) Segment-boundary states. A Mamba hit needs exactly the single
        # state block ending on the boundary (no window, and draft models have
        # no mamba layers, so no eagle shift). Block ``i`` ends at token
        # ``(i + 1) * block_size``.
        # (1) 段边界状态。Mamba 命中恰好需要以边界结尾的单个状态块
        # （没有窗口，且草稿模型无 mamba 层，故无 eagle 偏移）。
        # 块 i 结束于 token (i + 1) * block_size
        segment_tokens = None if retention_interval == 0 else retention_interval  # 0 表示无段边界
        if segment_tokens is not None:  # 需要保留段边界状态
            per_segment = segment_tokens // block_size  # 每段包含的块数
            if per_segment <= 1:  # 间隔不超过一个块
                # Interval at/below the block size: every block is a boundary.
                # 间隔等于或小于块大小：每个块都是边界
                return None  # 全部缓存
            first_boundary = (  # 范围内第一个段边界块索引
                start_block + per_segment  # 起点向后推一段
            ) // per_segment * per_segment - 1  # 对齐到段边界（边界为段末块）
            for i in range(first_boundary - start_block, len(mask), per_segment):  # 每隔一段标记一个边界块
                mask[i] = True  # 标记为可达

        # (2) Reachable-boundary states: the replay boundary (``num_prompt - 1``,
        # capped by ``get_computed_blocks``) and any shared-prefix junction, both
        # of which segments would otherwise skip under sparse retention. A Mamba
        # hit needs exactly the single state block ending on the boundary.
        # (2) 可达边界状态：回放边界（num_prompt - 1，受 get_computed_blocks 钳制）
        # 与共享前缀交汇点，稀疏保留下段机制可能跳过它们。
        # Mamba 命中恰好需要以边界结尾的单个状态块
        for boundary_tokens in reachable_boundaries:  # 遍历每个边界
            aligned = boundary_tokens // alignment_tokens * alignment_tokens  # 边界向下对齐
            boundary_block = aligned // block_size - 1  # 以该边界结尾的状态块索引
            if start_block <= boundary_block < end_block:  # 在缓存范围内
                mask[boundary_block - start_block] = True  # 标记为可达

        return mask  # 返回最终掩码

    def remove_skipped_blocks(  # 覆盖：释放 Mamba 不再需要的状态块
        self,
        request_id: str,  # 请求 ID
        processed_computed_tokens: int,  # 已完整处理的已计算 token 前缀长度
        num_prompt_tokens: int | None = None,  # 可选 prompt 长度（此处未使用）
    ) -> None:  # 无返回值
        assert isinstance(self.kv_cache_spec, MambaSpec)  # 断言规格为 Mamba

        super().remove_skipped_blocks(  # 先走基类逻辑（按 get_num_skipped_tokens 释放头部块）
            request_id, processed_computed_tokens, num_prompt_tokens  # 透传参数
        )
        if self.mamba_cache_mode == "align":  # "align" 模式需要额外释放旧状态块
            # `last_state_block_idx` refers to the block index allocated two steps ago.
            # The block allocated in the previous step is used to copy Mamba states
            # into the block allocated in the current step; the earlier block is
            # no longer needed and should be freed here.
            # last_state_block_idx 指向前两步分配的块索引。上一步分配的块
            # 用于把 Mamba 状态拷贝到当前步分配的块中；更早的块
            # 已不再需要，应在此释放
            last_state_block_idx = self.last_state_block_idx.get(request_id)  # 取上上步的状态块索引
            # Blocks allocated during prefill may be non-contiguous. Use
            # `last_state_block_idx` to free the appropriate block and replace it
            # with a null block.
            # prefill 期间分配的块可能不连续。用 last_state_block_idx
            # 释放对应的块并用空块替换
            if (
                last_state_block_idx is not None  # 确实记录过旧状态块
                and last_state_block_idx
                < cdiv(processed_computed_tokens, self.block_size) - 1  # 且不是当前正在用的状态块
            ):
                blocks = self.req_to_blocks[request_id]  # 获取请求块表
                if blocks[last_state_block_idx] != self._null_block:  # 该块尚未被释放
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])  # 交块池释放
                    blocks[last_state_block_idx] = self._null_block  # 块表中置为空块

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:  # 公共前缀块数（Mamba 恒为 0）
        """
        cascade attention is not supported by mamba
        mamba 不支持级联注意力
        """
        return 0  # 恒返回 0

    def get_num_blocks_to_allocate(  # 覆盖：Mamba 特有的块数估算
        self,
        request_id: str,  # 请求 ID
        num_tokens: int,  # 需要槽位的总 token 数
        new_computed_blocks: Sequence[KVCacheBlock],  # 新命中的缓存块
        total_computed_tokens: int,  # 已计算 token 总数
        num_local_computed_tokens: int,  # 本地命中的已计算 token 数
        num_tokens_main_model: int,  # 主模型 token 数
        apply_admission_cap: bool = False,  # 是否应用准入上限
    ) -> int:  # 返回还需分配的块数
        assert isinstance(self.kv_cache_spec, MambaSpec)  # 断言规格为 Mamba
        if (
            len(new_computed_blocks) > 0  # 有命中块
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step  # 且最后一块是本步刚缓存的
        ):
            # Mamba can't rely on blocks generated by other requests in the current step
            # To put it in the next step, we return num_gpu_blocks + 1 so
            # that kv_cache_manager will think there is no enough blocks to allocate now
            # and don't schedule it in the current step.
            # Mamba 不能依赖本步中由其他请求生成的块。
            # 为把它推迟到下一步，返回 num_gpu_blocks + 1，
            # 使 kv_cache_manager 认为当前没有足够的块可分配，
            # 从而本步不调度该请求
            return self.block_pool.num_gpu_blocks + 1  # 返回不可能满足的巨大值
        if self.mamba_cache_mode != "align":  # 非 align 模式
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            # 为线性注意力的投机解码（MTP/EAGLE）额外分配 num_speculative_blocks 个块
            if self.num_speculative_blocks > 0:  # 需要投机块时
                num_tokens += (  # token 数加上投机块覆盖的 token
                    self.kv_cache_spec.block_size * self.num_speculative_blocks  # 块大小 × 投机块数
                )
            return super().get_num_blocks_to_allocate(  # 走基类通用估算
                request_id,  # 请求 ID
                num_tokens,  # 调整后的 token 数
                new_computed_blocks,  # 命中块
                total_computed_tokens,  # 已计算总数
                num_local_computed_tokens,  # 本地命中数
                num_tokens_main_model,  # 主模型 token 数
                apply_admission_cap=apply_admission_cap,  # 准入上限标志
            )
        else:  # align 模式
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            # align 模式下不为前瞻 token 分配块，因为若调度了 x * block_size 个 token，
            # num_tokens 会是 x * block_size + num_lookahead_tokens，破坏对齐。
            # 当前草稿模型没有 mamba 层，因此可以忽略前瞻 token
            num_tokens = num_tokens_main_model  # 只用主模型 token 数

            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            # 注：这是对所需块数的高估，因为 num_tokens 可能包含
            # 之后会被拒绝的草稿 token
            num_required_blocks = (  # 所需块数 = token 块数 + 投机块数
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks  # 向上取整 + 投机块
            )
            num_new_blocks = (  # 还需分配的块数
                num_required_blocks  # 所需块数
                - len(new_computed_blocks)  # 减去命中块
                - len(self.req_to_blocks[request_id])  # 减去已持有块
            )
            has_partial_hit = (  # 是否存在部分命中
                self._has_partial_local_hit(  # 本次命中是否以部分命中收尾
                    new_computed_blocks, num_local_computed_tokens  # 命中块与本地 token 数
                )
                or request_id in self._partial_hit_reqs  # 或已登记过部分命中
            )
            if has_partial_hit:  # 有部分命中
                num_new_blocks = max(num_new_blocks, 0) + 1  # 额外预留一个 CoW 块
            if num_new_blocks > 0:  # 确实需要新块
                if request_id in self._allocated_block_reqs:  # 旧请求（已分配过块）
                    # Old request. Needs at most 1 more blocks as we can reuse the
                    # speculative blocks in previous step.
                    # 旧请求最多再需要 1 个块，因为可复用
                    # 上一步的投机块
                    num_new_blocks = 1 + int(has_partial_hit)  # 1 个运行状态块（+可能的 CoW 块）
                else:  # 首次 prefill
                    # First prefill. Allocate 1 block for running state, the
                    # speculative blocks, and one extra block if a partial cache
                    # hit must be copy-on-written before the new tokens run.
                    # 首次 prefill：分配 1 个运行状态块、
                    # 投机块，若部分命中需要在新 token 运行前 CoW，
                    # 再额外分配一个块
                    num_new_blocks = (  # 1 + 投机块数 + 可能的 CoW 块
                        1 + self.num_speculative_blocks + int(has_partial_hit)  # 求和
                    )

            num_evictable_computed_blocks = self._get_num_evictable_blocks(  # 命中块中可驱逐的数量
                new_computed_blocks  # 统计所有命中块
            )
            return num_new_blocks + num_evictable_computed_blocks  # 新块 + 会占空闲额度的可驱逐块

    def allocate_new_blocks(  # 覆盖：Mamba align 模式的块分配
        self, request_id: str, num_tokens: int, num_tokens_main_model: int  # 请求 ID、总 token 数、主模型 token 数
    ) -> list[KVCacheBlock]:  # 返回新分配的块列表
        assert isinstance(self.kv_cache_spec, MambaSpec)  # 断言规格为 Mamba
        if self.mamba_cache_mode != "align":  # 非 align 模式
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            # 为线性注意力的投机解码（MTP/EAGLE）额外分配 num_speculative_blocks 个块
            if self.num_speculative_blocks > 0:  # 需要投机块时
                num_tokens += self.block_size * self.num_speculative_blocks  # token 数加上投机块覆盖的 token
            return super().allocate_new_blocks(  # 走基类通用分配
                request_id, num_tokens, num_tokens_main_model  # 透传参数
            )
        else:  # align 模式
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            # align 模式下不为前瞻 token 分配块（原因同 get_num_blocks_to_allocate）
            num_tokens = num_tokens_main_model  # 只用主模型 token 数
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]  # 获取请求块表
            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            # 注：这是对所需块数的高估（草稿 token 可能被拒绝）
            num_required_blocks = (  # 所需块数 = token 块数 + 投机块数
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks  # 向上取整 + 投机块
            )
            partial_hit = self._partial_hit_reqs.get(request_id)  # 取该请求的部分命中记录（可能为 None）
            has_partial_hit = partial_hit is not None  # 是否存在部分命中
            # `num_required_blocks` might be less than `len(req_blocks)` if blocks are
            # over-allocated at last round.
            # 若上一轮过度分配了块，num_required_blocks 可能小于 len(req_blocks)
            if num_required_blocks <= len(req_blocks) and not has_partial_hit:  # 块已足够且无部分命中
                return []  # 无需分配
            else:  # 需要分配或需要 CoW
                prev_block_len = len(req_blocks)  # 分配前的块数
                blocks_allocated = request_id in self._allocated_block_reqs  # 该请求是否已分配过块（旧请求）
                # Record the last state block
                # 记录最后的状态块索引
                if blocks_allocated:  # 旧请求
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    # 运行状态始终保存在倒数 (1 + num_speculative_blocks) 个块处
                    self.last_state_block_idx[request_id] = (  # 记录上一步的运行状态块索引
                        prev_block_len - 1 - self.num_speculative_blocks  # 末尾减去投机块数再减 1
                    )
                elif prev_block_len > 0:  # 新请求但已有命中块
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    # 新请求命中前缀缓存时，最后一个块保存命中状态
                    self.last_state_block_idx[request_id] = prev_block_len - 1  # 命中状态在最后一个块

                num_skipped_blocks = (  # 不需要保存状态的块数（除运行状态块与投机块外）
                    num_required_blocks - self.num_speculative_blocks - 1  # 所需块 - 投机块 - 运行状态块
                )
                # null blocks
                # 用空块占位被跳过的状态位置
                if prev_block_len < num_skipped_blocks:  # 现有块数不足跳过数
                    req_blocks.extend(  # 补空块直到 num_skipped_blocks
                        [
                            self._null_block  # 空块占位
                            for _ in range(prev_block_len, num_skipped_blocks)  # 补齐差额
                        ]
                    )

                if blocks_allocated:  # 旧请求：复用投机块
                    # reuse previous speculative blocks in this step
                    # 本步复用上一步的投机块
                    for block_idx in range(  # 遍历末尾的投机块
                        prev_block_len - self.num_speculative_blocks, prev_block_len  # 投机块区间
                    ):
                        if block_idx < num_skipped_blocks:  # 该投机块落在跳过区
                            req_blocks.append(req_blocks[block_idx])  # 移到末尾继续用作投机块
                            req_blocks[block_idx] = self._null_block  # 原位置置空
                        else:  # 已超出跳过区
                            break  # 无需再移动
                num_new_blocks = num_required_blocks - len(req_blocks)  # 还差多少块
                if has_partial_hit:  # 有部分命中
                    num_new_blocks = max(num_new_blocks, 0) + 1  # 额外预留 CoW 块
                if blocks_allocated:  # 旧请求
                    assert num_new_blocks <= 1 + int(has_partial_hit)  # 断言：最多 1 个运行状态块（+CoW）
                else:  # 首次分配
                    assert num_new_blocks <= self.num_speculative_blocks + 1 + int(  # 断言：投机块 + 运行状态块（+CoW）
                        has_partial_hit
                    )
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)  # 从块池取新块
                returned_blocks = req_blocks[prev_block_len:]  # 本次新增的块（用于返回给调用方）
                if partial_hit is not None:  # 存在部分命中需要 CoW
                    block_idx, source_block = partial_hit  # 解包 (块索引, 源共享块)
                    cow_block = new_blocks[0]  # 第一个新块作为 CoW 目标块
                    new_blocks = new_blocks[1:]  # 剩余块留作普通分配
                    if blocks_allocated:  # 旧请求的 CoW 处理
                        # The worker block table of a running request is
                        # append-only, so the request must stay on
                        # source_block. Move the cache entry to cow_block
                        # instead; the queued copy fills it before forward
                        # overwrites source_block.
                        # 运行中请求的 worker 块表是只追加的，
                        # 因此请求必须留在 source_block 上。
                        # 改为把缓存条目移到 cow_block；
                        # 排队的拷贝会在 forward 覆盖 source_block 之前填充它
                        assert req_blocks[block_idx] is source_block  # 断言该位置是源块
                        self.block_pool.move_block_hashes(source_block, cow_block)  # 缓存条目从源块移到 CoW 块
                        self._pending_cow_copies.append((source_block, cow_block))  # 登记待执行拷贝
                        source_block.ref_cnt += 1  # 源块加引用，防止拷贝前被回收
                        boundary_tokens = self._producer_partial_tail_reqs.pop(  # 取生产者边界 token 数（若有）
                            request_id, None  # 不存在返回 None
                        )
                        if boundary_tokens is not None:  # 是生产者请求
                            # This CoW preserved a producer's own boundary
                            # state in cow_block; hand it to the connector for
                            # partial-tail offload once the copy has run.
                            # 本次 CoW 在 cow_block 中保存了生产者自己的边界状态；
                            # 拷贝完成后交给 connector 做部分尾部卸载
                            self._pending_partial_tail_offloads.append(  # 登记卸载交接
                                (
                                    request_id,  # 请求 ID
                                    self.kv_cache_group_id,  # 分组 ID
                                    cow_block,  # CoW 块
                                    boundary_tokens,  # 边界 token 数
                                )
                            )
                        if cow_block.block_hash is not None:  # CoW 块有哈希
                            # The moved entry is only filled by this step's
                            # copy, so defer same-step hits on it.
                            # 移过来的条目要等本步拷贝完成后才有效，
                            # 因此推迟同一步内对它的命中
                            self.cached_blocks_this_step.add(cow_block.block_hash)  # 记入本步缓存集合
                    else:  # 新请求的 CoW 处理
                        self._apply_cow(request_id, block_idx, source_block, cow_block)  # 标准 CoW 重定向
                        returned_blocks = [cow_block] + returned_blocks  # CoW 块也计入返回块
                req_blocks.extend(new_blocks)  # 追加剩余新块到请求块表
                self._allocated_block_reqs.add(request_id)  # 标记该请求已分配过块
                self._partial_hit_reqs.pop(request_id, None)  # 清除已处理的部分命中记录
                returned_blocks.extend(new_blocks)  # 新块计入返回
                return returned_blocks  # 返回本步新增的所有块

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:  # 覆盖：弹出块前清理 align 模式状态
        if self.mamba_cache_mode == "align":  # align 模式有额外记账需要清理
            self._allocated_block_reqs.discard(request_id)  # 从已分配集合中移除
            self.last_state_block_idx.pop(request_id, None)  # 清除状态块索引记录
            self._producer_partial_tail_reqs.pop(request_id, None)  # 清除生产者部分尾部记录
            # A hand-off whose request died in this same scheduling pass must
            # not reach the connector: its unpin hook (free) has already run.
            # 同一调度轮内死亡的请求，其交接不能到达 connector：
            # 它的 unpin 钩子（free）已经执行过了
            self._pending_partial_tail_offloads = [  # 过滤掉该请求的待卸载交接
                entry  # 保留的交接项
                for entry in self._pending_partial_tail_offloads  # 遍历现有交接
                if entry[0] != request_id  # 排除属于该请求的项
            ]
        return super().pop_blocks_for_free(request_id)  # 调用基类完成块弹出

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:  # 计算 Mamba 可跳过的 token 数
        """
        Get the number of tokens whose mamba state are not needed anymore. Mamba only
        need to keep the state of the last computed token, so we return
        num_computed_tokens - 1.
        获取其 mamba 状态不再需要的 token 数。Mamba 只需保留
        最后一个已计算 token 的状态，因此返回 num_computed_tokens - 1。
        """
        return num_computed_tokens - 1  # 除最后一个 token 外全部可跳过

    def cache_blocks(  # 覆盖：缓存块并记录本步缓存的哈希
        self,
        request: Request,  # 请求对象
        num_tokens: int,  # 需缓存的总 token 数
        retention_interval: int | None = None,  # 稀疏保留粒度
    ) -> None:  # 无返回值
        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)  # 缓存前的已缓存块数
        super().cache_blocks(request, num_tokens, retention_interval=retention_interval)  # 走基类缓存逻辑
        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)  # 缓存后的已缓存块数
        if self.mamba_cache_mode == "align":  # align 模式额外缓存部分尾块
            partial_hash = self._cache_partial_tail_block(request, num_tokens)  # 缓存 prompt 尾部（若在块内部结束）
            if partial_hash is not None:  # 确实注册了部分哈希
                self.cached_blocks_this_step.add(partial_hash)  # 记入本步缓存集合
        if num_cached_blocks_after > num_cached_blocks_before:  # 本次有新增缓存块
            for block in self.req_to_blocks[request.request_id][  # 遍历新增缓存的块
                num_cached_blocks_before:num_cached_blocks_after  # 新增区间
            ]:
                # Skip null blocks (align-mode skipped states) and blocks that
                # were not cached this step — with sparse retention
                # (reachable_block_mask) the intermediate state snapshots carry
                # no hash and must not be recorded as cached-this-step.
                # 跳过空块（align 模式被跳过的状态）与本步未缓存的块——
                # 稀疏保留（reachable_block_mask）下中间状态快照
                # 不带哈希，不能记为本步已缓存
                if block.is_null or block.block_hash is None:  # 空块或无哈希
                    continue  # 跳过
                self.cached_blocks_this_step.add(block.block_hash)  # 记入本步缓存集合

    def new_step_starts(self) -> None:  # 新调度步开始：清空本步缓存记录
        self.cached_blocks_this_step.clear()  # 清空集合

    def _cache_partial_tail_block(  # Mamba align 模式：缓存 prompt 部分尾部状态块
        self,
        request: Request,  # 请求对象
        num_tokens: int,  # 当前需缓存的 token 数
    ) -> BlockHashWithGroupId | None:  # 返回注册的部分哈希（无则 None）
        hash_block_size = self.block_pool.hash_block_size  # 块池的哈希粒度
        if self.block_size == hash_block_size:  # 块大小与哈希粒度一致
            return None  # 不存在块内边界，无需处理
        if num_tokens % self.block_size == 0:  # 恰好落在整块边界
            return None  # 无部分尾部
        if num_tokens % hash_block_size != 0:  # 未落在哈希边界上
            return None  # 无法注册部分哈希
        latest_prompt_hash_boundary = (  # 最后一个 prompt 哈希边界的 token 位置
            request.num_prompt_tokens // hash_block_size  # prompt 长度向下对齐
        ) * hash_block_size  # 得到边界 token 数
        if num_tokens != latest_prompt_hash_boundary:  # 当前缓存量不是最后一个 prompt 边界
            return None  # 只缓存最终 prompt 边界，其余跳过

        block_idx = num_tokens // self.block_size  # 该边界所在块的索引
        blocks = self.req_to_blocks[request.request_id]  # 获取请求块表
        if block_idx >= len(blocks):  # 块尚未分配
            return None  # 无法缓存
        source_block = blocks[block_idx]  # 目标源块
        if source_block.is_null:  # 空块无法承载状态
            return None  # 跳过

        partial_hash = self.block_pool.cache_partial_block(  # 通过块池注册部分块缓存条目
            request=request,  # 请求对象
            block=source_block,  # 目标块
            num_tokens=num_tokens,  # 该块内有效的 token 数
            kv_cache_group_id=self.kv_cache_group_id,  # 分组 ID
            block_size=self.block_size,  # 块大小
        )
        if partial_hash is not None:  # 注册成功
            self._partial_hit_reqs[request.request_id] = (block_idx, source_block)  # 记录 (块索引, 源块) 供后续 CoW
            self.num_cached_block[request.request_id] = block_idx  # 已缓存数回退到完整块数
            # Producer of this partial tail: the boundary state currently lives
            # in ``source_block`` but the next step's forward overwrites it. The
            # upcoming CoW copies it into a durable cow_block; record the req so
            # allocate_new_blocks hands that block to the connector for offload.
            # 该部分尾部的生产者：边界状态当前在 source_block 中，
            # 但下一步的 forward 会覆盖它。即将到来的 CoW 会把它
            # 拷贝到持久的 cow_block；记录该请求，使
            # allocate_new_blocks 把该块交给 connector 卸载
            self._producer_partial_tail_reqs[request.request_id] = num_tokens  # 记录生产者边界 token 数
        return partial_hash  # 返回部分哈希


class CrossAttentionManager(SingleTypeKVCacheManager):  # 交叉注意力管理器
    """Manager for cross-attention KV cache in encoder-decoder models.
    编码器-解码器模型中交叉注意力 KV cache 的管理器。
    """

    def add_local_computed_blocks(  # 覆盖：交叉注意力不接受缓存命中块
        self,
        request_id: str,  # 请求 ID
        new_computed_blocks: Sequence[KVCacheBlock],  # 新命中的缓存块
        num_local_computed_tokens: int,  # 本地已计算 token 数
        num_external_computed_tokens: int,  # 外部已计算 token 数
    ) -> None:  # 无返回值
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        # 交叉注意力的块不做跨请求共享缓存，
        # 因此 new_computed_blocks 必须始终为空
        assert len(new_computed_blocks) == 0  # 断言没有命中块

    def allocate_external_computed_blocks(  # 覆盖：交叉注意力不使用外部 KV
        self,
        request_id: str,  # 请求 ID
        num_local_computed_tokens: int,  # 本地已计算 token 数
        num_external_computed_tokens: int,  # 外部已计算 token 数
    ) -> None:  # 无返回值
        # Cross-attention does not use prefix caching / external KV loads.
        # 交叉注意力不使用前缀缓存 / 外部 KV 加载
        return  # 直接返回，什么都不做

    def cache_blocks(  # 覆盖：交叉注意力不缓存块
        self,
        request: Request,  # 请求对象
        num_tokens: int,  # 需缓存的 token 数
        retention_interval: int | None = None,  # 稀疏保留粒度
    ) -> None:  # 无返回值
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        # 交叉注意力的块不做跨请求共享缓存，因此本方法不应被调用
        raise ValueError("Should not be called as prefix caching is disabled.")  # 抛出异常：前缀缓存已禁用

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:  # 公共前缀块数恒为 0
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        # 交叉注意力块包含请求私有的编码器状态，
        # 不在不同请求之间共享
        return 0  # 返回 0

    @classmethod  # 类方法：交叉注意力不支持缓存命中
    def find_longest_cache_hit(  # 直接抛出未实现异常
        cls,
        block_hashes: BlockHashList,  # 请求的块哈希序列
        max_length: int,  # 命中前缀最大长度
        kv_cache_group_ids: list[int],  # KV cache 分组 ID 列表
        block_pool: BlockPool,  # 块池
        kv_cache_spec: KVCacheSpec,  # KV cache 规格
        drop_eagle_block: bool,  # 是否为 EAGLE/MTP 丢弃最后一块
        alignment_tokens: int,  # 对齐粒度
        dcp_world_size: int = 1,  # 解码上下文并行度
        pcp_world_size: int = 1,  # 预填充上下文并行度
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:  # 返回类型（实际不会返回）
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (  # 断言规格为交叉注意力
            "CrossAttentionManager can only be used for cross-attention groups"  # 仅可用于交叉注意力分组
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        # 交叉注意力无法从前缀缓存中受益，原因：
        # 1. 编码器状态每请求唯一（不同的音频/图像输入）
        # 2. 编码器状态每请求只计算一次，非增量式
        # 3. 不同多模态输入之间不存在可复用的前缀
        # 返回空块表示无缓存命中
        raise NotImplementedError("CrossAttentionManager does not support caching")  # 不支持缓存


class SinkFullAttentionManager(FullAttentionManager):  # 带 sink 块的全注意力管理器（StreamingLLM 风格）
    def __init__(  # 构造函数：初始化时预分配 sink 块
        self,
        kv_cache_spec: SinkFullAttentionSpec,  # sink 全注意力规格
        block_pool: BlockPool,  # 块池
        enable_caching: bool,  # 是否启用前缀缓存
        kv_cache_group_id: int,  # KV cache 分组 ID
        scheduler_block_size: int,  # 调度粒度
        dcp_world_size: int = 1,  # 解码上下文并行度
        pcp_world_size: int = 1,  # 预填充上下文并行度
    ):  # 无返回值
        super().__init__(  # 调用全注意力管理器初始化
            kv_cache_spec=kv_cache_spec,  # 规格
            block_pool=block_pool,  # 块池
            enable_caching=enable_caching,  # 缓存开关
            kv_cache_group_id=kv_cache_group_id,  # 分组 ID
            scheduler_block_size=scheduler_block_size,  # 调度粒度
            dcp_world_size=dcp_world_size,  # DCP 并行度
            pcp_world_size=pcp_world_size,  # PCP 并行度
        )
        sink_len = kv_cache_spec.sink_len  # sink 区域的 token 长度
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0  # sink 长度必须为正且整块对齐
        num_sink_block = sink_len // self.block_size  # 需要的 sink 块数
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)  # 从空闲队列头部预取 sink 块


def get_manager_for_kv_cache_spec(  # 工厂函数：根据规格创建对应的管理器
    kv_cache_spec: KVCacheSpec,  # KV cache 规格实例
    max_in_flight_tokens: int,  # 已调度但未落定的最大 token 数
    max_model_len: int,  # 模型可服务的最大上下文长度
    **kwargs,  # 透传给管理器构造函数的其余参数
) -> SingleTypeKVCacheManager:  # 返回对应类型的管理器实例
    """
    Get the appropriate manager for a given KVCacheSpec.
    为给定的 KVCacheSpec 获取合适的管理器。

    Uses the KVCacheSpecRegistry to look up the manager class, supporting
    both built-in and custom specs registered via @register_kv_cache_spec
    and KVCacheSpecRegistry.register.
    通过 KVCacheSpecRegistry 查找管理器类，同时支持内置规格与
    经 @register_kv_cache_spec 和 KVCacheSpecRegistry.register
    注册的自定义规格。

    Args:
        kv_cache_spec: The KVCacheSpec instance  # KVCacheSpec 实例
        max_in_flight_tokens: The max tokens scheduled but not yet settled
            (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`
            # 已调度但尚未落定的最大 token 数（每个并发步一个批次）；
            # 见 VllmConfig.max_in_flight_tokens
        max_model_len: The maximum context length the model could serve
        # 模型可服务的最大上下文长度
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass
        # 合适的 SingleTypeKVCacheManager 子类实例
    """
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)  # 从注册表查管理器类
    assert manager_class is not None, (  # 断言已注册
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"  # 未注册的规格报错
    )
    # SlidingWindow / ChunkedLocalAttention managers recycle blocks;
    # the runtime admission cap must match the recycling-aware bound the
    # startup pool sizer uses (single source of truth: the spec method).
    # R-SWA also recycles gap blocks but peak physical KV still fits the
    # full-attention bound (prefix + window <= max_model_len), so it inherits
    # FullAttentionSpec sizing without a separate admission cap.
    # 滑动窗口 / 分块局部注意力管理器会回收块；
    # 运行时准入上限必须与启动时池容量估算使用的感知回收边界一致
    # （唯一事实来源：规格上的方法）。
    # R-SWA 也回收空隙块，但峰值物理 KV 仍满足全注意力边界
    # （前缀 + 窗口 <= max_model_len），因此沿用
    # FullAttentionSpec 的容量估算，不设单独的准入上限
    if isinstance(
        kv_cache_spec,  # 规格实例
        (SlidingWindowSpec, ChunkedLocalAttentionSpec),  # 滑动窗口或分块局部注意力
    ):
        kwargs["max_admission_blocks_per_request"] = (  # 注入每请求准入块数上限
            kv_cache_spec.max_admission_blocks_per_request(  # 由规格方法计算（单一事实来源）
                max_in_flight_tokens=max_in_flight_tokens,  # 在途 token 上限
                max_model_len=max_model_len,  # 最大模型长度
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)  # 实例化管理器
    return manager  # 返回管理器实例


def register_all_kvcache_specs(vllm_config):  # 注册所有内置 KV cache 规格
    """Built-in spec registration 内置规格注册"""
    KVCacheSpecRegistry.register(  # 注册全注意力规格
        FullAttentionSpec,  # 规格类
        FullAttentionManager,  # 管理器类
        uniform_type_base_spec=FullAttentionSpec,  # 统一类型基规格
    )

    KVCacheSpecRegistry.register(  # 注册滑动窗口规格
        SlidingWindowSpec,  # 规格类
        SlidingWindowManager,  # 管理器类
        uniform_type_base_spec=SlidingWindowSpec,  # 统一类型基规格
    )
    KVCacheSpecRegistry.register(  # 注册 MLA 滑动窗口规格
        SlidingWindowMLASpec,  # 规格类
        SlidingWindowManager,  # 复用滑动窗口管理器
        uniform_type_base_spec=SlidingWindowMLASpec,  # 统一类型基规格
    )

    KVCacheSpecRegistry.register(  # 注册 Mamba 规格
        MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec  # 规格、管理器、基规格
    )
    KVCacheSpecRegistry.register(  # 注册分块局部注意力规格
        ChunkedLocalAttentionSpec,  # 规格类
        ChunkedLocalAttentionManager,  # 管理器类
        uniform_type_base_spec=ChunkedLocalAttentionSpec,  # 统一类型基规格
    )
    KVCacheSpecRegistry.register(  # 注册交叉注意力规格
        CrossAttentionSpec,  # 规格类
        CrossAttentionManager,  # 管理器类
        uniform_type_base_spec=CrossAttentionSpec,  # 统一类型基规格
    )

    # FullAttentionSpec subclasses — grouped with FullAttentionSpec
    # FullAttentionSpec 的子类 —— 与 FullAttentionSpec 归为一组
    KVCacheSpecRegistry.register(  # 注册 TQ 全注意力规格
        TQFullAttentionSpec,  # 规格类
        FullAttentionManager,  # 复用全注意力管理器
        uniform_type_base_spec=FullAttentionSpec,  # 基规格为全注意力
    )
    KVCacheSpecRegistry.register(  # 注册 MLA 注意力规格
        MLAAttentionSpec, FullAttentionManager, uniform_type_base_spec=FullAttentionSpec  # 规格、管理器、基规格
    )
    KVCacheSpecRegistry.register(  # 注册 R-SWA 规格
        RSWASpec, RSWAManager, uniform_type_base_spec=FullAttentionSpec  # 规格、管理器、基规格为全注意力
    )
    # NOTE(Mengqing): HiddenStateCacheSpec won't take part in
    # grouping, thus the uniform_type_base_spec is just a
    # placeholder.
    # 注（Mengqing）：HiddenStateCacheSpec 不参与分组，
    # 因此 uniform_type_base_spec 仅是占位符
    KVCacheSpecRegistry.register(  # 注册隐藏状态缓存规格
        HiddenStateCacheSpec,  # 规格类
        FullAttentionManager,  # 复用全注意力管理器
        uniform_type_base_spec=FullAttentionSpec,  # 占位基规格
    )
    KVCacheSpecRegistry.register(  # 注册 sink 全注意力规格
        SinkFullAttentionSpec,  # 规格类
        SinkFullAttentionManager,  # 管理器类
        uniform_type_base_spec=FullAttentionSpec,  # 基规格为全注意力
    )

    from vllm.platforms import current_platform  # 延迟导入平台对象

    current_platform.register_custom_kv_cache_specs(vllm_config)  # 注册平台自定义规格
