# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""
# 定义 KV 连接器功能的模型 runner 混入(Mixin)。
# 在 disaggregated serving 场景下,通过 KVConnector 与 prefill/decode 实例
# 交换 KV cache(no_forward / pre_forward / post_forward),并支持统一 KV 布局。

# 导入 Generator 类型,用于标注生成器返回类型(上下文管理器)。
from collections.abc import Generator
# 导入上下文管理器工具:AbstractContextManager、contextmanager、nullcontext。
from contextlib import AbstractContextManager, contextmanager, nullcontext
# 导入 TYPE_CHECKING,用于仅类型检查时导入。
from typing import TYPE_CHECKING

# 导入 PyTorch,用于张量操作。
import torch

# 导入 VllmConfig 配置类。
from vllm.config import VllmConfig
# 导入缓存 dtype 枚举 CacheDType。
from vllm.config.cache import CacheDType
# 导入 KV 传输组的全局访问器:get_kv_transfer_group 与 has_kv_transfer_group。
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
# 导入 KV 连接器基类 KVConnectorBase,用于类型断言。
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
# 导入前向上下文访问器:get_forward_context 与 set_forward_context。
from vllm.forward_context import get_forward_context, set_forward_context
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入注意力后端基类。
from vllm.v1.attention.backend import AttentionBackend
# 导入 KV cache 规格类型:AttentionSpec 与 KVCacheConfig。
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
# 导入模型 runner 输出类型:KVConnectorOutput 与 ModelRunnerOutput。
from vllm.v1.outputs import (
    KVConnectorOutput,
    ModelRunnerOutput,
)
# 导入注意力分组类。
from vllm.v1.worker.utils import AttentionGroup

# 仅在类型检查时导入 SchedulerOutput。
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# 创建本模块的日志记录器。
logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
# 定义为 Model Runner(GPU/TPU)共用的 KV 连接器功能混入。
class KVConnectorModelRunnerMixin:
    # KV 连接器混入类:提供 KV cache 跨实例传输的保存/加载生命周期管理,
    # 以及统一跨层 KV 布局的选择与分配能力。

    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        # 即使没有前向工作量,也要进行 KV 发送/接收。
        # 在空的前向上下文与 KV 连接器输出上下文中执行(不等待保存完成)。
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ) as kv_connector_output,
        ):
            # 无需任何实际前向工作。
            pass

        # 返回仅含 KV 连接器输出的模型输出。
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        defer_finalize: bool = False,
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        # 根据是否配置 KV 传输组,返回“获取连接器输出”的上下文管理器或空上下文。
        # 参数:
        #   scheduler_output: 调度输出(含连接器元数据)。
        #   defer_finalize: 是否延迟收尾(用于草稿模型前向)。
        return (
            # 配置了 KV 传输:返回真正执行连接器生命周期的上下文管理器。
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, defer_finalize=defer_finalize
            )
            if has_kv_transfer_group()
            # 未配置:返回无操作上下文。
            else nullcontext()
        )

    @staticmethod
    def finalize_kv_connector() -> None:
        """Finalize the KV connector: wait_for_save and clear metadata.

        Call after draft model forward when defer_finalize=True was used.
        """
        # 收尾 KV 连接器:等待保存完成并清除元数据。
        # 在草稿模型前向之后、且使用了 defer_finalize=True 时调用。
        # 若配置了 KV 传输组:
        if has_kv_transfer_group():
            # 获取 KV 传输组连接器。
            kv_connector = get_kv_transfer_group()
            # 等待所有 KV 保存完成。
            kv_connector.wait_for_save()
            # 清除连接器元数据。
            kv_connector.clear_connector_metadata()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    # 说明:该上下文管理器必须在活跃的前向上下文内使用,
    # 它封装了 execute_model 中 KV 连接器的完整生命周期。
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        # 创建 KV 连接器输出容器(记录发送/接收完成、失效块、统计与事件)。
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        # 使用调度输出中的 KV 连接器元数据更新连接器。
        # 获取 KV 传输组连接器。
        kv_connector = get_kv_transfer_group()
        # 断言连接器实现了 KVConnectorBase 接口。
        assert isinstance(kv_connector, KVConnectorBase)
        # 断言调度输出携带了连接器元数据。
        assert scheduler_output.kv_connector_metadata is not None
        # 绑定本次迭代的连接器元数据。
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        # 说明:后台 KV cache 传输在此发生。这些传输设计为异步,涉及的请求
        # 可能与当前运行请求不同;在此处触发以节省一次 collective_rpc。
        # 基于当前前向上下文启动 KV 加载(跨实例拉取 KV)。
        kv_connector.start_load_kv(get_forward_context())
        try:
            # 让出执行权,使 with 体内的前向在 KV 加载已启动的情况下运行。
            yield output
        finally:
            # 若需等待保存且未延迟收尾:
            if wait_for_save and not defer_finalize:
                # 等待所有 KV 保存完成。
                kv_connector.wait_for_save()

            # 查询已完成请求的发送/接收完成标志。
            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            # 获取有加载错误的块 id。
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            # 获取 KV 连接器的统计信息。
            output.kv_connector_stats = kv_connector.get_kv_connector_stats()
            # 获取 KV cache 事件。
            output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
            # 构建连接器 worker 元数据。
            output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

            # 若未延迟收尾:
            if not defer_finalize:
                # 清除连接器元数据。
                kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
    ) -> bool:
        # 判断是否使用统一 KV 布局。
        # 统一布局指所有层共享同一底层张量,对给定块号,所有层的 KV 数据连续。
        # 这允许一次高效地传输所有层的按块 KV 数据。
        # 注意:该布局仅在三条件下应用:
        # 1. KV cache 配置只含一个组,所有层页大小相同;
        # 2. 配置了 KV 连接器,且连接器偏好该布局(prefer_cross_layer_blocks 为 True);
        # 3. 注意力后端按块跨步索引 KV(indexes_kv_by_block_stride),
        #    即 num_blocks 是最外层物理维,使按块的所有层数据连续。
        # 参数:
        #   attn_groups: 模型的注意力组列表。
        # 返回:
        #   是否应使用统一 KV cache 布局。
        # 未配置 KV 传输组,返回 False。
        if not has_kv_transfer_group():
            return False
        # 连接器不偏好跨层块,返回 False。
        if not get_kv_transfer_group().prefer_cross_layer_blocks:
            return False

        # 必须有且仅有一个注意力组且组内仅一个分组。
        if len(attn_groups) != 1 or len(attn_groups[0]) != 1:
            return False

        # 取唯一的注意力组。
        attn_group = attn_groups[0][0]
        # 取该组的 KV cache 规格。
        kv_cache_spec = attn_group.kv_cache_spec
        # 若非注意力规格,返回 False。
        if not isinstance(kv_cache_spec, AttentionSpec):
            return False
        # Per-token-head quant carves inline-scale views that assume per-layer
        # contiguous KV buffers; the cross-layer layout breaks this and corrupts KV.
        # 逐 token-head 量化会切出内联缩放视图,假设每层 KV 缓冲连续;
        # 跨层布局会破坏该假设并损坏 KV。
        if kv_cache_spec.kv_quant_mode.is_per_token_head:
            return False
        # 返回后端是否按块跨步索引 KV。
        return kv_cache_spec.indexes_kv_by_block_stride

    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, type[AttentionBackend]]:
        # 为所有层布局相同的简单场景初始化并重塑 KV caches。
        # 此函数假定 use_uniform_kv_cache() 已返回 True。
        # 参数:
        #   kv_cache_config: KV cache 配置。
        #   attn_groups: 注意力组列表。
        #   cache_dtype: KV cache 数据类型。
        #   device: 分配设备。
        #   kernel_block_sizes: 各 KV cache 组的 kernel 块大小。
        # 返回:
        #   元组 (kv_caches, cross_layers_kv_cache, attn_backend):
        #     kv_caches: 层名 -> 对应 KV 内存缓冲的字典。
        #     cross_layers_kv_cache: 跨层 KV cache 张量。
        #     attn_backend: 与该张量匹配的注意力后端类。
        # 取唯一的注意力组。
        attn_group = attn_groups[0][0]
        # 取 KV cache 规格。
        kv_cache_spec = attn_group.kv_cache_spec
        # 断言是注意力规格。
        assert isinstance(kv_cache_spec, AttentionSpec)

        # 收集所有 KV cache 张量的大小(应为同一个)。
        tensor_sizes = set(
            kv_cache_tensor.size for kv_cache_tensor in kv_cache_config.kv_cache_tensors
        )
        # 断言只有一个尺寸。
        assert len(tensor_sizes) == 1
        # 取出张量大小。
        tensor_size = tensor_sizes.pop()

        # 页大小(字节)。
        page_size = kv_cache_spec.page_size_bytes
        # 断言张量大小为页大小的整数倍。
        assert tensor_size % page_size == 0
        # 计算块数。
        num_blocks = tensor_size // page_size
        # 层数 = KV cache 张量条目数。
        num_layers = len(kv_cache_config.kv_cache_tensors)
        # 总大小 = 单张量大小 × 层数。
        total_size = tensor_size * num_layers

        # 断言 kernel 块大小只有一个。
        assert len(kernel_block_sizes) == 1
        # 取 kernel 块大小。
        kernel_block_size = kernel_block_sizes[0]
        # 计算每 KV 块的 kernel 块数(虚拟块拆分)。
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        # 计算 kernel 级块总数。
        kernel_num_blocks = num_blocks * num_blocks_per_kv_block

        # 取注意力后端类。
        attn_backend = attn_group.backend
        # 获取后端的 KV cache 形状(块优先)。
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        # prepend a num_layers dimension into the shape
        # 在形状前附加一个层数维度。
        kv_cache_shape = (num_layers,) + kv_cache_shape

        # 尝试获取后端的 KV cache 跨步顺序(含层数维度)。
        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
            # 断言跨步顺序长度与形状维度数一致。
            assert len(kv_cache_stride_order) == len(kv_cache_shape)
        # 后端未实现时:
        except (AttributeError, NotImplementedError):
            # 使用自然顺序(0,1,2,...)。
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        # 按跨步顺序重排形状(使层维度作为最内/最外根据后端决定)。
        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

        # 记录分配跨层 KV cache 的日志。
        logger.info("Allocating a cross layer KV cache of shape %s", kv_cache_shape)

        # allocate one contiguous buffer for all layers
        # 为所有层分配一个连续缓冲。
        cross_layers_kv_cache = (
            # 先分配 total_size 字节的 int8 张量,再视图为规格 dtype,
            torch.zeros(total_size, dtype=torch.int8, device=device)
            .view(kv_cache_spec.dtype)
            # 最后视图为目标形状。
            .view(kv_cache_shape)
        )

        # Maintain original KV shape view.
        # 保持原始 KV 形状视图。
        # 计算逆置换顺序(把跨步顺序还原)。
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        # 置换回原始顺序(层维度在前)。
        permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)

        # 初始化层名 -> KV 张量的字典。
        kv_caches = {}
        # 遍历每个 KV cache 张量条目:
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            # 取第 i 层的统一张量视图。
            tensor = permuted_kv_cache[i]
            # 把共享该条目的所有层名映射到该张量。
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = tensor

        # 返回 (层 KV 字典, 跨层张量, 注意力后端类)。
        return kv_caches, cross_layers_kv_cache, attn_backend