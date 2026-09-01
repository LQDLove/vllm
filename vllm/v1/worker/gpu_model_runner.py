# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 导入 functools,用于 cached_property、partial 等工具。
import functools
# 导入 gc,用于垃圾回收。
import gc
# 导入 itertools,用于迭代工具。
import itertools
# 导入 threading,用于线程相关操作。
import threading
# 导入 time,用于计时。
import time
# 导入 defaultdict,用于带默认值的字典。
from collections import defaultdict
# 导入抽象集合类型标注。
from collections.abc import Callable, Iterable, Iterator, Sequence
# 导入上下文管理器工具。
from contextlib import AbstractContextManager, contextmanager, nullcontext
# 导入拷贝与深拷贝。
from copy import copy, deepcopy
# 导入数据类工具。
from dataclasses import dataclass, replace
# 导入 reduce,用于归并迭代。
from functools import reduce
# 导入类型标注工具。
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

# 导入 numpy,用于 CPU 侧数组运算。
import numpy as np
# 导入 PyTorch,用于张量与 CUDA 操作。
import torch
# 导入 torch.distributed,用于分布式原语。
import torch.distributed
# 导入 nn 模块,用于模型类型。
import torch.nn as nn
# 导入 tqdm,用于进度条显示。
from tqdm import tqdm

# 导入环境变量配置。
import vllm.envs as envs
# 导入可打断 CUDA graph 包装与启用判断。
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
# 导入编译计数器。
from vllm.compilation.counter import compilation_counter
# 导入 CUDA graph 统计与包装器。
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
# 导入 CUDA graph 捕获开关设置。
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
# 导入编译配置相关项。
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
    update_config,
)
# 导入缓存配置。
from vllm.config.cache import CacheConfig
# 导入编码器缓存管理器元数据。
from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
# 导入已处理的 logprobs 模式常量。
from vllm.config.model import PROCESSED_LOGPROBS_MODES
# 导入 EC(嵌入缓存)传输工具。
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
# 导入 EPLB 状态。
from vllm.distributed.eplb.eplb_state import EplbState
# 导入 KV 传输工具。
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
# 导入 KV 块拷贝工具。
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
# 导入并行状态工具(图捕获上下文、各并行组访问器等)。
from vllm.distributed.parallel_state import (
    GraphCaptureContext,
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
)
# 导入前向上下文工具。
from vllm.forward_context import (
    BatchDescriptor,
    set_forward_context,
)
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 LoRA 层基类与映射类型。
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping, LoRAMappingType
# 导入注意力层与 MLA 注意力。
from vllm.model_executor.layers.attention import Attention, MLAAttention
# 导入注意力层基类接口。
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
# 导入 EP all2all 管理器访问器。
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
# 导入路由专家捕获器及绑定函数。
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
# 导入 Mamba SSU 后端初始化函数。
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
# 导入旋转位置编码类(MRoPE 与 XDRoPE)。
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
# 导入模型加载器工厂。
from vllm.model_executor.model_loader import get_model_loader
# 导入分层重载的初始化与收尾函数。
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
# 导入模型接口(MoE、多模态、mRoPE/xdrope 等)。
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_realtime,
    supports_transcription,
    supports_xdrope,
)
# 导入基础模型接口(池化、生成判断)。
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
# 导入层卸载器工具。
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)
# 导入多模态注册表。
from vllm.multimodal import MULTIMODAL_REGISTRY
# 导入多模态编码预算。
from vllm.multimodal.encoder_budget import MultiModalBudget
# 导入多模态输入结构。
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
# 导入多模态工具(模态拷贝、窗口特征、批量 kwargs)。
from vllm.multimodal.utils import (
    copy_mm_embedding_modality,
    get_mm_features_in_window,
    group_and_batch_mm_kwargs,
    set_mm_embedding_modality,
)
# 导入当前平台抽象。
from vllm.platforms import current_platform
# 导入池化参数。
from vllm.pooling_params import PoolingParams
# 导入采样类型枚举。
from vllm.sampling_params import SamplingType
# 导入中间张量容器。
from vllm.sequence import IntermediateTensors
# 导入任务类型枚举。
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
# 导入追踪装饰器。
from vllm.tracing import instrument
# 导入 prompt 长度计算工具。
from vllm.utils import length_from_prompt_token_ids_or_embeds
# 导入数学工具(向上整除与向上取整)。
from vllm.utils.math_utils import cdiv, round_up
# 导入显存工具(分析器与格式化)。
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
# 导入 PyTorch NVTX 钩子。
from vllm.utils.nvtx_pytorch_hooks import PytHooks
# 导入计算单元数量工具。
from vllm.utils.platform_utils import num_compute_units
# 导入 torch 工具(锁页、异步 H2D、流、量化 KV dtype 等)。
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    async_tensor_h2d,
    current_stream,
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
# 导入注意力后端基础接口。
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
# 导入 GDN 注意力元数据构建器。
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
# 导入 Bailing 线性注意力元数据构建器。
from vllm.v1.attention.backends.linear_attn import (
    BailingLinearAttentionMetadataBuilder,
)
# 导入 Mamba2 注意力元数据构建器。
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
# 导入注意力后端通用工具。
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
)
# 导入调度器的新请求数据结构。
from vllm.v1.core.sched.output import NewRequestData
# 导入 CUDA graph 分发器。
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
# 导入 KV cache 规格相关类型。
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVQuantMode,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)
# 导入 KV cache 规格注册表。
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
# 导入输出相关结构。
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    RoutedExpertsLists,
    RoutedExpertsTensors,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
# 导入后期交互运行器(池化)。
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner
# 导入池化元数据与状态。
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
# 导入 logits 处理器工具。
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
# 导入 logits 处理器接口。
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
# 导入采样元数据。
from vllm.v1.sample.metadata import SamplingMetadata
# 导入拒绝采样器。
from vllm.v1.sample.rejection_sampler import RejectionSampler
# 导入采样器。
from vllm.v1.sample.sampler import Sampler
# 导入投机解码各类提案器。
from vllm.v1.spec_decode.custom_class_proposer import create_custom_proposer
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.spec_decode.medusa import MedusaProposer
# 导入投机解码元数据。
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
# 导入 GPU n-gram 提案器相关工具。
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    NgramProposerGPU,
    copy_num_valid_draft_tokens,
    update_ngram_gpu_tensors_incremental,
    update_scheduler_for_invalid_drafts,
)
# 导入 Step3p5 MTP 提案器。
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
# 导入后缀解码提案器。
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
# 导入批次变化时的 token 数更新工具。
from vllm.v1.spec_decode.utils import update_num_computed_tokens_for_batch_change
# 导入语法位掩码应用函数。
from vllm.v1.structured_output.utils import apply_grammar_bitmask
# 导入 CPU/GPU 双缓冲与函数记录上下文。
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
# 导入 mamba 工具模块。
from vllm.v1.worker import mamba_utils
# 导入 slot 映射模式枚举。
from vllm.v1.worker.block_table import SlotMappingMode
# 导入 CP 工具(兼容性检查、dummy 上下文长度与元数据)。
from vllm.v1.worker.cp_utils import (
    check_attention_cp_compatibility,
    get_dcp_dummy_context_len,
    prepare_dcp_dummy_context_metadata,
)
# 导入 DP 批次协调工具。
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
# 导入 EC 连接器运行器混入类。
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
# 导入注意力 KV cache 重塑函数。
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
# 导入 v1 输入批次与缓存请求状态。
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
# 导入微批次(ubatch)包装器。
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
# 导入 KV 连接器运行器混入类。
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
# 导入 LoRA 运行器混入类。
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
# 导入微批次工具(切片、阈值检查、注意力元数据拆分)。
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
    split_attn_metadata,
)
# 导入 worker 工具(NaN 检查、SP 残差判断)。
from vllm.v1.worker.utils import is_residual_scattered_for_sp, raise_if_nan_logits
# 导入工作区锁定上下文。
from vllm.v1.worker.workspace import lock_workspace

# 导入同目录 worker 工具(注意力组、块清零、KV 共享等)。
from .utils import (
    AttentionGroup,
    KVBlockZeroer,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    copy_kv_cache_blocks_inplace,
    prepare_kernel_block_sizes,
    sanity_check_mm_encoder_outputs,
)

# 仅类型检查时导入(避免循环依赖)。
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

# 创建本模块的日志记录器。
logger = init_logger(__name__)


def _get_parameter_for_reload(model: nn.Module, name: str) -> nn.Parameter:
    """按检查点名称解析参数,不改变模型的模块树。"""
    # 从右侧分割出模块名与参数名。
    module_name, _, parameter_name = name.rpartition(".")
    # 获取子模块。
    module = model.get_submodule(module_name)
    if isinstance(module, BaseLayerWithLoRA):
        # LoRA 层则取其基础层。
        module = module.base_layer
    # 返回目标参数。
    return module.get_parameter(parameter_name)


# 注意力元数据字典类型别名(层名 -> 元数据)。
AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# 启用 ubatch 时为列表。
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


# ModelRunnerOutput 的包装器,支持重叠执行。
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        # 同步部分的模型运行器输出。
        model_runner_output: ModelRunnerOutput,
        # 采样的 token ids 设备张量。
        sampled_token_ids: torch.Tensor,
        # logprobs 张量(可选)。
        logprobs_tensors: LogprobsTensors | None,
        # 无效请求索引列表。
        invalid_req_indices: list[int],
        # 异步输出拷贝专用流。
        async_output_copy_stream: torch.cuda.Stream,
        # 词表大小。
        vocab_size: int,
        # 路由专家张量(可选)。
        routed_experts: RoutedExpertsTensors | None = None,
        # 是否检查 EP 容错。
        check_ep_fault: bool = False,
    ):
        # 保存同步部分输出。
        self._model_runner_output = model_runner_output
        # 保存无效请求索引。
        self._invalid_req_indices = invalid_req_indices

        # 拷贝流上的事件,用于同步非阻塞拷贝。
        # 阻塞(sleep)事件以避免忙轮询 CUDA 驱动锁。
        self.async_copy_ready_event = torch.cuda.Event(blocking=True)

        # 保留设备张量引用,避免在拷贝到主机完成前被释放。
        self._sampled_token_ids = sampled_token_ids
        # 保存词表大小。
        self.vocab_size = vocab_size
        # 保存 logprobs 张量。
        self._logprobs_tensors = logprobs_tensors
        # 保存路由专家张量。
        self._routed_experts = routed_experts
        # EP 故障标志张量默认 None。
        self._has_fault: torch.Tensor | None = None

        # 在独立流上启动拷贝,但不立即同步。
        # 取默认流。
        default_stream = torch.cuda.current_stream()
        # 切换到异步拷贝流。
        with torch.cuda.stream(async_output_copy_stream):
            # 拷贝流等待默认流完成。
            async_output_copy_stream.wait_stream(default_stream)
            # 非阻塞拷贝采样 token 到 CPU。
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            # 非阻塞拷贝 logprobs 到 CPU(若有)。
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            # 非阻塞拷贝路由专家到 CPU(若有)。
            self._routed_experts_cpu = (
                self._routed_experts.to_cpu_nonblocking()
                if self._routed_experts is not None
                else None
            )
            if check_ep_fault:
                # 查询 EP all2all 故障标志并拷到 CPU。
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            # 记录拷贝完成事件。
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """把设备张量拷到主机并返回 ModelRunnerOutput。

        本函数会阻塞直到拷贝完成。
        """
        # 取采样 token 的最大生成长度。
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        # 等待异步拷贝完成。
        self.async_copy_ready_event.synchronize()

        # 拷贝完成后释放设备张量。
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            # 常规单 token 生成:直接转列表。
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                # 清空无效请求的采样结果。
                valid_sampled_token_ids[i].clear()
            # logprobs 列表默认 None。
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                # 转换 logprobs 张量。
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            # 投机解码多 token:用拒绝采样器解析输出。
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        # 取保存的输出对象并填充结果。
        output = self._model_runner_output
        # 填入有效采样 token ids。
        output.sampled_token_ids = valid_sampled_token_ids
        # 填入 logprobs。
        output.logprobs = logprobs_lists

        if self._routed_experts_cpu is not None:
            # 填入路由专家列表。
            output.routed_experts = self._routed_experts_cpu.tolists()
        # 释放设备端路由专家张量。
        del self._routed_experts

        if self._has_fault is not None and self._has_fault.item():
            # 检测到 EP 通信故障则抛出异常。
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        # 返回最终输出。
        return output


def _copy_pooler_output_to_cpu(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    # 把池化输出按完成掩码拷贝到 CPU,仅保留已完成请求的结果。
    # 取请求数。
    num_reqs = len(finished_mask)

    if isinstance(raw_pooler_output, torch.Tensor):
        # 张量形式输出。
        if raw_pooler_output.shape[0] != num_reqs:
            # 批大小与掩码不一致时报错。
            raise ValueError(
                "Pooler output batch size does not match finished mask size: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}."
            )

        # 统计已完成请求数。
        num_finished = sum(finished_mask)
        if num_finished == 0:
            # 无完成请求,全部为 None。
            return [None] * num_reqs
        if num_finished == num_reqs:
            # 全部完成:整批拷贝。
            return list(raw_pooler_output.to("cpu", non_blocking=True))

        # 部分完成。
        # 取完成请求的索引列表。
        finished_indices = [i for i, include in enumerate(finished_mask) if include]
        # 构建索引张量。
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        # 按索引选择并拷贝到 CPU。
        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
            "cpu", non_blocking=True
        )
        # 初始化部分输出列表。
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            # 填入完成请求的结果。
            partial_pooler_output[i] = out
        # 返回部分结果。
        return partial_pooler_output

    # 列表形式输出。
    assert isinstance(raw_pooler_output, list)
    if len(raw_pooler_output) != num_reqs:
        # 长度不一致时报错。
        raise ValueError(
            "Pooler output batch size does not match finished mask size: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )

    # 初始化结果列表。
    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
        if include and out is not None:
            # 仅拷贝已完成且非空的条目。
            pooler_output[i] = out.to("cpu", non_blocking=True)
    # 返回结果列表。
    return pooler_output


class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    # 池化模型的异步输出包装器。

    def __init__(
        self,
        # 同步部分输出。
        model_runner_output: ModelRunnerOutput,
        # 原始池化输出(设备上)。
        raw_pooler_output: PoolerOutput,
        # 完成掩码。
        finished_mask: list[bool],
        # 异步拷贝流。
        async_output_copy_stream: torch.cuda.Stream,
    ):
        # 保存同步部分输出。
        self._model_runner_output = model_runner_output

        # 拷贝流上的事件,用于同步非阻塞拷贝。
        # 阻塞(sleep)事件以避免忙轮询 CUDA 驱动锁。
        self.async_copy_ready_event = torch.cuda.Event(blocking=True)

        # 保留设备张量引用,避免在拷贝到主机完成前被释放。
        self._raw_pooler_output = raw_pooler_output

        # 在独立流上启动拷贝,但不立即同步。
        # 取默认流。
        default_stream = torch.cuda.current_stream()
        # 切换到异步拷贝流。
        with torch.cuda.stream(async_output_copy_stream):
            # 拷贝流等待默认流。
            async_output_copy_stream.wait_stream(default_stream)
            # 按完成掩码拷贝池化输出到 CPU。
            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=self._raw_pooler_output,
                finished_mask=finished_mask,
            )
            # 记录拷贝完成事件。
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """把设备张量拷到主机并返回 ModelRunnerOutput。
        本函数会阻塞直到拷贝完成。
        """
        # 等待异步拷贝完成。
        self.async_copy_ready_event.synchronize()

        # 拷贝完成后释放设备张量。
        del self._raw_pooler_output
        # 返回最终输出。
        return self._model_runner_output


class ExecuteModelState(NamedTuple):
    """execute_model() 返回 None 之后,在其与 sample_tokens() 之间
    传递的临时缓存状态。"""

    # 调度器输出。
    scheduler_output: "SchedulerOutput"
    # logits 张量。
    logits: torch.Tensor
    # 投机解码元数据。
    spec_decode_metadata: SpecDecodeMetadata | None
    # 投机解码公共注意力元数据。
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    # 全部隐藏状态。
    hidden_states: torch.Tensor
    # 采样用隐藏状态。
    sample_hidden_states: torch.Tensor
    # 辅助隐藏状态列表。
    aux_hidden_states: list[torch.Tensor] | None
    # EC 连接器输出。
    ec_connector_output: ECConnectorOutput | None
    # CUDA graph 统计。
    cudagraph_stats: CUDAGraphStat | None
    # slot 映射(启用 ubatch 时为列表)。
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    # GPU v1 模型运行器:混入 LoRA、KV 连接器与 EC 连接器能力。

    def __init__(
        self,
        # vLLM 总配置。
        vllm_config: VllmConfig,
        # 目标设备。
        device: torch.device,
    ):
        # 保存总配置。
        self.vllm_config = vllm_config
        # 保存模型配置。
        self.model_config = vllm_config.model_config
        # 保存缓存配置。
        self.cache_config = vllm_config.cache_config
        # 保存卸载配置。
        self.offload_config = vllm_config.offload_config
        # 保存编译配置。
        self.compilation_config = vllm_config.compilation_config
        # 保存 LoRA 配置。
        self.lora_config = vllm_config.lora_config
        # 保存加载配置。
        self.load_config = vllm_config.load_config
        # 保存并行配置。
        self.parallel_config = vllm_config.parallel_config
        # 保存调度器配置。
        self.scheduler_config = vllm_config.scheduler_config
        # 保存投机解码配置。
        self.speculative_config = vllm_config.speculative_config
        # 保存可观测性配置。
        self.observability_config = vllm_config.observability_config

        # 局部引用以便简写。
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        # 保存设备。
        self.device = device
        # 模型数据类型。
        self.dtype = self.model_config.dtype

        # EP 容错检查默认关闭。
        self.check_ep_fault = False
        if parallel_config.data_parallel_size > 1 and self.model_config.is_moe:
            # MoE + DP 时查询后端容错能力。
            self.check_ep_fault = get_ep_all2all_manager().support_fault_tolerance

        # 解析 KV cache 数据类型。
        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        # 是否为池化模型。
        self.is_pooling_model = model_config.runner_type == "pooling"
        # 是否启用 prompt embeds 输入。
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        # 是否为仅接受多模态原始输入的模型。
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # 以下两项将在 load_model() 中覆盖。
        self.is_multimodal_pruning_enabled = False
        self.requires_sequential_video_encoding = False
        # 在 init_routed_experts_capturer() 完成后置 True。
        # 防止路由专家代码在 profiling/dummy 运行时执行。
        self.routed_experts_initialized = False
        # 最大模型长度。
        self.max_model_len = model_config.max_model_len

        # 首次前向后总是置 False。
        # DCP 世界大小。
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        # DCP rank(未启用时为 0)。
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        # 单批最大 token 数。
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        # 最大并发请求数。
        self.max_num_reqs = scheduler_config.max_num_seqs

        # 为 external_launcher(torchrun)广播 PP 输出,
        # 以确保各 PP rank 同步。
        # TODO: 支持微批次重叠
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # 模型相关。
        # query 注意力头数。
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        # 输入嵌入维度。
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        # 仅对使用 ALiBi 的模型(如 MPT)有意义。
        self.use_alibi = model_config.uses_alibi

        # 是否启用级联注意力。
        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        # 是否为 mm prefix LM。
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # 多模态数据支持。
        # 多模态注册表。
        self.mm_registry = MULTIMODAL_REGISTRY
        # 是否使用 mrope。
        self.uses_mrope = model_config.uses_mrope
        # xdrope 维度(0 表示不使用)。
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        # 是否支持多模态输入。
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # 编码器输入最大长度,仅 encoder-decoder 模型使用。
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # 异步调度开关。
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # 采样器。
        self.sampler = Sampler(
            logprobs_mode=self.model_config.logprobs_mode,
            use_fp64_gumbel=self.model_config.use_fp64_gumbel,
        )

        # EPLB 状态(模型加载后惰性初始化)。
        self.eplb_state: EplbState | None = None
        # MoE 模型接口缓存。
        self._moe_model: MixtureOfExperts | None = None
        # 注意(yongji): 扩容/缩容期间临时禁用 EPLB 的标志。
        self.eep_eplb_suppressed = False
        """
        专家并行负载均衡器的状态。

        模型加载后惰性初始化。
        """

        # 惰性初始化项。
        # self.model: nn.Module  # load_model 后设置
        # 在 initialize_kv_cache 中初始化。
        self.kv_caches: list[torch.Tensor] = []
        # 在 initialize_kv_cache_tensors 中初始化。
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # 索引: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash -> 编码器输出。
        self.encoder_cache: dict[str, torch.Tensor] = {}
        # 后期交互运行器。
        self.late_interaction_runner = LateInteractionRunner()

        # 编码器 CUDA graph 管理器(模型加载后如启用则初始化)。
        self.encoder_cudagraph_manager: EncoderCudaGraphManager | None = None

        # 是否使用辅助隐藏状态输出。
        self.use_aux_hidden_state_outputs = False
        # 设置投机解码。
        # 注意(Jiayi): 目前把整个草稿模型放在最后一个 PP rank 上。
        # 若草稿模型层数很多,这并不理想。
        if self.speculative_config and get_pp_group().is_last_rank:
            # 草稿提案器类型联合。
            self.drafter: (
                NgramProposer  # noqa: F823
                | NgramProposerGPU
                | SuffixDecodingProposer
                | EagleProposer
                | DFlashProposer
                | DraftModelProposer
                | MedusaProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer
                | Step3p5MTPProposer
            )
            if self.speculative_config.method == "custom_class":
                # 自定义提案器类。
                self.drafter = create_custom_proposer(  # type: ignore[assignment]
                    self.vllm_config
                )
            elif self.speculative_config.method == "ngram":
                # CPU n-gram 提案器(局部导入避免循环依赖)。
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer

                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.uses_draft_model():
                # 草稿模型提案器(EAGLE 等)。
                self.drafter = DraftModelProposer(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    runner=self,
                )
            elif self.speculative_config.use_ngram_gpu():
                # GPU n-gram 提案器。
                self.drafter = NgramProposerGPU(self.vllm_config, self.device, self)
                # GPU 上记录不含投机 token 的数量。
                self.num_tokens_no_spec_gpu = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32, device=device
                )
                # GPU 上的 token ids 缓冲。
                self.token_ids_gpu_tensor = torch.zeros(
                    self.max_num_reqs,
                    self.max_model_len,
                    dtype=torch.int32,
                    device=device,
                )
                # 锁页索引缓冲(n-gram 用)。
                self._ngram_pinned_idx_buf = torch.zeros(
                    self.max_num_reqs, dtype=torch.long, pin_memory=True
                )
                # 锁页值缓冲(n-gram 用)。
                self._ngram_pinned_val_buf = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32, pin_memory=True
                )
            elif self.speculative_config.use_gemma4_mtp():
                # Gemma4 MTP 提案器。
                self.drafter = Gemma4Proposer(self.vllm_config, self.device, self)
            elif self.speculative_config.use_step3p5_mtp():
                # Step3p5 MTP 提案器。
                self.drafter = Step3p5MTPProposer(self.vllm_config, self.device, self)
            elif self.speculative_config.use_dflash():
                # DFlash 提案器(需要辅助隐藏状态)。
                self.drafter = DFlashProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "suffix":
                # 后缀解码提案器。
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                # EAGLE 提案器。
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    # eagle3 视配置决定是否使用辅助隐藏状态。
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
                # Medusa 提案器。
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            elif self.speculative_config.method == "extract_hidden_states":
                # 隐藏状态提取提案器(需要辅助隐藏状态)。
                self.drafter = ExtractHiddenStatesProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
                self.use_aux_hidden_state_outputs = True
            else:
                # 未知的投机解码方法。
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            # 创建拒绝采样器。
            self.rejection_sampler = RejectionSampler(
                self.sampler, self.speculative_config, self.device
            )

        # 投机 token 数默认 0。
        self.num_spec_tokens = 0
        # 上一步的投机 token 数。
        self.prev_num_spec_tokens = 0
        # GPU 上的有效采样 token 计数。
        self.valid_sampled_token_count_gpu: torch.Tensor | None = None
        if self.speculative_config:
            # 记录投机 token 数。
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            self.prev_num_spec_tokens = self.num_spec_tokens
            # 取草稿模型配置。
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                # 草稿模型的最大长度。
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                # 无草稿配置时与主模型相同。
                self.effective_drafter_max_model_len = self.max_model_len
        # 异步调度 + 投机解码同时启用时为 True。
        self.use_async_spec_decode = (
            self.use_async_scheduling and self.num_spec_tokens > 0
        )

        # 请求状态。
        # req_id -> 缓存请求状态映射。
        self.requests: dict[str, CachedRequestState] = {}
        # 注意(rob): num_prompt_logprobs 只包含当前处于 prefill 阶段的请求。
        self.num_prompt_logprobs: dict[str, int] = {}

        # 输入批次。
        # 注意(Chen): 理想情况下应根据 kv cache 配置在 initialize_kv_cache
        # 中初始化输入批次。但如同 https://github.com/vllm-project/vllm/pull/18298,
        # 由于某些未知原因,必须在 load_model 之前初始化输入批次,
        # 否则量化 + 权重卸载会失败。作为临时方案,先在此初始化,
        # 并在 initialize_kv_cache 中当此处的 block_sizes 与 kv cache
        # 配置不同时重新初始化。
        # 取模型配置中的 logits 处理器。
        logits_processors = model_config.logits_processors
        # 自定义 logits 处理器序列。
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        # 占位块大小(初始化时 KV 配置未知)。
        placeholder_block_size = (
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
        )
        # 占位最大块数。
        placeholder_max_num_blocks = cdiv(
            max(self.max_model_len, self.max_encoder_len), placeholder_block_size
        )
        # 记录初始化块大小(供 initialize_kv_cache 重建时比较)。
        self._init_block_sizes = [placeholder_block_size]
        self._init_kernel_block_sizes = [placeholder_block_size]
        self._init_max_num_blocks = [placeholder_max_num_blocks]
        self._init_slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT]
        # 创建输入批次。
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # encoder-decoder 需使用编码器长度,
            # 因为交叉注意力也有 KV cache。
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[placeholder_block_size],
            kernel_block_sizes=[placeholder_block_size],
            max_num_blocks_per_req=[placeholder_max_num_blocks],
            num_spec_tokens=self.num_spec_tokens,
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                PIN_MEMORY,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # 目前无法得知某个自定义 logits 处理器是否使用输出 token ids,
            # 因此保守设置。思考预算跟踪在有预算请求入批时动态请求。
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            reasoning_config=self.vllm_config.reasoning_config,
            use_replayssm=self.cache_config.use_replayssm,
        )

        # 异步调度启用时,用于采样 token 从 GPU 到 CPU 重叠传输的
        # 独立 CUDA 流。
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # 异步调度启用时,用于同步跨步骤复用的 CPU 张量。
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            # 创建异步拷贝流。
            self.async_output_copy_stream = torch.cuda.Stream()
            # 阻塞(sleep)事件以避免忙轮询 CUDA 驱动锁;
            # TP 争用时自旋会膨胀并使该 rank 掉队。
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)

        # self.cudagraph_batch_sizes 按升序排序。
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # 取配置的捕获尺寸并排序。
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )
        else:
            # 禁用时为空列表。
            self.cudagraph_batch_sizes = []

        # 缓存设备属性。
        self._init_device_properties()

        # 可观测性用的编码器计时注册表。
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        # 编码器计时的线程锁。
        self._encoder_timing_lock = threading.Lock()

        # CUDA graph 的持久缓冲区。
        # 输入 token ids 缓冲。
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # 位置缓冲。
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        # query 起点前缀和缓冲。
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        # 序列长度缓冲。
        self.seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # 乐观序列长度的 CPU 锁页缓冲。
        self.optimistic_seq_lens_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        # 已计算 token 数缓冲。
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # 上一步草稿 token 数缓冲。
        self.prev_num_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        # token -> 请求索引映射缓冲。
        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        # 映射当前批次位置 -> 上一批次位置(新请求为 -1)。
        self.prev_positions = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        # 调度 token 数缓冲。
        self.num_scheduled_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # 编码器序列长度缓冲。
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            # DCP 启用时准备本地序列长度缓冲。
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # 由于 inputs_embeds 可能是 bfloat16 且无需 numpy 版本,
        # 不创建 numpy 缓冲以避免 RuntimeError。
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        # token id 标记缓冲。
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        # 丢弃请求掩码缓冲。
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        # 解码草稿 token 数缓冲。
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        # 接受 token 数缓冲。
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # 仅对使用 M-RoPE 的模型(如 Qwen2-VL)有意义。
        if self.uses_mrope:
            # 注意: mrope_positions 特意多加一个 dummy 位置使其非连续,
            # 以便与 torch.compile 配合。
            # 详细解释见 https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # 注意: 启用 M-RoPE 时,无论输入模态如何位置 id 都是 3D。
            # 对纯文本输入,各维位置 id 相同,使 M-RoPE 功能等价于 1D-RoPE。
            # 见 https://arxiv.org/abs/2409.12191 第 5 页
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # 仅对使用 XD-RoPE 的模型(如 HunYuan-VL)有意义。
        if self.uses_xdrope_dim > 0:
            # 类似 mrope,但使用指定维数(默认 4)。
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # 首个 PP rank 为 None;其余在 load_model 后设置。
        self.intermediate_tensors: IntermediateTensors | None = None

        # 优化: 缓存 arange 张量而非每步重建。
        # 保持 int64 以避免长上下文溢出。
        # - arange_np: 不可变的 [0, 1, 2, ...] 作为批量计算源
        # - query_pos: 存放计算出的批量 arange 结果的 CpuGpuBuffer
        # 缓冲大小取两者较大值。
        arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
        # 创建 arange numpy 数组。
        self.arange_np = np.arange(arange_size, dtype=np.int64)
        # 创建 query 位置缓冲。
        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        # arange 计算的临时数组。
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)

        # 跨层 KV 共享的层配对。
        # 若注意力层 `layer_name` 是此字典的键,表示该层将使用
        # `shared_kv_cache_layers[layer_name]` 的 KV cache 做注意力。
        self.shared_kv_cache_layers: dict[str, str] = {}
        # 可用快速 prefill 的 KV 共享层集合。
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        # 快速 prefill 的 logits 索引默认 None。
        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            # 启用时分配索引缓冲。
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        # 均匀解码的 query 长度 = 1 + 投机 token 数。
        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # 用于运行时 CUDA graph 分发的分发器。
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # 多模态编码预算(支持 mm 输入时创建)。
        self.mm_budget = (
            MultiModalBudget(self.vllm_config, self.mm_registry)
            if self.supports_mm_inputs
            else None
        )

        # 批次重排阈值默认 None。
        self.reorder_batch_threshold: int | None = None

        # 仅存在于本运行器 KVCacheConfig 中的注意力层
        # (如 KV 共享、encoder-only 注意力),
        # 而不在调度器的 KVCacheConfig 中。
        self.runner_only_attn_layers: set[str] = set()

        # 缓存的草稿输出。
        # 草稿 token ids。
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        # 草稿概率。
        self._draft_probs: torch.Tensor | None = None
        # 草稿概率对应的请求 id 列表。
        self._draft_prob_req_ids: list[str] | None = None
        # N-gram GPU 路径: 每请求有效草稿数的异步 D2H 缓冲/事件。
        self._num_valid_draft_tokens: torch.Tensor | None = None
        self._num_valid_draft_tokens_cpu: torch.Tensor | None = None
        self._num_valid_draft_tokens_event: torch.cuda.Event | None = None
        self._num_valid_draft_tokens_copy_stream: torch.cuda.Stream | None = None
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            # GPU n-gram 时创建锁页 CPU 缓冲、事件与拷贝流。
            self._num_valid_draft_tokens_cpu = torch.empty(
                self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
            )
            self._num_valid_draft_tokens_event = torch.cuda.Event()
            self._num_valid_draft_tokens_copy_stream = torch.cuda.Stream()

        # 草稿 token 对应的请求 id 列表。
        self._draft_token_req_ids: list[str] | None = None
        # 传输事件。
        self.transfer_event = torch.Event()
        # 采样 token 的锁页 CPU 缓冲。
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        # 预分配的"有效采样 token 计数"拷贝 CPU 张量,
        # 带专用流做重叠、事件做协调。
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # 草稿 token 也会异步拷贝到 CPU,以备结构化输出需要。
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        self.num_accepted_tokens_event: torch.Event | None = None
        if self.num_spec_tokens:
            # 启用投机解码时创建相关事件、流与 CPU 缓冲。
            self.draft_token_ids_event = torch.Event()
            self.num_accepted_tokens_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            if self.use_async_scheduling:
                # 异步调度时为有效采样计数创建事件、流与缓冲。
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )

        # 模型权重卸载器。
        # 确保在任何 get_offloader 调用之前执行。
        set_offloader(create_offloader(self.offload_config))

        # execute_model() 与 sample_tokens() 之间传递的临时状态。
        self.execute_model_state: ExecuteModelState | None = None
        # KV 连接器输出。
        self.kv_connector_output: KVConnectorOutput | None = None
        # Mamba 状态索引映射。
        self.mamba_state_idx: dict[str, int] = {}
        # Mamba 缓冲区(惰性创建)。
        self._mamba_bufs: mamba_utils.MambaBuffers | None = None
        # 上一步最后调度的 Mamba 索引缓冲。
        self.mamba_prev_last_scheduled_idx: CpuGpuBuffer | None = None
        if self.cache_config.mamba_cache_mode == "all" and self.num_spec_tokens > 0:
            # "all" 模式 + 投机解码时创建。
            self.mamba_prev_last_scheduled_idx = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # 分层 NVTX 钩子注册标志。
        self.layerwise_nvtx_hooks_registered = False

    def update_max_model_len(self, max_model_len: int) -> None:
        # 更新最大模型长度(profile 时可能调整)。
        self.max_model_len = max_model_len
        if self.speculative_config:
            # 取草稿模型配置。
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                # 草稿无独立长度限制时与主模型同步。
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        """
        清理 profiling 期间使用、推理阶段不再需要的多模态缓存。
        """
        if self.mm_budget:
            # 重置多模态预算缓存。
            self.mm_budget.reset_cache()
        # 清空后期交互运行器。
        self.late_interaction_runner.clear()

    def reset_encoder_cache(self) -> None:
        """清空存放视觉嵌入的 GPU 侧编码器缓存。

        更新模型权重时应调用,以确保不会复用旧权重算出的过期嵌入。
        """
        # 清空编码器缓存字典。
        self.encoder_cache.clear()
        # 清空后期交互运行器。
        self.late_interaction_runner.clear()

    def post_kv_cache_wake_up(self) -> None:
        # KV cache 从休眠唤醒后的处理。
        self.init_fp8_kv_scales()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """
        从休眠唤醒后重新初始化 KV cache 与 FP8 缩放因子。
        1. 将 KV cache 张量清零,去除重新分配带来的垃圾数据。
        2. 把注意力层缩放因子(_k_scale、_v_scale)重置为 1.0。
          若留在 0.0(唤醒后的默认值),所有 KV cache 值实际变为零,
          导致输出乱码。
        """
        if not is_quantized_kv_cache(self.cache_config.cache_dtype):
            # 非量化 KV cache 无需处理。
            return

        # 取 KV cache 条目列表。
        kv_caches = getattr(self, "kv_caches", [])
        for cache_entry in kv_caches:
            if cache_entry is None:
                # 跳过空条目。
                continue
            # 混合模型(Mamba、DeltaNet)按层以张量列表而非单张量
            # 存放状态。
            if isinstance(cache_entry, list):
                for t in cache_entry:
                    # 逐个清零。
                    t.zero_()
            else:
                # 单张量直接清零。
                cache_entry.zero_()

        # K 缩放因子可能的属性名。
        k_attr_names = ("_k_scale", "k_scale")
        # V 缩放因子可能的属性名。
        v_attr_names = ("_v_scale", "v_scale")

        # 取静态前向上下文中的各层。
        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: 一般而言,用户使用在线 fp8 kv cache 量化时缩放
                # 因子为 1.0。但为获得更好精度,llm-compressors 等压缩
                # 框架允许用户调整缩放因子。未来可能需要在此恢复
                # 特定的校准缩放。
                k_scale_val, v_scale_val = 1.0, 1.0

                # 处理 K 缩放。
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        # 取参数并填充。
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # 处理 V 缩放。
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        # 取参数并填充。
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        # 获取位置张量:整数为切片长度,序列为索引集合。
        if isinstance(num_tokens, int):
            # 按长度切片。
            if self.uses_mrope:
                # mrope:返回 3D 位置切片。
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                # xdrope:返回多维位置切片。
                return self.xdrope_positions.gpu[:, :num_tokens]
            # 常规 1D 位置切片。
            return self.positions[:num_tokens]
        else:
            # 按索引集合切片。
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        # 创建 CPU/GPU 双缓冲。
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            with_numpy=numpy,
        )

    def _get_mamba_bufs(self) -> mamba_utils.MambaBuffers:
        # 仅在 ``mamba_cache_mode == "align"`` 路径可达。
        # 后处理子对象额外要求投机解码 + hybrid 模型。
        # 断言模式正确。
        assert self.cache_config.mamba_cache_mode == "align"
        if self._mamba_bufs is None:
            # 惰性创建 Mamba 缓冲区。
            self._mamba_bufs = mamba_utils.MambaBuffers.create(
                max_num_reqs=self.max_num_reqs,
                kv_cache_config=self.kv_cache_config,
                copy_funcs=self.model.get_mamba_state_copy_func(),
                make_buffer=self._make_buffer,
                device=self.device,
                with_postprocess_align=(
                    self.speculative_config is not None and self.model_config.is_hybrid
                ),
            )
        # 返回缓冲区。
        return self._mamba_bufs

    def _init_model_kwargs(self):
        # 构建模型额外关键字参数(池化模型的 token_type_ids)。
        # 初始化空字典。
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            # 非池化模型直接返回。
            return model_kwargs

        # 取请求数与池化参数。
        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        # 收集带压缩 token_type_ids 的请求。
        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                # 记录该请求的 token 类型。
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            # 无此类请求,返回空 kwargs。
            return model_kwargs

        # 使用 CPU 侧 seq_lens 上界在 CPU 上构建 id;
        # 带GPU 标量的 torch.arange(seq_lens[i]) 会强制同步。
        # 取各请求序列长度列表。
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs].tolist()
        # token 类型 id 列表。
        token_type_ids = []

        for i in range(num_reqs):
            # 该请求的序列长度。
            seq_len_i = seq_lens_cpu[i]
            # 压缩 token 类型起点(默认在末尾)。
            pos = token_type_id_requests.get(i, seq_len_i)
            # 位置 >= pos 的标记为 1。
            ids = (torch.arange(seq_len_i) >= pos).int()
            token_type_ids.append(ids)

        # 拼接进锁页 CPU 缓冲。
        token_type_ids_cpu = torch.empty(
            sum(seq_lens_cpu), dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        # 拼接所有 token 类型 id。
        torch.cat(token_type_ids, out=token_type_ids_cpu)
        # 异步拷贝到 GPU 并写入 kwargs。
        model_kwargs["token_type_ids"] = token_type_ids_cpu.to(
            device=self.device, non_blocking=True
        )
        # 返回模型 kwargs。
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        根据注意力后端的需要更新批次中请求的顺序。例如某些注意力
        后端(即 MLA)可能希望按注意力计算是计算受限还是内存受限
        来分离请求。

        Args:
            scheduler_output: 调度器输出。
        """
        # 无注意力模型没有 kv_cache_groups,但 Mamba 等模型虽无注意力
        # 却用 kv_cache 保存内部状态。因此检查 kv_cache 组数量而非
        # 仅看 self.model_config.is_attention_free。
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            # 无 KV cache 组,直接返回。
            return

        if self.reorder_batch_threshold is not None:
            # 设置了重排阈值时执行解码/prefill 分离重排。
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    def _init_kv_zero_meta(self) -> None:
        """_zero_block_ids 的一次性预计算。

        由 gpu_worker.py 在 CuMem 池上下文之外调用。
        """
        # 创建 KV 块清零器。
        self._kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=self.compilation_config.static_forward_context,
        )

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """对给定块 id 清零 KV cache 显存。"""
        if hasattr(self, "_kv_block_zeroer"):
            # 清零器已初始化则执行清零。
            self._kv_block_zeroer.zero_block_ids(block_ids)

    # 注意: 供子类 model runner 覆盖。
    def _init_device_properties(self) -> None:
        """从 torch.cuda.get_device_properties 初始化属性"""

        # 记录 SM 数量。
        self.num_sms = num_compute_units(self.device.index)

    # 注意: 供子类 model runner 覆盖。
    def _sync_device(self) -> None:
        # 同步设备。
        torch.accelerator.synchronize()

    # 获取或创建异步输出复制专用的 CUDA 流(惰性初始化)。
    def _get_or_create_async_output_copy_stream(self) -> torch.cuda.Stream:
        # 读取当前已保存的异步输出复制流。
        stream = self.async_output_copy_stream
        # 若尚未创建,则新建一个 CUDA 流并缓存起来。
        if stream is None:
            stream = torch.cuda.Stream()
            self.async_output_copy_stream = stream
        # 返回该 CUDA 流。
        return stream

    # 请求状态被移除时的钩子方法。
    def _on_request_state_removed(
        self,
        req_id: str,
        req_state: CachedRequestState | None,
    ) -> None:
        """供平台相关 runner 清理请求级旁路缓存的钩子。"""
        # 显式丢弃参数,表示默认无需额外清理。
        del req_id, req_state

    # 处理调度器输出中的编码器缓存生命周期更新。
    def _process_encoder_cache_scheduler_output(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """应用调度器侧的编码器缓存生命周期更新。"""
        # 遍历调度器要求释放的多模态哈希。
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            # 从编码器缓存中弹出对应条目(若存在)。
            self.encoder_cache.pop(mm_hash, None)

    # 根据调度器输出更新缓存请求状态与持久批次。
    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
        """根据调度器输出更新缓存请求状态和持久批次。

        更新后的状态将由 `_prepare_inputs` 函数用于创建模型的输入 GPU 张量。

        当批次中存在新增/恢复/暂停/已完成的请求时,SamplingMetadata 会被更新
        并复制到 GPU。
        """
        # 从缓存状态中移除已完成的请求。
        for req_id in scheduler_output.finished_req_ids:
            # 弹出该请求的缓存状态(可能不存在)。
            req_state = self.requests.pop(req_id, None)
            # 调用移除钩子,便于平台侧做额外清理。
            self._on_request_state_removed(req_id, req_state)
            # 移除该请求的 prompt logprobs 数量记录。
            self.num_prompt_logprobs.pop(req_id, None)
        # 通知 late_interaction runner 这些请求已结束。
        self.late_interaction_runner.on_requests_finished(
            scheduler_output.finished_req_ids
        )
        # 将已完成的请求从持久批次中移除。
        # 注意(woosuk): 存在 finished_req_ids 与 scheduled_req_ids 重叠的边界情况。
        # 这发生在请求被中止后又以相同 ID 重新提交时。此时我们把它们视为两个不同的
        # 请求——先清除第一个请求的缓存状态,再把第二个当作新请求处理。
        for req_id in scheduler_output.finished_req_ids:
            # 从持久批次中移除该请求。
            self.input_batch.remove_request(req_id)

        # 为新分配的缓存块清零 GPU 显存,防止残留的 NaN/旧数据污染注意力或 SSM 计算。
        if scheduler_output.new_block_ids_to_zero:
            # 对这些新块执行清零操作。
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
        # 若调度器指定了 KV cache 块拷贝,则原地执行拷贝。
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

        # 释放已缓存的编码器输出。
        self._process_encoder_cache_scheduler_output(scheduler_output)

        # 将未被调度的请求从持久批次中移除。
        # 注意(woosuk): 未被调度的请求要么是被抢占的请求,要么是本步未被调度但将来
        # 仍可能被再次调度的运行中请求。我们将它们移出持久批次,但保留其缓存状态,
        # 因为它们将来某个时刻会被再次调度。
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # 注意(zhuohan): cached_req_ids 与 resumed_req_ids 通常不相交,因此除
        # reset_prefix_cache 中的强制抢占场景外,`(scheduled_req_ids - resumed_req_ids)
        # == scheduled_req_ids` 成立。在该场景下我们把 resumed_req_ids 也纳入未调度
        # 集合,使它们在经由正常的恢复请求路径重新调度之前,先从持久批次中被清除。
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # 注意(woosuk): 持久批次优化假设相邻批次包含的请求大多相同。若批次间请求
        # 重叠度很低(例如在两组不同请求之间交替),该优化将变得非常低效。
        for req_id in unscheduled_req_ids:
            # 从持久批次中移除未被调度的请求。
            self.input_batch.remove_request(req_id)

        # 判断是否启用 ngram_gpu 投机解码。
        is_ngram_gpu = (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        )
        # 若启用 ngram_gpu,则准备跟踪新增请求的列表。
        if is_ngram_gpu:
            ngram_gpu_new_reqs: list[CachedRequestState] = []

        # 待加入持久批次的请求列表。
        reqs_to_add: list[CachedRequestState] = []
        # 延迟到模型前向之后再执行的投机解码修正列表。
        deferred_spec_decode_corrections = []

        # 将新请求加入缓存状态。
        for new_req_data in scheduler_output.scheduled_new_reqs:
            # 取出新请求的请求 ID。
            req_id = new_req_data.req_id
            # 若该 ID 已存在于缓存状态中(仅流式场景会出现)。
            if req_id in self.requests:
                # 更新流式请求状态并加入待添加列表。
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            # 取出采样参数与池化参数。
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            # 若采样类型为"指定随机种子"。
            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                # 在当前设备上创建随机数生成器并设置种子。
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                # 否则不使用专属生成器。
                generator = None

            # 若是池化模型,则校验并应用池化参数更新。
            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            # 为该新请求构建缓存请求状态对象。
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                prompt_is_token_ids=new_req_data.prompt_is_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )
            # 将新请求状态登记到请求字典。
            self.requests[req_id] = req_state
            # 在 late_interaction runner 中注册该请求。
            self.late_interaction_runner.register_request(req_id, pooling_params)

            # 若请求了 prompt logprobs,则记录其数量(-1 表示整个词表)。
            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # 仅对使用 M-RoPE 的模型有效(例如 Qwen2-VL)。
            if self.uses_mrope:
                # 初始化 M-RoPE 位置。
                self._init_mrope_positions(req_state)

            # 仅对使用 XD-RoPE 的模型有效(例如 HunYuan-VL)。
            if self.uses_xdrope_dim > 0:
                # 初始化 XD-RoPE 位置。
                self._init_xdrope_positions(req_state)

            # 将新请求加入待添加列表。
            reqs_to_add.append(req_state)
            # 为 ngram_gpu 全张量拷贝跟踪新请求。
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)

        # 更新运行中/恢复中请求的状态。
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # 在裁剪前保存调度器分配的投机 token 数,使 prev_num_draft_len
        # 保留乐观计数,用于后续的拒绝修正。
        original_num_spec_per_req: dict[str, int] = {}
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            # 记录每个请求原始的投机 token 数量。
            for req_id, toks in scheduled_spec_tokens.items():
                original_num_spec_per_req[req_id] = len(toks)
            # 针对无效草稿同步更新调度器侧的计数。
            update_scheduler_for_invalid_drafts(
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens_cpu,
                scheduler_output,
                self.input_batch.req_id_to_index,
            )
        # 若启用异步投机解码,先清零上一轮的草稿数量数组。
        if self.use_async_spec_decode:
            self.prev_num_draft_tokens.np.fill(0)

        # 遍历调度器返回的缓存请求数据,逐个更新状态。
        for i, req_id in enumerate(req_data.req_ids):
            # 取出该请求的缓存状态。
            req_state = self.requests[req_id]
            # 取出本次已计算的 token 数。
            num_computed_tokens = req_data.num_computed_tokens[i]
            # 取出新增的块 ID 列表。
            new_block_ids = req_data.new_block_ids[i]
            # 判断该请求是否从抢占中恢复。
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            # 取出该请求的输出 token 总数。
            num_output_tokens = req_data.num_output_tokens[i]
            # 查询该请求在持久批次中的索引(可能不存在)。
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # 异步调度+投机解码下,依据上一轮草稿长度决定是否修正已计算 token 数。
            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len 用于带投机解码的异步调度模式,表示是否需要
                # 更新该请求的 num_computed_tokens。例如:
                # 第一步: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0。
                # 第二步: num_computed_tokens = 100(prompt 长度),
                # spec_tokens = [a,b], prev_num_draft_len = 0。
                # 第三步: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2。
                # 第一步和第二步的 num_computed_tokens 不包含投机 token 的长度,
                # 但第三步包含。只有当 prev_num_draft_len > 0 时才需要更新
                # num_computed_tokens。
                if req_index is None:
                    # 请求不在批次中,直接清零草稿长度。
                    req_state.prev_num_draft_len = 0
                else:
                    # 乐观假设草稿全部被接受;排队一个修正操作,在模型前向之后
                    # 调用,以保持异步调度。该修正在 _prepare_inputs 中于 GPU 侧完成。
                    optimistic_num_accepted = req_state.prev_num_draft_len
                    # 用 -1 占位扩展输出 token 列表(等待后续修正)。
                    req_state.output_token_ids.extend([-1] * optimistic_num_accepted)

                    # 将该请求加入延迟修正列表。
                    deferred_spec_decode_corrections.append(
                        (req_id, optimistic_num_accepted, req_state)
                    )

                    # 查询该请求在上一轮批次中的索引。
                    prev_req_index = (
                        self.input_batch.prev_req_id_to_index.get(req_id)
                        if self.input_batch.prev_req_id_to_index
                        else None
                    )
                    # 记录乐观接受的草稿 token 数。
                    if prev_req_index is not None:
                        self.prev_num_draft_tokens.np[prev_req_index] = (
                            optimistic_num_accepted
                        )

                    # ngram_gpu 模式下同步增加无投机 token 数。
                    if is_ngram_gpu and optimistic_num_accepted > 0:
                        self.input_batch.num_tokens_no_spec[req_index] += (
                            optimistic_num_accepted
                        )

            # 更新缓存状态:记录本次已计算的 token 数。
            req_state.num_computed_tokens = num_computed_tokens

            # 非最后一个 PP rank 时,需要从调度器数据恢复新 token。
            if not is_last_rank:
                if not req_data.new_token_ids:
                    # 异步调度的流水线并行:采样 token 通过 GPU 广播传播。
                    new_token_ids: list[int] = []
                else:
                    # 非异步调度的流水线并行:调度器回传采样 token ID,
                    # 因为第一阶段 worker 与最后阶段 worker 之间没有直接通信。
                    new_token_ids = req_data.new_token_ids[i]
                    # 追加上一步的采样 token(若有)。这不包括投机 token 等"未验证" token。
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # 最常见情形下避免列表切片。
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        # 扩展输出 token 列表。
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                # 某些输出 token 因同步 KV 加载失败被丢弃,或 output_token_ids
                # 因上面的乐观扩展(async spec decode)而虚高。对齐缓存状态。
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    # 计算该请求的 token 边界(prompt+输出)。
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    # 更新无投机 token 数。
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # 更新块 ID。
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # 将新块追加到现有块 ID 之后。
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                # 断言该请求不在持久批次中且新块 ID 存在。
                assert req_index is None
                assert new_block_ids is not None
                # 该请求从抢占中恢复。
                # 用新块 ID 替换现有块 ID。
                req_state.block_ids = new_block_ids

            if req_index is None:
                # 该请求不在持久批次中。
                # 它要么被抢占后恢复,要么上一步未被调度而需要重新加入。

                if self.use_async_scheduling and num_output_tokens > 0:
                    # 异步调度场景下必须恢复被恢复请求的输出 token ID,
                    # 以获得正确的 input_ids。
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                # 将该请求加入待添加列表。
                reqs_to_add.append(req_state)
                # 为 ngram_gpu 全张量拷贝跟踪恢复的请求。
                if is_ngram_gpu:
                    ngram_gpu_new_reqs.append(req_state)
                continue

            # 更新持久批次。
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                # 向块表写入新块 ID 行。
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # 最后一个 rank 无需更新 token_ids_cpu,因为采样 token 已被缓存。
            if not is_last_rank:
                # 起始 token 索引。
                start_token_index = self.input_batch.num_tokens_no_spec[req_index]
                # 对分块 prefill,num_computed_tokens 可能小于 num_tokens_no_spec。
                # 异步调度的流水线并行:没有 new_token_ids,按 num_computed_tokens
                # 推进 num_tokens_no_spec。
                end_token_index = max(
                    start_token_index,
                    num_computed_tokens + len(new_token_ids),
                )
                # 若结束索引大于起始索引,则写入新 token。
                if end_token_index > start_token_index:
                    if new_token_ids:
                        # 将 new_token_ids 写入 token_ids_cpu。
                        num_new_tokens = end_token_index - start_token_index
                        tokens_to_append = new_token_ids[-num_new_tokens:]
                        self.input_batch.token_ids_cpu[
                            req_index, start_token_index:end_token_index
                        ] = tokens_to_append
                    # 标记这些位置为 token ID。
                    self.input_batch.is_token_ids[
                        req_index, start_token_index:end_token_index
                    ] = True
                    # 推进无投机 token 数。
                    self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # 将投机 token ID 写入 token_ids_cpu。
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
            # 在 ngram 裁剪后恢复调度器侧的草稿数量。
            if original_num_spec_per_req:
                # 取出原始投机 token 数。
                orig = original_num_spec_per_req.get(req_id, 0)
                # 若与当前草稿长度不同则恢复。
                if orig != req_state.prev_num_draft_len:
                    req_state.prev_num_draft_len = orig

        # 将新增或恢复的请求加入持久批次。
        # 较小的空闲索引优先被填充。
        for request in reqs_to_add:
            # 将请求加入批次。
            self.input_batch.add_request(request)
            # 写入投机 token ID。
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # 若移除请求留下了空隙,则压缩批次状态。
        self.input_batch.condense()
        # 允许注意力后端潜在地重排批次。
        self._may_reorder_batch(scheduler_output)
        # 用待处理的更新刷新批次元数据。
        self.input_batch.refresh_metadata()

        # 在批次稳定后增量更新 ngram_gpu 张量。
        if is_ngram_gpu:
            update_ngram_gpu_tensors_incremental(
                self.input_batch,
                self.token_ids_gpu_tensor,
                self.num_tokens_no_spec_gpu,
                ngram_gpu_new_reqs,
                self.device,
                _pinned_idx_buf=self._ngram_pinned_idx_buf,
                _pinned_val_buf=self._ngram_pinned_val_buf,
            )

        # 若存在延迟的投机解码修正,则返回修正回调。
        if deferred_spec_decode_corrections:

            def correct_spec_decode_token_counts():
                # 获取 GPU 侧统计的有效采样 token 数。
                valid_sampled_token_count = self._get_valid_sampled_token_count()
                if not valid_sampled_token_count:
                    # 无有效数据则直接返回。
                    return
                prev_req_id_to_index = self.input_batch.prev_req_id_to_index
                if not prev_req_id_to_index:
                    return
                # 遍历延迟修正项。
                for (
                    req_id,
                    optimistic_num_accepted,
                    req_state,
                ) in deferred_spec_decode_corrections:
                    # 查询该请求在上一轮批次中的索引。
                    prev_req_index = prev_req_id_to_index.get(req_id)
                    if prev_req_index is None:
                        continue
                    # 有效接受数 = 有效采样数 - 1(最后一个为真实采样 token)。
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    # 计算乐观值与实际接受数的差值。
                    correction = optimistic_num_accepted - num_accepted
                    # 修正请求状态的已计算 token 数。
                    req_state.num_computed_tokens -= correction
                    # 查询该请求在当前批次中的索引。
                    cur_req_index = self.input_batch.req_id_to_index.get(req_id)
                    if cur_req_index is None:
                        continue
                    # 修正持久批次中的已计算 token 数。
                    self.input_batch.num_computed_tokens_cpu[cur_req_index] -= (
                        correction
                    )
                    # ngram_gpu 模式下同步修正无投机 token 数(CPU 与 GPU 两侧)。
                    if is_ngram_gpu and correction > 0:
                        self.input_batch.num_tokens_no_spec[cur_req_index] -= correction
                        self.num_tokens_no_spec_gpu[cur_req_index] -= correction

            # 返回修正函数,供调用方在模型前向之后执行。
            return correct_spec_decode_token_counts
        else:
            # 无需修正,返回 None。
            return None

    # 在模型执行后更新缓存状态(用于混合模型的 MTP/EAGLE 投机解码)。
    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """在模型执行后更新缓存状态。

        用于混合模型的 MTP/EAGLE:在线性注意力机制下只保留最后一个 token 的状态。
        在 MTP/EAGLE 中,草稿 token 的状态会一直保留,直到我们确定每个序列接受了
        多少 token,并在下一次迭代中根据接受的 token 数量进行状态移位。
        """
        # 非投机解码或非混合模型时无需处理。
        if not self.speculative_config or not self.model_config.is_hybrid:
            return

        # 统计每个序列接受的 token 数。
        # 有效 token 从位置 0 起连续,因此统计非 -1 的 token 数即得到第一个
        # -1 出现的位置(即接受的 token 数)。
        num_reqs = output_token_ids.size(0)
        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)

        # mamba 缓存模式为 align 时走融合 GPU 后处理路径。
        if self.cache_config.mamba_cache_mode == "align":
            # 融合的 GPU 后处理:状态拷贝 + 每请求接受 token 更新,无需 CPU-GPU 同步。
            # 元数据(num_scheduled_tokens、num_draft_tokens、num_computed_tokens)
            # 已在 _prepare_inputs 中预先放置到 GPU 缓冲区。
            mamba_utils.postprocess_mamba_align_gpu(
                bufs=self._get_mamba_bufs(),
                num_reqs=num_reqs,
                num_accepted_tokens_gpu=self.num_accepted_tokens.gpu,
                num_accepted_tokens_cpu_tensor=(
                    self.input_batch.num_accepted_tokens_cpu_tensor
                ),
                input_batch=self.input_batch,
                kv_cache_config=self.kv_cache_config,
                forward_context=self.compilation_config.static_forward_context,
                mamba_state_copy_funcs=self.model.get_mamba_state_copy_func(),
            )

            # 断言事件已创建。
            assert self.num_accepted_tokens_event is not None
            # 在当前流上记录事件。
            self.num_accepted_tokens_event.record()
        else:
            # 将 GPU 上的接受 token 数异步拷贝到 CPU 张量。
            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
            )
            # 断言事件已创建。
            assert self.num_accepted_tokens_event is not None
            # 在当前流上记录事件。
            self.num_accepted_tokens_event.record()

            # mamba 缓存模式为 all 时,后处理全部 mamba 状态。
            if self.cache_config.mamba_cache_mode == "all":
                mamba_utils.postprocess_mamba_all(
                    scheduler_output,
                    self.kv_cache_config,
                    self.input_batch,
                    self.requests,
                    self.mamba_state_idx,
                    self.num_spec_tokens,
                    num_reqs,
                )

    # 从 scheduled_new_reqs 更新流式会话请求的状态。
    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """根据 `scheduled_new_reqs` 更新流式会话请求。

        从 InputBatch 中移除该请求(若存在),更新缓存状态,
        并为重新加入批次做好准备。

        注意: prompt_token_ids 包含中间输出 token——之前生成但现在成为
        输入上下文(即提示词的一部分)的 token。
        """
        # 先从持久批次中移除该请求。
        self.input_batch.remove_request(req_id)
        # 取出该请求的缓存状态。
        req_state = self.requests[req_id]

        # 用最新的请求数据更新各字段。
        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        # 在 late_interaction runner 中重新注册该请求。
        self.late_interaction_runner.register_request(req_id, req_state.pooling_params)
        # 更新块 ID 与已计算 token 数。
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        # 重新计算 prompt token 总数。
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # 清空 `output_token_ids`,因为之前的输出 token 现在已成为
        # `prompt_token_ids` 的一部分。
        req_state.output_token_ids.clear()

        # 若模型使用 M-RoPE,重新初始化其位置。
        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        # 返回更新后的请求状态。
        return req_state

    # 为指定请求初始化 M-RoPE 位置。
    def _init_mrope_positions(self, req_state: CachedRequestState):
        # 获取当前模型实例。
        model = self.get_model()
        # 断言模型支持 M-RoPE。
        assert supports_mrope(model), "M-RoPE support is not implemented."
        # 类型转换为支持 M-RoPE 的模型。
        mrope_model = cast(SupportsMRoPE, model)

        # `prompt_embeds` 是直通模态(没有 grid_thw),而模型的 M-RoPE 代码假定
        # 每个特征都有网格信息,因此将其过滤掉。prompt_embeds 的位置按文本位置处理。
        mrope_features = [
            f for f in req_state.mm_features if f.modality != "prompt_embeds"
        ]

        # 依据输入形式确定输入 token 序列。
        if req_state.prompt_token_ids is not None:
            # 使用 prompt token ID 列表。
            input_tokens = req_state.prompt_token_ids
        elif req_state.prompt_embeds is not None:
            # 对仅嵌入的输入,当 mm_features 为空时(即上面的过滤结果),
            # get_mrope_input_positions 只需要序列长度。
            seq_len = req_state.prompt_embeds.shape[0]
            # 以连续序号作为输入 token。
            input_tokens = list(range(seq_len))
        else:
            # 两者都不存在时抛出错误。
            raise ValueError(
                "M-RoPE requires either prompt_token_ids or prompt_embeds."
            )

        # 调用模型的 M-RoPE 位置计算方法,保存结果与位置偏移量。
        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                input_tokens,
                mrope_features,
            )
        )

    # 为指定请求初始化 XD-RoPE 位置。
    def _init_xdrope_positions(self, req_state: CachedRequestState):
        # 获取当前模型实例。
        model = self.get_model()
        # 类型转换为支持 XD-RoPE 的模型。
        xdrope_model = cast(SupportsXDRoPE, model)
        # 断言 prompt token ID 可用。
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        # 断言模型支持 XD-RoPE。
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        # 调用模型的 XD-RoPE 位置计算方法并保存结果。
        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    # 从调度器输出中提取多模态的 kwargs 输入。
    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        # 调度器输出为空或模型不是多模态原始输入模型时,返回空字典。
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        # 收集所有新请求的多模态特征数据。
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append((feature.modality, feature.data))

        # 一次性输入所有模态
        mm_kwargs_combined: BatchedTensorInputs = {}
        # 按模态分组并批量化,再合并到一个字典。
        for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=self.device,
            pin_memory=PIN_MEMORY,
        ):
            mm_kwargs_combined.update(mm_kwargs_batch)

        # 返回合并后的多模态输入。
        return mm_kwargs_combined

    # 构建用于性能分析的虚拟多模态 kwargs 输入。
    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        # 非多模态原始输入模型时返回空字典。
        if not self.is_multimodal_raw_input_only_model:
            return {}

        # 获取多模态预算管理器。
        mm_budget = self.mm_budget
        assert mm_budget is not None

        # 无塔式模态(仅嵌入模式)时返回空字典。
        if not mm_budget.mm_max_toks_per_item:
            return {}  # 无塔式模态(仅嵌入模式)

        # 获取 token 数最多的模态并构造虚拟批次。
        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    # 计算给定数组的累积和与批内 arange。
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        arange_out: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        """获取给定数组的累积和与批内 arange。
        例如,[2, 5, 3] -> [2, 7, 10],arange 写入 arange_out[:10]:
        [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]。
        等价于但快于:
        np.concatenate([np.arange(n) for n in num_tokens])
        """
        # 第 1 步: [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        # 总 token 数。
        total_num_tokens = cu_num_tokens[-1]
        # 第 2 步: [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # 第 3 步: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        np.subtract(
            self.arange_np[:total_num_tokens],
            cumsums_offsets,
            out=arange_out[:total_num_tokens],
        )

        # 返回累积和数组。
        return cu_num_tokens

    # 构建 prev_positions 映射:当前位置 -> 上一轮位置(新请求为 -1)。
    def _compute_prev_positions(self, num_reqs: int) -> None:
        """构建 prev_positions 映射:当前位置 -> 上一轮位置(新请求为 -1)。

        将映射结果填充到 self.prev_positions.np[:num_reqs]。
        """
        # 获取上一轮的请求 ID 到索引映射。
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        prev_positions = self.prev_positions.np[:num_reqs]

        # 上一轮映射为空时,全部填充 -1。
        if not prev_req_id_to_index:
            prev_positions.fill(-1)
            return

        # 逐请求查询上一轮的位置。
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            prev_positions[i] = prev_req_id_to_index.get(req_id, -1)

    # 为当前批次准备输入 ID。
    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """为当前批次准备输入 ID。

        谨慎处理 `prev_sampled_token_ids`:它可能缓存自上一轮引擎迭代,
        此时需要把 GPU 上这些 token 拷贝到 input_ids 中对应的槽位。

        使用 self.prev_positions[:num_reqs],它将当前位置映射到上一轮位置
        (新请求为 -1)。
        """

        # 上一轮采样 token 不可用时,按正常调度情形处理。
        if self.input_batch.prev_sampled_token_ids is None:
            # 正常调度情形
            # 将 input_ids 拷贝到 GPU。
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                # 启用提示词嵌入时同步拷贝嵌入与 token 标记。
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # 异步调度情形:上一轮的部分 decode 请求在 input_ids_cpu 中没有条目,
        # 需要在 GPU 上从 prev_sampled_token_ids 拷贝。
        prev_positions = self.prev_positions.np[:num_reqs]
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        # 采样 token 的扁平化目标索引列表。
        sample_flattened_indices: list[int] = []
        # 投机 token 的扁平化目标索引列表。
        spec_flattened_indices: list[int] = []
        # 上一轮草稿 token 的源索引列表。
        prev_draft_token_indices: list[int] = []
        # 上一轮请求索引列表。
        prev_indices: list[int] = []
        # 是否所有公共请求的索引都一一对应。
        common_indices_match = True
        # 最大的扁平化索引。
        max_flattened_index = -1
        # 投机 token 总数。
        total_num_spec_tokens = 0

        # 遍历当前批次中的每个请求。
        for cur_index in range(num_reqs):
            # 获取该请求的上一轮位置。
            prev_index = prev_positions[cur_index]
            if prev_index < 0:
                # 新请求,跳过。
                continue
            # 记录上一轮索引。
            prev_indices.append(prev_index)
            # 获取请求 ID。
            req_id = self.input_batch.req_ids[cur_index]
            # 我们需要计算每个公共请求最后一个 token 的扁平化
            # input_ids 索引。
            draft_len = len(scheduled_spec_tokens.get(req_id, ()))
            # 累加投机 token 数。
            total_num_spec_tokens += draft_len
            flattened_index = cu_num_tokens[cur_index].item() - 1
            # 示例: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
            # sample_flattened_indices = [0, 2, 5]
            # spec_flattened_indices = [1,   3, 4,    6, 7]
            sample_flattened_indices.append(flattened_index - draft_len)
            spec_flattened_indices.extend(
                range(flattened_index - draft_len + 1, flattened_index + 1)
            )
            start = prev_index * self.prev_num_spec_tokens
            # prev_draft_token_indices 用于确定哪些 draft_tokens_id
            # 应被拷贝到 input_ids。
            # 示例: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
            # 展平 draft_tokens_id [1,2,3,4,5,6]
            # 每个请求的 draft_len [1, 2, 1]
            # 则 prev_draft_token_indices 为 [0,   2, 3,   4]
            prev_draft_token_indices.extend(range(start, start + draft_len))
            # 判断索引是否完全对应。
            common_indices_match &= prev_index == flattened_index
            # 更新最大扁平化索引。
            max_flattened_index = max(max_flattened_index, flattened_index)

        # 公共请求(与上一轮相同)的数量。
        num_common_tokens = len(sample_flattened_indices)
        # 不含投机 token 的总调度数。
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if self.enable_prompt_embeds:
            # 多模态嵌入路径读取 is_token_ids.gpu;其 .cpu 副本每步都会刷新,
            # 但下面的异步快速路径只散写 input_ids.gpu,因此这里也要刷新
            # is_token_ids.gpu。
            self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens < total_without_spec:
            # 若并非所有请求都是上一轮的 decode,需要先把 input_ids_cpu
            # 拷贝到 GPU。
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
        if num_common_tokens == 0:
            # 与上一轮没有公共请求。
            # 因此 input_ids.cpu 已包含全部输入 ID。
            return
        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # 常见情形优化:批次未变化且没有发生重排。
            # 两组索引都是 0..N-1 的同一排列,因此可以用单一切片直接拷贝。
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            return
        # 异步上传索引张量,使 scatter 可以非阻塞。
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        # 将上一轮的采样 token 散写到 input_ids 的对应位置。
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # 在采样 token 散写完成后再散写草稿 token。
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        # 断言草稿 token ID 为张量类型。
        assert isinstance(self._draft_token_ids, torch.Tensor)
        # 异步上传草稿 token 的目标索引张量。
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        # 异步上传草稿 token 的源索引张量。
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)

        # 因为 input_ids 的 dtype 是 torch.int32,
        # 所以这里把 draft_token_ids 转换为 torch.int32。
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        # 将草稿 token 散写到 input_ids 的对应位置。
        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    # 计算编码器序列长度张量(交叉注意力场景)。
    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
        for_cudagraph_capture: bool = False,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        # 非交叉注意力 KV 规格时无需编码器长度。
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # 为未实际调度的填充请求清零缓冲区(CUDA 图场景)。
        self.encoder_seq_lens.np[:num_reqs] = 0

        # 构建 encoder_seq_lens 数组:请求索引 -> 本批次已调度输入的编码器长度。
        for req_id in num_scheduled_tokens:
            # 获取请求索引与缓存状态。
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                # 无多模态特征时编码器长度为 0。
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # 获取运行中编码器请求的编码器输入 token 总数——
            # 无论编码是否完成都计入,以便交叉注意力知道要关注多少编码器 token。
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            # 写入编码器长度。
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens
        if for_cudagraph_capture:
            # CUDA 图捕获期间需要使用真实的编码器长度,
            # 以便 max_seqlen_k 以正确的值被捕获。
            max_encoder_len = getattr(
                self.model_config.hf_config,
                "max_source_positions",
                self.max_encoder_len,
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len

        # 将编码器长度拷贝到 GPU。
        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        # 返回 GPU 与 CPU 两侧的编码器长度。
        return encoder_seq_lens, encoder_seq_lens_cpu

    # 为模型前向准备输入张量。
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """
        返回:
            tuple[logits_indices, spec_decode_metadata]
        """
        # 本步调度的 token 总数。
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        # 持久批次中的请求数。
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # 优化:先开始拷贝块表。
        # 这样可以与后续的 CPU 操作重叠执行。
        self.input_batch.block_table.commit_block_table(num_reqs)

        # 获取请求索引。
        # 例如,[2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )

        # 获取各 token 的位置。
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )

        # 计算 M-RoPE 位置。
        # 仅对使用 M-RoPE 的模型有效(例如 Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # 计算 XD-RoPE 位置。
        # 仅对使用 XD-RoPE 的模型有效(例如 HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)

        # 获取 token 索引。
        # 例如,[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # 其中 M 是 max_model_len。
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        # 转换为 torch 张量以供索引选择使用。
        token_indices_tensor = torch.from_numpy(token_indices)

        # 注意(woosuk): 这里用 torch.index_select 而不是 np.take,
        # 因为对大张量而言 torch.index_select 比 np.take 快得多。
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        if self.enable_prompt_embeds:
            # 启用提示词嵌入时,同步选取 token 标记位。
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # 因为没有在 InputBatch 上预分配巨大的 prompt_embeds CPU 张量,
        # 所以需要把提示词嵌入填充到 GpuModelRunner 预分配张量的对应位置。
        if self.input_batch.req_prompt_embeds:
            # 输出的起始索引。
            output_idx = 0
            for req_idx in range(num_reqs):
                # 该请求调度的 token 数。
                num_sched = num_scheduled_tokens[req_idx]

                # 若该请求没有嵌入则跳过。
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # 若该请求没有调度任何 token 则跳过。
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                # 取出该请求的嵌入及已计算位置。
                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # 若读取位置超出可用嵌入则跳过。
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # 拷贝可用的嵌入。
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                # 推进输出索引。
                output_idx += num_sched

        # 准备注意力元数据。
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # 注意: 将 query_start_loc 填充为非递减,因为 FlashAttention 等
        # 内核要求如此
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        # 计算乐观序列长度(假设上一轮所有草稿 token 都被接受)。
        # 存入 optimistic_seq_lens_cpu,供 _build_attention_metadata(max_seq_len)
        # 与 discard_request_mask 使用。
        # seq_lens(GPU)稍后将使用同样的乐观值计算。
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        # 填充请求位置清零。
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # 构建 prev_positions 映射:当前位置 -> 上一轮位置(新请求为 -1)。
        # 用于从上一轮的 GPU 张量中收集数据。
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)

        # 收集每个请求的总 token 数。
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # 记录哪些请求不应被采样,
        # 以便在返回前清除这些请求的采样 token。
        self.discard_request_mask.np[:num_reqs] = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # 从 CPU 同步接受 token 数(混合模型由 _update_states_after_model_execute
        # 设置)。异步调度(非 align)下跳过:CPU 副本会与在途 D2H 拷贝及
        # 批次行移动产生竞争。
        needs_cpu_accepted_counts = self.num_accepted_tokens_event is not None and not (
            self.use_async_scheduling and self.cache_config.mamba_cache_mode != "align"
        )
        if needs_cpu_accepted_counts:
            # 断言事件已创建。
            assert self.num_accepted_tokens_event is not None
            # 等待 GPU 拷贝完成。
            self.num_accepted_tokens_event.synchronize()
            # 异步模式:condense() 重排了索引,需要用 prev_positions 映射。
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                # 新请求填充为 1。
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # 非异步模式:直接使用 CPU 值。
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
            # 填充位置填充为 1。
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()
        else:
            # 默认填 1;下方 update_num_computed_tokens_for_batch_change 会依据
            # valid_sampled_token_count 修正有草稿的行。
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

        # mamba prev_last_scheduled_idx 存在时,预处理 mamba all 的投机解码。
        if self.mamba_prev_last_scheduled_idx is not None:
            mamba_utils.preprocess_mamba_all_specdec(
                scheduler_output,
                self.input_batch,
                self.mamba_state_idx,
                num_reqs,
                self.mamba_prev_last_scheduled_idx,
            )

        # 更新 GPU 上的 num_computed_tokens。异步投机解码下 CPU 值是乐观的
        # (假设全部草稿被接受),内核会在 GPU 上用上一步的
        # valid_sampled_token_count_gpu 进行修正。否则直接从 CPU 拷贝。
        if (
            self.use_async_spec_decode
            and self.valid_sampled_token_count_gpu is not None
            and prev_req_id_to_index
        ):
            # 拷贝位置映射与草稿数量到 GPU。
            self.prev_positions.copy_to_gpu(num_reqs)
            self.prev_num_draft_tokens.copy_to_gpu()
            # 异步上载 CPU 侧的已计算 token 数。
            cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
                device=self.device, non_blocking=True
            )
            # 用 GPU 内核按批次变化修正已计算 token 数。
            update_num_computed_tokens_for_batch_change(
                self.num_computed_tokens,
                self.num_accepted_tokens.gpu[:num_reqs],
                self.prev_positions.gpu[:num_reqs],
                self.valid_sampled_token_count_gpu,
                self.prev_num_draft_tokens.gpu,
                cpu_values,
            )
        else:
            # 直接异步拷贝 CPU 值到 GPU。
            self.num_computed_tokens[:num_reqs].copy_(
                self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )

        # 写入请求索引并拷贝到 GPU。
        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        # 拷贝查询位置并更新调度数量。
        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        # 计算各 token 的最终位置(已计算数 + 批内偏移)。
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
        # 计算各请求的序列长度并清零填充部分。
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        # 计算槽位映射。
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # 将张量拷贝到 GPU。
        self._prepare_input_ids(
            scheduler_output,
            num_reqs,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            # 仅对使用 M-RoPE 的模型有效(例如 Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # 仅对使用 XD-RoPE 的模型有效(例如 HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        # 异步投机解码下修正 M/XD-RoPE 位置的漂移。
        if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
            # 计算 GPU 与 CPU 侧已计算 token 数的差值。
            drift = self.num_computed_tokens[req_indices_gpu].to(
                torch.int64
            ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
                device=self.device, dtype=torch.int64, non_blocking=True
            )
            # 选择对应的位置张量并累加漂移。
            target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
            target.gpu[:, :total_num_scheduled_tokens] += drift

        # 判断是否使用投机解码。
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # 注意(woosuk): 由于分块 prefill,批次中可能包含部分请求。虽然不应从
            # 这些部分请求中采样 token,但为简单起见我们仍然采样,之后忽略来自
            # 部分请求的采样 token。
            # TODO: 支持 prompt logprobs。
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # 获取每个请求的草稿 token 数。
            # 遍历字典而非全部请求,因为并非所有请求都有草稿 token。
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # 对分块 prefill 使用 -1 而非 0 作为掩码,因为引导解码可能回滚投机 token。
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                # 获取请求索引与草稿长度。
                req_idx = self.input_batch.req_id_to_index[req_id]
                draft_len = len(draft_token_ids)
                num_draft_tokens[req_idx] = draft_len
                # 纯 decode 请求(调度数 = 草稿数 + 1)单独记录。
                if num_scheduled_tokens[req_idx] == draft_len + 1:
                    num_decode_draft_tokens[req_idx] = draft_len
            # 计算投机解码元数据。
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # 供部分注意力后端(如 GDN)的仅 decode CUDA 图使用。
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # LoRA 热交换模型
        if self.lora_config:
            # 断言采样 token 总数不超过最大批处理 token 数。
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            # 设置当前激活的 LoRA。
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )

        # 返回 logits 索引与投机解码元数据。
        return (
            logits_indices,
            spec_decode_metadata,
        )

    # 为所有注意力层构建元数据。
    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        返回:
            tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # 无注意力模型不需要注意力元数据
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        # 未提供填充数量时使用实际数量。
        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        # 每层注意力元数据字典。
        attn_metadata: PerLayerAttnMetadata = {}
        # 存在微批切片时,为每个微批创建一个字典。
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # 对某些带滑动窗口模型的注意力后端(如 FA),捕获时需要确保后端看到的
            # max_seq_len 不小于滑动窗口大小,以便选择正确的内核。
            max_seq_len = self.max_model_len
        else:
            # 取乐观序列长度的最大值。
            max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        # 获取 KV cache 组列表。
        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        # 内部函数:获取指定 KV 组的块表张量。
        def _get_block_table(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                # 仅编码器注意力使用全零块表。
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                # 获取设备侧块表张量。
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

            # 用 NULL_BLOCK_ID(空块)填充未使用的块表条目,以支持 CUDAGraph 填充。
            # 块 0 预留给填充。
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
            return blk_table_tensor

        # 断言槽位映射已提供。
        assert slot_mappings is not None
        # 获取第 0 组的块表与槽位映射。
        block_table_gid_0 = _get_block_table(0)
        slot_mapping_gid_0 = slot_mappings[0]

        # 路由专家已初始化时,快照本步的槽位映射到私有设备缓冲区。
        if self.routed_experts_initialized:
            # 将本步的注意力 slot_mapping 拷贝到私有设备缓冲区。共享的
            # ``slot_mappings[attn_gid]`` 由注意力块表持有,并会被下一次
            # ``_prepare_inputs`` 覆盖;我们需要一个稳定的快照,因为异步 D2H
            # 拷贝可能仍在下一步运行时于拷贝流上处于在途状态。
            slot_mapping_attn = slot_mappings[self.routed_experts_capturer.attn_gid]
            self.routed_experts_slot_mapping_device[:num_tokens].copy_(
                slot_mapping_attn[:num_tokens]
            )

        # 获取 CPU 侧的各计数张量切片。
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
            :num_reqs_padded
        ]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
        seq_lens_cpu_upper_bound = seq_lens_cpu

        # is_prefilling: 请求是否仍处于 prefill 阶段。
        # 供 mamba 后端区分真正的 decode 与短 extend。
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
        # 填充行清零,防止 condense() 的旧数据在 CUDA 图模式下把填充误判为 prefill。
        is_prefilling[num_reqs:] = False

        if self.use_async_spec_decode:
            # 异步模式下以 GPU 张量为准。
            seq_lens_cpu = None
            num_computed_tokens_cpu = None

        # 在构建注意力元数据之前计算 mm_prefix 双向区间,使构建器可以在 build()
        # 中处理它们。默认跳过超出 sliding_window 的区间,防止较早的 token 跨
        # 整张图像进行注意力。在内核内将 mm_prefix 限制到滑动窗口的模型
        # (如 Gemma4——它需要在滑动层上同时满足 HF 的 (causal OR blockwise) AND
        # sliding_window)选择不跳过,以便大于窗口的图像保留双向区间;随后由内核
        # 按查询逐个界定。
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            hf_text_config = self.model_config.hf_text_config
            # 获取滑动窗口大小。
            _bidi_sw = getattr(hf_text_config, "sliding_window", None)
            # 判断模型是否在内核中执行限制。
            _clamps_in_kernel = getattr(
                self.model, "mm_prefix_clamp_sliding_window", False
            )
            for req_id in self.input_batch.req_ids:
                # 该请求的图像文档区间列表。
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    if mm_feature.modality == "audio":
                        # 音频模态跳过。
                        continue
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    for r in img_doc_range:
                        if (
                            not _clamps_in_kernel
                            and _bidi_sw is not None
                            and (r[1] - r[0] + 1) > _bidi_sw
                        ):
                            # 超出滑动窗口且内核不处理时跳过。
                            continue
                        image_doc_ranges.append(r)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

        # 参考滑动窗口注意力(R-SWA):传递每请求的 prompt 长度,使注意力后端
        # 能保持前缀全局可见。后端拥有持久且 CUDA 图安全的 GPU 缓冲区。
        rswa_prefix_lens = None
        if self.model_config.rswa_window is not None:
            rswa_prefix_lens = num_prompt_tokens_cpu

        # replayssm 模式的 decode 基址 CPU 张量。
        replayssm_decode_base_cpu = None
        if self.cache_config.use_replayssm:
            replayssm_decode_base_cpu = (
                self.input_batch.replayssm_decode_base_cpu_tensor[:num_reqs_padded]
            )

        # 构建公共注意力元数据对象。
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens[:num_reqs_padded],
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            replayssm_decode_base_cpu=replayssm_decode_base_cpu,
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
            is_prefilling=is_prefilling,
            positions=self.positions[:num_tokens_padded],
            mm_req_doc_ranges=req_doc_ranges,
            rswa_prefix_lens=rswa_prefix_lens,
        )

        # DCP 世界规模大于 1 时计算本地序列长度。
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.optimistic_seq_lens_cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            # 将 DCP 本地序列长度写入公共元数据。
            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        # KV 共享快速 prefill 模式下记录 logits 索引。
        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # 在混合 KV cache 组之间缓存注意力元数据的构建结果。
        # 当相同的元数据构建器和 KVCacheSpec 跨不同混合组复用时,唯一变化的是块表,
        # 因此可以缓存构建结果,并在构建器支持时用 `builder.update_block_table`
        # 仅更新块表。
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        # 内部函数:为指定注意力组构建元数据。
        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            # 获取注意力组及其元数据构建器。
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            # 构建缓存键。
            cache_key = (kv_cache_spec, type(builder))

            # 获取级联注意力的前缀长度。
            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            # 额外的元数据构建参数。
            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(
                builder,
                (
                    Mamba2AttentionMetadataBuilder,
                    GDNAttentionMetadataBuilder,
                    BailingLinearAttentionMetadataBuilder,
                ),
            ):
                assert ubid is None, (
                    "UBatching not supported with GDN or linear attn yet"
                )
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )
                if (
                    isinstance(builder, Mamba2AttentionMetadataBuilder)
                    and self.mamba_prev_last_scheduled_idx is not None
                ):
                    extra_attn_metadata_args["prev_last_scheduled_idx"] = (
                        self.mamba_prev_last_scheduled_idx.gpu[:num_reqs_padded]
                    )

            # CUDA 图捕获时使用专用构建。
            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                # 缓存命中时仅更新块表。
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                # 正常构建元数据。
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                # 支持更新块表时缓存构建结果。
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i

            # 依据是否微批选择目标字典。
            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            # 同组的所有层共享同一份元数据。
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # 为每个 KV cache 组准备注意力元数据,并让同组内的层共享同一份元数据。
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            # 浅拷贝公共元数据。
            cm = copy(cm_base)  # 浅拷贝

            # 对每个 kv_cache_group,基本上只有编码器 seq_lens、块表和槽位映射会变化。
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
                for_cudagraph_capture=for_cudagraph_capture,
            )
            if kv_cache_gid > 0:
                # 非 0 组使用各自的块表与槽位映射。
                cm.block_table_tensor = _get_block_table(kv_cache_gid)
                cm.slot_mapping = slot_mappings[kv_cache_gid]

            # 投机解码时保存第一份公共注意力元数据供 drafter 使用。
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(
                    self.drafter,
                    (
                        EagleProposer,
                        DFlashProposer,
                        Gemma4Proposer,
                        ExtractHiddenStatesProposer,
                    ),
                ):
                    # 仅当 drafter 的 KV 组匹配时保存。
                    if self.drafter.kv_cache_gid == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm
            # 为多组提议者捕获每组块表。
            if self.speculative_config and isinstance(self.drafter, Step3p5MTPProposer):
                self.drafter.set_per_group_attn_metadata(
                    kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
                )
            elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):
                self.drafter.set_per_group_block_table(
                    kv_cache_gid, cm.block_table_tensor
                )

            # 为组内每个注意力组构建元数据。
            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    # 微批切片时逐个构建。
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        # 投机解码且发生了填充时,去除填充(drafter 目前仍只使用分块 CUDA 图并
        # 直接修改注意力元数据,因此不想使用带填充的元数据)。
        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )

        # 返回注意力元数据与投机解码公共元数据。
        return attn_metadata, spec_decode_common_attn_metadata

    # 计算每个 KV 组、每个注意力组的级联注意力前缀长度。
    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        返回:
            Optional[cascade_attn_prefix_lens]
                cascade_attn_prefix_lens 是二维的:
                ``[kv_cache_group_id][attn_group_idx]``,
                若不应使用级联注意力则为 None
        """

        # 是否使用了级联注意力。
        use_cascade_attn = False
        # KV cache 组数量。
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        # 为每个 KV 组初始化前缀长度列表。
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        # 遍历每个 KV 组中的注意力组。
        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    # 仅编码器注意力不使用级联。
                    cascade_attn_prefix_len = 0
                else:
                    # 不应使用级联注意力时为 0
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                # 记录该注意力组的前缀长度。
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                # 更新是否使用了级联注意力。
                use_cascade_attn |= cascade_attn_prefix_len > 0

        # 使用级联时返回前缀长度矩阵,否则返回 None。
        return cascade_attn_prefix_lens if use_cascade_attn else None

    # 计算级联注意力的公共前缀长度。
    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """计算级联注意力的公共前缀长度。

        注意(woosuk): 本函数返回的公共前缀长度专门用于级联注意力,而非请求间
        实际共享的 token 数。当级联注意力被禁用(use_cascade=False)时,即使请求
        共享公共 token,本函数也返回 0。此外,公共前缀长度会被截断为块大小的
        整数倍,并可能因下面说明的实现细节被进一步截断。

        Args:
            num_scheduled_tokens: 每个请求调度的 token 数。
            num_common_prefix_blocks: 共享的 KV cache 块数量。

        Returns:
            int: 以 token 计的公共前缀长度。
        """

        # 初始公共前缀长度 = 共享块数 × 块大小。
        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # 常见情形。
            return 0

        # 注意(woosuk): 级联注意力使用两个注意力内核:一个处理公共前缀,
        # 另一个处理其余部分。对第一个内核,我们把所有查询 token(可能来自
        # 不同请求)拼接起来,当作来自同一请求处理。然后用双向注意力处理
        # KV cache 中的公共前缀。重要的是,这意味着第一个内核不做任何掩码。

        # 考虑如下示例:
        # 请求 1 的输入查询: [D, E, X]
        # 请求 1 的 kv cache: [A, B, C, D, E, X]
        # 请求 1 的 num_computed_tokens: 3(即 [A, B, C])
        # 请求 2 的输入查询: [E, Y]
        # 请求 2 的 kv cache: [A, B, C, D, E, Y]
        # 请求 2 的 num_computed_tokens: 4(即 [A, B, C, D])

        # 若用 [A, B, C, D, E] 作为公共前缀,则第一个内核会计算输入查询
        # [D, E, X, E, Y] 与公共前缀 [A, B, C, D, E] 之间的双向注意力。
        # 但这是错的,因为请求 1 中的 D 不应关注公共前缀中的 E(即需要掩码)。
        # 为避免这一点,应以 [A, B, C, D] 作为公共前缀。
        # 也就是说,公共前缀应被各请求中最小的 num_computed_tokens 截断,
        # 再加一以包含查询的第一个 token。

        # 实际上,我们用 [A, B, C] 作为公共前缀,而不是 [A, B, C, D]
        # (即公共前缀被最小的 num_computed_tokens 截断,不加一)。
        # 这是出于一个实现细节:我们希望级联注意力始终使用两个内核。设想:
        # 请求 3 的输入查询: [D]
        # 请求 3 的 kv cache: [A, B, C, D]
        # 请求 3 的 num_computed_tokens: 3(即 [A, B, C])
        # 若用 [A, B, C, D] 作为请求 1-3 的公共前缀,则请求 3 将只由第一个内核
        # 处理,而第二个内核会得到空输入。这虽不是根本性问题,但当前实现不支持
        # 这种情况。
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # common_prefix_len 应为块大小的整数倍。
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        # 判断是否使用滑动窗口注意力。
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        # 判断是否使用局部注意力。
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        # 断言 KV 规格为注意力规格。
        assert isinstance(kv_cache_spec, AttentionSpec)
        # 询问构建器是否使用级联注意力。
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        # 使用级联时返回前缀长度,否则返回 0。
        return common_prefix_len if use_cascade else 0

    # 计算本批次所有 token 的 M-RoPE 位置。
    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        # 输出位置的写入指针。
        mrope_pos_ptr = 0
        # 遍历批次中的每个请求。
        for index, req_id in enumerate(self.input_batch.req_ids):
            # 取出请求状态。
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            # 获取已计算 token 数、调度 token 数与 prompt 长度。
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            # 划分本次调度中 prompt 部分与生成部分的长度。
            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            # 断言两部分之和等于调度数。
            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt 的 mrope_positions 已预计算
                # 确定源与目标区间。
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                # 从预计算结果拷贝 prompt 部分位置。
                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                # 推进写入指针。
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # 现场计算 completion 部分的 mrope_positions
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                # 断言位置偏移量已存在。
                assert req.mrope_position_delta is not None
                # 调用 M-RoPE 旋转嵌入获取后续位置张量。
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                # 推进写入指针。
                mrope_pos_ptr += completion_part_len

    # 计算本批次所有 token 的 XD-RoPE 位置。
    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        # 输出位置的写入指针。
        xdrope_pos_ptr = 0
        # 遍历批次中的每个请求。
        for index, req_id in enumerate(self.input_batch.req_ids):
            # 取出请求状态。
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            # 获取已计算 token 数、调度 token 数与 prompt 长度。
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            # 划分本次调度中 prompt 部分与生成部分的长度。
            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            # 断言两部分之和等于调度数。
            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt 的 xdrope_positions 已预计算
                # 确定源与目标区间。
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                # 从预计算结果拷贝 prompt 部分位置。
                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                # 推进写入指针。
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # 现场计算 completion 部分的 xdrope_positions
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                # 调用 XD-RoPE 旋转嵌入获取后续位置张量。
                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                # 推进写入指针。
                xdrope_pos_ptr += completion_part_len

    # 计算投机解码所需的元数据。
    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # 输入:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # 输出:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # 计算 logits 索引。
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # 第 1 步。
        # cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # _arange_scratch[:11]: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        # 第 2 步。 [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # 第 3 步。 [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += self._arange_scratch[: cu_num_sampled_tokens[-1]]

        # 计算 bonus logits 索引。
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # 计算 draft logits 索引。
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # _arange_scratch[:6]: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens = self._get_cumsum_and_arange(
            num_draft_tokens, self._arange_scratch, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += self._arange_scratch[: cu_num_draft_tokens[-1]]

        # 将各索引数组异步上传到 GPU。
        cu_num_draft_tokens = async_tensor_h2d(cu_num_draft_tokens, device=self.device)
        cu_num_sampled_tokens = async_tensor_h2d(
            cu_num_sampled_tokens, device=self.device
        )
        logits_indices = async_tensor_h2d(logits_indices, device=self.device)
        target_logits_indices = async_tensor_h2d(
            target_logits_indices, device=self.device
        )
        bonus_logits_indices = async_tensor_h2d(
            bonus_logits_indices, device=self.device
        )

        # 计算草稿 token ID。
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        # 构建并返回投机解码元数据。
        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    # 为 KV 共享快速 prefill 准备填充后的 logits 索引。
    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        # 断言预分配的索引缓冲区存在。
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        # logits 数量。
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        # 拷贝 logits 索引到缓冲区。
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # 上一轮迭代可能在 logits_indices[num_logits:] 中残留旧索引,其值可能
        # 大于当前迭代的批大小。为确保索引始终有效,用最后一个索引填充填充位。
        # 在 GPU 侧广播标量,以避免 .item() 触发 D2H 同步。
        self.kv_sharing_fast_prefill_logits_indices[num_logits:] = logits_indices[-1]
        # 为模型解码器部分分派 CUDA 图模式。
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_logits, invalid_modes={CUDAGraphMode.FULL}
        )
        # 填充后的 logits 数量。
        num_logits_padded = batch_desc.num_tokens
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        # 返回填充后的 logits 索引。
        return logits_indices_padded

    # 从调度器输出批量整理多模态编码器输入。
    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[tuple[str, MultiModalKwargsItem]],
        list[tuple[str, PlaceholderRange]],
    ]:
        """从已调度的编码器输入批量整理多模态输入。

        Args:
            scheduler_output: 包含已调度编码器输入的调度器输出。

        Returns:
            元组 (mm_hashes, mm_kwargs, mm_lora_refs),其中:
            - mm_hashes: 每个条目的多模态哈希列表
            - mm_kwargs: 每个条目的多模态 kwargs 列表
            - mm_lora_refs: 每个条目的 (req_id, placeholder_range) 列表
        """
        # 获取已调度的编码器输入。
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            # 无编码器输入时返回三个空列表。
            return [], [], []

        # 多模态哈希列表。
        mm_hashes = list[str]()
        # 多模态 kwargs 列表。
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # 多模态 LoRA 引用信息,用于把每个多模态条目映射回其请求与位置
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        # 遍历每个请求的编码器输入 ID。
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            # 取出请求状态。
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                # 取出多模态特征。
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    # 无数据则跳过。
                    continue

                # 记录哈希、kwargs 与 LoRA 引用。
                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        # 返回哈希、kwargs 与 LoRA 引用。
        return mm_hashes, mm_kwargs, mm_lora_refs

    # 缓存编码器输出,供后续多模态嵌入收集使用。
    def _cache_encoder_output(
        self,
        mm_hash: str,
        output: torch.Tensor,
        ec_manager_metadata: "EncoderCacheManagerMetadata | None",
        free_encoder_mm_hashes: list[str],
    ) -> None:
        """存储编码器输出,供后续多模态嵌入收集使用。"""
        # 显式丢弃未用参数。
        del ec_manager_metadata, free_encoder_mm_hashes
        # 按哈希存入编码器缓存。
        self.encoder_cache[mm_hash] = output
        # 视情况将编码器输出保存到连接器。
        self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

    # 执行多模态编码器并返回各条目的编码输出。
    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        # 从调度器输出批量整理多模态输入。
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        # 无待编码输入时直接返回。
        if not mm_kwargs:
            return []

        # `prompt_embeds` 是直通模态,张量已处于模型嵌入空间,无需运行编码器。
        # 这里把每个 `prompt_embeds` 张量直接注入编码器缓存,使
        # `_gather_mm_embeddings` 能通过标准的 `is_mm_embed` 路径拼接它。
        pe_indices = [
            i
            for i, (modality, _) in enumerate(mm_kwargs)
            if modality == "prompt_embeds"
        ]
        if pe_indices:
            # 逐个缓存 prompt_embeds 张量。
            for i in pe_indices:
                pe_tensor = mm_kwargs[i][1]["embedding"].data
                assert isinstance(pe_tensor, torch.Tensor)

                self._cache_encoder_output(
                    mm_hashes[i],
                    pe_tensor.to(self.device),
                    scheduler_output.ec_manager_metadata,
                    scheduler_output.free_encoder_mm_hashes,
                )
            # 从 mm_kwargs/mm_hashes/mm_lora_refs 中过滤掉 `prompt_embeds` 条目,
            # 因为它们不需要进一步的编码器处理。
            mm_hashes = [h for i, h in enumerate(mm_hashes) if i not in pe_indices]
            mm_kwargs = [k for i, k in enumerate(mm_kwargs) if i not in pe_indices]
            mm_lora_refs = [
                r for i, r in enumerate(mm_lora_refs) if i not in pe_indices
            ]
            if not mm_kwargs:
                # 过滤掉 `prompt_embeds` 后已无可编码内容
                return []

        # 判断是否启用编码器计时统计。
        should_time = bool(
            self.observability_config
            and self.observability_config.enable_mm_processor_stats
            and scheduler_output.scheduled_encoder_inputs
        )

        # 尽可能批量处理多模态输入:若批次中某请求包含多个模态,或与前一个请求
        # 模态不同,则单独处理以保持条目顺序。
        # FIXME(ywang96): 这是处理同一批次内多模态的临时方案,同时仍能受益于
        # 多模态输入批处理。正确的方案应是重排编码器输出。
        model = cast(SupportsMultiModal, self.model)

        # 启用 LoRA 且支持塔式连接器 LoRA 时,为编码器输入独立构建映射。
        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # (编码器批次结构与主批次不同)
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []

            # 遍历每个多模态条目并构建 LoRA 映射。
            for req_id, pos_info in mm_lora_refs:
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                # 优先使用 pos_info.get_num_embeds 精确统计 MM 嵌入 token 数。
                num_tokens = self.model.get_num_mm_encoder_tokens(  # type: ignore[attr-defined]
                    pos_info.get_num_embeds()
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * num_tokens)
                encoder_token_counts.append(num_tokens)

                if lora_id > 0:
                    # 收集涉及的 LoRA 请求。
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # 设置塔式适配器映射
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            # 仅当模型确实有连接器时才设置连接器映射。
            # 某些多模态模型从 `SupportsMultiModal` 继承了桩实现
            # `get_num_mm_connector_tokens`,它返回 None,不应被视为支持
            # 连接器 LoRA 的信号。
            mm_mapping = (
                self.model.get_mm_mapping()  # type: ignore[attr-defined]
                if hasattr(self.model, "get_mm_mapping")
                else None
            )
            if (
                mm_mapping is not None
                and mm_mapping.connector
                and hasattr(self.model, "get_num_mm_connector_tokens")
            ):
                # 计算连接器后的 token 数。
                post_op_counts = [
                    self.model.get_num_mm_connector_tokens(num_tokens)  # type: ignore[attr-defined]
                    for num_tokens in encoder_token_counts
                ]

                # 生成连接器 token 级映射。
                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(post_op_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                # 设置连接器 LoRA 映射。
                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        # 编码器输出列表。
        encoder_outputs: list[torch.Tensor] = []
        # 跟踪 mm_kwargs/mm_lora_refs 中的当前索引,把组映射回请求 ID
        current_item_idx = 0
        # 按模态分组批量处理。
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=PIN_MEMORY
        ):
            # 组输出。
            batch_outputs: MultiModalEmbeddings

            # EVS 与动态分辨率视频相关改动。
            # (ekhvedchenia): 处理多模态数据时限制峰值显存的临时 hack。
            # 这解决了调度器把过多视频样本放进单个批次的问题。调度器用剪枝后的
            # 视觉 token 数与计算预算比较,这是不正确的(应考虑输入媒体大小或
            # 未剪枝的输出视觉 token 数)。
            # nemotron 的动态分辨率视频通过 requires_sequential_video_encoding
            # 暂时使用此 hack,因为它尚不支持视频批处理。
            # TODO(ywang96): 修复内存剖析以考虑 EVS 并移除此 hack。
            if (
                (
                    self.is_multimodal_pruning_enabled
                    or self.requires_sequential_video_encoding
                )
                and modality == "video"
                and num_items > 1
            ):
                # 视频逐条顺序编码。
                batch_outputs_lst = list[torch.Tensor]()
                for video_idx in range(num_items):
                    # 取出单个视频条目。
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]
                    with self.timed_encoder_operation(
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1
                    ):
                        # 为单个视频构建微批输入。
                        _, _, micro_batch_mm_inputs = next(
                            group_and_batch_mm_kwargs(
                                [video_mm_kwargs_item],
                                device=self.device,
                                pin_memory=PIN_MEMORY,
                            )
                        )

                        # 逐个编码视频。
                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )

                        # 收集微批输出。
                        batch_outputs_lst.extend(micro_batch_outputs)

                batch_outputs = batch_outputs_lst
            else:
                # 运行编码器。
                # `batch_outputs` 是以下两者之一:
                # 1. 形状为 (num_items, feature_size, hidden_size) 的张量,
                # 适用于 feature_size 在所有多模态条目间固定的情形。
                # 2. 长度为 num_items 的列表或元组,每个元素形状为
                # (feature_size, hidden_size),适用于 feature_size 依赖于
                # 输入多模态条目的动态情形。

                with self.timed_encoder_operation(
                    should_time, mm_lora_refs, current_item_idx, num_items
                ):
                    # CUDA 图输出。
                    cudagraph_output = None
                    if (
                        self.encoder_cudagraph_manager is not None
                        and self.encoder_cudagraph_manager.supports_modality(modality)
                    ):
                        # 支持时用 CUDA 图执行编码。
                        cudagraph_output = self.encoder_cudagraph_manager.execute(
                            mm_kwargs_batch,
                        )

                    if cudagraph_output is not None:
                        # 使用 CUDA 图输出。
                        batch_outputs = cudagraph_output
                    else:
                        # 直接调用模型的嵌入方法。
                        batch_outputs = model.embed_multimodal(**mm_kwargs_batch)

            # 校验输出条目数。
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            # 收集编码输出。
            encoder_outputs.extend(batch_outputs)

            # 推进条目索引。
            current_item_idx += num_items

        # 按 mm_hash 缓存编码器输出
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self._cache_encoder_output(
                mm_hash,
                output,
                scheduler_output.ec_manager_metadata,
                scheduler_output.free_encoder_mm_hashes,
            )
            logger.debug("Finish execute for mm hash %s", mm_hash)

        # 返回编码器输出列表。
        return encoder_outputs

    # 从缓存中获取指定哈希的编码器输出。
    def _get_encoder_output_from_cache(self, mm_hash: str) -> torch.Tensor | None:
        """返回缓存中用于多模态嵌入收集的编码器输出。"""
        # 按哈希查询缓存(可能不存在)。
        return self.encoder_cache.get(mm_hash, None)

    # 收集本批次的多模态嵌入。
    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # 本步调度的 token 总数。
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # 多模态嵌入列表。
        mm_embeds = list[torch.Tensor]()
        # 标记每个 token 位置是否为多模态嵌入。
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens,
            dtype=torch.bool,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        # 当前请求在扁平输出中的起始索引。
        req_start_idx = 0
        # 是否需要同步 M-RoPE 位置。
        should_sync_mrope_positions = False
        # 是否需要同步 XD-RoPE 位置。
        should_sync_xdrope_positions = False

        # 遍历批次中的每个请求。
        for req_id in self.input_batch.req_ids:
            # 该请求的嵌入列表。
            mm_embeds_req: list[torch.Tensor] = []

            # 获取调度 token 数与请求状态。
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            # 取出多模态特征并确定当前窗口内的特征范围。
            mm_features = req_state.mm_features
            lo, hi = get_mm_features_in_window(
                mm_features,
                start=num_computed_tokens,
                end=num_computed_tokens + num_scheduled_tokens,
            )
            # 遍历窗口内的每个特征。
            for i in range(lo, hi):
                mm_feature = mm_features[i]
                pos_info = mm_feature.mm_position
                # 特征的起始偏移与编码器 token 数。
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # 计算本步涉及的编码器 token 区间。
                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                # 获取该区间内的嵌入索引。
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # 当前区间没有嵌入时跳过收集
                if curr_embeds_start == curr_embeds_end:
                    continue

                # 按哈希查询编码器输出。
                mm_hash = mm_feature.identifier
                encoder_output = self._get_encoder_output_from_cache(mm_hash)
                if encoder_output is None:
                    # 起始于已处理边界之后的特征仅经由 drafter 的 +1 前瞻到达,
                    # 可能尚未编码;起草时回退到 token 嵌入。
                    if (
                        start_pos
                        >= req_state.num_computed_tokens + num_scheduled_tokens
                    ):
                        continue
                    # 编码器缓存未命中,抛出错误。
                    raise RuntimeError(f"Encoder cache miss for {mm_hash}.")

                if (is_embed := pos_info.is_embed) is not None:
                    # 有嵌入标记时按嵌入区间切片。
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    # 无嵌入标记时直接按区间切片。
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                # 计算该特征在请求输出中的起始位置。
                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                # 对重叠的 mm_features(如 use_audio_in_video)取或掩码
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                        True
                    )
                else:
                    is_mm_embed[
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed
                # 设置嵌入的模态信息。
                set_mm_embedding_modality(mm_embeds_item, mm_feature.modality)
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                # 启用剪枝且使用 M-RoPE 时重算位置。
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                old_mm_embeds_req = mm_embeds_req
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                # 把模态信息复制到重算后的嵌入。
                mm_embeds_req = [
                    copy_mm_embedding_modality(src, dst)
                    for src, dst in zip(old_mm_embeds_req, mm_embeds_req)
                ]
                # 更新请求状态中的位置。
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            # 收集该请求的嵌入并推进起始索引。
            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        if should_sync_mrope_positions:
            # 重算并上传 M-RoPE 位置。
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            # 重算并上传 XD-RoPE 位置。
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        # 返回嵌入列表与嵌入标记掩码。
        return mm_embeds, is_mm_embed

    # 获取当前模型(未包装的原始模型)。
    def get_model(self) -> nn.Module:
        # 模型尚未初始化时报错。
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        if isinstance(
            self.model, (CUDAGraphWrapper, UBatchWrapper, BreakableCUDAGraphWrapper)
        ):
            # 从 CUDA 图包装器中取出原始模型。
            return self.model.unwrap()
        return self.model

    # 获取起草模型(drafter 内的模型)。
    def get_draft_model(self) -> nn.Module | None:
        drafter = getattr(self, "drafter", None)
        if drafter is None:
            return None
        model = getattr(drafter, "model", None)
        if isinstance(
            model, (CUDAGraphWrapper, UBatchWrapper, BreakableCUDAGraphWrapper)
        ):
            return cast(nn.Module, model.unwrap())
        return cast(nn.Module | None, model)

    # 获取支持的生成任务列表。
    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        # 获取原始模型。
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            # 文本生成模型支持 generate。
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                # 仅支持转写时直接返回。
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            # 支持实时任务。
            supported_tasks.append("realtime")

        # 返回支持的生成任务。
        return supported_tasks

    # 获取支持的池化任务列表。
    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        # 获取原始模型。
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        # 返回池化器支持的任务。
        return list(model.pooler.get_supported_tasks())

    # 获取支持的全部任务。
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            # 生成型 runner。
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            # 池化型 runner。
            tasks.extend(self.get_supported_pooling_tasks())

        # 返回任务元组。
        return tuple(tasks)

    # 同步并收集中间张量(流水线并行/序列并行场景)。
    def sync_and_gather_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        # 断言中间张量缓冲区已初始化。
        assert self.intermediate_tensors is not None

        # 张量并行大小。
        tp = self.vllm_config.parallel_config.tensor_parallel_size
        # 判断残差张量是否因序列并行而被分散。
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # 启用序列并行时,"residual" 张量在 TP 秩间分片。这里进行 all-gather,
        # 因为下游 QKV + Attention 需要在 SP 切分点之前拿到完整的 residual。
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                # 判断该张量是否为被分散的 residual。
                is_scattered = k == "residual" and is_rs
                if is_scattered:
                    # 先 all-gather 本地分片得到完整残差。
                    local_len = num_tokens // tp
                    v = get_tp_group().all_gather(v[:local_len], dim=0)

                # 异步拷贝到预分配的中间张量缓冲区。
                self.intermediate_tensors[k][:num_tokens].copy_(
                    v[:num_tokens], non_blocking=True
                )

        # 返回切片后的中间张量视图。
        return IntermediateTensors(
            {k: v[:num_tokens] for k, v in self.intermediate_tensors.items()}
        )

    # 推进 EPLB(专家并行负载均衡)状态。
    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        EPLB(专家并行负载均衡)状态的单步推进。
        """
        # 未启用 EPLB 或被抑制时直接返回。
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        # 断言状态与 MoE 模型已就绪。
        assert self.eplb_state is not None
        assert self._moe_model is not None
        # 推进 EPLB 状态。
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    # 依据物理-逻辑专家映射重建 EPLB 状态。
    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        # 断言 MoE 模型已就绪。
        assert self._moe_model is not None

        # 从映射构建新的 EPLB 状态。
        self.eplb_state = EplbState.from_mapping(
            model=self._moe_model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )

    # 执行池化模型并构造输出。
    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # 持久批次中的请求数。
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        # 截取本步调度的隐藏状态。
        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]

        # 获取池化元数据并构建池化游标。
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu,
            device=hidden_states.device,
            query_start_loc_gpu=self.query_start_loc.gpu[: num_reqs + 1],
        )

        # 调用模型的池化器。
        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        # 标记序列长度已达 prompt 长度(即已完成)的请求。
        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]
        # 对池化输出做 late_interaction 后处理。
        raw_pooler_output = self.late_interaction_runner.postprocess_pooler_output(
            raw_pooler_output=raw_pooler_output,
            pooling_params=pooling_metadata.pooling_params,
            req_ids=self.input_batch.req_ids,
            finished_mask=finished_mask,
        )

        # 构造基础的 ModelRunnerOutput。
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        # 无原始池化输出或没有已完成的请求时,同步设备并返回空池化输出。
        if raw_pooler_output is None or not any(finished_mask):
            self._sync_device()
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        if not current_platform.is_cuda_alike():
            # cpu/xpu runner 不能使用基于 CUDA 流/事件的包装。
            model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                raw_pooler_output=raw_pooler_output,
                finished_mask=finished_mask,
            )
            self._sync_device()
            return model_runner_output

        # 返回异步 GPU 池化输出包装。
        return AsyncGPUPoolingModelRunnerOutput(
            model_runner_output=model_runner_output,
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
            async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
        )

    # 序列并行启用时,把 token 数向上填充到张量并行大小的整数倍。
    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        # 为 SP 启用集合通信融合时,把 token 数填充为 tensor_parallel_size 的倍数
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config.enable_sp and tp_size > 1:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    # 准备多模态模型的输入张量。
    def _prepare_mm_inputs(
        self, num_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.model.requires_raw_input_tokens:
            # 需要原始输入 token 时提供 input_ids。
            input_ids = self.input_ids.gpu[:num_tokens]
        else:
            input_ids = None

        # 返回输入 ID 与嵌入。
        inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
        return input_ids, inputs_embeds

    # 模型前向前的预处理:准备输入 ID/嵌入、位置与中间张量。
    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # 已填充
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        # 本步调度的 token 总数。
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # 是否为流水线并行的第一个 rank。
        is_first_rank = get_pp_group().is_first_rank
        # 是否为编码器-解码器模型。
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # 在嵌入查找前截断投机调度器的占位符(-1)。
        if self.speculative_config is not None:
            self.input_ids.gpu[:num_input_tokens].clamp_(min=0)

        # _prepare_inputs 可能重排批次,因此必须在之后收集多模态输出,
        # 以保证顺序正确
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # 若有多模态编码器则运行之。
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                # 收集多模态嵌入。
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # 注意(woosuk): 为统一 token ID 与软 token(视觉嵌入),
            # 即使输入是文本,我们始终用嵌入(而非 token ID)作为多模态模型的输入。
            if self.enable_prompt_embeds and self.input_batch.req_prompt_embeds:
                # 某些位置带有预计算的 prompt_embeds:它们已在 self.inputs_embeds
                # 中,并标记为 is_token_ids=False。仅对 token ID 位置做嵌入
                # (把 prompt_embeds 位置的占位 ID 置 0,使嵌入收集不会读到
                # 越界 ID),写回时不覆盖 prompt_embeds 位置。
                is_token_ids = self.is_token_ids.gpu[:num_scheduled_tokens]
                safe_input_ids = torch.where(
                    is_token_ids,
                    self.input_ids.gpu[:num_scheduled_tokens],
                    0,
                )
                # 对 token ID 位置计算嵌入。
                inputs_embeds_scheduled = self.model.embed_input_ids(
                    safe_input_ids,
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
                # 只写回 token ID 位置,保留 prompt_embeds 位置。
                target = self.inputs_embeds.gpu[:num_scheduled_tokens]
                self.inputs_embeds.gpu[:num_scheduled_tokens] = torch.where(
                    is_token_ids.unsqueeze(-1),
                    inputs_embeds_scheduled,
                    target,
                )
            else:
                # 对全部位置计算嵌入。
                inputs_embeds_scheduled = self.model.embed_input_ids(
                    self.input_ids.gpu[:num_scheduled_tokens],
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )

                # TODO(woosuk): 避免该拷贝。待优化。
                self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
                    inputs_embeds_scheduled
                )

            # 准备多模态输入与模型 kwargs。
            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # 获取非输入嵌入 token 的输入嵌入,
            # 然后放入合适的位置。
            # TODO(qthequartermasterman): 即使启用了 prompt embeds,(a)并非所有
            # 请求都会使用 prompt embeds,(b)初始 prompt 处理完后,其余生成的
            # token 仍是 token ID,因此让嵌入层始终位于 CUDA 图之外并不理想。
            # v0 引擎通过对 CUDA 图"双重编译"来避免这一点:先用 input_ids 再用
            # inputs_embeds 编译全部 num_tokens。若批次只有 token ID,把嵌入层
            # 纳入 CUDA 图会更有性能优势(如下面的 else 分支)。
            is_token_ids = self.is_token_ids.np[:num_scheduled_tokens]
            # 找出需要转为嵌入的 token ID 位置。
            token_ids_idx_np = np.nonzero(is_token_ids)[0]
            # 部分 token ID 可能需要转为嵌入
            if token_ids_idx_np.size > 0:
                token_ids_idx = async_tensor_h2d(token_ids_idx_np, device=self.device)
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = None
        else:
            # 对纯文本模型,使用 token ID 作为输入。
            # 虽然可以像多模态模型那样用嵌入作为输入,但出于性能考虑并不理想,
            # 因为那样嵌入层不会包含在 CUDA 图中。
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        # 依据模型类型选择位置张量。
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
            if num_input_tokens > num_scheduled_tokens:
                # 清零填充位置的 position。
                self.positions[num_scheduled_tokens:num_input_tokens].zero_()

        if is_first_rank:
            # 第一个 rank 无需中间张量。
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            # 同步并收集来自上一 rank 的中间张量。
            intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # 运行编码器,与其他多模态输入的处理方式类似。
            # 对编码器-解码器模型,这里的处理更简单,因为输出直接传给解码器。
            # 我们不做任何 prompt 替换。也只会有单个编码器输入。
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            # 把编码器输出写入模型 kwargs。
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        # 返回预处理结果元组。
        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    # 执行采样并返回 SamplerOutput。
    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # 采样下一个 token 并按需获取 logprobs。
        sampling_metadata = self.input_batch.sampling_metadata
        # 异步调度且当前采样参数需要时,
        # 用上一步采样的 token 更新输出 token ID。
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            # 无投机解码时直接调用采样器。
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # 仅在需要 output_token_ids(使用 penalties 或 bad_words)时,
        # 用上一步的真实草稿 token 更新 spec_token_ids。
        if self.use_async_scheduling and self._draft_token_req_ids is not None:
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(draft_token_ids_cpu)

        # 获取草稿概率并用拒绝采样器采样。
        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output

    # 同步模式下的采样后簿记(更新缓存、计算 logprobs 等)。
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        # 统计 logits 中的 NaN 数量(可选)。
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # 持久批次请求数。
        num_reqs = self.input_batch.num_reqs
        # 找出应丢弃采样 token 的请求索引。
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            # 回退对应随机生成器的偏移,保证可复现性。
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # 拷贝部分对象,避免返回后被修改。
        # 使用异步调度时这一点很重要。
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        # 采样的 token 数。
        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        if not self.use_async_scheduling:
            # 同步调度:在下方 ``_to_list`` 之前,把路由专家的 D2H 结果放入
            # 锁页 CPU 缓冲区。``_to_list`` 会在异步拷贝流上做
            # ``event.synchronize()``,它等待自上次同步以来在默认流上排队的所有
            # D2H,因此这次入队自然被覆盖,无需单独同步。
            if self.routed_experts_initialized:
                buf = self.routed_experts_capturer.get_device_buffer()
                total = scheduler_output.total_num_scheduled_tokens
                self.routed_experts_cpu[:total].copy_(buf[:total], non_blocking=True)
                self.routed_experts_slot_mapping_cpu[:total].copy_(
                    self.routed_experts_slot_mapping_device[:total],
                    non_blocking=True,
                )

            # 获取有效生成的 token。
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # 无投机解码 token。
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # 屏蔽不应被采样的 token。
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            else:
                # 包含投机解码 token。
                valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    logprobs_tensors=logprobs_tensors,
                )
        else:
            # 异步调度:不在本步取回 token。
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # 把采样 token 缓存在 GPU 上,避免 CPU 同步。
            # 它们会在下一步准备输入时被拷贝进 input_ids。
            # 使用投机解码时,这一步在 propose_draft_token_ids() 中完成。
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # 在模型 runner 中缓存采样 token,使调度器无需把它们回传。
        # 注意(woosuk): 例外是使用 PP 时,调度器会把采样 token 回传,因为
        # 第一阶段 worker 与最后阶段 worker 之间没有直接通信。
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                # 异步调度用 -1 占位。
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            # 计算写入区间并断言不超过最大模型长度。
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            # 写入 token_ids_cpu 并更新标记与长度。
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            # 追加到请求状态的输出 token 列表。
            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # 按需计算 prompt logprobs。
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        # 返回簿记结果元组。
        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    # 上下文管理器:确保输入准备与上一步的 CPU 张量复用同步。
    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # 确保上一步已完成对复用 CPU 张量的使用。
        # 异步调度下必须如此,因为 CPU->GPU 传输是异步的。
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            # 在结束时记录事件。
            self.prepare_inputs_event.record()

    # 调用模型前向的辅助方法。
    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """调用模型前向传播的辅助方法。

        子类可覆盖本方法以执行模型。
        动机:可以只检查本方法,而不必检查带有额外逻辑的整个 execute_model。

        Args:
            input_ids: 输入 token ID
            positions: token 位置
            intermediate_tensors: 来自前序流水线阶段的张量
            inputs_embeds: 输入嵌入(input_ids 的替代)
            **model_kwargs: 额外的模型参数

        Returns:
            模型输出张量
        """
        # 调用模型。
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # 判断是否为所有请求调度数量相同的均匀 decode 批次。
    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        检查是否为所有请求调度 token 数相同的 decode 批次。
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    # 确定批次执行方式(CUDA 图模式)与填充数量。
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool,
        allow_microbatching: bool = True,
        force_eager: bool = False,
        # 供 CUDA 图捕获使用 TODO(lucas): 重构 CUDA 图捕获方式
        # (将在 model runner v2 中改进)
        force_uniform_decode: bool | None = None,
        force_has_lora: bool | None = None,
        force_num_active_loras: int | None = None,
        num_encoder_reqs: int = 0,
    ) -> tuple[
        CUDAGraphMode,
        BatchDescriptor,
        bool,
        torch.Tensor | None,
        CUDAGraphStat | None,
    ]:
        # 判断是否为均匀 decode 批次。
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
        )
        # 编码器-解码器模型仅支持 decoder_step > 0 时的 CUDA 图(不存在
        # enc_output)。此外分块 prefill 被禁用,因此批次是均匀的。
        has_encoder_output = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # 为 CUDA 图分派计算 LoRA 状态。
        num_active_loras = (
            force_num_active_loras
            if force_num_active_loras is not None
            else len(self.input_batch.lora_id_to_lora_request)
        )
        has_lora = num_active_loras > 0 if force_has_lora is None else force_has_lora

        # 按序列并行要求填充 token 数。
        num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)

        # 内部函数:分派 CUDA 图模式。
        def dispatch_cudagraph(num_tokens, disable_full=False, valid_modes=None):
            return self.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                has_lora=has_lora,
                uniform_decode=uniform_decode,
                num_active_loras=num_active_loras,
                valid_modes={CUDAGraphMode.NONE} if force_eager else valid_modes,
                invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
            )

        # 首次分派 CUDA 图模式。
        cudagraph_mode, batch_descriptor = dispatch_cudagraph(
            num_tokens_padded, disable_full=use_cascade_attn or has_encoder_output
        )
        num_tokens_padded = batch_descriptor.num_tokens
        if self.compilation_config.pass_config.enable_sp:
            assert (
                batch_descriptor.num_tokens
                % self.vllm_config.parallel_config.tensor_parallel_size
                == 0
            ), (
                "Sequence parallelism requires num_tokens to be "
                "a multiple of tensor parallel size"
            )

        # 数据并行运行时的额外协调,因为需要跨秩协同。
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.parallel_config,
                    allow_microbatching=allow_microbatching,
                    num_tokens_padded=num_tokens_padded,
                    uniform_decode=uniform_decode,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )

            # 提取 DP 同步后的值。
            if num_tokens_across_dp is not None:
                dp_rank = self.parallel_config.data_parallel_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # 用 DP 填充重新分派,以获得正确的 batch_descriptor。
                cudagraph_mode, batch_descriptor = dispatch_cudagraph(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # 断言以确保约定的 token 数正确,否则
                # num_tokens_across_dp 将失效。
                assert batch_descriptor.num_tokens == num_tokens_padded

        # 可选的 CUDA 图统计信息。
        cudagraph_stats = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            cudagraph_stats = CUDAGraphStat(
                num_unpadded_tokens=num_tokens,
                num_padded_tokens=batch_descriptor.num_tokens,
                num_paddings=batch_descriptor.num_tokens - num_tokens,
                runtime_mode=str(cudagraph_mode),
            )

        # 返回分派结果元组。
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    # 注册逐层 NVTX 钩子(若启用逐层 NVTX 追踪)。
    def _register_layerwise_nvtx_hooks(self) -> None:
        """
        当启用 --enable-layerwise-nvtx-tracing 时注册逐层 NVTX 钩子,
        以追踪模型中每一层或每个模块的详细信息。
        """

        if (
            self.vllm_config.observability_config.enable_layerwise_nvtx_tracing
            and not self.layerwise_nvtx_hooks_registered
        ):
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when CUDA graph is "
                    "turned off; you may observe part or all of the model "
                    "missing NVTX markers"
                )

            # 在 STOCK_TORCH_COMPILE 模式下,在这里注册钩子后,nn.Module 的
            # __call__ 函数会以 fullgraph=True 被重新编译。由于 nvtx.range_push/pop
            # 无法被 torch dynamo 追踪,我们不能在这里注册钩子函数,
            # 因为钩子函数也会被 torch dynamo 追踪。
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                logger.debug_once(
                    "layerwise NVTX tracing is not supported when "
                    "CompilationMode is STOCK_TORCH_COMPILE, skipping "
                    "function hooks registration"
                )
            else:
                # 注册 PyTorch 钩子并标记已完成。
                pyt_hooks = PytHooks()
                pyt_hooks.register_hooks(self.model, self.model.__class__.__name__)
                self.layerwise_nvtx_hooks_registered = True

    # 以两种格式构建槽位映射。
    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """
        构建系统所需的两种格式的槽位映射。

        Args:
            num_tokens_padded: token 总数(已填充)
            num_reqs_padded: 请求总数(已填充)
            num_tokens_unpadded: 实际 token 数(未填充)
            ubatch_slices: DBO 使用的可选微批切片信息

        Returns:
            元组:
            - slot_mappings_by_gid: 供注意力元数据使用的 dict[int, torch.Tensor]
            - slot_mappings_by_layer: 供 ForwardContext 使用的 dict[str, torch.Tensor]
              或列表
        """
        # 无 KV cache 配置时返回空。
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        # 内部函数:获取指定 KV 组的槽位映射。
        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                # 仅编码器注意力使用全零槽位映射。
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                # 获取块表的 GPU 槽位映射。
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # 用 -1 填充未使用部分。完整 CUDA 图模式下的 reshape_and_cache 需要它。
            # `blk_table_tensor` 取 -1 以匹配 mamba 的 PAD_SLOT_ID
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        # 为每个 KV 组构建槽位映射。
        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        # 把组级映射展开为层级映射。
        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        if ubatch_slices is not None:
            # 微批切片时按 token 区间切片。
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        # 返回组级与层级两种映射。
        return slot_mappings_by_gid, slot_mappings_by_layer

    # 检查是否所有已调度请求都被标记为丢弃采样 token。
    def _is_all_reqs_chunked_prefill(self) -> bool:
        """检查是否所有已调度请求都被标记为丢弃采样 token。

        当每个已调度请求的 `discard_request_mask` 都被置位时返回 True
        (例如,非最后一个 prefill 块的分块 prefill 请求)。"""
        num_reqs = self.input_batch.num_reqs
        return bool(self.discard_request_mask.np[:num_reqs].all())

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(scheduler_output)

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            logger.debug(
                "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                "should_ubatch: %s, num_tokens_across_dp: %s",
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            logger.debug(
                "ubatch_slices: %s, ubatch_slices_padded: %s",
                ubatch_slices,
                ubatch_slices_padded,
            )

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            has_separate_kv_update = not all(
                all(
                    g.backend.forward_includes_kv_cache_update
                    for g in self.attn_groups[id]
                )
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )
            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            if self.cache_config.mamba_cache_mode == "align":
                # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                # to decide copy operations, so we must apply deferred
                # corrections before it runs.
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                mamba_bufs = self._get_mamba_bufs()
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    mamba_bufs.preprocess,
                    align_ctx=mamba_bufs.postprocess_align,
                )
                # preprocess_mamba resets num_accepted_tokens_cpu to 1
                # for requests whose state was copied to a new block.
                # Re-sync to GPU so the mamba kernel reads from the
                # correct initial state slot (init_token_idx = 0).
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)

                # Stage per-request inputs for the fused postprocess kernel
                # only when that kernel will actually run. The kernel is
                # gated on spec-decode + hybrid (see MambaBuffers.create);
                # without it, ``mamba_bufs.postprocess_align`` is None and
                # the staging buffers don't exist.
                if mamba_bufs.postprocess_align is not None:
                    mamba_utils.stage_postprocess_inputs_to_gpu(
                        mamba_bufs.postprocess_align,
                        scheduler_output,
                        self.input_batch.req_ids,
                        num_reqs,
                        self.requests,
                        self.mamba_state_idx,
                    )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded,
                num_reqs_padded=(
                    num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
                ),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    slot_mappings=slot_mappings_by_group,
                )
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = (
            self.model_config.is_encoder_decoder and num_encoder_reqs > 0
        )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None
        # Update the EPLB meta.
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(
                self.model_config,
                num_tokens_unpadded,
                ubatch_slices_padded,
            )
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,
                skip_compiled=has_encoder_input,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        return None

    def _input_fits_in_drafter(
        self, common_attn_metadata: CommonAttentionMetadata | None
    ) -> bool:
        if common_attn_metadata is None:
            return False
        assert self.speculative_config is not None
        # DFlash queries one extra token (the bonus token) beyond num_spec_tokens
        num_drafter_query_tokens = self.num_spec_tokens + (
            1 if self.speculative_config.use_dflash() else 0
        )
        return (
            common_attn_metadata.max_seq_len + num_drafter_query_tokens
            <= self.effective_drafter_max_model_len
        )

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        if self.use_async_scheduling:
            pp = get_pp_group()
            # For torchrun external_launcher PP mode with broadcast_pp_output=True,
            # PP outputs have been broadcasted to all ranks at logits computation.
            # Therefore, here is no need to send sampled token ids again in this case.
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(
                    sampler_output.sampled_token_ids
                )

        self._draft_token_ids = None
        self._draft_probs = None
        self._draft_prob_req_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        draft_after_bookkeeping = False
        if spec_config is not None:
            # Decide whether to run the drafter or zero out draft tokens.
            input_fits_in_drafter = self._input_fits_in_drafter(
                spec_decode_common_attn_metadata
            )
            # Whether the drafter runs a GPU model forward (and thus carries
            # TP/EP/DP collectives), independent of padded-batch timing.
            drafter_runs_model_forward = (
                spec_config.use_eagle()
                or spec_config.uses_draft_model()
                or spec_config.uses_extract_hidden_states()
            )
            use_gpu_toks = (
                drafter_runs_model_forward
                and not spec_config.disable_padded_drafter_batch
            )
            if use_gpu_toks:
                # EAGLE/DraftModel speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                else:
                    if self.valid_sampled_token_count_event is not None:
                        assert spec_decode_common_attn_metadata is not None
                        next_token_ids, valid_sampled_tokens_count = (
                            self.drafter.prepare_next_token_ids_padded(
                                sampled_token_ids,
                                self.requests,
                                self.input_batch,
                                self.discard_request_mask.gpu,
                            )
                        )
                        self._copy_valid_sampled_token_count(
                            next_token_ids, valid_sampled_tokens_count
                        )
                    if self.parallel_config.data_parallel_size > 1:
                        # Prevent hang when DP ranks disagree on input_fits_in_drafter
                        self.drafter.dummy_run(num_tokens=1)
            elif (
                spec_config.use_ngram_gpu()
                and not spec_config.disable_padded_drafter_batch
            ):
                assert isinstance(self.drafter, NgramProposerGPU)
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count, _ = (
                        self.drafter.update_token_ids_ngram(
                            sampled_token_ids,
                            self.input_batch,
                            self.token_ids_gpu_tensor,
                            self.num_tokens_no_spec_gpu,
                            self.discard_request_mask.gpu,
                        )
                    )
                    self._copy_valid_sampled_token_count(
                        next_token_ids, valid_sampled_tokens_count
                    )
            else:
                # These drafters consume CPU sampled tokens, so they run
                # after bookkeeping.
                draft_after_bookkeeping = True

            if not input_fits_in_drafter:
                # Zero out draft tokens so the scheduler doesn't schedule
                # stale drafts from the previous step.
                # For Nemotron-H: it is necessary to zero out the draft tokens,
                # otherwise the stale tokens will corrupt Mamba recurrent
                # state and logprobs for sequences near max_model_len.
                self._draft_token_ids = torch.zeros(
                    1, device=self.device, dtype=torch.int32
                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                self._draft_probs = None
                self._draft_prob_req_ids = None
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
            )

        if draft_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            if input_fits_in_drafter:
                propose_draft_token_ids(valid_sampled_token_ids)
            elif (
                drafter_runs_model_forward
                and self.parallel_config.data_parallel_size > 1
            ):
                # Prevent hang when DP ranks disagree on input_fits_in_drafter
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                self.drafter.dummy_run(num_tokens=1)

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if spec_config is not None:
            self.finalize_kv_connector()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # self.kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts=None,
            )

        if not self.use_async_scheduling:
            if self.routed_experts_initialized:
                # Sync path: D2H was issued in ``_bookkeeping_sync`` and
                # synchronized by ``_to_list``'s event.synchronize(), so
                # the pinned buffers are ready to be wrapped as numpy.
                total = scheduler_output.total_num_scheduled_tokens
                output.routed_experts = RoutedExpertsLists(
                    routing_data=self.routed_experts_cpu[:total].numpy(),
                    slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
                )
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            # Async path: produce a device-side snapshot that the async
            # copy stream can D2H later. Both tensors must be private
            # clones because:
            #   - ``routing_data`` source is the shared capturer buffer,
            #     which the next forward overwrites on the default stream.
            #   - ``slot_mapping`` source is our own
            #     ``routed_experts_slot_mapping_device``, which the
            #     next ``_prepare_inputs`` overwrites on the default
            #     stream while the D2H is still pending on the copy
            #     stream.
            # Without clones, the copy stream would read torn data.
            routed_experts_snapshot = self.get_routed_experts(
                scheduler_output.total_num_scheduled_tokens
            )

            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
                vocab_size=self.input_batch.vocab_size,
                routed_experts=routed_experts_snapshot,
                check_ep_fault=self.check_ep_fault,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """Broadcast sampled token ids (GPU) from last PP stage"""
        pp = get_pp_group()
        assert pp.is_last_rank
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        # Skip for chunked prefill: sampled tokens are dummy
        # and will be discarded, no need to broadcast.
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(
                sampled_token_ids, src=pp.rank, group=pp.device_group
            )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """Receive sampled token ids broadcast from last PP stage"""
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        # skip for chunked prefill.
        if not self._is_all_reqs_chunked_prefill():
            torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder (-1) token id.
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
            pos = self.input_batch.num_tokens_no_spec[i]
            self.input_batch.is_token_ids[i, pos] = True
            self.input_batch.num_tokens_no_spec[i] = pos + 1
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        if torch.is_tensor(self._draft_token_ids):
            assert isinstance(self._draft_token_ids, torch.Tensor)
            self.prev_num_spec_tokens = self._draft_token_ids.shape[1]
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.cuda.current_stream()
        num_reqs = draft_token_ids.shape[0]
        num_spec_tokens = draft_token_ids.shape[1]
        with torch.cuda.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        assert isinstance(self._draft_token_ids, torch.Tensor)
        num_spec_tokens = self._draft_token_ids.shape[1]
        return self.draft_token_ids_cpu[
            : len(req_ids), :num_spec_tokens
        ].tolist(), req_ids

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        if self.use_async_spec_decode:
            # Stash for GPU-side correction in _prepare_inputs.
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def _get_spec_decode_draft_probs(
        self, spec_decode_metadata: SpecDecodeMetadata
    ) -> torch.Tensor | None:
        if self._draft_probs is None or self._draft_prob_req_ids is None:
            return None

        row_by_req_id = {
            req_id: idx for idx, req_id in enumerate(self._draft_prob_req_ids)
        }
        draft_probs_rows: list[torch.Tensor] = []
        for req_id, num_draft in zip(
            self.input_batch.req_ids, spec_decode_metadata.num_draft_tokens
        ):
            if num_draft == 0:
                continue
            row_idx = row_by_req_id.get(req_id)
            if row_idx is None:
                logger.warning(
                    "Missing cached draft probabilities for request %s; "
                    "falling back to legacy speculative rejection behavior.",
                    req_id,
                )
                return None
            draft_probs_rows.append(self._draft_probs[row_idx, :num_draft])

        if not draft_probs_rows:
            return None
        return torch.cat(draft_probs_rows, dim=0).contiguous()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        num_spec_tokens_to_schedule = scheduler_output.num_spec_tokens_to_schedule
        self._draft_probs = None
        self._draft_prob_req_ids = None
        if spec_config.method == "ngram":
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                num_spec_tokens_to_schedule,
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        elif spec_config.method == "custom_class":
            assert isinstance(sampled_token_ids, list)
            draft_token_ids = cast(Any, self.drafter).propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        elif spec_config.use_ngram_gpu():
            assert isinstance(self.drafter, NgramProposerGPU)
            (
                next_token_ids,
                valid_sampled_tokens_count,
                valid_sampled_token_ids_gpu,
            ) = self.drafter.update_token_ids_ngram(
                sampled_token_ids,
                self.input_batch,
                self.token_ids_gpu_tensor,
                self.num_tokens_no_spec_gpu,
                self.discard_request_mask.gpu,
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

            batch_size = next_token_ids.shape[0]

            draft_token_ids, num_valid_draft_tokens = self.drafter.propose(
                num_spec_tokens_to_schedule,
                self.num_tokens_no_spec_gpu[:batch_size],
                self.token_ids_gpu_tensor[:batch_size],
                valid_sampled_token_ids_gpu,
                valid_sampled_tokens_count,
            )

            # Cache valid draft counts for scheduler-side trimming.
            self._num_valid_draft_tokens = num_valid_draft_tokens

            # Async D2H copy on a dedicated stream.
            copy_num_valid_draft_tokens(
                self._num_valid_draft_tokens_cpu,
                self._num_valid_draft_tokens_copy_stream,
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens,
                self.input_batch.num_reqs,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(
                num_spec_tokens_to_schedule,
                self.input_batch,
                sampled_token_ids,
                slot_mappings=slot_mappings,
            )
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = async_tensor_h2d(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                num_speculative_tokens=num_spec_tokens_to_schedule,
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
                slot_mappings=slot_mappings,
            )
        elif spec_config.uses_extract_hidden_states():
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            assert isinstance(sampled_token_ids, torch.Tensor), (
                "sampled_token_ids should be a torch.Tensor for "
                "extract_hidden_states method."
            )
            if not self.use_aux_hidden_state_outputs or aux_hidden_states is None:
                raise ValueError(
                    "aux_hidden_states are required when using `extract_hidden_states`"
                )
            target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]

            draft_token_ids = self.drafter.propose(
                num_speculative_tokens=num_spec_tokens_to_schedule,
                sampled_token_ids=sampled_token_ids,
                target_hidden_states=target_hidden_states,
                common_attn_metadata=common_attn_metadata,
                slot_mappings=slot_mappings,
            )
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

        elif (
            spec_config.use_eagle()
            or spec_config.use_dflash()
            or spec_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer | DFlashProposer | DraftModelProposer | Gemma4Proposer,
            )

            if spec_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). Safe to
            # rebind here: hidden_states was already consumed for sampling
            # above and is not used again in this branch.
            alt = getattr(
                self.get_model(), "get_mtp_target_hidden_states", lambda: None
            )()
            if alt is not None:
                hidden_states = alt

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = self._get_positions(token_indices)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    # When padding the batch, token_indices is just a range
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = self._get_positions(total_num_tokens)
                    if self.use_aux_hidden_state_outputs:
                        assert aux_hidden_states is not None
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
                num_speculative_tokens=num_spec_tokens_to_schedule,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )
            if hasattr(self.drafter, "take_last_draft_probs"):
                draft_probs = self.drafter.take_last_draft_probs()
                if draft_probs is not None:
                    self._draft_probs = draft_probs
                    self._draft_prob_req_ids = self.input_batch.req_ids.copy()

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            if config_name not in allowed_config_names:
                allowed = ", ".join(sorted(allowed_config_names))
                raise ValueError(
                    f"Config override '{config_name}' is not supported. "
                    f"Supported configs: {allowed}"
                )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    @instrument(span_name="Loading (GPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        """
        Args:
            load_dummy_weights: load dummy weights instead of real weights.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0

        try:
            with DeviceMemoryProfiler() as m:
                time_before_load = time.perf_counter()
                if load_dummy_weights:
                    self.load_config.load_format = "dummy"
                model_loader = get_model_loader(self.load_config)
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    if hasattr(self.drafter, "load_model"):
                        self.drafter.load_model(self.model)
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        assert not self.parallel_config.enable_elastic_ep, (
                            "Elastic EP is not supported with drafter model."
                        )
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                        )
                        assert hasattr(self.drafter, "set_eplb_state")
                        self.drafter.set_eplb_state(self.eplb_state)
                        eplb_models += 1

                self._setup_eagle3_aux_hidden_state_outputs()

                # Resolve the MoE model, unwrapping VLM wrappers if needed.
                # VLM models (e.g. KimiK25ForConditionalGeneration) wrap the
                # actual MoE language model but don't implement
                # MixtureOfExperts themselves.
                moe_candidate = self.model
                if not is_mixture_of_experts(moe_candidate) and isinstance(
                    moe_candidate, SupportsMultiModal
                ):
                    moe_candidate = moe_candidate.get_language_model()
                if is_mixture_of_experts(moe_candidate):
                    self._moe_model = moe_candidate

                if (
                    self._moe_model is not None
                    and self.parallel_config.enable_eplb
                    and not load_dummy_weights
                ):
                    logger.info_once(
                        "EPLB is enabled for model %s.",
                        self.model_config.model,
                    )
                    assert self.eplb_state is not None
                    self.eplb_state.add_model(
                        self._moe_model,
                        self.model_config,
                    )
                    eplb_models += 1

                time_after_load = time.perf_counter()
            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            combined_msg = f"{msg} (original error: {e})"
            logger.error(combined_msg)
            raise e
        logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
        )

        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )
        self.requires_sequential_video_encoding = hasattr(
            self.get_model(), "requires_sequential_video_encoding"
        )  # Temporary hack for dynamic res video w/o support for bs>1 yet

        if (
            self._moe_model is not None
            and self.parallel_config.enable_eplb
            and not load_dummy_weights
            and self.eplb_state is not None
            and self.eplb_state.is_async
        ):
            self.eplb_state.start_async_loop()

        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
        ):
            from vllm.env_override import _apply_constrain_to_fx_strides_patch

            _apply_constrain_to_fx_strides_patch()
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation modes, cudagraph behavior is controlled by
        # CudagraphWrapper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        if (
            is_breakable_cudagraph_enabled()
            and cudagraph_mode != CUDAGraphMode.NONE
            and not self.parallel_config.use_ubatching
        ):
            self.model = BreakableCUDAGraphWrapper(self.model, self.vllm_config)
            drafter = getattr(self, "drafter", None)
            if drafter is not None and hasattr(drafter, "model"):
                drafter.model = BreakableCUDAGraphWrapper(
                    drafter.model, self.vllm_config
                )
        elif (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

        get_offloader().post_init()

    def _setup_eagle3_aux_hidden_state_outputs(self) -> None:
        if not self.use_aux_hidden_state_outputs:
            return

        if not supports_eagle3(self.get_model()):
            raise RuntimeError(
                "Model does not support EAGLE3 interface but "
                "aux_hidden_state_outputs was requested"
            )
        # Try to get auxiliary layers from speculative config,
        # otherwise use model's default layers
        aux_layers = self._get_eagle3_aux_layers_from_config()
        if aux_layers:
            logger.info(
                "Using auxiliary layers from speculative config: %s", aux_layers
            )
        else:
            aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()

        self.model.set_aux_hidden_state_layers(aux_layers)

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config

        layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if not layer_ids:
            dflash_config = getattr(hf_config, "dflash_config", None)
            eagle_config = getattr(hf_config, "eagle_config", None)

            if dflash_config and isinstance(dflash_config, dict):
                # Add 1 to convert DFlash's aux layer id semantics
                layer_ids = [
                    i + 1 for i in (dflash_config.get("target_layer_ids") or [])
                ]

            if eagle_config and isinstance(eagle_config, dict):
                layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")

        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """
        Reload weights from a weights iterator or from disk

        Args:
            weights_iterator: weights to load into model
            weights_path: path to load weights from if weights_iterator is not
                provided. Use path of original model if neither is provided.
            is_checkpoint_format: set to False if weights have already been
                processed into kernel format (repacking, renaming, etc.)
        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        model = self.get_model()
        weights_to_load = {
            name.replace(".base_layer.", ".") if self.lora_config else name
            for name, _ in model.named_parameters()
        }
        counter_before_reloading = time.perf_counter()

        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            if weights_path is not None:
                # The revision belongs to the model we are reloading away from,
                # so it must not be carried over to the new path.
                self.model_config.model = weights_path
                self.model_config.revision = None
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # begin loading weights
        logger.info_once("Reloading weights inplace...")
        if is_checkpoint_format:
            # load weights from checkpoint/ original model format
            initialize_layerwise_reload(model)
            loaded_weights = model.load_weights(weights_iterator)
            finalize_layerwise_reload(model, self.model_config)

        else:
            # load weights from kernel format
            logger.warning_once(
                "Reloading with `is_checkpoint_format=True` requires that "
                "weights be in kernel format and already sharded",
            )
            loaded_weights = set()
            for name, loaded_weight in weights_iterator:
                param = _get_parameter_for_reload(model, name)  # TODO: buffers?
                param.copy_(loaded_weight)
                loaded_weights.add(name)

        self.reset_lora_state()

        # logging and validation
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
        )
        if self.model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

        self.reset_encoder_cache()
        self.reset_mm_cache()

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = async_tensor_h2d(
                request.prompt_token_ids, device=self.device
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = request.in_progress_prompt_logprobs_cpu
            if logprobs_tensors is None:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                request.in_progress_prompt_logprobs_cpu = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt scores respecting logprobs_mode.
            # NOTE: prompt tokens skip sampling processors, so
            # processed_* and raw_* yield the same scores here.
            if self.model_config.logprobs_mode in ("raw_logits", "processed_logits"):
                scores = logits.to(torch.float32)
            else:
                scores = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                scores, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(num_nans_in_logits)
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        # Don't use `max_items_per_batch` here to avoid redundant computation
        dummy_mm_inputs = self.mm_registry.get_dummy_mm_inputs(
            self.model_config,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]

        # We use the cache so that the item is saved to the cache,
        # but not read from the cache
        assert dummy_mm_item is not None, "Item should not already be cached"

        return next(
            mm_kwargs_batch
            for _, _, mm_kwargs_batch in group_and_batch_mm_kwargs(
                [(modality, dummy_mm_item)] * max_items_per_batch,
                device=self.device,
                pin_memory=PIN_MEMORY,
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
            profile_seq_lens: If provided, use this value for seq_lens instead
                of max_query_len. Used to profile attention workspace that
                scales with context length.
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.is_valid_runtime_mode()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.max_num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        dcp_dummy_context_len = get_dcp_dummy_context_len(
            self.dcp_world_size,
            self.parallel_config.cp_kv_cache_interleave_size,
            hasattr(self, "kv_cache_config"),
            create_mixed_batch,
            is_graph_capturing,
            uniform_decode,
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        # Dummy runs have no real slot assignments — fill with -1 so
        # concat_and_cache kernels skip the KV write.
        if slot_mappings_by_group is not None:
            for sm in slot_mappings_by_group.values():
                sm.fill_(-1)

        # _dummy_run shares pinned CPU buffers (seq_lens, query_start_loc,
        # etc.) with execute_model.  It must participate in the same event
        # protocol so that back-to-back dummy/real steps don't overwrite
        # pinned memory while a prior non_blocking H2D DMA is still reading.
        with self.synchronize_input_prep():
            # If force_attention is True, we always capture attention.
            # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens  # type: ignore[assignment]
                elif create_mixed_batch:
                    # In the mixed batch mode (used for FI warmup), we use
                    # shorter sequence lengths to run faster.
                    # TODO(luka) better system for describing dummy batches
                    if dcp_dummy_context_len > 0:
                        seq_lens = torch.tensor(  # type: ignore[assignment]
                            [1 + dcp_dummy_context_len] * num_decode_tokens
                            + [num_prefill_tokens + dcp_dummy_context_len],
                            dtype=torch.int,
                        )
                    else:
                        seq_lens = torch.tensor(  # type: ignore[assignment]
                            [1] * num_decode_tokens + [num_prefill_tokens + 1],
                            dtype=torch.int,
                        )
                elif dcp_dummy_context_len > 0:
                    seq_lens = max_query_len + dcp_dummy_context_len  # type: ignore[assignment]
                else:
                    seq_lens = max_query_len  # type: ignore[assignment]
                self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
                self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
                self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

                cum_num_tokens = self._get_cumsum_and_arange(
                    num_scheduled_tokens, self.query_pos.np
                )
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
                self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
                    cum_num_tokens[-1]
                )
                self.query_start_loc.copy_to_gpu()

                prepare_dcp_dummy_context_metadata(
                    input_batch=self.input_batch,
                    kv_cache_config=getattr(self, "kv_cache_config", None),
                    query_pos=self.query_pos,
                    positions=self.positions,
                    query_start_loc=self.query_start_loc,
                    num_reqs=num_reqs,
                    num_tokens_unpadded=num_tokens_unpadded,
                    dcp_dummy_context_len=dcp_dummy_context_len,
                )

                # Sync block table CPU->GPU so cleared rows from
                # remove_request() are visible to the attention metadata
                # builder. Without this, stale block IDs from finished
                # requests can corrupt Mamba state.
                self.input_batch.block_table.commit_block_table(num_reqs_padded)

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    # FULL replay reads capture-time metadata buffers. Re-stage them
                    # from the zeroed dummy block tables instead of retaining state
                    # indices from the previous real batch.
                    for_cudagraph_capture=(
                        is_graph_capturing
                        or cudagraph_runtime_mode == CUDAGraphMode.FULL
                    ),
                    slot_mappings=slot_mappings_by_group,
                    use_spec_decode=self.speculative_config is not None,
                )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_gather_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer
                    | DFlashProposer
                    | DraftModelProposer
                    | ExtractHiddenStatesProposer
                    | Gemma4Proposer,
                )
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if (
                    self.compilation_config.cudagraph_specialize_lora
                    and num_active_loras > 0
                ):
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])

        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            logprob_token_ids=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
            # Also warm forward_native (taken when generators dict is non-empty),
            # but skip the extra call in 'processed_logits' / 'processed_logprobs'
            # modes — there TopKTopPSampler binds forward = forward_native at
            # init time, so the warmup call is redundant and only inflates peak
            # memory during profile_run.
            # No .clone() of logits: warmup output is discarded, so any in-place
            # mutation by forward_native does not affect correctness.
            if self.sampler.logprobs_mode not in PROCESSED_LOGPROBS_MODES:
                self.sampler(
                    logits=logits,
                    sampling_metadata=replace(
                        dummy_metadata,
                        generators={
                            0: torch.Generator(device=self.device).manual_seed(0)
                        },
                    ),
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            draft_probs = None
            if (
                self.speculative_config.rejection_sample_method == "standard"
                and self.speculative_config.draft_sample_method == "probabilistic"
            ):
                draft_probs = torch.rand(
                    num_tokens,
                    logits.shape[-1],
                    device=self.device,
                    dtype=torch.float32,
                )
                draft_probs = torch.softmax(draft_probs, dim=-1)
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_np = np.full(num_reqs, min_tokens_per_req)
        num_scheduled_tokens_np[-1] += num_tokens % num_reqs
        assert np.sum(num_scheduled_tokens_np) == num_tokens
        assert len(num_scheduled_tokens_np) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.from_numpy(num_scheduled_tokens_np)
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            prompt_token_ids_cpu=dummy_token_ids.cpu(),
            pooling_params=[dummy_pooling_params] * num_reqs,
            pooling_states=[PoolingStates() for i in range(num_reqs)],
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_cpu=dummy_prompt_lens,
            device=hidden_states.device,
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model not need to run pooler.
            return torch.tensor([])

        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            raise RuntimeError(
                f"Model {self.model_config.model} does not support "
                "any pooling tasks. See "
                "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                "to learn more."
            )

        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output if o is not None)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and mm_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    if not mm_budget.mm_max_toks_per_item:
                        # All modality limits are 0 — embedding-only mode.
                        # Budget is non-zero for embedding storage, but
                        # there's no encoder to profile.
                        logger.info(
                            "Skipping encoder profiling for embedding-only "
                            "mode (all modality limits=0 with "
                            "enable_mm_embeds=True).",
                        )
                    else:
                        # NOTE: Currently model is profiled with a single
                        # non-text modality with the max possible input
                        # tokens even when it supports multiple.
                        dummy_modality = mm_budget.get_modality_with_max_tokens()
                        max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[
                            dummy_modality
                        ]

                        logger.info_once(
                            "Encoder cache will be initialized with a "
                            "budget of %s tokens, and profiled with "
                            "%s %s items of the maximum feature size.",
                            encoder_budget,
                            max_mm_items_per_batch,
                            dummy_modality,
                        )

                        # Create dummy batch of multimodal inputs.
                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                            dummy_modality,
                            max_mm_items_per_batch,
                        )

                        # Run multimodal encoder.
                        dummy_encoder_outputs = self.model.embed_multimodal(
                            **batched_dummy_mm_inputs
                        )

                        sanity_check_mm_encoder_outputs(
                            dummy_encoder_outputs,
                            expected_num_items=max_mm_items_per_batch,
                        )
                        for i, output in enumerate(dummy_encoder_outputs):
                            self.encoder_cache[f"tmp_{i}"] = output

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def _init_minimal_kv_cache_for_profiling(self) -> None:
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )

        kv_cache_spec = self.get_kv_cache_spec()
        KVCacheSpecRegistry.check_kv_cache_spec_registry(kv_cache_spec)
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        # the minimum number of blocks required is 1 block *per sequence*
        min_blocks = (
            min(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
            or 1
        )

        # Temporarily change num_gpu_blocks_override to allocate a minimal KV cache
        saved_override = self.cache_config.num_gpu_blocks_override
        self.cache_config.num_gpu_blocks_override = min_blocks
        minimal_config = get_kv_cache_config_from_groups(
            self.vllm_config, kv_cache_groups, available_memory=0
        )
        self.cache_config.num_gpu_blocks_override = saved_override

        self.initialize_kv_cache(minimal_config, is_profiling=True)
        self.cache_config.num_gpu_blocks = minimal_config.num_blocks

        logger.debug("Initialized minimal KV cache for CUDA graph profiling")

    @staticmethod
    @contextmanager
    def _freeze_gc():
        gc.collect()
        should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
        if should_freeze:
            gc.freeze()
        try:
            yield
        finally:
            if should_freeze:
                gc.unfreeze()
                gc.collect()

    def shutdown(self) -> None:
        """Release GPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
        from vllm.v1.worker.workspace import reset_workspace_manager

        # Calls torch.accelerator.synchronize()
        self._cleanup_profiling_kv_cache()
        if current_platform.is_rocm():
            # Drop captured graphs before distributed teardown. On ROCm, delayed
            # graph destruction can surface HSA faults in the next engine startup.
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            self.encoder_cudagraph_manager = None
        self.compilation_config.static_forward_context.clear()
        self.model = None  # type: ignore[assignment]
        _ROPE_DICT.clear()

        reset_workspace_manager()
        if current_platform.is_rocm() or current_platform.is_xpu():
            gc.collect()
            torch.accelerator.empty_cache()
            torch.accelerator.synchronize()

    def _cleanup_profiling_kv_cache(self) -> None:
        torch.accelerator.synchronize()
        if hasattr(self, "kv_caches") and self.kv_caches:
            for i in range(len(self.kv_caches)):
                self.kv_caches[i] = None  # type: ignore
            self.kv_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None

        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                kv_cache = layer.kv_cache
                layer.kv_cache = (
                    torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
                )
            # Clean up quantized KV cache scale views
            # (int8_per_token_head, fp8_per_token_head)
            if hasattr(layer, "impl"):
                if hasattr(layer.impl, "_k_scale_cache"):
                    layer.impl._k_scale_cache = None
                if hasattr(layer.impl, "_v_scale_cache"):
                    layer.impl._v_scale_cache = None

        gc.collect()
        torch.accelerator.empty_cache()

        logger.debug("Cleaned up profiling KV cache and CUDA graphs")

    @torch.inference_mode()
    def _create_encoder_cudagraph_manager(self) -> "EncoderCudaGraphManager | None":
        if not (
            self.compilation_config.cudagraph_mm_encoder and self.supports_mm_inputs
        ):
            return None

        # Use get_model() to unwrap CUDAGraphWrapper/UBatchWrapper, because
        # @runtime_checkable Protocol isinstance() checks do not work through
        # __getattr__ forwarding.
        from vllm.model_executor.models.interfaces import (
            SupportsEncoderCudaGraph,
            supports_encoder_cudagraph,
        )
        from vllm.v1.worker.encoder_cudagraph import (
            EncoderCudaGraphManager,
        )

        raw_model = self.get_model()
        if not supports_encoder_cudagraph(raw_model):
            return None

        return EncoderCudaGraphManager(
            vllm_config=self.vllm_config,
            device=self.device,
            dtype=self.dtype,
            model=cast(SupportsEncoderCudaGraph, raw_model),
        )

    @torch.inference_mode()
    def _maybe_init_encoder_cudagraph_manager(self) -> None:
        if self.encoder_cudagraph_manager is None:
            self.encoder_cudagraph_manager = self._create_encoder_cudagraph_manager()
            if self.encoder_cudagraph_manager is not None:
                logger.info("Initialized EncoderCudaGraphManager for vision encoder")

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        with set_current_vllm_config(self.vllm_config):
            self._init_minimal_kv_cache_for_profiling()

        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured

        capture_descs = self.cudagraph_dispatcher.get_capture_descs()
        # Use a temporary manager for memory profiling. The persistent manager
        # is initialized later so it does not keep profiling-only graph state.
        encoder_cudagraph_manager = self._create_encoder_cudagraph_manager()

        decoder_graphs = sum(len(descs) for _, descs in capture_descs)
        encoder_graphs = (
            encoder_cudagraph_manager.get_num_graphs_to_capture()
            if encoder_cudagraph_manager is not None
            else 0
        )
        total_graphs = decoder_graphs + encoder_graphs
        if total_graphs == 0:
            logger.debug("No CUDA graphs will be captured, skipping profiling")
            self._cleanup_profiling_kv_cache()
            return 0

        graph_groups = [
            *(
                f"{mode.name}={len(descs)} (largest={descs[0].num_tokens})"
                for mode, descs in capture_descs
                if descs
            ),
        ]
        if encoder_graphs > 0:
            graph_groups.append(
                f"ENCODER={encoder_graphs} "
                f"(largest={encoder_cudagraph_manager.token_budgets[-1]})"
            )

        logger.info("Profiling CUDA graph memory: %s", ", ".join(graph_groups))

        # Use a temporary pool for profiling to avoid fragmentation in the main pool.
        profiling_pool = current_platform.graph_pool_handle()
        encoder_profiling_pool = current_platform.graph_pool_handle()
        original_pools: dict[int, Any] = {}
        all_wrappers = list(CUDAGraphWrapper._all_instances) + list(
            BreakableCUDAGraphWrapper._all_instances
        )
        for instance in all_wrappers:
            original_pools[id(instance)] = instance.graph_pool
            instance.graph_pool = profiling_pool

        shared_memory_estimate = {}
        per_graph_estimate = {}
        encoder_memory_estimate = 0

        # On ROCm, capture these throwaway profiling graphs on vLLM's dedicated
        # compute stream instead of the fresh side stream graph_capture()
        # allocates by default. torch's allocator pools free blocks per stream,
        # so a side-stream forward strands a persistent aiter scratch buffer in
        # a separate pool, shifting the physical placement of the real KV cache
        # allocated afterward and slowing bandwidth-bound decode ~20%. The
        # graphs are discarded, so a side stream is unnecessary here.
        # Use current_stream(), not torch.cuda.current_stream(): before vLLM
        # initializes its dedicated stream, torch returns the per-thread default
        # stream (cuda_stream=0), which cannot be used for cudagraph capture.
        # cap_ctx=None keeps the side-stream path on CUDA.
        cap_ctx = (
            GraphCaptureContext(current_stream())
            if current_platform.is_rocm()
            else None
        )

        # Cleanup-only guard: CUDA graph capture errors should still propagate
        # because encoder graph capture is opt-in.
        try:
            set_cudagraph_capturing_enabled(True)
            with (
                self._freeze_gc(),
                graph_capture(device=self.device, graph_capture_context=cap_ctx),
            ):
                torch.accelerator.synchronize()
                torch.accelerator.empty_cache()

                for mode, descs in capture_descs:
                    profile_descs = descs[:2]
                    mem_samples: list[int] = []

                    for i, desc in enumerate(profile_descs):
                        mem_before = torch.accelerator.get_memory_info()[0]
                        self._warmup_and_capture(
                            desc,
                            cudagraph_runtime_mode=mode,
                            profile_seq_lens=(
                                min(
                                    self.max_model_len,
                                    self.max_num_tokens // desc.num_tokens,
                                )
                                if mode == CUDAGraphMode.FULL and i == 0
                                else None
                            ),
                        )
                        torch.accelerator.synchronize()
                        free_after = torch.accelerator.get_memory_info()[0]
                        mem_samples.append(mem_before - free_after)

                    first_capture = mem_samples[0]
                    # Use at least 1 MiB per graph for driver overhead
                    per_graph = max(
                        mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20
                    )

                    shared_memory_estimate[mode] = first_capture
                    per_graph_estimate[mode] = per_graph * (len(descs) - 1)

                    logger.debug(
                        "Estimated %s CUDA graph memory: "
                        "%.2f MiB first-capture + (%d-1) × %.2f MiB per-graph",
                        mode.name,
                        first_capture / (1 << 20),
                        len(descs),
                        per_graph / (1 << 20),
                    )

                if encoder_cudagraph_manager is not None:
                    mem_before = torch.accelerator.get_memory_info()[0]
                    encoder_cudagraph_manager.capture(graph_pool=encoder_profiling_pool)
                    torch.accelerator.synchronize()
                    free_after = torch.accelerator.get_memory_info()[0]
                    encoder_memory_estimate = max(mem_before - free_after, 0)

                    logger.debug(
                        "Estimated encoder CUDA graph memory: %.2f MiB for %d graphs",
                        encoder_memory_estimate / (1 << 20),
                        encoder_graphs,
                    )
        finally:
            set_cudagraph_capturing_enabled(False)
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            if encoder_cudagraph_manager is not None:
                encoder_cudagraph_manager.clear()
            all_wrappers = list(CUDAGraphWrapper._all_instances) + list(
                BreakableCUDAGraphWrapper._all_instances
            )
            for instance in all_wrappers:
                if id(instance) in original_pools:
                    instance.graph_pool = original_pools[id(instance)]
            for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():
                key_set.clear()
            self.cudagraph_dispatcher.keys_initialized = False
            self.maybe_remove_all_loras(self.lora_config)
            self._cleanup_profiling_kv_cache()
            compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured

        # FULL and PIECEWISE graphs share the global pool at runtime and are
        # never replayed concurrently, so the pool overlays their memory.
        # Take the max to avoid double-counting the overlap.
        decoder_estimate = max(shared_memory_estimate.values(), default=0) + sum(
            per_graph_estimate.values()
        )
        # Encoder graphs use a manager-local pool at runtime, separate from the
        # decoder pool, so add their estimate instead of overlaying it.
        total_estimate = decoder_estimate + encoder_memory_estimate
        logger.info(
            "Estimated CUDA graph memory: %.2f GiB total",
            total_estimate / (1 << 30),
        )

        return int(total_estimate)

    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        # Initialize encoder CUDA graph manager if enabled.
        self._maybe_init_encoder_cudagraph_manager()

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)

        # Setup torch profiler for graph capture traces (conditional)
        from vllm.distributed.parallel_state import get_world_group

        local_rank = get_world_group().local_rank
        enable_profiler = (
            local_rank == 0
        ) and self.vllm_config.profiler_config.capture_torch_profiler
        if enable_profiler:
            trace_dir = (
                self.vllm_config.profiler_config.torch_profiler_dir + "/capture_traces"
            )
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    trace_dir,
                    worker_name=f"graph_capture_rank_{local_rank}",
                    use_gzip=True,
                ),
            )
            logger.info_once(
                "Rank %d: Torch profiler enabled for CUDA graph capture, "
                "traces will be saved to: %s",
                local_rank,
                trace_dir,
            )
        else:
            profiler = nullcontext()
            logger.info_once(
                "Rank %d: Torch profiler disabled for CUDA graph capture", local_rank
            )

        with self._freeze_gc(), graph_capture(device=self.device):
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

            for (
                runtime_mode,
                batch_descs,
            ) in self.cudagraph_dispatcher.get_capture_descs():
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                    profiler=profiler,
                )
                torch.accelerator.synchronize()

            # Capture encoder CUDA graphs if enabled
            if self.encoder_cudagraph_manager is not None:
                encoder_graph_pool = current_platform.graph_pool_handle()
                self.encoder_cudagraph_manager.capture(graph_pool=encoder_graph_pool)

            torch.accelerator.synchronize()
            end_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()

        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
        profiler: AbstractContextManager[Any] | None = None,
    ):
        if profiler is None:
            profiler = nullcontext()
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        for _ in range(num_warmups):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=force_attention,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                profile_seq_lens=profile_seq_lens,
            )
        if num_warmups > 0:
            # Warmups may use auxiliary streams. Ensure all of their work has
            # completed before beginning CUDA graph capture.
            torch.accelerator.synchronize()
        with (
            profiler,
            torch.profiler.record_function(
                f"capture_{desc.num_tokens}_{cudagraph_runtime_mode.name}"
            ),
        ):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                is_graph_capturing=True,
                profile_seq_lens=profile_seq_lens,
            )

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
        profiler: AbstractContextManager[Any] | None = None,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        uniform_decode = batch_descriptors[0].uniform

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            batch_descriptors = tqdm(
                batch_descriptors,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # We skip EPLB here since we don't want to record dummy metrics
        for batch_desc in batch_descriptors:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=batch_desc.num_tokens,
                    uniform_decode=uniform_decode,
                )
            )
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
                profiler=profiler,
            )
            torch.accelerator.synchronize()
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            """Deduplication key for attention groups within a KV cache group.

            Splits on per-rank ``num_heads_q`` in addition to backend + spec
            so layers with different Q-head counts (e.g. a spec-decode draft
            with fewer attention heads than its target) get separate metadata
            builders. The builders' scratch (e.g. ``softmax_segm_*`` in
            ``triton_attn``, ``num_qo_heads`` in FlashInfer) is sized by
            ``num_heads_q`` and assumes uniformity within the group; see
            ``get_num_attention_heads_from_layers`` in
            ``vllm/v1/attention/backends/utils.py``.
            """

            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec
            num_heads_q: int

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                # Non-Attention layer types (e.g. Mamba1, ShortConv) do not
                # expose ``num_heads``; fall back to 0 so they cluster as
                # before. Such layers never coexist with Attention in a
                # single KV cache group (different KVCacheSpec), so the
                # fallback can never spuriously merge them with attention
                # layers.
                num_heads_q = getattr(layers[layer_name], "num_heads", 0)
                key = (full_cls_name, layer_kv_cache_spec, num_heads_q)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec, num_heads_q
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for key, layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    key.attn_backend,
                    layer_names,
                    key.kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(
            attention_backend_list,
            kv_cache_config.kv_cache_groups,
            is_profiling=is_profiling,
        )

        # Check if attention backend supports PCP&DCP and related features.
        check_attention_cp_compatibility(self.vllm_config)

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

        # Initialize drafter attention backend
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer | DFlashProposer | DraftModelProposer | Gemma4Proposer,
            )
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
        is_profiling: bool = False,
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_attn_backend = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_attn_backend = attn_backend.__name__
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            min_cg_support,
            min_cg_attn_backend,
            self.uniform_decode_query_len,
            use_v2_model_runner=False,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
            is_profiling=is_profiling,
        )
        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer
                | DFlashProposer
                | DraftModelProposer
                | ExtractHiddenStatesProposer
                | Gemma4Proposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        what it was originally created with. This happens when the final
        block size (determined after model loading) differs from the
        placeholder used during __init__, or when there are multiple
        KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = []
        max_num_blocks = []
        slot_mapping_modes = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            kv_cache_spec_kind = get_kv_cache_spec_kind(kv_cache_spec)
            if kv_cache_spec_kind == KVCacheSpecKind.ENCODER_ONLY_ATTENTION:
                continue
            block_size = kv_cache_spec.block_size
            block_sizes.append(block_size)
            if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:
                slot_mapping_modes.append(SlotMappingMode.NONE)
            else:
                slot_mapping_modes.append(SlotMappingMode.TOKEN_TO_KV_SLOT)
            max_num_blocks_per_req = kv_cache_spec.max_num_blocks_per_req(
                self.vllm_config, max_model_len
            )
            max_num_blocks.append(max_num_blocks_per_req)

        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
            or max_num_blocks != self._init_max_num_blocks
            or slot_mapping_modes != self._init_slot_mapping_modes
        ):
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
            self._init_max_num_blocks = max_num_blocks
            self._init_slot_mapping_modes = slot_mapping_modes
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                num_spec_tokens=self.num_spec_tokens,
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
                cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
                reasoning_config=self.vllm_config.reasoning_config,
                use_replayssm=self.cache_config.use_replayssm,
                slot_mapping_modes=slot_mapping_modes,
            )

        assert self._init_block_sizes == block_sizes, (
            f"InputBatch block_sizes {self._init_block_sizes} != "
            f"kv_cache block_sizes {block_sizes}"
        )
        assert self._init_kernel_block_sizes == kernel_block_sizes, (
            f"InputBatch kernel_block_sizes {self._init_kernel_block_sizes} "
            f"!= kv_cache kernel_block_sizes {kernel_block_sizes}"
        )

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        packed_backing: torch.Tensor | None = None
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            if kv_cache_tensor.block_stride > 0:
                # Allocate once; all packed tensors alias the same backing.
                if packed_backing is None:
                    packed_backing = torch.zeros(
                        kv_cache_tensor.size,
                        dtype=torch.int8,
                        device=self.device,
                    )
                tensor = packed_backing
            else:
                tensor = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=self.device
                )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False

        # Map layer names to (offset, block_stride) within the packed
        # backing tensor so we can create strided views per layer.
        layer_packing: dict[str, tuple[int, int]] = {}
        for kv_tensor in self.kv_cache_config.kv_cache_tensors:
            if kv_tensor.block_stride > 0:
                for ln in kv_tensor.shared_by:
                    layer_packing[ln] = (kv_tensor.offset, kv_tensor.block_stride)
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                packing = layer_packing.get(layer_name)
                if packing is not None:
                    _, blk_stride = packing
                    num_blocks = raw_tensor.numel() // blk_stride
                else:
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    # For MLA with compression, storage_block_size != block_size
                    if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
                        shape_block_size = kv_cache_spec.storage_block_size
                    else:
                        shape_block_size = kernel_block_size

                    # Skipped layers (--kv-cache-dtype-skip-layers) need
                    # the unquantized shape.
                    layer_cache_dtype_str = (
                        "auto"
                        if kv_cache_spec.kv_quant_mode == KVQuantMode.NONE
                        else getattr(
                            kv_cache_spec,
                            "cache_dtype_str",
                            None,
                        )
                        or self.cache_config.cache_dtype
                    )
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        shape_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=layer_cache_dtype_str,
                    )
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    kv_caches[layer_name] = _reshape_attention_kv_cache(
                        raw_tensor,
                        kv_cache_spec,
                        kv_cache_shape,
                        kv_cache_stride_order,
                        kernel_num_blocks,
                        packing,
                    )

                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    page_size_bytes = kv_cache_spec.page_size_bytes
                    # Hold a single contiguous [num_blocks, 1, 1, page_size_bytes]
                    # int8 page view per layer; the layer's bind_kv_cache unpacks
                    # each block's bytes into its conv/ssm state views. Keeping
                    # one tensor per layer lets the KV connector register it
                    # without special-casing Mamba.
                    kv_caches[layer_name] = raw_tensor[
                        : num_blocks * page_size_bytes
                    ].view(num_blocks, 1, 1, page_size_bytes)
                else:
                    raise NotImplementedError

        # Reconcile divergent KV layouts to blocks-first. Triggered by hybrid
        # attention/mamba models, and by encoder-decoder models whose shared
        # decoder/cross-attention allocation mixes K/V-first and blocks-first
        # backends (see _has_mixed_attention_kv_layout).
        if has_attn and (
            has_mamba or self._has_mixed_attention_kv_layout(kernel_block_sizes)
        ):
            self._update_hybrid_attention_mamba_layout(kv_caches, kernel_block_sizes)

        return kv_caches

    def _has_mixed_attention_kv_layout(self, kernel_block_sizes: list[int]) -> bool:
        """Whether attention groups disagree on the physical KV cache layout.

        Encoder-decoder models (e.g. Whisper) share one raw KV allocation
        between a decoder self-attention layer (K/V-first ROCM_ATTN, block dim
        1) and a cross-attention layer (blocks-first, block dim 0). Mixed block
        dims mean a block ID maps to different bytes per layer, so the shared
        buffer must be normalized to a single (blocks-first) layout.
        """
        block_dims: set[int] = set()
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue
            block_dims.add(
                group.backend.get_kv_cache_block_dim(
                    kernel_block_sizes[group.kv_cache_group_id],
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=self.cache_config.cache_dtype,
                )
            )
        return len(block_dims) > 1

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor], kernel_block_sizes: list[int]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                continue
            block_dim = group.backend.get_kv_cache_block_dim(
                kernel_block_sizes[group.kv_cache_group_id],
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype_str=self.cache_config.cache_dtype,
            )
            # block_dim: 0 means (num_blocks, 2, ...); 1 means (2, num_blocks, ...).
            if block_dim == 0:
                continue
            assert block_dim == 1
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                hidden_size = kv_cache.shape[2:].numel()
                kv_cache.as_strided_(
                    size=kv_cache.shape,
                    stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_raw_tensors, kernel_block_sizes
            )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self._mamba_bufs = None
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        self._kernel_block_sizes = kernel_block_sizes

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        if (
            self.speculative_config
            and self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, ExtractHiddenStatesProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group() and not is_profiling:
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend
                )
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

    def get_routed_experts(
        self,
        num_tokens: int,
    ) -> RoutedExpertsTensors | None:
        if not self.routed_experts_initialized:
            return None

        device_buffer = self.routed_experts_capturer.get_device_buffer()
        return RoutedExpertsTensors(
            routing_data=device_buffer[:num_tokens].clone(),
            slot_mapping=self.routed_experts_slot_mapping_device[:num_tokens].clone(),
        )

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        self.routed_experts_capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
        )
        bind_routed_experts_capturer(self.model, self.routed_experts_capturer)

        # Pinned CPU buffer for non-blocking D2H of ``routing_data`` on
        # the sync scheduling path. Shape / dtype mirror the device
        # capturer exactly so ``copy_`` is a straight memcpy.
        self.routed_experts_cpu = torch.empty(
            self.routed_experts_capturer.device_buffer.shape,
            dtype=self.routed_experts_capturer.device_buffer.dtype,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        # ``slot_mapping`` dtype is fixed to int64 by
        # ``block_table.slot_mapping``; we mirror that here.
        max_tokens = self.scheduler_config.max_num_batched_tokens
        self.routed_experts_slot_mapping_cpu = torch.empty(
            (max_tokens,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        # Private device buffer so the shared ``block_table.slot_mapping``
        # can be overwritten by the next ``_prepare_inputs`` while the
        # D2H is still pending on the copy stream. Written in
        # ``_prepare_inputs``, read in ``_bookkeeping_sync`` (sync path)
        # or cloned into a snapshot (async path).
        self.routed_experts_slot_mapping_device = torch.empty(
            (max_tokens,),
            dtype=torch.int64,
            device=self.device,
        )
        self.routed_experts_initialized = True

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        if has_ec_transfer() and not get_ec_transfer().is_consumer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                if isinstance(spec, AttentionSpec):
                    backend = attn_module.get_attn_backend()
                    # indexes_kv_by_block_stride() -> get_kv_cache_stride_order()
                    # -> get_kv_cache_layout() needs the current vLLM config.
                    with set_current_vllm_config(self.vllm_config):
                        indexes = backend.indexes_kv_by_block_stride()
                    spec = replace(spec, indexes_kv_by_block_stride=indexes)
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """
        Get encoder timing stats for all requests and clear the registry.

        Returns:
            Dictionary mapping request_id to stats dict.
        """
        with self._encoder_timing_lock:
            stats = {
                req_id: stats_obj.to_dict()
                for req_id, stats_obj in self.encoder_timing_registry.items()
            }
            self.encoder_timing_registry.clear()
            return stats

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """
        Context manager to time encoder forward operations.

        Args:
            should_time: Whether timing is enabled
            group_lora_refs: Full list of (request_id, pos_info) tuples
            current_item_idx: Starting index for this group
            num_items: Number of items in this group
        """
        if not should_time:
            yield
            return

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.accelerator.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_secs += per_request_time
                    stats.num_encoder_calls += 1


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
