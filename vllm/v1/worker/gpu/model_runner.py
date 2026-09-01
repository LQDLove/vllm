# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
注意: 本文件的编码风格指南:
此 model runner 被所有模型共享: 文本与多模态、生成与嵌入、公开与私有。
因此, 本文件只能包含所有模型通用的代码。模型特定的行为应放在
对应的模型特定文件中。

换句话说:
* 对修改此文件要极其谨慎, 它应保持稳定。
* 对新增代码行要更加谨慎, 它应保持精简。

即使是共享特性(例如不同的并行模式), 也要把复杂性挡在此路径之外。
特性越不常见, 越应被隐藏。优先使用在其他地方定义的工具函数并
从这里调用, 而不是直接把特性特定逻辑嵌入其中。
"""

# 导入 functools,用于 cached_property 等装饰器。
import functools
# 导入 gc,用于垃圾回收(显存释放前触发)。
import gc
# 导入 time,用于计时统计。
import time
# 导入 deepcopy,用于深拷贝 KV cache 配置。
from copy import deepcopy
# 导入类型标注工具。
from typing import Any, NamedTuple

# 导入 numpy,用于 CPU 侧数组运算。
import numpy as np
# 导入 PyTorch,用于张量与 CUDA 操作。
import torch
# 导入 nn 模块,用于模型类型标注。
import torch.nn as nn

# 导入环境变量配置。
import vllm.envs as envs
# 导入编译计数器,统计 capture 触发次数。
from vllm.compilation.counter import compilation_counter
# 导入 vLLM 总配置。
from vllm.config import VllmConfig
# 导入 CUDA graph 模式枚举。
from vllm.config.compilation import CUDAGraphMode
# 导入并行状态组访问器(DCP 与 PP)。
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
)
# 导入前向上下文工具(批次描述符与上下文设置)。
from vllm.forward_context import BatchDescriptor, set_forward_context
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 EP all2all 管理器访问器。
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
# 导入路由专家捕获器及其绑定函数。
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
# 导入 Mamba SSU 后端初始化函数。
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
# 导入模型加载器工厂。
from vllm.model_executor.model_loader import get_model_loader
# 导入多模态注册表。
from vllm.multimodal import MULTIMODAL_REGISTRY
# 导入多模态编码预算与 dummy 编码器输入构造。
from vllm.multimodal.encoder_budget import (
    MultiModalBudget,
    get_dummy_encoder_profile_inputs,
)
# 导入中间张量容器。
from vllm.sequence import IntermediateTensors
# 导入支持任务类型枚举。
from vllm.tasks import SupportedTask
# 导入向上整除工具。
from vllm.utils.math_utils import cdiv
# 导入显存分析器与格式化工具。
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
# 导入字符串 dtype 到 torch dtype 的映射。
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
# 导入语法输出与调度器输出。
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
# 导入 KV cache 配置与 Mamba 规格。
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
# 导入输出相关结构(草稿 token、输出、路由专家张量等)。
from vllm.v1.outputs import (
    DraftTokenIds,
    ModelRunnerOutput,
    RoutedExpertsTensors,
    make_empty_encoder_model_runner_output,
)
# 导入块表宽度计算函数。
from vllm.v1.worker.block_table import get_block_table_width
# 导入 CP 注意力兼容性检查。
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
# 导入 PCP(流水线上下文并行)管理器模块。
from vllm.v1.worker.gpu import pcp_manager as pcp
# 导入异步输出结构。
from vllm.v1.worker.gpu.async_utils import AsyncOutput, AsyncPoolingOutput
# 导入注意力工具(slot 映射构建、KV spec、后端与缓存初始化)。
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
# 导入 GPU v2 块表管理。
from vllm.v1.worker.gpu.block_table import BlockTables
# 导入缓冲区工具(异步拷贝与默认并发设置)。
from vllm.v1.worker.gpu.buffer_utils import (
    async_copy_to_gpu,
    set_default_max_concurrency,
)
# 导入 DCP 本地序列长度准备函数。
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
# 导入 CUDA graph 工具(批次执行描述符、图管理器、均匀 token 数)。
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
    get_uniform_token_count,
)
# 导入 DP 分发与同步工具。
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
# 导入 EC(嵌入缓存)连接器工厂。
from vllm.v1.worker.gpu.ec_connector import get_ec_connector
# 导入 EPLB 控制器与步进函数。
from vllm.v1.worker.gpu.eplb_utils import EPLBController, step_eplb_after
# 导入输入批次及相关工具函数。
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
# 导入 KV 连接器(空操作连接器、类型、工厂)。
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
# 导入 LoRA 状态与相关工具。
from vllm.v1.worker.gpu.lora_utils import (
    LoraState,
    create_lora_capture_hook,
    get_lora_capture_cases,
    get_num_active_loras_for_dispatch,
)
# 导入编码器缓存。
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
# 导入多模态 LoRA 激活函数。
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
# 导入模型状态初始化函数。
from vllm.v1.worker.gpu.model_states import init_model_state
# 导入池化运行器。
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
# 导入 PP(流水线并行)处理器。
from vllm.v1.worker.gpu.pp_utils import PPHandler
# 导入采样器输出。
from vllm.v1.worker.gpu.sample.output import SamplerOutput
# 导入 prompt logprobs 工作器。
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
# 导入采样器。
from vllm.v1.worker.gpu.sample.sampler import Sampler
# 导入关机前的释放工具。
from vllm.v1.worker.gpu.shutdown import free_before_shutdown
# 导入投机解码初始化函数。
from vllm.v1.worker.gpu.spec_decode import init_speculator
# 导入 EAGLE3 辅助隐藏层设置函数。
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
# 导入拒绝采样器。
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
# 导入草稿模型投机器。
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
# 导入草稿 token 处理器。
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
# 导入请求状态管理。
from vllm.v1.worker.gpu.states import RequestState
# 导入结构化输出工作器。
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
# 导入 LoRA 模型运行器混入类。
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
# 导入 KV 块清零器与 KV 块拷贝工具。
from vllm.v1.worker.utils import KVBlockZeroer, copy_kv_cache_blocks_inplace

# 创建本模块的日志记录器。
logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):
    # GPU v2 模型运行器:所有模型的通用执行入口,
    # 继承 LoRA 混入类以获得 LoRA 加载与激活能力。

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # 保存总配置。
        self.vllm_config = vllm_config
        # 保存模型配置。
        self.model_config = vllm_config.model_config
        # 保存缓存配置。
        self.cache_config = vllm_config.cache_config
        # 保存编译(含 CUDA graph)配置。
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

        # 保存目标设备。
        self.device = device
        # 模型(激活)数据类型。
        self.dtype = self.model_config.dtype
        # 是否为 encoder-only 模型。
        self.is_encoder_only = vllm_config.is_encoder_only
        # KV cache 数据类型默认与模型相同。
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # 量化 KV cache:按字符串配置转换为 torch dtype。
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # KV cache 需要清零时(如 hybrid 模型 + fp8 KV cache),
        # 在 _init_kv_zero_meta() 中惰性创建。
        self.kv_block_zeroer: KVBlockZeroer | None = None

        # 词表大小。
        self.vocab_size = self.model_config.get_vocab_size()
        # 模型最大上下文长度。
        self.max_model_len = self.model_config.max_model_len
        # 单批最大调度 token 数。
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        # 最大并发请求数。
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        # 是否为 encoder-decoder 模型。
        self.is_encoder_decoder = self.model_config.is_encoder_decoder

        # 创建输出拷贝专用 CUDA 流(与主计算流重叠)。
        self.output_copy_stream = torch.cuda.Stream(self.device)

        # 流水线并行相关。
        # 是否使用 PP。
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        # 是否为 PP 首个 rank。
        self.is_first_pp_rank = get_pp_group().is_first_rank
        # 是否为 PP 最后一个 rank。
        self.is_last_pp_rank = get_pp_group().is_last_rank

        # 按最大并发飞行步数确定 UVA 缓冲池大小。
        # 必须在任何池化缓冲区构造之前运行。
        set_default_max_concurrency(vllm_config.max_concurrent_batches)

        # PP 广播/接收辅助器,集合通信在侧流上运行。
        self.pp_handler: PPHandler | None = None

        # 中间张量的持久缓冲区(非首个 PP rank 使用)。
        self.intermediate_tensors: IntermediateTensors | None = None

        # 数据并行相关。
        # DP 组大小。
        self.dp_size = self.parallel_config.data_parallel_size
        # DP rank。
        self.dp_rank = self.parallel_config.data_parallel_rank

        # 检测 EP all2all 对端故障,防止输出损坏结果。
        # 仅对使用支持容错 all2all 后端的 MoE + DP 有意义。
        self.check_ep_fault = False
        if self.dp_size > 1 and self.model_config.is_moe:
            # MoE + DP 时查询后端是否支持容错。
            self.check_ep_fault = get_ep_all2all_manager().support_fault_tolerance

        # 解码上下文并行(DCP)相关。
        # DCP 组大小。
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        # 是否启用 DCP。
        self.use_dcp = self.dcp_size > 1
        # 本 rank 在 DCP 组内的序号(未启用时为 0)。
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        # CP KV cache 交错大小。
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # 多模态相关。
        # 多模态注册表。
        self.mm_registry = MULTIMODAL_REGISTRY
        # 是否支持多模态输入。
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        # 编码器缓存默认为 None。
        self.encoder_cache = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # 支持多模态且为首个 PP rank 时创建编码器缓存。
            self.encoder_cache = EncoderCache()
        # 创建 EC 连接器。
        self.ec_connector = get_ec_connector(vllm_config, self.encoder_cache)

        # 投机解码相关。
        # 投机器(speculator)默认为 None。
        self.speculator = None
        # 是否需要使用目标模型的辅助隐藏层输出。
        self.use_aux_hidden_state_outputs = False
        # 投机步数(每步草稿 token 数)。
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        if self.speculative_config is not None:
            if self.is_last_pp_rank:
                # 仅最后一个 PP rank 需要投机器。
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method in ("eagle3", "dflash", "dspark"):
                # 起草可能需要目标模型输出的辅助隐藏状态。
                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    # 这些方法不支持流水线并行。
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )

        # 草稿 token 传播 - 用于投机解码 + 结构化输出。
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        # PCP(流水线上下文并行)管理器,初始化 KV cache 时构建。
        self.pcp_manager: pcp.PCPManager | None = None

        # 池化模型相关。
        # 是否为池化模型。
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        # 池化运行器(最后一个 PP rank 上创建)。
        self.pooling_runner: PoolingRunner | None = None

        # 多模块 MTP 在 chunked prefill 期间向其模块喂入
        # 随后 num_speculative_steps 个 prefill token;
        # 其他投机器只读紧接着的一个。
        num_prefill_lookahead = (
            self.num_speculative_steps
            if self.speculative_config is not None
            and self.speculative_config.use_multi_module_mtp()
            else 1
        )
        # 通用请求状态管理器。
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        # 输入缓冲区集合(GPU 上的持久输入张量)。
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        if self.use_pp:
            # 使用 PP 时创建 PP 处理器。
            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )

        # 采样器与 decode_query_len 在 load_model() 中创建,
        # 因为需要 model_state 存在后才知道每步新采样 token 数。
        self.sampler: Sampler | None = None
        self.rejection_sampler: RejectionSampler | None = None
        self.prompt_logprobs_worker: PromptLogprobsWorker | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None
        self.cudagraph_manager: ModelCudaGraphManager | None = None

        # LoRA 相关工作器。
        # LoRA 状态管理器。
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        # LoRA 捕获用例(默认 0)。
        self.lora_capture_cases = [0]
        if self.lora_config:
            # 按 LoRA 配置与编译配置确定捕获用例。
            self.lora_capture_cases = get_lora_capture_cases(
                self.lora_config, self.compilation_config
            )

        # 配置了 KV 连接器时使用;默认为空操作连接器。
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # 用于把 execute_model 的状态传递给后续 sample_tokens 调用。
        self.execute_model_state: ExecuteModelState | None = None

        # 专家并行负载均衡器(EPLB)。
        self.eplb = EPLBController(self.parallel_config, self.device)
        # 路由专家捕获器(按需初始化)。
        self.routed_experts_capturer: RoutedExpertsCapturer | None = None

    def update_max_model_len(self, max_model_len: int) -> None:
        # 更新最大模型长度(profile 时可能调整)。
        self.max_model_len = max_model_len
        # 同步更新请求状态中的最大长度。
        self.req_states.max_model_len = max_model_len

    def init_routed_experts_capturer(self) -> None:
        """在每个参与 worker 上初始化目标模型的专家捕获。"""
        # 创建路由专家捕获器。
        self.routed_experts_capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=self.max_num_tokens,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
        )
        # 把捕获器绑定到模型上。
        bind_routed_experts_capturer(self.model, self.routed_experts_capturer)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 收集本运行器支持的任务类型。
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            # 生成类任务由模型状态给出。
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            # 不依赖 pooling_runner,因为首个 PP rank 也需要该信息,
            # 而 pooling_runner 只在最后一个 PP rank 初始化。
            tasks.extend(
                PoolingRunner.get_supported_tasks(self.model, self.model_config)
            )
        # 转为元组返回。
        return tuple(tasks)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        # 加载模型权重并初始化依赖模型的组件。
        # 记录加载开始时间。
        time_before_load = time.perf_counter()
        if load_dummy_weights:
            # 使用 dummy 权重(profile 用)。
            self.load_config.load_format = "dummy"
        # EPLB 加载前准备。
        self.eplb.prepare_load()
        # 标记是否向 EPLB 注册了额外模型。
        eplb_models_added = False
        # 在显存分析器上下文中加载,统计显存占用。
        with DeviceMemoryProfiler() as m:
            # 获取模型加载器。
            model_loader = get_model_loader(self.vllm_config.load_config)
            # 记录日志(只输出一次)。
            logger.info_once("Loading model from scratch...")

            # 从头加载模型。
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.vllm_config.model_config
            )
            if self.lora_config:
                # 配置了 LoRA 时加载 LoRA 模型包装。
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )

            if self.use_aux_hidden_state_outputs:
                # EAGLE3 等方法需要设置辅助隐藏状态层。
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
            if isinstance(self.speculator, DraftModelSpeculator):
                # 草稿模型投机器需要加载草稿模型。
                self.speculator.load_model(self.model)
                # 尝试把投机器注册到 EPLB。
                eplb_models_added = self.eplb.maybe_register_speculator(
                    self.speculator, self.speculative_config, load_dummy_weights
                )
        # 记录加载结束时间。
        time_after_load = time.perf_counter()

        # 记录模型显存占用。
        self.model_memory_usage = m.consumed_memory
        # 输出加载耗时与显存日志。
        logger.info(
            "Model loading took %s GiB and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        # 初始化依赖模型的组件。
        # 创建模型状态。
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )

        # 解码 query 长度 = 投机步数 + 每步新采样 token 数。
        self.decode_query_len = (
            self.num_speculative_steps
            + self.model_state.num_new_sampled_tokens_per_step
        )

        # 初始化采样器。模型状态可能通过 custom_sampler() 覆盖。
        if self.is_last_pp_rank and not self.is_pooling_model:
            # 最后一个 PP rank 且非池化模型时创建采样器。
            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            )
            # 让模型状态有机会提供自定义采样器。
            custom = self.model_state.custom_sampler(self.sampler)

            if custom:
                # 使用自定义采样器(含拒绝采样器)。
                self.sampler, self.rejection_sampler = custom
            elif self.speculative_config is not None:
                # 启用投机解码时创建拒绝采样器。
                self.rejection_sampler = RejectionSampler(
                    self.sampler,
                    self.speculative_config,
                    self.device,
                )
            # 创建 prompt logprobs 工作器。
            self.prompt_logprobs_worker = PromptLogprobsWorker(
                self.max_num_reqs,
                logprobs_mode=self.model_config.logprobs_mode,
            )
            # 创建结构化输出工作器。
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * self.decode_query_len,
                vocab_size=self.vocab_size,
                device=self.device,
            )

        if self.is_pooling_model and self.is_last_pp_rank:
            # 池化模型在最后一个 PP rank 上创建池化运行器。
            self.pooling_runner = PoolingRunner(self.model, self.vllm_config)
        # 尝试把目标模型注册到 EPLB。
        eplb_models_added |= self.eplb.maybe_register_model(
            self.model,
            self.model_config,
            load_dummy_weights,
        )
        # 需要时启动 EPLB 异步循环。
        self.eplb.maybe_start_async_loop(eplb_models_added)

        if not self.is_first_pp_rank:
            # 非首个 PP rank:创建按最大 capture 尺寸定型的中间张量,
            # 以便每批切片使用。保存为持久成员,使运行时能把收到的数据
            # 拷入 CUDA graph 捕获的相同地址。
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

    def get_model(self) -> nn.Module:
        # 返回目标模型。
        return self.model

    def get_draft_model(self) -> nn.Module | None:
        # 返回草稿模型;非 DraftModelSpeculator 时为 None。
        speculator = self.speculator
        if not isinstance(speculator, DraftModelSpeculator):
            return None
        return speculator.model

    def reload_weights(self, *args, **kwargs) -> None:
        # TODO(Wentao): 完全迁移到 v2 后改用完整实现而非导入。
        # 借用 v1 运行器的重载实现。
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.reload_weights(self, *args, **kwargs)  # type: ignore[arg-type]

    def update_config(self, *args, **kwargs) -> None:
        # TODO(Wentao): 完全迁移到 v2 后改用完整实现而非导入。
        # 借用 v1 运行器的配置更新实现。
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.update_config(self, *args, **kwargs)  # type: ignore[arg-type]

        # v2 通过 self.vllm_config 读取配置(如 load_model 中),
        # 因此需与 v1 辅助函数刚替换的属性保持同步。
        self.vllm_config.model_config = self.model_config
        self.vllm_config.load_config = self.load_config

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # 缓存默认 CUDA 流以避免查找开销。
        return torch.cuda.current_stream(self.device)

    def get_kv_cache_spec(self):
        # 返回 KV cache 规格;encoder-only 模型为空。
        if self.is_encoder_only:
            return {}
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        # 初始化 KV cache:构建块表、注意力后端、CUDA graph 管理器等。
        # 深拷贝配置避免外部修改。
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        # 块表需要覆盖的最大长度。
        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # 交叉注意力的块表需要索引 encoder token,
            # 其数量可能超过 decoder 的 max_model_len。
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        # 各 KV cache 组的块大小列表。
        block_sizes = []
        # 每组每请求最大块数列表。
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            # 取该组的 KV cache 规格。
            spec = kv_cache_group.kv_cache_spec
            # 记录块大小。
            block_sizes.append(spec.block_size)
            # 使用 DCP 时每个请求的 KV cache 分片在不同 rank 上,
            # 当前 rank 的一个块覆盖全局(未分片)序列的
            # block_size * cp_size 个 token。
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )
            # Mamba/Hybrid 模型的 KV cache 需要为投机 token 额外的块。
            if isinstance(spec, MambaSpec):
                # 前缀缓存开启时保留原块数否则为 1,再加投机块数。
                max_num_blocks = (
                    max_num_blocks if self.cache_config.enable_prefix_caching else 1
                ) + spec.num_speculative_blocks
                # Mamba 组不做 token 对齐地计算行宽。
                max_num_blocks = get_block_table_width(
                    max_num_blocks, spec.block_size, token_alignment=None
                )
            else:
                # 常规 attention 组按 128 token 对齐计算行宽。
                max_num_blocks = get_block_table_width(max_num_blocks, spec.block_size)
            # 记录该组的最大块数。
            max_num_blocks_per_group.append(max_num_blocks)

        # 初始化注意力后端,得到组、CG 支持与 kernel 块大小。
        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        # 按模型状态提供的额外约束收窄 CG 支持范围。
        attn_cg_support = attn_cg_support.narrow(
            *self.model_state.get_additional_cg_support()
        )
        # 创建块表管理器。
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        # 需要时构建 PCP 管理器。
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.req_states,
            self.block_tables,
            cls=self.pcp_manager_cls,
        )
        # 初始化 Mamba SSU 后端。
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        # 解析最终 CUDA graph 模式与捕获尺寸。
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.decode_query_len,
            use_v2_model_runner=True,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
        )
        # 创建 CUDA graph 管理器。
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            lora_capture_cases=self.lora_capture_cases,
        )
        # 检查注意力 CP 兼容性。
        check_attention_cp_compatibility(self.vllm_config)
        if isinstance(self.speculator, DraftModelSpeculator):
            # HACK(woosuk): 为草稿模型设置注意力相关组件。
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            # 在 set_attn 之后调用,使投机器能按自己的注意力支持
            # 确定 CUDA graph 模式。
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        # KV cache 张量列表。
        self.kv_caches: list[torch.Tensor] = []
        # 分配 KV cache 并返回层名 -> 张量的字典。
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_groups,
            self.device,
            self.cache_config.cache_dtype,
            self.kernel_block_sizes,
            self.vllm_config,
        )
        # 据配置创建 KV 连接器。
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _init_kv_zero_meta(self) -> None:
        """构建 KV 块清零元数据;由 gpu_worker 调用。"""
        # 创建 KV 块清零器(惰性初始化)。
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
        )

    @torch.inference_mode()
    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        # 本 dummy 步的 token 数。
        num_tokens: int,
        *args,
        # 是否跳过注意力元数据准备。
        skip_attn: bool = False,
        # 是否为均匀解码(每请求 token 数相同)。
        uniform_decode: bool = False,
        # 是否跳过 EPLB 步进。
        skip_eplb: bool = False,
        # 是否为显存 profiling 用途。
        is_profile: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.is_encoder_only:
            # encoder-only 模型无需 dummy 前向,返回空张量。
            empty = torch.empty(0, device=self.device)
            return empty, empty
        if skip_attn and not is_profile:
            # skip_attn 只允许在初始显存 profiling 时使用。
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        # 构造 dummy 调度器输出。
        # 请求数取 token 数与上限的较小值。
        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            # HACK(lucas): 目前 worker 在 MRV1 与 MRV2 间共享,
            # 且 MTP 投机解码希望 dummy 运行用 1+num_speculative_tokens,
            # 因此这里取 max;最终可能会在 worker 中改变,
            # 见 https://github.com/vllm-project/vllm/pull/35243
            num_tokens = max(num_tokens, self.decode_query_len)
            # 均匀解码:每请求 token 数相同。
            num_reqs = num_tokens // self.decode_query_len
            # 断言整除。
            assert num_tokens % self.decode_query_len == 0
        # 均匀分摊余数,确保无 dummy 请求超过
        # ceil(num_tokens / num_reqs) <= max_model_len 个 token。
        num_tokens_per_request = [
            num_tokens // num_reqs + (i >= num_reqs - num_tokens % num_reqs)
            for i in range(num_reqs)
        ]

        # 断言分配的 token 总数正确。
        assert sum(num_tokens_per_request) == num_tokens
        # 构造请求 id -> 调度 token 数映射。
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        # 创建空的调度器输出并填充调度信息。
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # dummy 运行期间禁用一切 KV 连接器行为。
        self.kv_connector.set_disabled(True)

        # 取 dummy 运行所需的中间张量。
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            # 非首个 PP rank 需要切出本批大小的中间张量。
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        # LoRA 配置存在时取最大 LoRA 数。
        max_loras = self.lora_config.max_loras if self.lora_config is not None else 0
        # 在 LoRA dummy 上下文中执行模型。
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens=np.array(num_tokens_per_request, dtype=np.int32),
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=max_loras,
        ):
            # 执行模型(dummy_run 标记)。
            self.execute_model(
                dummy_scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                is_profile=is_profile,
            )
        # 恢复 KV 连接器。
        self.kv_connector.set_disabled(False)

        # 非最后的 PP rank 不产生采样输出。
        if not self.is_last_pp_rank:
            return None, None

        # 断言 execute_model 状态存在。
        assert self.execute_model_state is not None
        # 取出输入批次、注意力元数据、slot 映射、隐藏状态等。
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        # 清空执行状态。
        self.execute_model_state = None

        # dummy 运行 eagle 投机器的 propose 以确保 DP/EP 同步。
        if self.speculator is not None:
            # 断言采样器存在。
            assert self.sampler is not None
            # 多模态输入占位。
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
            if self.speculator.supports_mm_inputs:
                # 构造形状正确的空 mm 输入。
                mm_inputs = (
                    [],
                    torch.zeros(
                        input_batch.num_tokens,
                        dtype=torch.bool,
                        device="cpu",
                    ),
                )

            # 让目标模型覆盖喂给起草器的隐藏状态
            # (如 DeepSeek V4 MTP 需要 hc_head 前的残差)。
            # 目标返回按 max_num_batched_tokens 定型的持久缓冲;
            # 切到 propose() 期望的活动 token 数。
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                # 取目标模型提供的特殊隐藏状态。
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            # 执行投机器 propose(dummy 参数)。
            self.speculator.propose(
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings_by_layer,
                last_hidden_states=spec_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=torch.ones(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                num_rejected=torch.zeros(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                last_sampled=self.req_states.last_sampled_tokens,
                next_prefill_tokens=self.req_states.next_prefill_tokens,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seeds=self.sampler.sampling_states.seeds.gpu,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )

        # 断言隐藏状态存在(最后的 PP rank 总是有)。
        assert hidden_states is not None
        # 按 logits 索引切出采样用隐藏状态。
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        # 返回全部隐藏状态与采样用隐藏状态。
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        # dummy 运行采样器,预热/统计显存。
        # 请求数取隐藏状态首维。
        num_reqs = hidden_states.shape[0]
        # 计算 logits。
        logits = self.model.compute_logits(hidden_states)
        # 构造 dummy 输入批次。
        dummy_input_batch = InputBatch.make_dummy(
            num_reqs, num_reqs, self.input_buffers
        )

        # 注意(woosuk): 初始显存 profiling 时,采样器可能跳过
        # top_k、top_p 与 logprobs,使用的 GPU 显存比实际执行时少。
        # 断言采样器存在并执行。
        assert self.sampler is not None
        self.sampler(logits, dummy_input_batch)

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        # dummy 运行池化器。
        # 断言池化运行器存在并执行。
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self) -> None:
        # 显存 profiling 运行:用最大批量跑一遍模型与采样器/池化器。
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # 支持多模态且为首个 PP rank 时,profile 编码器。
            # 取多模态配置。
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and not mm_config.skip_mm_profiling:
                # 未跳过 mm profiling 时构建预算与 dummy 输入。
                mm_budget = MultiModalBudget(
                    self.vllm_config,
                    self.mm_registry,
                    enable_cache=False,
                )
                dummy_mm_inputs = get_dummy_encoder_profile_inputs(
                    self.mm_registry,
                    mm_budget,
                )
                # profile 编码器缓存占用。
                self.model_state.encoder_runner.profile_encoder_cache(
                    dummy_mm_inputs, mm_budget
                )

        if self.is_encoder_only:
            # encoder-only:同步、重置编码器缓存并返回。
            torch.accelerator.synchronize()
            self.reset_encoder_cache()
            gc.collect()
            return

        # 以最大 token 数运行 dummy 前向(跳过注意力)。
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        # 仅在最后 PP rank 运行采样器/池化器(其余 rank 返回 None)。
        if self.is_last_pp_rank:
            # 断言采样隐藏状态存在。
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                # 生成模型:dummy 运行采样器。
                self._dummy_sampler_run(sample_hidden_states)
            else:
                # 池化模型:dummy 运行池化器。
                self._dummy_pooler_run(hidden_states)

        # 同步并释放临时张量。
        torch.accelerator.synchronize()
        del hidden_states, sample_hidden_states
        # 重置编码器缓存并触发 GC。
        self.reset_encoder_cache()
        gc.collect()

    def post_kv_cache_wake_up(self) -> None:
        # KV cache 唤醒后(如 offloading 恢复)初始化块表布局张量。
        self.block_tables.init_block_table_layout_tensors()

    def reset_mm_cache(self) -> None:
        # 重置多模态缓存。
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        # 重置编码器缓存。
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()
        if self.pooling_runner is not None:
            # 池化运行器存在时同时清空。
            self.pooling_runner.clear()

    def profile_cudagraph_memory(self) -> int:
        # 注意(woosuk): 是否保留此 API 待定。
        return 0

    @torch.inference_mode()
    def capture_model(self) -> int:
        # 捕获 CUDA graph,返回占用的显存字节数。
        if self.is_encoder_only:
            # encoder-only 无需捕获。
            return 0

        # 断言 CUDA graph 管理器存在。
        assert self.cudagraph_manager is not None
        if not self.cudagraph_manager.needs_capture():
            # 无需捕获(模式为 NONE)时警告并返回。
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        # 计数器自增。
        compilation_counter.num_gpu_runner_capture_triggers += 1

        # 记录开始时间。
        start_time = time.perf_counter()
        # 触发 GC 与清空缓存以获得准确基线。
        gc.collect()
        torch.accelerator.empty_cache()
        # 记录捕获前空闲显存。
        start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        # 在 dummy LoRA 设置上下文中捕获模型与投机器的图。
        with self.maybe_setup_dummy_loras(self.lora_config):
            # 执行 CUDA graph 捕获。
            self.cudagraph_manager.capture(
                self.model,
                self.model_state,
                self.input_buffers,
                self.intermediate_tensors,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                has_lora=self.lora_config is not None,
                use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
                lora_capture_hook=create_lora_capture_hook(self.lora_config, self),
            )
            if self.speculator is not None:
                # 投机器也捕获自己的图。
                self.speculator.capture()

        # 记录结束时间与空闲显存,计算耗时与占用。
        end_time = time.perf_counter()
        end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # 这通常需要 5~20 秒。
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        # 返回 CUDA graph 占用的显存。
        return cuda_graph_size

    def _remove_request(self, req_id: str) -> bool:
        # 必须先调用 model_state.remove_request 再调用
        # req_states.remove_request,使 model_state 仍能查询槽位索引。
        # 从模型状态移除请求。
        self.model_state.remove_request(req_id)
        # 从请求状态移除并拿到索引。
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            # 请求不存在,返回 False。
            return False
        if self.pooling_runner is not None:
            # 池化运行器存在则移除其登记。
            self.pooling_runner.remove_request(req_idx)
        if self.pp_handler is not None:
            # 通知 PP 处理器索引已释放。
            self.pp_handler.on_req_idx_freed(req_idx)
        if self.encoder_cache is not None:
            # 编码器缓存存在则移除其多模态特征。
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            # prompt logprobs 工作器存在则移除登记。
            self.prompt_logprobs_worker.remove_request(req_id)
        # 从 LoRA 状态移除。
        self.lora_state.remove_request(req_id)
        # 返回移除成功。
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        # 处理本步完成的(与被抢占的)请求,清理其状态。
        # 取完成请求 id 集合。
        finished_req_ids = scheduler_output.finished_req_ids
        if self.pooling_runner is not None:
            # 被抢占的文档在重新调度前保留其 query 使用预留。
            self.pooling_runner.on_requests_finished(finished_req_ids)
        # 取被抢占请求 id 集合。
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            # 合并到完成集合。
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            # 逐个移除请求状态。
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        # 释放调度器标记的编码器缓存条目。
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                # 逐个释放编码器缓存。
                self.encoder_cache.free_encoder_cache(mm_hash)

    def update_pp_decode_requests(self):
        # 对非最后 PP rank:用 pp_size 步之前调度的那一步的采样输出
        # 更新解码请求。
        if self.pp_handler is not None:
            # 取上一步的采样输出。
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                # 有输出则执行采样后处理。
                self.postprocess_sampled(**outputs)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        # 添加本步调度的新请求到各组件状态。
        for new_req_data in scheduler_output.scheduled_new_reqs:
            # 断言 prompt token ids 存在。
            assert new_req_data.prompt_token_ids is not None
            # 断言 prefill token ids 存在。
            assert new_req_data.prefill_token_ids is not None
            # 取请求 id。
            req_id = new_req_data.req_id

            # 流式输入更新:请求可能来自之前的 chunk。
            # 先移除旧状态,以便下方用更新后的 prompt_token_ids
            # 与 mm_features 干净地重新添加。
            self._remove_request(req_id)

            # 计算 prompt 长度。
            prompt_len = len(new_req_data.prompt_token_ids)
            # 取采样参数。
            sampling_params = new_req_data.sampling_params
            # 添加请求基础状态(分阶段写入)。
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
            # 查询该请求的批次索引。
            req_index = self.req_states.req_id_to_index[req_id]

            if self.pooling_runner is not None:
                # 池化模型:登记池化参数。
                assert new_req_data.pooling_params is not None
                self.pooling_runner.add_request(
                    req_id,
                    req_index,
                    new_req_data.pooling_params,
                    new_req_data.prompt_token_ids,
                )

            if self.encoder_cache is not None:
                # 登记该请求的多模态特征。
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            # 模型状态添加请求。
            self.model_state.add_request(req_index, new_req_data)
            # 写入(覆盖)该请求的块 id。
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            # 登记 LoRA 请求。
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                # 最后 PP rank 且有采样参数时登记采样相关信息。
                # 断言采样器存在。
                assert self.sampler is not None
                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )
                # 断言 prompt logprobs 工作器存在。
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            # 有新请求时,应用各组件的分阶段写入。
            self.req_states.apply_staged_writes()
            self.model_state.apply_staged_writes()
        if self.sampler is not None:
            # 应用采样器的分阶段写入。
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # 为已有请求添加新块并更新已计算 token 数。
        # 取缓存的已调度请求信息。
        reqs = scheduler_output.scheduled_cached_reqs
        # 取已计算 token 数的 numpy 视图。
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        # 遍历每个请求更新。
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            # 查询批次索引。
            req_index = self.req_states.req_id_to_index[req_id]
            # 写入新的已计算 token 数。
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
                # 追加新分配的块 id(不覆盖)。
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # 更新 CPU 侧 prefill 已计算 token 数
        # (取 min(prefill 已计算, prompt 长度))。
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

        # 对新分配的缓存块清零 GPU 显存,
        # 防止残留 NaN/旧数据破坏注意力或 SSM 计算。
        if scheduler_output.new_block_ids_to_zero:
            # 断言清零器已初始化并执行清零。
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # 对部分前缀缓存命中执行写时复制的块拷贝,
        # 在新块清零之后、前向读取之前进行。
        if scheduler_output.kv_cache_block_copies:
            # 原地拷贝 KV cache 块。
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> InputBatch:
        # 准备本批所有输入并填入持久输入缓冲区,返回 InputBatch。
        # 本批实际调度 token 数。
        num_tokens = scheduler_output.total_num_scheduled_tokens
        # padding 后的 token 数(CUDA graph 尺寸)。
        num_tokens_after_padding = batch_desc.num_tokens
        # 断言至少有 token 要跑。
        assert num_tokens > 0
        if envs.VLLM_MOE_SKIP_PADDING:
            # 标记 cudagraph padding 的尾部行,
            # 使 kernel 在支持时可跳过这些行的计算。
            # 有效 token 行标记为非 padding。
            self.input_buffers.is_padding[:num_tokens].fill_(False)
            # padding 行标记为 True。
            self.input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(
                True
            )
        # 请求 id -> 本步调度 token 数映射。
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        # 请求数。
        num_reqs = len(num_tokens_per_req)

        # batch_idx -> req_id
        # 排序批次请求 id(解码在前)。
        req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)
        # 按排序后的顺序取每请求调度 token 数。
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        # 构建旧索引 -> 新批次索引 的映射。
        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        # 异步拷贝索引映射到 GPU。
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # 取每请求的草稿 token 数。
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # 无草稿 token 调度(常见情形)。
            # 草稿 token 总数为 0。
            total_num_draft_tokens = 0
            # logits 总数即请求数。
            total_num_logits = num_reqs
            # CPU 侧累计 logits 前缀和(0..n)。
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            # GPU 侧同形状前缀和。
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            # 展开后索引映射与原映射相同。
            expanded_idx_mapping = idx_mapping
            # 每请求的局部偏移全为 0。
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            # 有草稿 token:统计每请求数量。
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            # 每步新采样(奖励)token 数。
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            # 草稿 token 总数。
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            # logits 总数 = 请求数*奖励数 + 草稿总数。
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            # 每请求的 logits 数 = 草稿数 + 奖励数。
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            # 构建前缀和数组。
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            # 累计求和。
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            # 异步拷贝到 GPU。
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            # 最大扩展长度为解码 query 长度。
            max_expand_len = self.decode_query_len
            # 展开索引映射与局部偏移(每 logits 一行)。
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # 取 query_start_loc。
        # PIECEWISE 图无需请求 padding,此处为 None。
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        # 分配 query 起点前缀和数组。
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        # 累计各请求 token 数。
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # 为完整 CUDA graph 模式填充。
        # 某些注意力后端(如 FA3)要求 query_start_loc 非递减。
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        # 异步拷贝进输入缓冲区。
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        # 截断到 padding 后请求数。
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        # 取 GPU 侧切片。
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
        # 按索引映射取各请求 prompt 长度。
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        # 取 prefill 已计算 token 数视图。
        computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens
        # 按索引映射取各请求的值。
        num_computed_prefill_tokens_np = computed_prefill_tokens_np[idx_mapping_np]
        # 是否仍在 prefill:已计算 < prompt 长度。
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np

        # 存在 prefill 请求时取 prefill 输入 token。
        if np.any(is_prefilling_np):
            # 填充 input_ids 缓冲区的 prefill 段。
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # 准备位置与序列长度。
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        # 截取本批的序列长度。
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

        # DCP 本地序列长度默认 None。
        dcp_local_seq_lens = None
        if self.use_dcp:
            # 启用 DCP 时准备本地序列长度。
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            # 截取本批切片。
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # 部分 input token 直接读取自上一步采样 token 与草稿 token;
        # 同时得到用于采样的 logits 索引。
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
        )

        # seq_lens 的 CPU 上界;padding 条目保持为 0。
        # 取各请求已计算 token 数。
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        # 分配上界数组。
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        # 上界 = 已计算 + 本步调度数。
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        # 转为 torch 张量。
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        # 最大序列长度默认 None。
        max_seq_len_np = None
        if self.use_pp:
            # max_seq_len 只被 PP 的 compute_need_sampled_mask 使用。
            max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

        # prompt 长度默认 None。
        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens 只在 R-SWA(滑动窗口)情形使用。
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        # 组装 InputBatch。
        input_batch = InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            max_seq_len_np=max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
        )
        # 需要时对批次做 PCP 分区。
        return pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        # 准备注意力元数据:块表与 slot 映射。
        if self.pcp_manager is not None:
            # PCP 启用时由管理器处理。
            return self.pcp_manager.prepare_attn(input_batch)

        # 块表: num_kv_cache_groups x [num_reqs_padded, max_num_blocks]。
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        # slot 映射: [num_kv_cache_groups, num_tokens_padded]。
        # kernel 会把超出 num_tokens 的部分填充为 PAD_SLOT_ID。
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        # 返回块表与 slot 映射。
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        # 准备 dummy 运行用的注意力元数据。
        # 生成 dummy 块表。
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        # 需要时取 PCP dummy slot 映射。
        slot_mappings = pcp.maybe_get_pcp_dummy_slot_mappings(
            self.pcp_manager, self.block_tables, input_batch.num_tokens
        )
        # 返回块表与 slot 映射。
        return block_tables, slot_mappings

    def sample(
        self,
        # 模型输出的隐藏状态。
        hidden_states: torch.Tensor,
        # 当前输入批次。
        input_batch: InputBatch,
        # 语法(结构化输出)结果,可为 None。
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        # 采样 token:计算 logits -> 应用语法掩码 -> 采样/拒绝采样。
        # 按 logits 索引切出采样用隐藏状态。
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        # 由模型计算 logits。
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # 有结构化输出请求时,原地应用语法位掩码。
            # 断言结构化输出工作器存在。
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            # 无投机解码:直接使用普通采样器。
            # 断言采样器存在并采样。
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # 投机解码的拒绝采样。
            # 断言拒绝采样器与投机器存在。
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # 概率化拒绝采样需要草稿 logits。
                self.speculator.draft_logits,
            )

        # 返回采样输出、接受数与拒绝数。
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        # 索引映射(可含 -1 表示被掩码条目)。
        idx_mapping: torch.Tensor,
        # 采样得到的 token ids。
        sampled_tokens: torch.Tensor,
        # 每请求接受 token 数。
        num_sampled: torch.Tensor,
        # 每请求拒绝 token 数。
        num_rejected: torch.Tensor,
        # query 起点前缀和(可选)。
        query_start_loc: torch.Tensor | None = None,
    ) -> None:
        # 采样后处理:更新请求状态。
        # 最后 PP rank 上取惩罚直方图统计。
        if self.is_last_pp_rank:
            # 断言采样器存在。
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            # 其他 rank 无惩罚统计。
            output_bin_counts = None
        # 批量更新 token 状态(计算 token 数、最后采样、惩罚等)。
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )

        # 更新模型状态(如 Mamba 递归状态)。
        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )

    @torch.inference_mode()
    def execute_model(
        self,
        # 调度器输出(本步调度计划)。
        scheduler_output: SchedulerOutput,
        # 上游 PP rank 传来的中间张量(可选)。
        intermediate_tensors: IntermediateTensors | None = None,
        # 是否为 dummy 运行。
        dummy_run: bool = False,
        # dummy 运行时是否跳过注意力准备。
        skip_attn_for_dummy_run: bool = False,
        # 是否为显存 profiling 用途。
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if not dummy_run:
            # 常规运行:先更新请求状态。
            # 更新 PP 解码请求(延迟一步的采样结果)。
            self.update_pp_decode_requests()
            # 处理完成/被抢占请求。
            self.finish_requests(scheduler_output)
            # 释放编码器缓存条目。
            self.free_states(scheduler_output)
            # 添加新请求。
            self.add_requests(scheduler_output)
            # 更新已有请求。
            self.update_requests(scheduler_output)
            # 应用块表的分阶段写入。
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # 本步没有 token 需要运行模型。
                # 由 KV 连接器做空转发并返回。
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # 获取批次描述符并在 DP 各 rank 间同步。
        # 请求数。
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        # 调度 token 总数。
        num_toks = scheduler_output.total_num_scheduled_tokens
        # 最大单请求 query 长度。
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        # 判断是否为均匀 token 计数(全解码批)。
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)

        # 活动 LoRA 数默认 0。
        num_active_loras = 0
        if self.lora_config:
            # 取本批请求 id 列表。
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            # 统计分发所需的活动 LoRA 数。
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config, self.lora_state, req_ids, dummy_run
            )

        # 是否跳过编译路径。
        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Whisper 等 encoder-decoder 模型在调度了编码器输入时应
            # 以 eager/非编译方式运行,因为本步会用动态编码器输出
            # 更新交叉注意力缓存。
            skip_compiled = True

        # 分发 CUDA graph 描述符并同步 DP 各 rank。
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            need_eager=is_profile or skip_compiled,
            num_active_loras=num_active_loras,
        )

        if batch_desc.num_tokens == 0:
            # 所有 DP rank 均无 token 可运行。
            # 由 KV 连接器做空转发并返回。
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # 常见路径。
            # 准备全部输入并拷入输入缓冲区。
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            # 准备块表与 slot 映射。
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            # Mamba "对齐"预拷贝:在前向之前跨块边界迁移递归状态。
            # 仅对真实批次运行,且在 model_state.prepare_attn 读取
            # num_accepted_tokens 之前,使边界重置对注意力元数据可见。
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )

            if self.lora_config:
                # 激活 LoRA 适配器。
                # 构建 LoRA 输入。
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                # 设置活动 LoRA。
                self._set_active_loras(*lora_inputs)
        else:
            # 无实际 token。用于 DP 或显存 profiling 的 dummy 运行。
            # 构造 dummy 输入批次。
            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
            )
            if not skip_attn_for_dummy_run:
                # 准备 dummy 注意力元数据。
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                # 跳过注意力:FULL 模式下不允许。
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

        # 注意力元数据与按层 slot 映射默认 None。
        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            # 需要准备注意力元数据。
            # 断言 slot 映射存在。
            assert slot_mappings is not None
            # 按层构建 slot 映射字典。
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            # 断言块表存在。
            assert block_tables is not None
            # 由模型状态准备注意力元数据。
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
                # FULL 重放读取捕获时的元数据缓冲。改从清零的 dummy
                # 块表重新暂存,而不是保留上一个真实批次的索引。
                for_capture=dummy_run and batch_desc.cg_mode == CUDAGraphMode.FULL,
            )

        # 输入 token ids 默认取自批次缓冲。
        input_ids = input_batch.input_ids
        # 输入嵌入默认 None。
        inputs_embeds = None
        # EC 连接器输出默认 None。
        ec_connector_output = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # 运行 MM 编码器(如需)并获取多模态嵌入。
            # 仅首个 PP rank 准备多模态嵌入。
            if dummy_run:
                # dummy 运行:获取形状正确的 mm 嵌入以支持编译模型。
                inputs_embeds = self.model_state.dummy_inputs_embeds(
                    input_batch.num_tokens_after_padding
                )
            else:
                # 取本步调度的编码器输入。
                scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                if self.lora_config is not None:
                    # LoRA 启用时激活多模态 LoRA。
                    set_active_mm_loras(
                        model=self.model,
                        lora_manager=self.lora_manager,
                        encoder_cache=self.encoder_cache,
                        req_id_to_index=self.req_states.req_id_to_index,
                        lora_state=self.lora_state,
                        scheduled_encoder_inputs=scheduled_encoder_inputs,
                    )
                # 在 EC 连接器上下文中获取 mm 嵌入。
                with self.ec_connector.maybe_get_output(
                    scheduler_output
                ) as ec_connector_output:
                    inputs_embeds = self.model_state.get_mm_embeddings(
                        scheduled_encoder_inputs, input_batch, self.req_states
                    )
            if inputs_embeds is not None and not self.model.requires_raw_input_tokens:
                # 模型直接消费嵌入时无需原始 token ids。
                input_ids = None

        if self.is_encoder_only:
            # encoder-only 模型:返回空输出并携带 EC 连接器输出。
            output = make_empty_encoder_model_runner_output(scheduler_output)
            output.ec_connector_output = ec_connector_output
            return output

        # 组装模型输入字典。
        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
            # 注意: prepare_inputs 返回的值会覆盖上面的默认值。
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            # 非首个 PP rank 的更新。
            # 无需原始输入,置空。
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            # 准备中间张量。
            # 断言上游中间张量与本地持久缓冲存在。
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            # 本批 token 数。
            n = input_batch.num_tokens_after_padding
            # 切片(或拷贝收到的数据)到持久缓冲同地址。
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }
            # 包装为 IntermediateTensors。
            model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
            # 释放上游引用。
            del intermediate_tensors

        # 更新 EPLB 元数据。
        self.eplb.prepare_forward(self.model_config, input_batch.num_tokens)

        # 运行模型。
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # FULL 模式使用显式 cudagraph 重放。
            # 注意(woosuk): 这里无需传入输入张量,
            # 因为它们已被拷贝到 CUDA graph 输入缓冲区。
            # 断言图管理器存在。
            assert self.cudagraph_manager is not None
            # 前向前的 KV 连接器操作。
            self.kv_connector.pre_forward(scheduler_output)
            # 重放完整图。
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # piecewise 与 eager 模式直接调用 model()。
            # 构建批次描述符。
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )

            # 在前向上下文中执行(记录注意力元数据等)。
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=input_batch.is_padding,
            ):
                # 前向前的 KV 连接器操作。
                self.kv_connector.pre_forward(scheduler_output)
                if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # 运行 PIECEWISE 图(编译 PW cudagraph 或可打断
                    # cudagraph,由 run_pw_graph 内部选择)。图管理器
                    # 存在后 cg_mode 才会是 PIECEWISE。
                    # 断言图管理器存在。
                    assert self.cudagraph_manager is not None
                    model_output = self.cudagraph_manager.run_pw_graph(
                        self.model, model_inputs
                    )
                else:
                    # Eager(NONE):直接调用原始模型。
                    model_output = self.model(**model_inputs)

        if self.is_last_pp_rank:
            # 最后 PP rank:处理最终隐藏状态。
            if self.use_aux_hidden_state_outputs:
                # 需要辅助隐藏状态:输出为元组。
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
            else:
                # 常规:输出为张量。
                assert isinstance(model_output, torch.Tensor)
                hidden_states = model_output
                aux_hidden_states = None
            # 输出中间张量为 None。
            output_intermediate_tensors = None
        else:
            # 非最后 rank:输出为中间张量。
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        # 路由专家张量默认 None。
        routed_experts = None
        if not dummy_run and (capturer := self.routed_experts_capturer) is not None:
            # 非 dummy 且捕获器存在时提取路由专家。
            # 断言 slot 映射存在。
            assert slot_mappings is not None
            routed_experts = capturer.get_routed_experts(slot_mappings, num_toks)

        # 取完成请求 id 集合。
        finished_req_ids = scheduler_output.finished_req_ids
        # 保存执行状态,供后续 sample_tokens/pool 使用。
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
            routed_experts=routed_experts,
        )

        if not self.is_last_pp_rank:
            # 非最后 PP rank:返回待发送的 IntermediateTensors。
            return output_intermediate_tensors
        # 最后 rank 返回 None(采样在 sample_tokens 中进行)。
        return None

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        # 采样 token 阶段:在 execute_model 之后调用,与采样相关的
        # 集合通信与后处理都在这里进行。
        if self.execute_model_state is None:
            # 之前的 execute_model 调用必定失败了。
            return None

        # 取出保存的执行状态。
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        routed_experts = self.execute_model_state.routed_experts
        # 清空执行状态。
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # 非最后 PP rank: hidden_states 为 None,因为该 rank 产出的是
            # IntermediateTensors 而非最终隐藏状态。接收最后 rank 广播的
            # 采样 token 并更新本地状态。
            # 断言 PP 处理器存在并接收。
            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # 在此乐观地更新整个批次的 num_computed_tokens;
            # 若有拒绝采样,会在 update_requests 中调整。
            self.postprocess_num_computed_tokens(input_batch)
            if not all_decode_next:
                # 可能包含非最终 prefill 块,它们会在紧接着的下一步
                # (而非 pp_size 步后)被调度。
                self.model_state.postprocess_state(input_batch.idx_mapping, 0)

            # 步后的 KV 连接器相关操作。
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            # 返回仅含连接器输出的结果。
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # 最后 rank:采样 token。
        # 需要时从 PCP 分区恢复以进行采样。
        hidden_states, input_batch = pcp.maybe_restore_pcp_for_sampling(
            self.pcp_manager, hidden_states, input_batch
        )

        # 执行采样,得到输出与接受/拒绝数。
        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.pp_handler is not None:
            # 广播给非最后 PP rank(处理投机解码多 token)。
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        # 断言 prompt logprobs 工作器存在。
        assert self.prompt_logprobs_worker is not None
        # 计算 prompt logprobs。
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )

        # 准备模型运行器输出。
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # 注意(woosuk): req_id_to_index 在此运行器中未使用,
            # 仅为兼容现有模型运行器与调度器。
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )
        # 在此启动异步输出拷贝,使其与投机器提案重叠。
        # 构建异步输出对象。
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            check_ep_fault=self.check_ep_fault,
            routed_experts=routed_experts,
        )

        # 投机器的多模态输入占位。
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if self.speculator is not None and self.speculator.supports_mm_inputs:
            # 取缓存的多模态嵌入供草稿前向使用。
            # 注意: 在此处进行,因为后处理会更新 num_computed_prefill_tokens。
            # EAGLE/MTP 起草器读取比目标超前一个位置。
            # TODO(TheEpicDolphin): 多模块 MTP 时为所有投机步收集 mm 嵌入。
            mm_inputs = self.model_state.gather_mm_embeddings(
                input_batch, draft_lookahead=1
            )

        # 后处理结果并更新请求状态。
        # 注意: 特意在创建 AsyncOutput 之后进行,确保 copy_event
        # 在调用后处理前已记录。此顺序可略微降低延迟,
        # 因为异步 D2H 拷贝无需等待后处理完成。
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        if self.speculator is not None:
            # 投机器存在时起草下一步 token。
            # 断言采样器存在。
            assert self.sampler is not None
            # 让目标模型覆盖喂给起草器的隐藏状态
            # (如 DeepSeek V4 MTP 需要 hc_head 前的残差)。
            # 目标返回按 max_num_batched_tokens 定型的持久缓冲;
            # 切到 propose() 期望的活动 token 数。
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                # 取目标模型提供的特殊隐藏状态。
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            # 执行投机器提案。
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                spec_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                mm_inputs=mm_inputs,
            )
            # 把草稿 token 写入请求状态。
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

        if self.num_speculative_steps > 0:
            # 投机解码与扩散 LLM 都使用草稿 token,
            # 但后者没有投机器(即 self.speculator 为 None)。
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )

        # 步后的 KV 连接器相关操作。
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        # 挂载到输出。
        model_runner_output.kv_connector_output = kv_connector_output

        # 返回异步输出。
        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # 取出并清空待上报的草稿 token ids。
        return self.draft_tokens_handler.get_draft_tokens()

    @torch.inference_mode()
    @step_eplb_after()
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None:
        # 池化阶段:在 execute_model 之后调用。
        if self.execute_model_state is None:
            # 之前的 execute_model 调用必定失败了。
            return None

        # 取出保存的执行状态。
        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        # 清空执行状态。
        self.execute_model_state = None

        # 步后的 KV 连接器相关操作。
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)

        if not self.is_last_pp_rank:
            # 非最后 rank:更新已计算 token 数并返回。
            self.postprocess_num_computed_tokens(input_batch)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # 断言池化运行器存在并执行池化。
        assert self.pooling_runner is not None
        pooler_output, finished_mask = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )

        # 构建模型运行器输出。
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
        )
        # 构建异步池化输出(异步 D2H 拷贝)。
        async_output = AsyncPoolingOutput(
            model_runner_output=model_runner_output,
            pooler_output=pooler_output,
            finished_mask=finished_mask,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        # 更新已计算 token 数。
        self.postprocess_num_computed_tokens(input_batch)
        # 返回异步输出。
        return async_output

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        # 更新已计算 token 数(解码步后处理)。
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    def shutdown(self) -> None:
        """释放 GPU 张量(模型权重、KV cache、工作区),
        以便同进程运行时显存可回收。"""
        # 同步设备。
        torch.accelerator.synchronize()
        if hasattr(self, "kv_caches"):
            # 清空 KV cache 列表。
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            # 清空注意力组。
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            # 删除 KV cache 配置。
            del self.kv_cache_config
        # 关机前的通用释放。
        free_before_shutdown(self.vllm_config)
        if hasattr(self, "model_state"):
            # 删除模型状态。
            del self.model_state
        if getattr(self, "speculator", None) is not None:
            # 置空投机器。
            self.speculator = None
        if hasattr(self, "model"):
            # 删除模型。
            del self.model

        # 触发 GC 并清空缓存。
        gc.collect()
        torch.accelerator.empty_cache()
        logger.debug("Cleaned up model weights, KV caches, and workspace")

    ########### EPLB 方法开始 ###########
    @property
    def eplb_state(self):
        # 读取 EPLB 状态。
        return self.eplb.state

    @eplb_state.setter
    def eplb_state(self, state) -> None:
        # 写入 EPLB 状态。
        self.eplb.state = state

    @property
    def eep_eplb_suppressed(self) -> bool:
        # 读取 EPLB 抑制标志。
        return self.eplb.suppressed

    @eep_eplb_suppressed.setter
    def eep_eplb_suppressed(self, suppressed: bool) -> None:
        # 写入 EPLB 抑制标志。
        self.eplb.suppressed = suppressed

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        # 按给定映射设置 EPLB。
        self.eplb.setup_from_mapping(
            self.model,
            self.model_config,
            expanded_physical_to_logical,
            old_num_physical_experts,
        )

    ########### EPLB 方法结束 ###########

    # 树外硬件运行器可选择 PCP 管理器类。
    @property
    def pcp_manager_cls(self) -> type[pcp.PCPManager]:
        # 返回 PCP 管理器类。
        return pcp.PCPManager


class ExecuteModelState(NamedTuple):
    # execute_model 与 sample_tokens/pool 之间传递的状态。
    # 当前输入批次。
    input_batch: InputBatch
    # 注意力元数据。
    attn_metadata: dict[str, Any] | None
    # 按层 slot 映射。
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    # 最终隐藏状态。
    hidden_states: torch.Tensor | None
    # 辅助隐藏状态列表(EAGLE3 等)。
    aux_hidden_states: list[torch.Tensor] | None
    # 已完成请求 id 集合。
    finished_req_ids: set[str]
    # 路由专家张量。
    routed_experts: RoutedExpertsTensors | None


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int], decode_query_len: int
) -> list[str]:
    # 排序顺序: 解码 -> 短扩展 -> prefill。
    # split_decodes_and_prefills 依赖均匀解码
    # (query_len == decode_query_len)排在前面。
    # 排序键: 非解码请求排后,再按 token 数升序。
    key = lambda r: ((num := num_tokens_per_req[r]) != decode_query_len, num)
    # 返回排序后的请求 id 列表。
    return sorted(num_tokens_per_req, key=key)
