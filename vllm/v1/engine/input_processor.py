# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import time  # time：时间模块（生成到达时间戳）
from collections.abc import Mapping  # Mapping：映射类型（trace_headers 用）
from typing import Any, Literal  # Any：通用类型；Literal：字面量类型标注

import vllm.envs as envs  # vllm.envs：vLLM 环境变量配置
from vllm.config import VllmConfig  # vLLM 全局配置
from vllm.inputs import (
    EngineInput,  # 引擎输入类型（预处理后的输入）
    PromptType,  # 原始 prompt 类型（文本/token IDs/embeds 等）
    SingletonInput,  # 单一输入类型（encoder 或 decoder 输入）
    split_enc_dec_input,  # 拆分 encoder-decoder 输入
)
from vllm.inputs.preprocess import InputPreprocessor  # 输入预处理器（tokenize 等）
from vllm.logger import init_logger  # 初始化 vLLM 日志记录器
from vllm.lora.request import LoRARequest  # LoRA 请求
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
# 多模态注册表（管理多模态处理器）
from vllm.multimodal.encoder_budget import MultiModalBudget  # 多模态编码器预算
from vllm.multimodal.inputs import MultiModalFeatureSpec  # 多模态特征规格
from vllm.multimodal.utils import argsort_mm_positions  # 按位置排序多模态索引
from vllm.platforms import current_platform  # 当前平台抽象
from vllm.pooling_params import PoolingParams  # 池化参数（embedding 任务）
from vllm.renderers import BaseRenderer, renderer_from_config
# renderer：tokenizer + 多模态处理封装
from vllm.sampling_params import SamplingParams  # 采样参数（生成任务）
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS, SupportedTask
# 任务分类：生成任务、池化任务、支持的任务类型
from vllm.tokenizers import TokenizerLike  # tokenizer 接口类型
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid
# 计算 token/embedding 序列长度；生成随机 UUID
from vllm.utils.jsontree import json_iter_leaves  # 遍历 JSON 树叶子节点
from vllm.v1.engine import EngineCoreRequest  # 引擎核心请求类型

logger = init_logger(__name__)  # 模块级日志记录器


class InputProcessor:
    # 输入处理器：将原始输入转换为 EngineCoreRequest

    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置
        renderer: BaseRenderer | None = None,  # renderer（可选，默认从配置创建）
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表
    ) -> None:
        self.vllm_config = vllm_config  # 保存全局配置
        self.model_config = model_config = vllm_config.model_config  # 模型配置
        self.cache_config = vllm_config.cache_config  # 缓存配置
        self.lora_config = vllm_config.lora_config  # LoRA 配置
        self.scheduler_config = vllm_config.scheduler_config  # 调度器配置
        self.speculative_config = vllm_config.speculative_config
        # 投机解码（speculative decoding）配置
        self.structured_outputs_config = vllm_config.structured_outputs_config
        # 结构化输出配置
        self.observability_config = vllm_config.observability_config
        # 可观测性配置
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # 是否使用 V2 模型运行器

        self.generation_config_fields = model_config.try_get_generation_config()
        # 获取模型的 generation 配置字段（如 eos_token_id 等）

        self.renderer = renderer or renderer_from_config(vllm_config)
        # 使用传入的 renderer 或从配置创建

        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
        # 检查模型是否支持多模态输入
        self.mm_encoder_cache_size = 0  # 多模态编码器缓存大小初始化为 0
        self.skip_prompt_length_check = False  # 是否跳过 prompt 长度检查
        if self.supports_mm_inputs:
            # 如果支持多模态输入
            mm_budget = MultiModalBudget(vllm_config, mm_registry)
            # 创建多模态预算管理器
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size
            # 获取编码器缓存大小
            self.skip_prompt_length_check = (
                mm_budget.processor.info.skip_prompt_length_check
            )
            # 从处理器信息获取是否跳过长度检查
            mm_budget.reset_cache()  # Not used anymore
            # 重置预算缓存（已不再使用）

        self.input_preprocessor = InputPreprocessor(
            vllm_config,  # 配置
            renderer=renderer,  # renderer
            mm_registry=mm_registry,  # 多模态注册表
        )
        # 创建输入预处理器

        # Raw-prompt preprocessing (tokenization and multimodal processing)
        # is blocking, so async callers should run it on the renderer's
        # thread pool to keep their event loop responsive.
        self.process_inputs_async = make_async(
            self.process_inputs, executor=self.renderer._executor
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        # 属性：获取 tokenizer（可能为 None）
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        # 获取 tokenizer 实例（保证非 None）
        return self.renderer.get_tokenizer()

    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,  # 采样/池化参数
        supported_tasks: tuple[SupportedTask, ...],  # 模型支持的任务类型
    ) -> None:
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid."""
        # 如果 SamplingParams 或 PoolingParams 无效则抛出 ValueError
        if isinstance(params, SamplingParams):
            # 如果是采样参数（生成任务）
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            # 筛选支持的生成任务
            if not supported_generation_tasks:
                # 如果没有生成任务
                raise ValueError("This model does not support generation")
                # 抛出错误

            params.verify(
                # 验证采样参数
                self.model_config,  # 模型配置
                self.speculative_config,  # 投机解码配置
                self.structured_outputs_config,  # 结构化输出配置
                self.tokenizer,  # tokenizer
            )

            if params.thinking_token_budget is not None:
                # 如果设置了思考 token 预算（推理模型）
                if (
                    self.vllm_config.reasoning_config is None
                    or not self.vllm_config.reasoning_config.enabled
                ):
                    # 如果未配置推理配置
                    raise ValueError(
                        # 抛出错误
                        "thinking_token_budget is set but reasoning_config is "
                        "not configured. Please set --reasoning-parser "
                        "and/or --reasoning-config to use thinking_token_budget."
                    )
                if self.use_v2_model_runner:
                    # 如果使用 V2 模型运行器
                    raise ValueError(
                        # 抛出错误（V2 运行器暂不支持）
                        "thinking_token_budget is not yet supported by the V2 "
                        "model runner. Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 "
                        "to use thinking_token_budget."
                    )
        elif isinstance(params, PoolingParams):
            # 如果是池化参数（embedding 任务）
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            # 筛选支持的池化任务
            if not supported_pooling_tasks:
                # 如果没有池化任务
                raise ValueError("This model does not support pooling")
                # 抛出错误

            if params.task is None:
                # 如果未指定池化任务类型
                if "token_embed" in supported_pooling_tasks:
                    # 如果支持 token 嵌入
                    params.task = "token_embed"  # 默认选择 token_embed
                elif "token_classify" in supported_pooling_tasks:
                    # 如果支持 token 分类
                    params.task = "token_classify"  # 默认选择 token_classify
                elif "plugin" in supported_pooling_tasks:
                    # 如果支持插件
                    params.task = "plugin"  # 默认选择 plugin

            if params.task not in supported_pooling_tasks:
                # 如果任务类型不支持
                raise ValueError(
                    # 抛出错误
                    f"Unsupported task: {params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

            params.verify(self.model_config)  # 验证池化参数
        else:
            # 其他参数类型
            raise TypeError(
                # 抛出类型错误
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        # 验证 LoRA 请求
        if lora_request is None:
            # 如果没有 LoRA 请求
            return  # 直接返回

        # LoRA request passed in while LoRA is not enabled
        # 传入了 LoRA 请求但未启用 LoRA
        if not self.lora_config:
            # 如果未启用 LoRA
            raise ValueError(
                # 抛出错误
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            # 如果有 tokenizer
            logger.warning_once(
                # 记录一次性废弃警告
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    def _get_mm_identifier(
        self,
        mm_hash: str,  # 多模态内容哈希
        lora_request: LoRARequest | None,  # LoRA 请求（可选）
    ) -> str:
        """
        When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore, the mm_hash must be
        generated based on the LoRA request to prevent incorrect cache hits.
        """
        # 当 enable_tower_connector_lora 为 True 时，多模态嵌入随 LoRA 请求
        # 而变化。因此 mm_hash 必须基于 LoRA 请求生成，防止错误的缓存命中。
        if (
            lora_request is None  # 无 LoRA
            or self.lora_config is None  # 无 LoRA 配置
            or not self.lora_config.enable_tower_connector_lora
            # 未启用 tower connector LoRA
        ):
            return mm_hash  # 直接返回原始哈希
        return f"{lora_request.lora_name}:{mm_hash}"
        # 否则生成包含 LoRA 名称的唯一哈希

    def inject_into_mm_cache(
        self,
        mm_hashes: dict[str, list[str]],  # 多模态哈希字典
        mm_kwargs: dict[str, list],  # 多模态处理参数
    ) -> None:
        """Inject pre-processed mm_kwargs into the processor cache.

        Call this when mm_kwargs have already been through the HF processor
        externally (e.g. by a frontend that transfers pre-processed tensors
        to the backend).  This ensures MM cache hit rate metrics are reported
        accurately and avoids redundant processing on subsequent requests
        with the same images.

        Uses ``get_and_update_item()`` with an empty prompt_updates list,
        since token expansion has already been handled externally.
        """
        # 将预处理后的 mm_kwargs 注入处理器缓存。
        # 当 mm_kwargs 已在外部经过 HF 处理器时调用（例如前端将预处理后的
        # 张量传输到后端）。这确保多模态缓存命中率指标准确，并避免对
        # 相同图片的后续请求进行冗余处理。
        # 使用 get_and_update_item() 和空 prompt_updates 列表，
        # 因为 token 扩展已在外部处理。
        cache = self.renderer.mm_processor_cache  # 获取多模态处理器缓存
        if cache is None:
            # 如果没有缓存
            return  # 直接返回
        try:
            for modality, hashes in mm_hashes.items():
                # 遍历每个模态的哈希列表
                items = mm_kwargs.get(modality, [])  # 获取该模态的处理参数
                for i, mm_hash in enumerate(hashes):
                    # 遍历每个哈希
                    if i < len(items) and items[i] is not None:
                        # 如果索引有效且项目非空
                        # Insert into cache via get_and_update_item.
                        # Use the returned item (may be an address for SHM
                        # cache or the original item for LRU cache).
                        # 通过 get_and_update_item 插入缓存。
                        # 使用返回的项目（对于 SHM 缓存可能是地址，
                        # 对于 LRU 缓存可能是原始项目）。
                        items[i], _ = cache.get_and_update_item(
                            (items[i], []),  # (项目, 空 prompt 更新)
                            mm_hash,  # 缓存键
                        )
            # Update cache stats to reflect the externally processed items
            # 更新缓存统计以反映外部处理的项目
            self.renderer.update_mm_cache_stats()
        except Exception:
            # 捕获所有异常
            logger.warning(
                # 记录警告
                "Failed to inject mm_kwargs into processor cache",
                exc_info=True,  # 包含异常堆栈
            )

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        # 用添加了 8 个随机字符的内部请求 ID 替换外部提供的请求 ID，确保唯一性。
        if request.external_req_id is not None:
            # 如果 external_req_id 已被设置（内部字段）
            raise ValueError(
                # 抛出错误
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        # 保存用户提供的原始请求 ID 到 external_req_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            # 如果禁用了请求 ID 随机化（调试模式）
            logger.warning_once(
                # 记录一次性警告
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
            # 生成内部唯一 ID：外部 ID + 8 位随机字符

    def process_inputs(
        self,
        request_id: str,  # 用户提供的请求 ID
        prompt: PromptType | EngineInput,  # 原始输入
        params: SamplingParams | PoolingParams,  # 采样/池化参数
        supported_tasks: tuple[SupportedTask, ...],  # 支持的任务
        arrival_time: float | None = None,  # 到达时间（可选）
        lora_request: LoRARequest | None = None,  # LoRA 请求（可选）
        tokenization_kwargs: dict[str, Any] | None = None,  # tokenize 参数（可选）
        trace_headers: Mapping[str, str] | None = None,  # 追踪头（可选）
        priority: int = 0,  # 优先级
        data_parallel_rank: int | None = None,  # 目标 DP rank（可选）
        resumable: bool = False,  # 是否可续传（流式输入）
    ) -> EngineCoreRequest:
        # 核心方法：将原始输入处理为 EngineCoreRequest
        self._validate_params(params, supported_tasks)  # 验证参数
        self._validate_lora(lora_request)  # 验证 LoRA

        parallel_config = self.vllm_config.parallel_config  # 并行配置
        dp_size = parallel_config.data_parallel_size  # DP 大小
        dp_local_size = parallel_config.data_parallel_size_local  # 本地 DP 大小
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        # 有效 rank 数（仅本地引擎时用本地数，否则用全局数）
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            # 如果指定了无效的 DP rank
            raise ValueError(
                # 抛出错误
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            # 如果 prompt 已是 EngineInput 字典（已预处理）
            if tokenization_kwargs:
                # 如果传入了 tokenize 参数（已废弃）
                logger.warning_once(
                    # 记录一次性警告
                    "Passing tokenization_kwargs to InputProcessor is deprecated "
                    "and will be removed in v0.18. You should instead pass "
                    "them to Renderer.render_cmpl() or Renderer.render_chat()."
                )

            if arrival_time is None:
                # 如果未提供到达时间
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]
                # 从输入中获取或使用当前时间

            processed_inputs: EngineInput = prompt  # type: ignore[assignment]
            # 直接使用已预处理的输入
        else:
            # 否则是原始 prompt（需预处理）
            logger.warning_once(
                # 记录一次性废弃警告
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                # 如果未提供到达时间
                arrival_time = time.time()  # 使用当前时间

            processed_inputs = self.input_preprocessor.preprocess(
                prompt,  # 原始 prompt
                tokenization_kwargs=tokenization_kwargs,  # tokenize 参数
            )
            # 通过预处理器处理（tokenize、多模态预处理等）

        current_platform.validate_request(processed_inputs, params)
        # 平台特定请求验证

        encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
        # 拆分 encoder-decoder 输入（非 enc-dec 模型时 encoder 为 None）
        self._validate_model_inputs(encoder_inputs, decoder_inputs)
        # 验证模型输入（长度、vocab 范围等）

        # Mypy can be conservative for TypedDict unions; normalize access.
        # mypy 对 TypedDict 联合可能过于保守；归一化访问方式。
        if decoder_inputs["type"] == "embeds":
            # 如果是 embedding 输入（非 token）
            prompt_embeds = decoder_inputs["prompt_embeds"]
            # 获取预计算 embedding
            prompt_token_ids = decoder_inputs.get("prompt_token_ids")
            # 可能同时有 token IDs（混合模式）
            prompt_is_token_ids = decoder_inputs.get("is_token_ids")
            # 混合模式标记（True=token，False=embedding）
        else:
            # 纯 token 输入
            prompt_token_ids = decoder_inputs["prompt_token_ids"]  # token IDs
            prompt_embeds = None  # 无 embedding
            prompt_is_token_ids = None  # 无混合标记

        sampling_params = None  # 采样参数初始化为 None
        pooling_params = None  # 池化参数初始化为 None
        if isinstance(params, SamplingParams):
            # 如果是采样参数（生成任务）
            # TODO: can we avoid cloning here in multiproc case?
            # TODO：在多进程场景下能否避免克隆？
            sampling_params = params.clone()  # 克隆采样参数
            # If unset max tokens, then generate up to the max_model_len.
            # 如果未设置 max_tokens，则生成到 max_model_len 为止
            if sampling_params.max_tokens is None:
                # 如果未指定生成 token 上限
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                # 计算 prompt 序列长度
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len
                # 默认生成量 = 最大模型长度 - prompt 长度

            sampling_params.update_from_generation_config(
                self.generation_config_fields,  # generation 配置字段
                self.renderer.get_eos_token_id(),  # EOS token ID
            )
            # 从 generation 配置更新采样参数（如 stop token）
            if self.tokenizer is not None:
                # 如果有 tokenizer
                sampling_params.update_from_tokenizer(self.tokenizer)
                # 从 tokenizer 更新（如 bos/eos token 相关）
        else:
            pooling_params = params.clone()  # 克隆池化参数

        # Multimodal related.
        # 多模态相关处理
        mm_features: list[MultiModalFeatureSpec] | None = None  # 多模态特征列表

        if decoder_inputs["type"] == "multimodal":
            # 如果是多模态输入
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]  # 多模态处理参数
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            # 多模态占位符位置
            decoder_mm_hashes = decoder_inputs["mm_hashes"]  # 多模态内容哈希

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                # 验证所有哈希叶子节点都是字符串
                raise ValueError(
                    # 抛出错误
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            # 将多模态占位符、哈希和输入从字典合并并展平为列表，
            # 并按每个项目在输入序列中的位置排序。
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)
            # 按位置排序多模态索引

            mm_features = []  # 多模态特征列表
            for modality, idx in sorted_mm_idxs:
                # 遍历排序后的多模态索引
                base_mm_hash = decoder_mm_hashes[modality][idx]  # 基础哈希
                mm_features.append(
                    MultiModalFeatureSpec(  # 创建特征规格
                        data=decoder_mm_inputs[modality][idx],  # 输入数据
                        modality=modality,  # 模态类型
                        identifier=self._get_mm_identifier(
                            base_mm_hash,  # 多模态标识（考虑 LoRA）
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        # 在序列中的位置
                        mm_hash=base_mm_hash,  # 内容哈希
                    )
                )

        return EngineCoreRequest(
            # 构建并返回 EngineCoreRequest
            request_id=request_id,  # 请求 ID
            prompt_token_ids=prompt_token_ids,  # prompt token IDs
            prompt_embeds=prompt_embeds,  # prompt embeddings（可选）
            prompt_is_token_ids=prompt_is_token_ids,  # 混合标记
            mm_features=mm_features,  # 多模态特征
            sampling_params=sampling_params,  # 采样参数
            pooling_params=pooling_params,  # 池化参数
            arrival_time=arrival_time,  # 到达时间
            lora_request=lora_request,  # LoRA
            cache_salt=decoder_inputs.get("cache_salt"),  # 缓存 salt
            priority=priority,  # 优先级
            data_parallel_rank=data_parallel_rank,  # DP rank
            trace_headers=trace_headers,  # 追踪头
            resumable=resumable,  # 可续传标记
        )

    def _validate_prompt_len(
        self,
        prompt_len: int,  # prompt 长度
        prompt_type: Literal["encoder", "decoder"],  # prompt 类型
    ):
        # 验证 prompt 长度
        if self.skip_prompt_length_check:
            # 如果跳过长度检查（多模态模型可能跳过）
            return  # 直接返回

        if prompt_len == 0 and prompt_type == "decoder":
            # 如果 decoder prompt 为空
            raise ValueError(f"The {prompt_type} prompt cannot be empty")
            # 抛出错误

        model_config = self.model_config  # 模型配置
        max_prompt_len = (
            model_config.max_model_len  # decoder 用最大模型长度
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size  # encoder 用编码器缓存大小
        )
        if prompt_len > max_prompt_len:
            # 如果 prompt 超过最大长度
            if self.supports_mm_inputs:
                # 多模态模型的建议
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                # 文本模型的建议
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise ValueError(
                # 抛出错误
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            # 如果恰好等于最大长度且是生成模型
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            # 生成至少需要 1 个输出 token
            raise ValueError(
                # 抛出错误
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def _validate_model_input(
        self,
        prompt_input: SingletonInput,  # 单一输入
        prompt_type: Literal["encoder", "decoder"],  # prompt 类型
    ) -> None:
        # 验证模型输入（长度 + vocab 范围）
        model_config = self.model_config  # 模型配置
        tokenizer = self.tokenizer  # tokenizer

        prompt_ids = (
            None  # embedding 输入无 token IDs
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )
        # 根据输入类型提取 embedding

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        # 计算序列长度
        self._validate_prompt_len(prompt_len, prompt_type)  # 验证长度

        if prompt_input["type"] == "multimodal":
            # 如果是多模态输入
            decoder_mm_positions = prompt_input["mm_placeholders"]
            # 多模态占位符
            for modality, mm_positions in decoder_mm_positions.items():
                # 遍历每个模态
                for mm_position in mm_positions:
                    # 遍历每个位置
                    num_embeds = mm_position.get_num_embeds()  # 嵌入数量
                    if num_embeds > self.mm_encoder_cache_size:
                        # 如果嵌入数超过缓存大小
                        raise ValueError(
                            # 抛出错误
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        if prompt_ids and tokenizer is not None:
            # 如果有 token IDs 和 tokenizer
            max_input_id = max(prompt_ids, default=0)  # 最大 token ID

            # NOTE: tokenizer.max_token_id is the tokenizer's vocab size while
            # self.model_config.get_vocab_size() is the model's vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # 注意：tokenizer.max_token_id 是 tokenizer 的词汇表大小，
            # 而 self.model_config.get_vocab_size() 是模型的词汇表大小。
            # 对于 Qwen3 模型，语言模型有 tokenizer 中不存在的额外 token，
            # 反之某些多模态模型的占位 token 也不在 tokenizer 中。
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            # 这里取两者的最大值来判断 token ID 是否真正超出词汇表。
            model_vocab_size = model_config.get_vocab_size()  # 模型词汇表大小
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                # 如果 token ID 超出两者最大值
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")
                # 抛出错误

    def _validate_model_inputs(
        self,
        encoder_input: SingletonInput | None,  # encoder 输入（可能为 None）
        decoder_input: SingletonInput,  # decoder 输入
    ):
        # 验证 encoder 和 decoder 的模型输入
        if encoder_input is not None:
            # 如果有 encoder 输入（enc-dec 模型）
            self._validate_model_input(encoder_input, prompt_type="encoder")
            # 验证 encoder 输入

        self._validate_model_input(decoder_input, prompt_type="decoder")
        # 验证 decoder 输入