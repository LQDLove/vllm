# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/inputs/preprocess.py
# 本文件实现 InputPreprocessor 类：将用户传入的原始 PromptType 转换为
# 引擎内部可直接消费的 EngineInput 格式（token IDs + 多模态占位符等）。
# 核心路径：preprocess() → 分派到 _process_decoder_only_prompt 或
# _process_encoder_decoder_prompt → 内部再按 prompt 类型（text/tokens/embeds/mm）
# 调用对应的 _process_xxx 方法完成 tokenization 与多模态处理。
# =============================================================================

from collections.abc import Mapping
# 导入 Mapping：类型标注（mm_processor_kwargs 参数）。
from typing import Any, overload
# 导入 Any（宽松类型）、overload（方法重载标注）。

from typing_extensions import assert_never
# 导入 assert_never：提示 mypy 穷尽性检查已被覆盖，运行时抛出异常。

from vllm.config import VllmConfig
# 导入 VllmConfig：顶层配置对象（含 model_config、parallel_config 等）。
from vllm.inputs import build_enc_dec_input
# 导入 build_enc_dec_input：构造 EncoderDecoderInput 的辅助函数。
from vllm.logger import init_logger
# 导入 init_logger：初始化模块级日志记录器。
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
# 导入 MULTIMODAL_REGISTRY（全局多模态注册表单例）、MultiModalRegistry（类型）。
from vllm.renderers import BaseRenderer, renderer_from_config
# 导入 BaseRenderer（渲染器抽象基类）、renderer_from_config（工厂函数）。
from vllm.renderers.inputs import (
    DecoderDictPrompt,
    # decoder-only 模型的字典 prompt 语义类型（文本/token/embeds/mm 的判别）。
    DecoderOnlyDictPrompt,
    # decoder-only 模型的单例字典 prompt（本质是 DecoderDictPrompt 的子集）。
    EncoderDecoderDictPrompt,
    # encoder-decoder 模型的字典 prompt（含 encoder_prompt 和 decoder_prompt）。
    EncoderDictPrompt,
    # encoder 侧的字典 prompt。
    SingletonDictPrompt,
    # 单例字典 prompt 的语义别名。
)
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt, parse_enc_dec_prompt
# 导入两个解析函数：将 PromptType 映射为结构化的字典 prompt。
from vllm.tokenizers import TokenizerLike
# 导入 TokenizerLike：tokenizer 接口类型（HuggingFace tokenizer 或 Dummy）。

from .engine import (
    DecoderEngineInput,
    # decoder 引擎输入（decoder-only 模型中 decoder 侧已预处理的输入）。
    DecoderOnlyEngineInput,
    # decoder-only 模型已预处理的引擎输入。
    EmbedsInput,
    # embeddings 形式的引擎输入（用户直接传入 prompt_embeds）。
    EncoderDecoderInput,
    # encoder-decoder 模型的引擎输入（含 encoder_input 和 decoder_input）。
    EncoderInput,
    # encoder 侧的引擎输入。
    EngineInput,
    # 顶层引擎输入类型别名。
    MultiModalInput,
    # 多模态引擎输入（含 token_ids + mm_placeholders + embeddings）。
    SingletonInput,
    # 单例引擎输入类型别名。
    TokensInput,
    # token IDs 形式的引擎输入。
    tokens_input,
    # tokens_input 工厂函数：构造最小的 TokensInput 结构体。
)
from .llm import (
    EmbedsPrompt,
    # EmbedsPrompt：用户传入 prompt_embeds 的 Prompt 类型。
    MultiModalDataDict,
    # MultiModalDataDict：多模态数据字典（图片/音频等原始输入）。
    MultiModalUUIDDict,
    # MultiModalUUIDDict：多模态资源的 UUID 映射。
    PromptType,
    # PromptType = DecoderOnlyPrompt | EncoderDecoderPrompt：用户级 prompt 类型别名。
    TextPrompt,
    # TextPrompt：用户传入纯文本的 Prompt 类型。
    TokensPrompt,
    # TokensPrompt：用户传入 token ID 列表的 Prompt 类型。
)

logger = init_logger(__name__)
# 初始化本模块日志记录器。


class InputPreprocessor:
    # =========================================================================
    # InputPreprocessor：将用户传入的 PromptType → EngineInput 的转换层。
    # 持有 renderer（tokenizer + 多模态处理器）和 mm_registry 引用，
    # 核心方法是 preprocess()，内部按 prompt 类型分派到具体的处理函数。
    # =========================================================================
    """Pre-processor that converts user prompts into engine inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        # 顶层配置对象。
        renderer: BaseRenderer | None = None,
        # 渲染器（可选；若为 None，则从 vllm_config 自动创建）。
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        # 多模态注册表（默认使用全局单例）。
    ) -> None:
        # -------------------------------------------------------------------
        # 构造函数：保存配置、renderer 和多模态注册表引用。
        # -------------------------------------------------------------------
        super().__init__()
        # 调用基类构造。

        self.model_config = vllm_config.model_config
        # 保存模型配置（模型名、max_model_len、embedding_size 等）。
        self.renderer = renderer or renderer_from_config(vllm_config)
        # 保存 renderer；若未传入则从配置自动创建。
        self.mm_registry = mm_registry
        # 保存多模态注册表引用。

    @property
    def tokenizer(self) -> TokenizerLike | None:
        # -------------------------------------------------------------------
        # 属性：返回 renderer 持有的 tokenizer（可能为 None）。
        # -------------------------------------------------------------------
        return self.renderer.tokenizer
        # 直接透传 renderer 的 tokenizer。

    def get_tokenizer(self) -> TokenizerLike:
        # -------------------------------------------------------------------
        # 获取 tokenizer 实例（保证非 None，若为 None 则抛异常）。
        # -------------------------------------------------------------------
        return self.renderer.get_tokenizer()
        # 委托给 renderer（内部有 None 检查）。

    def _tokenize_prompt(
        self,
        prompt: str,
        # 原始文本 prompt。
        tokenization_kwargs: dict[str, Any] | None = None,
        # 额外的 tokenizer 参数（已废弃，建议在 Renderer 层传入）。
    ) -> list[int]:
        # -------------------------------------------------------------------
        # 对纯文本 prompt 进行 tokenize，返回 token ID 列表。
        # -------------------------------------------------------------------
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        # 文档字符串：对文本 prompt 应用模型 tokenizer，返回对应 token IDs。
        renderer = self.renderer
        # 本地引用 renderer。

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )
        # 基于默认补全 tokenization 参数，合并用户传入的额外 kwargs。

        tok_prompt = renderer._tokenize_singleton_prompt(
            TextPrompt(prompt=prompt),
            # 将纯文本包装为 TextPrompt。
            tok_params,
            # tokenization 参数。
        )
        # 调用 renderer 的 tokenize 方法。

        return tok_prompt["prompt_token_ids"]
        # 返回 token ID 列表。

    def _process_multimodal(
        self,
        prompt: str | list[int],
        # prompt 文本或 token ID 列表。
        mm_data: MultiModalDataDict,
        # 多模态数据（图片/音频等原始输入）。
        mm_processor_kwargs: Mapping[str, object] | None = None,
        # 多模态处理器额外参数。
        tokenization_kwargs: dict[str, Any] | None = None,
        # 额外的 tokenizer 参数。
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
        # 多模态资源 UUID 映射（用于缓存命中）。
    ) -> MultiModalInput:
        # -------------------------------------------------------------------
        # 对多模态 prompt 进行 tokenize + 多模态占位符替换，
        # 返回 MultiModalInput（token_ids + mm_placeholders + embeddings）。
        # -------------------------------------------------------------------
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        # 文档字符串：对多模态 prompt 应用多模态处理器，返回 token IDs 与元数据。
        return self.renderer._process_multimodal(
            prompt,
            # prompt 文本或 token IDs。
            mm_data,
            # 多模态原始数据。
            mm_uuids=mm_uuids,
            # UUID 映射（用于多模态缓存）。
            mm_processor_kwargs=mm_processor_kwargs,
            # 多模态处理器额外参数。
            tokenization_kwargs=tokenization_kwargs,
            # tokenizer 额外参数。
        )
        # 完全委托给 renderer 的多模态处理方法。

    def _process_embeds(
        self,
        parsed_content: EmbedsPrompt,
        # 用户传入的 prompt_embeds Prompt 类型。
    ) -> EmbedsInput:
        # -------------------------------------------------------------------
        # 处理用户直接传入 embeddings 的情况（跳过 tokenization）。
        # -------------------------------------------------------------------
        return self.renderer._process_embeds(parsed_content)
        # 委托给 renderer 的 embeddings 处理方法。

    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: dict[str, Any] | None = None
    ) -> list[int]:
        # -------------------------------------------------------------------
        # 对 token ID 列表进行截断（按模型 max_model_len+truncation 策略）。
        # 通过重新走一次 _tokenize_singleton_prompt 让 tokenizer 做截断。
        # -------------------------------------------------------------------
        renderer = self.renderer
        # 本地引用 renderer。

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )
        # 构造 tokenization 参数（含 truncation 相关配置）。

        tok_prompt = renderer._tokenize_singleton_prompt(
            TokensPrompt(prompt_token_ids=inputs),
            # 将 token ID 列表包装为 TokensPrompt（避免文本 tokenize 开销）。
            tok_params,
            # 参数（含 max_length、truncation 策略）。
        )
        # 让 tokenizer 按配置截断。

        return tok_prompt["prompt_token_ids"]
        # 返回截断后的 token ID 列表。

    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        # 用户传入的 token IDs Prompt 类型。
        tokenization_kwargs: dict[str, Any] | None = None,
        # 额外的 tokenizer 参数。
    ) -> TokensInput | MultiModalInput:
        # -------------------------------------------------------------------
        # 处理用户以 token ID 列表形式传入的 prompt。
        # 先截断，如果有伴随的多模态数据则走多模态处理。
        # -------------------------------------------------------------------
        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )
        # 截断 token ID 列表。

        inputs: TokensInput | MultiModalInput
        # 初始化局部变量（后面赋值）。
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            # 如果该 prompt 包含多模态数据。
            inputs = self._process_multimodal(
                prompt_token_ids,
                # 截断后的 token IDs。
                multi_modal_data,
                # 多模态原始数据。
                parsed_content.get("mm_processor_kwargs"),
                # 多模态处理器额外参数。
                tokenization_kwargs=tokenization_kwargs,
                # tokenizer 额外参数。
                mm_uuids=parsed_content.get("multi_modal_uuids"),
                # UUID 映射。
            )
            # 走多模态处理路径。
        else:
            inputs = tokens_input(prompt_token_ids)
            # 纯 token IDs：构造最小 TokensInput。

        if prompt_text := parsed_content.get("prompt"):
            # 如果用户额外提供了 prompt 文本（用于日志/调试）。
            inputs["prompt"] = prompt_text
            # 保存 prompt 文本。
        if cache_salt := parsed_content.get("cache_salt"):
            # 如果用户提供了缓存盐值（用于区分相同 token IDs 的不同语义）。
            inputs["cache_salt"] = cache_salt
            # 保存缓存盐值。

        return inputs
        # 返回 TokensInput 或 MultiModalInput。

    def _process_text(
        self,
        parsed_content: TextPrompt,
        # 用户传入的纯文本 Prompt 类型。
        tokenization_kwargs: dict[str, Any] | None = None,
        # 额外的 tokenizer 参数。
    ) -> TokensInput | MultiModalInput:
        # -------------------------------------------------------------------
        # 处理用户以纯文本形式传入的 prompt。
        # 先 tokenize，如果有伴随的多模态数据则走多模态处理。
        # -------------------------------------------------------------------
        prompt_text = parsed_content["prompt"]
        # 取出文本内容。

        inputs: TokensInput | MultiModalInput
        # 初始化局部变量。
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            # 如果该 prompt 包含多模态数据。
            inputs = self._process_multimodal(
                prompt_text,
                # 原始文本 prompt。
                multi_modal_data,
                # 多模态原始数据。
                parsed_content.get("mm_processor_kwargs") or {},
                # 多模态处理器额外参数（缺省为空字典）。
                tokenization_kwargs=tokenization_kwargs,
                # tokenizer 额外参数。
            )
            # 走多模态处理路径（内部会同时做 tokenize + 占位符替换）。
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            # 纯文本：走 tokenization 路径。
            inputs = tokens_input(prompt_token_ids)
            # 构造 TokensInput。

        inputs["prompt"] = prompt_text
        # 保存原始文本 prompt。

        if cache_salt := parsed_content.get("cache_salt"):
            # 如果用户提供了缓存盐值。
            inputs["cache_salt"] = cache_salt
            # 保存缓存盐值。

        return inputs
        # 返回 TokensInput 或 MultiModalInput。

    @overload
    def _prompt_to_llm_inputs(
        self,
        prompt: EncoderDictPrompt,
        # encoder 字典 prompt。
        tokenization_kwargs: dict[str, Any] | None = None,
        # tokenizer 额外参数。
    ) -> EncoderInput: ...
    # 重载签名 1：EncoderDictPrompt → EncoderInput。

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderDictPrompt,
        # decoder 字典 prompt。
        tokenization_kwargs: dict[str, Any] | None = None,
        # tokenizer 额外参数。
    ) -> DecoderEngineInput: ...
    # 重载签名 2：DecoderDictPrompt → DecoderEngineInput。

    @overload
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: DecoderOnlyDictPrompt,
        # decoder-only 字典 prompt。
        tokenization_kwargs: dict[str, Any] | None = None,
        # tokenizer 额外参数。
    ) -> DecoderOnlyEngineInput: ...
    # 重载签名 3：DecoderOnlyDictPrompt → DecoderOnlyEngineInput。

    def _prompt_to_llm_inputs(
        self,
        prompt: SingletonDictPrompt,
        # 单例字典 prompt（text/token/embeds 三种格式之一）。
        tokenization_kwargs: dict[str, Any] | None = None,
        # tokenizer 额外参数。
    ) -> SingletonInput:
        # -------------------------------------------------------------------
        # 分派单例 prompt 到具体的处理函数：
        #   - 含 "prompt_embeds" → _process_embeds()
        #   - 含 "prompt_token_ids" → _process_tokens()
        #   - 含 "prompt" → _process_text()
        # -------------------------------------------------------------------
        if "prompt_embeds" in prompt:
            # 如果 prompt 中包含 embeddings。
            return self._process_embeds(prompt)  # type: ignore[arg-type]
            # 走 embeddings 路径。

        if "prompt_token_ids" in prompt:
            # 如果 prompt 中包含 token IDs。
            return self._process_tokens(prompt)  # type: ignore[arg-type]
            # 走 tokens 截断/处理路径。

        if "prompt" in prompt:
            # 如果 prompt 中包含文本。
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
            )
            # 走文本 tokenization 路径。

        assert_never(prompt)  # type: ignore[arg-type]
        # 穷尽性断言：不应到达此处，若到达则运行时抛出 AssertionError。

    def _process_encoder_decoder_prompt(
        self,
        prompt: EncoderDecoderDictPrompt,
        # encoder-decoder 字典 prompt（含 encoder_prompt 和 decoder_prompt）。
        tokenization_kwargs: dict[str, Any] | None = None,
        # tokenizer 额外参数。
    ) -> EncoderDecoderInput:
        # -------------------------------------------------------------------
        # 处理 encoder-decoder 模型的 prompt：
        # 分别处理 encoder 侧和 decoder 侧的输入，然后组装为 EncoderDecoderInput。
        # -------------------------------------------------------------------
        encoder_prompt = prompt["encoder_prompt"]
        # 取出 encoder 侧 prompt。
        decoder_prompt = prompt["decoder_prompt"]
        # 取出 decoder 侧 prompt（可能为 None）。

        skip_decoder_start_token = False
        # 是否跳过 decoder 起始 token（默认不跳过）。
        if self.renderer.mm_processor is not None:
            # 如果有多模态处理器。
            from vllm.multimodal.processing import EncDecMultiModalProcessor
            # 延迟导入 encoder-decoder 多模态处理器类。

            if isinstance(self.renderer.mm_processor, EncDecMultiModalProcessor):
                # 如果处理器是 EncDecMultiModalProcessor 实例。
                skip_decoder_start_token = (
                    self.renderer.mm_processor.skip_decoder_start_token
                )
                # 读取处理器的配置：是否跳过 decoder 起始 token。

        return build_enc_dec_input(
            encoder_input=self._prompt_to_llm_inputs(
                encoder_prompt,
                tokenization_kwargs=tokenization_kwargs,
            ),
            # 将 encoder 侧 prompt 转为 EncoderInput。
            decoder_input=(
                None
                if decoder_prompt is None
                # decoder 侧没有 prompt 则为 None（仅编码场景）。
                else self._prompt_to_llm_inputs(
                    decoder_prompt,
                    tokenization_kwargs=tokenization_kwargs,
                )
                # 将 decoder 侧 prompt 转为 DecoderEngineInput。
            ),
            decoder_start_token_id=self.renderer.get_dec_start_token_id(),
            # decoder 起始 token ID。
            skip_decoder_start_token=skip_decoder_start_token,
            # 是否跳过 decoder 起始 token。
        )
        # 组装为 EncoderDecoderInput 返回。

    def _process_decoder_only_prompt(
        self,
        prompt: DecoderOnlyDictPrompt,
        # decoder-only 字典 prompt。
        tokenization_kwargs: dict[str, Any] | None = None,
        # tokenizer 额外参数。
    ) -> DecoderOnlyEngineInput:
        # -------------------------------------------------------------------
        # 处理 decoder-only 模型的 prompt：直接分派到 _prompt_to_llm_inputs。
        # -------------------------------------------------------------------
        return self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
        )
        # 分派到单例处理逻辑。

    def preprocess(
        self,
        prompt: PromptType,
        # 用户传入的 prompt（可以是 str、list[int]、TextPrompt、TokensPrompt 等）。
        tokenization_kwargs: dict[str, Any] | None = None,
        # 额外的 tokenizer 参数（已废弃）。
    ) -> EngineInput:
        # -------------------------------------------------------------------
        # 公共入口：将 PromptType 转换为 EngineInput。
        # 先判断模型是 encoder-decoder 还是 decoder-only，再分派到对应处理方法。
        # -------------------------------------------------------------------
        """Preprocess the input prompt."""
        # 文档字符串：预处理输入 prompt。
        if self.model_config.is_encoder_decoder:
            # 如果模型是 encoder-decoder 架构。
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            # 注释：encoder-decoder 模型需将输入 prompt 特殊映射为 encoder 与 decoder。
            return self._process_encoder_decoder_prompt(
                parse_enc_dec_prompt(prompt),
                # 先将 PromptType 解析为 EncoderDecoderDictPrompt。
                tokenization_kwargs,
                # 透传 tokenization 参数。
            )
            # 走 encoder-decoder 路径。

        return self._process_decoder_only_prompt(
            parse_dec_only_prompt(prompt),
            # 先将 PromptType 解析为 DecoderOnlyDictPrompt。
            tokenization_kwargs=tokenization_kwargs,
            # 透传 tokenization 参数。
        )
        # 走 decoder-only 路径（最常见）。