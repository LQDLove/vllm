# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import sys  # sys：系统模块（sys.maxsize 用作最大整数比较）
from abc import ABC, abstractmethod  # ABC：抽象基类；abstractmethod：抽象方法装饰器

import tokenizers  # tokenizers：HuggingFace tokenizers 库
import tokenizers.decoders  # decoders：tokenizers 库的解码器模块（DecodeStream）
from packaging import version  # version：版本号解析比较库
from tokenizers import Tokenizer  # Tokenizer：tokenizers 核心类
from transformers import TokenizersBackend  # TokenizersBackend：transformers 后端抽象

from vllm.logger import init_logger  # 初始化 vLLM 日志记录器
from vllm.tokenizers import TokenizerLike  # tokenizer 接口类型
from vllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,  # 将 prompt token IDs 转换为 token 字符串列表
    detokenize_incrementally,  # 增量式 detokenize（慢速路径用）
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
# 计算 token/embedding 序列长度
from vllm.v1.engine import EngineCoreRequest  # 引擎核心请求类型

logger = init_logger(__name__)  # 模块级日志记录器

# Only tokenizers >= 0.22.0 supports DecodeStream with native prefill
# (ids parameter) used for FastIncrementalDetokenizer.
# 仅 tokenizers >= 0.22.0 支持带原生 prefill（ids 参数）的 DecodeStream，
# 用于 FastIncrementalDetokenizer。
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")
# 是否使用快速 detokenizer（版本 >= 0.22.0）

# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
# 来自 tokenizers 库源码的错误字符串（无效前缀错误）
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"
# 无效前缀错误消息


class IncrementalDetokenizer:
    # 增量式 detokenizer 基类（纯占位实现，不做实际解码）
    def __init__(self):
        self.token_ids: list[int] = []  # 生成的 token ID 列表

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids  # 属性：返回生成的 token ID 列表

    def num_output_tokens(self) -> int:
        return len(self.token_ids)  # 返回生成的 token 数量

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        self.token_ids.extend(new_token_ids)  # 扩展 token ID 列表
        return None  # 返回 None（不做解码）

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""  # 返回空字符串（不做解码）

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,  # tokenizer（可为 None）
        request: EngineCoreRequest,  # 引擎核心请求
    ) -> "IncrementalDetokenizer":
        # 工厂方法：根据请求创建合适的 detokenizer
        assert request.sampling_params is not None  # 断言采样参数存在

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            # 无 tokenizer => 跳过 detokenization。
            return IncrementalDetokenizer()  # 返回纯占位实例

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            # Fast tokenizer => use tokenizers library DecodeStream.
            # 快速 tokenizer => 使用 tokenizers 库的 DecodeStream。
            return FastIncrementalDetokenizer(tokenizer, request)
            # 返回快速增量 detokenizer

        # Fall back to slow python-based incremental detokenization.
        # 回退到慢速 Python 增量 detokenization。
        return SlowIncrementalDetokenizer(tokenizer, request)
        # 返回慢速增量 detokenizer


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    # 带实际解码逻辑的增量 detokenizer 基类（抽象）
    def __init__(self, request: EngineCoreRequest):
        super().__init__()  # 调用父类初始化（token_ids 列表）

        # Stop strings
        # 停止字符串配置
        params = request.sampling_params  # 获取采样参数
        assert params is not None  # 断言参数存在
        if params.stop is None:
            # 如果未配置停止字符串
            self.stop = []  # 空列表
        elif isinstance(params.stop, str):
            # 如果是单个字符串
            self.stop = [params.stop]  # 转为列表
        else:
            self.stop = params.stop  # 直接使用列表
        self.min_tokens = params.min_tokens  # 最小 token 数
        self.include_stop_str_in_output = params.include_stop_str_in_output
        # 是否在输出中包含停止字符串

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        # 当停止字符串需从流式输出中排除时，需要保留的字符数。
        if self.stop and not self.include_stop_str_in_output:
            # 如果有停止字符串且不包含在输出中
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
            # 缓冲长度 = 最长停止字符串长度 - 1
        else:
            self.stop_buffer_length = 0  # 否则无缓冲
        self._last_output_text_offset: int = 0  # 上次输出文本偏移

        # Generation data
        # 生成数据
        self.output_text = ""  # 累计输出文本

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        # 按以下方式更新请求的 RequestState：
        # 1) 增量式 detokenize 新 token IDs。
        # 2) 评估停止条件。
        # 返回匹配的停止字符串或 None。
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            # 如果没有新 token ID 则跳过 detokenization。
            return None  # 返回 None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            # 如果因停止而终止，根据 include_stop_str_in_output 参数
            # 从 detokenization 中排除最后一个 token。
            skipped_stop_token_id = new_token_ids[-1]  # 跳过的停止 token
            new_token_ids = new_token_ids[:-1]  # 排除最后一个 token
        else:
            skipped_stop_token_id = None  # 无跳过的 token

        # 1) Detokenize the new token ids incrementally.
        # 1) 增量式 detokenize 新 token IDs。
        stop_check_offset = len(self.output_text)  # 停止检查偏移
        for new_token_id in new_token_ids:
            # 遍历新 token
            self.token_ids.append(new_token_id)  # 添加 token ID
            self.output_text += self.decode_next(new_token_id)  # 解码并累加
            # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
            # 支持 min_tokens，见 issue 链接
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                # 如果设置了最小 token 数且未达到
                stop_check_offset = len(self.output_text)
                # 更新停止检查偏移（min_tokens 内不检查停止）

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            # 跳过 detokenization 后的清理。
            self.token_ids.append(skipped_stop_token_id)  # 重新添加 token ID

        # 2) Evaluate stop strings.
        # 2) 评估停止字符串。
        stop_string = None  # 停止字符串
        if self.stop and self.num_output_tokens() > self.min_tokens:
            # 如果有停止字符串且超过最小 token 数
            stop = check_stop_strings(
                output_text=self.output_text,  # 当前输出文本
                new_char_count=len(self.output_text) - stop_check_offset,
                # 新增字符数
                stop=self.stop,  # 停止字符串列表
                include_in_output=self.include_stop_str_in_output,
                # 是否包含停止字符串
            )
            if stop is not None:
                # 如果匹配到停止字符串
                stop_string, truncate_to = stop  # 解包
                if truncate_to != -1:
                    # 如果需要截断
                    self.output_text = self.output_text[:truncate_to]
                    # 截断输出文本

        return stop_string  # 返回匹配的停止字符串

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:
        # 抽象方法：解码单个 token
        raise NotImplementedError

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""
        # 如果 delta 为 True，只返回自上次调用本方法以来的新文本

        # We return the full output text if the sequence is finished.
        # 如果序列已完成，返回完整输出文本。
        buffer_length = 0 if finished else self.stop_buffer_length
        # 缓冲长度（完成时无缓冲）
        if not delta:
            # 非 delta 模式
            if not buffer_length:
                # 如果无缓冲
                return self.output_text  # 返回完整文本
            return self.output_text[:-buffer_length]
            # 返回去除缓冲的文本

        length = len(self.output_text) - buffer_length  # 有效长度
        last_offset = self._last_output_text_offset  # 上次偏移
        if last_offset < length:
            # 如果有新文本
            self._last_output_text_offset = length  # 更新偏移
            return self.output_text[last_offset:length]  # 返回增量文本
        return ""  # 无新文本


class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    # 快速增量 detokenizer（使用 tokenizers 库 DecodeStream）
    def __init__(self, tokenizer: TokenizersBackend, request: EngineCoreRequest):
        # 构造函数
        super().__init__(request)  # 调用父类初始化

        sampling_params = request.sampling_params  # 采样参数
        assert sampling_params is not None  # 断言存在

        self.request_id = request.request_id  # 请求 ID（用于错误日志）
        self.skip_special_tokens = sampling_params.skip_special_tokens
        # 是否跳过特殊 token

        self.tokenizer: Tokenizer = tokenizer._tokenizer  # 底层 tokenizer

        # Use native prefill to prime the decode stream with prompt tokens.
        # Look up DecodeStream on the module so backend patches (e.g. the
        # fastokens shim that replaces ``tokenizers.decoders.DecodeStream``)
        # are honored regardless of import order.
        # 使用原生 prefill 将解码流初始化为 prompt token。
        # 在模块上查找 DecodeStream，使后端补丁（如替换
        # ``tokenizers.decoders.DecodeStream`` 的 fastokens shim）
        # 无论导入顺序如何都生效。
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,  # 用 prompt token IDs 预填充
            skip_special_tokens=self.skip_special_tokens,  # 跳过特殊 token
        )
        # 创建解码流（原生 prefill）

        self.spaces_between_special_tokens = (
            # 特殊 token 间是否需要空格
            sampling_params.skip_special_tokens  # 跳过特殊 token 时需要
            or sampling_params.spaces_between_special_tokens  # 或显式配置
        )

        if not self.spaces_between_special_tokens:
            # 如果特殊 token 间不需要空格
            # Store dict of added token ids so that we can suppress
            # the spaces between them.
            # 存储添加的 token ID 字典，以便抑制它们之间的空格。
            added_token_ids = getattr(self.tokenizer, "added_token_ids", None)
            # 获取已添加的 token ID
            if added_token_ids is None:
                # 如果未缓存
                self.tokenizer.added_token_ids = added_token_ids = {
                    tid: tok.content  # token ID → 内容
                    for tid, tok in self.tokenizer.get_added_tokens_decoder().items()
                    # 遍历添加的 token
                }
                # 构建并缓存字典

            if added_token_ids:
                # 如果有添加的 token
                self.last_special = False  # 上次是否特殊 token
                self.added_token_ids = added_token_ids  # 保存字典
            else:
                # No added tokens.
                # 无添加的 token
                self.spaces_between_special_tokens = True  # 恢复空格

    def decode_next(self, next_token_id: int) -> str:
        # 解码单个 token（快速路径）
        token = self._protected_step(next_token_id)  # 保护性步进

        if not self.spaces_between_special_tokens:
            # 如果特殊 token 间不需要空格
            special_token = self.added_token_ids.get(next_token_id)
            # 查找是否特殊 token
            is_special = special_token is not None  # 是否特殊
            if is_special and self.last_special:
                # Return raw token string without any prefixed spaces.
                # 如果连续两个特殊 token，返回无前缀空格的原始 token 字符串。
                token = special_token  # 使用原始内容
            self.last_special = is_special  # 更新上次特殊标志

        return token or ""  # 返回 token 或空字符串

    def _protected_step(self, next_token_id: int) -> str | None:
        # 保护性步进（处理异常）
        try:
            token = self.stream.step(self.tokenizer, next_token_id)  # 步进解码
        except (OverflowError, TypeError):
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            # 处理罕见观察到的溢出，仍需诊断。
            # 见 issue 链接
            logger.exception("Encountered invalid token id: %r", next_token_id)
            # 记录异常（无效 token ID）
            token = None  # token 设为 None
        except Exception as e:
            # 捕获其他异常
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                # 如果不是无效前缀错误
                raise e  # 重新抛出
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            # 从边界情况恢复：tokenizer 可能产生非单调、无效的 UTF-8 输出，
            # 破坏 tokenizers DecodeStream 的内部状态。
            # 见 issue 链接
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            # 记录警告
            self.stream = tokenizers.decoders.DecodeStream(
                skip_special_tokens=self.skip_special_tokens
            )
            # 重置解码流（不预填充）
            token = self.stream.step(self.tokenizer, next_token_id)
            # 重新步进
        return token  # 返回 token


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    # 慢速增量 detokenizer（纯 Python 实现）
    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):
        # 构造函数
        super().__init__(request)  # 调用父类初始化

        self.tokenizer = tokenizer  # 保存 tokenizer
        params = request.sampling_params  # 采样参数
        assert params is not None  # 断言存在

        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        # 计算 prompt 长度

        # Metadata for incremental detokenization.
        # 增量 detokenization 的元数据。
        if request.prompt_token_ids is not None:
            # 如果有 prompt token IDs
            self.tokens, self.prefix_offset, self.read_offset = (
                convert_prompt_ids_to_tokens(
                    tokenizer=tokenizer,  # tokenizer
                    prompt_ids=request.prompt_token_ids,  # prompt token IDs
                    skip_special_tokens=params.skip_special_tokens,
                    # 跳过特殊 token
                )
            )
            # 将 prompt token 转换为字符串列表（预计算）
        else:
            # Prompt embedding requests cannot be detokenized, in general.
            # 通常无法对 prompt embedding 请求进行 detokenize。
            self.tokens = [""] * self.prompt_len  # 用空字符串填充
            self.prefix_offset = 0  # 前缀偏移 0
            self.read_offset = 0  # 读取偏移 0

        self.token_ids.extend(request.prompt_token_ids or [0] * self.prompt_len)
        # 扩展 token ID 列表（embedding 用 0 填充）

        self.skip_special_tokens = params.skip_special_tokens  # 跳过特殊 token
        self.spaces_between_special_tokens = params.spaces_between_special_tokens
        # 特殊 token 间是否有空格

    @property
    def output_token_ids(self) -> list[int]:
        # 属性：返回生成的 token ID（不含 prompt）
        if self.prompt_len:
            # 如果有 prompt
            return self.token_ids[self.prompt_len:]
            # 返回生成部分
        return self.token_ids  # 返回全部

    def num_output_tokens(self) -> int:
        # 返回生成的 token 数量（不含 prompt）
        return len(self.token_ids) - self.prompt_len  # 减去 prompt 长度

    def decode_next(self, next_token_id: int) -> str:
        # 解码单个 token（慢速路径）
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,  # tokenizer
            all_input_ids=self.token_ids,  # 所有输入 token IDs
            prev_tokens=self.tokens,  # 之前的 token 字符串
            prefix_offset=self.prefix_offset,  # 前缀偏移
            read_offset=self.read_offset,  # 读取偏移
            skip_special_tokens=self.skip_special_tokens,  # 跳过特殊 token
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            # 特殊 token 间空格
        )
        # 调用增量 detokenize 工具函数

        self.tokens.extend(new_tokens)  # 扩展 token 列表
        self.prefix_offset = prefix_offset  # 更新前缀偏移
        self.read_offset = read_offset  # 更新读取偏移

        return decoded_text  # 返回解码文本


def check_stop_strings(
    output_text: str,  # 完整输出文本
    new_char_count: int,  # 新增字符数
    stop: list[str],  # 停止字符串列表
    include_in_output: bool,  # 是否包含停止字符串在输出中
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.

    When several stop strings match within the newly generated text (for
    example when speculative decoding appends multiple tokens in a single
    step), the stop string that completes earliest in the text is selected,
    so the result matches appending one token at a time. Ties are broken by
    stop-list order.
    """
    # 检查是否有停止字符串匹配，并相应截断序列输出文本。
    # 返回 (stop_string, offset) 元组（匹配时）或 None。
    # stop_string 是匹配的停止字符串；offset 是 output_text 应截断到的长度，
    # 或 -1 表示不截断。
    # 当多个停止字符串在新生成文本中匹配时（例如投机解码在单步中
    # 追加多个 token），选择在文本中最早完成的停止字符串，
    # 使结果与逐个追加 token 一致。平局按停止列表顺序打破。
    if not new_char_count or not stop:
        # 如果没有新增字符或无停止字符串
        return None  # 返回 None

    best_stop_str: str | None = None  # 最佳停止字符串
    best_stop_index = 0  # 最佳停止字符串起始索引
    best_end = sys.maxsize  # 最佳结束位置（初始化为最大值）
    for stop_str in stop:
        # 遍历停止字符串
        stop_string_len = len(stop_str)  # 停止字符串长度
        # Avoid searching already-searched text.
        # 避免搜索已搜索过的文本。
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        # 在新文本区域搜索停止字符串（从当前位置减去新字符数和停止长度）
        if stop_index == -1:
            # 如果未找到
            continue  # 继续下一个

        # Prefer the stop string that completes earliest in the text.
        # 优先选择在文本中最早完成的停止字符串。
        end = stop_index + stop_string_len  # 结束位置
        if end < best_end:
            # 如果结束位置更早
            best_stop_str = stop_str  # 更新最佳停止字符串
            best_stop_index = stop_index  # 更新起始索引
            best_end = end  # 更新结束位置

    if best_stop_str is None:
        # 如果没有匹配的停止字符串
        return None  # 返回 None

    if include_in_output:
        # 如果要在输出中包含停止字符串
        # Truncate to end of stop string.
        # 截断到停止字符串结尾。
        if best_end >= len(output_text):
            # No truncation required.
            # 无需截断。
            return best_stop_str, -1  # 返回 -1（不截断）
        return best_stop_str, best_end  # 返回截断位置

    # Truncate the output text to the beginning of the stop string.
    # 将输出文本截断到停止字符串的起始位置。
    return best_stop_str, best_stop_index  # 返回起始索引