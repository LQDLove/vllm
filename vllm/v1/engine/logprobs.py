# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import itertools  # itertools：迭代工具库（repeat 无限重复）
from collections.abc import Iterable  # 类型标注：可迭代对象
from dataclasses import dataclass  # dataclasses：数据类装饰器

from vllm.logger import init_logger  # 初始化 vLLM 日志记录器
from vllm.logprobs import (
    FlatLogprobs,  # 扁平结构的 logprobs 容器
    PromptLogprobs,  # prompt logprobs 类型（list[dict]）
    SampleLogprobs,  # 采样 logprobs 类型
    append_logprobs_for_next_position,  # 为下一个位置追加 logprob 容器
    create_prompt_logprobs,  # 创建 prompt logprobs 容器
    create_sample_logprobs,  # 创建 sample logprobs 容器
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike,  # tokenizer 接口类型
    convert_ids_list_to_tokens,  # 将 token ID 列表转换为 token 字符串列表
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
# 引擎核心输出、引擎核心请求类型（跨进程数据载体）
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
# LogprobsLists：logprobs 列表（采样阶段）；LogprobsTensors：logprobs 张量（prompt 阶段）

logger = init_logger(__name__)  # 模块级日志记录器

NONES = itertools.repeat(None)
# NONES：无限重复 None 的迭代器，用于禁止 detokenization 时的占位


@dataclass
class LogprobsProcessor:
    # Logprobs 处理器：管理单个请求的 logprobs 累积与处理
    # Tokenizer for this request,
    # None if detokenization is disabled.
    # 该请求的 tokenizer；禁用 detokenization 时为 None
    tokenizer: TokenizerLike | None

    # Logprobs for this request
    # 该请求的 logprobs 数据
    logprobs: SampleLogprobs | None  # 采样 logprobs 累积容器
    prompt_logprobs: PromptLogprobs | None  # prompt logprobs 累积容器
    cumulative_logprob: float | None  # 累积对数概率（所有生成 token 的 logprob 之和）
    num_logprobs: int | None  # 请求的采样 logprobs 数量
    num_prompt_logprobs: int | None  # 请求的 prompt logprobs 数量

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,  # tokenizer（可为 None）
        request: EngineCoreRequest,  # 引擎核心请求
    ) -> "LogprobsProcessor":
        # 工厂方法：根据新请求创建 LogprobsProcessor
        sampling_params = request.sampling_params  # 获取采样参数
        assert sampling_params is not None  # 断言采样参数存在
        num_logprobs = sampling_params.num_logprobs  # 采样 logprobs 数量
        num_prompt_logprobs = sampling_params.prompt_logprobs  # prompt logprobs 数量
        return cls(
            tokenizer=tokenizer,  # 保存 tokenizer
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            # 累积对数概率初始化为 0.0（未请求时保持 None）
            logprobs=(
                None  # 未请求采样 logprobs 时为 None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
                # 否则根据 flat_logprobs 标志创建容器
            ),
            prompt_logprobs=(
                None  # 未请求 prompt logprobs 时为 None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
                # 否则创建 prompt logprobs 容器
            ),
            num_prompt_logprobs=num_prompt_logprobs,  # 保存 prompt logprobs 数量
            num_logprobs=num_logprobs,  # 保存采样 logprobs 数量
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """
        # 从 EngineCore 更新采样 logprobs。
        # 只有当 EngineCore 在先前 step 生成了 >1 个 token 时（如 spec decoding），
        # 外层列表长度才 >1。
        # 参数 logprobs_lists：logprob tokens、logprobs 和 ranks 的列表。

        assert self.num_logprobs is not None  # 断言已请求 logprobs
        assert self.logprobs is not None  # 断言容器已创建
        assert self.cumulative_logprob is not None  # 断言累积概率存在

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists
        # 解包：token ID 列表、logprobs 列表、rank 列表（第 4 个元素忽略）

        for rank_np, logprobs_np, token_ids_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst
        ):
            # 并行遍历每组（每个位置）的 rank、logprobs、token IDs
            rank = rank_np.tolist()  # numpy 数组转 Python 列表
            logprobs = logprobs_np.tolist()  # logprobs 转为列表
            token_ids = token_ids_np.tolist()  # token IDs 转为列表
            # Detokenize (non-incrementally).
            # 非增量式 detokenize（一次性解码）
            decoded_tokens: list[str] | Iterable[None]  # 解码后的 token 字符串
            if self.tokenizer is None:
                decoded_tokens = NONES  # 无 tokenizer：用 None 占位
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
                # 将 token ID 列表转为 token 字符串列表
                context_token_ids = self._get_sampled_context_ids(self.logprobs)
                # 获取此前已采样 token ID 作为上下文（用于 UTF-8 修正）
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list,
                    tokens=token_ids,  # token ID
                    context_token_ids=context_token_ids,  # 上下文
                )
                # 验证并修正包含替换字符（U+FFFD）的解码结果

            # Sampler puts the sampled logprob in first.
            # 采样器将采样的 logprob 放在第一位
            sampled_token_logprob = logprobs[0]  # 取第一个（被采样 token）的 logprob
            self.cumulative_logprob += sampled_token_logprob
            # 累积到总对数概率

            # Update with the Logprob container for this pos.
            # 更新该位置的 Logprob 容器
            append_logprobs_for_next_position(
                self.logprobs,  # 目标容器
                token_ids,  # token IDs（top-k 备选）
                logprobs,  # 对应的 logprobs
                decoded_tokens,  # 解码后的字符串
                rank,  # token 排名
                self.num_logprobs,  # 请求的 logprobs 数量
            )

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,  # prompt logprobs 张量
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """
        # 从 EngineCore 更新 prompt logprobs。
        # 参数：包含 prompt logprobs 张量的元组。

        # Prompt logprobs are enabled.
        # prompt logprobs 已启用
        assert self.num_prompt_logprobs is not None  # 断言已请求
        assert self.prompt_logprobs is not None  # 断言容器已创建

        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors
        # 解包：token IDs、logprobs、ranks 张量（第 4 个元素忽略）

        # Recover shapes.
        # 恢复张量形状
        num_prompt_tokens, num_logprobs = logprobs.shape
        # 第一维 = prompt token 数；第二维 = 每位置的 logprob 数

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        # 非增量式 detokenize。
        # 输出是扁平的：[num_tok, num_lps] -> [num_tok * num_lps]
        all_decoded_tokens: list[str] | None = (
            None  # 无 tokenizer 时为 None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(
                self.tokenizer, token_ids.flatten().tolist()
            )
            # 将扁平化的 token ID 列表一次性解码
        )

        # Pythonize the torch tensors.
        # 将 torch 张量转为 Python 列表
        prompt_token_ranks = ranks.tolist()  # rank 转为列表
        prompt_logprobs = logprobs.tolist()  # logprobs 转为列表
        token_ids_list = token_ids.tolist()  # token IDs 转为列表

        # Make Logprob for each position.
        # 为每个位置构建 Logprob 容器
        for pos in range(num_prompt_tokens):
            # 遍历每个 prompt token 位置
            # Handle flattening and UTF-8 correction per position
            # 处理每个位置的扁平化和 UTF-8 修正
            offset = pos * num_logprobs  # 扁平化偏移起始
            offset_end = offset + num_logprobs  # 扁平化偏移结束

            decoded_tokens_for_pos: list[str] | Iterable[None]
            # 该位置的解码 token 列表
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES  # 无 tokenizer：占位 None
            else:
                # Extract decoded tokens for this position
                # 提取该位置的解码 token 切片
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]
                # Context: preceding prompt tokens accumulated in
                # self.prompt_logprobs from previous loop iterations.
                # 上下文：此前循环迭代累积在前缀 token（在 self.prompt_logprobs 中）
                context_token_ids = self._get_sampled_context_ids(self.prompt_logprobs)
                # 获取已处理的 prompt token ID 作为上下文
                # Apply UTF-8 correction within this position's token boundaries
                # 在该位置 token 边界内应用 UTF-8 修正
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice,
                    tokens=token_ids_list[pos],  # 该位置的 token IDs
                    context_token_ids=context_token_ids,  # 上下文
                )

            # Update with the Logprob container for this pos.
            # 更新该位置的 Logprob 容器
            append_logprobs_for_next_position(
                self.prompt_logprobs,  # 目标容器
                token_ids_list[pos],  # 该位置 token IDs
                prompt_logprobs[pos],  # 该位置 logprobs
                decoded_tokens_for_pos,  # 解码 token
                prompt_token_ranks[pos],  # rank
                self.num_prompt_logprobs,  # 请求数量
            )

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        # 弹出并返回所有请求的 prompt logprobs。
        # logprobs 处理器在一个或多个 prefill chunk 中聚合 prompt logprobs。
        # 此方法一次性返回所有 prompt logprobs 并遗忘它们。
        # 确保 RequestOutputKind.DELTA 语义正确，即所有 prompt logprobs
        # 在 prefill 结束时一次性返回。
        # 返回：请求禁用 prompt logprobs 时为 None；否则返回全部列表。
        plp = self.prompt_logprobs  # 获取当前累积的 prompt logprobs
        if plp:
            # 如果有数据
            self.prompt_logprobs = []  # 清空（重置为空列表）
        return plp  # 返回累积的数据

    @staticmethod
    def _get_sampled_context_ids(
        logprobs_source: SampleLogprobs | PromptLogprobs | None,  # logprobs 数据源
        max_context: int = 4,  # 最大上下文长度（默认 4）
    ) -> list[int]:
        """Extract recent sampled token IDs from a logprobs source.

        The sampled (or prompt) token at each position is the first
        entry, since it is always inserted first by
        append_logprobs_for_next_position.

        Args:
            logprobs_source: The logprobs container to extract from.
            max_context: Maximum number of preceding tokens to return.
                4 is sufficient for any UTF-8 multi-byte sequence.

        Returns:
            List of sampled token IDs, oldest first, most recent last.
        """
        # 从 logprobs 数据源提取近期采样的 token ID。
        # 每个位置被采样（或 prompt）的 token 是第一个条目，因为
        # append_logprobs_for_next_position 总是先插入它。
        # 参数：logprobs_source 数据源；max_context 要返回的最大前置 token 数
        # （4 足以覆盖任何 UTF-8 多字节序列）。
        # 返回：采样 token ID 列表，最旧在前，最新在后。
        if not logprobs_source:
            # 如果数据源为空
            return []  # 返回空列表

        n = len(logprobs_source)  # 数据源长度
        start = max(0, n - max_context)  # 起始索引（只保留最近 max_context 个）

        # Efficient path for FlatLogprobs: access token_ids directly.
        # FlatLogprobs 的高效路径：直接访问 token_ids
        if isinstance(logprobs_source, FlatLogprobs):
            return [
                logprobs_source.token_ids[logprobs_source.start_indices[i]]
                # 通过 start_indices 定位该位置的第一个 token ID
                for i in range(start, n)  # 遍历最近位置
                if logprobs_source.start_indices[i] < logprobs_source.end_indices[i]
                # 仅包含非空的位置
            ]

        # list[dict] path
        # list[dict] 形式的路径（非扁平结构）
        result: list[int] = []  # 结果列表
        for i in range(start, n):
            # 遍历最近位置
            entry = logprobs_source[i]  # 获取该位置条目（dict）
            if entry is not None:  # 如果非空
                result.append(next(iter(entry)))  # 取第一个键（被采样 token ID）
        return result  # 返回结果

    def _correct_decoded_token(
        self, token_id: int, context_token_ids: list[int]  # token ID 和上下文
    ) -> str:
        """Correct a decoded token that contains the replacement character.

        When byte-fallback tokenization splits multi-byte UTF-8
        characters across tokens, individual token decoding produces
        the replacement character U+FFFD. This method uses preceding
        sampled tokens as context to reconstruct the correct text.

        Args:
            token_id: The single token ID to correct.
            context_token_ids: Preceding sampled token IDs in sequential
                order (oldest first). These are the actual tokens in
                the generated sequence, NOT top-k alternatives.

        Returns:
            The corrected decoded string, or empty string if the byte
            sequence is genuinely incomplete at this point.
        """
        # 修正包含替换字符（U+FFFD）的解码 token。
        # 当 byte-fallback tokenization 将多字节 UTF-8 字符拆分到多个 token 时，
        # 单独解码每个 token 会产生替换字符 U+FFFD。此方法使用前置采样 token
        # 作为上下文，重构正确的文本。
        # 参数：token_id 要修正的单个 token ID；context_token_ids 按顺序排列的
        # 前置采样 token ID（最旧在前）。这些是生成序列中的实际 token，而非 top-k 备选。
        # 返回：修正后的解码字符串；如果字节序列此时确实不完整则返回空字符串。
        assert self.tokenizer is not None  # 断言 tokenizer 存在

        max_ctx = min(len(context_token_ids), 4)  # 最多使用 4 个上下文 token

        for num_ctx in range(1, max_ctx + 1):
            # 逐步增加上下文长度尝试
            context = context_token_ids[-num_ctx:]  # 取最近 num_ctx 个上下文 token
            full_decoded = self.tokenizer.decode(context + [token_id])
            # 解码上下文 + 目标 token 的完整序列

            if full_decoded.endswith("\ufffd"):
                # 如果仍以替换字符结尾（序列仍不完整）
                continue  # 增加上下文长度重试

            # Find the boundary between "clean" context tokens and
            # byte-fallback tokens that are part of the same incomplete
            # sequence. Byte-fallback context tokens returned "" when
            # they were processed, so their text must be attributed to
            # this completing token.
            # 找到"干净"上下文 token 与同属不完整序列的 byte-fallback token
            # 之间的边界。byte-fallback 上下文 token 在处理时返回 "",
            # 因此它们的文本必须归因于这个完成的 token。
            clean_end = len(context)  # 干净边界初始化为上下文末尾
            for j in range(len(context) - 1, -1, -1):
                # 从后向前遍历上下文
                if self.tokenizer.decode([context[j]]).endswith("\ufffd"):
                    # 如果该上下文 token 解码以替换字符结尾（是 byte-fallback）
                    clean_end = j  # 移动干净边界
                else:
                    break  # 遇到干净 token 停止

            # Decode only the clean (non-byte-fallback) prefix.
            # 仅解码干净（非 byte-fallback）前缀
            if clean_end > 0:
                # 如果有干净前缀
                clean_prefix = self.tokenizer.decode(context[:clean_end])
                # 解码干净前缀
            else:
                clean_prefix = ""  # 否则空字符串

            if full_decoded.startswith(clean_prefix):
                # 如果完整解码以干净前缀开始
                return full_decoded[len(clean_prefix):]
                # 返回去除前缀后的剩余部分（即该 token 应贡献的文本）
            # Tokenizer normalization may cause prefix mismatch.
            # tokenizer 归一化可能导致前缀不匹配
            # Find the longest common prefix between them.
            # 在两者间查找最长公共前缀
            common_len = 0  # 公共前缀长度
            for a, b in zip(clean_prefix, full_decoded):
                # 逐个字符比较
                if a != b:  # 遇到不同字符
                    break  # 停止
                common_len += 1  # 公共长度 +1
            return full_decoded[common_len:]  # 返回去除公共前缀后的部分

        return ""  # 所有尝试失败：返回空字符串

    def _verify_tokens(
        self,
        decoded_tokens_list: list[str],  # 待验证的解码 token 列表
        tokens: list[int],  # 对应的 token IDs
        context_token_ids: list[int] | None = None,  # 上下文 token IDs（可选）
    ) -> list[str]:
        """Verify and correct decoded tokens with replacement characters.

        Args:
            decoded_tokens_list: Decoded token strings to verify.
            tokens: Token IDs corresponding to decoded_tokens_list.
                These are alternatives at the SAME position (e.g.
                [sampled, top1, top2]), NOT sequential tokens.
            context_token_ids: Preceding sampled token IDs providing
                sequential context. If None, extracted from
                self.logprobs.
        """
        # 验证并修正包含替换字符的解码 token。
        # 参数：decoded_tokens_list 待验证的解码字符串；tokens 与解码列表对应的
        # token IDs，它们是同一位置的备选（如 [sampled, top1, top2]），
        # 而非顺序 token；context_token_ids 提供顺序上下文的前置采样 token ID，
        # 如果为 None 则从 self.logprobs 提取。
        if context_token_ids is None:
            # 如果未提供上下文
            context_token_ids = self._get_sampled_context_ids(self.logprobs)
            # 从 self.logprobs 提取

        corrected_decoded_token_map = dict()  # 记录需要修正的索引 → 修正值
        for idx, text in enumerate(decoded_tokens_list):
            # 遍历解码列表
            if text.endswith("\ufffd"):
                # 如果以替换字符结尾
                # Replacement char at the end means a potential
                # unfinished byte sequence from byte-fallback
                # tokenization. Correct each token independently
                # using only the sequential context.
                # 结尾的替换字符表示可能存在 byte-fallback tokenization
                # 产生的未完成字节序列。仅使用顺序上下文独立修正每个 token。
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    tokens[idx], context_token_ids  # 修正该 token
                )

        for idx, text in corrected_decoded_token_map.items():
            # 遍历需要修正的 token
            decoded_tokens_list[idx] = text  # 用修正值替换
        # 遍历所有需要修正的 token

        return decoded_tokens_list  # 返回修正后的列表

    def update_from_output(self, output: EngineCoreOutput) -> None:
        # 从引擎核心输出更新 logprobs 状态
        if output.new_logprobs is not None:
            # 如果有采样 logprobs
            self._update_sample_logprobs(output.new_logprobs)
            # 更新采样 logprobs
        if output.new_prompt_logprobs_tensors is not None:
            # 如果有 prompt logprobs 张量
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
            # 更新 prompt logprobs