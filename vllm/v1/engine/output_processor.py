# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

import asyncio  # asyncio：异步 I/O 框架（RequestOutputCollector 用）
from collections import defaultdict, deque
# defaultdict：带默认值字典；deque：双端队列（流式输入缓冲）
from collections.abc import Iterable  # Iterable：可迭代对象类型
from dataclasses import dataclass  # dataclass：数据类装饰器
from typing import Any, cast  # Any：通用类型；cast：类型转换标注

import numpy as np  # numpy：科学计算库（routed experts 拼接）
import torch  # torch：PyTorch

from vllm.lora.request import LoRARequest  # LoRA 请求
from vllm.outputs import (
    STREAM_FINISHED,  # 流式完成哨兵
    CompletionOutput,  # 完成输出
    PoolingOutput,  # 池化输出
    PoolingRequestOutput,  # 池化请求输出
    RequestOutput,  # 请求输出
)
from vllm.sampling_params import RequestOutputKind  # 输出类型枚举
from vllm.tokenizers import TokenizerLike  # tokenizer 接口类型
from vllm.tracing import (
    SpanAttributes,  # 分布式追踪 Span 属性
    SpanKind,  # Span 类型
    extract_trace_context,  # 提取追踪上下文
    instrument_manual,  # 手动追踪工具
)
from vllm.utils import length_from_prompt_token_ids_or_embeds  # 计算序列长度
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
# 引擎核心输出；引擎核心请求；完成原因
from vllm.v1.engine.detokenizer import IncrementalDetokenizer  # 增量式 detokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor  # logprobs 处理器
from vllm.v1.engine.parallel_sampling import ParentRequest  # 并行采样父请求
from vllm.v1.metrics.stats import (
    IterationStats,  # 迭代统计
    LoRARequestStates,  # LoRA 请求状态
    RequestStateStats,  # 请求状态统计
    SchedulerStats,  # 调度器统计
)

# shared empty CPU tensor used as a placeholder pooling output
# 共享的空 CPU 张量，用作池化输出的占位符
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")
# 空 CPU 张量


class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """
    # 按单个请求收集流式 RequestOutput，
    # 用于传递给消费的 asyncio generate 任务。
    # 流式 delta 模式下，如果生产者超过消费者，RequestOutput 会被合并。

    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        # 构造函数
        self.aggregate = output_kind == RequestOutputKind.DELTA
        # 是否聚合（仅在 DELTA 模式下聚合）
        self.request_id = request_id  # 请求 ID
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
        # 当前输出（单槽位）
        self.ready = asyncio.Event()  # 就绪事件（有数据时设置）

        self._input_stream_task: asyncio.Task | None = None
        # 输入流处理任务（流式输入用）

    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        # 非阻塞放入操作
        if self.output is None or isinstance(output, Exception):
            # 如果当前无输出或放入的是异常
            self.output = output  # 直接保存
            self.ready.set()  # 设置就绪事件
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            # 这确保具有不同请求索引（n>1 时）的请求输出不会互相覆盖。
            self.output.add(output, aggregate=self.aggregate)
            # 合并输出（聚合或覆盖）
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output  # 池化输出直接替换

    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get operation blocks on put event."""
        # 获取操作：阻塞等待放入事件
        while (output := self.output) is None:
            # 循环直到有输出
            await self.ready.wait()  # 等待就绪事件
        self.output = None  # 清空输出
        self.ready.clear()  # 清除就绪事件
        if isinstance(output, Exception):
            # 如果输出是异常
            raise output  # 抛出异常
        return output  # 返回输出

    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """Non-blocking get operation."""
        # 非阻塞获取操作
        output = self.output  # 获取输出
        if output is not None:
            # 如果有输出
            self.output = None  # 清空
            self.ready.clear()  # 清除就绪
        if isinstance(output, Exception):
            # 如果输出是异常
            raise output  # 抛出异常
        return output  # 返回输出（可能为 None）

    def close(self):
        # 关闭收集器
        if self._input_stream_task is not None:
            # 如果有输入流任务
            self._input_stream_task.cancel()  # 取消任务

    def __del__(self):
        # 析构函数
        if (task := self._input_stream_task) is not None:
            # 如果有输入流任务
            task.get_loop().call_soon_threadsafe(task.cancel)
            # 线程安全取消任务
            self._input_stream_task = None  # 清除引用


@dataclass
class OutputProcessorOutput:
    # 输出处理器输出结果
    request_outputs: list[RequestOutput | PoolingRequestOutput]  # 请求输出列表
    reqs_to_abort: list[str]  # 需要中止的请求 ID 列表


@dataclass
class StreamingUpdate:
    """Streaming input update data for output processor.

    Contains the incremental prompt data to be applied to a request state
    when the current sub-request completes.
    """
    # 流式输入更新数据。
    # 包含当前子请求完成时要应用到请求状态的增量 prompt 数据。

    prompt: str | None  # 增量 prompt 文本
    prompt_token_ids: list[int] | None  # 增量 prompt token IDs
    arrival_time: float  # 到达时间
    final: bool = False  # 是否最终更新


class RequestState:
    # 请求状态：管理单个请求在前端的全部状态
    def __init__(
        self,
        request_id: str,  # 内部请求 ID
        external_req_id: str,  # 外部请求 ID
        parent_req: ParentRequest | None,  # 父请求（可选）
        request_index: int,  # 子请求索引
        lora_request: LoRARequest | None,  # LoRA 请求（可选）
        output_kind: RequestOutputKind,  # 输出类型
        prompt: str | None,  # prompt 文本（可选）
        prompt_token_ids: list[int] | None,  # prompt token IDs
        prompt_embeds: torch.Tensor | None,  # prompt embeddings（可选）
        logprobs_processor: LogprobsProcessor | None,  # logprobs 处理器
        detokenizer: IncrementalDetokenizer | None,  # detokenizer
        max_tokens_param: int | None,  # max_tokens 参数
        arrival_time: float,  # 到达时间
        queue: RequestOutputCollector | None,  # 输出收集器
        log_stats: bool,  # 是否记录统计
        stream_interval: int,  # 流式间隔
        top_p: float | None = None,  # top_p 参数（可选）
        n: int | None = None,  # 并行采样数（可选）
        temperature: float | None = None,  # 温度参数（可选）
        stream_input: bool = False,  # 是否流式输入
    ):
        self.request_id = request_id  # 内部请求 ID
        self.external_req_id = external_req_id  # 外部请求 ID
        self.parent_req = parent_req  # 父请求
        self.request_index = request_index  # 子请求索引
        self.lora_request = lora_request  # LoRA 请求
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        # LoRA 名称
        self.output_kind = output_kind  # 输出类型
        self.prompt = prompt  # prompt 文本
        self.prompt_token_ids = prompt_token_ids  # prompt token IDs
        self.prompt_embeds = prompt_embeds  # prompt embeddings
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        # 计算 prompt 长度
        self.logprobs_processor = logprobs_processor  # logprobs 处理器
        self.detokenizer = detokenizer  # detokenizer
        self.max_tokens_param = max_tokens_param  # max_tokens 参数
        self.top_p = top_p  # top_p
        self.n = n  # 并行采样数
        self.temperature = temperature  # 温度
        self.is_prefilling = True  # 是否正在 prefill
        self.queue = queue  # 输出收集器
        self.num_cached_tokens = 0  # 缓存 token 数
        self.num_cache_creation_tokens = 0  # 缓存创建 token 数

        self.stats = RequestStateStats(arrival_time=arrival_time) if log_stats else None
        # 创建请求状态统计（如果启用日志）

        # Routed experts accumulation (prompt + sample chunks)
        # 路由专家累积（prompt + 采样块）
        self.routed_experts_chunks: list[np.ndarray] = []
        # 路由专家块列表

        # Stream Interval
        # 流式间隔
        self.stream_interval = stream_interval  # 流式间隔
        self.sent_tokens_offset = 0  # Offset of sent tokens
        # 已发送 token 的偏移量

        # Streaming input queue
        # 流式输入队列
        self.streaming_input = stream_input  # 是否流式输入
        self.input_chunk_queue: deque[StreamingUpdate] | None = (
            deque() if stream_input else None
        )
        # 流式输入块队列（仅流式输入时创建）

    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        # 应用流式更新
        self.streaming_input = not update.final  # 更新流式输入标志
        # TODO also include relevant output tokens in new prompt here
        #     (match scheduler behavior).
        # TODO：此处也应包括新 prompt 中的相关输出 token（匹配调度器行为）。
        if update.prompt:
            # 如果有 prompt 文本
            self.prompt = (
                (self.prompt + update.prompt) if self.prompt else update.prompt
            )
            # 拼接 prompt 文本
        if self.prompt_token_ids:
            # 如果有 token IDs
            self.prompt_token_ids.extend(update.prompt_token_ids or ())
            # 扩展 token IDs
        else:
            self.prompt_token_ids = update.prompt_token_ids or []
            # 否则直接设置
        assert self.prompt_token_ids is not None  # 断言非空
        self.prompt_len = len(self.prompt_token_ids)  # 更新 prompt 长度
        if self.stats is not None:
            # 如果有统计
            self.stats.arrival_time = update.arrival_time  # 更新到达时间
        self.is_prefilling = True  # 标记重新 prefill

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,  # tokenizer（可选）
        request: EngineCoreRequest,  # 引擎核心请求
        prompt: str | None,  # prompt 文本
        parent_req: ParentRequest | None,  # 父请求
        request_index: int,  # 子请求索引
        queue: RequestOutputCollector | None,  # 输出收集器
        log_stats: bool,  # 是否记录统计
        stream_interval: int,  # 流式间隔
    ) -> "RequestState":
        # 工厂方法：根据新请求创建请求状态
        if sampling_params := request.sampling_params:
            # 如果有采样参数（生成任务）
            if not sampling_params.detokenize:
                # 如果禁用了 detokenize
                tokenizer = None  # 不使用 tokenizer
            output_kind = sampling_params.output_kind  # 输出类型
            if sampling_params.stream_interval is not None:
                # 如果请求指定了流式间隔
                # clamp to the engine-level stream interval.
                # 限制到引擎级流式间隔（取较大值）。
                stream_interval = max(sampling_params.stream_interval, stream_interval)
                # 取较大值
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,  # tokenizer
                request=request,  # 请求
            )
            # 创建 logprobs 处理器
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,  # tokenizer
                request=request,  # 请求
            )
            # 创建 detokenizer
            max_tokens_param = sampling_params.max_tokens  # max_tokens
            top_p = sampling_params.top_p  # top_p
            n = sampling_params.n  # 并行采样数
            temperature = sampling_params.temperature  # 温度
        else:
            # 否则是池化请求
            logprobs_processor = None  # 无 logprobs
            detokenizer = None  # 无 detokenizer
            max_tokens_param = None  # 无 max_tokens
            top_p = None  # 无 top_p
            n = None  # 无 n
            temperature = None  # 无温度
            assert request.pooling_params is not None  # 断言池化参数存在
            output_kind = request.pooling_params.output_kind  # 输出类型

        assert request.external_req_id is not None  # 断言外部 ID 存在
        return cls(
            request_id=request.request_id,  # 内部请求 ID
            external_req_id=request.external_req_id,  # 外部请求 ID
            parent_req=parent_req,  # 父请求
            request_index=request_index,  # 子请求索引
            lora_request=request.lora_request,  # LoRA
            output_kind=output_kind,  # 输出类型
            prompt=prompt,  # prompt 文本
            prompt_token_ids=request.prompt_token_ids,  # prompt token IDs
            prompt_embeds=request.prompt_embeds,  # prompt embeddings
            logprobs_processor=logprobs_processor,  # logprobs 处理器
            detokenizer=detokenizer,  # detokenizer
            max_tokens_param=max_tokens_param,  # max_tokens
            top_p=top_p,  # top_p
            n=n,  # 并行采样数
            temperature=temperature,  # 温度
            arrival_time=request.arrival_time,  # 到达时间
            queue=queue,  # 输出收集器
            log_stats=log_stats,  # 日志统计
            stream_interval=stream_interval,  # 流式间隔
            stream_input=request.resumable,  # 流式输入标志
        )

    def make_request_output(
        self,
        new_token_ids: list[int],  # 新 token IDs
        pooling_output: torch.Tensor | None,  # 池化输出（可能）
        finish_reason: FinishReason | None,  # 完成原因
        stop_reason: int | str | None,  # 停止原因
        kv_transfer_params: dict[str, Any] | None = None,  # KV 传输参数（可选）
        ec_transfer_params: dict[str, Any] | None = None,  # 专家缓存传输参数（可选）
    ) -> RequestOutput | PoolingRequestOutput | None:
        # 创建请求输出
        finished = finish_reason is not None  # 是否完成
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY
        # 是否仅最终输出

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            # 仅在 FINAL_ONLY 模式需要最终输出。
            return None  # 返回 None

        if self.stream_interval > 1:
            # 如果流式间隔 > 1
            assert self.detokenizer is not None  # 断言 detokenizer 存在

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            # 仅在以下情况发送输出：
            # 1. 已完成，或
            # 2. 是第一个 token，或
            # 3. 达到流式间隔数量的 token
            if not (
                finished  # 已完成
                or self.sent_tokens_offset == 0  # 第一个 token
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                >= self.stream_interval  # 达到间隔
            ):
                return None  # 不发送

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                # DELTA 模式从偏移处发送 token，否则发送所有 token。
                new_token_ids = self.detokenizer.output_token_ids[
                    self.sent_tokens_offset:
                ]
                # 获取增量 token IDs
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()
                # 更新偏移

        external_req_id = self.external_req_id  # 外部请求 ID

        if pooling_output is not None:
            # 如果是池化输出
            return self._new_request_output(
                external_req_id,  # 外部 ID
                [self._new_pooling_output(pooling_output)],  # 池化输出
                finished,  # 完成标志
            )
            # 返回池化请求输出

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)
        # 创建完成输出

        if self.parent_req is None:
            # 如果没有父请求
            outputs = [output]  # 单个输出
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            # 通过父请求聚合输出
            if not outputs:
                # 如果无输出（还有子请求未完成）
                return None  # 返回 None
            external_req_id = self.parent_req.external_req_id  # 使用外部 ID

        return self._new_request_output(
            external_req_id,  # 外部 ID
            outputs,  # 输出列表
            finished,  # 完成标志
            kv_transfer_params,  # KV 传输参数
            ec_transfer_params,  # 专家缓存传输参数
        )
        # 返回请求输出

    def _new_request_output(
        self,
        external_req_id: str,  # 外部请求 ID
        outputs: list[CompletionOutput] | list[PoolingOutput],  # 输出列表
        finished: bool,  # 完成标志
        kv_transfer_params: dict[str, Any] | None = None,  # KV 传输参数（可选）
        ec_transfer_params: dict[str, Any] | None = None,  # 专家缓存传输参数（可选）
    ) -> RequestOutput | PoolingRequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        # 如果使用了 prompt embeddings，使用占位 prompt token IDs
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
            # 用 0 填充
        assert prompt_token_ids is not None  # 断言非空

        first_output = outputs[0]  # 第一个输出
        if isinstance(first_output, PoolingOutput):
            # 如果是池化输出
            assert len(outputs) == 1  # 断言只有一个
            return PoolingRequestOutput(
                request_id=external_req_id,  # 外部请求 ID
                outputs=first_output,  # 池化输出
                num_cached_tokens=self.num_cached_tokens,  # 缓存 token 数
                prompt_token_ids=prompt_token_ids,  # prompt token IDs
                finished=finished,  # 完成标志
            )
        assert self.logprobs_processor is not None  # 断言 logprobs 处理器存在
        if self.output_kind == RequestOutputKind.DELTA:
            # 如果是 DELTA 模式
            # Side effect: logprobs processor forgets prompt logprobs
            # 副作用：logprobs 处理器遗忘 prompt logprobs（一次性弹出）
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
            # 弹出并返回 prompt logprobs
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs
            # 直接引用

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            # request_id 是外部提供的原始 ID
            lora_request=self.lora_request,  # LoRA 请求
            prompt=self.prompt,  # prompt 文本
            prompt_token_ids=prompt_token_ids,  # prompt token IDs
            prompt_logprobs=prompt_logprobs,  # prompt logprobs
            outputs=cast(list[CompletionOutput], outputs),  # 完成输出列表
            finished=finished,  # 完成标志
            kv_transfer_params=kv_transfer_params,  # KV 传输参数
            ec_transfer_params=ec_transfer_params,  # 专家缓存传输参数
            num_cached_tokens=self.num_cached_tokens,  # 缓存 token 数
            num_cache_creation_tokens=self.num_cache_creation_tokens,
            # 缓存创建 token 数
            metrics=self.stats,  # 请求统计（指标）
        )

    def _new_completion_output(
        self,
        token_ids: list[int],  # 新 token IDs
        finish_reason: FinishReason | None,  # 完成原因
        stop_reason: int | str | None,  # 停止原因
    ) -> CompletionOutput:
        # 创建完成输出
        assert self.detokenizer is not None  # 断言 detokenizer 存在
        assert self.logprobs_processor is not None  # 断言 logprobs 处理器存在
        finished = finish_reason is not None  # 是否完成
        delta = self.output_kind == RequestOutputKind.DELTA  # 是否 DELTA 模式

        # Prepare text and token_ids, based on delta mode
        # 根据 DELTA 模式准备文本和 token IDs
        text = self.detokenizer.get_next_output_text(finished, delta)
        # 获取下一个输出文本
        if not delta:
            # 非 DELTA 模式
            token_ids = self.detokenizer.output_token_ids
            # 使用全部 token IDs

        # Prepare logprobs, based on delta mode
        # 根据 DELTA 模式准备 logprobs
        logprobs = self.logprobs_processor.logprobs  # 全部 logprobs
        if delta and logprobs:
            # DELTA 模式且有 logprobs
            logprobs = logprobs[-len(token_ids):]
            # 只取最近 token 数量的 logprobs

        # Concatenate routed experts on finish
        # 完成时拼接路由专家
        routed_experts = None  # 路由专家
        if finished and self.routed_experts_chunks:
            # 如果完成且有多块的专家数据
            routed_experts = np.concatenate(self.routed_experts_chunks, axis=0)
            # 拼接所有块

        return CompletionOutput(
            index=self.request_index,  # 请求索引
            text=text,  # 文本
            token_ids=token_ids,  # token IDs
            routed_experts=routed_experts,  # 路由专家
            logprobs=logprobs,  # logprobs
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            # 累积对数概率
            finish_reason=str(finish_reason) if finished else None,
            # 完成原因（字符串）
            stop_reason=stop_reason if finished else None,
            # 停止原因（仅完成时）
        )

    def _new_pooling_output(self, pooling_output: torch.Tensor) -> PoolingOutput:
        # 创建池化输出
        return PoolingOutput(data=pooling_output)  # 包装池化数据


class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""
    # 将 EngineCoreOutputs 处理为 RequestOutputs

    def __init__(
        self,
        tokenizer: TokenizerLike | None,  # tokenizer（可 None）
        *,
        log_stats: bool,  # 是否记录统计
        stream_interval: int = 1,  # 流式间隔（默认 1）
        tracing_enabled: bool = False,  # 是否启用追踪
    ):
        self.log_stats = log_stats  # 日志统计标志
        self.tokenizer = tokenizer  # tokenizer
        self.stream_interval = stream_interval  # 流式间隔
        self.request_states: dict[str, RequestState] = {}
        # 请求状态字典（内部请求 ID → 请求状态）
        self.parent_requests: dict[str, ParentRequest] = {}
        # 父请求字典
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        # 外部请求 ID → 内部请求 ID 列表映射
        self.lora_states = LoRARequestStates(log_stats)  # LoRA 请求状态
        self.tracing_enabled = tracing_enabled  # 追踪标志

    def get_num_unfinished_requests(self):
        # 获取未完成请求数
        return len(self.request_states)  # 返回字典长度

    def has_unfinished_requests(self) -> bool:
        # 是否有未完成请求
        return len(self.request_states) > 0  # 字典非空

    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""
        # 向所有 generate() 任务传播错误
        for _, state in self.request_states.items():
            # 遍历所有请求状态
            assert state.queue is not None  # 断言队列存在
            state.queue.put(e)  # 推送错误到队列

    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        # 中止请求列表。
        # request_ids 可以是外部请求 ID（传给 InputProcessor.process_inputs() 的）
        # 或内部请求 ID（创建 EngineCoreRequest 时随机生成的）。
        # 如果提供外部请求 ID，且该外部请求 ID 被多个请求使用，
        # 则与该外部请求 ID 关联的所有请求都会被中止。
        # 在并行采样情况下，请求 ID 可能用于标识父请求，
        # 此时关联的子请求也会被中止。
        internal_req_ids = []  # 内部请求 ID 列表
        for request_id in request_ids:
            # 遍历请求 ID
            if internal:
                # Internal ID - this may be a parent request
                # 内部 ID - 可能是父请求
                internal_req_ids.append(request_id)  # 添加到列表

                # Remove internal ID from the external->internal mapping
                # 从外部到内部映射中移除内部 ID
                if req_state := self.request_states.get(request_id):
                    # 如果找到请求状态
                    external_req_id = req_state.external_req_id  # 外部 ID
                    internal_ids = self.external_req_ids[external_req_id]
                    # 内部 ID 列表
                    internal_ids.remove(request_id)  # 移除
                    if not internal_ids:
                        # 如果列表为空
                        del self.external_req_ids[external_req_id]
                        # 删除映射
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                # 外部 ID - 中止映射中的所有请求
                internal_req_ids.extend(internal_ids)  # 扩展列表

        request_ids_to_abort = []  # 需要中止的请求 ID
        for request_id in internal_req_ids:
            # 遍历内部请求 ID
            req_state = self.request_states.pop(request_id, None)
            # 弹出请求状态
            if req_state is not None:
                # 如果有请求状态
                self.lora_states.request_finished(request_id, req_state.lora_name)
                # 记录 LoRA 请求完成
                request_ids_to_abort.append(request_id)  # 添加到中止列表
                # Produce final abort output.
                # 产生最终中止输出。
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],  # 无新 token
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        # 设置 pooling_output 非 None 以正确进入中止池化分支
                        pooling_output=EMPTY_CPU_TENSOR  # 空张量占位
                        if req_state.detokenizer is None  # 无 detokenizer
                        else None,
                        finish_reason=FinishReason.ABORT,  # 中止原因
                        stop_reason=None,  # 无停止原因
                        kv_transfer_params=None,  # 无 KV 传输
                        ec_transfer_params=None,  # 无专家缓存传输
                    )
                ):
                    # 如果创建了中止输出
                    req_state.queue.put(request_output)  # 推送输出
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                # 移除父请求前先中止子请求。
                if parent.child_requests:
                    # 如果有子请求
                    child_reqs = list(parent.child_requests)  # 子请求列表
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    # 递归中止子请求
                    request_ids_to_abort.extend(child_reqs)  # 扩展列表
                self.parent_requests.pop(request_id, None)
                # 移除父请求
        return request_ids_to_abort  # 返回需要中止的请求 ID

    def add_request(
        self,
        request: EngineCoreRequest,  # 引擎核心请求
        prompt: str | None,  # prompt 文本
        parent_req: ParentRequest | None = None,  # 父请求（可选）
        request_index: int = 0,  # 子请求索引
        queue: RequestOutputCollector | None = None,  # 输出收集器（可选）
    ) -> None:
        # 添加请求
        request_id = request.request_id  # 内部请求 ID
        req_state = self.request_states.get(request_id)  # 查找请求状态
        if req_state is not None:
            # 如果已存在（流式输入的后续块）
            self._update_streaming_request_state(req_state, request, prompt)
            # 更新流式请求状态
            return  # 返回

        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,  # tokenizer
            request=request,  # 请求
            prompt=prompt,  # prompt 文本
            parent_req=parent_req,  # 父请求
            request_index=request_index,  # 子请求索引
            queue=queue,  # 输出收集器
            log_stats=self.log_stats,  # 日志统计
            stream_interval=self.stream_interval,  # 流式间隔
        )
        self.request_states[request_id] = req_state  # 保存请求状态
        if parent_req:
            # 如果有父请求
            self.parent_requests[parent_req.request_id] = parent_req
            # 保存父请求

        # Track the external_req_id -> [internal_req_id, ...] mapping
        # 跟踪外部请求 ID → [内部请求 ID, ...] 映射
        self.external_req_ids[req_state.external_req_id].append(request_id)
        # 添加映射

    def _update_streaming_request_state(
        self, req_state: RequestState, request: EngineCoreRequest, prompt: str | None
    ) -> None:
        """Queue a streaming update instead of immediately applying it."""
        # 排队流式更新而不是立即应用
        if not request.resumable:
            # Final request - just mark completion, don't add its dummy tokens.
            # 最终请求 - 仅标记完成，不添加其占位 token。
            if req_state.input_chunk_queue is None:
                # Engine already finished - emit final output and clean up.
                # 引擎已完成 - 发送最终输出并清理。
                self._finish_request(req_state)  # 完成请求
                if req_state.queue is not None:
                    # Emit a final output with finished=True
                    # to unblock the generate() loop.
                    # 发送完成输出以解除 generate() 循环阻塞。
                    req_state.queue.put(STREAM_FINISHED)  # 推送流完成哨兵
            elif req_state.input_chunk_queue:
                # 如果有输入块
                req_state.input_chunk_queue[-1].final = True
                # 标记最后一个块为最终
            else:
                req_state.streaming_input = False  # 标记非流式
            return  # 返回

        update = StreamingUpdate(
            prompt=prompt,  # 增量 prompt
            prompt_token_ids=request.prompt_token_ids,  # 增量 token IDs
            arrival_time=request.arrival_time,  # 到达时间
        )
        # 创建流式更新

        # Apply request updates now if the last input already completed.
        # 如果上次输入已完成，立即应用请求更新。
        if req_state.input_chunk_queue is None:
            # 如果无输入块队列
            req_state.apply_streaming_update(update)  # 立即应用
            req_state.input_chunk_queue = deque()  # 创建队列
        else:
            # Queue the streaming update otherwise.
            # 否则排队流式更新。
            req_state.input_chunk_queue.append(update)  # 追加到队列

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],  # 引擎核心输出列表
        engine_core_timestamp: float | None = None,  # 引擎核心时间戳（可选）
        iteration_stats: IterationStats | None = None,  # 迭代统计（可选）
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.

            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.

        If you need to touch every element of the batch, do it from
        within the loop below.
        """
        # 处理 EngineCoreOutputs：
        # 1) 为日志计算统计
        # 2) Detokenize
        # 3) 创建并处理 RequestOutput 对象：
        #    - 如果有队列（AsyncLLM 使用），将 RequestOutput 放入队列
        #      由 per-request generate() 任务处理。
        #    - 如果没有队列（LLMEngine 使用），返回 RequestOutput 列表。
        # 开发者注意事项：
        # vLLM V1 最小化对整个批次的 Python 循环次数以确保系统开销最小。
        # 这是唯一应该循环 EngineCoreOutputs 的函数。
        # 如需接触批次每个元素，应在下方循环内进行。

        request_outputs: list[RequestOutput | PoolingRequestOutput] = []
        # 请求输出列表
        reqs_to_abort: list[str] = []  # 需要中止的请求
        for engine_core_output in engine_core_outputs:
            # 遍历引擎核心输出
            req_id = engine_core_output.request_id  # 请求 ID
            req_state = self.request_states.get(req_id)  # 查找请求状态
            if req_state is None:
                # Ignore output for already-aborted request.
                # 忽略已中止请求的输出。
                continue  # 继续

            # 1) Compute stats for this iteration.
            # 1) 为此迭代计算统计。
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )
            # 更新统计

            new_token_ids = engine_core_output.new_token_ids  # 新 token IDs
            pooling_output = engine_core_output.pooling_output  # 池化输出
            finish_reason = engine_core_output.finish_reason  # 完成原因
            stop_reason = engine_core_output.stop_reason  # 停止原因
            kv_transfer_params = engine_core_output.kv_transfer_params  # KV 传输
            ec_transfer_params = engine_core_output.ec_transfer_params  # 专家缓存
            if engine_core_output.routed_experts is not None:
                # 如果有路由专家数据
                req_state.routed_experts_chunks.append(
                    engine_core_output.routed_experts
                )
                # 追加到块列表

            if req_state.is_prefilling:
                # 如果正在 prefill
                if engine_core_output.prefill_stats is not None:
                    # 如果有 prefill 统计
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                    # 更新缓存 token 数
                    req_state.num_cache_creation_tokens = (
                        engine_core_output.prefill_stats.num_cache_creation_tokens
                    )
                    # 更新缓存创建 token 数
                req_state.is_prefilling = False  # 标记 prefill 完成

            if pooling_output is None:
                # 如果是生成输出（非池化）
                assert req_state.detokenizer is not None  # 断言 detokenizer 存在
                assert req_state.logprobs_processor is not None  # 断言处理器存在
                # 2) Detokenize the token ids into text and perform stop checks.
                # 2) 将 token IDs 解码为文本并执行停止检查。
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                # 更新 detokenizer（增量解码）
                if stop_string:
                    # 如果匹配到停止字符串
                    finish_reason = FinishReason.STOP  # 更新完成原因
                    stop_reason = stop_string  # 更新停止原因

                # 3) Compute sample and prompt logprobs for request,
                # if required.
                # 3) 如需要，为请求计算采样和 prompt logprobs。
                req_state.logprobs_processor.update_from_output(engine_core_output)
                # 更新 logprobs

            # 4) Create and handle RequestOutput objects.
            # 4) 创建并处理 RequestOutput 对象。
            if request_output := req_state.make_request_output(
                new_token_ids,  # 新 token IDs
                pooling_output,  # 池化输出
                finish_reason,  # 完成原因
                stop_reason,  # 停止原因
                kv_transfer_params,  # KV 传输参数
                ec_transfer_params,  # 专家缓存传输参数
            ):
                # 如果创建了请求输出
                if req_state.streaming_input:
                    # 如果是流式输入
                    request_output.finished = False  # 强制标记未完成

                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    # AsyncLLM：放入队列由 generate() 处理。
                    req_state.queue.put(request_output)  # 推送输出
                else:
                    # LLMEngine: return list of RequestOutputs.
                    # LLMEngine：返回 RequestOutput 列表。
                    request_outputs.append(request_output)  # 添加到列表

            # Free completed requests.
            # 释放已完成的请求。
            if finish_reason is not None:
                # 如果请求已完成
                if req_state.streaming_input:
                    # 如果是流式输入
                    if req_state.input_chunk_queue:
                        # 如果有输入块
                        update = req_state.input_chunk_queue.popleft()
                        # 弹出最早的输入块
                        req_state.apply_streaming_update(update)  # 应用更新
                    else:
                        req_state.input_chunk_queue = None  # 清空队列
                else:
                    self._finish_request(req_state)  # 完成请求
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        # 如果请求在 EngineCore 未完成，但 Detokenizer
                        # 检测到停止字符串，需要在 EngineCore 中止。
                        reqs_to_abort.append(req_id)  # 添加到中止列表

                    # Track per-request stats
                    # 跟踪每请求统计
                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    # 更新完成统计
                    if self.tracing_enabled:
                        # 如果启用追踪
                        self.do_tracing(engine_core_output, req_state, iteration_stats)
                        # 执行追踪

        return OutputProcessorOutput(
            request_outputs=request_outputs,  # 请求输出列表
            reqs_to_abort=reqs_to_abort,  # 需要中止的请求
        )
        # 返回处理结果

    def _finish_request(self, req_state: RequestState) -> None:
        # 完成请求并清理
        req_id = req_state.request_id  # 内部请求 ID
        self.request_states.pop(req_id)  # 弹出请求状态

        internal_ids = self.external_req_ids[req_state.external_req_id]
        # 获取内部 ID 列表
        internal_ids.remove(req_id)  # 移除
        if not internal_ids:
            # 如果列表为空
            del self.external_req_ids[req_state.external_req_id]
            # 删除映射

        # Remove parent request if applicable.
        # 如适用，移除父请求。
        parent_req = req_state.parent_req  # 父请求
        if parent_req and not parent_req.child_requests:
            # 如果有父请求且无子请求
            self.parent_requests.pop(parent_req.request_id, None)
            # 移除父请求

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        # 更新调度器统计
        self.lora_states.update_scheduler_stats(scheduler_stats)  # 委托 LoRA 状态

    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,  # 引擎核心输出
        req_state: RequestState,  # 请求状态
        iteration_stats: IterationStats | None,  # 迭代统计
    ) -> None:
        # 执行分布式追踪
        assert req_state.stats is not None  # 断言统计存在
        assert iteration_stats is not None  # 断言迭代统计存在

        metrics = req_state.stats  # 请求统计
        arrival_time_ns = int(metrics.arrival_time * 1e9)  # 到达时间（纳秒）
        trace_context = extract_trace_context(engine_core_output.trace_headers)
        # 提取追踪上下文
        prompt_length = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )
        # 计算 prompt 长度

        # Calculate timing metrics
        # 计算时间指标
        e2e_time = iteration_stats.iteration_timestamp - metrics.arrival_time
        # 端到端时间
        queued_time = metrics.scheduled_ts - metrics.queued_ts  # 排队时间
        prefill_time = metrics.first_token_ts - metrics.scheduled_ts  # prefill 时间
        decode_time = metrics.last_token_ts - metrics.first_token_ts  # 解码时间
        inference_time = metrics.last_token_ts - metrics.scheduled_ts  # 推理时间

        # Build attributes dict
        # 构建属性字典
        attributes: dict[str, Any] = {
            SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: (
                metrics.first_token_latency  # 首个 token 延迟
            ),
            SpanAttributes.GEN_AI_LATENCY_E2E: e2e_time,  # 端到端延迟
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE: queued_time,  # 排队时间
            SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: prompt_length,  # prompt token
            SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: (
                metrics.num_generation_tokens  # 生成 token 数
            ),
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: prefill_time,
            # prefill 时间
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: decode_time,
            # 解码时间
            SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: inference_time,
            # 推理时间
            SpanAttributes.GEN_AI_REQUEST_ID: req_state.external_req_id,
            # 请求 ID
        }

        # Add optional request parameters
        # 添加可选请求参数
        if req_state.top_p:
            # 如果有 top_p
            attributes[SpanAttributes.GEN_AI_REQUEST_TOP_P] = req_state.top_p
            # 添加 top_p
        if req_state.max_tokens_param:
            # 如果有 max_tokens
            attributes[SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS] = (
                req_state.max_tokens_param
            )
            # 添加 max_tokens
        if req_state.temperature:
            # 如果有温度
            attributes[SpanAttributes.GEN_AI_REQUEST_TEMPERATURE] = (
                req_state.temperature
            )
            # 添加温度
        if req_state.n:
            # 如果有并行采样数
            attributes[SpanAttributes.GEN_AI_REQUEST_N] = req_state.n
            # 添加 n

        instrument_manual(
            span_name="llm_request",  # Span 名称
            start_time=arrival_time_ns,  # 开始时间
            attributes=attributes,  # 属性
            context=trace_context,  # 追踪上下文
            kind=SpanKind.SERVER,  # Span 类型（服务器）
        )
        # 执行手动追踪

    def _update_stats_from_output(
        self,
        req_state: RequestState,  # 请求状态
        engine_core_output: EngineCoreOutput,  # 引擎核心输出
        engine_core_timestamp: float | None,  # 引擎核心时间戳
        iteration_stats: IterationStats | None,  # 迭代统计
    ):
        # 从输出更新统计
        if iteration_stats is None:
            # 如果无迭代统计
            return  # 返回

        assert engine_core_timestamp is not None  # 断言时间戳存在
        assert req_state.stats is not None  # 断言请求统计存在
        iteration_stats.update_from_output(
            engine_core_output,  # 引擎核心输出
            engine_core_timestamp,  # 时间戳
            req_state.is_prefilling,  # 是否 prefill
            req_state.stats,  # 请求统计
            self.lora_states,  # LoRA 状态
            req_state.lora_name,  # LoRA 名称
        )
        # 更新迭代统计

    def _update_stats_from_finished(
        self,
        req_state: RequestState,  # 请求状态
        finish_reason: FinishReason | None,  # 完成原因
        iteration_stats: IterationStats | None,  # 迭代统计
    ):
        # 从完成更新统计
        if iteration_stats is None:
            # 如果无迭代统计
            return  # 返回

        assert finish_reason is not None  # 断言完成原因存在
        assert req_state.stats is not None  # 断言请求统计存在
        iteration_stats.update_from_finished_request(
            finish_reason=finish_reason,  # 完成原因
            request_id=req_state.external_req_id,  # 外部请求 ID
            num_prompt_tokens=req_state.prompt_len,  # prompt token 数
            max_tokens_param=req_state.max_tokens_param,  # max_tokens 参数
            req_stats=req_state.stats,  # 请求统计
            num_cached_tokens=req_state.num_cached_tokens,  # 缓存 token 数
        )
        # 更新完成统计
        self.lora_states.request_finished(req_state.request_id, req_state.lora_name)
        # 记录 LoRA 请求完成

        ParentRequest.observe_finished_request(
            req_state.parent_req, iteration_stats, req_state.stats.num_generation_tokens
        )
        # 观察父请求完成（并行采样统计）