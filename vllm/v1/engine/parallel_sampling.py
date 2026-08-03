# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

from copy import copy  # copy 模块：用于浅拷贝对象（复制 sampling_params）
from typing import cast  # cast：类型转换标注（让类型检查器信任我们的转换）

from vllm.outputs import CompletionOutput  # 完成输出类型（包含生成的 token、logprobs 等）
from vllm.sampling_params import RequestOutputKind, SamplingParams
# SamplingParams：采样参数（n、seed、output_kind 等）
# RequestOutputKind：输出类型枚举（FINAL_ONLY 只返回最终结果 / DELTA 流式增量 / 等）
from vllm.v1.engine import EngineCoreRequest  # 引擎核心请求类型（前端发往核心的请求载体）
from vllm.v1.metrics.stats import IterationStats  # 迭代统计（记录每步的指标数据）


class ParentRequest:
    """Info, state & processing for parallel sampling request.
    # 并行采样（n>1）请求的信息、状态与处理逻辑

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    # 存储父请求 ID 和采样参数；便于生成子请求的采样参数
    """

    request_id: str  # 父请求 ID（内部请求 ID，含 salt）
    external_req_id: str  # 外部请求 ID（用户提供的原始请求 ID）
    sampling_params: SamplingParams  # 父请求的采样参数（含 n>1）

    # To track the completion of child requests
    # 用于跟踪子请求的完成状态
    child_requests: set[str]  # 尚未完成的子请求 ID 集合

    # To aggregate child completions when not streaming
    # 非流式模式下聚合所有子请求的最终输出
    output_aggregator: list[CompletionOutput]  # 按 index 存放每个子请求的最终 CompletionOutput

    # To find the max number of generated tokens across all children
    # 统计所有子请求中生成 token 数的最大值
    max_num_generation_tokens: int

    # To efficiently obtain child sampling params
    # 高效获取子请求采样参数的缓存（避免重复拷贝）
    cached_child_sampling_params: SamplingParams | None

    def __init__(self, request: EngineCoreRequest) -> None:
        # 构造函数：从父请求提取信息
        assert request.external_req_id is not None  # 确保外部请求 ID 存在
        sampling_params = request.params  # 获取采样参数（sampling 或 pooling）
        self.request_id = request.request_id  # 保存内部请求 ID
        self.external_req_id = request.external_req_id  # 保存外部请求 ID
        self.sampling_params = sampling_params  # 保存采样参数

        self.child_requests = set()  # 初始化子请求集合（空）
        self.output_aggregator = (
            # 非流式（FINAL_ONLY）模式下：
            [cast(CompletionOutput, None)] * sampling_params.n
            # 创建长度为 n 的列表，预先占位 None，每个位置存一个子请求的最终输出
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )
        # 非 FINAL_ONLY 模式（流式）下用空列表，不聚合（直接流式返回）
        self.max_num_generation_tokens = 0  # 最大生成 token 数初始化为 0
        self.cached_child_sampling_params = None  # 子采样参数缓存初始化为 None

    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """Efficiently obtain child `sampling_params`
        # 高效获取子请求的采样参数

        If `sampling_params.seed` is not `None` then
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.
        # 如果父采样参数 seed 不为 None，则每个子请求需要
        # 父采样参数的一个唯一克隆（带唯一 seed）

        Args:
          index: index within `n` child requests
        # 参数：index — 在 n 个子请求中的索引

        Returns:
          Child `sampling_params` instance.
        # 返回：子请求的采样参数实例
        """
        seed = self.sampling_params.seed  # 读取父采样参数的 seed
        if self.cached_child_sampling_params:
            # 如果已有缓存（seed 为 None 时缓存）：
            return self.cached_child_sampling_params
            # 直接复用缓存的子采样参数（所有子请求参数相同，且无需唯一 seed）
        # Build child sampling_params
        # 构建子采样参数：
        child_sampling_params = copy(self.sampling_params)  # 浅拷贝父采样参数
        child_sampling_params.n = 1  # 子请求 n 设为 1（每个子请求只生成一个序列）
        if seed is None:
            # Cache child sampling_params for later reuse
            # seed 为 None：子请求之间参数完全相同，缓存以便后续复用
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            # seed 不为 None：每个子请求需要一个带唯一 seed 的克隆
            child_sampling_params.seed = seed + index
            # 子请求的 seed = 父 seed + 子索引（保证每个子请求使用不同随机种子，
            # 从而生成多样化的输出）
        return child_sampling_params  # 返回子采样参数

    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.
        # 获取子请求 ID 和采样参数

        Args:
          index: index within `n` child requests.
        # 参数：index — 在 n 个子请求中的索引

        Returns:
          (request ID, sampling_params) tuple
        # 返回：(请求 ID, 采样参数) 元组
        """
        child_req_id = f"{index}_{self.request_id}"
        # 子请求 ID = "{索引}_{父请求ID}"，保证唯一且可追溯回父请求
        self.child_requests.add(child_req_id)
        # 将子请求 ID 加入未完成集合（跟踪完成状态）
        return child_req_id, self._get_child_sampling_params(index)
        # 返回子请求 ID 和该子请求的采样参数

    @property
    def n(self) -> int:
        return self.sampling_params.n  # 属性：返回并行采样数量 n

    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        # 处理子请求的输出，返回 (要发送给客户端的输出列表, 父请求是否全部完成)
        already_finished_and_returned: bool = False
        # 标记：该子请求是否已经完成且结果已返回过客户端
        if completion_output.finished():
            # 如果这个子请求已完成：
            if child_request_id in self.child_requests:
                # 如果该子请求还在未完成集合中：
                self.child_requests.remove(child_request_id)
                # 从未完成集合中移除（表示它刚完成）
            else:
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                # 子请求 ID 不在集合中，说明它在之前已完成的 batch step 中
                # 已经完成并返回给客户端了（本次是重复的完成通知）
                already_finished_and_returned = True
                # 标记为"已提前完成并返回"

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # 如果是流式输出（非 FINAL_ONLY）模式：
            # If streaming, just return the current output
            # 流式模式下直接返回当前的增量输出
            #
            # DO NOT output finished and already returned child request to client again
            # 不要再次向客户端输出已提前完成并返回过的子请求
            outputs = [] if already_finished_and_returned else [completion_output]
            # 已提前返回 → 空列表（不重复发送）；否则返回当前输出
        else:
            # If not streaming, aggregate the n final outputs.
            # 非流式（FINAL_ONLY）模式：聚合 n 个子请求的最终输出
            self.output_aggregator[completion_output.index] = completion_output
            # 按子请求的 index 存储其最终输出到聚合器对应位置
            outputs = [] if self.child_requests else self.output_aggregator
            # 如果还有子请求未完成 → 返回空列表（继续等待聚合）
            # 所有子请求都完成 → 返回聚合的完整输出列表

        finished = not self.child_requests
        # 父请求"整体完成" = 没有剩余未完成的子请求
        return outputs, finished  # 返回 (输出列表, 是否全部完成)

    def observe_num_generation_tokens(self, num_generation_tokens: int):
        # 记录并更新所有子请求中的最大生成 token 数
        self.max_num_generation_tokens = max(
            num_generation_tokens, self.max_num_generation_tokens
        )  # 取当前值与历史最大值中的较大者
        return self.max_num_generation_tokens  # 返回更新后的最大值

    @staticmethod
    def observe_finished_request(
        parent_req: "ParentRequest | None",
        iteration_stats: IterationStats,
        num_generation_tokens: int,
    ):
        # 静态方法：当请求（父或子）完成时记录迭代统计
        n_param = parent_req.n if parent_req is not None else 1
        # 并行采样数：有父请求则取 n，否则为 1（单请求）

        if parent_req is not None:
            # 如果有父请求（即这是子请求完成）：
            num_generation_tokens = parent_req.observe_num_generation_tokens(
                num_generation_tokens
            )  # 更新父请求的最大生成 token 数，并取回该最大值
            # 这样多个子请求完成时，只记录一次最大的生成 token 数

        # Child requests finished, we can now record to iteration stats
        # 子请求已全部完成，现在可以记录到迭代统计
        if parent_req is None or not parent_req.child_requests:
            # 如果是普通请求完成，或者父请求的所有子请求都已完成：
            iteration_stats.max_num_generation_tokens_iter.append(num_generation_tokens)
            # 记录该请求（或这组子请求）的最大生成 token 数
            iteration_stats.n_params_iter.append(n_param)
            # 记录该请求（或这组子请求）的并行采样数