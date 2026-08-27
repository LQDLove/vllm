# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX 文件版权声明：vLLM 项目贡献者

from vllm.logger import init_logger  # 日志初始化工具
from vllm.v1.core.sched.output import SchedulerOutput  # 调度器输出类型
from vllm.v1.core.sched.scheduler import Scheduler  # 基础调度器
from vllm.request import Request, RequestStatus  # 请求对象与状态枚举

logger = init_logger(__name__)  # 初始化本模块日志器


class AsyncScheduler(Scheduler):  # 异步调度器，继承自 Scheduler
    def __init__(self, *args, **kwargs) -> None:  # 构造函数
        super().__init__(*args, **kwargs)  # 调用父类初始化
        # reusable read-only placeholder list for speculative decoding.
        # 可复用的只读占位符列表，用于投机解码
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        # 长度为 num_spec_tokens 的 -1 列表：作为 spec token 的占位初值
        self.pp_size = self.parallel_config.pipeline_parallel_size
        # 流水线并行大小：用于 PP 微批次调度计算下一步可解码步

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # 调度完成后的状态更新（重写父类方法）
        super()._update_after_schedule(scheduler_output)  # 先执行父类的通用更新
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # 取本次调度为各请求分配的投机解码 token
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        # 用下一步将调度的 draft token 数量重建占位符列表
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        # 按最新 spec token 数重建占位符列表（长度随调度变化）
        for req_id in scheduler_output.num_scheduled_tokens:
            # 遍历本次调度的所有请求（含已调度 token 数）
            request = self.requests[req_id]  # 取请求对象
            if request.is_prefill_chunk:
                continue  # 处于 prefill 分块阶段的请求跳过本逻辑

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # 标记是否存在待处理的结构化输出 token：
            # 该请求启用结构化输出且尚有未填实的占位符
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            # 该请求本步会生成 num_sampled_tokens_per_step 个新 token
            # 加上 num_spec_tokens 个 spec token。扩散模型无 AR 奖励 token
            # （num_sampled_tokens_per_step == 0），仅含画布（spec）token
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            # 该请求本次调度的 spec token 数量
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # 累加占位符数量：本步将产出的新 token + spec token 都先记为占位
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            # 为新增的 draft/spec token 添加占位符
            # 实际 spec token id 将在 worker 进程中回填
            request.spec_token_ids = self._spec_token_placeholders
            # 把占位符列表挂到请求上（worker 后续替换为真实 id）

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                # 使用 v2 runner 时：设置该请求下一次可被调度 decode 的步索引
                # （用于流水线微批次调度）
                request.next_decode_eligible_step = self.current_step + self.pp_size
                # 下一步可解码步 = 当前步 + pp_size（等流水线走完一轮再解码）

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # 用 worker 产出的新 token 更新请求（重写父类方法）
        status_before_update = request.status  # 记录更新前状态
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )
        # 调父类完成 token 追加、状态推进、停止判定

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        # 抢占时占位符已清零；陈旧投递不能再扣减占位符（否则下溢）
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            # 非陈旧投递：扣减本次实到的 token 数对应的占位符
            assert request.num_output_placeholders >= 0
            # 断言占位符不会变负（陈旧/越界投递会破坏不变量）

        # Cache the new tokens. Preempted requests should be skipped.
        # 缓存新 token。被抢占的请求应跳过
        if status_before_update == RequestStatus.RUNNING:
            # 仅更新前为 RUNNING 的请求才缓存（被抢占的不动）
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
            # 缓存 KV block：以"已计算 token 数 - 未填实占位符数"
            # 作为实际可落盘的 token 数，避免把占位符也当成真实 token 缓存
        return new_token_ids, stopped  # 返回新 token 与是否停止
