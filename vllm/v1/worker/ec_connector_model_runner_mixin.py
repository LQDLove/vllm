# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define EC connector functionality mixin for model runners.
"""
# 定义 EC(Encoder-Connector)连接器功能的模型 runner 混入(Mixin)。
# 用于 EPD(Encoder-Decoder Pairing)分离部署:将仅编码器实例的输出写入
# 跨实例连接器供配对解码实例消费,或从连接器加载编码器缓存。

# 导入 Generator 类型,用于标注生成器返回类型(上下文管理器实现)。
from collections.abc import Generator
# 导入上下文管理器相关工具:AbstractContextManager 抽象类、contextmanager
# 装饰器与 nullcontext(无操作上下文)。
from contextlib import AbstractContextManager, contextmanager, nullcontext
# 导入 TYPE_CHECKING,用于仅类型检查时的条件导入。
from typing import TYPE_CHECKING

# 导入 PyTorch,用于标注张量类型。
import torch

# 导入 EC 传输的全局访问器:get_ec_transfer 获取连接器,has_ec_transfer 判断是否配置。
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
# 导入 EC 连接器基类,用于类型断言。
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
# 导入日志初始化函数,用于创建模块日志记录器。
from vllm.logger import init_logger
# 导入 EC 连接器输出容器(记录发送/接收完成状态)。
from vllm.v1.outputs import ECConnectorOutput

# 仅在类型检查时导入 SchedulerOutput(避免运行时循环依赖)。
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# 创建本模块的日志记录器。
logger = init_logger(__name__)


# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)
# 定义为 Model Runner(GPU/TPU)共用的 EC 连接器功能混入。
class ECConnectorModelRunnerMixin:
    # EC 连接器混入类:提供编码器缓存的保存与读取能力,
    # 与具体模型 runner 无关,通过静态方法组合使用。

    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        # 若配置了 EC 传输,则把本轮多模态编码器缓存保存到连接器队列。
        # 参数:
        #   encoder_cache: 多模态编码器输出缓存(哈希 -> 张量)。
        #   mm_hash: 多模态输入内容的哈希键,用于区分不同缓存条目。
        # 未配置 EC 传输时直接返回(不保存)。
        if not has_ec_transfer():
            # 记录调试日志:提醒当前未配置 EC 传输。
            logger.debug("Not have ec transfer please check")
            return
        # 获取全局 EC 连接器实例。
        connector = get_ec_transfer()
        # 将编码器缓存按 mm_hash 保存到连接器,供配对实例消费。
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        # 根据是否配置 EC 传输,返回“获取连接器输出”的上下文管理器或空上下文,
        # 使调用方可以在 execute_model 中安全地消费编码器输出。
        # 参数:
        #   scheduler_output: 当前迭代的调度输出(含连接器元数据)。
        #   encoder_cache: 多模态编码器缓存(作为消费者时从连接器加载进来)。
        #   **kwargs: 传给 _get_ec_connector_output 的其它参数。
        return (
            # 配置了 EC 传输:返回真正从连接器获取输出的上下文管理器。
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            # 未配置:返回无操作上下文(nullcontext),不执行任何连接器逻辑。
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire EC connector lifecycle within execute_model
    # 说明:该上下文管理器必须在活跃的前向上下文内使用,
    # 它封装了 execute_model 中 EC 连接器的完整生命周期。
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        # 创建 EC 连接器输出容器(记录 finished_sending/finished_recving 状态)。
        output = ECConnectorOutput()

        # 获取全局 EC 连接器实例。
        ec_connector = get_ec_transfer()
        # 断言连接器实现了 ECConnectorBase 接口。
        assert isinstance(ec_connector, ECConnectorBase)
        # 断言调度输出中携带了连接器元数据(未携带则说明调用时机错误)。
        assert scheduler_output.ec_connector_metadata is not None
        # 绑定本迭代的连接器元数据(目标实例、缓存键等)。
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        # Load caches for consumer or both roles
        # 若当前实例是消费者(或同时承担消费者角色),则从连接器加载编码器缓存。
        if ec_connector.is_consumer:
            # 启动异步/同步的缓存加载流程。
            ec_connector.start_load_caches(encoder_cache, **kwargs)

        try:
            # 让出执行权,使 with 语句体内的前向逻辑在已加载缓存的情况下运行。
            yield output
        finally:
            # 无论如何退出,都要记录发送/接收完成状态:
            # 根据已完成请求集合查询连接器的发送与接收完成标志。
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )

            # 清理本次迭代绑定的连接器元数据,避免泄漏到下一迭代。
            ec_connector.clear_connector_metadata()