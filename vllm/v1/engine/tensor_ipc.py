# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 文件头部：开源许可证声明（Apache 2.0 版权）

"""Tensor IPC transport via torch.multiprocessing.Queue.
# 模块文档：通过 torch.multiprocessing.Queue 实现跨进程张量 IPC 传输

This module contains the queue-based transport logic for sharing tensors
between processes (e.g., API server -> engine core). The msgpack layer
emits/consumes lightweight :class:`TensorIpcData` values, while transport
state such as request association, handle generation, queue routing, buffering,
and cleanup lives here.
"""  # 说明 msgpack 层只传输轻量句柄(TensorIpcData)，真实的张量数据经由本模块的队列传输

import dataclasses  # dataclasses 模块：用于定义数据类
import uuid  # uuid 模块：生成唯一 ID（用于 sender_id 标识发送者）
from collections import defaultdict
# defaultdict：带默认值的字典，访问不存在的键时自动创建默认值（避免 KeyError）
from dataclasses import field  # field：数据类字段配置工具（如 default_factory）
from multiprocessing.queues import Queue as MPQueue
# 导入 multiprocessing 的 Queue 并别名为 MPQueue：多进程安全队列，用于跨进程传张量
from typing import Any  # Any：通用类型标注

import torch  # PyTorch：张量操作（share_memory_ 等）

from vllm.logger import init_logger  # 初始化 vLLM 的日志记录器
from vllm.v1.serial_utils import OOBTensorConsumer
# OOBTensorConsumer：带外(out-of-band)张量消费者基类，msgpack 层通过它把张量"旁路"出去

logger = init_logger(__name__)  # 模块级日志记录器，__name__ = "vllm.v1.engine.tensor_ipc"

TensorIpcQueue = MPQueue  # 类型别名：语义化命名 multiprocessing.Queue


@dataclasses.dataclass  # 数据类装饰器：自动生成 __init__、__repr__ 等
class TensorIpcData:
    """
    Data sent via torch.multiprocessing.Queue for zero-copy IPC.
    # 通过 torch.multiprocessing.Queue 发送的数据载体，用于零拷贝 IPC

    Contains the tensor_id and the actual tensor. The tensor is
    shared in memory (GPU or CPU) for efficient inter-process communication.
    # 含张量 ID 和真实张量；张量位于共享内存（GPU 或 CPU），实现高效跨进程通信
    """

    sender_id: str  # 发送者 ID（8位hex字符串，唯一标识发送进程）
    message_id: int  # 消息编号（一次请求处理 = 一条消息，可含多个张量）
    tensor_id: int  # 张量序号（同一消息内第几个张量）
    tensor: torch.Tensor  # 实际传输的 torch.Tensor（位于共享内存）


class TensorIpcSender(OOBTensorConsumer):
    """Send-side logic for tensor IPC via torch.multiprocessing.Queue.
    # 张量 IPC 发送端逻辑

    Uses a single queue targeting rank 0 (the only rank that consumes
    multimodal tensors during TP>1 / PP>1. Note: DP>1 not supported).
    # 使用单队列，目标为 rank 0（TP>1 / PP>1 时唯一消费多模态张量的 rank；DP>1 不支持）
    """

    def __init__(self, queue: TensorIpcQueue):  # 构造函数：接收张量传输队列
        self.queue = queue  # 保存队列引用
        self._tensor_id_counter = 0  # 张量 ID 计数器，从 0 起每发一个 +1
        self._message_counter = 0  # 消息计数器，从 0 起每条新消息 +1
        self._sender_id = uuid.uuid4().hex[:8]
        # 生成 8 位 hex 唯一 ID 标识本发送者；接收端用它区分不同发送方

    def set_target_engine(self, target_engine: int) -> None:  # 设置目标引擎编号（多引擎路由）
        if target_engine != 0:  # 当前实现只支持目标引擎 0（单队列）
            raise IndexError(  # 抛异常拒绝其他目标
                "TensorIpcSender only supports a single queue; "
                f"got target engine {target_engine}"
            )  # 错误消息：说明仅支持单队列，并报告收到的引擎编号

    def new_message(self) -> None:  # 开始一条新消息（每条 EngineCoreRequest 序列化前调用）
        self._message_counter += 1  # 消息计数 +1
        self._tensor_id_counter = 0  # 张量 ID 重置为 0（新消息从 0 重新编号）

    def __call__(self, tensor: torch.Tensor) -> dict[str, Any] | None:
        # 使实例可调用；msgpack 序列化器遇到张量时调用
        # 参数：要传输的张量；返回：张量句柄(metadata)，失败返回 None
        """Send tensor via queue, return its handle. Returns None if failed."""
        # 通过队列发送张量，返回其句柄；失败返回 None

        try:  # 异常处理：任何失败走 except 分支降级回退
            # Move tensor to shared memory for IPC
            # This is required for proper inter-process communication
            # 将张量移到共享内存（IPC 必需），跨进程直接访问，实现零拷贝
            if not tensor.is_shared():  # 检查张量是否已在共享内存中
                tensor = tensor.share_memory_()
            # 不在则原地转为共享内存（关键零拷贝前提）

            metadata = {  # 创建张量句柄字典（只含 ID 信息，不含数据本体）
                "sender_id": self._sender_id,  # 发送者 ID
                "message_id": self._message_counter,  # 当前消息编号
                "tensor_id": self._tensor_id_counter,  # 当前张量编号
            }

            self._tensor_id_counter += 1  # 张量编号自增，为下一个张量做准备

            ipc_data = TensorIpcData(**metadata, tensor=tensor)  # type: ignore[arg-type]
            # 用 metadata + tensor 构造 TensorIpcData（解包元数据字段）

            # Use a timeout to avoid blocking indefinitely
            # 使用超时避免无限阻塞
            self.queue.put(ipc_data, timeout=10.0)
            # 将张量数据放入多进程队列（共享内存引用，零拷贝），10 秒超时保护

            logger.debug(  # 记录调试日志
                "Sent tensor %s for (shape=%s, device=%s) "
                "via IPC queue (shared memory)",  # 日志格式：句柄、形状、设备
                metadata,  # 张量元数据
                tensor.shape,  # 张量形状
                tensor.device,  # 张量设备
            )

            return metadata  # 成功：返回句柄，msgpack 层把它放入 ZMQ 消息

        except Exception as e:  # 捕获所有异常（队列满、序列化失败等）
            logger.warning(  # 记录警告日志
                "Failed to send tensor via IPC queue: %s. "
                "Falling back to standard serialization.",  # 提示将回退标准序列化
                e,  # 异常对象
            )

            return None  # 失败：返回 None，msgpack 层回退到普通序列化（张量走 ZMQ）


@dataclasses.dataclass  # 数据类装饰器
class _Sender:
    # 私有辅助类：管理单个发送者的缓冲状态
    current_message_id: int = -1
    # 当前已处理的最大消息 ID，初始 -1（用于清理过期消息）
    tensors: dict[int, dict[int, torch.Tensor]] = field(default_factory=dict)
    # 张量缓冲：外层键=message_id，内层键=tensor_id，值为张量
    # field(default_factory=dict) 确保每个实例独立字典（避免共享可变默认值）


class TensorIpcReceiver:
    """Receive-side logic for tensor IPC via torch.multiprocessing.Queue.
    # 张量 IPC 接收端逻辑

    Wraps the queue receive logic previously embedded in MsgpackDecoder.
    # 封装了原先嵌入在 MsgpackDecoder 中的队列接收逻辑
    """

    def __init__(self, queue: TensorIpcQueue):  # 构造函数：接收张量队列
        self.queue = queue  # 保存队列引用
        self._tensor_buffers = defaultdict[str, _Sender](_Sender)
        # 按 sender_id 索引的缓冲字典；defaultdict 自动为新 sender 创建 _Sender

    def __call__(
        # 使实例可调用；msgpack 解码器遇到张量句柄时调用
        self, dtype: str, shape: tuple[int, ...], meta: dict[str, Any]
        # 参数：期望的 dtype、形状、以及 msgpack 中携带的句柄(meta)
    ) -> torch.Tensor:  # 返回：真正的 torch.Tensor（从共享内存取回）

        """Retrieve a tensor from torch.multiprocessing.Queue.
        # 从 torch.multiprocessing.Queue 取回张量

        Uses a drain-and-buffer pattern: drains all available tensors from
        the queue, buffering them, until the requested tensor is found.
        Works for CUDA and CPU.
        # 使用"排空+缓冲"模式：持续取队列中所有张量并缓存，直到找到目标；支持 CUDA/CPU
        """

        # Create lookup key from handle
        # 从句柄创建查找键
        sender_id: str = meta["sender_id"]  # 提取发送者 ID
        message_id: int = meta["message_id"]  # 提取消息编号
        tensor_id: int = meta["tensor_id"]  # 提取张量编号

        # Drain all available tensors. We save them regardless if this is
        # the one we're waiting for as they may arrive out of order from
        # multiple producers.
        # 排空所有可用张量：无论是否目标都保存，因为多个生产者的张量可能乱序到达
        while True:  # 无限循环直到找到目标张量；超时由 queue.get 的 timeout 抛异常退出
            sender = self._tensor_buffers.get(sender_id)
            # 获取该发送者的缓冲对象（可能为 None = 还没收到该发送者数据）

            if sender is not None:  # 如果该发送者有缓冲数据
                tensors = sender.tensors
                # 取张量缓冲（message_id -> {tensor_id: tensor}）
                tensor = tensors.get(message_id, {}).pop(tensor_id, None)
                # 从缓冲中取出目标张量（pop 取出并删除；找不到返回 None）

                if tensor is not None:  # 如果找到了目标张量
                    if sender.current_message_id != message_id:
                    # 如果这条消息比当前已处理的消息新（新消息到达）
                        while tensors and (mid := next(iter(tensors))) < message_id:
                        # 清理所有早于当前消息 ID 的过期消息（海象运算符 := 赋值并比较）
                            if sender.tensors.pop(mid):  # 移除过期消息条目
                                logger.warning(  # 记录警告日志
                                    "Discarding %d stale tensors from sender %s",
                                    # 日志格式：丢弃的旧张量数量
                                    sender_id,  # 传入 sender_id
                                )
                        sender.current_message_id = message_id
                        # 更新当前消息 ID 为这个新消息

                    logger.debug(  # 记录调试日志
                        "Received tensor %s from sender %s for (shape=%s, device=%s) "
                        "via IPC queue (shared memory)",  # 日志格式：张量元数据和属性
                        (message_id, tensor_id),  # (消息ID, 张量ID)
                        sender_id,  # 发送者 ID
                        tensor.shape,  # 张量形状
                        tensor.device,  # 张量设备
                    )

                    return tensor  # 找到目标，返回张量

            ipc_data: TensorIpcData = self.queue.get(timeout=10.0)
            # 缓冲中没有目标：从队列取新张量（10 秒超时保护）

            # Store tensor
            # 存储张量到缓冲
            sender = self._tensor_buffers[ipc_data.sender_id]
            # 获取（或自动创建）该发送者的缓冲对象

            if sender.current_message_id > ipc_data.message_id:
            # 如果这个张量属于更早的过期消息
                logger.warning(  # 记录警告日志
                    "Ignoring stale tensor from sender %s", ipc_data.sender_id
                )  # 传入发送者 ID

                continue  # 丢弃过期张量，继续循环等待目标

            sender.tensors.setdefault(ipc_data.message_id, {})[ipc_data.tensor_id] = (
                ipc_data.tensor
            )
            # 将取到的张量存入缓冲：
            # setdefault 获取（或创建）该消息 ID 的字典，
            # 然后按 tensor_id 存储张量