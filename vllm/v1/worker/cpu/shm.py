# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CPU 共享内存兼容补丁模块:在纯 CPU 环境中,将 torch.cuda / torch.accelerator
# 以及 vLLM 的 GPU 缓冲/UVA 相关 API 替换为 CPU 占位实现或无操作(no-op),
# 使依赖 CUDA 的模型 runner 代码可在 CPU 后端正常 import 与运行。

# 跳过本文件的 import 排序检查(isort 全局跳过指令)。
# isort: skip_file
# 禁止本文件的 ruff E402(模块层级 import 位置)告警:允许在代码中段 import。
# ruff: noqa: E402
# 禁止 mypy 的 misc / assignment 类错误,因为这里大量动态替换 torch 属性。
# mypy: disable-error-code="misc, assignment"

# 导入 Any 类型,用于标注占位函数的任意参数。
from typing import Any

# 导入 numpy,用于在 async_tensor_h2d 中把数组转换为张量。
import numpy as np

# 补丁 torch API:先导入 torch 模块,随后对其 Event/Stream 等属性进行替换。
# Patch torch APIs
import torch


def noop(*args: Any, **kwargs: Any) -> None:
    # 通用 no-op 函数:忽略所有位置参数与关键字参数,不做任何事。
    # 用于替换 CPU 上无意义或会报错的 CUDA 同步/缓存清理等调用。
    pass


# Distinct no-op so empty_cache does not alias synchronize: Dynamo's
# handle_synchronize is keyed on that object and asserts on CPU-only hosts.
def empty_cache_noop(*args: Any, **kwargs: Any) -> None:
    pass


def fake_pin_memory(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    # 伪造的 pin_memory 方法:CPU 张量本身即可被零拷贝访问,无需锁页,
    # 因此直接返回张量自身(原样返回,不做任何拷贝)。
    return self


class _EventPlaceholder:
    # CUDA 事件的 CPU 占位类:在 CPU 上没有真正的流事件语义,
    # 提供记录(record)与同步(synchronize)两个空方法即可让调用方正常执行。

    def __init__(self, *args, **kwargs) -> None:
        # 占位构造:忽略所有参数,仅挂载两个 no-op 方法。
        # 将 record 方法替换为 no-op:记录事件(在 CPU 上无事可做)。
        self.record = noop
        # 将 synchronize 方法替换为 no-op:等待事件(在 CPU 上无事可做)。
        self.synchronize = noop


class _StreamPlaceholder:
    # CUDA 流的 CPU 占位类:代替 torch.cuda.Stream,使依赖流的代码在
    # CPU 后端可以无障碍运行。

    def __init__(self, *args, **kwargs) -> None:
        # 占位构造:忽略所有参数,挂载 wait_stream 空方法,并把设备设为 CPU。
        # 将 wait_stream 方法替换为 no-op:等待另一流(CPU 上无事可做)。
        self.wait_stream = noop
        # 将设备固定为 CPU 设备,使读取 .device 的代码得到 torch.device("cpu")。
        self.device = torch.device("cpu")

    def __enter__(self, *args, **kwargs):
        # 上下文管理器进入:返回自身,支持 with 语句。
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 上下文管理器退出:无事可做,直接通过(不吞噬任何异常)。
        pass


# 导入 CPU 资源工具,用于在 get_memory_info 中读取 NUMA 节点的内存信息。
from vllm.utils.cpu_resource_utils import get_memory_node_info


def get_memory_info(*args: Any, **kwargs: Any) -> tuple[int, int]:
    # 获取当前 CPU 内存信息,替代 torch.accelerator.get_memory_info。
    # 返回值: (available_memory, total_memory),单位字节。
    # 读取当前 NUMA 节点的内存信息。
    meminfo = get_memory_node_info()
    # 返回(可用内存, 总内存)。
    return meminfo.available_memory, meminfo.total_memory


# 补丁 torch.cuda / torch.accelerator 相关 API 为 CPU 占位实现:
# 将 torch.Event 替换为 CPU 占位类。
torch.Event = _EventPlaceholder
# 将 torch.cuda.Event 同样替换为 CPU 占位类。
torch.cuda.Event = _EventPlaceholder
# 将 torch.cuda.Stream 替换为 CPU 占位类。
torch.cuda.Stream = _StreamPlaceholder
# 将 torch.cuda.set_stream 替换为 no-op(CPU 上切换流无意义)。
torch.cuda.set_stream = noop
# 将 torch.cuda.current_stream 替换为总是返回一个新的 CPU 占位流。
torch.cuda.current_stream = lambda *args, **kwargs: _StreamPlaceholder()
# 将 torch.accelerator.synchronize 替换为 no-op(CPU 天然同步)。
torch.accelerator.synchronize = noop
# 将 torch.accelerator.empty_cache 替换为 no-op(CPU 无需清缓存)。
torch.accelerator.empty_cache = noop
# 将 torch.Tensor.pin_memory 替换为 fake_pin_memory(CPU 免锁页)。
torch.Tensor.pin_memory = fake_pin_memory
# 将 torch.accelerator.get_memory_info 替换为读取 NUMA 节点内存的真实实现。
torch.accelerator.get_memory_info = get_memory_info

# 补丁 vLLM torch 工具:导入 vllm.utils.torch_utils 以便覆盖其异步拷贝函数。
# Patch vLLM torch utils
import vllm.utils.torch_utils as torch_utils


def async_tensor_h2d(
    # CPU 版的主机到设备异步拷贝:在 CPU 后端无异步语义(不会返回 Stream/Event),
    # 直接同步地(to)完成数据搬运并返回 CPU 张量。
    data: list | np.ndarray | torch.Tensor,
    # 输入数据:可以是列表、numpy 数组或 torch.Tensor。
    device: str | torch.device,
    # 目标设备:在 CPU 实现中被忽略,始终返回 CPU 张量。
    dtype: torch.dtype | None = None,
    # 目标数据类型:为空时沿用输入本身的 dtype。
) -> torch.Tensor:
    # 若输入是 numpy 数组,先转换为 torch 张量(共享底层内存,高效)。
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    # 若输入已是 torch 张量,则按目标 dtype 转换到 CPU(默认留在原 dtype)。
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    # 其余情况(纯 list)以“列表 -> CPU 张量”方式创建,并应用目标 dtype。
    return torch.tensor(data, dtype=dtype, device="cpu")


# 用 CPU 实现覆盖 vllm 工具模块中的异步拷贝函数。
torch_utils.async_tensor_h2d = async_tensor_h2d

# 补丁 model runner API:导入 GPU 与 CPU 的缓冲工具模块,准备替换 UvaBuffer。
# Patch model runner APIs
import vllm.v1.worker.gpu.buffer_utils as gpu_buffer_utils
import vllm.v1.worker.cpu.buffer_utils as cpu_buffer_utils

# 将 GPU 侧的 UvaBuffer 替换为 CPU 侧基于 CPU 张量的 UVA 实现,
# 使 GPU 模型 runner 在 CPU 后端也能完成零拷贝缓冲的初始化。
gpu_buffer_utils.UvaBuffer = cpu_buffer_utils.UvaBuffer