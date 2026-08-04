# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# XPU(Intel 加速卡)Model Runner:分别对应 V1 与 V2 的 GPUModelRunner,
# 通过 _torch_cuda_wrapper 上下文把 torch.cuda 相关 API 临时映射为 torch.xpu,
# 使 GPU 模型 runner 代码可在 XPU 设备上运行。

# 导入 contextmanager 装饰器,用于定义 _torch_cuda_wrapper 上下文管理器。
from contextlib import contextmanager
# 导入 partial 工具,用于将 torch.xpu 中的方法包装成与 cuda API 兼容的偏函数。
from functools import partial

# 导入 PyTorch,用于访问 torch.cuda / torch.xpu 设备接口。
import torch

# 导入 VllmConfig,模型 runner 构造时接收完整的 vLLM 配置。
from vllm.config import VllmConfig
# 导入 supports_xpu_graph,用于判断当前 XPU 环境是否支持 CUDA Graph 等价特性。
from vllm.utils.torch_utils import supports_xpu_graph
# 导入 V2 GPUModelRunner,并命名为 GPUModelRunnerV2 以便引用。
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
# 导入 V1 GPUModelRunner(顶层实现)。
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""
    # XPU 平台模型 runner(V1 版本):以 GPUModelRunner 为基类,
    # 仅在初始化时通过 _torch_cuda_wrapper 把 cuda API 映射到 XPU。

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # 初始化 XPU V1 模型 runner。
        # 参数:
        #   vllm_config: 完整的 vLLM 配置。
        #   device: 目标 XPU 设备。
        # 在 cuda->xpu 映射上下文中调用父类初始化,使 GPU 代码运行于 XPU。
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        # FIXME: To be verified.
        # 标记:级联注意力(cascade attention)在 XPU 上暂不启用,待验证后移除。
        self.cascade_attn_enabled = False


class XPUModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""
    # XPU 平台模型 runner(V2 版本):以 V2 GPUModelRunner 为基类,
    # 适配新 Model Runner V2 执行管线。

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # 初始化 XPU V2 模型 runner。
        # 参数:
        #   vllm_config: 完整的 vLLM 配置。
        #   device: 目标 XPU 设备。
        # 在 cuda->xpu 映射上下文中调用父类初始化,使 V2 代码运行于 XPU。
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    # 把 torch.cuda 的相关 API 临时替换为 torch.xpu 实现,供模型 runner
    # 初始化期间使用。使用 contextmanager 以便 with 语句进出时生效。
    # Replace cuda APIs with xpu APIs. Each callable gets its own functools.partial
    # so it is not the same object as torch.xpu.* (Torch Dynamo _get_handlers()
    # asserts on duplicate registration when cuda aliases xpu directly).
    # 说明:每个可调用对象都被包装成独立的 functools.partial,避免与 torch.xpu.*
    # 引用同一个对象(否则 Torch Dynamo 的 _get_handlers() 会在重复注册时报错)。
    # 将 torch.cuda.Stream 指向 torch.xpu.Stream。
    torch.cuda.Stream = torch.xpu.Stream
    # 将 torch.cuda.default_stream 包装为 torch.xpu.current_stream。
    torch.cuda.default_stream = partial(torch.xpu.current_stream)
    # 将 torch.cuda.current_stream 包装为 torch.xpu.current_stream。
    torch.cuda.current_stream = partial(torch.xpu.current_stream)
    # 将 torch.cuda.stream 包装为 torch.xpu.stream(上下文切换流)。
    torch.cuda.stream = partial(torch.xpu.stream)
    # 将 torch.cuda.set_stream 包装为 torch.xpu.set_stream。
    torch.cuda.set_stream = partial(torch.xpu.set_stream)

    # torch.xpu.Event does not accept the ``blocking`` kwarg that
    # torch.cuda.Event supports, so drop it here.
    # 说明:torch.xpu.Event 不接受 torch.cuda.Event 支持的 blocking 关键字参数,
    # 因此在此包装函数中丢弃该参数。
    def _xpu_event(*args, blocking=None, **kwargs):
        # XPU 事件包装:忽略 blocking 参数,构造 torch.xpu.Event。
        return torch.xpu.Event(*args, **kwargs)

    # 将 torch.cuda.Event 替换为 XPU 事件包装函数。
    torch.cuda.Event = _xpu_event
    # 若当前 XPU 环境支持 CUDA Graph 等价特性(设备能力足够):
    if supports_xpu_graph():
        # 将 torch.cuda.graph 包装为 torch.xpu.graph(图捕获上下文)。
        torch.cuda.graph = partial(torch.xpu.graph)
        # 将 torch.cuda.CUDAGraph 指向 XPU 的图类。
        torch.cuda.CUDAGraph = torch.xpu.XPUGraph
        # 将 torch.cuda.graph_pool_handle 包装为 torch.xpu.graph_pool_handle。
        torch.cuda.graph_pool_handle = partial(torch.xpu.graph_pool_handle)
    # 让出执行权,使 with 语句体内的代码在补丁生效后运行。
    yield