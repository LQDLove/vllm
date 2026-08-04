# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CPU 侧缓冲工具:提供基于 UVA(Unified Virtual Addressing,统一虚拟寻址)的
# CPU 张量缓冲,使 CPU 张量可直接被 GPU 以零拷贝方式访问。

# 导入抽象序列类型 Sequence,用于表示尺寸既可传整数也可传序列。
from collections.abc import Sequence

# 导入 PyTorch,用于创建张量。
import torch

# 导入平台工具函数 is_uva_available,用于检测当前环境是否支持 UVA。
from vllm.utils.platform_utils import is_uva_available


class UvaBuffer:
    # UVA 缓冲类:在 CPU 上分配一块零拷贝张量,供 GPU 直接读取。
    # 适用于 CPU 后端需要与 GPU 共享数据、又不想执行设备间显式拷贝的场景。

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        # 初始化 UVA 缓冲。
        # 参数:
        #   size: 缓冲的形状,可接收单个整数(一维)或形状序列。
        #   dtype: 张量的数据类型(如 torch.float32)。
        # 检查当前环境是否支持 UVA;不支持则说明无法建立零拷贝缓冲。
        if not is_uva_available():
            # 抛出运行时错误,提示 UVA 不可用。
            raise RuntimeError("UVA is not available")
        # 在 CPU 上分配指定形状与类型的全零张量作为底层存储。
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        # 暴露该 CPU 张量的 numpy 视图,方便与 numpy 代码互操作。
        self.np = self.cpu.numpy()
        # UVA 视图与 CPU 张量共享同一块内存(零拷贝)。
        self.uva = self.cpu