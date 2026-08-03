# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 本模块是 vllm.v1.executor 包的初始化文件，负责对外导出执行器相关类。
from .abstract import Executor
# 从 abstract 模块导入 Executor 抽象基类，它是所有执行器的统一接口定义。
from .uniproc_executor import UniProcExecutor
# 从 uniproc_executor 模块导入 UniProcExecutor，即单进程执行器实现。

__all__ = ["Executor", "UniProcExecutor"]
# 定义 __all__ 列表，声明本包对外公开的符号：仅 Executor 与 UniProcExecutor。
# 注意：MultiprocExecutor / Ray 系列执行器不在此导出，因为它们是按需在
# abstract.get_class() 工厂方法中动态导入的，避免强制依赖 multiprocessing 或 ray。