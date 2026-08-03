# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/ray_env_utils.py
# 本文件提供「从 driver 向 Ray worker 传播环境变量」的工具函数。
# 核心逻辑：返回 os.environ 中除 worker 专属变量与用户配置排除项之外的
# 全部环境变量，供 Ray actor 在 initialize_worker 时以 setdefault 语义应用。
# =============================================================================
import os
# 导入 os：读取当前进程（driver）的环境变量。

from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS
# 导入 Ray 官方「不随 runtime_env 继承」的环境变量集合。
# 这些变量携带节点/进程特定信息（如 Ray 内部的临时路径、端口等），
# 复制给 worker 会导致行为错误，因此必须排除。


def get_driver_env_vars(
    worker_specific_vars: set[str],
    # 调用方传入的「worker 专属」变量集合（如 WORKER_SPECIFIC_ENV_VARS），
    # 这些变量必须在 GPU 发现后按 worker 单独设置，不能从 driver 复制。
) -> dict[str, str]:
    # =========================================================================
    # 返回需要传播给 Ray worker 的 driver 环境变量字典。
    # =========================================================================
    """Return driver env vars to propagate to Ray workers.

    Returns everything from ``os.environ`` except ``worker_specific_vars``
    and user-configured exclusions (``RAY_NON_CARRY_OVER_ENV_VARS``).
    """
    # 文档字符串：返回需要传播给 Ray worker 的 driver 环境变量——
    # 即 os.environ 中除去 worker_specific_vars 与 RAY_NON_CARRY_OVER_ENV_VARS
    # 之外的全部变量。
    exclude_vars = worker_specific_vars | RAY_NON_CARRY_OVER_ENV_VARS
    # 合并两套排除集合：worker 专属变量 + Ray 官方不继承变量。

    return {key: value for key, value in os.environ.items() if key not in exclude_vars}
    # 字典推导式：遍历 driver 全部环境变量，保留不在排除集中的项。
    # 返回的字典随后由 RayExecutorV2.initialize_worker 用 setdefault 应用到
    # worker 进程——只补缺不覆盖，保证节点本地值优先。