# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CPU worker 子包:标识 vllm.v1.worker.cpu 为 Python 包。
# 该包包含 CPU 后端的缓冲工具(UvaBuffer)、CPU 模型执行器(CPUModelRunner)
# 以及将 CUDA 相关 API 替换为 CPU 占位实现的共享内存补丁模块(shm)。