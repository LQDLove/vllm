# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CPU Model Runner(V2 版本):继承 GPU 侧的 V2 GPUModelRunner,仅对接 CPU 编译方案,
# 提供模型编译预热(profile_run)以生成通用形状的推理图。

# 导入 vLLM 的日志初始化函数,用于输出日志。
from vllm.logger import init_logger

# 导入 GPU 侧的 V2 GPUModelRunner,CPU 实现复用它的大部分执行管线。
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

# 创建本模块的日志记录器,用于输出编译预热相关信息。
logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    # CPU 模型执行器:以 GPUModelRunner 为基类,CPU 后端复用其全部输入
    # 准备、注意力元数据构建与采样流程,只调整编译/预热行为。

    # TBD: Whether need to move this to Worker?
    # 注:是否将预热逻辑迁移到 Worker 层仍待定(TBD)。

    def warming_up_model(self) -> None:
        # 执行模型编译预热:为 CPU 后端编译生成推理计算图,避免运行期首次
        # 前向触发即时编译(all-at-once compilation)造成的大幅延迟。
        # 记录预热开始日志。
        logger.info("Warming up model for the compilation...")
        # 仅对通用(generic)输入形状生成编译图,以控制编译内存与时间。
        # 执行一次 dummy profile 前向,触发 torch.compile 编译该通用形状。
        self.profile_run()
        # 记录预热完成日志。
        logger.info("Warming up done.")