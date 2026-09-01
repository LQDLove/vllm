# SPDX-License-Identifier: Apache-2.0  # Apache 2.0 开源许可证标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权归属 vLLM 项目贡献者
"""A GPU worker class.
GPU worker 类：vLLM v1 中单个 GPU 工作进程的核心实现，
负责设备初始化、模型加载、内存剖析、KV cache 分配、
CUDA 图捕获/预热、模型执行与采样等全生命周期管理。
"""

import gc  # 垃圾回收：用于冻结/解冻 GC 堆以稳定推理延迟
import os  # 操作系统接口：读取/清理环境变量（如 NCCL、PyTorch 分配器配置）
import time  # 时间工具：sleep 模式内存释放轮询、单调时钟计时
from collections.abc import Callable  # 可调用类型注解：通信后处理回调
from contextlib import AbstractContextManager, contextmanager, nullcontext  # 上下文管理器工具：内存池上下文、空上下文
from datetime import timedelta  # 时间跨度：分布式初始化超时配置
from types import NoneType  # None 类型：execute_model 返回类型 isinstance 判断
from typing import TYPE_CHECKING, Any  # 类型检查开关与 Any 类型注解

import numpy as np  # NumPy：调度 token 数组、块表等数组操作
import regex as re  # regex 模块（增强版 re）：解析 PyTorch 分配器配置字符串
import torch  # PyTorch 核心库：张量、设备管理、推理模式
import torch.nn as nn  # 神经网络模块：get_model 等方法的返回类型

import vllm.envs as envs  # vLLM 环境变量：float32 精度、CUDA 图内存剖析开关等
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config  # 全局配置：CUDA 图模式、vLLM 配置、线程局部配置切换
from vllm.config.compilation import CompilationMode  # 编译模式枚举：VLLM_COMPILE / NONE 等
from vllm.device_allocator import get_mem_allocator_instance  # 内存分配器实例：CuMem 内存池（sleep 模式/权重/KV cache 隔离）
from vllm.distributed import (  # 分布式初始化工具
    ensure_model_parallel_initialized,  # 确保张量/流水线/上下文并行组初始化完成
    init_distributed_environment,  # 初始化 torch.distributed 进程组环境
    set_custom_all_reduce,  # 启用/禁用自定义 allreduce kernel
)
from vllm.distributed.ec_transfer import (  # Encoder Cache 传输连接器（EPD 分离式部署）
    ensure_ec_transfer_initialized,  # 初始化 EC 传输
    ensure_ec_transfer_shutdown,  # 关闭 EC 传输
)
from vllm.distributed.eplb.eplb_utils import override_envs_for_eplb  # 专家并行负载均衡（EPLB）环境变量覆盖
from vllm.distributed.kv_transfer import (  # KV cache 传输连接器（分离式预填充/解码等）
    ensure_kv_transfer_initialized,  # 初始化 KV 传输（需要 kv_cache_config）
    ensure_kv_transfer_shutdown,  # 关闭 KV 传输
    get_kv_transfer_group,  # 获取 KV 传输进程组
    has_kv_transfer_group,  # 判断是否已创建 KV 传输组
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # KV 连接器基类
    KVConnectorHandshakeMetadata,  # worker 间握手元数据类型
)
from vllm.distributed.parallel_state import (  # 并行状态管理
    Handle,  # 非阻塞通信句柄（异步 send/recv 的等待对象）
    checkpoint_prepare_distributed_state,  # 准备分布式状态检查点
    checkpoint_restore_distributed_state,  # 恢复分布式状态检查点
    get_pp_group,  # 流水线并行（PP）进程组
    get_tp_group,  # 张量并行（TP）进程组
)
from vllm.distributed.weight_transfer import (  # 权重传输引擎（RLHF/在线训练更新权重）
    WeightTransferEngine,  # 权重传输引擎基类
    WeightTransferEngineFactory,  # 按配置创建具体引擎的工厂
)
from vllm.logger import init_logger  # vLLM 日志工具：创建模块级 logger
from vllm.lora.request import LoRARequest  # LoRA 适配器请求：动态加载/卸载 LoRA
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup  # kernel 级预热：在 CUDA 图捕获前预热/调优 kernel
from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory  # 多模态 IPC 共享显存预留（API 进程间零拷贝传递）
from vllm.platforms import current_platform  # 当前硬件平台抽象：CUDA/ROCm/XPU/CPU 判断与平台定制
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper  # 性能剖析器包装：torch profiler 与 CUDA profiler
from vllm.sequence import IntermediateTensors  # PP 中间层张量容器：流水线阶段间传递的隐藏状态
from vllm.tasks import SupportedTask  # 支持的任务类型枚举（生成/池化等）
from vllm.tracing import instrument  # 追踪装饰器：为方法添加 span 追踪
from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback  # GC 工具：冻结堆避免推理期 GC 扫描大对象
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check, with_gpu_sync_check  # GPU 同步检查调试：VLLM_GPU_SYNC_CHECK 门控
from vllm.utils.mem_constants import GiB_bytes  # 常量：1 GiB 的字节数
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling  # 内存工具：内存快照、GiB 格式化、剖析上下文
from vllm.utils.torch_utils import set_random_seed, set_torch_threads_for_runtime  # torch 工具：设置随机种子、运行期线程数
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput  # 调度器输出：每步调度计划与文法（结构化输出）结果
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec  # KV cache 接口：缓存配置与各层规格
from vllm.v1.outputs import (  # 模型输出类型
    AsyncModelRunnerOutput,  # 异步调度模式下的输出（延迟采样）
    DraftTokenIds,  # 投机解码草稿 token id 输出
    ModelRunnerOutput,  # 常规模型执行输出（采样 token id 等）
)
from vllm.v1.utils import compute_iteration_details, report_usage_stats  # 工具：迭代明细统计、匿名使用统计上报
from vllm.v1.worker.sentinel.gpu_worker_sentinel import WorkerSentinel  # 容错哨兵：处理故障容忍（FT）控制命令
from vllm.v1.worker.startup_plan import (  # 启动计划工具
    maybe_apply_startup_plan,  # 应用预保存的启动计划（KV cache 内存建议等）
    maybe_save_startup_plan,  # 保存启动计划供下次启动复用
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp  # 判断序列并行下 residual 是否分散存储
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase  # worker 基类与编译耗时记录
from vllm.v1.worker.workspace import init_workspace_manager  # 工作空间管理器：为临时张量提供统一 workspace

from ...model_executor.model_loader import TensorizerLoader  # Tensorizer 模型加载/保存器（相对导入上层包）
from .gpu.warmup import warmup_kernels  # V2 model runner 的 kernel 预热入口
from .utils import request_memory  # 根据 gpu_memory_utilization 计算目标请求显存

logger = init_logger(__name__)  # 创建本模块的日志记录器

if TYPE_CHECKING:  # 仅类型检查时导入，避免运行时循环依赖
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackend  # sleep 模式后端类型
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig  # Tensorizer 配置类型
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # V1 GPU model runner 类型


class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization
    带惰性通信同步的中间张量容器：PP 非阻塞接收的包装，
    在真正访问 tensors 之前才等待通信完成。
    """

    def __init__(  # 构造函数
        self,
        tensors: dict[str, torch.Tensor],  # 中间层张量字典（如 "hidden_states"）
        comm_handles: list[Handle] | None = None,  # 未完成的通信句柄列表（异步 recv）
        comm_postprocess: list[Callable[[], None]] | None = None,  # 通信后处理回调列表
    ) -> None:  # 无返回值
        super().__init__(tensors)  # 调用基类初始化张量容器
        self._comm_handles = comm_handles  # 保存通信句柄
        self._comm_postprocess = comm_postprocess  # 保存后处理回调
        self._comm_waited = False  # 标记是否已等待通信完成

    def wait_for_comm(self) -> None:  # 等待全部通信完成（幂等）
        if self._comm_waited:  # 已经等待过
            return  # 直接返回，避免重复等待
        if self._comm_handles:  # 存在未完成的通信句柄
            for handle in self._comm_handles:  # 逐个等待
                handle.wait()  # 阻塞直到该通信完成
        if self._comm_postprocess:  # 存在后处理回调
            for fn in self._comm_postprocess:  # 逐个执行
                fn()  # 如 allgather 后的切分/重排
        self._comm_waited = True  # 标记已完成

    def __getattribute__(self, name: str):  # 属性访问拦截：保证使用前通信已就绪
        # ensure `.tensors` is ready before use
        # 确保访问 `.tensors` 前通信数据已就绪
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):  # 首次访问 tensors 且未等待
            object.__getattribute__(self, "wait_for_comm")()  # 先等待通信完成
        return object.__getattribute__(self, name)  # 返回正常属性访问结果


class Worker(WorkerBase):  # GPU worker：继承 worker 基类
    def __init__(  # 构造函数
        self,
        vllm_config: VllmConfig,  # 全局 vLLM 配置
        local_rank: int,  # 节点内本地 rank（决定可见 GPU）
        rank: int,  # 全局 rank（分布式进程组中的编号）
        distributed_init_method: str,  # 分布式初始化方法（如 "env://"、tcp 地址）
        is_driver_worker: bool = False,  # 是否为驱动 worker（rank 0 负责与引擎核心交互）
    ):
        super().__init__(  # 调用基类构造
            vllm_config=vllm_config,  # 透传全局配置
            local_rank=local_rank,  # 本地 rank
            rank=rank,  # 全局 rank
            distributed_init_method=distributed_init_method,  # 初始化方法
            is_driver_worker=is_driver_worker,  # 是否驱动 worker
        )

        # configure float32 matmul precision according to vLLM env.
        # 按 vLLM 环境变量配置 float32 矩阵乘法精度（"highest"/"high"/"medium"）
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION  # 读取精度配置
        torch.set_float32_matmul_precision(precision)  # 应用到 PyTorch 全局设置

        from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor  # 弹性专家并行（EP）扩缩容执行器

        self.elastic_ep_executor = ElasticEPScalingExecutor(self)  # 创建 EP 弹性扩缩容执行器（传入自身引用）
        self.worker_sentinel: WorkerSentinel | None = None  # 容错哨兵（默认无）
        if self.parallel_config.enable_fault_tolerance:  # 启用故障容忍时
            self.worker_sentinel = WorkerSentinel(worker=self)  # 创建哨兵以处理 FT 命令
        # Buffers saved before sleep
        # level 2 sleep 前保存的缓冲区（非权重持久张量，如 RoPE 缓存）
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}  # 主模型保存的缓冲区（名称 -> CPU 副本）
        self._sleep_saved_draft_buffers: dict[str, torch.Tensor] = {}  # 草稿模型的缓冲区副本

        # Weight transfer engine is created in `load_model` once the model
        # is available, since the engine needs a reference to the model.
        # 权重传输引擎在 load_model 中创建（引擎需要模型引用）
        self.weight_transfer_engine: WeightTransferEngine | None = None  # 权重传输引擎（延迟创建）
        self._weight_update_active = False  # 是否有进行中的权重更新会话
        self._weight_update_is_draft = False  # 当前会话是否针对草稿模型

        # Torch/CUDA profiler. Enabled and configured through profiler_config.
        # Profiler wrapper is created lazily in profile() when start is called,
        # so we have all the information needed for proper trace naming.
        # torch/CUDA 剖析器，经 profiler_config 启用与配置；
        # 包装器在 profile() 开始时惰性创建，以获得完整的 trace 命名信息
        self.profiler: Any | None = None  # 剖析器包装实例（延迟创建）
        self.profiler_config = vllm_config.profiler_config  # 保存剖析器配置

        # Only validate profiler config is valid, don't instantiate yet
        # 仅校验剖析器配置合法性，暂不实例化
        if self.profiler_config.profiler not in ("torch", "cuda", None):  # 非法类型
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")  # 报错

        self.use_v2_model_runner = vllm_config.use_v2_model_runner  # 是否使用 V2 版 model runner
        # pending non-blocking PP send work from the previous iteration
        # 上一次迭代遗留的 PP 非阻塞发送任务
        self._pp_send_work: list[Handle] = []  # 待等待的通信句柄列表

        # Resolved lazily on first sleep/wake; persists worker-process state.
        # sleep 后端在首次 sleep/wake 时惰性解析；随 worker 进程存活
        self._sleep_mode_backend: SleepModeBackend | None = None  # sleep 模式后端（延迟创建）

    def _get_sleep_mode_backend(self) -> "SleepModeBackend":  # 获取（并按需创建）sleep 后端
        if self._sleep_mode_backend is None:  # 尚未创建
            from vllm.device_allocator.sleep_mode_backend import (  # 延迟导入避免循环依赖
                SleepModeBackendFactory,  # 后端工厂
            )

            self._sleep_mode_backend = SleepModeBackendFactory.create_backend(  # 按模型配置创建具体后端
                self.vllm_config.model_config  # 模型配置决定后端实现
            )
        return self._sleep_mode_backend  # 返回后端实例

    def sleep(self, level: int = 1) -> None:  # 进入 sleep 模式释放显存
        torch.accelerator.synchronize()  # 等待所有 GPU 任务完成，确保测量准确
        free_bytes_before_sleep = torch.accelerator.get_memory_info()[0]  # 记录 sleep 前空闲显存

        # Save the buffers before level 2 sleep
        # level 2 sleep 会销毁显存数据，先保存模型缓冲区
        if level == 2:  # 深度睡眠
            model = self.model_runner.model  # 获取主模型
            self._sleep_saved_buffers = {  # 把所有具名 buffer 拷贝到 CPU
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()  # 名称 -> CPU 副本
            }
            draft = self.get_draft_model()  # 获取草稿模型（投机解码）
            if draft is not None:  # 存在草稿模型时同样保存
                self._sleep_saved_draft_buffers = {  # 草稿模型缓冲区 -> CPU 副本
                    name: buffer.cpu().clone() for name, buffer in draft.named_buffers()  # 遍历具名 buffer
                }

        self._get_sleep_mode_backend().suspend(level)  # 调用后端挂载：level 1 释放权重，level 2 连 KV cache 一起释放

        torch.accelerator.synchronize()  # 再次同步，确保释放操作完成
        deadline = time.monotonic() + (5.0 if current_platform.is_rocm() else 0)  # ROCm 上设置 5 秒等待期限（释放可能延迟）
        while True:  # 轮询等待内存统计刷新
            free_bytes_after_sleep, total = torch.accelerator.get_memory_info()  # 读取当前空闲/总显存
            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep  # 计算实际释放量
            if freed_bytes >= 0 or time.monotonic() >= deadline:  # 已释放或超时
                break  # 退出轮询
            time.sleep(0.1)  # 短暂等待后重试（ROCm 统计可能延迟更新）

        used_bytes = total - free_bytes_after_sleep  # 计算仍在使用的显存
        assert freed_bytes >= 0, "Memory usage increased after sleeping."  # 断言 sleep 后内存未增加
        logger.info(  # 记录释放结果
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",  # 日志模板
            format_gib(freed_bytes),  # 释放的 GiB 数
            format_gib(used_bytes),  # 仍在使用的 GiB 数
        )

    def wake_up(self, tags: list[str] | None = None) -> None:  # 从 sleep 模式唤醒（可按标签部分唤醒）
        self._get_sleep_mode_backend().resume(tags)  # 调用后端恢复（tags 指定恢复 weights/kv_cache）

        # Restore the buffers after level 2 sleep
        # level 2 sleep 后恢复先前保存的缓冲区
        wake_weights = tags is None or "weights" in tags  # 判断是否需要恢复权重相关内容
        if wake_weights and len(self._sleep_saved_buffers):  # 需要恢复且有已保存 buffer
            model = self.model_runner.model  # 获取主模型
            for name, buffer in model.named_buffers():  # 遍历所有具名 buffer
                if name in self._sleep_saved_buffers:  # 该 buffer 之前被保存过
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)  # 从 CPU 副本恢复数据
            self._sleep_saved_buffers = {}  # 清空已保存副本

        if wake_weights and len(self._sleep_saved_draft_buffers):  # 恢复草稿模型 buffer
            draft = self.get_draft_model()  # 获取草稿模型
            if draft is not None:  # 存在时
                for name, buffer in draft.named_buffers():  # 遍历草稿模型 buffer
                    if name in self._sleep_saved_draft_buffers:  # 之前保存过的
                        buffer.data.copy_(self._sleep_saved_draft_buffers[name].data)  # 恢复数据
            self._sleep_saved_draft_buffers = {}  # 清空副本

        if tags is None or "kv_cache" in tags:  # 需要恢复 KV cache 时
            self.model_runner.post_kv_cache_wake_up()  # 通知 model runner 执行 KV cache 唤醒后处理

    def checkpoint_prepare(self) -> None:  # 准备分布式状态检查点
        checkpoint_prepare_distributed_state()  # 委托全局函数（容错快照前调用）

    def checkpoint_restore(self) -> None:  # 恢复分布式状态检查点
        checkpoint_restore_distributed_state()  # 委托全局函数（故障恢复时调用）

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:  # 获取内存池上下文（tag="weights"/"kv_cache"）
        if (
            current_platform.is_cuda_alike()  # CUDA 类平台（CUDA/ROCm）
            and not self.vllm_config.model_config.enable_cumem_allocator  # 且未启用 CuMem 分配器
        ):
            return nullcontext()  # 不需要内存池，返回空上下文

        if (
            current_platform.is_xpu()  # XPU 平台
            and not self.vllm_config.model_config.enable_sleep_mode  # 且未启用 sleep 模式
        ):
            return nullcontext()  # 无需内存池

        if current_platform.is_cpu():  # CPU 平台
            return nullcontext()  # 无需 GPU 内存池

        allocator = get_mem_allocator_instance()  # 获取 CuMem 分配器实例
        if tag == "weights":  # 权重内存池
            assert allocator.get_current_usage() == 0, (  # 断言当前无活跃使用
                "CuMem allocator can only be used for one instance per process."  # 每进程只能有一个实例
            )
        return allocator.use_memory_pool(tag=tag)  # 返回内存池上下文：池内分配的张量按 tag 归组

    @contextmanager  # 上下文管理器装饰器
    def _scoped_allocator_max_split(self, max_split_size_mb: int):  # 临时设置分配器 max_split_size_mb
        """Temporarily set max_split_size_mb to reduce allocator fragmentation at the
        cost of more cudaMalloc calls (negligible in practice). Restores the original
        value on exit.
        临时降低 max_split_size_mb 以减少分配器碎片（代价是更多 cudaMalloc，实际可忽略），退出时恢复原值。
        """
        if not current_platform.is_cuda():  # 非 CUDA 平台
            yield  # 直接通过，不做任何设置
            return

        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")  # 读取现有分配器配置
        match = re.search(r"max_split_size_mb:(\d+)", conf)  # 提取已有的 max_split_size_mb 值
        original_value = match.group(1) if match else None  # 保存原始值（可能为 None）

        torch._C._accelerator_setAllocatorSettings(  # 设置新的分配器参数
            f"max_split_size_mb:{max_split_size_mb}"  # 使用传入的小块拆分阈值
        )
        try:
            yield  # 执行受保护代码块
        finally:
            # PyTorch defaults to SIZE_MAX (no limit).
            # PyTorch 默认为 SIZE_MAX（无限制）
            _SIZE_MAX_MB = (2**64 - 1) // (1024 * 1024)  # 计算无限制对应的 MB 值
            restore = original_value if original_value else str(_SIZE_MAX_MB)  # 恢复原值或默认无限制
            torch._C._accelerator_setAllocatorSettings(  # 恢复分配器设置
                f"max_split_size_mb:{restore}"  # 恢复目标值
            )

    @instrument(span_name="Init device")  # 追踪：设备初始化 span
    def init_device(self):  # 初始化 GPU 设备与分布式环境
        if self.device_config.device_type == "cuda":  # CUDA 设备
            # This env var set by Ray causes exceptions with graph building.
            # Ray 设置的该环境变量会导致 CUDA 图构建异常，移除之
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)  # 删除该环境变量
            parallel_config = self.parallel_config  # 缓存并行配置
            if (
                parallel_config.distributed_executor_backend  # 执行器后端
                not in ("ray", "external_launcher")  # 非 Ray/外部启动器
                and parallel_config.data_parallel_backend != "ray"  # DP 后端也非 Ray
                and parallel_config.nnodes_within_dp == 1  # DP 内单节点
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                # 优先使用 DP 本地 rank，否则使用全局 DP rank
                dp_local_rank = self.parallel_config.data_parallel_rank_local  # DP 本地 rank
                if dp_local_rank is None:  # 未设置本地 rank
                    dp_local_rank = self.parallel_config.data_parallel_index  # 回退到全局 DP index

                tp_pp_world_size = (  # TP × PP 的世界大小
                    self.parallel_config.pipeline_parallel_size  # PP 大小
                    * self.parallel_config.tensor_parallel_size  # TP 大小
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                # 将本地 rank 调整为 DP 展开后的实际 GPU 序号
                self.local_rank += dp_local_rank * tp_pp_world_size  # 偏移本地 rank

            # Publish the logical-to-physical mapping for topology queries
            # such as NIC affinity and P2P checks.
            # 发布逻辑->物理 GPU 映射，供网卡亲和性与 P2P 检查等拓扑查询使用
            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids  # 分配的物理 GPU id 列表
            if assigned_physical_gpu_ids is not None:  # 指定了物理映射
                from vllm.platforms.interface import set_assigned_physical_gpu_ids  # 设置全局映射

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)  # 发布映射
                assert self.local_rank < len(assigned_physical_gpu_ids), (  # 断言本地 rank 在映射范围内
                    f"local_rank {self.local_rank} is out of bounds for "  # 越界信息
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"  # 映射内容
                )
                # NOTE(patch pr45026): local_world_size is derived from
                # parallel_config.nnodes, which is only set for the "mp"
                # multi-node backend. With the "ray"/"external_launcher"
                # backends nnodes stays 1, so local_world_size collapses to
                # the full world_size and this check wrongly fires on
                # cross-node deployments. assigned_physical_gpu_ids is already
                # per-node and the local_rank bound above fully validates the
                # mapping for these backends, so skip the check for them.
                # 注意（pr45026）：local_world_size 仅在 "mp" 多节点后端下正确；
                # ray/external_launcher 后端 nnodes 恒为 1，会导致误报，
                # 而上面的 local_rank 越界检查已足够，故跳过该检查
                if parallel_config.distributed_executor_backend not in (  # 非 ray/外部启动器后端
                    "ray",
                    "external_launcher",
                ):
                    assert self.parallel_config.local_world_size <= len(  # 断言本地世界大小不超过映射数
                        assigned_physical_gpu_ids
                    ), (
                        f"local_world_size ({self.parallel_config.local_world_size})"  # 本地世界大小
                        " exceeds assigned_physical_gpu_ids count "  # 超出提示
                        f"({len(assigned_physical_gpu_ids)})"  # 映射数量
                    )
            else:  # 未指定物理映射
                assert self.local_rank < torch.accelerator.device_count(), (  # 断言本地 rank 在可见设备数内
                    f"DP adjusted local rank {self.local_rank} is out of "  # 越界信息
                    f"bounds for {torch.accelerator.device_count()} devices."  # 设备数
                )

            visible_device_index = (  # 计算可见设备索引
                current_platform.logical_device_id_to_visible_device_id(self.local_rank)  # 逻辑 id -> 物理 id
            )
            self.device = torch.device(f"cuda:{visible_device_index}")  # 创建 torch 设备对象
            torch.accelerator.set_device_index(self.device)  # 设置当前设备

            current_platform.check_if_supports_dtype(self.model_config.dtype)  # 校验平台支持该模型精度

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            # 在内存快照前初始化分布式环境，
            # 确保 NCCL 缓冲区已计入可用内存测量
            init_worker_distributed_environment(  # 初始化分布式环境（本文件底部函数）
                self.vllm_config,  # 全局配置
                self.rank,  # 全局 rank
                self.distributed_init_method,  # 初始化方法
                self.local_rank,  # 本地 rank
                current_platform.dist_backend,  # 平台指定的通信后端（nccl 等）
            )

            if self.use_v2_model_runner:  # 使用 V2 model runner 时
                logger.info_once("Using V2 Model Runner")  # 打印一次提示

            # Set random seed.
            # 设置随机种子
            set_random_seed(self.model_config.seed)  # 保证各 worker 随机一致性

            # Now take memory snapshot after NCCL is initialized
            # NCCL 初始化完成后收集内存并做快照
            gc.collect()  # 触发垃圾回收
            torch.accelerator.empty_cache()  # 清空缓存分配器

            # take current memory snapshot
            # 记录当前内存快照
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)  # 初始内存快照
            self.requested_memory = request_memory(init_snapshot, self.cache_config)  # 按 gpu_memory_utilization 计算目标内存
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)  # 调试日志：快照
            logger.debug(  # 调试日志：请求内存
                "worker requested memory: %sGiB", format_gib(self.requested_memory)  # GiB 格式
            )
        else:  # 非 CUDA 设备
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")  # 不支持则报错

        # Initialize workspace manager
        # 初始化工作空间管理器
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1  # 启用 DBO（双批次重叠）时需要 2 个子批
        init_workspace_manager(self.device, num_ubatches)  # 按设备与子批数初始化

        # Construct the model runner
        # 构造 model runner
        if self.use_v2_model_runner:  # V2 版本
            from vllm.v1.worker.gpu.model_runner import (  # 延迟导入 V2 runner
                GPUModelRunner as GPUModelRunnerV2,  # 别名
            )

            # HACK(woosuk): This is a temporary fix to avoid type errors.
            # 临时方案：绕过类型检查（V2 runner 接口与 V1 略有差异）
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore  # 创建 V2 model runner
                self.vllm_config, self.device  # 传入配置与设备
            )
        else:  # V1 版本
            from vllm.v1.worker.gpu_model_runner import (  # 延迟导入 V1 runner
                GPUModelRunner as GPUModelRunnerV1,  # 别名
            )

            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)  # 创建 V1 model runner

        if self.rank == 0:  # 仅 rank 0
            # If usage stat is enabled, collect relevant info.
            # 若启用使用统计，收集相关信息上报
            report_usage_stats(self.vllm_config)  # 匿名使用统计

    def handle_ft_command(self, ft_request):  # 处理故障容忍控制命令
        assert self.worker_sentinel is not None  # 哨兵必须已创建
        return self.worker_sentinel.handle_command(ft_request)  # 委托哨兵处理

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    # 待办：改用 TorchDispatchMode 而非内存池来拦截张量分配
    def load_model(self, *, load_dummy_weights: bool = False) -> None:  # 加载模型权重
        with (
            self._maybe_get_memory_pool_context(tag="weights"),  # 权重分配进入 weights 内存池（支持 sleep 释放）
            set_current_vllm_config(self.vllm_config),  # 设置线程局部配置，供加载过程中的模块读取
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            # 20 MiB 是 PyTorch 允许的 max_split_size_mb 最小值
            self._scoped_allocator_max_split(max_split_size_mb=20),  # 减少权重加载时的碎片
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)  # 实际加载（load_dummy_weights 用于 dummy 权重）

        if self.vllm_config.weight_transfer_config is not None:  # 配置了权重传输
            self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(  # 按配置创建引擎
                self.vllm_config.weight_transfer_config,  # 传输配置
                self.vllm_config,  # 全局配置
                self.device,  # 当前设备
                self.model_runner.get_model(),  # 模型引用
            )

    def update_config(self, overrides: dict[str, Any]) -> None:  # 运行期更新配置
        self.model_runner.update_config(overrides)  # 委托 model runner

    def reload_weights(self, *args, **kwargs) -> None:  # 重新加载权重
        with set_current_vllm_config(self.vllm_config):  # 配置上下文
            self.model_runner.reload_weights(*args, **kwargs)  # 委托 model runner

    @torch.inference_mode()  # 推理模式：禁用 autograd 跟踪
    def determine_available_memory(self) -> int:  # 剖析并确定可用于 KV cache 的显存
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.

        剖析模型峰值显存，确定不 OOM 情况下可用于 KV cache 的字节数。
        引擎先做一次内存剖析，再计算 KV cache 可用字节数。
        提示：可通过 gpu_memory_utilization 参数限制显存使用。
        """
        maybe_apply_startup_plan(self)  # 应用启动计划（若存在预保存的 KV cache 内存值）

        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:  # 用户显式指定了 KV cache 内存
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            # 仍需一次 profile run 以针对 max_num_batched_tokens 编译模型
            self.model_runner.profile_run()  # 执行剖析运行（触发编译）

            msg = (  # 提示信息
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} "  # 初始空闲内存
                f"GiB, reserved {format_gib(kv_cache_memory_bytes)} GiB memory for "  # 预留的 KV cache 内存
                "KV Cache as specified by kv_cache_memory_bytes config and "  # 由配置指定
                "skipped memory profiling. This does not respect the "  # 跳过了内存剖析
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "  # 不遵循 gpu_memory_utilization
                "config when you want manual control of KV cache memory "  # 仅在手动控制时使用
                "size. If OOM'ed, check the difference of initial free "  # OOM 时对比
                "memory between the current run and the previous run "  # 两次运行的初始空闲内存差异
                "where kv_cache_memory_bytes is suggested and update it "  # 并据此调整
                "correspondingly."  # 对应值
            )
            logger.info(msg)  # 打印提示
            return reserve_mm_ipc_gpu_memory(  # 预留多模态 IPC 显存后返回最终值
                kv_cache_memory_bytes,  # 用户指定的 KV cache 内存
                self.model_config.multimodal_config,  # 多模态配置
                getattr(self.parallel_config, "_api_process_count", 1),  # API 进程数（IPC 预留倍数）
            )

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        # 用 dummy 输入执行前向，剖析模型内存使用
        with memory_profiling(  # 内存剖析上下文
            self.init_snapshot,  # 初始快照作为基线
            weights_memory=int(self.model_runner.model_memory_usage),  # 已知权重内存
        ) as profile_result:  # 剖析结果
            self.model_runner.profile_run()  # dummy 前向运行

        # Profile CUDA graph memory if graphs will be captured.
        # ROCm is included: #44825 moved the profiler to
        # torch.accelerator.get_memory_info (reliable on ROCm, as used by
        # the AMD-CI mem tests), and graph_pool_handle resolves to the same
        # torch.cuda handle the live capture path already uses on ROCm.
        # XPU stays excluded (see #39977).
        # 若将捕获 CUDA 图则剖析其内存；ROCm 已包含（剖析器已迁移至
        # torch.accelerator.get_memory_info，在 ROCm 上可靠）；XPU 仍排除（见 #39977）
        cudagraph_memory_estimate = 0  # CUDA 图内存估计（默认 0）
        if (
            current_platform.is_cuda_alike()  # CUDA 类平台
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE  # 且启用 CUDA 图
        ):
            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()  # 剖析图内存占用

        # Respect the opt-in flag as originally designed.
        # 尊重可选开关的原始设计意图
        cudagraph_memory_estimate_applied = (  # 实际计入剖析的图内存
            cudagraph_memory_estimate  # 开启时使用估计值
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS  # 环境变量开关
            else 0  # 关闭时计 0
        )

        self.total_consumed = profile_result.total_consumed  # 记录总消耗（权重 + 非 torch）
        self.peak_activation_memory = (  # 记录峰值激活内存
            profile_result.transient_peak_headroom + cudagraph_memory_estimate_applied  # 瞬时峰值余量 + 图内存
        )
        self.cudagraph_memory_estimate = cudagraph_memory_estimate  # 保存图内存估计供后续比对

        free_gpu_memory = profile_result.after_profile.free_memory  # 剖析后空闲显存
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        # 注意：这里假设共享同 GPU 的其他进程在剖析期间未改变内存使用
        assert self.init_snapshot.free_memory >= free_gpu_memory, (  # 断言空闲内存未反常增加
            "Error in memory profiling. "  # 剖析错误提示
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "  # 初始空闲
            f"current free memory {format_gib(free_gpu_memory)} GiB. "  # 当前空闲
            "This happens when other processes sharing the same container "  # 常见于同容器其他进程
            "release GPU memory while vLLM is profiling during initialization. "  # 在剖析期间释放显存
            "To fix this, ensure consistent GPU memory allocation or "  # 修复：保持显存分配一致
            "isolate vLLM in its own container."  # 或隔离 vLLM
        )
        self.available_kv_cache_memory_bytes = (  # 计算可用 KV cache 内存
            self.requested_memory  # 目标请求内存
            - profile_result.non_kv_cache_memory  # 减去非 KV cache 内存
            - cudagraph_memory_estimate_applied  # 减去 CUDA 图内存
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory  # 请求之外的内存（未申请部分）
        logger.debug(  # 调试日志：初始空闲/请求
            "Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",  # 模板
            format_gib(self.init_snapshot.free_memory),  # 初始空闲
            self.cache_config.gpu_memory_utilization,  # 利用率
            format_gib(self.requested_memory),  # 请求内存
        )
        logger.debug(  # 调试日志：剖析后空闲
            "Free memory after profiling: %s GiB (total), %s GiB (within requested)",  # 模板
            format_gib(free_gpu_memory),  # 总空闲
            format_gib(free_gpu_memory - unrequested_memory),  # 请求范围内的空闲
        )
        logger.debug(profile_result)  # 调试日志：剖析明细
        logger.info_once(  # 打印一次：可用 KV cache 内存
            "Available KV cache memory: %s GiB",  # 模板
            format_gib(self.available_kv_cache_memory_bytes),  # 可用内存
        )

        if cudagraph_memory_estimate > 0:  # 有 CUDA 图内存估计时
            total_mem = self.init_snapshot.total_memory  # 总显存
            current_util = self.cache_config.gpu_memory_utilization  # 当前利用率配置
            cg_util_delta = cudagraph_memory_estimate / total_mem  # 图内存等效的利用率占比
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:  # 启用了图内存剖析
                equiv_util = round(current_util - cg_util_delta, 4)  # 等效的无剖析利用率
                suggested_util = min(  # 建议利用率（不超过 1.0）
                    round(current_util + cg_util_delta, 4),  # 当前值加上图内存占比
                    1.0,  # 上限
                )
                logger.info(  # 打印解释信息
                    "CUDA graph memory profiling is enabled (default since "  # 剖析已启用
                    "v0.21.0). The current --gpu-memory-utilization=%.4f is "  # 当前值
                    "equivalent to --gpu-memory-utilization=%.4f without "  # 等效值
                    "CUDA graph memory profiling. To maintain the same "  # 保持相同
                    "effective KV cache size as before, increase "  # 有效 KV cache 容量
                    "--gpu-memory-utilization to %.4f. To disable, set "  # 建议值
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",  # 关闭方法
                    current_util,  # 参数：当前值
                    equiv_util,  # 参数：等效值
                    suggested_util,  # 参数：建议值
                )
            else:  # 未启用图内存剖析
                suggested_util = min(  # 建议值
                    round(current_util + cg_util_delta, 4),  # 加上图内存占比
                    1.0,  # 上限
                )
                logger.warning(  # 警告：关闭可能导致 OOM
                    "CUDA graph memory profiling is disabled "  # 已禁用
                    "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). "  # 开关来源
                    "Without it, CUDA graph memory is not accounted for "  # 图内存未计入
                    "during KV cache allocation, which may require lowering "  # 可能需要降低
                    "--gpu-memory-utilization to avoid OOM. Consider "  # 利用率以避免 OOM
                    "re-enabling it (the default as of v0.21.0) and increasing "  # 建议重新启用
                    "--gpu-memory-utilization from %.4f to %.4f.",  # 并提高利用率
                    current_util,  # 当前值
                    suggested_util,  # 建议值
                )

        return reserve_mm_ipc_gpu_memory(  # 最终返回：预留多模态 IPC 显存后的 KV cache 可用内存
            int(self.available_kv_cache_memory_bytes),  # 计算出的可用内存
            self.model_config.multimodal_config,  # 多模态配置
            getattr(self.parallel_config, "_api_process_count", 1),  # API 进程数
        )

    def get_kv_connector_handshake_metadata(  # 获取 KV 连接器握手元数据
        self,
    ) -> dict[tuple[int, int], KVConnectorHandshakeMetadata] | None:  # (pp_rank, tp_rank) -> 元数据
        """Get KV connector metadata from this worker if available.

        Returned dict is keyed by `(pp_rank, tp_rank)`.
        获取本 worker 的 KV 连接器元数据（若可用），以 (pp_rank, tp_rank) 为键。
        """

        if not has_kv_transfer_group():  # 未创建 KV 传输组
            return None  # 无元数据

        connector = get_kv_transfer_group()  # 获取连接器
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        # 无需跨 worker 交换握手元数据的连接器返回 None
        if (metadata := connector.get_handshake_metadata()) is None:  # 无元数据
            return None

        pp_rank = get_pp_group().rank_in_group  # 本 worker 在 PP 组内的 rank
        tp_rank = get_tp_group().rank_in_group  # 本 worker 在 TP 组内的 rank
        return {(pp_rank, tp_rank): metadata}  # 返回映射

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:  # 获取各层 KV cache 规格
        return self.model_runner.get_kv_cache_spec()  # 委托 model runner

    def update_max_model_len(self, max_model_len: int) -> None:  # 自动适配显存后更新最大模型长度
        """Update max_model_len after auto-fit to GPU memory.
        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.

        当 max_model_len=-1 时引擎自动确定适配显存的最大上下文长度，
        worker 需同步更新其缓存的 max_model_len。
        """
        self.model_config.max_model_len = max_model_len  # 更新本地配置
        if self.model_runner is not None:  # model runner 已创建
            self.model_runner.update_max_model_len(max_model_len)  # 同步更新 runner
        logger.debug("Updated max_model_len to %d", max_model_len)  # 调试日志

    @instrument(span_name="Allocate KV cache")  # 追踪：KV cache 分配 span
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:  # 按配置分配 KV cache
        """Allocate GPU KV cache with the specified kv_cache_config.
        按指定的 kv_cache_config 分配 GPU KV cache。"""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        # 用剖析后调整的块数更新本地配置，供预热阶段使用
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks  # 更新 GPU 块数

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        # 此处初始化 KV 连接器（需要 kv_cache_config）；
        # 必须在 initialize_kv_cache 之前，因为后者会注入与连接器无关的组
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)  # 初始化 KV 传输

        with self._maybe_get_memory_pool_context(tag="kv_cache"):  # KV cache 进入专用内存池
            self.model_runner.initialize_kv_cache(kv_cache_config)  # 实际分配 KV cache

        if self.model_config.enable_return_routed_experts:  # 需要返回专家路由信息
            self.model_runner.init_routed_experts_capturer()  # 初始化路由捕获器

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        # 在 CuMem 池外构建 KV-zero 元数据，使簿记张量使用标准分配器，
        # 避免 sleep/wake 周期中被丢弃
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(  # 需要清零且 runner 支持时
            self.model_runner, "_init_kv_zero_meta"
        ):
            self.model_runner._init_kv_zero_meta()  # 初始化清零元数据

    @instrument(span_name="Warmup (GPU)")  # 追踪：GPU 预热 span
    def compile_or_warm_up_model(self) -> CompilationTimes:  # 编译与预热模型
        warmup_sizes: list[int] = []  # 需要预热的 token 数集合

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:  # 启用 vLLM 编译
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            # 预热不在 CUDA 图捕获尺寸内但用户希望编译的尺寸
            # （如 chunked prefill 的 max-num-batched-token）
            compile_sizes = self.vllm_config.compilation_config.compile_sizes  # 用户指定编译尺寸
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]  # 拷贝为初始集
            cg_capture_sizes: list[int] = []  # CUDA 图捕获尺寸

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:  # 启用 CUDA 图
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes  # 读取捕获尺寸
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes  # 空则无
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]  # 排除已由图覆盖的尺寸

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()  # 编译范围列表
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            # 对每个编译范围，若无任何尺寸落在其中，则加入范围末端以保证编译覆盖
            all_sizes = set(cg_capture_sizes)  # 已覆盖尺寸集合
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])  # 加入预热尺寸
            for compile_range in compile_ranges:  # 遍历范围
                if not any(x in compile_range for x in all_sizes):  # 范围内无任何已知尺寸
                    warmup_sizes.append(compile_range.end)  # 加入范围末端

        # We skip EPLB here since we don't want to record dummy metrics
        # 跳过 EPLB，避免记录 dummy 指标
        for size in sorted(warmup_sizes, reverse=True):  # 从大到小预热
            logger.info("Compile and warming up model for size %d", size)  # 日志
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)  # dummy 前向触发编译
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)  # 预热后清理 LoRA

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        # 在 CUDA 图捕获前预热/调优模型执行用到的 kernel
        kernel_warmup(self)  # kernel 级预热

        cuda_graph_memory_bytes = 0  # 实际图内存（默认 0）
        if not self.model_config.enforce_eager:  # 非 eager 模式
            cuda_graph_memory_bytes = self.model_runner.capture_model()  # 捕获 CUDA 图并返回其内存

        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        # 比较实际与预估的 CUDA 图内存（若做过预估）
        if (
            hasattr(self, "cudagraph_memory_estimate")  # 存在估计值
            and self.cudagraph_memory_estimate > 0  # 且大于 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)  # 字节转 GiB 的辅助函数
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)  # 差值
            logger.info(  # 记录对比结果
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "  # 实际/估计
                "difference: %s GiB (%.1f%%).",  # 差值与百分比
                GiB(cuda_graph_memory_bytes),  # 实际
                GiB(self.cudagraph_memory_estimate),  # 估计
                GiB(diff),  # 差值
                100 * diff / max(cuda_graph_memory_bytes, 1),  # 相对偏差（避免除零）
            )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(  # 未手动指定 KV cache 内存且做过剖析
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            # 依赖 memory_profiling 估计 KV cache 大小时给出优化建议；
            # 剖析未计入 CUDA 图内存，且可能低估消耗，故预留 150MiB 缓冲避免 OOM
            redundancy_buffer_memory = 150 * (1 << 20)  # 150 MiB 安全缓冲

            non_kv_cache_memory = (  # 非 KV cache 内存总量
                self.total_consumed  # 已消耗（权重+非 torch）
                + self.peak_activation_memory  # 峰值激活
                + cuda_graph_memory_bytes  # CUDA 图
            )
            kv_cache_memory_bytes_to_gpu_limit = (  # 按显存上限计算的可用 KV cache 内存
                self.init_snapshot.free_memory  # 初始空闲
                - non_kv_cache_memory  # 减非 KV 内存
                - redundancy_buffer_memory  # 减缓冲
            )
            kv_cache_memory_bytes_to_requested_limit = (  # 按请求上限计算的可用 KV cache 内存
                int(self.requested_memory)  # 请求内存
                - non_kv_cache_memory  # 减非 KV 内存
                - redundancy_buffer_memory  # 减缓冲
            )

            msg = (  # 构造建议信息
                f"Free memory on device "  # 设备空闲内存
                f"({format_gib(self.init_snapshot.free_memory)}/"  # 空闲
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "  # /总内存
                f"Desired GPU memory utilization is "  # 期望利用率
                f"({self.cache_config.gpu_memory_utilization}, "  # 数值
                f"{format_gib(self.requested_memory)} GiB). "  # 对应内存
                f"Actual usage is {format_gib(self.total_consumed)} "  # 实际消耗
                f"GiB for consumed memory (weights + non-torch), "  # 权重+非 torch
                f"{format_gib(self.peak_activation_memory)} GiB "  # 峰值激活
                f"for peak activation, and {format_gib(cuda_graph_memory_bytes)} "  # CUDA 图内存
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "  # 建议：用精确的
                f"config with `--kv-cache-memory="  # kv-cache-memory 配置
                f"{kv_cache_memory_bytes_to_requested_limit}` "  # 请求上限方案
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "  # 适配请求内存
                f"into requested memory, or `--kv-cache-memory="  # 或
                f"{kv_cache_memory_bytes_to_gpu_limit}` "  # 显存上限方案
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "  # 充分利用显存
                f"utilize gpu memory. Current kv cache memory in use is "  # 当前
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."  # 正在使用的 KV cache
            )

            logger.info(msg)  # 打印建议

            maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)  # 保存启动计划（请求上限方案）

        if self.use_v2_model_runner:  # V2 runner
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            # V2：完整运行 execute_model + sample_tokens 以 JIT 编译 triton kernel
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)  # kernel 预热
        elif get_pp_group().is_last_rank:  # V1：仅 PP 最后一级需要预热采样器
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            # V1：预热采样器并预分配 logits 等最大形状缓冲以避免碎片；
            # 特意放在 capture_model 之后，防止缓冲被 empty_cache 清除
            max_num_reqs = min(  # 最大并发请求数
                self.scheduler_config.max_num_seqs,  # 最大序列数
                self.scheduler_config.max_num_batched_tokens,  # 最大批 token 数
            )

            # We skip EPLB here since we don't want to record dummy metrics
            # 跳过 EPLB，避免记录 dummy 指标
            hidden_states, last_hidden_states = self.model_runner._dummy_run(  # dummy 运行取隐藏状态
                num_tokens=max_num_reqs,  # token 数 = 最大请求数（uniform decode 场景）
                skip_eplb=True,  # 跳过 EPLB，不记录 dummy 指标
                cudagraph_runtime_mode=CUDAGraphMode.NONE,  # 不使用 CUDA 图
            )
            if self.model_runner.is_pooling_model:  # 池化模型
                self.model_runner._dummy_pooler_run(hidden_states)  # 预热池化器
            else:  # 生成模型
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)  # 预热采样器（预分配 logits 缓冲）

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        # 重置随机种子，避免初始化/剖析影响运行期随机状态
        set_random_seed(self.model_config.seed)  # 恢复种子

        # Eagerly trigger inductor's once-per-process lazy inits during
        # warmup (rather than on a later compile cache-miss at runtime).
        # 在预热期主动触发 inductor 的每进程一次性惰性初始化
        # （而非运行期编译缓存未命中时才触发）
        c_config = self.compilation_config  # 缓存编译配置
        if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":  # 使用 inductor 后端时
            from vllm.compilation.compiler_interface import (  # 延迟导入接口
                trigger_inductor_lazy_init,  # 惰性初始化触发器
            )

            trigger_inductor_lazy_init(self.device)  # 立即初始化 inductor

        # All warmup is done — start monitoring for unexpected JIT
        # compilations that would cause latency spikes during inference.
        # 预热完毕——开始监控推理期意外的 JIT 编译（会导致延迟尖峰）
        from vllm.utils.jit_monitor import activate as activate_jit_monitor  # JIT 监控器

        activate_jit_monitor(  # 激活监控
            mode=self.observability_config.jit_monitor_mode,  # 监控模式
            verbose=self.observability_config.jit_monitor_verbose,  # 是否详细输出
        )

        # Freeze the worker heap so the GC won't scan static objects
        # (model weights, KV caches, CUDA graphs) during inference.
        # 冻结 worker 堆，使 GC 不再扫描静态对象（权重/KV cache/CUDA 图）
        freeze_gc_heap()  # 冻结 GC 堆以稳定延迟
        maybe_attach_gc_debug_callback()  # 按需挂载 GC 调试回调

        # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
        # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
        # 预热/首次编译完成——激活 VLLM_GPU_SYNC_CHECK 门控，
        # 使后续 execute_model/sample_tokens 调用强制同步检查
        enable_gpu_sync_check()  # 启用 GPU 同步检查

        # Startup is done; steady-state serving gets no benefit from torch
        # intra-op parallelism.
        # 启动完成；稳态服务无需 torch 算子内并行
        set_torch_threads_for_runtime()  # 调整 torch 线程数以适配运行期

        return CompilationTimes(  # 返回编译耗时统计
            language_model=self.compilation_config.compilation_time,  # 语言模型编译耗时
            encoder=self.compilation_config.encoder_compilation_time,  # 编码器编译耗时
        )

    def reset_mm_cache(self) -> None:  # 重置多模态处理器缓存
        self.model_runner.reset_mm_cache()  # 委托 model runner

    def reset_encoder_cache(self) -> None:  # 重置编码器缓存
        self.model_runner.reset_encoder_cache()  # 委托 model runner

    def get_model(self) -> nn.Module:  # 获取主模型
        return self.model_runner.get_model()  # 委托 model runner

    def get_draft_model(self) -> nn.Module | None:  # 获取草稿模型（投机解码）
        return self.model_runner.get_draft_model()  # 无则返回 None

    def _set_draft_weight_update_target(self) -> None:  # 将权重更新目标设为草稿模型
        assert self.weight_transfer_engine is not None  # 引擎必须已创建

        draft_model = self.get_draft_model()  # 获取草稿模型
        if draft_model is None:  # 不存在
            raise RuntimeError(  # 报错
                "Draft model weight update requested, but no draft model is configured."  # 未配置草稿模型
            )

        speculative_config = self.speculative_config  # 投机解码配置
        if speculative_config is None or speculative_config.draft_model_config is None:  # 未配置
            raise RuntimeError(  # 报错
                "Draft model weight update requested, but no draft model "  # 请求了草稿模型更新
                "config is configured."  # 但无对应配置
            )

        self.weight_transfer_engine.set_weight_update_target(  # 设置更新目标
            draft_model, speculative_config.draft_model_config  # 草稿模型及其配置
        )

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:  # 获取支持的任务类型
        return self.model_runner.get_supported_tasks()  # 委托 model runner

    def get_compilation_match_table(self) -> dict[str, int]:  # 获取编译 pass 匹配表
        from vllm.compilation.passes.vllm_inductor_pass import get_match_table  # 延迟导入

        return get_match_table()  # 返回 pass 名称 -> 优先级映射

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:  # 获取编码器计时统计
        """Get encoder timing stats from model runner.
        从 model runner 获取编码器计时统计。"""
        return self.model_runner.get_encoder_timing_stats()  # 委托 model runner

    def annotate_profile(self, scheduler_output):  # 为剖析 trace 添加本轮迭代注解
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        # 添加 trace 注解以便区分每轮迭代的预填充/生成请求数；
        # context 请求指尚未生成任何 token 的请求
        if not self.profiler:  # 未启用剖析器
            return nullcontext()  # 返回空上下文

        self.profiler.step()  # 推进剖析器步数（trace 中区分迭代）

        iteration_details = compute_iteration_details(scheduler_output)  # 计算本轮迭代明细

        if self.vllm_config.profiler_config.detailed_trace_annotation:  # 启用详细注解
            # Compute roofline-model metrics per request, split by phase
            # (context vs generation). These help estimate compute and
            # memory intensity from the trace.
            #
            # Per-request quantities:
            #   query_len = number of scheduled (new) tokens for this request
            #   seq_len   = total sequence length (computed + scheduled tokens)
            #
            # Aggregated across requests in each phase
            # (ctx_=context, gen_=generation):
            #   seq_len_sum = sum of seq_len   (total KV length)
            #   qq_compute  = sum of query_len*query_len
            #                 (proxy for QK^T compute cost)
            #   qk_compute  = sum of query_len*seq_len
            #                 (proxy for QK^T compute cost for decode and
            #                  chunked prefill)
            #   total_scheduled_tokens = scheduled tokens across all requests
            # 按阶段（预填充/生成）计算每请求的 roofline 指标，
            # 用于从 trace 估算算力与访存强度：
            #   query_len = 本请求调度 token 数；seq_len = 总序列长度
            #   聚合量：seq_len_sum（KV 总长）、qq/qk_compute（QK^T 代理）
            ctx_seq_len_sum = 0  # 预填充阶段 seq_len 之和
            ctx_qq_compute = 0  # 预填充阶段 QK^T 代理（query²）
            ctx_qk_compute = 0  # 预填充阶段 QK^T 代理（query×seq）
            gen_seq_len_sum = 0  # 生成阶段 seq_len 之和
            gen_qq_compute = 0  # 生成阶段 QK^T 代理
            gen_qk_compute = 0  # 生成阶段 QK^T 代理
            total_scheduled_tokens = 0  # 本轮调度 token 总数

            # Build a map of req_id -> num_computed_tokens for all requests
            # 构建所有请求的 req_id -> 已计算 token 数映射
            new_req_ids = {  # 新请求 id 集合
                new_req.req_id for new_req in scheduler_output.scheduled_new_reqs  # 遍历新请求
            }
            num_computed_tokens_ids = {  # 新请求的已计算 token 数
                new_req.req_id: new_req.num_computed_tokens  # id -> 已计算数
                for new_req in scheduler_output.scheduled_new_reqs  # 遍历新请求
            }
            for req_id, num_computed_tokens in zip(  # 合并已缓存请求
                scheduler_output.scheduled_cached_reqs.req_ids,  # 缓存请求 id
                scheduler_output.scheduled_cached_reqs.num_computed_tokens,  # 对应已计算数
            ):
                num_computed_tokens_ids[req_id] = num_computed_tokens  # 更新映射

            # Accumulate per-phase metrics
            # 按阶段累加指标
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():  # 遍历调度计划
                query_len = num_tokens  # query 长度 = 调度 token 数
                total_scheduled_tokens += query_len  # 累加总调度数
                seq_len = num_computed_tokens_ids.get(req_id, 0) + query_len  # 序列总长
                if (
                    scheduler_output.scheduled_cached_reqs.is_context_phase(req_id)  # 处于预填充阶段
                    or req_id in new_req_ids  # 或为新请求
                ):
                    ctx_seq_len_sum += seq_len  # 累加预填充指标
                    ctx_qq_compute += query_len * query_len  # query² 代理
                    ctx_qk_compute += query_len * seq_len  # query×seq 代理
                else:  # 生成阶段
                    gen_seq_len_sum += seq_len  # 累加生成指标
                    gen_qq_compute += query_len * query_len  # query² 代理
                    gen_qk_compute += query_len * seq_len  # query×seq 代理
            annotation = "".join(  # 拼接详细注解字符串
                [
                    "execute_",  # 前缀
                    str(total_scheduled_tokens),  # 总调度 token
                    "_context_",  # 预填充部分
                    str(iteration_details.num_ctx_requests),  # 请求数
                    "(sq",  # 调度 token 数
                    str(iteration_details.num_ctx_tokens),  # 值
                    "sk",  # seq 长度和
                    str(ctx_seq_len_sum),  # 值
                    "sqsq",  # query² 代理
                    str(ctx_qq_compute),  # 值
                    "sqsk",  # query×seq 代理
                    str(ctx_qk_compute),  # 值
                    ")_generation_",  # 生成部分
                    str(iteration_details.num_generation_requests),  # 请求数
                    "(sq",  # 调度 token 数
                    str(iteration_details.num_generation_tokens),  # 值
                    "sk",  # seq 长度和
                    str(gen_seq_len_sum),  # 值
                    "sqsq",  # query² 代理
                    str(gen_qq_compute),  # 值
                    "sqsk",  # query×seq 代理
                    str(gen_qk_compute),  # 值
                    ")",  # 结束
                ]
            )
        else:  # 简单注解模式
            annotation = "".join(  # 仅包含请求数与 token 数
                [
                    "execute_context_",  # 预填充部分
                    str(iteration_details.num_ctx_requests),  # 请求数
                    "(",  # 括号
                    str(iteration_details.num_ctx_tokens),  # token 数
                    ")",  # 结束
                    "_generation_",  # 生成部分
                    str(iteration_details.num_generation_requests),  # 请求数
                    "(",  # 括号
                    str(iteration_details.num_generation_tokens),  # token 数
                    ")",  # 结束
                ]
            )
        return self.profiler.annotate_context_manager(annotation)  # 返回带注解的上下文管理器

    @torch.inference_mode()  # 推理模式：禁用 autograd
    @with_gpu_sync_check  # GPU 同步检查调试装饰器（VLLM_GPU_SYNC_CHECK 门控）
    def sample_tokens(  # 采样 token（与 execute_model 分离，支持异步调度）
        self, grammar_output: "GrammarOutput | None"  # 文法（结构化输出）采样结果
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:  # 返回采样输出
        return self.model_runner.sample_tokens(grammar_output)  # 委托 model runner

    @torch.inference_mode()  # 推理模式：禁用 autograd
    @with_gpu_sync_check  # GPU 同步检查调试装饰器
    def execute_model(  # 执行模型前向（一个调度步）
        self, scheduler_output: "SchedulerOutput"  # 调度器输出
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:  # 中间级返回 None（继续传给下一 PP 级）
        # ensure any previous non-blocking PP sends are complete
        # 确保上一次迭代的非阻塞 PP 发送已完成
        if self._pp_send_work:  # 存在遗留发送任务
            for handle in self._pp_send_work:  # 逐个等待
                handle.wait()  # 阻塞直到完成
            self._pp_send_work = []  # 清空列表

        intermediate_tensors = None  # 上一 PP 级传来的中间张量（首级为 None）
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0  # 本轮是否有前向计算
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 调度 token 总数
        all_gather_tensors = {}  # SP 场景需要在 TP 组 allgather 的张量集合
        compilation_config = self.vllm_config.compilation_config  # 缓存编译配置
        parallel_config = self.vllm_config.parallel_config  # 缓存并行配置

        if (
            parallel_config.pipeline_parallel_size > 1  # 启用流水线并行
            and compilation_config.pass_config.enable_sp  # 且启用序列并行
            and forward_pass  # 且本轮有前向
        ):
            # currently only supported by V1 GPUModelRunner
            # 目前仅 V1 GPUModelRunner 支持
            assert not self.use_v2_model_runner  # 断言非 V2
            num_scheduled_tokens_np = np.array(  # 每请求调度 token 数数组
                list(scheduler_output.num_scheduled_tokens.values()),  # 取值列表
                dtype=np.int32,  # int32 类型
            )
            # TODO(lucas): This is pretty gross; ideally we should only ever call
            # `_determine_batch_execution_and_padding` once (will get called again
            # in `execute_model`) but this requires a larger refactor of PP.
            # 待办：理想情况下只调用一次批描述计算（execute_model 内会再调一次），
            # 但这需要较大规模的 PP 重构
            _, batch_desc, _, _, _ = (  # 计算批执行描述
                self.model_runner._determine_batch_execution_and_padding(  # 确定填充与批描述
                    num_tokens=num_scheduled_tokens,  # 总 token 数
                    num_reqs=len(num_scheduled_tokens_np),  # 请求数
                    num_scheduled_tokens_np=num_scheduled_tokens_np,  # 每请求 token 数组
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),  # 最大值
                    use_cascade_attn=False,  # TODO(lucas): Handle cascade attention  # 暂不支持级联注意力
                )
            )
            all_gather_tensors = {  # residual 是否需要 allgather
                "residual": not is_residual_scattered_for_sp(  # 若未分散存储则需要聚合
                    self.vllm_config, batch_desc.num_tokens  # 依据配置与 token 数判断
                )
            }

        if forward_pass and not get_pp_group().is_first_rank:  # 非首级且需前向：先接收
            tensor_dict, comm_handles, comm_postprocess = (  # 非阻塞接收中间张量
                get_pp_group().irecv_tensor_dict(  # 异步接收张量字典
                    all_gather_group=get_tp_group(),  # allgather 使用 TP 组
                    all_gather_tensors=all_gather_tensors,  # 需要 allgather 的张量
                )
            )
            assert tensor_dict is not None  # 断言接收成功
            intermediate_tensors = AsyncIntermediateTensors(  # 包装为异步中间张量
                tensor_dict,  # 张量字典
                comm_handles=comm_handles,  # 通信句柄（惰性等待）
                comm_postprocess=comm_postprocess,  # 后处理回调
            )

        with self.annotate_profile(scheduler_output):  # 剖析注解上下文
            output = self.model_runner.execute_model(  # 执行模型前向
                scheduler_output, intermediate_tensors  # 传入调度计划与中间张量
            )
            if (
                self.use_v2_model_runner  # V2 runner
                and self.model_runner.is_pooling_model  # 且为池化模型
                and output is None  # 且无输出
            ):
                output = self.model_runner.pool()  # type: ignore  # 执行池化获得输出
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType  # 已是最终输出类型
            ):
                return output  # 直接返回

        assert isinstance(output, IntermediateTensors)  # 中间级应返回中间张量
        parallel_config = self.vllm_config.parallel_config  # 重新读取并行配置
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"  # 非外部启动器
            and not get_pp_group().is_last_rank  # 且非最后一级（最后一级应返回最终输出）
        )

        # launch non-blocking send of intermediate tensors
        # 发起非阻塞的中间张量发送（传给下一 PP 级）
        self._pp_send_work = get_pp_group().isend_tensor_dict(  # 异步发送并保存句柄
            output.tensors,  # 中间张量
            all_gather_group=get_tp_group(),  # allgather 使用 TP 组
            all_gather_tensors=all_gather_tensors,  # 需要 allgather 的张量
        )

        return None  # 中间级：无最终输出（张量已发送给下一级）

    def take_draft_token_ids(self) -> DraftTokenIds | None:  # 取出投机解码草稿 token id
        return self.model_runner.take_draft_token_ids()  # 委托 model runner

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):  # 启动/停止性能剖析
        # Check if profiling is enabled
        # 检查剖析是否已启用
        if self.profiler_config is None or self.profiler_config.profiler is None:  # 未配置
            raise RuntimeError(  # 报错
                "Profiling is not enabled. Please set --profiler-config to enable "  # 提示启用方法
                "profiling. Example: "  # 示例
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"  # torch 剖析器
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"  # 输出目录
            )

        if is_start:  # 启动剖析
            # Generate the trace name by combining prefix with comprehensive rank suffix
            # 用前缀 + 全局 rank 后缀生成 trace 文件名
            from vllm.distributed.utils import get_worker_rank_suffix  # 延迟导入

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)  # rank 后缀

            # Build the full trace name
            # 构建完整 trace 名
            if profile_prefix:  # 指定了前缀
                trace_name = f"{profile_prefix}_{rank_suffix}"  # 前缀_rank
            else:  # 无前缀
                trace_name = rank_suffix  # 仅 rank 后缀

            # Create the profiler wrapper only on the first start call
            # 仅在首次启动时创建剖析器包装
            if self.profiler is None:  # 尚未创建
                profiler_type = self.profiler_config.profiler  # 剖析器类型
                if profiler_type == "torch":  # torch profiler
                    self.profiler = TorchProfilerWrapper(  # 创建包装
                        self.profiler_config,  # 配置
                        worker_name=trace_name,  # worker 名称（trace 文件名）
                        local_rank=self.local_rank,  # 本地 rank
                        activities=["CPU", "CUDA"],  # 同时剖析 CPU 与 CUDA
                    )
                    logger.debug(  # 调试日志
                        "Starting torch profiler with trace name: %s", trace_name  # trace 名
                    )
                elif profiler_type == "cuda":  # CUDA profiler（nsys 等）
                    self.profiler = CudaProfilerWrapper(self.profiler_config)  # 创建包装
                    logger.debug("Starting CUDA profiler")  # 调试日志
                else:  # 其他
                    # Config validation should prevent this code being reached
                    # 配置校验应已阻止到达此处
                    raise ValueError(  # 报错
                        f"Invalid profiler value of {self.profiler_config.profiler}"  # 非法值
                    )

            # If profiler already initialized, restart profiling but keep
            # the original trace name from the first initialization.
            # 若已初始化则重启剖析，但保留首次的 trace 名
            self.profiler.start()  # 启动
        else:  # 停止剖析
            if self.profiler is None:  # 未启动过
                logger.warning("Profiler was not started, nothing to stop.")  # 警告
                return  # 直接返回
            self.profiler.stop()  # 停止并导出 trace

    def execute_dummy_batch(self) -> None:  # 执行 dummy 批（如同步调度模式的预热）
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)  # uniform decode 的 query 长度（默认 1）
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)  # dummy 运行

    def add_lora(self, lora_request: LoRARequest) -> bool:  # 动态加载 LoRA
        return self.model_runner.add_lora(lora_request)  # 委托 model runner

    def remove_lora(self, lora_id: int) -> bool:  # 卸载 LoRA
        return self.model_runner.remove_lora(lora_id)  # 委托 model runner

    def list_loras(self) -> set[int]:  # 列出已加载的 LoRA id
        return self.model_runner.list_loras()  # 委托 model runner

    def pin_lora(self, lora_id: int) -> bool:  # 固定 LoRA（防止被逐出）
        return self.model_runner.pin_lora(lora_id)  # 委托 model runner

    def check_health(self) -> None:  # 健康检查
        # worker will always be healthy as long as it's running.
        # 只要 worker 进程在运行即视为健康
        return

    def save_sharded_state(  # 保存分片模型状态
        self,
        path: str,  # 保存路径
        pattern: str | None = None,  # 文件名模式
        max_size: int | None = None,  # 分片最大大小
    ) -> None:  # 无返回值
        from vllm.model_executor.model_loader import ShardedStateLoader  # 延迟导入分片加载器

        ShardedStateLoader.save_model(  # 按当前 TP 分片保存模型
            self.model_runner.model,  # 模型
            path,  # 路径
            pattern=pattern,  # 模式
            max_size=max_size,  # 大小上限
        )

    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:  # 保存为 Tensorizer 格式
        TensorizerLoader.save_model(  # 保存模型
            self.get_model(),  # 模型
            tensorizer_config=tensorizer_config,  # Tensorizer 配置
            model_config=self.model_config,  # 模型配置
        )

    def _check_weight_transfer_engine(self) -> None:  # 校验权重传输引擎已配置
        if self.weight_transfer_engine is None:  # 未配置
            raise RuntimeError(  # 报错
                "Weight transfer not configured. "  # 未配置提示
                "Please set weight_transfer_config to enable weight transfer."  # 启用方法
            )

    def init_weight_transfer_engine(self, init_info: dict) -> None:  # 初始化权重传输机制
        """
        Initialize weight transfer mechanism.
        For NCCL backend, this creates a process group with the trainer.

        Args:
            init_info: Dictionary containing backend-specific initialization info

        初始化权重传输机制。NCCL 后端会与训练端创建进程组。
        init_info：包含后端特定初始化信息的字典。
        """
        self._check_weight_transfer_engine()  # 校验引擎存在
        assert self.weight_transfer_engine is not None  # 类型收窄
        # Parse dict into backend-specific typed dataclass
        # 将字典解析为后端特定的类型化数据类
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)  # 解析初始化信息
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)  # 初始化传输引擎

    def start_weight_update(self) -> None:  # 开始权重更新会话（主模型）
        """
        Start a new weight update session.

        Delegates engine-specific preparation (e.g. layerwise reload setup) to
        the configured weight transfer engine. The worker only tracks that a
        session is active.

        开始新的权重更新会话。引擎相关准备委托给传输引擎，
        worker 仅跟踪会话是否活跃。
        """
        with set_current_vllm_config(self.vllm_config):  # 配置上下文
            self._start_weight_update()  # 调用内部实现

    def start_draft_weight_update(self) -> None:  # 开始权重更新会话（草稿模型）
        """
        Like start_weight_update, but retargets the engine at the speculative
        draft model for this session.
        类似 start_weight_update，但本次会话目标是投机解码草稿模型。
        """
        with set_current_vllm_config(self.vllm_config):  # 配置上下文
            self._start_weight_update(is_draft=True)  # 草稿模式

    def _start_weight_update(self, is_draft: bool = False) -> None:  # 内部：启动权重更新
        self._check_weight_transfer_engine()  # 校验引擎存在
        assert self.weight_transfer_engine is not None  # 类型收窄

        if is_draft and not self.weight_transfer_engine.supports_draft_weight_update:  # 草稿但引擎不支持
            raise RuntimeError(  # 报错
                f"{type(self.weight_transfer_engine).__name__} does not support "  # 引擎类名
                "draft model weight updates."  # 不支持草稿更新
            )

        if self._weight_update_active:  # 已有活跃会话
            raise RuntimeError(  # 报错
                "start_weight_update called while a weight update is already "  # 已有会话进行中
                "active. Call finish_weight_update first."  # 需先结束
            )

        try:
            if is_draft:  # 草稿模式
                self._set_draft_weight_update_target()  # 设置目标为草稿模型
            self.weight_transfer_engine.start_weight_update()  # 引擎启动会话
        except BaseException:  # 异常时回滚
            self.weight_transfer_engine.reset_weight_update_target()  # 重置目标
            raise  # 重新抛出
        self._weight_update_active = True  # 标记会话活跃
        self._weight_update_is_draft = is_draft  # 记录是否草稿

    def update_weights(self, update_info: dict) -> None:  # 接收一个权重更新分块
        """
        Receive one weight update chunk from the trainer.

        start_weight_update must be called before update_weights and
        finish_weight_update must be called after all chunks have been sent.
        Every chunk loads into whichever model the session's start_weight_update
        / start_draft_weight_update call selected.

        Args:
            update_info: Dictionary containing backend-specific update info

        从训练端接收一个权重更新分块。必须先调用 start_weight_update，
        全部分块发送后调用 finish_weight_update。
        分块加载到会话开始时选定的模型。
        update_info：后端特定的更新信息字典。
        """
        self._check_weight_transfer_engine()  # 校验引擎存在
        assert self.weight_transfer_engine is not None  # 类型收窄

        if not self._weight_update_active:  # 无活跃会话
            raise RuntimeError(  # 报错
                "start_weight_update must be called before update_weights."  # 需先启动会话
            )

        with set_current_vllm_config(self.vllm_config):  # 配置上下文
            try:
                self.weight_transfer_engine.update_weights(update_info)  # 引擎应用分块
            except BaseException:  # 异常时终止会话
                self._weight_update_active = False  # 取消活跃标记
                self.weight_transfer_engine.reset_weight_update_target()  # 重置目标
                raise  # 重新抛出

    def finish_weight_update(self) -> None:  # 结束权重更新会话
        """Finish the current weight update session.
        结束当前权重更新会话。"""
        self._check_weight_transfer_engine()  # 校验引擎存在
        assert self.weight_transfer_engine is not None  # 类型收窄

        if not self._weight_update_active:  # 无活跃会话
            raise RuntimeError(  # 报错
                "finish_weight_update called without a matching start_weight_update."  # 未启动会话
            )

        with set_current_vllm_config(self.vllm_config):  # 配置上下文
            self.weight_transfer_engine.finish_weight_update()  # 引擎结束会话
            self.weight_transfer_engine.reset_weight_update_target()  # 重置更新目标
            self._weight_update_active = False  # 取消活跃标记

        # Weight transfer bypasses GPUModelRunner.reload_weights().
        # 权重传输绕过了 reload_weights，需手动重置 LoRA 状态
        if not self._weight_update_is_draft:  # 非草稿会话
            self.model_runner.reset_lora_state()  # 重置 LoRA 状态

    def shutdown(self) -> None:  # 关闭 worker 并释放资源
        gc.unfreeze()  # 解冻 GC 堆

        # has_kv_transfer_group can be None during interpreter shutdown.
        # 解释器关闭期间相关函数可能为 None，需判空
        if ensure_kv_transfer_shutdown is not None:  # 函数可用
            ensure_kv_transfer_shutdown()  # 关闭 KV 传输
        if ensure_ec_transfer_shutdown is not None:  # 函数可用
            ensure_ec_transfer_shutdown()  # 关闭 EC 传输
        if self.profiler is not None:  # 剖析器存在
            self.profiler.shutdown()  # 关闭剖析器

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):  # 引擎存在
            weight_transfer_engine.shutdown()  # 关闭引擎

        # Release GPU resources held by the model runner so that memory
        # can be reclaimed when running in-process
        # 释放 model runner 持有的 GPU 资源，便于进程内运行时回收内存
        if model_runner := getattr(self, "model_runner", None):  # runner 存在
            model_runner.shutdown()  # 关闭 runner

        # Release kept-alive cumem pools while the pluggable allocator wrappers
        # and callbacks are still alive, so MemPool teardown is not deferred to
        # interpreter finalization (pytorch/pytorch#145168).
        # 在分配器包装与回调仍存活时释放 CuMem 池，
        # 避免内存池销毁被推迟到解释器终结阶段（pytorch/pytorch#145168）
        if current_platform.is_cuda_alike():  # CUDA 类平台
            from vllm.device_allocator.cumem import CuMemAllocator  # 延迟导入

            if CuMemAllocator.instance is not None:  # 实例存在
                CuMemAllocator.instance.release_pools()  # 释放内存池

    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):  # 执行弹性 EP 扩缩容操作
        return self.elastic_ep_executor.execute(execute_method, *args, **kwargs)  # 委托 EP 执行器


def init_worker_distributed_environment(  # 初始化 worker 的分布式环境
    vllm_config: VllmConfig,  # 全局配置
    rank: int,  # 全局 rank
    distributed_init_method: str | None = None,  # 初始化方法（默认 env://）
    local_rank: int = -1,  # 本地 rank
    backend: str = "nccl",  # 通信后端
) -> None:  # 无返回值
    """Initialize the distributed environment.
    初始化分布式环境。"""
    parallel_config = vllm_config.parallel_config  # 并行配置
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance  # 延迟导入批不变性初始化

    init_batch_invariance()  # 初始化批不变性（保证不同批大小下结果一致）

    override_envs_for_eplb(  # 按需覆盖 EPLB 相关环境变量
        parallel_config,  # 并行配置
        moe_backend=getattr(vllm_config.kernel_config, "moe_backend", None),  # MoE 后端
    )
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)  # 配置自定义 allreduce 开关

    init_method = distributed_init_method or "env://"  # 默认使用环境变量初始化

    timeout = None  # 超时（默认无）
    if parallel_config.distributed_timeout_seconds is not None:  # 配置了超时
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)  # 转为 timedelta

    init_distributed_environment(  # 初始化 torch.distributed 进程组
        parallel_config.world_size,  # 世界大小
        rank,  # rank
        init_method,  # 初始化方法
        local_rank,  # 本地 rank
        backend,  # 后端
        timeout,  # 超时
    )

    ensure_model_parallel_initialized(  # 初始化/等待模型并行组
        parallel_config.tensor_parallel_size,  # TP 大小
        parallel_config.pipeline_parallel_size,  # PP 大小
        parallel_config.prefill_context_parallel_size,  # 预填充上下文并行大小
        parallel_config.decode_context_parallel_size,  # 解码上下文并行大小
    )

    # Init ec connector here before KV caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    # 在 KV cache 初始化前初始化 EC 连接器；
    # 注意：EPD 分离模式下纯编码器实例不初始化 KV cache
    ensure_ec_transfer_initialized(vllm_config)  # 初始化 EC 传输
