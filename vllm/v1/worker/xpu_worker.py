# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# XPU(Intel 加速卡)Worker:继承 GPU Worker,适配 Intel 平台。负责 XPU 设备
# 初始化(含 DP 下的 local_rank 偏移)、oneCCL 分布式环境初始化、内存快照、
# workspace 管理器与 XPU Model Runner 的构造,以及 profile 与关闭流程。

# 导入 gc 模块,用于在内存快照前主动触发垃圾回收。
import gc
# 导入 os 模块,用于读写 oneCCL/分布式相关环境变量。
import os

# 导入 PyTorch,用于 XPU 设备 API 与分布式通信。
import torch

# 导入 VllmConfig,Worker 构造时接收完整配置。
from vllm.config import VllmConfig
# 导入日志初始化函数,用于创建模块日志记录器。
from vllm.logger import init_logger
# 导入 current_platform,用于查询平台类型(XPU)。
from vllm.platforms import current_platform
# 导入 TorchProfilerWrapper,用于 XPU profile 追踪。
from vllm.profiler.wrapper import TorchProfilerWrapper
# 导入内存快照与大小格式化工具。
from vllm.utils.mem_utils import MemorySnapshot, format_gib
# 导入随机种子设置函数,保证分布式下各 rank 可复现。
from vllm.utils.torch_utils import set_random_seed
# 导入上报使用统计信息的工具(可选遥测)。
from vllm.v1.utils import report_usage_stats
# 导入 GPU Worker 基类与分布式环境初始化函数。
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
# 导入 workspace 管理器初始化函数。
from vllm.v1.worker.workspace import init_workspace_manager
# 导入 XPU 的 V1/V2 模型 runner。
from vllm.v1.worker.xpu_model_runner import XPUModelRunner, XPUModelRunnerV2

# 导入请求内存计算工具(相对路径导入同目录 utils 模块)。
from .utils import request_memory

# 创建本模块的日志记录器。
logger = init_logger(__name__)


class XPUWorker(Worker):
    """A XPU worker class."""
    # XPU Worker 类:适配 Intel 加速卡的 Worker 实现。

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # 初始化 XPU Worker。
        # 参数:
        #   vllm_config: 完整的 vLLM 配置。
        #   local_rank: 本机设备索引。
        #   rank: 全局 rank。
        #   distributed_init_method: 分布式初始化方式。
        #   is_driver_worker: 是否承担 driver 职责。
        # 调用父类(GPU Worker)完成通用组件初始化。
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        # 取设备配置用于后续校验。
        device_config = self.device_config
        # 断言设备类型必须是 xpu。
        assert device_config.device_type == "xpu"
        # 断言当前运行平台确实是 XPU。
        assert current_platform.is_xpu()

    def init_device(self):
        # 初始化 XPU 设备:调整 local_rank、设置设备、初始化分布式环境、
        # 建立内存快照、初始化 workspace 与模型 runner。
        # In DP mode, XPU workers see all visible devices.
        # Offset local_rank by the local DP shard.
        # 说明:在 DP 模式下,每个 XPU worker 可见全部设备,
        # 需要按本地 DP 分片偏移 local_rank 以选中自己的设备。
        # 取并行配置引用。
        parallel_config = self.parallel_config
        # 若非 ray/external_launcher 后端,且 DP 后端不是 ray,且单节点单 DP 组:
        if (
            parallel_config.distributed_executor_backend
            not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.nnodes_within_dp == 1
        ):
            # 优先取本地 DP rank。
            dp_local_rank = parallel_config.data_parallel_rank_local
            # 本地 DP rank 不存在时退回全局 DP 索引。
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            # 计算 TP×PP 世界大小,用于把 DP rank 映射到物理设备。
            tp_pp_world_size = (
                parallel_config.pipeline_parallel_size
                * parallel_config.tensor_parallel_size
            )
            # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
            self.local_rank += dp_local_rank * tp_pp_world_size

            # 查询可见设备总数。
            visible_device_count = torch.accelerator.device_count()
            # 断言调整后的 local_rank 在可见设备范围内。
            assert self.local_rank < visible_device_count, (
                f"DP adjusted local rank {self.local_rank} is out of bounds. "
            )
            # 断言本机 world size 不超过可见设备数。
            assert parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )

        # 读取配置中的设备。
        device = self.device_config.device
        # 若设备是 xpu 类型的 torch.device 且当前平台是 XPU:
        if (
            isinstance(device, torch.device)
            and device.type == "xpu"
            and current_platform.is_xpu()
        ):
            # 构造当前 worker 的 XPU 设备对象。
            self.device = torch.device(f"xpu:{self.local_rank}")
            # 将 accelerator 的当前设备设置为该 XPU 设备。
            torch.accelerator.set_device_index(self.device)
            # 检查当前平台是否支持模型所需的 dtype(如 bf16)。
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            # 清空 XPU 缓存,释放预热阶段的残留显存。
            torch.accelerator.empty_cache()
            # 记录该 XPU 设备的总显存(用作内存规划基线)。
            self.init_gpu_memory = torch.xpu.get_device_properties(
                self.local_rank
            ).total_memory
        else:
            # 设备类型不支持时抛出运行时错误。
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")

        # 读取 oneCCL 传输协议环境变量(默认 "ofi")。
        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        # 读取本机 world size 环境变量(默认使用并行配置的 world_size)。
        ENV_LOCAL_WORLD_SIZE = os.getenv(
            "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        )
        # 设置 oneCCL 传输协议环境变量。
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        # 设置本机 world size 环境变量。
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        # 设置本机 rank 环境变量(供 oneCCL/分布式后端读取)。
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        # 初始化分布式环境(TP/PP/CP 进程组,后端为 XPU 的 dist_backend)。
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # global all_reduce needed for overall oneccl warm up
        # 执行一次全局 all_reduce,用于整个 oneCCL 的预热(建立通信上下文)。
        if torch.distributed.is_xccl_available():
            # 在 XPU 上对全 0 张量执行一次 all_reduce,初始化集合通信。
            torch.distributed.all_reduce(torch.zeros(1).xpu())

        # 若使用 V2 模型 runner:
        if self.use_v2_model_runner:
            # 记录“正在使用 V2 Model Runner”日志(仅一次)。
            logger.info_once("Using V2 Model Runner")

        # Set random seed.
        # 设置随机种子,保证后续采样结果可复现。
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        # 分布式/NCCL 初始化完成后,先做 GC 与清缓存,再拍内存快照。
        gc.collect()
        # 再次清空 XPU 缓存。
        torch.accelerator.empty_cache()

        # take current memory snapshot
        # 记录当前内存快照,用于后续 profiling 与 KV cache 内存规划。
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        # 根据内存快照与 cache 配置计算请求内存(目标占用)。
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        # 记录初始化内存快照的调试日志。
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        # 记录请求内存大小的调试日志。
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        # Initialize workspace manager
        # 初始化 workspace 管理器:DBO 启用时使用 2 个微批槽位,否则 1 个。
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        # 构造模型 runner:按 use_v2_model_runner 选择 V2 或 V1 XPU runner。
        model_runner = XPUModelRunnerV2 if self.use_v2_model_runner else XPUModelRunner
        self.model_runner = model_runner(  # type: ignore
            self.vllm_config, self.device
        )

        # 仅 rank 0 上报使用统计信息(如启用了遥测)。
        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            # 若启用了使用统计,则收集并上报相关信息。
            report_usage_stats(self.vllm_config)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # 启动/停止 profiler(XPU 侧,懒创建包装器)。
        # 参数:
        #   is_start: True 表示启动,False 表示停止。
        #   profile_prefix: trace 文件名前缀。
        # 未启用 profiling 配置时抛错。
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        # 启动且尚未创建 profiler 实例时懒创建:
        if is_start and self.profiler is None:
            # 获取 worker 的 rank 后缀,用于生成唯一的 trace 名。
            from vllm.distributed.utils import get_worker_rank_suffix

            # 计算全局 rank 对应的后缀。
            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
            # 组合 trace 名:带前缀则前缀_后缀,否则仅后缀。
            trace_name = (
                f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix
            )

            # 创建 Torch profiler 包装器(XPU 平台的活动为 CPU+XPU)。
            self.profiler = TorchProfilerWrapper(
                self.profiler_config,
                worker_name=trace_name,
                local_rank=self.local_rank,
                activities=["CPU", "XPU"],
            )
            # 记录 profiler 启动调试日志。
            logger.debug("Starting torch profiler with trace name: %s", trace_name)

        # 调用父类完成 start/stop 实际逻辑。
        super().profile(is_start=is_start, profile_prefix=profile_prefix)

    def shutdown(self) -> None:
        # 关闭 XPU Worker:记录清理开始日志,释放父类资源,再释放 XPU 显存池。
        # 记录关闭开始的日志(含 rank 信息)。
        logger.info(
            "XPUWorker shutdown: cleaning up (rank=%d, local_rank=%d)",
            self.rank,
            self.local_rank,
        )
        # 调用父类清理(模型 runner、传输、profiler 等)。
        super().shutdown()
        # 导入 XPU 显存分配器,用于释放显存池。
        from vllm.device_allocator.xpumem import XpuMemAllocator

        # 若 XPU 显存分配器实例存在:
        if XpuMemAllocator.instance is not None:
            # 释放所有 XPU 显存池。
            XpuMemAllocator.instance.release_pools()
        # 记录关闭完成的日志。
        logger.info(
            "XPUWorker shutdown: done (rank=%d, local_rank=%d)",
            self.rank,
            self.local_rank,
        )