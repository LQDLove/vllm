# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# CPU Worker 实现。
# 继承 GPU Worker,基于 CPU 内存(而非 GPU 显存)规划 KV cache,
# 绑定 NUMA 内存节点,并禁用不适用于 CPU 的设施(sleep、自定义 all-reduce 等)。

# Must be imported firstly
# 必须最先导入:该模块会替换 torch.cuda 等为 CPU 占位实现。
import vllm.v1.worker.cpu.shm  # noqa # isort: skip

# 导入 math,用于内存计算向上取整。
import math
# 导入 os,用于读写环境变量(LD_PRELOAD、VLLM_DIST_IDENT)。。
import os
# 导入 sys,用于平台判断(Linux)。
import sys
# 导入 Any 类型,用于 profiler 标注。
from typing import Any

# 导入 psutil,用于获取当前进程 RSS 内存。
import psutil
# 导入 PyTorch,用于设备与张量操作。
import torch

# 导入 VllmConfig 配置类。
from vllm.config import VllmConfig
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 CPU 架构枚举与当前平台。
from vllm.platforms import CpuArchEnum, current_platform
# 导入 Torch profiler 包装器。
from vllm.profiler.wrapper import TorchProfilerWrapper
# 导入 CPU 资源工具:CPU 列表、NUMA 内存节点信息与可见内存节点。
from vllm.utils.cpu_resource_utils import (
    get_allowed_cpu_list,
    get_memory_node_info,
    get_visible_memory_node,
)
# 导入内存单位格式化工具。
from vllm.utils.mem_utils import format_gib
# 导入随机种子设置函数。
from vllm.utils.torch_utils import set_random_seed
# 导入 V1 CPU 模型 runner。
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
# 导入 GPU Worker 基类与分布式环境初始化函数。
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
# 导入编译耗时元组类型。
from vllm.v1.worker.worker_base import CompilationTimes

# 创建本模块的日志记录器。
logger = init_logger(__name__)


class CPUWorker(Worker):
    # CPU Worker 类:以 GPU Worker 为基类,适配 CPU 平台的内存与线程管理。

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # TODO: use numactl for process setup
        # TODO: optimize for `interleaved` policy
        # 说明:计划使用 numactl 进行进程设置;并优化 `interleaved` 内存策略。
        # Bind memory node
        # 绑定内存节点:获取可见内存节点列表。
        allowed_memory_nodes = get_visible_memory_node()
        # 获取允许的 CPU 核心列表。
        allowed_cpu_list = get_allowed_cpu_list()
        # 取第一个 CPU 核心作为代表。
        cpu_core = allowed_cpu_list[0]

        # TODO: some CI hosts are not correctly set, change to assertion after fix
        # 说明:部分 CI 主机配置不正确,修复后改为断言。
        # 若该核心的 NUMA 节点不在可见内存节点中:
        if cpu_core.numa_node not in allowed_memory_nodes:
            # 记录警告:节点不在可用内存节点列表中。
            logger.warning(
                "Node %s is not in available memory nodes %s.",
                cpu_core.numa_node,
                allowed_memory_nodes,
            )

        # On s390x, numa_node may be a synthetic book ID that doesn't
        # correspond to a real memory node. Fall back to first visible node.
        # 说明:s390x 上 numa_node 可能是合成 book ID,不对应真实内存节点,
        # 回退到第一个可见节点。
        if cpu_core.numa_node in allowed_memory_nodes:
            # 使用该核心的 NUMA 节点。
            memory_node = cpu_core.numa_node
        else:
            # 记录警告并回退。
            logger.warning(
                "CPU group key %s is not a valid memory node. "
                "Falling back to memory node %s.",
                cpu_core.numa_node,
                allowed_memory_nodes[0],
            )
            # 使用第一个可见内存节点。
            memory_node = allowed_memory_nodes[0]

        # 用所选内存节点初始化 CPU 内存环境(C++ 侧)。
        torch.ops._C.init_cpu_memory_env([memory_node])

        # 获取该内存节点的信息。
        memory_status = get_memory_node_info(memory_node)
        # 内存利用率(此处实为 CPU 内存占比)。
        memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        # 计算请求的 CPU 内存 = 总内存 × 利用率(向上取整)。
        self.requested_cpu_memory = math.ceil(
            memory_status.total_memory * memory_fraction
        )
        # 取当前可用内存。
        available_memory = memory_status.available_memory

        # 若未显式指定 KV cache 内存,且请求内存超过可用内存:
        if (
            vllm_config.cache_config.kv_cache_memory_bytes is None
            and self.requested_cpu_memory > available_memory
        ):
            # 抛出错误:可用内存不足。
            raise ValueError(
                f"Available memory on node {cpu_core.numa_node} "
                f"({format_gib(available_memory)}/"
                f"{format_gib(memory_status.total_memory)} GiB) on startup "
                f"is less than desired CPU memory utilization "
                f"({vllm_config.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_cpu_memory)} GiB). "
                "On the CPU backend, the `--gpu-memory-utilization` flag "
                "controls the fraction of CPU memory reserved (despite its "
                "name). To resolve: decrease `--gpu-memory-utilization` "
                "(e.g. `--gpu-memory-utilization 0.5`) "
                "or reduce CPU memory used by other processes."
            )

        # 调用父类(GPU Worker)完成通用初始化。
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # CPU 后端禁用自定义 all-reduce(使用默认 gloo 路径)。
        self.parallel_config.disable_custom_all_reduce = True

        # Torch profiler. Enabled and configured through profiler_config.
        # Torch profiler:由 profiler_config 启用并配置。
        # profiler 实例初始为 None。
        self.profiler: Any | None = None
        # 取 profiler 配置。
        profiler_config = vllm_config.profiler_config
        # 若启用了 torch profiler:
        if profiler_config.profiler == "torch":
            # 构造 worker trace 名。
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            # 创建 CPU 侧 Torch profiler 包装器(活动仅 CPU)。
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )

    def init_device(self):
        # 初始化 CPU 设备:设置 device、检查预加载库、初始化分布式环境与模型 runner。
        # 设备固定为 CPU。
        self.device = torch.device("cpu")

        # Check whether critical libraries are loaded
        # 检查关键库是否已通过 LD_PRELOAD 预加载:
        def check_preloaded_libs(name: str) -> bool:
            # 读取 LD_PRELOAD 环境变量。
            ld_preload_list = os.environ.get("LD_PRELOAD", "")
            # 若库名不在预加载列表中:
            if name not in ld_preload_list:
                # 记录警告:建议按文档设置 LD_PRELOAD。
                logger.warning(
                    "%s is not found in LD_PRELOAD. "
                    "For best performance, please follow the section "
                    "`set LD_PRELOAD` in "
                    "https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ "
                    "to setup required pre-loaded libraries.",
                    name,
                )
                # 返回 False:未预加载。
                return False
            # 已预加载,返回 True。
            return True

        # 若运行在 Linux 平台:
        if sys.platform.startswith("linux"):
            # 检查 tcmalloc 是否预加载(内存分配性能)。
            check_preloaded_libs("libtcmalloc")
            # 若 CPU 架构为 x86:
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                # 检查 Intel OpenMP 是否预加载。
                iomp_loaded = check_preloaded_libs("libiomp")
                # 若未加载 iomp 且配置了规范化解码:
                if not iomp_loaded and self.vllm_config.speculative_config is not None:
                    # 记录警告:缺少 Intel OpenMP 将显著降低规范化解码性能。
                    logger.warning(
                        "Speculative decoding on CPU without Intel OpenMP in "
                        "LD_PRELOAD will cause significant performance loss. "
                        "Please follow the section `set LD_PRELOAD` in "
                        "https://docs.vllm.ai/en/latest/getting_started/"
                        "installation/cpu/ "
                        "to setup libiomp5.",
                    )

        def skip_set_num_threads(x: int):
            # 屏蔽 torch.set_num_threads:线程绑定后不允许修改线程数。
            logger.warning(
                "CPU backend doesn't allow to use "
                "`torch.set_num_threads` after the thread binding, skip it."
            )

        # 替换 torch.set_num_threads 为屏蔽版本。
        torch.set_num_threads = skip_set_num_threads

        # Note: unique identifier for creating allreduce shared memory
        # 说明:为创建 all-reduce 共享内存设置唯一标识符。
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]
        # Initialize the distributed environment.
        # 初始化分布式环境(TP/PP 等进程组)。
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        # Set random seed.
        # 设置随机种子,保证可复现。
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        # 构造模型 runner:按 use_v2_model_runner 选择 V2 或 V1 CPU runner。
        if self.use_v2_model_runner:
            # 导入 V2 CPU 模型 runner。
            from vllm.v1.worker.cpu.model_runner import (
                CPUModelRunner as CPUModelRunnerV2,
            )

            # 使用 V2 CPU 模型 runner。
            self.model_runner: CPUModelRunner = CPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            # 使用 V1 CPU 模型 runner。
            self.model_runner = CPUModelRunner(self.vllm_config, torch.device("cpu"))

    def sleep(self, level: int = 1) -> None:
        # CPU 后端不支持睡眠模式,记录警告并忽略。
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        # CPU 后端不支持唤醒模式,记录警告并忽略。
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        # 确定可用于 KV cache 的 CPU 内存大小。
        # 先预热模型(触发编译)。
        self.model_runner.warming_up_model()

        # 获取允许的 CPU 核心列表。
        allowed_cpu_list = get_allowed_cpu_list()
        # 取第一个核心。
        cpu_core = allowed_cpu_list[0]

        # 获取该核心 NUMA 节点的内存信息。
        memory_status = get_memory_node_info(cpu_core.numa_node)
        # 取可用内存。
        available_memory = memory_status.available_memory
        # 取显式指定的 KV cache 内存(可能为 None)。
        explicit_kv_cache_size = self.cache_config.kv_cache_memory_bytes

        # 初始化 KV cache 大小与提示消息。
        kv_cache_size = None
        msg = None
        # 若用户显式指定了 KV cache 内存:
        if explicit_kv_cache_size is not None:
            # 若指定大小超过可用内存:
            if explicit_kv_cache_size > available_memory:
                # 抛出错误。
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(explicit_kv_cache_size)} GiB). "
                    "Decrease --kv-cache-memory-bytes, VLLM_CPU_KVCACHE_SPACE, "
                    "or reduce CPU memory used by other processes."
                )
            # 使用显式指定的大小。
            kv_cache_size = explicit_kv_cache_size
            # 构造提示消息。
            msg = (
                f"Explicitly set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}."
            )
        # 否则自动计算:
        else:
            # 获取当前进程已消耗的 RSS 内存。
            consumed_memory = psutil.Process(os.getpid()).memory_info().rss
            # 可用于 KV 的内存 = 请求内存 - 已消耗内存。
            requested_memory_for_kv = int(self.requested_cpu_memory - consumed_memory)
            # 若可用 KV 内存非正或超过可用内存:
            if (
                requested_memory_for_kv <= 0
                or requested_memory_for_kv > available_memory
            ):
                # 抛出错误。
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(requested_memory_for_kv)}/"
                    f"{format_gib(self.requested_cpu_memory)} GiB). "
                    "Reduce CPU memory used by other processes."
                )
            # 使用计算得到的 KV 内存。
            kv_cache_size = requested_memory_for_kv
            # 构造提示消息。
            msg = (
                f"Auto set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}, with "
                f"{format_gib(self.requested_cpu_memory)} GiB requested memory"
                f" for the worker. {format_gib(consumed_memory)} GiB"
                f" memory was consumed by non-kv usages."
            )

        # 记录 KV cache 内存分配日志。
        logger.info(msg)

        # 返回 KV cache 内存大小。
        return kv_cache_size

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Note: the model has been compiled in determine_available_memory(),
        # Only compile here for models without kv cache
        # 说明:模型已在 determine_available_memory() 中编译;
        # 此处仅为无 KV cache 的模型编译。
        if len(self.model_runner.kv_caches) == 0:
            # 预热模型。
            self.model_runner.warming_up_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        # 重置随机种子,确保随机状态不受模型初始化与 profiling 影响。
        set_random_seed(self.model_config.seed)
        # 返回编译耗时。
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # 启动/停止 CPU profiler。
        # 若未创建 profiler,抛错。
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        # 启动时调用 start。
        if is_start:
            self.profiler.start()
        # 停止时调用 stop。
        else:
            self.profiler.stop()