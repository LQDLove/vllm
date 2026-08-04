# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Worker 抽象接口与进程包装器。
# - WorkerBase: 定义所有硬件 Worker 的统一生命周期与推理接口,支持两阶段执行。
# - WorkerWrapperBase: 代表 executor/engine 中的一个进程,负责懒初始化 Worker、
#   动态注入 worker_extension_cls 扩展、管理多模态共享缓存。

# 导入 Callable 类型,用于标注 apply_model 等接受函数参数的接口。
from collections.abc import Callable
# 导入类型工具:TYPE_CHECKING 用于类型检查时的条件导入;Any 任意类型;
# NamedTuple 用于定义编译耗时元组;TypeVar 用于泛型。
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

# 导入 PyTorch 主模块(用于标注设备与张量)。
import torch
# 导入 nn 模块,模型模块类型标注。
import torch.nn as nn

# 导入 vllm.ir(编译器 IR 相关,设置 IR op 优先级与 torch-wrap)。
import vllm.ir
# 导入 VllmConfig(完整配置)与 set_current_vllm_config(配置上下文管理器)。
from vllm.config import VllmConfig, set_current_vllm_config
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 LoRARequest,用于 LoRA 管理接口。
from vllm.lora.request import LoRARequest
# 导入多模态注册表,用于创建 worker 侧的多模态共享缓存。
from vllm.multimodal import MULTIMODAL_REGISTRY
# 导入 tracing 的 instrument 装饰器,用于给 Worker init 加 span。
from vllm.tracing import instrument
# 导入按限定名解析对象的工具(worker_cls 字符串 -> 类)。
from vllm.utils.import_utils import resolve_obj_by_qualname
# 导入环境变量更新工具,用于更新子进程环境变量。
from vllm.utils.system_utils import update_environment_variables
# 导入 KV cache 规范类型(KVCacheSpec)。
from vllm.v1.kv_cache_interface import KVCacheSpec

# 仅类型检查时导入调度输出与模型输出类型(避免运行时循环依赖)。
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    # 运行时用 object 占位,保持 API 签名稳定。
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

# 创建本模块的日志记录器。
logger = init_logger(__name__)

# 泛型类型变量,用于 apply_model 的返回类型标注。
_R = TypeVar("_R")


class CompilationTimes(NamedTuple):
    # 编译耗时统计元组。
    # language_model: 语言模型编译/预热耗时(秒)。
    language_model: float
    # encoder: 编码器编译/预热耗时(秒)。
    encoder: float


class WorkerBase:
    # Worker 抽象基类:将不同硬件(GPU/CPU/XPU/TPU)的实现与 vLLM 解耦,
    # 并抽象控制面通信(例如向其他 worker 传递请求元数据)。
    # 所有硬件 Worker 都须继承本类并实现其接口。
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        # 初始化 worker 的公共组件。
        # Args:
        #     vllm_config: 完整的 vLLM 配置。
        #     local_rank: 本机设备索引。
        #     rank: 全局 rank。
        #     distributed_init_method: 分布式初始化方式。
        #     is_driver_worker: 是否承担 driver 职责。
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        # 保存完整配置。
        self.vllm_config = vllm_config
        # 抽取模型配置。
        self.model_config = vllm_config.model_config
        # 抽取缓存配置。
        self.cache_config = vllm_config.cache_config
        # 抽取 LoRA 配置。
        self.lora_config = vllm_config.lora_config
        # 抽取加载配置。
        self.load_config = vllm_config.load_config
        # 抽取并行配置。
        self.parallel_config = vllm_config.parallel_config
        # 抽取调度配置。
        self.scheduler_config = vllm_config.scheduler_config
        # 抽取设备配置。
        self.device_config = vllm_config.device_config
        # 抽取规范化解码配置。
        self.speculative_config = vllm_config.speculative_config
        # 抽取可观测性配置。
        self.observability_config = vllm_config.observability_config
        # 抽取 KV 传输配置。
        self.kv_transfer_config = vllm_config.kv_transfer_config
        # 抽取编译配置。
        self.compilation_config = vllm_config.compilation_config

        # 延迟导入当前平台模块。
        from vllm.platforms import current_platform

        # 保存当前平台对象。
        self.current_platform = current_platform

        # 把 rank 写回并行配置,便于后续访问。
        self.parallel_config.rank = rank
        # 保存本机秩。
        self.local_rank = local_rank
        # 保存全局秩。
        self.rank = rank
        # 保存分布式初始化方式。
        self.distributed_init_method = distributed_init_method
        # 保存是否 driver worker 标志。
        self.is_driver_worker = is_driver_worker

        # Device and model state
        # 设备与模型状态初始为 None,由子类初始化。
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

        # IR op priority and torch-wrap state are constant for the worker's
        # lifetime.
        # IR op 优先级与 torch-wrap 状态在 worker 生命周期内恒定:
        # 设置默认的 IR op 优先级。
        vllm_config.kernel_config.ir_op_priority.set_default()
        # 设置默认的 torch wrap 启用状态。
        vllm.ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        # 获取 KV cache 实现规范(dict: layer -> KVCacheSpec)。
        """Get specifications for KV cache implementation."""
        # 子类实现。
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # 通过编译/预热准备模型执行。
        # Returns:
        #     编译耗时(language_model, encoder),单位为秒。
        """Prepare model for execution through compilation/warmup.

        Returns:
            Compilation times (language_model, encoder) in seconds.
        """
        # 子类实现。
        raise NotImplementedError

    def check_health(self) -> None:
        # 基础健康检查(可由设备特定实现覆盖)。
        """Basic health check (override for device-specific checks)."""
        return

    def init_device(self) -> None:
        # 初始化设备状态,例如加载模型或进行设备内存分配。
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        # 子类实现。
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        # 重置多模态缓存:若 model_runner 提供 reset_mm_cache 则调用之。
        # 获取 model_runner 上的 reset_mm_cache 方法(可能不存在)。
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        # 若该方法可调用:
        if callable(reset_fn):
            # 调用它重置多模态缓存。
            reset_fn()

    def get_model(self) -> nn.Module:
        # 获取本 worker 内的模型模块。
        # 子类实现。
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        # 对 worker 内的模型应用给定函数 fn,返回其结果。
        """Apply a function on the model inside this worker."""
        # 获取模型并调用 fn。
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        # 返回模型的分层结构视图(transformers 风格字符串)。
        """Return a transformers-style hierarchical view of the model."""
        # 导入模型检查格式化工具。
        from vllm.model_inspection import format_model_inspection

        # 生成并返回模型检查字符串。
        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # 将模型加载到目标设备。
        # Args:
        #     load_dummy_weights: 是否仅加载随机权重(用于测量内存)。
        """Load model onto target device."""
        # 子类实现。
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # 执行模型前向。若返回 None,则应紧随其后调用 sample_tokens 获取
        # ModelRunnerOutput。该设计可随结构化输出并行的重构而调整。
        # 子类实现。
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # 仅当 execute_model 返回 None 时需立即调用:基于语法约束进行采样,
        # 返回 ModelRunnerOutput。
        """Should be called immediately after execute_model iff it returned None."""
        # 子类实现。
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        # 返回单个 cache block 的大小(字节),用于推测解码。
        # 子类实现。
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 加载 LoRA 适配器,返回是否成功。
        # 子类实现。
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        # 卸载指定 id 的 LoRA,返回是否成功。
        # 子类实现。
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        # 固定 LoRA 防止被 LRU 逐出,返回是否成功。
        # 子类实现。
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        # 列出当前已加载的 LoRA id 集合。
        # 子类实现。
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        # (属性)从模型配置获取词汇表大小。
        """Get vocabulary size from model configuration."""
        # 委托给 model_config。
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        # 释放 worker 持有的资源(默认空实现)。
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    # 代表 executor/engine 中的一个进程,负责懒初始化真实 Worker 并管理其生命周期。
    # 先实例化包装器(记下 worker 模块/类名),调用 update_environment_variables 后,
    # 真正初始化发生在 init_worker。
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        # 以给定 vllm_config 与 rpc_rank 初始化包装器。
        # rpc_rank 为 worker 在 executor 中的 rank,多数与分布式组内 rank 相同;
        # 多 executor 协同(如 SPMD 离线推理 TP=2 启动两个引擎)时二者可能不同。
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        # 保存 rpc rank。
        self.rpc_rank: int = rpc_rank
        # 全局 rank 默认与 rpc rank 相同,可显式指定。
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        # 以下字段在 init_worker 调用后才赋值:
        # 底层 worker 实例。
        self.worker: WorkerBase
        # 完整 vLLM 配置。
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        # 关闭底层 worker(若已初始化)。
        if self.worker is not None:
            # 调用 worker 的 shutdown。
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        # 按本进程的 rpc_rank 更新对应的环境变量。
        # 取当前 rpc_rank 对应的环境变量字典。
        envs = envs_list[self.rpc_rank]
        # 应用环境变量更新。
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        # 真正初始化 worker:解析并注入公共逻辑,最后调用 worker 构造函数。
        # 参数 all_kwargs 按 rpc_rank 索引,取出本进程的参数。
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        # 取当前 rpc_rank 对应的构造参数。
        kwargs = all_kwargs[self.rpc_rank]

        # 从参数中取出 vllm_config。
        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        # 断言 vllm_config 必须提供。
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        # 保存 vllm_config。
        self.vllm_config = vllm_config

        # 为当前线程启用函数调用追踪。
        vllm_config.enable_trace_function_call_for_thread()

        # 加载通用插件。
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        # 取并行配置。
        parallel_config = vllm_config.parallel_config
        # worker_cls 以字符串(限定名)形式给出:
        if isinstance(parallel_config.worker_cls, str):
            # 按限定名解析出 worker 类。
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            # 已不再支持直接传类对象。
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        # 若配置了 worker 扩展类:
        if parallel_config.worker_extension_cls:
            # 解析扩展类。
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            # 记录将被扩展调用的方法名。
            extended_calls = []
            # 若扩展类还不是 worker 类的基类:
            if worker_extension_cls not in worker_class.__bases__:
                # 检查 worker 类与扩展类之间的属性冲突。
                for attr in dir(worker_extension_cls):
                    # 跳过魔法属性。
                    if attr.startswith("__"):
                        continue
                    # 断言扩展类的属性不在 worker 类中(避免覆盖)。
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    # 若属性可调用,记录为扩展方法。
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # 动态把扩展类加入 worker 类的基类(多重继承)。
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                # 记录注入日志。
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        # 取出分配的物理 GPU id(可能不存在)。
        assigned_physical_gpu_ids = kwargs.pop("assigned_physical_gpu_ids", None)
        # 若提供了 GPU 分配信息:
        if assigned_physical_gpu_ids is not None:
            # 写入并行配置。
            vllm_config.parallel_config.assigned_physical_gpu_ids = (
                assigned_physical_gpu_ids
            )

        # 取出共享 worker 锁(多模态共享缓存需要)。
        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        # 若未提供锁:
        if shared_worker_lock is None:
            # 构造缺失提示消息。
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            # 取多模态配置。
            mm_config = vllm_config.model_config.multimodal_config
            # 若配置了 shm 类型的多模态处理器缓存,则必须提供锁:
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                # 缺少锁属于错误,抛出 ValueError。
                raise ValueError(msg)
            else:
                # 其它情况仅警告一次。
                logger.warning_once(msg)

            # 无锁时多模态接收缓存为 None。
            self.mm_receiver_cache = None
        else:
            # 有锁:通过多模态注册表创建 worker 侧接收缓存。
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        # 在 vllm config 上下文中实例化 worker(使配置在初始化期间可访问)。
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            # 调用 worker 构造函数完成真正初始化。
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        # 按全局 rank 选取对应 kv_cache_config,并下发给底层 worker。
        # 取本进程全局 rank 对应的 KV cache 配置。
        kv_cache_config = kv_cache_configs[self.global_rank]
        # 断言 vllm_config 已设置。
        assert self.vllm_config is not None
        # 在 vllm config 上下文中转发给底层 worker。
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        # 转发到底层 worker 的 init_device(在 vllm config 上下文中执行)。
        # 断言 vllm_config 已设置。
        assert self.vllm_config is not None
        # 在 vllm config 上下文中调用底层 worker 的设备初始化。
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        # 属性代理:将未在包装器上定义的属性/方法转发给底层 worker。
        # 调用底层 worker 的对应属性。
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        # 将共享内存缓存中的多模态特征合并入新调度的请求(mm_features)。
        # 获取多模态接收缓存。
        mm_cache = self.mm_receiver_cache
        # 无缓存时直接返回。
        if mm_cache is None:
            return

        # 遍历本轮新调度的请求:
        for req_data in scheduler_output.scheduled_new_reqs:
            # 用缓存中的特征更新请求的多模态特征。
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # 先应用多模态缓存,再委托底层 worker 执行模型。
        self._apply_mm_cache(scheduler_output)

        # 转发给底层 worker 的 execute_model。
        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        # 清空共享缓存并转发给底层 worker 重置多模态缓存。
        # 获取多模态接收缓存。
        mm_receiver_cache = self.mm_receiver_cache
        # 若存在则清空。
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        # 转发给底层 worker 重置多模态缓存。
        self.worker.reset_mm_cache()