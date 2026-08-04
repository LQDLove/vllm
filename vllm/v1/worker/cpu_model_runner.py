# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# V1 CPU Model Runner:继承顶层 V1 GPUModelRunner,通过补丁把 CUDA API
# 替换为 CPU 占位实现,并把设备张量迁移到 CPU;无 Triton 时使用 CPU 的
# Triton-kernel 替代实现(slot mapping、规范化解码 EAGLE/DFlash、拒绝采样等)。

# 导入 sys 模块,用于按模块名查询已加载的模块(动态替换 kernel)。
import sys
# 导入 contextmanager 装饰器,用于定义 _torch_cuda_wrapper 等上下文管理器。
from contextlib import contextmanager
# 导入 Any 类型,用于 _postprocess_tensors 中泛化的对象属性处理。
from typing import Any

# 导入 PyTorch 主模块,用于张量操作与设备 API。
import torch
# 导入 nn 模块,用于标注模型类型。
import torch.nn as nn

# 导入 CPU 侧的 Triton kernel 工具(提供 CPU 版 kernel 实现)。
import vllm.utils.cpu_triton_utils as cpu_tl
# 导入 VllmConfig,模型 runner 构造时接收完整配置。
from vllm.config import VllmConfig
# 导入日志初始化函数,用于创建模块日志记录器。
from vllm.logger import init_logger
# 导入 get_model,用于加载模型。
from vllm.model_executor.model_loader import get_model
# 导入 instrument 装饰器,用于给关键方法加 span 追踪。
from vllm.tracing import instrument
# 导入 KV cache 接口:FullAttentionSpec 用于区分全注意力层,KVCacheConfig 配置。
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
# 导入 CpuGpuBuffer,用于把其 gpu 侧指向 cpu 侧。
from vllm.v1.utils import CpuGpuBuffer
# 导入 V1 GPUModelRunner 作为基类。
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# 创建本模块的日志记录器。
logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    # CPU 模型执行器:以 V1 GPUModelRunner 为基类,所有输入准备/采样的
    # 逻辑复用 GPU 实现,仅替换设备相关 API 与 kernel。

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # 初始化 CPU 模型 runner。
        # 参数:
        #   vllm_config: 完整的 vLLM 配置。
        #   device: 目标设备(必须为 CPU)。
        # avoid calling accelerator APIs for methods inherited from super class
        # 避免父类方法中调用 accelerator(CUDA)API:先把 accelerator 相关调用置为 no-op。
        _set_torch_accelerator_to_noop()

        # 在 torch.cuda 流/事件占位补丁上下文中调用父类初始化,
        # 使 GPU runner 代码可运行于无 CUDA 的 CPU 环境。
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # 断言设备确实是 CPU(否则说明调用方传入了错误设备)。
        assert device == torch.device("cpu")
        # Note: speculative decoding is now supported on CPU with C++ native impls
        # 注:CPU 现已通过 C++ 原生实现支持规范化解码。

        # CPU 后端禁用 CUDA Graph。
        self.use_cuda_graph = False
        # CPU 后端不启用级联注意力(cascade attention)。
        self.cascade_attn_enabled = False

        # 把父类初始化产生的设备(CPU/GPU)张量缓冲统一指到 CPU 侧。
        self._postprocess_tensors()
        # 配置 Triton 相关设置(无 Triton-CPU 时替换默认 kernel 为 C++ 实现)。
        self._postprocess_triton()

    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        # 说明:把对象上的设备张量替换为 CPU 张量(CpuGpuBuffer 的 gpu 指向 cpu)。
        # 定义内部辅助函数:若对象同时具有 cpu_attr_name 与 device_attr_name
        # 两个 Tensor 属性,则把 device 属性指向 cpu 属性的张量。
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            # 读取 CPU 属性对应的张量(可能不存在)。
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            # 读取设备属性对应的张量(可能不存在)。
            device_tensor = getattr(obj, device_attr_name, None)
            # 仅当两者都是 torch.Tensor 时才进行替换。
            if isinstance(cpu_tensor, torch.Tensor) and isinstance(
                device_tensor, torch.Tensor
            ):
                # 把设备属性设置为 CPU 张量(共享底层数据,免拷贝)。
                setattr(obj, device_attr_name, cpu_tensor)

        # 遍历 runner 的所有实例属性:
        for v in vars(self).values():
            # 若属性是 CpuGpuBuffer(同时持有 cpu/gpu 两个缓冲):
            if isinstance(v, CpuGpuBuffer):
                # 让 gpu 缓冲指向 cpu 缓冲(CPU 上二者同址)。
                v.gpu = v.cpu

        # 遍历 input_batch 的全部属性:
        for k, v in vars(self.input_batch).items():
            # 找形如 "xxx_cpu_tensor" 的 CPU 张量属性:
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                # 用该 CPU 张量替换掉对应的(设备)属性(去 "_cpu_tensor" 后缀)。
                replace_tensor(self.input_batch, k, k[:-11])

        # 遍历输入批处理中每个 KV cache 组的块表:
        for block_table in self.input_batch.block_table.block_tables:
            # 遍历块表的每个属性:
            for v in vars(block_table).values():
                # 若为 CpuGpuBuffer,同样把 gpu 缓冲指向 cpu 缓冲。
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu

    def _postprocess_triton(self) -> None:
        # 配置 CPU 的 Triton 相关实现。
        # 导入 Triton 可用性标志。
        from vllm.triton_utils import HAS_TRITON

        # 若环境支持 Triton(含 Triton-CPU 后端):
        if HAS_TRITON:
            # 记录日志:已可用 Triton-CPU,跳过 C++ monkey-patch。
            logger.info(
                "Triton-CPU backend is available; skipping C++ monkey-patches "
                "for Triton kernels."
            )
            # 直接返回,不再替换 kernel。
            return

        # 无 Triton 时,将 vLLM 默认的 Triton kernel 替换为 CPU(C++)实现:
        # 导入块表模块以替换 slot mapping kernel。
        import vllm.v1.worker.block_table

        # 把计算 slot mapping 的 Triton kernel 替换为 CPU 版。
        vllm.v1.worker.block_table._compute_slot_mapping_kernel = (
            cpu_tl.compute_slot_mapping_kernel
        )

        # Speculative decoding fallbacks
        # 规范化解码的 CPU 回退实现:
        # 导入拒绝采样器模块。
        import vllm.v1.sample.rejection_sampler
        # 导入规范化解码的 proposer 与工具模块。
        import vllm.v1.spec_decode.llm_base_proposer
        import vllm.v1.spec_decode.utils as spec_decode_utils

        # 替换 EAGLE prefill 输入准备的 kernel 为 CPU 版。
        vllm.v1.spec_decode.llm_base_proposer.eagle_prepare_inputs_padded_kernel = (
            cpu_tl.eagle_prepare_inputs_padded_kernel
        )
        # 替换 EAGLE 下一 token 准备的 kernel 为 CPU 版。
        vllm.v1.spec_decode.llm_base_proposer.eagle_prepare_next_token_padded_kernel = (
            cpu_tl.eagle_prepare_next_token_padded_kernel
        )
        # 替换 EAGLE 输入拷贝/展开 kernel 为 CPU 版。
        vllm.v1.spec_decode.llm_base_proposer.copy_and_expand_eagle_inputs_kernel = (
            cpu_tl.copy_and_expand_eagle_inputs_kernel
        )
        # 替换 DFlash 输入拷贝/展开 kernel 为 CPU 版(工具模块层面)。
        spec_decode_utils.copy_and_expand_dflash_inputs_kernel = (
            cpu_tl.copy_and_expand_dflash_inputs_kernel
        )
        # 查询 dflash 模块是否已加载。
        dflash_module = sys.modules.get("vllm.v1.spec_decode.dflash")
        # 若 dflash 模块已加载:
        if dflash_module is not None:
            # 记录要设置的 kernel 属性名。
            dflash_kernel_name = "copy_and_expand_dflash_inputs_kernel"
            # 在 dflash 模块上动态设置 CPU 版 kernel。
            setattr(
                dflash_module,
                dflash_kernel_name,
                cpu_tl.copy_and_expand_dflash_inputs_kernel,
            )
        # 替换 EAGLE step slot mapping 元数据 kernel 为 CPU 版。
        spec_decode_utils.eagle_step_slot_mapping_metadata_kernel = (
            cpu_tl.eagle_step_slot_mapping_metadata_kernel
        )
        # 替换拒绝采样的贪心采样 kernel 为 CPU 版。
        vllm.v1.sample.rejection_sampler.rejection_greedy_sample_kernel = (
            cpu_tl.rejection_greedy_sample_kernel
        )
        # 替换拒绝采样的随机采样 kernel 为 CPU 版。
        vllm.v1.sample.rejection_sampler.rejection_random_sample_kernel = (
            cpu_tl.rejection_random_sample_kernel
        )
        # 替换拒绝采样的 expand kernel 为 CPU 版。
        vllm.v1.sample.rejection_sampler.expand_kernel = cpu_tl.expand_kernel
        # 替换拒绝采样的恢复 token 采样 kernel 为 CPU 版。
        vllm.v1.sample.rejection_sampler.sample_recovered_tokens_kernel = (
            cpu_tl.sample_recovered_tokens_kernel
        )

        # 导入 Mamba 工具模块,替换其中的批量内存拷贝 kernel。
        import vllm.v1.worker.mamba_utils

        # 把 Mamba 的 batch_memcpy kernel 替换为 CPU 版。
        vllm.v1.worker.mamba_utils.batch_memcpy_kernel = cpu_tl.batch_memcpy_kernel

    @instrument(span_name="Loading (CPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        # 在 CPU 后端加载模型(带 span 追踪)。
        # 参数:
        #   load_dummy_weights: 是否加载随机权重(CPU 不支持)。
        # 若请求加载 dummy 权重:
        if load_dummy_weights:
            # 抛出异常:CPU 不支持加载 dummy 权重(弹性 EP 扩缩容所需)。
            raise ValueError(
                "Loading dummy weights (needed for elastic EP scale-up) "
                "Is not supported by the CPU Model Runner."
            )
        # 记录开始加载模型的日志。
        logger.info("Starting to load model %s...", self.model_config.model)
        # 通过模型加载器加载真实模型。
        self.model = get_model(vllm_config=self.vllm_config)

        # 若启用了 LoRA:
        if self.lora_config:
            # 将 LoRA 适配器集成到已加载模型中。
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        # 若存在 drafter(规范化解码的草稿模型):
        if hasattr(self, "drafter"):
            # 记录加载草稿模型的日志。
            logger.info_once("Loading drafter model...")
            # 加载草稿模型并关联到已加载的目标模型。
            self.drafter.load_model(self.model)

        # 配置 EAGLE-3 辅助隐藏状态输出层。
        self._setup_eagle3_aux_hidden_state_outputs()

    def get_model(self) -> nn.Module:
        # 返回模型模块实例。
        return self.model

    @instrument(span_name="Warmup (CPU)")
    def warming_up_model(self) -> None:
        # CPU 模型编译预热(带 span 追踪):为通用形状生成编译图,
        # 避免运行期首次前向的即时编译延迟。
        # 记录预热开始日志。
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        # 仅对通用(generic)形状生成编译图。
        # 在全局编译设置(MKLDNN/CPPGEMM 需冻结参数)上下文中执行 profile。
        with _set_global_compilation_settings(self.vllm_config):
            # 执行 dummy profile 前向,触发 torch.compile 编译。
            self.profile_run()
        # 记录预热完成日志。
        logger.info("Warming up done.")

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        # 初始化 CPU 侧 KV cache。
        # 参数:
        #   kv_cache_config: KV cache 分配与规格配置。
        #   is_profiling: 是否处于 profiling 阶段。
        # 先调用父类完成 KV cache 分配与绑定。
        super().initialize_kv_cache(kv_cache_config, is_profiling)

        # 若配置了规范化解码:
        if self.speculative_config:
            # 若使用 EAGLE 方法:
            if self.speculative_config.use_eagle():
                # 记录 EAGLE 草稿模型的 KV cache 已初始化。
                logger.info("EAGLE drafter KV cache initialized for CPU backend")
            # 若使用其它草稿模型:
            elif self.speculative_config.uses_draft_model():
                # 记录草稿模型的 KV cache 已初始化。
                logger.info("Draft model KV cache initialized for CPU backend")

    def _init_device_properties(self) -> None:
        # CPU 后端无需初始化设备属性(空实现)。
        pass

    def _sync_device(self) -> None:
        # CPU 后端无需显式设备同步(天然同步,空实现)。
        pass

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        # 清零指定的 KV cache 块,防止部分写入导致陈旧数据损坏。
        # Zero full-attention blocks to prevent stale data corruption on partial writes.
        # Encoder-only (runner-only) layers are not FullAttentionSpec, so the
        # spec filter below already excludes them; no runner-only skip needed.
        # 说明:仅清零全注意力(FullAttentionSpec)块;仅编码器层不是
        # FullAttentionSpec,下面按 spec 过滤即可排除,无需额外跳过。
        # 用集合记录已清零的块地址,避免同一块被重复清零。
        seen_ptrs: set[int] = set()
        # 遍历每个 KV cache 组:
        for group in self.kv_cache_config.kv_cache_groups:
            # 若不是全注意力规格(如 Mamba/编码器层),跳过该组。
            if not isinstance(group.kv_cache_spec, FullAttentionSpec):
                continue
            # 遍历该组包含的层:
            for layer_name in group.layer_names:
                # 从静态前向上下文获取该层上下文。
                ctx = self.compilation_config.static_forward_context.get(layer_name)
                # 上下文不存在时跳过。
                if ctx is None:
                    continue
                # 获取该层的 KV cache 张量。
                kv = ctx.kv_cache
                # 若 KV cache 不是 torch.Tensor,跳过。
                if not isinstance(kv, torch.Tensor):
                    continue
                # 若该张量的数据地址已处理过(多条层共享同一 KV 块),跳过。
                if kv.data_ptr() in seen_ptrs:
                    continue
                # 记录已处理的数据地址。
                seen_ptrs.add(kv.data_ptr())
                # 遍历待清零的块 id:
                for block_id in block_ids:
                    # 将该块清零(原位置零)。
                    kv[block_id].zero_()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        """CPU-safe version: direct tolist() without CUDA events."""
        # 采样 token 转列表的 CPU 安全版本:直接 tolist(),无需 CUDA 事件。
        # 直接调用 tolist() 把张量转为嵌套列表。
        return sampled_token_ids.tolist()


@contextmanager
def _torch_cuda_wrapper():
    # 定义 torch.cuda 流/事件的 CPU 占位上下文管理器,使 GPU runner 代码
    # 可在纯 CPU 环境运行;with 语句中替换,退出时恢复。
    # 定义 CUDA 事件的 CPU 占位类:
    class _EventPlaceholder:
        # 构造:忽略参数,挂载两个 no-op lambda。
        def __init__(self, *args, **kwargs) -> None:
            # record 为 no-op(记录事件,CPU 上无事可做)。
            self.record = lambda: None
            # synchronize 为 no-op(等待事件,CPU 上无事可做)。
            self.synchronize = lambda: None

    # 定义 CUDA 流的 CPU 占位类:
    class _StreamPlaceholder:
        # 构造:忽略参数,挂载 wait_stream no-op 与 cpu 设备。
        def __init__(self, *args, **kwargs) -> None:
            # wait_stream 为 no-op(等待另一流,CPU 上无事可做)。
            self.wait_stream = lambda *a, **kw: None
            # 设备固定为 CPU。
            self.device = torch.device("cpu")

    # 保存原始的 torch.Event。
    cuda_event = torch.Event
    # 保存原始的 torch.cuda.Stream。
    cuda_stream = torch.cuda.Stream
    try:
        # 将 torch.Event 替换为 CPU 占位类。
        torch.Event = _EventPlaceholder
        # 将 torch.cuda.Stream 替换为 CPU 占位类。
        torch.cuda.Stream = _StreamPlaceholder
        # 让出执行权,使 with 体内代码在补丁生效下运行。
        yield
    finally:
        # 恢复原始 torch.Event。
        torch.Event = cuda_event
        # 恢复原始 torch.cuda.Stream。
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    # 设置/恢复全局编译配置的上下文管理器。
    # 说明:MKLDNN 与 CPPGEMM 后端需要冻结参数(freezing),因此
    # 在启用 max_autotune 时临时打开 freezing,退出时恢复。
    # 导入 torch._inductor.config 以读取/修改 inductor 编译配置。
    import torch._inductor.config as torch_inductor_config

    # 获取用户在编译配置里提供的 inductor 配置字典。
    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    # 说明:MKLDNN/CPPGEMM 后端要求冻结参数(freezing)。
    # 保存当前 freezing 设置的原始值,便于退出时恢复。
    freezing_value = torch_inductor_config.freezing
    try:
        # 若用户开启了 max_autotune(自动调优):
        if inductor_config.get("max_autotune", False):
            # 开启参数冻结(满足后端要求)。
            torch_inductor_config.freezing = True
        # 让出执行权,使调用方在修改后的配置下运行。
        yield
    finally:
        # 恢复原始的 freezing 设置。
        torch_inductor_config.freezing = freezing_value


def _set_torch_accelerator_to_noop() -> None:
    # 把 torch.accelerator 的同步/清缓存 API 替换为 no-op,
    # 避免 CPU 环境调用 CUDA 相关实现报错。
    # 定义 no-op 函数,忽略任意参数。
    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    # 将 accelerator.synchronize 替换为 no-op(CPU 天然同步)。
    torch.accelerator.synchronize = noop
    # 将 accelerator.empty_cache 替换为 no-op(CPU 无需清缓存)。
    torch.accelerator.empty_cache = noop