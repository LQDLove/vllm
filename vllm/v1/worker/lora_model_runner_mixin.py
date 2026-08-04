# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define LoRA functionality mixin for model runners.
"""
# 定义 LoRA 功能的模型 runner 混入(Mixin)。
# 为模型 runner 提供 LoRA 适配器加载、激活、LRU 缓存管理,
# 以及 CUDA Graph 捕获时的 dummy LoRA 支持。

# 导入 contextmanager 装饰器,用于定义上下文管理器。
from contextlib import contextmanager
# 导入 TypeAlias,用于定义输入批处理联合类型。
from typing import TypeAlias

# 导入 numpy,用于映射数组构造。
import numpy as np
# 导入 PyTorch,用于设备与模型类型标注。
import torch
# 导入 nn 模块,用于模型类型标注。
import torch.nn as nn

# 导入 VllmConfig 配置类。
from vllm.config import VllmConfig
# 导入 LoRA 配置类。
from vllm.config.lora import LoRAConfig
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 LoRA 映射数据结构与映射类型。
from vllm.lora.layers import LoRAMapping, LoRAMappingType
# 导入 LoRA 请求类型。
from vllm.lora.request import LoRARequest
# 导入 LRU 缓存 worker LoRA 管理器。
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
# 导入 supports_lora,用于检查模型是否支持 LoRA。
from vllm.model_executor.models import supports_lora
# 导入 GPU 输入批处理。
from vllm.v1.worker.gpu_input_batch import InputBatch as GPUInputBatch
# 导入 TPU 输入批处理。
from vllm.v1.worker.tpu_input_batch import InputBatch as TPUInputBatch

# 类型别名:输入批处理可以是 TPU 或 GPU 的实现。
InputBatch: TypeAlias = TPUInputBatch | GPUInputBatch

# 创建本模块的日志记录器。
logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
# 定义为 GPUModelRunner(及其变体)共用的混入。
class LoRAModelRunnerMixin:
    # LoRA 模型 runner 混入:提供 LoRA 适配器加载、激活与管理能力。

    def load_lora_model(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> nn.Module:
        # 把 LoRA 适配器集成到已加载模型中。
        # 参数:
        #   model: 已加载的基础模型。
        #   vllm_config: 完整配置。
        #   device: 目标设备。
        # 若模型不支持 LoRA:
        if not supports_lora(model):
            # 抛出错误。
            raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")

        # Add LoRA Manager to the Model Runner
        # 向模型 runner 添加 LoRA 管理器:
        # 创建基于 LRU 缓存的 worker LoRA 管理器。
        self.lora_manager = LRUCacheWorkerLoRAManager(
            vllm_config,
            device,
            model.embedding_modules,
        )
        # 用管理器包装模型并返回(嵌入模块映射)。
        return self.lora_manager.create_lora_manager(model, vllm_config)

    def _set_active_loras(
        self,
        prompt_lora_mapping: tuple[int, ...],
        token_lora_mapping: tuple[int, ...],
        lora_requests: set[LoRARequest],
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        # 设置当前活跃的 LoRA 适配器集合。
        # 参数:
        #   prompt_lora_mapping: 每个采样 token 对应的 LoRA id(长度=采样 token 数)。
        #   token_lora_mapping: 每个调度 token 对应的 LoRA id(长度=调度 token 数)。
        #   lora_requests: 活跃的 LoRA 请求集合。
        #   mapping_type: LoRA 映射类型(默认语言)。
        # 确保 LoRA 已启用。
        self._ensure_lora_enabled()

        # Set is_prefill to True, so we always use the SGMV kernels on
        # non-cuda platforms.
        # On cuda platforms we use the same kernels for prefill and
        # decode and this flag is generally ignored.
        # 说明:将 is_prefill 置为 True,使非 CUDA 平台始终使用 SGMV kernel;
        # CUDA 平台上 prefill 与 decode 使用相同 kernel,此标志通常被忽略。
        # 构造 LoRA 映射。
        lora_mapping = LoRAMapping(
            token_lora_mapping,
            prompt_lora_mapping,
            is_prefill=True,
            type=mapping_type,
        )
        # 激活适配器。
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def _ensure_lora_enabled(self) -> None:
        # 断言 LoRA 已启用(lora_manager 存在)。
        if not hasattr(self, "lora_manager"):
            raise RuntimeError("LoRA is not enabled. Use --enable-lora to enable LoRA.")

    def set_active_loras(
        self,
        input_batch: InputBatch,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray | None = None,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ) -> None:
        # 从输入批处理生成映射并激活 LoRA。
        # 参数:
        #   input_batch: 当前输入批处理。
        #   num_scheduled_tokens: 各请求的调度 token 数。
        #   num_sampled_tokens: 各请求的采样 token 数(默认取 1)。
        #   mapping_type: LoRA 映射类型。
        # 若未提供采样 token 数,默认每个请求采样 1 个 token。
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        # 声明映射与请求集合变量。
        prompt_lora_mapping: tuple[int, ...]  # of size np.sum(num_sampled_tokens)
        token_lora_mapping: tuple[int, ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
        # 从输入批处理生成 LoRA 输入(映射与请求)。
        prompt_lora_mapping, token_lora_mapping, lora_requests = (
            input_batch.make_lora_inputs(num_scheduled_tokens, num_sampled_tokens)
        )
        # 激活相应 LoRA。
        return self._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests, mapping_type
        )

    @contextmanager
    def maybe_setup_dummy_loras(
        self, lora_config: LoRAConfig | None, remove_lora: bool = True
    ):
        # 为 CUDA Graph 捕获搭建若干 dummy LoRA 适配器。
        # 参数:
        #   lora_config: LoRA 配置(为 None 表示未启用)。
        #   remove_lora: 退出后是否移除所有适配器。
        # 若未启用 LoRA,直接让出。
        if lora_config is None:
            yield
        else:
            # __enter__ code
            # 进入代码:断言 LoRA 管理器存在。
            assert self.lora_manager is not None, "LoRA is not enabled"

            # 取最大 LoRA 数量。
            num_loras = lora_config.max_loras
            # 计算 warmup 秩(上限 8)。
            lora_warmup_rank: int = (
                lora_config.max_lora_rank if lora_config.max_lora_rank < 8 else 8
            )
            # 从管理器获取合适的 warmup 秩。
            lora_warmup_rank = self.lora_manager.get_dummy_lora_warmup_rank(
                lora_warmup_rank
            )
            # Make dummy lora requests
            # 构造 dummy LoRA 请求(路径为假路径)。
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, num_loras + 1)
            }

            # 进入 dummy LoRA 缓存上下文。
            with self.lora_manager.dummy_lora_cache():
                # Add the dummy LoRAs here so _set_active_loras doesn't try to
                # load from disk.
                # 在此添加 dummy LoRA,使 _set_active_loras 不从磁盘加载。
                for lr in lora_requests:
                    # 添加 dummy LoRA(指定秩)。
                    self.lora_manager.add_dummy_lora(lr, rank=lora_warmup_rank)

                # 让出,使调用方在 dummy LoRA 就绪时运行。
                yield

            # __exit__ code
            # 退出代码:若需移除:
            if remove_lora:
                # 移除所有适配器。
                self.lora_manager.remove_all_adapters()

    @contextmanager
    def maybe_select_dummy_loras(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
        num_sampled_tokens: np.ndarray | None = None,
        num_active_loras: int = 0,
    ):
        # 选择用于当前 dummy 运行的 LoRA 组合(匹配捕获用例)。
        # 参数:
        #   lora_config: LoRA 配置(为 None 表示禁用)。
        #   num_scheduled_tokens: 各请求调度 token 数数组。
        #   num_sampled_tokens: 各请求采样 token 数数组。
        #   num_active_loras: 使用的不同 LoRA 数量:
        #       - 0: 无 LoRA 活跃(设置空映射)。
        #       - >0: 使用恰好这么多不同的 LoRA。
        # 若未提供采样 token 数,默认每个请求采样 1 个。
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)

        # Skip LoRA setup entirely only if no LoRA config
        # 仅当无 LoRA 配置时完全跳过 LoRA 设置。
        if lora_config is None:
            yield
        else:
            # __enter__ code
            # 进入代码:断言 LoRA 管理器存在。
            assert self.lora_manager is not None, "LoRA is not enabled"

            # 请求数。
            num_reqs = len(num_scheduled_tokens)
            # 最大 LoRA 数。
            max_loras = lora_config.max_loras

            # Determine how many distinct LoRAs to use and whether to include
            # no-LoRA tokens (-1 entries).
            # When num_active_loras > max_loras (e.g., max_loras + 1), we need
            # to include -1 entries to simulate batches with both LoRA and
            # no-LoRA tokens. This ensures prepare_tensors computes the correct
            # num_active_loras that matches the cudagraph capture key.
            # 说明:确定使用的不同 LoRA 数及是否包含无 LoRA token(-1 项)。
            # 当 num_active_loras > max_loras 时,需包含 -1 项以模拟混合批,
            # 确保 prepare_tensors 计算出与 CUDA Graph 捕获键匹配的活跃数。
            if num_active_loras == 0:
                # No LoRA active - use 0 mappings like the original code
                # 无 LoRA 活跃:使用 0 映射(与原代码一致)。
                effective_num_loras = 0
                include_no_lora = False
            elif num_active_loras > max_loras:
                # num_active_loras > max_loras means we want max_loras adapters
                # PLUS no-LoRA tokens (-1). This is the max_loras + 1 case.
                # 需求为 max_loras 个适配器加无 LoRA token(-1),即 max_loras+1 情形。
                effective_num_loras = max_loras
                include_no_lora = True
            else:
                # Specific number of active LoRAs requested
                # 请求特定数量的活跃 LoRA。
                effective_num_loras = min(num_active_loras, max_loras)
                include_no_lora = False

            # Make prompt lora mapping
            # Assign LoRA IDs cyclically to simulate a worst-case scenario.
            # LoRA IDs are 1-indexed (1 to max_loras) as required by LoRARequest.
            # convert_mapping() will convert these to 0-indexed slot indices.
            # 构造 prompt LoRA 映射:循环分配 LoRA id 以模拟最坏情形。
            # LoRA id 从 1 开始(1 到 max_loras),convert_mapping() 会转为 0 基槽位。
            if effective_num_loras > 0:
                if include_no_lora:
                    # Include -1 (no-LoRA) entries by cycling through
                    # -1, 1, 2, ..., effective_num_loras
                    # This ensures prepare_tensors sees both LoRA and no-LoRA
                    # tokens, computing num_active_loras = effective_num_loras+1
                    # 通过循环 -1, 1, 2, ... 包含无 LoRA 项,使 prepare_tensors
                    # 同时看到 LoRA 与无 LoRA token,计算活跃数为 effective_num_loras+1。
                    cycle_values = np.array(
                        list(range(1, effective_num_loras + 1)),
                        dtype=np.int32,
                    )
                    prompt_lora_mapping = cycle_values[
                        np.arange(num_reqs, dtype=np.int32) % len(cycle_values)
                    ]
                else:
                    # Use 1 to effective_num_loras (1-indexed lora IDs)
                    # 使用 1 到 effective_num_loras(1 基 LoRA id)。
                    prompt_lora_mapping = (
                        np.arange(num_reqs, dtype=np.int32) % effective_num_loras
                    ) + 1
            else:
                # No LoRA active - use 0 for all tokens (original behavior)
                # 无 LoRA 活跃:所有 token 用 0(原始行为)。
                prompt_lora_mapping = np.zeros(num_reqs, dtype=np.int32)

            # Make sample lora mapping
            # 构造采样映射:按采样 token 数重复 prompt 映射。
            sample_lora_mapping = np.repeat(prompt_lora_mapping, num_sampled_tokens)

            # Make token lora mapping
            # 构造 token 映射:按调度 token 数重复 prompt 映射。
            token_lora_mapping = np.repeat(prompt_lora_mapping, num_scheduled_tokens)

            # Make dummy lora requests (only for the active LoRAs)
            # 仅构造活跃 LoRA 的 dummy 请求。
            lora_requests: set[LoRARequest] = {
                LoRARequest(
                    lora_name=f"warmup_{lora_id}",
                    lora_int_id=lora_id,
                    lora_path="/not/a/real/path",
                )
                for lora_id in range(1, effective_num_loras + 1)
            }

            # 激活这些 dummy LoRA。
            self._set_active_loras(
                tuple(sample_lora_mapping),
                tuple(token_lora_mapping),
                lora_requests,
                mapping_type,
            )

            # 让出,使调用方在选中 LoRA 后运行。
            yield

    @contextmanager
    def maybe_dummy_run_with_lora(
        self,
        lora_config: LoRAConfig | None,
        num_scheduled_tokens: np.ndarray,
        num_sampled_tokens: np.ndarray,
        remove_lora: bool = True,
        num_active_loras: int = 0,
        mapping_type: LoRAMappingType = LoRAMappingType.LANGUAGE,
    ):
        # 在 dummy LoRA 状态下执行 dummy 运行的上下文管理器。
        # 参数:
        #   lora_config: LoRA 配置。
        #   num_scheduled_tokens: 各请求调度 token 数。
        #   num_sampled_tokens: 各请求采样 token 数。
        #   remove_lora: 退出后是否移除 LoRA。
        #   num_active_loras: 使用的不同 LoRA 数量(>0 才激活 LoRA)。
        # 组合搭建与选择两个上下文。
        with (
            self.maybe_setup_dummy_loras(lora_config, remove_lora),
            self.maybe_select_dummy_loras(
                lora_config,
                num_scheduled_tokens,
                mapping_type,
                num_sampled_tokens,
                num_active_loras,
            ),
        ):
            # 让出执行。
            yield

    def maybe_remove_all_loras(self, lora_config: LoRAConfig | None):
        # 若启用了 LoRA,移除所有适配器。
        # 未启用 LoRA 直接返回。
        if lora_config is None:
            return
        # 移除所有适配器。
        self.lora_manager.remove_all_adapters()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 添加一个 LoRA 适配器并返回是否成功。
        # 确保 LoRA 已启用。
        self._ensure_lora_enabled()
        # 调用管理器添加适配器。
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        # 移除指定 id 的 LoRA 并返回是否成功。
        # 确保 LoRA 已启用。
        self._ensure_lora_enabled()
        # 调用管理器移除适配器。
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        # 固定指定 id 的 LoRA,防止被 LRU 逐出。
        # 确保 LoRA 已启用。
        self._ensure_lora_enabled()
        # 调用管理器固定适配器。
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        # 列出当前已加载的 LoRA id 集合。
        # 确保 LoRA 已启用。
        self._ensure_lora_enabled()
        # 从管理器获取适配器列表。
        return self.lora_manager.list_adapters()