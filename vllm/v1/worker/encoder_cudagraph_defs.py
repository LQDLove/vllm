# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 编码器 CUDA Graph 的数据结构定义(DTO)。
# 包含单条多模态编码器输入项的规格(EncoderItemSpec)、编码器 CUDA Graph 的配置
# (EncoderCudaGraphConfig)、图捕获输入(EncoderCudaGraphCaptureInputs)与
# 图重放缓冲区(EncoderCudaGraphReplayBuffers),供模型与图管理器交换数据。

# 导入 Callable 类型,用于定义 padding 逻辑回调的类型别名。
from collections.abc import Callable
# 导入 dataclass 装饰器与 field 工厂函数,用于定义轻量数据类。
from dataclasses import dataclass, field

# 导入 PyTorch,数据类用 torch.Tensor 标注张量字段。
import torch

# 类型别名:每缓冲区的重放填充/拷贝逻辑。签名: (capture_buffer, replay_buffer) -> None
EncoderCudaGraphPaddingLogic = Callable[[torch.Tensor, torch.Tensor], None]


@dataclass
class EncoderItemSpec:
    """Description of a single encoder input item.

    Returned by ``get_encoder_cudagraph_item_specs()`` to describe each
    image or video in a batch without the manager needing to understand
    model-specific input formats.
    """
    # 编码器输入项规格:描述一条输入(图像/视频)的尺寸与输出 token 数,
    # 使管理器无需理解模型特定的输入格式。

    input_size: int
    """Number of input patches/rows for this item."""
    # 本条输入(input item)的 patch/行数。

    output_tokens: int
    """Number of output tokens after encoder processing (e.g. after
    spatial merge)."""
    # 编码器处理后(如空间合并)的输出 token 数。

    global_output_tokens: int = 0
    """Number of output tokens from the global image path.
    Only used when ``EncoderCudaGraphConfig.enable_dual_path_graph`` is True."""
    # 全局图像路径的输出 token 数;仅当启用双路径图(dual-path graph)时使用。

    local_output_tokens: int = 0
    """Number of output tokens from the local patch path.
    Only used when ``EncoderCudaGraphConfig.enable_dual_path_graph`` is True."""
    # 局部 patch 路径的输出 token 数;仅当启用双路径图(dual-path graph)时使用。


@dataclass
class EncoderCudaGraphConfig:
    """Configuration for encoder CUDA graph management.

    Provided by the model at init time via
    ``get_encoder_cudagraph_config()``. Values are fixed for the
    lifetime of the manager.
    """
    # 编码器 CUDA Graph 管理配置。由模型在初始化时通过
    # get_encoder_cudagraph_config() 提供,配置值在管理器生命周期内固定。

    modalities: list[str]
    """Supported modalities (e.g. ["image"])."""
    # 支持的模态列表(如 ["image"]、["video"] 等)。

    buffer_keys: list[str]
    """Keys for the tensor buffers recorded into the CUDA graph.
    Before replay the manager zeros then slice-copies new data
    into these buffers."""
    # 被录制成 CUDA Graph 的张量缓冲区键名。重放前管理器先将这些缓冲清零,
    # 再把新数据以切片(slice-copy)方式拷入,保证缓冲区地址不变量。

    out_hidden_size: int
    """Output hidden dim of the vision encoder.
    Used for DP gather buffer allocation."""
    # 视觉编码器的输出隐藏维度;用于数据并行(DP)收集缓冲区的分配。

    padding_logics: dict[str, EncoderCudaGraphPaddingLogic] = field(
        default_factory=dict
    )
    """Optional per-buffer replay padding/copy logic.
    If absent for a key, the manager zeros the capture buffer and slice-copies
    the replay buffer into it."""
    # 可选的按缓冲区的重放填充/拷贝逻辑。若某个键缺失,管理器将执行默认行为:
    # 清零捕获缓冲,并把重放缓冲切片拷贝进去。

    max_frames_per_video: int = 1
    """Maximum number of frames per video.
    Only relevant when "video" is in ``modalities``.
    Image-only models can use the default of 1."""
    # 单个视频的最大帧数;仅当 modalities 含 "video" 时相关(纯图像模型用默认值 1)。

    enable_dual_path_graph: bool = False
    """If True, the manager captures two independent graph sets
    (global + local) and runs dual-path graph selection during inference."""
    # 若为 True,管理器捕获两套独立的图(全局 + 局部),并在推理时执行双路径图选择。

    global_token_per_image: int = 0
    """Tokens per global image (e.g. 272 for DeepSeek-OCR).
    Only used when ``enable_dual_path_graph`` is True."""
    # 每张全局图像对应的 token 数(如 DeepSeek-OCR 为 272);仅双路径图时使用。

    local_token_per_patch: int = 0
    """Tokens per local patch (e.g. 100 for DeepSeek-OCR).
    Only used when ``enable_dual_path_graph`` is True."""
    # 每个局部 patch 对应的 token 数(如 DeepSeek-OCR 为 100);仅双路径图时使用。


@dataclass
class EncoderCudaGraphCaptureInputs:
    """Everything needed for one CUDA graph capture.

    Returned by ``prepare_encoder_cudagraph_capture_inputs()``.
    """
    # 一次 CUDA Graph 捕获所需的全部内容。由
    # prepare_encoder_cudagraph_capture_inputs() 返回。

    values: dict[str, torch.Tensor]
    """Precomputed tensor buffers that will be recorded into the
    CUDA graph.  The manager stores references to these exact
    tensor objects and copies new data into them before each
    ``graph.replay()`` call (buffer identity invariant)."""
    # 将被记录进 CUDA Graph 的预计算张量缓冲。管理器保存这些张量对象的精确引用,
    # 并在每次 graph.replay() 前把新数据拷入其中(保持缓冲区身份不变量)。


@dataclass
class EncoderCudaGraphReplayBuffers:
    """New buffer values for graph replay, computed by the model from
    actual batch inputs.

    Returned by ``prepare_encoder_cudagraph_replay_buffers()``.
    Keys match ``EncoderCudaGraphConfig.buffer_keys``.
    """
    # 图重放时的新缓冲值,由模型根据真实 batch 输入计算得到。
    # 由 prepare_encoder_cudagraph_replay_buffers() 返回;键与
    # EncoderCudaGraphConfig.buffer_keys 对应。

    values: dict[str, torch.Tensor | None]
    """Data to copy into the captured buffers before replay.
    ``None`` values leave the corresponding captured buffer
    unchanged."""
    # 重放前拷贝到已捕获缓冲中的数据;值为 None 的键保持对应缓冲不变。