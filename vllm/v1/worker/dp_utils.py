# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 数据并行(DP)执行协调工具。
# 通过跨 DP ranks 的 all_reduce 同步实际/填充后的 token 数、是否启用微批(ubatch)
# 以及 CUDA Graph 模式,确保各 rank 以一致的方式执行当前迭代。

# 导入 PyTorch 主模块,用于张量操作。
import torch
# 导入 torch.distributed,用于跨 rank 的 all_reduce 集合通信。
import torch.distributed as dist

# 导入 ParallelConfig,用于读取 DP 相关配置。
from vllm.config import ParallelConfig
# 导入 get_dp_group,用于获取 DP 进程组(设备组与 CPU 组)。
from vllm.distributed.parallel_state import get_dp_group
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入微批工具:check_ubatch_thresholds 检查是否达到启用微批的条件;
# is_last_ubatch_empty 判断最后一个微批是否为空。
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_last_ubatch_empty,
)

# 创建本模块的日志记录器。
logger = init_logger(__name__)


def _get_device_and_group(parallel_config: ParallelConfig):
    # 获取 DP 组对应的设备与进程组。
    # Use the actual device assigned to the DP group, not just the device type
    # 使用 DP 组实际分配的设备(而非仅设备类型)。
    device = get_dp_group().device
    # 获取 DP 组的设备进程组(用于 GPU 上的 NCCL all_reduce)。
    group = get_dp_group().device_group

    # Transferring this tensor from GPU to CPU will introduce a GPU sync
    # point that could adversely affect performance of vllm with asynch
    # scheduling. This environment variable exists to quickly disable
    # this optimization if we run into this case.
    # 说明:把上述张量从 GPU 传回 CPU 会引入 GPU 同步点,可能影响
    # 异步调度下的性能;此环境变量用于在遇到问题时快速禁用该优化。
    if parallel_config.disable_nccl_for_dp_synchronization:
        # 记录已改用 CPU all_reduce 同步 DP padding。
        logger.info_once(
            "Using CPU all reduce to synchronize DP padding between ranks.",
        )
        # 改用 CPU 设备做 all_reduce。
        device = "cpu"
        # 改用 CPU 进程组。
        group = get_dp_group().cpu_group
    # 返回 (设备, 进程组)。
    return device, group


def _run_ar(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    # 执行一次跨 DP ranks 的 all_reduce,汇总各 rank 的 token 元信息。
    # 参数:
    #   should_ubatch: 本 rank 是否尝试启用微批。
    #   orig_num_tokens_per_ubatch: 未填充的每微批 token 数。
    #   padded_num_tokens_per_ubatch: 填充后的每微批 token 数。
    #   cudagraph_mode: 本 rank 的 CUDA Graph 模式(0=NONE,1=PIECEWISE,2=FULL)。
    #   parallel_config: 并行配置。
    # 取 DP 世界大小(rank 总数)。
    dp_size = parallel_config.data_parallel_size
    # 取本 rank 在 DP 组中的编号。
    dp_rank = parallel_config.data_parallel_rank
    # 获取执行 all_reduce 的设备与进程组。
    device, group = _get_device_and_group(parallel_config)
    # Populate this rank's contribution on CPU to reduce GPU syncs.
    # 在 CPU 上填充本 rank 的贡献(形状 (4, dp_size) 的全零 int32 张量),
    # 以减少 GPU 同步。
    tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)
    # 第 0 行第 dp_rank 列:本 rank 未填充的每微批 token 数。
    tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
    # 第 1 行第 dp_rank 列:本 rank 填充后的每微批 token 数。
    tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch
    # 第 2 行第 dp_rank 列:本 rank 是否尝试微批(1 或 0)。
    tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0
    # 第 3 行第 dp_rank 列:本 rank 的 CUDA Graph 模式。
    tensor_cpu[3][dp_rank] = cudagraph_mode
    # 将 CPU 张量异步拷贝到目标设备(NCCL 同步避免阻塞)。
    tensor = tensor_cpu.to(device, non_blocking=True)
    # 跨 DP 进程组执行 all_reduce 求和,得到所有 rank 的汇总信息。
    dist.all_reduce(tensor, group=group)
    # 返回汇总后的张量。
    return tensor


def _post_process_ubatch(tensor: torch.Tensor, num_ubatches: int) -> bool:
    # 判断所有 DP ranks 是否都同意启用微批。
    # 参数:
    #   tensor: all_reduce 后的汇总张量(形状 (4, dp_size))。
    #   num_ubatches: 微批数量。
    # 取各 rank 未填充的每微批 token 数。
    orig_num_tokens_tensor = tensor[0, :]
    # 取各 rank 填充后的每微批 token 数。
    padded_num_tokens_tensor = tensor[1, :]

    # First determine if we are going to be ubatching.
    # 首先判断是否启用微批:仅当所有 rank 都标记了尝试微批(tensor[2] 全为 1)。
    should_ubatch: bool = bool(torch.all(tensor[2] == 1).item())
    # 任一 rank 未尝试微批则直接返回 False。
    if not should_ubatch:
        return False
    # If the DP ranks are planning to ubatch, make sure that
    # there are no "empty" second ubatches
    # 若各 DP rank 计划启用微批,还需确保不存在“空的”第二个微批:
    # 取所有 rank 中未填充 token 的最小值。
    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    # 取所有 rank 中填充后 token 的最大值。
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    # 若最后一个微批为空(仅存在 padding 中):
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens, num_ubatches):
        # 记录调试日志并放弃微批。
        logger.debug(
            "Aborting ubatching %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        # 置 should_ubatch 为 False。
        should_ubatch = False
    # 返回最终是否启用微批。
    return should_ubatch


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    # 按需将各 rank 的 token 数填充到全 DP 组最大值。
    # 参数:
    #   tensor: all_reduce 后的汇总张量。
    #   should_dp_pad: 是否需要做 DP padding。
    # 取各 rank 填充后的每微批 token 数(第 1 行)。
    num_tokens_across_dp = tensor[1, :]
    # 若需要 DP padding:
    if should_dp_pad:
        # If DP padding is enabled, ensure that each rank is processing the same number
        # of tokens
        # 启用 DP padding 时,确保每个 rank 处理相同数量的 token:
        # 取所有 rank 中 token 数的最大值。
        max_num_tokens = int(num_tokens_across_dp.max().item())
        # 返回一个长度 dp_size 的 CPU 张量,每个元素均为最大值。
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    else:
        # 不需要 padding 时,返回各 rank 原始 token 数的 CPU 张量。
        return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    # 跨 DP ranks 同步 cudagraph 模式:取最小值。
    # 若任一 rank 为 NONE(0),则所有 rank 都用 NONE。
    # 这保证所有 rank 发送一致的数值(全部填充或全部不填充)。
    # 取第 3 行(各 rank 的 cudagraph 模式)的最小值并转为整数。
    return int(tensor[3, :].min().item())


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do.

    2. Determines the total number of tokens that each rank will run.
    When running microbatched or if cudagraph is enabled (synced across ranks),
    all ranks will be padded out so that they run with the same number of tokens.

    3. Synchronizes cudagraph_mode across ranks by taking the minimum.

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including any DP padding.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]
    """
    # 跨 DP ranks 协调:
    # 1. 决定每个 DP rank 是否执行微批(要么全部微批,要么全不微批)。
    # 2. 确定每个 rank 将执行的 token 总数;微批或启用 cudagraph 时,
    #    所有 rank 都会被填充到相同 token 数。
    # 3. 通过取最小值同步各 rank 的 cudagraph 模式。
    # 返回:
    #   should_ubatch: 所有 DP rank 是否都执行微批。
    #   num_tokens_after_padding: 每个 DP rank 每微批含 padding 的 token 数张量。
    #   synced_cudagraph_mode: 同步后的 cudagraph 模式(跨 rank 的最小值)。
    # 断言填充后的 token 数 ≥ 未填充的 token 数。
    assert num_tokens_padded >= num_tokens_unpadded

    # Coordinate between the DP ranks via an All Reduce
    # to determine the total number of tokens that each rank
    # will run and if we are using ubatching or not.
    # 通过 all_reduce 协调各 DP rank,确定每个 rank 将执行的 token 数
    # 以及是否使用微批。
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    # Synchronize cudagraph_mode across ranks first (take min).
    # This is needed before DP padding decision since we use the synced
    # cudagraph mode to determine whether DP padding is needed.
    # 先同步 cudagraph 模式(取最小值)。这必须在 DP padding 决策之前,
    # 因为我们用同步后的 cudagraph 模式决定是否需要 DP padding。
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

    # Check conditions for microbatching
    # 检查启用微批的条件。
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)

    # DP padding is needed when cudagraph is enabled (synced across ranks)
    # or when ubatching/DBO is active (ubatching requires uniform batch
    # sizes across DP ranks currently).
    # Use the synced runtime cudagraph mode rather than the compilation config
    # so we can avoid padding when cudagraph is not enabled for this step.
    # 当 cudagraph 启用(跨 rank 同步)或微批/DBO 激活时(微批当前要求
    # DP ranks 间 batch 大小一致)需要 DP padding。
    # 使用同步后的运行时 cudagraph 模式而非编译配置,
    # 以便在本步未启用 cudagraph 时避免 padding。
    should_dp_pad = synced_cudagraph_mode != 0 or should_ubatch

    # Pad all DP ranks up to the maximum token count across ranks if
    # should_dp_pad is True
    # 当 should_dp_pad 为 True 时,把所有 DP rank 填充到跨 rank 的最大 token 数。
    num_tokens_after_padding = _post_process_dp_padding(
        tensor,
        should_dp_pad,
    )

    # 返回 (是否微批, 填充后的 token 数张量, 同步后的 cudagraph 模式)。
    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode


def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Args:
        num_tokens_unpadded: Number of tokens without accounting for padding
        allow_microbatching: If microbatching should be attempted
        parallel_config: The parallel config
        num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,
            TP, etc)
        uniform_decode: Only used if allow_microbatching is True. True if the batch
            only contains single token decodes
        cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL).
            DP padding is enabled when synced cudagraph mode across ranks is not NONE.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        padded up to the max value across all DP ranks when cudagraph is enabled.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]
    """
    # 在全部 DP ranks 间协调 full batch 是否需要拆分为微批(ubatch)。
    # Args:
    #   num_tokens_unpadded: 未考虑 padding 的 token 数。
    #   allow_microbatching: 是否允许尝试微批。
    #   parallel_config: 并行配置。
    #   num_tokens_padded: 含非 DP padding(CUDA graphs、TP 等)的 token 数。
    #   uniform_decode: 仅 allow_microbatching=True 时使用;若 batch 只含
    #       单 token 解码则为 True。
    #   cudagraph_mode: 本 rank 的 cudagraph 模式(0=NONE,1=PIECEWISE,2=FULL)。
    #       跨 rank 同步后的模式非 NONE 时启用 DP padding。
    # Returns:
    #   should_ubatch: 若设置,则所有 DP rank 同意微批。
    #   num_tokens_after_padding: 每个 DP rank 每微批含 padding 的 token 数张量;
    #       cudagraph 启用时会被填充到跨 rank 的最大值。
    #   synced_cudagraph_mode: 同步后的 cudagraph 模式(跨 rank 最小值)。
    # 若 DP 大小为 1(单个 rank),无需协调,直接返回不微批。
    if parallel_config.data_parallel_size == 1:
        # Early exit.
        # 提前退出。
        return False, None, cudagraph_mode

    # If the caller has explicitly enabled microbatching.
    # 若调用方显式允许微批:
    should_attempt_ubatching = False
    if allow_microbatching:
        # Check preconditions for microbatching
        # 检查微批前置条件:
        # 断言 uniform_decode 参数必须提供(仅微批路径需要)。
        assert uniform_decode is not None
        # 通过阈值检查判断是否应尝试微批。
        should_attempt_ubatching = check_ubatch_thresholds(
            parallel_config,
            num_tokens_unpadded,
            uniform_decode=uniform_decode,
        )

    # 若未提供含 padding 的 token 数,默认等于未填充的 token 数。
    if num_tokens_padded is None:
        num_tokens_padded = num_tokens_unpadded

    # 调用内部协调函数,得到 (是否微批, 填充后 token 数张量, 同步 cudagraph 模式)。
    (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode) = (
        _synchronize_dp_ranks(
            num_tokens_unpadded,
            num_tokens_padded,
            should_attempt_ubatching,
            cudagraph_mode,
            parallel_config,
        )
    )

    # 返回最终结果:(是否微批, 各 rank 填充后的每微批 token 数张量, 同步后的 cudagraph 模式)。
    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode
