# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Workspace 临时缓冲管理器。
# 按微批(ubatch)槽位管理一块可增长的 GPU 工作区,供注意力等 kernel 复用
# 临时内存;执行期间可锁定以防止继续增长。

# 导入 inspect 模块,用于在 get_caller_info 中遍历调用栈定位调用方。
import inspect
# 导入 os 模块,用于在 get_caller_info 中取文件名。
import os
# 导入 accumulate,用于计算各张量在 workspace 中的累积偏移。
from itertools import accumulate
# 导入 prod,用于计算形状各维度乘积(张量元素个数)。
from math import prod

# 导入 PyTorch,用于创建 workspace 张量。
import torch

# 导入 vllm.envs,用于读取 VLLM_DEBUG_WORKSPACE 调试开关。
import vllm.envs as envs
# 导入日志初始化函数,用于创建模块日志记录器。
from vllm.logger import init_logger
# 导入 round_up,用于把字节数按 256 对齐。
from vllm.utils.math_utils import round_up
# 导入 dbo_current_ubatch_id,用于获取当前微批 id 以选择对应槽位。
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

# 创建本模块的日志记录器。
logger = init_logger(__name__)


def _compute_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    # 计算给定 shape 与 dtype 的张量所需字节数。
    # 元素个数 = 形状各维乘积;字节数 = 元素个数 × 每元素字节数。
    return prod(shape) * dtype.itemsize


# Constants
# 常量定义:
# 1 MB = 1024^2 字节(用于日志中的 MB 换算)。
_MB = 1024**2
# 1 GiB = 1024^3 字节。
_GiB = 1024**3

# Global workspace manager instance
# 全局 workspace 管理器单例(初始为 None)。
_manager: "WorkspaceManager | None" = None


class WorkspaceManager:
    # workspace 分配管理器。
    # 为每个活跃的微批(ubatch)槽位维护一块 workspace 缓冲,供注意力等 kernel
    # 复用临时内存;可锁定以防止执行期继续增长。
    """Manager for workspace allocation.

    Manages one workspace buffer per active ubatch slot.
    Can be locked to prevent further growth during execution.
    """

    def __init__(self, device: torch.device, num_ubatches: int | None = None):
        # 初始化管理器:记录设备与微批槽位数,创建初始 workspace 列表。
        # 保存分配 workspace 的目标设备。
        self._device = device
        # Cache num ubatches at init based on configuration (default to 1)
        # 根据配置缓存微批槽位数(未指定时默认 1)。
        self._num_ubatches = num_ubatches if num_ubatches is not None else 1
        # 为每个微批槽位创建初始为 None 的 workspace 槽(懒分配)。
        self._current_workspaces: list[torch.Tensor | None] = [
            None
        ] * self._num_ubatches
        # 锁定标记:False 表示允许增长,True 表示锁定。
        self._locked: bool = False

    @staticmethod
    def _workspace_size_bytes(workspace: torch.Tensor | None) -> int:
        # 获取 workspace 张量的字节大小。
        """Get size of workspace in bytes."""
        # 空 workspace 大小为 0。
        if workspace is None:
            return 0
        # 否则 = 元素个数 × 每元素字节数。
        return workspace.numel() * workspace.element_size()

    def lock(self) -> None:
        # 锁定 workspace,禁止继续增长;此后更大的分配请求将抛 AssertionError,
        # 确保执行期间 workspace 尺寸固定。
        """Lock the workspace to prevent further growth.

        After locking, any attempt to allocate a larger workspace will raise
        an assertion error. This ensures workspace size is fixed during execution.
        """
        # 置锁定标记为 True。
        self._locked = True
        # 若开启了 workspace 调试:
        if envs.VLLM_DEBUG_WORKSPACE:
            # 记录锁定日志及各槽位当前大小(MB)。
            logger.info(
                "[WORKSPACE DEBUG] Workspace locked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def unlock(self) -> None:
        # 解锁 workspace 允许增长(弹性 EP 扩缩容时使用)。
        """Unlock the workspace to allow growth.

        This is used during elastic EP scaling when the workspace size
        needs to grow due to changes in the number of experts.
        """
        # 置锁定标记为 False。
        self._locked = False
        # 若开启了 workspace 调试:
        if envs.VLLM_DEBUG_WORKSPACE:
            # 记录解锁日志及各槽位当前大小(MB)。
            logger.info(
                "[WORKSPACE DEBUG] Workspace unlocked. Current sizes: %s",
                [
                    self._workspace_size_bytes(ws) / _MB
                    for ws in self._current_workspaces
                    if ws is not None
                ],
            )

    def is_locked(self) -> bool:
        # 查询 workspace 是否已锁定。
        """Check if workspace is locked."""
        # 返回锁定标记。
        return self._locked

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        # 从单次分配中获得多个 (shape, dtype) 张量视图。
        # 按 256 字节对齐计算各张量偏移后,在 workspace 缓冲中切分视图返回。
        """Get multiple workspace tensors simultaneously from a single allocation.

        Args:
            *shapes_and_dtypes: One or more (shape, dtype) tuples.

        Returns:
            List of tensor views into the workspace buffer, one per shape/dtype pair.
        """
        # 计算每个 (shape, dtype) 所需的实际字节数。
        actual_bytes = [_compute_bytes(s, d) for s, d in shapes_and_dtypes]
        # 把每个字节数向上对齐到 256 字节。
        aligned_bytes = [round_up(actual, 256) for actual in actual_bytes]
        # 求和得到所需总字节数。
        total_bytes = sum(aligned_bytes)

        # Calculate cumulative offsets using itertools.accumulate
        # 用 accumulate 计算各张量在缓冲中的累积偏移(前缀 [0, ...])。
        offsets = list(accumulate([0] + aligned_bytes[:-1]))

        # 确保 workspace 槽位足够大并返回当前槽位张量。
        current_workspace = self._ensure_workspace_size(total_bytes)

        # 按偏移切出每个张量的视图,并 reshape 为目标 shape 返回。
        return [
            current_workspace[offsets[i] : offsets[i] + actual_bytes[i]]
            .view(shapes_and_dtypes[i][1])
            .reshape(shapes_and_dtypes[i][0])
            for i in range(len(shapes_and_dtypes))
        ]

    def _ensure_workspace_size(self, required_bytes: int) -> torch.Tensor:
        # 确保当前微批槽位的 workspace 已分配且不小于所需字节;
        # 不足时若已锁定则报错,否则释放旧段并重新分配更大的缓冲。
        """Ensure workspace is allocated and large enough, return current workspace.

        Args:
            required_bytes: The number of bytes required.

        Returns:
            The current workspace tensor.
        """
        # 获取当前微批 id,以选定对应槽位。
        ubatch_id = dbo_current_ubatch_id()
        # 取当前槽位的 workspace(可能为 None)。
        current_workspace = self._current_workspaces[ubatch_id]
        # 计算当前 workspace 的大小。
        current_size = self._workspace_size_bytes(current_workspace)

        # 若当前大小不足以容纳所需字节:
        if current_size < required_bytes:

            def get_caller_info() -> str:
                # 查找第一个在 WorkspaceManager 之外的调用帧(用于报错/日志定位)。
                """Find first frame outside WorkspaceManager."""
                # 获取当前帧。
                curr_frame = inspect.currentframe()
                # 获取失败时返回 unknown。
                if curr_frame is None:
                    return "unknown"
                # Walk up the stack skipping WorkspaceManager frames
                # 沿调用栈向上跳过 WorkspaceManager 内部帧。
                curr_frame = curr_frame.f_back
                while curr_frame is not None:
                    # TODO: This only catches instance methods (self), missing
                    # classmethods and staticmethods. Once Python 3.11+ is the
                    # minimum supported version, use co_qualname instead:
                    #   qualname = curr_frame.f_code.co_qualname
                    #   if qualname.startswith("WorkspaceManager."):
                    # 注:当前只识别实例方法(self);classmethod/staticmethod 可能
                    # 遗漏。Python 3.11+ 可用 co_qualname 更可靠地判断。
                    # 若该帧的局部变量含 WorkspaceManager 实例(self):
                    if isinstance(curr_frame.f_locals.get("self"), WorkspaceManager):
                        # 继续向上查找。
                        curr_frame = curr_frame.f_back
                        continue
                    # 取调用方文件名(仅基名)。
                    filename = os.path.basename(curr_frame.f_code.co_filename)
                    # 返回 "文件:行号:函数名" 定位信息。
                    return (
                        f"{filename}:{curr_frame.f_lineno}:{curr_frame.f_code.co_name}"
                    )
                # 未找到外部帧时返回 unknown。
                return "unknown"

            # 若 workspace 已锁定:
            if self._locked:
                # 抛出断言错误:锁定后不允许增长。
                raise AssertionError(
                    f"Workspace is locked but allocation from '{get_caller_info()}' "
                    f"requires {required_bytes / _MB:.2f} MB, current size is "
                    f"{current_size / _MB:.2f} MB. "
                    "Workspace growth is not allowed after locking."
                )

            # Only resize the requesting ubatch's workspace.  Other
            # ubatches resize lazily on their next get_simultaneous call.
            # Resizing all ubatches here would orphan the other ubatch's
            # old tensor when it still holds views into it (DBO leak).
            # 说明:只调整请求方微批的 workspace;其它微批在下次调用时懒分配。
            # 若在此处调整所有微批,会使仍持有旧视图的其它微批张量失效(DBO 泄漏)。
            # 先把该槽位置为 None。
            self._current_workspaces[ubatch_id] = None
            # 删除旧张量引用以便释放。
            del current_workspace
            # Release the freed segment back to CUDA so the caching
            # allocator can reuse the GPU memory for the larger
            # allocation below. Without this, each resize may leave a
            # dead segment in reserved memory which can cause higher peak
            # memory usage.
            # 说明:把释放的段归还 CUDA,使缓存分配器可为下面的更大分配复用内存;
            # 否则每次扩容都会在预留内存中留下死段,导致更高峰值内存。
            torch.accelerator.empty_cache()
            # 重新分配一个足够大的 uint8 workspace(大小为 required_bytes 字节)。
            self._current_workspaces[ubatch_id] = torch.empty(
                (required_bytes,), dtype=torch.uint8, device=self._device
            )
            # 更新当前 workspace 引用。
            current_workspace = self._current_workspaces[ubatch_id]

            # 若开启了 workspace 调试:
            if envs.VLLM_DEBUG_WORKSPACE:
                # 记录扩容日志(调用方、旧大小、新大小、微批 id)。
                logger.info(
                    "[WORKSPACE DEBUG] Resized workspace from '%s': %.2f MB -> "
                    "%.2f MB (ubatch %d)",
                    get_caller_info(),
                    current_size / _MB,
                    required_bytes / _MB,
                    ubatch_id,
                )

        # 返回当前(可能刚扩容的)workspace 张量。
        return current_workspace


def is_workspace_manager_initialized() -> bool:
    # 查询 workspace 管理器是否已初始化。
    """Check if workspace manager has been initialized.

    Returns:
        True if workspace manager is initialized, False otherwise.
    """
    # 全局管理器非 None 即为已初始化。
    return _manager is not None


def current_workspace_manager() -> "WorkspaceManager":
    # 获取当前全局 workspace 管理器实例(未初始化则断言失败)。
    """Get the current workspace manager instance.

    Raises:
        AssertionError: If workspace manager has not been initialized.
    """
    # 断言管理器已初始化,否则报错提示先调用 init_workspace_manager()。
    assert _manager is not None, (
        "WorkspaceManager not initialized. Call init_workspace_manager() "
        "with a device before using workspace functions."
    )
    # 返回全局管理器实例。
    return _manager


def init_workspace_manager(
    device: torch.device, num_ubatches: int | None = None
) -> None:
    # 以指定设备(与微批槽位数)初始化全局 workspace 管理器。
    # 须在使用任何 workspace 函数前调用,通常在 GPUModelRunner.__init__ 中调用。
    """Initialize the workspace manager with a device.

    Must be called before using any workspace functions. Typically called
    from GPUModelRunner.__init__.

    Args:
        device: The device to allocate workspace on.
        num_ubatches: Number of workspace ubatch slots. Defaults to 1.
    """
    # 引用全局管理器变量以便重赋值。
    global _manager
    # 若管理器已初始化过(重复初始化):
    if _manager is not None:
        # 记录警告:在旧设备上已初始化,现正重新初始化为新设备。
        logger.warning(
            "WorkspaceManager already initialized on device %s, "
            "reinitializing on device %s",
            _manager._device,
            device,
        )
    # 创建新的 WorkspaceManager 并赋给全局变量。
    _manager = WorkspaceManager(device, num_ubatches)


def lock_workspace() -> None:
    # 锁定全局 workspace,防止热路径上的意外内存分配。
    # 调用全局管理器实例的 lock。
    """Lock the workspace to prevent further growth.

    After calling this function, any attempt to allocate a workspace larger
    than the current size will raise an AssertionError. This ensures that
    workspace size is fixed during execution and prevents unexpected memory
    allocations in the hot path.

    Example:
        # During initialization
        init_workspace_manager(device)
        reserve_workspace(shape1, dtype1)
        reserve_workspace(shape2, dtype2)

        # Lock after warmup/profiling
        lock_workspace()

        # Now all get_workspace calls must fit in pre-allocated size
    """
    current_workspace_manager().lock()


def unlock_workspace() -> None:
    # 解锁全局 workspace 以允许增长(弹性 EP 扩缩容场景)。
    # 调用全局管理器实例的 unlock。
    """Unlock the workspace to allow growth.

    This is used during elastic EP scaling when the workspace size
    needs to grow due to changes in the number of experts.
    After scaling operations complete, lock_workspace() should be
    called again to prevent unexpected allocations.
    """
    current_workspace_manager().unlock()


def reset_workspace_manager() -> None:
    # 将全局 workspace 管理器重置为未初始化状态。
    # 主要用于测试场景,使测试可干净地重新初始化管理器。
    """Reset the workspace manager to uninitialized state.

    This is primarily intended for testing purposes to allow tests
    to reinitialize the workspace manager cleanly.
    """
    # 引用全局管理器变量。
    global _manager
    # 置为 None(未初始化)。
    _manager = None