# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# =============================================================================
# vllm/v1/executor/vllm_net_devices.py
# GPU→NIC 网络设备映射工具（供 RDMA 传输使用：UCX、NVSHMEM、NCCL_IB_HCA 等）。
# 被 UniProcExecutor 与 MultiprocExecutor 共享调用。
#
# 需要两个环境变量同时设置才生效：
#   - VLLM_GPU_NIC_PCIE_MAPPING：逗号分隔的 GPU_BDF=NIC_BDF 映射。
#   - VLLM_NIC_SELECTION_VARS：逗号分隔的要设置的环境变量名列表，
#     每一项可选后缀（如 UCX_NET_DEVICES:1,NCCL_IB_HCA:1）。
# =============================================================================
"""GPU-to-NIC net-device mapping for RDMA transports (UCX, NVSHMEM, ...).

Shared by both UniProcExecutor (TP=1) and MultiprocExecutor (TP>1).
All transport-specific env vars and sysfs lookups live here so executor
files only need to call ``set_worker_net_device(local_rank, vllm_config)``.

Requires two env vars set together:
- ``VLLM_GPU_NIC_PCIE_MAPPING`` -- comma-separated GPU_BDF=NIC_BDF pairs.
- ``VLLM_NIC_SELECTION_VARS`` -- comma-separated list of env vars to set,
  each optionally suffixed (e.g. ``UCX_NET_DEVICES:1,NCCL_IB_HCA:1``).
"""
# 模块文档字符串：说明本模块为 RDMA 传输服务的 GPU→NIC 映射。
# 被 UniProcExecutor（TP=1）与 MultiprocExecutor（TP>1）共享。
# 所有传输相关环境变量与 sysfs 查询集中在此，executor 文件只需调用
# set_worker_net_device(local_rank, vllm_config)。
# 需要两个环境变量同时设置：
#   - VLLM_GPU_NIC_PCIE_MAPPING：GPU_BDF=NIC_BDF 对（逗号分隔）。
#   - VLLM_NIC_SELECTION_VARS：要设置的 env var 列表（逗号分隔，可带后缀）。

import os
# 导入 os：设置/读取环境变量（UCX_NET_DEVICES、NCCL_IB_HCA 等）。
from pathlib import Path
# 导入 Path：遍历 /sys/class/infiniband/ 目录解析 RDMA 设备名。

import vllm.envs as envs
# 导入 vllm 环境变量模块（读取 VLLM_GPU_NIC_PCIE_MAPPING 等）。
from vllm.config import VllmConfig
# 导入 VllmConfig（获取并行配置以调整 DP 场景的 local_rank）。
from vllm.logger import init_logger
# 导入日志初始化函数。
from vllm.platforms import current_platform
# 导入当前平台抽象（获取 GPU PCI BDF、设备 ID 映射等）。

logger = init_logger(__name__)
# 初始化本模块日志。


def normalize_pci(addr: str) -> tuple[int, int, int, int]:
    # =========================================================================
    # 把 PCI BDF / 域-总线-设备-功能 字符串解析为可比较的整数元组（全 16 进制）。
    # =========================================================================
    """Parse PCI BDF/domain-bus-device-function into comparable ints (all hex).

    Supported shapes:
    - ``domain:bus:dev.fn`` -- domain width varies (e.g. ``00000001:00:00.0``,
      ``0001:00:00.0``, ``0000:3f:00.0``).
    - ``bus:dev.fn`` -- domain **0** (e.g. ``01:00.0``, ``40:00.0``).

    Function suffix is hex (typically ``0``--``7``). Raises ``ValueError`` if malformed.
    """
    # 文档字符串：解析 PCI BDF 为可比较的 int 元组（均为 16 进制）。
    # 支持的格式：domain:bus:dev.fn（domain 宽度可变）与 bus:dev.fn（domain 视为 0）。
    # 功能后缀为 16 进制（通常 0-7）。格式错误抛 ValueError。
    s = addr.strip().lower().replace(" ", "")
    # 去除首尾空白、转小写、删除空格。
    if s.startswith("0x"):
        # 若以 0x 前缀开头。
        s = s[2:]
        # 去掉前缀。
    if "." not in s:
        # 若缺少函数后缀点号。
        raise ValueError(f"invalid PCI BDF (missing function suffix): {addr!r}")
        # 报错。
    body, fn_s = s.rsplit(".", 1)
    # 从右侧最后一个点拆分：主体与函数。
    if not fn_s or any(c not in "0123456789abcdef" for c in fn_s):
        # 函数为空或含非法字符。
        raise ValueError(f"invalid PCI function in BDF: {addr!r}")
        # 报错。
    fn = int(fn_s, 16)
    # 解析函数为 16 进制整数。
    if fn > 0xFF:
        # 超出 8 位范围。
        raise ValueError(f"PCI function out of range: {addr!r}")
        # 报错。

    parts = body.split(":")
    # 按冒号拆分主体。
    if len(parts) == 2:
        # 两段：bus:dev。
        domain = 0
        # domain 视为 0。
        bus = int(parts[0], 16)
        # 总线号。
        device = int(parts[1], 16)
        # 设备号。
    elif len(parts) == 3:
        # 三段：domain:bus:dev。
        domain = int(parts[0], 16)
        # domain。
        bus = int(parts[1], 16)
        # 总线号。
        device = int(parts[2], 16)
        # 设备号。
    else:
        raise ValueError(
            f"invalid PCI BDF (want domain:bus:dev.fn or bus:dev.fn): {addr!r}"
        )
        # 非法段数报错。

    if bus > 0xFF or device > 0x1F:
        # 总线超过 255 或设备超过 31。
        raise ValueError(f"PCI bus or device out of range: {addr!r}")
        # 报错。
    return (domain, bus, device, fn)
    # 返回四元组。


def parse_gpu_nic_mapping(
    raw: str,
) -> dict[tuple[int, int, int, int], tuple[int, int, int, int]]:
    # =========================================================================
    # 解析 VLLM_GPU_NIC_PCIE_MAPPING 原始字符串为 {GPU_BDF: NIC_BDF} 映射。
    # =========================================================================
    out: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}
    # 初始化输出字典。
    for segment in raw.split(","):
        # 按逗号分片。
        segment = segment.strip()
        # 去空白。
        if not segment:
            # 空段。
            continue
            # 跳过。
        if "=" not in segment:
            # 无等号。
            raise ValueError(
                "VLLM_GPU_NIC_PCIE_MAPPING: expected comma-separated"
                f" gpu_bdf=nic_bdf pairs; ambiguous segment: {segment!r}"
            )
            # 报错。
        gpu_s, nic_s = segment.split("=", 1)
        # 按首个等号拆分 GPU 与 NIC。
        gpu_key = normalize_pci(gpu_s.strip())
        # 规范化 GPU BDF。
        nic_val = normalize_pci(nic_s.strip())
        # 规范化 NIC BDF。
        out[gpu_key] = nic_val
        # 记录映射。
    return out
    # 返回。


def rdma_name_for_nic_pci(nic_pci: tuple[int, int, int, int]) -> str:
    # =========================================================================
    # 把 NIC PCI BDF 映射为 sysfs 中的 RDMA 设备名（mlx5_*、ibp* 等）。
    # =========================================================================
    """Map NIC PCI BDF to sysfs RDMA name (mlx5_*, ibp*, ...).

    Under ``/sys/class/infiniband/<name>/``, ``device`` is a **symlink** to the PCI
    device directory (e.g. ``.../0101:00:00.0``). We take ``Path(...).resolve().name``
    as the BDF string.

    ``VLLM_GPU_NIC_PCIE_MAPPING`` NIC keys must **normalize** (via ``normalize_pci``)
    to the same tuple as this basename.
    """
    # 文档字符串：把 NIC PCI BDF 映射为 sysfs RDMA 名。
    # /sys/class/infiniband/<name>/device 是指向 PCI 设备目录的符号链接；
    # 取 resolve().name 作为 BDF 字符串。映射中的 NIC 键需经 normalize_pci 规范化。
    ib = Path("/sys/class/infiniband")
    # RDMA 设备基目录。
    if not ib.is_dir():
        # 目录不存在。
        raise RuntimeError("/sys/class/infiniband not found or not a directory")
        # 报错。
    names = sorted(p.name for p in ib.iterdir() if p.is_dir())
    # 列出所有 RDMA 设备名（排序保证确定性）。
    for name in names:
        # 遍历设备。
        dev_link = ib / name / "device"
        # 设备 → PCI 的符号链接。
        if not dev_link.exists():
            # 无链接。
            continue
            # 跳过。
        try:
            resolved = dev_link.resolve()
            # 解析真实路径。
        except OSError:
            continue
            # 解析失败跳过。
        pci_name = resolved.name
        # 取目录名作为 BDF 字符串。
        try:
            if normalize_pci(pci_name) == nic_pci:
                # 规范化后匹配目标 NIC。
                return name
                # 返回 RDMA 设备名。
        except ValueError:
            continue
            # BDF 解析失败跳过。
    raise RuntimeError(
        f"No /sys/class/infiniband device for NIC PCI {nic_pci}; have entries: {names}"
    )
    # 未找到匹配设备报错。


def parse_nic_selection_vars(raw: str) -> list[tuple[str, str]]:
    # =========================================================================
    # 解析 VLLM_NIC_SELECTION_VARS 为 [(环境变量名, 后缀)] 列表。
    # =========================================================================
    """Parse ``VLLM_NIC_SELECTION_VARS`` into ``(env_var_name, suffix)`` pairs.

    Each entry is ``VAR_NAME`` or ``VAR_NAME:<suffix>``.  The colon and
    everything after it is appended verbatim to the RDMA device name.
    """
    # 文档字符串：把 VLLM_NIC_SELECTION_VARS 解析为 (变量名, 后缀) 对。
    # 每项是 VAR_NAME 或 VAR_NAME:<后缀>；冒号及之后内容原样追加到 RDMA 设备名。
    result: list[tuple[str, str]] = []
    # 初始化结果。
    for entry in raw.split(","):
        # 按逗号分片。
        entry = entry.strip()
        # 去空白。
        if not entry:
            # 空项。
            continue
            # 跳过。
        if ":" in entry:
            # 含冒号后缀。
            var_name, suffix = entry.split(":", 1)
            # 拆分变量名与后缀。
            result.append((var_name, ":" + suffix))
            # 记录（后缀补回冒号）。
        else:
            result.append((entry, ""))
            # 无后缀。
    return result
    # 返回。


def set_worker_gpu_nic_mapping(local_rank: int) -> None:
    # =========================================================================
    # 根据 VLLM_GPU_NIC_PCIE_MAPPING 为某个 worker 设置 NIC 选择环境变量。
    # 具体设置哪些变量由 VLLM_NIC_SELECTION_VARS 控制。
    # =========================================================================
    """Set NIC selection env vars from VLLM_GPU_NIC_PCIE_MAPPING for a worker.

    Which env vars are set is controlled by ``VLLM_NIC_SELECTION_VARS``.
    """
    # 文档字符串：为 worker 设置由映射决定的 NIC 选择环境变量。
    raw = envs.VLLM_GPU_NIC_PCIE_MAPPING.strip()
    # 读取映射原文。
    if not raw:
        # 空映射。
        return
        # 直接返回。
    selection_raw = envs.VLLM_NIC_SELECTION_VARS.strip()
    # 读取要设置的变量列表。
    selection_vars = parse_nic_selection_vars(selection_raw)
    # 解析变量列表。
    mapping = parse_gpu_nic_mapping(raw)
    # 解析 GPU→NIC 映射。
    pci_by_index = current_platform.get_all_gpu_pci_bus_ids()
    # 获取所有 GPU 物理索引 → PCI BDF 的映射。
    # Translate CUDA-relative local_rank to the physical device index,
    # which accounts for CUDA_VISIBLE_DEVICES narrowing (e.g. DP sharding).
    # 注释：把 CUDA 相对 local_rank 转成物理设备索引；
    # 这考虑 CUDA_VISIBLE_DEVICES 收窄（如 DP 分片）。
    physical_id = current_platform.device_id_to_physical_device_id(local_rank)
    # 转物理设备 ID。
    if physical_id not in pci_by_index:
        # 无对应 PCI。
        raise RuntimeError(
            f"No GPU PCI for physical device index {physical_id} "
            f"(local_rank={local_rank}) in map "
            f"(have indices {sorted(pci_by_index.keys())})"
        )
        # 报错。
    gpu_bdf = pci_by_index[physical_id]
    # 取 GPU BDF。
    gpu_key = normalize_pci(gpu_bdf)
    # 规范化 GPU BDF。
    if gpu_key not in mapping:
        # 映射中无该 GPU。
        keys_fmt = ", ".join(
            f"{d:04x}:{b:02x}:{dev:02x}.{fn}"
            # 格式化。
            for d, b, dev, fn in sorted(mapping.keys())
            # 遍历排序后的键。
        )
        # 格式化已有键。
        raise RuntimeError(
            f"No VLLM_GPU_NIC_PCIE_MAPPING entry for GPU PCI {gpu_bdf} "
            f"(worker local_rank={local_rank}); mapped GPUs: {keys_fmt}"
        )
        # 报错。
    nic_pci = mapping[gpu_key]
    # 查 NIC BDF。
    rdma_dev = rdma_name_for_nic_pci(nic_pci)
    # 解析 RDMA 设备名。

    set_vars: list[str] = []
    # 记录已设置的变量（日志用）。
    for var_name, suffix in selection_vars:
        # 遍历要设置的变量。
        value = f"{rdma_dev}{suffix}"
        # 构造值：RDMA 名 + 可选后缀。
        existing = os.environ.get(var_name, "").strip()
        # 读现有值。
        if existing:
            # 已有值。
            value = f"{value},{existing}"
            # 前插（新值优先）。
        os.environ[var_name] = value
        # 设置环境变量。
        set_vars.append(f"{var_name}={value}")
        # 记录。

    nic_fmt = f"{nic_pci[0]:04x}:{nic_pci[1]:02x}:{nic_pci[2]:02x}.{nic_pci[3]}"
    # 格式化 NIC BDF 用于日志。
    logger.info(
        "GPU rank %s (PCIe addr %s) mapped to NIC %s (PCIe addr %s) via env vars: %s",
        local_rank,
        # rank。
        gpu_bdf,
        # GPU BDF。
        rdma_dev,
        # RDMA 设备名。
        nic_fmt,
        # NIC BDF。
        ", ".join(set_vars),
        # 设置的变量。
    )
    # 记录映射日志。


def _dp_adjusted_local_rank(tp_local_rank: int, vllm_config: VllmConfig) -> int:
    # =========================================================================
    # 计算考虑数据并行（DP）后的节点级 GPU 索引。
    # =========================================================================
    """Compute the node-wide GPU index accounting for data parallelism.

    On CUDA-alike platforms without env-var device isolation (the common
    MP-backend path), the worker sees *all* GPUs on the node and selects
    its device via ``torch.accelerator.set_device_index()`` using::

        dp_local_rank * tp_pp_world_size + tp_local_rank

    This mirrors the adjustment in ``Worker.init_device()`` so we resolve
    the correct GPU PCI address *before* the CUDA device is initialised.
    """
    # 文档字符串：计算考虑 DP 后的节点级 GPU 索引。
    # 无环境变量设备隔离的平台（常见 MP 后端路径）上，worker 可见节点全部 GPU，
    # 用 dp_local_rank * tp_pp_world_size + tp_local_rank 公式选择设备。
    # 这与 Worker.init_device() 的调整保持一致，从而在 CUDA 设备初始化前
    # 解析正确的 GPU PCI 地址。
    pc = vllm_config.parallel_config
    # 取并行配置。
    if (
        pc.distributed_executor_backend not in ("ray", "external_launcher")
        # 非 Ray/外部 launcher。
        and pc.data_parallel_backend != "ray"
        # DP 后端非 Ray。
        and pc.nnodes_within_dp == 1
        # 单 DP 节点。
    ):
        # 常规 MP 路径需要调整。
        dp_local_rank = pc.data_parallel_rank_local
        # 取本地 DP rank。
        if dp_local_rank is None:
            # 未设置则回退。
            dp_local_rank = pc.data_parallel_index
            # 用全局 DP 索引。
        tp_pp_world_size = pc.pipeline_parallel_size * pc.tensor_parallel_size
        # TP×PP 世界大小。
        return dp_local_rank * tp_pp_world_size + tp_local_rank
        # 返回调整后的节点级索引。
    return tp_local_rank
    # 其他路径无需调整。


def set_worker_net_device(local_rank: int, vllm_config: VllmConfig) -> None:
    # =========================================================================
    # 顶层入口：UniProcExecutor 与 MultiprocExecutor 都调用本函数。
    # 若设置了 VLLM_GPU_NIC_PCIE_MAPPING 与 VLLM_NIC_SELECTION_VARS 则设置
    # 网卡选择环境变量；否则为 no-op。
    # =========================================================================
    """Top-level entry point for both UniProcExecutor and MultiprocExecutor.

    Sets NIC selection env vars from ``VLLM_GPU_NIC_PCIE_MAPPING`` and
    ``VLLM_NIC_SELECTION_VARS`` if present; no-op otherwise.
    """
    # 文档字符串：UniProc/Multiproc 的顶层入口。若两个环境变量存在则设置
    # NIC 选择 env var；否则 no-op。
    has_pcie_mapping = bool(envs.VLLM_GPU_NIC_PCIE_MAPPING.strip())
    # 是否设置映射。
    has_selection_vars = bool(envs.VLLM_NIC_SELECTION_VARS.strip())
    # 是否设置变量列表。
    if has_pcie_mapping and not has_selection_vars:
        # 只设映射。
        raise RuntimeError(
            "VLLM_GPU_NIC_PCIE_MAPPING is set but VLLM_NIC_SELECTION_VARS "
            "is not; both must be set together."
        )
        # 要求同时设置。
    if has_selection_vars and not has_pcie_mapping:
        # 只设变量列表。
        raise RuntimeError(
            "VLLM_NIC_SELECTION_VARS is set but VLLM_GPU_NIC_PCIE_MAPPING "
            "is not; both must be set together."
        )
        # 要求同时设置。
    # No-op when neither env var is present.
    # 注释：两个环境变量都未设置时为 no-op。
    if not has_pcie_mapping and not has_selection_vars:
        # 均未设置。
        return
        # 直接返回。
    # Both env vars are present, so set the NIC selection env vars.
    # 注释：两个环境变量都存在，设置 NIC 选择环境变量。
    adjusted_rank = _dp_adjusted_local_rank(local_rank, vllm_config)
    # 计算 DP 调整后的 rank。
    set_worker_gpu_nic_mapping(adjusted_rank)
    # 设置映射。