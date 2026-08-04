# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================================
# VLLM-WORKER-COMMENT
# 启动计划:持久化并复用内存 profiling 结果。
# 以(模型/配置/硬件/库)指纹为 key,在 VLLM_CACHE_ROOT/startup_plan 下缓存
# KV cache 内存值,后续启动在指纹匹配且空闲内存充足时跳过 profiling。
# ============================================================================
"""Persist and reuse the memory-profiling result across engine boots.

On startup, vLLM measures how much GPU memory the KV cache can use and
computes the ``--kv-cache-memory`` value that reproduces that allocation.
For a fixed (model, config, hardware, library) combination the result is
deterministic, yet it is re-measured on every boot.

When ``VLLM_ENABLE_STARTUP_PLAN=1``, each worker persists that value under
``{VLLM_CACHE_ROOT}/startup_plan/`` (regenerable derived state, alongside
the torch.compile cache), keyed by a fingerprint of everything the value
depends on, and later boots apply it automatically -- skipping the
memory-profiling measurement and the CUDA-graph memory estimation pass --
if and only if the fingerprint matches and the device has at least as much
free memory as when the plan was recorded. On any mismatch the worker
falls back to full profiling, so a stale plan costs nothing and is never
trusted.
"""
# 模块说明:在每次引擎启动时持久化并复用内存 profiling 结果。
# 启动时 vLLM 测量 KV cache 可用显存并计算对应的 --kv-cache-memory 值;
# 对固定(模型/配置/硬件/库)组合结果确定,却每次启动都重新测量。
# 当 VLLM_ENABLE_STARTUP_PLAN=1 时,每个 worker 将该值保存到
# {VLLM_CACHE_ROOT}/startup_plan/(可再生的派生状态,与 torch.compile 缓存并列),
# 以一切影响该值的因素指纹为键;后续启动在指纹匹配且设备空闲内存不少于
# 记录基线时自动应用,跳过内存 profiling 与 CUDA graph 内存估算。
# 任何不匹配都会回退到完整 profiling,因此陈旧计划代价为零且永不被信任。

# 导入 hashlib,用于计算指纹的 SHA256 摘要。
import hashlib
# 导入 json,用于序列化/反序列化计划文件。
import json
# 导入 os,用于路径拼接与目录创建。
import os
# 导入 TYPE_CHECKING,用于仅类型检查时的导入。
from typing import TYPE_CHECKING

# 导入 PyTorch(通过 torch.__version__ 参与指纹计算)。
import torch

# 导入 vllm.envs,用于读取 VLLM_ENABLE_STARTUP_PLAN 与 VLLM_CACHE_ROOT。
import vllm.envs as envs
# 导入 VllmConfig,compute_plan_fingerprint 使用其 compute_hash。
from vllm.config import VllmConfig
# 导入日志初始化函数。
from vllm.logger import init_logger
# 导入 current_platform,用于获取设备名称/总内存/算力。
from vllm.platforms import current_platform

# 仅在类型检查时导入 Worker(避免循环依赖)。
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

# 创建本模块的日志记录器。
logger = init_logger(__name__)

# 启动计划文件的 schema 版本,用于校验兼容性。
PLAN_SCHEMA_VERSION = 1


def compute_plan_fingerprint(
    vllm_config: VllmConfig, rank: int, world_size: int
) -> str:
    """Hash everything the profiled KV-cache memory value depends on.

    ``VllmConfig.compute_hash()`` covers the vLLM version and the model,
    cache, parallel, and compilation configs, but deliberately contains no
    device identity (``DeviceConfig.compute_hash`` is empty), so device
    name, total memory, compute capability, and the torch/CUDA build are
    added here. The vLLM version is also pinned as an explicit factor so
    version invalidation holds no matter how ``compute_hash`` evolves.
    Rank is included because per-rank memory use differs under TP/PP.
    Driver-only changes are not part of the key; the free-memory gate at
    apply time bounds the residual risk.
    """
    # 计算启动计划指纹:对影响 profiling 出的 KV cache 内存值的所有因素做哈希。
    # VllmConfig.compute_hash() 覆盖 vLLM 版本与模型/缓存/并行/编译配置,
    # 但故意不含设备身份,因此这里补充设备名称、总内存、算力与 torch/CUDA 构建。
    # vLLM 版本也显式作为因子,确保版本失效不依赖 compute_hash 的演化;
    # rank 被包含,因为 TP/PP 下各 rank 内存使用不同。
    # 仅驱动(驱动)层面的变化不参与密钥;应用时的空闲内存门控约束残余风险。
    # Imported here (as VllmConfig.compute_hash does) to avoid a cycle with
    # the top-level vllm package.
    # 在此处导入(与 VllmConfig.compute_hash 一致),避免与顶层 vllm 包循环依赖。
    from vllm import __version__ as vllm_version

    # 获取设备算力(可能为 None)。
    capability = current_platform.get_device_capability()
    # 构建指纹因子字典:包含 schema、vllm 版本、配置哈希、设备信息等。
    factors = {
        "schema": PLAN_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": vllm_config.compute_hash(),
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(capability) if capability else "",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "rank": rank,
        "world_size": world_size,
    }
    # 对因子做排序 JSON 序列化后取 SHA256 十六进制摘要。
    digest = hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()
    # 截取前 16 个字符作为指纹标识。
    return digest[:16]


def _plan_path(fingerprint: str) -> str:
    """Plans are regenerable derived state, so they live under the standard
    vLLM cache root (like the torch.compile cache) and relocate with
    ``VLLM_CACHE_ROOT`` instead of needing a location knob of their own."""
    # 返回启动计划文件的路径。
    # VLLM_CACHE_ROOT is already user-expanded by envs.py.
    # 说明:VLLM_CACHE_ROOT 已由 envs.py 完成用户展开。
    # 拼接 VLLM_CACHE_ROOT/startup_plan/startup_plan_{fingerprint}.json。
    return os.path.join(
        envs.VLLM_CACHE_ROOT, "startup_plan", f"startup_plan_{fingerprint}.json"
    )


def _load_plan(fingerprint: str) -> dict | None:
    """Load a plan for this fingerprint; None if absent or unreadable."""
    # 加载指定指纹的启动计划;不存在或不可读时返回 None。
    # 计算计划文件路径。
    path = _plan_path(fingerprint)
    # 尝试打开并读取 JSON。
    try:
        with open(path) as f:
            plan = json.load(f)
    # 文件不存在时返回 None。
    except FileNotFoundError:
        return None
    # 其它 I/O 或解析错误:记录警告并返回 None。
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable startup plan %s: %s", path, e)
        return None
    # 校验 schema 版本与指纹匹配:
    if (
        plan.get("schema") != PLAN_SCHEMA_VERSION
        or plan.get("fingerprint") != fingerprint
    ):
        # 不匹配则视为无效,返回 None。
        return None
    # 校验通过,返回计划字典。
    return plan


def _applicable_kv_cache_memory_bytes(
    plan: dict, current_free_memory: int
) -> int | None:
    """The apply-time OOM-safety gate.

    The recorded value is only valid if the device has at least as much
    free memory now as when the plan was measured (co-tenants, leaked
    allocations, or MIG changes all reduce it). Outside that envelope,
    return None and let the caller re-profile.
    """
    # 应用时的 OOM 安全门控:仅当当前空闲内存不少于计划记录基线时才有效。
    # 若设备空闲内存减少(同租户、泄漏或 MIG 变化),超出该范围返回 None,
    # 让调用方重新 profiling。
    # 取计划中的 KV cache 内存字节数。
    kv_bytes = plan.get("kv_cache_memory_bytes")
    # 取计划中的空闲内存基线。
    baseline = plan.get("free_memory_baseline")
    # 若两者类型不合法,返回 None。
    if not isinstance(kv_bytes, int) or not isinstance(baseline, int):
        return None
    # 若 KV 内存值非正,返回 None。
    if kv_bytes <= 0:
        return None
    # 若当前空闲内存小于基线:
    if current_free_memory < baseline:
        # 记录日志:当前空闲内存低于记录基线,回退到完整内存 profiling。
        logger.info(
            "Startup plan not applied: current free memory (%.2f GiB) is "
            "below the recorded baseline (%.2f GiB); falling back to full "
            "memory profiling.",
            current_free_memory / (1 << 30),
            baseline / (1 << 30),
        )
        # 返回 None,表示不应用计划。
        return None
    # 门控通过,返回缓存的 KV cache 内存值。
    return kv_bytes


def maybe_apply_startup_plan(worker: "Worker") -> None:
    """If enabled and ``--kv-cache-memory`` was not set explicitly, apply a
    persisted plan by setting ``worker.cache_config.kv_cache_memory_bytes``.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``."""
    # 若启用且用户未显式指定 --kv-cache-memory,则应用持久化计划:
    # 通过设置 worker.cache_config.kv_cache_memory_bytes 生效。
    # 未启用计划或无显式配置时什么都不做(VLLM_ENABLE_STARTUP_PLAN=1 才有效)。
    # 未启用计划,或用户已显式指定 KV cache 内存时直接返回。
    if (
        not envs.VLLM_ENABLE_STARTUP_PLAN
        or worker.cache_config.kv_cache_memory_bytes is not None
    ):
        return
    # 计算本 rank 的计划指纹。
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    # 加载该指纹对应的计划。
    plan = _load_plan(fingerprint)
    # 未找到计划时直接返回。
    if plan is None:
        return
    # 取当前空闲内存。
    current_free_memory = worker.init_snapshot.free_memory
    # 通过 OOM 安全门控判断是否可用。
    kv_bytes = _applicable_kv_cache_memory_bytes(plan, current_free_memory)
    # 门控不通过时返回(执行完整 profiling)。
    if kv_bytes is None:
        return
    # 记录应用计划的日志(指纹、KV 内存值、记录/当前空闲内存)。
    logger.info(
        "Applying persisted startup plan (fingerprint %s): "
        "kv_cache_memory_bytes=%d (%.2f GiB), recorded free-memory "
        "baseline %.2f GiB, current %.2f GiB. Memory profiling will "
        "be skipped.",
        fingerprint,
        kv_bytes,
        kv_bytes / (1 << 30),
        plan["free_memory_baseline"] / (1 << 30),
        current_free_memory / (1 << 30),
    )
    # 将缓存的 KV 内存值写入 cache 配置,使后续初始化跳过 profiling。
    worker.cache_config.kv_cache_memory_bytes = kv_bytes


def maybe_save_startup_plan(worker: "Worker", kv_cache_memory_bytes: int) -> None:
    """Atomically persist this boot's profiling result for future boots.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``; failures are logged,
    never raised."""
    # 原子地持久化本次启动的 profiling 结果供后续复用。
    # 未启用 VLLM_ENABLE_STARTUP_PLAN=1 时不做任何事;
    # 失败仅记录日志,绝不抛出异常。
    # 未启用计划时直接返回。
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return
    # 计算本 rank 的计划指纹。
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    # 计算计划文件路径。
    path = _plan_path(fingerprint)
    # 尝试写入计划文件:
    try:
        # 确保存放目录存在。
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 组装计划负载:schema、指纹、KV 内存值、空闲内存基线。
        payload = {
            "schema": PLAN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
            "free_memory_baseline": int(worker.init_snapshot.free_memory),
        }
        # 将负载原子写入临时文件并替换为最终路径。
        with open(path, "w") as f:
            json.dump(payload, f)  # type: ignore[arg-type]
        # 记录已保存计划的日志。
        logger.info(
            "Saved startup plan (fingerprint: %s, kv_cache_memory_bytes=%s)",
            fingerprint,
            kv_cache_memory_bytes,
        )
    # 任何写入失败都只记录异常,不向上抛出。
    except Exception as e:
        logger.warning("Failed to save startup plan: %s", e)