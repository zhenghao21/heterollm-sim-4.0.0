"""Vectorized screening of GEMM architecture and mapping candidates.

This module deliberately accelerates only independent analytical formulae.  It
does not attempt to move the simulator's ordered event queue, topology routing,
or OR-Tools solver onto a GPU.  NumPy is the required implementation; CuPy is
loaded lazily and is an optional drop-in array backend when a usable CUDA device
is present.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .cost_models import mma_output_tile_wave_contract, mma_output_tile_wave_proxy


_BACKENDS = frozenset(("auto", "numpy", "cupy"))


@dataclass(frozen=True)
class BatchedGemmInput:
    """Columnar inputs for independent GEMM candidate estimates.

    Every numeric field may be a scalar or a one-dimensional array.  Scalars
    are broadcast over the candidate dimension, so a workload can efficiently
    be compared against many hardware profiles (or vice versa).

    ``io_*`` describes an optional additional byte-oriented transfer such as
    host/storage I/O.  Its bandwidth is GB/s.  ``communication_*`` describes
    an already-aggregated mapping communication path; its bandwidth is Gbit/s.
    Route discovery and contention remain the responsibility of the regular
    simulator.
    """

    m: Any
    k: Any
    n: Any
    peak_tops: Any
    memory_bandwidth_gb_s: Any
    activation_bits: Any = 8
    weight_bits: Any = 8
    output_bits: Any = 16
    weight_metadata_bytes: Any = 0
    compute_efficiency: Any = 1.0
    memory_efficiency: Any = 1.0
    kernel_launch_ns: Any = 0.0
    io_bytes: Any = 0
    io_bandwidth_gb_s: Any = 0.0
    io_latency_ns: Any = 0.0
    communication_bytes: Any = 0
    communication_bandwidth_gbps: Any = 0.0
    communication_latency_ns: Any = 0.0
    candidate_ids: Optional[Sequence[str]] = None
    sm_count: Any = 0
    tensor_cores_per_sm: Any = 0
    mma_m: Any = 0
    mma_n: Any = 0
    mma_k: Any = 0
    occupancy: Any = 0.0


@dataclass(frozen=True)
class BatchedGemmResult:
    """Host-resident, explainable columns returned by the batch evaluator.

    All arrays use input order.  ``sort_order`` contains indices from fastest
    to slowest and ``rank`` is the inverse zero-based rank.  Ties retain input
    order, including after CuPy execution, because ranking is performed with a
    stable NumPy sort after copying the numeric columns to the host.
    """

    candidate_ids: Optional[Tuple[str, ...]]
    backend_requested: str
    backend_used: str
    diagnostics: Tuple[str, ...]
    hbm_bandwidth_model: str
    hbm_bandwidth_contract: Mapping[str, object]
    operations: np.ndarray
    activation_bytes: np.ndarray
    weight_bytes: np.ndarray
    output_bytes: np.ndarray
    gemm_io_bytes: np.ndarray
    m_tile_count: np.ndarray
    n_tile_count: np.ndarray
    serial_k_tile_count: np.ndarray
    independent_output_tile_count: np.ndarray
    warp_equivalent_count: np.ndarray
    parallel_tile_slots: np.ndarray
    resident_warp_equivalent_capacity: np.ndarray
    output_tile_wave_count: np.ndarray
    output_tile_wave_utilization: np.ndarray
    peak_effective_hbm_bandwidth_gb_s: np.ndarray
    shape_effective_hbm_bandwidth_gb_s: np.ndarray
    hbm_bandwidth_proxy_enabled: np.ndarray
    fallback_to_fixed_bandwidth: np.ndarray
    compute_ns: np.ndarray
    memory_ns: np.ndarray
    roofline_ns: np.ndarray
    kernel_launch_ns: np.ndarray
    io_ns: np.ndarray
    communication_ns: np.ndarray
    total_ns: np.ndarray
    compute_utilization: np.ndarray
    bound: np.ndarray
    sort_order: np.ndarray
    rank: np.ndarray

    @property
    def candidate_count(self) -> int:
        return int(self.total_ns.size)

    @property
    def ranked_candidate_ids(self) -> Optional[Tuple[str, ...]]:
        if self.candidate_ids is None:
            return None
        return tuple(self.candidate_ids[index] for index in self.sort_order)


def evaluate_batched_gemm(
    candidates: BatchedGemmInput, *, backend: str = "auto"
) -> BatchedGemmResult:
    """Evaluate independent GEMM candidates with vectorized roofline formulae.

    ``backend`` accepts ``"auto"``, ``"numpy"``, or ``"cupy"``.  Both auto
    selection and an explicit CuPy request safely fall back to NumPy when CuPy
    or CUDA is unavailable; the Chinese ``diagnostics`` explain that choice.
    CuPy never appears in module-level imports or package dependencies.
    """

    requested = _normalize_backend(backend)
    xp, used, diagnostics = _select_backend(requested)
    columns = _numeric_columns(candidates, xp)
    size = int(columns["m"].size)
    candidate_ids = _validate_candidate_ids(candidates.candidate_ids, size)

    _validate_columns(columns, xp)

    # All arithmetic below is element-wise over the candidate dimension.  No
    # candidate-specific Python dispatch is used, including for 10k+ batches.
    m = columns["m"]
    k = columns["k"]
    n = columns["n"]
    operations = 2.0 * m * k * n
    activation_bytes = xp.ceil(
        m * k * columns["activation_bits"] / 8.0
    )
    weight_bytes = (
        xp.ceil(k * n * columns["weight_bits"] / 8.0)
        + columns["weight_metadata_bytes"]
    )
    output_bytes = xp.ceil(m * n * columns["output_bits"] / 8.0)
    gemm_io_bytes = activation_bytes + weight_bytes + output_bytes

    attainable_tops = columns["peak_tops"] * columns["compute_efficiency"]
    peak_effective_memory_bandwidth = (
        columns["memory_bandwidth_gb_s"] * columns["memory_efficiency"]
    )
    # Structural fields are optional for compatibility with existing batched
    # callers.  The proxy is enabled only for complete, valid structural rows;
    # all other rows retain the historical fixed effective bandwidth.
    integral_structure = (
        (columns["sm_count"] == xp.floor(columns["sm_count"]))
        & (
            columns["tensor_cores_per_sm"]
            == xp.floor(columns["tensor_cores_per_sm"])
        )
        & (columns["mma_m"] == xp.floor(columns["mma_m"]))
        & (columns["mma_n"] == xp.floor(columns["mma_n"]))
        & (columns["mma_k"] == xp.floor(columns["mma_k"]))
    )
    hbm_bandwidth_proxy_enabled = (
        integral_structure
        & (columns["sm_count"] > 0.0)
        & (columns["tensor_cores_per_sm"] > 0.0)
        & (columns["mma_m"] > 0.0)
        & (columns["mma_n"] > 0.0)
        & (columns["mma_k"] > 0.0)
        & (columns["occupancy"] > 0.0)
        & (columns["occupancy"] <= 1.0)
    )
    tile_wave = mma_output_tile_wave_proxy(
        m, k, n,
        sm_count=xp.where(hbm_bandwidth_proxy_enabled, columns["sm_count"], 1.0),
        tensor_cores_per_sm=xp.where(hbm_bandwidth_proxy_enabled, columns["tensor_cores_per_sm"], 1.0),
        mma_m=xp.where(hbm_bandwidth_proxy_enabled, columns["mma_m"], 1.0),
        mma_n=xp.where(hbm_bandwidth_proxy_enabled, columns["mma_n"], 1.0),
        mma_k=xp.where(hbm_bandwidth_proxy_enabled, columns["mma_k"], 1.0),
        occupancy=xp.where(hbm_bandwidth_proxy_enabled, columns["occupancy"], 1.0),
        array_module=xp,
    )
    tile_wave = {
        key: xp.where(hbm_bandwidth_proxy_enabled, value, 1.0 if key == "output_tile_wave_utilization" else 0.0)
        for key, value in tile_wave.items()
    }
    output_tile_wave_utilization = tile_wave["output_tile_wave_utilization"]
    effective_memory_bandwidth = (
        peak_effective_memory_bandwidth * output_tile_wave_utilization
    )
    compute_ns = operations / (attainable_tops * 1000.0)
    memory_ns = gemm_io_bytes / effective_memory_bandwidth
    roofline_ns = xp.maximum(compute_ns, memory_ns)

    io_active = columns["io_bytes"] > 0.0
    safe_io_bandwidth = xp.where(
        columns["io_bandwidth_gb_s"] > 0.0,
        columns["io_bandwidth_gb_s"],
        1.0,
    )
    io_ns = xp.where(
        io_active,
        columns["io_latency_ns"] + columns["io_bytes"] / safe_io_bandwidth,
        0.0,
    )

    communication_active = columns["communication_bytes"] > 0.0
    safe_communication_bandwidth = xp.where(
        columns["communication_bandwidth_gbps"] > 0.0,
        columns["communication_bandwidth_gbps"],
        1.0,
    )
    communication_ns = xp.where(
        communication_active,
        columns["communication_latency_ns"]
        + 8.0
        * columns["communication_bytes"]
        / safe_communication_bandwidth,
        0.0,
    )

    total_ns = (
        columns["kernel_launch_ns"]
        + roofline_ns
        + io_ns
        + communication_ns
    )
    compute_utilization = compute_ns / roofline_ns
    compute_bound = compute_ns >= memory_ns

    host = {
        "operations": _to_numpy(operations, used, xp),
        "activation_bytes": _to_numpy(activation_bytes, used, xp),
        "weight_bytes": _to_numpy(weight_bytes, used, xp),
        "output_bytes": _to_numpy(output_bytes, used, xp),
        "gemm_io_bytes": _to_numpy(gemm_io_bytes, used, xp),
        **{key: _to_numpy(value, used, xp) for key, value in tile_wave.items()},
        "peak_effective_hbm_bandwidth_gb_s": _to_numpy(
            peak_effective_memory_bandwidth, used, xp
        ),
        "shape_effective_hbm_bandwidth_gb_s": _to_numpy(
            effective_memory_bandwidth, used, xp
        ),
        "hbm_bandwidth_proxy_enabled": _to_numpy(
            hbm_bandwidth_proxy_enabled, used, xp
        ),
        "fallback_to_fixed_bandwidth": _to_numpy(
            ~hbm_bandwidth_proxy_enabled, used, xp
        ),
        "compute_ns": _to_numpy(compute_ns, used, xp),
        "memory_ns": _to_numpy(memory_ns, used, xp),
        "roofline_ns": _to_numpy(roofline_ns, used, xp),
        "kernel_launch_ns": _to_numpy(
            columns["kernel_launch_ns"], used, xp
        ),
        "io_ns": _to_numpy(io_ns, used, xp),
        "communication_ns": _to_numpy(communication_ns, used, xp),
        "total_ns": _to_numpy(total_ns, used, xp),
        "compute_utilization": _to_numpy(compute_utilization, used, xp),
        "compute_bound": _to_numpy(compute_bound, used, xp),
    }
    sort_order = np.argsort(host["total_ns"], kind="stable").astype(
        np.int64, copy=False
    )
    rank = np.empty(size, dtype=np.int64)
    rank[sort_order] = np.arange(size, dtype=np.int64)
    bound = np.where(host.pop("compute_bound"), "compute", "memory")

    for value in tuple(host.values()) + (sort_order, rank, bound):
        value.setflags(write=False)

    return BatchedGemmResult(
        candidate_ids=candidate_ids,
        backend_requested=requested,
        backend_used=used,
        diagnostics=diagnostics,
        hbm_bandwidth_model=str(mma_output_tile_wave_contract()["model"]),
        hbm_bandwidth_contract=mma_output_tile_wave_contract(),
        bound=bound,
        sort_order=sort_order,
        rank=rank,
        **host,
    )


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise ValueError("backend 必须是 auto、numpy 或 cupy")
    normalized = backend.strip().lower()
    if normalized not in _BACKENDS:
        raise ValueError(
            "backend 不支持 {!r}；可选值为 auto、numpy、cupy".format(backend)
        )
    return normalized


def _select_backend(requested: str) -> Tuple[Any, str, Tuple[str, ...]]:
    if requested == "numpy":
        return np, "numpy", ("已使用 NumPy CPU 批量后端。",)

    cupy, reason = _probe_cupy()
    if cupy is not None:
        return cupy, "cupy", ("已使用 CuPy/CUDA 批量后端。",)

    if requested == "cupy":
        prefix = "请求的 CuPy/CUDA 后端不可用，已回退到 NumPy CPU"
    else:
        prefix = "未检测到可用的 CuPy/CUDA，auto 已回退到 NumPy CPU"
    return np, "numpy", ("{}：{}。".format(prefix, reason),)


def _probe_cupy() -> Tuple[Optional[Any], str]:
    """Import and minimally validate CuPy without making it a dependency."""

    try:
        cupy = importlib.import_module("cupy")
    except ModuleNotFoundError:
        return None, "CuPy 未安装"
    except ImportError:
        return None, "CuPy 导入失败，请检查 Python、CuPy 与 CUDA 版本是否兼容"
    except OSError:
        return None, "CuPy 动态库无法加载，请检查 CUDA 运行库与 CuPy 版本"
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return None, "没有可用的 CUDA 设备"
        # This catches a present CuPy package with a broken driver/runtime.
        cupy.empty((1,), dtype=cupy.float32)
    except Exception:  # CuPy exposes backend-specific exception types.
        return None, "CUDA 初始化失败，请检查显卡驱动、CUDA 运行库与设备可用性"
    return cupy, ""


def _numeric_columns(candidates: BatchedGemmInput, xp: Any) -> dict:
    names = (
        "m",
        "k",
        "n",
        "peak_tops",
        "memory_bandwidth_gb_s",
        "activation_bits",
        "weight_bits",
        "output_bits",
        "weight_metadata_bytes",
        "compute_efficiency",
        "memory_efficiency",
        "kernel_launch_ns",
        "io_bytes",
        "io_bandwidth_gb_s",
        "io_latency_ns",
        "communication_bytes",
        "communication_bandwidth_gbps",
        "communication_latency_ns",
        "sm_count",
        "tensor_cores_per_sm",
        "mma_m",
        "mma_n",
        "mma_k",
        "occupancy",
    )
    raw = []
    for name in names:
        try:
            value = xp.atleast_1d(xp.asarray(getattr(candidates, name)))
        except Exception as exc:
            raise ValueError("字段 {} 不能转换为数值数组：{}".format(name, exc))
        if value.dtype.kind == "b":
            raise ValueError("字段 {} 不接受布尔值".format(name))
        raw.append(value)
    try:
        broadcast = xp.broadcast_arrays(*raw)
    except ValueError as exc:
        raise ValueError("候选字段的批量形状无法广播：{}".format(exc))
    if not broadcast or broadcast[0].ndim != 1:
        raise ValueError("所有候选字段必须可广播为一维批次")
    if int(broadcast[0].size) == 0:
        raise ValueError("候选批次不能为空")
    try:
        return {
            name: value.astype(xp.float64, copy=False)
            for name, value in zip(names, broadcast)
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("候选字段必须全部为数值：{}".format(exc))


def _validate_columns(columns: dict, xp: Any) -> None:
    for name, value in columns.items():
        if _device_bool(xp.any(~xp.isfinite(value))):
            raise ValueError("字段 {} 必须全部是有限数值".format(name))

    integer_fields = (
        "m",
        "k",
        "n",
        "activation_bits",
        "weight_bits",
        "output_bits",
        "weight_metadata_bytes",
        "io_bytes",
        "communication_bytes",
    )
    for name in integer_fields:
        value = columns[name]
        if _device_bool(xp.any(value != xp.floor(value))):
            raise ValueError("字段 {} 必须全部是整数".format(name))

    positive = (
        "m",
        "k",
        "n",
        "peak_tops",
        "memory_bandwidth_gb_s",
        "activation_bits",
        "weight_bits",
        "output_bits",
    )
    for name in positive:
        if _device_bool(xp.any(columns[name] <= 0.0)):
            raise ValueError("字段 {} 必须全部大于 0".format(name))

    non_negative = (
        "weight_metadata_bytes",
        "kernel_launch_ns",
        "io_bytes",
        "io_bandwidth_gb_s",
        "io_latency_ns",
        "communication_bytes",
        "communication_bandwidth_gbps",
        "communication_latency_ns",
    )
    for name in non_negative:
        if _device_bool(xp.any(columns[name] < 0.0)):
            raise ValueError("字段 {} 必须全部大于等于 0".format(name))

    for name in ("compute_efficiency", "memory_efficiency"):
        value = columns[name]
        if _device_bool(xp.any((value <= 0.0) | (value > 1.0))):
            raise ValueError("字段 {} 必须全部位于 (0, 1]".format(name))

    missing_io_bandwidth = (
        (columns["io_bytes"] > 0.0) & (columns["io_bandwidth_gb_s"] <= 0.0)
    )
    if _device_bool(xp.any(missing_io_bandwidth)):
        raise ValueError("io_bytes 大于 0 时 io_bandwidth_gb_s 必须大于 0")
    missing_communication_bandwidth = (
        (columns["communication_bytes"] > 0.0)
        & (columns["communication_bandwidth_gbps"] <= 0.0)
    )
    if _device_bool(xp.any(missing_communication_bandwidth)):
        raise ValueError(
            "communication_bytes 大于 0 时 communication_bandwidth_gbps 必须大于 0"
        )


def _validate_candidate_ids(
    candidate_ids: Optional[Sequence[str]], size: int
) -> Optional[Tuple[str, ...]]:
    if candidate_ids is None:
        return None
    ids = tuple(candidate_ids)
    if len(ids) != size:
        raise ValueError(
            "candidate_ids 长度 {} 与候选数量 {} 不一致".format(len(ids), size)
        )
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("candidate_ids 必须是非空字符串")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids 不允许重复")
    return ids


def _device_bool(value: Any) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def _to_numpy(value: Any, backend: str, xp: Any) -> np.ndarray:
    if backend == "cupy":
        return np.asarray(xp.asnumpy(value))
    return np.asarray(value)


__all__ = (
    "BatchedGemmInput",
    "BatchedGemmResult",
    "evaluate_batched_gemm",
)
