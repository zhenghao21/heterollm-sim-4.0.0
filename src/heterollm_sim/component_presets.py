"""Offline component preset catalog for HardwareIR components.

The catalog returns one standalone :class:`ComponentSpec` template per preset.
It intentionally does not include links or placement changes: callers add the
component to the current topology and connect it explicitly when appropriate.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, replace
import re
import json
import os
import tempfile
import threading
import time
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .ir import ComponentSpec, LinkSpec, PortSpec, SCHEMA_VERSION
from .serde import to_primitive


CATALOG_VERSION = "1.3.0"
CATALOG_CUTOFF_AT = "2026-08-27T00:00:00Z"

CAPABILITY_UNITS: Mapping[str, str] = {
    "capacity_bytes": "B",
    "peak_ops_per_s": "op/s",
    "bandwidth_gbps": "Gb/s_decimal_shared_total",
    "read_bandwidth_gbps": "Gb/s_decimal_one_way",
    "write_bandwidth_gbps": "Gb/s_decimal_one_way",
    "read_latency_ns": "ns",
    "write_latency_ns": "ns",
    "transfer_granularity_bytes": "B",
    "max_outstanding_requests": "request_count",
    "dma_bandwidth_gbps": "Gb/s_decimal_one_way",
    "dma_latency_ns": "ns",
    "dma_energy_pj_per_byte": "pJ/B",
}
PORT_CAPABILITY_UNITS: Mapping[str, str] = {
    "lanes": "lane_count",
    "bandwidth_gbps": "Gb/s_decimal_one_way",
    "max_links": "link_count",
}
LINK_CAPABILITY_UNITS: Mapping[str, str] = {
    "lanes": "lane_count",
    "bandwidth_gbps": "Gb/s_decimal_one_way",
    "latency_ns": "ns",
}

# Component templates are commonly inserted into the bundled V4 scenario.  A
# typed cost component therefore carries the same explicit binding used by
# that scenario's registry.  Consumers with a different registry must replace
# this id deliberately; no code here infers a profile from a multi-entry
# registry.
_DEFAULT_COST_PROFILE_IDS: Mapping[str, str] = {
    "gpu": "legacy-gpu",
    "hbm": "legacy-hbm",
    "gddr": "legacy-gddr",
    "cpu": "legacy-cpu",
    "host_memory": "legacy-host-memory",
    "cim": "legacy-cim",
}

_CATALOG_PROCESS_LOCK = threading.RLock()
_CATALOG_FILE_LOCK_TIMEOUT_S = 10.0


class ComponentPresetPersistenceError(ValueError):
    """The user-owned component-preset persistence is unavailable."""


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("组件预设目录包含重复的 JSON 字段：{}".format(key))
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value):
    raise ValueError("组件预设目录包含非有限 JSON 数值：{}".format(value))


@contextmanager
def _catalog_file_lock_unchecked(path: Path):
    """Serialize catalog mutations across instances and processes."""

    with _CATALOG_PROCESS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".lock")
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + _CATALOG_FILE_LOCK_TIMEOUT_S
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("组件预设目录锁定超时")
                        time.sleep(0.01)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _catalog_file_lock(path: Path):
    try:
        with _catalog_file_lock_unchecked(path):
            yield
    except ComponentPresetPersistenceError:
        raise
    except OSError as exc:
        raise ComponentPresetPersistenceError("组件预设目录锁定失败：{}".format(path)) from exc


def _default_cost_profile_id(kind: str) -> Optional[str]:
    normalized = kind.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"digital_sram_cim", "compute_in_memory", "compute_in_memory_tile"}:
        normalized = "cim"
    elif normalized == "hbm_stack":
        normalized = "hbm"
    return _DEFAULT_COST_PROFILE_IDS.get(normalized)

S1_STANDARD = "S1_STANDARD"
S2_VENDOR_DECLARED = "S2_VENDOR_DECLARED"
S3_VENDOR_PREPRODUCTION = "S3_VENDOR_PREPRODUCTION"
S4_PRIMARY_RESEARCH = "S4_PRIMARY_RESEARCH"
A_ANALYTICAL = "A_ANALYTICAL"
EVIDENCE_LEVELS = frozenset(
    {
        S1_STANDARD,
        S2_VENDOR_DECLARED,
        S3_VENDOR_PREPRODUCTION,
        S4_PRIMARY_RESEARCH,
        A_ANALYTICAL,
    }
)

COMPONENT_KINDS = frozenset(
    {
        "cpu",
        "gpu",
        "hbm",
        "gddr",
        "host_memory",
        "hbf",
        "ssd",
        "high_io_ssd",
        "digital_sram_cim",
    }
)

USAGE_HINT = "该接口只返回一个组件模板；前端应将其添加到当前拓扑，并由用户按需重命名和连接端口。"
BUNDLE_USAGE_HINT = "该预设是组合拓扑；前端应原子添加全部组件、内部链路和默认展开的视觉分组，并作为一次操作撤销。"


@dataclass(frozen=True)
class ComponentSource:
    title: str
    url: str
    evidence_level: str
    publisher: str
    published_at: str = ""
    accessed_at: str = "2026-08-27"

    def to_metadata(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "evidence_level": self.evidence_level,
            "publisher": self.publisher,
            "published_at": self.published_at,
            "accessed_at": self.accessed_at,
        }


@dataclass(frozen=True)
class ComponentPresetDefinition:
    preset_id: str
    name: str
    family: str
    component: ComponentSpec
    sources: Tuple[ComponentSource, ...]
    evidence_level: str
    limitations: Tuple[str, ...]
    notes: str
    tags: Tuple[str, ...] = ()

    @property
    def component_kind(self) -> str:
        return self.component.kind


@dataclass(frozen=True)
class TopologyBundleDefinition:
    preset_id: str
    name: str
    family: str
    components: Tuple[ComponentSpec, ...]
    links: Tuple[LinkSpec, ...]
    group: Mapping[str, Any]
    sources: Tuple[ComponentSource, ...]
    evidence_level: str
    limitations: Tuple[str, ...]
    notes: str
    tags: Tuple[str, ...] = ()

    @property
    def component_kind(self) -> str:
        return "topology_bundle"


def _gb(value: float) -> int:
    return int(value * 1_000_000_000)


def _tb(value: float) -> int:
    return int(value * 1_000_000_000_000)


def _source(
    title: str,
    url: str,
    evidence_level: str,
    *,
    publisher: str,
    published_at: str = "",
    accessed_at: str = "2026-08-27",
) -> ComponentSource:
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError("未知的组件来源证据等级")
    return ComponentSource(
        title,
        url,
        evidence_level,
        publisher,
        published_at,
        accessed_at,
    )


def _source_metadata(sources: Sequence[ComponentSource]) -> Tuple[Dict[str, str], ...]:
    return tuple(source.to_metadata() for source in sources)


def _port_metadata(extras: Mapping[str, Any] = {}) -> Dict[str, Any]:
    return {
        "capability_units": dict(PORT_CAPABILITY_UNITS),
        **dict(extras),
    }


def _link_metadata(extras: Mapping[str, Any] = {}) -> Dict[str, Any]:
    return {
        "capability_units": dict(LINK_CAPABILITY_UNITS),
        **dict(extras),
    }


def _component_metadata(
    preset_id: str,
    *,
    technology: Mapping[str, Any],
    sources: Sequence[ComponentSource],
    evidence_level: str,
    limitations: Sequence[str],
    notes: str,
    measurement_basis: str,
    extras: Mapping[str, Any] = {},
) -> Dict[str, Any]:
    metadata = {
        "preset": {
            "id": preset_id,
            "catalog_version": CATALOG_VERSION,
            "usage_hint": USAGE_HINT,
        },
        "technology": dict(technology),
        "sources": list(_source_metadata(sources)),
        "evidence_level": evidence_level,
        "applicability_limitations": list(limitations),
        "applicability": list(limitations),
        "measurement_basis": measurement_basis,
        "capability_units": dict(CAPABILITY_UNITS),
        "revision": "catalog-{}".format(CATALOG_VERSION),
        "value_scope": "单组件模板",
        "conditions": [],
        "derived_formula": "",
        "expires_at": "",
        "supersedes": [],
        "notes": notes,
    }
    metadata.update(dict(extras))
    return metadata


# Runtime profiles are deliberately kept as plain JSON-compatible objects here.
# The web UI can copy one into ``profiles.components`` and remap resource IDs
# without importing the Python cost-model classes.  ``*_parameter_basis`` is a
# field-level audit trail: public product/standard values are separated from
# editable analytical defaults (especially latency and efficiency).
def _hbm_cost_profile_template(
    component_id: str,
    bandwidth_gbps: float,
    *,
    read_latency_ns: float = 40.0,
    write_latency_ns: float = 40.0,
    transaction_bytes: int = 256,
    max_outstanding_requests: int = 32,
    parallel_lanes: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    bandwidth_gb_s = bandwidth_gbps / 8.0
    profile = {
        "bandwidth_gb_s": bandwidth_gb_s,
        "efficiency": 1.0,
        "energy_pj_per_byte": 4.0,
        "resource_id": "{}.hbm_fabric".format(component_id),
        "read_latency_ns": read_latency_ns,
        "write_latency_ns": write_latency_ns,
        "transaction_bytes": transaction_bytes,
        "max_outstanding_requests": max_outstanding_requests,
        "parallel_lanes": parallel_lanes,
        "read_bandwidth_gb_s": bandwidth_gb_s,
        "write_bandwidth_gb_s": bandwidth_gb_s,
    }
    basis = {
        "bandwidth_gb_s": "component.read_bandwidth_gbps / 8; public interface speed and width",
        "read_bandwidth_gb_s": "component.read_bandwidth_gbps / 8; symmetric peak envelope",
        "write_bandwidth_gb_s": "component.write_bandwidth_gbps / 8; symmetric peak envelope",
        "efficiency": "A_ANALYTICAL editable derating default; no vendor workload efficiency claimed",
        "energy_pj_per_byte": "A_ANALYTICAL editable energy default; no product measurement claimed",
        "resource_id": "derived from the materialized component ID",
        "read_latency_ns": "A_ANALYTICAL editable HBM service-latency default; JEDEC does not specify end-to-end controller latency",
        "write_latency_ns": "A_ANALYTICAL editable HBM service-latency default; JEDEC does not specify end-to-end controller latency",
        "transaction_bytes": "A_ANALYTICAL simulator transaction granularity",
        "max_outstanding_requests": "A_ANALYTICAL simulator queue-overlap default",
        "parallel_lanes": "A_ANALYTICAL aggregate DRAM service-lane parameter",
    }
    return profile, basis


def _host_memory_cost_profile_template(
    component_id: str,
    bandwidth_gbps: float,
    *,
    name: str,
    read_latency_ns: float = 100.0,
    write_latency_ns: float = 100.0,
    transaction_bytes: int = 256,
    max_outstanding_requests: int = 32,
    parallel_lanes: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    bandwidth_gb_s = bandwidth_gbps / 8.0
    profile = {
        "bandwidth_gb_s": bandwidth_gb_s,
        "efficiency": 1.0,
        "energy_pj_per_byte": 12.0,
        "resource_id": "{}.memory".format(component_id),
        "name": name,
        "read_latency_ns": read_latency_ns,
        "write_latency_ns": write_latency_ns,
        "transaction_bytes": transaction_bytes,
        "max_outstanding_requests": max_outstanding_requests,
        "parallel_lanes": parallel_lanes,
        "read_bandwidth_gb_s": bandwidth_gb_s,
        "write_bandwidth_gb_s": bandwidth_gb_s,
    }
    basis = {
        "bandwidth_gb_s": "component.read_bandwidth_gbps / 8; vendor per-Grace aggregate",
        "read_bandwidth_gb_s": "component.read_bandwidth_gbps / 8; symmetric analytical envelope",
        "write_bandwidth_gb_s": "component.write_bandwidth_gbps / 8; symmetric analytical envelope",
        "efficiency": "A_ANALYTICAL editable memory-controller derating default",
        "energy_pj_per_byte": "A_ANALYTICAL editable host-memory energy default",
        "resource_id": "derived from the materialized component ID",
        "name": "derived from product family",
        "read_latency_ns": "A_ANALYTICAL editable CPU-visible LPDDR service-latency default; product page does not publish end-to-end latency",
        "write_latency_ns": "A_ANALYTICAL editable CPU-visible LPDDR service-latency default; product page does not publish end-to-end latency",
        "transaction_bytes": "A_ANALYTICAL simulator transaction granularity",
        "max_outstanding_requests": "A_ANALYTICAL simulator queue-overlap default",
        "parallel_lanes": "A_ANALYTICAL aggregate memory service-lane parameter",
    }
    return profile, basis


def _gpu_cost_profile_template(
    component_id: str,
    *,
    sm_count: int = 132,
    frequency_ghz: float = 1.98,
    peak_bf16_tflops: float = 989.5,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    # The structural tensor-core cycle value is calibrated to the published
    # dense BF16 peak, while the other issue/cache quantities remain editable
    # analytical defaults.  It is not a claim about a vendor's microcode.
    operations_per_mma = 2 * 16 * 16 * 16
    cycles_per_mma = (
        sm_count * 4 * frequency_ghz * operations_per_mma
        / (peak_bf16_tflops * 1000.0)
    )
    profile = {
        "tensor_core": {
            "sm_count": sm_count,
            "tensor_cores_per_sm": 4,
            "frequency_ghz": frequency_ghz,
            "mma_m": 16,
            "mma_n": 16,
            "mma_k": 16,
            "cycles_per_mma": cycles_per_mma,
            "supported_dtypes": ["fp16", "bf16", "int8"],
            # peak_bf16_tflops is the dense BF16 baseline; Hopper's dense
            # FP16 rate is equal and dense INT8 is 2x that baseline.
            "dtype_throughput_scale": {"fp16": 1.0, "bf16": 1.0, "int8": 2.0},
            "resource_id": "{}.tensor_core".format(component_id),
        },
        "cache_hierarchy": {
            "levels": [
                {
                    "name": "l1_shared",
                    "capacity_bytes": 30 * 1024 * 1024,
                    "line_bytes": 128,
                    "hit_latency_ns": 20.0,
                    "bandwidth_gb_s": 24000.0,
                    "associativity": 16,
                    "banks": sm_count * 32,
                    "read_ports": 2,
                    "write_ports": 1,
                    "max_outstanding": 32,
                    "energy_pj_per_byte": 0.15,
                    "resource_id": "{}.l1_shared".format(component_id),
                },
                {
                    "name": "l2",
                    "capacity_bytes": 50 * 1024 * 1024,
                    "line_bytes": 128,
                    "hit_latency_ns": 120.0,
                    "bandwidth_gb_s": 12000.0,
                    "associativity": 16,
                    "banks": 128,
                    "read_ports": 2,
                    "write_ports": 1,
                    "max_outstanding": 128,
                    "energy_pj_per_byte": 0.6,
                    "resource_id": "{}.l2".format(component_id),
                },
            ],
            "write_back": True,
            "write_allocate": True,
        },
        "scalar_lanes_per_sm": 128,
        "scalar_ops_per_cycle": 1.0,
        "reduction_ops_per_cycle_per_sm": 64.0,
        "special_function_units_per_sm": 16,
        "special_function_ops_per_cycle": 1.0,
        "host_gemm_offload": None,
        "host_recurrent_offload": None,
        "quantized_matmul_capabilities": [],
        "occupancy": 0.85,
        "attainable_efficiency": 0.65,
        "kernel_launch_ns": 1000.0,
        "tensor_energy_pj_per_op": 0.2,
        "scalar_energy_pj_per_op": 0.35,
        "special_function_energy_pj_per_op": 1.2,
        "launch_energy_pj": 10000.0,
        "scalar_resource_id": "{}.scalar".format(component_id),
        "special_function_resource_id": "{}.sfu".format(component_id),
        "launch_resource_id": "{}.frontend".format(component_id),
        "default_tensor_dtype": "bf16",
        "name": "{}-gpu-profile".format(component_id),
    }
    basis = {
        "tensor_core.sm_count": "S2 public Hopper implementation reference; verify active SM count for the exact SKU",
        "tensor_core.frequency_ghz": "A_ANALYTICAL editable clock reference; the cited product pages do not guarantee one boost clock",
        "tensor_core.cycles_per_mma": "derived to reproduce published dense BF16 peak; analytical structural calibration",
        "tensor_core.tensor_cores_per_sm": "S2_VENDOR_DECLARED architecture-level count",
        "tensor_core.supported_dtypes": "S2_VENDOR_DECLARED architecture capability",
        "cache_hierarchy": "A_ANALYTICAL editable cache/issue defaults; product pages do not publish all simulator fields",
        "occupancy": "A_ANALYTICAL editable scheduler default",
        "attainable_efficiency": "A_ANALYTICAL editable workload efficiency default",
        "kernel_launch_ns": "A_ANALYTICAL editable launch-latency default",
        "*_energy": "A_ANALYTICAL editable energy defaults; no application measurement claimed",
        "resource_id": "derived from the materialized component ID",
    }
    return profile, basis


def _cpu_cost_profile_template(
    component_id: str,
    *,
    core_count: int = 72,
    frequency_ghz: float = 3.0,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    profile = {
        "pipeline": {
            "core_count": core_count,
            "frequency_ghz": frequency_ghz,
            "simd_width_bits": 128,
            "decode_width": 4,
            "issue_width": 8,
            "retire_width": 8,
            "vector_fma_units_per_core": 4,
            "vector_alu_units_per_core": 4,
            "load_units_per_core": 2,
            "store_units_per_core": 1,
            "branch_units_per_core": 2,
            "special_function_units_per_core": 1,
            "special_function_cycles_per_vector": 12.0,
            "reorder_buffer_entries": 256,
            "load_store_queue_entries": 96,
            "memory_level_parallelism": 16,
            "branch_mispredict_ns": 5.0,
            "resource_id": "{}.pipeline".format(component_id),
        },
        "cache_hierarchy": {
            "levels": [
                {
                    "name": "l1d",
                    "capacity_bytes": core_count * 64 * 1024,
                    "line_bytes": 64,
                    "hit_latency_ns": 1.0,
                    "bandwidth_gb_s": 3000.0,
                    "associativity": 4,
                    "banks": core_count * 8,
                    "read_ports": 2,
                    "write_ports": 1,
                    "max_outstanding": 16,
                    "energy_pj_per_byte": 0.2,
                    "resource_id": "{}.l1d".format(component_id),
                },
                {
                    "name": "l2",
                    "capacity_bytes": core_count * 1 * 1024 * 1024,
                    "line_bytes": 64,
                    "hit_latency_ns": 4.0,
                    "bandwidth_gb_s": 1500.0,
                    "associativity": 16,
                    "banks": core_count * 8,
                    "read_ports": 2,
                    "write_ports": 1,
                    "max_outstanding": 32,
                    "energy_pj_per_byte": 0.8,
                    "resource_id": "{}.l2".format(component_id),
                },
                {
                    "name": "l3",
                    "capacity_bytes": 117 * 1024 * 1024,
                    "line_bytes": 64,
                    "hit_latency_ns": 18.0,
                    "bandwidth_gb_s": 800.0,
                    "associativity": 16,
                    "banks": 64,
                    "read_ports": 2,
                    "write_ports": 1,
                    "max_outstanding": 64,
                    "energy_pj_per_byte": 2.0,
                    "resource_id": "{}.l3".format(component_id),
                },
            ],
            "write_back": True,
            "write_allocate": True,
        },
        "quantized_dot_capabilities": [],
        "attainable_efficiency": 0.72,
        "dispatch_ns": 80.0,
        "gemm_energy_pj_per_op": 1.5,
        "elementwise_energy_pj_per_op": 1.0,
        "reduction_energy_pj_per_op": 1.2,
        "special_function_energy_pj_per_op": 4.0,
        "dispatch_energy_pj": 800.0,
        "name": "{}-cpu-profile".format(component_id),
    }
    basis = {
        "pipeline.core_count": "S2_VENDOR_DECLARED NVIDIA Grace Arm Neoverse V2 core count",
        "pipeline.frequency_ghz": "A_ANALYTICAL editable clock assumption; the cited Grace public pages do not guarantee one SKU-wide frequency",
        "pipeline.simd_width_bits": "S2_VENDOR_DECLARED 128-bit SVE2",
        "pipeline.vector_fma_units_per_core": "S2 architecture fact (4 × 128-bit SVE2 units/core) mapped to an editable analytical FMA-unit assumption",
        "pipeline.vector_alu_units_per_core": "S2 architecture fact (4 × 128-bit SVE2 units/core) mapped to an editable analytical ALU-unit assumption",
        "pipeline.*": "A_ANALYTICAL editable issue/queue defaults; product pages do not publish all simulator fields",
        "cache_hierarchy.levels[0].capacity_bytes": "S2_VENDOR_DECLARED/architecture reference L1D class",
        "cache_hierarchy.levels[1].capacity_bytes": "S2_VENDOR_DECLARED/architecture reference private L2 class",
        "cache_hierarchy.levels[2].capacity_bytes": "S2_VENDOR_DECLARED Grace system-cache capacity",
        "cache_hierarchy": "A_ANALYTICAL editable latency/bandwidth/port defaults where exact implementation detail is not public",
        "*_energy": "A_ANALYTICAL editable energy defaults; no application measurement claimed",
        "resource_id": "derived from the materialized component ID",
    }
    return profile, basis


def _cim_cost_profile_template(component_id: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    profile = {
        "array_count": 512,
        "p_m": 1,
        "p_k": 128,
        "p_n": 128,
        "frequency_ghz": 1.0,
        "input_parallel_bits": 4,
        "weight_parallel_bits": 4,
        "cycles_per_eval": 1,
        "weight_capacity_bytes": 512 * 1024 * 1024,
        "max_m_replication": 4,
        "load_bandwidth_gb_s": 512.0,
        "activation_bandwidth_gb_s": 1024.0,
        "output_bandwidth_gb_s": 1024.0,
        "noc_bandwidth_gb_s": 2048.0,
        "accumulator_outputs_per_cycle": 4096.0,
        "peripheral_elements_per_cycle": 4096.0,
        "load_latency_ns": 20.0,
        "noc_hop_latency_ns": 2.0,
        "noc_reduce_fan_in": 4,
        "peripheral_latency_ns": 5.0,
        "accumulator_bits": 32,
        "accumulator_guard_bits": 0,
        "supported_activation_bits": [1, 2, 4, 8, 16],
        "supported_weight_bits": [1, 2, 4, 8, 16],
        "eval_energy_pj": 1.0,
        "load_energy_pj_per_byte": 0.5,
        "activation_energy_pj_per_byte": 0.2,
        "output_energy_pj_per_byte": 0.2,
        "noc_energy_pj_per_byte": 0.1,
        "accumulator_energy_pj_per_op": 0.05,
        "peripheral_energy_pj_per_element": 0.1,
        "array_resource_id": "{}.array".format(component_id),
        "load_resource_id": "{}.load".format(component_id),
        "activation_resource_id": "{}.activation".format(component_id),
        "noc_resource_id": "{}.noc".format(component_id),
        "accumulator_resource_id": "{}.accumulator".format(component_id),
        "peripheral_resource_id": "{}.peripheral".format(component_id),
        "name": "{}-digital-sram-cim-profile".format(component_id),
        "arithmetic_mode": "integer_bit_slice",
        "weight_conversion_mode": "disabled",
        "weight_decode_elements_per_ns": 0.0,
        "conversion_scratch_capacity_bytes": 0,
        "activation_fp32_to_fp16_elements_per_ns": 0.0,
        "conversion_contract_basis": "",
        "conversion_read_energy_pj_per_byte": 0.0,
        "conversion_write_energy_pj_per_byte": 0.0,
    }
    basis = {
        "*": "A_ANALYTICAL internal simulator reference; not a fabricated product specification",
        "weight_capacity_bytes": "component capacity_bytes and internal analysis reference",
        "resource_id": "derived from the materialized component ID",
    }
    return profile, basis


def _hbm_ports(
    *,
    generation: str,
    channels: int,
    bandwidth_gbps: float,
    protocol: str = "HBM",
) -> Tuple[PortSpec, ...]:
    return (
        PortSpec(
            port_id="host",
            protocol=protocol,
            role="device",
            version=generation,
            lanes=channels,
            bandwidth_gbps=bandwidth_gbps,
            metadata=_port_metadata({"channels": channels}),
        ),
    )


def _hbm_preset(
    preset_id: str,
    name: str,
    *,
    generation: str,
    pin_speed_gbps: float,
    io_bits: int,
    channels: int,
    capacity_gb: float,
    bandwidth_gbps: float,
    evidence_level: str,
    limitations: Sequence[str],
    notes: str,
    sources: Sequence[ComponentSource],
    protocol: str = "HBM",
    family: str = "HBM",
) -> ComponentPresetDefinition:
    component_id = preset_id.replace("-", "_")
    cost_profile_template, cost_profile_parameter_basis = _hbm_cost_profile_template(
        component_id,
        bandwidth_gbps,
    )
    technology = {
        "generation": generation,
        "pin_speed_gbps": pin_speed_gbps,
        "interface_bits": io_bits,
        "channels": channels,
        "capacity_gb": capacity_gb,
        "kind_policy": "{} 介质代际写入 metadata；ComponentSpec.kind 固定保持为 hbm，以复用 GPU 本地显存服务模型。".format(generation),
        "interface_protocol": protocol,
    }
    component = ComponentSpec(
        component_id=component_id,
        kind="hbm",
        cost_profile_id=_default_cost_profile_id("hbm"),
        ports=_hbm_ports(
            generation=generation,
            channels=channels,
            bandwidth_gbps=bandwidth_gbps,
            protocol=protocol,
        ),
        capacity_bytes=_gb(capacity_gb),
        read_bandwidth_gbps=bandwidth_gbps,
        write_bandwidth_gbps=bandwidth_gbps,
        metadata=_component_metadata(
            preset_id,
            technology=technology,
            sources=sources,
            evidence_level=evidence_level,
            limitations=limitations,
            notes=notes,
            measurement_basis="由公开速度档与接口宽度换算为 IR 内部带宽字段；详情页按十进制 GB/s/TB/s 展示。",
            extras={
                "revision": {
                    "HBM3": "JESD238B.01",
                    "HBM4": "JESD270-4A",
                }.get(generation, "{}-{}".format(generation, CATALOG_VERSION)),
                "value_scope": "单个 {} 本地显存介质的峰值读写带宽模板".format(generation),
                "conditions": [
                    "十进制带宽单位",
                    "未扣除 ECC、控制器开销、封装损耗或热降频",
                ],
                "derived_formula": "公开速度档乘以接口宽度，结果写入 IR 带宽字段",
                "expires_at": "2027-08-22",
                "read_latency_ns": cost_profile_template["read_latency_ns"],
                "write_latency_ns": cost_profile_template["write_latency_ns"],
                "transfer_granularity_bytes": cost_profile_template["transaction_bytes"],
                "max_outstanding_requests": cost_profile_template["max_outstanding_requests"],
                "cost_profile_template": cost_profile_template,
                "cost_profile_parameter_basis": cost_profile_parameter_basis,
                "cost_profile_key": "hbm",
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family=family,
        component=component,
        sources=tuple(sources),
        evidence_level=evidence_level,
        limitations=tuple(limitations),
        notes=notes,
        tags=("memory", generation.lower()),
    )


def _gddr_preset(
    preset_id: str,
    name: str,
    *,
    generation: str,
    data_rate_gbps: float,
    interface_bits: int,
    capacity_gb: float,
    evidence_level: str,
    limitations: Sequence[str],
    notes: str,
    sources: Sequence[ComponentSource],
    family: str = "GDDR Graphics Memory",
    signal_encoding: str = "NRZ",
) -> ComponentPresetDefinition:
    """Build a first-class GDDR component template.

    ``data_rate_gbps`` is the effective per-pin data rate.  The resulting
    component bandwidth is stored in the IR's one-way Gb/s field exactly once;
    no additional DDR/PAM multiplier is applied.
    """
    component_id = preset_id.replace("-", "_")
    bandwidth_gbps = data_rate_gbps * interface_bits
    # DramCore reserves one command interval per data lane.  Keep that
    # reservation from becoming an accidental second bandwidth ceiling: a
    # lane carrying ``data_width_bits`` pins at ``data_rate_gbps`` Gb/s needs
    # this long to move one burst.  The value is a scheduling budget derived
    # from the declared interface rate, not a JEDEC tCCD measurement.
    burst_bytes = 64
    data_lanes = max(1, interface_bits // 32)
    data_width_bits = max(1, interface_bits // data_lanes)
    lane_bandwidth_gb_s = data_rate_gbps * data_width_bits / 8.0
    burst_interval_ns = burst_bytes / lane_bandwidth_gb_s
    cost_profile_template, cost_profile_parameter_basis = _hbm_cost_profile_template(
        component_id,
        bandwidth_gbps,
        read_latency_ns=35.0,
        write_latency_ns=35.0,
        transaction_bytes=256,
        max_outstanding_requests=64,
        parallel_lanes=max(1, interface_bits // 32),
    )
    cost_profile_template["resource_id"] = "{}.gddr_fabric".format(component_id)
    technology = {
        "generation": generation,
        "memory_type": generation,
        "data_rate_gbps": data_rate_gbps,
        "interface_bits": interface_bits,
        "interface_bandwidth_gb_s": bandwidth_gbps / 8.0,
        "signal_encoding": signal_encoding,
        "kind_policy": "GDDR 代际保留在 technology；ComponentSpec.kind 固定为 gddr。",
        "interface_protocol": generation,
    }
    component = ComponentSpec(
        component_id=component_id,
        kind="gddr",
        cost_profile_id=_default_cost_profile_id("gddr"),
        ports=(
            PortSpec(
                port_id="host",
                protocol="GDDR",
                role="device",
                version=generation,
                lanes=interface_bits,
                bandwidth_gbps=bandwidth_gbps,
                metadata=_port_metadata({
                    "interface_width_bits": interface_bits,
                    "data_rate_gbps": data_rate_gbps,
                    "bandwidth_scope": "shared_component_total_one_way",
                }),
            ),
        ),
        capacity_bytes=_gb(capacity_gb),
        read_bandwidth_gbps=bandwidth_gbps,
        write_bandwidth_gbps=bandwidth_gbps,
        metadata=_component_metadata(
            preset_id,
            technology=technology,
            sources=sources,
            evidence_level=evidence_level,
            limitations=limitations,
            notes=notes,
            measurement_basis="有效引脚数据率 × 总接口位宽 ÷ 8，结果按十进制 GB/s 写入 IR。",
            extras={
                "revision": "{}-{}".format(generation, CATALOG_VERSION),
                "value_scope": "单套 GDDR 显存子系统的峰值读写带宽模板",
                "conditions": ["十进制容量与带宽单位", "读写共享同一数据通路，不将方向带宽相加"],
                "derived_formula": "data_rate_gbps × interface_bits = IR one-way Gb/s",
                "expires_at": "2027-10-05",
                "read_latency_ns": cost_profile_template["read_latency_ns"],
                "write_latency_ns": cost_profile_template["write_latency_ns"],
                "transfer_granularity_bytes": cost_profile_template["transaction_bytes"],
                "max_outstanding_requests": cost_profile_template["max_outstanding_requests"],
                "cost_profile_template": cost_profile_template,
                "cost_profile_parameter_basis": cost_profile_parameter_basis,
                "cost_profile_key": "gddr",
                "physical_memory_config": {
                    "kind": "GDDR",
                    "generation": generation,
                    "channels": 1,
                    "data_lanes": data_lanes,
                    "data_width_bits": data_width_bits,
                    "data_rate_mt_s": data_rate_gbps * 1000.0,
                    "interface_bandwidth_gb_s": bandwidth_gbps / 8.0,
                    "stacks": 1,
                    "dies_per_stack": 1,
                    "ranks_per_channel": 1,
                    "bank_groups_per_rank": 4,
                    "banks_per_group": 4,
                    "rows_per_bank": 131072,
                    "row_bytes": 8192,
                    "burst_bytes": burst_bytes,
                    "interleave_bytes": burst_bytes,
                    "open_ns": 14.0,
                    "close_ns": 14.0,
                    "read_latency_ns": 35.0,
                    "write_latency_ns": 35.0,
                    "burst_interval_ns": burst_interval_ns,
                    "max_outstanding_requests": 64,
                    "capacity_bytes": _gb(capacity_gb),
                    "metadata": {
                        "parameter_basis": "A_ANALYTICAL" if evidence_level == A_ANALYTICAL else evidence_level,
                        "geometry_scope": "synthetic_equivalent_gddr_subsystem",
                        "bandwidth_input_mode": "explicit_aggregate_interface",
                        "pin_data_rate_gbps": data_rate_gbps,
                        "signal_encoding": signal_encoding,
                        "command_interval_model": "per_lane_burst_payload_budget",
                        "command_interval_basis": "burst_bytes / (pin_data_rate_gbps * data_width_bits / 8); analytical scheduling budget, not JEDEC timing",
                        "lane_bandwidth_gb_s": lane_bandwidth_gb_s,
                        "burst_interval_ns": burst_interval_ns,
                    },
                },
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family=family,
        component=component,
        sources=tuple(sources),
        evidence_level=evidence_level,
        limitations=tuple(limitations),
        notes=notes,
        tags=("memory", "gddr", generation.lower()),
    )


def _hbm_product_slice_preset(
    preset_id: str,
    name: str,
    *,
    generation: str,
    visible_capacity_gb: float,
    raw_capacity_gb: float,
    bandwidth_gbps: float,
    product_stack_count: int,
    unit_count_status: str,
    unit_count_formula: str,
    product_family: str,
    sources: Sequence[ComponentSource],
    evidence_level: str = S2_VENDOR_DECLARED,
    analytical_pim: bool = False,
) -> ComponentPresetDefinition:
    """Create a per-stack product slice derived from public product totals."""

    limitations = (
        "该模板是产品总可见容量与总显存带宽按物理堆叠数守恒拆分的单颗切片，不代表厂商逐堆叠公布了独立可用值。",
        "原始堆叠容量与产品可见容量分别记录；未扣除 ECC、刷新、控制器、封装或热限制。",
    ) + (
        ("HBM-PIM 数值是分析参考，不对应通用量产部件。",)
        if analytical_pim
        else ()
    )
    source_basis = "{} product aggregate and stack-count provenance".format(
        product_family
    )
    component_id = preset_id.replace("-", "_")
    cost_profile_template, cost_profile_parameter_basis = _hbm_cost_profile_template(
        component_id,
        bandwidth_gbps,
    )
    component = ComponentSpec(
        component_id=component_id,
        kind="hbm",
        cost_profile_id=_default_cost_profile_id("hbm"),
        ports=(
            PortSpec(
                port_id="host",
                protocol="HBM",
                role="device",
                version=generation,
                lanes=16,
                bandwidth_gbps=bandwidth_gbps,
                metadata=_port_metadata({
                    "channels": 16,
                    "physical_channel_total": 16,
                    "channels_per_stack": 16,
                    "bandwidth_scope": "derived_product_stack_slice",
                }),
            ),
        ),
        capacity_bytes=_gb(visible_capacity_gb),
        read_bandwidth_gbps=bandwidth_gbps,
        write_bandwidth_gbps=bandwidth_gbps,
        metadata=_component_metadata(
            preset_id,
            technology={
                "generation": generation,
                "product_family": product_family,
                "visible_capacity_gb": visible_capacity_gb,
                "raw_capacity_gb": raw_capacity_gb,
                "product_stack_count": product_stack_count,
                "kind_policy": "HBM 代际只写入 metadata；ComponentSpec.kind 固定保持为 hbm。",
            },
            sources=sources,
            evidence_level=evidence_level,
            limitations=limitations,
            notes="用于物理 HBM 拓扑展开的单颗产品切片模板。",
            measurement_basis="产品总可见容量和单向显存带宽按堆叠数等分；原始容量另存，不把推导值伪装成逐堆叠厂商规格。",
            extras={
                "revision": "{}-product-slice".format(generation),
                "value_scope": "单颗物理 HBM 堆叠的产品可见切片",
                "conditions": ["十进制容量与带宽单位", unit_count_status],
                "derived_formula": unit_count_formula,
                "physical_composition": {
                    "simulator_representation": "single_physical_unit_node",
                    "simulator_node_count": 1,
                    "physical_unit_kind": "HBM_stack",
                    "physical_unit_count": 1,
                    "physical_unit_count_status": "explicit_physical_node",
                    "known_multiple": product_stack_count > 1,
                    "controller_component_id": "",
                    "memory_subsystem_id": "standalone_component_preset",
                    "unit_index": 0,
                    "unit_count_in_product": product_stack_count,
                    "unit_count_status": unit_count_status,
                    "unit_count_formula": unit_count_formula,
                    "unit_capacity_bytes": _gb(visible_capacity_gb),
                    "product_total_capacity_bytes": _gb(
                        visible_capacity_gb * product_stack_count
                    ),
                    "unit_raw_capacity_bytes": _gb(raw_capacity_gb),
                    "product_total_raw_capacity_bytes": _gb(
                        raw_capacity_gb * product_stack_count
                    ),
                    "capacity_accounting": (
                        "product_visible_and_raw_capacity_recorded_separately"
                        if raw_capacity_gb != visible_capacity_gb
                        else "product_visible_capacity_equals_modeled_raw_capacity"
                    ),
                    "unit_bandwidth_gbps": bandwidth_gbps,
                    "product_total_bandwidth_gbps": bandwidth_gbps
                    * product_stack_count,
                    "component_preset_id": preset_id,
                    "component_preset_status": "self",
                    "source_basis": source_basis,
                },
                "parameter_basis": {
                    "capacity_bytes": "product visible capacity / unit_count_in_product",
                    "read_bandwidth_gbps": "product total bandwidth / unit_count_in_product",
                    "write_bandwidth_gbps": "symmetric analytical envelope from the same per-stack interface",
                    "unit_count": unit_count_status,
                    "unit_count_formula": unit_count_formula,
                    "source_basis": source_basis,
                },
                "provenance": {
                    "value_status": (
                        "analytical_reference"
                        if analytical_pim
                        else "derived_per_physical_stack"
                    ),
                    "unit_count_status": unit_count_status,
                    "source_basis": source_basis,
                },
                "read_latency_ns": cost_profile_template["read_latency_ns"],
                "write_latency_ns": cost_profile_template["write_latency_ns"],
                "transfer_granularity_bytes": cost_profile_template["transaction_bytes"],
                "max_outstanding_requests": cost_profile_template["max_outstanding_requests"],
                "cost_profile_template": cost_profile_template,
                "cost_profile_parameter_basis": cost_profile_parameter_basis,
                "cost_profile_key": "hbm",
                "expires_at": "2027-08-23",
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family="HBM Product Slice",
        component=component,
        sources=tuple(sources),
        evidence_level=evidence_level,
        limitations=limitations,
        notes="物理 HBM 拓扑展开使用的单颗产品切片；推导与来源状态可机读。",
        tags=("memory", generation.lower(), "physical-stack", "product-slice"),
    )
def _gpu_hbm_controller_ports(
    *,
    generation: str,
    stack_count: int,
    aggregate_bandwidth_gbps: float,
) -> Tuple[PortSpec, ...]:
    per_stack = aggregate_bandwidth_gbps / stack_count
    ports = [
        PortSpec(
            port_id="hbm{}".format(index),
            protocol="HBM",
            role="controller",
            version=generation,
            lanes=16,
            bandwidth_gbps=per_stack,
            metadata=_port_metadata({
                "expected_stack_index": index,
                "bandwidth_scope": "per_stack_controller_slice",
                "physical_channel_total": 16,
                "channels_per_stack": 16,
            }),
        )
        for index in range(stack_count)
    ]
    ports.extend(
        [
            PortSpec(
                port_id="nvlink0",
                protocol="NVLink",
                role="endpoint",
                version="4.0",
                lanes=18,
                bandwidth_gbps=3600.0,
                metadata=_port_metadata({
                    "bandwidth_direction": "unidirectional",
                    "source_display_value": "每方向 450 GB/s；官方双向聚合 900 GB/s",
                    "aggregate_bidirectional_gbps": 7200.0,
                    "aggregate_bidirectional_display_value": "900 GB/s",
                }),
            ),
            PortSpec(
                port_id="pcie0",
                protocol="PCIe",
                role="endpoint",
                version="5.0",
                lanes=16,
                bandwidth_gbps=512.0,
                payload="NVMe/CXL/control",
                metadata=_port_metadata({
                    "source_display_value": "每方向 64 GB/s；官方双向聚合 128 GB/s",
                    "aggregate_bidirectional_display_value": "128 GB/s",
                }),
            ),
        ]
    )
    return tuple(ports)


def _gpu_preset(
    preset_id: str,
    name: str,
    *,
    hbm_generation: str,
    hbm_stack_count: int,
    device_memory_gb: float,
    device_memory_bandwidth_gbps: float,
    bf16_dense_tflops: float,
    fp8_dense_tflops: float,
    fp8_sparse_tflops: float,
    bf16_sparse_tflops: float,
    tdp_watts: int,
    mig_profiles: str,
    sources: Sequence[ComponentSource],
) -> ComponentPresetDefinition:
    limitations = (
        "模板表示计算侧单 GPU；显存容量和 HBM 带宽写入 metadata，不自动创建或连接 HBM 组件。",
        "peak_ops_per_s 使用保守 dense BF16 Tensor Core 峰值；稀疏和 FP8 峰值仅作为条件化 metadata。",
        "L2/片上容量未在该页面完整建模，capacity_bytes 仅保留为计算侧本地缓存占位。",
    )
    notes = "NVIDIA Hopper SXM 计算侧模板；如果拓扑需要显式内存节点，请另外添加 HBM 堆叠预设。"
    component_id = preset_id.replace("-", "_")
    cost_profile_template, cost_profile_parameter_basis = _gpu_cost_profile_template(
        component_id,
        peak_bf16_tflops=bf16_dense_tflops,
    )
    component = ComponentSpec(
        component_id=component_id,
        kind="gpu",
        cost_profile_id=_default_cost_profile_id("gpu"),
        ports=_gpu_hbm_controller_ports(
            generation=hbm_generation,
            stack_count=hbm_stack_count,
            aggregate_bandwidth_gbps=device_memory_bandwidth_gbps,
        ),
        capacity_bytes=50 * 1024 * 1024,
        peak_ops_per_s=bf16_dense_tflops * 1_000_000_000_000,
        metadata=_component_metadata(
            preset_id,
            technology={
                "architecture": "NVIDIA Hopper",
                "memory_generation": hbm_generation,
                "memory_stack_count": hbm_stack_count,
                "device_memory_gb": device_memory_gb,
                "device_memory_bandwidth_gbps": device_memory_bandwidth_gbps,
                "bf16_tensor_dense_tflops": bf16_dense_tflops,
                "fp8_tensor_dense_tflops": fp8_dense_tflops,
                "fp8_tensor_sparse_tflops": fp8_sparse_tflops,
                "bf16_tensor_sparse_tflops": bf16_sparse_tflops,
                "peak_ops_condition": "dense BF16 Tensor Core, no sparsity",
                "tdp_watts": tdp_watts,
                "mig_profiles": mig_profiles,
            },
            sources=sources,
            evidence_level=S2_VENDOR_DECLARED,
            limitations=limitations,
            notes=notes,
            measurement_basis="采用厂商规格表中的 dense BF16 Tensor Core 峰值；显存带宽按十进制 TB/s 换算到 IR 带宽字段。",
            extras={
                "component_scope": "compute_side_only",
                "external_memory_modeled_in_metadata": True,
                "value_scope": "单个 SXM GPU 计算侧模板",
                "conditions": [
                    "dense BF16 不使用结构化稀疏",
                    "FP8 与稀疏峰值仅供对照，不写入 peak_ops_per_s",
                    "PCIe 端口按每方向 64 GB/s 建模；官方双向聚合值为 128 GB/s",
                    "NVLink 端口按每方向 450 GB/s 建模；900 GB/s 仅作为双向聚合展示值",
                ],
                "derived_formula": "公开 GB/s 或 TB/s 峰值乘以 8，写入 IR 带宽字段",
                "expires_at": "2027-08-22",
                "cost_profile_template": cost_profile_template,
                "cost_profile_parameter_basis": cost_profile_parameter_basis,
                "cost_profile_key": "gpu",
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family="NVIDIA Hopper GPU",
        component=component,
        sources=tuple(sources),
        evidence_level=S2_VENDOR_DECLARED,
        limitations=limitations,
        notes=notes,
        tags=("gpu", "hopper", hbm_generation.lower()),
    )


def _ssd_preset(
    preset_id: str,
    name: str,
    *,
    family: str,
    kind: str,
    capacity_bytes: int,
    read_gbps: float,
    write_gbps: float,
    interface_bandwidth_gbps: float,
    interface: str,
    protocol_version: str,
    lanes: int,
    endurance: str,
    limitations: Sequence[str],
    notes: str,
    sources: Sequence[ComponentSource],
    evidence_level: str,
    conditions: Sequence[str] = (),
) -> ComponentPresetDefinition:
    default_conditions = (
        "顺序读写峰值",
        "max_outstanding_requests 仅重叠启动延迟；不模拟完整 NVMe 队列调度、热降频或写放大",
    )
    high_io = kind == "high_io_ssd"
    analytical_transport = {
        "read_latency_ns": 25_000.0 if high_io else 80_000.0,
        "write_latency_ns": 40_000.0 if high_io else 100_000.0,
        "transfer_granularity_bytes": 4096,
        "max_outstanding_requests": 64 if high_io else 32,
        "dma_bandwidth_gbps": max(read_gbps, write_gbps),
        "dma_latency_ns": 1_200.0 if high_io else 2_000.0,
        "dma_energy_pj_per_byte": 0.0,
        "unknown_value_sentinels": {
            "dma_energy_pj_per_byte": "0.0 means unknown/not declared, not zero energy"
        },
        "storage_transport_parameter_basis": (
            "editable analytical controller/queue defaults; vendor evidence only covers "
            "the separately declared capacity and sequential media bandwidth"
        ),
    }
    component = ComponentSpec(
        component_id=preset_id.replace("-", "_"),
        kind=kind,
        cost_profile_id=_default_cost_profile_id(kind),
        ports=(
            PortSpec(
                port_id="pcie0",
                protocol="PCIe",
                role="endpoint",
                version=protocol_version,
                lanes=lanes,
                bandwidth_gbps=interface_bandwidth_gbps,
                payload="NVMe",
                metadata=_port_metadata(
                    {
                        "interface": interface,
                        "bandwidth_basis": "decoded_phy_line_rate",
                        "media_bandwidth_modeled_separately": True,
                    }
                ),
            ),
        ),
        capacity_bytes=capacity_bytes,
        read_bandwidth_gbps=read_gbps,
        write_bandwidth_gbps=write_gbps,
        metadata=_component_metadata(
            preset_id,
            technology={
                "media": "NAND flash",
                "interface": interface,
                "nvme": True,
                "endurance": endurance,
            },
            sources=sources,
            evidence_level=evidence_level,
            limitations=limitations,
            notes=notes,
            measurement_basis="read_gbps/write_gbps 已是 IR 十进制 Gb/s；调用方必须先把厂商 MB/s 或 GB/s 除/乘换算后传入；随机 IOPS 不直接折算。",
            extras={
                **analytical_transport,
                "value_scope": "单个 NVMe SSD 组件模板",
                "conditions": list(conditions or default_conditions),
                "derived_formula": "厂商 MB/s ÷ 1000 × 8 = IR Gb/s（或厂商 GB/s × 8）；具体换算保存在 vendor_parameter_provenance",
                "expires_at": "2027-08-22",
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family=family,
        component=component,
        sources=tuple(sources),
        evidence_level=evidence_level,
        limitations=tuple(limitations),
        notes=notes,
        tags=("storage", kind),
    )


JEDEC_HBM3_SOURCE = _source(
    "JEDEC JESD238B.01 High Bandwidth Memory DRAM (HBM3)",
    "https://www.jedec.org/standards-documents/docs/jesd238b01",
    S1_STANDARD,
    publisher="JEDEC",
)
JEDEC_HBM4_SOURCE = _source(
    "JEDEC JESD270-4A High Bandwidth Memory DRAM (HBM4)",
    "https://www.jedec.org/standards-documents/docs/jesd270-4a",
    S1_STANDARD,
    publisher="JEDEC",
)
SAMSUNG_HBM3_SOURCE = _source(
    "Samsung HBM3 product specifications",
    "https://semiconductor.samsung.com/dram/hbm/hbm3/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
)
SAMSUNG_HBM3E_SOURCE = _source(
    "Samsung HBM3E product specifications",
    "https://semiconductor.samsung.com/dram/hbm/hbm3e/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
)
SAMSUNG_HBM4_SOURCE = _source(
    "Samsung HBM4 product specifications",
    "https://semiconductor.samsung.com/dram/hbm/hbm4/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
)
SAMSUNG_HBM_SOURCE = _source(
    "Samsung HBM product family",
    "https://semiconductor.samsung.com/dram/hbm/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
)
MICRON_HBM3E_SOURCE = _source(
    "Micron HBM3E product specifications",
    "https://www.micron.com/products/memory/hbm/hbm3e",
    S2_VENDOR_DECLARED,
    publisher="Micron",
)
NVIDIA_H100_SOURCE = _source(
    "NVIDIA H100 GPU product specifications",
    "https://www.nvidia.com/en-us/data-center/h100/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
)
NVIDIA_H200_SOURCE = _source(
    "NVIDIA H200 GPU product specifications",
    "https://www.nvidia.com/en-us/data-center/h200/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
)
NVIDIA_GH200_SOURCE = _source(
    "NVIDIA Grace Hopper Superchip product specifications",
    "https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
)
NVIDIA_GB200_SOURCE = _source(
    "NVIDIA GB200 NVL72 reference architecture",
    "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
)
NVIDIA_B200_SOURCE = _source(
    "NVIDIA DGX B200 system specifications",
    "https://www.nvidia.com/en-us/data-center/dgx-b200/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
)
NVIDIA_BLACKWELL_SOURCE = _source(
    "NVIDIA Blackwell Architecture",
    "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
)
AMD_MI300A_SOURCE = _source(
    "AMD Instinct MI300A product specifications",
    "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html",
    S2_VENDOR_DECLARED,
    publisher="AMD",
)
AMD_MI300X_SOURCE = _source(
    "AMD Instinct MI300X product specifications",
    "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
    S2_VENDOR_DECLARED,
    publisher="AMD",
)
AMD_RYZEN_9_9950X3D_SOURCE = _source(
    "AMD Ryzen 9 9950X3D Desktop Processor specifications",
    "https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html",
    S2_VENDOR_DECLARED,
    publisher="AMD",
    accessed_at="2026-09-27",
)
NVIDIA_RTX_5080_SOURCE = _source(
    "NVIDIA GeForce RTX 5080 graphics card specifications",
    "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
    accessed_at="2026-09-27",
)
MICRON_GDDR_SOURCE = _source(
    "Micron GDDR graphics memory family overview",
    "https://www.micron.com/products/memory/graphics-memory",
    S2_VENDOR_DECLARED,
    publisher="Micron",
    accessed_at="2026-10-05",
)
MICRON_GDDR7_SOURCE = _source(
    "Micron GDDR7 product and technology overview",
    "https://www.micron.com/products/memory/graphics-memory/gddr7",
    S2_VENDOR_DECLARED,
    publisher="Micron",
    accessed_at="2026-10-05",
)
NVIDIA_BLACKWELL_ARCHITECTURE_PDF_SOURCE = _source(
    "NVIDIA RTX Blackwell GPU Architecture technical brief (GeForce RTX 5080 table)",
    "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
    accessed_at="2026-09-27",
)
INTEL_GAUDI3_SOURCE = _source(
    "Intel Gaudi 3 AI accelerator product specifications",
    "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html",
    S2_VENDOR_DECLARED,
    publisher="Intel",
)
SAMSUNG_HBM_PIM_SOURCE = _source(
    "Samsung HBM-PIM technology reference",
    "https://semiconductor.samsung.com/news-events/tech-blog/hbm-pim-cutting-edge-memory-technology-to-accelerate-next-generation-ai/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
)
SAMSUNG_PM1743_WHITE_PAPER_SOURCE = _source(
    "Samsung PM1743 white paper",
    "https://download.semiconductor.samsung.com/resources/white-paper/PM1743_White_Paper_240510.pdf",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
    published_at="2024-05-10",
)
SAMSUNG_PM1743_PRODUCT_SOURCE = _source(
    "Samsung PM1743 product page",
    "https://semiconductor.samsung.com/ssd/enterprise-ssd/pm1743/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
)
SOLIDIGM_D7_P5810_SOURCE = _source(
    "Solidigm D7-P5810 product brief",
    "https://www.solidigm.com/content/dam/solidigm/en/site/products/data-center/product-briefs/d7-5810/documents/solidigm-d7-p5810-product-brief.pdf",
    S2_VENDOR_DECLARED,
    publisher="Solidigm",
)
SK_HYNIX_HBF_SOURCE = _source(
    "SK hynix and Sandisk HBF OCP specification announcement",
    "https://news.skhynix.com/en/hbf-at-fms-2026/",
    S3_VENDOR_PREPRODUCTION,
    publisher="SK hynix",
    published_at="2026-08-04",
)
SANDISK_HBF_SOURCE = _source(
    "Sandisk High Bandwidth Flash fact sheet",
    "https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/collateral/company/Sandisk-HBF-Fact-Sheet.pdf",
    S3_VENDOR_PREPRODUCTION,
    publisher="Sandisk",
    published_at="2025-07",
)
INTERNAL_CIM_SOURCE = _source(
    "HeteroLLM Simulator digital SRAM-CIM analytical reference",
    "",
    A_ANALYTICAL,
    publisher="HeteroLLM Simulator",
)

# Curated vendor sources.  The bundled catalog is intentionally offline: the
# URL and access date are recorded so a reviewer can reproduce the values, but
# importing the simulator never depends on a live vendor site.
SAMSUNG_DDR5_32GB_SOURCE = _source(
    "Samsung DDR5 DRAM / 32GB UDIMM family",
    "https://semiconductor.samsung.com/dram/ddr/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
    accessed_at="2026-09-27",
)
SAMSUNG_HBM3E_OFFICIAL_SOURCE = _source(
    "Samsung HBM3E product specifications",
    "https://semiconductor.samsung.com/dram/hbm/hbm3e/",
    S2_VENDOR_DECLARED,
    publisher="Samsung Semiconductor",
    accessed_at="2026-09-27",
)
YMTC_ZHITAI_TIPRO9100_SOURCE = _source(
    "YMTC / ZhiTai TiPlus9100 official product specification (user name: Ti Pro9100)",
    "https://www.ymtc.com/cn/products/77.html?cat=44",
    S2_VENDOR_DECLARED,
    publisher="Yangtze Memory Technologies / ZhiTai",
    accessed_at="2026-09-27",
)
SK_HYNIX_HBF_OFFICIAL_SOURCE = _source(
    "SK hynix HBF standard specification announcement",
    "https://news.skhynix.com/en/hbf-at-fms-2026/",
    S3_VENDOR_PREPRODUCTION,
    publisher="SK hynix",
    published_at="2026-08-04",
    accessed_at="2026-09-27",
)
NVIDIA_B200_OFFICIAL_SOURCE = _source(
    "NVIDIA B200 Tensor Core GPU product specification",
    "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
    accessed_at="2026-09-27",
)
NVIDIA_DGX_B200_OFFICIAL_SOURCE = _source(
    "NVIDIA DGX B200 system specifications",
    "https://www.nvidia.com/en-us/data-center/dgx-b200/",
    S2_VENDOR_DECLARED,
    publisher="NVIDIA",
    accessed_at="2026-09-27",
)


def _grace_lpddr_preset(
    preset_id: str,
    name: str,
    *,
    product_family: str,
    bandwidth_gbps: float,
    source: ComponentSource,
) -> ComponentPresetDefinition:
    limitations = (
        "该节点是每颗 Grace 的主机内存子系统聚合，不表示单颗 LPDDR5X DRAM。",
        "物理 DRAM 数量未可靠披露，因此保持 unknown/null，不进行人为拆分。",
    )
    bandwidth_gb_per_s = bandwidth_gbps / 8.0
    component_id = preset_id.replace("-", "_")
    cost_profile_template, cost_profile_parameter_basis = _host_memory_cost_profile_template(
        component_id,
        bandwidth_gbps,
        name="{}-memory-profile".format(component_id),
    )
    component = ComponentSpec(
        component_id=component_id,
        kind="host_memory",
        cost_profile_id=_default_cost_profile_id("host_memory"),
        ports=(
            PortSpec(
                port_id="host",
                protocol="LPDDR5X",
                role="device",
                version=product_family,
                bandwidth_gbps=bandwidth_gbps,
                metadata=_port_metadata(
                    {"bandwidth_scope": "vendor_per_grace_aggregate"}
                ),
            ),
        ),
        capacity_bytes=_gb(480.0),
        read_bandwidth_gbps=bandwidth_gbps,
        write_bandwidth_gbps=bandwidth_gbps,
        metadata=_component_metadata(
            preset_id,
            technology={
                "generation": "LPDDR5X",
                "product_family": product_family,
                "capacity_gb": 480.0,
                "bandwidth_gbps": bandwidth_gbps,
            },
            sources=(source,),
            evidence_level=S2_VENDOR_DECLARED,
            limitations=limitations,
            notes="供 {} 架构预设关联的 Grace 主机内存聚合模板。".format(
                product_family
            ),
            measurement_basis=(
                "采用每颗 Grace 最高 480 GB 与 {:.0f} GB/s 的产品级聚合口径。".format(
                    bandwidth_gb_per_s
                )
            ),
            extras={
                "physical_composition": {
                    "simulator_representation": "aggregate_node",
                    "simulator_node_count": 1,
                    "physical_unit_kind": "LPDDR5X_device",
                    "physical_unit_count": None,
                    "physical_unit_count_status": "not_reliably_disclosed",
                    "known_multiple": True,
                    "component_preset_id": preset_id,
                    "source_basis": "vendor per-Grace host-memory subsystem aggregate",
                },
                "parameter_basis": {
                    "capacity_bytes": "vendor per-Grace aggregate up to 480 GB",
                    "bandwidth_gbps": (
                        "vendor per-Grace aggregate {:.0f} GB/s converted to {:.0f} Gb/s".format(
                            bandwidth_gb_per_s, bandwidth_gbps
                        )
                    ),
                },
                "value_scope": "每颗 Grace 的 LPDDR5X 主机内存聚合",
                "conditions": ["物理 DRAM 数量未知", "产品级聚合口径"],
                "derived_formula": "{:.0f} GB/s × 8 = {:.0f} Gb/s".format(
                    bandwidth_gb_per_s, bandwidth_gbps
                ),
                "expires_at": "2027-08-27",
                "read_latency_ns": cost_profile_template["read_latency_ns"],
                "write_latency_ns": cost_profile_template["write_latency_ns"],
                "transfer_granularity_bytes": cost_profile_template["transaction_bytes"],
                "max_outstanding_requests": cost_profile_template["max_outstanding_requests"],
                "cost_profile_template": cost_profile_template,
                "cost_profile_parameter_basis": cost_profile_parameter_basis,
                "cost_profile_key": "host_memory",
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family="Grace Host Memory Aggregate",
        component=component,
        sources=(source,),
        evidence_level=S2_VENDOR_DECLARED,
        limitations=limitations,
        notes="Grace 主机内存产品级聚合组件。",
        tags=("memory", "host-memory", "lpddr5x", "aggregate"),
    )


def _grace_cpu_preset(
    preset_id: str,
    name: str,
    *,
    product_family: str,
    gpu_port_count: int,
    memory_bandwidth_gbps: float,
    source: ComponentSource,
) -> ComponentPresetDefinition:
    limitations = (
        "厂商未声明可跨工作负载使用的 CPU 推理峰值，因此 peak_ops_per_s 保持 0.0 unknown sentinel；执行成本只认该组件经 cost_profile_id 显式绑定的 profiles.components.cpu registry 条目。",
        "CPU 节点不内嵌 LPDDR5X 容量；主机内存必须使用独立 host_memory 组件和显式链路。",
    )
    component_id = preset_id.replace("-", "_")
    cost_profile_template, cost_profile_parameter_basis = _cpu_cost_profile_template(
        component_id,
    )
    ports = tuple(
        PortSpec(
            port_id="gpu{}".format(index),
            protocol="NVLink-C2C",
            role="endpoint",
            version=product_family,
            bandwidth_gbps=3600.0,
            metadata=_port_metadata(
                {
                    "source_display_value": "每方向 450 GB/s；官方界面聚合展示 900 GB/s",
                    "bandwidth_basis": "symmetric_half_of_vendor_aggregate_interface",
                }
            ),
        )
        for index in range(gpu_port_count)
    ) + (
        PortSpec(
            port_id="memory",
            protocol="LPDDR5X",
            role="controller",
            version=product_family,
            bandwidth_gbps=memory_bandwidth_gbps,
            metadata=_port_metadata(
                {"bandwidth_scope": "vendor_per_grace_aggregate"}
            ),
        ),
    )
    component = ComponentSpec(
        component_id=component_id,
        kind="cpu",
        cost_profile_id=_default_cost_profile_id("cpu"),
        ports=ports,
        capacity_bytes=0,
        peak_ops_per_s=0.0,
        metadata=_component_metadata(
            preset_id,
            technology={
                "architecture": "Arm Neoverse V2",
                "core_count": 72,
                "core_count_semantics": "physical_core_count",
                "vector_isa": "4×128 位 SVE2",
                "product_family": product_family,
            },
            sources=(source,),
            evidence_level=S2_VENDOR_DECLARED,
            limitations=limitations,
            notes="Grace CPU 计算侧模板；载入后须显式绑定 profiles.components.cpu 中的 Profile，才可校准分类吞吐、延迟和能耗。",
            measurement_basis="采用厂商公开的 72 核 Grace CPU、产品级一致性接口和 LPDDR5X 聚合带宽；不推导未公开的通用 OPS。",
            extras={
                "component_scope": "compute_side_only",
                "external_memory_required": True,
                "unknown_value_sentinels": {
                    "peak_ops_per_s": "0.0 means not vendor-declared; the component's explicit profiles.components.cpu binding is authoritative"
                },
                "capability_applicability": {
                    "capacity_bytes": "not_applicable; capacity belongs to a separate host_memory component",
                    "peak_ops_per_s": "reporting mirror only; the component-bound profiles.components.cpu entry owns categorized execution cost",
                },
                "cpu_profile_required": True,
                "value_scope": "单颗 Grace CPU 计算侧组件",
                "conditions": [
                    "CPU 与可达 host_memory 组件必须各自显式绑定 profiles.components.cpu / profiles.components.host_memory registry 条目",
                    "NVLink-C2C 单向值由 900 GB/s 聚合界面值对称除二",
                ],
                "derived_formula": "450 GB/s × 8 = 3600 Gb/s；LPDDR5X GB/s × 8 写入 IR",
                "expires_at": "2027-08-27",
                "cost_profile_template": cost_profile_template,
                "cost_profile_parameter_basis": cost_profile_parameter_basis,
                "cost_profile_key": "cpu",
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family="NVIDIA Grace CPU",
        component=component,
        sources=(source,),
        evidence_level=S2_VENDOR_DECLARED,
        limitations=limitations,
        notes="Grace CPU 计算侧模板；吞吐成本由可编辑 CPU Profile 提供。",
        tags=("cpu", "grace", "arm", "neoverse-v2"),
    )


def _desktop_cpu_preset(
    preset_id: str,
    name: str,
    *,
    model: str,
    core_count: int,
    thread_count: int,
    base_clock_ghz: float,
    boost_clock_ghz: float,
    l2_capacity_bytes: int,
    l3_capacity_bytes: int,
    tdp_watts: int,
    memory_data_rate_mt_s: int,
    memory_channels: int,
    source: ComponentSource,
) -> ComponentPresetDefinition:
    """Build a desktop CPU compute endpoint from a vendor product page.

    Capacity and memory bandwidth intentionally remain on separate memory
    components.  The CPU component only exposes compute/cache facts and a
    DDR5 controller port so an architecture can connect an explicit DRAM
    preset without duplicating its capacity.
    """
    component_id = preset_id.replace("-", "_")
    profile, profile_basis = _cpu_cost_profile_template(
        component_id,
        core_count=core_count,
        frequency_ghz=boost_clock_ghz,
    )
    # _cpu_cost_profile_template is shared with the Grace reference catalog;
    # replace its vendor-specific wording so this AMD preset cannot imply Arm
    # Neoverse facts or NVIDIA provenance.
    profile_basis.update({
        "pipeline.core_count": "AMD Ryzen 9 9950X3D official 16 physical cores",
        "pipeline.frequency_ghz": "AMD Ryzen 9 9950X3D official max boost clock; sustained clocks remain workload dependent",
        "pipeline.simd_width_bits": "A_ANALYTICAL editable SIMD width; AMD product page does not publish this simulator field",
        "pipeline.vector_fma_units_per_core": "A_ANALYTICAL editable execution-unit assumption; not vendor-declared",
        "pipeline.vector_alu_units_per_core": "A_ANALYTICAL editable execution-unit assumption; not vendor-declared",
        "pipeline.*": "A_ANALYTICAL editable issue/queue defaults; product page does not publish all simulator fields",
        "cache_hierarchy.levels[0].capacity_bytes": "AMD Ryzen 9 9950X3D official L1D cache class; byte conversion uses 1 MiB = 1024² B",
        "cache_hierarchy.levels[1].capacity_bytes": "AMD Ryzen 9 9950X3D official total L2 cache 16 MB; byte conversion uses 1 MiB = 1024² B",
        "cache_hierarchy.levels[2].capacity_bytes": "AMD Ryzen 9 9950X3D official total L3 cache 128 MB; byte conversion uses 1 MiB = 1024² B",
    })
    # The generic analytical profile uses a scalable cache shape.  Replace
    # cache capacities with the product's published cache totals while keeping
    # unreported issue/latency values explicitly analytical.
    levels = profile["cache_hierarchy"]["levels"]
    if levels:
        levels[0]["capacity_bytes"] = 1280 * 1024  # AMD page: total L1 cache 1280 KB
    if len(levels) >= 2:
        levels[1]["capacity_bytes"] = l2_capacity_bytes
    if len(levels) >= 3:
        levels[2]["capacity_bytes"] = l3_capacity_bytes
    memory_bandwidth_gbps = (
        memory_data_rate_mt_s * 64.0 * memory_channels / 1_000.0
    )  # MT/s × 64-bit payload × channels, expressed as decimal Gb/s
    ports = (
        PortSpec(
            port_id="memory",
            protocol="DDR5",
            role="controller",
            version="DDR5-{}".format(memory_data_rate_mt_s),
            lanes=64 * memory_channels,
            bandwidth_gbps=memory_bandwidth_gbps,
            metadata=_port_metadata(
                {
                    "data_rate_mt_s": memory_data_rate_mt_s,
                    "channels": memory_channels,
                    "interface_width_bits": 64,
                    "bandwidth_scope": "theoretical_payload_one_way",
                }
            ),
        ),
    )
    limitations = (
        "AMD 页面给出频率、核心/线程、缓存和 TDP；端到端内存延迟及跨工作负载 OPS 不在产品规格中。",
        "capacity_bytes 保持 0；主机内存容量和介质吞吐必须由独立 host_memory 组件表达。",
        "peak_ops_per_s 保持 0；CPU 分类执行成本由组件绑定的 profiles.components.cpu 模板提供。",
    )
    component = ComponentSpec(
        component_id=component_id,
        kind="cpu",
        cost_profile_id=_default_cost_profile_id("cpu"),
        ports=ports,
        capacity_bytes=0,
        peak_ops_per_s=0.0,
        metadata=_component_metadata(
            preset_id,
            technology={
                "vendor": "AMD",
                "architecture": "Zen 5 with 3D V-Cache",
                "model": model,
                "core_count": core_count,
                "thread_count": thread_count,
                "base_clock_ghz": base_clock_ghz,
                "boost_clock_ghz": boost_clock_ghz,
                "l2_capacity_bytes": l2_capacity_bytes,
                "l3_capacity_bytes": l3_capacity_bytes,
                "tdp_watts": tdp_watts,
                "socket": "AM5",
                "memory_type": "DDR5",
                "memory_data_rate_mt_s": memory_data_rate_mt_s,
                "memory_channels": memory_channels,
            },
            sources=(source,),
            evidence_level=S2_VENDOR_DECLARED,
            limitations=limitations,
            notes="AMD Ryzen 9 9950X3D 桌面 CPU 计算端点；内存容量由独立 DDR5 组件提供。",
            measurement_basis="核心/线程、频率、缓存、TDP 和内存速度来自 AMD 产品页；OPS、延迟和效率是可编辑分析参数。",
            extras={
                "component_scope": "compute_side_only",
                "external_memory_required": True,
                "cpu_profile_required": True,
                "capability_applicability": {
                    "capacity_bytes": "not_applicable; use an explicit host_memory component",
                    "read_bandwidth_gbps": "not_applicable; use an explicit host_memory component",
                    "write_bandwidth_gbps": "not_applicable; use an explicit host_memory component",
                    "peak_ops_per_s": "profile-bound categorized CPU cost; no vendor-general OPS claim",
                },
                "unknown_value_sentinels": {
                    "peak_ops_per_s": "0.0 means not vendor-declared",
                },
                "value_scope": "单颗 AMD Ryzen 9 9950X3D 计算端点",
                "conditions": [
                    "DDR5 theoretical payload bandwidth: 5600 MT/s × 64-bit × 2 channels = 89.6 GB/s",
                    "CPU cache and memory-controller latency remain analytical defaults",
                ],
                "derived_formula": "5600 MT/s × 64-bit × 2 channels ÷ 8 = 89.6 GB/s; IR port bandwidth = 716.8 Gb/s",
                "cost_profile_template": profile,
                "cost_profile_parameter_basis": profile_basis,
                "cost_profile_key": "cpu",
                "vendor_parameter_provenance": {
                    "core_count": {"value": core_count, "unit": "core", "source_field": "CPU Cores"},
                    "thread_count": {"value": thread_count, "unit": "thread", "source_field": "Threads"},
                    "base_clock_ghz": {"value": base_clock_ghz, "unit": "GHz", "source_field": "Base Clock"},
                    "boost_clock_ghz": {"value": boost_clock_ghz, "unit": "GHz", "source_field": "Max. Boost Clock"},
                    "l2_capacity_bytes": {"value": l2_capacity_bytes, "unit": "B", "source_field": "L2 Cache"},
                    "l3_capacity_bytes": {"value": l3_capacity_bytes, "unit": "B", "source_field": "L3 Cache"},
                    "tdp_watts": {"value": tdp_watts, "unit": "W", "source_field": "Default TDP"},
                    "l2_capacity_bytes": {"value": l2_capacity_bytes, "unit": "B", "source_field": "L2 Cache; converted from 16 MB using 1 MiB = 1024² B"},
                    "l3_capacity_bytes": {"value": l3_capacity_bytes, "unit": "B", "source_field": "L3 Cache; converted from 128 MB using 1 MiB = 1024² B"},
                    "memory": {"value": memory_data_rate_mt_s, "unit": "MT/s", "source_field": "System Memory Specification"},
                },
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family="AMD Ryzen 9000 Desktop CPU",
        component=component,
        sources=(source,),
        evidence_level=S2_VENDOR_DECLARED,
        limitations=limitations,
        notes="AMD Ryzen 9 9950X3D 桌面 CPU 计算端点；内存容量由独立 DDR5 组件提供。",
        tags=("cpu", "amd", "ryzen", "zen5", "desktop"),
    )


def _consumer_gpu_preset(
    preset_id: str,
    name: str,
    *,
    model: str,
    architecture: str,
    cuda_cores: int,
    sm_count: int,
    tensor_cores_per_sm: int,
    boost_clock_ghz: float,
    memory_gb: float,
    memory_data_rate_gbps: float,
    memory_bus_bits: int,
    memory_bandwidth_gb_s: float,
    l2_capacity_bytes: int,
    tgp_watts: int,
    bf16_dense_tflops: float,
    source: ComponentSource,
    architecture_source: Optional[ComponentSource] = None,
) -> ComponentPresetDefinition:
    """Build a consumer GPU compute endpoint with explicit GDDR interface port."""
    component_id = preset_id.replace("-", "_")
    source_items = (source,) + ((architecture_source,) if architecture_source else ())
    profile, profile_basis = _gpu_cost_profile_template(
        component_id,
        sm_count=sm_count,
        frequency_ghz=boost_clock_ghz,
        peak_bf16_tflops=bf16_dense_tflops,
    )
    profile["cache_hierarchy"]["levels"][-1]["capacity_bytes"] = l2_capacity_bytes
    profile["cache_hierarchy"]["levels"][0]["capacity_bytes"] = 10752 * 1024  # Blackwell PDF: 10,752 KB L1 cache
    profile_basis["tensor_core.sm_count"] = "NVIDIA RTX 5080 CUDA cores ÷ 128 CUDA cores/SM = 84 SMs"
    profile_basis["tensor_core.frequency_ghz"] = "NVIDIA RTX 5080 official boost clock"
    profile_basis["tensor_core.tensor_cores_per_sm"] = "NVIDIA Blackwell architecture mapping; verify exact SKU microarchitecture"
    profile_basis["cache_hierarchy.levels[0].capacity_bytes"] = "NVIDIA RTX Blackwell Architecture Appendix B: 10,752 KB L1 cache; converted using 1 KB = 1024 B"
    profile_basis["cache_hierarchy.levels[1].capacity_bytes"] = "NVIDIA RTX Blackwell Architecture Appendix B: 65,536 KB L2 cache; converted using 1 KB = 1024 B"
    port_bandwidth_gbps = memory_bandwidth_gb_s * 8.0
    component = ComponentSpec(
        component_id=component_id,
        kind="gpu",
        cost_profile_id=_default_cost_profile_id("gpu"),
        ports=(
            PortSpec(
                port_id="gddr7",
                protocol="GDDR",
                role="controller",
                version="GDDR7",
                lanes=memory_bus_bits,
                bandwidth_gbps=port_bandwidth_gbps,
                metadata=_port_metadata(
                    {
                        "memory_data_rate_gbps": memory_data_rate_gbps,
                        "memory_bus_width_bits": memory_bus_bits,
                        "bandwidth_scope": "vendor_memory_interface_one_way",
                    }
                ),
            ),
        ),
        # GPU compute endpoints do not own memory capacity or media bandwidth;
        # those are represented by an attached memory component/link.
        capacity_bytes=0,
        peak_ops_per_s=bf16_dense_tflops * 1_000_000_000_000,
        read_bandwidth_gbps=0.0,
        write_bandwidth_gbps=0.0,
        metadata=_component_metadata(
            preset_id,
            technology={
                "vendor": "NVIDIA",
                "architecture": architecture,
                "model": model,
                "cuda_cores": cuda_cores,
                "sm_count": sm_count,
                "tensor_cores_per_sm": tensor_cores_per_sm,
                "boost_clock_ghz": boost_clock_ghz,
                "device_memory_gb": memory_gb,
                "device_memory_type": "GDDR7",
                "device_memory_data_rate_gbps": memory_data_rate_gbps,
                "device_memory_bus_width_bits": memory_bus_bits,
                "device_memory_bandwidth_gb_s": memory_bandwidth_gb_s,
                "l2_capacity_bytes": l2_capacity_bytes,
                "tgp_watts": tgp_watts,
                "bf16_dense_tflops": bf16_dense_tflops,
            },
            sources=source_items,
            evidence_level=S2_VENDOR_DECLARED,
            limitations=(
                "显存容量和 GDDR7 媒体带宽记录在 technology/端口元数据中；GPU 组件不重复拥有 capacity/read/write 字段。",
                "BF16 Tensor Core 峰值采用 NVIDIA Blackwell 技术简报 RTX 5080 表格的 112.6 TFLOPS（FP32 accumulate）；实际应用吞吐取决于内核和占用率。",
            ),
            notes="NVIDIA GeForce RTX 5080 Blackwell 计算端点；显存通过 GDDR7 接口端口建模。",
            measurement_basis="NVIDIA 官方页提供 CUDA 核心、显存、总线、速度、带宽和功耗；BF16 峰值是架构参数推导。",
            extras={
                "component_scope": "compute_side_only",
                "external_memory_modeled_in_metadata": False,
                "memory_requires_explicit_component": True,
                "capability_applicability": {
                    "capacity_bytes": "not_applicable_on_gpu_compute_endpoint; use attached GDDR7 memory metadata/port",
                    "read_bandwidth_gbps": "not_applicable_on_gpu_compute_endpoint; use GDDR7 port",
                    "write_bandwidth_gbps": "not_applicable_on_gpu_compute_endpoint; use GDDR7 port",
                    "peak_ops_per_s": "derived BF16 Tensor Core compute envelope",
                },
                "value_scope": "单颗 NVIDIA GeForce RTX 5080 计算端点",
                "conditions": [
                    "RTX 5080 official memory bandwidth is 960 GB/s one-way",
                    "BF16 dense throughput is vendor-declared in NVIDIA RTX Blackwell GPU Architecture Table 4 (112.6 TFLOPS FP32 accumulate)",
                ],
                "derived_formula": "960 GB/s × 8 = 7680 Gb/s; BF16 tensor peak copied from NVIDIA Blackwell Table 4",
                "cost_profile_template": profile,
                "cost_profile_parameter_basis": profile_basis,
                "cost_profile_key": "gpu",
                "vendor_parameter_provenance": {
                    "cuda_cores": {"value": cuda_cores, "unit": "count", "source_field": "CUDA Cores"},
                    "boost_clock_ghz": {"value": boost_clock_ghz, "unit": "GHz", "source_field": "Boost Clock"},
                    "memory_gb": {"value": memory_gb, "unit": "GB_decimal", "source_field": "Standard Memory Config"},
                    "memory_data_rate_gbps": {"value": memory_data_rate_gbps, "unit": "Gb/s_per_pin", "source_field": "Memory Speed"},
                    "memory_bus_bits": {"value": memory_bus_bits, "unit": "bit", "source_field": "Memory Interface Width"},
                    "memory_bandwidth_gb_s": {"value": memory_bandwidth_gb_s, "unit": "GB/s_decimal", "source_field": "Memory Bandwidth"},
                    "tgp_watts": {"value": tgp_watts, "unit": "W", "source_field": "Total Graphics Power"},
                    "l2_capacity_bytes": {"value": l2_capacity_bytes, "unit": "B", "source_field": "NVIDIA RTX Blackwell Architecture Table 3; 65,536 KB"},
                    "bf16_dense_tflops": {"value": bf16_dense_tflops, "unit": "TFLOP/s", "status": "vendor_declared_blackwell_table_4"},
                },
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family="NVIDIA GeForce RTX 50 Series",
        component=component,
        sources=source_items,
        evidence_level=S2_VENDOR_DECLARED,
        limitations=component.metadata["applicability_limitations"],
        notes="NVIDIA GeForce RTX 5080 Blackwell 计算端点；显存通过 GDDR7 接口端口建模。",
        tags=("gpu", "nvidia", "blackwell", "gddr7", "consumer"),
    )


_LEGACY_PRESETS: Tuple[ComponentPresetDefinition, ...] = (
    _grace_cpu_preset(
        "nvidia-grace-cpu-gh200",
        "NVIDIA Grace CPU for GH200",
        product_family="GH200",
        gpu_port_count=1,
        memory_bandwidth_gbps=4000.0,
        source=NVIDIA_GH200_SOURCE,
    ),
    _grace_cpu_preset(
        "nvidia-grace-cpu-gb200",
        "NVIDIA Grace CPU for GB200",
        product_family="GB200",
        gpu_port_count=2,
        memory_bandwidth_gbps=4096.0,
        source=NVIDIA_GB200_SOURCE,
    ),
    _hbm_preset(
        "hbm2e-16gb-3_2",
        "HBM2E 16GB 409.6 GB/s Analytical Stack",
        generation="HBM2E",
        pin_speed_gbps=3.2,
        io_bits=1024,
        channels=16,
        capacity_gb=16.0,
        bandwidth_gbps=3276.8,
        evidence_level=A_ANALYTICAL,
        limitations=(
            "HBM2E 每引脚 3.2 的速度档按 1024 位接口换算，作为分析模板而非指定产品逐堆叠规格。",
            "模板不自动代表 Gaudi 3 的产品级等分带宽；Gaudi 3 请使用对应 product-slice 预设。",
        ),
        notes="单颗 HBM2E 16GB 分析模板。",
        sources=(SAMSUNG_HBM_SOURCE,),
    ),
    _hbm_product_slice_preset(
        "hbm3-16gb-0_670tbs-h100-slice",
        "H100 HBM3 16GB 670 GB/s Product Slice",
        generation="HBM3",
        visible_capacity_gb=16.0,
        raw_capacity_gb=16.0,
        bandwidth_gbps=5360.0,
        product_stack_count=5,
        unit_count_status="vendor_documented_active_stacks",
        unit_count_formula="5 active HBM3 stacks × 16 GB = 80 GB; 3.35 TB/s / 5 = 670 GB/s",
        product_family="NVIDIA H100 SXM",
        sources=(NVIDIA_H100_SOURCE,),
    ),
    _hbm_product_slice_preset(
        "hbm3-16gb-0_667tbs-product-slice",
        "GH200 HBM3 16GB 666.667 GB/s Product Slice",
        generation="HBM3",
        visible_capacity_gb=16.0,
        raw_capacity_gb=16.0,
        bandwidth_gbps=32000.0 / 6.0,
        product_stack_count=6,
        unit_count_status="derived_from_GH200_total_plus_full_GH100_6_stack_implementation",
        unit_count_formula="96 GB / 16 GB = 6; 4 TB/s / 6 = 666.667 GB/s",
        product_family="NVIDIA GH200 96GB HBM3 variant",
        sources=(NVIDIA_GH200_SOURCE, NVIDIA_H100_SOURCE),
    ),
    _hbm_product_slice_preset(
        "hbm3e-24gb-0_833tbs-gh200-slice",
        "GH200 HBM3E 24GB 833.333 GB/s Product Slice",
        generation="HBM3E",
        visible_capacity_gb=24.0,
        raw_capacity_gb=24.0,
        bandwidth_gbps=40000.0 / 6.0,
        product_stack_count=6,
        unit_count_status="derived_from_product_total_and_24GB_stack_class",
        unit_count_formula="144 GB / 24 GB = 6; 5 TB/s / 6 = 833.333 GB/s",
        product_family="NVIDIA GH200 144GB HBM3E variant",
        sources=(NVIDIA_GH200_SOURCE, SAMSUNG_HBM3E_SOURCE),
    ),
    _hbm_product_slice_preset(
        "hbm3-16gb-0_6625tbs-mi300a-slice",
        "MI300A HBM3 16GB 662.5 GB/s Product Slice",
        generation="HBM3",
        visible_capacity_gb=16.0,
        raw_capacity_gb=16.0,
        bandwidth_gbps=5300.0,
        product_stack_count=8,
        unit_count_status="derived_cross_source_product_architecture",
        unit_count_formula="128 GB / 8 = 16 GB; 5.3 TB/s / 8 = 662.5 GB/s",
        product_family="AMD MI300A",
        sources=(AMD_MI300A_SOURCE,),
    ),
    _hbm_product_slice_preset(
        "hbm3-24gb-0_665625tbs-mi300x-slice",
        "MI300X HBM3 24GB 665.625 GB/s Product Slice",
        generation="HBM3",
        visible_capacity_gb=24.0,
        raw_capacity_gb=24.0,
        bandwidth_gbps=5325.0,
        product_stack_count=8,
        unit_count_status="vendor_documented_stacks",
        unit_count_formula="192 GB / 8 = 24 GB; 5.325 TB/s / 8 = 665.625 GB/s",
        product_family="AMD MI300X",
        sources=(AMD_MI300X_SOURCE,),
    ),
    _hbm_product_slice_preset(
        "hbm3e-24gb-0_800tbs-h200-slice",
        "H200 HBM3E 24GB-class 800 GB/s Product Slice",
        generation="HBM3E",
        visible_capacity_gb=23.5,
        raw_capacity_gb=24.0,
        bandwidth_gbps=6400.0,
        product_stack_count=6,
        unit_count_status="derived_from_product_total_and_24GB_stack_class",
        unit_count_formula="144 GB raw / 24 GB = 6; 141 GB visible / 6 = 23.5 GB; 4.8 TB/s / 6 = 800 GB/s",
        product_family="NVIDIA H200 SXM",
        sources=(NVIDIA_H200_SOURCE, SAMSUNG_HBM3E_SOURCE),
    ),
    _hbm_product_slice_preset(
        "hbm3e-24gb-1_000tbs-gb200-slice",
        "GB200 HBM3E 24GB-class 1 TB/s Product Slice",
        generation="HBM3E",
        visible_capacity_gb=23.25,
        raw_capacity_gb=24.0,
        bandwidth_gbps=8000.0,
        product_stack_count=8,
        unit_count_status="vendor_documented_stack_sites",
        unit_count_formula="8 stack sites per GPU; 186 GB visible / 8 = 23.25 GB; 8 TB/s / 8 = 1 TB/s",
        product_family="NVIDIA GB200",
        sources=(NVIDIA_GB200_SOURCE, SAMSUNG_HBM3E_SOURCE),
    ),
    _hbm_product_slice_preset(
        "nvidia-b200-hbm3e-90gb-4tbps-analysis",
        "NVIDIA B200 HBM3E 90GB 4TB/s Analytical Endpoint",
        generation="HBM3E",
        visible_capacity_gb=90.0,
        raw_capacity_gb=90.0,
        bandwidth_gbps=32_000.0,
        product_stack_count=2,
        unit_count_status="analytical_two_modeled_hbm_nodes",
        unit_count_formula=(
            "NVIDIA DGX B200 1,440 GB / 8 GPUs = 180 GB and 64 TB/s / 8 GPUs = "
            "8 TB/s per B200; user-requested two modeled endpoints split equally"
        ),
        product_family="NVIDIA B200 SXM 180GB",
        sources=(NVIDIA_B200_SOURCE, NVIDIA_BLACKWELL_SOURCE, SAMSUNG_HBM3E_SOURCE),
        evidence_level=A_ANALYTICAL,
    ),
    _hbm_product_slice_preset(
        "hbm2e-16gb-0_4625tbs-gaudi3-slice",
        "Gaudi 3 HBM2E 16GB 462.5 GB/s Product Slice",
        generation="HBM2E",
        visible_capacity_gb=16.0,
        raw_capacity_gb=16.0,
        bandwidth_gbps=3700.0,
        product_stack_count=8,
        unit_count_status="vendor_documented_stacks",
        unit_count_formula="128 GB / 8 = 16 GB; 3.7 TB/s / 8 = 462.5 GB/s",
        product_family="Intel Gaudi 3",
        sources=(INTEL_GAUDI3_SOURCE,),
    ),
    _hbm_product_slice_preset(
        "hbm-pim-16gb-3_2-analysis",
        "HBM2E-PIM 16GB 409.6 GB/s Analytical Stack",
        generation="HBM2E",
        visible_capacity_gb=16.0,
        raw_capacity_gb=16.0,
        bandwidth_gbps=3276.8,
        product_stack_count=1,
        unit_count_status="experimental_reference_scope",
        unit_count_formula="one analytical HBM2E-PIM reference stack",
        product_family="Samsung HBM-PIM analytical reference",
        sources=(SAMSUNG_HBM_PIM_SOURCE,),
        evidence_level=A_ANALYTICAL,
        analytical_pim=True,
    ),
    _grace_lpddr_preset(
        "lpddr5x-gh200-480gb-500gbs-aggregate",
        "GH200 Grace LPDDR5X 480GB 500 GB/s Aggregate",
        product_family="GH200",
        bandwidth_gbps=4000.0,
        source=NVIDIA_GH200_SOURCE,
    ),
    _grace_lpddr_preset(
        "lpddr5x-gb200-480gb-512gbs-aggregate",
        "GB200 Grace LPDDR5X 480GB 512 GB/s Aggregate",
        product_family="GB200",
        bandwidth_gbps=4096.0,
        source=NVIDIA_GB200_SOURCE,
    ),
    _hbm_preset(
        "jedec-hbm3-24gb-6_4",
        "JEDEC HBM3 24GB 819.2 GB/s Stack",
        generation="HBM3",
        pin_speed_gbps=6.4,
        io_bits=1024,
        channels=16,
        capacity_gb=24.0,
        bandwidth_gbps=6553.6,
        evidence_level=S1_STANDARD,
        limitations=(
            "容量选用常见 24GB 堆叠作为模板；JESD238B.01 还允许其他芯粒密度和堆叠高度。",
            "模板只表示单个 HBM 堆叠，不包含 GPU 控制器或封装互连。",
        ),
        notes="HBM3 代际写入 metadata，同时保持 ComponentSpec.kind 为 hbm。",
        sources=(JEDEC_HBM3_SOURCE, SAMSUNG_HBM3_SOURCE),
    ),
    _hbm_preset(
        "hbm3e-24gb-8_0",
        "HBM3E 24GB 1.024 TB/s Stack",
        generation="HBM3E",
        pin_speed_gbps=8.0,
        io_bits=1024,
        channels=16,
        capacity_gb=24.0,
        bandwidth_gbps=8192.0,
        evidence_level=A_ANALYTICAL,
        limitations=(
            "1.024 TB/s 是 HBM3E 常见速度档分析模板；公开供应商页面也可能展示更高峰值。",
            "带宽按公开 HBM3E 接口形态线性换算，未引入供应商时序、ECC 或功耗模型。",
        ),
        notes="适用于需要早期或常见 HBM3E 速度档、而不是最高公开速度档的拓扑分析。",
        sources=(SAMSUNG_HBM3E_SOURCE, MICRON_HBM3E_SOURCE),
    ),
    _hbm_preset(
        "hbm3e-36gb-9_2",
        "HBM3E 36GB 1177.6 GB/s Stack",
        generation="HBM3E",
        pin_speed_gbps=9.2,
        io_bits=1024,
        channels=16,
        capacity_gb=36.0,
        bandwidth_gbps=9420.8,
        evidence_level=S2_VENDOR_DECLARED,
        limitations=(
            "36GB 表示 12-high HBM3E 模板；若使用 8-high 24GB 器件请调整 capacity_bytes。",
            "供应商页面给出的是每堆叠峰值带宽，实际系统值取决于控制器和封装。",
        ),
        notes="供应商公开的 HBM3E 高速堆叠模板，适合 H200 或 AI 加速器风格的显存系统分析。",
        sources=(SAMSUNG_HBM3E_SOURCE, MICRON_HBM3E_SOURCE),
    ),
    _hbm_preset(
        "jedec-hbm4-64gb-8_0",
        "HBM4 64GB 2.048 TB/s Analytical Stack",
        generation="HBM4",
        pin_speed_gbps=8.0,
        io_bits=2048,
        channels=32,
        capacity_gb=64.0,
        bandwidth_gbps=16384.0,
        evidence_level=A_ANALYTICAL,
        limitations=(
            "HBM4 标准接口基准可折算为每堆叠 2.048 TB/s；64 GB 容量是分析模板，不作为 JEDEC 公开容量保证。",
            "Samsung 等厂商产品可采用不同容量和更高带宽档，使用时必须按目标器件校准。",
        ),
        notes="标准接口事实与分析容量组合，因此整体证据等级为 A_ANALYTICAL；HBM4 仍编码为 kind=hbm。",
        sources=(JEDEC_HBM4_SOURCE, SAMSUNG_HBM4_SOURCE),
    ),
    _gpu_preset(
        "nvidia-h100-sxm-gpu",
        "NVIDIA H100 SXM Compute-Side GPU",
        hbm_generation="HBM3",
        hbm_stack_count=5,
        device_memory_gb=80.0,
        device_memory_bandwidth_gbps=26800.0,
        bf16_dense_tflops=989.5,
        fp8_dense_tflops=1979.0,
        fp8_sparse_tflops=3958.0,
        bf16_sparse_tflops=1979.0,
        tdp_watts=700,
        mig_profiles="up to 7 MIGs @ 10GB each",
        sources=(NVIDIA_H100_SOURCE,),
    ),
    _gpu_preset(
        "nvidia-h200-sxm-gpu",
        "NVIDIA H200 SXM Compute-Side GPU",
        hbm_generation="HBM3E",
        hbm_stack_count=6,
        device_memory_gb=141.0,
        device_memory_bandwidth_gbps=38400.0,
        bf16_dense_tflops=989.5,
        fp8_dense_tflops=1979.0,
        fp8_sparse_tflops=3958.0,
        bf16_sparse_tflops=1979.0,
        tdp_watts=700,
        mig_profiles="up to 7 MIGs @ 18GB each",
        sources=(NVIDIA_H200_SOURCE,),
    ),
    _ssd_preset(
        "samsung-pm1743-15_36tb",
        "Samsung PM1743 15.36TB PCIe 5.0 NVMe SSD",
        family="Enterprise NVMe SSD",
        kind="high_io_ssd",
        capacity_bytes=_tb(15.36),
        read_gbps=112.0,
        write_gbps=56.8,
        interface_bandwidth_gbps=126.03076923076924,
        interface="PCIe 5.0 x4, NVMe",
        protocol_version="5.0",
        lanes=4,
        endurance="enterprise server SSD; random write endurance not modeled",
        limitations=(
            "使用当前白皮书顺序读写峰值作为组件带宽；随机 IOPS 不直接转换为带宽。",
            "容量选用 15.36TB 模板；PM1743 系列存在其他容量和形态。",
        ),
        notes="PCIe 5.0 高 I/O SSD 模板，适合模型卸载、检查点或数据集暂存分析。",
        sources=(SAMSUNG_PM1743_WHITE_PAPER_SOURCE, SAMSUNG_PM1743_PRODUCT_SOURCE),
        evidence_level=S2_VENDOR_DECLARED,
        conditions=(
            "128KiB sequential",
            "FIO",
            "Ubuntu 18.04.2",
            "QD256",
            "1 worker",
        ),
    ),
    _ssd_preset(
        "solidigm-d7-p5810-800gb",
        "Solidigm D7-P5810 800GB PCIe 4.0 SLC NVMe SSD",
        family="Enterprise NVMe SSD",
        kind="ssd",
        capacity_bytes=_gb(800.0),
        read_gbps=51.2,
        write_gbps=35.2,
        interface_bandwidth_gbps=63.01538461538462,
        interface="PCIe 4.0 x4, NVMe",
        protocol_version="4.0",
        lanes=4,
        endurance="up to 50 DWPD random write workload",
        limitations=(
            "模板强调高写耐久 SLC/持久缓存型 SSD；不代表所有 D7 系列或 QLC 缓存部署。",
            "顺序峰值带宽和 4KB 随机 IOPS 属于不同工作负载，仿真时不要混用。",
        ),
        notes="写密集持久缓存型 SSD 模板，适合缓存、日志和热卸载层分析。",
        sources=(SOLIDIGM_D7_P5810_SOURCE,),
        evidence_level=S2_VENDOR_DECLARED,
    ),
    ComponentPresetDefinition(
        preset_id="ocp-hbf-2026-512gb",
        name="2026 OCP High Bandwidth Flash 512GB Stack",
        family="High Bandwidth Flash",
        component=ComponentSpec(
            component_id="ocp_hbf_2026_512gb",
            kind="hbf",
            ports=(
                PortSpec(
                    port_id="ucie0",
                    # HBF is the logical service protocol.  The physical
                    # package transport remains recorded in metadata.
                    protocol="HBF",
                    role="endpoint",
                    version="2.0",
                    lanes=64,
                    bandwidth_gbps=2048.0,
                    payload="xPU-HBF",
                    metadata=_port_metadata(
                        {
                            "performance_grade": "grade_3",
                            "bandwidth_basis": "analytical_HBF_x64_32_GTps_envelope",
                            "physical_transport_protocol": "UCIe",
                            "media_bandwidth_modeled_separately": True,
                        }
                    ),
                ),
            ),
            capacity_bytes=_gb(512.0),
            # ComponentSpec stores one-way rates in Gb/s; these are the
            # requested decimal 488 GB/s read and 27.2 GB/s write values.
            read_bandwidth_gbps=3904.0,
            write_bandwidth_gbps=217.6,
            metadata=_component_metadata(
                "ocp-hbf-2026-512gb",
                technology={
                    "generation": "HBF 2026 OCP",
                    "media": "NAND flash",
                    "interface": "HBF logical service over UCIe",
                    "capacity_gb": 512,
                    "read_bandwidth_gbps": 3904.0,
                    "write_bandwidth_gbps": 217.6,
                    "non_volatile": True,
                },
                sources=(SK_HYNIX_HBF_SOURCE, SANDISK_HBF_SOURCE),
                evidence_level=S3_VENDOR_PREPRODUCTION,
                limitations=(
                    "2026 OCP HBF 仍是新兴生态参考；本预设的读写带宽和延迟是用户指定的分析坐标，不是量产 SKU 实测。",
                    "HBF 通过 HBF 逻辑协议链路建模，metadata.physical_transport_protocol=UCIe 保留封装承载信息。",
                    "读写方向带宽不对称；端到端吞吐仍受链路、控制器、事务粒度和队列深度限制。",
                    "active-memory 语义仅用于显式 KV Cache 敏感性分析，不代表公开 HBF 产品已经提供透明 load/store。",
                ),
                notes="High Bandwidth Flash 位于 HBM 与 SSD 层之间；拓扑使用 HBF 逻辑链路并记录 UCIe 物理承载。",
                measurement_basis="用户指定 488/27.2 GB/s 和 4/75 us 分析坐标；带宽换算到 IR Gb/s 字段。",
                extras={
                    "read_latency_ns": 4000.0,
                    "write_latency_ns": 75000.0,
                    "transfer_granularity_bytes": 4096,
                    "max_outstanding_requests": 32,
                    "dma_bandwidth_gbps": 2048.0,
                    "dma_latency_ns": 800.0,
                    "dma_energy_pj_per_byte": 0.0,
                    "unknown_value_sentinels": {
                        "dma_energy_pj_per_byte": "0.0 means unknown/not declared, not zero energy",
                    },
                    "storage_transport_parameter_basis": (
                        "editable analytical HBF controller/queue defaults; public evidence "
                        "only supports the separately declared capacity and read envelope"
                    ),
                    "component_scope": "near_package_nonvolatile_memory",
                    "physical_composition": {
                        "simulator_representation": "single_physical_unit_node",
                        "simulator_node_count": 1,
                        "physical_unit_kind": "HBF_stack",
                        "physical_unit_count": 1,
                        "physical_unit_count_status": "explicit_reference_node",
                        "known_multiple": False,
                        "unit_index": 0,
                        "unit_count_in_product": 1,
                        "unit_count_status": "preproduction_reference_scope",
                        "unit_count_formula": "one HBF reference stack per standalone component preset",
                        "unit_capacity_bytes": _gb(512.0),
                        "product_total_capacity_bytes": _gb(512.0),
                        "component_preset_id": "ocp-hbf-2026-512gb",
                        "source_basis": "OCP HBF preproduction up-to envelope",
                    },
                    "capacity_scope": "up_to_512GB",
                    "bandwidth_scope": "grade3_up_to_3TB_per_s",
                    "internal_nand_composition": {
                        "dies_per_stack": "8-high_or_16-high",
                        "correlation_status": "not_reliably_disclosed",
                    },
                    "value_scope": "单个近封装非易失存储组件模板",
                    "conditions": [
                        "预生产/工作流级公开信息与用户指定分析坐标",
                        "读写带宽和读写延迟均是可替换分析参数",
                        "需要显式 HBF 逻辑链路，物理承载仍记录为 UCIe",
                    ],
                    "derived_formula": "用户指定十进制 GB/s 乘以 8，写入 IR Gb/s 字段",
                    "access_mode": "memory",
                    "read_only": False,
                    "writable": True,
                    "write_buffer_bytes": 0,
                    "memory_service_owner": "ocp_hbf_2026_512gb.memory",
                    "cost_profile_key": "host_memory",
                    "cost_profile_template": {
                        "bandwidth_gb_s": 488.0,
                        "efficiency": 1.0,
                        "energy_pj_per_byte": 12.0,
                        "resource_id": "ocp_hbf_2026_512gb.memory",
                        "name": "HBF analytical active-memory profile",
                        "read_latency_ns": 4000.0,
                        "write_latency_ns": 75000.0,
                        "transaction_bytes": 4096,
                        "max_outstanding_requests": 32,
                        "read_bandwidth_gb_s": 488.0,
                        "write_bandwidth_gb_s": 27.2,
                    },
                    "expires_at": "2027-08-22",
                },
            ),
        ),
        sources=(SK_HYNIX_HBF_SOURCE, SANDISK_HBF_SOURCE),
        evidence_level=S3_VENDOR_PREPRODUCTION,
        limitations=(
            "2026 OCP HBF 仍是新兴生态参考；本预设的读写带宽和延迟是用户指定的分析坐标，不是量产 SKU 实测。",
            "HBF 使用 HBF 逻辑协议链路，metadata.physical_transport_protocol=UCIe 保留封装承载信息。",
            "active-memory 语义仅用于显式 KV Cache 敏感性分析，不代表公开 HBF 产品已经提供透明 load/store。",
        ),
        notes="High Bandwidth Flash 位于 HBM 与 SSD 层之间；拓扑使用 HBF 逻辑链路并记录 UCIe 物理承载。",
        tags=("memory", "hbf", "ucie", "flash"),
    ),
    ComponentPresetDefinition(
        preset_id="digital-sram-cim-analysis",
        name="Digital SRAM-CIM Analytical Tile",
        family="Compute-in-Memory",
        component=ComponentSpec(
            component_id="digital_sram_cim_analysis",
            kind="digital_sram_cim",
            cost_profile_id=_default_cost_profile_id("digital_sram_cim"),
            ports=(
                PortSpec(
                    port_id="ucie0",
                    protocol="UCIe",
                    role="endpoint",
                    version="1.1",
                    lanes=64,
                    bandwidth_gbps=2048.0,
                    payload="streaming",
                    metadata=_port_metadata(),
                ),
            ),
            capacity_bytes=512 * 1024 * 1024,
            peak_ops_per_s=512_000_000_000_000.0,
            read_bandwidth_gbps=2048.0,
            write_bandwidth_gbps=2048.0,
            metadata=_component_metadata(
                "digital-sram-cim-analysis",
                technology={
                    "architecture": "digital SRAM compute-in-memory",
                    "weight_residency": "on_tile_sram",
                    "capacity_mib": 512,
                    "interconnect": "UCIe streaming",
                    "bandwidth_display": "256 GB/s",
                },
                sources=(INTERNAL_CIM_SOURCE,),
                evidence_level=A_ANALYTICAL,
                limitations=(
                    "分析模板，不对应具体流片产品；能效、面积、频率和编译器约束需要按研究假设调整。",
                    "适合 MoE expert/MLP 权重驻留或近存计算探索，不自动修改 placement。",
                ),
                notes="内置数字 SRAM-CIM 分析模板，与参考场景的容量和互连规模保持一致。",
                measurement_basis="内部分析占位值；带宽显示为 256 GB/s，写入 IR 字段时采用比特带宽。",
                extras={
                    "component_scope": "analysis_template",
                    "value_scope": "单个数字 SRAM-CIM 分析 tile",
                    "conditions": [
                        "非产品规格",
                        "需要按研究假设调整能效、面积、频率和编译器约束",
                    ],
                    "derived_formula": "256 GB/s 乘以 8，写入 IR 带宽字段",
                    "expires_at": "2027-08-22",
                    "cost_profile_template": _cim_cost_profile_template(
                        "digital_sram_cim_analysis"
                    )[0],
                    "cost_profile_parameter_basis": _cim_cost_profile_template(
                        "digital_sram_cim_analysis"
                    )[1],
                    "cost_profile_key": "cim",
                },
            ),
        ),
        sources=(INTERNAL_CIM_SOURCE,),
        evidence_level=A_ANALYTICAL,
        limitations=(
            "分析模板，不对应具体流片产品；能效、面积、频率和编译器约束需要按研究假设调整。",
            "适合 MoE expert/MLP 权重驻留或近存计算探索，不自动修改 placement。",
        ),
        notes="内置数字 SRAM-CIM 分析模板，与参考场景的容量和互连规模保持一致。",
        tags=("cim", "sram", "analysis"),
    ),
)


def _curated_clone(
    definition: ComponentPresetDefinition,
    *,
    preset_id: str,
    name: str,
    family: Optional[str] = None,
    sources: Optional[Sequence[ComponentSource]] = None,
    evidence_level: Optional[str] = None,
    notes: Optional[str] = None,
    limitations: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    component_updates: Optional[Mapping[str, Any]] = None,
    technology_updates: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ComponentPresetDefinition:
    """Clone a legacy template while making the curated evidence explicit."""
    source_items = tuple(sources or definition.sources)
    metadata = deepcopy(dict(definition.component.metadata))
    metadata.setdefault("preset", {})["id"] = preset_id
    metadata["preset"]["catalog_version"] = CATALOG_VERSION
    metadata["sources"] = list(_source_metadata(source_items))
    if technology_updates:
        metadata["technology"] = {
            **dict(metadata.get("technology", {})),
            **dict(technology_updates),
        }
    if provenance:
        metadata["vendor_parameter_provenance"] = deepcopy(dict(provenance))
    component = replace(
        definition.component,
        component_id=preset_id.replace("-", "_"),
        metadata=metadata,
        **dict(component_updates or {}),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family=family or definition.family,
        component=component,
        sources=source_items,
        evidence_level=evidence_level or definition.evidence_level,
        limitations=tuple(limitations or definition.limitations),
        notes=notes or definition.notes,
        tags=tuple(tags or definition.tags),
    )


def _legacy_component(preset_id: str) -> ComponentPresetDefinition:
    return next(item for item in _LEGACY_PRESETS if item.preset_id == preset_id)


# Public component catalog.  Legacy definitions remain importable through the
# private compatibility registry below because saved scenarios and old tests
# may still refer to their stable IDs; they are deliberately absent from the
# public catalog returned by list_component_presets().
_SAMSUNG_HBM3E = _hbm_preset(
    "samsung-hbm3e-36gb-9_2",
    "Samsung HBM3E 12-High 36GB (9.2Gb/s per pin)",
    generation="HBM3E", pin_speed_gbps=9.2, io_bits=1024, channels=16,
    capacity_gb=36.0, bandwidth_gbps=9.2 * 1024,
    evidence_level=S2_VENDOR_DECLARED,
    limitations=(
        "Samsung 公布的是 HBM3E 器件/堆叠速度档；端到端可用吞吐仍取决于 GPU 控制器、封装和工作负载。",
        "read/write 是同一接口的方向性峰值，不应相加宣传为同时双倍吞吐。",
    ),
    notes="Samsung HBM3E 12-high 36GB 产品档；每堆叠峰值带宽由 9.2Gb/s/pin × 1024-bit ÷ 8 换算。",
    sources=(SAMSUNG_HBM3E_OFFICIAL_SOURCE,),
)
_SAMSUNG_HBM3E = _curated_clone(
    _SAMSUNG_HBM3E,
    preset_id="samsung-hbm3e-36gb-9_2",
    name="Samsung HBM3E 12-High 36GB (9.2Gb/s per pin)",
    provenance={
        "capacity_bytes": {"value": "36 GB", "unit": "GB_decimal", "source_field": "Samsung HBM3E 12-high product capacity"},
        "pin_speed_gbps": {"value": 9.2, "unit": "Gb/s_per_pin", "source_field": "Samsung HBM3E speed grade"},
        "interface_bits": {"value": 1024, "unit": "bit", "source_field": "HBM3E standard interface width"},
        "read_bandwidth_gbps": {"formula": "9.2 Gb/s/pin × 1024 bit ÷ 8 = 1177.6 GB/s = 9420.8 Gb/s", "unit": "Gb/s_decimal_one_way"},
        "write_bandwidth_gbps": {"formula": "same interface peak used as a directional write envelope", "status": "derived_not_separately_published"},
        "source_excerpt": "Samsung HBM3E page: 12-layer stack, up to 1,180 GB/s bandwidth at 9.2 Gbps, 36 GB capacity",
        "vendor_display_bandwidth": {"value": 1180, "unit": "GB/s_decimal", "status": "rounded_vendor_display"},
    },
)
_GENERIC_HBM3E = _hbm_preset(
    "hbm3e-standard-24gb-8_0",
    "HBM3E Standard 24GB 1.024TB/s Analytical Stack",
    generation="HBM3E", pin_speed_gbps=8.0, io_bits=1024, channels=16,
    capacity_gb=24.0, bandwidth_gbps=8.0 * 1024,
    evidence_level=A_ANALYTICAL,
    limitations=(
        "这是接口速度与常见 24GB 容量组合的分析模板，不对应单一厂商 SKU。",
        "延迟、效率、队列和可用吞吐属于可编辑仿真参数，不是 JEDEC 端到端保证。",
    ),
    notes="仅保留一个通用 HBM3E 分析档，避免把不同厂商产品切片误当成独立标准器件。",
    sources=(SAMSUNG_HBM3E_OFFICIAL_SOURCE,),
)
_GENERIC_HBM3E = _curated_clone(
    _GENERIC_HBM3E,
    preset_id="hbm3e-standard-24gb-8_0",
    name="HBM3E Standard 24GB 1.024TB/s Analytical Stack",
    provenance={
        "capacity_bytes": {"value": "24 GB", "unit": "GB_decimal", "status": "analytical_common_stack_class"},
        "pin_speed_gbps": {"value": 8.0, "unit": "Gb/s_per_pin", "status": "analytical_speed_grade"},
        "interface_bits": {"value": 1024, "unit": "bit", "status": "HBM interface convention"},
        "read_bandwidth_gbps": {"formula": "8.0 Gb/s/pin × 1024 bit ÷ 8 = 1024 GB/s = 8192 Gb/s", "unit": "Gb/s_decimal_one_way"},
        "write_bandwidth_gbps": {"formula": "same interface peak used as directional analytical envelope", "status": "derived_not_vendor_specific"},
    },
)

# GDDR generations share the same lightweight DRAM service model while
# retaining first-class component identity and generation-specific metadata.
_GDDR6_REFERENCE = _gddr_preset(
    "gddr6-16gb-20_0-256bit",
    "GDDR6 16GB 20Gb/s 256-bit Graphics Memory",
    generation="GDDR6",
    data_rate_gbps=20.0,
    interface_bits=256,
    capacity_gb=16.0,
    evidence_level=A_ANALYTICAL,
    limitations=(
        "GDDR6 20Gb/s 与 16GB/256-bit 是可编辑分析组合，不对应单一厂商 SKU。",
        "时序、效率、能耗和持续带宽保留为分析参数，不能解读为器件级实测。",
    ),
    notes="GDDR6 分析参考：20Gb/s/pin × 256-bit ÷ 8 = 640GB/s。",
    sources=(MICRON_GDDR_SOURCE,),
)
_GDDR6X_REFERENCE = _gddr_preset(
    "gddr6x-16gb-21_0-256bit",
    "GDDR6X 16GB 21Gb/s 256-bit Graphics Memory",
    generation="GDDR6X",
    data_rate_gbps=21.0,
    interface_bits=256,
    capacity_gb=16.0,
    evidence_level=A_ANALYTICAL,
    limitations=(
        "GDDR6X 21Gb/s 与 16GB/256-bit 是可编辑分析组合，不对应单一厂商 SKU。",
        "GDDR6X 的 PAM4 仅记录为信号方式说明；有效数据率不再额外乘编码系数。",
    ),
    notes="GDDR6X 分析参考：21Gb/s/pin × 256-bit ÷ 8 = 672GB/s。",
    sources=(MICRON_GDDR_SOURCE,),
    signal_encoding="PAM4",
)
_RTX5080_GDDR7 = _gddr_preset(
    "gddr7-16gb-30_0-256bit",
    "GDDR7 16GB 30Gb/s 256-bit GPU Memory",
    generation="GDDR7",
    data_rate_gbps=30.0,
    interface_bits=256,
    capacity_gb=16.0,
    evidence_level=S2_VENDOR_DECLARED,
    limitations=(
        "该预设采用 RTX 5080 公布的 GDDR7 16GB、30Gb/s、256-bit 接口参数；实际可用容量和持续吞吐受驱动、控制器和工作负载影响。",
        "读写带宽是同一 GDDR7 接口的方向性上限，不应相加理解为同时双倍预算。",
        "ComponentSpec.kind 正式为 gddr；该节点表示 GPU 本地 GDDR7 显存子系统。",
    ),
    notes="NVIDIA GeForce RTX 5080 的 GDDR7 本地显存预设；30Gb/s × 256-bit ÷ 8 = 960GB/s，写入 IR 为 7680Gb/s。",
    sources=(NVIDIA_RTX_5080_SOURCE,),
    family="GDDR7 GPU Memory",
    signal_encoding="PAM3",
)
_RTX5080_GDDR7 = _curated_clone(
    _RTX5080_GDDR7,
    preset_id="gddr7-16gb-30_0-256bit",
    name="GDDR7 16GB 30Gb/s 256-bit GPU Memory",
    tags=("memory", "gddr7", "gpu_memory"),
    technology_updates={
        "vendor": "NVIDIA GeForce RTX 5080 configuration",
        "memory_type": "GDDR7",
        "data_rate_gbps": 30.0,
        "interface_bits": 256,
        "bandwidth_gb_s": 960.0,
        "simulator_component_kind": "gddr",
        "simulator_node_role": "gpu_local_memory_subsystem",
    },
    provenance={
        "capacity_bytes": {"value": 16, "unit": "GB_decimal", "source_field": "RTX 5080 standard memory configuration"},
        "pin_speed_gbps": {"value": 30.0, "unit": "Gb/s_per_pin", "source_field": "RTX 5080 memory speed"},
        "interface_bits": {"value": 256, "unit": "bit", "source_field": "RTX 5080 memory interface width"},
        "read_bandwidth_gbps": {"formula": "30 Gb/s/pin × 256 bit ÷ 8 = 960 GB/s = 7680 Gb/s", "unit": "Gb/s_decimal_one_way", "source_field": "RTX 5080 memory bandwidth"},
        "write_bandwidth_gbps": {"formula": "same GDDR7 interface peak used as a directional write envelope", "status": "derived_directional_envelope"},
    },
)

_HBF_BASE = _legacy_component("ocp-hbf-2026-512gb")
_SK_HYNIX_HBF = _curated_clone(
    _HBF_BASE,
    preset_id="sk-hynix-hbf-512gb",
    name="SK hynix HBF 512GB (488 GB/s read, 27.2 GB/s write analytical)",
    family="High Bandwidth Flash",
    sources=(SK_HYNIX_HBF_OFFICIAL_SOURCE,),
    evidence_level=S3_VENDOR_PREPRODUCTION,
    limitations=(
        "读写带宽和读写延迟采用用户指定分析坐标，不是量产 SKU 的持续实测吞吐。",
        "HBF 逻辑链路 protocol=HBF；metadata.physical_transport_protocol=UCIe 记录物理封装承载。",
        "active-memory 语义用于 KV Cache 敏感性分析；端到端吞吐仍受链路、控制器和队列约束。",
    ),
    notes="SK hynix HBF 512GB 参考节点；读写带宽和延迟是用户指定的可替换分析参数。",
    component_updates={
        "read_bandwidth_gbps": 3904.0,
        "write_bandwidth_gbps": 217.6,
        "bandwidth_gbps": 3904.0,
        "cost_profile_id": "sk_hynix_hbf_512gb_memory",
    },
    technology_updates={
        "vendor": "SK hynix",
        "generation": "HBF 2026 OCP",
        "bandwidth_scope": "user_configured_488GB_per_s_read_27_2GB_per_s_write",
        "read_bandwidth_gbps": 3904.0,
        "write_bandwidth_gbps": 217.6,
        "read_bandwidth_gb_s": 488.0,
        "write_bandwidth_gb_s": 27.2,
        "read_latency_us": 4.0,
        "write_latency_us": 75.0,
        "logical_protocol": "HBF",
        "physical_transport_protocol": "UCIe",
    },
    provenance={
        "capacity_bytes": {"value": 512, "unit": "GB_decimal", "source_field": "capacity specifications up to 512GB"},
        "read_bandwidth_gbps": {"value": 488.0, "unit": "GB/s_decimal", "formula": "user-specified 488 GB/s × 8 = 3904 Gb/s", "status": "analytical_user_configured"},
        "write_bandwidth_gbps": {"value": 27.2, "unit": "GB/s_decimal", "formula": "user-specified 27.2 GB/s × 8 = 217.6 Gb/s", "status": "analytical_user_configured"},
        "read_latency_ns": {"value": 4000.0, "unit": "ns", "formula": "user-specified 4 us × 1000", "status": "analytical_user_configured"},
        "write_latency_ns": {"value": 75000.0, "unit": "ns", "formula": "user-specified 75 us × 1000", "status": "analytical_user_configured"},
        "interface": {"value": "HBF", "unit": "logical_protocol", "physical_transport": "UCIe", "status": "analytical_protocol_alias"},
        "source_excerpt": "SK hynix announcement: HBF uses UCIe physical connection; this preset applies user-provided logical HBF service parameters",
    },
)
_SK_HYNIX_HBF.component.metadata["capability_status"] = {
    "read_bandwidth_gbps": "analytical_user_configured",
    "write_bandwidth_gbps": "analytical_user_configured",
    "read_latency_ns": "analytical_user_configured",
    "write_latency_ns": "analytical_user_configured",
    "dma_energy_pj_per_byte": "not_published",
}
_SK_HYNIX_HBF.component.metadata["facts"] = {
    "读带宽": "488 GB/s（用户指定分析参数）",
    "写带宽": "27.2 GB/s（用户指定分析参数）",
    "读延迟": "4 us（用户指定分析参数）",
    "写延迟": "75 us（用户指定分析参数）",
    "链路协议": "HBF（逻辑）；UCIe（物理承载）",
    "DMA 能耗状态": "未公开（not_published）；不得按 0 pJ/B 宣称零能耗",
}
_SK_HYNIX_HBF.component.metadata.update({
    # Opt in to the typed active-memory contract so the default B200
    # architecture can be used for KV placement scans.  The profile remains
    # analytical and is remapped per materialized component instance.
    "access_mode": "memory",
    "read_only": False,
    "writable": True,
    "write_buffer_bytes": 0,
    "memory_service_owner": "sk_hynix_hbf_512gb.memory",
    "transfer_granularity_bytes": 4096,
    "max_outstanding_requests": 32,
    "read_latency_ns": 4000.0,
    "write_latency_ns": 75000.0,
    "cost_profile_key": "host_memory",
    "cost_profile_template": {
        "bandwidth_gb_s": 488.0,
        "efficiency": 1.0,
        "energy_pj_per_byte": 12.0,
        "resource_id": "sk_hynix_hbf_512gb.memory",
        "name": "HBF analytical active-memory profile",
        "read_latency_ns": 4000.0,
        "write_latency_ns": 75000.0,
        "transaction_bytes": 4096,
        "max_outstanding_requests": 32,
        "read_bandwidth_gb_s": 488.0,
        "write_bandwidth_gb_s": 27.2,
    },
})

# The user called this product "Ti Pro9100"; YMTC's official page names it
# TiPlus9100. Keep the stable simulator ID while recording the exact official
# 1TB MPN, test conditions and decimal conversion formulas.
_TIPRO9100 = _ssd_preset(
    "ymtc-zhitai-ti-pro9100",
    "长江存储致态 TiPlus9100 1TB SSD（用户原称 Ti Pro9100）",
    family="Consumer NVMe SSD", kind="ssd", capacity_bytes=_gb(1024.0),
    read_gbps=96.0, write_gbps=85.6, interface_bandwidth_gbps=126.03076923076924,
    interface="PCIe Gen5 x4, NVMe 2.0", protocol_version="5.0", lanes=4,
    endurance="600 TBW; 5-year limited warranty",
    limitations=(
        "厂家页面产品名为 TiPlus9100；本项目保留用户原称 Ti Pro9100 作为稳定 ID。",
        "顺序读写为致态实验室内部测试峰值（不是 PCIe 链路理论值）；随机 IOPS 不折算为带宽。",
        "接口解码带宽仅用于链路上限，媒体带宽仍由厂家顺序读写字段决定。",
    ),
    notes="YMTC 官方 TiPlus9100 1TB（MPN ZTSS3CB08D6CMC）；用户请求中的 Ti Pro9100 视为该官方产品别名。",
    sources=(YMTC_ZHITAI_TIPRO9100_SOURCE,), evidence_level=S2_VENDOR_DECLARED,
    conditions=("官方实验室测试: AMD Ryzen 9 9950X + ROG STRIX X870E-E GAMING WIFI", "CrystalDiskMark 8.0.4; sequential 1MiB Q8T1; random 4KiB Q32T16", "read/write values are decimal MB/s converted to decimal Gb/s"),
)
_TIPRO9100 = _curated_clone(
    _TIPRO9100,
    preset_id="ymtc-zhitai-ti-pro9100",
    name="长江存储致态 TiPlus9100 1TB SSD（用户原称 Ti Pro9100）",
    evidence_level=S2_VENDOR_DECLARED,
    provenance={
        "mpn": {"value": "ZTSS3CB08D6CMC", "source_field": "MPN 1024GB variant"},
        "capacity_bytes": {"value": 1024, "unit": "GB_decimal", "formula": "1024 GB × 1,000,000,000 = 1,024,000,000,000 B"},
        "read_bandwidth_gbps": {"value": 12000, "unit": "MB/s_decimal", "formula": "12000 MB/s ÷ 1000 = 12 GB/s × 8 = 96 Gb/s", "test_condition": "CDM 1MiB Q8T1"},
        "write_bandwidth_gbps": {"value": 10700, "unit": "MB/s_decimal", "formula": "10700 MB/s ÷ 1000 = 10.7 GB/s × 8 = 85.6 Gb/s", "test_condition": "CDM 1MiB Q8T1"},
        "interface_bandwidth_gbps": {"value": 126.03076923076924, "unit": "Gb/s_decimal_one_way", "formula": "PCIe Gen5 ×4 decoded payload: 31.5076923077 Gb/s/lane × 4; link ceiling only"},
        "random_iops": {"value": 1850, "unit": "KIOPS", "test_condition": "CDM 4KiB Q32T16"},
        "interface": {"value": "PCIe Gen5 x4 / NVMe 2.0", "unit": "protocol", "bandwidth_scope": "link ceiling only"},
        "endurance": {"value": 600, "unit": "TBW_decimal", "scope": "1TB variant"},
    },
)
_TIPRO9100.component.metadata["measurement_basis"] = (
    "采用 YMTC 官方 TiPlus9100 1TB 实验室顺序读写峰值；容量和带宽均保留原始十进制单位及转换公式。"
)
_TIPRO9100.component.metadata["simulation_usable"] = True
_TIPRO9100.component.metadata["vendor_parameter_status"] = "vendor_declared_tested"

_SAMSUNG_DDR5 = _grace_lpddr_preset(
    "samsung-ddr5-32gb-udimm-5600",
    "Samsung DDR5 32GB UDIMM DDR5-5600 series",
    product_family="DDR5 UDIMM 32GB", bandwidth_gbps=358.4,
    source=SAMSUNG_DDR5_32GB_SOURCE,
)
_SAMSUNG_DDR5 = _curated_clone(
    _SAMSUNG_DDR5,
    preset_id="samsung-ddr5-32gb-udimm-5600",
    name="Samsung DDR5 32GB UDIMM DDR5-5600 series",
    family="Samsung DDR5 DRAM",
    limitations=(
        "32GB 指整条 UDIMM 模块容量，非单颗 DRAM die；这是 Samsung 官方 UDIMM 32GB/5600 速度等级，不虚构未在该页面列出的 MPN。",
        "Samsung 页面公开 UDIMM 最高 32GB、5600Mbps 速度；未公开端到端 CPU 控制器延迟，延迟和效率保留分析参数。",
    ),
    notes="Samsung DDR5 32GB UDIMM 模块；峰值 payload 带宽按 5600MT/s × 64-bit ÷ 8 = 44.8GB/s 换算。",
    component_updates={
        "capacity_bytes": _gb(32.0),
        "read_bandwidth_gbps": 358.4,
        "write_bandwidth_gbps": 358.4,
        "ports": (PortSpec(port_id="host", protocol="DDR5", role="device", version="DDR5-5600", lanes=64, bandwidth_gbps=358.4, metadata=_port_metadata({"data_rate_mt_s": 5600, "interface_width_bits": 64, "bandwidth_scope": "module_payload_one_way"})),),
    },
    technology_updates={"vendor": "Samsung", "generation": "DDR5", "module_capacity_gb": 32, "data_rate_mt_s": 5600, "interface_width_bits": 64, "is_module": True},
    provenance={
        "module_series": {"value": "Samsung DDR5 UDIMM 32GB / 5600 Mbps", "status": "vendor_family_page_declared"},
        "capacity_bytes": {"value": 32, "unit": "GB_decimal", "scope": "one_UDIMM_module"},
        "data_rate": {"value": 5600, "unit": "MT/s", "source_field": "DDR5 speed grade"},
                    "interface_width": {"value": 64, "unit": "bit_payload", "ECC_note": "bandwidth uses the 64-bit payload width; any ECC/physical overhead is not included"},
        "read_bandwidth_gbps": {"formula": "5600 MT/s × 64 bit ÷ 8 = 44.8 GB/s × 8 = 358.4 Gb/s", "unit": "Gb/s_decimal_one_way"},
        "write_bandwidth_gbps": {"formula": "same DDR5 payload envelope", "status": "derived_directional_envelope"},
        "source_excerpt": "Samsung DDR page: UDIMM capacity up to 32GB and speed up to 5,600Mbps at 1.1V",
    },
)
_SAMSUNG_DDR5.component.metadata["physical_composition"] = {
    "simulator_representation": "single_module_node",
    "physical_unit_kind": "DDR5_UDIMM_module",
    "physical_unit_count": 1,
    "physical_unit_count_status": "one_modeled_module",
    "unit_capacity_bytes": _gb(32.0),
    "module_series": "Samsung DDR5 UDIMM 32GB / 5600 Mbps",
    "source_basis": "Samsung DDR module page; 32GB/5600 speed class recorded as the selected module profile",
}
_SAMSUNG_DDR5.component.metadata["parameter_basis"] = {
    "capacity_bytes": "one 32GB module; not a die or a system aggregate",
    "data_rate_mt_s": "Samsung DDR5 module speed class 5600 MT/s",
    "read_bandwidth_gbps": "5600 MT/s × 64-bit payload ÷ 8 × 8 = 358.4 Gb/s",
    "write_bandwidth_gbps": "directional payload envelope; controller efficiency remains analytical",
}
_SAMSUNG_DDR5.component.metadata["value_scope"] = "单条 Samsung DDR5 32GB UDIMM 模块"
_SAMSUNG_DDR5.component.metadata["conditions"] = [
    "Samsung DDR5 module page; DDR5-5600 speed class",
    "64-bit payload width; ECC bits excluded from payload bandwidth",
    "latency/efficiency are editable analytical controller defaults",
]

# Local measured configuration from Win32_PhysicalMemory on the development
# host: four 32 GiB DIMMs, two memory channels, and a currently trained
# DDR5-5600 data rate.  This is a machine snapshot, not a vendor SKU claim.
_ACER_LOCAL_DDR5 = _curated_clone(
    _SAMSUNG_DDR5,
    preset_id="acer-local-ddr5-128gb-5600-dual-channel",
    name="本机 Acer DDR5 128 GiB DDR5-5600 双通道",
    family="Local measured DDR5",
    sources=(),
    evidence_level=S4_PRIMARY_RESEARCH,
    limitations=(
        "这是本机 Win32_PhysicalMemory 快照，不是 Acer 的公开产品 SKU；更换 DIMM、BIOS memory training 或通道数后应重新读取。",
        "第二组 DIMM 的 SPD Speed 为 4800，但当前 ConfiguredClockSpeed 为 5600；带宽按当前配置速率和两个通道计算。",
        "系统未提供端到端内存访问延迟；延迟、效率和事务粒度仍是可编辑分析参数。",
    ),
    notes="本机实测安装配置：4×32 GiB，A/B 双通道，当前 DDR5-5600；理论 payload 带宽 5600 MT/s × 64 bit ÷ 8 × 2 = 89.6 GB/s。",
    component_updates={
        "capacity_bytes": 4 * 32 * 1024 ** 3,
        "bandwidth_gbps": 716.8,
        "read_bandwidth_gbps": 716.8,
        "write_bandwidth_gbps": 716.8,
        "ports": (PortSpec(
            port_id="host",
            protocol="DDR5",
            role="device",
            version="DDR5-5600",
            lanes=128,
            bandwidth_gbps=716.8,
            metadata=_port_metadata({
                "data_rate_mt_s": 5600,
                "interface_width_bits": 64,
                "channel_count": 2,
                "bandwidth_scope": "configured_system_payload_one_way",
            }),
        ),),
    },
    technology_updates={
        "vendor": "Acer (SMBIOS manufacturer)",
        "generation": "DDR5",
        "capacity_gib": 128,
        "installed_dimm_count": 4,
        "module_capacity_gib": 32,
        "configured_data_rate_mt_s": 5600,
        "spd_data_rates_mt_s": [5600, 4800, 5600, 4800],
        "channel_count": 2,
        "part_numbers": ["BL.9BWWR.424", "BL.9BWWR.373"],
    },
    provenance={
        "manufacturer": {"value": "Acer", "source_field": "Win32_PhysicalMemory.Manufacturer", "status": "local_snapshot"},
        "part_numbers": {"value": ["BL.9BWWR.424", "BL.9BWWR.373"], "source_field": "Win32_PhysicalMemory.PartNumber", "status": "local_snapshot"},
        "capacity_bytes": {"value": 137438953472, "unit": "B", "formula": "4 × 32 GiB", "source_field": "Win32_PhysicalMemory.Capacity", "status": "local_snapshot"},
        "configured_data_rate_mt_s": {"value": 5600, "unit": "MT/s", "source_field": "Win32_PhysicalMemory.ConfiguredClockSpeed", "status": "local_snapshot"},
        "bandwidth_gb_s": {"value": 89.6, "unit": "GB/s_decimal", "formula": "5600 MT/s × 64 bit ÷ 8 × 2 channels", "status": "derived_from_local_snapshot"},
    },
)
_ACER_LOCAL_DDR5 = replace(
    _ACER_LOCAL_DDR5,
    sources=(),
    component=replace(
        _ACER_LOCAL_DDR5.component,
        metadata={
            **_ACER_LOCAL_DDR5.component.metadata,
            "local_hardware_snapshot": {
                "source": "Win32_PhysicalMemory",
                "captured_at": "2026-09-27",
                "manufacturer": "Acer",
                "part_numbers": ["BL.9BWWR.424", "BL.9BWWR.373"],
                "module_count": 4,
                "module_capacity_bytes": 34359738368,
                "capacity_bytes": 137438953472,
                "spd_speed_mt_s": [5600, 4800, 5600, 4800],
                "configured_speed_mt_s": 5600,
                "channel_count": 2,
                "theoretical_bandwidth_gb_s": 89.6,
            },
            "facts": {
                "本机厂商": "Acer（SMBIOS）",
                "料号": "BL.9BWWR.424 ×2；BL.9BWWR.373 ×2",
                "安装容量": "128 GiB（4 × 32 GiB）",
                "当前速率": "5600 MT/s",
                "通道": "双通道",
                "理论带宽": "89.6 GB/s",
            },
            "sources": [],
        },
    ),
)
_ACER_LOCAL_PROFILE, _ACER_LOCAL_PROFILE_BASIS = _host_memory_cost_profile_template(
    "acer_local_ddr5_128gb_5600_dual_channel",
    716.8,
    name="acer-local-ddr5-128gb-5600-dual-channel-memory-profile",
    read_latency_ns=100.0,
    write_latency_ns=100.0,
    transaction_bytes=256,
    max_outstanding_requests=32,
)
_ACER_LOCAL_DDR5 = replace(
    _ACER_LOCAL_DDR5,
    component=replace(
        _ACER_LOCAL_DDR5.component,
        metadata={
            **_ACER_LOCAL_DDR5.component.metadata,
            "cost_profile_template": _ACER_LOCAL_PROFILE,
            "cost_profile_parameter_basis": _ACER_LOCAL_PROFILE_BASIS,
        },
    ),
)

_SRAM_CIM = _curated_clone(
    _legacy_component("digital-sram-cim-analysis"),
    preset_id="sram-cim-analytical-tile",
    name="SRAM 存算一体分析 Tile（非量产 SKU）",
    family="SRAM Compute-in-Memory",
    evidence_level=A_ANALYTICAL,
    limitations=(
        "仅为分析参考，不对应公开量产器件或特定厂家 SKU。",
        "容量、OPS、片上带宽、能效和频率均需按研究假设或实测校准。",
    ),
    notes="保留一个 SRAM-CIM 分析模板，明确不冒充厂家参数。",
    provenance={
        "all_fields": {"status": "analytical_reference_only", "source": "no vendor SKU claimed"},
    },
)

_B200_GPU = _gpu_preset(
    "nvidia-b200-sxm-gpu",
    "NVIDIA B200 SXM GPU (180GB HBM3E, 8TB/s)",
    hbm_generation="HBM3E", hbm_stack_count=8, device_memory_gb=180.0,
    device_memory_bandwidth_gbps=64000.0,
    bf16_dense_tflops=2250.0, fp8_dense_tflops=4500.0,
    fp8_sparse_tflops=9000.0, bf16_sparse_tflops=4500.0,
    tdp_watts=1000, mig_profiles="未公开 MIG 配置",
    sources=(NVIDIA_DGX_B200_OFFICIAL_SOURCE, NVIDIA_B200_OFFICIAL_SOURCE),
)
_B200_GPU = _curated_clone(
    _B200_GPU,
    preset_id="nvidia-b200-sxm-gpu",
    name="NVIDIA B200 SXM GPU (180GB HBM3E, 8TB/s)",
    family="NVIDIA Blackwell GPU",
    notes="NVIDIA B200 单 GPU 规格按 DGX B200/Blackwell 厂家聚合值折算：180GB HBM3E、8TB/s、dense BF16 2.25PFLOPS。",
    component_updates={
        # GPU ComponentSpec is the compute endpoint.  Its attached HBM must
        # be represented by explicit hbm components/links; otherwise these
        # fields would double count memory in architecture_scan and serving.
        "capacity_bytes": 0,
        "read_bandwidth_gbps": 0.0,
        "write_bandwidth_gbps": 0.0,
    },
    technology_updates={"architecture": "NVIDIA Blackwell", "vendor": "NVIDIA", "device_memory_gb": 180.0, "device_memory_bandwidth_gbps": 64000.0, "bf16_tensor_dense_tflops": 2250.0, "tdp_watts": 1000},
    provenance={
        "device_memory_gb": {"value": 180, "unit": "GB_decimal", "formula": "DGX B200 1440GB ÷ 8 GPUs"},
        "device_memory_bandwidth_gbps": {"value": 64000, "unit": "Gb/s_decimal_one_way", "formula": "DGX B200 64TB/s ÷ 8 GPUs × 8"},
        "component_read_bandwidth_gbps": {"formula": "device memory aggregate 8TB/s × 8 = 64000Gb/s", "scope": "attached HBM3E aggregate"},
        "component_write_bandwidth_gbps": {"formula": "same directional HBM3E aggregate envelope", "status": "derived_directional_envelope"},
        "bf16_dense_tflops": {"value": 2250, "unit": "TFLOP/s", "status": "vendor Blackwell dense tensor aggregate"},
        "tdp_watts": {"value": 1000, "unit": "W", "status": "vendor product power class"},
    },
)
_B200_GPU_PORTS = tuple(
    PortSpec(
        port_id="hbm{}".format(index), protocol="HBM", role="controller",
        version="HBM3E", lanes=16, bandwidth_gbps=8000.0,
        metadata=_port_metadata({"expected_stack_index": index, "bandwidth_scope": "analytical_equal_share_of_8TB_per_s"}),
    )
    for index in range(8)
) + (
    PortSpec(
        port_id="nvlink0", protocol="NVLink", role="endpoint", version="5.0",
        lanes=18, bandwidth_gbps=7200.0,
        metadata=_port_metadata({"bandwidth_scope": "one_way", "vendor_display_value": "1.8 TB/s bidirectional per GPU"}),
    ),
    PortSpec(
        port_id="pcie0", protocol="PCIe", role="endpoint", version="6.0",
        lanes=16, bandwidth_gbps=1008.2461538461539,
        payload="NVMe/control",
        metadata=_port_metadata({"bandwidth_scope": "analytical_decoded_link_ceiling", "source_status": "protocol_ceiling_not_B200_media_bandwidth"}),
    ),
)
_B200_GPU = replace(_B200_GPU, component=replace(_B200_GPU.component, ports=_B200_GPU_PORTS))
_B200_GPU.component.metadata["technology"].update({
    "memory_stack_count": None,
    "memory_stack_count_status": "not_vendor_disclosed",
    "attached_memory_scope": "DGX B200 8-GPU aggregate divided by 8",
})
_B200_GPU.component.metadata["technology"].pop("mig_profiles", None)
_B200_GPU.component.metadata["component_scope"] = "B200 compute endpoint; attached HBM3E requires explicit memory components"
_B200_GPU.component.metadata["external_memory_modeled_in_metadata"] = False
_B200_GPU.component.metadata["memory_requires_explicit_component"] = True
_B200_GPU.component.metadata["capability_applicability"] = {
    "capacity_bytes": "not_applicable_on_gpu_compute_endpoint; use hbm.capacity_bytes",
    "read_bandwidth_gbps": "not_applicable_on_gpu_compute_endpoint; use hbm.read_bandwidth_gbps and links",
    "write_bandwidth_gbps": "not_applicable_on_gpu_compute_endpoint; use hbm.write_bandwidth_gbps and links",
    "peak_ops_per_s": "GPU compute capability",
}
_B200_GPU.component.metadata["conditions"] = [
    "NVIDIA DGX B200: 1,440GB total / 8 GPUs = 180GB per GPU",
    "NVIDIA DGX B200: 64TB/s total HBM3E / 8 GPUs = 8TB/s per GPU",
    "dense BF16 2.25PFLOPS is a Blackwell architecture reference; scheduler/cache/efficiency fields are analytical",
]
_B200_GPU.component.metadata["derived_formula"] = "DGX B200 aggregate product values divided by 8 GPUs; decimal TB/s × 8000 = IR Gb/s"
_B200_GPU.component.metadata["cost_profile_parameter_basis"] = {
    key: str(value).replace("Hopper", "Blackwell")
    for key, value in _B200_GPU.component.metadata.get("cost_profile_parameter_basis", {}).items()
}

_RYZEN_9_9950X3D = _desktop_cpu_preset(
    "amd-ryzen-9-9950x3d",
    "AMD Ryzen 9 9950X3D（16 核 / 32 线程）",
    model="AMD Ryzen 9 9950X3D",
    core_count=16,
    thread_count=32,
    base_clock_ghz=4.3,
    boost_clock_ghz=5.7,
    l2_capacity_bytes=16 * 1024 * 1024,
    l3_capacity_bytes=128 * 1024 * 1024,
    tdp_watts=170,
    memory_data_rate_mt_s=5600,
    memory_channels=2,
    source=AMD_RYZEN_9_9950X3D_SOURCE,
)

_RTX_5080 = _consumer_gpu_preset(
    "nvidia-rtx-5080",
    "NVIDIA GeForce RTX 5080 16GB GDDR7",
    model="NVIDIA GeForce RTX 5080",
    architecture="NVIDIA Blackwell",
    cuda_cores=10752,
    sm_count=84,
    tensor_cores_per_sm=4,
    boost_clock_ghz=2.617,
    memory_gb=16.0,
    memory_data_rate_gbps=30.0,
    memory_bus_bits=256,
    memory_bandwidth_gb_s=960.0,
    l2_capacity_bytes=64 * 1024 * 1024,
    tgp_watts=360,
    bf16_dense_tflops=112.6,
    source=NVIDIA_RTX_5080_SOURCE,
    architecture_source=NVIDIA_BLACKWELL_ARCHITECTURE_PDF_SOURCE,
)

_PRESETS: Tuple[ComponentPresetDefinition, ...] = (
    _SAMSUNG_HBM3E,
    _GDDR6_REFERENCE,
    _GDDR6X_REFERENCE,
    _RTX5080_GDDR7,
    _SK_HYNIX_HBF,
    _TIPRO9100,
    _SAMSUNG_DDR5,
    _ACER_LOCAL_DDR5,
    _SRAM_CIM,
    _B200_GPU,
    _RYZEN_9_9950X3D,
    _RTX_5080,
)

# Legacy definitions are intentionally not registered.  The curated catalog
# is the complete hardware component surface; old IDs must fail closed in the
# public detail/materialize APIs.
_LEGACY_PRESETS = ()


_COMPONENT_BY_ID: Mapping[str, ComponentPresetDefinition] = {
    **{item.preset_id: item for item in _LEGACY_PRESETS},
    **{item.preset_id: item for item in _PRESETS},
}
if len(_COMPONENT_BY_ID) != len({item.preset_id for item in (*_LEGACY_PRESETS, *_PRESETS)}):  # import-time invariant, never user input
    raise RuntimeError("组件预设 ID 重复")
if any(item.component_kind not in COMPONENT_KINDS for item in _PRESETS):
    raise RuntimeError("组件预设使用了不支持的 ComponentSpec.kind")



_BUNDLES: Tuple[TopologyBundleDefinition, ...] = ()


_BUNDLE_BY_ID: Mapping[str, TopologyBundleDefinition] = {
    item.preset_id: item for item in _BUNDLES
}
_BY_ID: Mapping[str, Any] = {**_COMPONENT_BY_ID, **_BUNDLE_BY_ID}
if len(_BY_ID) != len(set(_COMPONENT_BY_ID) | set(_BUNDLE_BY_ID)):
    raise RuntimeError("组件或组合拓扑预设 ID 重复")

class ComponentPresetMutationError(ValueError):
    """A client-correctable component-preset mutation error."""

    def __init__(self, code: str, message: str, *, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _catalog_definitions() -> Tuple[Any, ...]:
    return _PRESETS


def _metadata(definition: Any) -> Dict[str, Any]:
    if isinstance(definition, TopologyBundleDefinition):
        aggregate_capacity = sum(component.capacity_bytes for component in definition.components)
        root = next(
            component
            for component in definition.components
            if component.component_id == definition.group["root"]
        )
        memory_bandwidth = sum(
            component.read_bandwidth_gbps
            for component in definition.components
            if component.kind == "hbm"
        )
        return {
            "id": definition.preset_id,
            "name": definition.name,
            "family": definition.family,
            "preset_type": "topology_bundle",
            "component_kind": "topology_bundle",
            "root_component_kind": root.kind,
            "component_count": len(definition.components),
            "link_count": len(definition.links),
            "capacity_bytes": aggregate_capacity,
            "peak_ops_per_s": root.peak_ops_per_s,
            "read_bandwidth_gbps": memory_bandwidth,
            "write_bandwidth_gbps": memory_bandwidth,
            "capability_units": dict(CAPABILITY_UNITS),
            "port_count": sum(len(component.ports) for component in definition.components),
            "evidence_level": definition.evidence_level,
            "sources": list(_source_metadata(definition.sources)),
            "limitations": list(definition.limitations),
            "notes": definition.notes,
            "value_scope": "一个 GPU 根、显式 HBM 堆叠、内部专用链路和视觉分组",
            "conditions": ["原子载入", "默认展开", "GPU 为分组根", "一次撤销"],
            "revision": "catalog-{}".format(CATALOG_VERSION),
            "expires_at": "2027-08-22",
            "supersedes": [],
            "tags": list(definition.tags),
            "catalog_version": CATALOG_VERSION,
            "usage_hint": BUNDLE_USAGE_HINT,
        }
    component = definition.component
    component_metadata = component.metadata
    return {
        "id": definition.preset_id,
        "name": definition.name,
        "family": definition.family,
        "preset_type": "component",
        "component_kind": component.kind,
        "capacity_bytes": component.capacity_bytes,
        "peak_ops_per_s": component.peak_ops_per_s,
        "read_bandwidth_gbps": component.read_bandwidth_gbps,
        "write_bandwidth_gbps": component.write_bandwidth_gbps,
        "capability_units": dict(CAPABILITY_UNITS),
        "port_count": len(component.ports),
        "evidence_level": definition.evidence_level,
        "sources": list(_source_metadata(definition.sources)),
        "limitations": list(definition.limitations),
        "notes": definition.notes,
        "facts": dict(component_metadata.get("facts", {})),
        "capability_status": dict(component_metadata.get("capability_status", {})),
        "value_scope": component_metadata.get("value_scope", ""),
        "conditions": list(component_metadata.get("conditions", ())),
        "revision": component_metadata.get("revision", ""),
        "expires_at": component_metadata.get("expires_at", ""),
        "supersedes": list(component_metadata.get("supersedes", ())),
        "tags": list(definition.tags),
        "catalog_version": CATALOG_VERSION,
        "usage_hint": USAGE_HINT,
    }


def list_component_presets() -> Tuple[Dict[str, Any], ...]:
    """List stable metadata ordered by URL-safe preset ID."""

    return tuple(
        _metadata(item)
        for item in sorted(_catalog_definitions(), key=lambda item: item.preset_id)
    )


def component_preset_filters(items=None) -> Dict[str, Any]:
    """Return available filter values for the list API envelope."""

    items = list_component_presets() if items is None else items
    tags = sorted({tag for item in items for tag in item["tags"]})
    return {
        "component_kind": sorted({item["component_kind"] for item in items}),
        "family": sorted({item["family"] for item in items}),
        "evidence_level": sorted({item["evidence_level"] for item in items}),
        "tag": tags,
    }


def component_preset_page(
    *,
    query: str = "",
    family: str = "",
    component_kind: str = "",
    evidence_level: str = "",
    tag: str = "",
    items=None,
) -> Dict[str, Any]:
    """Return a small filtered list envelope for UI integration."""

    normalized_query = query.strip().lower()
    normalized_family = family.strip().lower()
    normalized_kind = component_kind.strip().lower()
    normalized_evidence = evidence_level.strip().lower()
    normalized_tag = tag.strip().lower()
    catalog_items = list_component_presets() if items is None else items
    items = []
    for item in catalog_items:
        if normalized_family and item["family"].lower() != normalized_family:
            continue
        if normalized_kind and item["component_kind"].lower() != normalized_kind:
            continue
        if normalized_evidence and item["evidence_level"].lower() != normalized_evidence:
            continue
        if normalized_tag and normalized_tag not in {value.lower() for value in item["tags"]}:
            continue
        if normalized_query:
            searchable = " ".join(
                [
                    item["id"],
                    item["name"],
                    item["family"],
                    item["component_kind"],
                    item["evidence_level"],
                    item["notes"],
                    " ".join(item["tags"]),
                ]
            ).lower()
            if normalized_query not in searchable:
                continue
        items.append(item)
    return {
        "items": items,
        "total": len(items),
        "filters": component_preset_filters(catalog_items),
        "query": {
            "query": query,
            "family": family,
            "component_kind": component_kind,
            "evidence_level": evidence_level,
            "tag": tag,
        },
        "catalog": {
            "version": CATALOG_VERSION,
            "cutoff_at": CATALOG_CUTOFF_AT,
        },
        "usage_hint": USAGE_HINT,
    }


def get_component_preset(preset_id: str) -> Any:
    """Return an immutable catalog definition, raising ``KeyError`` if absent."""

    return _BY_ID[preset_id]


def materialize_component_payload(preset_id: str) -> Dict[str, Any]:
    """Return one JSON-compatible ``ComponentSpec`` template."""

    definition = get_component_preset(preset_id)
    if not isinstance(definition, ComponentPresetDefinition):
        raise TypeError("组合拓扑预设不能物化为单组件 payload")
    return to_primitive(definition.component)


def component_preset_detail(preset_id: str) -> Dict[str, Any]:
    """Build an API detail object for a component or topology bundle."""

    definition = get_component_preset(preset_id)
    return _component_definition_detail(definition)


def _component_definition_detail(definition: Any) -> Dict[str, Any]:
    if isinstance(definition, TopologyBundleDefinition):
        return {
            "preset": _metadata(definition),
            "components": [to_primitive(component) for component in definition.components],
            "links": [to_primitive(link) for link in definition.links],
            "group": {
                **dict(definition.group),
                "members": list(definition.group["members"]),
            },
            "usage_hint": BUNDLE_USAGE_HINT,
            "catalog": {
                "version": CATALOG_VERSION,
                "cutoff_at": CATALOG_CUTOFF_AT,
            },
        }
    return {
        "preset": _metadata(definition),
        "component": to_primitive(definition.component),
        "links": [],
        "usage_hint": USAGE_HINT,
        "catalog": {
            "version": CATALOG_VERSION,
            "cutoff_at": CATALOG_CUTOFF_AT,
        },
    }


_PRESET_MUTATION_FIELDS = frozenset({
    "id", "preset_id", "name", "family", "component", "component_spec",
    "evidence_level", "limitations", "notes", "tags", "sources",
})
_PRESET_MUTATION_IGNORED_FIELDS = frozenset({
    "preset_type", "component_kind", "root_component_kind", "component_count", "link_count",
    "capacity_bytes", "peak_ops_per_s", "read_bandwidth_gbps", "write_bandwidth_gbps",
    "capability_units", "port_count", "evidence_level", "sources", "limitations", "notes",
    "value_scope", "conditions", "revision", "expires_at", "supersedes", "tags",
    "catalog_version", "usage_hint", "catalog", "links",
    "capability_status", "facts",
})
_COMPONENT_MUTATION_FIELDS = frozenset({
    "component_id", "kind", "cost_profile_id", "ports", "package_id", "die_id",
    "capacity_bytes", "peak_ops_per_s", "read_bandwidth_gbps",
    "write_bandwidth_gbps", "bandwidth_gbps", "metadata", "schema_version",
})
_PORT_MUTATION_FIELDS = frozenset({
    "port_id", "protocol", "role", "direction", "version", "lanes",
    "bandwidth_gbps", "max_links", "payload", "metadata", "schema_version",
})
_PRESET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,80}$")


def _mutation_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComponentPresetMutationError("invalid_payload", "{} 必须是对象".format(label))
    return dict(value)


def _mutation_list(value: Any, label: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise ComponentPresetMutationError("invalid_payload", "{} 必须是数组".format(label))
    return list(value)


def _validate_metadata_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata 对象键必须是字符串")
            _validate_metadata_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_metadata_keys(item)


def _strict_payload_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return (
            set(left) == set(right)
            and all(_strict_payload_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_payload_equal(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    return left == right


def _parse_port_payload(value: Any, index: int) -> PortSpec:
    payload = _mutation_mapping(value, "component.ports[{}]".format(index))
    unknown = sorted(set(payload) - _PORT_MUTATION_FIELDS)
    if unknown:
        raise ComponentPresetMutationError("unknown_fields", "端口包含未知字段：{}".format(", ".join(unknown)))
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("direction", "bidirectional")
    payload.setdefault("version", "1.0")
    payload.setdefault("lanes", 1)
    payload.setdefault("bandwidth_gbps", 0.0)
    payload.setdefault("max_links", 1)
    metadata = dict(_mutation_mapping(payload.get("metadata", {}), "port.metadata"))
    try:
        _validate_metadata_keys(metadata)
        to_primitive(metadata)
    except (TypeError, ValueError) as exc:
        raise ComponentPresetMutationError("invalid_component", "端口 metadata 无法序列化：{}".format(exc)) from exc
    payload["metadata"] = metadata
    try:
        return PortSpec(**payload)
    except (TypeError, ValueError) as exc:
        raise ComponentPresetMutationError("invalid_component", "端口参数无效：{}".format(exc)) from exc


def _parse_component_payload(value: Any) -> ComponentSpec:
    payload = _mutation_mapping(value, "component")
    unknown = sorted(set(payload) - _COMPONENT_MUTATION_FIELDS)
    if unknown:
        raise ComponentPresetMutationError("unknown_fields", "组件包含未知字段：{}".format(", ".join(unknown)))
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["ports"] = tuple(_parse_port_payload(item, index) for index, item in enumerate(_mutation_list(payload.get("ports", ()), "component.ports")))
    payload.setdefault("capacity_bytes", 0)
    payload.setdefault("peak_ops_per_s", 0.0)
    payload.setdefault("read_bandwidth_gbps", 0.0)
    payload.setdefault("write_bandwidth_gbps", 0.0)
    # Keep an omitted shared total omitted. The IR derives active-memory
    # defaults; flash/SSD write capability must never be inferred from reads.
    payload.setdefault("bandwidth_gbps", 0.0)
    if str(payload.get("kind", "")).strip().lower().replace("-", "_") == "gpu":
        # GPU presets own compute capability.  Device-memory capacity and
        # media bandwidth must be represented by explicit HBM/memory nodes.
        payload["capacity_bytes"] = 0
        payload["read_bandwidth_gbps"] = 0.0
        payload["write_bandwidth_gbps"] = 0.0
        payload["bandwidth_gbps"] = 0.0
    metadata = dict(_mutation_mapping(payload.get("metadata", {}), "component.metadata"))
    for field in ("facts", "capability_status"):
        if field in metadata and not isinstance(metadata[field], Mapping):
            raise ComponentPresetMutationError("invalid_component", "component.metadata.{} 必须是对象".format(field))
    for field in ("conditions", "supersedes"):
        if field in metadata and (
            not isinstance(metadata[field], (list, tuple))
            or any(not isinstance(item, str) for item in metadata[field])
        ):
            raise ComponentPresetMutationError("invalid_component", "component.metadata.{} 必须是字符串数组".format(field))
    for field in ("value_scope", "revision", "expires_at"):
        if field in metadata and not isinstance(metadata[field], str):
            raise ComponentPresetMutationError("invalid_component", "component.metadata.{} 必须是字符串".format(field))
    try:
        _validate_metadata_keys(metadata)
        to_primitive(metadata)
    except (TypeError, ValueError) as exc:
        raise ComponentPresetMutationError("invalid_component", "组件 metadata 无法序列化：{}".format(exc)) from exc
    template = metadata.get("cost_profile_template")
    # Profile rates describe a calibrated service model and are bounded by
    # hardware during scenario resolution. Preserve them across catalog edits.
    payload["metadata"] = metadata
    try:
        return ComponentSpec(**payload)
    except (TypeError, ValueError) as exc:
        raise ComponentPresetMutationError("invalid_component", "组件参数无效：{}".format(exc)) from exc


def _validate_ignored_fields(raw: Mapping[str, Any], component: ComponentSpec, *, fields: Optional[Sequence[str]] = None) -> None:
    metadata = component.metadata
    expected = {
        "preset_type": "component",
        "component_kind": component.kind,
        "capacity_bytes": component.capacity_bytes,
        "peak_ops_per_s": component.peak_ops_per_s,
        "read_bandwidth_gbps": component.read_bandwidth_gbps,
        "write_bandwidth_gbps": component.write_bandwidth_gbps,
        "capability_units": dict(CAPABILITY_UNITS),
        "port_count": len(component.ports),
        "facts": dict(metadata.get("facts", {})),
        "capability_status": dict(metadata.get("capability_status", {})),
        "value_scope": metadata.get("value_scope", ""),
        "conditions": list(metadata.get("conditions", ())),
        "revision": metadata.get("revision", ""),
        "expires_at": metadata.get("expires_at", ""),
        "supersedes": list(metadata.get("supersedes", ())),
        "catalog_version": CATALOG_VERSION,
        "usage_hint": USAGE_HINT,
        "links": [],
        "catalog": {"version": CATALOG_VERSION, "cutoff_at": CATALOG_CUTOFF_AT},
    }
    fields = set(raw) if fields is None else set(fields)
    type_conflicts = {
        field
        for field in ("capacity_bytes", "port_count", "peak_ops_per_s", "read_bandwidth_gbps", "write_bandwidth_gbps")
        if field in fields and isinstance(raw.get(field), bool)
    }
    conflicts = sorted(
        field for field, expected_value in expected.items()
        if field in fields
        and field not in _PRESET_MUTATION_FIELDS
        and (field in type_conflicts or not _strict_payload_equal(raw[field], expected_value))
    )
    if conflicts:
        raise ComponentPresetMutationError(
            "conflicting_fields",
            "只读派生字段与 component 不一致：{}".format(", ".join(conflicts)),
        )


_SOURCE_FIELDS = frozenset({
    "title", "url", "evidence_level", "publisher", "published_at", "accessed_at",
})


def _parse_sources(value: Any, *, strict: bool = False) -> Tuple[ComponentSource, ...]:
    if value is None:
        return ()
    sources = []
    for index, raw in enumerate(_mutation_list(value, "sources")):
        source = _mutation_mapping(raw, "sources[{}]".format(index))
        if strict:
            unknown = sorted(set(source) - _SOURCE_FIELDS)
            if unknown:
                raise ComponentPresetMutationError("invalid_source", "来源包含未知字段：{}".format(", ".join(unknown)))
            if any(not isinstance(source.get(field), str) for field in source):
                raise ComponentPresetMutationError("invalid_source", "来源字段必须是字符串")
        title = str(source.get("title", "")).strip()
        url = str(source.get("url", "")).strip()
        publisher = str(source.get("publisher", "")).strip() or "用户录入"
        evidence = str(source.get("evidence_level", S2_VENDOR_DECLARED)).strip()
        if not title or (not url and evidence != A_ANALYTICAL):
            raise ComponentPresetMutationError("invalid_source", "来源必须包含 title 和 url")
        if evidence not in EVIDENCE_LEVELS:
            raise ComponentPresetMutationError("invalid_source", "来源 evidence_level 无效：{}".format(evidence))
        sources.append(ComponentSource(title, url, evidence, publisher, str(source.get("published_at", "")), str(source.get("accessed_at", ""))))
    return tuple(sources)


def _definition_from_mutation(payload: Mapping[str, Any], *, existing: Optional[ComponentPresetDefinition] = None, strict: bool = False) -> ComponentPresetDefinition:
    raw = _mutation_mapping(payload, "payload")
    validation_fields = set(raw)
    wrapper_fields = set()
    if isinstance(raw.get("preset"), Mapping):
        preset_fields = dict(raw["preset"])
        wrapper_fields = set(preset_fields)
        conflicting = sorted(
            key for key in set(preset_fields) & (set(raw) - {"preset"})
            if not _strict_payload_equal(preset_fields[key], raw[key])
        )
        if conflicting:
            raise ComponentPresetMutationError("conflicting_fields", "preset 包装层与外层字段冲突：{}".format(", ".join(conflicting)))
        raw = {**preset_fields, **{key: value for key, value in raw.items() if key != "preset"}}
    if "id" in raw and "preset_id" in raw and not _strict_payload_equal(raw["id"], raw["preset_id"]):
        raise ComponentPresetMutationError("conflicting_fields", "id 和 preset_id 不能指向不同预设")
    if "component" in raw and "component_spec" in raw and not _strict_payload_equal(raw["component"], raw["component_spec"]):
        raise ComponentPresetMutationError("conflicting_fields", "component 和 component_spec 不能同时使用不同值")
    unknown = sorted(set(raw) - _PRESET_MUTATION_FIELDS - _PRESET_MUTATION_IGNORED_FIELDS)
    if unknown:
        raise ComponentPresetMutationError("unknown_fields", "预设包含未知字段：{}".format(", ".join(unknown)))
    if strict:
        for field in ("id", "preset_id"):
            if field in raw and not isinstance(raw[field], str):
                raise ComponentPresetMutationError("invalid_id", "{} 必须是字符串".format(field))
        for field in ("name", "family", "evidence_level", "notes"):
            if field in raw and not isinstance(raw[field], str):
                raise ComponentPresetMutationError("invalid_payload", "预设字段必须是字符串：{}".format(field))
        for field in ("limitations", "tags"):
            if field in raw and (
                not isinstance(raw[field], list)
                or any(not isinstance(item, str) for item in raw[field])
            ):
                raise ComponentPresetMutationError("invalid_payload", "预设字段必须是字符串数组：{}".format(field))
        if "sources" in raw and raw["sources"] is None:
            raise ComponentPresetMutationError("invalid_source", "sources 必须是数组")
    base_component = existing.component if existing else None
    component_payload = raw.get("component", raw.get("component_spec"))
    if component_payload is None and base_component is not None:
        component_payload = to_primitive(base_component)
    if component_payload is None:
        raise ComponentPresetMutationError("missing_component", "缺少必填字段 component")
    component = _parse_component_payload(component_payload)
    if strict:
        if wrapper_fields and (
            existing is None
            or component_payload == to_primitive(existing.component)
        ):
            validation_fields |= wrapper_fields
        _validate_ignored_fields(raw, component, fields=validation_fields)
    preset_id = str(raw.get("id", raw.get("preset_id", existing.preset_id if existing else ""))).strip()
    if not _PRESET_ID_RE.fullmatch(preset_id):
        raise ComponentPresetMutationError("invalid_id", "id 必须是 2-81 个字母、数字、点、下划线或连字符")
    name = str(raw.get("name", existing.name if existing else component.component_id)).strip()
    family = str(raw.get("family", existing.family if existing else "用户自定义")).strip()
    if not name or not family:
        raise ComponentPresetMutationError("missing_field", "name 和 family 不能为空")
    evidence = str(raw.get("evidence_level", existing.evidence_level if existing else A_ANALYTICAL)).strip()
    if evidence not in EVIDENCE_LEVELS:
        raise ComponentPresetMutationError("invalid_evidence_level", "evidence_level 无效：{}".format(evidence))
    limitations = tuple(str(item).strip() for item in raw.get("limitations", existing.limitations if existing else ()) if str(item).strip())
    notes = str(raw.get("notes", existing.notes if existing else "用户自定义硬件预设")).strip()
    tags = tuple(str(item).strip() for item in raw.get("tags", existing.tags if existing else ()) if str(item).strip())
    inherited_sources = [source.to_metadata() for source in existing.sources] if existing else ()
    sources = _parse_sources(raw.get("sources", inherited_sources), strict=strict)
    return ComponentPresetDefinition(preset_id, name, family, component, sources, evidence, limitations, notes, tags)


def _definition_from_persisted_record(value: Any) -> ComponentPresetDefinition:
    raw = _mutation_mapping(value, "presets")
    if "preset" in raw:
        raise ValueError("硬件预设目录记录不允许使用 preset 包装层")
    if "id" in raw and "preset_id" in raw:
        if not isinstance(raw["id"], str) or not isinstance(raw["preset_id"], str) or not _strict_payload_equal(raw["id"], raw["preset_id"]):
            raise ValueError("硬件预设目录中的 id 和 preset_id 冲突")
    if "component" in raw and "component_spec" in raw and not _strict_payload_equal(raw["component"], raw["component_spec"]):
        raise ValueError("硬件预设目录中的 component 和 component_spec 冲突")
    return _definition_from_mutation(raw, strict=True)


class ComponentPresetCatalog:
    """Per-server editable catalog, persisted atomically in a user-owned file.

    The lock serializes requests handled by the local HTTP server. Bundled
    definitions never change, and page/detail refresh user state under the
    instance lock (module imports remain deterministic for CLI/tests).
    """

    def __init__(self, cache_dir=None):
        from .model_catalog import default_catalog_cache_dir

        configured = os.environ.get("HETEROLLM_SIM_HARDWARE_PRESET_CACHE") or os.environ.get("HETEROLLM_SIM_HARDWARE_PRESET_CACHE_DIR")
        directory = Path(cache_dir) if cache_dir is not None else (
            Path(configured).expanduser() if configured else default_catalog_cache_dir().parent / "component-presets"
        )
        self.path = directory / "presets.json"
        self._lock = threading.RLock()
        self.overrides: Dict[str, ComponentPresetDefinition] = {}
        self.removed = set()
        self._load_persisted()

    def _load_persisted(self):
        overrides: Dict[str, ComponentPresetDefinition] = {}
        removed = set()
        try:
            if self.path.exists():
                raw = json.loads(
                    self.path.read_text(encoding="utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                    parse_constant=_reject_nonfinite_json_constant,
                )
                if not isinstance(raw, dict) or set(raw) != {"version", "presets", "removed"}:
                    raise ValueError("硬件预设目录顶层结构无效：{}".format(self.path))
                if type(raw.get("version")) is not int or raw["version"] != 1:
                    raise ValueError("硬件预设目录版本无效：{}".format(self.path))
                parsed = [
                    _definition_from_persisted_record(value)
                    for value in _mutation_list(raw.get("presets", []), "presets")
                ]
                ids = [item.preset_id for item in parsed]
                if len(ids) != len(set(ids)):
                    raise ValueError("硬件预设目录包含重复的预设 ID：{}".format(self.path))
                overrides = {item.preset_id: item for item in parsed}
                removed_values = _mutation_list(raw.get("removed", []), "removed")
                if any(
                    not isinstance(item, str) or not _PRESET_ID_RE.fullmatch(item)
                    for item in removed_values
                ):
                    raise ValueError("硬件预设目录包含无效的删除标记：{}".format(self.path))
                if len(removed_values) != len(set(removed_values)):
                    raise ValueError("硬件预设目录包含重复的删除标记：{}".format(self.path))
                removed = set(removed_values)
                if set(overrides) & removed:
                    raise ValueError("硬件预设目录同时保留并删除同一预设：{}".format(self.path))
        except ComponentPresetPersistenceError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ComponentPresetPersistenceError("组件预设目录无法读取：{}".format(self.path)) from exc
        self.overrides, self.removed = overrides, removed

    def _save(self, overrides, removed):
        records = [{
            "id": item.preset_id, "name": item.name, "family": item.family,
            "component": to_primitive(item.component),
            "sources": [source.to_metadata() for source in item.sources],
            "evidence_level": item.evidence_level, "limitations": list(item.limitations),
            "notes": item.notes, "tags": list(item.tags),
        } for item in sorted(overrides.values(), key=lambda item: item.preset_id)]
        temp_path = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=".presets-", suffix=".json", delete=False) as handle:
                temp_path = Path(handle.name)
                json.dump({"version": 1, "presets": records, "removed": sorted(removed)}, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except OSError as exc:
            raise ComponentPresetPersistenceError("组件预设目录无法写入：{}".format(self.path)) from exc
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError as exc:
                    if sys.exc_info()[0] is None:
                        raise ComponentPresetPersistenceError("组件预设临时文件无法清理：{}".format(temp_path)) from exc
        # Publish only after the atomic write succeeds: failed saves do not
        # expose state that disappears after restarting the server.
        self.overrides, self.removed = overrides, removed

    def _get(self, preset_id):
        if preset_id in self.removed:
            raise KeyError(preset_id)
        return self.overrides[preset_id] if preset_id in self.overrides else get_component_preset(preset_id)

    def page(self, **filters):
        with self._lock:
            self._load_persisted()
            definitions = [item for item in _PRESETS if item.preset_id not in self.removed and item.preset_id not in self.overrides]
            definitions.extend(self.overrides.values())
            return component_preset_page(items=[_metadata(item) for item in sorted(definitions, key=lambda item: item.preset_id)], **filters)

    def detail(self, preset_id):
        with self._lock:
            self._load_persisted()
            return _component_definition_detail(self._get(preset_id))

    def create(self, payload):
        with self._lock:
            with _catalog_file_lock(self.path):
                self._load_persisted()
                definition = _definition_from_mutation(payload, strict=True)
                if definition.preset_id in _BY_ID or definition.preset_id in self.overrides:
                    raise ComponentPresetMutationError("already_exists", "组件预设 ID 已存在：{}".format(definition.preset_id), status=409)
                self._save({**self.overrides, definition.preset_id: definition}, self.removed - {definition.preset_id})
                return _component_definition_detail(definition)

    def update(self, preset_id, payload):
        with self._lock:
            with _catalog_file_lock(self.path):
                self._load_persisted()
                current = self._get(preset_id)
                if not isinstance(current, ComponentPresetDefinition):
                    raise ComponentPresetMutationError("unsupported_preset", "组合拓扑预设暂不支持编辑")
                definition = _definition_from_mutation(payload, existing=current, strict=True)
                if definition.preset_id != preset_id:
                    raise ComponentPresetMutationError("id_change_not_allowed", "编辑时不允许修改预设 ID")
                self._save({**self.overrides, preset_id: definition}, self.removed - {preset_id})
                return _component_definition_detail(definition)

    def delete(self, preset_id):
        with self._lock:
            with _catalog_file_lock(self.path):
                self._load_persisted()
                if not isinstance(self._get(preset_id), ComponentPresetDefinition):
                    raise ComponentPresetMutationError("unsupported_preset", "组合拓扑预设暂不支持删除")
                self._save({key: value for key, value in self.overrides.items() if key != preset_id}, self.removed | {preset_id})


__all__ = [
    "A_ANALYTICAL",
    "CATALOG_CUTOFF_AT",
    "CATALOG_VERSION",
    "CAPABILITY_UNITS",
    "COMPONENT_KINDS",
    "EVIDENCE_LEVELS",
    "S1_STANDARD",
    "S2_VENDOR_DECLARED",
    "S3_VENDOR_PREPRODUCTION",
    "S4_PRIMARY_RESEARCH",
    "BUNDLE_USAGE_HINT",
    "USAGE_HINT",
    "ComponentPresetDefinition",
    "ComponentPresetPersistenceError",
    "ComponentSource",
    "TopologyBundleDefinition",
    "component_preset_detail",
    "component_preset_filters",
    "component_preset_page",
    "ComponentPresetMutationError",
    "ComponentPresetCatalog",
    "get_component_preset",
    "list_component_presets",
    "materialize_component_payload",
]
