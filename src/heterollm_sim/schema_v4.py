"""V4 authoring/runtime profiles and the explicit offline V3 importer.

The normal scenario parser is intentionally *not* a compatibility layer.  It
accepts only :data:`AUTHORING_SCHEMA_VERSION`; callers that still own a V3
document must run the explicit offline ``heterollm-import-v3`` command first.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field, fields
import math
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar


AUTHORING_SCHEMA_VERSION = "4.0.0"
SIMULATION_SCHEMA_VERSION = "4.0.0"
RUNTIME_PROFILE_SCHEMA_VERSION = AUTHORING_SCHEMA_VERSION
CONTROLLER_PROFILE_SCHEMA_VERSION = AUTHORING_SCHEMA_VERSION
LEGACY_AUTHORING_SCHEMA_VERSION = "3.0.0"
LEGACY_MAPPING_FINGERPRINT_SCHEMA = "automatic-mapping-deployment-v3"
CONTROL_PLANE_FINGERPRINT_SCHEMA = "runtime-control-plane-v4"


def _positive_int(value: int, field_name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(field_name, minimum))


def _positive_number(
    value: float, field_name: str, *, allow_zero: bool = False
) -> None:
    minimum = 0.0 if allow_zero else 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < minimum
        or (not allow_zero and value == 0)
    ):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError("{} must be a finite {} number".format(field_name, comparator))


@dataclass(frozen=True)
class CPUControlPlaneProfile:
    """CPU cache/DRAM resources used by scheduling and submission work."""

    cache_capacity_bytes: int = 32 * 1024**2
    cache_line_bytes: int = 64
    cache_hit_latency_ns: float = 12.0
    dram_read_bandwidth_gb_s: float = 100.0
    dram_write_bandwidth_gb_s: float = 80.0
    dram_latency_ns: float = 90.0
    memory_channels: int = 8
    request_queue_depth: int = 256
    max_outstanding_requests: int = 128
    request_batch_size: int = 32

    def __post_init__(self) -> None:
        for name in (
            "cache_capacity_bytes",
            "cache_line_bytes",
            "memory_channels",
            "request_queue_depth",
            "max_outstanding_requests",
            "request_batch_size",
        ):
            _positive_int(getattr(self, name), "cpu.{}".format(name))
        for name in (
            "cache_hit_latency_ns",
            "dram_read_bandwidth_gb_s",
            "dram_write_bandwidth_gb_s",
            "dram_latency_ns",
        ):
            _positive_number(getattr(self, name), "cpu.{}".format(name))


@dataclass(frozen=True)
class NVMePageCacheProfile:
    """NVMe queues and the host page cache used for weight/state offload."""

    page_cache_capacity_bytes: int = 8 * 1024**3
    page_size_bytes: int = 4096
    nvme_read_bandwidth_gb_s: float = 7.0
    nvme_write_bandwidth_gb_s: float = 5.0
    nvme_read_latency_ns: float = 80_000.0
    nvme_write_latency_ns: float = 120_000.0
    submission_queue_depth: int = 128
    completion_queue_depth: int = 128
    max_outstanding_io: int = 64
    io_batch_size: int = 16

    def __post_init__(self) -> None:
        for name in (
            "page_cache_capacity_bytes",
            "page_size_bytes",
            "submission_queue_depth",
            "completion_queue_depth",
            "max_outstanding_io",
            "io_batch_size",
        ):
            _positive_int(getattr(self, name), "nvme_page_cache.{}".format(name))
        for name in (
            "nvme_read_bandwidth_gb_s",
            "nvme_write_bandwidth_gb_s",
            "nvme_read_latency_ns",
            "nvme_write_latency_ns",
        ):
            _positive_number(getattr(self, name), "nvme_page_cache.{}".format(name))


@dataclass(frozen=True)
class PCIeDMAIOMMUProfile:
    """Host/device transport lanes, DMA queues, and IOMMU translation."""

    pcie_lane_count: int = 16
    pcie_bandwidth_gb_s_per_lane: float = 3.938
    dma_engine_count: int = 2
    dma_queue_depth: int = 128
    dma_max_outstanding: int = 64
    dma_batch_bytes: int = 1024 * 1024
    iommu_tlb_entries: int = 2048
    iommu_page_size_bytes: int = 4096
    iommu_miss_latency_ns: float = 250.0
    iommu_max_outstanding_walks: int = 32

    def __post_init__(self) -> None:
        for name in (
            "pcie_lane_count",
            "dma_engine_count",
            "dma_queue_depth",
            "dma_max_outstanding",
            "dma_batch_bytes",
            "iommu_tlb_entries",
            "iommu_page_size_bytes",
            "iommu_max_outstanding_walks",
        ):
            _positive_int(getattr(self, name), "pcie_dma_iommu.{}".format(name))
        _positive_number(
            self.pcie_bandwidth_gb_s_per_lane,
            "pcie_dma_iommu.pcie_bandwidth_gb_s_per_lane",
        )
        _positive_number(
            self.iommu_miss_latency_ns,
            "pcie_dma_iommu.iommu_miss_latency_ns",
        )

    @property
    def aggregate_pcie_bandwidth_gb_s(self) -> float:
        return self.pcie_lane_count * self.pcie_bandwidth_gb_s_per_lane


@dataclass(frozen=True)
class GPUCommandProcessorProfile:
    command_processor_count: int = 1
    hardware_queue_count: int = 32
    queue_depth: int = 1024
    max_outstanding_kernels: int = 128
    launch_batch_size: int = 16
    command_submission_latency_ns: float = 5_000.0

    def __post_init__(self) -> None:
        for name in (
            "command_processor_count",
            "hardware_queue_count",
            "queue_depth",
            "max_outstanding_kernels",
            "launch_batch_size",
        ):
            _positive_int(getattr(self, name), "gpu.command_processor.{}".format(name))
        _positive_number(
            self.command_submission_latency_ns,
            "gpu.command_processor.command_submission_latency_ns",
        )


@dataclass(frozen=True)
class GPUMMUTLBProfile:
    tlb_entries: int = 8192
    page_size_bytes: int = 64 * 1024
    page_walk_latency_ns: float = 300.0
    max_outstanding_page_walks: int = 64
    translation_batch_size: int = 32

    def __post_init__(self) -> None:
        for name in (
            "tlb_entries",
            "page_size_bytes",
            "max_outstanding_page_walks",
            "translation_batch_size",
        ):
            _positive_int(getattr(self, name), "gpu.mmu_tlb.{}".format(name))
        _positive_number(
            self.page_walk_latency_ns, "gpu.mmu_tlb.page_walk_latency_ns"
        )


@dataclass(frozen=True)
class GPUL2CacheProfile:
    capacity_bytes: int = 50 * 1024**2
    line_size_bytes: int = 128
    hit_latency_ns: float = 120.0
    miss_queue_depth: int = 256
    max_outstanding_misses: int = 128
    request_batch_size: int = 32

    def __post_init__(self) -> None:
        for name in (
            "capacity_bytes",
            "line_size_bytes",
            "miss_queue_depth",
            "max_outstanding_misses",
            "request_batch_size",
        ):
            _positive_int(getattr(self, name), "gpu.l2_cache.{}".format(name))
        _positive_number(self.hit_latency_ns, "gpu.l2_cache.hit_latency_ns")


@dataclass(frozen=True)
class GPUVRAMControllerProfile:
    controller_count: int = 8
    channel_count: int = 8
    lanes_per_channel: int = 4
    queue_depth: int = 256
    max_outstanding_requests: int = 128
    request_batch_size: int = 32
    read_bandwidth_gb_s: float = 3_000.0
    write_bandwidth_gb_s: float = 3_000.0
    access_latency_ns: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "controller_count",
            "channel_count",
            "lanes_per_channel",
            "queue_depth",
            "max_outstanding_requests",
            "request_batch_size",
        ):
            _positive_int(getattr(self, name), "gpu.vram_controller.{}".format(name))
        for name in (
            "read_bandwidth_gb_s",
            "write_bandwidth_gb_s",
            "access_latency_ns",
        ):
            _positive_number(getattr(self, name), "gpu.vram_controller.{}".format(name))


@dataclass(frozen=True)
class GPUControllerProfile:
    command_processor: GPUCommandProcessorProfile = field(
        default_factory=GPUCommandProcessorProfile
    )
    mmu_tlb: GPUMMUTLBProfile = field(default_factory=GPUMMUTLBProfile)
    l2_cache: GPUL2CacheProfile = field(default_factory=GPUL2CacheProfile)
    vram_controller: GPUVRAMControllerProfile = field(
        default_factory=GPUVRAMControllerProfile
    )

    def __post_init__(self) -> None:
        expected = (
            ("command_processor", GPUCommandProcessorProfile),
            ("mmu_tlb", GPUMMUTLBProfile),
            ("l2_cache", GPUL2CacheProfile),
            ("vram_controller", GPUVRAMControllerProfile),
        )
        for name, profile_type in expected:
            if not isinstance(getattr(self, name), profile_type):
                raise ValueError("gpu.{} must be a {}".format(name, profile_type.__name__))


@dataclass(frozen=True)
class ControllerProfile:
    """Aggregate V4 runtime controller profile for one architecture."""

    cpu: CPUControlPlaneProfile = field(default_factory=CPUControlPlaneProfile)
    nvme_page_cache: NVMePageCacheProfile = field(
        default_factory=NVMePageCacheProfile
    )
    pcie_dma_iommu: PCIeDMAIOMMUProfile = field(
        default_factory=PCIeDMAIOMMUProfile
    )
    gpu_controllers: Mapping[str, GPUControllerProfile] = field(
        default_factory=lambda: {"gpu0": GPUControllerProfile()}
    )
    schema_version: str = AUTHORING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORING_SCHEMA_VERSION:
            raise ValueError(
                "runtime profile schema_version must be exactly {}; got {}".format(
                    AUTHORING_SCHEMA_VERSION, self.schema_version
                )
            )
        if not isinstance(self.cpu, CPUControlPlaneProfile):
            raise ValueError("runtime profile cpu must be a CPUControlPlaneProfile")
        if not isinstance(self.nvme_page_cache, NVMePageCacheProfile):
            raise ValueError("runtime profile nvme_page_cache must be a NVMePageCacheProfile")
        if not isinstance(self.pcie_dma_iommu, PCIeDMAIOMMUProfile):
            raise ValueError("runtime profile pcie_dma_iommu must be a PCIeDMAIOMMUProfile")
        if not isinstance(self.gpu_controllers, Mapping) or not self.gpu_controllers:
            raise ValueError("runtime profile gpu_controllers must be a non-empty mapping")
        for component_id, profile in self.gpu_controllers.items():
            if not isinstance(component_id, str) or not component_id.strip():
                raise ValueError("runtime profile GPU component id must not be empty")
            if not isinstance(profile, GPUControllerProfile):
                raise ValueError(
                    "runtime profile gpu_controllers.{} must be a GPUControllerProfile".format(
                        component_id
                    )
                )

    @classmethod
    def architecture_default(cls, hardware: Any) -> "ControllerProfile":
        """Build a deterministic default keyed by declared GPU component IDs."""

        components = (
            hardware.get("components", ())
            if isinstance(hardware, Mapping)
            else getattr(hardware, "components", ())
        )
        gpu_ids = []
        for raw in components or ():
            kind = raw.get("kind", "") if isinstance(raw, Mapping) else getattr(raw, "kind", "")
            component_id = (
                raw.get("component_id", "")
                if isinstance(raw, Mapping)
                else getattr(raw, "component_id", "")
            )
            normalized = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized == "gpu" and str(component_id).strip():
                gpu_ids.append(str(component_id))
        if not gpu_ids:
            gpu_ids = ["gpu0"]
        return cls(
            gpu_controllers={
                component_id: GPUControllerProfile()
                for component_id in sorted(set(gpu_ids))
            }
        )


_T = TypeVar("_T")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(field_name))
    return value


def _dataclass_values(
    data: Mapping[str, Any], field_name: str, profile_type: Type[_T]
) -> Dict[str, Any]:
    allowed = {item.name for item in fields(profile_type)}
    unknown = sorted(str(name) for name in data if name not in allowed)
    if unknown:
        raise ValueError("{} has unknown fields: {}".format(field_name, ", ".join(unknown)))
    return dict(data)


def _leaf_profile_from_dict(
    data: Mapping[str, Any], field_name: str, profile_type: Type[_T]
) -> _T:
    return profile_type(**_dataclass_values(data, field_name, profile_type))


def gpu_controller_profile_from_dict(data: Mapping[str, Any]) -> GPUControllerProfile:
    values = _dataclass_values(data, "runtime.gpu_controllers.*", GPUControllerProfile)
    nested = (
        ("command_processor", GPUCommandProcessorProfile),
        ("mmu_tlb", GPUMMUTLBProfile),
        ("l2_cache", GPUL2CacheProfile),
        ("vram_controller", GPUVRAMControllerProfile),
    )
    for name, profile_type in nested:
        if name in values:
            values[name] = _leaf_profile_from_dict(
                _mapping(values[name], "runtime.gpu_controllers.*.{}".format(name)),
                "runtime.gpu_controllers.*.{}".format(name),
                profile_type,
            )
    return GPUControllerProfile(**values)


def runtime_profile_from_dict(data: Mapping[str, Any]) -> ControllerProfile:
    values = _dataclass_values(data, "profiles.runtime", ControllerProfile)
    if "cpu" in values:
        values["cpu"] = _leaf_profile_from_dict(
            _mapping(values["cpu"], "profiles.runtime.cpu"),
            "profiles.runtime.cpu",
            CPUControlPlaneProfile,
        )
    if "nvme_page_cache" in values:
        values["nvme_page_cache"] = _leaf_profile_from_dict(
            _mapping(values["nvme_page_cache"], "profiles.runtime.nvme_page_cache"),
            "profiles.runtime.nvme_page_cache",
            NVMePageCacheProfile,
        )
    if "pcie_dma_iommu" in values:
        values["pcie_dma_iommu"] = _leaf_profile_from_dict(
            _mapping(values["pcie_dma_iommu"], "profiles.runtime.pcie_dma_iommu"),
            "profiles.runtime.pcie_dma_iommu",
            PCIeDMAIOMMUProfile,
        )
    if "gpu_controllers" in values:
        raw_controllers = _mapping(
            values["gpu_controllers"], "profiles.runtime.gpu_controllers"
        )
        values["gpu_controllers"] = {
            str(component_id): gpu_controller_profile_from_dict(
                _mapping(
                    raw,
                    "profiles.runtime.gpu_controllers.{}".format(component_id),
                )
            )
            for component_id, raw in raw_controllers.items()
        }
    return ControllerProfile(**values)


def _upgrade_authoring_versions(value: Any) -> Any:
    """Upgrade V3 authoring nodes without rewriting opaque user metadata."""

    if isinstance(value, Mapping):
        return {
            str(name): (
                AUTHORING_SCHEMA_VERSION
                if name == "schema_version" and item == LEGACY_AUTHORING_SCHEMA_VERSION
                else deepcopy(item)
                if name in {"metadata", "attributes"}
                else _upgrade_authoring_versions(item)
            )
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_upgrade_authoring_versions(item) for item in value]
    if isinstance(value, tuple):
        return [_upgrade_authoring_versions(item) for item in value]
    return value


_POLICY_KEYS = ("locked_op_keys", "locked_tensor_ids", "options")
_DECISION_KEYS = (
    "aggregate_weight_backing",
    "aggregate_weight_backing_bytes",
    "operator_execution_targets",
    "rank_weight_shards",
    "logical_weight_views",
    "logical_weight_aliases",
    "cold_cim_streaming_tensors",
    "resident_cim_replicas",
    "weight_tensor_details",
    "route",
    "routes",
)


def _migrate_control_plane_metadata(placement: Mapping[str, Any]) -> Dict[str, Any]:
    migrated = deepcopy(dict(placement))
    metadata = dict(_mapping(migrated.get("metadata", {}), "placement.metadata"))
    legacy_explicit_placement = {
        field_name: deepcopy(
            dict(_mapping(migrated.get(field_name, {}), "placement.{}".format(field_name)))
        )
        for field_name in (
            "op_to_component",
            "tensor_to_component",
            "tensor_bytes",
        )
    }
    for field_name in legacy_explicit_placement:
        migrated[field_name] = {}
    raw_auto = metadata.pop("auto_mapping", None)
    old_marker: Optional[str] = None
    if raw_auto is not None or any(legacy_explicit_placement.values()):
        auto: Dict[str, Any] = {}
        if raw_auto is not None:
            auto = dict(_mapping(raw_auto, "placement.metadata.auto_mapping"))
        existing = metadata.get("control_plane")
        if existing is not None:
            raise ValueError(
                "V3 placement metadata cannot contain both auto_mapping and control_plane"
            )
    if raw_auto is not None:
        marker = auto.get("fingerprint_schema")
        if marker is not None:
            old_marker = str(marker)
            if old_marker != LEGACY_MAPPING_FINGERPRINT_SCHEMA:
                raise ValueError(
                    "unsupported V3 placement.metadata.auto_mapping fingerprint_schema: {}".format(
                        old_marker
                    )
                )
    if raw_auto is not None or any(legacy_explicit_placement.values()):
        legacy_policy = {
            key: deepcopy(auto[key]) for key in _POLICY_KEYS if key in auto
        }
        legacy_decision = {
            key: deepcopy(auto[key]) for key in _DECISION_KEYS if key in auto
        }
        metadata["control_plane"] = {
            "policy": {},
            "decision": {},
            "evidence": {
                "status": "migrated",
                "migrated_from": {
                    "schema_version": LEGACY_AUTHORING_SCHEMA_VERSION,
                    "artifact": "placement.metadata.auto_mapping",
                    "fingerprint_schema": old_marker,
                },
                "legacy_v3_explicit_placement": legacy_explicit_placement,
                "legacy_v3_auto_mapping": {
                    "policy": legacy_policy,
                    "decision": legacy_decision,
                },
            },
        }
    migrated["metadata"] = metadata
    return migrated


def _migrate_v3_scenario_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an explicit V4 payload without invoking the normal parser.

    The importer is deterministic and offline.  It preserves namespaced /v1
    artifacts and retains old placement, lock, option, and solver output only
    as read-only evidence.  It never turns legacy decisions into executable
    V4 authoring placement or control-plane policy.
    """

    source = _mapping(data, "scenario")
    source_version = source.get("schema_version")
    if source_version != LEGACY_AUTHORING_SCHEMA_VERSION:
        raise ValueError(
            "V3 importer requires scenario.schema_version exactly {}; got {}".format(
                LEGACY_AUTHORING_SCHEMA_VERSION, source_version
            )
        )
    migrated = _upgrade_authoring_versions(deepcopy(dict(source)))
    migrated["placement"] = _migrate_control_plane_metadata(
        _mapping(migrated.get("placement"), "placement")
    )
    profiles = dict(_mapping(migrated.get("profiles"), "profiles"))
    if "runtime" in profiles:
        raise ValueError("V3 payload unexpectedly contains profiles.runtime")
    runtime = ControllerProfile.architecture_default(
        _mapping(migrated.get("hardware"), "hardware")
    )
    from .serde import to_primitive

    profiles["runtime"] = to_primitive(runtime)
    migrated["profiles"] = profiles
    return migrated


def _import_v3_scenario_file(
    source: Path, destination: Path, *, overwrite: bool = False
) -> Path:
    """Migrate and validate one JSON file, refusing accidental overwrite."""

    from .config import scenario_from_dict
    from .serde import read_json, write_json

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different paths")
    if destination.exists() and not overwrite:
        raise FileExistsError("destination already exists: {}".format(destination))
    migrated = _migrate_v3_scenario_payload(read_json(source))
    scenario_from_dict(migrated)
    write_json(destination, migrated)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicit offline HeteroLLM V3-to-V4 scenario importer"
    )
    parser.add_argument("source", type=Path, help="V3 scenario JSON input")
    parser.add_argument("destination", type=Path, help="V4 scenario JSON output")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing destination"
    )
    args = parser.parse_args(argv)
    _import_v3_scenario_file(
        args.source,
        args.destination,
        overwrite=args.force,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the entry point
    raise SystemExit(main())


__all__ = [
    "AUTHORING_SCHEMA_VERSION",
    "CONTROLLER_PROFILE_SCHEMA_VERSION",
    "CONTROL_PLANE_FINGERPRINT_SCHEMA",
    "ControllerProfile",
    "CPUControlPlaneProfile",
    "GPUCommandProcessorProfile",
    "GPUControllerProfile",
    "GPUL2CacheProfile",
    "GPUMMUTLBProfile",
    "GPUVRAMControllerProfile",
    "LEGACY_AUTHORING_SCHEMA_VERSION",
    "LEGACY_MAPPING_FINGERPRINT_SCHEMA",
    "NVMePageCacheProfile",
    "PCIeDMAIOMMUProfile",
    "RUNTIME_PROFILE_SCHEMA_VERSION",
    "SIMULATION_SCHEMA_VERSION",
    "gpu_controller_profile_from_dict",
    "runtime_profile_from_dict",
]
