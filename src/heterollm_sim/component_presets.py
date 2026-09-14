"""Offline component preset catalog for HardwareIR components.

The catalog returns one standalone :class:`ComponentSpec` template per preset.
It intentionally does not include links or placement changes: callers add the
component to the current topology and connect it explicitly when appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .ir import ComponentSpec, LinkSpec, PortSpec
from .serde import to_primitive


CATALOG_VERSION = "1.2.0"
CATALOG_CUTOFF_AT = "2026-08-27T00:00:00Z"

CAPABILITY_UNITS: Mapping[str, str] = {
    "capacity_bytes": "B",
    "peak_ops_per_s": "op/s",
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
    "cpu": "legacy-cpu",
    "host_memory": "legacy-host-memory",
    "cim": "legacy-cim",
}


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


def _hbm_ports(
    *,
    generation: str,
    channels: int,
    bandwidth_gbps: float,
) -> Tuple[PortSpec, ...]:
    return (
        PortSpec(
            port_id="host",
            protocol="HBM",
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
) -> ComponentPresetDefinition:
    technology = {
        "generation": generation,
        "pin_speed_gbps": pin_speed_gbps,
        "interface_bits": io_bits,
        "channels": channels,
        "capacity_gb": capacity_gb,
        "kind_policy": "HBM 代际只写入 metadata；ComponentSpec.kind 固定保持为 hbm。",
    }
    component = ComponentSpec(
        component_id=preset_id.replace("-", "_"),
        kind="hbm",
        cost_profile_id=_default_cost_profile_id("hbm"),
        ports=_hbm_ports(
            generation=generation,
            channels=channels,
            bandwidth_gbps=bandwidth_gbps,
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
                "value_scope": "单个 HBM 堆叠的峰值读写带宽模板",
                "conditions": [
                    "十进制带宽单位",
                    "未扣除 ECC、控制器开销、封装损耗或热降频",
                ],
                "derived_formula": "公开速度档乘以接口宽度，结果写入 IR 带宽字段",
                "expires_at": "2027-08-22",
            },
        ),
    )
    return ComponentPresetDefinition(
        preset_id=preset_id,
        name=name,
        family="HBM",
        component=component,
        sources=tuple(sources),
        evidence_level=evidence_level,
        limitations=tuple(limitations),
        notes=notes,
        tags=("memory", generation.lower()),
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
    component = ComponentSpec(
        component_id=preset_id.replace("-", "_"),
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
    component = ComponentSpec(
        component_id=preset_id.replace("-", "_"),
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
            measurement_basis="采用公开顺序读写峰值，按十进制 GB/s 换算到 IR 带宽字段；随机 IOPS 不直接折算。",
            extras={
                **analytical_transport,
                "value_scope": "单个 NVMe SSD 组件模板",
                "conditions": list(conditions or default_conditions),
                "derived_formula": "公开 GB/s 峰值乘以 8，写入 IR 带宽字段",
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
    component = ComponentSpec(
        component_id=preset_id.replace("-", "_"),
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
        component_id=preset_id.replace("-", "_"),
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


_PRESETS: Tuple[ComponentPresetDefinition, ...] = (
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
                    protocol="UCIe",
                    role="endpoint",
                    version="2.0",
                    lanes=64,
                    bandwidth_gbps=2048.0,
                    payload="xPU-HBF",
                    metadata=_port_metadata(
                        {
                            "performance_grade": "grade_3",
                            "bandwidth_basis": "analytical_UCIe_x64_32_GTps_envelope",
                            "media_bandwidth_modeled_separately": True,
                        }
                    ),
                ),
            ),
            capacity_bytes=_gb(512.0),
            read_bandwidth_gbps=24000.0,
            write_bandwidth_gbps=0.0,
            metadata=_component_metadata(
                "ocp-hbf-2026-512gb",
                technology={
                    "generation": "HBF 2026 OCP",
                    "media": "NAND flash",
                    "interface": "UCIe / xPU-HBF",
                    "capacity_gb": 512,
                    "read_bandwidth_gbps": 24000.0,
                    "write_bandwidth_gbps": None,
                    "non_volatile": True,
                },
                sources=(SK_HYNIX_HBF_SOURCE, SANDISK_HBF_SOURCE),
                evidence_level=S3_VENDOR_PREPRODUCTION,
                limitations=(
                    "2026 OCP HBF 仍是新兴开放规范/生态模板；供应链、控制器和软件栈可用性需要单独确认。",
                    "公开信息主要强调读带宽和容量，写带宽未建模，IR 写带宽字段保持 0.0。",
                    "UCIe 端口采用每方向 256 GB/s 分析 envelope；3 TB/s 是媒体读峰值，不是端口线速。",
                    "适合 AI inference 权重/冷数据近封装读取分析，不应当替代低延迟 HBM 建模。",
                ),
                notes="High Bandwidth Flash 位于 HBM 与 SSD 层之间；拓扑中需要显式添加 UCIe 链路。",
                measurement_basis="采用公开 3 TB/s 读带宽等级，按十进制单位换算到 IR 带宽字段。",
                extras={
                    "read_latency_ns": 2500.0,
                    "write_latency_ns": 0.0,
                    "transfer_granularity_bytes": 4096,
                    "max_outstanding_requests": 32,
                    "dma_bandwidth_gbps": 2048.0,
                    "dma_latency_ns": 800.0,
                    "dma_energy_pj_per_byte": 0.0,
                    "unknown_value_sentinels": {
                        "write_bandwidth_gbps": "0.0 means unknown/not declared, not physical zero",
                        "write_latency_ns": "0.0 means unknown/not declared, not zero latency",
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
                        "预生产/工作流级公开信息",
                        "只建模读峰值，不建模写入性能",
                        "需要显式 UCIe 链路，且链路与媒体带宽分别受限",
                    ],
                    "derived_formula": "公开 TB/s 峰值乘以 8000，写入 IR 带宽字段",
                    "expires_at": "2027-08-22",
                },
            ),
        ),
        sources=(SK_HYNIX_HBF_SOURCE, SANDISK_HBF_SOURCE),
        evidence_level=S3_VENDOR_PREPRODUCTION,
        limitations=(
            "2026 OCP HBF 仍是新兴开放规范/生态模板；供应链、控制器和软件栈可用性需要单独确认。",
            "公开信息主要强调读带宽和容量，写带宽未建模，IR 写带宽字段保持 0.0。",
            "UCIe 端口采用每方向 256 GB/s 分析 envelope；3 TB/s 是媒体读峰值，不是端口线速。",
            "适合 AI inference 权重/冷数据近封装读取分析，不应当替代低延迟 HBM 建模。",
        ),
        notes="High Bandwidth Flash 位于 HBM 与 SSD 层之间；拓扑中需要显式添加 UCIe 链路。",
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


_COMPONENT_BY_ID: Mapping[str, ComponentPresetDefinition] = {
    item.preset_id: item for item in _PRESETS
}
if len(_COMPONENT_BY_ID) != len(_PRESETS):  # import-time invariant, never user input
    raise RuntimeError("组件预设 ID 重复")
if any(item.component_kind not in COMPONENT_KINDS for item in _PRESETS):
    raise RuntimeError("组件预设使用了不支持的 ComponentSpec.kind")


def _hopper_sxm_bundle(
    preset_id: str,
    name: str,
    *,
    gpu_preset_id: str,
    hbm_generation: str,
    hbm_stack_count: int,
    visible_memory_gb: float,
    aggregate_bandwidth_gbps: float,
    source: ComponentSource,
    raw_stack_capacity_gb: float,
    unit_count_status: str,
    unit_count_formula: str,
    component_preset_id: str,
) -> TopologyBundleDefinition:
    """Build the stable-ID template later remapped atomically by the Web UI."""

    gpu_template = _COMPONENT_BY_ID[gpu_preset_id].component
    gpu = replace(
        gpu_template,
        component_id="gpu0",
        package_id="package0",
        die_id="gpu0_die",
        metadata={
            **dict(gpu_template.metadata),
            "component_scope": "topology_bundle_root",
            "topology_bundle_id": preset_id,
            "external_memory_modeled_in_metadata": False,
        },
    )
    per_stack_capacity_gb = visible_memory_gb / hbm_stack_count
    per_stack_bandwidth_gbps = aggregate_bandwidth_gbps / hbm_stack_count
    memories = tuple(
        ComponentSpec(
            component_id="hbm{}".format(index),
            kind="hbm",
            cost_profile_id=_default_cost_profile_id("hbm"),
            ports=(
                PortSpec(
                    port_id="host",
                    protocol="HBM",
                    role="device",
                    version=hbm_generation,
                    lanes=16,
                    bandwidth_gbps=per_stack_bandwidth_gbps,
                    metadata=_port_metadata({
                        "channels": 16,
                        "physical_channel_total": 16,
                        "channels_per_stack": 16,
                        "bandwidth_scope": "one_stack_equal_share_of_vendor_gpu_total",
                    }),
                ),
            ),
            package_id="package0",
            die_id="hbm{}_die".format(index),
            capacity_bytes=_gb(per_stack_capacity_gb),
            read_bandwidth_gbps=per_stack_bandwidth_gbps,
            write_bandwidth_gbps=per_stack_bandwidth_gbps,
            metadata={
                "capability_units": dict(CAPABILITY_UNITS),
                "topology_bundle_id": preset_id,
                "physical_composition": {
                    "simulator_representation": "single_physical_unit_node",
                    "simulator_node_count": 1,
                    "physical_unit_kind": "HBM_stack",
                    "physical_unit_count": 1,
                    "physical_unit_count_status": "explicit_physical_node",
                    "known_multiple": hbm_stack_count > 1,
                    "controller_component_id": "gpu0",
                    "memory_subsystem_id": "gpu0",
                    "unit_index": index,
                    "unit_count_in_product": hbm_stack_count,
                    "unit_count_status": unit_count_status,
                    "unit_count_formula": unit_count_formula,
                    "unit_capacity_bytes": _gb(per_stack_capacity_gb),
                    "product_total_capacity_bytes": _gb(visible_memory_gb),
                    "unit_raw_capacity_bytes": _gb(raw_stack_capacity_gb),
                    "product_total_raw_capacity_bytes": _gb(
                        raw_stack_capacity_gb * hbm_stack_count
                    ),
                    "capacity_accounting": (
                        "product_visible_and_raw_capacity_recorded_separately"
                        if raw_stack_capacity_gb != per_stack_capacity_gb
                        else "product_visible_capacity_equals_modeled_raw_capacity"
                    ),
                    "unit_bandwidth_gbps": per_stack_bandwidth_gbps,
                    "product_total_bandwidth_gbps": aggregate_bandwidth_gbps,
                    "component_preset_id": component_preset_id,
                    "component_preset_status": "catalog_reference",
                    "source_basis": "vendor product aggregate and documented/derived stack composition",
                },
                "parameter_basis": {
                    "capacity_bytes": "product visible capacity / unit_count_in_product",
                    "read_bandwidth_gbps": "product total bandwidth / unit_count_in_product",
                    "write_bandwidth_gbps": "symmetric analytical envelope from the same per-stack interface",
                    "unit_count": unit_count_status,
                    "unit_count_formula": unit_count_formula,
                    "source_basis": "vendor product aggregate and documented/derived stack composition",
                },
                "provenance": {
                    "value_status": "derived_per_physical_stack",
                    "unit_count_status": unit_count_status,
                    "source_basis": "vendor product aggregate and documented/derived stack composition",
                },
                "technology": {
                    "generation": hbm_generation,
                    "stack_index": index,
                    "visible_capacity_gb": per_stack_capacity_gb,
                    "capacity_scope": "equal_share_of_vendor_visible_memory",
                },
                "sources": list(_source_metadata((source,))),
                "evidence_level": S2_VENDOR_DECLARED,
                "measurement_basis": "厂商公开的整卡可见容量和总显存带宽按组合预设中的堆叠数等分。",
                "derived_formula": "总可见容量与总显存带宽分别除以堆叠数；带宽以 bit/s 写入 IR。",
                "conditions": ["系统级等分", "十进制容量和带宽单位"],
                "applicability_limitations": [
                    "等分不代表厂商公开了每个物理堆叠的独立容量或带宽。",
                    "未扣除 ECC、控制器、封装、刷新或热限制。",
                ],
            },
        )
        for index in range(hbm_stack_count)
    )
    links = tuple(
        LinkSpec(
            link_id="hbm_link{}".format(index),
            source_component="gpu0",
            source_port="hbm{}".format(index),
            target_component="hbm{}".format(index),
            target_port="host",
            protocol="HBM",
            version=hbm_generation,
            lanes=16,
            bandwidth_gbps=per_stack_bandwidth_gbps,
            latency_ns=40.0,
            bidirectional=True,
            metadata=_link_metadata({
                "topology_bundle_id": preset_id,
                "bandwidth_semantics": "one_stack_one_way_capacity",
                "physical_stack_count_asserted": unit_count_status.startswith(
                    "vendor_documented"
                ),
                "unit_count_status": unit_count_status,
                "unit_count_formula": unit_count_formula,
                "displayed_bandwidth_scope": "单个堆叠的单向等分带宽；读写不相加。",
            }),
        )
        for index in range(hbm_stack_count)
    )
    limitations = (
        "组合预设只表示一个 SXM GPU 与其显式 HBM 堆叠；不自动添加 NVLink 交换机、CPU、PCIe 根复合体或主机内存。",
        "HBM 堆叠的容量和带宽由厂商公开整卡值等分，适合系统级拓扑，不是封装反向工程。",
        "自动布局与视觉分组不改变仿真链路语义，分组默认展开并以 GPU 为根。",
    )
    return TopologyBundleDefinition(
        preset_id=preset_id,
        name=name,
        family="NVIDIA Hopper SXM Bundle",
        components=(gpu,) + memories,
        links=links,
        group={
            "group_id": "sxm_group0",
            "label": name,
            "members": tuple(component.component_id for component in (gpu,) + memories),
            "root": "gpu0",
            "collapsed": False,
        },
        sources=(source,),
        evidence_level=S2_VENDOR_DECLARED,
        limitations=limitations,
        notes="一次载入 GPU 根、全部 HBM 组件和专用内部链路；由前端统一重新编号、布局和提交撤销历史。",
        tags=("gpu", "hopper", "sxm", "bundle", hbm_generation.lower()),
    )


_BUNDLES: Tuple[TopologyBundleDefinition, ...] = (
    _hopper_sxm_bundle(
        "nvidia-h100-sxm-bundle",
        "NVIDIA H100 SXM + 5×HBM3",
        gpu_preset_id="nvidia-h100-sxm-gpu",
        hbm_generation="HBM3",
        hbm_stack_count=5,
        visible_memory_gb=80.0,
        aggregate_bandwidth_gbps=26800.0,
        source=NVIDIA_H100_SOURCE,
        raw_stack_capacity_gb=16.0,
        unit_count_status="vendor_documented_active_stacks",
        unit_count_formula="vendor-documented 5 active HBM3 stacks × 16 GB = 80 GB",
        component_preset_id="hbm3-16gb-0_670tbs-h100-slice",
    ),
    _hopper_sxm_bundle(
        "nvidia-h200-sxm-bundle",
        "NVIDIA H200 SXM + 6×HBM3E",
        gpu_preset_id="nvidia-h200-sxm-gpu",
        hbm_generation="HBM3E",
        hbm_stack_count=6,
        visible_memory_gb=141.0,
        aggregate_bandwidth_gbps=38400.0,
        source=NVIDIA_H200_SOURCE,
        raw_stack_capacity_gb=24.0,
        unit_count_status="derived_from_product_total_and_24GB_stack_class",
        unit_count_formula="144 GB raw capacity / 24 GB HBM3E stack class = 6; 141 GB is product-visible capacity",
        component_preset_id="hbm3e-24gb-0_800tbs-h200-slice",
    ),
)


_BUNDLE_BY_ID: Mapping[str, TopologyBundleDefinition] = {
    item.preset_id: item for item in _BUNDLES
}
_BY_ID: Mapping[str, Any] = {**_COMPONENT_BY_ID, **_BUNDLE_BY_ID}
if len(_BY_ID) != len(_PRESETS) + len(_BUNDLES):
    raise RuntimeError("组件或组合拓扑预设 ID 重复")


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
        for item in sorted(_PRESETS + _BUNDLES, key=lambda item: item.preset_id)
    )


def component_preset_filters() -> Dict[str, Any]:
    """Return available filter values for the list API envelope."""

    items = list_component_presets()
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
) -> Dict[str, Any]:
    """Return a small filtered list envelope for UI integration."""

    normalized_query = query.strip().lower()
    normalized_family = family.strip().lower()
    normalized_kind = component_kind.strip().lower()
    normalized_evidence = evidence_level.strip().lower()
    normalized_tag = tag.strip().lower()
    items = []
    for item in list_component_presets():
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
        "filters": component_preset_filters(),
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
    "ComponentSource",
    "TopologyBundleDefinition",
    "component_preset_detail",
    "component_preset_filters",
    "component_preset_page",
    "get_component_preset",
    "list_component_presets",
    "materialize_component_payload",
]
