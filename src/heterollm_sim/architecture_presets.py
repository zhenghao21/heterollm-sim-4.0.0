"""Offline architecture-topology presets for :class:`HardwareSpec`.

This catalog is deliberately separate from component presets.  Each entry is
a complete, hardware-only graph that replaces the current hardware topology;
it never contains a model, workload, placement, or rank mapping.  Public
logical relationships are kept distinct from analytical collapsed fabrics so
that a useful system model is not mistaken for undisclosed physical wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec
from .serde import to_primitive
from .topology import validate_topology


CATALOG_VERSION = "1.3.0"
CATALOG_CUTOFF_AT = "2026-08-27T00:00:00Z"

COMPONENT_CAPABILITY_UNITS: Mapping[str, str] = {
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

# Architecture payloads are consumed by the V4 scenario adapter alongside the
# bundled reference profile registry.  Keep the binding in the generated
# component itself; a caller can then either retain the explicit id or replace
# it deliberately when importing a different registry.  In particular, this
# is not a "pick the only profile" fallback: an explicit id remains valid when
# a registry contains several profiles, and an absent id is rejected by the
# scenario validator.
_DEFAULT_COST_PROFILE_IDS: Mapping[str, str] = {
    "gpu": "legacy-gpu",
    "hbm": "legacy-hbm",
    "cpu": "legacy-cpu",
    "host_memory": "legacy-host-memory",
    "dram": "legacy-host-memory",
    "cim": "legacy-cim",
}


def _default_cost_profile_id(kind: str) -> Optional[str]:
    normalized = kind.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"digital_sram_cim", "compute_in_memory", "compute_in_memory_tile"}:
        normalized = "cim"
    elif normalized in {"hbm_stack"}:
        normalized = "hbm"
    return _DEFAULT_COST_PROFILE_IDS.get(normalized)

EXACT_PUBLIC_TOPOLOGY = "exact_public_topology"
ANALYTICAL_APPROXIMATION = "analytical_approximation"
EXPERIMENTAL_REFERENCE = "experimental_reference"
SUPPORT_LEVELS = frozenset(
    {EXACT_PUBLIC_TOPOLOGY, ANALYTICAL_APPROXIMATION, EXPERIMENTAL_REFERENCE}
)

REPLACEMENT_POLICY: Mapping[str, Any] = {
    "mode": "replace_hardware",
    "preserve": ["model", "workload"],
    "invalidate": [
        "placement",
        "rank_mapping",
        "hardware_cost_profiles",
        "host_orchestration",
    ],
    "confirmation_required": True,
    "undo_scope": "single_operation",
}


@dataclass(frozen=True)
class ArchitectureSource:
    title: str
    url: str
    publisher: str
    source_kind: str
    published_at: str = ""
    accessed_at: str = "2026-08-27"
    primary_source: bool = True

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "source_kind": self.source_kind,
            "published_at": self.published_at,
            "accessed_at": self.accessed_at,
            "primary_source": self.primary_source,
        }


@dataclass(frozen=True)
class ArchitecturePresetDefinition:
    preset_id: str
    name: str
    vendor: str
    family: str
    topology_class: str
    scale: str
    support_level: str
    hardware: HardwareSpec
    sources: Tuple[ArchitectureSource, ...]
    limitations: Tuple[str, ...]
    notes: str
    tags: Tuple[str, ...] = ()
    loadable: bool = True

    @property
    def protocols(self) -> Tuple[str, ...]:
        return tuple(sorted({link.protocol for link in self.hardware.links}))

    @property
    def groups(self) -> Tuple[Mapping[str, Any], ...]:
        view = self.hardware.metadata.get("topology_view", {})
        return tuple(view.get("groups", ())) if isinstance(view, Mapping) else ()


def _source(title: str, url: str, publisher: str, source_kind: str) -> ArchitectureSource:
    return ArchitectureSource(title, url, publisher, source_kind)


NVIDIA_HOPPER = _source(
    "NVIDIA DGX H100 system",
    "https://www.nvidia.com/en-us/data-center/dgx-h100/",
    "NVIDIA",
    "vendor_reference_architecture",
)
NVIDIA_H200 = _source(
    "NVIDIA H200 Tensor Core GPU",
    "https://www.nvidia.com/en-us/data-center/h200/",
    "NVIDIA",
    "vendor_product_specification",
)
NVIDIA_GH200 = _source(
    "NVIDIA Grace Hopper Superchip",
    "https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/",
    "NVIDIA",
    "vendor_product_specification",
)
NVIDIA_GB200 = _source(
    "NVIDIA GB200 NVL72",
    "https://www.nvidia.com/en-us/data-center/gb200-nvl72/",
    "NVIDIA",
    "vendor_reference_architecture",
)
AMD_MI300A = _source(
    "AMD Instinct MI300A accelerated processing unit",
    "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html",
    "AMD",
    "vendor_product_specification",
)
AMD_MI300X = _source(
    "AMD Instinct MI300X accelerator",
    "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
    "AMD",
    "vendor_product_specification",
)
INTEL_GAUDI3 = _source(
    "Intel Gaudi 3 AI accelerator",
    "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html",
    "Intel",
    "vendor_product_specification",
)
CXL_SPEC = _source(
    "Compute Express Link specifications",
    "https://computeexpresslink.org/cxl-specification/",
    "CXL Consortium",
    "consortium_specification",
)
UCIE_SPEC = _source(
    "Universal Chiplet Interconnect Express specifications",
    "https://www.uciexpress.org/specifications",
    "UCIe Consortium",
    "consortium_specification",
)
JEDEC_HBM = _source(
    "JEDEC High Bandwidth Memory standards",
    "https://www.jedec.org/standards-documents/focus/memory-ssd-ufs-emmc/hbm",
    "JEDEC",
    "industry_standard",
)
SAMSUNG_PIM = _source(
    "Samsung HBM-PIM technology",
    "https://semiconductor.samsung.com/news-events/news/samsung-develops-industrys-first-high-bandwidth-memory-with-ai-processing-power/",
    "Samsung Electronics",
    "vendor_technology_disclosure",
)
NVIDIA_GDS = _source(
    "NVIDIA GPUDirect Storage overview guide",
    "https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html",
    "NVIDIA",
    "vendor_software_architecture",
)
OCP_HBF = _source(
    "Open Compute Project High Bandwidth Flash",
    "https://www.opencompute.org/projects/high-bandwidth-flash",
    "Open Compute Project",
    "consortium_reference",
)
SAMSUNG_PM1743 = _source(
    "Samsung PM1743 enterprise NVMe SSD",
    "https://semiconductor.samsung.com/ssd/enterprise-ssd/pm1743/",
    "Samsung Semiconductor",
    "vendor_product_specification",
)
SAMSUNG_PM1743_WHITE_PAPER = _source(
    "Samsung PM1743 white paper (2024-05-10)",
    "https://download.semiconductor.samsung.com/resources/white-paper/PM1743_White_Paper_240510.pdf",
    "Samsung Semiconductor",
    "vendor_white_paper",
)


def _gb(value: float) -> int:
    return int(value * 1_000_000_000)


def _capacity_display(capacity_bytes: int, *, aggregate: bool = False) -> str:
    if capacity_bytes <= 0:
        return ""
    suffix = "IEC 聚合显示口径" if aggregate else "IEC 显示口径"
    if capacity_bytes >= 1024 ** 4:
        return "{:.3f} TiB（{}）".format(capacity_bytes / (1024 ** 4), suffix)
    if capacity_bytes >= 1024 ** 3:
        return "{:.3f} GiB（{}）".format(capacity_bytes / (1024 ** 3), suffix)
    if capacity_bytes >= 1024 ** 2:
        return "{:.3f} MiB（{}）".format(capacity_bytes / (1024 ** 2), suffix)
    return "{:.3f} KiB（{}）".format(capacity_bytes / 1024, suffix)


def _bandwidth_metadata(
    protocol: str,
    version: str,
    lanes: int,
    payload: Optional[str],
) -> Dict[str, Any]:
    """Return an additive, machine-readable bandwidth accounting contract.

    ``bandwidth_gbps`` remains the established IR field and always stores a
    decimal gigabit-per-second value for one direction.  The extra fields make
    it explicit whether that number is a raw PHY envelope, a decoded line-rate
    envelope, a vendor aggregate, or an analytical operating point.
    """

    metadata: Dict[str, Any] = {
        "capability_units": dict(PORT_CAPABILITY_UNITS),
        "bandwidth_semantics": "one_way_capacity",
        "bandwidth_value_unit": "Gb/s_decimal",
        "direction_accounting": "one_direction_only",
        "duplex_accounting": "bidirectional_does_not_double_bandwidth_gbps",
        "lane_count": lanes,
        "lane_count_semantics": "declared_transfer_units",
    }
    normalized = protocol.lower()
    if normalized == "pcie" and version == "5.0":
        metadata.update(
            {
                "bandwidth_basis": "decoded_phy_line_rate",
                "line_rate_gtps_per_lane": 32.0,
                "encoding": "128b/130b",
                "encoding_overhead_included": True,
                "transaction_overhead_included": False,
                "bandwidth_scope": "aggregate_across_declared_lanes",
            }
        )
    elif normalized == "cxl" and version == "3.0":
        metadata.update(
            {
                "bandwidth_basis": "raw_phy_envelope",
                "line_rate_gtps_per_lane": 64.0,
                "signaling": "PCIe_6.0_PHY_PAM4",
                "flit_fec_overhead_included": False,
                "protocol_payload_overhead_included": False,
                "bandwidth_scope": "aggregate_across_declared_lanes",
            }
        )
    elif normalized == "ucie" and version == "2.0":
        metadata.update(
            {
                "bandwidth_basis": "analytical_raw_phy_operating_point",
                "line_rate_gtps_per_lane": 32.0,
                "protocol_payload_overhead_included": False,
                "bandwidth_scope": "aggregate_across_declared_lanes",
                "payload_mapping": payload or "unspecified",
            }
        )
    else:
        metadata.update(
            {
                "bandwidth_basis": "vendor_or_analytical_aggregate_envelope",
                "bandwidth_scope": "aggregate_endpoint_capacity",
            }
        )
    return metadata


def _aggregate_composition(
    physical_unit_kind: str,
    *,
    physical_unit_count: Optional[int],
    count_status: str,
    known_multiple: bool,
    source_basis: str,
) -> Dict[str, Any]:
    """Describe physical composition without multiplying simulator nodes."""

    return {
        "simulator_representation": "aggregate_node",
        "simulator_node_count": 1,
        "physical_unit_kind": physical_unit_kind,
        "physical_unit_count": physical_unit_count,
        "physical_unit_count_status": count_status,
        "known_multiple": known_multiple,
        "source_basis": source_basis,
    }


def _split_integer_total(total: int, count: int, index: int) -> int:
    """Split an integer total without losing bytes to truncation."""

    if count <= 0:
        raise ValueError("physical HBM stack count must be positive")
    quotient, remainder = divmod(total, count)
    return quotient + (1 if index < remainder else 0)


def _hbm_controller_ports(
    stack_count: int,
    generation: str,
    product_total_bandwidth_gbps: float,
    *,
    prefix: str = "hbm",
) -> Tuple[PortSpec, ...]:
    """Create one controller port for every modeled physical HBM stack."""

    unit_bandwidth = product_total_bandwidth_gbps / stack_count
    return tuple(
        _port(
            "{}{}".format(prefix, index),
            "HBM",
            "controller",
            version=generation,
            lanes=16,
            bandwidth_gbps=unit_bandwidth,
            metadata={
                "physical_unit_kind": "HBM_stack",
                "unit_index": index,
                "unit_count_in_product": stack_count,
                "physical_channel_total": 16,
                "channels_per_stack": 16,
                "bandwidth_derivation": "product_total_bandwidth_gbps / unit_count_in_product",
            },
        )
        for index in range(stack_count)
    )


def _physical_hbm_stacks(
    *,
    root_component_id: str,
    root_port_prefix: str,
    stack_id_prefix: str,
    link_id_prefix: str,
    package_id: str,
    generation: str,
    stack_count: int,
    product_total_capacity_bytes: int,
    product_total_bandwidth_gbps: float,
    role: str,
    product_model: str,
    unit_count_status: str,
    unit_count_formula: str,
    source_basis: str,
    component_preset_id: str = "",
    product_total_raw_capacity_bytes: Optional[int] = None,
    latency_ns: float = 40.0,
) -> Tuple[Tuple[ComponentSpec, ...], Tuple[LinkSpec, ...]]:
    """Expand a product memory subsystem into one node/link per HBM stack.

    Product-visible capacity and aggregate one-way bandwidth remain the source
    totals.  Per-stack values are deterministic shares whose formula and
    provenance are carried on every node, port, and link.  This is a physical
    component-count model, not a claim about undisclosed channel routing.
    """

    if stack_count <= 0:
        raise ValueError("physical HBM stack count must be positive")
    unit_bandwidth = product_total_bandwidth_gbps / stack_count
    components = []
    links = []
    for index in range(stack_count):
        component_id = "{}{}".format(stack_id_prefix, index)
        capacity_bytes = _split_integer_total(
            product_total_capacity_bytes,
            stack_count,
            index,
        )
        raw_capacity_bytes = (
            _split_integer_total(
                product_total_raw_capacity_bytes,
                stack_count,
                index,
            )
            if product_total_raw_capacity_bytes is not None
            else capacity_bytes
        )
        physical_composition = {
            "simulator_representation": "single_physical_unit_node",
            "simulator_node_count": 1,
            "physical_unit_kind": "HBM_stack",
            "physical_unit_count": 1,
            "physical_unit_count_status": "explicit_physical_node",
            "known_multiple": stack_count > 1,
            "controller_component_id": root_component_id,
            "memory_subsystem_id": root_component_id,
            "unit_index": index,
            "unit_count_in_product": stack_count,
            "unit_count_status": unit_count_status,
            "unit_count_formula": unit_count_formula,
            "unit_capacity_bytes": capacity_bytes,
            "product_total_capacity_bytes": product_total_capacity_bytes,
            "unit_raw_capacity_bytes": raw_capacity_bytes,
            "product_total_raw_capacity_bytes": (
                product_total_raw_capacity_bytes
                if product_total_raw_capacity_bytes is not None
                else product_total_capacity_bytes
            ),
            "capacity_accounting": (
                "product_visible_and_raw_capacity_recorded_separately"
                if product_total_raw_capacity_bytes is not None
                and product_total_raw_capacity_bytes != product_total_capacity_bytes
                else "product_visible_capacity_equals_modeled_raw_capacity"
            ),
            "unit_bandwidth_gbps": unit_bandwidth,
            "product_total_bandwidth_gbps": product_total_bandwidth_gbps,
            "component_preset_id": component_preset_id,
            "component_preset_status": (
                "catalog_reference"
                if component_preset_id
                else "no_exact_per_stack_component_preset"
            ),
            "source_basis": source_basis,
        }
        parameter_basis = {
            "capacity_bytes": "product_total_capacity_bytes divmod unit_count_in_product; remainder assigned by unit_index",
            "read_bandwidth_gbps": "product_total_bandwidth_gbps / unit_count_in_product",
            "write_bandwidth_gbps": "symmetric analytical envelope from the same per-stack shared interface",
            "unit_count": unit_count_status,
            "unit_count_formula": unit_count_formula,
            "source_basis": source_basis,
        }
        components.append(
            _component(
                component_id,
                "hbm",
                (
                    _port(
                        "host",
                        "HBM",
                        "device",
                        version=generation,
                        lanes=16,
                        bandwidth_gbps=unit_bandwidth,
                        metadata={
                            "physical_unit_kind": "HBM_stack",
                            "unit_index": index,
                            "unit_count_in_product": stack_count,
                            "physical_channel_total": 16,
                            "channels_per_stack": 16,
                            "bandwidth_derivation": "product_total_bandwidth_gbps / unit_count_in_product",
                        },
                    ),
                ),
                package_id=package_id,
                die_id="{}_die".format(component_id),
                capacity_bytes=capacity_bytes,
                read_bandwidth_gbps=unit_bandwidth,
                write_bandwidth_gbps=unit_bandwidth,
                role=role,
                model="{} physical {} stack {} of {}".format(
                    product_model,
                    generation,
                    index + 1,
                    stack_count,
                ),
                approximation=(
                    "单颗节点的可见容量与带宽由产品总量按物理堆叠数守恒拆分；"
                    "不表示厂商逐堆叠公布了独立可用值。"
                ),
                physical_composition=physical_composition,
                parameter_basis=parameter_basis,
                extra_metadata={
                    "technology": {
                        "generation": generation,
                        "stack_index": index,
                    },
                    "provenance": {
                        "value_status": "derived_per_physical_stack",
                        "unit_count_status": unit_count_status,
                        "source_basis": source_basis,
                    },
                },
            )
        )
        links.append(
            _link(
                "{}{}".format(link_id_prefix, index),
                root_component_id,
                "{}{}".format(root_port_prefix, index),
                component_id,
                "host",
                "HBM",
                version=generation,
                lanes=16,
                bandwidth_gbps=unit_bandwidth,
                latency_ns=latency_ns,
                metadata={
                    "bandwidth_basis": "derived_per_physical_stack_share",
                    "bandwidth_derivation": "product_total_bandwidth_gbps / unit_count_in_product",
                    "physical_unit_kind": "HBM_stack",
                    "unit_index": index,
                    "unit_count_in_product": stack_count,
                    "unit_count_status": unit_count_status,
                    "unit_count_formula": unit_count_formula,
                    "source_basis": source_basis,
                    "physical_stack_count_asserted": unit_count_status.startswith(
                        "vendor_documented"
                    ),
                },
            )
        )
    return tuple(components), tuple(links)


def _peak_ops_basis(
    precision: str,
    *,
    source_basis: str,
    sparsity: str = "dense_no_structured_sparsity_credit",
) -> Dict[str, Any]:
    return {
        "precision": precision,
        "sparsity": sparsity,
        "operation_scope": "matrix_or_tensor_operations",
        "source_basis": source_basis,
        "ir_contract": "peak_ops_per_s is not a cross-precision maximum",
    }


def _port(
    port_id: str,
    protocol: str,
    role: str,
    *,
    version: str = "1.0",
    lanes: int = 1,
    bandwidth_gbps: float = 0.0,
    max_links: int = 1,
    payload: Optional[str] = None,
    semantics: str = "one_way_capacity",
    metadata: Optional[Mapping[str, Any]] = None,
) -> PortSpec:
    port_metadata = _bandwidth_metadata(protocol, version, lanes, payload)
    port_metadata["bandwidth_semantics"] = semantics
    if metadata:
        port_metadata.update(metadata)
    return PortSpec(
        port_id=port_id,
        protocol=protocol,
        role=role,
        version=version,
        lanes=lanes,
        bandwidth_gbps=bandwidth_gbps,
        max_links=max_links,
        payload=payload,
        metadata=port_metadata,
    )


def _component(
    component_id: str,
    kind: str,
    ports: Sequence[PortSpec],
    *,
    package_id: str = "",
    die_id: str = "",
    capacity_bytes: int = 0,
    peak_ops_per_s: float = 0.0,
    read_bandwidth_gbps: float = 0.0,
    write_bandwidth_gbps: float = 0.0,
    role: str,
    model: str = "",
    approximation: str = "",
    physical_composition: Optional[Mapping[str, Any]] = None,
    parameter_basis: Optional[Mapping[str, Any]] = None,
    peak_ops_basis: Optional[Mapping[str, Any]] = None,
    topology_evidence: str = "",
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> ComponentSpec:
    component_metadata: Dict[str, Any] = {
        "capability_units": dict(COMPONENT_CAPABILITY_UNITS),
        "architecture_role": role,
        "model": model,
        "hardware_only": True,
    }
    if kind.lower() == "cpu" and peak_ops_per_s <= 0:
        component_metadata.update(
            {
                "unknown_value_sentinels": {
                    "peak_ops_per_s": "0.0 means not vendor-declared; the component's explicit profiles.components.cpu binding is authoritative"
                },
                "cpu_profile_contract": (
                    "CPU and reachable host-memory components each require explicit "
                    "profiles.components.cpu / profiles.components.host_memory bindings; "
                    "component peak_ops_per_s is only a reporting mirror"
                ),
            }
        )
    if capacity_bytes:
        component_metadata["capacity_display_value"] = _capacity_display(capacity_bytes)
        component_metadata["capacity_source_basis"] = (
            "底层字节值按十进制产品规格或分析参数录入；界面容量统一换算为 IEC 单位。"
            if capacity_bytes % 1_000_000_000 == 0
            else "底层字节值按二进制分析参数录入；界面容量统一使用 IEC 单位。"
        )
    if approximation:
        component_metadata["modeling_approximation"] = approximation
    if physical_composition:
        component_metadata["physical_composition"] = dict(physical_composition)
    if parameter_basis:
        component_metadata["parameter_basis"] = dict(parameter_basis)
    if peak_ops_per_s:
        component_metadata["peak_ops_basis"] = dict(
            peak_ops_basis
            or _peak_ops_basis(
                "analytical_precision_not_fixed",
                source_basis="analytical_parameter_requires_calibration",
            )
        )
    if topology_evidence:
        component_metadata["topology_evidence"] = topology_evidence
    if extra_metadata:
        component_metadata.update(dict(extra_metadata))
    return ComponentSpec(
        component_id=component_id,
        kind=kind,
        cost_profile_id=_default_cost_profile_id(kind),
        ports=tuple(ports),
        package_id=package_id,
        die_id=die_id,
        capacity_bytes=capacity_bytes,
        peak_ops_per_s=peak_ops_per_s,
        read_bandwidth_gbps=read_bandwidth_gbps,
        write_bandwidth_gbps=write_bandwidth_gbps,
        metadata=component_metadata,
    )


def _link(
    link_id: str,
    source_component: str,
    source_port: str,
    target_component: str,
    target_port: str,
    protocol: str,
    *,
    version: str = "1.0",
    lanes: int = 1,
    bandwidth_gbps: float = 0.0,
    latency_ns: float = 0.0,
    payload: Optional[str] = None,
    approximation: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> LinkSpec:
    link_metadata = _bandwidth_metadata(protocol, version, lanes, payload)
    link_metadata["capability_units"] = dict(LINK_CAPABILITY_UNITS)
    if approximation:
        link_metadata["modeling_approximation"] = approximation
    if metadata:
        link_metadata.update(metadata)
    return LinkSpec(
        link_id=link_id,
        source_component=source_component,
        source_port=source_port,
        target_component=target_component,
        target_port=target_port,
        protocol=protocol,
        version=version,
        lanes=lanes,
        bandwidth_gbps=bandwidth_gbps,
        latency_ns=latency_ns,
        bidirectional=True,
        payload=payload,
        metadata=link_metadata,
    )


def _group(
    group_id: str,
    label: str,
    members: Iterable[str],
    root: str,
    *,
    collapsed: bool = False,
) -> Dict[str, Any]:
    return {
        "group_id": group_id,
        "label": label,
        "members": tuple(members),
        "root": root,
        "collapsed": collapsed,
    }


def _planner_structure(
    components: Sequence[ComponentSpec],
) -> Dict[str, Any]:
    """Return the shared V4 CPU/GPU attachment contract for a hardware graph."""

    gpu_count = sum(
        component.kind.lower() == "gpu" for component in components
    )
    cpu_count = sum(
        component.kind.lower() == "cpu" for component in components
    )
    return {
        "planner_executable": gpu_count > 0 and cpu_count > 0,
        "requires_gpu_attachment": gpu_count == 0,
        "requires_cpu_attachment": cpu_count == 0,
        "requires_profile_review": True,
        "planner_gpu_count": gpu_count,
        "planner_cpu_count": cpu_count,
    }


def _hardware(
    preset_id: str,
    name: str,
    components: Sequence[ComponentSpec],
    links: Sequence[LinkSpec],
    groups: Sequence[Mapping[str, Any]],
    positions: Mapping[str, Mapping[str, float]],
    *,
    support_level: str,
    sources: Sequence[ArchitectureSource],
    limitations: Sequence[str],
) -> HardwareSpec:
    planner_structure = _planner_structure(components)
    physical_wiring_claimed = False
    if support_level == EXACT_PUBLIC_TOPOLOGY:
        evidence_scope = "public_logical_composition_and_relationships"
    elif support_level == ANALYTICAL_APPROXIMATION:
        evidence_scope = "public_endpoints_with_analytical_collapsed_topology"
    else:
        evidence_scope = "protocol_conformant_experimental_reference"
    topology_evidence = {
        "level": support_level,
        "scope": evidence_scope,
        "physical_wiring_claimed": physical_wiring_claimed,
        "source_scope": "sources support public product/protocol facts, not undisclosed physical wiring",
    }
    parameter_basis = {
        "policy": "per_field_metadata_is_authoritative",
        "capacity_storage_unit": "bytes",
        "bandwidth_storage_unit": "Gb/s_decimal_one_way",
        "peak_ops_policy": "precision_and_sparsity_are_declared_per_compute_component",
        "capability_units": {
            "component": dict(COMPONENT_CAPABILITY_UNITS),
            "port": dict(PORT_CAPABILITY_UNITS),
            "link": dict(LINK_CAPABILITY_UNITS),
        },
        "source_scope": "sources are bibliography; derived and analytical values are labeled in field metadata",
    }
    topology_view = {
        "version": 1,
        "layout": {
            "positions": {key: dict(value) for key, value in positions.items()},
            "viewport": {"x": 0.0, "y": 0.0, "scale": 1.0},
        },
        "groups": [
            {**dict(group), "members": list(group["members"])} for group in groups
        ],
    }
    return HardwareSpec(
        name=name,
        components=tuple(components),
        links=tuple(links),
        require_connected=True,
        metadata={
            "architecture_preset": {
                "id": preset_id,
                "catalog_version": CATALOG_VERSION,
                "hardware_only": True,
                "support_level": support_level,
                **planner_structure,
            },
            "topology_evidence": topology_evidence,
            "parameter_basis": parameter_basis,
            "topology_view": topology_view,
            "sources": [source.to_metadata() for source in sources],
            "applicability_limitations": list(limitations),
            "replacement_policy": dict(REPLACEMENT_POLICY),
        },
    )


def _definition(
    preset_id: str,
    name: str,
    vendor: str,
    family: str,
    topology_class: str,
    scale: str,
    support_level: str,
    components: Sequence[ComponentSpec],
    links: Sequence[LinkSpec],
    groups: Sequence[Mapping[str, Any]],
    positions: Mapping[str, Mapping[str, float]],
    sources: Sequence[ArchitectureSource],
    limitations: Sequence[str],
    notes: str,
    tags: Sequence[str],
) -> ArchitecturePresetDefinition:
    hardware = _hardware(
        preset_id,
        name,
        components,
        links,
        groups,
        positions,
        support_level=support_level,
        sources=sources,
        limitations=limitations,
    )
    return ArchitecturePresetDefinition(
        preset_id=preset_id,
        name=name,
        vendor=vendor,
        family=family,
        topology_class=topology_class,
        scale=scale,
        support_level=support_level,
        hardware=hardware,
        sources=tuple(sources),
        limitations=tuple(limitations),
        notes=notes,
        tags=tuple(tags),
    )


def _hbm_accelerator_cluster(
    *,
    preset_id: str,
    name: str,
    gpu_model: str,
    memory_kind: str,
    memory_capacity_gb: float,
    memory_bandwidth_gbps: float,
    peak_ops_per_s: float,
    count: int,
    fabric_protocol: str,
    fabric_version: str,
    fabric_bandwidth_gbps: float,
    vendor: str,
    family: str,
    sources: Sequence[ArchitectureSource],
    support_level: str,
    fabric_kind: str = "fabric_switch",
    fabric_lanes: int = 1,
    scale: str = "eight_accelerator",
    memory_physical_unit_count: Optional[int] = None,
    memory_count_status: str = "multiple_not_reliably_disclosed",
    peak_precision: str = "BF16_tensor",
    peak_source_basis: str = "vendor_product_specification",
    fabric_member_rate_gbps: Optional[float] = None,
    fabric_total_physical_port_count: Optional[int] = None,
    memory_raw_capacity_gb: Optional[float] = None,
    memory_count_formula: str = "",
    memory_source_basis: str = "",
    memory_component_preset_id: str = "",
) -> ArchitecturePresetDefinition:
    if memory_physical_unit_count is None:
        raise ValueError(
            "product architecture presets require an explicit or derived HBM stack count"
        )
    components = []
    links = []
    groups = []
    positions: Dict[str, Dict[str, float]] = {}
    switch_ports = []
    approximation = (
        "每个加速器到单个逻辑非阻塞 Fabric 端点；不表示厂商未公开的交换芯片内部布线。"
    )
    for index in range(count):
        accelerator_id = "accelerator{}".format(index)
        switch_port = "endpoint{}".format(index)
        fabric_metadata: Dict[str, Any] = {}
        if fabric_member_rate_gbps is not None:
            fabric_metadata.update(
                {
                    "bandwidth_basis": "analytical_aggregate_physical_port_envelope",
                    "lane_count_semantics": "aggregated_physical_ethernet_ports",
                    "aggregate_member_count": fabric_lanes,
                    "aggregate_member_rate_gbps": fabric_member_rate_gbps,
                    "aggregation_formula": "member_count_x_member_rate",
                    "protocol_payload_overhead_included": False,
                }
            )
        if fabric_total_physical_port_count is not None:
            fabric_metadata.update(
                {
                    "device_total_physical_port_count": fabric_total_physical_port_count,
                    "modeled_port_subset_count": fabric_lanes,
                    "modeled_port_subset_status": "analytical_endpoint_envelope",
                }
            )
        components.append(
            _component(
                accelerator_id,
                "gpu",
                _hbm_controller_ports(
                    memory_physical_unit_count,
                    memory_kind,
                    memory_bandwidth_gbps,
                )
                + (
                    _port("fabric", fabric_protocol, "endpoint", version=fabric_version, lanes=fabric_lanes, bandwidth_gbps=fabric_bandwidth_gbps, metadata=fabric_metadata),
                ),
                package_id="module{}".format(index),
                die_id="accelerator_die{}".format(index),
                peak_ops_per_s=peak_ops_per_s,
                role="accelerator",
                model=gpu_model,
                peak_ops_basis=_peak_ops_basis(
                    peak_precision,
                    source_basis=peak_source_basis,
                ),
                approximation=(
                    "Gaudi 3 使用当前仿真器可执行的 gpu 计算成本类；model 字段保留真实加速器类型。"
                    if vendor == "Intel"
                    else ""
                ),
            )
        )
        stack_ids = tuple(
            "memory{}_hbm{}".format(index, stack_index)
            for stack_index in range(memory_physical_unit_count)
        )
        stack_components, stack_links = _physical_hbm_stacks(
            root_component_id=accelerator_id,
            root_port_prefix="hbm",
            stack_id_prefix="memory{}_hbm".format(index),
            link_id_prefix="memory_link{}_hbm".format(index),
            package_id="module{}".format(index),
            generation=memory_kind,
            stack_count=memory_physical_unit_count,
            product_total_capacity_bytes=_gb(memory_capacity_gb),
            product_total_bandwidth_gbps=memory_bandwidth_gbps,
            role="local_accelerator_memory",
            product_model=gpu_model,
            unit_count_status=memory_count_status,
            unit_count_formula=memory_count_formula or (
                "vendor-documented product stack count = {}".format(
                    memory_physical_unit_count
                )
                if memory_count_status.startswith("vendor_documented")
                else "architecture preset parameter stack_count = {}".format(
                    memory_physical_unit_count
                )
            ),
            source_basis=memory_source_basis or (
                "vendor product architecture and aggregate memory specification"
                if memory_count_status.startswith("vendor_documented")
                else "explicit derived architecture-preset parameter and vendor aggregate memory specification"
            ),
            product_total_raw_capacity_bytes=(
                _gb(memory_raw_capacity_gb)
                if memory_raw_capacity_gb is not None
                else None
            ),
            component_preset_id=memory_component_preset_id,
        )
        components.extend(stack_components)
        switch_ports.append(
            _port(switch_port, fabric_protocol, "switch", version=fabric_version, lanes=fabric_lanes, bandwidth_gbps=fabric_bandwidth_gbps, metadata=fabric_metadata)
        )
        links.extend(stack_links)
        links.append(
            _link("fabric_link{}".format(index), accelerator_id, "fabric", "fabric0", switch_port, fabric_protocol, version=fabric_version, lanes=fabric_lanes, bandwidth_gbps=fabric_bandwidth_gbps, latency_ns=100.0, approximation=approximation, metadata=fabric_metadata)
        )
        groups.append(
            _group(
                "module{}".format(index),
                "{} module {}".format(gpu_model, index),
                (accelerator_id,) + stack_ids,
                accelerator_id,
                collapsed=count > 4,
            )
        )
        positions[accelerator_id] = {"x": float((index % 4) * 320), "y": float(220 + (index // 4) * 360)}
        for stack_index, stack_id in enumerate(stack_ids):
            positions[stack_id] = {
                "x": float((index % 4) * 320 + (stack_index % 4) * 56),
                "y": float(340 + (index // 4) * 360 + (stack_index // 4) * 72),
            }
    components.append(
        _component(
            "fabric0",
            fabric_kind,
            switch_ports,
            role="collapsed_nonblocking_fabric",
            model="logical fabric",
            approximation=approximation,
        )
    )
    groups.append(_group("fabric", "Logical nonblocking fabric", ("fabric0",), "fabric0", collapsed=True))
    positions["fabric0"] = {"x": 480.0, "y": 20.0}
    limitations = (
        approximation,
        "链路带宽是每个加速器端点的一向容量，不能乘作任意两卡之间同时可用的全网带宽。",
        "每颗物理 HBM 堆叠使用独立组件和专用链路；单颗容量与带宽由公开产品总量按堆叠数守恒拆分，不反向推断通道或封装走线。",
    ) + (
        ("Gaudi 3 映射到当前仿真器支持的 gpu 计算成本类；组件 model 元数据保留实际产品类型。",)
        if vendor == "Intel"
        else ()
    )
    if vendor == "Intel":
        limitations += (
            "每个逻辑 RoCE 端点采用八个 200GbE 端口的一向物理层 envelope；Gaudi 3 的 24 个集成端口没有被虚构成唯一板级分组。",
        )
    return _definition(
        preset_id,
        name,
        vendor,
        family,
        "accelerator_fabric",
        scale,
        support_level,
        components,
        links,
        groups,
        positions,
        sources,
        limitations,
        "可运行的系统级端点图；折叠 Fabric 只表达公开的互联能力边界。",
        (vendor.lower(), "accelerator", "hbm", fabric_protocol.lower(), "collapsed-fabric"),
    )


def _gh200_superchip(
    *,
    preset_id: str = "nvidia-gh200-superchip",
    name: str = "NVIDIA GH200 Grace Hopper Superchip (96GB HBM3 variant)",
    hbm_generation: str = "HBM3",
    hbm_stack_count: int = 6,
    hbm_capacity_gb: float = 96.0,
    hbm_bandwidth_gbps: float = 32000.0,
    hbm_unit_count_status: str = "derived_from_GH200_total_plus_full_GH100_6_stack_implementation",
    hbm_unit_count_formula: str = "GH200 96 GB total / 16 GB HBM3 stack class = 6",
    hbm_source_basis: str = "GH200 product total plus full-H100 six-stack implementation; count is derived, not labeled vendor-documented",
    hbm_component_preset_id: str = "hbm3-16gb-0_667tbs-product-slice",
) -> ArchitecturePresetDefinition:
    components = [
        _component("grace0", "cpu", (
            _port("gpu", "NVLink-C2C", "endpoint", version="GH200", bandwidth_gbps=3600.0),
            _port("memory", "LPDDR5X", "controller", version="GH200", bandwidth_gbps=4000.0),
        ), package_id="superchip0", die_id="grace_die", peak_ops_per_s=0.0, role="host_cpu", model="Grace CPU"),
        _component("hopper0", "gpu", (
            _port("cpu", "NVLink-C2C", "endpoint", version="GH200", bandwidth_gbps=3600.0),
        ) + _hbm_controller_ports(hbm_stack_count, hbm_generation, hbm_bandwidth_gbps), package_id="superchip0", die_id="hopper_die", peak_ops_per_s=989_500_000_000_000.0, role="accelerator", model="Hopper GPU", peak_ops_basis=_peak_ops_basis("BF16_tensor", source_basis="vendor_product_specification")),
        _component("lpddr0", "host_memory", (_port("host", "LPDDR5X", "device", version="GH200", bandwidth_gbps=4000.0),), package_id="superchip0", die_id="lpddr_array", capacity_bytes=_gb(480), read_bandwidth_gbps=4000.0, write_bandwidth_gbps=4000.0, role="coherent_host_memory", model="LPDDR5X aggregate", physical_composition={**_aggregate_composition("LPDDR5X_device", physical_unit_count=None, count_status="not_reliably_disclosed", known_multiple=True, source_basis="vendor_product_level_memory_subsystem"), "controller_component_id": "grace0", "memory_subsystem_id": "grace0", "component_preset_id": "lpddr5x-gh200-480gb-500gbs-aggregate", "component_preset_status": "catalog_reference"}, parameter_basis={"capacity_bytes": "vendor_product_aggregate_decimal", "bandwidth_gbps": "vendor_product_aggregate_memory_bandwidth"}, extra_metadata={"component_preset_id": "lpddr5x-gh200-480gb-500gbs-aggregate", "capacity_scope": "per_Grace_up_to_480GB", "bandwidth_scope": "vendor_per_Grace_500GBps_aggregate"}),
    ]
    hbm_components, hbm_links = _physical_hbm_stacks(
        root_component_id="hopper0",
        root_port_prefix="hbm",
        stack_id_prefix="hbm",
        link_id_prefix="hbm_link",
        package_id="superchip0",
        generation=hbm_generation,
        stack_count=hbm_stack_count,
        product_total_capacity_bytes=_gb(hbm_capacity_gb),
        product_total_bandwidth_gbps=hbm_bandwidth_gbps,
        role="accelerator_memory",
        product_model="GH200 {} GB {}".format(hbm_capacity_gb, hbm_generation),
        unit_count_status=hbm_unit_count_status,
        unit_count_formula=hbm_unit_count_formula,
        source_basis=hbm_source_basis,
        component_preset_id=hbm_component_preset_id,
    )
    components.extend(hbm_components)
    links = [
        _link("c2c0", "grace0", "gpu", "hopper0", "cpu", "NVLink-C2C", version="GH200", bandwidth_gbps=3600.0, latency_ns=50.0),
        _link("lpddr0", "grace0", "memory", "lpddr0", "host", "LPDDR5X", version="GH200", bandwidth_gbps=4000.0, latency_ns=80.0),
    ]
    links.extend(hbm_links)
    hbm_ids = tuple(component.component_id for component in hbm_components)
    groups = (
        _group(
            "superchip0",
            "GH200 superchip",
            ("grace0", "hopper0", "lpddr0") + hbm_ids,
            "hopper0",
        ),
    )
    limitations = (
        "只表达公开的 Grace、Hopper、LPDDR5X、HBM 与 NVLink-C2C 逻辑关系，不表达封装物理走线。",
        "LPDDR5X 仍按产品级容量和带宽聚合；{} 已按参数化的 {} 颗物理节点展开，数量 provenance 记录在每颗节点中。".format(hbm_generation, hbm_stack_count),
    )
    positions = {
        "grace0": {"x": 80.0, "y": 100.0},
        "hopper0": {"x": 400.0, "y": 100.0},
        "lpddr0": {"x": 80.0, "y": 320.0},
    }
    for index, hbm_id in enumerate(hbm_ids):
        positions[hbm_id] = {
            "x": 360.0 + float((index % 3) * 100),
            "y": 300.0 + float((index // 3) * 100),
        }
    return _definition(preset_id, name, "NVIDIA", "Grace Hopper", "coherent_superchip", "single_superchip", EXACT_PUBLIC_TOPOLOGY, components, links, groups, positions, (NVIDIA_GH200,), limitations, "硬件-only 的公开逻辑拓扑；载入后由调用方使现有映射失效。", ("nvidia", "grace", "hopper", hbm_generation.lower(), "coherent", "c2c"))


def _gh200_nvl2(
    *,
    preset_id: str = "nvidia-gh200-nvl2",
    name: str = "NVIDIA GH200 NVL2 (2×144GB HBM3E current variant)",
    hbm_generation: str = "HBM3E",
    hbm_stack_count: int = 6,
    hbm_capacity_gb_per_superchip: float = 144.0,
    hbm_bandwidth_gbps_per_superchip: float = 40000.0,
    hbm_unit_count_status: str = "derived_from_product_total_and_24GB_stack_class",
    hbm_unit_count_formula: str = "144 GB per-Superchip total / 24 GB HBM3E stack class = 6",
    hbm_source_basis: str = "current GH200 NVL2 product aggregate plus 24 GB HBM3E stack class; count is derived",
    hbm_component_preset_id: str = "hbm3e-24gb-0_833tbs-gh200-slice",
) -> ArchitecturePresetDefinition:
    components = []
    links = []
    groups = []
    positions: Dict[str, Dict[str, float]] = {}
    for index in range(2):
        x = float(index * 620)
        grace = "grace{}".format(index)
        hopper = "hopper{}".format(index)
        lpddr = "lpddr{}".format(index)
        components.extend((
            _component(grace, "cpu", (_port("gpu", "NVLink-C2C", "endpoint", version="GH200", bandwidth_gbps=3600.0), _port("memory", "LPDDR5X", "controller", version="GH200", bandwidth_gbps=4000.0)), package_id="superchip{}".format(index), die_id="grace{}".format(index), role="host_cpu", model="Grace CPU"),
            _component(hopper, "gpu", (_port("cpu", "NVLink-C2C", "endpoint", version="GH200", bandwidth_gbps=3600.0),) + _hbm_controller_ports(hbm_stack_count, hbm_generation, hbm_bandwidth_gbps_per_superchip) + (_port("peer", "NVLink", "endpoint", version="4.0", bandwidth_gbps=3600.0),), package_id="superchip{}".format(index), die_id="hopper{}".format(index), peak_ops_per_s=989_500_000_000_000.0, role="accelerator", model="Hopper GPU", peak_ops_basis=_peak_ops_basis("BF16_tensor", source_basis="vendor_product_specification")),
            _component(lpddr, "host_memory", (_port("host", "LPDDR5X", "device", version="GH200", bandwidth_gbps=4000.0),), package_id="superchip{}".format(index), die_id="lpddr{}".format(index), capacity_bytes=_gb(480), read_bandwidth_gbps=4000.0, write_bandwidth_gbps=4000.0, role="host_memory", model="LPDDR5X aggregate", physical_composition={**_aggregate_composition("LPDDR5X_device", physical_unit_count=None, count_status="not_reliably_disclosed", known_multiple=True, source_basis="vendor_product_level_memory_subsystem"), "controller_component_id": grace, "memory_subsystem_id": grace, "component_preset_id": "lpddr5x-gh200-480gb-500gbs-aggregate", "component_preset_status": "catalog_reference"}, parameter_basis={"capacity_bytes": "vendor_product_aggregate_decimal", "bandwidth_gbps": "vendor_product_aggregate_memory_bandwidth"}, extra_metadata={"component_preset_id": "lpddr5x-gh200-480gb-500gbs-aggregate", "capacity_scope": "per_Grace_up_to_480GB", "bandwidth_scope": "vendor_per_Grace_500GBps_aggregate"}),
        ))
        hbm_components, hbm_links = _physical_hbm_stacks(
            root_component_id=hopper,
            root_port_prefix="hbm",
            stack_id_prefix="hbm{}_".format(index),
            link_id_prefix="hbm_link{}_".format(index),
            package_id="superchip{}".format(index),
            generation=hbm_generation,
            stack_count=hbm_stack_count,
            product_total_capacity_bytes=_gb(hbm_capacity_gb_per_superchip),
            product_total_bandwidth_gbps=hbm_bandwidth_gbps_per_superchip,
            role="accelerator_memory",
            product_model="GH200 {} GB {}".format(hbm_capacity_gb_per_superchip, hbm_generation),
            unit_count_status=hbm_unit_count_status,
            unit_count_formula=hbm_unit_count_formula,
            source_basis=hbm_source_basis,
            component_preset_id=hbm_component_preset_id,
        )
        components.extend(hbm_components)
        links.extend((
            _link("c2c{}".format(index), grace, "gpu", hopper, "cpu", "NVLink-C2C", version="GH200", bandwidth_gbps=3600.0, latency_ns=50.0),
            _link("lpddr{}".format(index), grace, "memory", lpddr, "host", "LPDDR5X", version="GH200", bandwidth_gbps=4000.0, latency_ns=80.0),
        ))
        links.extend(hbm_links)
        hbm_ids = tuple(component.component_id for component in hbm_components)
        groups.append(_group("superchip{}".format(index), "GH200 superchip {}".format(index), (grace, hopper, lpddr) + hbm_ids, hopper))
        positions.update({grace: {"x": x, "y": 100.0}, hopper: {"x": x + 280.0, "y": 100.0}, lpddr: {"x": x, "y": 300.0}})
        for stack_index, hbm_id in enumerate(hbm_ids):
            positions[hbm_id] = {
                "x": x + 260.0 + float((stack_index % 3) * 90),
                "y": 280.0 + float((stack_index // 3) * 90),
            }
    links.append(_link("nvl2_peer", "hopper0", "peer", "hopper1", "peer", "NVLink", version="4.0", bandwidth_gbps=3600.0, latency_ns=100.0))
    limitations = (
        "两颗 GH200 之间仅建模公开的 GPU 对等 NVLink 逻辑连接；不推断板级链路分配。",
        "主机内存仍按每颗 Superchip 聚合；{} 按每颗 Superchip 参数化的 {} 颗物理节点展开，数量 provenance 记录在每颗节点中。".format(hbm_generation, hbm_stack_count),
    )
    return _definition(preset_id, name, "NVIDIA", "Grace Hopper", "coherent_superchip", "dual_superchip", ANALYTICAL_APPROXIMATION, components, links, groups, positions, (NVIDIA_GH200,), limitations, "双 Superchip 系统级分析图；产品变体与显存口径在名称和节点 provenance 中显式区分。", ("nvidia", "gh200", "nvl2", hbm_generation.lower(), "coherent"))


def _gb200_system(
    count: int,
    preset_id: str,
    name: str,
    scale: str,
    *,
    hbm_stack_count_per_gpu: int = 8,
    hbm_unit_count_status: str = "vendor_documented_stack_sites",
) -> ArchitecturePresetDefinition:
    components = []
    links = []
    groups = []
    positions: Dict[str, Dict[str, float]] = {}
    switch_ports = []
    superchips = count // 2
    if count <= 4:
        lpddr_bandwidth_gbps_per_grace = 4096.0
        lpddr_bandwidth_basis = (
            "NVL4 two-Grace 1.024 TB/s aggregate equally attributed to two Grace memory domains"
        )
        lpddr_system_bandwidth_gbps = 8192.0
        lpddr_component_preset_status = "catalog_reference"
    else:
        lpddr_system_bandwidth_gbps = 112000.0
        lpddr_bandwidth_gbps_per_grace = (
            lpddr_system_bandwidth_gbps / superchips
        )
        lpddr_bandwidth_basis = (
            "NVL72 rack 14 TB/s host-memory envelope equally attributed for analysis; not a vendor per-Grace total"
        )
        lpddr_component_preset_status = "catalog_reference_with_system_bandwidth_override"
    approximation = "以单个参数化非阻塞 NVLink Fabric 表达系统边界；不声称还原未公开的交换芯片级物理布线。"
    grace_group_collapsed = count > 4
    gpu_hbm_group_collapsed = True
    for module in range(superchips):
        base_x = float((module % 6) * 1400)
        base_y = float(280 + (module // 6) * 520)
        cpu = "grace{}".format(module)
        lpddr = "lpddr{}".format(module)
        components.append(_component(cpu, "cpu", tuple(_port("gpu{}".format(local), "NVLink-C2C", "endpoint", bandwidth_gbps=3600.0) for local in range(2)) + (_port("memory", "LPDDR5X", "controller", version="GB200", bandwidth_gbps=lpddr_bandwidth_gbps_per_grace, metadata={"bandwidth_basis": lpddr_bandwidth_basis}),), package_id="superchip{}".format(module), die_id="grace{}".format(module), role="host_cpu", model="Grace CPU"))
        components.append(
            _component(
                lpddr,
                "host_memory",
                (
                    _port(
                        "host",
                        "LPDDR5X",
                        "device",
                        version="GB200",
                        bandwidth_gbps=lpddr_bandwidth_gbps_per_grace,
                        metadata={"bandwidth_basis": lpddr_bandwidth_basis},
                    ),
                ),
                package_id="superchip{}".format(module),
                die_id="lpddr_array{}".format(module),
                capacity_bytes=_gb(480),
                read_bandwidth_gbps=lpddr_bandwidth_gbps_per_grace,
                write_bandwidth_gbps=lpddr_bandwidth_gbps_per_grace,
                role="coherent_host_memory",
                model="Grace LPDDR5X aggregate",
                physical_composition={
                    **_aggregate_composition(
                        "LPDDR5X_device",
                        physical_unit_count=None,
                        count_status="not_reliably_disclosed",
                        known_multiple=True,
                        source_basis="vendor per-Grace capacity aggregate; physical DRAM count unknown",
                    ),
                    "controller_component_id": cpu,
                    "memory_subsystem_id": cpu,
                    "component_preset_id": "lpddr5x-gb200-480gb-512gbs-aggregate",
                    "component_preset_status": lpddr_component_preset_status,
                },
                parameter_basis={
                    "capacity_bytes": "vendor per-Grace aggregate up to 480 GB",
                    "bandwidth_gbps": lpddr_bandwidth_basis,
                    "system_total_capacity_bytes": _gb(480) * superchips,
                    "system_total_bandwidth_gbps": lpddr_system_bandwidth_gbps,
                },
                extra_metadata={
                    "component_preset_id": "lpddr5x-gb200-480gb-512gbs-aggregate",
                    "capacity_scope": "per_Grace_up_to_480GB",
                    "bandwidth_scope": (
                        "per_Grace_derived_from_NVL4_aggregate"
                        if count <= 4
                        else "per_Grace_analysis_share_of_NVL72_rack_14TBps_envelope"
                    ),
                },
            )
        )
        positions[cpu] = {"x": base_x + 460.0, "y": base_y}
        positions[lpddr] = {"x": base_x + 680.0, "y": base_y}
        links.append(
            _link(
                "lpddr{}".format(module),
                cpu,
                "memory",
                lpddr,
                "host",
                "LPDDR5X",
                version="GB200",
                bandwidth_gbps=lpddr_bandwidth_gbps_per_grace,
                latency_ns=80.0,
                metadata={
                    "bandwidth_basis": lpddr_bandwidth_basis,
                    "system_total_bandwidth_gbps": lpddr_system_bandwidth_gbps,
                    "physical_device_count_asserted": False,
                },
            )
        )
        groups.append(
            _group(
                "grace{}_lpddr".format(module),
                "GB200 Grace {} + LPDDR5X".format(module),
                (cpu, lpddr),
                cpu,
                collapsed=grace_group_collapsed,
            )
        )
        for local in range(2):
            index = module * 2 + local
            gpu = "gpu{}".format(index)
            endpoint = "endpoint{}".format(index)
            components.append(
                _component(
                    gpu,
                    "gpu",
                    (_port("cpu", "NVLink-C2C", "endpoint", bandwidth_gbps=3600.0),)
                    + _hbm_controller_ports(
                        hbm_stack_count_per_gpu,
                        "HBM3E",
                        64000.0,
                    )
                    + (_port("fabric", "NVLink", "endpoint", version="5.0", bandwidth_gbps=7200.0),),
                    package_id="superchip{}".format(module),
                    die_id="blackwell{}".format(index),
                    peak_ops_per_s=2_500_000_000_000_000.0,
                    role="accelerator",
                    model="Blackwell GPU",
                    peak_ops_basis=_peak_ops_basis(
                        "BF16_tensor",
                        source_basis="vendor table sparse value divided by two to dense; then divided by GPU count",
                    ),
                )
            )
            hbm_components, hbm_links = _physical_hbm_stacks(
                root_component_id=gpu,
                root_port_prefix="hbm",
                stack_id_prefix="hbm{}_".format(index),
                link_id_prefix="hbm_link{}_".format(index),
                package_id="superchip{}".format(module),
                generation="HBM3E",
                stack_count=hbm_stack_count_per_gpu,
                product_total_capacity_bytes=_gb(186),
                product_total_bandwidth_gbps=64000.0,
                role="accelerator_memory",
                product_model="GB200 per-GPU 186 GB HBM3E",
                unit_count_status=hbm_unit_count_status,
                unit_count_formula="official Blackwell brief: 8 HBM3E stack sites per GPU (8×2 per Superchip)",
                source_basis="NVIDIA Blackwell architecture brief stack-site count and GB200 aggregate memory specification",
                product_total_raw_capacity_bytes=_gb(192),
                component_preset_id="hbm3e-24gb-1_000tbs-gb200-slice",
            )
            components.extend(hbm_components)
            switch_ports.append(_port(endpoint, "NVLink", "switch", version="5.0", bandwidth_gbps=7200.0))
            links.extend((
                _link("c2c{}".format(index), cpu, "gpu{}".format(local), gpu, "cpu", "NVLink-C2C", bandwidth_gbps=3600.0, latency_ns=50.0),
                _link("fabric{}".format(index), gpu, "fabric", "fabric0", endpoint, "NVLink", version="5.0", bandwidth_gbps=7200.0, latency_ns=100.0, approximation=approximation),
            ))
            links.extend(hbm_links)
            hbm_ids = tuple(component.component_id for component in hbm_components)
            groups.append(
                _group(
                    "gpu{}_hbm".format(index),
                    "GB200 GPU {} + {} HBM3E".format(
                        index,
                        hbm_stack_count_per_gpu,
                    ),
                    (gpu,) + hbm_ids,
                    gpu,
                    collapsed=gpu_hbm_group_collapsed,
                )
            )
            gpu_base_x = base_x + float(local * 650)
            positions[gpu] = {"x": gpu_base_x, "y": base_y + 150.0}
            for stack_index, hbm_id in enumerate(hbm_ids):
                positions[hbm_id] = {
                    "x": gpu_base_x + float((stack_index % 4) * 150),
                    "y": base_y + float(270 + (stack_index // 4) * 90),
                }
    components.append(_component("fabric0", "fabric_switch", switch_ports, role="collapsed_nonblocking_fabric", model="NVLink Fabric", approximation=approximation))
    groups.append(_group("fabric", "Collapsed NVLink Fabric", ("fabric0",), "fabric0", collapsed=True))
    positions["fabric0"] = {"x": float(max(0, min(superchips, 6) - 1) * 280), "y": 20.0}
    limitations = (
        approximation,
        "模型保留 {} 个 GPU 端点并只创建 {} 条 GPU-Fabric 链路，避免全互联链路爆炸。".format(count, count),
        "每颗物理 HBM 使用独立组件与专用链路；链路值是单颗的一向等分容量，不是全机双向聚合宣传值。",
        "每 GPU 的公开 186GB 可见 HBM3E、192GB 原始堆叠容量与每秒 8TB 总带宽按官方八个堆叠位点守恒拆分。",
        "每颗 Grace 增加一个最高 480GB 的 LPDDR5X 聚合节点，物理 DRAM 数量保持 unknown/null；其带宽 provenance 区分 NVL4 聚合等分与 NVL72 机架每秒 14TB 分析口径。",
        "peak_ops_per_s 采用公开 FP16/BF16 稀疏规格除以二后的 dense 值，不计结构化稀疏加速。",
    )
    return _definition(preset_id, name, "NVIDIA", "GB200", "accelerator_fabric", scale, ANALYTICAL_APPROXIMATION, components, links, groups, positions, (NVIDIA_GB200,), limitations, "可扩展的折叠 Fabric 分析模型，不声称还原物理交换布线。", ("nvidia", "gb200", "blackwell", "nvlink", "collapsed-fabric"))


def _mi300a(hbm_stack_count: int = 8) -> ArchitecturePresetDefinition:
    components = [
        _component("zen4_cpu0", "cpu", (_port("gpu", "InfinityFabric", "endpoint", version="4.0", bandwidth_gbps=2048.0),), package_id="mi300a0", die_id="cpu_complex", role="host_cpu_chiplets", model="Zen 4 CPU complex", approximation="CPU chiplets are collapsed into one logical component."),
        _component("cdna_gpu0", "gpu", (_port("cpu", "InfinityFabric", "endpoint", version="4.0", bandwidth_gbps=2048.0),) + _hbm_controller_ports(hbm_stack_count, "HBM3", 42400.0), package_id="mi300a0", die_id="gpu_complex", peak_ops_per_s=980_600_000_000_000.0, role="gpu_chiplets", model="CDNA 3 GPU complex", approximation="GPU chiplets are collapsed into one logical component.", peak_ops_basis=_peak_ops_basis("BF16_matrix", source_basis="vendor_product_specification")),
    ]
    hbm_components, hbm_links = _physical_hbm_stacks(
        root_component_id="cdna_gpu0",
        root_port_prefix="hbm",
        stack_id_prefix="hbm",
        link_id_prefix="hbm_link",
        package_id="mi300a0",
        generation="HBM3",
        stack_count=hbm_stack_count,
        product_total_capacity_bytes=_gb(128),
        product_total_bandwidth_gbps=42400.0,
        role="unified_hbm",
        product_model="MI300A 128 GB unified HBM3",
        unit_count_status="derived_cross_source_product_architecture",
        unit_count_formula="cross-source MI300A package architecture count = 8 HBM3 stacks",
        source_basis="AMD product aggregate plus cross-source package-architecture count",
        component_preset_id="hbm3-16gb-0_6625tbs-mi300a-slice",
    )
    components.extend(hbm_components)
    links = [
        _link("if0", "zen4_cpu0", "gpu", "cdna_gpu0", "cpu", "InfinityFabric", version="4.0", bandwidth_gbps=2048.0, latency_ns=60.0, approximation="Parameterized package-level coherent path."),
    ]
    links.extend(hbm_links)
    hbm_ids = tuple(component.component_id for component in hbm_components)
    groups = (_group("package0", "MI300A APU package", ("zen4_cpu0", "cdna_gpu0") + hbm_ids, "cdna_gpu0"),)
    limitations = ("CPU、GPU 与 HBM 的公开逻辑组成是准确的；计算 chiplet 保持逻辑折叠，但八颗物理 HBM 堆叠分别建模。", "Infinity Fabric 的带宽和时延是可覆盖的分析参数，不代表封装内部逐链路测量。")
    positions = {"zen4_cpu0": {"x": 40.0, "y": 120.0}, "cdna_gpu0": {"x": 360.0, "y": 120.0}}
    for index, hbm_id in enumerate(hbm_ids):
        positions[hbm_id] = {"x": 260.0 + float((index % 4) * 100), "y": 320.0 + float((index // 4) * 90)}
    return _definition("amd-mi300a-apu", "AMD Instinct MI300A APU", "AMD", "MI300", "coherent_apu", "single_package", EXACT_PUBLIC_TOPOLOGY, components, links, groups, positions, (AMD_MI300A,), limitations, "公开封装逻辑关系的系统级表示。", ("amd", "mi300a", "apu", "unified-memory"))


def _mi300x_single(hbm_stack_count: int = 8) -> ArchitecturePresetDefinition:
    components = [
        _component("mi300x0", "gpu", _hbm_controller_ports(hbm_stack_count, "HBM3", 42600.0), package_id="mi300x0", die_id="gpu_complex", peak_ops_per_s=1_307_400_000_000_000.0, role="accelerator", model="MI300X", peak_ops_basis=_peak_ops_basis("BF16_matrix", source_basis="vendor_product_specification")),
    ]
    hbm_components, hbm_links = _physical_hbm_stacks(
        root_component_id="mi300x0",
        root_port_prefix="hbm",
        stack_id_prefix="hbm",
        link_id_prefix="hbm_link",
        package_id="mi300x0",
        generation="HBM3",
        stack_count=hbm_stack_count,
        product_total_capacity_bytes=_gb(192),
        product_total_bandwidth_gbps=42600.0,
        role="accelerator_memory",
        product_model="MI300X 192 GB HBM3",
        unit_count_status="vendor_documented_stacks",
        unit_count_formula="AMD product architecture documents 8 HBM3 stacks",
        source_basis="AMD product architecture and aggregate memory specification",
        component_preset_id="hbm3-24gb-0_665625tbs-mi300x-slice",
    )
    components.extend(hbm_components)
    hbm_ids = tuple(component.component_id for component in hbm_components)
    groups = (_group("package0", "MI300X package", ("mi300x0",) + hbm_ids, "mi300x0"),)
    limitations = ("八颗物理 HBM3 堆叠分别建模；单颗可见容量和带宽由公开产品总量守恒拆分。", "不反向推断 XCD/IOD 间物理 Infinity Fabric 走线。")
    positions = {"mi300x0": {"x": 260.0, "y": 100.0}}
    for index, hbm_id in enumerate(hbm_ids):
        positions[hbm_id] = {"x": 120.0 + float((index % 4) * 100), "y": 300.0 + float((index // 4) * 90)}
    return _definition("amd-mi300x", "AMD Instinct MI300X", "AMD", "MI300", "integrated_accelerator", "single_package", EXACT_PUBLIC_TOPOLOGY, components, hbm_links, groups, positions, (AMD_MI300X,), limitations, "公开加速器与 HBM 逻辑关系。", ("amd", "mi300x", "hbm3"))


def _cxl_type3() -> ArchitecturePresetDefinition:
    components = (
        _component("host0", "cpu", (_port("cxl0", "CXL", "host", version="3.0", lanes=16, bandwidth_gbps=1024.0),), role="cxl_host", model="generic CXL host"),
        _component("cxl_memory0", "cxl_memory", (_port("cxl0", "CXL", "device", version="3.0", lanes=16, bandwidth_gbps=1024.0),), capacity_bytes=_gb(512), read_bandwidth_gbps=1024.0, write_bandwidth_gbps=1024.0, role="type3_memory_device", model="parameterized CXL Type-3 memory", approximation="Capacity and media latency are reference parameters."),
    )
    links = (_link("cxl0", "host0", "cxl0", "cxl_memory0", "cxl0", "CXL", version="3.0", lanes=16, bandwidth_gbps=1024.0, latency_ns=180.0),)
    groups = (_group("host", "CXL host", ("host0",), "host0"), _group("memory", "Type-3 memory", ("cxl_memory0",), "cxl_memory0"))
    limitations = ("Host 到 Type-3 设备的协议角色与直接连接是规范级拓扑；容量、介质时延和有效负载效率是可覆盖参数。", "链路带宽是物理层一向上限，未伪造统一的 CXL.mem 有效负载折扣。")
    return _definition("cxl-type3-memory-expander", "CXL 3.0 Type-3 Memory Expander", "CXL Consortium", "CXL", "disaggregated_memory", "direct_attach", EXACT_PUBLIC_TOPOLOGY, components, links, groups, {"host0": {"x": 80.0, "y": 160.0}, "cxl_memory0": {"x": 440.0, "y": 160.0}}, (CXL_SPEC,), limitations, "规范角色正确的 Type-3 直接连接模板。", ("cxl", "type3", "memory-expansion"))


def _cxl_pool() -> ArchitecturePresetDefinition:
    components = [
        _component("host0", "cpu", (_port("cxl0", "CXL", "host", version="3.0", lanes=16, bandwidth_gbps=1024.0),), role="cxl_host", model="host 0"),
        _component("host1", "cpu", (_port("cxl0", "CXL", "host", version="3.0", lanes=16, bandwidth_gbps=1024.0),), role="cxl_host", model="host 1"),
        _component("cxl_switch0", "fabric_switch", (
            _port("host0", "CXL", "device", version="3.0", lanes=16, bandwidth_gbps=1024.0),
            _port("host1", "CXL", "device", version="3.0", lanes=16, bandwidth_gbps=1024.0),
            _port("memory0", "CXL", "host", version="3.0", lanes=16, bandwidth_gbps=1024.0),
            _port("memory1", "CXL", "host", version="3.0", lanes=16, bandwidth_gbps=1024.0),
        ), role="logical_cxl_switch", model="CXL 3.0 logical switch", approximation="Fabric manager and internal switch contention are not modeled."),
    ]
    for index in range(2):
        components.append(_component("memory{}".format(index), "cxl_memory", (_port("cxl0", "CXL", "device", version="3.0", lanes=16, bandwidth_gbps=1024.0),), capacity_bytes=_gb(512), read_bandwidth_gbps=1024.0, write_bandwidth_gbps=1024.0, role="pooled_type3_memory", model="parameterized Type-3 device"))
    links = (
        _link("host0", "host0", "cxl0", "cxl_switch0", "host0", "CXL", version="3.0", lanes=16, bandwidth_gbps=1024.0, latency_ns=120.0),
        _link("host1", "host1", "cxl0", "cxl_switch0", "host1", "CXL", version="3.0", lanes=16, bandwidth_gbps=1024.0, latency_ns=120.0),
        _link("memory0", "cxl_switch0", "memory0", "memory0", "cxl0", "CXL", version="3.0", lanes=16, bandwidth_gbps=1024.0, latency_ns=180.0),
        _link("memory1", "cxl_switch0", "memory1", "memory1", "cxl0", "CXL", version="3.0", lanes=16, bandwidth_gbps=1024.0, latency_ns=180.0),
    )
    groups = (_group("hosts", "CXL hosts", ("host0", "host1"), "host0"), _group("fabric", "CXL fabric", ("cxl_switch0",), "cxl_switch0", collapsed=True), _group("pool", "Pooled Type-3 memory", ("memory0", "memory1"), "memory0"))
    limitations = ("表达 CXL 3.0 交换与内存池的规范级角色；Fabric Manager、动态绑定和一致性状态机未建模。", "交换争用、介质时延与容量是分析参数，不代表某一商用设备。")
    return _definition("cxl-memory-pool", "CXL 3.0 Shared Memory Pool", "CXL Consortium", "CXL", "disaggregated_memory", "multi_host_pool", ANALYTICAL_APPROXIMATION, components, links, groups, {"host0": {"x": 0.0, "y": 80.0}, "host1": {"x": 0.0, "y": 300.0}, "cxl_switch0": {"x": 360.0, "y": 190.0}, "memory0": {"x": 720.0, "y": 80.0}, "memory1": {"x": 720.0, "y": 300.0}}, (CXL_SPEC,), limitations, "多主机共享内存池的逻辑 Fabric 模板。", ("cxl", "memory-pool", "switch", "multi-host"))


def _ucie_chiplet() -> ArchitecturePresetDefinition:
    pkg = "chiplet_package0"
    components = (
        _component("cpu_chiplet0", "cpu", (_port("io", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming"),), package_id=pkg, die_id="cpu_die", role="cpu_chiplet", model="parameterized CPU chiplet"),
        _component("accelerator_chiplet0", "generic_accelerator", (_port("io", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming"),), package_id=pkg, die_id="accelerator_die", peak_ops_per_s=512_000_000_000_000.0, role="accelerator_chiplet", model="parameterized accelerator"),
        _component("io_die0", "io_die", (_port("cpu", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming"), _port("accelerator", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming")), package_id=pkg, die_id="io_die", role="package_io_die", model="parameterized I/O die"),
    )
    links = (
        _link("ucie_cpu", "cpu_chiplet0", "io", "io_die0", "cpu", "UCIe", version="2.0", lanes=64, bandwidth_gbps=2048.0, latency_ns=20.0, payload="streaming"),
        _link("ucie_accelerator", "accelerator_chiplet0", "io", "io_die0", "accelerator", "UCIe", version="2.0", lanes=64, bandwidth_gbps=2048.0, latency_ns=20.0, payload="streaming"),
    )
    groups = (_group("compute", "Compute chiplets", ("cpu_chiplet0", "accelerator_chiplet0"), "accelerator_chiplet0"), _group("io", "Package I/O", ("io_die0",), "io_die0"))
    limitations = ("端口角色、封装内约束与 Streaming payload 符合 UCIe 语义；组件能力和链路有效效率是实验参数。", "这不是任何厂商产品的裸片布局或 bump map。")
    return _definition("ucie-chiplet-package", "UCIe 2.0 CPU + Accelerator Chiplet Package", "UCIe Consortium", "UCIe", "chiplet_package", "single_package", EXPERIMENTAL_REFERENCE, components, links, groups, {"cpu_chiplet0": {"x": 0.0, "y": 80.0}, "accelerator_chiplet0": {"x": 0.0, "y": 300.0}, "io_die0": {"x": 420.0, "y": 190.0}}, (UCIE_SPEC,), limitations, "协议合法的芯粒实验参考，不对应具体流片。", ("ucie", "chiplet", "accelerator", "experimental"))


def _gpu_cim() -> ArchitecturePresetDefinition:
    pkg = "heterogeneous_package0"
    components = (
        _component("gpu0", "gpu", (_port("memory", "HBM", "controller", version="HBM3", lanes=16, bandwidth_gbps=6553.6), _port("cim", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming")), package_id=pkg, die_id="gpu_die", peak_ops_per_s=989_500_000_000_000.0, role="host_accelerator", model="parameterized GPU"),
        _component("hbm0", "hbm", (_port("host", "HBM", "device", version="HBM3", lanes=16, bandwidth_gbps=6553.6),), package_id=pkg, die_id="hbm_die", capacity_bytes=_gb(24), read_bandwidth_gbps=6553.6, write_bandwidth_gbps=6553.6, role="active_memory", model="HBM3 analytical stack", physical_composition={"simulator_representation": "single_physical_unit_node", "simulator_node_count": 1, "physical_unit_kind": "HBM_stack", "physical_unit_count": 1, "physical_unit_count_status": "explicit_physical_node", "known_multiple": False, "controller_component_id": "gpu0", "memory_subsystem_id": "gpu0", "unit_index": 0, "unit_count_in_product": 1, "unit_count_status": "experimental_reference_scope", "unit_count_formula": "one analytical HBM3 stack in the experimental preset", "unit_capacity_bytes": _gb(24), "product_total_capacity_bytes": _gb(24), "unit_raw_capacity_bytes": _gb(24), "product_total_raw_capacity_bytes": _gb(24), "capacity_accounting": "analytical_reference_capacity", "unit_bandwidth_gbps": 6553.6, "product_total_bandwidth_gbps": 6553.6, "component_preset_id": "jedec-hbm3-24gb-6_4", "component_preset_status": "catalog_reference", "source_basis": "analytical HBM3 reference-stack parameter"}, parameter_basis={"capacity_bytes": "analytical HBM3 stack parameter", "bandwidth_gbps": "analytical per-stack interface envelope"}, extra_metadata={"provenance": {"value_status": "analytical_reference", "unit_count_status": "experimental_reference_scope", "source_basis": "analytical HBM3 reference-stack parameter"}}),
        _component("cim0", "digital_sram_cim", (_port("host", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming"),), package_id=pkg, die_id="cim_die", capacity_bytes=512 * 1024 * 1024, peak_ops_per_s=512_000_000_000_000.0, read_bandwidth_gbps=2048.0, write_bandwidth_gbps=2048.0, role="near_memory_compute", model="digital SRAM-CIM analytical tile"),
    )
    links = (
        _link("hbm0", "gpu0", "memory", "hbm0", "host", "HBM", version="HBM3", lanes=16, bandwidth_gbps=6553.6, latency_ns=40.0),
        _link("cim0", "gpu0", "cim", "cim0", "host", "UCIe", version="2.0", lanes=64, bandwidth_gbps=2048.0, latency_ns=20.0, payload="streaming"),
    )
    groups = (_group("host", "GPU and HBM", ("gpu0", "hbm0"), "gpu0"), _group("near_memory", "CIM tile", ("cim0",), "cim0"))
    limitations = ("HBM3 与 UCIe 的协议边界可运行，但 CIM 容量、算力和调度是分析参数。", "模板不自动创建模型 placement，也不假定任意算子适合 CIM。")
    return _definition("gpu-hbm-cim", "GPU + HBM + SRAM-CIM", "Cross-vendor", "Near-memory Compute", "near_memory_compute", "single_package", EXPERIMENTAL_REFERENCE, components, links, groups, {"gpu0": {"x": 200.0, "y": 80.0}, "hbm0": {"x": 0.0, "y": 300.0}, "cim0": {"x": 420.0, "y": 300.0}}, (JEDEC_HBM, UCIE_SPEC), limitations, "协议合法的异构计算实验参考。", ("gpu", "hbm", "cim", "ucie"))


def _hbm_pim() -> ArchitecturePresetDefinition:
    pkg = "pim_package0"
    components = (
        _component("gpu0", "gpu", (_port("pim", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming"),), package_id=pkg, die_id="gpu_die", peak_ops_per_s=989_500_000_000_000.0, role="host_accelerator", model="parameterized GPU"),
        _component("pim0", "pim_accelerator", (_port("host", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming"), _port("memory", "HBM", "controller", version="HBM2E", lanes=8, bandwidth_gbps=3276.8)), package_id=pkg, die_id="pim_logic", peak_ops_per_s=1_200_000_000_000.0, role="processing_in_memory_logic", model="HBM-PIM analytical logic"),
        _component("hbm_pim0", "hbm", (_port("logic", "HBM", "device", version="HBM2E", lanes=8, bandwidth_gbps=3276.8),), package_id=pkg, die_id="hbm_pim_stack", capacity_bytes=_gb(16), read_bandwidth_gbps=3276.8, write_bandwidth_gbps=3276.8, role="pim_memory_array", model="HBM2E-PIM reference stack", physical_composition={"simulator_representation": "single_physical_unit_node", "simulator_node_count": 1, "physical_unit_kind": "HBM_stack", "physical_unit_count": 1, "physical_unit_count_status": "explicit_physical_node", "known_multiple": False, "controller_component_id": "pim0", "memory_subsystem_id": "pim0", "unit_index": 0, "unit_count_in_product": 1, "unit_count_status": "experimental_reference_scope", "unit_count_formula": "one analytical HBM2E-PIM stack in the experimental preset", "unit_capacity_bytes": _gb(16), "product_total_capacity_bytes": _gb(16), "unit_raw_capacity_bytes": _gb(16), "product_total_raw_capacity_bytes": _gb(16), "capacity_accounting": "analytical_reference_capacity", "unit_bandwidth_gbps": 3276.8, "product_total_bandwidth_gbps": 3276.8, "component_preset_id": "hbm-pim-16gb-3_2-analysis", "component_preset_status": "catalog_reference", "source_basis": "analytical HBM-PIM reference parameter"}, parameter_basis={"capacity_bytes": "analytical HBM-PIM reference parameter", "bandwidth_gbps": "analytical per-stack interface envelope"}, extra_metadata={"provenance": {"value_status": "analytical_reference", "unit_count_status": "experimental_reference_scope", "source_basis": "analytical HBM-PIM reference parameter"}}),
    )
    links = (
        _link("ucie0", "gpu0", "pim", "pim0", "host", "UCIe", version="2.0", lanes=64, bandwidth_gbps=2048.0, latency_ns=20.0, payload="streaming"),
        _link("hbm0", "pim0", "memory", "hbm_pim0", "logic", "HBM", version="HBM2E", lanes=8, bandwidth_gbps=3276.8, latency_ns=40.0),
    )
    groups = (_group("host", "Host GPU", ("gpu0",), "gpu0"), _group("pim", "HBM-PIM subsystem", ("pim0", "hbm_pim0"), "pim0"))
    limitations = ("HBM-PIM 是技术参考而非通用量产部件模板；PIM 算力与指令覆盖需按目标研究覆盖。", "UCIe 主机接口是为保持当前 IR 可运行而采用的实验集成边界，不声称属于 Samsung 公布封装。")
    return _definition("hbm-pim", "HBM-PIM Processing-in-Memory Reference", "Samsung Electronics", "HBM-PIM", "near_memory_compute", "single_package", EXPERIMENTAL_REFERENCE, components, links, groups, {"gpu0": {"x": 0.0, "y": 160.0}, "pim0": {"x": 360.0, "y": 160.0}, "hbm_pim0": {"x": 720.0, "y": 160.0}}, (SAMSUNG_PIM, UCIE_SPEC), limitations, "明确标注接口假设的 HBM-PIM 实验参考。", ("hbm", "pim", "near-memory", "experimental"))


def _gds() -> ArchitecturePresetDefinition:
    components = (
        _component("host0", "cpu", (_port("gpu", "PCIe", "root", version="5.0", lanes=16, bandwidth_gbps=504.12307692307695), _port("ssd", "PCIe", "root", version="5.0", lanes=4, bandwidth_gbps=126.03076923076924)), role="pcie_host", model="GDS-capable host"),
        _component("gpu0", "gpu", (_port("host", "PCIe", "endpoint", version="5.0", lanes=16, bandwidth_gbps=504.12307692307695),), peak_ops_per_s=989_500_000_000_000.0, role="gds_accelerator", model="Hopper-class GDS-capable GPU analysis node", peak_ops_basis=_peak_ops_basis("BF16_tensor", source_basis="analytical H100-class compute default; GDS source does not specify compute throughput")),
        _component("nvme0", "high_io_ssd", (_port("host", "PCIe", "endpoint", version="5.0", lanes=4, bandwidth_gbps=126.03076923076924),), capacity_bytes=_gb(7680), read_bandwidth_gbps=112.0, write_bandwidth_gbps=56.8, role="nvme_storage", model="Samsung PM1743-class 7.68 TB NVMe SSD", physical_composition={"simulator_representation": "single_device_node", "simulator_node_count": 1, "physical_unit_kind": "NVMe_SSD", "physical_unit_count": 1, "physical_unit_count_status": "preset_scope"}, parameter_basis={"capacity_bytes": "vendor_product_decimal_capacity", "read_bandwidth_gbps": "vendor_sequential_read_14_GB_per_s", "write_bandwidth_gbps": "vendor_sequential_write_7.1_GB_per_s", "interface_bandwidth_gbps": "PCIe_5_x4_decoded_PHY_32_GTps_128b130b"}, extra_metadata={"read_latency_ns": 25_000.0, "write_latency_ns": 40_000.0, "transfer_granularity_bytes": 4096, "max_outstanding_requests": 256, "dma_bandwidth_gbps": 112.0, "dma_latency_ns": 1_200.0, "dma_energy_pj_per_byte": 0.0, "unknown_value_sentinels": {"dma_energy_pj_per_byte": "0.0 means unknown/not declared, not zero energy"}, "storage_transport_parameter_basis": "editable analytical GDS controller defaults; vendor evidence covers capacity and sequential media bandwidth"}),
    )
    links = (
        _link("gpu_pcie", "host0", "gpu", "gpu0", "host", "PCIe", version="5.0", lanes=16, bandwidth_gbps=504.12307692307695, latency_ns=150.0),
        _link("ssd_pcie", "host0", "ssd", "nvme0", "host", "PCIe", version="5.0", lanes=4, bandwidth_gbps=126.03076923076924, latency_ns=200.0, metadata={"bandwidth_basis": "decoded_phy_line_rate", "decoded_phy_capacity_gbps": 126.03076923076924, "device_read_envelope_gbps": 112.0, "media_bandwidth_modeled_separately": True}),
    )
    groups = (_group("compute", "GDS compute", ("host0", "gpu0"), "gpu0"), _group("storage", "NVMe storage", ("nvme0",), "nvme0"))
    limitations = ("拓扑表达 GPU、NVMe 与同一 PCIe 主机域的 GDS 数据路径边界；max_outstanding_requests 仅重叠启动延迟，不模拟文件系统、完整 DMA/NVMe 队列调度或软件栈开销。", "SSD 性能和 PCIe 负载效率是可覆盖参数，不保证端到端吞吐等于链路峰值。")
    return _definition("gpu-nvme-gds", "GPU + NVMe GPUDirect Storage", "NVIDIA", "GPUDirect Storage", "storage_offload", "single_host", EXACT_PUBLIC_TOPOLOGY, components, links, groups, {"host0": {"x": 280.0, "y": 40.0}, "gpu0": {"x": 0.0, "y": 300.0}, "nvme0": {"x": 560.0, "y": 300.0}}, (NVIDIA_GDS, SAMSUNG_PM1743, SAMSUNG_PM1743_WHITE_PAPER), limitations, "GDS 所需硬件逻辑关系的硬件-only 模板。", ("gpu", "nvme", "gds", "pcie"))


def _gpu_hbf() -> ArchitecturePresetDefinition:
    pkg = "hbf_package0"
    components = (
        _component("gpu0", "gpu", (_port("hbf", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming"),), package_id=pkg, die_id="gpu_die", peak_ops_per_s=989_500_000_000_000.0, role="host_accelerator", model="parameterized GPU"),
        _component("hbf0", "hbf", (_port("host", "UCIe", "endpoint", version="2.0", lanes=64, bandwidth_gbps=2048.0, payload="streaming", metadata={"media_bandwidth_modeled_separately": True}),), package_id=pkg, die_id="hbf_die", capacity_bytes=_gb(512), read_bandwidth_gbps=24000.0, write_bandwidth_gbps=0.0, role="near_package_flash", model="OCP HBF analysis device", physical_composition={"simulator_representation": "single_physical_unit_node", "simulator_node_count": 1, "physical_unit_kind": "HBF_stack", "physical_unit_count": 1, "physical_unit_count_status": "explicit_reference_node", "known_multiple": False, "unit_index": 0, "unit_count_in_product": 1, "unit_count_status": "experimental_reference_scope", "unit_count_formula": "one HBF reference stack in this experimental preset", "unit_capacity_bytes": _gb(512), "product_total_capacity_bytes": _gb(512), "source_basis": "OCP HBF preproduction up-to envelope"}, parameter_basis={"capacity_bytes": "up_to_512_GB_reference_capacity", "read_bandwidth_gbps": "grade3_up_to_3_TB_per_s_converted_to_24000_Gb_per_s", "write_bandwidth_gbps": "unknown_kept_zero", "interface_bandwidth_gbps": "analytical_UCIe_x64_32_GTps_envelope"}, extra_metadata={"capacity_scope": "up_to_512GB", "bandwidth_scope": "grade3_up_to_3TB_per_s", "read_latency_ns": 2500.0, "write_latency_ns": 0.0, "transfer_granularity_bytes": 4096, "max_outstanding_requests": 32, "dma_bandwidth_gbps": 2048.0, "dma_latency_ns": 800.0, "dma_energy_pj_per_byte": 0.0, "unknown_value_sentinels": {"write_bandwidth_gbps": "0.0 means unknown/not declared, not physical zero", "write_latency_ns": "0.0 means unknown/not declared, not zero latency", "dma_energy_pj_per_byte": "0.0 means unknown/not declared, not zero energy"}, "storage_transport_parameter_basis": "editable analytical HBF controller defaults bounded by the declared UCIe path", "internal_nand_composition": {"dies_per_stack": "8-high_or_16-high", "correlation_status": "not_reliably_disclosed"}}),
    )
    links = (_link("ucie0", "gpu0", "hbf", "hbf0", "host", "UCIe", version="2.0", lanes=64, bandwidth_gbps=2048.0, latency_ns=30.0, payload="streaming"),)
    groups = (_group("package0", "GPU and High Bandwidth Flash", ("gpu0", "hbf0"), "gpu0"),)
    limitations = ("HBF 仍是新兴生态参考；容量、读带宽、控制器和软件栈必须按目标实现校准。", "UCIe 链路与 HBF 媒体峰值分开建模，端到端吞吐受较小者和协议开销限制。", "写带宽没有可靠通用公开值，因此保持 0。")
    return _definition("gpu-hbf", "GPU + High Bandwidth Flash", "Open Compute Project", "HBF", "storage_offload", "near_package", EXPERIMENTAL_REFERENCE, components, links, groups, {"gpu0": {"x": 80.0, "y": 160.0}, "hbf0": {"x": 440.0, "y": 160.0}}, (OCP_HBF, UCIE_SPEC), limitations, "面向权重与冷数据近封装读取的实验参考。", ("gpu", "hbf", "ucie", "flash"))



def _soc_2x_dram_sram_cim(*, shared_phy_noc: bool = False) -> ArchitecturePresetDefinition:
    """User-authored 3D topology, not a calibrated SoC or a UCIe proxy.

    Default rates/capacities are replaceable analytical sweep coordinates.
    The upper DRAM has a dedicated through-stack path, not a forwarding DRAM.
    An auxiliary host supplies the existing V4 CPU orchestration contract.
    """

    preset_id = "soc-2x-dram-sram-cim" + ("-shared-phy-noc" if shared_phy_noc else "")
    name = "SoC + 2× stacked DRAM + SRAM-CIM" + (" (shared PHY/NoC)" if shared_phy_noc else " (independent controllers)")
    package, stack, domain = "soc_package0", "stack0", "stack0.thermal"
    bandwidth = 2048.0  # Analytical 256 GB/s one-way interface, not a device specification.
    assumptions = {
        "value_status": "uncalibrated_analytical_scaffold",
        "calibrated": False,
        "parameter_basis": {
            "capacity_bytes": "user-replaceable analytical capacity, not measured silicon",
            "bandwidth_gbps": "user-replaceable one-way analytical operating point",
            "latency_ns": "user-replaceable analytical service time; not a measured timing",
            "cost_profile_id": "reference registry scaffold; replace with domain-specific profiles before prediction",
        },
    }
    stack_metadata = {**assumptions, "stack_id": stack, "thermal_domain_id": domain}
    components = [
        _component(
            "soc0", "gpu",
            tuple(_port("dram{}".format(i), "TSV", "controller", bandwidth_gbps=bandwidth) for i in range(2))
            + (_port("cim", "NoC", "endpoint", bandwidth_gbps=bandwidth),
               _port("host", "PCIe", "endpoint", version="5.0", lanes=16, bandwidth_gbps=504.0, payload="coherent_dma")),
            package_id=package, die_id="soc_die0", peak_ops_per_s=512e12,
            role="analytical_soc_compute_proxy", model="GPU cost-model proxy for uncalibrated SoC compute",
            approximation="kind=gpu 仅复用已有执行器；不代表真实 SoC kernel 映射或已校准性能。",
            extra_metadata={**stack_metadata, "stack_layer": 0, "performance_model": "analytical_gpu_proxy",
                            "kernel_mapping_status": "unknown_requires_measurement"},
        ),
    ]
    links = []
    owners = {}
    for index in range(2):
        dram_id, link_id = "dram{}".format(index), "vertical_dram{}".format(index)
        controller = dram_id + ".controller"
        components.append(_component(
            dram_id, "dram", (_port("soc", "TSV", "device", bandwidth_gbps=bandwidth),),
            package_id=package, die_id=dram_id + "_die", capacity_bytes=32 * 1024**3,
            read_bandwidth_gbps=bandwidth, write_bandwidth_gbps=bandwidth,
            role="active_memory", model="Analytical stacked DRAM die",
            extra_metadata={**stack_metadata, "stack_layer": index + 1, "memory_service_owner": controller,
                            "controller_id": controller, "vertical_link_id": link_id,
                            "resident_access_path": "topology",
                            "read_latency_ns": 60.0, "write_latency_ns": 60.0,
                            "profile_resource_id_required": controller},
        ))
        phy_owner = "stack0.shared_phy_noc" if shared_phy_noc else "stack0.phy{}".format(index)
        links.append(_link(
            link_id, "soc0", dram_id, dram_id, "soc", "TSV",
            bandwidth_gbps=bandwidth, latency_ns=10.0,
            approximation="分析用垂直通路；不是 UCIe，也不表示实测 bump/TSV 布线。",
            metadata={**assumptions, "vertical_link": True, "stack_id": stack,
                      "thermal_domain_id": domain, "physical_technology": "TSV",
                      "read_path": [dram_id, "soc0"], "write_path": ["soc0", dram_id],
                      "pass_through_layers": list(range(1, index + 1)),
                      "forwarding_model": "dedicated_through_stack_no_dram_forwarding",
                      "shared_bidirectional": True, "physical_resource_owner": phy_owner},
        ))
        owners["link." + link_id] = phy_owner
    components.extend((
        _component(
            "cim0", "digital_sram_cim", (_port("soc", "NoC", "endpoint", bandwidth_gbps=bandwidth),),
            package_id=package, die_id="soc_die0", capacity_bytes=512 * 1024**2,
            peak_ops_per_s=512e12, read_bandwidth_gbps=bandwidth, write_bandwidth_gbps=bandwidth,
            role="on_die_sram_cim", model="Uncalibrated digital SRAM-CIM reference tile",
            extra_metadata={**stack_metadata, "stack_layer": 0, "memory_service_owner": "cim0.sram",
                            "weight_residency_policy": "placement_and_capacity_must_be_declared"},
        ),
        _component(
            "cpu0", "cpu",
            (_port("soc", "PCIe", "root", version="5.0", lanes=16, bandwidth_gbps=504.0, payload="coherent_dma"),
             _port("memory", "DDR", "controller", bandwidth_gbps=3276.8)),
            package_id="host_package0", die_id="host_cpu_die0", role="reference_host_orchestration",
            model="Auxiliary reference host, outside the 3D stack", extra_metadata=assumptions,
        ),
        _component(
            "host_memory0", "host_memory", (_port("cpu", "DDR", "device", bandwidth_gbps=3276.8),),
            package_id="host_memory_package0", die_id="host_memory_die0", capacity_bytes=64 * 1024**3,
            read_bandwidth_gbps=3276.8, write_bandwidth_gbps=3276.8,
            role="reference_host_memory", model="Auxiliary host memory, not a third stacked DRAM",
            extra_metadata={**assumptions, "memory_service_owner": "host_memory0.controller"},
        ),
    ))
    noc_owner = "stack0.shared_phy_noc" if shared_phy_noc else "soc0.noc"
    owners["link.soc_cim_noc"] = noc_owner
    links.extend((
        _link("soc_cim_noc", "soc0", "cim", "cim0", "soc", "NoC",
              bandwidth_gbps=bandwidth, latency_ns=2.0,
              metadata={**assumptions, "on_die": True, "thermal_domain_id": domain,
                        "shared_bidirectional": True, "physical_resource_owner": noc_owner}),
        _link("host_soc", "cpu0", "soc", "soc0", "host", "PCIe", version="5.0", lanes=16,
              bandwidth_gbps=504.0, latency_ns=800.0, payload="coherent_dma", metadata=assumptions),
        _link("host_memory", "cpu0", "memory", "host_memory0", "cpu", "DDR",
              bandwidth_gbps=3276.8, latency_ns=60.0, metadata={**assumptions, "shared_bidirectional": True}),
    ))
    limitations = (
        "用户指定结构的分析预设；容量、带宽、延迟与参考 profile 都不是实测或校准 SoC 数据。",
        "soc0 的 kind=gpu 是执行模型代理；真实 SoC kernel 支持、精度、缓存和算力需另行测量。",
        "dram1 通过独立穿层 TSV 通路接到 SoC，不经 dram0 存储控制器转发；不宣称所有 3D 封装都如此。",
        "两片 DRAM 控制器独立；共享版本将垂直通路与 CIM NoC 串行化到同一分析用 PHY/NoC owner，不自动增加峰值带宽。",
        "必须为两片 DRAM 分配独立 profile/resource_id；legacy-host-memory 只是载入 scaffold，不能当作两片器件的校准值。",
        "cpu0/host_memory0 是栈外辅助主控参考；热参数默认关闭，仅支持显式给定的静态热工作点降额，不模拟动态温升。",
    )
    result = _definition(
        preset_id, name, "User-defined", "3D DRAM + SRAM-CIM", "stacked_dram_soc", "single_package_with_reference_host",
        EXPERIMENTAL_REFERENCE, components, links,
        (_group("stack", "3D stack: SoC + DRAM", ("soc0", "dram0", "dram1", "cim0"), "soc0"),
         _group("host", "Reference host (outside stack)", ("cpu0", "host_memory0"), "cpu0")),
        {"soc0": {"x": 400.0, "y": 400.0}, "cim0": {"x": 700.0, "y": 400.0},
         "dram0": {"x": 400.0, "y": 220.0}, "dram1": {"x": 400.0, "y": 40.0},
         "cpu0": {"x": 0.0, "y": 400.0}, "host_memory0": {"x": 0.0, "y": 180.0}},
        (), limitations, "未校准的、可审计的 3D DRAM / SRAM-CIM 参数扫描起点。",
        ("3d", "dram", "sram-cim", "noc", "tsv", "analytical", "shared-phy" if shared_phy_noc else "independent-phy"),
    )
    metadata = {**result.hardware.metadata,
                "physical_resource_owners": owners,
                "controller_mode": "independent",
                "phy_noc_mode": "shared" if shared_phy_noc else "independent",
                "thermal_operating_point": {"enabled": False, "mode": "static_derating_only", "calibrated": False},
                "topology_evidence": {**result.hardware.metadata["topology_evidence"],
                                      "scope": "user_authored_analytical_topology",
                                      "source_scope": "user design brief only; no vendor or calibration evidence"}}
    return replace(result, hardware=replace(result.hardware, metadata=metadata))


_PRESETS: Tuple[ArchitecturePresetDefinition, ...] = (
    _hbm_accelerator_cluster(preset_id="nvidia-h100-sxm-8-nvswitch", name="8× NVIDIA H100 SXM + NVSwitch", gpu_model="H100 SXM", memory_kind="HBM3", memory_capacity_gb=80, memory_bandwidth_gbps=26800.0, peak_ops_per_s=989_500_000_000_000.0, count=8, fabric_protocol="NVLink", fabric_version="4.0", fabric_bandwidth_gbps=3600.0, fabric_lanes=18, vendor="NVIDIA", family="Hopper", sources=(NVIDIA_HOPPER,), support_level=ANALYTICAL_APPROXIMATION, memory_physical_unit_count=5, memory_count_status="vendor_documented_active_stacks", memory_component_preset_id="hbm3-16gb-0_670tbs-h100-slice"),
    _hbm_accelerator_cluster(preset_id="nvidia-h200-sxm-8-nvswitch", name="8× NVIDIA H200 SXM + NVSwitch", gpu_model="H200 SXM", memory_kind="HBM3E", memory_capacity_gb=141, memory_bandwidth_gbps=38400.0, peak_ops_per_s=989_500_000_000_000.0, count=8, fabric_protocol="NVLink", fabric_version="4.0", fabric_bandwidth_gbps=3600.0, fabric_lanes=18, vendor="NVIDIA", family="Hopper", sources=(NVIDIA_H200,), support_level=ANALYTICAL_APPROXIMATION, memory_physical_unit_count=6, memory_count_status="derived_from_product_total_and_24GB_stack_class", memory_raw_capacity_gb=144.0, memory_count_formula="144 GB raw product capacity / 24 GB HBM3E stack class = 6; 141 GB is product-visible capacity", memory_source_basis="NVIDIA H200 141 GB visible aggregate plus six 24 GB raw HBM3E stack-class derivation", memory_component_preset_id="hbm3e-24gb-0_800tbs-h200-slice"),
    _gh200_superchip(),
    _gh200_superchip(
        preset_id="nvidia-gh200-superchip-144gb-hbm3e",
        name="NVIDIA GH200 Grace Hopper Superchip (144GB HBM3E variant)",
        hbm_generation="HBM3E",
        hbm_capacity_gb=144.0,
        hbm_bandwidth_gbps=40000.0,
        hbm_unit_count_status="derived_from_product_total_and_24GB_stack_class",
        hbm_unit_count_formula="144 GB product total / 24 GB HBM3E stack class = 6",
        hbm_source_basis="GH200 144 GB product total plus 24 GB HBM3E stack class; count is derived",
        hbm_component_preset_id="hbm3e-24gb-0_833tbs-gh200-slice",
    ),
    _gh200_nvl2(),
    _gh200_nvl2(
        preset_id="nvidia-gh200-nvl2-96gb-hbm3",
        name="NVIDIA GH200 NVL2 (2×96GB HBM3 variant)",
        hbm_generation="HBM3",
        hbm_capacity_gb_per_superchip=96.0,
        hbm_bandwidth_gbps_per_superchip=32000.0,
        hbm_unit_count_status="derived_from_GH200_total_plus_full_GH100_6_stack_implementation",
        hbm_unit_count_formula="96 GB per-Superchip total / 16 GB HBM3 stack class = 6",
        hbm_source_basis="GH200 product total plus full-H100 six-stack implementation; count is derived",
        hbm_component_preset_id="hbm3-16gb-0_667tbs-product-slice",
    ),
    _gb200_system(4, "nvidia-gb200-nvl4", "NVIDIA GB200 NVL4", "four_gpu"),
    _gb200_system(72, "nvidia-gb200-nvl72", "NVIDIA GB200 NVL72", "rack_scale"),
    _mi300a(),
    _mi300x_single(),
    _hbm_accelerator_cluster(preset_id="amd-mi300x-8-infinity-fabric", name="8× AMD Instinct MI300X + Infinity Fabric", gpu_model="MI300X", memory_kind="HBM3", memory_capacity_gb=192, memory_bandwidth_gbps=42600.0, peak_ops_per_s=1_307_400_000_000_000.0, count=8, fabric_protocol="InfinityFabric", fabric_version="MI300X", fabric_bandwidth_gbps=3584.0, fabric_lanes=8, vendor="AMD", family="MI300", sources=(AMD_MI300X,), support_level=ANALYTICAL_APPROXIMATION, memory_physical_unit_count=8, memory_count_status="vendor_documented_stacks", peak_precision="BF16_matrix", memory_component_preset_id="hbm3-24gb-0_665625tbs-mi300x-slice"),
    _hbm_accelerator_cluster(preset_id="intel-gaudi3-8-roce", name="8× Intel Gaudi 3 RoCE Reference Baseboard", gpu_model="Gaudi 3", memory_kind="HBM2E", memory_capacity_gb=128, memory_bandwidth_gbps=29600.0, peak_ops_per_s=1_835_000_000_000_000.0, count=8, fabric_protocol="RoCE", fabric_version="v2", fabric_bandwidth_gbps=1600.0, fabric_lanes=8, vendor="Intel", family="Gaudi", sources=(INTEL_GAUDI3,), support_level=ANALYTICAL_APPROXIMATION, fabric_kind="ethernet_switch", memory_physical_unit_count=8, memory_count_status="vendor_documented_stacks", peak_precision="BF16_MME", fabric_member_rate_gbps=200.0, fabric_total_physical_port_count=24, memory_component_preset_id="hbm2e-16gb-0_4625tbs-gaudi3-slice"),
    _cxl_type3(),
    _cxl_pool(),
    _ucie_chiplet(),
    _gpu_cim(),
    _hbm_pim(),
    _gds(),
    _gpu_hbf(),
    _soc_2x_dram_sram_cim(),
    _soc_2x_dram_sram_cim(shared_phy_noc=True),
)


_BY_ID: Mapping[str, ArchitecturePresetDefinition] = {
    item.preset_id: item for item in _PRESETS
}
if len(_BY_ID) != len(_PRESETS):
    raise RuntimeError("架构拓扑预设 ID 重复")
if any(item.support_level not in SUPPORT_LEVELS for item in _PRESETS):
    raise RuntimeError("架构拓扑预设使用了未知 support_level")
for _item in _PRESETS:
    _report = validate_topology(_item.hardware)
    if _item.loadable and not _report.is_valid:
        raise RuntimeError(
            "架构拓扑预设 {} 无法载入：{}".format(
                _item.preset_id, _report.format_en()
            )
        )


def _compatibility(item: ArchitecturePresetDefinition) -> Dict[str, Any]:
    planner_structure = _planner_structure(item.hardware.components)
    gpu_count = planner_structure["planner_gpu_count"]
    compute_count = sum(
        component.kind in {"gpu", "generic_accelerator", "digital_sram_cim", "pim_accelerator"}
        for component in item.hardware.components
    )
    if item.topology_class == "accelerator_fabric":
        model_classes = ["dense_transformer", "moe_transformer", "hybrid_transformer"]
        parallelism = ["tp", "pp", "ep_for_moe"]
        storage_roles = ["accelerator_memory"]
    elif item.topology_class in {"coherent_superchip", "coherent_apu"}:
        model_classes = ["dense_transformer", "hybrid_transformer", "long_context"]
        parallelism = ["single_rank"] + (["tp", "pp"] if compute_count > 1 else [])
        storage_roles = ["coherent_host_memory", "accelerator_memory"]
    elif item.topology_class == "disaggregated_memory":
        model_classes = ["memory_capacity_bound", "long_context"]
        parallelism = ["single_rank", "host_partitioning"]
        storage_roles = ["cxl_memory_expansion", "weight_capacity", "kv_cache_capacity"]
    elif item.topology_class == "stacked_dram_soc":
        model_classes = ["dense_transformer", "hybrid_transformer", "long_context", "operator_offload"]
        parallelism = ["single_rank", "operator_offload"]
        storage_roles = ["active_memory", "resident_weights", "kv_cache_capacity", "linear_state"]
    elif item.topology_class == "near_memory_compute":
        model_classes = ["mlp_heavy", "moe_transformer", "operator_offload"]
        parallelism = ["single_rank", "operator_offload", "ep_for_moe"]
        storage_roles = ["resident_weights", "active_memory"]
    elif item.topology_class == "storage_offload":
        model_classes = ["memory_capacity_bound", "long_context", "weight_offload"]
        parallelism = ["single_rank", "storage_offload"]
        storage_roles = ["weight_offload", "cold_tensor_storage", "kv_cache_offload"]
    elif item.topology_class == "chiplet_package":
        model_classes = ["dense_transformer", "hybrid_transformer", "operator_offload"]
        parallelism = ["single_rank", "operator_offload"]
        storage_roles = []
    else:
        model_classes = ["dense_transformer"]
        parallelism = ["single_rank"]
        storage_roles = ["accelerator_memory"]
    constraints = [
        "hardware_only",
        "placement_must_be_regenerated",
        "protocol_reachability_required",
        "capacity_must_fit_selected_mapping",
    ]
    if item.topology_class == "stacked_dram_soc":
        constraints.extend(("uncalibrated_soc_gpu_proxy", "per_dram_controller_profile_binding_required", "thermal_disabled_without_explicit_operating_point"))
    if "ep_for_moe" in parallelism:
        constraints.append("ep_requires_moe")
    if gpu_count:
        constraints.append(
            "world_size_must_not_exceed_{}_planner_gpu_endpoints".format(
                gpu_count
            )
        )
    requires_gpu_attachment = planner_structure["requires_gpu_attachment"]
    requires_cpu_attachment = planner_structure["requires_cpu_attachment"]
    if requires_gpu_attachment:
        constraints.append("gpu_attachment_required_before_reference_planner")
    if requires_cpu_attachment:
        constraints.append("cpu_attachment_required_for_host_orchestration")
    constraints.extend(
        (
            "hardware_cost_profiles_must_be_regenerated_and_reviewed",
            "single_logical_host_orchestration_sequence_per_realized_cohort",
        )
    )
    return {
        "schema_version": "1.1",
        **planner_structure,
        "logical_compute_endpoint_count": compute_count,
        "planner_requirement_basis": (
            "reference planner executes ranks only on kind=gpu; V4 host "
            "orchestration also requires real kind=cpu and kind=gpu component "
            "references"
        ),
        "host_orchestration_scope": (
            "single_aggregate_prepare_pack_dma_sequence_per_realized_cohort_"
            "with_physical_invocation_counted_driver_and_gpu_command_work"
        ),
        "recommended_model_classes": model_classes,
        "recommended_parallelism": parallelism,
        "supported_storage_roles": storage_roles,
        "feature_tags": sorted(set(item.tags) | {protocol.lower() for protocol in item.protocols}),
        "constraints": constraints,
    }


def _metadata(item: ArchitecturePresetDefinition) -> Dict[str, Any]:
    total_capacity = sum(component.capacity_bytes for component in item.hardware.components)
    hardware_metadata = item.hardware.metadata
    return {
        "id": item.preset_id,
        "name": item.name,
        "vendor": item.vendor,
        "family": item.family,
        "topology_class": item.topology_class,
        "scale": item.scale,
        "support_level": item.support_level,
        "evidence_level": item.support_level,
        "topology_evidence": dict(hardware_metadata["topology_evidence"]),
        "parameter_basis": dict(hardware_metadata["parameter_basis"]),
        "loadable": item.loadable,
        "hardware_only": True,
        "component_count": len(item.hardware.components),
        "link_count": len(item.hardware.links),
        "group_count": len(item.groups),
        "capacity_bytes": total_capacity,
        "capability_units": dict(COMPONENT_CAPABILITY_UNITS),
        "capacity_display_value": _capacity_display(total_capacity, aggregate=True),
        "capacity_source_basis": (
            "底层字节值可能组合十进制产品规格与二进制分析参数；界面容量统一换算为 IEC 单位。"
            if total_capacity
            else ""
        ),
        "protocols": list(item.protocols),
        "component_kinds": sorted({component.kind for component in item.hardware.components}),
        "sources": [source.to_metadata() for source in item.sources],
        "limitations": list(item.limitations),
        "notes": item.notes,
        "tags": list(item.tags),
        "compatibility": _compatibility(item),
        "catalog_version": CATALOG_VERSION,
    }


def list_architecture_presets() -> Tuple[Dict[str, Any], ...]:
    """Return stable compact catalog rows without the full hardware graph."""

    return tuple(_metadata(item) for item in sorted(_PRESETS, key=lambda value: value.preset_id))


def architecture_preset_filters() -> Dict[str, Sequence[Any]]:
    items = list_architecture_presets()
    return {
        "vendor": sorted({item["vendor"] for item in items}),
        "family": sorted({item["family"] for item in items}),
        "topology_class": sorted({item["topology_class"] for item in items}),
        "scale": sorted({item["scale"] for item in items}),
        "support_level": sorted({item["support_level"] for item in items}),
        "protocol": sorted({protocol for item in items for protocol in item["protocols"]}),
        "tag": sorted({tag for item in items for tag in item["tags"]}),
        "loadable": [True, False],
    }


def architecture_preset_page(
    *,
    query: str = "",
    vendor: str = "",
    family: str = "",
    topology_class: str = "",
    scale: str = "",
    support_level: str = "",
    protocol: str = "",
    tag: str = "",
    loadable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return a filtered list envelope suitable for UI/API integration."""

    normalized = {
        "query": query.strip().lower(),
        "vendor": vendor.strip().lower(),
        "family": family.strip().lower(),
        "topology_class": topology_class.strip().lower(),
        "scale": scale.strip().lower(),
        "support_level": support_level.strip().lower(),
        "protocol": protocol.strip().lower(),
        "tag": tag.strip().lower(),
    }
    items = []
    for item in list_architecture_presets():
        if loadable is not None and item["loadable"] is not loadable:
            continue
        if any(
            normalized[field] and item[field].lower() != normalized[field]
            for field in ("vendor", "family", "topology_class", "scale", "support_level")
        ):
            continue
        if normalized["protocol"] and normalized["protocol"] not in {value.lower() for value in item["protocols"]}:
            continue
        if normalized["tag"] and normalized["tag"] not in {value.lower() for value in item["tags"]}:
            continue
        haystack = " ".join((item["id"], item["name"], item["vendor"], item["family"], item["topology_class"], item["notes"], " ".join(item["protocols"]), " ".join(item["tags"]))).lower()
        if normalized["query"] and normalized["query"] not in haystack:
            continue
        items.append(item)
    return {
        "items": items,
        "total": len(items),
        "filters": architecture_preset_filters(),
        "query": {"query": query, "vendor": vendor, "family": family, "topology_class": topology_class, "scale": scale, "support_level": support_level, "protocol": protocol, "tag": tag, "loadable": loadable},
        "catalog": {"version": CATALOG_VERSION, "cutoff_at": CATALOG_CUTOFF_AT},
        "replacement_policy": dict(REPLACEMENT_POLICY),
    }


def get_architecture_preset(preset_id: str) -> ArchitecturePresetDefinition:
    return _BY_ID[preset_id]


def materialize_architecture_payload(preset_id: str) -> Dict[str, Any]:
    """Return a new JSON-compatible complete ``HardwareSpec`` payload."""

    item = get_architecture_preset(preset_id)
    if not item.loadable:
        raise ValueError(
            "架构拓扑预设 {} 当前不可载入：{}".format(
                preset_id, "；".join(item.limitations)
            )
        )
    return to_primitive(item.hardware)


def architecture_preset_detail(preset_id: str) -> Dict[str, Any]:
    item = get_architecture_preset(preset_id)
    hardware = materialize_architecture_payload(preset_id) if item.loadable else None
    topology_view = hardware["metadata"]["topology_view"] if hardware else None
    return {
        "preset": _metadata(item),
        "hardware": hardware,
        "components": hardware["components"] if hardware else [],
        "links": hardware["links"] if hardware else [],
        "groups": topology_view["groups"] if topology_view else [],
        "topology_view": topology_view,
        "compatibility": _compatibility(item),
        "replacement_policy": dict(REPLACEMENT_POLICY),
        "catalog": {"version": CATALOG_VERSION, "cutoff_at": CATALOG_CUTOFF_AT},
    }


__all__ = [
    "ANALYTICAL_APPROXIMATION",
    "CATALOG_CUTOFF_AT",
    "CATALOG_VERSION",
    "EXACT_PUBLIC_TOPOLOGY",
    "EXPERIMENTAL_REFERENCE",
    "REPLACEMENT_POLICY",
    "SUPPORT_LEVELS",
    "ArchitecturePresetDefinition",
    "ArchitectureSource",
    "architecture_preset_detail",
    "architecture_preset_filters",
    "architecture_preset_page",
    "get_architecture_preset",
    "list_architecture_presets",
    "materialize_architecture_payload",
]
