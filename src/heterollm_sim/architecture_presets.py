"""Offline architecture-topology presets for :class:`HardwareSpec`.

Each entry is a complete, hardware-only graph composed from component preset
instances plus explicit links and layout. It replaces the current hardware
topology and never contains a model, workload, placement, or rank mapping.
Public logical relationships are kept distinct from analytical collapsed
fabrics so that a useful system model is not mistaken for undisclosed wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec
from .component_presets import (
    _cpu_cost_profile_template,
    _gpu_cost_profile_template,
    _hbm_cost_profile_template,
    _host_memory_cost_profile_template,
    materialize_component_payload,
)
from .physical_presets import make_analytical_dram_config, make_analytical_nand_config
from .serde import to_primitive
from .schema_v4 import ControllerProfile, GPUControllerProfile
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
    "gddr": "legacy-gddr",
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
        protocols = {link.protocol for link in self.hardware.links}
        # Keep UCIe discoverable as a physical-transport alias after HBF
        # links become the logical protocol used by the simulator.
        if "HBF" in protocols:
            protocols.add("UCIe")
        return tuple(sorted(protocols))

    @property
    def groups(self) -> Tuple[Mapping[str, Any], ...]:
        view = self.hardware.metadata.get("topology_view", {})
        return tuple(view.get("groups", ())) if isinstance(view, Mapping) else ()


def _source(title: str, url: str, publisher: str, source_kind: str) -> ArchitectureSource:
    return ArchitectureSource(title, url, publisher, source_kind)


NVIDIA_GH200 = _source(
    "NVIDIA Grace Hopper Superchip",
    "https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/",
    "NVIDIA",
    "vendor_product_specification",
)
NVIDIA_B200 = _source(
    "NVIDIA DGX B200 system specifications",
    "https://www.nvidia.com/en-us/data-center/dgx-b200/",
    "NVIDIA",
    "vendor_product_specification",
)
NVIDIA_BLACKWELL = _source(
    "NVIDIA Blackwell Architecture",
    "https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/",
    "NVIDIA",
    "vendor_architecture_reference",
)
UCIE_SPEC = _source(
    "Universal Chiplet Interconnect Express specifications",
    "https://www.uciexpress.org/specifications",
    "UCIe Consortium",
    "consortium_specification",
)
OCP_HBF = _source(
    "Open Compute Project High Bandwidth Flash",
    "https://www.opencompute.org/projects/high-bandwidth-flash",
    "Open Compute Project",
    "consortium_reference",
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
    channels_per_stack: int,
    prefix: str = "hbm",
) -> Tuple[PortSpec, ...]:
    """Create one controller port for every modeled physical HBM stack."""

    if type(channels_per_stack) is not int or channels_per_stack <= 0:
        raise ValueError("HBM channels per stack must be a positive integer")
    unit_bandwidth = product_total_bandwidth_gbps / stack_count
    return tuple(
        _port(
            "{}{}".format(prefix, index),
            "HBM",
            "controller",
            version=generation,
            lanes=channels_per_stack,
            bandwidth_gbps=unit_bandwidth,
            metadata={
                "physical_unit_kind": "HBM_stack",
                "unit_index": index,
                "unit_count_in_product": stack_count,
                "physical_channel_total": channels_per_stack,
                "channels_per_stack": channels_per_stack,
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
    channels_per_stack: int,
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
    if type(channels_per_stack) is not int or channels_per_stack <= 0:
        raise ValueError("HBM channels per stack must be a positive integer")
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
                        lanes=channels_per_stack,
                        bandwidth_gbps=unit_bandwidth,
                        metadata={
                            "physical_unit_kind": "HBM_stack",
                            "unit_index": index,
                            "unit_count_in_product": stack_count,
                            "physical_channel_total": channels_per_stack,
                            "channels_per_stack": channels_per_stack,
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
                lanes=channels_per_stack,
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
    bandwidth_gbps: float = 0.0,
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
    # This builder is used only by the bundled architecture definitions.  Its
    # synthetic arrays are explicit preset assumptions; catalog/user payloads
    # pass through _catalog_component and are never repaired implicitly here.
    if kind in {"hbm", "dram", "host_memory", "cxl_memory", "ssd", "high_io_ssd", "hbf"} and "physical_memory_config" not in component_metadata:
        read_gbps = read_bandwidth_gbps or bandwidth_gbps
        write_gbps = write_bandwidth_gbps or bandwidth_gbps or read_gbps
        if capacity_bytes <= 0 or read_gbps <= 0 or write_gbps <= 0:
            raise ValueError("{} preset storage needs capacity and directional bandwidth".format(component_id))
        version = ports[0].version if ports else ""
        protocol = ports[0].protocol.upper() if ports else ""
        if kind in {"ssd", "high_io_ssd", "hbf"}:
            component_metadata["physical_memory_config"] = make_analytical_nand_config(
                kind="HBF" if kind == "hbf" else "SSD",
                capacity_bytes=capacity_bytes,
                read_bandwidth_gbps=read_gbps,
                write_bandwidth_gbps=write_gbps,
                read_latency_ns=float(component_metadata.get("read_latency_ns", 50_000.0)),
                write_latency_ns=float(component_metadata.get("write_latency_ns", 800_000.0)),
                scope="architecture_preset_{}".format(component_id),
            )
        else:
            memory_kind = "HBM" if kind == "hbm" else "LPDDR" if "LPDDR" in protocol else "DDR"
            if kind == "hbm":
                # HBM PortSpec.lanes denotes channels, not DQ pins. A single
                # endpoint is the authority for this bundled component's
                # interface; multiple endpoints need an explicit Core config
                # to state whether their channels are shared or independent.
                hbm_ports = [port for port in ports
                             if port.protocol.upper() == "HBM"
                             and port.role.lower() in {"device", "endpoint"}]
                if len(hbm_ports) != 1:
                    raise ValueError(
                        "{} HBM preset requires exactly one HBM memory endpoint "
                        "or an explicit physical_memory_config".format(component_id)
                    )
                memory_channels = hbm_ports[0].lanes
                version = hbm_ports[0].version
            else:
                memory_channels = 8 if kind == "host_memory" else 1
            component_metadata["physical_memory_config"] = make_analytical_dram_config(
                kind=memory_kind, generation=version or memory_kind,
                capacity_bytes=capacity_bytes,
                read_bandwidth_gbps=read_gbps,
                write_bandwidth_gbps=write_gbps,
                channels=memory_channels,
                read_latency_ns=float(component_metadata.get("read_latency_ns", 35.0)),
                write_latency_ns=float(component_metadata.get("write_latency_ns", 35.0)),
                scope="architecture_preset_{}".format(component_id),
            )
    # Product-visible budgets are not raw binary DRAM densities. Keep the
    # selected reference geometry's raw capacity separate from product facts.
    if kind == "hbm" and "physical_composition" in component_metadata:
        geometry = component_metadata["physical_memory_config"].get("metadata", {})
        if geometry.get("geometry_status") == "standard_organization":
            composition = component_metadata["physical_composition"]
            composition["nominal_raw_capacity_budget_bytes"] = composition.get("unit_raw_capacity_bytes")
            composition["unit_raw_capacity_bytes"] = geometry["raw_capacity_bytes"]
            composition["product_total_raw_capacity_bytes"] = (
                geometry["raw_capacity_bytes"] * composition.get("unit_count_in_product", 1)
            )
            composition["raw_capacity_status"] = "selected_JEDEC_reference_not_verified_product_geometry"
            composition["capacity_accounting"] = "product_visible_budget_separate_from_selected_reference_raw_capacity"
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
        bandwidth_gbps=(bandwidth_gbps or max(read_bandwidth_gbps, write_bandwidth_gbps)),
        metadata=component_metadata,
    )


def _catalog_component(
    preset_id: str,
    component_id: str,
    *,
    package_id: str,
    die_id: str,
    ports: Optional[Sequence[PortSpec]] = None,
    metadata_updates: Optional[Mapping[str, Any]] = None,
    value_overrides: Optional[Mapping[str, Any]] = None,
    component_catalog=None,
) -> ComponentSpec:
    """Materialize an architecture node from the curated component catalog.

    Architecture code supplies only placement identity, endpoint ports and
    topology-specific derived values.  Vendor capabilities and profile
    metadata stay owned by the component preset, avoiding a second copy that
    can drift when a hardware preset is edited.
    """

    payload = (component_catalog.detail(preset_id)["component"] if component_catalog is not None
               else materialize_component_payload(preset_id))
    source_id = str(payload.get("component_id", ""))
    source_ports = tuple(PortSpec(**port) for port in payload.get("ports", ()))
    metadata = dict(payload.get("metadata", {}))
    if source_id and source_id != component_id:
        def remap(value):
            if isinstance(value, str):
                return value.replace(source_id, component_id)
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            return value
        metadata = remap(metadata)
    metadata["component_preset_id"] = preset_id
    metadata["component_preset_status"] = "catalog_reference"
    if metadata_updates:
        metadata.update(dict(metadata_updates))
    values = {
        "component_id": component_id,
        "package_id": package_id,
        "die_id": die_id,
        "ports": tuple(ports) if ports is not None else source_ports,
        "metadata": metadata,
    }
    values.update(dict(value_overrides or {}))
    base_payload = dict(payload)
    base_payload["ports"] = source_ports
    return replace(ComponentSpec(**base_payload), **values)


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
    # Keep the architecture as a composition of catalog components plus
    # links.  This index is intentionally derived from each component's
    # provenance metadata so a topology consumer can see which editable
    # hardware preset supplied every node without having to infer it from the
    # display name.  Auxiliary nodes (for example the reference CPU) may not
    # have a public preset and are marked accordingly by an empty value.
    component_preset_ids = {
        component.component_id: component.metadata.get("component_preset_id", "")
        for component in components
        if component.metadata.get("component_preset_id")
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
            "component_preset_ids": component_preset_ids,
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
    channels_per_stack: int,
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
                    channels_per_stack=channels_per_stack,
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
            channels_per_stack=channels_per_stack,
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
    channels_per_stack: int,
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
        ) + _hbm_controller_ports(hbm_stack_count, hbm_generation, hbm_bandwidth_gbps, channels_per_stack=channels_per_stack), package_id="superchip0", die_id="hopper_die", peak_ops_per_s=989_500_000_000_000.0, role="accelerator", model="Hopper GPU", peak_ops_basis=_peak_ops_basis("BF16_tensor", source_basis="vendor_product_specification")),
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
        channels_per_stack=channels_per_stack,
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
    channels_per_stack: int,
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
            _component(hopper, "gpu", (_port("cpu", "NVLink-C2C", "endpoint", version="GH200", bandwidth_gbps=3600.0),) + _hbm_controller_ports(hbm_stack_count, hbm_generation, hbm_bandwidth_gbps_per_superchip, channels_per_stack=channels_per_stack) + (_port("peer", "NVLink", "endpoint", version="4.0", bandwidth_gbps=3600.0),), package_id="superchip{}".format(index), die_id="hopper{}".format(index), peak_ops_per_s=989_500_000_000_000.0, role="accelerator", model="Hopper GPU", peak_ops_basis=_peak_ops_basis("BF16_tensor", source_basis="vendor_product_specification")),
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
            channels_per_stack=channels_per_stack,
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


def _b200_hbf_hbm(component_catalog=None) -> ArchitecturePresetDefinition:
    """B200 accelerator package plus a reference host for execution."""

    package = "b200_package0"
    def catalog_payload(preset_id):
        return (component_catalog.detail(preset_id)["component"] if component_catalog is not None
                else materialize_component_payload(preset_id))
    hbm_preset_id = "samsung-hbm3e-36gb-9_2"
    hbm_catalog_payload = catalog_payload(hbm_preset_id)
    hbm_unit_capacity_bytes = int(hbm_catalog_payload["capacity_bytes"])
    hbm_unit_bandwidth_gbps = float(hbm_catalog_payload["read_bandwidth_gbps"])
    hbm_total_capacity_bytes = hbm_unit_capacity_bytes * 2
    hbm_total_bandwidth_gbps = hbm_unit_bandwidth_gbps * 2
    pcie_gbps = 504.12307692307695  # PCIe 5.0 x16 decoded 128b/130b envelope

    # Compute and cache defaults are owned by the B200 component preset; the
    # architecture contributes only endpoint identity and links.
    hbm_ports = _hbm_controller_ports(2, "HBM3E", hbm_total_bandwidth_gbps, channels_per_stack=16)
    gpu_ports = hbm_ports + (
        _port("hbf0", "HBF", "endpoint", version="2.0", lanes=64, bandwidth_gbps=3_904.0, payload="streaming", metadata={"physical_transport_protocol": "UCIe"}),
        _port("hbf1", "HBF", "endpoint", version="2.0", lanes=64, bandwidth_gbps=3_904.0, payload="streaming", metadata={"physical_transport_protocol": "UCIe"}),
        _port("pcie0", "PCIe", "endpoint", version="5.0", lanes=16, bandwidth_gbps=pcie_gbps, payload="coherent_dma"),
    )
    components = [
        _catalog_component(
            "nvidia-b200-sxm-gpu", "gpu0", package_id=package, die_id="b200_die",
            component_catalog=component_catalog,
            ports=gpu_ports,
            metadata_updates={
                "architecture": "Blackwell", "memory_product": "HBM3E",
                "memory_total_bandwidth_tb_per_s": hbm_total_bandwidth_gbps / 8_000.0,
                "memory_total_capacity_gb": hbm_total_capacity_bytes / 1_000_000_000.0,
                "product_memory_reference": {"capacity_gb": 180.0, "bandwidth_tb_per_s": 8.0, "source": "NVIDIA DGX B200 aggregate / 8 GPUs"},
                "attached_memory_component_preset_id": hbm_preset_id,
                "hbm_node_split": "two equal analytical HBM endpoints; physical stack count intentionally unspecified",
                "provenance": {"value_status": "vendor_aggregate_derived_and_public_product_specification", "source_basis": "NVIDIA DGX B200 aggregate system specification and Blackwell Technical Overview Table 3"},
            },
        )
    ]

    hbf_physical = lambda index: {
        "simulator_representation": "single_reference_endpoint",
        "simulator_node_count": 1,
        "physical_unit_kind": "HBF_endpoint",
        "physical_unit_count": 1,
        "physical_unit_count_status": "explicit_reference_node",
        "known_multiple": False,
        "unit_index": index,
        "unit_count_in_product": 1,
        "unit_count_status": "experimental_reference_scope",
        "unit_count_formula": "one HBF reference endpoint per user-requested node",
        "unit_capacity_bytes": _gb(512.0),
        "product_total_capacity_bytes": _gb(512.0),
        "source_basis": "OCP HBF preproduction up-to envelope",
    }
    def hbf_component(index: int) -> ComponentSpec:
        hbf_payload = catalog_payload("sk-hynix-hbf-512gb")
        catalog_port = PortSpec(**hbf_payload["ports"][0])
        hbf_port_metadata = dict(catalog_port.metadata)
        hbf_port_metadata["media_bandwidth_modeled_separately"] = True
        hbf_port = replace(catalog_port, port_id="hbf0", protocol="HBF", bandwidth_gbps=3_904.0, payload="streaming", metadata=hbf_port_metadata)
        component = _catalog_component(
            "sk-hynix-hbf-512gb", "hbf{}".format(index),
            component_catalog=component_catalog,
            package_id=package, die_id="hbf{}_die".format(index),
            ports=(hbf_port,),
            metadata_updates={
                "architecture_role": "near_package_flash",
                "model": "SK hynix HBF analysis device {}".format(index),
                "physical_composition": {
                    **hbf_physical(index),
                    "unit_capacity_bytes": hbf_payload["capacity_bytes"],
                    "product_total_capacity_bytes": hbf_payload["capacity_bytes"],
                },
                "write_capability_status": "unknown" if hbf_payload["write_bandwidth_gbps"] <= 0 else "user_configured",
            },
        )
        # Each HBF endpoint owns an independent memory service resource.  The
        # catalog preset is reusable, but a two-endpoint architecture must not
        # make both components resolve to one profile/resource ID.
        metadata = dict(component.metadata)
        template = dict(metadata.get("cost_profile_template", {}))
        resource_id = "hbf{}.memory".format(index)
        template["resource_id"] = resource_id
        metadata["cost_profile_template"] = template
        metadata["memory_service_owner"] = resource_id
        return replace(
            component,
            cost_profile_id="sk_hynix_hbf_512gb_memory_{}".format(index),
            metadata=metadata,
        )
    components.extend(hbf_component(index) for index in range(2))

    # HBM endpoints are direct instances of the curated Samsung component
    # preset.  Architecture-specific information is limited to placement,
    # endpoint identity and the GPU↔HBM links; capacity, media bandwidth and
    # profile defaults remain owned by the component preset.
    hbm_components = []
    hbm_links = []
    for index in range(2):
        component_id = "hbm{}".format(index)
        source_port = PortSpec(**hbm_catalog_payload["ports"][0])
        hbm_port = replace(source_port, port_id="host")
        hbm_component = _catalog_component(
            hbm_preset_id,
            component_id,
            component_catalog=component_catalog,
            package_id=package,
            die_id="{}_die".format(component_id),
            ports=(hbm_port,),
            metadata_updates={
                "architecture_role": "local_accelerator_memory",
                "model": "Samsung HBM3E endpoint {} of 2".format(index + 1),
                "physical_composition": {
                    "simulator_representation": "catalog_component_instance",
                    "simulator_node_count": 1,
                    "physical_unit_kind": "HBM3E_stack_class",
                    "physical_unit_count": 1,
                    "physical_unit_count_status": "component_preset_instance",
                    "unit_index": index,
                    "unit_count_in_product": 2,
                    "component_preset_id": hbm_preset_id,
                    "source_basis": "Samsung HBM3E component preset",
                },
            },
        )
        hbm_components.append(hbm_component)
        hbm_links.append(_link(
            "gpu_hbm{}".format(index), "gpu0", "hbm{}".format(index),
            component_id, "host", "HBM", version="HBM3E", lanes=hbm_port.lanes,
            bandwidth_gbps=min(hbm_port.bandwidth_gbps, hbm_ports[index].bandwidth_gbps), latency_ns=40.0,
        ))
    components.extend(hbm_components)

    cpu_profile, cpu_basis = _cpu_cost_profile_template("cpu0", core_count=72, frequency_ghz=3.0)
    ddr_preset_id = "samsung-ddr5-32gb-udimm-5600"
    ddr_payload = catalog_payload(ddr_preset_id)
    ddr_source_port = PortSpec(**ddr_payload["ports"][0])
    ddr_port = replace(ddr_source_port, port_id="ddr0", role="device")
    hostmem_component = _catalog_component(
        ddr_preset_id,
        "hostmem0",
        component_catalog=component_catalog,
        package_id="host0",
        die_id="ddr_die",
        ports=(ddr_port,),
        metadata_updates={
            "auxiliary_host": True,
            "architecture_role": "auxiliary_host_memory",
            "model": "Samsung DDR5 32GB reference host memory",
        },
    )
    host_bandwidth_gbps = float(hostmem_component.read_bandwidth_gbps)
    host_read_latency_ns = float(hostmem_component.metadata.get("read_latency_ns", 60.0) or 60.0)
    host_write_latency_ns = float(hostmem_component.metadata.get("write_latency_ns", host_read_latency_ns) or host_read_latency_ns)
    host_profile, host_basis = _host_memory_cost_profile_template(
        "hostmem0", host_bandwidth_gbps,
        name="hostmem0-reference-memory-profile",
        read_latency_ns=host_read_latency_ns,
        write_latency_ns=host_write_latency_ns,
    )
    components.extend((
        _component(
            "cpu0", "cpu",
            (_port("pcie0", "PCIe", "root", version="5.0", lanes=16, bandwidth_gbps=pcie_gbps, payload="coherent_dma"), _port("ddr0", "DDR5", "controller", version="DDR5-5600", lanes=ddr_port.lanes, bandwidth_gbps=host_bandwidth_gbps)),
            package_id="host0", die_id="cpu_die", role="auxiliary_host_cpu", model="Reference Grace-class host CPU",
            extra_metadata={"auxiliary_host": True, "cost_profile_key": "cpu", "cost_profile_template": cpu_profile, "cost_profile_parameter_basis": cpu_basis, "cpu_profile_required": True},
        ),
        hostmem_component,
    ))
    links = list(hbm_links)
    for index in range(2):
        hbf_endpoint = next(component for component in components if component.component_id == "hbf{}".format(index)).ports[0]
        links.append(_link("gpu_hbf{}".format(index), "gpu0", "hbf{}".format(index), "hbf{}".format(index), "hbf0", "HBF", version="2.0", lanes=min(64, hbf_endpoint.lanes), bandwidth_gbps=min(3_904.0, hbf_endpoint.bandwidth_gbps), latency_ns=4_000.0, payload="streaming", metadata={"physical_transport_protocol": "UCIe"}))
    links.extend((
        _link("cpu_gpu_pcie", "cpu0", "pcie0", "gpu0", "pcie0", "PCIe", version="5.0", lanes=16, bandwidth_gbps=pcie_gbps, latency_ns=800.0, payload="coherent_dma"),
        _link("cpu_hostmem_ddr", "cpu0", "ddr0", "hostmem0", "ddr0", "DDR5", version="DDR5-5600", lanes=ddr_port.lanes, bandwidth_gbps=host_bandwidth_gbps, latency_ns=80.0),
    ))
    groups = (
        _group("b200_package", "NVIDIA B200 with 2×HBM3E + 2×HBF", ("gpu0", "hbm0", "hbm1", "hbf0", "hbf1"), "gpu0"),
        _group("reference_host", "Auxiliary host CPU and memory", ("cpu0", "hostmem0"), "cpu0"),
    )
    limitations = (
        "NVIDIA DGX B200 页面公开的是 8-GPU 系统聚合值（1,440 GB、64 TB/s），本预设另以 Samsung HBM3E 组件预设实例化两个可编辑 HBM 端点；DGX 聚合值仅保留为 GPU 产品参考。",
        "两个 HBM 节点的容量、读写带宽和 profile 均来自 samsung-hbm3e-36gb-9_2 组件预设；物理 B200 封装堆叠数量未由本拓扑推断。",
        "B200 峰值采用 NVIDIA Blackwell Technical Overview Table 3 的 HGX B200 dense FP16/BF16 tensor 2.2 PFLOPS；SM 数、时钟、缓存、效率和时序字段均为可编辑分析假设。",
        "HBF 容量和媒体读带宽由 SK hynix HBF 组件预设提供；写带宽没有可靠通用公开值，因此以 unknown 状态保留。",
        "cpu0/hostmem0 仅为使仿真执行器具备 host orchestration 路径的栈外辅助参考；hostmem0 由 Samsung DDR5 32GB 组件预设提供，不改变 1 GPU + 2 HBM + 2 HBF 加速器组成。",
    )
    return _definition(
        "nvidia-b200-1gpu-2hbf-2hbm", "NVIDIA B200 + 2×HBF + 2×HBM3E", "NVIDIA", "Blackwell",
        "storage_offload", "single_package", ANALYTICAL_APPROXIMATION,
        components, links, groups,
        {"gpu0": {"x": 260.0, "y": 180.0}, "hbm0": {"x": 560.0, "y": 40.0}, "hbm1": {"x": 780.0, "y": 40.0}, "hbf0": {"x": 560.0, "y": 300.0}, "hbf1": {"x": 780.0, "y": 300.0}, "cpu0": {"x": 0.0, "y": 140.0}, "hostmem0": {"x": 0.0, "y": 360.0}},
        (NVIDIA_B200, NVIDIA_BLACKWELL, NVIDIA_GH200, OCP_HBF, UCIE_SPEC), limitations,
        "用户指定的单 B200、双 HBM 和双 HBF 统一封装分析预设，并附带不计入加速器数量的辅助主机执行路径。",
        ("nvidia", "b200", "blackwell", "hbm3e", "hbf", "ucie", "flash"),
    )


def _b200_hbf_hbm_3hbf(component_catalog=None) -> ArchitecturePresetDefinition:
    """Capacity-expanded B200 analysis graph for full BF16 DeepSeek-V3 weights.

    The third HBF is an explicit analytical capacity coordinate. It does not
    claim that a production B200 package contains three HBF devices.
    """

    base = _b200_hbf_hbm(component_catalog)
    hardware = base.hardware
    components = list(hardware.components)
    gpu = next(item for item in components if item.component_id == "gpu0")
    hbf0 = next(item for item in components if item.component_id == "hbf0")
    metadata = dict(hbf0.metadata)
    template = dict(metadata.get("cost_profile_template", {}))
    resource_id = "hbf2.memory"
    template["resource_id"] = resource_id
    metadata["cost_profile_template"] = template
    metadata["memory_service_owner"] = resource_id
    metadata["model"] = "SK hynix HBF analysis device 2"
    physical = dict(metadata.get("physical_composition", {}))
    physical["unit_index"] = 2
    physical["unit_count_in_product"] = 3
    physical["unit_count_status"] = "analytical_capacity_expansion"
    metadata["physical_composition"] = physical
    hbf2 = replace(
        hbf0,
        component_id="hbf2",
        die_id="hbf2_die",
        cost_profile_id="sk_hynix_hbf_512gb_memory_2",
        ports=tuple(replace(port, port_id="hbf0") for port in hbf0.ports),
        metadata=metadata,
    )
    gpu_port = _port(
        "hbf2", "HBF", "endpoint", version="2.0", lanes=64,
        bandwidth_gbps=3_904.0, payload="streaming",
        metadata={"physical_transport_protocol": "UCIe"},
    )
    components = [replace(item, ports=item.ports + (gpu_port,)) if item.component_id == "gpu0" else item for item in components]
    components.append(hbf2)
    links = list(hardware.links)
    links.append(_link(
        "gpu_hbf2", "gpu0", "hbf2", "hbf2", "hbf0", "HBF",
        version="2.0", lanes=64, bandwidth_gbps=3_904.0,
        latency_ns=4_000.0, payload="streaming",
        metadata={"physical_transport_protocol": "UCIe"},
    ))
    groups = []
    for group in base.groups:
        item = dict(group)
        if item.get("group_id") == "b200_package":
            item["members"] = list(item.get("members", ())) + ["hbf2"]
            item["label"] = "NVIDIA B200 with 2×HBM3E + 3×HBF"
        groups.append(item)
    positions = dict(hardware.metadata.get("topology_view", {}).get("layout", {}).get("positions", {}))
    positions["hbf2"] = {"x": 1000.0, "y": 300.0}
    limitations = tuple(base.limitations) + (
        "这是为 DeepSeek-V3 BF16 全量权重敏感性分析增加的第三个 HBF 容量坐标，不代表量产 B200 封装拓扑。",
        "总可用 HBF 容量约 1.536 TB；仍需为运行时、KV 和控制面保留余量。",
    )
    return _definition(
        "nvidia-b200-1gpu-3hbf-2hbm",
        "NVIDIA B200 + 3×HBF + 2×HBM3E（DeepSeek-V3 capacity analysis）",
        "NVIDIA", "Blackwell", "storage_offload", "single_package",
        ANALYTICAL_APPROXIMATION, components, links, groups, positions,
        base.sources, limitations,
        "为 DeepSeek-V3 BF16 权重容量分析准备的扩展 HBF 拓扑；第三个 HBF 是分析容量坐标。",
        ("nvidia", "b200", "blackwell", "hbm3e", "hbf", "ucie", "flash", "deepseek"),
    )


_NATIVE_RTX5080_SOURCE = _source(
    "native-llama-parity-rtx5080.json hardware input",
    "file:///C:/Users/A/Downloads/native-llama-parity-rtx5080.json",
    "本机 native 采集",
    "local_native_hardware_snapshot",
)


def _native_rtx5080_local(component_catalog=None) -> ArchitecturePresetDefinition:
    """The measured local host/RTX 5080 hardware graph as a loadable preset.

    The native file also contains model, workload, placement and timing data.
    Only its hardware meaning is retained here; all editable capabilities come
    from the curated RTX 5080, GDDR7, Ryzen and local DDR5 component presets.
    """

    preset_id = "local-native-rtx5080-9950x3d-gddr7-ddr5"
    pcie_gbps = 252.032
    gddr7_gbps = 7680.0
    ddr5_gbps = 716.8

    gpu = _catalog_component(
        "nvidia-rtx-5080",
        "gpu0",
        package_id="gpu_package0",
        die_id="gpu_die0",
        component_catalog=component_catalog,
        ports=(
            _port(
                "gddr7",
                "GDDR",
                "controller",
                version="GDDR7",
                lanes=256,
                bandwidth_gbps=gddr7_gbps,
                metadata={"bandwidth_source": "component_preset"},
            ),
            _port(
                "pcie0",
                "PCIe",
                "endpoint",
                version="5.0",
                lanes=8,
                bandwidth_gbps=pcie_gbps,
                payload="coherent_dma",
                metadata={"bandwidth_source": "native_negotiated_link"},
            ),
        ),
        metadata_updates={
            "architecture_role": "native_gpu_compute",
            "native_source_component_id": "gpu0",
            "memory_interface_protocol": "GDDR7",
        },
    )
    cpu = _catalog_component(
        "amd-ryzen-9-9950x3d",
        "cpu0",
        package_id="host_package0",
        die_id="cpu_die0",
        component_catalog=component_catalog,
        ports=(
            _port(
                "pcie0",
                "PCIe",
                "root",
                version="5.0",
                lanes=8,
                bandwidth_gbps=pcie_gbps,
                payload="coherent_dma",
                metadata={"bandwidth_source": "native_negotiated_link"},
            ),
            _port(
                "memory",
                "DDR5",
                "controller",
                version="DDR5-5600",
                lanes=128,
                bandwidth_gbps=ddr5_gbps,
                metadata={"bandwidth_source": "component_preset"},
            ),
        ),
        metadata_updates={
            "architecture_role": "native_host_cpu",
            "native_source_component_id": "cpu0",
        },
    )
    hostmem = _catalog_component(
        "acer-local-ddr5-128gb-5600-dual-channel",
        "hostmem0",
        package_id="host_package0",
        die_id="host_memory0",
        component_catalog=component_catalog,
        ports=(
            _port(
                "memory",
                "DDR5",
                "device",
                version="DDR5-5600",
                lanes=128,
                bandwidth_gbps=ddr5_gbps,
                metadata={"bandwidth_source": "component_preset"},
            ),
        ),
        metadata_updates={
            "architecture_role": "native_host_memory",
            "native_source_component_id": "hostmem0",
            "memory_service_owner": "hostmem0.memory",
            "resident_access_path": "topology",
            "physical_hardware_component": True,
        },
    )
    gpu_memory = _catalog_component(
        "gddr7-16gb-30_0-256bit",
        "gddr0",
        package_id="gpu_package0",
        die_id="gddr7_memory0",
        component_catalog=component_catalog,
        ports=(
            _port(
                "memory",
                "GDDR",
                "device",
                version="GDDR7",
                lanes=256,
                bandwidth_gbps=gddr7_gbps,
                metadata={"bandwidth_source": "component_preset"},
            ),
        ),
        metadata_updates={
            "architecture_role": "native_gpu_local_memory",
            "native_source_component_id": "gddr0",
            "attached_memory_preset_id": "gddr7-16gb-30_0-256bit",
            "component_preset_status": "logical_gpu_attached_memory",
            "memory_type": "GDDR7",
            "memory_service_owner": "gddr0.gddr_fabric",
            "resident_access_path": "topology",
            "physical_hardware_component": False,
            "simulator_memory_kind": "gddr",
        },
    )

    components = (gpu, cpu, hostmem, gpu_memory)
    links = (
        _link(
            "cpu-gpu-pcie",
            "cpu0",
            "pcie0",
            "gpu0",
            "pcie0",
            "PCIe",
            version="5.0",
            lanes=8,
            bandwidth_gbps=pcie_gbps,
            latency_ns=800.0,
            payload="coherent_dma",
            metadata={
                "bandwidth_source": "native_negotiated_link",
                "native_source_link_id": "cpu-gpu-pcie",
            },
        ),
        _link(
            "cpu-hostmem-ddr",
            "cpu0",
            "memory",
            "hostmem0",
            "memory",
            "DDR5",
            version="DDR5-5600",
            lanes=128,
            bandwidth_gbps=ddr5_gbps,
            # memory_component service is authoritative for this logical DDR path.
            latency_ns=100.0,
            metadata={
                "bandwidth_source": "memory_component",
                "service_ref": "hostmem0.access",
                "bandwidth_resource_id": "hostmem0.memory",
                "native_source_link_id": "cpu-hostmem-ddr",
            },
        ),
        _link(
            "gpu-gddr7",
            "gpu0",
            "gddr7",
            "gddr0",
            "memory",
            "GDDR",
            version="GDDR7",
            lanes=256,
            bandwidth_gbps=gddr7_gbps,
            # This link is a logical memory-service interface.  The attached
            # GDDR7 component preset is authoritative for service latency
            # (35 ns); keeping the two values aligned avoids the
            # memory_component contract rejecting an otherwise valid preset.
            latency_ns=35.0,
            metadata={
                "bandwidth_source": "memory_component",
                "service_ref": "gddr0.access",
                "bandwidth_resource_id": "gddr0.gddr_fabric",
                "native_source_link_id": "gpu-gddr7",
                "logical_memory_interface": True,
            },
        ),
    )
    limitations = (
        "这是 native-llama-parity-rtx5080.json 的硬件输入提炼；模型、workload、placement、profile 和 timing 不属于架构预设。",
        "native 文件中的 hbm1-hbm7 是历史参考拓扑残留，已删除；RTX 5080 本机本地显存使用 GDDR7 组件预设。",
        "hostmem0 使用本机 Acer DDR5-5600 双通道组件预设；当前配置带宽为 5600 MT/s × 64 bit ÷ 8 × 2 = 89.6 GB/s。",
        "链路 bandwidth_gbps 是对应协议的传输上限，采用单向 Gb/s 口径；PCIe 采用本机协商出的 PCIe 5.0 x8 值。",
        "CPU 到主机内存的物理接口是 DDR5；PCIe 只用于 CPU 到 GPU 的独立互连。",
    )
    result = _definition(
        preset_id,
        "本机 Native RTX 5080 + Ryzen 9 9950X3D",
        "Local measured hardware",
        "RTX 5080 / Zen 5",
        "single_gpu_host",
        "single_host",
        EXPERIMENTAL_REFERENCE,
        components,
        links,
        (
            _group("gpu_local", "RTX 5080 + GDDR7", ("gpu0", "gddr0"), "gpu0"),
            _group("host", "Ryzen 9 9950X3D + DDR5", ("cpu0", "hostmem0"), "cpu0"),
        ),
        {
            "gpu0": {"x": 360.0, "y": 200.0},
            "gddr0": {"x": 650.0, "y": 200.0},
            "cpu0": {"x": 40.0, "y": 200.0},
            "hostmem0": {"x": 40.0, "y": 430.0},
        },
        (_NATIVE_RTX5080_SOURCE,),
        limitations,
        "本机 native RTX 5080 推理硬件场景的可直接导入架构预设；硬件能力由组件预设统一提供。",
        ("local", "native", "rtx5080", "gddr7", "ddr5", "pcie", "single-gpu"),
    )
    metadata = {
        **result.hardware.metadata,
        "native_input": {
            "source_file": "native-llama-parity-rtx5080.json",
            "source_path": "C:/Users/A/Downloads/native-llama-parity-rtx5080.json",
            "hardware_only_projection": True,
            "excluded_sections": ["model", "workload", "placement", "profiles", "timing"],
        },
        "bandwidth_contract": {
            "protocol_bandwidth_is_transfer_upper_bound": True,
            "unit": "Gb/s_decimal_one_way",
            "read_write_accounting": "shared_bidirectional_resource",
        },
    }
    return replace(result, hardware=replace(result.hardware, metadata=metadata))


# Current loadable architecture catalog.
_PRESETS: Tuple[ArchitecturePresetDefinition, ...] = (
    _b200_hbf_hbm(),
    _b200_hbf_hbm_3hbf(),
    _native_rtx5080_local(),
)
PUBLIC_ARCHITECTURE_PRESET_IDS = frozenset({
    "nvidia-b200-1gpu-2hbf-2hbm",
    "nvidia-b200-1gpu-3hbf-2hbm",
    "local-native-rtx5080-9950x3d-gddr7-ddr5",
})


_BY_ID: Mapping[str, ArchitecturePresetDefinition] = {
    item.preset_id: item for item in _PRESETS
    if item.preset_id in PUBLIC_ARCHITECTURE_PRESET_IDS
}
if len(_BY_ID) != len(PUBLIC_ARCHITECTURE_PRESET_IDS):
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
        constraints.extend(("uncalibrated_soc_gpu_proxy", "per_dram_controller_profile_binding_required"))
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

    return tuple(
        _metadata(item)
        for item in sorted(_PRESETS, key=lambda value: value.preset_id)
        if item.preset_id in PUBLIC_ARCHITECTURE_PRESET_IDS
    )


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
        "loadable": sorted({item["loadable"] for item in items}),
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


def materialize_architecture_payload(preset_id: str, *, component_catalog=None) -> Dict[str, Any]:
    """Return a new JSON-compatible complete ``HardwareSpec`` payload."""

    item = get_architecture_preset(preset_id)
    if component_catalog is not None:
        if preset_id == "nvidia-b200-1gpu-2hbf-2hbm":
            item = _b200_hbf_hbm(component_catalog)
        elif preset_id == "nvidia-b200-1gpu-3hbf-2hbm":
            item = _b200_hbf_hbm_3hbf(component_catalog)
        elif preset_id == "local-native-rtx5080-9950x3d-gddr7-ddr5":
            item = _native_rtx5080_local(component_catalog)
    if not item.loadable:
        raise ValueError(
            "架构拓扑预设 {} 当前不可载入：{}".format(
                preset_id, "；".join(item.limitations)
            )
        )
    return to_primitive(item.hardware)


def architecture_preset_detail(preset_id: str, *, component_catalog=None) -> Dict[str, Any]:
    item = get_architecture_preset(preset_id)
    hardware = materialize_architecture_payload(preset_id, component_catalog=component_catalog) if item.loadable else None
    topology_view = hardware["metadata"]["topology_view"] if hardware else None
    profiles: Dict[str, Any] = {}
    if hardware is not None:
        # These are the existing analytical scenario policies, now explicitly
        # supplied with the selected architecture rather than inherited from a
        # previously loaded reference scenario. They are not device measurements.
        profiles["fusion"] = {
            "qkv_rope": True,
            "flash_attention": True,
            "gemm_epilogue_activation": True,
            "residual_norm": True,
            "max_fused_working_set_bytes": 30 * 1024**2,
        }
        gpu_ids = {
            component["component_id"] for component in hardware["components"]
            if str(component["kind"]).strip().lower().replace("-", "_").replace(" ", "_") == "gpu"
        }
        if gpu_ids:
            # architecture_default historically invents gpu0 for GPU-free
            # hardware. Bind only devices that the selected preset contains.
            profiles["runtime"] = to_primitive(ControllerProfile(
                gpu_controllers={gpu_id: GPUControllerProfile() for gpu_id in sorted(gpu_ids)},
            ))
    return {
        "preset": _metadata(item),
        "hardware": hardware,
        "profiles": profiles,
        "profiles_parameter_basis": {
            "kind": "existing_analytical_template",
            "runtime_source": "schema_v4.ControllerProfile / GPUControllerProfile",
            "fusion_source": "reference.build_reference_scenario.fusion_policy",
            "is_native_measurement": False,
            "is_manufacturer_specification": False,
        } if profiles else {},
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
