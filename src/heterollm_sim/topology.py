"""Static legality checks for HardwareIR component graphs."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import re
from typing import Dict, List, Mapping, Optional, Set, Tuple

from .ir import (
    ComponentSpec,
    HardwareSpec,
    LinkSpec,
    PortSpec,
    normalize_component_kind,
)


_PROTOCOL_ALIASES = {
    "pcie": "pcie",
    "pci_e": "pcie",
    "cxl": "cxl",
    "ucie": "ucie",
    "nvlink": "nvlink",
    "hbm": "hbm",
    # Canonical authoring uses protocol=GDDR and version=<generation>.
    # Generation-shaped names remain accepted during input normalization.
    "gddr": "gddr",
    "gddr6": "gddr6",
    "gddr_6": "gddr6",
    "gddr6x": "gddr6x",
    "gddr_6x": "gddr6x",
    "gddr7": "gddr7",
    "gddr_7": "gddr7",
    "ddr3": "ddr",
    "ddr4": "ddr",
    "ddr5": "ddr",
}

_GDDR_PROTOCOLS = frozenset({"gddr", "gddr6", "gddr6x", "gddr7"})
_DIRECTION_ALIASES = {
    "in": "input",
    "input": "input",
    "rx": "input",
    "out": "output",
    "output": "output",
    "tx": "output",
    "bi": "bidirectional",
    "bidir": "bidirectional",
    "bidirectional": "bidirectional",
    "inout": "bidirectional",
}


def _key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _protocol(value: str) -> str:
    normalized = _key(value)
    return _PROTOCOL_ALIASES.get(normalized, normalized)


def _direction(value: str) -> str:
    return _DIRECTION_ALIASES.get(_key(value), _key(value))


def _role(value: str) -> str:
    normalized = _key(value)
    aliases = {
        "root_complex": "root",
        "rc": "root",
        "host_bridge": "root",
        "ep": "endpoint",
        "memory": "device",
        "d2d": "endpoint",
        "die": "endpoint",
        "peer": "endpoint",
    }
    return aliases.get(normalized, normalized)


def _component_kind(component: ComponentSpec) -> str:
    normalized = normalize_component_kind(component.kind)
    if normalized in {"hbm", "hbm_stack"}:
        return "hbm"
    # Component normalization is intentionally kept in one place in ir.py,
    # but accept generation-shaped authoring values here as well so topology
    # validation remains fail-closed while a config is being normalized.
    if normalized in {"gddr", "gddr_memory", "gddr6", "gddr6x", "gddr7"}:
        return "gddr"
    return normalized


def _gddr_generation(value: object) -> Optional[str]:
    """Extract a canonical GDDR generation from a protocol/version value."""

    text = re.sub(r"^GDDR[-_ ]+", "GDDR", str(value or "").strip().upper())
    matched = re.match(r"^(GDDR6X|GDDR6|GDDR7)(?:$|[-_ ].*)", text)
    return matched.group(1) if matched is not None else None


def _metadata_generations(metadata: object) -> Set[str]:
    if not isinstance(metadata, Mapping):
        return set()
    values = []
    for key in (
        "generation",
        "memory_generation",
        "supported_generations",
        "gddr_generations",
        "supported_gddr_generations",
        "memory_generations",
    ):
        raw = metadata.get(key)
        if isinstance(raw, str):
            values.extend(re.split(r"[,;\s]+", raw))
        elif isinstance(raw, (tuple, list, set, frozenset)):
            values.extend(raw)
    return {
        generation
        for value in values
        if (generation := _gddr_generation(value)) is not None
    }


def _component_gddr_generation(component: ComponentSpec) -> Optional[str]:
    metadata = component.metadata
    physical_config = metadata.get("physical_memory_config")
    physical_generation = (
        physical_config.get("generation")
        if isinstance(physical_config, Mapping)
        else getattr(physical_config, "generation", None)
    )
    candidates = [
        metadata.get("generation"),
        metadata.get("memory_generation"),
        metadata.get("technology", {}).get("generation")
        if isinstance(metadata.get("technology"), Mapping)
        else None,
        physical_generation,
        component.kind,
    ]
    for value in candidates:
        generation = _gddr_generation(value)
        if generation is not None:
            return generation
    return None


def _protocols_compatible(port_protocol: str, link_protocol: str) -> bool:
    """Allow a generic GDDR port to negotiate a generation explicitly."""

    if port_protocol == link_protocol:
        return True
    return port_protocol in _GDDR_PROTOCOLS and link_protocol in _GDDR_PROTOCOLS


def _version_key(version: str) -> Optional[Tuple[int, ...]]:
    numbers = re.findall(r"\d+", version)
    if not numbers:
        return None
    return tuple(int(number) for number in numbers)


def _version_supported(negotiated: str, maximum: str) -> bool:
    negotiated_key = _version_key(negotiated)
    maximum_key = _version_key(maximum)
    if negotiated_key is None or maximum_key is None:
        return _key(negotiated) == _key(maximum)
    width = max(len(negotiated_key), len(maximum_key))
    negotiated_key += (0,) * (width - len(negotiated_key))
    maximum_key += (0,) * (width - len(maximum_key))
    return negotiated_key <= maximum_key


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    component_id: Optional[str] = None
    port_id: Optional[str] = None
    link_id: Optional[str] = None
    message_en: Optional[str] = None

    def __str__(self) -> str:
        location = self.link_id or self.component_id or "拓扑"
        return "{} [{}]: {}".format(location, self.code, self.message)

    @property
    def message_zh(self) -> str:
        """Return the explicit Chinese diagnostic message."""

        return self.message


@dataclass(frozen=True)
class TopologyValidationReport:
    errors: Tuple[ValidationIssue, ...] = ()
    warnings: Tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def ok(self) -> bool:
        return self.is_valid

    @property
    def error_codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.errors)

    def format(self) -> str:
        if self.is_valid:
            return "拓扑校验通过"
        return "拓扑校验失败：\n- " + "\n- ".join(
            str(issue) for issue in self.errors
        )

    def format_en(self) -> str:
        """Return the English report for logs and machine diagnostics."""

        if self.is_valid:
            return "topology is valid"
        return "topology validation failed:\n- " + "\n- ".join(
            "{} [{}]: {}".format(
                issue.link_id or issue.component_id or "topology",
                issue.code,
                issue.message_en or issue.message,
            )
            for issue in self.errors
        )


class TopologyValidationError(ValueError):
    def __init__(self, report: TopologyValidationReport):
        self.report = report
        super().__init__(report.format())


def _validate_protocol_rules(
    link: LinkSpec,
    source_component: ComponentSpec,
    source_port: PortSpec,
    target_component: ComponentSpec,
    target_port: PortSpec,
    errors: List[ValidationIssue],
) -> None:
    protocol = _protocol(link.protocol)

    def add(code: str, message: str, message_en: str) -> None:
        errors.append(
            ValidationIssue(
                code=code,
                message=message,
                message_en=message_en,
                link_id=link.link_id,
            )
        )

    if protocol == "hbm":
        source_is_hbm = _component_kind(source_component) == "hbm"
        target_is_hbm = _component_kind(target_component) == "hbm"
        if source_is_hbm == target_is_hbm:
            add(
                "hbm_dedicated_endpoint",
                "HBM 本地显存链路必须且只能连接一个 HBM 显存组件",
                "HBM links must connect exactly one HBM local-memory component",
            )
            return
        memory_port = source_port if source_is_hbm else target_port
        controller_port = target_port if source_is_hbm else source_port
        if _role(memory_port.role) not in {"device", "endpoint"}:
            add(
                "hbm_device_role",
                "HBM 显存侧端口必须使用 device 角色",
                "the HBM memory-side port must have device role",
            )
        if _role(controller_port.role) not in {"controller", "host", "root"}:
            add(
                "hbm_controller_role",
                "HBM 控制器侧端口必须使用 controller 角色",
                "the HBM controller-side port must have controller role",
            )

    elif protocol in _GDDR_PROTOCOLS:
        source_is_gddr = _component_kind(source_component) == "gddr"
        target_is_gddr = _component_kind(target_component) == "gddr"
        if source_is_gddr == target_is_gddr:
            add(
                "gddr_dedicated_endpoint",
                "GDDR 本地显存链路必须且只能连接一个 GDDR 显存组件",
                "GDDR links must connect exactly one GDDR local-memory component",
            )
            return
        memory_component = source_component if source_is_gddr else target_component
        memory_port = source_port if source_is_gddr else target_port
        controller_component = target_component if source_is_gddr else source_component
        controller_port = target_port if source_is_gddr else source_port
        if _role(memory_port.role) not in {"device", "endpoint"}:
            add(
                "gddr_device_role",
                "GDDR 显存侧端口必须使用 device 角色",
                "the GDDR memory-side port must have device role",
            )
        if _role(controller_port.role) not in {"controller", "host", "root"}:
            add(
                "gddr_controller_role",
                "GDDR 控制器侧端口必须使用 controller 角色",
                "the GDDR controller-side port must have controller role",
            )

        memory_generation = _component_gddr_generation(memory_component)
        link_generation = _gddr_generation(link.protocol) or _gddr_generation(link.version)
        port_generation = _gddr_generation(memory_port.protocol) or _gddr_generation(memory_port.version)
        controller_generation = _gddr_generation(controller_port.protocol) or _gddr_generation(controller_port.version)
        controller_supported = (
            _metadata_generations(controller_port.metadata)
            | _metadata_generations(controller_component.metadata)
        )
        required = memory_generation or link_generation or port_generation
        if memory_generation and link_generation and memory_generation != link_generation:
            add(
                "gddr_generation_mismatch",
                "GDDR 显存代际 {} 与链路代际 {} 不一致".format(memory_generation, link_generation),
                "GDDR memory generation {} does not match link generation {}".format(memory_generation, link_generation),
            )
        if link_generation and port_generation and link_generation != port_generation:
            add(
                "gddr_memory_port_generation_mismatch",
                "GDDR 链路代际 {} 与显存端口代际 {} 不一致".format(link_generation, port_generation),
                "GDDR link generation {} does not match memory port generation {}".format(link_generation, port_generation),
            )
        if required is None:
            add(
                "gddr_generation_required",
                "GDDR 链路必须明确声明 GDDR6、GDDR6X 或 GDDR7 代际",
                "GDDR links must explicitly declare GDDR6, GDDR6X or GDDR7",
            )
        elif controller_supported:
            if required not in controller_supported:
                add(
                    "gddr_controller_unsupported_generation",
                    "控制器未声明支持 GDDR 代际 {}".format(required),
                    "controller does not declare support for GDDR generation {}".format(required),
                )
        elif controller_generation is not None:
            if required != controller_generation:
                add(
                    "gddr_controller_generation_mismatch",
                    "控制器端口仅声明 {}，不支持 {}".format(controller_generation, required),
                    "controller port declares {} and does not support {}".format(controller_generation, required),
                )
        else:
            add(
                "gddr_controller_capability_missing",
                "GDDR 控制器必须声明端口代际或 supported_generations",
                "GDDR controller must declare a port generation or supported_generations",
            )

    elif protocol in {"ucie", "hbf"}:
        if not source_component.package_id or not target_component.package_id:
            add("ucie_package_missing", "HBF/UCIe 端点必须声明 package_id", "HBF/UCIe endpoints must declare package_id")
        elif source_component.package_id != target_component.package_id:
            add("ucie_cross_package", "HBF/UCIe 仅支持封装内的裸片互连", "HBF/UCIe is package-local die-to-die connectivity")
        if not source_component.die_id or not target_component.die_id:
            add("ucie_die_missing", "HBF/UCIe 端点必须声明 die_id", "HBF/UCIe endpoints must declare die_id")
        elif source_component.die_id == target_component.die_id:
            add("ucie_same_die", "HBF/UCIe 必须连接不同的裸片", "HBF/UCIe must connect different dies")
        if not link.payload or not link.payload.strip():
            add("ucie_payload_missing", "HBF/UCIe 链路必须声明 payload", "HBF/UCIe links must declare a payload")
        else:
            link_payload = _key(link.payload)
            for port, endpoint_name in (
                (source_port, "source"),
                (target_port, "target"),
            ):
                if port.payload and _key(port.payload) != link_payload:
                    add(
                        "ucie_payload_mismatch",
                        "{}端口不支持协商后的 payload {}".format(
                            "源" if endpoint_name == "source" else "目标", link.payload
                        ),
                        "{} port does not support negotiated payload {}".format(
                            endpoint_name, link.payload
                        ),
                    )

    elif protocol == "pcie":
        roles = {_role(source_port.role), _role(target_port.role)}
        if roles != {"root", "endpoint"}:
            add("pcie_role_pair", "PCIe 链路必须连接一个 root 和一个 endpoint", "PCIe links require one root and one endpoint")

    elif protocol == "cxl":
        roles = {_role(source_port.role), _role(target_port.role)}
        if roles != {"host", "device"}:
            add("cxl_role_pair", "CXL 链路必须连接一个 host 和一个 device", "CXL links require one host and one device")

    elif protocol == "nvlink":
        roles = (_role(source_port.role), _role(target_port.role))
        allowed = {"endpoint", "switch"}
        if any(role not in allowed for role in roles) or roles == ("switch", "switch"):
            add(
                "nvlink_endpoint_role",
                "NVLink 必须连接 endpoint 对端，也可以经由 NVLink switch 中转",
                "NVLink requires endpoint peers, optionally through an NVLink switch",
            )



def _validate_stack_metadata(hardware: HardwareSpec, errors: List[ValidationIssue]) -> None:
    """Check opt-in 3D annotations without imposing a layout on legacy graphs.

    Several functional blocks can share one die/layer (SoC compute and CIM).
    A vertical link describes an end-to-end path; it need not stop at every die.
    """

    def add(code: str, message: str, *, component_id=None, link_id=None) -> None:
        errors.append(ValidationIssue(code=code, message=message, message_en=message, component_id=component_id, link_id=link_id))

    components = hardware.component_map()
    links = {link.link_id: link for link in hardware.links}
    layers = {}
    dies = {}
    packages = {}
    for component in hardware.components:
        metadata = component.metadata
        for key in ("stack_id", "thermal_domain_id"):
            if key in metadata and (not isinstance(metadata[key], str) or not metadata[key].strip()):
                add("invalid_" + key, key + " must be non-empty text", component_id=component.component_id)
        if "stack_id" in metadata or "stack_layer" in metadata:
            stack, layer = metadata.get("stack_id"), metadata.get("stack_layer")
            if not isinstance(stack, str) or not stack.strip() or isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                add("invalid_stack_location", "stack_id and a non-negative integer stack_layer are required together", component_id=component.component_id)
            elif not component.package_id or not component.die_id:
                add("stack_identity_missing", "stacked components require package_id and die_id", component_id=component.component_id)
            else:
                if packages.setdefault(stack, component.package_id) != component.package_id:
                    add("stack_cross_package", "one stack_id cannot span packages", component_id=component.component_id)
                if layers.setdefault((stack, layer), component.die_id) != component.die_id:
                    add("stack_layer_collision", "different dies cannot occupy the same stack layer", component_id=component.component_id)
                if dies.setdefault((component.package_id, component.die_id), (stack, layer)) != (stack, layer):
                    add("stack_die_location_conflict", "blocks on one die must agree on stack_id and stack_layer", component_id=component.component_id)
        if "vertical_link_id" in metadata:
            link_id = metadata["vertical_link_id"]
            link = links.get(link_id) if isinstance(link_id, str) else None
            if link is None or component.component_id not in (link.source_component, link.target_component) or link.metadata.get("vertical_link") is not True:
                add("invalid_vertical_link_reference", "vertical_link_id must reference an incident vertical link", component_id=component.component_id)

    for link in hardware.links:
        metadata = link.metadata
        domain = metadata.get("thermal_domain_id")
        if "thermal_domain_id" in metadata and (not isinstance(domain, str) or not domain.strip()):
            add("invalid_thermal_domain_id", "thermal_domain_id must be non-empty text", link_id=link.link_id)
        for flag in ("vertical_link", "on_die"):
            if flag in metadata and not isinstance(metadata[flag], bool):
                add("invalid_" + flag, flag + " must be boolean", link_id=link.link_id)
        source, target = components.get(link.source_component), components.get(link.target_component)
        if source is None or target is None:
            continue  # The main validator reports missing endpoints.
        if metadata.get("on_die") is True:
            if not source.die_id or not source.package_id or (source.package_id, source.die_id) != (target.package_id, target.die_id):
                add("on_die_link_cross_die", "an on_die link must remain on the same package and die", link_id=link.link_id)
        if metadata.get("vertical_link") is not True:
            continue
        if _protocol(link.protocol) not in {"tsv", "hybridbonding", "hybrid_bonding"}:
            add("vertical_link_protocol", "vertical links must explicitly use TSV or HybridBonding, not UCIe", link_id=link.link_id)
        stack = metadata.get("stack_id")
        if not isinstance(stack, str) or not stack.strip() or stack != source.metadata.get("stack_id") or stack != target.metadata.get("stack_id"):
            add("vertical_link_stack_mismatch", "vertical link and both endpoints must share a stack_id", link_id=link.link_id)
        if not source.package_id or source.package_id != target.package_id or not source.die_id or not target.die_id or source.die_id == target.die_id:
            add("vertical_link_die_mismatch", "vertical links must connect distinct dies in the same package", link_id=link.link_id)
        first, last = source.metadata.get("stack_layer"), target.metadata.get("stack_layer")
        if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in (first, last)) or first == last:
            add("vertical_link_layers", "vertical endpoints must declare different non-negative stack layers", link_id=link.link_id)
        if not link.bidirectional:
            add("vertical_link_directions", "this vertical-path contract requires bidirectional read/write connectivity", link_id=link.link_id)
        for key, path in (("read_path", [link.target_component, link.source_component]),
                          ("write_path", [link.source_component, link.target_component])):
            declared = metadata.get(key)
            if not isinstance(declared, (tuple, list)) or list(declared) != path:
                add("vertical_link_" + key, key + " must explicitly match the link endpoints and direction", link_id=link.link_id)


def validate_topology(hardware: HardwareSpec) -> TopologyValidationReport:
    """Return every discoverable static topology error in one report."""

    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []

    def add(
        code: str,
        message: str,
        message_en: str,
        component_id: Optional[str] = None,
        port_id: Optional[str] = None,
        link_id: Optional[str] = None,
    ) -> None:
        errors.append(
            ValidationIssue(
                code=code,
                message=message,
                message_en=message_en,
                component_id=component_id,
                port_id=port_id,
                link_id=link_id,
            )
        )

    component_ids = [component.component_id for component in hardware.components]
    if not hardware.name or not hardware.name.strip():
        add("hardware_name_empty", "硬件名称不能为空", "hardware name must not be empty")
    if not hardware.schema_version or not hardware.schema_version.strip():
        add("schema_version_empty", "schema_version 不能为空", "schema_version must not be empty")
    if not hardware.components:
        add("components_empty", "硬件至少需要包含一个组件", "hardware must contain at least one component")
    for component_id, count in Counter(component_ids).items():
        if not component_id or not component_id.strip():
            add("component_id_empty", "component_id 不能为空", "component_id must not be empty")
        if count > 1:
            add("duplicate_component", "component_id 必须唯一", "component_id must be unique", component_id)

    components: Dict[str, ComponentSpec] = {}
    ports: Dict[Tuple[str, str], PortSpec] = {}
    for component in hardware.components:
        components.setdefault(component.component_id, component)
        if not component.kind or not component.kind.strip():
            add("component_kind_empty", "组件 kind 不能为空", "component kind must not be empty", component.component_id)
        if component.capacity_bytes < 0:
            add("negative_capacity", "capacity_bytes 不能为负数", "capacity_bytes must be non-negative", component.component_id)
        if (
            component.peak_ops_per_s < 0
            or component.read_bandwidth_gbps < 0
            or component.write_bandwidth_gbps < 0
        ):
            add(
                "negative_component_rate",
                "组件的计算速率和带宽不能为负数",
                "component compute and bandwidth rates must be non-negative",
                component.component_id,
            )

        port_ids = [port.port_id for port in component.ports]
        for port_id, count in Counter(port_ids).items():
            if not port_id or not port_id.strip():
                add("port_id_empty", "port_id 不能为空", "port_id must not be empty", component.component_id)
            if count > 1:
                add(
                    "duplicate_port",
                    "同一组件内的 port_id 必须唯一",
                    "port_id must be unique within a component",
                    component.component_id,
                    port_id,
                )
        for port in component.ports:
            ports.setdefault((component.component_id, port.port_id), port)
            if not port.protocol or not port.protocol.strip():
                add("port_protocol_empty", "端口 protocol 不能为空", "port protocol must not be empty", component.component_id, port.port_id)
            if _direction(port.direction) not in {"input", "output", "bidirectional"}:
                add("invalid_direction", "端口 direction 无法识别", "unknown port direction", component.component_id, port.port_id)
            if not port.version or not port.version.strip():
                add("port_version_empty", "端口 version 不能为空", "port version must not be empty", component.component_id, port.port_id)
            if port.lanes <= 0:
                add("invalid_port_lanes", "端口 lanes 必须为正数", "port lanes must be positive", component.component_id, port.port_id)
            if port.bandwidth_gbps < 0:
                add("negative_port_bandwidth", "端口带宽不能为负数", "port bandwidth must be non-negative", component.component_id, port.port_id)
            if port.max_links <= 0:
                add("invalid_port_occupancy", "max_links 必须为正数", "max_links must be positive", component.component_id, port.port_id)

    link_ids = [link.link_id for link in hardware.links]
    for link_id, count in Counter(link_ids).items():
        if not link_id or not link_id.strip():
            add("link_id_empty", "link_id 不能为空", "link_id must not be empty")
        if count > 1:
            add("duplicate_link", "link_id 必须唯一", "link_id must be unique", link_id=link_id)

    occupancy: Counter[Tuple[str, str]] = Counter()
    adjacency: Dict[str, Set[str]] = {component_id: set() for component_id in components}
    for link in hardware.links:
        if not link.protocol or not link.protocol.strip():
            add("link_protocol_empty", "链路 protocol 不能为空", "link protocol must not be empty", link_id=link.link_id)
        if not link.version or not link.version.strip():
            add("link_version_empty", "链路 version 不能为空", "link version must not be empty", link_id=link.link_id)
        if link.lanes <= 0:
            add("invalid_link_lanes", "链路 lanes 必须为正数", "link lanes must be positive", link_id=link.link_id)
        if link.bandwidth_gbps < 0 or link.latency_ns < 0:
            add("negative_link_rate", "链路带宽和延迟不能为负数", "link bandwidth and latency must be non-negative", link_id=link.link_id)

        source_key = (link.source_component, link.source_port)
        target_key = (link.target_component, link.target_port)
        source_component = components.get(link.source_component)
        target_component = components.get(link.target_component)
        source_port = ports.get(source_key)
        target_port = ports.get(target_key)
        for endpoint_key, component, port in (
            (source_key, source_component, source_port),
            (target_key, target_component, target_port),
        ):
            if component is None:
                add(
                    "unknown_component",
                    "链路引用了未知组件 {}".format(endpoint_key[0]),
                    "link references unknown component {}".format(endpoint_key[0]),
                    link_id=link.link_id,
                )
            elif port is None:
                add(
                    "unknown_port",
                    "链路引用了未知端口 {}.{}".format(*endpoint_key),
                    "link references unknown port {}.{}".format(*endpoint_key),
                    component_id=endpoint_key[0],
                    port_id=endpoint_key[1],
                    link_id=link.link_id,
                )
            else:
                occupancy[endpoint_key] += 1

        if source_component is None or target_component is None or source_port is None or target_port is None:
            continue
        if source_key == target_key or link.source_component == link.target_component:
            add("self_link", "链路必须连接不同的组件", "links must connect distinct components", link_id=link.link_id)
        adjacency[link.source_component].add(link.target_component)
        adjacency[link.target_component].add(link.source_component)

        protocol = _protocol(link.protocol)
        for port, endpoint_name in ((source_port, "source"), (target_port, "target")):
            port_protocol = _protocol(port.protocol)
            if not _protocols_compatible(port_protocol, protocol):
                add(
                    "protocol_mismatch",
                    "{}端口 protocol {} 与链路 protocol {} 不匹配".format(
                        "源" if endpoint_name == "source" else "目标",
                        port.protocol,
                        link.protocol,
                    ),
                    "{} port protocol {} does not match link protocol {}".format(
                        endpoint_name, port.protocol, link.protocol
                    ),
                    link_id=link.link_id,
                )
            # GDDR generations are negotiated by the explicit generation
            # checks below.  A generic GDDR controller commonly uses a
            # controller-specific numeric version (for example ``1.0``),
            # which must not be compared as if it were a JEDEC generation.
            version_supported = (
                True
                if protocol in _GDDR_PROTOCOLS and port_protocol in _GDDR_PROTOCOLS
                else _version_supported(link.version, port.version)
            )
            if not version_supported:
                add(
                    "version_unsupported",
                    "{}端口 version {} 不支持协商后的 version {}".format(
                        "源" if endpoint_name == "source" else "目标",
                        port.version,
                        link.version,
                    ),
                    "{} port version {} cannot support negotiated version {}".format(
                        endpoint_name, port.version, link.version
                    ),
                    link_id=link.link_id,
                )
            if link.lanes > port.lanes:
                add(
                    "lanes_exceed_port",
                    "链路 lanes 超过{}端口容量".format(
                        "源" if endpoint_name == "source" else "目标"
                    ),
                    "link lanes exceed {} port capacity".format(endpoint_name),
                    link_id=link.link_id,
                )
            if port.bandwidth_gbps > 0 and link.bandwidth_gbps > port.bandwidth_gbps:
                add(
                    "bandwidth_exceeds_port",
                    "链路带宽超过{}端口容量".format(
                        "源" if endpoint_name == "source" else "目标"
                    ),
                    "link bandwidth exceeds {} port capacity".format(endpoint_name),
                    link_id=link.link_id,
                )

        source_direction = _direction(source_port.direction)
        target_direction = _direction(target_port.direction)
        if source_direction not in {"output", "bidirectional"}:
            add("source_direction", "源端口无法发送数据", "source port cannot transmit", link_id=link.link_id)
        if target_direction not in {"input", "bidirectional"}:
            add("target_direction", "目标端口无法接收数据", "target port cannot receive", link_id=link.link_id)
        if link.bidirectional and (
            source_direction != "bidirectional" or target_direction != "bidirectional"
        ):
            add(
                "bidirectional_port_required",
                "双向链路两端都必须使用双向端口",
                "bidirectional links require bidirectional ports",
                link_id=link.link_id,
            )

        _validate_protocol_rules(
            link,
            source_component,
            source_port,
            target_component,
            target_port,
            errors,
        )

    for endpoint, count in occupancy.items():
        port = ports[endpoint]
        if count > port.max_links:
            add(
                "port_overoccupied",
                "端口已连接 {} 条链路，超过 max_links={} 的限制".format(count, port.max_links),
                "port is used by {} links but max_links is {}".format(count, port.max_links),
                endpoint[0],
                endpoint[1],
            )
        component = components[endpoint[0]]
        component_kind = _component_kind(component)
        port_protocol = _protocol(port.protocol)
        if (component_kind == "hbm" or port_protocol == "hbm") and count > 1:
            add(
                "hbm_port_not_dedicated",
                "HBM 本地显存端口只能由一条链路独占使用",
                "HBM local-memory ports may be used by only one link",
                endpoint[0],
                endpoint[1],
            )
        elif (component_kind == "gddr" or port_protocol in _GDDR_PROTOCOLS) and count > 1:
            add(
                "gddr_port_not_dedicated",
                "GDDR 本地显存端口只能由一条链路独占使用",
                "GDDR local-memory ports may be used by only one link",
                endpoint[0],
                endpoint[1],
            )

    for component in hardware.components:
        component_kind = _component_kind(component)
        incident = [
            link
            for link in hardware.links
            if link.source_component == component.component_id
            or link.target_component == component.component_id
        ]
        for link in incident:
            protocol = _protocol(link.protocol)
            if component_kind == "hbm" and protocol != "hbm":
                add(
                    "hbm_non_dedicated_protocol",
                    "HBM 本地显存组件只能使用 HBM 专用链路",
                    "HBM local-memory components may only use dedicated HBM links",
                    component.component_id,
                    link_id=link.link_id,
                )
            elif component_kind == "gddr" and protocol not in _GDDR_PROTOCOLS:
                add(
                    "gddr_non_dedicated_protocol",
                    "GDDR 本地显存组件只能使用 GDDR 专用链路",
                    "GDDR local-memory components may only use dedicated GDDR links",
                    component.component_id,
                    link_id=link.link_id,
                )
            elif component_kind == "hbf" and protocol not in {"ucie", "hbf"}:
                add(
                    "hbf_non_ucie_protocol",
                    "V4 中的 HBF 必须通过 HBF 逻辑链路或其 UCIe 物理承载连接",
                    "V4 HBF must connect through the HBF logical protocol or its UCIe physical carrier",
                    component.component_id,
                    link_id=link.link_id,
                )
            elif component_kind in {"ssd", "high_io_ssd"} and protocol not in {
                "pcie",
                "cxl",
            }:
                add(
                    "ssd_non_storage_protocol",
                    "SSD 和 High-I/O SSD 组件必须通过 PCIe 或 CXL 连接",
                    "SSD and High-I/O SSD components must connect through PCIe or CXL",
                    component.component_id,
                    link_id=link.link_id,
                )

    if hardware.require_connected and components:
        start = next(iter(components))
        visited = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        missing = sorted(set(components) - visited)
        if missing:
            add(
                "disconnected_topology",
                "以下组件未接入硬件拓扑图：{}".format(
                    ", ".join(missing)
                ),
                "components are not connected to the hardware graph: {}".format(
                    ", ".join(missing)
                ),
            )

    _validate_stack_metadata(hardware, errors)
    return TopologyValidationReport(errors=tuple(errors), warnings=tuple(warnings))


def assert_valid_topology(hardware: HardwareSpec) -> None:
    """Raise :class:`TopologyValidationError` unless ``hardware`` is legal."""

    report = validate_topology(hardware)
    if not report.is_valid:
        raise TopologyValidationError(report)


__all__ = [
    "TopologyValidationError",
    "TopologyValidationReport",
    "ValidationIssue",
    "assert_valid_topology",
    "validate_topology",
]
