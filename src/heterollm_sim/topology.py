"""Static legality checks for HardwareIR component graphs."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Set, Tuple

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
}
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
    return "hbm" if normalized in {"hbm", "hbm_stack"} else normalized


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
                "HBM 链路必须且只能连接一个 HBM 组件",
                "HBM links must connect exactly one HBM component",
            )
            return
        memory_port = source_port if source_is_hbm else target_port
        controller_port = target_port if source_is_hbm else source_port
        if _role(memory_port.role) not in {"device", "endpoint"}:
            add(
                "hbm_device_role",
                "HBM 侧端口必须使用 device 角色",
                "the HBM-side port must have device role",
            )
        if _role(controller_port.role) not in {"controller", "host", "root"}:
            add(
                "hbm_controller_role",
                "非 HBM 侧端口必须使用 controller 角色",
                "the non-HBM port must have controller role",
            )

    elif protocol == "ucie":
        if not source_component.package_id or not target_component.package_id:
            add("ucie_package_missing", "UCIe 端点必须声明 package_id", "UCIe endpoints must declare package_id")
        elif source_component.package_id != target_component.package_id:
            add("ucie_cross_package", "UCIe 仅支持封装内的裸片互连", "UCIe is package-local die-to-die connectivity")
        if not source_component.die_id or not target_component.die_id:
            add("ucie_die_missing", "UCIe 端点必须声明 die_id", "UCIe endpoints must declare die_id")
        elif source_component.die_id == target_component.die_id:
            add("ucie_same_die", "UCIe 必须连接不同的裸片", "UCIe must connect different dies")
        if not link.payload or not link.payload.strip():
            add("ucie_payload_missing", "UCIe 链路必须声明 payload", "UCIe links must declare a payload")
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
            if _protocol(port.protocol) != protocol:
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
            if not _version_supported(link.version, port.version):
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
        if (_component_kind(component) == "hbm" or _protocol(port.protocol) == "hbm") and count > 1:
            add(
                "hbm_port_not_dedicated",
                "HBM 端口只能由一条链路独占使用",
                "HBM ports may be used by only one link",
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
                    "HBM 组件只能使用专用 HBM 链路",
                    "HBM components may only use dedicated HBM links",
                    component.component_id,
                    link_id=link.link_id,
                )
            elif component_kind == "hbf" and protocol != "ucie":
                add(
                    "hbf_non_ucie_protocol",
                    "V4 中的 HBF 是 NAND 闪存，必须通过 UCIe 连接",
                    "V4 HBF is NAND flash and must connect through UCIe",
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
