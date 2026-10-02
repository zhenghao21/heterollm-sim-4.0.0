"""Small, shared physical data-access service.

This module deliberately stops at service generation.  It does not schedule
tasks or choose routes; callers can lower the returned phases into the
existing :class:`ResourceDemand`/``TaskSpec`` contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import EvidenceStatus, ResourceDemand, TaskCategory, TaskSpec
from .ir import ComponentSpec, LinkSpec, OFFLOAD_STORAGE_COMPONENT_KINDS, default_memory_resource_id
from .memory_service import realtime_memory_metrics


class AccessKind(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    COPY = "COPY"


# Short names are convenient for planners and preserve the wording in the
# design document.
READ = AccessKind.READ
WRITE = AccessKind.WRITE
COPY = AccessKind.COPY


def _positive(value: Any, name: str) -> float:
    number = _non_negative(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite non-negative number") from None
    if isinstance(value, bool) or not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = _non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _value(source: Any, *names: str, default: Any = None) -> Any:
    """Read a field from a mapping or a profile object."""

    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if source is not None and hasattr(source, name):
            return getattr(source, name)
    return default


def memory_service(
    *,
    read_bytes: int,
    write_bytes: int,
    bandwidth_gb_s: float,
    read_bandwidth_gb_s: Optional[float] = None,
    write_bandwidth_gb_s: Optional[float] = None,
    read_latency_ns: float = 0.0,
    write_latency_ns: float = 0.0,
    transaction_bytes: int = 256,
    max_outstanding_requests: int = 32,
    parallel_lanes: int = 1,
    service_model: str = "analytical",
) -> Mapping[str, object]:
    """Analytical bandwidth/latency envelope for one resolved payload stream.

    Reads and writes share an aggregate request window of
    ``max_outstanding_requests * parallel_lanes``.  Request-slot time divided
    by that window is an optimistic concurrency bound, not a schedule.
    A nonempty direction also cannot complete before its single-request latency.
    Bandwidth and these bounds overlap (max, not sum).  Zero latency reproduces
    the historical payload/BW model exactly, including partial transactions.

    Inputs must already reflect cache misses or other physical-flow decisions.
    No addresses, access ordering or achieved MLP are known: transaction counts
    assume coalesced per-direction streams, and payload bytes are NOT rounded up
    to transaction size.  Unaligned/strided overfetch must be supplied upstream.
    """

    _non_negative_int(read_bytes, "read_bytes")
    _non_negative_int(write_bytes, "write_bytes")
    _positive(bandwidth_gb_s, "bandwidth_gb_s")
    read_bw = bandwidth_gb_s if read_bandwidth_gb_s is None else read_bandwidth_gb_s
    write_bw = bandwidth_gb_s if write_bandwidth_gb_s is None else write_bandwidth_gb_s
    _positive(read_bw, "read_bandwidth_gb_s")
    _positive(write_bw, "write_bandwidth_gb_s")
    _non_negative(read_latency_ns, "read_latency_ns")
    _non_negative(write_latency_ns, "write_latency_ns")
    _positive_int(transaction_bytes, "transaction_bytes")
    _positive_int(max_outstanding_requests, "max_outstanding_requests")
    _positive_int(parallel_lanes, "parallel_lanes")
    logical_read_bytes, logical_write_bytes = read_bytes, write_bytes
    if service_model != "analytical":
        read_bytes = ((read_bytes + transaction_bytes - 1) // transaction_bytes) * transaction_bytes
        write_bytes = ((write_bytes + transaction_bytes - 1) // transaction_bytes) * transaction_bytes
    physical_bytes = read_bytes + write_bytes
    read_transactions = (read_bytes + transaction_bytes - 1) // transaction_bytes
    write_transactions = (write_bytes + transaction_bytes - 1) // transaction_bytes
    transaction_count = read_transactions + write_transactions
    request_window = max_outstanding_requests * parallel_lanes
    effective_outstanding = min(request_window, transaction_count)
    read_bandwidth_service_ns = read_bytes / read_bw if read_bytes else 0.0
    write_bandwidth_service_ns = write_bytes / write_bw if write_bytes else 0.0
    # One shared controller: directions add; preserve exact legacy rounding
    # when their rates coincide (rather than adding two rounded divisions).
    bandwidth_service_ns = (
        physical_bytes / read_bw if read_bw == write_bw else
        read_bandwidth_service_ns + write_bandwidth_service_ns
    )
    single_request_latency_ns = max(
        read_latency_ns if read_transactions else 0.0,
        write_latency_ns if write_transactions else 0.0,
    )
    request_slot_time_ns = (
        read_transactions * read_latency_ns
        + write_transactions * write_latency_ns
    )
    concurrency_service_ns = (
        request_slot_time_ns / effective_outstanding
        if effective_outstanding else 0.0
    )
    latency_service_ns = max(single_request_latency_ns, concurrency_service_ns)
    if service_model not in {"analytical", "serialized", "overlapped"}:
        raise ValueError("memory_service_model must be analytical, serialized or overlapped")
    if service_model != "analytical":
        latency_service_ns = (
            math.ceil(read_transactions / request_window) * read_latency_ns
            + math.ceil(write_transactions / request_window) * write_latency_ns
        )
    service_ns = (bandwidth_service_ns + latency_service_ns
                  if service_model == "serialized" else max(bandwidth_service_ns, latency_service_ns))
    service_source = (
        "zero_traffic" if not physical_bytes else
        "latency_concurrency" if latency_service_ns > bandwidth_service_ns else
        "bandwidth_and_latency_concurrency" if latency_service_ns == bandwidth_service_ns else
        "bandwidth"
    )
    if service_source == "latency_concurrency":
        bottleneck = "latency_concurrency"
    elif service_source == "bandwidth_and_latency_concurrency":
        bottleneck = "bandwidth_and_latency_concurrency"
    elif service_source == "bandwidth":
        bottleneck = "bandwidth"
    else:
        bottleneck = "none"
    metrics = realtime_memory_metrics(
        logical_read_bytes + logical_write_bytes,
        service_ns,
        physical_bytes=physical_bytes,
        bandwidth_ceiling_gb_s=(
            physical_bytes / bandwidth_service_ns
            if bandwidth_service_ns > 0 else 0.0
        ),
        queue_wait_ns=0.0,
        request_window_utilization=(
            min(1.0, transaction_count / float(request_window))
            if transaction_count else 0.0
        ),
        bottleneck=bottleneck,
    )
    return {
        "model": "memory_bandwidth_latency_concurrency_bound_v1",
        "evidence": EvidenceStatus.ANALYTICAL.value,
        "cycle_accurate": False,
        "latency_model_enabled": read_latency_ns > 0.0 or write_latency_ns > 0.0,
        "byte_scope": "resolved_backing_payload",
        "logical_read_bytes": logical_read_bytes,
        "logical_write_bytes": logical_write_bytes,
        "logical_bytes": logical_read_bytes + logical_write_bytes,
        "physical_read_bytes": read_bytes,
        "physical_write_bytes": write_bytes,
        "physical_bytes": physical_bytes,
        "physical_traffic_basis": ("payload_only_no_transaction_padding" if service_model == "analytical"
                                   else "rounded_directional_transactions"),
        "transaction_bytes": transaction_bytes,
        "read_transactions": read_transactions,
        "write_transactions": write_transactions,
        "transaction_count": transaction_count,
        "max_outstanding_requests": max_outstanding_requests,
        "parallel_lanes": parallel_lanes,
        "request_window": request_window,
        "effective_outstanding": effective_outstanding,
        "outstanding_basis": "configured_limit_times_parallel_lanes_capped_by_transaction_count",
        "concurrency_assumption": "independent_requests_shared_read_write_window_across_aggregate_lanes",
        "read_latency_ns": read_latency_ns,
        "write_latency_ns": write_latency_ns,
        "bandwidth_gb_s": bandwidth_gb_s,
        "read_bandwidth_gb_s": read_bw,
        "write_bandwidth_gb_s": write_bw,
        "read_bandwidth_service_ns": read_bandwidth_service_ns,
        "write_bandwidth_service_ns": write_bandwidth_service_ns,
        "bandwidth_model": "shared_controller_directional_serial_payload",
        "bandwidth_service_ns": bandwidth_service_ns,
        "single_request_latency_ns": single_request_latency_ns,
        "request_slot_time_ns": request_slot_time_ns,
        "concurrency_service_ns": concurrency_service_ns,
        "latency_service_ns": latency_service_ns,
        "latency_bound": latency_service_ns > bandwidth_service_ns,
        "service_source": service_source,
        "service_ns": service_ns,
        **metrics,
        "estimated_internal_wait_ns": max(0.0, concurrency_service_ns - single_request_latency_ns),
        "unmodeled_terms": (
            "request_dependencies_and_achieved_mlp",
            "transaction_alignment_fragmentation_and_padding",
            "bank_row_buffer_refresh_and_read_write_turnaround",
        ),
    }


@dataclass(frozen=True)
class PhysicalService:
    """One authoritative data service for a physical owner."""

    service_id: str
    physical_owner: str
    resource_id: str
    read_bandwidth_gb_s: float
    write_bandwidth_gb_s: float
    read_latency_ns: float = 0.0
    write_latency_ns: float = 0.0
    transaction_granularity: int = 256
    queue_depth: int = 1
    parallel_lanes: int = 1
    latency_scope: str = "service"
    efficiency: float = 1.0
    service_model: str = "analytical"
    storage_id: str = ""
    component: Optional[ComponentSpec] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("service_id", "physical_owner", "resource_id", "latency_scope"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")
        _non_negative(self.read_bandwidth_gb_s, "read_bandwidth_gb_s")
        _non_negative(self.write_bandwidth_gb_s, "write_bandwidth_gb_s")
        _non_negative(self.read_latency_ns, "read_latency_ns")
        _non_negative(self.write_latency_ns, "write_latency_ns")
        _positive_int(self.transaction_granularity, "transaction_granularity")
        _positive_int(self.queue_depth, "queue_depth")
        _positive_int(self.parallel_lanes, "parallel_lanes")
        if not 0 < float(self.efficiency) <= 1:
            raise ValueError("efficiency must be in (0, 1]")

    @property
    def bandwidth_gb_s(self) -> float:
        return min(self.read_bandwidth_gb_s, self.write_bandwidth_gb_s)

    def price(
        self,
        kind: AccessKind,
        byte_count: int,
        *,
        transaction_bytes: Optional[int] = None,
        service_model: Optional[str] = None,
        page_offset_bytes: Optional[int] = None,
    ) -> Mapping[str, object]:
        kind = AccessKind(kind)
        _non_negative_int(byte_count, "byte_count")
        if kind is AccessKind.COPY:
            raise ValueError("COPY must be expanded before billing")
        read = kind is AccessKind.READ
        if page_offset_bytes is not None:
            _non_negative_int(page_offset_bytes, "page_offset_bytes")
        bandwidth = self.read_bandwidth_gb_s if read else self.write_bandwidth_gb_s
        if byte_count and bandwidth <= 0:
            raise ValueError(f"storage service {self.service_id} requires a positive {kind.value.lower()} bandwidth")
        if (self.component is not None
                and (self.component.metadata.get("nand_media") is not None
                     or self.component.metadata.get("hbf_media") is not None)):
            from .hbf_media import nand_media_service
            return nand_media_service(
                self.component,
                byte_count,
                read,
                page_offset_bytes=page_offset_bytes,
            )
        model = self.service_model if service_model is None else service_model
        if model not in {"analytical", "serialized", "overlapped"}:
            raise ValueError("memory_service_model must be analytical, serialized or overlapped")
        if model != "analytical" and self.component is not None:
            granularity = _non_negative_int(self.component.metadata.get("transfer_granularity_bytes", 0), "transfer_granularity_bytes")
            if self.component.is_active_memory:
                granularity = self.transaction_granularity
            transaction_bytes = transaction_bytes or granularity or max(1, byte_count)
        result = memory_service(
            read_bytes=byte_count if read else 0,
            write_bytes=0 if read else byte_count,
            bandwidth_gb_s=max(self.read_bandwidth_gb_s, self.write_bandwidth_gb_s, 1e-30),
            read_bandwidth_gb_s=self.read_bandwidth_gb_s or 1e-30,
            write_bandwidth_gb_s=self.write_bandwidth_gb_s or 1e-30,
            read_latency_ns=self.read_latency_ns,
            write_latency_ns=self.write_latency_ns,
            transaction_bytes=transaction_bytes or self.transaction_granularity,
            max_outstanding_requests=self.queue_depth,
            parallel_lanes=self.parallel_lanes,
            service_model=model,
        )
        if self.component is not None:
            metadata = self.component.metadata
            energy = metadata.get(("read" if read else "write") + "_energy_pj_per_byte",
                                  metadata.get("memory_service", {}).get("energy_pj_per_byte", 0.0))
            result["energy_pj"] = result["physical_bytes"] * _non_negative(energy, "energy_pj_per_byte")
        return result

    def bill(self, kind: AccessKind, byte_count: int) -> float:
        return float(self.price(kind, byte_count)["service_ns"])


@dataclass(frozen=True)
class LinkService:
    """An independent physical interconnect; a topology view has no service."""

    service_id: str
    physical_owner: str
    resource_id: str
    bandwidth_gb_s: float
    latency_ns: float = 0.0
    queue_depth: int = 1

    def __post_init__(self) -> None:
        for name in ("service_id", "physical_owner", "resource_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")
        _positive(self.bandwidth_gb_s, "bandwidth_gb_s")
        _non_negative(self.latency_ns, "latency_ns")
        _positive_int(self.queue_depth, "queue_depth")

    def demands(self, byte_count: int, *, energy_pj_per_byte: float = 0.0) -> Tuple[ResourceDemand, ...]:
        """Sending holds bandwidth; arrival holds one shared in-flight slot.

        Only the sender records bytes and energy. A new logical resource name
        for the same owner never creates another physical request window.
        """
        _non_negative_int(byte_count, "byte_count")
        energy = _non_negative(energy_pj_per_byte, "energy_pj_per_byte")
        if byte_count == 0:
            return ()
        send_ns = byte_count / self.bandwidth_gb_s
        return (
            ResourceDemand(self.resource_id, send_ns, bytes_moved=byte_count,
                           energy_pj=byte_count * energy),
            ResourceDemand(self.physical_owner + ".inflight",
                           send_ns + self.latency_ns),
        )

    @property
    def resource_capacities(self) -> Mapping[str, int]:
        return {self.physical_owner + ".inflight": self.queue_depth}

    def bill(self, byte_count: int) -> float:
        return max(demand.service_ns for demand in self.demands(byte_count))


@dataclass(frozen=True)
class MemoryPosition:
    service_id: str
    offset_bytes: int = 0
    resource_id: Optional[str] = None
    allocation_id: str = ""

    def __post_init__(self) -> None:
        if not self.service_id:
            raise ValueError("service_id must be non-empty")
        _non_negative_int(self.offset_bytes, "offset_bytes")


@dataclass(frozen=True)
class DataAccess:
    operation_id: str
    kind: AccessKind
    byte_count: int
    source: Optional[MemoryPosition] = None
    target: Optional[MemoryPosition] = None
    dependencies: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must be non-empty")
        kind = AccessKind(self.kind)
        object.__setattr__(self, "kind", kind)
        _non_negative_int(self.byte_count, "byte_count")
        if kind is AccessKind.READ and self.source is None:
            raise ValueError("READ requires source")
        if kind is AccessKind.WRITE and self.target is None:
            raise ValueError("WRITE requires target")
        if kind is AccessKind.COPY and (self.source is None or self.target is None):
            raise ValueError("COPY requires source and target")
        if any(not isinstance(dep, str) or not dep for dep in self.dependencies):
            raise ValueError("dependencies must be non-empty strings")


@dataclass(frozen=True)
class MotionPhase:
    operation_id: str
    kind: AccessKind
    service_id: str
    physical_owner: str
    resource_id: str
    byte_count: int
    service_ns: float
    dependencies: Tuple[str, ...] = ()
    extra_demands: Tuple[ResourceDemand, ...] = ()
    physical_bytes: Optional[int] = None
    energy_pj: float = 0.0

    @property
    def demand(self) -> ResourceDemand:
        return ResourceDemand(self.resource_id, self.service_ns, bytes_moved=self.byte_count if self.physical_bytes is None else self.physical_bytes, energy_pj=self.energy_pj)

    @property
    def demands(self) -> Tuple[ResourceDemand, ...]:
        return (self.demand,) + self.extra_demands


@dataclass(frozen=True)
class ExpandedMotion:
    operation_id: str
    phases: Tuple[MotionPhase, ...]
    resource_owners: Mapping[str, str]
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    dependencies: Tuple[str, ...] = ()

    @property
    def demands(self) -> Tuple[ResourceDemand, ...]:
        return tuple(demand for phase in self.phases for demand in phase.demands)

    def to_tasks(self, request_id: str) -> Tuple[TaskSpec, ...]:
        tasks = tuple(TaskSpec(
            phase.operation_id, request_id, phase.operation_id, TaskCategory.COMMUNICATION,
            dependencies=phase.dependencies, demands=phase.demands,
        ) for phase in self.phases)
        if not tasks or tasks[-1].task_id != self.operation_id:
            tasks += (TaskSpec(
                self.operation_id, request_id, self.operation_id, TaskCategory.COMMUNICATION,
                dependencies=(tasks[-1].task_id,) if tasks else self.dependencies,
            ),)
        return tasks


def resolve_service(
    component: ComponentSpec,
    profile: Any = None,
    *,
    service_id: Optional[str] = None,
) -> PhysicalService:
    """Resolve one component service; calibration never replaces its physical cap.

    ScenarioConfig supplies ``memory_service`` after migrating legacy profiles.
    Direct ComponentSpec callers use the same component fields and explicit
    effective overrides. A directional zero remains unknown, including HBF writes.
    """
    metadata = component.metadata
    resolved = metadata.get("memory_service", {})
    if not isinstance(resolved, Mapping):
        raise ValueError("memory_service must be a mapping")
    profile = profile if profile is not None else metadata.get("profile", metadata.get("memory_profile"))
    physical = _positive(resolved.get("physical_bandwidth_gb_s", component.shared_bandwidth_gbps / 8.0), "component physical bandwidth")
    efficiency = float(_value(profile, "efficiency", default=metadata.get("efficiency", 1.0)))
    if not math.isfinite(efficiency) or not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    measured = _value(profile, "measured_effective_bandwidth_gb_s", default=metadata.get("measured_effective_bandwidth_gb_s"))
    shared = resolved.get("bandwidth_gb_s")
    if shared is None:
        shared = measured if measured is not None else _value(profile, "effective_bandwidth_gb_s")
    if shared is None:
        shared = float(_value(profile, "bandwidth_gb_s", default=physical)) * efficiency
    shared = _positive(shared, "effective bandwidth_gb_s")
    if shared > physical + 1e-12:
        raise ValueError("effective bandwidth cannot exceed component physical bandwidth")

    def direction_bandwidth(direction: str) -> float:
        ceiling = float(resolved.get("physical_" + direction + "_bandwidth_gb_s", component.directional_bandwidth_gbps(direction) / 8.0))
        raw = resolved.get(direction + "_bandwidth_gb_s")
        if raw is None:
            raw = _value(profile, "effective_" + direction + "_bandwidth_gb_s")
        if raw is None:
            raw = measured
        if raw is None:
            raw = _value(profile, direction + "_bandwidth_gb_s")
            raw = (float(raw) * efficiency if raw is not None else min(shared, ceiling * efficiency))
        value = _non_negative(raw, direction + "_bandwidth_gb_s")
        if value > min(physical, ceiling) + 1e-12:
            raise ValueError(f"effective {direction} bandwidth cannot exceed component physical bandwidth")
        return value

    def parameter(name: str, *aliases: str, default: Any) -> Any:
        return _value(resolved, name, *aliases, default=_value(metadata, name, *aliases, default=_value(profile, name, *aliases, default=default)))

    owner = str(resolved.get("physical_owner") or metadata.get("physical_owner") or default_memory_resource_id(component))
    model = str(parameter("service_model", "memory_service_model", default="analytical" if component.is_active_memory else "serialized"))
    if model not in {"analytical", "serialized", "overlapped"}:
        raise ValueError("memory_service_model must be analytical, serialized or overlapped")
    granularity = parameter("transaction_bytes", "transfer_granularity_bytes", default=256)
    # Legacy endpoint granularity=0 means no padding, not a zero transaction.
    _non_negative_int(granularity, "transaction_bytes")
    return PhysicalService(
        service_id=service_id or str(resolved.get("service_id") or metadata.get("service_id") or component.component_id),
        physical_owner=owner,
        resource_id=str(resolved.get("resource_id") or metadata.get("memory_resource_id") or metadata.get("resource_id") or owner),
        read_bandwidth_gb_s=direction_bandwidth("read"),
        write_bandwidth_gb_s=direction_bandwidth("write"),
        read_latency_ns=_non_negative(parameter("read_latency_ns", default=0.0), "read_latency_ns"),
        write_latency_ns=_non_negative(parameter("write_latency_ns", default=0.0), "write_latency_ns"),
        transaction_granularity=granularity or 256,
        queue_depth=parameter("max_outstanding_requests", "queue_depth", default=32 if component.is_active_memory else 1),
        parallel_lanes=parameter("parallel_lanes", default=1),
        efficiency=efficiency,
        service_model=model,
        storage_id=str(metadata.get("physical_storage_id") or component.component_id),
        component=component,
    )


def resolve_link_service(
    link: LinkSpec,
    source: ComponentSpec,
    target: ComponentSpec,
) -> Optional[LinkService]:
    """Return a service only for a real interconnect, never a local view."""

    local_view = (
        str(link.metadata.get("bandwidth_source", "")).strip().lower()
        in {"component", "memory_component", "shared_component"}
    )
    if link.metadata.get("service_ref") or local_view:
        memories = [component for component in (source, target) if component.is_storage]
        reference = link.metadata.get("service_ref")
        if reference:
            memories = [component for component in memories if reference in {
                component.component_id, component.component_id + ".access",
                component.metadata.get("service_id"),
                component.metadata.get("memory_service", {}).get("service_id"),
            }]
        if len(memories) != 1:
            raise ValueError(f"link {link.link_id} service_ref must identify one connected storage service")
        memory = memories[0]
        service = resolve_service(memory)
        resource = link.metadata.get("bandwidth_resource_id")
        if resource is not None and resource not in {service.resource_id, service.physical_owner}:
            raise ValueError(f"link {link.link_id} bandwidth_resource_id conflicts with referenced memory service")
        if link.bandwidth_gbps > 0 and not math.isclose(link.bandwidth_gbps, memory.shared_bandwidth_gbps):
            raise ValueError(f"link {link.link_id} bandwidth conflicts with referenced memory service")
        if link.latency_ns > 0:
            raise ValueError(f"link {link.link_id} service_ref cannot declare extra latency")
        return None
    bandwidth_gbps = float(link.bandwidth_gbps)
    if bandwidth_gbps <= 0:
        raise ValueError(f"link {link.link_id} requires positive bandwidth")
    owner = str(link.metadata.get("physical_owner", link.link_id))
    resource = str(link.metadata.get("resource_id", link.link_id))
    return LinkService(link.link_id, owner, resource, bandwidth_gbps / 8.0, float(link.latency_ns), int(link.metadata.get("queue_depth", 1)))


def expand_access(
    access: DataAccess,
    services: Mapping[str, PhysicalService],
    *,
    links: Mapping[Tuple[str, str], LinkService] = (),
) -> ExpandedMotion:
    """Expand READ/WRITE/COPY into physical phases and explicit dependencies."""

    if (
        access.kind is AccessKind.COPY
        and access.source is not None
        and access.target is not None
        and (services[access.source.service_id].storage_id or services[access.source.service_id].service_id)
        == (services[access.target.service_id].storage_id or services[access.target.service_id].service_id)
        and access.source.offset_bytes == access.target.offset_bytes
        and access.source.allocation_id
        and access.source.allocation_id == access.target.allocation_id
    ):
        # An alias of the exact same range is already resident in the target
        # view.  Keep this a real zero-cost operation instead of charging a
        # read and write against the shared owner.
        return ExpandedMotion(access.operation_id, (), {}, dependencies=access.dependencies)

    def phase(kind: AccessKind, position: MemoryPosition, deps: Tuple[str, ...]) -> MotionPhase:
        service = services[position.service_id]
        bill = service.price(
            kind,
            access.byte_count,
            page_offset_bytes=position.offset_bytes,
        )
        return MotionPhase(access.operation_id, kind, service.service_id, service.physical_owner,
                           service.resource_id, access.byte_count, float(bill["service_ns"]), deps,
                           physical_bytes=int(bill.get("physical_bytes", access.byte_count)),
                           energy_pj=float(bill.get("energy_pj", 0.0)))

    if access.kind is AccessKind.READ:
        phases = (phase(AccessKind.READ, access.source, access.dependencies),)
    elif access.kind is AccessKind.WRITE:
        phases = (phase(AccessKind.WRITE, access.target, access.dependencies),)
    else:
        read = phase(AccessKind.READ, access.source, access.dependencies)
        deps = access.dependencies + (read.operation_id + ".read",)
        read = MotionPhase(read.operation_id + ".read", read.kind, read.service_id, read.physical_owner, read.resource_id, read.byte_count, read.service_ns, read.dependencies, physical_bytes=read.physical_bytes, energy_pj=read.energy_pj)
        phases_list = [read]
        link = links.get((access.source.service_id, access.target.service_id)) if hasattr(links, "get") else None
        if link is not None:
            link_demands = link.demands(access.byte_count)
            phases_list.append(MotionPhase(
                access.operation_id + ".link", AccessKind.COPY, link.service_id,
                link.physical_owner, link.resource_id, access.byte_count, link_demands[0].service_ns, deps,
                extra_demands=link_demands[1:],
            ))
            deps = (access.operation_id + ".link",)
        write = phase(AccessKind.WRITE, access.target, deps)
        phases_list.append(MotionPhase(access.operation_id + ".write", write.kind, write.service_id, write.physical_owner, write.resource_id, write.byte_count, write.service_ns, deps, physical_bytes=write.physical_bytes, energy_pj=write.energy_pj))
        phases = tuple(phases_list)
    owners = {}
    for item in phases:
        previous = owners.get(item.resource_id)
        if previous is not None and previous != item.physical_owner:
            raise ValueError(
                "resource {} has conflicting physical owners {} and {}".format(
                    item.resource_id, previous, item.physical_owner
                )
            )
        owners[item.resource_id] = item.physical_owner
    capacities = {}
    for item in phases:
        for demand in item.extra_demands:
            # Inflight slots are owned by the physical link service, so two
            # logical aliases of one link share one queue window.
            owners[demand.resource_id] = item.physical_owner + ".inflight"
            capacities[demand.resource_id] = link.queue_depth
    return ExpandedMotion(access.operation_id, phases, owners, capacities, access.dependencies)


def expand_accesses(
    accesses: Iterable[DataAccess],
    services: Mapping[str, PhysicalService],
    *,
    links: Mapping[Tuple[str, str], LinkService] = (),
) -> Tuple[ExpandedMotion, ...]:
    """Expand a batch once per operation id.

    Frontends may describe one physical operation from both an operator and a
    DMA view.  The operation id is the stable identity; the first description
    is retained only if later descriptions have the same physical semantics.
    """

    result = []
    seen = {}
    for access in accesses:
        signature = (access.kind, access.byte_count, access.source, access.target, access.dependencies)
        if access.operation_id in seen:
            if seen[access.operation_id] != signature:
                raise ValueError(f"conflicting descriptions for operation {access.operation_id}")
            continue
        seen[access.operation_id] = signature
        result.append(expand_access(access, services, links=links))
    return tuple(result)


# Friendly names for callers that prefer nouns over verbs.
price_access = expand_access
resolve_memory_service = resolve_service



@dataclass(frozen=True)
class EndpointService:
    name: str
    demands: Tuple[ResourceDemand, ...]
    metadata: Mapping[str, object]


def endpoint_service(
    component: ComponentSpec,
    byte_count: int,
    *,
    read: bool,
    name: str,
    page_offset_bytes: Optional[int] = None,
) -> Optional[EndpointService]:
    _non_negative_int(byte_count, "byte_count")
    bandwidth = float(component.directional_bandwidth_gbps("read" if read else "write"))
    if bandwidth <= 0:
        # Offload media are active transfer endpoints.  Silently omitting
        # a missing direction used to make an HBF with unknown write
        # bandwidth accept state/KV writes at zero cost.  Active HBM/DRAM
        # interfaces may rely on their declared topology link or global
        # memory profile, but HBF/SSD media must fail closed when a real
        # transfer uses an unknown direction.
        if (
            byte_count > 0
            and component.normalized_kind
            in OFFLOAD_STORAGE_COMPONENT_KINDS
        ):
            direction = "read" if read else "write"
            raise ValueError(
                "storage component {} requires a positive {} bandwidth "
                "for this transfer".format(component.component_id, direction)
            )
        return None
    direction = "read" if read else "write"
    # Opt-in OCP-style cold-page accounting.  The link still carries the
    # host-visible request bytes; this endpoint demand charges the
    # physical page traffic and media latency separately.  We intentionally
    # keep one logical endpoint demand: a shared physical owner cannot
    # accept two demands from the same task, and the diagnostic metadata
    # retains RMW read bytes for write requests.
    if (component.normalized_kind in OFFLOAD_STORAGE_COMPONENT_KINDS
            and (component.metadata.get("nand_media") is not None
                 or component.metadata.get("hbf_media") is not None)):
        if byte_count == 0:
            return None
        from .hbf_media import nand_media_service
        media = nand_media_service(
            component,
            byte_count,
            read,
            page_offset_bytes=page_offset_bytes,
        )
        physical_bytes = media["physical_read_bytes"] if read else media["physical_bytes"]
        return EndpointService(
            name="{}.{}.{}.cold_page".format(name, component.component_id, direction),
            demands=(ResourceDemand(
                resource_id="component.{}.{}".format(component.component_id, direction),
                service_ns=media["service_ns"],
                bytes_moved=physical_bytes,
                energy_pj=media["energy_pj"],
            ),),
            metadata={
                "event_kind": "memory_{}".format(direction),
                "component_id": component.component_id,
                "bytes": byte_count,
                "transferred_bytes": media["host_transfer_bytes"],
                "physical_bytes": media["physical_bytes"],
                "nand_media": media,
                **({"hbf_media": media}
                   if component.metadata.get("hbf_media") is not None else {}),
                "transfer_granularity_bytes": media["media_page_bytes"],
                "transactions": media["command_count"],
                "max_outstanding_requests": media["command_queue_depth"],
                "latency_ns": (media["page_read_latency_ns"] if read
                               else media["page_program_latency_ns"]),
                "latency_batches": media["media_waves"],
                "bandwidth_service_ns": media["host_service_ns"],
                "latency_service_ns": media["media_read_service_ns"] if read else media["service_ns"],
                "physical_kind": component.normalized_kind,
                "access_mode": component.metadata.get("access_mode", "default"),
                "memory_service_model": media["version"],
                "timing_evidence": "ANALYTICAL",
            },
        )
    physical_service = resolve_service(component)
    latency = physical_service.read_latency_ns if read else physical_service.write_latency_ns
    granularity = _non_negative_int(component.metadata.get("transfer_granularity_bytes", 0), "transfer_granularity_bytes")
    transaction_count = math.ceil(byte_count / granularity) if granularity and byte_count else int(byte_count > 0)
    transferred_bytes = transaction_count * granularity if granularity else byte_count
    if component.is_active_memory:
        granularity = physical_service.transaction_granularity
        transaction_count = math.ceil(byte_count / granularity)
        transferred_bytes = transaction_count * granularity
    max_outstanding = physical_service.queue_depth
    parallel_lanes = physical_service.parallel_lanes
    request_window = max_outstanding * parallel_lanes
    latency_batches = math.ceil(transaction_count / request_window) if transaction_count else 0
    service_model = physical_service.service_model
    if service_model == "analytical":
        transferred_bytes = byte_count
        granularity = physical_service.transaction_granularity
        transaction_count = math.ceil(byte_count / granularity)
        latency_batches = math.ceil(transaction_count / request_window) if transaction_count else 0
    bandwidth_gb_s = physical_service.read_bandwidth_gb_s if read else physical_service.write_bandwidth_gb_s
    billed = physical_service.price(
        AccessKind.READ if read else AccessKind.WRITE, byte_count,
        transaction_bytes=(physical_service.transaction_granularity if service_model == "analytical"
                           else granularity or max(1, byte_count)),
        service_model=service_model,
    )
    bandwidth_ns = billed["bandwidth_service_ns"]
    latency_ns = billed["latency_service_ns"]
    service_ns = billed["service_ns"]
    energy_key = "{}_energy_pj_per_byte".format(direction)
    energy = _non_negative(component.metadata.get(energy_key, component.metadata.get("memory_service", {}).get("energy_pj_per_byte", 0.0)), energy_key)
    if service_model in {"analytical", "overlapped"}:
        bottleneck = "latency" if latency_ns > bandwidth_ns else "bandwidth"
        if latency_ns == bandwidth_ns:
            bottleneck = "bandwidth_and_latency"
    else:
        bottleneck = "serialized_bandwidth_and_latency"
    throughput_metrics = realtime_memory_metrics(
        byte_count,
        service_ns,
        physical_bytes=transferred_bytes,
        # Component fields are Gbit/s; the shared helper reports GB/s.
        bandwidth_ceiling_gb_s=bandwidth_gb_s,
        queue_wait_ns=0.0,
        request_window_utilization=min(
            1.0, transaction_count / float(request_window)
        ) if transaction_count else 0.0,
        bottleneck=bottleneck,
    )
    return EndpointService(
        name="{}.{}.{}".format(name, component.component_id, direction),
        demands=(
            ResourceDemand(
                resource_id=(physical_service.resource_id if component.is_active_memory
                             else "component.{}.{}".format(component.component_id, direction)),
                service_ns=service_ns,
                bytes_moved=transferred_bytes,
                energy_pj=transferred_bytes * energy,
            ),
        ),
        metadata={
            "event_kind": "memory_{}".format(direction),
            "component_id": component.component_id,
            "bytes": byte_count,
            "transferred_bytes": transferred_bytes,
            "transfer_granularity_bytes": granularity,
            "transactions": transaction_count,
            "max_outstanding_requests": max_outstanding,
            "parallel_lanes": parallel_lanes,
            "request_window": request_window,
            "effective_outstanding": billed["effective_outstanding"],
            "latency_batches": latency_batches,
            "latency_ns": latency,
            "physical_kind": component.normalized_kind,
            "access_mode": component.metadata.get("access_mode", "default"),
            "memory_service_model": service_model,
            "physical_service_id": physical_service.service_id,
            "physical_owner": physical_service.physical_owner,
            "physical_resource_id": physical_service.resource_id,
            "access_kind": "READ" if read else "WRITE",
            "bandwidth_service_ns": bandwidth_ns,
            "latency_service_ns": latency_ns,
            **throughput_metrics,
            "estimated_internal_wait_ns": billed["estimated_internal_wait_ns"],
            "timing_evidence": "ANALYTICAL",
        },
    )



# Public spelling used by communication and cost model callers.
shared_memory_service = memory_service
