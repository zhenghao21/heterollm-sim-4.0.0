"""Small, shared physical data-access service.

This module deliberately stops at service generation.  It does not schedule
tasks or choose routes; callers can lower the returned phases into the
existing :class:`ResourceDemand`/``TaskSpec`` contracts.
"""

from __future__ import annotations

import math
import copy
import contextvars
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import EvidenceStatus, ResourceDemand, TaskCategory, TaskSpec
from .ir import ComponentSpec, LinkSpec, OFFLOAD_STORAGE_COMPONENT_KINDS, default_memory_resource_id
from .memory_service import realtime_memory_metrics


@dataclass
class _PhysicalRuntime:
    core: Any
    clock_ns: float = 0.0
    signature: str = ""


@dataclass
class PhysicalRuntimeContext:
    """State for one simulation run's physical devices.

    Runtime ownership is explicit and context-local.  A context never silently
    merges two configurations under one physical owner.
    """

    config: Any = None
    physical_owner: Optional[str] = None
    runtimes: dict[str, _PhysicalRuntime] = field(default_factory=dict)
    timeline: Any = None
    _committed_owners: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.timeline is None:
            from .memory_transfer import ResourceTimeline
            self.timeline = ResourceTimeline()
        if self.config is not None or self.physical_owner is not None:
            if self.config is None or not self.physical_owner:
                raise ValueError("config and physical_owner must be provided together")
            self.runtime(self.config, self.physical_owner)

    def runtime(self, config: Any, owner: str) -> _PhysicalRuntime:
        from .memory_types import DramConfig, NandConfig, parse_physical_memory_config
        if not isinstance(config, (DramConfig, NandConfig)):
            config = parse_physical_memory_config(config)
        kind = str(getattr(config.kind, "value", config.kind))
        signature = kind + ":" + repr(config)
        key = str(owner)
        current = self.runtimes.get(key)
        if current is not None:
            if current.signature != signature:
                raise ValueError(
                    f"physical owner {owner} was configured with multiple physical geometries"
                )
            return current
        from .dram_core import DramCore
        from .nand_core import NandCore
        metadata = dict(config.metadata)
        if isinstance(config, DramConfig):
            metadata.setdefault("bank_resource_prefix", f"{key}:dram:bank")
            metadata.setdefault("command_resource_prefix", f"{key}:dram:command")
            metadata.setdefault("data_resource_prefix", f"{key}:dram:data")
            config = replace(config, metadata=metadata)
        else:
            metadata.setdefault("array_resource_prefix", f"{key}:nand:array")
            metadata.setdefault("parallel_resource_prefix", f"{key}:nand:parallel")
            metadata.setdefault("channel_resource_prefix", f"{key}:nand:channel")
            metadata.setdefault("host_resource_id", f"{key}:nand:host")
            config = replace(config, metadata=metadata)
        core = DramCore(config, self.timeline) if isinstance(config, DramConfig) else NandCore(config, self.timeline)
        current = _PhysicalRuntime(core=core, signature=signature)
        self.runtimes[key] = current
        return current

    def preview_runtime(self, config: Any, owner: str) -> _PhysicalRuntime:
        from .memory_types import DramConfig, NandConfig, parse_physical_memory_config
        if not isinstance(config, (DramConfig, NandConfig)):
            config = parse_physical_memory_config(config)
        kind = str(getattr(config.kind, "value", config.kind))
        from .dram_core import DramCore
        from .nand_core import NandCore
        core = DramCore(config) if isinstance(config, DramConfig) else NandCore(config)
        return _PhysicalRuntime(core=core, signature=kind + ":" + repr(config))

    def snapshot(self) -> dict[str, Any]:
        import copy as _copy
        return _copy.deepcopy((self.runtimes, self.timeline.__dict__, self._committed_owners))

    def restore(self, snapshot: dict[str, Any]) -> None:
        runtimes, timeline_state, owners = snapshot
        self.runtimes.clear(); self.runtimes.update(runtimes)
        lane_ref = getattr(self.timeline, "lane_available", None)
        self.timeline.__dict__.clear(); self.timeline.__dict__.update(timeline_state)
        if lane_ref is not None and "lane_available" in timeline_state:
            lane_ref.clear(); lane_ref.update(timeline_state["lane_available"])
            self.timeline.lane_available = lane_ref
        for active in self.runtimes.values():
            active.core.timeline = self.timeline
        self._committed_owners.clear(); self._committed_owners.update(owners)


_CURRENT_PHYSICAL_CONTEXT: contextvars.ContextVar[PhysicalRuntimeContext | None] = contextvars.ContextVar(
    "heterollm_sim_physical_context", default=None
)


def current_physical_runtime_context() -> PhysicalRuntimeContext:
    context = _CURRENT_PHYSICAL_CONTEXT.get()
    if context is None:
        context = PhysicalRuntimeContext()
        _CURRENT_PHYSICAL_CONTEXT.set(context)
    return context


def reset_physical_runtimes() -> None:
    """Start a fresh context-local physical simulation timeline."""
    _CURRENT_PHYSICAL_CONTEXT.set(PhysicalRuntimeContext())


def _physical_runtime(
    config: Any,
    owner: str,
    context: Optional[PhysicalRuntimeContext] = None,
    *,
    preview: bool = False,
) -> _PhysicalRuntime:
    context = context or current_physical_runtime_context()
    return context.preview_runtime(config, owner) if preview else context.runtime(config, owner)


def resolve_physical_task(
    task: TaskSpec,
    runtime: PhysicalRuntimeContext,
    arrival_ns: float,
) -> TaskSpec:
    """Commit one physical task at its event-kernel arrival time.

    Planner pricing remains a preview.  This function is the single formal
    submission point used by the event kernel; the core's own resource
    timeline supplies the task completion time and per-resource busy demand.
    """
    metadata = dict(task.metadata)
    raw_config = metadata.get("physical_memory_config")
    access = metadata.get("memory_access")
    accesses = metadata.get("memory_accesses", access)
    if raw_config is None or not isinstance(accesses, (Mapping, tuple, list)) or not accesses:
        raise ValueError("physical task requires physical_memory_config and memory_access")
    if is_dataclass(raw_config):
        raw_config = asdict(raw_config)
    if not isinstance(raw_config, Mapping):
        raise ValueError("physical_memory_config must be a DRAM/NAND config or mapping")
    from .memory_types import AccessRequest, DramConfig, Operation, parse_physical_memory_config
    config = parse_physical_memory_config(raw_config)
    is_dram = isinstance(config, DramConfig)
    if isinstance(arrival_ns, bool) or not isinstance(arrival_ns, (int, float)) or arrival_ns < 0:
        raise ValueError("physical task arrival_ns must be non-negative")
    if isinstance(accesses, Mapping):
        accesses = (accesses,)
    # Validate every descriptor before creating a core or reserving a resource.
    validated = []
    for access in accesses:
        if not isinstance(access, Mapping):
            raise ValueError("physical memory_access entries must be mappings")
        operation = Operation(str(access.get("operation", "")).lower())
        address, byte_count = access.get("address"), access.get("byte_count")
        if isinstance(address, bool) or not isinstance(address, int) or address < 0:
            raise ValueError("physical task address must be a non-negative integer")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("physical task byte_count must be a positive integer")
        validated.append((access, operation, address, byte_count))
    snapshot = runtime.snapshot() if len(validated) > 1 else None
    placeholder_ids = set()
    for access, _op, _address, _bytes in validated:
        for key in ("resource_id", "physical_resource_id", "physical_owner"):
            value = access.get(key)
            if value:
                placeholder_ids.add(str(value))
    results = []
    next_arrival = float(arrival_ns)
    try:
      for index, (access, operation, address, byte_count) in enumerate(validated):
        owner = str(access.get("physical_owner") or metadata.get("physical_owner") or task.task_id)
        active = runtime.runtime(config, owner)
        request = AccessRequest(f"{task.request_id or task.task_id}:{index}", operation, address, byte_count, next_arrival)
        submit = getattr(active.core, "submit", active.core.execute)
        result = submit(request)
        active.clock_ns = max(active.clock_ns, result.completion_ns)
        results.append(result)
        next_arrival = result.completion_ns
    except Exception:
        if snapshot is not None:
            runtime.restore(snapshot)
        raise
    result = results[-1]
    counters = dict(result.counters)
    resource_busy = {}
    resource_bytes = {}
    resource_intervals = {}
    resource_interval_payloads = {}
    resource_last = {}
    for item in results:
        for resource_id, value in item.counters.get("resource_busy_ns", {}).items():
            resource_busy[resource_id] = resource_busy.get(resource_id, 0.0) + value
        for resource_id, value in item.counters.get("resource_bytes", {}).items():
            resource_bytes[resource_id] = resource_bytes.get(resource_id, 0) + value
        for resource_id, value in item.counters.get("resource_intervals", {}).items():
            resource_intervals.setdefault(resource_id, []).extend(value)
        for resource_id, value in item.counters.get("resource_interval_payloads", {}).items():
            resource_interval_payloads.setdefault(resource_id, []).extend(value)
        resource_last.update(item.counters.get("resource_last_intervals", {}))
    counters.update({"resource_busy_ns": resource_busy, "resource_bytes": resource_bytes,
                     "resource_intervals": resource_intervals, "resource_last_intervals": resource_last,
                     "resource_interval_payloads": resource_interval_payloads,
                     "operation_count": len(results)})
    for key in (
        "logical_bytes", "physical_bytes", "physical_read_bytes", "physical_write_bytes",
        "host_transfer_bytes", "internal_transfer_bytes", "pages_read", "pages_programmed",
        "erase_operations", "burst_count", "row_hits", "row_misses", "row_conflicts",
    ):
        if key == "physical_bytes":
            counters[key] = sum(item.transfer_bytes for item in results)
        elif key == "logical_bytes":
            counters[key] = sum(item.logical_bytes for item in results)
        else:
            values = [item.counters.get(key) for item in results]
            if any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                counters[key] = sum(float(value or 0) for value in values)
    demands = tuple(ResourceDemand(str(resource_id), float(duration), bytes_moved=int(resource_bytes.get(resource_id, 0)))
                    for resource_id, duration in sorted(resource_busy.items()) if float(duration) > 0)
    if not demands:
        demands = (ResourceDemand(owner, result.latency_ns, bytes_moved=result.transfer_bytes),)
    metadata.update({
        "physical_execution": {
            **counters,
            "logical_bytes": sum(item.logical_bytes for item in results),
            "physical_bytes": sum(item.transfer_bytes for item in results),
            "service_ns": next_arrival - float(arrival_ns),
            "arrival_ns": float(arrival_ns),
            "completion_ns": next_arrival,
            "operation": results[-1].operation.value if len(results) == 1 else "read_write",
        },
        "physical_arrival_ns": float(arrival_ns),
        "physical_completion_ns": next_arrival,
        "physical_resource_intervals": tuple(
            item for result in results for item in result.stages
            if item.name not in {"READ_PIPELINE", "WRITE_PIPELINE"}
        ),
        "physical_resource_last_intervals": resource_last,
        "physical_demands_resource_ids": tuple(sorted(resource_busy)),
        "physical_placeholder_resource_ids": tuple(sorted(placeholder_ids)),
    })
    # Keep unrelated compute/link demands. Only preview demands for resources
    # identified by this transaction are replaced by core-owned reservations.
    memory_ids = set(metadata["physical_demands_resource_ids"]) | placeholder_ids
    preserved = tuple(d for d in task.demands if d.resource_id not in memory_ids)
    metadata["physical_nonmemory_demand_ids"] = tuple(d.resource_id for d in preserved)
    return replace(task, demands=preserved + demands, metadata=metadata)


def is_physical_task(task: TaskSpec) -> bool:
    metadata = task.metadata
    config = metadata.get("physical_memory_config")
    accesses = metadata.get("memory_accesses", metadata.get("memory_access"))
    # Endpoint metadata always carries these optional keys.  It becomes a
    # formal physical transaction only when both values are present and the
    # descriptor is non-empty; an explicitly supplied malformed descriptor is
    # rejected instead of silently falling back to analytical pricing.
    if config is None:
        if accesses not in (None, (), []):
            raise ValueError("physical task requires physical_memory_config and memory_access")
        return False
    if not isinstance(accesses, Mapping) and not isinstance(accesses, (tuple, list)):
        raise ValueError("physical task requires physical_memory_config and memory_access")
    if not accesses:
        raise ValueError("physical task requires a non-empty memory access descriptor")
    return True


def physical_task_demands(task: TaskSpec) -> Tuple[ResourceDemand, ...]:
    """Return preview demands that remain externally scheduled for a physical task."""
    metadata = task.metadata
    physical_ids = set(str(item) for item in metadata.get("physical_demands_resource_ids", ()))
    physical_ids.update(str(item) for item in metadata.get("physical_placeholder_resource_ids", ()))
    accesses = metadata.get("memory_accesses", metadata.get("memory_access"))
    if isinstance(accesses, Mapping):
        accesses = (accesses,)
    for access in accesses or ():
        if isinstance(access, Mapping):
            for key in ("resource_id", "physical_resource_id"):
                value = access.get(key)
                if value:
                    physical_ids.add(str(value))
    return tuple(demand for demand in task.demands if str(demand.resource_id) not in physical_ids)


class AccessKind(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    COPY = "COPY"
    ERASE = "ERASE"


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
        runtime: Optional[PhysicalRuntimeContext] = None,
        arrival_ns: Optional[float] = None,
        preview: bool = False,
    ) -> Mapping[str, object]:
        kind = AccessKind(kind)
        _non_negative_int(byte_count, "byte_count")
        if kind is AccessKind.COPY:
            raise ValueError("COPY must be expanded before billing")
        read = kind is AccessKind.READ
        if page_offset_bytes is not None:
            _non_negative_int(page_offset_bytes, "page_offset_bytes")
        if page_offset_bytes is None and self.component is not None:
            page_offset_bytes = self.component.metadata.get("memory_access_offset_bytes")
            if page_offset_bytes is not None:
                _non_negative_int(page_offset_bytes, "memory_access_offset_bytes")
        bandwidth = self.read_bandwidth_gb_s if read else self.write_bandwidth_gb_s
        if (byte_count and kind is not AccessKind.ERASE and bandwidth <= 0
                and not (self.component is not None and self.component.metadata.get("physical_memory_config") is not None)):
            raise ValueError(f"storage service {self.service_id} requires a positive {kind.value.lower()} bandwidth")
        # The component configuration is the single physical transaction contract.
        # and therefore gets explicit array and data-path stages.
        if self.component is not None and self.component.metadata.get("physical_memory_config") is not None:
            from .memory_types import AccessRequest, DramConfig, Operation, parse_physical_memory_config
            raw_config = self.component.metadata["physical_memory_config"]
            if is_dataclass(raw_config):
                raw_config = asdict(raw_config)
            if not isinstance(raw_config, Mapping):
                raise ValueError("physical_memory_config must be a DRAM/NAND config or mapping")
            if kind is AccessKind.ERASE:
                raw_kind = str(getattr(raw_config.get("kind", ""), "value", raw_config.get("kind", ""))).strip().upper().replace("-", "")
                if raw_kind in {"DDR", "LPDDR", "HBM", "GDDR", "GDDR6", "GDDR6X", "GDDR7"}:
                    raise ValueError("DRAM does not support ERASE operations")
            if page_offset_bytes is None and self.component.metadata.get("memory_access_offset_bytes") is None:
                raise ValueError("physical_memory_config requires an explicit memory address")
            if byte_count == 0:
                return {
                    "logical_bytes": 0, "physical_bytes": 0,
                    "physical_read_bytes": 0, "physical_write_bytes": 0,
                    "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
                    "service_ns": 0.0, "actual_bandwidth_gb_s": 0.0,
                    "bandwidth_ceiling_gb_s": 0.0,
                    "operation": kind.value.lower(),
                    "timing_model": "physical_transaction_v1",
                    "energy_pj": 0.0,
                    "pages_touched": 0,
                }
            config = parse_physical_memory_config(raw_config)
            is_dram = isinstance(config, DramConfig)
            operation = Operation.READ if kind is AccessKind.READ else Operation.WRITE if kind is AccessKind.WRITE else Operation.ERASE
            if arrival_ns is not None:
                _non_negative(arrival_ns, "arrival_ns")
            context = runtime if runtime is not None else current_physical_runtime_context()
            active_runtime = _physical_runtime(config, self.physical_owner, context, preview=preview or runtime is None)
            request_arrival_ns = active_runtime.clock_ns if arrival_ns is None else float(arrival_ns)
            request = AccessRequest(
                f"{self.service_id}:physical",
                operation,
                0 if page_offset_bytes is None else page_offset_bytes,
                max(1, byte_count),
                arrival_ns=request_arrival_ns,
            )
            submit = getattr(active_runtime.core, "submit", active_runtime.core.execute)
            result = submit(request)
            if not preview and runtime is not None:
                active_runtime.clock_ns = max(active_runtime.clock_ns, result.completion_ns)
            counters = dict(result.counters)
            counters.update({
                "logical_bytes": result.logical_bytes,
                "physical_bytes": result.transfer_bytes,
                "physical_read_bytes": result.physical_read_bytes,
                "physical_write_bytes": result.physical_write_bytes,
                "host_transfer_bytes": result.host_transfer_bytes,
                "internal_transfer_bytes": result.internal_transfer_bytes,
                "service_ns": result.latency_ns,
                "arrival_ns": result.arrival_ns,
                "completion_ns": result.completion_ns,
                "actual_bandwidth_gb_s": result.actual_bandwidth_gb_s,
                "bandwidth_ceiling_gb_s": result.bandwidth_ceiling_gb_s,
                "operation": operation.value,
                "timing_model": "physical_transaction_v1",
                "energy_pj": 0.0,
                "pages_touched": result.counters.get("page_count", 0) if not is_dram else 0,
                "write_completion": "media_program_complete" if operation is Operation.WRITE else "n/a",
            })
            return counters
        if kind is AccessKind.ERASE:
            raise ValueError("ERASE requires physical_memory_config")
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

    def price_batch(
        self,
        requests,
        *,
        start_ns=0.0,
        state=None,
        runtime: Optional[PhysicalRuntimeContext] = None,
    ):
        """Price an explicit NAND request batch and return its next queue state."""
        if self.component is not None and self.component.metadata.get("physical_memory_config") is not None:
            from .memory_types import AccessRequest, Operation, parse_physical_memory_config
            from dataclasses import asdict, is_dataclass
            raw = self.component.metadata["physical_memory_config"]
            if is_dataclass(raw):
                raw = asdict(raw)
            config = parse_physical_memory_config(raw)
            explicit_context = runtime or (state if isinstance(state, PhysicalRuntimeContext) else None)
            context = explicit_context or current_physical_runtime_context()
            active_runtime = _physical_runtime(
                config,
                self.physical_owner,
                context,
                preview=explicit_context is None,
            )
            parsed = []
            for index, item in enumerate(requests):
                op = str(item.get("operation", "")).lower()
                if "address" in item:
                    address = item["address"]
                elif "page_offset_bytes" in item:
                    address = item["page_offset_bytes"]
                else:
                    raise ValueError("physical batch request requires an explicit address")
                if isinstance(address, bool) or not isinstance(address, int) or address < 0:
                    raise ValueError("physical batch request address must be a non-negative integer")
                if isinstance(item.get("byte_count"), bool) or not isinstance(item.get("byte_count"), int):
                    raise ValueError("physical batch request byte_count must be an integer")
                parsed.append(AccessRequest(
                    str(item.get("request_id", f"request-{index}")),
                    Operation(op), int(address), int(item["byte_count"]),
                    float(item.get("arrival_ns", max(start_ns, active_runtime.clock_ns))),
                ))
            batch = active_runtime.core.run(parsed)
            if explicit_context is not None:
                active_runtime.clock_ns = max(active_runtime.clock_ns, batch.last_completion_ns)
            return {
                "model": "physical_transaction_queue_v1",
                "requests": tuple(batch.requests),
                "request_count": len(batch.requests),
                "logical_bytes": batch.logical_bytes,
                "physical_bytes": batch.transfer_bytes,
                "host_transfer_bytes": batch.host_transfer_bytes,
                "internal_transfer_bytes": batch.internal_transfer_bytes,
                "physical_read_bytes": batch.physical_read_bytes,
                "physical_write_bytes": batch.physical_write_bytes,
                "pages_read": batch.pages_read,
                "pages_programmed": batch.pages_programmed,
                "erase_operations": batch.erase_operations,
                "queue_wait_ns": batch.queue_wait_ns,
                "service_ns": batch.duration_ns,
                "end_ns": batch.last_completion_ns,
                "actual_bandwidth_gb_s": batch.actual_bandwidth_gb_s,
                "bandwidth_ceiling_gb_s": batch.bandwidth_ceiling_gb_s,
            }
        if self.component is None or self.component.metadata.get("physical_memory_config") is None:
            raise ValueError("price_batch requires physical_memory_config")
        raise RuntimeError("unreachable physical batch branch")


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
        if kind in (AccessKind.WRITE, AccessKind.ERASE) and self.target is None:
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
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
            metadata=dict(phase.metadata),
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
        metadata: dict[str, Any] = {}
        raw_config = service.component.metadata.get("physical_memory_config") if service.component is not None else None
        if raw_config is not None:
            metadata.update({
                "physical_memory_config": raw_config,
                "memory_access": {
                    "operation": kind.value.lower(),
                    "address": position.offset_bytes,
                    "byte_count": access.byte_count,
                    "physical_owner": service.physical_owner,
                    "resource_id": service.resource_id,
                },
            })
        return MotionPhase(access.operation_id, kind, service.service_id, service.physical_owner,
                           service.resource_id, access.byte_count, float(bill["service_ns"]), deps,
                           physical_bytes=int(bill.get("physical_bytes", access.byte_count)),
                           energy_pj=float(bill.get("energy_pj", 0.0)), metadata=metadata)

    if access.kind is AccessKind.READ:
        phases = (phase(AccessKind.READ, access.source, access.dependencies),)
    elif access.kind is AccessKind.WRITE:
        phases = (phase(AccessKind.WRITE, access.target, access.dependencies),)
    elif access.kind is AccessKind.ERASE:
        phases = (phase(AccessKind.ERASE, access.target, access.dependencies),)
    else:
        read = phase(AccessKind.READ, access.source, access.dependencies)
        deps = access.dependencies + (read.operation_id + ".read",)
        read = MotionPhase(read.operation_id + ".read", read.kind, read.service_id, read.physical_owner, read.resource_id, read.byte_count, read.service_ns, read.dependencies, physical_bytes=read.physical_bytes, energy_pj=read.energy_pj, metadata=read.metadata)
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
        phases_list.append(MotionPhase(access.operation_id + ".write", write.kind, write.service_id, write.physical_owner, write.resource_id, write.byte_count, write.service_ns, deps, physical_bytes=write.physical_bytes, energy_pj=write.energy_pj, metadata=write.metadata))
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
    dram_address_bytes: Optional[int] = None,
    operation: Optional[str] = None,
    runtime: Optional[PhysicalRuntimeContext] = None,
    arrival_ns: Optional[float] = None,
    preview: bool = False,
) -> Optional[EndpointService]:
    """Lower one memory access using the canonical physical transaction model."""
    _non_negative_int(byte_count, "byte_count")
    address = page_offset_bytes if page_offset_bytes is not None else dram_address_bytes
    if address is None:
        address = component.metadata.get("memory_access_offset_bytes")
    if address is None and component.metadata.get("physical_memory_config") is not None:
        raise ValueError("physical_memory_config requires an explicit memory address")
    if address is None:
        address = 0
    _non_negative_int(address, "memory access address")
    op_name = ("read" if read else "write") if operation is None else str(operation).lower()
    if op_name not in {"read", "write", "erase"}:
        raise ValueError("operation must be read, write or erase")
    if op_name == "read" and not read:
        raise ValueError("read operation requires read=True")
    if op_name == "write" and read:
        raise ValueError("write operation requires read=False")
    if byte_count > 0 and component.metadata.get("physical_memory_config") is None:
        declared_bandwidth = float(component.directional_bandwidth_gbps("read" if read else "write"))
        if declared_bandwidth <= 0:
            return None
    service = resolve_service(component)
    billed = service.price(
        AccessKind.READ if op_name == "read" else AccessKind.WRITE if op_name == "write" else AccessKind.ERASE,
        byte_count,
        page_offset_bytes=address,
        runtime=runtime,
        arrival_ns=arrival_ns,
        preview=preview,
    )
    physical = int(billed.get("physical_bytes", 0))
    if op_name == "erase":
        physical = 0
    demand = ResourceDemand(
        resource_id=service.resource_id,
        service_ns=float(billed.get("service_ns", 0.0)),
        bytes_moved=physical,
        energy_pj=float(billed.get("energy_pj", 0.0)),
    )
    endpoint_metadata = {
            "event_kind": f"memory_{op_name}",
            "component_id": component.component_id,
            "bytes": byte_count,
            "address": address,
            "physical_memory_config": component.metadata.get("physical_memory_config"),
            "physical_execution": dict(billed),
            "physical_bytes": physical,
            "transferred_bytes": int(billed.get("host_transfer_bytes", billed.get("logical_bytes", byte_count))),
            "physical_kind": component.normalized_kind,
            "physical_service_id": service.service_id,
            "physical_owner": service.physical_owner,
            "physical_resource_id": service.resource_id,
            "access_kind": op_name.upper(),
            "timing_evidence": "ANALYTICAL",
        }
    if component.metadata.get("physical_memory_config") is not None:
        endpoint_metadata["memory_access"] = {
            "operation": op_name,
            "address": address,
            "byte_count": byte_count,
            "physical_owner": service.physical_owner,
            "resource_id": service.resource_id,
        }
    return EndpointService(
        name=f"{name}.{component.component_id}.{op_name}",
        demands=(demand,),
        metadata=endpoint_metadata,
    )


# Public spelling used by communication and cost model callers.
shared_memory_service = memory_service
