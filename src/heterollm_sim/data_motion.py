"""Small, shared physical data-access service.

This module deliberately stops at service generation.  It does not schedule
tasks or choose routes; callers can lower the returned phases into the
existing :class:`ResourceDemand`/``TaskSpec`` contracts.
"""

from __future__ import annotations

import math
import copy
import contextvars
from bisect import bisect_left
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import EvidenceStatus, ResourceDemand, TaskCategory, TaskSpec
from .ir import ComponentSpec, LinkSpec, OFFLOAD_STORAGE_COMPONENT_KINDS, default_memory_resource_id
from .memory_service import realtime_memory_metrics


class _OverlapCompletionIndex:
    """Online interval overlap maxima for physical access ordering.

    Each update stores a completion timestamp on the canonical segment-tree
    cover of one byte range.  A query returns the maximum value among all
    stored ranges intersecting the requested range.  The node-local value is
    included on partial traversal because every range stored there covers the
    whole node interval, hence it intersects whenever that node intersects.
    This preserves the old pairwise ordering rule while reducing each access
    from an O(previous_accesses) scan to O(log endpoints).
    """

    def __init__(self, endpoints: Sequence[int]) -> None:
        self._endpoints = tuple(endpoints)
        self._segments = max(1, len(self._endpoints) - 1)
        self._values = [0.0] * (4 * self._segments + 4)
        self._subtree = [0.0] * (4 * self._segments + 4)

    def _indices(self, address: int, byte_count: int) -> Tuple[int, int]:
        start = bisect_left(self._endpoints, address)
        end = bisect_left(self._endpoints, address + byte_count)
        # Validated accesses contribute both endpoints, so this is normally
        # an exact segment range.  Keep the bounds checked for defensive use.
        return max(0, min(self._segments - 1, start)), max(1, min(self._segments, end))

    def update(self, address: int, byte_count: int, completion_ns: float) -> None:
        left, right = self._indices(address, byte_count)
        if right <= left:
            return

        def visit(node: int, lo: int, hi: int) -> None:
            if right <= lo or hi <= left:
                return
            if left <= lo and hi <= right:
                self._values[node] = max(self._values[node], completion_ns)
            else:
                mid = (lo + hi) // 2
                visit(node * 2, lo, mid)
                visit(node * 2 + 1, mid, hi)
            child_max = 0.0
            if hi - lo > 1:
                child_max = max(self._subtree[node * 2], self._subtree[node * 2 + 1])
            self._subtree[node] = max(self._values[node], child_max)

        visit(1, 0, self._segments)

    def query(self, address: int, byte_count: int) -> float:
        left, right = self._indices(address, byte_count)
        if right <= left:
            return 0.0

        def visit(node: int, lo: int, hi: int) -> float:
            if right <= lo or hi <= left:
                return 0.0
            if left <= lo and hi <= right:
                return self._subtree[node]
            mid = (lo + hi) // 2
            return max(
                self._values[node],
                visit(node * 2, lo, mid),
                visit(node * 2 + 1, mid, hi),
            )

        return visit(1, 0, self._segments)


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
    allocators: dict[str, Any] = field(default_factory=dict)
    timeline: Any = None
    # Exact event/replay runs retain expanded stage/mapping details. Aggregate
    # cohort estimates set this false to avoid allocating those transient
    # tuples; scalar timing and traffic counters remain unchanged.
    capture_details: bool = True
    _committed_owners: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.timeline is None:
            from .memory_transfer import ResourceTimeline
            self.timeline = ResourceTimeline()
        if not isinstance(self.capture_details, bool):
            raise ValueError("capture_details must be a boolean")
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
        from .memory_allocator import PhysicalAddressAllocator
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
        core = (
            DramCore(config, self.timeline, capture_details=self.capture_details)
            if isinstance(config, DramConfig)
            else NandCore(config, self.timeline, capture_details=self.capture_details)
        )
        current = _PhysicalRuntime(core=core, signature=signature)
        self.runtimes[key] = current
        capacity = int(getattr(config, "effective_capacity_bytes", getattr(config, "capacity_bytes", 0)) or 0)
        if capacity > 0:
            alignment = int(getattr(config, "burst_bytes", 64) or 64)
            self.allocators.setdefault(key, PhysicalAddressAllocator(capacity, alignment))
        return current

    def preview_runtime(
        self,
        config: Any,
        owner: str,
        *,
        capture_details: Optional[bool] = None,
    ) -> _PhysicalRuntime:
        from .memory_types import DramConfig, NandConfig, parse_physical_memory_config
        if not isinstance(config, (DramConfig, NandConfig)):
            config = parse_physical_memory_config(config)
        kind = str(getattr(config.kind, "value", config.kind))
        from .dram_core import DramCore
        from .nand_core import NandCore
        # Preserve detailed preview bills by default.  Summary-only planner
        # callers can disable transient stage/mapping tuples explicitly.
        details = True if capture_details is None else capture_details
        if not isinstance(details, bool):
            raise ValueError("capture_details must be a boolean")
        core = (
            DramCore(config, capture_details=details)
            if isinstance(config, DramConfig)
            else NandCore(config, capture_details=details)
        )
        return _PhysicalRuntime(core=core, signature=kind + ":" + repr(config))

    def snapshot(self) -> dict[str, Any]:
        """Capture a transaction-local rollback point without copying history.

        Physical tasks are submitted one at a time.  The old implementation
        deep-copied every runtime, allocator and the complete timeline before
        each task; once burst interval detail accumulated this duplicated the
        whole run for every in-flight transaction.  Core state and allocator
        entries are small mutable indexes, while timeline interval payloads
        are replaced at ``metrics_snapshot`` and can therefore be retained by
        reference for rollback.
        """
        runtime_state = {}
        for owner, active in self.runtimes.items():
            core = active.core
            # Core indexes (DRAM bank rows / NAND array clocks) grow with the
            # physical address stream.  Deep-copying them for every access
            # recreated the original quadratic memory behaviour.  Their
            # immutable configuration and scalar admission state are enough
            # for the normal commit path; mutable indexes remain owned by the
            # live core and are only consulted again after a successful task.
            core_state = {}
            for key, value in vars(core).items():
                if key in {"config", "timeline"}:
                    continue
                if key == "_banks" and isinstance(value, dict):
                    # Bank rows are mutable dataclasses; copy the small bank
                    # index so a failed transaction cannot leak row state.
                    core_state[key] = {
                        name: copy.copy(bank) for name, bank in value.items()
                    }
                elif key in {"_array_ready", "_buffer_ready"} and isinstance(value, dict):
                    # NAND readiness values are scalars, so a shallow dict
                    # copy is sufficient and avoids copying timeline history.
                    core_state[key] = dict(value)
                elif key == "_inflight" and isinstance(value, list):
                    core_state[key] = list(value)
                else:
                    core_state[key] = value
            core_state["config"] = core.config
            runtime_state[owner] = {
                "active": active,
                "clock_ns": active.clock_ns,
                "core": core_state,
            }
        allocator_state = {
            owner: dict(getattr(allocator, "_allocations", {}))
            for owner, allocator in self.allocators.items()
        }
        timeline = self.timeline
        timeline_state = {
            "ready_ns": dict(timeline.ready_ns),
            "directions": dict(timeline.directions),
            "busy_ns": dict(timeline.busy_ns),
            "bytes_moved": dict(timeline.bytes_moved),
            "last_intervals": dict(timeline.last_intervals),
            # submit() replaces these containers before appending details.
            "_intervals": timeline._intervals,
            "_interval_limit": timeline._interval_limit,
            "_interval_count": timeline._interval_count,
            "_intervals_truncated": timeline._intervals_truncated,
            "_touched": timeline._touched,
            "_interval_payloads": timeline._interval_payloads,
            "lane_available": {
                resource: tuple(lanes)
                for resource, lanes in timeline.lane_available.items()
            },
        }
        return {
            "runtimes": dict(self.runtimes),
            "runtime_state": runtime_state,
            "allocators": dict(self.allocators),
            "allocator_state": allocator_state,
            "timeline": timeline_state,
            "committed_owners": set(self._committed_owners),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.runtimes.clear()
        self.runtimes.update(snapshot["runtimes"])
        self.allocators.clear()
        self.allocators.update(snapshot["allocators"])
        for owner, state in snapshot.get("runtime_state", {}).items():
            active = self.runtimes.get(owner)
            if active is None:
                continue
            active.clock_ns = state["clock_ns"]
            active.core.__dict__.clear()
            active.core.__dict__.update(state["core"])
            active.core.timeline = self.timeline
        for owner, allocations in snapshot.get("allocator_state", {}).items():
            allocator = self.allocators.get(owner)
            if allocator is not None and hasattr(allocator, "_allocations"):
                allocator._allocations.clear()
                allocator._allocations.update(allocations)
        timeline_state = snapshot["timeline"]
        timeline = self.timeline
        for name in ("ready_ns", "directions", "busy_ns", "bytes_moved", "last_intervals"):
            target = getattr(timeline, name)
            target.clear()
            target.update(timeline_state[name])
        timeline._intervals = timeline_state["_intervals"]
        timeline._interval_limit = timeline_state["_interval_limit"]
        timeline._interval_count = timeline_state["_interval_count"]
        timeline._intervals_truncated = timeline_state["_intervals_truncated"]
        timeline._touched = timeline_state["_touched"]
        timeline._interval_payloads = timeline_state["_interval_payloads"]
        timeline.lane_available.clear()
        timeline.lane_available.update({
            resource: list(lanes)
            for resource, lanes in timeline_state["lane_available"].items()
        })
        self._committed_owners.clear()
        self._committed_owners.update(snapshot.get("committed_owners", ()))


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
    capture_details: Optional[bool] = None,
) -> _PhysicalRuntime:
    context = context or current_physical_runtime_context()
    return (
        context.preview_runtime(
            config, owner, capture_details=capture_details
        )
        if preview
        else context.runtime(config, owner)
    )


def _owner_physical_config(metadata: Mapping[str, Any], owner: str, primary: Any) -> Any:
    """Resolve a declared physical geometry without borrowing another owner."""
    from .memory_types import DramConfig, NandConfig, parse_physical_memory_config

    configurations = metadata.get("physical_memory_configs")
    if configurations is None:
        return primary
    if not isinstance(configurations, Mapping):
        raise ValueError("physical_memory_configs must map owners to configurations")
    raw = configurations.get(owner)
    if raw is None:
        raise ValueError("physical owner {} has no declared configuration".format(owner))
    if isinstance(raw, (DramConfig, NandConfig)):
        return raw
    return parse_physical_memory_config(raw)


def register_physical_allocations(task: TaskSpec, runtime: PhysicalRuntimeContext, *, config=None) -> None:
    """Materialize declared buffers into the owner-scoped run allocator."""
    metadata = task.metadata
    raw_config = config if config is not None else metadata.get("physical_memory_config")
    from .memory_types import DramConfig, NandConfig, parse_physical_memory_config
    if not isinstance(raw_config, (DramConfig, NandConfig)):
        raw_config = parse_physical_memory_config(raw_config)
    declarations = metadata.get("physical_allocations", ())
    if isinstance(declarations, Mapping):
        declarations = (declarations,)
    if declarations is None:
        declarations = ()
    if not isinstance(declarations, (tuple, list)):
        raise ValueError("physical_allocations must be a mapping or sequence")
    # Planner-created descriptors carry a stable buffer identity and an
    # allocation extent, but older task metadata did not promote a separate
    # declaration table. Derive declarations from those descriptors once;
    # this is registration, never an address guess.
    if not declarations:
        contract = metadata.get("stateful_l2")
        default_owner = str(metadata.get("physical_owner") or
                            (contract.get("memory_resource") if isinstance(contract, Mapping) else "") or "")
        raw_accesses = metadata.get("memory_accesses", metadata.get("memory_access"))
        if isinstance(raw_accesses, Mapping):
            raw_accesses = (raw_accesses,)
        if isinstance(contract, Mapping):
            # The physical preview may already be cache-trimmed. Allocate the
            # logical working set before the cache uses it, using the same
            # buffer identities and preserving explicit address/alias facts.
            previews = {(str(row.get("physical_owner") or default_owner), str(row.get("buffer_id")), row.get("allocation_generation", row.get("generation", 0))): row
                        for row in raw_accesses or () if isinstance(row, Mapping) and row.get("buffer_id")}
            cache_accesses = []
            for row in contract.get("accesses", ()):
                buffer_id = row["buffer_id"]
                if buffer_id.startswith("@tasklocal"):
                    buffer_id = task.task_id + buffer_id[10:]
                preview = previews.get((str(row.get("physical_owner") or default_owner), buffer_id, row.get("allocation_generation", 0)), {})
                cache_accesses.append({**preview, **row, "buffer_id": buffer_id,
                                       "byte_count": row["size_bytes"]})
            raw_accesses = tuple(cache_accesses)
        if isinstance(raw_accesses, (tuple, list)):
            derived = {}
            for access in raw_accesses:
                if not isinstance(access, Mapping):
                    continue
                buffer_id = access.get("buffer_id") or access.get("tensor_id")
                if not buffer_id:
                    continue
                # Descriptor fields are part of the physical-address
                # contract.  Do not coerce floats/strings with ``int`` here:
                # ``64.9`` silently becoming ``64`` would register and later
                # access a different extent than the caller declared.
                offset = _descriptor_non_negative_int(
                    access.get("offset_bytes", access.get("offset", 0)),
                    "offset_bytes",
                )
                size = _descriptor_non_negative_int(
                    access.get("byte_count", access.get("size_bytes", 0)),
                    "byte_count",
                )
                declared_extent = access.get("allocation_size_bytes")
                if declared_extent is None:
                    declared_extent = access.get("buffer_size_bytes")
                extent = (
                    _descriptor_non_negative_int(declared_extent, "allocation_size_bytes")
                    if declared_extent is not None
                    else offset + size
                )
                if declared_extent is None and isinstance(contract, Mapping):
                    line_bytes = _positive_descriptor_int(
                        contract["line_bytes"], "line_bytes"
                    )
                    extent = ((extent + line_bytes - 1) // line_bytes) * line_bytes
                generation = _descriptor_non_negative_int(
                    access.get("generation", access.get("allocation_generation", 0)),
                    "generation",
                )
                owner = str(access.get("physical_owner") or default_owner)
                key = (owner, str(buffer_id), generation)
                previous = derived.get(key)
                if previous is None or extent > previous["size_bytes"]:
                    declaration = {"buffer_id": str(buffer_id), "size_bytes": extent,
                                   "generation": generation, "physical_owner": owner,
                                   "inferred": declared_extent is None}
                    if access.get("alias_of") is not None:
                        declaration.update({
                            "alias_of": str(access["alias_of"]),
                            "alias_generation": _descriptor_non_negative_int(
                                access.get("alias_generation", generation),
                                "alias_generation",
                            ),
                            "alias_offset_bytes": _descriptor_non_negative_int(
                                access.get("alias_offset_bytes", 0),
                                "alias_offset_bytes",
                            ),
                        })
                    if str(access.get("address_source", "")).startswith("explicit"):
                        explicit_address = access.get("address")
                        if not isinstance(explicit_address, int) or explicit_address < offset:
                            raise ValueError("explicit physical address is before buffer offset")
                        declaration["address"] = explicit_address - offset
                    derived[key] = declaration
            declarations = tuple(derived.values())
    pending = sorted(
        declarations,
        key=lambda item: 1 if isinstance(item, Mapping) and item.get("alias_of") else 0,
    )
    while pending:
        deferred = []
        progress = 0
        for declaration in pending:
            if not isinstance(declaration, Mapping):
                raise ValueError("physical allocation entries must be mappings")
            alias_of = declaration.get("alias_of")
            owner = str(declaration.get("physical_owner") or metadata.get("physical_owner") or "")
            generation = _descriptor_non_negative_int(
                declaration.get("generation", declaration.get("allocation_generation", 0)),
                "generation",
            )
            alias_generation = declaration.get("alias_generation")
            if alias_of is not None and alias_generation is None:
                alias_generation = generation
            elif alias_generation is not None:
                alias_generation = _descriptor_non_negative_int(
                    alias_generation, "alias_generation"
                )
            if alias_of is not None and owner:
                existing_allocator = runtime.allocators.get(owner)
                if existing_allocator is not None and existing_allocator.get_allocation(str(alias_of), alias_generation) is None:
                    deferred.append(declaration)
                    continue
            try:
                _register_physical_allocation(
                    declaration, metadata, runtime,
                    _owner_physical_config(metadata, owner, raw_config),
                )
                progress += 1
            except ValueError:
                raise
        if not deferred:
            break
        if progress == 0:
            raise ValueError("physical allocation alias target is not registered")
        pending = deferred


def _register_physical_allocation(declaration, metadata, runtime, raw_config):
    """Register one declaration after its optional alias target exists."""
    if not isinstance(declaration, Mapping):
        raise ValueError("physical allocation entries must be mappings")
    buffer_id = declaration.get("buffer_id")
    if not buffer_id:
        raise ValueError("physical allocation requires buffer_id")
    owner = str(declaration.get("physical_owner") or metadata.get("physical_owner") or "")
    if not owner:
        raise ValueError("physical allocation requires physical_owner")
    size_bytes = declaration.get("size_bytes", declaration.get("buffer_size_bytes"))
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("physical allocation requires positive size_bytes")
    runtime.runtime(raw_config, owner)
    allocator = runtime.allocators.get(owner)
    if allocator is None:
        raise ValueError("physical owner has no address allocator: {}".format(owner))
    allocator.allocate(
        str(buffer_id), size_bytes,
        _descriptor_non_negative_int(
            declaration.get("generation", declaration.get("allocation_generation", 0)),
            "generation",
        ),
        alias_of=declaration.get("alias_of"),
        address=declaration.get("address", declaration.get("base_address")),
        alias_generation=declaration.get("alias_generation"),
        alias_offset_bytes=_descriptor_non_negative_int(
            declaration.get("alias_offset_bytes", 0), "alias_offset_bytes"
        ),
        inferred=_descriptor_bool(declaration.get("inferred", False), "inferred"),
    )


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
    energy_per_byte = _non_negative(
        metadata.get("physical_energy_pj_per_byte", 0.0), "physical_energy_pj_per_byte"
    )
    raw_energy_by_owner = metadata.get("physical_energy_pj_per_byte_by_owner")
    if raw_energy_by_owner is not None and not isinstance(raw_energy_by_owner, Mapping):
        raise ValueError("physical_energy_pj_per_byte_by_owner must map owners to rates")
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
    if isinstance(arrival_ns, bool) or not isinstance(arrival_ns, (int, float)) or arrival_ns < 0:
        raise ValueError("physical task arrival_ns must be non-negative")
    if isinstance(accesses, Mapping):
        accesses = (accesses,)
    owner_energy = {}
    for item in accesses:
        if not isinstance(item, Mapping):
            raise ValueError("physical memory_access entries must be mappings")
        owner = str(item.get("physical_owner") or metadata.get("physical_owner") or "")
        if not owner:
            raise ValueError("physical access requires physical_owner")
        if raw_energy_by_owner is None:
            owner_energy[owner] = energy_per_byte
        else:
            if owner not in raw_energy_by_owner:
                raise ValueError("physical owner {} has no declared energy rate".format(owner))
            owner_energy[owner] = _non_negative(
                raw_energy_by_owner[owner],
                "physical_energy_pj_per_byte_by_owner[{}]".format(owner),
            )
    if raw_energy_by_owner is None and len(owner_energy) > 1:
        raise ValueError("multi-owner physical task requires per-owner energy rates")
    if metadata.get("physical_memory_configs") is None and len(owner_energy) > 1:
        raise ValueError("multi-owner physical task requires per-owner configurations")
    register_physical_allocations(task, runtime, config=config)

    # Validate every descriptor before creating a core or reserving a resource.
    # A physical address may be omitted only when the buffer has an explicit
    # allocation; this keeps L2 victims and partial line fills tied to it.
    declarations = metadata.get("physical_allocations", ())
    if isinstance(declarations, Mapping):
        declarations = (declarations,)
    if declarations is None:
        declarations = ()
    if not isinstance(declarations, (tuple, list)):
        raise ValueError("physical_allocations must be a mapping or sequence")
    validated = []
    for access in accesses:
        if not isinstance(access, Mapping):
            raise ValueError("physical memory_access entries must be mappings")
        operation = Operation(str(access.get("operation", "")).lower())
        address, byte_count = access.get("address"), access.get("byte_count")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("physical task byte_count must be a positive integer")
        owner = str(access.get("physical_owner") or metadata.get("physical_owner") or "")
        if not owner:
            raise ValueError("physical access requires physical_owner")
        buffer_id = access.get("buffer_id") or access.get("tensor_id")
        address_source = str(access.get("address_source", ""))
        if buffer_id and address_source and not address_source.startswith("explicit"):
            allocator = runtime.allocators.get(owner)
            if allocator is None:
                raise ValueError("physical owner has no address allocator: {}".format(owner))
            generation = _descriptor_non_negative_int(
                access.get("generation", access.get("allocation_generation", 0)),
                "generation",
            )
            offset = _descriptor_non_negative_int(
                access.get("offset_bytes", access.get("offset", 0)),
                "offset_bytes",
            )
            address = allocator.address(str(buffer_id), offset, byte_count, generation)
        elif address is None:
            if not buffer_id:
                raise ValueError("physical access without address requires buffer_id")
            generation = _descriptor_non_negative_int(
                access.get("generation", access.get("allocation_generation", 0)),
                "generation",
            )
            allocator = runtime.allocators.get(owner)
            if allocator is None:
                raise ValueError("physical owner has no address allocator: {}".format(owner))
            if not any(a.buffer_id == str(buffer_id) and a.generation == generation
                       for a in allocator.allocations()):
                raise ValueError("physical access has no declared allocation: {}".format(buffer_id))
            address = allocator.address(
                str(buffer_id),
                _descriptor_non_negative_int(
                    access.get("offset_bytes", access.get("offset", 0)),
                    "offset_bytes",
                ),
                byte_count,
                generation,
            )
        if isinstance(address, bool) or not isinstance(address, int) or address < 0:
            raise ValueError("physical task address must be a non-negative integer")
        access_config = _owner_physical_config(metadata, owner, config)
        if address + byte_count > access_config.capacity_bytes:
            raise ValueError(
                "physical access exceeds owner {} capacity".format(owner)
            )
        validated.append((access, operation, address, byte_count, owner, access_config))
    # Previous accesses only constrain a new request when at least one side
    # writes.  The old implementation scanned every prior descriptor for
    # every descriptor (O(n**2)); a large line-expanded projection can contain
    # tens of thousands of descriptors.  Compress address endpoints once and
    # keep range-max indexes for all accesses and writes.  Each query and
    # update is O(log n), while submission order and per-access timing stay
    # unchanged.
    endpoints_by_owner = {}
    for _access, _op, address, byte_count, owner, _config in validated:
        endpoints_by_owner.setdefault(owner, set()).update((address, address + byte_count))
    all_completion = {
        owner: _OverlapCompletionIndex(sorted(endpoints))
        for owner, endpoints in endpoints_by_owner.items()
    }
    write_completion = {
        owner: _OverlapCompletionIndex(sorted(endpoints))
        for owner, endpoints in endpoints_by_owner.items()
    }

    snapshot = runtime.snapshot()
    placeholder_ids = set()
    for access, _op, _address, _bytes, _owner, _config in validated:
        for key in ("resource_id", "physical_resource_id", "physical_owner"):
            value = access.get(key)
            if value:
                placeholder_ids.add(str(value))
    results = []
    resolved_accesses = []
    try:
      for index, (access, operation, address, byte_count, owner, access_config) in enumerate(validated):
        active = runtime.runtime(access_config, owner)
        access_arrival = float(arrival_ns)
        if operation is Operation.WRITE:
            access_arrival = max(
                access_arrival,
                all_completion[owner].query(address, byte_count),
            )
        else:
            access_arrival = max(
                access_arrival,
                write_completion[owner].query(address, byte_count),
            )
        request = AccessRequest(f"{task.request_id or task.task_id}:{index}", operation, address, byte_count, access_arrival)
        submit = getattr(active.core, "submit", active.core.execute)
        result = submit(request)
        active.clock_ns = max(active.clock_ns, result.completion_ns)
        results.append(result)
        all_completion[owner].update(address, byte_count, result.completion_ns)
        if operation is Operation.WRITE:
            write_completion[owner].update(address, byte_count, result.completion_ns)
        resolved_accesses.append({**dict(access), "address": address})
      completion_ns = max(item.completion_ns for item in results)
    except Exception:
        if snapshot is not None:
            runtime.restore(snapshot)
        raise
    result = max(results, key=lambda item: item.completion_ns)
    counters = dict(result.counters)
    resource_busy = {}
    resource_bytes = {}
    resource_owners = {}
    resource_intervals = {}
    resource_interval_payloads = {}
    resource_last = {}
    for (access, _operation, _address, _byte_count, owner, _config), item in zip(validated, results):
        for resource_id, value in item.counters.get("resource_busy_ns", {}).items():
            resource_busy[resource_id] = resource_busy.get(resource_id, 0.0) + value
            resource_owners[resource_id] = owner
        for resource_id, value in item.counters.get("resource_bytes", {}).items():
            resource_bytes[resource_id] = resource_bytes.get(resource_id, 0) + value
        for resource_id, value in item.counters.get("resource_intervals", {}).items():
            resource_intervals.setdefault(resource_id, []).extend(value)
        for resource_id, value in item.counters.get("resource_interval_payloads", {}).items():
            resource_interval_payloads.setdefault(resource_id, []).extend(value)
        resource_last.update(item.counters.get("resource_last_intervals", {}))
    counters.update({"resource_busy_ns": resource_busy, "resource_bytes": resource_bytes,
                     "resource_owners": resource_owners,
                     "resource_intervals": resource_intervals, "resource_last_intervals": resource_last,
                     "resource_interval_payloads": resource_interval_payloads,
                     "operation_count": len(results)})
    execution_by_owner = {}
    operations_by_owner = {}
    for (_access, operation, _address, _byte_count, owner, access_config), item in zip(validated, results):
        row = execution_by_owner.setdefault(owner, {
            "kind": str(getattr(access_config.kind, "value", access_config.kind)),
            "operation_count": 0,
            "logical_bytes": 0,
            "logical_read_bytes": 0,
            "logical_write_bytes": 0,
            "physical_bytes": 0,
            "physical_read_bytes": 0,
            "physical_write_bytes": 0,
            "energy_pj": 0.0,
            "resource_busy_ns": {},
            "resource_owners": {},
            "completion_ns": float(arrival_ns),
        })
        operations_by_owner.setdefault(owner, set()).add(operation.value)
        row["operation_count"] += 1
        row["logical_bytes"] += item.logical_bytes
        row["logical_read_bytes" if operation is Operation.READ else "logical_write_bytes"] += item.logical_bytes
        row["physical_bytes"] += item.transfer_bytes
        row["physical_read_bytes"] += int(item.counters.get("physical_read_bytes", 0))
        row["physical_write_bytes"] += int(item.counters.get("physical_write_bytes", 0))
        row["completion_ns"] = max(row["completion_ns"], item.completion_ns)
        for resource_id, value in item.counters.get("resource_busy_ns", {}).items():
            row["resource_busy_ns"][resource_id] = (
                row["resource_busy_ns"].get(resource_id, 0.0) + value
            )
            row["resource_owners"][resource_id] = owner
        for key in (
            "burst_count", "row_hits", "row_misses", "row_conflicts",
            "read_write_switches", "queue_wait_ns", "refresh_wait_ns",
            "turnaround_wait_ns", "pages_read", "pages_programmed",
            "pages_touched", "media_waves", "erase_operations",
            "host_transfer_bytes", "internal_transfer_bytes",
        ):
            value = item.counters.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = row.get(key, 0) + value
    for owner, row in execution_by_owner.items():
        row["service_ns"] = row["completion_ns"] - float(arrival_ns)
        operations = operations_by_owner[owner]
        row["operation"] = next(iter(operations)) if len(operations) == 1 else "read_write"
    for key in (
        "logical_bytes", "physical_bytes", "physical_read_bytes", "physical_write_bytes",
        "host_transfer_bytes", "internal_transfer_bytes", "pages_read", "pages_programmed",
        "pages_touched", "media_waves", "erase_operations", "burst_count",
        "row_hits", "row_misses", "row_conflicts", "read_write_switches",
        "queue_wait_ns", "refresh_wait_ns", "turnaround_wait_ns",
    ):
        if key == "physical_bytes":
            counters[key] = sum(item.transfer_bytes for item in results)
        elif key == "logical_bytes":
            counters[key] = sum(item.logical_bytes for item in results)
        else:
            values = [item.counters.get(key) for item in results]
            if any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                counters[key] = sum(float(value or 0) for value in values)
    # Timing comes from the physical core. An explicitly carried energy
    # coefficient remains analytical, priced once against resolved bursts.
    demands = tuple(ResourceDemand(str(resource_id), float(duration),
                    bytes_moved=int(resource_bytes.get(resource_id, 0)),
                    energy_pj=int(resource_bytes.get(resource_id, 0))
                    * owner_energy[resource_owners[str(resource_id)]])
                    for resource_id, duration in sorted(resource_busy.items()) if float(duration) > 0)
    if not demands:
        raise ValueError("physical memory core returned no resource timing for a positive-byte access")
    for row in execution_by_owner.values():
        row["energy_pj"] = 0.0
    for demand in demands:
        owner = resource_owners[demand.resource_id]
        execution_by_owner[owner]["energy_pj"] += demand.energy_pj
    metadata.update({
        "memory_access": resolved_accesses[0] if len(resolved_accesses) == 1 else tuple(resolved_accesses),
        "memory_accesses": tuple(resolved_accesses),
        "physical_execution": {
            **counters,
            "logical_bytes": sum(item.logical_bytes for item in results),
            "logical_read_bytes": sum(item.logical_bytes for item in results if item.operation is Operation.READ),
            "logical_write_bytes": sum(item.logical_bytes for item in results if item.operation is Operation.WRITE),
            "physical_bytes": sum(item.transfer_bytes for item in results),
            "energy_pj": sum(demand.energy_pj for demand in demands),
            "service_ns": completion_ns - float(arrival_ns),
            "arrival_ns": float(arrival_ns),
            "completion_ns": completion_ns,
            "operation": results[-1].operation.value if len(results) == 1 else "read_write",
        },
        "physical_execution_by_owner": execution_by_owner,
        "physical_arrival_ns": float(arrival_ns),
        "physical_completion_ns": completion_ns,
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


def _descriptor_non_negative_int(value: Any, name: str) -> int:
    """Validate an optional integer field from physical descriptors.

    Descriptor producers may omit optional offsets/generations (represented
    as ``None``), but an explicitly supplied value must already be an integer.
    In particular, avoid ``int(value)`` because it silently truncates a
    malformed float or accepts a numeric string and changes the declared
    physical range.
    """

    if value is None:
        return 0
    return _non_negative_int(value, name)


def _positive_descriptor_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be a positive integer")
    return _positive_int(value, name)


def _descriptor_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
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
        summary_only: bool = False,
    ) -> Mapping[str, object]:
        kind = AccessKind(kind)
        from .physical_contract import require_physical_memory_config
        if self.component is None:
            raise ValueError("physical memory service requires an explicitly configured component; aggregate-cost fallback is disabled")
        require_physical_memory_config(self.component)
        _non_negative_int(byte_count, "byte_count")
        if kind is AccessKind.COPY:
            raise ValueError("COPY must be expanded before billing")
        read = kind is AccessKind.READ
        if page_offset_bytes is not None:
            _non_negative_int(page_offset_bytes, "page_offset_bytes")
        if not isinstance(summary_only, bool):
            raise TypeError("summary_only must be a boolean")
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
            active_runtime = _physical_runtime(
                config,
                self.physical_owner,
                context,
                preview=preview or runtime is None,
                capture_details=(False if summary_only else None),
            )
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
        raise ValueError("physical memory service requires physical_memory_config")

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
        if self.component is None:
            raise ValueError("price_batch requires a physical memory component")
        from .physical_contract import require_physical_memory_config
        config = require_physical_memory_config(self.component)
        if self.component is not None and self.component.metadata.get("physical_memory_config") is not None:
            from .memory_types import AccessRequest, Operation
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
    from .physical_contract import require_physical_memory_config
    require_physical_memory_config(component)
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
    compact_preview: bool = False,
) -> Optional[EndpointService]:
    """Lower one memory access using the canonical physical transaction model.

    ``compact_preview`` is for planner-authored physical tasks.  It removes
    expanded interval/payload traces from the analytical preview while
    preserving scalar counters; dispatch still recomputes the full trace.
    """
    _non_negative_int(byte_count, "byte_count")
    # Compute/controller endpoints have no DRAM/NAND array. Their pipelines
    # and local SRAM are accounted by the compute models.
    if not component.is_storage:
        return None
    address = page_offset_bytes if page_offset_bytes is not None else dram_address_bytes
    if address is None:
        address = component.metadata.get("memory_access_offset_bytes")
    from .physical_contract import require_physical_memory_config
    require_physical_memory_config(component)
    if address is None:
        raise ValueError("physical_memory_config requires an explicit memory address")
    _non_negative_int(address, "memory access address")
    op_name = ("read" if read else "write") if operation is None else str(operation).lower()
    if op_name not in {"read", "write", "erase"}:
        raise ValueError("operation must be read, write or erase")
    if op_name == "read" and not read:
        raise ValueError("read operation requires read=True")
    if op_name == "write" and read:
        raise ValueError("write operation requires read=False")
    service = resolve_service(component)
    billed = service.price(
        AccessKind.READ if op_name == "read" else AccessKind.WRITE if op_name == "write" else AccessKind.ERASE,
        byte_count,
        page_offset_bytes=address,
        runtime=runtime,
        arrival_ns=arrival_ns,
        preview=preview,
        summary_only=compact_preview,
    )
    if compact_preview and runtime is None and component.metadata.get("physical_memory_config") is not None:
        # Planner-authored endpoint tasks only need scalar preview counters.
        # The event kernel resolves the physical descriptor again at dispatch,
        # where the complete interval/payload trace is attached to the result.
        # Avoid retaining one expanded burst tuple per task in a serving
        # schedule while keeping the default endpoint API unchanged.
        heavy_keys = {
            "resource_intervals",
            "resource_interval_payloads",
            "resource_last_intervals",
            "physical_resource_intervals",
            "physical_resource_last_intervals",
        }
        billed = {
            key: value for key, value in billed.items() if key not in heavy_keys
        }
        billed["details_truncated"] = True
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
