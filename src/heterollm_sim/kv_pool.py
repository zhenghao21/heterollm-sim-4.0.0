"""Experimental paged KV-cache pool.

This module deliberately lives beside the static serving ledger.  It provides
the page ownership and migration primitives needed by the experimental
``paged_pool`` layout without changing the llama.cpp-compatible static path.
The pool owns no scheduler state: callers resize a request atomically and can
use the returned component/page statistics when lowering serving events.

The implementation is intentionally single-machine.  A page can move only
between components that are writable active/offload memory and reachable in
the supplied topology.  An external physical ledger can be supplied through
the small ``can_adjust``/``adjust``/``transfer`` interface, so allocations
participate in the serving ledger instead of maintaining a disconnected
capacity counter.  Physical migration timing is supplied explicitly through
``transfer_executor``; without that page-aware callback, migration remains
fail-closed instead of using an aggregate bandwidth estimate.  A real shared
queue returns ``KvPoolTransferPlan`` so admission can commit or cancel the
physical reservation together with the capacity ledger.
"""

from __future__ import annotations

import copy
from uuid import uuid4
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple


_OFFLOAD_STORAGE_KINDS = frozenset(
    {"hbf", "ssd", "high_io_ssd", "nvme", "flash", "storage"}
)


class KvPoolError(ValueError):
    """Base error raised for invalid or unsupported paged-pool operations."""


class KvPoolUnsupported(KvPoolError):
    """Raised when a request would require an unsupported pool capability."""


class PhysicalLedger(Protocol):
    """Minimal interface implemented by ``_PhysicalCapacityLedger``."""

    def can_adjust(self, component_id: Optional[str], delta_bytes: int) -> bool: ...

    def adjust(self, component_id: Optional[str], delta_bytes: int) -> bool: ...


@dataclass(frozen=True)
class KvPoolComponent:
    """A memory component eligible for page placement or offload.

    ``active`` distinguishes a resident KV location from a backing tier.
    HBF/SSD components should therefore be passed with ``active=False`` unless
    their IR explicitly models memory access.  ``machine_id`` is used to fail
    closed on cross-machine pools; the P6 external KV-store design is outside
    this module.
    """

    component_id: str
    capacity_bytes: int
    kind: str = "memory"
    tier: str = "active"
    active: bool = True
    writable: bool = True
    machine_id: str = "local"
    read_bandwidth_gbps: float = 0.0
    write_bandwidth_gbps: float = 0.0
    latency_ns: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id:
            raise KvPoolError("component_id must be non-empty")
        if isinstance(self.capacity_bytes, bool) or int(self.capacity_bytes) < 0:
            raise KvPoolError("component capacity_bytes must be non-negative")
        if float(self.read_bandwidth_gbps) < 0 or float(self.write_bandwidth_gbps) < 0:
            raise KvPoolError("component bandwidth must be non-negative")
        if float(self.latency_ns) < 0:
            raise KvPoolError("component latency_ns must be non-negative")
        if not self.machine_id:
            raise KvPoolError("machine_id must be non-empty")
        if not isinstance(self.active, bool) or not isinstance(self.writable, bool):
            raise KvPoolError("component active and writable must be booleans")

    @classmethod
    def from_component(cls, component: Any) -> "KvPoolComponent":
        """Project an IR component, or keep an explicit logical pool budget."""
        from .ir import ComponentSpec
        if isinstance(component, cls):
            return component
        if not isinstance(component, ComponentSpec):
            raise KvPoolError("pool components must be ComponentSpec or KvPoolComponent values")
        metadata = dict(component.metadata)
        active_value = component.is_active_memory
        return cls(
            component_id=component.component_id,
            capacity_bytes=component.capacity_bytes,
            kind=component.kind,
            tier=str(metadata.get("memory_tier", "active" if active_value else "offload")),
            active=active_value,
            writable=component.is_writable,
            machine_id=str(metadata.get("machine_id", "local")),
            read_bandwidth_gbps=component.read_bandwidth_gbps,
            write_bandwidth_gbps=component.write_bandwidth_gbps,
            latency_ns=float(metadata.get("latency_ns", 0.0)),
            metadata=metadata,
        )


@dataclass
class KvPage:
    """A physical page and its logical request/prefix metadata."""

    request_id: str
    logical_page_id: int
    token_start: int
    token_end: int
    layer_group: str
    owner_component: str
    resident_tier: str
    bytes: int
    ref_count: int = 1
    pinned: bool = False
    last_access: int = 0
    prefix_key: Optional[str] = None
    allocation_generation: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.layer_group or not self.owner_component:
            raise KvPoolError("page request_id, layer_group, and owner_component are required")
        if self.logical_page_id < 0 or self.token_start < 0 or self.token_end < self.token_start:
            raise KvPoolError("invalid page token range")
        if self.bytes <= 0 or self.ref_count < 1 or self.last_access < 0:
            raise KvPoolError("page bytes/ref_count/last_access are invalid")


@dataclass(frozen=True)
class KvPoolTransfer:
    """Timing and accounting record for migration, offload, or restore."""

    kind: str
    logical_page_id: int
    source_component: str
    target_component: str
    bytes: int
    duration_ns: float
    start_ns: Optional[float] = None
    completion_ns: Optional[float] = None


@dataclass(frozen=True)
class KvPoolTransferRequest:
    """Immutable page description passed to a physical transfer executor."""

    logical_page_id: int
    request_id: str
    layer_group: str
    source_component: str
    target_component: str
    bytes: int
    allocation_generation: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class KvPoolTransferPlan:
    """Prepared physical transfer with an optional commit/rollback pair.

    A shared physical event queue returns this plan: preparation
    reserves the event, ``commit`` publishes it after the capacity ledger
    succeeds, and ``rollback`` cancels it on any failed admission.
    """

    duration_ns: float
    start_ns: Optional[float] = None
    completion_ns: Optional[float] = None
    commit: Optional[Callable[[], None]] = field(default=None, repr=False, compare=False)
    rollback: Optional[Callable[[], None]] = field(default=None, repr=False, compare=False)
    # A staged asynchronous plan exposes the current phase end until advance
    # has submitted every downstream phase. Only a plan without advance is
    # a final completion contract that may publish the destination.
    advance: Optional[Callable[[], "KvPoolTransferPlan"]] = field(default=None, repr=False, compare=False)
    # Cancellation drains already-submitted work and releases only this copy's
    # destination. Unlike admission rollback it never rewinds shared calendars.
    cancel: Optional[Callable[[], Mapping[str, Any]]] = field(default=None, repr=False, compare=False)
    # Only the currently submitted physical phase. A closed native writer
    # can report its exact calendar footprint; unknown writers retain the
    # complete shared-calendar publication represented by None.
    native_changed_resource_ids: Optional[tuple[str, ...]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            duration = float(self.duration_ns)
        except (TypeError, ValueError) as error:
            raise KvPoolError("transfer plan duration_ns must be numeric") from error
        if not isfinite(duration) or duration < 0:
            raise KvPoolError("transfer plan duration_ns must be finite and non-negative")
        object.__setattr__(self, "duration_ns", duration)
        if self.start_ns is not None:
            try:
                start = float(self.start_ns)
            except (TypeError, ValueError) as error:
                raise KvPoolError("transfer plan start_ns must be numeric") from error
            if not isfinite(start) or start < 0:
                raise KvPoolError("transfer plan start_ns must be finite and non-negative")
            object.__setattr__(self, "start_ns", start)
        if self.completion_ns is not None:
            try:
                completion = float(self.completion_ns)
            except (TypeError, ValueError) as error:
                raise KvPoolError("transfer plan completion_ns must be numeric") from error
            if not isfinite(completion) or completion < 0:
                raise KvPoolError("transfer plan completion_ns must be finite and non-negative")
            object.__setattr__(self, "completion_ns", completion)
        if self.start_ns is not None and self.completion_ns is not None:
            if self.completion_ns < self.start_ns:
                raise KvPoolError("transfer plan completion_ns precedes start_ns")
        for name in ("commit", "rollback", "advance", "cancel"):
            hook = getattr(self, name)
            if hook is not None and not callable(hook):
                raise KvPoolError("transfer plan {} hook must be callable".format(name))
        if self.native_changed_resource_ids is not None and (
                type(self.native_changed_resource_ids) is not tuple
                or any(type(resource) is not str for resource in self.native_changed_resource_ids)):
            raise KvPoolError("native changed resource IDs must be a tuple of strings or None")


@dataclass(frozen=True)
class KvPoolResizeResult:
    success: bool
    request_id: str
    page_count: int
    allocated_pages: int = 0
    released_pages: int = 0
    error: Optional[str] = None


class _LocalLedger:
    """Physical byte ledger used when serving has not supplied one."""

    def __init__(self, limits: Mapping[str, int]) -> None:
        self.limits = {str(key): max(0, int(value)) for key, value in limits.items()}
        self.used_bytes = {key: 0 for key in self.limits}

    def can_adjust(self, component_id: Optional[str], delta_bytes: int) -> bool:
        if not component_id or delta_bytes <= 0:
            return True
        if component_id not in self.limits:
            return True
        return self.used_bytes[component_id] + int(delta_bytes) <= self.limits[component_id]

    def adjust(self, component_id: Optional[str], delta_bytes: int) -> bool:
        if not self.can_adjust(component_id, delta_bytes):
            return False
        if component_id and component_id in self.limits:
            value = self.used_bytes[component_id] + int(delta_bytes)
            if value < 0:
                raise RuntimeError("KV pool physical ledger underflow on " + component_id)
            self.used_bytes[component_id] = value
        return True

    def can_transfer(self, source: str, target: str, byte_count: int) -> bool:
        if source == target:
            return self.used_bytes.get(source, 0) >= byte_count
        return self.used_bytes.get(source, 0) >= byte_count and self.can_adjust(target, byte_count)

    def transfer(self, source: str, target: str, byte_count: int) -> bool:
        if not self.can_transfer(source, target, byte_count):
            return False
        if source != target:
            self.adjust(source, -byte_count)
            self.adjust(target, byte_count)
        return True


class _LedgerAdapter:
    def __init__(
        self,
        limits: Mapping[str, int],
        ledger: Optional[Any] = None,
    ) -> None:
        self._local = _LocalLedger(limits)
        self.ledger = self._local if ledger is None else ledger
        if not callable(getattr(self.ledger, "can_adjust", None)) or not callable(getattr(self.ledger, "adjust", None)):
            raise KvPoolError("ledger must implement can_adjust and adjust")

    @property
    def limits(self) -> Mapping[str, int]:
        return getattr(self.ledger, "limits", self._local.limits)

    @property
    def used_bytes(self) -> Mapping[str, int]:
        # Minimal external ledgers need not publish their usage. In that case
        # the pool uses its own tracked page ownership, not an empty local ledger.
        return getattr(self.ledger, "used_bytes", {})

    def can_adjust(self, component: str, delta: int) -> bool:
        return bool(self.ledger.can_adjust(component, int(delta)))

    def adjust(self, component: str, delta: int) -> bool:
        return bool(self.ledger.adjust(component, int(delta)))

    def can_transfer(self, source: str, target: str, bytes_: int) -> bool:
        method = getattr(self.ledger, "can_transfer", None)
        if method is not None:
            return bool(method(source, target, int(bytes_)))
        return self.can_adjust(source, -int(bytes_)) and self.can_adjust(target, int(bytes_))

    def adjust_many(self, deltas: Mapping[str, int]) -> bool:
        """Apply distinct physical footprints, reverting rejected callbacks."""
        if not all(self.can_adjust(component, delta) for component, delta in deltas.items()):
            return False
        method = getattr(self.ledger, "adjust_many", None)
        if method is not None:
            return bool(method(deltas))
        changed = []
        def undo():
            while changed:
                component, delta = changed.pop()
                if not self.adjust(component, -delta):
                    raise RuntimeError("KV pool ledger could not reverse failed capacity adjustment")
        try:
            for component, delta in deltas.items():
                if not self.adjust(component, delta):
                    undo()
                    return False
                changed.append((component, delta))
        except Exception:
            # Exceptions are outside the boolean callback contract. Restore
            # earlier successful entries before propagating the failure.
            undo()
            raise
        return True

    def can_transfer_sized(self, source: str, target: str, source_bytes: int, target_bytes: int) -> bool:
        if source_bytes == target_bytes:
            return self.can_transfer(source, target, source_bytes)
        # The pool already owns this source page. Minimal external ledgers
        # need only expose can_adjust/adjust, not a readable usage mapping.
        return (self.can_adjust(source, -source_bytes)
                and self.can_adjust(target, target_bytes))

    def transfer_sized(self, source: str, target: str, source_bytes: int, target_bytes: int) -> bool:
        if source_bytes == target_bytes:
            return self.transfer(source, target, source_bytes)
        if not self.can_transfer_sized(source, target, source_bytes, target_bytes):
            return False
        return self.adjust_many({source: -source_bytes, target: target_bytes})

    def transfer(self, source: str, target: str, bytes_: int) -> bool:
        method = getattr(self.ledger, "transfer", None)
        if method is not None:
            return bool(method(source, target, int(bytes_)))
        if not self.can_transfer(source, target, bytes_):
            return False
        # A minimal ledger has already been checked for both sides.
        # Release first so an active->backing move does not transiently exceed
        # the shared physical limit.
        if source != target:
            if not self.adjust(source, -int(bytes_)):
                return False
            if not self.adjust(target, int(bytes_)):
                # No transactional callback exists.  Keep this fail-closed:
                # callers must provide can/adjust that are atomic in practice.
                raise RuntimeError("KV pool ledger transfer lost atomicity")
        return True


class DynamicKVPool:
    """A bounded, topology-aware page allocator for the ``paged_pool`` mode."""

    def __init__(
        self,
        components: Iterable[Any],
        *,
        tokens_per_page: int = 16,
        page_bytes: int = 1,
        page_bytes_by_layer: Optional[Mapping[str, int]] = None,
        ledger: Optional[Any] = None,
        topology: Optional[Any] = None,
        offload_components: Optional[Sequence[str]] = None,
        allow_cross_machine: bool = False,
        transfer_executor: Optional[
            Callable[[KvPoolTransferRequest], KvPoolTransferPlan]
        ] = None,
    ) -> None:
        if isinstance(tokens_per_page, bool) or int(tokens_per_page) <= 0:
            raise KvPoolError("tokens_per_page must be positive")
        if isinstance(page_bytes, bool) or int(page_bytes) <= 0:
            raise KvPoolError("page_bytes must be positive")
        if isinstance(components, Mapping):
            raise KvPoolError("components must be an iterable of ComponentSpec or KvPoolComponent values")
        from .ir import ComponentSpec
        from .physical_contract import (
            parse_execution_physical_config, require_physical_memory_config,
        )

        normalized = []
        physical_configs = {}
        for component in components:
            item = KvPoolComponent.from_component(component)
            raw = item.metadata.get("physical_memory_config")
            if raw is not None:
                # An IR component carries the physical capacity contract. A
                # pool descriptor may instead carry a smaller logical budget;
                # require complete core geometry without equating the two.
                physical_configs[item.component_id] = (
                    require_physical_memory_config(component)
                    if isinstance(component, ComponentSpec)
                    else parse_execution_physical_config(raw)
                )
            normalized.append(item)
        if not normalized:
            raise KvPoolError("paged KV pool needs at least one component")
        if len({item.component_id for item in normalized}) != len(normalized):
            raise KvPoolError("paged KV pool component ids must be unique")
        machine_ids = {item.machine_id for item in normalized}
        if len(machine_ids) > 1 and not allow_cross_machine:
            raise KvPoolUnsupported("cross-machine KV pools require the future external KV store")
        for layer, value in (page_bytes_by_layer or {}).items():
            if isinstance(value, bool) or int(value) <= 0:
                raise KvPoolError("page_bytes_by_layer values must be positive")
        self.components: Dict[str, KvPoolComponent] = {item.component_id: item for item in normalized}
        self._allocation_granularity = {}
        for item in normalized:
            config = physical_configs.get(item.component_id)
            alignment = 1
            if config is not None:
                from .memory_types import DramConfig, SramConfig
                alignment = config.burst_bytes if isinstance(config, (DramConfig, SramConfig)) else config.page_bytes
            self._allocation_granularity[item.component_id] = int(alignment)
        self.tokens_per_page = int(tokens_per_page)
        self.default_page_bytes = int(page_bytes)
        self.page_bytes_by_layer = {str(key): int(value) for key, value in (page_bytes_by_layer or {}).items()}
        limits = {item.component_id: item.capacity_bytes for item in normalized}
        self._ledger = _LedgerAdapter(limits, ledger)
        if transfer_executor is not None and not callable(transfer_executor):
            raise KvPoolError("transfer_executor must be callable")
        # The pool owns page state and capacity accounting.  A caller that has
        # the physical address map and shared event queue can inject the
        # actual source-read/link/target-write lowering here.  The callback
        # receives an immutable page description so it can resolve a real
        # physical address and return a commit/rollback transfer plan.
        self._transfer_executor = None
        self.topology = topology
        explicit_offload = {str(item) for item in (offload_components or ())}
        unknown = explicit_offload - set(self.components)
        if unknown:
            raise KvPoolError("unknown offload component(s): " + ", ".join(sorted(unknown)))
        self.offload_components: Set[str] = explicit_offload or {
            item.component_id for item in normalized if not item.active and item.writable
        }
        self._pages: Dict[int, KvPage] = {}
        self._requests: Dict[str, List[int]] = {}
        self._request_components: Dict[str, Optional[str]] = {}
        self._prefixes: Dict[str, List[int]] = {}
        # Destination capacity is reserved while the authoritative source page
        # remains live.  Plans are owned by the caller's completion queue; this
        # plain metadata can be copied by admission rollback/probe code.
        self._pending_moves: Dict[int, Tuple[str, str, int, str]] = {}
        self._bindings: Dict[int, Set[str]] = defaultdict(set)
        self._owned_bytes: Dict[str, int] = defaultdict(int)
        self._cache_namespace = "kv_pool:" + uuid4().hex
        self._page_counter = 0
        self._clock = 0
        self._events: List[KvPoolTransfer] = []
        self._last_error: Optional[str] = None
        self.set_transfer_executor(transfer_executor)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def ledger(self) -> Any:
        """Return the shared physical ledger object used for accounting."""

        return self._ledger.ledger

    @property
    def transfer_executor(self) -> Any:
        """Return the optional physical migration lowerer attached to the pool."""

        return self._transfer_executor

    def set_transfer_executor(
        self,
        transfer_executor: Optional[
            Callable[[KvPoolTransferRequest], KvPoolTransferPlan]
        ],
    ) -> None:
        """Attach the shared physical transfer executor after pool creation."""

        if transfer_executor is not None and not callable(transfer_executor):
            raise KvPoolError("transfer_executor must be callable")
        if self._pending_moves:
            raise KvPoolError("cannot replace transfer executor with in-flight pages")
        footprint = getattr(transfer_executor, "allocation_bytes", None)
        granularity = dict(self._allocation_granularity)
        if callable(footprint):
            for component in self.components:
                value = footprint(component, 1)
                if type(value) is not int or value <= 0:
                    raise KvPoolError("physical allocation granularity must be a positive integer")
                granularity[component] = value
        totals = dict.fromkeys(self.components, 0)
        for page in self._pages.values():
            unit = granularity[page.owner_component]
            totals[page.owner_component] += ((page.bytes + unit - 1) // unit) * unit
        deltas = {component: total - self._owned_bytes.get(component, 0)
                  for component, total in totals.items() if total != self._owned_bytes.get(component, 0)}
        if deltas and self._transfer_executor is not None and self._pages:
            raise KvPoolError("cannot change allocation granularity of already bound pages")
        if deltas and not self._ledger.adjust_many(deltas):
            raise KvPoolError("existing pages exceed capacity at physical allocation granularity")
        self._allocation_granularity = granularity
        self._owned_bytes = defaultdict(int, {key: value for key, value in totals.items() if value})
        self._transfer_executor = transfer_executor

    def allocation_bytes(self, component_id: str, payload_bytes: int) -> int:
        """Reserved capacity for one allocation; transfer payload is unchanged."""
        unit = self._allocation_granularity[component_id]
        return ((payload_bytes + unit - 1) // unit) * unit

    @property
    def capacity_bytes_by_component(self) -> Mapping[str, int]:
        return {component_id: int(component.capacity_bytes) for component_id, component in self.components.items()}

    @property
    def used_bytes_by_component(self) -> Mapping[str, int]:
        return {component_id: int(self._component_used(component_id)) for component_id in self.components}

    @property
    def page_owners(self) -> Mapping[int, str]:
        return {page_id: page.owner_component for page_id, page in self._pages.items()}

    @property
    def events(self) -> Tuple[KvPoolTransfer, ...]:
        return tuple(self._events)

    @property
    def migration_time_ns(self) -> float:
        return sum(event.duration_ns for event in self._events)

    def clear_events(self) -> None:
        self._events.clear()

    def snapshot_transfer_executor(self) -> Any:
        """Capture optional external queue state for serving rollback.

        A physical executor that owns mutable queue state can expose matching
        ``snapshot()`` and ``restore(state)`` methods. Stateless plan producers
        have no queue state to capture and return ``None`` here.
        """

        executor = self._transfer_executor
        snapshot = getattr(executor, "snapshot", None)
        restore = getattr(executor, "restore", None)
        if snapshot is None and restore is None:
            return None
        if not callable(snapshot) or not callable(restore):
            raise KvPoolUnsupported(
                "transfer_executor must expose both snapshot() and restore(state)"
            )
        # The executor owns its snapshot representation. Physical snapshots
        # deliberately retain live Core handles alongside copied mutable
        # state; deepcopy would clone those handles and detach observers and
        # other owners of the shared runtime on restore.
        return snapshot()

    def restore_transfer_executor(self, state: Any) -> None:
        """Restore optional external queue state captured above."""

        if state is None:
            return
        executor = self._transfer_executor
        restore = getattr(executor, "restore", None)
        if not callable(restore):
            raise KvPoolUnsupported(
                "transfer_executor does not support restoring queue state"
            )
        restore(state)

    def _bytes_for_layer(self, layer_group: str) -> int:
        value = self.page_bytes_by_layer.get(str(layer_group), self.default_page_bytes)
        if value <= 0:
            raise KvPoolError("page bytes must be positive for layer " + str(layer_group))
        return value

    def _component_used(self, component_id: str) -> int:
        used = self._ledger.used_bytes
        if component_id in used:
            # Existing serving allocations are part of the shared physical
            # ledger.  This is the value used for capacity/ranking decisions.
            return max(0, int(used[component_id]))
        return max(0, int(self._owned_bytes.get(component_id, 0)))

    def _component_free(self, component_id: str) -> int:
        component = self.components[component_id]
        return max(0, int(component.capacity_bytes) - self._component_used(component_id))

    def _route(self, source: str, target: str, bytes_: int) -> Tuple[Any, ...]:
        if source == target or self.topology is None:
            return ()
        route = getattr(self.topology, "route", None)
        if route is None:
            raise KvPoolUnsupported("topology must expose route(source, target, bytes)")
        return tuple(route(source, target, int(bytes_)))

    def _reachable(self, source: Optional[str], target: str, bytes_: int) -> bool:
        if not source or source == target or self.topology is None:
            return True
        try:
            self._route(source, target, bytes_)
            return True
        except (ValueError, KeyError, KvPoolUnsupported):
            return False

    def _transfer_plan(
        self, page: KvPage, source: str, target: str, bytes_: int, *, staged: bool = False
    ) -> KvPoolTransferPlan:
        """Prepare a migration through an injected shared physical executor.

        The pool knows ownership and capacity, but it does not know physical
        addresses or queue ownership.  Without an executor the operation stays
        fail-closed rather than silently pricing a second, aggregate transfer.
        """
        if source == target or bytes_ == 0:
            return KvPoolTransferPlan(0.0)
        executor = self._transfer_executor
        if executor is None:
            raise KvPoolUnsupported(
                "paged_pool migration requires physical address and shared memory-event "
                "lowering; aggregate transfer costs are disabled. Provide transfer_executor "
                "or use fixed/manual KV placement."
            )
        request = KvPoolTransferRequest(
            logical_page_id=int(page.logical_page_id),
            request_id=str(page.request_id),
            layer_group=str(page.layer_group),
            source_component=str(source),
            target_component=str(target),
            bytes=int(bytes_),
            allocation_generation=int(page.allocation_generation),
            token_start=int(page.token_start),
            token_end=int(page.token_end),
        )
        try:
            prepare = getattr(executor, "prepare_async", None) if staged else None
            raw_plan = prepare(request) if callable(prepare) else executor(request)
        except KvPoolError:
            raise
        except Exception as error:
            raise KvPoolUnsupported(
                "paged_pool physical transfer executor failed: {}".format(error)
            ) from error
        if isinstance(raw_plan, KvPoolTransferPlan):
            return raw_plan
        raise KvPoolUnsupported("paged_pool physical transfer executor must return KvPoolTransferPlan")

    @staticmethod
    def _run_transfer_hook(plan: KvPoolTransferPlan, name: str) -> None:
        hook = getattr(plan, name)
        if hook is None:
            return
        try:
            hook()
        except Exception as error:
            raise KvPoolUnsupported(
                "paged_pool transfer {} hook failed: {}".format(name, error)
            ) from error

    @staticmethod
    def _require_storage_bandwidth(
        component: KvPoolComponent, *, read: bool
    ) -> None:
        """Reject an unknown service direction for offload media.

        A topology link cannot stand in for the medium's own service time.
        Active HBM/DRAM service rates come from the physical executor's config;
        offload media must also declare the direction the operation uses.
        """

        normalized_kind = (
            str(component.kind)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if normalized_kind not in _OFFLOAD_STORAGE_KINDS:
            return
        bandwidth = (
            component.read_bandwidth_gbps
            if read
            else component.write_bandwidth_gbps
        )
        if float(bandwidth) <= 0.0:
            direction = "read" if read else "write"
            raise KvPoolUnsupported(
                "storage component {} requires a positive {} bandwidth".format(
                    component.component_id, direction
                )
            )

    def _candidate_ids(
        self,
        layer_group: str,
        bytes_: int,
        *,
        request_component: Optional[str],
        compatible_components: Optional[Iterable[str] | Mapping[str, Iterable[str]]],
        active: bool,
    ) -> List[str]:
        allowed: Optional[Set[str]] = None
        if compatible_components is not None:
            if isinstance(compatible_components, Mapping):
                values = compatible_components.get(layer_group, compatible_components.get("*", ()))
            else:
                values = compatible_components
            allowed = {str(value) for value in values}
        candidates: List[Tuple[Tuple[float, float, float, str], str]] = []
        for component_id, component in self.components.items():
            if active != component.active or not component.writable:
                continue
            if allowed is not None and component_id not in allowed:
                continue
            if not self._reachable(request_component, component_id, bytes_):
                continue
            reserved = self.allocation_bytes(component_id, bytes_)
            if self._component_free(component_id) < reserved or not self._ledger.can_adjust(component_id, reserved):
                continue
            bandwidth = max(float(component.write_bandwidth_gbps), float(component.read_bandwidth_gbps), 0.0)
            # More free bytes first; then faster links/memory and lower local latency.
            score = (-float(self._component_free(component_id)), -bandwidth, float(component.latency_ns), component_id)
            candidates.append((score, component_id))
        candidates.sort(key=lambda item: item[0])
        return [item[1] for item in candidates]

    def _touch(self, page: KvPage) -> None:
        self._clock += 1
        page.last_access = self._clock

    def _register_page(self, page: KvPage) -> None:
        self._pages[page.logical_page_id] = page
        self._owned_bytes[page.owner_component] += self.allocation_bytes(page.owner_component, page.bytes)
        self._bindings[page.logical_page_id].add(page.request_id)

    def _free_page(self, page_id: int) -> None:
        # Commit the shared ledger release before mutating pool indexes.  A
        # rejecting external ledger must leave the page and its ownership
        # metadata available for a retry.
        page = self._pages.get(page_id)
        if page is None:
            return
        reserved = self.allocation_bytes(page.owner_component, page.bytes)
        if not self._ledger.adjust(page.owner_component, -reserved):
            raise RuntimeError("KV pool physical ledger rejected page release")
        self._pages.pop(page_id, None)
        self._owned_bytes[page.owner_component] -= reserved
        if self._owned_bytes[page.owner_component] <= 0:
            self._owned_bytes.pop(page.owner_component, None)
        self._bindings.pop(page_id, None)
        for key, ids in list(self._prefixes.items()):
            if page_id in ids:
                ids[:] = [value for value in ids if value != page_id]
            if not ids:
                self._prefixes.pop(key, None)

    def _drop_ref(self, page_id: int, binding: Optional[str] = None) -> None:
        page = self._pages.get(page_id)
        if page is None:
            return
        if page.ref_count <= 0:
            raise RuntimeError("KV pool page refcount underflow")
        # The final unpinned reference frees the page.  Do this before
        # removing the binding/decrementing ref_count so a rejected ledger
        # release leaves the whole page state unchanged.
        if page.ref_count == 1 and not page.pinned:
            self._free_page(page_id)
            return
        if binding is not None:
            self._bindings.get(page_id, set()).discard(binding)
        page.ref_count -= 1

    def _attach_ref(self, page_id: int, request_id: str) -> None:
        page = self._pages[page_id]
        page.ref_count += 1
        self._bindings[page_id].add(request_id)
        self._touch(page)

    def _allocate_one(
        self,
        request_id: str,
        logical_index: int,
        token_count: Optional[int],
        layer_group: str,
        *,
        request_component: Optional[str],
        compatible_components: Optional[Iterable[str] | Mapping[str, Iterable[str]]],
    ) -> Optional[KvPage]:
        bytes_ = self._bytes_for_layer(layer_group)
        candidates = self._candidate_ids(
            layer_group,
            bytes_,
            request_component=request_component,
            compatible_components=compatible_components,
            active=True,
        )
        if not candidates:
            return None
        owner = candidates[0]
        if not self._ledger.adjust(owner, self.allocation_bytes(owner, bytes_)):
            return None
        token_start = logical_index * self.tokens_per_page
        token_end = token_start + self.tokens_per_page
        if token_count is not None:
            token_end = min(token_end, max(token_start, int(token_count)))
        page = KvPage(
            request_id=request_id,
            logical_page_id=self._page_counter,
            token_start=token_start,
            token_end=token_end,
            layer_group=str(layer_group),
            owner_component=owner,
            resident_tier=self.components[owner].tier,
            bytes=bytes_,
            last_access=self._clock,
        )
        self._page_counter += 1
        self._register_page(page)
        self._touch(page)
        return page

    def resize_detailed(
        self,
        request_id: str,
        target_pages: Optional[int] = None,
        *,
        target_tokens: Optional[int] = None,
        layer_group: str = "default",
        request_component: Optional[str] = None,
        compatible_components: Optional[Iterable[str] | Mapping[str, Iterable[str]]] = None,
        prefix_key: Optional[str] = None,
    ) -> KvPoolResizeResult:
        """Resize one request atomically; return a machine-readable result."""

        self._last_error = None
        if not request_id:
            raise KvPoolError("request_id must be non-empty")
        if target_pages is None:
            if target_tokens is None:
                raise KvPoolError("target_pages or target_tokens is required")
            if isinstance(target_tokens, bool) or int(target_tokens) < 0:
                raise KvPoolError("target_tokens must be non-negative")
            target_pages = (int(target_tokens) + self.tokens_per_page - 1) // self.tokens_per_page
        if isinstance(target_pages, bool) or int(target_pages) < 0:
            raise KvPoolError("target_pages must be non-negative")
        target_pages = int(target_pages)
        current_ids = list(self._requests.get(request_id, ()))
        current_count = len(current_ids)
        if any(page_id in self._pending_moves for page_id in current_ids):
            self._last_error = "cannot resize KV pages with an in-flight transfer"
            return KvPoolResizeResult(False, request_id, current_count, error=self._last_error)
        if target_pages == current_count:
            for page_id in current_ids:
                self._touch(self._pages[page_id])
            self._request_components.setdefault(request_id, request_component)
            return KvPoolResizeResult(True, request_id, current_count)

        if target_pages < current_count:
            removed = current_ids[target_pages:]
            kept = current_ids[:target_pages]
            pending = list(removed)
            released = 0
            while pending:
                page_id = pending[0]
                try:
                    self._drop_ref(page_id, request_id)
                except RuntimeError:
                    # Preserve only the refs that still belong to this
                    # request.  A retry must not drop a shared page twice.
                    self._requests[request_id] = kept + pending
                    raise
                pending.pop(0)
                released += 1
                self._requests[request_id] = kept + pending
            self._requests[request_id] = kept
            if not self._requests[request_id]:
                self._requests.pop(request_id, None)
                self._request_components.pop(request_id, None)
            return KvPoolResizeResult(True, request_id, target_pages, released_pages=released)

        # All newly attached/allocated pages are rolled back if one page cannot
        # be placed.  This is the central atomicity guarantee of the pool.
        working = current_ids[:]
        attached: List[int] = []
        allocated: List[int] = []
        try:
            if prefix_key and not current_ids:
                for page_id in self._prefixes.get(str(prefix_key), ())[:target_pages]:
                    self._attach_ref(page_id, request_id)
                    working.append(page_id)
                    attached.append(page_id)
            while len(working) < target_pages:
                page = self._allocate_one(
                    request_id,
                    len(working),
                    target_tokens,
                    layer_group,
                    request_component=request_component,
                    compatible_components=compatible_components,
                )
                if page is None:
                    raise KvPoolError(
                        "no compatible writable active component has capacity for page {}".format(len(working))
                    )
                working.append(page.logical_page_id)
                allocated.append(page.logical_page_id)
            self._requests[request_id] = working
            self._request_components[request_id] = request_component
            return KvPoolResizeResult(True, request_id, target_pages, len(allocated), 0)
        except (KvPoolError, RuntimeError) as exc:
            for page_id in reversed(allocated):
                self._drop_ref(page_id, request_id)
            for page_id in reversed(attached):
                self._drop_ref(page_id, request_id)
            self._last_error = str(exc)
            return KvPoolResizeResult(False, request_id, current_count, error=self._last_error)

    def can_resize(self, request_id: str, target_pages: int) -> bool:
        """Probe the same placement algorithm without changing live identities.

        A shadow ledger accumulates reservations so external capacity checks see
        total growth, not just one page. No external adjust callback is invoked.
        """
        shadow = copy.copy(self)
        for name in ("_pages", "_requests", "_request_components", "_prefixes",
                     "_bindings", "_owned_bytes", "_events", "_pending_moves"):
            setattr(shadow, name, copy.deepcopy(getattr(self, name)))
        initial = dict(self._ledger.used_bytes)
        ledger = _LocalLedger(self._ledger.limits)
        ledger.used_bytes.update(initial)
        local_can_adjust = ledger.can_adjust
        ledger.can_adjust = lambda component, delta: (
            local_can_adjust(component, delta)
            and self._ledger.can_adjust(
                component, ledger.used_bytes.get(component, 0)
                - initial.get(component, 0) + delta)
        )
        shadow._ledger = _LedgerAdapter(self._ledger.limits, ledger)
        return shadow.resize_detailed(request_id, target_pages).success

    def handoff_request(self, source_request: str, target_request: str) -> Tuple[int, ...]:
        """Move exclusive page references between turns without copying KV."""
        if source_request == target_request or self._requests.get(target_request):
            raise KvPoolError("KV handoff requires a distinct empty destination request")
        pages = self.request_pages(source_request)
        if any(p.ref_count != 1 or p.pinned or p.prefix_key is not None
               or p.logical_page_id in self._pending_moves for p in pages):
            raise KvPoolError("KV handoff supports exclusive pages without in-flight copies only")
        ids = self._requests.pop(source_request, [])
        self._requests[target_request] = ids
        component = self._request_components.pop(source_request, None)
        if component is not None:
            self._request_components[target_request] = component
        for page in pages:
            self._bindings[page.logical_page_id].remove(source_request)
            self._bindings[page.logical_page_id].add(target_request)
            page.request_id = target_request
        return tuple(ids)

    def release(self, request_id: str) -> None:
        if any(page in self._pending_moves for page in self._requests.get(request_id, ())):
            raise KvPoolError("cannot release KV pages with an in-flight transfer")
        pending = list(self._requests.get(request_id, ()))
        while pending:
            page_id = pending[0]
            try:
                self._drop_ref(page_id, request_id)
            except RuntimeError:
                self._requests[request_id] = pending
                raise
            pending.pop(0)
            self._requests[request_id] = pending
        self._requests.pop(request_id, None)
        self._request_components.pop(request_id, None)

    def request_pages(self, request_id: str) -> Tuple[KvPage, ...]:
        return tuple(self._pages[page_id] for page_id in self._requests.get(request_id, ()) if page_id in self._pages)

    def cache_accesses(self, request_id: str, *, first_token: int, token_count: int,
                       operation: str = "read"):
        """Export byte ranges from actual live page owners, including prefixes.

        IDs are pool-monotonic, so release/reallocate does not resurrect cache
        lines. Every migration advances a generation, including a return to
        the original component. Independent pools never alias one another.
        """
        from .kernel_memory import paged_buffer_accesses
        from dataclasses import replace
        pages = self.request_pages(request_id)
        if any(page.logical_page_id in self._pending_moves for page in pages):
            raise KvPoolError("KV consumer must wait for the in-flight transfer completion")
        sizes = {p.bytes for p in pages}
        if not pages:
            # Still validate range and operation for empty requests.
            return paged_buffer_accesses(
                buffer_id=self._cache_namespace, page_ids=(),
                tokens_per_page=self.tokens_per_page, bytes_per_token=1,
                first_token=first_token, token_count=token_count, operation=operation)
        if len(sizes) != 1 or next(iter(sizes)) % self.tokens_per_page:
            raise KvPoolUnsupported("KV cache range requires uniform complete token rows")
        page_bytes = next(iter(sizes))
        accesses = paged_buffer_accesses(
            buffer_id=self._cache_namespace, page_ids=tuple(p.logical_page_id for p in pages),
            tokens_per_page=self.tokens_per_page,
            bytes_per_token=page_bytes // self.tokens_per_page,
            first_token=first_token, token_count=token_count, operation=operation)
        by_id = {p.logical_page_id: p for p in pages}
        result = []
        for access in accesses:
            page = by_id[int(access.buffer_id.rsplit(":", 1)[1])]
            if access.operation == "write" and page.ref_count > 1:
                raise KvPoolUnsupported(
                    "shared KV page writes require explicit copy-on-write; the pool only reuses read-only prefixes")
            export = getattr(self._transfer_executor, "cache_access", None)
            if callable(export):
                result.append((page.owner_component, export(page,
                    offset_bytes=access.offset_bytes, size_bytes=access.size_bytes,
                    operation=access.operation)))
                continue
            identity = (f"{page.owner_component}:{self._cache_namespace}:"
                        f"generation:{page.allocation_generation}:page:{page.logical_page_id}")
            result.append((page.owner_component, replace(access, buffer_id=identity)))
        return tuple(result)

    def page(self, logical_page_id: int) -> KvPage:
        try:
            return self._pages[int(logical_page_id)]
        except KeyError as exc:
            raise KvPoolError("unknown KV page {}".format(logical_page_id)) from exc

    def pages(self) -> Tuple[KvPage, ...]:
        return tuple(self._pages.values())

    def register_prefix(self, prefix_key: str, pages: Sequence[KvPage | int]) -> None:
        if not prefix_key:
            raise KvPoolError("prefix_key must be non-empty")
        if prefix_key in self._prefixes:
            raise KvPoolError("prefix_key already registered: " + prefix_key)
        ids: List[int] = []
        for value in pages:
            page_id = value.logical_page_id if isinstance(value, KvPage) else int(value)
            if page_id not in self._pages:
                raise KvPoolError("cannot register unknown page {}".format(page_id))
            if page_id in ids:
                raise KvPoolError("a prefix cannot contain the same physical page twice")
            ids.append(page_id)
        for page_id in ids:
            page = self._pages[page_id]
            page.ref_count += 1
            page.prefix_key = str(prefix_key)
            self._touch(page)
        self._prefixes[str(prefix_key)] = ids

    def release_prefix(self, prefix_key: str) -> None:
        key = str(prefix_key)
        pending = list(self._prefixes.get(key, ()))
        while pending:
            page_id = pending[0]
            try:
                self._drop_ref(page_id)
            except RuntimeError:
                self._prefixes[key] = pending
                raise
            pending.pop(0)
            if key in self._prefixes:
                # _free_page edits this index in place. Keep the work list
                # separate or freeing the next page also consumes pending,
                # skipping a reference (or popping an empty list).
                self._prefixes[key] = list(pending)
            if page_id in self._pages:
                # A page may belong to several registered prefixes. Keep a
                # representative live key, and clear it after its last prefix
                # reference goes away so exclusive handoff becomes legal.
                self._pages[page_id].prefix_key = next(
                    (other for other, ids in self._prefixes.items() if page_id in ids), None)
        self._prefixes.pop(key, None)

    def prefix_pages(self, prefix_key: str) -> Tuple[KvPage, ...]:
        return tuple(self._pages[page_id] for page_id in self._prefixes.get(str(prefix_key), ()) if page_id in self._pages)

    def pin(self, request_id: str, logical_page_id: Optional[int] = None) -> int:
        pages = self.request_pages(request_id)
        if logical_page_id is not None:
            pages = tuple(page for page in pages if page.logical_page_id == int(logical_page_id))
        for page in pages:
            page.pinned = True
            self._touch(page)
        return len(pages)

    def unpin(self, request_id: str, logical_page_id: Optional[int] = None) -> int:
        pages = self.request_pages(request_id)
        if logical_page_id is not None:
            pages = tuple(page for page in pages if page.logical_page_id == int(logical_page_id))
        count = 0
        for page in pages:
            if page.ref_count == 0:
                self._free_page(page.logical_page_id)
            else:
                page.pinned = False
            count += 1
        return count

    def unpin_page(self, logical_page_id: int) -> bool:
        """Unpin a page after its request binding has already been released."""

        page = self.page(logical_page_id)
        if not page.pinned:
            return False
        if page.ref_count == 0:
            self._free_page(page.logical_page_id)
        else:
            page.pinned = False
        return True

    def touch(self, request_id: str) -> None:
        for page in self.request_pages(request_id):
            self._touch(page)

    def evict(self, bytes_needed: int, *, component_ids: Optional[Iterable[str]] = None) -> int:
        """Evict unbound, unpinned LRU pages and return freed physical bytes."""

        if isinstance(bytes_needed, bool) or int(bytes_needed) < 0:
            raise KvPoolError("bytes_needed must be non-negative")
        allowed = {str(item) for item in component_ids} if component_ids is not None else None
        candidates = [
            page for page in self._pages.values()
            if page.ref_count == 1 and not page.pinned and not self._bindings.get(page.logical_page_id)
            and (allowed is None or page.owner_component in allowed)
        ]
        candidates.sort(key=lambda page: (page.last_access, page.logical_page_id))
        freed = 0
        for page in candidates:
            if freed >= int(bytes_needed):
                break
            # Drop the prefix reference (the only legal unbound reference).
            self._free_page(page.logical_page_id)
            freed += self.allocation_bytes(page.owner_component, page.bytes)
        return freed

    def _move(self, page: KvPage, target_component: str, *, kind: str) -> bool:
        if page.logical_page_id in self._pending_moves:
            self._last_error = "KV page already has an in-flight transfer"
            return False
        if target_component not in self.components:
            self._last_error = "unknown target component " + target_component
            return False
        if page.owner_component == target_component:
            self._last_error = None
            return True
        target = self.components[target_component]
        if not target.writable:
            self._last_error = "target component is read-only: " + target_component
            return False
        self._require_storage_bandwidth(
            self.components[page.owner_component], read=True
        )
        self._require_storage_bandwidth(target, read=False)
        if not self._reachable(page.owner_component, target_component, page.bytes):
            self._last_error = "no topology route from {} to {}".format(page.owner_component, target_component)
            return False
        source_bytes = self.allocation_bytes(page.owner_component, page.bytes)
        target_bytes = self.allocation_bytes(target_component, page.bytes)
        if not self._ledger.can_transfer_sized(page.owner_component, target_component, source_bytes, target_bytes):
            self._last_error = "target component has insufficient physical capacity: " + target_component
            return False
        source = page.owner_component
        # Prepare the physical event before changing ownership.  A two-phase
        # plan can cancel a queue reservation if the capacity ledger rejects
        # the transfer; a numeric result is a side-effect-free timing estimate.
        try:
            plan = self._transfer_plan(page, source, target_component, page.bytes)
        except KvPoolError as error:
            self._last_error = str(error)
            raise
        if not self._ledger.transfer_sized(source, target_component, source_bytes, target_bytes):
            try:
                self._run_transfer_hook(plan, "rollback")
            except KvPoolError:
                self._last_error = "physical transfer reservation could not be rolled back"
                raise
            self._last_error = "physical ledger rejected transfer {} -> {}".format(source, target_component)
            return False
        try:
            self._run_transfer_hook(plan, "commit")
        except KvPoolError as error:
            rollback_error: Optional[BaseException] = None
            try:
                self._run_transfer_hook(plan, "rollback")
            except BaseException as hook_error:
                rollback_error = hook_error
            reverted = False
            try:
                reverted = self._ledger.transfer_sized(target_component, source, target_bytes, source_bytes)
            except BaseException as ledger_error:
                rollback_error = rollback_error or ledger_error
            if not reverted:
                rollback_error = rollback_error or RuntimeError(
                    "physical ledger could not reverse failed KV transfer"
                )
            self._last_error = str(error)
            if rollback_error is not None:
                raise KvPoolUnsupported(
                    "paged_pool transfer commit failed and rollback was incomplete"
                ) from rollback_error
            raise
        self._owned_bytes[source] -= source_bytes
        self._owned_bytes[target_component] += target_bytes
        page.allocation_generation += 1
        page.owner_component = target_component
        page.resident_tier = target.tier
        self._touch(page)
        self._events.append(KvPoolTransfer(
            kind,
            page.logical_page_id,
            source,
            target_component,
            page.bytes,
            plan.duration_ns,
            plan.start_ns,
            plan.completion_ns,
        ))
        self._last_error = None
        return True

    def prepare_restore_page(
        self, logical_page_id: int, target_component: str, *, kind: str = "prefetch",
    ) -> Optional[KvPoolTransferPlan]:
        """Reserve a copy without publishing its destination to consumers.

        The caller must complete the returned plan at/after completion_ns.
        Abort is only legal immediately, before unrelated work is scheduled:
        executor rollback restores the reservation calendar, not just one I/O.
        """
        page = self.page(logical_page_id)
        target = self.components.get(target_component)
        if (page.logical_page_id in self._pending_moves or target is None
                or not target.active or not target.writable
                or page.owner_component == target_component
                or page.ref_count != 1 or page.pinned):
            self._last_error = "invalid or already pending asynchronous restore"
            return None
        source = page.owner_component
        target_bytes = self.allocation_bytes(target_component, page.bytes)
        self._require_storage_bandwidth(self.components[source], read=True)
        self._require_storage_bandwidth(target, read=False)
        if (not self._reachable(source, target_component, page.bytes)
                or not self._ledger.can_adjust(target_component, target_bytes)):
            self._last_error = "asynchronous restore has no route or destination capacity"
            return None
        plan = self._transfer_plan(page, source, target_component, page.bytes, staged=True)
        if plan.start_ns is None or plan.completion_ns is None:
            self._run_transfer_hook(plan, "rollback")
            raise KvPoolUnsupported("asynchronous restore requires absolute physical completion timestamps")
        if not self._ledger.adjust(target_component, target_bytes):
            self._run_transfer_hook(plan, "rollback")
            self._last_error = "physical ledger rejected asynchronous restore reservation"
            return None
        self._pending_moves[page.logical_page_id] = (
            source, target_component, page.allocation_generation, str(kind))
        self._last_error = None
        return plan

    def abort_prepared_restore(self, logical_page_id: int, plan: KvPoolTransferPlan) -> None:
        """Undo a just-prepared restore; callers abort a batch in reverse order."""
        page = self.page(logical_page_id)
        _source, target, _generation, _kind = self._pending_moves[logical_page_id]
        target_bytes = self.allocation_bytes(target, page.bytes)
        if not self._ledger.adjust(target, -target_bytes):
            raise KvPoolError("could not release asynchronous restore reservation")
        try:
            self._run_transfer_hook(plan, "rollback")
        except BaseException:
            if not self._ledger.adjust(target, target_bytes):
                raise KvPoolUnsupported("could not restore failed KV abort reservation")
            raise
        del self._pending_moves[logical_page_id]

    def complete_prepared_restore(
        self, logical_page_id: int, plan: KvPoolTransferPlan, *, now_ns: float,
    ) -> None:
        """Publish a completed copy and release only its original source."""
        page = self.page(logical_page_id)
        source, target, generation, kind = self._pending_moves[logical_page_id]
        if (not isfinite(now_ns) or now_ns < 0 or plan.advance is not None
                or plan.completion_ns is None or now_ns < plan.completion_ns):
            raise KvPoolError("cannot consume a KV transfer before its completion")
        if page.owner_component != source or page.allocation_generation != generation:
            raise KvPoolError("in-flight KV source identity changed")
        source_bytes = self.allocation_bytes(source, page.bytes)
        target_bytes = self.allocation_bytes(target, page.bytes)
        if not self._ledger.adjust(source, -source_bytes):
            raise KvPoolError("could not release completed KV transfer source")
        try:
            self._run_transfer_hook(plan, "commit")
        except BaseException:
            if not self._ledger.adjust(source, source_bytes):
                raise KvPoolUnsupported("could not restore failed KV transfer source reservation")
            raise
        self._owned_bytes[source] -= source_bytes
        self._owned_bytes[target] += target_bytes
        page.owner_component = target
        page.resident_tier = self.components[target].tier
        page.allocation_generation += 1
        self._touch(page)
        del self._pending_moves[logical_page_id]
        self._events.append(KvPoolTransfer(
            kind, logical_page_id, source, target, page.bytes,
            plan.duration_ns, plan.start_ns, plan.completion_ns))

    def cancel_prepared_restore(self, logical_page_id: int, plan: KvPoolTransferPlan,
                                *, now_ns: float) -> Mapping[str, Any]:
        """Drain the submitted phase, retain source ownership, discard target."""
        page = self.page(logical_page_id)
        source, target, generation, _kind = self._pending_moves[logical_page_id]
        if (not isfinite(now_ns) or now_ns < 0
                or plan.completion_ns is None or now_ns < plan.completion_ns):
            raise KvPoolError("cancellation must drain the currently submitted physical phase")
        if page.owner_component != source or page.allocation_generation != generation:
            raise KvPoolError("in-flight KV source identity changed")
        if plan.cancel is None:
            raise KvPoolUnsupported("executor does not support cost-preserving cancellation")
        target_bytes = self.allocation_bytes(target, page.bytes)
        if not self._ledger.adjust(target, -target_bytes):
            raise KvPoolError("could not release cancelled restore reservation")
        try:
            billed = plan.cancel()
        except BaseException:
            if not self._ledger.adjust(target, target_bytes):
                raise KvPoolUnsupported("could not restore failed KV cancellation reservation")
            raise
        del self._pending_moves[logical_page_id]
        return billed

    def migrate_page(self, logical_page_id: int, target_component: str) -> bool:
        page = self.page(logical_page_id)
        if not self.components[target_component].active:
            raise KvPoolUnsupported("migrate_page target must be active memory; use offload_page for backing tiers")
        return self._move(page, str(target_component), kind="migration")

    def offload_page(self, logical_page_id: int, target_component: Optional[str] = None,
                     *, kind: str = "offload") -> bool:
        page = self.page(logical_page_id)
        targets = [str(target_component)] if target_component is not None else sorted(self.offload_components)
        for target in targets:
            if target not in self.components or target not in self.offload_components:
                continue
            if self._move(page, target, kind=str(kind)):
                return True
        self._last_error = self._last_error or "no writable offload component has capacity"
        return False

    def restore_page(
        self,
        logical_page_id: int,
        target_component: Optional[str] = None,
        *,
        kind: str = "restore",
    ) -> bool:
        page = self.page(logical_page_id)
        candidates = [str(target_component)] if target_component is not None else self._candidate_ids(
            page.layer_group,
            page.bytes,
            request_component=page.owner_component,
            compatible_components=None,
            active=True,
        )
        for target in candidates:
            if target in self.components and self.components[target].active and self._move(page, target, kind=str(kind)):
                return True
        self._last_error = self._last_error or "no active component has capacity for restore"
        return False

    def component_stats(self) -> Mapping[str, Mapping[str, int | str]]:
        page_counts: Dict[str, int] = defaultdict(int)
        for page in self._pages.values():
            page_counts[page.owner_component] += 1
        return {
            component_id: {
                "capacity_bytes": int(component.capacity_bytes),
                "used_bytes": int(self._component_used(component_id)),
                "pool_bytes": int(self._owned_bytes.get(component_id, 0)),
                "logical_pool_bytes": sum(page.bytes for page in self._pages.values()
                                          if page.owner_component == component_id),
                "allocation_granularity_bytes": self._allocation_granularity[component_id],
                "free_bytes": int(self._component_free(component_id)),
                "page_count": int(page_counts.get(component_id, 0)),
                "active": component.active,
                "tier": component.tier,
            }
            for component_id, component in self.components.items()
        }

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "layout_mode": "paged_pool",
            "tokens_per_page": self.tokens_per_page,
            "page_count": len(self._pages),
            "request_count": len(self._requests),
            "prefix_count": len(self._prefixes),
            "component_stats": self.component_stats(),
            "migration_time_ns": self.migration_time_ns,
            "transfer_count": len(self._events),
            "last_error": self._last_error,
        }


__all__ = [
    "DynamicKVPool",
    "KvPage",
    "KvPoolComponent",
    "KvPoolError",
    "KvPoolResizeResult",
    "KvPoolTransfer",
    "KvPoolTransferPlan",
    "KvPoolTransferRequest",
    "KvPoolUnsupported",
    "PhysicalLedger",
]
