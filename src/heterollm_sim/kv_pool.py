"""Experimental paged KV-cache pool.

This module deliberately lives beside the legacy serving ledger.  It provides
the page ownership and migration primitives needed by the experimental
``paged_pool`` layout without changing the llama.cpp-compatible static path.
The pool owns no scheduler state: callers resize a request atomically and can
use the returned component/page statistics when lowering serving events.

The implementation is intentionally single-machine.  A page can move only
between components that are writable active/offload memory and reachable in
the supplied topology.  An external physical ledger can be supplied through
the small ``can_adjust``/``adjust``/``transfer`` interface, so allocations
participate in the serving ledger instead of maintaining a disconnected
capacity counter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple


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

    @classmethod
    def from_component(cls, component: Any) -> "KvPoolComponent":
        """Build a pool descriptor from an IR ``ComponentSpec`` or mapping."""

        if isinstance(component, cls):
            return component
        if isinstance(component, Mapping):
            get = component.get
            component_id = get("component_id", get("id"))
            metadata = get("metadata", {}) or {}
            kind = str(get("kind", "memory"))
            active = get("active", get("is_active_memory", None))
            writable = get("writable", get("is_writable", None))
            if active is None:
                active = str(kind).lower() not in {"hbf", "ssd", "nvme", "flash", "storage"}
            if writable is None:
                writable = metadata.get("writable", metadata.get("read_only") is not True)
            return cls(
                component_id=str(component_id),
                capacity_bytes=int(get("capacity_bytes", 0)),
                kind=kind,
                tier=str(get("tier", metadata.get("memory_tier", "active" if active else "offload"))),
                active=bool(active),
                writable=bool(writable),
                machine_id=str(get("machine_id", metadata.get("machine_id", "local"))),
                read_bandwidth_gbps=float(get("read_bandwidth_gbps", metadata.get("read_bandwidth_gbps", 0.0))),
                write_bandwidth_gbps=float(get("write_bandwidth_gbps", metadata.get("write_bandwidth_gbps", 0.0))),
                latency_ns=float(get("latency_ns", metadata.get("latency_ns", 0.0))),
                metadata=dict(metadata),
            )
        metadata = dict(getattr(component, "metadata", {}) or {})
        component_id = getattr(component, "component_id", getattr(component, "id", None))
        if component_id is None:
            raise KvPoolError("pool component must expose component_id")
        kind = str(getattr(component, "kind", "memory"))
        active_value = getattr(component, "is_active_memory", None)
        if active_value is None:
            active_value = str(kind).lower() not in {"hbf", "ssd", "nvme", "flash", "storage"}
        writable_value = getattr(component, "is_writable", None)
        if writable_value is None:
            writable_value = metadata.get("writable", metadata.get("read_only") is not True)
        return cls(
            component_id=str(component_id),
            capacity_bytes=int(getattr(component, "capacity_bytes", 0)),
            kind=kind,
            tier=str(metadata.get("memory_tier", "active" if active_value else "offload")),
            active=bool(active_value),
            writable=bool(writable_value),
            machine_id=str(metadata.get("machine_id", "local")),
            read_bandwidth_gbps=float(getattr(component, "read_bandwidth_gbps", 0.0)),
            write_bandwidth_gbps=float(getattr(component, "write_bandwidth_gbps", 0.0)),
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
        *,
        can_adjust: Optional[Callable[[str, int], bool]] = None,
        adjust: Optional[Callable[[str, int], bool]] = None,
        can_transfer: Optional[Callable[[str, str, int], bool]] = None,
        transfer: Optional[Callable[[str, str, int], bool]] = None,
    ) -> None:
        self._local = _LocalLedger(limits)
        self.ledger = ledger or self._local
        self._can_adjust_callback = can_adjust
        self._adjust_callback = adjust
        self._can_transfer_callback = can_transfer
        self._transfer_callback = transfer
        self._callback_used: Dict[str, int] = {str(key): 0 for key in limits}

    @property
    def limits(self) -> Mapping[str, int]:
        return getattr(self.ledger, "limits", self._local.limits)

    @property
    def used_bytes(self) -> Mapping[str, int]:
        if self.ledger is self._local and (
            self._can_adjust_callback is not None
            or self._adjust_callback is not None
            or self._can_transfer_callback is not None
            or self._transfer_callback is not None
        ):
            return self._callback_used
        return getattr(self.ledger, "used_bytes", self._local.used_bytes)

    def can_adjust(self, component: str, delta: int) -> bool:
        if self._can_adjust_callback is not None:
            return bool(self._can_adjust_callback(component, int(delta)))
        method = getattr(self.ledger, "can_adjust", None)
        return bool(method(component, int(delta))) if method is not None else True

    def adjust(self, component: str, delta: int) -> bool:
        if self._adjust_callback is not None:
            result = bool(self._adjust_callback(component, int(delta)))
            if result:
                self._callback_used[component] = self._callback_used.get(component, 0) + int(delta)
            return result
        method = getattr(self.ledger, "adjust", None)
        result = bool(method(component, int(delta))) if method is not None else True
        if result and self.ledger is self._local and self._can_adjust_callback is not None:
            self._callback_used[component] = self._callback_used.get(component, 0) + int(delta)
        return result

    def can_transfer(self, source: str, target: str, bytes_: int) -> bool:
        if self._can_transfer_callback is not None:
            return bool(self._can_transfer_callback(source, target, int(bytes_)))
        method = getattr(self.ledger, "can_transfer", None)
        if method is not None:
            return bool(method(source, target, int(bytes_)))
        return self.can_adjust(source, -int(bytes_)) and self.can_adjust(target, int(bytes_))

    def transfer(self, source: str, target: str, bytes_: int) -> bool:
        if self._transfer_callback is not None:
            result = bool(self._transfer_callback(source, target, int(bytes_)))
            if result and source != target:
                self._callback_used[source] = self._callback_used.get(source, 0) - int(bytes_)
                self._callback_used[target] = self._callback_used.get(target, 0) + int(bytes_)
            return result
        method = getattr(self.ledger, "transfer", None)
        if method is not None:
            return bool(method(source, target, int(bytes_)))
        if not self.can_transfer(source, target, bytes_):
            return False
        # A callback-only ledger has already been checked for both sides.
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
        components: Mapping[str, Any] | Iterable[Any],
        *,
        tokens_per_page: int = 16,
        page_bytes: int = 1,
        page_bytes_by_layer: Optional[Mapping[str, int]] = None,
        ledger: Optional[Any] = None,
        topology: Optional[Any] = None,
        offload_components: Optional[Sequence[str]] = None,
        allow_cross_machine: bool = False,
        ledger_can_adjust: Optional[Callable[[str, int], bool]] = None,
        ledger_adjust: Optional[Callable[[str, int], bool]] = None,
        ledger_can_transfer: Optional[Callable[[str, str, int], bool]] = None,
        ledger_transfer: Optional[Callable[[str, str, int], bool]] = None,
    ) -> None:
        if isinstance(tokens_per_page, bool) or int(tokens_per_page) <= 0:
            raise KvPoolError("tokens_per_page must be positive")
        if isinstance(page_bytes, bool) or int(page_bytes) <= 0:
            raise KvPoolError("page_bytes must be positive")
        if isinstance(components, Mapping):
            raw_components = components.values()
        else:
            raw_components = components
        normalized = [KvPoolComponent.from_component(item) for item in raw_components]
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
        self.tokens_per_page = int(tokens_per_page)
        self.default_page_bytes = int(page_bytes)
        self.page_bytes_by_layer = {str(key): int(value) for key, value in (page_bytes_by_layer or {}).items()}
        limits = {item.component_id: item.capacity_bytes for item in normalized}
        self._ledger = _LedgerAdapter(
            limits,
            ledger,
            can_adjust=ledger_can_adjust,
            adjust=ledger_adjust,
            can_transfer=ledger_can_transfer,
            transfer=ledger_transfer,
        )
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
        self._bindings: Dict[int, Set[str]] = defaultdict(set)
        self._owned_bytes: Dict[str, int] = defaultdict(int)
        self._page_counter = 0
        self._clock = 0
        self._events: List[KvPoolTransfer] = []
        self._last_error: Optional[str] = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def events(self) -> Tuple[KvPoolTransfer, ...]:
        return tuple(self._events)

    @property
    def migration_time_ns(self) -> float:
        return sum(event.duration_ns for event in self._events)

    def clear_events(self) -> None:
        self._events.clear()

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
        try:
            return tuple(route(source, target, int(bytes_)))
        except TypeError:
            return tuple(route(source, target, int(bytes_), policy="lowest_latency"))

    def _reachable(self, source: Optional[str], target: str, bytes_: int) -> bool:
        if not source or source == target or self.topology is None:
            return True
        try:
            self._route(source, target, bytes_)
            return True
        except (ValueError, KeyError, KvPoolUnsupported):
            return False

    def _transfer_time_ns(self, source: str, target: str, bytes_: int) -> float:
        if source == target:
            return 0.0
        if self.topology is not None:
            hops = self._route(source, target, bytes_)
            return float(sum(float(hop.transfer_ns(bytes_)) for hop in hops))
        source_desc, target_desc = self.components[source], self.components[target]
        rates = [rate for rate in (source_desc.write_bandwidth_gbps, target_desc.read_bandwidth_gbps) if rate > 0]
        if not rates:
            return float(source_desc.latency_ns + target_desc.latency_ns)
        return float(source_desc.latency_ns + target_desc.latency_ns + 8.0 * bytes_ / min(rates))

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
            if self._component_free(component_id) < bytes_ or not self._ledger.can_adjust(component_id, bytes_):
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
        self._owned_bytes[page.owner_component] += page.bytes
        self._bindings[page.logical_page_id].add(page.request_id)

    def _free_page(self, page_id: int) -> None:
        page = self._pages.pop(page_id, None)
        if page is None:
            return
        if not self._ledger.adjust(page.owner_component, -page.bytes):
            raise RuntimeError("KV pool physical ledger rejected page release")
        self._owned_bytes[page.owner_component] -= page.bytes
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
        if binding is not None:
            self._bindings.get(page_id, set()).discard(binding)
        if page.ref_count <= 0:
            raise RuntimeError("KV pool page refcount underflow")
        page.ref_count -= 1
        if page.ref_count == 0 and not page.pinned:
            self._free_page(page_id)

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
        if not self._ledger.adjust(owner, bytes_):
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
        if target_pages == current_count:
            for page_id in current_ids:
                self._touch(self._pages[page_id])
            self._request_components.setdefault(request_id, request_component)
            return KvPoolResizeResult(True, request_id, current_count)

        if target_pages < current_count:
            removed = current_ids[target_pages:]
            for page_id in removed:
                self._drop_ref(page_id, request_id)
            self._requests[request_id] = current_ids[:target_pages]
            if not self._requests[request_id]:
                self._requests.pop(request_id, None)
                self._request_components.pop(request_id, None)
            return KvPoolResizeResult(True, request_id, target_pages, released_pages=len(removed))

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

    def resize(self, request_id: str, target_pages: int, **kwargs: Any) -> bool:
        """Resize and return only success, matching the legacy ledger API."""

        return self.resize_detailed(request_id, target_pages, **kwargs).success

    def resize_tokens(self, request_id: str, target_tokens: int, **kwargs: Any) -> bool:
        return self.resize_detailed(request_id, target_tokens=target_tokens, **kwargs).success

    def allocate(self, request_id: str, page_count: int, **kwargs: Any) -> bool:
        return self.resize(request_id, page_count, **kwargs)

    def release(self, request_id: str) -> None:
        page_ids = self._requests.pop(request_id, [])
        self._request_components.pop(request_id, None)
        for page_id in page_ids:
            self._drop_ref(page_id, request_id)

    def request_pages(self, request_id: str) -> Tuple[KvPage, ...]:
        return tuple(self._pages[page_id] for page_id in self._requests.get(request_id, ()) if page_id in self._pages)

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
            ids.append(page_id)
        for page_id in ids:
            page = self._pages[page_id]
            page.ref_count += 1
            page.prefix_key = str(prefix_key)
            self._touch(page)
        self._prefixes[str(prefix_key)] = ids

    def release_prefix(self, prefix_key: str) -> None:
        ids = self._prefixes.pop(str(prefix_key), [])
        for page_id in ids:
            self._drop_ref(page_id)

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
            page.pinned = False
            count += 1
            if page.ref_count == 0:
                self._free_page(page.logical_page_id)
        return count

    def unpin_page(self, logical_page_id: int) -> bool:
        """Unpin a page after its request binding has already been released."""

        page = self.page(logical_page_id)
        if not page.pinned:
            return False
        page.pinned = False
        if page.ref_count == 0:
            self._free_page(page.logical_page_id)
        return True

    def touch(self, request_id: str) -> None:
        for page in self.request_pages(request_id):
            self._touch(page)

    def evict(self, bytes_needed: int, *, component_ids: Optional[Iterable[str]] = None) -> int:
        """Evict unbound, unpinned LRU pages and return freed bytes."""

        if isinstance(bytes_needed, bool) or int(bytes_needed) < 0:
            raise KvPoolError("bytes_needed must be non-negative")
        allowed = {str(item) for item in component_ids} if component_ids is not None else None
        candidates = [
            page for page in self._pages.values()
            if page.ref_count > 0 and not page.pinned and not self._bindings.get(page.logical_page_id)
            and (allowed is None or page.owner_component in allowed)
        ]
        candidates.sort(key=lambda page: (page.last_access, page.logical_page_id))
        freed = 0
        for page in candidates:
            if freed >= int(bytes_needed):
                break
            freed += page.bytes
            # Drop the prefix reference (the only legal unbound reference).
            page.ref_count = 1
            self._free_page(page.logical_page_id)
        return freed

    def _move(self, page: KvPage, target_component: str, *, kind: str) -> bool:
        if target_component not in self.components:
            self._last_error = "unknown target component " + target_component
            return False
        target = self.components[target_component]
        if not target.writable:
            self._last_error = "target component is read-only: " + target_component
            return False
        if not self._reachable(page.owner_component, target_component, page.bytes):
            self._last_error = "no topology route from {} to {}".format(page.owner_component, target_component)
            return False
        if not self._ledger.can_transfer(page.owner_component, target_component, page.bytes):
            self._last_error = "target component has insufficient physical capacity: " + target_component
            return False
        source = page.owner_component
        if not self._ledger.transfer(source, target_component, page.bytes):
            self._last_error = "physical ledger rejected transfer {} -> {}".format(source, target_component)
            return False
        duration = self._transfer_time_ns(source, target_component, page.bytes)
        self._owned_bytes[source] -= page.bytes
        self._owned_bytes[target_component] += page.bytes
        page.owner_component = target_component
        page.resident_tier = target.tier
        self._touch(page)
        self._events.append(KvPoolTransfer(kind, page.logical_page_id, source, target_component, page.bytes, duration))
        self._last_error = None
        return True

    def migrate_page(self, logical_page_id: int, target_component: str) -> bool:
        page = self.page(logical_page_id)
        if not self.components[target_component].active:
            raise KvPoolUnsupported("migrate_page target must be active memory; use offload_page for backing tiers")
        return self._move(page, str(target_component), kind="migration")

    def offload_page(self, logical_page_id: int, target_component: Optional[str] = None) -> bool:
        page = self.page(logical_page_id)
        targets = [str(target_component)] if target_component is not None else sorted(self.offload_components)
        for target in targets:
            if target not in self.components or target not in self.offload_components:
                continue
            if self._move(page, target, kind="offload"):
                return True
        self._last_error = self._last_error or "no writable offload component has capacity"
        return False

    def restore_page(self, logical_page_id: int, target_component: Optional[str] = None) -> bool:
        page = self.page(logical_page_id)
        candidates = [str(target_component)] if target_component is not None else self._candidate_ids(
            page.layer_group,
            page.bytes,
            request_component=page.owner_component,
            compatible_components=None,
            active=True,
        )
        for target in candidates:
            if target in self.components and self.components[target].active and self._move(page, target, kind="restore"):
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


# Small aliases make the experimental API discoverable without committing to
# the longer class name in serving/config integration code.
KvPool = DynamicKVPool
PagedKVPool = DynamicKVPool
KVPage = KvPage


__all__ = [
    "DynamicKVPool",
    "KVPage",
    "KvPage",
    "KvPool",
    "KvPoolComponent",
    "KvPoolError",
    "KvPoolResizeResult",
    "KvPoolTransfer",
    "KvPoolUnsupported",
    "PagedKVPool",
    "PhysicalLedger",
]
