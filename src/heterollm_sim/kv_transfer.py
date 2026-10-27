"""Physical KV-page transfer executor.

This module owns the page-to-address mapping used by the paged KV pool.  It
never derives an address from a logical page number: every live page is
materialised through :class:`PhysicalAddressAllocator`.  A transfer reserves
source-memory, link/relay, and target-memory work on one shared
``PhysicalRuntimeContext``.  The returned ``KvPoolTransferPlan`` makes that
reservation reversible until the pool commits its capacity ledger.
"""
from __future__ import annotations

import copy
import heapq
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType as _MethodType, SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from .data_motion import (AccessKind, PhysicalRuntimeContext, PhysicalService, resolve_service,
                          _NATIVE_TIMELINE_HISTORY_SNAPSHOT)
from .ir import ComponentSpec
from .memory_allocator import PhysicalAddressAllocator, PhysicalAllocation
from .memory_types import DramConfig, NandConfig, SramConfig
from .physical_contract import require_physical_memory_config
from .kv_pool import KvPoolError, KvPoolTransferPlan, KvPoolTransferRequest


@dataclass
class _PageBinding:
    page_id: int
    request_id: str
    layer_group: str
    component_id: str
    owner: str
    bytes: int
    generation: int
    allocation: PhysicalAllocation


class PhysicalKVTransferExecutor:
    """Lower KV page movement and ordinary page access into physical events.

    ``components`` is a sequence of ``ComponentSpec`` values.
    ``topology`` uses the ``TopologyRouter`` route contract.  The optional
    ``relay_components`` sequence forces transfers through those components;
    each relay is written and then read, so relay memory and its own queue are
    charged explicitly.
    """

    def __init__(
        self,
        components: Sequence[ComponentSpec],
        *,
        runtime: Optional[PhysicalRuntimeContext] = None,
        topology: Any = None,
        relay_components: Sequence[str] = (),
        arrival_ns: Optional[float | Callable[[KvPoolTransferRequest], float]] = 0.0,
        namespace: str = "kv-pool",
        cache_states: Optional[Mapping[str, Any]] = None,
        cache_owners: Optional[Mapping[str, Any]] = None,
        retain_phase_details: bool = True,
    ) -> None:
        self.runtime = runtime or PhysicalRuntimeContext()
        self.topology = topology
        self.arrival_ns = arrival_ns
        self.namespace = str(namespace)
        self.cache_states = {} if cache_states is None else cache_states
        self.cache_owners = {} if cache_owners is None else cache_owners
        if not self.namespace:
            raise ValueError("namespace must be non-empty")
        if not isinstance(components, Sequence) or any(not isinstance(item, ComponentSpec) for item in components):
            raise ValueError("components must be a sequence of ComponentSpec")
        self.components = {item.component_id: item for item in components}
        self.relay_components = tuple(str(value) for value in relay_components)
        unknown = [value for value in self.relay_components if value not in self.components]
        if unknown:
            raise ValueError("unknown relay component(s): " + ", ".join(unknown))
        self.services: dict[str, PhysicalService] = {}
        self._configs: dict[str, DramConfig | NandConfig | SramConfig] = {}
        self._bindings: dict[int, _PageBinding] = {}
        # Plan/phase ownership belongs to snapshot-restorable executor state,
        # including synchronous plans whose commit is retained by the pool.
        self._async_plan_states: dict[int, dict[str, Any]] = {}
        # IDs deliberately do not rewind on restore: stale callbacks from a
        # discarded future must never name a newly prepared transfer.
        self._next_async_plan_id = 0
        self._next_async_phase_token = 0
        self._async_restore_epoch = 0
        self.events: list[dict[str, Any]] = []
        # Submitted work survives cancellation and detail-event retention. Rows
        # are immutable after append; snapshots need not clone the full history.
        self._cost_phases: list[dict[str, Any]] = []
        self.retain_phase_details = bool(retain_phase_details)
        self._cost_groups: dict[tuple[str, ...], dict[str, Any]] = {}
        self._cost_phase_total = 0
        self._compact_cost_savepoints = []
        self.counters = {"read_bytes": 0, "write_bytes": 0, "link_bytes": 0,
                         "relay_bytes": 0, "transfer_count": 0,
                         "service_ns": 0.0, "link_ns": 0.0}
        self._validate_components()

    def _validate_components(self) -> None:
        # Hardware graphs commonly include compute/IO components without a
        # physical memory core. Validate memory descriptors eagerly when they
        # are present, while resolving only page source/target components on
        # demand.
        for ident, component in self.components.items():
            if not ident:
                raise ValueError("physical component id must be non-empty")
            raw = component.metadata.get("physical_memory_config")
            if raw is None:
                continue
            self._ensure_service(ident)

    def _ensure_service(self, component_id: str) -> PhysicalService:
        ident = str(component_id)
        if ident in self.services:
            return self.services[ident]
        component = self.components.get(ident)
        if component is None:
            raise KvPoolError("unknown physical component " + ident)
        raw = component.metadata.get("physical_memory_config")
        if raw is None:
            raise KvPoolError("component {} requires physical_memory_config".format(ident))
        config = require_physical_memory_config(component)
        service = resolve_service(component)
        self.runtime.runtime(config, service.physical_owner)
        self._configs[ident] = config
        self.services[ident] = service
        return service

    def _page_id(self, page: Any) -> int:
        if not hasattr(page, "logical_page_id"):
            raise KvPoolError("page must provide logical_page_id")
        value = page.logical_page_id
        if type(value) is not int or value < 0:
            raise KvPoolError("logical page id must be an integer")
        return value

    def _binding(self, page: Any) -> _PageBinding:
        try:
            return self._bindings[self._page_id(page)]
        except KeyError as error:
            raise KvPoolError("KV page is not bound to a physical allocation") from error

    def _allocator(self, owner: str) -> PhysicalAddressAllocator:
        allocator = self.runtime.allocators.get(owner)
        if allocator is None:
            raise KvPoolError("physical owner {} has no address allocator".format(owner))
        return allocator

    def _buffer_id(self, page: Any) -> str:
        request = page.request_id
        return "{}:{}:page:{}".format(self.namespace, request, self._page_id(page))

    def allocation_bytes(self, component_id: str, payload_bytes: int) -> int:
        """Capacity footprint of an independently aligned physical page."""
        if type(payload_bytes) is not int or payload_bytes < 0:
            raise ValueError("page payload bytes must be a non-negative integer")
        service = self._ensure_service(component_id)
        alignment = self._allocator(service.physical_owner).alignment_bytes
        return ((payload_bytes + alignment - 1) // alignment) * alignment

    def bind_page(self, page: Any) -> int:
        """Allocate and bind one page at its declared owner component."""
        page_id = self._page_id(page)
        component_id = page.owner_component
        if component_id not in self.components:
            raise KvPoolError("unknown page owner component " + component_id)
        size = page.bytes
        generation = page.allocation_generation
        if type(size) is not int or size <= 0 or type(generation) is not int or generation < 0:
            raise KvPoolError("page bytes and allocation generation are invalid")
        existing = self._bindings.get(page_id)
        if existing is not None:
            if existing.component_id != component_id or existing.bytes != size or existing.generation != generation:
                raise KvPoolError("page binding changed without migration commit")
            if existing.request_id != page.request_id:
                # A session handoff changes logical ownership only. Preserve
                # the existing allocation and its physical address.
                self._bindings[page_id] = _PageBinding(existing.page_id,
                    str(page.request_id), existing.layer_group, existing.component_id,
                    existing.owner, existing.bytes, existing.generation, existing.allocation)
            return existing.allocation.base_address
        service = self._ensure_service(component_id)
        allocator = self._allocator(service.physical_owner)
        allocation = allocator.allocate(self._buffer_id(page), self.allocation_bytes(component_id, size), generation,
                                        allocation_arena="resident")
        self._bindings[page_id] = _PageBinding(
            page_id, page.request_id,
            page.layer_group, component_id,
            service.physical_owner, size, generation, allocation,
        )
        return allocation.base_address

    def sync_pages(self, pages: Sequence[Any]) -> None:
        """Bind pages that were allocated by a pool after a resize."""
        current = {self._page_id(page) for page in pages}
        # Pool release can remove a page without involving this executor.  Drop
        # the corresponding physical allocation before the next migration so
        # allocator capacity and the logical pool stay in lockstep.
        for page_id in tuple(self._bindings):
            if page_id not in current:
                self._release_page_id(page_id)
        for page in pages:
            self.bind_page(page)

    def address(self, page: Any, offset_bytes: int = 0, size_bytes: Optional[int] = None) -> int:
        binding = self._binding(page)
        if type(offset_bytes) is not int or offset_bytes < 0:
            raise ValueError("offset_bytes must be a non-negative integer")
        if size_bytes is not None and (type(size_bytes) is not int or size_bytes < 0):
            raise ValueError("size_bytes must be a non-negative integer")
        size = binding.bytes - offset_bytes if size_bytes is None else size_bytes
        if size < 0 or offset_bytes + size > binding.bytes:
            raise ValueError("KV access exceeds logical page payload")
        return binding.allocation.address(offset_bytes, size)

    def release_page(self, page: Any) -> None:
        self._release_page_id(self._page_id(page))

    def _release_page_id(self, page_id: int) -> None:
        binding = self._bindings.get(page_id)
        if binding is None:
            return
        self._release_allocation(binding.owner, binding.allocation.buffer_id, binding.generation)
        self._bindings.pop(page_id)

    def cache_access(self, page: Any, *, offset_bytes: int, size_bytes: int,
                     operation: str):
        """Export a cache range that borrows this page's actual allocation.

        Logical session handoff does not rename storage, and migration versions
        are allocation generations rather than part of a second buffer name.
        """
        from .cache_state import CacheAccess
        binding = self._binding(page)
        self.address(page, offset_bytes, size_bytes)
        return CacheAccess(binding.allocation.buffer_id, offset_bytes, size_bytes,
            operation, binding.allocation.size_bytes, binding.generation)

    def _release_allocation(self, owner: str, buffer_id: str, generation: int) -> None:
        allocator = self._allocator(owner)
        allocation = allocator.get_allocation(buffer_id, generation)
        active = self.runtime.runtimes[owner]
        if allocation is not None and isinstance(active.core.config, NandConfig):
            # NAND allocations start on page boundaries; their rounded tail
            # is also exclusively owned. Never leave dirty staging data that
            # could later program this address after the allocation is reused.
            page_bytes = active.core.config.page_bytes
            rounded_bytes = ((allocation.size_bytes + page_bytes - 1) // page_bytes) * page_bytes
            active.core.discard_staged_pages(allocation.base_address, rounded_bytes,
                                            now_ns=active.clock_ns)
        allocator.release(buffer_id, generation)
        for cache in self._owner_caches(owner):
            cache.discard_buffer(buffer_id, generation)

    def _owner_caches(self, owner):
        return tuple(self.cache_states[key][1] for key in self.cache_owners.get(owner, ())
                     if key in self.cache_states)

    def _snapshot_caches(self):
        return {key: (state[0], state[1], copy.deepcopy(vars(state[1])))
                for key, state in self.cache_states.items()}

    def _restore_caches(self, snapshot):
        self.cache_states.clear()
        for key, (signature, cache, state) in snapshot.items():
            vars(cache).clear()
            vars(cache).update(copy.deepcopy(state))
            self.cache_states[key] = (signature, cache)

    def _flush_source_caches(self, binding, arrival_ns):
        """Make dirty GPU cache lines visible before the migration source read."""
        current, phases = arrival_ns, []
        for cache in self._owner_caches(binding.owner):
            flush = cache.flush(binding.allocation.buffer_id, allocation_generation=binding.generation)
            for access in flush.backing_accesses:
                address = binding.allocation.address(access.offset_bytes, access.size_bytes)
                bills = self._price_chunks(binding.component_id, AccessKind.WRITE,
                    address, access.size_bytes, arrival_ns=current, caller_managed_transaction=True)
                phases.extend({"kind": "source_cache_writeback", **bill} for bill in bills)
                current = float(bills[-1]["completion_ns"]) if bills else current
        return current, phases

    def _arrival(self, request: Optional[KvPoolTransferRequest], value: Optional[float]) -> float:
        if value is not None:
            result = float(value)
        elif callable(self.arrival_ns):
            result = float(self.arrival_ns(request))
        elif self.arrival_ns is None:
            result = 0.0
        else:
            result = float(self.arrival_ns)
        if not math.isfinite(result) or result < 0:
            raise ValueError("arrival_ns must be finite and non-negative")
        return result

    @staticmethod
    def _sum_counters(total: dict[str, Any], result: Mapping[str, Any]) -> None:
        for key, value in result.items():
            if key == "nand_observations" and value is not None:
                from .nand_core import summarize_nand_observations
                # Raw Core observations and an already summarized prefix both
                # retain their events and per-owner geometry. Recompute the
                # derived queue/plane statistics over their joint population.
                total[key] = summarize_nand_observations((total.get(key), value))
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key == "arrival_ns":
                    total[key] = min(total.get(key, value), value)
                elif key in {"completion_ns", "host_ack_completion_ns", "media_drain_completion_ns", "write_buffer_peak_bytes",
                             "write_buffer_release_completion_ns",
                             "host_outstanding_at_admission_peak", "host_read_outstanding_at_admission_peak",
                             "host_write_outstanding_at_admission_peak"}:
                    total[key] = max(total.get(key, value), value)
                elif key in {"write_buffer_occupancy_bytes", "bandwidth_ceiling_gb_s", "array_units", "actual_bandwidth_gb_s"}:
                    total[key] = value
                else:
                    total[key] = total.get(key, 0) + value
            elif isinstance(value, Mapping):
                nested = total.setdefault(key, {})
                for resource, amount in value.items():
                    if isinstance(amount, (int, float)):
                        nested[resource] = nested.get(resource, 0) + amount
                    elif key in {"resource_intervals", "resource_interval_payloads"}:
                        nested[resource] = (*nested.get(resource, ()), *amount)
                    else:
                        nested[resource] = amount
            else:
                total[key] = value

    @contextmanager
    def _io_transaction(self, *, owners=None, _owned_timeline=False):
        """Roll back one logical access/phase without copying event history."""
        saved_runtime = (self.runtime.snapshot() if owners is None else
            self.runtime.begin_transaction(owners=owners, _owned_timeline=_owned_timeline))
        saved_counters = dict(self.counters)
        cost_count = len(self._cost_phases)
        compact_costs = None
        if not self.retain_phase_details:
            protect_all = (owners is None or not self._native_compact_cost_writer()
                or _owned_timeline and not saved_runtime.get("_owned_native_writers", False))
            # A nested phase may introduce an owner outside its parent's
            # declared phase. Save that group's original value in each outer
            # checkpoint before an unknown writer can mutate the shared
            # object. Closed native retention replaces flat group maps;
            # its previous values remain immutable for every ancestor.
            if protect_all:
                for ancestor in self._compact_cost_savepoints:
                    for key, group in self._cost_groups.items():
                        if ancestor.get(key) is group:
                            ancestor[key] = copy.deepcopy(group)
                saved_groups = copy.deepcopy(self._cost_groups)
            else:
                saved_groups = dict(self._cost_groups)
            compact_costs = saved_groups, self._cost_phase_total
            self._compact_cost_savepoints.append(saved_groups)
        cache_transactions = []
        try:
            for _key, (_signature, cache) in self.cache_states.items():
                cache_transactions.append((cache, cache.begin_transaction()))
            yield saved_runtime
        except Exception:
            self.runtime.restore(saved_runtime)
            self.counters = saved_counters
            del self._cost_phases[cost_count:]
            if compact_costs is not None:
                self._cost_groups, self._cost_phase_total = compact_costs
            for cache, transaction in reversed(cache_transactions):
                cache.rollback_transaction(transaction)
            raise
        else:
            if owners is not None:
                self.runtime.commit_transaction(saved_runtime)
            for cache, transaction in cache_transactions:
                cache.commit_transaction(transaction)
        finally:
            if compact_costs is not None:
                self._compact_cost_savepoints.pop()

    def _native_compact_cost_writer(self):
        """Unknown statistics hooks keep the original whole-ledger rollback."""
        return all(getattr(getattr(self, name), "__func__", getattr(self, name)) is original
            for name, original in _NATIVE_COMPACT_COST_WRITERS)

    def _retain_cost_phase(self, phase):
        if self.retain_phase_details:
            self._cost_phases.append(phase)
            return
        # Only reporting retention changes. Each phase has already made its
        # original Core/link submission, and every byte/service counter stays
        # additive. Interval and page-event populations are explicitly absent
        # rather than growing with the number of imported prompt chunks.
        key = (phase["kind"], str(phase.get("owner", phase.get("link_id"))),
            str(phase.get("component_id", "")), str(phase["operation"]))
        previous = self._cost_groups.get(key)
        # Native aggregation mutates only each group's outer dictionary and
        # its flat metric maps. Copy these columns in C instead of recursively
        # cloning accumulated scalar resource counters before every PD chunk.
        # Unknown callbacks retain the full transactional copy above.
        group = {} if previous is None else {name: dict(value) if isinstance(value, Mapping) else value
                                             for name, value in previous.items()}
        count = group.get("submitted_phase_count", 0) + 1
        start = min(group.get("start_ns", phase.get("start_ns", phase["arrival_ns"])),
            phase.get("start_ns", phase["arrival_ns"]))
        skip = {"resource_intervals", "resource_interval_payloads", "nand_observations",
            "resource_busy_by_direction_ns", "start_ns", "duration_ns"}
        # Physical bills may expose expanded stages/pages under opt-in trace
        # settings. Keep fixed resource maps and scalar/owner facts only.
        values = {name: value for name, value in phase.items() if name not in skip
            and (value is None or isinstance(value, (str, bool, int, float))
                 or (isinstance(value, Mapping) and all(
                     isinstance(amount, (str, bool, int, float)) for amount in value.values()))
                 or name in {"host_admission_depth_scopes"})}
        values = {name: value for name, value in values.items()
            if not (name in group and group[name] is None and isinstance(value, (int, float)))}
        self._sum_counters(group, values)
        direction = "read_write" if phase["kind"] == "link" else str(phase["operation"]).lower()
        self._cost_groups[key] = group
        group.update(start_ns=start, duration_ns=group["completion_ns"]-group["arrival_ns"],
            phase_id="paged-kv-physical-group-" + str(tuple(self._cost_groups).index(key)),
            submitted_phase_count=count, intervals_truncated=True,
            resource_busy_by_direction_ns={direction: dict(group.get("resource_busy_ns", {}))})
        self._cost_phase_total += 1

    def _cost_phase_count(self):
        return len(self._cost_phases) if self.retain_phase_details else self._cost_phase_total

    def _record_memory_cost(self, component_id, kind, bill):
        owner = self.services[component_id].physical_owner
        raw = dict(bill)
        raw["logical_read_bytes"] = int(raw.get("logical_bytes", 0)) if kind is AccessKind.READ else 0
        raw["logical_write_bytes"] = int(raw.get("logical_bytes", 0)) if kind is AccessKind.WRITE else 0
        direction = "read" if kind is AccessKind.READ else "write"
        raw.update({"phase_id": "paged-kv-physical-" + str(self._cost_phase_count()),
            "kind": "memory", "component_id": component_id, "owner": owner,
            "memory_kind": self._configs[component_id].kind.value,
            "resource_owners": {resource: owner for resource in raw.get("resource_busy_ns", {})},
            "resource_busy_by_direction_ns": {direction: dict(raw.get("resource_busy_ns", {}))},
            "direction_semantics": "attributed to the submitted transaction direction; a NAND write may include RMW reads"})
        self._retain_cost_phase(raw)
        return raw

    def _submit_link_phase(self, hop, size, arrival_ns):
        """Use the same payload-wire and inflight-credit contract as DAG hops."""
        timeline = self.runtime.timeline
        demands = hop.demands(size)
        for resource, capacity in hop.resource_capacities.items():
            resource = timeline.canonical_resource_id(resource)
            existing = timeline.lane_available.get(resource)
            if existing is None:
                timeline.lane_available[resource] = [timeline.available_ns(resource)] * capacity
            elif len(existing) != capacity:
                raise KvPoolError("link inflight capacity conflicts with the shared resource calendar")
        combined = {}
        for demand in demands:
            resource = timeline.canonical_resource_id(demand.resource_id)
            row = combined.setdefault(resource, {"busy": 0., "bytes": 0})
            row["busy"] += demand.service_ns
            row["bytes"] += demand.bytes_moved
        start = max((arrival_ns, *(timeline.available_ns(resource) for resource in combined)))
        busy, moved, intervals, payloads = {}, {}, {}, {}
        end = start
        for resource, cost in combined.items():
            reservation = timeline.reserve(resource, start, cost["busy"], direction="read_write")
            end = max(end, reservation.end_ns)
            busy[resource], moved[resource] = cost["busy"], cost["bytes"]
            timeline.bytes_moved[resource] = timeline.bytes_moved.get(resource, 0) + cost["bytes"]
            intervals[resource] = ((reservation.start_ns, reservation.end_ns),)
            payloads[resource] = ((reservation.start_ns, reservation.end_ns, cost["bytes"]),)
        phase = {
            "phase_id": "paged-kv-physical-" + str(self._cost_phase_count()),
            "kind": "link", "link_id": hop.link_id, "operation": "copy",
            "resource_id": timeline.canonical_resource_id(hop.resource_id), "bytes": size,
            "arrival_ns": arrival_ns, "start_ns": start, "completion_ns": end,
            "duration_ns": end - start,
            "logical_bytes": size, "physical_bytes": size, "resource_busy_ns": busy, "resource_bytes": moved,
            "resource_intervals": intervals,
            "resource_interval_payloads": payloads,
            "resource_busy_by_direction_ns": {"read_write": busy},
            "intervals_truncated": False}
        self._retain_cost_phase(phase)
        self.counters["link_bytes"] += size
        self.counters["link_ns"] += end - start
        return phase

    def cost_ledger(self) -> Mapping[str, Any]:
        """Summarize already submitted Core/link work, without executing it again.

        Times are admission/completion envelopes, not occupied-time sums. Busy
        time is summed per resource and may exceed elapsed time for parallel
        slots. Core interval detail can be truncated independently of totals.
        """
        from .planner import _summarize_dram_task_traffic, _summarize_nand_task_traffic
        result = {"schema": "heterollm.paged-kv-physical-cost/v1",
            "phase_count": self._cost_phase_count(), "resource_accounted_bytes": 0, "resource_busy_ns": {},
            "resource_bytes": {}, "resource_busy_by_direction_ns": {}, "owners": {}, "links": {},
            "first_arrival_ns": None, "last_completion_ns": None}
        adapters = []
        phases = self._cost_phases if self.retain_phase_details else self._cost_groups.values()
        memory_counts, operation_counts = {}, {}
        for phase in phases:
            count = phase.get("submitted_phase_count", 1)
            start, end = phase["arrival_ns"], phase["completion_ns"]
            result["first_arrival_ns"] = start if result["first_arrival_ns"] is None else min(start, result["first_arrival_ns"])
            result["last_completion_ns"] = end if result["last_completion_ns"] is None else max(end, result["last_completion_ns"])
            for key in ("resource_busy_ns", "resource_bytes"):
                for resource, value in phase.get(key, {}).items():
                    result[key][resource] = result[key].get(resource, 0) + value
            for direction, values in phase["resource_busy_by_direction_ns"].items():
                target = result["resource_busy_by_direction_ns"].setdefault(direction, {})
                for resource, value in values.items():
                    target[resource] = target.get(resource, 0) + value
            if phase["kind"] == "memory":
                owner = phase["owner"]
                row = result["owners"].setdefault(owner, {"component_ids": [],
                    "memory_kind": phase["memory_kind"], "transaction_count": 0})
                if phase["component_id"] not in row["component_ids"]:
                    row["component_ids"].append(phase["component_id"])
                row["transaction_count"] += count
                for key in ("logical_bytes", "logical_read_bytes", "logical_write_bytes",
                            "physical_bytes", "physical_read_bytes", "physical_write_bytes",
                            "host_transfer_bytes", "internal_transfer_bytes"):
                    # Unknown counters must not become measured zero.
                    value = phase.get(key)
                    row[key] = None if value is None or (key in row and row[key] is None) else row.get(key, 0) + value
                raw = {**phase, "kind": phase["memory_kind"]}
                traffic = ("storage_traffic" if str(phase["memory_kind"]).upper() in {"HBF", "SSD"}
                    else "dram_traffic")
                memory_counts[traffic] = memory_counts.get(traffic, 0) + count
                operations = operation_counts.setdefault(traffic, {})
                operation = str(phase.get("operation", "unknown"))
                operations[operation] = operations.get(operation, 0) + count
                adapters.append(SimpleNamespace(task_id=phase["phase_id"],
                    metadata={"physical_execution_by_owner": {owner: raw}},
                    demands=tuple(SimpleNamespace(resource_id=resource)
                                  for resource in phase.get("resource_busy_ns", {}))))
            else:
                row = result["links"].setdefault(phase["link_id"], {"phase_count": 0, "bytes": 0})
                row["phase_count"] += count
                row["bytes"] += phase["physical_bytes"]
        result["resource_accounted_bytes"] = sum(result["resource_bytes"].values())
        result["storage_traffic"] = _summarize_nand_task_traffic(adapters)
        result["dram_traffic"] = _summarize_dram_task_traffic(adapters)
        for key in ("storage_traffic", "dram_traffic"):
            result[key] = {**result[key], "count_semantics": "task_count here counts submitted migration memory phases, not kernel tasks"}
            if not self.retain_phase_details:
                result[key]["task_count"] = memory_counts.get(key, 0)
                if "operation_counts" in result[key]:
                    result[key]["operation_counts"] = dict(sorted(operation_counts.get(key, {}).items()))
                result[key]["detail_scope"] = "bounded owner/direction totals; interval and NAND page-event detail not retained"
                for availability in result[key].get("metric_availability", {}).values():
                    for field in ("recorded_owner_rows", "physical_owner_rows"):
                        if field in availability:
                            availability[field] = memory_counts.get(key, 0) if (
                                field == "physical_owner_rows" or availability["status"] == "recorded") else 0
        result["phases"] = tuple(copy.deepcopy(self._cost_phases))
        if not self.retain_phase_details:
            result.update(phase_detail_retained=False, phase_group_count=len(self._cost_groups))
        return result

    def _price_chunks(self, component_id: str, kind: AccessKind, address: int,
                      size: int, *, arrival_ns: float,
                      caller_managed_transaction: bool = False) -> tuple[dict[str, Any], ...]:
        service = self._ensure_service(component_id)
        price = service._price if caller_managed_transaction else service.price
        config = self._configs[component_id]
        limit = config.write_buffer_bytes if isinstance(config, NandConfig) and kind is AccessKind.WRITE else 0
        granularity = config.host_granularity_bytes if isinstance(config, NandConfig) else config.burst_bytes
        if limit > 0 and limit < granularity:
            raise KvPoolError("write buffer is smaller than one host transaction granularity")
        chunk = min(size, limit) if limit > 0 else size
        if limit > 0:
            chunk = max(granularity, (chunk // granularity) * granularity)
        chunk = max(1, chunk)
        results = []
        offset = 0
        current = arrival_ns
        while offset < size:
            amount = min(chunk, size - offset)
            if limit > 0:
                # A partial first host granule still consumes a full transfer.
                # End chunks on a granule boundary so rounding cannot exceed
                # the finite staging capacity.
                amount = min(amount, chunk - (address + offset) % granularity)
            bill = price(kind, amount, page_offset_bytes=address + offset,
                         runtime=self.runtime, arrival_ns=current)
            self._record_memory_cost(component_id, kind, bill)
            results.append(dict(bill))
            current = float(bill.get("completion_ns", current))
            offset += amount
        if kind is AccessKind.READ:
            self.counters["read_bytes"] += size
        elif kind is AccessKind.WRITE:
            self.counters["write_bytes"] += size
        self.counters["service_ns"] += sum(
            float(item.get("service_ns", 0.0)) for item in results
        )
        return tuple(results)

    def read_page(self, page: Any, *, arrival_ns: Optional[float] = None,
                  offset_bytes: int = 0, size_bytes: Optional[int] = None) -> Mapping[str, Any]:
        return self._page_access(page, AccessKind.READ, arrival_ns=arrival_ns,
                                 offset_bytes=offset_bytes, size_bytes=size_bytes)

    def write_page(self, page: Any, *, arrival_ns: Optional[float] = None,
                   offset_bytes: int = 0, size_bytes: Optional[int] = None) -> Mapping[str, Any]:
        return self._page_access(page, AccessKind.WRITE, arrival_ns=arrival_ns,
                                 offset_bytes=offset_bytes, size_bytes=size_bytes)

    def _page_access(self, page: Any, kind: AccessKind, *, arrival_ns,
                     offset_bytes, size_bytes) -> Mapping[str, Any]:
        # A caller observes one logical page operation. A rejected final chunk
        # or failed statistics reduction must not publish its successful prefix.
        with self._io_transaction():
            binding = self._binding(page)
            address = self.address(page, offset_bytes, size_bytes)
            size = binding.bytes - offset_bytes if size_bytes is None else size_bytes
            total: dict[str, Any] = {"operation": kind.value, "logical_bytes": 0}
            for bill in self._price_chunks(binding.component_id, kind, address, size,
                    arrival_ns=self._arrival(None, arrival_ns), caller_managed_transaction=True):
                self._sum_counters(total, bill)
            if total.get("completion_ns") is None:
                total["completion_ns"] = total.get("arrival_ns", 0.0)
            total["service_ns"] = total["completion_ns"] - total.get("arrival_ns", 0.0)
            total["actual_bandwidth_gb_s"] = total["logical_bytes"] / total["service_ns"] if total["service_ns"] else 0.0
            return total

    def _route(self, source: str, target: str, size: int) -> Sequence[Any]:
        if source == target:
            return ()
        if self.topology is None:
            return ()
        return tuple(self.topology.route(source, target, size))

    def prepare_external_import(self, *, source_component: str, destinations: Sequence[Mapping[str, Any]],
                                buffer_bytes: int, chunk_bytes: int, request_id: str,
                                arrival_ns: float) -> KvPoolTransferPlan:
        """Stream external historical KV into the native lowerer's declared buffers.

        The destination ranges split into one stream per target component
        (a rank's own KV store behind its own PD link); the streams advance
        concurrently. ``buffer_bytes`` is this request's own source staging
        buffer, cut into ``chunk_bytes`` slots dealt round-robin to the
        streams, at least one each. A chunk holds its slot from its source
        read until its target write completes, as an RDMA source buffer stays
        valid until the remote write is acknowledged, so a stream keeps at
        most its slot count in flight while the reads, links and writes of
        adjacent chunks overlap on the native resource timelines. Every
        submitted action after preparation is one plan phase. Consumers may
        publish the imported context only after every stream's last write
        completes and commit. External Prefill production is outside this
        Decode-node runtime, never a free destination-memory initialization.
        """
        if (type(buffer_bytes) is not int or type(chunk_bytes) is not int
                or min(buffer_bytes, chunk_bytes) <= 0 or not request_id):
            raise KvPoolError("external KV requires positive finite buffer/chunk sizes and a request id")
        rows = []
        streams = {}
        for item in destinations:
            row = dict(item)
            if (not row.get("component_id") or not row.get("buffer_id")
                    or type(row.get("offset_bytes")) is not int or row["offset_bytes"] < 0
                    or type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0):
                raise KvPoolError("invalid external KV destination range")
            service = self._ensure_service(row["component_id"])
            row["physical_owner"] = service.physical_owner
            row["generation"] = int(row.get("generation", 0))
            # Validate exact owner-scoped bounds before reserving any I/O.
            self._allocator(service.physical_owner).address(row["buffer_id"], row["offset_bytes"],
                row["size_bytes"], row["generation"])
            streams.setdefault(str(row["component_id"]), []).append(len(rows))
            rows.append(row)
        if not rows:
            raise KvPoolError("external KV import requires at least one nonempty destination")
        slots = buffer_bytes // chunk_bytes
        if slots < len(streams):
            raise KvPoolError("external KV source buffer must hold one chunk slot per target stream")
        snapshot_method = self.snapshot
        snapshot = (snapshot_method(_immutable_timeline=True)
            if type(snapshot_method) is _MethodType
            and snapshot_method.__func__ is _IMMUTABLE_EXECUTOR_SNAPSHOT else snapshot_method())
        source = self._ensure_service(source_component)
        size = slots * chunk_bytes
        identity = self.namespace + ":external:" + request_id
        start = self._arrival(None, arrival_ns)
        try:
            allocation = self._allocator(source.physical_owner).allocate(identity, size, 0,
                allocation_arena="workspace")
            relays = []
            for component in self.relay_components:
                owner = self._ensure_service(component).physical_owner
                allocated = self._allocator(owner).allocate(identity + ":relay:" + component,
                    self.allocation_bytes(component, size), 0, allocation_arena="workspace")
                relays.append((component, owner, allocated))
            plan_id = self._next_async_plan_id
            self._next_async_plan_id += 1
            # A stream is [destination row indices, row cursor, byte offset,
            # free slots]; a chunk is [stream, slot, actions, next action].
            state = {"plan_id": plan_id, "done": False, "phase": 0,
                "request_id": request_id, "source_component": source_component,
                "source_owner": source.physical_owner, "source_allocation": allocation,
                "relays": tuple(relays), "destinations": tuple(rows),
                "streams": [[tuple(indices), 0, 0, list(range(index, slots, len(streams)))]
                            for index, indices in enumerate(streams.values())],
                "chunk_bytes": chunk_bytes, "slot_count": slots, "chunks": {}, "events": [],
                "next_chunk_id": 0, "submitted_actions": 0, "submitted_write_bytes": 0,
                "last_write": None, "logical_bytes": sum(r["size_bytes"] for r in rows),
                "start_ns": start, "completion_ns": start, "submitted_until_ns": start,
                "rollback_snapshot": snapshot}
            self._async_plan_states[plan_id] = state
            # Fill every stream's slots round-robin, so all target ranks start
            # together and share the source in arrival order.
            changed, progressed = [], True
            while progressed:
                progressed = False
                for index, stream in enumerate(state["streams"]):
                    if stream[3] and stream[1] < len(stream[0]):
                        changed.append(self._start_external_chunk(state, index, stream[3][0], start))
                        stream[3].pop(0)
                        progressed = True
            return self._external_plan(state, state, None if None in changed
                else tuple(dict.fromkeys(resource for value in changed for resource in value)))
        except BaseException:
            self.restore(snapshot)
            raise

    def _external_chunk_actions(self, state, row, offset, amount, slot):
        base = slot * state["chunk_bytes"]
        actions = [("memory", state["source_component"], "read",
                    state["source_allocation"].address(base, amount), amount)]
        previous = state["source_component"]
        for component, _owner, allocation in state["relays"]:
            actions.extend(("link", hop, amount) for hop in self._route(previous, component, amount)
                           if not hop.service_ref)
            address = allocation.address(base, amount)
            actions.extend((("memory", component, "write", address, amount),
                            ("memory", component, "read", address, amount)))
            previous = component
        actions.extend(("link", hop, amount) for hop in self._route(previous, row["component_id"], amount)
                       if not hop.service_ref)
        address = self._allocator(row["physical_owner"]).address(row["buffer_id"],
            row["offset_bytes"] + offset, amount, row["generation"])
        actions.append(("memory", row["component_id"], "write", address, amount))
        return tuple(actions)

    def _start_external_chunk(self, state, stream_index, slot, now):
        """Submit the next chunk of a stream into ``slot``; returns its changed lanes.

        The cursor moves only after the source read was submitted, so a
        rejected submission leaves the plan state unchanged.
        """
        stream = state["streams"][stream_index]
        row = state["destinations"][stream[0][stream[1]]]
        offset = stream[2]
        amount = min(state["chunk_bytes"], row["size_bytes"] - offset)
        actions = self._external_chunk_actions(state, row, offset, amount, slot)
        changed, completion = self._submit_external_action(state, actions[0], now)
        if offset + amount == row["size_bytes"]:
            stream[1] += 1
            stream[2] = 0
        else:
            stream[2] = offset + amount
        chunk_id = state["next_chunk_id"]
        state["next_chunk_id"] += 1
        state["chunks"][chunk_id] = [stream_index, slot, actions, 1]
        heapq.heappush(state["events"], (completion, chunk_id))
        return changed

    def _submit_external_action(self, state, action, now):
        """Submit one source/relay/target memory or link action at ``now``."""
        changed_resources = None
        if action[0] == "memory":
            _, component, operation, address, amount = action
            # Protect exactly the submitted owner, using the native Core undo
            # journal. PhysicalService.price would snapshot all GPU cores for
            # every prompt chunk even though only this owner can be changed.
            service = self._ensure_service(component)
            owner = service.physical_owner
            native_writer = (type(self) is PhysicalKVTransferExecutor
                and self._native_compact_cost_writer()
                and type(service) is PhysicalService
                and getattr(service._price, "__func__", service._price) is _NATIVE_EXTERNAL_MEMORY_PRICE)
            # This private native phase submits one known memory owner. Its
            # Core may journal the touched resource columns. Unknown service
            # or statistics hooks can edit other owners and keep a full undo.
            with self._io_transaction(owners=(owner,) if native_writer else None,
                                      _owned_timeline=native_writer) as checkpoint:
                bills = self._price_chunks(component, AccessKind.READ if operation == "read" else AccessKind.WRITE,
                    address, amount, arrival_ns=now, caller_managed_transaction=True)
            completion = float(bills[-1]["completion_ns"])
            journal = checkpoint["timeline"].get("_owned_transaction")
            if native_writer and checkpoint.get("_owned_native_writers") and journal is not None:
                # The committed undo journal names every potentially written
                # lane, including virtual NAND array frontiers and HOST's
                # direct native publication. It is independent of retained
                # traffic summaries and never infers an owner from a prefix.
                changed_resources = tuple(dict.fromkeys(resource
                    for entries in journal.entries.values() for resource in entries))
        else:
            _, hop, amount = action
            bill = self._submit_link_phase(hop, amount, now)
            completion = bill["completion_ns"]
            changed_resources = _native_external_link_resources(self, hop, bill)
        state["submitted_actions"] += 1
        state["submitted_until_ns"] = max(state["submitted_until_ns"], completion)
        return changed_resources, completion

    def _external_plan(self, state, callback_state, changed_resources):
        # The next phase is the earliest completion that submits further
        # work. With none left, the import ends when its last write does.
        final = not state["events"]
        completion = state["submitted_until_ns"] if final else state["events"][0][0]
        token = self._new_phase_token()
        state.update(completion_ns=completion, phase_token=token, final=final)
        phase = state["phase"]
        return KvPoolTransferPlan(duration_ns=completion-state["start_ns"], start_ns=state["start_ns"],
            completion_ns=completion,
            native_changed_resource_ids=changed_resources,
            advance=None if final else lambda: self._advance_external_phase(callback_state, token),
            commit=lambda: self._finish_external_import(callback_state, token, cancelled=False),
            cancel=lambda: self._finish_external_import(callback_state, token, cancelled=True),
            rollback=lambda: self._rollback_external_import(callback_state, token, phase))

    def _advance_external_phase(self, callback_state, token):
        """Handle the earliest completed action: at most one new submission.

        A completed source read, relay access or link submits its chunk's next
        action. A completed target write frees its slot, which at once starts
        the stream's next chunk. Plan state changes only after the submission
        succeeded, so a rejected phase leaves the import exactly as it was.
        """
        state = self._transfer_state(callback_state)
        self._check_phase_token(state, token)
        if state["done"]:
            raise KvPoolError("external KV import is no longer active")
        if state["final"]:
            raise KvPoolError("external KV import has no unsubmitted phases")
        now, chunk_id = state["events"][0]
        chunk = state["chunks"][chunk_id]
        stream_index, slot, actions, position = chunk
        stream = state["streams"][stream_index]
        unstarted = stream[1] < len(stream[0])
        write = None
        if position < len(actions):
            action = actions[position]
            changed, completion = self._submit_external_action(state, action, now)
            heapq.heappop(state["events"])
            chunk[3] = position + 1
            if position + 1 < len(actions) or unstarted:
                # Only a write whose slot release can start another chunk
                # needs its completion as a phase.
                heapq.heappush(state["events"], (completion, chunk_id))
            else:
                del state["chunks"][chunk_id]
            if position + 1 == len(actions):
                write = (completion, action[4])
                state["submitted_write_bytes"] += action[4]
        else:
            changed = (self._start_external_chunk(state, stream_index, slot, now)
                       if unstarted else ())
            if not unstarted:
                stream[3].append(slot)
            # A started chunk pushed a later (completion, id) than this
            # minimum, so the released phase is still at the heap's top.
            heapq.heappop(state["events"])
            del state["chunks"][chunk_id]
        state["phase"] += 1
        state["last_write"] = write
        state.pop("rollback_snapshot", None)
        return self._external_plan(state, callback_state, changed)

    def _finish_external_import(self, callback_state, token, *, cancelled):
        with self._finalization_transaction(callback_state):
            state = self._transfer_state(callback_state)
            self._check_phase_token(state, token)
            if state["done"]:
                if state.get("cancelled") != cancelled:
                    raise KvPoolError("external KV import already finalized with a different outcome")
                return state["terminal_event"]
            if not cancelled and not state["final"]:
                raise KvPoolError("cannot publish external KV before all phases complete")
            self._release_allocation(state["source_owner"], state["source_allocation"].buffer_id, 0)
            for _component, owner, allocation in state["relays"]:
                self._release_allocation(owner, allocation.buffer_id, 0)
            # A cancelled import still drains every action already submitted;
            # the caller waits for that before the buffers can be reused.
            billed = {"kind": "external_kv_cancelled" if cancelled else "external_kv_import",
                "request_id": state["request_id"], "logical_bytes": state["logical_bytes"],
                "start_ns": state["start_ns"], "completion_ns": state["submitted_until_ns"],
                "submitted_phases": state["submitted_actions"],
                "buffer_bytes": state["source_allocation"].size_bytes,
                "slot_count": state["slot_count"], "stream_count": len(state["streams"])}
            self.events.append(billed)
            counter = "cancelled_transfer_count" if cancelled else "transfer_count"
            self.counters[counter] = self.counters.get(counter, 0) + 1
            state.update(cancelled=cancelled, terminal_event=billed)
            state.pop("rollback_snapshot", None)
            self._finish_transfer(state, callback_state)
            return billed

    def _rollback_external_import(self, callback_state, token, phase):
        state = self._transfer_state(callback_state)
        self._check_phase_token(state, token)
        if phase != 0 or state["phase"] != 0:
            raise KvPoolError("only a just-prepared external import can roll back")
        snapshot = state["rollback_snapshot"]
        self.restore(snapshot)
        callback_state.update(done=True, terminal_epoch=self._async_restore_epoch)

    def _link_segment(self, source: str, target: str, size: int, *, arrival_ns: float) -> tuple[float, list[dict[str, Any]]]:
        current = arrival_ns
        phases: list[dict[str, Any]] = []
        for hop in self._route(source, target, size):
            # TopologyRouter exposes a local memory-service view as a route
            # hop with ``service_ref``.  The source/target PhysicalService
            # bills already charge that queue; reserving the view again would
            # double-count the same medium.  Only independent fabric links
            # consume link timeline capacity here.
            if hop.service_ref:
                continue
            phase = self._submit_link_phase(hop, size, current)
            current = phase["completion_ns"]
            phases.append(phase)
        return current, phases

    def _transfer(self, request: KvPoolTransferRequest) -> KvPoolTransferPlan:
        page_id = self._page_id(request)
        binding = self._bindings.get(page_id)
        if binding is None:
            raise KvPoolError("KV page {} is not physically bound".format(page_id))
        if (type(request.bytes) is not int or binding.component_id != request.source_component
                or binding.bytes != request.bytes):
            raise KvPoolError("transfer request does not match the bound source page")
        if (type(request.allocation_generation) is not int
                or request.allocation_generation != binding.generation):
            raise KvPoolError("transfer request generation does not match bound page")
        target_component = str(request.target_component)
        if target_component not in self.components:
            raise KvPoolError("unknown transfer target " + target_component)
        target_service = self._ensure_service(target_component)
        target_owner = target_service.physical_owner
        snapshot = self.runtime.snapshot()
        cache_snapshot = self._snapshot_caches()
        bindings_snapshot = copy.deepcopy(self._bindings)
        events_snapshot = len(self.events)
        counters_snapshot = copy.deepcopy(self.counters)
        cost_count = len(self._cost_phases)
        temp_allocations: list[tuple[str, str, int]] = []
        target_allocator = self._allocator(target_owner)
        target_generation = binding.generation + 1
        start = self._arrival(request, None)
        phases: list[dict[str, Any]] = []
        try:
            target_alloc = target_allocator.allocate(self._buffer_id(request),
                self.allocation_bytes(target_component, binding.bytes), target_generation,
                allocation_arena="resident")
            source_addr = binding.allocation.base_address
            source_start, writebacks = self._flush_source_caches(binding, start)
            phases.extend(writebacks)
            read_bills = self._price_chunks(binding.component_id, AccessKind.READ, source_addr,
                                            binding.bytes, arrival_ns=source_start, caller_managed_transaction=True)
            current = float(read_bills[-1].get("completion_ns", start)) if read_bills else start
            for bill in read_bills:
                phases.append({"kind": "source_read", **bill})
            route_stages = (self.relay_components + (target_component,)) if self.relay_components else (target_component,)
            current_component = binding.component_id
            for stage in route_stages[:-1]:
                current, link_phases = self._link_segment(current_component, stage, binding.bytes, arrival_ns=current)
                phases.extend(link_phases)
                relay_owner = self._ensure_service(stage).physical_owner
                relay_allocator = self._allocator(relay_owner)
                relay_allocation = relay_allocator.allocate(
                    self._buffer_id(request) + ":relay:" + stage,
                    self.allocation_bytes(stage, binding.bytes), target_generation,
                    allocation_arena="workspace")
                temp_allocations.append((relay_owner, relay_allocation.buffer_id, target_generation))
                relay_bills = self._price_chunks(stage, AccessKind.WRITE, relay_allocation.base_address,
                    binding.bytes, arrival_ns=current, caller_managed_transaction=True)
                for bill in relay_bills:
                    phases.append({"kind": "relay_write", "component": stage, **bill})
                current = float(relay_bills[-1].get("completion_ns", current)) if relay_bills else current
                relay_read = self._price_chunks(stage, AccessKind.READ,
                    relay_allocation.base_address,
                    binding.bytes, arrival_ns=current, caller_managed_transaction=True)
                for bill in relay_read:
                    phases.append({"kind": "relay_read", "component": stage, **bill})
                current = float(relay_read[-1].get("completion_ns", current)) if relay_read else current
                self.counters["relay_bytes"] += 2 * binding.bytes
                current_component = stage
            current, link_phases = self._link_segment(current_component, target_component, binding.bytes, arrival_ns=current)
            phases.extend(link_phases)
            target_bills = self._price_chunks(target_component, AccessKind.WRITE, target_alloc.base_address,
                                              binding.bytes, arrival_ns=current, caller_managed_transaction=True)
            for bill in target_bills:
                phases.append({"kind": "target_write", **bill})
            completion = float(target_bills[-1].get("completion_ns", current)) if target_bills else current
        except Exception:
            self.runtime.restore(snapshot)
            self._restore_caches(cache_snapshot)
            self._bindings = bindings_snapshot
            self.counters = counters_snapshot
            del self._cost_phases[cost_count:]
            raise
        plan_id = self._next_async_plan_id
        self._next_async_plan_id += 1
        phase_token = self._new_phase_token()
        # All physical work is prepared at once, so this single synchronous
        # phase is also final. Register it just like asynchronous phases: a
        # serving snapshot can restore it after the closure already committed.
        state = {"done": False, "phase": 0, "phase_count": 0, "plan_id": plan_id,
                 "phase_token": phase_token}
        self._async_plan_states[plan_id] = state
        plan = KvPoolTransferPlan(
            duration_ns=max(0.0, completion - start), start_ns=start,
            completion_ns=completion,
            commit=lambda: self._commit_transfer(page_id, target_component, target_owner, target_alloc,
                                                 target_generation, phases, temp_allocations, state,
                                                 expected_token=phase_token),
            rollback=lambda: self._rollback_transfer(snapshot, bindings_snapshot, counters_snapshot, events_snapshot, target_component,
                                                     target_owner, target_alloc, temp_allocations, state,
                                                     expected_token=phase_token, cost_count=cost_count, cache_snapshot=cache_snapshot),
        )
        return plan

    def prepare_async(self, request: KvPoolTransferRequest) -> KvPoolTransferPlan:
        """Reserve capacity and source read, then submit downstream work in time order.

        A future HBM write must not be submitted while its HBF read is still
        pending: the physical cores use ordered admission calendars. The
        caller advances this plan only when its current phase has completed,
        interleaving those submissions with ordinary live-kernel tasks.
        """
        page_id = self._page_id(request)
        binding = self._bindings.get(page_id)
        if (binding is None or type(request.bytes) is not int
                or type(request.allocation_generation) is not int
                or binding.component_id != request.source_component
                or binding.bytes != request.bytes
                or binding.generation != request.allocation_generation):
            raise KvPoolError("asynchronous transfer does not match the bound source page")
        target_component = str(request.target_component)
        target_service = self._ensure_service(target_component)
        target_owner = target_service.physical_owner
        snapshot = self.runtime.snapshot()
        cache_snapshot = self._snapshot_caches()
        bindings_snapshot = copy.deepcopy(self._bindings)
        events_snapshot = len(self.events)
        counters_snapshot = copy.deepcopy(self.counters)
        cost_count = len(self._cost_phases)
        target_generation = binding.generation + 1
        temp_allocations = []
        start = self._arrival(request, None)
        try:
            target_alloc = self._allocator(target_owner).allocate(
                self._buffer_id(request), self.allocation_bytes(target_component, binding.bytes), target_generation,
                allocation_arena="resident")
            # Allocate relay buffers before beginning I/O, so a later phase
            # cannot discover an unreserved capacity requirement.
            downstream = []
            source = binding.component_id
            for component in (*self.relay_components, target_component):
                for hop in self._route(source, component, binding.bytes):
                    if not hop.service_ref:
                        downstream.append(("link", hop))
                if component == target_component:
                    downstream.append(("target_write", component, target_alloc.base_address, AccessKind.WRITE))
                else:
                    relay_owner = self._ensure_service(component).physical_owner
                    allocator = self._allocator(relay_owner)
                    allocation = allocator.allocate(self._buffer_id(request) + ":relay:" + component,
                                                    self.allocation_bytes(component, binding.bytes), target_generation,
                                                    allocation_arena="workspace")
                    temp_allocations.append((relay_owner, allocation.buffer_id, target_generation))
                    downstream.extend((
                        ("relay_write", component, allocation.base_address, AccessKind.WRITE),
                        ("relay_read", component, allocation.base_address, AccessKind.READ)))
                source = component
            source_start, writebacks = self._flush_source_caches(binding, start)
            read_bills = self._price_chunks(binding.component_id, AccessKind.READ,
                binding.allocation.base_address, binding.bytes, arrival_ns=source_start,
                caller_managed_transaction=True)
            phases = (*writebacks, *({"kind": "source_read", **bill} for bill in read_bills))
            ready = float(read_bills[-1].get("completion_ns", start)) if read_bills else start
        except Exception:
            self.runtime.restore(snapshot)
            self._restore_caches(cache_snapshot)
            self._bindings = bindings_snapshot
            self.counters = counters_snapshot
            del self._cost_phases[cost_count:]
            raise
        plan_id = self._next_async_plan_id
        self._next_async_plan_id += 1
        phase_token = self._new_phase_token()
        state = {"done": False, "phase": 0, "phase_count": len(downstream), "plan_id": plan_id,
                 "phase_token": phase_token}
        self._async_plan_states[plan_id] = state

        def make_plan(index, phase_end, completed_phases, token):
            final = index == len(downstream)

            def advance():
                live_state = self._transfer_state(state)
                if live_state["done"]:
                    raise KvPoolError("physical transfer was already committed or cancelled")
                if live_state["phase"] != index:
                    raise KvPoolError("physical transfer phase was already advanced")
                self._check_phase_token(live_state, token)
                # Other requests may have run since prepare_async. Roll back
                # only this newly submitted phase; the plan-wide rollback is
                # reserved for immediate, just-prepared admission failures.
                with self._io_transaction():
                    next_token = self._new_phase_token()
                    next_plan = advance_phase(next_token)
                    live_state["phase"] = index + 1
                    live_state["phase_token"] = next_token
                    return next_plan

            def advance_phase(next_token):
                descriptor = downstream[index]
                kind = descriptor[0]
                if kind == "link":
                    hop = descriptor[1]
                    phase = self._submit_link_phase(hop, binding.bytes, phase_end)
                    new_phases = (phase,)
                    end = phase["completion_ns"]
                else:
                    _, component, address, access_kind = descriptor
                    bills = self._price_chunks(component, access_kind, address, binding.bytes,
                                               arrival_ns=phase_end, caller_managed_transaction=True)
                    new_phases = tuple({"kind": kind, "component": component, **bill} for bill in bills)
                    end = float(bills[-1].get("completion_ns", phase_end)) if bills else phase_end
                    if kind.startswith("relay_"):
                        self.counters["relay_bytes"] += binding.bytes
                return make_plan(index + 1, end, (*completed_phases, *new_phases), next_token)

            return KvPoolTransferPlan(
                duration_ns=max(0.0, phase_end - start), start_ns=start, completion_ns=phase_end,
                commit=(lambda: self._commit_transfer(page_id, target_component, target_owner, target_alloc,
                    target_generation, list(completed_phases), temp_allocations, state,
                    expected_token=token)) if final else None,
                rollback=lambda: self._rollback_transfer(snapshot, bindings_snapshot, counters_snapshot,
                    events_snapshot, target_component, target_owner, target_alloc, temp_allocations, state,
                    expected_token=token, cost_count=cost_count, cache_snapshot=cache_snapshot),
                advance=None if final else advance,
                cancel=lambda: self._cancel_transfer(page_id, target_component, target_owner,
                    target_alloc, target_generation, completed_phases, temp_allocations, state,
                    expected_phase=index, expected_token=token),
            )

        return make_plan(0, ready, phases, phase_token)

    def _new_phase_token(self):
        # Replaying a snapshot creates a new physical submission. Its callback
        # must not accept a discarded future's same-index phase and old bill.
        token = self._next_async_phase_token
        self._next_async_phase_token += 1
        return token

    @staticmethod
    def _check_phase_token(state, expected_token):
        if expected_token is not None and state.get("phase_token") != expected_token:
            raise KvPoolError("physical transfer phase belongs to a discarded future")

    def _finish_transfer(self, live_state, callback_state):
        live_state["done"] = True
        if "plan_id" in live_state:
            # Keep terminal idempotence in callbacks already owned by callers,
            # without retaining every completed migration in the executor or
            # copying that history into subsequent snapshots.
            callback_state.update(live_state)
            callback_state["terminal_epoch"] = self._async_restore_epoch
            self._async_plan_states.pop(live_state["plan_id"], None)

    def _transfer_state(self, state):
        if "plan_id" not in state:
            return state
        try:
            return self._async_plan_states[state["plan_id"]]
        except KeyError as exc:
            if state["done"] and state.get("terminal_epoch") == self._async_restore_epoch:
                return state
            raise KvPoolError("physical transfer plan is no longer active") from exc

    def _cancel_transfer(self, page_id, target_component, target_owner, target_alloc,
                         target_generation, phases, temp_allocations, state, *, expected_phase=None,
                         expected_token=None):
        with self._finalization_transaction(state):
            return self._cancel_transfer_impl(page_id, target_component, target_owner,
                target_alloc, target_generation, phases, temp_allocations, state,
                expected_phase=expected_phase, expected_token=expected_token)

    @contextmanager
    def _finalization_transaction(self, callback_state):
        """Publish/free atomically while retaining the already completed I/O."""
        bindings = dict(self._bindings)
        states = copy.deepcopy(self._async_plan_states)
        callback = dict(callback_state)
        event_count = len(self.events)
        try:
            with self._io_transaction():
                yield
        except Exception:
            self._bindings = bindings
            self._async_plan_states = states
            callback_state.clear()
            callback_state.update(callback)
            del self.events[event_count:]
            raise

    def _cancel_transfer_impl(self, page_id, target_component, target_owner, target_alloc,
                             target_generation, phases, temp_allocations, state, *, expected_phase=None,
                             expected_token=None):
        callback_state = state
        state = self._transfer_state(state)
        if state["done"]:
            raise KvPoolError("physical transfer was already committed or cancelled")
        if expected_phase is not None and state["phase"] != expected_phase:
            raise KvPoolError("cannot cancel using an earlier physical transfer phase")
        self._check_phase_token(state, expected_token)
        # Caller waits for the submitted phase's completion; no allocator can
        # reuse an address still being written by that phase.
        self._release_allocation(target_owner, target_alloc.buffer_id, target_generation)
        for owner, buffer_id, generation in temp_allocations:
            self._release_allocation(owner, buffer_id, generation)
        billed = {"logical_page_id": page_id, "kind": "cancelled",
                  "source_component": self._bindings[page_id].component_id,
                  "target_component": target_component, "bytes": self._bindings[page_id].bytes,
                  "phases": tuple(phases),
                  "start_ns": phases[0].get("arrival_ns", 0.0) if phases else 0.0,
                  "completion_ns": phases[-1].get("completion_ns", 0.0) if phases else 0.0}
        billed["duration_ns"] = billed["completion_ns"] - billed["start_ns"]
        self.events.append(billed)
        self.counters["cancelled_transfer_count"] = self.counters.get("cancelled_transfer_count", 0) + 1
        self._finish_transfer(state, callback_state)
        return billed

    def _commit_transfer(self, page_id: int, target_component: str, target_owner: str,
                         target_alloc: PhysicalAllocation, target_generation: int,
                         phases: list[dict[str, Any]], temp_allocations: list[tuple[str, str, int]],
                          state: dict[str, Any], *, expected_token=None) -> None:
        with self._finalization_transaction(state):
            self._commit_transfer_impl(page_id, target_component, target_owner,
                target_alloc, target_generation, phases, temp_allocations, state,
                expected_token=expected_token)

    def _commit_transfer_impl(self, page_id: int, target_component: str, target_owner: str,
                             target_alloc: PhysicalAllocation, target_generation: int,
                             phases: list[dict[str, Any]], temp_allocations: list[tuple[str, str, int]],
                             state: dict[str, Any], *, expected_token=None) -> None:
        callback_state = state
        state = self._transfer_state(state)
        if "phase_count" in state and state["phase"] != state["phase_count"]:
            raise KvPoolError("cannot commit before every physical transfer phase completes")
        self._check_phase_token(state, expected_token)
        if state["done"]:
            return
        binding = self._bindings[page_id]
        old = binding
        self._release_allocation(old.owner, old.allocation.buffer_id, old.generation)
        self._bindings[page_id] = _PageBinding(old.page_id, old.request_id, old.layer_group,
                                               target_component, target_owner, old.bytes,
                                               target_generation, target_alloc)
        for owner, buffer_id, generation in temp_allocations:
            self._release_allocation(owner, buffer_id, generation)
        self.events.append({"logical_page_id": page_id, "source_component": old.component_id,
                            "target_component": target_component, "bytes": old.bytes,
                            "start_ns": phases[0].get("arrival_ns", 0.0) if phases else 0.0,
                            "completion_ns": phases[-1].get("completion_ns", 0.0) if phases else 0.0,
                            "duration_ns": 0.0 if not phases else phases[-1].get("completion_ns", 0.0) - phases[0].get("arrival_ns", 0.0),
                            "phases": tuple(phases)})
        self.counters["transfer_count"] += 1
        self._finish_transfer(state, callback_state)

    def _rollback_transfer(self, snapshot: Mapping[str, Any], bindings: Mapping[int, _PageBinding],
                           counters: Mapping[str, Any], events_len: int, target_component: str, target_owner: str,
                           target_alloc: PhysicalAllocation, temp_allocations: list[tuple[str, str, int]],
                           state: dict[str, Any], *, expected_token=None, cost_count=0, cache_snapshot=None) -> None:
        callback_state = state
        state = self._transfer_state(state)
        if state["done"]:
            return
        if state.get("phase", 0) != 0:
            raise KvPoolError("only a just-prepared physical transfer can be rolled back")
        self._check_phase_token(state, expected_token)
        self.runtime.restore(snapshot)
        if cache_snapshot is not None:
            self._restore_caches(cache_snapshot)
        self._bindings = copy.deepcopy(bindings)
        self.counters = copy.deepcopy(counters)
        del self.events[events_len:]
        del self._cost_phases[cost_count:]
        self._finish_transfer(state, callback_state)

    def __call__(self, request: KvPoolTransferRequest) -> KvPoolTransferPlan:
        return self._transfer(request)

    def snapshot(self, *, _immutable_timeline: bool = False) -> dict[str, Any]:
        runtime_method = self.runtime.snapshot
        runtime_snapshot = (runtime_method(_immutable_timeline=True)
            if _immutable_timeline and type(runtime_method) is _MethodType
            and runtime_method.__func__ is _NATIVE_TIMELINE_HISTORY_SNAPSHOT else runtime_method())
        return {"runtime": runtime_snapshot, "bindings": copy.deepcopy(self._bindings),
                "cache_states": self._snapshot_caches(),
                "events": copy.deepcopy(self.events), "counters": copy.deepcopy(self.counters),
                "cost_phases": tuple(self._cost_phases),
                "cost_groups": copy.deepcopy(self._cost_groups), "cost_phase_total": self._cost_phase_total,
                "async_plan_states": _copy_plan_states(self._async_plan_states)}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.runtime.restore(snapshot["runtime"])
        self._restore_caches(snapshot["cache_states"])
        self._bindings = copy.deepcopy(snapshot["bindings"])
        self.events = copy.deepcopy(snapshot["events"])
        self.counters = copy.deepcopy(snapshot["counters"])
        self._cost_phases = list(snapshot["cost_phases"])
        self._cost_groups = copy.deepcopy(snapshot["cost_groups"])
        self._cost_phase_total = snapshot["cost_phase_total"]
        self._async_plan_states = _copy_plan_states(snapshot["async_plan_states"])
        # An absent plan cannot inherit terminal authority from a discarded
        # future. Active phase tokens explicitly contained in the snapshot
        # remain valid, including when the same snapshot is restored again.
        self._async_restore_epoch += 1

    def stats(self) -> Mapping[str, Any]:
        return {**self.counters, "events": len(self.events)}


_IMMUTABLE_EXECUTOR_SNAPSHOT = PhysicalKVTransferExecutor.snapshot


_PLAN_COPY_ATOMS = (str, int, float, bool, type(None))
_PLAN_COPY_DEEPCOPY = copy.deepcopy
_PLAN_COPY_DISPATCH = copy._deepcopy_dispatch
_PLAN_COPY_KEEP_ALIVE = copy._keep_alive
_PLAN_COPY_HANDLERS = tuple((kind, copy._deepcopy_dispatch[kind])
                           for kind in (*_PLAN_COPY_ATOMS, dict, tuple))


def _copy_plan_destinations(value, memo):
    """Copy ordinary destination rows without dispatching each scalar.

    The memo belongs to the complete plan-state snapshot, not this field.
    Unknown containers/values keep deepcopy's callbacks, recursion and order.
    """
    if (type(value) is not tuple
            or (id(memo) in memo and type(memo[id(memo)]) is not list)
            or copy.deepcopy is not _PLAN_COPY_DEEPCOPY
            or copy._deepcopy_dispatch is not _PLAN_COPY_DISPATCH
            or copy._keep_alive is not _PLAN_COPY_KEEP_ALIVE
            or any(_PLAN_COPY_DISPATCH.get(kind) is not handler
                   for kind, handler in _PLAN_COPY_HANDLERS)):
        return copy.deepcopy(value, memo)
    if id(value) in memo:
        return memo[id(value)]
    result = []
    changed = False
    for row in value:
        if id(row) in memo:
            previous = memo[id(row)]
        else:
            # A custom earlier field can seed even scalar entries in memo.
            # Such rows must retain deepcopy's lookup of both values and keys.
            if (type(row) is not dict or any(
                    type(key) is not str or not (type(item) is str or type(item) is int
                        or type(item) is float or type(item) is bool or item is None)
                    or id(key) in memo or id(item) in memo
                    for key, item in row.items())):
                # Already copied rows are in memo exactly where deepcopy
                # would have put them. Falling back neither repeats callbacks
                # nor changes aliases, including tuple/list cycles.
                return copy.deepcopy(value, memo)
            previous = row.copy()
            memo[id(row)] = previous
            _PLAN_COPY_KEEP_ALIVE(row, memo)
        result.append(previous)
        changed |= previous is not row
    if not changed:
        return value
    result = tuple(result)
    memo[id(value)] = result
    _PLAN_COPY_KEEP_ALIVE(value, memo)
    return result


def _copy_plan_states(states):
    """Copy in-flight plan states, sharing their rollback snapshots.

    A rollback snapshot is a value that restore() only reads. Deep-copying it
    copied the whole runtime once per just-prepared import, nested through
    every later snapshot, so each import cost grew with the batch.
    """
    memo = {}
    return {plan_id: {key: value if key == "rollback_snapshot" else
                      _copy_plan_destinations(value, memo)
                      if type(key) is str and key == "destinations" else copy.deepcopy(value, memo)
                      for key, value in state.items()}
            for plan_id, state in states.items()}


_NATIVE_COMPACT_COST_WRITERS = tuple((name, getattr(PhysicalKVTransferExecutor, name))
    for name in ("_price_chunks", "_record_memory_cost", "_retain_cost_phase", "_sum_counters"))
_NATIVE_EXTERNAL_MEMORY_PRICE = PhysicalService._price


# Link callbacks have no memory-owner journal. Their native demand builder
# closes the footprint over the byte wire and its in-flight credit resource.
# A replaced class, demand constructor, timeline writer or scalar validation
# hook keeps the complete publication contract instead.
from . import communication as _sparse_communication
from . import data_motion as _sparse_data_motion
from .contracts import ResourceDemand as _SparseResourceDemand
from .memory_transfer import ResourceTimeline as _SparseTimeline, _OWNED_TIMELINE_METHODS as _SPARSE_TIMELINE_METHODS
_SPARSE_LINK_CLASSES = tuple((cls, tuple((name, value) for name, value in vars(cls).items()
    if name != "__slotnames__")) for cls in (
        _sparse_communication.RouteHop, _sparse_data_motion.LinkService, _SparseResourceDemand))
_SPARSE_LINK_CALLBACKS = (
    (_sparse_communication, "RouteHop", _sparse_communication.RouteHop),
    (_sparse_communication, "LinkService", _sparse_data_motion.LinkService),
    (_sparse_communication, "_non_negative_integer", _sparse_communication._non_negative_integer),
    *tuple((_sparse_data_motion, name, getattr(_sparse_data_motion, name)) for name in (
        "ResourceDemand", "_positive", "_non_negative", "_positive_int", "_non_negative_int")),
)
_SPARSE_LINK_SUBMIT = PhysicalKVTransferExecutor._submit_link_phase


def _native_external_link_resources(executor, hop, bill):
    from ._nand_pack import can_own_resource_aliases, can_own_timeline
    from .memory_transfer import _TIMELINE_MAPS
    timeline = executor.runtime.timeline
    if (type(executor) is not PhysicalKVTransferExecutor
            or type(hop) is not _SPARSE_LINK_CLASSES[0][0]
            or type(timeline) is not _SparseTimeline
            or getattr(executor._submit_link_phase, "__func__", None) is not _SPARSE_LINK_SUBMIT
            or not executor._native_compact_cost_writer()
            or any(getattr(module, name) is not original for module, name, original in _SPARSE_LINK_CALLBACKS)
            or any(getattr(getattr(timeline, name), "__func__", getattr(timeline, name)) is not original
                   for name, original in _SPARSE_TIMELINE_METHODS)
            or any(type(getattr(timeline, name)) is not dict for name in _TIMELINE_MAPS)
            or not can_own_resource_aliases(timeline.resource_aliases)
            or not can_own_timeline(timeline, tuple(bill["resource_busy_ns"]))
            or any(type(value) not in (str, int, float, bool, type(None)) for value in vars(hop).values())):
        return None
    for cls, original in _SPARSE_LINK_CLASSES:
        current = vars(cls)
        cached_slots = current.get("__slotnames__")
        if (cached_slots is not None and (type(cached_slots) is not list or cached_slots)
                or len(current) - ("__slotnames__" in current) != len(original)
                or any(current.get(name) is not value for name, value in original)):
            return None
    # These are the actual emitted demands, including credit lanes, after
    # canonical alias resolution. No guessed per-device resource list.
    return tuple(bill["resource_busy_ns"])


_NATIVE_EXTERNAL_PUBLICATION_METHODS = tuple((name, getattr(PhysicalKVTransferExecutor, name)) for name in (
    "prepare_external_import", "_start_external_chunk", "_submit_external_action", "_external_plan",
    "_advance_external_phase", "_external_chunk_actions",
    "_native_compact_cost_writer", "_io_transaction", "_price_chunks", "_record_memory_cost", "_retain_cost_phase",
    "_sum_counters", "_submit_link_phase", "_ensure_service", "_allocator", "_route"))
_NATIVE_EXTERNAL_RUNTIME_METHODS = tuple((name, getattr(PhysicalRuntimeContext, name)) for name in (
    "snapshot", "begin_transaction", "commit_transaction", "restore", "_owners_have_native_hooks", "_can_own_timeline"))
_NATIVE_EXTERNAL_LINK_RESOURCE_EXPORT = _native_external_link_resources


def native_external_publication_supported(executor):
    """An authored transfer plan is never trusted merely for carrying IDs."""
    return (type(executor) is PhysicalKVTransferExecutor
        and type(executor.runtime) is PhysicalRuntimeContext
        and _native_external_link_resources is _NATIVE_EXTERNAL_LINK_RESOURCE_EXPORT
        and all(getattr(getattr(executor, name), "__func__", getattr(executor, name)) is original
                for name, original in _NATIVE_EXTERNAL_PUBLICATION_METHODS)
        and all(getattr(getattr(executor.runtime, name), "__func__", getattr(executor.runtime, name)) is original
                for name, original in _NATIVE_EXTERNAL_RUNTIME_METHODS))


__all__ = ["PhysicalKVTransferExecutor"]
