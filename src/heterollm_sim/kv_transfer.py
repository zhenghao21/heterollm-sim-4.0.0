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
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from .data_motion import AccessKind, PhysicalRuntimeContext, PhysicalService, resolve_service
from .ir import ComponentSpec
from .memory_allocator import PhysicalAddressAllocator, PhysicalAllocation
from .memory_types import DramConfig, NandConfig
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
        self._configs: dict[str, DramConfig | NandConfig] = {}
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
    def _io_transaction(self):
        """Roll back one logical access/phase without copying event history."""
        saved_runtime = self.runtime.snapshot()
        saved_counters = dict(self.counters)
        cost_count = len(self._cost_phases)
        cache_transactions = []
        try:
            for _key, (_signature, cache) in self.cache_states.items():
                cache_transactions.append((cache, cache.begin_transaction()))
            yield
        except Exception:
            self.runtime.restore(saved_runtime)
            self.counters = saved_counters
            del self._cost_phases[cost_count:]
            for cache, transaction in reversed(cache_transactions):
                cache.rollback_transaction(transaction)
            raise
        else:
            for cache, transaction in cache_transactions:
                cache.commit_transaction(transaction)

    def _record_memory_cost(self, component_id, kind, bill):
        owner = self.services[component_id].physical_owner
        raw = dict(bill)
        raw["logical_read_bytes"] = int(raw.get("logical_bytes", 0)) if kind is AccessKind.READ else 0
        raw["logical_write_bytes"] = int(raw.get("logical_bytes", 0)) if kind is AccessKind.WRITE else 0
        direction = "read" if kind is AccessKind.READ else "write"
        raw.update({"phase_id": "paged-kv-physical-" + str(len(self._cost_phases)),
            "kind": "memory", "component_id": component_id, "owner": owner,
            "memory_kind": self._configs[component_id].kind.value,
            "resource_owners": {resource: owner for resource in raw.get("resource_busy_ns", {})},
            "resource_busy_by_direction_ns": {direction: dict(raw.get("resource_busy_ns", {}))},
            "direction_semantics": "attributed to the submitted transaction direction; a NAND write may include RMW reads"})
        self._cost_phases.append(raw)
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
            "phase_id": "paged-kv-physical-" + str(len(self._cost_phases)),
            "kind": "link", "link_id": hop.link_id, "operation": "copy",
            "resource_id": timeline.canonical_resource_id(hop.resource_id), "bytes": size,
            "arrival_ns": arrival_ns, "start_ns": start, "completion_ns": end,
            "duration_ns": end - start,
            "logical_bytes": size, "physical_bytes": size, "resource_busy_ns": busy, "resource_bytes": moved,
            "resource_intervals": intervals,
            "resource_interval_payloads": payloads,
            "resource_busy_by_direction_ns": {"read_write": busy},
            "intervals_truncated": False}
        self._cost_phases.append(phase)
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
            "phase_count": len(self._cost_phases), "resource_accounted_bytes": 0, "resource_busy_ns": {},
            "resource_bytes": {}, "resource_busy_by_direction_ns": {}, "owners": {}, "links": {},
            "first_arrival_ns": None, "last_completion_ns": None}
        adapters = []
        for phase in self._cost_phases:
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
                row["transaction_count"] += 1
                for key in ("logical_bytes", "logical_read_bytes", "logical_write_bytes",
                            "physical_bytes", "physical_read_bytes", "physical_write_bytes",
                            "host_transfer_bytes", "internal_transfer_bytes"):
                    # Unknown counters must not become measured zero.
                    value = phase.get(key)
                    row[key] = None if value is None or (key in row and row[key] is None) else row.get(key, 0) + value
                raw = {**phase, "kind": phase["memory_kind"]}
                adapters.append(SimpleNamespace(task_id=phase["phase_id"],
                    metadata={"physical_execution_by_owner": {owner: raw}},
                    demands=tuple(SimpleNamespace(resource_id=resource)
                                  for resource in phase.get("resource_busy_ns", {}))))
            else:
                row = result["links"].setdefault(phase["link_id"], {"phase_count": 0, "bytes": 0})
                row["phase_count"] += 1
                row["bytes"] += phase["physical_bytes"]
        result["resource_accounted_bytes"] = sum(result["resource_bytes"].values())
        result["storage_traffic"] = _summarize_nand_task_traffic(adapters)
        result["dram_traffic"] = _summarize_dram_task_traffic(adapters)
        for key in ("storage_traffic", "dram_traffic"):
            result[key] = {**result[key], "count_semantics": "task_count here counts submitted migration memory phases, not kernel tasks"}
        result["phases"] = tuple(copy.deepcopy(self._cost_phases))
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

    def snapshot(self) -> dict[str, Any]:
        return {"runtime": self.runtime.snapshot(), "bindings": copy.deepcopy(self._bindings),
                "cache_states": self._snapshot_caches(),
                "events": copy.deepcopy(self.events), "counters": copy.deepcopy(self.counters),
                "cost_phases": tuple(self._cost_phases),
                "async_plan_states": copy.deepcopy(self._async_plan_states)}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.runtime.restore(snapshot["runtime"])
        self._restore_caches(snapshot["cache_states"])
        self._bindings = copy.deepcopy(snapshot["bindings"])
        self.events = copy.deepcopy(snapshot["events"])
        self.counters = copy.deepcopy(snapshot["counters"])
        self._cost_phases = list(snapshot["cost_phases"])
        self._async_plan_states = copy.deepcopy(snapshot["async_plan_states"])
        # An absent plan cannot inherit terminal authority from a discarded
        # future. Active phase tokens explicitly contained in the snapshot
        # remain valid, including when the same snapshot is restored again.
        self._async_restore_epoch += 1

    def stats(self) -> Mapping[str, Any]:
        return {**self.counters, "events": len(self.events)}


__all__ = ["PhysicalKVTransferExecutor"]
