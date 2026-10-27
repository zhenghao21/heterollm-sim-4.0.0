"""Position-indexed execution of a live closed Token graph.

The program stores scheduling state by integer position. Device transactions,
resource accounting and allocation retirement use the same primitives as the
incremental kernel. This module is also compiled by the strict Cython build.
"""
from __future__ import annotations

import heapq
from dataclasses import replace
from typing import Dict, List

try:
    import cython
except ImportError:
    class _PythonMode:
        Py_ssize_t = int
        @staticmethod
        def cclass(cls):
            return cls
        @staticmethod
        def locals(**types):
            return lambda function: function
    cython = _PythonMode()


@cython.cclass
class TokenFrame:
    kernel: object
    chunk: tuple
    layout: object
    phases: tuple
    ids: tuple
    indegree: list
    dependency_ready: list
    sorted_demands: tuple
    groups: tuple
    ready_by_group: list
    versions: list
    heap: list
    remaining: cython.Py_ssize_t
    completed: list
    owned_physical_frontiers: object
    frontiers_initialized: object
    shared_physical_transaction: object

    def __init__(self, kernel, chunk, layout, phases, index_physical_allocations,
                 prepared_seen=None, owned_physical_frontiers=False):
        self.kernel = kernel
        self.chunk = chunk
        self.layout = layout
        self.phases = phases
        self.ids = tuple(task.task_id for task in chunk)
        self.indegree = list(layout.indegree_template)
        self.dependency_ready = [0.0] * len(chunk)
        self.sorted_demands = tuple(tuple(task.demands[i] for i in order)
                                    for task, order in zip(chunk, layout.demand_order))
        self.groups = tuple(sorted(set(layout.resource_groups)))
        self.ready_by_group = [[] for _ in self.groups]
        self.versions = [kernel._group_versions.get(group, 0) for group in self.groups]
        self.heap = []
        self.remaining = len(chunk)
        self.completed = [False] * len(chunk)
        self.owned_physical_frontiers = owned_physical_frontiers
        self.frontiers_initialized = False
        # A closed native cohort is already protected by the serving-level
        # rollback snapshot.  The drain may install one physical transaction
        # for the whole graph and expose a lightweight per-task view below;
        # this removes one allocator/timeline snapshot and commit per task.
        self.shared_physical_transaction = None

        # Submission remains atomic: validation has finished before any live
        # state changes. An open dynamic submission owns a different lifetime.
        kernel._physical_allocation_uses.clear()
        kernel._external_physical_allocation_keys.clear()
        kernel._physical_allocation_arenas.clear()
        kernel._physical_allocation_task_keys.clear()
        kernel._physical_allocation_last_end.clear()
        kernel._pending_physical_releases.clear()
        kernel._tasks.update(zip(self.ids, chunk))
        if prepared_seen is None:
            kernel._seen_ids.update(self.ids)
        else:
            kernel._commit_compact_seen_namespace(prepared_seen)
        initial = set()
        for position in layout.root_positions:
            changed = self.add_ready(position)
            if changed is not None:
                initial.add(changed)
        for group in sorted(initial):
            self.refresh(group)
        kernel._token_frame = self
        if index_physical_allocations:
            kernel._index_physical_allocation_uses(chunk)

    def static_key(self, position):
        task = self.chunk[position]
        dependency = self.dependency_ready[position]
        effective = dependency
        if task.earliest_start_ns > effective:
            effective = task.earliest_start_ns
        phase, sequence = self.phases[position]
        return (effective, dependency, task.earliest_start_ns, phase, sequence,
                task.task_id, position)

    def invalidate_frontiers(self):
        """An external observer may edit calendars outside the last task."""
        self.frontiers_initialized = False

    def ready_key(self, position):
        from .data_motion import is_physical_task, physical_task_demands
        task = self.chunk[position]
        dependency = self.dependency_ready[position]
        effective = dependency
        if task.earliest_start_ns > effective:
            effective = task.earliest_start_ns
        demands = physical_task_demands(task) if is_physical_task(task) else task.demands
        if demands:
            iterator = iter(demands)
            ready = self.kernel._resource_ready_ns(next(iterator).resource_id)
            for demand in iterator:
                value = self.kernel._resource_ready_ns(demand.resource_id)
                if value > ready:
                    ready = value
        else:
            ready = 0.0
        actual = effective
        if ready > actual:
            actual = ready
        phase, sequence = self.phases[position]
        return (actual, effective, dependency, task.earliest_start_ns,
                phase, sequence, task.task_id)

    def refresh(self, group):
        self.versions[group] += 1
        ready = self.ready_by_group[group]
        if ready:
            position = ready[0][-1]
            heapq.heappush(self.heap, (*self.ready_key(position), self.groups[group],
                                       self.versions[group], position))

    def add_ready(self, position):
        group = self.layout.group_indices[position]
        ready = self.ready_by_group[group]
        previous = ready[0] if ready else None
        heapq.heappush(ready, self.static_key(position))
        return group if previous is None or ready[0] != previous else None

    def pop_ready(self):
        while self.heap:
            queued = heapq.heappop(self.heap)
            position = queued[9]
            group = self.layout.group_indices[position]
            if queued[8] != self.versions[group]:
                continue
            ready = self.ready_by_group[group]
            if not ready or ready[0][-1] != position:
                continue
            current = self.ready_key(position)
            if current != queued[:7]:
                self.refresh(group)
                continue
            return queued
        return None

    def peek_ready_key(self):
        queued = self.pop_ready()
        if queued is None:
            return None
        heapq.heappush(self.heap, queued)
        return queued[0], queued[1], queued[2], queued[3], queued[6]

    def step(self):
        queued = self.pop_ready()
        if queued is None:
            return None
        position = queued[9]
        group = self.layout.group_indices[position]
        start, effective, dependency = queued[:3]
        kernel = self.kernel
        task = self.chunk[position]
        try:
            task, demands, physical, registered, snapshot = kernel._resolve_task_execution(task, start, self.sorted_demands[position])
        except Exception:
            heapq.heappush(self.heap, queued)
            raise
        heapq.heappop(self.ready_by_group[group])
        kernel._tasks.pop(task.task_id)
        event = kernel._account_task_execution(task, demands, physical, snapshot,
                                               start, effective, dependency)
        end = event.end_ns
        self.completed[position] = True
        self.remaining -= 1
        kernel._dependents[task.task_id] = [self.ids[p] for p in self.layout.dependent_positions[position]]
        changed_groups = [group]
        for dependent in self.layout.dependent_positions[position]:
            if end > self.dependency_ready[dependent]:
                self.dependency_ready[dependent] = end
            self.indegree[dependent] -= 1
            if self.indegree[dependent] == 0:
                changed = self.add_ready(dependent)
                if changed is not None and changed not in changed_groups:
                    changed_groups.append(changed)
        changed_groups.sort()
        for changed in changed_groups:
            self.refresh(changed)
        if registered:
            kernel._retire_transient_physical_allocations(task.task_id, end)
            if not kernel._tasks:
                kernel._reclaim_physical_allocations(float("inf"))
        if not self.remaining:
            self.publish_scheduler()
            kernel._token_frame = None
        return event

    def publish_scheduler(self):
        kernel = self.kernel
        for group, version, ready in zip(self.groups, self.versions, self.ready_by_group):
            kernel._group_versions[group] = version
            # Public incremental work uses the same six-field static key.
            kernel._ready_by_group[group] = [item[:6] for item in ready]

    def export_incremental(self):
        """Transfer a suspended frame when the graph becomes open/dynamic."""
        kernel = self.kernel
        self.publish_scheduler()
        for position, task in enumerate(self.chunk):
            if self.completed[position]:
                continue
            key = task.task_id
            kernel._dependents[key] = [self.ids[p] for p in self.layout.dependent_positions[position]]
            kernel._dependency_ready[key] = self.dependency_ready[position]
            kernel._indegree[key] = self.indegree[position]
            kernel._resource_groups[key] = self.layout.resource_groups[position]
            kernel._sorted_demands[key] = self.sorted_demands[position]
            kernel._phase_sequence[key] = self.phases[position]
        for queued in self.heap:
            heapq.heappush(kernel._ready_heap, queued[:9])
        kernel._token_frame = None


@cython.locals(frame=TokenFrame)
def drain_token_program(kernel, chunk, layout, *, validate_tasks=True,
                        event_observer=None, before_step=None,
                        index_physical_allocations=False, on_submitted=None,
                        seen_namespace=None, owned_physical_frontiers=False):
    from .event_kernel import (_validated_phase_sequence, _validate_task_time,
                               _validate_aggregate_metadata)
    if len(chunk) != layout.task_count:
        raise ValueError("task graph does not match compiled structure")
    if validate_tasks and not layout.matches_structure(chunk):
        raise ValueError("task graph does not match compiled structure")
    ids = tuple(task.task_id for task in chunk)
    if len(set(ids)) != len(ids):
        raise ValueError("task chunk contains duplicate task ids")
    prepared_seen = None
    if seen_namespace is None:
        for key in ids:
            if kernel._has_seen_task_id(key):
                raise ValueError("duplicate task_id: {}".format(key))
    else:
        prefix, suffixes = seen_namespace
        if len(suffixes) != len(ids) or any(
                key != prefix + suffix for key, suffix in zip(ids, suffixes)):
            raise ValueError("compact namespace does not match Token task ids")
        prepared_seen = kernel._prepare_compact_seen_namespace(prefix, suffixes)
    phases = []
    for task in chunk:
        kernel._validate_owner_demands(task)
        if validate_tasks:
            _validate_task_time(task)
            _validate_aggregate_metadata(task)
        phases.append(_validated_phase_sequence(task))
    shared_snapshot = None
    # External/prefetch callbacks and TP mirroring can write the physical
    # calendar between tasks, so retain their original per-task transactions.
    # A closed physical graph without those writers can safely use one
    # transaction because the serving cohort already has an outer rollback
    # boundary.
    if owned_physical_frontiers and before_step is None and getattr(kernel, "_native_tp_mirror", None) is None:
        from .data_motion import is_physical_task, physical_task_transaction_owners
        owners = set()
        has_physical = False
        for task in chunk:
            if is_physical_task(task):
                has_physical = True
                owners.update(physical_task_transaction_owners(task))
        if has_physical and owners:
            shared_snapshot = kernel.physical_runtime.begin_transaction(
                owners=owners, _owned_timeline=True)
    try:
        frame = TokenFrame(kernel, chunk, layout, tuple(phases), index_physical_allocations,
                           prepared_seen, owned_physical_frontiers)
        frame.shared_physical_transaction = shared_snapshot
        kernel._token_frame = frame
        if not frame.remaining:
            frame.publish_scheduler()
            kernel._token_frame = None
        if on_submitted is not None:
            on_submitted()
        events = []
        while frame.remaining:
            if before_step is not None:
                try:
                    before_step(kernel)
                finally:
                    frame.invalidate_frontiers()
            # A callback may open the graph through the incremental public API.
            event = frame.step() if kernel._token_frame is frame else kernel.step()
            if event is None:
                kernel.assert_drained()
                raise ValueError("schedule contains a dependency cycle")
            if kernel._token_frame is not frame and frame.remaining:
                frame.remaining -= 1
            if event_observer is None:
                events.append(event)
            else:
                event_observer(event)
        if shared_snapshot is not None:
            kernel.physical_runtime.commit_transaction(shared_snapshot)
            shared_snapshot = None
        return tuple(events)
    except BaseException:
        if shared_snapshot is not None:
            kernel.physical_runtime.restore(shared_snapshot)
        raise


def resolve_task_execution(self, task, start_ns, sorted_demands=None):
    """Resolve one live transaction; the scheduler owns its ready entry."""

    frame: TokenFrame
    task_id = task.task_id
    self._reclaim_physical_allocations(start_ns)
    from .data_motion import is_physical_task
    is_physical = is_physical_task(task)
    registered_physical_allocations = is_physical
    l2_snapshot = None
    l2_transaction = None
    l2_owner = None
    physical_snapshot = None
    shared_physical = False
    if is_physical:
        try:
            from .data_motion import physical_task_transaction_owners
            frame = getattr(self, "_token_frame", None)
            owned_physical = frame is not None and frame.owned_physical_frontiers
            shared = (getattr(frame, "shared_physical_transaction", None)
                      if owned_physical else None)
            if shared is not None:
                from .data_motion import physical_task_demands
                resource_ids = {demand.resource_id for demand in physical_task_demands(task)}
                resource_ids.update(task.metadata.get("physical_resource_last_intervals", {}))
                physical_snapshot = {
                    "_owned_native_writers": shared.get("_owned_native_writers", False),
                    "timeline": {
                        "_owned_transaction": shared["timeline"].get("_owned_transaction"),
                        "lane_available": {
                            resource_id: tuple(self.physical_runtime.timeline.lane_available.get(resource_id, ()))
                            for resource_id in resource_ids
                        },
                    },
                }
                shared_physical = True
            else:
                physical_snapshot = self.physical_runtime.begin_transaction(
                    owners=physical_task_transaction_owners(task) if owned_physical else None,
                    _owned_timeline=owned_physical)
            # Local import avoids making the planner/data-motion module
            # part of the kernel import cycle; it also keeps the physical
            # adapter optional for analytical-only users.
            from .data_motion import register_physical_allocations, resolve_physical_task
            # A physical GDDR descriptor is only a preview until the
            # mutable L2 contract has run.  Resolve that contract first,
            # then submit the remaining miss/write-back traffic to the
            # DRAM core.  This preserves cache reuse while retaining the
            # physical event path for cold traffic.
            if "stateful_l2" in task.metadata:
                contract = task.metadata["stateful_l2"]
                l2_owner = str(contract["owner"])
                previous_l2_state = self._l2_states.get(l2_owner)
                l2_snapshot = (l2_owner, previous_l2_state)
                if previous_l2_state is not None:
                    l2_transaction = previous_l2_state[1].begin_transaction()
                register_physical_allocations(task, self.physical_runtime)
                registered_physical_allocations = True
                task = resolve_l2_task(task, self._l2_states, self.physical_runtime)
                task, is_physical = _materialize_l2_physical_access(task, self.physical_runtime)
            if is_physical:
                backing_offset_ns = task.metadata.get("l2_execution", {}).get(
                    "backing_arrival_offset_ns", 0.0
                )
                from . import data_motion
                timeline = self.physical_runtime.timeline
                previous_orders = timeline._owned_metric_orders
                # This scope ends before imports, observers, TP mirroring or
                # physical rollback can edit public dictionaries directly.
                timeline._owned_metric_orders = None
                accesses = task.metadata.get("memory_accesses") if type(task.metadata) is dict else None
                if (owned_physical and type(accesses) in (tuple, list) and len(accesses) > 1
                        and physical_snapshot.get("_owned_native_writers", False)
                        and physical_snapshot["timeline"].get("_owned_transaction") is not None
                        and all(getattr(data_motion, name) is original
                                for name, original in data_motion._NATIVE_METRIC_DISPATCH)
                        and timeline._can_own_metric_orders()):
                    timeline._owned_metric_orders = {}
                try:
                    task = resolve_physical_task(
                        task, self.physical_runtime, start_ns + backing_offset_ns,
                        _caller_managed_transaction=True,
                    )
                finally:
                    # A nested dispatch may have changed the outer task's
                    # roots/order; resume it with no retained ordinals.
                    if previous_orders is not None:
                        previous_orders.clear()
                    timeline._owned_metric_orders = previous_orders
                contract = task.metadata.get("stateful_l2")
                if contract is not None:
                    completion_span = float(task.metadata["physical_completion_ns"]) - float(start_ns)
                    order_resource = contract.get("order_resource")
                    envelopes = set(contract["envelope_resources"])
                    excluded_ids = envelopes | set(task.metadata["physical_demands_resource_ids"])
                    compute_span = max((d.service_ns for d in task.demands
                                        if d.resource_id not in excluded_ids), default=0.0)
                    duration = max(compute_span, completion_span) + contract.get("dependency_ns", 0.0)
                    demands = tuple(
                        replace(d, service_ns=completion_span if d.resource_id == order_resource else duration)
                        if d.resource_id in envelopes else d
                        for d in task.demands
                    )
                    task = replace(task, demands=demands, metadata={
                        **_l2_execution_metadata(task.metadata, duration, demands,
                            physical=True), "l2_execution": {
                        **task.metadata["l2_execution"],
                        "memory_completion_offset_ns": completion_span,
                    }})
            # Validation belongs to the same transaction as the Core
            # reservations and mutable L2 update. A rejected completion
            # must not leave either device state or an absent ready item.
            _validate_event_timing(task, start_ns, task.demands)
        except Exception:
            if l2_snapshot is not None:
                owner, previous = l2_snapshot
                if previous is None:
                    self._l2_states.pop(owner, None)
                else:
                    self._l2_states[owner] = previous
                if l2_transaction is not None:
                    previous[1].rollback_transaction(l2_transaction)
            if physical_snapshot is not None and not shared_physical:
                self.physical_runtime.restore(physical_snapshot)
            raise
        else:
            if l2_transaction is not None:
                cache = self._l2_states[l2_owner][1]
                cache.commit_transaction(l2_transaction)
            if physical_snapshot is not None and not shared_physical:
                self.physical_runtime.commit_transaction(physical_snapshot)
        demands = tuple(sorted(task.demands, key=lambda d: d.resource_id))
        # Even a zero-duration reservation can advance a resource's
        # frontier to its arrival. Such a resource has no positive busy
        # demand below, but must still be visible to ordinary tasks.
        if is_physical:
            from ._resource_ledger import publish_timeline_frontiers
            frame = getattr(self, "_token_frame", None)
            if (frame is not None and frame.owned_physical_frontiers
                    and frame.frontiers_initialized
                    and physical_snapshot.get("_owned_native_writers", False)):
                # Only the trusted native caller owns all timeline writes in
                # this closed frame. The merged task bill includes every
                # Core's touched resources, including zero-busy reservations;
                # timeline._touched alone contains only the last Core call.
                timeline_lanes = self.physical_runtime.timeline.lane_available
                for resource_id in task.metadata["physical_resource_last_intervals"]:
                    lanes = timeline_lanes.get(resource_id)
                    if lanes:
                        self.resource_available[resource_id] = lanes[0] if len(lanes) == 1 else min(lanes)
            else:
                publish_timeline_frontiers(
                    self.physical_runtime.timeline.lane_available, self.resource_available)
                if frame is not None:
                    frame.frontiers_initialized = True
    elif "stateful_l2" in task.metadata:
        try:
            l2_owner = str(task.metadata["stateful_l2"]["owner"])
            previous_l2_state = self._l2_states.get(l2_owner)
            if previous_l2_state is not None:
                l2_transaction = previous_l2_state[1].begin_transaction()
            task = resolve_l2_task(task, self._l2_states)
            _validate_event_timing(task, start_ns, task.demands)
        except Exception:
            if l2_owner is not None:
                if previous_l2_state is None:
                    self._l2_states.pop(l2_owner, None)
                else:
                    self._l2_states[l2_owner] = previous_l2_state
                    if l2_transaction is not None:
                        previous_l2_state[1].rollback_transaction(l2_transaction)
            # The task remains in its group queue; retain its global ready
            # entry too, so validation failure cannot silently drop it.
            raise
        else:
            if l2_transaction is not None:
                self._l2_states[l2_owner][1].commit_transaction(l2_transaction)
        demands = tuple(sorted(task.demands, key=lambda d: d.resource_id))
    else:
        demands = self._sorted_demands[task_id] if sorted_demands is None else sorted_demands
        try:
            _validate_event_timing(task, start_ns, demands)
        except Exception:
            raise

    return task, demands, is_physical, registered_physical_allocations, physical_snapshot


def account_task_execution(self, task, demands, is_physical, physical_snapshot,
                            start_ns, effective_ready_ns, dependency_ready_ns):
    """Commit one task bill to the shared persistent event calendar."""
    task_id = task.task_id
    from ._resource_ledger import account_task_demands
    queue_wait_ns = start_ns - effective_ready_ns
    end_ns, physical_ids, resource_predecessors, resource_lanes = account_task_demands(
        self, task, demands, is_physical, start_ns, effective_ready_ns)
    resource_last_interval = self.resource_last_interval
    resource_available = self.resource_available

    if is_physical:
        if not task.metadata.get("physical_resource_intervals"):
            physical_lane_completions = _publish_physical_summary(
                self, task, physical_ids, physical_snapshot, resource_predecessors)
        else:
            physical_intervals: Dict[str, List[Dict[str, object]]] = {}
            physical_lane_completions = []
            for stage in task.metadata.get("physical_resource_intervals", ()):
                if stage.resource_id is None:
                    continue
                resource_id = stage.resource_id
                interval = {
                    "task_id": task_id, "resource_id": resource_id,
                    "start_ns": stage.start_ns, "end_ns": stage.end_ns,
                }
                physical_intervals.setdefault(resource_id, []).append(interval)
            # Summary execution still retains the final real reservation. It
            # must publish the same readiness and task predecessor as detailed
            # execution whenever the lane can be identified unambiguously.
            for resource_id, (start, end) in task.metadata.get(
                    "physical_resource_last_intervals", {}).items():
                physical_intervals.setdefault(resource_id, []).append({
                    "task_id": task_id, "resource_id": resource_id,
                    "start_ns": start, "end_ns": end,
                })
            for resource_id in physical_ids:
                lanes = self.physical_runtime.timeline.lane_available.get(resource_id, ())
                before_lanes = physical_snapshot["timeline"]["lane_available"].get(resource_id, ())
                intervals = physical_intervals.get(resource_id, ())
                previous = resource_last_interval.get(resource_id)
                if (len(lanes) == 1 and previous is not None
                        and any(previous["end_ns"] == item["start_ns"] for item in intervals)):
                    resource_predecessors.setdefault(resource_id, dict(previous))
                if intervals:
                    last = max(intervals, key=lambda item: item["end_ns"])
                    if previous is None or last["end_ns"] >= previous["end_ns"]:
                        resource_last_interval[resource_id] = last
                for lane_index, end in enumerate(lanes):
                    before = before_lanes[lane_index] if lane_index < len(before_lanes) else 0.0
                    if end <= before:
                        continue
                    physical_lane_completions.append((resource_id, lane_index, end))
                    matches = [item for item in intervals
                               if item["end_ns"] == end and item["start_ns"] >= before]
                    key = (resource_id, lane_index)
                    if matches:
                        self._resource_lane_last_interval[key] = {
                            **max(matches, key=lambda item: item["start_ns"]), "lane": lane_index}
                    else:
                        # A truncated trace may omit the interval, but the
                        # changed lane clock proves this task's release point.
                        # Do not invent its unknown reservation start time.
                        self._resource_lane_last_interval[key] = {
                            "task_id": task_id, "resource_id": resource_id,
                            "end_ns": end, "lane": lane_index}
                if lanes:
                    # The next task may use any free lane. Stage detail is
                    # history, not a replacement for this calendar authority.
                    resource_available[resource_id] = min(lanes)
        task = replace(task, metadata={**task.metadata,
            "physical_resource_lane_completions": tuple(physical_lane_completions)})

    self._completed_end[task_id] = end_ns
    self._completion_leases[task_id] = 1
    self.completed_count += 1
    self.total_queue_wait_ns += queue_wait_ns
    self.total_service_ns += end_ns - start_ns
    if end_ns > self.makespan_ns:
        self.makespan_ns = end_ns

    return KernelEvent(
        task=task,
        start_ns=start_ns,
        end_ns=end_ns,
        dependency_ready_ns=dependency_ready_ns,
        effective_ready_ns=effective_ready_ns,
        demands=demands,
        resource_predecessors=resource_predecessors,
        resource_lanes=resource_lanes,
    )


from .event_kernel import (
    KernelEvent, _materialize_l2_physical_access, _validate_event_timing,
    _publish_physical_summary,
)
from .kernel_memory import resolve_l2_task, _l2_execution_metadata
