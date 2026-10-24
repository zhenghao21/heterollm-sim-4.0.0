"""Boundary-conditioned windows of the existing native physical Task graph.

Each window runs complete selected layers (including their communications)
on an independent copy of the real post-ingress kernel. Omitted predecessors
are declared boundary inputs, never fabricated physical events. A sampled
kernel cannot resume the full cohort, and never commits a serving Token.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, replace
import json
import time
from typing import Mapping

from .event_kernel import CompiledGraphLayout
from .live_execution_metrics import LiveCohortMetrics
from .native_row_bounds import first_access_ns


_IDENTITY_FIELDS = frozenset(("layer_id", "name", "label", "operator_id", "source_operator_id"))


class NativeLayerSamplingGuardError(ValueError):
    """A valid native graph cannot satisfy this sampling contract."""


def _semantic(value):
    if isinstance(value, Mapping):
        return {key: _semantic(item) for key, item in value.items() if key not in _IDENTITY_FIELDS}
    if isinstance(value, (tuple, list)):
        return [_semantic(item) for item in value]
    return value


def layer_inventory(scenario, specs):
    """Classify by native typed geometry/semantics, never model size alone."""
    view = scenario.model._execution_view
    by_layer = defaultdict(list)
    for task in specs:
        identity = task.metadata.get("layer_id")
        if identity is not None:
            by_layer[str(identity)].append(task)
    rows = []
    known = set()
    for index, descriptor in enumerate(view.layer_instances):
        layer = descriptor.layer
        identity = str(layer.layer_id)
        if identity in known:
            raise ValueError("native layer sampler requires unique expanded layer identities")
        known.add(identity)
        signature = _semantic(asdict(layer))
        key = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        tasks = tuple(by_layer.get(identity, ()))
        if not tasks:
            raise NativeLayerSamplingGuardError("native layer has no complete authored Task subgraph: " + identity)
        rows.append({"layer_id": identity, "layer_index": index, "class_key": key,
                     "signature": signature, "task_count": len(tasks)})
    if set(by_layer) - known:
        raise NativeLayerSamplingGuardError("native Task graph contains unknown layer identities")
    if not rows:
        raise NativeLayerSamplingGuardError("native model has no expanded layers to sample")
    return rows


def _graph_facts(specs, inventory):
    """Prove serial layer composition on the actual dependency graph."""
    by_id = {task.task_id: task for task in specs}
    if len(by_id) != len(specs):
        raise ValueError("native layer graph contains duplicate Task ids")
    layer_indices = {row["layer_id"]: row["layer_index"] for row in inventory}
    groups = {task.task_id: task.metadata.get("layer_id") for task in specs}
    children = defaultdict(list)
    indegree = {}
    for task in specs:
        if any(parent not in by_id for parent in task.dependencies):
            raise ValueError("native layer sampler requires the original closed cohort graph")
        indegree[task.task_id] = len(task.dependencies)
        for parent in task.dependencies:
            children[parent].append(task.task_id)
    queue = deque(task.task_id for task in specs if not indegree[task.task_id])
    order = []
    while queue:
        identity = queue.popleft()
        order.append(identity)
        for child in children[identity]:
            indegree[child] -= 1
            if not indegree[child]:
                queue.append(child)
    if len(order) != len(specs):
        raise ValueError("native layer Task graph is cyclic")
    roots, terminals = defaultdict(list), defaultdict(list)
    for task in specs:
        layer = groups[task.task_id]
        if layer is None:
            continue
        if not any(groups[parent] == layer for parent in task.dependencies):
            roots[layer].append(task.task_id)
        if not any(groups[child] == layer for child in children[task.task_id]):
            terminals[layer].append(task.task_id)
    terminal_bits, required = {}, {}
    position = 0
    for row in inventory:
        mask = 0
        for identity in terminals[row["layer_id"]]:
            terminal_bits[identity] = 1 << position
            mask |= 1 << position
            position += 1
        required[row["layer_id"]] = mask
    masks, layer_before = {}, {}
    for identity in order:
        mask, before = 0, 0
        for parent in by_id[identity].dependencies:
            mask |= masks[parent] | terminal_bits.get(parent, 0)
            before |= layer_before[parent]
            parent_layer = groups[parent]
            if parent_layer is not None:
                before |= 1 << layer_indices[parent_layer]
        masks[identity], layer_before[identity] = mask, before
    layer_after = {}
    for identity in reversed(order):
        after = 0
        for child in children[identity]:
            after |= layer_after[child]
            child_layer = groups[child]
            if child_layer is not None:
                after |= 1 << layer_indices[child_layer]
        layer_after[identity] = after
    failures = []
    for previous, current in zip(inventory, inventory[1:]):
        expected = required[previous["layer_id"]]
        if not expected or any(masks[root] & expected != expected for root in roots[current["layer_id"]]):
            failures.append({"before": previous["layer_id"], "after": current["layer_id"],
                             "reason": "not_all_previous_terminals_precede_every_next_root"})
    boundary = defaultdict(list)
    for task in specs:
        if groups[task.task_id] is not None:
            continue
        before, after = layer_before[task.task_id], layer_after[task.task_id]
        kind = ("before_first_layer" if not before and after else
                "after_last_layer" if before and not after else
                "interior" if before and after else "independent")
        boundary[kind].append(task.task_id)
    # A layer may have independent rank branches, but every branch must
    # join before the next layer starts. Non-layer work can only be added
    # directly when it too is separated by the native dependency edges.
    nonserial_boundary = []
    for kind, ids in boundary.items():
        wanted = frozenset(ids)
        boundary_roots = [identity for identity in ids
                          if not any(parent in wanted for parent in by_id[identity].dependencies)]
        boundary_terminals = [identity for identity in ids
                              if not any(child in wanted for child in children[identity])]
        if kind == "before_first_layer":
            # Prove each prefix terminal is an ancestor of every first
            # layer root, with compact bitsets instead of ancestor sets.
            bits = {identity: 1 << index for index, identity in enumerate(boundary_terminals)}
            ancestry = {}
            for identity in order:
                mask = 0
                for parent in by_id[identity].dependencies:
                    mask |= ancestry[parent] | bits.get(parent, 0)
                ancestry[identity] = mask
            required_mask = (1 << len(bits))-1
            separated = all(ancestry[identity] & required_mask == required_mask
                            for identity in roots[inventory[0]["layer_id"]])
        elif kind == "after_last_layer":
            required_mask = required[inventory[-1]["layer_id"]]
            separated = all(masks[identity] & required_mask == required_mask
                            for identity in boundary_roots)
        else:
            separated = False
        if not separated:
            nonserial_boundary.append(kind)
    return {"qualified": not failures, "basis": "all_previous_layer_terminals_are_ancestors_of_all_next_layer_roots",
            "serial_layer_boundaries_proven": not failures,
            "nonserial_non_layer_groups": nonserial_boundary,
            "layer_order": [row["layer_id"] for row in inventory], "failures": failures,
            "roots": dict(roots), "terminals": dict(terminals)}, dict(boundary)


# Per-request KV records are the same rows seen through several views of a
# task's metadata. Window execution reads them only for the tasks it runs.
PER_REQUEST_RECORD_KEYS = (
    "memory_accesses", "physical_address_bindings", "buffer_accesses", "physical_allocations",
    "rhs_buffer_accesses", "kv_writeback_accesses", "persistent_request_buffers",
    "persistent_buffer_ids", "generic_kv_storage")


def drop_unsampled_request_records(metadata_rows, kept_layers):
    """Delete the per-request records of tasks in layers no window will run.

    ``metadata_rows`` are the task metadata dicts of every view that shares
    the records; the rows are only released once no view keeps them. Tasks
    outside any layer, and tasks of ``kept_layers``, are untouched.
    """
    kept = frozenset(kept_layers)
    dropped = 0
    for metadata in metadata_rows:
        layer = metadata.get("layer_id")
        if layer is None or str(layer) in kept:
            continue
        for key in PER_REQUEST_RECORD_KEYS:
            if metadata.pop(key, None) is not None:
                dropped += 1
    return dropped


def _independent_kernel(source, executor=None):
    if source.has_active_tasks:
        raise NativeLayerSamplingGuardError("native layer boundary requires no active full-cohort tasks")
    # Resource ownership is immutable; copy all mutable controllers, calendars
    # and allocations together so their internal shared references survive.
    memo = {id(source.resource_owners): source.resource_owners}
    kernel = deepcopy(source, memo)
    if executor is None:
        return kernel, None
    # The native PD executor reserves this kernel's own physical runtime and
    # cache states. One memo rebinds the copy to the copied calendars.
    copied = deepcopy(executor, memo)
    if (copied.runtime is not kernel.physical_runtime
            or copied.cache_states is not kernel._l2_states
            or copied.cache_owners is not kernel._physical_allocation_cache_owners):
        raise NativeLayerSamplingGuardError("native PD executor is not bound to the sampled kernel")
    return kernel, copied


class _ConcurrentImports:
    """Continue real in-flight native PD imports on an independent copy.

    Each import keeps its own executor state (row, offset, action, token), so
    the continuation submits exactly the phases the live runtime would. As in
    ``_advance_prefetches_before_kernel_step``, a phase due no later than the
    next ready Task is submitted first, then every frontier is republished.
    Final commit/publication of the imported context is not performed.
    """

    def __init__(self, kernel, executor, request_ids):
        self.kernel, self.executor = kernel, executor
        states = {state.get("request_id"): state for state in executor._async_plan_states.values()
                  if not state.get("done") and "destinations" in state}
        missing = sorted(set(request_ids)-set(states))
        if missing:
            raise NativeLayerSamplingGuardError("native PD import is not in flight: "+", ".join(missing))
        self.states = [states[identity] for identity in request_ids]
        self.writes, self.submitted = [], 0

    def advance_until(self, time_ns):
        from ._resource_ledger import publish_owner_frontiers, publish_timeline_frontiers
        kernel = self.kernel
        while True:
            due = min(((state["completion_ns"], index) for index, state in enumerate(self.states)
                       if not state["final"]), default=None)
            if due is None or due[0] > time_ns:
                return
            state = self.states[due[1]]
            self.executor._advance_external_phase(state, state["phase_token"])
            action = state["actions"][state["action"]-1]
            if action[0] == "memory" and action[2] == "write":
                self.writes.append((state["completion_ns"], action[4], state["request_id"]))
            self.submitted += 1
            publish_timeline_frontiers(kernel.physical_runtime.timeline.lane_available,
                                       kernel.resource_available)
            publish_owner_frontiers(kernel.resource_owners, kernel.resource_available)

    def before_step(self, kernel):
        ready = kernel.peek_ready_key()
        if ready is not None:
            self.advance_until(float(ready[0]))

    def observation(self, start_ns, end_ns):
        inside = [row for row in self.writes if start_ns < row[0] <= end_ns]
        identities = [state["request_id"] for state in self.states]
        return {"request_ids": identities,
            "submitted_phases": self.submitted,
            "written_bytes_in_window": sum(row[1] for row in inside),
            "write_completions_ns": [row[0] for row in inside],
            "write_phases_in_window": len(inside),
            "written_bytes_by_request": {identity: sum(row[1] for row in inside if row[2] == identity)
                                         for identity in identities},
            "write_phases_by_request": {identity: sum(row[2] == identity for row in inside)
                                        for identity in identities},
            "final_reached": [state["request_id"] for state in self.states if state["final"]],
            "import_spans_ns": {state["request_id"]: (state["start_ns"], state["completion_ns"])
                                for state in self.states if state["final"]},
            "scope": "real_in_flight_native_PD_continued_on_the_sample_copy; destination_commit_not_performed"}


def remaining_import_bytes(state):
    """Logical bytes of a live import plan not yet covered by its row cursor."""
    done = sum(row["size_bytes"] for row in state["destinations"][:state["row"]])+state["offset"]
    return state["logical_bytes"]-done


def _execute_window(source, all_specs, selected_ids, *, origin_ns, control, sample_id,
                    concurrent_imports=None, row_terms=None):
    """Run the selected tasks on an independent copy of ``source``.

    With ``row_terms`` (task id -> term key) the record also carries
    ``first_access_ns``: per term, how far the boundary's DRAM row state
    can move it (see native_row_bounds).
    """
    started = time.perf_counter()
    kernel, executor = _independent_kernel(
        source, None if concurrent_imports is None else concurrent_imports[0])
    imports = None
    if executor is not None:
        imports = _ConcurrentImports(kernel, executor, concurrent_imports[1])
        # Reach the window boundary with every earlier PD phase submitted.
        imports.advance_until(origin_ns)
    clone_wall = time.perf_counter()-started
    wanted = frozenset(selected_ids)
    selected = []
    omitted = set()
    inputs = {}
    for task in all_specs:
        if task.task_id not in wanted:
            continue
        omitted.update(parent for parent in task.dependencies if parent not in wanted)
        for access in task.metadata.get("memory_accesses", ()):
            if access.get("operation") != "read" or not access.get("buffer_id"):
                continue
            owner = access.get("physical_owner", task.metadata.get("physical_owner"))
            generation = access.get("allocation_generation", 0)
            key = (owner, access["buffer_id"], generation)
            allocator = kernel.physical_runtime.allocators.get(owner)
            exists = allocator is not None and allocator.get_allocation(access["buffer_id"], generation) is not None
            inputs.setdefault(key, {"physical_owner": owner, "buffer_id": access["buffer_id"],
                "generation": generation, "present_at_real_boundary": exists,
                "content_semantics": "declared_boundary_input_no_prior_write_event" if not exists else "real_post_ingress_allocation"})
        selected.append(replace(task, dependencies=tuple(parent for parent in task.dependencies if parent in wanted),
                                earliest_start_ns=max(float(task.earliest_start_ns), origin_ns)))
    selected = tuple(selected)
    if not selected:
        raise ValueError("native physical layer window is empty")
    # The source is read only throughout independent window execution.
    metrics = LiveCohortMetrics(resource_owners=kernel.resource_owners, origin_ns=origin_ns)
    first, last, count, physical = None, origin_ns, 0, 0
    layer_ends = {}
    facts = [] if row_terms is not None else None
    def observe(event):
        nonlocal first, last, count, physical
        metrics.observe(event)
        if facts is not None:
            md = event.task.metadata
            owner = md.get("physical_owner")
            facts.append((event.task.task_id, event.start_ns, event.end_ns, event.task.dependencies,
                          tuple(demand.resource_id for demand in event.task.demands),
                          tuple((access.get("physical_owner", owner), access["address"], access["byte_count"])
                                for access in md.get("memory_accesses", ())
                                if type(access.get("address")) is int)))
        first = event.start_ns if first is None else min(first, event.start_ns)
        last = max(last, event.end_ns)
        layer_id = event.task.metadata.get("layer_id")
        if layer_id is not None:
            layer_ends[layer_id] = max(layer_ends.get(layer_id, event.end_ns), event.end_ns)
        count += 1
        physical += int(bool(event.task.metadata.get("physical_execution_by_owner")))
        if count == len(selected) or count % 256 == 0:
            try:
                control.raise_if_cancelled()
                control.report("native_layer_tasks", count, len(selected), unit="schedule_tasks",
                    simulated_time_ns=event.end_ns, message="正在执行原生完整层窗口",
                    metadata={"sample_id": sample_id, "full_cohort_committed": False,
                              "visible_output_tokens": 0})
            finally:
                frame = getattr(kernel, "_token_frame", None)
                if frame is not None:
                    frame.invalidate_frontiers()
    control.raise_if_cancelled()
    before = time.perf_counter()
    layout = CompiledGraphLayout.compile(selected)
    compiled_wall = time.perf_counter()-before
    physical_started = time.perf_counter()
    kernel._drain_prevalidated_compiled(selected, layout, event_observer=observe,
        index_physical_allocations=True, owned_physical_frontiers=True,
        before_step=None if imports is None else imports.before_step)
    physical_wall = time.perf_counter()-physical_started
    metadata_started = time.perf_counter()
    metadata = metrics.metadata()
    metadata_wall = time.perf_counter()-metadata_started
    terminal = wanted - {parent for task in selected for parent in task.dependencies}
    row_state = None if facts is None else first_access_ns(
        facts, {owner: active.core.config for owner, active in kernel.physical_runtime.runtimes.items()},
        row_terms)
    carry = max(0., max(source.physical_runtime.timeline.ready_ns.values(), default=origin_ns)-origin_ns)
    return {"sample_id": sample_id, "window_id": sample_id,
        "concurrent_imports": None if imports is None else imports.observation(first, last),
        "layer_completion_ns": layer_ends, "first_access_ns": row_state,
        "start_ns": first, "first_task_start_ns": first,
        "end_ns": last, "duration_ns": last-first, "boundary_origin_ns": origin_ns,
        "elapsed_from_boundary_ns": last-origin_ns, "boundary_wait_ns": first-origin_ns,
        "wall_seconds": time.perf_counter()-started,
        "wall_breakdown_seconds": {"source_kernel_copy": clone_wall,
            "compiled_subgraph": compiled_wall, "physical_tasks": physical_wall,
            "physical_metrics": metadata_wall},
        "task_count": count, "executed_task_count": count,
        "physical_task_count": physical,
        "visible_output_tokens": 0, "full_cohort_committed": False,
        "sample_state": "independent_native_kernel_copy_not_full_cohort_prefix",
        "physical_observations": {key: metadata.get(key) for key in (
            "storage_traffic", "dram_traffic", "sram_traffic", "resource_busy_ns", "category_time_ns")},
        "input_boundary": {"qualified": True,
                           "qualification_scope": "native_logical_inputs_of_complete_selected_subgraph; controller_phase_bounded_separately",
                           "initial_carry_upper_ns": carry,
                           "omitted_dependency_task_ids": sorted(omitted),
                           "logical_buffer_inputs": list(inputs.values()),
                           "predecessor_events_fabricated": False},
        "output_boundary": {"qualified": True,
            "qualification_scope": "all_selected_native_tasks_and_terminal_events_completed; not_in_situ_full_cohort_state",
            "terminal_task_ids": sorted(terminal),
            "resource_frontiers": dict(kernel.resource_available),
            "controller_state_resumable_as_full_cohort": False}}
