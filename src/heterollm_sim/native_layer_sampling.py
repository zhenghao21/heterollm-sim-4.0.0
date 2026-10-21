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


def _select_windows(inventory, options):
    size = options.layer_window_size
    calibration = options.calibration_layer_windows
    validation = options.validation_layer_windows
    classes = defaultdict(list)
    for row in inventory:
        classes[row["class_key"]].append(row)
    windows = []
    for class_key, rows in classes.items():
        candidates = []
        for start in range(len(rows) - size + 1):
            chunk = rows[start:start+size]
            if all(right["layer_index"] == left["layer_index"] + 1 for left, right in zip(chunk, chunk[1:])):
                candidates.append(chunk)
        if not candidates:
            raise NativeLayerSamplingGuardError("native class has no complete contiguous layer window")
        count = calibration + validation
        if len(rows) < count * size:
            # A small class can be executed exhaustively. It is never
            # duplicated to pretend an independent holdout exists.
            windows.extend({"role": "exhaustive", "class_key": class_key, "layers": [row]}
                           for row in rows)
            continue
        selected, used = [], set()
        for index in range(count):
            target = round(index * (len(candidates) - 1) / max(1, count - 1))
            available = sorted(range(len(candidates)), key=lambda item: (abs(item-target), item))
            chosen = next((item for item in available
                if not {row["layer_id"] for row in candidates[item]} & used), None)
            if chosen is None:
                raise NativeLayerSamplingGuardError("native layer calibration/validation windows overlap")
            chunk = candidates[chosen]
            used.update(row["layer_id"] for row in chunk)
            selected.append({"role": "calibration" if index < calibration else "validation",
                             "class_key": class_key, "layers": chunk})
        windows.extend(selected)
    if len(windows) > options.max_sample_layer_windows:
        raise NativeLayerSamplingGuardError("native layer class coverage exceeds the explicit sample-window limit")
    return windows


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
        return {"request_ids": [state["request_id"] for state in self.states],
            "submitted_phases": self.submitted,
            "written_bytes_in_window": sum(row[1] for row in inside),
            "write_completions_ns": [row[0] for row in inside],
            "write_phases_in_window": len(inside),
            "final_reached": [state["request_id"] for state in self.states if state["final"]],
            "scope": "real_in_flight_native_PD_continued_on_the_sample_copy; destination_commit_not_performed"}


def import_progress_alone(source, executor, request_ids, *, start_ns, end_ns):
    """Advance only the in-flight imports on a copy; nothing else runs."""
    kernel, copied = _independent_kernel(source, executor)
    imports = _ConcurrentImports(kernel, copied, request_ids)
    imports.advance_until(start_ns)
    imports.advance_until(end_ns)
    return imports.observation(start_ns, end_ns)


def remaining_import_bytes(state):
    """Logical bytes of a live import plan not yet covered by its row cursor."""
    done = sum(row["size_bytes"] for row in state["destinations"][:state["row"]])+state["offset"]
    return state["logical_bytes"]-done


# Paired PD interference windows: a long calibration run and a shorter
# holdout run of one layer class, at different import progress.
PD_CALIBRATION_LAYERS, PD_HOLDOUT_LAYERS = 24, 12
PD_PREROLL_NS, PD_HOLDOUT_SHIFT_NS = 1e6, 10e6


def _pd_pair(source, specs, ids, *, origin_ns, control, sample_id, imports, progress_metadata):
    pair = {}
    for name, concurrent in (("quiet", None), ("concurrent", imports)):
        control.raise_if_cancelled()
        record = _execute_window(source, specs, ids, origin_ns=origin_ns, control=control,
            sample_id=f"{sample_id}-{name}", concurrent_imports=concurrent,
            progress_metadata=progress_metadata)
        pair[name] = {key: record[key] for key in ("start_ns", "end_ns", "duration_ns",
            "boundary_origin_ns", "elapsed_from_boundary_ns", "task_count",
            "layer_completion_ns", "concurrent_imports", "wall_seconds")}
    return pair


def sample_native_pd_interference(runtime, prepared, control, *, prefix="", progress_metadata=None,
                                  calibration_layers=PD_CALIBRATION_LAYERS,
                                  holdout_layers=PD_HOLDOUT_LAYERS,
                                  preroll_ns=PD_PREROLL_NS, holdout_shift_ns=PD_HOLDOUT_SHIFT_NS):
    """Measure native work under one real in-flight PD import (k=1).

    Each pair runs the same complete layers (or non-layer group) from one
    boundary on independent copies: quiet, and with the in-flight import
    holding the most remaining bytes continued phase by phase; any other
    in-flight import stays frozen on the copy. Calibration and holdout differ
    in layers and in import progress (the holdout preroll is longer).
    """
    started = time.perf_counter()
    base = {"schema": "native-PD-interference-samples/v1", "qualified": False,
        "calibration_layers": calibration_layers, "holdout_layers": holdout_layers,
        "preroll_ns": preroll_ns, "holdout_shift_ns": holdout_shift_ns}
    executor, pending = runtime._external_executor, list(runtime._external_pending)
    if executor is None or not pending:
        return {**base, "reason": "native_PD_import_not_in_flight_at_cutoff"}
    states = {state.get("request_id"): state for state in executor._async_plan_states.values()
              if not state.get("done") and "destinations" in state}
    if set(pending)-set(states):
        return {**base, "reason": "native_PD_in_flight_plan_state_unavailable"}
    remaining = {identity: remaining_import_bytes(states[identity]) for identity in pending}
    continued = max(pending, key=lambda identity: (remaining[identity], identity))
    inventory = prepared["inventory"]
    if len({row["class_key"] for row in inventory}) != 1:
        return {**base, "reason": "native_PD_interference_requires_one_layer_class"}
    count = len(inventory)
    calibration_start, holdout_start = count//12, count*9//16
    if (min(calibration_layers, holdout_layers) < 2 or calibration_start+calibration_layers > holdout_start
            or holdout_start+holdout_layers > count):
        return {**base, "reason": "native_PD_interference_layer_windows_do_not_fit"}
    source, specs, by_layer = prepared["source"], prepared["specs"], prepared["by_layer"]
    imports, origin = (executor, [continued]), prepared["origin"]
    layers = []
    for role, start, size, shift in (("calibration", calibration_start, calibration_layers, 0.),
                                     ("holdout", holdout_start, holdout_layers, holdout_shift_ns)):
        identities = [row["layer_id"] for row in inventory[start:start+size]]
        ids = [identity for layer in identities for identity in by_layer[layer]]
        window_origin = origin+preroll_ns+shift
        layers.append({"role": role, "layer_ids": identities, "origin_ns": window_origin,
            **_pd_pair(source, specs, ids, origin_ns=window_origin, control=control,
                sample_id=prefix+"pd-interference-"+role, imports=imports,
                progress_metadata=progress_metadata)})
    non_layer = [{"position": kind, "origin_ns": origin+preroll_ns,
        **_pd_pair(source, specs, ids, origin_ns=origin+preroll_ns, control=control,
            sample_id=prefix+"pd-interference-non-layer-"+kind, imports=imports,
            progress_metadata=progress_metadata)}
        for kind, ids in prepared["boundary"].items()]
    return {**base, "qualified": True, "reason": None, "continued_request_id": continued,
        "in_flight_request_ids": pending, "remaining_bytes_at_cutoff": remaining,
        "boundary_origin_ns": origin, "layer_count": count,
        "layer_windows": layers, "non_layer_windows": non_layer,
        "wall_seconds": time.perf_counter()-started,
        "scope": "paired_same_boundary_quiet_and_single_real_in_flight_PD_import; other_in_flight_imports_frozen_on_the_copy",
        "physical_state_advanced": False, "source_kernel_modified": False}


def _execute_window(source, all_specs, selected_ids, *, origin_ns, control, sample_id,
                    replay_capture=None, progress_metadata=None, concurrent_imports=None):
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
    # The budget interpreter needs the actual allocator-resolved access
    # descriptors and native resource bills, rather than the authored
    # placeholder addresses or a synthetic fixed-gap IO stream. Keep this
    # private, in-process capture outside every serialized sample field.
    # The source is read only throughout independent window execution.
    resolved_events = [] if replay_capture is not None else None
    metrics = LiveCohortMetrics(resource_owners=kernel.resource_owners, origin_ns=origin_ns)
    first, last, count, physical = None, origin_ns, 0, 0
    layer_ends = {}
    def observe(event):
        nonlocal first, last, count, physical
        metrics.observe(event)
        if resolved_events is not None:
            resolved_events.append(event)
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
                    metadata={**(progress_metadata or {}),
                              "sample_id": sample_id, "full_cohort_committed": False,
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
    carry = max(0., max(source.physical_runtime.timeline.ready_ns.values(), default=origin_ns)-origin_ns)
    if replay_capture is not None:
        replay_capture.update(source_kernel=source, selected_specs=selected,
            resolved_events=tuple(resolved_events), boundary_origin_ns=origin_ns,
            sample_id=sample_id, source_kernel_modified=False,
            complete_native_window=True)
    return {"sample_id": sample_id, "window_id": sample_id,
        "concurrent_imports": None if imports is None else imports.observation(first, last),
        "layer_completion_ns": layer_ends,
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


def _subset_stages(stages, ids):
    wanted = frozenset(ids)
    rows = []
    # Prepared Task identities survive the namespacing adapter in name.
    for stage in stages:
        tasks = tuple(task for task in stage.execution_tasks if task.task_id in wanted)
        if tasks:
            rows.append(replace(stage, execution_tasks=tasks))
    return tuple(rows)


def prepare_native_layer_windows(runtime, cohort, cost, stages, request_ready_ns, control):
    """Prepare one unchanged post-PD source and its original native IR once."""
    from .native_window_producer import summarize_native_stages
    started = time.perf_counter()
    control.raise_if_cancelled()
    source = runtime._execution_resource_kernel
    paid_frontier = max(source.physical_runtime.timeline.ready_ns.values(), default=float(runtime.now))
    # Each independent window declares a quiescent paid-input boundary.
    # Its copied calendars contain all real PD reservations. Waiting for
    # those reservations here avoids multiplying one outstanding PD tail
    # into every layer's calibration bias. The whole-cohort budget retains
    # this initial wait once; no source clock, event or Token is advanced.
    origin = max(float(runtime.now), float(request_ready_ns), float(source.makespan_ns), paid_frontier)
    namespace = "native-layer-physical-sample." + cohort.cohort_id
    specs, sources = runtime._stage_task_specs(stages,
        {stage.stage_id: origin for stage in stages}, namespace)
    inventory = layer_inventory(runtime.plan.scenario, specs)
    composition, boundary = _graph_facts(specs, inventory)
    if not composition["qualified"]:
        raise NativeLayerSamplingGuardError("native layer windows require proven serial layer composition")
    by_layer = defaultdict(list)
    tasks_by_id = {task.task_id: task for task in specs}
    names = {task.task_id: task.name for task in specs}
    from .native_collective_row_reuse import native_allocation_scope
    allocation_scope = native_allocation_scope(stages, composition, boundary, names, source)
    for task in specs:
        if task.metadata.get("layer_id") is not None:
            by_layer[task.metadata["layer_id"]].append(task.task_id)
    inventory_by_id = {row["layer_id"]: row for row in inventory}
    for row in inventory:
        control.raise_if_cancelled()
        original_ids = [names[identity] for identity in by_layer[row["layer_id"]]]
        summary = summarize_native_stages(_subset_stages(stages, original_ids),
            runtime.plan.scenario, source.physical_runtime, allow_bounded_clock=True,
            native_runtime=runtime, allocation_scope=allocation_scope)
        row["native_ir_summary"] = summary
        # Typed model semantics alone do not prove that two actual native
        # subgraphs carry the same routed expert work or memory shape.
        # Addresses stay outside the class: their row/page/cache effects
        # are explicitly charged by the physical phase budget.
        native_shape = {"operator_signature": summary["operator_signature"],
            "balanced_expert_shape": [item[1:] for item in summary["native_balanced_expert_signature"]],
            "task_count": summary["task_count"],
            "collective_row_reuse_layouts": summary["collective_row_reuse_layouts"],
            "owner_io_shape": {owner: {key: facts[key] for key in (
                "read_count", "write_count", "read_bytes", "write_bytes", "context_read_bytes")}
                for owner, facts in summary["owner_accesses"].items()}}
        row["signature"]["native_operation_geometry"] = native_shape
        row["class_key"] = json.dumps(row["signature"], ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"))
    return {"runtime": runtime, "cohort": cohort, "cost": cost, "stages": stages,
        "request_ready_ns": request_ready_ns, "source": source,
        "source_runtime_now": runtime.now, "source_makespan_ns": source.makespan_ns,
        "paid_frontier": paid_frontier, "origin": origin, "specs": specs,
        "inventory": inventory, "composition": composition, "boundary": boundary,
        "by_layer": by_layer, "tasks_by_id": tasks_by_id, "names": names,
        "inventory_by_id": inventory_by_id, "non_layer_samples": None,
        "pd_interference_samples": None, "allocation_scope": allocation_scope,
        "preparation_wall_seconds": time.perf_counter()-started}


def sample_native_layer_windows(runtime, cohort, cost, stages, request_ready_ns, options, control,
                                *, prepared=None):
    """Execute complete consecutive native layers on unchanged source copies.

    A preparation is reusable only within the same candidate, cohort and
    real post-PD source. Each attempt still executes its selected complete
    windows; no old attempt is merged into its calibration or holdout set.
    """
    from .native_window_producer import summarize_native_stages
    started = time.perf_counter()
    control.raise_if_cancelled()
    reused_preparation = prepared is not None
    if prepared is None:
        prepared = prepare_native_layer_windows(runtime, cohort, cost, stages,
            request_ready_ns, control)
    if (prepared["runtime"] is not runtime or prepared["cohort"] is not cohort
            or prepared["cost"] is not cost or prepared["stages"] is not stages
            or prepared["source"] is not runtime._execution_resource_kernel
            or prepared["source_runtime_now"] != runtime.now
            or prepared["source_makespan_ns"] != runtime._execution_resource_kernel.makespan_ns
            or prepared["request_ready_ns"] != request_ready_ns):
        raise NativeLayerSamplingGuardError("native_layer_preparation_source_or_IR_changed")
    source, origin = prepared["source"], prepared["origin"]
    specs, inventory = prepared["specs"], prepared["inventory"]
    by_layer, tasks_by_id = prepared["by_layer"], prepared["tasks_by_id"]
    names, inventory_by_id = prepared["names"], prepared["inventory_by_id"]
    composition, boundary = dict(prepared["composition"]), prepared["boundary"]
    paid_frontier = prepared["paid_frontier"]
    windows = _select_windows(inventory, options)
    attempt = getattr(runtime, "_native_layer_sampling_attempt", {})
    attempt_tag = ({"sampling_attempt_index": attempt["attempt_index"],
        "layer_window_size": attempt["layer_window_size"]} if attempt else {})
    prefix = (f"attempt-{attempt['attempt_index']:02d}.w{options.layer_window_size}."
              if attempt else "")
    previous_windows = attempt.get("previous_completed_windows", 0)
    previous_tasks = attempt.get("previous_executed_tasks", 0)
    samples = []
    # The orchestration wall deadline may fire in a later window. Expose
    # completed windows immediately, never the unfinished physical graph.
    runtime._native_layer_sample_records = samples
    capture_replays = getattr(runtime, "_capture_native_layer_phase_replays", False) is True
    if capture_replays:
        runtime._native_layer_phase_replays = {}
    context = {item.request_id: item.context_tokens for item in cohort.items}
    for index, window in enumerate(windows):
        control.raise_if_cancelled()
        identities = [row["layer_id"] for row in window["layers"]]
        ids = [identity for layer in identities for identity in by_layer[layer]]
        original_ids = [names[identity] for identity in ids]
        ranks = tuple(sorted({tasks_by_id[identity].metadata["rank"] for identity in ids
                              if type(tasks_by_id[identity].metadata.get("rank")) is int}))
        if not ranks:
            raise NativeLayerSamplingGuardError("native complete layer window has no authored rank identity")
        sample_id = prefix+f"layer-window-{index:03d}"
        replay = {} if capture_replays else None
        record = _execute_window(source, specs, ids, origin_ns=origin, control=control,
                                 sample_id=sample_id, replay_capture=replay,
                                 progress_metadata=attempt_tag)
        if capture_replays:
            runtime._native_layer_phase_replays[sample_id] = replay
        record.update(role=window["role"], class_key=window["class_key"], layer_ids=tuple(identities),
            cohort_id=cohort.cohort_id, cohort_kind=cohort.kind,
            decode_step=int(runtime._completed_decode_steps), rank_ids=ranks,
            context_tokens_by_request=dict(context),
            layer_indices=[inventory_by_id[identity]["layer_index"] for identity in identities],
            native_ir_summary=summarize_native_stages(_subset_stages(stages, original_ids),
                runtime.plan.scenario, source.physical_runtime, allow_bounded_clock=True,
                native_runtime=runtime, allocation_scope=prepared["allocation_scope"]))
        record["physical_observations"].update(class_key=window["class_key"],
            sample_state=record["sample_state"], input_boundary=record["input_boundary"],
            output_boundary=record["output_boundary"],
            boundary_wait_ns=record["boundary_wait_ns"],
            wall_breakdown_seconds=record["wall_breakdown_seconds"],
            native_ir_summary=record["native_ir_summary"])
        samples.append(record)
        control.report("native_layer_sample_completed", previous_windows+len(samples),
            previous_windows+len(windows), unit="work_units",
            message="完整物理层窗口已保存，不计完整批次或输出词元",
            metadata={**attempt_tag, "physical_layer_progress": {
                **attempt_tag, "attempt_completed_windows": len(samples),
                "completed_windows": previous_windows+len(samples),
                "total_windows": previous_windows+len(windows),
                "executed_tasks": previous_tasks+sum(row["executed_task_count"] for row in samples),
                "complete_cohorts": 0, "visible_output_tokens": 0}})
    reused_non_layer = prepared["non_layer_samples"] is not None
    if reused_non_layer:
        non_layer = prepared["non_layer_samples"]
    else:
        non_layer = []
        for kind, ids in boundary.items():
            control.raise_if_cancelled()
            record = _execute_window(source, specs, ids, origin_ns=origin, control=control,
                                     sample_id=prefix+"non-layer-"+kind,
                                     progress_metadata=attempt_tag)
            wanted = frozenset(ids)
            record.update(role="non_layer", position=kind,
                original_task_positions=[index for index, task in enumerate(specs) if task.task_id in wanted],
                native_ir_summary=summarize_native_stages(_subset_stages(stages, [names[identity] for identity in ids]),
                    runtime.plan.scenario, source.physical_runtime, allow_bounded_clock=True,
                    native_runtime=runtime))
            non_layer.append(record)
        prepared["non_layer_samples"] = non_layer
    # Future B=1 imports are budgeted from measured interference only; the
    # pairs depend on the source and import state, never on the window size.
    pd_interference = prepared["pd_interference_samples"]
    reused_pd_interference = pd_interference is not None
    executor = runtime._external_executor
    if (not reused_pd_interference and runtime.plan.scheduler.max_num_seqs == 1
            and executor is not None
            and any(spec.request_id not in runtime._external_initialized for spec in runtime.plan.requests)):
        pd_interference = sample_native_pd_interference(runtime, prepared, control,
            prefix=prefix, progress_metadata=attempt_tag)
        prepared["pd_interference_samples"] = pd_interference
    composition["boundary_guard_ns"] = sum(row["duration_ns"] for row in non_layer
        if row["position"] in composition["nonserial_non_layer_groups"])
    composition["boundary_guard_basis"] = ("native_DAG_separated_prefix_suffix; potentially_overlapping_non_layer_groups_charged_in_full; "
        "controller_row_cache_refresh_and_allocation_state_are_separate_per_layer_IR_phase_guards")
    composition["source_pending_carry_upper_ns"] = max(0., paid_frontier-float(runtime.now))
    composition["sample_boundary_origin_ns"] = origin
    composition["sample_boundary_scope"] = "independent_copy_after_paid_resource_frontiers; source_clock_not_advanced"
    return {"schema": "native-layer-physical-samples/v1", "layer_inventory": inventory,
        "records": samples, "layer_window_samples": samples, "non_layer_samples": non_layer,
        "pd_interference": pd_interference, "pd_interference_reused": reused_pd_interference,
        "non_layer_coverage_complete": sum(map(len, boundary.values())) == sum(
            task.metadata.get("layer_id") is None for task in specs),
        "layer_composition": composition, "cohort_kind": cohort.kind,
        "context_tokens": [{"request_id": item.request_id, "context_tokens": item.context_tokens,
                            "phase": item.phase, "token_count": item.token_count} for item in cohort.items],
        "original_task_count": len(specs), "sampled_task_count": sum(row["task_count"] for row in (*samples, *non_layer)),
        "wall_seconds": time.perf_counter()-started, "source_cutoff_ns": origin,
        "prepared_source_IR_reused": reused_preparation,
        "non_layer_samples_reused": reused_non_layer,
        "physical_state_advanced": False, "source_kernel_modified": False,
        "full_cohorts_committed": 0, "visible_output_tokens": 0,
        "controller_state": "independent_samples_discarded_not_a_full_prefix"}
