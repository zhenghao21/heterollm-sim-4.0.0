"""Stationary continuous-batching forecast from native layer samples.

Staggered preloaded seeds put a continuous-refill Decode node in its steady
state from the first step: B running requests with uniformly spread contexts,
freed slots refilled at the next step from a pool of imported requests. Every
step then serves the same native work, so one step is composed from native
samples taken on independent copies of the first cohort boundary:

* calibration run: the step prefix and the first layers, all in situ;
* holdout run: the last layers and the step suffix, later in time;
* the step's cold ends once more at the other origin;
* two real in-flight PD imports, each continued alone beside the step
  prefix and first two layers, against the calibration run without imports;
* when a group starts k > 1 imports at one boundary of the measured steps,
  two sets of k real in-flight imports, each set continued together beside
  the step prefix and every calibration layer.

Each observation is first widened by the row-state bound of the DRAM banks
its run reaches first: those start from the boundary's open rows, not from
the rows the previous step leaves (see native_row_bounds). Layer steps take
the per-class hull of both runs widened by the drift of the class means
between them. The prefix with the entry layer, and the suffix, take the
hull of their two observations widened by its width. The imports landing
in one step all start at the boundary before it, so k of them run together
and add their measured joint E(k) (not k times the single e) to the quiet
work they overlap; with B running requests of ``output`` tokens B/output
imports land per step. The model holds only while one boundary's imports
finish before the group starts its next ones. No IR worst-case guard is
used. The default ``pd_concurrency_envelope`` samples only k=1 and the busiest k_max and
bounds intermediate E(k) and span by their endpoint intervals, conditional
on both quantities being nondecreasing with k. Disabling it measures every
concurrency term.

One assumption is not bounded (SAME_CLASS_ADDRESS_ASSUMPTION): an
unsampled layer takes its class's sampled step. Layers of one class run the
same native work on different buffer addresses, so their DRAM row openings
differ where the buffers straddle row boundaries differently. A per-command
path bound of those openings is far wider than the 10% budget, while
sampled same-class layers at different addresses measure the same step; the
representative-point full native runs check it layer by layer.

With attention data parallelism (``ep_degree`` > 1) the replica's B requests
sit in ep groups of exactly B/ep (the cohort is rejected otherwise), each
group refilling its own freed slots from its own share of the import pool.
Layers stay synchronous across groups, so a step lasts as long as its
slowest group: each step's import term is E(k) of its busiest group's k
landings (see ``_group_refill_schedule``).
"""
from __future__ import annotations

from array import array
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace
from fractions import Fraction
import json
import math
from statistics import fmean
import time

from .execution_control import ExecutionControl, NativeFastWallBudgetError, _NativeLayerSamplingCutoff
from .native_fast_window import (NativeLayerWindowSample, NativeWindowForecast, _checked_budget,
                                 _eligibility_reasons)
from .native_layer_sampling import (NativeLayerSamplingGuardError, _execute_window, _graph_facts, _layer_counts,
                                    drop_unsampled_request_records, layer_inventory,
                                    remaining_import_bytes)
from .planner import _compilation_scope
from .serving import RequestStatus, _OnlineRuntime, compile_serving_plan

HOLDOUT_SHIFT_NS = 10e6
SAME_CLASS_ADDRESS_ASSUMPTION = (
    "Unsampled layers take their class's sampled step: the DRAM row openings that differ between "
    "same-class layers because their buffer addresses straddle row boundaries differently are "
    "assumed negligible and are not bounded; representative-point full native runs check every "
    "unsampled layer against its class interval.")
_RUN_FIELDS = ("sample_id", "start_ns", "end_ns", "duration_ns", "boundary_origin_ns",
               "elapsed_from_boundary_ns", "task_count", "layer_completion_ns", "first_access_ns",
               "concurrent_imports", "wall_seconds", "physical_observations")


def _context_sensitive(metadata):
    name = str(metadata.get("operator_class", "")).lower()
    kind = str(metadata.get("model_operator_kind", "")).lower()
    event = str(metadata.get("event_kind", "")).lower()
    return ("attention" in name or "attention" in kind or name.startswith("kv_")
            or event.startswith("kv_") or bool(metadata.get("persistent_request_buffers"))
            or any(metadata.get(key) is not None for key in (
                "generic_kv_storage", "kv_operation", "kv_cache_component", "kv_operand",
                "kv_read_semantics", "native_kv_work", "source_f32_kv")))


def native_layer_shape(tasks):
    """Native operation geometry of one layer subgraph, without addresses.

    Typed LayerSpec semantics alone do not prove that two layers carry the
    same routed expert work or memory traffic, so the class also fixes the
    operator multiset, the balanced expert histogram and each owner's IO.
    """
    operators, experts, owners = defaultdict(int), set(), {}
    for task in tasks:
        md = task.metadata
        sensitive = _context_sensitive(md)
        operators[(str(md.get("operator_class", "")), str(md.get("model_operator_kind", "")), sensitive)] += 1
        if md.get("routing_model") == "deterministic_balanced_histogram" and "expert_index" in md:
            experts.add((md.get("rank"), md["expert_index"], md.get("expert_tokens"), md.get("routed_tokens")))
        for access in md.get("memory_accesses", ()):
            owner = access.get("physical_owner", md.get("physical_owner"))
            operation, count = access.get("operation"), access.get("byte_count")
            if operation not in {"read", "write"} or type(count) is not int:
                continue
            row = owners.setdefault(str(owner), {"read_count": 0, "write_count": 0, "read_bytes": 0,
                                                 "write_bytes": 0, "context_read_bytes": 0})
            row[operation + "_count"] += 1
            row[operation + "_bytes"] += count
            if sensitive and operation == "read":
                row["context_read_bytes"] += count
    return {"operator_signature": sorted([*key, count] for key, count in operators.items()),
            "balanced_expert_shape": sorted(experts, key=repr), "task_count": len(tasks),
            "owner_io_shape": owners}


class _IRDigest:
    """Per-task fingerprint of an IR: what the query/decode comparison reads.

    Dependency wiring, category, operator, layer, rank, expert and the KV/IO
    rows (owner, operation, bytes, generation-0 flag, KV flag) go into one
    64-bit hash per task, resource id/bytes/work units into a second, and the
    demands' service times are kept exactly. Nothing per-request is kept, so
    an IR can be freed as soon as it is digested.
    """

    __slots__ = ("graph", "shape", "demand_count", "service_ns")

    def __init__(self, stages):
        tasks = [task for stage in stages for task in stage.execution_tasks]
        positions = {task.task_id: index for index, task in enumerate(tasks)}
        categories = {}
        self.graph, self.shape = array("q"), array("q")
        self.demand_count, self.service_ns = array("i"), array("d")
        for task in tasks:
            md = task.metadata
            default_owner = md.get("physical_owner")
            accesses = tuple((access.get("physical_owner", default_owner), access.get("operation"),
                              access.get("byte_count"), access.get("allocation_generation", 0) == 0,
                              str(access.get("buffer_id", "")).startswith("kv:"))
                             for access in md.get("memory_accesses", ()))
            category = categories.get(task.category)
            if category is None:
                category = categories[task.category] = str(task.category)
            self.graph.append(hash((tuple(positions.get(parent, -1) for parent in task.dependencies),
                category, str(md.get("operator_class", "")), md.get("layer_id"), md.get("rank"),
                md.get("expert_index"), md.get("expert_tokens"), len(accesses), hash(accesses),
                sum(map(hash, accesses)))))
            self.shape.append(hash(tuple((row.resource_id, row.bytes_moved, row.work_units)
                                         for row in task.demands)))
            self.demand_count.append(len(task.demands))
            self.service_ns.extend(row.service_ns for row in task.demands)


def _query_decode_guard(runtime, cohort, stages, *, before_decode=None):
    """Prove the sampled final-query IR is a Decode step's IR.

    The seeds' final prompt queries are one-token steps at their real
    contexts. Lowering the same cohort as Decode must give the same graph,
    physical IO and resource work; nonphysical service differences are
    returned as a symmetric per-step guard. The first IR is digested before
    ``before_decode`` runs and the Decode IR is lowered, so at most one full
    IR is alive at a time.
    """
    from .serving import _lower_cost

    if not cohort.items or cohort.kind != "prefill" or any(item.token_count != 1 for item in cohort.items):
        raise NativeLayerSamplingGuardError("native_stationary_requires_final_prompt_query_cohort")
    items = tuple(replace(item, phase="decode", token_count=1, main_tokens=1, draft_tokens=0,
        verifier_tokens=1, committed_tokens=1, kv_append_tokens=1, kv_materialized_tokens=1,
        logit_tokens=1, completion_cursor=2) for item in cohort.items)
    projected = replace(cohort, cohort_id=cohort.cohort_id + ".native-stationary-decode",
                        kind="decode", items=items)
    left = _IRDigest(stages)
    if before_decode is not None:
        before_decode()
    cost = _lower_cost(runtime.lowerer, runtime.plan.scenario, projected)
    candidate, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
    if fallback or not candidate:
        raise NativeLayerSamplingGuardError("native_stationary_decode_ir_unavailable")
    candidate = runtime._with_gpu_controller_stages(candidate, projected)
    right = _IRDigest(candidate)
    del candidate, cost
    if len(left.graph) != len(right.graph):
        raise NativeLayerSamplingGuardError("native_stationary_query_decode_task_counts_differ")
    guard, offset = 0., 0
    for index in range(len(left.graph)):
        if left.graph[index] != right.graph[index]:
            raise NativeLayerSamplingGuardError("native_stationary_query_decode_graph_or_io_differs")
        count = left.demand_count[index]
        if count != right.demand_count[index] or left.shape[index] != right.shape[index]:
            raise NativeLayerSamplingGuardError("native_stationary_query_decode_work_shape_differs")
        guard += sum(abs(left.service_ns[offset + step] - right.service_ns[offset + step])
                     for step in range(count))
        offset += count
    return guard


def _sample_roles(inventory, layers):
    """Layer ids each sample run executes, or None when the stack is too short.

    Entry, exit and PD runs lie inside the calibration and holdout layers.
    """
    calibration, holdout = layers
    count = len(inventory)
    if calibration < 2 or holdout < 2:
        return None
    calibration, holdout = _layer_counts([row["class_key"] for row in inventory], calibration, holdout)
    if calibration + holdout > count:
        return None
    head = [row["layer_id"] for row in inventory[:calibration]]
    tail = [row["layer_id"] for row in inventory[count - holdout:]]
    return {"calibration": head, "holdout": tail, "entry": head[:1], "exit": tail[-1:],
            "pd_import": head[:2]}


def _component_groups(scenario):
    """Component id -> ep group, through the rank mapping and each rank's package."""
    packages = {component.component_id: component.package_id for component in scenario.hardware.components}
    groups, package_group = {}, {}
    for mapping in scenario.placement.parallel.rank_mapping:
        for component in (mapping.component_id, mapping.memory_component_id, mapping.cim_component_id):
            if component is not None:
                groups[component] = mapping.ep_rank
        package = packages.get(mapping.component_id)
        if package is not None:
            # A package shared by ranks of different groups names none of them.
            package_group[package] = (mapping.ep_rank
                                      if package_group.get(package, mapping.ep_rank) == mapping.ep_rank else None)
    for component, package in packages.items():
        if package_group.get(package) is not None:
            groups.setdefault(component, package_group[package])
    return groups


def _ep_groups(runtime, cohort):
    """Per-group request counts, mean contexts and rank sets of the first cohort.

    Layers are synchronous across ep groups, so the stationary composition is
    only exact when every group serves exactly B/ep requests. A cohort short
    of B is a KV capacity shortfall: the seeds fill the groups in turn, so it
    is uneven, and admission rejects it with the per-group holdings the
    search needs to jump to the native capacity.
    """
    scenario = runtime.plan.scenario
    parallel = scenario.placement.parallel
    degree, batch = parallel.ep_degree, runtime.plan.scheduler.max_num_seqs
    requests, contexts, group_of = [0] * degree, [0] * degree, {}
    for item in cohort.items:
        if not 0 <= item.ep_group < degree:
            raise NativeLayerSamplingGuardError("native_stationary_ep_groups_unbalanced")
        requests[item.ep_group] += 1
        contexts[item.ep_group] += item.context_tokens
        group_of[item.request_id] = item.ep_group
    if degree > 1 and sum(requests) == batch and any(count * degree != batch for count in requests):
        raise NativeLayerSamplingGuardError("native_stationary_ep_groups_unbalanced")
    return {"degree": degree, "requests": requests, "group_of": group_of,
            "mean_context_tokens": [total / count if count else 0. for total, count in zip(contexts, requests)],
            "rank_ids": [sorted(mapping.rank for mapping in parallel.rank_mapping if mapping.ep_rank == group)
                         for group in range(degree)],
            "component_group": _component_groups(scenario) if degree > 1 else {}}


def _import_pool_occupancy(runtime, degree):
    """Prepared imports (in flight or ready) held for each ep group."""
    held = (*runtime._external_pending, *runtime._external_ready)
    if degree == 1:
        return [len(held)]
    occupancy = [0] * degree
    for request_id in held:
        occupancy[runtime.states[request_id].ep_group] += 1
    return occupancy


def _destination_group(prepared, state):
    """The ep group an in-flight import writes its KV to."""
    if prepared["ep_groups"]["degree"] == 1:
        return 0
    known = prepared["ep_groups"]["component_group"]
    groups = {known.get(row["component_id"]) for row in state["destinations"]}
    if len(groups) != 1 or None in groups:
        raise NativeLayerSamplingGuardError("native_stationary_import_destination_group_ambiguous")
    return groups.pop()


def _task_metadata_views(specs, stages, cost):
    """Every task metadata dict that shares the cohort's per-request records."""
    for task in specs:
        yield task.metadata
    for stage in stages:
        for task in stage.execution_tasks:
            yield task.metadata
    for stage in cost.metadata.get("execution_stages") or ():
        for fact in stage.get("execution_tasks", ()):
            metadata = fact.get("metadata")
            if metadata is not None:
                yield metadata


def cohort_origin(runtime, request_ready_ns):
    """The native cohort boundary origin and its already-paid resource frontier."""
    source = runtime._execution_resource_kernel
    ready = source.physical_runtime.timeline.ready_ns
    executor = runtime._external_executor
    if executor is not None:
        # The PD source serves only external imports: its queued chunk reads
        # are import work no Decode task waits for, not an outstanding tail.
        owner = executor.services[str(runtime._external_ingress["source_component"])].physical_owner + ":"
        ready = {resource: value for resource, value in ready.items() if not resource.startswith(owner)}
    paid_frontier = max(ready.values(), default=float(runtime.now))
    # Each copy starts after every reservation already paid on the source,
    # so no outstanding tail is multiplied into the samples.
    origin = max(float(runtime.now), float(request_ready_ns), float(source.makespan_ns), paid_frontier)
    return origin, paid_frontier


def prefill_stage_ready(runtime, cohort, stages, request_ready_ns, *, host_ns=0.0):
    """Read the native Prefill stage constraints without advancing the boundary.

    This is the readiness calculation in ``_schedule_execution_stages`` for
    the supplied closed native stages, before they execute. Callers must
    supply the same stages and host duration; this does not create owner
    residency transfers or runtime extension stages.
    """
    if cohort.kind != "prefill" or any(item.phase != "prefill" for item in cohort.items):
        raise NativeLayerSamplingGuardError("native_prefill_stage_ready_requires_prefill")
    component_count = len({stage.component_id for stage in stages})
    metadata = runtime.plan.scenario.workload.metadata
    if not isinstance(metadata, Mapping):
        metadata = {}
    pipeline_prefill = (component_count > 1 and not bool(
        metadata.get("llama_cpp_single_request_prefill_serialized", False)))
    host_request_ready_ns = (max((max(runtime.states[item.request_id].spec.arrival_ns,
        runtime._runtime_origin_ns) for item in cohort.items), default=runtime.now)
        if pipeline_prefill else request_ready_ns)
    previous_device_available_ns = runtime._gpu_available_ns
    host_start_ns = max(runtime._host_available_ns, host_request_ready_ns)
    host_end_ns = host_start_ns + host_ns
    ready = {}
    for stage in stages:
        participates = runtime._component_participates_in_device_fence(stage.component_id)
        requires_fence = stage.requires_device_fence and participates
        device_fence_ready_ns = 0.0
        if participates:
            device_fence_ready_ns = runtime._last_serial_device_end_ns
            if requires_fence:
                device_fence_ready_ns = max(device_fence_ready_ns, previous_device_available_ns)
        cross_chunk_ready = max((runtime._request_stage_ready_ns.get(
            (request_id, stage.component_id, stage.stage_index),
            max(runtime.states[request_id].spec.arrival_ns, runtime._runtime_origin_ns))
            for request_id in stage.request_ids), default=host_end_ns)
        ready[stage.stage_id] = max(host_end_ns, cross_chunk_ready, device_fence_ready_ns)
    return ready


def cohort_layers(runtime, cohort, stages, request_ready_ns, namespace, *, ready_by_stage=None):
    """Task specs of an unexecuted cohort on the unchanged source, with its layer classes.

    The part every native layer-sample preparation shares: the boundary
    origin, the serial layer composition and each layer's native class.
    """
    source = runtime._execution_resource_kernel
    origin, paid_frontier = cohort_origin(runtime, request_ready_ns)
    native_ready = ready_by_stage is not None
    if native_ready:
        if (cohort.kind != "prefill" or set(ready_by_stage) != {stage.stage_id for stage in stages}
                or any(not math.isfinite(value) or value < 0 for value in ready_by_stage.values())):
            raise NativeLayerSamplingGuardError("native_prefill_stage_ready_invalid")
    else:
        ready_by_stage = {stage.stage_id: origin for stage in stages}
    specs, _sources = runtime._stage_task_specs(stages, ready_by_stage,
                                                namespace + cohort.cohort_id)
    # The live run places the cohort's shared resident buffers when it
    # indexes the cohort, before any task runs; every copy inherits them.
    source._place_shared_resident_allocations(specs)
    inventory = layer_inventory(runtime.plan.scenario, specs)
    composition, boundary = _graph_facts(specs, inventory)
    if not composition["qualified"]:
        raise NativeLayerSamplingGuardError("native_stationary_requires_serial_layer_composition")
    if set(boundary) - {"before_first_layer", "after_last_layer"} or composition["nonserial_non_layer_groups"]:
        raise NativeLayerSamplingGuardError("native_stationary_non_layer_work_not_a_serial_prefix_or_suffix")
    by_layer = defaultdict(list)
    for task in specs:
        if task.metadata.get("layer_id") is not None:
            by_layer[str(task.metadata["layer_id"])].append(task)
    for row in inventory:
        row["native_shape"] = native_layer_shape(by_layer[row["layer_id"]])
        row["class_key"] = json.dumps({"semantics": row["signature"], "native": row["native_shape"]},
                                      ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)
    return {"cohort": cohort, "source": source, "origin": origin, "paid_frontier": paid_frontier,
            **({"native_stage_ready": True} if native_ready else {}),
            "specs": specs, "inventory": inventory, "composition": composition, "boundary": boundary,
            "by_layer": {layer: [task.task_id for task in tasks] for layer, tasks in by_layer.items()},
            "ranks": {task.task_id: task.metadata["rank"] for task in specs
                      if type(task.metadata.get("rank")) is int},
            "context": {item.request_id: item.context_tokens for item in cohort.items}}


def prepare_stationary(runtime, cohort, stages, request_ready_ns, control, cost, layers):
    """One unchanged post-boundary source, its graph and layer classes.

    Runs only on the sampler's cutoff path: once the layer summaries and the
    first IR's digest exist, the per-request records of layers no sample
    window will run are dead data and are dropped from every view of the
    cohort before the Decode IR is lowered. ``cost`` is the cohort's lowering
    and ``layers`` the (calibration, holdout) layer counts.
    """
    started = time.perf_counter()
    control.raise_if_cancelled()
    groups = _ep_groups(runtime, cohort)
    prepared = cohort_layers(runtime, cohort, stages, request_ready_ns, "native-stationary-sample.")
    specs = prepared["specs"]
    roles = _sample_roles(prepared["inventory"], layers)
    # A too-short stack is rejected when sampling starts; nothing is cut.
    kept = None if roles is None else frozenset(layer for ids in roles.values() for layer in ids)

    def drop_dead_records():
        if kept is not None:
            drop_unsampled_request_records(_task_metadata_views(specs, stages, cost), kept)

    guard = _query_decode_guard(runtime, cohort, stages, before_decode=drop_dead_records)
    control.raise_if_cancelled()
    return {**prepared, "record_layers": kept, "ep_groups": groups,
            "import_pool_occupancy_by_group": _import_pool_occupancy(runtime, groups["degree"]),
            "query_decode_guard_ns": guard, "preparation_wall_seconds": time.perf_counter() - started}


def window_terms(prepared, layer_ids, *, entry, exit):
    """Terms a composition reads from a run, for its row-state bounds.

    The prefix with the run's first layer (``entry``), every later layer on
    its own, and the suffix after the last layer (``exit``).
    """
    by_layer, boundary = prepared["by_layer"], prepared["boundary"]
    keys = {identity: layer for layer in layer_ids[1:] for identity in by_layer[layer]}
    if entry:
        keys.update((identity, "entry") for identity in (
            *boundary.get("before_first_layer", []), *by_layer[layer_ids[0]]))
    if exit:
        keys.update((identity, "exit") for identity in boundary.get("after_last_layer", []))
    return keys


def run_window(prepared, control, records, role, layer_ids, groups, origin, sample_id, *,
               imports=None, row_terms=None):
    """Run ``groups`` and the tasks of ``layer_ids`` on a copy of the source; record the sample.

    The finished run is appended to ``records`` at once, so a wall limit in
    a later run keeps the physical samples already taken.
    """
    control.raise_if_cancelled()
    by_layer, cohort = prepared["by_layer"], prepared["cohort"]
    ids = [identity for group in groups for identity in group] + [
        identity for layer in layer_ids for identity in by_layer[layer]]
    specs, execution_origin = prepared["specs"], origin
    if prepared.get("native_stage_ready"):
        # Native Prefill host/controller preparation can overlap earlier
        # chunks. Preserve its absolute constraints instead of fencing it at
        # the device boundary; later-origin samples shift those same inputs.
        selected = frozenset(ids)
        shift = origin - prepared["origin"]
        specs = tuple(replace(task, earliest_start_ns=task.earliest_start_ns + shift)
                      for task in specs if task.task_id in selected)
        execution_origin = min(origin, min((task.earliest_start_ns for task in specs), default=origin))
    record = _execute_window(prepared["source"], specs, ids, origin_ns=execution_origin,
                             control=control, sample_id=sample_id, concurrent_imports=imports,
                             row_terms=row_terms,
                             **({"capture_prefill_terms": True} if prepared.get("native_stage_ready") else {}),
                             **({"tp_symmetry": prepared["tp_symmetry"]}
                                 if "tp_symmetry" in prepared else {}))
    row = {"layer_ids": layer_ids, "origin_ns": origin, **{key: record[key] for key in _RUN_FIELDS}}
    if prepared.get("native_stage_ready"):
        row["prefill_terms"] = record["prefill_terms"]
        row.update(boundary_origin_ns=origin, elapsed_from_boundary_ns=row["end_ns"] - origin,
                   execution_origin_ns=execution_origin)
    records.append(NativeLayerWindowSample(window_id=row["sample_id"], role=role,
        cohort_id=cohort.cohort_id, cohort_kind=cohort.kind, decode_step=0,
        layer_ids=tuple(layer_ids),
        rank_ids=tuple(sorted({prepared["ranks"][identity] for identity in ids
                               if identity in prepared["ranks"]})),
        context_tokens_by_request=prepared["context"], executed_task_count=row["task_count"],
        start_ns=row["start_ns"], end_ns=row["end_ns"], wall_seconds=row["wall_seconds"],
        physical_observations=row["physical_observations"]))
    control.report("native_stationary_sample", len(records), unit="work_units",
        message="原生层样本已保存，不计完整批次或输出词元",
        metadata={"sample_id": row["sample_id"], "role": role, "executed_tasks": row["task_count"]})
    return row


def sample_stationary(runtime, prepared, control, layers, records, *, holdout_shift_ns=HOLDOUT_SHIFT_NS,
                      pd_concurrency_envelope=True):
    """Run the stationary step samples; the source kernel never advances.

    Each finished run is appended to ``records`` at once, so a wall limit
    in a later run keeps the physical samples already taken.
    """
    started = time.perf_counter()
    calibration, holdout = layers
    inventory, boundary = prepared["inventory"], prepared["boundary"]
    count = len(inventory)
    roles = _sample_roles(inventory, layers)
    if roles is None:
        raise NativeLayerSamplingGuardError("native_stationary_stack_too_short_for_disjoint_runs")
    kept = prepared["record_layers"]
    if kept is not None and any(not kept.issuperset(ids) for ids in roles.values()):
        # Preparation dropped the per-request records of every other layer.
        raise NativeLayerSamplingGuardError("native_stationary_sample_layer_records_dropped")
    prefix, suffix = boundary.get("before_first_layer", []), boundary.get("after_last_layer", [])

    def run(role, layer_ids, groups, origin, tag, imports=None, entry=False, exit=False):
        return run_window(prepared, control, records, role, layer_ids, groups, origin, "stationary-" + tag,
                          imports=imports, row_terms=None if imports else window_terms(
                              prepared, layer_ids, entry=entry, exit=exit))

    first, second = prepared["origin"], prepared["origin"] + holdout_shift_ns
    runs = {"calibration": run("calibration", roles["calibration"], (prefix,), first, "calibration", entry=True),
            "holdout": run("holdout", roles["holdout"], (suffix,), second, "holdout", exit=True),
            # The cold ends again at the other origin.
            "entry_at_holdout_origin": run("entry", roles["entry"], (prefix,), second, "entry-second",
                                           entry=True),
            "exit_at_calibration_origin": run("exit", roles["exit"], (suffix,), first, "exit-second", exit=True)}
    executor, pending = runtime._external_executor, list(runtime._external_pending)
    imports = {"in_flight_request_ids": pending, "samples": []}
    if executor is not None and pending:
        states = {state.get("request_id"): state for state in executor._async_plan_states.values()
                  if not state.get("done") and "destinations" in state}
        # The two imports with the most remaining bytes, each alone; any
        # import outside a sampled set stays frozen on its copy. Every task of a
        # layer waits for all terminals of the layer before it, so the
        # prefix and first two layers end exactly as in the calibration run
        # without imports; later layers are not needed for the comparison.
        # With several ep groups the second import is the one with the most
        # remaining bytes outside the first one's group, when there is one.
        ordered = sorted((identity for identity in pending if identity in states),
                         key=lambda identity: (-remaining_import_bytes(states[identity]), identity))
        group_of = {identity: _destination_group(prepared, states[identity]) for identity in ordered}
        chosen = ordered[:1]
        if chosen:
            other = next((identity for identity in ordered[1:] if group_of[identity] != group_of[chosen[0]]),
                         None)
            chosen += [other] if other is not None else ordered[1:2]
        sets = [[identity] for identity in chosen]
        # A group starts all the imports landing in one step at the same
        # boundary, so they run together: for every busiest-group count k > 1
        # of the measured steps, two sets of k imports also run together.
        # The step stalls until such a set is written, so the set runs beside
        # all calibration layers: it finishes inside the window and its joint
        # extra is its measured later end, not a slope extrapolation.
        schedule = _group_refill_schedule(*refill_inputs(runtime, prepared))
        counts = schedule["busiest_landings"]
        if pd_concurrency_envelope:
            counts = (max(counts, default=1),)
        for count in counts:
            if count > 1:
                sets += _concurrent_sets(ordered, group_of, count)
        for identities in sets:
            rows = [states[identity] for identity in identities]
            imports["samples"].append({"request_ids": identities,
                "logical_bytes": sum(state["logical_bytes"] for state in rows),
                "remaining_bytes_at_cutoff": sum(remaining_import_bytes(state) for state in rows),
                "destination_group": group_of[identities[0]],
                "concurrent": run("pd_import", roles["pd_import" if len(identities) == 1 else "calibration"],
                                  (prefix,), first, "pd-" + "+".join(identities), (executor, identities))})
    return {"schema": "native-stationary-samples/v1", "runs": runs, "pd_imports": imports,
            "layers": {role: len(roles[role]) for role in ("calibration", "holdout")}, "layer_count": count,
            "holdout_shift_ns": holdout_shift_ns, "wall_seconds": time.perf_counter() - started,
            "physical_state_advanced": False, "source_kernel_modified": False}


def _concurrent_sets(ordered, group_of, count):
    """Two disjoint sets of ``count`` in-flight imports of one destination group each.

    Groups are taken in the order of their import with the most remaining
    bytes. The first set is the first group holding ``count``; the second,
    the next such group's, else the same group's next ``count``. Fewer sets
    when the pool does not hold them.
    """
    members = defaultdict(list)
    for identity in ordered:
        members[group_of[identity]].append(identity)
    groups = [group for group in members if len(members[group]) >= count]
    if not groups:
        return []
    sets = [members[groups[0]][:count]]
    if len(groups) > 1:
        sets.append(members[groups[1]][:count])
    elif len(members[groups[0]]) >= 2 * count:
        sets.append(members[groups[0]][count:2 * count])
    return sets


def _hull(values, widen):
    return min(values) - widen, max(values) + widen


def _interval(lower, upper):
    # The harmonic mean puts the estimate at equal relative distance from
    # both bounds: (upper-lower)/(upper+lower).
    estimate = 2. * lower * upper / (lower + upper)
    return {"estimate": estimate, "lower": lower, "upper": upper,
            "relative_error_budget": (upper - lower) / (upper + lower)}


def _within(row, target):
    value, lower, upper = (Fraction(row[key]) for key in ("estimate", "lower", "upper"))
    return value <= (1 + Fraction(target)) * lower and value >= (1 - Fraction(target)) * upper


def _completions(run, layer_ids):
    done = run["layer_completion_ns"] or {}
    times = [done.get(layer) for layer in layer_ids]
    if not all(isinstance(value, float) and math.isfinite(value) for value in times) or any(
            after <= before for before, after in zip(times, times[1:])):
        raise ValueError("native_stationary_layer_completions_invalid")
    return times


def _widened_hull(values, widths, widen):
    """Hull of observations each widened by its own bound, then by ``widen``."""
    return (min(value - width for value, width in zip(values, widths)) - widen,
            max(value + width for value, width in zip(values, widths)) + widen)


def compose_layers(inventory, runs, *, exit):
    """Lower/upper native service of the entry, every later layer and (``exit``) the suffix.

    Every observation is first widened by the boundary row-state bound of
    its run (``first_access_ns``): the DRAM banks a run reaches first start
    from the cohort boundary's row state, not the steady one. Without
    ``exit`` the service ends with the last layer and no exit runs are read.
    """
    classes = {row["layer_id"]: row["class_key"] for row in inventory}
    steps = defaultdict(lambda: defaultdict(list))
    widths = defaultdict(lambda: defaultdict(list))
    for role in ("calibration", "holdout"):
        run = runs[role]
        times = _completions(run, run["layer_ids"])
        for layer, before, after in zip(run["layer_ids"][1:], times, times[1:]):
            steps[classes[layer]][role].append(after - before)
            widths[classes[layer]][role].append(run["first_access_ns"].get(layer, 0.))
    calibration, holdout, entry_second = runs["calibration"], runs["holdout"], runs["entry_at_holdout_origin"]
    entry = [_completions(calibration, calibration["layer_ids"][:1])[0] - calibration["origin_ns"],
             _completions(entry_second, entry_second["layer_ids"])[0] - entry_second["origin_ns"]]
    entry_widths = [run["first_access_ns"].get("entry", 0.) for run in (calibration, entry_second)]
    exits, exit_widths = [0., 0.], [0., 0.]
    if exit:
        exit_second = runs["exit_at_calibration_origin"]
        exits = [holdout["end_ns"] - _completions(holdout, holdout["layer_ids"])[-1],
                 exit_second["end_ns"] - _completions(exit_second, exit_second["layer_ids"])[0]]
        exit_widths = [run["first_access_ns"].get("exit", 0.) for run in (holdout, exit_second)]
    if min(entry) <= 0 or min(exits) < 0:
        raise ValueError("native_stationary_step_ends_invalid")
    rows, intervals = [], {}
    for key in dict.fromkeys(classes.values()):
        cal, held = steps[key]["calibration"], steps[key]["holdout"]
        if not cal or not held:
            raise ValueError("native_stationary_runs_do_not_step_every_layer_class")
        drift = abs(fmean(cal) - fmean(held))
        row_widths = [*widths[key]["calibration"], *widths[key]["holdout"]]
        lo, hi = _widened_hull([*cal, *held], row_widths, drift)
        if lo <= 0:
            raise ValueError("native_stationary_layer_drift_exhausts_lower_bound")
        intervals[key] = (lo, hi)
        rows.append({"class_key": key, "layers": sum(value == key for value in classes.values()),
                     "calibration_steps_ns": cal, "holdout_steps_ns": held,
                     "calibration_holdout_drift_ns": drift, "first_access_ns": row_widths,
                     "step_ns": {"lower": lo, "upper": hi}})
    entry_lo, entry_hi = _widened_hull(entry, entry_widths, max(entry) - min(entry))
    exit_lo, exit_hi = _widened_hull(exits, exit_widths, max(exits) - min(exits))
    lower, upper = max(0., entry_lo) + max(0., exit_lo), entry_hi + exit_hi
    for row in inventory[1:]:
        lo, hi = intervals[row["class_key"]]
        lower += lo
        upper += hi
    parts = {"classes": rows, "entry_with_prefix_ns": entry, "entry_first_access_ns": entry_widths,
             "entry_interval_ns": [max(0., entry_lo), entry_hi]}
    if exit:
        parts.update(exit_suffix_ns=exits, exit_first_access_ns=exit_widths,
                     exit_interval_ns=[max(0., exit_lo), exit_hi])
    return lower, upper, parts


def _step_service(prepared, samples):
    """Lower/upper native service of one Decode step, and its parts."""
    lower, upper, parts = compose_layers(prepared["inventory"], samples["runs"], exit=True)
    guard = prepared["query_decode_guard_ns"]
    lower, upper = lower - guard, upper + guard
    if lower <= 0:
        raise ValueError("native_stationary_step_lower_bound_not_positive")
    return lower, upper, {**parts, "query_decode_guard_ns": guard}


def _import_extra(samples):
    """Joint quiet-work delay E(k) of k fresh imports run together, and their span T(k).

    Each sample continues one set of in-flight imports (k = 1: one import
    alone). A set finished inside the window gives E(k) directly as the
    later end of the same work, scaled from the bytes written in the window
    to the set's whole imports. Otherwise the post-entry layer slope gives
    the slowdown s and the set's observed write rate the span, with
    E(k) = T(k)*s/(1+s). Every sampled k needs two sets; returns
    {k: {"extra_ns": (lo, hi), "span_ns": (lo, hi)}} and the sample rows.
    """
    quiet = samples["runs"]["calibration"]
    rows = []
    for sample in samples["pd_imports"]["samples"]:
        identities, total, busy = sample["request_ids"], sample["logical_bytes"], sample["concurrent"]
        observed = busy["concurrent_imports"]
        written = sum(observed["written_bytes_by_request"][identity] for identity in identities)
        layers = busy["layer_ids"]
        if len(layers) < 2 or layers != quiet["layer_ids"][:len(layers)] or written <= 0:
            raise ValueError("native_stationary_PD_sample_invalid")
        # The busy run ends with its last layer; the quiet run reached the
        # same layer at its completion.
        q, c = _completions(quiet, layers), _completions(busy, layers)
        spans = [observed["import_spans_ns"].get(identity) for identity in identities]
        complete = all(span is not None and span[1] <= busy["end_ns"] for span in spans)
        rate = written / ((max(span[1] for span in spans) if complete else busy["end_ns"]) - busy["start_ns"])
        imported = total / rate
        row = {"request_ids": identities, "concurrent_imports": len(identities),
               "write_rate_bytes_per_ns": rate, "span_ns": imported,
               "written_bytes_in_window": written, "logical_bytes": total}
        if complete:
            row.update(mode="complete_import_in_window",
                       extra_ns=max(0., (c[-1] - q[-1]) * total / written))
        else:
            if min(observed["write_phases_by_request"][identity] for identity in identities) < 2:
                raise ValueError("native_stationary_PD_window_has_too_few_import_writes")
            slowdown = (c[-1] - c[0]) / (q[-1] - q[0]) - 1.
            row.update(mode="post_entry_layer_slope", slowdown=slowdown,
                       extra_ns=max(0., imported * slowdown / (1. + slowdown)))
        rows.append(row)
    terms = {}
    for count in sorted({row["concurrent_imports"] for row in rows}):
        sets = [row for row in rows if row["concurrent_imports"] == count]
        if len(sets) < 2:
            continue
        extras, spans = [row["extra_ns"] for row in sets], [row["span_ns"] for row in sets]
        extra_lo, extra_hi = _hull(extras, max(extras) - min(extras))
        span_lo, span_hi = _hull(spans, max(spans) - min(spans))
        terms[count] = {"extra_ns": (max(0., extra_lo), extra_hi), "span_ns": (max(0., span_lo), span_hi)}
    if 1 not in terms:
        raise ValueError("native_stationary_requires_two_in_flight_PD_imports")
    return terms, rows


def _monotone_import_envelope(terms, counts):
    """Bound unsampled intermediate k by measured k=1 and k_max endpoints.

    Each endpoint must already have two samples. No endpoint is invented,
    no extrapolation is made, and endpoint measurements are left unchanged.
    These conditional bounds assume both E(k) and T(k) are nondecreasing.
    """
    maximum = max(counts, default=1)
    missing = sorted(count for count in counts if 1 < count < maximum and count not in terms)
    if not missing or 1 not in terms or maximum not in terms:
        return terms
    envelope = {key: (terms[1][key][0], terms[maximum][key][1])
                for key in ("extra_ns", "span_ns")}
    if any(lower > upper for lower, upper in envelope.values()):
        raise ValueError("native_stationary_PD_monotonicity_inconsistent")
    return {count: terms[count] if count in terms else dict(envelope)
            for count in sorted(set(terms) | set(missing))}


def _refill_schedule(ages, output, pool, warmup, measured):
    """The continuous-refill schedule in step units.

    Step 1 is the seeds' final-prompt cohort; Decode steps count from step
    2, so the window holds steps warmup+2 .. warmup+measured+1. Boundary k
    ends step k. A slot freed at boundary k is refilled for step k+1 from
    the ready pool in request order, which is topped up at that boundary;
    the first pool starts importing at boundary 0. For each fresh request
    complete inside the window, returns the steps from its import start to
    its first token; also the most admissions at one boundary and the
    fewest steps between an admitted import's start and its admission.
    """
    finishes = defaultdict(int)
    for age in ages:
        finishes[output - age] += 1
    starts = [0] * pool
    first, last = warmup + 2, warmup + measured + 1
    waits, most, lag = [], 0, math.inf
    for boundary in range(1, last):
        freed = finishes.pop(boundary, 0)
        most = max(most, freed)
        for _ in range(min(freed, len(starts))):
            start = starts.pop(0)
            lag = min(lag, boundary - start)
            finishes[boundary + output] += 1
            if boundary + 1 >= first and boundary + output <= last:
                waits.append(boundary + 1 - start)
            starts.append(boundary)
    return waits, most, lag


def _refill_landings(ages, output, pool, last):
    """Requests landing in each step: the slots freed at the previous boundary
    that the pool can refill, following ``_refill_schedule``'s slot rules."""
    finishes = defaultdict(int)
    for age in ages:
        finishes[output - age] += 1
    landed = {}
    for boundary in range(1, last):
        refilled = min(finishes.pop(boundary, 0), pool)
        if refilled:
            landed[boundary + 1] = refilled
            finishes[boundary + output] += refilled
    return landed


def _group_refill_schedule(ages_by_group, output, pool, warmup, measured):
    """Per-ep-group continuous refill: a freed slot is refilled from its own
    group's pool of ``pool`` imports.

    Waits are aggregated over the groups; ``most`` is the largest per-group
    admission count at one boundary, ``lag`` the smallest import lead. ``rho``
    is the busiest-group share of landings over the measured steps: layers
    are synchronous across groups, so a step is delayed by the group with the
    most landing imports, ``sum_s max_g k_g(s) / sum_s sum_g k_g(s)``. It is
    exactly 1 for a single group.

    The imports landing in step s all start at the boundary before it, so a
    group's k_g(s) imports run together. ``busiest_landings`` counts the
    measured steps by their busiest group's k = max_g k_g(s) (k > 0), and
    ``fewest_landing_steps`` is the shortest distance between two landing
    steps of one group: the imports started at one boundary run alone only
    while they finish within that many steps.
    """
    first, last = warmup + 2, warmup + measured + 1
    rows, waits, landings = [], [], []
    for ages in ages_by_group:
        group_waits, most, lag = _refill_schedule(ages, output, pool, warmup, measured)
        waits += group_waits
        landings.append(_refill_landings(ages, output, pool, last))
        rows.append({"requests": len(ages), "pool": pool, "most_admissions_per_step": most,
                     "fewest_import_steps": lag, "complete_fresh_requests": len(group_waits)})
    peak = total = 0
    busiest = defaultdict(int)
    for step in range(first, last + 1):
        counts = [landed.get(step, 0) for landed in landings]
        peak += max(counts)
        total += sum(counts)
        if max(counts):
            busiest[max(counts)] += 1
    gap = min((after - before for landed in landings
               for before, after in zip(sorted(landed), sorted(landed)[1:])), default=math.inf)
    return {"waits": waits, "most": max(row["most_admissions_per_step"] for row in rows),
            "lag": min(row["fewest_import_steps"] for row in rows), "groups": rows,
            "rho": peak / total if total else 1., "peak_group_landings": peak, "total_landings": total,
            "busiest_landings": dict(sorted(busiest.items())), "fewest_landing_steps": gap}


def _required_group_pool(ages_by_group, output, pool, warmup, measured, lead):
    """Smallest per-group pool above ``pool`` (found too small) whose refill keeps every
    admission ``lead`` steps behind its import start.

    ``lead`` is the measured import span upper bound in lower-bound steps,
    plus one step so the re-sampled span and step of the re-run still fit.
    A larger pool consumes each prepared import later and admits no more per
    boundary, so both checks are monotone in the pool and a bisection finds
    the smallest. None when no pool can: the first refills take imports that
    started at boundary 0, whose lead the seeds' ages fix.
    """
    def fits(size):
        schedule = _group_refill_schedule(ages_by_group, output, size, warmup, measured)
        return schedule["most"] <= size and schedule["lag"] >= lead

    # One prepared import per admission in the window is the most a pool
    # can use; beyond it the lead no longer grows.
    high = max(len(ages) for ages in ages_by_group) * ((warmup + measured + 1) // output + 2)
    if high <= pool or not fits(high):
        return None
    low = pool
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            high = middle
        else:
            low = middle
    return high


def kv_requests_held(runtime, prepared):
    """KV reservations each EP group holds at the cutoff: first-cohort seeds plus prepared imports.

    Every seed and import reserves the same max-live KV, so when KV refused
    one more these are the groups' capacities in requests.
    """
    held = list(prepared["import_pool_occupancy_by_group"])
    for request_id in prepared["context"]:
        held[0 if len(held) == 1 else runtime.states[request_id].ep_group] += 1
    return held


def _preloaded_seed_admission(runtime, diagnostics):
    """Reject a proven seed-capacity shortfall before lowering any cohort.

    A future arrival, non-preloaded ingress, or a different seed/output
    contract must still follow ordinary admission and the existing guards.
    The caller runs this check once, immediately after the first preload.
    """
    if not runtime._external_ingress.get("preload_warmup_seed_kv", False):
        return
    batch = runtime.plan.scheduler.max_num_seqs
    if runtime._external_preloaded_requests >= batch:
        return
    seeds, outputs = [], set()
    for state in runtime._state_values:
        if state.spec.metadata.get("warmup_seed", False):
            seeds.append(state)
        else:
            outputs.add(state.spec.output_tokens)
    if (len(seeds) != batch or len(outputs) != 1
            or any(state.spec.arrival_ns > runtime.now for state in seeds)):
        return
    held = [0] * runtime._ep_degree
    for state in seeds:
        if state.spec.request_id not in runtime._external_initialized:
            # Pre-admission rejection or another state is not proof that
            # preload attempted allocation and found insufficient capacity.
            if state.status != RequestStatus.WAITING:
                return
            continue
        group = state.ep_group
        # An invalid assignment is not evidence of a capacity limit; leave
        # it to the existing cohort/group guard rather than relabeling it.
        if type(group) is not int or not 0 <= group < len(held):
            return
        held[group] += 1
    diagnostics.update(physical_admission_cutoff_ns=runtime.now,
                       kv_requests_held_by_group=held)
    raise NativeLayerSamplingGuardError("native_stationary_first_cohort_not_a_full_batch")


def admission_reasons(runtime, prepared):
    """Why the prepared boundary is not one full stationary batch; checked before any sample runs."""
    batch = runtime.plan.scheduler.max_num_seqs
    group_pool = runtime._external_ingress.get("max_inflight_requests", 2) // prepared["ep_groups"]["degree"]
    seeds = sum(1 for state in runtime._state_values if state.spec.metadata.get("warmup_seed", False))
    outputs = {state.spec.output_tokens for state in runtime._state_values
               if not state.spec.metadata.get("warmup_seed", False)}
    if len(outputs) != 1 or seeds != batch:
        return ["native_stationary_requires_one_fresh_output_length_and_a_seed_per_slot"]
    if len(prepared["context"]) != batch:
        # KV capacity did not admit a full batch of seeds.
        return ["native_stationary_first_cohort_not_a_full_batch"]
    if min(prepared["import_pool_occupancy_by_group"]) < group_pool:
        # KV capacity holds fewer prepared imports than the pool.
        return ["native_stationary_import_pool_not_full_at_cutoff"]
    return []


def refill_inputs(runtime, prepared):
    """Seed ages per ep group, the fresh output length, the per-group pool and
    the warmup/measured steps of an admitted stationary boundary's refill, in
    ``_group_refill_schedule``'s argument order."""
    ingress = runtime._external_ingress
    groups = prepared["ep_groups"]
    degree = groups["degree"]
    (output,) = {state.spec.output_tokens for state in runtime._state_values
                 if not state.spec.metadata.get("warmup_seed", False)}
    ages_by_group = [[] for _ in range(degree)]
    for state in runtime._state_values:
        spec = state.spec
        if not spec.metadata.get("warmup_seed", False):
            continue
        owner = 0 if degree == 1 else groups["group_of"].get(spec.request_id)
        if owner is None:
            raise NativeLayerSamplingGuardError("native_stationary_ep_groups_unbalanced")
        ages_by_group[owner].append(output - spec.output_tokens)
    return (ages_by_group, output, ingress.get("max_inflight_requests", 2) // degree,
            ingress["warmup_decode_cohorts"], ingress["measurement_decode_cohorts"])


def evaluate_stationary(runtime, prepared, samples, options):
    """Compose the stationary window metrics and their error budget."""
    plan = runtime.plan
    target = options.target_relative_error
    limit = target / (2 + target)
    batch, pool = plan.scheduler.max_num_seqs, runtime._external_ingress.get("max_inflight_requests", 2)
    groups = prepared["ep_groups"]
    degree = groups["degree"]
    result = {"accepted": False, "reasons": admission_reasons(runtime, prepared), "metrics": {},
              "metric_bounds": {}, "error_budget": {"target_relative_error": target,
                                                    "effective_independent_target": limit,
                                                    "conditional_only": True}}
    reasons = result["reasons"]
    if reasons:
        return result
    try:
        step_lo, step_hi, step = _step_service(prepared, samples)
        terms, imports = _import_extra(samples)
        ages_by_group, output, group_pool, warmup, measured = refill_inputs(runtime, prepared)
    except ValueError as error:
        reasons.append(str(error))
        return result
    schedule = _group_refill_schedule(ages_by_group, output, group_pool, warmup, measured)
    waits, most, lag, rho = schedule["waits"], schedule["most"], schedule["lag"], schedule["rho"]
    busiest, total = schedule["busiest_landings"], schedule["total_landings"]
    enveloped = []
    if options.pd_concurrency_envelope:
        measured_counts = set(terms)
        try:
            terms = _monotone_import_envelope(terms, busiest)
        except ValueError as error:
            reasons.append(str(error))
            return result
        enveloped = sorted(set(terms) - measured_counts)
    if set(busiest) - set(terms):
        reasons.append("native_stationary_concurrent_PD_imports_not_sampled")
        return result
    # Each running request finishes after ``output`` steps, so B/output
    # imports land per step. The k imports a group starts at one boundary
    # run together and add their joint E(k) to the next step; layers are
    # synchronous across ep groups, so a step waits for its busiest group.
    # With n_k measured steps whose busiest group lands k of all ``total``
    # landings, every step grows by B/output * sum_k E(k) n_k/total; with
    # single imports only that is B*e*rho/output.
    shares = {count: steps / total for count, steps in busiest.items()} if total else {1: 1.}
    tpot_lo = step_lo + sum(batch * terms[count]["extra_ns"][0] * share / output for count, share in shares.items())
    tpot_hi = step_hi + sum(batch * terms[count]["extra_ns"][1] * share / output for count, share in shares.items())
    span_lo = min(terms[count]["span_ns"][0] for count in shares)
    span_hi = max(terms[count]["span_ns"][1] for count in shares)
    gap = schedule["fewest_landing_steps"]
    if span_hi > gap * tpot_lo:
        # Imports a group started at one boundary would still run when it
        # starts the next ones: no sampled E(k) covers that overlap.
        reasons.append("native_stationary_import_concurrency_exceeds_model")
    if most > group_pool or lag * tpot_lo < span_hi:
        # A freed slot would wait for an unfinished import. The pool this
        # measured span needs lets the runner re-run the batch once.
        reasons.append("native_stationary_refill_not_always_ready")
        required = _required_group_pool(ages_by_group, output, group_pool, warmup, measured,
                                        math.ceil(span_hi / tpot_lo) + 1)
        result["error_budget"]["import_pool_required"] = None if required is None else required * degree
    complete = len(waits)
    if not complete:
        reasons.append("native_stationary_window_holds_no_complete_fresh_request")
        return result
    parallel = plan.scenario.placement.parallel
    world = parallel.tp_degree * parallel.pp_degree * parallel.ep_degree
    wait = fmean(waits)
    bounds = {"mean_complete_request_tpot_ns": _interval(tpot_lo, tpot_hi),
              "measurement_duration_ns": _interval(measured * tpot_lo, measured * tpot_hi),
              "cluster_tokens_per_second": _interval(batch * 1e9 / tpot_hi, batch * 1e9 / tpot_lo),
              "mean_complete_request_ingress_ttft_ns": _interval(tpot_lo * wait, tpot_hi * wait)}
    if any(not _within(row, limit) for row in bounds.values()):
        reasons.append("native_stationary_independent_budget_exceeded")
    tokens = measured * batch
    result.update(metric_bounds=bounds, predicted_visible_output_tokens=(warmup + measured) * batch,
        window={"warmup_decode_cohorts": warmup, "requested_decode_cohorts": measured,
                "measurement_visible_output_tokens": tokens, "complete_fresh_requests": complete,
                "forecast_only": True},
        error_budget={**result["error_budget"], "step_service_ns": {"lower": step_lo, "upper": step_hi},
            "step": step, "pd_import_extra_ns": {"lower": terms[1]["extra_ns"][0], "upper": terms[1]["extra_ns"][1]},
            "pd_import_span_ns": {"lower": span_lo, "upper": span_hi}, "pd_imports": imports,
            "pd_import_terms": [{"concurrent_imports": count, "measured_steps": busiest.get(count, 0),
                                 "share_of_landings": shares.get(count, 0.),
                                 "extra_ns": {"lower": row["extra_ns"][0], "upper": row["extra_ns"][1]},
                                 "span_ns": {"lower": row["span_ns"][0], "upper": row["span_ns"][1]}}
                                for count, row in terms.items()],
            "import_pool": pool,
            "import_overlap": {"fewest_landing_steps": None if gap == math.inf else gap,
                               "span_upper_steps": span_hi / tpot_lo},
            "refill_schedule": {"most_admissions_per_step": most, "fewest_import_steps": lag,
                                "mean_ingress_wait_steps": wait, "groups": schedule["groups"]},
            "pd_import_group_factor": {"rho": rho, "peak_group_landings": schedule["peak_group_landings"],
                                       "total_landings": total, "busiest_landings": busiest},
            "ep_groups": {"degree": degree, "requests": groups["requests"],
                          "mean_context_tokens": groups["mean_context_tokens"],
                          "rank_ids": groups["rank_ids"], "pool_per_group": group_pool,
                          "import_pool_occupancy": prepared["import_pool_occupancy_by_group"]},
            "basis": "stationary_continuous_refill_step; calibration_holdout_layer_hull_widened_by_drift; "
                     "in_situ_step_ends_two_origins; boundary_first_access_row_state_bounds; "
                     "measured_joint_PD_import_extra_per_concurrent_count",
            "assumptions": [SAME_CLASS_ADDRESS_ASSUMPTION]})
    if enveloped:
        result["error_budget"]["pd_concurrency_envelope"] = {
            "sampled_endpoints": [1, max(busiest)], "enveloped_counts": enveloped,
            "basis": "E(k) and import span T(k) are nondecreasing with concurrent import count; "
                     "intermediate bounds use the k=1 lower and k_max upper endpoints"}
        result["error_budget"]["assumptions"].append(
            "Joint PD import extra E(k) and span T(k) are nondecreasing in k; "
            "intermediate concurrency counts use a conditional endpoint envelope, not measured windows.")
        result["error_budget"]["basis"] += "; monotone_PD_concurrency_endpoint_envelope"
    if not reasons:
        result.update(accepted=True, metrics={
            "cluster_tokens_per_second": bounds["cluster_tokens_per_second"]["estimate"],
            "tokens_per_second_per_gpu": bounds["cluster_tokens_per_second"]["estimate"] / world,
            "mean_complete_request_tpot_ns": bounds["mean_complete_request_tpot_ns"]["estimate"],
            "mean_complete_request_ingress_ttft_ns": bounds["mean_complete_request_ingress_ttft_ns"]["estimate"],
            "complete_fresh_requests": complete, "measurement_visible_output_tokens": tokens,
            "measurement_duration_ns": bounds["measurement_duration_ns"]["estimate"]})
    return result


def deadline_control(control, deadline):
    """``control`` with the sample wall deadline (None: none) added to its cancellation."""
    if type(control) is ExecutionControl and control.cancellation_callback is None:
        # Native file/deadline checks cannot edit a physical calendar; keep
        # them native so PD phases avoid the unknown-callback rollback path.
        deadlines = [value for value in (deadline, control.deadline_monotonic) if value is not None]
        return ExecutionControl(progress_callback=control.progress_callback,
            progress_queue=control.progress_queue, stop_request_path=control.stop_request_path,
            deadline_monotonic=min(deadlines, default=None))

    def cancel():
        control.raise_if_cancelled()
        if deadline is not None and time.perf_counter() >= deadline:
            raise NativeFastWallBudgetError("native layer samples exceeded the candidate time budget")
        return False

    return ExecutionControl(progress_callback=control.progress_callback,
                            progress_queue=control.progress_queue, cancellation_callback=cancel)


def simulate_native_stationary(scenario, *, batch_lowerer, execution_control=None,
                               residency_manager=None, execution_kernel=None, runtime_origin_ns=0.,
                               custom_lowerer=False):
    """Sample the first cohort boundary natively and forecast the window."""
    from .fast_simulation import FastSimulationOutcome

    options = scenario.simulation.native_window
    started = time.perf_counter()
    deadline = started + options.max_wall_seconds
    control = execution_control or ExecutionControl()
    diagnostics = {"method": "native_stationary_continuous_refill/v1", "requested_mode": "fast",
        "effective_mode": "forecast_rejected", "sampling_options": asdict(options),
        "target_relative_error": options.target_relative_error,
        "validation_status": "conditional_budget_not_full_window_validation",
        "accuracy_target_is_guarantee": False, "future_physical_state_advanced": False,
        "physical_runtime_resumable_from_forecast": False,
        "sampled_decode_cohorts": 0, "sampled_visible_output_tokens": 0,
        "predicted_decode_cohorts": 0, "rejection_reasons": [], "fallback_reasons": []}
    # Only the enabled concurrency approximation adds diagnostic fields;
    # the full-sampling result retains its established representation.
    if not options.pd_concurrency_envelope:
        diagnostics["sampling_options"].pop("pd_concurrency_envelope")

    sample_control = deadline_control(control, deadline)
    records, prepared, samples, runtime = [], {}, {}, None
    context = getattr(batch_lowerer, "_compilation_context", None)
    with _compilation_scope(scenario, context):
        sample_control.raise_if_cancelled()
        plan = compile_serving_plan(scenario)
        reasons = _eligibility_reasons(scenario, plan, custom_lowerer=custom_lowerer,
                                      shared_residency=residency_manager is not None)
        if reasons:
            diagnostics.update(rejection_reasons=list(reasons), fallback_reasons=list(reasons),
                               wall_seconds=time.perf_counter() - started)
            return FastSimulationOutcome(diagnostics=diagnostics)
        try:
            runtime = _OnlineRuntime(plan, batch_lowerer, execution_control=sample_control,
                residency_manager=residency_manager, execution_kernel=execution_kernel,
                runtime_origin_ns=runtime_origin_ns)

            layers = (options.calibration_layers, options.holdout_layers)

            def sampler(active, cohort, cost, stages, request_ready_ns):
                prepared.update(prepare_stationary(active, cohort, stages, request_ready_ns,
                                                   sample_control, cost, layers))
                raise _NativeLayerSamplingCutoff()

            runtime._native_layer_sampler = sampler
            runtime._native_preload_admission_guard = lambda active: _preloaded_seed_admission(active, diagnostics)
            try:
                runtime.run_until_decode_steps(1, strict_cohort_cutoff=True)
            except _NativeLayerSamplingCutoff:
                pass
            else:
                raise RuntimeError("native stationary sampling unexpectedly committed an ordinary cohort")
            prepared["tp_symmetry"] = (scenario, options.tp_rank_symmetry)
            diagnostics["physical_admission_cutoff_ns"] = runtime.now
            diagnostics["kv_requests_held_by_group"] = kv_requests_held(runtime, prepared)
            admission = admission_reasons(runtime, prepared)
            if admission:
                budget = {"accepted": False, "reasons": admission}
            else:
                samples = sample_stationary(runtime, prepared, sample_control, layers, records,
                    pd_concurrency_envelope=options.pd_concurrency_envelope)
                budget = _checked_budget(evaluate_stationary(runtime, prepared, samples, options))
        except NativeFastWallBudgetError:
            budget = {"accepted": False, "reasons": ["native_stationary_sample_wall_budget_exhausted"]}
        except NativeLayerSamplingGuardError as error:
            budget = {"accepted": False, "reasons": [str(error)]}
        layer_samples = tuple(records)
        if runtime is not None:
            diagnostics["physical_ingress_accounting"] = {**runtime._external_ingress_report(),
                "first_output_status": "not_committed_by_isolated_layer_samples"}
            diagnostics["physical_boundary_accounting"] = {
                "layer_composition": prepared.get("composition", {}),
                "sample_boundary_origin_ns": prepared.get("origin"),
                "source_pending_carry_upper_ns": (max(0., prepared["paid_frontier"] - float(runtime.now))
                                                  if prepared else None),
                "full_cohort_committed": False, "visible_output_tokens": 0}
        reasons = list(budget.get("reasons", ()))
        diagnostics.update(error_budget=budget.get("error_budget", {}),
            metric_bounds=budget.get("metric_bounds", {}),
            rejection_reasons=reasons, fallback_reasons=reasons,
            stationary_samples={key: samples.get(key) for key in (
                "layers", "layer_count", "holdout_shift_ns", "wall_seconds")} if samples else {},
            preparation_wall_seconds=prepared.get("preparation_wall_seconds"),
            physical_layer_window_count=len(layer_samples),
            physical_complete_cohort_count=0, physical_visible_output_token_count=0,
            wall_seconds=time.perf_counter() - started)
        expired = time.perf_counter() >= deadline
        try:
            control.raise_if_cancelled()
        except NativeFastWallBudgetError:
            expired = True
        if expired:
            budget = {**budget, "accepted": False}
            diagnostics.update(rejection_reasons=[*reasons, "native_total_candidate_wall_budget_exhausted"],
                fallback_reasons=[*reasons, "native_total_candidate_wall_budget_exhausted"])
        if budget.get("accepted") is not True:
            control.report("native_fast_rejected", len(layer_samples), unit="work_units",
                message="原生稳态样本已保留，窗口预测未通过预算")
            return FastSimulationOutcome(diagnostics=diagnostics, layer_window_samples=layer_samples)
        ingress = runtime._external_ingress
        count = ingress["warmup_decode_cohorts"] + ingress["measurement_decode_cohorts"]
        forecast = NativeWindowForecast(runtime.now, 0, count, 0,
            budget["predicted_visible_output_tokens"],
            {"warmup_decode_cohorts": ingress["warmup_decode_cohorts"],
             "measurement_decode_cohorts": ingress["measurement_decode_cohorts"]},
            budget["window"], budget["metrics"], budget["metric_bounds"],
            tuple(budget.get("request_forecasts", ())), budget["error_budget"])
        diagnostics.update(effective_mode="fast", predicted_decode_cohorts=count,
            predicted_visible_output_tokens=forecast.predicted_visible_output_tokens)
        control.report("native_layer_forecast_complete", len(layer_samples), unit="work_units",
            message="原生稳态样本和预测结果分别保存；没有提交完整批次或Token")
        return FastSimulationOutcome(diagnostics=diagnostics, native_window_forecast=forecast,
                                     layer_window_samples=layer_samples)


__all__ = ["native_layer_shape", "prepare_stationary", "sample_stationary", "evaluate_stationary",
           "simulate_native_stationary"]
