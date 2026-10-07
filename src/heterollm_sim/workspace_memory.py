"""Capacity bounds from physical buffer lifetimes, without pricing DRAM.

A cut is valid only when its final task joins every prefix task and every
newly ready suffix task depends on it.  That task is a serial barrier regardless of device
timing.  Buffers within one region may all overlap; buffers spanning regions
remain live.  This is a conservative DAG bound, not a measured peak.
"""
from __future__ import annotations

from collections import defaultdict, OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from threading import RLock

_AUDIT_CACHE = OrderedDict()
_AUDIT_IN_FLIGHT = {}
_AUDIT_LOCK = RLock()


def physical_workspace_bound(tasks):
    """Return per-owner simultaneous transient bytes across serial DAG cuts."""
    from .event_kernel import UnifiedEventKernel

    positions = {task.task_id: index for index, task in enumerate(tasks)}
    children = [set() for _ in tasks]
    indegrees = [len(task.dependencies) for task in tasks]
    for index, task in enumerate(tasks):
        for dependency in task.dependencies:
            if dependency not in positions or positions[dependency] >= index:
                raise ValueError("workspace analysis requires a complete topologically ordered DAG")
            source = positions[dependency]
            children[source].add(index)
    # Independent roots can run ahead of an apparent barrier.  Keep them in
    # one region until their paths join, rather than treating list order as
    # execution order.
    ready = {index for index, count in enumerate(indegrees) if count == 0}
    region = 0
    regions = []
    cuts = []
    prefix_sinks = set()
    for index, task in enumerate(tasks):
        for dependency in tasks[index].dependencies:
            prefix_sinks.discard(positions[dependency])
        prefix_sinks.add(index)
        ready.remove(index)
        for child in children[index]:
            indegrees[child] -= 1
            if indegrees[child] == 0:
                ready.add(child)
        regions.append(region)
        if prefix_sinks == {index} and ready and ready <= children[index]:
            cuts.append(index)
            region += 1

    buffers = {}
    residents = {}
    aliases = {}
    for index, task in enumerate(tasks):
        if task.metadata.get("physical_memory_config") is None:
            continue
        default_owner = (task.metadata.get("physical_owner")
                         or task.metadata.get("stateful_l2", {}).get("memory_resource"))
        configurations = task.metadata.get("physical_memory_configs", {})
        for row in UnifiedEventKernel._physical_descriptor_rows(task):
            identity = row.get("buffer_id") or row.get("tensor_id")
            if not identity:
                continue
            owner = row.get("physical_owner") or default_owner
            if str(identity).startswith("@tasklocal"):
                identity = task.task_id + str(identity)[len("@tasklocal"):]
            generation = int(row.get("allocation_generation", row.get("generation", 0)))
            key = (str(owner), str(identity), generation)
            if row.get("alias_of"):
                target_id = str(row["alias_of"])
                if target_id.startswith("@tasklocal"):
                    target_id = task.task_id + target_id[len("@tasklocal"):]
                aliases[key] = (str(owner), target_id,
                    int(row.get("alias_generation", generation)))
            offset = int(row.get("offset_bytes", row.get("offset", 0)))
            accessed_extent = offset + int(row.get("byte_count", row.get("size_bytes", 0)))
            declared_extent = row.get("allocation_size_bytes", row.get("buffer_size_bytes"))
            extent = accessed_extent if declared_extent is None else int(declared_extent)
            if extent < accessed_extent:
                raise ValueError("physical workspace access exceeds its declared allocation")
            contract = task.metadata.get("stateful_l2")
            if declared_extent is None and contract is not None:
                line = int(contract["line_bytes"])
                extent = ((extent + line - 1) // line) * line
            config = configurations.get(owner, task.metadata["physical_memory_config"])
            alignment = int(config.get("burst_bytes", 64))
            extent = ((extent + alignment - 1) // alignment) * alignment
            registry = residents if generation == 0 else buffers
            previous = registry.get(key)
            if previous is None:
                registry[key] = [extent, regions[index], regions[index]]
            else:
                previous[0] = max(previous[0], extent)
                previous[2] = max(previous[2], regions[index])
    def root_of(key):
        seen = set()
        while key in aliases:
            if key in seen:
                raise ValueError("physical workspace aliases contain a cycle")
            seen.add(key)
            key = aliases[key]
        return key

    for alias in aliases:
        target = root_of(alias)
        view = (residents if alias[2] == 0 else buffers).pop(alias)
        if target[2] == 0:
            # A view of resident memory adds no transient allocation.
            continue
        if target not in buffers:
            raise ValueError("workspace alias requires its physical target")
        buffers[target][1] = min(buffers[target][1], view[1])
        buffers[target][2] = max(buffers[target][2], view[2])
    deltas = defaultdict(lambda: defaultdict(int))
    for (owner, _, _), (extent, first, last) in buffers.items():
        deltas[owner][first] += extent
        deltas[owner][last + 1] -= extent
    peaks = {}
    for owner, changes in deltas.items():
        live = peak = 0
        for index in sorted(changes):
            live += changes[index]
            peak = max(peak, live)
        peaks[owner] = peak
    return {"bytes_by_owner": peaks, "serial_region_count": region + 1,
        "resident_bytes_by_owner": {owner: sum(value[0] for key, value in residents.items() if key[0] == owner)
                                     for owner in {key[0] for key in residents}},
        "buffer_count": len(buffers), "task_count": len(tasks),
        "bound_kind": "dependency_barrier_lifetime_upper_bound"}


def device_workspace_reservation(scenario):
    """Reuse internally computed audits, excluding only nonsemantic UI status."""
    from .serde import stable_hash, to_primitive
    cache_input = to_primitive(scenario)
    # Validation clears these presentation flags before run-estimate. They
    # neither affect lowering nor replace the actual control-plane fingerprint.
    # Preserve every other field, including other UI metadata and saved decisions.
    ui = cache_input.get("placement", {}).get("metadata", {}).get("ui")
    if isinstance(ui, dict):
        ui.pop("mapping_stale", None)
        ui.pop("mapping_stale_reason", None)
    key = stable_hash(cache_input)
    with _AUDIT_LOCK:
        if key in _AUDIT_CACHE:
            _AUDIT_CACHE.move_to_end(key)
            return deepcopy(_AUDIT_CACHE[key])
        pending = _AUDIT_IN_FLIGHT.get(key)
        owns_calculation = pending is None
        if owns_calculation:
            pending = Future()
            _AUDIT_IN_FLIGHT[key] = pending
    if not owns_calculation:
        return deepcopy(pending.result())
    try:
        result = _device_workspace_reservation(scenario)
    except BaseException as exc:
        with _AUDIT_LOCK:
            _AUDIT_IN_FLIGHT.pop(key, None)
            pending.set_exception(exc)
        raise
    with _AUDIT_LOCK:
        _AUDIT_IN_FLIGHT.pop(key, None)
        _AUDIT_CACHE[key] = deepcopy(result)
        while len(_AUDIT_CACHE) > 32:
            _AUDIT_CACHE.popitem(last=False)
        pending.set_result(result)
    return deepcopy(result)


def _device_workspace_reservation(scenario):
    """Lower configured maximal ordinary and speculative cohorts for capacity.

    The first-fit tier policy originally omitted every activation.  Its
    densely packed expert tensors exposed this issue before dense presets.
    MTP includes the maximum configured verifier width and both acceptance
    endpoints.  The result remains a conservative graph bound, not a claim
    that arbitrary allocation sequences cannot fragment their workspace.
    """
    from . import planner
    from .serving import BatchCohort, BatchItem

    config = scenario.llama_cpp_config
    mtp = planner._mtp_configuration(scenario)
    requests = tuple(planner.materialize_requests(scenario))
    if not requests:
        return None
    largest_prompt = max(requests, key=lambda item: item.prompt_tokens)
    longest_context = max(requests, key=lambda item: item.prompt_tokens + item.output_tokens)
    rows = min(largest_prompt.prompt_tokens, config.ubatch, config.batch)
    history = max(0, largest_prompt.prompt_tokens - rows)
    examples = [BatchCohort("cohort-workspace-prefill", "prefill", 0.0, (
        BatchItem(largest_prompt.request_id, "prefill", rows, history,
            kv_append_tokens=rows, kv_materialized_tokens=rows, logit_tokens=1,
            completion_cursor=largest_prompt.prompt_tokens),))]
    if longest_context.output_tokens > 1:
        prior = longest_context.prompt_tokens + longest_context.output_tokens - 2
        examples.append(BatchCohort("cohort-workspace-decode", "decode", 0.0, (
            BatchItem(longest_context.request_id, "decode", 1, prior,
                kv_append_tokens=1, kv_materialized_tokens=1, logit_tokens=1,
                completion_cursor=longest_context.output_tokens),)))
        if mtp is not None:
            width = min(1 + mtp.candidate_tokens, longest_context.output_tokens - 1, config.batch)
            prior = longest_context.prompt_tokens + longest_context.output_tokens - 1 - width
            for accepted in sorted({1, width}):
                examples.append(BatchCohort("cohort-workspace-mtp-{}".format(accepted), "mtp", 0.0, (
                    BatchItem(longest_context.request_id, "mtp", width, prior,
                        proposed_tokens=width, expected_accepted_tokens=float(accepted),
                        kv_append_tokens=accepted, kv_materialized_tokens=width,
                        main_tokens=1, draft_tokens=width - 1, verifier_tokens=width,
                        committed_tokens=accepted, logit_tokens=width,
                        completion_cursor=longest_context.output_tokens),),
                    proposal_cost_scale=mtp.proposal_cost_scale))
    peaks = {}
    residents = {}
    audits = []
    with planner._compilation_scope(scenario):
        for example in examples:
            lowering = planner._lower_serving_cohort(scenario, example)
            tasks = planner._promote_physical_allocation_extents(lowering.schedule.tasks)
            audit = physical_workspace_bound(tasks)
            audit["phase"] = example.kind
            audits.append(audit)
            for owner, size in audit["bytes_by_owner"].items():
                # Multiple configured request slots can overlap.  Reserving
                # their individual complete DAG bounds is safe and explicit.
                peaks[owner] = max(peaks.get(owner, 0), size * config.parallel)
            for owner, size in audit["resident_bytes_by_owner"].items():
                residents[owner] = max(residents.get(owner, 0), size)
    owner_components = {}
    for component in scenario.hardware.components:
        memory = component.metadata.get("memory_service", {})
        from .ir import default_memory_resource_id
        owner = (memory.get("physical_owner") or component.metadata.get("physical_owner")
                 or default_memory_resource_id(component))
        owner_components[owner] = component.component_id
    unmapped = (set(peaks) | set(residents)) - set(owner_components)
    if unmapped:
        raise ValueError("physical workspace owners have no component mapping: " + ", ".join(sorted(unmapped)))
    return {"schema": "physical-workspace-reservation/v1",
        "scope": "configured_prefill_final_decode_and_maximum_mtp_acceptance_endpoints",
        "slot_multiplier": config.parallel,
        "bytes_by_component": {owner_components[owner]: size for owner, size in peaks.items()},
        "resident_physical_bytes_by_component": {owner_components[owner]: size for owner, size in residents.items()},
        "cohorts": audits}
