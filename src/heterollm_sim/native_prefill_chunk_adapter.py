"""Adapt native layer samples into a pure, conditional chunk composition DAG.

Prefix task starts and per-lane releases stay separate: a short demand must
not hold its resource until the whole task completes.  Layers are serial
sampled terms, not reconstructed physical task executions or controller
states.  No serving cursor, clock or physical kernel is changed here.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import fmean

from .native_layer_sampling import NativeLayerSamplingGuardError
from .native_stationary import SAME_CLASS_ADDRESS_ASSUMPTION, _completions, _widened_hull


PREFIX_SERVICE_ASSUMPTION = (
    "Prefix service and per-lane release intervals use the hull of calibration and "
    "second-origin native observations widened by their drift. Future boundary state "
    "and fixed lane/order representativeness remain conditional, not proven extrema.")
LAYER_COLLAPSE_ASSUMPTION = (
    "Native serial layer barriers are retained as sampled layer terms on one virtual "
    "device lane. Preparation keeps its native task DAG, same-stage barriers and device "
    "fences; unsampled per-task memory resource calendars are not reconstructed.")


def _guard(condition, reason):
    if not condition:
        raise NativeLayerSamplingGuardError(reason)


def _hull(values, widths=()):
    values = tuple(float(value) for value in values)
    widths = tuple(float(value) for value in widths) or (0.0,) * len(values)
    _guard(len(values) == len(widths) and all(math.isfinite(value) and value >= 0 for value in (*values, *widths)),
           "native_prefill_chunk_invalid_service_observation")
    drift = max(values) - min(values)
    return {"lower": max(0.0, min(value - width for value, width in zip(values, widths)) - drift),
            "upper": max(value + width for value, width in zip(values, widths)) + drift}


def _class_parts(prepared, runs, probe=None):
    """Experimental non-negative class intervals, without accepting a prompt.

    The cold and optional history-probe formulas retain their original upper
    bounds. Intersecting each interval with non-negative service permits a
    zero lower bound; the independent prompt budget is checked only after
    the full preparation/layer graph has been composed. No shared Decode
    composition or positive-service observation guard is changed.
    """
    classes = {row["layer_id"]: row["class_key"] for row in prepared["inventory"]}
    steps = defaultdict(lambda: defaultdict(list))
    widths = defaultdict(lambda: defaultdict(list))
    probe_layer_ids = defaultdict(list)
    roles = ["calibration", "holdout"]
    all_runs = dict(runs)
    if probe is not None:
        roles.append("boundary_history_probe")
        all_runs["boundary_history_probe"] = probe
    for role in roles:
        run = all_runs[role]
        layer_ids = run["layer_ids"]
        _guard(bool(layer_ids) and all(layer in classes for layer in layer_ids),
               "native_prefill_chunk_class_layer_unknown")
        times = _completions(run, layer_ids)
        for layer, before, after in zip(layer_ids[1:], times, times[1:]):
            width = (run.get("first_access_ns") or {}).get(layer, 0.0)
            _guard(type(width) in (int, float) and math.isfinite(width) and width >= 0,
                   "native_prefill_chunk_class_row_width_invalid")
            key = classes[layer]
            steps[key][role].append(after - before)
            widths[key][role].append(width)
            if role == "boundary_history_probe":
                probe_layer_ids[key].append(layer)
    rows = []
    for key in dict.fromkeys(classes.values()):
        cal, held = steps[key]["calibration"], steps[key]["holdout"]
        _guard(bool(cal) and bool(held), "native_stationary_runs_do_not_step_every_layer_class")
        cold_widths = [*widths[key]["calibration"], *widths[key]["holdout"]]
        drift = abs(fmean(cal) - fmean(held))
        raw_lower, upper = _widened_hull([*cal, *held], cold_widths, drift)
        row = {"class_key": key, "layers": sum(value == key for value in classes.values()),
               "calibration_steps_ns": cal, "holdout_steps_ns": held,
               "calibration_holdout_drift_ns": drift, "first_access_ns": cold_widths,
               "cold_raw_lower": raw_lower}
        if probe is not None:
            history = steps[key]["boundary_history_probe"]
            history_widths = widths[key]["boundary_history_probe"]
            observed, row_widths = [*cal, *held, *history], [*cold_widths, *history_widths]
            means = [fmean(cal), fmean(held)] + ([fmean(history)] if history else [])
            combined_drift = max(means) - min(means)
            probe_lower, probe_upper = _widened_hull(observed, row_widths, combined_drift)
            raw_lower, upper = min(raw_lower, probe_lower), max(upper, probe_upper)
            row.update(boundary_history_probe_steps_ns=history,
                boundary_history_probe_first_access_ns=history_widths,
                boundary_history_probe_layer_ids=probe_layer_ids[key],
                boundary_history_probe_run_mean_ns=fmean(history) if history else None,
                boundary_history_probe_combined_drift_ns=combined_drift,
                boundary_history_probe_raw_lower=probe_lower)
        row.update(raw_lower=raw_lower, lower_clamped_to_zero=raw_lower < 0.0,
                   zero_lower_bound=raw_lower <= 0.0,
                   step_ns={"lower": max(0.0, raw_lower), "upper": upper})
        rows.append(row)
    result = {"classes": rows, "local_lower_bound_policy": "intersect_with_nonnegative_service",
              "prompt_budget_evaluated": False}
    if probe is not None:
        from .native_prefill_chunk_probe import (SAMPLE_PIPELINE_ASSUMPTION,
            UNBOUNDED_HISTORY_REPRESENTATIVENESS_ASSUMPTION)
        result["assumptions"] = [SAMPLE_PIPELINE_ASSUMPTION, UNBOUNDED_HISTORY_REPRESENTATIVENESS_ASSUMPTION]
        result["boundary_history_probe"] = {"sample_id": probe.get("sample_id"),
            "origin_ns": probe["origin_ns"], "layer_ids": list(probe["layer_ids"]),
            "representativeness_bounded": False,
            "sample_pipeline": dict(probe.get("boundary_history_probe", {}))}
    return result


def build_prefill_chunk_terms(prepared, samples, chunk_index, *, frontier=None):
    """Return ``chunk/extra_edges/assumptions/frontier`` for the pure composer.

    ``prepared`` is the chunk-input producer's ordinary Prepared graph with
    stable stage keys/descriptors. ``samples`` is ``sample_prefill`` output
    with explicitly captured ``prefill_terms``.  ``frontier`` is the prior
    return value's immutable-by-convention frontier; it contains only graph
    node references, never runtime/controller states.
    """
    _guard(isinstance(chunk_index, int) and not isinstance(chunk_index, bool) and chunk_index >= 0,
           "native_prefill_chunk_invalid_index")
    runs = samples["runs"]
    calibration, second = runs["calibration"], runs["entry_at_holdout_origin"]
    prefix_calibration = runs.get("entry_at_calibration_origin", calibration)
    inventory = prepared["inventory"]
    _guard(bool(inventory), "native_prefill_chunk_has_no_layers")
    history_probe = samples.get("boundary_history_probe")
    parts = _class_parts(prepared, runs, history_probe)
    classes = {row["class_key"]: dict(row["step_ns"]) for row in parts["classes"]}
    first_layer = inventory[0]["layer_id"]
    first_services, first_widths = [], []
    for run in ((prefix_calibration, second) if history_probe is None else (prefix_calibration, second, history_probe)):
        start = run["prefill_terms"]["layer_start_ns"].get(first_layer)
        end = run["layer_completion_ns"].get(first_layer)
        _guard(start is not None and end is not None and end > start,
               "native_prefill_chunk_first_layer_service_not_observed")
        first_services.append(end - start)
        first_widths.append((run.get("first_access_ns") or {}).get("entry", 0.0))
    first_interval = _hull(first_services, first_widths)
    first_class = classes[inventory[0]["class_key"]]
    first_interval = {"lower": min(first_interval["lower"], first_class["lower"]),
                      "upper": max(first_interval["upper"], first_class["upper"])}

    chunk_id = prepared.get("chunk_id", "prefill-chunk-{:04d}".format(chunk_index))
    stage_keys = prepared["stage_key_by_id"]
    stage_descriptors = {row["stage_id"]: row for row in prepared["stage_descriptors"]}
    stage_name = {identity: "native-stage:" + json.dumps(key, separators=(",", ":"))
                  for identity, key in stage_keys.items()}
    specs = {task.task_id: task for task in prepared["specs"]}
    prefix_ids = frozenset(prepared["boundary"].get("before_first_layer", ()))
    observed = []
    for run in (prefix_calibration, second):
        rows = {row["node_id"]: row for row in run["prefill_terms"]["prefix_tasks"]
                if row["node_id"] in prefix_ids}
        _guard(set(rows) == prefix_ids, "native_prefill_chunk_prefix_tasks_not_observed")
        observed.append(rows)

    previous = frontier or {}
    resource_releases = dict(previous.get("resource_releases", {}))
    prior_last_layer = previous.get("last_layer_terminal")
    nodes, edges, prefix_starts, prefix_ends = [], [], {}, {}
    stage_roots = {}
    origin = float(prepared["origin"])

    def add(identity, stage, duration, parents=(), ready=None, resources=()):
        row = {"node_id": identity, "stage_id": stage, "duration_ns": duration,
               "dependencies": list(dict.fromkeys(parents)), "resource_ids": list(resources),
               "ready_ns": {"lower": origin, "upper": origin} if ready is None else ready}
        nodes.append(row)
        return row

    zero = {"lower": 0.0, "upper": 0.0}
    # Native completion order can differ from start order for independent
    # resources. Sort by starts before rebuilding each observed lane order.
    ordered_prefix = sorted(prefix_ids, key=lambda identity: (
        observed[0][identity]["start_ns"], observed[0][identity]["end_ns"], identity))
    for identity in ordered_prefix:
        rows = [group[identity] for group in observed]
        stage_id = rows[0]["stage_id"]
        _guard(stage_id in stage_name, "native_prefill_chunk_unknown_prefix_stage")
        _guard(all(row["stage_id"] == stage_id for row in rows),
               "native_prefill_chunk_prefix_stage_changed")
        start, work, end = "prefix-start:" + identity, "prefix-work:" + identity, "prefix-end:" + identity
        prefix_starts[identity], prefix_ends[identity] = start, end
        ready = float(rows[0]["earliest_start_ns"])
        start_row = add(start, stage_name[stage_id], zero, ready={"lower": ready, "upper": ready})
        stage_roots.setdefault(stage_id, []).append(start)
        add(work, stage_name[stage_id], _hull(row["duration_ns"] for row in rows), (start,))
        releases = []
        demand_maps = [{row["resource_id"]: float(row["service_ns"]) for row in item["demands"]}
                       for item in rows]
        _guard(set(demand_maps[0]) == set(demand_maps[1]),
               "native_prefill_chunk_prefix_resource_set_changed")
        for number, resource in enumerate(sorted(demand_maps[0])):
            if not any(group[resource] > 0 for group in demand_maps):
                continue
            lanes = [row["resource_lanes"].get(resource, 0) for row in rows]
            _guard(lanes[0] == lanes[1], "native_prefill_chunk_prefix_lane_assignment_changed")
            lane = (resource, lanes[0])
            release = "prefix-release:{}:{}".format(identity, number)
            add(release, stage_name[stage_id], _hull(group[resource] for group in demand_maps), (start,))
            previous_release = resource_releases.get(lane)
            if previous_release is not None:
                if previous_release[0] == chunk_id:
                    start_row["dependencies"].append(previous_release[1])
                else:
                    edges.append((previous_release, (chunk_id, start)))
            resource_releases[lane] = (chunk_id, release)
            releases.append(release)
        add(end, stage_name[stage_id], zero, (work, *releases))

    # Replace only local prefix dependencies with their task completion
    # nodes. Suffix observations in a full reference are deliberately ignored.
    node_by_id = {node["node_id"]: node for node in nodes}
    for identity, start in prefix_starts.items():
        dependencies = specs[identity].dependencies
        _guard(not (set(dependencies) - prefix_ids), "native_prefill_chunk_prefix_has_nonprefix_parent")
        node_by_id[start]["dependencies"].extend(prefix_ends[parent] for parent in dependencies)
    prefix_used = {parent for identity in prefix_ids for parent in specs[identity].dependencies if parent in prefix_ids}
    prefix_terminals = [prefix_ends[identity] for identity in prefix_ids - prefix_used]
    layer_intervals, previous_layer = {}, None
    for position, layer in enumerate(inventory):
        layer_id = layer["layer_id"]
        interval = first_interval if position == 0 else classes[layer["class_key"]]
        layer_intervals[layer_id] = dict(interval)
        native_stages = sorted({specs[identity].metadata["execution_stage_id"]
                                for identity in prepared["by_layer"][layer_id]})
        _guard(bool(native_stages), "native_prefill_chunk_layer_has_no_stage")
        before, after = [], []
        parents = prefix_terminals if previous_layer is None else [previous_layer]
        for number, stage_id in enumerate(native_stages):
            _guard(stage_id in stage_name, "native_prefill_chunk_unknown_layer_stage")
            start = "layer-stage-start:{}:{}".format(position, number)
            ready = float(prepared["stage_ready_ns"][stage_id])
            add(start, stage_name[stage_id], zero, parents, {"lower": ready, "upper": ready})
            stage_roots.setdefault(stage_id, []).append(start)
            before.append(start)
            after.append(("layer-stage-end:{}:{}".format(position, number), stage_id))
        body = "layer-service:" + layer_id
        add(body, "sampled-layer:" + layer_id, interval, before,
            resources=("native-prefill-device-serial-lane",))
        for end, stage_id in after:
            add(end, stage_name[stage_id], zero, (body,))
        # Body completion is the last native layer task completion. Stage
        # completion markers are zero-width and impose no extra service.
        previous_layer = body

    if prior_last_layer is not None:
        for stage_id, roots in stage_roots.items():
            descriptor = stage_descriptors[stage_id]
            if descriptor.get("participates_in_device_fence") and descriptor.get("requires_device_fence"):
                edges.extend((prior_last_layer, (chunk_id, root)) for root in roots)
    assumptions = list(dict.fromkeys((SAME_CLASS_ADDRESS_ASSUMPTION, PREFIX_SERVICE_ASSUMPTION,
        LAYER_COLLAPSE_ASSUMPTION, *([
            "Layer calibration windows declare their boundary independently of prefix execution; "
            "both in-situ prefix entries are sampled, while cache/allocator representativeness remains conditional."]
            if prepared.get("chunk_separate_prefix") else []),
        *(["PP Prefill layer classes canonicalize owner-local resource and expert-placement names; "
           "stage activation transfers remain explicit in the sampled task graph and are covered by "
           "the full-reference error check."]
          if prepared.get("pipeline_layer_class_normalized") else []),
        *parts.get("assumptions", ()),
        *prepared.get("chunk_input_boundary", {}).get("assumptions", ()))))
    return {"chunk": {"chunk_id": chunk_id, "nodes": nodes, "last_layer_node_id": previous_layer},
            "extra_edges": edges, "assumptions": assumptions,
            "frontier": {"resource_releases": resource_releases,
                         "last_layer_terminal": (chunk_id, previous_layer)},
            "layer_intervals": layer_intervals,
            "diagnostics": {"prefix_task_count": len(prefix_ids),
                "first_layer_service_observations_ns": first_services,
                "first_layer_entry_row_width_ns": first_widths,
                "sampled_class_parts": parts["classes"]}}


__all__ = ["build_prefill_chunk_terms", "PREFIX_SERVICE_ASSUMPTION", "LAYER_COLLAPSE_ASSUMPTION"]
