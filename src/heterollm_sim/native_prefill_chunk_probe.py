"""Opt-in short physical-history probes for experimental chunk sampling.

The probe executes a shortened prefix/head/tail sample pipeline on an
independent kernel. It emits only selected native Tasks, never physical
events for omitted layers. A sampled tail follows the head's real terminal
Tasks, so its controller and allocation history differs from a cold holdout.
This is a sensitivity observation, not a bound on all unsampled histories.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import math
from statistics import fmean

from .native_layer_sampling import NativeLayerSamplingGuardError
from .native_stationary import _completions, _widened_hull, run_window, window_terms


SAMPLE_PIPELINE_ASSUMPTION = (
    "The boundary-history probe is a shortened native sample pipeline: selected head layers precede "
    "selected tail layers through explicit real-Task dependencies. Omitted layers execute no Tasks "
    "and create no physical events; their allocation/controller state is not reconstructed.")
UNBOUNDED_HISTORY_REPRESENTATIVENESS_ASSUMPTION = (
    "Per-class cold and short-history observations, widened by their cross-run mean drift, are "
    "conditionally representative of unsampled internal allocation/address/controller histories. "
    "The probe does not establish a finite global representativeness bound or guarantee coverage.")


def _guard(condition, reason):
    if not condition:
        raise NativeLayerSamplingGuardError(reason)


def _short_pipeline(prepared, layer_ids):
    """Reconnect a missing layer interval on a separate sample-spec tuple."""
    inventory = [row["layer_id"] for row in prepared["inventory"]]
    positions = {layer: index for index, layer in enumerate(inventory)}
    specs = prepared["specs"]
    by_id = {task.task_id: task for task in specs}
    by_layer = prepared["by_layer"]
    prefix = frozenset(prepared["boundary"].get("before_first_layer", ()))
    selected = prefix | frozenset(identity for layer in layer_ids for identity in by_layer[layer])
    replacements, seams = {}, []
    for previous, current in zip(layer_ids, layer_ids[1:]):
        if positions[current] == positions[previous] + 1:
            continue
        previous_ids, current_ids = frozenset(by_layer[previous]), frozenset(by_layer[current])
        used = {parent for identity in previous_ids for parent in by_id[identity].dependencies
                if parent in previous_ids}
        terminals = tuple(sorted(previous_ids - used))
        _guard(bool(terminals), "native_prefill_chunk_history_probe_has_no_head_terminal")
        rebound = []
        for identity in sorted(current_ids):
            task = by_id[identity]
            omitted = tuple(parent for parent in task.dependencies if parent not in selected)
            # Reconnect only roots whose original incoming layer history is
            # omitted. All selected/prefix dependencies keep their order.
            if omitted and not any(parent in current_ids for parent in task.dependencies):
                dependencies = tuple(dict.fromkeys((*(parent for parent in task.dependencies
                                                     if parent not in omitted), *terminals)))
                replacements[identity] = replace(task, dependencies=dependencies)
                rebound.append({"task_id": identity, "omitted_parent_ids": list(omitted)})
        _guard(bool(rebound), "native_prefill_chunk_history_probe_tail_has_no_reconnectable_root")
        seams.append({"head_last_layer": previous, "tail_first_layer": current,
                      "head_terminal_ids": list(terminals), "reconnected_tail_roots": rebound})
    sample = dict(prepared)
    sample["specs"] = tuple(replacements.get(task.task_id, task) for task in specs)
    return sample, seams


def run_chunk_boundary_history_probe(prepared, samples, control, records, *, origin_ns=None):
    """Run prefix + up to four calibration and four holdout layers once.

    ``origin_ns`` defaults to the first sample origin. A caller may explicitly
    supply the holdout origin for a separate sensitivity experiment. The
    original Prepared and cold runs are unchanged; run_window appends this
    real selected-Task window to ``records``.
    """
    control.raise_if_cancelled()
    runs = samples["runs"]
    chosen = set(runs["calibration"]["layer_ids"][:4]) | set(runs["holdout"]["layer_ids"][-4:])
    layer_ids = [row["layer_id"] for row in prepared["inventory"] if row["layer_id"] in chosen]
    _guard(len(layer_ids) >= 2 and len(layer_ids) == len(chosen),
           "native_prefill_chunk_history_probe_layers_invalid")
    origin = prepared["origin"] if origin_ns is None else origin_ns
    _guard(type(origin) in (int, float) and math.isfinite(origin) and origin >= 0,
           "native_prefill_chunk_history_probe_origin_invalid")
    sample, seams = _short_pipeline(prepared, layer_ids)
    prefix = prepared["boundary"].get("before_first_layer", ())
    probe = run_window(sample, control, records, "chunk_boundary_history_probe", layer_ids,
        (prefix,), float(origin), "prefill-chunk-boundary-history-probe",
        row_terms=window_terms(sample, layer_ids, entry=True, exit=False))
    # Explicitly require the real observed selected-layer order. A broken
    # seam must fail instead of turning overlapping executions into steps.
    _completions(probe, layer_ids)
    probe["boundary_history_probe"] = {"experimental": True, "default_enabled": False,
        "sample_pipeline": "prefix_then_selected_head_then_selected_tail",
        "source_kernel_modified": False, "omitted_layers_physically_executed": False,
        "reconnected_seams": seams,
        "assumptions": [SAMPLE_PIPELINE_ASSUMPTION, UNBOUNDED_HISTORY_REPRESENTATIVENESS_ASSUMPTION]}
    return probe


def expand_class_intervals(prepared, samples, probe, parts):
    """Return a new parts dict with per-class cold/history step hulls.

    Each observation carries its own boundary row width. As in the original
    two-run class formula, the hull is widened by the range of run means;
    with two means this is exactly abs(mean(calibration)-mean(holdout)). The
    probe's first layer is an entry observation and is left to the adapter.
    No returned interval claims a proven bound on unsampled histories.
    """
    classes = {row["layer_id"]: row["class_key"] for row in prepared["inventory"]}
    probe_ids = probe["layer_ids"]
    _guard(bool(probe_ids) and all(layer in classes for layer in probe_ids),
           "native_prefill_chunk_history_probe_class_unknown")
    times = _completions(probe, probe_ids)
    steps, widths, layer_steps = defaultdict(list), defaultdict(list), defaultdict(list)
    for layer, before, after in zip(probe_ids[1:], times, times[1:]):
        width = (probe.get("first_access_ns") or {}).get(layer, 0.0)
        _guard(type(width) in (int, float) and math.isfinite(width) and width >= 0,
               "native_prefill_chunk_history_probe_row_width_invalid")
        key = classes[layer]
        steps[key].append(after - before)
        widths[key].append(width)
        layer_steps[key].append(layer)
    result = deepcopy(parts)
    _guard({row["class_key"] for row in result["classes"]} == set(classes.values()),
           "native_prefill_chunk_history_probe_parts_class_mismatch")
    for row in result["classes"]:
        key = row["class_key"]
        cal, held = row["calibration_steps_ns"], row["holdout_steps_ns"]
        original_widths = row["first_access_ns"]
        _guard(bool(cal) and bool(held) and len(original_widths) == len(cal) + len(held),
               "native_prefill_chunk_history_probe_cold_parts_invalid")
        observed, row_widths = [*cal, *held, *steps[key]], [*original_widths, *widths[key]]
        _guard(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in observed)
               and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in row_widths),
               "native_prefill_chunk_history_probe_observation_invalid")
        means = [fmean(cal), fmean(held)] + ([fmean(steps[key])] if steps[key] else [])
        drift = max(means) - min(means)
        lower, upper = _widened_hull(observed, row_widths, drift)
        _guard(lower > 0, "native_prefill_chunk_history_probe_drift_exhausts_lower_bound")
        row["step_ns"] = {"lower": min(row["step_ns"]["lower"], lower),
                          "upper": max(row["step_ns"]["upper"], upper)}
        row["boundary_history_probe_steps_ns"] = steps[key]
        row["boundary_history_probe_first_access_ns"] = widths[key]
        row["boundary_history_probe_layer_ids"] = layer_steps[key]
        row["boundary_history_probe_run_mean_ns"] = fmean(steps[key]) if steps[key] else None
        row["boundary_history_probe_combined_drift_ns"] = drift
    result["boundary_history_probe"] = {"sample_id": probe.get("sample_id"),
        "origin_ns": probe["origin_ns"], "layer_ids": list(probe_ids),
        "representativeness_bounded": False,
        "sample_pipeline": deepcopy(probe.get("boundary_history_probe", {}))}
    result["assumptions"] = list(dict.fromkeys((*result.get("assumptions", ()),
        SAMPLE_PIPELINE_ASSUMPTION, UNBOUNDED_HISTORY_REPRESENTATIVENESS_ASSUMPTION)))
    return result


__all__ = ["run_chunk_boundary_history_probe", "expand_class_intervals",
           "SAMPLE_PIPELINE_ASSUMPTION", "UNBOUNDED_HISTORY_REPRESENTATIVENESS_ASSUMPTION"]
