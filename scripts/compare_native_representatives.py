"""Step 8: check the stationary fast forecast against a full native run of the same scenario.

One representative point runs twice on the same scenario (real prompt length,
shortened output so the full window stays affordable): the fast stationary
forecast and the complete native detailed run. The fast intervals of
throughput, mean fresh-request TPOT and ingress TTFT are checked against the
full run, and every layer step of every full-run Decode cohort is checked
against the step interval of its layer class. Results go to one JSON file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_native_config import build_native_comparison_scenario
from heterollm_sim import native_stationary
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.reporting import run_scenario
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig

_COHORT_PREFIX = "serving.online."


def scenario_for(args):
    return build_native_comparison_scenario(args.model, args.hardware, args.workload,
        batch_size=args.batch, tp=args.tp, pp=1, ep=args.ep, output_override=args.output)


def fast_run(scenario, **layers):
    seen = {}
    original = native_stationary.prepare_stationary

    def prepare(runtime, *rest, **options):
        prepared = original(runtime, *rest, **options)
        seen["classes"] = {row["layer_id"]: row["class_key"] for row in prepared["inventory"]}
        seen["layers"] = [row["layer_id"] for row in prepared["inventory"]]
        return prepared

    native_stationary.prepare_stationary = prepare
    started = time.perf_counter()
    try:
        result = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
            native_window=NativeWindowFastConfig(max_wall_seconds=6 * 3600., **layers))), retention_policy="aggregate")
    finally:
        native_stationary.prepare_stationary = original
    budget = result.approximation.get("error_budget") or {}
    intervals = {row["class_key"]: (row["step_ns"]["lower"], row["step_ns"]["upper"])
                 for row in (budget.get("step") or {}).get("classes", ())}
    return {"wall_seconds": time.perf_counter() - started,
            "rejection_reasons": list(result.approximation.get("rejection_reasons", ())),
            "metric_bounds": result.approximation.get("metric_bounds") or {},
            "step_service_ns": budget.get("step_service_ns"),
            "entry_interval_ns": (budget.get("step") or {}).get("entry_interval_ns"),
            "exit_interval_ns": (budget.get("step") or {}).get("exit_interval_ns"),
            "pd_imports": budget.get("pd_imports"),
            "layers": seen.get("layers", []), "classes": seen.get("classes", {}),
            "class_intervals": intervals}


def detailed_run(scenario):
    """The full native run, with every task's layer and end time per cohort."""
    ends = defaultdict(dict)
    original = UnifiedEventKernel._drain_prevalidated_compiled

    def drain(kernel, specs, layout, *rest, event_observer=None, **options):
        def observe(event):
            if event_observer is not None:
                event_observer(event)
            identity, layer = event.task.task_id, event.task.metadata.get("layer_id")
            if layer is not None and identity.startswith(_COHORT_PREFIX):
                cohort = identity[len(_COHORT_PREFIX):].split(".", 1)[0]
                previous = ends[cohort].get(layer)
                ends[cohort][layer] = event.end_ns if previous is None else max(previous, event.end_ns)
        return original(kernel, specs, layout, *rest, event_observer=observe, **options)

    UnifiedEventKernel._drain_prevalidated_compiled = drain
    started = time.perf_counter()
    try:
        result = run_scenario(scenario, retention_policy="aggregate")
    finally:
        UnifiedEventKernel._drain_prevalidated_compiled = original
    return result, dict(ends), time.perf_counter() - started


def observed_metrics(scenario, serving):
    """The full window's metrics as the fast forecast defines them (native_toy)."""
    window = serving.decode_measurement
    (output,) = {row.output_tokens for row in scenario.workload.requests if not row.metadata["warmup_seed"]}
    seeds = {row.request_id for row in scenario.workload.requests if row.metadata["warmup_seed"]}
    imports = {event.request_id: event.timestamp_ns for event in serving.events
               if event.event_type == "external_kv_import_start"}
    tokens = defaultdict(list)
    for event in serving.events:
        if event.event_type == "tokens_committed":
            tokens[event.request_id].extend([event.timestamp_ns] * event.details["visible_tokens"])
    complete = {key: times for key, times in tokens.items() if key not in seeds and len(times) == output
                and window["start_ns"] < times[0] <= times[-1] <= window["end_ns"]}
    return {"mean_complete_request_tpot_ns": statistics.mean(
                (times[-1] - times[0]) / (output - 1) for times in complete.values()),
            "cluster_tokens_per_second": window["visible_output_tokens"] * 1e9 / (window["end_ns"] - window["start_ns"]),
            "mean_complete_request_ingress_ttft_ns": statistics.mean(
                times[0] - imports[key] for key, times in complete.items())}, len(complete), window


def layer_checks(fast, ends, window):
    """Every Decode cohort's layer steps against their class intervals."""
    layers, classes, intervals = fast["layers"], fast["classes"], fast["class_intervals"]
    rows, outside = [], []
    for cohort, done in sorted(ends.items()):
        if not all(layer in done for layer in layers):
            continue
        times = [done[layer] for layer in layers]
        last = times[-1]
        for layer, before, after in zip(layers[1:], times, times[1:]):
            lower, upper = intervals[classes[layer]]
            step = after - before
            row = {"cohort": cohort, "layer_id": layer, "step_ns": step, "lower_ns": lower, "upper_ns": upper,
                   "in_window": window["start_ns"] < last <= window["end_ns"]}
            rows.append(row)
            if not lower <= step <= upper:
                outside.append({**row, "excess_ns": step - upper if step > upper else step - lower})
    return rows, outside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3_405b")
    parser.add_argument("--hardware", default="HBM4")
    parser.add_argument("--workload", default="SHORT")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=int, default=8, help="output tokens; warmup out-1, measurement 2(out-1)")
    parser.add_argument("--calibration-layers", type=int)
    parser.add_argument("--holdout-layers", type=int)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    scenario = scenario_for(args)
    fast = fast_run(scenario, **{key: value for key, value in (("calibration_layers", args.calibration_layers),
                                                               ("holdout_layers", args.holdout_layers)) if value})
    result, ends, detailed_wall = detailed_run(scenario)
    observed, complete, window = observed_metrics(scenario, result.serving)
    metrics = {}
    for name, value in observed.items():
        bounds = fast["metric_bounds"].get(name)
        metrics[name] = {"full_native": value, "fast": bounds,
                         "inside": bool(bounds) and bounds["lower"] <= value <= bounds["upper"],
                         "estimate_relative_error": None if not bounds else bounds["estimate"] / value - 1}
    rows, outside = layer_checks(fast, ends, window)
    in_window = [row for row in rows if row["in_window"]]
    payload = {"point": vars(args) | {"result": str(args.result)},
               "fast": fast, "detailed_wall_seconds": detailed_wall,
               "complete_fresh_requests": complete,
               "window": {key: window.get(key) for key in ("start_ns", "end_ns", "measured_decode_cohorts", "batch_sizes")},
               "metrics": metrics,
               "layer_steps": {"checked": len(rows), "checked_in_window": len(in_window),
                               "outside": len(outside), "outside_in_window": sum(row["in_window"] for row in outside),
                               "worst": sorted(outside, key=lambda row: -abs(row["excess_ns"]))[:40]}}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("metrics", "layer_steps", "complete_fresh_requests",
                                                    "detailed_wall_seconds")}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
