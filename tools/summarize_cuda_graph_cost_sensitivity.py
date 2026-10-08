"""Summarize measured host-service variation in an already completed report.

This tool reads the report's realized lifecycle events and cost registry only.
It does not run native code, repeat measurements, change simulation parameters,
or infer end-to-end latency bounds from host service times.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


PHASES = (
    "ordinary_submit", "capture", "instantiate", "update", "update_failure",
    "first_launch_submit", "replay_submit", "destroy_exec", "destroy_graph",
)
SUMMARY_FIELDS = ("median_ns", "minimum_ns", "maximum_ns")
TOTAL_FIELDS = (
    "sum_of_event_medians_ns", "sum_of_event_minima_ns", "sum_of_event_maxima_ns",
)
INTERPRETATION = (
    "These are additive CPU host-service amounts, not elapsed simulation time.",
    "Each lifecycle occurrence contributes once. An ordinary-submit occurrence "
    "is the whole measured enqueue loop, even when split across many host tasks.",
    "Sums of observed per-event minima/maxima are sensitivity scenarios, not "
    "confidence intervals or observations of a complete run.",
    "CPU submission can overlap GPU execution. End-to-end sensitivity requires "
    "rescheduling the same dependency graph and is not computed here.",
    "Measurement repeat scatter does not quantify synthetic-to-real kernel "
    "bias or prediction error on unseen models.",
)


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_number(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (not allow_zero and value == 0)):
        raise ValueError(f"{name} must be a finite {'non-negative' if allow_zero else 'positive'} number")
    return value


def _measurement(value):
    value = _mapping(value, "independent phase measurement")
    result = {key: _positive_number(value.get(key), key) for key in SUMMARY_FIELDS}
    if not result["minimum_ns"] <= result["median_ns"] <= result["maximum_ns"]:
        raise ValueError("phase measurement requires minimum <= median <= maximum")
    result["repeat_mad_ns"] = _positive_number(value.get("repeat_mad_ns"), "repeat_mad_ns", allow_zero=True)
    count = value.get("repeat_count")
    if type(count) is not int or count < 3:
        raise ValueError("independent phase measurement requires at least three repeats")
    result["repeat_count"] = count
    return result


def _structure(value, registry):
    value = _mapping(value, "phase structure binding")
    identity = value.get("structure_id")
    count = value.get("node_count")
    if (not isinstance(identity, str) or identity not in registry
            or type(count) is not int or count <= 0):
        raise ValueError("phase structure has no matching report registry entry")
    registered = _mapping(registry[identity], "registered structure")
    if registered.get("node_count") != count or not isinstance(registered.get("topology"), str):
        raise ValueError("phase structure differs from its report registry entry")
    result = {"structure_id": identity, "node_count": count}
    for role in ("previous", "current"):
        if role in value:
            result[role] = _structure(value[role], registry)
    return result


def _empty_totals():
    return {key: 0.0 for key in TOTAL_FIELDS}


def _add_measurement(totals, measurement, count):
    for destination, source in zip(TOTAL_FIELDS, SUMMARY_FIELDS):
        totals[destination] += count * measurement[source]


def summarize_completed_report(document, *, source=None):
    """Count realized lifecycle occurrences, not repeated task audit references."""
    document = _mapping(document, "input")
    job_id = None
    if "report" in document:
        if document.get("status") != "completed":
            raise ValueError("a completed simulation job is required")
        job_id = document.get("job_id")
        report = _mapping(document["report"], "completed job report")
    else:
        report = document
    summary = _mapping(report.get("summary"), "report summary")
    requests = _mapping(report.get("requests"), "report requests")
    if not requests or any(not isinstance(row, dict) or row.get("status") != "finished"
                           for row in requests.values()):
        raise ValueError("every reported request must be finished")
    lifecycle = _mapping(summary.get("llama_cuda_graph_lifecycle"), "CUDA Graph lifecycle")
    if lifecycle.get("remaining_compiled_invocations") != 0:
        raise ValueError("CUDA Graph structural program is incomplete")
    transitions = lifecycle.get("transitions")
    if not isinstance(transitions, (tuple, list)) or not transitions:
        raise ValueError("completed lifecycle transitions are required")
    costs = _mapping(lifecycle.get("cuda_runtime_cost_registry"), "CUDA runtime cost registry")
    structures = _mapping(lifecycle.get("structures_registry"), "CUDA structure registry")
    expected_counts = _mapping(lifecycle.get("event_counts"), "lifecycle event counts")
    if any(key not in PHASES or type(count) is not int or count <= 0
           for key, count in expected_counts.items()):
        raise ValueError("lifecycle event counts must contain supported phases and positive integers")

    seen, counts, groups, limitations = set(), Counter(), {}, set()
    for transition in transitions:
        transition = _mapping(transition, "lifecycle transition")
        identity, events = transition.get("invocation_id"), transition.get("events")
        if not isinstance(identity, str) or not identity or identity in seen or identity not in costs:
            raise ValueError("each lifecycle invocation requires one unique registered cost audit")
        if (transition.get("pricing_ready") is not True
                or transition.get("unresolved_update_count") != 0
                or transition.get("body_executions") != 1
                or transition.get("capture_executes_body") is not False):
            raise ValueError("lifecycle work is unresolved or duplicated")
        if not isinstance(events, (list, tuple)) or not events or any(phase not in PHASES for phase in events):
            raise ValueError("lifecycle contains an unsupported or missing cost phase")
        seen.add(identity)
        event_counts = Counter(events)
        counts.update(event_counts)
        audit = _mapping(costs[identity], "registered cost audit")
        if audit.get("schema") != "cuda-runtime-cost/v1" or audit.get("audit_id") != identity:
            raise ValueError("runtime registry must contain the complete matching cost audit")
        if (audit.get("runtime_measurement_mode") != "experimental_exact_structure"
                or audit.get("prediction_qualified") is not False or audit.get("qualified") is not False):
            raise ValueError("host scatter summary requires explicit unqualified exact-structure measurements")
        uncertainty = _mapping(audit.get("independent_measurement_uncertainty"), "phase measurements")
        bindings = _mapping(audit.get("phase_structures"), "phase structures")
        if set(uncertainty) != set(events) or set(bindings) != set(events):
            raise ValueError("cost measurements and structure bindings must match realized phases exactly")
        evidence = audit.get("evidence")
        if not isinstance(evidence, str) or not evidence:
            raise ValueError("independent measurement provenance is required")
        audit_limits = audit.get("independent_measurement_limitations")
        if not isinstance(audit_limits, (list, tuple)) or not audit_limits or any(
                not isinstance(item, str) or not item for item in audit_limits):
            raise ValueError("independent measurement limitations must be retained")
        limitations.update(audit_limits)
        for phase, count in event_counts.items():
            measurement = _measurement(uncertainty[phase])
            binding = _structure(bindings[phase], structures)
            if phase == "update_failure" and not {"previous", "current"}.issubset(binding):
                raise ValueError("failed updates require the measured old/new structure pair")
            key = phase, evidence, json.dumps(binding, sort_keys=True)
            if key not in groups:
                groups[key] = {"phase": phase, "structure_binding": binding,
                    "measurement_evidence": evidence, "measurement": measurement,
                    "event_count": 0, **_empty_totals()}
            group = groups[key]
            if group["measurement"] != measurement:
                raise ValueError("the same measured phase/structure has conflicting summaries")
            group["event_count"] += count
            _add_measurement(group, measurement, count)
    if seen != set(costs):
        raise ValueError("cost registry includes invocations absent from completed lifecycle events")
    if dict(counts) != expected_counts:
        raise ValueError("lifecycle event summary differs from realized transitions")

    phases, total = {}, _empty_totals()
    for phase in PHASES:
        selected = [groups[key] for key in sorted(groups) if key[0] == phase]
        if not selected:
            continue
        row = {"event_count": counts[phase], "measurement_groups": selected, **_empty_totals()}
        for group in selected:
            for key in TOTAL_FIELDS:
                row[key] += group[key]
                total[key] += group[key]
        phases[phase] = row
    return {"schema": "heterollm.cuda-graph-host-cost-sensitivity/v1",
        "source": source, "job_id": job_id,
        "scope": "all_completed_report_invocations_including_startup_and_warmup",
        "invocation_count": len(seen), "event_count": sum(counts.values()),
        "event_counts": dict(counts), "phases": phases, "host_service_totals_ns": total,
        "prediction_qualified": False, "end_to_end_sensitivity_computed": False,
        "native_execution_performed": False, "new_measurements_performed": False,
        "simulation_parameters_modified": False,
        "unseen_structure_policy": "reject; no interpolation, extrapolation, or ordinary-chain fallback",
        "interpretation": list(INTERPRETATION),
        "independent_measurement_limitations": sorted(limitations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Completed job JSON or standalone simulation report")
    parser.add_argument("--output", type=Path, help="Write summary JSON; otherwise print it")
    args = parser.parse_args()
    try:
        document = json.loads(args.report.read_text(encoding="utf-8-sig"))
        result = summarize_completed_report(document, source=str(args.report.resolve()))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
