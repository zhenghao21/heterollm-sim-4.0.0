"""Host-service scatter must not be presented as end-to-end uncertainty."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "tools/summarize_cuda_graph_cost_sensitivity.py"
spec = importlib.util.spec_from_file_location("cuda_graph_cost_sensitivity", path)
sensitivity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensitivity)


def completed_report():
    phase_lists = [["ordinary_submit"], ["replay_submit"], ["replay_submit"],
                   ["destroy_exec", "update_failure"]]
    transitions, registry = [], {}
    old = {"structure_id": "old", "node_count": 2}
    new = {"structure_id": "new", "node_count": 3}
    values = {"ordinary_submit": (100, 80, 150), "replay_submit": (10, 5, 30),
              "destroy_exec": (20, 15, 25), "update_failure": (40, 30, 70)}
    counts = {}
    for index, phases in enumerate(phase_lists):
        identity = f"invoke:{index}"
        transitions.append({"invocation_id": identity, "label": f"decode:{index}",
            "events": phases, "pricing_ready": True, "unresolved_update_count": 0,
            "body_executions": 1, "capture_executes_body": False})
        bindings, measurements = {}, {}
        for phase in phases:
            counts[phase] = counts.get(phase, 0) + 1
            med, low, high = values[phase]
            measurements[phase] = {"median_ns": med, "minimum_ns": low,
                "maximum_ns": high, "repeat_mad_ns": 2, "repeat_count": 99}
            bindings[phase] = old if phase == "destroy_exec" else new
            if phase == "update_failure":
                bindings[phase] = {**new, "previous": old, "current": new}
        registry[identity] = {"schema": "cuda-runtime-cost/v1", "audit_id": identity,
            "runtime_measurement_mode": "experimental_exact_structure", "qualified": False,
            "prediction_qualified": False, "independent_measurement_uncertainty": measurements,
            "phase_structures": bindings, "evidence": "independent_measurements.json",
            "independent_measurement_limitations": ["Synthetic kernels are not real kernels."]}
    return {"requests": {"r": {"status": "finished"}}, "summary": {
        "llama_cuda_graph_lifecycle": {"remaining_compiled_invocations": 0,
            "transitions": transitions, "event_counts": counts,
            "cuda_runtime_cost_registry": registry, "structures_registry": {
                "old": {"topology": "old-chain", "node_count": 2},
                "new": {"topology": "new-chain", "node_count": 3}}}}}


def lifecycle(report):
    return report["summary"]["llama_cuda_graph_lifecycle"]


def test_counts_completed_invocations_once_and_keeps_host_sums_separate_from_e2e():
    report = completed_report()
    # Ordinary submission can be split over many task audit references. Those
    # references must not multiply the measured cost of the whole enqueue loop.
    report["events"] = [{"phase": "cuda_runtime_ordinary_submit", "audit_id": "invoke:0"}] * 20
    result = sensitivity.summarize_completed_report({"status": "completed", "job_id": "j", "report": report})
    assert result["job_id"] == "j"
    assert result["invocation_count"] == 4
    assert result["event_count"] == 5
    assert result["phases"]["ordinary_submit"]["event_count"] == 1
    replay = result["phases"]["replay_submit"]
    assert replay["event_count"] == 2 and len(replay["measurement_groups"]) == 1
    assert replay["sum_of_event_medians_ns"] == 20
    assert result["host_service_totals_ns"] == {
        "sum_of_event_medians_ns": 180, "sum_of_event_minima_ns": 135,
        "sum_of_event_maxima_ns": 305}
    assert result["end_to_end_sensitivity_computed"] is False
    assert result["prediction_qualified"] is False
    assert result["new_measurements_performed"] is False


def test_destroy_and_failed_update_preserve_the_registered_old_new_bindings():
    result = sensitivity.summarize_completed_report(completed_report())
    old = result["phases"]["destroy_exec"]["measurement_groups"][0]["structure_binding"]
    pair = result["phases"]["update_failure"]["measurement_groups"][0]["structure_binding"]
    assert old["structure_id"] == pair["previous"]["structure_id"] == "old"
    assert pair["current"]["structure_id"] == "new"


@pytest.mark.parametrize("change", ["not_finished", "incomplete", "event_count", "missing_cost",
    "extra_cost", "duplicate_invocation", "unresolved", "missing_phase", "qualified",
    "unseen_structure", "missing_pair", "conflicting_measurement", "nan", "invalid_range"])
def test_rejects_incomplete_unmatched_or_unqualified_input(change):
    report = deepcopy(completed_report())
    state = lifecycle(report)
    audit = state["cuda_runtime_cost_registry"]["invoke:1"]
    if change == "not_finished":
        report["requests"]["r"]["status"] = "running"
    elif change == "incomplete":
        state["remaining_compiled_invocations"] = 1
    elif change == "event_count":
        state["event_counts"]["replay_submit"] = 3
    elif change == "missing_cost":
        state["cuda_runtime_cost_registry"].pop("invoke:1")
    elif change == "extra_cost":
        state["cuda_runtime_cost_registry"]["unused"] = audit
    elif change == "duplicate_invocation":
        state["transitions"][2]["invocation_id"] = "invoke:1"
    elif change == "unresolved":
        state["transitions"][1]["pricing_ready"] = False
    elif change == "missing_phase":
        audit["independent_measurement_uncertainty"].clear()
    elif change == "qualified":
        audit["prediction_qualified"] = True
    elif change == "unseen_structure":
        state["structures_registry"].pop("new")
    elif change == "missing_pair":
        state["cuda_runtime_cost_registry"]["invoke:3"]["phase_structures"]["update_failure"].pop("previous")
    elif change == "conflicting_measurement":
        audit["independent_measurement_uncertainty"]["replay_submit"]["median_ns"] = 11
    elif change == "nan":
        audit["independent_measurement_uncertainty"]["replay_submit"]["maximum_ns"] = float("nan")
    else:
        audit["independent_measurement_uncertainty"]["replay_submit"]["minimum_ns"] = 50
    with pytest.raises(ValueError):
        sensitivity.summarize_completed_report(report)


def test_job_wrapper_must_be_completed():
    with pytest.raises(ValueError, match="completed simulation"):
        sensitivity.summarize_completed_report({"status": "running", "report": completed_report()})
