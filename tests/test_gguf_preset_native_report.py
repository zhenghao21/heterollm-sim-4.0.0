"""Accuracy reports must not turn a derived model or missing pair into evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location("gguf_preset_native_report", TOOLS / "analyze_gguf_preset_validation.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
finally:
    sys.path.remove(str(TOOLS))


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def comparison(sim, native):
    return {"status": "compared", "metrics": {key: {
        "simulation_ns": sim, "native": {"median_ns": native},
        "signed_error_percent": (sim / native - 1) * 100,
    } for key, _ in report.METRICS}}


def test_history_keeps_prediction_change_separate_from_native_drift():
    result = report.history_changes(comparison(120, 110), comparison(100, 100))
    metric = result["metrics"]["engine_e2e_ns"]
    assert metric["simulation_delta_ns"] == 20
    assert metric["simulation_delta_percent"] == pytest.approx(20)
    assert metric["native_delta_percent"] == pytest.approx(10)
    assert metric["current_signed_error_percent"] == pytest.approx(100 / 11)
    assert report.history_changes({"status": "failed"}, comparison(100, 100)) is None


@pytest.mark.parametrize("metadata", [
    {"model_preset_id": "wrong"},
    {"model_preset_id": "original", "gguf_preset_changes": {"hidden_size": 1024}},
])
def test_changed_or_wrong_preset_cannot_retain_native_metrics(tmp_path, monkeypatch, metadata):
    write(tmp_path / "cases.json", {"cases": [{"case_id": "case", "preset_id": "original", "name": "Example"}]})
    for mode in ("off", "on"):
        write(tmp_path / f"ui_case_graph_{mode}_submission.json", {"scenario": {"model": {"metadata": metadata}}})
    monkeypatch.setattr(report, "compare", lambda *args: deepcopy(comparison(100, 100)))
    summary = report.collect(tmp_path, tmp_path / "no_history")
    assert summary["expected_comparisons"] == 2
    assert summary["completed_comparisons"] == 0
    for mode in summary["cases"][0]["modes"].values():
        assert mode["comparison"]["status"] == "configuration_mismatch"
        assert "metrics" not in mode["comparison"]
    html = report.render(tmp_path, summary).read_text(encoding="utf-8")
    assert "0/2 组有效配对" in html
    assert "未完成有效实测" in html
    assert "0.000 ms" not in html


def test_structure_summary_does_not_count_missing_or_failed_models(tmp_path):
    cases = [{"case_id": slug} for slug in ("good", "failed", "missing")]
    assert report.structure_summary(tmp_path, cases)["qualified_models"] == 0
    scope = "source_lifecycle_predicates_only_not_cuda_node_topology_or_timing"
    write(tmp_path / "structure_lifecycle_validation.json", {
        "schema": "heterollm.cuda-graph-source-live-validation/v1", "target_llm_latency_used": False,
        "all_qualified": True,  # An aggregate claim cannot override individual records.
        "models": [
            {"case_id": "good", "qualified": True, "dry_call_count": 386, "live_call_count": 386,
             "mismatch_count": 0, "scope": scope, "native_decisions_used_as_prediction_inputs": False},
            {"case_id": "failed", "qualified": False, "dry_call_count": 386, "live_call_count": 386,
             "mismatch_count": 1, "scope": scope, "native_decisions_used_as_prediction_inputs": False},
        ]})
    summary = report.structure_summary(tmp_path, cases)
    assert summary["status"] == "incomplete"
    assert summary["expected_models"] == 3
    assert summary["qualified_models"] == 1
    assert summary["matched_invocations"] == 386
    assert summary["mismatch_count"] == 1
    assert summary["scope"] == scope


def test_monitoring_retains_failure_history_and_only_uses_complete_samples(tmp_path):
    path = tmp_path / "execution_monitor.jsonl"
    first = {"time_utc": "2026-10-08T11:00:00+00:00", "available_memory_bytes": 1000,
             "processes": [{"private_bytes": 20, "rss_bytes": 10}],
             "cases": [{"case_id": "one", "mode": "off", "status": "failed", "error": "allocation"}]}
    second = {"time_utc": "2026-10-08T11:01:00+00:00", "available_memory_bytes": 900,
              "processes": [{"private_bytes": 18, "rss_bytes": 12}],
              "cases": [{"case_id": "one", "mode": "off", "status": "completed"}]}
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + '\n{"time_utc":', encoding="utf-8")
    summary = report.monitoring_summary(tmp_path)
    assert summary["sample_count"] == 2
    assert summary["observed_span_seconds"] == 60
    assert summary["trailing_partial_record"] is True
    assert summary["latest_case_status_counts"] == {"completed": 1}
    assert summary["peak_service_private_bytes"] == 20
    assert summary["peak_service_rss_bytes"] == 12
    assert summary["minimum_available_memory_bytes"] == 900
    assert summary["failed_cases_observed"][0]["error"] == "allocation"
    assert summary["confirmed_simulation_failed_cases_observed"] == []
    assert summary["latest_simulation_status_counts"] == {"unrecorded": 1}


def test_physical_participation_counts_only_successfully_paired_reports():
    cases = [{"modes": {"off": {"comparison": {
        "case_id": "one", "graph_mode": "off", "status": "compared", "physical_participation": {"status": "pass"}}},
        "on": {"comparison": {"case_id": "one", "graph_mode": "on", "status": "configuration_mismatch",
                               "physical_participation": {"status": "pass"}}}}}]
    summary = report.participation_summary(cases)
    assert summary["expected_comparisons"] == 2
    assert summary["paired_comparisons"] == summary["passed_comparisons"] == 1


def test_ui_observer_failure_does_not_become_backend_failure_after_recovery(tmp_path):
    write(tmp_path / "ui_runs.json", {
        "runs": [{"case_id": "ui", "mode": "off", "job_id": "ui-job", "status": "failed",
                  "simulation_status": "running", "error": "browser closed"},
                 {"case_id": "backend", "mode": "off", "job_id": "new-job", "status": "completed",
                  "simulation_status": "completed"}],
        "failed_attempts": [{"case_id": "backend", "mode": "off", "job_id": "old-job",
                             "status": "failed", "simulation_status": "failed", "error": "allocation"}],
        "observer_interruptions": [{"case_id": "ui", "mode": "off", "job_id": "ui-job",
                                    "status": "failed", "simulation_status": "running", "error": "browser closed"}],
    })
    summary = report.ui_run_summary(tmp_path)
    assert summary["latest_observer_status_counts"] == {"failed": 1, "completed": 1}
    assert summary["latest_simulation_status_counts"] == {"running": 1, "completed": 1}
    assert len(summary["confirmed_simulation_failed_attempts"]) == 1
    assert summary["confirmed_simulation_failed_attempts"][0]["job_id"] == "old-job"
    assert len(summary["observer_interruptions"]) == 1


def test_ui_backend_status_uses_matching_result_and_never_guesses_legacy_failure(tmp_path):
    write(tmp_path / "completed.json", {"job_id": "done", "status": "completed"})
    write(tmp_path / "ui_runs.json", {"runs": [
        {"case_id": "done", "mode": "off", "job_id": "done", "status": "completed",
         "result_path": "completed.json", "simulation_status": "running"},
        {"case_id": "legacy", "mode": "off", "job_id": "legacy", "status": "failed"},
    ]})
    summary = report.ui_run_summary(tmp_path)
    assert summary["latest_simulation_status_counts"] == {"completed": 1, "unrecorded": 1}
    assert summary["confirmed_simulation_failed_attempts"] == []
    assert summary["runs"][0]["actual_result_status"] == "completed"


def test_ui_backend_status_rejects_result_from_another_job(tmp_path):
    write(tmp_path / "result.json", {"job_id": "other", "status": "completed"})
    write(tmp_path / "ui_runs.json", {"runs": [
        {"case_id": "one", "mode": "off", "job_id": "one", "status": "completed",
         "result_path": str(tmp_path / "result.json")},
    ]})
    with pytest.raises(ValueError, match="job_id differs"):
        report.ui_run_summary(tmp_path)
