"""A rerun report preserves pairing gates and excludes unfinished jobs."""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location("cost_logic_rerun_report", TOOLS / "render_cost_logic_rerun_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
finally:
    sys.path.remove(str(TOOLS))


def test_mape_pairs_are_equal_weight_and_failed_rows_do_not_count():
    cases = []
    for index, (old, new, status) in enumerate([(40, -20, "compared"), (-10, 20, "compared"),
                                               (None, -90, "compared"), (90, 1, "failed")]):
        history = {"metrics": {key: {"previous_signed_error_percent": old} for key, _ in report.METRICS}} if old is not None else None
        cases.append({"case_id": str(index), "modes": {"on": {"history": history, "comparison": {
            "status": status, "metrics": {key: {"signed_error_percent": new} for key, _ in report.METRICS}}}}})
    actual = report.metric_error_summary(cases, "engine_e2e_ns")
    assert actual["current_pairs"] == 3
    assert actual["matched_old_new_pairs"] == 2
    assert actual["current_mape_percent"] == pytest.approx(130 / 3)
    assert actual["previous_mape_percent"] == 25
    assert actual["matched_current_mape_percent"] == 20
    assert actual["improved_pairs"] == actual["worsened_pairs"] == 1
    assert report.metric_error_summary(cases, "engine_e2e_ns", "off")["current_mape_percent"] is None


def test_compact_job_retains_predicates_that_reject_invalid_lifecycle():
    lifecycle = {"remaining_compiled_invocations": 1, "event_counts": {"replay_submit": 1},
                 "cuda_runtime_cost_registry": {"huge": list(range(100))},
                 "transitions": [{"events": ["replay_submit"], "body_executions": 2,
                                  "capture_executes_body": False, "pricing_ready": False,
                                  "unresolved_update_count": 7, "irrelevant": "large payload"}]}
    job = {"status": "completed", "report": {"summary": {"llama_cuda_graph_lifecycle": lifecycle},
          "requests": {"measured": {"status": "finished"}}, "batch_history": ["discard"],
          "measurement_semantics": {"latency": {"primary_boundary": "engine"}}}}
    saved = deepcopy(job)
    compact = report.compact_job(job)
    retained = report.graph_analysis.find_lifecycle(compact["report"])
    assert retained["remaining_compiled_invocations"] == 1
    assert retained["transitions"][0]["body_executions"] == 2
    assert retained["transitions"][0]["pricing_ready"] is False
    assert retained["transitions"][0]["unresolved_update_count"] == 7
    assert "cuda_runtime_cost_registry" not in retained
    assert "batch_history" not in compact["report"]
    assert compact["report"]["requests"] == job["report"]["requests"]
    assert job == saved


def test_read_gzip_fallback_prefers_actual_uncompressed_result(tmp_path):
    path = tmp_path / "job_result.json"
    with gzip.open(str(path) + ".gz", "wt", encoding="utf-8") as stream:
        json.dump({"status": "failed"}, stream)
    assert report.read_json(path)["status"] == "failed"
    path.write_text('{"status":"completed"}', encoding="utf-8")
    assert report.read_json(path)["status"] == "completed"


def test_wall_clock_uses_execution_span_instead_of_sum_for_parallel_jobs():
    rows = [{"case_id": "a", "started_at": "2026-10-10T02:00:00Z", "finished_at": "2026-10-10T02:00:10Z"},
            {"case_id": "b", "started_at": "2026-10-10T10:00:02+08:00", "finished_at": "2026-10-10T10:00:12+08:00"},
            {"case_id": "untimed", "started_at": None, "finished_at": None}]
    actual = report.wall_clock_summary(rows)
    assert actual["timed_jobs"] == 2
    assert actual["median_wall_seconds"] == 10
    assert actual["observed_execution_span_seconds"] == 12
    assert actual["rows"][-1]["wall_seconds"] is None
    with pytest.raises(ValueError, match="before"):
        report.wall_clock_summary([{**rows[0], "finished_at": "2026-10-10T01:59:59Z"}])


def test_unfinished_matrix_cannot_write_comparison_or_final_html(tmp_path):
    output = tmp_path / "new"
    output.mkdir()
    baseline = tmp_path / "old"
    baseline.mkdir()
    for directory in (output, baseline):
        (directory / "native_timing_manifest.json").write_text('{"status":"completed"}', encoding="utf-8")
    (output / "correction_inputs.json").write_text(json.dumps({
        "native_measurements_reused": True, "scenario_inputs_changed": False}), encoding="utf-8")
    (output / "cases.json").write_text(json.dumps({"cases": [
        {"case_id": f"case_{i}", "preset_id": f"preset-{i}", "name": f"Model {i}"}
        for i in range(10)]}), encoding="utf-8")
    (output / "ui_case_0_graph_off_result.json").write_text('{"status":"running"}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonterminal"):
        report.checked_collect(output, baseline)
    assert not (output / "report.html").exists()
    assert not list(output.glob("comparison_*.json"))
