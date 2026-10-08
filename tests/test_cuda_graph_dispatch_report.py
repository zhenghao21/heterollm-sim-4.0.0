"""A correction comparison must not relabel reused native timings as new."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("dispatch_report", ROOT / "tools/validate_cuda_graph_dispatch_changes.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_report_uses_actual_metric_keys_and_retains_missing_pairs(tmp_path, monkeypatch):
    metrics = {key: {"simulation_ns": 1200000, "native": {"median_ns": 1000000},
                     "signed_error_percent": 20.0} for key, _ in report.METRICS}
    previous = {key: {"previous_simulation_ns": 1300000, "previous_signed_error_percent": 30.0}
                for key, _ in report.METRICS}
    summary = {"cases": [{"name": "Example <model>", "modes": {
        "on": {"comparison": {"status": "compared", "metrics": metrics}, "history": {"metrics": previous}},
        "off": {"comparison": {"status": "pending"}, "history": None}}, "graph_speedup": None}],
        "completed_comparisons": 1, "expected_comparisons": 2}
    monkeypatch.setattr(report, "collect", lambda *_: summary)
    (tmp_path / "correction_inputs.json").write_text(json.dumps({
        "native_measurements_reused": True, "scenario_inputs_changed": False}), encoding="utf-8")
    assert report.report(tmp_path, tmp_path / "baseline") == {"completed": 1, "expected": 2}
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Example &lt;model&gt;" in html and "pending" in html
    assert "+20.00%" in html and "+30.00%" in html and "1.2000" in html
    assert "+10.00 pp" in html
    assert "修复前 30.000% → 修复后 20.000%" in html and "纳入 1/2 组" in html
    assert "并非本轮重新计时" in html and "尚未完整解决" in html
    assert json.loads((tmp_path / "comparison_summary.json").read_text())["graph_device_dispatch_fix_complete"] is False


def test_prepare_cannot_overwrite_baseline(tmp_path):
    with pytest.raises(ValueError, match="must not overwrite"):
        report.prepare(tmp_path, tmp_path)


def test_report_keeps_current_result_without_history_and_links_available_evidence(tmp_path, monkeypatch):
    metrics = {key: {"simulation_ns": 900000, "native": {"median_ns": 1000000},
                     "signed_error_percent": -10.0} for key, _ in report.METRICS}
    summary = {"cases": [{"name": "Model", "modes": {
        "on": {"comparison": {"status": "compared", "metrics": metrics}, "history": None},
        "off": {"comparison": {"status": "failed", "error": "bad <input>"}, "history": None}},
        "graph_speedup": {"native_e2e_reduction_percent": 12.5, "simulation_e2e_reduction_percent": None}}],
        "completed_comparisons": 1, "expected_comparisons": 2}
    monkeypatch.setattr(report, "collect", lambda *_: summary)
    records = {
        "correction_inputs.json": {"native_measurements_reused": True, "scenario_inputs_changed": False},
        "native_timing_manifest.json": {"started_local": "2026-10-08 19:32:40", "finished_local": "2026-10-08 19:36:17",
                                        "environment": {"timezone": "China Standard Time"}},
        "independent_dispatch_summary.json": {"status": "rejected_as_general_predictive_calibration", "prediction_qualified": False},
        "source_dispatch_coverage.json": {"summary": {"matched_models": 9, "model_count": 10,
                                                       "matched_regions": 36, "audited_regions": 36}},
        "queue_model_probe/experiment_plan.json": {"cases": []},
        "queue_model_probe/queue_model_identification.json": {"prediction_qualified": False},
        "queue_model_probe_quiet/quiet_comparison.json": {
            "started_beijing": "2026-10-09T00:04:21+08:00", "finished_beijing": "2026-10-09T00:04:32+08:00",
            "case_count": 31, "repetitions": 21, "parameter_refit": False, "fitted_model_equal": True,
            "prediction_qualified": False, "holdout_comparison": [
                {"case_id": "nodes_holdout_255", "low_cpu": {"span_error_percent": -0.5712}},
                {"case_id": "mixed_alternating_holdout", "low_cpu": {"span_error_percent": 10.5223}},
                {"case_id": "mixed_shuffled_holdout", "low_cpu": {"span_error_percent": 5.8045}}],
            "observer_controls": [{"difference_percent": 7.2297}, {"difference_percent": 10.0737}]},
        "queue_model_probe_quiet/queue_model_identification.json": {"prediction_qualified": False},
        "queue_model_probe_quiet/preflight.json": {"cpu_samples": [{"cpu_percent": 2.2}, {"cpu_percent": 3.9}],
                                                   "gpu_status_csv": "2026/10/09 00:04:20.921, 25 %, 17 %"},
        "http_observer_disconnect_review.json": {"completed_simulations": 20, "failed_simulations": 0,
                                                 "observed_polling_disconnects": 2, "service_ports": [8800, 8809],
                                                 "fix_stage": "after all 20 simulations completed",
                                                 "matrix_rerun_after_http_fix": False},
    }
    for name, value in records.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "source_dispatch_coverage.html").write_text("<html></html>", encoding="utf-8")
    report.report(tmp_path, tmp_path / "baseline")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "0.9000" in html and "-10.00%" in html and "12.500%" in html
    assert "bad &lt;input&gt;" in html and "colspan=\"7\"" in html
    assert "2026-10-08 19:32:40" in html and "2026-10-08 19:36:17" in html
    assert 'href="independent_dispatch_summary.json"' in html
    assert 'href="source_dispatch_coverage.html"' in html and "9/10" in html and "36/36" in html
    assert 'href="queue_candidate_holdout.json"' not in html
    assert 'href="queue_model_probe/queue_model_identification.json"' in html
    assert "31 组先声明训练与留出划分" in html and "事后探索" in html
    assert 'href="queue_model_probe_quiet/quiet_comparison.json"' in html
    assert 'href="queue_model_probe_quiet/queue_model_identification.json"' in html
    assert "2026-10-09 00:04:21" in html and "31 个配置 × 21 次重复" in html
    assert "2.2%–3.9%" in html and "GPU 仍有 25% 活动且归属未完全确认" in html
    assert "-0.57%" in html and "+10.52% / +5.80%" in html and "+7.23% / +10.07%" in html
    assert "固定原模型，没有重新训练或调整参数" in html
    assert "20 组完成、0 组失败" in html and "2 次轮询通信异常" in html
    assert "此修复发生在 20 组复跑结束后" in html and "没有在 HTTP 修复后再次重跑这 20 组" in html
    assert 'href="http_observer_disconnect_review.json"' in html
    assert 'href="../../tests/test_web_disconnected_response.py"' in html
    assert "GET_ROWS" in html and "同格式 Q8" in html and "未写入预测预设" in html
    stored = json.loads((tmp_path / "comparison_summary.json").read_text(encoding="utf-8"))
    assert stored["supporting_reports"]["independent_dispatch"]["prediction_qualified"] is False


@pytest.mark.parametrize("field,value", [("native_measurements_reused", False), ("scenario_inputs_changed", True),
                                       ("native_values_used_for_calibration", True),
                                       ("device_dispatch_calibration_installed", True)])
def test_report_rejects_inconsistent_reuse_claim(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(report, "collect", lambda *_: {})
    record = {"native_measurements_reused": True, "scenario_inputs_changed": False, field: value}
    (tmp_path / "correction_inputs.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance is inconsistent"):
        report.report(tmp_path, tmp_path / "baseline")
    assert not (tmp_path / "report.html").exists()


@pytest.mark.parametrize("changes,preset,status,expected", [
    ({}, "qwen3-0_6b", "compared", (1, 1, 1)),
    ({"hidden_size": 64}, "qwen3-0_6b", "compared", (0, 0, 1)),
    (None, "qwen3-0_6b", "pending", (0, 0, 0)),
    ({}, "different-preset", "compared", (0, 1, 1)),
])
def test_input_summary_reads_actual_submission_and_requires_explicit_empty_changes(tmp_path, changes, preset, status, expected):
    metadata = {"model_preset_id": preset, "gguf_preset_origin": {"filename": "Qwen3-0.6B-f16.gguf"}}
    if changes is not None:
        metadata["gguf_preset_changes"] = changes
    (tmp_path / "ui_qwen_graph_on_submission.json").write_text(json.dumps({
        "scenario": {"model": {"metadata": metadata}}}), encoding="utf-8")
    cases = [{"case_id": "qwen", "preset_id": "qwen3-0_6b", "name": "Qwen", "modes": {
        "on": {"comparison": {"status": status, "physical_participation": {"status": "pass"}}}}}]
    summary = report._input_participation_summary(tmp_path, cases)
    assert tuple(summary[key] for key in ("original_gguf_presets", "explicit_empty_changes", "physical_participation_passed")) == expected
    assert summary["rows"][0]["gguf_filename"] == "Qwen3-0.6B-f16.gguf"


def test_e2e_summary_is_equal_weight_and_excludes_missing_invalid_pairs():
    cases = []
    for index, (old, new, status) in enumerate([(-40, -20, "compared"), (10, 20, "compared"),
                                               (10, -10, "compared"), (None, 0, "compared"),
                                               (100, 1, "failed"), (float("nan"), 0, "compared")]):
        cases.append({"name": str(index), "case_id": str(index), "modes": {"on": {
            "comparison": {"status": status, "metrics": {"engine_e2e_ns": {"signed_error_percent": new}}},
            "history": {"metrics": {"engine_e2e_ns": {"previous_signed_error_percent": old}}}}}})
    actual = report._e2e_error_summary(cases)
    assert actual["matched_pairs"] == 3 and actual["excluded_pairs"] == 3
    assert actual["previous_mape_percent"] == 20
    assert actual["current_mape_percent"] == pytest.approx(50 / 3)
    assert actual["improved_pairs"] == actual["worsened_pairs"] == actual["unchanged_pairs"] == 1
    assert [pair["absolute_error_reduction_percentage_points"] for pair in actual["pairs"]] == [20, -10, 0]
    empty = report._e2e_error_summary(cases[3:])
    assert empty["matched_pairs"] == 0 and empty["previous_mape_percent"] is None
