"""Formal fast-scan summaries distinguish forecasts from executed transactions."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report():
    source = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan_report.py"
    spec = importlib.util.spec_from_file_location("b200_fast_scan_report_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_directory(tmp_path, rows):
    (tmp_path / "points").mkdir()
    (tmp_path / "progress").mkdir()
    (tmp_path / "logs").mkdir()
    manifest = {
        "models": [{"model_id": "small", "label": "Small", "parameters_b": .5}],
        "architectures": ["HBM", "HBF"], "prompts": [65536],
        "outputs": [1024, 5120], "parameters": {"simulation_mode": "fast"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for index, row in enumerate(rows):
        (tmp_path / "points" / f"{index}.json").write_text(json.dumps(row), encoding="utf-8")
    return tmp_path


def point(architecture="HBM", output=1024, status="forecast_completed", **updates):
    return {
        "point_id": f"small-{architecture}-{output}", "model_id": "small",
        "architecture": architecture, "prompt_tokens": 65536, "output_tokens": output,
        "simulation_mode": "fast", "status": status, "ttft_ns": 2e9, "tpot_ns": 3e6,
        "wall_seconds": 7.5,
        "approximation": {
            "sampled_decode_steps": 80, "predicted_decode_steps": output - 1 - 80,
            "method": "maintenance-aware", "target_relative_error": .1,
            "conditional_relative_error_budget": .0339,
            "validation_status": "not_validated_against_full_detailed",
            "sampling_attempts": [{"sampled_steps": 52}, {"sampled_steps": 80}],
            "sampling_resets": [{"reason": "ordinary_erase_cleanup"}],
            "error_budget_breakdown": {"combined": .0339},
        },
        **updates,
    }


def test_terminal_counts_combine_success_without_double_counting(report):
    rows = [{"status": status} for status in (
        "completed", "forecast_completed", "forecast_rejected", "capacity_rejected", "failed")]
    counts = report._counts(rows, 72)
    assert counts["completed"] == 2
    assert counts["detailed_completed"] == counts["forecast_completed"] == 1
    assert counts["forecast_rejected"] == counts["capacity_rejected"] == counts["failed"] == 1
    assert counts["pending"] == 67


def test_csv_preserves_forecast_budget_and_rejections_without_fake_latencies(report, tmp_path, monkeypatch):
    rejected = point("HBF", status="forecast_rejected", ttft_ns=None, tpot_ns=None,
                     reason="prefix_calibration_exceeds_error_budget")
    rejected["approximation"]["fallback_reasons"] = ["prefix_calibration_exceeds_error_budget"]
    path = scan_directory(tmp_path, [point(), rejected,
        point(output=5120, status="capacity_rejected", ttft_ns=None, tpot_ns=None, reason="capacity")])
    plotted = []
    monkeypatch.setattr(report, "_plot", lambda root, manifest, rows, total: plotted.extend(rows))
    counts = report.write_report(path)
    assert counts["completed"] == 1 and counts["pending"] == 1
    assert [row["status"] for row in plotted] == ["forecast_completed"]
    with (path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    indexed = {row["status"]: row for row in rows}
    completed = indexed["forecast_completed"]
    assert completed["simulation_mode"] == "fast"
    assert completed["ttft_fidelity"] == "detailed" and completed["tpot_fidelity"] == "estimated"
    assert completed["physical_execution_scope"] == "detailed_prefill_and_decode_prefix_only"
    assert completed["sampled_decode_steps"] == "80" and completed["predicted_decode_steps"] == "943"
    assert float(completed["conditional_relative_error_budget"]) == .0339
    assert json.loads(completed["sampling_attempts"])[-1]["sampled_steps"] == 80
    assert json.loads(completed["sampling_resets"])[0]["reason"] == "ordinary_erase_cleanup"
    assert json.loads(completed["approximation"])["target_relative_error"] == .1
    for status in ("forecast_rejected", "capacity_rejected"):
        assert all(indexed[status][metric] == "" for metric in ("ttft_ns", "tpot_ns", "ttft_s", "tpot_ms"))
        assert indexed[status]["reason"]
    data = json.loads((path / "results.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["approximation"]["sampling_resets"]
    markdown = (path / "report.md").read_text(encoding="utf-8")
    assert "资源占用、事务及物理状态只覆盖已执行前缀" in markdown
    assert "10% 是误差目标" in markdown
    assert "not_validated_against_full_detailed" in markdown
    assert "prefix_calibration_exceeds_error_budget" in markdown


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), True])
def test_forecast_success_requires_positive_finite_latency(report, tmp_path, value):
    path = scan_directory(tmp_path, [point(tpot_ns=value)])
    with pytest.raises(ValueError, match="invalid tpot_ns"):
        report._read_inputs(path)


def test_markdown_only_refresh_leaves_csv_json_and_plots_unchanged(report, tmp_path):
    path = scan_directory(tmp_path, [point()])
    for filename in ("summary.csv", "results.json", "comparison.png", "comparison.svg"):
        (path / filename).write_text("do not regenerate", encoding="utf-8")
    report.refresh_markdown(path)
    assert (path / "report.md").exists()
    for filename in ("summary.csv", "results.json", "comparison.png", "comparison.svg"):
        assert (path / filename).read_text(encoding="utf-8") == "do not regenerate"


def test_forecast_completion_does_not_become_real_decode_progress(report, tmp_path):
    path = scan_directory(tmp_path, [])
    row = point(output=5120)
    (path / "logs" / f"{row['point_id']}.log").write_text("\n".join(json.dumps(event) for event in (
        {"stage": "serving_cohorts", "completed": 48, "wall_seconds": 90},
        {"stage": "fast_forecast_complete", "completed": 5119, "total": 5119, "wall_seconds": 91},
    )), encoding="utf-8")
    (path / "progress" / f"{row['point_id']}.json").write_text(json.dumps({
        "wall_seconds": 91, "progress": {"stage": "fast_forecast_complete", "completed": 5119, "total": 5119}
    }), encoding="utf-8")
    progress = report._running_progress(path, row, 2048)
    assert progress["completed_cohorts"] == 48
    assert "真实执行 16" in progress["decode"]
    assert "预测步数不计作真实执行" in progress["detail"]


def test_plot_explicitly_labels_forecast_tail(report, tmp_path):
    path = scan_directory(tmp_path, [point()])
    report.write_report(path)
    svg = (path / "comparison.svg").read_text(encoding="utf-8")
    assert "Fast mode: detailed TTFT; mean TPOT includes a forecast tail" in svg
    assert "forecast-rejected" in svg
    assert (path / "comparison.png").stat().st_size > 0


def test_explicit_real_counters_override_forecast_sized_cohort_event(report, tmp_path):
    path = scan_directory(tmp_path, [])
    row = point(output=5120)
    (path / "progress" / f"{row['point_id']}.json").write_text(json.dumps({
        "simulation_mode": "fast", "real_prefill_steps": 32, "real_decode_steps": 80,
        "wall_seconds": 100, "progress": {"stage": "serving_cohorts", "completed": 5151}
    }), encoding="utf-8")
    progress = report._running_progress(path, row, 2048)
    assert progress["completed_cohorts"] == 112
    assert "32/32 块" in progress["prefill"]
    assert "真实执行 80" in progress["decode"]
    assert "5119" not in progress["decode"]


@pytest.mark.parametrize("status,label", [
    ("ready", "待启动"), ("completed_with_rejections", "已结束（含拒绝点）")])
def test_simulation_manifest_shows_mode_before_any_point_and_queue_labels(report, tmp_path, status, label):
    path = scan_directory(tmp_path, [])
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parameters"] = {}
    manifest["simulation"] = {"mode": "fast", "fast_decode": {
        "sampling_strategy": "adaptive", "target_relative_error": .1, "max_sample_tokens": 128}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (path / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    report.refresh_markdown(path)
    markdown = (path / "report.md").read_text(encoding="utf-8")
    assert label in markdown
    assert "快速模式口径" in markdown
    assert "| 仿真模式 | fast |" in markdown
    assert "| Decode 采样策略 | adaptive |" in markdown
    assert "| 误差目标（非逐点实测保证） | 10.0% |" in markdown
    assert "| Decode 真实采样上限（步） | 128 |" in markdown


def test_watch_stops_when_all_points_handled_with_rejections(report, tmp_path, monkeypatch):
    path = scan_directory(tmp_path, [])
    (path / "status.json").write_text('{"status":"completed_with_rejections"}', encoding="utf-8")

    def forbidden_sleep(_seconds):
        pytest.fail("A finished queue must not remain in the watcher loop")

    monkeypatch.setattr(report.time, "sleep", forbidden_sleep)
    report.watch_report(path)
    assert (path / "report.md").exists()
