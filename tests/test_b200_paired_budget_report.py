"""Paired conditional bounds and archived retries never become measured errors."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report():
    source = Path(__file__).resolve().parents[1] / "scripts" / "b200_hbm_hbf_scan_report.py"
    spec = importlib.util.spec_from_file_location("b200_paired_report_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision(*, accepted=True, paired=True):
    return {
        "schema": "directional_decode_interval_pair/v1",
        "semantics": "conditional_model_bounds_not_measured_error",
        "strategy": "matched_directional_pair" if paired else "independent_symmetric_sufficient_bound",
        "target_relative_error": .1,
        "independent_symmetric_target_relative_error": .1 / 2.1,
        "reference_source_id": "small-HBM-64k-10k" if paired else None,
        "candidate_memory_kind": "HBF", "reference_memory_kind": "HBM" if paired else None,
        "candidate_relative_error_budget": (.0533 if paired else .0333) if accepted else .103,
        "reference_relative_error_budget": .0339 if paired else None,
        "pair_relative_error_budget": (.0873 if accepted else .123) if paired else None,
        "pair_ratio_direction": "hbf/hbm" if paired else None,
        "pair_ratio_interval": {"estimate": .9, "lower": .85, "upper": .93} if paired else None,
        "candidate_within_target": accepted, "reference_within_target": paired,
        "pair_within_target": accepted if paired else None,
        "accepted": accepted, "reasons": [] if accepted else ["paired_ratio_exceeds_target"],
    }


def point(*, accepted=True, paired=True):
    paired_decision = decision(accepted=accepted, paired=paired)
    return {
        "physical_weight_semantics": "canonical_shared_resident_weights_v1",
        "point_id": "small-HBF-64k-10k", "model_id": "small", "architecture": "HBF",
        "prompt_tokens": 65536, "output_tokens": 10240,
        "simulation_mode": "fast", "status": "forecast_completed" if accepted else "forecast_rejected",
        "ttft_ns": 2e9 if accepted else None, "tpot_ns": 3e6 if accepted else None,
        "reason": None if accepted else "paired_ratio_exceeds_target",
        "approximation": {
            "sampled_decode_steps": 80, "predicted_decode_steps": 10159,
            "conditional_relative_error_budget": paired_decision["candidate_relative_error_budget"],
            "validation_status": "not_validated_against_full_detailed",
            "error_budget_breakdown": {"future_erase_upper_ns": 20, "future_erase_parallel_groups": 2},
            "paired_budget_decision": paired_decision,
            "sampling_attempts": [{"sampled_steps": 80, "paired_budget_decision": paired_decision}],
        },
    }


def scan_directory(tmp_path, row):
    for directory in ("points", "logs", "progress", "points_history"):
        (tmp_path / directory).mkdir()
    manifest = {
        "physical_weight_semantics": "canonical_shared_resident_weights_v1",
        "models": [{"model_id": "small", "label": "Small", "parameters_b": .5}],
        "architectures": ["HBM", "HBF"], "prompts": [65536], "outputs": [10240],
        "parameters": {"simulation_mode": "fast"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "points" / "hbf.json").write_text(json.dumps(row), encoding="utf-8")
    return tmp_path


def test_report_exports_pair_bounds_without_claiming_measured_accuracy(report, tmp_path, monkeypatch):
    path = scan_directory(tmp_path, point())
    monkeypatch.setattr(report, "_plot", lambda *_args: None)
    report.write_report(path)
    with (path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["budget_acceptance_strategy"] == "matched_directional_pair"
    assert row["budget_accepted"] == "True"
    assert row["paired_reference_source_id"] == "small-HBM-64k-10k"
    assert float(row["candidate_relative_error_budget"]) == .0533
    assert float(row["reference_relative_error_budget"]) == .0339
    assert float(row["pair_relative_error_budget"]) == .0873
    assert row["pair_ratio_direction"] == "HBF/HBM"
    assert json.loads(row["paired_budget_decision"])["semantics"] == "conditional_model_bounds_not_measured_error"
    assert json.loads(row["paired_budget_decision"])["pair_ratio_direction"] == "hbf/hbm"
    assert json.loads(row["sampling_attempts"])[0]["paired_budget_decision"]["accepted"] is True
    assert json.loads(row["error_budget_breakdown"])["future_erase_upper_ns"] == 20
    assert row["validation_status"] == "not_validated_against_full_detailed"
    data = json.loads((path / "results.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["paired_budget_decision"]["pair_ratio_interval"]["lower"] == .85
    markdown = (path / "report.md").read_text(encoding="utf-8")
    assert "5.330%" in markdown and "8.730%" in markdown
    assert "small-HBM-64k-10k" in markdown
    assert "不是与完整详细模式对照得到的实测误差" in markdown
    assert "各自的条件预算及 HBF/HBM 比值条件预算都必须 ≤10%" in markdown
    assert "独立对称预算 ≤4.7619%" in markdown


def test_rejected_pair_keeps_blank_metrics_and_real_rejection(report, tmp_path, monkeypatch):
    path = scan_directory(tmp_path, point(accepted=False))
    monkeypatch.setattr(report, "_plot", lambda *_args: pytest.fail("Rejected forecasts must not be plotted"))
    report.write_report(path)
    with (path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["budget_accepted"] == "False"
    assert row["ttft_ns"] == row["tpot_ns"] == ""
    assert row["reason"] == "paired_ratio_exceeds_target"
    assert float(row["pair_relative_error_budget"]) == .123


def test_retry_history_does_not_duplicate_active_counts_and_markdown_only_refresh(report, tmp_path, monkeypatch):
    path = scan_directory(tmp_path, point())
    history = point(accepted=False)
    (path / "points_history" / "small-HBF-64k-10k__old.json").write_text(json.dumps(history), encoding="utf-8")
    (path / "status.json").write_text(json.dumps({
        "status": "running", "reason": "budget_correction_retry",
        "retry_priority_points": ["small-HBF-64k-10k"], "retry_group_completed": 1,
        "retry_group_total": 4, "retry_group_accepted": None,
    }), encoding="utf-8")
    monkeypatch.setattr(report, "_plot", lambda *_args: None)
    counts = report.write_report(path)
    assert counts["completed"] == 1 and counts["forecast_rejected"] == 0 and counts["pending"] == 1
    data = json.loads((path / "results.json").read_text(encoding="utf-8"))
    assert len(data["rows"]) == 1
    assert data["historical_attempts"]["record_count"] == 1
    for filename in ("summary.csv", "results.json", "comparison.png", "comparison.svg"):
        (path / filename).write_text("untouched", encoding="utf-8")
    report.refresh_markdown(path)
    for filename in ("summary.csv", "results.json", "comparison.png", "comparison.svg"):
        assert (path / filename).read_text(encoding="utf-8") == "untouched"
    markdown = (path / "report.md").read_text(encoding="utf-8")
    assert "历史拒绝不重复计入" in markdown and "保留 **1** 份旧结果" in markdown
    assert "本轮修复重跑已处理 **1/4**" in markdown
    assert "验收状态：**尚未全部处理**" in markdown
    assert "优先重跑原拒绝点" in markdown


def test_independent_guard_has_no_invented_reference_or_ratio(report, tmp_path):
    path = scan_directory(tmp_path, point(paired=False))
    row = report._summary_row(point(paired=False))
    assert row["paired_reference_source_id"] is None and row["pair_relative_error_budget"] is None
    report.refresh_markdown(path)
    markdown = (path / "report.md").read_text(encoding="utf-8")
    assert "独立对称充分条件" in markdown
    assert "无匹配参考" in markdown


@pytest.mark.parametrize("accepted,label", [
    (None, "尚未全部处理"), (True, "全部通过"), (False, "未全部通过"),
])
def test_retry_acceptance_boolean_is_a_state_not_a_count(report, tmp_path, accepted, label):
    path = scan_directory(tmp_path, point())
    (path / "status.json").write_text(json.dumps({
        "status": "running", "retry_group_completed": 4 if accepted is not None else 1,
        "retry_group_total": 4, "retry_group_accepted": accepted,
    }), encoding="utf-8")
    report.refresh_markdown(path)
    markdown = (path / "report.md").read_text(encoding="utf-8")
    assert f"验收状态：**{label}**" in markdown
    assert "None 个" not in markdown and "True 个" not in markdown and "False 个" not in markdown
