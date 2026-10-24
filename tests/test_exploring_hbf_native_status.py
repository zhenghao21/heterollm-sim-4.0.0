"""Dashboard observation must not manufacture results or resume experiments."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exploring_hbf_native_status as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def point(hw, *, model="405b", workload="SHORT", state="completed", tps=100., tpot=10e6, dtype="bf16",
          layout=(1, 8, 1, 1), batch=8):
    case = {"case_id": f"{model}-{hw}-{workload}", "model": model, "hardware": hw,
        "workload": workload, "input_tokens": 1660, "output_tokens": 373, "gpu_count": 8,
        "weight_dtype": dtype, "kv_dtype": dtype}
    return {"case_id": case["case_id"], "case": case, "status": state,
        "selected": {"status": state, "backend": "native_serving_physical",
            "parameters": {**dict(zip(("dp", "tp", "pp", "ep"), layout)), "batch_size": batch},
            "metrics": {"tokens_per_second_per_gpu": tps,
                "cluster_tokens_per_second": 8*tps, "mean_complete_request_tpot_ns": tpot}}}


@pytest.fixture(autouse=True)
def no_process_observation(monkeypatch):
    monkeypatch.setattr(report, "_process_info", lambda pid: {"pid": pid, "alive": None,
        "cpu_seconds": None, "rss_bytes": None})


def test_empty_run_does_not_invent_metrics_or_whole_run_progress(tmp_path):
    progress = {"case_id": "a", "pid": 123, "stage": "cohort_tasks", "completed": 98560,
        "total": 123197, "wall_seconds": 26000., "updated_at": "2026-10-20T07:00:00+08:00"}
    write(tmp_path / "status.json", {"state": "running", "counts": {"running": 1, "pending": 29},
        "running": [{"case_id": "a", "candidate_id": "a__DP1_TP8_PP1_EP1__B1", "launcher_pid": 456}],
        "total_cases": 30, "total_candidates": 60})
    write(tmp_path / "results.json", {"cases": []})
    write(tmp_path / "progress/a__DP1_TP8_PP1_EP1__B1.json", progress)
    snapshot = report.build_snapshot(tmp_path)
    assert snapshot["completed_results"] == []
    assert snapshot["eta"]["remaining_min_seconds"] is None
    assert snapshot["eta"]["remaining_max_seconds"] is None
    assert snapshot["running"][0]["completed"] == 98560
    assert snapshot["running"][0]["pid"] == 123
    assert snapshot["running"][0]["candidate_id"] == "a__DP1_TP8_PP1_EP1__B1"
    assert "percentage" not in snapshot["running"][0]
    assert snapshot["observed_at"].endswith("+08:00")
    markdown = report.render_markdown(snapshot)
    assert "尚无真实完成配置" in markdown
    assert "不是Token数" in markdown
    assert "0/30" in markdown and "0/60" in markdown


def test_ratios_have_fixed_storage_over_hbm4_direction(tmp_path):
    points = [point("HBM4"), point("HBF", tps=200., tpot=15e6)]
    write(tmp_path / "results.json", {"cases": points})
    snapshot = report.build_snapshot(tmp_path)
    assert snapshot["cases"] == points
    a, b = snapshot["completed_results"]
    assert (a["throughput_ratio_vs_hbm4"], a["tpot_ratio_vs_hbm4"]) == (1., 1.)
    assert (b["throughput_ratio_vs_hbm4"], b["tpot_ratio_vs_hbm4"]) == (2., 1.5)
    assert b["cluster_tps"] == 1600.
    assert b["tpot_ms"] == 15.


@pytest.mark.parametrize("change", [{"model": "other"}, {"workload": "MID"}, {"dtype": "fp32"}])
def test_missing_or_mismatched_baseline_never_becomes_one(tmp_path, change):
    write(tmp_path / "results.json", {"cases": [point("HBM4"), point("HBF", **change)]})
    row = report.build_snapshot(tmp_path)["completed_results"][1]
    assert row["throughput_ratio_vs_hbm4"] is None
    assert row["tpot_ratio_vs_hbm4"] is None


def test_rejections_historical_backend_and_partial_candidate_do_not_become_selected_results(tmp_path):
    rejected = point("HBF", state="slo_rejected")
    historical = point("CONV")
    historical["selected"]["backend"] = "paper_serving"
    partial = point("HBF+", state="running")
    partial["selected"]["status"] = "completed"
    write(tmp_path / "results.json", {"cases": [rejected, historical, partial]})
    write(tmp_path / "candidates/b1.json", {"status": "completed"})
    assert report.build_snapshot(tmp_path)["completed_results"] == []


def test_eta_is_explicit_and_can_be_updated_without_source_changes(tmp_path):
    assessment = {"summary": "完整候选评估", "remaining_min_seconds": 100.,
        "remaining_max_seconds": 200., "method": "真实壁钟", "caveats": ["不同硬件速度仍未知。"]}
    write(tmp_path / "eta_assessment.json", assessment)
    assert report.build_snapshot(tmp_path)["eta"] == assessment
    replacement = {**assessment, "remaining_min_seconds": None, "remaining_max_seconds": None}
    assert report.build_snapshot(tmp_path, eta=replacement)["eta"] == replacement


def test_expired_local_batch_estimate_is_not_presented_as_current_remaining_time(tmp_path):
    assessment = {"assessed_at": "2020-01-01T00:00:00+08:00", "summary": "整轮未知",
        "current_batch_estimates": [{"case_id": "a", "remaining_min_seconds": 100.}]}
    snapshot = report.build_snapshot(tmp_path, eta=assessment)
    assert snapshot["eta"]["assessment_expired"] is True
    assert snapshot["eta"]["current_batch_estimates"] == []
    assert snapshot["eta"]["remaining_min_seconds"] is None
    assert "评估已超过20分钟" in report.render_markdown(snapshot)


def test_incomplete_baseline_precision_does_not_prove_a_matching_comparison(tmp_path):
    baseline, row = point("HBM4"), point("HBF")
    for value in (baseline, row):
        del value["case"]["weight_dtype"]
    write(tmp_path / "results.json", {"cases": [baseline, row]})
    assert all(value["throughput_ratio_vs_hbm4"] is None
               for value in report.build_snapshot(tmp_path)["completed_results"])


def test_configuration_omits_native_model_graphs_and_keeps_real_hardware(tmp_path):
    write(tmp_path / "plan.json", {"models": {"405b": {"layers": 126}}, "measurement": {"slo": 100},
        "deployment_search": {"fixed": "TP8"}})
    write(tmp_path / "native_inputs.json", {"models": {"huge_graph": ["DO NOT PUBLISH"]},
        "hardware": {"HBM4": {"per_gpu_table": {"capacity": 288},
            "native_hbm_config": {"bandwidth": 12800}, "other_unneeded": "omit"}}})
    write(tmp_path / "run_manifest.json", {"batch_sizes": [1, 8], "layout": {"dp": 1, "tp": 8, "pp": 1, "ep": 1}})
    configuration = report.build_snapshot(tmp_path)["configuration"]
    assert configuration["models"] == {"405b": {"layers": 126}}
    assert configuration["hardware"]["HBM4"]["native_hbm_config"]["bandwidth"] == 12800
    assert configuration["run"]["batch_sizes"] == [1, 8]
    assert configuration["run"]["layout"] == {"dp": 1, "tp": 8, "pp": 1, "ep": 1}
    assert "DO NOT PUBLISH" not in json.dumps(configuration)


def test_publish_writes_only_own_reports_not_manager_files_or_markers(tmp_path):
    status = {"state": "running"}
    write(tmp_path / "status.json", status)
    (tmp_path / "report.md").write_text("manager owned", encoding="utf-8")
    (tmp_path / "stop.request").write_text("preserve", encoding="utf-8")
    result = report.write_snapshot(tmp_path)
    assert json.loads((tmp_path / "progress_analysis.json").read_text(encoding="utf-8")) == result
    assert (tmp_path / "progress_analysis.md").exists()
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8")) == status
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == "manager owned"
    assert (tmp_path / "stop.request").read_text(encoding="utf-8") == "preserve"


def test_manager_final_state_overrides_old_eta(tmp_path):
    write(tmp_path / "status.json", {"state": "completed_with_rejections"})
    write(tmp_path / "eta_assessment.json", {"summary": "旧估计", "remaining_min_seconds": 100})
    result = report.build_snapshot(tmp_path)
    assert result["eta"]["remaining_min_seconds"] == result["eta"]["remaining_max_seconds"] == 0.


def test_temporarily_incomplete_json_is_unknown_not_a_fabricated_failure(tmp_path):
    (tmp_path / "status.json").write_text("{", encoding="utf-8")
    snapshot = report.build_snapshot(tmp_path)
    assert snapshot["status"].get("state") is None
    assert any("暂时无法读取" in note for note in snapshot["notes"])


def test_paused_attempt_keeps_real_counters_and_does_not_publish_stale_eta(tmp_path):
    write(tmp_path / "status.json", {"state": "paused", "running": [], "counts": {"paused": 1}})
    write(tmp_path / "eta_assessment.json", {"summary": "旧估计", "current_batch_estimates": [
        {"case_id": "a", "remaining_min_seconds": 5., "remaining_max_seconds": 10.}]})
    (tmp_path / "stop.request").write_text("user paused", encoding="utf-8")
    write(tmp_path / "progress/a.json", {"state": "paused", "batch_size": 1, "wall_seconds": 28000.,
        "cumulative_progress": {"cohorts": 2, "prefill_compute_tokens": 1,
            "visible_output_tokens": 2, "decode_cohorts": 1, "target_decode_cohorts": 1116,
            "warmup_decode_cohorts": 1, "measurement_decode_cohorts": 0},
        "wall_breakdown_seconds": {"cohort_lowering": 21000., "cohort_tasks": 7000.}})
    snapshot = report.build_snapshot(tmp_path)
    assert snapshot["running"] == []
    assert snapshot["paused"][0]["cumulative_progress_available"] is True
    assert snapshot["status"]["pause_requested"] is True
    assert snapshot["eta"]["remaining_min_seconds"] is None
    assert "current_batch_estimates" not in snapshot["eta"]
    markdown = report.render_markdown(snapshot)
    assert "已暂停尝试" in markdown
    assert "21000.000" in markdown and "批次任务图生成" in markdown
    assert "1/1116" in markdown


def test_old_paused_attempt_never_infers_tokens_from_completed_tasks(tmp_path):
    write(tmp_path / "status.json", {"state": "paused"})
    write(tmp_path / "progress/a.json", {"state": "paused", "stage": "cohort_tasks",
        "completed": 112640, "total": 123197})
    snapshot = report.build_snapshot(tmp_path)
    row, = snapshot["paused"]
    assert row["cumulative_progress_available"] is False
    assert "cumulative_progress" not in row
    assert "未知" in report.render_markdown(snapshot)


def test_online_layer_result_keeps_bounds_without_repeating_physical_ledgers(tmp_path):
    row = point("HBF")
    selected = row["selected"]
    selected["native_window_forecast"] = {"sampling_unit": "layer_windows",
        "metric_bounds": {"mean_complete_request_tpot_ns": {"lower": 10., "upper": 11.}},
        "error_budget": {"large_ledger": ["resource"] * 10000}}
    selected["physical_layer_samples"] = [{"executed_task_count": 99}]
    selected["approximation"] = {"physical_ingress_accounting": {"large_ledger": ["resource"] * 10000}}
    write(tmp_path / "results.json", {"cases": [row]})
    snapshot = report.build_snapshot(tmp_path)
    published = snapshot["cases"][0]["selected"]
    assert published["metrics"] == selected["metrics"]
    assert published["native_window_forecast"]["metric_bounds"] == selected["native_window_forecast"]["metric_bounds"]
    assert published["physical_layer_summary"] == {"completed_windows": 1, "executed_tasks": 99,
        "complete_cohorts": 0, "visible_output_tokens": 0}
    assert "large_ledger" not in json.dumps(published)


def test_optimization_diagnostics_do_not_count_as_formal_candidate_results(tmp_path):
    write(tmp_path / "status.json", {"state": "paused", "counts": {"completed": 0}})
    write(tmp_path / "actual_runtime_profile/summary.json", {"schema": "native-runtime-diagnostics/v1",
        "rows": [{"id": "probe", "hardware": "HBF", "workload": "SHORT", "batch_size": 1,
            "status": "forecast_rejected", "wall_seconds": 147., "layer_windows": 4,
            "cohort_relative_budget": .66}], "findings": ["正式扫描仍暂停。"]})
    snapshot = report.build_snapshot(tmp_path)
    assert snapshot["completed_results"] == [] and snapshot["saved_candidate_counts"] == {}
    assert snapshot["optimization"]["rows"][0]["layer_windows"] == 4
    assert "优化诊断（不计正式结果）" in report.render_markdown(snapshot)


def test_live_diagnostic_reads_real_progress_and_terminal_without_changing_scan(tmp_path, monkeypatch):
    write(tmp_path / "status.json", {"state": "paused", "counts": {"completed": 0}})
    write(tmp_path / "actual_runtime_profile/summary.json", {"schema": "native-runtime-diagnostics/v1",
        "rows": [{"id": "probe", "case_id": "actual", "hardware": "HBF", "workload": "LONG",
            "batch_size": 1, "status": "running", "wall_seconds": 0., "layer_windows": 0}]})
    write(tmp_path / "actual_runtime_profile/probe/progress/actual.json", {"pid": 456,
        "stage": "serving_boundary", "candidate_wall_seconds": 120.,
        "cumulative_progress": {"external_submitted_target_write_bytes": 3*2**30,
                                "external_imported_requests": 0}})
    monkeypatch.setattr(report, "_process_info", lambda _pid: {"alive": True})
    snapshot = report.build_snapshot(tmp_path)
    row, = snapshot["optimization"]["rows"]
    assert row["wall_seconds"] == 120. and row["submitted_target_write_gib"] == 3.
    assert row["complete_imported_requests"] == 0 and row["status"] == "running"
    assert snapshot["status"]["state"] == "paused" and snapshot["completed_results"] == []
    write(tmp_path / "actual_runtime_profile/probe/result.json", {"status": "forecast_rejected",
        "wall_seconds": 700., "metrics": None, "rejection_reason": "budget_exceeded"})
    row, = report.build_snapshot(tmp_path)["optimization"]["rows"]
    assert row["status"] == "forecast_rejected" and row["wall_seconds"] == 700.


def test_completed_rows_carry_the_selected_layout_and_baseline_ignores_it(tmp_path):
    """Each store is compared at its own best layout and batch, not at a fixed one."""
    baseline = point("HBM4", layout=(1, 8, 1, 1), batch=8)
    other = point("HBF", tps=200., tpot=15e6, layout=(2, 2, 1, 2), batch=16)
    write(tmp_path / "results.json", {"cases": [baseline, other]})
    a, b = report.build_snapshot(tmp_path)["completed_results"]
    assert [(a[key], b[key]) for key in ("dp", "tp", "pp", "ep", "batch_size")] == [
        (1, 2), (8, 2), (1, 1), (1, 2), (8, 16)]
    assert (b["throughput_ratio_vs_hbm4"], b["tpot_ratio_vs_hbm4"], b["baseline_batch_size"]) == (2., 1.5, 8)
    assert a["throughput_ratio_vs_hbm4"] == 1.
    markdown = report.render_markdown(report.build_snapshot(tmp_path))
    assert "选定布局DP/TP/PP/EP" in markdown and "|2/2/1/2|16|" in markdown and "|1/8/1/1|8|" in markdown


def test_running_candidates_are_read_by_candidate_id_so_same_case_candidates_do_not_collide(tmp_path):
    ids = ["c__DP1_TP8_PP1_EP1__B1", "c__DP2_TP4_PP1_EP1__B8"]
    write(tmp_path / "status.json", {"state": "running", "running": [
        {"case_id": "c", "candidate_id": identity, "launcher_pid": 1 + index} for index, identity in enumerate(ids)]})
    for index, identity in enumerate(ids):
        write(tmp_path / "progress" / (identity + ".json"), {"case_id": "c", "candidate_id": identity,
            "pid": 10 + index, "batch_size": 1 + 7 * index, "stage": "cohort_tasks", "completed": index, "total": 5})
    write(tmp_path / "progress" / "d__DP1_TP8_PP1_EP1__B1.json", {"case_id": "d", "state": "paused", "batch_size": 1})
    snapshot = report.build_snapshot(tmp_path)
    assert [(row["case_id"], row["candidate_id"], row["pid"]) for row in snapshot["running"]] == [
        ("c", ids[0], 10), ("c", ids[1], 11)]
    paused, = snapshot["paused"]
    assert (paused["case_id"], paused["candidate_id"]) == ("d", "d__DP1_TP8_PP1_EP1__B1")
    markdown = report.render_markdown(snapshot)
    assert ids[0] in markdown and ids[1] in markdown and "d__DP1_TP8_PP1_EP1__B1" in markdown
