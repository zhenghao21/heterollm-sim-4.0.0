"""Report-only regressions: synthetic JSON, real pure validators, no simulation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import multiworkload_architecture_matrix as matrix
from tools import render_multiworkload_results as report


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("report tests must never build scenarios, load GGUF, summarize or run simulation")

    for name in ("build_case", "run_cell", "summarize", "main"):
        monkeypatch.setattr(matrix, name, forbidden)
    for name in ("load_model", "build_scenario", "execute_scenario", "run_scenario"):
        monkeypatch.setattr(matrix.q, name, forbidden)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def make_run(tmp_path, *, kind="topology", cases=("independent", "shared"), values=None):
    load = ["L_test", 2048, 128, 16]
    group = {"id": "P4" if kind == "scratch" else "I2", "question": "report regression", "kind": kind,
             "cases": list(cases), "values": list(values or cases)}
    if kind == "scratch":
        group["direction"] = "nonincreasing"
    manifest = {"source": {"frozen": "test"}, "workloads": [load], "groups": [group]}
    write_json(tmp_path / "manifest.json", manifest)
    return manifest


def add_cell(run_dir, manifest, case, *, value=None, status="SIMULATED", config_exists=True, mutate=None):
    workload = dict(zip(("id", "prompt_tokens", "output_tokens", "batch"), manifest["workloads"][0]))
    config = {"model_config_sha256": "model", "workload": {"requests": 16},
              "hardware": {"metadata": {"phy_noc_mode": case}},
              "profiles": {"components": {"cim/legacy-cim": {"conversion_scratch_capacity_bytes": value}}}}
    if manifest["groups"][0]["kind"] == "scratch":
        config["hardware"]["metadata"]["phy_noc_mode"] = "independent"
    stem = workload["id"] + "__" + case
    observation = {"metrics": {"ttft_ns": 1e6, "tpot_ns": 2e6, "e2e_ns": 3e6},
                   "summary": {"completed_requests": 16, "rejected_requests": 0,
                               "throughput": {"visible_output_tokens_per_s": 100}},
                   "expected_requests": 16, "expected_output_tokens": 2048, "actual_output_tokens": 2048,
                   "reported_resource_bytes": 1024, "complete_resource_accounting": True,
                   "model_sha256": "model", "workload_sha256": "workload", "placement_sha256": "placement",
                   "placement": {"op_to_component": {"mlp": "soc"}, "tensor_to_component": {"kv_cache": "dram0"}}}
    cell = {"id": stem, "variant": case, "status": status, "workload": workload,
            "source_stable": True, "source_sha256": matrix.q.stable_hash(manifest["source"]),
            "observation": observation, "checks": [{"check": "request_completion", "status": "PASS"}]}
    if mutate:
        mutate(cell, config)
    cell["config_sha256"] = matrix.q.stable_hash(config)
    if config_exists:
        write_json(run_dir / "configs" / (stem + ".json"), config)
    write_json(run_dir / "cells" / (stem + ".json"), cell)
    return cell


@pytest.mark.parametrize("status", ["PENDING", "BLOCKED", "HOST_TIMEOUT", "CHECK_FAILED", "FAILED", "INVALIDATED"])
def test_non_simulated_never_displays_performance(tmp_path, status):
    point = {"status": status, "metrics_ms": {"ttft_ms": 123456.75, "tpot_ms": 234567.75, "e2e_ms": None},
             "throughput_tokens_per_s": 345678.75, "resource_accounted_bytes": 456789.75}
    rendered = report.point_html(tmp_path, point)
    assert all(str(value) not in rendered for value in (123456.75, 234567.75, 345678.75, 456789.75))
    assert "— ms" in rendered and "— token/s" in rendered


def test_failed_none_metrics_show_diagnostics_not_predictions(tmp_path):
    manifest = make_run(tmp_path)

    def failed(cell, config):
        obs = cell["observation"]
        obs["metrics"] = {"ttft_ns": 123456750000, "tpot_ns": None, "e2e_ns": None}
        obs["summary"].update(completed_requests=0, rejected_requests=16)
        obs["actual_output_tokens"] = 8
        cell["error"] = None
        cell["checks"] = [{"check": name, "status": "FAIL", "detail": "incomplete"}
                          for name in ("request_completion", "timing_finite")]

    add_cell(tmp_path, manifest, "independent", status="CHECK_FAILED", mutate=failed)
    row = report.summary_from_cells(tmp_path)["rows"][0]
    point = row["points"][0]
    assert point["metrics_ms"] == {} and point["throughput_tokens_per_s"] is None
    assert (point["completed_requests"], point["rejected_requests"], point["actual_output_tokens"],
            point["expected_output_tokens"]) == (0, 16, 8, 2048)
    assert not row["comparable"] and not row["feasible_points_comparable"] and row["trend_check"] is None
    rendered = report.point_html(tmp_path, point)
    assert "request_completion=FAIL" in rendered and "timing_finite=FAIL" in rendered
    assert "实际 8 / 期望 2048" in rendered and "拒绝 16" in rendered
    assert "123456.75" not in rendered
    assert "CHECK_FAILED" in report.conclusion(row)


def test_pending_observation_fields_empty_and_missing_links_inert(tmp_path):
    manifest = make_run(tmp_path)
    row = report.summary_from_cells(tmp_path)["rows"][0]
    point = row["points"][0]
    assert point["status"] == "PENDING" and point["metrics_ms"] == {}
    for field in ("completed_requests", "rejected_requests", "expected_requests", "actual_output_tokens",
                  "expected_output_tokens", "throughput_tokens_per_s", "resource_accounted_bytes", "error"):
        assert point[field] is None
    rendered = report.point_html(tmp_path, point, {"batch": 16, "output_tokens": 128})
    assert "请求：完成" not in rendered and '<a href=' not in rendered
    assert "无有效配置" in rendered and "待生成" not in rendered
    assert '<a href=' not in report.link(tmp_path, "configs/nonexistent.json", "输入配置")
    assert '<a href=' not in report.link(tmp_path, "../outside.json", "输入配置")


def test_p4_missing_blocked_config_preserves_feasible_pair_and_flat_limit(tmp_path):
    cases = ("cim_scratch_262144", "cim_independent", "cim_scratch_4194304")
    manifest = make_run(tmp_path, kind="scratch", cases=cases, values=(262144, 1048576, 4194304))
    for case, value in zip(cases, manifest["groups"][0]["values"]):
        add_cell(tmp_path, manifest, case, value=value,
                 status="BLOCKED" if case == cases[-1] else "SIMULATED", config_exists=case != cases[-1])
    data = report.summary_from_cells(tmp_path)
    row = data["rows"][0]
    assert not row["comparable"] and row["feasible_points_comparable"] and row["fixed_mapping"]
    assert row["compared_cases"] == list(cases[:2]) and row["unexpected_changes"] == []
    assert row["trend_check"]["status"] == "PASS" and not row["trend_check"]["observed_sensitivity"]
    assert row["credibility"] == "PARTIAL_CAPACITY_LIMIT_UNVALIDATED"
    conclusion = report.conclusion(row)
    assert "4 MiB scratch + 2 MiB array = 6 MiB" in conclusion
    assert "2/3" in conclusion and "未观察到参数敏感性" in conclusion and "单槽" in conclusion
    point_html = report.point_html(tmp_path, row["points"][-1])
    assert 'href="configs/' not in point_html and "无有效配置" in point_html


@pytest.mark.parametrize("change,expected", [
    ("hardware", "/hardware/capacity"), ("model", "/model_config_sha256"),
    ("workload", "/workload/requests"), ("mapping", None), ("identity", None),
    ("missing_config", None), ("tampered_config", None), ("missing_timing", None),
    ("missing_identity", None), ("source_mismatch", None),
])
def test_strict_comparability_rejects_undeclared_or_incomplete_points(tmp_path, change, expected):
    manifest = make_run(tmp_path)
    add_cell(tmp_path, manifest, "independent")

    def mutate(cell, config):
        if change == "hardware":
            config["hardware"]["capacity"] = 123
        elif change == "model":
            config["model_config_sha256"] = "other"
        elif change == "workload":
            config["workload"]["requests"] = 2
        elif change == "mapping":
            cell["observation"]["placement"]["op_to_component"]["mlp"] = "other"
        elif change == "identity":
            cell["observation"]["workload_sha256"] = "other"
        elif change == "missing_timing":
            cell["observation"]["metrics"]["tpot_ns"] = None
        elif change == "missing_identity":
            cell["observation"]["model_sha256"] = None
        elif change == "source_mismatch":
            cell["source_sha256"] = "not-the-frozen-source"

    add_cell(tmp_path, manifest, "shared", mutate=mutate, config_exists=change != "missing_config")
    if change == "tampered_config":
        path = tmp_path / "configs/L_test__shared.json"
        config = report.read_json(path)
        config["hardware"]["metadata"]["phy_noc_mode"] = "tampered"
        write_json(path, config)
    row = report.summary_from_cells(tmp_path)["rows"][0]
    assert not row["comparable"] and not row["feasible_points_comparable"]
    if expected:
        assert expected in row["unexpected_changes"]


def test_missing_successful_input_is_not_mislabeled_capacity_block(tmp_path):
    manifest = make_run(tmp_path, cases=("independent", "shared", "missing"))
    for case in manifest["groups"][0]["cases"]:
        add_cell(tmp_path, manifest, case, config_exists=case != "missing")
    row = report.summary_from_cells(tmp_path)["rows"][0]
    assert not row["comparable"] and row["feasible_points_comparable"]
    assert row["credibility"] == "NOT_ACCEPTED"
    assert row["compared_cases"] == ["independent", "shared"]


def test_successful_topology_zero_difference_is_not_contention_validation(tmp_path):
    manifest = make_run(tmp_path)
    for case in ("independent", "shared"):
        add_cell(tmp_path, manifest, case)
    row = report.summary_from_cells(tmp_path)["rows"][0]
    assert row["comparable"] and row["fixed_mapping"]
    assert "未观察到端到端差异" in report.conclusion(row)
    assert "不等于已验证共享争用代价" in report.conclusion(row)


def test_snapshot_retries_a_changed_file_and_deduplicates_shared_cases(tmp_path, monkeypatch):
    manifest = make_run(tmp_path)
    manifest["groups"].append({**deepcopy(manifest["groups"][0]), "id": "I3"})
    write_json(tmp_path / "manifest.json", manifest)
    add_cell(tmp_path, manifest, "independent")
    calls, real_read, replaced = {}, report.read_json, False

    def reading(path):
        nonlocal replaced
        data = real_read(path)
        calls[path] = calls.get(path, 0) + 1
        if path.name == "L_test__independent.json" and path.parent.name == "cells" and not replaced:
            replaced = True
            write_json(path, {**data, "status": "CHECK_FAILED", "error": "replacement during snapshot"})
        return data

    monkeypatch.setattr(report, "read_json", reading)
    data = report.summary_from_cells(tmp_path)
    assert len(data["rows"]) == 2
    assert all(row["points"][0]["status"] == "CHECK_FAILED" for row in data["rows"])
    assert calls[tmp_path / "cells/L_test__independent.json"] == 2  # once per attempt, not per question


def test_cli_modes_write_only_report_outputs_and_keep_original_summary(tmp_path, monkeypatch):
    manifest = make_run(tmp_path)
    for case in ("independent", "shared"):
        add_cell(tmp_path, manifest, case)
    original = {"schema": "legacy", "validation_status": "UNVALIDATED", "rows": []}
    write_json(tmp_path / "summary.json", original)
    frozen = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    monkeypatch.setattr(sys, "argv", ["report", str(tmp_path), "--from-cells"])
    report.main()
    saved = report.read_json(tmp_path / "report_summary.json")
    assert saved["data_source"] == "report_summary.json" and saved["rows"][0]["comparable"]
    html = (tmp_path / "results.html").read_text(encoding="utf-8")
    assert 'href="report_summary.json"' in html and "manifest/cells/configs" in html
    assert "MLP 权重在活动 HBF、KV 仍在 HBM0" in html and "未测试 KV swap" in html
    assert "同一问题、同一负载行内" in html and "I3 不是 CIM 开/关实验" in html
    assert "诊断重跑" in html
    assert all(path.read_bytes() == content for path, content in frozen.items())
    report_bytes = (tmp_path / "report_summary.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["report", str(tmp_path)])
    report.main()
    assert (tmp_path / "report_summary.json").read_bytes() == report_bytes
    assert 'href="summary.json"' in (tmp_path / "results.html").read_text(encoding="utf-8")


def test_user_cancel_removes_point_without_hiding_success(tmp_path):
    manifest = make_run(tmp_path, cases=("independent", "shared", "cancelled"))
    for case in ("independent", "shared"):
        add_cell(tmp_path, manifest, case)
    manifest["cancelled_cells"] = ["L_test__cancelled"]
    write_json(tmp_path / "manifest.json", manifest)
    data = report.summary_from_cells(tmp_path)
    row = data["rows"][0]
    assert len(row["points"]) == 2
    assert not row["comparable"] and row["feasible_points_comparable"]
    assert row["credibility"] == "PARTIAL_USER_CANCELLED_UNVALIDATED"
    assert "用户取消" in report.conclusion(row)
    assert all(p["status"] == "SIMULATED" for p in row["points"])
    assert "运行中快照" not in report.render(tmp_path, data)
    add_cell(tmp_path, manifest, "cancelled")
    with pytest.raises(ValueError, match="隐藏已落盘"):
        report.summary_from_cells(tmp_path)
