"""Report numerical units distinguish repeat scatter, errors and host service."""
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "tools/render_cuda_graph_validation_report.py"
spec = importlib.util.spec_from_file_location("cuda_graph_report_render", path)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def native(mode, times_ms):
    return {"status": "completed", "configuration": {"cuda_graphs_mode_requested": mode,
        "effective_env": {"GGML_CUDA_DISABLE_GRAPHS": "1" if mode == "off" else None}},
        "warmups": [{}, {}], "samples": [{"status": "completed", "prompt_n": 512,
            "visible_output_tokens": 128, "cache_n": 0,
            **{key: time * 1e6 for key, _ in renderer.METRICS}} for time in times_ms]}


def test_renderer_includes_mad_absolute_error_percentage_points_and_host_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "CASES", [("one", "Qwen example")])
    for mode, measured, predicted in (("off", 100, 120), ("on", 80, 90)):
        write(tmp_path / f"native_one_graph_{mode}.json",
              native(mode, [measured - 20, measured - 10, measured, measured + 10, measured + 100]))
        write(tmp_path / f"comparison_one_graph_{mode}.json", {"status": "compared",
            "metrics": {key: {"simulation_ns": predicted * 1e6,
                "absolute_error_ns": abs(predicted - measured) * 1e6,
                "signed_error_percent": (predicted - measured) / measured * 100}
                for key, _ in renderer.METRICS}})
    write(tmp_path / "host_cost_sensitivity_one_graph_on.json", {
        "schema": "heterollm.cuda-graph-host-cost-sensitivity/v1",
        "scope": "all_completed_report_invocations_including_startup_and_warmup",
        "invocation_count": 5, "event_count": 8, "event_counts": {"replay_submit": 8},
        "host_service_totals_ns": {"sum_of_event_medians_ns": 4e6,
            "sum_of_event_minima_ns": 3e6, "sum_of_event_maxima_ns": 9e6},
        "prediction_qualified": False, "end_to_end_sensitivity_computed": False})
    html = renderer.render(tmp_path).read_text(encoding="utf-8")
    summary = renderer.read(tmp_path / "comparison_summary.json")
    gain = summary["graph_speedups"][0]
    assert gain["native_e2e_reduction_percent"] == pytest.approx(20)
    assert gain["simulation_e2e_reduction_percent"] == pytest.approx(25)
    assert gain["signed_reduction_error_percentage_points"] == pytest.approx(5)
    assert gain["native_speedup_ratio"] == pytest.approx(1.25)
    assert gain["simulation_speedup_ratio"] == pytest.approx(120 / 90)
    mode = summary["cases"][0]["modes"]["on"]
    assert mode["native"]["engine_e2e_ns"]["mad_ns"] == 10e6
    assert mode["host_cost_sensitivity"]["host_service_totals_ns"]["sum_of_event_medians_ns"] == 4e6
    for text in ("MAD 10.000 ms", "绝对误差 10.000 ms", "+12.50%", "+5.00 个百分点",
                 "包含启动探测、预热和正式请求", "不是请求 E2E", "早期通用插值探索（失败记录）",
                 "capture 计时夹带了节点数量查询", "不构成跨模型家族"):
        assert text in html


def test_speedup_stays_unavailable_until_both_predictions_are_valid():
    modes = {mode: {"native": {"engine_e2e_ns": {"median_ns": 100}},
                   "comparison": {"status": "configuration_mismatch"}}
             for mode in ("off", "on")}
    gain = renderer.graph_speedup(modes)
    assert gain["simulation_e2e_reduction_percent"] is None
    assert gain["signed_reduction_error_percentage_points"] is None


def test_host_summary_refuses_an_e2e_interval_disguised_as_host_service(tmp_path):
    path = tmp_path / "host.json"
    write(path, {"schema": "heterollm.cuda-graph-host-cost-sensitivity/v1",
        "scope": "all_completed_report_invocations_including_startup_and_warmup",
        "end_to_end_sensitivity_computed": True})
    with pytest.raises(ValueError, match="scope mismatch"):
        renderer.host_cost_summary(path)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, None, "100"])
def test_native_metrics_rejects_invalid_sample_even_when_median_would_be_valid(tmp_path, value):
    path = tmp_path / "native.json"
    data = native("off", [100] * 5)
    data["samples"][0]["engine_ttft_ns"] = value
    write(path, data)
    with pytest.raises(ValueError, match="positive and finite"):
        renderer.native_metrics(path, "off")
