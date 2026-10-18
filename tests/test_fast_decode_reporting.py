"""Public fast-mode results keep forecasts distinct from physical traces."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import RunManifest, SIMULATION_SCHEMA_VERSION
from heterollm_sim.fast_simulation import DecodeForecast
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import (
    FastScenarioResult, OnlineScenarioResult, format_report, online_summary_dict,
    report_dict, run_scenario,
)
from heterollm_sim.run_jobs import RunJobManager, RunJobTraceError
from heterollm_sim.simulation_options import SimulationConfig
from test_hbf_host_staging_integration import strict_staging_scenario


def forecast_result():
    scenario = replace(build_reference_scenario(), simulation=SimulationConfig(mode="fast"))
    manifest = RunManifest(schema_version=SIMULATION_SCHEMA_VERSION, run_id="fast-test",
        random_seed=0, simulator_version="test", model_name=scenario.model.name,
        hardware_name=scenario.hardware.name, workload_name=scenario.workload.name)
    forecast = DecodeForecast("r", 0., 100., 156., 220., 0.,
        1000, 121, 56, 64, 100., 1., (), (), {"physical_state_advanced": False})
    return FastScenarioResult(scenario, manifest,
        {"effective_mode": "fast", "target_relative_error": .1,
         "validation_status": "not_validated_against_full_detailed",
         "sampled_decode_steps": 56, "predicted_decode_steps": 64}, (), "aggregate",
        forecast=forecast)


def test_forecast_report_does_not_invent_complete_physical_metrics():
    result = forecast_result()
    report = report_dict(result)
    assert report["simulation_mode"] == "fast" and report["trace_fidelity"] == "approximate"
    assert report["requests"]["r"]["status"] == "forecast_completed"
    assert report["requests"]["r"]["metric_semantics"]["ttft_ns"] == "detailed"
    assert report["summary"]["task_count"] is None
    assert report["resource_utilization"] == {} and report["batch_history"] == []
    assert report["storage_traffic"]["full_request_available"] is False
    assert report["component_timeseries"]["status"] == "unavailable_for_forecast"
    assert "尚无完整详细对照验证" in format_report(result, data=report)
    assert online_summary_dict(result) == {"summary": report["summary"], "requests": report["requests"]}


def test_explicit_detailed_mode_and_fast_fallback_preserve_real_request_results():
    scenario = strict_staging_scenario(prompt=4, output=2, all_hbf=True, capacity=65536)
    implicit = run_scenario(scenario)
    explicit = run_scenario(replace(scenario, simulation=SimulationConfig(mode="detailed")))
    assert isinstance(implicit, OnlineScenarioResult)
    assert implicit == explicit
    fallback = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast")))
    assert isinstance(fallback, FastScenarioResult) and fallback.forecast is None
    assert fallback.detailed.serving.request_metrics == implicit.serving.request_metrics
    assert fallback.detailed.serving.batches == implicit.serving.batches
    report = report_dict(fallback, include_visualization=False)
    reference = report_dict(implicit, include_visualization=False)
    for field in ("summary", "requests", "scheduler", "storage_traffic", "dram_traffic"):
        assert report[field] == reference[field]
    assert report["approximation"]["effective_mode"] == "detailed"
    assert report["manifest"]["metadata"]["effective_simulation_mode"] == "detailed"
    assert "使用详细执行" in format_report(fallback, data=report)


def test_background_trace_request_explains_forecast_and_unwraps_detailed_fallback():
    result = forecast_result()
    with RunJobManager(max_workers=1) as manager:
        manager._jobs["forecast"] = SimpleNamespace(status="completed", result=result)
        with pytest.raises(RunJobTraceError) as caught:
            manager._trace_result_locked("forecast")
        assert caught.value.code == "fast_forecast_has_no_physical_trace"
        detailed = OnlineScenarioResult(result.scenario, None, result.manifest, (), "aggregate")
        manager._jobs["fallback"] = SimpleNamespace(status="completed",
            result=replace(result, forecast=None, detailed=detailed))
        assert manager._trace_result_locked("fallback") is detailed
        # Keep shutdown concerned only with real submitted job records.
        manager._jobs.clear()
