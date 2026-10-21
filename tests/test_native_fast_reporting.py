"""A native forecast never relabels its physical sample as a full window."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import RunManifest, SIMULATION_SCHEMA_VERSION
from heterollm_sim.native_fast_window import NativeLayerWindowSample, NativeWindowForecast
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim import reporting
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig


def result(accepted):
    scenario = replace(build_reference_scenario(), simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig()))
    manifest = RunManifest(schema_version=SIMULATION_SCHEMA_VERSION,
        run_id="native-window-report", random_seed=0, simulator_version="test",
        model_name=scenario.model.name, hardware_name=scenario.hardware.name,
        workload_name=scenario.workload.name)
    forecast = NativeWindowForecast(200., 5, 1111, 6, 1111,
        {"warmup_decode_cohorts": 372, "measurement_decode_cohorts": 744},
        {"duration_ns": 74400.},
        {"cluster_tokens_per_second": 1.e7, "mean_complete_request_tpot_ns": 100.,
         "complete_fresh_requests": 2},
        {"mean_complete_request_tpot_ns": {"lower_ns": 99., "upper_ns": 101.}},
        ({"request_id": "fresh", "ttft_ns": 200., "tpot_ns": 100.},),
        {"accepted": True}) if accepted else None
    return reporting.FastScenarioResult(scenario, manifest,
        {"effective_mode": "fast" if accepted else "forecast_rejected",
         "target_relative_error": .1, "native_window_prefix_only": True,
         "sampled_decode_cohorts": 5, "predicted_decode_cohorts": 1111 if accepted else 0,
         "rejection_reasons": [] if accepted else ["future_pd_budget_unavailable"]},
        (), "aggregate", native_window_forecast=forecast)


@pytest.mark.parametrize("accepted", [False, True])
def test_native_window_report_distinguishes_forecast_from_physical_prefix(accepted, monkeypatch):
    value = result(accepted)
    prefix = reporting.OnlineScenarioResult(value.scenario, object(), value.manifest, (), "aggregate")
    value = replace(value, sampled=prefix)
    real = {"summary": {"completed_requests": 0, "task_count": 222},
            "requests": {"fresh": {"status": "running"}},
            "dram_traffic": {"physical_bytes": 1234}}
    monkeypatch.setattr(reporting, "_report_dict_in_context",
        lambda item, **_kwargs: real if item is prefix else pytest.fail("not the actual sample"))
    payload = reporting._fast_report_dict(value, include_visualization=False)
    assert payload["physical_prefix"] is real
    assert payload["summary"]["task_count"] is None
    assert payload["dram_traffic"]["full_window_available"] is False
    assert payload["physical_prefix"]["requests"]["fresh"]["status"] == "running"
    assert value.detailed is value.forecast is None
    if accepted:
        assert payload["requests"]["fresh"]["status"] == "forecast_completed"
        assert payload["summary"]["completed_requests_semantics"] == "forecast"
        assert payload["native_window_forecast"]["forecast_has_physical_trace"] is False
    else:
        assert payload["requests"] == {} and payload["native_window_forecast"] is None
        assert payload["summary"]["completed_requests"] is None
    text = reporting.format_report(value, data=payload)
    assert "不是完整窗口实测误差保证" in text
    assert ("预算结果：接受" if accepted else "future_pd_budget_unavailable") in text


def layer_sample(role="calibration", window_id="layer-window-0"):
    return NativeLayerWindowSample(window_id, role, "cohort-000000", "decode", 0,
        ("dense0",), (0, 1), {"request0": 1660}, 27, 10., 200., .5,
        {"dram_traffic": {"physical_bytes": 512, "burst_count": 8}})


@pytest.mark.parametrize("accepted", (False, True))
def test_layer_samples_never_become_observed_completed_cohorts_or_tokens(accepted):
    value = result(accepted)
    scenario = replace(value.scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(sampling_unit="layer_windows")))
    forecast = replace(value.native_window_forecast, sampled_decode_cohorts=0,
        sampled_visible_output_tokens=0, sampling_unit="layer_windows") if accepted else None
    value = replace(value, scenario=scenario,
        approximation={**value.approximation, "sampling_unit": "layer_windows",
            "sampled_decode_cohorts": 0, "sampled_visible_output_tokens": 0,
            "physical_ingress_accounting": {"read_bytes": 64, "write_bytes": 64,
                "scope": "paid_native_PD_only"}},
        native_window_forecast=forecast,
        layer_window_samples=(layer_sample(), layer_sample("validation", "layer-window-1")))
    payload = reporting._fast_report_dict(value, include_visualization=False)
    assert payload["physical_prefix"] is None
    assert payload["result_semantics"]["sampling_unit"] == "layer_windows"
    assert payload["result_semantics"]["complete_physical_decode_cohorts_observed"] == 0
    assert payload["result_semantics"]["visible_output_tokens_observed"] == 0
    assert len(payload["physical_layer_samples"]) == 2
    assert all(not row["complete_native_cohort"] and row["visible_output_tokens_observed"] == 0
               for row in payload["physical_layer_samples"])
    assert payload["physical_ingress_accounting"]["read_bytes"] == 64
    assert payload["dram_traffic"]["full_window_available"] is False
    if accepted:
        assert payload["native_window_forecast"]["sampling_unit"] == "layer_windows"
        assert payload["native_window_forecast"]["controller_state"] == "discarded_after_boundary_conditioned_layer_windows"
        assert payload["summary"]["completed_requests_semantics"] == "forecast"
    else:
        assert payload["summary"]["completed_requests"] is None
        assert payload["native_window_forecast"] is None
    text = reporting.format_report(value, data=payload)
    assert "真实层窗口 2" in text
    assert "完整Decode批次和可见输出Token实测均为0" in text
    assert "不是完整窗口实测误差保证" in text


def test_layer_report_rejects_a_fabricated_complete_serving_prefix(monkeypatch):
    value = result(False)
    scenario = replace(value.scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(sampling_unit="layer_windows")))
    prefix = reporting.OnlineScenarioResult(scenario, object(), value.manifest, (), "aggregate")
    value = replace(value, scenario=scenario, sampled=prefix)
    monkeypatch.setattr(reporting, "_report_dict_in_context",
        lambda *_args, **_kwargs: pytest.fail("partial layer cohort must be rejected before serving report generation"))
    with pytest.raises(ValueError, match="partial cohort"):
        reporting._fast_report_dict(value)


def test_layer_forecast_cannot_claim_a_completed_decode_token_or_cohort():
    value = result(True).native_window_forecast
    with pytest.raises(ValueError, match="cannot claim completed"):
        replace(value, sampling_unit="layer_windows")


def test_exhaustive_layer_class_is_not_relabelled_as_independent_holdout():
    row = replace(layer_sample(), role="exhaustive").to_metadata()
    assert row["validation_semantics"] == "all_native_members_of_this_layer_class_physically_executed"
    assert not row["complete_native_cohort"] and row["visible_output_tokens_observed"] == 0


def test_public_native_layer_result_keeps_manifest_and_sample_scope_separate(monkeypatch):
    from heterollm_sim.fast_simulation import FastSimulationOutcome
    from test_native_external_kv import ingress_scenario

    options = NativeWindowFastConfig(sampling_unit="layer_windows")
    scenario = replace(ingress_scenario(count=1), simulation=SimulationConfig(mode="fast", native_window=options))
    forecast = replace(result(True).native_window_forecast, sampling_unit="layer_windows",
        sampled_decode_cohorts=0, sampled_visible_output_tokens=0)
    samples = (layer_sample(),)
    monkeypatch.setattr(reporting, "simulate_fast", lambda *_args, **_kwargs:
        FastSimulationOutcome(native_window_forecast=forecast, layer_window_samples=samples,
            diagnostics={"effective_mode": "fast", "sampling_unit": "layer_windows",
                "target_relative_error": .1, "physical_complete_cohort_count": 0}))
    value = reporting.run_scenario(scenario)
    assert value.layer_window_samples == samples
    assert value.sampled is value.detailed is None
    metadata = value.manifest.metadata["native_sampling"]
    assert metadata["unit"] == "layer_windows"
    assert metadata["max_total_layer_windows"] == 32
    assert metadata["max_full_batch_layer_windows"] == metadata["max_seed_shape_anchor_layer_windows"] == 16
    assert metadata["calibration_layer_windows_per_class"] == 2
    assert metadata["validation_layer_windows_per_class"] == 2
    assert "not_complete_cohorts_or_visible_tokens" in metadata["layer_sample_scope"]
    payload = reporting.report_dict(value, include_visualization=False)
    assert payload["physical_prefix"] is None
    assert payload["result_semantics"]["sampling_unit"] == "layer_windows"


def test_forecast_PD_label_permutations_preserve_multiple_logical_request_slots():
    value = result(True)
    slots = tuple({"request_id": None, "logical_fifo_slot_id": identity,
        "per_request_label_order_proven": False, "forecast_only": True,
        "tpot_ns": {"estimate": 100., "lower": 99., "upper": 101.}}
        for identity in ("slot0", "slot1"))
    value = replace(value, native_window_forecast=replace(value.native_window_forecast,
        request_forecasts=slots))
    payload = reporting._fast_report_dict(value, include_visualization=False)
    assert set(payload["requests"]) == {"slot0", "slot1"}
    assert all(row["request_id"] is None and row["status"] == "forecast_completed"
               for row in payload["requests"].values())


def test_forecast_duplicate_logical_slots_are_rejected():
    value = result(True)
    slot = {"request_id": None, "logical_fifo_slot_id": "duplicate"}
    value = replace(value, native_window_forecast=replace(value.native_window_forecast,
        request_forecasts=(slot, slot)))
    with pytest.raises(ValueError, match="distinct"):
        reporting._fast_report_dict(value, include_visualization=False)


@pytest.mark.parametrize("field,value", [("layer_ids", ()), ("rank_ids", ()),
    ("rank_ids", (True,)), ("executed_task_count", 0), ("role", "predicted"),
    ("end_ns", 9.), ("wall_seconds", float("inf")), ("context_tokens_by_request", {})])
def test_physical_layer_sample_requires_real_complete_window_observations(field, value):
    with pytest.raises(ValueError):
        replace(layer_sample(), **{field: value})
