"""A native forecast never relabels its physical samples as a full window."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import RunManifest, SIMULATION_SCHEMA_VERSION
from heterollm_sim.native_fast_window import NativeLayerWindowSample, NativeWindowForecast
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim import reporting
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig


def layer_sample(role="calibration", window_id="stationary-calibration"):
    return NativeLayerWindowSample(window_id, role, "cohort-000000", "prefill", 0,
        ("dense0",), (0, 1), {"request0": 1660}, 27, 10., 200., .5,
        {"dram_traffic": {"physical_bytes": 512, "burst_count": 8}})


def result(accepted):
    scenario = replace(build_reference_scenario(), simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig()))
    manifest = RunManifest(schema_version=SIMULATION_SCHEMA_VERSION,
        run_id="native-window-report", random_seed=0, simulator_version="test",
        model_name=scenario.model.name, hardware_name=scenario.hardware.name,
        workload_name=scenario.workload.name)
    forecast = NativeWindowForecast(200., 0, 1111, 0, 1111,
        {"warmup_decode_cohorts": 372, "measurement_decode_cohorts": 744},
        {"duration_ns": 74400.},
        {"cluster_tokens_per_second": 1.e7, "mean_complete_request_tpot_ns": 100.,
         "complete_fresh_requests": 2},
        {"mean_complete_request_tpot_ns": {"lower_ns": 99., "upper_ns": 101.}},
        ({"request_id": "fresh", "ttft_ns": 200., "tpot_ns": 100.},),
        {"accepted": True}) if accepted else None
    return reporting.FastScenarioResult(scenario, manifest,
        {"effective_mode": "fast" if accepted else "forecast_rejected",
         "target_relative_error": .1, "sampled_decode_cohorts": 0,
         "predicted_decode_cohorts": 1111 if accepted else 0,
         "rejection_reasons": [] if accepted else ["native_stationary_independent_budget_exceeded"],
         "physical_ingress_accounting": {"read_bytes": 64, "write_bytes": 64,
                                         "scope": "paid_native_PD_only"}},
        (), "aggregate", native_window_forecast=forecast,
        layer_window_samples=(layer_sample(), layer_sample("holdout", "stationary-holdout")))


@pytest.mark.parametrize("accepted", (False, True))
def test_layer_samples_never_become_observed_completed_cohorts_or_tokens(accepted):
    value = result(accepted)
    payload = reporting._fast_report_dict(value, include_visualization=False)
    assert "physical_prefix" not in payload
    assert payload["result_semantics"]["complete_physical_decode_cohorts_observed"] == 0
    assert payload["result_semantics"]["visible_output_tokens_observed"] == 0
    assert len(payload["physical_layer_samples"]) == 2
    assert all(not row["complete_native_cohort"] and row["visible_output_tokens_observed"] == 0
               for row in payload["physical_layer_samples"])
    assert payload["physical_ingress_accounting"]["read_bytes"] == 64
    assert payload["dram_traffic"]["full_window_available"] is False
    assert payload["summary"]["task_count"] is None
    if accepted:
        assert payload["requests"]["fresh"]["status"] == "forecast_completed"
        assert payload["summary"]["completed_requests_semantics"] == "forecast"
        assert payload["native_window_forecast"]["forecast_has_physical_trace"] is False
        assert payload["native_window_forecast"]["controller_state"] == "discarded_after_independent_native_layer_samples"
    else:
        assert payload["requests"] == {} and payload["native_window_forecast"] is None
        assert payload["summary"]["completed_requests"] is None
    text = reporting.format_report(value, data=payload)
    assert "真实层样本 2" in text
    assert "完整Decode批次和可见输出Token实测均为0" in text
    assert "不是完整窗口实测误差保证" in text
    assert ("预算结果：接受" if accepted else "native_stationary_independent_budget_exceeded") in text


def test_report_rejects_a_fabricated_complete_serving_prefix(monkeypatch):
    value = result(False)
    prefix = reporting.OnlineScenarioResult(value.scenario, object(), value.manifest, (), "aggregate")
    value = replace(value, sampled=prefix)
    monkeypatch.setattr(reporting, "_report_dict_in_context",
        lambda *_args, **_kwargs: pytest.fail("partial cohort must be rejected before serving report generation"))
    with pytest.raises(ValueError, match="partial cohort"):
        reporting._fast_report_dict(value)


def test_forecast_cannot_claim_a_completed_decode_token_or_cohort():
    value = result(True).native_window_forecast
    with pytest.raises(ValueError, match="cannot claim completed"):
        replace(value, sampled_decode_cohorts=1)
    with pytest.raises(ValueError, match="cannot claim completed"):
        replace(value, sampled_visible_output_tokens=1)


@pytest.mark.parametrize("envelope", [True, False])
def test_public_native_result_keeps_manifest_and_sample_scope_separate(monkeypatch, envelope):
    from heterollm_sim.fast_simulation import FastSimulationOutcome
    from test_native_external_kv import ingress_scenario

    options = NativeWindowFastConfig(calibration_layers=6, holdout_layers=3,
                                    pd_concurrency_envelope=envelope)
    scenario = replace(ingress_scenario(count=1), simulation=SimulationConfig(mode="fast", native_window=options))
    forecast = result(True).native_window_forecast
    samples = (layer_sample(),)
    monkeypatch.setattr(reporting, "simulate_fast", lambda *_args, **_kwargs:
        FastSimulationOutcome(native_window_forecast=forecast, layer_window_samples=samples,
            diagnostics={"effective_mode": "fast", "target_relative_error": .1,
                         "physical_complete_cohort_count": 0}))
    value = reporting.run_scenario(scenario)
    assert value.layer_window_samples == samples
    assert value.sampled is value.detailed is None
    metadata = value.manifest.metadata["native_sampling"]
    assert metadata["model"] == "stationary_continuous_refill"
    assert (metadata["calibration_layers"], metadata["holdout_layers"]) == (6, 3)
    if envelope:
        assert metadata["pd_concurrency_envelope"] is True
    else:
        assert "pd_concurrency_envelope" not in metadata
    assert "not_complete_cohorts_or_visible_tokens" in metadata["layer_sample_scope"]
    payload = reporting.report_dict(value, include_visualization=False)
    assert payload["result_semantics"]["complete_physical_decode_cohorts_observed"] == 0


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
    ("rank_ids", (True,)), ("executed_task_count", 0), ("role", "validation"),
    ("end_ns", 9.), ("wall_seconds", float("inf")), ("context_tokens_by_request", {})])
def test_physical_layer_sample_requires_real_complete_window_observations(field, value):
    with pytest.raises(ValueError):
        replace(layer_sample(), **{field: value})


@pytest.mark.parametrize("options", ({"calibration_layers": 1}, {"calibration_layers": 3}, {"holdout_layers": 1},
    {"tp_rank_symmetry": "invalid"},
    {"max_wall_seconds": 0.}, {"max_wall_seconds": float("inf")}, {"target_relative_error": .2}))
def test_stationary_options_keep_the_sampling_and_budget_limits(options):
    with pytest.raises(ValueError):
        NativeWindowFastConfig(**options)


def test_decode_defaults_are_class_covering_minimum_counts():
    options = NativeWindowFastConfig()
    assert (options.calibration_layers, options.holdout_layers) == (4, 2)
    from heterollm_sim.native_stationary import _sample_roles
    inventory = [{"layer_id": str(i), "class_key": ("dense", "chunked", "full")[i % 3]}
                 for i in range(12)]
    roles = _sample_roles(inventory, (4, 2))
    assert [len(roles[key]) for key in ("calibration", "holdout", "pd_import")] == [4, 4, 2]
    for role in ("calibration", "holdout"):
        assert {inventory[int(i)]["class_key"] for i in roles[role][1:]} == {"dense", "chunked", "full"}
