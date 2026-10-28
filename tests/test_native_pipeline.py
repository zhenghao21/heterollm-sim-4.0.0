"""PP native-window execution uses an independent stage sample and forecast."""
from dataclasses import replace

from heterollm_sim.reporting import FastScenarioResult, _fast_report_dict, format_report, run_scenario
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig
from native_toy import _analytical_toy_hardware, toy_scenario


def pipeline_scenario(*, pp=2, microbatches=2, batch=4):
    model = toy_scenario(layer_count=4).model
    from exploring_hbf_native_config import build_native_comparison_scenario
    scenario = build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", tp=1, pp=pp,
        pipeline_microbatches=microbatches, batch_size=batch, model_override=model, prompt_override=20,
        output_override=3, warmup_override=2, measurement_override=4)
    return _analytical_toy_hardware(scenario, .00001)


def test_pipeline_fast_request_forecasts_from_a_real_pipeline_prefix():
    scenario = pipeline_scenario()
    full = run_scenario(scenario, retention_policy="aggregate")
    options = NativeWindowFastConfig(max_wall_seconds=120)
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=options)), retention_policy="aggregate")
    assert isinstance(fast, FastScenarioResult)
    assert fast.detailed is None, fast.approximation
    assert fast.native_pipeline_forecast is not None and fast.layer_window_samples == ()
    assert full.serving.decode_measurement["complete"]
    diagnostics = fast.approximation
    assert diagnostics["physical_complete_cohort_count"] == 0
    assert diagnostics["physical_visible_output_token_count"] == 0
    assert diagnostics["pipeline_stage_sampling"] is True
    assert diagnostics["stage_sample_task_count"] < diagnostics["stage_sample_total_task_count"]
    assert diagnostics["forecast_used"] is True
    assert diagnostics["predicted_visible_output_tokens"] > 0
    assert fast.sampled is None
    assert "native_sampling" not in fast.manifest.metadata
    assert fast.manifest.metadata["native_pipeline_forecast"]["microbatches"] == 2
    assert fast.native_pipeline_forecast.sampled_decode_cohorts == 4
    assert options.pipeline_probe_decode_cohorts == 2
    payload = _fast_report_dict(fast, include_visualization=False)
    assert payload["result_semantics"]["forecast_used"] is True
    assert payload["native_pipeline_forecast"]["forecast_has_physical_trace"] is False
    from run_exploring_hbf_native import native_fast_result
    params = {"dp": 1, "tp": 1, "pp": 2, "ep": 1, "microbatches": 2,
              "gpu_count": 2, "batch_size": 4, "output_tokens": 3,
              "measurement_decode_cohorts": 4, "target_relative_error": .10}
    row = native_fast_result(fast, params)
    assert row["fidelity"] == "native_pipeline_stage_samples_and_virtual_window_forecast"
    assert row["forecast_tokens"] > 0
    assert row["sampled_tokens"] == 0
    assert row["native_pipeline_forecast"]["forecast_has_physical_trace"] is False


def test_pipeline_wall_budget_rejects_without_forecast_or_completed_result():
    scenario = pipeline_scenario()
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(max_wall_seconds=1e-9))), retention_policy="aggregate")
    assert fast.detailed is fast.native_window_forecast is None
    assert fast.approximation["rejection_reasons"]
    payload = _fast_report_dict(fast)
    assert payload["result_semantics"].get("forecast_used") is False
    assert "原生PP" in format_report(fast, data=payload)


def test_pipeline_stage_sample_budget_rejection_is_reported(monkeypatch):
    """A failed stage-window composition is a forecast rejection, not a crash."""
    import heterollm_sim.native_pipeline as pipeline

    def reject(*_args, **_kwargs):
        raise ValueError("native_pipeline_independent_budget_exceeded")

    monkeypatch.setattr(pipeline, "compose_pipeline_window", reject)
    scenario = pipeline_scenario()
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(max_wall_seconds=120))), retention_policy="aggregate")
    assert fast.detailed is fast.native_window_forecast is None
    assert fast.approximation["rejection_reasons"] == ["native_pipeline_independent_budget_exceeded"]
    assert fast.approximation["fallback_reasons"] == ["native_pipeline_independent_budget_exceeded"]


def test_single_microbatch_pipeline_transfer_gap_is_reported_without_crash():
    """Embedded PP transfers must not escape the forecast rejection path."""
    scenario = pipeline_scenario(microbatches=1)
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(max_wall_seconds=120))), retention_policy="aggregate")
    assert fast.detailed is fast.native_window_forecast is None
    reasons = fast.approximation["rejection_reasons"]
    assert reasons
    assert "native_pipeline_activation_transfer_sample_missing" not in reasons
    assert "native_pipeline_stage_sampling_transfer_tasks_missing" not in reasons


def test_pipeline_prefill_uses_complete_live_forward_including_stage_transfer():
    model = toy_scenario(layer_count=4).model
    from exploring_hbf_native_config import build_native_prefill_scenario
    scenario = build_native_prefill_scenario("llama3_405b", "SHORT", tp=1, pp=2,
        full_pass=True, model_override=model, prompt_override=20)
    scenario = _analytical_toy_hardware(scenario, .00001)
    result = run_scenario(scenario, retention_policy="aggregate")
    passed = result.approximation["prefill_pass"]
    assert passed["accepted"], result.approximation
    bounds = passed["metric_bounds"]["prefill_pass_ns"]
    assert bounds["lower"] == bounds["upper"] > 0
    assert passed["error_budget"]["native_chunk_count"] == 1
    assert passed["error_budget"]["assumptions"] == []
    payload = _fast_report_dict(result)
    assert payload["result_semantics"]["complete_physical_cohorts_observed"] == 1
    assert payload["result_semantics"]["visible_output_tokens_observed"] > 0
    assert payload["result_semantics"]["forecast_used"] is False
    assert "native_sampling" not in result.manifest.metadata
    assert "完整原生前向" in format_report(result, data=payload)


def test_pipeline_slow_paid_import_rejects_an_unfilled_measurement_batch():
    from native_toy import slow_pd_ingress
    scenario = slow_pd_ingress(pipeline_scenario(), .00001)
    result = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(max_wall_seconds=120))), retention_policy="aggregate")
    assert result.detailed is result.native_window_forecast is None
    assert result.approximation["rejection_reasons"] == []
    assert result.approximation["physical_complete_cohort_count"] == 0
    assert result.approximation["physical_visible_output_token_count"] == 0


def test_report_does_not_select_a_saved_pp_point_from_the_old_execution_contract(tmp_path):
    pipeline_scenario()
    import run_exploring_hbf_native as runner
    case = runner.case_matrix()[0]
    layout = runner.make_layout(1, 4, 2, 1)
    identity = runner.candidate_id(case, layout, 4)
    runner.write_json(tmp_path / "points" / (case["case_id"] + ".json"), {
        "case": case, "case_id": case["case_id"], "backend": runner.BACKEND,
        "runner_contract": runner.CONTRACT, "status": "completed",
        "planned_candidate_ids": [identity], "selected": {"old_pp_metrics": True}})
    runner.refresh(tmp_path, [case], {case["case_id"]: [(layout, 4)]})
    point, = runner.read_json(tmp_path / "results.json")["cases"]
    assert point["status"] == "failed" and point["selected"] is None
    runner.write_case_point(tmp_path, case, layout, [4])
    runner.refresh(tmp_path, [case], {case["case_id"]: [(layout, 4)]})
    point, = runner.read_json(tmp_path / "results.json")["cases"]
    assert point["status"] == "running" and point["selected"] is None
