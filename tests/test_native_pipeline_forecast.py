import pytest

from heterollm_sim.native_pipeline_forecast import (PipelineStageSample, compose_pipeline_window,
    forecast_complete_request_metrics, samples_from_execution_schedule)


def samples():
    rows=[]
    for cohort in range(2):
        for mb in range(2):
            rows.extend([
                PipelineStageSample(cohort,mb,0,0,10),
                PipelineStageSample(cohort,mb,0,10,12,kind="pipeline_activation_transfer"),
                PipelineStageSample(cohort,mb,1,12,27),
            ])
    return rows


def test_pipeline_forecast_preserves_stage_transfer_and_is_virtual():
    result=compose_pipeline_window(samples(),sampled_decode_cohorts=2,
        predicted_decode_cohorts=6,batch_size=4,output_tokens=3)
    assert result.predicted_decode_cohorts == 6
    assert result.predicted_visible_output_tokens == 16
    assert any(row["kind"] == "pipeline_activation_transfer" for row in result.stage_samples)
    assert result.error_budget["physical_runtime_state_advanced"] is False
    assert result.error_budget["conditional_only"] is True
    assert result.to_metadata()["forecast_has_physical_trace"] is False


def test_pipeline_forecast_rejects_wide_sample_interval():
    rows=samples()
    rows.append(PipelineStageSample(2,0,1,0,100))
    with pytest.raises(ValueError,match="independent_budget"):
        compose_pipeline_window(rows,sampled_decode_cohorts=3,predicted_decode_cohorts=4,
            batch_size=2,output_tokens=3)

def test_realized_schedule_rows_bind_microbatch_and_transfer_kind():
    rows=samples_from_execution_schedule((
        {"stage_id":"operator-invocation-group-0001.stage0000","invocation_group_id":"operator-invocation-group-0001","start_ns":1,"end_ns":4,"dependencies":()},
        {"stage_id":"operator-invocation-group-0001.pp0000.activation_transfer","invocation_group_id":"operator-invocation-group-0001","start_ns":4,"end_ns":7,"dependencies":("compute",)},
    ),cohort=2)
    assert rows[0].microbatch == rows[1].microbatch == 1
    assert rows[0].kind == "compute" and rows[1].kind == "pipeline_activation_transfer"

def test_small_complete_window_is_within_the_conditional_ten_percent_budget():
    import sys
    from dataclasses import replace
    sys.path[:0] = ["scripts", "tests"]
    from test_native_pipeline import pipeline_scenario
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig
    scenario = pipeline_scenario()
    full = run_scenario(scenario, retention_policy="aggregate")
    fast = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(max_wall_seconds=120))), retention_policy="aggregate")
    observed = forecast_complete_request_metrics(
        decode_measurement=full.serving.decode_measurement)
    assert observed["accepted"]
    estimate = fast.native_pipeline_forecast.metrics["mean_complete_request_tpot_ns"]
    assert abs(estimate - observed["mean_complete_request_tpot_ns"]) / observed["mean_complete_request_tpot_ns"] <= .10


def test_iteration_interval_cannot_be_used_as_complete_request_tpot():
    result = forecast_complete_request_metrics(
        decode_measurement={"mean_iteration_ns": 10.0}, virtual_requests=())
    assert result["accepted"] is False
    assert result["reason"] == "complete_request_tpot_unavailable"
