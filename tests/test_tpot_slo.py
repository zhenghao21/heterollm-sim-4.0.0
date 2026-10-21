"""TPOT is a mean visible-token interval, independent of a maximum TBT SLO."""
from dataclasses import replace

import pytest

from heterollm_sim.config import scheduler_from_dict
from heterollm_sim.contracts import RunManifest
from heterollm_sim.ir import RequestSpec, SchedulerSpec
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import (
    OnlineScenarioResult, _online_report_core, _slo_semantics, report_dict,
)
from heterollm_sim.serde import to_primitive
from heterollm_sim.serving import (
    BatchCost, _OnlineRuntime, compile_serving_plan, evaluate_tpot_slo,
)


def run_intervals(*, tpot=None, tbt=None, output=4, finish_tail_ns=0.0):
    """Exercise real request state/commit accounting with known cohort costs."""
    scenario = build_reference_scenario()
    scenario = replace(scenario, workload=replace(
        scenario.workload, requests=(RequestSpec("one", 0.0, 8, output),),
        request_count=1, mtp=None, scheduler=replace(
            scenario.workload.scheduler, mode="continuous", max_num_seqs=1,
            max_num_batched_tokens=8, prefill_chunk_tokens=8,
            preemption_enabled=False, slo_ttft_ns=None,
            slo_tbt_ns=tbt, slo_tpot_ns=tpot,
        ),
    ))
    durations = iter((10.0, 20.0, 80.0, 10.0))
    runner = _OnlineRuntime(compile_serving_plan(scenario),
                            lambda *_: BatchCost(next(durations)))
    if finish_tail_ns:
        finish = runner._finish
        runner._finish = lambda state, timestamp: finish(state, timestamp + finish_tail_ns)
    serving = runner.run()
    manifest = RunManifest(
        schema_version="4.0.0", run_id="tpot-slo", random_seed=0,
        simulator_version="test", model_name=scenario.model.name,
        hardware_name=scenario.hardware.name, workload_name=scenario.workload.name,
    )
    return OnlineScenarioResult(scenario, serving, manifest, (), "aggregate")


@pytest.mark.parametrize("tpot,tbt,expected_tpot,qualified", [
    (40.0, None, True, 1),     # mean 110/3 passes, while the 80 ns gap exceeds 60
    (None, 60.0, None, 0),
    (40.0, 60.0, True, 0),
    (30.0, None, False, 0),    # mean fails 30, although every gap meets 100
    (None, 100.0, None, 1),
    (30.0, 100.0, False, 0),
    (40.0, 100.0, True, 1),
    (None, None, None, 1),
])
def test_mean_tpot_and_max_tbt_constraints_are_independent(tpot, tbt, expected_tpot, qualified):
    result = run_intervals(tpot=tpot, tbt=tbt)
    metric = result.serving.request_metrics["one"]
    report = _online_report_core(result)
    assert metric.tpot_ns == pytest.approx(110.0 / 3)
    assert metric.last_token_ns == 120.0
    assert metric.slo_tpot_met is expected_tpot
    assert report.requests["one"]["tbt_ns"] == [20.0, 80.0, 10.0]
    assert report.requests["one"]["slo_tpot_met"] is expected_tpot
    assert report.summary["goodput"]["qualified_requests"] == qualified
    assert report.summary["completed_requests"] == 1
    assert report.summary["goodput"]["visible_output_tokens_per_s"] == pytest.approx(
        qualified * 4e9 / 120.0)


@pytest.mark.parametrize("output", [0, 1])
def test_fewer_than_two_visible_tokens_is_undefined_and_not_qualified(output):
    result = run_intervals(tpot=100.0, output=output)
    metric = result.serving.request_metrics["one"]
    report = _online_report_core(result)
    assert metric.tpot_ns is None
    assert metric.slo_tpot_met is None
    assert report.requests["one"]["slo_tpot_met"] is None
    assert report.summary["goodput"]["qualified_requests"] == 0
    assert evaluate_tpot_slo(0.0, output, 100.0) is None


def test_single_token_keeps_old_qualification_when_tpot_filter_is_disabled():
    result = run_intervals(output=1)
    assert _online_report_core(result).summary["goodput"]["qualified_requests"] == 1


def test_post_token_request_finish_tail_does_not_change_mean_tpot():
    result = run_intervals(tpot=40.0, finish_tail_ns=1000.0)
    metric = result.serving.request_metrics["one"]
    assert metric.finish_ns == 1120.0
    assert metric.last_token_ns == 120.0
    assert metric.tpot_ns == pytest.approx(110.0 / 3)
    assert metric.slo_tpot_met is True
    assert _online_report_core(result).summary["goodput"]["qualified_requests"] == 1


def test_tpot_filter_does_not_change_dispatch_or_physical_execution_records():
    baseline = run_intervals()
    filtered = run_intervals(tpot=30.0, tbt=100.0)
    assert filtered.serving.events == baseline.serving.events
    assert filtered.serving.batches == baseline.serving.batches
    assert filtered.serving.runtime_kernel_metrics == baseline.serving.runtime_kernel_metrics
    assert filtered.serving.makespan_ns == baseline.serving.makespan_ns
    assert filtered.serving.kv_metrics == baseline.serving.kv_metrics


def test_scheduler_parse_serialization_and_runtime_policy_propagate_tpot():
    payload = to_primitive(SchedulerSpec(mode="continuous", slo_tpot_ns=100e6))
    parsed = scheduler_from_dict(payload)
    assert parsed.slo_tpot_ns == 100e6
    assert to_primitive(parsed) == payload
    del payload["slo_tpot_ns"]
    assert scheduler_from_dict(payload).slo_tpot_ns is None
    result = run_intervals(tpot=40.0)
    assert result.serving.plan.scheduler.slo_tpot_ns == 40.0
    assert to_primitive(result.serving.request_metrics["one"])["slo_tpot_met"] is True


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf"), True])
def test_tpot_threshold_must_be_positive_finite_number(threshold):
    payload = to_primitive(SchedulerSpec())
    payload["slo_tpot_ns"] = threshold
    with pytest.raises(ValueError, match="slo_tpot_ns"):
        scheduler_from_dict(payload)
    with pytest.raises(ValueError, match="slo_tpot_ns"):
        SchedulerSpec(slo_tpot_ns=threshold)


def test_full_report_exposes_filter_scope_formula_and_undefined_rule():
    result = run_intervals(tpot=40.0)
    data = report_dict(result, include_visualization=False)
    slo = data["result_semantics"]["slo"]
    assert slo["configured"] is True
    assert slo["slo_tpot_ns"] == 40.0
    assert slo["slo_ttft_ns"] is None and slo["slo_tbt_ns"] is None
    assert slo["scheduler_target"] is False
    assert slo["applied_to_goodput"] is True
    assert "visible_output_tokens-1" in slo["tpot_definition"]
    assert "not qualified" in slo["tpot_undefined"]
    assert data["requests"]["one"]["slo_tpot_met"] is True
    static_slo = _slo_semantics(result.scenario, online=False)
    assert static_slo["slo_tpot_ns"] == 40.0
    assert static_slo["applied_to_goodput"] is False


@pytest.mark.parametrize("threshold,expected", [(1.0, True), (0.5, False)])
def test_forecast_report_propagates_tpot_filter_without_claiming_measured_goodput(threshold, expected):
    from test_fast_decode_reporting import forecast_result

    result = forecast_result()
    result = replace(result, scenario=replace(result.scenario, workload=replace(
        result.scenario.workload, scheduler=replace(
            result.scenario.workload.scheduler, slo_tpot_ns=threshold))))
    data = report_dict(result, include_visualization=False)
    assert data["requests"]["r"]["slo_tpot_met"] is expected
    assert data["requests"]["r"]["metric_semantics"]["tpot_ns"] == "estimated_mean"
    slo = data["result_semantics"]["slo"]
    assert slo["slo_tpot_ns"] == threshold
    assert slo["applied_to_goodput"] is False
    assert "estimated mean only" in slo["goodput_filter"]
    assert "goodput" not in data["summary"]
