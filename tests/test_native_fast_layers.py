"""Layer windows cannot silently turn into completed physical cohorts."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as NS
import sys

import pytest

from heterollm_sim import native_fast_layers as native
from heterollm_sim.execution_control import ExecutionControl, ExecutionCancelledError
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig


def harness(monkeypatch, accepted=True):
    scenario = NS(simulation=SimulationConfig(mode="fast", native_window=NativeWindowFastConfig(
        sampling_unit="layer_windows")), workload=NS(metadata={"external_kv_ingress": {
        "warmup_decode_cohorts": 3, "measurement_decode_cohorts": 6, "require_full_batch": True}}))
    plan = NS(mtp=NS(enabled=False), scheduler=NS(preemption_enabled=False),
              kv_policy=NS(offload_component=None, prefetch_distance=0))
    clock, runtimes = [0.], []
    record = {"window_id": "layer0", "role": "calibration", "cohort_id": "c0",
        "cohort_kind": "prefill", "decode_step": 0, "layer_ids": ("layer0",),
        "rank_ids": (0,), "context_tokens_by_request": {"r0": 7},
        "executed_task_count": 12, "start_ns": 0., "end_ns": 100.,
        "wall_seconds": 1., "physical_observations": {"physical_bytes": 256}}

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.now = 10.
            self.control = kwargs["execution_control"]
            self.batches, self.events = [], []
            runtimes.append(self)

        def run_until_decode_steps(self, target, **kwargs):
            self._native_layer_sampler(self, object(), object(), (), 10.)
            pytest.fail("a partial cohort must not commit")

        def _external_ingress_report(self):
            return {"imported_logical_bytes": 100, "imported_requests": 1}

        def _result(self):
            pytest.fail("layer sampling must not create a sampled ServingResult")

    def sample(*args, **kwargs):
        return {"records": [record]}

    def evaluate(*args, **kwargs):
        return {"accepted": accepted, "reasons": [] if accepted else ["layer_phase_budget_exceeded"],
            "metrics": {"cluster_tokens_per_second": 80., "tokens_per_second_per_gpu": 10.,
                "mean_complete_request_tpot_ns": 100., "mean_complete_request_ingress_ttft_ns": 200.,
                "measurement_duration_ns": 1000., "complete_fresh_requests": 2,
                "measurement_visible_output_tokens": 8},
            "metric_bounds": {}, "window": {}, "error_budget": {},
            "predicted_visible_output_tokens": 12}

    from heterollm_sim.native_layer_sampling import NativeLayerSamplingGuardError
    sampling = NS(sample_native_layer_windows=sample,
                  prepare_native_layer_windows=lambda *args: {"prepared": True},
                  NativeLayerSamplingGuardError=NativeLayerSamplingGuardError)
    monkeypatch.setattr(native, "_compilation_scope", lambda *_args: nullcontext())
    monkeypatch.setattr(native, "compile_serving_plan", lambda _s: plan)
    monkeypatch.setattr(native, "_OnlineRuntime", Runtime)
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    from heterollm_sim import execution_control
    monkeypatch.setattr(execution_control, "perf_counter", lambda: clock[0])
    monkeypatch.setitem(sys.modules, "heterollm_sim.native_layer_sampling", sampling)
    monkeypatch.setitem(sys.modules, "heterollm_sim.native_layer_budget", NS(evaluate_native_layer_window=evaluate))
    return scenario, clock, runtimes, record, sampling


@pytest.mark.parametrize("accepted", (False, True))
def test_layer_windows_keep_paid_ingress_and_samples_without_committing_tokens(monkeypatch, accepted):
    scenario, _, runtimes, _, _ = harness(monkeypatch, accepted)
    outcome = native.simulate_native_layer_windows(scenario, batch_lowerer=object())
    assert outcome.sampled is outcome.detailed is outcome.forecast is None
    assert runtimes[0].batches == runtimes[0].events == []
    assert outcome.layer_window_samples[0].to_metadata()["visible_output_tokens_observed"] == 0
    assert outcome.diagnostics["physical_ingress_accounting"]["imported_logical_bytes"] == 100
    if accepted:
        forecast = outcome.native_window_forecast
        assert forecast.sampling_unit == "layer_windows"
        assert forecast.sampled_decode_cohorts == forecast.sampled_visible_output_tokens == 0
        assert forecast.predicted_decode_cohorts == 9
    else:
        assert outcome.native_window_forecast is None
        assert outcome.diagnostics["rejection_reasons"] == ["layer_phase_budget_exceeded"]


@pytest.mark.parametrize("outer_deadline", (False, True))
def test_layer_sample_timeout_retains_only_fully_completed_windows(monkeypatch, outer_deadline):
    scenario, clock, _, record, sampling = harness(monkeypatch)
    def sample(runtime, cohort, cost, stages, ready, options, control, **kwargs):
        runtime._native_layer_sample_records = [record]
        if outer_deadline:
            raise native.NativeFastWallBudgetError("whole candidate deadline")
        clock[0] = scenario.simulation.native_window.max_wall_seconds + 1.
        control.raise_if_cancelled()
    sampling.sample_native_layer_windows = sample
    outcome = native.simulate_native_layer_windows(scenario, batch_lowerer=object())
    assert outcome.native_window_forecast is None and len(outcome.layer_window_samples) == 1
    assert "native_layer_sample_wall_budget_exhausted" in outcome.diagnostics["rejection_reasons"]


def test_outer_deadline_remains_expired_and_finished_samples_still_survive(monkeypatch):
    scenario, _, _, record, sampling = harness(monkeypatch)
    expired = [False]
    def deadline():
        if expired[0]:
            raise native.NativeFastWallBudgetError("whole candidate deadline")
        return False
    def sample(runtime, cohort, cost, stages, ready, options, control, **kwargs):
        runtime._native_layer_sample_records = [record]
        expired[0] = True
        control.raise_if_cancelled()
    sampling.sample_native_layer_windows = sample
    outcome = native.simulate_native_layer_windows(scenario, batch_lowerer=object(),
        execution_control=ExecutionControl(cancellation_callback=deadline))
    assert outcome.native_window_forecast is None
    assert len(outcome.layer_window_samples) == 1
    assert "native_total_candidate_wall_budget_exhausted" in outcome.diagnostics["rejection_reasons"]


def test_layer_sampling_guard_rejects_but_physical_failure_still_raises(monkeypatch):
    scenario, _, _, record, sampling = harness(monkeypatch)
    def sample(runtime, *args, **kwargs):
        runtime._native_layer_sample_records = [record]
        raise sampling.NativeLayerSamplingGuardError("native_layer_class_coverage_incomplete")
    sampling.sample_native_layer_windows = sample
    outcome = native.simulate_native_layer_windows(scenario, batch_lowerer=object())
    assert len(outcome.layer_window_samples) == 1
    assert outcome.diagnostics["rejection_reasons"] == ["native_layer_class_coverage_incomplete"]
    def broken(*args, **kwargs):
        raise ValueError("physical_controller_failure")
    sampling.sample_native_layer_windows = broken
    with pytest.raises(ValueError, match="physical_controller_failure"):
        native.simulate_native_layer_windows(scenario, batch_lowerer=object())


def test_layer_forecast_uses_default_full_batch_without_key_error(monkeypatch):
    scenario, *_ = harness(monkeypatch)
    del scenario.workload.metadata["external_kv_ingress"]["require_full_batch"]
    outcome = native.simulate_native_layer_windows(scenario, batch_lowerer=object())
    assert outcome.native_window_forecast is not None
    assert outcome.native_window_forecast.sampling_unit == "layer_windows"


@pytest.mark.parametrize("pause", (False, True))
def test_layer_user_pause_wins_and_final_publication_is_inside_wall_budget(monkeypatch, pause):
    scenario, clock, _, _, _ = harness(monkeypatch)
    paused = [False]
    def progress(value):
        if value.stage == "native_layer_forecast_complete":
            clock[0] = scenario.simulation.native_window.max_wall_seconds + 1.
            paused[0] = pause
    def execute():
        return native.simulate_native_layer_windows(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=progress,
                cancellation_callback=lambda: paused[0]))
    if pause:
        with pytest.raises(ExecutionCancelledError):
            execute()
    else:
        result = execute()
        assert result.native_window_forecast is None
        assert result.diagnostics["rejection_reasons"] == ["native_layer_publication_wall_budget_exhausted"]


def test_layer_controlled_cutoff_does_not_emit_native_execution_failure():
    from test_native_external_kv import ingress_scenario, native_runtime
    from heterollm_sim.execution_control import _NativeLayerSamplingCutoff
    runtime = native_runtime(ingress_scenario("hbm_tp2", warmup=3, measured=6))
    def cutoff(*args):
        raise _NativeLayerSamplingCutoff()
    runtime._native_layer_sampler = cutoff
    with pytest.raises(_NativeLayerSamplingCutoff):
        runtime.run_until_decode_steps(1)
    assert runtime.batches == []
    assert not any(event.event_type == "tokens_committed" for event in runtime.events)
    assert runtime._external_ingress_report()["imported_logical_bytes"] > 0
    assert not any(row.get("event_type") == "execution_failed" for row in runtime._schedule_trace)


@pytest.mark.parametrize("accepted_width", (2, 4))
def test_adaptive_windows_reuse_one_runtime_preparation_PD_and_keep_attempts_separate(monkeypatch, accepted_width):
    scenario, _, runtimes, record, sampling = harness(monkeypatch)
    scenario.simulation = replace(scenario.simulation, native_window=replace(
        scenario.simulation.native_window, max_layer_window_size=4))
    preparations, widths, contexts = [], [], []
    def prepare(*args):
        result = {"cohort": args[1], "source": args[0]}
        preparations.append(result)
        return result
    def sample(runtime, cohort, cost, stages, ready, options, control, *, prepared):
        widths.append(options.layer_window_size)
        contexts.append(prepared)
        current = {**record, "layer_ids": tuple(f"L{i}" for i in range(options.layer_window_size))}
        runtime._native_layer_sample_records = [current]
        return {"records": [current], "non_layer_samples_reused": len(widths)>1}
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(args[-1].layer_window_size)
        accepted = calls[-1] == accepted_width
        return {"accepted": accepted,
            "reasons": [] if accepted else ["native_layer_conditional_single_or_independent_pair_budget_exceeded"],
            "error_budget": {"width": calls[-1]},
            "metrics": {"cluster_tokens_per_second": 80., "tokens_per_second_per_gpu": 10.,
                "mean_complete_request_tpot_ns": 100., "mean_complete_request_ingress_ttft_ns": 200.,
                "measurement_duration_ns": 1000., "complete_fresh_requests": 2,
                "measurement_visible_output_tokens": 8},
            "metric_bounds": {},
            "window": {}, "predicted_visible_output_tokens": 9}
    sampling.prepare_native_layer_windows = prepare
    sampling.sample_native_layer_windows = sample
    sys.modules["heterollm_sim.native_layer_budget"].evaluate_native_layer_window = evaluate
    result = native.simulate_native_layer_windows(scenario, batch_lowerer=object())
    assert len(runtimes) == len(preparations) == 1
    expected_widths = [1, 2] if accepted_width == 2 else [1, 2, 4]
    assert widths == calls == expected_widths
    assert all(context is preparations[0] for context in contexts)
    attempts = result.diagnostics["layer_sampling_attempts"]
    assert [row["status"] for row in attempts] == ["budget_rejected"]*(len(widths)-1)+["accepted"]
    assert [row["executed_physical_layers"] for row in attempts] == expected_widths
    assert [row["budget"]["error_budget"]["width"] for row in attempts] == widths
    assert result.diagnostics["final_layer_window_size"] == accepted_width
    assert result.diagnostics["physical_layer_windows_all_attempts"] == len(widths)
    assert len(result.layer_window_samples) == 1  # Earlier rounds are historical, not holdout.
    assert result.native_window_forecast is not None
    assert all(row["physical_complete_cohorts"] == row["physical_visible_output_tokens"] == 0 for row in attempts)


@pytest.mark.parametrize("maximum,reason,expected", ((None,
    "native_layer_conditional_single_or_independent_pair_budget_exceeded", [1]),
    (4, "native_layer_future_physical_allocation_bound_unavailable", [1]),
    (4, "native_layer_conditional_single_or_independent_pair_budget_exceeded", [1, 2, 4])))
def test_adaptive_sampling_requires_explicit_limit_and_only_budget_refusals_expand(
        monkeypatch, maximum, reason, expected):
    scenario, _, _, record, sampling = harness(monkeypatch, accepted=False)
    scenario.simulation = replace(scenario.simulation, native_window=replace(
        scenario.simulation.native_window, max_layer_window_size=maximum))
    widths = []
    def sample(runtime, cohort, cost, stages, ready, options, control, **kwargs):
        widths.append(options.layer_window_size)
        return {"records": [record]}
    sampling.sample_native_layer_windows = sample
    sys.modules["heterollm_sim.native_layer_budget"].evaluate_native_layer_window = lambda *a, **kw: {
        "accepted": False, "reasons": [reason], "error_budget": {}}
    result = native.simulate_native_layer_windows(scenario, batch_lowerer=object())
    assert widths == expected
    assert result.native_window_forecast is None
    assert result.diagnostics["rejection_reasons"] == [reason]
    assert result.diagnostics["final_layer_window_size"] == expected[-1]


@pytest.mark.parametrize("pause", (False, True))
def test_adaptive_expansion_does_not_reset_total_deadline_and_preserves_completed_attempts(monkeypatch, pause):
    scenario, clock, _, record, sampling = harness(monkeypatch, accepted=False)
    scenario.simulation = replace(scenario.simulation, native_window=replace(
        scenario.simulation.native_window, max_layer_window_size=4))
    widths = []
    def sample(runtime, cohort, cost, stages, ready, options, control, **kwargs):
        widths.append(options.layer_window_size)
        runtime._native_layer_sample_records = [record]
        if options.layer_window_size == 2:
            clock[0] = 1801.
            if pause:
                raise ExecutionCancelledError("new user pause")
            control.raise_if_cancelled()
        return {"records": [record]}
    sampling.sample_native_layer_windows = sample
    sys.modules["heterollm_sim.native_layer_budget"].evaluate_native_layer_window = lambda *a, **kw: {
        "accepted": False, "reasons": ["native_layer_conditional_single_or_independent_pair_budget_exceeded"],
        "error_budget": {}}
    if pause:
        with pytest.raises(ExecutionCancelledError) as caught:
            native.simulate_native_layer_windows(scenario, batch_lowerer=object())
        attempts = caught.value.native_layer_sampling_attempts
        assert [row["status"] for row in attempts] == ["budget_rejected", "execution_stopped"]
    else:
        result = native.simulate_native_layer_windows(scenario, batch_lowerer=object())
        attempts = result.diagnostics["layer_sampling_attempts"]
        assert [row["status"] for row in attempts] == ["budget_rejected", "wall_budget_exhausted"]
        assert result.native_window_forecast is None
        assert "native_layer_sample_wall_budget_exhausted" in result.diagnostics["rejection_reasons"]
    assert widths == [1, 2]
    assert [row["complete_physical_layer_windows"] for row in attempts] == [1, 1]


@pytest.mark.parametrize("maximum", (0, True, 1.5))
def test_adaptive_layer_maximum_rejects_unknown_limits(maximum):
    with pytest.raises(ValueError):
        NativeWindowFastConfig(sampling_unit="layer_windows", max_layer_window_size=maximum)
