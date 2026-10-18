"""Forecast acceptance, exact fallback, cancellation and honest result scope."""
from contextlib import nullcontext
from dataclasses import replace
import math
from types import SimpleNamespace as NS

import pytest

from heterollm_sim import fast_simulation as fast
from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim.simulation_options import FastDecodeConfig, SimulationConfig


def test_context_envelope_balances_constant_and_linear_extremes():
    prediction, lower, upper = fast.model_envelope(100., 1000., 1100, 100)
    assert lower == 10000.
    assert upper == pytest.approx(11495.)
    assert (prediction - lower) / lower == pytest.approx((upper - prediction) / upper)
    assert lower <= prediction <= upper
    assert fast.model_envelope(100., 1000., 1000, 0) == (0., 0., 0.)


@pytest.mark.parametrize("cost,context", [(0, 2), (math.nan, 2), (2, 0), (2, math.inf)])
def test_context_envelope_rejects_invalid_anchor(cost, context):
    with pytest.raises(ValueError):
        fast.model_envelope(cost, context, 10, 2)


def harness(monkeypatch, *, growth=.12, cost=lambda step: 100., guard=True,
            options=None, erase_counts=None):
    options = options or FastDecodeConfig(warmup_tokens=2, calibration_tokens=4,
        validation_tokens=2, max_sample_tokens=16, segment_tokens=5)
    scenario = NS(simulation=SimulationConfig(mode="fast", fast_decode=options))
    request = NS(request_id="r0", prompt_tokens=1000,
                 output_tokens=int(growth * 1000) + 1, arrival_ns=0.)
    plan = NS(requests=(request,))
    runners = []

    class Runtime:
        def __init__(self, plan, lowerer, **kwargs):
            self.plan = plan
            self.now = 1000.
            self.events = [NS(event_type="tokens_committed", request_id="r0",
                timestamp_ns=1000., details={"source": "prefill", "committed": 1})]
            prefill_metadata = {"task_count": 3}
            if erase_counts is not None:
                prefill_metadata["storage_traffic"] = {"erase_operations": erase_counts(0)}
            self.batches = [NS(kind="prefill", cost=NS(metadata=prefill_metadata))]
            self.state = NS(first_token_ns=1000., committed=1, spec=request)
            self._state_values = [self.state]
            self.decode_steps = 0
            self._completed_decode_steps = 0
            self.run_calls = 0
            runners.append(self)

        def run_until_decode_steps(self, target):
            for step in range(self.decode_steps + 1, min(target, request.output_tokens - 1) + 1):
                self.now += cost(step)
                self.state.committed += 1
                self.events.append(NS(event_type="tokens_committed", request_id="r0",
                    timestamp_ns=self.now, details={"committed": step + 1, "visible_tokens": 1}))
                metadata = {"task_count": 2}
                if erase_counts is not None:
                    metadata["storage_traffic"] = {"erase_operations": erase_counts(step)}
                self.batches.append(NS(kind="decode", cost=NS(metadata=metadata)))
                self.decode_steps = step
                self._completed_decode_steps = step
            return self.state.committed == request.output_tokens

        def run(self):
            self.run_calls += 1
            self.run_until_decode_steps(request.output_tokens - 1)
            return NS(batches=tuple(self.batches), finish=self.now)

    class Projection:
        safe = guard
        hbf_devices = ()
        kv_buffers = ()
        safety_reasons = () if guard else ("hbf_program_headroom_may_require_gc:hbf0",)
        def to_metadata(self):
            return {"safe": self.safe, "physical_state_advanced": False}

    monkeypatch.setattr(fast, "_compilation_scope", lambda *args: nullcontext())
    monkeypatch.setattr(fast, "compile_serving_plan", lambda scenario: plan)
    monkeypatch.setattr(fast, "_OnlineRuntime", Runtime)
    monkeypatch.setattr(fast, "_eligibility_reasons", lambda *args, **kwargs: ())
    monkeypatch.setattr(fast, "hbf_kv_page_period", lambda scenario: 1)
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: Projection())
    return scenario, request, runners


def test_fast_executes_only_contiguous_prefix_and_keeps_original_full_budget(monkeypatch):
    scenario, request, runners = harness(monkeypatch)
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    runner = runners[0]
    assert outcome.detailed is None
    assert outcome.forecast.sampled_decode_steps == runner.decode_steps == 8
    assert outcome.forecast.predicted_decode_steps == 112
    assert runner.state.committed == 9 and runner.run_calls == 0
    assert runner.plan.requests[0] is request and request.output_tokens == 121
    assert sum(segment.decode_steps for segment in outcome.forecast.segments) == 112
    assert outcome.forecast.ttft_ns == 1000.
    assert outcome.forecast.tpot_ns == pytest.approx(
        (outcome.forecast.finish_ns - 1000.) / 120)
    assert outcome.diagnostics["validation_status"] == "not_validated_against_full_detailed"
    assert outcome.diagnostics["future_physical_state_advanced"] is False


def test_storage_guard_resumes_the_same_runtime_to_detailed_completion(monkeypatch):
    scenario, request, runners = harness(monkeypatch, guard=False)
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    assert len(runners) == 1 and runners[0].run_calls == 1
    assert runners[0].decode_steps == request.output_tokens - 1
    assert outcome.forecast is None and outcome.detailed.finish == 13000.
    assert outcome.diagnostics["predicted_decode_steps"] == 0
    assert "hbf_program_headroom_may_require_gc:hbf0" in outcome.diagnostics["fallback_reasons"]


def test_unobserved_hbf_page_flush_period_cannot_be_predicted(monkeypatch):
    scenario, request, runners = harness(monkeypatch)
    projection = NS(safe=True, safety_reasons=(),
        hbf_devices=({"physical_owner": "hbf0.memory"},),
        kv_buffers=({"physical_owner": "hbf0.memory", "bytes_per_token": 64},),
        to_metadata=lambda: {"physical_state_advanced": False})
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projection)
    monkeypatch.setattr(fast, "hbf_kv_page_period", lambda scenario: 64)
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    assert outcome.forecast is None and runners[0].state.committed == request.output_tokens
    assert outcome.diagnostics["hbf_kv_page_period_tokens"] == 64
    assert outcome.diagnostics["fallback_reasons"] == ["calibration_does_not_cover_hbf_kv_page_period"]


def test_unstable_window_uses_more_samples_then_continues_exactly(monkeypatch):
    scenario, request, runners = harness(monkeypatch, cost=lambda step: 100. * 1.2 ** step)
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    assert outcome.forecast is None
    assert runners[0].run_calls == 1 and runners[0].state.committed == request.output_tokens
    assert "prefix_calibration_exceeds_error_budget" in outcome.diagnostics["fallback_reasons"]


@pytest.mark.parametrize("growth,reason", [(.3, "context_growth_exceeds_fast_domain"),
                                         (.004, "output_shorter_than_calibration_budget")])
def test_outside_domain_does_not_attempt_a_forecast(monkeypatch, growth, reason):
    scenario, _, runners = harness(monkeypatch, growth=growth)
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    assert outcome.diagnostics["fallback_reasons"] == [reason]
    assert runners[0].run_calls == 1 and outcome.forecast is None


def test_cancelled_forecast_does_not_report_completion(monkeypatch):
    scenario, _, runners = harness(monkeypatch)
    progress = []
    with pytest.raises(ExecutionCancelledError):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(cancellation_callback=lambda: True,
                                               progress_callback=progress.append))
    assert not any(item.stage == "fast_forecast_complete" for item in progress)
    assert runners[0].run_calls == 0


@pytest.mark.parametrize("period,expected", [(1, (4, 8, 4)), (4, (4, 8, 4)),
                                            (16, (4, 32, 16)), (64, (4, 128, 64))])
def test_adaptive_window_covers_whole_training_half_and_holdout_periods(period, expected):
    window = fast.sampling_window(FastDecodeConfig(), period)
    assert (window.warmup, window.training, window.holdout) == expected
    assert window.training % (2 * period) == window.holdout % period == 0


def test_fixed_window_preserves_original_calibration_estimate(monkeypatch):
    options = FastDecodeConfig(sampling_strategy="fixed")
    scenario, _, runners = harness(monkeypatch, cost=lambda step: 100., options=options)
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    assert runners[0].decode_steps == 56
    assert outcome.forecast.samples[8].step == 9
    calibration = fast._calibration(outcome.forecast.samples, options)
    expected = fast.forecast_segments(calibration["base_ns"], calibration["anchor_context"],
        1000, 56, 120, options.segment_tokens)
    assert outcome.forecast.segments == expected
    assert outcome.diagnostics["training_steps"] == 32
    assert outcome.diagnostics["holdout_steps"] == 16


def test_adaptive_default_executes_sixteen_steps_on_stable_hbm(monkeypatch):
    scenario, _, runners = harness(monkeypatch, options=FastDecodeConfig())
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    assert outcome.forecast.sampled_decode_steps == runners[0].decode_steps == 16
    assert outcome.diagnostics["per_storage_conditional_error_target"] == pytest.approx(.1 / 2.1)
    assert outcome.diagnostics["pairwise_ratio_target_relative_error"] == .1


def test_adaptive_expands_both_windows_without_discarding_earlier_training(monkeypatch):
    # A first-period offset is visible to the early/late comparison. Enlarging
    # both windows dilutes it, rather than sliding past and concealing it.
    scenario, _, _ = harness(monkeypatch,
        cost=lambda step: 110. if 5 <= step <= 8 else 100.,
        options=FastDecodeConfig())
    outcome = fast.simulate_fast(scenario, batch_lowerer=object())
    attempts = outcome.diagnostics["sampling_attempts"]
    assert len(attempts) > 1
    for first, second in zip(attempts, attempts[1:]):
        assert first["warmup"] == second["warmup"] == 4
        assert second["training"] > first["training"]
        assert second["holdout"] > first["holdout"]
        assert second["sampled_steps"] == sum(second[key] for key in ("warmup", "training", "holdout"))


def test_maintenance_bounds_cover_both_unseen_erase_and_prefix_positive_bias():
    batches = [NS(kind="decode", cost=NS(metadata={"storage_traffic": {
        "erase_operations": count}})) for count in (99, 1, 0, 2, 0, 0, 0)]
    runner = NS(batches=batches)
    projection = NS(hbf_devices=({"block_erase_ns": 10.,
                                  "ordinary_erase_serial_time_guard_ns": 200.},))
    result = fast.maintenance_error_envelope(runner, projection,
        fast.SamplingWindow(1, 4, 2, 1), remaining=60, context_scale=1.1)
    assert result["sampled_erase_operations"] == 102
    assert result["previous_cohort_erase_operations"] == 99
    assert result["erase_guard_batches"] == 7
    assert result["erase_guard_sample_steps"] == 6
    assert result["prefix_erase_bias_guard_ns"] == pytest.approx(11220.)
    assert result["future_erase_serial_guard_ns"] == 200.
    assert result["erase_observations_available"]


def test_unavailable_erase_observation_is_not_zero_maintenance():
    runner = NS(batches=[NS(kind="decode", cost=NS(metadata={})) for _ in range(6)])
    projection = NS(hbf_devices=({"block_erase_ns": 10.,
                                  "ordinary_erase_serial_time_guard_ns": 200.},))
    result = fast.maintenance_error_envelope(runner, projection,
        fast.SamplingWindow(1, 4, 2, 1), remaining=60, context_scale=1.)
    assert not result["erase_observations_available"]


def test_zero_warmup_erase_guard_includes_final_prefill_cohort_cleanup():
    def batch(kind, count):
        return NS(kind=kind, cost=NS(metadata={"storage_traffic": {"erase_operations": count}}))
    runner = NS(batches=[batch("prefill", 7)] + [batch("decode", 0) for _ in range(6)])
    projection = NS(hbf_devices=({"block_erase_ns": 10.,
                                  "ordinary_erase_serial_time_guard_ns": 200.},))
    result = fast.maintenance_error_envelope(runner, projection,
        fast.SamplingWindow(0, 4, 2, 1), remaining=60, context_scale=1.)
    assert result["erase_observations_available"]
    assert result["previous_cohort_erase_operations"] == 7
    assert result["prefix_erase_bias_guard_ns"] == 700.


def test_unbounded_erase_budget_expands_then_falls_back_with_json_safe_diagnostics(monkeypatch):
    import json

    # Many simultaneous bank erases have a serial guard larger than elapsed
    # sample time. This invalidates the timing lower bound, not the request.
    scenario, request, runners = harness(monkeypatch,
        options=FastDecodeConfig(max_sample_tokens=40), erase_counts=lambda step: 8)
    projection = NS(safe=True, safety_reasons=(),
        hbf_devices=({"physical_owner": "hbf0.memory", "block_erase_ns": 100.,
                      "ordinary_erase_serial_time_guard_ns": 10000.},),
        to_metadata=lambda: {"safe": True, "physical_state_advanced": False})
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projection)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is None and result.detailed is not None
    assert runners[0].run_calls == 1 and runners[0].decode_steps == request.output_tokens - 1
    attempts = result.diagnostics["sampling_attempts"]
    assert len(attempts) > 1
    assert all(attempt["conditional_relative_error_budget"] is None for attempt in attempts)
    assert all(attempt["budget_unavailable_reason"] ==
        "prefix_erase_budget_exhausts_lower_timing_bound" for attempt in attempts)
    assert result.diagnostics["conditional_tail_lower_ns"] is None
    assert result.diagnostics["conditional_relative_error_budget"] is None
    assert result.diagnostics["fallback_reasons"] == ["maintenance_recalibration_exceeds_sample_budget"]
    json.dumps(result.diagnostics, allow_nan=False)


def test_unbounded_local_error_can_be_reported_after_detailed_fallback(monkeypatch):
    import json

    scenario, _, runners = harness(monkeypatch,
        cost=lambda step: 10000. if 3 <= step <= 6 else 100.)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is None and runners[0].run_calls == 1
    assert result.diagnostics["sampling_attempts"][0]["conditional_relative_error_budget"] is None
    assert result.diagnostics["sampling_attempts"][0]["budget_unavailable_reason"] == \
        "local_calibration_error_not_bounded_below_one"
    json.dumps(result.diagnostics, allow_nan=False)


def test_two_sided_storage_budget_controls_the_ratio_under_declared_bounds():
    target = .1
    epsilon = target / (2 + target)
    assert (1 + epsilon) / (1 - epsilon) - 1 == pytest.approx(target)
    assert 1 - (1 - epsilon) / (1 + epsilon) <= target


def test_fixed_incomplete_phase_windows_use_detailed_instead_of_biased_means(monkeypatch):
    scenario, _, runners = harness(monkeypatch, options=FastDecodeConfig(
        sampling_strategy="fixed", warmup_tokens=0, calibration_tokens=10,
        validation_tokens=4, max_sample_tokens=20))
    monkeypatch.setattr(fast, "hbf_kv_page_period", lambda scenario: 4)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is None and runners[0].run_calls == 1
    assert result.diagnostics["fallback_reasons"] == ["calibration_does_not_cover_hbf_kv_page_period"]


def maintenance_harness(monkeypatch, *, erases=(), period=16, cap=128,
                        future_guard=10000., hbf=True):
    """Synthetic media maintenance at known cohorts, never a reference fit."""
    def cost(step):
        return 100. + 100. * (step in erases) + 50. * (step - 1 in erases)

    scenario, request, runners = harness(monkeypatch, options=FastDecodeConfig(
        max_sample_tokens=cap), cost=cost,
        erase_counts=lambda step: 256 if step in erases else 0)
    request.prompt_tokens, request.output_tokens = 65536, 5120
    devices = ({"physical_owner": "hbf0.memory", "block_erase_ns": 2.,
                "ordinary_erase_serial_time_guard_ns": future_guard},) if hbf else ()
    projection = NS(safe=True, safety_reasons=(), hbf_devices=devices,
        to_metadata=lambda: {"safe": True, "physical_state_advanced": False})
    monkeypatch.setattr(fast, "hbf_kv_page_period", lambda scenario: period)
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projection)
    return scenario, request, runners, cost


def test_ordinary_erase_recalibrates_at_eighty_without_removing_prefix_cost(monkeypatch):
    import json

    scenario, request, runners, cost = maintenance_harness(monkeypatch, erases=(23,))
    progress = []
    result = fast.simulate_fast(scenario, batch_lowerer=object(),
        execution_control=ExecutionControl(progress_callback=progress.append))
    assert result.forecast is not None and runners[0].run_calls == 0
    diagnostics = result.diagnostics
    attempts = diagnostics["sampling_attempts"]
    assert [row["sampled_steps"] for row in attempts] == [52, 80]
    assert attempts[0]["forecast_accepted"] is False
    assert attempts[0]["prefix_erase_bias_guard_ns"] > 0
    assert attempts[1]["forecast_accepted"] is True
    assert attempts[1]["prefix_erase_bias_guard_ns"] == 0
    assert attempts[1]["future_erase_serial_guard_ns"] == 10000.
    assert attempts[1]["acceptance_target_relative_error"] == pytest.approx(.1 / 2.1)
    assert attempts[1]["training_first_decode_step"] == 33
    assert attempts[1]["training_last_decode_step"] == 64
    assert attempts[1]["holdout_first_decode_step"] == 65
    assert attempts[1]["holdout_last_decode_step"] == 80
    assert attempts[1]["refit_first_decode_step"] == 33
    assert attempts[1]["refit_last_decode_step"] == 80
    assert attempts[1]["holdout_disjoint_from_training"] is True
    assert attempts[1]["final_refit_includes_holdout"] is True
    reset, = diagnostics["sampling_resets"]
    assert reset["old_warmup"] == 4 and reset["new_warmup"] == 32
    assert reset["last_erase_decode_step"] == 23 and reset["last_erase_operations"] == 256
    assert reset["cleanup_through_decode_step"] == 24
    assert reset["next_target_decode_steps"] == 80
    assert reset["within_sample_budget"] is True
    assert reset["exact_prefix_samples_and_latency_preserved"] is True
    assert reset["future_erase_service_budget_preserved"] is True
    assert len(result.forecast.samples) == len(diagnostics["calibration_samples"]) == 80
    assert [sample.step for sample in result.forecast.samples] == list(range(1, 81))
    assert [sample.duration_ns for sample in result.forecast.samples] == [cost(step) for step in range(1, 81)]
    assert result.forecast.cutoff_ns == 1000. + math.fsum(cost(step) for step in range(1, 81))
    assert result.forecast.tpot_ns == pytest.approx(
        (math.fsum(cost(step) for step in range(1, 81))
         + math.fsum(segment.duration_ns for segment in result.forecast.segments))
        / (request.output_tokens - 1))
    calibration_events = [event for event in progress if event.stage == "fast_calibration"]
    assert calibration_events[0].metadata["approximation"]["sampling_resets"] == []
    assert len(calibration_events[1].metadata["approximation"]["sampling_resets"]) == 1
    assert [event.stage for event in progress] == [
        "fast_calibration", "fast_recalibration", "fast_calibration", "fast_forecast_complete"]
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("erase_step,warmup,target", [(23, 32, 80), (32, 48, 96), (52, 64, 112)])
def test_cleanup_boundary_excludes_next_cohort_before_rounding_period(
        monkeypatch, erase_step, warmup, target):
    scenario, _, runners, _ = maintenance_harness(monkeypatch, erases=(erase_step,))
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is not None
    assert result.forecast.sampled_decode_steps == runners[0].decode_steps == target
    reset, = result.diagnostics["sampling_resets"]
    assert reset["cleanup_through_decode_step"] == erase_step + 1
    assert reset["new_warmup"] == reset["aligned_cleanup_through_decode_step"] == warmup
    final = result.diagnostics["sampling_attempts"][-1]
    assert final["training_first_decode_step"] == warmup + 1
    assert final["previous_cohort_erase_operations"] == final["sampled_erase_operations"] == 0


def test_erase_reset_beyond_cap_is_reported_before_detailed_execution(monkeypatch):
    scenario, _, runners, _ = maintenance_harness(monkeypatch, erases=(23,), cap=64)
    progress = []

    class RequireForecast(RuntimeError):
        pass

    def stop_fallback(event):
        progress.append(event)
        if event.stage == "fast_mode_fallback":
            raise RequireForecast()

    with pytest.raises(RequireForecast):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=stop_fallback))
    assert runners[0].decode_steps == 52 and runners[0].run_calls == 0
    approximation = progress[-1].metadata["approximation"]
    assert approximation["fallback_reasons"] == ["maintenance_recalibration_exceeds_sample_budget"]
    assert approximation["sampling_resets"][-1]["next_target_decode_steps"] == 80
    assert approximation["sampling_resets"][-1]["within_sample_budget"] is False
    assert len(approximation["calibration_samples"]) == 52


def test_repeated_erases_cannot_escape_cap_or_mutate_earlier_reset_records(monkeypatch):
    import json

    scenario, _, runners, _ = maintenance_harness(monkeypatch, erases=(23, 70, 110))
    progress = []

    def stop_fallback(event):
        progress.append(event)
        if event.stage == "fast_mode_fallback":
            raise ExecutionCancelledError("test stops before detailed tail")

    with pytest.raises(ExecutionCancelledError):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=stop_fallback))
    approximation = progress[-1].metadata["approximation"]
    assert approximation["fallback_reasons"] == ["maintenance_recalibration_exceeds_sample_budget"]
    assert [row["sampled_steps"] for row in approximation["sampling_attempts"]] == [52, 80, 128]
    assert [row["next_target_decode_steps"] for row in approximation["sampling_resets"]] == [80, 128, 160]
    assert runners[0].decode_steps == 128 and runners[0].run_calls == 0
    assert len(approximation["calibration_samples"]) == 128
    reset_events = [event for event in progress if event.stage == "fast_recalibration"]
    assert [len(event.metadata["approximation"]["sampling_resets"]) for event in reset_events] == [1, 2, 3]
    assert all(row["future_erase_serial_guard_ns"] == 10000.
               for row in approximation["sampling_attempts"])
    json.dumps(approximation, allow_nan=False)


@pytest.mark.parametrize("hbf,period", [(False, 1), (True, 4)])
def test_no_erase_hbm_and_hbf_keep_original_sixteen_step_estimate(monkeypatch, hbf, period):
    scenario, _, runners, _ = maintenance_harness(monkeypatch, hbf=hbf, period=period)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast.sampled_decode_steps == runners[0].decode_steps == 16
    assert result.diagnostics["sampling_resets"] == []
    calibration = fast._calibration(result.forecast.samples, scenario.simulation.fast_decode,
        fast.sampling_window(scenario.simulation.fast_decode, period))
    expected = fast.forecast_segments(calibration["base_ns"], calibration["anchor_context"],
        65536, 16, 5119, scenario.simulation.fast_decode.segment_tokens)
    assert result.forecast.segments == expected


def test_reset_does_not_remove_future_erase_budget_or_force_a_forecast(monkeypatch):
    scenario, _, _, _ = maintenance_harness(monkeypatch, erases=(23,), future_guard=1000000.)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is None
    assert result.diagnostics["fallback_reasons"] == ["prefix_calibration_exceeds_error_budget"]
    attempts = result.diagnostics["sampling_attempts"]
    assert len(result.diagnostics["sampling_resets"]) == 1
    assert all(row["future_erase_serial_guard_ns"] == 1000000. for row in attempts)
    assert all(not row["forecast_accepted"] for row in attempts)
    assert all(row["prefix_erase_bias_guard_ns"] == 0 for row in attempts[1:])


def test_late_reset_extends_whole_training_periods_without_rewinding_runtime(monkeypatch):
    from types import SimpleNamespace as NS

    def batch(step):
        return NS(kind="decode", cost=NS(metadata={"storage_traffic": {
            "erase_operations": 3 if step == 23 else 0}}))
    runner = NS(batches=[batch(step) for step in range(1, 101)])
    candidate, record = fast.maintenance_recalibration(runner,
        fast.SamplingWindow(4, 64, 32, 16), FastDecodeConfig())
    assert candidate == fast.SamplingWindow(32, 64, 16, 16)
    assert candidate.total == 112 >= len(runner.batches)
    assert record["new_warmup"] == 32


@pytest.mark.parametrize("period,erase_step,warmup", [(1, 5, 6), (4, 12, 16)])
def test_maintenance_reset_preserves_original_minimum_small_period_windows(period, erase_step, warmup):
    runner = NS(batches=[NS(kind="decode", cost=NS(metadata={"storage_traffic": {
        "erase_operations": int(step == erase_step)}})) for step in range(1, 17)])
    original = fast.sampling_window(FastDecodeConfig(), period)
    candidate, record = fast.maintenance_recalibration(runner, original, FastDecodeConfig())
    assert candidate.warmup == warmup
    assert candidate.training == original.training == 8
    assert candidate.holdout == original.holdout == 4
    assert record["next_target_decode_steps"] >= len(runner.batches)


def test_duration_outlier_without_physical_erase_cannot_trigger_a_window_reset(monkeypatch):
    scenario, _, _ = harness(monkeypatch, options=FastDecodeConfig(),
        cost=lambda step: 300. if step in (23, 24) else 100., erase_counts=lambda step: 0)
    projection = NS(safe=True, safety_reasons=(),
        hbf_devices=({"physical_owner": "hbf0.memory", "block_erase_ns": 2.,
                      "ordinary_erase_serial_time_guard_ns": 0.},),
        to_metadata=lambda: {"safe": True, "physical_state_advanced": False})
    monkeypatch.setattr(fast, "hbf_kv_page_period", lambda scenario: 16)
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projection)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.diagnostics["sampling_resets"] == []
    assert all(attempt["warmup"] == 4 for attempt in result.diagnostics["sampling_attempts"])
