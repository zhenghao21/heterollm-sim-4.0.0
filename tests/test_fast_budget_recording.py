"""Conditional budget arithmetic and durable per-attempt diagnostics.

All execution uses the tiny synthetic prefix runtime; no physical simulation
or completed detailed reference is used to select a forecast.
"""
from dataclasses import asdict
import json
import math

import pytest

from heterollm_sim import fast_simulation as fast
from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim.simulation_options import FastDecodeConfig
from test_fast_decode_forecast import harness


def assert_interval(actual, lower, upper, estimate):
    assert actual["lower_ns"] == pytest.approx(lower)
    assert actual["upper_ns"] == pytest.approx(upper)
    over = estimate / lower - 1
    under = 1 - estimate / upper
    assert actual["overestimate_relative_error"] == pytest.approx(over)
    assert actual["underestimate_relative_error"] == pytest.approx(under)
    assert actual["relative_error_budget"] == pytest.approx(max(over, under))


def test_budget_records_each_interval_and_combines_bounds_instead_of_percentages():
    result = fast.conditional_error_budget(100., 140., 120., .1,
        {"prefix_erase_bias_guard_ns": 10., "future_erase_serial_guard_ns": 30.})
    parts = result["error_budget_breakdown"]
    assert parts["semantics"] == "conditional_interval_bounds_not_additive_percentages"
    assert parts["forecast_tail_ns"] == 120.
    assert parts["local_relative_error"] == .1
    assert parts["prefix_erase_bias_guard_ns"] == 10.
    assert parts["future_erase_serial_guard_ns"] == 30.
    expected = {
        "model_only": (100., 140.),
        "model_with_local_error": (100. / 1.1, 140. / .9),
        "model_with_prefix_erase_bias": (90., 140.),
        "model_with_future_erase": (100., 170.),
        "model_with_maintenance": (90., 170.),
        "combined": (90. / 1.1, 170. / .9),
    }
    for name, (lower, upper) in expected.items():
        assert_interval(parts[name], lower, upper, 120.)
    combined = parts["combined"]
    assert result["conditional_tail_lower_ns"] == combined["lower_ns"]
    assert result["conditional_tail_upper_ns"] == combined["upper_ns"]
    assert result["conditional_relative_error_budget"] == combined["relative_error_budget"]
    additive = sum(parts[name]["relative_error_budget"] for name in (
        "model_only", "model_with_local_error", "model_with_prefix_erase_bias",
        "model_with_future_erase"))
    assert combined["relative_error_budget"] != pytest.approx(additive)
    assert result["budget_unavailable_reason"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("local,prefix,reason", [
    (.01, 100., "prefix_erase_budget_exhausts_lower_timing_bound"),
    (.01, 101., "prefix_erase_budget_exhausts_lower_timing_bound"),
    (1., 0., "local_calibration_error_not_bounded_below_one"),
    (math.nan, 0., "local_calibration_error_not_bounded_below_one"),
    (math.inf, 0., "local_calibration_error_not_bounded_below_one"),
])
def test_unavailable_budget_remains_strict_json_without_infinite_error(local, prefix, reason):
    result = fast.conditional_error_budget(100., 140., 120., local,
        {"prefix_erase_bias_guard_ns": prefix, "future_erase_serial_guard_ns": 30.})
    assert result["conditional_relative_error_budget"] is None
    assert result["conditional_tail_lower_ns"] is None
    assert result["conditional_tail_upper_ns"] is None
    assert result["budget_unavailable_reason"] == reason
    json.dumps(result, allow_nan=False)


def test_every_sampling_attempt_emits_its_own_immutable_diagnostic_record(monkeypatch):
    scenario, _, runners = harness(monkeypatch,
        cost=lambda step: 105. if 5 <= step <= 8 else 100.,
        options=FastDecodeConfig())
    progress = []
    outcome = fast.simulate_fast(scenario, batch_lowerer=object(),
        execution_control=ExecutionControl(progress_callback=progress.append))
    events = [event for event in progress if event.stage == "fast_calibration"]
    attempts = outcome.diagnostics["sampling_attempts"]
    assert len(events) == len(attempts) > 1
    for index, (event, attempt) in enumerate(zip(events, attempts), 1):
        approximation = event.metadata["approximation"]
        assert event.completed == attempt["sampled_steps"]
        assert event.metadata["calibration_attempt"] == attempt
        assert len(approximation["sampling_attempts"]) == index
        assert approximation["sampling_attempts"][-1] == attempt
        assert len(approximation["calibration_samples"]) == event.completed
        assert approximation["calibration_samples"][-1]["step"] == event.completed
        json.dumps(event.metadata, allow_nan=False)
    assert runners[0].run_calls == 0


def test_rejected_forecast_callback_can_stop_before_any_detailed_tail(monkeypatch):
    scenario, request, runners = harness(monkeypatch,
        cost=lambda step: 100. * 1.2 ** step)
    progress = []

    class RequireForecast(RuntimeError):
        pass

    def record_then_reject(event):
        progress.append(event)
        if event.stage == "fast_mode_fallback":
            raise RequireForecast("diagnostic run must not execute detailed tail")

    with pytest.raises(RequireForecast):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=record_then_reject))
    runner = runners[0]
    assert runner.run_calls == 0
    assert runner.decode_steps < request.output_tokens - 1
    fallback = progress[-1]
    assert fallback.stage == "fast_mode_fallback"
    approximation = fallback.metadata["approximation"]
    assert approximation["effective_mode"] == "forecast_rejected"
    assert approximation["validation_status"] == "forecast_rejected_before_detailed_fallback"
    assert approximation["physical_tail_transactions_recorded"] is False
    assert approximation["future_physical_state_advanced"] is False
    assert approximation["predicted_decode_steps"] == 0
    assert approximation["sampled_decode_steps"] == runner.decode_steps
    assert len(approximation["calibration_samples"]) == runner.decode_steps
    final_attempt = approximation["sampling_attempts"][-1]
    assert final_attempt["sampled_steps"] == runner.decode_steps
    assert final_attempt["forecast_accepted"] is False
    assert final_attempt["error_budget_breakdown"]["combined"]["relative_error_budget"] == \
        final_attempt["conditional_relative_error_budget"]
    assert not any(event.stage == "fast_forecast_complete" for event in progress)
    json.dumps(fallback.metadata, allow_nan=False)


def test_success_retains_raw_samples_and_separate_training_holdout_observations(monkeypatch):
    options = FastDecodeConfig()
    # Independent holdout is slightly slower, so a copied training total would
    # be observable. The tiny drift still fits the unchanged ratio budget.
    cost = lambda step: 101. if 13 <= step <= 16 else 100.
    scenario, _, runners = harness(monkeypatch, options=options, cost=cost)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is not None and result.detailed is None
    assert result.forecast.sampled_decode_steps == 16
    diagnostics = result.diagnostics
    assert diagnostics["calibration_samples"] == [asdict(row) for row in result.forecast.samples]
    assert [sample.duration_ns for sample in result.forecast.samples] == [cost(step) for step in range(1, 17)]
    assert diagnostics["training_mean_ns"] == 100.
    assert diagnostics["training_early_mean_ns"] == 100.
    assert diagnostics["training_late_mean_ns"] == 100.
    assert diagnostics["holdout_observed_ns"] == 404.
    assert diagnostics["holdout_estimated_ns"] != diagnostics["holdout_observed_ns"]
    assert diagnostics["training_steps"] == 8 and diagnostics["holdout_steps"] == 4
    attempt = diagnostics["sampling_attempts"][-1]
    assert attempt["forecast_accepted"] is True
    assert attempt["holdout_observed_ns"] == 404.
    assert len(diagnostics["calibration_cycles"]) == 16
    assert math.fsum(row["duration_ns"] for row in diagnostics["calibration_cycles"]) == \
        math.fsum(sample.duration_ns for sample in result.forecast.samples)
    assert runners[0].run_calls == 0
    json.dumps(diagnostics, allow_nan=False)


def test_cancellation_after_calibration_keeps_partial_record_without_completion(monkeypatch):
    scenario, request, runners = harness(monkeypatch)
    progress = []
    cancelled = False

    def cancel_on_calibration(event):
        nonlocal cancelled
        progress.append(event)
        if event.stage == "fast_calibration":
            cancelled = True

    with pytest.raises(ExecutionCancelledError):
        fast.simulate_fast(scenario, batch_lowerer=object(), execution_control=ExecutionControl(
            progress_callback=cancel_on_calibration, cancellation_callback=lambda: cancelled))
    assert len(progress) == 1 and progress[0].stage == "fast_calibration"
    approximation = progress[0].metadata["approximation"]
    assert approximation["validation_status"] == "not_validated_against_full_detailed"
    assert approximation["future_physical_state_advanced"] is False
    assert approximation["physical_tail_transactions_recorded"] is False
    assert "forecast_segments" not in approximation
    assert len(approximation["calibration_samples"]) == runners[0].decode_steps
    assert runners[0].run_calls == 0 and runners[0].decode_steps < request.output_tokens - 1
    json.dumps(progress[0].metadata, allow_nan=False)
