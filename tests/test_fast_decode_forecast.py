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


def harness(monkeypatch, *, growth=.12, cost=lambda step: 100., guard=True, options=None):
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
            self.batches = [NS(kind="prefill", cost=NS(metadata={"task_count": 3}))]
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
                self.batches.append(NS(kind="decode", cost=NS(metadata={"task_count": 2})))
                self.decode_steps = step
                self._completed_decode_steps = step
            return self.state.committed == request.output_tokens

        def run(self):
            self.run_calls += 1
            self.run_until_decode_steps(request.output_tokens - 1)
            return NS(batches=tuple(self.batches), finish=self.now)

    class Projection:
        safe = guard
        kv_buffers = ()
        safety_reasons = () if guard else ("fast_tail_requires_hbm_kv:hbf0",)
        def to_metadata(self):
            return {"safe": self.safe, "physical_state_advanced": False}

    monkeypatch.setattr(fast, "_compilation_scope", lambda *args: nullcontext())
    monkeypatch.setattr(fast, "compile_serving_plan", lambda scenario: plan)
    monkeypatch.setattr(fast, "_OnlineRuntime", Runtime)
    monkeypatch.setattr(fast, "_eligibility_reasons", lambda *args, **kwargs: ())
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
    assert "fast_tail_requires_hbm_kv:hbf0" in outcome.diagnostics["fallback_reasons"]


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


def test_adaptive_window_has_an_even_training_length():
    window = fast.sampling_window(FastDecodeConfig())
    assert (window.warmup, window.training, window.holdout) == (4, 8, 4)
    odd = fast.sampling_window(FastDecodeConfig(calibration_tokens=5, validation_tokens=3))
    assert (odd.training, odd.holdout) == (6, 3)


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
