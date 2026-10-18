"""A maintenance point estimate changes neither its fixed bounds nor samples."""
from dataclasses import asdict
import math

import pytest

from heterollm_sim import fast_simulation as fast
from heterollm_sim.fast_budget import (PairedTimingReference, minimax_decode_estimate,
    paired_timing_decision)
from test_fast_gc_budget import gc_harness, tiny_live_projection


def reference(kind="hbm", *, estimate=100., lower=98., upper=101.):
    return PairedTimingReference("model", 1000, 1024, kind, .1, estimate,
        lower, upper, "matching-existing-fast-reference")


def decision(estimate, lower, upper, *, kind="hbf", ref=None):
    return paired_timing_decision(decode_estimate_ns=estimate,
        decode_lower_ns=lower, decode_upper_ns=upper, target_relative_error=.1,
        candidate_memory_kind=kind, reference=ref)


@pytest.mark.parametrize("kind", ["hbm", "hbf"])
def test_joint_minimax_really_minimizes_fixed_direction_and_single_point(kind):
    ref = reference("hbf" if kind == "hbm" else "hbm")
    lower, upper = 90., 108.
    center = minimax_decode_estimate(lower, upper, candidate_memory_kind=kind, reference=ref)
    def score(estimate):
        result = decision(estimate, lower, upper, kind=kind, ref=ref)
        return max(result["candidate_relative_error_budget"], result["pair_relative_error_budget"])
    optimum = score(center)
    assert lower <= center <= upper
    for index in range(1001):
        assert optimum <= score(lower + (upper - lower) * index / 1000.) + 1e-15
    result = decision(center, lower, upper, kind=kind, ref=ref)
    assert result["candidate_decode_interval"]["lower_ns"] == lower
    assert result["candidate_decode_interval"]["upper_ns"] == upper
    assert result["pair_ratio_direction"] == "hbf/hbm"


def test_hbm_denominator_joint_minimax_is_not_just_its_harmonic_midpoint():
    center = minimax_decode_estimate(95., 105., candidate_memory_kind="hbm",
        reference=reference("hbf", lower=100., upper=100.))
    assert center == pytest.approx(math.sqrt(95. * 105.))


def test_minimax_cannot_make_a_too_wide_interval_pass():
    ref = reference(lower=100., upper=100.)
    center = minimax_decode_estimate(89., 111., candidate_memory_kind="hbf", reference=ref)
    result = decision(center, 89., 111., ref=ref)
    assert result["accepted"] is False
    assert result["candidate_within_target"] is False
    assert result["pair_within_target"] is False
    assert result["target_relative_error"] == .1


def test_exact_ten_percent_center_does_not_get_a_new_margin():
    ref = reference(lower=100., upper=100.)
    center = minimax_decode_estimate(90., 110., candidate_memory_kind="hbf", reference=ref)
    assert center == pytest.approx(99.)
    assert decision(center, 90., 110., ref=ref)["accepted"] is True
    assert decision(center, 90., 110.000001, ref=ref)["accepted"] is False


def test_adjacent_float_above_target_is_rejected_without_epsilon():
    ref = reference(lower=100., upper=100.)
    exact = decision(110., 100., 110., ref=ref)
    adjacent = math.nextafter(110., math.inf)
    exceeded = decision(adjacent, 100., adjacent, ref=ref)
    assert exact["accepted"] is True
    assert exceeded["candidate_within_target"] is False
    assert exceeded["pair_within_target"] is False
    assert exceeded["accepted"] is False
    with pytest.raises(ValueError, match="single-point"):
        reference("hbf", estimate=adjacent, lower=100., upper=adjacent)


def test_pair_endpoint_uses_exact_products_for_either_direction():
    for kind in ("hbf", "hbm"):
        ref = reference("hbm" if kind == "hbf" else "hbf", lower=100., upper=100.)
        if kind == "hbf":
            lower, estimate, upper = 100., 110., 110.
            beyond = math.nextafter(estimate, math.inf)
            result = decision(beyond, lower, beyond, kind=kind, ref=ref)
        else:
            # An exact reference numerator and an HBM denominator at 100/1.1
            # lie just beyond the target once rounded toward a smaller HBM.
            lower = math.nextafter(100. / 1.1, -math.inf)
            result = decision(lower, lower, 100., kind=kind, ref=ref)
        assert result["pair_within_target"] is False
        assert result["accepted"] is False


def test_no_reference_retains_original_independent_sufficient_guard():
    center = minimax_decode_estimate(95., 105., candidate_memory_kind="hbf")
    result = decision(center, 95., 105.)
    assert result["candidate_within_target"] is True
    assert result["accepted"] is False
    assert result["independent_symmetric_target_relative_error"] == .1 / 2.1


@pytest.mark.parametrize("bounds", [(0., 1.), (-1., 1.), (2., 1.), (1., math.inf),
                                    (math.nan, 1.), (True, 1.)])
def test_invalid_fixed_endpoints_are_never_repaired(bounds):
    with pytest.raises(ValueError):
        minimax_decode_estimate(*bounds, candidate_memory_kind="hbf")


def test_center_changes_only_predicted_durations_and_keeps_baseline_bounds():
    segments = (fast.ForecastSegment(3, 2, 1003, 1004, 30., 29., 31.),
                fast.ForecastSegment(5, 3, 1005, 1007, 50., 49., 51.))
    budget = {"conditional_tail_lower_ns": 60., "conditional_tail_upper_ns": 100.,
        "conditional_relative_error_budget": 1 / 3,
        "error_budget_breakdown": {"forecast_tail_ns": 80.,
            "model_only": {"lower_ns": 78., "upper_ns": 82.},
            "combined": {"lower_ns": 60., "upper_ns": 100.,
                         "relative_error_budget": 1 / 3}}}
    centered, selected, record = fast._center_maintenance_forecast(
        segments, 20., budget, memory_kind="hbf", reference=None)
    assert math.fsum(row.duration_ns for row in centered) + 20. == pytest.approx(96.)
    assert record["exact_sampled_decode_ns"] == 20.
    assert record["physical_prefix_rescaled"] is False
    assert record["conditional_bounds_recomputed_after_centering"] is False
    assert record["baseline_model_segments"] == [asdict(row) for row in segments]
    assert selected["conditional_tail_lower_ns"] == budget["conditional_tail_lower_ns"]
    assert selected["conditional_tail_upper_ns"] == budget["conditional_tail_upper_ns"]
    assert selected["error_budget_breakdown"]["model_only"] is budget["error_budget_breakdown"]["model_only"]
    assert selected["error_budget_breakdown"]["baseline_model_forecast_tail_ns"] == 80.
    assert selected["error_budget_breakdown"]["combined"]["estimate_ns"] == 76.
    for original, updated in zip(segments, centered, strict=True):
        assert (original.first_decode_step, original.decode_steps, original.first_context_tokens,
                original.last_context_tokens, original.model_lower_ns, original.model_upper_ns) == (
            updated.first_decode_step, updated.decode_steps, updated.first_context_tokens,
            updated.last_context_tokens, updated.model_lower_ns, updated.model_upper_ns)
    assert segments[0].duration_ns == 30.
    assert budget["error_budget_breakdown"]["forecast_tail_ns"] == 80.


def test_gc_forecast_finish_tpot_and_diagnostics_match_unchanged_exact_samples(monkeypatch):
    scenario, request, runners = gc_harness(monkeypatch)
    projected = tiny_live_projection()
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projected)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    forecast, diagnostics = result.forecast, result.diagnostics
    assert forecast is not None and result.detailed is None
    assert runners[0].run_calls == 0
    assert forecast.samples == tuple(fast.DecodeSample(step, 1000 + step, 100.)
                                    for step in range(1, 17))
    exact = math.fsum(row.duration_ns for row in forecast.samples)
    tail = math.fsum(row.duration_ns for row in forecast.segments)
    assert exact == diagnostics["exact_sampled_decode_ns"] == 1600.
    assert forecast.finish_ns == pytest.approx(forecast.cutoff_ns + tail)
    assert forecast.tpot_ns == pytest.approx((exact + tail) / (request.output_tokens - 1))
    assert diagnostics["decode_forecast_estimate_ns"] == pytest.approx(exact + tail)
    center = diagnostics["forecast_estimate_centering"]
    assert center["applied"] is True
    assert center["selected_tail_estimate_ns"] == pytest.approx(tail)
    assert diagnostics["sampling_attempts"][-1]["forecast_estimate_centering"] == center
    assert diagnostics["future_physical_state_advanced"] is False
    assert diagnostics["physical_tail_transactions_recorded"] is False
