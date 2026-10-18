"""Whole-Decode interval pairing, including unequal and directional errors."""
from dataclasses import FrozenInstanceError, replace
import json
import math
from types import SimpleNamespace as NS

import pytest

from heterollm_sim import fast_simulation as fast
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.fast_budget import (
    PairedTimingReference, paired_timing_decision, validate_paired_reference,
)
from heterollm_sim.simulation_options import FastDecodeConfig
from test_fast_decode_forecast import harness


def reference(**changes):
    values = dict(model_name="model", prompt_tokens=1000, output_tokens=157,
        memory_kind="hbm", target_relative_error=.10, decode_estimate_ns=100.,
        decode_lower_ns=96., decode_upper_ns=104., source_id="completed-hbm-point")
    values.update(changes)
    return PairedTimingReference(**values)


def decide(estimate=100., lower=96., upper=106., *, ref=None, kind="hbf"):
    return paired_timing_decision(decode_estimate_ns=estimate,
        decode_lower_ns=lower, decode_upper_ns=upper, target_relative_error=.10,
        candidate_memory_kind=kind, reference=ref)


def test_pair_accepts_unequal_intervals_without_widening_ten_percent_target():
    unpaired = decide()
    paired = decide(ref=reference())
    assert unpaired["accepted"] is False
    assert paired["accepted"] is True
    assert paired["candidate_relative_error_budget"] > .10 / 2.10
    assert paired["candidate_relative_error_budget"] < .10
    assert paired["pair_ratio_direction"] == "hbf/hbm"
    assert paired["pair_ratio_interval"]["lower"] == pytest.approx(96. / 104.)
    assert paired["pair_ratio_interval"]["upper"] == pytest.approx(106. / 96.)
    assert paired["pair_ratio_interval"]["overestimate_relative_error"] == pytest.approx(104. / 96. - 1.)
    assert paired["pair_ratio_interval"]["underestimate_relative_error"] == pytest.approx(1. - 96. / 106.)
    assert paired["pair_relative_error_budget"] < .10
    assert paired["reference_source_id"] == "completed-hbm-point"
    assert paired["semantics"] == "conditional_model_bounds_not_measured_error"


@pytest.mark.parametrize("candidate_kind", ["hbf", "hbm"])
def test_ratio_uses_requested_hbf_over_hbm_direction_for_either_candidate(candidate_kind):
    if candidate_kind == "hbf":
        result = decide(110., 100., 110., ref=reference(
            decode_estimate_ns=100., decode_lower_ns=100., decode_upper_ns=100.))
    else:
        result = decide(100., 100., 100., kind="hbm", ref=reference(memory_kind="hbf",
            decode_estimate_ns=110., decode_lower_ns=100., decode_upper_ns=110.))
    assert result["accepted"] is True
    assert result["pair_ratio_direction"] == "hbf/hbm"
    assert result["pair_ratio_interval"]["estimate"] == pytest.approx(1.10)
    assert result["pair_ratio_interval"]["lower"] == pytest.approx(1.)
    assert result["pair_ratio_interval"]["upper"] == pytest.approx(1.10)


def test_exact_ten_percent_boundary_passes_without_an_added_margin():
    exact_ref = reference(decode_estimate_ns=100., decode_lower_ns=100., decode_upper_ns=100.)
    boundary = decide(110., 100., 110., ref=exact_ref)
    above = decide(110.001, 100., 110.001, ref=exact_ref)
    assert boundary["accepted"] is True
    assert above["accepted"] is False
    assert above["candidate_within_target"] is False
    assert "candidate_single_point_conditional_error_exceeds_target" in above["reasons"]


def test_pair_can_fail_even_when_both_single_point_bounds_pass():
    result = decide(100., 94., 106., ref=reference(decode_lower_ns=94., decode_upper_ns=106.))
    assert result["candidate_within_target"] and result["reference_within_target"]
    assert result["pair_within_target"] is False
    assert result["accepted"] is False
    assert "directional_pair_conditional_error_exceeds_target" in result["reasons"]


def test_reverse_candidate_cannot_pass_only_the_reciprocal_ratio_target():
    # HBM/HBF would underestimate by 9.52%, but the requested HBF/HBM
    # ratio can be overestimated by 10.53%.  Only the latter is the target.
    result = decide(100., 100., 105., kind="hbm", ref=reference(memory_kind="hbf",
        decode_lower_ns=95., decode_upper_ns=100.))
    assert result["candidate_within_target"] and result["reference_within_target"]
    assert result["pair_ratio_interval"]["overestimate_relative_error"] == pytest.approx(105. / 95. - 1.)
    assert result["pair_relative_error_budget"] > .10
    assert result["accepted"] is False


def test_missing_conditional_interval_is_a_recorded_refusal_not_a_zero_budget():
    result = decide(lower=None, ref=reference())
    assert result["accepted"] is False
    assert result["candidate_relative_error_budget"] is None
    assert result["reference_relative_error_budget"] is not None
    assert result["reasons"] == ["candidate_conditional_decode_interval_unavailable"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("changes", [
    {"model_name": ""}, {"source_id": ""}, {"prompt_tokens": 0},
    {"output_tokens": True}, {"memory_kind": "HBM"}, {"weight_dtype": "fp16"},
    {"kv_dtype": "int8"}, {"weight_dtype": None}, {"memory_kind": []},
    {"decode_lower_ns": 0}, {"decode_estimate_ns": math.inf},
    {"decode_upper_ns": math.nan}, {"decode_lower_ns": 101.},
    {"decode_upper_ns": 99.}, {"target_relative_error": .11},
    {"decode_lower_ns": 89.},
])
def test_reference_rejects_invalid_domain_and_its_own_error_bound(changes):
    with pytest.raises(ValueError):
        reference(**changes)


@pytest.mark.parametrize("changes", [
    {"model_name": "other"}, {"prompt_tokens": 1024}, {"output_tokens": 158},
    {"memory_kind": "hbm"}, {"target_relative_error": .09},
])
def test_request_match_is_strict(changes):
    values = dict(model_name="model", prompt_tokens=1000, output_tokens=157,
                  memory_kind="hbf", target_relative_error=.10)
    values.update(changes)
    with pytest.raises(ValueError):
        validate_paired_reference(reference(), **values)


def test_reference_is_frozen():
    ref = reference()
    with pytest.raises(FrozenInstanceError):
        ref.model_name = "another"


def paired_runtime(monkeypatch):
    scenario, request, runners = harness(monkeypatch, growth=.156,
        options=FastDecodeConfig(max_sample_tokens=16), erase_counts=lambda step: 0)
    scenario.model = NS(name="model")
    scenario.hardware = NS(component_map=lambda: {"hbf0": NS(normalized_kind="hbf")})
    plan = NS(requests=(request,), kv_policy=NS(cache_component="hbf0"))
    projection = NS(safe=True, safety_reasons=(),
        hbf_devices=({"block_erase_ns": 1., "ordinary_erase_serial_time_guard_ns": 300.},),
        to_metadata=lambda: {"safe": True, "physical_state_advanced": False})
    monkeypatch.setattr(fast, "compile_serving_plan", lambda scenario: plan)
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projection)
    ref = reference(decode_estimate_ns=15600., decode_lower_ns=15000., decode_upper_ns=16200.)
    return scenario, request, runners, ref


def test_synthetic_runtime_pairs_exact_prefix_and_never_executes_detailed_suffix(monkeypatch):
    scenario, request, runners, ref = paired_runtime(monkeypatch)
    progress = []
    outcome = fast.simulate_fast(scenario, batch_lowerer=object(), paired_reference=ref,
        execution_control=ExecutionControl(progress_callback=progress.append))
    assert outcome.forecast is not None and outcome.detailed is None
    assert runners[0].run_calls == 0 and runners[0].decode_steps == 16
    diagnostics = outcome.diagnostics
    decision = diagnostics["paired_budget_decision"]
    assert decision["accepted"] is True
    assert decision["candidate_relative_error_budget"] > .10 / 2.10
    exact_prefix = math.fsum(sample.duration_ns for sample in outcome.forecast.samples)
    assert diagnostics["exact_sampled_decode_ns"] == exact_prefix == 1600.
    assert decision["candidate_decode_interval"]["estimate_ns"] == pytest.approx(
        outcome.forecast.tpot_ns * (request.output_tokens - 1))
    assert decision["candidate_decode_interval"]["lower_ns"] == pytest.approx(
        exact_prefix + diagnostics["conditional_tail_lower_ns"])
    assert diagnostics["sampling_attempts"][-1]["paired_budget_decision"] == decision
    event = next(event for event in progress if event.stage == "fast_calibration")
    assert event.metadata["approximation"]["paired_budget_decision"] == decision
    assert diagnostics["validation_status"] == "not_validated_against_full_detailed"
    json.dumps(diagnostics, allow_nan=False)


def test_same_synthetic_runtime_without_reference_retains_independent_guard(monkeypatch):
    scenario, _, runners, _ = paired_runtime(monkeypatch)
    progress = []

    class RefusedBeforeDetailed(RuntimeError):
        pass

    def require_forecast(event):
        progress.append(event)
        if event.stage == "fast_mode_fallback":
            raise RefusedBeforeDetailed()

    with pytest.raises(RefusedBeforeDetailed):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=require_forecast))
    assert runners[0].run_calls == 0 and runners[0].decode_steps == 16
    decision = progress[-1].metadata["approximation"]["paired_budget_decision"]
    assert decision["strategy"] == "independent_symmetric_sufficient_bound"
    assert decision["accepted"] is False


@pytest.mark.parametrize("changes", [{"model_name": "wrong"}, {"prompt_tokens": 999},
                                     {"output_tokens": 1024}, {"memory_kind": "hbf"}])
def test_wrong_reference_cannot_start_sampling_or_detailed_fallback(monkeypatch, changes):
    scenario, _, runners, ref = paired_runtime(monkeypatch)
    with pytest.raises(ValueError):
        fast.simulate_fast(scenario, batch_lowerer=object(), paired_reference=replace(ref, **changes))
    assert not runners


def test_non_bf16_candidate_cannot_use_a_reference_to_bypass_domain_guard(monkeypatch):
    scenario, _, runners, ref = paired_runtime(monkeypatch)
    monkeypatch.setattr(fast, "_eligibility_reasons", lambda *args, **kwargs: ("fast_mode_requires_bf16_kv",))
    with pytest.raises(ValueError, match="BF16 fast domain"):
        fast.simulate_fast(scenario, batch_lowerer=object(), paired_reference=ref)
    assert not runners
