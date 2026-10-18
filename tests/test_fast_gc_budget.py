"""GC service bounds retain independent lower/upper and paired 10% guards."""
from copy import deepcopy
from itertools import permutations, product
import json
import math
from types import SimpleNamespace as NS

import pytest

from heterollm_sim import fast_simulation as fast
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.fast_budget import PairedTimingReference, paired_timing_decision
from heterollm_sim.simulation_options import FastDecodeConfig
from test_fast_decode_forecast import harness


OWNER = "hbf0.memory"


def observation(cycles=0, copies=0):
    return {"available": True, "gc_cycles": cycles, "gc_copied_pages": copies,
        "pressure_flushes": 0, "rmw_pages": 0, "recorded_task_count": 2,
        "expected_task_count": 2, "semantics": "cohort_local_staging_counter_deltas"}


def projection(**changes):
    guard = {"required": True, "qualified": True, "zero_copy_proven": True,
        "cycles_upper": 3, "copied_pages_upper": 0,
        "future_gc_erase_serial_ns": 6., "future_gc_non_erase_serial_ns": 0.,
        "future_write_placement_serial_guard_ns": 12., "carry_in_serial_guard_ns": 4.,
        "future_read_placement_serial_guard_ns": 13.,
        "temporary_read_pages_per_decode_upper": 2,
        "per_unit_costs": {"gc_copy_page_serial_upper_ns": 3.,
                           "ordinary_program_page_serial_upper_ns": 2.,
                           "ordinary_read_page_serial_upper_ns": 1.,
                           "read_cache_initial_state_serial_upper_ns": 3.}}
    guard.update(changes)
    device = {"physical_owner": OWNER, "observed_gc_cycles": 147380,
        "program_page_guard_budget": 50, "free_program_pages": 1,
        "block_erase_ns": 2., "ordinary_erase_serial_time_guard_ns": 8.,
        "gc_projection": guard}
    return NS(hbf_devices=(device,), safe=True, safety_reasons=(),
              to_metadata=lambda: {"safe": True, "physical_state_advanced": False})


def runtime():
    batches = []
    for index, (programs, cycles, copies, erases) in enumerate(
            [(10, 2, 2, 2), (9, 0, 0, 1), (8, 1, 1, 1),
             (7, 0, 0, 0), (6, 0, 0, 0)]):
        metadata = {"storage_gc_observations": {OWNER: observation(cycles, copies)},
                    "storage_traffic": {"pages_programmed": programs,
                                        "erase_operations": erases}}
        batches.append(NS(kind="decode", cost=NS(metadata=metadata)))
    return NS(batches=batches)


def envelope(runner=None, projected=None):
    if runner is None:
        runner = runtime()
        for batch in runner.batches:
            batch.cost.metadata["storage_gc_observations"][OWNER]["gc_copied_pages"] = 0
    return fast.maintenance_error_envelope(runner, projected or projection(),
        fast.SamplingWindow(0, 2, 2, 1), 100, 1.1)


def test_gc_prefix_uses_cohort_deltas_including_previous_cleanup_not_history():
    runner, projected = runtime(), projection()
    before = deepcopy([batch.cost.metadata for batch in runner.batches])
    guard_before = deepcopy(projected.hbf_devices)
    result = envelope(runner, projected)
    assert result["gc_budget_required"] is True and result["gc_observations_available"] is True
    assert result["gc_prefix_mapping_stable"] is False
    assert result["gc_prefix_recalibration_required"] is True
    assert result["prefix_gc_bias_guard_ns"] == pytest.approx(3 * 3. * 100 / 4 * 1.1)
    assert result["prefix_write_placement_bias_guard_ns"] == pytest.approx(37 * 2. * 100 / 4 * 1.1)
    assert result["prefix_erase_bias_guard_ns"] == pytest.approx(4 * 2. * 100 / 4 * 1.1)
    assert result["future_gc_erase_serial_guard_ns"] == 6.
    assert result["future_gc_non_erase_serial_guard_ns"] == 0.
    assert result["future_write_placement_serial_guard_ns"] == 12.
    assert result["future_gc_carry_in_serial_guard_ns"] == 4.
    assert result["prefix_read_placement_bias_guard_ns"] == pytest.approx((5 * 2 * 1. + 3.) * 100 / 4 * 1.1)
    assert result["future_read_placement_serial_guard_ns"] == 13.
    owner = result["gc_guard_by_owner"][0]
    assert owner["sampled_gc_cycles"] == 3  # Cumulative Prefill history is 147380.
    assert owner["sampled_gc_copied_pages"] == 3
    assert owner["selected_cohort_gc_copied_pages"] == 1
    assert owner["previous_cohort"]["gc_copied_pages"] == 2
    assert owner["sampled_ordinary_program_pages"] == 37
    assert result["gc_guard_physical_state_advanced"] is False
    assert [batch.cost.metadata for batch in runner.batches] == before
    assert projected.hbf_devices == guard_before
    assert envelope(runner, projected) == result
    json.dumps(result, allow_nan=False)


def test_gc_bounds_subtract_fit_bias_and_add_future_service_without_double_erase():
    maintenance = envelope()
    result = fast.conditional_error_budget(10000., 11000., 10500., .01, maintenance)
    prefix = sum(maintenance[name] for name in ("prefix_erase_bias_guard_ns", "prefix_gc_bias_guard_ns",
        "prefix_write_placement_bias_guard_ns", "prefix_read_placement_bias_guard_ns"))
    future = 8. + 6. + 0. + 12. + 13. + 4.
    assert result["conditional_tail_lower_ns"] == pytest.approx((10000. - prefix) / 1.01)
    assert result["conditional_tail_upper_ns"] == pytest.approx((11000. + future) / .99)
    parts = result["error_budget_breakdown"]
    assert parts["prefix_maintenance_total_guard_ns"] == prefix
    assert parts["future_maintenance_total_guard_ns"] == future
    assert parts["model_with_future_gc"]["upper_ns"] == 11006.
    assert parts["model_with_gc_carry_in"]["upper_ns"] == 11004.
    assert parts["model_with_future_read_placement"]["upper_ns"] == 11013.
    assert parts["semantics"] == "conditional_interval_bounds_not_additive_percentages"
    assert result["budget_unavailable_reason"] is None


def test_observed_live_copy_requires_a_clean_mapping_window_even_with_known_service():
    maintenance = envelope(runtime())
    assert maintenance["gc_observations_available"] is True
    assert maintenance["prefix_gc_bias_guard_ns"] > 0
    assert maintenance["gc_prefix_mapping_stable"] is False
    result = fast.conditional_error_budget(10000., 11000., 10500., .01, maintenance)
    assert result["conditional_relative_error_budget"] is None
    assert result["budget_unavailable_reason"] == "hbf_gc_prefix_mapping_requires_clean_window"
    # Known service breakdowns remain visible; they cannot waive mapping.
    assert result["error_budget_breakdown"]["future_read_placement_serial_guard_ns"] == 13.


def test_decode_read_geometry_cannot_stand_in_for_previous_prefill_cleanup():
    runner = runtime()
    runner.batches[0].kind = "prefill"
    result = envelope(runner)
    assert result["gc_observations_available"] is False
    assert result["gc_budget_unavailable_reason"] == "hbf_gc_previous_decode_read_geometry_unavailable"


@pytest.mark.parametrize("mutation", [
    lambda row: row.pop("storage_gc_observations"),
    lambda row: row.update(storage_gc_observations=None),
    lambda row: row["storage_gc_observations"].update({OWNER: None}),
    lambda row: row["storage_gc_observations"][OWNER].update(available=False),
    lambda row: row["storage_gc_observations"][OWNER].pop("gc_copied_pages"),
    lambda row: row["storage_gc_observations"][OWNER].update(gc_cycles=True),
    lambda row: row["storage_gc_observations"][OWNER].update(gc_cycles=-1),
    lambda row: row["storage_gc_observations"][OWNER].update(gc_cycles=0, gc_copied_pages=1),
    lambda row: row["storage_gc_observations"][OWNER].update(expected_task_count=3),
    lambda row: row["storage_gc_observations"][OWNER].update(recorded_task_count=0),
    lambda row: row["storage_gc_observations"][OWNER].update(semantics="cumulative"),
    lambda row: row["storage_traffic"].pop("pages_programmed"),
    lambda row: row["storage_traffic"].update(pages_programmed=0),
])
def test_missing_or_inconsistent_gc_observations_are_unavailable_never_zero(mutation):
    runner = runtime()
    mutation(runner.batches[0].cost.metadata)  # Previous cleanup must also be observed.
    maintenance = envelope(runner)
    assert maintenance["gc_observations_available"] is False
    assert maintenance["prefix_gc_bias_guard_ns"] is None
    assert maintenance["future_gc_erase_serial_guard_ns"] is None
    result = fast.conditional_error_budget(10000., 11000., 10500., .01, maintenance)
    assert result["conditional_tail_lower_ns"] is None
    assert result["conditional_tail_upper_ns"] is None
    assert result["conditional_relative_error_budget"] is None
    assert result["budget_unavailable_reason"] == "hbf_gc_maintenance_observations_unavailable"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key", ["pressure_flushes", "rmw_pages"])
def test_gc_budget_does_not_bypass_pressure_or_rmw(key):
    runner = runtime()
    runner.batches[-1].cost.metadata["storage_gc_observations"][OWNER][key] = 1
    result = envelope(runner)
    assert result["gc_observations_available"] is False
    assert result["gc_budget_unavailable_reason"] == "hbf_gc_prefix_has_pressure_or_rmw:" + OWNER


@pytest.mark.parametrize("changes", [
    {"qualified": False}, {"zero_copy_proven": False},
    {"future_gc_erase_serial_ns": None}, {"future_gc_non_erase_serial_ns": math.inf},
    {"future_write_placement_serial_guard_ns": -1.}, {"carry_in_serial_guard_ns": True},
    {"future_read_placement_serial_guard_ns": None},
    {"temporary_read_pages_per_decode_upper": None},
    {"per_unit_costs": {}},
])
def test_missing_or_unclosed_future_gc_projection_cannot_be_used(changes):
    result = envelope(projected=projection(**changes))
    assert result["gc_observations_available"] is False
    assert result["prefix_write_placement_bias_guard_ns"] is None
    assert result["future_gc_carry_in_serial_guard_ns"] is None


def test_gc_maintenance_can_exhaust_lower_bound_and_still_refuse():
    result = fast.conditional_error_budget(1000., 1100., 1050., .01, envelope())
    assert result["conditional_tail_lower_ns"] is None
    assert result["budget_unavailable_reason"] == "prefix_maintenance_budget_exhausts_lower_timing_bound"


def test_no_gc_path_preserves_original_interval_arithmetic_without_new_observations():
    runner, projected = runtime(), projection()
    projected.hbf_devices[0].update(observed_gc_cycles=0, program_page_guard_budget=1,
                                   free_program_pages=100, gc_projection={"required": False})
    for batch in runner.batches:
        batch.cost.metadata.pop("storage_gc_observations")
    maintenance = envelope(runner, projected)
    result = fast.conditional_error_budget(10000., 11000., 10500., .01, maintenance)
    assert maintenance["gc_budget_required"] is False
    assert result["conditional_tail_lower_ns"] == (10000. - maintenance["prefix_erase_bias_guard_ns"]) / 1.01
    assert result["conditional_tail_upper_ns"] == (11000. + 8.) / .99
    assert result["error_budget_breakdown"]["future_gc_erase_serial_guard_ns"] == 0.


def test_required_gc_fields_cannot_default_to_zero_in_conditional_composition():
    result = fast.conditional_error_budget(100., 100., 100., 0., {
        "prefix_erase_bias_guard_ns": 0., "future_erase_serial_guard_ns": 0.,
        "gc_budget_required": True, "gc_observations_available": True})
    assert result["conditional_relative_error_budget"] is None
    assert result["budget_unavailable_reason"] == "hbf_gc_maintenance_observations_unavailable"


def test_gc_candidate_still_requires_original_single_and_directional_pair_limits():
    parts = envelope()
    # Maintenance biases widen this otherwise narrow model, making its lower
    # bound insufficient. A matched excellent HBM reference cannot waive it.
    result = fast.conditional_error_budget(10000., 10010., 10005., 0., parts)
    reference = PairedTimingReference("model", 262144, 1024, "hbm", .1,
                                     10000., 9999., 10001., "existing-hbm-fast")
    decision = paired_timing_decision(decode_estimate_ns=10005.,
        decode_lower_ns=result["conditional_tail_lower_ns"],
        decode_upper_ns=result["conditional_tail_upper_ns"],
        target_relative_error=.1, candidate_memory_kind="hbf", reference=reference)
    assert decision["accepted"] is False
    assert decision["candidate_within_target"] is False
    assert decision["pair_ratio_direction"] == "hbf/hbm"
    assert decision["target_relative_error"] == .1


def gc_harness(monkeypatch, *, missing=False, copy_steps=(), options=None, growth=.03,
               erases=lambda step: 0):
    scenario, request, runners = harness(monkeypatch, growth=growth,
        options=options or FastDecodeConfig(), erase_counts=erases)
    base_runtime = fast._OnlineRuntime

    class ObservedRuntime(base_runtime):
        def record(self):
            for step, batch in enumerate(self.batches):
                batch.cost.metadata["storage_traffic"]["pages_programmed"] = 1
                if not missing:
                    copies = int(step in copy_steps)
                    batch.cost.metadata["storage_gc_observations"] = {OWNER: observation(1, copies)}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.record()

        def run_until_decode_steps(self, target):
            result = super().run_until_decode_steps(target)
            self.record()
            return result

    projected = projection(per_unit_costs={"gc_copy_page_serial_upper_ns": .01,
        "ordinary_program_page_serial_upper_ns": .01,
        "ordinary_read_page_serial_upper_ns": .01,
        "read_cache_initial_state_serial_upper_ns": .01})
    monkeypatch.setattr(fast, "_OnlineRuntime", ObservedRuntime)
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projected)
    return scenario, request, runners


def test_gc_forecast_uses_same_exact_prefix_and_never_runs_a_detailed_suffix(monkeypatch):
    scenario, request, runners = gc_harness(monkeypatch)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is not None and result.detailed is None
    assert runners[0].run_calls == 0
    assert runners[0].decode_steps == result.forecast.sampled_decode_steps == 16
    assert result.forecast.samples == tuple(fast.DecodeSample(step, 1000 + step, 100.)
                                             for step in range(1, 17))
    diagnostics = result.diagnostics
    assert diagnostics["method"] == "closed_empty_gc_maintenance_directional_paired_envelope/v5"
    assert diagnostics["gc_budget_required"] is True
    assert diagnostics["paired_budget_decision"]["accepted"] is True
    assert diagnostics["paired_budget_decision"]["target_relative_error"] == .1
    assert diagnostics["validation_status"] == "not_validated_against_full_detailed"
    assert diagnostics["physical_tail_transactions_recorded"] is False
    assert diagnostics["future_physical_state_advanced"] is False
    assert diagnostics["sampling_attempts"][-1]["error_budget_breakdown"]["gc_budget_required"] is True
    json.dumps(diagnostics, allow_nan=False)


def test_missing_gc_observation_refuses_before_any_detailed_fallback(monkeypatch):
    scenario, _, runners = gc_harness(monkeypatch, missing=True)
    progress = []

    class RequireForecast(RuntimeError):
        pass

    def callback(event):
        progress.append(event)
        if event.stage == "fast_mode_fallback":
            raise RequireForecast()

    with pytest.raises(RequireForecast):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=callback))
    assert runners[0].run_calls == 0 and runners[0].decode_steps == 16
    diagnostics = progress[-1].metadata["approximation"]
    assert diagnostics["effective_mode"] == "forecast_rejected"
    assert diagnostics["gc_observations_available"] is False
    assert diagnostics["fallback_reasons"] == ["hbf_gc_cohort_observations_unavailable:" + OWNER]
    assert diagnostics["predicted_decode_steps"] == 0


def test_one_live_copy_resets_after_cleanup_and_retains_all_real_tokens(monkeypatch):
    scenario, _, runners = gc_harness(monkeypatch, growth=.06, copy_steps=(10,))
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is not None and result.detailed is None
    assert runners[0].run_calls == 0
    assert result.forecast.sampled_decode_steps == runners[0].decode_steps == 23
    assert [sample.step for sample in result.forecast.samples] == list(range(1, 24))
    assert result.diagnostics["exact_sampled_decode_ns"] == 2300.
    attempts = result.diagnostics["sampling_attempts"]
    assert attempts[0]["budget_unavailable_reason"] == "hbf_gc_prefix_mapping_requires_clean_window"
    assert attempts[0]["forecast_accepted"] is False
    assert attempts[-1]["gc_prefix_mapping_stable"] is True
    reset = result.diagnostics["sampling_resets"][0]
    assert reset["cause"] == "observed_live_gc_copy_and_following_cohort_cleanup"
    assert reset["last_gc_copy_decode_step"] == 10
    assert reset["cleanup_through_decode_step"] == 11
    assert reset["new_warmup"] == 11 and reset["next_target_decode_steps"] == 23
    assert reset["exact_prefix_samples_and_latency_preserved"] is True
    assert "erase_cohorts" not in reset


def test_budgeted_empty_collections_do_not_move_clean_window_on_every_erase(monkeypatch):
    scenario, _, runners = gc_harness(monkeypatch, erases=lambda step: 1)
    result = fast.simulate_fast(scenario, batch_lowerer=object())
    assert result.forecast is not None
    assert result.diagnostics["sampling_resets"] == []
    assert runners[0].decode_steps == 16 and runners[0].run_calls == 0


def test_rejected_empty_gc_budget_expands_without_repeated_erase_resets(monkeypatch):
    options = FastDecodeConfig(max_sample_tokens=28)
    scenario, _, runners = gc_harness(monkeypatch, growth=.06, options=options,
                                      erases=lambda step: 1)
    projected = projection(future_write_placement_serial_guard_ns=1000.)
    monkeypatch.setattr(fast, "project_fast_tail", lambda *args, **kwargs: projected)
    progress = []

    class RequireForecast(RuntimeError):
        pass

    def callback(event):
        progress.append(event)
        if event.stage == "fast_mode_fallback":
            raise RequireForecast()

    with pytest.raises(RequireForecast):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=callback))
    diagnostics = progress[-1].metadata["approximation"]
    assert diagnostics["sampling_resets"] == []
    assert [attempt["sampled_steps"] for attempt in diagnostics["sampling_attempts"]] == [16, 19, 22, 25, 28]
    assert runners[0].decode_steps == 28 and runners[0].run_calls == 0
    assert diagnostics["fallback_reasons"] == ["prefix_calibration_exceeds_error_budget"]


def test_live_copy_reset_over_cap_refuses_without_running_detailed_suffix(monkeypatch):
    scenario, _, runners = gc_harness(monkeypatch, growth=.06, copy_steps=(15,),
                                      options=FastDecodeConfig(max_sample_tokens=16))
    progress = []

    class RequireForecast(RuntimeError):
        pass

    def callback(event):
        progress.append(event)
        if event.stage == "fast_mode_fallback":
            raise RequireForecast()

    with pytest.raises(RequireForecast):
        fast.simulate_fast(scenario, batch_lowerer=object(),
            execution_control=ExecutionControl(progress_callback=callback))
    diagnostics = progress[-1].metadata["approximation"]
    assert runners[0].decode_steps == 16 and runners[0].run_calls == 0
    reset = diagnostics["sampling_resets"][0]
    assert reset["next_target_decode_steps"] == 28
    assert reset["within_sample_budget"] is False
    assert diagnostics["fallback_reasons"] == ["maintenance_recalibration_exceeds_sample_budget"]


def lru_misses(capacity, initial, trace):
    cache, misses = list(initial), 0
    for tag in trace:
        if tag not in cache:
            misses += 1
        else:
            cache.remove(tag)
        cache.append(tag)
        if len(cache) > capacity:
            cache.pop(0)
    return misses


@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_lru_temporary_insertion_bound_against_persistent_only_trace(capacity):
    # Both traces share their arbitrary initial state. After k distinct
    # persistent accesses an insertion's state difference is gone; before
    # that, at most k persistent misses plus its own miss can be charged.
    tags = tuple(range(capacity + 2))
    for initial in permutations(tags, capacity):
        for trace in product(range(capacity + 1), repeat=5):
            baseline = lru_misses(capacity, initial, trace)
            for position in range(len(trace) + 1):
                polluted = trace[:position] + ("temporary",) + trace[position:]
                assert lru_misses(capacity, initial, polluted) - baseline <= capacity + 1


def test_lru_k_extra_persistent_misses_are_possible_so_coefficient_cannot_be_one():
    for capacity in (1, 2, 3):
        initial = tuple(range(capacity))  # Oldest first.
        persistent = initial
        assert lru_misses(capacity, initial, persistent) == 0
        assert lru_misses(capacity, initial, ("temporary",) + persistent) == capacity + 1
