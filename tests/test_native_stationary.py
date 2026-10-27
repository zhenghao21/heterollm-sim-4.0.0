"""The stationary fast forecast composes native samples of one refill step."""
from dataclasses import replace

import pytest

from heterollm_sim import native_stationary as stationary
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig
import native_toy


def test_refill_schedule_follows_slots_freed_at_each_step():
    # Centered ages 0..3 free one slot per step; with a pool of two, each
    # request waits two steps in the pool and emits at the end of the third.
    waits, most, lag = stationary._refill_schedule([0, 1, 2, 3], 4, 2, 3, 6)
    assert waits == [3, 3, 3] and most == 1 and lag == 1
    # Two slots freed at once outrun a pool of one.
    waits, most, lag = stationary._refill_schedule([1, 1], 4, 1, 3, 6)
    assert most == 2


def test_schedule_counts_the_imports_each_group_starts_together():
    # One slot frees per step: every measured step lands one import.
    single = stationary._group_refill_schedule([[0, 1, 2, 3]], 4, 2, 3, 6)
    assert single["busiest_landings"] == {1: 6} and single["fewest_landing_steps"] == 1
    # Two seeds finishing together start two imports at each refill boundary.
    pairs = stationary._group_refill_schedule([[1, 1]], 4, 2, 3, 6)
    assert pairs["busiest_landings"] == {2: 1} and pairs["fewest_landing_steps"] == 4
    assert pairs["rho"] == 1.
    # Across ep groups a step counts its busiest group.
    groups = stationary._group_refill_schedule([[1, 1], [1, 2]], 4, 2, 3, 6)
    assert groups["busiest_landings"] == {1: 1, 2: 1}


def test_required_pool_is_the_smallest_keeping_the_measured_lead():
    # One slot frees per step from boundary 3; a pool of p keeps later imports
    # p steps ahead, but the first refill takes an import started at boundary 0.
    ages = [[0, 1, 2, 3]]
    assert [stationary._group_refill_schedule(ages, 6, pool, 5, 10)["lag"] for pool in (1, 2, 3, 4)] == [1, 2, 3, 3]
    assert stationary._required_group_pool(ages, 6, 1, 5, 10, 2) == 2
    assert stationary._required_group_pool(ages, 6, 2, 5, 10, 3) == 3
    assert stationary._required_group_pool(ages, 6, 1, 5, 10, 4) is None
    # Two slots freed at one boundary need two prepared imports.
    assert stationary._required_group_pool([[1, 1]], 4, 1, 3, 6, 1) == 2


def test_harmonic_estimate_meets_the_independent_gate_up_to_a_ten_percent_width():
    target = .1 / 2.1
    assert stationary._interval(100., 110.)["relative_error_budget"] == pytest.approx(target)
    assert stationary._within(stationary._interval(100., 109.999), target)
    assert not stationary._within(stationary._interval(100., 110.001), target)


def _run(origin, layers, completions, end, first_access=None):
    return {"origin_ns": origin, "start_ns": origin, "end_ns": end, "task_count": 10,
            "layer_ids": layers, "layer_completion_ns": dict(zip(layers, completions)),
            "first_access_ns": first_access or {}}


def _prepared(classes, guard=0.):
    return {"inventory": [{"layer_id": f"L{index}", "class_key": key} for index, key in enumerate(classes)],
            "query_decode_guard_ns": guard}


def test_step_composes_measured_class_steps_and_widens_by_their_drift():
    prepared = _prepared(["a"] * 6)
    samples = {"runs": {
        "calibration": _run(0., ["L0", "L1", "L2"], [30., 40., 50.], 50.),
        "holdout": _run(100., ["L4", "L5"], [130., 142.], 145.),
        "entry_at_holdout_origin": _run(100., ["L0"], [131.], 131.),
        "exit_at_calibration_origin": _run(0., ["L5"], [10.], 13.)}}
    lower, upper, parts = stationary._step_service(prepared, samples)
    # Steps 10, 10 (calibration) and 12 (holdout): means drift by 2.
    assert parts["classes"][0]["step_ns"] == {"lower": 8., "upper": 14.}
    # Entry 30/31 widened by 1, exit 3/3, five composed steps.
    assert (lower, upper) == (29. + 3. + 5 * 8., 32. + 3. + 5 * 14.)


def test_each_observation_is_widened_by_its_boundary_row_state_bound():
    prepared = _prepared(["a"] * 6)
    samples = {"runs": {
        "calibration": _run(0., ["L0", "L1", "L2"], [30., 40., 50.], 50., {"entry": 2., "L1": 1.}),
        "holdout": _run(100., ["L4", "L5"], [130., 142.], 145., {"L5": 3., "exit": .5}),
        "entry_at_holdout_origin": _run(100., ["L0"], [131.], 131., {"entry": 0.}),
        "exit_at_calibration_origin": _run(0., ["L5"], [10.], 13., {"exit": 1.})}}
    lower, upper, parts = stationary._step_service(prepared, samples)
    # Steps 10+-1, 10, 12+-3 widened again by the drift of 2.
    assert parts["classes"][0]["step_ns"] == {"lower": 7., "upper": 17.}
    # Entry 30+-2 and 31 widened by 1; exit 3+-.5 and 3+-1.
    assert parts["entry_interval_ns"] == [27., 33.] and parts["exit_interval_ns"] == [2., 4.]
    assert (lower, upper) == (27. + 2. + 5 * 7., 33. + 4. + 5 * 17.)


def test_every_class_must_step_in_both_runs():
    prepared = _prepared(["a", "a", "a", "b", "b"])
    samples = {"runs": {
        "calibration": _run(0., ["L0", "L1", "L2"], [30., 40., 50.], 50.),
        "holdout": _run(100., ["L3", "L4"], [130., 140.], 145.),
        "entry_at_holdout_origin": _run(100., ["L0"], [130.], 130.),
        "exit_at_calibration_origin": _run(0., ["L4"], [10.], 15.)}}
    with pytest.raises(ValueError, match="every_layer_class"):
        stationary._step_service(prepared, samples)


def _imports(*rows):
    """Each row: one import (id) or a set run together (tuple of ids, with
    per-id written bytes, spans and phases)."""
    quiet = _run(0., ["L0", "L1", "L2"], [30., 40., 50.], 50.)
    samples = []
    for identity, total, written, span, layers, end, phases in rows:
        identities = list(identity) if isinstance(identity, tuple) else [identity]
        per = lambda value: dict(zip(identities, value if isinstance(identity, tuple) else (value,)))
        observed = {"written_bytes_by_request": per(written), "write_phases_by_request": per(phases),
                    "import_spans_ns": {key: value for key, value in per(span).items() if value is not None}}
        busy = {**_run(0., ["L0", "L1", "L2"], layers, end), "concurrent_imports": observed}
        samples.append({"request_ids": identities, "logical_bytes": total, "concurrent": busy})
    return {"runs": {"calibration": quiet}, "pd_imports": {"samples": samples}}


def test_finished_import_scales_its_later_end_and_unfinished_import_uses_the_layer_slope():
    terms, rows = stationary._import_extra(_imports(
        ("x", 100, 50, (-10., 20.), [32., 42., 52.], 52., 3),
        # 25 % slower post-entry steps while writing 50 bytes over 60 ns.
        ("y", 100, 50, None, [30., 42.5, 55.], 60., 3)))
    assert rows[0]["mode"] == "complete_import_in_window" and rows[0]["extra_ns"] == 4.
    assert rows[1]["mode"] == "post_entry_layer_slope"
    assert rows[1]["span_ns"] == pytest.approx(120.) and rows[1]["extra_ns"] == pytest.approx(24.)
    assert list(terms) == [1] and terms[1]["extra_ns"] == pytest.approx((0., 44.))


def test_imports_run_together_give_their_joint_extra_and_span():
    terms, rows = stationary._import_extra(_imports(
        ("x", 100, 100, (0., 20.), [32., 42., 52.], 52., 3),
        ("y", 100, 100, (0., 20.), [32., 42., 52.], 52., 3),
        # Two imports of 100 bytes written together by 40 ns delay the
        # quiet work by 9 ns, not twice the single 2 ns.
        (("x", "y"), 200, (100, 100), ((0., 30.), (0., 40.)), [35., 49., 59.], 59., (3, 3)),
        (("z", "w"), 200, (100, 100), ((0., 40.), (0., 40.)), [35., 47., 57.], 57., (3, 3))))
    assert [row["concurrent_imports"] for row in rows] == [1, 1, 2, 2]
    assert rows[2]["mode"] == "complete_import_in_window" and rows[2]["span_ns"] == 40.
    assert terms[1]["extra_ns"] == (2., 2.) and terms[2]["extra_ns"] == (5., 11.)
    assert terms[2]["span_ns"] == (40., 40.)
    # One set of a count is not enough to bound its joint extra.
    terms, _rows = stationary._import_extra(_imports(*[
        ("x", 100, 100, (0., 20.), [32., 42., 52.], 52., 3),
        ("y", 100, 100, (0., 20.), [32., 42., 52.], 52., 3),
        (("x", "y"), 200, (100, 100), ((0., 30.), (0., 40.)), [35., 49., 59.], 59., (3, 3))]))
    assert list(terms) == [1]


def test_two_in_flight_imports_are_required():
    with pytest.raises(ValueError, match="two_in_flight"):
        stationary._import_extra(_imports(("x", 100, 50, (-10., 20.), [32., 42., 52.], 52., 3)))


def _fast(scenario, calibration, holdout):
    from heterollm_sim.reporting import run_scenario
    return run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(calibration_layers=calibration, holdout_layers=holdout))),
        retention_policy="aggregate")


@pytest.mark.parametrize("scenario,calibration,holdout,together", [
    # Eight requests of six tokens: some boundaries start two imports at once.
    (lambda: native_toy.toy_scenario(8, 1., batch_size=8, output=6, warmup=5, measurement=10), 4, 3, [2]),
    # The steady state keeps the DRAM rows each step leaves open; the
    # samples start from the cohort boundary's rows (first-access bounds).
    (lambda: native_toy.moe_toy_scenario(8, batch_size=4, output=6, warmup=5, measurement=10), 4, 4, []),
], ids=["dense", "moe"])
def test_forecast_brackets_the_full_native_continuous_refill_window(scenario, calibration, holdout, together):
    scenario = scenario()
    fast, window, observed, complete = native_toy.stationary_and_full_window(scenario, calibration, holdout)
    assert fast.approximation["rejection_reasons"] == []
    assert fast.sampled is fast.detailed is None
    busiest = fast.approximation["error_budget"]["pd_import_group_factor"]["busiest_landings"]
    assert [count for count in busiest if count > 1] == together
    # Two single imports, and two sets of k for every k started together.
    assert sorted(row.role for row in fast.layer_window_samples) == [
        "calibration", "entry", "exit", "holdout", *["pd_import"] * (2 + 2 * len(together))]
    assert window["batch_sizes"] == [scenario.workload.scheduler.max_num_seqs] * window["measured_decode_cohorts"]
    bounds = fast.native_window_forecast.metric_bounds
    for name, value in observed.items():
        assert bounds[name]["lower"] <= value <= bounds[name]["upper"], name
    assert fast.native_window_forecast.metrics["complete_fresh_requests"] == complete


def test_short_stack_and_slow_refill_are_rejected_not_extrapolated():
    short = _fast(native_toy.toy_scenario(2, 1., batch_size=2, output=3), 4, 2)
    assert short.native_window_forecast is None
    assert short.approximation["rejection_reasons"] == ["native_stationary_stack_too_short_for_disjoint_runs"]
    # Maverick layer classes (dense, MoE chunked, MoE full) each step in
    # both runs; over 0.1 Gb/s PD links its imports outlast a step, so a
    # slot would wait.
    slow = _fast(native_toy.slow_pd_ingress(native_toy.maverick_pattern_scenario(
        8, batch_size=4, output=6, warmup=5, measurement=10), .1), 4, 4)
    assert slow.native_window_forecast is None
    assert "native_stationary_refill_not_always_ready" in slow.approximation["rejection_reasons"]
    assert len(slow.approximation["error_budget"]["step"]["classes"]) == 3
