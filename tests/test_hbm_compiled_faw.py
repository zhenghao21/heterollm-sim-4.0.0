"""Compiled tFAW is a bounded representation of the HBM command controller."""
import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from heterollm_sim._compiled_hbm import execute_hbm_numeric
from heterollm_sim._hbm_control import _Controller
from heterollm_sim._hbm_numeric import _faw_ready
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration, assert_same, assert_core_state_equal, assert_batch_matches_scalar


def test_fifth_activation_waits_for_the_first_anchor_plus_faw():
    c = configuration(channels=1, four_activate_window_ns=20.,
                      activate_same_group_ns=0., activate_different_group_ns=0.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    anchors = []
    for bank in range(5):
        assert_same(reference, numeric, AccessRequest(str(bank), "read", bank * c.lane_count * c.burst_bytes, 32))
        anchors.append(max(value["last_activate"] for value in numeric._hbm_control_state["banks"].values()))
    assert anchors == [1., 3., 5., 7., 21.]
    assert numeric.timeline.last_intervals["dram:row_command:0:0"] == (20., 21.5)


@pytest.mark.parametrize("calendar,earliest,frontier,expected", [
    ([100.] * 4, 1., 0., 1.),
    ([100.] * 4, 91., 0., 120.),
    ([1., 3., 5., 7., 40., 42., 44., 46.], 8., 0., 21.),
    ([1., 3., 5., 7., 40., 42., 44., 46.], 29., 22., 60.),
    ([1., 3., 5., 7.], 21., 21., 21.),
])
def test_faw_future_reservations_and_expiration_have_independent_expectations(calendar, earliest, frontier, expected):
    c = configuration(four_activate_window_ns=20.,
                      activate_same_group_ns=0., activate_different_group_ns=0.)
    state = {"activate": {}, "activate_window": calendar.copy()}
    windows = np.zeros((1, len(calendar) + 5))
    windows[0, :len(calendar)] = calendar
    sizes = np.array([len(calendar)], dtype=np.int64)
    timing = np.zeros(31)
    timing[0], timing[30] = 1., 20.
    assert DramCore(c)._bank_group_activate_ready(state, 0, earliest, frontier, clock_quantum=.5) == expected
    assert _faw_ready(0, earliest, frontier, timing, windows, sizes) == expected
    assert windows[0, :sizes[0]].tolist() == state["activate_window"]


@pytest.mark.parametrize("future,group_gap,width,expected_anchor", [
    ([100., 104., 108., 112.], 0., 20.1, 2.),
    ([16., 20., 24., 28.], 3.1, 24.5, 42.),
])
def test_future_calendar_insertion_remains_sorted_and_rechecks_clock_rounding(future, group_gap, width, expected_anchor):
    c = configuration(command_clock_ns=2., data_rate_mt_s=2000., read_latency_ns=16.,
        write_latency_ns=8., four_activate_window_ns=width,
        activate_same_group_ns=group_gap, activate_different_group_ns=0.,
        column_same_group_ns=8., column_different_group_ns=4.,
        column_different_sid_ns=6., precharge_to_precharge_ns=4.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for core in (reference, numeric):
        pc = _Controller(core, AccessRequest("seed", "read", 0, 32)).pc((0, 0, 0))
        pc["activate_window"] = future.copy()
        pc["activate"] = {0: 0.}
    assert_same(reference, numeric, AccessRequest("before-future", "read", 0, 32))
    calendar = numeric._hbm_control_state["pcs"][(0, 0, 0)]["activate_window"]
    assert calendar == sorted(calendar)
    assert expected_anchor in calendar
    assert all(calendar[i + 4] >= calendar[i] + width for i in range(len(calendar) - 4))


@pytest.mark.parametrize("clock", [1., .7])
@pytest.mark.parametrize("refresh", [False, True])
def test_faw_mixed_commands_refresh_and_idle_match_exactly(clock, refresh):
    c = configuration(command_clock_ns=clock, data_rate_mt_s=4000. / clock,
        read_latency_ns=8. * clock, write_latency_ns=4. * clock,
        four_activate_window_ns=31.3, maximum_active_ns=900.,
        refresh_policy="all_bank" if refresh else "disabled",
        refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    span = c.lane_count * c.bank_count * c.row_bytes
    for i, (op, address, count, arrival) in enumerate((
        ("read", 3, 9007, 0.), ("write", 65, 4096, 0.),
        ("read", span + 5, 9001, 5.), ("read", 32, 1024, 1e6))):
        assert_same(reference, numeric, AccessRequest(str(i), op, address, count, arrival))


def test_large_request_keeps_faw_calendar_bounded_and_remains_compiled():
    c = configuration(four_activate_window_ns=31., refresh_policy="all_bank",
                      refresh_interval_ns=400., refresh_cycle_ns=40.)
    core = DramCore(c, capture_details=False)
    result = core.submit(AccessRequest("large", "read", 0, 1024**2))
    assert result.counters["compiled_hbm_float64"]
    assert result.counters["activate_commands"] > 1000
    assert result.counters["refresh_commands"] > 0
    assert max(len(pc["activate_window"]) for pc in core._hbm_control_state["pcs"].values()) <= 5


def test_faw_batch_and_copied_runtime_state_preserve_following_execution():
    c = configuration(four_activate_window_ns=31., max_outstanding_requests=2,
                      refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.)
    scalar, batch = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    requests = tuple(AccessRequest(str(i), "write" if i % 2 else "read", i * 157, 1024, i * 2.) for i in range(6))
    expected = tuple(scalar.submit(request) for request in requests)
    assert_batch_matches_scalar(batch.submit_many(requests), expected)
    assert all(result.counters["compiled_hbm_float64"] for result in expected)
    assert_core_state_equal(batch, scalar)
    restored = copy.deepcopy(batch)
    following = AccessRequest("after-snapshot", "read", 131072, 4096, 1e6)
    assert restored.submit(following) == batch.submit(following)
    assert_core_state_equal(restored, batch)


def test_faw_numeric_overflow_declines_without_live_mutation():
    c = configuration(four_activate_window_ns=20.)
    core = DramCore(c, capture_details=False)
    core.submit(AccessRequest("warm", "read", 0, 512))
    saved = copy.deepcopy(core)
    assert execute_hbm_numeric(core, AccessRequest("huge", "read", 4096, 32, math.ldexp(1., 64))) is None
    assert_core_state_equal(core, saved)


@pytest.mark.parametrize("mapping", ["bank_interleaved", "row_local"])
def test_both_mappings_keep_non_power_of_two_geometry_and_multiple_stacks(mapping):
    c = configuration(channels=3, stacks=2, bank_groups_per_rank=3, rows_per_bank=24,
                      activated_row_msb=4, address_mapping=mapping,
                      four_activate_window_ns=31., refresh_policy="all_bank",
                      refresh_interval_ns=400., refresh_cycle_ns=40.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for index, (op, address, count, arrival) in enumerate((
        ("read", 3, 19001, 0.), ("write", 12301, 17007, 1.),
        ("read", c.computed_capacity_bytes - 1025, 1025, 5000.))):
        assert_same(reference, numeric, AccessRequest(str(index), op, address, count, arrival))


@pytest.mark.parametrize("mapping", ["bank_interleaved", "row_local"])
@pytest.mark.parametrize("scheduler", ["fcfs", "row_hit_first", "read_write_batch"])
@pytest.mark.parametrize("queue_scope", ["device", "channel"])
def test_full_timing_numeric_matches_reference_with_scheduler_and_queue(mapping, scheduler, queue_scope, monkeypatch):
    from heterollm_sim import _compiled_hbm
    c = configuration(address_mapping=mapping, scheduler=scheduler, scheduler_window=4,
                      queue_scope=queue_scope, max_outstanding_requests=2,
                      four_activate_window_ns=31., refresh_policy="all_bank",
                      refresh_interval_ns=400., refresh_cycle_ns=40.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    span = c.lane_count * c.bank_count * c.row_bytes
    requests = tuple(AccessRequest(str(i), op, address, count, arrival)
        for i, (op, address, count, arrival) in enumerate((
            ("read", 0, 64, 0.), ("write", span, 64, 0.), ("read", 0, 64, 0.),
            ("write", 64, 96, 0.), ("read", 65536, 4096, 1.),
            ("write", 0, 129, 10000.), ("read", 0, 1024, 10000.))))
    with monkeypatch.context() as patch:
        patch.setattr(_compiled_hbm, "_pack_hbm_numeric", lambda *args, **kwargs: None)
        expected = reference.submit_many(requests)
    actual = numeric.submit_many(requests)
    assert all(row.counters["compiled_hbm_float64"] for row in actual)
    assert tuple(replace(row, counters={k: v for k, v in row.counters.items()
                                       if k != "compiled_hbm_float64"}) for row in actual) == expected
    assert_core_state_equal(numeric, reference)
