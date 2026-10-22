"""Explicit, uncalibrated bank-group eligibility constraints on the real core."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_mapping import map_dram_address
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig


def config(**changes):
    changes.setdefault("capacity_bytes", None)
    return replace(DramConfig(
        kind="HBM", bank_groups_per_rank=2, banks_per_group=2,
        rows_per_bank=8, row_bytes=256, burst_bytes=64,
        open_ns=0, close_ns=0, read_latency_ns=0, write_latency_ns=0,
        burst_interval_ns=1, read_recovery_ns=0, write_recovery_ns=0,
        interface_bandwidth_gb_s=64,
    ), **changes)


# The rotated mapping alternates groups: in the default 2x2 geometry address
# 64 reaches bank 2 (group 1) and address 128 reaches bank 1 (group 0).
SAME_GROUP, OTHER_GROUP = 128, 64


def request(name, address, operation="read", *, arrival=0, byte_count=64):
    return AccessRequest(name, operation, address, byte_count, arrival_ns=arrival)


def stage_start(result, name="COMMAND_RESERVATION"):
    return next(stage.start_ns for stage in result.stages if stage.name == name)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_column_spacing_retains_same_group_across_another_group(operation):
    core = DramCore(config(column_same_group_ns=10, column_different_group_ns=2))
    rows = core.submit_many(tuple(request(str(i), address, operation)
                                  for i, address in enumerate((0, OTHER_GROUP, SAME_GROUP))))
    assert [stage_start(row) for row in rows] == [0, 2, 10]
    assert [row.completion_ns for row in rows] == [1, 3, 11]
    assert sum(core.timeline.busy_ns.values()) == 6  # 3 command + 3 data
    assert sum(core.timeline.bytes_moved.values()) == 192


def test_column_histories_are_per_direction_and_survive_opposite_commands():
    core = DramCore(config(column_same_group_ns=50, column_different_group_ns=20))
    rows = core.submit_many((request("readA", 0), request("writeB", OTHER_GROUP, "write"),
                             request("readA-again", SAME_GROUP)))
    assert [stage_start(row) for row in rows] == [0, 1, 50]


def test_regrouping_fixed_bank_count_changes_timing_only_when_constraints_enabled():
    def run(groups, banks, same_gap, different_gap):
        core = DramCore(config(bank_groups_per_rank=groups, banks_per_group=banks,
                               column_same_group_ns=same_gap, column_different_group_ns=different_gap))
        rows = core.submit_many((request("bank0", 0), request("second-bank", 64)))
        return core, rows[-1].completion_ns
    disabled = [run(groups, banks, 0, 0) for groups, banks in ((1, 4), (2, 2))]
    enabled = [run(groups, banks, 10, 2) for groups, banks in ((1, 4), (2, 2))]
    assert [completion for _, completion in disabled] == [2, 2]
    assert [completion for _, completion in enabled] == [11, 3]
    for core, _ in disabled + enabled:
        assert core.config.bank_count == 4
        assert core.config.computed_capacity_bytes == 8192
        assert sum(core.timeline.busy_ns.values()) == 4
        assert sum(core.timeline.bytes_moved.values()) == 128


def test_activation_spacing_retains_group_history_and_allows_act_overlap():
    core = DramCore(config(open_ns=3, activate_same_group_ns=10, activate_different_group_ns=2))
    rows = core.submit_many(tuple(request(str(i), address)
                                  for i, address in enumerate((0, OTHER_GROUP, SAME_GROUP))))
    assert [stage_start(row, "ACTIVATE") for row in rows] == [0, 2, 10]
    assert [stage_start(row) for row in rows] == [3, 5, 13]
    assert sum(core.timeline.busy_ns.values()) == 15  # 9 ACT + 3 command + 3 data


def test_fifth_activation_obeys_sliding_four_activation_window():
    core = DramCore(config(bank_groups_per_rank=16, banks_per_group=1,
                           open_ns=1, four_activate_window_ns=25))
    rows = core.submit_many(tuple(request(str(i), i * 64) for i in range(9)))
    assert [stage_start(row, "ACTIVATE") for row in rows] == [0] * 4 + [25] * 4 + [50]
    assert core._bank_group_state[(0, 0, 0, 0)]["activate_window"] == [0] * 4 + [25] * 4 + [50]
    assert sum(core.timeline.busy_ns.values()) == 27


@pytest.mark.parametrize("address,expected", [(0, 11), (SAME_GROUP, 11), (OTHER_GROUP, 6)])
def test_write_to_read_gates_read_command_from_write_data_end(address, expected):
    core = DramCore(config(write_latency_ns=3, read_latency_ns=1,
                           write_to_read_same_group_ns=7, write_to_read_different_group_ns=2))
    first = core.submit(request("writeA", 0, "write"))
    second = core.submit(request("read", address))
    assert first.completion_ns == 4
    assert stage_start(second) == expected
    assert second.completion_ns == expected + 2


def test_write_to_read_keeps_older_group_write_instead_of_only_latest_write():
    core = DramCore(config(write_latency_ns=5,
                           write_to_read_same_group_ns=20, write_to_read_different_group_ns=2))
    rows = core.submit_many((request("writeA", 0, "write"), request("writeB", OTHER_GROUP, "write"),
                             request("readA", SAME_GROUP)))
    assert [row.completion_ns for row in rows[:2]] == [6, 7]
    assert stage_start(rows[2]) == 26  # max(A-end 6 + 20, B-end 7 + 2)


def test_group_wait_and_data_bus_turnaround_take_max_without_double_billing():
    core = DramCore(config(write_latency_ns=3, read_latency_ns=1, write_to_read_ns=100,
                           write_to_read_same_group_ns=7))
    first = core.submit(request("writeA", 0, "write"))
    second = core.submit(request("readA", SAME_GROUP))
    assert first.completion_ns == 4
    assert stage_start(second) == 11
    assert stage_start(second, "BURST_TRANSFER") == 104
    assert second.completion_ns == 105
    assert sum(core.timeline.busy_ns.values()) == 4
    assert sum(core.timeline.bytes_moved.values()) == 128


@pytest.mark.parametrize("gate", ["activate_same_group_ns", "four_activate_window_ns"])
def test_enabling_activation_gate_does_not_enable_zero_write_to_read_gate(gate):
    core = DramCore(config(write_latency_ns=100, **{gate: 10}))
    first = core.submit(request("writeA", 0, "write"))
    second = core.submit(request("readA", 0))
    assert first.completion_ns == 101
    assert stage_start(second) == 1  # WTR=0 imposes no command gate
    assert second.completion_ns == 102  # real shared data bus still waits


def test_zero_cross_group_wtr_is_disabled_even_when_same_group_wtr_is_active():
    core = DramCore(config(write_latency_ns=100, write_to_read_same_group_ns=20))
    core.submit(request("writeA", 0, "write"))
    row = core.submit(request("readB", OTHER_GROUP))
    assert stage_start(row) == 1
    assert row.completion_ns == 102


@pytest.mark.parametrize("dimension", ["stacks", "dies_per_stack", "channels", "ranks_per_channel"])
def test_group_constraints_are_independent_across_each_scope_dimension(dimension):
    geometry = config(**{dimension: 2}, column_same_group_ns=100, activate_same_group_ns=100,
                      four_activate_window_ns=100, write_to_read_same_group_ns=100)
    def scope(mapping):
        return mapping.stack, mapping.die, mapping.lane, mapping.rank
    first_scope = scope(map_dram_address(geometry, 0))
    other_address = next(address for address in range(64, geometry.effective_capacity_bytes, 64)
                         if scope(map_dram_address(geometry, address)) != first_scope)
    core = DramCore(geometry)
    core.submit(request("first-scope", 0, "write"))
    other = core.submit(request("other-scope", other_address))
    assert stage_start(other, "ACTIVATE") == 0
    assert stage_start(other) < 100
    assert other.completion_ns < 100
    assert len(core._bank_group_state) == 2


def test_future_bank_reservation_and_reversed_arrivals_preserve_act_history():
    timeline = ResourceTimeline(
        ready_ns={"dram:bank:0:0:0:0:0:0": 100},
        lane_available={"dram:command:0": [0., 0.]},
    )
    core = DramCore(config(column_same_group_ns=10, column_different_group_ns=2,
                           four_activate_window_ns=25), timeline)
    rows = core.submit_many((request("A", 0, arrival=1), request("B", OTHER_GROUP, arrival=0),
                             request("A2", SAME_GROUP, arrival=0)))
    assert [stage_start(row, "ACTIVATE") for row in rows] == [100, 1, 1]
    assert [stage_start(row) for row in rows] == [100, 102, 110]
    assert rows[1].counters["queue_wait_ns"] == 1
    state = core._bank_group_state[(0, 0, 0, 0)]
    assert state["activate"] == {0: 100, 1: 1}
    assert state["activate_window"] == [1, 1, 100]


def test_future_read_does_not_gate_independent_write_command():
    core = DramCore(config(write_latency_ns=100, write_to_read_same_group_ns=7),
                    ResourceTimeline(lane_available={"dram:command:0": [0., 0.]}))
    rows = core.submit_many((request("writeA", 0, "write"), request("readA", 0),
                             request("writeB", OTHER_GROUP, "write")))
    assert [stage_start(row) for row in rows] == [0, 108, 1]
    assert core._bank_group_state[(0, 0, 0, 0)]["column"] == {
        "read": {0: 108}, "write": {0: 0, 1: 1},
    }


@pytest.mark.parametrize("field", ["write_to_read_same_group_ns", "write_to_read_different_group_ns"])
def test_write_to_read_gate_does_not_serialize_read_only_work(field):
    core = DramCore(config(banks_per_group=1, burst_interval_ns=0, **{field: 1}),
                    ResourceTimeline(lane_available={"dram:command:0": [0., 0.],
                                                      "dram:data:0": [0., 0.]}))
    core.timeline.reserve("dram:bank:0:0:0:0:0:0", 0, 100)
    rows = core.submit_many((request("future", 0), request("now", 64)))
    assert [stage_start(row, "ACTIVATE") for row in rows] == [100, 0]
    assert [stage_start(row) for row in rows] == [100, 0]
    assert [row.completion_ns for row in rows] == [101, 1]


def test_write_history_retains_latest_data_end_when_actual_time_runs_backwards():
    core = DramCore(config(burst_interval_ns=0, write_to_read_same_group_ns=7),
                    ResourceTimeline(lane_available={"dram:command:0": [0., 0., 0.],
                                                      "dram:data:0": [0., 0., 0.]}))
    core.timeline.reserve("dram:bank:0:0:0:0:0:0", 0, 100)
    rows = core.submit_many((request("future-write", 0, "write"),
                             request("earlier-write", SAME_GROUP, "write"), request("read", SAME_GROUP)))
    assert [row.completion_ns for row in rows[:2]] == [101, 1]
    assert stage_start(rows[2]) == 108
    assert core._bank_group_state[(0, 0, 0, 0)]["write_end"][0] == 101


def activation_calendar_run(bank_ready, window=25):
    count = len(bank_ready)
    core = DramCore(config(bank_groups_per_rank=count, banks_per_group=1,
                           burst_interval_ns=0, four_activate_window_ns=window),
                    ResourceTimeline(lane_available={"dram:command:0": [0.] * count,
                                                      "dram:data:0": [0.] * count}))
    for index, ready in enumerate(bank_ready):
        bank_id = core._bank_id(map_dram_address(core.config, index * 64))
        core.timeline.reserve(bank_id, 0, ready)
    rows = core.submit_many(tuple(request(str(i), i * 64) for i in range(count)))
    return core, rows


@pytest.mark.parametrize("fifth_ready,expected", [(0, 0), (90, 125)])
def test_future_acts_and_actual_bank_calendar_are_considered_before_window(fifth_ready, expected):
    core, rows = activation_calendar_run([100, 100, 100, 100, fifth_ready])
    assert [stage_start(row, "ACTIVATE") for row in rows] == [100] * 4 + [expected]
    assert rows[-1].completion_ns == expected + 1
    assert core._bank_group_state[(0, 0, 0, 0)]["activate_window"] == sorted([100] * 4 + [expected])


def test_middle_insertion_moves_to_earliest_legal_activation_boundary():
    _, rows = activation_calendar_run([0, 10, 20, 30, 15], window=40)
    assert [stage_start(row, "ACTIVATE") for row in rows] == [0, 10, 20, 30, 40]


def test_future_activation_does_not_expire_earlier_window_history():
    _, rows = activation_calendar_run([0, 0, 0, 100, 0, 0])
    assert [stage_start(row, "ACTIVATE") for row in rows] == [0, 0, 0, 100, 0, 25]


def test_group_histories_do_not_forget_future_starts_after_earlier_other_group():
    core = DramCore(config(burst_interval_ns=0, column_same_group_ns=10, activate_same_group_ns=10),
                    ResourceTimeline(lane_available={"dram:command:0": [0., 0., 0.],
                                                      "dram:data:0": [0., 0., 0.]}))
    core.timeline.reserve("dram:bank:0:0:0:0:0:0", 0, 100)
    rows = core.submit_many((request("A-future", 0), request("B-earlier", OTHER_GROUP),
                             request("A-again", SAME_GROUP)))
    assert [stage_start(row, "ACTIVATE") for row in rows] == [100, 0, 110]
    assert [stage_start(row) for row in rows] == [100, 0, 110]


def test_window_deadline_rounding_does_not_loop_at_large_timestamps():
    core = DramCore(config(four_activate_window_ns=1))
    state = {"activate": {}, "activate_window": [1e20] * 4}
    assert core._bank_group_activate_ready(state, 0, 1e20) == 1e20


def test_enabled_summary_and_batch_do_not_use_unadapted_numeric_or_fold_paths():
    geometry = config(column_same_group_ns=7, column_different_group_ns=2,
                      activate_same_group_ns=9, activate_different_group_ns=3,
                      four_activate_window_ns=20, write_to_read_same_group_ns=6,
                      write_to_read_different_group_ns=4)
    requests = (request("read", 0, byte_count=4096),
                request("write", 4096, "write", byte_count=4096),
                request("read-again", 0, byte_count=4096))
    detailed = DramCore(geometry)
    expected = detailed.submit_many(requests)
    summary = DramCore(geometry, capture_details=False)
    with patch("heterollm_sim._compiled_dram.execute_compiled", side_effect=AssertionError("numeric")), \
         patch("heterollm_sim._compiled_dram.execute_compiled_many", side_effect=AssertionError("batch")), \
         patch.object(summary, "_execute_accelerated_accepted", side_effect=AssertionError("fold")):
        actual = summary.submit_many(requests)
    assert [r.completion_ns for r in actual] == [r.completion_ns for r in expected]
    assert summary._banks == detailed._banks
    assert summary._bank_group_state == detailed._bank_group_state
    assert summary.timeline.busy_ns == detailed.timeline.busy_ns
    assert summary.timeline.bytes_moved == detailed.timeline.bytes_moved
    assert all(r.counters["bank_group_timing_fallback"] for r in actual)
    assert all(not r.counters["bank_group_timing_fallback"] for r in expected)


def test_all_zero_defaults_retain_legacy_execution_and_no_group_history():
    core = DramCore(config(write_latency_ns=100))
    first = core.submit(request("write", 0, "write"))
    second = core.submit(request("read", 0))
    assert [first.completion_ns, second.completion_ns] == [101, 102]
    assert stage_start(second) == 1
    assert core._bank_group_state == {}
    assert "bank_group_timing_enabled" not in first.counters
    assert DramCore(config(), capture_details=False)._can_accelerate()


def test_new_large_timing_is_included_in_atomic_overflow_guard():
    core = DramCore(config(column_same_group_ns=1e308))
    core.submit(request("first", 0, arrival=1e308))
    before = copy.deepcopy(core.__dict__)
    # The second bank's ACT is scheduled before its overflowing CAS gate.
    # Shadow prevalidation must keep both bank and group state unchanged.
    with pytest.raises(ValueError):
        core.submit(request("overflow", SAME_GROUP))
    assert core.__dict__ == before
