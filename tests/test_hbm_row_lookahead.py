"""Row-command lookahead keeps one fixed row-bus order per request."""
from dataclasses import replace
import random

import pytest

from heterollm_sim._hbm_control import row_lookahead_depth
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_mapping import rotated_bank_step
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import assert_same, configuration


def address(c, *, lane=0, bank=0, row=0, inner=0):
    step = rotated_bank_step(bank, c.bank_count, c.banks_per_group)
    block = (row * (c.row_bytes // c.burst_bytes) + inner) * c.bank_count + step
    return (block * c.lane_count + lane) * c.burst_bytes


def stage_starts(result, name):
    return [stage.start_ns for stage in result.stages
            if stage.name == name and stage.resource_id == "dram:row_command:0:0"]


def test_depth_covers_one_tRP_of_back_to_back_PRE_ACT_pairs():
    c = configuration()
    # tRP 3 ns over a 2 ns PRE+ACT pair needs two later PREs, plus one.
    assert row_lookahead_depth(c) == 3
    # Two sweeps of a channel's banks would ask one bank for two rows.
    assert row_lookahead_depth(replace(c, precharge_to_activate_ns=1000.)) == 2 * c.bank_count - 1


def test_conflicting_rows_are_precharged_ahead_and_ACTs_overlap_tRP():
    c = configuration(precharge_to_activate_ns=12.)
    sweep = c.lane_count * c.bank_count * c.burst_bytes
    core = DramCore(c)
    core.submit(AccessRequest("open-row-0", "read", 0, sweep))
    result = core.submit(AccessRequest("row-1", "read", address(c, row=1), sweep))
    assert result.counters["row_conflicts"] == c.lane_count * c.bank_count
    assert result.counters["row_misses"] == 0
    names = [stage.name for stage in result.stages if stage.resource_id == "dram:row_command:0:0"]
    depth = row_lookahead_depth(c)
    # The first burst of the channel decides its own PRE and the next depth.
    assert names[:depth + 2] == ["PRECHARGE"] * (depth + 1) + ["ACTIVATE"]
    acts = stage_starts(result, "ACTIVATE")
    # Without lookahead each conflict waits a whole tRP after the previous ACT.
    assert acts[1] - acts[0] < c.precharge_to_activate_ns


def test_bank_needed_by_an_earlier_pending_burst_keeps_its_own_turn():
    # One lane block spans two rows, so a block's own bursts cross a row.
    c = configuration(interleave_bytes=2 * 1024)
    core = DramCore(c)
    core.submit(AccessRequest("open-row-0", "read", 0, 32))
    bursts = c.row_bytes // c.burst_bytes
    result = core.submit(AccessRequest("cross", "read", (bursts - 2) * c.burst_bytes, 4 * c.burst_bytes))
    assert result.counters["row_hits"] == 3
    assert result.counters["row_conflicts"] == 1
    names = [stage.name for stage in result.stages if stage.resource_id == "dram:row_command:0:0"]
    # Two hits of row 0 still pending: the PRE for row 1 waits its own burst.
    assert names == ["PRECHARGE", "ACTIVATE"]
    reads = [stage.start_ns for stage in result.stages if stage.name == "READ_COMMAND"]
    assert stage_starts(result, "PRECHARGE")[0] > reads[1]


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_compiled_lookahead_matches_reference_on_conflict_streams(seed, refresh):
    c = configuration(precharge_to_activate_ns=12., four_activate_window_ns=11.,
                      refresh_policy="all_bank" if refresh else "disabled",
                      refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    rng = random.Random(seed)
    arrival = 0.
    for index in range(12):
        arrival += rng.choice((0., 3., 40.))
        start = address(c, lane=rng.randrange(c.lane_count), bank=rng.randrange(c.bank_count),
                        row=rng.randrange(4))
        size = rng.choice((32, 96, 2048, 4096))
        operation = rng.choice(("read", "write"))
        assert_same(reference, numeric, AccessRequest(str(index), operation, start, size, arrival))


def test_moving_an_arrival_earlier_never_delays_any_request():
    """Row hits and misses do not depend on timing without REF or expiry."""
    c = configuration(precharge_to_activate_ns=12., four_activate_window_ns=11.)
    rng = random.Random(7)

    def run(requests):
        core = DramCore(c, capture_details=False)
        return [result.completion_ns for result in core.submit_many(requests)]

    for _ in range(40):
        requests, arrival = [], 0.
        for index in range(6):
            arrival += rng.choice((0., 2., 9.))
            start = address(c, lane=rng.randrange(c.lane_count), bank=rng.randrange(c.bank_count),
                            row=rng.randrange(3))
            requests.append(AccessRequest(str(index), "read", start, rng.choice((32, 512, 2048)), arrival))
        base = run(requests)
        for index in range(1, len(requests)):
            earlier = max(requests[index - 1].arrival_ns, requests[index].arrival_ns - 5.)
            moved = requests[:index] + [replace(requests[index], arrival_ns=earlier)] + requests[index + 1:]
            assert all(after <= before for before, after in zip(base, run(moved)))
