"""tFAW checks against legal-window enumeration, not path equivalence."""
import random

import numpy as np
import pytest

from heterollm_sim._hbm_numeric import _faw_ready
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


def legal(calendar, width):
    ordered = sorted(calendar)
    return all(ordered[index + 4] - ordered[index] >= width
               for index in range(len(ordered) - 4))


def test_future_insertion_matches_first_legal_integer_time_with_safe_expiration():
    randomizer = random.Random(1701)
    width = 7
    config = configuration(four_activate_window_ns=width,
                           activate_same_group_ns=0, activate_different_group_ns=0)
    timing = np.zeros(31)
    timing[0], timing[30] = 1., width
    for _ in range(40):
        calendar = []
        for _ in range(12):
            candidate = (calendar[-1] if calendar else 0) + randomizer.randrange(5)
            if len(calendar) >= 4:
                candidate = max(candidate, calendar[-4] + width)
            calendar.append(float(candidate))
        assert legal(calendar, width)
        for frontier in (0., 9., 20.):
            for requested in (frontier, frontier + 1., frontier + 6., frontier + 13.):
                expected = requested
                while not legal([*calendar, expected], width):
                    expected += 1.
                windows = np.zeros((1, len(calendar) + 5))
                windows[0, :len(calendar)] = calendar
                sizes = np.array([len(calendar)], dtype=np.int64)
                state = {"activate": {}, "activate_window": calendar.copy()}
                numeric = _faw_ready(0, requested, frontier, timing, windows, sizes)
                reference = DramCore(config)._bank_group_activate_ready(
                    state, 0, requested, frontier, clock_quantum=.5)
                assert numeric == reference == expected


@pytest.mark.parametrize("details", [False, True])
@pytest.mark.parametrize("arrival,width", [(45., 100.), (500., 1000.)])
def test_refresh_and_idle_folding_do_not_erase_unexpired_activate_limits(details, arrival, width):
    c = configuration(channels=1, four_activate_window_ns=width,
                      activate_same_group_ns=0., activate_different_group_ns=0.,
                      refresh_policy="all_bank", refresh_interval_ns=40., refresh_cycle_ns=2.)
    core = DramCore(c, capture_details=details)
    for bank in range(4):
        core.submit(AccessRequest(str(bank), "read", bank * c.lane_count * 32, 32))
    assert core._hbm_control_state["pcs"][(0, 0, 0)]["activate_window"] == [1., 3., 5., 7.]
    result = core.submit(AccessRequest("after-refresh", "read", 4 * c.lane_count * 32, 32, arrival))
    assert result.counters["refresh_commands"] > 0
    if not details:
        assert result.counters["compiled_hbm_float64"]
    last_anchor = max(bank["last_activate"] for bank in core._hbm_control_state["banks"].values())
    # REFab and its short recovery do not grant another ACT inside the
    # already occupied four-ACT window. This expectation is path-independent.
    assert last_anchor >= 1. + width
    assert legal([1., 3., 5., 7., last_anchor], width)
