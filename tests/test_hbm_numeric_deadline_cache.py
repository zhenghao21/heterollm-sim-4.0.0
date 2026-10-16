"""Incremental active deadlines preserve the full controller's maintenance."""
import copy
import math

import pytest

from heterollm_sim._compiled_hbm import execute_hbm_numeric
from heterollm_sim._hbm_control import _Controller
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import assert_same, assert_core_state_equal, configuration


@pytest.mark.parametrize("clock", [1., .7])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("refresh", [False, True])
def test_deadline_owner_replacement_and_later_maintenance_match(clock, tied, refresh):
    config = configuration(channels=1, command_clock_ns=clock,
        data_rate_mt_s=4000. / clock, read_latency_ns=8. * clock,
        write_latency_ns=4. * clock, maximum_active_ns=120.,
        refresh_policy="all_bank" if refresh else "disabled",
        refresh_interval_ns=100. if refresh else 0., refresh_cycle_ns=20. if refresh else 0.)
    reference = DramCore(config, capture_details=False)
    bank_step = config.lane_count * config.burst_bytes
    row_step = config.lane_count * config.bank_count * config.row_bytes
    # Three open banks give the channel one earliest deadline. Replacing the
    # newest bank must retain it; replacing its owner must find the next one.
    for bank in range(3):
        _Controller(reference, AccessRequest(f"seed-{bank}", "read", bank * bank_step, 32)).execute()
    if tied:
        # Exercise a restored state with equal earliest deadlines explicitly.
        # Equal minima must remain after either owner is closed.
        banks = list(reference._hbm_control_state["banks"].values())
        banks[1]["last_activate"] = banks[0]["last_activate"]
    numeric = copy.deepcopy(reference)
    for index, (operation, address, arrival) in enumerate((
        ("write", row_step + 2 * bank_step, -0.),
        ("read", row_step, 0.),
        ("write", row_step + bank_step, 90.),
        ("read", row_step + bank_step, math.nextafter(120., math.inf)),
        ("read", 0, 300.),
        ("write", 0, 301.),
    )):
        assert_same(reference, numeric, AccessRequest(str(index), operation, address, 32, arrival))


def test_deadline_maintenance_keeps_negative_zero_arrival_and_mixed_rows():
    config = configuration()
    reference, numeric = DramCore(config, capture_details=False), DramCore(config, capture_details=False)
    row_step = config.lane_count * config.bank_count * config.row_bytes
    for index in range(5):
        assert_same(reference, numeric, AccessRequest(str(index), "read" if index % 2 else "write",
                    (index % 2) * row_step, 512, -0.))


def test_deadline_clock_overflow_declines_without_committing_state():
    config = configuration()
    core = DramCore(config, capture_details=False)
    core.submit(AccessRequest("warm", "read", 0, 32))
    # A restored active bank beyond the exact clock range must still be
    # checked when building the request's deadline cache, before any reuse.
    next(iter(core._hbm_control_state["banks"].values()))["last_activate"] = float(2 ** 51)
    before = copy.deepcopy(core)
    assert execute_hbm_numeric(core, AccessRequest("oversized-deadline", "read", 0, 32)) is None
    assert_core_state_equal(core, before)
