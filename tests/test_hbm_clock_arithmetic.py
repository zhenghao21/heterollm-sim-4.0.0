"""Independent clock-count expectations for HBM, including late arrivals."""
import math
from dataclasses import replace

import numpy as np
import pytest

from heterollm_sim._hbm_control import _Controller
from heterollm_sim._hbm_numeric import _command
from heterollm_sim._compiled_hbm import execute_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_presets import resolve_physical_config


def main_hbm_config():
    return resolve_physical_config("hbm3e_reference", overrides={
        "capacity_bytes": 512 * 1024 ** 3,
        "interface_bandwidth_gb_s": 1600.,
        "read_bandwidth_gb_s": 1600., "write_bandwidth_gb_s": 1600.})


@pytest.mark.parametrize("execution", ["details", "scalar", "numeric"])
def test_main_profile_three_same_address_reads_have_independent_clock_expectations(execution):
    c = main_hbm_config()
    assert (c.command_clock_ns, c.read_latency_ns, c.column_same_group_ns) == (2.56, 81.92, 10.24)
    core = DramCore(c, capture_details=execution == "details")
    # ACT at CK0, its array anchor at CK1, tRCD=14 ns -> first RD at CK7.
    # Same-group tCCD is four CK; RL=32 CK and the burst occupies two CK.
    for index, rd_tick in enumerate((7, 11, 15)):
        request = AccessRequest(str(index), "read", 0, 32)
        result = (core._submit_validated(request, execute=lambda accepted: execute_hbm_numeric(core, accepted))
                  if execution == "numeric" else core.submit(request))
        assert core.timeline.last_intervals["dram:column_command:0:0"] == (
            rd_tick * c.command_clock_ns, (rd_tick + 1) * c.command_clock_ns)
        assert core.timeline.last_intervals["dram:data:0"] == (
            (rd_tick + 32) * c.command_clock_ns, (rd_tick + 34) * c.command_clock_ns)
        assert result.completion_ns == (rd_tick + 34) * c.command_clock_ns
        assert result.counters.get("compiled_hbm_float64", False) is (execution == "numeric")


@pytest.mark.parametrize("cycles,rising,step_half_ticks", [(.5, False, 1), (1.5, True, 4)])
@pytest.mark.parametrize("clock", [2.56, .7, .625])
def test_internal_row_command_chain_keeps_integer_half_ticks(cycles, rising, step_half_ticks, clock):
    c = replace(main_hbm_config(), command_clock_ns=clock, data_rate_mt_s=4000 / clock,
                read_latency_ns=32 * clock, write_latency_ns=8 * clock)
    core = DramCore(c, capture_details=True)
    controller = _Controller(core, AccessRequest("clock-chain", "read", 0, 32))
    # Exercise command occupancy directly, independently of bank timing gates.
    geometry = np.zeros(11, dtype=np.int64)
    geometry[0] = 2  # one channel, two pseudo-channels
    timing = np.zeros(30, dtype=np.float64)
    timing[0] = c.command_clock_ns
    resources = np.zeros((4, 4), dtype=np.float64)
    touched, counts = np.zeros(4, dtype=np.bool_), np.zeros(7)
    for index in range(128):
        half_tick = index * step_half_ticks
        expected = (half_tick * (c.command_clock_ns / 2),
                    (half_tick + int(2 * cycles)) * (c.command_clock_ns / 2))
        command = controller.command((0, 0), "row", 0., cycles, "CLOCK_TEST", rising=rising)
        assert (command.start_ns, command.end_ns) == expected
        assert _command(0, 0, 0., cycles, rising, geometry, timing,
                        resources, touched, counts) == expected


@pytest.mark.parametrize("details", [True, False])
@pytest.mark.parametrize("warm", [True, False])
@pytest.mark.parametrize("late", [False, True])
def test_external_arrival_one_ulp_after_edge_never_issues_on_that_edge(details, warm, late):
    c = main_hbm_config()
    core = DramCore(c, capture_details=details)
    if warm:
        core.submit(AccessRequest("warm", "read", 0, 32))
    edge = 100 * c.command_clock_ns
    arrival = math.nextafter(edge, math.inf) if late else edge
    result = core.submit(AccessRequest("arrival", "read", 0, 32, arrival))
    issue_tick = 100 + int(late)
    rd_tick = issue_tick if warm else issue_tick + 7
    assert core.timeline.last_intervals["dram:column_command:0:0"][0] == rd_tick * c.command_clock_ns
    if not warm:
        assert core.timeline.last_intervals["dram:row_command:0:0"][0] == issue_tick * c.command_clock_ns
    assert result.completion_ns == (rd_tick + 34) * c.command_clock_ns


@pytest.mark.parametrize("details", [True, False])
@pytest.mark.parametrize("late", [False, True])
def test_external_dq_reservation_keeps_fixed_rl_and_strict_late_edge(details, late):
    c = main_hbm_config()
    core = DramCore(c, capture_details=details)
    core.submit(AccessRequest("warm", "read", 0, 32))
    edge = 100 * c.command_clock_ns
    until = math.nextafter(edge, math.inf) if late else edge
    # Install an actual external endpoint, without a duration subtraction
    # that could independently erase its nextafter boundary.
    core.timeline.ready_ns["dram:data:0"] = until
    core.timeline.lane_available["dram:data:0"][0] = until
    result = core.submit(AccessRequest("reserved-dq", "read", 0, 32))
    data_tick = 100 + int(late)
    assert core.timeline.last_intervals["dram:column_command:0:0"][0] == (data_tick - 32) * c.command_clock_ns
    assert core.timeline.last_intervals["dram:data:0"] == (
        data_tick * c.command_clock_ns, (data_tick + 2) * c.command_clock_ns)
    assert result.completion_ns == (data_tick + 2) * c.command_clock_ns


def test_compiled_clock_index_overflow_declines_without_mutating_live_state():
    c = main_hbm_config()
    core = DramCore(c, capture_details=False)
    request = AccessRequest("large-clock-index", "read", 0, 32, 2. ** 64 * c.command_clock_ns)
    assert execute_hbm_numeric(core, request) is None
    assert not core._hbm_control_state
    assert not core._banks
    assert not core.timeline.ready_ns
