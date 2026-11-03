"""Tick and row-hit cycle execution stops before every real refresh boundary."""

from dataclasses import replace

import pytest

from heterollm_sim import _hbm_numeric
from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration
from test_hbm_write_ticks import same_result


def compare(config, requests, monkeypatch, workers=1):
    monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", workers)
    core = DramCore(config, capture_details=False)
    packed = _pack_hbm_numeric(core, requests[0])
    assert _hbm_numeric._column_tick_profile(packed.g, packed.t, True)[0] == 1
    before = tuple(array.copy() for array in packed.arrays)
    for request in requests:
        packed.arrays = tuple(array.copy() for array in before)
        packed.g[12] = 0
        expected = packed.execute(request)
        saved = tuple(array.copy() for array in packed.arrays)
        packed.arrays = tuple(array.copy() for array in before)
        packed.g[12] = 1
        actual = packed.execute(request)
        same_result(expected, actual)
        for original, current in zip(saved, packed.arrays):
            assert original.dtype == current.dtype
            assert original.shape == current.shape
            assert original.tobytes() == current.tobytes()
        before = saved


@pytest.mark.parametrize("mapping", ("bank_interleaved", "row_local"))
@pytest.mark.parametrize("clock", (.625, 1.))
def test_refresh_recovery_partial_ranges_and_direction_changes_are_bitwise(mapping, clock, monkeypatch):
    config = configuration(channels=8, rows_per_bank=8192, activated_row_msb=12,
        command_clock_ns=clock, data_rate_mt_s=4000. / clock,
        read_latency_ns=8. * clock, write_latency_ns=4. * clock,
        read_to_precharge_ns=0., write_to_read_same_group_ns=0.,
        write_to_read_different_group_ns=0., write_to_read_ns=0.,
        read_to_write_command_ns=0., read_to_write_ns=0.,
        refresh_policy="all_bank", refresh_interval_ns=400., refresh_cycle_ns=40.,
        maximum_active_ns=3600., address_mapping=mapping,
        interleave_bytes=128 if mapping == "bank_interleaved" else None)
    row_span = config.lane_count * config.bank_count * config.row_bytes
    compare(config, (
        AccessRequest("seed", "read", 193, row_span + 37, .125),
        AccessRequest("write", "write", 31, row_span // 2 + 115, 111.25),
        AccessRequest("at-refresh", "read", 17, row_span + 71, 400.),
        AccessRequest("late-write", "write", 333, row_span + 17, 70000.001),
    ), monkeypatch)
