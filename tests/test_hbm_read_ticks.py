"""Grid-valued READ commands retain every generic JEDEC state update."""
from dataclasses import replace

import numpy as np
import pytest

from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_column_pack_equivalence import b200_memory
from test_hbm_compiled_control import configuration


@pytest.mark.parametrize("config", [
    b200_memory(),
    configuration(channels=3, rows_per_bank=8192, activated_row_msb=12,
                  read_to_precharge_ns=0., write_to_read_same_group_ns=0.,
                  write_to_read_different_group_ns=0., write_to_read_ns=0.),
    replace(b200_memory(), maximum_active_ns=200.),
    replace(b200_memory(), interleave_bytes=128),
    replace(b200_memory(), address_mapping="row_local", interleave_bytes=None),
])
def test_read_ticks_match_generic_controller_bitwise(config):
    requests = (
        AccessRequest("read", "read", 193, 2 ** 22 + 37, 0.125),
        AccessRequest("write", "write", 31, 2 ** 16 + 115, 111.25),
        AccessRequest("repeat", "read", 17, 2 ** 22 + 71, 121.125),
        AccessRequest("late", "read", 333, 2 ** 22 + 17, 70000.001),
    )
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False), requests[0])
    before = tuple(array.copy() for array in packed.arrays)
    for request in requests:
        packed.arrays = tuple(array.copy() for array in before)
        packed.g[12] = 0
        expected = packed.execute(request)
        expected_arrays = tuple(array.copy() for array in packed.arrays)
        packed.arrays = tuple(array.copy() for array in before)
        packed.g[12] = 1
        actual = packed.execute(request)
        assert actual == expected
        for actual_array, expected_array in zip(packed.arrays, expected_arrays):
            assert np.array_equal(actual_array.view(np.uint8), expected_array.view(np.uint8))
        before = expected_arrays


def test_off_grid_restored_column_history_disables_reduction():
    config = b200_memory()
    core = DramCore(config, capture_details=False)
    core.submit(AccessRequest("seed", "read", 0, 32))
    pc = next(iter(core._hbm_control_state["pcs"].values()))
    for key in pc["column"]["read"]:
        pc["column"]["read"][key] = np.nextafter(pc["column"]["read"][key], np.inf)
    packed = _pack_hbm_numeric(core, AccessRequest("read", "read", 0, 2 ** 20))
    assert packed.g[12] == 0


@pytest.mark.parametrize("clock", [.512, .3, 1., 1.28])
def test_read_tick_gap_rounding_requires_a_provable_margin(clock):
    from heterollm_sim._hbm_numeric import _read_tick_profile

    packed = _pack_hbm_numeric(DramCore(b200_memory(), capture_details=False),
                               AccessRequest("read", "read", 0, 32))
    packed.t[0] = clock
    packed.t[11:14] = [5 * clock, 2 * clock, 4 * clock]
    packed.t[22] = 32 * clock
    packed.t[26:28] = 2 * clock
    assert _read_tick_profile(packed.g, packed.t)[0] == 1
    packed.t[11] = np.nextafter(5 * clock, np.inf)
    assert _read_tick_profile(packed.g, packed.t)[0] == 0


@pytest.mark.parametrize("interleave,arrival", [(32, 1e9 + .001), (128, 2 ** 36 * .512 - 1.)])
def test_folded_cycles_keep_serial_float_accumulators_at_late_clock_edges(interleave, arrival):
    config = replace(b200_memory(), interleave_bytes=interleave)
    request = AccessRequest("late-partial", "read", 193, 2 ** 24 + 117, arrival)
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False), request)
    packed.arrays[8][:, 1] = np.linspace(0.001, 2 ** 34 + .3, len(packed.ids))
    before = tuple(array.copy() for array in packed.arrays)
    packed.g[12] = 0
    expected = packed.execute(request)
    saved = tuple(array.copy() for array in packed.arrays)
    packed.arrays = tuple(array.copy() for array in before)
    packed.g[12] = 1
    assert packed.execute(request) == expected
    for actual, original in zip(packed.arrays, saved):
        assert np.array_equal(actual.view(np.uint8), original.view(np.uint8))
