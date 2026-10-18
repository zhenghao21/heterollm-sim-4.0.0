"""WRITE tick execution retains recovery anchors and full controller state."""
from dataclasses import replace

import numpy as np
import pytest

from heterollm_sim import _hbm_numeric
from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_column_pack_equivalence import b200_memory


def same_result(expected, actual):
    assert replace(actual, counters={}) == replace(expected, counters={})
    assert actual.counters.keys() == expected.counters.keys()
    for key, value in expected.counters.items():
        if isinstance(value, float):
            assert np.float64(actual.counters[key]).tobytes() == np.float64(value).tobytes(), key
        else:
            assert actual.counters[key] == value, key


@pytest.mark.parametrize("changes", [{}, {"interleave_bytes": 128},
    {"address_mapping": "row_local", "interleave_bytes": None},
    {"maximum_active_ns": 200.}, {"write_bandwidth_gb_s": 4000.}])
def test_write_ticks_match_generic_controller_through_direction_and_row_changes(changes, monkeypatch):
    monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 1)
    config = replace(b200_memory(), **changes)
    requests = (
        AccessRequest("seed-read", "read", 31, 2 ** 16 + 17, .125),
        AccessRequest("write-rows", "write", 193, 2 ** 24 + 37, 100.125),
        AccessRequest("read-after-write", "read", 17, 2 ** 19 + 71, 111.125),
        AccessRequest("late-write", "write", 333, 2 ** 19 + 17, 70000.001),
    )
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False), requests[0])
    assert _hbm_numeric._column_tick_profile(packed.g, packed.t, False)[0] == 1
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
        for original, current in zip(saved, packed.arrays, strict=True):
            assert original.dtype == current.dtype
            assert original.shape == current.shape
            assert original.tobytes() == current.tobytes()
        before = saved


@pytest.mark.parametrize("field", [18, 24])
def test_unproved_read_to_write_gap_does_not_enable_ticks(field):
    packed = _pack_hbm_numeric(DramCore(b200_memory(), capture_details=False),
                               AccessRequest("write", "write", 0, 32))
    packed.t[field] = .125
    assert _hbm_numeric._column_tick_profile(packed.g, packed.t, False)[0] == 0


def test_write_profile_uses_write_latency_and_not_read_sid_gap():
    packed = _pack_hbm_numeric(DramCore(b200_memory(), capture_details=False),
                               AccessRequest("write", "write", 0, 32))
    profile = _hbm_numeric._column_tick_profile(packed.g, packed.t, False)
    assert profile[2] == 8
    assert profile[3] == profile[4] == 2
    assert profile[6] == profile[7] == 2
    assert packed.t[10] == 15.  # Analog tWR remains anchored to data end.


def test_group_spanning_sids_declines_write_ticks_and_preserves_direction_history(monkeypatch):
    monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 1)
    config = replace(b200_memory(), bank_groups_per_rank=1, banks_per_group=32)
    requests = (
        AccessRequest("read-seed", "read", 31, 2 ** 16 + 17, .125),
        AccessRequest("write-across-sids", "write", 193, 2 ** 23 + 37, 100.125),
        AccessRequest("read-after-write", "read", 17, 2 ** 19 + 71, 111.125),
        AccessRequest("following-write", "write", 333, 2 ** 19 + 17, 70000.001),
    )
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False), requests[0])
    assert packed.g[4] == 32
    assert _hbm_numeric._column_tick_profile(packed.g, packed.t, False)[0] == 0
    assert _hbm_numeric._column_tick_profile(packed.g, packed.t, True)[0] == 1
    before = tuple(array.copy() for array in packed.arrays)
    for request in requests:
        packed.arrays = tuple(array.copy() for array in before)
        packed.g[12] = 0
        expected = packed.execute(request)
        saved = tuple(array.copy() for array in packed.arrays)
        packed.arrays = tuple(array.copy() for array in before)
        packed.g[12] = 1
        same_result(expected, packed.execute(request))
        for original, current in zip(saved, packed.arrays, strict=True):
            assert original.tobytes() == current.tobytes()
        before = saved


def test_late_write_orbit_preserves_busy_sum_rounding_and_data_end_recovery(monkeypatch):
    monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 1)
    config = replace(b200_memory(), interleave_bytes=128)
    request = AccessRequest("late-write", "write", 193, 2 ** 24 + 117, 2 ** 36 * .512 - 1.)
    packed = _pack_hbm_numeric(DramCore(config, capture_details=False), request)
    packed.arrays[8][:, 1] = np.linspace(.001, 2 ** 54, len(packed.ids))
    initial = tuple(array.copy() for array in packed.arrays)
    packed.g[12] = 0
    expected = packed.execute(request)
    saved = tuple(array.copy() for array in packed.arrays)
    packed.arrays = tuple(array.copy() for array in initial)
    packed.g[12] = 1
    same_result(expected, packed.execute(request))
    for original, current in zip(saved, packed.arrays, strict=True):
        assert original.tobytes() == current.tobytes()
