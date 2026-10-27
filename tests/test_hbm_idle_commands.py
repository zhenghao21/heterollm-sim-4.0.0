"""Idle maintenance keeps every original command and floating addition."""
import math
import types

import numpy as np
import pytest
from numba import njit

from heterollm_sim import _hbm_numeric as hbm
from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_compiled_control import configuration


@njit(nogil=True, fastmath=False)
def original_refresh_until(lane, horizon, arrival, g, t, rows, bf, addresses, pc,
                           pc_seen, channel, channel_int, resources, touched, counts):
    if not g[8]:
        return False
    ch = lane // 2
    if channel[ch, 1] < 0:
        channel[ch, 1] = max(t[20], math.floor(arrival / t[20]) * t[20])
    refreshed = False
    while channel[ch, 1] <= horizon:
        if hbm._idle_refresh(lane, arrival, g, t, rows, bf, addresses, pc, pc_seen,
                             channel, resources, touched, counts):
            refreshed = True
            continue
        due = channel[ch, 1]
        for pseudo in range(2):
            hbm._refresh_pc(ch * 2 + pseudo, due, g, t, rows, bf, addresses, pc,
                            pc_seen, channel, channel_int, resources, touched, counts)
        channel[ch, 1] = hbm._after(channel[ch, 1], t[20], t[0] / 2)
        refreshed = True
    return refreshed


def clone_dispatcher(dispatcher, namespace):
    source = dispatcher.py_func
    function = types.FunctionType(source.__code__, namespace, source.__name__, source.__defaults__)
    function.__module__ = __name__
    return njit(nogil=True, fastmath=False)(function)


@pytest.fixture(scope="module")
def original_range():
    namespace = dict(vars(hbm), _refresh_until=original_refresh_until)
    namespace["_precharge_ahead"] = clone_dispatcher(hbm._precharge_ahead, namespace)
    return clone_dispatcher(hbm._run_hbm_channel_range, namespace)


def config(shared=False, **changes):
    values = dict(command_clock_ns=.64, data_rate_mt_s=6250., read_latency_ns=5.12,
                  write_latency_ns=2.56, refresh_policy="all_bank", refresh_interval_ns=3900.,
                  refresh_cycle_ns=530.)
    if shared:
        values.update(generation="HBM2", data_rate_mt_s=3125., activate_command_cycles=2., precharge_command_cycles=1.,
                      data_width_bits=64, activated_row_msb=None)
    values.update(changes)
    return configuration(**values)


def packed_state(current):
    packed = _pack_hbm_numeric(DramCore(current, capture_details=False), AccessRequest("seed", "read", 0, 32))
    assert packed is not None and len(packed.arrays) == 14
    return packed


def bits(arrays):
    return tuple((array.dtype.str, array.shape, array.tobytes()) for array in arrays)


def execute_range(function, packed, arrays, arrival, read, first=6, stop=1031):
    scratch = hbm._new_channel_scratch(packed.g, arrays)
    return function(first, stop, arrival, read, packed.g, packed.t, *arrays, scratch, 0, int(packed.g[0]) // 2)


@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("negative_busy", (False, True))
def test_late_off_grid_complete_numeric_matches_all_fourteen_arrays(shared, negative_busy, original_range):
    packed = packed_state(config(shared))
    # The unchanged controller builds the common warm state, including open
    # rows. The new helper first waits for ordinary refresh to close them.
    initial = tuple(array.copy() for array in packed.arrays)
    execute_range(original_range, packed, initial, .125, True, 0, 128)
    if negative_busy:
        initial[8][:2 * (int(packed.g[0]) // 2), 1] = -float(2 ** 60)
    outputs, states = [], []
    for function in (original_range, hbm._run_hbm_channel_range):
        arrays = tuple(array.copy() for array in initial)
        output = execute_range(function, packed, arrays, 2.e6 + .001, not shared)
        outputs.append((output[:4], output[4].tobytes()))
        states.append(bits(arrays))
    assert outputs[0] == outputs[1]
    assert states[0] == states[1]


def refresh(function, packed, arrays, horizon, arrival):
    rows, bf, addresses, pc, _, pc_seen, channel, channel_int, resources, _, _, touched, _, _ = arrays
    counts = np.zeros(7, dtype=np.float64)
    error = None
    try:
        result = function(0, horizon, arrival, packed.g, packed.t, rows, bf, addresses,
                          pc, pc_seen, channel, channel_int, resources, touched, counts)
    except OverflowError as exc:
        result, error = None, (type(exc), str(exc))
    return result, error, counts.tobytes(), bits(arrays)


def test_shared_bus_overlap_flushes_completed_epoch_without_replay():
    packed = packed_state(config(True, refresh_interval_ns=1.6, refresh_cycle_ns=.32))
    initial = tuple(array.copy() for array in packed.arrays)
    initial[2][:64] = np.arange(64) * 32
    initial[6][0, 1] = .001
    results = [refresh(function, packed, tuple(array.copy() for array in initial), .001, 1000.001)
               for function in (original_refresh_until, hbm._refresh_until)]
    assert results[0] == results[1]
    assert np.frombuffer(results[1][2], dtype=np.float64)[4] == 2.


def test_clock_magnitude_decline_preserves_overflow_mutation_prefix():
    packed = packed_state(config())
    initial = tuple(array.copy() for array in packed.arrays)
    due = packed.t[0] / 2 * 2. ** 52
    initial[6][0, 1] = due
    before = bits(initial)
    rows, bf, addresses, pc, _, pc_seen, channel, _, resources, _, _, touched, _, _ = initial
    counts = np.zeros(7, dtype=np.float64)
    assert not hbm._idle_refresh_commands(0, due, packed.g, packed.t, rows, bf, addresses,
                                          pc, pc_seen, channel, resources, touched, counts)
    assert bits(initial) == before and not counts.any()
    results = [refresh(function, packed, tuple(array.copy() for array in initial), due, due)
               for function in (original_refresh_until, hbm._refresh_until)]
    assert results[0] == results[1]
    assert results[1][1][0] is OverflowError
    assert results[1][3][5][2] != before[5][2]  # first PC was marked seen before overflow
