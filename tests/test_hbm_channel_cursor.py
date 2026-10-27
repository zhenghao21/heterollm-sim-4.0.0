"""Exact integer channel cursor specialization and unchanged numeric arrays."""
from dataclasses import replace

import numpy as np
import pytest
from numba import njit

from heterollm_sim import _hbm_numeric
from heterollm_sim._compiled_hbm import _pack_hbm_numeric
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from test_hbm_read_ticks import b200_memory


def wrapped(value):
    return (value + 2**63) % 2**64 - 2**63


def oracle_index(cursor, lane_span, period):
    return wrapped((cursor // period) * 2 * lane_span + (cursor % period) % (2 * lane_span))


def oracle_cursor(index, channel, lane_span, period):
    return wrapped((index // (2 * lane_span)) * period + channel * 2 * lane_span + index % (2 * lane_span))


@pytest.mark.parametrize("span_shift,period_shift", [(0, 1), (0, 9), (5, 14), (11, 20), (61, 62)])
def test_binary_channel_cursor_matches_floor_modulo_for_signed_int64_boundaries(span_shift, period_shift):
    span, period = 1 << span_shift, 1 << period_shift
    values = {-2**63, -2**63 + 1, -period-1, -period, -period+1, -2*span-1, -2*span,
              -1, 0, 1, 2*span-1, 2*span, 2*span+1, period-1, period, period+1, 2**63-1}
    for index in sorted(value for value in values if -2**63 <= value < 2**63):
        assert _hbm_numeric._channel_index(index, span, period, span_shift, period_shift) == oracle_index(index, span, period)
        for channel in {0, period // (2 * span) - 1}:
            assert _hbm_numeric._channel_cursor(index, channel, span, period, span_shift, period_shift) == oracle_cursor(index, channel, span, period)


@pytest.mark.parametrize("span,period", [(3, 18), (32, 192), (1, 6)])
def test_non_binary_channel_cursor_keeps_original_integer_path(span, period):
    for index in (-2**63, -period-1, -1, 0, 1, period-1, period, 2**63-1):
        assert _hbm_numeric._channel_index(index, span, period, -1, -1) == oracle_index(index, span, period)
        for channel in (0, period // (2 * span) - 1):
            assert _hbm_numeric._channel_cursor(index, channel, span, period, -1, -1) == oracle_cursor(index, channel, span, period)


@njit(cache=True, nogil=True, fastmath=False)
def original_index(cursor, lane_span, period, span_shift, period_shift):
    return (cursor // period) * 2 * lane_span + (cursor % period) % (2 * lane_span)


@njit(cache=True, nogil=True, fastmath=False)
def original_cursor(index, channel, lane_span, period, span_shift, period_shift):
    return (index // (2 * lane_span)) * period + channel * 2 * lane_span + index % (2 * lane_span)


@pytest.mark.parametrize("changes", [{}, {"address_mapping": "row_local", "interleave_bytes": None}, {"interleave_bytes": 96}])
@pytest.mark.parametrize("read", [False, True])
def test_complete_numeric_partial_stripes_match_original_cursor_formulas(changes, read):
    # Two dispatcher variants are built from the same outer native algorithm;
    # only its cursor callees differ. No production writer is monkeypatched.
    namespace = dict(vars(_hbm_numeric))
    namespace.update(_channel_index=original_index, _channel_cursor=original_cursor)
    import types
    ahead = _hbm_numeric._precharge_ahead.py_func
    old_ahead = types.FunctionType(ahead.__code__, namespace, ahead.__name__, ahead.__defaults__)
    old_ahead.__module__ = __name__
    namespace["_precharge_ahead"] = njit(nogil=True, fastmath=False)(old_ahead)
    source = _hbm_numeric._run_hbm_channel_range.py_func
    old_function = types.FunctionType(source.__code__, namespace, source.__name__, source.__defaults__)
    old_function.__module__ = __name__
    old_range = njit(nogil=True, fastmath=False)(old_function)
    request = AccessRequest("partial-channel-stripes", "read" if read else "write", 193, 2**20+117, 1e9+.001)
    packed = _pack_hbm_numeric(DramCore(replace(b200_memory(), **changes), capture_details=False), request)
    assert packed is not None
    before = tuple(array.copy() for array in packed.arrays)
    first, stop = request.address // packed.core.config.burst_bytes, (request.address+request.byte_count-1)//packed.core.config.burst_bytes+1
    outputs, states = [], []
    for function in (old_range, _hbm_numeric._run_hbm_channel_range):
        arrays = tuple(array.copy() for array in before)
        scratch = _hbm_numeric._new_channel_scratch(packed.g, arrays)
        output = function(first, stop, request.arrival_ns, read, packed.g, packed.t, *arrays, scratch, 0, int(packed.g[0])//2)
        outputs.append(output)
        states.append(tuple((array.dtype.str, array.shape, array.tobytes()) for array in arrays))
    assert outputs[0][:4] == outputs[1][:4]
    assert np.array_equal(outputs[0][4].view(np.uint8), outputs[1][4].view(np.uint8))
    assert states[0] == states[1]
