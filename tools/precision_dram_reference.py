"""Validation-only DRAM reference that visits every burst without row folding.

Uses the production float64 single-burst transition (fastmath=False). The
separate Python per-burst reference checks its arithmetic and calendar state.
This module is not imported by normal simulation execution.
"""
from numba import njit

from heterollm_sim._dram_numeric import _burst_range


@njit(cache=True, nogil=True, fastmath=False)
def run_numeric_serial(first, stop, arrival, read, geometry, durations, shared,
                       bankrows, bankready, banktouched, calendar, busy, moved,
                       directions, last_start, last_end, touched):
    lanes, banks, inner, bursts_per_row, rows_per_bank, dies_per_stack = geometry
    row_span = lanes * banks * bursts_per_row
    cursor, completion = first, arrival
    hits = misses = conflicts = 0
    while cursor < stop:
        linear_row = cursor // row_span
        unit, row = linear_row // rows_per_bank, linear_row % rows_per_bank
        row_stop = min(stop, (linear_row + 1) * row_span)
        cursor, completion, h, m, c = _burst_range(
            cursor, row_stop, row, unit, arrival, read, geometry, durations,
            shared, bankrows, bankready, banktouched, calendar, busy, moved,
            directions, last_start, last_end, touched, completion)
        hits += h
        misses += m
        conflicts += c
    return completion, hits, misses, conflicts, 0
