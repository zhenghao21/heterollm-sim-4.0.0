"""Compiled float64 implementation of the summary DRAM state transitions.

The row-hit closed form is identical to DramCore's Python implementation.
Compilation removes Python objects from its boundary-burst loops; no
fast-math, parallel reductions, sampling or reduced precision is used.
"""
from __future__ import annotations

import math

from numba import njit
import numpy as np


@njit(cache=True, nogil=True, fastmath=False)
def run_numeric_many(firsts, stops, arrivals, reads, max_outstanding, acceptance,
                     inflight, inflight_count, geometry, duration_pairs, shared,
                     bankrows, bankready, banktouched, calendar, busy, moved,
                     directions, last_start, last_end, touched):
    """Execute ordered requests with the same admission gate as submit()."""
    count, resources = len(firsts), len(calendar)
    ends = np.empty(count, dtype=np.float64)
    waits = np.empty(count, dtype=np.float64)
    counts = np.empty((count, 4), dtype=np.int64)
    busy_delta = np.empty((count, resources), dtype=np.float64)
    byte_delta = np.empty((count, resources), dtype=np.int64)
    last_starts = np.empty((count, resources), dtype=np.float64)
    last_ends = np.empty((count, resources), dtype=np.float64)
    touches = np.empty((count, resources), dtype=np.bool_)
    before_busy, before_moved = np.empty(resources), np.empty(resources, dtype=np.int64)
    for index in range(count):
        effective = max(arrivals[index], acceptance)
        remaining = 0
        for pos in range(inflight_count):
            if inflight[pos] > effective:
                inflight[remaining] = inflight[pos]
                remaining += 1
        inflight_count = remaining
        if inflight_count >= max_outstanding:
            earliest = inflight[0]
            for pos in range(1, inflight_count):
                earliest = min(earliest, inflight[pos])
            effective = max(effective, earliest)
            remaining = 0
            for pos in range(inflight_count):
                if inflight[pos] > effective:
                    inflight[remaining] = inflight[pos]
                    remaining += 1
            inflight_count = remaining
        before_busy[:] = busy
        before_moved[:] = moved
        touched[:] = False
        first, stop, read = firsts[index], stops[index], reads[index]
        durations = duration_pairs[0 if read else 1]
        if stop - first >= 64:
            completion, hits, misses, conflicts, skipped = run_numeric(
                first, stop, effective, read, geometry, durations, shared,
                bankrows, bankready, banktouched, calendar, busy, moved,
                directions, last_start, last_end, touched)
        else:
            # Small requests normally use the Python burst loop. Preserve its
            # individual transitions and addition order, without row folding.
            cursor, completion = first, effective
            hits = misses = conflicts = skipped = 0
            row_span = geometry[0] * geometry[1] * geometry[3]
            while cursor < stop:
                linear_row = cursor // row_span
                unit, row = linear_row // geometry[4], linear_row % geometry[4]
                row_stop = min(stop, (linear_row + 1) * row_span)
                cursor, completion, h, m, c = _burst_range(
                    cursor, row_stop, row, unit, effective, read, geometry, durations, shared,
                    bankrows, bankready, banktouched, calendar, busy, moved,
                    directions, last_start, last_end, touched, completion)
                hits += h
                misses += m
                conflicts += c
        ends[index], waits[index] = completion, effective - arrivals[index]
        counts[index, 0], counts[index, 1], counts[index, 2], counts[index, 3] = hits, misses, conflicts, skipped
        busy_delta[index] = busy - before_busy
        byte_delta[index] = moved - before_moved
        last_starts[index], last_ends[index], touches[index] = last_start, last_end, touched
        acceptance = effective
        inflight[inflight_count] = completion
        inflight_count += 1
    return (ends, waits, counts, busy_delta, byte_delta, last_starts, last_ends,
            touches, acceptance, inflight_count)


@njit(cache=True, nogil=True, fastmath=False)
def _repeat_add(value, increment, count):
    while count:
        value += increment
        count -= 1
        if not count or not math.isfinite(value):
            break
        step = (value + increment) - value
        if not math.isfinite(step):
            return value + increment
        if step == 0.0:
            break
        exponent = math.frexp(value)[1]
        upper = 1.7976931348623157e308 if exponent == 1024 else math.ldexp(1.0, exponent)
        steps = min(count, max(0, int((upper - value) / step) - 1))
        if steps:
            value += steps * step
            count -= steps
    return value


@njit(cache=True, nogil=True, fastmath=False)
def _reserve(rid, earliest, duration, byte_count, direction, rtw, wtr,
             calendar, busy, moved, directions, last_start, last_end, touched):
    previous_end = calendar[rid]
    if direction and directions[rid] and directions[rid] != direction:
        previous_end += rtw if directions[rid] == 1 else wtr
    start = max(earliest, previous_end)
    end = start + duration
    calendar[rid] = end
    busy[rid] += duration
    last_start[rid], last_end[rid] = start, end
    touched[rid] = True
    if direction:
        directions[rid] = direction
        moved[rid] += byte_count
    return start, end


@njit(cache=True, nogil=True, fastmath=False)
def _burst_range(cursor, target, row, unit, arrival, read, geometry, durations,
                 shared, bankrows, bankready, banktouched, calendar, busy,
                 moved, directions, last_start, last_end, touched, completion):
    lanes, banks, inner = geometry[0], geometry[1], geometry[2]
    period = lanes * banks * inner
    total_banks = len(bankrows)
    commands = 1 if shared else lanes
    burst = int(durations[0])
    q, p, latency, recovery = durations[1], durations[2], durations[3], durations[4]
    open_ns, close_ns, rtw, wtr = durations[5], durations[6], durations[7], durations[8]
    direction = 1 if read else 2
    hits = misses = conflicts = 0
    while cursor < target:
        slot = (cursor % period) // inner
        lane, bank = slot % lanes, slot // lanes
        bank_id = (unit * lanes + lane) * banks + bank
        banktouched[bank_id] = True
        dependency = arrival
        if bankrows[bank_id] == row:
            hits += 1
        else:
            if bankrows[bank_id] >= 0:
                conflicts += 1
                _, dependency = _reserve(
                    bank_id, max(dependency, bankready[bank_id]), close_ns, 0, 0, rtw, wtr,
                    calendar, busy, moved, directions, last_start, last_end, touched)
            else:
                misses += 1
            _, dependency = _reserve(
                bank_id, max(dependency, bankready[bank_id]), open_ns, 0, 0, rtw, wtr,
                calendar, busy, moved, directions, last_start, last_end, touched)
            bankrows[bank_id], bankready[bank_id] = row, dependency
        command_id = total_banks + (0 if shared else lane)
        data_id = total_banks + commands + lane
        command_start, _ = _reserve(
            command_id, dependency, q, 0, 0, rtw, wtr,
            calendar, busy, moved, directions, last_start, last_end, touched)
        _, end = _reserve(
            data_id, command_start + latency, p, burst, direction, rtw, wtr,
            calendar, busy, moved, directions, last_start, last_end, touched)
        bankready[bank_id] = max(bankready[bank_id], end + recovery)
        completion = max(completion, end + (0.0 if read else recovery))
        cursor += 1
    return cursor, completion, hits, misses, conflicts


@njit(cache=True, nogil=True, fastmath=False)
def run_numeric(first, stop, arrival, read, geometry, durations, shared,
                bankrows, bankready, banktouched, calendar, busy, moved,
                directions, last_start, last_end, touched):
    lanes, banks, inner, bursts_per_row, rows_per_bank, dies_per_stack = geometry
    period = lanes * banks * inner
    row_span = lanes * banks * bursts_per_row
    total_banks = len(bankrows)
    command_count = 1 if shared else lanes
    burst = int(durations[0])
    q, p, latency = durations[1], durations[2], durations[3]
    command_before = np.empty(lanes, dtype=np.float64)
    cursor, completion = first, arrival
    hits = misses = conflicts = skipped = 0
    while cursor < stop:
        linear_row = cursor // row_span
        unit, row = linear_row // rows_per_bank, linear_row % rows_per_bank
        row_stop = min(stop, (linear_row + 1) * row_span)
        aligned = cursor % period == 0
        target = min(row_stop, (cursor // period + 1) * period)
        cursor, completion, h, m, c = _burst_range(
            cursor, target, row, unit, arrival, read, geometry, durations, shared,
            bankrows, bankready, banktouched, calendar, busy, moved, directions,
            last_start, last_end, touched, completion)
        hits += h; misses += m; conflicts += c
        if not aligned and cursor + period <= row_stop:
            cursor, completion, h, m, c = _burst_range(
                cursor, cursor + period, row, unit, arrival, read, geometry, durations, shared,
                bankrows, bankready, banktouched, calendar, busy, moved, directions,
                last_start, last_end, touched, completion)
            hits += h; misses += m; conflicts += c
        cycles = (row_stop - cursor) // period
        count = max(0, cycles - 1)
        if count:
            per_lane = banks * inner
            transfers = count * per_lane
            commands = count * (period if shared else per_lane)
            for lane in range(lanes):
                command_before[lane] = calendar[total_banks + (0 if shared else lane)]
            for lane in range(lanes):
                data_id = total_banks + command_count + lane
                end = _repeat_add(calendar[data_id], p, transfers)
                if shared:
                    for ci in range(2):
                        cycle = 0 if ci == 0 else count - 1
                        for bi in range(2):
                            bank = 0 if bi == 0 else banks - 1
                            for ii in range(2):
                                inner_index = 0 if ii == 0 else inner - 1
                                command_index = cycle * period + bank * lanes * inner + lane * inner + inner_index
                                transfer_index = cycle * per_lane + bank * inner + inner_index
                                start = _repeat_add(command_before[lane], q, command_index)
                                end = max(end, _repeat_add(start + latency + p, p, transfers - transfer_index - 1))
                else:
                    end = max(end, _repeat_add(command_before[lane] + latency + p, p, transfers - 1),
                              _repeat_add(command_before[lane], q, commands - 1) + latency + p)
                calendar[data_id] = end
                busy[data_id] = _repeat_add(busy[data_id], p, transfers)
                moved[data_id] += transfers * burst
                last_start[data_id], last_end[data_id] = end - p, end
            for command in range(command_count):
                rid = total_banks + command
                end = _repeat_add(calendar[rid], q, commands)
                calendar[rid] = end
                busy[rid] = _repeat_add(busy[rid], q, commands)
                last_start[rid], last_end[rid] = end - q, end
            skipped += count * period
            hits += count * period
            cursor += count * period
        cursor, completion, h, m, c = _burst_range(
            cursor, row_stop, row, unit, arrival, read, geometry, durations, shared,
            bankrows, bankready, banktouched, calendar, busy, moved, directions,
            last_start, last_end, touched, completion)
        hits += h; misses += m; conflicts += c
    return completion, hits, misses, conflicts, skipped
