"""Float64 execution of the JEDEC HBM normal-operation controller.

This is a compiled representation of _hbm_control, not the analytical DRAM
accelerator. Physical CA ports, PC data pipes, SID histories and refresh
remain distinct. No fast-math or parallel reductions are used.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Condition
import numpy as np
from numba import njit
from ._hbm_clock import align_clock, clock_after

_CHANNEL_WORKERS = min(8, os.cpu_count() or 1)
_CHANNEL_EXECUTOR = ThreadPoolExecutor(max_workers=_CHANNEL_WORKERS,
                                       thread_name_prefix="hbm-channel")

# Timing vector indices, shared with the packing module.
TIMING_FIELDS = (
    "command_clock_ns", "activate_command_cycles", "precharge_command_cycles",
    "column_command_cycles", "activate_to_read_ns", "activate_to_write_ns",
    "precharge_to_activate_ns", "minimum_active_ns", "activate_cycle_ns",
    "read_to_precharge_ns", "write_to_precharge_ns", "column_same_group_ns",
    "column_different_group_ns", "column_different_sid_ns", "activate_same_group_ns",
    "activate_different_group_ns", "write_to_read_same_group_ns",
    "write_to_read_different_group_ns", "read_to_write_command_ns",
    "precharge_to_precharge_ns", "refresh_interval_ns", "refresh_cycle_ns",
    "read_latency_ns", "write_latency_ns", "read_to_write_ns", "write_to_read_ns",
)


_align_clock = njit(cache=True, nogil=True, fastmath=False)(align_clock)
_clock_after = njit(cache=True, nogil=True, fastmath=False)(clock_after)


@njit(cache=True, nogil=True, fastmath=False)
def _align(value, quantum):
    # Numba converts floor/round to int64. Decline before that conversion
    # leaves the range where adjacent integer ticks remain distinguishable.
    if abs(value / quantum) >= 2. ** 52:
        raise OverflowError("HBM clock index exceeds compiled exact-integer range")
    return _align_clock(value, quantum)


@njit(cache=True, nogil=True, fastmath=False)
def _after(value, delay, quantum):
    if ((math.isfinite(value) and abs(value / quantum) >= 2. ** 52)
            or (math.isfinite(delay) and abs(delay / quantum) >= 2. ** 52)):
        raise OverflowError("HBM clock index exceeds compiled exact-integer range")
    return _clock_after(value, delay, quantum)


@njit(cache=True, nogil=True, fastmath=False)
def _command(channel, kind, earliest, cycles, rising, g, t, resources, touched, counts):
    channels = g[0] // 2
    rid = channel + kind * channels
    quantum = t[0] if rising or kind == 1 or g[7] else t[0] / 2
    start = _align(max(earliest, resources[rid, 0]), quantum)
    duration = cycles * t[0]
    end = _after(start, duration, t[0] / 2)
    resources[rid, 0] = end
    resources[rid, 1] += duration
    resources[rid, 2], resources[rid, 3] = start, end
    touched[rid] = True
    # The detail controller counts Transfer.duration_ns, while the calendar
    # accumulates the requested duration; retain both addition orders.
    counts[5 + kind] += end - start
    return start, end


@njit(cache=True, nogil=True, fastmath=False)
def _pre_ready(lane, earliest, pc, t):
    if pc[lane, 0] >= 0:
        earliest = max(earliest, _after(pc[lane, 0], t[19], t[0] / 2))
    return earliest


@njit(cache=True, nogil=True, fastmath=False)
def _faw_ready(lane, earliest, frontier, t, windows, window_counts):
    """Check only consecutive five-ACT windows containing the insertion."""
    width = t[30]
    if width <= 0:
        return earliest
    count = window_counts[lane]
    expired = 0
    while expired < count and _after(windows[lane, expired], width, t[0] / 2) <= frontier:
        expired += 1
    if expired:
        for index in range(count - expired):
            windows[lane, index] = windows[lane, index + expired]
        count -= expired
        window_counts[lane] = count
    while True:
        position = 0
        while position < count and windows[lane, position] <= earliest:
            position += 1
        next_start = earliest
        for offset in range(max(0, position - 4), min(position, count - 4) + 1):
            first = earliest if offset == position else windows[lane, offset]
            last_index = offset + 4
            last = earliest if last_index == position else windows[lane, last_index - 1]
            if last < _after(first, width, t[0] / 2):
                next_start = max(next_start, _after(windows[lane, offset], width, t[0] / 2))
        if next_start == earliest:
            return earliest
        earliest = next_start


@njit(cache=True, nogil=True, fastmath=False)
def _record_activate(lane, anchor, windows, window_counts):
    count = window_counts[lane]
    if count == windows.shape[1]:
        raise OverflowError("HBM ACT calendar exceeds packed capacity")
    position = count
    while position > 0 and windows[lane, position - 1] > anchor:
        windows[lane, position] = windows[lane, position - 1]
        position -= 1
    windows[lane, position] = anchor
    window_counts[lane] = count + 1


@njit(cache=True, nogil=True, fastmath=False)
def _activate_start(lane, bank, group, earliest, arrival, g, t, bf, pc, history, resources,
                    windows, window_counts):
    ready = max(earliest, bf[bank, 1], pc[lane, 1], resources[lane // 2, 0])
    if bf[bank, 0] >= 0:
        ready = max(ready, _after(_after(bf[bank, 0], t[8], t[0] / 2), -t[0], t[0] / 2))
    anchor = _after(_align(ready, t[0]), t[0], t[0] / 2)
    for previous_group in range(g[5]):
        stamp = history[lane, previous_group, 0]
        gap = t[14] if previous_group == group else t[15]
        if stamp >= 0 and gap > 0:
            anchor = max(anchor, _after(stamp, gap, t[0] / 2))
    frontier = max(arrival, resources[lane // 2, 0])
    while True:
        anchor = _faw_ready(lane, anchor, frontier, t, windows, window_counts)
        start = _align(_after(anchor, -t[0], t[0] / 2), t[0])
        aligned = _after(start, t[0], t[0] / 2)
        if t[30] <= 0 or aligned == anchor:
            return start
        anchor = aligned


@njit(cache=True, nogil=True, fastmath=False)
def _column_start(lane, bank, group, bank_ready, earliest, read, g, t, pc, history,
                  resources, directions):
    channels = g[0] // 2
    ready = max(earliest, bank_ready, pc[lane, 1], resources[channels + lane // 2, 0])
    sid = (bank % g[1]) // 16
    op_index = 1 if read else 2
    for entry in range(history.shape[1]):
        previous_group = entry * g[10] // g[4]
        stamp = history[lane, entry, op_index]
        if stamp >= 0:
            gap = t[11] if previous_group == group else t[12]
            if read and entry * g[10] // 16 != sid:
                gap = t[13]
            ready = max(ready, _after(stamp, gap, t[0] / 2))
        if read:
            end = history[lane, entry, 3]
            if end >= 0:
                gap = t[16] if previous_group == group else t[17]
                ready = max(ready, _after(end, gap, t[0] / 2))
        else:
            previous = history[lane, entry, 1]
            if previous >= 0:
                ready = max(ready, _after(previous, t[18], t[0] / 2))
    rid = 2 * channels + lane
    data_ready = resources[rid, 0]
    direction = 1 if read else 2
    if directions[rid] and directions[rid] != direction:
        data_ready = _after(data_ready, t[25] if read else t[24], t[0] / 2)
    data_ready = max(data_ready, pc[lane, 2 if read else 3])
    latency = t[22 if read else 23]
    start = _align(max(ready, _after(data_ready, -latency, t[0] / 2)), t[0])
    if _after(start, latency, t[0] / 2) < data_ready:
        start = _after(start, t[0], t[0] / 2)
    return start


@njit(cache=True, nogil=True, fastmath=False)
def _estimate(lane, bank, group, row, arrival, activate_not_before, read, g, t, rows, bf, pc, history,
              channel, channel_int, resources, directions, windows, window_counts):
    earliest = arrival
    if rows[bank] != row:
        if rows[bank] >= 0:
            pre = _align(max(_pre_ready(lane, max(earliest, bf[bank, 2]), pc, t),
                             resources[lane // 2, 0]), t[0] if g[7] else t[0] / 2)
            if channel_int[lane // 2, 0] == bank:
                pre = max(pre, _after(channel[lane // 2, 0], 2 * t[0], t[0] / 2))
            earliest = _after(pre, max(t[6], t[2] * t[0]), t[0] / 2)
        anchor = _after(_activate_start(lane, bank, group, max(earliest, activate_not_before), arrival,
            g, t, bf, pc, history, resources, windows, window_counts), t[0], t[0] / 2)
        bank_ready = _after(anchor, t[4 if read else 5], t[0] / 2)
    else:
        anchor = bf[bank, 0]
        bank_ready = bf[bank, 3 if read else 4]
    return (_column_start(lane, bank, group, bank_ready, earliest, read, g, t, pc, history,
                          resources, directions), anchor)


@njit(cache=True, nogil=True, fastmath=False)
def _next_active_deadline(ch, g, t, rows, bf, addresses):
    deadline = math.inf
    if t[29] > 0:
        for bank in range(ch * 2 * g[1], (ch + 1) * 2 * g[1]):
            if addresses[bank] >= 0 and rows[bank] >= 0:
                deadline = min(deadline, _after(bf[bank, 0], t[29], t[0] / 2))
    return deadline


@njit(cache=True, nogil=True, fastmath=False)
def _close_expiring(ch, horizon, g, t, rows, bf, addresses, pc, channel,
                     channel_int, resources, touched, counts):
    pending_banks = np.empty(2 * g[1], dtype=np.int64)
    deadlines = np.empty(2 * g[1], dtype=np.float64)
    starts = np.empty(2 * g[1], dtype=np.float64)
    count = 0
    for bank in range(ch * 2 * g[1], (ch + 1) * 2 * g[1]):
        if addresses[bank] >= 0 and rows[bank] >= 0 and _after(bf[bank, 0], t[29], t[0] / 2) <= horizon:
            pending_banks[count], deadlines[count] = bank, _after(bf[bank, 0], t[29], t[0] / 2)
            count += 1
    order = np.argsort(deadlines[:count])
    next_start = math.inf
    next_pc = np.full(2, math.inf)
    duration = t[2] * t[0]
    step = t[0] if g[7] else t[0] / 2
    for index in range(count - 1, -1, -1):
        item = order[index]
        bank = pending_banks[item]
        lane = bank // g[1]
        latest = min(deadlines[item], _after(next_start, -duration, t[0] / 2),
                     _after(next_pc[lane % 2], -t[19], t[0] / 2))
        start = math.floor(latest / step) * step
        while _align(start, step) > latest:
            start = _after(start, -step, t[0] / 2)
        start = _align(start, step)
        earliest = _pre_ready(lane, max(bf[bank, 2], pc[lane, 1], resources[ch, 0]), pc, t)
        if channel_int[ch, 0] == bank:
            earliest = max(earliest, _after(channel[ch, 0], 2 * t[0], t[0] / 2))
        if start < earliest:
            raise ValueError("HBM tRASmax deadline cannot be met by a legal PRE command")
        starts[item] = start
        next_start, next_pc[lane % 2] = start, start
    for index in range(count):
        item = order[index]
        bank = pending_banks[item]
        lane = bank // g[1]
        earliest = _pre_ready(lane, max(starts[item], bf[bank, 2], pc[lane, 1]), pc, t)
        if channel_int[ch, 0] == bank:
            earliest = max(earliest, _after(channel[ch, 0], 2 * t[0], t[0] / 2))
        start, _ = _command(ch, 0, earliest, t[2], False, g, t, resources, touched, counts)
        counts[1] += 1
        pc[lane, 0] = start
        rows[bank] = -1
        bf[bank, 1] = _after(start, t[6], t[0] / 2)
        bf[bank, 5] = bf[bank, 1]


@njit(cache=True, nogil=True, fastmath=False)
def _refresh_pc(lane, due, g, t, rows, bf, addresses, pc, pc_seen, channel,
                channel_int, resources, touched, counts):
    pc_seen[lane] = True
    first_bank, end_bank = lane * g[1], (lane + 1) * g[1]
    earliest = max(due, pc[lane, 1])
    opened = False
    for bank in range(first_bank, end_bank):
        if addresses[bank] >= 0 and rows[bank] >= 0:
            opened = True
            earliest = max(earliest, bf[bank, 2])
    if opened:
        earliest = _pre_ready(lane, earliest, pc, t)
        if channel_int[lane // 2, 1] == lane:
            earliest = max(earliest, _after(channel[lane // 2, 0], 2 * t[0], t[0] / 2))
        pre_start = _align(max(earliest, resources[lane // 2, 0]), t[0] if g[7] else t[0] / 2)
        if t[29] > 0:
            for bank in range(first_bank, end_bank):
                if addresses[bank] >= 0 and rows[bank] >= 0 and pre_start > _after(bf[bank, 0], t[29], t[0] / 2):
                    raise ValueError("HBM tRASmax deadline cannot be met by the refresh PRE command")
        start, _ = _command(lane // 2, 0, earliest, t[2], False, g, t, resources, touched, counts)
        counts[1] += 1
        pc[lane, 0] = start
        earliest = _after(start, t[6], t[0] / 2)
        for bank in range(first_bank, end_bank):
            if addresses[bank] >= 0 and rows[bank] >= 0:
                rows[bank] = -1
                bf[bank, 1] = earliest
                bf[bank, 5] = earliest
    for bank in range(first_bank, end_bank):
        if addresses[bank] >= 0:
            earliest = max(earliest, bf[bank, 1])
    last_act = channel[lane // 2, 0]
    if last_act >= 0:
        earliest = max(earliest, _after(last_act, 2 * t[0], t[0] / 2))
    if g[7]:
        earliest = max(earliest, resources[g[0] // 2 + lane // 2, 0])
    start, _ = _command(lane // 2, 0, earliest, t[2], True, g, t, resources, touched, counts)
    if g[7]:
        _command(lane // 2, 1, start, 1., True, g, t, resources, touched, counts)
    if start > due + 8 * t[20]:
        raise ValueError("HBM periodic refresh missed the maximum eight postponed intervals")
    pc[lane, 1] = _after(start, t[21], t[0] / 2)
    for bank in range(first_bank, end_bank):
        if addresses[bank] >= 0:
            rows[bank] = -1
            bf[bank, 1] = pc[lane, 1]
            bf[bank, 5] = pc[lane, 1]
    counts[4] += 1


@njit(cache=True, nogil=True, fastmath=False)
def _idle_refresh(lane, arrival, g, t, rows, bf, addresses, pc, pc_seen,
                  channel, resources, touched, counts):
    """Fold fully idle, closed-bank epochs only on an exact arithmetic grid.

    Under the bounds below every start, end, repeated busy-time addition
    and refresh deadline is an exactly representable multiple of 0.25 ns.
    Non-dyadic clocks keep the ordinary per-command loop.
    """
    ch, interval, due = lane // 2, t[20], channel[lane // 2, 1]
    if interval < t[21] + 2 * t[0]:
        return False
    for value in (t[0], interval, due, arrival, t[21]):
        if value < 0 or value >= 2. ** 49 or value * 2 != math.floor(value * 2):
            return False
    cutoff = arrival - t[21] - 2 * t[0]
    periods = math.floor((cutoff - due) / interval) + 1
    if periods < 2:
        return False
    for bank in range(ch * 2 * g[1], (ch + 1) * 2 * g[1]):
        if addresses[bank] >= 0 and (rows[bank] >= 0 or bf[bank, 1] > due):
            return False
    row_id, column_id = ch, g[0] // 2 + ch
    if max(resources[row_id, 0], resources[column_id, 0]) > due:
        return False
    for pseudo in range(2):
        if pc[ch * 2 + pseudo, 1] > due:
            return False
    for kind in range(2 if g[7] else 1):
        rid = ch + kind * (g[0] // 2)
        if resources[rid, 1] >= 2. ** 49 or resources[rid, 1] * 4 != math.floor(resources[rid, 1] * 4):
            return False
    last_due = due + (periods - 1) * interval
    last_start = _align(last_due, t[0])
    commands = periods * 2
    for kind in range(2 if g[7] else 1):
        rid = ch + kind * (g[0] // 2)
        occupancy = (t[2] if kind == 0 else 1.) * t[0]
        start = last_start + t[0]
        end = start + occupancy
        resources[rid, 0] = end
        resources[rid, 1] += commands * occupancy
        resources[rid, 2], resources[rid, 3] = start, end
        touched[rid] = True
        counts[5 + kind] += commands * occupancy
    for pseudo in range(2):
        target = ch * 2 + pseudo
        pc_seen[target] = True
        pc[target, 1] = last_start + pseudo * t[0] + t[21]
        for bank in range(target * g[1], (target + 1) * g[1]):
            if addresses[bank] >= 0:
                bf[bank, 1] = pc[target, 1]
                bf[bank, 5] = pc[target, 1]
    channel[ch, 1] = due + periods * interval
    counts[4] += commands
    return True


@njit(cache=True, nogil=True, fastmath=False)
def _refresh_until(lane, horizon, arrival, g, t, rows, bf, addresses, pc, pc_seen,
                   channel, channel_int, resources, touched, counts):
    if not g[8]:
        return False
    ch = lane // 2
    if channel[ch, 1] < 0:
        channel[ch, 1] = max(t[20], math.floor(arrival / t[20]) * t[20])
    refreshed = False
    while channel[ch, 1] <= horizon:
        if _idle_refresh(lane, arrival, g, t, rows, bf, addresses, pc, pc_seen,
                         channel, resources, touched, counts):
            refreshed = True
            continue
        due = channel[ch, 1]
        for pseudo in range(2):
            _refresh_pc(ch * 2 + pseudo, due, g, t, rows, bf, addresses, pc, pc_seen,
                        channel, channel_int, resources, touched, counts)
        channel[ch, 1] = _after(channel[ch, 1], t[20], t[0] / 2)
        refreshed = True
    return refreshed


@njit(cache=True, nogil=True, fastmath=False)
def _run_hbm_channel_range(first, stop, arrival, read, g, t, rows, bf, addresses, pc,
                    history, pc_seen, channel, channel_int, resources, moved, directions, touched,
                    windows, window_counts, channel_first, channel_stop):
    """Run complete bursts; caller commits arrays only after successful return."""
    completion = arrival
    hits = misses = conflicts = 0
    counts = np.zeros(7, dtype=np.float64)
    channels = g[0] // 2
    active_deadlines = np.empty(channels, dtype=np.float64)
    for ch in range(channel_first, channel_stop):
        active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
    # Recovery depends only on this request's direction and timing profile.
    # Keep the empty-range path free of timing arithmetic, as before.
    recovery = 0.
    if first < stop:
        recovery = t[9] if read else _after(_after(t[23], t[26], t[0] / 2), t[10], t[0] / 2)
    # Each physical channel owns both PCs and their banks/CA/data ports.
    # Walk only this channel range's stripes, retaining its original burst
    # order even for row-local mappings and partial first/last stripes.
    lane_span = g[3] * g[1] if g[11] else g[2]
    period = g[0] * lane_span
    offset = 2 * channel_first * lane_span
    width = 2 * (channel_stop - channel_first) * lane_span
    stripe_start = (first // period) * period + offset
    while stripe_start < stop:
        for cursor in range(max(first, stripe_start), min(stop, stripe_start + width)):
            if g[11]:  # row_local: complete rows, then banks, then lanes.
                stripe = cursor // g[3]
                bank_within = stripe % g[1]
                lane = (stripe // g[1]) % g[0]
                row = stripe // (g[1] * g[0])
            else:
                block, inner = cursor // g[2], cursor % g[2]
                lane = block % g[0]
                bank_linear = block // g[0]
                bank_within = bank_linear % g[1]
                row = ((bank_linear // g[1]) * g[2] + inner) // g[3]
            bank = lane * g[1] + bank_within
            group = bank_within // g[4]
            ch = lane // 2
            if g[6] >= 0:
                row &= ~(1 << g[6])
            if addresses[bank] < 0:
                addresses[bank] = cursor * g[9]
            pc_seen[lane] = True
            activate_not_before = arrival
            for iteration in range(2 * g[1] + 16):
                command_start, anchor = _estimate(lane, bank, group, row, arrival, activate_not_before, read, g, t, rows,
                                                 bf, pc, history, channel, channel_int, resources, directions,
                                                 windows, window_counts)
                horizon = _after(command_start, recovery, t[0] / 2)
                if active_deadlines[ch] <= horizon:
                    if g[8] and channel[ch, 1] < 0:
                        channel[ch, 1] = max(t[20], math.floor(arrival / t[20]) * t[20])
                    if g[8] and channel[ch, 1] <= active_deadlines[ch]:
                        _refresh_until(lane, channel[ch, 1], arrival, g, t, rows, bf, addresses,
                                       pc, pc_seen, channel, channel_int, resources, touched, counts)
                    else:
                        _close_expiring(ch, horizon, g, t, rows, bf, addresses, pc, channel,
                                        channel_int, resources, touched, counts)
                    active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
                    continue
                if _refresh_until(lane, command_start, arrival, g, t, rows, bf, addresses,
                                  pc, pc_seen, channel, channel_int, resources, touched, counts):
                    active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
                    continue
                if t[29] > 0 and rows[bank] != row and horizon > _after(anchor, t[29], t[0] / 2):
                    rcd = t[4] if read else t[5]
                    if _after(rcd, recovery, t[0] / 2) > t[29]:
                        raise ValueError("HBM access recovery exceeds the tRASmax activation window")
                    activate_not_before = _align(max(_after(activate_not_before, t[0], t[0] / 2),
                        _after(_after(horizon, -t[29], t[0] / 2), -t[0], t[0] / 2)), t[0])
                    continue
                break
            else:
                raise ValueError("HBM timing inputs leave no column-command window between maintenance commands")
            if rows[bank] == row:
                hits += 1
                # The final estimate saw this open row and no maintenance command
                # ran afterward. Its column start therefore uses exactly the same
                # state as a second _column_start call here would use.
                start = command_start
            else:
                if rows[bank] < 0:
                    misses += 1
                else:
                    conflicts += 1
                    earliest = _pre_ready(lane, max(arrival, bf[bank, 2], pc[lane, 1]), pc, t)
                    if channel_int[lane // 2, 0] == bank:
                        earliest = max(earliest, _after(channel[lane // 2, 0], 2 * t[0], t[0] / 2))
                    start, _ = _command(lane // 2, 0, earliest, t[2], False, g, t,
                                         resources, touched, counts)
                    counts[1] += 1
                    pc[lane, 0] = start
                    rows[bank] = -1
                    bf[bank, 1] = _after(start, t[6], t[0] / 2)
                    bf[bank, 5] = bf[bank, 1]
                start = _activate_start(lane, bank, group, activate_not_before, arrival,
                    g, t, bf, pc, history, resources, windows, window_counts)
                start, _ = _command(lane // 2, 0, start, t[1], True, g, t,
                                     resources, touched, counts)
                anchor = _after(start, t[0], t[0] / 2)
                history[lane, group, 0] = anchor
                if t[30] > 0:
                    _record_activate(lane, anchor, windows, window_counts)
                channel[lane // 2, 0] = start
                channel_int[lane // 2, 0], channel_int[lane // 2, 1] = bank, lane
                bf[bank, 0] = anchor
                bf[bank, 3], bf[bank, 4] = _after(anchor, t[4], t[0] / 2), _after(anchor, t[5], t[0] / 2)
                bf[bank, 2] = _after(anchor, t[7], t[0] / 2)
                rows[bank] = row
                bf[bank, 6] = min(bf[bank, 3], bf[bank, 4])
                bf[bank, 5] = bf[bank, 2]
                counts[0] += 1
                active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
                start = _column_start(lane, bank, group, bf[bank, 3 if read else 4], arrival,
                                      read, g, t, pc, history, resources, directions)
            start, _ = _command(lane // 2, 1, start, t[3], True, g, t, resources, touched, counts)
            counts[2 if read else 3] += 1
            entry = bank_within // g[10]
            history[lane, entry, 1 if read else 2] = start
            data_start = max(_after(start, t[22 if read else 23], t[0] / 2), resources[2 * channels + lane, 0])
            data_end = _after(data_start, t[26], t[0] / 2)
            rid = 2 * channels + lane
            resources[rid, 0] = data_end
            resources[rid, 1] += t[26]
            resources[rid, 2], resources[rid, 3] = data_start, data_end
            moved[rid] += g[9]
            directions[rid] = 1 if read else 2
            touched[rid] = True
            pc[lane, 2 if read else 3] = _after(data_start, t[27 if read else 28], t[0] / 2)
            if read:
                bf[bank, 2] = max(bf[bank, 2], _after(start, t[9], t[0] / 2))
            else:
                bf[bank, 2] = max(bf[bank, 2], _after(data_end, t[10], t[0] / 2))
                history[lane, entry, 3] = data_end
            bf[bank, 5] = bf[bank, 2]
            completion = max(completion, data_end)
        if period >= stop - stripe_start:
            break
        stripe_start += period
    return completion, hits, misses, conflicts, counts


def run_hbm_numeric(first, stop, arrival, read, g, t, *arrays):
    """Execute independent channels concurrently, preserving each command."""
    channels = int(g[0]) // 2
    workers = min(_CHANNEL_WORKERS, channels)
    if stop - first < 65536 or workers < 2:
        return _run_hbm_channel_range(first, stop, arrival, read, g, t, *arrays, 0, channels)
    # Compile/cache-load on the calling thread before touching shared arrays.
    # An empty interval performs no mutation and avoids concurrent first JIT.
    _run_hbm_channel_range(first, first, arrival, read, g, t, *arrays, 0, 0)
    condition = Condition()
    cancelled = False
    active = 0

    def execute_channel(channel_first, channel_stop):
        nonlocal active
        with condition:
            if cancelled:
                return None
            active += 1
        try:
            return _run_hbm_channel_range(first, stop, arrival, read, g, t,
                                          *arrays, channel_first, channel_stop)
        finally:
            with condition:
                active -= 1
                condition.notify_all()

    jobs = []
    try:
        for index in range(workers):
            jobs.append(_CHANNEL_EXECUTOR.submit(execute_channel,
                index * channels // workers, (index + 1) * channels // workers))
        wait(jobs)
        results = [job.result() for job in jobs]
    except BaseException:
        # submit() can enqueue work before raising, without returning a future.
        # Guard the actual mutations, including such untracked queued work:
        # after cancellation no new worker enters, and rollback waits for every
        # worker already inside the numeric controller, even on interruption.
        with condition:
            cancelled = True
            while active:
                try:
                    condition.wait()
                except BaseException:
                    continue
        raise
    completion = max(result[0] for result in results)
    hits = sum(result[1] for result in results)
    misses = sum(result[2] for result in results)
    conflicts = sum(result[3] for result in results)
    counts = np.zeros(7, dtype=np.float64)
    for result in results:
        counts += result[4]
    return completion, hits, misses, conflicts, counts
