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
from ._dram_numeric import _repeat_add

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
    # For stamps on one represented clock grid, clock_after(stamp, gap) is
    # monotone for each fixed gap: either integer tick addition is used for
    # every stamp, or ordinary IEEE addition is used for every stamp. Reduce
    # equal-gap categories before doing the relatively expensive clock math.
    # The packer proves initial membership. New command stamps come from
    # _align; the conservative magnitude bound preserves their membership.
    grouped = g[12]
    grid_limit = t[0] * 2. ** 48
    same = different = different_sid = previous_read = -1.
    for entry in range(history.shape[1]):
        previous_group = entry * g[10] // g[4]
        stamp = history[lane, entry, op_index]
        if stamp >= 0:
            if grouped and stamp < grid_limit:
                if read and entry * g[10] // 16 != sid:
                    different_sid = max(different_sid, stamp)
                elif previous_group == group:
                    same = max(same, stamp)
                else:
                    different = max(different, stamp)
            else:
                gap = t[11] if previous_group == group else t[12]
                if read and entry * g[10] // 16 != sid:
                    gap = t[13]
                ready = max(ready, _after(stamp, gap, t[0] / 2))
        if read:
            end = history[lane, entry, 3]
            if end >= 0:
                gap = t[16] if previous_group == group else t[17]
                ready = max(ready, end if gap == 0. else _after(end, gap, t[0] / 2))
        else:
            previous = history[lane, entry, 1]
            if previous >= 0:
                if grouped and previous < grid_limit:
                    previous_read = max(previous_read, previous)
                else:
                    ready = max(ready, _after(previous, t[18], t[0] / 2))
    if same >= 0:
        ready = max(ready, _after(same, t[11], t[0] / 2))
    if different >= 0:
        ready = max(ready, _after(different, t[12], t[0] / 2))
    if different_sid >= 0:
        ready = max(ready, _after(different_sid, t[13], t[0] / 2))
    if previous_read >= 0:
        ready = max(ready, previous_read if t[18] == 0. else _after(previous_read, t[18], t[0] / 2))
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
        half_clock = t[0] / 2
        delay_ticks = round(t[29] / half_clock)
        if t[29] != delay_ticks * half_clock:
            # A non-grid delay always uses ordinary floating addition for
            # every anchor, including externally restored non-grid anchors.
            # IEEE addition with one fixed finite operand is monotone.
            anchor = math.inf
            for bank in range(ch * 2 * g[1], (ch + 1) * 2 * g[1]):
                if addresses[bank] >= 0 and rows[bank] >= 0:
                    anchor = min(anchor, bf[bank, 0])
            if math.isfinite(anchor):
                return _after(anchor, t[29], half_clock)
            return deadline
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
def _column_tick_profile(g, t, read):
    """Prove the integer recurrence used by same-direction row hits.

    No command is omitted. This changes the representation of grid-valued
    clocks only. Analog column gaps must lie far enough inside their tick
    interval that all IEEE rounding errors below the bounded horizon cannot
    change the first legal command edge.
    """
    result = np.zeros(8, dtype=np.int64)
    limit = 2 ** 38
    result[1] = limit
    if (g[8] or t[3] != 1.
            or (read and (t[9] != 0. or t[16] != 0. or t[17] != 0. or t[25] != 0.))
            # WRITE has no SID-specific gap. A custom group spanning several
            # SIDs would need the same-group gap across those SID histories.
            or (not read and (g[4] > 16 or t[18] != 0. or t[24] != 0.))):
        return result
    clock = t[0]
    for index, field in enumerate((22 if read else 23, 26, 27 if read else 28)):
        value = t[field]
        if abs(value / clock) >= limit:
            return result
        ticks = round(value / clock)
        if value != ticks * clock or ticks < 0:
            return result
        result[2 + index] = ticks
    for index, field in enumerate((11, 12, 13 if read else 12)):
        value = t[field]
        if value < 0 or value / clock >= limit:
            return result
        ticks = math.ceil(value / clock)
        if value != ticks * clock:
            margin = min(value - (ticks - 1) * clock, ticks * clock - value)
            # Covers the multiplication, addition, division and final edge
            # comparison, with a deliberately conservative factor of 64.
            if margin <= 64. * 2. ** -52 * (limit * clock + value):
                return result
        result[5 + index] = ticks
    result[0] = 1
    return result


@njit(cache=True, nogil=True, fastmath=False)
def _bounded_tick(value, clock, limit, ceiling):
    if abs(value / clock) >= limit:
        return -2
    edge = _align(value, clock) if ceiling else value
    tick = round(edge / clock)
    return tick if edge == tick * clock and 0 <= tick < limit else -2


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _tick_top_record(top, key, value):
    """Maintain two largest distinct keys under monotone per-key updates."""
    if top[1] == key:
        top[0] = value
    elif value >= top[0]:
        top[2], top[3] = top[0], top[1]
        top[0], top[1] = value, key
    elif top[3] == key or value >= top[2]:
        top[2], top[3] = value, key


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _tick_history_record(lane, entry, tick, stride, group_tops, sid_tops):
    sid = entry * stride // 16
    _tick_top_record(group_tops[lane, sid], entry, tick)
    _tick_top_record(sid_tops[lane], sid, group_tops[lane, sid, 0])


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _binary_shift(value):
    if value <= 0 or value & (value - 1):
        return -1
    shift = 0
    while value > 1:
        value >>= 1
        shift += 1
    return shift


@njit(cache=True, nogil=True, fastmath=False)
def _fold_column_cycles(cycles, channel_first, channel_stop, read, recovery, g, t, profile,
                      previous_columns, previous_data, previous_history,
                      tick_columns, tick_data, tick_history, tick_base, tick_banks,
                      cycle_starts, group_tops, sid_tops, deadlines,
                      bf, pc, history, resources, moved, counts):
    """Repeat an exact translation orbit of the integer max-plus controller.

    Every dynamic input of the directional recurrence must translate by one common
    integer delta within its independent physical channel. The fixed lower
    bounds must already lie behind that channel's old column frontier. This
    proves translation equivariance for all subsequent cycles, rather than
    extrapolating timing from sampled commands. A row or maintenance boundary
    ends the orbit. Serial float accumulators retain their original ordering.
    """
    channels = g[0] // 2
    deltas = np.empty(channels, dtype=np.int64)
    for ch in range(channel_first, channel_stop):
        delta = tick_columns[ch] - previous_columns[ch]
        if delta <= 0 or resources[channels + ch, 1] < 0:
            return 0, 0.
        deltas[ch] = delta
        cycles = min(cycles, max(0, (profile[1] - tick_columns[ch]
                     - profile[2] - max(profile[3], profile[4])) // delta))
        for lane in range(2 * ch, 2 * ch + 2):
            if (tick_data[lane] - previous_data[lane] != delta
                    or tick_base[lane] > previous_columns[ch]
                    or resources[2 * channels + lane, 1] < 0):
                return 0, 0.
            for entry in range(history.shape[1]):
                if (tick_history[lane, entry] < 0
                        or tick_history[lane, entry] - previous_history[lane, entry] != delta):
                    return 0, 0.
            for bank in range(lane * g[1], (lane + 1) * g[1]):
                if tick_banks[bank] < 0 or tick_banks[bank] > previous_columns[ch]:
                    return 0, 0.
        while cycles > 0:
            last = tick_columns[ch] - 1 + cycles * delta
            horizon = last * t[0] if read else _after(last * t[0], recovery, t[0] / 2)
            if (last + profile[2] + max(profile[3], profile[4]) < profile[1]
                    and horizon < deadlines[ch]):
                break
            cycles -= 1
        if cycles == 0:
            return 0, 0.
    # Counts contain actual Transfer.duration_ns = end-start, whose rounding
    # can vary with the edge. Replay only those subtractions/additions in the
    # original bank/channel/PC/burst order; no controller work is repeated.
    for repeat in range(1, cycles + 1):
        for bank_within in range(g[1]):
            for ch in range(channel_first, channel_stop):
                shift = repeat * deltas[ch]
                for lane in range(2 * ch, 2 * ch + 2):
                    bank = lane * g[1] + bank_within
                    for inner in range(g[2]):
                        tick = cycle_starts[bank, inner] + shift
                        counts[6] += (tick + 1) * t[0] - tick * t[0]
    completion = 0.
    pc_bursts = cycles * g[1] * g[2]
    for ch in range(channel_first, channel_stop):
        shift = cycles * deltas[ch]
        column = channels + ch
        last = tick_columns[ch] - 1 + shift
        resources[column, 0] = (last + 1) * t[0]
        resources[column, 1] = _repeat_add(resources[column, 1], t[0], 2 * pc_bursts)
        resources[column, 2], resources[column, 3] = last * t[0], (last + 1) * t[0]
        tick_columns[ch] += shift
        for lane in range(2 * ch, 2 * ch + 2):
            for entry in range(history.shape[1]):
                tick_history[lane, entry] += shift
                history[lane, entry, 1 if read else 2] = tick_history[lane, entry] * t[0]
                if not read:
                    history[lane, entry, 3] = (tick_history[lane, entry] + profile[2] + profile[3]) * t[0]
            for sid in range(group_tops.shape[1]):
                for index in (0, 2):
                    if group_tops[lane, sid, index] >= 0:
                        group_tops[lane, sid, index] += shift
            for index in (0, 2):
                if sid_tops[lane, index] >= 0:
                    sid_tops[lane, index] += shift
            for bank in range(lane * g[1], (lane + 1) * g[1]):
                last = cycle_starts[bank, g[2] - 1] + shift
                if read:
                    bf[bank, 2] = max(bf[bank, 2], last * t[0])
                else:
                    data_end = (last + profile[2] + profile[3]) * t[0]
                    bf[bank, 2] = max(bf[bank, 2], _after(data_end, t[10], t[0] / 2))
                bf[bank, 5] = bf[bank, 2]
                for inner in range(g[2]):
                    cycle_starts[bank, inner] += shift
            last = cycle_starts[(lane + 1) * g[1] - 1, g[2] - 1]
            data_start = (last + profile[2]) * t[0]
            data_end = (last + profile[2] + profile[3]) * t[0]
            rid = 2 * channels + lane
            resources[rid, 0] = data_end
            resources[rid, 1] = _repeat_add(resources[rid, 1], t[26], pc_bursts)
            resources[rid, 2], resources[rid, 3] = data_start, data_end
            moved[rid] += pc_bursts * g[9]
            pc[lane, 2 if read else 3] = (last + profile[2] + profile[4]) * t[0]
            tick_data[lane] += shift
            completion = max(completion, data_end)
    counts[2 if read else 3] += 2 * (channel_stop - channel_first) * pc_bursts
    return cycles, completion


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
    tick_candidate = g[12] and stop - first >= 64
    tick_profile = _column_tick_profile(g, t, read) if tick_candidate else np.zeros(8, dtype=np.int64)
    tick_mode = tick_candidate and tick_profile[0]
    tick_lanes = g[0] if tick_mode else 0
    tick_channels = channels if tick_mode else 0
    tick_history = np.empty((tick_lanes, history.shape[1]), dtype=np.int64)
    tick_banks = np.empty(rows.shape[0] if tick_mode else 0, dtype=np.int64)
    tick_base = np.empty(tick_lanes, dtype=np.int64)
    tick_data = np.empty(tick_lanes, dtype=np.int64)
    tick_columns = np.empty(tick_channels, dtype=np.int64)
    tick_valid = np.empty(tick_lanes, dtype=np.bool_)
    tick_group_tops = np.empty((tick_lanes, (g[1] + 15) // 16, 4), dtype=np.int64)
    tick_sid_tops = np.empty((tick_lanes, 4), dtype=np.int64)
    fold_mode = (tick_mode and not g[11] and g[2] <= min(g[3], 64)
                 and g[3] % g[2] == 0 and rows.shape[0] * g[2] <= 1048576)
    cycle_starts = np.empty((rows.shape[0] if fold_mode else 0, g[2] if fold_mode else 0), dtype=np.int64)
    previous_columns = np.empty(tick_channels, dtype=np.int64)
    previous_data = np.empty(tick_lanes, dtype=np.int64)
    previous_history = np.empty_like(tick_history)
    previous_valid = False
    cycle_fast = False
    if tick_mode:
        # A worker never reads another channel's scratch state. Initialize
        # only its own banks/history, just as the live state is partitioned.
        tick_banks[2 * channel_first * g[1]:2 * channel_stop * g[1]] = -2
        tick_group_tops[2 * channel_first:2 * channel_stop] = -1
        tick_sid_tops[2 * channel_first:2 * channel_stop] = -1
        for ch in range(channel_first, channel_stop):
            tick_columns[ch] = _bounded_tick(resources[channels + ch, 0], t[0], tick_profile[1], True)
            for lane in range(2 * ch, 2 * ch + 2):
                valid = True
                base = max(arrival, pc[lane, 1])
                for entry in range(history.shape[1]):
                    stamp = history[lane, entry, 1 if read else 2]
                    tick_history[lane, entry] = -1 if stamp < 0 else _bounded_tick(stamp, t[0], tick_profile[1], False)
                    if tick_history[lane, entry] == -2:
                        valid = False
                    elif tick_history[lane, entry] >= 0:
                        _tick_history_record(lane, entry, tick_history[lane, entry], g[10],
                                             tick_group_tops, tick_sid_tops)
                    # The opposite direction's history stays fixed throughout
                    # this homogeneous request. Both selected turnaround gaps
                    # are proved zero by _column_tick_profile.
                    base = max(base, history[lane, entry, 3 if read else 1])
                tick_base[lane] = _bounded_tick(base, t[0], tick_profile[1], True)
                tick_data[lane] = _bounded_tick(max(resources[2 * channels + lane, 0], pc[lane, 2 if read else 3]),
                                                 t[0], tick_profile[1], False)
                tick_valid[lane] = valid and tick_base[lane] >= 0
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
    lane_shift, bank_shift = _binary_shift(g[0]), _binary_shift(g[1])
    interleave_shift, row_shift = _binary_shift(g[2]), _binary_shift(g[3])
    group_shift, history_shift = _binary_shift(g[4]), _binary_shift(g[10])
    binary_mapping = min(lane_shift, bank_shift, interleave_shift, row_shift,
                         group_shift, history_shift) >= 0
    while stripe_start < stop:
        if fold_mode:
            cycle_span = g[0] * g[1] * g[2]
            if (stripe_start - offset) % cycle_span == 0:
                cycle_index = (stripe_start - offset) // cycle_span
                row_phase = cycle_index % (g[3] // g[2])
                if previous_valid and cycle_fast and row_phase:
                    cycles = min(g[3] // g[2] - row_phase,
                                 (stop - (stripe_start - offset)) // cycle_span)
                    if cycles > 0:
                        folded, folded_completion = _fold_column_cycles(cycles, channel_first, channel_stop,
                            read, recovery, g, t, tick_profile, previous_columns, previous_data, previous_history,
                            tick_columns, tick_data, tick_history, tick_base, tick_banks, cycle_starts,
                            tick_group_tops, tick_sid_tops, active_deadlines,
                            bf, pc, history, resources, moved, counts)
                        if folded:
                            hits += folded * 2 * (channel_stop - channel_first) * g[1] * g[2]
                            completion = max(completion, folded_completion)
                            stripe_start += folded * cycle_span
                            previous_valid = False
                            cycle_fast = False
                            if stripe_start >= stop:
                                break
                for ch in range(channel_first, channel_stop):
                    previous_columns[ch] = tick_columns[ch]
                    for lane in range(2 * ch, 2 * ch + 2):
                        previous_data[lane] = tick_data[lane]
                        for entry in range(history.shape[1]):
                            previous_history[lane, entry] = tick_history[lane, entry]
                previous_valid = True
                cycle_fast = first <= stripe_start
        for cursor in range(max(first, stripe_start), min(stop, stripe_start + width)):
            if binary_mapping:
                if g[11]:
                    stripe = cursor >> row_shift
                    bank_within = stripe & (g[1] - 1)
                    lane = (stripe >> bank_shift) & (g[0] - 1)
                    row = stripe >> (bank_shift + lane_shift)
                else:
                    block, inner = cursor >> interleave_shift, cursor & (g[2] - 1)
                    lane = block & (g[0] - 1)
                    bank_linear = block >> lane_shift
                    bank_within = bank_linear & (g[1] - 1)
                    row = (((bank_linear >> bank_shift) << interleave_shift) + inner) >> row_shift
                group = bank_within >> group_shift
                history_entry = bank_within >> history_shift
            elif g[11]:  # row_local: complete rows, then banks, then lanes.
                stripe = cursor // g[3]
                bank_within = stripe % g[1]
                lane = (stripe // g[1]) % g[0]
                row = stripe // (g[1] * g[0])
                group, history_entry = bank_within // g[4], bank_within // g[10]
            else:
                block, inner = cursor // g[2], cursor % g[2]
                lane = block % g[0]
                bank_linear = block // g[0]
                bank_within = bank_linear % g[1]
                row = ((bank_linear // g[1]) * g[2] + inner) // g[3]
                group, history_entry = bank_within // g[4], bank_within // g[10]
            bank = lane * g[1] + bank_within
            ch = lane // 2
            if g[6] >= 0:
                row &= ~(1 << g[6])
            if addresses[bank] < 0:
                addresses[bank] = cursor * g[9]
            pc_seen[lane] = True
            if (tick_mode and tick_valid[lane] and rows[bank] == row
                    and tick_data[lane] >= 0 and tick_columns[ch] >= 0):
                if tick_banks[bank] == -2:
                    tick_banks[bank] = _bounded_tick(bf[bank, 3 if read else 4], t[0], tick_profile[1], True)
                if tick_banks[bank] >= 0:
                    tick = max(tick_base[lane], tick_banks[bank], tick_columns[ch],
                               tick_data[lane] - tick_profile[2])
                    sid = bank_within // 16
                    entry = history_entry
                    previous = tick_history[lane, entry]
                    if previous >= 0:
                        tick = max(tick, previous + tick_profile[5])
                    group_top = tick_group_tops[lane, sid]
                    previous = group_top[0] if group_top[1] != entry else group_top[2]
                    if previous >= 0:
                        tick = max(tick, previous + tick_profile[6])
                    sid_top = tick_sid_tops[lane]
                    previous = sid_top[0] if sid_top[1] != sid else sid_top[2]
                    if previous >= 0:
                        tick = max(tick, previous + tick_profile[7])
                    start = tick * t[0]
                    last_tick = tick + tick_profile[2] + max(tick_profile[3], tick_profile[4])
                    horizon = start if read else _after(start, recovery, t[0] / 2)
                    if last_tick < tick_profile[1] and horizon < active_deadlines[ch]:
                        # These are exactly clock_after's integer tick sums,
                        # expressed in full rather than half clock units.
                        end = (tick + 1) * t[0]
                        column = channels + ch
                        resources[column, 0] = end
                        resources[column, 1] += t[0]
                        resources[column, 2], resources[column, 3] = start, end
                        touched[column] = True
                        counts[6] += end - start
                        counts[2 if read else 3] += 1
                        hits += 1
                        tick_columns[ch] = tick + 1
                        entry = history_entry
                        history[lane, entry, 1 if read else 2] = start
                        tick_history[lane, entry] = tick
                        _tick_history_record(lane, entry, tick, g[10], tick_group_tops, tick_sid_tops)
                        if fold_mode:
                            cycle_starts[bank, inner] = tick
                        data_start = (tick + tick_profile[2]) * t[0]
                        data_end = (tick + tick_profile[2] + tick_profile[3]) * t[0]
                        rid = 2 * channels + lane
                        resources[rid, 0] = data_end
                        resources[rid, 1] += t[26]
                        resources[rid, 2], resources[rid, 3] = data_start, data_end
                        moved[rid] += g[9]
                        directions[rid] = 1 if read else 2
                        touched[rid] = True
                        tick_data[lane] = tick + tick_profile[2] + max(tick_profile[3], tick_profile[4])
                        pc[lane, 2 if read else 3] = (tick + tick_profile[2] + tick_profile[4]) * t[0]
                        if read:
                            bf[bank, 2] = max(bf[bank, 2], start)
                        else:
                            # tWR may be analog (15 ns on the B200 reference).
                            # Keep the original data-end anchor and rounding.
                            bf[bank, 2] = max(bf[bank, 2], _after(data_end, t[10], t[0] / 2))
                            history[lane, entry, 3] = data_end
                        bf[bank, 5] = bf[bank, 2]
                        completion = max(completion, data_end)
                        continue
            cycle_fast = False
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
            entry = history_entry
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
            if tick_mode:
                tick_banks[bank] = -2
                tick_columns[ch] = _bounded_tick(resources[channels + ch, 0], t[0], tick_profile[1], True)
                tick_history[lane, entry] = _bounded_tick(start, t[0], tick_profile[1], False)
                tick_valid[lane] = tick_valid[lane] and tick_history[lane, entry] >= 0
                if tick_history[lane, entry] >= 0:
                    _tick_history_record(lane, entry, tick_history[lane, entry], g[10],
                                         tick_group_tops, tick_sid_tops)
                    if fold_mode:
                        cycle_starts[bank, inner] = tick_history[lane, entry]
                tick_data[lane] = _bounded_tick(max(data_end, pc[lane, 2 if read else 3]), t[0], tick_profile[1], False)
        if period >= stop - stripe_start:
            break
        stripe_start += period
    return completion, hits, misses, conflicts, counts


def run_hbm_numeric(first, stop, arrival, read, g, t, *arrays):
    """Execute independent channels concurrently, preserving each command."""
    channels = int(g[0]) // 2
    # Small requests often reach only a few adjacent channels. Their command
    # loop never consults another channel, so neither should its initial
    # deadline/history packing. A wrapped range keeps the original traversal.
    lane_span = int(g[3]) * int(g[1]) if g[11] else int(g[2])
    first_lane, last_lane = first // lane_span, (stop - 1) // lane_span
    channel_first, channel_stop = 0, channels
    if first < stop and first_lane // int(g[0]) == last_lane // int(g[0]):
        channel_first = (first_lane % int(g[0])) // 2
        channel_stop = (last_lane % int(g[0])) // 2 + 1
    physical_bytes = (stop - first) * int(g[9])
    workers = min(_CHANNEL_WORKERS, channel_stop - channel_first,
                  physical_bytes // (32 * 1024))
    if physical_bytes < 128 * 1024 or workers < 2:
        return _run_hbm_channel_range(first, stop, arrival, read, g, t, *arrays, channel_first, channel_stop)
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
                channel_first + index * (channel_stop - channel_first) // workers,
                channel_first + (index + 1) * (channel_stop - channel_first) // workers))
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
