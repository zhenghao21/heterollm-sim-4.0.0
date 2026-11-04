"""Float64 execution of the JEDEC HBM normal-operation controller.

This is a compiled representation of _hbm_control, not the analytical DRAM
accelerator. Physical CA ports, PC data pipes, SID histories and refresh
remain distinct. No fast-math or parallel reductions are used.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Condition, local
import numpy as np
from numba import njit
from ._hbm_clock import align_clock, clock_after
from ._dram_numeric import _repeat_add, rotated_bank
from .native_hbm_memo_experiment import _ACTIVE as _NUMERIC_MEMO_CONTEXT

# A manager running several simulator processes caps each process's channel
# threads so that together they do not oversubscribe the host cores.
_CHANNEL_WORKERS = max(1, min(8, int(os.environ.get("HETEROLLM_HBM_CHANNEL_THREADS") or os.cpu_count() or 1)))
_CHANNEL_EXECUTOR = ThreadPoolExecutor(max_workers=_CHANNEL_WORKERS,
                                       thread_name_prefix="hbm-channel")
_CHANNEL_SCRATCH = local()


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
    ratio = value / quantum
    if abs(ratio) >= 2. ** 52:
        raise OverflowError("HBM clock index exceeds compiled exact-integer range")
    # align_clock with its one division shared.
    tick = math.floor(ratio)
    edge = tick * quantum
    if edge < value:
        edge = (tick + 1) * quantum
    return edge


@njit(cache=True, nogil=True, fastmath=False)
def _after(value, delay, quantum):
    # clock_after with its two divisions shared.
    if math.isfinite(value) and math.isfinite(delay):
        value_ratio, delay_ratio = value / quantum, delay / quantum
        if abs(value_ratio) >= 2. ** 52 or abs(delay_ratio) >= 2. ** 52:
            raise OverflowError("HBM clock index exceeds compiled exact-integer range")
        value_tick, delay_tick = round(value_ratio), round(delay_ratio)
        if value == value_tick * quantum and delay == delay_tick * quantum:
            return (value_tick + delay_tick) * quantum
    elif ((math.isfinite(value) and abs(value / quantum) >= 2. ** 52)
            or (math.isfinite(delay) and abs(delay / quantum) >= 2. ** 52)):
        raise OverflowError("HBM clock index exceeds compiled exact-integer range")
    return value + delay


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
    # Grid stamps reduce per gap before the clock math, as in _column_start.
    grid_limit = t[0] * 2. ** 48
    other = -1.
    for previous_group in range(g[5]):
        stamp = history[lane, previous_group, 0]
        gap = t[14] if previous_group == group else t[15]
        if stamp >= 0 and gap > 0:
            if g[12] and stamp < grid_limit and previous_group != group:
                other = max(other, stamp)
            else:
                anchor = max(anchor, _after(stamp, gap, t[0] / 2))
    if other >= 0:
        anchor = max(anchor, _after(other, t[15], t[0] / 2))
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
def _precharge_start(lane, bank, earliest, g, t, bf, pc, channel, channel_int, resources):
    """_Controller.precharge_start: the in-order PRE edge of one bank."""
    ready = _pre_ready(lane, max(earliest, bf[bank, 2], pc[lane, 1]), pc, t)
    if channel_int[lane // 2, 0] == bank:
        ready = max(ready, _after(channel[lane // 2, 0], 2 * t[0], t[0] / 2))
    return _align(max(ready, resources[lane // 2, 0]), t[0] if g[7] else t[0] / 2)


@njit(cache=True, nogil=True, fastmath=False)
def _precharge(lane, bank, earliest, g, t, rows, bf, pc, channel, channel_int,
               resources, touched, counts):
    start = _precharge_start(lane, bank, earliest, g, t, bf, pc, channel, channel_int, resources)
    start, _ = _command(lane // 2, 0, start, t[2], False, g, t, resources, touched, counts)
    counts[1] += 1
    pc[lane, 0] = start
    rows[bank] = -1
    bf[bank, 1] = _after(start, t[6], t[0] / 2)
    bf[bank, 5] = bf[bank, 1]


@njit(cache=True, nogil=True, fastmath=False)
def _estimate(lane, bank, group, row, arrival, activate_not_before, read, g, t, rows, bf, pc, history,
              channel, channel_int, resources, directions, windows, window_counts):
    earliest = arrival
    activate = -1.
    if rows[bank] != row:
        if rows[bank] >= 0:
            pre = _precharge_start(lane, bank, earliest, g, t, bf, pc, channel, channel_int, resources)
            earliest = _after(pre, max(t[6], t[2] * t[0]), t[0] / 2)
        activate = _activate_start(lane, bank, group, max(earliest, activate_not_before), arrival,
            g, t, bf, pc, history, resources, windows, window_counts)
        anchor = _after(activate, t[0], t[0] / 2)
        bank_ready = _after(anchor, t[4 if read else 5], t[0] / 2)
    else:
        anchor = bf[bank, 0]
        bank_ready = bf[bank, 3 if read else 4]
    return (_column_start(lane, bank, group, bank_ready, earliest, read, g, t, pc, history,
                          resources, directions), anchor, activate)


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
def _idle_refresh_commands(lane, arrival, g, t, rows, bf, addresses, pc, pc_seen,
                           channel, resources, touched, counts):
    """Keep each non-dyadic command/addition; publish closed-bank state once."""
    ch, clock, interval, cycle = lane // 2, t[0], t[20], t[21]
    due = channel[ch, 1]
    if (not math.isfinite(clock) or clock / 2 <= 0 or due < 0 or cycle < 0
            or interval < cycle + 2 * clock):
        return False
    half = clock / 2
    # Exclude all clock-index exceptions before the first state mutation.
    for value in (arrival, due, interval, cycle, channel[ch, 0]):
        if not math.isfinite(value) or abs(value / half) >= 2. ** 49:
            return False
    cutoff = arrival - cycle - 2 * clock
    if due + interval > cutoff or t[2] != (1. if g[7] else .5):
        return False
    row_id, column_id = ch, g[0] // 2 + ch
    if not (resources[row_id, 0] <= due and resources[column_id, 0] <= due):
        return False
    for pseudo in range(2):
        if not pc[ch * 2 + pseudo, 1] <= due:
            return False
    for bank in range(ch * 2 * g[1], (ch + 1) * 2 * g[1]):
        if addresses[bank] >= 0 and (rows[bank] >= 0 or not bf[bank, 1] <= due):
            return False
    if channel[ch, 0] >= 0 and _after(channel[ch, 0], 2 * clock, half) > due:
        return False
    # Non-grid delays always take clock_after's ordinary IEEE addition.
    # The scope proof above already excludes clock-index exceptions.
    cycle_off_grid = cycle != round(cycle / half) * half
    interval_off_grid = interval != round(interval / half) * half
    blocked0 = blocked1 = 0.
    while due <= cutoff:
        for pseudo in range(2):
            earliest = max(due, resources[column_id, 0]) if g[7] else due
            start, _ = _command(ch, 0, earliest, t[2], True, g, t,
                                resources, touched, counts)
            if g[7]:
                _command(ch, 1, start, 1., True, g, t, resources, touched, counts)
            blocked = start + cycle if cycle_off_grid else _after(start, cycle, half)
            if pseudo == 0:
                blocked0 = blocked
            else:
                blocked1 = blocked
            counts[4] += 1
        next_due = due + interval if interval_off_grid else _after(due, interval, half)
        channel[ch, 1] = next_due
        # Preserve this successful epoch, then let the full controller handle
        # overlap. Returning False here would replay commands and busy time.
        if (blocked0 > next_due or blocked1 > next_due
                or resources[row_id, 0] > next_due or resources[column_id, 0] > next_due):
            break
        due = next_due
    for pseudo in range(2):
        target = ch * 2 + pseudo
        pc_seen[target] = True
        pc[target, 1] = blocked0 if pseudo == 0 else blocked1
        for bank in range(target * g[1], (target + 1) * g[1]):
            if addresses[bank] >= 0:
                bf[bank, 1] = pc[target, 1]
                bf[bank, 5] = pc[target, 1]
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
        if (_idle_refresh(lane, arrival, g, t, rows, bf, addresses, pc, pc_seen,
                          channel, resources, touched, counts)
                or _idle_refresh_commands(lane, arrival, g, t, rows, bf, addresses, pc, pc_seen,
                                          channel, resources, touched, counts)):
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
    if (t[3] != 1.
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
def _rotated_bank(step, g):
    return rotated_bank(step, g[1], g[4])


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _binary_shift(value):
    if value <= 0 or value & (value - 1):
        return -1
    shift = 0
    while value > 1:
        value >>= 1
        shift += 1
    return shift


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _burst_location(cursor, g, binary_mapping, lane_shift, bank_shift, interleave_shift, row_shift):
    """Lane, bank within the lane, activated row and inner burst of a cursor."""
    inner = 0
    if g[13]:
        # HBM4 SID-aware mixed radix: burst | BG-in-SID | lane | column |
        # bank-in-group | SID | row. The two low bursts stay within one PC
        # and alternate its two bank groups, avoiding a tCCDR crossing.
        cursor, bg_in_sid = divmod(cursor, 2)
        lane = cursor % g[0]
        cursor //= g[0]
        column = cursor % g[3]
        cursor //= g[3]
        bank_in_group = cursor % g[4]
        cursor //= g[4]
        sid = cursor % g[14]
        row = cursor // g[14]
        bank_within = sid * g[15] * g[4] + (bg_in_sid % g[15]) * g[4] + bank_in_group
    elif binary_mapping:
        if g[11]:
            stripe = cursor >> row_shift
            bank_within = stripe & (g[1] - 1)
            lane = (stripe >> bank_shift) & (g[0] - 1)
            row = stripe >> (bank_shift + lane_shift)
        else:
            block, inner = cursor >> interleave_shift, cursor & (g[2] - 1)
            lane = block & (g[0] - 1)
            bank_linear = block >> lane_shift
            bank_within = _rotated_bank(bank_linear & (g[1] - 1), g)
            row = (((bank_linear >> bank_shift) << interleave_shift) + inner) >> row_shift
    elif g[11]:  # row_local: complete rows, then banks, then lanes.
        stripe = cursor // g[3]
        bank_within = stripe % g[1]
        lane = (stripe // g[1]) % g[0]
        row = stripe // (g[1] * g[0])
    else:
        block, inner = cursor // g[2], cursor % g[2]
        lane = block % g[0]
        bank_linear = block // g[0]
        bank_within = _rotated_bank(bank_linear % g[1], g)
        row = ((bank_linear // g[1]) * g[2] + inner) // g[3]
    if g[6] >= 0:
        row &= ~(1 << g[6])
    return lane, bank_within, row, inner


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _lookahead_depth(g, t):
    """row_lookahead_depth in _hbm_control."""
    depth = math.ceil(max(t[6], t[2] * t[0]) / ((t[1] + t[2]) * t[0])) + 1
    return min(depth, 2 * g[1] - 1)


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _channel_index(cursor, lane_span, period, span_shift, period_shift):
    """Position of a cursor among its physical channel's bursts."""
    if span_shift >= 0:
        # P is a multiple of 2L. Arithmetic shifts and low-bit masks are
        # floor division/modulo by positive powers of two, including negatives.
        return ((cursor >> period_shift) << (span_shift + 1)) + (cursor & (2 * lane_span - 1))
    return (cursor // period) * 2 * lane_span + (cursor % period) % (2 * lane_span)


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _channel_cursor(index, ch, lane_span, period, span_shift, period_shift):
    if span_shift >= 0:
        return (((index >> (span_shift + 1)) << period_shift)
                + (ch << (span_shift + 1)) + (index & (2 * lane_span - 1)))
    return (index // (2 * lane_span)) * period + ch * 2 * lane_span + index % (2 * lane_span)


@njit(cache=True, nogil=True, fastmath=False)
def _precharge_ahead(target, served, ch, arrival, g, t, binary_mapping, lane_shift, bank_shift,
                     interleave_shift, row_shift, lane_span, period, span_shift, period_shift,
                     rows, bf, addresses, pc,
                     pc_seen, channel, channel_int, resources, touched, counts, active_deadlines,
                     ring_tags, ring_banks, stale):
    """_Controller.precharge_ahead; returns the precharged bank or -1.

    ``ring_tags``/``ring_banks`` cache the bank of each channel position.
    The bank is a pure function of the position, so a matching tag is exact.
    """
    ring = ring_tags.shape[1]
    lane, bank_within, row, _ = _burst_location(_channel_cursor(target, ch, lane_span, period, span_shift, period_shift), g,
        binary_mapping, lane_shift, bank_shift, interleave_shift, row_shift)
    bank = lane * g[1] + bank_within
    slot = target % ring
    ring_tags[ch, slot], ring_banks[ch, slot] = target, bank
    if rows[bank] < 0 or rows[bank] == row:
        return -1
    for index in range(served, target):
        slot = index % ring
        if ring_tags[ch, slot] != index:
            other_lane, other_bank, _, _ = _burst_location(_channel_cursor(index, ch, lane_span, period, span_shift, period_shift), g,
                binary_mapping, lane_shift, bank_shift, interleave_shift, row_shift)
            ring_tags[ch, slot], ring_banks[ch, slot] = index, other_lane * g[1] + other_bank
        if ring_banks[ch, slot] == bank:
            return -1
    for iteration in range(2 * g[1] + 16):
        start = _precharge_start(lane, bank, arrival, g, t, bf, pc, channel, channel_int, resources)
        if active_deadlines[ch] <= start:
            if g[8] and channel[ch, 1] < 0:
                channel[ch, 1] = max(t[20], math.floor(arrival / t[20]) * t[20])
            if g[8] and channel[ch, 1] <= active_deadlines[ch]:
                _refresh_until(lane, channel[ch, 1], arrival, g, t, rows, bf, addresses,
                               pc, pc_seen, channel, channel_int, resources, touched, counts)
            else:
                _close_expiring(ch, start, g, t, rows, bf, addresses, pc, channel,
                                channel_int, resources, touched, counts)
        elif not _refresh_until(lane, start, arrival, g, t, rows, bf, addresses,
                                pc, pc_seen, channel, channel_int, resources, touched, counts):
            break
        active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
        if rows[bank] < 0 or rows[bank] == row:
            return -1
    else:
        raise ValueError("HBM timing inputs leave no row-command window between maintenance commands")
    _precharge(lane, bank, arrival, g, t, rows, bf, pc, channel, channel_int,
               resources, touched, counts)
    _close_deadline(ch, bank, g, t, rows, bf, addresses, active_deadlines, stale)
    return bank


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _close_deadline(ch, bank, g, t, rows, bf, addresses, active_deadlines, stale):
    """The channel deadline a full rescan would give after ``bank`` closed.

    The deadline is the minimum over mapped open banks of a monotone function
    of their ACT anchor. Closing a bank whose own deadline lies above that
    minimum leaves it unchanged; otherwise the channel is rescanned. An open
    bank mapped since the last scan (``stale``) always forces the rescan.
    """
    if stale[ch] or (t[29] > 0 and _after(bf[bank, 0], t[29], t[0] / 2) <= active_deadlines[ch]):
        active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
        stale[ch] = 0


@njit(cache=True, nogil=True, fastmath=False)
def _fold_column_cycles(cycles, channel_first, channel_stop, read, recovery, g, t, profile,
                      previous_columns, previous_data, previous_history,
                      tick_columns, tick_data, tick_history, tick_base, tick_banks,
                      cycle_starts, group_tops, sid_tops, deadlines,
                      bf, pc, history, channel, resources, moved, counts):
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
            deadline = deadlines[ch]
            if g[8]:
                deadline = min(deadline, channel[ch, 1])
            if (last + profile[2] + max(profile[3], profile[4]) < profile[1]
                    and horizon < deadline):
                break
            cycles -= 1
        if cycles == 0:
            return 0, 0.
    # Counts contain actual Transfer.duration_ns = end-start, whose rounding
    # can vary with the edge. Replay only those subtractions/additions in the
    # original bank/channel/PC/burst order; no controller work is repeated.
    for repeat in range(1, cycles + 1):
        for step in range(g[1]):
            bank_within = _rotated_bank(step, g)
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
            last = cycle_starts[lane * g[1] + _rotated_bank(g[1] - 1, g), g[2] - 1]
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
                    windows, window_counts, scratch, channel_first, channel_stop):
    """Run complete bursts; caller commits arrays only after successful return."""
    completion = arrival
    hits = misses = conflicts = 0
    counts = np.zeros(7, dtype=np.float64)
    # Parallel dispatch probes this exact specialization on the calling
    # thread before workers can mutate live arrays. An empty interval must
    # not initialize geometry-sized workspaces or inspect live deadlines.
    if first >= stop or channel_first >= channel_stop:
        return completion, hits, misses, conflicts, counts
    channels = g[0] // 2
    active_deadlines, integers = scratch
    lanes, bank_count, entries = g[0], rows.shape[0], history.shape[1]
    history_size = lanes * entries
    group_size = lanes * ((g[1] + 15) // 16) * 4
    fold_geometry = (not g[11] and g[2] <= min(g[3], 64)
                     and g[3] % g[2] == 0 and bank_count * g[2] <= 1048576)
    cycle_size = bank_count * g[2] if fold_geometry else 0
    position = 0
    tick_history = integers[position:position + history_size].reshape((lanes, entries))
    position += history_size
    tick_banks = integers[position:position + bank_count]
    position += bank_count
    tick_base = integers[position:position + lanes]
    position += lanes
    tick_data = integers[position:position + lanes]
    position += lanes
    tick_columns = integers[position:position + channels]
    position += channels
    tick_valid = integers[position:position + lanes]
    position += lanes
    tick_group_tops = integers[position:position + group_size].reshape((lanes, (g[1] + 15) // 16, 4))
    position += group_size
    tick_sid_tops = integers[position:position + lanes * 4].reshape((lanes, 4))
    position += lanes * 4
    cycle_starts = integers[position:position + cycle_size].reshape(
        (bank_count if fold_geometry else 0, g[2] if fold_geometry else 0))
    position += cycle_size
    previous_columns = integers[position:position + channels]
    position += channels
    previous_data = integers[position:position + lanes]
    position += lanes
    previous_history = integers[position:position + history_size].reshape((lanes, entries))
    position += history_size
    look = integers[position:position + channels]
    position += channels
    early_banks = integers[position:position + bank_count]
    position += bank_count
    ring_tags = integers[position:position + bank_count].reshape((channels, 2 * g[1]))
    position += bank_count
    ring_banks = integers[position:position + bank_count].reshape((channels, 2 * g[1]))
    position += bank_count
    stale = integers[position:position + channels]
    position += channels
    look[channel_first:channel_stop] = -1
    early_banks[2 * channel_first * g[1]:2 * channel_stop * g[1]] = 0
    ring_tags[channel_first:channel_stop] = -1
    stale[channel_first:channel_stop] = 0
    depth = _lookahead_depth(g, t)
    for ch in range(channel_first, channel_stop):
        active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
    tick_candidate = g[12] and stop - first >= 64
    tick_profile = _column_tick_profile(g, t, read) if tick_candidate else np.zeros(8, dtype=np.int64)
    tick_mode = tick_candidate and tick_profile[0]
    fold_mode = tick_mode and fold_geometry
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
    # Prove positive, representable P and 2L once. Do not reinterpret overflowed
    # or non-binary geometry in the per-burst cursor helpers. Row-local uses
    # an entire row/bank lane stripe; ordinary mapping uses its interleave.
    span_shift = row_shift + bank_shift if g[11] else interleave_shift
    period_shift = span_shift + lane_shift
    if (not binary_mapping or not 0 <= span_shift < 62
            or not span_shift + 1 <= period_shift < 63):
        span_shift, period_shift = -1, -1
    elif lane_span != 1 << span_shift or period != 1 << period_shift:
        span_shift, period_shift = -1, -1
    while stripe_start < stop:
        if fold_mode:
            cycle_span = g[0] * g[1] * g[2]
            if (stripe_start - offset) % cycle_span == 0:
                cycle_index = (stripe_start - offset) // cycle_span
                row_phase = cycle_index % (g[3] // g[2])
                if previous_valid and cycle_fast and row_phase:
                    # The row's last cycle runs burst by burst: lookahead
                    # within it reaches the next row's bursts.
                    cycles = min(g[3] // g[2] - row_phase - 1,
                                 (stop - (stripe_start - offset)) // cycle_span)
                    if cycles > 0:
                        folded, folded_completion = _fold_column_cycles(cycles, channel_first, channel_stop,
                            read, recovery, g, t, tick_profile, previous_columns, previous_data, previous_history,
                            tick_columns, tick_data, tick_history, tick_base, tick_banks, cycle_starts,
                            tick_group_tops, tick_sid_tops, active_deadlines,
                            bf, pc, history, channel, resources, moved, counts)
                        if folded:
                            hits += folded * 2 * (channel_stop - channel_first) * g[1] * g[2]
                            completion = max(completion, folded_completion)
                            stripe_start += folded * cycle_span
                            # Every lookahead target inside the folded cycles
                            # and one cycle beyond is a hit of this open row.
                            following = ((stripe_start - offset) // period) * 2 * lane_span + depth
                            for ch in range(channel_first, channel_stop):
                                look[ch] = max(look[ch], following)
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
            lane, bank_within, row, inner = _burst_location(cursor, g, binary_mapping, lane_shift,
                                                            bank_shift, interleave_shift, row_shift)
            if binary_mapping:
                group, history_entry = bank_within >> group_shift, bank_within >> history_shift
            else:
                group, history_entry = bank_within // g[4], bank_within // g[10]
            bank = lane * g[1] + bank_within
            ch = lane // 2
            served = _channel_index(cursor, lane_span, period, span_shift, period_shift)
            if look[ch] < 0:
                look[ch] = served
            while (look[ch] <= served + depth
                   and _channel_cursor(look[ch], ch, lane_span, period, span_shift, period_shift) < stop):
                early = _precharge_ahead(look[ch], served, ch, arrival, g, t, binary_mapping,
                    lane_shift, bank_shift, interleave_shift, row_shift, lane_span, period, span_shift, period_shift,
                    rows, bf, addresses, pc, pc_seen, channel, channel_int, resources, touched,
                    counts, active_deadlines, ring_tags, ring_banks, stale)
                if early >= 0:
                    early_banks[early] = 1
                    cycle_fast = False
                    if tick_mode:
                        tick_banks[early] = -2
                look[ch] += 1
            if addresses[bank] < 0:
                addresses[bank] = cursor * g[9]
                # A restored open row without a mapping joins the deadline
                # set only at the next full scan, as before.
                if rows[bank] >= 0:
                    stale[ch] = 1
            pc_seen[lane] = True
            if (tick_mode and tick_valid[lane] and rows[bank] == row
                    and tick_data[lane] >= 0 and tick_columns[ch] >= 0):
                if g[8] and channel[ch, 1] < 0:
                    # Establish the ordinary first-burst refresh phase before
                    # a row-hit shortcut. Untouched channels remain absent.
                    channel[ch, 1] = max(t[20], math.floor(arrival / t[20]) * t[20])
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
                    if (last_tick < tick_profile[1] and horizon < active_deadlines[ch]
                            and (not g[8] or start < channel[ch, 1])):
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
                command_start, anchor, activate = _estimate(lane, bank, group, row, arrival,
                    activate_not_before, read, g, t, rows, bf, pc, history, channel, channel_int,
                    resources, directions, windows, window_counts)
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
                # A closed bank changes no state between the final estimate
                # and its ACT: a False _refresh_until only seeds the refresh
                # phase, which neither ACT nor column timing reads, and the
                # ACT command writes only the row bus. Both estimated edges
                # are therefore exactly the ones a recomputation would give.
                closed = rows[bank] < 0
                if closed:
                    # A row precharged ahead for this burst was a conflict.
                    if early_banks[bank]:
                        conflicts += 1
                        early_banks[bank] = 0
                    else:
                        misses += 1
                    start = activate
                else:
                    conflicts += 1
                    _precharge(lane, bank, arrival, g, t, rows, bf, pc, channel, channel_int,
                               resources, touched, counts)
                    _close_deadline(ch, bank, g, t, rows, bf, addresses, active_deadlines, stale)
                    start = _activate_start(lane, bank, group, activate_not_before, arrival,
                        g, t, bf, pc, history, resources, windows, window_counts)
                start, _ = _command(lane // 2, 0, start, t[1], True, g, t,
                                     resources, touched, counts)
                closed = closed and start == activate
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
                if stale[ch]:
                    active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
                    stale[ch] = 0
                elif t[29] > 0:
                    # The opened bank joins the channel minimum; see _close_deadline.
                    active_deadlines[ch] = min(active_deadlines[ch], _after(anchor, t[29], t[0] / 2))
                start = command_start if closed else _column_start(lane, bank, group,
                    bf[bank, 3 if read else 4], arrival, read, g, t, pc, history, resources, directions)
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
                if g[8]:
                    # REFab changes the PC's blocked frontier. The next
                    # row-hit recurrence must include that new lower bound.
                    base = max(arrival, pc[lane, 1])
                    for entry in range(history.shape[1]):
                        base = max(base, history[lane, entry, 3 if read else 1])
                    tick_base[lane] = _bounded_tick(base, t[0], tick_profile[1], True)
                    tick_valid[lane] = tick_valid[lane] and tick_base[lane] >= 0
        if period >= stop - stripe_start:
            break
        stripe_start += period
    return completion, hits, misses, conflicts, counts


def _new_channel_scratch(g, arrays):
    """One channel-partitioned workspace; it contains no controller state."""
    lanes, banks = int(g[0]), int(g[1])
    bank_count, history_entries = len(arrays[0]), arrays[4].shape[1]
    fold = (not g[11] and g[2] <= min(g[3], 64) and g[3] % g[2] == 0
            and bank_count * int(g[2]) <= 1048576)
    integers = (2 * lanes * history_entries + bank_count + 4 * lanes + lanes
                + lanes * ((banks + 15) // 16) * 4 + lanes * 4
                + (bank_count * int(g[2]) if fold else 0) + 2 * (lanes // 2) + 3 * bank_count)
    # Two flat buffers avoid boxing thirteen separate NumPy descriptors at
    # every compiled worker entry. All views are made inside the controller.
    return np.empty(lanes // 2, dtype=np.float64), np.empty(integers, dtype=np.int64)


def run_hbm_numeric(first, stop, arrival, read, g, t, *arrays):
    """Lease scratch until every channel finishes, including failure paths.

    Each worker initializes and uses only its own physical channels. Scratch
    survives successful execution and rollback alike, but no value survives
    as an input to a later request. Caller-local leases also isolate different
    controllers, concurrent callers and signal-handler reentrant submissions.
    """
    key = (int(g[0]), int(g[1]), int(g[2]), int(g[3]), bool(g[11]),
           len(arrays[0]), arrays[4].shape[1])
    cached = getattr(_CHANNEL_SCRATCH, "lease", None)
    if cached is None or cached[0] != key or cached[2]:
        lease = [key, _new_channel_scratch(g, arrays), True]
        if cached is None or not cached[2]:
            _CHANNEL_SCRATCH.lease = lease
    else:
        lease = cached
        lease[2] = True
    try:
        memo = _NUMERIC_MEMO_CONTEXT.get()
        if memo is not None:
            from . import _compiled_hbm as memo_bridge
            policy = (memo_bridge._numeric_memo_execution_policy()
                      if memo_bridge._numeric_memo_execution_policy is memo_bridge._OWNED_NUMERIC_MEMO_POLICY else None)
            if policy is not None:
                return memo.execute(_run_hbm_numeric, first, stop, arrival, read, g, t, arrays, lease[1],
                                    execution_policy=policy)
        return _run_hbm_numeric(first, stop, arrival, read, g, t, arrays, lease[1])
    finally:
        lease[2] = False


def _run_hbm_numeric(first, stop, arrival, read, g, t, arrays, scratch):
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
        return _run_hbm_channel_range(first, stop, arrival, read, g, t, *arrays, scratch,
                                      channel_first, channel_stop)
    # Compile/cache-load on the calling thread before touching shared arrays.
    # An empty interval performs no mutation and avoids concurrent first JIT.
    _run_hbm_channel_range(first, first, arrival, read, g, t, *arrays, scratch, 0, 0)
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
                                          *arrays, scratch, channel_first, channel_stop)
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


# Per-PC scheduler counters kept with the controller state, one row per lane.
FR_STAT_FIELDS = ("row_hits", "row_misses", "row_conflicts", "activate_commands",
                  "precharge_commands", "direction_switches", "write_drains",
                  "peak_read_queue", "peak_write_queue", "max_read_wait_ns")


# Clock delay ticks a kernel precomputes; a delay off the half-clock grid has none.
_NO_TICK = -(1 << 62)


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _fr_after(value, delay, delay_tick, half, inverse):
    """_after for a precomputed delay tick, without its divisions.

    Below 2**50 ticks a represented grid value lies within a few ulps of its
    integer under either form of the ratio, so both round to the same tick;
    a value off the grid fails the membership test under both.
    """
    ratio = value * inverse
    if delay_tick != _NO_TICK and abs(ratio) < 2. ** 50:
        tick = round(ratio)
        if value == tick * half:
            return (tick + delay_tick) * half
        return value + delay
    return _after(value, delay, half)


@njit(cache=True, nogil=True, fastmath=False, inline="always")
def _fr_tick(delay, half):
    tick = round(delay / half)
    return tick if abs(delay / half) < 2. ** 50 and delay == tick * half else _NO_TICK


@njit(cache=True, nogil=True, fastmath=False)
def _fr_column_start(lane, group, sid, bank_ready, earliest, read, t, ticks, half, inverse, entry_group,
                     entry_sid, grouped, grid_limit, pc, history, column_bus, data, resources, directions):
    """_column_start over precomputed group tables.

    Every term only raises ``ready``. A term whose plain sum lies half a clock
    below it cannot pass it, so its exact clock addition is not computed.
    """
    ready = max(earliest, bank_ready, pc[lane, 1], resources[column_bus, 0])
    op_index = 1 if read else 2
    same = different = different_sid = previous_read = -1.
    for entry in range(entry_group.shape[0]):
        previous_group = entry_group[entry]
        stamp = history[lane, entry, op_index]
        if stamp >= 0:
            if grouped and stamp < grid_limit:
                if read and entry_sid[entry] != sid:
                    different_sid = max(different_sid, stamp)
                elif previous_group == group:
                    same = max(same, stamp)
                else:
                    different = max(different, stamp)
            else:
                gap = t[11] if previous_group == group else t[12]
                if read and entry_sid[entry] != sid:
                    gap = t[13]
                ready = max(ready, _after(stamp, gap, half))
        if read:
            end = history[lane, entry, 3]
            if end >= 0:
                index = 16 if previous_group == group else 17
                if end + t[index] + half > ready:
                    ready = max(ready, end if t[index] == 0. else _fr_after(end, t[index], ticks[index], half, inverse))
        else:
            previous = history[lane, entry, 1]
            if previous >= 0:
                if grouped and previous < grid_limit:
                    previous_read = max(previous_read, previous)
                else:
                    ready = max(ready, _after(previous, t[18], half))
    if same >= 0 and same + t[11] + half > ready:
        ready = max(ready, _fr_after(same, t[11], ticks[11], half, inverse))
    if different >= 0 and different + t[12] + half > ready:
        ready = max(ready, _fr_after(different, t[12], ticks[12], half, inverse))
    if different_sid >= 0 and different_sid + t[13] + half > ready:
        ready = max(ready, _fr_after(different_sid, t[13], ticks[13], half, inverse))
    if previous_read >= 0 and previous_read + t[18] + half > ready:
        ready = max(ready, previous_read if t[18] == 0.
                    else _fr_after(previous_read, t[18], ticks[18], half, inverse))
    data_ready = resources[data, 0]
    if directions[data] and directions[data] != (1 if read else 2):
        turnaround = 25 if read else 24
        data_ready = _fr_after(data_ready, t[turnaround], ticks[turnaround], half, inverse)
    data_ready = max(data_ready, pc[lane, 2 if read else 3])
    latency = 22 if read else 23
    start = _align(max(ready, _fr_after(data_ready, -t[latency],
                                        -ticks[latency] if ticks[latency] != _NO_TICK else _NO_TICK,
                                        half, inverse)), t[0])
    if _fr_after(start, t[latency], ticks[latency], half, inverse) < data_ready:
        start = _fr_after(start, t[0], ticks[0], half, inverse)
    return start


@njit(cache=True, nogil=True, fastmath=False)
def _fr_faw_ready(lane, earliest, frontier, width, width_tick, half, inverse, windows, window_counts):
    """_faw_ready on a precomputed window tick."""
    if width <= 0:
        return earliest
    count = window_counts[lane]
    expired = 0
    while expired < count and _fr_after(windows[lane, expired], width, width_tick, half, inverse) <= frontier:
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
            if last < _fr_after(first, width, width_tick, half, inverse):
                next_start = max(next_start, _fr_after(windows[lane, offset], width, width_tick, half, inverse))
        if next_start == earliest:
            return earliest
        earliest = next_start


@njit(cache=True, nogil=True, fastmath=False)
def _fr_activate_start(lane, bank, group, earliest, groups, grouped, t, ticks, clock_back, half, inverse, grid_limit,
                       bf, pc, history, resources, windows, window_counts):
    """_activate_start for an entry already on the queue, on precomputed ticks."""
    ready = max(earliest, bf[bank, 1], pc[lane, 1], resources[lane // 2, 0])
    if bf[bank, 0] >= 0:
        ready = max(ready, _fr_after(_fr_after(bf[bank, 0], t[8], ticks[8], half, inverse),
                                     -t[0], clock_back, half, inverse))
    anchor = _fr_after(_align(ready, t[0]), t[0], ticks[0], half, inverse)
    other = -1.
    for previous_group in range(groups):
        stamp = history[lane, previous_group, 0]
        same_group = previous_group == group
        gap = t[14] if same_group else t[15]
        if stamp >= 0 and gap > 0:
            if grouped and stamp < grid_limit and not same_group:
                other = max(other, stamp)
            else:
                anchor = max(anchor, _fr_after(stamp, gap, ticks[14 if same_group else 15], half, inverse))
    if other >= 0:
        anchor = max(anchor, _fr_after(other, t[15], ticks[15], half, inverse))
    frontier = max(0., resources[lane // 2, 0])
    while True:
        anchor = _fr_faw_ready(lane, anchor, frontier, t[30], ticks[30], half, inverse, windows, window_counts)
        start = _align(_fr_after(anchor, -t[0], clock_back, half, inverse), t[0])
        aligned = _fr_after(start, t[0], ticks[0], half, inverse)
        if t[30] <= 0 or aligned == anchor:
            return start
        anchor = aligned


@njit(cache=True, nogil=True, fastmath=False)
def _fr_precharge_start(lane, bank, earliest, step, t, ticks, follow_tick, half, inverse, bf, pc, channel, channel_int,
                        resources):
    """_precharge_start on precomputed ticks."""
    ready = max(earliest, bf[bank, 2], pc[lane, 1])
    if pc[lane, 0] >= 0:
        ready = max(ready, _fr_after(pc[lane, 0], t[19], ticks[19], half, inverse))
    if channel_int[lane // 2, 0] == bank:
        ready = max(ready, _fr_after(channel[lane // 2, 0], 2 * t[0], follow_tick, half, inverse))
    return _align(max(ready, resources[lane // 2, 0]), step)


@njit(cache=True, nogil=True, fastmath=False)
def _fr_lane_cursor(first, stop, lane, lanes):
    """First burst of [first, stop) on a lane; the stripe is two bursts wide."""
    block = first >> 1
    offset = (lane - block) % lanes
    cursor = first if offset == 0 else (block + offset) << 1
    return cursor if cursor < stop else -1


@njit(cache=True, nogil=True, fastmath=False)
def _fr_next_request(start, direction, lane, lanes, firsts, stops, reads):
    """The next request of one direction reaching this lane, with its burst."""
    for request in range(start, firsts.shape[0]):
        if reads[request] == (direction == 0):
            cursor = _fr_lane_cursor(firsts[request], stops[request], lane, lanes)
            if cursor >= 0:
                return request, cursor
    return -1, -1


@njit(cache=True, nogil=True, fastmath=False)
def _fr_recount(ch, banks, rows, count, head, q, q_bank, q_row, pending_hit, dirty):
    """Rebuild each direction's open-row hit counts after maintenance closed rows."""
    for pseudo in range(2):
        target = 2 * ch + pseudo
        for direction in range(2):
            dirty[pseudo, direction] = True
            pending_hit[pseudo, direction, :] = 0
            slot = head[pseudo, direction]
            for _ in range(count[pseudo, direction]):
                bank = q_bank[pseudo, direction, slot]
                if rows[bank] == q_row[pseudo, direction, slot]:
                    pending_hit[pseudo, direction, bank - target * banks] += 1
                slot += 1
                if slot == q[direction]:
                    slot = 0


# A batch mirrors channels when it averages this many bursts per channel, and
# keeps at most this many simulated channels to mirror from.
FR_MIRROR_MIN_BURSTS = 64
FR_MIRROR_SOURCES = 128


@njit(cache=True, nogil=True, fastmath=False)
def _fr_signature(ch, lanes, firsts, stops, signature):
    """Each lane's stream per request: its first burst with the lane bits removed, and its length.

    The bank and row of a burst do not depend on its lane, so equal signatures
    are equal command streams.
    """
    position = 0
    for pseudo in range(2):
        lane = 2 * ch + pseudo
        for request in range(firsts.shape[0]):
            cursor = _fr_lane_cursor(firsts[request], stops[request], lane, lanes)
            if cursor < 0:
                signature[position], signature[position + 1] = -1, 0
            else:
                span = stops[request] - cursor
                signature[position] = cursor - 2 * lane
                signature[position + 1] = 2 * (span // (2 * lanes)) + min(span % (2 * lanes), 2 - (cursor & 1))
            position += 2


@njit(cache=True, nogil=True, fastmath=False)
def _fr_first_live(windows, count, frontier, t, half):
    """Index of the first ACT window entry the row bus frontier has not expired."""
    index = 0
    while index < count and _after(windows[index], t[30], half) <= frontier:
        index += 1
    return index


@njit(cache=True, nogil=True, fastmath=False)
def _fr_state_matches(source, ch, lowest, g, t, half, rows, bf, addresses, pc, history, channel, channel_int,
                      resources, directions, windows, window_counts, sources):
    """Whether this channel behaves as the source channel did from its saved state.

    Bus and data frontiers must be equal. A timestamp may differ only where
    no command of the batch can see it: every later edge is at or above
    ``lowest``, the earlier of the batch's first arrival and the next refresh,
    so a ready time below that and a stamp whose largest gap ends below it
    are the same state. Open-row deadlines are checked by the caller.
    """
    rows_before, addresses_before, bf_before, pc_before = sources[4], sources[6], sources[8], sources[10]
    history_before, windows_before, counts_before = sources[12], sources[14], sources[15]
    channel_before, channel_int_before = sources[18], sources[20]
    resources_before, directions_before = sources[22], sources[24]
    lanes, banks = g[0], g[1]
    channels = lanes // 2
    first_lane, first_bank = 2 * ch, 2 * ch * banks
    if channel[ch, 0] != channel_before[source, 0] or channel[ch, 1] != channel_before[source, 1]:
        return False
    for index in range(2):
        offset = first_bank if index == 0 else first_lane
        value = channel_int[ch, index]
        if (value - offset if value >= 0 else value) != channel_int_before[source, index]:
            return False
    rids = (ch, channels + ch, 2 * channels + first_lane, 2 * channels + first_lane + 1)
    for index in range(4):
        if resources[rids[index], 0] != resources_before[source, index, 0]:
            return False
    for pseudo in range(2):
        if directions[rids[2 + pseudo]] != directions_before[source, pseudo]:
            return False
    pre_floor = lowest - t[19] - half
    for pseudo in range(2):
        lane = first_lane + pseudo
        if (pc[lane, 2] != pc_before[source, pseudo, 2] or pc[lane, 3] != pc_before[source, pseudo, 3]
                or max(pc[lane, 1], lowest) != max(pc_before[source, pseudo, 1], lowest)
                or max(pc[lane, 0], pre_floor) != max(pc_before[source, pseudo, 0], pre_floor)):
            return False
    activate_floor = lowest - t[8] + t[0] - half
    for offset in range(2 * banks):
        bank = first_bank + offset
        if (rows[bank] != rows_before[source, offset]
                or (addresses[bank] >= 0) != (addresses_before[source, offset] >= 0)
                or max(bf[bank, 0], activate_floor) != max(bf_before[source, offset, 0], activate_floor)):
            return False
        for field in range(1, 7):
            if max(bf[bank, field], lowest) != max(bf_before[source, offset, field], lowest):
                return False
    row_floor = lowest - max(t[14], t[15]) - half
    column_floor = lowest - max(t[11], t[12], t[13], t[18]) - half
    end_floor = lowest - max(t[16], t[17]) - half
    for pseudo in range(2):
        lane = first_lane + pseudo
        for entry in range(history.shape[1]):
            if (max(history[lane, entry, 0], row_floor) != max(history_before[source, pseudo, entry, 0], row_floor)
                    or max(history[lane, entry, 1], column_floor) != max(history_before[source, pseudo, entry, 1], column_floor)
                    or max(history[lane, entry, 2], column_floor) != max(history_before[source, pseudo, entry, 2], column_floor)
                    or max(history[lane, entry, 3], end_floor) != max(history_before[source, pseudo, entry, 3], end_floor)):
                return False
        # Expired ACT entries leave at the next ACT plan on both channels.
        frontier = resources[ch, 0]
        mine = _fr_first_live(windows[lane], window_counts[lane], frontier, t, half)
        theirs = _fr_first_live(windows_before[source, pseudo], counts_before[source, pseudo], frontier, t, half)
        if window_counts[lane] - mine != counts_before[source, pseudo] - theirs:
            return False
        for index in range(window_counts[lane] - mine):
            if windows[lane, mine + index] != windows_before[source, pseudo, theirs + index]:
                return False
    return True


@njit(cache=True, nogil=True, fastmath=False)
def _fr_capture_before(source, ch, g, rows, bf, addresses, pc, history, channel, channel_int, resources,
                       directions, windows, window_counts, sources):
    source_channel = sources[0]
    rows_before, addresses_before, bf_before, pc_before = sources[4], sources[6], sources[8], sources[10]
    history_before, windows_before, counts_before = sources[12], sources[14], sources[15]
    channel_before, channel_int_before = sources[18], sources[20]
    resources_before, directions_before = sources[22], sources[24]
    banks, channels = g[1], g[0] // 2
    first_lane, first_bank = 2 * ch, 2 * ch * banks
    source_channel[source] = ch
    rows_before[source] = rows[first_bank:first_bank + 2 * banks]
    addresses_before[source] = addresses[first_bank:first_bank + 2 * banks]
    bf_before[source] = bf[first_bank:first_bank + 2 * banks]
    pc_before[source] = pc[first_lane:first_lane + 2]
    history_before[source] = history[first_lane:first_lane + 2]
    windows_before[source] = windows[first_lane:first_lane + 2]
    counts_before[source] = window_counts[first_lane:first_lane + 2]
    channel_before[source] = channel[ch]
    for index in range(2):
        offset = first_bank if index == 0 else first_lane
        value = channel_int[ch, index]
        channel_int_before[source, index] = value - offset if value >= 0 else value
    rids = (ch, channels + ch, 2 * channels + first_lane, 2 * channels + first_lane + 1)
    for index in range(4):
        resources_before[source, index] = resources[rids[index]]
    for pseudo in range(2):
        directions_before[source, pseudo] = directions[rids[2 + pseudo]]


@njit(cache=True, nogil=True, fastmath=False)
def _fr_capture_after(source, ch, usable, reach_open, reach, g, rows, bf, addresses, pc, history, channel, channel_int, resources,
                      directions, windows, window_counts, pc_seen, touched, partial_moved, partial_stats,
                      partial_int, partial_float, partial_counts, sources):
    (_, source_usable, source_reach, _, _, rows_after, _, addresses_after, _, bf_after, _, pc_after, _, history_after,
     _, _, windows_after, counts_after, _, channel_after, _, channel_int_after, _, resources_after, _,
     directions_after, touched_after, seen_after, moved_after, stats_after, int_after, float_after,
     counts_total) = sources
    banks, channels = g[1], g[0] // 2
    first_lane, first_bank = 2 * ch, 2 * ch * banks
    source_usable[source] = usable
    source_reach[source, 0], source_reach[source, 1] = reach_open, reach
    rows_after[source] = rows[first_bank:first_bank + 2 * banks]
    addresses_after[source] = addresses[first_bank:first_bank + 2 * banks]
    bf_after[source] = bf[first_bank:first_bank + 2 * banks]
    pc_after[source] = pc[first_lane:first_lane + 2]
    history_after[source] = history[first_lane:first_lane + 2]
    windows_after[source] = windows[first_lane:first_lane + 2]
    counts_after[source] = window_counts[first_lane:first_lane + 2]
    channel_after[source] = channel[ch]
    for index in range(2):
        offset = first_bank if index == 0 else first_lane
        value = channel_int[ch, index]
        channel_int_after[source, index] = value - offset if value >= 0 else value
    rids = (ch, channels + ch, 2 * channels + first_lane, 2 * channels + first_lane + 1)
    for index in range(4):
        resources_after[source, index] = resources[rids[index]]
        touched_after[source, index] = touched[rids[index]]
    for pseudo in range(2):
        directions_after[source, pseudo] = directions[rids[2 + pseudo]]
        seen_after[source, pseudo] = pc_seen[first_lane + pseudo]
        moved_after[source, pseudo] = partial_moved[pseudo]
    stats_after[source] = partial_stats
    int_after[source] = partial_int
    float_after[source] = partial_float
    counts_total[source] = partial_counts


@njit(cache=True, nogil=True, fastmath=False)
def _fr_merge_tallies(ch, g, saved_busy, partial_busy, partial_moved, partial_stats, partial_int, partial_float,
                      partial_counts, resources, moved, stats, out_int, out_float, counts):
    """Add one channel's own tallies to the batch's, in channel order.

    Busy time and bus counters are summed per channel, so a mirrored channel
    adds exactly what the channel it mirrors added.
    """
    channels = g[0] // 2
    first_lane = 2 * ch
    rids = (ch, channels + ch, 2 * channels + first_lane, 2 * channels + first_lane + 1)
    for index in range(4):
        resources[rids[index], 1] = saved_busy[index] + partial_busy[index]
    for pseudo in range(2):
        moved[rids[2 + pseudo]] += partial_moved[pseudo]
        lane = first_lane + pseudo
        for field in range(7):
            stats[lane, field] += partial_stats[pseudo, field]
        for field in range(7, 10):
            stats[lane, field] = max(stats[lane, field], partial_stats[pseudo, field])
    for request in range(out_int.shape[0]):
        for field in range(3):
            out_int[request, field] += partial_int[request, field]
        out_float[request, 0] = max(out_float[request, 0], partial_float[request, 0])
        out_float[request, 1] = max(out_float[request, 1], partial_float[request, 1])
    for field in range(7):
        counts[field] += partial_counts[field]


@njit(cache=True, nogil=True, fastmath=False)
def _fr_apply_source(source, ch, g, rows, bf, addresses, pc, history, pc_seen, channel, channel_int, resources, moved,
                     directions, touched, windows, window_counts, stats, out_int, out_float, counts, sources):
    """Give this channel the source channel's batch, keeping what that batch did not write."""
    (source_channel, _, _, _, _, rows_after, addresses_before, addresses_after, bf_before, bf_after, pc_before, pc_after,
     history_before, history_after, windows_before, counts_before, windows_after, counts_after, _, channel_after, _,
     channel_int_after, resources_before, resources_after, _, directions_after, touched_after, seen_after, moved_after,
     stats_after, int_after, float_after, counts_total) = sources
    banks, channels = g[1], g[0] // 2
    first_lane, first_bank = 2 * ch, 2 * ch * banks
    # Two bursts of a stripe per lane; first touches of a bank differ by the lane bits.
    shift = 4 * (ch - source_channel[source]) * g[9]
    for offset in range(2 * banks):
        bank = first_bank + offset
        rows[bank] = rows_after[source, offset]
        if addresses_before[source, offset] < 0 <= addresses_after[source, offset]:
            addresses[bank] = addresses_after[source, offset] + shift
        for field in range(7):
            if bf_after[source, offset, field] != bf_before[source, offset, field]:
                bf[bank, field] = bf_after[source, offset, field]
    for pseudo in range(2):
        lane = first_lane + pseudo
        pc_seen[lane] = pc_seen[lane] or seen_after[source, pseudo]
        for field in range(4):
            if pc_after[source, pseudo, field] != pc_before[source, pseudo, field]:
                pc[lane, field] = pc_after[source, pseudo, field]
        for entry in range(history.shape[1]):
            for field in range(4):
                if history_after[source, pseudo, entry, field] != history_before[source, pseudo, entry, field]:
                    history[lane, entry, field] = history_after[source, pseudo, entry, field]
        changed = counts_after[source, pseudo] != counts_before[source, pseudo]
        if not changed:
            for index in range(counts_after[source, pseudo]):
                if windows_after[source, pseudo, index] != windows_before[source, pseudo, index]:
                    changed = True
                    break
        if changed:
            windows[lane] = windows_after[source, pseudo]
            window_counts[lane] = counts_after[source, pseudo]
    channel[ch] = channel_after[source]
    for index in range(2):
        offset = first_bank if index == 0 else first_lane
        value = channel_int_after[source, index]
        channel_int[ch, index] = value + offset if value >= 0 else value
    rids = (ch, channels + ch, 2 * channels + first_lane, 2 * channels + first_lane + 1)
    busy = np.empty(4, dtype=np.float64)
    partial_busy = np.empty(4, dtype=np.float64)
    for index in range(4):
        rid = rids[index]
        busy[index] = resources[rid, 1]
        partial_busy[index] = resources_after[source, index, 1]
        resources[rid, 0] = resources_after[source, index, 0]
        for field in (2, 3):
            if resources_after[source, index, field] != resources_before[source, index, field]:
                resources[rid, field] = resources_after[source, index, field]
        touched[rid] = touched[rid] or touched_after[source, index]
    for pseudo in range(2):
        directions[rids[2 + pseudo]] = directions_after[source, pseudo]
    _fr_merge_tallies(ch, g, busy, partial_busy, moved_after[source], stats_after[source], int_after[source],
                      float_after[source], counts_total[source], resources, moved, stats, out_int, out_float, counts)


@njit(cache=True, nogil=True, fastmath=False)
def _fr_open_rows_safe(source, ch, g, t, rows, bf, addresses, half, source_reach):
    """Whether no row this channel holds open reaches tRASmax while it can still matter.

    The first refresh closes every addressed bank, so those rows only have to
    outlast the source's first refresh; a row a refresh leaves open has to
    outlast its whole batch. Rows opened in the batch are the source's own.
    """
    if t[29] > 0:
        for bank in range(2 * ch * g[1], (2 * ch + 2) * g[1]):
            if rows[bank] >= 0 and _after(bf[bank, 0], t[29], half) <= source_reach[source, 0 if addresses[bank] >= 0 else 1]:
                return False
    return True


@njit(cache=True, nogil=True, fastmath=False)
def _run_fr_fcfs_channels(firsts, stops, arrivals, reads, q, g, t, rows, bf, addresses, pc,
                          history, pc_seen, channel, channel_int, resources, moved, directions,
                          touched, windows, window_counts, stats, out_int, out_float,
                          channel_first, channel_stop, span, channel_counts):
    """Issue one batch's bursts command by command through per-PC queues.

    Each physical channel owns both PCs, their banks and its CA buses, so one
    worker can finish a channel alone. Inside it the lane with the earliest
    legal next command issues first, which keeps both command buses in time
    order. A lane's next command is the earlier of the best row hit's column
    and the row command of its oldest entry without a queued hit on the open
    row; the latter is what closes and opens banks ahead of their turn.
    """
    requests, lanes, banks = firsts.shape[0], g[0], g[1]
    channels = lanes // 2
    half = t[0] / 2
    if requests == 0 or channel_first >= channel_stop:
        return
    inverse = 1. / half
    ticks = np.empty(t.shape[0], dtype=np.int64)
    for index in range(t.shape[0]):
        ticks[index] = _fr_tick(t[index], half)
    # Runtime divisions on the burst path become these tables and shifts.
    bank_group = np.empty(banks, dtype=np.int64)
    bank_entry = np.empty(banks, dtype=np.int64)
    bank_sid = np.empty(banks, dtype=np.int64)
    for bank_within in range(banks):
        bank_group[bank_within] = bank_within // g[4]
        bank_entry[bank_within] = bank_within // g[10]
        bank_sid[bank_within] = bank_within // 16
    entry_group = np.empty(history.shape[1], dtype=np.int64)
    entry_sid = np.empty(history.shape[1], dtype=np.int64)
    for entry in range(history.shape[1]):
        entry_group[entry] = entry * g[10] // g[4]
        entry_sid[entry] = entry * g[10] // 16
    group_sid_group = np.empty(g[5], dtype=np.int64)
    for group in range(g[5]):
        group_sid_group[group] = group // g[15]
    grouped = g[12] != 0
    grid_limit = t[0] * 2. ** 48
    clock_back = -ticks[0] if ticks[0] != _NO_TICK else _NO_TICK
    follow_tick = _fr_tick(2 * t[0], half)
    precharge_step = t[0] if g[7] else half
    lane_shift, column_shift, bank_shift, sid_shift = (_binary_shift(g[0]), _binary_shift(g[3]),
                                                        _binary_shift(g[4]), _binary_shift(g[14]))
    binary = min(lane_shift, column_shift, bank_shift, sid_shift, _binary_shift(g[15])) >= 0
    row_mask = ~(1 << g[6]) if g[6] >= 0 else -1
    # A channel with the traffic and the live state of a simulated one takes
    # its result. The dyadic idle-refresh fold reads busy totals, so a clock it
    # can apply to does not mirror.
    total_bursts = 0
    for request in range(requests):
        total_bursts += stops[request] - firsts[request]
    mirror = (span >= 2 and t[0] * 2 != math.floor(t[0] * 2)
              and total_bursts >= FR_MIRROR_MIN_BURSTS * span)
    capacity_sources = min(FR_MIRROR_SOURCES, channel_stop - channel_first) if mirror else 0
    span_banks, entries = 2 * banks, history.shape[1]
    sources = (np.empty(capacity_sources, dtype=np.int64), np.empty(capacity_sources, dtype=np.bool_),
               np.empty((capacity_sources, 2), dtype=np.float64),
               np.empty((capacity_sources, 4 * requests), dtype=np.int64),
               np.empty((capacity_sources, span_banks), dtype=np.int64),
               np.empty((capacity_sources, span_banks), dtype=np.int64),
               np.empty((capacity_sources, span_banks), dtype=np.int64),
               np.empty((capacity_sources, span_banks), dtype=np.int64),
               np.empty((capacity_sources, span_banks, 7), dtype=np.float64),
               np.empty((capacity_sources, span_banks, 7), dtype=np.float64),
               np.empty((capacity_sources, 2, 4), dtype=np.float64),
               np.empty((capacity_sources, 2, 4), dtype=np.float64),
               np.empty((capacity_sources, 2, entries, 4), dtype=np.float64),
               np.empty((capacity_sources, 2, entries, 4), dtype=np.float64),
               np.empty((capacity_sources, 2, windows.shape[1]), dtype=np.float64),
               np.empty((capacity_sources, 2), dtype=np.int64),
               np.empty((capacity_sources, 2, windows.shape[1]), dtype=np.float64),
               np.empty((capacity_sources, 2), dtype=np.int64),
               np.empty((capacity_sources, 2), dtype=np.float64),
               np.empty((capacity_sources, 2), dtype=np.float64),
               np.empty((capacity_sources, 2), dtype=np.int64),
               np.empty((capacity_sources, 2), dtype=np.int64),
               np.empty((capacity_sources, 4, 4), dtype=np.float64),
               np.empty((capacity_sources, 4, 4), dtype=np.float64),
               np.empty((capacity_sources, 2), dtype=np.int8),
               np.empty((capacity_sources, 2), dtype=np.int8),
               np.empty((capacity_sources, 4), dtype=np.bool_),
               np.empty((capacity_sources, 2), dtype=np.bool_),
               np.empty((capacity_sources, 2), dtype=np.int64),
               np.empty((capacity_sources, 2, stats.shape[1]), dtype=np.float64),
               np.empty((capacity_sources, requests, 3), dtype=np.int64),
               np.empty((capacity_sources, requests, 2), dtype=np.float64),
               np.empty((capacity_sources, 7), dtype=np.float64))
    source_usable, source_reach, source_signature = sources[1], sources[2], sources[3]
    source_count = 0
    signature = np.empty(4 * requests if mirror else 0, dtype=np.int64)
    first_arrival = math.inf
    for request in range(requests):
        first_arrival = min(first_arrival, arrivals[request])
    tally = np.zeros(7, dtype=np.float64) if mirror else channel_counts[0]
    ch_int = np.zeros((requests, 3), dtype=np.int64) if mirror else out_int
    ch_float = np.zeros((requests, 2), dtype=np.float64) if mirror else out_float
    saved_busy = np.empty(4, dtype=np.float64)
    partial_busy = np.empty(4, dtype=np.float64)
    saved_moved = np.empty(2, dtype=np.int64)
    partial_moved = np.empty(2, dtype=np.int64)
    saved_stats = np.empty((2, stats.shape[1]), dtype=np.float64)
    partial_stats = np.empty((2, stats.shape[1]), dtype=np.float64)
    depth = max(q[0], q[1])
    q_bank = np.empty((2, 2, depth), dtype=np.int64)
    q_row = np.empty((2, 2, depth), dtype=np.int64)
    q_request = np.empty((2, 2, depth), dtype=np.int64)
    q_entry = np.empty((2, 2, depth), dtype=np.float64)
    head = np.empty((2, 2), dtype=np.int64)
    count = np.empty((2, 2), dtype=np.int64)
    bypass = np.empty((2, 2), dtype=np.int64)
    iter_request = np.empty((2, 2), dtype=np.int64)
    iter_cursor = np.empty((2, 2), dtype=np.int64)
    slot_free = np.empty((2, 2), dtype=np.float64)
    dirty = np.empty((2, 2), dtype=np.bool_)
    pending_hit = np.empty((2, 2, banks), dtype=np.int64)
    early = np.empty((2, banks), dtype=np.int8)
    active_deadlines = np.empty(channels, dtype=np.float64)
    stale_deadline = np.zeros(channels, dtype=np.int64)
    own_last = np.empty((requests, 2), dtype=np.float64)
    own_wait = np.empty((requests, 2), dtype=np.float64)
    draining = np.empty(2, dtype=np.bool_)
    stale = np.empty(2, dtype=np.bool_)
    reads_since = np.empty(2, dtype=np.int64)
    last_group = np.empty(2, dtype=np.int64)
    lane_request = np.empty(2, dtype=np.int64)
    plan_kind = np.empty(2, dtype=np.int64)
    plan_direction = np.empty(2, dtype=np.int64)
    plan_offset = np.empty(2, dtype=np.int64)
    last_column = np.empty(2, dtype=np.float64)
    act_floor = np.empty(2, dtype=np.float64)
    plan_start = np.empty(2, dtype=np.float64)
    traffic_floor = np.empty(2, dtype=np.float64)
    recovery = (t[9], _after(_after(t[23], t[26], half), t[10], half))
    recovery_ticks = (_fr_tick(recovery[0], half), _fr_tick(recovery[1], half))
    column_duration = t[3] * t[0]
    column_duration_tick = _fr_tick(column_duration, half)
    for ch in range(channel_first, channel_stop):
        row_bus, column_bus = ch, channels + ch
        reached = False
        for pseudo in range(2):
            lane = 2 * ch + pseudo
            traffic_floor[pseudo] = _after(resources[2 * channels + lane, 0], -t[22], half)
            for direction in range(2):
                iter_request[pseudo, direction], iter_cursor[pseudo, direction] = _fr_next_request(
                    0, direction, lane, lanes, firsts, stops, reads)
                reached = reached or iter_request[pseudo, direction] >= 0
        if not reached:
            continue
        capture = -1
        if mirror:
            if channel[ch, 1] >= 0:
                _fr_signature(ch, lanes, firsts, stops, signature)
                lowest = min(first_arrival, channel[ch, 1])
                mirrored = False
                for source in range(source_count):
                    if not source_usable[source] or not np.array_equal(source_signature[source], signature):
                        continue
                    if (_fr_state_matches(source, ch, lowest, g, t, half, rows, bf, addresses, pc, history, channel,
                                          channel_int, resources, directions, windows, window_counts, sources)
                            and _fr_open_rows_safe(source, ch, g, t, rows, bf, addresses, half, source_reach)):
                        _fr_apply_source(source, ch, g, rows, bf, addresses, pc, history, pc_seen, channel, channel_int,
                                         resources, moved, directions, touched, windows, window_counts, stats,
                                         out_int, out_float, channel_counts[ch - channel_first], sources)
                        mirrored = True
                        break
                if mirrored:
                    continue
                if source_count < capacity_sources:
                    capture = source_count
                    source_count += 1
                    source_signature[capture] = signature
                    _fr_capture_before(capture, ch, g, rows, bf, addresses, pc, history, channel, channel_int,
                                       resources, directions, windows, window_counts, sources)
            # The channel's own tallies, added to the batch's when it finishes.
            for pseudo in range(2):
                lane = 2 * ch + pseudo
                data = 2 * channels + lane
                saved_busy[2 + pseudo], resources[data, 1] = resources[data, 1], 0.
                saved_moved[pseudo], moved[data] = moved[data], 0
                saved_stats[pseudo] = stats[lane]
                stats[lane] = 0.
            saved_busy[0], resources[row_bus, 1] = resources[row_bus, 1], 0.
            saved_busy[1], resources[column_bus, 1] = resources[column_bus, 1], 0.
            tally[:] = 0.
            ch_int[:] = 0
            ch_float[:, 0] = -math.inf
            ch_float[:, 1] = 0.
        reach = reach_open = -math.inf
        refreshed = deadline_reached = False
        active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
        head[:] = 0
        count[:] = 0
        bypass[:] = 0
        slot_free[:] = 0.
        dirty[:] = False
        pending_hit[:] = 0
        early[:] = 0
        own_last[:] = -1.
        own_wait[:] = 0.
        draining[:] = False
        stale[:] = True
        reads_since[:] = q[4]
        last_group[:] = -1
        lane_request[:] = -1
        plan_kind[:] = -1
        last_column[:] = -1.
        act_floor[:] = 0.
        plan_start[:] = math.inf
        last_pseudo = 1
        while True:
            for pseudo in range(2):
                if not stale[pseudo]:
                    continue
                stale[pseudo] = False
                lane = 2 * ch + pseudo
                if draining[pseudo] and count[pseudo, 1] <= q[3]:
                    draining[pseudo] = False
                    reads_since[pseudo] = 0
                for direction in range(2):
                    # A drain lowers the write queue only if refill waits for its end.
                    if direction == 1 and draining[pseudo]:
                        continue
                    capacity = q[direction]
                    while count[pseudo, direction] < capacity and iter_request[pseudo, direction] >= 0:
                        request = iter_request[pseudo, direction]
                        cursor = iter_cursor[pseudo, direction]
                        if binary:
                            # The g[13] arm of _burst_location for non-negative cursors.
                            block = cursor >> (1 + lane_shift + column_shift)
                            bank_within = ((block >> bank_shift) & (g[14] - 1)) * g[15] * g[4]
                            bank_within += ((cursor & 1) & (g[15] - 1)) * g[4] + (block & (g[4] - 1))
                            row = (block >> (bank_shift + sid_shift)) & row_mask
                        else:
                            _, bank_within, row, _ = _burst_location(cursor, g, False, 0, 0, 0, 0)
                        bank = lane * banks + bank_within
                        slot = head[pseudo, direction] + count[pseudo, direction]
                        if slot >= capacity:
                            slot -= capacity
                        q_bank[pseudo, direction, slot] = bank
                        q_row[pseudo, direction, slot] = row
                        q_request[pseudo, direction, slot] = request
                        q_entry[pseudo, direction, slot] = max(arrivals[request], slot_free[pseudo, direction])
                        count[pseudo, direction] += 1
                        if addresses[bank] < 0:
                            addresses[bank] = cursor * g[9]
                            if rows[bank] >= 0:
                                stale_deadline[ch] = 1
                        pc_seen[lane] = True
                        if rows[bank] == row:
                            pending_hit[pseudo, direction, bank_within] += 1
                        else:
                            dirty[pseudo, direction] = True
                        cursor = cursor + 1 if not cursor & 1 else cursor - 1 + 2 * lanes
                        if cursor < stops[request]:
                            iter_cursor[pseudo, direction] = cursor
                        else:
                            iter_request[pseudo, direction], iter_cursor[pseudo, direction] = _fr_next_request(
                                request + 1, direction, lane, lanes, firsts, stops, reads)
                    if count[pseudo, direction] > stats[lane, 7 + direction]:
                        stats[lane, 7 + direction] = count[pseudo, direction]
                waiting_reads, waiting_writes = count[pseudo, 0], count[pseudo, 1]
                if waiting_reads + waiting_writes == 0:
                    plan_start[pseudo] = math.inf
                    plan_kind[pseudo] = -1
                    continue
                if (not draining[pseudo] and waiting_reads and waiting_writes >= q[2]
                        and reads_since[pseudo] >= q[4]):
                    draining[pseudo] = True
                    stats[lane, 6] += 1
                column_floor = resources[column_bus, 0]
                if draining[pseudo] or not waiting_reads:
                    direction = 1
                elif not waiting_writes:
                    direction = 0
                else:
                    # A read that has arrived always goes ahead of a queued write.
                    direction = 0 if (q_entry[pseudo, 0, head[pseudo, 0]]
                                      <= max(q_entry[pseudo, 1, head[pseudo, 1]], column_floor)) else 1
                read = direction == 0
                capacity = q[direction]
                waiting = count[pseudo, direction]
                first_slot = head[pseudo, direction]
                forced = bypass[pseudo, direction] >= capacity
                previous, previous_group = last_column[pseudo], last_group[pseudo]
                floor = column_floor if previous < 0 else max(column_floor, previous + t[12])
                hit_offset = row_offset = -1
                best = math.inf
                for offset in range(1 if forced else waiting):
                    slot = first_slot + offset
                    if slot >= capacity:
                        slot -= capacity
                    bank = q_bank[pseudo, direction, slot]
                    if rows[bank] != q_row[pseudo, direction, slot]:
                        continue
                    bank_within = bank - lane * banks
                    group = bank_group[bank_within]
                    ready = max(floor, bf[bank, 3 if read else 4], q_entry[pseudo, direction, slot])
                    stamp = history[lane, bank_entry[bank_within], 1 if read else 2]
                    if stamp >= 0:
                        ready = max(ready, stamp + t[11])
                    if previous >= 0:
                        if previous_group == group:
                            ready = max(ready, previous + t[11])
                        elif read and group_sid_group[previous_group] != group_sid_group[group]:
                            ready = max(ready, previous + t[13])
                    if ready < best:
                        best, hit_offset = ready, offset
                        if ready <= floor:
                            break
                if dirty[pseudo, direction]:
                    unserved = False
                    for offset in range(waiting):
                        slot = first_slot + offset
                        if slot >= capacity:
                            slot -= capacity
                        bank = q_bank[pseudo, direction, slot]
                        if rows[bank] == q_row[pseudo, direction, slot]:
                            continue
                        unserved = True
                        if (rows[bank] >= 0 and pending_hit[pseudo, direction, bank - lane * banks] > 0
                                and not (forced and offset == 0)):
                            continue
                        row_offset = offset
                        break
                    dirty[pseudo, direction] = unserved
                start = math.inf
                kind = -1
                chosen = -1
                if hit_offset >= 0:
                    slot = first_slot + hit_offset
                    if slot >= capacity:
                        slot -= capacity
                    bank = q_bank[pseudo, direction, slot]
                    bank_within = bank - lane * banks
                    start = _fr_column_start(lane, bank_group[bank_within], bank_sid[bank_within],
                                             bf[bank, 3 if read else 4], q_entry[pseudo, direction, slot], read,
                                             t, ticks, half, inverse, entry_group, entry_sid, grouped, grid_limit,
                                             pc, history, column_bus, 2 * channels + lane, resources, directions)
                    kind, chosen = 0, hit_offset
                if row_offset >= 0:
                    slot = first_slot + row_offset
                    if slot >= capacity:
                        slot -= capacity
                    bank = q_bank[pseudo, direction, slot]
                    entry = q_entry[pseudo, direction, slot]
                    if rows[bank] >= 0:
                        row_start = _fr_precharge_start(lane, bank, entry, precharge_step, t, ticks, follow_tick, half,
                                                        inverse, bf, pc, channel, channel_int, resources)
                        row_kind = 1
                    else:
                        # The row bus alone bounds later ACTs, so it alone may prune tFAW history.
                        row_start = _fr_activate_start(lane, bank, bank_group[bank - lane * banks],
                                                       max(entry, act_floor[pseudo]), g[5], grouped, t, ticks,
                                                       clock_back, half, inverse, grid_limit, bf, pc, history,
                                                       resources, windows, window_counts)
                        row_kind = 2
                    if row_start < start:
                        start, kind, chosen = row_start, row_kind, row_offset
                if kind < 0:
                    raise ValueError("HBM scheduler has queued work but no eligible command")
                plan_start[pseudo], plan_kind[pseudo] = start, kind
                plan_direction[pseudo], plan_offset[pseudo] = direction, chosen
            if plan_start[0] == plan_start[1]:
                if plan_kind[0] < 0:
                    break
                pseudo = 1 - last_pseudo
            else:
                pseudo = 0 if plan_start[0] < plan_start[1] else 1
            kind = plan_kind[pseudo]
            start = plan_start[pseudo]
            # The other lane's issue may have passed this plan's bus slot.
            if start < resources[column_bus if kind == 0 else row_bus, 0]:
                stale[pseudo] = True
                continue
            lane = 2 * ch + pseudo
            direction, chosen = plan_direction[pseudo], plan_offset[pseudo]
            read = direction == 0
            capacity = q[direction]
            slot = head[pseudo, direction] + chosen
            if slot >= capacity:
                slot -= capacity
            bank, row = q_bank[pseudo, direction, slot], q_row[pseudo, direction, slot]
            request, entry = q_request[pseudo, direction, slot], q_entry[pseudo, direction, slot]
            bank_within = bank - lane * banks
            group, history_entry = bank_group[bank_within], bank_entry[bank_within]
            horizon = (_fr_after(start, recovery[direction], recovery_ticks[direction], half, inverse)
                       if kind == 0 else start)
            if stale_deadline[ch]:
                active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
                stale_deadline[ch] = 0
            maintained = False
            reach = max(reach, horizon)
            if active_deadlines[ch] <= horizon:
                deadline_reached = True
                if g[8] and channel[ch, 1] < 0:
                    channel[ch, 1] = max(t[20], math.floor(entry / t[20]) * t[20])
                if g[8] and channel[ch, 1] <= active_deadlines[ch]:
                    _refresh_until(lane, channel[ch, 1], entry, g, t, rows, bf, addresses,
                                   pc, pc_seen, channel, channel_int, resources, touched, tally)
                else:
                    _close_expiring(ch, horizon, g, t, rows, bf, addresses, pc, channel,
                                    channel_int, resources, touched, tally)
                maintained = True
            elif (g[8] and channel[ch, 1] <= start
                    and _refresh_until(lane, start, entry, g, t, rows, bf, addresses, pc, pc_seen,
                                       channel, channel_int, resources, touched, tally)):
                maintained = True
            if maintained:
                if not refreshed:
                    refreshed = True
                    reach_open = max(reach, resources[row_bus, 0])
                active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
                _fr_recount(ch, banks, rows, count, head, q, q_bank, q_row, pending_hit, dirty)
                stale[:] = True
                continue
            if kind == 1:
                _precharge(lane, bank, entry, g, t, rows, bf, pc, channel, channel_int,
                           resources, touched, tally)
                _close_deadline(ch, bank, g, t, rows, bf, addresses, active_deadlines, stale_deadline)
                stats[lane, 4] += 1
                early[pseudo, bank_within] = 1
                pending_hit[pseudo, 0, bank_within] = 0
                pending_hit[pseudo, 1, bank_within] = 0
                dirty[pseudo, 0] = dirty[pseudo, 1] = True
                stale[pseudo] = True
                last_pseudo = pseudo
                continue
            if kind == 2:
                anchor = _after(start, t[0], half)
                if t[29] > 0:
                    rcd = t[4 if read else 5]
                    column = _fr_column_start(lane, group, bank_sid[bank_within], _after(anchor, rcd, half), entry,
                                              read, t, ticks, half, inverse, entry_group, entry_sid, grouped,
                                              grid_limit, pc, history, column_bus, 2 * channels + lane, resources,
                                              directions)
                    latest = _after(column, recovery[direction], half)
                    if latest > _after(anchor, t[29], half):
                        if _after(rcd, recovery[direction], half) > t[29]:
                            raise ValueError("HBM access recovery exceeds the tRASmax activation window")
                        # A later DQ reservation keeps the row closed rather than
                        # holding it open for longer than tRASmax.
                        act_floor[pseudo] = _align(max(_after(start, t[0], half),
                            _after(_after(latest, -t[29], half), -t[0], half)), t[0])
                        stale[pseudo] = True
                        continue
                command, _ = _command(ch, 0, start, t[1], True, g, t, resources, touched, tally)
                anchor = _after(command, t[0], half)
                history[lane, group, 0] = anchor
                if t[30] > 0:
                    _record_activate(lane, anchor, windows, window_counts)
                channel[ch, 0] = command
                channel_int[ch, 0], channel_int[ch, 1] = bank, lane
                bf[bank, 0] = anchor
                bf[bank, 3], bf[bank, 4] = _after(anchor, t[4], half), _after(anchor, t[5], half)
                bf[bank, 2] = _after(anchor, t[7], half)
                rows[bank] = row
                bf[bank, 6] = min(bf[bank, 3], bf[bank, 4])
                bf[bank, 5] = bf[bank, 2]
                tally[0] += 1
                stats[lane, 3] += 1
                if stale_deadline[ch]:
                    active_deadlines[ch] = _next_active_deadline(ch, g, t, rows, bf, addresses)
                    stale_deadline[ch] = 0
                elif t[29] > 0:
                    active_deadlines[ch] = min(active_deadlines[ch], _after(anchor, t[29], half))
                early[pseudo, bank_within] = 2 if early[pseudo, bank_within] == 1 else 3
                act_floor[pseudo] = 0.
                for side in range(2):
                    hits_now = 0
                    position = head[pseudo, side]
                    for _ in range(count[pseudo, side]):
                        if q_bank[pseudo, side, position] == bank and q_row[pseudo, side, position] == row:
                            hits_now += 1
                        position += 1
                        if position == q[side]:
                            position = 0
                    pending_hit[pseudo, side, bank_within] = hits_now
                    dirty[pseudo, side] = True
                stale[pseudo] = True
                last_pseudo = pseudo
                continue
            # _command on the column bus; a rising edge on the command clock either way.
            command = _align(max(start, resources[column_bus, 0]), t[0])
            command_end = _fr_after(command, column_duration, column_duration_tick, half, inverse)
            resources[column_bus, 0] = command_end
            resources[column_bus, 1] += column_duration
            resources[column_bus, 2], resources[column_bus, 3] = command, command_end
            touched[column_bus] = True
            tally[6] += command_end - command
            tally[2 if read else 3] += 1
            history[lane, history_entry, 1 if read else 2] = command
            data = 2 * channels + lane
            if directions[data] and directions[data] != (1 if read else 2):
                stats[lane, 5] += 1
            data_start = max(_fr_after(command, t[22 if read else 23], ticks[22 if read else 23], half, inverse),
                             resources[data, 0])
            data_end = _fr_after(data_start, t[26], ticks[26], half, inverse)
            resources[data, 0] = data_end
            resources[data, 1] += t[26]
            resources[data, 2], resources[data, 3] = data_start, data_end
            moved[data] += g[9]
            directions[data] = 1 if read else 2
            touched[data] = True
            pc[lane, 2 if read else 3] = _fr_after(data_start, t[27 if read else 28], ticks[27 if read else 28],
                                                   half, inverse)
            if read:
                bf[bank, 2] = max(bf[bank, 2], _fr_after(command, t[9], ticks[9], half, inverse))
                reads_since[pseudo] += 1
                # Waits only count time another request's traffic held this lane.
                own = own_last[request, pseudo]
                if own >= 0:
                    if lane_request[pseudo] != request:
                        own_wait[request, pseudo] += max(0., command - max(entry, own + t[12]))
                elif lane_request[pseudo] >= 0:
                    own_wait[request, pseudo] += max(0., command - entry)
                else:
                    own_wait[request, pseudo] += max(0., min(command, traffic_floor[pseudo]) - entry)
                own_last[request, pseudo] = command
                lane_request[pseudo] = request
            else:
                bf[bank, 2] = max(bf[bank, 2], _fr_after(data_end, t[10], ticks[10], half, inverse))
                history[lane, history_entry, 3] = data_end
            bf[bank, 5] = bf[bank, 2]
            ch_float[request, 0] = max(ch_float[request, 0], data_end)
            outcome = early[pseudo, bank_within]
            early[pseudo, bank_within] = 0
            # Early tags: 0 row hit, 2 opened after a conflict, 3 opened from closed.
            outcome = 0 if outcome == 0 else 2 if outcome == 2 else 1
            ch_int[request, outcome] += 1
            stats[lane, outcome] += 1
            pending_hit[pseudo, direction, bank_within] -= 1
            # Remove the entry, keeping the others in age order.
            first_slot = head[pseudo, direction]
            position = slot
            for _ in range(chosen):
                source = position - 1 if position else capacity - 1
                q_bank[pseudo, direction, position] = q_bank[pseudo, direction, source]
                q_row[pseudo, direction, position] = q_row[pseudo, direction, source]
                q_request[pseudo, direction, position] = q_request[pseudo, direction, source]
                q_entry[pseudo, direction, position] = q_entry[pseudo, direction, source]
                position = source
            head[pseudo, direction] = first_slot + 1 if first_slot + 1 < capacity else 0
            count[pseudo, direction] -= 1
            bypass[pseudo, direction] = bypass[pseudo, direction] + 1 if chosen else 0
            slot_free[pseudo, direction] = command
            last_column[pseudo], last_group[pseudo] = command, group
            stale[pseudo] = True
            last_pseudo = pseudo
        for pseudo in range(2):
            lane = 2 * ch + pseudo
            for request in range(requests):
                pending_wait = own_wait[request, pseudo]
                if pending_wait > ch_float[request, 1]:
                    ch_float[request, 1] = pending_wait
                if pending_wait > stats[lane, 9]:
                    stats[lane, 9] = pending_wait
        if mirror:
            partial_busy[0], partial_busy[1] = resources[row_bus, 1], resources[column_bus, 1]
            for pseudo in range(2):
                lane = 2 * ch + pseudo
                data = 2 * channels + lane
                partial_busy[2 + pseudo] = resources[data, 1]
                partial_moved[pseudo] = moved[data]
                moved[data] = saved_moved[pseudo]
                partial_stats[pseudo] = stats[lane]
                stats[lane] = saved_stats[pseudo]
            if capture >= 0:
                reach = max(reach, resources[row_bus, 0], resources[column_bus, 0])
                _fr_capture_after(capture, ch, not deadline_reached, reach_open if refreshed else reach, reach, g, rows, bf, addresses, pc, history, channel,
                                  channel_int, resources, directions, windows, window_counts, pc_seen, touched,
                                  partial_moved, partial_stats, ch_int, ch_float, tally, sources)
            _fr_merge_tallies(ch, g, saved_busy, partial_busy, partial_moved, partial_stats, ch_int, ch_float, tally,
                              resources, moved, stats, out_int, out_float, channel_counts[ch - channel_first])


def run_fr_fcfs(firsts, stops, arrivals, reads, q, g, t, rows, bf, addresses, pc, history, pc_seen,
                channel, channel_int, resources, moved, directions, touched, windows, window_counts,
                stats):
    """Run one batch, one worker per channel range; returns per-request outcomes.

    Workers own disjoint channels and their own output rows. The outcome rows
    are reduced in channel order, so results do not depend on thread timing.
    """
    lanes, channels = int(g[0]), int(g[0]) // 2
    channel_first, channel_stop = channels, 0
    for first, stop in zip(firsts.tolist(), stops.tolist()):
        first_block, last_block = first >> 1, (stop - 1) >> 1
        if first_block // lanes != last_block // lanes:
            channel_first, channel_stop = 0, channels
            break
        channel_first = min(channel_first, (first_block % lanes) // 2)
        channel_stop = max(channel_stop, (last_block % lanes) // 2 + 1)
    arrays = (rows, bf, addresses, pc, history, pc_seen, channel, channel_int, resources, moved,
              directions, touched, windows, window_counts, stats)
    requests = len(firsts)

    span = channel_stop - channel_first

    def execute(left, right):
        out_int = np.zeros((requests, 3), dtype=np.int64)
        out_float = np.zeros((requests, 2), dtype=np.float64)
        out_float[:, 0] = arrivals
        channel_counts = np.zeros((max(1, right - left), 7), dtype=np.float64)
        _run_fr_fcfs_channels(firsts, stops, arrivals, reads, q, g, t, *arrays,
                              out_int, out_float, left, right, span, channel_counts)
        return out_int, out_float, channel_counts

    def reduced(channel_counts):
        counts = np.zeros(7, dtype=np.float64)
        for row in channel_counts:
            counts += row
        return counts

    physical_bytes = int((stops - firsts).sum()) * int(g[9])
    bursts = int((stops - firsts).sum())
    workers = min(_CHANNEL_WORKERS, span, physical_bytes // (32 * 1024))
    clock = float(t[0])
    if (physical_bytes < 128 * 1024 or workers < 2
            or (clock * 2 != math.floor(clock * 2) and bursts < FR_MIRROR_MIN_BURSTS * span)):
        out_int, out_float, channel_counts = execute(channel_first, channel_stop)
        return out_int, out_float, reduced(channel_counts)
    # Compile or load the kernel on the calling thread before workers start.
    execute(0, 0)
    condition = Condition()
    cancelled = False
    active = 0

    def guarded(left, right):
        nonlocal active
        with condition:
            if cancelled:
                return None
            active += 1
        try:
            return execute(left, right)
        finally:
            with condition:
                active -= 1
                condition.notify_all()

    jobs = []
    try:
        for index in range(workers):
            jobs.append(_CHANNEL_EXECUTOR.submit(guarded,
                channel_first + index * (channel_stop - channel_first) // workers,
                channel_first + (index + 1) * (channel_stop - channel_first) // workers))
        wait(jobs)
        results = [job.result() for job in jobs]
    except BaseException:
        with condition:
            cancelled = True
            while active:
                try:
                    condition.wait()
                except BaseException:
                    continue
        raise
    out_int = sum(result[0] for result in results)
    out_float = np.stack([result[1] for result in results]).max(axis=0)
    return out_int, out_float, reduced(np.concatenate([result[2] for result in results]))
