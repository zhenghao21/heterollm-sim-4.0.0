"""Dispatch of the JEDEC HBM per-PC FR-FCFS controller over its channel workers.

The command kernel is C, in _hbm_fr_core.c; _hbm_control is its reference.
Physical CA ports, PC data pipes, SID histories and refresh remain distinct.
No fast-math or parallel reductions are used.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Condition
import numpy as np

try:
    from ._hbm_fr_kernel import FR_MIRROR_MIN_BURSTS, _run_fr_fcfs_channels
except ImportError as error:
    # dram_core takes an ImportError from here to mean no compiled HBM controller,
    # and would go on through the reference controller at a fraction of the speed.
    raise RuntimeError("The HBM FR-FCFS kernel is not built: run "
                       "scripts/build_resource_ledger.py --module _hbm_fr_kernel --inplace") from error

# A manager running several simulator processes caps each process's channel
# threads so that together they do not oversubscribe the host cores.
_CHANNEL_WORKERS = max(1, min(8, int(os.environ.get("HETEROLLM_HBM_CHANNEL_THREADS") or os.cpu_count() or 1)))
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


# Per-PC scheduler counters kept with the controller state, one row per lane.
FR_STAT_FIELDS = ("row_hits", "row_misses", "row_conflicts", "activate_commands",
                  "precharge_commands", "direction_switches", "write_drains",
                  "peak_read_queue", "peak_write_queue", "max_read_wait_ns")


def run_fr_fcfs(firsts, stops, arrivals, reads, q, g, t, rows, bf, addresses, pc, history, pc_seen,
                channel, channel_int, resources, moved, directions, touched, windows, window_counts,
                stats):
    """Run one batch, one worker per channel range; returns per-request outcomes.

    Workers own disjoint channels and their own output rows. The outcome rows
    are reduced in channel order, so results do not depend on thread timing.
    """
    lanes, channels, stripe = int(g[0]), int(g[0]) // 2, int(g[15])
    channel_first, channel_stop = channels, 0
    for first, stop in zip(firsts.tolist(), stops.tolist()):
        first_block, last_block = first // stripe, (stop - 1) // stripe
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
