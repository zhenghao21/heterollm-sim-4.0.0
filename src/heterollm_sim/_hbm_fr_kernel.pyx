# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Python entry of the HBM FR-FCFS command kernel; the kernel itself is _hbm_fr_core.c."""
from libc.stdint cimport int8_t, int64_t

import numpy as np


cdef extern from "_hbm_fr_core.h" nogil:
    enum:
        _MIRROR_MIN_BURSTS "FR_MIRROR_MIN_BURSTS"
        FR_OK
        FR_ERR_CLOCK_RANGE
        FR_ERR_ACT_CALENDAR
        FR_ERR_TRAS_PRECHARGE
        FR_ERR_TRAS_REFRESH
        FR_ERR_REFRESH_POSTPONED
        FR_ERR_RECOVERY_WINDOW
        FR_ERR_NO_COMMAND
        FR_ERR_ZERO_DIVISION
        FR_ERR_MEMORY

    ctypedef struct fr_args:
        const int64_t *firsts
        const int64_t *stops
        const double *arrivals
        const int8_t *reads
        const int64_t *q
        const int64_t *g
        const double *t
        int64_t *rows
        double *bf
        int64_t *addresses
        double *pc
        double *history
        int8_t *pc_seen
        double *channel
        int64_t *channel_int
        double *resources
        int64_t *moved
        int8_t *directions
        int8_t *touched
        double *windows
        int64_t *window_counts
        double *stats
        int64_t *out_int
        double *out_float
        double *channel_counts
        int64_t requests
        int64_t entries
        int64_t window_capacity
        int64_t stat_fields
        int64_t channel_first
        int64_t channel_stop
        int64_t span
        int64_t stripe_shift

    int fr_run_channels(const fr_args *args)


FR_MIRROR_MIN_BURSTS = _MIRROR_MIN_BURSTS


cdef flags(array):
    """A typed memoryview refuses NumPy's bool format, so a bool array is viewed as int8."""
    if array.dtype != np.bool_:
        raise TypeError("HBM kernel flags must be a bool array")
    return array.view(np.int8)


cdef raise_failure(int code):
    if code == FR_ERR_CLOCK_RANGE:
        raise OverflowError("HBM clock index exceeds compiled exact-integer range")
    if code == FR_ERR_ACT_CALENDAR:
        raise OverflowError("HBM ACT calendar exceeds packed capacity")
    if code == FR_ERR_TRAS_PRECHARGE:
        raise ValueError("HBM tRASmax deadline cannot be met by a legal PRE command")
    if code == FR_ERR_TRAS_REFRESH:
        raise ValueError("HBM tRASmax deadline cannot be met by the refresh PRE command")
    if code == FR_ERR_REFRESH_POSTPONED:
        raise ValueError("HBM periodic refresh missed the maximum eight postponed intervals")
    if code == FR_ERR_RECOVERY_WINDOW:
        raise ValueError("HBM access recovery exceeds the tRASmax activation window")
    if code == FR_ERR_NO_COMMAND:
        raise ValueError("HBM scheduler has queued work but no eligible command")
    if code == FR_ERR_ZERO_DIVISION:
        raise ZeroDivisionError("division by zero")
    if code == FR_ERR_MEMORY:
        raise MemoryError()


def _run_fr_fcfs_channels(firsts, stops, arrivals, reads, q, g, t, rows, bf, addresses, pc, history, pc_seen,
                          channel, channel_int, resources, moved, directions, touched, windows, window_counts,
                          stats, out_int, out_float, Py_ssize_t channel_first, Py_ssize_t channel_stop,
                          Py_ssize_t span, channel_counts):
    """Issue one batch's bursts command by command through per-PC queues, for channels in a range."""
    cdef const int64_t[::1] view_firsts = firsts, view_stops = stops, view_q = q, view_g = g
    cdef const double[::1] view_arrivals = arrivals, view_t = t
    cdef const int8_t[::1] view_reads = flags(reads)
    cdef int64_t[::1] view_rows = rows, view_addresses = addresses, view_moved = moved, view_window_counts = window_counts
    cdef int8_t[::1] view_pc_seen = flags(pc_seen), view_directions = directions, view_touched = flags(touched)
    cdef double[:, ::1] view_bf = bf, view_pc = pc, view_channel = channel, view_resources = resources
    cdef double[:, ::1] view_windows = windows, view_stats = stats, view_out_float = out_float
    cdef double[:, ::1] view_counts = channel_counts
    cdef int64_t[:, ::1] view_channel_int = channel_int, view_out_int = out_int
    cdef double[:, :, ::1] view_history = history
    cdef fr_args args
    cdef int code
    if (view_t.shape[0] < 34 or view_q.shape[0] < 5 or view_g.shape[0] < 16 or view_bf.shape[1] != 7
            or view_pc.shape[1] != 7 or view_history.shape[2] != 4 or view_channel.shape[1] != 2
            or view_channel_int.shape[1] != 2 or view_resources.shape[1] != 4 or view_stats.shape[1] < 10
            or view_out_int.shape[1] != 3 or view_out_float.shape[1] != 2 or view_counts.shape[1] != 7):
        raise ValueError("HBM kernel arrays do not have the packed controller's layout")
    if view_g[15] < 2 or view_g[15] & (view_g[15] - 1):
        raise ValueError("HBM kernel needs a power-of-two stripe of bank groups")
    args.firsts = &view_firsts[0]
    args.stops = &view_stops[0]
    args.arrivals = &view_arrivals[0]
    args.reads = &view_reads[0]
    args.q = &view_q[0]
    args.g = &view_g[0]
    args.t = &view_t[0]
    args.rows = &view_rows[0]
    args.bf = &view_bf[0, 0]
    args.addresses = &view_addresses[0]
    args.pc = &view_pc[0, 0]
    args.history = &view_history[0, 0, 0]
    args.pc_seen = &view_pc_seen[0]
    args.channel = &view_channel[0, 0]
    args.channel_int = &view_channel_int[0, 0]
    args.resources = &view_resources[0, 0]
    args.moved = &view_moved[0]
    args.directions = &view_directions[0]
    args.touched = &view_touched[0]
    args.windows = &view_windows[0, 0]
    args.window_counts = &view_window_counts[0]
    args.stats = &view_stats[0, 0]
    args.out_int = &view_out_int[0, 0]
    args.out_float = &view_out_float[0, 0]
    args.channel_counts = &view_counts[0, 0]
    args.requests = view_firsts.shape[0]
    args.entries = view_history.shape[1]
    args.window_capacity = view_windows.shape[1]
    args.stat_fields = view_stats.shape[1]
    args.channel_first = channel_first
    args.channel_stop = channel_stop
    args.span = span
    args.stripe_shift = 0
    while (view_g[15] >> args.stripe_shift) > 1:
        args.stripe_shift += 1
    with nogil:
        code = fr_run_channels(&args)
    if code != FR_OK:
        raise_failure(code)
