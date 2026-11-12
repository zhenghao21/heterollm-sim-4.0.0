/* The HBM per-PC FR-FCFS command kernel; see _hbm_fr_core.c. */
#ifndef HBM_FR_CORE_H
#define HBM_FR_CORE_H

#include <stdint.h>

/* A batch mirrors channels when it averages this many bursts per channel. */
#define FR_MIRROR_MIN_BURSTS 64

enum {
    FR_OK = 0,
    FR_ERR_CLOCK_RANGE,
    FR_ERR_ACT_CALENDAR,
    FR_ERR_TRAS_PRECHARGE,
    FR_ERR_TRAS_REFRESH,
    FR_ERR_REFRESH_POSTPONED,
    FR_ERR_RECOVERY_WINDOW,
    FR_ERR_NO_COMMAND,
    FR_ERR_ZERO_DIVISION,
    FR_ERR_MEMORY
};

/* Flat C-order views of the packed controller arrays; bool arrays are int8. */
typedef struct fr_args {
    const int64_t *firsts, *stops;
    const double *arrivals;
    const int8_t *reads;
    const int64_t *q, *g;
    const double *t;
    int64_t *rows;
    double *bf;
    int64_t *addresses;
    double *pc, *history;
    int8_t *pc_seen;
    double *channel;
    int64_t *channel_int;
    double *resources;
    int64_t *moved;
    int8_t *directions, *touched;
    double *windows;
    int64_t *window_counts;
    double *stats;
    int64_t *out_int;
    double *out_float, *channel_counts;
    int64_t requests, entries, window_capacity, stat_fields;
    int64_t channel_first, channel_stop, span, stripe_shift;
    unsigned char *audit_history;
    int64_t audit_stride;
} fr_args;

/* Runs without the GIL; returns FR_OK or the error for the caller to raise. */
int fr_run_channels(const fr_args *args);

#include <stdio.h>
#include <stddef.h>
size_t fr_audit_image_size(int64_t banks,int64_t bgs,int64_t sids);
FILE *fr_audit_take_file(int summary);

#endif
