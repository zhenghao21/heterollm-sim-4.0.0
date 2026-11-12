/* The HBM per-PC FR-FCFS command kernel.
 *
 * Each physical channel owns both PCs, their banks and its CA buses, so one
 * worker can finish a channel alone. Build with strict IEEE doubles: no
 * fast-math and no contraction. max/min keep Python's operand order, which
 * decides the sign of an equal zero; round is nearest-even, floor truncates
 * towards minus infinity, and both give int64 as the clock indices require.
 */
#include "_hbm_fr_core.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>

#if defined(_MSC_VER)
#include <intrin.h>
#include <windows.h>
#define FR_INLINE static __forceinline
#define FR_NOINLINE static __declspec(noinline)
#define FR_NORETURN __declspec(noreturn)
#define FR_THREAD __declspec(thread)
#else
#define FR_INLINE static inline __attribute__((always_inline))
#define FR_NOINLINE static __attribute__((noinline))
#define FR_NORETURN __attribute__((noreturn))
#define FR_THREAD _Thread_local
#endif

#define FR_TWO49 562949953421312.0
#define FR_TWO50 1125899906842624.0
#define FR_TWO52 4503599627370496.0
#define FR_TWO48 281474976710656.0
#define FR_TWO44 17592186044416.0
/* Clock delay ticks the kernel precomputes. A finite delay off the half-clock grid
 * has none and always adds plainly; one too large for the fast form has none either. */
#define FR_NO_TICK (-(INT64_C(1) << 62))
#define FR_OFF_GRID (-(INT64_C(1) << 61))
/* A batch keeps at most this many simulated channels to mirror from. */
#define FR_MIRROR_SOURCES 128
#define FR_FIELDS 34
/* A PC's columns: last PRE, blocked until, next read and write data, then its
 * per-bank refresh: the tRREFD gate, the next due time, the tRFCpb gate after
 * the last REFpb completing a 16-bank SID set, and the banks of the current
 * set already refreshed (bit i is bank i of the SID). */
#define FR_PC 8
/* Below this a product with the SID count's inverse divides exactly. */
#define FR_SID_FAST (INT64_C(1) << 24)

static FR_THREAD jmp_buf *fr_escape;

/* Optional audit-only observer. It is completely inert unless the caller
 * supplies an audit path; each worker uses its own fragment to avoid locking
 * the numeric command path. */
static FR_THREAD FILE *fr_trace_file;
static FR_THREAD FILE *fr_trace_summary_file;
static FR_THREAD long long fr_trace_total;
static FR_THREAD long long fr_trace_counts[7];
static FR_THREAD long long fr_trace_limit;
static FR_THREAD double fr_trace_min_time, fr_trace_max_time;
/* The limit is process/thread-wide, because a fast simulation opens the
 * observer once for every small numeric batch.  Resetting it per batch would
 * still allow millions of sampled rows. */
static FR_THREAD long long fr_trace_emitted;

enum {
    AUDIT_BANK_OPEN, AUDIT_TRP, AUDIT_ROW_MATCH, AUDIT_REFRESH_BLOCK,
    AUDIT_PRE_CLOSED, AUDIT_TRC, AUDIT_TRAS, AUDIT_TRASMAX, AUDIT_TRCD,
    AUDIT_RTP, AUDIT_WR, AUDIT_TPPD, AUDIT_TRRD, AUDIT_TFAW, AUDIT_WTR,
    AUDIT_RTW, AUDIT_DQ, AUDIT_TCCDL, AUDIT_TCCDS, AUDIT_TCCDR,
    AUDIT_BUS_ROW, AUDIT_BUS_COL, AUDIT_REFAB_CLOSED, AUDIT_REFAB_TRP,
    AUDIT_REFAB_NOP, AUDIT_REFAB_INTERVAL, AUDIT_REF_CLOSED, AUDIT_REF_TRP,
    AUDIT_ACT_REF, AUDIT_REF_ACT, AUDIT_REF_SPACING, AUDIT_REFRESH_ORDER,
    AUDIT_REFRESH_GROUP_GAP, AUDIT_REFRESH_INTERVAL, AUDIT_REFRESH_RATE,
    AUDIT_RULE_COUNT
};

static const char *fr_audit_rule_names[AUDIT_RULE_COUNT] = {
    "1.bank_open", "1.tRP", "1.row_match", "7.refresh_block",
    "1.pre_closed", "2.tRC", "2.tRAS", "2.tRASmax", "2.tRCD",
    "2.tRTP", "2.tWR", "3.tPPD", "3.tRRD", "3.tFAW", "4.WTR",
    "4.RTW", "5.DQ_nonoverlap", "6.tCCDL", "6.tCCDS", "6.tCCDR",
    "bus.row_serial", "bus.col_serial", "6.refab_closed", "6.refab_tRP",
    "6.refab_nop", "6.refab_interval", "7.refresh_closed", "7.refresh_tRP",
    "8.act_to_refpb", "8.refpb_to_act", "9.refpb_spacing", "10.refresh_order",
    "10.refresh_group_gap", "11.refresh_interval", "11.refresh_rate"
};

typedef struct fr_audit_state {
    int initialized, lanes, banks, groups, bgs, sids, channels;
    int64_t token;
    int64_t *open_row;
    double *act_anchor, *pre_time, *last_rd, *last_wr_end, *blocked, *last_ref;
    double *last_pre, *last_rd_lane, *data_available;
    double *act_group, *act_group_top1, *act_group_top2;
    int64_t *act_group_bank1, *act_group_bank2;
    double *act_top1, *act_top2;
    int64_t *act_top_group1, *act_top_group2;
    double *ref_top1, *ref_top2;
    int64_t *ref_bank1, *ref_bank2;
    double *last_col, *last_col_sid_top1, *last_col_sid_top2, *last_col_lane_top1, *last_col_lane_top2, *write_bg;
    int64_t *last_col_sid_group1, *last_col_sid_group2, *last_col_lane_sid1, *last_col_lane_sid2;
    double *faw, *refresh_ring, *round_end, *pc_group_gate, *bank_ref_ring, *refresh_t0;
    uint8_t *faw_count, *refresh_count, *refresh_ring_count, *round_seen, *bank_ref_count;
    uint64_t *refresh_mask;
    long long *refresh_total;
    double *row_busy, *col_busy, *refab_blocked, *last_refab;
    int64_t *open_count;
    long long commands, by_type[7];
    long long delta_commands;
    long long checks[AUDIT_RULE_COUNT], violations[AUDIT_RULE_COUNT];
    long long delta_checks[AUDIT_RULE_COUNT], delta_violations[AUDIT_RULE_COUNT];
    uint8_t delta_first_seen[AUDIT_RULE_COUNT];
    char first_kind[AUDIT_RULE_COUNT][8];
    double first_time[AUDIT_RULE_COUNT];
    int64_t first_stack[AUDIT_RULE_COUNT], first_channel[AUDIT_RULE_COUNT], first_pc[AUDIT_RULE_COUNT];
    int64_t first_sid[AUDIT_RULE_COUNT], first_bg[AUDIT_RULE_COUNT], first_bank[AUDIT_RULE_COUNT], first_row[AUDIT_RULE_COUNT];
} fr_audit_state;

static FR_THREAD fr_audit_state fr_audit;
static FR_THREAD unsigned char *fr_audit_empty;

static void fr_audit_free(void);
static void fr_audit_image_transfer(unsigned char *image, int64_t ch, int writing);
static void fr_audit_recent_update(double *top1, double *top2, int64_t *key1, int64_t *key2, int64_t key, double time);

static void fr_audit_reset_delta(void)
{
    fr_audit.delta_commands = 0;
    memset(fr_audit.delta_checks, 0, sizeof(fr_audit.delta_checks));
    memset(fr_audit.delta_violations, 0, sizeof(fr_audit.delta_violations));
    memset(fr_audit.delta_first_seen, 0, sizeof(fr_audit.delta_first_seen));
}

static int fr_trace_kind_index(const char *kind)
{
    if (strcmp(kind, "ACT") == 0) return 0;
    if (strcmp(kind, "PRE") == 0) return 1;
    if (strcmp(kind, "PRE_ALL") == 0) return 2;
    if (strcmp(kind, "REFab") == 0) return 3;
    if (strcmp(kind, "REFpb") == 0) return 4;
    if (strcmp(kind, "RD") == 0) return 5;
    if (strcmp(kind, "WR") == 0) return 6;
    return -1;
}

static void fr_trace_open(void)
{
    const char *path = getenv("HETEROLLM_HBM_TRACE_FILE");
    const char *summary = getenv("HETEROLLM_HBM_TRACE_SUMMARY_FILE");
    const char *limit = getenv("HETEROLLM_HBM_TRACE_LIMIT");
    const char *minimum = getenv("HETEROLLM_HBM_TRACE_MIN_NS");
    const char *maximum = getenv("HETEROLLM_HBM_TRACE_MAX_NS");
    char trace_path[4096], summary_path[4096];
    const char *trace_name = path;
    const char *summary_name = summary;
#if defined(_MSC_VER)
    unsigned long thread_id = (unsigned long)GetCurrentThreadId();
    if (path != NULL && path[0] != '\0') {
        _snprintf_s(trace_path, sizeof(trace_path), _TRUNCATE, "%s.thread_%lu", path, thread_id);
        trace_name = trace_path;
    }
    if (summary != NULL && summary[0] != '\0') {
        _snprintf_s(summary_path, sizeof(summary_path), _TRUNCATE, "%s.thread_%lu", summary, thread_id);
        summary_name = summary_path;
    }
#endif
    fr_trace_total = 0;
    memset(fr_trace_counts, 0, sizeof(fr_trace_counts));
    fr_trace_min_time = (minimum != NULL && minimum[0] != '\0') ? strtod(minimum, NULL) : -DBL_MAX;
    fr_trace_max_time = (maximum != NULL && maximum[0] != '\0') ? strtod(maximum, NULL) : DBL_MAX;
    if (!isfinite(fr_trace_min_time)) fr_trace_min_time = -DBL_MAX;
    if (!isfinite(fr_trace_max_time)) fr_trace_max_time = DBL_MAX;
    /* History belongs to the component's channel image, not this worker. */
    fr_audit_reset_delta();
    memset(fr_audit.first_kind,0,sizeof(fr_audit.first_kind));
    fr_trace_limit = (limit != NULL && limit[0] != '\0') ? strtoll(limit, NULL, 10) : LLONG_MAX;
    if (fr_trace_limit < 0)
        fr_trace_limit = LLONG_MAX;
    fr_trace_file = (trace_name != NULL && trace_name[0] != '\0') ? tmpfile() : NULL;
    fr_trace_summary_file = (summary_name != NULL && summary_name[0] != '\0') ? tmpfile() : NULL;
    if (fr_trace_file != NULL)
        setvbuf(fr_trace_file, NULL, _IOFBF, 1 << 20);
    if (fr_trace_summary_file != NULL)
        setvbuf(fr_trace_summary_file, NULL, _IOFBF, 1 << 20);
    if (fr_trace_file != NULL)
        fputs("#BEGIN\n", fr_trace_file);
}

static void fr_trace_close(void)
{
    if (fr_trace_file != NULL) {
        fputs("#END\n", fr_trace_file);
        fflush(fr_trace_file);

    }
    if (fr_trace_summary_file != NULL) {
        fprintf(fr_trace_summary_file,
                "SUMMARY\ttotal=%lld\tACT=%lld\tPRE=%lld\tPRE_ALL=%lld\tREFab=%lld\tREFpb=%lld\tRD=%lld\tWR=%lld\n",
                fr_trace_total, fr_trace_counts[0], fr_trace_counts[1], fr_trace_counts[2],
                fr_trace_counts[3], fr_trace_counts[4], fr_trace_counts[5], fr_trace_counts[6]);
        if (fr_audit.initialized) {
            int rule;
            fprintf(fr_trace_summary_file, "AUDIT_DELTA\tcommands=%lld\tchecks=", fr_audit.delta_commands);
            for (rule = 0; rule < AUDIT_RULE_COUNT; ++rule) fprintf(fr_trace_summary_file, "%s%lld", rule ? "," : "", fr_audit.delta_checks[rule]);
            fputs("\tviolations=", fr_trace_summary_file);
            for (rule = 0; rule < AUDIT_RULE_COUNT; ++rule) fprintf(fr_trace_summary_file, "%s%lld", rule ? "," : "", fr_audit.delta_violations[rule]);
            fputs("\tfirst=", fr_trace_summary_file);
            for (rule = 0; rule < AUDIT_RULE_COUNT; ++rule) if (fr_audit.delta_first_seen[rule]) {
                if (rule) fputc(';', fr_trace_summary_file);
                fprintf(fr_trace_summary_file, "%d,%s,%.17g,%lld,%lld,%lld,%lld,%lld,%lld,%lld", rule, fr_audit.first_kind[rule], fr_audit.first_time[rule],
                        (long long)fr_audit.first_stack[rule], (long long)fr_audit.first_channel[rule], (long long)fr_audit.first_pc[rule],
                        (long long)fr_audit.first_sid[rule], (long long)fr_audit.first_bg[rule], (long long)fr_audit.first_bank[rule], (long long)fr_audit.first_row[rule]);
            }
            fputc('\n', fr_trace_summary_file);
        }
        fflush(fr_trace_summary_file);

    }
}

FR_NORETURN FR_NOINLINE void fr_fail(int code)
{
    longjmp(*fr_escape, code);
}

FR_INLINE double fr_abs(double value)
{
    return fabs(value);
}

FR_INLINE int fr_isfinite(double value)
{
    return fr_abs(value) <= DBL_MAX;
}

/* max(accumulator, value) in Python's order: maxsd returns its second operand on a tie or NaN. */
FR_INLINE double fr_max(double accumulator, double value)
{
    return _mm_cvtsd_f64(_mm_max_sd(_mm_set_sd(value), _mm_set_sd(accumulator)));
}

FR_INLINE double fr_min(double accumulator, double value)
{
    return _mm_cvtsd_f64(_mm_min_sd(_mm_set_sd(value), _mm_set_sd(accumulator)));
}

FR_INLINE int64_t fr_round(double value)
{
    return _mm_cvtsd_si64(_mm_set_sd(value));
}

/* The floor of a ratio already bounded below 2**63 in magnitude; the x86 result on NaN. */
FR_INLINE int64_t fr_floor(double value)
{
    int64_t tick = (int64_t)value;
    return tick - ((double)tick > value);
}

/* The floor of any value, outside the burst path. */
FR_INLINE int64_t fr_ifloor(double value)
{
    return (int64_t)floor(value);
}

FR_INLINE double fr_divide(double value, double divisor)
{
    if (divisor == 0.)
        fr_fail(FR_ERR_ZERO_DIVISION);
    return value / divisor;
}

FR_INLINE int64_t fr_floordiv(int64_t value, int64_t divisor)
{
    int64_t quotient;
    if (divisor == 0)
        fr_fail(FR_ERR_ZERO_DIVISION);
    quotient = value / divisor;
    return quotient - ((value % divisor != 0) && ((value < 0) != (divisor < 0)));
}

FR_INLINE int64_t fr_mod(int64_t value, int64_t divisor)
{
    int64_t remainder;
    if (divisor == 0)
        fr_fail(FR_ERR_ZERO_DIVISION);
    remainder = value % divisor;
    return remainder + ((remainder != 0 && ((remainder < 0) != (divisor < 0))) ? divisor : 0);
}

FR_INLINE int64_t fr_imax(int64_t left, int64_t right)
{
    return right > left ? right : left;
}

FR_INLINE int64_t fr_imin(int64_t left, int64_t right)
{
    return right < left ? right : left;
}

FR_INLINE double fr_align(double value, double quantum)
{
    /* Decline before the conversion leaves the range where adjacent integer
     * ticks remain distinguishable. */
    double ratio = value / quantum;
    int64_t tick;
    double edge;
    if (fr_abs(ratio) >= FR_TWO52)
        fr_fail(FR_ERR_CLOCK_RANGE);
    tick = fr_floor(ratio);
    edge = (double)tick * quantum;
    if (edge < value)
        edge = (double)(tick + 1) * quantum;
    return edge;
}

/* clock_after with its two divisions shared. */
static double fr_after(double value, double delay, double quantum)
{
    if (fr_isfinite(value) && fr_isfinite(delay)) {
        double value_ratio = value / quantum, delay_ratio = delay / quantum;
        int64_t value_tick, delay_tick;
        if (fr_abs(value_ratio) >= FR_TWO52 || fr_abs(delay_ratio) >= FR_TWO52)
            fr_fail(FR_ERR_CLOCK_RANGE);
        value_tick = fr_round(value_ratio);
        delay_tick = fr_round(delay_ratio);
        if (value == (double)value_tick * quantum && delay == (double)delay_tick * quantum)
            return (double)(value_tick + delay_tick) * quantum;
    } else if ((fr_isfinite(value) && fr_abs(value / quantum) >= FR_TWO52)
               || (fr_isfinite(delay) && fr_abs(delay / quantum) >= FR_TWO52)) {
        fr_fail(FR_ERR_CLOCK_RANGE);
    }
    return value + delay;
}

/* fr_after for a precomputed delay tick, without its divisions.
 *
 * Below 2**50 ticks a represented grid value lies within a few ulps of its
 * integer under either form of the ratio, so both round to the same tick; a
 * value off the grid fails the membership test under both, and so does any
 * value with a delay off the grid.
 */
FR_INLINE double fr_after_tick(double value, double delay, int64_t delay_tick, double half, double inverse)
{
    double ratio = value * inverse;
    if (delay_tick != FR_NO_TICK && fr_abs(ratio) < FR_TWO50) {
        if (delay_tick != FR_OFF_GRID) {
            int64_t tick = fr_round(ratio);
            if (value == (double)tick * half)
                return (double)(tick + delay_tick) * half;
        }
        return value + delay;
    }
    return fr_after(value, delay, half);
}

FR_INLINE int64_t fr_tick(double delay, double half)
{
    double ratio = delay / half;
    int64_t tick = fr_round(ratio);
    if (!(fr_abs(ratio) < FR_TWO50))
        return FR_NO_TICK;
    return delay == (double)tick * half ? tick : FR_OFF_GRID;
}

FR_INLINE int64_t fr_negative_tick(int64_t tick)
{
    return tick == FR_NO_TICK || tick == FR_OFF_GRID ? tick : -tick;
}

/* Everything a helper reads: the controller arrays and the tables derived once per call. */
typedef struct fr_state {
    const fr_args *a;
    const int64_t *g, *q;
    const double *t;
    int64_t *rows, *addresses, *channel_int, *window_counts;
    double *bf, *pc, *history, *channel, *resources, *windows;
    int8_t *pc_seen, *touched;
    int64_t banks, channels, entries, window_capacity;
    double half, inverse, grid_limit, precharge_step, end_gap;
    const int64_t *entry_group, *entry_sid, *read_order, *write_order, *read_splits, *write_splits;
    double act_duration, pre_duration;
    int64_t ticks[FR_FIELDS], follow_tick, clock_back, act_duration_tick, pre_duration_tick;
    int64_t *close_banks, *close_order;
    double *close_deadlines, *close_starts;
    int grouped;
    double *act_gates, *act_floors;
    uint8_t *act_gate_ready, *act_floor_ready;
    uint64_t *miss_seen;
} fr_state;

static void fr_audit_free(void)
{
#define FR_AUDIT_FREE(name) do { free(fr_audit.name); fr_audit.name = NULL; } while (0)
    FR_AUDIT_FREE(open_row); FR_AUDIT_FREE(act_anchor); FR_AUDIT_FREE(pre_time);
    FR_AUDIT_FREE(last_rd); FR_AUDIT_FREE(last_wr_end); FR_AUDIT_FREE(blocked); FR_AUDIT_FREE(last_ref);
    FR_AUDIT_FREE(last_pre); FR_AUDIT_FREE(last_rd_lane); FR_AUDIT_FREE(data_available);
    FR_AUDIT_FREE(act_group); FR_AUDIT_FREE(act_group_top1); FR_AUDIT_FREE(act_group_top2);
    FR_AUDIT_FREE(act_group_bank1); FR_AUDIT_FREE(act_group_bank2);
    FR_AUDIT_FREE(act_top1); FR_AUDIT_FREE(act_top2); FR_AUDIT_FREE(act_top_group1); FR_AUDIT_FREE(act_top_group2);
    FR_AUDIT_FREE(ref_top1); FR_AUDIT_FREE(ref_top2); FR_AUDIT_FREE(ref_bank1); FR_AUDIT_FREE(ref_bank2);
    FR_AUDIT_FREE(last_col); FR_AUDIT_FREE(last_col_sid_top1); FR_AUDIT_FREE(last_col_sid_top2);
    FR_AUDIT_FREE(last_col_sid_group1); FR_AUDIT_FREE(last_col_sid_group2);
    FR_AUDIT_FREE(last_col_lane_top1); FR_AUDIT_FREE(last_col_lane_top2);
    FR_AUDIT_FREE(last_col_lane_sid1); FR_AUDIT_FREE(last_col_lane_sid2);
    FR_AUDIT_FREE(write_bg); FR_AUDIT_FREE(faw); FR_AUDIT_FREE(refresh_ring); FR_AUDIT_FREE(round_end);
    FR_AUDIT_FREE(pc_group_gate); FR_AUDIT_FREE(bank_ref_ring); FR_AUDIT_FREE(refresh_t0);
    FR_AUDIT_FREE(faw_count); FR_AUDIT_FREE(refresh_count); FR_AUDIT_FREE(refresh_ring_count); FR_AUDIT_FREE(round_seen); FR_AUDIT_FREE(bank_ref_count); FR_AUDIT_FREE(refresh_mask);
    FR_AUDIT_FREE(refresh_total);
    FR_AUDIT_FREE(row_busy); FR_AUDIT_FREE(col_busy); FR_AUDIT_FREE(refab_blocked); FR_AUDIT_FREE(last_refab); FR_AUDIT_FREE(open_count);
#undef FR_AUDIT_FREE
    free(fr_audit_empty); fr_audit_empty = NULL;
    memset(&fr_audit, 0, sizeof(fr_audit));
}

static int fr_audit_alloc(void **out, size_t count, size_t size)
{
    *out = calloc(count ? count : 1, size);
    return *out != NULL;
}

static void fr_audit_fill_double(double *values, size_t count, double value)
{
    size_t index;
    for (index = 0; index < count; ++index)
        values[index] = value;
}

static void fr_audit_seed(const fr_state *s, int64_t first_channel, int64_t last_channel)
{
    int64_t lane, bank, sid, group, ch, index;
    int64_t lanes = s->g[0], banks = s->banks, groups = s->g[5], sids = s->g[14];
    if (first_channel < 0) first_channel = 0;
    if (last_channel > s->channels) last_channel = s->channels;
    for (lane = 2 * first_channel; lane < 2 * last_channel; ++lane) {
        int64_t bank_open = 0;
        double data_available = -DBL_MAX, last_rd = -DBL_MAX;
        fr_audit.last_pre[lane] = s->pc[lane * FR_PC] >= 0 ? s->pc[lane * FR_PC] : -DBL_MAX;
        if (s->pc[lane * FR_PC + 6] > 0)
            fr_audit.pc_group_gate[lane] = s->pc[lane * FR_PC + 6];

        /* The numeric state may enter this observer between two REFpb
         * commands.  Seed the current SID's partial 16-bank group from the
         * controller's next-refresh cursor instead of pretending that the
         * first command of this observer is the group's first command. */
        if (s->t[33] > 0 && s->pc[lane * FR_PC + 5] >= 0) {
            int64_t next = fr_mod(fr_round(s->pc[lane * FR_PC + 5] / s->t[33]), banks);
            int64_t next_sid = next / 16, position = next % 16;
            int64_t sid_item = lane * sids + next_sid;
            if (position > 0 && next_sid >= 0 && next_sid < sids) {
                fr_audit.refresh_count[sid_item] = (uint8_t)position;
                fr_audit.refresh_mask[sid_item] = position == 64 ? UINT64_MAX : ((UINT64_C(1) << position) - 1);
            }
            /* The bank ready frontier is the completed REFpb edge plus
             * tRFCpb.  It supplies the same-SID boundary when this stream
             * starts at the first bank of a later group. */
            for (sid = 0; sid < sids; ++sid) {
                int64_t last = sid * 16 + 15;
                if (last < banks) {
                    double ready = s->bf[(lane * banks + last) * 7 + 1];
                    if (ready >= s->t[31] - 1e-7)
                        fr_audit.round_end[lane * sids + sid] = ready - s->t[31];
                }
            }
        }
        for (group = 0; group < groups; ++group) {
            fr_audit.act_group[lane * groups + group] = -DBL_MAX;
            fr_audit.write_bg[lane * groups + group] = -DBL_MAX;
        }
        /* Seed only the first observed state of a component. Every subsequent
         * batch imports its exact channel image, including closed-bank ACTs,
         * ACT groups, columns, WTR/DQ and both command buses. */
        fr_audit.data_available[lane] = data_available;
        fr_audit.last_rd_lane[lane] = last_rd;
        for (index = 0; index < s->window_counts[lane] && index < 5; ++index)
            fr_audit.faw[lane * 5 + index] = s->windows[lane * s->window_capacity + index];
        fr_audit.faw_count[lane] = (uint8_t)(s->window_counts[lane] < 5 ? s->window_counts[lane] : 5);
        ch = lane >> 1;
        for (bank = 0; bank < banks; ++bank) {
            int64_t item = lane * banks + bank;
            fr_audit.open_row[item] = s->rows[item];
            if (s->rows[item] >= 0) {
                fr_audit.act_anchor[item] = s->bf[item * 7];
                bank_open += 1;
            } else {
                fr_audit.act_anchor[item] = -DBL_MAX;
            }
            fr_audit.blocked[item] = s->bf[item * 7 + 1];
        }
        fr_audit.open_count[lane] += bank_open;
    }
    /* This seed runs once per component/channel. Later calls restore the
     * committed image, never a worker-local stream with cleared history. */
    for (ch = first_channel; ch < last_channel; ++ch) {
        fr_audit.row_busy[ch] = s->resources[ch*4];
        fr_audit.col_busy[ch] = s->resources[(s->channels+ch)*4];
    }
}

static int fr_audit_ensure(const fr_state *s)
{
    /* g[5] is the global bank-group count.  g[15] is the groups-per-SID
     * stride used by the address mapper; command fields carry global BGs. */
    int64_t lanes = s->g[0], banks = s->banks, bgs = s->g[5], sids = s->g[14], channels = s->channels;
    int64_t bank_n = lanes * banks, lane_group_n = lanes * bgs, lane_sid_n = lanes * sids;
    int64_t column_n = lane_sid_n * bgs, faw_n = lanes * 5, ring_n = lanes * 49;
    int64_t bank_ref_n = bank_n * 9;
    int64_t index;
    if (lanes <= 0 || banks <= 0 || bgs <= 0 || sids <= 0 || channels <= 0)
        return 0;
    if (fr_audit.initialized && fr_audit.lanes == lanes && fr_audit.banks == banks
            && fr_audit.bgs == bgs && fr_audit.sids == sids && fr_audit.channels == channels)
        return 1;
    if (fr_audit.initialized)
        fr_audit_free();
    fr_audit.lanes = (int)lanes; fr_audit.banks = (int)banks; fr_audit.bgs = (int)bgs;
    fr_audit.sids = (int)sids; fr_audit.channels = (int)channels; fr_audit.token = s->g[16];
#define FR_AUDIT_ALLOC(name, count, type) if (!fr_audit_alloc((void **)&fr_audit.name, (size_t)(count), sizeof(type))) goto failed
    FR_AUDIT_ALLOC(open_row, bank_n, int64_t); FR_AUDIT_ALLOC(act_anchor, bank_n, double);
    FR_AUDIT_ALLOC(pre_time, bank_n, double); FR_AUDIT_ALLOC(last_rd, bank_n, double);
    FR_AUDIT_ALLOC(last_wr_end, bank_n, double); FR_AUDIT_ALLOC(blocked, bank_n, double); FR_AUDIT_ALLOC(last_ref, bank_n, double);
    FR_AUDIT_ALLOC(last_pre, lanes, double); FR_AUDIT_ALLOC(last_rd_lane, lanes, double); FR_AUDIT_ALLOC(data_available, lanes, double);
    FR_AUDIT_ALLOC(act_group, lane_group_n, double); FR_AUDIT_ALLOC(act_group_top1, lane_group_n, double); FR_AUDIT_ALLOC(act_group_top2, lane_group_n, double);
    FR_AUDIT_ALLOC(act_group_bank1, lane_group_n, int64_t); FR_AUDIT_ALLOC(act_group_bank2, lane_group_n, int64_t);
    FR_AUDIT_ALLOC(act_top1, lanes, double); FR_AUDIT_ALLOC(act_top2, lanes, double); FR_AUDIT_ALLOC(act_top_group1, lanes, int64_t); FR_AUDIT_ALLOC(act_top_group2, lanes, int64_t);
    FR_AUDIT_ALLOC(ref_top1, lanes, double); FR_AUDIT_ALLOC(ref_top2, lanes, double); FR_AUDIT_ALLOC(ref_bank1, lanes, int64_t); FR_AUDIT_ALLOC(ref_bank2, lanes, int64_t);
    FR_AUDIT_ALLOC(last_col, 2 * column_n, double);
    FR_AUDIT_ALLOC(last_col_sid_top1, 2 * lane_sid_n, double); FR_AUDIT_ALLOC(last_col_sid_top2, 2 * lane_sid_n, double);
    FR_AUDIT_ALLOC(last_col_sid_group1, 2 * lane_sid_n, int64_t); FR_AUDIT_ALLOC(last_col_sid_group2, 2 * lane_sid_n, int64_t);
    FR_AUDIT_ALLOC(last_col_lane_top1, lanes, double); FR_AUDIT_ALLOC(last_col_lane_top2, lanes, double);
    FR_AUDIT_ALLOC(last_col_lane_sid1, lanes, int64_t); FR_AUDIT_ALLOC(last_col_lane_sid2, lanes, int64_t);
    FR_AUDIT_ALLOC(write_bg, lane_group_n, double);
    FR_AUDIT_ALLOC(faw, faw_n, double); FR_AUDIT_ALLOC(refresh_ring, ring_n, double); FR_AUDIT_ALLOC(round_end, lane_sid_n, double);
    FR_AUDIT_ALLOC(pc_group_gate, lanes, double); FR_AUDIT_ALLOC(bank_ref_ring, bank_ref_n, double); FR_AUDIT_ALLOC(refresh_t0, lanes, double);
    FR_AUDIT_ALLOC(faw_count, lanes, uint8_t); FR_AUDIT_ALLOC(refresh_count, lane_sid_n, uint8_t); FR_AUDIT_ALLOC(refresh_ring_count, lanes, uint8_t); FR_AUDIT_ALLOC(round_seen, lane_sid_n, uint8_t); FR_AUDIT_ALLOC(bank_ref_count, bank_n, uint8_t); FR_AUDIT_ALLOC(refresh_mask, lane_sid_n, uint64_t);
    FR_AUDIT_ALLOC(refresh_total, lanes, long long);
    FR_AUDIT_ALLOC(row_busy, channels, double); FR_AUDIT_ALLOC(col_busy, channels, double); FR_AUDIT_ALLOC(refab_blocked, lanes, double); FR_AUDIT_ALLOC(last_refab, lanes, double); FR_AUDIT_ALLOC(open_count, lanes, int64_t);
#undef FR_AUDIT_ALLOC
    for (index = 0; index < bank_n; ++index) {
        fr_audit.open_row[index] = -1;
        fr_audit.act_anchor[index] = fr_audit.pre_time[index] = fr_audit.last_rd[index] = fr_audit.last_wr_end[index] = fr_audit.blocked[index] = fr_audit.last_ref[index] = -DBL_MAX;
    }
    fr_audit_fill_double(fr_audit.last_pre, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.last_rd_lane, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.data_available, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.act_group, (size_t)lane_group_n, -DBL_MAX);
    fr_audit_fill_double(fr_audit.act_group_top1, (size_t)lane_group_n, -DBL_MAX);
    fr_audit_fill_double(fr_audit.act_group_top2, (size_t)lane_group_n, -DBL_MAX);
    fr_audit_fill_double(fr_audit.act_top1, (size_t)lanes, -DBL_MAX); fr_audit_fill_double(fr_audit.act_top2, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.ref_top1, (size_t)lanes, -DBL_MAX); fr_audit_fill_double(fr_audit.ref_top2, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.last_col, (size_t)(2 * column_n), -DBL_MAX);
    fr_audit_fill_double(fr_audit.last_col_sid_top1, (size_t)(2 * lane_sid_n), -DBL_MAX);
    fr_audit_fill_double(fr_audit.last_col_sid_top2, (size_t)(2 * lane_sid_n), -DBL_MAX);
    fr_audit_fill_double(fr_audit.last_col_lane_top1, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.last_col_lane_top2, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.write_bg, (size_t)lane_group_n, -DBL_MAX);
    fr_audit_fill_double(fr_audit.faw, (size_t)faw_n, -DBL_MAX); fr_audit_fill_double(fr_audit.refresh_ring, (size_t)ring_n, -DBL_MAX); fr_audit_fill_double(fr_audit.round_end, (size_t)lane_sid_n, -DBL_MAX);
    fr_audit_fill_double(fr_audit.pc_group_gate, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.bank_ref_ring, (size_t)bank_ref_n, -DBL_MAX);
    fr_audit_fill_double(fr_audit.refresh_t0, (size_t)lanes, -DBL_MAX);
    fr_audit_fill_double(fr_audit.row_busy, (size_t)channels, -DBL_MAX); fr_audit_fill_double(fr_audit.col_busy, (size_t)channels, -DBL_MAX); fr_audit_fill_double(fr_audit.refab_blocked, (size_t)lanes, -DBL_MAX); fr_audit_fill_double(fr_audit.last_refab, (size_t)lanes, -DBL_MAX);
    for (index = 0; index < lane_group_n; ++index) { fr_audit.act_group_bank1[index] = fr_audit.act_group_bank2[index] = -1; }
    for (index = 0; index < 2 * lane_sid_n; ++index) { fr_audit.last_col_sid_group1[index] = fr_audit.last_col_sid_group2[index] = -1; }
    for (index = 0; index < lanes; ++index) {
        fr_audit.act_top_group1[index] = fr_audit.act_top_group2[index] = -1;
        fr_audit.ref_bank1[index] = fr_audit.ref_bank2[index] = -1;
        fr_audit.last_col_lane_sid1[index] = fr_audit.last_col_lane_sid2[index] = -1;
    }
    for (index = 0; index < AUDIT_RULE_COUNT; ++index) { fr_audit.first_time[index] = DBL_MAX; fr_audit.first_kind[index][0] = '\0'; }
    fr_audit_empty=(unsigned char *)calloc(fr_audit_image_size(banks,bgs,sids),1);
    if(!fr_audit_empty) goto failed;
    fr_audit_image_transfer(fr_audit_empty,0,1);
    fr_audit.initialized = 1;
    fr_audit_reset_delta();
    return 1;
failed:
#undef FR_AUDIT_ALLOC
    fr_audit_free();
    return 0;
}

size_t fr_audit_image_size(int64_t banks, int64_t bgs, int64_t sids)
{
    int64_t lanes = 2, channels = 1;
    int64_t bank_n = lanes*banks, lane_group_n = lanes*bgs, lane_sid_n = lanes*sids;
    int64_t column_n = lane_sid_n*bgs, faw_n = lanes*5, ring_n = lanes*49, bank_ref_n = bank_n*9;
    return 8 + (bank_n)*sizeof(int64_t) + (bank_n)*sizeof(double) + (bank_n)*sizeof(double) + (bank_n)*sizeof(double) + (bank_n)*sizeof(double) + (bank_n)*sizeof(double) + (bank_n)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(double) + (lane_group_n)*sizeof(double) + (lane_group_n)*sizeof(double) + (lane_group_n)*sizeof(double) + (lane_group_n)*sizeof(int64_t) + (lane_group_n)*sizeof(int64_t) + (lanes)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(int64_t) + (lanes)*sizeof(int64_t) + (lanes)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(int64_t) + (lanes)*sizeof(int64_t) + (2 * column_n)*sizeof(double) + (2 * lane_sid_n)*sizeof(double) + (2 * lane_sid_n)*sizeof(double) + (2 * lane_sid_n)*sizeof(int64_t) + (2 * lane_sid_n)*sizeof(int64_t) + (lanes)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(int64_t) + (lanes)*sizeof(int64_t) + (lane_group_n)*sizeof(double) + (faw_n)*sizeof(double) + (ring_n)*sizeof(double) + (lane_sid_n)*sizeof(double) + (lanes)*sizeof(double) + (bank_ref_n)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(uint8_t) + (lane_sid_n)*sizeof(uint8_t) + (lanes)*sizeof(uint8_t) + (lane_sid_n)*sizeof(uint8_t) + (bank_n)*sizeof(uint8_t) + (lane_sid_n)*sizeof(uint64_t) + (lanes)*sizeof(long long) + (channels)*sizeof(double) + (channels)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(double) + (lanes)*sizeof(int64_t);
}

static void fr_audit_image_transfer(unsigned char *image, int64_t ch, int writing)
{
    int64_t lanes=fr_audit.lanes, banks=fr_audit.banks, bgs=fr_audit.bgs;
    int64_t sids=fr_audit.sids, channels=fr_audit.channels;
    int64_t bank_n=lanes*banks, lane_group_n=lanes*bgs, lane_sid_n=lanes*sids;
    int64_t column_n=lane_sid_n*bgs, faw_n=lanes*5, ring_n=lanes*49, bank_ref_n=bank_n*9;
    unsigned char *cursor=image+8;
#define IMAGE_FIELD(name, count, type) do { \
    size_t n=(size_t)(count)/(size_t)channels, bytes=n*sizeof(type); \
    if(writing) memcpy(cursor,fr_audit.name+ch*n,bytes); \
    else memcpy(fr_audit.name+ch*n,cursor,bytes); \
    cursor+=bytes; } while(0)
    IMAGE_FIELD(open_row, bank_n, int64_t);
    IMAGE_FIELD(act_anchor, bank_n, double);
    IMAGE_FIELD(pre_time, bank_n, double);
    IMAGE_FIELD(last_rd, bank_n, double);
    IMAGE_FIELD(last_wr_end, bank_n, double);
    IMAGE_FIELD(blocked, bank_n, double);
    IMAGE_FIELD(last_ref, bank_n, double);
    IMAGE_FIELD(last_pre, lanes, double);
    IMAGE_FIELD(last_rd_lane, lanes, double);
    IMAGE_FIELD(data_available, lanes, double);
    IMAGE_FIELD(act_group, lane_group_n, double);
    IMAGE_FIELD(act_group_top1, lane_group_n, double);
    IMAGE_FIELD(act_group_top2, lane_group_n, double);
    IMAGE_FIELD(act_group_bank1, lane_group_n, int64_t);
    IMAGE_FIELD(act_group_bank2, lane_group_n, int64_t);
    IMAGE_FIELD(act_top1, lanes, double);
    IMAGE_FIELD(act_top2, lanes, double);
    IMAGE_FIELD(act_top_group1, lanes, int64_t);
    IMAGE_FIELD(act_top_group2, lanes, int64_t);
    IMAGE_FIELD(ref_top1, lanes, double);
    IMAGE_FIELD(ref_top2, lanes, double);
    IMAGE_FIELD(ref_bank1, lanes, int64_t);
    IMAGE_FIELD(ref_bank2, lanes, int64_t);
    IMAGE_FIELD(last_col, 2 * column_n, double);
    IMAGE_FIELD(last_col_sid_top1, 2 * lane_sid_n, double);
    IMAGE_FIELD(last_col_sid_top2, 2 * lane_sid_n, double);
    IMAGE_FIELD(last_col_sid_group1, 2 * lane_sid_n, int64_t);
    IMAGE_FIELD(last_col_sid_group2, 2 * lane_sid_n, int64_t);
    IMAGE_FIELD(last_col_lane_top1, lanes, double);
    IMAGE_FIELD(last_col_lane_top2, lanes, double);
    IMAGE_FIELD(last_col_lane_sid1, lanes, int64_t);
    IMAGE_FIELD(last_col_lane_sid2, lanes, int64_t);
    IMAGE_FIELD(write_bg, lane_group_n, double);
    IMAGE_FIELD(faw, faw_n, double);
    IMAGE_FIELD(refresh_ring, ring_n, double);
    IMAGE_FIELD(round_end, lane_sid_n, double);
    IMAGE_FIELD(pc_group_gate, lanes, double);
    IMAGE_FIELD(bank_ref_ring, bank_ref_n, double);
    IMAGE_FIELD(refresh_t0, lanes, double);
    IMAGE_FIELD(faw_count, lanes, uint8_t);
    IMAGE_FIELD(refresh_count, lane_sid_n, uint8_t);
    IMAGE_FIELD(refresh_ring_count, lanes, uint8_t);
    IMAGE_FIELD(round_seen, lane_sid_n, uint8_t);
    IMAGE_FIELD(bank_ref_count, bank_n, uint8_t);
    IMAGE_FIELD(refresh_mask, lane_sid_n, uint64_t);
    IMAGE_FIELD(refresh_total, lanes, long long);
    IMAGE_FIELD(row_busy, channels, double);
    IMAGE_FIELD(col_busy, channels, double);
    IMAGE_FIELD(refab_blocked, lanes, double);
    IMAGE_FIELD(last_refab, lanes, double);
    IMAGE_FIELD(open_count, lanes, int64_t);
#undef IMAGE_FIELD
    if(writing) image[0]=1;
}

static void fr_audit_prepare(const fr_state *s)
{
    int64_t ch;
    size_t size=fr_audit_image_size(s->banks,s->g[5],s->g[14]);
    if(!s->a->audit_history) return;
    if(s->a->audit_stride!=(int64_t)size || !fr_audit_ensure(s)) fr_fail(FR_ERR_MEMORY);
    for(ch=s->a->channel_first;ch<s->a->channel_stop;++ch) {
        unsigned char *image=s->a->audit_history+(ch-s->a->channel_first)*size;
        if(image[0]) fr_audit_image_transfer(image,ch,0);
        else {
            fr_audit_image_transfer(fr_audit_empty,ch,0);
            fr_audit_seed(s,ch,ch+1);
        }
    }
}

static void fr_audit_save(const fr_args *args)
{
    int64_t ch;
    if(!args->audit_history || !fr_audit.initialized) return;
    for(ch=args->channel_first;ch<args->channel_stop;++ch)
        fr_audit_image_transfer(args->audit_history+(ch-args->channel_first)*args->audit_stride,ch,1);
}

static int fr_audit_before(double later, double earlier, double minimum)
{
    return later - earlier < minimum - 1e-7;
}

static int fr_audit_after(double later, double earlier, double maximum)
{
    return later - earlier > maximum + 1e-7;
}

static void fr_audit_test(int rule, int bad, const char *kind, double time, int64_t stack, int64_t channel,
                          int64_t pc, int64_t sid, int64_t bg, int64_t bank, int64_t row)
{
    fr_audit.checks[rule] += 1; fr_audit.delta_checks[rule] += 1;
    if (!bad) return;
    fr_audit.violations[rule] += 1; fr_audit.delta_violations[rule] += 1;
    if (fr_audit.first_kind[rule][0] == '\0') {
        fr_audit.delta_first_seen[rule] = 1;
        strncpy(fr_audit.first_kind[rule], kind, sizeof(fr_audit.first_kind[rule]) - 1);
        fr_audit.first_kind[rule][sizeof(fr_audit.first_kind[rule]) - 1] = '\0';
        fr_audit.first_time[rule] = time; fr_audit.first_stack[rule] = stack; fr_audit.first_channel[rule] = channel;
        fr_audit.first_pc[rule] = pc; fr_audit.first_sid[rule] = sid; fr_audit.first_bg[rule] = bg;
        fr_audit.first_bank[rule] = bank; fr_audit.first_row[rule] = row;
    }
}

static void fr_audit_recent_update(double *top1, double *top2, int64_t *key1, int64_t *key2, int64_t key, double time)
{
    if (*key1 == key) { *top1 = time; return; }
    if (*key2 == key) {
        *top2 = time;
        if (*top2 > *top1) { double stamp = *top1; int64_t old = *key1; *top1 = *top2; *key1 = *key2; *top2 = stamp; *key2 = old; }
        return;
    }
    if (*top1 <= -DBL_MAX / 2 || time > *top1) { *top2 = *top1; *key2 = *key1; *top1 = time; *key1 = key; }
    else if (*top2 <= -DBL_MAX / 2 || time > *top2) { *top2 = time; *key2 = key; }
}

static double fr_audit_recent_other(double top1, double top2, int64_t key1, int64_t key2, int64_t key)
{
    if (key1 != key && top1 > -DBL_MAX / 2) return top1;
    if (key2 != key && top2 > -DBL_MAX / 2) return top2;
    return -DBL_MAX;
}

static void fr_audit_faw(const fr_state *s, int64_t lane, const char *kind, double time, int64_t stack, int64_t ch,
                         int64_t pc, int64_t sid, int64_t bg, int64_t bank, int64_t row)
{
    int64_t count = fr_audit.faw_count[lane], index;
    double *window = fr_audit.faw + lane * 5;
    while (count > 0 && window[0] <= time - s->t[30] + 1e-7) {
        for (index = 1; index < count; ++index) window[index - 1] = window[index];
        count -= 1;
    }
    fr_audit_test(AUDIT_TFAW, count >= 4, kind, time, stack, ch, pc, sid, bg, bank, row);
    if (count < 5) window[count++] = time;
    else { window[0] = window[1]; window[1] = window[2]; window[2] = window[3]; window[3] = time; }
    fr_audit.faw_count[lane] = (uint8_t)count;
}

static void fr_audit_event(const fr_state *s, const char *kind, int64_t lane, int64_t sid, int64_t group,
                           int64_t bank, int64_t row, int64_t column, double start, double data_start, double data_end)
{
    int64_t ch, pc, bank_index, group_index, sid_index, column_index, index;
    int64_t stack, local_ch;
    double anchor, previous, other;
    fr_audit_state *a = &fr_audit;
    if (!fr_audit_ensure(s)) return;
    ch = lane >> 1; pc = lane & 1; stack = s->g[11] > 0 ? lane / s->g[11] : 0; local_ch = s->g[11] > 0 ? (lane % s->g[11]) / 2 : ch;
    bank_index = lane * a->banks + bank; group_index = lane * a->bgs + group; sid_index = lane * a->sids + sid;
    column_index = (lane * a->sids + sid) * a->bgs + group;
    if (kind[0] == 'A') a->by_type[0] += 1;
    else if (kind[0] == 'P' && kind[1] == 'R' && kind[2] == 'E') a->by_type[1] += 1;
    else if (strcmp(kind, "PRE_ALL") == 0) a->by_type[2] += 1;
    else if (strcmp(kind, "REFab") == 0) a->by_type[3] += 1;
    else if (strcmp(kind, "REFpb") == 0) a->by_type[4] += 1;
    else if (kind[0] == 'R') a->by_type[5] += 1;
    else if (kind[0] == 'W') a->by_type[6] += 1;
    a->commands += 1; a->delta_commands += 1;

    /* The two CA buses are shared by the two PCs of one channel. */
    if (strcmp(kind, "ACT") == 0 || strcmp(kind, "PRE") == 0 || strcmp(kind, "PRE_ALL") == 0 || strcmp(kind, "REFpb") == 0 || strcmp(kind, "REFab") == 0) {
        fr_audit_test(AUDIT_BUS_ROW, start < a->row_busy[ch] - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
        a->row_busy[ch] = start + (strcmp(kind, "ACT") == 0 ? 1.5 : .5) * s->t[0];
    } else if (strcmp(kind, "RD") == 0 || strcmp(kind, "WR") == 0) {
        fr_audit_test(AUDIT_BUS_COL, start < a->col_busy[ch] - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
        a->col_busy[ch] = start + s->t[0];
    }

    if (strcmp(kind, "REFab") == 0) {
        fr_audit_test(AUDIT_REFAB_CLOSED, a->open_count[lane] != 0, kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->last_pre[lane] > -DBL_MAX / 2)
            fr_audit_test(AUDIT_REFAB_TRP, fr_audit_before(start, a->last_pre[lane], s->t[6]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (start < a->refab_blocked[lane] - 1e-7)
            fr_audit_test(AUDIT_REFAB_NOP, 1, kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->last_refab[lane] > -DBL_MAX / 2) fr_audit_test(AUDIT_REFAB_INTERVAL, fr_audit_after(start, a->last_refab[lane], 9 * s->t[20]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        a->last_refab[lane] = start; a->refab_blocked[lane] = start + s->t[21];
        return;
    }
    if (start < a->refab_blocked[lane] - 1e-7) fr_audit_test(AUDIT_REFAB_NOP, 1, kind, start, stack, local_ch, pc, sid, group, bank, row);
    if (strcmp(kind, "PRE_ALL") == 0) {
        a->open_count[lane] = 0;
        if (a->last_pre[lane] > -DBL_MAX/2)
            fr_audit_test(AUDIT_TPPD, fr_audit_before(start,a->last_pre[lane],s->t[19]),kind,start,stack,local_ch,pc,sid,group,bank,row);
        for (index = 0; index < a->banks; ++index) {
            int64_t item = lane * a->banks + index;
            if(a->open_row[item] >= 0) {
                if(a->act_anchor[item] > -DBL_MAX/2) {
                    fr_audit_test(AUDIT_TRAS,fr_audit_before(start,a->act_anchor[item],s->t[7]),kind,start,stack,local_ch,pc,sid,group,index,row);
                    fr_audit_test(AUDIT_TRASMAX,fr_audit_after(start,a->act_anchor[item],s->t[29]),kind,start,stack,local_ch,pc,sid,group,index,row);
                }
                if(a->last_rd[item] > -DBL_MAX/2) fr_audit_test(AUDIT_RTP,fr_audit_before(start,a->last_rd[item],s->t[9]),kind,start,stack,local_ch,pc,sid,group,index,row);
                if(a->last_wr_end[item] > -DBL_MAX/2) fr_audit_test(AUDIT_WR,fr_audit_before(start,a->last_wr_end[item],s->t[10]),kind,start,stack,local_ch,pc,sid,group,index,row);
            }
            a->open_row[item] = -1; a->pre_time[item]=start;
            a->last_rd[item]=a->last_wr_end[item]=-DBL_MAX;
        }
        a->last_pre[lane]=start;
        return;
    }
    if (bank < 0 || bank >= a->banks) return;

    if (strcmp(kind, "ACT") == 0) {
        anchor = start + s->t[0];
        fr_audit_test(AUDIT_BANK_OPEN, a->open_row[bank_index] >= 0, kind, start, stack, local_ch, pc, sid, group, bank, row);
        fr_audit_test(AUDIT_REFRESH_BLOCK, anchor < a->blocked[bank_index] - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->pre_time[bank_index] > -DBL_MAX / 2) fr_audit_test(AUDIT_TRP, fr_audit_before(start, a->pre_time[bank_index], s->t[6]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->act_anchor[bank_index] > -DBL_MAX / 2) fr_audit_test(AUDIT_TRC, fr_audit_before(anchor, a->act_anchor[bank_index], s->t[8]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        previous = a->act_group[group_index];
        if (previous > -DBL_MAX / 2) fr_audit_test(AUDIT_TRRD, fr_audit_before(anchor, previous, s->t[14]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        other = fr_audit_recent_other(a->act_top1[lane], a->act_top2[lane], a->act_top_group1[lane], a->act_top_group2[lane], group);
        if (other > -DBL_MAX / 2) fr_audit_test(AUDIT_TRRD, fr_audit_before(anchor, other, s->t[15]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        other = fr_audit_recent_other(a->ref_top1[lane], a->ref_top2[lane], a->ref_bank1[lane], a->ref_bank2[lane], bank);
        if (other > -DBL_MAX / 2) fr_audit_test(AUDIT_REF_ACT, fr_audit_before(anchor, other, s->t[32]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        fr_audit_faw(s, lane, kind, anchor, stack, local_ch, pc, sid, group, bank, row);
        fr_audit_recent_update(&a->act_top1[lane], &a->act_top2[lane], &a->act_top_group1[lane], &a->act_top_group2[lane], group, anchor);
        a->act_group[group_index] = anchor;
        fr_audit_recent_update(&a->act_group_top1[group_index], &a->act_group_top2[group_index], &a->act_group_bank1[group_index], &a->act_group_bank2[group_index], bank, anchor);
        a->open_row[bank_index] = row; a->act_anchor[bank_index] = anchor; a->pre_time[bank_index] = a->last_rd[bank_index] = a->last_wr_end[bank_index] = -DBL_MAX; a->blocked[bank_index] = -DBL_MAX; a->open_count[lane] += 1;
    } else if (strcmp(kind, "RD") == 0 || strcmp(kind, "WR") == 0) {
        fr_audit_test(AUDIT_ROW_MATCH, a->open_row[bank_index] < 0 || a->open_row[bank_index] != row, kind, start, stack, local_ch, pc, sid, group, bank, row);
        fr_audit_test(AUDIT_REFRESH_BLOCK, start < a->blocked[bank_index] - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->act_anchor[bank_index] > -DBL_MAX / 2) fr_audit_test(AUDIT_TRCD, fr_audit_before(start, a->act_anchor[bank_index], strcmp(kind, "RD") == 0 ? s->t[4] : s->t[5]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (strcmp(kind, "RD") == 0) a->last_rd[bank_index] = start;
        else a->last_wr_end[bank_index] = data_end > -DBL_MAX / 2 ? data_end : start + s->t[23] + s->t[26];
        if (strcmp(kind, "RD") == 0 || strcmp(kind, "WR") == 0) {
            int read = strcmp(kind, "RD") == 0;
            int64_t typed_sid_index = 2 * (lane * a->sids + sid) + (read ? 0 : 1);
            column_index = typed_sid_index * a->bgs + group;
            previous = a->last_col[column_index];
            if (previous > -DBL_MAX / 2) fr_audit_test(AUDIT_TCCDL, fr_audit_before(start, previous, s->t[11]), kind, start, stack, local_ch, pc, sid, group, bank, row);
            previous = fr_audit_recent_other(a->last_col_sid_top1[typed_sid_index],
                                             a->last_col_sid_top2[typed_sid_index],
                                             a->last_col_sid_group1[typed_sid_index],
                                             a->last_col_sid_group2[typed_sid_index], group);
            if (previous > -DBL_MAX / 2)
                fr_audit_test(AUDIT_TCCDS, fr_audit_before(start, previous, s->t[12]), kind, start, stack, local_ch, pc, sid, group, bank, row);
            if (read) {
                previous = fr_audit_recent_other(a->last_col_lane_top1[lane], a->last_col_lane_top2[lane],
                                                 a->last_col_lane_sid1[lane], a->last_col_lane_sid2[lane], sid);
                if (previous > -DBL_MAX / 2)
                    fr_audit_test(AUDIT_TCCDR, fr_audit_before(start, previous, s->t[13]), kind, start, stack, local_ch, pc, sid, group, bank, row);
            }
            a->last_col[column_index] = start;
            fr_audit_recent_update(&a->last_col_sid_top1[typed_sid_index], &a->last_col_sid_top2[typed_sid_index],
                                   &a->last_col_sid_group1[typed_sid_index], &a->last_col_sid_group2[typed_sid_index], group, start);
            if (read)
                fr_audit_recent_update(&a->last_col_lane_top1[lane], &a->last_col_lane_top2[lane],
                                       &a->last_col_lane_sid1[lane], &a->last_col_lane_sid2[lane], sid, start);
        }
        if (strcmp(kind, "RD") == 0) {
            previous = a->last_rd_lane[lane]; if (previous > -DBL_MAX / 2) { /* kept for the next WR */ }
            other = a->data_available[lane];
            if (data_start > -DBL_MAX / 2) fr_audit_test(AUDIT_DQ, data_start < other - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
            a->data_available[lane] = data_end > -DBL_MAX / 2 ? data_end : start + s->t[22] + s->t[26];
            a->last_rd_lane[lane] = start;
            for (index = 0; index < a->bgs; ++index) if (a->write_bg[lane * a->bgs + index] > -DBL_MAX / 2) {
                double gap = start - a->write_bg[lane * a->bgs + index];
                fr_audit_test(index == group ? AUDIT_WTR : AUDIT_WTR, gap < (index == group ? s->t[16] : s->t[17]) - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
            }
        } else {
            if (a->last_rd_lane[lane] > -DBL_MAX / 2) fr_audit_test(AUDIT_RTW, fr_audit_before(start, a->last_rd_lane[lane], s->t[18]), kind, start, stack, local_ch, pc, sid, group, bank, row);
            other = a->data_available[lane]; if (data_start > -DBL_MAX / 2) fr_audit_test(AUDIT_DQ, data_start < other - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
            a->data_available[lane] = data_end > -DBL_MAX / 2 ? data_end : start + s->t[23] + s->t[26];
            a->write_bg[lane * a->bgs + group] = a->data_available[lane];
        }
    } else if (strcmp(kind, "PRE") == 0) {
        fr_audit_test(AUDIT_PRE_CLOSED, a->open_row[bank_index] < 0, kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->act_anchor[bank_index] > -DBL_MAX / 2) {
            fr_audit_test(AUDIT_TRAS, fr_audit_before(start, a->act_anchor[bank_index], s->t[7]), kind, start, stack, local_ch, pc, sid, group, bank, row);
            fr_audit_test(AUDIT_TRASMAX, fr_audit_after(start, a->act_anchor[bank_index], s->t[29]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        }
        if (a->last_rd[bank_index] > -DBL_MAX / 2) fr_audit_test(AUDIT_RTP, fr_audit_before(start, a->last_rd[bank_index], s->t[9]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->last_wr_end[bank_index] > -DBL_MAX / 2) fr_audit_test(AUDIT_WR, fr_audit_before(start, a->last_wr_end[bank_index], s->t[10]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->last_pre[lane] > -DBL_MAX / 2) fr_audit_test(AUDIT_TPPD, fr_audit_before(start, a->last_pre[lane], s->t[19]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        a->open_row[bank_index] = -1; a->pre_time[bank_index] = start; a->last_rd[bank_index] = a->last_wr_end[bank_index] = -DBL_MAX; if (a->open_count[lane] > 0) a->open_count[lane] -= 1; a->last_pre[lane] = start;
    } else if (strcmp(kind, "REFpb") == 0) {
        fr_audit_test(AUDIT_REF_CLOSED, a->open_row[bank_index] >= 0, kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (a->pre_time[bank_index] > -DBL_MAX / 2) fr_audit_test(AUDIT_REF_TRP, fr_audit_before(start, a->pre_time[bank_index], s->t[6]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        previous = a->act_anchor[bank_index]; if (previous > -DBL_MAX / 2) fr_audit_test(AUDIT_ACT_REF, fr_audit_before(start, previous, s->t[8]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        previous = fr_audit_recent_other(a->act_group_top1[group_index], a->act_group_top2[group_index], a->act_group_bank1[group_index], a->act_group_bank2[group_index], bank); if (previous > -DBL_MAX / 2) fr_audit_test(AUDIT_ACT_REF, fr_audit_before(start, previous, s->t[14]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        other = fr_audit_recent_other(a->act_top1[lane], a->act_top2[lane], a->act_top_group1[lane], a->act_top_group2[lane], group); if (other > -DBL_MAX / 2) fr_audit_test(AUDIT_ACT_REF, fr_audit_before(start, other, s->t[15]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        previous = a->last_ref[bank_index];
        if (previous > -DBL_MAX / 2) {
            fr_audit_test(AUDIT_REF_SPACING, fr_audit_before(start, previous, s->t[31]), kind, start, stack, local_ch, pc, sid, group, bank, row);
            fr_audit_test(AUDIT_REFRESH_INTERVAL, fr_audit_after(start, previous, 9 * s->t[20]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        }
        other = fr_audit_recent_other(a->ref_top1[lane], a->ref_top2[lane], a->ref_bank1[lane], a->ref_bank2[lane], bank); if (other > -DBL_MAX / 2) fr_audit_test(AUDIT_REF_SPACING, fr_audit_before(start, other, s->t[32]), kind, start, stack, local_ch, pc, sid, group, bank, row);
        a->blocked[bank_index] = start + s->t[31]; a->last_ref[bank_index] = start;
        fr_audit_recent_update(&a->ref_top1[lane], &a->ref_top2[lane], &a->ref_bank1[lane], &a->ref_bank2[lane], bank, start);
        if (a->pc_group_gate[lane] > -DBL_MAX / 2)
            fr_audit_test(AUDIT_REFRESH_GROUP_GAP, start < a->pc_group_gate[lane] - 1e-7, kind, start, stack, local_ch, pc, sid, group, bank, row);
        if (sid >= 0 && sid < a->sids) {
            uint64_t bit = UINT64_C(1) << (bank % 64), mask = a->refresh_mask[sid_index];
            fr_audit_test(AUDIT_REFRESH_ORDER, (mask & bit) != 0, kind, start, stack, local_ch, pc, sid, group, bank, row);
            if (a->refresh_count[sid_index] == 0 && a->round_end[sid_index] > -DBL_MAX / 2) fr_audit_test(AUDIT_REFRESH_GROUP_GAP, fr_audit_before(start, a->round_end[sid_index], s->t[31]), kind, start, stack, local_ch, pc, sid, group, bank, row);
            a->refresh_mask[sid_index] = mask | bit; a->refresh_count[sid_index] += 1;
            if (a->refresh_count[sid_index] >= a->banks / a->sids) {
                a->refresh_count[sid_index] = 0; a->refresh_mask[sid_index] = 0; a->round_end[sid_index] = start;
                a->pc_group_gate[lane] = start + s->t[31];
            }
        }
        { uint8_t count = a->bank_ref_count[bank_index]; double *ring = a->bank_ref_ring + bank_index * 9;
          while (count > 0 && start - ring[0] >= s->t[20] - 1e-7) {
              memmove(ring, ring + 1, (size_t)(count - 1) * sizeof(double));
              count -= 1;
          }
          fr_audit_test(AUDIT_REFRESH_RATE, count >= 9, kind, start, stack, local_ch, pc, sid, group, bank, row);
          if (count < 9) ring[count++] = start;
          else { memmove(ring, ring + 1, 8 * sizeof(double)); ring[8] = start; }
          a->bank_ref_count[bank_index] = count;
        }
        if (a->refresh_t0[lane] <= -DBL_MAX / 2)
            a->refresh_t0[lane] = start;
        a->refresh_total[lane] += 1;
        { double due = floor((start - a->refresh_t0[lane]) / s->t[33]) - 8. * 48.;
          fr_audit_test(AUDIT_REFRESH_RATE, due > (double)a->refresh_total[lane], kind, start, stack, local_ch, pc, sid, group, bank, row);
        }
        fr_audit_faw(s, lane, kind, start, stack, local_ch, pc, sid, group, bank, row);
    }
}

FR_INLINE void fr_trace(const fr_state *s, const char *kind, int64_t lane, int64_t sid,
                        int64_t group, int64_t bank, int64_t row, int64_t column, double start,
                        double data_start, double data_end)
{
    int64_t lanes_per_stack, stack, local_channel;
    int kind_index;
    if (fr_trace_file == NULL && fr_trace_summary_file == NULL)
        return;
    if (fr_trace_summary_file != NULL)
        fr_audit_event(s, kind, lane, sid, group, bank, row, column, start, data_start, data_end);
    kind_index = fr_trace_kind_index(kind);
    fr_trace_total++;
    if (kind_index >= 0)
        fr_trace_counts[kind_index]++;
    if (fr_trace_file == NULL || fr_trace_emitted >= fr_trace_limit
            || start < fr_trace_min_time || start > fr_trace_max_time)
        return;
    lanes_per_stack = s->g[11] > 0 ? s->g[11] : s->g[0];
    stack = lane / lanes_per_stack;
    local_channel = (lane % lanes_per_stack) / 2;
    fprintf(fr_trace_file, "%s\t%.17g\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\n",
            kind, start, (long long)stack, (long long)local_channel,
            (long long)(lane & 1), (long long)sid, (long long)group,
            (long long)bank, (long long)row, (long long)column);
    fr_trace_emitted++;
}

static void fr_command(const fr_state *s, int64_t channel, int64_t kind, double earliest, double cycles, int rising,
                       double *counts, double *start_out)
{
    const double *t = s->t;
    double *resource = s->resources + 4 * (channel + kind * s->channels);
    double quantum = rising || kind == 1 || s->g[7] ? t[0] : s->half;
    double start = fr_align(fr_max(earliest, resource[0]), quantum);
    double duration = cycles * t[0];
    double end = fr_after(start, duration, s->half);
    resource[0] = end;
    resource[1] += duration;
    resource[2] = start;
    resource[3] = end;
    s->touched[channel + kind * s->channels] = 1;
    counts[5 + kind] += duration;
    *start_out = start;
}

/* A command on the row bus, on its precomputed duration. */
FR_INLINE double fr_row_command(const fr_state *s, int64_t ch, double earliest, double quantum, double duration,
                                int64_t duration_tick, double *counts)
{
    double *resource = s->resources + 4 * ch;
    double start = fr_align(fr_max(earliest, resource[0]), quantum);
    double end = fr_after_tick(start, duration, duration_tick, s->half, s->inverse);
    resource[0] = end;
    resource[1] += duration;
    resource[2] = start;
    resource[3] = end;
    s->touched[ch] = 1;
    counts[5] += duration;
    return start;
}

FR_INLINE double fr_pre_ready(const fr_state *s, int64_t lane, double earliest)
{
    if (s->pc[lane * FR_PC] >= 0)
        earliest = fr_max(earliest, fr_after(s->pc[lane * FR_PC], s->t[19], s->half));
    return earliest;
}

FR_NOINLINE void fr_record_activate(const fr_state *s, int64_t lane, double anchor)
{
    double *row = s->windows + lane * s->window_capacity;
    int64_t count = s->window_counts[lane], position = count;
    if (count == s->window_capacity)
        fr_fail(FR_ERR_ACT_CALENDAR);
    while (position > 0 && row[position - 1] > anchor) {
        row[position] = row[position - 1];
        position -= 1;
    }
    row[position] = anchor;
    s->window_counts[lane] = count + 1;
}

/* The in-order PRE edge of one bank, and the command itself. */
static double fr_precharge_start(const fr_state *s, int64_t lane, int64_t bank, double earliest)
{
    const double *t = s->t;
    const double *lane_pc = s->pc + lane * FR_PC;
    double half = s->half, inverse = s->inverse;
    double ready = fr_max(fr_max(earliest, s->bf[bank * 7 + 2]), lane_pc[1]);
    if (lane_pc[0] >= 0)
        ready = fr_max(ready, fr_after_tick(lane_pc[0], t[19], s->ticks[19], half, inverse));
    if (s->channel_int[(lane >> 1) * 2] == bank)
        ready = fr_max(ready, fr_after_tick(s->channel[(lane >> 1) * 2], 2 * t[0], s->follow_tick, half, inverse));
    return fr_align(fr_max(ready, s->resources[(lane >> 1) * 4]), s->precharge_step);
}

static void fr_precharge(const fr_state *s, int64_t lane, int64_t bank, double earliest, double *counts)
{
    double start = fr_row_command(s, lane >> 1, fr_precharge_start(s, lane, bank, earliest), s->precharge_step,
                                  s->pre_duration, s->pre_duration_tick, counts);
    {
        int64_t within = bank % s->banks;
        fr_trace(s, "PRE", lane, within / (s->g[15] * s->g[4]), within / s->g[4], within,
                 s->rows[bank], -1, start, -DBL_MAX, -DBL_MAX);
    }
    double next = fr_after_tick(start, s->t[6], s->ticks[6], s->half, s->inverse);
    counts[1] += 1;
    s->pc[lane * FR_PC] = start;
    s->rows[bank] = -1;
    s->bf[bank * 7 + 1] = next;
    s->bf[bank * 7 + 5] = next;
}

static double fr_next_active_deadline(const fr_state *s, int64_t ch)
{
    const double *t = s->t;
    const int64_t *rows = s->rows, *addresses = s->addresses;
    const double *bf = s->bf;
    double deadline = HUGE_VAL, half = s->half;
    int64_t first = ch * 2 * s->banks, stop = (ch + 1) * 2 * s->banks, bank;
    if (t[29] > 0) {
        int64_t delay_ticks = fr_round(t[29] / half);
        if (t[29] != (double)delay_ticks * half) {
            /* A non-grid delay always uses ordinary floating addition for
             * every anchor, including externally restored non-grid anchors.
             * IEEE addition with one fixed finite operand is monotone. */
            double anchor = HUGE_VAL;
            for (bank = first; bank < stop; ++bank)
                if (addresses[bank] >= 0 && rows[bank] >= 0)
                    anchor = fr_min(anchor, bf[bank * 7]);
            if (fr_isfinite(anchor))
                return fr_after(anchor, t[29], half);
            return deadline;
        }
        for (bank = first; bank < stop; ++bank)
            if (addresses[bank] >= 0 && rows[bank] >= 0)
                deadline = fr_min(deadline, fr_after(bf[bank * 7], t[29], half));
    }
    return deadline;
}

/* numpy.argsort's quicksort as Numba compiles it, which is not stable. */
static void fr_argsort(const double *keys, int64_t size, int64_t *order)
{
    int64_t stack_low[100], stack_high[100], n = 1, index;
    for (index = 0; index < size; ++index)
        order[index] = index;
    if (size < 2)
        return;
    stack_low[0] = 0;
    stack_high[0] = size - 1;
/* Keys here are deadlines that passed a comparison, so none is NaN. */
#define FR_LESS(left, right) ((left) < (right))
#define FR_SWAP(left, right) do { int64_t swap_ = order[left]; order[left] = order[right]; order[right] = swap_; } while (0)
    while (n > 0) {
        int64_t low, high, i, j;
        n -= 1;
        low = stack_low[n];
        high = stack_high[n];
        while (high - low >= 15) {
            int64_t mid = (low + high) >> 1;
            double pivot;
            if (FR_LESS(keys[order[mid]], keys[order[low]]))
                FR_SWAP(low, mid);
            if (FR_LESS(keys[order[high]], keys[order[mid]]))
                FR_SWAP(high, mid);
            if (FR_LESS(keys[order[mid]], keys[order[low]]))
                FR_SWAP(low, mid);
            pivot = keys[order[mid]];
            FR_SWAP(high, mid);
            i = low;
            j = high - 1;
            for (;;) {
                while (i < high && FR_LESS(keys[order[i]], pivot))
                    i += 1;
                while (j >= low && FR_LESS(pivot, keys[order[j]]))
                    j -= 1;
                if (i >= j)
                    break;
                FR_SWAP(i, j);
                i += 1;
                j -= 1;
            }
            FR_SWAP(i, high);
            if (high - i > i - low) {
                if (high > i) {
                    stack_low[n] = i + 1;
                    stack_high[n] = high;
                    n += 1;
                }
                high = i - 1;
            } else {
                if (i > low) {
                    stack_low[n] = low;
                    stack_high[n] = i - 1;
                    n += 1;
                }
                low = i + 1;
            }
        }
        if (high > low) {
            for (i = low + 1; i < high + 1; ++i) {
                int64_t k = order[i];
                double value = keys[k];
                j = i;
                while (j > low && FR_LESS(value, keys[order[j - 1]])) {
                    order[j] = order[j - 1];
                    j -= 1;
                }
                order[j] = k;
            }
        }
    }
#undef FR_LESS
#undef FR_SWAP
}

FR_NOINLINE void fr_close_expiring(const fr_state *s, int64_t ch, double before, double *counts)
{
    const double *t = s->t;
    int64_t *rows = s->rows;
    const int64_t *addresses = s->addresses, *channel_int = s->channel_int;
    double *bf = s->bf, *pc = s->pc;
    const double *channel = s->channel, *resources = s->resources;
    double half = s->half;
    int64_t *pending_banks = s->close_banks, *order = s->close_order;
    double *deadlines = s->close_deadlines, *starts = s->close_starts;
    int64_t count = 0, bank, index;
    double next_start = HUGE_VAL, next_pc[2] = {HUGE_VAL, HUGE_VAL};
    double duration, step;
    for (bank = ch * 2 * s->banks; bank < (ch + 1) * 2 * s->banks; ++bank) {
        if (addresses[bank] >= 0 && rows[bank] >= 0 && fr_after(bf[bank * 7], t[29], half) < before) {
            pending_banks[count] = bank;
            deadlines[count] = fr_after(bf[bank * 7], t[29], half);
            count += 1;
        }
    }
    fr_argsort(deadlines, count, order);
    duration = t[2] * t[0];
    step = s->precharge_step;
    for (index = count - 1; index >= 0; --index) {
        int64_t item = order[index], lane;
        double latest, start, earliest;
        bank = pending_banks[item];
        lane = fr_floordiv(bank, s->banks);
        latest = fr_min(fr_min(deadlines[item], fr_after(next_start, -duration, half)),
                        fr_after(next_pc[lane & 1], -t[19], half));
        start = (double)fr_ifloor(latest / step) * step;
        while (fr_align(start, step) > latest)
            start = fr_after(start, -step, half);
        start = fr_align(start, step);
        earliest = fr_pre_ready(s, lane, fr_max(fr_max(bf[bank * 7 + 2], pc[lane * FR_PC + 1]), resources[ch * 4]));
        if (channel_int[ch * 2] == bank)
            earliest = fr_max(earliest, fr_after(channel[ch * 2], 2 * t[0], half));
        if (start < earliest)
            fr_fail(FR_ERR_TRAS_PRECHARGE);
        starts[item] = start;
        next_start = start;
        next_pc[lane & 1] = start;
    }
    for (index = 0; index < count; ++index) {
        int64_t item = order[index], lane;
        double earliest, start, next;
        bank = pending_banks[item];
        lane = fr_floordiv(bank, s->banks);
        earliest = fr_pre_ready(s, lane, fr_max(fr_max(starts[item], bf[bank * 7 + 2]), pc[lane * FR_PC + 1]));
        if (channel_int[ch * 2] == bank)
            earliest = fr_max(earliest, fr_after(channel[ch * 2], 2 * t[0], half));
        fr_command(s, ch, 0, earliest, t[2], 0, counts, &start);
        {
            int64_t within = bank % s->banks;
            fr_trace(s, "PRE", lane, within / (s->g[15] * s->g[4]), within / s->g[4], within,
                     rows[bank], -1, start, -DBL_MAX, -DBL_MAX);
        }
        counts[1] += 1;
        pc[lane * FR_PC] = start;
        rows[bank] = -1;
        next = fr_after(start, t[6], half);
        bf[bank * 7 + 1] = next;
        bf[bank * 7 + 5] = next;
    }
}

FR_NOINLINE void fr_refresh_pc(const fr_state *s, int64_t lane, double due, double *counts)
{
    const double *t = s->t;
    int64_t *rows = s->rows;
    const int64_t *addresses = s->addresses, *g = s->g;
    double *bf = s->bf, *pc = s->pc;
    const double *channel = s->channel;
    double half = s->half;
    int64_t first_bank = lane * s->banks, end_bank = (lane + 1) * s->banks, ch = lane >> 1, bank;
    double earliest = fr_max(due, pc[lane * FR_PC + 1]);
    double start, last_act;
    int opened = 0;
    s->pc_seen[lane] = 1;
    for (bank = first_bank; bank < end_bank; ++bank) {
        if (addresses[bank] >= 0 && rows[bank] >= 0) {
            opened = 1;
            earliest = fr_max(earliest, bf[bank * 7 + 2]);
        }
    }
    if (opened) {
        double pre_start;
        earliest = fr_pre_ready(s, lane, earliest);
        if (s->channel_int[ch * 2 + 1] == lane)
            earliest = fr_max(earliest, fr_after(channel[ch * 2], 2 * t[0], half));
        pre_start = fr_align(fr_max(earliest, s->resources[ch * 4]), s->precharge_step);
        if (t[29] > 0) {
            for (bank = first_bank; bank < end_bank; ++bank)
                if (addresses[bank] >= 0 && rows[bank] >= 0 && pre_start > fr_after(bf[bank * 7], t[29], half))
                    fr_fail(FR_ERR_TRAS_REFRESH);
        }
        fr_command(s, ch, 0, earliest, t[2], 0, counts, &start);
        fr_trace(s, "PRE_ALL", lane, -1, -1, -1, -1, -1, start, -DBL_MAX, -DBL_MAX);
        counts[1] += 1;
        pc[lane * FR_PC] = start;
        earliest = fr_after(start, t[6], half);
        for (bank = first_bank; bank < end_bank; ++bank) {
            if (addresses[bank] >= 0 && rows[bank] >= 0) {
                rows[bank] = -1;
                bf[bank * 7 + 1] = earliest;
                bf[bank * 7 + 5] = earliest;
            }
        }
    }
    for (bank = first_bank; bank < end_bank; ++bank)
        if (addresses[bank] >= 0)
            earliest = fr_max(earliest, bf[bank * 7 + 1]);
    last_act = channel[ch * 2];
    if (last_act >= 0)
        earliest = fr_max(earliest, fr_after(last_act, 2 * t[0], half));
    if (g[8])
        earliest = fr_max(earliest, s->resources[(s->channels + ch) * 4]);
    fr_command(s, ch, 0, earliest, t[2], 1, counts, &start);
    fr_trace(s, "REFab", lane, -1, -1, -1, -1, -1, start, -DBL_MAX, -DBL_MAX);
    if (g[8]) {
        double ignored;
        fr_command(s, ch, 1, start, 1., 1, counts, &ignored);
    }
    if (start > due + 8 * t[20])
        fr_fail(FR_ERR_REFRESH_POSTPONED);
    pc[lane * FR_PC + 1] = fr_after(start, t[21], half);
    for (bank = first_bank; bank < end_bank; ++bank) {
        if (addresses[bank] >= 0) {
            rows[bank] = -1;
            bf[bank * 7 + 1] = pc[lane * FR_PC + 1];
            bf[bank * 7 + 5] = pc[lane * FR_PC + 1];
        }
    }
    counts[4] += 1;
}

/* Fold fully idle, closed-bank epochs only on an exact arithmetic grid.
 *
 * Under the bounds below every start, end, repeated busy-time addition and
 * refresh deadline is an exactly representable multiple of 0.25 ns. Non-dyadic
 * clocks keep the ordinary per-command loop.
 */
FR_NOINLINE int fr_idle_refresh(const fr_state *s, int64_t lane, double arrival, double *counts)
{
    const double *t = s->t;
    const int64_t *g = s->g, *rows = s->rows, *addresses = s->addresses;
    double *bf = s->bf, *pc = s->pc, *channel = s->channel, *resources = s->resources;
    int64_t ch = lane >> 1, kind, pseudo, bank, periods, commands;
    double interval = t[20], due = channel[ch * 2 + 1];
    double values[5], cutoff, last_due, last_start;
    int64_t index, kinds = g[8] ? 2 : 1;
    if (interval < t[21] + 2 * t[0])
        return 0;
    values[0] = t[0];
    values[1] = interval;
    values[2] = due;
    values[3] = arrival;
    values[4] = t[21];
    for (index = 0; index < 5; ++index) {
        double value = values[index];
        if (value < 0 || value >= FR_TWO49 || value * 2 != (double)fr_ifloor(value * 2))
            return 0;
    }
    cutoff = arrival - t[21] - 2 * t[0];
    periods = fr_ifloor(fr_divide(cutoff - due, interval)) + 1;
    if (periods < 2)
        return 0;
    for (bank = ch * 2 * s->banks; bank < (ch + 1) * 2 * s->banks; ++bank)
        if (addresses[bank] >= 0 && (rows[bank] >= 0 || bf[bank * 7 + 1] > due))
            return 0;
    if (fr_max(resources[ch * 4], resources[(s->channels + ch) * 4]) > due)
        return 0;
    for (pseudo = 0; pseudo < 2; ++pseudo)
        if (pc[(ch * 2 + pseudo) * FR_PC + 1] > due)
            return 0;
    for (kind = 0; kind < kinds; ++kind) {
        double busy = resources[(ch + kind * s->channels) * 4 + 1];
        if (busy >= FR_TWO49 || busy * 4 != (double)fr_ifloor(busy * 4))
            return 0;
    }
    last_due = due + (double)(periods - 1) * interval;
    last_start = fr_align(last_due, t[0]);
    commands = periods * 2;
    for (kind = 0; kind < kinds; ++kind) {
        int64_t rid = ch + kind * s->channels;
        double occupancy = (kind == 0 ? t[2] : 1.) * t[0];
        double start = last_start + t[0];
        double end = start + occupancy;
        resources[rid * 4] = end;
        resources[rid * 4 + 1] += (double)commands * occupancy;
        resources[rid * 4 + 2] = start;
        resources[rid * 4 + 3] = end;
        s->touched[rid] = 1;
        counts[5 + kind] += (double)commands * occupancy;
    }
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        int64_t target = ch * 2 + pseudo;
        s->pc_seen[target] = 1;
        pc[target * FR_PC + 1] = last_start + (double)pseudo * t[0] + t[21];
        for (bank = target * s->banks; bank < (target + 1) * s->banks; ++bank) {
            if (addresses[bank] >= 0) {
                bf[bank * 7 + 1] = pc[target * FR_PC + 1];
                bf[bank * 7 + 5] = pc[target * FR_PC + 1];
            }
        }
    }
    channel[ch * 2 + 1] = due + (double)periods * interval;
    counts[4] += (double)commands;
    return 1;
}

/* Keep each non-dyadic command/addition; publish closed-bank state once. */
FR_NOINLINE int fr_idle_refresh_commands(const fr_state *s, int64_t lane, double arrival, double *counts)
{
    const double *t = s->t;
    const int64_t *g = s->g, *rows = s->rows, *addresses = s->addresses;
    double *bf = s->bf, *pc = s->pc, *channel = s->channel;
    const double *resources = s->resources;
    int64_t ch = lane >> 1, row_id = ch, column_id = s->channels + ch, pseudo, bank, index;
    double clock = t[0], interval = t[20], cycle = t[21], due = channel[ch * 2 + 1];
    double half, cutoff, values[5], blocked0 = 0., blocked1 = 0.;
    int cycle_off_grid, interval_off_grid;
    if (!fr_isfinite(clock) || clock / 2 <= 0 || due < 0 || cycle < 0 || interval < cycle + 2 * clock)
        return 0;
    half = clock / 2;
    /* Exclude all clock-index exceptions before the first state mutation. */
    values[0] = arrival;
    values[1] = due;
    values[2] = interval;
    values[3] = cycle;
    values[4] = channel[ch * 2];
    for (index = 0; index < 5; ++index)
        if (!fr_isfinite(values[index]) || fr_abs(values[index] / half) >= FR_TWO49)
            return 0;
    cutoff = arrival - cycle - 2 * clock;
    if (due + interval > cutoff || t[2] != (g[7] ? 1. : .5))
        return 0;
    if (!(resources[row_id * 4] <= due && resources[column_id * 4] <= due))
        return 0;
    for (pseudo = 0; pseudo < 2; ++pseudo)
        if (!(pc[(ch * 2 + pseudo) * FR_PC + 1] <= due))
            return 0;
    for (bank = ch * 2 * s->banks; bank < (ch + 1) * 2 * s->banks; ++bank)
        if (addresses[bank] >= 0 && (rows[bank] >= 0 || !(bf[bank * 7 + 1] <= due)))
            return 0;
    if (channel[ch * 2] >= 0 && fr_after(channel[ch * 2], 2 * clock, half) > due)
        return 0;
    /* Non-grid delays always take clock_after's ordinary IEEE addition.
     * The scope proof above already excludes clock-index exceptions. */
    cycle_off_grid = cycle != (double)fr_round(cycle / half) * half;
    interval_off_grid = interval != (double)fr_round(interval / half) * half;
    while (due <= cutoff) {
        double next_due;
        for (pseudo = 0; pseudo < 2; ++pseudo) {
            double earliest = g[8] ? fr_max(due, resources[column_id * 4]) : due;
            double start, blocked;
            fr_command(s, ch, 0, earliest, t[2], 1, counts, &start);
            fr_trace(s, "REFab", 2 * ch + pseudo, -1, -1, -1, -1, -1, start, -DBL_MAX, -DBL_MAX);
            if (g[8]) {
                double ignored;
                fr_command(s, ch, 1, start, 1., 1, counts, &ignored);
            }
            blocked = cycle_off_grid ? start + cycle : fr_after(start, cycle, half);
            if (pseudo == 0)
                blocked0 = blocked;
            else
                blocked1 = blocked;
            counts[4] += 1;
        }
        next_due = interval_off_grid ? due + interval : fr_after(due, interval, half);
        channel[ch * 2 + 1] = next_due;
        /* Preserve this successful epoch, then let the full controller handle
         * overlap. Returning false here would replay commands and busy time. */
        if (blocked0 > next_due || blocked1 > next_due
                || resources[row_id * 4] > next_due || resources[column_id * 4] > next_due)
            break;
        due = next_due;
    }
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        int64_t target = ch * 2 + pseudo;
        s->pc_seen[target] = 1;
        pc[target * FR_PC + 1] = pseudo == 0 ? blocked0 : blocked1;
        for (bank = target * s->banks; bank < (target + 1) * s->banks; ++bank) {
            if (addresses[bank] >= 0) {
                bf[bank * 7 + 1] = pc[target * FR_PC + 1];
                bf[bank * 7 + 5] = pc[target * FR_PC + 1];
            }
        }
    }
    return 1;
}

FR_NOINLINE int fr_refresh_until(const fr_state *s, int64_t lane, double horizon, double arrival, double *counts)
{
    const double *t = s->t;
    double *channel = s->channel;
    int64_t ch = lane >> 1;
    int refreshed = 0;
    if (!s->g[8])
        return 0;
    if (channel[ch * 2 + 1] < 0)
        channel[ch * 2 + 1] = fr_max(t[20], (double)fr_ifloor(fr_divide(arrival, t[20])) * t[20]);
    while (channel[ch * 2 + 1] <= horizon) {
        double due;
        if (fr_idle_refresh(s, lane, arrival, counts) || fr_idle_refresh_commands(s, lane, arrival, counts)) {
            refreshed = 1;
            continue;
        }
        due = channel[ch * 2 + 1];
        fr_refresh_pc(s, ch * 2, due, counts);
        fr_refresh_pc(s, ch * 2 + 1, due, counts);
        channel[ch * 2 + 1] = fr_after(channel[ch * 2 + 1], t[20], s->half);
        refreshed = 1;
    }
    return refreshed;
}

/* memory_mapping.rotated_bank: alternate groups inside a 16-bank SID. */
/* Bank within the lane and activated row of a cursor: burst | BG-in-SID | lane |
 * column | bank-in-group | SID | row. */
FR_NOINLINE void fr_burst_location(const int64_t *g, int64_t cursor, int64_t *bank_within, int64_t *row)
{
    int64_t bg_in_sid = fr_mod(cursor, g[15]), bank_in_group, sid;
    cursor = fr_floordiv(cursor, g[15]);
    cursor = fr_floordiv(cursor, g[0]);
    cursor = fr_floordiv(cursor, g[3]);
    bank_in_group = fr_mod(cursor, g[4]);
    cursor = fr_floordiv(cursor, g[4]);
    sid = fr_mod(cursor, g[14]);
    *row = fr_floordiv(cursor, g[14]);
    *bank_within = sid * g[15] * g[4] + bg_in_sid * g[4] + bank_in_group;
}

static int64_t fr_binary_shift(int64_t value)
{
    int64_t shift = 0;
    if (value <= 0 || (value & (value - 1)))
        return -1;
    while (value > 1) {
        value >>= 1;
        shift += 1;
    }
    return shift;
}

/* The last terms of a column start: the data bus, then the clock edge. */
FR_INLINE double fr_column_edge(const fr_state *s, int64_t lane, int read, int64_t data, double ready)
{
    const double *t = s->t;
    const int64_t *ticks = s->ticks;
    const double *lane_pc = s->pc + lane * FR_PC;
    double half = s->half, inverse = s->inverse;
    double data_ready = s->resources[data * 4], start;
    int64_t direction = read ? 1 : 2, latency = read ? 22 : 23;
    if (s->a->directions[data] && s->a->directions[data] != direction) {
        int64_t turnaround = read ? 25 : 24;
        data_ready = fr_after_tick(data_ready, t[turnaround], ticks[turnaround], half, inverse);
    }
    data_ready = fr_max(data_ready, lane_pc[read ? 2 : 3]);
    start = fr_align(fr_max(ready, fr_after_tick(data_ready, -t[latency], fr_negative_tick(ticks[latency]),
                                                 half, inverse)), t[0]);
    if (fr_after_tick(start, t[latency], ticks[latency], half, inverse) < data_ready)
        start = fr_after_tick(start, t[0], ticks[0], half, inverse);
    return start;
}

/* _column_start over precomputed group tables, for any history.
 *
 * Every term only raises ``ready``. A term whose plain sum lies half a clock
 * below it cannot pass it, so its exact clock addition is not computed.
 */
FR_NOINLINE double fr_column_start_general(const fr_state *s, int64_t lane, int64_t group, int64_t sid, double bank_ready,
                                       double earliest, int read, int64_t column_bus, int64_t data)
{
    const int64_t *entry_group = s->entry_group, *entry_sid = s->entry_sid;
    const double *t = s->t;
    const int64_t *ticks = s->ticks;
    const double *resources = s->resources;
    const double *lane_pc = s->pc + lane * FR_PC;
    const double *lane_history = s->history + lane * s->entries * 4;
    double half = s->half, inverse = s->inverse, grid_limit = s->grid_limit;
    int grouped = s->grouped;
    int64_t entries = s->entries, entry, op_index = read ? 1 : 2;
    double ready = fr_max(fr_max(fr_max(earliest, bank_ready), lane_pc[1]), resources[column_bus * 4]);
    double same = -1., different = -1., different_sid = -1., previous_read = -1.;
    for (entry = 0; entry < entries; ++entry) {
        int64_t previous_group = entry_group[entry];
        double stamp = lane_history[entry * 4 + op_index];
        if (stamp >= 0) {
            if (grouped && stamp < grid_limit) {
                if (read && entry_sid[entry] != sid)
                    different_sid = fr_max(different_sid, stamp);
                else if (previous_group == group)
                    same = fr_max(same, stamp);
                else
                    different = fr_max(different, stamp);
            } else {
                double gap = previous_group == group ? t[11] : t[12];
                if (read && entry_sid[entry] != sid)
                    gap = t[13];
                ready = fr_max(ready, fr_after(stamp, gap, half));
            }
        }
        if (read) {
            double end = lane_history[entry * 4 + 3];
            if (end >= 0) {
                int64_t index = previous_group == group ? 16 : 17;
                if (end + t[index] + half > ready)
                    ready = fr_max(ready, t[index] == 0. ? end
                                   : fr_after_tick(end, t[index], ticks[index], half, inverse));
            }
        } else {
            double previous = lane_history[entry * 4 + 1];
            if (previous >= 0) {
                if (grouped && previous < grid_limit)
                    previous_read = fr_max(previous_read, previous);
                else
                    ready = fr_max(ready, fr_after(previous, t[18], half));
            }
        }
    }
    if (same >= 0 && same + t[11] + half > ready)
        ready = fr_max(ready, fr_after_tick(same, t[11], ticks[11], half, inverse));
    if (different >= 0 && different + t[12] + half > ready)
        ready = fr_max(ready, fr_after_tick(different, t[12], ticks[12], half, inverse));
    if (different_sid >= 0 && different_sid + t[13] + half > ready)
        ready = fr_max(ready, fr_after_tick(different_sid, t[13], ticks[13], half, inverse));
    if (previous_read >= 0 && previous_read + t[18] + half > ready)
        ready = fr_max(ready, t[18] == 0. ? previous_read
                       : fr_after_tick(previous_read, t[18], ticks[18], half, inverse));
    return fr_column_edge(s, lane, read, data, ready);
}

/* A column start when the lane's stamps are on the clock grid: each class of
 * history entries reduces to its latest stamp before the clock math, because
 * clock_after is then monotone for a fixed gap. ``end_top`` bounds the lane's
 * write ends, which rarely reach ``ready`` and are then not visited.
 */
FR_INLINE double fr_column_start(const fr_state *s, int64_t lane, int64_t within, int64_t group, int64_t sid,
                                 double bank_ready, double earliest, int read, int64_t column_bus, int64_t data,
                                 double end_top)
{
    const double *t = s->t;
    const int64_t *ticks = s->ticks;
    const double *lane_history = s->history + lane * s->entries * 4;
    const int64_t *order = (read ? s->read_order : s->write_order) + within * s->entries;
    const int64_t *splits = (read ? s->read_splits : s->write_splits) + within * 2;
    double half = s->half, inverse = s->inverse;
    int64_t entries = s->entries, position, first_split = splits[0], second_split = splits[1];
    const double *stamps = lane_history + (read ? 1 : 2);
    double ready, same = -1., different = -1., different_sid = -1., top;
    if (!s->grouped)
        return fr_column_start_general(s, lane, group, sid, bank_ready, earliest, read, column_bus, data);
    for (position = 0; position < first_split; ++position)
        same = fr_max(same, stamps[order[position]]);
    for (; position < second_split; ++position)
        different = fr_max(different, stamps[order[position]]);
    for (; position < entries; ++position)
        different_sid = fr_max(different_sid, stamps[order[position]]);
    top = fr_max(fr_max(same, different), different_sid);
    if (top >= s->grid_limit)
        return fr_column_start_general(s, lane, group, sid, bank_ready, earliest, read, column_bus, data);
    ready = fr_max(fr_max(fr_max(earliest, bank_ready), s->pc[lane * FR_PC + 1]), s->resources[column_bus * 4]);
    if (read) {
        if (end_top + s->end_gap + half > ready) {
            for (position = 0; position < entries; ++position) {
                double end = lane_history[position * 4 + 3];
                if (end >= 0) {
                    int64_t index = s->entry_group[position] == group ? 16 : 17;
                    if (end + t[index] + half > ready)
                        ready = fr_max(ready, t[index] == 0. ? end
                                       : fr_after_tick(end, t[index], ticks[index], half, inverse));
                }
            }
        }
    } else {
        double previous_read = -1.;
        for (position = 0; position < entries; ++position)
            previous_read = fr_max(previous_read, lane_history[position * 4 + 1]);
        if (previous_read >= s->grid_limit)
            return fr_column_start_general(s, lane, group, sid, bank_ready, earliest, read, column_bus, data);
        if (previous_read >= 0 && previous_read + t[18] + half > ready)
            ready = fr_max(ready, t[18] == 0. ? previous_read
                           : fr_after_tick(previous_read, t[18], ticks[18], half, inverse));
    }
    if (same >= 0 && same + t[11] + half > ready)
        ready = fr_max(ready, fr_after_tick(same, t[11], ticks[11], half, inverse));
    if (different >= 0 && different + t[12] + half > ready)
        ready = fr_max(ready, fr_after_tick(different, t[12], ticks[12], half, inverse));
    if (different_sid >= 0 && different_sid + t[13] + half > ready)
        ready = fr_max(ready, fr_after_tick(different_sid, t[13], ticks[13], half, inverse));
    return fr_column_edge(s, lane, read, data, ready);
}

/* Drop ACT entries the row bus frontier has expired, and return when the oldest
 * remaining entry expires: until the frontier reaches it this changes nothing. */
FR_NOINLINE double fr_faw_prune(const fr_state *s, int64_t lane, double frontier)
{
    double width = s->t[30], half = s->half, inverse = s->inverse;
    int64_t width_tick = s->ticks[30];
    double *row = s->windows + lane * s->window_capacity;
    int64_t count, expired = 0, index;
    if (width <= 0)
        return HUGE_VAL;
    count = s->window_counts[lane];
    while (expired < count && fr_after_tick(row[expired], width, width_tick, half, inverse) <= frontier)
        expired += 1;
    if (expired) {
        for (index = 0; index < count - expired; ++index)
            row[index] = row[index + expired];
        count -= expired;
        s->window_counts[lane] = count;
    }
    return count ? fr_after_tick(row[0], width, width_tick, half, inverse) : HUGE_VAL;
}

/* _faw_ready on a precomputed window tick: only consecutive five-ACT windows containing the insertion. */
static double fr_faw_ready(const fr_state *s, int64_t lane, double earliest)
{
    double width = s->t[30], half = s->half, inverse = s->inverse;
    int64_t width_tick = s->ticks[30];
    const double *row = s->windows + lane * s->window_capacity;
    int64_t count;
    if (width <= 0)
        return earliest;
    count = s->window_counts[lane];
    for (;;) {
        int64_t position = 0, offset, stop;
        double next_start = earliest;
        while (position < count && row[position] <= earliest)
            position += 1;
        stop = fr_imin(position, count - 4) + 1;
        for (offset = fr_imax(0, position - 4); offset < stop; ++offset) {
            double first = offset == position ? earliest : row[offset];
            int64_t last_index = offset + 4;
            double last = last_index == position ? earliest : row[last_index - 1];
            if (last < fr_after_tick(first, width, width_tick, half, inverse))
                next_start = fr_max(next_start, fr_after_tick(row[offset], width, width_tick, half, inverse));
        }
        if (next_start == earliest)
            return earliest;
        earliest = next_start;
    }
}

/* tRRD is bank-independent within a group and one plan. Evaluate its
 * existing clock operations once for each group actually inspected. */
FR_INLINE double fr_activate_group_gate(const fr_state *s, int64_t lane, int64_t group)
{
    const double *t = s->t;
    const int64_t *ticks = s->ticks;
    const double *lane_history = s->history + lane * s->entries * 4;
    double half = s->half, inverse = s->inverse, grid_limit = s->grid_limit;
    double gate = -1., other = -1.;
    int64_t groups = s->g[5], previous_group;
    int grouped = s->grouped;
    if (s->act_gate_ready[group])
        return s->act_gates[group];
    for (previous_group = 0; previous_group < groups; ++previous_group) {
        double stamp = lane_history[previous_group * 4];
        int same_group = previous_group == group;
        double gap = same_group ? t[14] : t[15];
        if (stamp >= 0 && gap > 0) {
            if (grouped && stamp < grid_limit && !same_group)
                other = fr_max(other, stamp);
            else
                gate = fr_max(gate, fr_after_tick(stamp, gap, ticks[same_group ? 14 : 15], half, inverse));
        }
    }
    if (other >= 0)
        gate = fr_max(gate, fr_after_tick(other, t[15], ticks[15], half, inverse));
    s->act_gates[group] = gate;
    s->act_gate_ready[group] = 1;
    return gate;
}

/* The ACT edge from a ready time that already holds the bank's own terms, under tRRD
 * gate ``gate``. ``base`` gets the anchor before the gate, ``fixed`` the anchor the
 * tFAW loop settles on. The edge never falls when ``ready`` or ``gate`` rises: every
 * step is a max, a round up to the clock, or the earliest tFAW slot no earlier than its
 * argument; the loop runs that inflationary map to its least fixed point. */
FR_INLINE double fr_activate_from(const fr_state *s, int64_t lane, double gate, double ready, double *base,
                                  double *fixed)
{
    const double *t = s->t;
    const int64_t *ticks = s->ticks;
    const double *lane_pc = s->pc + lane * FR_PC;
    double half = s->half, inverse = s->inverse;
    double anchor = fr_max(fr_after_tick(fr_align(ready, t[0]), t[0], ticks[0], half, inverse), lane_pc[4]);
    *base = anchor;
    anchor = fr_max(anchor, gate);
    for (;;) {
        double start, aligned;
        anchor = fr_faw_ready(s, lane, anchor);
        start = fr_align(fr_after_tick(anchor, -t[0], s->clock_back, half, inverse), t[0]);
        aligned = fr_after_tick(start, t[0], ticks[0], half, inverse);
        if (t[30] <= 0 || aligned == anchor) {
            *fixed = t[30] <= 0 ? *base : anchor;
            return start;
        }
        anchor = aligned;
    }
}

/* _activate_start for an entry already on the queue, on precomputed ticks.
 * The row bus alone bounds later ACTs, so the caller has pruned tFAW history to it. */
FR_INLINE double fr_activate_start(const fr_state *s, int64_t lane, int64_t bank, int64_t group, double earliest)
{
    const double *t = s->t;
    const int64_t *ticks = s->ticks;
    const double *bf = s->bf, *lane_pc = s->pc + lane * FR_PC;
    double half = s->half, inverse = s->inverse;
    double row_bus = s->resources[(lane >> 1) * 4];
    double ready = fr_max(fr_max(fr_max(earliest, bf[bank * 7 + 1]), lane_pc[1]), row_bus);
    double base, fixed;
    if (bf[bank * 7] >= 0)
        ready = fr_max(ready, fr_after_tick(fr_after_tick(bf[bank * 7], t[8], ticks[8], half, inverse),
                                            -t[0], s->clock_back, half, inverse));
    return fr_activate_from(s, lane, fr_activate_group_gate(s, lane, group), ready, &base, &fixed);
}

/* The lowest ACT edge any closed bank of ``group`` can have in this row scan: the edge
 * from the lane's bank-free ready time. Every bank's own terms only raise its ready time,
 * so its edge is at least this. Slot ``groups`` holds the edge without a gate and
 * ``groups + 1`` its fixed point; a gate at or below that point leaves the edge as is. */
FR_INLINE double fr_activate_floor(const fr_state *s, int64_t lane, int64_t group, double act_floor)
{
    int64_t groups = s->g[5];
    double *floors = s->act_floors, gate, base, fixed;
    if (s->act_floor_ready[group])
        return floors[group];
    if (!s->act_floor_ready[groups]) {
        double ready = fr_max(fr_max(act_floor, s->pc[lane * FR_PC + 1]), s->resources[(lane >> 1) * 4]);
        floors[groups] = fr_activate_from(s, lane, -HUGE_VAL, ready, &base, &floors[groups + 1]);
        s->act_floor_ready[groups] = 1;
    }
    gate = fr_activate_group_gate(s, lane, group);
    if (gate <= floors[groups + 1]) {
        floors[group] = floors[groups];
    } else {
        double ready = fr_max(fr_max(act_floor, s->pc[lane * FR_PC + 1]), s->resources[(lane >> 1) * 4]);
        floors[group] = fr_activate_from(s, lane, gate, ready, &base, &fixed);
    }
    s->act_floor_ready[group] = 1;
    return floors[group];
}

/* Exact queued PRE/ACT edge, unless a conservative lower bound already loses.
 * A tie loses too: the column or older bank was visited first. */
FR_INLINE double fr_queued_row_start(const fr_state *s, int64_t lane, int64_t bank, int64_t group,
                                      double entry, double act_floor, double best, double *faw_expiry,
                                      double plain_limit)
{
    const double *t = s->t, *bf = s->bf + bank * 7, *pc = s->pc + lane * FR_PC;
    double row_bus = s->resources[(lane >> 1) * 4];
    double row_floor;
    if (s->rows[bank] >= 0) {
        row_floor = fr_max(fr_max(fr_max(entry, bf[2]), pc[1]), row_bus);
        if (row_floor >= best)
            return HUGE_VAL;
        return fr_precharge_start(s, lane, bank, entry);
    }
    row_floor = fr_max(fr_max(fr_max(fr_max(entry, act_floor), bf[1]), pc[1]), row_bus);
    /* Only the row-port frontier can discard old tFAW entries. */
    if (*faw_expiry <= fr_max(0., row_bus))
        *faw_expiry = fr_faw_prune(s, lane, fr_max(0., row_bus));
    if (bf[0] >= 0)
        row_floor = fr_max(row_floor, bf[0] + t[8] - t[0]);
    if (t[30] > 0 && s->window_counts[lane] >= 4) {
        const double *window = s->windows + lane * s->window_capacity;
        int64_t live = s->window_counts[lane];
        if (row_floor >= window[live - 1])
            row_floor = fr_max(row_floor, window[live - 4] + t[30] - t[0]);
    }
    /* The half-clock margin covers rounding of the plain sums below 2**44
     * half clocks; outside that range use the exact clock helpers. */
    if (row_floor - s->half >= best && row_floor < plain_limit)
        return HUGE_VAL;
    /* A tie loses too, so a group floor at or after ``best`` settles every bank of it. */
    if (fr_activate_floor(s, lane, group, act_floor) >= best)
        return HUGE_VAL;
    return fr_activate_start(s, lane, bank, group, fr_max(entry, act_floor));
}

/* First burst of [first, stop) on a lane; a stripe holds one burst of each bank group of a SID. */
FR_INLINE int64_t fr_lane_cursor(int64_t first, int64_t stop, int64_t lane, int64_t lanes, int64_t stripe_shift)
{
    int64_t block = first >> stripe_shift;
    int64_t offset = fr_mod(lane - block, lanes);
    int64_t cursor = offset == 0 ? first : (block + offset) << stripe_shift;
    return cursor < stop ? cursor : -1;
}

/* The next request of one direction reaching this lane, with its burst. */
static void fr_next_request(const fr_args *a, int64_t start, int64_t direction, int64_t lane, int64_t lanes,
                            int64_t *request_out, int64_t *cursor_out)
{
    int64_t request;
    for (request = start; request < a->requests; ++request) {
        if ((a->reads[request] != 0) == (direction == 0)) {
            int64_t cursor = fr_lane_cursor(a->firsts[request], a->stops[request], lane, lanes, a->stripe_shift);
            if (cursor >= 0) {
                *request_out = request;
                *cursor_out = cursor;
                return;
            }
        }
    }
    *request_out = -1;
    *cursor_out = -1;
}

/* One entry of a scheduling queue; ``within`` is its bank within the lane. */
typedef struct fr_item {
    int64_t within, row, column, request;
    double entry;
} fr_item;

/* One FIFO scheduling queue of a PC: its entries in ring slots, age order from
 * head. Bit i of ``hits`` says whether the i-th oldest entry's row is open, so
 * planning visits row hits and row misses without scanning the queue. Bit i of
 * ``firsts`` says whether the i-th oldest entry is its bank's oldest row miss:
 * a row command only ever serves that entry, so planning visits one per bank. */
typedef struct fr_queue {
    fr_item *items;
    uint64_t *hits, *firsts;
    int64_t *pending_hit, *queued;
    int64_t head, count, bypass, iter_request, iter_cursor;
    double slot_free;
    int dirty;
} fr_queue;

FR_INLINE int64_t fr_lowest_bit(uint64_t value)
{
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, value);
    return (int64_t)index;
#else
    return __builtin_ctzll(value);
#endif
}

FR_INLINE void fr_set_bit(uint64_t *bits, int64_t offset)
{
    bits[offset >> 6] |= UINT64_C(1) << (offset & 63);
}

/* Remove one age position: younger positions move down by one. */
FR_INLINE void fr_remove_bit(uint64_t *bits, int64_t words, int64_t offset)
{
    int64_t word = offset >> 6;
    uint64_t below = (UINT64_C(1) << (offset & 63)) - 1;
    uint64_t value = bits[word];
    value = (value & below) | ((value >> 1) & ~below);
    for (; word + 1 < words; ++word) {
        value |= bits[word + 1] << 63;
        bits[word] = value;
        value = bits[word + 1] >> 1;
    }
    bits[word] = value;
}

/* Point ``firsts`` of one bank at its oldest entry whose row is not ``open_row``
 * (-1: the bank is closed, so its oldest entry). */
static void fr_first_of_bank(fr_queue *queue, int64_t capacity, int64_t within, int64_t open_row)
{
    int64_t position = queue->head, index, found = 0;
    for (index = 0; index < queue->count; ++index) {
        const fr_item *item = &queue->items[position];
        if (item->within == within) {
            queue->firsts[index >> 6] &= ~(UINT64_C(1) << (index & 63));
            if (!found && item->row != open_row) {
                queue->firsts[index >> 6] |= UINT64_C(1) << (index & 63);
                found = 1;
            }
        }
        position += 1;
        if (position == capacity)
            position = 0;
    }
}

/* Rebuild each direction's open-row hits after maintenance closed rows. */
FR_NOINLINE void fr_recount(const fr_state *s, int64_t ch, fr_queue queues[2][2], int64_t hit_words)
{
    const int64_t *q = s->q;
    int64_t pseudo, direction, index;
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        const int64_t *lane_rows = s->rows + (2 * ch + pseudo) * s->banks;
        for (direction = 0; direction < 2; ++direction) {
            fr_queue *queue = &queues[pseudo][direction];
            int64_t slot = queue->head;
            queue->dirty = 1;
            memset(queue->pending_hit, 0, (size_t)s->banks * sizeof(int64_t));
            memset(queue->hits, 0, (size_t)hit_words * sizeof(uint64_t));
            memset(queue->firsts, 0, (size_t)hit_words * sizeof(uint64_t));
            memset(s->miss_seen, 0, (size_t)((s->banks + 63) >> 6) * sizeof(uint64_t));
            for (index = 0; index < queue->count; ++index) {
                const fr_item *item = &queue->items[slot];
                if (lane_rows[item->within] == item->row) {
                    queue->pending_hit[item->within] += 1;
                    fr_set_bit(queue->hits, index);
                } else if (!(s->miss_seen[item->within >> 6] & (UINT64_C(1) << (item->within & 63)))) {
                    s->miss_seen[item->within >> 6] |= UINT64_C(1) << (item->within & 63);
                    fr_set_bit(queue->firsts, index);
                }
                slot += 1;
                if (slot == q[direction])
                    slot = 0;
            }
        }
    }
}

/* A closed bank's queued hits leave both directions' bitsets. */
static void fr_forget_bank(fr_queue *lane_queues, const int64_t *q, int64_t hit_words, int64_t within)
{
    int64_t side;
    for (side = 0; side < 2; ++side) {
        fr_queue *other = &lane_queues[side];
        if (other->pending_hit[within] > 0) {
            /* Only this bank's hits leave the bitset. */
            int64_t word;
            for (word = 0; word < hit_words; ++word) {
                uint64_t rest = other->hits[word];
                while (rest) {
                    int64_t offset = word * 64 + fr_lowest_bit(rest), position = other->head + offset;
                    rest &= rest - 1;
                    if (position >= q[side])
                        position -= q[side];
                    if (other->items[position].within == within)
                        other->hits[word] &= ~(UINT64_C(1) << (offset & 63));
                }
            }
            other->pending_hit[within] = 0;
        }
        if (other->queued[within] > 0)
            fr_first_of_bank(other, q[side], within, -1);
        other->dirty = 1;
    }
}

/* The first burst of a bank: what a bank first published for its refresh names. */
static int64_t fr_bank_cursor(const int64_t *g, int64_t lane, int64_t within)
{
    int64_t per_sid = g[15] * g[4];
    int64_t sid = within / per_sid, rest = within % per_sid;
    return ((sid * g[4] + rest % g[4]) * g[3] * g[0] + lane) * g[15] + rest / g[4];
}

FR_INLINE int64_t fr_bits16(uint64_t value)
{
    int64_t count = 0;
    for (value &= 0xFFFF; value; value &= value - 1)
        count += 1;
    return count;
}

/* The bank a lane's due REFpb takes. Within an SID's set of 16 the order is the
 * controller's choice: a bank with no queued work goes first, then one whose
 * queued entries all need a row command anyway, and a bank with queued row hits
 * last; ties go to the lowest bank. Without queues this is plain bank order. */
FR_NOINLINE int64_t fr_refresh_pick(const fr_state *s, int64_t lane, double due, const fr_queue *lane_queues)
{
    int64_t base = fr_mod(fr_round(due / s->t[33]), s->banks) / 16 * 16, slot, best = -1, best_rank = 3;
    uint64_t done = (uint64_t)s->pc[lane * FR_PC + 7];
    for (slot = 0; slot < 16; ++slot) {
        int64_t within = base + slot, rank = 0;
        if (done >> slot & 1)
            continue;
        if (lane_queues) {
            if (lane_queues[0].pending_hit[within] > 0 || lane_queues[1].pending_hit[within] > 0)
                rank = 2;
            else if (lane_queues[0].queued[within] > 0 || lane_queues[1].queued[within] > 0)
                rank = 1;
        }
        if (rank < best_rank) {
            best = within;
            best_rank = rank;
            if (!rank)
                break;
        }
    }
    return best;
}

/* REFpb from step ``first`` until bank ``within`` is refreshed, in bank order
 * from a set of which ``done`` is already refreshed. */
static int64_t fr_refresh_turn(uint64_t done, int64_t first, int64_t banks, int64_t within)
{
    int64_t slot = fr_mod(first, 16), base = fr_mod(first, banks) / 16 * 16;
    if (base <= within && within < base + 16) {
        int64_t bit = within - base;
        if (done >> bit & 1)
            return 16 - slot + banks - 16 + bit;
        return bit - fr_bits16(done & ((UINT64_C(1) << bit) - 1));
    }
    return 16 - slot + fr_mod(within - base - 16, banks);
}

/* A REFpb's rising edge. It needs the bank closed past tRP and tRC, tRREFD from
 * the lane's last REFpb, the tRFCpb gate of the last completed set, and the ACT
 * gaps and tFAW, which count it as an ACT. */
FR_NOINLINE double fr_refresh_bank_start(const fr_state *s, int64_t lane, int64_t within, double earliest)
{
    const double *t = s->t;
    const int64_t *ticks = s->ticks;
    const double *bank_bf = s->bf + (lane * s->banks + within) * 7, *lane_pc = s->pc + lane * FR_PC;
    const double *lane_history = s->history + lane * s->entries * 4;
    double half = s->half, inverse = s->inverse, row_bus = s->resources[(lane >> 1) * 4];
    double ready = fr_max(fr_max(fr_max(fr_max(fr_max(earliest, bank_bf[1]), lane_pc[1]), lane_pc[4]), lane_pc[6]), row_bus);
    double anchor;
    int64_t group = within / s->g[4], previous;
    if (bank_bf[0] >= 0)
        ready = fr_max(ready, fr_after_tick(bank_bf[0], t[8], ticks[8], half, inverse));
    anchor = fr_align(ready, t[0]);
    for (previous = 0; previous < s->g[5]; ++previous) {
        double stamp = lane_history[previous * 4], gap = previous == group ? t[14] : t[15];
        if (stamp >= 0 && gap > 0)
            anchor = fr_max(anchor, fr_after_tick(stamp, gap, ticks[previous == group ? 14 : 15], half, inverse));
    }
    for (;;) {
        double start;
        anchor = fr_faw_ready(s, lane, anchor);
        start = fr_align(anchor, t[0]);
        if (t[30] <= 0 || start == anchor)
            return start;
        anchor = start;
    }
}

/* Issue a lane's next REFpb and move its due time on. */
FR_NOINLINE void fr_refresh_bank(const fr_state *s, int64_t lane, int64_t within, double earliest, double *counts)
{
    const double *t = s->t;
    int64_t ch = lane >> 1, bank = lane * s->banks + within;
    double *lane_pc = s->pc + lane * FR_PC, *bank_bf = s->bf + bank * 7;
    double due = lane_pc[5], start, done;
    start = fr_row_command(s, ch, fr_max(earliest, lane_pc[6]), t[0], s->pre_duration, s->pre_duration_tick, counts);
    fr_trace(s, "REFpb", lane, within / (s->g[15] * s->g[4]), within / s->g[4], within, -1, -1, start, -DBL_MAX, -DBL_MAX);
    counts[4] += 1;
    if (start > due + 8 * t[20])
        fr_fail(FR_ERR_REFRESH_POSTPONED);
    lane_pc[4] = fr_after_tick(start, t[32], s->ticks[32], s->half, s->inverse);
    done = fr_after_tick(start, t[31], s->ticks[31], s->half, s->inverse);
    bank_bf[1] = done;
    bank_bf[5] = done;
    {
        uint64_t set = (uint64_t)lane_pc[7] | (UINT64_C(1) << (within & 15));
        if (set == 0xFFFF) {
            /* The set is complete: no REFpb on this PC for tRFCpb. */
            set = 0;
            lane_pc[6] = fr_after_tick(start, t[31], s->ticks[31], s->half, s->inverse);
        }
        lane_pc[7] = (double)set;
    }
    if (t[30] > 0) {
        fr_faw_prune(s, lane, fr_max(0., s->resources[ch * 4]));
        fr_record_activate(s, lane, start);
    }
    lane_pc[5] = (double)(fr_round(due / t[33]) + 1) * t[33];
}

/* Whether a lane's due REFpb, or the PRE before it, replaces the plan it was
 * compared with. It goes when no later than the plan, and ahead of a row command
 * once its bank is ready: an ACT's anchor lies a clock after its start, so a
 * REFpb under the same gaps would otherwise lose to every ACT. An open row with
 * queued hits stays open for them. A REFpb one tREFI late takes the lane.
 *
 * Returns the plan start below which this cannot change. The times it compares
 * only rise, and the bank chosen stays, until a command or a new queue entry
 * reaches a bank of the set, which resets the caller's copy.
 */
FR_NOINLINE double fr_plan_refresh(const fr_state *s, int64_t lane, const fr_queue *lane_queues,
                                   double *plan_start, int64_t *plan_kind, int64_t *plan_offset, int64_t *target)
{
    const double *t = s->t;
    const double *lane_pc = s->pc + lane * FR_PC;
    double planned = *plan_start, due = lane_pc[5], late = due + t[20], when, floor_;
    int64_t within = fr_refresh_pick(s, lane, due, lane_queues), bank = lane * s->banks + within, kind;
    int take;
    *target = within;
    if (s->rows[bank] >= 0) {
        kind = 4;
        if (lane_queues[0].pending_hit[within] > 0 || lane_queues[1].pending_hit[within] > 0) {
            if (late > planned)
                return late;
            take = 0;
            when = fr_precharge_start(s, lane, bank, due);
            floor_ = late;
        } else {
            when = fr_precharge_start(s, lane, bank, due);
            take = when <= planned;
            floor_ = fr_min(when, late);
        }
    } else {
        const double *bank_bf = s->bf + bank * 7;
        double ready = fr_max(fr_max(fr_max(due, bank_bf[1]), lane_pc[4]), lane_pc[6]);
        if (bank_bf[0] >= 0)
            ready = fr_max(ready, fr_after_tick(bank_bf[0], t[8], s->ticks[8], s->half, s->inverse));
        kind = 3;
        when = fr_refresh_bank_start(s, lane, within, due);
        take = when <= planned || ((*plan_kind == 1 || *plan_kind == 2) && ready <= planned);
        floor_ = fr_min(ready, late);
    }
    if (take || late <= planned) {
        *plan_start = when;
        *plan_kind = kind;
        *plan_offset = within;
        return due;
    }
    return floor_;
}

/* From this time on nothing the channel has done can delay a REFpb or its PRE. */
FR_NOINLINE double fr_refresh_floor(const fr_state *s, int64_t ch)
{
    const double *t = s->t;
    const double *history = s->history + 2 * ch * s->entries * 4;
    double gap = fr_max(t[14], t[15]);
    double floor_ = fr_max(s->resources[ch * 4], s->channel[ch * 2] + 2 * t[0]);
    int64_t lane, bank, entry;
    for (lane = 2 * ch; lane < 2 * ch + 2; ++lane) {
        const double *lane_pc = s->pc + lane * FR_PC;
        int64_t count = s->window_counts[lane];
        floor_ = fr_max(fr_max(fr_max(fr_max(floor_, lane_pc[1]), lane_pc[4]), lane_pc[6]), lane_pc[0] + t[19]);
        if (count)
            floor_ = fr_max(floor_, s->windows[lane * s->window_capacity + count - 1] + t[30]);
    }
    for (entry = 0; entry < 2 * s->entries; ++entry)
        floor_ = fr_max(floor_, history[entry * 4] + gap);
    for (bank = 2 * ch * s->banks; bank < (2 * ch + 2) * s->banks; ++bank) {
        const double *bank_bf = s->bf + bank * 7;
        floor_ = fr_max(fr_max(floor_, bank_bf[1]), bank_bf[0] + t[8]);
        if (s->rows[bank] >= 0)
            floor_ = fr_max(floor_, bank_bf[2]);
    }
    return floor_;
}

/* The tRFCpb gate left by the last completed group in a skipped interval.
 * Idle-gap skipping does not issue each command, so retain the same group
 * boundary state that the command loop would have produced.  There are at
 * most one such boundary every 16 banks; this is outside the hot per-command
 * path and keeps the exact clock-grid alignment of the real command. */
static double fr_refresh_skip_group_gate(const fr_state *s, int64_t lane, int64_t first, int64_t skipped)
{
    const double *t = s->t;
    double gate = s->pc[lane * FR_PC + 6], step = t[33];
    int64_t resume = first + skipped;
    int64_t boundary = first + (15 - fr_mod(first, 16));
    for (; boundary < resume; boundary += 16) {
        double due = (double)boundary * step;
        double start = fr_align(fr_max(due, gate), t[0]);
        gate = fr_after_tick(start, t[31], s->ticks[31], s->half, s->inverse);
    }
    return gate;
}

/* Skip the REFpb of a gap that nothing after the gap can see.
 *
 * Past the floor each goes at its due time, or a tRP after its row's PRE, and
 * every delay it adds has ended before ``latest``. A skipped one therefore adds
 * only its counts and bus time and closes its open row. The REFpb from ``latest``
 * on are simulated, which leaves the row bus, the gates and the tFAW window as
 * if none had been skipped; both lanes then have the same due time.
 */
FR_NOINLINE int fr_skip_refresh_gap(const fr_state *s, int64_t ch, double latest, double *counts)
{
    if (getenv("HETEROLLM_HBM_AUDIT_DISABLE_EQUIV") != NULL
            || getenv("HETEROLLM_HBM_AUDIT_DISABLE_SKIP") != NULL)
        return 0;
    const double *t = s->t;
    double step = t[33], total;
    int64_t resume = fr_ifloor(latest / step), banks = s->banks, commands = 0;
    int64_t pseudo, within;
    double group_gate[2] = {0., 0.};
    if (step < fr_max(t[30], t[6] + t[32]) + 4 * t[0])
        return 0;
    if (t[29] > 0) {
        for (pseudo = 0; pseudo < 2; ++pseudo) {
            int64_t lane = 2 * ch + pseudo, first = fr_round(s->pc[lane * FR_PC + 5] / step);
            int64_t skipped = resume - first;
            uint64_t done = (uint64_t)s->pc[lane * FR_PC + 7];
            if (skipped <= 0)
                continue;
            group_gate[pseudo] = fr_refresh_skip_group_gate(s, lane, first, skipped);
            if (group_gate[pseudo] > latest)
                return 0;
            for (within = 0; within < banks; ++within) {
                int64_t bank = lane * banks + within, turn = fr_refresh_turn(done, first, banks, within);
                if (s->rows[bank] >= 0
                        && s->bf[bank * 7] + t[29] <= (double)(turn < skipped ? first + turn : resume) * step)
                    return 0;
            }
        }
    }
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        int64_t lane = 2 * ch + pseudo, first = fr_round(s->pc[lane * FR_PC + 5] / step);
        int64_t skipped = resume - first, taken;
        uint64_t done = (uint64_t)s->pc[lane * FR_PC + 7];
        if (skipped <= 0)
            continue;
        for (within = 0; within < banks; ++within) {
            int64_t bank = lane * banks + within, turn = fr_refresh_turn(done, first, banks, within);
            if (s->rows[bank] >= 0 && turn < skipped) {
                s->rows[bank] = -1;
                counts[1] += 1;
                commands += 1;
            }
        }
        if (skipped < 16 - fr_mod(first, 16)) {
            /* The lowest unrefreshed banks of the current set, as REFpb without traffic takes them. */
            for (taken = 0; taken < skipped; ++taken)
                done |= ~done & (done + 1);
        } else {
            done = (UINT64_C(1) << fr_mod(resume, 16)) - 1;
        }
        counts[4] += (double)skipped;
        commands += skipped;
        s->pc[lane * FR_PC + 5] = (double)resume * step;
        s->pc[lane * FR_PC + 6] = group_gate[pseudo];
        s->pc[lane * FR_PC + 7] = (double)done;
    }
    if (!commands)
        return 0;
    total = (double)commands * s->pre_duration;
    s->resources[ch * 4 + 1] += total;
    counts[5] += total;
    return 1;
}

/* REFpb that fell due in the gap before this batch, in time order. They run
 * before the mirror check, so channels that ended the last batch a few REFpb
 * apart meet the batch in the same state. */
FR_NOINLINE void fr_refresh_gap(const fr_state *s, int64_t ch, double first_arrival, double *counts)
{
    const double *t = s->t;
    const double *pc = s->pc + 2 * ch * FR_PC;
    double step = s->precharge_step, duration = s->pre_duration;
    /* Rebuild a full 16-bank SID group after the bulk skip. A skipped
     * boundary can include PREs on both PCs; its delayed group gate and
     * bank-ready edges must expire before row-candidate eligibility sees
     * them. The final SID group restores the exact shared-CA phase. */
    double latest = first_arrival - (16 * t[33] + t[31] + t[6] + t[30] + 4 * t[0]);
    double deadline = fr_next_active_deadline(s, ch);
    double skip_floor = latest - fr_min(pc[5], pc[FR_PC + 5]) >= t[33] ? fr_refresh_floor(s, ch) : HUGE_VAL;
    for (;;) {
        int64_t lane = 2 * ch + (pc[FR_PC + 5] < pc[5]), within, bank;
        double due = s->pc[lane * FR_PC + 5], when, horizon;
        int opened;
        if (!(due < first_arrival))
            return;
        if (due >= skip_floor) {
            skip_floor = HUGE_VAL;
            if (fr_skip_refresh_gap(s, ch, latest, counts)) {
                deadline = fr_next_active_deadline(s, ch);
                continue;
            }
        }
        within = fr_refresh_pick(s, lane, due, NULL);
        bank = lane * s->banks + within;
        opened = s->rows[bank] >= 0;
        when = opened ? fr_precharge_start(s, lane, bank, due) : fr_refresh_bank_start(s, lane, within, due);
        horizon = fr_after(when, opened ? fr_max(duration, t[19]) : duration, s->half);
        if (deadline < horizon + step && deadline < fr_align(horizon, step)) {
            fr_close_expiring(s, ch, fr_align(horizon, step), counts);
            deadline = fr_next_active_deadline(s, ch);
        } else if (opened) {
            fr_precharge(s, lane, bank, due, counts);
            deadline = fr_next_active_deadline(s, ch);
        } else {
            fr_refresh_bank(s, lane, within, when, counts);
        }
    }
}

/* Each lane's stream per request: its first burst with the lane bits removed,
 * and its length. The bank and row of a burst do not depend on its lane, so
 * equal signatures are equal command streams. */
static void fr_signature(const fr_args *a, int64_t ch, int64_t lanes, int64_t *signature)
{
    int64_t position = 0, pseudo, request, stripe = a->g[15];
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        int64_t lane = 2 * ch + pseudo;
        for (request = 0; request < a->requests; ++request) {
            int64_t cursor = fr_lane_cursor(a->firsts[request], a->stops[request], lane, lanes, a->stripe_shift);
            if (cursor < 0) {
                signature[position] = -1;
                signature[position + 1] = 0;
            } else {
                int64_t span = a->stops[request] - cursor;
                signature[position] = cursor - stripe * lane;
                signature[position + 1] = stripe * fr_floordiv(span, stripe * lanes)
                    + fr_imin(fr_mod(span, stripe * lanes), stripe - (cursor & (stripe - 1)));
            }
            position += 2;
        }
    }
}

/* What a simulated channel saved for later channels to mirror. */
typedef struct fr_sources {
    int64_t *source_channel, *signature, *rows_before, *rows_after, *addresses_before, *addresses_after;
    int64_t *counts_before, *counts_after, *channel_int_before, *channel_int_after, *moved_after, *int_after;
    int8_t *usable, *directions_before, *directions_after, *touched_after, *seen_after;
    double *reach, *bf_before, *bf_after, *pc_before, *pc_after, *history_before, *history_after;
    double *windows_before, *windows_after, *channel_before, *channel_after, *resources_before, *resources_after;
    double *stats_after, *float_after, *counts_total;
} fr_sources;

/* Index of the first ACT window entry the row bus frontier has not expired. */
static int64_t fr_first_live(const double *row, int64_t count, double frontier, double width, double half)
{
    int64_t index = 0;
    while (index < count && fr_after(row[index], width, half) <= frontier)
        index += 1;
    return index;
}

/* Whether this channel behaves as the source channel did from its saved state.
 *
 * Bus and data frontiers must be equal. A timestamp may differ only where no
 * command of the batch can see it: every later edge is at or above ``lowest``,
 * the earlier of the batch's first arrival and the next refresh, so a ready
 * time below that and a stamp whose largest gap ends below it are the same
 * state. Open-row deadlines are checked by the caller.
 */
static int fr_state_matches(const fr_state *s, const fr_sources *o, int64_t source, int64_t ch, double lowest)
{
    const double *t = s->t;
    const int8_t *directions = s->a->directions;
    int64_t banks = s->banks, channels = s->channels, entries = s->entries, capacity = s->window_capacity;
    int64_t first_lane = 2 * ch, first_bank = 2 * ch * banks, index, pseudo, offset, field, entry;
    int64_t span_banks = 2 * banks;
    int64_t rids[4];
    double half = s->half, pre_floor, activate_floor, row_floor, column_floor, end_floor, frontier;
    const double *resources_before = o->resources_before + source * 16;
    const double *pc_before = o->pc_before + source * 2 * FR_PC;
    const double *bf_before = o->bf_before + source * span_banks * 7;
    const double *history_before = o->history_before + source * 2 * entries * 4;
    if (s->channel[ch * 2] != o->channel_before[source * 2] || s->channel[ch * 2 + 1] != o->channel_before[source * 2 + 1])
        return 0;
    for (index = 0; index < 2; ++index) {
        int64_t base = index == 0 ? first_bank : first_lane;
        int64_t value = s->channel_int[ch * 2 + index];
        if ((value >= 0 ? value - base : value) != o->channel_int_before[source * 2 + index])
            return 0;
    }
    rids[0] = ch;
    rids[1] = channels + ch;
    rids[2] = 2 * channels + first_lane;
    rids[3] = 2 * channels + first_lane + 1;
    for (index = 0; index < 4; ++index)
        if (s->resources[rids[index] * 4] != resources_before[index * 4])
            return 0;
    for (pseudo = 0; pseudo < 2; ++pseudo)
        if (directions[rids[2 + pseudo]] != o->directions_before[source * 2 + pseudo])
            return 0;
    pre_floor = lowest - t[19] - half;
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        const double *mine = s->pc + (first_lane + pseudo) * FR_PC;
        const double *theirs = pc_before + pseudo * FR_PC;
        if (mine[2] != theirs[2] || mine[3] != theirs[3] || mine[5] != theirs[5] || mine[7] != theirs[7]
                || fr_max(mine[4], lowest) != fr_max(theirs[4], lowest)
                || fr_max(mine[6], lowest) != fr_max(theirs[6], lowest)
                || fr_max(mine[1], lowest) != fr_max(theirs[1], lowest)
                || fr_max(mine[0], pre_floor) != fr_max(theirs[0], pre_floor))
            return 0;
    }
    /* A REFpb waits for tRC from its own edge, an ACT from one clock before it. */
    activate_floor = lowest - t[8] + (s->g[6] ? 0. : t[0]) - half;
    for (offset = 0; offset < span_banks; ++offset) {
        int64_t bank = first_bank + offset;
        const double *mine = s->bf + bank * 7;
        const double *theirs = bf_before + offset * 7;
        if (s->rows[bank] != o->rows_before[source * span_banks + offset]
                || (s->addresses[bank] >= 0) != (o->addresses_before[source * span_banks + offset] >= 0)
                || fr_max(mine[0], activate_floor) != fr_max(theirs[0], activate_floor))
            return 0;
        for (field = 1; field < 7; ++field)
            if (fr_max(mine[field], lowest) != fr_max(theirs[field], lowest))
                return 0;
    }
    row_floor = lowest - fr_max(t[14], t[15]) - half;
    column_floor = lowest - fr_max(fr_max(fr_max(t[11], t[12]), t[13]), t[18]) - half;
    end_floor = lowest - fr_max(t[16], t[17]) - half;
    frontier = s->resources[ch * 4];
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        int64_t lane = first_lane + pseudo, mine_live, theirs_live;
        const double *window_row = s->windows + lane * capacity;
        const double *before_row = o->windows_before + (source * 2 + pseudo) * capacity;
        int64_t count_mine = s->window_counts[lane], count_theirs = o->counts_before[source * 2 + pseudo];
        for (entry = 0; entry < entries; ++entry) {
            const double *mine = s->history + (lane * entries + entry) * 4;
            const double *theirs = history_before + (pseudo * entries + entry) * 4;
            if (fr_max(mine[0], row_floor) != fr_max(theirs[0], row_floor)
                    || fr_max(mine[1], column_floor) != fr_max(theirs[1], column_floor)
                    || fr_max(mine[2], column_floor) != fr_max(theirs[2], column_floor)
                    || fr_max(mine[3], end_floor) != fr_max(theirs[3], end_floor))
                return 0;
        }
        /* Expired ACT entries leave at the next ACT plan on both channels. */
        mine_live = fr_first_live(window_row, count_mine, frontier, t[30], half);
        theirs_live = fr_first_live(before_row, count_theirs, frontier, t[30], half);
        if (count_mine - mine_live != count_theirs - theirs_live)
            return 0;
        for (index = 0; index < count_mine - mine_live; ++index)
            if (window_row[mine_live + index] != before_row[theirs_live + index])
                return 0;
    }
    return 1;
}

static void fr_capture_before(const fr_state *s, const fr_sources *o, int64_t source, int64_t ch)
{
    int64_t banks = s->banks, channels = s->channels, entries = s->entries, capacity = s->window_capacity;
    int64_t first_lane = 2 * ch, first_bank = 2 * ch * banks, span_banks = 2 * banks, index, pseudo;
    const int8_t *directions = s->a->directions;
    int64_t rids[4];
    o->source_channel[source] = ch;
    memcpy(o->rows_before + source * span_banks, s->rows + first_bank, (size_t)span_banks * sizeof(int64_t));
    memcpy(o->addresses_before + source * span_banks, s->addresses + first_bank, (size_t)span_banks * sizeof(int64_t));
    memcpy(o->bf_before + source * span_banks * 7, s->bf + first_bank * 7, (size_t)span_banks * 7 * sizeof(double));
    memcpy(o->pc_before + source * 2 * FR_PC, s->pc + first_lane * FR_PC, 2 * FR_PC * sizeof(double));
    memcpy(o->history_before + source * 2 * entries * 4, s->history + first_lane * entries * 4,
           (size_t)(2 * entries * 4) * sizeof(double));
    memcpy(o->windows_before + source * 2 * capacity, s->windows + first_lane * capacity,
           (size_t)(2 * capacity) * sizeof(double));
    memcpy(o->counts_before + source * 2, s->window_counts + first_lane, 2 * sizeof(int64_t));
    o->channel_before[source * 2] = s->channel[ch * 2];
    o->channel_before[source * 2 + 1] = s->channel[ch * 2 + 1];
    for (index = 0; index < 2; ++index) {
        int64_t base = index == 0 ? first_bank : first_lane;
        int64_t value = s->channel_int[ch * 2 + index];
        o->channel_int_before[source * 2 + index] = value >= 0 ? value - base : value;
    }
    rids[0] = ch;
    rids[1] = channels + ch;
    rids[2] = 2 * channels + first_lane;
    rids[3] = 2 * channels + first_lane + 1;
    for (index = 0; index < 4; ++index)
        memcpy(o->resources_before + (source * 4 + index) * 4, s->resources + rids[index] * 4, 4 * sizeof(double));
    for (pseudo = 0; pseudo < 2; ++pseudo)
        o->directions_before[source * 2 + pseudo] = directions[rids[2 + pseudo]];
}

static void fr_capture_after(const fr_state *s, const fr_sources *o, int64_t source, int64_t ch, int usable,
                             double reach_open, double reach, const int64_t *partial_moved, const double *partial_stats,
                             const int64_t *partial_int, const double *partial_float, const double *partial_counts)
{
    const fr_args *a = s->a;
    int64_t banks = s->banks, channels = s->channels, entries = s->entries, capacity = s->window_capacity;
    int64_t first_lane = 2 * ch, first_bank = 2 * ch * banks, span_banks = 2 * banks, index, pseudo;
    int64_t rids[4];
    o->usable[source] = (int8_t)usable;
    o->reach[source * 2] = reach_open;
    o->reach[source * 2 + 1] = reach;
    memcpy(o->rows_after + source * span_banks, s->rows + first_bank, (size_t)span_banks * sizeof(int64_t));
    memcpy(o->addresses_after + source * span_banks, s->addresses + first_bank, (size_t)span_banks * sizeof(int64_t));
    memcpy(o->bf_after + source * span_banks * 7, s->bf + first_bank * 7, (size_t)span_banks * 7 * sizeof(double));
    memcpy(o->pc_after + source * 2 * FR_PC, s->pc + first_lane * FR_PC, 2 * FR_PC * sizeof(double));
    memcpy(o->history_after + source * 2 * entries * 4, s->history + first_lane * entries * 4,
           (size_t)(2 * entries * 4) * sizeof(double));
    memcpy(o->windows_after + source * 2 * capacity, s->windows + first_lane * capacity,
           (size_t)(2 * capacity) * sizeof(double));
    memcpy(o->counts_after + source * 2, s->window_counts + first_lane, 2 * sizeof(int64_t));
    o->channel_after[source * 2] = s->channel[ch * 2];
    o->channel_after[source * 2 + 1] = s->channel[ch * 2 + 1];
    for (index = 0; index < 2; ++index) {
        int64_t base = index == 0 ? first_bank : first_lane;
        int64_t value = s->channel_int[ch * 2 + index];
        o->channel_int_after[source * 2 + index] = value >= 0 ? value - base : value;
    }
    rids[0] = ch;
    rids[1] = channels + ch;
    rids[2] = 2 * channels + first_lane;
    rids[3] = 2 * channels + first_lane + 1;
    for (index = 0; index < 4; ++index) {
        memcpy(o->resources_after + (source * 4 + index) * 4, s->resources + rids[index] * 4, 4 * sizeof(double));
        o->touched_after[source * 4 + index] = s->touched[rids[index]];
    }
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        o->directions_after[source * 2 + pseudo] = a->directions[rids[2 + pseudo]];
        o->seen_after[source * 2 + pseudo] = s->pc_seen[first_lane + pseudo];
        o->moved_after[source * 2 + pseudo] = partial_moved[pseudo];
    }
    memcpy(o->stats_after + source * 2 * a->stat_fields, partial_stats, (size_t)(2 * a->stat_fields) * sizeof(double));
    memcpy(o->int_after + source * a->requests * 3, partial_int, (size_t)(a->requests * 3) * sizeof(int64_t));
    memcpy(o->float_after + source * a->requests * 2, partial_float, (size_t)(a->requests * 2) * sizeof(double));
    memcpy(o->counts_total + source * 7, partial_counts, 7 * sizeof(double));
}

/* Add one channel's own tallies to the batch's, in channel order.
 *
 * Busy time and bus counters are summed per channel, so a mirrored channel
 * adds exactly what the channel it mirrors added.
 */
static void fr_merge_tallies(const fr_state *s, int64_t ch, const double *saved_busy, const double *partial_busy,
                             const int64_t *partial_moved, const double *partial_stats, const int64_t *partial_int,
                             const double *partial_float, const double *partial_counts, double *counts)
{
    const fr_args *a = s->a;
    int64_t channels = s->channels, first_lane = 2 * ch, index, pseudo, field, request;
    int64_t stat_fields = a->stat_fields;
    int64_t rids[4];
    rids[0] = ch;
    rids[1] = channels + ch;
    rids[2] = 2 * channels + first_lane;
    rids[3] = 2 * channels + first_lane + 1;
    for (index = 0; index < 4; ++index)
        s->resources[rids[index] * 4 + 1] = saved_busy[index] + partial_busy[index];
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        double *lane_stats = a->stats + (first_lane + pseudo) * stat_fields;
        const double *partial = partial_stats + pseudo * stat_fields;
        a->moved[rids[2 + pseudo]] += partial_moved[pseudo];
        for (field = 0; field < 7; ++field)
            lane_stats[field] += partial[field];
        for (field = 7; field < 10; ++field)
            lane_stats[field] = fr_max(lane_stats[field], partial[field]);
    }
    for (request = 0; request < a->requests; ++request) {
        for (field = 0; field < 3; ++field)
            a->out_int[request * 3 + field] += partial_int[request * 3 + field];
        a->out_float[request * 2] = fr_max(a->out_float[request * 2], partial_float[request * 2]);
        a->out_float[request * 2 + 1] = fr_max(a->out_float[request * 2 + 1], partial_float[request * 2 + 1]);
    }
    for (field = 0; field < 7; ++field)
        counts[field] += partial_counts[field];
}

/* Give this channel the source channel's batch, keeping what that batch did not write. */
static void fr_apply_source(const fr_state *s, const fr_sources *o, int64_t source, int64_t ch, double *counts)
{
    const fr_args *a = s->a;
    int64_t banks = s->banks, channels = s->channels, entries = s->entries, capacity = s->window_capacity;
    int64_t first_lane = 2 * ch, first_bank = 2 * ch * banks, span_banks = 2 * banks;
    int64_t shift, offset, pseudo, field, entry, index, stat_fields = a->stat_fields;
    int64_t rids[4];
    double busy[4], partial_busy[4];
    /* Two bursts of a stripe per lane; first touches of a bank differ by the lane bits. */
    shift = 4 * (ch - o->source_channel[source]) * s->g[9];
    for (offset = 0; offset < span_banks; ++offset) {
        int64_t bank = first_bank + offset;
        const double *after = o->bf_after + (source * span_banks + offset) * 7;
        const double *before = o->bf_before + (source * span_banks + offset) * 7;
        s->rows[bank] = o->rows_after[source * span_banks + offset];
        if (o->addresses_before[source * span_banks + offset] < 0 && 0 <= o->addresses_after[source * span_banks + offset])
            s->addresses[bank] = o->addresses_after[source * span_banks + offset] + shift;
        for (field = 0; field < 7; ++field)
            if (after[field] != before[field])
                s->bf[bank * 7 + field] = after[field];
    }
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        int64_t lane = first_lane + pseudo;
        const double *pc_after = o->pc_after + (source * 2 + pseudo) * FR_PC;
        const double *pc_before = o->pc_before + (source * 2 + pseudo) * FR_PC;
        const double *history_after = o->history_after + (source * 2 + pseudo) * entries * 4;
        const double *history_before = o->history_before + (source * 2 + pseudo) * entries * 4;
        const double *windows_after = o->windows_after + (source * 2 + pseudo) * capacity;
        const double *windows_before = o->windows_before + (source * 2 + pseudo) * capacity;
        int64_t counts_after = o->counts_after[source * 2 + pseudo];
        int changed;
        s->pc_seen[lane] = s->pc_seen[lane] || o->seen_after[source * 2 + pseudo];
        for (field = 0; field < FR_PC; ++field)
            if (pc_after[field] != pc_before[field])
                s->pc[lane * FR_PC + field] = pc_after[field];
        for (entry = 0; entry < entries; ++entry)
            for (field = 0; field < 4; ++field)
                if (history_after[entry * 4 + field] != history_before[entry * 4 + field])
                    s->history[(lane * entries + entry) * 4 + field] = history_after[entry * 4 + field];
        changed = counts_after != o->counts_before[source * 2 + pseudo];
        if (!changed) {
            for (index = 0; index < counts_after; ++index) {
                if (windows_after[index] != windows_before[index]) {
                    changed = 1;
                    break;
                }
            }
        }
        if (changed) {
            memcpy(s->windows + lane * capacity, windows_after, (size_t)capacity * sizeof(double));
            s->window_counts[lane] = counts_after;
        }
    }
    s->channel[ch * 2] = o->channel_after[source * 2];
    s->channel[ch * 2 + 1] = o->channel_after[source * 2 + 1];
    for (index = 0; index < 2; ++index) {
        int64_t base = index == 0 ? first_bank : first_lane;
        int64_t value = o->channel_int_after[source * 2 + index];
        s->channel_int[ch * 2 + index] = value >= 0 ? value + base : value;
    }
    rids[0] = ch;
    rids[1] = channels + ch;
    rids[2] = 2 * channels + first_lane;
    rids[3] = 2 * channels + first_lane + 1;
    for (index = 0; index < 4; ++index) {
        int64_t rid = rids[index];
        const double *after = o->resources_after + (source * 4 + index) * 4;
        const double *before = o->resources_before + (source * 4 + index) * 4;
        busy[index] = s->resources[rid * 4 + 1];
        partial_busy[index] = after[1];
        s->resources[rid * 4] = after[0];
        for (field = 2; field < 4; ++field)
            if (after[field] != before[field])
                s->resources[rid * 4 + field] = after[field];
        s->touched[rid] = s->touched[rid] || o->touched_after[source * 4 + index];
    }
    for (pseudo = 0; pseudo < 2; ++pseudo)
        a->directions[rids[2 + pseudo]] = o->directions_after[source * 2 + pseudo];
    fr_merge_tallies(s, ch, busy, partial_busy, o->moved_after + source * 2, o->stats_after + source * 2 * stat_fields,
                     o->int_after + source * a->requests * 3, o->float_after + source * a->requests * 2,
                     o->counts_total + source * 7, counts);
}

/* Whether no row this channel holds open reaches tRASmax while it can still matter.
 *
 * The first REFab, or a whole round of REFpb on both PCs, closes every addressed
 * bank, so those rows only have to outlast that point of the source; a row a
 * refresh leaves open has to outlast its whole batch. Rows opened in the batch
 * are the source's own.
 */
static int fr_open_rows_safe(const fr_state *s, const fr_sources *o, int64_t source, int64_t ch)
{
    const double *t = s->t;
    int64_t bank;
    if (t[29] > 0) {
        for (bank = 2 * ch * s->banks; bank < (2 * ch + 2) * s->banks; ++bank)
            if (s->rows[bank] >= 0 && fr_after(s->bf[bank * 7], t[29], s->half)
                    <= o->reach[source * 2 + (s->addresses[bank] >= 0 ? 0 : 1)])
                return 0;
    }
    return 1;
}

/* A single allocation carved into every per-call table. */
typedef struct fr_arena {
    char *next;
    size_t size;
    int measuring;
} fr_arena;

static void *fr_carve(fr_arena *arena, int64_t count, size_t width)
{
    size_t bytes = ((size_t)count * width + 63) & ~(size_t)63;
    void *result = arena->next;
    if (arena->measuring)
        arena->size += bytes;
    else
        arena->next += bytes;
    return result;
}

typedef struct fr_tables {
    int64_t *bank_group, *bank_entry, *bank_sid, *entry_group, *entry_sid, *group_sid_group;
    int64_t *read_order, *write_order, *read_splits, *write_splits;
    int64_t *signature, *ch_int_own, *close_banks, *close_order;
    double *tally_own, *ch_float_own, *own_last, *own_wait;
    double *saved_stats, *partial_stats, *close_deadlines, *close_starts, *gaps, *no_gaps;
    fr_item *items;
    uint64_t *hits, *firsts, *row_seen;
    int64_t *pending_hit, *queued;
    int8_t *early;
    double *act_gates, *act_floors;
    uint8_t *act_gate_ready, *act_floor_ready;
    uint64_t *miss_seen;
} fr_tables;

static void fr_lay_out(fr_arena *arena, const fr_args *a, int mirror, int64_t capacity_sources, int64_t depth,
                       int64_t hit_words, int64_t groups, fr_sources *o, fr_tables *w)
{
    int64_t banks = a->g[1], span_banks = 2 * banks, entries = a->entries, requests = a->requests;
    int64_t capacity = a->window_capacity, c = capacity_sources, stat_fields = a->stat_fields;
    w->bank_group = (int64_t *)fr_carve(arena, banks, sizeof(int64_t));
    w->bank_entry = (int64_t *)fr_carve(arena, banks, sizeof(int64_t));
    w->bank_sid = (int64_t *)fr_carve(arena, banks, sizeof(int64_t));
    w->entry_group = (int64_t *)fr_carve(arena, entries, sizeof(int64_t));
    w->entry_sid = (int64_t *)fr_carve(arena, entries, sizeof(int64_t));
    w->group_sid_group = (int64_t *)fr_carve(arena, groups, sizeof(int64_t));
    w->read_order = (int64_t *)fr_carve(arena, banks * entries, sizeof(int64_t));
    w->write_order = (int64_t *)fr_carve(arena, banks * entries, sizeof(int64_t));
    w->read_splits = (int64_t *)fr_carve(arena, banks * 2, sizeof(int64_t));
    w->write_splits = (int64_t *)fr_carve(arena, banks * 2, sizeof(int64_t));
    w->signature = (int64_t *)fr_carve(arena, mirror ? 4 * requests : 0, sizeof(int64_t));
    w->tally_own = (double *)fr_carve(arena, 7, sizeof(double));
    w->ch_int_own = (int64_t *)fr_carve(arena, mirror ? requests * 3 : 0, sizeof(int64_t));
    w->ch_float_own = (double *)fr_carve(arena, mirror ? requests * 2 : 0, sizeof(double));
    w->saved_stats = (double *)fr_carve(arena, 2 * stat_fields, sizeof(double));
    w->partial_stats = (double *)fr_carve(arena, 2 * stat_fields, sizeof(double));
    w->items = (fr_item *)fr_carve(arena, 4 * depth, sizeof(fr_item));
    w->hits = (uint64_t *)fr_carve(arena, 4 * hit_words, sizeof(uint64_t));
    w->firsts = (uint64_t *)fr_carve(arena, 4 * hit_words, sizeof(uint64_t));
    w->row_seen = (uint64_t *)fr_carve(arena, (banks + 63) >> 6, sizeof(uint64_t));
    w->miss_seen = (uint64_t *)fr_carve(arena, (banks + 63) >> 6, sizeof(uint64_t));
    w->act_gates = (double *)fr_carve(arena, groups, sizeof(double));
    w->act_gate_ready = (uint8_t *)fr_carve(arena, groups, sizeof(uint8_t));
    w->act_floors = (double *)fr_carve(arena, groups + 2, sizeof(double));
    w->act_floor_ready = (uint8_t *)fr_carve(arena, groups + 1, sizeof(uint8_t));
    w->pending_hit = (int64_t *)fr_carve(arena, 4 * banks, sizeof(int64_t));
    w->queued = (int64_t *)fr_carve(arena, 4 * banks, sizeof(int64_t));
    w->early = (int8_t *)fr_carve(arena, 2 * banks, sizeof(int8_t));
    w->gaps = (double *)fr_carve(arena, 2 * groups * groups, sizeof(double));
    w->no_gaps = (double *)fr_carve(arena, groups, sizeof(double));
    w->own_last = (double *)fr_carve(arena, requests * 2, sizeof(double));
    w->own_wait = (double *)fr_carve(arena, requests * 2, sizeof(double));
    w->close_banks = (int64_t *)fr_carve(arena, span_banks, sizeof(int64_t));
    w->close_order = (int64_t *)fr_carve(arena, span_banks, sizeof(int64_t));
    w->close_deadlines = (double *)fr_carve(arena, span_banks, sizeof(double));
    w->close_starts = (double *)fr_carve(arena, span_banks, sizeof(double));
    o->source_channel = (int64_t *)fr_carve(arena, c, sizeof(int64_t));
    o->usable = (int8_t *)fr_carve(arena, c, sizeof(int8_t));
    o->reach = (double *)fr_carve(arena, c * 2, sizeof(double));
    o->signature = (int64_t *)fr_carve(arena, c * 4 * requests, sizeof(int64_t));
    o->rows_before = (int64_t *)fr_carve(arena, c * span_banks, sizeof(int64_t));
    o->rows_after = (int64_t *)fr_carve(arena, c * span_banks, sizeof(int64_t));
    o->addresses_before = (int64_t *)fr_carve(arena, c * span_banks, sizeof(int64_t));
    o->addresses_after = (int64_t *)fr_carve(arena, c * span_banks, sizeof(int64_t));
    o->bf_before = (double *)fr_carve(arena, c * span_banks * 7, sizeof(double));
    o->bf_after = (double *)fr_carve(arena, c * span_banks * 7, sizeof(double));
    o->pc_before = (double *)fr_carve(arena, c * 2 * FR_PC, sizeof(double));
    o->pc_after = (double *)fr_carve(arena, c * 2 * FR_PC, sizeof(double));
    o->history_before = (double *)fr_carve(arena, c * 2 * entries * 4, sizeof(double));
    o->history_after = (double *)fr_carve(arena, c * 2 * entries * 4, sizeof(double));
    o->windows_before = (double *)fr_carve(arena, c * 2 * capacity, sizeof(double));
    o->counts_before = (int64_t *)fr_carve(arena, c * 2, sizeof(int64_t));
    o->windows_after = (double *)fr_carve(arena, c * 2 * capacity, sizeof(double));
    o->counts_after = (int64_t *)fr_carve(arena, c * 2, sizeof(int64_t));
    o->channel_before = (double *)fr_carve(arena, c * 2, sizeof(double));
    o->channel_after = (double *)fr_carve(arena, c * 2, sizeof(double));
    o->channel_int_before = (int64_t *)fr_carve(arena, c * 2, sizeof(int64_t));
    o->channel_int_after = (int64_t *)fr_carve(arena, c * 2, sizeof(int64_t));
    o->resources_before = (double *)fr_carve(arena, c * 16, sizeof(double));
    o->resources_after = (double *)fr_carve(arena, c * 16, sizeof(double));
    o->directions_before = (int8_t *)fr_carve(arena, c * 2, sizeof(int8_t));
    o->directions_after = (int8_t *)fr_carve(arena, c * 2, sizeof(int8_t));
    o->touched_after = (int8_t *)fr_carve(arena, c * 4, sizeof(int8_t));
    o->seen_after = (int8_t *)fr_carve(arena, c * 2, sizeof(int8_t));
    o->moved_after = (int64_t *)fr_carve(arena, c * 2, sizeof(int64_t));
    o->stats_after = (double *)fr_carve(arena, c * 2 * stat_fields, sizeof(double));
    o->int_after = (int64_t *)fr_carve(arena, c * requests * 3, sizeof(int64_t));
    o->float_after = (double *)fr_carve(arena, c * requests * 2, sizeof(double));
    o->counts_total = (double *)fr_carve(arena, c * 7, sizeof(double));
}

/* Issue one batch's bursts command by command through per-PC queues.
 *
 * Inside a channel the lane with the earliest legal next command issues first,
 * which keeps both command buses in time order. A lane's next command is the
 * earlier of the best row hit's column and the row command of its oldest entry
 * without a queued hit on the open row; the latter is what closes and opens
 * banks ahead of their turn.
 */
FR_NOINLINE void fr_channels(const fr_args *a, char *volatile *arena_out)
{
    const int64_t *g = a->g, *q = a->q;
    const double *t = a->t, *arrivals = a->arrivals;
    const int64_t *firsts = a->firsts, *stops = a->stops;
    const int8_t *reads = a->reads;
    int64_t *rows = a->rows, *addresses = a->addresses, *channel_int = a->channel_int;
    double *bf = a->bf, *pc = a->pc, *history = a->history, *channel = a->channel, *resources = a->resources;
    int64_t *moved = a->moved;
    int8_t *pc_seen = a->pc_seen, *directions = a->directions, *touched = a->touched;
    double *stats = a->stats;
    int64_t requests = a->requests, lanes = g[0], banks = g[1], channels = lanes / 2, entries = a->entries;
    int64_t stat_fields = a->stat_fields;
    double half = t[0] / 2, inverse;
    fr_state state;
    const fr_state *s = &state;
    fr_sources sources;
    const fr_sources *o = &sources;
    fr_tables tables;
    fr_arena arena;
    fr_queue queues[2][2];
    int64_t ticks[FR_FIELDS];
    int64_t index, request, ch, pseudo, direction, depth, hit_words, total_bursts, capacity_sources, source_count;
    int64_t lane_shift, column_shift, bank_shift, follow_tick, clock_back;
    int64_t stripe_shift = a->stripe_shift, stripe_mask = g[15] - 1, stripe_step;
    uint64_t sid_inverse;
    int64_t column_duration_tick, recovery_ticks[2], groups;
    double grid_limit, plain_limit, column_duration, recovery[2], first_arrival;
    int binary, mirror, grouped;
    const int64_t *bank_group, *bank_entry, *bank_sid, *group_sid_group;
    int8_t *early;
    double *tally;
    int64_t *ch_int;
    double *ch_float, *own_last, *own_wait;
    int64_t saved_moved[2], partial_moved[2];
    double saved_busy[4], partial_busy[4];
    if (requests == 0 || a->channel_first >= a->channel_stop)
        return;
    if (half == 0.)
        fr_fail(FR_ERR_ZERO_DIVISION);
    inverse = 1. / half;
    for (index = 0; index < FR_FIELDS; ++index)
        ticks[index] = fr_tick(t[index], half);
    groups = g[5];
    stripe_step = g[15] * lanes - stripe_mask;
    lane_shift = fr_binary_shift(g[0]);
    column_shift = fr_binary_shift(g[3]);
    bank_shift = fr_binary_shift(g[4]);
    sid_inverse = (UINT64_C(0x100000000) + (uint64_t)g[14] - 1) / (uint64_t)g[14];
    binary = fr_imin(fr_imin(lane_shift, column_shift), bank_shift) >= 0;
    grouped = g[12] != 0;
    grid_limit = t[0] * FR_TWO48;
    plain_limit = half * FR_TWO44;
    clock_back = fr_negative_tick(ticks[0]);
    follow_tick = fr_tick(2 * t[0], half);
    /* A channel with the traffic and the live state of a simulated one takes
     * its result. The dyadic idle-refresh fold reads busy totals, so a clock it
     * can apply to does not mirror. */
    total_bursts = 0;
    for (request = 0; request < requests; ++request)
        total_bursts += stops[request] - firsts[request];
    mirror = getenv("HETEROLLM_HBM_AUDIT_DISABLE_EQUIV") == NULL
        && getenv("HETEROLLM_HBM_AUDIT_DISABLE_MIRROR") == NULL
        && a->span >= 2 && t[0] * 2 != (double)fr_ifloor(t[0] * 2)
        && total_bursts >= FR_MIRROR_MIN_BURSTS * a->span;
    capacity_sources = mirror ? fr_imin(FR_MIRROR_SOURCES, a->channel_stop - a->channel_first) : 0;
    depth = fr_imax(q[0], q[1]);
    hit_words = (depth + 63) >> 6;

    memset(&arena, 0, sizeof(arena));
    arena.measuring = 1;
    fr_lay_out(&arena, a, mirror, capacity_sources, depth, hit_words, groups, &sources, &tables);
    arena.next = (char *)malloc(arena.size + 64);
    if (arena.next == NULL)
        fr_fail(FR_ERR_MEMORY);
    *arena_out = arena.next;
    arena.next = (char *)(((uintptr_t)arena.next + 63) & ~(uintptr_t)63);
    arena.measuring = 0;
    fr_lay_out(&arena, a, mirror, capacity_sources, depth, hit_words, groups, &sources, &tables);

    /* Runtime divisions on the burst path become these tables and shifts. */
    for (index = 0; index < banks; ++index) {
        tables.bank_group[index] = fr_floordiv(index, g[4]);
        tables.bank_entry[index] = fr_floordiv(index, g[10]);
        tables.bank_sid[index] = fr_floordiv(index, 16);
    }
    for (index = 0; index < entries; ++index) {
        tables.entry_group[index] = fr_floordiv(index * g[10], g[4]);
        tables.entry_sid[index] = fr_floordiv(index * g[10], 16);
    }
    for (index = 0; index < groups; ++index)
        tables.group_sid_group[index] = fr_floordiv(index, g[15]);
    /* What a column's group adds to the next column of each group, read and write;
     * minus infinity where nothing is added. */
    for (index = 0; index < groups; ++index) {
        int64_t other;
        tables.no_gaps[index] = -HUGE_VAL;
        for (other = 0; other < groups; ++other) {
            tables.gaps[index * groups + other] = index == other ? t[11]
                : tables.group_sid_group[index] != tables.group_sid_group[other] ? t[13] : -HUGE_VAL;
            tables.gaps[(groups + index) * groups + other] = index == other ? t[11] : -HUGE_VAL;
        }
    }
    /* Each bank's history entries by class of column gap: same group, other group, other SID. */
    for (index = 0; index < banks; ++index) {
        int64_t *read_order = tables.read_order + index * entries, *write_order = tables.write_order + index * entries;
        int64_t pass, entry, position = 0;
        for (pass = 0; pass < 3; ++pass) {
            for (entry = 0; entry < entries; ++entry) {
                int other_sid = tables.entry_sid[entry] != tables.bank_sid[index];
                int same_group = tables.entry_group[entry] == tables.bank_group[index];
                if ((other_sid ? 2 : same_group ? 0 : 1) == pass)
                    read_order[position++] = entry * 4;
            }
            if (pass < 2)
                tables.read_splits[index * 2 + pass] = position;
        }
        position = 0;
        for (pass = 0; pass < 2; ++pass) {
            for (entry = 0; entry < entries; ++entry)
                if ((tables.entry_group[entry] == tables.bank_group[index] ? 0 : 1) == pass)
                    write_order[position++] = entry * 4;
            tables.write_splits[index * 2 + pass] = position;
        }
    }
    bank_group = tables.bank_group;
    bank_entry = tables.bank_entry;
    bank_sid = tables.bank_sid;
    group_sid_group = tables.group_sid_group;
    early = tables.early;
    own_last = tables.own_last;
    own_wait = tables.own_wait;
    tally = mirror ? tables.tally_own : a->channel_counts;
    ch_int = mirror ? tables.ch_int_own : a->out_int;
    ch_float = mirror ? tables.ch_float_own : a->out_float;
    if (mirror) {
        memset(tally, 0, 7 * sizeof(double));
        memset(ch_int, 0, (size_t)(requests * 3) * sizeof(int64_t));
        memset(ch_float, 0, (size_t)(requests * 2) * sizeof(double));
    }
    for (pseudo = 0; pseudo < 2; ++pseudo) {
        for (direction = 0; direction < 2; ++direction) {
            fr_queue *queue = &queues[pseudo][direction];
            queue->items = tables.items + (pseudo * 2 + direction) * depth;
            queue->hits = tables.hits + (pseudo * 2 + direction) * hit_words;
            queue->firsts = tables.firsts + (pseudo * 2 + direction) * hit_words;
            queue->pending_hit = tables.pending_hit + (pseudo * 2 + direction) * banks;
            queue->queued = tables.queued + (pseudo * 2 + direction) * banks;
        }
    }

    state.a = a;
    state.g = g;
    state.q = q;
    state.t = t;
    state.rows = rows;
    state.addresses = addresses;
    state.channel_int = channel_int;
    state.window_counts = a->window_counts;
    state.bf = bf;
    state.pc = pc;
    state.history = history;
    state.channel = channel;
    state.resources = resources;
    state.windows = a->windows;
    state.pc_seen = pc_seen;
    state.touched = touched;
    state.banks = banks;
    state.channels = channels;
    state.entries = entries;
    state.window_capacity = a->window_capacity;
    state.half = half;
    state.inverse = inverse;
    state.grid_limit = grid_limit;
    state.end_gap = fr_max(t[16], t[17]);
    state.entry_group = tables.entry_group;
    state.entry_sid = tables.entry_sid;
    state.read_order = tables.read_order;
    state.write_order = tables.write_order;
    state.read_splits = tables.read_splits;
    state.write_splits = tables.write_splits;
    state.precharge_step = g[7] ? t[0] : half;
    memcpy(state.ticks, ticks, sizeof(ticks));
    state.follow_tick = follow_tick;
    state.clock_back = clock_back;
    state.act_duration = t[1] * t[0];
    state.act_duration_tick = fr_tick(state.act_duration, half);
    state.pre_duration = t[2] * t[0];
    state.pre_duration_tick = fr_tick(state.pre_duration, half);
    state.close_banks = tables.close_banks;
    state.close_order = tables.close_order;
    state.close_deadlines = tables.close_deadlines;
    state.close_starts = tables.close_starts;
    state.grouped = grouped;
    state.act_gates = tables.act_gates;
    state.act_gate_ready = tables.act_gate_ready;
    state.act_floors = tables.act_floors;
    state.miss_seen = tables.miss_seen;
    state.act_floor_ready = tables.act_floor_ready;
    fr_audit_prepare(s);

    source_count = 0;
    first_arrival = HUGE_VAL;
    for (request = 0; request < requests; ++request)
        first_arrival = fr_min(first_arrival, arrivals[request]);
    recovery[0] = t[9];
    recovery[1] = fr_after(fr_after(t[23], t[26], half), t[10], half);
    recovery_ticks[0] = fr_tick(recovery[0], half);
    recovery_ticks[1] = fr_tick(recovery[1], half);
    column_duration = t[3] * t[0];
    column_duration_tick = fr_tick(column_duration, half);

    for (ch = a->channel_first; ch < a->channel_stop; ++ch) {
        int64_t row_bus = ch, column_bus = channels + ch, capture = -1, stale_deadline = 0;
        int64_t last_pseudo = 1;
        int reached = 0, refreshed = 0, deadline_reached = 0;
        double reach = -HUGE_VAL, reach_open = -HUGE_VAL, active_deadline, close_before;
        double initial_busy[4];
        int64_t initial_moved[2];
        double traffic_floor[2], last_column[2], act_floor[2], plan_start[2], faw_expiry[2], end_top[2];
        int64_t reads_since[2], last_group[2], lane_request[2], plan_kind[2], plan_direction[2], plan_offset[2];
        double refresh_floor[2];
        int64_t refresh_target[2], refresh_issued[2];
        int draining[2], stale[2];
        for (pseudo = 0; pseudo < 2; ++pseudo) {
            int64_t lane = 2 * ch + pseudo;
            traffic_floor[pseudo] = fr_after(resources[(2 * channels + lane) * 4], -t[22], half);
            for (direction = 0; direction < 2; ++direction) {
                fr_queue *queue = &queues[pseudo][direction];
                fr_next_request(a, 0, direction, lane, lanes, &queue->iter_request, &queue->iter_cursor);
                reached = reached || queue->iter_request >= 0;
            }
        }
        if (!reached)
            continue;
        if (!mirror)
            tally = a->channel_counts + (ch - a->channel_first) * 7;
        initial_busy[0] = resources[row_bus * 4 + 1];
        initial_busy[1] = resources[column_bus * 4 + 1];
        for (pseudo = 0; pseudo < 2; ++pseudo) {
            int64_t data = 2 * channels + 2 * ch + pseudo;
            initial_busy[2 + pseudo] = resources[data * 4 + 1];
            initial_moved[pseudo] = moved[data];
        }
        if (g[6]) {
            for (pseudo = 0; pseudo < 2; ++pseudo) {
                int64_t lane = 2 * ch + pseudo;
                double *due = pc + lane * FR_PC + 5;
                pc_seen[lane] = 1;
                if (*due < 0) {
                    /* The current maintenance epoch, not history before the model; bank 0 opens it. */
                    *due = fr_max(t[20], (double)fr_ifloor(fr_divide(first_arrival, t[20])) * t[20]);
                    for (index = 0; index < banks; ++index)
                        if (addresses[lane * banks + index] < 0)
                            addresses[lane * banks + index] = fr_bank_cursor(g, lane, index) * g[9];
                }
            }
            fr_refresh_gap(s, ch, first_arrival, mirror ? a->channel_counts + (ch - a->channel_first) * 7 : tally);
        }
        if (mirror) {
            if (g[6] || channel[ch * 2 + 1] >= 0) {
                double lowest = g[6] ? fr_min(fr_min(first_arrival, pc[2 * ch * FR_PC + 5]), pc[(2 * ch + 1) * FR_PC + 5])
                                     : fr_min(first_arrival, channel[ch * 2 + 1]);
                int mirrored = 0;
                int64_t source;
                fr_signature(a, ch, lanes, tables.signature);
                for (source = 0; source < source_count; ++source) {
                    if (!o->usable[source] || memcmp(o->signature + source * 4 * requests, tables.signature,
                                                     (size_t)(4 * requests) * sizeof(int64_t)) != 0)
                        continue;
                    if (fr_state_matches(s, o, source, ch, lowest) && fr_open_rows_safe(s, o, source, ch)) {
                        fr_apply_source(s, o, source, ch, a->channel_counts + (ch - a->channel_first) * 7);
                        mirrored = 1;
                        break;
                    }
                }
                if (mirrored)
                    goto channel_done;
                if (source_count < capacity_sources) {
                    capture = source_count;
                    source_count += 1;
                    memcpy(o->signature + capture * 4 * requests, tables.signature,
                           (size_t)(4 * requests) * sizeof(int64_t));
                    fr_capture_before(s, o, capture, ch);
                }
            }
            /* The channel's own tallies, added to the batch's when it finishes. */
            for (pseudo = 0; pseudo < 2; ++pseudo) {
                int64_t lane = 2 * ch + pseudo, data = 2 * channels + lane;
                saved_busy[2 + pseudo] = resources[data * 4 + 1];
                resources[data * 4 + 1] = 0.;
                saved_moved[pseudo] = moved[data];
                moved[data] = 0;
                memcpy(tables.saved_stats + pseudo * stat_fields, stats + lane * stat_fields,
                       (size_t)stat_fields * sizeof(double));
                memset(stats + lane * stat_fields, 0, (size_t)stat_fields * sizeof(double));
            }
            saved_busy[0] = resources[row_bus * 4 + 1];
            resources[row_bus * 4 + 1] = 0.;
            saved_busy[1] = resources[column_bus * 4 + 1];
            resources[column_bus * 4 + 1] = 0.;
            memset(tally, 0, 7 * sizeof(double));
            memset(ch_int, 0, (size_t)(requests * 3) * sizeof(int64_t));
            for (request = 0; request < requests; ++request) {
                ch_float[request * 2] = -HUGE_VAL;
                ch_float[request * 2 + 1] = 0.;
            }
        }
        active_deadline = fr_next_active_deadline(s, ch);
        for (pseudo = 0; pseudo < 2; ++pseudo) {
            for (direction = 0; direction < 2; ++direction) {
                fr_queue *queue = &queues[pseudo][direction];
                queue->head = 0;
                queue->count = 0;
                queue->bypass = 0;
                queue->slot_free = 0.;
                queue->dirty = 0;
                memset(queue->pending_hit, 0, (size_t)banks * sizeof(int64_t));
                memset(queue->queued, 0, (size_t)banks * sizeof(int64_t));
                memset(queue->hits, 0, (size_t)hit_words * sizeof(uint64_t));
                memset(queue->firsts, 0, (size_t)hit_words * sizeof(uint64_t));
            }
            draining[pseudo] = 0;
            stale[pseudo] = 1;
            reads_since[pseudo] = q[4];
            last_group[pseudo] = -1;
            lane_request[pseudo] = -1;
            plan_kind[pseudo] = -1;
            last_column[pseudo] = -1.;
            act_floor[pseudo] = 0.;
            plan_start[pseudo] = HUGE_VAL;
            faw_expiry[pseudo] = -HUGE_VAL;
            /* Never reached without per-bank refresh: a planned start is finite. */
            refresh_floor[pseudo] = g[6] ? pc[(2 * ch + pseudo) * FR_PC + 5] : HUGE_VAL;
            refresh_target[pseudo] = -1;
            refresh_issued[pseudo] = 0;
            end_top[pseudo] = -1.;
            for (index = 0; index < entries; ++index)
                end_top[pseudo] = fr_max(end_top[pseudo], history[((2 * ch + pseudo) * entries + index) * 4 + 3]);
        }
        memset(early, 0, (size_t)(2 * banks) * sizeof(int8_t));
        for (index = 0; index < requests * 2; ++index) {
            own_last[index] = -1.;
            own_wait[index] = 0.;
        }
        for (;;) {
            int64_t lane, kind, chosen, slot, bank, row, bank_within, group, history_entry, capacity;
            int read, maintained;
            double start, entry, horizon;
            fr_queue *queue;
            const fr_item *item;
            for (pseudo = 0; pseudo < 2; ++pseudo) {
                int64_t waiting_reads, waiting_writes, waiting, first_slot, hit_offset, offset;
                int forced;
                double column_floor, floor_, best, previous;
                int64_t previous_group;
                double *lane_stats;
                const int64_t *lane_rows;
                const double *lane_bf;
                if (!stale[pseudo])
                    continue;
                stale[pseudo] = 0;
                lane = 2 * ch + pseudo;
                lane_stats = stats + lane * stat_fields;
                lane_rows = rows + lane * banks;
                lane_bf = bf + lane * banks * 7;
                if (draining[pseudo] && queues[pseudo][1].count <= q[3]) {
                    draining[pseudo] = 0;
                    reads_since[pseudo] = 0;
                }
                for (direction = 0; direction < 2; ++direction) {
                    /* A drain lowers the write queue only if refill waits for its end. */
                    if (direction == 1 && draining[pseudo])
                        continue;
                    queue = &queues[pseudo][direction];
                    capacity = q[direction];
                    while (queue->count < capacity && queue->iter_request >= 0) {
                        int64_t cursor = queue->iter_cursor, column;
                        request = queue->iter_request;
                        if (binary) {
                            /* The burst location for non-negative cursors. */
                            int64_t block = cursor >> (stripe_shift + lane_shift + column_shift);
                            int64_t rest = block >> bank_shift;
                            row = rest < FR_SID_FAST ? (int64_t)(((uint64_t)rest * sid_inverse) >> 32) : rest / g[14];
                            bank_within = (rest - row * g[14]) * g[15] * g[4];
                            bank_within += (cursor & stripe_mask) * g[4] + (block & (g[4] - 1));
                        } else {
                            fr_burst_location(g, cursor, &bank_within, &row);
                        }
                        column = fr_mod(fr_floordiv(fr_floordiv(cursor, g[15]), g[0]), g[3]);
                        bank = lane * banks + bank_within;
                        slot = queue->head + queue->count;
                        if (slot >= capacity)
                            slot -= capacity;
                        queue->items[slot].within = bank_within;
                        queue->items[slot].row = row;
                        queue->items[slot].column = column;
                        queue->items[slot].request = request;
                        queue->items[slot].entry = fr_max(arrivals[request], queue->slot_free);
                        if (addresses[bank] < 0) {
                            addresses[bank] = cursor * g[9];
                            if (rows[bank] >= 0)
                                stale_deadline = 1;
                        }
                        pc_seen[lane] = 1;
                        if (rows[bank] != row && queue->queued[bank_within] == queue->pending_hit[bank_within])
                            fr_set_bit(queue->firsts, queue->count);
                        queue->queued[bank_within] += 1;
                        /* Queued work changes which bank of the set a due REFpb takes. */
                        if (refresh_target[pseudo] >= 0 && (bank_within >> 4) == (refresh_target[pseudo] >> 4))
                            refresh_floor[pseudo] = pc[lane * FR_PC + 5];
                        if (rows[bank] == row) {
                            queue->pending_hit[bank_within] += 1;
                            fr_set_bit(queue->hits, queue->count);
                        } else {
                            queue->dirty = 1;
                        }
                        queue->count += 1;
                        cursor = (cursor & stripe_mask) != stripe_mask ? cursor + 1 : cursor + stripe_step;
                        if (cursor < stops[request])
                            queue->iter_cursor = cursor;
                        else
                            fr_next_request(a, request + 1, direction, lane, lanes,
                                            &queue->iter_request, &queue->iter_cursor);
                    }
                    if ((double)queue->count > lane_stats[7 + direction])
                        lane_stats[7 + direction] = (double)queue->count;
                }
                waiting_reads = queues[pseudo][0].count;
                waiting_writes = queues[pseudo][1].count;
                if (waiting_reads + waiting_writes == 0) {
                    plan_start[pseudo] = HUGE_VAL;
                    plan_kind[pseudo] = -1;
                    if (g[6])
                        refresh_floor[pseudo] = fr_plan_refresh(s, lane, queues[pseudo], plan_start + pseudo, plan_kind + pseudo,
                                                                plan_offset + pseudo, refresh_target + pseudo);
                    continue;
                }
                if (!draining[pseudo] && waiting_reads && waiting_writes >= q[2] && reads_since[pseudo] >= q[4]) {
                    draining[pseudo] = 1;
                    lane_stats[6] += 1;
                }
                column_floor = resources[column_bus * 4];
                if (draining[pseudo] || !waiting_reads) {
                    direction = 1;
                } else if (!waiting_writes) {
                    direction = 0;
                } else {
                    /* A read that has arrived always goes ahead of a queued write. */
                    const fr_queue *read_queue = &queues[pseudo][0], *write_queue = &queues[pseudo][1];
                    direction = read_queue->items[read_queue->head].entry
                        <= fr_max(write_queue->items[write_queue->head].entry, column_floor) ? 0 : 1;
                }
                queue = &queues[pseudo][direction];
                read = direction == 0;
                capacity = q[direction];
                waiting = queue->count;
                first_slot = queue->head;
                forced = queue->bypass >= capacity;
                previous = last_column[pseudo];
                previous_group = last_group[pseudo];
                floor_ = previous < 0 ? column_floor : fr_max(column_floor, previous + t[12]);
                hit_offset = -1;
                best = HUGE_VAL;
                {
                    int64_t bank_ready_field = read ? 3 : 4, history_field = read ? 1 : 2, word;
                    int64_t last_word = forced ? 1 : hit_words;
                    const double *lane_history = history + lane * entries * 4;
                    const double *previous_gaps = previous < 0 ? tables.no_gaps
                        : tables.gaps + (read ? 0 : groups * groups) + previous_group * groups;
                    for (word = 0; word < last_word; ++word) {
                        uint64_t rest = queue->hits[word];
                        if (forced)
                            rest &= 1;
                        while (rest) {
                            double ready, stamp;
                            offset = word * 64 + fr_lowest_bit(rest);
                            rest &= rest - 1;
                            /* This wrap flips with the ring position, so it does not branch. */
                            slot = first_slot + offset;
                            slot -= capacity & -(int64_t)(slot >= capacity);
                            item = &queue->items[slot];
                            bank_within = item->within;
                            ready = fr_max(fr_max(floor_, lane_bf[bank_within * 7 + bank_ready_field]), item->entry);
                            stamp = lane_history[bank_entry[bank_within] * 4 + history_field];
                            if (stamp >= 0)
                                ready = fr_max(ready, stamp + t[11]);
                            ready = fr_max(ready, previous + previous_gaps[bank_group[bank_within]]);
                            if (ready < best) {
                                best = ready;
                                hit_offset = offset;
                                if (ready <= floor_)
                                    goto hits_scanned;
                            }
                        }
                    }
                }
            hits_scanned:
                start = HUGE_VAL;
                kind = -1;
                chosen = -1;
                if (hit_offset >= 0) {
                    slot = first_slot + hit_offset;
                    if (slot >= capacity)
                        slot -= capacity;
                    item = &queue->items[slot];
                    bank_within = item->within;
                    start = fr_column_start(s, lane, bank_within, bank_group[bank_within], bank_sid[bank_within],
                                            lane_bf[bank_within * 7 + (read ? 3 : 4)], item->entry, read,
                                            column_bus, 2 * channels + lane, end_top[pseudo]);
                    kind = 0;
                    chosen = hit_offset;
                }
                /* No row edge precedes the row bus; a column wins a tie.
                 * Keep dirty until a later plan can consider a row. */
                if (queue->dirty && start > resources[row_bus * 4]) {
                    int unserved = 0, eligible = 0;
                    int64_t word, in_refresh = -1;
                    double refresh_floor = resources[row_bus * 4] + t[6];
                    memset(tables.act_gate_ready, 0, (size_t)groups * sizeof(uint8_t));
                    memset(tables.act_floor_ready, 0, (size_t)(groups + 1) * sizeof(uint8_t));
                    /* Only the oldest miss of each bank participates (``firsts``). Compare
                     * its exact command edge; queue order breaks ties and a column wins a tie. */
                    for (word = 0; word << 6 < waiting; ++word) {
                        uint64_t rest = ~queue->hits[word];
                        if (waiting - (word << 6) < 64)
                            rest &= (UINT64_C(1) << (waiting - (word << 6))) - 1;
                        if (rest)
                            unserved = 1;
                        rest = queue->firsts[word];
                        if (forced)
                            rest &= word == 0 ? UINT64_C(1) : UINT64_C(0);
                        while (rest) {
                            double row_start;
                            offset = word * 64 + fr_lowest_bit(rest);
                            rest &= rest - 1;
                            slot = first_slot + offset;
                            if (slot >= capacity)
                                slot -= capacity;
                            item = &queue->items[slot];
                            bank_within = item->within;
                            if (lane_rows[bank_within] >= 0 && queue->pending_hit[bank_within] > 0
                                    && !(forced && offset == 0))
                                continue;
                            if (g[6] && lane_rows[bank_within] < 0 && lane_bf[bank_within * 7 + 1] > refresh_floor
                                    && !(forced && offset == 0)) {
                                if (in_refresh < 0)
                                    in_refresh = offset;
                                continue;
                            }
                            eligible = 1;
                            bank = lane * banks + bank_within;
                            row_start = fr_queued_row_start(s, lane, bank, bank_group[bank_within],
                                item->entry, act_floor[pseudo], start, faw_expiry + pseudo, plain_limit);
                            if (row_start < start) {
                                start = row_start;
                                kind = lane_rows[bank_within] >= 0 ? 1 : 2;
                                chosen = offset;
                            }
                        }
                    }
                    /* Refreshing banks remain a fallback, in the original queue order. */
                    if (!eligible && in_refresh >= 0) {
                        double row_start;
                        slot = first_slot + in_refresh;
                        if (slot >= capacity)
                            slot -= capacity;
                        item = &queue->items[slot];
                        bank_within = item->within;
                        bank = lane * banks + bank_within;
                        row_start = fr_queued_row_start(s, lane, bank, bank_group[bank_within],
                            item->entry, act_floor[pseudo], start, faw_expiry + pseudo, plain_limit);
                        if (row_start < start) {
                            start = row_start;
                            kind = 2;
                            chosen = in_refresh;
                        }
                    }
                    queue->dirty = unserved;
                }
                if (kind < 0)
                    fr_fail(FR_ERR_NO_COMMAND);
                plan_start[pseudo] = start;
                plan_kind[pseudo] = kind;
                plan_direction[pseudo] = direction;
                plan_offset[pseudo] = chosen;
                if (refresh_floor[pseudo] <= start)
                    refresh_floor[pseudo] = fr_plan_refresh(s, lane, queues[pseudo], plan_start + pseudo, plan_kind + pseudo,
                                                            plan_offset + pseudo, refresh_target + pseudo);
            }
            /* An idle lane plans REFpb for as long as the other has traffic. */
            if (g[6] && queues[0][0].count + queues[0][1].count + queues[1][0].count + queues[1][1].count == 0)
                break;
            if (plan_start[0] == plan_start[1]) {
                if (plan_kind[0] < 0)
                    break;
                pseudo = 1 - last_pseudo;
            } else {
                pseudo = plan_start[0] < plan_start[1] ? 0 : 1;
            }
            kind = plan_kind[pseudo];
            start = plan_start[pseudo];
            /* The other lane's issue may have passed this plan's bus slot. */
            if (start < resources[(kind == 0 ? column_bus : row_bus) * 4]) {
                stale[pseudo] = 1;
                continue;
            }
            lane = 2 * ch + pseudo;
            chosen = plan_offset[pseudo];
            /* A row command holds the row bus, and a PRE also blocks the PC's next
             * PRE: a deadline inside that span could no longer get its own PRE. */
            if (kind >= 3) {
                bank_within = chosen;
                bank = lane * banks + bank_within;
                entry = start;
                horizon = fr_after(start, kind == 4 ? fr_max(t[2] * t[0], t[19]) : t[2] * t[0], half);
            } else {
                direction = plan_direction[pseudo];
                read = direction == 0;
                queue = &queues[pseudo][direction];
                capacity = q[direction];
                slot = queue->head + chosen;
                if (slot >= capacity)
                    slot -= capacity;
                item = &queue->items[slot];
                bank_within = item->within;
                bank = lane * banks + bank_within;
                row = item->row;
                request = item->request;
                entry = item->entry;
                group = bank_group[bank_within];
                history_entry = bank_entry[bank_within];
                horizon = kind == 0 ? fr_after_tick(start, recovery[direction], recovery_ticks[direction], half, inverse)
                                    : fr_after(start, kind == 1 ? fr_max(t[2] * t[0], t[19]) : t[1] * t[0], half);
            }
            if (stale_deadline) {
                active_deadline = fr_next_active_deadline(s, ch);
                stale_deadline = 0;
            }
            maintained = 0;
            reach = fr_max(reach, horizon + s->precharge_step);
            /* A PRE on the first row grid point after the horizon still meets a
             * deadline at that point, so only earlier deadlines close first. */
            close_before = active_deadline < horizon + s->precharge_step ? fr_align(horizon, s->precharge_step)
                                                                         : -HUGE_VAL;
            if (active_deadline < close_before) {
                deadline_reached = 1;
                if (g[8] && channel[ch * 2 + 1] < 0)
                    channel[ch * 2 + 1] = fr_max(t[20], (double)fr_ifloor(fr_divide(entry, t[20])) * t[20]);
                if (g[8] && channel[ch * 2 + 1] <= active_deadline)
                    fr_refresh_until(s, lane, channel[ch * 2 + 1], entry, tally);
                else
                    fr_close_expiring(s, ch, close_before, tally);
                maintained = 1;
            } else if (g[8] && channel[ch * 2 + 1] <= start && fr_refresh_until(s, lane, start, entry, tally)) {
                maintained = 1;
            }
            if (maintained) {
                if (!refreshed) {
                    refreshed = 1;
                    reach_open = fr_max(reach, resources[row_bus * 4]);
                }
                active_deadline = fr_next_active_deadline(s, ch);
                fr_recount(s, ch, queues, hit_words);
                stale[0] = 1;
                stale[1] = 1;
                if (g[6]) {
                    refresh_floor[0] = pc[2 * ch * FR_PC + 5];
                    refresh_floor[1] = pc[(2 * ch + 1) * FR_PC + 5];
                }
                continue;
            }
            if (refresh_target[pseudo] >= 0 && (bank_within >> 4) == (refresh_target[pseudo] >> 4))
                refresh_floor[pseudo] = pc[lane * FR_PC + 5];
            if (kind == 3) {
                fr_refresh_bank(s, lane, bank_within, start, tally);
                refresh_floor[pseudo] = pc[lane * FR_PC + 5];
                refresh_target[pseudo] = -1;
                faw_expiry[pseudo] = -HUGE_VAL;
                refresh_issued[pseudo] += 1;
                /* A set in progress is finished in any order, so every bank is only
                 * certain to have had its REFpb 15 after a whole round. */
                if (!refreshed && refresh_issued[0] >= banks + 15 && refresh_issued[1] >= banks + 15) {
                    refreshed = 1;
                    reach_open = fr_max(reach, resources[row_bus * 4]);
                }
                stale[pseudo] = 1;
                if (plan_kind[1 - pseudo] != 0 || queues[1 - pseudo][0].dirty || queues[1 - pseudo][1].dirty)
                    stale[1 - pseudo] = 1;
                last_pseudo = pseudo;
                continue;
            }
            if (kind == 1 || kind == 4) {
                /* A PRE for a REFpb is due at the REFpb's time and is not an early close. */
                fr_precharge(s, lane, bank, kind == 4 ? pc[lane * FR_PC + 5] : entry, tally);
                /* The channel deadline a full rescan would give after this bank
                 * closed: closing a bank whose own deadline lies above the
                 * minimum leaves it unchanged. An open bank mapped since the
                 * last scan always forces the rescan. */
                if (stale_deadline || (t[29] > 0 && fr_after_tick(bf[bank * 7], t[29], ticks[29], half, inverse)
                                       <= active_deadline)) {
                    active_deadline = fr_next_active_deadline(s, ch);
                    stale_deadline = 0;
                }
                if (kind == 1) {
                    stats[lane * stat_fields + 4] += 1;
                    early[pseudo * banks + bank_within] = 1;
                }
                fr_forget_bank(queues[pseudo], q, hit_words, bank_within);
                stale[pseudo] = 1;
                if (plan_kind[1 - pseudo] != 0 || queues[1 - pseudo][0].dirty || queues[1 - pseudo][1].dirty)
                    stale[1 - pseudo] = 1;
                last_pseudo = pseudo;
                continue;
            }
            if (kind == 2) {
                double anchor = fr_after_tick(start, t[0], ticks[0], half, inverse), command;
                double *bank_bf = bf + bank * 7;
                int64_t side;
                if (t[29] > 0) {
                    int64_t rcd_index = read ? 4 : 5;
                    double rcd = t[rcd_index];
                    double column = fr_column_start(s, lane, bank_within, group, bank_sid[bank_within],
                                                    fr_after_tick(anchor, rcd, ticks[rcd_index], half, inverse),
                                                    entry, read, column_bus, 2 * channels + lane, end_top[pseudo]);
                    double latest = fr_align(fr_after_tick(column, recovery[direction], recovery_ticks[direction],
                                                           half, inverse), s->precharge_step);
                    if (latest > fr_after_tick(anchor, t[29], ticks[29], half, inverse)) {
                        if (fr_after(rcd, recovery[direction], half) > t[29])
                            fr_fail(FR_ERR_RECOVERY_WINDOW);
                        /* A later DQ reservation keeps the row closed rather than
                         * holding it open for longer than tRASmax. */
                        act_floor[pseudo] = fr_align(fr_max(fr_after(start, t[0], half),
                                                            fr_after(fr_after(latest, -t[29], half), -t[0], half)), t[0]);
                        stale[pseudo] = 1;
                        continue;
                    }
                }
                command = fr_row_command(s, ch, start, t[0], s->act_duration, s->act_duration_tick, tally);
                fr_trace(s, "ACT", lane, bank_sid[bank_within], bank_group[bank_within], bank_within,
                         row, -1, command, -DBL_MAX, -DBL_MAX);
                anchor = fr_after_tick(command, t[0], ticks[0], half, inverse);
                history[(lane * entries + group) * 4] = anchor;
                if (t[30] > 0) {
                    fr_record_activate(s, lane, anchor);
                    faw_expiry[pseudo] = -HUGE_VAL;
                }
                channel[ch * 2] = command;
                channel_int[ch * 2] = bank;
                channel_int[ch * 2 + 1] = lane;
                bank_bf[0] = anchor;
                bank_bf[3] = fr_after_tick(anchor, t[4], ticks[4], half, inverse);
                bank_bf[4] = fr_after_tick(anchor, t[5], ticks[5], half, inverse);
                bank_bf[2] = fr_after_tick(anchor, t[7], ticks[7], half, inverse);
                rows[bank] = row;
                bank_bf[6] = fr_min(bank_bf[3], bank_bf[4]);
                bank_bf[5] = bank_bf[2];
                tally[0] += 1;
                stats[lane * stat_fields + 3] += 1;
                if (stale_deadline) {
                    active_deadline = fr_next_active_deadline(s, ch);
                    stale_deadline = 0;
                } else if (t[29] > 0) {
                    active_deadline = fr_min(active_deadline, fr_after_tick(anchor, t[29], ticks[29], half, inverse));
                }
                early[pseudo * banks + bank_within] = early[pseudo * banks + bank_within] == 1 ? 2 : 3;
                act_floor[pseudo] = 0.;
                for (side = 0; side < 2; ++side) {
                    fr_queue *other = &queues[pseudo][side];
                    int64_t hits_now = 0, position = other->head, iteration;
                    for (iteration = 0; iteration < other->count; ++iteration) {
                        if (other->items[position].within == bank_within && other->items[position].row == row) {
                            hits_now += 1;
                            fr_set_bit(other->hits, iteration);
                        }
                        position += 1;
                        if (position == q[side])
                            position = 0;
                    }
                    other->pending_hit[bank_within] = hits_now;
                    if (other->queued[bank_within] > 0)
                        fr_first_of_bank(other, q[side], bank_within, row);
                    other->dirty = 1;
                }
                stale[pseudo] = 1;
                /* Row CA, refresh eligibility and the shared last-ACT bank
                 * changed. Pure column-only work has no row dependency. */
                if (plan_kind[1 - pseudo] != 0 || queues[1 - pseudo][0].dirty || queues[1 - pseudo][1].dirty)
                    stale[1 - pseudo] = 1;
                last_pseudo = pseudo;
                continue;
            }
            {
                /* _command on the column bus; a rising edge on the command clock either way. */
                double *column_resource = resources + column_bus * 4;
                int64_t data = 2 * channels + lane, position, first_slot, iteration, outcome;
                double *data_resource = resources + data * 4;
                double *lane_stats = stats + lane * stat_fields;
                double command = fr_align(fr_max(start, column_resource[0]), t[0]);
                double command_end = fr_after_tick(command, column_duration, column_duration_tick, half, inverse);
                double data_start, data_end;
                int64_t latency = read ? 22 : 23, turnaround = read ? 27 : 28;
                column_resource[0] = command_end;
                column_resource[1] += column_duration;
                column_resource[2] = command;
                column_resource[3] = command_end;
                touched[column_bus] = 1;
                tally[6] += column_duration;
                tally[read ? 2 : 3] += 1;
                history[(lane * entries + history_entry) * 4 + (read ? 1 : 2)] = command;
                if (directions[data] && directions[data] != (read ? 1 : 2))
                    lane_stats[5] += 1;
                data_start = fr_max(fr_after_tick(command, t[latency], ticks[latency], half, inverse), data_resource[0]);
                data_end = fr_after_tick(data_start, t[26], ticks[26], half, inverse);
                fr_trace(s, read ? "RD" : "WR", lane, bank_sid[bank_within], bank_group[bank_within],
                         bank_within, row, item->column, command, data_start, data_end);
                data_resource[0] = data_end;
                data_resource[1] += t[26];
                data_resource[2] = data_start;
                data_resource[3] = data_end;
                moved[data] += g[9];
                directions[data] = read ? 1 : 2;
                touched[data] = 1;
                pc[lane * FR_PC + (read ? 2 : 3)] = fr_after_tick(data_start, t[turnaround], ticks[turnaround], half, inverse);
                if (read) {
                    double own;
                    bf[bank * 7 + 2] = fr_max(bf[bank * 7 + 2], fr_after_tick(command, t[9], ticks[9], half, inverse));
                    reads_since[pseudo] += 1;
                    /* Waits only count time another request's traffic held this lane. */
                    own = own_last[request * 2 + pseudo];
                    if (own >= 0) {
                        if (lane_request[pseudo] != request)
                            own_wait[request * 2 + pseudo] += fr_max(0., command - fr_max(entry, own + t[12]));
                    } else if (lane_request[pseudo] >= 0) {
                        own_wait[request * 2 + pseudo] += fr_max(0., command - entry);
                    } else {
                        own_wait[request * 2 + pseudo] += fr_max(0., fr_min(command, traffic_floor[pseudo]) - entry);
                    }
                    own_last[request * 2 + pseudo] = command;
                    lane_request[pseudo] = request;
                } else {
                    bf[bank * 7 + 2] = fr_max(bf[bank * 7 + 2], fr_after_tick(data_end, t[10], ticks[10], half, inverse));
                    history[(lane * entries + history_entry) * 4 + 3] = data_end;
                    end_top[pseudo] = fr_max(end_top[pseudo], data_end);
                }
                bf[bank * 7 + 5] = bf[bank * 7 + 2];
                ch_float[request * 2] = fr_max(ch_float[request * 2], data_end);
                outcome = early[pseudo * banks + bank_within];
                early[pseudo * banks + bank_within] = 0;
                /* Early tags: 0 row hit, 2 opened after a conflict, 3 opened from closed. */
                outcome = outcome == 0 ? 0 : outcome == 2 ? 2 : 1;
                ch_int[request * 3 + outcome] += 1;
                lane_stats[outcome] += 1;
                queue->pending_hit[bank_within] -= 1;
                queue->queued[bank_within] -= 1;
                fr_remove_bit(queue->hits, hit_words, chosen);
                fr_remove_bit(queue->firsts, hit_words, chosen);
                /* Remove the entry, keeping the others in age order. */
                first_slot = queue->head;
                position = slot;
                for (iteration = 0; iteration < chosen; ++iteration) {
                    int64_t from = position ? position - 1 : capacity - 1;
                    queue->items[position] = queue->items[from];
                    position = from;
                }
                queue->head = first_slot + 1 < capacity ? first_slot + 1 : 0;
                queue->count -= 1;
                queue->bypass = chosen ? queue->bypass + 1 : 0;
                queue->slot_free = command;
                last_column[pseudo] = command;
                last_group[pseudo] = group;
                stale[pseudo] = 1;
                /* The shared column floor changes hit order or read/write
                 * priority. A fixed-direction row winner can only benefit
                 * when its alternative column is pushed later. */
                if (plan_kind[1 - pseudo] == 0 || (plan_direction[1 - pseudo] == 1
                        && queues[1 - pseudo][0].count && !draining[1 - pseudo]))
                    stale[1 - pseudo] = 1;
                last_pseudo = pseudo;
            }
        }
        for (pseudo = 0; pseudo < 2; ++pseudo) {
            double *lane_stats = stats + (2 * ch + pseudo) * stat_fields;
            for (request = 0; request < requests; ++request) {
                double pending_wait = own_wait[request * 2 + pseudo];
                if (pending_wait > ch_float[request * 2 + 1])
                    ch_float[request * 2 + 1] = pending_wait;
                if (pending_wait > lane_stats[9])
                    lane_stats[9] = pending_wait;
            }
        }
        if (mirror) {
            partial_busy[0] = resources[row_bus * 4 + 1];
            partial_busy[1] = resources[column_bus * 4 + 1];
            for (pseudo = 0; pseudo < 2; ++pseudo) {
                int64_t lane = 2 * ch + pseudo, data = 2 * channels + lane;
                partial_busy[2 + pseudo] = resources[data * 4 + 1];
                partial_moved[pseudo] = moved[data];
                moved[data] = saved_moved[pseudo];
                memcpy(tables.partial_stats + pseudo * stat_fields, stats + lane * stat_fields,
                       (size_t)stat_fields * sizeof(double));
                memcpy(stats + lane * stat_fields, tables.saved_stats + pseudo * stat_fields,
                       (size_t)stat_fields * sizeof(double));
            }
            if (capture >= 0) {
                reach = fr_max(fr_max(reach, resources[row_bus * 4]), resources[column_bus * 4]);
                fr_capture_after(s, o, capture, ch, !deadline_reached, refreshed ? reach_open : reach, reach,
                                 partial_moved, tables.partial_stats, ch_int, ch_float, tally);
            }
            fr_merge_tallies(s, ch, saved_busy, partial_busy, partial_moved, tables.partial_stats, ch_int, ch_float,
                             tally, a->channel_counts + (ch - a->channel_first) * 7);
        }
    channel_done:
        {
            /* Bill configured occupancies once per channel, independent of
             * mirroring, idle REFpb folding and the order of double additions. */
            double *bill = a->channel_counts + (ch - a->channel_first) * 7;
            bill[5] = bill[0] * s->act_duration + (bill[1] + bill[4]) * s->pre_duration;
            bill[6] = (bill[2] + bill[3]) * column_duration + (g[7] ? bill[4] * t[0] : 0.);
            resources[row_bus * 4 + 1] = fr_after(initial_busy[0], bill[5], half);
            resources[column_bus * 4 + 1] = fr_after(initial_busy[1], bill[6], half);
            for (pseudo = 0; pseudo < 2; ++pseudo) {
                int64_t data = 2 * channels + 2 * ch + pseudo;
                double bursts = (double)((moved[data] - initial_moved[pseudo]) / g[9]);
                resources[data * 4 + 1] = fr_after(initial_busy[2 + pseudo], bursts * t[26], half);
            }
        }
    }
}

int fr_run_channels(const fr_args *args)
{
    jmp_buf escape;
    char *volatile arena = NULL;
    int code;
    fr_trace_open();
    fr_escape = &escape;
    code = setjmp(escape);
    if (code == 0)
        fr_channels(args, &arena);
    if(code == 0) fr_audit_save(args);
    free(arena);
    fr_escape = NULL;
    fr_trace_close();
    return code;
}

/* Transfer buffered observations to the GIL owner. No external audit file is
 * written until the physical transaction commits. */
FILE *fr_audit_take_file(int summary)
{
    FILE *result=summary ? fr_trace_summary_file : fr_trace_file;
    if(summary) fr_trace_summary_file=NULL; else fr_trace_file=NULL;
    return result;
}
