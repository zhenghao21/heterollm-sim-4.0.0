"""Read-only QPC/read-window assessment. Sampling cadence is not a kernel calibration."""
from __future__ import annotations
import math
import statistics


def percentile(values, fraction):
    values = sorted(values)
    if not values: return None
    i = (len(values) - 1) * fraction; lo = int(i)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (i - lo)


def distribution(values):
    if not values: return {'count': 0}
    return {'count': len(values), 'minimum': min(values), 'median': statistics.median(values),
            'p90': percentile(values, .9), 'p99': percentile(values, .99), 'maximum': max(values)}


def summarize(raw):
    f = raw['qpc_frequency']; samples = raw['samples']; errors = []
    for row in samples:
        ticks = [row[n] for n in ('sample_begin_qpc', 'sm_read_begin_qpc', 'sm_read_end_qpc', 'sample_end_qpc')]
        if ticks != sorted(ticks): errors.append('raw_QPC_order_invalid')
    starts = [r['sample_begin_qpc'] for r in samples]
    if any(b <= a for a, b in zip(starts, starts[1:])): errors.append('duplicate_or_reverse_sample_start')
    indexes = [r['index'] for r in samples]
    if indexes != sorted(set(indexes)): errors.append('duplicate_or_reverse_deadline_index')
    valid = [r for r in samples if r['sm_mhz'].get('status') == 0 and type(r['sm_mhz'].get('value')) is int]
    if len(valid) != len(samples): errors.append('SM_readback_failure')
    intervals = [(b-a)*1000/f for a,b in zip(starts, starts[1:])]
    reads = [(r['sm_read_end_qpc']-r['sm_read_begin_qpc'])*1000/f for r in samples]
    wake = [r['wake_lateness_qpc']*1000/f for r in samples]
    values = sorted({r['sm_mhz']['value'] for r in valid})
    return {'status': 'cadence_diagnostic_complete' if not errors else 'cadence_diagnostic_with_errors',
        'errors': errors, 'requested_period_ms': raw['period_ns']/1e6,
        'requested_slots': raw['requested_slots'], 'captured_samples': len(samples),
        'deadline_slots_missed': sum(max(0,r['next_index']-r['first_index']) for r in raw['missed_deadlines']),
        'sample_start_intervals_ms': distribution(intervals), 'SM_read_cost_ms': distribution(reads),
        'wake_lateness_ms': distribution(wake), 'intervals_over25ms': sum(v>25 for v in intervals),
        'intervals_in4_to6ms_fraction': sum(4<=v<=6 for v in intervals)/max(1,len(intervals)),
        'intervals_in4_to8ms_fraction': sum(4<=v<=8 for v in intervals)/max(1,len(intervals)),
        'observed_SM_clock_MHz': values,
        'observed_values_in2400plusminus30': bool(values) and all(abs(v-2400)<=30 for v in values),
        'formal_clock_gate': {'status':'not_evaluated','reason':'No inference/formal kernel was run; cadence test alone cannot prove the inference clock domain.'},
        'cost_calibration_eligible': False, 'quality_thresholds_changed': False}


def formal_clock_gate(raw, formal_intervals, *, maximum_gap_ms=25, target_mhz=2400, tolerance_mhz=30):
    """Bracket using explicit SM READ WINDOWS, with original frozen25ms/2400+/-30.

    Maximum possible separation uses the start of the prior read and end of the
    later read. Single timestamps cannot silently substitute for a read window.
    """
    if (maximum_gap_ms,target_mhz,tolerance_mhz)!=(25,2400,30):
        raise ValueError('formal clock quality thresholds are frozen')
    f = raw.get('qpc_frequency')
    if type(f) is not int or f<=0: return {'passed':False,'issues':['invalid_QPC_frequency'],'coverage':[]}
    samples=raw.get('samples',[]);issues=[];ordered=[]
    for sample in samples:
        lo,hi=sample.get('sm_read_begin_qpc'),sample.get('sm_read_end_qpc')
        if type(lo) is not int or type(hi) is not int or lo>hi:
            issues.append('missing_or_invalid_SM_read_window');continue
        ordered.append(sample)
    ordered.sort(key=lambda r:r['sm_read_begin_qpc'])
    if len({(r['sm_read_begin_qpc'],r['sm_read_end_qpc']) for r in ordered})!=len(ordered):issues.append('duplicate_read_window')
    if not formal_intervals:issues.append('formal_intervals_missing')
    used=set();coverage=[]
    for row in formal_intervals:
        begin,end=row.get('qpc_start'),row.get('qpc_end')
        if type(begin) is not int or type(end) is not int or begin>end:
            issues.append('invalid_formal_window');continue
        before=[(i,s) for i,s in enumerate(ordered) if s['sm_read_end_qpc']<=begin]
        after=[(i,s) for i,s in enumerate(ordered) if s['sm_read_begin_qpc']>=end]
        if not before or not after:
            issues.append('formal_interval_not_bracketed');continue
        li,left=before[-1];ri,right=after[0]
        before_ms=(begin-left['sm_read_begin_qpc'])*1000/f
        after_ms=(right['sm_read_end_qpc']-end)*1000/f
        if max(before_ms,after_ms)>25:issues.append('formal_clock_bracket_too_wide')
        used.update(range(li,ri+1))
        coverage.append({'index':row.get('index'),'before_read_begin_qpc':left['sm_read_begin_qpc'],
            'before_read_end_qpc':left['sm_read_end_qpc'],'after_read_begin_qpc':right['sm_read_begin_qpc'],
            'after_read_end_qpc':right['sm_read_end_qpc'],'maximum_before_gap_ms':before_ms,'maximum_after_gap_ms':after_ms})
    values=[]
    for i in sorted(used):
        value=ordered[i].get('sm_mhz',{})
        if value.get('status')!=0 or type(value.get('value')) is not int:
            issues.append('SM_readback_failed');continue
        values.append(value['value'])
        if abs(value['value']-2400)>30:issues.append('SM_clock_outside_target_domain')
    if not values:issues.append('no_valid_bracketing_readback')
    return {'passed':not issues,'issues':sorted(set(issues)),'coverage':coverage,'target_SM_MHz':2400,
        'tolerance_MHz':30,'maximum_bracket_ms':25,'measured_clock_locked_inferred':False,
        'scope':'Sampled SM read windows, not a continuous per-kernel clock waveform.'}
