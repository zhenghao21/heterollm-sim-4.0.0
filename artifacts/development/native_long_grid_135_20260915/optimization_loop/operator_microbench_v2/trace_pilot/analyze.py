"""Read-only Nsight SQLite extraction for the fixed synthetic pilot.

Attribution follows the installed NVIDIA nvtx_gpu_proj_trace report: CUDA API
correlation ID plus global process ID, with the issuing CPU thread contained
in the named external NVTX range. Kernel duration is an interval union, not a
sum of overlapping kernels, and is never added to the enclosing host wall.
"""
from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
from pilot import HERE, BASE, ref, utc, verify_build, write_json

MARKER = re.compile(r'^GENERIC_GEMM\|phase=(first_call|warmup|formal)\|index=(\d+)\|format=Q4_K\|M=64\|N=4096\|K=1024$')
GLOBAL_PID_MASK = 0xFFFFFFFFFF000000
ATTRIBUTION_SOURCE = Path(r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\reports\nvtx_gpu_proj_trace.py')


def union_ns(intervals):
    total = 0
    left = right = None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('invalid backwards event interval')
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0)


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0, 'minimum_ns': None, 'median_ns': None, 'p95_nearest_rank_ns': None,
                'maximum_ns': None, 'mean_ns': None}
    return {'count': len(values), 'minimum_ns': values[0], 'median_ns': statistics.median(values),
            'p95_nearest_rank_ns': values[math.ceil(len(values) * 0.95) - 1],
            'maximum_ns': values[-1], 'mean_ns': statistics.mean(values)}


def attributed_kernels(marker, apis, kernels):
    launches = [row for row in apis if row['globalTid'] == marker['globalTid']
        and marker['start'] <= row['start'] and row['end'] <= marker['end']]
    keys = {(row['globalTid'] & GLOBAL_PID_MASK, row['correlationId']) for row in launches}
    attributed = [row for row in kernels if (row['globalPid'], row['correlationId']) in keys]
    return launches, attributed


def clock_summary(label):
    source = HERE / label / 'telemetry.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    result = {'source': ref(source), 'sample_count': len(data['samples']), 'clock_locked': False,
              'scope': 'whole process; 50 ms sampling can miss individual short calls',
              'gpu_uuid': data['gpu_identity']['uuid']}
    for key in ('graphics_mhz', 'sm_mhz', 'memory_mhz', 'pstate', 'temperature_c', 'power_mw'):
        values = [row[key]['value'] for row in data['samples']
                  if key in row and row[key].get('status') == 0]
        result[key] = {'min': min(values), 'max': max(values), 'median': statistics.median(values)} if values else {'status': 'unavailable'}
    return result


def module_check(data, manifest):
    actual = {row['name'].lower(): row for row in data['loaded_modules_before']}
    expected = {Path(row['path']).name.lower(): row for row in [manifest['executable'], *manifest['dependencies']]}
    comparisons = []
    for name in actual.keys() & expected.keys():
        row, wanted = actual[name], expected[name]
        comparisons.append({'name': name, 'matches': row['sha256'] == wanted['sha256']
            and Path(row['path']).resolve() == Path(wanted['path']).resolve(), 'actual': row})
    return {'module_snapshot_stable_within_process': data['modules_stable'],
        'frozen_modules_match': all(row['matches'] for row in comparisons),
        'frozen_module_comparisons': comparisons, 'manifest_dependencies_not_loaded': sorted(expected.keys() - actual.keys()),
        'additional_runtime_module_refs': [actual[name] for name in sorted(actual.keys() - expected.keys())]}


def analyze():
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    manifest = verify_build()
    source = HERE / 'profile' / 'gemm.sqlite'
    profile = json.loads((HERE / 'profile' / 'microbench.json').read_text(encoding='utf-8'))
    control = json.loads((HERE / 'direct_control' / 'microbench.json').read_text(encoding='utf-8'))
    connection = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    table_names = {r['name'] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {
        'StringIds': {'id', 'value'}, 'NVTX_EVENTS': {'start','end','text','globalTid'},
        'CUPTI_ACTIVITY_KIND_KERNEL': {'start','end','globalPid','correlationId','demangledName','shortName','deviceId'},
        'CUPTI_ACTIVITY_KIND_RUNTIME': {'start','end','globalTid','correlationId','nameId','returnValue'},
        'DIAGNOSTIC_EVENT': {'text','severity'}, 'TARGET_INFO_GPU': {'id','name','uuid','l2CacheSize'}}
    schema = {}
    for name, columns in required.items():
        if name not in table_names:
            raise ValueError('missing captured event table: ' + name)
        actual = {row['name'] for row in connection.execute('PRAGMA table_info("' + name + '")')}
        if not columns.issubset(actual):
            raise ValueError('unrecognized event schema: ' + name)
        schema[name] = sorted(actual)
    ids = dict(connection.execute('SELECT id,value FROM StringIds'))
    read = lambda table: [dict(row) for row in connection.execute('SELECT rowid AS evidence_rowid,* FROM "' + table + '"')]
    nvtx = read('NVTX_EVENTS')
    kernels = read('CUPTI_ACTIVITY_KIND_KERNEL')
    runtime = read('CUPTI_ACTIVITY_KIND_RUNTIME')
    diagnostics = read('DIAGNOSTIC_EVENT')
    gpu_info = read('TARGET_INFO_GPU')
    sync = read('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION') if 'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION' in table_names else []
    memcpy = read('CUPTI_ACTIVITY_KIND_MEMCPY') if 'CUPTI_ACTIVITY_KIND_MEMCPY' in table_names else []
    for row in kernels:
        row['demangled_name_text'] = ids[row['demangledName']]
        row['short_name_text'] = ids[row['shortName']]
    for row in runtime:
        row['name_text'] = ids[row['nameId']]
    for row in nvtx:
        row['resolved_text'] = row['text'] if row['text'] is not None else ids.get(row.get('textId'))
    compatibility = [row for row in diagnostics if 'not supported' in row['text'].lower()
                     or 'unsupported' in row['text'].lower() or 'using libraries for' in row['text'].lower()]
    warnings = [row for row in diagnostics if row['severity'] >= 2]
    markers = sorted([row for row in nvtx if MARKER.fullmatch(row['resolved_text'] or '')], key=lambda row: row['start'])
    call_runs = {(row['phase'], row['index']): row for row in profile['runs']}
    if len(markers) != len(call_runs):
        raise ValueError('not every synthetic call has exactly one NVTX marker')
    rows, seen, selected = [], set(), set()
    for marker in markers:
        match = MARKER.fullmatch(marker['resolved_text'])
        key = (match.group(1), int(match.group(2)))
        if key in seen or key not in call_runs:
            raise ValueError('duplicate or unmatched invocation marker')
        seen.add(key)
        launch_apis, owned = attributed_kernels(marker, runtime, kernels)
        for kernel in owned:
            if kernel['evidence_rowid'] in selected:
                raise ValueError('one kernel mapped to multiple nonnested synthetic invocations')
            selected.add(kernel['evidence_rowid'])
        internal_sync = [event for event in launch_apis if 'Synchronize' in event['name_text']]
        union = union_ns((event['start'], event['end']) for event in owned)
        rows.append({'phase': key[0], 'index': key[1], 'qpc_derived_run_verbatim': call_runs[key],
            'nvtx_range': {key: marker[key] for key in ('start','end','globalTid','resolved_text','evidence_rowid')},
            'nvtx_range_duration_ns': marker['end'] - marker['start'],
            'kernel_count': len(owned), 'cuda_kernel_union_ns': union,
            'cuda_kernel_sum_ns_diagnostic_only': sum(event['end'] - event['start'] for event in owned),
            'kernel_span_ns': max(event['end'] for event in owned) - min(event['start'] for event in owned) if owned else 0,
            'all_kernels_contained_in_nvtx': all(marker['start'] <= event['start'] and event['end'] <= marker['end'] for event in owned),
            'host_cuda_runtime_api_union_ns': union_ns((event['start'], event['end']) for event in launch_apis),
            'host_cuda_synchronize_api_union_ns': union_ns((event['start'], event['end']) for event in internal_sync),
            'kernel_events': owned, 'runtime_api_events': launch_apis,
            'timing_policy': 'Host wall includes launch and synchronize. Kernel union is a nested GPU view. Never add these fields.'})
    outside = [row for row in kernels if row['evidence_rowid'] not in selected]
    all_correct = (profile['status'] == control['status'] == 'measured'
        and all(data[stage]['passed'] and data[stage]['finite_all_outputs']
            for data in (profile, control) for stage in ('first_call_correctness','final_correctness')))
    invariants = {
        'expected_24_nvtx_ranges': len(markers) == 24,
        'expected_24_application_calls_each': profile['graph_compute_calls'] == control['graph_compute_calls'] == 24,
        'all_application_compute_status_success': all(row['ggml_status'] == 0 for data in (profile, control) for row in data['runs']),
        'exact_shape_each': all(all(data[key] == protocol['shape'][key] for key in ('M','N','K')) for data in (profile, control)),
        'same_quantized_weight_and_input_hashes': profile['quantization'] == control['quantization'],
        'same_explicit_app_environment': profile['environment'] == control['environment'],
        'annotations_explicitly_zero': profile['environment']['LLAMA_TRACE_ANNOTATIONS'] == '0',
        'cuda_graphs_explicitly_disabled': profile['environment']['GGML_CUDA_DISABLE_GRAPHS'] == '1',
        'all_24_calls_have_three_correlated_kernels': all(row['kernel_count'] == 3 for row in rows),
        'all_correlated_kernels_within_outer_nvtx': all(row['all_kernels_contained_in_nvtx'] for row in rows),
        'all_24_eviction_kernels_outside_measured_ranges': len(outside) == 24 and all(row['short_name_text'] == 'scale_f32' for row in outside),
        'eviction_kernels_have_no_overlap_with_nvtx': all(event['end'] <= marker['start'] or event['start'] >= marker['end'] for event in outside for marker in markers),
        'no_internal_llm_nvtx_annotations': len(nvtx) == len(markers),
        'gpu_uuid_matches_protocol': any('GPU-' + row['uuid'] == protocol['gpu_identity']['uuid'] and row['id'] == 0 for row in gpu_info),
        'first_and_final_numerical_check_pass': all_correct,
        'tolerances_unchanged': all(data['correctness_contract']['absolute_tolerance'] == 0.05 and data['correctness_contract']['relative_tolerance'] == 0.03 and data['final_correctness']['sample_count'] == 256 for data in (profile, control))}
    modules = {'profile': module_check(profile, manifest), 'direct_control': module_check(control, manifest)}
    invariants['frozen_loaded_module_identity_matches'] = all(data['frozen_modules_match'] and data['module_snapshot_stable_within_process'] for data in modules.values())
    signatures = lambda data: sorted((row['name'],row['path'],row['sha256']) for row in data['loaded_modules_before'])
    invariants['same_loaded_runtime_module_identities'] = signatures(profile) == signatures(control)
    host_profile = distribution([row['elapsed_ns'] for row in profile['runs'] if row['phase'] == 'formal'])
    host_control = distribution([row['elapsed_ns'] for row in control['runs'] if row['phase'] == 'formal'])
    gpu_formal = distribution([row['cuda_kernel_union_ns'] for row in rows if row['phase'] == 'formal'])
    delta = 100 * (host_profile['median_ns'] / host_control['median_ns'] - 1)
    raw_events = {'schema': 'synthetic-pilot-nsys-event-extract/v1', 'source_sqlite': ref(source),
        'timestamp_unit': 'ns, Nsight domain; separate from application relative steady_clock epoch',
        'nvtx': nvtx, 'cuda_kernels': kernels, 'cuda_runtime': runtime, 'cuda_sync': sync, 'cuda_memcpy': memcpy,
        'diagnostics': diagnostics, 'publication': 'raw event data: keep local only'}
    write_json(HERE / 'profile' / 'raw_events.json', raw_events)
    write_json(HERE / 'mapped_calls.json', {'schema': 'synthetic-pilot-correlated-calls/v1',
        'protocol': ref(HERE / 'protocol.json'), 'source_sqlite': ref(source),
        'attribution_reference': ref(ATTRIBUTION_SOURCE), 'clock_epochs_not_aligned': True,
        'host_and_kernel_union_not_additive': True, 'calls': rows, 'outside_measured_kernel_events': outside})
    summary = {'schema': 'synthetic-gemm-trace-pilot-result/v1', 'created_utc': utc(),
        'collection_status': 'completed_with_unsupported_driver_warning' if compatibility else 'completed',
        'structural_extraction_status': 'passed' if all(invariants.values()) else 'failed',
        'source_protocol': ref(HERE / 'protocol.json'), 'source_sqlite': ref(source),
        'source_nsys_report': ref(HERE / 'profile' / 'gemm.nsys-rep'),
        'extraction_tool': ref(Path(__file__)), 'schema_columns_checked': schema,
        'attempt_counts': {'profile': 1, 'direct_control': 1, 'export_sqlite': 1, 'retries': 0},
        'shape': protocol['shape'], 'environment': profile['environment'],
        'invariants': invariants, 'compatibility_gate_passed': not compatibility,
        'compatibility_evidence': compatibility, 'all_profiler_warnings': warnings,
        'counts': {'nvtx': len(nvtx), 'kernels_total': len(kernels), 'kernels_in_gemm': len(selected),
            'eviction_kernels_outside_gemm': len(outside), 'runtime_api_events': len(runtime)},
        'observed_kernel_names': [{'demangled_name': name, 'count': count} for name,count in Counter(row['demangled_name_text'] for row in kernels).items()],
        'formal_host_profile': host_profile, 'formal_host_direct_control': host_control,
        'formal_cuda_kernel_union_profile': gpu_formal,
        'profiling_comparison': {'median_relative_change_percent': delta,
            'interpretation': 'one unlocked ordered pair only; timing difference is confounded by drift and profiling. Negative difference is not proof of negative profiler overhead. Neither run is calibration truth.'},
        'first_and_final_correctness': {label:{stage:{k:data[stage][k] for k in ('passed','finite_all_outputs','sample_count','max_absolute_error','max_relative_error','rmse')} for stage in ('first_call_correctness','final_correctness')} for label,data in [('profile',profile),('direct_control',control)]},
        'actual_eviction': {'requested_mib': 128, 'device_l2_bytes': profile['gpu_l2_bytes'],
            'effective_input_bytes': profile['cache_eviction_bytes'], 'sweep_reads_and_writes_separate_buffers': True,
            'four_l2_rule_satisfied': profile['cache_eviction_bytes'] >= 4*profile['gpu_l2_bytes'],
            'dram_transaction_truth_proven': False},
        'clocks': {label:clock_summary(label) for label in ('profile','direct_control')},
        'modules': modules, 'frozen_build_refs_still_match': True,
        'absolute_per_call_qpc_ticks_available': False, 'qpc_derived_call_timestamps_preserved': True,
        'host_and_cuda_are_not_additive': True, 'native_llm_executed': False,
        'calibration_eligible': False, 'llm_latency_fitting': False, 'less_than_5_percent_claim_allowed': False,
        'limits': ['Nsight reports unsupported CUDA 13.4 driver and fallback to CUDA 12.8 collection libraries.',
            'No frequency lock, thermal control, randomized order, or repeated process-level profiling comparison.',
            'Existing executable lacks absolute raw per-call QPC ticks; relative nanosecond values retained exactly.',
            'Original transitive header dependency closure is incomplete; supplemental hashes cannot prove original includes.',
            'Only 256 uniform output samples checked at first and last call, plus all-output finite checks; no full numeric comparison for every call.',
            'Cache eviction is an explicit sweep, not measured HBM traffic; kernel view is nested in the host wall.']}
    write_json(HERE / 'result_summary.json', summary)
    connection.close()
    print(json.dumps({'collection_status': summary['collection_status'], 'structural_extraction_status': summary['structural_extraction_status'],
        'counts': summary['counts'], 'profile_host_median_ns':host_profile['median_ns'],
        'direct_host_median_ns':host_control['median_ns'],'kernel_union_median_ns':gpu_formal['median_ns'],
        'failed_invariants':[key for key,value in invariants.items() if not value], 'calibration_eligible':False},ensure_ascii=False))
    return summary


if __name__ == '__main__':
    analyze()
