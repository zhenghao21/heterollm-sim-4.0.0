"""Append the 27 GPU cells and select an immutable, native-only stable dataset.

Consumer contract (native-stable-dataset/v1): selected_cells[] contains cell_id,
model_key, prompt_tokens/output_tokens/parallel, config (the frozen native plan),
metrics[ttft|tpot|e2e], native_actuals[] (block, repeat, request_index,
engine_start_rank, metrics_ms, raw_ref and JSON pointers), source_per_cell, and
static_hardware (frozen_hardware, hardware_refs, configured_clock, state_refs).
excluded_cells[] preserves every exclusion reason; coverage has all six groups.
The selector never reads simulator predictions, fits, or prediction errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import native_long_grid_report as legacy

METRICS = ('ttft', 'tpot', 'e2e')
BASE_GROUPS = ('qwen25', 'qwen35', 'qwen38', 'smollm2', 'tinyllama')
GROUPS = (*BASE_GROUPS, 'qwen38_gpu')
GRID = tuple(itertools.product((128, 512, 1536), (32, 128, 256), (1, 2, 4)))
THRESHOLD_PCT = 5.0
GPU_EXPLICIT_LAYERS = 66
CLOCK_EXCEPTION_COORDINATES = frozenset(((128, 128, 2), (512, 128, 2),
                                         (1536, 128, 2), (512, 32, 2)))
CLOCK_EXCEPTION_AUTHORIZATION = '那四格硬件状态失败的也进入有效记录吧，频率没有差很多'
VERIFICATION_SCOPE = {
    'rechecked': [
        'Frozen source/protocol/source snapshot hashes and existing raw/receipt/config bindings are verified through native_long_grid_report.',
        'Saved request engine timepoints are parsed again and the six native deviation statistics require full positive sample counts and strict <5%.',
        'GPU plans require exactly 66 requested layers with fit_params=false; same GGUF bytes and matching frozen native exe/DLL hashes are compared.',
        'CPU/GPU effective allowlisted environments are compared per corresponding plan, including is_set and value; no process-noise normalization is applied.',
    ],
    'retained_from_capture_not_fully_reexecuted': [
        'Complete capture status and bound runtime-baseline/before/after records retain the collector acceptance; every runtime/state acceptance predicate is not rerun by this report.',
        'Loaded runtime and hardware/clock snapshots remain readable through raw and baseline references; no live hardware, clock, affinity, or native-process probe occurs.',
        'Large GGUF and native runtime binaries are compared by frozen identities rather than rehashed by this report.',
    ],
}
SCHEMA = 'native-stable-dataset/v1'
LIMITATIONS = [
    'Three measured repeats in one process are a development initial screen only.',
    'Strictly less than 5% observed native variability is not a formal repeatability guarantee.',
    'Selection is independent of simulator predictions and errors; no prediction acceptance is implied.',
    'CPU and GPU placements and all freeze/protocol/extractor identities remain separate.',
    'The original 135-cell denominator and failure history are retained; 27 GPU cells are appended.',
    'Large GGUF and runtime files use their frozen SHA256 identities; this report does not rehash large binaries.',
]


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    return value


def _finite(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _coordinate(row):
    return row['prompt_tokens'], row['output_tokens'], row['parallel']


def _plan_coordinate(plan):
    return (plan.get('expected_prompt_tokens', len(plan.get('prompt_token_ids', []))),
            plan['output'], plan['parallel'])


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None


def _bound_artifact(freeze, path):
    normalized = str(Path(path).resolve()).casefold()
    refs = [r for r in freeze.get('artifact_refs', [])
            if str(Path(r.get('path', '')).resolve()).casefold() == normalized]
    if len(refs) != 1 or not _sha(refs[0].get('sha256')):
        raise ValueError('missing_or_ambiguous_frozen_artifact: ' + str(path))
    return refs[0]


def _runtime_refs(freeze):
    refs = [r for r in freeze.get('artifact_refs', [])
            if Path(r.get('path', '')).suffix.lower() in ('.exe', '.dll')]
    digests = legacy._runtime_digests(freeze)
    if not any(name.endswith('.dll') for name in digests) or not all(_sha(v) for v in digests.values()):
        raise ValueError('native_exe_or_dll_identity_missing')
    return refs, digests


def strict_exclusion_reasons(row):
    """Reject missing, nonfinite, zero-count, failed, unverified, or >=5% cells."""
    reasons = []
    if row.get('status') != 'complete':
        reasons.append('cell_incomplete_or_failed')
    if row.get('evidence_verified') is not True:
        reasons.append('evidence_or_freeze_not_verified')
    if row.get('diagnostic_only') is True:
        reasons.append('diagnostic_only')
    for planned, captured, label in (('planned_blocks', 'complete_blocks', 'process_blocks'),
                                      ('planned_batches', 'captured_batches', 'measurement_batches')):
        n = row.get(planned)
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or row.get(captured) != n:
            reasons.append(label + '_coverage_incomplete_or_zero')
    raw_metrics = row.get('raw_summary', {}).get('metrics', {})
    for metric in METRICS:
        value = row.get('metrics', {}).get(metric, {})
        if value.get('status') != 'measured' or value.get('diagnostic_only') is True:
            reasons.append(metric + '_not_measured')
        if not _finite(value.get('native_median_ms'), positive=True):
            reasons.append(metric + '_native_median_invalid')
        for field in ('batch_worst_abs_pct', 'rank_worst_abs_pct'):
            pct = value.get(field)
            if not _finite(pct):
                reasons.append(metric + '_' + field + '_missing_or_nonfinite')
            elif pct >= THRESHOLD_PCT:
                reasons.append(metric + '_' + field + '_not_strictly_below_5')
        source = raw_metrics.get(metric, {})
        distributions = [source.get('batch_medians', {})]
        ranks = source.get('request_ranks', [])
        if len(ranks) != row.get('parallel') or {r.get('rank') for r in ranks} != set(range(row.get('parallel', 0))):
            reasons.append(metric + '_rank_coverage_invalid')
        distributions.extend(ranks)
        if any(not isinstance(d.get('planned_samples'), int) or d.get('planned_samples', 0) <= 0
               or d.get('observed_samples') != d.get('planned_samples')
               or d.get('planned_samples') != row.get('planned_batches') for d in distributions):
            reasons.append(metric + '_sample_coverage_incomplete_or_zero')
    return list(dict.fromkeys(reasons + row.get('metadata_errors', []) + row.get('actuals_errors', [])))


def _validate_original_grid(base):
    expected = {(model, *point) for model in BASE_GROUPS for point in GRID}
    actual = [(r.get('model_key'), *_coordinate(r)) for r in base.get('cells', [])]
    if base.get('planned_cells') != 135 or len(actual) != 135 or set(actual) != expected:
        raise ValueError('original_report_must_preserve_exact_135_cell_grid')
    if len({r['cell_id'] for r in base['cells']}) != 135:
        raise ValueError('duplicate_original_cell_ids')


def _identity_errors(source, freeze):
    errors = []
    if source.get('end_verified') is not True or source.get('verification_errors'):
        errors.append('campaign_end_or_freeze_unverified')
    if freeze.get('schema') != 'native-repeatability-freeze/v1':
        errors.append('freeze_schema_invalid')
    for name in ('freeze_ref', 'protocol_ref', 'source_snapshot_ref'):
        ref = source.get(name, {}) or {}
        if not ref.get('path') or not _sha(ref.get('sha256')):
            errors.append(name + '_identity_missing')
    extractor = freeze.get('extractor_identity', {})
    if not _sha(extractor.get('sha256')) or not _sha(extractor.get('file_sha256')):
        errors.append('extractor_identity_missing')
    elif not any(r.get('sha256') == extractor['file_sha256']
                 and Path(r.get('path', '')).name == 'native_llama_compare.py'
                 for r in freeze.get('source_refs', [])):
        errors.append('extractor_not_bound_to_frozen_source')
    return errors



def _frozen_environment(freeze, plan):
    environments = freeze.get('environments')
    value = environments.get(plan['key']) if isinstance(environments, dict) else None
    if not isinstance(value, dict) or set(value) != set(legacy.experiment.ENV_KEYS):
        raise ValueError('frozen_effective_environment_missing_or_invalid')
    for entry in value.values():
        if (not isinstance(entry, dict) or set(entry) != {'is_set', 'value'}
                or type(entry['is_set']) is not bool
                or (entry['is_set'] and not isinstance(entry['value'], str))
                or (not entry['is_set'] and entry['value'] is not None)):
            raise ValueError('frozen_effective_environment_missing_or_invalid')
    return value


def _gpu_metadata(cpu_freeze, gpu_freeze):
    """Compare immutable identities, never launch a native/model/profiler process."""
    errors = []
    cpu = {_plan_coordinate(p): p for p in cpu_freeze['plans'] if p.get('model_key') == 'qwen38'}
    gpu = {_plan_coordinate(p): p for p in gpu_freeze['plans']}
    if set(cpu) != set(GRID) or set(gpu) != set(GRID) or len(gpu_freeze['plans']) != 27:
        errors.append('gpu_or_cpu_qwen38_grid_mismatch')
    try:
        _, cpu_runtime = _runtime_refs(cpu_freeze)
        _, gpu_runtime = _runtime_refs(gpu_freeze)
        if cpu_runtime != gpu_runtime:
            errors.append('gpu_native_exe_or_dll_sha256_mismatch')
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))
    if not cpu_freeze.get('hardware_fingerprint') or cpu_freeze.get('hardware_fingerprint') != gpu_freeze.get('hardware_fingerprint'):
        errors.append('gpu_hardware_identity_mismatch')
    for point, plan in gpu.items():
        original = cpu.get(point)
        if original is None:
            continue
        if plan.get('model_key') != 'qwen38_gpu':
            errors.append('gpu_model_key_must_be_qwen38_gpu')
        if original.get('gpu_layers') != 0:
            errors.append('cpu_gpu_placement_identity_invalid')
        if type(plan.get('gpu_layers')) is not int or plan['gpu_layers'] != GPU_EXPLICIT_LAYERS:
            errors.append('gpu_layers_must_be_explicit_66')
        if plan.get('fit_params') is not False:
            errors.append('gpu_fit_params_must_be_explicit_false')
        try:
            if _frozen_environment(cpu_freeze, original) != _frozen_environment(gpu_freeze, plan):
                errors.append('gpu_effective_environment_mismatch')
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
        try:
            if _bound_artifact(cpu_freeze, original['model'])['sha256'] != _bound_artifact(gpu_freeze, plan['model'])['sha256']:
                errors.append('gpu_gguf_sha256_mismatch')
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
        tokens = plan.get('prompt_token_ids')
        if (not isinstance(tokens, list) or len(tokens) != point[0]
                or tokens != original.get('prompt_token_ids')):
            errors.append('gpu_prompt_token_ids_mismatch')
        # Explicit GPU placement disables auto-fit. CPU may omit fit_params;
        # this approved placement control is not a source/protocol identity merge.
        excluded = {'model_key', 'job_id', 'key', 'model', 'gpu_layers', 'fit_params'}
        before = {k: v for k, v in original.items() if k not in excluded}
        after = {k: v for k, v in plan.items() if k not in excluded}
        if before != after:
            errors.append('gpu_nonplacement_configuration_mismatch')
    return list(dict.fromkeys(errors))




def _clock_supplement_metadata(base_rows, sources, freezes, supplement):
    """Bind exactly four new complete captures to the approved clock exception."""
    freeze = supplement['freeze']
    errors, comparisons = [], []
    expected = {('smollm2', *point) for point in CLOCK_EXCEPTION_COORDINATES}
    plans = freeze.get('plans', [])
    actual = [(plan.get('model_key'), *_plan_coordinate(plan)) for plan in plans]
    if len(plans) != 4 or set(actual) != expected:
        errors.append('clock_exception_requires_exact_four_smol_cells')
    original_rows = {(row['model_key'], *_coordinate(row)): row for row in base_rows}
    for plan in plans:
        coordinate = (plan.get('model_key'), *_plan_coordinate(plan))
        if coordinate not in expected:
            continue
        row = original_rows.get(coordinate)
        if row is None:
            errors.append('clock_exception_original_cell_missing')
            continue
        source_id = row['source_per_cell']['source_id']
        old_freeze = freezes[source_id]
        original_plans = [item for item in old_freeze['plans']
                          if item['job_id'] + '__' + item['condition_id'] == row['cell_id']]
        if len(original_plans) != 1:
            errors.append('clock_exception_requires_one_original_process_block')
            continue
        original = original_plans[0]
        if row.get('status') == 'complete':
            errors.append('clock_exception_cannot_replace_complete_original_cell')
        if (plan.get('gpu_sm_clock_tolerance_mhz') != 60
                or original.get('gpu_sm_clock_tolerance_mhz') != 30
                or plan.get('expected_gpu_sm_clock_mhz') != 2400
                or original.get('expected_gpu_sm_clock_mhz') != 2400):
            errors.append('clock_exception_requires_2400mhz_30_to_60mhz_tolerance')
        if not old_freeze.get('hardware_fingerprint') or old_freeze.get('hardware_fingerprint') != freeze.get('hardware_fingerprint'):
            errors.append('clock_exception_hardware_identity_mismatch')
        try:
            old_model = _bound_artifact(old_freeze, original['model'])
            new_model = _bound_artifact(freeze, plan['model'])
            if old_model['sha256'] != new_model['sha256']:
                errors.append('clock_exception_gguf_sha256_mismatch')
            if _runtime_refs(old_freeze)[1] != _runtime_refs(freeze)[1]:
                errors.append('clock_exception_native_exe_or_dll_sha256_mismatch')
            if _frozen_environment(old_freeze, original) != _frozen_environment(freeze, plan):
                errors.append('clock_exception_effective_environment_mismatch')
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
        # Model path relocation is permitted only through the equal SHA above.
        excluded = {'model', 'gpu_sm_clock_tolerance_mhz'}
        if ({k: v for k, v in original.items() if k not in excluded}
                != {k: v for k, v in plan.items() if k not in excluded}):
            errors.append('clock_exception_other_configuration_mismatch')
        comparisons.append({'cell_id': row['cell_id'], 'original_status': row['status'],
                            'original_source_per_cell': copy.deepcopy(row['source_per_cell']),
                            'original_freeze_ref': sources[source_id]['freeze_ref'],
                            'supplement_freeze_ref': supplement['identity']['freeze_ref'],
                            'expected_gpu_sm_clock_mhz': 2400,
                            'original_tolerance_mhz': 30, 'exception_tolerance_mhz': 60})
    lineage = {'schema': 'native-clock-exception-supplement-lineage/v1',
               'source_id': 'clock_exception_supplement',
               'status': 'verified' if not errors else 'invalid',
               'user_authorization': CLOCK_EXCEPTION_AUTHORIZATION,
               'exception_scope': 'Only the four named Smol cells use a 60 MHz tolerance around 2400 MHz; all six native variability deviations must still be strictly below 5%.',
               'replacement_strategy': 'New complete 2-warmup/3-measurement captures replace failed cells; original raw/receipt statuses are never changed.',
               'allowed_cell_coordinates': [{'model_key': 'smollm2', 'prompt_tokens': point[0],
                                             'output_tokens': point[1], 'parallel': point[2]}
                                            for point in sorted(CLOCK_EXCEPTION_COORDINATES)],
               'supplement_freeze_ref': supplement['identity']['freeze_ref'],
               'verification_errors': list(dict.fromkeys(errors)), 'cells': comparisons}
    return list(dict.fromkeys(errors)), lineage


def _apply_clock_supplement(rows, sources, freezes, failures, campaign):
    supplement = legacy._campaign(campaign, 'clock_exception_supplement')
    errors, lineage = _clock_supplement_metadata(rows, sources, freezes, supplement)
    sources['clock_exception_supplement'] = supplement['identity']
    freezes['clock_exception_supplement'] = supplement['freeze']
    failures.extend(supplement['failed_attempts'])
    new_rows = {(r['model_key'], *_coordinate(r)): (key, r)
                for key, r in supplement['rows'].items()}
    result = []
    for original in rows:
        coordinate = _coordinate(original)
        if original['model_key'] != 'smollm2' or coordinate not in CLOCK_EXCEPTION_COORDINATES:
            result.append(original)
            continue
        match = new_rows.get(('smollm2', *coordinate))
        if match is None:
            row = copy.deepcopy(original)
            row.setdefault('metadata_errors', []).extend([*errors, 'clock_exception_replacement_cell_missing'])
            result.append(row)
            continue
        key, fresh = match
        row = copy.deepcopy(fresh)
        row.update(cell_id=original['cell_id'], native_cell_id=legacy._cell_id(key),
                   metadata_errors=list(errors), diagnostic_only=not fresh.get('evidence_verified'),
                   source_per_cell={'source_id': 'clock_exception_supplement',
                                    'freeze_ref': supplement['identity']['freeze_ref'],
                                    'extractor_identity': supplement['identity'].get('extractor_identity'),
                                    'replacement': True, 'original_status': original['status'],
                                    'replacement_reason': 'user_approved_clock_tolerance_exception',
                                    'original_source_per_cell': copy.deepcopy(original['source_per_cell'])},
                   clock_exception={'approved': True, 'target_mhz': 2400, 'tolerance_mhz': 60,
                                    'original_tolerance_mhz': 30, 'variability_threshold_unchanged': True},
                   attempt_history=[*copy.deepcopy(original.get('attempt_history', [])),
                                    *[event for event in supplement['failed_attempts']
                                      if event.get('cell_id') == legacy._cell_id(key)]])
        result.append(row)
    return result, lineage


def _hardware_index(source, freeze, plan, evidence):
    """Expose only this cell's own immutable hardware and measured-clock sources.

    `frozen_hardware` is convenient static model input. Its captured clocks may
    predate the sampling window: use configured_clock and state_refs for clocks.
    No current machine probe or simulator output is read here.
    """
    plan_index = freeze['plans'].index(plan)
    state_refs = []
    for entry in evidence:
        state_refs.append({'key': entry['key'], 'raw_ref': entry['raw_ref'],
            'state_before': '/state_before',
            'state_measurement_before': '/state_measurement_before',
            'state_after': '/state_after',
            'runtime_before': '/runtime_before', 'runtime_after': '/runtime_after',
            'actual_argv': '/actual_argv',
            'gpu_clock_before': '/state_measurement_before/gpu_state',
            'gpu_clock_after': '/state_after/gpu_state'})
    hardware_refs = [copy.deepcopy(ref) for ref in freeze.get('artifact_refs', [])
                    if Path(ref.get('path', '')).name.lower() in ('hardware.json', 'cpu_topology.json')]
    config_keys = ('threads', 'threads_batch', 'worker_cpu_mask', 'resolved_cpu_affinity',
                   'gpu_layers', 'fit_params', 'poll', 'poll_batch', 'ctx', 'batch',
                   'ubatch', 'parallel', 'kv_unified_per_slot', 'expected_power_scheme')
    return {'schema': 'native-selected-cell-hardware/v1',
            'source_id': source['source_id'],
            'hardware_fingerprint': freeze.get('hardware_fingerprint'),
            'frozen_hardware': copy.deepcopy(freeze.get('hardware', {})),
            'frozen_hardware_ref': {'file_ref': source['freeze_ref'], 'json_pointer': '/hardware'},
            'hardware_refs': hardware_refs,
            'native_config_ref': {'file_ref': source['freeze_ref'], 'json_pointer': '/plans/' + str(plan_index)},
            'native_config': {key: copy.deepcopy(plan[key]) for key in config_keys if key in plan},
            'configured_clock': {key: plan.get(key) for key in
                                 ('expected_gpu_sm_clock_mhz', 'gpu_sm_clock_tolerance_mhz')},
            'state_refs': state_refs,
            'clock_scope': 'sampling clocks come from raw state_measurement_before/state_after; frozen hardware clocks are capture-time only'}


def _collect_actuals(row, source, freeze, *, records=None):
    """Keep per-request engine measurements and immutable raw locations reusable."""
    errors, actuals, evidence = [], [], []
    key = row['native_cell_id']
    plans = [p for p in freeze['plans'] if p['job_id'] + '__' + p['condition_id'] == key]
    if not plans:
        return {}, ['frozen_cell_plan_missing']
    try:
        model_ref = _bound_artifact(freeze, plans[0]['model'])
        runtime_refs, _ = _runtime_refs(freeze)
    except (KeyError, ValueError) as exc:
        return {}, [str(exc)]
    for plan in plans:
        plan_index = freeze['plans'].index(plan)
        if plan.get('measure_batches') != 3 or plan.get('warmup_batches') != 2 or plan.get('process_blocks') != 1:
            errors.append('initial_screen_plan_must_be_2_warmup_3_measure_1_process')
        try:
            data = Path(source['campaign']) / 'native'
            record = next((r for r in (records or []) if r.get('key') == plan['key']), None)
            if record is None:
                record = legacy.experiment.load_completed(data, freeze, plan)
            if record is None or record.get('status') != 'complete':
                errors.append('verified_complete_raw_missing')
                continue
            raw_ref = record.get('_verified_raw_ref') or legacy.reference(data / (plan['key'] + '.json'))
            receipt_ref = legacy.reference(data / (plan['key'] + '.receipt.json'))
            if not record.get('runtime_baseline_ref'):
                errors.append('loaded_runtime_identity_missing')
            evidence.append({'key': plan['key'], 'raw_ref': raw_ref, 'receipt_ref': receipt_ref,
                             'runtime_baseline_ref': record.get('runtime_baseline_ref'),
                             'batch_journal_refs': record.get('batch_journal_refs', []),
                             'config_ref': {'freeze_ref': source['freeze_ref'], 'json_pointer': '/plans/' + str(plan_index)},
                             'config_sha256': _digest(plan),
                             'hardware_state_refs': {name: {'file_ref': raw_ref, 'json_pointer': '/' + name}
                                 for name in ('state_before', 'state_measurement_before', 'state_after',
                                              'runtime_before', 'runtime_after', 'actual_argv')}})
            for phase in ('warmup', 'runs'):
                batches = record.get(phase, [])
                count = plan['warmup_batches'] if phase == 'warmup' else plan['measure_batches']
                if len(batches) != count:
                    errors.append(phase + '_raw_batch_coverage_incomplete')
                for repeat, batch in enumerate(batches):
                    requests = batch.get('requests', [])
                    if (batch.get('status') != 'complete' or len(requests) != plan['parallel']
                            or {r.get('request_index') for r in requests} != set(range(plan['parallel']))
                            or {r.get('engine_start_rank') for r in requests} != set(range(plan['parallel']))):
                        errors.append(phase + '_raw_request_coverage_incomplete')
                    for index, request in enumerate(requests):
                        timing = request.get('response', {}).get('timings', {})
                        inspected = legacy.experiment.inspect_request(request.get('response', {}),
                            request.get('boundary', {}), plan['output'], row['prompt_tokens'])
                        if inspected.get('status') != 'measured':
                            errors.append(phase + '_invalid_engine_request')
                        values = inspected.get('metrics_ms', {})
                        if not all(_finite(values.get(m), positive=True) for m in METRICS):
                            errors.append(phase + '_nonfinite_or_nonpositive_engine_metrics')
                        if phase == 'runs':
                            pointer = '/runs/' + str(repeat) + '/requests/' + str(index)
                            actuals.append({'block': plan.get('block', 0), 'repeat': repeat,
                                'request_index': request.get('request_index'),
                                'engine_start_rank': request.get('engine_start_rank'),
                                'metrics_ms': values, 'raw_ref': raw_ref, 'raw_pointer': pointer,
                                'engine_timing_pointer': pointer + '/response/timings',
                                'engine_request_begin_us': timing.get('engine_request_begin_us'),
                                'engine_first_token_us': timing.get('engine_prompt_last_us'),
                                'engine_last_token_us': timing.get('engine_last_token_us'),
                                'engine_token_times_count': len(timing.get('engine_token_times_us') or [])})
        except Exception as exc:
            errors.append('raw_evidence_verification_failed: ' + repr(exc))
    if len(actuals) != row.get('planned_batches', 0) * row['parallel'] or not actuals:
        errors.append('actual_request_coverage_incomplete_or_zero')
    return {'config': copy.deepcopy(plans[0]), 'plans': copy.deepcopy(plans),
            'static_hardware': _hardware_index(source, freeze, plans[0], evidence),
            'model_ref': model_ref, 'native_runtime_refs': runtime_refs,
            'native_actuals': actuals, 'evidence_index': evidence,
            'config_sha256': _digest(plans[0]),
            'prompt_token_ids_sha256': _digest(plans[0].get('prompt_token_ids'))}, list(dict.fromkeys(errors))


def _pending_gpu_rows(base):
    rows = []
    for cpu in base['cells']:
        if cpu['model_key'] != 'qwen38':
            continue
        row = copy.deepcopy(cpu)
        row['cell_id'] = cpu['cell_id'].replace('qwen38_', 'qwen38_gpu_', 1)
        row.update(id=cpu['id'].replace('qwen38_', 'qwen38_gpu_', 1), model_key='qwen38_gpu',
                   native_cell_id=row['cell_id'], status='not_collected', evidence_verified=False,
                   diagnostic_only=True, complete_blocks=0, captured_batches=0, accepted_captured_batches=0,
                   metrics={m: {'status': 'missing'} for m in METRICS}, raw_summary={},
                   source_per_cell={'source_id': 'gpu_extension', 'freeze_ref': None, 'replacement': False},
                   attempt_history=[], metadata_errors=['gpu_freeze_missing'])
        rows.append(row)
    return rows


def build_dataset(campaign, gpu_campaign, supplement=None, lineage=None, clock_supplement=None):
    campaign, gpu_campaign = Path(campaign).resolve(), Path(gpu_campaign).resolve()
    if supplement is None and (campaign / 'supplement_sse_v2/native/freeze.json').exists():
        supplement = campaign / 'supplement_sse_v2'
    base = legacy.build_report(campaign, supplement=supplement, lineage=lineage)
    _validate_original_grid(base)
    rows = copy.deepcopy(base['cells'])
    sources = copy.deepcopy(base['sources'])
    freezes = {name: _json(identity['freeze_ref']['path']) for name, identity in sources.items()}
    failures = copy.deepcopy(base.get('failed_attempts', []))
    clock_lineage = None
    if clock_supplement is not None:
        rows, clock_lineage = _apply_clock_supplement(rows, sources, freezes, failures, clock_supplement)
    gpu_source = None
    gpu_errors = []
    if (gpu_campaign / 'native/freeze.json').exists():
        gpu_source = legacy._campaign(gpu_campaign, 'gpu_extension')
        sources['gpu_extension'] = gpu_source['identity']
        freezes['gpu_extension'] = gpu_source['freeze']
        gpu_errors = _gpu_metadata(freezes['primary'], gpu_source['freeze'])
        lookup = {_coordinate(r): (key, r) for key, r in gpu_source['rows'].items()}
        # Keep all planned slots even if the GPU freeze is partial or malformed.
        for pending in _pending_gpu_rows(base):
            match = lookup.get(_coordinate(pending))
            if match is None:
                pending['metadata_errors'] = [*gpu_errors, 'gpu_planned_cell_missing']
                rows.append(pending)
                continue
            key, gpu_row = match
            row = copy.deepcopy(gpu_row)
            row.update(cell_id=pending['cell_id'], native_cell_id=legacy._cell_id(key), model_key='qwen38_gpu',
                       metadata_errors=list(gpu_errors), diagnostic_only=not gpu_row.get('evidence_verified'),
                       source_per_cell={'source_id': 'gpu_extension', 'freeze_ref': sources['gpu_extension']['freeze_ref'],
                                        'extractor_identity': sources['gpu_extension'].get('extractor_identity'),
                                        'replacement': False, 'appended_to_135': True},
                       attempt_history=[a for a in gpu_source['failed_attempts'] if a.get('cell_id') == legacy._cell_id(key)])
            rows.append(row)
        failures.extend(gpu_source['failed_attempts'])
    else:
        rows.extend(_pending_gpu_rows(base))
        gpu_errors = ['gpu_freeze_missing']
    selected, excluded = [], []
    for row in rows:
        row.setdefault('native_cell_id', row['cell_id'])
        source_id = row['source_per_cell']['source_id']
        row['placement_group'] = row['model_key']
        row['initial_screen_only'] = True
        row['formal_repeatability_accepted'] = False
        row['formal_prediction_accepted'] = False
        source = sources.get(source_id)
        if source is not None:
            freeze = freezes[source_id]
            row.setdefault('metadata_errors', []).extend(_identity_errors(source, freeze))
            row['source_per_cell'].update(protocol_ref=source.get('protocol_ref'),
                source_snapshot_ref=source.get('source_snapshot_ref'), hardware_fingerprint=source.get('hardware_fingerprint'))
            if row.get('status') == 'complete' and row.get('evidence_verified') is True:
                records = None
                if source_id == 'gpu_extension' and gpu_source is not None:
                    records = gpu_source['records'].get(tuple(row['native_cell_id'].split('__', 1)), [])
                actual, errors = _collect_actuals(row, source, freeze, records=records)
                row.update(actual)
                row['actuals_errors'] = errors
            else:
                row['actuals_errors'] = ['complete_verified_actuals_unavailable']
        reasons = strict_exclusion_reasons(row)
        row['strict_all_metrics_below_5pct'] = not reasons
        row['selection_exclusion_reasons'] = reasons
        if reasons:
            excluded.append({'cell_id': row['cell_id'], 'model_key': row['model_key'],
                             'prompt_tokens': row['prompt_tokens'], 'output_tokens': row['output_tokens'],
                             'parallel': row['parallel'], 'reasons': reasons,
                             'source_per_cell': row['source_per_cell'], 'attempt_history': row.get('attempt_history', [])})
        else:
            selected.append(copy.deepcopy(row))
    if len(rows) != 162 or len({r['cell_id'] for r in rows}) != 162:
        raise ValueError('combined_denominator_or_unique_cell_identity_not_162')
    coverage = []
    for name in GROUPS:
        group = [r for r in rows if r['model_key'] == name]
        coverage.append({'model_key': name, 'placement_group': name, 'planned_cells': 27,
                         'complete_cells': sum(r['status'] == 'complete' for r in group),
                         'verified_cells': sum(r.get('evidence_verified') is True and not r.get('metadata_errors') and not r.get('actuals_errors') for r in group),
                         'selected_cells': sum(r['strict_all_metrics_below_5pct'] for r in group),
                         'excluded_cells': sum(not r['strict_all_metrics_below_5pct'] for r in group),
                         'exclusion_reason_counts': dict(Counter(reason for r in group for reason in r['selection_exclusion_reasons']))})
    created = datetime.now(timezone.utc).isoformat()
    source_per_cell = {r['cell_id']: r['source_per_cell'] for r in rows}
    selection = {'schema': SCHEMA, 'created_utc': created,
                 'selection_rule': {'basis': 'native_engine_variability_only', 'metrics': list(METRICS),
                                    'fields': ['batch_worst_abs_pct', 'rank_worst_abs_pct'],
                                    'threshold_pct': THRESHOLD_PCT, 'comparison': '<',
                                    'full_valid_coverage_required': True, 'simulator_values_read': False},
                 'planned_cells': 162, 'selected_count': len(selected), 'excluded_count': len(excluded),
                 'selected_cell_ids': [r['cell_id'] for r in selected], 'selected_cells': selected,
                 'excluded_cells': excluded, 'coverage': coverage, 'sources': sources,
                 'source_per_cell': source_per_cell, 'failed_attempts': failures,
                 'supplement_lineage': base.get('supplement_lineage'),
                 'clock_exception_lineage': clock_lineage,
                 'selector_source_ref': legacy.reference(Path(__file__)),
                 'verification_scope': copy.deepcopy(VERIFICATION_SCOPE),
                 'initial_screen_only': True, 'formal_repeatability_accepted': False,
                 'formal_prediction_accepted': False, 'limitations': LIMITATIONS}
    selection = _clean(selection)
    selection['payload_sha256'] = _digest(selection)
    report = {'schema': 'native-long-grid-162-screen/v1', 'created_utc': created,
              'planned_cells': 162, 'original_planned_cells': 135, 'appended_gpu_planned_cells': 27,
              'complete_cells': sum(r['status'] == 'complete' for r in rows),
              'strict_all_metrics_below_5pct_cells': len(selected), 'coverage': coverage,
              'sources': sources, 'source_per_cell': source_per_cell, 'mixed_freezes': True,
              'gpu_append_validation_errors': gpu_errors, 'failed_attempts': failures,
              'verification_scope': copy.deepcopy(VERIFICATION_SCOPE),
              'supplement_lineage': base.get('supplement_lineage'),
              'clock_exception_lineage': clock_lineage,
              'original_verification_errors': base.get('verification_errors', []),
              'initial_screen_only': True, 'formal_repeatability_accepted': False,
              'formal_prediction_accepted': False, 'limitations': LIMITATIONS,
              'selection_payload_sha256': selection['payload_sha256'], 'cells': rows}
    return _clean(report), selection


def verify_selection_payload(selection):
    payload = dict(selection)
    claimed = payload.pop('payload_sha256', None)
    return _sha(claimed) and _digest(payload) == claimed


def _svg(report, metric):
    width, panel_w, panel_h = 1260, 420, 228
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="580" viewBox="0 0 {width} 580">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#172033}.small{font-size:11px}</style>',
             f'<text x="20" y="26" font-size="21">{metric.upper()} native variability: max(batch, request-rank) deviation %</text>',
             '<text x="20" y="48" class="small">Engine measurements only. Strict &lt;5% screen; 3 repeats in one process; no formal guarantee. Gray = missing/unverified.</text>']
    for gi, name in enumerate(GROUPS):
        x0, y0 = (gi % 3) * panel_w + 14, (gi // 3) * panel_h + 78
        parts.append(f'<text x="{x0}" y="{y0}" font-size="16">{html.escape(name)}</text>')
        group = {_coordinate(r): r for r in report['cells'] if r['model_key'] == name}
        for ci, concurrency in enumerate((1, 2, 4)):
            x = x0 + ci * 134
            parts.append(f'<text x="{x}" y="{y0 + 21}" class="small">Concurrency {concurrency}</text>')
            for oi, output in enumerate((32, 128, 256)):
                parts.append(f'<text x="{x + 32 + oi * 32}" y="{y0 + 39}" class="small">{output}</text>')
            for pi, prompt in enumerate((128, 512, 1536)):
                parts.append(f'<text x="{x}" y="{y0 + 62 + pi * 34}" class="small">{prompt}</text>')
                for oi, output in enumerate((32, 128, 256)):
                    row = group[(prompt, output, concurrency)]
                    m = row.get('metrics', {}).get(metric, {})
                    vals = [m.get(k) for k in ('batch_worst_abs_pct', 'rank_worst_abs_pct')]
                    valid = (row.get('evidence_verified') is True and row['status'] == 'complete'
                             and not row.get('metadata_errors') and not row.get('actuals_errors')
                             and all(_finite(v) for v in vals))
                    value = max(vals) if valid else None
                    fill = '#e2e8f0' if value is None else '#bbf7d0' if value < 5 else '#fde68a' if value < 10 else '#fca5a5'
                    label = 'NA' if value is None else f'{value:.1f}'
                    xx, yy = x + 31 + oi * 32, y0 + 45 + pi * 34
                    tip = html.escape(f"{row['cell_id']}; batch={vals[0]}; rank={vals[1]}")
                    parts.append(f'<g><title>{tip}</title><rect x="{xx}" y="{yy}" width="30" height="31" rx="3" fill="{fill}"/>'
                                 f'<text x="{xx + 15}" y="{yy + 20}" text-anchor="middle" font-size="10">{label}</text></g>')
        cov = report['coverage'][gi]
        parts.append(f'<text x="{x0}" y="{y0 + 178}" class="small">All 3 metrics strict &lt;5%: {cov["selected_cells"]}/27; completed: {cov["complete_cells"]}/27</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    legacy._atomic_text(output / 'report.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for metric in METRICS:
        legacy._atomic_text(output / (metric + '_native_variability.svg'), _svg(report, metric))
    table = '<table><tr><th>组</th><th>完成/计划</th><th>严格入选/计划</th></tr>' + ''.join(
        f'<tr><td>{c["model_key"]}</td><td>{c["complete_cells"]}/27</td><td>{c["selected_cells"]}/27</td></tr>' for c in report['coverage']) + '</table>'
    text = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>162格原生波动初筛</title>'
            '<style>body{font-family:Arial,sans-serif;margin:30px;color:#172033}img{width:100%;max-width:1260px}td,th{padding:7px 20px;border-bottom:1px solid #ddd}</style>'
            '<h1>162格原生波动初筛</h1><p>原135格 + 27B GPU版27格。三项Engine指标的批次与请求序位最大偏差均严格小于5%，且原始证据完整有效才入选。</p>'
            '<p>三次单进程重复仅为开发初筛，不构成正式稳定性保证；图中是原生波动，不是仿真预测误差。灰格为缺失或未核验。</p>' + table +
            ''.join(f'<p><img src="{m}_native_variability.svg" alt="{m.upper()}原生波动热图"></p>' for m in METRICS) + '</html>')
    legacy._atomic_text(output / 'report.html', text)
    lines = ['# 162格原生初筛', '', f'完成 {report["complete_cells"]}/162；三指标严格 <5% 且证据完整有效 {report["strict_all_metrics_below_5pct_cells"]}/162。',
             '', '三次单进程重复仅为开发初筛；原生波动不等于仿真预测误差。',
             '复核范围：重验冻结引用、raw绑定与Engine时间点；CPU/GPU环境按实际is_set/value逐格比对；GPU要求显式66层且fit关闭。',
             '模型与exe/DLL比对冻结摘要，未重新读取大二进制；运行baseline和前后硬件状态保留采集门禁结论及原始引用，未重新执行所有状态验收谓词。', '',
             '| 组 | 完成/计划 | 严格入选/计划 |', '|---|---:|---:|']
    lines += [f'| {c["model_key"]} | {c["complete_cells"]}/27 | {c["selected_cells"]}/27 |' for c in report['coverage']]
    lines += ['', f'保留失败尝试 {len(report["failed_attempts"])} 条。每格原始来源与全部排除原因见 report.json。', '']
    legacy._atomic_text(output / 'report.md', '\n'.join(lines))


def write_selection(selection, path):
    """Exclusive creation prevents later simulation work from changing membership."""
    if not verify_selection_payload(selection):
        raise ValueError('selection_payload_sha256_invalid')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    # O_EXCL is intentional: never overwrite a previously frozen selection.
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gpu-campaign', type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--lineage', type=Path)
    parser.add_argument('--clock-supplement', type=Path, help='Explicit four-cell Smol rerun with user-approved 60 MHz clock tolerance')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true', help='Render progress without freezing selection membership')
    args = parser.parse_args(argv)
    if not args.report_only and args.selection.exists():
        parser.error('selection already exists; use another explicit path or --report-only')
    report, selection = build_dataset(args.campaign, args.gpu_campaign, args.supplement, args.lineage, args.clock_supplement)
    if not args.report_only:
        report['selection_ref'] = write_selection(selection, args.selection)
    _write_report(report, args.output)
    print(json.dumps({'planned_cells': 162, 'selected_count': selection['selected_count'],
                      'excluded_count': selection['excluded_count'], 'coverage': selection['coverage'],
                      'selection_ref': report.get('selection_ref'), 'output': str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())