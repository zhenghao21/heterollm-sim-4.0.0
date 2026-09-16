"""R19 R2 acceptance-gate regressions; no compile, DLL load, GPU, or timing probe."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[6]
PROBE = ROOT / 'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_019/cpu_sampling_probe'
SPEC = importlib.util.spec_from_file_location('cpu_sampling_probe_entry_r2', PROBE / 'entry.py')
entry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(entry)

CPU = 0
IDENTITY = {
    'cpu_brand': 'Synthetic CPU identity only',
    'cpuid_signature': 1234,
    'active_processor_group_count': 1,
    'active_processor_count_group0': 16,
    'group': 0,
    'logical_cpu': CPU,
    'thread_affinity_mask': 1,
}
FREQUENCY_FIELDS = ['os_max_mhz', 'os_reported_current_mhz', 'os_limit_mhz']


def observation(*, maximum=5000, current=4500, limit=100):
    return {
        'group': 0,
        'logical_processor': CPU,
        'thread_affinity_mask': 1,
        'os_max_mhz': maximum,
        'os_reported_current_mhz': current,
        'os_limit_mhz': limit,
    }


def stage(name, *, maximum_after=5000, current_after=4500, limit_after=100, steady_ticks=None):
    before, after = observation(), observation(maximum=maximum_after, current=current_after, limit=limit_after)
    stable = before['os_max_mhz'] == after['os_max_mhz'] and before['os_reported_current_mhz'] == after['os_reported_current_mhz'] and before['os_limit_mhz'] == after['os_limit_mhz']
    return {
        'stage': name,
        'numeric_quality': 'exact_pass',
        'warmup_calls': 16,
        'steady_repeats': 64,
        'first_use_ticks': 1,
        'steady_raw_ticks': [100] * 64 if steady_ticks is None else steady_ticks,
        'reported_frequency_fields_checked': FREQUENCY_FIELDS,
        'reported_frequency_stable': stable,
        'timing_usable': stable,
        'diagnostic_only': not stable,
        'frequency_changed_diagnostic_only': not stable,
        'cpu_before': before,
        'cpu_after': after,
    }


def document(index=0, *, maximum_after=5000, current_after=4500, limit_after=100, steady_ticks=None, observer_ticks=None):
    cases = []
    for vocabulary in (32768, 131072, 262144):
        for pattern in ('monotone_ascending', 'deterministic_random_permutation'):
            cases.append({
                'vocabulary_size': vocabulary,
                'pattern': pattern,
                'split': 'holdout' if vocabulary == 131072 else 'train',
                'candidate_record_bytes': 12,
                'top_k': 1,
                'first_use_not_pooled_with_steady': True,
                'stages': [stage('candidate_loop', steady_ticks=steady_ticks), stage('original_dll_topk_apply', maximum_after=maximum_after, current_after=current_after, limit_after=limit_after, steady_ticks=steady_ticks)],
            })
    return {
        'status': 'complete',
        'process_index': index,
        'process_id': 100 + index,
        'logical_cpu': CPU,
        'GPU_context_created': False,
        'model_loaded': False,
        'full_sampler_chain_measured': False,
        'actual_cpu_identity': copy.deepcopy(IDENTITY),
        'qpc_frequency': 10_000_000,
        'observer_empty_bracket_ticks': [1] * 64 if observer_ticks is None else observer_ticks,
        'cases': cases,
    }


def test_matching_identity_and_all_three_frequency_fields_are_timing_usable():
    result = entry.validate_result(document(), 0, CPU, IDENTITY)
    assert result['stable_stages'] == 12 and result['total_stages'] == 12
    assert all(stage['timing_usable'] for stage in result['stage_quality'])


@pytest.mark.parametrize('changed', [
    {'maximum_after': 4900},
    {'current_after': 4400},
    {'limit_after': 99},
])
def test_each_reported_frequency_field_drift_is_preserved_but_never_accepted_as_timing_evidence(changed):
    results = []
    for index in range(3):
        doc = document(index, **(changed if index == 1 else {}))
        results.append({'document': doc, 'frequency': entry.validate_result(doc, index, CPU, IDENTITY)})
    quality = entry.summarize_quality(results, {'path': 'identity-freeze', 'sha256': '0' * 64, 'bytes': 1})
    assert quality['frequency_changed_stages'] == 6
    assert quality['timing_usable'] is False
    assert quality['diagnostic_only'] is True
    assert quality['accepted_for_timing_evidence'] is False



def _quality_for(documents):
    results = []
    for index, doc in enumerate(documents):
        results.append({'document': doc, 'frequency': entry.validate_result(doc, index, CPU, IDENTITY)})
    return entry.summarize_quality(results, {'path': 'identity-freeze', 'sha256': '0' * 64, 'bytes': 1})


def test_steady_p90_p10_jitter_gate_is_diagnostic_only():
    jitter = [100] * 57 + [200] * 7
    quality = _quality_for([document(index, steady_ticks=jitter if index == 0 else None) for index in range(3)])
    assert quality['steady_dispersion_pass_stages'] == 24
    assert quality['timing_usable'] is False
    assert 'steady_dispersion' in quality['timing_quality_failure_reasons']


def test_cross_process_median_outlier_gate_is_diagnostic_only():
    quality = _quality_for([document(0), document(1), document(2, steady_ticks=[106] * 64)])
    assert quality['cross_process_case_stage_groups_total'] == 12
    assert quality['cross_process_case_stage_groups_pass'] == 0
    assert quality['timing_usable'] is False
    assert 'cross_process_median_deviation' in quality['timing_quality_failure_reasons']


def test_observer_overhead_and_zero_steady_denominator_are_diagnostic_only_without_division():
    overhead = _quality_for([document(index, observer_ticks=[2] * 64) for index in range(3)])
    assert overhead['observer_overhead_pass_stages'] == 0
    assert overhead['timing_usable'] is False
    assert 'observer_overhead' in overhead['timing_quality_failure_reasons']
    zero = _quality_for([document(index, steady_ticks=[0] * 64) for index in range(3)])
    assert zero['timing_usable'] is False
    assert zero['steady_dispersion_pass_stages'] == 0
    assert 'steady_dispersion' in zero['timing_quality_failure_reasons']


def test_nonpositive_frequency_is_rejected_not_misclassified_as_stable():
    doc = document()
    doc['cases'][0]['stages'][0]['cpu_before']['os_limit_mhz'] = 0
    with pytest.raises(ValueError, match='frequency observation unavailable'):
        entry.validate_result(doc, 0, CPU, IDENTITY)


def test_cpu_identity_mismatch_is_rejected_before_series_quality():
    doc = document()
    doc['actual_cpu_identity']['thread_affinity_mask'] = 2
    with pytest.raises(ValueError, match='affinity differs'):
        entry.validate_result(doc, 0, CPU, IDENTITY)


def test_frequency_status_cannot_claim_usable_when_max_mhz_changed():
    doc = document(maximum_after=4900)
    changed = doc['cases'][0]['stages'][1]
    changed['reported_frequency_stable'] = True
    changed['timing_usable'] = True
    changed['diagnostic_only'] = False
    changed['frequency_changed_diagnostic_only'] = False
    with pytest.raises(ValueError, match='frequency status'):
        entry.validate_result(doc, 0, CPU, IDENTITY)


def test_source_and_protocol_require_identity_capture_and_three_field_frequency_gate():
    source = (PROBE / 'cpu_sampling_probe.cpp').read_text(encoding='utf-8')
    protocol = json.loads((PROBE / 'protocol.json').read_text(encoding='utf-8'))
    assert 'before.max_mhz == after.max_mhz' in source
    assert 'before.reported_mhz == after.reported_mhz' in source
    assert 'before.limit_mhz == after.limit_mhz' in source
    assert '--identity-only' in source
    assert 'actual_cpu_identity' in source
    assert protocol['schema'] == 'cpu-sampling-work-probe-protocol/v3'
    assert protocol['quality_gate']['timing_acceptance'].count('max/current/limit') == 1
    assert protocol['identity_capture']['before_measurement'] is True
    assert protocol['timing_quality_gate']['stage_steady_dispersion'].endswith('diagnostic-only.')
    assert 'P90/P10' in protocol['timing_quality_gate']['stage_steady_dispersion']
    assert '0.05' in protocol['timing_quality_gate']['cross_process_medians']
    assert '0.01' in protocol['timing_quality_gate']['observer_overhead']
    assert '36 stage records' in protocol['timing_quality_gate']['denominators']







