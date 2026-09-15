"""Pure synthetic tests: no native process, model hashing, GPU, or predictions."""
import copy
import json
from pathlib import Path

import pytest

from tools import native_162_dataset as dataset
from tools import native_long_grid_report as legacy


def cell(*, parallel=1, values=(100., 101., 99.)):
    plan = {'job_id': 'qwen25_p128_o32_c1', 'model_key': 'qwen25',
            'condition_id': 'fixed_runtime', 'model': 'fixture.gguf',
            'expected_prompt_tokens': 128, 'output': 32, 'parallel': parallel,
            'measure_batches': 3}
    batches = [{'status': 'complete', 'batch_medians_ms': {m: value for m in dataset.METRICS},
                'requests': [{'status': 'measured', 'engine_start_rank': rank,
                              'metrics_ms': {m: value for m in dataset.METRICS}}
                             for rank in range(parallel)]}
               for value in values]
    result = legacy.summarize_cell([plan], [{'status': 'complete', 'runs': batches}])
    result.update(evidence_verified=True, diagnostic_only=False, cell_id=plan['job_id'] + '__fixed_runtime')
    return result


def test_strict_less_than_five_instead_of_legacy_inclusive_threshold():
    row = cell(values=(95., 100., 105.))
    assert row['observed_all_metrics_within_5pct'] is True
    reasons = dataset.strict_exclusion_reasons(row)
    assert 'ttft_batch_worst_abs_pct_not_strictly_below_5' in reasons
    assert 'e2e_rank_worst_abs_pct_not_strictly_below_5' in reasons
    assert dataset.strict_exclusion_reasons(cell(values=(95.0001, 100., 104.9999))) == []


@pytest.mark.parametrize('invalid', [None, float('nan'), float('inf'), -1, True])
@pytest.mark.parametrize('field', ['batch_worst_abs_pct', 'rank_worst_abs_pct'])
def test_nonfinite_missing_negative_boolean_deviation_is_rejected(invalid, field):
    row = cell()
    row['metrics']['tpot'][field] = invalid
    assert 'tpot_' + field + '_missing_or_nonfinite' in dataset.strict_exclusion_reasons(row)


def test_request_rank_variability_cannot_hide_behind_stable_batch_medians():
    row = cell(parallel=2)
    row['metrics']['ttft']['batch_worst_abs_pct'] = 0
    row['metrics']['ttft']['rank_worst_abs_pct'] = 7
    assert 'ttft_rank_worst_abs_pct_not_strictly_below_5' in dataset.strict_exclusion_reasons(row)


@pytest.mark.parametrize('change,reason', [
    ({'evidence_verified': False}, 'evidence_or_freeze_not_verified'),
    ({'status': 'failed'}, 'cell_incomplete_or_failed'),
    ({'planned_batches': 0, 'captured_batches': 0}, 'measurement_batches_coverage_incomplete_or_zero'),
    ({'complete_blocks': 0}, 'process_blocks_coverage_incomplete_or_zero'),
    ({'captured_batches': 2}, 'measurement_batches_coverage_incomplete_or_zero'),
    ({'diagnostic_only': True}, 'diagnostic_only'),
])
def test_incomplete_unverified_or_zero_counts_excluded(change, reason):
    row = cell()
    row.update(change)
    assert reason in dataset.strict_exclusion_reasons(row)


def test_every_rank_and_every_metric_needs_exact_sample_coverage():
    row = cell(parallel=2)
    row['raw_summary']['metrics']['e2e']['request_ranks'][1]['observed_samples'] = 0
    assert 'e2e_sample_coverage_incomplete_or_zero' in dataset.strict_exclusion_reasons(row)
    row['raw_summary']['metrics']['ttft']['request_ranks'][1]['rank'] = 0
    assert 'ttft_rank_coverage_invalid' in dataset.strict_exclusion_reasons(row)


def source_fixture(tmp_path, models, gpu=False):
    root = tmp_path / ('gpu' if gpu else 'primary')
    (root / 'native').mkdir(parents=True)
    refs = []
    for filename, content in [('fixture.gguf', b'same model'), ('llama-server.exe', b'same exe'),
                              ('ggml-cuda.dll', b'same dll'), ('native_llama_compare.py', b'extractor')]:
        path = root / filename
        path.write_bytes(content)
        refs.append(legacy.reference(path))
    plans, groups, rows = [], {}, {}
    for model in models:
        for prompt, output, parallel in dataset.GRID:
            job = f'{model}_p{prompt}_o{output}_c{parallel}'
            plan = {'job_id': job, 'model_key': model, 'condition_id': 'fixed_runtime',
                    'key': job + '__fixed_runtime__b00', 'model': str(root / 'fixture.gguf'),
                    'expected_prompt_tokens': prompt, 'prompt_token_ids': list(range(prompt)),
                    'output': output, 'parallel': parallel, 'measure_batches': 3,
                    'warmup_batches': 2, 'process_blocks': 1, 'block': 0,
                    'gpu_layers': 66 if gpu else 0, 'poll': 50, 'log_verbosity': 0,
                    'expected_gpu_sm_clock_mhz': 2400, 'gpu_sm_clock_tolerance_mhz': 30}
            if gpu:
                plan['fit_params'] = False
            plans.append(plan)
            key = (job, 'fixed_runtime')
            groups[key] = [plan]
            row = cell(parallel=parallel)
            row.update(id=job, cell_id=job + '__fixed_runtime', model_key=model,
                       prompt_tokens=prompt, output_tokens=output,
                       source_per_cell={'source_id': 'gpu_extension' if gpu else 'primary'}, attempt_history=[])
            rows[key] = row
    freeze = {'schema': 'native-repeatability-freeze/v1', 'plans': plans,
              'environments': {p['key']: {key: {'is_set': False, 'value': None}
                              for key in legacy.experiment.ENV_KEYS} for p in plans},
              'artifact_refs': refs[:3], 'source_refs': [refs[3]], 'hardware_fingerprint': 'same-device',
              'extractor_identity': {'sha256': 'a' * 64, 'file_sha256': refs[3]['sha256']}}
    path = root / 'native/freeze.json'
    path.write_text(json.dumps(freeze), encoding='utf-8')
    freeze_ref = legacy.reference(path)
    identity = {'source_id': 'gpu_extension' if gpu else 'primary', 'campaign': str(root),
                'freeze_ref': freeze_ref, 'end_verified': True, 'verification_errors': [],
                'protocol_ref': refs[3], 'source_snapshot_ref': refs[3],
                'extractor_identity': freeze['extractor_identity'], 'hardware_fingerprint': 'same-device'}
    for row in rows.values():
        row['source_per_cell']['freeze_ref'] = freeze_ref
    return {'identity': identity, 'freeze': freeze, 'groups': groups, 'rows': rows,
            'records': {}, 'failed_attempts': []}


def test_gpu_append_checks_same_gguf_tokens_native_identity_and_placement(tmp_path):
    cpu = source_fixture(tmp_path, ['qwen38'])
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    assert dataset._gpu_metadata(cpu['freeze'], gpu['freeze']) == []
    changed = copy.deepcopy(gpu['freeze'])
    changed['plans'][0]['prompt_token_ids'][0] = 900
    assert 'gpu_prompt_token_ids_mismatch' in dataset._gpu_metadata(cpu['freeze'], changed)
    changed = copy.deepcopy(gpu['freeze'])
    changed['artifact_refs'][0]['sha256'] = '0' * 64
    assert 'gpu_gguf_sha256_mismatch' in dataset._gpu_metadata(cpu['freeze'], changed)
    changed = copy.deepcopy(gpu['freeze'])
    changed['artifact_refs'][2]['sha256'] = '0' * 64
    assert 'gpu_native_exe_or_dll_sha256_mismatch' in dataset._gpu_metadata(cpu['freeze'], changed)
    changed = copy.deepcopy(gpu['freeze'])
    changed['plans'][0]['gpu_layers'] = 0
    assert 'gpu_layers_must_be_explicit_66' in dataset._gpu_metadata(cpu['freeze'], changed)


def setup_build(tmp_path, monkeypatch, *, include_gpu=True):
    cpu = source_fixture(tmp_path, dataset.BASE_GROUPS)
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    base = {'planned_cells': 135, 'cells': list(cpu['rows'].values()),
            'sources': {'primary': cpu['identity']}, 'verification_errors': [],
            'failed_attempts': [{'source_id': 'primary', 'cell_id': 'old_failed', 'status': 'failed'}]}
    monkeypatch.setattr(legacy, 'build_report', lambda *args, **kwargs: copy.deepcopy(base))
    monkeypatch.setattr(legacy, '_campaign', lambda *args, **kwargs: copy.deepcopy(gpu))
    def actuals(row, source, freeze, **kwargs):
        plan = next((p for p in freeze['plans'] if p['job_id'] + '__' + p['condition_id'] == row['native_cell_id']), None)
        if plan is None:
            return {}, ['frozen_cell_plan_missing']
        return {'config': plan, 'native_actuals': [{'metrics_ms': {m: 1. for m in dataset.METRICS},
                'raw_ref': source['freeze_ref'], 'engine_timing_pointer': '/runs/0/requests/0/response/timings'}]}, []
    monkeypatch.setattr(dataset, '_collect_actuals', actuals)
    gpu_path = Path(gpu['identity']['campaign']) if include_gpu else tmp_path / 'absent_gpu'
    return Path(cpu['identity']['campaign']), gpu_path, base, gpu


def test_135_plus_27_preserves_failed_history_and_six_group_coverage(tmp_path, monkeypatch):
    campaign, gpu_campaign, base, gpu = setup_build(tmp_path, monkeypatch)
    report, selection = dataset.build_dataset(campaign, gpu_campaign)
    assert report['planned_cells'] == 162
    assert selection['selected_count'] == 162
    assert len(selection['selected_cell_ids']) == len(set(selection['selected_cell_ids'])) == 162
    assert selection['failed_attempts'] == base['failed_attempts']
    assert len(report['coverage']) == 6
    assert all(c['planned_cells'] == c['selected_cells'] == 27 for c in report['coverage'])
    assert selection['sources']['primary']['freeze_ref'] != selection['sources']['gpu_extension']['freeze_ref']
    assert selection['selection_rule']['simulator_values_read'] is False
    assert dataset.verify_selection_payload(selection)
    assert not selection['formal_repeatability_accepted']
    assert not selection['formal_prediction_accepted']


def test_missing_gpu_still_has_full_162_denominator_and_explicit_exclusions(tmp_path, monkeypatch):
    campaign, gpu_campaign, _, _ = setup_build(tmp_path, monkeypatch, include_gpu=False)
    report, selection = dataset.build_dataset(campaign, gpu_campaign)
    assert len(report['cells']) == 162
    assert selection['selected_count'] == 135
    assert selection['excluded_count'] == 27
    assert all('gpu_freeze_missing' in c['reasons'] for c in selection['excluded_cells'])
    assert selection['coverage'][-1]['selected_cells'] == 0


def test_gpu_configuration_drift_excludes_only_appended_gpu_cells(tmp_path, monkeypatch):
    campaign, gpu_campaign, _, gpu = setup_build(tmp_path, monkeypatch)
    gpu['freeze']['plans'][0]['poll'] = 0
    report, selection = dataset.build_dataset(campaign, gpu_campaign)
    assert selection['selected_count'] == 135
    assert 'gpu_nonplacement_configuration_mismatch' in report['gpu_append_validation_errors']
    assert {r['model_key'] for r in selection['excluded_cells']} == {'qwen38_gpu'}


def test_selection_hash_tamper_and_exclusive_write(tmp_path):
    selection = {'schema': dataset.SCHEMA, 'created_utc': '2026-09-15T00:00:00+00:00', 'selected_cells': []}
    selection['payload_sha256'] = dataset._digest(selection)
    path = tmp_path / 'selection.json'
    ref = dataset.write_selection(selection, path)
    assert ref == legacy.reference(path)
    with pytest.raises(FileExistsError):
        dataset.write_selection(selection, path)
    selection['selected_cells'].append({'cell_id': 'tamper'})
    assert not dataset.verify_selection_payload(selection)
    with pytest.raises(ValueError, match='sha256_invalid'):
        dataset.write_selection(selection, tmp_path / 'other.json')


def test_svg_renders_162_slots_without_numpy_or_prediction_values(tmp_path, monkeypatch):
    campaign, gpu_campaign, _, _ = setup_build(tmp_path, monkeypatch, include_gpu=False)
    report, _ = dataset.build_dataset(campaign, gpu_campaign)
    image = dataset._svg(report, 'tpot')
    assert image.count('<g><title>') == 162
    assert 'qwen38_gpu' in image
    assert 'Engine measurements only' in image
    assert '>NA<' in image

@pytest.mark.parametrize('fit', [None, True, 0, 'false'])
def test_gpu_requires_boolean_explicit_fit_off(tmp_path, fit):
    cpu = source_fixture(tmp_path, ['qwen38'])
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    gpu['freeze']['plans'][0]['fit_params'] = fit
    assert 'gpu_fit_params_must_be_explicit_false' in dataset._gpu_metadata(cpu['freeze'], gpu['freeze'])


def test_explicit_full_layer_placement_with_fit_off_is_allowed(tmp_path):
    cpu = source_fixture(tmp_path, ['qwen38'])
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    for plan in gpu['freeze']['plans']:
        plan['gpu_layers'] = 66
    assert dataset._gpu_metadata(cpu['freeze'], gpu['freeze']) == []


def test_static_hardware_keeps_own_freeze_config_and_before_after_raw_refs(tmp_path):
    source = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    source['freeze']['hardware'] = {'gpu': {'name': 'fixture GPU', 'clocks': {'sm_mhz': 1800}}}
    plan = source['freeze']['plans'][0]
    plan.update(expected_gpu_sm_clock_mhz=2400, gpu_sm_clock_tolerance_mhz=30)
    raw_ref = {'path': str(tmp_path / 'bound_raw.json'), 'sha256': 'b' * 64}
    result = dataset._hardware_index(source['identity'], source['freeze'], plan,
                                    [{'key': plan['key'], 'raw_ref': raw_ref}])
    assert result['source_id'] == 'gpu_extension'
    assert result['frozen_hardware_ref']['file_ref'] == source['identity']['freeze_ref']
    assert result['native_config_ref']['json_pointer'] == '/plans/0'
    assert result['frozen_hardware']['gpu']['clocks']['sm_mhz'] == 1800
    assert result['configured_clock']['expected_gpu_sm_clock_mhz'] == 2400
    assert result['state_refs'][0]['raw_ref'] == raw_ref
    assert result['state_refs'][0]['gpu_clock_before'] == '/state_measurement_before/gpu_state'
    assert result['state_refs'][0]['gpu_clock_after'] == '/state_after/gpu_state'


@pytest.mark.parametrize('layers', [-1, 0, 1, 65, 67, True, 66.0, None])
def test_gpu_full_placement_requires_integer_66(tmp_path, layers):
    cpu = source_fixture(tmp_path, ['qwen38'])
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    gpu['freeze']['plans'][0]['gpu_layers'] = layers
    assert 'gpu_layers_must_be_explicit_66' in dataset._gpu_metadata(cpu['freeze'], gpu['freeze'])


def test_effective_frozen_environment_is_checked_beyond_config_overrides(tmp_path):
    cpu = source_fixture(tmp_path, ['qwen38'])
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    key = gpu['freeze']['plans'][0]['key']
    gpu['freeze']['environments'][key]['CUDA_VISIBLE_DEVICES'] = {'is_set': True, 'value': '0'}
    assert 'gpu_effective_environment_mismatch' in dataset._gpu_metadata(cpu['freeze'], gpu['freeze'])


def test_unset_and_empty_environment_value_are_distinct(tmp_path):
    cpu = source_fixture(tmp_path, ['qwen38'])
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    key = gpu['freeze']['plans'][0]['key']
    gpu['freeze']['environments'][key]['CUDA_VISIBLE_DEVICES'] = {'is_set': True, 'value': ''}
    assert 'gpu_effective_environment_mismatch' in dataset._gpu_metadata(cpu['freeze'], gpu['freeze'])


@pytest.mark.parametrize('mutation', ['missing_plan', 'missing_variable', 'invalid_type'])
def test_missing_or_malformed_effective_environment_rejected(tmp_path, mutation):
    cpu = source_fixture(tmp_path, ['qwen38'])
    gpu = source_fixture(tmp_path, ['qwen38_gpu'], gpu=True)
    key = gpu['freeze']['plans'][0]['key']
    if mutation == 'missing_plan':
        del gpu['freeze']['environments'][key]
    elif mutation == 'missing_variable':
        del gpu['freeze']['environments'][key]['CUDA_VISIBLE_DEVICES']
    else:
        gpu['freeze']['environments'][key]['CUDA_VISIBLE_DEVICES']['is_set'] = 0
    assert 'frozen_effective_environment_missing_or_invalid' in dataset._gpu_metadata(cpu['freeze'], gpu['freeze'])


def test_split_admission_is_preserved_but_not_an_extra_stability_gate():
    row = cell()
    row['split_admission_batches'] = 3
    row['raw_summary']['strict_within_5pct'] = False
    assert dataset.strict_exclusion_reasons(row) == []



def clock_fixture(tmp_path, monkeypatch):
    campaign, gpu_campaign, base, gpu = setup_build(tmp_path, monkeypatch)
    clock = source_fixture(tmp_path / 'clock', ['smollm2'])
    keep = lambda plan: dataset._plan_coordinate(plan) in dataset.CLOCK_EXCEPTION_COORDINATES
    clock['freeze']['plans'] = [plan for plan in clock['freeze']['plans'] if keep(plan)]
    clock['rows'] = {key: row for key, row in clock['rows'].items()
                     if dataset._coordinate(row) in dataset.CLOCK_EXCEPTION_COORDINATES}
    for plan in clock['freeze']['plans']:
        plan['gpu_sm_clock_tolerance_mhz'] = 60
    clock['identity']['source_id'] = 'clock_exception_supplement'
    for row in base['cells']:
        if row['model_key'] == 'smollm2' and dataset._coordinate(row) in dataset.CLOCK_EXCEPTION_COORDINATES:
            row['status'] = 'incomplete_or_failed'
            event = {'source_id': 'primary', 'cell_id': row['cell_id'], 'status': 'failed',
                     'error': 'expected GPU SM clock not observed: 2445 MHz'}
            row['attempt_history'] = [event]
            base['failed_attempts'].append(event)
    monkeypatch.setattr(legacy, '_campaign', lambda path, label: copy.deepcopy(
        clock if label == 'clock_exception_supplement' else gpu))
    return campaign, gpu_campaign, Path(clock['identity']['campaign']), base, clock


def test_clock_supplement_replaces_only_four_failed_cells_and_retains_history(tmp_path, monkeypatch):
    campaign, gpu_campaign, clock_path, base, _ = clock_fixture(tmp_path, monkeypatch)
    report, selection = dataset.build_dataset(campaign, gpu_campaign, clock_supplement=clock_path)
    assert selection['planned_cells'] == selection['selected_count'] == 162
    assert report['clock_exception_lineage']['status'] == 'verified'
    assert len(report['clock_exception_lineage']['cells']) == 4
    assert selection['failed_attempts'] == base['failed_attempts']
    replaced = [row for row in selection['selected_cells']
                if row['source_per_cell']['source_id'] == 'clock_exception_supplement']
    assert len(replaced) == 4
    assert {dataset._coordinate(row) for row in replaced} == dataset.CLOCK_EXCEPTION_COORDINATES
    assert all(row['attempt_history'][0]['status'] == 'failed' for row in replaced)
    assert all(row['source_per_cell']['original_status'] == 'incomplete_or_failed' for row in replaced)
    assert all(row['clock_exception']['variability_threshold_unchanged'] for row in replaced)
    assert all(row['status'] == 'incomplete_or_failed' for row in base['cells']
               if row['model_key'] == 'smollm2' and dataset._coordinate(row) in dataset.CLOCK_EXCEPTION_COORDINATES)


@pytest.mark.parametrize('field,value,reason', [
    ('gpu_sm_clock_tolerance_mhz', 61, 'clock_exception_requires_2400mhz_30_to_60mhz_tolerance'),
    ('threads', 8, 'clock_exception_other_configuration_mismatch'),
    ('expected_gpu_sm_clock_mhz', 2430, 'clock_exception_requires_2400mhz_30_to_60mhz_tolerance'),
])
def test_clock_exception_cannot_change_other_runtime_or_frequency_contract(tmp_path, monkeypatch, field, value, reason):
    campaign, gpu_campaign, clock_path, _, clock = clock_fixture(tmp_path, monkeypatch)
    clock['freeze']['plans'][0][field] = value
    report, selection = dataset.build_dataset(campaign, gpu_campaign, clock_supplement=clock_path)
    assert selection['selected_count'] == 158
    assert reason in report['clock_exception_lineage']['verification_errors']
    assert all(reason in row['reasons'] for row in selection['excluded_cells'])


def test_clock_exception_checks_effective_environment_and_model_sha(tmp_path, monkeypatch):
    campaign, gpu_campaign, clock_path, _, clock = clock_fixture(tmp_path, monkeypatch)
    plan = clock['freeze']['plans'][0]
    clock['freeze']['environments'][plan['key']]['CUDA_VISIBLE_DEVICES'] = {'is_set': True, 'value': '1'}
    clock['freeze']['artifact_refs'][0]['sha256'] = 'e' * 64
    report, selection = dataset.build_dataset(campaign, gpu_campaign, clock_supplement=clock_path)
    assert selection['selected_count'] == 158
    errors = report['clock_exception_lineage']['verification_errors']
    assert 'clock_exception_effective_environment_mismatch' in errors
    assert 'clock_exception_gguf_sha256_mismatch' in errors


def test_clock_supplement_needs_exact_scope_and_cannot_replace_complete_original(tmp_path, monkeypatch):
    campaign, gpu_campaign, clock_path, base, clock = clock_fixture(tmp_path, monkeypatch)
    base_row = next(row for row in base['cells'] if row['model_key'] == 'smollm2'
                    and dataset._coordinate(row) in dataset.CLOCK_EXCEPTION_COORDINATES)
    base_row['status'] = 'complete'
    clock['freeze']['plans'].pop()
    report, _ = dataset.build_dataset(campaign, gpu_campaign, clock_supplement=clock_path)
    errors = report['clock_exception_lineage']['verification_errors']
    assert 'clock_exception_requires_exact_four_smol_cells' in errors
    assert 'clock_exception_cannot_replace_complete_original_cell' in errors


def test_clock_tolerance_exception_does_not_override_strict_five_percent_rule(tmp_path, monkeypatch):
    campaign, gpu_campaign, clock_path, _, clock = clock_fixture(tmp_path, monkeypatch)
    key = next(iter(clock['rows']))
    clock['rows'][key]['metrics']['e2e']['rank_worst_abs_pct'] = 5.0
    report, selection = dataset.build_dataset(campaign, gpu_campaign, clock_supplement=clock_path)
    assert selection['selected_count'] == 161
    assert report['clock_exception_lineage']['status'] == 'verified'
    assert 'e2e_rank_worst_abs_pct_not_strictly_below_5' in selection['excluded_cells'][0]['reasons']
