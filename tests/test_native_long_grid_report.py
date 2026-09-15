from tools.native_long_grid_report import summarize_cell


def make_case(values, *, planned=3, complete=True):
    cfg={'job_id':'qwen25_p128_o32_c1','model_key':'qwen25','condition_id':'fixed',
         'model':'fixture.gguf','expected_prompt_tokens':128,'output':32,'parallel':1,'measure_batches':planned}
    batches=[]
    for value in values:
        metric={m:value for m in ('ttft','tpot','e2e')}
        batches.append({'status':'complete','batch_medians_ms':metric,'split_admission':False,'engine_start_spread_ms':0,
          'requests':[{'status':'measured','engine_start_rank':0,'metrics_ms':metric}]})
    return cfg,{'status':'complete' if complete else 'failed','runs':batches}


def test_small_observed_deviation_does_not_become_formal_acceptance():
    cfg,record=make_case([99.,100.,101.])
    row=summarize_cell([cfg],[record])
    assert row['observed_all_metrics_within_5pct'] is True
    assert row['formal_repeatability_accepted'] is False
    assert row['formal_prediction_accepted'] is False
    assert row['metrics']['ttft']['rank_worst_abs_ms']==1.


def test_missing_batch_keeps_full_denominator_and_cannot_pass():
    cfg,record=make_case([100.,100.],complete=False)
    row=summarize_cell([cfg],[record])
    assert row['planned_batches']==3
    assert row['captured_batches']==2
    assert row['metrics']['tpot']['rank_fraction_within_5pct']==2/3
    assert row['observed_all_metrics_within_5pct'] is False


def test_constant_batch_median_does_not_hide_request_variation():
    cfg,record=make_case([1.,1.,1.]);cfg['parallel']=2
    for batch,(left,right) in zip(record['runs'],[(.5,1.5),(.8,1.2),(.5,1.5)]):
        batch['requests']=[{'status':'measured','engine_start_rank':rank,
          'metrics_ms':{m:value for m in ('ttft','tpot','e2e')}} for rank,value in enumerate((left,right))]
    row=summarize_cell([cfg],[record])
    assert row['metrics']['ttft']['batch_worst_abs_pct']==0
    assert row['metrics']['ttft']['rank_worst_abs_pct']>5
    assert row['observed_all_metrics_within_5pct'] is False

import copy
import json
from pathlib import Path
import pytest
from tools import native_long_grid_report as report


def make_proof(primary, child):
    cells, refs = [], [primary['identity']['freeze_ref']]
    for source in (primary, child):
        refs += source['freeze']['source_refs']
    for key, records in primary['records'].items():
        if primary['rows'][key]['status'] != 'complete':
            continue
        plan = primary['groups'][key][0]
        raw = records[0]
        counts = {'warmup': plan['warmup_batches'], 'runs': plan['measure_batches']}
        requests = [{'phase': phase, 'repeat': repeat, 'request_index': rank,
            'engine_equivalent': True, 'raw_final_equivalent': True, 'validation_errors': [],
            'saved_engine_sha256': 'equal', 'replayed_engine_sha256': 'equal', 'raw_final_engine_sha256': 'equal'}
            for phase, count in counts.items() for repeat in range(count) for rank in range(plan['parallel'])]
        request_counts = {phase: count * plan['parallel'] for phase, count in counts.items()}
        refs.append(raw['_verified_raw_ref'])
        cells.append({'key': plan['key'], 'status': 'complete', 'full_batch_coverage': True,
            'formal_measurement': True, 'engine_equivalent': True, 'raw_ref': raw['_verified_raw_ref'],
            'batch_counts': counts, 'request_counts': request_counts,
            'total_requests': len(requests), 'requests': requests})
    totals = {phase: sum(c['request_counts'][phase] for c in cells) for phase in ('warmup','runs')}
    return {'schema': 'native-grid-parser-replay-equivalence/v1', 'status': 'passed',
        'engine_equivalent': True, 'completed_cells': len(cells), 'expected_completed_cells': len(cells),
        'source_refs_verified_after_replay': True, 'original_evidence_modified': False,
        'diagnostics_included_in_completed_counts': False, 'source_refs': refs,
        'old_parser': {'file_ref': primary['freeze']['source_refs'][0]},
        'new_parser': {'file_ref': child['freeze']['source_refs'][0]}, 'cells': cells,
        'request_counts': totals, 'total_requests': sum(totals.values()),
        'batch_counts': {phase: sum(c['batch_counts'][phase] for c in cells) for phase in totals}}


def supplement_fixture(tmp_path, monkeypatch, *, total=135, complete=100):
    def source(label, indices, completed):
        freeze_path = tmp_path / label / 'native/freeze.json'
        model = tmp_path / label / 'fixture.gguf'
        freeze = {'hardware_fingerprint': 'same-hardware',
            'extractor_identity': {'sha256': label + '-extractor', 'file_sha256': label + '-file'},
            'artifact_refs': [{'path': str(model), 'sha256': 'same-model'},
                {'path': str(tmp_path / label / 'llama-server.exe'), 'sha256': 'same-server'},
                {'path': str(tmp_path / label / 'ggml-cuda.dll'), 'sha256': 'same-dll'}],
            'environments': {}, 'source_refs': []}
        groups, rows, records = {}, {}, {}
        for index in indices:
            cfg, record = make_case([99., 100., 101.], complete=index in completed)
            cfg.update(job_id='qwen25_cell' + str(index), model=str(model), key='key' + str(index), block=0,
                prompt_token_ids=[1, 2, 3], ctx=2048, batch=64, ubatch=64, threads=16, gpu_layers=-1,
                worker_cpu_mask='0x55555555', poll=0, warmup_batches=2, environment={'LLAMA_ENGINE_TOKEN_TIMES':'1'})
            if index not in completed:
                record['runs'] = []
            key = cfg['job_id'], cfg['condition_id']
            groups[key] = [cfg]
            record['key'] = cfg['key']
            record['_verified_raw_ref'] = {'path': str(tmp_path / label / 'native' / (cfg['key'] + '.json')), 'sha256': 'raw-' + str(index), 'bytes': 1}
            records[key] = [record] if index in completed else []
            rows[key] = report.summarize_cell([cfg], [record] if index in completed else [])
            rows[key]['evidence_verified'] = True
            freeze['environments'][cfg['key']] = {'LLAMA_ENGINE_TOKEN_TIMES': {'is_set': True, 'value': '1'}}
        freeze['source_refs'] = [{'path': str(tmp_path / label / 'execution/tools/native_llama_compare.py'), 'sha256': label + '-file', 'bytes': 1}]
        identity = {'source_id': label, 'freeze_ref': {'path': str(freeze_path), 'sha256': label + '-freeze'},
            'summary_ref': None, 'extractor_identity': freeze['extractor_identity'],
            'end_verified': True, 'execution_status': 'complete' if label == 'supplement' else 'failed', 'verification_errors': []}
        return {'identity': identity, 'freeze': freeze, 'groups': groups, 'rows': rows,
            'failed_attempts': [], 'records': records}
    primary = source('primary', range(total), set(range(complete)))
    child = source('supplement', range(complete, total), set(range(complete, total-1)))
    primary['failed_attempts'] = [{'source_id': 'primary', 'key': 'key100', 'cell_id': 'qwen25_cell100__fixed',
        'status': 'failed', 'error': 'original warmup failure', 'raw_ref': {'sha256': 'old-raw'}}]
    proof = tmp_path / 'offline_equivalence.json'
    proof.write_text(json.dumps(make_proof(primary, child)))
    lineage = {'schema': 'native-grid-supplement-lineage/v1',
        'primary_freeze_ref': primary['identity']['freeze_ref'], 'supplement_freeze_ref': child['identity']['freeze_ref'],
        'replacement_cell_ids': sorted(report._cell_id(key) for key in child['groups']),
        'extractor_patch': {'before_sha256': 'primary-extractor', 'after_sha256': 'supplement-extractor', 'reason': 'SSE heartbeat framing fix'},
        'primary_replay_equivalence': {'status': 'equivalent', 'checked_complete_cells': complete,
            'original_extractor_sha256': 'primary-extractor', 'replay_extractor_sha256': 'supplement-extractor',
            'evidence_ref': report.reference(proof)}}
    lineage_path = tmp_path / 'lineage.json'
    lineage_path.write_text(json.dumps(lineage))
    monkeypatch.setattr(report, '_campaign', lambda path, source_id: primary if source_id == 'primary' else child)
    return primary, child, lineage_path


def test_supplement_35_keeps_135_denominator_and_each_real_source(tmp_path, monkeypatch):
    primary, child, lineage = supplement_fixture(tmp_path, monkeypatch)
    result = report.build_report(tmp_path / 'primary', supplement=tmp_path / 'supplement', lineage=lineage)
    assert result['planned_cells'] == 135
    assert result['complete_cells'] == 134
    assert result['planned_measurement_batches'] == 405
    assert result['planned_requests'] == 405
    assert result['mixed_freezes'] is True
    assert result['sources']['primary']['freeze_ref']['sha256'] == 'primary-freeze'
    assert result['sources']['supplement']['freeze_ref']['sha256'] == 'supplement-freeze'
    assert result['source_per_cell']['qwen25_cell0__fixed']['source_id'] == 'primary'
    assert result['source_per_cell']['qwen25_cell100__fixed']['source_id'] == 'supplement'
    replacement = next(c for c in result['cells'] if c['cell_id'] == 'qwen25_cell100__fixed')
    assert replacement['attempt_history'][0]['error'] == 'original warmup failure'
    assert result['failed_attempts'] == primary['failed_attempts']
    assert result['formal_repeatability_accepted'] is False
    assert result['formal_prediction_accepted'] is False


@pytest.mark.parametrize('change', ['parallel', 'tokens', 'model', 'runtime', 'environment'])
def test_supplement_rejects_configuration_or_identity_drift(tmp_path, monkeypatch, change):
    primary, child, lineage = supplement_fixture(tmp_path, monkeypatch)
    plan = next(iter(child['groups'].values()))[0]
    if change == 'parallel':
        plan['parallel'] = 2
    elif change == 'tokens':
        plan['prompt_token_ids'][1] = 9
    elif change == 'model':
        child['freeze']['artifact_refs'][0]['sha256'] = 'different-model'
    elif change == 'runtime':
        child['freeze']['artifact_refs'][1]['sha256'] = 'different-server'
    elif change == 'environment':
        child['freeze']['environments'][plan['key']] = {'LLAMA_ENGINE_TOKEN_TIMES': {'is_set': False, 'value': None}}
    with pytest.raises(ValueError):
        report.build_report(tmp_path / 'primary', supplement=tmp_path / 'supplement', lineage=lineage)


def test_supplement_requires_offline_equivalence_for_all_reused_cells(tmp_path, monkeypatch):
    primary, child, lineage_path = supplement_fixture(tmp_path, monkeypatch)
    lineage = json.loads(lineage_path.read_text())
    lineage['primary_replay_equivalence']['checked_complete_cells'] = 99
    lineage_path.write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match='all reused complete primary cells'):
        report.build_report(tmp_path / 'primary', supplement=tmp_path / 'supplement', lineage=lineage_path)


def test_supplement_cannot_replace_a_complete_primary_cell(tmp_path, monkeypatch):
    primary, child, lineage = supplement_fixture(tmp_path, monkeypatch)
    key = next(iter(child['rows']))
    primary['rows'][key]['status'] = 'complete'
    payload = json.loads(lineage.read_text())
    payload['primary_replay_equivalence']['checked_complete_cells'] = 101
    lineage.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='already complete primary cell'):
        report.build_report(tmp_path / 'primary', supplement=tmp_path / 'supplement', lineage=lineage)


def test_render_writes_text_before_agg_and_records_crash_with_svg_fallback(tmp_path, monkeypatch):
    cfg, record = make_case([99., 100., 101.])
    row = report.summarize_cell([cfg], [record])
    row['evidence_verified'] = True
    result = {'planned_cells': 1, 'complete_cells': 1, 'captured_measurement_batches': 3,
        'planned_measurement_batches': 3, 'observed_all_metrics_within_5pct_cells': 1,
        'execution_status': 'complete', 'end_verified': True, 'cells': [row]}
    def crash(result_path, key, output_path, log_path):
        for filename in ('report.json', 'cells.csv', 'report.md', 'report.html'):
            assert (tmp_path / filename).is_file()
        log_path.write_text('NumPy incompatible extension')
        raise RuntimeError('simulated native extension crash')
    monkeypatch.setattr(report, '_run_chart', crash)
    state = report.render(result, tmp_path)
    assert state['status'] == 'complete_with_render_warnings'
    assert state['errors'][0]['stage'] == 'isolated_matplotlib_Agg'
    assert 'simulated native extension crash' in state['errors'][0]['error']
    assert (tmp_path / 'qwen25_variability.svg').is_file()
    assert 'qwen25_variability.svg' in (tmp_path / 'report.md').read_text(encoding='utf-8')
    assert json.loads((tmp_path / 'render_status.json').read_text(encoding='utf-8'))['status'] == state['status']


@pytest.mark.parametrize('change', ['top_false', 'count', 'missing_cell', 'wrong_key', 'cell_false',
    'new_parser', 'old_parser', 'missing_raw_ref', 'wrong_raw', 'missing_request',
    'duplicate_request', 'request_hash', 'request_error', 'aggregate'])
def test_proof_body_rejects_mutation_even_when_lineage_hash_is_updated(tmp_path, monkeypatch, change):
    primary, child, lineage_path = supplement_fixture(tmp_path, monkeypatch)
    lineage = json.loads(lineage_path.read_text())
    proof_path = Path(lineage['primary_replay_equivalence']['evidence_ref']['path'])
    proof = json.loads(proof_path.read_text())
    if change == 'top_false': proof['engine_equivalent'] = False
    elif change == 'count': proof['completed_cells'] = 99
    elif change == 'missing_cell': proof['cells'].pop()
    elif change == 'wrong_key': proof['cells'][0]['key'] = 'foreign-key'
    elif change == 'cell_false': proof['cells'][0]['engine_equivalent'] = False
    elif change in ('new_parser', 'old_parser'): proof[change]['file_ref']['sha256'] = 'wrong-parser'
    elif change == 'missing_raw_ref': proof['source_refs'] = [r for r in proof['source_refs'] if r['path'] != proof['cells'][0]['raw_ref']['path']]
    elif change == 'wrong_raw': proof['cells'][0]['raw_ref']['sha256'] = 'wrong-raw'
    elif change == 'missing_request': proof['cells'][0]['requests'].pop()
    elif change == 'duplicate_request': proof['cells'][0]['requests'][1] = proof['cells'][0]['requests'][0]
    elif change == 'request_hash': proof['cells'][0]['requests'][0]['replayed_engine_sha256'] = 'different'
    elif change == 'request_error': proof['cells'][0]['requests'][0]['validation_errors'] = ['bad-token']
    elif change == 'aggregate': proof['total_requests'] -= 1
    proof_path.write_text(json.dumps(proof))
    lineage['primary_replay_equivalence']['evidence_ref'] = report.reference(proof_path)
    lineage_path.write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match='replay'):
        report.build_report(tmp_path / 'primary', supplement=tmp_path / 'supplement', lineage=lineage_path)
