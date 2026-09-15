"""Pure parser replay audit checks; no native process or network access."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from tools import check_native_grid_parser_replay as audit

PARSER = Path(audit.__file__).with_name('native_llama_compare.py')


def sample_row(*, trailing=False):
    final = {'content': '', 'tokens': [], 'id_slot': 0, 'stop': True, 'stop_type': 'limit',
             'truncated': False, 'tokens_predicted': 2, 'tokens_evaluated': 1, 'tokens_cached': 2,
             'timings': {'prompt_n': 1, 'predicted_n': 2, 'cache_n': 0,
                         'engine_request_begin_us': 10, 'engine_prompt_last_us': 20,
                         'engine_last_token_us': 30, 'engine_token_times_us': [20, 30],
                         'engine_timepoints_complete': True}}
    events = [{'content': 'a', 'tokens': [1], 'stop': False},
              {'content': 'b', 'tokens': [2], 'stop': False}, final]
    lines = [':\n', '\n'] + ['data: '+json.dumps(e)+'\n\n' for e in events]
    row = {'request_index': 0, 'response': deepcopy(final),
           'raw_received_lines': [{'line': line, 'received_monotonic_s': 100.+i} for i, line in enumerate(lines)],
           'boundary': {'mode': 'sse', 'request_start_monotonic_s': 99., 'last_token_monotonic_s': 105.}}
    if trailing:
        row['raw_received_lines'] = row['raw_received_lines'][:3]
        row['trailing_body'] = ''.join(lines[3:])
        row['response'] = deepcopy(events[0])
        row['boundary']['mode'] = 'json'
    return row, final


def test_replay_compares_engine_fields_and_leaves_original_client_evidence_unchanged():
    identity, code = audit.parser_identity(PARSER)
    row, _ = sample_row();before = deepcopy(row)
    result, recovered = audit.check_request(code, row, {'output': 2}, 1)
    assert result['engine_equivalent'] and result['raw_final_equivalent']
    assert result['saved_engine_sha256'] == result['replayed_engine_sha256']
    assert recovered['timings']['engine_token_times_us'] == [20, 30]
    assert not result['client_timestamps_compared'] and not result['client_timestamps_reconstructed']
    assert row == before
    assert identity['function_ast_sha256'] and identity['file_ref']['sha256']


@pytest.mark.parametrize('field,value', [('predicted_n', 3), ('engine_token_times_us', [20, 31]), ('cache_n', 0.0)])
def test_every_timing_value_and_numeric_type_must_match(field, value):
    _, code = audit.parser_identity(PARSER);row, _ = sample_row()
    row['response']['timings'][field] = value
    result, _ = audit.check_request(code, row, {'output': 2}, 1)
    assert not result['engine_equivalent']
    assert 'timings' in result['different_fields']


def test_trailing_body_recovery_is_a_difference_not_equivalent_original_measurement():
    _, code = audit.parser_identity(PARSER);row, final = sample_row(trailing=True)
    result, recovered = audit.check_request(code, row, {'output': 2}, 1)
    assert not result['engine_equivalent']
    assert result['raw_final_equivalent'] and not result['validation_errors']
    assert recovered == audit.engine_fields(final)
    assert result['trailing_body_bytes'] > 0
    assert not result['client_timestamps_reconstructed']


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def campaign(tmp_path):
    native = tmp_path/'native';native.mkdir()
    old = tmp_path/'execution/tools';old.mkdir(parents=True)
    parser = old/'native_llama_compare.py';parser.write_bytes(PARSER.read_bytes())
    config = {'key': 'sample', 'model_key': 'test', 'parallel': 1, 'output': 2,
              'warmup_batches': 1, 'measure_batches': 1}
    freeze = {'plans': [config], 'source_refs': [audit.file_ref(parser)]}
    write(native/'freeze.json', freeze)
    record = {'key': 'sample', 'status': 'complete', 'config': config,
              'freeze_sha256': audit.sha(freeze), 'prompt_token_count': 1,
              'warmup': [], 'runs': [], 'batch_journal_refs': []}
    for phase in ('warmup', 'runs'):
        row, _ = sample_row()
        record[phase].append({'phase': phase, 'repeat': 0, 'requests': [row]})
        journal = {'key': 'sample', 'phase': phase, 'repeat': 0, 'captures': [deepcopy(row)],
                   'freeze_sha256': audit.sha(freeze), 'config_sha256': audit.sha(config)}
        path = native/(phase+'.batch.json');write(path, journal)
        record['batch_journal_refs'].append(audit.file_ref(path))
    raw = native/'sample.json';write(raw, record)
    write(native/'sample.receipt.json', {'key': 'sample', 'status': 'complete',
          'freeze_sha256': audit.sha(freeze), 'raw_ref': audit.file_ref(raw)})
    return tmp_path


def test_campaign_report_counts_all_batches_and_requests(tmp_path):
    root = campaign(tmp_path)
    result = audit.audit(root, PARSER, expected_complete=1)
    assert result['status'] == 'passed' and result['engine_equivalent']
    assert result['request_counts'] == {'warmup': 1, 'runs': 1}
    assert result['total_requests'] == 2 and result['completed_cells'] == 1
    assert result['cells'][0]['full_batch_coverage']
    assert result['source_refs_verified_after_replay']
    assert result['old_parser']['file_ref']['path'].endswith('native_llama_compare.py')
    assert not result['diagnostics_included_in_completed_counts']


def test_campaign_detects_reference_mutation_and_cell_count_mismatch(tmp_path):
    root = campaign(tmp_path)
    with pytest.raises(ValueError, match='expected 2'):
        audit.audit(root, PARSER, expected_complete=2)
    raw = root/'native/sample.json';raw.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='source evidence changed'):
        audit.audit(root, PARSER, expected_complete=1)


def test_completed_cell_cannot_pass_if_a_required_batch_is_missing(tmp_path):
    root = campaign(tmp_path)
    record = audit.load_json(root/'native/sample.json')
    record['runs'] = [];record['batch_journal_refs'] = record['batch_journal_refs'][:1]
    _, code = audit.parser_identity(PARSER)
    result = audit.inspect_cell(record, code, [])
    assert not result['full_batch_coverage'] and not result['engine_equivalent']
