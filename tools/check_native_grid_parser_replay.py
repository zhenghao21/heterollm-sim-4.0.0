"""Read-only engine equivalence audit for a frozen native campaign.

Reconstructs recorded SSE text plus any drained trailing body in memory. Parser
clocks are deliberately synthetic and discarded; this never recovers missing
original client timestamps, promotes failed warmups, or launches native code.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

SCHEMA = 'native-grid-parser-replay-equivalence/v1'
ENGINE_FIELDS = ('timings', 'tokens_evaluated', 'tokens_predicted', 'tokens_cached',
                 'truncated', 'stop_type', 'stop', 'id_slot')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_ref(path):
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


def verify_ref(ref):
    if file_ref(ref['path']) != ref:
        raise ValueError('source evidence changed: '+ref['path'])


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parser_identity(path):
    path = Path(path).resolve(strict=True)
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'post_stream_json']
    if len(nodes) != 1:
        raise ValueError('exactly one post_stream_json definition required')
    node = nodes[0]
    normalized = ast.dump(node, include_attributes=False)
    identity = {'file_ref': file_ref(path), 'function': node.name,
                'ast_basis': 'ast.dump(FunctionDef, include_attributes=False)',
                'function_ast_sha256': hashlib.sha256(normalized.encode('utf-8')).hexdigest(),
                'function_source': ast.get_source_segment(source, node)}
    return identity, compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec')


def wire_bytes(capture):
    rows = capture.get('raw_received_lines')
    if not isinstance(rows, list) or not rows:
        raise ValueError('recorded response lines missing')
    if any(not isinstance(row.get('line'), str) for row in rows):
        raise ValueError('invalid recorded response line')
    trailing = capture.get('trailing_body', '')
    if not isinstance(trailing, str):
        raise ValueError('invalid trailing body')
    return (''.join(row['line'] for row in rows)+trailing).encode('utf-8')


def replay(code, wire, payload):
    # Compile only the selected parser, without importing its simulator module.
    # No URL opener can access the network and no original client time is reused.
    ticks = itertools.count(1., .001)
    scope = {'json': json, 'Request': Request,
             'time': SimpleNamespace(perf_counter=lambda: next(ticks)),
             'urlopen': lambda *_args, **_kwargs: io.BytesIO(wire)}
    exec(code, scope)
    response, boundary = scope['post_stream_json']('http://offline.invalid/completion', payload)
    return response, boundary['mode']  # All generated clock values are discarded.


def engine_fields(response):
    return {name: response[name] for name in ENGINE_FIELDS if name in response}


def raw_final(wire):
    finals = []
    for line in wire.decode('utf-8').splitlines():
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            continue
        event = json.loads(data)
        if isinstance(event, dict) and event.get('stop') is True and isinstance(event.get('timings'), dict):
            finals.append(event)
    if len(finals) != 1:
        raise ValueError('exactly one raw final timing event required')
    return finals[0]


def validate_engine(response, prompt, output):
    timings = response.get('timings', {})
    times = timings.get('engine_token_times_us', [])
    errors = []
    for name, expected in (('prompt_n', prompt), ('predicted_n', output), ('cache_n', 0)):
        if type(timings.get(name)) is not int or timings[name] != expected:
            errors.append(name+'_mismatch')
    if not isinstance(times, list) or len(times) != output or any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
        errors.append('invalid_token_times')
    elif times != sorted(times) or not times or times[0] != timings.get('engine_prompt_last_us') or times[-1] != timings.get('engine_last_token_us'):
        errors.append('invalid_token_endpoints')
    elif type(timings.get('engine_request_begin_us')) not in (int, float) or timings['engine_request_begin_us'] > times[0]:
        errors.append('invalid_request_begin')
    if timings.get('engine_timepoints_complete') is not True:
        errors.append('incomplete_engine_timepoints')
    if response.get('stop') is not True or response.get('truncated') is not False or response.get('stop_type') != 'limit':
        errors.append('invalid_completion_stop')
    return errors


def check_request(code, row, config, prompt_count):
    wire = wire_bytes(row)
    decoded, mode = replay(code, wire, {'stream': True})
    saved = engine_fields(row.get('response', {}))
    actual = engine_fields(decoded)
    terminal = engine_fields(raw_final(wire))
    errors = validate_engine(decoded, prompt_count, config['output'])
    equivalent = canonical(saved) == canonical(actual)
    terminal_equivalent = canonical(actual) == canonical(terminal)
    result = {'request_index': row['request_index'], 'parser_mode': mode,
              'engine_equivalent': equivalent and terminal_equivalent and not errors,
              'saved_engine_sha256': sha(saved), 'replayed_engine_sha256': sha(actual),
              'raw_final_engine_sha256': sha(terminal),
              'raw_final_equivalent': terminal_equivalent, 'validation_errors': errors,
              'recorded_line_count': len(row['raw_received_lines']),
              'trailing_body_bytes': len(row.get('trailing_body', '').encode('utf-8')),
              'client_timestamps_compared': False, 'client_timestamps_reconstructed': False}
    if not equivalent:
        result['different_fields'] = [name for name in ENGINE_FIELDS
            if canonical({name: saved[name]} if name in saved else {}) != canonical({name: actual[name]} if name in actual else {})]
    return result, actual


def inspect_cell(record, code, refs, *, diagnostic=False):
    key = record['key'];config = record['config'];prompt_count = record['prompt_token_count']
    counts = {'warmup': 0, 'runs': 0};batches = {'warmup': 0, 'runs': 0}
    journal_map = {}
    for ref in record.get('batch_journal_refs', []):
        verify_ref(ref);refs.append(ref)
        journal = load_json(ref['path'])
        journal_key = (journal['phase'], journal['repeat'])
        if journal_key in journal_map:
            raise ValueError('duplicate batch journal')
        if journal['key'] != key or journal['config_sha256'] != sha(config) or journal['freeze_sha256'] != record['freeze_sha256']:
            raise ValueError('journal identity mismatch')
        journal_map[journal_key] = journal
    results = []
    for phase in ('warmup', 'runs'):
        for expected_repeat, batch in enumerate(record[phase]):
            if batch['phase'] != phase or batch['repeat'] != expected_repeat:
                raise ValueError('phase/repeat coverage mismatch')
            journal = journal_map.pop((phase, expected_repeat))
            rows = batch['requests']
            if len(rows) != config['parallel'] or len(journal['captures']) != len(rows):
                raise ValueError('parallel request coverage mismatch')
            batches[phase] += 1
            for index, (row, capture) in enumerate(zip(rows, journal['captures'])):
                if row['request_index'] != index:
                    raise ValueError('request index coverage mismatch')
                for name in ('raw_received_lines', 'trailing_body', 'response', 'boundary'):
                    if canonical(row.get(name)) != canonical(capture.get(name)):
                        raise ValueError('raw record/journal mismatch: '+name)
                result, recovered = check_request(code, row, config, prompt_count)
                result.update(phase=phase, repeat=expected_repeat)
                if diagnostic:
                    result.update(formal_measurement=False, old_parser_errors=row.get('errors', []),
                                  saved_engine=engine_fields(row.get('response', {})), recovered_engine=recovered,
                                  recovered_engine_valid=not result['validation_errors'])
                results.append(result);counts[phase] += 1
    if journal_map:
        raise ValueError('unused batch journals')
    coverage = all(batches[phase] == config[setting] for phase, setting in (('warmup', 'warmup_batches'), ('runs', 'measure_batches')))
    return {'key': key, 'model_key': config.get('model_key'), 'status': record['status'],
            'full_batch_coverage': coverage, 'batch_counts': batches, 'request_counts': counts,
            'total_requests': sum(counts.values()), 'engine_equivalent': coverage and all(row['engine_equivalent'] for row in results),
            'formal_measurement': not diagnostic, 'requests': results}


def audit(campaign, new_parser_path, expected_complete=100):
    campaign = Path(campaign).resolve(strict=True);native = campaign/'native'
    freeze_path = native/'freeze.json';freeze = load_json(freeze_path);freeze_hash = sha(freeze)
    refs = [file_ref(freeze_path)]
    old_path = campaign/'execution/tools/native_llama_compare.py'
    old_identity, _ = parser_identity(old_path)
    frozen_parser_refs = [ref for ref in freeze['source_refs'] if Path(ref['path']).resolve() == old_path.resolve()]
    if frozen_parser_refs != [old_identity['file_ref']]:
        raise ValueError('old parser differs from native freeze')
    new_identity, code = parser_identity(new_parser_path)
    refs += [old_identity['file_ref'], new_identity['file_ref']]
    plans = {plan['key']: plan for plan in freeze['plans']}
    cells = [];excluded = [];diagnostics = []
    for receipt_path in sorted(native.glob('*.receipt.json')):
        receipt = load_json(receipt_path);refs.append(file_ref(receipt_path))
        verify_ref(receipt['raw_ref']);refs.append(receipt['raw_ref'])
        record = load_json(receipt['raw_ref']['path'])
        if record['key'] != receipt['key'] or record['status'] != receipt['status'] or record['config'] != plans.get(record['key']):
            raise ValueError('receipt/raw/plan identity mismatch')
        if record['freeze_sha256'] != freeze_hash or receipt['freeze_sha256'] != freeze_hash:
            raise ValueError('receipt/raw freeze mismatch')
        if record['status'] == 'complete':
            cell = inspect_cell(record, code, refs)
            cell['raw_ref'] = receipt['raw_ref'];cells.append(cell)
        else:
            excluded.append({'key': record['key'], 'status': record['status'], 'error': record.get('error'),
                             'raw_ref': receipt['raw_ref'], 'eligible_for_completed_measurement_reuse': False})
            if record['warmup'] and not record['runs']:
                cell = inspect_cell(record, code, refs, diagnostic=True)
                cell['raw_ref'] = receipt['raw_ref'];diagnostics.append(cell)
    if len(cells) != expected_complete:
        raise ValueError(f'expected {expected_complete} completed cells, found {len(cells)}')
    dedup = {ref['path']: ref for ref in refs}
    for ref in dedup.values():verify_ref(ref)
    equivalent = bool(cells) and all(cell['engine_equivalent'] for cell in cells)
    return {'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'campaign': str(campaign), 'status': 'passed' if equivalent else 'failed',
            'engine_equivalent': equivalent, 'old_parser': old_identity, 'new_parser': new_identity,
            'source_refs': list(dedup.values()), 'source_refs_verified_after_replay': True,
            'comparison': {'fields': list(ENGINE_FIELDS), 'timings': 'entire object including every engine_token_times_us entry',
                           'equality': 'canonical JSON exact equality including numeric representation',
                           'wire_reconstruction': 'UTF-8 encode ordered recorded line strings then trailing_body; no new HTTP requests',
                           'client_timestamps_compared': False, 'client_timestamps_reconstructed': False,
                           'replayed_client_clock': 'synthetic and discarded; never measurement truth',
                           'trailing_body_without_arrival_timestamps': 'engine diagnostic only'},
            'completed_cells': len(cells), 'expected_completed_cells': expected_complete,
            'request_counts': {phase: sum(cell['request_counts'][phase] for cell in cells) for phase in ('warmup', 'runs')},
            'batch_counts': {phase: sum(cell['batch_counts'][phase] for cell in cells) for phase in ('warmup', 'runs')},
            'total_requests': sum(cell['total_requests'] for cell in cells), 'cells': cells,
            'failed_or_incomplete_cells_excluded': excluded,
            'failed_premeasurement_engine_diagnostics': diagnostics,
            'diagnostics_included_in_completed_counts': False, 'original_evidence_modified': False,
            'formal_prediction_acceptance': False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--new-parser', type=Path, default=Path(__file__).with_name('native_llama_compare.py'))
    ap.add_argument('--expected-complete', type=int, default=100)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    target = args.output or args.campaign/'parser_replay_equivalence.json'
    if target.exists():raise FileExistsError('audit output already exists: '+str(target))
    result = audit(args.campaign, args.new_parser, args.expected_complete)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(target.resolve()), 'status': result['status'],
                      'completed_cells': result['completed_cells'], 'request_counts': result['request_counts'],
                      'total_requests': result['total_requests']}, ensure_ascii=False))
    return 0 if result['engine_equivalent'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
