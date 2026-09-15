"""Recompute the frozen 135-cell native length screen from immutable raw evidence.

A three-repeat single-process screen never becomes formal repeatability or
prediction acceptance. Every planned cell remains in the report.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys
import traceback
import os
import time
import subprocess
import io

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import native_repeatability_experiment as experiment
METRICS=('ttft','tpot','e2e')
MODEL_NAMES={'qwen25':'Qwen2.5 0.5B','qwen35':'Qwen3.5 0.8B','qwen38':'Qwen3.8 27B','smollm2':'SmolLM2 1.7B','tinyllama':'TinyLlama 1.1B'}

def reference(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

def summarize_cell(plans,records):
    summary=experiment.summarize_group(plans,records)
    c=plans[0]
    result={'id':c['job_id'],'model_key':c.get('model_key',c['job_id'].split('_')[0]),
      'prompt_tokens':c.get('expected_prompt_tokens',len(c.get('prompt_token_ids',[]))),
      'output_tokens':c['output'],'parallel':c['parallel'],
      'planned_blocks':summary['planned_blocks'],'complete_blocks':summary['complete_blocks'],
      'planned_batches':summary['planned_batches'],'captured_batches':summary['captured_batches'],
      'split_admission_batches':summary['split_admission_batches'],
      'status':'complete' if summary['complete_blocks']==len(plans) else 'incomplete_or_failed',
      'initial_screen_only':True,'formal_repeatability_accepted':False,
      'formal_prediction_accepted':False,'metrics':{},'raw_summary':summary}
    for metric in METRICS:
        source=summary['metrics'][metric]
        if source['status']!='measured':
            result['metrics'][metric]={'status':source['status'],'observed_within_5pct':False};continue
        b=source['batch_medians'];ranks=source['request_ranks'];joint=source['all_request_deviations']
        complete=result['status']=='complete' and b['observed_samples']==b['planned_samples'] and all(r['observed_samples']==r['planned_samples'] for r in ranks)
        worst=joint.get('worst_abs_deviation_pct');batch_worst=b.get('worst_abs_deviation_pct')
        result['metrics'][metric]={
          'status':'measured','native_median_ms':b.get('median_ms'),
          'batch_worst_abs_pct':batch_worst,'batch_p90_abs_pct':b.get('p90_abs_deviation_pct'),
          'rank_worst_abs_pct':worst,'rank_p90_abs_pct':joint.get('p90_abs_deviation_pct'),
          'rank_fraction_within_5pct':joint.get('fraction_within_5pct'),
          'batch_sample_cv_pct':b.get('sample_cv_pct'),
          'rank_worst_abs_ms':max((abs(d)*r['median_ms']/100 for r in ranks for d in r.get('signed_deviation_pct',[]) if r.get('median_ms') is not None),default=None),
          'observed_within_5pct':bool(complete and worst is not None and batch_worst is not None and worst<=5 and batch_worst<=5)}
    result['observed_all_metrics_within_5pct']=all(result['metrics'][m].get('observed_within_5pct',False) for m in METRICS)
    return result

def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _groups(plans):
    groups = {}
    for plan in plans:
        groups.setdefault((plan['job_id'], plan['condition_id']), []).append(plan)
    return groups


def _cell_id(key):
    return key[0] + '__' + key[1]


def _campaign(campaign, source_id):
    campaign = Path(campaign).resolve()
    data = campaign / 'native'
    freeze_path = data / 'freeze.json'
    freeze = _read_json(freeze_path)
    errors = []
    # Recheck immutable source/protocol evidence, not restored live clock state.
    # Large model/runtime digests below are taken from the bound freeze; no 27GB
    # model rehash and no native/model/profiler invocation occurs in reporting.
    try:
        experiment.verify_refs(freeze['source_refs'])
        experiment.verify_refs([freeze['protocol_ref'], freeze['source_snapshot_ref']])
    except Exception as exc:
        errors.append({'source_id': source_id, 'stage': 'freeze_references', 'error': repr(exc)})
    groups = _groups(freeze['plans'])
    rows, records, failed = {}, {}, []
    for key, plans in groups.items():
        selected = []
        for plan in plans:
            try:
                record = experiment.load_completed(data, freeze, plan)
                if record is not None:
                    record['_verified_raw_ref'] = reference(data / (plan['key'] + '.json'))
                    selected.append(record)
                    if record.get('status') != 'complete':
                        raw_path = data / (plan['key'] + '.json')
                        receipt_path = data / (plan['key'] + '.receipt.json')
                        failed.append({'source_id': source_id, 'cell_id': _cell_id(key),
                            'key': plan['key'], 'status': record.get('status'),
                            'error': record.get('error', record.get('stop_reason')),
                            'warmup_batches': len(record.get('warmup', [])),
                            'formal_batches': len(record.get('runs', [])),
                            'started_utc': record.get('started_utc'), 'completed_utc': record.get('completed_utc'),
                            'raw_ref': reference(raw_path) if raw_path.exists() else None,
                            'receipt_ref': reference(receipt_path) if receipt_path.exists() else None})
            except Exception as exc:
                errors.append({'source_id': source_id, 'key': plan['key'], 'error': repr(exc)})
        records[key] = selected
        rows[key] = summarize_cell(plans, selected)
    summaries = sorted(data.glob('summary_*.json'))
    end = _read_json(summaries[-1]) if summaries else {}
    if summaries and (end.get('schema') != 'native-repeatability-results/v1' or end.get('freeze_sha256') != experiment.digest(freeze)):
        errors.append({'source_id': source_id, 'stage': 'summary_binding', 'error': 'summary schema/freeze digest mismatch'})
        end = {**end, 'end_verified': False}
    raw_failure_keys = {failure['key'] for failure in failed}
    plan_keys = {plan['key']: (plan['job_id'], plan['condition_id']) for plan in freeze['plans']}
    for event in end.get('failures', []):
        if event.get('key') not in raw_failure_keys:
            key = plan_keys.get(event.get('key'))
            failed.append({'source_id': source_id, 'cell_id': _cell_id(key) if key else None,
                'key': event.get('key'), 'status': 'failed', 'error': event.get('error'),
                'evidence_kind': 'summary_premeasurement_failure', 'raw_ref': None})
    identity = {'source_id': source_id, 'campaign': str(campaign), 'freeze_ref': reference(freeze_path),
        'summary_ref': reference(summaries[-1]) if summaries else None,
        'source_snapshot_ref': freeze['source_snapshot_ref'], 'protocol_ref': freeze['protocol_ref'],
        'extractor_identity': freeze.get('extractor_identity'),
        'hardware_fingerprint': freeze.get('hardware_fingerprint'),
        'execution_status': end.get('status', 'running_or_interrupted'),
        'end_verified': end.get('end_verified') is True, 'verification_errors': errors}
    for row in rows.values():
        row['evidence_verified'] = bool(identity['end_verified'] and not errors)
        if not row['evidence_verified']:
            row['observed_all_metrics_within_5pct'] = False
            for metric in row['metrics'].values():
                metric['observed_within_5pct'] = False
    return {'identity': identity, 'freeze': freeze, 'groups': groups, 'rows': rows,
            'records': records, 'failed_attempts': failed}


def _frozen_artifact_sha(freeze, path):
    path = str(Path(path).resolve()).casefold()
    matches = [item['sha256'] for item in freeze.get('artifact_refs', [])
               if str(Path(item['path']).resolve()).casefold() == path]
    if len(set(matches)) != 1:
        raise ValueError('one frozen model/runtime artifact SHA256 required: ' + path)
    return matches[0]


def _runtime_digests(freeze):
    result = {}
    for item in freeze.get('artifact_refs', []):
        path = Path(item['path'])
        if path.suffix.lower() in ('.dll', '.exe'):
            name = path.name.lower()
            if name in result and result[name] != item['sha256']:
                raise ValueError('ambiguous frozen runtime artifact: ' + name)
            result[name] = item['sha256']
    if 'llama-server.exe' not in result:
        raise ValueError('frozen llama-server.exe identity missing')
    return result


def _normalized_plan(plan, freeze):
    value = dict(plan)
    value['model'] = {'sha256': _frozen_artifact_sha(freeze, plan['model'])}
    return value


def _verify_replay_document(proof, primary, supplement):
    old, new = primary['freeze'], supplement['freeze']
    if (proof.get('schema') != 'native-grid-parser-replay-equivalence/v1'
            or proof.get('status') != 'passed' or proof.get('engine_equivalent') is not True
            or proof.get('source_refs_verified_after_replay') is not True
            or proof.get('original_evidence_modified') is not False
            or proof.get('diagnostics_included_in_completed_counts') is not False):
        raise ValueError('replay proof schema/status/equivalence invalid')
    refs = proof.get('source_refs', [])
    if not isinstance(refs, list):
        raise ValueError('replay source refs invalid')
    freeze_ref = primary['identity']['freeze_ref']
    if not any(r.get('path') == freeze_ref['path'] and r.get('sha256') == freeze_ref['sha256'] for r in refs):
        raise ValueError('replay proof not bound to primary freeze')
    for name, freeze in (('old_parser', old), ('new_parser', new)):
        parser_ref = proof.get(name, {}).get('file_ref', {})
        if not any(r.get('path') == parser_ref.get('path') and r.get('sha256') == parser_ref.get('sha256') for r in refs):
            raise ValueError('replay parser absent from source refs: ' + name)
        if not parser_ref.get('sha256') or parser_ref['sha256'] != freeze.get('extractor_identity', {}).get('file_sha256'):
            raise ValueError('replay parser file identity mismatch: ' + name)
        if not any(r.get('sha256') == parser_ref['sha256'] and Path(r['path']).name == 'native_llama_compare.py' for r in freeze.get('source_refs', [])):
            raise ValueError('replay parser not bound to frozen source: ' + name)
    expected = {}
    for key, row in primary['rows'].items():
        if row['status'] == 'complete':
            for plan in primary['groups'][key]:
                records = [r for r in primary['records'].get(key, []) if r.get('key') == plan['key'] and r.get('status') == 'complete']
                if len(records) != 1:
                    raise ValueError('replay requires one verified original raw per plan')
                expected[plan['key']] = (plan, records[0])
    cells = proof.get('cells', [])
    if (not isinstance(cells, list) or len(cells) != len(expected)
            or len({c.get('key') for c in cells}) != len(cells)
            or {c.get('key') for c in cells} != set(expected)
            or proof.get('completed_cells') != len(expected) or proof.get('expected_completed_cells') != len(expected)):
        raise ValueError('replay cell coverage mismatch')
    totals = {'warmup': 0, 'runs': 0}
    batch_totals = {'warmup': 0, 'runs': 0}
    for cell in cells:
        plan, raw = expected[cell['key']]
        raw_ref = raw['_verified_raw_ref']
        if not any(r.get('path') == raw_ref['path'] and r.get('sha256') == raw_ref['sha256'] for r in refs):
            raise ValueError('replay raw absent from source refs')
        if (cell.get('raw_ref', {}).get('path') != raw_ref['path']
                or cell.get('raw_ref', {}).get('sha256') != raw_ref['sha256']):
            raise ValueError('replay raw identity mismatch')
        batches = {'warmup': plan['warmup_batches'], 'runs': plan['measure_batches']}
        counts = {phase: count * plan['parallel'] for phase, count in batches.items()}
        requests = cell.get('requests', [])
        expected_indices = {(phase, repeat, rank) for phase, count in batches.items()
                            for repeat in range(count) for rank in range(plan['parallel'])}
        actual_indices = [(r.get('phase'), r.get('repeat'), r.get('request_index')) for r in requests]
        if (cell.get('status') != 'complete' or cell.get('engine_equivalent') is not True
                or cell.get('full_batch_coverage') is not True or cell.get('formal_measurement') is not True
                or cell.get('batch_counts') != batches or cell.get('request_counts') != counts
                or cell.get('total_requests') != sum(counts.values())
                or len(actual_indices) != len(expected_indices) or set(actual_indices) != expected_indices):
            raise ValueError('replay batch/request coverage mismatch')
        for request in requests:
            hashes = [request.get(k) for k in ('saved_engine_sha256', 'replayed_engine_sha256', 'raw_final_engine_sha256')]
            if (request.get('engine_equivalent') is not True or request.get('raw_final_equivalent') is not True
                    or request.get('validation_errors') != [] or not hashes[0] or len(set(hashes)) != 1):
                raise ValueError('replay request engine equivalence failure')
        for phase in totals:
            totals[phase] += counts[phase]
            batch_totals[phase] += batches[phase]
    if proof.get('request_counts') != totals or proof.get('batch_counts') != batch_totals or proof.get('total_requests') != sum(totals.values()):
        raise ValueError('replay aggregate counts mismatch')


def _verify_lineage(primary, supplement, lineage_path):
    """Lineage schema is intentionally explicit and does not merge freezes.

    Required JSON fields: schema=native-grid-supplement-lineage/v1,
    primary_freeze_ref, supplement_freeze_ref, replacement_cell_ids (job__condition),
    extractor_patch={before_sha256,after_sha256,reason}, and
    primary_replay_equivalence={status:equivalent,checked_complete_cells:N,
      original_extractor_sha256,replay_extractor_sha256,evidence_ref:{path,sha256}}.
    evidence_ref points to an immutable offline engine equivalence receipt. It
    must be created by the offline replay, never by this reporting tool.
    """
    lineage_path = Path(lineage_path).resolve()
    lineage = _read_json(lineage_path)
    if lineage.get('schema') != 'native-grid-supplement-lineage/v1':
        raise ValueError('explicit native-grid-supplement-lineage/v1 required')
    for label, source in (('primary', primary), ('supplement', supplement)):
        claimed = lineage.get(label + '_freeze_ref', {})
        expected = source['identity']['freeze_ref']
        if claimed.get('sha256') != expected['sha256'] or Path(claimed.get('path', '')).resolve() != Path(expected['path']):
            raise ValueError(label + ' lineage freeze identity mismatch')
    old, new = primary['freeze'], supplement['freeze']
    old_hash = old.get('extractor_identity', {}).get('sha256')
    new_hash = new.get('extractor_identity', {}).get('sha256')
    patch = lineage.get('extractor_patch', {})
    if not old_hash or not new_hash or patch.get('before_sha256') != old_hash or patch.get('after_sha256') != new_hash or not patch.get('reason'):
        raise ValueError('extractor patch before/after SHA256 and reason required')
    replay = lineage.get('primary_replay_equivalence', {})
    complete_count = sum(row['status'] == 'complete' for row in primary['rows'].values())
    if (replay.get('status') != 'equivalent' or replay.get('checked_complete_cells') != complete_count
            or replay.get('original_extractor_sha256') != old_hash
            or replay.get('replay_extractor_sha256') != new_hash):
        raise ValueError('all reused complete primary cells require bound offline engine replay equivalence')
    proof_ref = replay.get('evidence_ref')
    if not isinstance(proof_ref, dict) or not proof_ref.get('path') or not proof_ref.get('sha256'):
        raise ValueError('offline engine equivalence evidence_ref required')
    actual_proof = reference(proof_ref['path'])
    if proof_ref['sha256'] != actual_proof['sha256'] or ('bytes' in proof_ref and proof_ref['bytes'] != actual_proof['bytes']):
        raise ValueError('offline engine equivalence evidence SHA256/size differs')
    proof = _read_json(proof_ref['path'])
    subset = set(supplement['groups'])
    if not subset or not subset.issubset(primary['groups']):
        raise ValueError('supplement must contain only cells in the original planned denominator')
    claimed_ids = lineage.get('replacement_cell_ids', [])
    if len(claimed_ids) != len(set(claimed_ids)) or set(claimed_ids) != {_cell_id(key) for key in subset}:
        raise ValueError('lineage replacement cell scope differs from supplement freeze')
    if old.get('hardware_fingerprint') != new.get('hardware_fingerprint'):
        raise ValueError('supplement hardware fingerprint differs')
    if _runtime_digests(old) != _runtime_digests(new):
        raise ValueError('supplement runtime binary/DLL frozen identities differ')
    for key in sorted(subset):
        if primary['rows'][key]['status'] == 'complete':
            raise ValueError('supplement cannot replace already complete primary cell: ' + _cell_id(key))
        before = sorted(primary['groups'][key], key=lambda p: p['key'])
        after = sorted(supplement['groups'][key], key=lambda p: p['key'])
        if [_normalized_plan(p, old) for p in before] != [_normalized_plan(p, new) for p in after]:
            raise ValueError('supplement exact token/runtime configuration differs: ' + _cell_id(key))
        for plan in before:
            old_environment = old.get('environments', {}).get(plan['key'])
            new_environment = new.get('environments', {}).get(plan['key'])
            if old_environment is None or old_environment != new_environment:
                raise ValueError('supplement actual set/unset environment differs: ' + _cell_id(key))
    _verify_replay_document(proof, primary, supplement)
    return {'status': 'verified', 'lineage_ref': reference(lineage_path),
        'replacement_cell_ids': sorted(claimed_ids), 'extractor_patch': patch,
        'primary_replay_equivalence': replay,
        'configuration_comparison': 'all plan fields exact; model paths compared by frozen SHA256',
        'artifact_verification_scope': 'frozen model/runtime digests compared; large binaries not rehashed during report',
        'same_freeze': False, 'scope': 'native_repeatability_only_no_cost_fit'}


def build_report(campaign, supplement=None, lineage=None):
    primary = _campaign(campaign, 'primary')
    sources = {'primary': primary}
    comparison = None
    if supplement is not None:
        child = _campaign(supplement, 'supplement')
        comparison = _verify_lineage(primary, child, lineage or Path(supplement) / 'lineage.json')
        sources['supplement'] = child
    elif lineage is not None:
        raise ValueError('--lineage requires --supplement')
    cells, source_per_cell = [], {}
    for key, original in primary['rows'].items():
        source_id = 'supplement' if 'supplement' in sources and key in sources['supplement']['rows'] else 'primary'
        selected = sources[source_id]
        row = dict(selected['rows'][key])
        row['metrics'] = {name: dict(value) for name, value in row['metrics'].items()}
        accepted = [record for record in selected['records'].get(key, []) if record.get('status') == 'complete']
        accepted_row = summarize_cell(selected['groups'][key], accepted)
        row['accepted_metrics'] = accepted_row['metrics']
        row['accepted_captured_batches'] = accepted_row['captured_batches']
        row['diagnostic_captured_batches'] = row['captured_batches'] - accepted_row['captured_batches']
        row['diagnostic_only'] = row['status'] != 'complete' or not row.get('evidence_verified')
        if row['diagnostic_only']:
            row['observed_all_metrics_within_5pct'] = False
            for metric in row['metrics'].values():
                metric['observed_within_5pct'] = False
                metric['diagnostic_only'] = True
        row['cell_id'] = _cell_id(key)
        row['source_per_cell'] = {'source_id': source_id, 'freeze_ref': selected['identity']['freeze_ref'],
            'extractor_identity': selected['identity']['extractor_identity'],
            'original_status': original['status'], 'replacement': source_id == 'supplement'}
        row['attempt_history'] = [attempt for source in sources.values() for attempt in source['failed_attempts']
                                  if attempt.get('cell_id') == _cell_id(key)]
        source_per_cell[_cell_id(key)] = row['source_per_cell']
        cells.append(row)
    identities = {key: source['identity'] for key, source in sources.items()}
    errors = [error for source in sources.values() for error in source['identity']['verification_errors']]
    all_end_verified = all(source['identity']['end_verified'] for source in sources.values())
    status = primary['identity']['execution_status'] if len(sources) == 1 else (
        'combined_complete' if all(row['status'] == 'complete' for row in cells) and all_end_verified and not errors
        else 'combined_incomplete_or_failed')
    result = {'schema': 'native-long-grid-screen/v2' if len(sources) > 1 else 'native-long-grid-screen/v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'freeze_ref': primary['identity']['freeze_ref'],
        'summary_ref': primary['identity']['summary_ref'], 'source_ref': reference(Path(__file__)),
        'mixed_freezes': len(sources) > 1, 'sources': identities, 'source_per_cell': source_per_cell,
        'supplement_lineage': comparison, 'planned_cells': len(primary['groups']),
        'complete_cells': sum(c['status'] == 'complete' for c in cells),
        'observed_all_metrics_within_5pct_cells': sum(c['observed_all_metrics_within_5pct'] for c in cells),
        'planned_measurement_batches': sum(c['planned_batches'] for c in cells),
        'captured_measurement_batches': sum(c['accepted_captured_batches'] for c in cells),
        'diagnostic_captured_batches': sum(c['diagnostic_captured_batches'] for c in cells),
        'planned_requests': sum(c['planned_batches'] * c['parallel'] for c in cells),
        'end_verified': all_end_verified, 'execution_status': status, 'verification_errors': errors,
        'failed_attempts': [attempt for source in sources.values() for attempt in source['failed_attempts']],
        'formal_repeatability_accepted': False, 'formal_prediction_accepted': False,
        'limitations': ['Three repeats in one process are initial screening only.',
            'Relative variability is not simulator prediction error; absolute milliseconds are reported separately.',
            'All original planned cells, failed requests and missing samples remain in denominators.',
            'Supplement replacement retains both freeze and extractor identities and all failed attempt history.',
            'Longer output averages more decode intervals; a lower percentage alone does not prove elimination of absolute jitter.'],
        'cells': cells}
    if errors or not all_end_verified:
        result['observed_all_metrics_within_5pct_cells'] = 0
        for row in cells:
            row['evidence_verified'] = False
            row['diagnostic_only'] = True
            row['observed_all_metrics_within_5pct'] = False
            for metric in row['metrics'].values():
                metric['observed_within_5pct'] = False
                metric['diagnostic_only'] = True
    return result


def _atomic_text(path, text, encoding='utf-8'):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(text, encoding=encoding)
    os.replace(temporary, path)


def _model_rows(result, key):
    return [row for row in result['cells'] if row['model_key'] == key]


def _markdown(result):
    lines = ['# 长 token Native 初筛', '',
        f"采集状态：{result['execution_status']}；结束核验：{result['end_verified']}。",
        f"完成 {result['complete_cells']}/{result['planned_cells']} 场景、{result['captured_measurement_batches']}/{result['planned_measurement_batches']} 正式批次。",
        f"本次全部三指标观察偏差≤5%的场景：{result['observed_all_metrics_within_5pct_cells']}/{result['planned_cells']}。", '',
        '**这是三次重复的初筛，不是正式±5%保证，也不是仿真预测误差验收。**', '',
        '| 模型 | 完整场景 | 本次三项观察偏差≤5% |', '|---|---:|---:|']
    for key, name in MODEL_NAMES.items():
        rows = _model_rows(result, key)
        if rows:
            lines.append(f"| {name} | {sum(c['status']=='complete' for c in rows)}/{len(rows)} | {sum(c['observed_all_metrics_within_5pct'] for c in rows)}/{len(rows)} |")
    if result.get('mixed_freezes'):
        lines += ['', '## 原始采样与补采来源', '', '两轮拥有不同 freeze 和提取器身份；每格来源见 source_per_cell，未合成为同一轮。', '']
        for label, source in result.get('sources', {}).items():
            lines.append(f"- {label}: freeze SHA256 `{source['freeze_ref']['sha256']}`；extractor SHA256 `{source['extractor_identity']['sha256']}`。")
        lines += ['', '补采只替换原未完成格；原完成格由绑定凭据证明离线 engine 重放等价后复用。']
    failures = result.get('failed_attempts', [])
    if failures:
        lines += ['', '## 保留的失败尝试', '', '| 来源 | 场景 | 错误 |', '|---|---|---|']
        for event in failures:
            error = str(event.get('error') or event.get('status')).replace('|', '/').replace('\n', ' ')
            lines.append(f"| {event['source_id']} | {event.get('cell_id') or event.get('key')} | {error} |")
    lines += ['', '每格完整指标、P90、最大偏差、绝对毫秒、覆盖率及原始失败记录引用见 report.json / cells.csv。', '']
    return lines


def _html(result, images):
    body = '<h1>Native length screen</h1><p>' + html.escape(
        f"{result['complete_cells']}/{result['planned_cells']} complete; status {result['execution_status']}; end verified {result['end_verified']}.") + '</p>'
    body += '<p>Three repeats are initial screening only; no formal repeatability or prediction acceptance.</p>'
    if result.get('mixed_freezes'):
        body += '<p>Mixed source campaigns: each cell retains its actual freeze and extractor identity.</p>'
    body += '<p>Full metric tables and preserved failures: <a href="report.md">report.md</a>, <a href="cells.csv">cells.csv</a>, <a href="report.json">report.json</a>.</p>'
    for key, image in images.items():
        body += '<h2>' + html.escape(MODEL_NAMES.get(key, key)) + '</h2><img style="max-width:100%" src="' + html.escape(image, quote=True) + '">'
    return '<!doctype html><meta charset="utf-8"><title>Native length screen</title><body style="font-family:Arial;max-width:1200px;margin:32px auto">' + body + '</body>'


def _metric_value(rows, metric, prompt, output, concurrency):
    for row in rows:
        if (row['prompt_tokens'], row['output_tokens'], row['parallel']) == (prompt, output, concurrency):
            if row['status'] == 'complete' and row.get('evidence_verified'):
                return row['metrics'][metric].get('rank_worst_abs_pct')
    return None


def _draw_matplotlib(result, key, output):
    # Only called inside the disposable chart process, so native-extension import
    # crashes cannot destroy the already written Markdown/JSON/CSV in the parent.
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    rows = _model_rows(result, key)
    ps = sorted({r['prompt_tokens'] for r in rows})
    outs = sorted({r['output_tokens'] for r in rows})
    cs = sorted({r['parallel'] for r in rows})
    fig, axs = plt.subplots(3, len(cs), figsize=(11, 10), squeeze=False)
    try:
        cmap = plt.colormaps['YlOrRd'].copy()
        cmap.set_bad('#d0d0d0')
        for mi, metric in enumerate(METRICS):
            for ci, concurrency in enumerate(cs):
                matrix = np.full((len(ps), len(outs)), np.nan)
                for i, prompt in enumerate(ps):
                    for j, output_tokens in enumerate(outs):
                        value = _metric_value(rows, metric, prompt, output_tokens, concurrency)
                        if value is not None:
                            matrix[i, j] = value
                ax = axs[mi, ci]
                im = ax.imshow(matrix, vmin=0, vmax=20, cmap=cmap, aspect='auto')
                ax.set_xticks(range(len(outs)), outs)
                ax.set_yticks(range(len(ps)), ps)
                ax.set_xlabel('Output tokens')
                ax.set_ylabel('Input tokens')
                ax.set_title(f'{metric.upper()} | concurrency {concurrency}')
                for i in range(len(ps)):
                    for j in range(len(outs)):
                        value = matrix[i, j]
                        ax.text(j, i, 'missing' if np.isnan(value) else f'{value:.2f}%', ha='center', va='center',
                            color='white' if value > 13 else 'black', fontsize=10)
        fig.suptitle(MODEL_NAMES.get(key, key) + ' | Maximum per-request-rank deviation (%)\n3 repeats per cell; initial screen only', fontsize=14)
        fig.subplots_adjust(top=.9, bottom=.08, left=.08, right=.87, hspace=.5, wspace=.4)
        fig.colorbar(im, cax=fig.add_axes([.9, .2, .02, .55]), extend='max')
        target = Path(output)
        temporary = target.with_name(target.name + '.tmp.png')
        fig.savefig(temporary, dpi=155)
        os.replace(temporary, target)
    finally:
        plt.close(fig)


def _run_chart(result_path, key, output_path, log_path):
    command = [sys.executable, str(Path(__file__).resolve()), '--render-model', key,
               '--result-json', str(result_path), '--output', str(output_path)]
    with Path(log_path).open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=45,
            env={**os.environ, 'MPLBACKEND': 'Agg'},
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    if completed.returncode != 0 or not Path(output_path).is_file():
        raise RuntimeError('isolated Agg renderer failed, exit=' + str(completed.returncode))


def _draw_svg(result, key, output):
    """Dependency-free deterministic chart fallback after a logged Agg failure."""
    rows = _model_rows(result, key)
    ps = sorted({r['prompt_tokens'] for r in rows})
    outs = sorted({r['output_tokens'] for r in rows})
    cs = sorted({r['parallel'] for r in rows})
    width, height = 360 * len(cs), 810
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" fill="#222">',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-size="20">{html.escape(MODEL_NAMES.get(key,key))}</text>',
        f'<text x="{width/2}" y="48" text-anchor="middle" font-size="13">Maximum request-rank deviation (%); 3 repeats; initial screen only</text>']
    for mi, metric in enumerate(METRICS):
        for ci, concurrency in enumerate(cs):
            x0, y0 = ci * 360 + 72, mi * 236 + 105
            parts.append(f'<text x="{x0+125}" y="{y0-22}" text-anchor="middle" font-size="16">{metric.upper()} | concurrency {concurrency}</text>')
            cw, ch = 250 / len(outs), 150 / len(ps)
            for i, prompt in enumerate(ps):
                parts.append(f'<text x="{x0-8}" y="{y0+(i+.5)*ch+4}" text-anchor="end" font-size="12">{prompt}</text>')
                for j, out in enumerate(outs):
                    value = _metric_value(rows, metric, prompt, out, concurrency)
                    fraction = min(1., max(0., value / 20)) if value is not None else None
                    color = '#d0d0d0' if fraction is None else '#%02x%02x%02x' % (255, int(250-180*fraction), int(190-155*fraction))
                    label = 'missing' if value is None else f'{value:.2f}%'
                    parts.append(f'<rect x="{x0+j*cw}" y="{y0+i*ch}" width="{cw}" height="{ch}" fill="{color}" stroke="white"/>')
                    parts.append(f'<text x="{x0+(j+.5)*cw}" y="{y0+(i+.5)*ch+4}" text-anchor="middle" font-size="13">{label}</text>')
            for j, out in enumerate(outs):
                parts.append(f'<text x="{x0+(j+.5)*cw}" y="{y0+170}" text-anchor="middle" font-size="12">{out}</text>')
            parts.append(f'<text x="{x0+125}" y="{y0+190}" text-anchor="middle" font-size="12">Output tokens (rows: input tokens)</text>')
    parts.append(f'<text x="{width/2}" y="797" text-anchor="middle" font-size="12">Gray = incomplete/unverified; yellow to red = 0 to 20% or higher</text></g></svg>')
    _atomic_text(output, '\n'.join(parts))


def render(result, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = {'schema': 'native-grid-report-render/v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'writing_tables', 'images': {}, 'errors': []}
    def checkpoint():
        _atomic_text(output / 'render_status.json', json.dumps(state, ensure_ascii=False, indent=2))
    checkpoint()
    # All usable textual artifacts exist before importing/launching plotting code.
    _atomic_text(output / 'report.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    metrics_fields = ('native_median_ms','batch_worst_abs_pct','rank_worst_abs_pct','rank_p90_abs_pct','rank_worst_abs_ms','rank_fraction_within_5pct','batch_sample_cv_pct')
    columns = ['id','model_key','prompt_tokens','output_tokens','parallel','status','source_id','source_freeze_sha256','captured_batches','planned_batches','observed_all_metrics_within_5pct'] + [f'{m}_{k}' for m in METRICS for k in metrics_fields]
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=columns)
    writer.writeheader()
    for cell in result['cells']:
        row = {key: cell.get(key) for key in columns if key in cell}
        source = cell.get('source_per_cell', {})
        row.update(source_id=source.get('source_id'), source_freeze_sha256=source.get('freeze_ref', {}).get('sha256'))
        row.update({f'{m}_{k}':cell['metrics'][m].get(k) for m in METRICS for k in metrics_fields})
        writer.writerow(row)
    _atomic_text(output / 'cells.csv', table.getvalue(), encoding='utf-8-sig')
    lines = _markdown(result)
    _atomic_text(output / 'report.md', '\n'.join(lines) + '\n')
    _atomic_text(output / 'report.html', _html(result, {}))
    state['status'] = 'rendering_charts'
    checkpoint()
    for key, name in MODEL_NAMES.items():
        if not _model_rows(result, key):
            continue
        state['current_model'] = key
        checkpoint()
        image, log = key + '_variability.png', output / (key + '_render.log')
        try:
            _run_chart(output / 'report.json', key, output / image, log)
        except Exception as exc:
            state['errors'].append({'model_key': key, 'stage': 'isolated_matplotlib_Agg', 'error': repr(exc),
                'traceback': traceback.format_exc(), 'log_ref': reference(log) if log.is_file() else None})
            image = key + '_variability.svg'
            try:
                _draw_svg(result, key, output / image)
            except Exception as fallback_exc:
                state['errors'].append({'model_key': key, 'stage': 'svg_fallback', 'error': repr(fallback_exc),
                                        'traceback': traceback.format_exc()})
                checkpoint()
                continue
        state['images'][key] = image
        lines += [f'## {name}', '', f'![{name}]({image})', '']
        _atomic_text(output / 'report.md', '\n'.join(lines) + '\n')
        _atomic_text(output / 'report.html', _html(result, state['images']))
        checkpoint()
    state.pop('current_model', None)
    state['status'] = 'complete' if not state['errors'] else 'complete_with_render_warnings'
    state['finished_utc'] = datetime.now(timezone.utc).isoformat()
    if state['errors']:
        lines += ['', '绘图依赖异常已记录在 render_status.json 和各模型 render.log；可用的备用 SVG 图已直接嵌入。', '']
        _atomic_text(output / 'report.md', '\n'.join(lines))
    checkpoint()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--supplement', type=Path, help='new child campaign containing only original incomplete cells')
    parser.add_argument('--lineage', type=Path, help='explicit supplement lineage; defaults to child/lineage.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-model', choices=list(MODEL_NAMES), help=argparse.SUPPRESS)
    parser.add_argument('--result-json', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render_model:
        _draw_matplotlib(_read_json(args.result_json), args.render_model, args.output)
        return 0
    if args.campaign is None:
        parser.error('--campaign is required')
    try:
        result = build_report(args.campaign, supplement=args.supplement, lineage=args.lineage)
        state = render(result, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        _atomic_text(args.output / 'report_failure.json', json.dumps({'error': repr(exc),
            'traceback': traceback.format_exc(), 'created_utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2))
        raise
    print(json.dumps({**{k: result[k] for k in ('planned_cells','complete_cells','observed_all_metrics_within_5pct_cells','execution_status','end_verified')},
        'render_status': state['status'], 'render_warnings': len(state['errors'])}, ensure_ascii=False), flush=True)
    return 1 if result['verification_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
