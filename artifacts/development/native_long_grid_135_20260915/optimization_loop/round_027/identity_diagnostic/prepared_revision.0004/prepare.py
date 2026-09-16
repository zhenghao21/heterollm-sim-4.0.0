"""Prepare revision0004: arithmetically closed metadata budget; no GGUF reads."""
from copy import deepcopy
from pathlib import Path
import json
import diagnostic as d
import dual_hash as dh

HERE=Path(__file__).resolve().parent
R27=HERE.parent
TARGET=R27.parent.parent/'gpu_extension/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.readonly.gguf'
EXPECTED_SIZE=14865116128
EXPECTED_SHA='157479b0083662c7c9cfd8754cae24ab9cc1abbd4def91d49758512c90f1e406'


def prepare():
    protocol_path=HERE/'protocol.0004.json';manifest_path=HERE/'preparation_manifest.0004.json'
    if protocol_path.exists() or manifest_path.exists():raise ValueError('Revision0004 already prepared; no overwrite or automatic new attempt')
    budget=d.ReadBudget();old=d.document(HERE/'protocol.json',budget)
    if old['target']!={'path':str(TARGET.resolve()),'bytes':EXPECTED_SIZE,'sha256':EXPECTED_SHA}:
        raise ValueError('Old frozen target differs')
    before=dh.stat_record(TARGET.stat())
    if before['st_size']!=EXPECTED_SIZE:raise ValueError('Target stat size differs; no target content read')
    for item in old['campaign_refs']:d.check_ref(item,budget)
    barrier_path=R27/'predictions_complete.json';barrier=d.document(barrier_path,budget)
    if (barrier.get('schema')!='r27-full262-terminal-barrier/v1' or barrier.get('terminal_count')!=262
        or set(barrier.get('arms',{}))!={'off','on'} or barrier.get('failures_preserved') is not True):
        raise ValueError('Full262 immutable terminal snapshot required')
    controls=next(r for r in old['campaign_refs'] if Path(r['path']).name=='controls.json')
    if d.norm(barrier['controls_ref'])!=controls:raise ValueError('Terminal controls differ')
    predictions=[]
    for arm in ('off','on'):
        actual=barrier['arms'][arm];locked=old['arms'][arm]
        if d.norm(actual['freeze_ref'])!=locked['freeze_ref'] or set(actual['prediction_refs'])!=set(locked['cell_ids']) or len(actual['prediction_refs'])!=131:
            raise ValueError('Frozen131 arm differs')
        for cell_id,item in actual['prediction_refs'].items():
            expected=(R27/arm/'predictions'/(cell_id+'.prediction.json')).resolve()
            if Path(item['path']).resolve()!=expected or expected.stat().st_size!=item['bytes']:
                raise ValueError('Frozen prediction path/length differs')
            if item['bytes']>128*1024**2:raise ValueError('Prediction exceeds existing per-file metadata cap')
            predictions.append(item)
    prior_dir=HERE/'runs/read_comparison_0001'
    prior=d.document(prior_dir/'finish.json',budget)
    serial=d.document(prior_dir/'serial_01/finish.json',budget)
    if (prior.get('status')!='rejected' or prior.get('children')!=[] or set(prior.get('pass_receipts',{}))!={'serial_01'}
        or prior.get('actual_model_read_bytes_reported')!=EXPECTED_SIZE or prior.get('model_read_upper_bound_bytes')!=EXPECTED_SIZE
        or not any('Metadata byte budget exhausted' in value for value in prior.get('errors',[]))):
        raise ValueError('Expected one-pass budget-rejected attempt differs; not relabeling history')
    if (serial.get('status')!='agreed_expected_identity' or serial.get('bytes_read')!=EXPECTED_SIZE
        or serial.get('full_openssl_sha256')!=EXPECTED_SHA or serial.get('full_cng_sha256')!=EXPECTED_SHA
        or serial.get('chunk_digest_disagreements')!=[] or serial.get('errors')!=[]):
        raise ValueError('Prior serial record differs; new batch is not an identity repair')
    history_files=[HERE/'preparation_manifest.0003.json',prior_dir/'start.json',prior_dir/'finish.json',
        prior_dir/'serial_01/finish.json',prior_dir/'serial_01/chunks.jsonl',prior_dir/'serial_01.memory_control.json']
    history_refs=[d.ref(path,budget) for path in history_files]
    p=deepcopy(old);p.update(schema='dual-hash-diagnostic-protocol/v2',created_utc=dh.now(),
        status='prepared_not_executed',revision='0004',protocol_filename=protocol_path.name,
        run_directory='read_comparison_0002',target_stat_at_prepare=before,
        preparation_model_content_bytes_read=0,preparation_prediction_content_bytes_read=0,
        frozen_terminal_snapshot={'barrier_ref':d.ref(barrier_path,budget),'arms':barrier['arms'],
            'prediction_ref_count':len(predictions),'prediction_ref_bytes':sum(r['bytes'] for r in predictions)},
        prior_attempt={'status':'rejected','run_directory':'read_comparison_0001',
            'reason':'auxiliary_metadata_budget_inconsistent_with_three_required_full262_checks',
            'evidence_refs':history_refs,'reported_model_read_bytes':EXPECTED_SIZE,'reported_passes':1,
            'serial_evidence_status':'agreed_expected_identity_but_old_attempt_still_rejected',
            'serial_reused_in_new_batch':False,'old_failure_reclassified':False},
        maximum_model_read_bytes=5*EXPECTED_SIZE,maximum_model_open_count=5,
        cumulative_model_read_upper_bound_bytes=6*EXPECTED_SIZE,
        budget_revision_changes_identity_admission=False,automatic_retries=0)
    p['execution_gates']+=['preflight arithmetic covers all three full262 checks and all declared auxiliary reads before model content is touched']
    p['interpretation_limits']+=['The failed revision0003 used one model pass. This separately frozen batch may read five new passes; cumulative logical bound is six, and the failed batch remains rejected.',
        'Auxiliary length prechecks allocate a bounded read budget only; every required reference is still fully read and hashed at all three gates.']
    source_names=['dual_hash.py','diagnostic.py','prepare.py','test_identity_diagnostic.py','README.md']
    source_refs=[d.ref(HERE/name,budget) for name in source_names]
    runtime_refs=[d.ref(path,budget) for path in d.runtime_files()]
    placeholder={'path':str(protocol_path.resolve()),'bytes':0,'sha256':'0'*64}
    plan=d.planned_metadata_budget(p,{'files':source_refs+[placeholder],'runtime_files':runtime_refs})
    p['metadata_budget_plan']=plan;p['maximum_metadata_read_bytes']=plan['declared_metadata_read_limit_bytes']
    encoded=(json.dumps(p,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
    if len(encoded)>d.PROTOCOL_MAX_BYTES:raise ValueError('Prepared protocol exceeds declared cyclic-reference cap')
    dh.write_new(protocol_path,p)
    protocol_ref=d.ref(protocol_path,budget)
    m={'schema':'dual-hash-preparation/v1','created_utc':dh.now(),'revision':'0004',
        'protocol_ref':protocol_ref,'files':source_refs+[protocol_ref],'runtime_files':runtime_refs,
        'crypto_environment':d.crypto_environment(),'supersedes':d.ref(HERE/'preparation_manifest.0003.json',budget),
        'revision_fixes':['predeclared exact-reference length and read-count metadata budget',
            'three terminal and program/runtime full byte checks preserved',
            'new exclusive0002 five-pass batch; old rejected attempt and one-pass cumulative budget preserved'],
        'metadata_budget_plan':plan,'small_known_vector_self_test':dh.self_test(),
        'metadata_read_bytes':budget.used,'model_content_bytes_read':0,'prediction_content_bytes_read':0,
        'target_stat_after_prepare':dh.stat_record(TARGET.stat())}
    if not dh.same_file_stat(before,m['target_stat_after_prepare']):raise ValueError('Target stat changed during preparation')
    if len((json.dumps(m,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))>d.MANIFEST_MAX_BYTES:
        raise ValueError('Prepared manifest exceeds declared cyclic-reference cap')
    dh.write_new(manifest_path,m)
    check_budget=d.ReadBudget();prepared,_=d.verify_prepared(manifest_path,check_budget)
    checked=d.configure_execution_budget(prepared,manifest_path,check_budget)
    if checked!=plan:raise ValueError('Post-save byte budget preflight differs')
    print(json.dumps({'status':'revision0004_prepared_budget_precheck_passed','manifest':str(manifest_path),
        'required_metadata_upper_bound_bytes':plan['required_metadata_read_upper_bound_bytes'],
        'declared_metadata_limit_bytes':plan['declared_metadata_read_limit_bytes'],
        'new_batch_model_read_upper_bound_bytes':5*EXPECTED_SIZE,'prior_model_bytes':EXPECTED_SIZE,
        'cumulative_model_read_upper_bound_bytes':6*EXPECTED_SIZE,'model_content_bytes_read':0,'prediction_content_bytes_read':0}))

if __name__=='__main__':prepare()
