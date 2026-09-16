"""Freeze diagnostic inputs using metadata/stat only; never read GGUF content."""
from pathlib import Path
import json
import sys
import diagnostic as d
import dual_hash as dh

HERE=Path(__file__).resolve().parent
R27=HERE.parent
TARGET=R27.parent.parent/'gpu_extension/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.readonly.gguf'
EXPECTED_SIZE=14865116128
EXPECTED_SHA='157479b0083662c7c9cfd8754cae24ab9cc1abbd4def91d49758512c90f1e406'


def prepare():
    budget=d.ReadBudget()
    before=dh.stat_record(TARGET.stat())
    if before['st_size']!=EXPECTED_SIZE:raise ValueError('Target size differs; no target content read')
    campaign_files=[R27/'controls.json',R27/'freeze_receipt.json',R27/'protocol.json',R27/'off/freeze.json',R27/'on/freeze.json']
    refs=[d.ref(p,budget) for p in campaign_files]
    arms={}
    for arm in ('off','on'):
        path=R27/arm/'freeze.json';f=d.document(path,budget)
        ids=[c['cell_id'] for c in f['cells']]
        if len(ids)!=131 or len(set(ids))!=131:raise ValueError('Frozen131 required')
        arms[arm]={'freeze_ref':next(r for r in refs if Path(r['path'])==path.resolve()),
                   'cell_ids':ids,'source_sha256':f['source']['sha256'],'selection_sha256':f['selection_sha256']}
        # Expected digest must occur in each frozen arm's metadata, never inferred from a new scan.
        expected={'path':str(TARGET.resolve()),'bytes':EXPECTED_SIZE,'sha256':EXPECTED_SHA}
        refs_for_target=[d.norm(c['static_inputs']['prediction_model_ref']) for c in f['cells']
                         if c.get('static_inputs',{}).get('prediction_model_ref',{}).get('path')
                         and Path(c['static_inputs']['prediction_model_ref']['path']).resolve()==TARGET.resolve()]
        if not refs_for_target or any(r!=expected for r in refs_for_target):raise ValueError('Target path/size/SHA not bound to both frozen arms')
    p={'schema':'dual-hash-diagnostic-protocol/v1','created_utc':dh.now(),'status':'prepared_not_executed',
       'target':{'path':str(TARGET.resolve()),'bytes':EXPECTED_SIZE,'sha256':EXPECTED_SHA},
       'target_stat_at_prepare':before,'preparation_model_content_bytes_read':0,
       'chunk_bytes':dh.MAX_CHUNK,'passes':['serial_01','parallel_01','parallel_02','parallel_03','parallel_04'],
       'parallel_workers':4,'worker_kind':'independent Windows spawned Python processes; independent opens',
       'phase_order':['one serial memory-control plus file scan','four concurrent memory-controls plus independent file scans'],
       'memory_control':{'vector':'NIST million ASCII a','bytes':1000000,
          'expected_sha256':'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0',
          'single_large_update_per_engine':True,'repetitions_per_worker':1,'workers_total':5,'disk_input_bytes':0},
       'maximum_model_read_bytes':5*EXPECTED_SIZE,'maximum_metadata_read_bytes':d.MAX_METADATA_BYTES,
       'maximum_model_open_count':5,'automatic_retries':0,'cache_eviction':False,'file_mutation':False,
       'global_environment_or_CPU_BIOS_driver_changes':False,'model_latency_measured':False,'cost_parameters':0,
       'expected_identity_mismatch_rejected':True,'algorithm_disagreement_rejected':True,
       'later_agreement_can_repair_prior_failure':False,'campaign_refs':refs,'arms':arms,
       'observed_prior_bad_full_sha256':['86c0ee73142d34ff4e9e35251b0d4596db166c2463df6058f86e051a919e8485',
           '9df5e6ab054d2109059acf24092d8d4a469871e9ee9546bafd1ab3e6bd3b25a4'],
       'execution_gates':['R27 full262 terminal barrier with preserved failures','project idle before/between/after phases',
                          'frozen protocol/code/crypto DLL identities','two independent known-vector implementations'],
       'interpretation_limits':['Same chunks constrain algorithm comparison, not read integrity.',
          'Agreement between engines on wrong content is still rejection.',
          'Serial/concurrent differences are diagnostic only: cache, scheduling, reading and memory remain confounded.',
          'No inferred faulty hardware or storage from this experiment alone.',
          'Buffered OS cache is not bypassed; logical read bytes are bounded, physical disk bytes unknown.',
          'Terminal receipt checks inspect identity/status only, not model performance or scores.']}
    if (HERE/'protocol.json').exists():
        previous=d.document(HERE/'protocol.json',budget)
        for key in p:
            if key not in ('created_utc','target_stat_at_prepare') and previous.get(key)!=p[key]:
                raise ValueError('Existing protocol differs; never overwrite it')
        if not all(previous['target_stat_at_prepare'].get(k)==before.get(k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('Prepared target stat changed')
    else:
        dh.write_new(HERE/'protocol.json',p)
    source_files=['dual_hash.py','diagnostic.py','prepare.py','test_identity_diagnostic.py','README.md','protocol.json']
    small=dh.self_test()
    m={'schema':'dual-hash-preparation/v1','created_utc':dh.now(),'files':[d.ref(HERE/n,budget) for n in source_files],
       'runtime_files':[d.ref(x,budget) for x in d.runtime_files()],'crypto_environment':d.crypto_environment(),
       'revision':'0003','supersedes':d.ref(HERE/'preparation_manifest.0002.json',budget),
       'revision_fixes':['cross-API stat excludes ctime but includes birthtime; same-API retains ctime',
                         'exact GPU/relative runner/spawn-parent idle gate',
                         'CNG cleanup status checked and original exceptions preserved'],
       'small_known_vector_self_test':small,'metadata_read_bytes':budget.used,'model_content_bytes_read':0,
       'target_stat_after_prepare':dh.stat_record(TARGET.stat())}
    if not dh.same_file_stat(before,m['target_stat_after_prepare']):raise ValueError('Target stat changed during preparation')
    dh.write_new(HERE/'preparation_manifest.0003.json',m)
    print(json.dumps({'status':'prepared','protocol':str(HERE/'protocol.json'),
       'maximum_model_read_bytes':p['maximum_model_read_bytes'],'model_content_bytes_read':0}))


if __name__=='__main__':prepare()
