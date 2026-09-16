"""Independent host-only collector regression. No GPU/native process is started.

These tests state the intended rejection behavior. Current failures are actionable
review findings, not latency data or modifications to the collector itself.
"""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import pytest
HERE=Path(__file__).resolve().parent
COLLECTOR=HERE.parent/'collection'
sys.path.insert(0,str(COLLECTOR))
import common,extract,probe_adapter
spec=importlib.util.spec_from_file_location('review_fixture_source',COLLECTOR/'test_collection.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def test_completed_stage_rejects_absent_required_artifacts(monkeypatch,tmp_path):
    anchor={'path':str(tmp_path/'clock.json'),'sha256':'b'*64,'bytes':10}
    receipt={'clock_control_binding':{'receipt_ref':anchor},'external_approved_sha256':'a'*64,
             'child_process_exited':True,'status':'completed','returncode':0,
             'freeze_after':{'passed':True},'artifacts':[]}
    def fake_load(path):
        name=Path(path).name
        if name=='complete.json':return receipt
        if name=='protocol.json':return {'gpu_identity':{'uuid':'device0'}}
        if name=='clock-control-binding.json':return {'receipt_ref':anchor}
        raise AssertionError('unexpected read '+str(path))
    monkeypatch.setattr(extract,'load',fake_load)
    monkeypatch.setattr(extract,'verify_clock_receipt',lambda *args:{'receipt_ref':anchor})
    with pytest.raises((ValueError,KeyError,TypeError,OSError)):
        extract.stage_complete(tmp_path/'profile','a'*64)


def test_raw_audit_rejects_missing_process_and_pair_identity(tmp_path):
    config=probe_adapter.protocol_document()['configs'][1]
    records=fixtures.raw_fixture(config)
    # Frozen C++ actually emits these fields; a receipt with missing identity
    # cannot establish independent processes, paired argv, or originating time.
    for key in ('pid','argv','qpc_start','utc_start'):
        records[0].pop(key,None)
    records[1].pop('pair_id',None)
    path=tmp_path/'missing_identity.jsonl';path.write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
    try:result=probe_adapter.audit_raw(probe_adapter.read_raw(path),config,{})
    except ValueError:return
    assert result['valid_raw'] is False, result


def test_trace_rejects_different_native_process_identity():
    config=probe_adapter.protocol_document()['configs'][1]
    raw,app=fixtures.trace_fixture(config)
    # Fixture target globalPid is 1<<24. The native app claims another process;
    # identical labels/configuration do not authorize cross-process pairing.
    app['header']={'pid':22222}
    try:result=extract.correlate(raw,app,config)
    except ValueError:return
    assert result['all36_chain_complete'] is False, {
        'app_pid':app['header']['pid'],
        'trace_globalPid':raw['tables']['CUPTI_ACTIVITY_KIND_KERNEL'][0]['globalPid'],
        'accepted':result['all36_chain_complete']}


def test_graph_one_API_many_kernels_stays_counted_once():
    config=probe_adapter.protocol_document()['configs'][1]
    raw,app=fixtures.trace_fixture(config);app['header']={'pid':1}
    result=extract.correlate(raw,app,config)
    assert result['all36_chain_complete'] is True
    assert result['actual_kernels_mapped']==288
    assert all(c['actual_kernel_count']==8 for c in result['calls'])
    assert all(sum(c['exclusive_partition_ns'].values())==c['nvtx_full_call_duration_ns'] for c in result['calls'])


def test_missing_child_never_backfilled_from_planned_nodes():
    config=probe_adapter.protocol_document()['configs'][1]
    raw,app=fixtures.trace_fixture(config);app['header']={'pid':1}
    raw['tables']['CUPTI_ACTIVITY_KIND_KERNEL'].pop(0)
    result=extract.correlate(raw,app,config)
    assert result['all36_chain_complete'] is False
    assert result['calls'][0]['actual_kernel_count']==7


def test_incomplete_profile_keeps_failed_pair_in_denominator():
    import quality
    result=quality.quality({'id':'review'},[{'pair':i} for i in range(3)])
    assert result['pairs_observed']==result['pairs_required']==3
    assert result['measurement_cost_eligible'] is False
    assert result['calibration_eligible'] is False and result['fit_performed'] is False
