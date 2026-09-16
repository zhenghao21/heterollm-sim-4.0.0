"""Retained warmup adapter structure tests; never run native or LLM prediction."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
from dataclasses import replace
from types import SimpleNamespace
import pytest

from tools import predict_stable_native_dataset as a
from tests.test_predict_stable_native_dataset import fixture, document, seal
from tests.test_nonflash_kv_view import scenario as ordinary_scenario, SOURCE_SHA
from heterollm_sim.retained_kv_state import RetainedKVState, KEY, ENABLED
from heterollm_sim.serving import compile_serving_plan
from heterollm_sim import planner
from heterollm_sim.ir import RequestSpec

MAIN=Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")
ORIGINAL_SOURCE_FREEZE=a.source_freeze


def gguf(path,arch='llama',extra=None):
    def string(value):
        raw=value.encode();return struct.pack('<Q',len(raw))+raw
    fields={'general.architecture':arch,arch+'.block_count':2,arch+'.attention.head_count':4,arch+'.attention.head_count_kv':4,**(extra or {})}
    payload=b'GGUF'+struct.pack('<IQQ',3,0,len(fields))
    for key,value in fields.items():
        payload+=string(key)+(struct.pack('<I',8)+string(value) if isinstance(value,str) else struct.pack('<Ii',5,value))
    path.write_bytes(payload)
    return a.grid.file_ref(path)


@pytest.fixture
def data(tmp_path,monkeypatch):
    selection_path,selection,row,calls=fixture(tmp_path,monkeypatch)
    row['model_ref']=gguf(Path(row['config']['model']))
    plan={**row['config'],'key':'unit-cell','block':0,'warmup_batches':2,'measure_batches':3,'process_blocks':1,'cache_ram_mib':0,'kv_unified_per_slot':2048}
    runtime={'module_identity_sha256':'a'*64,'process_identity':{'pid':123,'creation_marker':'synthetic'},'actual_modules':row['native_runtime_refs']}
    raw={'key':'unit-cell','status':'complete','runtime_before':runtime,'runtime_after':runtime,
        'payload':{'cache_prompt':False,'n_predict':2},'actual_argv':['-np','2','-c','4096','-b','64','-ub','64','-fa','off','-kvu','--cache-ram','0','--spec-type','none','-ctk','f16','-ctv','f16'],
        'warmup':[{'status':'complete','phase':'warmup','process_block':0,'requests':[
            {'status':'measured','slot':slot,'response':{'truncated':False,'timings':{'prompt_n':4,'predicted_n':2,'cache_n':0,'prompt_ms':987654321}}}
            for slot in (0,1)]} for _ in range(2)],
        'measured_requests':[{'forbidden_target_latency_ms':987654321}]}
    raw_ref=document(tmp_path/'warmup.json',raw)
    misc=document(tmp_path/'source-proof.json',{'static':True})
    row['plans']=[plan];row['evidence_index']=[{'raw_ref':raw_ref,'receipt_ref':misc,'runtime_baseline_ref':misc,
        'config_sha256':hashlib.sha256(json.dumps(plan,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()}]
    row['source_per_cell']={'freeze_ref':misc};selection['selector_source_ref']=misc
    document(selection_path,seal(selection))
    scene=ordinary_scenario(2)
    view=copy.deepcopy(scene.workload.metadata['llama_cpp_nonflash_kv_view'])
    view_ref=document(tmp_path/'nonflash.json',view)
    nonflash={'contract_ref':view_ref,'contract':view,'cells':{row['cell_id']:view},'evidence_refs':[view_ref]}
    extractor=tmp_path/'original_extractor.py';extractor.write_bytes(a.RETAINED_WARMUP_EXTRACTOR.read_bytes())
    binding=a.freeze_retained_warmup_binding(selection_path,[row],tmp_path/'frozen',nonflash,extractor_path=extractor)
    inputs=a.static_inputs(row,selection,tmp_path,nonflash_kv_view=nonflash,retained_warmup=binding)
    requests=tuple(RequestSpec(request_id=f'request-{i:04d}',arrival_ns=0.0,prompt_tokens=4,output_tokens=2) for i in range(2))
    scene=replace(scene,placement=replace(scene.placement,kv_policy=replace(scene.placement.kv_policy,offload_ratio=0.0)),
        workload=replace(scene.workload,requests=requests,request_count=2,scheduler=replace(scene.workload.scheduler,preemption_enabled=False)))
    return dict(binding=binding,inputs=inputs,scene=scene,gguf=SimpleNamespace(sha256=row['model_ref']['sha256'],architecture='llama'),
        row=row,selection=selection,selection_path=selection_path,raw=raw,raw_path=tmp_path/'warmup.json',nonflash=nonflash,extractor=extractor)


def test_freeze_evidence_has_no_timing_and_copied_extractor_is_sufficient(data):
    proof=data['inputs']['retained_kv_warmup_evidence']
    assert proof['status']=='conditional' and proof['contract']['slots'][0]['prompt_tokens']==4
    text=json.dumps(proof)
    assert '987654321' not in text and 'prompt_ms' not in text and 'forbidden_target_latency_ms' not in text
    assert 'attention_family_scope' not in text and 'selection_payload_sha256' not in text
    assert proof['selection_ref']['sha256']==a.grid.file_ref(data['selection_path'])['sha256']
    data['extractor'].unlink()
    result=a.apply_retained_warmup_static_contract(data['scene'],data['inputs'],gguf=data['gguf'])
    ledger=RetainedKVState.from_plan(compile_serving_plan(result),tuple(planner._execution_layers(result)))
    assert ledger.rows=={0:5,1:5}
    assert result.workload.metadata[ENABLED] is True
    assert proof['post_warmup_native_lifecycle_proven'] is False
    assert proof['contract']['evidence_sha256']==a.grid.stable_hash({key:value for key,value in proof.items() if key!='contract'})


@pytest.mark.parametrize('change',['false','none','contract','slot','configuration','source_sha','selection_payload'])
def test_worker_rejects_forged_switch_contract_and_identity(data,change):
    inputs=copy.deepcopy(data['inputs'])
    if change=='false':inputs['retained_kv_warmup_state']=False
    if change=='none':inputs['retained_kv_warmup_evidence']=None
    if change=='contract':inputs['retained_kv_warmup_contract']=None
    if change=='slot':inputs['retained_kv_warmup_evidence']['contract']['slots'][0]['slot_id']=99
    if change=='configuration':inputs['nonflash_kv_view_contract']['configuration']['parallel']=1
    if change=='source_sha':inputs['retained_kv_warmup_evidence']['nonflash_contract']['source_sha256']['src/llama-kv-cache.cpp']='b'*64
    if change=='selection_payload':inputs['retained_kv_warmup_evidence']['selection_ref']['sha256']=data['selection']['payload_sha256']
    with pytest.raises(ValueError):a.apply_retained_warmup_static_contract(data['scene'],inputs,gguf=data['gguf'])


def test_raw_mutation_is_rejected_without_using_timing(data):
    data['raw']['warmup'][1]['requests'][0]['response']['timings']['prompt_n']=999
    document(data['raw_path'],data['raw'])
    with pytest.raises(ValueError):a.apply_retained_warmup_static_contract(data['scene'],data['inputs'],gguf=data['gguf'])


def test_actual_gguf_scope_overrides_model_key_labels(data,tmp_path):
    model=tmp_path/'hybrid.gguf';ref=gguf(model,'qwen35')
    scope=a.read_retained_gguf_scope(ref)
    assert scope['status']=='uncovered'
    proof=data['inputs']['retained_kv_warmup_evidence']
    hybrid=a.derive_retained_cell_proof(proof['warmup'],selection_ref=proof['selection_ref'],extractor_ref=proof['extractor_ref'],
        nonflash_contract=proof['nonflash_contract'],nonflash_ref=proof['nonflash_contract_ref'],model_scope=scope,native_refs=data['row']['native_runtime_refs'])
    assert hybrid['status']=='uncovered' and hybrid['contract'] is None
    assert 'model_key' not in hybrid
    sliding=gguf(tmp_path/'swa.gguf','llama',{'llama.attention.sliding_window':128})
    assert a.read_retained_gguf_scope(sliding)['status']=='uncovered'


def test_missing_warmup_and_unknown_count_are_uncovered(data):
    proof=data['inputs']['retained_kv_warmup_evidence'];warmup=copy.deepcopy(proof['warmup'])
    warmup['qualification']['warmup_record_and_static_protocol']='not_qualified'
    warmup['qualification']['missing_or_failed']=['warmup_batch_count_not_two']
    warmup['warmup_batches']=[]
    result=a.derive_retained_cell_proof(warmup,selection_ref=proof['selection_ref'],extractor_ref=proof['extractor_ref'],
        nonflash_contract=proof['nonflash_contract'],nonflash_ref=proof['nonflash_contract_ref'],model_scope=proof['model_scope'],native_refs=data['row']['native_runtime_refs'])
    assert result['status']=='uncovered' and result['contract'] is None


def test_default_off_keeps_static_identity_and_freeze_fields_absent(tmp_path,monkeypatch):
    selection_path,selection,row,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(a,'freeze_retained_warmup_binding',lambda *args,**kwargs:pytest.fail('off must not read warmup'))
    baseline=a.static_inputs(row,selection,tmp_path)
    frozen=a.freeze_selection(selection_path,tmp_path/'off',data_root=tmp_path,retained_kv_warmup_state=False)
    assert frozen['cells'][0]['static_inputs']==baseline and 'retained_kv_warmup' not in frozen
    sentinel=object()
    assert a.apply_retained_warmup_static_contract(sentinel,{},gguf=None) is sentinel


def test_initial_freeze_and_resume_protect_frozen_state(data,tmp_path,monkeypatch):
    monkeypatch.setattr(a,'verified_nonflash_kv_view_contract',lambda *args,**kwargs:data['nonflash'])
    frozen=a.freeze_selection(data['selection_path'],tmp_path/'on',data_root=tmp_path,
        nonflash_kv_view_source_contract_path=tmp_path/'nonflash.json',retained_kv_warmup_state=True,
        retained_kv_warmup_extractor_path=data['extractor'])
    a.verify_freeze_references(frozen)
    assert frozen['retained_kv_warmup_state'] is True
    assert frozen['retained_kv_warmup']['extractor_ref'] in frozen['source']['files']
    changed=copy.deepcopy(frozen)
    changed['retained_kv_warmup']['cells'][data['row']['cell_id']]['warmup']['initial_retained_slot_template']['retained_tokens_per_slot']=999
    with pytest.raises(ValueError,match='resume proof'):a.verify_retained_warmup_freeze(changed)
    frozen['cells'][0]['static_inputs']['retained_kv_warmup_state']=False
    with pytest.raises(ValueError,match='cell proof'):a.verify_retained_warmup_freeze(frozen)
    for option in ('--retained-kv-warmup-state','--no-retained-kv-warmup-state'):
        with pytest.raises(SystemExit):a.main(['--output',str(tmp_path),'--resume',option])
    with pytest.raises(ValueError,match='nonflash'):a.freeze_selection('unused.json',tmp_path/'bad',retained_kv_warmup_state=True)


def test_request_slot_mapping_never_uses_native_start_finish(data):
    scene=data['scene'];requests=list(scene.workload.requests);requests[0]=replace(requests[0],arrival_ns=100)
    scene=replace(scene,workload=replace(scene.workload,requests=tuple(requests)))
    with pytest.raises(ValueError,match='homogeneous simultaneous'):a.apply_retained_warmup_static_contract(scene,data['inputs'],gguf=data['gguf'])


def test_real131_header_and_warmup_prepare_62_ordinary_69_uncovered(tmp_path):
    if os.environ.get('R23_TEST_REAL_STATIC')!='1':pytest.skip('explicit real131 static preparation opt-in')
    path=MAIN/'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_022/current/freeze.json'
    frozen=json.loads(path.read_text(encoding='utf-8'))
    selection_path=Path(frozen['selection_ref']['path'])
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    rows=a.selected_rows(selection)
    binding=a.freeze_retained_warmup_binding(selection_path,rows,tmp_path/'real',frozen['nonflash_kv_view'],model_snapshot_map=frozen['model_snapshot_map'])
    assert binding['selection_ref']['sha256']=='cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5'
    assert binding['conditional_cell_count']==62 and binding['uncovered_cell_count']==69
    for proof in binding['cells'].values():
        if proof['status']=='conditional':
            assert proof['model_scope']['architecture'] in {'llama','qwen2'}
            assert proof['contract']['evidence_sha256'] and len(proof['contract']['slots'])==proof['contract']['configuration']['parallel']

            cfg=proof['contract']['configuration'];p=proof['warmup']['static_configuration']['prompt_tokens'];o=proof['warmup']['static_configuration']['output_tokens']
            case=ordinary_scenario(cfg['parallel'],architecture=proof['model_scope']['architecture']+'_decoder')
            requests=tuple(RequestSpec(request_id=ident,arrival_ns=0.0,prompt_tokens=p,output_tokens=o) for ident in proof['contract']['request_slots'])
            case=replace(case,placement=replace(case.placement,kv_policy=replace(case.placement.kv_policy,offload_ratio=0.0)),
                workload=replace(case.workload,requests=requests,request_count=len(requests),scheduler=replace(case.workload.scheduler,preemption_enabled=False),
                    metadata={**case.workload.metadata,'llama_cpp_nonflash_kv_view':proof['nonflash_contract'],
                        ENABLED:True,KEY:proof['contract'],'llama_cpp_retained_kv_identity':proof['contract']['identity']}))
            ledger=RetainedKVState.from_plan(compile_serving_plan(case),tuple(planner._execution_layers(case)))
            assert sum(ledger.rows.values())==cfg['parallel']*(p+o-1)
        else:assert proof['contract'] is None and any('GGUF_architecture' in reason for reason in proof['uncovered_reasons'])


def test_hybrid_uncovered_preserves_original_prediction_and_scorability(data,tmp_path,monkeypatch):
    """Use only fixture metrics; retain the existing numerical path and scoring."""
    row=data['row']
    row['model_ref']=gguf(Path(row['config']['model']),'qwen35')
    document(data['selection_path'],seal(data['selection']))
    monkeypatch.setattr(a,'verified_nonflash_kv_view_contract',lambda *args,**kwargs:data['nonflash'])
    monkeypatch.setattr(a.grid,'read_gguf_metadata',lambda path:SimpleNamespace(
        sha256=row['model_ref']['sha256'],architecture='qwen35',n_layer=64,
        metadata={'qwen35.block_count':65,'qwen35.nextn_predict_layers':1}))
    captured=[]
    original_fixture_run=a.grid.reporting.run_scenario
    def fixture_run(scene,**kwargs):
        captured.append(scene)
        return original_fixture_run(scene,**kwargs)
    monkeypatch.setattr(a.grid.reporting,'run_scenario',fixture_run)
    output=tmp_path/'hybrid-fallback'
    freeze=a.freeze_selection(data['selection_path'],output,data_root=tmp_path,
        nonflash_kv_view_source_contract_path=tmp_path/'nonflash.json',retained_kv_warmup_state=True,
        retained_kv_warmup_extractor_path=data['extractor'])
    inputs=freeze['cells'][0]['static_inputs']
    proof=inputs['retained_kv_warmup_evidence']
    assert freeze['cells'][0]['preparation_error'] is None
    assert proof['status']=='uncovered' and proof['contract'] is None
    baseline_inputs={key:value for key,value in inputs.items() if key not in (
        'retained_kv_warmup_state','retained_kv_warmup_evidence','retained_kv_warmup_contract')}
    baseline=a.predict_cell(baseline_inputs)
    retained=a.predict_cell(inputs)
    assert baseline['status']==retained['status']=='predicted'
    assert baseline['aggregate']==retained['aggregate']
    assert baseline['requests']==retained['requests']
    assert all(retained['aggregate'][metric]['median_ms'] is not None for metric in a.METRICS)
    original, fallback=captured
    qualification=fallback.workload.metadata['llama_cpp_retained_kv_warmup_qualification']
    assert qualification['status']=='uncovered'
    assert ENABLED not in fallback.workload.metadata and KEY not in fallback.workload.metadata
    remaining={key:value for key,value in fallback.workload.metadata.items() if key!='llama_cpp_retained_kv_warmup_qualification'}
    assert remaining==original.workload.metadata
    assert fallback.hardware==original.hardware and fallback.component_profiles==original.component_profiles
    declaration=next(row for row in retained['unsupported_dimensions'] if row['dimension']=='retained_kv_warmup_state')
    assert declaration['status']=='uncovered' and declaration['uncovered_reasons']
    # Exercise the independent scorer with synthetic fixture actuals only.
    # The retained qualification must not turn this complete cell into unscored.
    freeze_ref=a.grid.file_ref(output/'freeze.json')
    document(output/'predictions'/(inputs['cell_id']+'.prediction.json'),{**retained,'freeze_ref':freeze_ref})
    scored=a.score_predictions(output)
    assert scored['selected_denominator']==1
    assert all(scored['cells'][0]['metrics'][metric]['status']=='scored' for metric in a.METRICS)


def test_retained_off_on_share_exact_source_closure(data,tmp_path,monkeypatch):
    monkeypatch.setattr(a,'source_freeze',ORIGINAL_SOURCE_FREEZE)
    monkeypatch.setattr(a,'verified_nonflash_kv_view_contract',lambda *args,**kwargs:data['nonflash'])
    outputs={}
    for name,enabled in [('off',False),('on',True)]:
        outputs[name]=a.freeze_selection(data['selection_path'],tmp_path/name,data_root=tmp_path,
            nonflash_kv_view_source_contract_path=tmp_path/'nonflash.json',retained_kv_warmup_state=enabled)
    def closure(freeze):
        root=Path(freeze['source']['root'])
        return {Path(ref['path']).relative_to(root).as_posix():ref['sha256'] for ref in freeze['source']['files']}
    assert closure(outputs['off'])==closure(outputs['on'])
    assert closure(outputs['off'])['tools/retained_warmup_extractor.py']==a.RETAINED_WARMUP_EXTRACTOR_SHA256
    for freeze in outputs.values():
        paths=[str(Path(ref['path']).resolve()).casefold() for ref in freeze['source']['files']]
        assert len(paths)==len(set(paths))
        assert len([p for p in paths if p.endswith('retained_warmup_extractor.py')])==1
    assert 'retained_kv_warmup_state' not in outputs['off']['cells'][0]['static_inputs']
    assert outputs['on']['retained_kv_warmup']['extractor_ref'] in outputs['on']['source']['files']


def test_retained_existing_target_with_different_bytes_is_not_replaced(data,tmp_path):
    target=Path(data['binding']['extractor_ref']['path'])
    target.write_bytes(b'wrong copied extractor bytes')
    with pytest.raises(ValueError,match='different bytes'):
        a.freeze_retained_warmup_binding(data['selection_path'],[data['row']],tmp_path/'frozen',data['nonflash'])
    assert target.read_bytes()==b'wrong copied extractor bytes'


@pytest.mark.parametrize('conflict',[False,True])
def test_retained_source_manifest_deduplicates_only_matching_sha(data,tmp_path,monkeypatch,conflict):
    def duplicate_source(destination):
        source=ORIGINAL_SOURCE_FREEZE(destination)
        ref=next(ref for ref in source['files'] if Path(ref['path']).name=='retained_warmup_extractor.py')
        source['files'].append({**ref,'sha256':'0'*64 if conflict else ref['sha256']})
        source['sha256']=a.grid.stable_hash(source['files'])
        return source
    monkeypatch.setattr(a,'source_freeze',duplicate_source)
    monkeypatch.setattr(a,'verified_nonflash_kv_view_contract',lambda *args,**kwargs:data['nonflash'])
    kwargs=dict(data_root=tmp_path,nonflash_kv_view_source_contract_path=tmp_path/'nonflash.json',retained_kv_warmup_state=True)
    if conflict:
        with pytest.raises(ValueError,match='conflicting SHA'):
            a.freeze_selection(data['selection_path'],tmp_path/'duplicated',**kwargs)
    else:
        frozen=a.freeze_selection(data['selection_path'],tmp_path/'duplicated',**kwargs)
        refs=frozen['source']['files']
        assert len(refs)==len({str(Path(ref['path']).resolve()).casefold() for ref in refs})
        assert frozen['source']['sha256']==a.grid.stable_hash(refs)


def test_reviewed_extractor_bytes_survive_git_autocrlf_checkout(tmp_path):
    # The reviewed artifact has mixed line endings. Store its raw bytes and
    # ensure this Windows checkout configuration preserves the committed blob.
    import subprocess
    repository=tmp_path/'repository';repository.mkdir()
    subprocess.run(['git','init',str(repository)],check=True,capture_output=True)
    helper=repository/'tools/retained_warmup_extractor.py';helper.parent.mkdir()
    helper.write_bytes(a.RETAINED_WARMUP_EXTRACTOR.read_bytes())
    subprocess.run(['git','-C',str(repository),'-c','core.autocrlf=false','add','--','tools/retained_warmup_extractor.py'],check=True,capture_output=True)
    checkout=tmp_path/'checkout';checkout.mkdir()
    subprocess.run(['git','-C',str(repository),'-c','core.autocrlf=true','checkout-index','--all','--prefix='+str(checkout)+os.sep],check=True,capture_output=True)
    copied=(checkout/'tools/retained_warmup_extractor.py').read_bytes()
    assert hashlib.sha256(copied).hexdigest()==a.RETAINED_WARMUP_EXTRACTOR_SHA256


@pytest.mark.parametrize("field,value", [("prompt_n", 4.0), ("predicted_n", 2.0), ("cache_n", False), ("slot", 1.0), ("slot", True)])
def test_raw_per_request_integer_types_checked_before_dedup(data, tmp_path, field, value):
    # Keep the first request valid: set() must not erase the malformed second one.
    raw = copy.deepcopy(data["raw"])
    request = raw["warmup"][1]["requests"][1]
    if field == "slot":
        request["slot"] = value
    else:
        request["response"]["timings"][field] = value
    raw_ref = document(data["raw_path"], raw)
    data["row"]["evidence_index"][0]["raw_ref"] = raw_ref
    document(data["selection_path"], seal(data["selection"]))
    result = a.freeze_retained_warmup_binding(data["selection_path"], [data["row"]],
        tmp_path / "malformed", data["nonflash"], extractor_path=data["extractor"])
    proof = result["cells"][data["row"]["cell_id"]]
    assert proof["status"] == "uncovered" and proof["contract"] is None
    assert any("warmup_1_qualification_failed" in reason for reason in proof["uncovered_reasons"])
