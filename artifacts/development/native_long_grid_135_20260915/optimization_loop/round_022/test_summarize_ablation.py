"""Synthetic four-way structural tests; no native, model or simulator execution."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

P=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

s=module('r22_report_tests',P/'summarize_ablation.py')
d=module('r22_driver_tests',P/'evaluate_candidate.py')


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return s.reference(path)


def campaign(tmp_path,monkeypatch,*,predictions=True,scores=True,failed_pure=True):
    directory=tmp_path
    for name in ('evaluate_candidate.py','summarize_ablation.py'):
        (directory/name).write_bytes((P/name).read_bytes())
    monkeypatch.setattr(s,'__file__',str(directory/'summarize_ablation.py'))
    monkeypatch.setattr(d,'__file__',str(directory/'evaluate_candidate.py'))
    monkeypatch.setattr(d,'P',directory);monkeypatch.setattr(d,'s',s)
    selected_ref=put(directory/'selected.json',{'synthetic_static_identity':True})
    monkeypatch.setattr(s,'SELECTION_SHA',selected_ref['sha256'])
    ids=[f'qwen25_{i:02}' for i in range(17)]+[name+'_p512_o32_c1' for name in ('qwen35','smollm2','tinyllama')]
    universe=sorted(ids+[f'extra_{i:03}' for i in range(111)]);ids=sorted(ids)
    sampling_contract={'synthetic':True};sample_ref=put(directory/'sampling_contract.json',sampling_contract)
    nonflash={'schema':'heterollm.llama-nonflash-kv-view/v1','n_pad':1,'n_kv_padding':256,
        'context_allocation_alignment':256,'native_latency_used':False,'rules':{'occupied_lower_bound':'largest_current_sequence_retained_context_only'}}
    nonflash_ref=put(directory/'nonflash.json',nonflash)
    policy={'mode':'greedy','implementation':'llama_cpp_cpu_chain','top_k':1,'min_keep':0,'temperature':0.0}
    sample_binding={'contract_ref':sample_ref,'cells':{i:{'typed_policy':policy} for i in universe},'evidence_refs':[]}
    kv_binding={'contract_ref':nonflash_ref,'contract':nonflash,'cells':{i:{'cell_id':i} for i in universe},'evidence_refs':[]}
    freezes={};maps={}
    for variant in s.VARIANTS:
        mode=s.TREATMENTS[variant];source_root=directory/variant/'source';source_refs=[]
        for name in ('src/heterollm_sim/config.py','src/heterollm_sim/planner.py','src/heterollm_sim/cost_models.py','tools/predict_stable_native_dataset.py','tools/native_llama_compare.py'):
            path=source_root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# synthetic source\n');source_refs.append(s.reference(path))
        cells=[];issue_cells={}
        for ident in universe:
            model_key=ident.split('_')[0] if not ident.startswith('extra') else 'qwen38'
            proof={'contract':{'synthetic_gpu_contract':True},'mmq_source_costs_requested':mode['mmq'],'conversion_cta_costs_requested':mode['cta']}
            inputs={'cell_id':ident,'config':{'expected_prompt_tokens':512,'output':32,'parallel':1,'seed':42},
                'gpu_mmq_source_costs':mode['mmq'],'gpu_conversion_cta_costs':mode['cta'],
                'gpu_invocation_contract':proof['contract'],'gpu_invocation_evidence':proof,
                'sampling_binding':sample_binding['cells'][ident] if mode['sampling'] else None,
                'nonflash_kv_view_contract':kv_binding['cells'][ident] if mode['nonflash'] else None}
            if mode['issue']:
                cell={'requested':True,'status':'conditional','contract':{'sm_count':84},'native_latency_used':False,
                    'native_instruction_mapping_proven':False,'gpu_invocation_sha256':s.stable_hash(proof)}
                issue_cells[ident]=cell;inputs.update(mmvq_vector_issue_bound=True,mmvq_issue_contract=cell['contract'],mmvq_issue_evidence=cell)
            cells.append({'cell_id':ident,'model_key':model_key,'deployment':'synthetic','preparation_error':None,'static_inputs':inputs})
        freeze={'schema':'stable-native-simulation-freeze/v1','selected_denominator':131,'selection_sha256':s.SELECTION_SHA,
            'selection_ref':selected_ref,'created_utc':'2026-01-01T00:00:00+00:00','blind_evaluation':False,'calibration_applied':False,
            'source':{'root':str(source_root),'files':source_refs,'sha256':s.stable_hash(source_refs)},'cells':cells,
            'sampling':sample_binding if mode['sampling'] else None,'nonflash_kv_view':kv_binding if mode['nonflash'] else None}
        if mode['issue']:freeze.update(mmvq_vector_issue_bound=True,mmvq_issue_bound={'requested':True,'cells':issue_cells,'conditional_cell_count':131,'uncovered_cell_count':0})
        put(directory/variant/'freeze.json',freeze);freezes[variant]=freeze;maps[variant]={row['cell_id']:row for row in cells}
    protocol={'schema':'mmvq-four-way-ablation/v1','variants':list(s.VARIANTS),'candidate':s.CANDIDATE,'treatments':s.TREATMENTS,
        'full_denominator':131,'anchor_denominator':20,'anchor_ids':ids,'threshold_pct_strict':10.0,'metrics':list(s.METRICS),
        'native_lock':{'selection_sha256':s.SELECTION_SHA,'selected_cells':131},'is_blind':False,'formal_acceptance':False,
        'native_remeasurement':False,'calibration_added':False,'accuracy_selected_subset':False,'gate_B':'unvalidated',
        'driver_ref':s.reference(directory/'evaluate_candidate.py'),'summarizer_ref':s.reference(directory/'summarize_ablation.py'),
        'baseline_freeze_ref':s.reference(directory/'current/freeze.json')}
    put(directory/'evaluation_protocol.json',protocol)
    files=[directory/'evaluate_candidate.py',directory/'summarize_ablation.py',directory/'evaluation_protocol.json',*(directory/v/'freeze.json' for v in s.VARIANTS)]
    content=s.source_content(freezes['current'])
    put(directory/'evaluation_controls.json',{'schema':'mmvq-four-way-controls/v1','variants':list(s.VARIANTS),'same_source_closure':True,
        'files':[s.reference(path) for path in files],'source_content_sha256':s.stable_hash(content),'source_file_count':len(content)})
    controls=s.check_controls(directory)[1]
    def save_prediction(variant,ident,full=False):
        row=maps[variant][ident];failed=failed_pure and variant=='pure' and ident==ids[0]
        time=120-5*s.VARIANTS.index(variant)
        value={'schema':'stable-native-cell-prediction/v1','cell_id':ident,'model_key':row['model_key'],'deployment':row['deployment'],
            'status':'failed' if failed else 'predicted','reason':'synthetic failure' if failed else None,
            'freeze_ref':s.reference(directory/variant/'freeze.json'),'selection_sha256':s.SELECTION_SHA,'source_sha256':freezes[variant]['source']['sha256'],
            'created_utc':f'2026-01-01T0{6 if full else 1}:00:00+00:00','finished_utc':f'2026-01-01T0{7 if full else 2}:00:00+00:00',
            'native_answers_used':False,'calibration_applied':False,'formal_prediction_eligible':False,
            'input_identity':row['static_inputs'] if failed else {'static_inputs_sha256':s.stable_hash(row['static_inputs'])},
            'aggregate':{metric:{'median_ms':time} for metric in s.METRICS}}
        return put(directory/variant/'predictions'/(ident+'.prediction.json'),value)
    def save_score(variant,full=False):
        planned=universe if full else ids;rows=[]
        for ident,row in maps[variant].items():
            path=directory/variant/'predictions'/(ident+'.prediction.json');pred=s.read_json(path)[0] if ident in planned else None
            metrics={}
            for metric in s.METRICS:
                if pred is None or pred['status']!='predicted':metrics[metric]={'status':'unscored','reason':'synthetic failure' if pred else 'not run'}
                else:
                    value=pred['aggregate'][metric]['median_ms'];delta=value-100
                    metrics[metric]={'status':'scored','simulator_median_ms':value,'native_median_ms':100,'signed_error_ms':delta,
                        'absolute_error_ms':abs(delta),'signed_error_pct':delta,'absolute_percentage_error_pct':abs(delta),'native_run_medians_ms':[100,100,100]}
            rows.append({'cell_id':ident,'model_key':row['model_key'],'deployment':row['deployment'],
                'prediction_ref':s.reference(path) if pred else None,'metrics':metrics})
        value={'schema':'stable-native-simulation-errors/v1','selected_denominator':131,'freeze_ref':s.reference(directory/variant/'freeze.json'),
            'native_report_ref':selected_ref,'blind_evaluation':False,'formal_prediction_eligible':False,'calibration_applied':False,
            'created_utc':f'2026-01-01T0{9 if full else 4}:00:00+00:00','cells':rows}
        return put(directory/variant/('errors.0002.json' if full else 'errors.0001.json'),value)
    if predictions:
        refs={v:{'freeze_ref':s.reference(directory/v/'freeze.json'),'prediction_refs':{i:save_prediction(v,i) for i in ids}} for v in s.VARIANTS}
        barrier_ref=put(directory/'anchors_predictions.json',{'schema':'r22-prediction-barrier/v1','phase':'anchors','created_utc':'2026-01-01T03:00:00+00:00',
            'controls_ref':controls['controls'],'variants':refs,'native_answers_used':False,'terminal_failures_preserved':True})
        if scores:
            put(directory/'anchors_scores.json',{'schema':'r22-score-receipt/v1','phase':'anchors','created_utc':'2026-01-01T05:00:00+00:00',
                'controls_ref':controls['controls'],'prediction_barrier_ref':barrier_ref,'scores':{v:save_score(v) for v in s.VARIANTS}})
    def mocked_guard():
        return protocol,controls,s.frozen_bundles(directory,protocol)[0],protocol['native_lock']
    monkeypatch.setattr(d,'guard',mocked_guard)
    return {'directory':directory,'ids':ids,'universe':universe,'freezes':freezes,'controls':controls,
        'save_prediction':save_prediction,'save_score':save_score,'guard':mocked_guard}


def test_four_variants_declared_switches_and_source_semantics(tmp_path,monkeypatch):
    c=campaign(tmp_path,monkeypatch)
    result=s.assess(tmp_path,'anchors')
    assert result['variants']==['pure','current','cta_only','cta_issue']
    assert set(result['anchor_comparisons'])=={'cta_only_minus_current','cta_issue_minus_cta_only','current_minus_pure'}
    assert result['anchor_summary']['pure']['failed_or_unscored_cells']==1
    assert result['candidate131']['cells']==131 and result['gate_A']=='not_passed' and result['gate_B']=='unvalidated'
    assert result['anchor_comparisons']['cta_issue_minus_cta_only']['by_metric'][s.METRICS[0]]['denominator']==20
    before=s.static_variant_semantics(c['guard']()[2]['current'])
    bundle=copy.deepcopy(c['guard']()[2]['current']);bundle['cells'][c['ids'][0]]['static_inputs']['config']['seed']=999
    assert s.static_variant_semantics(bundle)!=before


def test_source_size_optional_only_for_declared_source_refs(tmp_path):
    source=tmp_path/'locked.cuh';source.write_text('header')
    ref=s.reference(source);small={key:ref[key] for key in ('path','sha256')}
    verified=s.verify_source_evidence_reference(s.normalized_source_evidence_ref(small))
    assert verified['observed_bytes']==6 and verified['declared_bytes'] is None
    with pytest.raises(s.EvidenceError,match='byte size'):s.verify_reference(small)
    good={'gpu_invocation':{'contract':{'source_refs':[small]},'evidence_refs':[small]}}
    assert len(s.evidence_closure(good))==1
    with pytest.raises(s.EvidenceError,match='byte size'):s.evidence_closure({'runtime_build_audit':{'evidence_refs':[small]}})
    wrong={**small,'size_bytes':7}
    with pytest.raises(s.EvidenceError,match='size'):s.verify_source_evidence_reference(s.normalized_source_evidence_ref(wrong))
    source.write_text('mutated')
    with pytest.raises(s.EvidenceError,match='changed'):s.verify_source_evidence_reference(small)


def test_fourth_variant_cannot_disappear(tmp_path,monkeypatch):
    campaign(tmp_path,monkeypatch)
    barrier=s.read_json(tmp_path/'anchors_predictions.json')[0];barrier['variants'].pop('cta_issue');put(tmp_path/'anchors_predictions.json',barrier)
    with pytest.raises(s.EvidenceError,match='every planned variant'):s.assess(tmp_path,'anchors')


def test_score_cannot_precede_four_way_barrier(tmp_path,monkeypatch):
    campaign(tmp_path,monkeypatch)
    score=s.read_json(tmp_path/'pure/errors.0001.json')[0];score['created_utc']='2026-01-01T02:30:00+00:00';ref=put(tmp_path/'pure/errors.0001.json',score)
    receipt=s.read_json(tmp_path/'anchors_scores.json')[0];receipt['scores']['pure']=ref;put(tmp_path/'anchors_scores.json',receipt)
    with pytest.raises(s.EvidenceError,match='all four'):s.assess(tmp_path,'anchors')


def test_prediction_hash_tamper_and_controls_tamper_rejected(tmp_path,monkeypatch):
    c=campaign(tmp_path,monkeypatch)
    prediction=tmp_path/'cta_issue/predictions'/(c['ids'][0]+'.prediction.json');prediction.write_text('{}')
    with pytest.raises(s.EvidenceError,match='changed evidence'):s.assess(tmp_path,'anchors')
    (tmp_path/'evaluate_candidate.py').write_text('changed')
    with pytest.raises(s.EvidenceError,match='changed evidence'):s.check_controls(tmp_path)


def test_full131_keeps_failed_anchors_and_strict_threshold(tmp_path,monkeypatch):
    c=campaign(tmp_path,monkeypatch)
    def predict(variant,ids,workers,timeout):
        assert variant=='cta_issue' and ids is None
        for ident in c['universe']:
            if ident not in c['ids']:c['save_prediction'](variant,ident,True)
    monkeypatch.setattr(d,'predict_variant',predict)
    monkeypatch.setattr(d,'run',lambda argv:c['save_score']('cta_issue',True))
    times=iter(['2026-01-01T08:00:00+00:00','2026-01-01T10:00:00+00:00']);monkeypatch.setattr(d,'now',lambda:next(times))
    d.full(workers=1)
    result=s.assess(tmp_path,'full')
    assert result['candidate131']['strict_all3_below10_cells']==131 and result['gate_A']=='passed' and result['gate_B']=='unvalidated'
    assert result['anchor_summary']['pure']['failed_or_unscored_cells']==1
    rows=copy.deepcopy(result['candidate131_rows']);rows[c['ids'][0]]['metrics'][s.METRICS[0]]['absolute_percentage_error_pct']=10.0
    assert s.summarize(rows)['strict_all3_below10_cells']==130
    before={p:s.reference(p) for p in tmp_path.glob('*/predictions/*.prediction.json')}
    d.full(workers=1)
    assert all(s.reference(p)==ref for p,ref in before.items())


def test_anchors_never_score_and_resume_reuses_all_four(tmp_path,monkeypatch):
    c=campaign(tmp_path,monkeypatch,predictions=False,scores=False);calls=[]
    def predict(variant,ids,workers,timeout):
        calls.append(variant)
        for ident in ids:c['save_prediction'](variant,ident)
    monkeypatch.setattr(d,'predict_variant',predict);monkeypatch.setattr(d,'run',lambda argv:pytest.fail('anchors must not score'))
    monkeypatch.setattr(d,'now',lambda:'2026-01-01T03:00:00+00:00')
    d.anchors(workers=1);assert calls==list(s.VARIANTS)
    d.anchors(workers=1);assert calls==list(s.VARIANTS)
    assert not list(tmp_path.glob('*/errors.*.json'))


def test_score_phase_requires_all_four_terminal_sets_and_resumes_partial_scores(tmp_path,monkeypatch):
    c=campaign(tmp_path,monkeypatch,scores=False);c['save_score']('pure');calls=[]
    def run(argv):
        assert '--score' in argv
        directory=Path(argv[argv.index('--output')+1]);variant=directory.name
        assert all((tmp_path/v/'predictions'/(i+'.prediction.json')).exists() for v in s.VARIANTS for i in c['ids'])
        calls.append(variant);c['save_score'](variant)
    monkeypatch.setattr(d,'run',run);monkeypatch.setattr(d,'now',lambda:'2026-01-01T05:00:00+00:00')
    d.score_anchors();assert calls==['current','cta_only','cta_issue']
    d.score_anchors();assert calls==['current','cta_only','cta_issue']


def test_report_outputs_exclusive_and_heatmap_contains_fourth_arm(tmp_path,monkeypatch):
    campaign(tmp_path,monkeypatch);result=s.assess(tmp_path,'anchors')
    paths=s.write_report(tmp_path,result,True);assert len(paths)==3
    svg=(tmp_path/'ablation_anchors_heatmap.svg').read_text(encoding='utf-8');assert 'cta_issue/TTFT' in svg and 'width="1370"' in svg
    with pytest.raises(s.EvidenceError,match='existing report'):s.write_report(tmp_path,result,True)


def test_freeze_arguments_keep_cta_contrast_isolated(tmp_path):
    ref={'path':'static.json'};baseline={key:{'audit_ref' if key=='runtime_build_audit' else 'contract_ref':ref} for key in ('runtime_build_audit','recurrent_batching','slot_order','host_offload_source','tensor_storage')}
    baseline.update(data_root=str(tmp_path),model_snapshot_map={})
    args={v:d.freeze_arguments(baseline,v) for v in s.VARIANTS}
    a,b,c=args['current'],args['cta_only'],args['cta_issue']
    assert {key for key in a if a[key]!=b[key]}=={'gpu_conversion_cta_costs'}
    assert {key for key in b if b[key]!=c[key]}=={'mmvq_vector_issue_bound'}
    assert a['nonflash_kv_view_source_contract_path'] is not None and args['pure']['sampling_contract_path'] is None


def test_no_simulator_or_execution_imports_in_report():
    text=(P/'summarize_ablation.py').read_text(encoding='utf-8')
    assert 'import subprocess' not in text and 'from heterollm_sim' not in text and 'round_018' not in text
    assert "'errors.0001.json'" in text and "'errors.0002.json'" in text


def test_duplicate_json_nonfinite_and_distribution(tmp_path):
    p=tmp_path/'duplicate.json';p.write_text('{"a":1,"a":2}')
    with pytest.raises(s.EvidenceError,match='duplicate'):s.read_json(p)
    p.write_text('{"a":NaN}')
    with pytest.raises(s.EvidenceError,match='nonfinite'):s.read_json(p)
    assert s.distribution([])=={'count':0,'median':None,'p90':None,'worst':None}
