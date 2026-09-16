"""R21 summarizer structural tests with temporary synthetic evidence only."""
import copy,importlib.util,json,sys
from pathlib import Path
import pytest
SPEC=importlib.util.spec_from_file_location('r21_summary',Path(__file__).with_name('summarize_ablation.py'));s=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(s)
STAMP='2026-01-01T00:00:00+00:00'

def write(path,doc):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc),encoding='utf-8');return s.reference(path)
def cells():
    rows=[{'cell_id':f'qwen25_{i}','model_key':'qwen25','deployment':'gpu','preparation_error':None,'static_inputs':{'config':{'prompt_tokens':512,'output_tokens':32,'parallel':1}}} for i in range(17)]
    rows += [{'cell_id':m+'_p512_o32_c1','model_key':m,'deployment':'gpu','preparation_error':None,'static_inputs':{'config':{'prompt_tokens':512,'output_tokens':32,'parallel':1}}} for m in ['qwen35','smollm2','tinyllama']]
    rows += [{'cell_id':f'other_{i}','model_key':'other','deployment':'gpu','preparation_error':None,'static_inputs':{'config':{}}} for i in range(111)]
    return rows

def record(value=105,native=100):
    delta=value-native
    return {'status':'scored','simulator_median_ms':value,'native_median_ms':native,'native_run_medians_ms':[native]*3,'signed_error_ms':delta,'absolute_error_ms':abs(delta),'signed_error_pct':100*delta/native,'absolute_percentage_error_pct':100*abs(delta)/native}
def scored(ident='a',value=105):return {'cell_id':ident,'model_key':'m','deployment':'gpu','prediction_status':'predicted','prediction_failure_reason':None,'metrics':{m:record(value) for m in s.METRICS}}

def bundle(tmp_path,monkeypatch):
    native=write(tmp_path/'native.json',{'fixture':True});monkeypatch.setattr(s,'SELECTION_SHA',native['sha256'])
    cs=cells();ids=s.anchors({x['cell_id']:x for x in cs});freeze={'schema':'stable-native-simulation-freeze/v1','created_utc':STAMP,'selection_sha256':native['sha256'],'selection_ref':native,'selected_denominator':131,'blind_evaluation':False,'calibration_applied':False,'source':{'sha256':'a'*64},'cells':cs};fr=write(tmp_path/'freeze.json',freeze)
    for row in cs:
        ident=row['cell_id']
        if ident not in ids:continue
        pred={'schema':'stable-native-cell-prediction/v1','cell_id':ident,'model_key':row['model_key'],'deployment':'gpu','status':'predicted','freeze_ref':fr,'selection_sha256':native['sha256'],'source_sha256':'a'*64,'created_utc':'2026-01-01T01:00:00+00:00','finished_utc':'2026-01-01T02:00:00+00:00','native_answers_used':False,'calibration_applied':False,'formal_prediction_eligible':False,'input_identity':{'static_inputs_sha256':s.stable_hash(row['static_inputs'])},'aggregate':{m:{'median_ms':105} for m in s.METRICS}}
        write(tmp_path/'predictions'/(ident+'.prediction.json'),pred)
    return s.load_bundle(tmp_path,'pure')

def score_doc(b):
    rows=[]
    for ident,cell in b['cells'].items():
        rows.append({'cell_id':ident,'model_key':cell['model_key'],'deployment':cell['deployment'],'prediction_ref':b['prediction_refs'].get(ident),'metrics':{m:record() if ident in b['predictions'] else {'status':'unscored','reason':'prediction unavailable or incomplete'} for m in s.METRICS}})
    return {'schema':'stable-native-simulation-errors/v1','selected_denominator':131,'freeze_ref':b['freeze_ref'],'native_report_ref':b['freeze']['selection_ref'],'blind_evaluation':False,'formal_prediction_eligible':False,'calibration_applied':False,'created_utc':'2026-01-01T03:00:00+00:00','cells':rows}

def test_anchor_selection_exact131_and20():
    cs=s.unique_cells({'cells':cells()});assert len(s.anchors(cs))==20
    with pytest.raises(s.EvidenceError,match='denominator'):s.unique_cells({'cells':cells()[:-1]})

def test_load_predictions_and131_retained_scores(tmp_path,monkeypatch):
    b=bundle(tmp_path,monkeypatch);doc=score_doc(b);write(tmp_path/'errors.0001.json',doc);out=s.load_score(b,'errors.0001.json',b['ids']);summary=s.summarize(out['rows'])
    assert summary['cells']==131 and summary['strict_all3_below10_cells']==20 and summary['failed_or_unscored_cells']==111
    assert all(x['absolute_percentage_error_pct']['count']==20 for x in summary['metrics'].values())

@pytest.mark.parametrize('change',['early','wrong_freeze','wrong_prediction','wrong_native','wrong_group','bad_arithmetic','duplicate_cell','missing_reason'])
def test_score_forgery_rejected(tmp_path,monkeypatch,change):
    b=bundle(tmp_path,monkeypatch);doc=score_doc(b)
    if change=='early':doc['created_utc']=STAMP
    elif change=='wrong_freeze':doc['freeze_ref']={**doc['freeze_ref'],'sha256':'b'*64}
    elif change=='wrong_prediction':doc['cells'][0]['prediction_ref']={**doc['cells'][0]['prediction_ref'],'sha256':'b'*64}
    elif change=='wrong_native':doc['native_report_ref']={**doc['native_report_ref'],'sha256':'b'*64}
    elif change=='wrong_group':doc['cells'][0]['model_key']='other'
    elif change=='bad_arithmetic':doc['cells'][0]['metrics'][s.METRICS[0]]['absolute_percentage_error_pct']=1
    elif change=='duplicate_cell':doc['cells'][-1]=copy.deepcopy(doc['cells'][0])
    else:doc['cells'][-1]['metrics'][s.METRICS[0]]['reason']=''
    write(tmp_path/'errors.0001.json',doc)
    with pytest.raises(s.EvidenceError):s.load_score(b,'errors.0001.json',b['ids'])

def test_failed_prediction_preserved(tmp_path,monkeypatch):
    b=bundle(tmp_path,monkeypatch);ident=b['ids'][0];path=tmp_path/'predictions'/(ident+'.prediction.json');pred=json.loads(path.read_text());pred.update(status='failed',reason='unsupported fixture',input_identity=b['cells'][ident]['static_inputs']);write(path,pred)
    b=s.load_bundle(tmp_path,'pure');doc=score_doc(b)
    row=next(x for x in doc['cells'] if x['cell_id']==ident);row['metrics']={m:{'status':'unscored','reason':'prediction unavailable or incomplete'} for m in s.METRICS};write(tmp_path/'errors.0001.json',doc)
    out=s.load_score(b,'errors.0001.json',b['ids']);assert s.summarize(out['rows'])['failed_or_unscored_cells']==112
    assert out['rows'][ident]['prediction_failure_reason']=='unsupported fixture'

def test_exact10_is_not_pass_and_comparison_keeps_failures():
    rows={'a':scored(value=110),'b':scored(value=109.99)};out=s.summarize(rows);assert out['strict_all3_below10_cells']==1
    before={'a':scored()};after={'a':scored(value=101)};assert s.compare(before,after,['a'])['outcomes']=={'improved':3}
    after['a']['metrics'][s.METRICS[0]]={'status':'unscored','reason':'failed'}
    assert s.compare(before,after,['a'])['outcomes']=={'unscored':1,'improved':2}

def test_changed_prediction_detected_and_duplicate_json_rejected(tmp_path):
    path=tmp_path/'a.json';ref=write(path,{'x':1});write(path,{'x':2})
    with pytest.raises(s.EvidenceError,match='changed evidence'):s.verify_reference(ref)
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(s.EvidenceError,match='duplicate JSON'):s.read_json(path)

def test_source_map_closure_requires_frozen_files(tmp_path):
    root=tmp_path/'source';names=['src/heterollm_sim/config.py','src/heterollm_sim/planner.py','src/heterollm_sim/cost_models.py','tools/predict_stable_native_dataset.py','tools/native_llama_compare.py'];refs=[]
    for name in names:
        q=root/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text('# fixture');refs.append(s.reference(q))
    freeze={'source':{'root':str(root),'files':refs,'sha256':s.stable_hash(refs)}}
    assert len(s.source_content(freeze))==5
    (root/names[0]).write_text('# changed')
    with pytest.raises(s.EvidenceError):s.source_content(freeze)

def semantic_fixture(tmp_path,variant):
    cs={x['cell_id']:x for x in cells()};binding={'cells':{i:{'typed_policy':{'mode':'greedy','implementation':'llama_cpp_cpu_chain','top_k':1,'min_keep':0,'temperature':0.0}} for i in cs}}
    binding['contract_ref']=write(tmp_path/'sampling.json',{'fixture':True})
    contract={'schema':'heterollm.llama-nonflash-kv-view/v1','n_pad':1,'n_kv_padding':256,'context_allocation_alignment':256,'native_latency_used':False,'rules':{'occupied_lower_bound':'largest_current_sequence_retained_context_only'}}
    nonflash={'contract':contract,'contract_ref':write(tmp_path/'physical.json',contract),'cells':{i:{**contract,'runtime_binding_status':'verified'} for i in cs}}
    freeze={'sampling':binding if variant!='pure' else None,'nonflash_kv_view':nonflash if variant=='physical' else None}
    for i,row in cs.items():
        row['static_inputs'].update(gpu_mmq_source_costs=variant!='pure',gpu_conversion_cta_costs=False,gpu_invocation_evidence={'mmq_source_costs_requested':variant!='pure','conversion_cta_costs_requested':False},sampling_binding=binding['cells'][i] if variant!='pure' else None,nonflash_kv_view_contract=nonflash['cells'][i] if variant=='physical' else None)
    return {'variant':variant,'freeze':freeze,'cells':cs}

def test_only_declared_static_treatments_normalized(tmp_path):
    bundles={v:semantic_fixture(tmp_path,v) for v in s.VARIANTS};normalized=[s.static_variant_semantics(b) for b in bundles.values()];assert normalized[0]==normalized[1]==normalized[2]
    bundles['physical']['cells']['qwen25_0']['static_inputs']['config']['parallel']=2
    assert s.static_variant_semantics(bundles['physical'])!=normalized[0]

def test_no_cross_round_score_reads_or_external_python_imports():
    text=Path(s.__file__).read_text(encoding='utf-8');assert 'round_018' not in text and 'round_017' not in text and 'heterollm_sim import' not in text
    assert "'errors.0001.json'" in text and "'errors.0002.json'" in text and 'subprocess' not in text

def minimal_result():
    rows={v:{'a':scored()} for v in s.VARIANTS};unscored={**scored('b'),'prediction_status':'not_run_at_this_score','metrics':{m:{'status':'unscored','reason':'missing'} for m in s.METRICS}}
    return {'phase':'anchors','physical131':s.summarize({'a':scored(),'b':unscored}),'gate_A':'not_passed','anchor_summary':{v:s.summarize(r) for v,r in rows.items()},'anchor_rows':rows,'anchor_comparisons':{},'coverage':{'physical_saved_predictions':20,'physical_missing_prediction_cells':111}}

def test_exclusive_outputs_and_svg(tmp_path):
    result=minimal_result();paths=s.write_report(tmp_path,result,True);assert len(paths)==3 and (tmp_path/'ablation_anchors_heatmap.svg').read_text().endswith('</svg>')
    before={p:Path(p).read_bytes() for p in paths}
    with pytest.raises(s.EvidenceError,match='existing report'):s.write_report(tmp_path,result,True)
    assert all(Path(p).read_bytes()==data for p,data in before.items())

def test_empty_valid_metric_distribution():assert s.distribution([])=={'count':0,'median':None,'p90':None,'worst':None}


def test_full_history_retains_unrun_anchor_score_rows(tmp_path,monkeypatch):
    b=bundle(tmp_path,monkeypatch);doc=score_doc(b);ident=next(i for i in b['cells'] if i not in b['ids'])
    b['predictions'][ident]={'status':'predicted','finished_utc':'2026-01-02T00:00:00+00:00'}
    b['prediction_refs'][ident]={'path':'future','bytes':1,'sha256':'a'*64}
    write(tmp_path/'errors.0001.json',doc);old=s.load_score(b,'errors.0001.json',b['ids'])
    assert old['rows'][ident]['prediction_status']=='not_run_at_this_score'
    assert old['rows'][ident]['metrics'][s.METRICS[0]]['status']=='unscored'

@pytest.mark.parametrize('phase,early_full', [('anchors',False),('full',True)])
def test_cross_variant_and_full_phase_timestamp_order(tmp_path,monkeypatch,phase,early_full):
    ids=[f'anchor{i}' for i in range(20)];universe=ids+[f'extra{i}' for i in range(111)]
    predictions={i:{'status':'predicted','created_utc':'2026-01-01T01:00:00+00:00','finished_utc':'2026-01-01T02:00:00+00:00'} for i in universe}
    bundles={v:{'variant':v,'directory':tmp_path/v,'freeze':{'sampling':None},'freeze_ref':{},'cells':{},'ids':ids,'planned':universe if phase=='full' and v=='physical' else ids,'predictions':copy.deepcopy({i:predictions[i] for i in universe if i in ids or phase=='full' and v=='physical'}),'prediction_refs':{}} for v in s.VARIANTS}
    monkeypatch.setattr(s,'check_controls',lambda _:({'anchor_ids':ids},{}));monkeypatch.setattr(s,'load_bundle',lambda directory,v,full=False:bundles[v]);monkeypatch.setattr(s,'source_content',lambda _:{});monkeypatch.setattr(s,'static_variant_semantics',lambda _:{});monkeypatch.setattr(s,'evidence_closure',lambda _:[])
    if not early_full:bundles['physical']['predictions'][ids[0]]['finished_utc']='2026-01-01T04:00:00+00:00'
    def score(b,name,expected):return {'created_utc':'2026-01-01T03:00:00+00:00','rows':{i:scored(i) for i in universe},'ref':{}}
    monkeypatch.setattr(s,'load_score',score)
    with pytest.raises(s.EvidenceError,match='before all three|before anchor scoring'):
        s.assess(tmp_path,phase)

def test_full131_strict_pass_denominator_and_failures():
    rows={f'cell{i}':scored(f'cell{i}',105) for i in range(131)};summary=s.summarize(rows)
    assert summary['cells']==131 and summary['strict_all3_below10_cells']==131
    rows['cell0']['metrics'][s.METRICS[0]]={'status':'unscored','reason':'unsupported physical dispatch'}
    summary=s.summarize(rows);assert summary['strict_all3_below10_cells']==130 and summary['full131_denominator']==131 and summary['failed_or_unscored_cells']==1
