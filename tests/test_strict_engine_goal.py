import copy
import unittest
from pathlib import Path
from tools.evaluate_strict_engine_goal import check_cell, METRICS
from tools.render_optimization_loop_report import rows_by_id

def fixture(scale=1):
    native={'parallel':1,'output_tokens':3,'prompt_tokens':128,'native_actuals':[{'block':0,'repeat':i,'request_index':0,'metrics_ms':{'ttft':100.,'tpot':100.,'e2e':300.}} for i in range(3)]}
    vals=dict(zip(METRICS,(100*scale,100*scale,300*scale)))
    request={'request_index':0,'visible_output_tokens':3,'prompt_tokens':128,'engine_request_begin_ns':0.,'engine_first_token_ns':100e6*scale,'engine_last_token_ns':300e6*scale,**vals}
    pred={'status':'predicted','native_answers_used':False,'requests':[request],'aggregate':{k:{'planned_requests':1,'observed_requests':1,'missing_requests':0,'median_ms':v} for k,v in vals.items()}}
    scored={'metrics':{}}
    for k,v in vals.items():
        actual=100 if k!=METRICS[2] else 300;delta=v-actual
        scored['metrics'][k]={'status':'scored','native_median_ms':actual,'simulator_median_ms':v,'signed_error_ms':delta,'absolute_error_ms':abs(delta),'signed_error_pct':100*delta/actual,'absolute_percentage_error_pct':100*abs(delta)/actual}
    return pred,native,scored

class GateTests(unittest.TestCase):
    def test_exact_and_boundary(self):
        for scale,want in [(1,True),(1.09999999,True),(1.1,False),(1.10000001,False)]:
            with self.subTest(scale=scale):self.assertEqual(check_cell(*fixture(scale))['all3_below10'],want)
    def test_bad_values(self):
        for x in [None,True,float('nan'),float('inf'),-1]:
            p,n,s=fixture();p['requests'][0][METRICS[0]]=x
            self.assertEqual(check_cell(p,n,s)['verdict'],'insufficient_evidence')
    def test_score_tamper(self):
        p,n,s=fixture(1.4);s['metrics'][METRICS[0]]['absolute_percentage_error_pct']=0
        self.assertEqual(check_cell(p,n,s)['verdict'],'insufficient_evidence')
    def test_aggregate_tamper(self):
        p,n,s=fixture();p['aggregate'][METRICS[0]]['median_ms']=50
        self.assertFalse(check_cell(p,n,s)['all3_below10'])
    def test_missing_metric(self):
        p,n,s=fixture();del s['metrics'][METRICS[0]]
        self.assertFalse(check_cell(p,n,s)['all3_below10'])
    def test_missing_request(self):
        p,n,s=fixture();p['requests']=[]
        self.assertFalse(check_cell(p,n,s)['all3_below10'])
    def test_native_answer_use(self):
        p,n,s=fixture();p['native_answers_used']=True
        self.assertFalse(check_cell(p,n,s)['all3_below10'])
    def test_native_repeat_missing(self):
        p,n,s=fixture();n['native_actuals'].pop()
        self.assertFalse(check_cell(p,n,s)['all3_below10'])
    def test_duplicate_cells(self):
        with self.assertRaises(ValueError):rows_by_id([{'cell_id':'a'},{'cell_id':'a'}])
    def test_wrong_tokens(self):
        p,n,s=fixture();p['requests'][0]['visible_output_tokens']=4
        self.assertFalse(check_cell(p,n,s)['all3_below10'])
if __name__=='__main__':unittest.main()

# Real on-disk 131-cell chain; only native invocation is absent.
import json
import hashlib
import pytest
from tools.evaluate_strict_engine_goal import evaluate

def save(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(doc),encoding='utf-8')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

@pytest.fixture
def full_chain(tmp_path):
    native_ref=save(tmp_path/'native.json',{'synthetic':True});selected=[];predictions=[];scores=[]
    for i in range(131):
        pred,native,score=fixture();ident=f'cell{i}';native.update(cell_id=ident,model_key='qwen25',metrics={alias:{'native_median_ms':v} for alias,v in [('ttft',100),('tpot',100),('e2e',300)]})
        for a in native['native_actuals']:a['raw_ref']=native_ref
        selected.append(native);predictions.append(pred);scores.append(score)
    sel={'selected_cells':selected,'selected_cell_ids':[x['cell_id'] for x in selected],'selected_count':131,'planned_cells':162}
    sel_ref=save(tmp_path/'selection.json',sel);src_ref=save(tmp_path/'source.json',{'frozen':True})
    freeze={'schema':'stable-native-simulation-freeze/v1','selection_ref':sel_ref,'selection_sha256':sel_ref['sha256'],'selected_denominator':131,'native_grid_denominator':162,'source':{'files':[src_ref]},'cells':[{'cell_id':x['cell_id'],'static_inputs':{'measurement_state_refs':[native_ref]}} for x in selected]}
    fr=save(tmp_path/'freeze.json',freeze)
    for n,(pred,score) in enumerate(zip(predictions,scores)):
        ident=f'cell{n}';pred.update(cell_id=ident,freeze_ref=fr,selection_sha256=sel_ref['sha256']);ref=save(tmp_path/'predictions'/f'{ident}.json',pred);score.update(cell_id=ident,prediction_ref=ref)
    score={'schema':'stable-native-simulation-errors/v1','freeze_ref':fr,'native_report_ref':sel_ref,'selected_denominator':131,'cells':scores}
    save(tmp_path/'errors.json',score)
    state={'schema':'fixed-native-optimization-loop/v1','native_selection_ref':sel_ref,'native_selected_cells':131,'native_remeasurement_allowed_in_this_loop':False,'native_raw_refs':[native_ref]};save(tmp_path/'state.json',state)
    return tmp_path,score

def test_full_chain_pass_does_not_promote_B(full_chain):
    root,_=full_chain;r=evaluate(root/'state.json',root,'errors.json')
    assert r['gate_A']['passed_cells']==131 and r['gate_A']['verdict']=='passed'
    assert r['gate_B']['verdict']=='unvalidated' and r['task_complete'] is False

@pytest.mark.parametrize('mutation',['missing_cell','duplicate_cell','cross_freeze','tampered_prediction','empty_sources'])
def test_whole_chain_rejects(full_chain,mutation):
    root,score=full_chain
    if mutation=='missing_cell':score['cells'].pop();save(root/'errors.json',score)
    elif mutation=='duplicate_cell':score['cells'][-1]=score['cells'][0];save(root/'errors.json',score)
    elif mutation in ('cross_freeze','tampered_prediction'):
        path=root/'predictions/cell0.json';p=json.loads(path.read_text());p['freeze_ref']['sha256']='0'*64 if mutation=='cross_freeze' else p['freeze_ref']['sha256'];p['requests'][0]['engine_ttft_ms']=1;save(path,p)
    else:
        path=root/'freeze.json';f=json.loads(path.read_text());f['source']['files']=[];save(path,f)
    with pytest.raises(ValueError):evaluate(root/'state.json',root,'errors.json')



from tools import evaluate_strict_engine_goal as gate


def test_mixed_metric_failure_preserves_both_flags():
    pred,native,score=fixture(1.2)
    del score['metrics'][METRICS[0]]
    result=check_cell(pred,native,score)
    assert result['verdict']=='insufficient_evidence'
    assert result['accuracy_failed'] is True and result['insufficient_evidence'] is True
    assert result['all3_below10'] is False and len(result['metrics'])==2
    assert all(value['passed'] is False for value in result['metrics'].values())


def test_invalid_request_metric_cannot_supply_an_accuracy_failure():
    pred,native,score=fixture(1.2)
    for metric in METRICS:pred['requests'][0][metric]=None
    result=check_cell(pred,native,score)
    assert result['insufficient_evidence'] is True
    assert result['accuracy_failed'] is False and result['metrics']=={}


@pytest.mark.parametrize('value',[None,True,False,0.0,'0',[],{},-1,1])
@pytest.mark.parametrize('side',['prediction','native'])
def test_malformed_request_identity_is_insufficient(value,side):
    pred,native,score=fixture()
    if side=='prediction':pred['requests'][0]['request_index']=value
    else:native['native_actuals'][0]['request_index']=value
    result=check_cell(pred,native,score)
    assert result['insufficient_evidence'] is True and result['all3_below10'] is False


@pytest.mark.parametrize('row',[None,[],0,'request'])
def test_malformed_request_object_is_insufficient(row):
    pred,native,score=fixture();pred['requests'][0]=row
    result=check_cell(pred,native,score)
    assert result['insufficient_evidence'] is True and result['all3_below10'] is False


def test_mixed_cell_failure_counts_overlap(full_chain):
    root,score=full_chain
    pred,_,cell_score=fixture(1.2)
    original=json.loads((root/'predictions/cell0.json').read_text())
    pred.update({key:original[key] for key in ('cell_id','freeze_ref','selection_sha256')})
    ref=save(root/'predictions/cell0.json',pred)
    cell_score.update(cell_id='cell0',prediction_ref=ref)
    del cell_score['metrics'][METRICS[0]]
    score['cells'][0]=cell_score;save(root/'errors.json',score)
    result=evaluate(root/'state.json',root,'errors.json')
    assert result['gate_A']['verdict']=='accuracy_failed'
    assert result['gate_A']['accuracy_failed_cells']==1
    assert result['gate_A']['insufficient_evidence_cells']==1
    assert result['gate_A']['passed_cells']==130
    assert result['gate_A']['passing_metrics']==390
    assert result['gate_A']['required_cells']==131 and result['gate_A']['required_metrics']==393
    assert result['gate_B']['verdict']=='unvalidated' and result['task_complete'] is False


def assert_snapshot(snapshot):
    raw=json.dumps(snapshot['document'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    assert hashlib.sha256(raw).hexdigest()==snapshot['sha256']
    assert len(raw)==snapshot['bytes']


def test_implementation_chain_and_embedded_snapshots(full_chain):
    root,_=full_chain;result=evaluate(root/'state.json',root,'errors.json')
    identity=result['sources']['implementation']
    assert identity['unchanged'] is True and identity['before']==identity['after']
    assert set(identity['before'])=={'evaluator','report','predictor','native_lock','grid'}
    for role,path in gate._implementation_paths().items():
        ref=identity['before'][role]
        assert ref['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert Path(ref['path'])==path.resolve()
    assert result['sources']['evaluator_sha256']==identity['before']['evaluator']['sha256']
    for snapshot in result['snapshots'].values():assert_snapshot(snapshot)
    policy=result['snapshots']['policy']['document']
    assert policy['gate_A']['required_cells']==131 and policy['gate_A']['required_metrics']==393
    assert policy['gate_A']['threshold_pct_strict']==10
    assert policy['gate_B']['development_gate_may_promote_B'] is False


def test_later_state_update_does_not_invalidate_saved_snapshot(full_chain):
    root,_=full_chain;result=evaluate(root/'state.json',root,'errors.json')
    output=root/'gate.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,allow_nan=False)
    original_output=output.read_bytes()
    live=json.loads((root/'state.json').read_text());live['rounds']=[{'round':14,'status':'planned'}]
    new_ref=save(root/'state.json',live)
    saved=json.loads(output.read_text(encoding='utf-8'))
    snapshot=saved['snapshots']['state']
    assert 'rounds' not in snapshot['document']
    assert snapshot['original_ref']['sha256']!=new_ref['sha256']
    assert_snapshot(snapshot)
    assert output.read_bytes()==original_output and saved['gate_A']['verdict']=='passed'


def arrange_implementation_mutation(root,monkeypatch,role):
    paths={}
    for name,path in gate._implementation_paths().items():
        copy=root/'implementation'/path.name;copy.parent.mkdir(parents=True,exist_ok=True)
        copy.write_bytes(path.read_bytes());paths[name]=copy
    monkeypatch.setattr(gate,'_implementation_paths',lambda:paths)
    original=gate.check_cell;changed=False
    def mutate_once(*args):
        nonlocal changed
        if not changed:
            paths[role].write_bytes(paths[role].read_bytes()+b'\n# changed during audit\n')
            changed=True
        return original(*args)
    monkeypatch.setattr(gate,'check_cell',mutate_once)


@pytest.mark.parametrize('role',['evaluator','report','predictor','native_lock','grid'])
def test_implementation_change_during_evaluation_fails_closed(full_chain,monkeypatch,role):
    root,_=full_chain;arrange_implementation_mutation(root,monkeypatch,role)
    with pytest.raises(gate.AuditEvidenceError,match='implementation changed during audit') as exc:
        evaluate(root/'state.json',root,'errors.json')
    identity=exc.value.audit_context['implementation']
    assert identity['unchanged'] is False
    assert identity['before'][role]['sha256']!=identity['after'][role]['sha256']
    assert_snapshot(exc.value.audit_context['snapshots']['state'])


def test_cli_retains_failed_audit_provenance_and_refuses_overwrite(full_chain,monkeypatch):
    root,_=full_chain;arrange_implementation_mutation(root,monkeypatch,'report')
    output=root/'failed-gate.json'
    monkeypatch.setattr(gate.sys,'argv',['gate','--state',str(root/'state.json'),'--evaluation',str(root),
        '--score-file',str(root/'errors.json'),'--output',str(output)])
    assert gate.main()==4
    result=json.loads(output.read_text(encoding='utf-8'))
    assert result['gate_A']['verdict']=='insufficient_evidence'
    assert result['sources']['implementation']['unchanged'] is False
    assert result['gate_B']['verdict']=='unvalidated' and result['task_complete'] is False
    assert_snapshot(result['snapshots']['state']);original=output.read_bytes()
    with pytest.raises(SystemExit,match='refusing overwrite'):gate.main()
    assert output.read_bytes()==original
