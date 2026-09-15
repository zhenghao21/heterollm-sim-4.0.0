"""Synthetic evidence only: never import/run simulator or native code."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import render_optimization_loop_report as report


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def metric(error_pct,native=100):
    sim=native*(1+error_pct/100);delta=sim-native
    return {'status':'scored','native_median_ms':native,'simulator_median_ms':sim,
        'signed_error_ms':delta,'absolute_error_ms':abs(delta),
        'signed_error_pct':100*delta/native,'absolute_percentage_error_pct':100*abs(delta)/native}


@pytest.fixture
def evidence(tmp_path):
    root=tmp_path/'source';loop=root/'loop';directory=loop/'candidate';native_refs=[];cells=[]
    for i in range(2):
        ref={'path':str(root/'raw'/f'cell{i}.json'),'sha256':hashlib.sha256(f'raw{i}'.encode()).hexdigest()}
        native_refs.append(ref)
        cells.append({'cell_id':f'cell{i}','model_key':'qwen25','placement_group':'qwen25',
            'metrics':{m:{'native_median_ms':100} for m in report.METRICS},
            'native_actuals':[{'raw_ref':ref},{'raw_ref':ref}]})
    selection={'schema':'native-stable-dataset/v1','planned_cells':4,'selected_count':2,'excluded_count':2,
        'selected_cells':cells,'excluded_cells':[{'cell_id':'bad0','reasons':['unstable']},{'cell_id':'bad1','reasons':['missing']}],
        'failed_attempts':[{'cell_id':'bad0','status':'failed','error':'recorded_clock_failure','source_id':'fixture'} for _ in range(14)],
        'coverage':[{'model_key':'qwen25','placement_group':'qwen25','planned_cells':4,'selected_cells':2,'excluded_cells':2}]}
    selection_ref=write(root/'selection.json',selection)
    state={'schema':'fixed-native-optimization-loop/v1','native_selection_ref':selection_ref,'native_raw_refs':native_refs,
        'status':'round_running','dataset_role':'Previously disclosed development data',
        'best_candidate':str(loop/'must-not-be-autoselected'),
        'budget':{'max_mechanism_rounds':6,'max_full_131_evaluations':4,'max_wall_seconds':28800},
        'full_131_evaluations_started':2,'full_131_evaluations_completed':1,
        'rounds':[{'round':2,'status':'no_effect_deferred','native_dispatch_proven':False},
                  {'round':3,'status':'paired_anchors_accuracy_failed'},
                  {'round':5,'status':'repaired','original_run_status':'zero_applied_preserved','timing_completeness':'unpriced'}],
        'version_control_receipts':[{'kind':'fixture','commit':'a'*40,'status':'pushed','pushed_to':'origin/main'}]}
    state_ref=write(loop/'state.json',state)
    freeze={'schema':'stable-native-simulation-freeze/v1','selection_ref':selection_ref,'selection_sha256':selection_ref['sha256'],
        'selected_denominator':2,'native_grid_denominator':4,'blind_evaluation':False,
        'tensor_storage':{'dequantization_timing':'unmodeled_selected_rows_only','native_dispatch_proven':False},
        'cells':[{'cell_id':r['cell_id'],'static_inputs':{'measurement_state_refs':[native_refs[i]]}} for i,r in enumerate(cells)]}
    freeze_ref=write(directory/'freeze.json',freeze)
    scored=[]
    for i,error in enumerate((9.,10.)):
        pred_ref=write(directory/'predictions'/f'cell{i}.prediction.json',{'cell_id':f'cell{i}','status':'predicted','freeze_ref':freeze_ref})
        scored.append({'cell_id':f'cell{i}','prediction_ref':pred_ref,'metrics':{k:metric(error) for k in report.METRICS.values()}})
    score={'schema':'stable-native-simulation-errors/v1','freeze_ref':freeze_ref,'native_report_ref':selection_ref,
        'selected_denominator':2,'evaluation_type':'development_post_selection','blind_evaluation':False,
        'formal_prediction_eligible':False,'calibration_applied':False,'cells':scored}
    write(directory/'errors.0001.json',score)
    return {'root':root,'loop':loop,'directory':directory,'selection':selection,'state':state,
            'state_path':Path(state_ref['path']),'freeze':freeze,'score':score,'raw_refs':native_refs}


def summarize(evidence,**kwargs):
    return report.make_summary(evidence['state_path'],[('fixture',evidence['directory'])],**kwargs)


def test_complete_score_preserves_denominators_strict_threshold_and_history(evidence):
    summary=summarize(evidence);ev=summary['evaluations'][0]
    assert summary['denominators']=={'original_grid':4,'fixed_selected':2,'excluded':2,'failed_native_attempts':14}
    assert ev['status']=='scored' and ev['prediction_completed']==2 and ev['all3_scored']==2
    assert ev['all3_strict_below10']==1  # exactly 10 is not a strict pass
    assert ev['group_completion'][0]=={'group':'qwen25','selected_cells':2,'all3_scored':2,'all3_strict_below10':1}
    assert ev['group_completion'][1]['all3_strict_below10'] is None
    assert len(ev['groups'])==18
    assert ev['native_identity']['checked_cells']==2
    assert ev['native_identity']['raw_payloads_rehashed'] is False
    assert not (evidence['root']/'raw').exists()  # No raw payload is necessary or opened.
    empty=next(r for r in ev['groups'] if r['group']=='qwen38_gpu')
    assert empty['absolute_percentage_error_pct']['median'] is None
    assert summary['rounds'][0]['flags']==['zero_effect','deferred','conditional']
    assert summary['rounds'][1]['flags']==['failed/regressed']
    assert 'unpriced' in summary['rounds'][2]['flags']
    assert ev['treatments']['tensor_storage']['unpriced_or_unmodeled_declared'] is True


def test_p90_interpolates_and_signed_worst_keeps_negative_sign():
    value=report.statistics_for([1,9,15]);assert value['p90']==pytest.approx(13.8)
    value=report.statistics_for([-30,1,2],signed=True)
    assert value['worst']==-30 and value['min']==-30 and value['max']==2
    assert report.statistics_for([])['median'] is None


def test_missing_score_is_pending_not_zero(evidence):
    (evidence['directory']/'errors.0001.json').unlink()
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='pending' and ev['groups']==[]
    assert ev['prediction_completed'] is None and ev['all3_strict_below10'] is None


def test_multiple_versions_need_explicit_pin_never_choose_latest_or_best(evidence):
    score=copy.deepcopy(evidence['score'])
    for row in score['cells']:
        row['metrics']={k:metric(0.) for k in report.METRICS.values()}
    write(evidence['directory']/'errors.9999.json',score)
    assert summarize(evidence)['evaluations'][0]['status']=='pending_ambiguous'
    pinned=summarize(evidence,score_files={'fixture':'errors.0001.json'})['evaluations'][0]
    assert pinned['all3_strict_below10']==1
    both=report.make_summary(evidence['state_path'],[('old',evidence['directory']),('new',evidence['directory'])],
        {'old':'errors.0001.json','new':'errors.9999.json'})
    assert [e['all3_strict_below10'] for e in both['evaluations']]==[1,2]
    assert both['accuracy_promotion'] is False


def test_pinned_score_not_yet_created_stays_pending(evidence):
    ev=summarize(evidence,score_files={'fixture':'errors.0002.json'})['evaluations'][0]
    assert ev['status']=='pending' and ev['all3_scored'] is None


def test_wrong_score_freeze_rejected(evidence):
    score=copy.deepcopy(evidence['score']);score['freeze_ref']['sha256']='0'*64
    write(evidence['directory']/'errors.0001.json',score)
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='integrity_failed' and 'freeze' in ev['reason']
    assert ev['groups']==[]


def test_swapped_cell_raw_identity_rejected_before_scoring(evidence):
    freeze=copy.deepcopy(evidence['freeze'])
    freeze['cells'][0]['static_inputs']['measurement_state_refs']=[evidence['raw_refs'][1]]
    write(evidence['directory']/'freeze.json',freeze)
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='integrity_failed' and 'raw SHA' in ev['reason']


def test_same_path_with_conflicting_raw_hash_rejected():
    with pytest.raises(report.EvidenceError,match='冲突'):
        report.raw_identity([{'path':'raw.json','sha256':'a'*64},{'path':'raw.json','sha256':'b'*64}])


def test_edited_native_median_cannot_hide_behind_matching_raw_refs(evidence):
    score=copy.deepcopy(evidence['score']);score['cells'][0]['metrics']['engine_ttft_ms']=metric(9,native=101)
    write(evidence['directory']/'errors.0001.json',score)
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='integrity_failed' and '原生中位数' in ev['reason']


@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),True])
def test_scored_invalid_metric_is_not_silently_dropped(evidence,value):
    score=copy.deepcopy(evidence['score']);score['cells'][0]['metrics']['engine_ttft_ms']['absolute_error_ms']=value
    write(evidence['directory']/'errors.0001.json',score)
    assert summarize(evidence)['evaluations'][0]['status']=='integrity_failed'


def test_missing_metric_retains_partial_denominator(evidence):
    score=copy.deepcopy(evidence['score']);score['cells'][0]['metrics']['engine_tpot_ms']={'status':'pending'}
    write(evidence['directory']/'errors.0001.json',score)
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='partial' and ev['prediction_completed']==2 and ev['all3_scored']==1
    assert ev['all3_strict_below10']==0
    row=next(r for r in ev['groups'] if r['group']=='qwen25' and r['metric']=='tpot')
    assert row['scored_cells']==1 and row['selected_cells']==2


def test_missing_prediction_file_does_not_become_zero_completed(evidence):
    (evidence['directory']/'predictions/cell0.prediction.json').unlink()
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='scored_snapshot_missing_predictions'
    assert ev['prediction_completed'] is None and ev['observed_existing_prediction_completed']==1
    assert ev['all3_scored']==2  # historical score data remains explicitly historical


def test_prediction_from_another_freeze_rejected_even_with_updated_score_hash(evidence):
    score=copy.deepcopy(evidence['score']);ref=write(evidence['directory']/'predictions/cell0.prediction.json',
        {'cell_id':'cell0','status':'predicted','freeze_ref':{'sha256':'f'*64}})
    score['cells'][0]['prediction_ref']=ref;write(evidence['directory']/'errors.0001.json',score)
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='integrity_failed' and '预测 SHA/cell/freeze' in ev['reason']


def test_claimed_blind_does_not_promote_disclosed_development_data(evidence):
    score=copy.deepcopy(evidence['score']);score['blind_evaluation']=True;write(evidence['directory']/'errors.0001.json',score)
    summary=summarize(evidence);ev=summary['evaluations'][0]
    assert ev['blind_evaluation'] is False and summary['accuracy_promotion'] is False
    assert ev['role_warning']


def test_report_and_portable_copy_escape_labels_keep_inputs_and_copy_no_raw(evidence,tmp_path):
    directory=evidence['directory'];(directory/'report.html').write_text('<p>existing visual</p>',encoding='utf-8')
    (directory/'heatmap_ttft.svg').write_text('<svg></svg>',encoding='utf-8')
    inputs=list(evidence['root'].rglob('*'));before={str(p):p.read_bytes() for p in inputs if p.is_file()}
    summary=report.make_summary(evidence['state_path'],[('<script>bad</script>',directory)])
    out=tmp_path/'out';portable=tmp_path/'portable'
    report.write_report(summary,out);report.portable_copy(summary,portable)
    assert set(p.name for p in out.iterdir())=={'summary.json','report.md','report.html'}
    page=(out/'report.html').read_text(encoding='utf-8')
    assert '<script>bad</script>' not in page and '&lt;script&gt;' in page
    assert 'lang="zh-CN"' in page and 'viewport' in page and 'http://' not in page
    assert (portable/'evaluations/1/heatmap_ttft.svg').is_file()
    assert not list(portable.rglob('*.prediction.json')) and not list(portable.rglob('freeze.json'))
    assert not list(portable.rglob('errors.*.json')) and not list(portable.rglob('raw'))
    assert before=={p:Path(p).read_bytes() for p in before}
    assert 'evaluations/1/report.html' in (portable/'report.html').read_text(encoding='utf-8')


def test_output_cannot_overwrite_existing_evaluation_report(evidence):
    (evidence['directory']/'report.html').write_text('original',encoding='utf-8')
    with pytest.raises(report.EvidenceError,match='冲突'):
        report.write_report(summarize(evidence),evidence['directory'])
    assert (evidence['directory']/'report.html').read_text(encoding='utf-8')=='original'


def test_cli_has_explicit_repeated_evaluations_and_no_automatic_latest(evidence,tmp_path,capsys):
    code=report.main(['--loop-state',str(evidence['state_path']),'--evaluation','done='+str(evidence['directory']),
        '--score-file','done=errors.0001.json','--evaluation','future='+str(tmp_path/'not_created'),
        '--output',str(tmp_path/'report')])
    result=json.loads(capsys.readouterr().out)
    assert code==0 and result['evaluations']=={'done':'scored','future':'pending'}


def test_duplicate_labels_and_unknown_score_labels_rejected(evidence):
    with pytest.raises(report.EvidenceError,match='重复'):report.assignments(['a=x','a=y'])
    with pytest.raises(report.EvidenceError,match='不存在'):summarize(evidence,score_files={'unknown':'errors.0001.json'})


def test_failed_prediction_cannot_have_scored_metrics(evidence):
    score=copy.deepcopy(evidence['score']);ref=write(evidence['directory']/'predictions/cell0.prediction.json',
        {'cell_id':'cell0','status':'failed','freeze_ref':score['freeze_ref']})
    score['cells'][0]['prediction_ref']=ref;write(evidence['directory']/'errors.0001.json',score)
    ev=summarize(evidence)['evaluations'][0]
    assert ev['status']=='integrity_failed' and '对应成功预测' in ev['reason']
