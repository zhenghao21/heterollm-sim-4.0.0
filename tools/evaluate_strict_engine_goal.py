"""Strict fixed-native development gate. Read only; formal acceptance fails closed."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools import render_optimization_loop_report as report
from tools import predict_stable_native_dataset as predictor
from tools.verify_fixed_native import verify_lock
METRICS=('engine_ttft_ms','engine_tpot_ms','engine_e2e_ms')
def finite_positive(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0

def check_cell(prediction, native, scored):
    """Recompute from per-request absolute timestamps, never trust stored APE."""
    issues=[];metrics={}
    if prediction.get('status')!='predicted':return {'verdict':'insufficient_evidence','issues':['prediction not successful'],'metrics':{},'all3_below10':False}
    if prediction.get('native_answers_used') is not False:issues.append('native answer usage missing or enabled')
    requests=prediction.get('requests');parallel=native.get('parallel')
    if type(parallel) is not int or parallel<1 or not isinstance(requests,list) or len(requests)!=parallel:
        return {'verdict':'insufficient_evidence','issues':['request coverage mismatch'],'metrics':{},'all3_below10':False}
    values={m:[] for m in METRICS};ids=set()
    for row in requests:
        ident=row.get('request_index');count=row.get('visible_output_tokens')
        if type(ident) is not int or ident in ids or not 0<=ident<parallel:issues.append('request identity missing/duplicate')
        if type(ident) is int:ids.add(ident)
        if type(count) is not int or count<=1 or count!=native.get('output_tokens'):issues.append('output token count mismatch');continue
        if row.get('prompt_tokens')!=native.get('prompt_tokens'):issues.append('prompt token count mismatch')
        begin,first,last=[row.get(k) for k in ('engine_request_begin_ns','engine_first_token_ns','engine_last_token_ns')]
        if not all(type(t) in (float,int) and math.isfinite(t) and t>=0 for t in (begin,first,last)) or not begin<first<last:
            issues.append('invalid engine timestamps');continue
        derived=dict(zip(METRICS,((first-begin)/1e6,(last-first)/(count-1)/1e6,(last-begin)/1e6)))
        for m,v in derived.items():
            if not finite_positive(row.get(m)) or not math.isclose(v,row[m],rel_tol=1e-9,abs_tol=1e-8):issues.append('request metric/timestamp mismatch:'+m)
            values[m].append(v)
    for m in METRICS:
        try:
            if len(values[m])!=parallel:raise ValueError('incomplete derived metrics')
            sim=statistics.median(values[m]);actual,runs=predictor.native_run_medians(native,m)
            if not finite_positive(actual):raise ValueError('invalid native median')
            aggregate=prediction.get('aggregate',{}).get(m,{})
            if aggregate.get('planned_requests')!=parallel or aggregate.get('observed_requests')!=parallel or aggregate.get('missing_requests')!=0:raise ValueError('aggregate request coverage')
            stored=aggregate.get('median_ms')
            if not finite_positive(stored) or not math.isclose(sim,stored,rel_tol=1e-9,abs_tol=1e-8):raise ValueError('aggregate differs from timestamps')
            item=scored.get('metrics',{}).get(m,{})
            report.metric_values(item)
            if item.get('status')!='scored' or not finite_positive(item.get('simulator_median_ms')) or not math.isclose(sim,item['simulator_median_ms'],rel_tol=1e-9,abs_tol=1e-8) or not math.isclose(actual,item['native_median_ms'],rel_tol=1e-9,abs_tol=1e-8):raise ValueError('score differs from native/request derivation')
            error=abs(sim-actual)/actual*100
            metrics[m]={'simulator_median_ms':sim,'native_median_ms':actual,'absolute_percentage_error_pct':error,'signed_error_ms':sim-actual,'absolute_error_ms':abs(sim-actual),'passed':error<10}
        except (ValueError,KeyError,TypeError) as exc:issues.append(m+':'+str(exc))
    passed=not issues and len(metrics)==3 and all(x['passed'] for x in metrics.values())
    return {'verdict':'insufficient_evidence' if issues else 'passed' if passed else 'accuracy_failed','issues':issues,'metrics':metrics,'all3_below10':passed}

def evaluate(state_path,directory,score_name):
    state,state_ref=report.read_json(state_path);lock=verify_lock(state_path)
    selection,selection_ref=report.read_json(state['native_selection_ref']['path'])
    selected=report.rows_by_id(selection['selected_cells']);expected=set(selection['selected_cell_ids'])
    if set(selected)!=expected or len(expected)!=131:raise ValueError('fixed 131-cell scope mismatch')
    freeze,freeze_ref=report.read_json(Path(directory)/'freeze.json')
    if not freeze.get('source',{}).get('files'):raise ValueError('empty frozen source map')
    predictor.verify_freeze_references(freeze)
    if set(report.rows_by_id(freeze['cells']))!=expected:raise ValueError('freeze scope missing/extra cells')
    evidence=report.load_evaluation('strict_A',directory,score_name,selection,selection_ref,selected,{c:report.native_cell_raws(r) for c,r in selected.items()})
    scores,score_ref=report.read_json(Path(directory)/score_name)
    scored=report.rows_by_id(scores.get('cells'))
    if set(scored)!=expected:raise ValueError('score scope missing/extra cells')
    rows=[];pred_refs=[]
    for cell in selection['selected_cell_ids']:
        row=scored[cell];ref=row.get('prediction_ref')
        try:
            pred,actual_ref=report.read_json(report.evidence_path(ref,directory))
            if actual_ref['sha256']!=ref.get('sha256') or pred.get('freeze_ref',{}).get('sha256')!=freeze_ref['sha256'] or pred.get('selection_sha256')!=selection_ref['sha256'] or pred.get('cell_id')!=cell:raise ValueError('prediction identity mismatch')
            pred_refs.append(actual_ref);result=check_cell(pred,selected[cell],row)
        except (ValueError,KeyError,TypeError,OSError) as exc:result={'verdict':'insufficient_evidence','issues':[str(exc)],'metrics':{},'all3_below10':False}
        rows.append({'cell_id':cell,**result})
    # Rehash all documents after extraction to detect changes during audit.
    for ref in [state_ref,selection_ref,freeze_ref,score_ref,*pred_refs]:
        if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('evidence changed during audit:'+ref['path'])
    verify_lock(state_path);predictor.verify_freeze_references(freeze)
    passed=sum(r['all3_below10'] for r in rows);missing=sum(r['verdict']=='insufficient_evidence' for r in rows);failed=sum(r['verdict']=='accuracy_failed' for r in rows)
    return {'schema':'strict-engine-acceptance/v1','gate_A':{'verdict':'accuracy_failed' if failed else 'insufficient_evidence' if missing else 'passed','passed_cells':passed,'accuracy_failed_cells':failed,'insufficient_evidence_cells':missing,'required_cells':131,'required_metrics':393,'passing_metrics':sum(v['passed'] for r in rows if not r['issues'] for v in r['metrics'].values()),'threshold_pct_strict':10,'prediction_coverage_pct':100*(131-missing)/131},'gate_B':{'verdict':'unvalidated','reason':'No registered independent acceptance and joint uncertainty evidence supplied; development gate never promotes B.'},'task_complete':False,'next_action':'repair_evidence_and_next_mechanism_round' if missing else 'next_mechanism_round' if failed else 'freeze_and_prepare_independent_acceptance','native_evaluable_coverage':{'selected':131,'planned':selection['planned_cells'],'pct':131/selection['planned_cells']*100},'native_lock':lock,'sources':{'state':state_ref,'selection':selection_ref,'freeze':freeze_ref,'score':score_ref,'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'cells':rows,'scope':'Fixed disclosed development gate only; no inference run, refit, or native remeasurement.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--score-file',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('refusing overwrite')
    try:result=evaluate(a.state,a.evaluation,a.score_file)
    except (ValueError,KeyError,TypeError,OSError) as exc:result={'schema':'strict-engine-acceptance/v1','gate_A':{'verdict':'insufficient_evidence','reason':str(exc)},'gate_B':{'verdict':'unvalidated'},'task_complete':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k in ('gate_A','gate_B','task_complete')},ensure_ascii=False));return 0 if result['gate_A']['verdict']=='passed' else 4
if __name__=='__main__':raise SystemExit(main())
