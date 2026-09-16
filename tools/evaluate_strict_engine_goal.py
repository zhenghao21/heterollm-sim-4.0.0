"""Strict fixed-native development gate. Read only; formal acceptance fails closed."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools import render_optimization_loop_report as report
from tools import predict_stable_native_dataset as predictor
from tools import verify_fixed_native as native_lock
from tools.verify_fixed_native import verify_lock
METRICS=('engine_ttft_ms','engine_tpot_ms','engine_e2e_ms')
REQUIRED_CELLS=131
THRESHOLD_PCT=10


def _implementation_paths():
    # Only the project modules used by this audit's validation call chain.
    return {'evaluator':Path(__file__), 'report':Path(report.__file__),
            'predictor':Path(predictor.__file__), 'native_lock':Path(native_lock.__file__),
            'grid':Path(predictor.grid.__file__)}


def _implementation_identity():
    refs={}
    for role,path in _implementation_paths().items():
        path=Path(path).resolve(strict=True);raw=path.read_bytes()
        refs[role]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    return refs


def _document_snapshot(document):
    raw=json.dumps(document,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return {'encoding':'canonical-json/utf-8','sha256':hashlib.sha256(raw).hexdigest(),
            'bytes':len(raw),'document':json.loads(raw)}


def _policy_snapshot():
    return _document_snapshot({'schema':'strict-engine-gate-policy/v1',
        'gate_A':{'required_cells':REQUIRED_CELLS,'required_metrics':REQUIRED_CELLS*len(METRICS),
                  'metrics':list(METRICS),'threshold_pct_strict':THRESHOLD_PCT,
                  'comparison':'absolute_percentage_error_pct < threshold_pct_strict',
                  'cell_pass_requires':'all three metrics and complete validated request/native evidence',
                  'native_run_groups':3,'native_remeasurement_allowed':False},
        'gate_B':{'verdict':'unvalidated','development_gate_may_promote_B':False},'task_complete':False})


class AuditEvidenceError(ValueError):
    def __init__(self,message,context):
        super().__init__(message)
        self.audit_context=context


def _cell_result(issues,metrics):
    insufficient=bool(issues) or len(metrics)!=len(METRICS)
    failed=any(not value['passed'] for value in metrics.values())
    passed=not insufficient and not failed
    # Keep the legacy verdict; independent flags preserve mixed failures.
    return {'verdict':'insufficient_evidence' if insufficient else 'passed' if passed else 'accuracy_failed',
            'issues':issues,'metrics':metrics,'all3_below10':passed,
            'accuracy_failed':failed,'insufficient_evidence':insufficient}

def finite_positive(x):
    return type(x) in (int,float) and math.isfinite(x) and x>0

def check_cell(prediction, native, scored):
    """Recompute from per-request absolute timestamps, never trust stored APE."""
    issues=[];metrics={}
    if prediction.get('status')!='predicted':return _cell_result(['prediction not successful'],{})
    if prediction.get('native_answers_used') is not False:issues.append('native answer usage missing or enabled')
    requests=prediction.get('requests');parallel=native.get('parallel')
    if type(parallel) is not int or parallel<1 or not isinstance(requests,list) or len(requests)!=parallel:
        return _cell_result([*issues,'request coverage mismatch'],{})
    actuals=native.get('native_actuals')
    if isinstance(actuals,list) and any(not isinstance(a,dict) or type(a.get('request_index')) is not int for a in actuals):
        return _cell_result([*issues,'native request identity invalid'],{})
    values={m:[] for m in METRICS};ids=set();metric_issues={m:[] for m in METRICS}
    for row in requests:
        if not isinstance(row,dict):issues.append('request must be an object');continue
        ident=row.get('request_index');count=row.get('visible_output_tokens')
        if type(ident) is not int or ident in ids or not 0<=ident<parallel:
            issues.append('request identity missing/duplicate');continue
        ids.add(ident)
        if type(count) is not int or count<=1 or count!=native.get('output_tokens'):issues.append('output token count mismatch');continue
        if row.get('prompt_tokens')!=native.get('prompt_tokens'):issues.append('prompt token count mismatch')
        begin,first,last=[row.get(k) for k in ('engine_request_begin_ns','engine_first_token_ns','engine_last_token_ns')]
        if not all(type(t) in (float,int) and math.isfinite(t) and t>=0 for t in (begin,first,last)) or not begin<first<last:
            issues.append('invalid engine timestamps');continue
        derived=dict(zip(METRICS,((first-begin)/1e6,(last-first)/(count-1)/1e6,(last-begin)/1e6)))
        for m,v in derived.items():
            if not finite_positive(row.get(m)) or not math.isclose(v,row[m],rel_tol=1e-9,abs_tol=1e-8):metric_issues[m].append('request metric/timestamp mismatch')
            values[m].append(v)
    for m in METRICS:
        try:
            if metric_issues[m]:raise ValueError(';'.join(metric_issues[m]))
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
            metrics[m]={'simulator_median_ms':sim,'native_median_ms':actual,'absolute_percentage_error_pct':error,'signed_error_ms':sim-actual,'absolute_error_ms':abs(sim-actual),'passed':error<THRESHOLD_PCT}
        except (ValueError,KeyError,TypeError,AttributeError) as exc:issues.append(m+':'+str(exc))
    return _cell_result(issues,metrics)

def evaluate(state_path,directory,score_name):
    context={'implementation':{'before':_implementation_identity()},'snapshots':{'policy':_policy_snapshot()}}
    error=None
    try:
        result=_evaluate(state_path,directory,score_name,context)
    except (ValueError,KeyError,TypeError,OSError,AttributeError) as exc:
        error=exc
    try:
        after=_implementation_identity()
    except (ValueError,TypeError,OSError) as exc:
        context['implementation'].update(unchanged=False,after_error=str(exc))
        raise AuditEvidenceError('implementation identity unavailable after audit: '+str(exc),context) from exc
    identity=context['implementation'];identity.update(after=after,unchanged=identity['before']==after)
    if not identity['unchanged']:
        changed=sorted(role for role in set(identity['before'])|set(after) if identity['before'].get(role)!=after.get(role))
        raise AuditEvidenceError('evaluator implementation changed during audit: '+', '.join(changed),context) from error
    if error is not None:raise AuditEvidenceError(str(error),context) from error
    result['sources']['implementation']=identity
    result['snapshots']=context['snapshots']
    return result


def _evaluate(state_path,directory,score_name,context):
    state,state_ref=report.read_json(state_path)
    context['snapshots']['state']={**_document_snapshot(state),'original_ref':state_ref,
        'scope':'Historical audit input; later intentional changes to the live state do not change this embedded snapshot.'}
    lock=verify_lock(state_path)
    selection,selection_ref=report.read_json(state['native_selection_ref']['path'])
    selected=report.rows_by_id(selection['selected_cells']);expected=set(selection['selected_cell_ids'])
    if set(selected)!=expected or len(expected)!=REQUIRED_CELLS:raise ValueError('fixed 131-cell scope mismatch')
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
        except (ValueError,KeyError,TypeError,OSError,AttributeError) as exc:result=_cell_result([str(exc)],{})
        rows.append({'cell_id':cell,**result})
    # Rehash all documents after extraction to detect changes during audit.
    for ref in [state_ref,selection_ref,freeze_ref,score_ref,*pred_refs]:
        if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('evidence changed during audit:'+ref['path'])
    verify_lock(state_path);predictor.verify_freeze_references(freeze)
    passed=sum(r['all3_below10'] for r in rows);missing=sum(r['insufficient_evidence'] for r in rows);failed=sum(r['accuracy_failed'] for r in rows)
    return {'schema':'strict-engine-acceptance/v1','gate_A':{'verdict':'accuracy_failed' if failed else 'insufficient_evidence' if missing else 'passed','passed_cells':passed,'accuracy_failed_cells':failed,'insufficient_evidence_cells':missing,'required_cells':REQUIRED_CELLS,'required_metrics':REQUIRED_CELLS*len(METRICS),'failure_counts_may_overlap':True,'passing_metrics':sum(v['passed'] for r in rows if not r['issues'] for v in r['metrics'].values()),'threshold_pct_strict':THRESHOLD_PCT,'prediction_coverage_pct':100*(REQUIRED_CELLS-missing)/REQUIRED_CELLS},'gate_B':{'verdict':'unvalidated','reason':'No registered independent acceptance and joint uncertainty evidence supplied; development gate never promotes B.'},'task_complete':False,'next_action':'repair_evidence_and_next_mechanism_round' if missing else 'next_mechanism_round' if failed else 'freeze_and_prepare_independent_acceptance','native_evaluable_coverage':{'selected':REQUIRED_CELLS,'planned':selection['planned_cells'],'pct':REQUIRED_CELLS/selection['planned_cells']*100},'native_lock':lock,'sources':{'state':state_ref,'selection':selection_ref,'freeze':freeze_ref,'score':score_ref,'evaluator_sha256':context['implementation']['before']['evaluator']['sha256']},'cells':rows,'scope':'Fixed disclosed development gate only; no inference run, refit, or native remeasurement.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--score-file',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('refusing overwrite')
    try:result=evaluate(a.state,a.evaluation,a.score_file)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        result={'schema':'strict-engine-acceptance/v1','gate_A':{'verdict':'insufficient_evidence','reason':str(exc)},'gate_B':{'verdict':'unvalidated'},'task_complete':False}
        if isinstance(exc,AuditEvidenceError):
            result['sources']={'implementation':exc.audit_context['implementation']}
            result['snapshots']=exc.audit_context['snapshots']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k in ('gate_A','gate_B','task_complete')},ensure_ascii=False));return 0 if result['gate_A']['verdict']=='passed' else 4
if __name__=='__main__':raise SystemExit(main())
