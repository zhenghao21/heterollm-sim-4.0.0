"""Recompute variance from preserved native responses; never filter slow runs."""
from __future__ import annotations
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/development/native_measurement_audit_20260915'
EXPERIMENTS=['variance_probe_v2','variance_logging_v4','variance_priority_v5','variance_cohort_v6b','variance_confirmation_v7','variance_confirmation_v8']

def cv(values):
    return 100*statistics.stdev(values)/statistics.mean(values) if len(values)>1 and statistics.mean(values)>0 else None

def main():
    rows=[];provenance=[];violations=[]
    for name in EXPERIMENTS:
        folder=BASE/name
        if not (folder/'summary.json').exists():continue
        groups={}
        for file in sorted(folder.glob('*.json')):
            raw=json.loads(file.read_text(encoding='utf-8'))
            if not isinstance(raw,dict) or 'runs' not in raw or not raw.get('model'):continue
            provenance.append({'path':str(file.relative_to(ROOT)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            condition=raw.get('condition') or raw.get('mode')
            bucket=groups.setdefault((Path(raw['model']).name,condition),[])
            batch_values={k:[] for k in ['ttft','tpot','e2e']};request_values={k:[] for k in batch_values};splits=0
            for run in raw['runs']:
                values={k:[] for k in batch_values};begins=[];firsts=[]
                for response,boundary in run['pairs']:
                    t=response['timings'];ts=t['engine_token_times_us'];begin=t['engine_request_begin_us']
                    if not (len(ts)==17==t['predicted_n'] and t['cache_n']==0 and ts==sorted(ts) and begin<=ts[0] and ts[0]==t['engine_prompt_last_us'] and ts[-1]==t['engine_last_token_us']):
                        violations.append(file.name)
                    values['ttft'].append((ts[0]-begin)/1000);values['tpot'].append((ts[-1]-ts[0])/16000);values['e2e'].append((ts[-1]-begin)/1000)
                    begins.append(begin);firsts.append(ts[0])
                splits+=int(max(begins)>min(firsts))
                for metric in values:
                    batch_values[metric].append(statistics.median(values[metric]));request_values[metric]+=values[metric]
            bucket.append({'file':file.name,'batches':len(raw['runs']),'split_batches':splits,'batch_values':batch_values,'request_values':request_values})
        for (model,condition),blocks in groups.items():
            metrics={}
            for metric in ['ttft','tpot','e2e']:
                batches=[v for block in blocks for v in block['batch_values'][metric]]
                requests=[v for block in blocks for v in block['request_values'][metric]]
                means=[statistics.mean(block['batch_values'][metric]) for block in blocks]
                metrics[metric]={'pooled_batch_cv_pct':cv(batches),'all_request_cv_pct':cv(requests),'median_ms':statistics.median(batches),'stdev_ms':statistics.stdev(batches),'mean_ms':statistics.mean(batches),'min_ms':min(batches),'max_ms':max(batches),'block_cv_pct':[cv(block['batch_values'][metric]) for block in blocks],'block_means_ms':means,'last_first_block_mean_change_pct':100*(means[-1]/means[0]-1)}
            rows.append({'experiment':name,'model':model,'condition':condition,'blocks':len(blocks),'batches':sum(b['batches'] for b in blocks),'requests':sum(len(b['request_values']['ttft']) for b in blocks),'split_batches':sum(b['split_batches'] for b in blocks),'metrics':metrics})
    result={'schema':'native-variance-review/v1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'interpretation':'empirical CV; not a confidence bound, independent accuracy validation, or observer equivalence','violations':violations,'rows':rows,'source_files':provenance,'failed_experiments':[{'experiment':'variance_persistent_v3','reason':'server silently closes SSE connection; no POST retry'},{'experiment':'variance_cohort_v6','reason':'initial parser rejected legitimate null begin event; failed experiment retained'}]}
    out=BASE/'variance_report.json'
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    lines=['# Native measurement variance report','', 'All slow runs retained. CV = sample standard deviation / mean of batch request medians. These are measurement fluctuations, not simulator prediction errors.','', '| Experiment | Model | Condition | Batches | TTFT CV | TPOT CV | E2E CV | Split batches |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        vals=[f"{r['metrics'][k]['pooled_batch_cv_pct']:.3f}%" for k in ['ttft','tpot','e2e']]
        lines.append('| '+' | '.join([r['experiment'],r['model'],r['condition'],str(r['batches']),*vals,str(r['split_batches'])])+' |')
    lines+=['','The former persistent-connection and initial cohort parser failures remain in their original directories. No default/native actual was overwritten. The atomic cohort is a different submission policy and must not replace independent HTTP results.','', 'Raw cross-block and per-request statistics, source hashes, and all failed experiments are in variance_report.json. Only two models at prompt 5 / output 17 / concurrency 4 are represented; other shapes, 27B, observer equivalence and prediction-error acceptance remain unvalidated.']
    with (BASE/'variance_report.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'groups':len(rows),'violations':len(violations),'requests':sum(r['requests'] for r in rows)}))
if __name__=='__main__':main()
