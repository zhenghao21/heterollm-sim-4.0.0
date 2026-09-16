"""Diagnostic only: compare immutable static predictions to completed synthetic traces."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,statistics
P=Path(__file__).resolve().parent
EXPECTED='66da88e6b8d24a1728945ddb17454ba42d028ec524f7b118fb3b175f1acb3c20'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def ref(p):
 p=p.resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def union(intervals):
 end=None;total=0
 for a,b in sorted(intervals):
  if b<a:raise ValueError('negative interval')
  if end is None or a>end:total+=b-a
  else:total+=max(0,b-end)
  end=max(end or b,b)
 return total
def dist(v):return {'count':len(v),'median_ns':statistics.median(v),'min_ns':min(v),'max_ns':max(v)}
def decompose(c):
 begin,end=c['nvtx_verbatim']['start'],c['nvtx_verbatim']['end'];kernels=[e['kernel'] for e in c['kernel_launch_pairs']];apis=c['api_events_verbatim']
 ki=[(k['start'],k['end']) for k in kernels];launch=[(a['start'],a['end']) for a in apis if 'LaunchKernel' in a.get('name_text','')];sync=[(a['start'],a['end']) for a in apis if 'Synchronize' in a.get('name_text','')]
 other=[(a['start'],a['end']) for a in apis if 'LaunchKernel' not in a.get('name_text','') and 'Synchronize' not in a.get('name_text','')]
 endpoints=sorted({begin,end,*[v for pair in ki+launch+sync+other for v in pair if begin<=v<=end]});exclusive={'gpu_active':0,'launch_only':0,'sync_only':0,'other_api_only':0,'unattributed_idle':0}
 for a,b in zip(endpoints,endpoints[1:]):
  mid=(a+b)/2
  label=next((name for name,spans in [('gpu_active',ki),('launch_only',launch),('sync_only',sync),('other_api_only',other)] if any(x<=mid<y for x,y in spans)),'unattributed_idle');exclusive[label]+=b-a
 if sum(exclusive.values())!=end-begin:raise ValueError('partition failure')
 first=min(a for a,b in ki);last=max(b for a,b in ki)
 return {**exclusive,'full_nvtx':end-begin,'kernel_union':union(ki),'kernel_span':last-first,'inter_kernel_gap':last-first-union(ki),'before_first_kernel':first-begin,'after_last_kernel':end-last,
 'launch_api_union':union(launch),'sync_api_union':union(sync),'other_api_union':union(other)}
def main():
 output=P/'device_gap_analysis.json'
 if output.exists():raise ValueError('immutable output already exists')
 baseline=P/'analytical_predictions.0001.json'
 if ref(baseline)['sha256']!=EXPECTED:raise ValueError('premeasurement prediction identity mismatch')
 predictions={r['config']['id']:r for r in load(baseline)['predictions']};paths=[]
 for summary in sorted((P/'collection_r2/analysis_0001').rglob('pair_summary.json')):
  d=load(summary)
  if d.get('trace_chain_complete') is True:paths.append(summary)
 if len(paths)!=8:raise ValueError('expected exactly eight completed r2 mapped pairs')
 paths.append(P/'collection_r3/first_pair_analysis/pair_summary.json');sources=[ref(baseline),ref(Path(__file__))];rows=[]
 for summary in paths:
  mapped_path=summary.parent/'mapped_calls.json';sources +=[ref(summary),ref(mapped_path)];s=load(summary);mapped=load(mapped_path);config=mapped['config'];prediction=predictions[config['id']];formal=[c for c in mapped['calls'] if c['phase']=='formal']
  if len(formal)!=30 or any(c['issues'] for c in formal):raise ValueError('completed trace contract not met')
  roles={};breakdowns=[decompose(c) for c in formal]
  for c in formal:
   for item in c['kernel_launch_pairs']:
    name=item['role'].split('_')[0];event=item['kernel'];roles.setdefault(name,[]).append(event['end']-event['start'])
  comparisons={}
  for name,times in roles.items():
   actual=statistics.median(times);analytic=prediction['source_qualified_roles'][name]['device_ns'];comparisons[name]={'actual_device':dist(times),'analytic_device_ns':analytic,'signed_relative_error_pct':100*(analytic-actual)/actual,'analytic_div_actual':analytic/actual,'conditional_floor_would_change_ns':max(analytic,actual)-analytic,'direction':'underpredict' if analytic<actual else 'overpredict'}
  union_med=statistics.median([d['kernel_union'] for d in breakdowns]);ana=prediction['device_total_ns']
  rows.append({'collection':summary.relative_to(P).parts[0],'config':config,'pair':s['pair'],'roles':comparisons,
    'device_total':{'actual_union_median_ns':union_med,'analytic_device_ns':ana,'signed_relative_error_pct':100*(ana-union_med)/union_med},
    'same_nsight_clock_partition':{key:dist([d[key] for d in breakdowns]) for key in breakdowns[0]},
    'individual_call_partitions':breakdowns,'profile_host':s['profile_host'],'direct_host':s['direct_host'],
    'host_median_relative_difference_pct':100*(s['profile_host']['median_ns']-s['direct_host']['median_ns'])/s['direct_host']['median_ns'],
    'analytic_launch_total_ns':prediction['launch_total_ns'],'calibration_eligible':False,
    'partition_policy':'GPU active first; outside GPU launch API then sync API then other API then unattributed idle. Exclusive per-call timeline buckets sum to NVTX. Category medians do not generally sum to median NVTX.',
    'no_cross_clock_subtraction':True})
 for r in sources:
  if ref(Path(r['path']))!=r:raise ValueError('diagnostic input changed')
 result={'schema':'r16-device-gap-diagnostic/v1','created_utc':datetime.now(timezone.utc).isoformat(),'sources':sources,'completed_pairs':len(rows),'formal_calls':270,'rows':rows,'diagnostic_only':True,'coefficients_fitted':False,'profile_created':False,'llm_actual_read':False,'gpu_executed':False,'all_samples_quality_ineligible_or_insufficient':True}
 with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 for row in rows:
  print(row['collection'],row['config']['id'],row['pair'],{k:round(v['signed_relative_error_pct'],1) for k,v in row['roles'].items()},'unionerr',round(row['device_total']['signed_relative_error_pct'],1),'part', {k:round(row['same_nsight_clock_partition'][k]['median_ns'],1) for k in ('kernel_union','full_nvtx','launch_api_union','before_first_kernel','inter_kernel_gap','after_last_kernel','gpu_active','launch_only','sync_only','other_api_only','unattributed_idle')})
if __name__=='__main__':main()
