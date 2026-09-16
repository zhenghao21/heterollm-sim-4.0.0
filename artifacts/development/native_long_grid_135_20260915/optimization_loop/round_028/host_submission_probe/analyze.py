"""Host-only validation/statistics; no execution and no cost-model ingestion."""
import json,math,statistics
from pathlib import Path
PHASES=('construction','backend_allocation','submit_batch','synchronize')
def require(ok,why):
 if not ok:raise ValueError(why)
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
def validate_record(record,protocol,mode,topology,protocol_sha):
 require(record.get('schema')=='host-submission-target-probe/v1','schema')
 require(record.get('mode')==mode and record.get('topology')==topology,'mode/topology')
 require(record.get('protocol_sha256')==protocol_sha and record.get('GPU_uuid')==protocol['gpu_uuid'],'protocol/hardware identity')
 require(record.get('status')==('path_qualified' if mode=='path' else 'timing_recorded_unadmitted'),'run not successful')
 require(record.get('profiling') is (mode=='path') and record.get('cost_model_admitted') is False and record.get('LLM_graph_reuse_internal_observed') is False,'qualification boundary')
 actual={x['path']:x['sha256'] for x in record.get('target_modules',[])}
 expected={x['path']:x['sha256'] for x in protocol['target_modules']}
 require(actual==expected,'loaded target modules differ')
 cases=record.get('cases',[])
 require(record.get('completed_cases')==3 and [x.get('nodes') for x in cases]==[1,4,16],'complete graph coverage')
 for case in cases:
  n=case['nodes']
  if mode=='path':
   require(case.get('executed_graph_nodes')==n and case.get('captured_launch_count')==n and case.get('qualified') is True,'node/kernel count')
   launches=case.get('launches',[]);require(len(launches)==n,'launch evidence incomplete')
   outputs=[];stream=None;first_input=None
   for i,x in enumerate(launches):
    require(x.get('index')==i and x.get('symbol')==protocol['kernel_symbol'] and (x.get('api_id'),x.get('api_name')) in ((211,'cudaLaunchKernel'),(430,'cudaLaunchKernelExC')),'kernel/API')
    require(x.get('attributes_source_qualified') is True,'launch attribute contract')
    require(x.get('geometry_observed') is True and x.get('grid')==[1,1,1] and x.get('block')==[1024,1,1] and x.get('shared_bytes')==128,'launch geometry')
    require(x.get('ncols')==4096 and x.get('exit_seen') is True and x.get('return_code')==0,'kernel return/shape')
    require(type(x.get('function')) is int and x['function']>0 and type(x.get('stream')) is int and x['stream']>=0,'unobserved function/stream')
    require(type(x.get('input')) is int and x['input']>0 and type(x.get('output')) is int and x['output']>0,'unobserved pointers')
    if i==0:stream=x['stream'];first_input=x['input']
    require(x['stream']==stream,'multiple streams unsupported')
    require(x['input']==(outputs[-1] if topology=='chain' and i else first_input),'graph dependence differs')
    require(x['output'] not in outputs and x['output']!=first_input,'in-place or shared outputs unsupported')
    outputs.append(x['output'])
   require(finite(case.get('max_abs_error')),'missing numerical verification')
  else:
   samples=case.get('samples',[]);require(len(samples)==31 and [x.get('sample') for x in samples]==list(range(31)),'timing sample coverage')
   for sample in samples:
    require(sample.get('graph_replays')==128 and sample.get('pure_GPU_service_ms') is None,'replay/envelope boundary')
    require(finite(sample.get('GPU_event_envelope_ms_includes_host_sync_gaps')) and finite(sample.get('max_abs_error')),'invalid event/correctness record')
    for phase in PHASES:
     row=sample[phase];require(finite(row.get('wall_ns')) and row.get('thread_CPU_service_inferred') is False,'false service inference')
     for key in ('thread_user_100ns_ticks','thread_kernel_100ns_ticks','thread_cycles'):require(type(row.get(key)) is int and row[key]>=0,'bad CPU counter')
     enough=row['thread_user_100ns_ticks']+row['thread_kernel_100ns_ticks']>=100
     require(row.get('thread_CPU_counter_threshold_met') is enough,'CPU counter resolution claim differs')
 return True

def variability(values):
 require(values and all(finite(x) for x in values),'invalid sample')
 med=statistics.median(values);mean=statistics.mean(values)
 cv=100*statistics.stdev(values)/mean if len(values)>1 and mean>0 else None
 deviation=max(abs(x-med)*100/med for x in values) if med>0 else None
 return {'count':len(values),'median':med,'sample_cv_pct':cv,'max_abs_deviation_from_median_pct':deviation}
def summarize_processes(records,protocol,protocol_sha,topology):
 require(len(records)==5,'five independent process records required')
 require(len({r.get('pid') for r in records})==5,'distinct process IDs required')
 for r in records:validate_record(r,protocol,'timing',topology,protocol_sha)
 result={}
 for case_index,n in enumerate((1,4,16)):
  phases={}
  for phase in PHASES:
   runs=[variability([r['cases'][case_index]['samples'][i][phase]['wall_ns'] for i in range(31)]) for r in records]
   between=variability([x['median'] for x in runs])
   stable=all(x['sample_cv_pct'] is not None and x['sample_cv_pct']<=5 for x in runs) and between['max_abs_deviation_from_median_pct'] is not None and between['max_abs_deviation_from_median_pct']<=5
   cpu_resolved=all(s[phase]['thread_CPU_counter_threshold_met'] for r in records for s in r['cases'][case_index]['samples'])
   phases[phase]={'within_process_wall':runs,'between_process_medians':between,'stable_wall_under_preregistered_rule':stable,'calling_thread_CPU_counter_threshold_met_all_samples':cpu_resolved,'thread_CPU_precision_validated':False,
    'thread_CPU_raw_100ns_ticks':variability([s[phase]['thread_user_100ns_ticks']+s[phase]['thread_kernel_100ns_ticks'] for r in records for s in r['cases'][case_index]['samples']]),
    'thread_cycles_raw':variability([s[phase]['thread_cycles'] for r in records for s in r['cases'][case_index]['samples']]),'service_cost_admitted':False}
  phases['GPU_event_envelope_ms_includes_host_sync_gaps']={'processes':[variability([s['GPU_event_envelope_ms_includes_host_sync_gaps'] for s in r['cases'][case_index]['samples']]) for r in records],'pure_GPU_service':None}
  result[str(n)]=phases
 return {'schema':'host-submission-statistics/v1','topology':topology,'fixed_process_count':5,'fixed_samples_each':31,'nodes':result,'outliers_removed':0,'cost_model_admitted':False,'GPU_event_is_envelope_not_kernel_service':True}
