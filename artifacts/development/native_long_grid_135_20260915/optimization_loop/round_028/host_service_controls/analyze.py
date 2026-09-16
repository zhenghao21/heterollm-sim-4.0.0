"""Raw counters never become service ns. ETW scheduling input is explicitly required.
This module does not capture ETW, execute probes, inspect native LLM timings or modify simulator costs.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,math,statistics
HERE=Path(__file__).resolve().parent
PHASES=('empty_submit_loop_before','submit','busy_sync','idle_sync','empty_submit_loop_after')
NODES=(1,4,16);GROUPS=(32,128,512,2048)
def need(ok,why):
 if not ok:raise ValueError(why)
def ref(p):
 p=Path(p).resolve();need(p.is_file(),'missing evidence');return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check_ref(r):
 need(isinstance(r,dict) and set(r)=={'path','bytes','sha256'} and type(r['bytes']) is int and r['bytes']>=0 and isinstance(r['sha256'],str) and len(r['sha256'])==64,'incomplete evidence reference')
 need(ref(r['path'])==r,'evidence identity differs');return r
def load_execution(record_path,sha,mode,observation=None):
 """Require the immutable runner terminal receipt, not just a plausible raw JSON."""
 import build
 path=Path(record_path).resolve();runroot=(HERE/'runs').resolve()
 need(path.name=='record.json' and path.parent.parent==runroot,'raw data must belong to the fixed run root')
 finish_path=path.with_name('finish.json');need(finish_path.is_file(),'runner terminal receipt missing')
 finish=json.loads(finish_path.read_text(encoding='utf8'))
 need(finish.get('schema')=='host-service-execution/v1' and finish.get('status')=='validated' and type(finish.get('returncode')) is int and finish['returncode']==0,'runner did not validate this execution')
 for key in ('inputs_unchanged','process_gates_passed','campaign_gates_passed','raw_validated'):need(finish.get(key) is True,'runner gate missing: '+key)
 need(check_ref(finish.get('record_ref'))==ref(path),'runner/raw binding differs')
 for key,name in [('start_ref','start.json'),('child_ref','child.json')]:need(check_ref(finish.get(key))==ref(path.with_name(name)),'runner evidence path differs')
 start=json.loads(Path(finish['start_ref']['path']).read_text(encoding='utf8'));child=json.loads(Path(finish['child_ref']['path']).read_text(encoding='utf8'))
 before=finish.get('identity_before');after=finish.get('identity_after')
 need(isinstance(before,dict) and before and before==after==start.get('identity_before'),'pre/post identity proof missing')
 check_ref(before['build_ref']);manifest=build.verify_manifest(before['build_ref']['path'])
 need(before['inputs']==manifest['inputs'] and before['compiled']=={k:v['executable'] for k,v in manifest['variants'].items()},'runner/build binding differs')
 need(before.get('loaded_code_refs'),'loaded code identities absent')
 for item in before['loaded_code_refs']:check_ref(item)
 need(finish.get('campaign_before')==finish.get('campaign_after')==start.get('campaign_before') and isinstance(start.get('campaign_before'),dict),'campaign barrier proof missing')
 check_ref(start['campaign_before']['barrier_ref']);check_ref(start['campaign_before']['controls_ref'])
 for item in start['campaign_before']['verifier_refs']:check_ref(item)
 need(start.get('process_before',{}).get('conflicts')==[] and finish.get('process_after',{}).get('conflicts')==[],'process exclusion evidence absent')
 need(start.get('mode')==mode and finish.get('mode')==mode,'execution mode differs')
 need(start.get('observation_requested')==finish.get('observation_requested'),'observation metadata differs')
 if observation is not None:need(finish['observation_requested']==observation,'observation cohort differs')
 begin=datetime.datetime.fromisoformat(start['created_utc']);end=datetime.datetime.fromisoformat(finish['finished_utc'])
 need(begin.tzinfo is not None and end.tzinfo is not None and end>=begin,'execution timestamps missing/reversed')
 record=json.loads(path.read_text(encoding='utf8'));validate_record(record,sha)
 need(child.get('pid')==record['pid'],'raw PID not child PID')
 topology=record.get('topology','chain');need(start.get('topology')==topology and finish.get('topology')==topology,'execution topology differs')
 exe=before['compiled']['pilot' if mode=='pilot' else 'service']['path'];expected=[exe,str(path)] if mode=='pilot' else [exe,mode,topology,str(path)]
 need(start.get('argv')==expected,'raw record not exact executed command')
 index=finish.get('process_index');need(type(index) is int and 1<=index<=(3 if mode=='pilot' else 5),'process budget identity')
 prefix=mode+('_'+finish['observation_requested'] if mode=='service' else '')+'_'+topology
 need(path.parent.name==prefix+'.%04d'%index,'run directory/index binding differs')
 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf8'));need(ref(HERE/'protocol.json')['sha256']==sha,'protocol SHA mismatch')
 if mode=='service':
  need(record.get('GPU_uuid')==protocol['gpu_uuid'],'actual device UUID differs')
  need({x['path']:x['sha256'] for x in record.get('target_modules',[])}=={x['path']:x['sha256'] for x in protocol['target_modules']},'loaded module identity differs')
 return record,dict(finish_ref=ref(finish_path),start_ref=finish['start_ref'],child_ref=finish['child_ref'],build_ref=before['build_ref'])

def stable_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def phase_ok(x,freq=None):
 a,b=x['start'],x['end'];need(all(type(v.get(k)) is int and v[k]>=0 for v in (a,b) for k in ('qpc_before','qpc_after')),'invalid QPC type')
 need(a['qpc_before']<=a['qpc_after']<=b['qpc_before']<=b['qpc_after'],'counter order')
 need(x['body_start_qpc']==a['qpc_after'] and x['body_end_qpc']==b['qpc_before'],'phase marker binding')
 for name in ('user_100ns','kernel_100ns','cycles'):need(type(a[name]) is int and type(b[name]) is int and 0<=a[name]<=b[name],'CPU counter reversal/type')
 
 for field,name in [('user_delta_100ns','user_100ns'),('kernel_delta_100ns','kernel_100ns'),('cycle_delta','cycles')]:need(x.get(field)==b[name]-a[name],'raw delta mismatch')
 need(x.get('snapshot_read_envelope_qpc_ticks')==a['qpc_after']-a['qpc_before']+b['qpc_after']-b['qpc_before'],'counter read envelope mismatch')
 if freq is not None:need(math.isclose(x['wall_ns'],(b['qpc_before']-a['qpc_after'])*1e9/freq,rel_tol=1e-12,abs_tol=1e-9),'QPC wall unit mismatch')
 need(x.get('CPU_service_ns') is None and x.get('counter_precision_validated') is False,'raw counters falsely claim precision')
def validate_record(r,protocol_sha):
 need(r.get('protocol_sha256')==protocol_sha and type(r.get('pid')) is int and type(r.get('tid')) is int and type(r.get('qpc_frequency')) is int and r['qpc_frequency']>0,'record identity')
 need(r.get('status')!='failed','failed probe stays failed')
 if r['schema']=='host-service-counter-pilot/v1':
  need(r['status']=='recorded_precision_unproven' and r.get('GPU_API_called') is False and r.get('cycles_converted_to_ns') is False and r.get('counter_precision_validated') is False,'pilot scope')
  need(len(r['samples'])==65,'complete pilot required')
  expected={(kind,amount,repeat) for repeat in range(5) for kind,amounts in [('empty_loop',(0,128,512,2048)),('busy',(.25,1,4,16,64)),('Sleep',(1,4,16,64))] for amount in amounts}
  need({(x.get('kind'),x.get('amount'),x.get('repeat')) for x in r['samples']}==expected,'pilot control coverage differs')
  start,end=r.get('sampling_start_qpc'),r.get('sampling_finish_qpc')
  need(type(start) is int and type(end) is int and 0<=start<=end and r.get('sampling_budget_ms')==20000 and (end-start)*1000/r['qpc_frequency']<=20000,'pilot sampling budget proof absent/overrun')
  for sample in r['samples']:phase_ok(sample['measurement'],r['qpc_frequency'])
  return
 need(r['schema']=='host-service-probe/v1' and r['mode']=='service' and r['status']=='recorded_ETW_required' and r['profiling'] is False and r['cost_model_applied'] is False and r['GPU_event_used'] is False,'service scope')
 need(r['topology'] in ('chain','fanout') and len(r['cases'])==12 and r['completed_cases']==12,'complete case count')
 need({(c['nodes'],c['groups']) for c in r['cases']}=={(n,g) for n in NODES for g in GROUPS},'case ladder differs')
 for c in r['cases']:
  need(c['kernels']==c['nodes']*c['groups'] and len(c['samples'])==31,'sample/kernel denominator')
  for i,s in enumerate(c['samples']):
   need(s['sample']==i and math.isfinite(s['max_abs_error']) and s['max_abs_error']>=0,'sample/correctness evidence')
   for phase in PHASES:phase_ok(s[phase],r['qpc_frequency'])
def pilot_summary(records,sha):
 need(len(records)==3 and len({r['pid'] for r in records})==3,'three independent pilot processes required')
 for r in records:validate_record(r,sha)
 ticks=[s['measurement']['user_delta_100ns']+s['measurement']['kernel_delta_100ns'] for r in records for s in r['samples']]
 positive=[v for v in ticks if v>0]
 windows=[]
 for kind,amount in sorted({(s['kind'],s['amount']) for r in records for s in r['samples']}):
  group=[s['measurement'] for r in records for s in r['samples'] if(s['kind'],s['amount'])==(kind,amount)]
  cpu=[(x['user_delta_100ns']+x['kernel_delta_100ns'])*100 for x in group];wall=[x['wall_ns'] for x in group]
  ratios=[c/w for c,w in zip(cpu,wall) if w>0]
  windows.append(dict(kind=kind,nominal_amount=amount,samples=len(group),nonzero_accounting_samples=sum(x>0 for x in cpu),median_reported_accounting_ns=statistics.median(cpu),median_wall_ns=statistics.median(wall),median_accounting_to_wall_ratio=statistics.median(ratios) if ratios else None,accounting_cv_pct=cv(cpu) if all(x>0 for x in cpu) else None))
 busy={x['nominal_amount']:x for x in windows if x['kind']=='busy'};tail=[busy[n]['median_accounting_to_wall_ratio'] for n in (16,64)]
 convergence=abs(tail[1]-tail[0])/statistics.median(tail)*100 if all(x is not None and x>0 for x in tail) else None
 return dict(schema='host-service-pilot-summary/v1',raw_sample_count=len(ticks),zero_counter_samples=sum(v==0 for v in ticks),observed_minimum_positive_tick_delta=min(positive) if positive else None,minimum_step_is_error_bound=False,counter_precision_validated=False,cycles_converted_to_ns=False,controlled_windows=windows,long_busy_window_ratio_difference_pct=convergence,conditional_development_use=True,absolute_service_accuracy_guarantee=False,assumptions=['GetThreadTimes is OS user+kernel accounting, not wall time','busy-window accounting/wall ratio is interpretable only with limited descheduling/interrupt work; this pilot does not independently verify that assumption','Sleep controls expose wait-versus-accounting behavior but do not prove service-time precision','minimum nonzero increment is observed visibility, not guaranteed timer resolution'],allowed_followup='use visible stable windows for conditional development diagnostics/cost candidates with explicit scope; formal service precision requires independent validation; no automatic simulator import')
def union_length(parts):
 total=0;end=None
 for lo,hi in sorted(parts):
  if hi<=lo:continue
  if end is None or lo>end:total+=hi-lo;end=hi
  elif hi>end:total+=hi-end;end=hi
 return total

def validate_etw(e,raw_ref,r):
 need(e.get('schema')=='host-service-etw-normalized/v1' and check_ref(e['probe_record_ref'])==raw_ref,'ETW/probe binding')
 check_ref(e['source_etl_ref']);check_ref(e['exporter_ref']);check_ref(e['export_receipt_ref'])
 receipt=json.loads(Path(e['export_receipt_ref']['path']).read_text(encoding='utf8'))
 payload={k:v for k,v in e.items() if k!='export_receipt_ref'}
 need(receipt.get('status')=='complete' and receipt.get('source_etl_ref')==e['source_etl_ref'] and receipt.get('exporter_ref')==e['exporter_ref'] and receipt.get('normalized_payload_sha256')==stable_hash(payload),'ETW exporter execution binding')
 need(e.get('clock_client_context')==1 and e.get('qpc_frequency')==r['qpc_frequency'],'ETW clock is not the same QPC domain')
 for name in ('lost_events','lost_buffers'):need(type(e.get(name)) is int and e[name]==0,'ETW loss rejects service evidence')
 need(e.get('scheduler_coverage')=='complete' and e.get('interrupt_coverage')=='complete','missing CSwitch/DPC/ISR coverage')
 need(type(e.get('coverage_start_qpc')) is int and type(e.get('coverage_end_qpc')) is int and e['coverage_start_qpc']<e['coverage_end_qpc'],'trace coverage')
 scheduled=e.get('scheduled_intervals');interrupts=e.get('interrupt_intervals')
 need(isinstance(scheduled,list) and scheduled and isinstance(interrupts,list),'interval data absent')
 by_cpu={};by_tid={}
 for x in scheduled:
  need(all(type(x.get(k)) is int for k in ('pid','tid','cpu','start_qpc','end_qpc')) and x['start_qpc']<x['end_qpc'],'invalid scheduled interval')
  by_cpu.setdefault(x['cpu'],[]).append((x['start_qpc'],x['end_qpc']));by_tid.setdefault(x['tid'],[]).append((x['start_qpc'],x['end_qpc']))
 for rows in list(by_cpu.values())+list(by_tid.values()):
  rows.sort();need(all(a[1]<=b[0] for a,b in zip(rows,rows[1:])),'overlapping scheduled CPU/thread intervals')
 for x in interrupts:need(x.get('kind') in ('DPC','ISR') and all(type(x.get(k)) is int for k in ('cpu','start_qpc','end_qpc')) and x['start_qpc']<x['end_qpc'],'invalid interrupt interval')

def active_service(e,pid,tid,lo,hi):
 need(e['coverage_start_qpc']<=lo<=hi<=e['coverage_end_qpc'],'phase outside ETW coverage')
 total=0;boundaries=2
 for x in e['scheduled_intervals']:
  if x['pid']!=pid or (tid is not None and x['tid']!=tid):continue
  a,b=max(lo,x['start_qpc']),min(hi,x['end_qpc'])
  if b<=a:continue
  pieces=[(max(a,z['start_qpc']),min(b,z['end_qpc'])) for z in e['interrupt_intervals'] if z['cpu']==x['cpu'] and z['end_qpc']>a and z['start_qpc']<b]
  total+=b-a-union_length(pieces);boundaries+=2+2*len(pieces)
 ns=total*1e9/e['qpc_frequency'];uncertainty=boundaries*1e9/e['qpc_frequency']
 return dict(active_CPU_ns=ns,timestamp_quantization_allowance_ns=uncertainty,allowance_scope='QPC tick quantization only; not a global clock accuracy guarantee')

def attach_etw(record_path,etw_path,sha):
 r,execution=load_execution(record_path,sha,'service','etw');rr=ref(record_path)
 need(r['schema']=='host-service-probe/v1','service record required')
 e=json.loads(Path(etw_path).read_text(encoding='utf8'));validate_etw(e,rr,r)
 cases=[]
 for c in r['cases']:
  samples=[]
  for s in c['samples']:
   measured={}
   for name in PHASES:
    phase=s[name];lo,hi=phase['body_start_qpc'],phase['body_end_qpc']
    caller=active_service(e,r['pid'],r['tid'],lo,hi);process=active_service(e,r['pid'],None,lo,hi)
    measured[name]={**caller,'other_same_process_thread_CPU_ns':max(0,process['active_CPU_ns']-caller['active_CPU_ns']),'other_thread_role_inferred':False,'raw_phase':phase}
   samples.append(dict(sample=s['sample'],phases=measured))
  cases.append(dict(nodes=c['nodes'],groups=c['groups'],kernels=c['kernels'],samples=samples))
 return dict(schema='host-service-etw-attached/v1',probe_record_ref=rr,execution_refs=execution,etw_ref=ref(etw_path),protocol_sha256=sha,topology=r['topology'],pid=r['pid'],cases=cases,counter_precision_validated=False,service_source='scheduled calling-thread time excluding DPC/ISR',simulator_modified=False)

def median(values):return statistics.median(values)
def cv(values):return 100*statistics.stdev(values)/statistics.mean(values) if len(values)>1 and statistics.mean(values)>0 else math.inf
def deviation(values):
 m=median(values);return max(abs(v-m)*100/m for v in values) if m>0 else math.inf

def service_matrix(attached):
 need(len(attached)==5 and len({a['pid'] for a in attached})==5,'full five process attached cohort required')
 results={}
 for key in [(n,g) for n in NODES for g in GROUPS]:
  per=[];all_samples=[]
  for a in attached:
   c=next(x for x in a['cases'] if (x['nodes'],x['groups'])==key);values=[]
   for s in c['samples']:
    p=s['phases'];submit=p['submit'];empty=[p[k]['active_CPU_ns'] for k in ('empty_submit_loop_before','empty_submit_loop_after')]
    raw=submit['active_CPU_ns'];need(raw>0,'zero active service is not a calibrated zero')
    need(submit['timestamp_quantization_allowance_ns']/raw<=.02,'timestamp precision exceeds 2% budget')
    need(max(empty)/raw<=.02,'observer/empty-loop fraction exceeds 2%')
    need(abs(empty[0]-empty[1])/raw<=.02,'empty-control drift')
    values.append(raw-median(empty));all_samples.append(dict(raw_active_ns=raw,empty_before_ns=empty[0],empty_after_ns=empty[1],other_thread_CPU_ns=submit['other_same_process_thread_CPU_ns']))
   need(len(values)==31 and cv(values)<=5,'within-process service instability/coverage');per.append(median(values))
  need(deviation(per)<=5,'between-process service instability');results[key]=dict(service_ns=median(per),process_medians_ns=per,raw_and_empty=all_samples)
 for n in NODES:
  tail=[results[n,g]['service_ns']/g for g in GROUPS[-2:]]
  need(abs(tail[1]-tail[0])*100/median(tail)<=5,'normalized cost not converged; backpressure/domain change')
 return results

def fit3(rows):
 # Allowed operator microbenchmark surface, never model-name or target-LLM fit.
 x=[(1.0,float(n*g),float(g)) for n,g in rows];y=[rows[k]['service_ns'] for k in rows]
 a=[[sum(row[i]*row[j] for row in x) for j in range(3)]+[sum(row[i]*v for row,v in zip(x,y))] for i in range(3)]
 for col in range(3):
  pivot=max(range(col,3),key=lambda r:abs(a[r][col]));need(abs(a[pivot][col])>1e-12,'K/G design rank deficient');a[col],a[pivot]=a[pivot],a[col];q=a[col][col];a[col]=[v/q for v in a[col]]
  for r in range(3):
   if r!=col:q=a[r][col];a[r]=[v-q*w for v,w in zip(a[r],a[col])]
 return [a[i][3] for i in range(3)]

def candidate(development,holdout,direct_records,sha,method):
 need(isinstance(method,dict) and method.get('schema')=='host-service-etw-method-qualification/v1' and method.get('protocol_sha256')==sha,'separate ETW collector/exporter method qualification required')
 need(method.get('scope')=='QPC CSwitch intervals with DPC/ISR exclusion','ETW method scope differs')
 # Frozen disabled until a real collector/exporter and measured method validator exist.
 # A caller-supplied status/boolean and matching file hashes are not accuracy proof.
 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
 need(protocol['etw_interface'].get('method_measurement_admission_enabled') is True,'ETW method admission is disabled in this frozen preparation; measured method validation implementation and a new reviewed freeze are required')
 need(protocol['etw_interface'].get('provided_ETW_capture_or_exporter_already_validated') is True,'ETW method has no independently validated collector/exporter')
 for key in ('collector_configuration_ref','exporter_source_ref','validation_receipt_ref'):check_ref(method[key])
 validation=json.loads(Path(method['validation_receipt_ref']['path']).read_text(encoding='utf8'))
 need(validation.get('schema')=='host-service-etw-exporter-validation/v1' and validation.get('status')=='passed' and validation.get('exporter_source_ref')==method['exporter_source_ref'],'exporter validation binding absent')
 expected={'busy_sleep_runtime_controls','context_switch_migration','DPC_ISR_overlap','lost_event_rejection','QPC_clock_identity','profiling_overhead_pair'}
 need(set(validation.get('passed_checks',[]))==expected,'ETW method has missing checks')
 for item in validation.get('evidence_refs',[]):check_ref(item)
 need(validation.get('evidence_refs'),'ETW runtime method validation evidence missing')
 for a in development+holdout:
  check_ref(a['probe_record_ref']);check_ref(a['etw_ref']);e=json.loads(Path(a['etw_ref']['path']).read_text(encoding='utf8'))
  need(e['exporter_ref']==method['exporter_source_ref'],'different ETW exporter')
  replay=attach_etw(a['probe_record_ref']['path'],a['etw_ref']['path'],sha)
  need(replay==a,'attached service was not rederived from immutable raw events')
 need(all(a['protocol_sha256']==sha for a in development+holdout),'attached protocol differs')
 need(all(a['topology']=='chain' for a in development) and all(a['topology']=='fanout' for a in holdout),'fanout must be held out intact')
 direct={t:[r for r in direct_records if r.get('topology')==t] for t in ('chain','fanout')}
 for t,group in direct.items():
  need(len(group)==5 and len({r['pid'] for r in group})==5,'five independent unprofiled processes per topology required')
  for r in group:validate_record(r,sha)
 # Profiling impact is compared in original wall units, never used as a CPU service conversion.
 for t,attached in [('chain',development),('fanout',holdout)]:
  for n in NODES:
   for g in GROUPS:
    traced=median([median([s['phases']['submit']['raw_phase']['wall_ns'] for s in next(c for c in a['cases'] if c['nodes']==n and c['groups']==g)['samples']]) for a in attached])
    unprofiled=median([median([s['submit']['wall_ns'] for s in next(c for c in r['cases'] if c['nodes']==n and c['groups']==g)['samples']]) for r in direct[t]])
    need(unprofiled>0 and abs(traced-unprofiled)/unprofiled<=.05,'ETW observation effect exceeds 5%')
 dev,test=service_matrix(development),service_matrix(holdout);beta=fit3(dev)
 need(all(math.isfinite(b) and b>=0 for b in beta),'negative/unidentifiable scalar parameters; retain raw surface only')
 validation=[]
 for key,row in test.items():
  n,g=key;prediction=beta[0]+beta[1]*n*g+beta[2]*g;ape=abs(prediction-row['service_ns'])/row['service_ns']*100
  need(ape<=5,'held-out topology exceeds 5%');validation.append(dict(nodes=n,groups=g,ape_pct=ape))
 return dict(schema='host-service-cost-candidate/v1',candidate_qualified=True,cost_model_applied=False,caller_only=True,counter_precision_validated=False,
     service_source='ETW scheduled intervals, DPC/ISR excluded; empty controls recorded',coefficients=dict(fixed_window_ns=beta[0],per_kernel_ns=beta[1],per_graph_call_ns=beta[2]),
     scope=dict(operator='F32 RMS_NORM 4096x1',nodes=list(NODES),groups=list(GROUPS),no_extrapolation=True,no_all_kernel_generalization=True),heldout=validation,
     owner_correction=dict(existing_250ns_semantics='host/API driver submission',existing_250ns_modeled_resource='gpu0.command_queue',gpu_frontend_1000ns_removed=False),
     method_qualification=method,limitations=['parameter uncertainty beyond existing process variability is not a guaranteed confidence interval','scope is caller-only RMS_NORM, not all-kernel or whole-host time'],native_LLM_latency_used=False)

def write_new(p,data):
 with Path(p).open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False)
def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='mode',required=True)
 q=sp.add_parser('pilot-summary');q.add_argument('--records',type=Path,nargs=3,required=True)
 e=sp.add_parser('attach-etw');e.add_argument('--record',type=Path,required=True);e.add_argument('--etw',type=Path,required=True)
 c=sp.add_parser('candidate');c.add_argument('--development',type=Path,nargs=5,required=True);c.add_argument('--holdout',type=Path,nargs=5,required=True);c.add_argument('--direct',type=Path,nargs=10,required=True);c.add_argument('--method-qualification',type=Path,required=True)
 for p in (q,e,c):p.add_argument('--protocol',type=Path,default=Path(__file__).with_name('protocol.json'));p.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();sha=ref(a.protocol)['sha256'];read=lambda p:json.loads(p.read_text(encoding='utf8'))
 if a.mode=='pilot-summary':
  loaded=[load_execution(p,sha,'pilot','direct') for p in a.records];out=pilot_summary([r for r,_ in loaded],sha);out['execution_refs']=[e for _,e in loaded]
 elif a.mode=='attach-etw':out=attach_etw(a.record,a.etw,sha)
 else:
  loaded=[load_execution(p,sha,'service','direct') for p in a.direct];out=candidate([read(p) for p in a.development],[read(p) for p in a.holdout],[r for r,_ in loaded],sha,read(a.method_qualification));out['direct_execution_refs']=[e for _,e in loaded]
 out['input_protocol_ref']=ref(a.protocol);write_new(a.output,out)
if __name__=='__main__':main()
