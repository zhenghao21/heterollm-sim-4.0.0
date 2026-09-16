"""Completed-collection QPC/NVML cadence audit; ignores numerical sample payloads and all LLM data."""
from pathlib import Path
import json,mmap,re,statistics,hashlib,datetime,collections
P=Path(__file__).resolve().parent;COLLECTION=P/'collection_r3'
def load(p):return json.loads(p.read_text())
def ref(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def quantile(v,q):
 v=sorted(v)
 if not v:return None
 i=(len(v)-1)*q;lo=int(i);return v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(i-lo)
def stats(v):
 if not v:return {'count':0}
 return {'count':len(v),'minimum':min(v),'median':statistics.median(v),'p90':quantile(v,.9),'p99':quantile(v,.99),'maximum':max(v)}
def headers(path):
 # Intentionally parse only36 tiny timing prefixes; 46MB numerical payload is not needed for this audit.
 rows=[]
 with path.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
  for marker in re.finditer(rb'\{"phase":"(first_call|warmup|formal)","index":\d+,',data):
   end=data.find(b',"correctness":',marker.start())
   if end<0 or end-marker.start()>5000:raise ValueError('timing prefix schema changed')
   rows.append(json.loads(data[marker.start():end]+b'}'))
 return rows
def audit_process(receipt_path):
 directory=receipt_path.parent;receipt=load(receipt_path);telemetry_path=directory/'telemetry.json';app_path=directory/'microbench.json'
 if not app_path.exists() or not telemetry_path.exists():return None
 freq=receipt['qpc_frequency'];telemetry=load(telemetry_path);samples=[s for s in telemetry['samples'] if type(s.get('qpc_ticks')) is int];samples.sort(key=lambda s:s['qpc_ticks'])
 intervals=[(b['qpc_ticks']-a['qpc_ticks'])*1000/freq for a,b in zip(samples,samples[1:])];rows=headers(app_path);formal=[r for r in rows if r['phase']=='formal'];brackets=[]
 for row in formal:
  left=[(i,s) for i,s in enumerate(samples) if s['qpc_ticks']<=row['qpc_start']];right=[(i,s) for i,s in enumerate(samples) if s['qpc_ticks']>=row['qpc_end']]
  if not left or not right:brackets.append({'index':row['index'],'missing':True});continue
  li,ls=left[-1];ri,rs=right[0];before=(row['qpc_start']-ls['qpc_ticks'])*1000/freq;after=(rs['qpc_ticks']-row['qpc_end'])*1000/freq
  brackets.append({'index':row['index'],'before_ms':before,'after_ms':after,'bracket_ms':(rs['qpc_ticks']-ls['qpc_ticks'])*1000/freq,'samples_inside':sum(row['qpc_start']<=x['qpc_ticks']<=row['qpc_end'] for x in samples[li:ri+1]),'within25ms':max(before,after)<=25,'before_qpc':ls['qpc_ticks'],'after_qpc':rs['qpc_ticks']})
 qpc_keys=[k for k in formal[0] if k.startswith('qpc_')] if formal else []
 validation=[(r['qpc_validation_end']-r['qpc_validation_start'])*1000/freq for r in formal]
 gaps=[(b['qpc_start']-a['qpc_end'])*1000/freq for a,b in zip(formal,formal[1:])]
 wall=[r['host_wall_ns'] for r in formal]
 return {'config':directory.parents[1].name,'pair':directory.parent.name,'mode':directory.name,'process_status':receipt['status'],'process_exit':receipt.get('returncode'),'sample_count':len(samples),'telemetry_error_samples':sum('error' in s for s in telemetry['samples']),
  'requested_period_ms':5,'start_to_start_sample_intervals_ms':stats(intervals),'interval_ge_25ms':sum(x>25 for x in intervals),'interval_14_to20ms_fraction':sum(14<=x<=20 for x in intervals)/max(len(intervals),1),'interval_4_to8ms_fraction':sum(4<=x<=8 for x in intervals)/max(len(intervals),1),
  'formal_wall_ns':stats(wall),'formal_wall_p90_p10':quantile(wall,.9)/quantile(wall,.1) if wall else None,'formal_validation_ms':stats(validation),'formal_intercall_gap_ms':stats(gaps),'formal_brackets':brackets,'formal_bracket_failures':sum(b.get('missing',False) or not b.get('within25ms',False) for b in brackets),
  'SM_clock_values':sorted({s.get('sm_mhz',{}).get('value') for s in samples if s.get('sm_mhz',{}).get('status')==0}),'original_clock_gate':receipt.get('clock_readback_gate'),
  'input_refs':[ref(receipt_path),ref(telemetry_path)],'app_timing_prefix_input':ref(app_path),'numeric_payload_parsed':False}
def main():
 output=P/'telemetry_stability_audit.json'
 if output.exists():raise ValueError('refuse overwrite')
 rows=[]
 for receipt in sorted(COLLECTION.glob('runs/*/pair_*/*/complete.json')):
  if receipt.parent.name not in ('profile','direct'):continue
  row=audit_process(receipt)
  if row:rows.append(row)
 target=[r for r in rows if r['config']=='train_Q8_0_m4_n4864_k896']
 result={'schema':'r16-telemetry-stability-audit/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_collection_only':True,'GPU_runs':0,'LLM_actuals_read':False,'quality_changed':False,'processes':len(rows),'by_mode':{mode:{'processes':len([r for r in rows if r['mode']==mode]),'median_process_sample_interval_ms':stats([r['start_to_start_sample_intervals_ms']['median'] for r in rows if r['mode']==mode]),'median_validation_ms':stats([r['formal_validation_ms'].get('median',0) for r in rows if r['mode']==mode]),'formal_host_p90_p10':stats([r['formal_wall_p90_p10'] for r in rows if r['mode']==mode and r['formal_wall_p90_p10']]),'processes_failing25ms':sum(r['formal_bracket_failures']>0 for r in rows if r['mode']==mode)} for mode in ('profile','direct')},'target_config':target,'rows':rows,
 'inference_limit':'Cadence includes NVML calls plus Event.wait and scheduling; no wait begin/end or NVML per-field timestamps captured. Cannot prove15.625ms timerquantum or exact scheduler/GIL rootcause from aggregate sample starts alone.',
 'source_refs':[ref(COLLECTION/'worker.py'),ref(COLLECTION/'common.py'),ref(P.parent/'round_015/trace_2026_5/pilot.py')]}
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'processes':len(rows),'by_mode':result['by_mode'],'target_bracket_failures':[x for r in target for x in r['formal_brackets'] if not x.get('within25ms',False)]}))
if __name__=='__main__':main()
