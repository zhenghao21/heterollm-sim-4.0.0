"""Audit every raw first/warmup/formal sample; read-only, no coefficient creation."""
from pathlib import Path
import json,math,sys,statistics,hashlib
P=Path(__file__).resolve().parent

def audit(doc,config):
 issues=[]
 def require(ok,message):
  if not ok:issues.append(message)
 require(doc.get('schema')=='single-operator-surface-probe/v2','schema')
 require(doc.get('status')=='measured','status')
 require(doc.get('M')==config['M'] and doc.get('N')==config['N'] and doc.get('K')==config['K'] and doc.get('weight_format')==config['quant'],'shape/quant')
 require(doc.get('input_dtype')=='F32' and doc.get('output_dtype')=='F32' and doc.get('layout')=='ordinary_contiguous_2d','dtype/layout')
 require(doc.get('device')=='cuda' and doc.get('cuda_index')==0 and doc.get('threads')==1,'placement')
 require(doc.get('graph_computations_per_batch')==1 and doc.get('graph_compute_calls')==36,'single-call counts')
 require(doc.get('warmup_requested')==5 and doc.get('formal_repeats_requested')==30,'sample counts')
 require(doc.get('nvtx_scope')=='one full graph call including event records, submit, wait and event queries; excluding cache sweep and numeric validation','full-call NVTX scope')
 require(doc.get('nvtx_enabled') is True and doc.get('expected_source_path')==config['expected_source_path'],'NVTX/source path')
 require(doc.get('actual_dispatch_trace_confirmed') is False,'do not promote expected to observed dispatch')
 require(doc.get('cache_policy')=='untimed_read_write_sweep_at_least_4x_device_L2','cache policy')
 l2,evict=doc.get('gpu_l2_bytes'),doc.get('cache_eviction_bytes')
 require(type(l2)==int and l2>0 and type(evict)==int and evict>=max(128<<20,l2*4),'cache sweep bytes')
 require(doc.get('modules_stable') is True and bool(doc.get('loaded_modules_before')) and doc.get('loaded_modules_before')==doc.get('loaded_modules_after'),'module stability')
 require(doc.get('environment',{}).get('GGML_CUDA_DISABLE_GRAPHS')=='1','graph environment')
 freq=doc.get('qpc_frequency');require(type(freq)==int and freq>0,'QPC frequency')
 control=doc.get('control_mode');require(type(control)==bool,'control mode')
 cc=doc.get('correctness_contract',{})
 require(cc.get('absolute_tolerance')==.05 and cc.get('relative_tolerance')==.03 and cc.get('path_absolute_tolerance')==.0001 and cc.get('path_relative_tolerance')==.00001 and cc.get('reference_mode')=='dual_math_and_source_path' and cc.get('source_runtime_equivalence_proven') is False,'correctness contract')
 runs=doc.get('runs',[]);expected=[('first_call',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]
 require([(r.get('phase'),r.get('index')) for r in runs]==expected,'phase/index order and count')
 def number(x):return type(x) in (int,float) and math.isfinite(x)
 def numeric(c,label):
  if not isinstance(c,dict):issues.append(label+':missing correctness');return
  count=min(4096,config['M']*config['N']);rows=c.get('samples',[])
  require(c.get('finite_all_outputs') is True and c.get('passed') is True,label+':finite/pass')
  require(c.get('sample_count')==count and len(rows)==count,label+':sample count')
  deltas=[];math_deltas=[];math_pass=[]
  for j,r in enumerate(rows):
   if not all(number(r.get(k)) for k in ('actual','reference','math_reference','math_absolute_error','path_reference','path_absolute_error')):issues.append(label+':nonfinite sample');continue
   flat=0 if count==1 else j*(config['M']*config['N']-1)//(count-1)
   require(r.get('n_index')==flat%config['N'] and r.get('m_index')==flat//config['N'],label+':sample position')
   md=abs(r['actual']-r['reference']);pd=abs(r['actual']-r['path_reference']);mp=md<=.05+.03*abs(r['reference']);pp=pd<=.0001+.00001*abs(r['path_reference'])
   require(r['reference']==r['math_reference'] and math.isclose(md,r['math_absolute_error'],abs_tol=1e-12) and math.isclose(pd,r['path_absolute_error'],abs_tol=1e-12),label+':error rederive')
   require(r.get('math_pass') is mp and r.get('path_pass') is pp and r.get('pass') is pp and pp,label+':numeric gates')
   deltas.append(pd);math_deltas.append(md);math_pass.append(mp)
  if deltas:
   require(math.isclose(c.get('path_max_absolute_error',-1),max(deltas),abs_tol=1e-12),label+':path max')
   require(math.isclose(c.get('path_rmse',-1),math.sqrt(sum(d*d for d in deltas)/len(deltas)),abs_tol=1e-12),label+':path rmse')
   require(c.get('math_passed') is all(math_pass),label+':math pass aggregate')
 numeric(doc.get('first_call_correctness'),'first')
 numeric(doc.get('final_correctness'),'final')
 last_end=0
 for r in runs:
  label=f"{r.get('phase')}[{r.get('index')}]"
  expected_label=f"operator_surface/v1|phase={r.get('phase')}|index={r.get('index')}|op=MUL_MAT|M={config['M']}|N={config['N']}|K={config['K']}|quant={config['quant']}|input=F32|output=F32|layout=contiguous2d|expected_path={config['expected_source_path']}"
  require(r.get('nvtx_label')==expected_label,label+':semantic label')
  keys=['qpc_pre_sync_start','qpc_pre_sync_end','qpc_evict_start','qpc_evict_submit_end','qpc_evict_end','qpc_nvtx_push_start','qpc_nvtx_push_end','qpc_start']
  if not control:keys+=['qpc_record_begin_start','qpc_record_begin_end']
  keys+=['qpc_submit_start','qpc_submit_end']
  if not control:keys+=['qpc_record_end_start','qpc_record_end_end']
  keys+=['qpc_wait_start','qpc_wait_end','qpc_end','qpc_nvtx_pop_start','qpc_nvtx_pop_end','qpc_validation_start','qpc_validation_end']
  ticks=[r.get(k) for k in keys];valid=all(type(t)==int and t>0 for t in ticks)
  require(valid and ticks==sorted(ticks) and ticks[0]>=last_end,label+':QPC absolute order')
  if valid:last_end=ticks[-1]
  require(r.get('graph_computations')==1 and all(r.get(k)==0 for k in ('ggml_status','cuda_submit_status','cuda_wait_status','eviction_status')),label+':status/count')
  if type(freq)==int and freq>0 and valid:
   wall=(r['qpc_end']-r['qpc_start'])*1e9/freq
   require(number(r.get('host_wall_ns')) and math.isclose(r['host_wall_ns'],wall,rel_tol=1e-8,abs_tol=1e-3),label+':wall rederive')
   require(r.get('host_per_graph_ns')==r.get('host_wall_ns'),label+':single-call wall')
  if control:
   require(r.get('event_envelope_ms') is None and all(r.get(k)==0 for k in ('qpc_record_begin_start','qpc_record_begin_end','qpc_record_end_start','qpc_record_end_end')),label+':control excludes events')
  else:
   require(number(r.get('event_envelope_ms')) and r['event_envelope_ms']>0 and all(r.get(k)==0 for k in ('cuda_begin_record_status','cuda_end_record_status','cuda_query_after_wait','cuda_elapsed_status')) and r.get('cuda_query_before_wait') in (0,600),label+':event validity')
  numeric(r.get('correctness'),label)
 return {'valid_raw':not issues,'issues':sorted(set(issues)),'sample_count':len(runs),'calibration_eligible':False,'reason':'Actual dispatch and profiling perturbation must be checked separately before cost calibration.'}

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('raw');ap.add_argument('--config',required=True);ap.add_argument('--output');a=ap.parse_args()
 config=next(c for c in json.loads((P/'protocol.json').read_text())['configs'] if c['id']==a.config);p=Path(a.raw);doc=json.loads(p.read_text(encoding='utf-8-sig'));result=audit(doc,config);result['raw']={'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 if a.output:
  out=Path(a.output)
  if out.exists():raise SystemExit('Refuse output overwrite')
  out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result));raise SystemExit(0 if result['valid_raw'] else 2)
if __name__=='__main__':main()
