"""Frozen generalization matrix runner for 5 models x 3 prompt x 3 output x 3 parallel.
Each cell is independently executed in a fresh llama-server.  The runner is
resumable and never changes calibration based on held-out results.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe'
MODELS={
 'qwen25': ROOT/'artifacts/multimodel_20260913/models/qwen2.5-0.5b-instruct-q4_k_m.gguf',
 'qwen35': ROOT/'artifacts/multimodel_20260913/models/Qwen3.5-0.8B-Q4_K_M.gguf',
 'qwen38': ROOT/'artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf',
 'tinyllama': ROOT/'artifacts/multimodel_20260913/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
 'smollm2': ROOT/'artifacts/multimodel_20260913/models/smollm2-1.7b-instruct-q4_k_m.gguf',
}
PROMPTS={
 'short':'Hi.',
 'medium':'Explain how deterministic benchmarking affects reproducibility in language model inference.',
 'long':' '.join(['Benchmarking a language model requires measuring prompt evaluation and steady state decode separately while keeping model, runtime, hardware, and scheduling configuration identical.']*8),
}
OUTPUTS={'short':8,'medium':32,'long':128}
PARALLEL=(1,2,4)
PROFILE={
 'qwen25': (ROOT/'artifacts/multimodel_next/qwen25_medium_prompt8_boundary_profile_v3.json', True, False, True),
 'qwen35': (ROOT/'artifacts/multimodel_next/qwen35_hi_prompt8_prefill_scoped_calibration_v1.json', True, False, True),
 'qwen38': (ROOT/'artifacts/multimodel_next/qwen38_cpu_semantic_calibration_prompt8_f9_v2.json', True, True, False),
 'tinyllama': (None, False, False, False),
 'smollm2': (None, False, False, False),
}
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def percentile(v,p):
 if not v:return None
 x=sorted(float(a) for a in v); pos=(len(x)-1)*p/100; lo=math.floor(pos); hi=math.ceil(pos)
 return x[lo] if lo==hi else x[lo]+(x[hi]-x[lo])*(pos-lo)
def cell_ids(models,prompts,outputs,pars,reps):
 for m in models:
  for pi in prompts:
   for oi in outputs:
    for par in pars:
     for rep in range(1,reps+1): yield f'{m}__{pi}__{oi}__p{par}__r{rep}',m,pi,oi,par,rep
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'artifacts/multimodel_next/generalization_matrix_v1.json'); ap.add_argument('--cells-dir',type=Path,default=None); ap.add_argument('--models',default=','.join(MODELS)); ap.add_argument('--repeats',type=int,default=3); ap.add_argument('--max-cells',type=int,default=None); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--timeout',type=int,default=900); args=ap.parse_args()
 selected=[x for x in args.models.split(',') if x]; bad=set(selected)-set(MODELS)
 if bad: raise SystemExit(f'unknown models: {sorted(bad)}')
 out=args.cells_dir or args.output.with_suffix(''); out=Path(out); out.mkdir(parents=True,exist_ok=True)
 freeze=ROOT/'artifacts/multimodel_next/blind_generalization_freeze_v1.json'
 freeze_data=json.loads(freeze.read_text(encoding='utf-8')) if freeze.exists() else {}
 cells=[]; planned=list(cell_ids(selected,PROMPTS,OUTPUTS,PARALLEL,args.repeats)); planned=planned[:args.max_cells] if args.max_cells else planned
 for cid,m,pi,oi,par,rep in planned:
  model=MODELS[m]; profile,stage,memory,phase=PROFILE[m]; path=out/(cid+'.json'); meta={'cell_id':cid,'model_key':m,'prompt_band':pi,'output_band':oi,'parallel':par,'repeat':rep,'prompt':PROMPTS[pi],'requested_output_tokens':OUTPUTS[oi],'model_path':str(model),'model_sha256':None if args.dry_run else (sha(model) if model.exists() else None),'binary_sha256':None if args.dry_run else (sha(BINARY) if BINARY.exists() else None),'freeze_manifest':str(freeze),'config':{'ctx':512,'parallel':par,'batch':64,'ubatch':64,'threads':16,'gpu_layers':0 if m=='qwen38' else -1,'request_timing':'stream'},'output':str(path)}
  if path.exists() and not args.dry_run:
   try: payload=json.loads(path.read_text(encoding='utf-8')); meta.update({'status':'valid' if payload.get('parity',{}).get('geometry',{}).get('ok') else 'invalid','payload':payload}); cells.append(meta); continue
   except Exception: pass
  if args.dry_run: meta['status']='planned'; cells.append(meta); continue
  cmd=[sys.executable,str(ROOT/'tools/native_llama_compare.py'),'--exe',str(BINARY),'--model',str(model),'--prompt',PROMPTS[pi],'--predict',str(OUTPUTS[oi]),'--ctx','512','--parallel',str(par),'--batch','64','--ubatch','64','--threads','16','--gpu-layers',str(0 if m=='qwen38' else -1),'--seed','42','--temperature','0','--top-k','1','--warmup-predict','2','--request-timing','stream','--output',str(path)]
  if profile and profile.exists(): cmd += ['--calibration-profile',str(profile)] + (['--apply-stage-calibration'] if stage else []) + (['--apply-memory-calibration'] if memory else []) + (['--apply-phase-boundary-calibration'] if phase else [])
  started=time.perf_counter()
  try: cp=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout); meta['returncode']=cp.returncode; meta['wall_s']=time.perf_counter()-started
  except subprocess.TimeoutExpired as e: meta.update({'returncode':-9,'wall_s':time.perf_counter()-started,'error':'timeout'}); cells.append(meta); continue
  if path.exists():
   try:
    payload=json.loads(path.read_text(encoding='utf-8')); meta.update({'status':'valid' if payload.get('parity',{}).get('geometry',{}).get('ok') else 'invalid','payload':payload})
   except Exception as e: meta.update({'status':'invalid','error':str(e)})
  else: meta.update({'status':'error','stderr':cp.stderr[-2000:]})
  cells.append(meta)
 groups={}
 for c in cells:
  if c.get('status')!='valid': continue
  p=c['payload']; er=p.get('relative_error_pct') or {}; key=(c['model_key'],c['prompt_band'],c['output_band'],c['parallel'])
  groups.setdefault(key,[]).append(er)
 agg={}
 for key,arr in groups.items():
  vals={}
  for metric in ('ttft_ms','tpot_ms','e2e_ms'):
   signed=[float(x[metric]) for x in arr if x.get(metric) is not None]; absolute=[abs(x) for x in signed]
   vals[metric]={'n':len(signed),'median_signed_pct':statistics.median(signed) if signed else None,'median_abs_pct':statistics.median(absolute) if absolute else None,'p90_abs_pct':percentile(absolute,90),'worst_abs_pct':max(absolute) if absolute else None}
  agg['|'.join(map(str,key))]=vals
 result={'schema':'generalization-matrix/v1','freeze_manifest':str(freeze),'models':selected,'prompt_bands':list(PROMPTS),'output_bands':list(OUTPUTS),'parallel_values':list(PARALLEL),'repeat_count':args.repeats,'planned_cell_count':len(planned),'cell_count':len(cells),'cells':cells,'aggregation':agg,'policy':'No held-out result may change profiles during this run; request timing uses stream when available, while prompt_eval diagnostics remain separate.'}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'output':str(args.output.resolve()),'planned':len(planned),'cells':len(cells),'valid':sum(x.get('status')=='valid' for x in cells),'aggregated_groups':len(agg)},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
