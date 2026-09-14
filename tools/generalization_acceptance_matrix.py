"""Run the frozen 5-model generalization acceptance matrix.

The matrix is resumable. A cell is never silently accepted from another
configuration or binary, and aggregation uses the concurrent batch p50 when
parallel > 1 plus absolute millisecond deltas.
"""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, subprocess, sys, time
from pathlib import Path
try:
    from .unified_evidence_manifest import validate_prediction_before_native
except ImportError:  # direct execution or importlib loading by the test runner
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from unified_evidence_manifest import validate_prediction_before_native
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
 'long':' '.join(['Benchmarking a language model requires measuring prompt evaluation and steady state decode separately while preserving identical model, runtime, hardware, and scheduling configuration.']*8),
}
OUTPUTS={'short':8,'medium':32,'long':128}; PARALLEL=(1,2,4)
CTX=2048
PROFILE={
 'qwen25':(ROOT/'artifacts/multimodel_next/qwen25_medium_prompt8_boundary_profile_v3.json',True,False,True),
 'qwen35':(ROOT/'artifacts/multimodel_next/qwen35_hi_prompt8_prefill_scoped_calibration_v1.json',True,False,True),
 'qwen38':(ROOT/'artifacts/multimodel_next/qwen38_cpu_semantic_calibration_prompt8_f9_v2.json',True,True,False),
 'tinyllama':(ROOT/'artifacts/multimodel_next/tinyllama_semantic_calibration_launch8us_locked_v1.json',False,False,False),
 'smollm2':(None,False,False,False),
}
def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def percentile(values,p):
 if not values:return None
 v=sorted(float(x) for x in values); pos=(len(v)-1)*p/100; lo=math.floor(pos); hi=math.ceil(pos)
 return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)
def planned_cells(models,repeats):
 for m in models:
  for pi in PROMPTS:
   for oi in OUTPUTS:
    for parallel in PARALLEL:
     for rep in range(1,repeats+1): yield f'{m}__{pi}__{oi}__p{parallel}__r{rep}',m,pi,oi,parallel,rep
def metric_records(payload):
 """Return one metric record for the concurrent batch represented by payload."""
 native=(payload.get('native') or {}); sim=(payload.get('simulator') or {})
 na=native.get('aggregate') or {}; sa=sim.get('aggregate') or {}
 specs={'ttft_ms':('request_to_first_token_ms','ttft_ms'),'tpot_ms':('tpot_ms','tpot_ms'),'e2e_ms':('request_to_end_ms','e2e_ms')}
 out={}
 for metric,(nk,sk) in specs.items():
  nobj=na.get(nk) or {}; sobj=sa.get(sk) or {}
  n=nobj.get('p50_ms') if isinstance(nobj,dict) else None; s=sobj.get('p50_ms') if isinstance(sobj,dict) else None
  if n is None or s is None:
   n={'ttft_ms':native.get('request_to_first_token_ms'),'tpot_ms':native.get('tpot_ms'),'e2e_ms':native.get('request_to_end_ms')}.get(metric)
   s=sim.get(metric)
  err=100*(float(s)-float(n))/float(n) if n not in (None,0) and s is not None else None
  out[metric]={'native_ms':float(n) if n is not None else None,'simulator_ms':float(s) if s is not None else None,'signed_error_pct':err,'absolute_error_pct':abs(err) if err is not None else None,'absolute_delta_ms':abs(float(s)-float(n)) if n is not None and s is not None else None}
 return out
def validate_payload(payload, *, model, prompt, output, parallel, model_sha, binary_sha, source_path=None):
 checks={}; checks['schema']=payload.get('schema')=='native-simulator-comparison/v2'; checks['geometry']=bool((payload.get('parity') or {}).get('geometry',{}).get('ok')); checks['tokens']=bool((payload.get('parity') or {}).get('tokens',{}).get('ok')); checks['parallel']=int(payload.get('configuration',{}).get('parallel',-1))==parallel; checks['prompt']=payload.get('request',{}).get('prompt')==prompt; checks['model_sha']=((payload.get('gguf') or {}).get('gguf') or {}).get('sha256')==model_sha; checks['binary_sha']=binary_sha==sha(BINARY); checks['stream']='request_boundary_aligned'==payload.get('validity_status') or payload.get('native',{}).get('request_boundary',{}).get('status')=='measured'; checks['prediction_before_native']=not validate_prediction_before_native(payload, source_path=source_path, require=True); return checks,all(checks.values())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'artifacts/multimodel_next/generalization_acceptance_v1.json'); ap.add_argument('--cells-dir',type=Path); ap.add_argument('--models',default=','.join(MODELS)); ap.add_argument('--repeats',type=int,default=3); ap.add_argument('--max-cells',type=int); ap.add_argument('--start-cell',type=int,default=0); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--timeout',type=int,default=900); ap.add_argument('--freeze-manifest',type=Path,default=ROOT/'artifacts/multimodel_next/blind_generalization_freeze_v3.json'); ap.add_argument('--output-mode',choices=('natural','fixed'),default='natural'); args=ap.parse_args()
 selected=[x for x in args.models.split(',') if x]; unknown=set(selected)-set(MODELS)
 if unknown: raise SystemExit(f'unknown models: {sorted(unknown)}')
 freeze_path=args.freeze_manifest.resolve(); freeze=json.loads(freeze_path.read_text(encoding='utf-8')) if freeze_path.exists() else {}
 binary_sha=sha(BINARY) if BINARY.exists() and not args.dry_run else freeze.get('binary_sha256')
 mismatches=[]
 if not args.dry_run:
  for rel,want in (freeze.get('source_sha256') or {}).items():
   p=ROOT/rel
   if p.exists() and sha(p).lower()!=str(want).lower(): mismatches.append(rel)
 if mismatches or binary_sha.lower()!=str(freeze.get('binary_sha256','')).lower():
   raise SystemExit(f'freeze identity mismatch; source={mismatches}, binary={binary_sha}')
 if freeze.get('schema') == 'blind-generalization-freeze/v4' and not args.dry_run:
  verifier=ROOT/'tools/check_freeze_manifest.py'
  check=subprocess.run([sys.executable,str(verifier),'--verify'],capture_output=True,text=True)
  if check.returncode != 0: raise SystemExit(f'v4 freeze verification failed: {check.stdout[-500:]} {check.stderr[-500:]}')
 out=args.cells_dir or args.output.with_suffix(''); Path(out).mkdir(parents=True,exist_ok=True)
 model_sha={m:(sha(MODELS[m]) if MODELS[m].exists() and not args.dry_run else None) for m in selected}; planned=list(planned_cells(selected,args.repeats))
 if str(freeze.get('schema','')).startswith('blind-generalization-freeze/v4'):
  allowed=freeze.get('scenarios') or {}
  planned=[item for item in planned if (item[2],item[3],item[4]) in {(x.get('prompt_band'),x.get('output_band'),int(x.get('parallel'))) for x in allowed.get(item[1], [])}]
 planned=planned[args.start_cell:]; planned=planned[:args.max_cells] if args.max_cells else planned; cells=[]
 for cid,m,pi,oi,parallel,rep in planned:
  model=MODELS[m]; profile,stage,memory,phase=PROFILE[m]; path=Path(out)/(cid+'.json'); meta={'cell_id':cid,'model_key':m,'prompt_band':pi,'output_band':oi,'parallel':parallel,'repeat':rep,'prompt':PROMPTS[pi],'requested_output_tokens':OUTPUTS[oi],'model_path':str(model),'model_sha256':model_sha[m],'binary_sha256':binary_sha,'freeze_manifest':str(freeze_path),'config':{'ctx':CTX,'parallel':parallel,'batch':64,'ubatch':64,'threads':16,'gpu_layers':0 if m=='qwen38' else -1,'request_timing':'stream','output_mode':args.output_mode},'output':str(path)}
  payload=None; returncode=0
  if path.exists() and not args.dry_run:
   try: payload=json.loads(path.read_text(encoding='utf-8'))
   except Exception: payload=None
  elif args.dry_run: meta['status']='planned'; cells.append(meta); continue
  if payload is None:
   cmd=[sys.executable,str(ROOT/'tools/native_llama_compare.py'),'--exe',str(BINARY),'--model',str(model),'--prompt',PROMPTS[pi],'--predict',str(OUTPUTS[oi]),'--ctx',str(CTX),'--parallel',str(parallel),'--batch','64','--ubatch','64','--threads','16','--gpu-layers',str(0 if m=='qwen38' else -1),'--seed','42','--temperature','0','--top-k','1','--warmup-predict','2','--request-timing','stream','--output-mode',args.output_mode,'--output',str(path)]
   if profile and profile.exists(): cmd += ['--calibration-profile',str(profile)] + (['--apply-stage-calibration'] if stage else []) + (['--apply-memory-calibration'] if memory else []) + (['--apply-phase-boundary-calibration'] if phase else [])
   try: cp=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout); returncode=cp.returncode
   except subprocess.TimeoutExpired: returncode=-9
   if path.exists():
    try: payload=json.loads(path.read_text(encoding='utf-8'))
    except Exception: payload=None
  if payload is None: meta.update({'status':'error','returncode':returncode}); cells.append(meta); continue
  checks,ok=validate_payload(payload,model=model,prompt=PROMPTS[pi],output=OUTPUTS[oi],parallel=parallel,model_sha=model_sha[m],binary_sha=binary_sha,source_path=path)
  identity_cfg=(payload.get('identity') or {}).get('configuration') or {}
  expected_gpu=0 if m=='qwen38' else -1
  checks['configuration']=all(identity_cfg.get(k)==v for k,v in {'ctx':CTX,'parallel':parallel,'batch':64,'ubatch':64,'threads':16,'gpu_layers':expected_gpu,'seed':42}.items())
  ok=ok and checks['configuration']
  rec=metric_records(payload); meta.update({'status':'valid' if returncode==0 and ok else 'invalid','returncode':returncode,'checks':checks,'metrics':rec,'validity_status':payload.get('validity_status'),'calibration':{'profile':str(profile) if profile else None,'stage':stage,'memory':memory,'phase_boundary':phase,'request_boundary':False},'actual_token_counts':payload.get('token_counts'),'parallel_support':payload.get('parallel_support'),'output_policy':payload.get('output_policy'),'prediction_artifact':payload.get('prediction_artifact'),'prediction_sha256':payload.get('prediction_sha256'),'identity':{'runtime_fingerprint':payload.get('runtime_fingerprint'),'gguf_sha256':((payload.get('gguf') or {}).get('gguf') or {}).get('sha256'),'hardware_fingerprint':(payload.get('calibration') or {}).get('hardware_fingerprint'),'configuration':identity_cfg}}); cells.append(meta)
 groups={}
 for c in cells:
  if c.get('status')!='valid': continue
  groups.setdefault((c['model_key'],c['prompt_band'],c['output_band'],c['parallel']),[]).append(c)
 aggregation={}
 for key,arr in groups.items():
  aggregation['|'.join(map(str,key))]={'n':len(arr),'metrics':{}}
  for metric in ('ttft_ms','tpot_ms','e2e_ms'):
   vals=[c['metrics'][metric] for c in arr if c['metrics'].get(metric,'').get('signed_error_pct') is not None]; signed=[x['signed_error_pct'] for x in vals]; abs_pct=[x['absolute_error_pct'] for x in vals]; abs_ms=[x['absolute_delta_ms'] for x in vals]
   aggregation['|'.join(map(str,key))]['metrics'][metric]={'n':len(vals),'median_signed_pct':statistics.median(signed) if signed else None,'median_abs_pct':statistics.median(abs_pct) if abs_pct else None,'p90_abs_pct':percentile(abs_pct,90),'worst_abs_pct':max(abs_pct) if abs_pct else None,'median_absolute_ms':statistics.median(abs_ms) if abs_ms else None,'p90_absolute_ms':percentile(abs_ms,90),'worst_absolute_ms':max(abs_ms) if abs_ms else None,'native_median_ms':statistics.median([x['native_ms'] for x in vals]) if vals else None,'simulator_median_ms':statistics.median([x['simulator_ms'] for x in vals]) if vals else None}
 result={'schema':'generalization-acceptance-matrix/v1','freeze_manifest':str(freeze_path),'models':selected,'prompt_bands':PROMPTS,'output_bands':OUTPUTS,'parallel_values':list(PARALLEL),'repeats':args.repeats,'planned_cell_count':len(planned),'cell_count':len(cells),'valid_cell_count':sum(c.get('status')=='valid' for c in cells),'cells':cells,'aggregation':aggregation,'policy':'Frozen blind run. No held-out result may modify profile or source. Concurrent cells aggregate request-level p50; p90/worst and absolute millisecond deltas are retained.','limitations':['Native request boundary is stream client timing and includes HTTP/client overhead; prompt_eval_ms remains separate diagnostic.','Requested output length can terminate early on EOS; actual token counts are retained per cell.','A complete 5x3x3x3 matrix with 3 repeats is 405 executions.']}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'output':str(args.output.resolve()),'planned':len(planned),'cells':len(cells),'valid':result['valid_cell_count'],'groups':len(aggregation)},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())


