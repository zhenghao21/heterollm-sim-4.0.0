import subprocess, json, hashlib, platform, time
from pathlib import Path
ROOT=Path(r'F:/codex_project/37_LLMsim/heterollm-sim-4.0.0')
exe=ROOT/'source/llama.cpp-semantic/build-semantic-direct/bin/llama-bench.exe'
model=ROOT/'artifacts/multimodel_20260913/models/qwen2.5-0.5b-instruct-q4_k_m.gguf'
out=ROOT/'artifacts/development/kernel_microbench_b05_v1.json'
Mvals=[1,4,16,64]; Tvals=[1,8,32]; reps=3

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
rows=[]; logs=[]
for backend,ngl in [('cpu',0),('gpu',-1)]:
 for m in Mvals:
  for t in Tvals:
   cmd=[str(exe),'-m',str(model),'-pg',f'{m},{t}','-r',str(reps),'-o','json','--no-warmup','-ngl',str(ngl),'-t','16','-b','64','-ub','64']
   ts=time.time(); p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=180); dur=time.time()-ts
   logs.append({'backend':backend,'m':m,'t':t,'returncode':p.returncode,'duration_s':dur,'stderr':p.stderr[-4000:]})
   if p.returncode!=0:
    rows.append({'backend':backend,'m':m,'t':t,'status':'failed','returncode':p.returncode}); continue
   try: data=json.loads(p.stdout)
   except Exception as e:
    rows.append({'backend':backend,'m':m,'t':t,'status':'invalid_json','error':str(e),'stdout_tail':p.stdout[-1000:]}); continue
   matches=[r for r in data if int(r.get('n_prompt',-1))==m and int(r.get('n_gen',-1))==t]
   if not matches:
    rows.append({'backend':backend,'m':m,'t':t,'status':'missing_match','records':len(data)}); continue
   r=matches[-1].copy(); r.update({'backend':backend,'requested_M':m,'requested_T':t,'status':'ok','command':cmd})
   rows.append(r)
artifact={'schema':'kernel-microbench-b05/v1','purpose':'development_only_operator_cost_evidence','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'binary':{'path':str(exe),'sha256':sha(exe)},'model':{'path':str(model),'sha256':sha(model)},'hardware':{'cpu':platform.processor(),'gpu':'NVIDIA GeForce RTX 5080','runtime':'CUDA'},'config':{'threads':16,'batch':64,'ubatch':64,'repetitions':reps,'M_values':Mvals,'T_values':Tvals,'kv_dtype':'f16','model_dtype':'Q4_K_M'},'rows':rows,'run_log':logs,'coverage':{'requested':len(Mvals)*len(Tvals)*2,'ok':sum(r.get('status')=='ok' for r in rows)},'decision':{'status':'blocked','reason':'llama-bench reports aggregate prompt/decode throughput; it does not provide per-operator semantic owner or kernel-level MMQ timing. Data may constrain aggregate trends but cannot enable generic operator cost calibration without semantic trace/microbench kernels.'}}
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(artifact,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps({'out':str(out),'coverage':artifact['coverage'],'decision':artifact['decision']},ensure_ascii=False))
