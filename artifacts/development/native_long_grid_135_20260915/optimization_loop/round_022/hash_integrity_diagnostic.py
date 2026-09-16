"""Bounded diagnostics of two recorded SHA failures; no prediction retry or evidence mutation."""
from pathlib import Path
import argparse,ctypes,datetime,hashlib,json,os,subprocess,sys
import _sha2
HERE=Path(__file__).resolve().parent
PDF=HERE/'cta_issue/evidence/nvidia-rtx-blackwell-gpu-architecture.pdf'
PDF_SHA='906ff2a409d7a7e4cbc56f5d3a179d574120d19aaba99520670e1a0c064595fa'
MODEL=HERE.parents[1]/'gpu_extension/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.readonly.gguf'
MODEL_SHA='157479b0083662c7c9cfd8754cae24ab9cc1abbd4def91d49758512c90f1e406'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def fingerprint(p):
 s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
def worker(core):
 kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
 kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 ok=bool(kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),1<<core))
 rows=[];before=fingerprint(PDF)
 for trial in range(16):
  payload=PDF.read_bytes();a=hashlib.sha256(payload).hexdigest();b=_sha2.sha256(payload).hexdigest()
  row={'trial':trial,'bytes':len(payload),'openssl_sha256':a,'hacl_sha256':b,'matches_expected':a==b==PDF_SHA}
  if not row['matches_expected']:
   out=HERE/f'hash_mismatch_cpu{core}_trial{trial}.pdf'
   with out.open('xb') as stream:stream.write(payload)
   row['captured_mismatch_payload']=str(out)
  rows.append(row)
 return {'cpu':core,'affinity_set':ok,'before':before,'after':fingerprint(PDF),'trials':rows}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
 if args.worker is not None:print(json.dumps(worker(args.worker)));return
 plan={'schema':'bounded-hash-diagnostic-plan/v1','created_utc':now(),'cpus':list(range(32)),'pdf_trials_per_cpu':16,'algorithms':['OpenSSL via hashlib','HACL via _sha2'],'model_full_read_passes':1,'target_llm_run':False,'prediction_retry':False,'existing_evidence_overwritten':False}
 write_new(HERE/'hash_integrity_protocol.json',plan)
 processes=[(cpu,subprocess.Popen([sys.executable,__file__,'--worker',str(cpu)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)) for cpu in plan['cpus']]
 results=[]
 for cpu,p in processes:
  stdout,stderr=p.communicate()
  results.append({'cpu':cpu,'returncode':p.returncode,'result':json.loads(stdout) if p.returncode==0 else None,'stderr':stderr.decode('utf-8',errors='replace')})
 before=fingerprint(MODEL);a=hashlib.sha256();b=_sha2.sha256();count=0
 with MODEL.open('rb') as stream:
  for payload in iter(lambda:stream.read(4*1024*1024),b''):
   a.update(payload);b.update(payload);count+=len(payload)
 model={'before':before,'after':fingerprint(MODEL),'bytes_read':count,'openssl_sha256':a.hexdigest(),'hacl_sha256':b.hexdigest(),'matches_expected':a.hexdigest()==b.hexdigest()==MODEL_SHA}
 failures=[{'cpu':r['cpu'],'trial':t} for r in results if r['result'] for t in r['result']['trials'] if not t['matches_expected']]
 result={'schema':'bounded-hash-diagnostic-result/v1','started_utc':plan['created_utc'],'finished_utc':now(),'pdf_processes':results,'pdf_bad_trials':failures,'model':model,'original_prediction_failures_preserved':True,'conclusion_scope':'Present rechecks only; success cannot erase historical failed reads or prove the machine never produced transient errors.'}
 write_new(HERE/'hash_integrity_result.json',result)
 print(json.dumps({'pdf_trials':sum(len(r['result']['trials']) for r in results if r['result']),'pdf_bad_trials':len(failures),'worker_errors':sum(r['returncode']!=0 for r in results),'model':model}))
if __name__=='__main__':main()
