"""Read-only source-path diagnosis, reproduces synthetic bytes; never fits timings."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import tempfile,subprocess,os
HERE=Path(__file__).resolve().parent
env=os.environ.copy();env["PATH"]=str(Path(json.loads((HERE/"identity_lock.json").read_text())["native_bin"]))+os.pathsep+env["PATH"]
root=Path(__file__).resolve().parents[6]
run=root/'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_008/stream_event_probe_v2/runs/20260915T225012796Z'
results=[]
for K in (896,1024):
 N=4096;M=4;s=20260914;values=np.empty(N*K+M*K,np.float32)
 for i in range(len(values)):
  s^=(s<<13)&0xffffffff;s^=s>>17;s^=(s<<5)&0xffffffff
  values[i]=((s&65535)-32768)/65536.
 w=values[:N*K].reshape(N,K//32,32);x=values[N*K:].reshape(M,K//32,32)
 idx=np.argmax(np.abs(w),axis=2);maximum=np.take_along_axis(w,idx[:,:,None],axis=2)[:,:,0]
 d=maximum/np.float32(-16);inv=np.divide(np.float32(1),d,out=np.zeros_like(d),where=d!=0)
 q=np.minimum(31,np.trunc(w*inv[:,:,None]+np.float32(16.5))).astype(np.uint8)
 dh=d.astype('<f2');qh=np.sum(((q>>4)&1).astype(np.uint32)*(np.uint32(1)<<np.arange(32,dtype=np.uint32)),axis=2,dtype=np.uint32)
 packed=np.empty((N,K//32,22),np.uint8);packed[:,:,:2]=dh.view(np.uint8).reshape(N,K//32,2);packed[:,:,2:6]=qh.astype('<u4').view(np.uint8).reshape(N,K//32,4);packed[:,:,6:]=(q[:,:,:16]&15)|((q[:,:,16:]&15)<<4)
 ad=np.max(np.abs(x),axis=2)/np.float32(127);scaled=np.divide(x,ad[:,:,None],out=np.zeros_like(x),where=ad[:,:,None]!=0)
 aq=(np.sign(scaled)*np.floor(np.abs(scaled)+np.float32(.5))).astype(np.int32)
 sums=x.copy()
 for off in (16,8,4,2,1):sums=(sums+sums[:,:,np.arange(32)^off]).astype(np.float32)
 asum=sums[:,:,0].astype(np.float16).astype(np.float32);adh=ad.astype(np.float16).astype(np.float32)
 for m in (1,2,4):
  path=run/f'{"dev" if K==896 else "validation"}_Q5_0_m{m}_k{K}.event.json';raw=json.loads(path.read_text());numerics=raw['runs'][0]['correctness'];pred=[];actual=[];mathref=[]
  for row in numerics['samples']:
   ni,mi=row['n_index'],row['m_index'];dots=np.sum(q[ni].astype(np.int32)*aq[mi],axis=1,dtype=np.int32)
   block=dh[ni].astype(np.float32)*(dots.astype(np.float32)*adh[mi]-np.float32(16)*asum[mi])
   pred.append(float(np.sum(block,dtype=np.float64)));actual.append(row['actual']);mathref.append(row['reference'])
  with tempfile.TemporaryDirectory() as tmp:
   wp=Path(tmp)/'w.bin';xp=Path(tmp)/'x.bin';wp.write_bytes(packed.tobytes());xp.write_bytes(values[N*K:].tobytes())
   cmd=subprocess.run([str(HERE/'reference-host.exe'),str(wp),str(xp),str(K)],input=''.join(f"{v['n_index']} {v['m_index']}\n" for v in numerics['samples']),text=True,capture_output=True,check=True,env=env)
   host=np.array([float(v) for v in cmd.stdout.splitlines()]);assert len(host)==len(pred)
  host_delta=np.abs(host-np.array(pred));assert host_delta.max()<1e-10,(K,m,float(host_delta.max()))
  delta=np.abs(host-actual);old=np.abs(np.array(mathref)-actual)
  record={'config':path.stem,'host_python_max_difference':float(host_delta.max()),'host_reference_max_actual_difference':float(delta.max()),'tight_path_failures':int(np.count_nonzero(delta>1e-4+1e-5*np.abs(host))),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'weight_sha_matches':hashlib.sha256(packed.tobytes()).hexdigest()==raw['quantization']['packed_weight_sha256'],'input_sha_matches':hashlib.sha256(values[N*K:N*K+m*K].tobytes()).hexdigest()==raw['quantization']['input_sha256'],'sample_count':len(pred),'source_path_max_absolute_error':float(delta.max()),'source_path_rmse':float(np.sqrt(np.mean(delta**2))),'mathematical_max_absolute_error':float(old.max()),'same_tolerance_path_failures':int(np.count_nonzero(delta>.05+.03*np.abs(pred))),'old_failures_retained':True}
  results.append(record);print(json.dumps(record),flush=True)
out=Path(__file__).with_name('host_reference_validation.json')
if out.exists():raise SystemExit('no overwrite')
out.write_text(json.dumps({'status':'source_path_diagnostic_not_new_acceptance','results':results,'limitation':'Block reduction order approximated with double accumulation; not bitwise kernel equivalence. Original failed runs remain rejected. No timings used or changed.'},indent=2))
