"""CPU-only independent references plus read-only old numeric evidence. No timings fitted."""
from pathlib import Path
import numpy as np, json,subprocess,tempfile,os,hashlib,datetime
P=Path(__file__).resolve().parent
ROOT=P.parents[5]
env=os.environ.copy();env['PATH']=json.loads((P/'identity_lock.json').read_text())['native_bin']+os.pathsep+env.get('PATH','')
def call(packed,x,k,q,path,pairs):
 with tempfile.TemporaryDirectory() as tmp:
  w=Path(tmp)/'w';v=Path(tmp)/'x';w.write_bytes(packed.tobytes());v.write_bytes(x.astype('<f4').tobytes())
  r=subprocess.run([str(P/'reference-host.exe'),str(w),str(v),str(k),q,path],input=''.join(f'{n} {m}\n' for n,m in pairs),capture_output=True,text=True,env=env,check=True)
  out=np.array([float(v) for v in r.stdout.splitlines()]);assert len(out)==len(pairs)
  return out

def reference(packed,x,k,q,path,pairs,return_blocks=False):
 width=22 if q=='Q5_0' else 34;p=packed.reshape(-1,k//32,width);xb=x.reshape(-1,k//32,32).astype(np.float32)
 wd=p[:,:,:2].copy().view('<f2').reshape(p.shape[:2]).astype(np.float32)
 if q=='Q5_0':
  lo=p[:,:,6:];hi=p[:,:,2:6].copy().view('<u4').reshape(p.shape[:2]);w=np.concatenate([lo&15,lo>>4],axis=2).astype(np.int32)+(((hi[:,:,None]>>np.arange(32,dtype=np.uint32))&1)<<4).astype(np.int32)
 else:w=p[:,:,2:].copy().view(np.int8).astype(np.int32)
 maximum=np.max(np.abs(xb),axis=2)
 if path=='MMVQ':
  d=maximum/np.float32(127);scaled=np.divide(xb,d[:,:,None],out=np.zeros_like(xb),where=d[:,:,None]!=0);scale=d.astype(np.float16).astype(np.float32)
 else:
  inv=np.divide(np.float32(127),maximum,out=np.zeros_like(maximum),where=maximum!=0);scaled=(xb*inv[:,:,None]).astype(np.float32);scale=np.divide(np.float32(1),inv,out=np.zeros_like(inv),where=inv!=0)
 aq=(np.sign(scaled)*np.floor(np.abs(scaled)+np.float32(.5))).astype(np.int32)
 sums=xb.copy()
 for off in [16,8,4,2,1]:sums=(sums+sums[:,:,np.arange(32)^off]).astype(np.float32)
 original_sum_half=sums[:,:,0].astype(np.float16).astype(np.float32)
 out=[];blocks=[]
 for n,m in pairs:
  if q=='Q5_0' and path=='MMVQ':dot=np.sum(w[n]*aq[m],axis=1,dtype=np.int32);v=wd[n]*(dot.astype(np.float32)*scale[m]-np.float32(16)*original_sum_half[m])
  else:dot=np.sum((w[n]-16 if q=='Q5_0' else w[n])*aq[m],axis=1,dtype=np.int32);v=(wd[n]*scale[m])*dot.astype(np.float32)
  blocks.append(v);out.append(float(np.sum(v,dtype=np.float64)))
 return (np.array(out),blocks) if return_blocks else np.array(out)

def synthetic_packed(k,q,seed=1947):
 rng=np.random.default_rng(seed);n=16;b=k//32;width=22 if q=='Q5_0' else 34
 p=np.empty((n,b,width),np.uint8);d=rng.uniform(-.04,.04,(n,b)).astype('<f2');p[:,:,:2]=d.view(np.uint8).reshape(n,b,2)
 if q=='Q5_0':
  w=rng.integers(0,32,(n,b,32),dtype=np.uint8);hi=np.sum(((w>>4)&1).astype(np.uint32)*(np.uint32(1)<<np.arange(32,dtype=np.uint32)),axis=2,dtype=np.uint32);p[:,:,2:6]=hi.astype('<u4').view(np.uint8).reshape(n,b,4);p[:,:,6:]=(w[:,:,:16]&15)|((w[:,:,16:]&15)<<4)
 else:p[:,:,2:]=rng.integers(-127,128,(n,b,32),dtype=np.int8).view(np.uint8)
 x=rng.uniform(-.5,.5,(64,k)).astype(np.float32);x[0]=0;x[1]=np.float32(.333251953125);x[2]=np.tile(np.array([.5,-.499,.03125,-.031249],np.float32),k//4)
 return p,x
def main():
 results=[];roundoff=[];paths_differ=False
 for k in (32,896,1024):
  for q in ('Q5_0','Q8_0'):
   packed,x=synthetic_packed(k,q);pairs=[(n,m) for n in range(16) for m in (0,1,2,3,7,15,31,63)]
   path_results={}
   for path in ('MMVQ','MMQ'):
    expected,blocks=reference(packed,x,k,q,path,pairs,True);actual=call(packed,x,k,q,path,pairs);delta=np.max(np.abs(expected-actual));assert delta<1e-10,(k,q,path,float(delta))
    order_max=0.;max_ratio=0.
    for ref,bs in zip(expected,blocks):
     for order in (bs,bs[::-1]):
      sequential=np.float32(0)
      for v in order:sequential=np.float32(sequential+v)
      e=abs(float(sequential)-ref);order_max=max(order_max,e);max_ratio=max(max_ratio,e/(1e-4+1e-5*abs(ref)))
    assert max_ratio<=1,(k,q,path,max_ratio)
    results.append({'K':k,'quant':q,'path':path,'samples':len(pairs),'cpp_python_max_abs_error':float(delta),'max_tested_float_reduction_difference':order_max,'path_tolerance_fraction':max_ratio})
    path_results[path]=actual
   paths_differ=paths_differ or bool(np.max(np.abs(path_results['MMVQ']-path_results['MMQ']))>1e-4)
 assert paths_differ
 # Source-dispatch reference supports M1/2/4 and M16/32/64; no GPU access is needed.
 protocol=json.loads((P/'protocol.json').read_text());assert len(protocol['configs'])==26
 assert len({c['id'] for c in protocol['configs']})==26
 assert [sum(c['group']==g for c in protocol['configs']) for g in ('training','validation','aligned_control')]==[18,4,4]
 # Negative tests: malformed scope is rejected by the host reference executable.
 for args in [('31','Q5_0','MMVQ'),('32','F16','MMQ'),('32','Q8_0','INVALID')]:
  r=subprocess.run([str(P/'reference-host.exe'),'absent','absent',*args],env=env,capture_output=True);assert r.returncode!=0
 out={'schema':'operator-surface-host-reference-validation/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed_host_only','gpu_access':False,'native_timing_used':False,'path_difference_assertion':paths_differ,'samples_total':sum(r['samples'] for r in results),'results':results,'limitation':'Tests exercise exact source arithmetic and selected alternate float reduction orders, not actual GPU dispatch, an exhaustive numerical bound, or runtime equivalence.'}
 (P/'host_reference_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'samples':out['samples_total'],'max_cpp_python_error':max(r['cpp_python_max_abs_error'] for r in results),'max_tolerance_fraction':max(r['path_tolerance_fraction'] for r in results)}))

if __name__=="__main__":main()
