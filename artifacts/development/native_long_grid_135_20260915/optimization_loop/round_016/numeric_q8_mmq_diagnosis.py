"""Bounded CPU sample arithmetic; no model payload, GPU, DLL, test or fitting invocation."""
from pathlib import Path
import hashlib,json,collections
import numpy as np
P=Path(__file__).resolve().parent

def main():
 root=P/'collection_r3/runs/aligned_Q8_0_m64_n896_k1024/pair_01'
 doc=json.loads((root/'direct/microbench.json').read_text());profile=json.loads((root/'profile/microbench.json').read_text())
 state=20260914;values=np.empty(896*1024+64*1024,np.float32)
 for i in range(len(values)):
  state^=(state<<13)&0xffffffff;state^=state>>17;state^=(state<<5)&0xffffffff;state&=0xffffffff;values[i]=((state&65535)-32768)/65536
 w=values[:896*1024].reshape(896,32,32);x=values[896*1024:].reshape(64,32,32)
 ws=np.abs(w).max(2)/np.float32(127);wi=np.float32(1)/ws;scaled=w*wi[:,:,None]
 wq=(np.sign(scaled)*np.floor(np.abs(scaled).astype(np.float64)+.5)).astype(np.int8)
 packed=np.empty((896,32,34),np.uint8);packed[:,:,:2]=ws.astype('<f2').view(np.uint8).reshape(896,32,2);packed[:,:,2:]=wq.view(np.uint8)
 input_hash=hashlib.sha256(x.tobytes()).hexdigest();weight_hash=hashlib.sha256(packed.tobytes()).hexdigest()
 assert input_hash==doc['quantization']['input_sha256'];assert weight_hash==doc['quantization']['packed_weight_sha256']
 maximum=np.abs(x).max(2);inverse=np.float32(127)/maximum;scaled=x*inverse[:,:,None]
 aq=(np.sign(scaled)*np.floor(np.abs(scaled).astype(np.float64)+.5)).astype(np.int32);scales=np.float32(1)/inverse;wd=ws.astype(np.float16).astype(np.float32)
 rows=doc['runs'][0]['correctness']['samples'];bad=[r for r in rows if not r['path_pass']]
 checks=[]
 for row in rows:
  if row['m_index']!=11:continue
  n,m=row['n_index'],row['m_index'];dot=(wq[n].astype(np.int32)*aq[m]).sum(1,dtype=np.int32)
  reference=float(((wd[n]*scales[m])*dot.astype(np.float32)).sum(dtype=np.float64))
  alternative_q=aq[m].copy();alternative_q[1,1]=63
  alternative_dot=(wq[n].astype(np.int32)*alternative_q).sum(1,dtype=np.int32)
  alternative=float(((wd[n]*scales[m])*alternative_dot.astype(np.float32)).sum(dtype=np.float64))
  checks.append({'n':n,'m':m,'actual':row['actual'],'stored_reference':row['path_reference'],'host_reference':reference,
   'host_reference_reproduction_abs':abs(reference-row['path_reference']),'alternative_one_code_63_reference':alternative,
   'residual_after_one_code_change':row['actual']-alternative,'observed_actual_minus_reference':row['actual']-reference,
   'predicted_delta_one_code_change':alternative-reference,'frozen_tolerance':.0001+.00001*abs(row['path_reference'])})
 inv=inverse[11,1];lower=np.nextafter(inv,np.float32(-np.inf));higher=np.nextafter(inv,np.float32(np.inf))
 result={'schema':'q8-mmq-bounded-arithmetic-diagnosis/v1','no_GPU_or_native_execution':True,'no_tolerance_changes':True,'no_fit':True,
  'input_sha256_match':input_hash,'packed_weight_sha256_match':weight_hash,'runs_each':len(doc['runs']),
  'all_direct_profile_outputs_identical':all(a['correctness']['samples']==b['correctness']['samples'] for a,b in zip(doc['runs'],profile['runs'])),
  'failed_samples_per_run':len(bad),'failed_M_rows':dict(collections.Counter(r['m_index'] for r in bad)),
  'all_runs_same_path_max_error':sorted({r['correctness']['path_max_absolute_error'] for r in doc['runs']}),
  'sensitive_input':{'M_index':11,'K_index':33,'block':1,'within_block':1,'x':float(x[11,1,1]),'amax':float(maximum[11,1]),
   'host_d_inv':float(inv),'host_product':float(scaled[11,1,1]),'host_roundf_code':int(aq[11,1,1]),
   'one_ulp_lower_d_inv':float(lower),'product_with_one_ulp_lower_inverse':float(np.float32(x[11,1,1]*lower)),
   'one_ulp_higher_d_inv':float(higher),'product_with_one_ulp_higher_inverse':float(np.float32(x[11,1,1]*higher))},
  'row11_sample_count':len(checks),'max_host_reference_reproduction_error':max(r['host_reference_reproduction_abs'] for r in checks),
  'max_residual_after_one_code_change':max(abs(r['residual_after_one_code_change']) for r in checks),
  'all_changed_rows_within_original_tolerance':all(abs(r['residual_after_one_code_change'])<=r['frozen_tolerance'] for r in checks),
  'checks':checks,'causal_limit':'One altered activation code explains all row11 outputs; GPU conversion buffer/instruction was not captured, so actual device reciprocal instruction cause remains hypothesis, not observed proof.'}
 with (P/'numeric_q8_mmq_diagnosis.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':main()
