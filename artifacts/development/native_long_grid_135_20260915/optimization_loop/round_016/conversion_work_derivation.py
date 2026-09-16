"""Source-expression derivation only; stdout JSON, no kernel/LLM timing input."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
ROOT=P.parents[4]
CUDA=ROOT/'source/llama.cpp-semantic/ggml/src/ggml-cuda'
def ref(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def derive(m,k,kind):
 if type(m)!=int or m<=0 or type(k)!=int or k<=0 or k%32:raise ValueError('positive M and K32-multiple required')
 padded=(k+511)//512*512;e=m*padded
 if kind=='MMVQ':
  return {'path':kind,'m':m,'k_logical':k,'k_padded':padded,'grid':[padded//256,m,1],'block':[256,1,1],
   'cta_count':m*padded//256,'source_threads':e,'active_warps':e//32,'actual_nonpadding_load_threads':m*k,'padding_zero_threads':m*(padded-k),
   'logical_read_bytes':4*m*k,'packed_write_bytes':36*e//32,'legacy_scalar_proxy_operations':11*e,
   'expressions':{'fabs':e,'max':5*e,'sum_add':5*e,'shuffle':10*e,'scale_divide_by_constant':e,
                  'variable_divide_upper_bound':e,'roundf_upper_bound':e,'int8_conversion_upper_bound':e,
                  'half_conversions':2*e//32,'int8_stores':e,'half2_metadata_stores':e//32},
   'warp_reduction_rounds_each':5,'dependency_chains':['load->abs->(shuffle,max)x5->scale->quantize/round/cvt->int8 store',
   'load->(shuffle,add)x5->half convert->half2 metadata store'],
   'notes':'zero-activation lanes may bypass variable divide/round; source expressions are not SASS or measured service'}
 if kind not in ('MMQ_D4','MMQ_DS4'):raise ValueError('layout outside audit scope')
 threads=e//4;summed=kind=='MMQ_DS4'
 return {'path':kind,'m':m,'k_logical':k,'k_padded':padded,'grid':[m,padded//512,1],'block':[128,1,1],
   'cta_count':m*padded//512,'source_threads':threads,'active_warps':threads//32,'actual_nonpadding_load_threads':m*k//4,'padding_zero_threads':m*(padded-k)//4,
   'logical_read_bytes':4*m*k,'packed_write_bytes':144*e//128,'legacy_scalar_proxy_operations':(20 if summed else 14)*threads,
   'expressions':{'fabs':4*threads,'max':6*threads,'multiply':4*threads,'sum_add':6*threads if summed else 0,
    'shuffle':(6 if summed else 3)*threads,'divide_or_reciprocal':2*threads,'roundf':4*threads,'int8_conversion':4*threads,
    'char4_stores':threads,'metadata_stores':threads//8,'half_conversions':threads//4 if summed else 0},
   'warp_reduction_rounds_each':3,'dependency_chains':['float4 load->four abs/three local max->(shuffle,max)x3->inverse scale->four quantize/round/cvt->char4 store',
    'optional DS4 local sum->(shuffle,add)x3->half2 metadata store; D4 reciprocal->F32 metadata store'],
   'notes':'D4 F32 metadata differs from DS4 half2; no experts/ids/scatter/batches; source expressions not instruction rates'}
def main():
 anchors={'quantize.cu':['float amax = fabsf(xi);','amax = warp_reduce_max<QK8_1>(amax);','sum  = warp_reduce_sum<QK8_1>(sum);','const float d_inv = 127.0f / amax;','y[ib].d4[iqs/32]  = d;'],
 'quantize.cuh':['#define CUDA_QUANTIZE_BLOCK_SIZE     256','#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128'],
 'common.cuh':['x += __shfl_xor_sync(0xffffffff, x, offset, width);','x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));']}
 sources=[]
 for name,strings in anchors.items():
  p=CUDA/name;s=p.read_text()
  for a in strings:
   if a not in s:raise ValueError('source anchor changed:'+a)
  sources.append({**ref(p),'anchors':[{'text':a,'lines':[i+1 for i,l in enumerate(s.splitlines()) if a in l]} for a in strings]})
 result={'schema':'source-conversion-expression-audit/v1','sources':sources,'gpu_executed':False,'latency_coefficients_derived':False,
  'rows':[derive(m,k,kind) for m in (1,4,64) for k in (896,1024) for kind in ('MMVQ','MMQ_D4','MMQ_DS4')]}
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
