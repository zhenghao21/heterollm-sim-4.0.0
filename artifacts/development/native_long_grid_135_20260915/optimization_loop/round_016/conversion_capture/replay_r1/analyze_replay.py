"""CPU decoding of observed owned-buffer bytes; never runs a GPU or changes a gate."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,struct
P=Path(__file__).resolve().parent
M,N,K=64,896,1024
BLOCK_SIZE,Q_OFFSET=144,16
OUTPUT_BYTES=M*(K//128)*BLOCK_SIZE
INPUT_BYTES=M*K*4
WEIGHT_BYTES=N*(K//32)*34
GRAPH_BYTES=M*N*4
SUFFIXES={'quantized':'.q8_1_d4.bin','input':'.input_f32.bin','weights':'.weights_q8_0.bin','graph':'.graph_f32.bin'}

def ref(path):
 path=Path(path).resolve(strict=True)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def fbits(v):return struct.unpack('<I',struct.pack('<f',v))[0]

def offsets(row,k):
 if not 0<=row<M or not 0<=k<K:raise ValueError('D4 decode index out of bounds')
 ib=(k//128)*M+row
 return ib*BLOCK_SIZE+Q_OFFSET+k%128,ib*BLOCK_SIZE+(k%128//32)*4

def decode(data):
 if len(data)!=OUTPUT_BYTES:raise ValueError('full D4 output must contain exactly 73728 bytes')
 rows=[]
 for m in range(M):
  ds=[];qs=[]
  for kb in range(K//128):
   offset=(kb*M+m)*BLOCK_SIZE
   ds.extend(struct.unpack_from('<4f',data,offset));qs.extend(struct.unpack_from('<128b',data,offset+Q_OFFSET))
  rows.append((ds,qs))
 return rows

def cpu_prediction(packed,n,d4,qs):
 if len(packed)!=WEIGHT_BYTES or not 0<=n<N:raise ValueError('original packed weights shape mismatch')
 total=0.0
 for b in range(K//32):
  off=(n*(K//32)+b)*34
  wd=struct.unpack_from('<e',packed,off)[0]
  wq=struct.unpack_from('<32b',packed,off+2)
  dot=sum(int(w)*int(q) for w,q in zip(wq,qs[b*32:b*32+32]))
  # This matches the archived host reference: float32 scale product and contribution,
  # followed by double accumulation. The input codes/scales here are OBSERVED bytes.
  total+=f32(f32(wd*d4[b])*f32(dot))
 return total

def analyze(result_path):
 result_path=Path(result_path).resolve(strict=True)
 metadata=json.loads(result_path.read_text(encoding='utf-8'))
 if metadata.get('status')!='replay_complete' or metadata.get('quantized_code_observed') is not True:raise ValueError('no completed observed replay to analyze')
 baseline=json.loads((P/'sample_baseline.json').read_text(encoding='utf-8'))
 paths={key:Path(str(result_path)+suffix) for key,suffix in SUFFIXES.items()}
 files={key:ref(path) for key,path in paths.items()}
 expected={'quantized':OUTPUT_BYTES,'input':INPUT_BYTES,'weights':WEIGHT_BYTES,'graph':GRAPH_BYTES}
 for key,size in expected.items():
  if files[key]['bytes']!=size:raise ValueError('incomplete observed '+key+' byte file')
 if files['input']['sha256']!=baseline['input_sha256']:raise ValueError('input hash differs from frozen sample')
 if files['weights']['sha256']!=baseline['packed_weight_sha256']:raise ValueError('packed weight hash differs from frozen sample')
 quantized=paths['quantized'].read_bytes();rows=decode(quantized)
 packed=paths['weights'].read_bytes();x=struct.unpack('<'+str(M*K)+'f',paths['input'].read_bytes());graph=struct.unpack('<'+str(M*N)+'f',paths['graph'].read_bytes())
 if any(not math.isfinite(s) for ds,_ in rows for s in ds):raise ValueError('non-finite D4 scale observed; preserve raw bytes for diagnosis')
 if any(not math.isfinite(g) for g in graph):raise ValueError('non-finite original graph output observed')
 qoff,doff=offsets(11,33);d4,qs=rows[11]
 target={'row':11,'k':33,'block_index':(33//128)*M+11,'q_byte_offset':qoff,'scale_byte_offset':doff,'input_f32':x[11*K+33],'q':qs[33],'d4':d4[1],'d4_bits':fbits(d4[1]),'q_raw_hex':quantized[qoff:qoff+1].hex(),'d4_raw_little_endian_hex':quantized[doff:doff+4].hex()}
 ct=metadata['target_row11_k33']
 for key in ('row','k','block_index','q_byte_offset','scale_byte_offset','q','d4_bits'):
  if ct[key]!=target[key]:raise ValueError('independent decoder differs from C++ target observation: '+key)
 if ct['d4']!=target['d4']:raise ValueError('C++/Python scale values differ')
 checks=[]
 for sample in baseline['samples']:
  m,n=sample['m'],sample['n'];scales,codes=rows[m]
  prediction=cpu_prediction(packed,n,scales,codes);actual=graph[m*N+n];tolerance=sample['frozen_tolerance']
  if tolerance!=0.0001+0.00001*abs(sample['stored_reference']):raise ValueError('frozen sample tolerance mismatch')
  checks.append({'m':m,'n':n,'replay_codes_cpu_prediction':prediction,'same_process_original_graph_actual':actual,'archived_operator_actual':sample['actual'],'archived_original_path_reference':sample['stored_reference'],'frozen_tolerance':tolerance,'prediction_minus_same_process_graph':prediction-actual,'prediction_minus_archived_actual':prediction-sample['actual'],'same_process_graph_equals_archived_actual':actual==sample['actual'],'prediction_matches_same_process_graph_under_frozen_tolerance':abs(prediction-actual)<=tolerance,'prediction_matches_archived_actual_under_frozen_tolerance':abs(prediction-sample['actual'])<=tolerance,'same_process_graph_passes_original_path_gate':abs(actual-sample['stored_reference'])<=tolerance})
 q_differences=[];d_differences=[];row_q_diff=Counter();q_count=0;d_count=0
 for m,(scales,codes) in enumerate(rows):
  for b in range(K//32):
   values=x[m*K+b*32:m*K+(b+1)*32];maximum=max(abs(v) for v in values)
   inverse=f32(127.0/maximum) if maximum else 0.0
   host_scale=f32(1.0/inverse) if inverse else 0.0
   if fbits(scales[b])!=fbits(host_scale):
    d_count+=1
    if len(d_differences)<256:d_differences.append({'row':m,'block32':b,'observed':scales[b],'observed_bits':fbits(scales[b]),'CPU_reference':host_scale,'CPU_bits':fbits(host_scale)})
   for lane,value in enumerate(values):
    product=f32(value*inverse)
    cpu_code=int(math.copysign(math.floor(abs(product)+0.5),product))
    k=b*32+lane
    if codes[k]!=cpu_code:
     q_count+=1;row_q_diff[m]+=1
     if len(q_differences)<256:q_differences.append({'row':m,'k':k,'observed_q':codes[k],'CPU_reference_q':cpu_code,'input_f32':value,'CPU_float32_inverse':inverse,'CPU_float32_product':product})
 selected=[{'row':m,'D4_scales_32':rows[m][0],'D4_scale_bits_32':[fbits(v) for v in rows[m][0]],'q_codes_1024':rows[m][1]} for m in (0,10,11,12,63)]
 result={'schema':'observed-owned-D4-replay-CPU-diagnosis/v1','diagnostic_only_not_gate_acceptance':True,'GPU_executed_by_analyzer':False,'no_LLM_actual_read':True,'no_tolerance_change':True,'no_performance_fit':True,'original_pool_read':False,'quantized_code_observation_scope':'Owned output of same-process same runtime func replay; not the original pool allocation.','metadata_ref':ref(result_path),'sample_baseline_ref':ref(P/'sample_baseline.json'),'observed_files':files,'input_and_packed_weight_hashes_match_frozen_case':True,'target_row11_k33':target,'sample_count':len(checks),'max_abs_prediction_minus_same_process_graph':max(abs(c['prediction_minus_same_process_graph']) for c in checks),'max_abs_prediction_minus_archived_actual':max(abs(c['prediction_minus_archived_actual']) for c in checks),'all_same_process_graph_samples_equal_archived_actual':all(c['same_process_graph_equals_archived_actual'] for c in checks),'all_replay_predictions_match_same_process_graph_under_frozen_tolerance':all(c['prediction_matches_same_process_graph_under_frozen_tolerance'] for c in checks),'all_replay_predictions_match_archived_actual_under_frozen_tolerance':all(c['prediction_matches_archived_actual_under_frozen_tolerance'] for c in checks),'same_process_original_path_failed_samples':sum(not c['same_process_graph_passes_original_path_gate'] for c in checks),'checks':checks,'observed_vs_CPU_activation_quantization':{'compared_codes':M*K,'different_code_count':q_count,'different_codes_by_row':dict(sorted(row_q_diff.items())),'first_256_code_differences':q_differences,'code_difference_list_truncated':q_count>256,'compared_scales':M*K//32,'different_scale_bits_count':d_count,'first_256_scale_differences':d_differences,'scale_difference_list_truncated':d_count>256},'selected_rows':selected,'causal_limit':'These are actual replay bytes from the captured runtime function and original DLL, with CPU output prediction. They do not directly reveal the freed original pool bytes or the exact reciprocal device instruction.'}
 return result

def host_test():
 blob=bytearray(OUTPUT_BYTES)
 for m in range(M):
  for k in range(K):
   qo,do=offsets(m,k);struct.pack_into('<b',blob,qo,(m+k)%255-127)
   if k%32==0:struct.pack_into('<f',blob,do,(m*32+k//32+1)/2048)
 decoded=decode(blob)
 for m,(ds,qs) in enumerate(decoded):
  assert qs==[(m+k)%255-127 for k in range(K)]
  assert ds==[(m*32+b+1)/2048 for b in range(K//32)]
 assert offsets(11,33)==(1633,1588)
 try:decode(blob[:-1]);raise AssertionError('short input accepted')
 except ValueError:pass
 for row,k in [(-1,0),(64,0),(0,-1),(0,1024)]:
  try:offsets(row,k);raise AssertionError('out of range accepted')
  except ValueError:pass
 packed=bytearray(WEIGHT_BYTES)
 # Every packed weight block has d=0.5 and 32 codes +1; observed input d=0.25, q=+2.
 # Each block contributes exactly 8 and 32 blocks must sum to 256.
 for b in range(K//32):struct.pack_into('<e32b',packed,b*34,0.5,*([1]*32))
 assert cpu_prediction(packed,0,[0.25]*32,[2]*1024)==256.0
 return {'all_65536_D4_cells_decoded_correctly':True,'target_offsets_verified':True,'short_file_and_out_of_range_rejected':True,'original_packed_weight_CPU_prediction_test':True,'GPU_executed':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--host-test',action='store_true');ap.add_argument('--result',type=Path);a=ap.parse_args()
 if a.host_test:print(json.dumps(host_test()));return
 if a.result is None:raise ValueError('explicit result path required')
 report=analyze(a.result);dest=Path(str(a.result.resolve())+'.analysis.json')
 with dest.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'analysis':ref(dest),'target':report['target_row11_k33'],'sample_count':report['sample_count'],'max_abs_prediction_minus_same_process_graph':report['max_abs_prediction_minus_same_process_graph'],'all_predictions_match':report['all_replay_predictions_match_same_process_graph_under_frozen_tolerance']}))
if __name__=='__main__':main()
