"""Source-expression/launch geometry derivation only. No timing fit, no GPU, no LLM actual."""
from pathlib import Path
import json,hashlib,math
P=Path(__file__).resolve().parent;ROOT=P.parents[4]
def ref(p):
 p=p.resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def derive(m,k,path):
 if type(m)!=int or m<1 or type(k)!=int or k<1 or k%32:raise ValueError('positive M,K32 aligned')
 kp=math.ceil(k/512)*512;e=m*kp
 if path=='MMVQ':
  t=e;b=m*kp//256
  expr={'abs':t,'float_max':5*t,'float_add':5*t,'shuffle_lane_ops':10*t,'normalization_divisions_nonzero_blocks_upper':2*t,'roundf_nonzero_blocks_upper':t,'int8_cast_nonzero_blocks_upper':t,'fp16_conversion_metadata_values':2*t//32}
  grid=[kp//256,m,1];threads=256;proxy=11*t;writes=36*e//32
 elif path in ('MMQ_D4','MMQ_DS4'):
  t=e//4;b=m*kp//512;summed=path=='MMQ_DS4'
  expr={'abs':4*t,'float_max':6*t,'float_add':6*t if summed else 0,'shuffle_lane_ops':(6 if summed else 3)*t,'normalization_division_expressions':2*t,'multiply':4*t,'roundf':4*t,'int8_cast':4*t,'fp16_conversion_metadata_values':t//4 if summed else 0,'fp32_scale_metadata_stores':0 if summed else t//8}
  grid=[m,kp//512,1];threads=128;proxy=(20 if summed else 14)*t;writes=144*e//128
 else:raise ValueError('unsupported path')
 return {'M':m,'Klogical':k,'Kpadded':kp,'path':path,'grid':grid,'block':[threads,1,1],'CTA_count':b,'launched_threads':t,'launched_warps':t//32,'maximum_simultaneously_occupied_SM_upper_bound_for_84SM':min(b,84),'original_scalar_proxy_operations':proxy,'read_bytes_logical':4*m*k,'write_bytes_physical':writes,'source_expression_counts':expr,'critical_path_notes':'shuffle-reduction dependency chains and division/round/pack latency have no independent cycle rates; expressions are not SASS instructions','no_fitted_duration':True}
def main():
 out=P/'conversion_mechanism_derivation.json'
 if out.exists():raise ValueError('no overwrite')
 rows=[derive(m,k,path) for path in ('MMVQ','MMQ_D4','MMQ_DS4') for m,k in ((1,896),(2,896),(4,896),(64,896),(64,1024))]
 observed=[]
 for directory in (P/'collection_r2/analysis_0001',P/'collection_r3/first_pair_analysis'):
  for file in sorted(directory.rglob('mapped_calls.json')):
   d=json.loads(file.read_text());config=d['config'];ks=[i['kernel'] for c in d['calls'] for i in c['kernel_launch_pairs'] if i['role'].startswith('conversion_')]
   if not ks:continue
   if any(c.get('issues') for c in d['calls']):continue
   path='MMVQ' if config['expected_source_path']=='MMVQ_Q8_1_HALF' else 'MMQ_D4'
   expected=derive(config['M'],config['K'],path)
   geometries={tuple(k[x] for x in ('gridX','gridY','gridZ','blockX','blockY','blockZ')) for k in ks}
   target=tuple(expected['grid']+expected['block'])
   observed.append({'input':ref(file),'config':config,'captured_calls':len(ks),'observed_geometry':[list(g) for g in geometries],'source_geometry':list(target),'geometry_match':geometries=={target},'timing_used':False})
 source=ROOT/'source/llama.cpp-semantic/ggml/src/ggml-cuda'
 evidence=[ref(source/x) for x in ('quantize.cu','quantize.cuh','common.cuh','mmq.cuh')]+[ref(ROOT/'src/heterollm_sim'/x) for x in ('planner.py','mmq_work.py','cost_models.py')]
 out.write_text(json.dumps({'schema':'conversion-source-derivation/v1','rows':rows,'completed_trace_geometry_crosschecks':observed,'sources':evidence,'GPU_runs':0,'LLM_actuals_read':False,'coefficients_fitted':False,'timing_values_consumed':False,'native_source_binary_equivalence_proven':False},indent=2)+'\n')
 print(json.dumps({'source_rows':len(rows),'trace_files_checked':len(observed),'geometry_matches':sum(x['geometry_match'] for x in observed),'timing_values_consumed':False}))
if __name__=='__main__':main()
