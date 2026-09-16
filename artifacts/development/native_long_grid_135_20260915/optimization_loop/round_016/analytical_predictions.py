"""Premeasurement analytical predictions; accepts only static synthetic protocol."""
from pathlib import Path
from dataclasses import replace, asdict
from datetime import datetime, timezone
import argparse, hashlib, json, math, sys
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.cost_models import GemmWorkload,TensorKernelWorkload,estimate_gpu_gemm,estimate_gpu_tensor_kernel
from heterollm_sim.mmq_work import derive_mmq_work,MMVQ_MAX_BATCH_SIZE
from heterollm_sim.projection_descriptors import ARTIFACT_QUANTIZATION_REGISTRY
P=Path(__file__).resolve().parent

def ref(path):
 path=Path(path).resolve();b=path.read_bytes()
 return {'path':str(path),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}

def profiles(clock_mhz=2400):
 if type(clock_mhz) not in (int,float) or not math.isfinite(clock_mhz) or clock_mhz<=0:
  raise ValueError('clock_mhz must be finite and positive')
 base=build_reference_scenario();gpu=base.component_profiles['gpu']['legacy-gpu'];tc=gpu.tensor_core
 cycles=84*4*2.617e9*(2*tc.mma_m*tc.mma_n*tc.mma_k)*.5/112.6e12
 gpu=replace(gpu,name='RTX5080-analytical',tensor_core=replace(tc,sm_count=84,frequency_ghz=clock_mhz/1000,cycles_per_mma=cycles))
 memory=replace(base.component_profiles['hbm']['legacy-hbm'],bandwidth_gb_s=960.)
 return gpu,memory

def phase_summary(est):
 device=[p for p in est.phases if p.name!='kernel_launch']
 return {'device_ns':sum(p.service_ns for p in device),'launch_ns':sum(p.service_ns for p in est.phases if p.name=='kernel_launch'),
         'phases':[asdict(p) for p in est.phases],'metadata':dict(est.metadata),'device_elapsed_interpretation':'analytical phase service; not measured GPU occupancy'}

def initial_workload(config):
 m,n,k,fmt=(config[f] for f in ('M','N','K','quant'))
 if any(type(v) is not int or v<=0 for v in (m,n,k)) or fmt not in ('Q5_0','Q8_0'):
  raise ValueError('Only positive synthetic Q5_0/Q8_0 shapes are supported')
 spec=ARTIFACT_QUANTIZATION_REGISTRY[fmt]
 if k%spec.block_size:raise ValueError('K must be an exact physical quantization-block multiple')
 blocks=n*(k//spec.block_size)
 # Match planner._layer_gemm's materialized projection path.  The inherited
 # analytical compute dtype is a16; physical native hidden/output storage is
 # F32.  It is not proof that MMVQ uses FP16 tensor cores on the device.
 # weight_storage_bytes excludes metadata: GemmWorkload.weight_bytes adds it.
 return GemmWorkload(m,k,n,activation_bits=16,weight_bits=spec.compute_weight_bits,output_bits=32,
     activation_storage_bytes=4*m*k,accumulator_bits=32,
     packed_weight_formats=(fmt,),packed_weight_transform_operations=blocks*spec.block_size*spec.dequant_operations_per_weight,
     packed_weight_format_segments=((fmt,n,blocks*spec.block_size*spec.dequant_operations_per_weight),),
     weight_metadata_bytes=blocks*spec.metadata_bytes,weight_storage_bytes=blocks*spec.payload_bytes)

def predict(config,gpu,memory):
 load=initial_workload(config);m,n,k,fmt=load.m,load.n,load.k,config['quant'];padded=((k+511)//512)*512
 initial=load
 pure=phase_summary(estimate_gpu_gemm(gpu,memory,load))
 stages={};work=None
 if m<=MMVQ_MAX_BATCH_SIZE[fmt]:
  conv=TensorKernelWorkload(operations=11*m*padded,read_bytes=4*m*k,write_bytes=36*m*padded//32,streaming_fraction=1.,name='llama_cpp_mmvq_f32_to_q8_1')
  load=replace(load,activation_storage_bytes=36*m*k//32);family='MMVQ'
 else:
  work=derive_mmq_work(m=m,k=k,n=n,weight_format=fmt,sm_count=gpu.sm_count,shared_memory_per_block=101376)
  conv=TensorKernelWorkload(operations=work.conversion_operations,read_bytes=work.conversion_read_bytes,write_bytes=work.conversion_write_bytes,streaming_fraction=1.,name='llama_cpp_mmq_f32_input_repacking')
  load=replace(load,activation_storage_bytes=work.consumer_unique_bytes,mmq_work=work);family='MMQ'
 stages['conversion']={**phase_summary(estimate_gpu_tensor_kernel(gpu,memory,conv)),'workload':asdict(conv)}
 stages['main']={**phase_summary(estimate_gpu_gemm(gpu,memory,load)),'workload':asdict(load)}
 if work is not None and work.fixup_launch:
  fix=TensorKernelWorkload(operations=work.fixup_operations,read_bytes=work.fixup_read_bytes,write_bytes=work.fixup_write_bytes,
      streaming_fraction=1.,name='llama_cpp_mmq_partial_result_fixup',launch_only=work.fixup_operations==0)
  stages['fixup']={**phase_summary(estimate_gpu_tensor_kernel(gpu,memory,fix)),'workload':asdict(fix)}
 return {'config':config,'initial_workload':asdict(initial),
     'compute_storage_contract':{'analytical_activation_bits':initial.activation_bits,'physical_input_storage_bits':32,
       'physical_output_storage_bits':32,'weight_payload_bytes':initial.weight_storage_bytes,
       'weight_metadata_bytes':initial.weight_metadata_bytes,'total_physical_weight_bytes':initial.weight_bytes,
       'generic_math_dtype_is_inherited_fallback_not_native_instruction_proof':True,
       'declared_quantized_capability':gpu.resolve_quantized_matmul_capability(initial) is not None},
     'role_prediction_status':'source_conditional_MMQ_roles' if family=='MMQ' else 'source_conversion_plus_generic_MMVQ_main_fallback',
     'predicted_family':family,'actual_dispatch_verified':False,'generic_analytical_main_only':pure,'source_qualified_roles':stages,
     'device_total_ns':sum(s['device_ns'] for s in stages.values()),'launch_total_ns':sum(s['launch_ns'] for s in stages.values()),
     'llm_prediction':False,'profile_used':False,'scope':'Current mechanism at explicit synthetic shape. Does not include host submission or synchronize. Source/runtime equivalence remains conditional.'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise ValueError('Predictions are immutable; choose a new path')
 protocol_ref=ref(a.protocol);protocol=json.loads(a.protocol.read_text(encoding='utf-8-sig'));gpu,mem=profiles()
 configs=protocol['configs']
 if len(configs)!=26 or len({c['id'] for c in configs})!=26:raise ValueError('Expected 26 unique preregistered shapes')
 shapes={(c['quant'],c['M'],c['N'],c['K']) for c in configs}
 wanted={(q,m,n,896) for q in ('Q5_0','Q8_0') for m in (1,4,64) for n in (128,896,4864)}
 wanted|={(q,m,1792,896) for q in ('Q5_0','Q8_0') for m in (2,32)}
 wanted|={(q,m,896,1024) for q in ('Q5_0','Q8_0') for m in (4,64)}
 if shapes!=wanted:raise ValueError('Joint-shape set differs from preregistered 26-shape domain')
 sources=[Path(__file__),ROOT/'tools/native_llama_compare.py',ROOT/'src/heterollm_sim/reference.py',ROOT/'src/heterollm_sim/cost_models.py',ROOT/'src/heterollm_sim/mmq_work.py',ROOT/'src/heterollm_sim/planner.py',ROOT/'src/heterollm_sim/projection_descriptors.py',P/'test_analytical_predictions.py']
 hardware_ref=ref(P.parent/'operator_microbench_v2/driver_device_properties.json')
 source_refs=[ref(p) for p in sources];rows=[predict(c,gpu,mem) for c in protocol['configs']]
 if source_refs!=[ref(p) for p in sources] or protocol_ref!=ref(a.protocol) or hardware_ref!=ref(hardware_ref['path']):raise ValueError('Source or static input changed during prediction')
 result={'schema':'premeasurement-synthetic-analytical-predictions/v1','created_utc':datetime.now(timezone.utc).isoformat(),
   'protocol_ref':protocol_ref,'sources':source_refs,'hardware_device_ref':hardware_ref,
   'effective_profiles':{'gpu':asdict(gpu),'memory':asdict(mem)},'target_sm_clock_mhz':2400,'cache_assumption':'stateless_compulsory_GEMM_IO',
   'uses_target_llm_timings':False,'kernel_surface_used':False,'predictions':rows,
   'limits':['Inherited structural efficiency/cache parameters are analytical assumptions, not measurements.',
   'MMVQ main retains generic fp16 tensor/scalar proxy with F32 storage; exact integer instruction mix is not resolved.',
   'Packed weight payload and metadata are separate; GemmWorkload adds metadata exactly once.',
   'Only device-kernel intervals are comparable; kernel launch demand and host envelopes reported separately.']}
 with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'predictions':len(rows),'output':str(a.output.resolve()),'sha256':ref(a.output)['sha256']}))
if __name__=='__main__':main()
