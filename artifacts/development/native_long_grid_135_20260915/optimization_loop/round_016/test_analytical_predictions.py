"""Static synthetic comparisons only; never starts native/GPU or reads LLM actuals."""
from dataclasses import asdict,replace
from pathlib import Path
import importlib.util,json,math,sys
import pytest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r16_analytical_predictions',P/'analytical_predictions.py')
analytical=importlib.util.module_from_spec(spec);spec.loader.exec_module(analytical)
from heterollm_sim import planner
from heterollm_sim.ir import LayerSpec
from heterollm_sim.cost_models import estimate_gpu_gemm,estimate_gpu_tensor_kernel
from heterollm_sim.projection_descriptors import ARTIFACT_QUANTIZATION_REGISTRY,WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA
from heterollm_sim.mmq_work import MMVQ_MAX_BATCH_SIZE
CONFIGS=json.loads((P/'operator_probe/r3/protocol.json').read_text(encoding='utf-8-sig'))['configs']

def planner_workload(config):
 k,n,fmt=config['K'],config['N'],config['quant'];s=ARTIFACT_QUANTIZATION_REGISTRY[fmt]
 metadata={'artifact_quantization':fmt,'weight_projection_descriptors':{
  'schema_version':WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA,'projections':{'p':{'segments':[{
   'segment_id':'synthetic','physical_tensor_name':'synthetic_weight','k':k,'n':n,'format':fmt,
   'physical_bytes':n*(k//s.block_size)*(s.payload_bytes+s.metadata_bytes),'tp_shard_axis':'replicated'}]}}}}
 layer=LayerSpec(layer_id='synthetic',kind='dense',hidden_size=k,intermediate_size=n,attention_heads=1,
                 dtype='fp16',quantization=fmt,metadata=metadata)
 return planner._layer_gemm(layer,config['M'],k,n,name='gemm',projection_id='p',f32_storage=True)

@pytest.mark.parametrize('config',CONFIGS,ids=lambda c:c['id'])
def test_initial_workload_exactly_matches_planner_materialized_projection(config):
 actual=analytical.initial_workload(config);expected=planner_workload(config)
 assert actual==expected
 assert actual.activation_bits==16 and actual.activation_bytes==4*actual.m*actual.k
 assert actual.output_bits==32 and actual.output_bytes==4*actual.m*actual.n
 s=ARTIFACT_QUANTIZATION_REGISTRY[config['quant']];blocks=actual.n*(actual.k//s.block_size)
 assert actual.weight_storage_bytes==blocks*s.payload_bytes
 assert actual.weight_metadata_bytes==blocks*s.metadata_bytes
 assert actual.weight_bytes==blocks*(s.payload_bytes+s.metadata_bytes)

@pytest.mark.parametrize('config',CONFIGS,ids=lambda c:c['id'])
def test_all26_compute_paths_generate_separate_finite_role_predictions(config):
 gpu,memory=analytical.profiles();row=analytical.predict(config,gpu,memory)
 assert row['compute_storage_contract']['declared_quantized_capability'] is False
 assert row['generic_analytical_main_only']['metadata']['tensor_dtype']=='fp16'
 assert row['generic_analytical_main_only']['metadata']['quantized_format_coverage']=='generic_quantized_fallback'
 assert row['actual_dispatch_verified'] is False and row['profile_used'] is False
 roles=row['source_qualified_roles']
 assert all(math.isfinite(r['device_ns']) and r['device_ns']>=0 for r in roles.values())
 assert row['device_total_ns']==sum(r['device_ns'] for r in roles.values())
 assert row['launch_total_ns']==sum(r['launch_ns'] for r in roles.values())
 for r in roles.values():
  assert r['launch_ns']==sum(max(d['service_ns'] for d in p['demands']) for p in r['phases'] if p['name']=='kernel_launch')
  assert r['device_ns']==sum(max(d['service_ns'] for d in p['demands']) for p in r['phases'] if p['name']!='kernel_launch')
 if config['M']<=MMVQ_MAX_BATCH_SIZE[config['quant']]:
  assert row['predicted_family']=='MMVQ' and set(roles)=={'conversion','main'}
  assert roles['main']['metadata']['tensor_dtype']=='fp16'
  assert roles['main']['metadata']['quantized_format_coverage']=='generic_quantized_fallback'
  assert 'fallback' in row['role_prediction_status']
 else:
  assert row['predicted_family']=='MMQ'
  assert roles['main']['metadata']['tensor_dtype']=='int8'
  assert roles['main']['metadata']['quantized_format_coverage']=='source_qualified_mmq'

@pytest.mark.parametrize('config',[c for c in CONFIGS if c['M']<=4],ids=lambda c:c['id'])
def test_mmvq_conversion_and_main_match_planner_helpers(config):
 gpu,memory=analytical.profiles();base=analytical.build_reference_scenario()
 scenario=replace(base,workload=replace(base.workload,metadata={**base.workload.metadata,'llama_cpp_f32_q8_1_mmvq':True}))
 original=planner_workload(config)
 conversion=planner._mmvq_activation_conversion_workload(scenario,original,max_m=MMVQ_MAX_BATCH_SIZE[config['quant']])
 row=analytical.predict(config,gpu,memory);roles=row['source_qualified_roles']
 assert roles['conversion']['workload']==asdict(conversion)
 expected_main=replace(original,activation_storage_bytes=36*original.m*original.k//32)
 assert roles['main']['workload']==asdict(expected_main)
 assert roles['main']['device_ns']==analytical.phase_summary(estimate_gpu_gemm(gpu,memory,expected_main))['device_ns']
 assert roles['conversion']['device_ns']==analytical.phase_summary(estimate_gpu_tensor_kernel(gpu,memory,conversion))['device_ns']


def test_profiles_keep_nativebuilder_structure_with_only_declared_2400mhz():
 base=analytical.build_reference_scenario();original=base.component_profiles['gpu']['legacy-gpu'];gpu,memory=analytical.profiles()
 assert gpu.tensor_core.sm_count==84 and gpu.tensor_core.frequency_ghz==2.4
 tc=original.tensor_core
 expected_cycles=84*4*2.617e9*(2*tc.mma_m*tc.mma_n*tc.mma_k)*.5/112.6e12
 assert gpu.tensor_core.cycles_per_mma==expected_cycles
 assert replace(gpu,name=original.name,tensor_core=original.tensor_core)==original
 assert memory.bandwidth_gb_s==960 and replace(memory,bandwidth_gb_s=base.component_profiles['hbm']['legacy-hbm'].bandwidth_gb_s)==base.component_profiles['hbm']['legacy-hbm']
 assert gpu.quantized_matmul_capabilities==()

@pytest.mark.parametrize('value',[0,-1,True,float('nan'),float('inf')])
def test_invalid_clocks_rejected(value):
 with pytest.raises(ValueError):analytical.profiles(value)

@pytest.mark.parametrize('change',[{'M':True},{'N':0},{'K':897},{'quant':'Q5_K'}])
def test_outside_synthetic_workload_contract_rejected(change):
 with pytest.raises(ValueError):analytical.initial_workload({**CONFIGS[0],**change})


def test_main_cli_static_predictions_are_immutable_and_protocol_bound(tmp_path,monkeypatch,capsys):
 output=tmp_path/'predictions.json';protocol=P/'operator_probe/r3/protocol.json'
 monkeypatch.setattr(sys,'argv',['analytical_predictions.py','--protocol',str(protocol),'--output',str(output)])
 analytical.main();data=json.loads(output.read_text());first=output.read_bytes()
 assert len(data['predictions'])==26 and data['uses_target_llm_timings'] is False
 assert data['kernel_surface_used'] is False and data['target_sm_clock_mhz']==2400
 assert data['protocol_ref']==analytical.ref(protocol)
 with pytest.raises(ValueError,match='immutable'):analytical.main()
 assert output.read_bytes()==first


def test_duplicate_shape_even_with_unique_ids_rejected(tmp_path,monkeypatch):
 protocol=json.loads((P/'operator_probe/r3/protocol.json').read_text(encoding='utf-8-sig'))
 protocol['configs'][-1]={**protocol['configs'][0],'id':'unique-but-duplicate-shape'}
 path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
 monkeypatch.setattr(sys,'argv',['analytical_predictions.py','--protocol',str(path),'--output',str(tmp_path/'never.json')])
 with pytest.raises(ValueError,match='Joint-shape'):analytical.main()
 assert not (tmp_path/'never.json').exists()
