from pathlib import Path
import json,sys,argparse
ROOT=Path(__file__).resolve().parents[5];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from heterollm_sim.llama_gpu_invocations import derive_llama_gpu_invocation_contract
from tools.predict_stable_native_dataset import freeze_selection
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['current','conversion_cta','pure']);a=ap.parse_args()
f=json.loads((P.parent/'round_014/tail_candidate_r2/freeze.json').read_text(encoding='utf-8'))
binding=json.loads(Path(f['host_offload_source']['contract_ref']['path']).read_text(encoding='utf-8'))['build_binding']
old=json.loads((P.parent/'round_006/repaired_source_contract.json').read_text(encoding='utf-8'))
contract=derive_llama_gpu_invocation_contract(binding,captured_kernel_environment={'GGML_CUDA_DISABLE_FUSION':None},cuda_compute_capability=1200,
 mmq_device_evidence={k:old['mmq_device_evidence'][k] for k in ('source_ref','selected_hardware_ref')})
path=P/'conversion_source_contract.json'
if not path.exists():
 with path.open('x',encoding='utf-8') as out:json.dump(contract,out,indent=2)
else:assert json.loads(path.read_text(encoding='utf-8'))==contract
result=freeze_selection(Path(f['selection_ref']['path']),P/a.variant,data_root=Path(f['data_root']),
 model_snapshot_map={k:v['path'] for k,v in f['model_snapshot_map'].items()},runtime_build_audit_path=Path(f['runtime_build_audit']['audit_ref']['path']),
 recurrent_batching_contract_path=Path(f['recurrent_batching']['contract_ref']['path']),slot_order_contract_path=Path(f['slot_order']['contract_ref']['path']),
 host_offload_source_contract_path=Path(f['host_offload_source']['contract_ref']['path']),tensor_storage_contract_path=Path(f['tensor_storage']['contract_ref']['path']),
 tensor_storage_f32_hidden=True,gpu_invocation_contract_path=path,gpu_mmq_source_costs=a.variant!='pure',gpu_conversion_cta_costs=a.variant=='conversion_cta')
print('Frozen '+a.variant+' with fixed native and current source closure; no prediction or native run.')
