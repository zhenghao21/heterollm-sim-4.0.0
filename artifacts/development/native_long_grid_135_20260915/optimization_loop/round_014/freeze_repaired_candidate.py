from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from heterollm_sim.llama_gpu_invocations import derive_llama_gpu_invocation_contract
from tools.predict_stable_native_dataset import freeze_selection
R=Path(__file__).resolve().parent
f=json.loads((R.parent/'round_006/physical_mapping_mmq_r2/freeze.json').read_text(encoding='utf-8'))
binding=json.loads(Path(f['host_offload_source']['contract_ref']['path']).read_text(encoding='utf-8'))['build_binding']
old=json.loads((R.parent/'round_006/repaired_source_contract.json').read_text(encoding='utf-8'))
contract=derive_llama_gpu_invocation_contract(binding,captured_kernel_environment={'GGML_CUDA_DISABLE_FUSION':None},cuda_compute_capability=1200,mmq_device_evidence={k:old['mmq_device_evidence'][k] for k in ('source_ref','selected_hardware_ref')})
p=R/'tail_source_contract.json'
assert json.loads(p.read_text(encoding='utf-8'))==contract, 'source contract changed'
freeze_selection(Path(f['selection_ref']['path']),R/'tail_candidate_r2',data_root=Path(f['data_root']),model_snapshot_map={k:v['path'] for k,v in f['model_snapshot_map'].items()},runtime_build_audit_path=Path(f['runtime_build_audit']['audit_ref']['path']),recurrent_batching_contract_path=Path(f['recurrent_batching']['contract_ref']['path']),slot_order_contract_path=Path(f['slot_order']['contract_ref']['path']),host_offload_source_contract_path=Path(f['host_offload_source']['contract_ref']['path']),tensor_storage_contract_path=Path(f['tensor_storage']['contract_ref']['path']),tensor_storage_f32_hidden=True,gpu_invocation_contract_path=p,gpu_mmq_source_costs=True)
print('R14 final-tail-read candidate r2 frozen; prior freezes retained')
