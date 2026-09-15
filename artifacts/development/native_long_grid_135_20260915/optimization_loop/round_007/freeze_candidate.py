from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
from tools.predict_stable_native_dataset import freeze_selection
R=Path(__file__).resolve().parent
f=json.loads((R.parent/"round_006/physical_mapping_mmq_r2/freeze.json").read_text(encoding="utf-8"))
freeze_selection(Path(f["selection_ref"]["path"]),R/"read_identity_r1",data_root=Path(f["data_root"]),model_snapshot_map={k:v["path"] for k,v in f["model_snapshot_map"].items()},runtime_build_audit_path=Path(f["runtime_build_audit"]["audit_ref"]["path"]),recurrent_batching_contract_path=Path(f["recurrent_batching"]["contract_ref"]["path"]),slot_order_contract_path=Path(f["slot_order"]["contract_ref"]["path"]),host_offload_source_contract_path=Path(f["host_offload_source"]["contract_ref"]["path"]),tensor_storage_contract_path=Path(f["tensor_storage"]["contract_ref"]["path"]),tensor_storage_f32_hidden=True,gpu_invocation_contract_path=R.parent/"round_006/repaired_source_contract.json",gpu_mmq_source_costs=True)
print("R7 read-identity candidate frozen")
