"""R21 source-only three-way comparison. Native actual remains fixed and read at scoring only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from tools.predict_stable_native_dataset import freeze_selection, verify_freeze_references
from tools.verify_fixed_native import verify_lock
P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'round_018'
STATE=P.parent/'state.json'
VARIANTS=('pure','current','physical')
SELECTION_SHA='cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):
 p=Path(p).resolve(strict=True)
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def write(p,data):
 with Path(p).open('x',encoding='utf-8') as out:json.dump(data,out,ensure_ascii=False,indent=2,allow_nan=False)
def check_controls():
 for x in read(P/'evaluation_controls.json')['files']:
  if ref(x['path'])!=x:raise ValueError('evaluation control changed: '+x['path'])

def run(argv):
 code=subprocess.call([str(x) for x in argv])
 if code:raise RuntimeError('command failed: '+str(code))

def freeze():
 lock=verify_lock(STATE)
 if lock['selection_sha256']!=SELECTION_SHA:raise ValueError('native selection changed')
 if (P/'evaluation_controls.json').exists():raise FileExistsError('controls already frozen')
 f=read(PREVIOUS/'sampling/freeze.json')
 ids=sorted(x['cell_id'] for x in f['cells'] if x['model_key']=='qwen25' or (x['model_key'] in {'qwen35','smollm2','tinyllama'} and '_p512_o32_c1' in x['cell_id']))
 if len(ids)!=20 or len(f['cells'])!=131:raise ValueError('scope mismatch')
 # This manifest is static-only; it contains no native latency or fitted rate.
 protocol={'schema':'nonflash-physical-kv-ablation/v1','created_utc':datetime.now(timezone.utc).isoformat(),
  'native_lock':lock,'variants':list(VARIANTS),'anchor_ids':ids,'full_denominator':131,
  'threshold_pct_strict':10,'metrics':['engine_ttft_ms','engine_tpot_ms','engine_e2e_ms'],
  'pure':'Existing analytical cost baseline, no MMQ source cost, no sampling contract, no physical KV rule',
  'current':'R18 sampling semantics and MMQ source costs, physical KV rule off',
  'physical':'current plus source-bound non-Flash KV lower-bound width; retained pool state unknown',
  'is_blind':False,'formal_acceptance':False,'native_remeasurement':False,'calibration_added':False,
  'workflow':'Save and validate all three anchor prediction sets before scoring any; preserve failures and 131 denominator. Continue physical to all131 after anchored review without modifying its freeze.',
  'continuation_criterion':'Mechanism/identity regressions pass and source correction retained. Mixed accuracy outcomes are reported, never censored. Full131 outcome needed for gate A.'}
 write(P/'evaluation_protocol.json',protocol)
 for variant in VARIANTS:
  freeze_selection(Path(f['selection_ref']['path']),P/variant,data_root=Path(f['data_root']),
   model_snapshot_map={k:v['path'] for k,v in f['model_snapshot_map'].items()},
   runtime_build_audit_path=Path(f['runtime_build_audit']['audit_ref']['path']),
   recurrent_batching_contract_path=Path(f['recurrent_batching']['contract_ref']['path']),
   slot_order_contract_path=Path(f['slot_order']['contract_ref']['path']),
   host_offload_source_contract_path=Path(f['host_offload_source']['contract_ref']['path']),
   tensor_storage_contract_path=Path(f['tensor_storage']['contract_ref']['path']),tensor_storage_f32_hidden=True,
   gpu_invocation_contract_path=PREVIOUS/'conversion_source_contract.json',
   gpu_mmq_source_costs=variant!='pure',gpu_conversion_cta_costs=False,
   sampling_contract_path=PREVIOUS/'sampling_parity/static_sampling_contract.json' if variant!='pure' else None,
   nonflash_kv_view_source_contract_path=P/'nonflash_kv_view_source_contract.json' if variant=='physical' else None)
  print('Frozen '+variant,flush=True)
 print(json.dumps(verify_lock(STATE)),flush=True)

def lock_controls():
 if (P/'evaluation_controls.json').exists():raise FileExistsError('controls already frozen')
 identity=[]
 for variant in VARIANTS:
  f=read(P/variant/'freeze.json');verify_freeze_references(f)
  identity.append({str(Path(x['path']).relative_to(Path(f['source']['root']))):x['sha256'] for x in f['source']['files']})
 if not identity or not all(x==identity[0] for x in identity):raise ValueError('variants differ in source closure')
 controls=[Path(__file__),P/'evaluation_protocol.json',P/'summarize_ablation.py',P/'nonflash_kv_view_source_contract.json',*(P/v/'freeze.json' for v in VARIANTS)]
 write(P/'evaluation_controls.json',{'schema':'ablation-controls/v1','created_utc':datetime.now(timezone.utc).isoformat(),'files':[ref(x) for x in controls],
  'source_file_count':len(identity[0]),'same_source_closure':True})
 print(json.dumps(verify_lock(STATE)),flush=True)

def execute(phase):
 check_controls();verify_lock(STATE)
 protocol=read(P/'evaluation_protocol.json')
 variants=VARIANTS if phase=='anchors' else ('physical',)
 for variant in variants:
  output=P/variant;verify_freeze_references(read(output/'freeze.json'))
  argv=[sys.executable,output/'source/tools/predict_stable_native_dataset.py','--output',output,'--resume','--workers','4']
  if phase=='anchors':
   for ident in protocol['anchor_ids']:argv.extend(['--cell-id',ident])
  print('Predict '+variant+' '+phase+'; no native execution',flush=True);run(argv)
 # Score is deliberately separated until every planned variant has saved predictions.
 for variant in variants:
  output=P/variant
  run([sys.executable,output/'source/tools/predict_stable_native_dataset.py','--output',output,'--score'])
 check_controls();print(json.dumps(verify_lock(STATE)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['freeze','lock','anchors','remaining']);a=p.parse_args()
 if a.phase=='freeze':freeze()
 elif a.phase=='lock':lock_controls()
 else:execute(a.phase)
