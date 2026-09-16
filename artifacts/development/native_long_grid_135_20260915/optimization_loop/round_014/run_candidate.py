from pathlib import Path
import json,subprocess,sys
r=Path(__file__).resolve().parent;out=r/'tail_candidate';f=json.loads((out/'freeze.json').read_text(encoding='utf-8'))
ids=[c['cell_id'] for c in f['cells'] if c['model_key']=='qwen25' or (c['model_key'] in {'qwen35','smollm2','tinyllama'} and '_p512_o32_c1' in c['cell_id'])]
assert len(ids)==20 and len(set(ids))==20
args=[sys.executable,str(out/'source/tools/predict_stable_native_dataset.py'),'--output',str(out),'--resume','--workers','4']
for cell in ids:args+=['--cell-id',cell]
print('17 Qwen2.5 cells plus 3 aligned controls; native unchanged',flush=True)
raise SystemExit(subprocess.call(args))
