from pathlib import Path
import json,subprocess,sys
P=Path(__file__).resolve().parent
for variant in ('current','sampling','pure'):
 out=P/variant;freeze=json.loads((out/'freeze.json').read_text(encoding='utf-8'))
 ids=[r['cell_id'] for r in freeze['cells'] if r['model_key']=='qwen25' or (r['model_key'] in {'qwen35','smollm2','tinyllama'} and '_p512_o32_c1' in r['cell_id'])]
 assert len(ids)==20
 argv=[sys.executable,str(out/'source/tools/predict_stable_native_dataset.py'),'--output',str(out),'--resume','--workers','4']
 for ident in ids:argv+=['--cell-id',ident]
 print('Running static-only '+variant+' 20 anchors; no native remeasurement',flush=True)
 code=subprocess.call(argv)
 if code:raise SystemExit(code)
 argv=[sys.executable,str(out/'source/tools/predict_stable_native_dataset.py'),'--output',str(out),'--score']
 code=subprocess.call(argv)
 if code:raise SystemExit(code)
print('All three independent frozen20-anchor variants completed. Unrun131 denominator retained.',flush=True)
