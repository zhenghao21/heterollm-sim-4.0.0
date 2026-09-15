from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent
old=r.parent/'round_010/stream_event_probe_dual_r2/runs/20260915T231129794Z';new=r/'long_batch/runs/20260915T232119418Z'
rows=[]
for p in sorted(new.glob('*.assessment.json')):
 a=json.loads((old/p.name).read_text());b=json.loads(p.read_text());s=a['statistics']['event'];t=b['statistics']['event']
 rows.append({'config':b['config'],'old_pass':a['accepted'],'new_pass':b['accepted'],'old_batch_calls':64,'new_batch_calls':1024,'old_event_p90_p10':s['event_p90_p10'],'new_event_p90_p10':t['event_p90_p10'],'old_event_max_deviation_pct':100*s['event_max_deviation_from_median_fraction'],'new_event_max_deviation_pct':100*t['event_max_deviation_from_median_fraction'],'old_per_graph_ns':s['event_per_graph_median_ns'],'new_per_graph_ns':t['event_per_graph_median_ns'],'new_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
summary=json.loads((new/'matrix_summary.json').read_text())
out=r/'window_comparison.json'
if out.exists():raise SystemExit('no overwrite')
out.write_text(json.dumps({'matrix_evidence_complete':summary['matrix_evidence_complete'],'accepted':summary['accepted_configs'],'total':12,'comparison_is_development':True,'llm_remeasured':False,'cost_model_updated':False,'all_event_max_deviation_below5':all(x['new_event_max_deviation_pct']<5 for x in rows),'rows':rows},indent=2))
print(summary['accepted_configs'],summary['matrix_evidence_complete'])
