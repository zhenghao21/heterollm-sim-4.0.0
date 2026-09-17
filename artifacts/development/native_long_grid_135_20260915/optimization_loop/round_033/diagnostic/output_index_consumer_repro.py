"""Small structural reproduction on unchanged frozen source; no model or timing reads."""
import sys,json,traceback,hashlib,datetime
from pathlib import Path
from dataclasses import replace
HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
ROOT=ROUND.parents[4]
arm=sys.argv[1]
assert arm in ('off','on')
frozen=ROUND/arm/'source'
sys.path[:0]=[str(frozen/'src'),str(ROOT)]
from heterollm_sim import planner
from tests.test_llama_final_norm_placement import scenario
from tests.test_final_layer_output_selection_planner import _cohort
from tests.model_helpers import model_from_layer_specs,execution_layers
case=scenario(ngl=0,offload=True);old=case.model
model=model_from_layer_specs(old.name,execution_layers(old),vocabulary_size=old.vocabulary_size,max_sequence_length=old.max_sequence_length,embedding_weight_bytes=old.embedding_weight_bytes,metadata=old.metadata,architecture='qwen3_5_hybrid_transformer')
case=replace(case,model=model)
original=planner._bind_output_index_upload
observations=[]
def observe(builder,first,ids):
 if ids:
  wanted=[t.metadata.get('final_layer_output_selection',{}).get('destination_component') for t in builder.tasks if t.task_id==ids[0]]
  gathers=[{'target':t.metadata.get('target_component'),'task_id':t.task_id} for t in builder.tasks[first:] if t.metadata.get('event_kind')=='output_row_selection']
  observations.append({'index_upload_target':wanted,'actual_gathers':gathers})
 return original(builder,first,ids)
planner._bind_output_index_upload=observe
rows=[]
try:
 for phase,tokens,context in [('prefill',64,0),('decode',1,64)]:
  observations.clear()
  row={'phase':phase,'token_rows':tokens,'context':context}
  try:
   result=planner.compile_serving_cohort_schedule(case,_cohort(tokens,1,context=context,phase=phase))
   row.update(status='compiled',task_count=len(result.tasks))
  except Exception as error:
   row.update(status='failed',exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
  row['observations']=list(observations);rows.append(row)
finally:planner._bind_output_index_upload=original
ref=lambda p:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
record={'schema':'r33-output-index-consumer-small-repro/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arm':arm,'source_ref':ref(Path(planner.__file__)),'script_ref':ref(Path(__file__)),'fixture_refs':[ref(ROOT/p) for p in ['tests/test_llama_final_norm_placement.py','tests/test_final_layer_output_selection_planner.py','tests/model_helpers.py']],'scope':'two-layer synthetic graph, same source architecture and F32 norm binding; excludes actual GGUF execution; formal worker stack unavailable','native_run':False,'target_timings_used':False,'frozen_files_modified':False,'results':rows}
with (HERE/('output_index_consumer_repro.'+arm+'.json')).open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'arm':arm,'results':[{'phase':x['phase'],'status':x['status'],'exception':x.get('exception'),'observations':x['observations']} for x in rows]},ensure_ascii=False))
