"""Complete request/resolved-settings projection with source-bound null-bias normalization."""
from pathlib import Path
import copy,hashlib,json
from export_static_sampling_contract import read_verified,ref,nonnull,write_new,digest,settings_projection,typed_candidate
P=Path(__file__).resolve().parent

def normalize_full_settings(settings):
 out=copy.deepcopy(settings);normalizations=[]
 for i,bias in enumerate(out.get('logit_bias',[])):
  if bias.get('bias','missing') is None:
   if out.get('ignore_eos') is not True:raise ValueError('unexplained null bias')
   bias['bias']={'value_kind':'negative_infinity','wire_representation':'json_null'}
   normalizations.append({'json_pointer':'/logit_bias/'+str(i)+'/bias','token':bias['token'],'semantic_value':'negative_infinity','source_basis':'common.cpp EOG -INFINITY construction; server-schema.cpp ignore_eos append; server-task.cpp format_logit_bias'})
 nonnull(out)
 return out,normalizations

def extract_from_verified_raw(raw_ref):
 """Hash-check the container and visit only payload and response generation_settings."""
 raw=read_verified(raw_ref)
 payload=copy.deepcopy(raw['payload']);nonnull(payload)
 variants={};counts={'warmup':0,'runs':0};first=''
 for phase in ('warmup','runs'):
  for bi,batch in enumerate(raw[phase]):
   for ri,request in enumerate(batch['requests']):
    # No generated content, tokens, timings, engine timestamp or latency keys read.
    settings=request['response']['generation_settings']
    core=settings_projection(settings,payload)
    full,normalizations=normalize_full_settings(settings)
    key=digest(full);variants.setdefault(key,(full,normalizations,core))
    counts[phase]+=1
    if not first:first=f'/{phase}/{bi}/requests/{ri}/response/generation_settings'
 if len(variants)!=1:raise ValueError('complete effective generation settings differ within raw block')
 full,normalizations,core=next(iter(variants.values()))
 return {'requested_payload':payload,'resolved_generation_settings':full,'sampling_policy':typed_candidate(core),'bias_normalizations':normalizations,'request_counts':counts,'evidence':{'raw_ref':raw_ref,'all_resolved_settings_checked':sum(counts.values()),'generation_settings_json_pointer_example':first,'payload_json_pointer':'/payload','requested_prompt_canonical_sha256':digest(payload['prompt']),'prompt_representation':'token_id_array' if isinstance(payload['prompt'],list) else 'text_string'}}

def main():
 original_path=P/'static_sampling_contract.json';original=json.loads(original_path.read_text(encoding='utf-8'))
 cells={}
 for cell in original['cells']:
  blocks=[extract_from_verified_raw(item['raw_ref']) for item in cell['raw_configuration_evidence']]
  signatures={digest({'requested_payload':b['requested_payload'],'resolved_generation_settings':b['resolved_generation_settings']}) for b in blocks}
  if len(signatures)!=1:raise ValueError('complete effective config differs within selected cell')
  b=blocks[0]
  cells[cell['cell_id']]={'cell_id':cell['cell_id'],'source_id':cell['source_id'],'requested_payload':b['requested_payload'],'resolved_generation_settings':b['resolved_generation_settings'],'sampling_policy':b['sampling_policy'],'backend_sampling':False,'bias_normalizations':b['bias_normalizations'],'raw_evidence':[v['evidence'] for v in blocks],'freeze_ref':cell['freeze_ref'],'collector_source_ref':cell['collector_source_ref'],'qualifications':{'native_backend':'host CPU sampling chain','candidate_materialization':'full vocabulary, F32 logits into 12-byte llama_token_data records','ordered_active_filters':['logit_bias_if_nonempty','top_k','top_p','min_p','temperature','dist'],'disabled_constructor_nodes':['penalties','dry','top_n_sigma','typ_p','xtc'],'temperature_zero_is_not_a_greedy_sampler_shortcut':True,'top_k_one_preserves_one_candidate':True,'native_min_keep_zero_legal':True,'dist_single_candidate_draws_uniform_rng':True,'grammar_empty':True,'speculative_decoding_disabled':True,'backend_sampling_disabled':True,'nonactive_parameters_preserved':True},'provenance_status':cell['provenance_status']}
 result={'schema':'native-sampling-static-contract/v1','status':'complete_source_request_and_resolved_config_projection','selection_ref':original['selection_ref'],'audit_source_contract_ref':ref(original_path),'exporter_ref':ref(Path(__file__)),'summary':original['summary'],'source_snapshots':original['source_snapshots'],'runtime_refs':original['runtime_refs'],'build_and_header_lineage':original['build_and_header_lineage'],'cells':cells,'preservation':{'complete_native_request_payload_preserved':True,'complete_generation_settings_preserved':True,'only_wire_null_bias_normalized_to_explicit_nonnull_negative_infinity':True,'nonactive_sampler_settings_preserved':True,'all_contract_keys_nonnull':True},'scope':{'GPU_executed':False,'simulation_or_native_inference_executed':False,'latency_fields_projected_or_used':False,'delay_or_cost_values_exported':False,'generated_response_text_or_tokens_used':False},'remaining_nonpolicy_scope':['Additional vocabulary model-suppress tokens are appended by common_sampler_init; generation_settings serializes request EOG biases but not that additional vocabulary list. Bind that static metadata separately if its work is modeled.','Empty sampler constructors do not execute penalty/DRY algorithm bodies; chain traversal and common previous-token ring commit remain source operations with no delay assigned here.']}
 write_new(P/'static_contract.json',result)
 handoff={'schema':'sampling-integration-handoff/v1','ready':True,'contract':ref(P/'static_contract.json'),'contract_schema':result['schema'],'cell_container':'cells mapping keyed by selected cell_id','policy_pointer':'/cells/{cell_id}/sampling_policy','requested_payload_pointer':'/cells/{cell_id}/requested_payload','resolved_settings_pointer':'/cells/{cell_id}/resolved_generation_settings','source_contract_binding':'selection_ref + per-cell raw_evidence/freeze_ref/collector_source_ref + runtime_refs + historical header/build lineage','extractor_module':ref(Path(__file__)),'extractor_function':'extract_from_verified_raw(raw_ref: dict) -> dict; verifies bytes+SHA256 and projects complete payload/resolved settings only','dependency_module':ref(P/'export_static_sampling_contract.py'),'policy_domain':'top_k positive integer; min_keep integer >=0 (not bool); preserve min_keep<=top_k for bounded implementation','all_131_complete':True,'no_timing_evidence_needed_for_semantic_integration':True}
 write_new(P/'INTEGRATION_HANDOFF.json',handoff)
 print(json.dumps(handoff,ensure_ascii=False))
if __name__=='__main__':main()
