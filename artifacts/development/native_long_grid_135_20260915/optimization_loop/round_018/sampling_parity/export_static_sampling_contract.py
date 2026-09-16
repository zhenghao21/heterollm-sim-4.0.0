"""Export source/config-only sampling parity. No GPU, simulation, or latency field projection."""
from pathlib import Path
from collections import Counter
import ast,copy,dataclasses,hashlib,json,math,struct,subprocess
from typing import Optional
P=Path(__file__).resolve().parent
ROOT=P.parents[5]
DATA=P.parents[2]
REQUIRED=('seed','temperature','dynatemp_range','dynatemp_exponent','top_k','top_p','min_p','top_n_sigma','xtc_probability','xtc_threshold','typical_p','repeat_last_n','repeat_penalty','presence_penalty','frequency_penalty','dry_multiplier','dry_base','dry_allowed_length','dry_penalty_last_n','dry_sequence_breakers','mirostat','mirostat_tau','mirostat_eta','adaptive_target','adaptive_decay','ignore_eos','stream','n_probs','min_keep','grammar','grammar_lazy','grammar_triggers','preserved_tokens','samplers','speculative.types','post_sampling_probs','backend_sampling','lora')
PAYLOAD_KEYS=('n_predict','ignore_eos','cache_prompt','temperature','top_k','seed','stream')
SAMPLER_ORDER=['penalties','dry','top_n_sigma','top_k','typ_p','top_p','min_p','xtc','temperature']
CLI_PREFIXES=('--temp','--temperature','--top-k','--top-p','--min-p','--min-keep','--samplers','--sampling-seq','--seed','--mirostat','--logit-bias','--ignore-eos','--backend-sampling','--no-backend-sampling','--grammar','--repeat-penalty','--presence-penalty','--frequency-penalty','--dry-')
CORE_NAMES=('llama-server.exe','llama-server-impl.dll','llama-common.dll','llama.dll')
SOURCES={
 'sampling_policy':'src/heterollm_sim/config.py','planner':'src/heterollm_sim/planner.py',
 'matching_scenario':'tools/native_llama_compare.py','static_predictor':'tools/predict_stable_native_dataset.py',
 'common_defaults':'source/llama.cpp-semantic/common/common.h','common_sampling':'source/llama.cpp-semantic/common/sampling.cpp',
 'common_initialization':'source/llama.cpp-semantic/common/common.cpp','native_samplers':'source/llama.cpp-semantic/src/llama-sampler.cpp',
 'native_sampler_header':'source/llama.cpp-semantic/include/llama.h',
 'server_schema':'source/llama.cpp-semantic/tools/server/server-schema.cpp','server_task':'source/llama.cpp-semantic/tools/server/server-task.cpp',
 'server_context':'source/llama.cpp-annotation-control/tools/server/server-context.cpp'}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()
def ref(path):
 path=Path(path).resolve(strict=True)
 with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h}

def verify_ref(expected):
 actual=ref(expected['path'])
 if actual['bytes']!=expected['bytes'] or actual['sha256']!=expected['sha256']:raise ValueError('frozen reference mismatch: '+expected['path'])
 return actual

def read_verified(expected):
 path=Path(expected['path']);raw=path.read_bytes()
 if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError('raw/config evidence changed: '+str(path))
 return json.loads(raw.decode('utf-8'))

def nonnull(v,where='root'):
 if v is None:raise ValueError('unresolved null in contract: '+where)
 if isinstance(v,dict):
  for k,x in v.items():nonnull(x,where+'.'+str(k))
 elif isinstance(v,(list,tuple)):
  for i,x in enumerate(v):nonnull(x,where+'['+str(i)+']')
 elif isinstance(v,float) and not math.isfinite(v):raise ValueError('non-finite numeric literal in contract: '+where)

def write_new(path,value):
 nonnull(value)
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

def settings_projection(settings,payload):
 if not isinstance(settings,dict):raise ValueError('resolved generation settings required')
 for k in REQUIRED:
  if k not in settings or settings[k] is None:raise ValueError('resolved sampling key missing/null: '+k)
 out={k:copy.deepcopy(settings[k]) for k in REQUIRED}
 if settings.get('generation_prompt','')!='':raise ValueError('generation prompt may enable extra reasoning behavior')
 if out['samplers']!=SAMPLER_ORDER:raise ValueError('unreviewed sampler order')
 if out['temperature']!=0 or out['top_k']!=1 or out['seed']!=42 or out['min_keep']!=0 or out['backend_sampling'] is not False:raise ValueError('unreviewed effective sampling policy')
 if out['top_p']!=f32(.95) or out['min_p']!=f32(.05):raise ValueError('unreviewed top-p/min-p defaults')
 if out['ignore_eos'] is not True or out['n_probs']!=0 or out['mirostat']!=0 or out['grammar']!='' or out['grammar_lazy'] is not False or out['speculative.types']!='none,none' or out['lora']!=[]:raise ValueError('unreviewed sampling modifiers')
 for k in ('temperature','top_k','seed','ignore_eos','stream'):
  if out[k]!=payload[k]:raise ValueError('wire request and resolved settings disagree: '+k)
 if 'logit_bias' not in settings or not isinstance(settings['logit_bias'],list):raise ValueError('resolved logit_bias required')
 biases=[]
 for entry in settings['logit_bias']:
  if not isinstance(entry,dict) or type(entry.get('token')) is not int:raise ValueError('unsupported serialized logit bias')
  bias=entry.get('bias','missing')
  if bias is None:
   # Collector sends no user logit bias. The locked server handler appends its
   # precomputed EOG -INFINITY list for ignore_eos=true, serialized as JSON null.
   if payload['ignore_eos'] is not True:raise ValueError('unexplained null bias')
   biases.append({'token':entry['token'],'bias_kind':'negative_infinity','wire_encoding':'json_null','source_basis':'ignore_eos EOG list; common.cpp populates -INFINITY; server-schema.cpp appends it'})
  elif isinstance(bias,(int,float)) and not isinstance(bias,bool) and math.isfinite(bias):
   biases.append({'token':entry['token'],'bias_kind':'finite','value':bias,'wire_encoding':'json_number'})
  else:raise ValueError('unsupported serialized bias value')
 out['logit_bias']=biases
 out['generation_prompt_empty']=True
 nonnull(out)
 return out

def typed_candidate(settings):
 return {'mode':'greedy','temperature':settings['temperature'],'implementation':'llama_cpp_cpu_chain','top_k':settings['top_k'],'top_p':settings['top_p'],'min_p':settings['min_p'],'min_keep':settings['min_keep']}

def source_snapshot():
 dest=P/'source_snapshots';dest.mkdir(exist_ok=False)
 result={}
 for key,relative in SOURCES.items():
  original=ROOT/relative;data=original.read_bytes();target=dest/(key+original.suffix)
  with target.open('xb') as f:f.write(data)
  result[key]={'original':ref(original),'snapshot':ref(target)}
 return result

def lineage_proof(runtime_refs):
 a=ROOT/'source/llama.cpp-annotation-control/evidence';n=ROOT/'source/llama.cpp-native-thread-control/evidence'
 headers=json.loads((a/'header_snapshot.json').read_text(encoding='utf-8'));ab=json.loads((a/'build_receipt.json').read_text(encoding='utf-8'));nb=json.loads((n/'build_receipt.json').read_text(encoding='utf-8'))
 if ref(a/'header_snapshot.json')['sha256']!=ab['header_snapshot_sha256']:raise ValueError('historical header snapshot not bound to annotation build')
 historical=[]
 for suffix in ('common/common.h','common/sampling.h','include/llama.h'):
  matches=[(path,h) for path,h in headers['files'].items() if path.replace('\\','/').endswith(suffix)]
  if len(matches)!=1:raise ValueError('historical header missing: '+suffix)
  path,h=matches[0];current=ref(path)
  if current['sha256']!=h:raise ValueError('sampling header drift from historical build')
  historical.append({'source':current,'historical_sha256':h,'snapshot_ref':ref(a/'header_snapshot.json')})
 binary_chain=[]
 for name in CORE_NAMES:
  selected=[v for v in runtime_refs.values() if Path(v['path']).name==name]
  if len(selected)!=1:raise ValueError('multiple selected runtime identities for '+name)
  item=selected[0];expected=nb['unchanged_runtime_sha256'].get(item['path'])
  if expected!=item['sha256']:raise ValueError('selected runtime not bound to native unchanged-runtime receipt: '+name)
  ancestor=Path(nb['runtime_base'])/name
  if ref(ancestor)['sha256']!=item['sha256']:raise ValueError('annotation-to-native unchanged module chain mismatch')
  binary_chain.append({'name':name,'selected':item,'annotation_ancestor':ref(ancestor),'native_receipt_ref':ref(n/'build_receipt.json')})
 gitroot=ROOT/'source/llama.cpp-semantic'
 commit=subprocess.run(['git','-C',str(gitroot),'rev-parse','HEAD'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,text=True).stdout.strip()
 tracked=['common/common.h','common/sampling.cpp','common/common.cpp','src/llama-sampler.cpp','tools/server/server-schema.cpp','tools/server/server-task.cpp']
 changes=subprocess.run(['git','-C',str(gitroot),'status','--porcelain','--',*tracked],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,text=True).stdout.strip()
 if changes:raise ValueError('uncommitted source drift in sampling implementation')
 smoke=(n/'version_smoke.log').read_text(encoding='utf-8')
 if commit[:7] not in smoke:raise ValueError('recorded runtime commit does not match clean source')
 return {'historical_header_bindings':historical,'unchanged_binary_chain':binary_chain,'clean_sampling_source_commit':commit,'recorded_runtime_version_ref':ref(n/'version_smoke.log'),'receipts':[ref(a/'source_manifest.json'),ref(a/'build_receipt.json'),ref(n/'source_manifest.json'),ref(n/'build_receipt.json')],'scope':'Effective values also checked directly in frozen generation_settings for every audited request. Source cleanliness and version marker are supporting lineage; no runtime instruction trace is claimed.'}

def ast_parity(snapshot_refs):
 config_path=Path(snapshot_refs['sampling_policy']['snapshot']['path']);config_text=config_path.read_text(encoding='utf-8')
 tree=ast.parse(config_text);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SamplingPolicy')
 module=ast.Module(body=[cls],type_ignores=[]);ast.fix_missing_locations(module)
 ns={'dataclass':dataclasses.dataclass,'Optional':Optional,'math':math};exec(compile(module,str(config_path),'exec'),ns)
 candidate={'mode':'greedy','temperature':0.0,'implementation':'llama_cpp_cpu_chain','top_k':1,'top_p':f32(.95),'min_p':f32(.05),'min_keep':0}
 try:ns['SamplingPolicy'](**candidate);admissible=True;error='none'
 except ValueError as e:admissible=False;error=str(e)
 cmp=ast.parse(Path(snapshot_refs['matching_scenario']['snapshot']['path']).read_text(encoding='utf-8'))
 fn=next(x for x in cmp.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='build_matching_scenario')
 scenario_sampling_written=any(isinstance(x,ast.keyword) and x.arg=='sampling_policy' for x in ast.walk(fn))
 pt=ast.parse(Path(snapshot_refs['static_predictor']['snapshot']['path']).read_text(encoding='utf-8'))
 keys=next(ast.literal_eval(x.value) for x in pt.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STATIC_KEYS' for t in x.targets))
 return {'source_snapshot_time':'at exporter start','current_snapshot_accepts_native_min_keep_zero':admissible,'constructor_result':error,'matching_scenario_writes_sampling_policy':scenario_sampling_written,'static_allowlist_sampling_keys':{k:k in keys for k in ('temperature','top_k','top_p','min_p','min_keep','sampling_policy','backend_sampling')},'execution_scope':'isolated dataclass validation and AST only; no scenario or predictor execution'}

def main():
 source_refs=source_snapshot()
 selection_path=DATA/'stable_native_dataset.json';selection=json.loads(selection_path.read_text(encoding='utf-8'))
 rows=selection['selected_cells'];assert len(rows)==131 and len(set(r['cell_id'] for r in rows))==131
 profiles={};cells=[];runtime_refs={};freeze_refs={};collector_refs={};request_count=Counter();raw_count=0;raw_bytes=0;source_count=Counter();bias_counts=Counter()
 for i,row in enumerate(rows):
  source_id=row['source_per_cell']['source_id'];source_count[source_id]+=1
  freeze_ref=row['source_per_cell']['freeze_ref'];freeze_key=freeze_ref['sha256']
  if freeze_key not in freeze_refs:
   freeze=read_verified(freeze_ref);freeze_refs[freeze_key]=freeze_ref
   candidates=[f for f in freeze['source_refs'] if Path(f['path']).name=='native_repeatability_experiment.py']
   if len(candidates)!=1:raise ValueError('one frozen collector source expected')
   collector_ref=verify_ref(candidates[0]);collector_text=Path(collector_ref['path']).read_text(encoding='utf-8')
   for literal in ("'temperature':0","'top_k':1","'ignore_eos':True","'cache_prompt':False","record['payload']=payload"):
    if literal not in collector_text:raise ValueError('collector frozen payload construction changed')
   collector_refs[freeze_key]=collector_ref
  refs=[];cell_profile_ids=set();cell_requests=Counter();payload_fingerprints=set();cell_payloads=[]
  for native_ref in row['native_runtime_refs']:
   if Path(native_ref['path']).name in CORE_NAMES:
    k=native_ref['path']
    if k not in runtime_refs:runtime_refs[k]=verify_ref(native_ref)
    elif runtime_refs[k]['sha256']!=native_ref['sha256']:raise ValueError('mixed selected runtime hashes')
  for evidence in row['evidence_index']:
   expected=evidence['raw_ref'];raw=read_verified(expected);raw_count+=1;raw_bytes+=expected['bytes']
   payload=raw['payload']
   if any(k not in payload or payload[k] is None for k in PAYLOAD_KEYS):raise ValueError('wire payload incomplete')
   extra=set(payload)-set(PAYLOAD_KEYS)-{'prompt'}
   if extra:raise ValueError('unreviewed native request fields: '+','.join(sorted(extra)))
   projection={k:payload[k] for k in PAYLOAD_KEYS}
   if projection['temperature']!=0 or projection['top_k']!=1 or projection['ignore_eos'] is not True or projection['cache_prompt'] is not False or projection['seed']!=42 or projection['stream'] is not True:raise ValueError('unreviewed native request policy')
   if raw['config']['seed']!=payload['seed'] or raw['config']['output']!=payload['n_predict']:raise ValueError('raw config/request mismatch')
   cli=[x for x in raw['actual_argv'][1:] if isinstance(x,str) and any(x==flag or (flag.endswith('-') and x.startswith(flag)) or x.startswith(flag+'=') for flag in CLI_PREFIXES)]
   if cli:raise ValueError('unreviewed sampling CLI overrides: '+repr(cli))
   pd=digest(projection)
   if pd not in payload_fingerprints:cell_payloads.append(projection);payload_fingerprints.add(pd)
   raw_profile_ids=set();first_location='';seen=0
   for phase in ('warmup','runs'):
    for bi,batch in enumerate(raw[phase]):
     for ri,request in enumerate(batch['requests']):
      # Deliberately never read response content/tokens/timings, batch metrics,
      # engine timestamps, selection native_actuals, or raw stream events.
      effective=settings_projection(request['response']['generation_settings'],payload)
      profile_id=digest(effective);profiles.setdefault(profile_id,effective);raw_profile_ids.add(profile_id);cell_profile_ids.add(profile_id)
      request_count[phase]+=1;cell_requests[phase]+=1;seen+=1
      if not first_location:first_location=f'/{phase}/{bi}/requests/{ri}/response/generation_settings'
   if seen==0 or len(raw_profile_ids)!=1:raise ValueError('missing or inconsistent resolved sampling settings within raw block')
   refs.append({'raw_ref':expected,'generation_settings_json_pointer_example':first_location,'all_request_settings_checked':seen,'sampling_cli_overrides':[]})
  if len(cell_profile_ids)!=1 or len(cell_payloads)!=1:raise ValueError('cell sampling configuration changed across blocks')
  profile_id=next(iter(cell_profile_ids));effective=profiles[profile_id];bias_counts[len(effective['logit_bias'])]+=1
  cell={'cell_id':row['cell_id'],'source_id':source_id,'sampling_profile_id':profile_id,'payload_projection':cell_payloads[0],'typed_policy_candidate':typed_candidate(effective),'backend_sampling':effective['backend_sampling'],'request_config_checks':dict(cell_requests),'raw_configuration_evidence':refs,'freeze_ref':freeze_ref,'collector_source_ref':collector_refs[freeze_key],'provenance_status':'payload_resolved_settings_native_modules_historical_headers_verified'}
  nonnull(cell);cells.append(cell)
  if (i+1)%25==0:print(json.dumps({'configuration_cells_projected':i+1,'of':len(rows)},ensure_ascii=False),flush=True)
 lineage=lineage_proof(runtime_refs)
 parity=ast_parity(source_refs)
 summary={'selected_cells':len(rows),'raw_blocks_verified':raw_count,'raw_container_bytes_hashed':raw_bytes,'resolved_request_settings_checked':dict(request_count),'source_cell_counts':dict(source_count),'distinct_sampling_profiles':len(profiles),'logit_bias_entry_count_by_cell':dict(sorted(bias_counts.items())),'all_temperature_zero':all(c['payload_projection']['temperature']==0 for c in cells),'all_top_k_one':all(c['payload_projection']['top_k']==1 for c in cells),'all_min_keep_zero':all(c['typed_policy_candidate']['min_keep']==0 for c in cells),'all_backend_sampling_false':all(not c['backend_sampling'] for c in cells),'all_effective_parameters_nonnull':True}
 result={'schema':'source-bound-native-sampling-parity/v1','status':'131_cell_static_sampling_configuration_verified','GPU_executed':False,'native_or_simulation_executed':False,'delay_or_cost_values_exported':False,'latency_fields_projected_or_used':False,'data_access_scope':'Hashed and parsed frozen raw containers but projected only request payload/config, response generation_settings, runtime refs, and source/header/build provenance. No timing values or generated content used.','selection_ref':ref(selection_path),'exporter_ref':ref(Path(__file__)),'summary':summary,'effective_profiles':profiles,'cells':cells,'source_snapshots':source_refs,'runtime_refs':list(runtime_refs.values()),'frozen_collector_refs':list(collector_refs.values()),'build_and_header_lineage':lineage,'simulator_source_snapshot_parity':parity,'canonicalization_note':'Only a logit_bias JSON null is replaced with an explicit negative_infinity enum, based on frozen ignore_eos request plus locked EOG-bias source path. No parameter is silently defaulted or coerced; native min_keep stays 0.','source_contract_limitations':['Additional model suppress tokens are merged by llama_vocab_get_suppress_tokens in common_sampler_init; their separate count is not serialized in generation_settings and must be bound through static vocabulary metadata if modeled.','Native disabled sampler constructors may leave empty sampler nodes; do not charge disabled penalty/DRY bodies as active algorithms.','This exporter establishes semantic policy/chain parity, never a measured latency or calibrated cost.']}
 write_new(P/'static_sampling_contract.json',result)
 write_new(P/'static_sampling_summary.json',{'schema':'sampling-static-audit-summary/v1','status':result['status'],'contract_ref':ref(P/'static_sampling_contract.json'),'summary':summary,'simulator_source_snapshot_parity':parity,'source_lineage_verified':True,'core_semantic_repair_authorized_by_task':True,'GPU_executed':False,'delays_exported':False})
 print(json.dumps({'completed':True,'contract_ref':ref(P/'static_sampling_contract.json'),'summary':summary,'parity':parity},ensure_ascii=False))
if __name__=='__main__':main()
