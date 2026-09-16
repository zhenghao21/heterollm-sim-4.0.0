"""Host-only tests; synthetic records, no GPU/target executable calls."""
from pathlib import Path
import copy,json,sys
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import analyze
import build
import run_probe

def fixture_record(mode='path',topology='chain',pid=1):
 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'));sha=build.ref(HERE/'protocol.json')['sha256']
 record={'schema':'host-submission-target-probe/v1','mode':mode,'topology':topology,'protocol_sha256':sha,'GPU_uuid':protocol['gpu_uuid'],
 'status':'path_qualified' if mode=='path' else 'timing_recorded_unadmitted','profiling':mode=='path','cost_model_admitted':False,'LLM_graph_reuse_internal_observed':False,'pid':pid,'completed_cases':3,
 'target_modules':[{'path':r['path'],'sha256':r['sha256']} for r in protocol['target_modules']],'cases':[]}
 for n in (1,4,16):
  if mode=='path':
   launches=[]
   for i in range(n):
    launches.append({'index':i,'symbol':protocol['kernel_symbol'],'api_id':430,'api_name':'cudaLaunchKernelExC','attributes_source_qualified':True,'geometry_observed':True,'grid':[1,1,1],'block':[1024,1,1],'shared_bytes':128,'ncols':4096,'exit_seen':True,'return_code':0,'function':123,'stream':0,
    'input':0x2000+(i-1)*0x100 if topology=='chain' and i else 0x1000,'output':0x2000+i*0x100})
   record['cases'].append({'nodes':n,'executed_graph_nodes':n,'captured_launch_count':n,'qualified':True,'max_abs_error':0,'launches':launches})
  else:
   phase={'wall_ns':1000.0,'thread_user_100ns_ticks':0,'thread_kernel_100ns_ticks':0,'thread_cycles':100,'thread_CPU_service_inferred':False,'thread_CPU_counter_threshold_met':False}
   sample={'graph_replays':128,'pure_GPU_service_ms':None,'GPU_event_envelope_ms_includes_host_sync_gaps':1,'max_abs_error':0,**{p:copy.deepcopy(phase) for p in analyze.PHASES}}
   record['cases'].append({'nodes':n,'samples':[{**copy.deepcopy(sample),'sample':i} for i in range(31)]})
 return record,protocol,sha

@pytest.mark.parametrize('topology',['chain','fanout'])
def test_qualified_topology_requires_actual_N_kernel_evidence(topology):
 r,p,h=fixture_record(topology=topology);assert analyze.validate_record(r,p,'path',topology,h)

@pytest.mark.parametrize('mutation',['kernel_count','graph_nodes','symbol','api','attributes','geometry_missing','stream_missing','pointer','dependency','exit','identity','profiling'])
def test_path_mismatch_rejected(mutation):
 r,p,h=fixture_record();case=r['cases'][1];x=case['launches'][1]
 if mutation=='kernel_count':case['captured_launch_count']=3
 elif mutation=='graph_nodes':case['executed_graph_nodes']=3
 elif mutation=='symbol':x['symbol']='another_kernel'
 elif mutation=='api':x['api_id']=999
 elif mutation=='attributes':x['attributes_source_qualified']=False
 elif mutation=='geometry_missing':x['geometry_observed']=False
 elif mutation=='stream_missing':x['stream']=None
 elif mutation=='pointer':x['output']=0
 elif mutation=='dependency':x['input']=0x1000
 elif mutation=='exit':x['exit_seen']=False
 elif mutation=='identity':r['target_modules'][0]['sha256']='0'*64
 elif mutation=='profiling':r['profiling']=False
 with pytest.raises(ValueError):analyze.validate_record(r,p,'path','chain',h)

def test_zero_thread_counters_are_unresolved_not_zero_service():
 records=[fixture_record('timing',pid=i)[0] for i in range(5)];_,p,h=fixture_record('timing')
 result=analyze.summarize_processes(records,p,h,'chain')
 assert result['cost_model_admitted'] is False
 for phases in result['nodes'].values():
  for key in analyze.PHASES:
   phase=phases[key]
   assert phase['stable_wall_under_preregistered_rule'] is True
   assert phase['calling_thread_CPU_counter_threshold_met_all_samples'] is False

def test_GPU_envelope_cannot_be_promoted_to_kernel_service():
 r,p,h=fixture_record('timing');r['cases'][0]['samples'][0]['pure_GPU_service_ms']=1
 with pytest.raises(ValueError,match='envelope'):analyze.validate_record(r,p,'timing','chain',h)

def test_timing_profiler_record_is_rejected():
 r,p,h=fixture_record('timing');r['profiling']=True
 with pytest.raises(ValueError):analyze.validate_record(r,p,'timing','chain',h)

def test_failed_or_censored_samples_remain_failure():
 records=[fixture_record('timing',pid=i)[0] for i in range(5)];_,p,h=fixture_record('timing')
 records[0]['cases'][0]['samples'].pop()
 with pytest.raises(ValueError,match='coverage'):analyze.summarize_processes(records,p,h,'chain')

def test_preregistered_variability_rule_rejects_instability():
 records=[fixture_record('timing',pid=i)[0] for i in range(5)];_,p,h=fixture_record('timing')
 records[0]['cases'][0]['samples'][0]['submit_batch']['wall_ns']=1e6
 result=analyze.summarize_processes(records,p,h,'chain')
 assert not result['nodes']['1']['submit_batch']['stable_wall_under_preregistered_rule']

def test_GPU_runner_requires_explicit_authorization_before_any_dependency_or_process(monkeypatch):
 monkeypatch.setattr(sys,'argv',['run_probe.py','path','--topology','chain','--manifest','unavailable.json'])
 monkeypatch.setattr(run_probe,'prerequisites',lambda *a:pytest.fail('must reject before run prerequisites'))
 with pytest.raises(ValueError,match='authorization'):run_probe.main()

def test_source_has_no_uncontrolled_target_run_in_build_entry():
 text=(HERE/'build.py').read_text(encoding='utf-8')
 assert "choices=('verify','build','host-test')" in text
 assert "subprocess.run([manifest['host_executable']['path']]" in text
 assert "subprocess.run([manifest['target_executable']['path']]" not in text

def test_stream_limitation_and_owners_are_explicit_in_protocol():
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
 assert p['node_counts']==[1,4,16] and p['timing_mode']['independent_processes']==5
 assert any('stream' in x for x in p['limitations'])
 assert any('1000ns/kernel' in x and '250ns/group' in x and '128+12G' in x for x in p['limitations'])
 assert p['acceptance']['no_cost_model_admission'] is True


def test_failed_postrun_identity_gate_blocks_future_timing(tmp_path):
 (tmp_path/'finish.json').write_text(json.dumps({'returncode':0,'inputs_unchanged':False}))
 with pytest.raises(ValueError,match='identity gate'):
  run_probe.validated_finish(tmp_path,{},tmp_path/'unused_manifest.json','path','chain')


@pytest.mark.parametrize('name', ['CUDA_INJECTION64_PATH', 'CUDA_LAUNCH_BLOCKING', 'GGML_CUDA_FORCE_MMQ',
    'GGML_UNKNOWN_FUTURE_DISPATCH', 'CUBLAS_WORKSPACE_CONFIG', 'NVTX_INJECTION64_PATH',
    'NSYS_INJECTION_LIBRARY_PATH', 'OMP_NUM_THREADS', 'SOME_PROFILER_HOOK', 'HOST_PROBE_AUTHORIZED_GPU_RUN'])
def test_dispatch_environment_cannot_be_silently_sanitized(name):
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
 with pytest.raises(ValueError,match='Inherited dispatch'):
  run_probe.environment_for(p, {'SystemRoot':r'C:\Windows',name:'untrusted-secret-value'})


def test_minimal_environment_has_locked_PATH_and_explicit_unqualified_policy():
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
 env,policy=run_probe.environment_for(p,{'SystemRoot':r'C:\Windows','PATH':'unlocked-dir',
   'UNRELATED_API_KEY':'do-not-record','GGML_CUDA_DISABLE_GRAPHS':'1'})
 assert env['GGML_CUDA_DISABLE_GRAPHS']=='1'
 assert env['HOST_PROBE_AUTHORIZED_GPU_RUN']=='1'
 assert 'unlocked-dir' not in env['PATH'] and 'UNRELATED_API_KEY' not in env
 assert 'do-not-record' not in json.dumps(policy)
 assert policy['inherited_environment_verified'] is False
 assert policy['runtime_dispatch_qualified_by_environment'] is False
 assert 'UNRELATED_API_KEY' in policy['omitted_parent_variable_names']
 assert policy['dll_directories']==list(dict.fromkeys(str(Path(r['path']).parent) for r in p['target_modules']))


def test_process_conflicts_allow_compilation_and_host_only_work():
 rows=[
  {'pid':1,'name':'llama-server.exe','cmdline':None},
  {'pid':2,'name':'python.exe','cmdline':['python','run_candidate.py','--full262']},
  {'pid':3,'name':'host-submission-target.0005.exe','cmdline':None},
  {'pid':4,'name':'python.exe','cmdline':None},
  {'pid':5,'name':'python.exe','cmdline':['python',r'F:\codex_project\37_LLMsim\unknown.py']},
  {'pid':6,'name':'python.exe','cmdline':['python',r'F:\codex_project\37_LLMsim\build.py','build']},
  {'pid':7,'name':'python.exe','cmdline':['python',r'F:\codex_project\37_LLMsim\build.py','host-test']},
  {'pid':8,'name':'python.exe','cmdline':['python','-m','pytest',r'F:\codex_project\37_LLMsim\test_host_probe.py']},
  {'pid':9,'name':'host-decoder-tests.0005.exe','cmdline':None},
  {'pid':10,'name':'cl.exe','cmdline':None},
  {'pid':11,'name':'explorer.exe','cmdline':None},
  {'pid':12,'name':'python.exe','cmdline':['python','run_probe.py']},
  {'pid':13,'name':'python.exe','cmdline':['python','run_probe.py']},
 ]
 conflicts=run_probe.process_conflicts(rows,current_pid=13)
 assert [r['pid'] for r in conflicts]==[1,2,3,4,5,12]
 assert all('cmdline' not in r for r in conflicts)


@pytest.fixture
def synthetic_execution(monkeypatch,tmp_path):
 from types import SimpleNamespace
 record,protocol,sha=fixture_record()
 manifest=tmp_path/'build_manifest.json';manifest.write_text('{}')
 frozen={'build_ref':build.ref(manifest),'protocol_ref':{'sha256':sha},'inputs':{'fixed':True},
         'loaded_code_refs':[{'runner':'one','analyze':'one','build':'one'}]}
 m={'target_executable':{'path':'NEVER_EXECUTE_TARGET.exe'}}
 monkeypatch.setattr(run_probe,'HERE',tmp_path)
 monkeypatch.setattr(run_probe,'prerequisites',lambda *a:(protocol,m,copy.deepcopy(frozen)))
 monkeypatch.setattr(run_probe,'identity_snapshot',lambda *a:copy.deepcopy(frozen))
 monkeypatch.setattr(run_probe,'environment_for',lambda p:({}, {'synthetic':True}))
 monkeypatch.setattr(run_probe,'process_snapshot',lambda:{'conflicts':[]})
 def launch(argv,env,directory):
  (directory/'record.json').write_text(json.dumps(record));return 0
 monkeypatch.setattr(run_probe,'launch_process',launch)
 monkeypatch.setattr(run_probe.subprocess,'Popen',lambda *a,**k:pytest.fail('GPU/process execution forbidden in synthetic tests'))
 a=SimpleNamespace(mode='path',topology='chain',index=1,manifest=manifest)
 return a,tmp_path/'runs/path_chain_0001',frozen,record


def test_success_receipt_binds_pre_post_and_start(synthetic_execution):
 a,d,frozen,_=synthetic_execution
 result=run_probe.execute(a)
 assert result['status']=='validated' and result['process_gates_passed'] is True
 assert result['identity_before']==result['identity_after']==frozen
 assert result['start_ref']==build.ref(d/'start.json')
 assert result['record_ref']==build.ref(d/'record.json')


@pytest.mark.parametrize('failure',['launch_oserror','missing_record','bad_record','exit_failure','post_verify_error',
 'runner_changed','analyzer_changed','helper_changed','protocol_changed','manifest_changed','post_process_conflict',
 'post_process_error','pre_process_conflict','environment_rejected','interrupted'])
def test_every_failure_has_immutable_finish(synthetic_execution,monkeypatch,failure):
 a,d,frozen,record=synthetic_execution
 if failure in ('launch_oserror','interrupted'):
  def launch(*args):
   if failure=='interrupted':raise KeyboardInterrupt('test interruption')
   raise OSError('test launch failed')
  monkeypatch.setattr(run_probe,'launch_process',launch)
 elif failure in ('missing_record','exit_failure'):
  monkeypatch.setattr(run_probe,'launch_process',lambda *a:0 if failure=='missing_record' else 9)
 elif failure=='bad_record':
  def launch(argv,env,directory):
   (directory/'record.json').write_text('{}');return 0
  monkeypatch.setattr(run_probe,'launch_process',launch)
 elif failure in ('post_verify_error','runner_changed','analyzer_changed','helper_changed','protocol_changed','manifest_changed'):
  calls=[]
  def identity(*args):
   calls.append(1)
   if len(calls)==1:return copy.deepcopy(frozen)
   if failure=='post_verify_error':raise ValueError('verify_inputs failure')
   changed=copy.deepcopy(frozen)
   if failure=='protocol_changed':changed['protocol_ref']={'sha256':'changed'}
   elif failure=='manifest_changed':changed['build_ref']={'sha256':'changed'}
   else:changed['loaded_code_refs'][0][{'runner_changed':'runner','analyzer_changed':'analyze','helper_changed':'build'}[failure]]='changed'
   return changed
  monkeypatch.setattr(run_probe,'identity_snapshot',identity)
 elif failure in ('post_process_conflict','post_process_error','pre_process_conflict'):
  calls=[]
  def processes():
   calls.append(1)
   if failure=='post_process_error' and len(calls)>1:raise OSError('inventory failed')
   return {'conflicts':[{'pid':77}] if failure=='pre_process_conflict' or len(calls)>1 else []}
  monkeypatch.setattr(run_probe,'process_snapshot',processes)
  if failure=='pre_process_conflict':
   monkeypatch.setattr(run_probe,'launch_process',lambda *a:pytest.fail('must not start while simulator live'))
 else:
  monkeypatch.setattr(run_probe,'environment_for',lambda *a:(_ for _ in ()).throw(ValueError('environment rejected')))
 with pytest.raises(RuntimeError,match='failed attempt preserved'):run_probe.execute(a)
 finish=json.loads((d/'finish.json').read_text(encoding='utf-8'))
 assert finish['status']=='failed' and finish['errors']
 assert finish['termination_requested'] is False and finish['cost_model_admitted'] is False
 with pytest.raises(ValueError,match='identity gate'):run_probe.validated_finish(d,{},a.manifest,'path','chain')
 with pytest.raises(FileExistsError):run_probe.execute(a)


def test_wait_interruption_does_not_kill_child(monkeypatch,tmp_path):
 class Child:
  pid=12345
  def wait(self):raise KeyboardInterrupt('fixture interruption')
  def poll(self):return None
  def kill(self):pytest.fail('must not kill')
  def terminate(self):pytest.fail('must not terminate')
 monkeypatch.setattr(run_probe.subprocess,'Popen',lambda *a,**k:Child())
 with pytest.raises(KeyboardInterrupt):run_probe.launch_process(['fake'],{},tmp_path)
 receipt=json.loads((tmp_path/'child_observation_error.json').read_text(encoding='utf-8'))
 assert receipt['observed_returncode'] is None and receipt['termination_requested'] is False
 assert not (tmp_path/'child_exited.json').exists()


def test_build_identity_includes_all_execution_helpers():
 assert {'run_probe.py','analyze.py','build.py','protocol.json','README.md','test_host_probe.py'}<=set(build.SOURCES)
 assert {Path(r['path']).name for r in run_probe.LOADED_CODE_REFS}=={'run_probe.py','analyze.py','build.py'}


def test_source_modified_after_import_cannot_be_blessed_by_new_manifest(tmp_path,monkeypatch):
 source=tmp_path/'helper.py';source.write_text('one')
 monkeypatch.setattr(run_probe,'LOADED_CODE_REFS',[build.ref(source)])
 source.write_text('two')
 with pytest.raises(ValueError,match='Loaded runner/analyzer/build helper code changed'):
  run_probe.identity_snapshot(tmp_path/'manifest_need_not_exist.json')



def test_toolchain_search_variables_are_omitted_not_dispatch_qualification():
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
 env,policy=run_probe.environment_for(p, {'CUDA_PATH':'unlocked-toolkit', 'CUDA_PATH_V12_9':'another-toolkit'})
 assert 'CUDA_PATH' not in env and 'CUDA_PATH_V12_9' not in env
 assert 'unlocked-toolkit' not in json.dumps(policy)
 assert set(policy['omitted_parent_variable_names'])=={'CUDA_PATH','CUDA_PATH_V12_9'}
 assert policy['inherited_environment_verified'] is False



def test_process_inventory_uses_OS_not_GPU_and_does_not_log_commandlines(monkeypatch):
 from types import SimpleNamespace
 def query(argv,**kwargs):
  assert argv[0].lower().endswith('powershell.exe')
  assert 'Get-CimInstance Win32_Process' in argv[-1]
  return SimpleNamespace(stdout=json.dumps([
   {'ProcessId':201,'Name':'python.exe','CommandLine':'python run_candidate.py --secret do-not-log'},
   {'ProcessId':202,'Name':'explorer.exe','CommandLine':'desktop-gpu-client'}]))
 monkeypatch.setattr(run_probe.subprocess,'run',query)
 result=run_probe.process_snapshot()
 assert [r['pid'] for r in result['conflicts']]==[201]
 assert 'do-not-log' not in json.dumps(result)
 assert result['desktop_GPU_process_absence_required'] is False
 assert result['background_GPU_isolation_verified'] is False
 assert result['continuous_exclusivity_verified'] is False


def test_inventory_query_failure_is_not_idle(monkeypatch):
 def fail(*args,**kwargs):raise OSError('CIM unavailable')
 monkeypatch.setattr(run_probe.subprocess,'run',fail)
 with pytest.raises(OSError):run_probe.process_snapshot()
