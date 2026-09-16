"""Pure host logic fixtures only; no counter, timing, CUDA or ETW capture is run."""
from pathlib import Path
import copy,importlib.util,json,sys
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import analyze,strict_gate

def phase():
 a=dict(qpc_before=0,qpc_after=10,user_100ns=0,kernel_100ns=0,cycles=100)
 b=dict(qpc_before=110,qpc_after=120,user_100ns=0,kernel_100ns=0,cycles=200)
 return dict(start=a,end=b,body_start_qpc=10,body_end_qpc=110,wall_ns=10000,snapshot_read_envelope_qpc_ticks=20,user_delta_100ns=0,kernel_delta_100ns=0,cycle_delta=100,CPU_service_ns=None,counter_precision_validated=False)

def pilot(pid=1):
 samples=[dict(kind=kind,amount=amount,repeat=repeat,measurement=phase()) for repeat in range(5) for kind,amounts in [('empty_loop',(0,128,512,2048)),('busy',(.25,1,4,16,64)),('Sleep',(1,4,16,64))] for amount in amounts]
 return dict(schema='host-service-counter-pilot/v1',status='recorded_precision_unproven',protocol_sha256='x',pid=pid,tid=1,qpc_frequency=10000000,sampling_start_qpc=0,sampling_finish_qpc=10000000,sampling_budget_ms=20000,GPU_API_called=False,cycles_converted_to_ns=False,counter_precision_validated=False,samples=samples)

def test_zero_counter_and_minimum_step_never_certify_time():
 out=analyze.pilot_summary([pilot(i) for i in (1,2,3)],'x')
 assert out['zero_counter_samples']==195 and out['counter_precision_validated'] is False and out['minimum_step_is_error_bound'] is False

def test_pilot_controls_and_total_budget_cannot_be_silently_truncated():
 r=pilot();r['samples'][0]=r['samples'][1]
 with pytest.raises(ValueError,match='control coverage'):analyze.validate_record(r,'x')
 r=pilot();r['sampling_finish_qpc']=200000001
 with pytest.raises(ValueError,match='sampling budget'):analyze.validate_record(r,'x')

def test_raw_counter_arithmetic_not_silently_corrected():
 x=phase();x['cycle_delta']=101
 with pytest.raises(ValueError,match='raw delta'):analyze.phase_ok(x)
 x=phase();x['CPU_service_ns']=0
 with pytest.raises(ValueError,match='falsely claim'):analyze.phase_ok(x)

def test_dpc_isr_union_is_subtracted_once_and_other_threads_separate():
 e=dict(coverage_start_qpc=0,coverage_end_qpc=1000,qpc_frequency=10000000,
 scheduled_intervals=[dict(pid=1,tid=2,cpu=0,start_qpc=0,end_qpc=100),dict(pid=1,tid=3,cpu=1,start_qpc=0,end_qpc=100)],
 interrupt_intervals=[dict(cpu=0,start_qpc=20,end_qpc=40,kind='DPC'),dict(cpu=0,start_qpc=30,end_qpc=50,kind='ISR')])
 a=analyze.active_service(e,1,2,0,100);b=analyze.active_service(e,1,None,0,100)
 assert a['active_CPU_ns']==7000 and b['active_CPU_ns']==17000
 assert a['timestamp_quantization_allowance_ns']>0
 with pytest.raises(ValueError,match='coverage'):analyze.active_service(e,1,2,-1,100)

def test_equal_k_triplet_and_full_rank_design():
 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
 assert [n*g for n,g in protocol['equal_kernel_controls']]==[512]*3
 rows={(n,g):{'service_ns':7+1000*n*g+250*g} for n in analyze.NODES for g in analyze.GROUPS}
 beta=analyze.fit3(rows)
 assert beta==pytest.approx([7,1000,250],rel=1e-6,abs=1e-6)
 with pytest.raises(ValueError,match='rank deficient'):analyze.fit3({(n,128):rows[n,128] for n in analyze.NODES})

def test_missing_etw_method_cannot_be_claimed_qualified():
 with pytest.raises(ValueError,match='separate ETW'):analyze.candidate([],[],[],'x',None)

def test_owner_is_command_queue_not_cpu_and_existing_frontend_unchanged():
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
 assert p['owner_correction']['modeled_resource']=='gpu0.command_queue'
 assert p['owner_correction']['kernel_frontend_ns']==1000
 assert p['owner_correction']['automatic_replacement_or_addition'] is False

def test_strict_inherited_environment_rejects_injection_and_dispatch_override():
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
 for key in ['CUDA_INJECTION64_PATH','GGML_CUDA_FORCE_MMQ','NSYS_INJECTION_LIBRARY_PATH']:
  with pytest.raises(ValueError,match='rejected'):strict_gate.environment_for(p,{key:'secret-should-not-print'})
 env,receipt=strict_gate.environment_for(p,{'SystemRoot':'C:/Windows','UNRELATED_SECRET':'omit'})
 assert 'UNRELATED_SECRET' not in env and receipt['runtime_dispatch_qualified_by_environment'] is False

def test_active_sim_and_new_probe_blocked_without_process_query():
 rows=[dict(pid=1,name='python.exe',cmdline=['python','F:/37_LLMsim/predict_stable_native_dataset.py']),dict(pid=2,name='host-service-probe.0001.exe',cmdline=[]),dict(pid=3,name='host-counter-probe.0001.exe',cmdline=[])]
 assert len(strict_gate.process_conflicts(rows,current_pid=99))==3

def test_pilot_budget_is_three_times_twenty_seconds():
 p=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))['pilot']
 assert p['processes']*p['sampling_budget_ms_per_process']==p['total_sampling_budget_ms']==60000
 assert p['any_timing_already_executed'] is False

def test_etw_missing_trace_binding_or_lost_events_fails(tmp_path):
 record=tmp_path/'record.json';record.write_text('{}')
 with pytest.raises(ValueError,match='ETW/probe binding'):analyze.validate_etw({},analyze.ref(record),{'qpc_frequency':100})

def test_build_and_runner_imports_do_not_start_native_or_counters(monkeypatch):
 import subprocess
 monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('no subprocess during module import'))
 monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('no subprocess during module import'))
 import build,run_controls
 assert build.HERE==HERE and run_controls.HERE==HERE


def test_crt_clock_dependency_does_not_excuse_explicit_probe_counters():
 import build
 evidence=build.validate_host_imports('QueryPerformanceCounter','ordinary_test_code','__security_init_cookie')
 assert evidence['CRT_startup_QPC_import'] and not evidence['runtime_clock_calls_instrumented']
 with pytest.raises(ValueError,match='object references'):build.validate_host_imports('QueryPerformanceCounter','__imp_QueryPerformanceCounter','__security_init_cookie')
 with pytest.raises(ValueError,match='experiment timing API'):build.validate_host_imports('GetThreadTimes','','__security_init_cookie')
 with pytest.raises(ValueError,match='unexplained'):build.validate_host_imports('QueryPerformanceCounter','','unknown')

def test_unbound_raw_and_success_without_gates_cannot_enter_etw(tmp_path,monkeypatch):
 monkeypatch.setattr(analyze,'HERE',tmp_path)
 folder=tmp_path/'runs'/'service_etw_chain.0001';folder.mkdir(parents=True);raw=folder/'record.json';raw.write_text('{}')
 with pytest.raises(ValueError,match='terminal receipt'):analyze.load_execution(raw,'x','service','etw')
 (folder/'finish.json').write_text(json.dumps(dict(schema='host-service-execution/v1',status='validated',returncode=0)))
 with pytest.raises(ValueError,match='runner gate missing'):analyze.load_execution(raw,'x','service','etw')
 with pytest.raises(ValueError,match='fixed run root'):analyze.load_execution(tmp_path/'record.json','x','service','etw')

def test_qualification_boolean_cannot_enable_measurement_admission():
 method=dict(schema='host-service-etw-method-qualification/v1',protocol_sha256='x',scope='QPC CSwitch intervals with DPC/ISR exclusion',passed=True)
 with pytest.raises(ValueError,match='admission is disabled'):analyze.candidate([],[],[],'x',method)

def test_path_adapter_preserves_predecessor_launch_and_dependence_validation():
 import run_controls
 p=json.loads((HERE.parent/'host_submission_probe'/'protocol.json').read_text(encoding='utf8'))
 r=dict(schema='host-service-probe/v1',mode='path',topology='chain',protocol_sha256='x',GPU_uuid=p['gpu_uuid'],status='path_qualified',profiling=True,cost_model_applied=False,GPU_event_used=False,counter_precision_validated=False,target_modules=p['target_modules'],completed_cases=3,cases=[])
 for n in (1,4,16):
  launches=[dict(index=i,symbol=p['kernel_symbol'],api_id=211,api_name='cudaLaunchKernel',attributes_source_qualified=True,geometry_observed=True,grid=[1,1,1],block=[1024,1,1],shared_bytes=128,ncols=4096,exit_seen=True,return_code=0,function=1,stream=0,input=100+i,output=101+i) for i in range(n)]
  r['cases'].append(dict(nodes=n,executed_graph_nodes=n,captured_launch_count=n,qualified=True,max_abs_error=0,launches=launches))
 run_controls.validate_raw(r,'path','chain','x')
 bad=copy.deepcopy(r);bad['cases'][1]['launches'][1]['input']=100
 with pytest.raises(ValueError,match='dependence'):run_controls.validate_raw(bad,'path','chain','x')
 bad=copy.deepcopy(r);bad['cases'][0]['launches'][0]['shared_bytes']=0
 with pytest.raises(ValueError,match='geometry'):run_controls.validate_raw(bad,'path','chain','x')
 bad=copy.deepcopy(r);bad['schema']='fabricated'
 with pytest.raises(ValueError,match='successor scope'):run_controls.validate_raw(bad,'path','chain','x')


def test_missing_full262_file_rejects_before_verifier_import(tmp_path):
 import run_controls
 runner=tmp_path/'run_candidate.py';runner.write_text('raise AssertionError("must not import")')
 with pytest.raises(ValueError,match='full262 terminal barrier missing'):run_controls.campaign_gate({'campaign_barrier':{'verifier_refs':[analyze.ref(runner)]}})

def test_full262_calls_existing_verifier_instead_of_accepting_marker(tmp_path):
 import run_controls
 runner=tmp_path/'run_candidate.py';runner.write_text('def barrier():\n raise ValueError("full identity validation rejected")\n')
 (tmp_path/'predictions_complete.json').write_text('{}')
 with pytest.raises(ValueError,match='full identity validation rejected'):run_controls.campaign_gate({'campaign_barrier':{'verifier_refs':[analyze.ref(runner)]}})

def test_budget_preflight_failure_has_retained_terminal_without_child(tmp_path,monkeypatch):
 import run_controls
 from types import SimpleNamespace
 monkeypatch.setattr(run_controls,'HERE',tmp_path)
 monkeypatch.setattr(run_controls,'process_gate',lambda:{'conflicts':[]})
 monkeypatch.setattr(run_controls.subprocess,'Popen',lambda *a,**k:pytest.fail('budget must precede child'))
 args=SimpleNamespace(mode='pilot',observation='direct',topology='chain',index=4,root=tmp_path/'runs',manifest=tmp_path/'manifest')
 with pytest.raises(ValueError,match='terminal/raw'):run_controls.execute(args)
 finish=json.loads((tmp_path/'runs/pilot_chain.0004/finish.json').read_text())
 assert finish['status']=='rejected' and finish['returncode'] is None and 'budget/index' in finish['error'] and finish['child_ref'] is None

def test_observation_interruption_never_kills_and_records_failed_terminal(tmp_path,monkeypatch):
 import run_controls
 from types import SimpleNamespace
 (tmp_path/'protocol.json').write_text('{}');fake={'compiled':{'pilot':{'path':'never-run.exe'}}}
 monkeypatch.setattr(run_controls,'HERE',tmp_path);monkeypatch.setattr(run_controls,'snapshot',lambda p:fake)
 monkeypatch.setattr(run_controls.strict_gate,'environment_for',lambda p:({},{}));monkeypatch.setattr(run_controls,'process_gate',lambda:{'conflicts':[]})
 calls=[]
 monkeypatch.setattr(run_controls,'campaign_gate',lambda p:calls.append('full262') or {'barrier':'fake'})
 class NeverExecutedChild:
  pid=123
  def wait(self):raise KeyboardInterrupt('synthetic observation interruption')
  def poll(self):return None
  def kill(self):pytest.fail('forbidden kill')
  def terminate(self):pytest.fail('forbidden terminate')
 monkeypatch.setattr(run_controls.subprocess,'Popen',lambda *a,**k:NeverExecutedChild())
 args=SimpleNamespace(mode='pilot',observation='direct',topology='chain',index=1,root=tmp_path/'runs',manifest=tmp_path/'manifest')
 with pytest.raises(ValueError,match='terminal/raw'):run_controls.execute(args)
 finish=json.loads((tmp_path/'runs/pilot_chain.0001/finish.json').read_text())
 assert finish['status']=='rejected' and finish['observation_interrupted'] and finish['child_may_be_live'] and finish['child_ref']
 assert calls==['full262','full262'] and finish['inputs_unchanged']

def test_relative_successor_runner_is_recognized_as_conflict():
 assert strict_gate.process_conflicts([dict(pid=33,name='python.exe',cmdline=['python','run_controls.py','--execute'])],current_pid=1)

def test_conditional_diagnostics_never_become_formal_precision():
 out=analyze.pilot_summary([pilot(i) for i in (1,2,3)],'x')
 assert len(out['controlled_windows'])==13 and out['conditional_development_use']
 assert not out['absolute_service_accuracy_guarantee'] and out['long_busy_window_ratio_difference_pct'] is None


def test_owned_runners_have_no_implicit_subprocess_run_or_termination():
 import ast
 for name in ('run_controls.py','build.py','strict_gate.py'):
  tree=ast.parse((HERE/name).read_text(encoding='utf8'))
  calls=[x.func for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
  assert not any(c.attr in ('kill','terminate') or (c.attr=='run' and isinstance(c.value,ast.Name) and c.value.id=='subprocess') for c in calls),name



def test_GPU_relative_runner_and_unknown_spawn_blocked():
 rows=[dict(pid=1,name='gpu-operator-timing.exe',cmdline=None),
       dict(pid=2,name='python.exe',cmdline='python runner.py run --run-dir run.0001'),
       dict(pid=3,ppid=2,name='python.exe',cmdline=['python','--multiprocessing-fork']),
       dict(pid=4,ppid=999,name='python.exe',cmdline=['python','--multiprocessing-fork']),
       dict(pid=5,name='python.exe',cmdline=['python','-m','pytest','test_controls.py']),
       dict(pid=6,ppid=5,name='python.exe',cmdline=['python','--multiprocessing-fork']),
       dict(pid=7,ppid=8,name='python.exe',cmdline=['python','--multiprocessing-fork']),
       dict(pid=8,ppid=7,name='python.exe',cmdline=['python','--multiprocessing-fork'])]
 bad=strict_gate.process_conflicts(rows,current_pid=99)
 assert [r['pid'] for r in bad]==[1,2,3,4,7,8]
 assert all('cmdline' not in r for r in bad)


def test_predecessor_analyzer_is_pinned_and_rechecked_after_import(tmp_path,monkeypatch):
 import run_controls
 local=tmp_path/'host_service_controls';local.mkdir()
 parent=tmp_path/'host_submission_probe';parent.mkdir();source=parent/'analyze.py'
 source.write_text('VALUE=42\n')
 (local/'protocol.json').write_text(json.dumps({'base_source_refs':[analyze.ref(source)]}))
 monkeypatch.setattr(run_controls,'HERE',local)
 module,pinned=run_controls.predecessor_analyzer();assert module.VALUE==42
 source.write_text('VALUE=43\n')
 with pytest.raises(ValueError,match='identity differs'):run_controls.predecessor_analyzer()
 source.write_text('from pathlib import Path\nPath(__file__).write_text("VALUE=44")\n')
 (local/'protocol.json').write_text(json.dumps({'base_source_refs':[analyze.ref(source)]}))
 with pytest.raises(ValueError,match='identity differs'):run_controls.predecessor_analyzer()


def test_unpinned_predecessor_analyzer_cannot_load(tmp_path,monkeypatch):
 import run_controls
 (tmp_path/'protocol.json').write_text('{"base_source_refs":[]}')
 monkeypatch.setattr(run_controls,'HERE',tmp_path)
 with pytest.raises(ValueError,match='uniquely pinned'):run_controls.predecessor_analyzer()


@pytest.mark.parametrize('failure',['start_ref','child_ref','record_ref','missing_record','post_identity','post_process','post_campaign'])
def test_final_failures_never_validated_and_leave_finish(tmp_path,monkeypatch,failure):
 import run_controls
 from types import SimpleNamespace
 monkeypatch.setattr(run_controls,'HERE',tmp_path)
 (tmp_path/'protocol.json').write_text('{}')
 fake={'compiled':{'pilot':{'path':'never-run.exe'}}};snapshots=[]
 def snapshot(path):
  snapshots.append(1)
  if failure=='post_identity' and len(snapshots)>1:raise OSError('synthetic identity error')
  return fake
 monkeypatch.setattr(run_controls,'snapshot',snapshot)
 monkeypatch.setattr(run_controls.strict_gate,'environment_for',lambda p:({},{}))
 processes=[]
 def process():
  processes.append(1)
  if failure=='post_process' and len(processes)==3:raise OSError('synthetic process error')
  return {'conflicts':[]}
 monkeypatch.setattr(run_controls,'process_gate',process)
 campaigns=[]
 def campaign(p):
  campaigns.append(1)
  if failure=='post_campaign' and len(campaigns)>1:raise OSError('synthetic campaign error')
  return {'barrier':'fixed'}
 monkeypatch.setattr(run_controls,'campaign_gate',campaign)
 directory=tmp_path/'runs/pilot_chain.0001'
 class Child:
  pid=123
  def wait(self):
   (directory/'record.json').write_text('{}');return 0
  def kill(self):pytest.fail('no kill')
  def terminate(self):pytest.fail('no terminate')
 monkeypatch.setattr(run_controls.subprocess,'Popen',lambda *a,**k:Child())
 def validate(*a):
  if failure=='missing_record':(directory/'record.json').unlink()
 monkeypatch.setattr(run_controls,'validate_raw',validate)
 original=run_controls.build.ref
 def broken_ref(path):
  target={'start_ref':'start.json','child_ref':'child.json','record_ref':'record.json'}.get(failure)
  if Path(path).name==target:raise OSError('synthetic final hash error')
  return original(path)
 monkeypatch.setattr(run_controls.build,'ref',broken_ref)
 args=SimpleNamespace(mode='pilot',observation='direct',topology='chain',index=1,root=tmp_path/'runs',manifest=tmp_path/'manifest')
 with pytest.raises(ValueError,match='rejected'):run_controls.execute(args)
 finish=json.loads((directory/'finish.json').read_text())
 assert finish['status']=='rejected' and finish['raw_validated'] is True
 if failure in ('start_ref','child_ref','record_ref','missing_record'):assert finish['reference_errors']
 if failure=='post_process':
  assert finish['campaign_gates_passed'] is True and 'post_process_error' in finish
