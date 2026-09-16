"""Explicit future execution only. Build and host tests never enter this module's execute path."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys
import build,analyze,strict_gate
HERE=Path(__file__).resolve().parent
LOADED=[build.ref(x) for x in (__file__,build.__file__,analyze.__file__,strict_gate.__file__)]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def snapshot(manifest):
 for r in LOADED:
  if build.ref(r['path'])!=r:raise ValueError('loaded code drift')
 m=build.verify_manifest(manifest)
 return dict(build_ref=build.ref(manifest),inputs=m['inputs'],compiled={k:v['executable'] for k,v in m['variants'].items()},loaded_code_refs=LOADED)
def process_gate():
 p=strict_gate.process_snapshot()
 if p['conflicts']:raise ValueError('project work still running: '+json.dumps(p['conflicts']))
 return p
def campaign_gate(protocol):
 refs=protocol['campaign_barrier']['verifier_refs']
 for item in refs:analyze.check_ref(item)
 runner=Path(refs[0]['path']);terminal=runner.parent/'predictions_complete.json'
 if not terminal.is_file():raise ValueError('R27 full262 terminal barrier missing')
 spec=importlib.util.spec_from_file_location('host_service_r27_terminal_guard',runner);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 record,bref=module.barrier() # Existing full identity/262 terminal validation; never score or predict.
 if record.get('schema')!='r27-full262-terminal-barrier/v1' or record.get('terminal_count')!=262 or record.get('failures_preserved') is not True or set(record.get('arms',{}))!={'off','on'}:raise ValueError('R27 full262 terminal barrier differs')
 for item in refs:analyze.check_ref(item)
 return dict(verifier_refs=refs,barrier_ref=bref,controls_ref=record['controls_ref'],terminal_count=262,scoring_executed=False)

def predecessor_analyzer():
 expected_path=(HERE.parent/'host_submission_probe'/'analyze.py').resolve()
 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
 matches=[r for r in protocol['base_source_refs'] if Path(r['path']).resolve()==expected_path]
 if len(matches)!=1:raise ValueError('predecessor analyzer must be uniquely pinned')
 pinned=matches[0];analyze.check_ref(pinned)
 # Execute these exact verified bytes, avoiding a stale timestamp-based pyc cache.
 data=expected_path.read_bytes()
 if len(data)!=pinned['bytes'] or hashlib.sha256(data).hexdigest()!=pinned['sha256']:raise ValueError('predecessor analyzer changed before import')
 import types
 module=types.ModuleType('host_service_predecessor_analyzer');module.__file__=str(expected_path)
 exec(compile(data,str(expected_path),'exec'),module.__dict__)
 analyze.check_ref(pinned)
 return module,pinned

def validate_raw(r,mode,topology,sha):
 if mode=='path':
  analyze.need(r.get('schema')=='host-service-probe/v1' and r.get('cost_model_applied') is False and r.get('GPU_event_used') is False and r.get('counter_precision_validated') is False,'path successor scope')
  # Exact predecessor decoder/shape gate; adapt field names only in memory.
  base=HERE.parent/'host_submission_probe';a,pinned=predecessor_analyzer()
  protocol=json.loads((base/'protocol.json').read_text(encoding='utf8'))
  v=dict(r,schema='host-submission-target-probe/v1',cost_model_admitted=False,LLM_graph_reuse_internal_observed=False)
  a.validate_record(v,protocol,'path',topology,sha);analyze.check_ref(pinned)
 else:analyze.validate_record(r,sha)
def required_path(root,manifest,sha):
 for topology in ('chain','fanout'):
  directory=root/('path_'+topology+'.0001');f=json.loads((directory/'finish.json').read_text(encoding='utf8'))
  if f['status']!='validated' or f['inputs_unchanged'] is not True or f['process_gates_passed'] is not True or f['returncode']!=0:raise ValueError('both qualified topology paths required')
  if f['identity_before']!=f['identity_after'] or f['identity_after']!=snapshot(manifest):raise ValueError('path qualification identity drift')
  analyze.check_ref(f['record_ref']);validate_raw(json.loads(Path(f['record_ref']['path']).read_text(encoding='utf8')),'path',topology,sha)
def execute(a):
 # Reserve the immutable attempt before *any* eligibility check, including budget.
 # A rejected preflight consumes that attempt; the next attempt cannot retry it.
 prefix=a.mode+('_'+a.observation if a.mode=='service' else '')+'_'+a.topology
 directory=HERE/'runs'/(prefix+'.%04d'%a.index);directory.mkdir(parents=True,exist_ok=False)
 finish=dict(schema='host-service-execution/v1',mode=a.mode,topology=a.topology,process_index=a.index,status='rejected',returncode=None,inputs_unchanged=False,process_gates_passed=False,campaign_gates_passed=False,cost_model_applied=False,observation_requested=a.observation,external_ETW_state_verified=False)
 start=dict(schema='host-service-start/v1',mode=a.mode,topology=a.topology,process_index=a.index,created_utc=now(),argv=None,identity_before=None,observation_requested=a.observation,background_GPU_isolation_verified=False)
 before=None;protocol=None;child=None
 try:
  if a.root.resolve()!=(HERE/'runs').resolve():raise ValueError('fixed run root prevents budget/retry bypass')
  if a.mode!='service' and a.observation!='direct':raise ValueError('pilot/path do not accept external ETW cohorts')
  if a.mode=='pilot' and a.topology!='chain':raise ValueError('pilot is topology-independent; only one three-process pilot cohort')
  limit=3 if a.mode=='pilot' else 1 if a.mode=='path' else 5
  if not 1<=a.index<=limit:raise ValueError('process budget/index exceeded')
  for i in range(1,a.index):
   previous=json.loads((HERE/'runs'/(prefix+'.%04d'%i)/'finish.json').read_text(encoding='utf8'))
   if previous['status']!='validated':raise ValueError('earlier failure retained; no automatic retry')
  protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf8'));sha=build.ref(HERE/'protocol.json')['sha256']
  before=snapshot(a.manifest);start['identity_before']=before;finish['identity_before']=before
  env,policy=strict_gate.environment_for(protocol);start['environment_policy']=policy
  start['process_before']=process_gate()
  start['campaign_before']=campaign_gate(protocol);finish['campaign_before']=start['campaign_before']
  if a.mode=='service':required_path(a.root,a.manifest,sha)
  start['process_before']=process_gate() # Recheck after expensive full262 validation.
  if a.mode=='pilot':exe=before['compiled']['pilot']['path'];env['HOST_SERVICE_COUNTER_AUTH']='1';args=[exe,str((directory/'record.json').resolve())]
  else:exe=before['compiled']['service']['path'];args=[exe,a.mode,a.topology,str((directory/'record.json').resolve())]
  start['argv']=args;build.write_new(directory/'start.json',start)
  with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
   child=subprocess.Popen(args,env=env,cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
   build.write_new(directory/'child.json',dict(pid=child.pid,created_utc=now()))
   try:finish['returncode']=child.wait()
   except BaseException as error:
    finish['observation_interrupted']=True;finish['child_may_be_live']=child.poll() is None
    raise RuntimeError('observation interrupted; child left running with saved PID') from error
  if finish['returncode']!=0:raise ValueError('probe failed (including sampling budget overrun)')
  raw=json.loads((directory/'record.json').read_text(encoding='utf8'));validate_raw(raw,a.mode,a.topology,sha);finish['raw_validated']=True
 except BaseException as error:finish['error']=type(error).__name__+': '+str(error)
 # Always retain a failed preflight/observation terminal. No timeout/termination path.
 if not(directory/'start.json').exists():build.write_new(directory/'start.json',start)
 if before is not None:
  try:
   after=snapshot(a.manifest);finish.update(identity_after=after,inputs_unchanged=after==before)
  except BaseException as error:finish['post_identity_error']=str(error)
 try:finish['process_after']=process_gate();finish['process_gates_passed']=isinstance(start.get('process_before'),dict)
 except BaseException as error:finish['post_process_error']=str(error)
 if protocol is not None and finish.get('campaign_before') is not None:
  try:
   finish['campaign_after']=campaign_gate(protocol);finish['campaign_gates_passed']=finish['campaign_after']==finish['campaign_before']
   finish['process_after']=process_gate()
  except BaseException as error:finish['post_campaign_error']=str(error);finish['campaign_gates_passed']=False
 finish['reference_errors']=[]
 for key,file in [('start_ref','start.json'),('child_ref','child.json'),('record_ref','record.json')]:
  try:
   path=directory/file
   finish[key]=build.ref(path) if path.exists() else None
   if finish.get('raw_validated') and finish[key] is None:raise ValueError('required final evidence missing: '+file)
  except BaseException as error:
   finish[key]=None;finish['reference_errors'].append(dict(file=file,error=type(error).__name__+': '+str(error)))
 # No captured error can be overridden by otherwise true gates.
 errors_present=any(key=='error' or key.startswith('post_') and key.endswith('_error') for key in finish) or bool(finish['reference_errors'])
 if not errors_present and finish.get('raw_validated') and finish['returncode']==0 and finish['inputs_unchanged'] and finish['process_gates_passed'] and finish['campaign_gates_passed']:finish['status']='validated'
 finish['finished_utc']=now();build.write_new(directory/'finish.json',finish)
 if finish['status']!='validated':raise ValueError('rejected; all terminal/raw evidence retained')
 print(json.dumps({'status':'validated','record':finish['record_ref'],'cost_model_applied':False}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True);m.add_argument('--check',action='store_true');m.add_argument('--execute',action='store_true');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--mode',choices=['pilot','path','service']);p.add_argument('--topology',choices=['chain','fanout'],default='chain');p.add_argument('--index',type=int,default=1);p.add_argument('--root',type=Path,default=HERE/'runs');p.add_argument('--observation',choices=['direct','etw'],default='direct');a=p.parse_args()
 if a.check:print(json.dumps({'identity':snapshot(a.manifest),'any_timing_executed':False,'execution_ready_not_checked':True,'requires_actual_R27_full262_and_idle':True}))
 else:
  if a.mode is None:p.error('--mode required for execution')
  execute(a)
