"""Prepare or explicitly launch the frozen 6-configuration synthetic collection matrix."""
from __future__ import annotations
import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
from common import HERE,PROBE,PILOT,POLICY,load,write_new,ref,verify_ref,verify_freeze,utc,IdentityError,verify_clock_receipt


def plan(configs):
    stages=[]
    ordered=list(enumerate(configs))
    for config_index,config in ordered:
        for pair in range(POLICY['process_pairs']):
            order=('direct','profile') if (config_index+pair)%2==0 else ('profile','direct')
            for mode in order:
                stages.append({'config_id':config['id'],'pair':pair,'mode':mode})
            stages.append({'config_id':config['id'],'pair':pair,'mode':'export'})
    return stages


def prepare(probe_root=PROBE,root_reviewed=False):
    if not root_reviewed:raise ValueError('root approval required before freezing this revision')
    probe_root=Path(probe_root).resolve(strict=True)
    destination=HERE/'protocol.json'
    if destination.exists() or (HERE/'freeze.json').exists():raise ValueError('collection is already frozen; verify, never overwrite')
    if probe_root!=PROBE.resolve():raise ValueError('this collection revision is scoped to graph_submit_probe')
    if not (probe_root/'build_manifest.json').is_file():
        raise ValueError('compiled graph build_manifest.json is required before collection freeze; no GPU run attempted')
    probe=load(probe_root/'protocol.json');build=load(probe_root/'build_manifest.json')
    import probe_adapter
    probe_adapter.validate_binding(probe_root,probe,build)
    for r in build['files']:verify_ref(r)
    prior=load(PILOT/'protocol.json');inventory=load(PILOT/'tool_inventory.json');signatures=load(PILOT/'tool_signatures.json')
    if signatures.get('critical_nvidia_binary_signatures_valid') is not True:raise ValueError('Nsight signature evidence unavailable')
    for r in inventory['files']:verify_ref(r)
    exe=ref(build['executable']['path'])
    if exe!=build['executable']:raise ValueError('probe executable differs from manifest/readiness')
    gpu_expected=probe['runtime']['gpu_expected']
    if gpu_expected['uuid']!=prior['gpu_identity']['uuid'] or gpu_expected['driver']!=prior['gpu_identity']['driver_version']:raise ValueError('probe GPU/driver differs from proven current tool protocol')
    environment=dict(probe['runtime']['environment'])
    environment.update({name:None for name in probe['runtime']['extra_clear_environment']})
    options=probe['profiler']['options'][1:]
    if '--kill=false' not in options or '--cuda-graph-trace=node' not in options:raise ValueError('safe graph-node profiler options required')
    if probe['profiler']['path']!=prior['nsys']['executable']['path']:raise ValueError('probe Nsight differs from frozen tool identity')
    protocol={'schema':'graph-matrix-collection-protocol/v1','probe_root':str(probe_root),'created_utc':utc(),'probe_protocol_ref':ref(probe_root/'protocol.json'),
        'probe_manifest_ref':ref(probe_root/'build_manifest.json'),
        'configs':probe['configs'],'quality_policy':POLICY,'execution_plan':plan(probe['configs']),
        'executable':exe,'native_bin':build['native_bin'],'nsys':prior['nsys'],'nsys_profile_options':options,
        'gpu_identity':prior['gpu_identity'],'environment_explicit':environment,
        'clock_domain':{'target_sm_clock_mhz':2400,'sm_clock_tolerance_mhz':30,'external_receipt_required':True,'actual_readback_each_stage_required':True,'collector_sets_clocks':False},
        'mode_contract':probe_adapter.mode_contract(),
        'event_free_control_scope':probe_adapter.mode_contract(),
        'root_review_required_before_gpu':True,'prepared_only':True,'native_llm_executed':False,'calibration_eligible':False,
        'sampler_method_change':'Copied high-resolution SM read-window sampler; multi-field NVML no longer sampled in formal; no cosmetic reuse of old telemetry identity',
        'source_runtime_dispatch_proof':'Native DLL observation is empirical; incomplete original build/source closure is retained as conditional transfer, never promoted to source equivalence.'}
    write_new(destination,protocol)
    paths=[*HERE.glob('*.py'),* (HERE/'telemetry').glob('*.py'),HERE/'telemetry/protocol.json',HERE/'README.md',HERE/'copy_provenance.json',HERE/'preparation_status.json',destination,probe_root/'protocol.json',probe_root/'build_manifest.json',
        PILOT/'pilot.py',PILOT/'protocol.json',PILOT/'tool_inventory.json',PILOT/'tool_signatures.json',Path(sys.executable),HERE/'clock_controller.py']
    paths.extend(Path(r['path']) for r in prior['nsys']['information'].values());paths.append(Path(prior['gpu_identity']['nvml_library']['path']))
    critical=[r for r in inventory['files'] if Path(r['path']).name in signatures['critical_names'] or 'cupti' in Path(r['path']).name.lower()]
    freeze={'schema':'graph-matrix-collection-freeze/v1','created_utc':utc(),'files':[ref(x) for x in sorted(set(paths))],
        'probe_files':build['files'],'tool_files':inventory['files'],'critical_tool_files':critical,'protocol_ref':ref(destination),'probe_manifest_ref':ref(probe_root/'build_manifest.json'),
        'tool_inventory_ref':ref(PILOT/'tool_inventory.json'),'tool_signatures_ref':ref(PILOT/'tool_signatures.json'),
        'python':ref(sys.executable),'source_closure_complete':False,'no_measurement_started':True}
    write_new(HERE/'freeze.json',freeze);freeze_ref=ref(HERE/'freeze.json')
    entry=[sys.executable,str(HERE/'runner.py'),'run','--expected-freeze-sha256',freeze_ref['sha256'],'--root-reviewed','--idle-window-confirmed','--clock-control-receipt','<ROOT_RECEIPT_PATH>','--clock-control-sha256','<ROOT_RECEIPT_SHA256>']
    write_new(HERE/'ready.json',{'schema':'graph-collection-readiness/v1','status':'prepared_pending_external_approval','freeze_ref':freeze_ref,
        'configs':6,'process_pairs':18,'profile_processes':18,'direct_processes':18,'export_processes':18,
        'expected_calls_per_process':36,'expected_formal_per_process':30,'gpu_executed':False,
        'execution_entry':entry,'resume_entry':entry,'external_approval_required_for_this_literal_sha256':True})
    check=verify_freeze(HERE/'freeze.json',freeze_ref['sha256'])
    write_new(HERE/'prepare_verification.json',check)
    return {'status':'ready_for_external_approval','freeze':freeze_ref,'verification':check,'gpu_executed':False}


def process_alive(pid):
    if os.name!='nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x100000,False,pid)
    if not handle:return False
    try:
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        return kernel.WaitForSingleObject(handle,0)==258
    finally:kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle)


def stage_path(stage):return HERE/'runs'/stage['config_id']/f"pair_{stage['pair']+1:02d}"/stage['mode']


def app_argv(config,output):
    import probe_adapter
    return probe_adapter.app_argv(config,output)


def commands(stage,protocol):
    import probe_adapter
    return probe_adapter.stage_command(stage,stage_path(stage),protocol)


def wait_checkpoint(proc,seconds=180):
    try:return {'status':'supervisor_exited','returncode':proc.wait(timeout=seconds)}
    except subprocess.TimeoutExpired:return {'status':'still_running','pid':proc.pid,'kill_on_deadline':False,'next_action':'resume only after the same supervisor records complete.json'}


def terminal_identity_stop(freeze,expected,origin,reason):
    path=HERE/'identity-stop.json'
    if not path.exists():
        try:write_new(path,{'schema':'graph-collection-identity-stop/v1','utc':utc(),'freeze_path':str(freeze),
            'external_approved_sha256':expected,'origin':origin,'reason':reason,'terminal_for_this_revision':True,
            'next_action':'Do not resume this revision; preserve evidence and create a new reviewed revision.'})
        except FileExistsError:pass
    return {'status':'identity_terminal_stop','stop_ref':ref(path),'reason':reason}


def checked_identity(freeze,expected,origin,full_tools=True):
    try:return verify_freeze(freeze,expected,full_tools)
    except IdentityError as exc:return terminal_identity_stop(freeze,expected,origin,str(exc))


def run(root_reviewed,idle_confirmed,max_stages=None,expected_freeze_sha256=None,clock_receipt_path=None,clock_receipt_sha256=None):
    if not root_reviewed or not idle_confirmed:raise ValueError('root review and idle window flags required; no launch')
    if max_stages is not None and max_stages<1:raise ValueError('max_stages must be positive')
    freeze=HERE/'freeze.json'
    if (HERE/'identity-stop.json').exists():return {'status':'identity_terminal_stop','stop_ref':ref(HERE/'identity-stop.json')}
    before=checked_identity(freeze,expected_freeze_sha256,'run_or_resume')
    if before.get('status')=='identity_terminal_stop':return before
    protocol=load(HERE/'protocol.json')
    try:clock_binding=verify_clock_receipt(clock_receipt_path,clock_receipt_sha256,protocol['gpu_identity']['uuid'])
    except (IdentityError,ValueError,KeyError,TypeError,OSError) as exc:return terminal_identity_stop(freeze,expected_freeze_sha256,'clock_control_identity',str(exc))
    clock_anchor=HERE/'clock-control-binding.json'
    if clock_anchor.exists() and load(clock_anchor).get('receipt_ref')!=clock_binding['receipt_ref']:return terminal_identity_stop(freeze,expected_freeze_sha256,'clock_control_receipt_changed','Different clock session cannot be mixed into this freeze')
    if not clock_anchor.exists():write_new(clock_anchor,clock_binding)
    for receipt in (HERE/'runs').glob('*/*/*/complete.json'):
        existing=load(receipt)
        if existing.get('status')=='failed_identity' or existing.get('freeze_after_error'):
            return terminal_identity_stop(freeze,expected_freeze_sha256,'retained_stage_identity_failure',str(receipt))
    attempts=HERE/'sessions';attempts.mkdir(exist_ok=True)
    session=attempts/f"session_{len(list(attempts.glob('session_*.json')))+1:04d}.json"
    write_new(session,{'utc':utc(),'pid':os.getpid(),'verification_before':before,'root_reviewed':True,'idle_window_confirmed':True})
    launched=0;completed=[]
    for stage in protocol['execution_plan']:
        directory=stage_path(stage);receipt=directory/'complete.json'
        if receipt.exists():
            state=load(receipt)
            if state.get('status')=='failed_identity' or state.get('freeze_after_error'):return terminal_identity_stop(freeze,expected_freeze_sha256,'existing_stage_identity_failure',str(receipt))
            if state.get('child_process_exited') is False:return {'status':'unresolved_running','stage':stage,'receipt':ref(receipt)}
            if state.get('status')=='completed':
                try:
                    from extract import stage_complete
                    stage_complete(directory,expected_freeze_sha256)
                except (ValueError,KeyError,TypeError,OSError) as exc:return terminal_identity_stop(freeze,expected_freeze_sha256,'completed_stage_origin_failure',str(exc))
            completed.append({'stage':stage,'receipt':ref(receipt),'status':state['status']});continue
        if directory.exists():
            launched_ref=directory/'supervisor.json'
            if not launched_ref.exists():return {'status':'unresolved_incomplete_launch','stage':stage,'next_action':'inspect existing evidence; no restart or next stage'}
            pid=load(launched_ref)['pid']
            return {'status':'still_running' if process_alive(pid) else 'unresolved_missing_exit_receipt','pid':pid,'stage':stage,'next_action':'no retry or next stage until exit evidence is resolved'}
        if max_stages is not None and launched>=max_stages:break
        verified=checked_identity(freeze,expected_freeze_sha256,'before_stage',False)
        if verified.get('status')=='identity_terminal_stop':return verified
        directory.mkdir(parents=True,exist_ok=False)
        if stage['mode']=='export' and not (directory.parent/'profile/trace.nsys-rep').exists():
            write_new(receipt,{'status':'blocked_missing_profile_report','stage':stage,'utc':utc(),'returncode':None});continue
        spec={'schema':'graph-collection-process-spec/v1','stage':stage,'directory':str(directory),'freeze':str(freeze),'expected_freeze_sha256':expected_freeze_sha256,'clock_control_binding':clock_binding,
            'argv':commands(stage,protocol),'cwd':protocol['probe_root'],'environment':protocol['environment_explicit'],
            'path_prefix':protocol['native_bin']+os.pathsep+r'E:\cuda\bin','telemetry':stage['mode']!='export','gpu_identity':protocol['gpu_identity'],
             'input_artifacts':[ref(directory.parent/'profile/trace.nsys-rep')] if stage['mode']=='export' else []}
        write_new(directory/'spec.json',spec)
        out=(directory/'supervisor.stdout.txt').open('xb');err=(directory/'supervisor.stderr.txt').open('xb')
        try:proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),str(directory/'spec.json')],cwd=HERE,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        finally:out.close();err.close()
        write_new(directory/'supervisor.json',{'pid':proc.pid,'utc':utc(),'session':ref(session)})
        result=wait_checkpoint(proc,POLICY['checkpoint_seconds']);launched+=1
        if result['status']=='still_running':
            write_new(directory/'checkpoint.json',{**result,'utc':utc(),'freeze_verification':checked_identity(freeze,expected_freeze_sha256,'checkpoint',False)})
            return {**result,'stage':stage}
        if not receipt.exists():return {'status':'unresolved_missing_exit_receipt','stage':stage,'supervisor_returncode':result['returncode']}
        state=load(receipt)
        if state.get('status')=='failed_identity' or state.get('freeze_after_error'):return terminal_identity_stop(freeze,expected_freeze_sha256,'new_stage_identity_failure',str(receipt))
        if state.get('child_process_exited') is False:return {'status':'unresolved_running','stage':stage,'receipt':ref(receipt)}
        if state.get('status')=='completed':
            try:
                from extract import stage_complete
                stage_complete(directory,expected_freeze_sha256)
            except (ValueError,KeyError,TypeError,OSError) as exc:return terminal_identity_stop(freeze,expected_freeze_sha256,'new_stage_origin_failure',str(exc))
        completed.append({'stage':stage,'receipt':ref(receipt),'status':state['status']})
    after=checked_identity(freeze,expected_freeze_sha256,'session_end')
    if after.get('status')=='identity_terminal_stop':return after
    all_receipts=all((stage_path(s)/'complete.json').exists() for s in protocol['execution_plan'])
    all_success=all_receipts and all(load(stage_path(s)/'complete.json').get('status')=='completed' for s in protocol['execution_plan'])
    result={'status':'matrix_success' if all_success else 'matrix_receipts_complete_with_failures' if all_receipts else 'bounded_stages_complete',
        'launched_this_session':launched,'completed_stage_count':len(completed),'planned_stage_count':len(protocol['execution_plan']),
        'verification_after':after,'all_stage_receipts_present':all_receipts,'all_stages_successful':all_success,'stages':completed}
    write_new(session.with_name(session.stem+'.finish.json'),result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','verify','status','run']);p.add_argument('--root-reviewed',action='store_true');p.add_argument('--idle-window-confirmed',action='store_true');p.add_argument('--max-stages',type=int);p.add_argument('--expected-freeze-sha256');p.add_argument('--probe-root',type=Path,default=PROBE);p.add_argument('--clock-control-receipt',type=Path);p.add_argument('--clock-control-sha256')
    a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.probe_root,a.root_reviewed)
    elif a.mode=='verify':result=verify_freeze(HERE/'freeze.json',a.expected_freeze_sha256)
    elif a.mode=='status':result={'ready':(HERE/'ready.json').exists(),'started_stages':len(list((HERE/'runs').glob('*/*/*/spec.json'))),'completed_stages':len(list((HERE/'runs').glob('*/*/*/complete.json')))}
    else:result=run(a.root_reviewed,a.idle_window_confirmed,a.max_stages,a.expected_freeze_sha256,a.clock_control_receipt,a.clock_control_sha256)
    import json;print(json.dumps(result,ensure_ascii=False));return 4 if result.get('status')=='identity_terminal_stop' else 0

if __name__=='__main__':raise SystemExit(main())
