"""Root-launched single clock session; checkpoint is never matrix completion.

Preparation/import has no device side effects. Only explicit --run with the
root-approved new collection SHA may lock clocks or launch the fixed matrix.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P=Path(__file__).resolve().parent
C=P
PY=Path(r'E:\anaconda\python.exe')
SMI=Path(r'C:\Windows\System32\nvidia-smi.exe')
PREFIX='graph_clock_control_' 
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)
TERMINAL={'matrix_success','matrix_receipts_complete_with_failures'}


def utc():return datetime.now(timezone.utc).isoformat()

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def ref(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path),'sha256':digest,'bytes':path.stat().st_size}


def capture(argv,name):
    out=P/(PREFIX+name+'.stdout.txt');err=P/(PREFIX+name+'.stderr.txt')
    with out.open('xb') as stdout,err.open('xb') as stderr:
        result=subprocess.run(list(map(str,argv)),stdout=stdout,stderr=stderr,creationflags=FLAGS)
    return {'command':list(map(str,argv)),'returncode':result.returncode,'stdout_ref':ref(out),'stderr_ref':ref(err)}


def wait_pid(pid):
    """Wait without terminating. Inability to verify exit is not evidence of exit."""
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x100000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error==87:return  # ERROR_INVALID_PARAMETER: PID no longer exists.
        raise OSError(error,'cannot verify owned process exit: '+str(pid))
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    try:
        while True:
            value=kernel.WaitForSingleObject(handle,1000)
            if value==0:return
            if value!=258:raise OSError(ctypes.get_last_error(),'owned process wait failed')
    finally:kernel.CloseHandle(handle)


def stage_directory(stage):
    return C/'runs'/stage['config_id']/f"pair_{stage['pair']+1:02d}"/stage['mode']


def verify_completed_stage(directory,approved_sha,clock_ref):
    receipt=load(directory/'complete.json')
    if receipt.get('status')=='failed_identity' or receipt.get('freeze_after_error') or receipt.get('identity_error'):
        raise RuntimeError('terminal stage identity failure: '+str(directory))
    if receipt.get('external_approved_sha256')!=approved_sha:raise RuntimeError('stage approved SHA mismatch')
    if receipt.get('child_process_exited') is not True:raise RuntimeError('stage child exit remains unresolved')
    after=receipt.get('freeze_after',{})
    if after.get('passed') is not True or after.get('external_approved_sha256')!=approved_sha:raise RuntimeError('stage final identity verification missing')
    if receipt.get('clock_control_binding',{}).get('receipt_ref')!=clock_ref:raise RuntimeError('stage clock session mismatch')
    if receipt.get('status') not in ('completed','failed'):raise RuntimeError('unsupported stage receipt state')
    return receipt


def wait_checkpoint(result,approved_sha,clock_ref):
    if result.get('status')!='still_running' or not isinstance(result.get('stage'),dict):raise RuntimeError('checkpoint lacks fixed stage')
    directory=stage_directory(result['stage']);record=load(directory/'supervisor.json')
    if result.get('pid')!=record.get('pid'):raise RuntimeError('checkpoint supervisor identity mismatch')
    # Same exact supervisor must finish before any new coordinator is launched.
    wait_pid(record['pid'])
    if not (directory/'complete.json').exists():
        # A crashed supervisor may still have a target child. Wait it too, then fail closed.
        launched=directory/'launched.json'
        if launched.exists():wait_pid(load(launched)['pid'])
        raise RuntimeError('supervisor exited without durable completion receipt')
    return verify_completed_stage(directory,approved_sha,clock_ref)


def wait_all_owned():
    """Finally-path safety. Wait recorded supervisors and primary children, including completed wrappers."""
    for directory in sorted((C/'runs').glob('*/*/*')):
        if not directory.is_dir():continue
        supervisor=directory/'supervisor.json';launched=directory/'launched.json'
        if supervisor.exists():wait_pid(load(supervisor)['pid'])
        if launched.exists():wait_pid(load(launched)['pid'])
        # No unrecorded child is presumed absent if a supervisor launch is incomplete.
        if (directory/'spec.json').exists() and not supervisor.exists():raise RuntimeError('unresolved launch bookkeeping; root must confirm owned process exit')
        if supervisor.exists() and not (directory/'complete.json').exists():raise RuntimeError('owned stage has no completion receipt after exit; root inspection required')


def drive_matrix(invoke,wait_same,count_first,verify_identity,*,first_pair_only=False):
    """Pure control-state machine; injectable functions allow no-subprocess regression tests."""
    while True:
        verify_identity()
        remaining=3-count_first() if first_pair_only else None
        if first_pair_only and remaining<=0:return {'status':'first_pair_complete'}
        outcome=invoke(remaining)
        status=outcome.get('status')
        if status=='identity_terminal_stop':raise RuntimeError('terminal identity stop from coordinator')
        if status=='still_running':
            completion=wait_same(outcome)
            if completion.get('status')=='failed_identity' or completion.get('freeze_after_error'):
                raise RuntimeError('terminal identity failure after checkpoint')
            if completion.get('child_process_exited') is not True:raise RuntimeError('checkpoint child exit not confirmed')
            verify_identity()
            continue
        if status in TERMINAL:
            if first_pair_only:raise RuntimeError('matrix finished unexpectedly during first-pair gate')
            verify_identity();return outcome
        if first_pair_only and status=='bounded_stages_complete':
            if count_first()<3:raise RuntimeError('bounded first pair did not produce three stage receipts')
            verify_identity();return {'status':'first_pair_complete'}
        raise RuntimeError('coordinator unresolved or unsupported status: '+str(status))


def load_collector():
    # Read collector helpers without writing into the frozen source tree.
    namespace={'__file__':str(C/'common.py'),'__name__':'controller_collection_common'}
    exec(compile((C/'common.py').read_bytes(),str(C/'common.py'),'exec'),namespace)
    return namespace


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true');ap.add_argument('--root-reviewed',action='store_true');ap.add_argument('--idle-window-confirmed',action='store_true')
    ap.add_argument('--expected-freeze-sha256',required=True);ap.add_argument('--review-seconds',type=int,default=600)
    args=ap.parse_args(argv)
    if not args.run:
        print(json.dumps({'status':'prepared_not_executed','collection':str(C),'expected_freeze_sha256':args.expected_freeze_sha256,'clock_session_start':str(P/(PREFIX+'start.json'))}));return 0
    if not args.root_reviewed or not args.idle_window_confirmed:raise RuntimeError('explicit root review and idle window required')
    if args.review_seconds<1:raise ValueError('review timeout must be positive')
    if len(args.expected_freeze_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.expected_freeze_sha256):raise ValueError('root-approved literal freeze SHA required')
    if any(P.glob(PREFIX+'*')):raise RuntimeError('graph-clock-control output prefix already exists; never overwrite or start another session')
    approved=args.expected_freeze_sha256
    locked=False;launched=False;result={'schema':'operator-clock-lifecycle/v2','started_utc':utc(),'pid':os.getpid(),'controller_ref':ref(Path(__file__)),
        'collection':str(C),'approved_freeze_sha256':approved,'automatic_resume_same_clock_session':True,'coordinator_calls':[]}
    clock_ref=None;support=None
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('elevated token required')
        support=load_collector();support['verify_freeze'](C/'freeze.json',approved)
        protocol=load(C/'protocol.json')
        if protocol['quality_policy']['config_count']!=6 or len(protocol['execution_plan'])!=54:raise RuntimeError('unapproved matrix scope/order')
        if (C/'runs').exists() and any((C/'runs').iterdir()):raise RuntimeError('new controller requires fresh graph collection; old session cannot be reused')
        capture([SMI,'--query-gpu=uuid,pstate,clocks.sm,clocks.mem','--format=csv,noheader'],'before')
        lock=capture([SMI,'-lgc','2400,2400'],'lock')
        if lock['returncode']!=0:raise RuntimeError('clock lock command failed')
        locked=True;time.sleep(1)
        check=capture([SMI,'--query-gpu=uuid,clocks.sm','--format=csv,noheader,nounits'],'readback')
        fields=Path(check['stdout_ref']['path']).read_text(encoding='utf-8-sig').strip().split(',')
        if check['returncode']!=0 or len(fields)!=2 or fields[0].strip()!=protocol['gpu_identity']['uuid'] or abs(int(fields[1].strip())-2400)>30:
            raise RuntimeError('actual GPU/readback does not match fixed domain')
        start=P/(PREFIX+'start.json')
        write(start,{'schema':'operator-clock-control-receipt/v1','created_utc':utc(),'gpu_uuid':fields[0].strip(),'target_sm_clock_mhz':2400,
            'sm_clock_tolerance_mhz':30,'requested_lock_min_mhz':2400,'requested_lock_max_mhz':2400,'lock_command_returncode':0,
            'command':lock['command'],'stdout_ref':lock['stdout_ref'],'stderr_ref':lock['stderr_ref'],'restore_on_exit_planned':True,
            'readback':check,'controller_ref':ref(Path(__file__))})
        clock_ref=ref(start)
        command=[str(PY),str(C/'runner.py'),'run','--root-reviewed','--idle-window-confirmed','--expected-freeze-sha256',approved,
            '--clock-control-receipt',str(start),'--clock-control-sha256',clock_ref['sha256']]
        def verify():
            if (C/'identity-stop.json').exists():raise RuntimeError('sticky collection identity stop')
            support['verify_freeze'](C/'freeze.json',approved)
            if ref(start)!=clock_ref:raise RuntimeError('same-session clock receipt changed')
        def count_first():
            count=0
            for stage in protocol['execution_plan'][:3]:
                directory=stage_directory(stage)
                if (directory/'complete.json').exists():
                    r=verify_completed_stage(directory,approved,clock_ref)
                    if r['status']!='completed':raise RuntimeError('first pair process failed; root must review retained evidence')
                    count+=1
            return count
        def invoke(limit):
            nonlocal launched
            argv=command+(['--max-stages',str(limit)] if limit is not None else [])
            launched=True
            item=capture(argv,'coordinator_'+f"{len(result['coordinator_calls'])+1:04d}")
            result['coordinator_calls'].append(item)
            text=Path(item['stdout_ref']['path']).read_text(encoding='utf-8-sig').strip()
            response=json.loads(text)
            write(P/(PREFIX+f"coordinator_result_{len(result['coordinator_calls']):04d}.json"),{'capture':item,'response':response})
            if item['returncode']!=0 and response.get('status')!='identity_terminal_stop':raise RuntimeError('coordinator exited nonzero')
            return response
        wait_same=lambda response:wait_checkpoint(response,approved,clock_ref)
        drive_matrix(invoke,wait_same,count_first,verify,first_pair_only=True)
        first_refs=[ref(stage_directory(stage)/'complete.json') for stage in protocol['execution_plan'][:3]]
        consent_path=P/(PREFIX+'continue.json');stop_path=P/(PREFIX+'stop.json')
        write(P/(PREFIX+'first_pair_ready.json'),{'utc':utc(),'approved_freeze_sha256':approved,'clock_receipt_ref':clock_ref,'receipts':first_refs,
            'controller_pid':os.getpid(),'continuation_file':str(consent_path),'all_quality_gates_unchanged':True})
        deadline=time.monotonic()+args.review_seconds
        while not consent_path.exists():
            if stop_path.exists():raise RuntimeError('root stopped after first finished pair')
            if time.monotonic()>deadline:raise RuntimeError('first-pair review window expired; no children killed')
            time.sleep(1)
        consent=load(consent_path)
        if consent.get('approved_freeze_sha256')!=approved or consent.get('first_pair_reviewed') is not True or consent.get('clock_receipt_sha256')!=clock_ref['sha256']:
            raise RuntimeError('first-pair continuation approval identity missing')
        result['matrix_result']=drive_matrix(invoke,wait_same,count_first,verify)
        # Never accept a matrix status string without all matching durable stage receipts.
        receipts=[load(stage_directory(stage)/'complete.json') for stage in protocol['execution_plan']]
        if len(receipts)!=54:raise RuntimeError('matrix final denominator mismatch')
        for stage in protocol['execution_plan']:
            r=load(stage_directory(stage)/'complete.json')
            if r.get('status')=='blocked_missing_profile_report':continue
            verify_completed_stage(stage_directory(stage),approved,clock_ref)
        result['status']=result['matrix_result']['status']
        result['complete_stage_receipts']=len(receipts)
    except BaseException as exc:
        result.update(status='failed_or_checkpoint_unresolved',error=type(exc).__name__+': '+str(exc))
    finally:
        # A reset cannot race an owned supervisor/child, even after a bookkeeping error.
        while launched:
            try:wait_all_owned();break
            except BaseException as exc:
                result['waiting_owned_exit_error']=type(exc).__name__+': '+str(exc)
                # No kill, no new GPU stage, no false final receipt. Root may inspect this checkpoint.
                checkpoint=P/(PREFIX+'owned_exit_unresolved.json')
                if not checkpoint.exists():write(checkpoint,{'utc':utc(),'error':result['waiting_owned_exit_error'],'controller_pid':os.getpid(),'reset_deferred':True})
                time.sleep(1)
        if locked:result['clock_reset']=capture([SMI,'-rgc'],'reset')
        result['finished_utc']=utc();write(P/(PREFIX+'finish.json'),result)
    print(json.dumps({'status':result['status'],'coordinator_calls':len(result['coordinator_calls']),'clock_reset_returncode':result.get('clock_reset',{}).get('returncode')},ensure_ascii=False))
    return 0 if result['status'] in TERMINAL and result.get('clock_reset',{}).get('returncode')==0 else 4

if __name__=='__main__':raise SystemExit(main())
