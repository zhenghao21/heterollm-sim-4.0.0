"""Durable child supervisor: no completion receipt while a launched child is unresolved."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import threading
import time
from common import load,write_new,utc,ref,verify_freeze,module,PILOT,IdentityError,POLICY,verify_clock_receipt,clock_readback_gate


def wait_for_real_exit(proc,errors=None):
    """Once Popen succeeded, every exception path waits for that same child; never kills."""
    errors=errors if errors is not None else []
    while True:
        try:
            code=proc.wait()
            if type(code) is not int:
                errors.append('wait returned no exit status');time.sleep(.05);continue
            return code
        except BaseException as exc:
            errors.append(type(exc).__name__+': '+str(exc))
            # A bookkeeping/wait exception is not permission to release process ownership.
            time.sleep(.05)


def supervise_child(spec,directory,receipt,support,popen=subprocess.Popen):
    proc=None;launch_error=None
    with (directory/'stdout.txt').open('xb') as out,(directory/'stderr.txt').open('xb') as err:
        try:
            proc=popen(spec['argv'],cwd=spec['cwd'],env=spec['_environment'],stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            # Save identity immediately before any operation that can throw.
            receipt['process_pid']=proc.pid;receipt['child_process_exited']=False
            write_new(directory/'launched.json',{'utc':utc(),'pid':proc.pid,'supervisor_pid':os.getpid(),'qpc_after_launch':support.qpc(),'argv':spec['argv']})
        except BaseException as exc:
            launch_error=type(exc).__name__+': '+str(exc);receipt['launch_bookkeeping_error']=launch_error
        finally:
            if proc is not None:
                receipt['returncode']=wait_for_real_exit(proc,receipt.setdefault('wait_observation_errors',[]))
                receipt['child_process_exited']=True
            else:
                receipt['child_process_exited']=True;receipt['child_was_started']=False
    receipt['status']='completed' if proc is not None and launch_error is None and receipt['returncode']==0 else 'failed'
    return receipt


def run(spec_path):
    from telemetry.sampler import Win32DeadlineTimer,NVMLSMReader,SamplerSession
    spec=load(spec_path);directory=Path(spec['directory']);freeze=Path(spec['freeze']);expected=spec['expected_freeze_sha256']
    session=None;timer=None;telemetry=None;identity_holder={}
    receipt={'schema':'graph-collection-process/v1','utc_started':utc(),'argv':spec['argv'],'cwd':spec['cwd'],
        'spec_ref':ref(spec_path),'supervisor_pid':os.getpid(),'external_approved_sha256':expected,
        'child_process_exited':True,'telemetry_errors':[],'sampling_method':'SM_read_windows_high_resolution_waitable_timer_absolute_QPC'}
    try:
        before=verify_freeze(freeze,expected,False)
        clock_ref=spec['clock_control_binding']['receipt_ref']
        receipt['clock_control_binding']=verify_clock_receipt(clock_ref['path'],clock_ref['sha256'],spec['gpu_identity']['uuid'])
        write_new(directory/'supervisor-started.json',{'utc':utc(),'pid':os.getpid(),'spec':ref(spec_path),'freeze_before':before})
        support=module(PILOT/'pilot.py','frozen_graph_QPC_support')
        receipt['qpc_frequency']=support.qpf()
        if spec['telemetry']:
            timer=Win32DeadlineTimer()
            if timer.frequency!=receipt['qpc_frequency']:raise IdentityError('sampler QPC frequency mismatch')
            def reader_factory():
                reader=NVMLSMReader()
                try:
                    identity=reader.identity;identity_holder.update(identity)
                    if any(identity[k]!=spec['gpu_identity'][k] for k in ('uuid','driver_version')):
                        raise IdentityError('observed GPU/driver differs from frozen protocol')
                    lo=timer.now();value=reader.read_sm();hi=timer.now()
                    identity_holder['preflight_SM']={'sm_read_begin_qpc':lo,'sm_read_end_qpc':hi,'sm_mhz':value}
                    if value['status']!=0 or type(value.get('value')) is not int or abs(value['value']-2400)>30:
                        raise IdentityError('actual prelaunch SM clock outside2400+/-30')
                    return reader
                except BaseException:
                    reader.close();raise
            session=SamplerSession(timer=timer,reader_factory=reader_factory,duration_ns=24*3600*1000000000,period_ns=5000000).start()
            if not session.ready.wait(30) or session.errors or session.done.is_set():
                raise IdentityError('high-resolution sampler not ready: '+repr(session.errors))
            receipt['gpu_identity']=dict(identity_holder)
        environment=os.environ.copy()
        for key,value in spec['environment'].items():
            environment.pop(key,None)
            if value is not None:environment[key]=value
        environment['PATH']=spec['path_prefix']+os.pathsep+environment.get('PATH','');spec['_environment']=environment
        receipt['qpc_launch_start']=support.qpc()
        supervise_child(spec,directory,receipt,support)
        receipt['qpc_process_complete']=support.qpc()
    except IdentityError as exc:receipt.update(status='failed_identity',identity_error=str(exc))
    except BaseException as exc:receipt.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        # supervise_child does not return until its native child has really exited.
        # A sampler join timeout retains ownership; never close NVML under a live reader.
        if session:
            session.stop()
            while session.thread.is_alive():
                try:session.join(1)
                except TimeoutError:continue
            telemetry=session.result
            try:session.close()
            except BaseException as exc:receipt['telemetry_errors'].append(str(exc))
            if telemetry is not None:
                telemetry['lifecycle']=session.lifecycle()
                write_new(directory/'telemetry.json',telemetry)
            if session.errors:receipt['telemetry_errors'].extend(session.errors)
        elif timer:
            timer.close()
        receipt['utc_finished']=utc()
        if spec['telemetry'] and (directory/'microbench.json').exists():
            try:
                if telemetry is None:raise IdentityError('high-resolution raw telemetry missing')
                import probe_adapter
                receipt['clock_readback_gate']=clock_readback_gate(probe_adapter.read_raw(directory/'microbench.json'),telemetry)
                if not receipt['clock_readback_gate']['passed']:receipt.update(status='failed_identity',identity_error='SM read-window clock gate failed')
            except (ValueError,KeyError,TypeError,OSError) as exc:receipt.update(status='failed_identity',identity_error='clock readback invalid: '+str(exc))
        try:receipt['freeze_after']=verify_freeze(freeze,expected,False)
        except Exception as exc:receipt.update(status='failed_identity',freeze_after_error=str(exc))
        if receipt.get('child_process_exited') is not True:raise RuntimeError('refusing completion receipt without confirmed child exit')
        if session and session.thread.is_alive():raise RuntimeError('refusing completion receipt with active sampler')
        receipt['artifacts']=[ref(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name!='complete.json']
        write_new(directory/'complete.json',receipt)
    return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('spec',type=Path);args=ap.parse_args();raise SystemExit(run(args.spec))
