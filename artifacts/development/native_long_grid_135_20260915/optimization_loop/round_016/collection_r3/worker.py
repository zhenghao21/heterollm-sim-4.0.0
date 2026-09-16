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
    spec=load(spec_path);directory=Path(spec['directory']);freeze=Path(spec['freeze']);expected=spec['expected_freeze_sha256']
    gpu=None;thread=None;stop=threading.Event();samples=[]
    receipt={'schema':'operator-collection-process/v2','utc_started':utc(),'argv':spec['argv'],'cwd':spec['cwd'],
        'spec_ref':ref(spec_path),'supervisor_pid':os.getpid(),'external_approved_sha256':expected,'child_process_exited':True,'telemetry_errors':[]}
    try:
        before=verify_freeze(freeze,expected,False)
        clock_ref=spec['clock_control_binding']['receipt_ref']
        receipt['clock_control_binding']=verify_clock_receipt(clock_ref['path'],clock_ref['sha256'],spec['gpu_identity']['uuid'])
        write_new(directory/'supervisor-started.json',{'utc':utc(),'pid':os.getpid(),'spec':ref(spec_path),'freeze_before':before})
        support=module(PILOT/'pilot.py','frozen_r15_telemetry_support')
        receipt.update(qpc_launch_start=support.qpc(),qpc_frequency=support.qpf())
        if spec['telemetry']:
            gpu=support.GpuTelemetry();identity=gpu.identity();receipt['gpu_identity']=identity
            first_sample=gpu.sample();samples.append(first_sample)
            reading=first_sample.get('sm_mhz',{})
            if reading.get('status')!=0 or type(reading.get('value')) is not int or abs(reading['value']-2400)>30:raise IdentityError('pre-launch actual SM clock outside 2400 +/-30 MHz')
            if any(identity[key]!=spec['gpu_identity'][key] for key in ('uuid','driver_version')):raise IdentityError('GPU/driver identity changed')
            def collect():
                while not stop.is_set():
                    try:samples.append(gpu.sample())
                    except Exception as exc:samples.append({'utc':utc(),'error':str(exc)})
                    stop.wait(POLICY['telemetry_period_seconds'])
            thread=threading.Thread(target=collect,daemon=True);thread.start()
        environment=os.environ.copy()
        for key,value in spec['environment'].items():
            environment.pop(key,None)
            if value is not None:environment[key]=value
        environment['PATH']=spec['path_prefix']+os.pathsep+environment.get('PATH','');spec['_environment']=environment
        supervise_child(spec,directory,receipt,support)
        receipt['qpc_process_complete']=support.qpc()
    except IdentityError as exc:receipt.update(status='failed_identity',identity_error=str(exc))
    except BaseException as exc:receipt.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if thread:thread.join(timeout=2)
        if gpu:
            try:samples.append(gpu.sample())
            except Exception as exc:receipt['telemetry_errors'].append(str(exc))
            try:gpu.close()
            except Exception as exc:receipt['telemetry_errors'].append(str(exc))
        write_new(directory/'telemetry.json',{'samples':samples,'clock_locked':None,'clock_locked_assertion':'No lock inferred; root receipt and actual sampled readbacks are separate evidence','period_seconds':POLICY['telemetry_period_seconds'],'target_sm_clock_mhz':2400,'sm_clock_tolerance_mhz':30,'scope':'whole process with absolute QPC, formal intervals independently audited'})
        receipt['utc_finished']=utc()
        if spec['telemetry'] and (directory/'microbench.json').exists():
            try:
                receipt['clock_readback_gate']=clock_readback_gate(load(directory/'microbench.json'),samples)
                if not receipt['clock_readback_gate']['passed']:receipt.update(status='failed_identity',identity_error='formal SM-clock domain readback gate failed')
            except (ValueError,KeyError,TypeError,OSError) as exc:receipt.update(status='failed_identity',identity_error='clock readback evidence invalid: '+str(exc))
        try:receipt['freeze_after']=verify_freeze(freeze,expected,False)
        except Exception as exc:receipt.update(status='failed_identity',freeze_after_error=str(exc))
        if receipt.get('child_process_exited') is not True:raise RuntimeError('refusing completion receipt without confirmed child exit')
        receipt['artifacts']=[ref(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name!='complete.json']
        write_new(directory/'complete.json',receipt)
    return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('spec',type=Path);args=ap.parse_args();raise SystemExit(run(args.spec))
