"""Keep an already started child alive until natural exit, even if observation is interrupted."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time


def write_note(path,value):
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    data=path.read_bytes()
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def wait_naturally(process,deadline_qpc_ns,directory,clock=time.perf_counter_ns):
    directory=Path(directory);overdue=False;notified=False;interruptions=[];note_errors=[]
    def note_deadline(observed):
        nonlocal overdue,notified
        if deadline_qpc_ns is None or observed<deadline_qpc_ns:return
        overdue=True
        if notified:return
        notified=True
        try:
            write_note(directory/'soft_deadline_exceeded.json',{
                'schema':'r28-natural-wait-overdue/v1','created_utc':datetime.now(timezone.utc).isoformat(),
                'observed_qpc_ns':observed,'deadline_qpc_ns':deadline_qpc_ns,
                'started_process_must_exit_naturally':True,'termination_requested':False,'evidence_eligible':False})
        except BaseException as error:
            note_errors.append(type(error).__name__+': '+str(error))
    while True:
        try:
            note_deadline(clock())
            rc=process.wait(timeout=1.0)
            ended=clock();note_deadline(ended)
            interrupted=bool(interruptions) or bool(note_errors)
            return {'returncode':rc,'natural_exit':True,'ended_qpc_ns':ended,'deadline_qpc_ns':deadline_qpc_ns,
                'soft_deadline_exceeded':overdue,'termination_requested':False,'observation_interrupted':interrupted,
                'observation_interruptions':interruptions,'observation_note_errors':note_errors,
                'stop_new_work':interrupted or overdue,'evidence_eligible':not (interrupted or overdue)}
        except subprocess.TimeoutExpired:
            continue
        except BaseException as error:
            value={'schema':'r31-observation-interrupted-natural-wait/v1',
                'created_utc':datetime.now(timezone.utc).isoformat(),'exception':type(error).__name__,
                'message':str(error),'pid':getattr(process,'pid',None),'stop_new_work':True,
                'started_process_must_exit_naturally':True,'termination_requested':False,'evidence_eligible':False}
            interruptions.append(value)
            try:
                value['record_ref']=write_note(directory/('observation_interrupted.%04d.json'%len(interruptions)),value)
            except BaseException as note_error:
                note_errors.append(type(note_error).__name__+': '+str(note_error))
            # A lost observation is not evidence that the child exited. Keep waiting.
