"""Explicit2s read-only NVML diagnostic; never starts inference or changes GPU clocks."""
from pathlib import Path
import argparse,datetime,json,sys
from sampler import Win32DeadlineTimer,NVMLSMReader,SamplerSession,file_ref
from assess import summarize
HERE=Path(__file__).resolve().parent

def write_new(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def run(output_root,seconds=2):
    if seconds!=2:raise ValueError('initial real diagnostic is frozen at2seconds')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(output_root).resolve()/stamp;out.mkdir(parents=True,exist_ok=False)
    before=[file_ref(HERE/name) for name in ('sampler.py','assess.py','run_diagnostic.py','test_sampler.py','protocol.json')]
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8-sig'))
    write_new(out/'manifest.json',{'schema':'telemetry-v2-host-diagnostic/v1','utc_started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sources':before,'protocol':protocol,'seconds':seconds,'GPU_clock_writes':False,'inference_launched':False})
    session=None;result=None;failure=None
    try:
        timer=Win32DeadlineTimer();session=SamplerSession(timer=timer,reader_factory=NVMLSMReader,duration_ns=2_000_000_000).start()
        if not session.ready.wait(10):raise TimeoutError('sampler not initialized after10seconds')
        result=session.join(10)
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        if session is not None and session.thread.is_alive():
            session.stop()
            # Keep ownership and do not close NVML/handles underneath a blocked read.
            try:session.join(10)
            except TimeoutError:failure+='; stop pending, worker ownership retained' 
    finally:
        if session is not None and not session.thread.is_alive():
            try:session.close()
            except BaseException as error:failure=(failure+'; ' if failure else '')+str(error)
    after=[file_ref(r['path']) for r in before]
    if after!=before:failure='source identity changed during diagnostic'
    if result is not None:write_new(out/'raw.json',result)
    assessment=summarize(result) if result else {'status':'failed_before_samples'}
    assessment.update(failure=failure,lifecycle=session.lifecycle() if session else None,sources_unchanged=after==before,
        utc_finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_new(out/'summary.json',assessment)
    print(json.dumps({'output':str(out),**{k:assessment[k] for k in ('status','captured_samples','deadline_slots_missed','sample_start_intervals_ms','SM_read_cost_ms','observed_SM_clock_MHz') if k in assessment}},ensure_ascii=False))
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--real-nvml',action='store_true');p.add_argument('--seconds',type=int,default=2);p.add_argument('--output-root',type=Path,default=HERE/'runs');a=p.parse_args()
    if not a.real_nvml:p.error('Explicit --real-nvml is required; otherwise run hostmock tests only')
    run(a.output_root,a.seconds)
