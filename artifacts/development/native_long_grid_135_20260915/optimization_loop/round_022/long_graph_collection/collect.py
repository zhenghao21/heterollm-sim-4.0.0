"""Finite R22 long-graph buffered collector; root-frozen, serial, no timed termination.
The only executed graphs are independent synthetic SCALE chains. No cost fitting.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,statistics,subprocess,sys,time
from strict_raw import audit,load,require
from native_trace import ref,analyze_trace
from telemetry.sampler import Win32DeadlineTimer,NVMLSMReader,SamplerSession
from telemetry.assess import formal_clock_gate
HERE=Path(__file__).resolve().parent
PROBE=HERE.parent/'long_graph_probe'
from contract import CONFIGS, domain

def utc():return datetime.now(timezone.utc).isoformat()
def write_new(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(doc,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def verify(r):
    a=ref(r['path']);require(a==r,'frozen reference changed: '+r['path']);return a

def plan(protocol,manifest):
    domain(protocol,manifest);result=[]
    for pair in range(1,4):
        configs=CONFIGS if pair%2 else list(reversed(CONFIGS))
        for config in configs:
            condition=config['id'];ix=CONFIGS.index(config)
            modes=['direct','profile'] if (pair+ix)%2 else ['profile','direct']
            for mode in modes+['export']:
                d=HERE/'runs'/condition/f'pair_{pair:02d}'/mode;raw=d/'raw.jsonl';pair_id=f'r22_long_graph.{condition}.pair{pair}'
                app=[manifest['executable']['path'],'--run','--config',condition,'--arm','buffered','--pair-id',pair_id,'--output',str(raw)]
                argv=app if mode=='direct' else ([manifest['profiler']['path'],*protocol['profiler']['profile_options'],'--output='+str(d/'trace'),*app] if mode=='profile' else [manifest['profiler']['path'],'export','--type=sqlite','--force-overwrite=false','--output='+str(d/'trace.sqlite'),str(d.parent/'profile/trace.nsys-rep')])
                result.append({'ordinal':len(result),'condition':condition,'arm':'buffered','pair':pair,'pair_id':pair_id,'mode':mode,'config':config,'directory':str(d),'raw':str(raw),'app_argv':app,'argv':argv})
    return result
def prepare():
    manifest=load(PROBE/'build_manifest.json');protocol=load(PROBE/'protocol.json')
    require(manifest.get('status')=='compiled_host_tested_not_gpu_executed' and manifest.get('gpu_access') is False,'clean compiled manifest required')
    domain(protocol,manifest)
    require(ref(PROBE/'protocol.json')==manifest['protocol'],'manifest protocol reference mismatch')
    for r in manifest['files']+[manifest['executable'],manifest['profiler']]:verify(r)
    proof_path=HERE.parent.parent/'round_020'/'graph_gap_runtime_identity.json'
    runtime_proof=load(proof_path)
    extra=runtime_proof['extra_runtime_refs']
    require(len(extra)==5,'complete external CUDA/driver runtime closure required')
    for r in extra:verify(r)
    require(ref(runtime_proof['previous_actual_raw']['path'])['sha256']==runtime_proof['previous_actual_raw']['sha256'],'historical actual runtime proof changed')
    paths=[proof_path,Path(__file__),HERE/'strict_raw.py',HERE/'native_trace.py',HERE/'quality_fixed.py',HERE/'contract.py',PROBE/'build_manifest.json',PROBE/'protocol.json',PROBE/'identity_check.py',PROBE/'identity_only_result.json']
    paths += list((HERE/'telemetry').glob('*.py'))
    smi=Path(shutil.which('nvidia-smi') or '')
    require(smi.is_file(),'nvidia-smi executable required')
    record={'schema':'r22-long-graph-collection-freeze/v1','created_utc':utc(),'manifest':ref(PROBE/'build_manifest.json'),'protocol':ref(PROBE/'protocol.json'),'collector_inputs':[ref(p) for p in paths], 'native_and_tool_refs':manifest['files']+extra,'extra_runtime_refs':extra,'runtime_proof':ref(proof_path),'smi':ref(smi),'plan':plan(protocol,manifest),'expected_native_processes':12,'expected_export_processes':6,'coefficients_allowed':False,'resume_policy':'unresolved or failed series is retained, no success-seeking retries','clock':{'target_mhz':2400,'tolerance_mhz':30,'period_ms':5,'max_bracket_ms':25},'timing_policy':{'p90_p10_max':1.5,'process_median_deviation_max':0.05,'profile_direct_wall_relative_max':0.20},'review_checkpoint_nonblocking':True,'stop_policy':'explicit matching stop.json only at natural stage boundaries'}
    write_new(HERE/'execution_freeze.json',record);print(json.dumps({'freeze':ref(HERE/'execution_freeze.json'),'native_processes':12,'exports':6}))

def verified_freeze(expected):
    require(type(expected) is str and len(expected)==64 and ref(HERE/'execution_freeze.json')['sha256']==expected,'approved freeze SHA required')
    f=load(HERE/'execution_freeze.json');require(f.get('collector_inputs') and f.get('native_and_tool_refs'),'empty freeze denied')
    for r in f['collector_inputs']+f['native_and_tool_refs']+[f['smi']]:verify(r)
    verify(f['manifest']);verify(f['protocol']);verify(f['runtime_proof'])
    manifest=load(f['manifest']['path']);protocol=load(f['protocol']['path'])
    require(manifest['protocol']==f['protocol'],'freeze manifest/protocol binding')
    require(f['plan']==plan(protocol,manifest),'frozen plan differs from canonical domain/argv')
    require(f.get('expected_native_processes')==12 and f.get('expected_export_processes')==6 and len(f.get('extra_runtime_refs',[]))==5,'full plan/runtime denominator')
    require(f['extra_runtime_refs']==load(f['runtime_proof']['path'])['extra_runtime_refs'] and f['native_and_tool_refs']==manifest['files']+f['extra_runtime_refs'],'runtime closure binding')
    return f

def wait_same(proc):
    # Preserve ownership and keep telemetry active even if bookkeeping is interrupted.
    while True:
        try:return proc.wait()
        except BaseException:time.sleep(.05)

def environment(protocol,manifest):
    env=os.environ.copy();env['PATH']=manifest['native_bin']+';E:\\cuda\\bin;'+env.get('PATH','')
    for k,v in {**protocol['runtime']['environment'],**{k:None for k in protocol['runtime']['extra_clear_environment']}}.items():
        if v is None:env.pop(k,None)
        else:env[k]=v
    return env

def stage_run(stage,freeze_sha,manifest,protocol,clock_ref):
    verified_freeze(freeze_sha);verify(clock_ref)
    d=Path(stage['directory']);d.mkdir(parents=True,exist_ok=False)
    receipt={'stage':stage,'created_utc':utc(),'freeze_sha256':freeze_sha,'clock_ref':clock_ref,'status':'not_launched','child_exited':True}
    write_new(d/'launch_intent.json',receipt)
    session=timer=proc=None;rawdoc=None
    try:
        if stage['mode']!='export':
            timer=Win32DeadlineTimer()
            def factory():
                reader=NVMLSMReader()
                try:
                    expected=protocol['runtime']['gpu_expected'];actual=reader.identity
                    require(actual['uuid']==expected['uuid'] and actual['driver_version']==expected['driver'],'actual GPU/driver mismatch')
                    sm=reader.read_sm();require(sm['status']==0 and abs(sm['value']-2400)<=30,'prelaunch SM outside frozen domain')
                    receipt['actual_gpu_identity']=actual
                    return reader
                except BaseException:reader.close();raise
            session=SamplerSession(timer=timer,reader_factory=factory,duration_ns=24*3600*10**9,period_ns=5000000).start()
            require(session.ready.wait(30) and not session.errors and not session.done.is_set(),'telemetry preflight failed')
        with (d/'stdout.txt').open('xb') as out,(d/'stderr.txt').open('xb') as err:
            try:
                proc=subprocess.Popen(stage['argv'],cwd=PROBE,env=environment(protocol,manifest),stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                receipt['launched_pid']=proc.pid;receipt['child_exited']=False
                write_new(d/'launched.json',{'pid':proc.pid,'argv':stage['argv'],'utc':utc(),'freeze_sha256':freeze_sha})
            finally:
                if proc is not None:receipt['returncode']=wait_same(proc);receipt['child_exited']=True
        require(proc is not None and receipt['returncode']==0,'child failed; preserve all artifacts')
        receipt['status']='executed'
    except BaseException as exc:receipt.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        cleanup_errors=[]
        if session:
            time.sleep(.01) # Obtain a genuine right-hand clock sample after child exit.
            try:session.stop()
            except BaseException as exc:cleanup_errors.append('stop: '+str(exc))
            while session.thread.is_alive():
                try:session.join(1)
                except TimeoutError:
                    try:session.stop()
                    except BaseException as exc:
                        if not cleanup_errors:cleanup_errors.append('stop: '+str(exc))
                except BaseException as exc:
                    if not cleanup_errors:cleanup_errors.append('join: '+str(exc))
            try:session.close()
            except BaseException as exc:cleanup_errors.append('close: '+str(exc))
            try:
                data=session.result
                if data is not None:data['lifecycle']=session.lifecycle();write_new(d/'telemetry.json',data)
                receipt['telemetry_errors']=list(session.errors)
                if session.errors:cleanup_errors.append('telemetry worker errors')
            except BaseException as exc:cleanup_errors.append('telemetry evidence: '+str(exc))
        elif timer:
            try:timer.close()
            except BaseException as exc:cleanup_errors.append('timer close: '+str(exc))
        if cleanup_errors:receipt.update(status='failed_cleanup',cleanup_errors=cleanup_errors)
        try:verified_freeze(freeze_sha);verify(clock_ref)
        except Exception as exc:receipt.update(status='failed_identity',error=str(exc))
        receipt['finished_utc']=utc();require(receipt['child_exited'],'cannot release a live child')
        receipt['artifacts']=[ref(p) for p in d.iterdir() if p.is_file()]
        write_new(d/'complete.json',receipt)
    if receipt['status']!='executed':raise RuntimeError('stage failed '+str(d))
    validation={'stage_ordinal':stage['ordinal'],'status':'failed','freeze_sha256':freeze_sha}
    try:
        if stage['mode']!='export':
            rawdoc=audit(stage['raw'],stage,manifest,protocol,proc.pid if stage['mode']=='direct' else None,verified_freeze(freeze_sha)['extra_runtime_refs'])
            telem=load(d/'telemetry.json');require(telem['qpc_frequency']==rawdoc['header']['qpc_frequency'],'telemetry/native QPC domain mismatch')
            rawdoc['clock_gate']=formal_clock_gate(telem,rawdoc['formal_clock_intervals'])
            write_new(d/'raw_audit.json',rawdoc);validation['raw_audit_ref']=ref(d/'raw_audit.json')
        else:
            require((d/'trace.sqlite').is_file(),'SQLite export missing');validation['sqlite_ref']=ref(d/'trace.sqlite')
        validation['status']='structurally_validated'
    except Exception as exc:validation['error']=str(exc);raise
    finally:write_new(d/'validation.json',validation)
    return receipt

def summarize_pairs(freeze_sha,partial=False):
    from quality_fixed import quality
    f=verified_freeze(freeze_sha);groups=[]
    for config in CONFIGS:
        pairs=[];failures=[]
        for pair in range(1,4):
            root=HERE/'runs'/config['id']/f'pair_{pair:02d}'
            try:
                for mode in ('direct','profile','export'):
                    validation=load(root/mode/'validation.json')
                    require(validation.get('status')=='structurally_validated','stage not validated: '+mode)
                    verify(validation['sqlite_ref'] if mode=='export' else validation['raw_audit_ref'])
                direct=load(root/'direct/raw_audit.json');profile=load(root/'profile/raw_audit.json')
                for raw in (direct,profile):
                    require(raw['condition']==config['id'],'summary config identity mismatch')
                    verify(raw['raw_ref']);verify(raw['sidecar_ref'])
                correlated=analyze_trace(root/'export/trace.sqlite',profile,config)
                output=root/'export/strict_trace.json'
                if not output.exists():write_new(output,correlated)
                else:require(load(output)==correlated,'immutable trace summary changed')
                pairs.append({'pair':pair-1,'clock_domain_validated':direct['clock_gate']['passed'] and profile['clock_gate']['passed'],'numerics_all_rows':direct['math_pass'] and profile['math_pass'],'trace_chain_complete':correlated['all36_chain_complete'],'source_path_matches_observed_family':correlated['source_path_matches_observed_family'],'trace_warning_free':not correlated['warning_diagnostics_verbatim'],'profile_kernel':correlated['formal_kernel_union'],'direct_host':direct['host_formal'],'profile_host':profile['host_formal'],'profile_raw':{'signature':{'environment':profile['setup']['environment'],'modules':profile['setup']['loaded_modules_before'],'graph_nodes':profile['setup']['graph_nodes'],'arm':'buffered'}},'raw_refs':[direct['raw_ref'],profile['raw_ref']],'per_call_final_validated':False,'condition':config['id'],'trace_ref':ref(output)})
            except Exception as exc:failures.append({'pair':pair,'reason':type(exc).__name__+': '+str(exc),'retained_in_required_three_pairs':True})
        result=quality({**config,'arm':'buffered'},pairs);result['pair_failures_or_unrun']=failures;groups.append(result)
    statuses=[stage_status(x) for x in f['plan']]
    return {'schema':'r22-long-graph-summary/v1','freeze_sha256':freeze_sha,'expected_stages':18,'completed_receipts':sum(x['process_status']!='not_run' for x in statuses),'successfully_validated_stages':sum(x['validation_status']=='structurally_validated' for x in statuses),'configurations':groups,'stage_status':statuses,'calibration_fitted':False,'coefficients_added':0,'partial':partial}

def stage_status(stage):
    result={key:stage[key] for key in ('ordinal','condition','arm','pair','mode')}
    for name,key,default in [('complete.json','process_status','not_run'),('validation.json','validation_status','not_validated')]:
        path=Path(stage['directory'])/name
        try:result[key]=load(path).get('status') if path.exists() else default
        except Exception as exc:
            result[key]='unreadable_receipt';result[key+'_error']=str(exc)
    return result

def stop_requested(expected):
    path=HERE/'stop.json'
    if not path.exists():return False
    document=load(path)
    require(document.get('freeze_sha256')==expected and document.get('stop') is True,'invalid explicit stop request')
    return True

def run(expected):
    f=verified_freeze(expected);manifest=load(f['manifest']['path']);protocol=load(f['protocol']['path']);session_path=HERE/'clock_session';session_path.mkdir(exist_ok=False)
    def smi(args,name):
        with (session_path/(name+'.stdout.txt')).open('xb') as out,(session_path/(name+'.stderr.txt')).open('xb') as err:
            p=subprocess.Popen([f['smi']['path'],*args],stdout=out,stderr=err);code=wait_same(p)
        r={'argv':[f['smi']['path'],*args],'returncode':code,'stdout':ref(session_path/(name+'.stdout.txt')),'stderr':ref(session_path/(name+'.stderr.txt'))};write_new(session_path/(name+'.json'),r);return r
    result={'freeze_sha256':expected,'started_utc':utc(),'status':'started','completed_stages':0,'attempted_stages':0};lock_attempted=False
    try:
        smi(['--query-gpu=uuid,driver_version,clocks.sm,clocks.mem','--format=csv,noheader'],'before')
        lock_attempted=True;locked=smi(['-lgc','2400,2400'],'lock');require(locked['returncode']==0,'clock lock failed')
        time.sleep(1)
        clock=readback=smi(['--query-gpu=uuid,driver_version,clocks.sm','--format=csv,noheader,nounits'],'readback')
        fields=Path(readback['stdout']['path']).read_text().strip().split(',');require(readback['returncode']==0 and len(fields)==3 and fields[0].strip()==protocol['runtime']['gpu_expected']['uuid'] and fields[1].strip()==protocol['runtime']['gpu_expected']['driver'] and abs(int(fields[2])-2400)<=30,'actual lock readback')
        clock_ref=ref(session_path/'readback.json')
        for stage in f['plan']:
            if stop_requested(expected):
                result['status']='stopped_by_explicit_request';break
            try:
                stage_run(stage,expected,manifest,protocol,clock_ref);result['completed_stages']+=1
            except Exception as exc:result.setdefault('stage_errors',[]).append({'ordinal':stage['ordinal'],'error':str(exc)})
            result['attempted_stages']+=1
            if result['attempted_stages']==6:
                checkpoint={'freeze_sha256':expected,'completed_stages':6,'review_required':False,'nonblocking':True,'clock_ref':clock_ref}
                try:
                    summary=summarize_pairs(expected,True);write_new(HERE/'first_pair_summary.json',summary)
                    checkpoint['summary_status']='saved'
                except Exception as exc:checkpoint.update(summary_status='failed',error=str(exc))
                write_new(HERE/'first_pair_checkpoint.json',checkpoint)
            print(json.dumps({'completed_stages':result['completed_stages'],'expected_stages':18}),flush=True)
        if result['attempted_stages']==18:
            write_new(HERE/'summary.json',summarize_pairs(expected));result['status']='completed_with_failures' if result.get('stage_errors') else 'completed'
    except BaseException as exc:
        result.update(status='failed_or_incomplete',error=type(exc).__name__+': '+str(exc))
    finally:
        if lock_attempted:
            try:
                result['reset']=smi(['-rgc'],'reset')
                if result['reset']['returncode']!=0:result['status']='failed_cleanup'
            except BaseException as exc:result.update(status='failed_cleanup',reset_error=type(exc).__name__+': '+str(exc))
        result['finished_utc']=utc();result['expected_stages']=18
        result['stage_status']=[stage_status(x) for x in f['plan']]
        result['remaining_stages']=[x for x in f['plan'] if not (Path(x['directory'])/'complete.json').exists()]
        write_new(HERE/'controller_result.json',result)
    print(json.dumps(result,ensure_ascii=False));return 0 if result['status']=='completed' and result.get('reset',{}).get('returncode')==0 else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','verify','run']);ap.add_argument('--expected-freeze-sha256');a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='verify':verified_freeze(a.expected_freeze_sha256);print('freeze verified')
    else:raise SystemExit(run(a.expected_freeze_sha256))