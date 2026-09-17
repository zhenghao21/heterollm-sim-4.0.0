"""One new Q4 wrapper path/numeric pass; no timing, retries or GPU by default."""
from pathlib import Path
import argparse,importlib.util,json,math,os,re,struct,subprocess
import psutil
P=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location("r31_qualification_builder",P/"build.py");b=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(b)
need=b.need
ref=b.ref
load=lambda path:json.loads(Path(path).read_text(encoding='utf8'))

def local_process_conflicts(rows,current_pid=None):
    """R29 supplement: direct/orphan target executables need no Python parent."""
    current_pid=os.getpid() if current_pid is None else current_pid
    blocked=[]
    for row in rows:
        if row['pid']==current_pid:continue
        name=(row.get('name') or '').casefold()
        executable=row.get('executable_path')
        path=Path(executable).resolve() if executable else None
        names={name,path.name.casefold() if path else ''}
        target=any(re.fullmatch(r'q4k-target\..+\.exe',value) or value in ('q4k-timing.exe','q4k-wrapper-qualification.exe') for value in names)
        local=path is not None and path.is_relative_to(P.resolve()) and path.suffix.casefold()=='.exe'
        host=path is not None and re.fullmatch(r'q4k-host-tests\.[0-9]+\.exe',path.name.casefold())
        if target or (local and not host):
            blocked.append({'pid':row['pid'],'ppid':row.get('ppid'),'name':row.get('name'),
                'executable_path':executable,'reason':'blocked_r29_target' if target else 'blocked_r29_local_executable'})
    return blocked


def idle():
    p=load(P/'protocol.json');pin=p['identities']['frozen_R28_guard'];b.verify_ref(pin)
    guard=b.load(Path(pin['path']),'frozen_r28_process_guard')
    records=[]
    for process in psutil.process_iter(['pid','ppid','name','cmdline','exe']):
        r=process.info;records.append({'pid':r['pid'],'ppid':r['ppid'],'name':r['name'],
            'cmdline':subprocess.list2cmdline(r['cmdline']) if r['cmdline'] else None,'executable_path':r['exe']})
    base=guard.process_conflicts(records);local=local_process_conflicts(records)
    conflicts={r['pid']:r for r in base+local}
    need(not conflicts,'project not idle: '+json.dumps(list(conflicts.values())))
    return {'created_utc':b.now(),'inventory_count':len(records),'conflicts':[],
        'checker_ref':ref(__file__),'base_guard_ref':pin,'continuous_exclusivity_verified':False}


def reference_module():
    p=load(P/'protocol.json');pin=p['identities']['Q4_reference'];b.verify_ref(pin)
    return b.load(Path(pin['path']),'r29_independent_Q4_reference')
reference=reference_module()

def validate_capture(raw,protocol):
    need(raw['schema']=='r31-q4k-wrapper-launch-capture/v1' and raw['status']=='path_observed_pending_independent_numeric','target path rejected')
    need(raw['timed'] is False and raw['performance_parameter_admitted'] is False and raw['target_LLM_latency_used'] is False,'invalid cost scope')
    need(raw['protocol_sha256']==ref(P/'protocol.json')['sha256'],'protocol mismatch')
    c=raw['configuration'];need(all(c[k]==v for k,v in protocol['shape'].items()),'shape differs')
    need(raw['hardware']['uuid']==protocol['GPU_uuid'] and raw['hardware']['SM_count']==84,'wrong hardware')
    need(raw['compile_CUPTI_API_version']==26 and raw['runtime_CUPTI_API_version']==130401,'callback ABI/runtime mismatch')
    need(Path(raw['CUPTI_path']).resolve()==Path(protocol['CUPTI']['path']).resolve(),'CUPTI loaded from other path')
    expected={Path(r['path']).name:r for r in protocol['target_modules']}
    for phase in ('loaded_modules_before','loaded_modules_after'):
        modules=raw[phase]
        need(len(modules)==len(expected),'module coverage differs')
        for r in modules:need(r['sha256']==expected[Path(r['path']).name]['sha256'] and Path(r['path']).resolve()==Path(expected[Path(r['path']).name]['path']).resolve(),'target module differs')
    need(raw['loaded_modules_before']==raw['loaded_modules_after'],'module identity changed')
    need(raw['overflow'] is False and raw['malformed_callback'] is False and raw['memory_api_count']==0,'capture incomplete/impure')
    auxiliary=raw['runtime_auxiliary_calls']
    need(raw['runtime_auxiliary_count']==len(auxiliary),'auxiliary callback coverage differs')
    sync_names={'cudaStreamSynchronize','cudaStreamSynchronize_ptsz','cudaDeviceSynchronize','cudaEventSynchronize'}
    for api in auxiliary:
        need(api['classification']=='synchronization' and api['api_name'] in sync_names,
             'captured allocation/free/copy/set or unknown auxiliary API')
        need(api['after_launch_pair'] is True and api['exit_seen'] is True and api['return_code']==0,
             'only successful synchronization after the launch pair is allowed')
    need(raw['launch_count']==2 and len(raw['launches'])==2,'exact conversion plus main required')
    need(all(v is True for v in raw['pointer_associations'].values()),'pointer associations invalid')
    for row,expected_launch in zip(raw['launches'],[protocol['expected_conversion'],protocol['expected_main']]):
        need(row['symbol']==expected_launch['symbol'] and row['api_id']==430 and row['exit_seen'] and row['return_code']==0,'wrong kernel/API/return')
        need(row['grid']==expected_launch['grid'] and row['block']==expected_launch['block'],'wrong actual launch geometry')
        need(row['geometry_observed'] and row['attributes_source_qualified'] and row['extended_api'],'unobserved geometry/PDL')
        need(row['shared']==0 and row['reported_attribute_count']==row['captured_attribute_count']==1,'wrong shared/attribute count')
        attrs=row['attributes'];need(len(attrs)==1 and attrs[0]['id']==6 and attrs[0]['programmaticStreamSerializationAllowed']==1,'wrong observed PDL value')
    conv,main=raw['launches']
    need(conv['context']==main['context'] and conv['context']>0 and conv['stream']==main['stream'],'launch stream/context mismatch')
    need(conv['correlation']!=main['correlation'] and conv['function']!=main['function'],'ambiguous launch identity')
    a,b=conv['arguments'],main['arguments']
    tensors=raw['graph_tensors']
    need(a['x']==tensors['input'] and b['vx']==tensors['weights'] and b['dst']==tensors['output'] and a['vy']==b['vy'],'actual pointer binding differs')
    need(len({a['x'],a['vy'],b['vx'],b['dst']})==4 and all(v>0 for v in (a['x'],a['vy'],b['vx'],b['dst'])),'invalid alias/null pointers')
    need(a['ne2_fastdiv']==[1,0,1] and b['channel_ratio_fastdiv']==b['sample_ratio_fastdiv']==[1,0,1] and b['nchannels_y_fastdiv']==[0,0,0],'fastdiv contract differs')
    need(a['ne00']==a['ne0']==a['s01']==a['s02']==a['s03']==2048 and a['ne1']==1,'conversion dimensions')
    need(b['ncols_x']==2048 and b['stride_row_x']==8 and b['stride_col_y']==64 and b['stride_col_dst']==2048,'main stride/reduction')
    need(b['stride_channel_x']==b['stride_sample_x']==16384 and b['stride_channel_y']==b['stride_sample_y']==64,'main channel/sample stride')
    need(b['stride_channel_dst']==b['stride_sample_dst']==2048 and b['ids_stride']==0 and b['ids']==0,'main output stride/ids')
    need(all(v==0 for v in b['fusion'].values()),'unexpected fused work')
    return True

def validate_numeric(directory,protocol):
    fixtures=protocol['fixture_files']
    for r in fixtures.values():b.check_ref(r)
    packed=Path(fixtures['weights.q4_k.bin']['path']).read_bytes()
    input_bytes=Path(fixtures['input.f32.bin']['path']).read_bytes()
    recomputed_q8=reference.expected_q8(input_bytes)
    expected_q8=Path(fixtures['expected.q8_1.bin']['path']).read_bytes()
    need(expected_q8==recomputed_q8,'frozen Q8 independent reference mismatch')
    rawq=(directory/'actual.q8_1.bin').read_bytes();need(len(rawq)==2304,'Q8 output size wrong')
    recomputed=reference.reference(packed,recomputed_q8)
    expected=Path(fixtures['reference.f64.bin']['path']).read_bytes();bounds=Path(fixtures['bounds.f64.bin']['path']).read_bytes()
    need(expected==struct.pack('<2048d',*recomputed['values']) and bounds==struct.pack('<2048d',*recomputed['bounds']),'frozen reference/bound changed')
    data=(directory/'actual.f32.bin').read_bytes();need(len(data)==8192,'main output size wrong')
    values=struct.unpack('<2048f',data);rows=[]
    for i,(actual,want,bound) in enumerate(zip(values,recomputed['values'],recomputed['bounds'])):
        finite=math.isfinite(actual);error=abs(actual-want) if finite else None
        rows.append({'row':i,'actual':actual if finite else None,'reference_f64':want,'bound':bound,
                     'absolute_error':error,'passed':finite and error<=bound})
    q8bad=[i for i,(a,b) in enumerate(zip(rawq,expected_q8)) if a!=b]
    result={'schema':'r29-q4k-independent-numerical/v1','passed':not q8bad and all(r['passed'] for r in rows),
            'Q8_byte_mismatch_count':len(q8bad),'Q8_first_mismatch_offsets':q8bad[:32],
            'full_outputs':rows,'raw_output_refs':[b.ref(directory/n) for n in ('actual.q8_1.bin','actual.f32.bin')],
            'reference_files':fixtures,'bound_fitted':False,'timing_or_cost_admitted':False}
    b.write_new(directory/'numerical.json',result)
    need(result['passed'],'independent full numerical qualification rejected')
    return b.ref(directory/'numerical.json')


def verify_target():
    p=load(P/'protocol.json');ids=p['identities']
    for k in ('Q4_target_finish','Q4_target_manifest','Q4_target_protocol'):b.verify_ref(ids[k])
    finish=load(ids['Q4_target_finish']['path'])
    need(finish.get('status')=='fixed_shape_target_path_and_numeric_qualified_cost_unmeasured' and
        finish.get('identity_unchanged') is True and finish.get('path_qualified') is True and finish.get('numeric_qualified') is True,
        'R29 target is not path/numerically qualified')
    for item in b.references(finish):b.verify_ref(item)
    return ids['Q4_target_finish']


def verify_numerical_evidence(directory,p):
    # Recompute all values without writing/replacing the retained numerical.json.
    for item in p['fixture_files'].values():b.verify_ref(item)
    files=p['fixture_files'];packed=Path(files['weights.q4_k.bin']['path']).read_bytes()
    q8=reference.expected_q8(Path(files['input.f32.bin']['path']).read_bytes())
    need(q8==Path(files['expected.q8_1.bin']['path']).read_bytes()==(directory/'actual.q8_1.bin').read_bytes(),'wrapper Q8 differs')
    expected=reference.reference(packed,q8)
    need(struct.pack('<2048d',*expected['values'])==Path(files['reference.f64.bin']['path']).read_bytes(),'reference changed')
    need(struct.pack('<2048d',*expected['bounds'])==Path(files['bounds.f64.bin']['path']).read_bytes(),'bound changed')
    actual=struct.unpack('<2048f',(directory/'actual.f32.bin').read_bytes())
    need(all(math.isfinite(a) and abs(a-e)<=bound for a,e,bound in zip(actual,expected['values'],expected['bounds'])),'wrapper numeric differs')


def verify_qualified():
    p=load(P/'protocol.json');m=b.verify_build();directory=P/'qualification.0001'
    result=load(directory/'finish.json')
    need(result.get('status')=='new_Q4_wrapper_path_and_independent_numeric_qualified' and result.get('identity_unchanged') is True,
        'new Q4 wrapper qualification required before any timing')
    need(result['build_ref']==ref(P/'build_manifest.json'),'wrapper qualified a different build')
    for item in b.references(result):b.verify_ref(item)
    verify_target();validate_capture(load(directory/'raw.json'),p);verify_numerical_evidence(directory,p)
    return {'wrapper_finish_ref':ref(directory/'finish.json'),'wrapper_raw_ref':ref(directory/'raw.json')}


def execute():
    p=load(P/'protocol.json');m=b.verify_build();host=load(P/'host_test_receipt.json')
    need(host.get('returncode')==0 and host.get('inputs_unchanged') is True and host.get('manifest_ref')==ref(P/'build_manifest.json'),'host-tested frozen build required')
    verify_target();before_idle=idle();directory=P/'qualification.0001';directory.mkdir(exist_ok=False)
    result={'schema':'r31-q4-wrapper-qualification/v1','status':'rejected','errors':[],
        'build_ref':ref(P/'build_manifest.json'),'protocol_ref':ref(P/'protocol.json'),'idle_before':before_idle,
        'timed':False,'cost_parameters':0,'termination_requested':False,'wrapper_object_ref':p['identities']['wrapper_object']}
    try:
        b.write_new(directory/'start.json',{'created_utc':b.now(),'build_ref':result['build_ref'],'protocol_ref':result['protocol_ref'],'budget':p['qualification_budget']})
        keep={'systemroot','windir','comspec','temp','tmp','localappdata','appdata','userprofile','homedrive','homepath','programdata','programfiles','programfiles(x86)'}
        env={k:v for k,v in os.environ.items() if k.lower() in keep};windows=os.environ.get('SystemRoot','C:/Windows')
        env.update(PATH=os.pathsep.join(list(dict.fromkeys(str(Path(r['path']).parent) for r in p['target_modules']))+[str(Path(windows)/'System32'),windows]),
            CAPTURE_RUNTIME_AUTHORIZED='1',GGML_CUDA_DISABLE_GRAPHS='1',GGML_CUDA_PDL='1',CAPTURE_CUPTI_DLL=p['CUPTI']['path'])
        with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
            child=subprocess.Popen([m['qualification_executable']['path'],'--run-recorder-only',str(directory/'raw.json')],cwd=P,env=env,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            b.write_new(directory/'child.json',{'pid':child.pid,'termination_requested':False})
            result['returncode']=child.wait()
        need(result['returncode']==0,'wrapper capture failed; no retry')
        validate_capture(load(directory/'raw.json'),p);result['path_qualified']=True
        validate_numeric(directory,p);result['numeric_qualified']=True
    except BaseException as e:result['errors'].append(type(e).__name__+': '+str(e))
    finally:
        try:
            result['idle_after']=idle();b.verify_build();result['identity_unchanged']=result['build_ref']==ref(P/'build_manifest.json')
            need(result['identity_unchanged'],'build identity changed')
            for name in ('start.json','child.json','raw.json','actual.q8_1.bin','actual.f32.bin','numerical.json'):
                result[name+'_ref']=ref(directory/name)
        except BaseException as e:result['errors'].append('postcheck: '+type(e).__name__+': '+str(e))
        if not result['errors']:result['status']='new_Q4_wrapper_path_and_independent_numeric_qualified'
        result['finished_utc']=b.now();b.write_new(directory/'finish.json',result)
    need(not result['errors'],'wrapper rejected, evidence retained: '+str(directory))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute-untimed-qualification',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args()
    if args.execute_untimed_qualification:print(json.dumps(execute()))
    elif args.check:print(json.dumps(verify_qualified()))
    else:raise ValueError('explicit execution or read-only check required')
