"""One untimed original-DLL Q4_K path/numerical qualification; no cost/timing mode."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,re,struct,subprocess,types
import build,reference
HERE=Path(__file__).resolve().parent
LOADED=[build.ref(p) for p in (__file__,build.__file__,reference.__file__)]


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def need(ok,message):
    if not ok:raise ValueError(message)


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))


def snapshot(manifest):
    for r in LOADED:build.check_ref(r)
    m=build.verify(manifest)
    return {'manifest_ref':build.ref(manifest),'inputs':m['inputs'],'target_executable':m['target_executable'],'loaded_code_refs':LOADED}


def load_guard(protocol):
    pin=protocol['execution_guard_refs'][0];build.check_ref(pin);data=Path(pin['path']).read_bytes()
    need(hashlib.sha256(data).hexdigest()==pin['sha256'],'guard import changed')
    mod=types.ModuleType('r29_host_control_idle');mod.__file__=pin['path'];exec(compile(data,pin['path'],'exec'),mod.__dict__)
    build.check_ref(pin);return mod


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
        target=any(re.fullmatch(r'q4k-target\..+\.exe',value) for value in names)
        local=path is not None and path.is_relative_to(HERE.resolve()) and path.suffix.casefold()=='.exe'
        host=path is not None and re.fullmatch(r'q4k-host-tests\.[0-9]+\.exe',path.name.casefold())
        if target or (local and not host):
            blocked.append({'pid':row['pid'],'ppid':row.get('ppid'),'name':row.get('name'),
                'executable_path':executable,'reason':'blocked_r29_target' if target else 'blocked_r29_local_executable'})
    return blocked


def process_snapshot(protocol):
    guard=load_guard(protocol)
    powershell=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    script="[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,ParentProcessId,Name,CommandLine,ExecutablePath) | ConvertTo-Json -Compress"
    child=subprocess.Popen([str(powershell),'-NoProfile','-NonInteractive','-Command',script],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8-sig',
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:output,error=child.communicate()
    except BaseException as exc:
        raise RuntimeError('process inventory interrupted; helper left running pid='+str(child.pid)) from exc
    need(child.returncode==0,'process inventory failed; returncode='+str(child.returncode))
    rows=json.loads(output);rows=[rows] if isinstance(rows,dict) else rows
    need(isinstance(rows,list) and bool(rows),'process inventory unavailable')
    records=[{'pid':int(r['ProcessId']),'ppid':int(r['ParentProcessId']),'name':r['Name'],
        'cmdline':r['CommandLine'],'executable_path':r['ExecutablePath']} for r in rows]
    base=guard.process_conflicts(records);local=local_process_conflicts(records)
    conflicts={r['pid']:r for r in base+local}
    return {'created_utc':now(),'inventory_count':len(records),'conflicts':list(conflicts.values()),
        'local_checker_ref':build.ref(__file__),'frozen_base_guard_ref':protocol['execution_guard_refs'][0],
        'scope':'frozen R28 exclusion plus R29 exact target names/local executables, including direct/orphan launches',
        'continuous_exclusivity_verified':False,'background_GPU_isolation_verified':False,
        'desktop_GPU_process_absence_required':False}


def idle(protocol):
    value=process_snapshot(protocol)
    need(not value['conflicts'],'project not idle: '+json.dumps(value['conflicts']))
    return value


def environment(protocol):
    source=dict(os.environ);known={'CUDA_PATH','CUDA_HOME','CUDA_ROOT'}
    rejected=[k for k in source if (k.upper().startswith(('CUDA_','GGML_','CUPTI_','NVTX_','NSYS_','CAPTURE_'))
        or 'INJECTION' in k.upper()) and k.upper() not in known and not k.upper().startswith('CUDA_PATH_V')]
    need(not rejected,'inherited dispatch/injection environment present: '+','.join(sorted(rejected)))
    keep={'SYSTEMROOT','WINDIR','COMSPEC','TEMP','TMP','LOCALAPPDATA','APPDATA','USERPROFILE','HOMEDRIVE','HOMEPATH','PROGRAMDATA','PROGRAMFILES','PROGRAMFILES(X86)'}
    env={k:v for k,v in source.items() if k.upper() in keep};system=source.get('SystemRoot','C:/Windows')
    dirs=list(dict.fromkeys(str(Path(r['path']).parent) for r in protocol['target_modules']))
    env['PATH']=os.pathsep.join(dirs+[str(Path(system)/'System32'),system])
    env.update(CAPTURE_RUNTIME_AUTHORIZED='1',GGML_CUDA_DISABLE_GRAPHS='1',GGML_CUDA_PDL='1',CAPTURE_CUPTI_DLL=protocol['CUPTI']['path'])
    return env


def validate_capture(raw,protocol):
    need(raw['schema']=='r29-q4k-target-launch-capture/v1' and raw['status']=='path_observed_pending_independent_numeric','target path rejected')
    need(raw['timed'] is False and raw['performance_parameter_admitted'] is False and raw['target_LLM_latency_used'] is False,'invalid cost scope')
    need(raw['protocol_sha256']==build.ref(HERE/'protocol.json')['sha256'],'protocol mismatch')
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
    for r in fixtures.values():build.check_ref(r)
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
            'full_outputs':rows,'raw_output_refs':[build.ref(directory/n) for n in ('actual.q8_1.bin','actual.f32.bin')],
            'reference_files':fixtures,'bound_fitted':False,'timing_or_cost_admitted':False}
    build.write_new(directory/'numerical.json',result)
    need(result['passed'],'independent full numerical qualification rejected')
    return build.ref(directory/'numerical.json')


def execute(manifest):
    protocol=load(HERE/'protocol.json');before=snapshot(manifest)
    directory=HERE/'runs'/'capture.0001';directory.mkdir(parents=True,exist_ok=False)
    finish={'schema':'r29-q4k-finish/v1','status':'rejected','returncode':None,'identity_before':before,
            'identity_unchanged':False,'path_qualified':False,'numeric_qualified':False,'errors':[],
            'timed':False,'cost_parameters':0,'target_DLL_rebuilt':False,'LLM_measured':False,'termination_requested':False}
    try:
        finish['idle_before']=idle(protocol);env=environment(protocol)
        argv=[before['target_executable']['path'],'--run-recorder-only',str(directory/'raw.json')]
        build.write_new(directory/'start.json',{'created_utc':now(),'identity_before':before,'argv':argv,
                         'fixture_manifest_ref':protocol['fixture_manifest'],'existing_authorization_serial_execution':True})
        with (directory/'stdout.log').open('xb') as out,(directory/'stderr.log').open('xb') as err:
            child=subprocess.Popen(argv,env=env,cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            build.write_new(directory/'child.json',{'pid':child.pid})
            try:finish['returncode']=child.wait()
            except BaseException as error:
                finish['child_may_be_live']=child.poll() is None;raise RuntimeError('natural child observation interrupted; no kill') from error
        need(finish['returncode']==0,'target capture process failed')
        validate_capture(load(directory/'raw.json'),protocol);finish['path_qualified']=True
        finish['numerical_ref']=validate_numeric(directory,protocol);finish['numeric_qualified']=True
    except BaseException as error:finish['errors'].append(type(error).__name__+': '+str(error))
    finally:
        try:
            finish['idle_after']=idle(protocol)
        except BaseException as error:finish['errors'].append('post idle: '+str(error))
        try:
            finish['identity_after']=snapshot(manifest);finish['identity_unchanged']=finish['identity_after']==before
            need(finish['identity_unchanged'],'execution input changed')
        except BaseException as error:finish['errors'].append('post identity: '+str(error))
        for name in ('raw.json','start.json','child.json','actual.q8_1.bin','actual.f32.bin'):
            try:
                if (directory/name).exists():finish[name+'_ref']=build.ref(directory/name)
                elif finish['path_qualified'] or finish['numeric_qualified']:raise ValueError('required final artifact missing: '+name)
            except BaseException as error:finish['errors'].append('final reference: '+str(error))
        if not finish['errors'] and finish['path_qualified'] and finish['numeric_qualified'] and finish['identity_unchanged']:
            finish['status']='fixed_shape_target_path_and_numeric_qualified_cost_unmeasured'
        finish['finished_utc']=now();build.write_new(directory/'finish.json',finish)
    need(not finish['errors'],'rejected attempt preserved; no retry')
    return finish


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--execute-untimed-qualification',action='store_true');a=parser.parse_args()
    need(a.execute_untimed_qualification,'explicit root-reviewed serial execution flag required')
    print(json.dumps(execute(a.manifest)))
