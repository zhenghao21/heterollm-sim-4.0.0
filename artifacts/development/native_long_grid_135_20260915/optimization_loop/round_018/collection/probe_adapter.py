"""Strict adapter for the graph probe protocol. No inferred legacy GEMM arguments."""
from pathlib import Path
import json
from datetime import datetime
import re
from common import PROBE,POLICY,IdentityError,ref,verify_ref,load

PROTOCOL_SCHEMA='graph-submit-probe-protocol/v1'


def protocol_document():
    path=PROBE/'protocol.json'
    if not path.is_file():raise IdentityError('graph probe protocol is not ready; no command or measurement interpretation permitted')
    protocol=load(path)
    if protocol.get('schema')!=PROTOCOL_SCHEMA:raise IdentityError('unsupported graph probe protocol schema')
    return protocol


def validate_protocol(protocol):
    if protocol.get('schema')!=PROTOCOL_SCHEMA:raise IdentityError('graph probe protocol schema differs')
    configs=protocol.get('configs',[])
    expected={(e,n) for e in (1024,262144) for n in (1,8,32)}
    if len(configs)!=6 or len({c.get('id') for c in configs})!=6 or {(c.get('elements'),c.get('nodes')) for c in configs}!=expected:
        raise IdentityError('graph six-config factorial differs')
    for config in configs:
        if {config.get('dtype'),config.get('operator'),config.get('layout')}!={'F32','SCALE','contiguous_1d'} or config.get('scale')!=.5 or config.get('bias')!=0:
            raise IdentityError('graph mathematical operation differs')
    execution=protocol.get('execution',{})
    for key,value in {'pairs':3,'first':1,'warmup':5,'formal':30,'graph_calls_per_sample':1,
        'final_backend_synchronize_calls_per_graph':1,'caller_threads':1,'cpu_backend_nodes':0,'cuda_index':0,
        'nvtx_in_both_arms':True,'probe_cuda_events':False}.items():
        if execution.get(key)!=value or type(execution.get(key)) is not type(value):raise IdentityError('graph execution contract mismatch: '+key)
    if execution.get('arms')!=['direct','profile']:raise IdentityError('unreviewed graph observer arms')
    options=protocol.get('profiler',{}).get('options',[])
    if '--kill=false' not in options or '--cuda-graph-trace=node' not in options:raise IdentityError('graph tracing/profiler lifetime options absent')
    if protocol.get('runtime',{}).get('environment',{}).get('GGML_CUDA_DISABLE_GRAPHS','not-listed') is not None:
        raise IdentityError('native default graph mode must remain explicitly absent')
    return configs


def validate_binding(root,protocol,build):
    validate_protocol(protocol)
    if build.get('schema')!='graph-submit-probe-build/v1' or build.get('status')!='compiled_host_tested_not_gpu_executed' or not build.get('files') or not build.get('executable'):raise IdentityError('graph probe not compiled/frozen')
    executable=build['executable'];verify_ref(executable)
    if Path(executable['path']).resolve()!=Path(root).resolve()/'graph-submit-probe.exe':raise IdentityError('graph manifest executable path differs')
    files={str(Path(r['path']).resolve()).casefold():r for r in build['files']}
    for name in ('protocol.json','graph_submit_probe.cpp','math_reference.h','frozen_module_guard.h'):
        key=str(Path(root).resolve()/name).casefold()
        if key not in files:raise IdentityError('required graph source missing from native freeze: '+name)
    if files.get(str(Path(executable['path']).resolve()).casefold())!=executable:raise IdentityError('graph executable not in full source closure')
    for r in build['files']:verify_ref(r)


def mode_contract():
    return 'Direct/profile are the same event-free graph program, native-default CUDA graphs and fusion. Profile adds reviewed Nsight node tracing only; raw kernel counts for untraced direct remain null.'


def kernel_role(kernel):
    return 'scale' if kernel.get('short_name_text')=='scale_f32' else 'unsupported'


def validate_kernel_chain(kernels,config):
    issues=[];ordered=sorted(kernels,key=lambda k:(k['start'],k['evidence_rowid']))
    if len(ordered)!=config['nodes']:issues.append('actual_SCALE_kernel_count_differs_from_unfused_planned_chain')
    for k in ordered:
        if kernel_role(k)!='scale':issues.append('unexpected_fused_or_other_kernel_path')
        if [k.get(n) for n in ('gridX','gridY','gridZ')]!=[(config['elements']+255)//256,1,1] or [k.get(n) for n in ('blockX','blockY','blockZ')]!=[256,1,1]:issues.append('actual_SCALE_launch_geometry_differs_from_locked_source')
        if k.get('dynamicSharedMemory')!=0:issues.append('unexpected_SCALE_dynamic_shared_memory')
    if len({(k.get('deviceId'),k.get('streamId'),k.get('globalPid')) for k in ordered})!=1:issues.append('multiple_or_missing_graph_device_stream_process')
    # PDL may permit overlapping outer execution intervals; this test does not
    # invent dependency edges from sorted timestamps. Actual source graph edges
    # must be checked by raw graph validation in audit_raw.
    return {'complete':not issues,'observed_family':'SCALE_F32' if not issues else 'unsupported',
        'roles':[kernel_role(k) for k in ordered],'issues':issues,'actual_kernel_count':len(ordered),
        'source_path_matches_observed_family':not issues,'temporal_nonoverlap_required':False,
        'dependency_policy':'Raw GGML predecessor-chain proof plus same-process single-stream kernel mapping; PDL interval overlap is not a graph-edge proof.'}



def app_argv(config,output):
    validate_protocol(protocol_document())
    if config not in protocol_document()['configs']:raise IdentityError('config absent from actual graph protocol')
    pair=Path(output).parent.parent.name
    if re.fullmatch(r'pair_0[123]',pair) is None:raise IdentityError('output path lacks explicit frozen pair identity')
    return ['--run','--config',config['id'],'--pair-id',config['id']+'/'+pair,'--output',str(output)]



def stage_command(stage,directory,protocol):
    """Single source of argv construction and later evidence checks, never executes."""
    directory=Path(directory).resolve()
    selected=[c for c in protocol['configs'] if c['id']==stage.get('config_id')]
    if len(selected)!=1 or type(stage.get('pair')) is not int or not 0<=stage['pair']<3:
        raise IdentityError('stage config/pair outside frozen matrix')
    if directory.name!=stage.get('mode') or directory.parent.name!=f"pair_{stage['pair']+1:02d}" or directory.parent.parent.name!=stage['config_id']:
        raise IdentityError('stage directory does not match config/pair/mode')
    config=selected[0];app=[protocol['executable']['path'],*app_argv(config,directory/'microbench.json')]
    if stage['mode']=='direct':return app
    if stage['mode']=='profile':return [protocol['nsys']['executable']['path'],'profile',*protocol['nsys_profile_options'],'--output='+str(directory/'trace'),*app]
    if stage['mode']=='export':return [protocol['nsys']['executable']['path'],'export','--type=sqlite','--force-overwrite=false','--output='+str(directory/'trace.sqlite'),str(directory.parent/'profile/trace.nsys-rep')]
    raise IdentityError('unknown stage mode')


def process_origin(document,config):
    """Validate actual C++ origin, without manufacturing identity from its case."""
    header,setup,footer=document['header'],document['setup'],document['footer']
    def positive(value,name):
        if type(value) is not int or value<=0:raise IdentityError('missing or invalid native '+name)
        return value
    pid=positive(header.get('pid'),'PID');start=positive(header.get('qpc_start'),'start QPC');end=positive(footer.get('qpc_end'),'end QPC')
    if pid>=1<<24:raise IdentityError('native PID exceeds reviewed Nsight24-bit process encoding')
    if end<=start:raise IdentityError('native process QPC lifetime reversed')
    for field,value in [('utc_start',header.get('utc_start')),('utc_end',footer.get('utc_end'))]:
        if not isinstance(value,str) or not value.endswith('Z'):raise IdentityError('missing native UTC origin: '+field)
        try:timestamp=datetime.fromisoformat(value.replace('Z','+00:00'))
        except ValueError as exc:raise IdentityError('invalid native UTC origin: '+field) from exc
        if timestamp.utcoffset().total_seconds()!=0:raise IdentityError('native UTC origin is not UTC')
    pair=setup.get('pair_id')
    if not isinstance(pair,str) or re.fullmatch(re.escape(config['id'])+r'/pair_0[123]',pair) is None:
        raise IdentityError('missing or invalid native pair identity')
    raw_path=document.get('source_raw',{}).get('path')
    if not isinstance(raw_path,str) or not Path(raw_path).is_absolute():raise IdentityError('actual raw source path absent')
    argv=header.get('argv')
    expected=[str((PROBE/'graph-submit-probe.exe').resolve()),'--run','--config',config['id'],'--pair-id',pair,'--output',str(Path(raw_path).resolve())]
    if not isinstance(argv,list) or any(not isinstance(x,str) for x in argv) or argv!=expected:
        raise IdentityError('native argv/executable/output does not match captured raw origin')
    for call in document.get('runs',[]):
        lo,hi=call.get('qpc_start'),call.get('qpc_end')
        if type(lo) is not int or type(hi) is not int or not start<=lo<hi<=end:
            raise IdentityError('graph call outside actual native QPC lifetime')
    tid=setup.get('scheduling',{}).get('caller_thread_id')
    positive(tid,'caller thread ID')
    if tid>=1<<24:raise IdentityError('native thread ID exceeds reviewed Nsight24-bit thread encoding')
    return {'pid':pid,'caller_thread_id':tid,'qpc_start':start,'qpc_end':end,
            'utc_start':header['utc_start'],'utc_end':footer['utc_end'],'pair_id':pair,
            'argv':argv,'raw_ref':document['source_raw'],'process_identity_source':'frozen_cpp_actual_header_and_footer'}


def unique_process_origins(pairs):
    """Reject copied process evidence; permit genuine OS PID reuse after exit."""
    seen={};intervals={};issues=[]
    for pair in pairs:
        for arm,origin in pair.get('process_origins',{}).items():
            key=(origin['pid'],origin['qpc_start']);owner=(pair.get('pair'),arm)
            previous=seen.get(key)
            if previous is not None:issues.append({'reason':'duplicate_native_process_origin','first':previous,'second':owner})
            seen[key]=owner
            for lo,hi,other in intervals.setdefault(origin['pid'],[]):
                if origin['qpc_start']<hi and origin['qpc_end']>lo:issues.append({'reason':'same_PID_overlapping_native_lifetimes','first':other,'second':owner})
            intervals[origin['pid']].append((origin['qpc_start'],origin['qpc_end'],owner))
    return issues


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate raw JSON field: '+key)
        result[key]=value
    return result


def read_raw(path):
    path=Path(path);before=ref(path)
    if before['bytes']>32*1024*1024:raise ValueError('graph JSONL exceeds bounded36call schema')
    records=[]
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():raise ValueError('empty raw JSONL record')
            records.append(json.loads(line,object_pairs_hook=_unique_object,
                parse_constant=lambda token:(_ for _ in ()).throw(ValueError('nonfinite raw JSON '+token))))
    if ref(path)!=before:raise ValueError('graph raw changed during parsing')
    if not records or len(records)>160:raise ValueError('graph record count outside36call bounds')
    kinds={'header','setup','graph_call','validation','intermediate_validation','footer'}
    if any(r.get('record') not in kinds for r in records):raise ValueError('raw error or unsupported record retained as failure')
    singleton={}
    for name in ('header','setup','footer'):
        selected=[r for r in records if r['record']==name]
        if len(selected)!=1:raise ValueError('missing/duplicate graph '+name)
        singleton[name]=selected[0]
    if records[0]['record']!='header' or records[-1]['record']!='footer':raise ValueError('raw header/footer order')
    header,setup,footer=(singleton[n] for n in ('header','setup','footer'))
    if header.get('schema')!='graph-submit-probe/v1':raise ValueError('raw schema differs from actual C++ emitter')
    f=header.get('qpc_frequency')
    if type(f) is not int or f<=0:raise ValueError('raw QPC frequency invalid')
    calls=[r for r in records if r['record']=='graph_call'];validation=[r for r in records if r['record']=='validation']
    by_label={r['label']:r for r in validation}
    if len(by_label)!=len(validation):raise ValueError('duplicate numerical validation label')
    runs=[]
    for call in calls:
        v=by_label.get(call['label'])
        if v is None:raise ValueError('graph call lacks its numerical validation')
        runs.append({'phase':v['phase'],'index':v['index'],'nvtx_label':call['label'],
            'qpc_start':call['qpc_submit_start'],'qpc_end':call['qpc_sync_end'],
            'host_wall_ns':(call['qpc_sync_end']-call['qpc_submit_start'])*1e9/f,
            'submit_ns':(call['qpc_submit_end']-call['qpc_submit_start'])*1e9/f,
            'wait_ns':(call['qpc_sync_end']-call['qpc_sync_start'])*1e9/f,
            'instrumentation_gap_ns':(call['qpc_sync_start']-call['qpc_submit_end'])*1e9/f})
    return {'schema':'graph-submit-normalized/v1','header':header,'setup':setup,'footer':footer,
        'records':records,'runs':runs,'qpc_frequency':f,'loaded_modules_before':setup.get('loaded_modules_before',[]),
        'loaded_modules_after':footer.get('loaded_modules_after',[]),'source_raw':before}


def audit_raw(document,config,freeze):
    issues=[]
    def check(condition,message):
        if not condition:issues.append(message)
    header,setup,footer=document['header'],document['setup'],document['footer']
    try:process_origin(document,config)
    except (IdentityError,KeyError,TypeError) as exc:issues.append('native_process_origin_invalid:'+str(exc))
    protocol=protocol_document();validate_protocol(protocol)
    check(config in protocol['configs'],'config_not_frozen')
    check(header.get('protocol_sha256')==ref(PROBE/'protocol.json')['sha256'],'probe_protocol_sha_mismatch')
    check(header.get('probe_cuda_events') is False and header.get('trace_clock_subtraction_allowed') is False,'event_or_clock_policy_mismatch')
    check(setup.get('config')==config['id'] and setup.get('elements')==config['elements'],'graph_config_shape_mismatch')
    check(setup.get('requested_nodes')==config['nodes'] and setup.get('actual_ggml_nodes')==config['nodes'],'actual_GGML_node_count_mismatch')
    check(setup.get('tensor_payload_bytes')==config['elements']*4,'tensor_bytes_mismatch')
    check(setup.get('allocated_payload_bytes')==(config['nodes']+1)*config['elements']*4,'graph_payload_bytes_mismatch')
    check(setup.get('logical_read_bytes')==setup.get('logical_write_bytes')==config['nodes']*config['elements']*4,'logical_graph_IO_mismatch')
    check(type(setup.get('allocated_buffer_bytes')) is int and setup['allocated_buffer_bytes']>=setup.get('allocated_payload_bytes',0),'buffer_capacity_mismatch')
    expected_env=dict(protocol['runtime']['environment']);expected_env.update({k:None for k in protocol['runtime']['extra_clear_environment']})
    check(setup.get('environment')==expected_env,'native_environment_mismatch')
    scheduling=setup.get('scheduling',{})
    check(scheduling.get('CPU_worker_pool_created') is False and scheduling.get('process_priority_class')==32 and scheduling.get('caller_thread_priority')==0,'unexpected_host_scheduling_policy')
    check(type(scheduling.get('process_affinity')) is int and scheduling['process_affinity']>0,'actual_affinity_missing')
    hardware=setup.get('hardware_actual',{});expected_gpu=protocol['runtime']['gpu_expected']
    check(hardware.get('uuid')==expected_gpu['uuid'] and hardware.get('name')==expected_gpu['name'] and (hardware.get('cc_major'),hardware.get('cc_minor'))==(12,0),'actual_GPU_identity_mismatch')
    for key in ('SMs','total_memory_bytes','L2_bytes'):check(type(hardware.get(key)) is int and hardware[key]>0,'actual_GPU_property_missing:'+key)
    graph=setup.get('graph_nodes',[])
    check(len(graph)==config['nodes'],'graph_nodes_incomplete')
    for i,node in enumerate(graph):
        check(node.get('index')==i and node.get('name')=='scale_'+str(i+1),'graph_node_identity_mismatch')
        check(node.get('src0')==('graph_input' if i==0 else 'scale_'+str(i)),'graph_dependency_mismatch')
        check(node.get('operator')=='SCALE' and node.get('dtype')=='F32' and node.get('scale')==.5 and node.get('bias')==0,'graph_operator_mismatch')
        check(node.get('ne')==[config['elements'],1,1,1] and node.get('nb')==[4,config['elements']*4,config['elements']*4,config['elements']*4],'graph_tensor_layout_mismatch')
    calls=[r for r in document['records'] if r['record']=='graph_call'];validations=[r for r in document['records'] if r['record']=='validation'];intermediate=[r for r in document['records'] if r['record']=='intermediate_validation']
    expected=[('first',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]
    check([(r['phase'],r['index']) for r in document['runs']]==expected,'full36calls_order_or_count_mismatch')
    check(len(calls)==len(validations)==36,'raw_call_validation_counts')
    last_completed_qpc=0
    for c,v,r in zip(calls,validations,document['runs']):
        check(c['label']==v['label']==f"graph_submit/{config['id']}/{r['phase']}/{r['index']}",'semantic_label_mismatch')
        check(v.get('stage')==config['nodes'],'final_validation_stage_mismatch')
        check(c.get('qpc_outer_push_start',0)>=last_completed_qpc,'cross_call_QPC_order')
        check(c.get('graph_status')==0 and c.get('observed_device_kernel_count') is None and c.get('observed_cuda_graph_launch_count') is None,'untraced_count_or_call_status_bad')
        keys=['qpc_outer_push_start','qpc_outer_push_end','qpc_submit_push_start','qpc_submit_push_end','qpc_submit_start','qpc_submit_end','qpc_submit_pop_start','qpc_submit_pop_end','qpc_sync_push_start','qpc_sync_push_end','qpc_sync_start','qpc_sync_end','qpc_sync_pop_start','qpc_sync_pop_end','qpc_outer_pop_start','qpc_outer_pop_end']
        ticks=[c.get(k) for k in keys]
        check(all(type(t) is int and t>0 for t in ticks) and ticks==sorted(ticks),'graph_QPC_boundary_order')
        check(type(v.get('qpc_validation_start')) is int and type(v.get('qpc_validation_end')) is int and c['qpc_outer_pop_end']<=v['qpc_validation_start']<=v['qpc_validation_end'],'validation_outside_graph_timer')
        last_completed_qpc=v['qpc_validation_end']
        for row in [x for x in intermediate if x['label']==c['label']]:
            check(type(row.get('qpc_start')) is int and type(row.get('qpc_end')) is int and last_completed_qpc<=row['qpc_start']<=row['qpc_end'],'intermediate_validation_time_order')
            last_completed_qpc=row['qpc_end']
    for v in [*validations,*intermediate]:
        result=v.get('result',{})
        check(result.get('checked')==config['elements'] and result.get('mismatches')==0 and result.get('nonfinite')==0 and result.get('first_bad_index')==-1 and result.get('max_abs_error')==0 and result.get('bitwise_pass') is True,'full_bitwise_numerical_validation_failed')
    expected_intermediate={(f"graph_submit/{config['id']}/{phase}/{index}",stage) for phase,index in (('first',0),('formal',29)) for stage in range(1,config['nodes'])}
    check(len(intermediate)==len(expected_intermediate) and {(r.get('label'),r.get('stage')) for r in intermediate}==expected_intermediate,'first_last_intermediate_validation_incomplete')
    check(footer.get('status')=='complete' and footer.get('math_pass') is True and footer.get('graph_calls')==36 and footer.get('modules_stable') is True,'footer_not_successful')
    check(document['loaded_modules_before']==document['loaded_modules_after'] and bool(document['loaded_modules_before']),'loaded_modules_changed_or_missing')
    return {'valid_raw':not issues,'issues':sorted(set(issues)),'graph_calls':len(calls),'calibration_eligible':False,
        'numeric_proof_scope':'Counts and result summaries from frozen independent closed-form C++ reference; raw tensor bit patterns not retained.'}


def raw_signature(document,config,freeze=None):
    setup=document['setup']
    scheduling=dict(setup.get('scheduling',{}));scheduling.pop('caller_thread_id',None)
    observer_paths={str(Path(r['path']).resolve()).casefold():r for r in (freeze or {}).get('tool_files',[])}
    runtime_modules=[]
    for item in document['loaded_modules_before']:
        observer=observer_paths.get(str(Path(item['path']).resolve()).casefold())
        if observer is not None:
            if observer['sha256']!=item['sha256']:raise IdentityError('captured observer module hash differs from frozen tool inventory')
        else:runtime_modules.append(item)
    return {'config':config,'environment':setup.get('environment'),'graph_nodes':setup.get('graph_nodes'),
        'hardware_actual':setup.get('hardware_actual'),'modules':runtime_modules,'scheduling':scheduling,
        'probe_cuda_events':False,'source_reference':'binary-exact half-per-stage closed-form, locked source'}


def expected_labels(document,config):
    return {r['nvtx_label']:(r['phase'],r['index']) for r in document['runs']}
