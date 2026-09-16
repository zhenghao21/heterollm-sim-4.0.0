"""Read-only actual-schema extraction and frozen matrix quality diagnostics."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sqlite3
from contextlib import closing
from common import HERE,PROBE,PROCESS_MASK,load,write_new,ref,module,verify_freeze,distribution,union_ns,expected_calls,utc,clock_readback_gate,verify_clock_receipt
from quality import quality

REQUIRED={
 'NVTX_EVENTS':{'start','end','globalTid','text'},
 'CUPTI_ACTIVITY_KIND_RUNTIME':{'start','end','globalTid','correlationId','nameId','returnValue'},
 'CUPTI_ACTIVITY_KIND_KERNEL':{'start','end','globalPid','correlationId','deviceId','streamId','demangledName','shortName','gridX','gridY','gridZ','blockX','blockY','blockZ','staticSharedMemory','dynamicSharedMemory'},
 'StringIds':{'id','value'},
}
def role(kernel):
    import probe_adapter
    return probe_adapter.kernel_role(kernel)


def validate_chain(kernels,config):
    import probe_adapter
    return probe_adapter.validate_kernel_chain(kernels,config)


def read_trace(path):
    path=Path(path).resolve(strict=True);before=ref(path);tables={};schema={}
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        names={r['name'] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name,required in REQUIRED.items():
            actual={r['name'] for r in db.execute('PRAGMA table_info("'+name+'")')}
            if not required.issubset(actual):raise ValueError('unsupported actual SQLite schema: '+name)
        for name in sorted(names):
            if name.startswith(('CUPTI_ACTIVITY_KIND_','NVTX','DIAGNOSTIC','PROFILER_OVERHEAD','TARGET_INFO_','ENUM_DIAGNOSTIC')) or name in ('StringIds','ENUM_NSYS_EVENT_CLASS'):
                schema[name]=[r['name'] for r in db.execute('PRAGMA table_info("'+name+'")')]
                records=[dict(r) for r in db.execute('SELECT rowid AS evidence_rowid,* FROM "'+name+'"')]
                for r in records:
                    for key,value in list(r.items()):
                        if isinstance(value,bytes):r[key]={'binary_hex':value.hex()}
                tables[name]=records
    if ref(path)!=before:raise ValueError('SQLite changed during read')
    strings={r['id']:r['value'] for r in tables['StringIds']}
    for r in tables['NVTX_EVENTS']:r['resolved_text']=strings.get(r.get('textId'),r.get('text'))
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        for r in tables.get(table,[]):r['name_text']=strings.get(r['nameId'])
    for r in tables['CUPTI_ACTIVITY_KIND_KERNEL']:
        r['demangled_name_text']=strings.get(r['demangledName'],'');r['short_name_text']=strings.get(r['shortName'],'')
    return {'schema':'graph-matrix-raw-nsys/v1','source_sqlite':before,'actual_schema':schema,'tables':tables,'time_unit':'ns'}


def api_base(name):
    return re.sub(r'_v[0-9]+$','',name or '')


def mutually_exclusive_partition(begin,end,kernels,apis):
    categories={'GPU':[(k['start'],k['end']) for k in kernels],
        'host_launch':[(a['start'],a['end']) for a in apis if 'Launch' in api_base(a.get('name_text'))],
        'host_sync':[(a['start'],a['end']) for a in apis if 'Synchronize' in api_base(a.get('name_text'))],
        'host_other':[(a['start'],a['end']) for a in apis]}
    points=sorted({begin,end,*[v for spans in categories.values() for pair in spans for v in pair if begin<=v<=end]})
    result={name:0 for name in (*categories,'unattributed')}
    for lo,hi in zip(points,points[1:]):
        mid=(lo+hi)/2;name=next((name for name,spans in categories.items() if any(a<=mid<b for a,b in spans)),'unattributed')
        result[name]+=hi-lo
    if sum(result.values())!=end-begin:raise ValueError('per-call union partition did not close')
    return result


def correlate(raw,app,config):
    import probe_adapter
    tables=raw['tables'];markers=tables['NVTX_EVENTS'];kernels=tables['CUPTI_ACTIVITY_KIND_KERNEL']
    apis=[{**r,'evidence_table':table} for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER') for r in tables.get(table,[])]
    wanted=probe_adapter.expected_labels(app,config)
    selected=[m for m in markers if m.get('resolved_text') in wanted]
    native_pid=app.get('header',{}).get('pid')
    native_tid=app.get('setup',{}).get('scheduling',{}).get('caller_thread_id')
    if type(native_pid) is not int or not 0<native_pid<(1<<24) or type(native_tid) is not int or not 0<native_tid<(1<<24):
        raise ValueError('actual native PID/caller TID missing or outside reviewed Nsight encoding')
    # NVIDIA Nsight Systems Analysis Guide: globalTid encodes 24-bit thread ID,
    # 24-bit process ID, then 16-bit VM/hardware namespace. Do not compare the
    # whole globalPid to pid<<24: live exports contain a nonzero upper namespace.
    # https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#global-process-and-thread-ids
    actual_namespaces=set()
    for marker in selected:
        value=marker.get('globalTid')
        if type(value) is not int or value<=0 or ((value>>24)&0xFFFFFF)!=native_pid or (value&0xFFFFFF)!=native_tid:
            raise ValueError('trace target process/thread does not match actual native header')
        actual_namespaces.add(value&PROCESS_MASK)
    if len(actual_namespaces)!=1:raise ValueError('trace spans multiple or missing process/VM namespaces')
    if len(selected)!=36 or len(wanted)!=36 or Counter(m['resolved_text'] for m in selected)!=Counter(wanted.keys()):
        raise ValueError('36 full graph semantic labels not uniquely captured')
    used=set();calls=[]
    for marker in sorted(selected,key=lambda m:m['start']):
        begin,end,tid=marker['start'],marker['end'],marker['globalTid']
        if type(begin) is not int or type(end) is not int or begin>=end:raise ValueError('invalid full graph NVTX interval')
        owned=[a for a in apis if a['globalTid']==tid and begin<=a['start'] and a['end']<=end]
        issues=[]
        for a in owned:
            if a.get('returnValue')!=0 and not(api_base(a.get('name_text'))=='cudaEventQuery' and a.get('returnValue')==600):
                issues.append('CUDA_runtime_or_driver_API_failed')
        direct_launch_names={'cudaLaunchKernel','cudaLaunchKernelExC','cuLaunchKernel','cuLaunchKernelEx'}
        graph_launch_names={'cudaGraphLaunch','cuGraphLaunch'}
        launches=[a for a in owned if api_base(a.get('name_text')) in direct_launch_names|graph_launch_names]
        mapped=[];pairs=[];per_call_used=set()
        for api in launches:
            found=[k for k in kernels if k['globalPid']==(tid&PROCESS_MASK) and k['correlationId']==api['correlationId']]
            # A runtime wrapper and its nested driver launch can refer to the same
            # captured child. Count the device event once; preserve both APIs.
            found=[k for k in found if k['evidence_rowid'] not in per_call_used]
            if not found:continue
            if api_base(api.get('name_text')) in direct_launch_names and len(found)!=1:
                issues.append('ordinary_launch_not_uniquely_correlated')
            for kernel in found:
                if kernel['evidence_rowid'] in used:raise ValueError('kernel assigned to multiple graph calls')
                used.add(kernel['evidence_rowid']);per_call_used.add(kernel['evidence_rowid']);mapped.append(kernel)
                if not begin<=kernel['start']<kernel['end']<=end:issues.append('kernel_outside_full_graph_nvtx')
                pairs.append({'api':api,'kernel':kernel,'role':role(kernel),'correlation_method':'same_process_launch_correlation',
                    'kernel_within_nvtx':begin<=kernel['start'] and kernel['end']<=end})
        inside=[k for k in kernels if k['globalPid']==(tid&PROCESS_MASK) and begin<=k['start'] and k['end']<=end]
        unmapped=[k for k in inside if k['evidence_rowid'] not in per_call_used]
        if unmapped:issues.append('captured_graph_child_launch_correlation_unresolved')
        if not launches:issues.append('captured_submission_launch_missing')
        chain=validate_chain(mapped,config);issues.extend(chain['issues'])
        devices={k['deviceId'] for k in mapped};external=[]
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
            for event in tables.get(table,[]):
                if table=='CUPTI_ACTIVITY_KIND_KERNEL' and event['evidence_rowid'] in per_call_used:continue
                if event.get('deviceId') in devices and event['start']<end and event['end']>begin:
                    external.append({'table':table,'event':event})
        if external:issues.append('captured_unowned_GPU_activity_overlaps_graph')
        sync=[a for a in owned if api_base(a.get('name_text')) in {'cudaStreamSynchronize','cuStreamSynchronize','cudaDeviceSynchronize'}]
        graph_launch=[a for a in owned if api_base(a.get('name_text')) in graph_launch_names]
        capture=[a for a in owned if any(x in api_base(a.get('name_text')) for x in ('BeginCapture','EndCapture','GraphInstantiate','GraphExecUpdate'))]
        if not sync:issues.append('actual_final_GPU_synchronization_API_missing')
        phase,index=wanted[marker['resolved_text']];intervals=[(k['start'],k['end']) for k in mapped]
        span=max((k['end'] for k in mapped),default=0)-min((k['start'] for k in mapped),default=0)
        calls.append({'phase':phase,'index':index,'nvtx_verbatim':marker,'api_events_verbatim':owned,'kernel_launch_pairs':pairs,
            'unmapped_inside_kernels_verbatim':unmapped,'captured_external_gpu_overlap_verbatim':external,'chain':chain,'issues':sorted(set(issues)),
            'actual_kernel_count':len(mapped),'planned_GGML_nodes':config['nodes'],'graph_launch_API_count':len(graph_launch),
            'graph_capture_update_API_count':len(capture),'actual_sync_API_count':len(sync),
            'observed_submission_mode':'graph_capture_or_update' if capture else 'graph_launch' if graph_launch else 'ordinary_kernel_launch',
            'kernel_union_ns':union_ns(intervals),'kernel_span_ns':span,'inter_kernel_gap_ns':span-union_ns(intervals),
            'launch_api_union_ns':union_ns((a['start'],a['end']) for a in launches),
            'host_runtime_api_union_ns':union_ns((a['start'],a['end']) for a in owned if a['evidence_table']=='CUPTI_ACTIVITY_KIND_RUNTIME'),
            'host_driver_api_union_ns':union_ns((a['start'],a['end']) for a in owned if a['evidence_table']=='CUPTI_ACTIVITY_KIND_DRIVER'),
            'host_sync_api_union_ns':union_ns((a['start'],a['end']) for a in sync),
            'nvtx_full_call_duration_ns':end-begin,'exclusive_partition_ns':mutually_exclusive_partition(begin,end,mapped,owned),
            'clock_policy':'All device/API partition arithmetic uses Nsight timestamps only; never subtract QPC from Nsight.'})
    if [(c['phase'],c['index']) for c in calls]!=expected_calls():raise ValueError('graph call order/count differs')
    severities={r['id']:r.get('label',r.get('name')) for r in tables.get('ENUM_DIAGNOSTIC_SEVERITY_LEVEL',[])}
    diagnostics=tables.get('DIAGNOSTIC_EVENT',[]);warnings=[d for d in diagnostics if severities.get(d.get('severity')) not in ('Info','Verbose')]
    formal=[c['kernel_union_ns'] for c in calls if c['phase']=='formal'];complete=all(not c['issues'] for c in calls)
    return {'schema':'graph-matrix-correlated-profile/v1','config':config,'calls':calls,'all36_chain_complete':complete,
        'source_path_matches_observed_family':all(c['chain']['source_path_matches_observed_family'] for c in calls),
        'formal_kernel_union':distribution(formal) if formal and all(v>0 for v in formal) else {},
        'actual_kernels_mapped':len(used),'unmapped_kernel_rowids':[k['evidence_rowid'] for k in kernels if k['evidence_rowid'] not in used],
        'all_diagnostics_verbatim':diagnostics,'warning_diagnostics_verbatim':warnings,
        'native_process_binding':{'native_header_pid':native_pid,'native_caller_thread_id':native_tid,
            'trace_global_pid':next(iter(actual_namespaces)),'trace_decoded_pid':(next(iter(actual_namespaces))>>24)&0xFFFFFF,
            'verified_actual_identity_match':True},
        'native_source_equivalence_proven':False,'calibration_eligible':False,'direct_untraced_kernel_count':None}


def numeric_document(path,config,freeze):
    before=ref(path)
    import probe_adapter
    doc=probe_adapter.read_raw(path)
    audit=probe_adapter.audit_raw(doc,config,freeze)
    identity={r['path'].casefold():r for r in freeze['probe_files']}
    unbound=[];critical={Path(r['path']).name.casefold() for r in freeze['probe_files'] if Path(r['path']).name.casefold() in {'ggml-base.dll','ggml.dll','ggml-cpu.dll','ggml-cuda.dll','graph-submit-probe.exe'}};seen=set()
    for item in doc.get('loaded_modules_before',[]):
        expected=identity.get(item['path'].casefold());name=Path(item['path']).name.casefold();seen.add(name)
        if name in critical and expected is None:audit['issues'].append('critical_loaded_module_unbound:'+name);audit['valid_raw']=False
        if expected and (item['sha256']!=expected['sha256'] or ('bytes' in item and item['bytes']!=expected['bytes'])):audit['issues'].append('loaded_module_identity_mismatch');audit['valid_raw']=False
        if expected is None:unbound.append(item)
    if not critical.issubset(seen):audit['issues'].append('critical_loaded_module_missing');audit['valid_raw']=False
    if ref(path)!=before:raise ValueError('application raw changed during numeric audit')
    signature=probe_adapter.raw_signature(doc,config,freeze)
    return doc,{'source':before,'audit':audit,'formal_host':distribution(r['host_wall_ns'] for r in doc['runs'] if r['phase']=='formal'),
        'signature':signature,'unbound_observed_cuda_or_os_runtime_modules':unbound,'QPC_absolute_boundaries_preserved_in_source':True}


def stage_complete(path,expected_freeze_sha256):
    """Bind each consumed artifact to the exact stage, launch, and actual process."""
    import probe_adapter
    path=Path(path).resolve(strict=True);protocol=load(HERE/'protocol.json');r=load(path/'complete.json')
    if path.name not in ('direct','profile','export') or re.fullmatch(r'pair_0[123]',path.parent.name) is None:
        raise ValueError('invalid completed stage location')
    stage={'config_id':path.parent.parent.name,'pair':int(path.parent.name[-2:])-1,'mode':path.name}
    expected_path=(HERE/'runs'/stage['config_id']/path.parent.name/path.name).resolve()
    if path!=expected_path:raise ValueError('completed stage outside this frozen collection')
    selected=[c for c in protocol['configs'] if c['id']==stage['config_id']]
    if len(selected)!=1:raise ValueError('completed stage configuration not frozen')
    config=selected[0];expected_argv=probe_adapter.stage_command(stage,path,protocol)
    if r.get('schema')!='graph-collection-process/v1':raise ValueError('invalid process receipt schema')
    clock=r.get('clock_control_binding',{}).get('receipt_ref',{})
    binding=verify_clock_receipt(clock.get('path'),clock.get('sha256'),protocol['gpu_identity']['uuid'])
    anchor=load(HERE/'clock-control-binding.json')
    if anchor.get('receipt_ref')!=binding['receipt_ref']:raise ValueError('stage clock session differs from approved session')
    if r.get('external_approved_sha256')!=expected_freeze_sha256 or r.get('child_process_exited') is not True:
        raise ValueError('stage not bound to approved freeze and confirmed exit')
    after=r.get('freeze_after',{})
    if r.get('status')!='completed' or r.get('returncode')!=0 or after.get('passed') is not True or after.get('external_approved_sha256')!=expected_freeze_sha256:
        raise ValueError('stage did not complete with the approved identity')
    # Some logs are legitimately empty; an absent or nonlocal artifact is not.
    artifacts=r.get('artifacts')
    if not isinstance(artifacts,list) or not artifacts:raise ValueError('required process artifact set missing')
    refs={}
    for artifact in artifacts:
        if not isinstance(artifact,dict) or set(artifact)!={'path','sha256','bytes'}:
            raise ValueError('malformed artifact reference')
        item_path=artifact['path']
        if not isinstance(item_path,str) or not Path(item_path).is_absolute() or Path(item_path).resolve().parent!=path:
            raise ValueError('artifact is not inside its completed stage')
        if not isinstance(artifact['sha256'],str) or re.fullmatch('[a-f0-9]{64}',artifact['sha256']) is None or type(artifact['bytes']) is not int or artifact['bytes']<0:
            raise ValueError('missing artifact hash or byte identity')
        name=Path(item_path).name
        if name.casefold() in refs or name=='complete.json':raise ValueError('duplicate or recursive artifact reference')
        refs[name.casefold()]=artifact
        # The supervisor can append its own redirected logs after its receipt.
        # They are explicitly not consumed as evidence by this extractor.
        if name not in ('supervisor.stdout.txt','supervisor.stderr.txt') and ref(item_path)!=artifact:
            raise ValueError('completed process artifact changed')
    required={'spec.json','supervisor-started.json','launched.json','stdout.txt','stderr.txt'}
    required|={'trace.sqlite'} if path.name=='export' else {'microbench.json','telemetry.json'}
    if path.name=='profile':required.add('trace.nsys-rep')
    if not required.issubset(refs):raise ValueError('missing necessary stage artifacts: '+','.join(sorted(required-set(refs))))
    for name in required:
        if Path(refs[name]['path'])!=path/name:raise ValueError('artifact reference path differs from consumed path')
        if name not in ('stdout.txt','stderr.txt') and refs[name]['bytes']<=0:raise ValueError('empty required stage evidence: '+name)
    spec_ref=refs['spec.json']
    if r.get('spec_ref')!=spec_ref:raise ValueError('process receipt spec binding mismatch')
    spec=load(spec_ref['path']);launched=load(refs['launched.json']['path']);supervisor=load(refs['supervisor-started.json']['path'])
    if spec.get('schema')!='graph-collection-process-spec/v1' or spec.get('stage')!=stage or spec.get('directory')!=str(path):
        raise ValueError('captured spec stage identity mismatch')
    if spec.get('freeze')!=str(HERE/'freeze.json') or spec.get('expected_freeze_sha256')!=expected_freeze_sha256:
        raise ValueError('captured spec freeze mismatch')
    if spec.get('argv')!=expected_argv or r.get('argv')!=expected_argv or launched.get('argv')!=expected_argv:
        raise ValueError('launched argv differs from frozen stage command')
    if spec.get('cwd')!=protocol['probe_root'] or r.get('cwd')!=spec['cwd'] or spec.get('environment')!=protocol['environment_explicit']:
        raise ValueError('captured cwd/runtime environment mismatch')
    if spec.get('gpu_identity')!=protocol['gpu_identity'] or spec.get('telemetry') is not (path.name!='export'):
        raise ValueError('captured device or telemetry stage identity mismatch')
    if spec.get('clock_control_binding',{}).get('receipt_ref')!=binding['receipt_ref']:
        raise ValueError('spec clock binding differs')
    if supervisor.get('spec')!=spec_ref or supervisor.get('freeze_before',{}).get('passed') is not True or supervisor.get('freeze_before',{}).get('external_approved_sha256')!=expected_freeze_sha256:
        raise ValueError('supervisor startup identity incomplete')
    for key in ('process_pid','supervisor_pid','qpc_frequency','qpc_launch_start','qpc_process_complete'):
        if type(r.get(key)) is not int or r[key]<=0:raise ValueError('missing actual receipt identity: '+key)
    if launched.get('pid')!=r['process_pid'] or launched.get('supervisor_pid')!=r['supervisor_pid'] or supervisor.get('pid')!=r['supervisor_pid']:
        raise ValueError('receipt/actual launched PID mismatch')
    if type(launched.get('qpc_after_launch')) is not int or not r['qpc_launch_start']<=launched['qpc_after_launch']<=r['qpc_process_complete']:
        raise ValueError('launched QPC lifetime mismatch')
    if path.name=='export':
        input_ref=ref(path.parent/'profile/trace.nsys-rep')
        if spec.get('input_artifacts')!=[input_ref]:raise ValueError('export source report content not bound to launch spec')
        origin=None
    else:
        if spec.get('input_artifacts')!=[]:raise ValueError('unexpected native-stage input artifacts')
        doc=probe_adapter.read_raw(refs['microbench.json']['path'])
        if doc['source_raw']!=refs['microbench.json']:raise ValueError('raw changed after artifact verification')
        origin=probe_adapter.process_origin(doc,config)
        if origin['pair_id']!=stage['config_id']+'/'+path.parent.name or origin['argv']!=[protocol['executable']['path'],*probe_adapter.app_argv(config,path/'microbench.json')]:
            raise ValueError('actual native process belongs to a different pair/stage')
        if doc['qpc_frequency']!=r['qpc_frequency'] or not r['qpc_launch_start']<=origin['qpc_start']<origin['qpc_end']<=r['qpc_process_complete']:
            raise ValueError('native QPC lifecycle is outside its actual parent launch')
        if path.name=='direct' and origin['pid']!=r['process_pid']:raise ValueError('direct native PID differs from launched child')
        if path.name=='profile' and origin['pid']==r['process_pid']:raise ValueError('profile target PID incorrectly claims the Nsight wrapper')
    return {**r,'evidence_binding':{'stage':stage,'artifacts':refs,'process_origin':origin,
            'spec_ref':spec_ref,'verified_actual_argv_and_PID':True,'export_input_content_bound':path.name=='export'}}


def extract_pair(config,pair,output,freeze,expected_freeze_sha256):
    base=HERE/'runs'/config['id']/f'pair_{pair+1:02d}';result={'pair':pair,'numerics_all_rows':False,'trace_chain_complete':False,
        'source_path_matches_observed_family':False,'trace_warning_free':False,'issues':[]}
    try:
        stages={}
        for mode in ('profile','direct','export'):
            try:
                receipt=stage_complete(base/mode,expected_freeze_sha256)
                stages[mode]={'complete':True,'evidence_binding':receipt['evidence_binding']}
            except (ValueError,KeyError,TypeError,OSError) as exc:stages[mode]={'complete':False,'reason':str(exc)}
        result['stage_statuses']=stages
        if not all(v['complete'] for v in stages.values()):raise ValueError('one or more stages failed or incomplete; all statuses retained')
        app,profile=numeric_document(base/'profile/microbench.json',config,freeze)
        direct_app,direct=numeric_document(base/'direct/microbench.json',config,freeze)
        if profile['source']!=stages['profile']['evidence_binding']['artifacts']['microbench.json'] or direct['source']!=stages['direct']['evidence_binding']['artifacts']['microbench.json']:
            raise ValueError('consumed raw differs from completed-stage artifact')
        origins={mode:stages[mode]['evidence_binding']['process_origin'] for mode in ('profile','direct')}
        result.update(profile_raw=profile,direct_raw=direct,profile_host=profile['formal_host'],direct_host=direct['formal_host'],process_origins=origins)
        gates={mode:clock_readback_gate(document,load(base/mode/'telemetry.json')) for mode,document in (('profile',app),('direct',direct_app))}
        result['clock_domain_gates']=gates;result['clock_domain_validated']=all(g['passed'] for g in gates.values())
        result['numerics_all_rows']=profile['audit']['valid_raw'] and direct['audit']['valid_raw'] and profile['signature']==direct['signature']
        trace=read_trace(base/'export/trace.sqlite')
        if trace['source_sqlite']!=stages['export']['evidence_binding']['artifacts']['trace.sqlite']:raise ValueError('consumed SQLite differs from completed export')
        write_new(output/'raw_events.json',trace)
        mapped=correlate(trace,app,config);write_new(output/'mapped_calls.json',mapped)
        result.update(trace_chain_complete=mapped['all36_chain_complete'],source_path_matches_observed_family=mapped['source_path_matches_observed_family'],profile_kernel=mapped['formal_kernel_union'])
        messages=[]
        for mode in ('profile','export'):
            for name in ('stdout.txt','stderr.txt'):
                for line in (base/mode/name).read_text(encoding='utf-8',errors='replace').splitlines():
                    if re.search(r'\b(warning|error|fatal|unsupported|incompatible)\b',line,re.I):messages.append({'mode':mode,'file':name,'line':line})
        result['tool_warning_lines']=messages;result['trace_warning_free']=not messages and not mapped['warning_diagnostics_verbatim']
        result['trace_sources']={'raw':ref(output/'raw_events.json'),'mapped':ref(output/'mapped_calls.json'),'sqlite':trace['source_sqlite']}
    except (ValueError,KeyError,TypeError,OSError,sqlite3.Error) as exc:result['issues'].append(type(exc).__name__+': '+str(exc))
    write_new(output/'pair_summary.json',result);return result


def extract(output,expected_freeze_sha256):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(HERE.resolve()):raise ValueError('extraction output must be new and inside collection')
    before=verify_freeze(HERE/'freeze.json',expected_freeze_sha256);freeze=load(HERE/'freeze.json');protocol=load(HERE/'protocol.json');output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for config in protocol['configs']:
        pairs=[extract_pair(config,p,output/config['id']/f'pair_{p+1:02d}',freeze,expected_freeze_sha256) for p in range(3)]
        import probe_adapter
        origin_issues=probe_adapter.unique_process_origins(pairs)
        if origin_issues:
            for pair in pairs:pair['numerics_all_rows']=False;pair.setdefault('issues',[]).append('independent_process_origins_not_proven')
        row=quality(config,pairs);row['process_origin_issues']=origin_issues
        write_new(output/config['id']/'quality.json',row);rows.append(row)
    after=verify_freeze(HERE/'freeze.json',expected_freeze_sha256)
    result={'schema':'graph-matrix-collection-summary/v1','utc':utc(),'freeze_ref':ref(HERE/'freeze.json'),'verification_before':before,'verification_after':after,
        'configs_required':6,'configs_reported':len(rows),'pairs_required':18,'formal_profile_calls_required':540,'formal_direct_calls_required':540,
        'diagnostic_accepted_configs':sum(r['measurement_cost_eligible'] for r in rows),'diagnostic_rejected_or_missing_configs':sum(not r['measurement_cost_eligible'] for r in rows),
        'by_group':{group:{'required':sum(c.get('group','graph_shape_factorial')==group for c in protocol['configs']),'accepted':sum(r['config'].get('group','graph_shape_factorial')==group and r['measurement_cost_eligible'] for r in rows)} for group in ('graph_shape_factorial',)},
        'source_runtime_equivalence_proven':False,'no_coefficients_emitted':True,'no_fit':True,'no_llm_actuals_read':True,'configs':rows}
    write_new(output/'summary.json',result);return {k:v for k,v in result.items() if k not in ('configs',)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-freeze-sha256',required=True);a=p.parse_args();print(json.dumps(extract(a.output,a.expected_freeze_sha256),ensure_ascii=False))
