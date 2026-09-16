"""Strict R20 adapter with original R18 trace attribution functions preserved.
R21 adds separate settle evidence without changing the R20 recorded-call attribution algorithm.
"""
from __future__ import annotations
from contextlib import closing
from collections import Counter,defaultdict
from pathlib import Path
import sqlite3,json,hashlib,statistics,math,re
PROCESS_MASK=0xFFFFFFFFFF000000
def ref(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}

def quantile(values,q):
    values=sorted(values)
    if not values or not 0<=q<=1:raise ValueError('invalid quantile')
    i=(len(values)-1)*q;a=int(i);return values[a]+(values[min(a+1,len(values)-1)]-values[a])*(i-a)

def distribution(values):
    values=list(values)
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid positive samples')
    lo,hi=quantile(values,.1),quantile(values,.9)
    return {'count':len(values),'minimum_ns':min(values),'maximum_ns':max(values),'median_ns':statistics.median(values),'p10_ns':lo,'p90_ns':hi,'p90_div_p10':hi/lo}

def union_ns(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if type(start) is not int or type(end) is not int or end<start:raise ValueError('invalid interval')
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return sum(b-a for a,b in merged)

def expected_calls():return [('first',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]

REQUIRED={
 'NVTX_EVENTS':{'start','end','globalTid','text'},
 'CUPTI_ACTIVITY_KIND_RUNTIME':{'start','end','globalTid','correlationId','nameId','returnValue'},
 'CUPTI_ACTIVITY_KIND_KERNEL':{'start','end','globalPid','correlationId','deviceId','streamId','demangledName','shortName','gridX','gridY','gridZ','blockX','blockY','blockZ','staticSharedMemory','dynamicSharedMemory'},
 'StringIds':{'id','value'},
}

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

role=kernel_role
validate_chain=validate_kernel_chain

def correlate(raw,app,config):
    tables=raw['tables'];markers=tables['NVTX_EVENTS'];kernels=tables['CUPTI_ACTIVITY_KIND_KERNEL']
    apis=[{**r,'evidence_table':table} for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER') for r in tables.get(table,[])]
    wanted={r['label']:(r['label'].rsplit('/',2)[1],int(r['label'].rsplit('/',1)[1])) for r in app['calls']}
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
        'unmapped_kernels_verbatim':[k for k in kernels if k['evidence_rowid'] not in used],
        'outside_recorded_calls_policy':'Unmapped events including settle work are retained separately and never included in 36 mapped calls or formal30 estimates.',
        'all_diagnostics_verbatim':diagnostics,'warning_diagnostics_verbatim':warnings,
        'native_process_binding':{'native_header_pid':native_pid,'native_caller_thread_id':native_tid,
            'trace_global_pid':next(iter(actual_namespaces)),'trace_decoded_pid':(next(iter(actual_namespaces))>>24)&0xFFFFFF,
            'verified_actual_identity_match':True},
        'native_source_equivalence_proven':False,'calibration_eligible':False,'direct_untraced_kernel_count':None}

def analyze_trace(path, raw_document, config):
    if 'records' in raw_document:
        rows=raw_document['records']
        for kind in ('header','setup','footer'):
            if len([x for x in rows if x.get('record')==kind])!=1:raise ValueError('unique raw '+kind+' required')
        doc={k:next(x for x in rows if x.get('record')==k) for k in ('header','setup','footer')}
        doc['calls']=[x for x in rows if x.get('record')=='graph_call']
    else:doc=raw_document
    wanted=[f"graph_submit/{config['id']}/{p}/{i}" for p,i in expected_calls()]
    if doc['setup'].get('config')!=config['id'] or [x.get('label') for x in doc['calls']]!=wanted:
        raise ValueError('config and ordered full36 labels must agree')
    if doc['footer'].get('graph_calls') not in (36,None):raise ValueError('call-count footer mismatch')
    trace=read_trace(path)
    result=correlate(trace,doc,config)
    if 'settle_metadata' in doc:
        settle=doc['settle_metadata'];ms=settle['condition_settle_ms']
        label=f"graph_submit/{config['id']}/settle/{ms}ms"
        markers=[r for r in trace['tables']['NVTX_EVENTS'] if r.get('resolved_text')==label]
        if len(markers)!=1:raise ValueError('unique settle NVTX marker required')
        marker=markers[0];first=result['calls'][0]['nvtx_verbatim'];warmup=result['calls'][1]['nvtx_verbatim']
        begin,end,tid=marker.get('start'),marker.get('end'),marker.get('globalTid')
        if type(begin) is not int or type(end) is not int or tid!=first['globalTid'] or not first['end']<=begin<=end<=warmup['start']:
            raise ValueError('settle trace process/time placement outside recorded calls')
        if end-begin<ms*1_000_000:raise ValueError('settle trace shorter than requested work window')
        apis=[{**r,'evidence_table':table} for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER') for r in trace['tables'].get(table,[]) if r['globalTid']==tid and begin<=r['start'] and r['end']<=end]
        kernels=[k for k in trace['tables']['CUPTI_ACTIVITY_KIND_KERNEL'] if k['globalPid']==(tid&PROCESS_MASK) and begin<=k['start'] and k['end']<=end]
        if ms>0 and not kernels:raise ValueError('settle trace lacks actual GPU work')
        mapped={pair['kernel']['evidence_rowid'] for call in result['calls'] for pair in call['kernel_launch_pairs']}
        if any(k['evidence_rowid'] in mapped for k in kernels):raise ValueError('settle kernel assigned to recorded call')
        result['settle_work']={'native_metadata':settle,'nvtx_verbatim':marker,'api_events_verbatim':apis,'kernels_verbatim':kernels,'observed_kernel_count':len(kernels),'estimator_time_used':False,'clock_policy':'Placement and duration checked independently inside native QPC and Nsight domains; no cross-domain subtraction.'}

    result['formal_denominator']=len([x for x in result['calls'] if x['phase']=='formal'])
    result['formal_30_complete']=result['formal_denominator']==30
    result['qpc_ns_subtraction_performed']=False
    return result
