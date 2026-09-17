"""R28 prepare/check/run/resume/extend/evaluate. Only run/resume/extend may create GPU work.
Evidence is never overwritten. No simulator input, prediction or LLM error is read.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ctypes,hashlib,importlib.util,json,math,os,statistics,struct,subprocess,sys,threading,time
import psutil
P=Path(__file__).resolve().parent
ROOT=P.parents[5]
LOOP=P.parents[1]
spec=importlib.util.spec_from_file_location('r28_timing_builder',P/'build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
need=b.need
ref=b.ref
verify_ref=b.verify_ref
write_new=b.write_new
now=b.now
CONV='_Z13quantize_q8_1PKfPvxxxxxj5uint3'
MAIN='_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def protocol():return load(P/'protocol.json')
def integer(value,name,positive=False):
    need(type(value) is int and value>=(1 if positive else 0),'invalid integer '+name);return value
def positive(value,name):
    need(type(value) in (int,float) and math.isfinite(value) and value>0,'invalid positive '+name);return value
def med(values):return statistics.median(values)

def median_ratio_interval(values,coverage=.90):
    values=sorted(positive(v,'ratio') for v in values);n=len(values)
    need(n in (5,10),'only frozen5/10 independent blocks allowed')
    choices=[]
    for k in range(1,(n+1)//2+1):
        tail=sum(math.comb(n,j) for j in range(k))/2**n
        confidence=1-2*tail
        if confidence>=coverage:choices.append((k,confidence))
    need(choices,'no finite interval at this sample size')
    k,confidence=choices[-1]
    return {'median':med(values),'low':values[k-1],'high':values[-k],'n_independent_blocks_assumed':n,
        'order_k':k,'coverage_under_iid_continuous_block_ratio_model':confidence,
        'method':'central_binomial_order_statistics','independence_proven':False,
        'two_look_single_comparison_union_lower_bound':1-(2/2**5+2*(1+10)/2**10),
        'familywise_coverage_claimed':False,'within_process_calls_not_independent_blocks':True}

def pair_metrics(kernels):
    need(len(kernels)==2,'two PDL kernels required')
    conv,main=sorted(kernels,key=lambda x:0 if x['name']==CONV else 1)
    need(conv['name']==CONV and main['name']==MAIN,'wrong pair symbols')
    for k in kernels:integer(k['start'],'kernel start',True);integer(k['end'],'kernel end',True);need(k['end']>k['start'],'nonpositive kernel interval')
    d0=conv['end']-conv['start'];d1=main['end']-main['start'];overlap=max(0,min(conv['end'],main['end'])-max(conv['start'],main['start']))
    span=max(k['end'] for k in kernels)-min(k['start'] for k in kernels);union=d0+d1-overlap
    return {'observed_conversion_lifespan_ns':d0,'observed_main_in_pair_lifespan_ns':d1,'observed_pair_span_ns':span,'observed_pair_union_ns':union,'observed_interval_overlap_ns':overlap,'observed_inter_kernel_gap_ns':span-union}

def validate_capability(raw):
    need(raw.get('schema')=='r28-gpu-operator-raw/v2' and raw.get('status')=='capability_observed_under_API_contract','capability did not finish')
    need(raw.get('protocol_ref')==ref(P/'protocol.json'),'capability protocol differs')
    for key,value in protocol()['hardware'].items():need(raw.get('hardware',{}).get(key)==value,'capability actual GPU differs')
    need(raw.get('timed') is False and raw.get('GPU_kernel_executed') is False,'capability unexpectedly timed/launched kernels')
    need(raw.get('kind')=='capability_hes','single legal HES capability role required')
    need(raw.get('budget_exceeded') is False,'capability exceeded synchronized deadline')
    for key in ('contexts_before','contexts_after'):
        state=raw.get(key,{})
        need(type(state.get('current_context')) is int and state['current_context']==0 and type(state.get('primary_active')) is int and state['primary_active']==0,'capability context was active')
    need(type(raw.get('HES_enable_returncode')) is int and raw['HES_enable_returncode']==0,'HES API did not succeed')
    trace=raw['CUPTI_evidence'];need(trace.get('compile_API_version')==26 and trace.get('runtime_API_version')==26 and trace.get('callback_overflow') is False and trace.get('STATE')==[],'capability ABI/STATE invalid')
    calls=trace.get('calls',[])
    need(not any('latency' in c['name'].lower() for c in calls),'illegal latency API used in initialized process')
    enables=[c for c in calls if c['name']=='cuptiActivityEnableHWTrace(1)']
    need(len(enables)==1 and enables[0]['returncode']==0 and enables[0]['phase']=='enable_HES_before_context','HES enable timing/result missing')
    need(raw['driver_init']['qpc_end']<=enables[0]['qpc'],'HES called before driver initialization')
    actual=raw.get('actual_mode_evidence',{})
    need(actual.get('kind')=='HWTrace_SDK26_API_contract_only' and actual.get('direct_mode_readback_available') is False and actual.get('silent_fallback_independently_excluded') is False and actual.get('latency_timestamps_requested') is False and actual.get('software_fallback_requested') is False,'capability overstates actual-mode evidence')
    return True

def validate_numeric(raw):
    refs=raw['references'];need(set(refs)=={'packed_q4','input_f32','expected_q8','reference_f64','bound_f64'},'reference fixture incomplete')
    for r in refs.values():verify_ref(r)
    fixtures=protocol()['fixture_files'];mapping={'packed_q4':'weights.q4_k.bin','input_f32':'input.f32.bin','expected_q8':'expected.q8_1.bin','reference_f64':'reference.f64.bin','bound_f64':'bounds.f64.bin'}
    for name,filename in mapping.items():
        verify_ref(fixtures[filename]);need(refs[name]['sha256']==fixtures[filename]['sha256'] and refs[name]['bytes']==fixtures[filename]['bytes'],'timing fixture changed from independently qualified R29 bytes')
    expected_q8=Path(refs['expected_q8']['path']).read_bytes();need(len(expected_q8)==2304,'Q8 reference shape wrong')
    expected_data=Path(refs['reference_f64']['path']).read_bytes();bound_data=Path(refs['bound_f64']['path']).read_bytes()
    need(len(expected_data)==len(bound_data)==2048*8,'output reference dimensions wrong')
    expected=struct.unpack('<2048d',expected_data);bounds=struct.unpack('<2048d',bound_data)
    need(all(math.isfinite(e) and math.isfinite(t) and t>0 for e,t in zip(expected,bounds)),'invalid mathematical reference/bound')
    role=raw['state']['role'];slots=raw['rotation']['slots']
    for phase in ('correctness_before','correctness_after'):
        value=raw[phase];need(value.get('passed') is True and len(value.get('rows',[]))==slots,'numerical qualification incomplete')
        need({r['slot'] for r in value['rows']}==set(range(slots)),'numerical slots incomplete')
        for row in value['rows']:
            verify_ref(row['actual_ref']);data=Path(row['actual_ref']['path']).read_bytes()
            if role=='convert':need(data==expected_q8 and row.get('byte_mismatches')==0,'Q8 conversion differs')
            else:
                need(len(data)==2048*4 and row.get('values')==2048 and row.get('failed')==0,'output dimensions/results differ')
                actual=struct.unpack('<2048f',data);errors=[abs(a-e) for a,e in zip(actual,expected)]
                need(all(math.isfinite(a) and e<=t for a,e,t in zip(actual,errors,bounds)),'output violates predeclared bounds')
                need(math.isclose(max(errors),row['max_absolute_error'],rel_tol=1e-12,abs_tol=1e-15),'reported numeric error differs')
                need(math.isclose(max(e/t for e,t in zip(errors,bounds)),row['max_bound_ratio'],rel_tol=1e-12,abs_tol=1e-15),'reported bound ratio differs')
    return {k:refs[k]['sha256'] for k in refs}

def validate_activity(raw):
    data=raw['activity'];need(data.get('kernel_record_ABI')=='CUpti_ActivityKernel9','activity ABI not matched SDK26')
    need(type(data.get('dropped_records')) is int and data['dropped_records']==0 and data.get('buffer_overflow') is False and data.get('unknown_kinds')==[],'dropped/unknown activity records')
    need(data.get('raw_buffers'),'no raw activity buffers')
    for row in data['raw_buffers']:
        verify_ref(row['ref']);need(row['valid_bytes']==row['ref']['bytes'],'raw buffer length mismatch')
    links={}
    for row in data['external_links']:
        need(row['kind']==3,'external correlation kind differs')
        corr=integer(row['correlation'],'correlation',True);external=integer(row['external_id'],'external id',True)
        need(corr not in links or links[corr]==external,'ambiguous external association');links[corr]=external
    api={}
    for row in data['runtime_APIs']:
        corr=integer(row['correlation'],'API correlation',True)
        need(corr not in api and row['cbid']==430 and type(row['return_value']) is int and row['return_value']==0,'unexpected/failed/duplicated launch API')
        need(integer(row['start'],'API start',True)<=integer(row['end'],'API end',True),'API timestamps invalid');api[corr]=row
    role=raw['state']['role'];expected_symbols=([CONV,MAIN] if role=='pair' else [MAIN if role=='main' else CONV]);width=len(expected_symbols)
    all_ids=set(range(1,65))|set(range(1000000,1000032));by_id={i:[] for i in all_ids};seen=set()
    need(len(data['kernels'])==96*width and len(api)==96*width,'formal+priming launch denominator differs')
    for k in data['kernels']:
        corr=integer(k['correlation'],'kernel correlation',True)
        need(corr not in seen and corr in links and corr in api,'kernel launch lacks unique API/external chain');seen.add(corr)
        need(links[corr] in by_id,'unregistered measurement id')
        need(k['kind']==10 and k['name'] in expected_symbols,'serialized/wrong kernel activity')
        need(integer(k['start'],'kernel start',True)<integer(k['end'],'kernel end',True),'nonzero kernel timestamp required')
        need(k['graph_id']==0 and k['graph_node_id']==0 and k['dynamic_shared']==0,'unexpected graph/shared path')
        shape=([16,1,1],[256,1,1]) if k['name']==CONV else ([2048,1,1],[32,4,1])
        need(k['grid']==shape[0] and k['block']==shape[1],'kernel geometry differs')
        by_id[links[corr]].append(k)
    contexts={(k['context'],k['stream'],k['device']) for k in data['kernels']};need(len(contexts)==1,'unexpected context/stream/device changes')
    metrics=[];prior_end=None
    for ident in list(range(1000000,1000032))+list(range(1,65)):
        rows=by_id[ident];need(sorted(k['name'] for k in rows)==sorted(expected_symbols),'sample kernel role set differs')
        begin=min(k['start'] for k in rows);end=max(k['end'] for k in rows)
        need(prior_end is None or begin>=prior_end,'independent samples overlap; throughput substituted for latency');prior_end=end
        if ident>=1000000:continue
        if role=='pair':
            value=pair_metrics(rows)
            conversion=next(k for k in rows if k['name']==CONV);main=next(k for k in rows if k['name']==MAIN)
            value['host_late_submission_lower_bound_ns']=min(value['observed_inter_kernel_gap_ns'],max(0,api[main['correlation']]['start']-conversion['end']))
        else:value={'observed_main_lifespan_ns' if role=='main' else 'observed_conversion_lifespan_ns':rows[0]['end']-rows[0]['start']}
        value['sample']=ident-1;metrics.append(value)
    return metrics

def validate_raw(raw,state_id,mode,verify_files=True):
    p=protocol();state=next((x for x in p['states'] if x['id']==state_id),None)
    need(state is not None and mode in p['modes'],'unknown state/mode; M4 is held out')
    need(raw.get('schema')=='r28-gpu-operator-raw/v2' and raw.get('budget_exceeded') is False and raw.get('service_cost_qualification')=='unvalidated' and raw.get('status')=='formal_observed_pending_independent_validation' and raw.get('kind')=='formal' and raw.get('state')==state and raw.get('mode')==mode,'formal state/status differs')
    need(raw.get('timed') is True and raw.get('GPU_kernel_executed') is True and raw.get('performance_parameter_admitted') is False,'wrong result scope')
    need(raw.get('protocol_ref')==ref(P/'protocol.json'),'raw result bound to another protocol')
    for key,value in p['hardware'].items():need(raw['hardware'].get(key)==value,'actual GPU identity differs')
    for key in ('contexts_before','contexts_after_HES_before_work'):
        need(raw[key]['primary_active']==0 and raw[key]['current_context']==0,'context preceded mode selection')
    frequency=positive(raw.get('QPC_frequency'),'QPC frequency');start=integer(raw['formal_begin_qpc'],'formal begin',True);end=integer(raw['formal_end_qpc'],'formal end',True)
    need(start<end,'formal window invalid')
    samples=raw['samples'];need(len(samples)==64 and [s['sample'] for s in samples]==list(range(64)),'formal sample denominator differs')
    for s in samples:
        need(start<=s['begin_qpc']<=s['submit_begin_qpc']<=s['submit_end_qpc']<=s['wait_begin_qpc']<=s['end_qpc']<=end,'QPC order invalid')
        need(math.isclose(s['wall_ns'],(s['end_qpc']-s['begin_qpc'])*1e9/frequency,rel_tol=1e-12,abs_tol=1e-6),'wall time not raw QPC derived')
        positive(s['wall_ns'],'wall duration')
        if p['modes'][mode]['events']:positive(s['event_ns'],'event envelope')
        else:need(s['event_ns'] is None,'event inserted in non-event control')
    for key in ('modules_before','modules_after'):
        for r in raw[key]:
            if verify_files:verify_ref(r)
        need(raw['modules_before']==raw['modules_after'],'loaded module set changed')
        expected={m['ref']['path']:m for m in p['identities']['native_modules']}
        for extra in ('CUPTI','nvperf_host','nvperf_target'):expected[p['identities'][extra]['path']]={'ref':p['identities'][extra],'required':extra=='CUPTI' and p['modes'][mode]['activity']}
        observed={item['path']:item for item in raw[key]}
        need(len(observed)==len(raw[key]) and all(path in expected and expected[path]['ref']==item for path,item in observed.items()),'unknown/duplicate module identity')
        need(all(not item['required'] or path in observed for path,item in expected.items()),'required loaded module evidence missing')
        if not p['modes'][mode]['activity']:need(not any(p['identities'][name]['path'] in observed for name in ('CUPTI','nvperf_host','nvperf_target')),'profiler module in unobserved control')
    warm=raw['warmup'];need(warm['calls']>=32 and (warm['end_qpc']-warm['begin_qpc'])*1e9/frequency>=500000000,'warmup contract failed')
    need(raw['observer_priming']['calls']==32,'observer startup not primed')
    rot=raw['rotation'];integer(rot['slots'],'rotation slots',True);need(rot['slots']<=128 and rot['total_weight_bytes']<=1073741824,'rotation budget exceeded')
    if state['cache']=='rotation':need(rot['total_weight_bytes']>=4*rot['L2_bytes'] and rot['slots']>1,'rotation working set not above L2')
    else:need(rot['slots']==1,'warm state changed buffers')
    need(all(s['slot']==(raw['formal_base_slot']+s['sample'])%rot['slots'] for s in samples),'rotation sequence differs')
    fingerprints=validate_numeric(raw) if verify_files else None
    traced=p['modes'][mode]['activity']
    if traced:
        trace=raw['CUPTI_evidence'];need(trace.get('compile_API_version')==trace.get('runtime_API_version')==26 and trace.get('STATE')==[] and trace.get('callback_overflow') is False,'HES ABI/STATE invalid')
        enables=[c for c in trace['calls'] if c['name']=='cuptiActivityEnableHWTrace(1)']
        need(len(enables)==1 and enables[0]['returncode']==0 and enables[0]['phase']=='enable_HES_before_context' and raw['driver_init']['qpc_end']<=enables[0]['qpc']<warm['begin_qpc'],'HES enable position/result unproven')
        need(not any('latency' in c['name'] for c in trace['calls']),'latency timestamps changed during formal HES run')
        actual=raw['actual_mode_evidence'];need(actual.get('kind')=='HWTrace_SDK26_API_contract_only' and actual.get('direct_mode_readback_available') is False and actual.get('silent_fallback_independently_excluded') is False and actual.get('latency_timestamps_requested') is False and actual.get('software_fallback_requested') is False,'actual HES API-contract evidence not explicit')
        if verify_files:
            verify_ref(raw['mode_proof_ref']);proof=load(raw['mode_proof_ref']['path'])
            need(proof.get('schema')=='r28-hes-api-contract/v2','legal HES capability proof missing')
            verify_ref(proof['capability_ref']);validate_capability(load(proof['capability_ref']['path']))
        metrics=validate_activity(raw) if verify_files else []
    else:
        need(raw['activity'] is None and raw['CUPTI_evidence'] is None and raw['actual_mode_evidence']['kind']=='unobserved_control','unobserved control included profiling')
        metrics=[]
    return {'wall_ns':med(s['wall_ns'] for s in samples),'event_ns':med(s['event_ns'] for s in samples) if p['modes'][mode]['events'] else None,
        'kernel_metrics':metrics,'fixture_fingerprints':fingerprints,'sample_count':64,'state':state_id,'mode':mode}


def assess_blocks(rows,n):
    p=protocol();need(n in (5,10),'not frozen block count');reports=[]
    for state in p['states']:
        group=[r for r in rows if r['state']==state['id'] and r['block']<=n]
        keys={(r['block'],r['mode']) for r in group};expected={(block,mode) for block in range(1,n+1) for mode in p['modes']}
        report={'state':state['id'],'required_processes':n*4,'reported_processes':len(group),'status':'rejected','reasons':[],'extension_eligible':False,'observed_duration_points':[],'service_cost_qualification':'unvalidated'}
        if keys!=expected or len(group)!=len(expected) or any(r.get('status')!='valid' for r in group):
            report['reasons'].append('failed_missing_or_duplicate_process');reports.append(report);continue
        fingerprints=[r['summary'].get('fixture_fingerprints') for r in group]
        if any(not isinstance(x,dict) or not x for x in fingerprints) or any(x!=fingerprints[0] for x in fingerprints):
            report['reasons'].append('data_fixture_changed_or_unbound');reports.append(report);continue
        index={(r['block'],r['mode']):r for r in group};intervals=[]
        for numerator,denominator,metric in p['acceptance']['ratio_pairs']:
            ratios=[index[(block,numerator)]['summary'][metric]/index[(block,denominator)]['summary'][metric] for block in range(1,n+1)]
            value=median_ratio_interval(ratios);value.update({'comparison':numerator+'/'+denominator,'metric':metric,'block_ratios':ratios})
            intervals.append(value)
        inside=lambda x:.95<=x<=1.05
        point_ok=all(inside(x['median']) for x in intervals);interval_ok=all(inside(x['low']) and inside(x['high']) for x in intervals)
        report['observer_intervals']=intervals
        if not point_ok:report['reasons'].append('observed_median_perturbation_exceeds_5pct')
        elif not interval_ok:report['reasons'].append('perturbation_interval_inconclusive')
        main_metric={'pair':'observed_pair_span_ns','main':'observed_main_lifespan_ns','convert':'observed_conversion_lifespan_ns'}[state['role']]
        metric_names=[k for k in index[(1,'B')]['summary']['kernel_metrics'][0] if k!='sample']
        all_stable=True
        for metric in metric_names:
            block_medians=[med(s[metric] for s in index[(block,'B')]['summary']['kernel_metrics']) for block in range(1,n+1)]
            center=med(block_medians);spread=max(abs(x-center)/center for x in block_medians) if center>0 else (0 if all(x==0 for x in block_medians) else math.inf)
            # Zero/small overlap and gap are diagnostic; stability gate applies to
            # a positive observed elapsed duration, not near-zero residuals; no service-cost qualification follows.
            if metric==main_metric:all_stable=math.isfinite(spread) and spread<=.05
            report['observed_duration_points'].append({'metric':metric,'median_ns':center,'block_medians_ns':block_medians,'max_block_relative_deviation':spread if math.isfinite(spread) else None,'is_primary_observed_elapsed_duration':metric==main_metric,'service_cost_qualification':'unvalidated'})
        if not all_stable:report['reasons'].append('cross_block_elapsed_cost_variation_exceeds_5pct')
        report['extension_eligible']=p['sampling']['maximum_extensions']>0 and n==5 and point_ok and not interval_ok and all_stable
        if interval_ok and all_stable:report['status']='envelope_controls_passed_service_unvalidated'
        report['cost_model_parameters_emitted']=False;report['scope']={**p['development'],'cache_policy':state['cache'],'origin':state['origin'],'is_cross_model_or_M4_validation':False,'familywise_confidence_guarantee':False}
        reports.append(report)
    return {'schema':'r28-observed-timing-evaluation/v2','blocks':n,'states':reports,'envelope_controlled_states':sum(r['status']=='envelope_controls_passed_service_unvalidated' for r in reports),'service_cost_qualified_states':0,'service_cost_qualification':'unvalidated','required_states':5,
        'envelope_controls_passed_all_states':all(r['status']=='envelope_controls_passed_service_unvalidated' for r in reports),'formal_success':False,'simulator_coefficients_emitted':False,'held_out_M4':'unmeasured',
        'extension_eligible':n==5 and any(r['extension_eligible'] for r in reports) and all(r['status']=='envelope_controls_passed_service_unvalidated' or r['extension_eligible'] for r in reports)}


def prepared():
    manifest=b.verify_build();receipt=load(P/'host_test_receipt.json')
    need(receipt.get('returncode')==0 and type(receipt.get('returncode')) is int and receipt.get('GPU_executed') is False and receipt.get('inputs_unchanged') is True,'host qualification missing')
    need(receipt['manifest_ref']==ref(P/'build_manifest.json'),'host tests cover another build')
    for item in b.references(receipt):verify_ref(item)
    return manifest,{'build_ref':ref(P/'build_manifest.json'),'host_test_ref':ref(P/'host_test_receipt.json'),'protocol_ref':ref(P/'protocol.json')}

def qualification_module():
    return b.load(P/'qualification.py','r31_Q4_qualification_guard')


def process_conflicts(records,own_pid=None):
    # Pure helper retained for inherited host tests; runtime uses the frozen strict guard plus local exact executable paths.
    old=b.load(LOOP/'round_026/run_target_capture_ex.py','r26_idle_logic')
    records=[r for r in records if r.get('pid')!=own_pid]
    bad=old.process_conflicts(records);seen={r['pid'] for r in bad}
    terms=('freeze_candidate.py','verify_frozen_preflight.py','run_wrapper_capture.py','run_target_capture_ex.py','host_submission_probe','gpu_operator_timing','q4k_timing','qualification.py','run_probe.py')
    for row in records:
        command=' '.join(row.get('cmdline') or []).replace('\\','/').lower();name=(row.get('name') or '').lower()
        if row['pid'] not in seen and (name in ('gpu-operator-timing.exe','q4k-timing.exe','q4k-wrapper-qualification.exe') or name.startswith('q4k-target.') or (name.startswith('python') and any(t in command for t in terms))):bad.append(row)
    return bad


def assert_idle():qualification_module().idle()


def runtime_prerequisites():
    assert_idle()
    r27_path=LOOP/'round_027/run_candidate.py';r27=b.load(r27_path,'r27_natural_terminal_guard')
    record,r27_ref=r27.barrier();need(record['terminal_count']==262 and record.get('failures_preserved') is True,'R27 terminal barrier missing')
    qualified=qualification_module().verify_qualified();assert_idle()
    return {'R27_barrier_ref':r27_ref,'R27_driver_ref':ref(r27_path),**qualified}


def verify_review(commit):
    need(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'full reviewed commit SHA required')
    resolved=subprocess.run(['git','rev-parse','--verify',commit+'^{commit}'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    need(resolved==commit,'reviewed commit differs')
    for name in b.SOURCES:
        path=P/name;relative=path.relative_to(ROOT).as_posix()
        value=subprocess.run(['git','show',commit+':'+relative],cwd=ROOT,capture_output=True,check=True).stdout
        need(value==path.read_bytes(),'reviewed source bytes differ: '+relative)
    return commit

def execution_env():
    p=protocol();keep={'systemroot','windir','comspec','temp','tmp','localappdata','appdata','userprofile','homedrive','homepath','programdata','programfiles','programfiles(x86)'}
    env={k:v for k,v in os.environ.items() if k.lower() in keep};windows=os.environ.get('SystemRoot','C:/Windows');env['SystemRoot']=windows
    dirs=sorted({str(Path(v['ref']['path']).parent) for v in p['identities']['native_modules']})
    env['PATH']=os.pathsep.join(dirs+[str(Path(windows)/'System32'),windows])
    env['GPU_OPERATOR_TIMING_AUTHORIZED']='1';env['GGML_CUDA_DISABLE_GRAPHS']='1';env['GGML_CUDA_PDL']='1'
    return env

class NvmlMonitor:
    """CPU-side NVML observations; no clock changes, only actual device readbacks."""
    def __init__(self):self.rows=[];self.error=None;self.stop_event=threading.Event();self.thread=None;self.lib=None
    def start(self):
        try:
            identity=protocol()['identities']['NVML'];verify_ref(identity);self.lib=ctypes.WinDLL(identity['path']);lib=self.lib
            lib.nvmlInit_v2.restype=ctypes.c_int;need(lib.nvmlInit_v2()==0,'NVML init failed')
            lib.nvmlDeviceGetHandleByUUID.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)];lib.nvmlDeviceGetHandleByUUID.restype=ctypes.c_int
            uuid=protocol()['hardware']['uuid'];handle=ctypes.c_void_p();need(lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(handle))==0,'NVML hardware UUID unavailable')
            lib.nvmlDeviceGetClockInfo.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)];lib.nvmlDeviceGetClockInfo.restype=ctypes.c_int
            lib.nvmlDeviceGetTemperature.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint)];lib.nvmlDeviceGetTemperature.restype=ctypes.c_int
            def poll():
                while not self.stop_event.is_set():
                    row={'qpc_ns':time.perf_counter_ns(),'uuid':uuid}
                    for name,kind in (('SM_MHz',1),('memory_MHz',2)):
                        v=ctypes.c_uint();rc=lib.nvmlDeviceGetClockInfo(handle,kind,ctypes.byref(v));row[name]=v.value if rc==0 else None;row[name+'_returncode']=rc
                    v=ctypes.c_uint();rc=lib.nvmlDeviceGetTemperature(handle,0,ctypes.byref(v));row['temperature_C']=v.value if rc==0 else None;row['temperature_returncode']=rc
                    self.rows.append(row);self.stop_event.wait(.1)
            self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
        except Exception as e:self.error=type(e).__name__+': '+str(e)
    def finish(self):
        self.stop_event.set()
        if self.thread:self.thread.join(2)
        if self.lib:self.lib.nvmlShutdown()
        return {'schema':'r28-read-only-NVML/v1','error':self.error,'samples':self.rows,'period_seconds':.1,
            'clock_domain':'Windows QPC represented by Python perf_counter_ns','scope':'sampled readbacks and outer bracketing, not every-kernel instantaneous clocks','clock_normalization_applied':False}

def make_plan(n):
    p=protocol();need(n in (5,10),'only frozen block counts')
    orders=[['U','A','AB','B'],['A','B','U','AB'],['B','AB','A','U'],['AB','U','B','A']]
    plan=[]
    for block in range(1,n+1):
        for si,state in enumerate(p['states']):
            for mode in orders[(block-1+si)%4]:plan.append({'block':block,'state':state['id'],'mode':mode,'id':f'b{block:02}_{state["id"]}_{mode}'})
    return plan

def remaining(start):
    elapsed=(time.perf_counter_ns()-start['budget_begin_qpc_ns'])/1e9
    need(elapsed>=0,'monotonic clock epoch changed; cannot resume budget')
    return max(0,1200-elapsed)

def immutable_guard(start):
    manifest,prepared_refs=prepared();need(prepared_refs==start['prepared_refs'],'prepared artifact changed')
    verify_review(start['reviewed_commit']);need(runtime_prerequisites()==start['runtime_prerequisites'],'R27/new-shim prerequisite changed')
    return manifest

def wait_naturally(process,deadline_qpc_ns,directory,clock=time.perf_counter_ns):
    """No process is cancelled. Soft deadline only invalidates evidence/stops new work."""
    overdue=False;notifications=0
    while True:
        now_ns=clock()
        if now_ns>=deadline_qpc_ns and not overdue:
            overdue=True;notifications+=1
            write_new(directory/'soft_deadline_exceeded.json',{'schema':'r28-natural-wait-overdue/v1',
                'created_utc':now(),'observed_qpc_ns':now_ns,'deadline_qpc_ns':deadline_qpc_ns,
                'started_process_must_exit_naturally':True,'termination_requested':False,'evidence_eligible':False})
        try:
            rc=process.wait(timeout=1.0)
            ended=clock();overdue=overdue or ended>=deadline_qpc_ns
            if overdue and notifications==0:
                write_new(directory/'soft_deadline_exceeded.json',{'schema':'r28-natural-wait-overdue/v1',
                    'created_utc':now(),'observed_qpc_ns':ended,'deadline_qpc_ns':deadline_qpc_ns,
                    'started_process_must_exit_naturally':True,'termination_requested':False,'evidence_eligible':False})
            return {'returncode':rc,'natural_exit':True,'ended_qpc_ns':ended,'deadline_qpc_ns':deadline_qpc_ns,
                'soft_deadline_exceeded':overdue,'termination_requested':False}
        except subprocess.TimeoutExpired:
            continue


def run_process(directory,kind,start,proof=None,state=None,mode=None):
    manifest=immutable_guard(start);need(remaining(start)>0,'20-minute budget exhausted')
    total_deadline=start['budget_begin_qpc_ns']+1200*10**9
    deadline=min(total_deadline,time.perf_counter_ns()+protocol()['sampling']['process_soft_budget_seconds']*10**9)
    directory.mkdir(exist_ok=False);argv=[manifest['executable']['path'],'--protocol',str(P/'protocol.json'),'--kind',kind,'--output',str(directory),'--deadline-qpc-ns',str(deadline)]
    if state is not None:argv+=['--state',state,'--mode',mode,'--proof',str(proof)]
    spec={'schema':'r28-gpu-timing-process-start/v2','created_utc':now(),'argv':argv,'kind':kind,'state':state,'mode':mode,
        'remaining_budget_seconds':remaining(start),'deadline_qpc_ns':deadline,'deadline_policy':'stop new calls at synchronized boundaries; natural exit without cancellation','prepared_refs':start['prepared_refs']}
    write_new(directory/'start.json',spec)
    finish={'schema':'r28-gpu-timing-process-finish/v2','status':'rejected','returncode':None,'start_ref':ref(directory/'start.json'),
        'created_utc':None,'performance_parameter_admitted':False,'service_cost_qualification':'unvalidated','natural_exit':None,'termination_requested':False}
    monitor=NvmlMonitor();monitor.start()
    try:
        need(monitor.error is None,'actual NVML telemetry unavailable: '+str(monitor.error));assert_idle()
        need(time.perf_counter_ns()<deadline,'soft budget exhausted before launch; no new process')
        with (directory/'stdout.log').open('xb') as stdout,(directory/'stderr.log').open('xb') as stderr:
            process=subprocess.Popen(argv,env=execution_env(),cwd=P,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            finish.update(wait_naturally(process,deadline,directory))
        need(not finish['soft_deadline_exceeded'],'collector finished beyond soft budget; retained but not qualified')
        need(finish['returncode']==0,'collector returned failure '+str(finish['returncode']))
        raw=load(directory/'raw.json')
        if kind=='capability_hes':validate_capability(raw)
        else:finish['summary']=validate_raw(raw,state,mode)
        finish['status']='valid'
    except Exception as error:finish['execution_error']=type(error).__name__+': '+str(error)
    telemetry=monitor.finish();write_new(directory/'telemetry.json',telemetry);finish['telemetry_ref']=ref(directory/'telemetry.json')
    try:immutable_guard(start);finish['identity_unchanged']=True
    except Exception as error:finish['identity_unchanged']=False;finish['identity_error']=type(error).__name__+': '+str(error);finish['status']='rejected'
    for key,name in (('raw_ref','raw.json'),('stdout_ref','stdout.log'),('stderr_ref','stderr.log'),('soft_deadline_ref','soft_deadline_exceeded.json')):
        finish[key]=ref(directory/name) if (directory/name).is_file() else None
    if finish['raw_ref'] is not None:
        try:
            raw=load(finish['raw_ref']['path']);finish['collector_budget_exceeded']=raw.get('budget_exceeded')
            if raw.get('budget_exceeded') is True:finish['status']='rejected'
        except Exception as error:finish['raw_decode_error']=type(error).__name__+': '+str(error);finish['status']='rejected'
    finish['created_utc']=now();write_new(directory/'finish.json',finish)
    return finish

def ensure_proof(directory,start):
    destination=directory/'capabilities.json'
    if destination.exists():
        proof=load(destination);need(proof.get('schema')=='r28-hes-api-contract/v2','capability scheme differs')
        verify_ref(proof['capability_ref']);validate_capability(load(proof['capability_ref']['path']))
        return destination
    out=directory/'capability_hes'
    if (out/'finish.json').is_file():
        result=load(out/'finish.json')
        for item in b.references(result):verify_ref(item)
    else:
        need(not out.exists(),'interrupted capability cannot be overwritten or silently retried')
        result=run_process(out,'capability_hes',start)
    need(result['status']=='valid' and result.get('identity_unchanged') is True,'SDK26 HES capability rejected; formal GPU observations blocked')
    proof={'schema':'r28-hes-api-contract/v2','created_utc':now(),'protocol_ref':ref(P/'protocol.json'),
        'capability_ref':ref(out/'raw.json'),'direct_mode_readback_available':False,'silent_fallback_independently_excluded':False,
        'mode_evidence':'documented SDK26 HWTrace success after driver initialization before context with clean STATE; conditional on implementation contract',
        'software_fallback_requested':False,'service_cost_qualification':'unvalidated'}
    write_new(destination,proof);return destination

def retained_rows(directory,n):
    rows=[]
    for item in make_plan(n):
        folder=directory/item['id'];path=folder/'finish.json'
        if not path.is_file():continue
        finish=load(path)
        for r in b.references(finish):verify_ref(r)
        row={**item,'status':finish['status'],'finish_ref':ref(path)}
        if finish['status']=='valid':
            need(finish.get('identity_unchanged') is True,'valid row has no after-identity check')
            summary=validate_raw(load(finish['raw_ref']['path']),item['state'],item['mode'])
            need(summary==finish['summary'],'saved row summary differs from original raw records');row['summary']=summary
        else:row['reason']=finish.get('execution_error',finish.get('identity_error','unknown_rejection'))
        rows.append(row)
    return rows

def evaluate(directory,n):
    start=load(directory/'start.json');immutable_guard(start);rows=retained_rows(directory,n)
    report=assess_blocks(rows,n);report.update({'created_utc':now(),'protocol_ref':ref(P/'protocol.json'),'campaign_start_ref':ref(directory/'start.json'),'process_rows':rows,
        'failed_or_missing_processes':n*20-sum(r['status']=='valid' for r in rows),'all_processes_preserved':True,'independent_operator_data_only':True})
    destination=directory/f'evaluation_{n:02}.json';need(not destination.exists(),'evaluation already exists; do not overwrite')
    write_new(destination,report);return report

def campaign(directory,phase,commit=None):
    need(phase!='extend','R31 has no extension budget; retain five-block result')
    directory=Path(directory).resolve();need(directory.parent==P and directory.name.startswith('run.'),'campaign must be a new run.N child of this directory')
    if phase=='run':
        verify_review(commit);_,refs=prepared();prereq=runtime_prerequisites();directory.mkdir(exist_ok=False)
        start={'schema':'r28-gpu-timing-campaign/v1','created_utc':now(),'reviewed_commit':commit,'prepared_refs':refs,'runtime_prerequisites':prereq,
            'budget_begin_qpc_ns':time.perf_counter_ns(),'budget_seconds':1200,'initial_plan':make_plan(5),'full_extension_plan':make_plan(10),'target_actuals_used_for_selection':False}
        write_new(directory/'start.json',start)
    else:start=load(directory/'start.json');immutable_guard(start)
    if phase=='extend':
        need(not (directory/'extension.json').exists(),'the only permitted extension already used')
        report=load(directory/'evaluation_05.json');recomputed=assess_blocks(retained_rows(directory,5),5)
        need(recomputed['extension_eligible'] and report['states']==recomputed['states'],'extension not justified by fixed inconclusive-interval rule')
        need(remaining(start)>0,'budget exhausted before extension')
        write_new(directory/'extension.json',{'schema':'r28-one-time-extension/v1','created_utc':now(),'from_blocks':5,'to_blocks':10,'prior_evaluation_ref':ref(directory/'evaluation_05.json'),'remaining_budget_seconds':remaining(start)})
    n=10 if (directory/'extension.json').exists() else 5
    terminal=directory/f'complete_{n:02}.json';need(not terminal.exists(),'campaign stage already terminal; no repeated execution')
    try:
        proof=ensure_proof(directory,start)
        for item in make_plan(n):
            folder=directory/item['id']
            if (folder/'finish.json').is_file():
                finish=load(folder/'finish.json')
                for r in b.references(finish):verify_ref(r)
                continue
            need(not folder.exists(),'interrupted process directory retained; cannot retry into same evidence slot')
            need(remaining(start)>0,'20-minute budget exhausted')
            result=run_process(folder,'formal',start,proof,item['state'],item['mode'])
            need(result.get('identity_unchanged') is True,'identity drift stops campaign')
            need(result.get('soft_deadline_exceeded') is not True and result.get('collector_budget_exceeded') is not True,'soft deadline hit; stop new slots, preserve natural-exit evidence')
        report=evaluate(directory,n)
        write_new(terminal,{'status':'completed','created_utc':now(),'evaluation_ref':ref(directory/f'evaluation_{n:02}.json'),'remaining_budget_seconds':remaining(start),'envelope_controls_passed_all_states':report['envelope_controls_passed_all_states'],'formal_success':False,'service_cost_qualification':'unvalidated'})
        return report
    except Exception as e:
        write_new(terminal,{'status':'rejected_or_budget_exhausted','created_utc':now(),'reason':type(e).__name__+': '+str(e),'remaining_budget_seconds':remaining(start),'formal_success':False,'failed_and_missing_slots_remain_in_denominator':True})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('check','run','resume','extend','evaluate'));parser.add_argument('--reviewed-commit');parser.add_argument('--run-dir',type=Path,default=P/'run.0001');parser.add_argument('--blocks',type=int,choices=(5,10),default=5);args=parser.parse_args()
    if args.phase=='check':prepared();print(json.dumps({'status':'compiled_and_host_tested','GPU_executed':False,'runtime_preconditions_checked':False,'M4_held_out':True}))
    elif args.phase=='evaluate':print(json.dumps(evaluate(args.run_dir,args.blocks),indent=2))
    else:
        result=campaign(args.run_dir,args.phase,args.reviewed_commit);print(json.dumps({'formal_success':result['formal_success'],'envelope_controlled_states':result['envelope_controlled_states'],'service_cost_qualification':'unvalidated','output':str(args.run_dir),'simulator_coefficients_emitted':False}))
