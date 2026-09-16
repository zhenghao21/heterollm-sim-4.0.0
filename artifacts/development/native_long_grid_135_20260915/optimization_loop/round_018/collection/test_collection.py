"""Host-only graph-collector preparation tests; never launches native, CUDA or NVML."""
import copy
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common,runner,worker,quality,probe_adapter
from extract import mutually_exclusive_partition,correlate


def test_six_configs_three_parity_pairs_full54stages():
    configs=[{'id':str(i)} for i in range(6)];stages=runner.plan(configs)
    assert len(stages)==54
    assert [s['mode'] for s in stages[:9]]==['direct','profile','export','profile','direct','export','direct','profile','export']
    assert [s['mode'] for s in stages[9:12]]==['profile','direct','export']
    assert all(sum(s['mode']==mode for s in stages)==18 for mode in ('profile','direct','export'))


def test_actual_graph_protocol_semantics_read_only():
    protocol=probe_adapter.protocol_document()
    assert len(probe_adapter.validate_protocol(protocol))==6
    assert protocol['execution']['probe_cuda_events'] is False
    assert protocol['runtime']['environment']['GGML_CUDA_DISABLE_GRAPHS'] is None
    changed=copy.deepcopy(protocol);changed['execution']['formal']=29
    with pytest.raises(common.IdentityError):probe_adapter.validate_protocol(changed)


def test_actual_graph_argv_eventfree_and_pair_identity():
    config=probe_adapter.protocol_document()['configs'][0]
    direct=runner.app_argv(config,Path('case/pair_01/direct/microbench.json'))
    profile=runner.app_argv(config,Path('case/pair_01/profile/microbench.json'))
    assert direct[:-1]==profile[:-1]
    assert direct[direct.index('--pair-id')+1]==config['id']+'/pair_01'
    assert '--control' not in direct and '--nvtx' not in direct and '--config' in direct


def test_checkpoint_keeps_child_ownership():
    class Running:
        pid=123
        def wait(self,timeout):raise subprocess.TimeoutExpired('mock',timeout)
        def kill(self):raise AssertionError('no kill')
        def terminate(self):raise AssertionError('no terminate')
    r=runner.wait_checkpoint(Running(),180)
    assert r['status']=='still_running' and r['kill_on_deadline'] is False


def test_true_exit_wait_retries_exception_without_termination():
    class Child:
        def __init__(self):self.calls=0
        def wait(self):
            self.calls+=1
            if self.calls==1:raise OSError('mock transient wait')
            return 0
    errors=[];assert worker.wait_for_real_exit(Child(),errors)==0;assert len(errors)==1


def clock_fixture(value=2392):
    rows=[];samples=[]
    for i in range(30):
        start=100000+i*100000;end=start+1000
        rows.append({'phase':'formal','index':i,'qpc_start':start,'qpc_end':end})
        for lo in (start-1000,end+1000):samples.append({'sm_read_begin_qpc':lo,'sm_read_end_qpc':lo+100,'sm_mhz':{'status':0,'value':value}})
    app={'qpc_frequency':1000000,'runs':rows};raw={'qpc_frequency':1000000,'samples':samples,
        'lifecycle':{'thread_exited':True,'NVML_shutdown_after_reads':True,'timer_handles_closed_after_join':True,'errors':[]}}
    return app,raw


def test_sm_read_windows_original_gate_and_complete30_required():
    app,raw=clock_fixture();assert common.clock_readback_gate(app,raw)['passed'] is True
    app['runs'].pop();assert common.clock_readback_gate(app,raw)['passed'] is False
    app,raw=clock_fixture(2700);assert common.clock_readback_gate(app,raw)['passed'] is False


def test_active_sampler_cannot_be_marked_valid():
    app,raw=clock_fixture();raw['lifecycle']['thread_exited']=False
    assert common.clock_readback_gate(app,raw)['passed'] is False


def test_partition_closes_with_overlapping_GPU_and_host_API():
    kernels=[{'start':20,'end':70}];apis=[{'name_text':'cudaGraphLaunch_v10000','start':10,'end':30},{'name_text':'cudaStreamSynchronize_v3020','start':30,'end':90}]
    p=mutually_exclusive_partition(0,100,kernels,apis)
    assert sum(p.values())==100 and p['GPU']==50 and p['host_launch']==10 and p['host_sync']==20 and p['unattributed']==20


def test_all_processes_and_quality_failures_stay_visible():
    c={'id':'x'};pairs=[{'pair':i,'numerics_all_rows':False} for i in range(3)]
    r=quality.quality(c,pairs)
    assert r['measurement_cost_eligible'] is False and r['pairs_observed']==3 and r['pairs_required']==3


def test_no_clock_or_GPU_action_on_import():
    assert common.POLICY['target_sm_clock_mhz']==2400 and common.POLICY['maximum_formal_clock_bracket_gap_ms']==25
    assert callable(worker.run) and callable(runner.prepare)
    assert common.POLICY['high_frequency_fields']==['sm_mhz']


def raw_fixture(config,output=None,pair=1,pid=1):
    protocol=probe_adapter.protocol_document();records=[]
    output=Path(output or 'synthetic_unused_output.jsonl').resolve();pair_id=config['id']+f'/pair_{pair:02d}'
    argv=[str((common.PROBE/'graph-submit-probe.exe').resolve()),'--run','--config',config['id'],'--pair-id',pair_id,'--output',str(output)]
    records.append({'pid':pid,'argv':argv,'qpc_start':1000,'utc_start':'2026-09-16T00:00:00.000Z','record':'header','schema':'graph-submit-probe/v1','qpc_frequency':1000000,'protocol_sha256':common.ref(common.PROBE/'protocol.json')['sha256'],'probe_cuda_events':False,'trace_clock_subtraction_allowed':False})
    environment=dict(protocol['runtime']['environment']);environment.update({k:None for k in protocol['runtime']['extra_clear_environment']})
    nodes=[];elements=config['elements'];count=config['nodes']
    for i in range(count):nodes.append({'index':i,'name':'scale_'+str(i+1),'src0':'graph_input' if i==0 else 'scale_'+str(i),'operator':'SCALE','dtype':'F32','scale':.5,'bias':0,'ne':[elements,1,1,1],'nb':[4,4*elements,4*elements,4*elements]})
    module={'name':'mock','path':'mock','sha256':'a'*64}
    records.append({'record':'setup','pair_id':pair_id,'config':config['id'],'elements':elements,'requested_nodes':count,'actual_ggml_nodes':count,'tensor_payload_bytes':4*elements,'allocated_payload_bytes':(count+1)*4*elements,'allocated_buffer_bytes':(count+1)*4*elements,'logical_read_bytes':count*4*elements,'logical_write_bytes':count*4*elements,'environment':environment,'scheduling':{'CPU_worker_pool_created':False,'process_priority_class':32,'caller_thread_priority':0,'process_affinity':255,'caller_thread_id':5},'hardware_actual':{'uuid':protocol['runtime']['gpu_expected']['uuid'],'name':protocol['runtime']['gpu_expected']['name'],'cc_major':12,'cc_minor':0,'SMs':84,'L2_bytes':67108864,'total_memory_bytes':16*1024**3},'graph_nodes':nodes,'loaded_modules_before':[module]})
    for ordinal,(phase,index) in enumerate([('first',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]):
        label=f"graph_submit/{config['id']}/{phase}/{index}";base=10000+ordinal*10000
        keys=['qpc_outer_push_start','qpc_outer_push_end','qpc_submit_push_start','qpc_submit_push_end','qpc_submit_start','qpc_submit_end','qpc_submit_pop_start','qpc_submit_pop_end','qpc_sync_push_start','qpc_sync_push_end','qpc_sync_start','qpc_sync_end','qpc_sync_pop_start','qpc_sync_pop_end','qpc_outer_pop_start','qpc_outer_pop_end']
        call={k:base+i*100 for i,k in enumerate(keys)};call.update(record='graph_call',label=label,graph_status=0,observed_device_kernel_count=None,observed_cuda_graph_launch_count=None);records.append(call)
        result={'checked':elements,'mismatches':0,'nonfinite':0,'first_bad_index':-1,'max_abs_error':0,'bitwise_pass':True}
        records.append({'record':'validation','label':label,'phase':phase,'index':index,'stage':count,'qpc_validation_start':base+2000,'qpc_validation_end':base+3000,'result':result})
        if ordinal in (0,35):
            records.extend({'record':'intermediate_validation','label':label,'stage':j,'qpc_start':base+3000+j*100,'qpc_end':base+3000+j*100+1,'result':result} for j in range(1,count))
    records.append({'record':'footer','qpc_end':400000,'utc_end':'2026-09-16T00:00:00.400Z','status':'complete','math_pass':True,'graph_calls':36,'modules_stable':True,'loaded_modules_after':[module]})
    return records


def test_graph_jsonl_shape_count_dependency_and_exact_validation(tmp_path):
    import json
    c=probe_adapter.protocol_document()['configs'][2];path=tmp_path/'graph.jsonl';records=raw_fixture(c,output=path)
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    doc=probe_adapter.read_raw(path);assert probe_adapter.audit_raw(doc,c,{})['valid_raw']
    assert len(probe_adapter.expected_labels(doc,c))==36
    assert doc['runs'][0]['phase']=='first' and doc['runs'][-1]['index']==29
    doc['setup']['graph_nodes'][1]['src0']='graph_input'
    assert 'graph_dependency_mismatch' in probe_adapter.audit_raw(doc,c,{})['issues']


def test_partial_or_numeric_failed_graph_record_denied(tmp_path):
    import json
    c=probe_adapter.protocol_document()['configs'][0];path=tmp_path/'bad.jsonl';records=raw_fixture(c,output=path);records[3]['result']['mismatches']=1
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    assert not probe_adapter.audit_raw(probe_adapter.read_raw(path),c,{})['valid_raw']
    records.pop();path.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    with pytest.raises(ValueError,match='footer'):probe_adapter.read_raw(path)


def test_observed_scale_count_never_filled_with_planned_count():
    c=probe_adapter.protocol_document()['configs'][1]
    kernel={'evidence_rowid':1,'start':1,'end':2,'short_name_text':'scale_f32','gridX':4,'gridY':1,'gridZ':1,'blockX':256,'blockY':1,'blockZ':1,'dynamicSharedMemory':0,'deviceId':0,'streamId':7,'globalPid':1}
    result=probe_adapter.validate_kernel_chain([kernel],c)
    assert result['actual_kernel_count']==1 and result['complete'] is False
    assert c['nodes']==8


def trace_fixture(config):
    """Synthetic schema-shaped evidence, not an actual Nsight export."""
    tables={'NVTX_EVENTS':[],'CUPTI_ACTIVITY_KIND_RUNTIME':[],
        'CUPTI_ACTIVITY_KIND_DRIVER':[],'CUPTI_ACTIVITY_KIND_KERNEL':[]}
    app={'runs':[],'header':{'pid':1},'setup':{'scheduling':{'caller_thread_id':5}}};pid=1<<24;tid=pid+5
    for ordinal,(phase,index) in enumerate(common.expected_calls()):
        base=ordinal*1000000;correlation=ordinal+100
        label=f"graph_submit/{config['id']}/{phase}/{index}"
        app['runs'].append({'nvtx_label':label,'phase':phase,'index':index})
        tables['NVTX_EVENTS'].append({'resolved_text':label,'start':base,'end':base+10000,'globalTid':tid})
        tables['CUPTI_ACTIVITY_KIND_RUNTIME'].extend([
            {'start':base+100,'end':base+200,'globalTid':tid,'correlationId':correlation,'returnValue':0,'name_text':'cudaGraphLaunch'},
            {'start':base+500,'end':base+9000,'globalTid':tid,'correlationId':correlation+1000,'returnValue':0,'name_text':'cudaStreamSynchronize'}])
        tables['CUPTI_ACTIVITY_KIND_DRIVER'].append({'start':base+110,'end':base+180,'globalTid':tid,'correlationId':correlation,'returnValue':0,'name_text':'cuGraphLaunch'})
        for node in range(config['nodes']):
            tables['CUPTI_ACTIVITY_KIND_KERNEL'].append({'evidence_rowid':ordinal*config['nodes']+node+1,
                'start':base+1000+node*400,'end':base+1200+node*400,'globalPid':pid,'correlationId':correlation,
                'deviceId':0,'streamId':7,'short_name_text':'scale_f32','gridX':(config['elements']+255)//256,
                'gridY':1,'gridZ':1,'blockX':256,'blockY':1,'blockZ':1,'dynamicSharedMemory':0})
    return {'tables':tables},app


def test_graph_launch_maps_all_children_without_runtime_driver_double_count():
    config=probe_adapter.protocol_document()['configs'][1]
    raw,app=trace_fixture(config);result=correlate(raw,app,config)
    assert result['all36_chain_complete'] is True
    assert result['actual_kernels_mapped']==36*8
    assert all(row['actual_kernel_count']==8 and len(row['api_events_verbatim'])==3 for row in result['calls'])
    assert all(sum(row['exclusive_partition_ns'].values())==row['nvtx_full_call_duration_ns'] for row in result['calls'])


def test_graph_child_without_launch_correlation_remains_rejected():
    config=probe_adapter.protocol_document()['configs'][1]
    raw,app=trace_fixture(config);raw['tables']['CUPTI_ACTIVITY_KIND_KERNEL'][0]['correlationId']=99999
    result=correlate(raw,app,config)
    assert result['all36_chain_complete'] is False
    assert result['calls'][0]['actual_kernel_count']==7
    assert 'captured_graph_child_launch_correlation_unresolved' in result['calls'][0]['issues']
    assert result['unmapped_kernel_rowids']==[1]


def test_same_kernel_cannot_be_assigned_to_two_graph_calls():
    config=probe_adapter.protocol_document()['configs'][1]
    raw,app=trace_fixture(config)
    raw['tables']['NVTX_EVENTS'][1]['start']=50
    with pytest.raises(ValueError,match='multiple graph calls'):correlate(raw,app,config)


def test_unknown_trace_diagnostic_severity_is_not_silently_accepted():
    config=probe_adapter.protocol_document()['configs'][1]
    raw,app=trace_fixture(config)
    raw['tables']['DIAGNOSTIC_EVENT']=[{'severity':999,'text':'unclassified diagnostic'}]
    result=correlate(raw,app,config)
    assert result['warning_diagnostics_verbatim']==raw['tables']['DIAGNOSTIC_EVENT']


# End-to-end evidence fixtures below isolate the stage binding gate. All files,
# PIDs and QPC values are synthetic; no native process, CUDA or NVML is invoked.
def completed_stage_fixture(monkeypatch,tmp_path,mode='direct'):
    import extract,json
    root=tmp_path/'collection';config=probe_adapter.protocol_document()['configs'][1]
    directory=root/'runs'/config['id']/'pair_01'/mode;directory.mkdir(parents=True)
    monkeypatch.setattr(extract,'HERE',root)
    probe=probe_adapter.protocol_document();environment=dict(probe['runtime']['environment'])
    environment.update({name:None for name in probe['runtime']['extra_clear_environment']})
    protocol={'configs':probe['configs'],'probe_root':str(common.PROBE),
        'executable':{'path':str(common.PROBE/'graph-submit-probe.exe')},
        'nsys':{'executable':{'path':probe['profiler']['path']}},'nsys_profile_options':probe['profiler']['options'][1:],
        'gpu_identity':{'uuid':probe['runtime']['gpu_expected']['uuid']},'environment_explicit':environment}
    def put(path,value):Path(path).write_text(json.dumps(value),encoding='utf-8')
    put(root/'protocol.json',protocol)
    anchor={'path':str(root/'clock.json'),'sha256':'b'*64,'bytes':10}
    put(root/'clock-control-binding.json',{'receipt_ref':anchor})
    def clock(path,sha,gpu):
        assert path==anchor['path'] and sha==anchor['sha256'] and gpu==protocol['gpu_identity']['uuid']
        return {'receipt_ref':anchor}
    monkeypatch.setattr(extract,'verify_clock_receipt',clock)
    stage={'config_id':config['id'],'pair':0,'mode':mode}
    argv=probe_adapter.stage_command(stage,directory,protocol)
    inputs=[]
    if mode=='export':
        source=directory.parent/'profile/trace.nsys-rep';source.parent.mkdir(exist_ok=True);source.write_bytes(b'synthetic report; never parsed as actual trace')
        inputs=[common.ref(source)];(directory/'trace.sqlite').write_bytes(b'synthetic SQLite; evidence identity test only')
    else:
        raw=directory/'microbench.json';raw.write_text('\n'.join(json.dumps(x) for x in raw_fixture(config,output=raw,pid=100))+'\n',encoding='utf-8')
        put(directory/'telemetry.json',{'synthetic_test_only':True})
        if mode=='profile':(directory/'trace.nsys-rep').write_bytes(b'synthetic report')
    spec={'schema':'graph-collection-process-spec/v1','stage':stage,'directory':str(directory),'freeze':str(root/'freeze.json'),
        'expected_freeze_sha256':'a'*64,'argv':argv,'cwd':protocol['probe_root'],'environment':environment,
        'gpu_identity':protocol['gpu_identity'],'telemetry':mode!='export','clock_control_binding':{'receipt_ref':anchor},'input_artifacts':inputs}
    put(directory/'spec.json',spec);spec_ref=common.ref(directory/'spec.json')
    actual_pid=100 if mode=='direct' else 200
    put(directory/'launched.json',{'pid':actual_pid,'supervisor_pid':50,'qpc_after_launch':800,'argv':argv})
    put(directory/'supervisor-started.json',{'pid':50,'spec':spec_ref,'freeze_before':{'passed':True,'external_approved_sha256':'a'*64}})
    (directory/'stdout.txt').write_bytes(b'');(directory/'stderr.txt').write_bytes(b'')
    receipt={'schema':'graph-collection-process/v1','clock_control_binding':{'receipt_ref':anchor},'external_approved_sha256':'a'*64,
        'child_process_exited':True,'status':'completed','returncode':0,'freeze_after':{'passed':True,'external_approved_sha256':'a'*64},
        'spec_ref':spec_ref,'argv':argv,'cwd':protocol['probe_root'],'process_pid':actual_pid,'supervisor_pid':50,
        'qpc_frequency':1000000,'qpc_launch_start':500,'qpc_process_complete':500000,
        'artifacts':[common.ref(f) for f in directory.iterdir() if f.is_file()]}
    def save():put(directory/'complete.json',receipt)
    save();return directory,receipt,save


@pytest.mark.parametrize('mode',['direct','profile','export'])
def test_full_stage_evidence_accepts_only_its_bound_process(monkeypatch,tmp_path,mode):
    import extract
    path,receipt,save=completed_stage_fixture(monkeypatch,tmp_path,mode)
    result=extract.stage_complete(path,'a'*64)
    assert result['evidence_binding']['stage']['mode']==mode
    assert result['evidence_binding']['verified_actual_argv_and_PID'] is True
    if mode!='export':assert result['evidence_binding']['process_origin']['pid']==100
    # Both empty native stdout and stderr are valid evidence with real zero size.
    assert result['evidence_binding']['artifacts']['stdout.txt']['bytes']==0


@pytest.mark.parametrize('mutation',['empty','missing_raw','missing_telemetry','duplicate','foreign_path','missing_hash'])
def test_full_stage_artifact_set_rejects_incomplete_or_foreign(monkeypatch,tmp_path,mutation):
    import extract
    path,receipt,save=completed_stage_fixture(monkeypatch,tmp_path)
    if mutation=='empty':receipt['artifacts']=[]
    elif mutation.startswith('missing_') and mutation!='missing_hash':
        name='microbench.json' if mutation=='missing_raw' else 'telemetry.json'
        receipt['artifacts']=[r for r in receipt['artifacts'] if Path(r['path']).name!=name]
    elif mutation=='duplicate':receipt['artifacts'].append(copy.deepcopy(receipt['artifacts'][0]))
    elif mutation=='foreign_path':
        foreign=tmp_path/'microbench.json';foreign.write_bytes((path/'microbench.json').read_bytes())
        receipt['artifacts']=[common.ref(foreign) if Path(r['path']).name=='microbench.json' else r for r in receipt['artifacts']]
    elif mutation=='missing_hash':receipt['artifacts'][0]['sha256']=None
    save()
    with pytest.raises((ValueError,KeyError,TypeError,OSError)):extract.stage_complete(path,'a'*64)


@pytest.mark.parametrize('mutation',['pid','pair','argv_output','before_parent_launch','missing_pid'])
def test_full_stage_rejects_raw_from_another_origin(monkeypatch,tmp_path,mutation):
    import extract,json
    path,receipt,save=completed_stage_fixture(monkeypatch,tmp_path)
    raw=path/'microbench.json';records=[json.loads(x) for x in raw.read_text().splitlines()]
    if mutation=='pid':records[0]['pid']=101
    elif mutation=='missing_pid':records[0].pop('pid')
    elif mutation=='pair':
        records[1]['pair_id']=records[1]['pair_id'].replace('pair_01','pair_02');records[0]['argv'][5]=records[1]['pair_id']
    elif mutation=='argv_output':records[0]['argv'][-1]=str(tmp_path/'other.jsonl')
    elif mutation=='before_parent_launch':records[0]['qpc_start']=200
    raw.write_text('\n'.join(json.dumps(x) for x in records)+'\n',encoding='utf-8')
    receipt['artifacts']=[common.ref(raw) if Path(r['path']).name=='microbench.json' else r for r in receipt['artifacts']];save()
    with pytest.raises((ValueError,KeyError,TypeError,OSError)):extract.stage_complete(path,'a'*64)


def test_trace_nonzero_VM_namespace_binds_actual_PID_and_TID():
    config=probe_adapter.protocol_document()['configs'][1];raw,app=trace_fixture(config)
    for table,field in [('NVTX_EVENTS','globalTid'),('CUPTI_ACTIVITY_KIND_RUNTIME','globalTid'),('CUPTI_ACTIVITY_KIND_DRIVER','globalTid'),('CUPTI_ACTIVITY_KIND_KERNEL','globalPid')]:
        for row in raw['tables'][table]:row[field]|=3<<48
    result=correlate(raw,app,config)
    assert result['all36_chain_complete'] is True
    assert result['native_process_binding']['trace_global_pid']==(3<<48)|(1<<24)
    app['setup']['scheduling']['caller_thread_id']=6
    with pytest.raises(ValueError,match='process/thread'):correlate(raw,app,config)


def test_independent_origins_reject_duplicate_but_allow_later_PID_reuse():
    one={'pid':10,'qpc_start':100,'qpc_end':200}
    two={'pid':10,'qpc_start':300,'qpc_end':400}
    pairs=[{'pair':0,'process_origins':{'direct':one}},{'pair':1,'process_origins':{'direct':two}}]
    assert probe_adapter.unique_process_origins(pairs)==[]
    pairs[1]['process_origins']['direct']=copy.deepcopy(one)
    assert any(i['reason']=='duplicate_native_process_origin' for i in probe_adapter.unique_process_origins(pairs))
