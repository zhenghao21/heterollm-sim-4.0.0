"""All tests are pure host logic; collector, CUDA and NVML are never executed."""
from pathlib import Path
from copy import deepcopy
import importlib.util,math,subprocess
import pytest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r28_host_logic',P/'runner.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_fixed_scope_and_complete_unseen_M4():
    p=r.protocol();assert p['development']['M']==1 and p['held_out']['M']==4
    assert {s['id'] for s in p['states']}=={'pair_warm','pair_rotate','main_warm','main_rotate','convert_warm'}
    assert len(r.make_plan(5))==100 and len(r.make_plan(10))==200
    assert len({a['id'] for a in r.make_plan(10)})==200
    assert r.make_plan(10)[:100]==r.make_plan(5)

@pytest.mark.parametrize('n,k,coverage',[(5,1,.9375),(10,2,.978515625)])
def test_exact_order_statistic_coverage_not_bootstrap(n,k,coverage):
    values=[1+i*.001 for i in range(n)];a=r.median_ratio_interval(values)
    assert a['order_k']==k and a['coverage_under_iid_continuous_block_ratio_model']==coverage
    assert a['low']==values[k-1] and a['high']==values[-k]
    assert a['independence_proven'] is False and a['familywise_coverage_claimed'] is False
    assert a['two_look_single_comparison_union_lower_bound']==.916015625

@pytest.mark.parametrize('values',[[1]*64,[float('nan')]*5,[0]*5,[True]*5,[1]*4])
def test_invalid_or_pseudoreplicated_intervals_rejected(values):
    with pytest.raises(ValueError):r.median_ratio_interval(values)


def test_PDL_overlap_and_gap_are_distinct():
    rows=[dict(name=r.CONV,start=100,end=160),dict(name=r.MAIN,start=140,end=250)]
    a=r.pair_metrics(rows);assert a['observed_pair_span_ns']==150 and a['observed_pair_union_ns']==150 and a['observed_interval_overlap_ns']==20 and a['observed_inter_kernel_gap_ns']==0
    rows[1].update(start=180,end=250);a=r.pair_metrics(rows)
    assert a['observed_pair_span_ns']==150 and a['observed_pair_union_ns']==130 and a['observed_inter_kernel_gap_ns']==20 and a['observed_interval_overlap_ns']==0
    assert a['observed_main_in_pair_lifespan_ns']==70


def capability():
    return {'schema':'r28-gpu-operator-raw/v2','status':'capability_observed_under_API_contract','kind':'capability_hes',
        'protocol_ref':r.ref(P/'protocol.json'),'hardware':r.protocol()['hardware'],'timed':False,'GPU_kernel_executed':False,
        'budget_exceeded':False,'HES_enable_returncode':0,
        'contexts_before':{'primary_active':0,'current_context':0},'contexts_after':{'primary_active':0,'current_context':0},'driver_init':{'qpc_end':10},
        'actual_mode_evidence':{'kind':'HWTrace_SDK26_API_contract_only','direct_mode_readback_available':False,
            'silent_fallback_independently_excluded':False,'latency_timestamps_requested':False,'software_fallback_requested':False},
        'CUPTI_evidence':{'compile_API_version':26,'runtime_API_version':26,'callback_overflow':False,'STATE':[],
            'calls':[dict(name='cuptiActivityEnableHWTrace(1)',returncode=0,qpc=20,phase='enable_HES_before_context')]}}


def test_single_legal_HES_capability_only_claims_SDK_contract():
    assert r.validate_capability(capability())
    for key in ('direct_mode_readback_available','silent_fallback_independently_excluded'):
        unsupported=capability();unsupported['actual_mode_evidence'][key]=True
        with pytest.raises(ValueError,match='overstates'):r.validate_capability(unsupported)

@pytest.mark.parametrize('damage',['late','warning','version','context','timestamp_only','failed_API','counterfactual','budget'])
def test_HES_request_flag_and_nonzero_timestamps_cannot_substitute_legal_contract_evidence(damage):
    a=capability()
    if damage=='late':a['CUPTI_evidence']['calls'][0]['qpc']=1
    if damage=='warning':a['CUPTI_evidence']['STATE']=[dict(id=3,result=27,phase='formal',message='warning')]
    if damage=='version':a['CUPTI_evidence']['runtime_API_version']=130401
    if damage=='context':a['contexts_before']['primary_active']=1
    if damage=='timestamp_only':a['CUPTI_evidence']['calls']=[];a['kernel_start']=100;a['kernel_end']=200
    if damage=='failed_API':a['HES_enable_returncode']=27
    if damage=='counterfactual':a['CUPTI_evidence']['calls'].append(dict(name='cuptiActivityEnableLatencyTimestamps(1)',returncode=27,qpc=21))
    if damage=='budget':a['budget_exceeded']=True
    with pytest.raises(ValueError):r.validate_capability(a)

def activity(tmp_path,role='pair'):
    path=tmp_path/'raw_buffer.bin';path.write_bytes(b'host synthetic raw fixture')
    trace={'kernel_record_ABI':'CUpti_ActivityKernel9','dropped_records':0,'buffer_overflow':False,'unknown_kinds':[],
        'raw_buffers':[{'ref':r.ref(path),'valid_bytes':path.stat().st_size}], 'kernels':[],'runtime_APIs':[],'external_links':[]}
    corr=1;symbols=[r.CONV,r.MAIN] if role=='pair' else [r.MAIN if role=='main' else r.CONV]
    for i,eid in enumerate(list(range(1000000,1000032))+list(range(1,65))):
        for j,symbol in enumerate(symbols):
            start=10000+i*1000+j*20;end=start+(50 if symbol==r.CONV else 100)
            geometry=r.protocol()['expected_conversion' if symbol==r.CONV else 'expected_main']
            trace['kernels'].append(dict(kind=10,name=symbol,start=start,end=end,correlation=corr,context=4,stream=5,device=0,graph_id=0,graph_node_id=0,dynamic_shared=0,
                grid=list(geometry['grid']),block=list(geometry['block'])))
            trace['runtime_APIs'].append(dict(cbid=430,start=start-10,end=start+5,correlation=corr,return_value=0))
            trace['external_links'].append(dict(kind=3,external_id=eid,correlation=corr));corr+=1
    return {'activity':trace,'state':{'role':role}}

@pytest.mark.parametrize('role',['pair','main','convert'])
def test_64_formal_calls_are_separate_from32_priming_and_throughput(tmp_path,role):
    values=r.validate_activity(activity(tmp_path,role));assert len(values)==64 and values[0]['sample']==0 and values[-1]['sample']==63
    if role=='pair':assert values[0]['observed_interval_overlap_ns']==30 and values[0]['observed_pair_span_ns']==120

@pytest.mark.parametrize('damage',['drop','zero','serialized','missing','correlation','graph','cross_sample_overlap','unknown','no_raw'])
def test_activity_completeness_and_path_failures_reject(tmp_path,damage):
    a=activity(tmp_path);t=a['activity']
    if damage=='drop':t['dropped_records']=1
    if damage=='zero':t['kernels'][0]['start']=0
    if damage=='serialized':t['kernels'][0]['kind']=3
    if damage=='missing':t['kernels'].pop()
    if damage=='correlation':t['external_links'][0]['external_id']=99
    if damage=='graph':t['kernels'][0]['graph_id']=2
    if damage=='cross_sample_overlap':t['kernels'][2]['start']=t['kernels'][0]['start']
    if damage=='unknown':t['unknown_kinds']=[3]
    if damage=='no_raw':t['raw_buffers']=[]
    with pytest.raises(ValueError):r.validate_activity(a)


def rows(n=5):
    result=[]
    for item in r.make_plan(n):
        role=next(s['role'] for s in r.protocol()['states'] if s['id']==item['state'])
        if role=='pair':values=dict(observed_pair_span_ns=100,observed_pair_union_ns=100,observed_conversion_lifespan_ns=50,observed_main_in_pair_lifespan_ns=70,observed_interval_overlap_ns=20,observed_inter_kernel_gap_ns=0,host_late_submission_lower_bound_ns=0)
        elif role=='main':values={'observed_main_lifespan_ns':100}
        else:values={'observed_conversion_lifespan_ns':30}
        result.append({**item,'status':'valid','summary':{'fixture_fingerprints':{'input':'host_test_same_fixture'},'wall_ns':1000,'event_ns':200 if item['mode'] in ('A','AB') else None,
            'kernel_metrics':[dict(sample=i,**values) for i in range(64)] if item['mode'] in ('B','AB') else []}})
    return result


def test_all5_states_and_two_controls_are_retained():
    a=r.assess_blocks(rows(),5);assert a['envelope_controls_passed_all_states'] and a['envelope_controlled_states']==5 and not a['extension_eligible']
    assert a['formal_success'] is False and a['service_cost_qualified_states']==0 and a['service_cost_qualification']=='unvalidated'
    assert all(s['service_cost_qualification']=='unvalidated' and all(point['service_cost_qualification']=='unvalidated' for point in s['observed_duration_points']) for s in a['states'])
    assert all(s['required_processes']==20 and s['cost_model_parameters_emitted'] is False for s in a['states'])


def test_inconclusive_interval_remains_retained_without_R31_extension():
    values=rows();target=next(v for v in values if v['state']=='pair_warm' and v['mode']=='B' and v['block']==1);target['summary']['wall_ns']=940
    a=r.assess_blocks(values,5);assert not a['formal_success'] and not a['extension_eligible'] and a['states'][0]['reasons']==['perturbation_interval_inconclusive']


def test_observed_median_failure_must_not_be_resampled_until_passing():
    values=rows()
    for v in values:
        if v['state']=='pair_warm' and v['mode']=='B':v['summary']['wall_ns']=1080
    a=r.assess_blocks(values,5);assert not a['extension_eligible'] and 'observed_median_perturbation_exceeds_5pct' in a['states'][0]['reasons']


def test_worst_block_is_not_silently_dropped():
    values=rows();target=next(v for v in values if v['state']=='main_warm' and v['mode']=='B' and v['block']==1)
    for k in target['summary']['kernel_metrics']:k['observed_main_lifespan_ns']=200
    a=r.assess_blocks(values,5);s=next(x for x in a['states'] if x['state']=='main_warm');assert 'cross_block_elapsed_cost_variation_exceeds_5pct' in s['reasons'] and not a['extension_eligible']


def test_failures_and_missing_modes_stay_in_denominator():
    values=rows();values[0]['status']='rejected';values.pop()
    a=r.assess_blocks(values,5);assert not a['formal_success'] and not a['extension_eligible'];assert sum(s['required_processes'] for s in a['states'])==100


def test_unknown_M4_state_cannot_enter_frozen_development():
    with pytest.raises(ValueError,match='held out'):r.validate_raw({},'pair_m4','B',verify_files=False)


def test_empty_prerequisites_cannot_pass(monkeypatch):
    monkeypatch.setattr(r,'assert_idle',lambda:None)
    def unavailable(*a):raise ValueError('no full262 barrier')
    monkeypatch.setattr(r.b,'load',unavailable)
    with pytest.raises(ValueError,match='full262'):r.runtime_prerequisites()


def test_budget_is_monotonic_and_does_not_reset_on_resume(monkeypatch):
    monkeypatch.setattr(r.time,'perf_counter_ns',lambda:2000000000000)
    assert r.remaining({'budget_begin_qpc_ns':1000000000000})==200
    assert r.remaining({'budget_begin_qpc_ns':0})==0
    with pytest.raises(ValueError):r.remaining({'budget_begin_qpc_ns':3000000000000})


def test_live_peer_and_freeze_conflicts(monkeypatch):
    records=[dict(pid=1,name='python.exe',cmdline=['runner.py','run','host_submission_probe']),dict(pid=2,name='python.exe',cmdline=['freeze_candidate.py']),dict(pid=3,name='gpu-operator-timing.exe',cmdline=[]),dict(pid=4,name='dwm.exe',cmdline=[])]
    assert {x['pid'] for x in r.process_conflicts(records,own_pid=99)}=={1,2,3}


def test_review_source_bytes_are_not_inferred_from_HEAD(monkeypatch):
    with pytest.raises(ValueError,match='full reviewed'):r.verify_review('HEAD')


def test_exclusive_evidence_write(tmp_path):
    p=tmp_path/'record.json';r.write_new(p,{'failed':True});before=p.read_bytes()
    with pytest.raises(FileExistsError):r.write_new(p,{'failed':False})
    assert p.read_bytes()==before


def test_control_and_observed_inputs_must_match():
    values=rows();values[0]['summary']['fixture_fingerprints']={'input':'different'}
    result=r.assess_blocks(values,5)
    assert result['states'][0]['reasons']==['data_fixture_changed_or_unbound'] and not result['extension_eligible']



def test_collector_no_illegal_latency_API_or_forced_process_end():
    source=(P/'collector.cpp').read_text(encoding='utf8')
    assert 'cuptiActivityEnableLatencyTimestamps' not in source
    assert 'capability_without_hes' not in source and 'capability_after_hes' not in source
    assert 'cuptiActivityEnableHWTrace' in source and 'HWTrace_SDK26_API_contract_only' in source
    assert 'after_formal_sync' in source and 'before_formal_call' in source
    runner=(P/'runner.py').read_text(encoding='utf8')
    assert '.kill(' not in runner and '.terminate(' not in runner


def test_overdue_process_is_awaited_naturally_and_evidence_retained(tmp_path):
    class Child:
        calls=0
        def wait(self,timeout):
            self.calls+=1
            if self.calls<=2:raise subprocess.TimeoutExpired('mock',timeout)
            return 0
        def kill(self):pytest.fail('must not kill')
        def terminate(self):pytest.fail('must not terminate')
    times=iter([0,11,12,13]);child=Child()
    result=r.wait_naturally(child,10,tmp_path,lambda:next(times))
    assert child.calls==3 and result['returncode']==0 and result['natural_exit'] is True
    assert result['soft_deadline_exceeded'] is True and result['termination_requested'] is False
    saved=r.load(tmp_path/'soft_deadline_exceeded.json')
    assert saved['evidence_eligible'] is False and saved['started_process_must_exit_naturally'] is True


def test_natural_process_exit_before_deadline_has_no_overdue_marker(tmp_path):
    class Child:
        def wait(self,timeout):return 0
    values=iter([1,3]);result=r.wait_naturally(Child(),10,tmp_path,lambda:next(values))
    assert not result['soft_deadline_exceeded'] and result['natural_exit']
    assert not (tmp_path/'soft_deadline_exceeded.json').exists()


def test_return_racing_soft_deadline_is_not_admitted(tmp_path):
    class Child:
        def wait(self,timeout):return 0
    values=iter([9,11]);result=r.wait_naturally(Child(),10,tmp_path,lambda:next(values))
    assert result['soft_deadline_exceeded'] and (tmp_path/'soft_deadline_exceeded.json').is_file()


def test_envelope_equivalence_does_not_establish_kernel_service_costs():
    values=rows()
    # Model a hypothetical observer that doubled every kernel duration while
    # control wall/event ratios happen to remain one. Service is still unvalidated.
    for row in values:
        for point in row['summary']['kernel_metrics']:
            for key in point:
                if key!='sample':point[key]*=2
    result=r.assess_blocks(values,5)
    assert result['envelope_controls_passed_all_states'] is True
    assert result['service_cost_qualified_states']==0 and result['formal_success'] is False
    assert result['service_cost_qualification']=='unvalidated'


def test_Q4_conversion_grid8_is_protocol_qualified_and_old16_rejected(tmp_path):
    raw=activity(tmp_path)
    assert r.protocol()['expected_conversion']['grid']==[8,1,1]
    assert r.validate_activity(raw)
    for kernel in raw['activity']['kernels']:
        if kernel['name']==r.CONV:kernel['grid']=[16,1,1]
    with pytest.raises(ValueError,match='geometry'):r.validate_activity(raw)


def test_Q4_main_geometry_is_read_from_protocol(tmp_path,monkeypatch):
    raw=activity(tmp_path);p=deepcopy(r.protocol());p['expected_main']['block']=[32,2,1]
    monkeypatch.setattr(r,'protocol',lambda:p)
    with pytest.raises(ValueError,match='geometry'):r.validate_activity(raw)


def test_wait_survives_keyboard_interrupt_and_observation_error_until_child_exits(tmp_path):
    class Child:
        pid=123;calls=0
        def wait(self,timeout):
            self.calls+=1
            assert not (tmp_path/'finish.json').exists()
            if self.calls==1:raise KeyboardInterrupt('synthetic user interruption')
            if self.calls==2:raise OSError('synthetic transient observation error')
            if self.calls==3:raise subprocess.TimeoutExpired('mock',timeout)
            return 0
        def kill(self):pytest.fail('must not kill')
        def terminate(self):pytest.fail('must not terminate')
    child=Child();result=r.wait_naturally(child,100,tmp_path,lambda:1)
    assert child.calls==4 and result['natural_exit'] and result['observation_interrupted'] and result['stop_new_work']
    assert not result['evidence_eligible'] and not result['termination_requested']
    assert [v['exception'] for v in result['observation_interruptions']]==['KeyboardInterrupt','OSError']
    assert len(list(tmp_path.glob('observation_interrupted.*.json')))==2


def test_interrupted_timing_process_finishes_only_after_natural_exit(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'immutable_guard',lambda *_:{'executable':{'path':'synthetic-no-GPU'}})
    monkeypatch.setattr(r,'remaining',lambda *_:1000)
    monkeypatch.setattr(r,'assert_idle',lambda:None)
    monkeypatch.setattr(r,'execution_env',lambda:{})
    class Monitor:
        error=None
        def start(self):pass
        def finish(self):return {'synthetic':True}
    monkeypatch.setattr(r,'NvmlMonitor',Monitor)
    directory=tmp_path/'process'
    class Child:
        pid=123;calls=0
        def wait(self,timeout):
            self.calls+=1;assert not (directory/'finish.json').exists()
            if self.calls==1:raise KeyboardInterrupt('synthetic')
            return 0
    child=Child();monkeypatch.setattr(r.subprocess,'Popen',lambda *a,**k:child)
    result=r.run_process(directory,'formal',{'budget_begin_qpc_ns':r.time.perf_counter_ns(),'prepared_refs':{}},state='main_warm',mode='U')
    assert child.calls==2 and result['status']=='rejected' and result['natural_exit'] and result['observation_interrupted']
    assert (directory/'finish.json').is_file()


def test_campaign_stops_new_slots_after_interrupted_child_natural_exit(tmp_path,monkeypatch):
    frozen_protocol=r.protocol();monkeypatch.setattr(r,'protocol',lambda:frozen_protocol)
    monkeypatch.setattr(r,'P',tmp_path)
    monkeypatch.setattr(r,'verify_review',lambda *_:None)
    monkeypatch.setattr(r,'prepared',lambda:({},{}))
    monkeypatch.setattr(r,'runtime_prerequisites',lambda:{})
    monkeypatch.setattr(r,'ensure_proof',lambda *args:tmp_path/'proof.json')
    calls=[]
    def child(*args):
        calls.append(args);return {'observation_interrupted':True,'natural_exit':True,'identity_unchanged':True}
    monkeypatch.setattr(r,'run_process',child)
    with pytest.raises(ValueError,match='observation interrupted'):r.campaign(tmp_path/'run.0001','run','a'*40)
    assert len(calls)==1 and (tmp_path/'run.0001/complete_05.json').is_file()
