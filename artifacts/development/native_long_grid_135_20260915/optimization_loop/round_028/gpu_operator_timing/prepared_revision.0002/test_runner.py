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
    a=r.pair_metrics(rows);assert a['pair_span_ns']==150 and a['pair_union_ns']==150 and a['interval_overlap_ns']==20 and a['inter_kernel_gap_ns']==0
    rows[1].update(start=180,end=250);a=r.pair_metrics(rows)
    assert a['pair_span_ns']==150 and a['pair_union_ns']==130 and a['inter_kernel_gap_ns']==20 and a['interval_overlap_ns']==0
    assert a['main_in_pair_lifespan_ns']==70


def capability(after):
    calls=[{'name':'counterfactual_latency_enable(1)','returncode':27 if after else 0,'qpc':30,'phase':'counterfactual_latency_api_only'}]
    if after:calls.insert(0,dict(name='cuptiActivityEnableHWTrace(1)',returncode=0,qpc=20,phase='enable_HES_before_context'))
    return {'schema':'r28-gpu-operator-raw/v1','status':'capability_observed','kind':'capability_after_hes' if after else 'capability_without_hes',
        'protocol_ref':r.ref(P/'protocol.json'),'hardware':r.protocol()['hardware'],'timed':False,'GPU_kernel_executed':False,'latency_api_returncode':27 if after else 0,'HES_enable_returncode':0 if after else None,
        'contexts_before':{'primary_active':0,'current_context':0},'contexts_after':{'primary_active':0,'current_context':0},'driver_init':{'qpc_end':10},
        'CUPTI_evidence':{'compile_API_version':26,'runtime_API_version':26,'callback_overflow':False,'STATE':[],'calls':calls}}


def test_counterfactual_needs_both_mode_outcomes():
    assert r.validate_capability(capability(False),False) and r.validate_capability(capability(True),True)
    ambiguous=capability(False);ambiguous['latency_api_returncode']=27
    with pytest.raises(ValueError,match='counterfactual'):r.validate_capability(ambiguous,False)
    silent=capability(True);silent['latency_api_returncode']=0
    with pytest.raises(ValueError,match='counterfactual'):r.validate_capability(silent,True)

@pytest.mark.parametrize('damage',['late','warning','version','context','timestamp_only'])
def test_HES_request_flag_and_nonzero_timestamps_cannot_substitute_mode_proof(damage):
    a=capability(True)
    if damage=='late':a['CUPTI_evidence']['calls'][0]['qpc']=1
    if damage=='warning':a['CUPTI_evidence']['STATE']=[dict(id=3,result=27,phase='formal',message='falling back')]
    if damage=='version':a['CUPTI_evidence']['runtime_API_version']=130401
    if damage=='context':a['contexts_before']['primary_active']=1
    if damage=='timestamp_only':a['CUPTI_evidence']['calls']=[];a['kernel_start']=100;a['kernel_end']=200
    with pytest.raises(ValueError):r.validate_capability(a,True)


def activity(tmp_path,role='pair'):
    path=tmp_path/'raw_buffer.bin';path.write_bytes(b'host synthetic raw fixture')
    trace={'kernel_record_ABI':'CUpti_ActivityKernel9','dropped_records':0,'buffer_overflow':False,'unknown_kinds':[],
        'raw_buffers':[{'ref':r.ref(path),'valid_bytes':path.stat().st_size}], 'kernels':[],'runtime_APIs':[],'external_links':[]}
    corr=1;symbols=[r.CONV,r.MAIN] if role=='pair' else [r.MAIN if role=='main' else r.CONV]
    for i,eid in enumerate(list(range(1000000,1000032))+list(range(1,65))):
        for j,symbol in enumerate(symbols):
            start=10000+i*1000+j*20;end=start+(50 if symbol==r.CONV else 100)
            trace['kernels'].append(dict(kind=10,name=symbol,start=start,end=end,correlation=corr,context=4,stream=5,device=0,graph_id=0,graph_node_id=0,dynamic_shared=0,
                grid=[16,1,1] if symbol==r.CONV else [3072,1,1],block=[256,1,1] if symbol==r.CONV else [32,4,1]))
            trace['runtime_APIs'].append(dict(cbid=430,start=start-10,end=start+5,correlation=corr,return_value=0))
            trace['external_links'].append(dict(kind=3,external_id=eid,correlation=corr));corr+=1
    return {'activity':trace,'state':{'role':role}}

@pytest.mark.parametrize('role',['pair','main','convert'])
def test_64_formal_calls_are_separate_from32_priming_and_throughput(tmp_path,role):
    values=r.validate_activity(activity(tmp_path,role));assert len(values)==64 and values[0]['sample']==0 and values[-1]['sample']==63
    if role=='pair':assert values[0]['interval_overlap_ns']==30 and values[0]['pair_span_ns']==120

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
        if role=='pair':values=dict(pair_span_ns=100,pair_union_ns=100,conversion_lifespan_ns=50,main_in_pair_lifespan_ns=70,interval_overlap_ns=20,inter_kernel_gap_ns=0,host_late_submission_lower_bound_ns=0)
        elif role=='main':values={'independent_main_ns':100}
        else:values={'independent_conversion_ns':30}
        result.append({**item,'status':'valid','summary':{'fixture_fingerprints':{'input':'host_test_same_fixture'},'wall_ns':1000,'event_ns':200 if item['mode'] in ('A','AB') else None,
            'kernel_metrics':[dict(sample=i,**values) for i in range(64)] if item['mode'] in ('B','AB') else []}})
    return result


def test_all5_states_and_two_controls_are_retained():
    a=r.assess_blocks(rows(),5);assert a['formal_success'] and a['qualified_states']==5 and not a['extension_eligible']
    assert all(s['required_processes']==20 and s['cost_model_parameters_emitted'] is False for s in a['states'])


def test_interval_only_inconclusive_allows_unique_predeclared_extension():
    values=rows();target=next(v for v in values if v['state']=='pair_warm' and v['mode']=='B' and v['block']==1);target['summary']['wall_ns']=940
    a=r.assess_blocks(values,5);assert not a['formal_success'] and a['extension_eligible'] and a['states'][0]['reasons']==['perturbation_interval_inconclusive']


def test_observed_median_failure_must_not_be_resampled_until_passing():
    values=rows()
    for v in values:
        if v['state']=='pair_warm' and v['mode']=='B':v['summary']['wall_ns']=1080
    a=r.assess_blocks(values,5);assert not a['extension_eligible'] and 'observed_median_perturbation_exceeds_5pct' in a['states'][0]['reasons']


def test_worst_block_is_not_silently_dropped():
    values=rows();target=next(v for v in values if v['state']=='main_warm' and v['mode']=='B' and v['block']==1)
    for k in target['summary']['kernel_metrics']:k['independent_main_ns']=200
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
