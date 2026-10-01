from dataclasses import replace
from itertools import product
import pytest
from test_cost_models import gpu_profile
from heterollm_sim.cost_models import GemmWorkload, HBMProfile, FusedAttentionWorkload, estimate_gpu_gemm, estimate_gpu_fused_attention
from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile, KernelSample, performance_surface, kernel_model_from_dict
from heterollm_sim.kernel_memory import attach_l2_contract
from heterollm_sim.cache_state import CacheAccess
from heterollm_sim.contracts import TaskSpec, TaskCategory, ResourceDemand
from heterollm_sim.event_kernel import UnifiedEventKernel, CompiledGraphLayout
from heterollm_sim.serde import to_primitive


def capability(**changes):
    return replace(KernelCapability('cuda_mmvq_q4_k', ('q4_k',), 'fp16', 'decode', 'dp4a', 'synthetic test only'), **changes)


def profile(*kernels, **changes):
    return replace(KernelModelProfile('test_gpu', 'test_runtime', 'test_arch', tuple(kernels)), **changes)


def work(**changes):
    return replace(GemmWorkload(1, 1024, 1024, activation_bits=16, weight_bits=4,
                   packed_weight_formats=('q4_k',), execution_phase='decode'), **changes)


def test_kernel_dispatch_phase_format_and_shape():
    model = profile(capability(max_shape=(4, 65536, 32768)), capability(kernel_family='cuda_mmq_q4_k', phase='prefill', compute_primitive='tensor'))
    gpu = replace(gpu_profile(), kernel_model=model)
    decode = estimate_gpu_gemm(gpu, HBMProfile(1000), work())
    prefill = estimate_gpu_gemm(gpu, HBMProfile(1000), work(m=128, execution_phase='prefill'))
    assert decode.metadata['kernel_model']['kernel_family'] == 'cuda_mmvq_q4_k'
    assert prefill.metadata['kernel_model']['kernel_family'] == 'cuda_mmq_q4_k'
    assert estimate_gpu_gemm(gpu, HBMProfile(1000), work(packed_weight_formats=('iq4_xs',))).metadata['model'] != 'kernel_aware'
    assert estimate_gpu_gemm(gpu, HBMProfile(1000), work(m=8)).metadata['model'] != 'kernel_aware'
    assert estimate_gpu_gemm(gpu, HBMProfile(1000), work(execution_phase='unspecified')).metadata['model'] != 'kernel_aware'
    with pytest.raises(ValueError, match='ambiguous'):
        estimate_gpu_gemm(replace(gpu, kernel_model=profile(capability(), capability())), HBMProfile(1000), work())


def test_occupancy_and_bandwidth_not_global_efficiency():
    gpu = replace(gpu_profile(), kernel_model=profile(capability()))
    first = estimate_gpu_gemm(gpu, HBMProfile(1000), work())
    highreg = replace(gpu, kernel_model=profile(capability(registers_per_thread=128)))
    second = estimate_gpu_gemm(highreg, HBMProfile(1000), work())
    assert second.metadata['kernel_model']['occupancy']['occupancy'] < first.metadata['kernel_model']['occupancy']['occupancy']
    assert second.service_ns > first.service_ns
    small = estimate_gpu_gemm(gpu, HBMProfile(1000), work(n=16))
    assert small.metadata['kernel_model']['achieved_bandwidth_gb_s'] < 1000
    with pytest.raises(ValueError, match='cannot reside'):
        profile().occupancy(capability(shared_memory_per_cta=999999))


def test_surface_complete_cells_and_no_silent_extrapolation():
    samples = tuple(KernelSample(m,n,k, 100*m, 50*m, 5, 10, 'synthetic') for m,n,k in product((1,4),(1024,4096),(1024,4096)))
    kernel = capability(samples=samples)
    predicted, _, info = performance_surface(kernel, (2,2048,2048), 70)
    assert predicted == pytest.approx(140)
    assert info['model'] == 'calibrated_analytical'
    exact, _, info = performance_surface(kernel, (1,1024,1024), 999)
    assert exact == pytest.approx(1998)
    assert info['model'] == 'calibrated_analytical'
    outside, _, info = performance_surface(kernel, (8,2048,2048), 70)
    assert outside == 70 and info['reason'] == 'outside_calibration_domain'
    assert info['distance_to_calibration_domain'] > 0
    _, _, info = performance_surface(replace(kernel,samples=samples[:-1]), (2,2048,2048),70)
    assert info['reason'] == 'joint_cell_not_measured'


def test_measured_replaces_high_analytical_wall():
    sample = KernelSample(1,1024,1024, 100, 10000, 7, 20, 'synthetic')
    gpu = replace(gpu_profile(),kernel_model=profile(capability(samples=(sample,), surface_model='measured_surrogate')))
    estimate = estimate_gpu_gemm(gpu, HBMProfile(1000), work())
    assert estimate.service_ns == pytest.approx(100)
    assert estimate.metadata['prediction']['uncertainty_ns'] == 7
    assert estimate.metadata['prediction']['validated_llm_scope'] is False


def test_shared_scalar_work_is_not_unphysical_max():
    gpu = replace(gpu_profile(),kernel_model=profile(capability(unpack_ops_per_weight=2,scale_ops_per_weight=1,reduction_ops_per_output=5)))
    estimate = estimate_gpu_gemm(gpu, HBMProfile(100000), work())
    pipe = estimate.metadata['kernel_model']['pipeline_ns']
    demand = next(d for d in estimate.phases[-1].demands if d.resource_id == gpu.scalar_resource_id)
    assert demand.service_ns == pytest.approx(pipe['unpack']+pipe['scale']+pipe['dot']+pipe['reduce'])


def test_attention_decode_context_and_distinct_prefill():
    gpu = replace(gpu_profile(),kernel_model=profile(
        capability(operator='attention',weight_formats=('fp16',),kernel_family='decode_attention',compute_primitive='simt'),
        capability(operator='attention',weight_formats=('fp16',),kernel_family='flash_prefill',phase='prefill',compute_primitive='tensor',internal_dtype='fp16')))
    w = FusedAttentionWorkload(1,1024,128,kv_hidden_size=32,execution_phase='decode')
    a = estimate_gpu_fused_attention(gpu,HBMProfile(1000),w)
    b = estimate_gpu_fused_attention(gpu,HBMProfile(1000),replace(w,context_tokens=32768))
    assert b.service_ns > a.service_ns
    assert b.metadata['kernel_model']['read_bytes'] > a.metadata['kernel_model']['read_bytes'] * 30
    c = estimate_gpu_fused_attention(gpu,HBMProfile(1000),replace(w,batch_tokens=128,execution_phase='prefill'))
    assert c.metadata['kernel_model']['attention_model'] == 'flash_prefill'


def test_profile_roundtrip_and_validation():
    model = profile(capability(samples=(KernelSample(1,1024,1024,100,120,2,3,'synthetic'),)))
    assert kernel_model_from_dict(to_primitive(model)) == model
    with pytest.raises(ValueError):
        kernel_model_from_dict({**to_primitive(model),'typo':1})
    with pytest.raises(ValueError):
        KernelSample(1,1,1,float('nan'),1,0,1,'bad')
    with pytest.raises(ValueError):
        profile(launch_ns=100)


def l2_task(ident, buffer, deps=(), operation='read'):
    gpu = gpu_profile()
    level = replace(gpu.cache_hierarchy.levels[-1],capacity_bytes=128,line_bytes=64)
    gpu = replace(gpu,cache_hierarchy=replace(gpu.cache_hierarchy,levels=(level,)))
    task = TaskSpec(ident,'r',ident,TaskCategory.COMPUTE,dependencies=deps,
                    demands=(ResourceDemand('hbm',128,bytes_moved=128),ResourceDemand('gpu.scalar',1)))
    return attach_l2_contract(task,gpu=gpu,hbm=HBMProfile(1,resource_id='hbm'),
            memory_resource='hbm',cache_resource='l2',owner='gpu.l2',
            accesses=(CacheAccess(buffer,0,128,operation),))


def drain(kernel):
    events=[]
    while kernel.has_active_tasks:
        events.append(kernel.step())
    return events


def test_l2_state_reuse_distance_writeback_and_isolation():
    tasks=(l2_task('a','w'),l2_task('b','w',('a',)),l2_task('c','other',('b',)),l2_task('d','w',('c',)))
    events=drain(UnifiedEventKernel.from_closed_graph(tasks))
    assert [e.task.metadata['l2_execution']['hit_lines'] for e in events] == [0,2,0,0]
    assert events[1].task.metadata['l2_execution']['hbm_read_bytes'] == 0
    again=drain(UnifiedEventKernel.from_closed_graph(tasks))
    assert again[0].task.metadata['l2_execution']['hit_lines'] == 0
    writes=drain(UnifiedEventKernel.from_closed_graph((l2_task('x','dirty',operation='write'),l2_task('y','other',('x',)))))
    assert writes[1].task.metadata['l2_execution']['dirty_eviction_bytes'] == 128


def test_compiled_and_incremental_l2_identical_and_persistent():
    tasks=(l2_task('a','w'),l2_task('b','w',('a',)))
    a=drain(UnifiedEventKernel.from_closed_graph(tasks))
    k=UnifiedEventKernel()
    b=k._drain_prevalidated_compiled(tasks,CompiledGraphLayout.compile(tasks))
    assert [(e.start_ns,e.end_ns) for e in a] == [(e.start_ns,e.end_ns) for e in b]
    k.submit((l2_task('c','w'),))
    assert k.step().task.metadata['l2_execution']['hit_lines'] == 2


def test_des_overlap_obeys_dependency_and_resource_identity():
    compute=TaskSpec('compute','r','compute',TaskCategory.COMPUTE,demands=(ResourceDemand('gpu.compute',100),ResourceDemand('hbm.read',40)))
    tx=TaskSpec('tx','r','tx',TaskCategory.COMMUNICATION,demands=(ResourceDemand('nvlink.tx',80),ResourceDemand('dma',20)))
    rx=replace(tx,task_id='rx',demands=(ResourceDemand('nvlink.rx',80),))
    events=drain(UnifiedEventKernel.from_closed_graph((compute,tx,rx)))
    assert max(e.end_ns for e in events)==100
    events=drain(UnifiedEventKernel.from_closed_graph((compute,replace(tx,dependencies=('compute',)))))
    assert max(e.end_ns for e in events)==180

def test_scenario_lowering_uses_kernel_costs_and_l2_contracts():
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.planner import compile_scenario
    scenario=build_reference_scenario()
    groups={kind:dict(values) for kind,values in scenario.component_profiles.items()}
    for key,gpu in groups['gpu'].items():
        kernels=tuple(capability(phase=phase,weight_formats=('int8',),activation_dtype='int8',compute_primitive='tensor',output_bits=16,kernel_family=phase+'_int8') for phase in ('prefill','decode'))
        groups['gpu'][key]=replace(gpu,kernel_model=profile(*kernels,stateful_l2=True))
    scenario=replace(scenario,component_profiles=groups)
    schedule=compile_scenario(scenario)
    selected=[t for t in schedule.tasks if t.metadata.get('cost_model',{}).get('model')=='kernel_aware']
    assert selected
    assert {t.metadata['cost_model']['kernel_model']['phase'] for t in selected} <= {'prefill','decode'}
    assert any('stateful_l2' in t.metadata for t in selected)
    kernel=UnifiedEventKernel.from_closed_graph(schedule.tasks)
    events=drain(kernel)
    assert any('l2_execution' in e.task.metadata for e in events)

def test_graph_launch_charged_once_only_for_explicit_capture():
    from heterollm_sim.kernel_model import apply_captured_graph_launch
    model=profile(graph_enabled=True,launch_ns=100,graph_launch_ns=25,launch_evidence='synthetic')
    tasks=tuple(TaskSpec(str(i),'r',str(i),TaskCategory.COMPUTE,
                demands=(ResourceDemand('frontend',100),),
                metadata={'phase':'kernel_launch','cuda_graph_id':'replay-1','cuda_graph_captured':True}) for i in range(3))
    revised=apply_captured_graph_launch(tasks,model)
    assert [t.demands[0].service_ns for t in revised] == [25,0,0]
    plain=tuple(replace(t,metadata={'phase':'kernel_launch'}) for t in tasks)
    assert apply_captured_graph_launch(plain,model)==plain


def test_fallback_exposes_low_confidence_not_silent_dispatch():
    gpu=replace(gpu_profile(),kernel_model=profile(capability()))
    estimate=estimate_gpu_gemm(gpu,HBMProfile(100),work(packed_weight_formats=('q4_0',)))
    assert estimate.metadata['prediction']['reason']=='kernel_dispatch_not_covered'
    assert estimate.metadata['prediction']['confidence']=='low'
    assert estimate.metadata['prediction']['fallback_kind']=='legacy_analytical'


def test_unknown_quantization_format_is_explicitly_marked_as_legacy_fallback():
    gpu=replace(gpu_profile(),kernel_model=profile(capability()))
    estimate=estimate_gpu_gemm(gpu,HBMProfile(100),work(packed_weight_formats=('mystery_q9',)))
    prediction=estimate.metadata['prediction']
    assert prediction['reason']=='unsupported_quantization_format'
    assert prediction['format_coverage']=='unsupported'
    assert prediction['unsupported_weight_formats']==('mystery_q9',)
    assert prediction['fallback_kind']=='legacy_analytical'


def test_missing_kernel_profile_is_explicitly_marked_as_legacy_fallback():
    estimate=estimate_gpu_gemm(gpu_profile(),HBMProfile(100),work())
    prediction=estimate.metadata['prediction']
    assert prediction['reason']=='kernel_model_profile_unavailable'
    assert prediction['format_coverage']=='profile_unavailable'
    assert prediction['fallback_kind']=='legacy_analytical'


def test_mixed_quantization_formats_are_explicitly_unresolved():
    gpu=replace(gpu_profile(),kernel_model=profile(capability()))
    estimate=estimate_gpu_gemm(gpu,HBMProfile(100),work(packed_weight_formats=('q4_k','q6_k')))
    prediction=estimate.metadata['prediction']
    assert prediction['reason']=='mixed_quantization_dispatch_unresolved'
    assert prediction['format_coverage']=='mixed_unresolved'

def test_online_kernel_path_preserves_dynamic_cache_handoff():
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.reporting import run_scenario, report_dict
    scenario=build_reference_scenario()
    groups={kind:dict(values) for kind,values in scenario.component_profiles.items()}
    for key,gpu in groups['gpu'].items():
        groups['gpu'][key]=replace(gpu,kernel_model=profile(stateful_l2=True))
    request=replace(scenario.workload.requests[0],prompt_tokens=2,output_tokens=2)
    scenario=replace(scenario,component_profiles=groups,workload=replace(scenario.workload,requests=(request,),mtp=None))
    result=run_scenario(scenario)
    output=report_dict(result)
    assert output['summary']['makespan_ns'] > 0


def test_explicit_buffers_cover_all_required_llm_buffer_kinds():
    from heterollm_sim.kernel_memory import resolve_l2_task
    for kind in ('weights','kv','activation','rope_table','norm_weight','logits','sampling'):
        first=l2_task('read-'+kind,kind)
        second=l2_task('reuse-'+kind,kind)
        state={}
        resolve_l2_task(first,state)
        hit=resolve_l2_task(second,state)
        assert hit.metadata['l2_execution']['hit_lines']==2


def test_graph_replay_zero_cost_nodes_wait_for_submission():
    from heterollm_sim.kernel_model import apply_captured_graph_launch
    model=profile(graph_enabled=True,launch_ns=100,graph_launch_ns=25,launch_evidence='synthetic')
    tasks=tuple(TaskSpec(str(i),'r',str(i),TaskCategory.COMPUTE,demands=(ResourceDemand('frontend',100),),
                metadata={'phase':'kernel_launch','cuda_graph_id':'one','cuda_graph_captured':True}) for i in range(3))
    events=drain(UnifiedEventKernel.from_closed_graph(apply_captured_graph_launch(tasks,model)))
    assert all(e.end_ns >= 25 for e in events)
    assert max(e.end_ns for e in events)==25

def test_raw_microbenchmark_import_recomputes_statistics_and_rejects_llm_fit(tmp_path):
    import hashlib,json
    from heterollm_sim.kernel_model import load_microbenchmark_surface
    k=capability()
    p=profile(k)
    doc={'schema':'heterollm.kernel-microbenchmark/v1','source_kind':'independent_synthetic_operator',
         'target_llm_latency_used':False,'measurement_boundary':'cuda_device_kernel_interval',
         'hardware_id':p.hardware_id,'runtime_id':p.runtime_id,'architecture':p.architecture,
         'kernel_family':k.kernel_family,'phase':k.phase,'weight_format':'q4_k',
         'activation_dtype':k.activation_dtype,'layout':k.layout,'output_bits':k.output_bits,
         'accumulator_bits':k.accumulator_bits,'epilogue':'','cache_protocol':k.cache_protocol,
         'samples':[{'m':1,'n':1024,'k':1024,'device_durations_ns':[90,100,110],
                     'hbm_bytes':100000,'hbm_durations_ns':[100,100,100]}]}
    path=tmp_path/'samples.json'
    def save():
        path.write_text(json.dumps(doc),encoding='utf-8')
        return hashlib.sha256(path.read_bytes()).hexdigest()
    loaded=load_microbenchmark_surface(path,expected_sha256=save(),profile=p,kernel=k,analytical_cost=lambda *args:200)
    assert loaded.samples[0].device_ns==100
    assert loaded.samples[0].stddev_ns==10
    assert loaded.samples[0].achieved_bandwidth_gb_s==1000
    # Preserve observed specialization through import; otherwise the importer
    # silently erases the dispatch boundary and permits unsafe interpolation.
    doc['samples'][0]['dispatch_signature'] = 'block128:regs53'
    doc['samples'].append({**doc['samples'][0], 'n':4096,
                           'dispatch_signature':'block128:regs80'})
    loaded=load_microbenchmark_surface(path,expected_sha256=save(),profile=p,kernel=k,analytical_cost=lambda *args:200)
    assert loaded.samples[0].dispatch_signature == 'block128:regs53'
    _,_,prediction = performance_surface(loaded,(1,2048,1024),200)
    assert prediction['reason'] == 'kernel_specialization_boundary'
    doc['target_llm_latency_used']=True
    with pytest.raises(ValueError,match='identity mismatch'):
        load_microbenchmark_surface(path,expected_sha256=save(),profile=p,kernel=k,analytical_cost=lambda *args:200)


def test_dtype_epilogue_and_runtime_isolation():
    gpu=replace(gpu_profile(),kernel_model=profile(capability()))
    assert estimate_gpu_gemm(gpu,HBMProfile(100),work(activation_dtype='bf16')).metadata['prediction']['model']=='analytical'
    assert estimate_gpu_gemm(gpu,HBMProfile(100),work(output_bits=32)).metadata['prediction']['model']=='analytical'
    assert estimate_gpu_gemm(gpu,HBMProfile(100),work(epilogue_name='silu',epilogue_operations=1024)).metadata['prediction']['model']=='analytical'


def test_mmvq_level2_surface_changes_service_ns_only_with_source_binding():
    from test_mmvq_work import contract
    from heterollm_sim.mmvq_work import derive_mmvq_work
    from heterollm_sim.kernel_model import mmvq_calibration_dispatch_signature

    def source(n):
        return derive_mmvq_work(m=1, n=n, k=1024, weight_format='Q4_K',
                                contract=contract(), allow_k_formats=True)

    anchor = source(1024)
    signature = mmvq_calibration_dispatch_signature(
        type('Workload', (), {'mmvq_work': anchor})())
    samples = (
        KernelSample(1, 1024, 1024, 100, 200, 2, 3, 'synthetic',
                     dispatch_signature=signature),
        KernelSample(1, 2048, 1024, 200, 400, 2, 3, 'synthetic',
                     dispatch_signature=signature),
    )
    descriptor = KernelCapability(
        'cuda_mmvq_q4_k', ('q4_k',), 'fp32', 'decode', 'dp4a',
        'synthetic source-bound test', output_bits=32,
        min_shape=(1, 1024, 1024), max_shape=(1, 2048, 1024),
        samples=samples, calibration_source_bound=True,
        cache_protocol='cold_streaming')
    model = KernelModelProfile(
        'test_gpu', 'test_runtime', 'test_arch', (descriptor,),
        calibration_runtime_sha256=anchor.runtime_binary_sha256)

    mid = source(1536)
    workload = GemmWorkload(
        1, 1024, 1536, activation_bits=32, weight_bits=4, output_bits=32,
        packed_weight_formats=('q4_k',), execution_phase='decode',
        activation_storage_bytes=mid.consumer_q8_1_unique_bytes,
        weight_storage_bytes=mid.logical_weight_bytes,
        output_storage_bytes=mid.output_f32_bytes, mmvq_work=mid)
    estimate = estimate_gpu_gemm(
        replace(gpu_profile(), kernel_model=model), HBMProfile(1000), workload)
    prediction = estimate.metadata['prediction']
    assert prediction['model'] == 'calibrated_analytical'
    assert prediction['reason'] == 'joint_grid_interpolation'
    assert estimate.service_ns != pytest.approx(
        estimate_gpu_gemm(replace(gpu_profile(), kernel_model=replace(model, kernels=(replace(descriptor, samples=(), calibration_source_bound=False),))),
                          HBMProfile(1000), workload).service_ns)

    unbound = replace(descriptor, calibration_source_bound=False)
    blocked = estimate_gpu_gemm(
        replace(gpu_profile(), kernel_model=replace(model, kernels=(unbound,))),
        HBMProfile(1000), workload)
    assert blocked.metadata['prediction']['model'] == 'analytical'
    assert blocked.metadata['prediction']['reason'] == 'source_calibration_binding_missing'


def test_l2_records_offsets_and_does_not_alias_other_buffers():
    from heterollm_sim.kernel_memory import resolve_l2_task
    task=l2_task('a','kv')
    c=dict(task.metadata['stateful_l2'])
    c['accesses']=({'buffer_id':'kv','offset_bytes':64,'size_bytes':64,'operation':'read'},)
    first=replace(task,metadata={**task.metadata,'stateful_l2':c})
    state={}; resolve_l2_task(first,state)
    c2={**c,'accesses':({'buffer_id':'kv','offset_bytes':0,'size_bytes':64,'operation':'read'},)}
    second=resolve_l2_task(replace(first,task_id='b',metadata={'stateful_l2':c2}),state)
    assert second.metadata['l2_execution']['hit_lines']==0

def test_llama_blackwell_preset_dispatch_and_no_fake_measurements():
    from heterollm_sim.kernel_model import llama_blackwell_analytical_profile
    p=llama_blackwell_analytical_profile('rtx5080','runtime')
    assert all(not k.samples for k in p.kernels)
    assert p.registers_per_sm == 65536
    assert p.shared_memory_per_sm == 102400
    assert p.max_ctas_per_sm == 24
    assert p.max_threads_per_sm == p.max_warps_per_sm * p.warp_size == 1536
    assert 'cuDeviceGetAttribute' in p.hardware_limits_evidence
    assert 'attainable_efficiency' in p.unverified_parameters
    def select(m,fmt):
        return p.dispatch(shape=(m,1024,1024),formats=(fmt,),dtype='fp16',phase='decode',output_bits=16,accumulator_bits=32,epilogue='')
    assert 'mmvq' in select(5,'q4_k').kernel_family
    assert 'mmq' in select(6,'q4_k').kernel_family
    assert 'mmvq' in select(8,'iq4_xs').kernel_family
    assert select(1,'q4_0').kernel_family != select(1,'q4_k').kernel_family
    assert select(1,'unknown') is None

def test_paged_kv_physical_aliases_generation_and_bounds():
    from heterollm_sim.kernel_memory import paged_buffer_accesses
    a=paged_buffer_accesses(buffer_id='kv',page_ids=(7,3),tokens_per_page=16,bytes_per_token=32,first_token=14,token_count=5)
    assert [(x.offset_bytes,x.size_bytes) for x in a]==[(448,64),(0,96)]
    assert a[0].buffer_id.endswith('page:7') and a[1].buffer_id.endswith('page:3')
    b=paged_buffer_accesses(buffer_id='kv',page_ids=(7,),tokens_per_page=16,bytes_per_token=32,first_token=14,token_count=2)
    assert a[0]==b[0]
    c=paged_buffer_accesses(buffer_id='kv',page_ids=(7,),tokens_per_page=16,bytes_per_token=32,first_token=14,token_count=2,allocation_generation=1)
    assert c[0].buffer_id!=a[0].buffer_id
    with pytest.raises(ValueError):
        paged_buffer_accesses(buffer_id='kv',page_ids=(7,),tokens_per_page=16,bytes_per_token=32,first_token=14,token_count=3)


def test_attention_dispatch_separates_gqa_storage_signatures():
    kernels=tuple(capability(operator='attention',weight_formats=('fp16',),attention_kv_hidden_size=w,attention_score_heads=8) for w in (32,64))
    p=profile(*kernels)
    selected=p.dispatch(shape=(1,1024,128),formats=('fp16',),dtype='fp16',phase='decode',operator='attention',attention_kv_hidden_size=32,attention_score_heads=8)
    assert selected.attention_kv_hidden_size==32

def test_causal_attention_tiles_keep_padding_and_skip_future_blocks():
    from heterollm_sim.kernel_model import attention_tile_work
    w=FusedAttentionWorkload(128,128,64,causal_query_positions=tuple(range(128)))
    work=attention_tile_work(w)
    assert work['useful_pairs']==128*129//2
    assert work['executed_pairs']==64*64+64*128
    assert work['causal_tiles_skipped']==1
    with pytest.raises(ValueError):
        replace(w,causal_query_positions=(0,)*128)
    assert not attention_tile_work(replace(w,causal_query_positions=()))['causal_positions_bound']

def test_surface_does_not_interpolate_across_observed_specializations():
    samples=(KernelSample(1,1024,1024,100,100,2,20,'trace-a',dispatch_signature='small-k'),
             KernelSample(1,1024,4096,200,200,2,20,'trace-b',dispatch_signature='large-k'))
    k=capability(samples=samples, surface_model='measured_surrogate')
    ns,_,info=performance_surface(k,(1,1024,2048),150)
    assert ns==150 and info['reason']=='kernel_specialization_boundary'
    assert performance_surface(k,(1,1024,1024),150)[0]==100

def test_l2_memory_completion_does_not_lock_unrelated_compute_tail():
    a=l2_task('a','weights')
    a=replace(a,demands=tuple(replace(d,resource_id='compute.a',service_ns=10000) if d.resource_id=='gpu.scalar' else d for d in a.demands))
    b=l2_task('b','weights')
    b=replace(b,demands=tuple(replace(d,resource_id='compute.b') if d.resource_id=='gpu.scalar' else d for d in b.demands))
    events=drain(UnifiedEventKernel.from_closed_graph((a,b)))
    first,second=events
    assert second.task.metadata['l2_execution']['hit_lines']==2
    assert second.start_ns >= first.task.metadata['l2_execution']['memory_completion_offset_ns']
    assert second.start_ns < first.end_ns

def test_measured_hot_cache_surface_cannot_price_default_cold_workload():
    k=capability(cache_protocol='hot', surface_model='measured_surrogate', samples=(KernelSample(1,1024,1024,100,100,2,20,'synthetic'),))
    gpu=replace(gpu_profile(),kernel_model=profile(k))
    cold=estimate_gpu_gemm(gpu,HBMProfile(1000),work())
    assert cold.metadata['prediction']['model']=='analytical'
    assert cold.metadata['prediction']['reason']=='measurement_cache_protocol_mismatch'
    hot=estimate_gpu_gemm(gpu,HBMProfile(1000),work(cache_protocol='hot'))
    assert hot.metadata['prediction']['model']=='measured_surrogate'
    assert hot.service_ns==pytest.approx(100)

def test_attention_mask_dispatch_changes_resource_descriptor():
    base = capability(operator='attention', phase='prefill', weight_formats=('fp16',),
                      compute_primitive='tensor')
    plain = replace(base, kernel_family='unmasked', attention_mask='none',
                    registers_per_thread=209, shared_memory_per_cta=17408)
    causal = replace(base, kernel_family='causal', attention_mask='causal',
                     registers_per_thread=230, shared_memory_per_cta=20992)
    gpu = replace(gpu_profile(), kernel_model=profile(plain, causal))
    workload = FusedAttentionWorkload(64, 1024, 256, score_heads=4,
                                     execution_phase='prefill')
    unmasked = estimate_gpu_fused_attention(gpu, HBMProfile(1000), workload)
    masked = estimate_gpu_fused_attention(gpu, HBMProfile(1000), replace(
        workload, causal_query_positions=tuple(range(960, 1024))))
    assert unmasked.metadata['kernel_model']['kernel_family'] == 'unmasked'
    assert masked.metadata['kernel_model']['kernel_family'] == 'causal'
    parsed = kernel_model_from_dict(to_primitive(gpu.kernel_model))
    assert parsed.kernels[1].attention_mask == 'causal'
    with pytest.raises(ValueError, match='attention_mask'):
        replace(causal, attention_mask='guessed')

def test_streamk_attention_cost_has_ordered_fixup_and_two_launches():
    descriptor = capability(operator='attention', phase='prefill', weight_formats=('fp16',),
        output_bits=32, internal_dtype='fp16', compute_primitive='tensor',
        attention_stream_k=True, registers_per_thread=64)
    gpu = replace(gpu_profile(), kernel_model=profile(descriptor, launch_ns=100, launch_evidence='test'))
    workload = FusedAttentionWorkload(64, 2048, 256, output_bits=32, score_heads=4,
        kv_hidden_size=128, execution_phase='prefill')
    estimate = estimate_gpu_fused_attention(gpu, HBMProfile(1000), workload)
    assert estimate.phase_names == ('kernel_launch', 'gpu_fused_attention', 'kernel_launch', 'gpu_attention_stream_k_fixup')
    assert estimate.metadata['kernel_count'] == 2
    main, fix = estimate.phases[1], estimate.phases[3]
    assert main.metadata['kernel_model']['stream_k']['fixup_required']
    assert fix.metadata['persistent_kv_read_bytes'] == 0
    traffic = main.metadata['kernel_model']['scratch_traffic']
    assert main.metadata['kernel_model']['write_bytes'] == workload.write_bytes + traffic['scratch_write_bytes']
    assert fix.metadata['kernel_model']['read_bytes'] == workload.write_bytes + traffic['scratch_read_bytes']
    assert traffic['scratch_write_bytes'] < traffic['allocation_bytes']
    assert next(d for d in fix.demands if d.resource_id == HBMProfile(1000).resource_id).bytes_moved == (workload.write_bytes * 2 + traffic['scratch_read_bytes'])
    assert fix.metadata['kernel_model']['write_bytes'] == workload.write_bytes
    tasks = tuple(TaskSpec(f'phase{i}', 'request', phase.name, phase.category, demands=phase.demands,
                           dependencies=(f'phase{i-1}',) if i else ())
                  for i, phase in enumerate(estimate.phases))
    events = drain(UnifiedEventKernel.from_closed_graph(tasks))
    assert events[-1].end_ns == pytest.approx(estimate.service_ns)
    assert events[-1].start_ns >= events[-2].end_ns
    assert kernel_model_from_dict(to_primitive(gpu.kernel_model)).kernels[0].attention_stream_k
    with pytest.raises(ValueError, match='stream-K'):
        replace(descriptor, samples=(KernelSample(64,2048,256,100,100,1,20,'test'),))


def test_streamk_rejects_non_divisible_score_heads_before_tile_accounting():
    descriptor = capability(operator='attention', phase='prefill', weight_formats=('fp16',),
        output_bits=32, internal_dtype='fp16', compute_primitive='tensor',
        attention_stream_k=True, registers_per_thread=64, attention_heads_per_tile=2)
    base = gpu_profile()
    gpu = replace(base, tensor_core=replace(base.tensor_core, sm_count=100),
                  kernel_model=profile(descriptor))
    workload = FusedAttentionWorkload(64, 2048, 192, kv_hidden_size=96,
        output_bits=32, score_heads=3, execution_phase='prefill')

    estimate = estimate_gpu_fused_attention(gpu, HBMProfile(1000), workload)

    assert estimate.metadata['model'] != 'kernel_aware'
    assert 'stream_k' not in estimate.metadata.get('kernel_model', {})

def test_uniform_fixup_traffic_separates_allocation_writes_and_reads():
    from heterollm_sim.kernel_model import attention_uniform_fixup_traffic
    traffic = attention_uniform_fixup_traffic(blocks=96, output_tiles=4,
        columns_per_tile=64, live_rows=256, head_dim=64)
    assert traffic['allocation_bytes'] == 96 * 64 * 272
    assert traffic['scratch_write_bytes'] == 92 * 64 * 256 + 96 * 64 * 8
    assert traffic['scratch_read_bytes'] == 256 * (23 * 256 + 24 * 8)
    assert traffic['scratch_write_bytes'] == traffic['scratch_read_bytes']
    padded = attention_uniform_fixup_traffic(blocks=96, output_tiles=4,
        columns_per_tile=64, live_rows=132, head_dim=64)
    assert padded['scratch_write_bytes'] == traffic['scratch_write_bytes']
    assert padded['scratch_read_bytes'] < traffic['scratch_read_bytes']
    assert padded['output_bytes'] == 132 * 256
    unsplit = attention_uniform_fixup_traffic(blocks=4, output_tiles=4,
        columns_per_tile=64, live_rows=256, head_dim=64)
    assert unsplit['scratch_write_bytes'] == unsplit['scratch_read_bytes'] == 0
    with pytest.raises(ValueError):
        attention_uniform_fixup_traffic(blocks=95, output_tiles=4,
            columns_per_tile=64, live_rows=256, head_dim=64)

def test_streamk_scratch_identity_producer_consumer_hits_and_isolation():
    from heterollm_sim.kernel_memory import attention_stream_k_accesses
    descriptor = capability(operator='attention', phase='prefill', weight_formats=('fp16',),
        output_bits=32, internal_dtype='fp16', compute_primitive='tensor',
        attention_stream_k=True, registers_per_thread=64)
    gpu = gpu_profile()
    level = replace(gpu.cache_hierarchy.levels[-1], capacity_bytes=8*1024*1024, line_bytes=64)
    gpu = replace(gpu, kernel_model=profile(descriptor, stateful_l2=True),
                  cache_hierarchy=replace(gpu.cache_hierarchy, levels=(level,)))
    hbm = HBMProfile(1000)
    workload = FusedAttentionWorkload(64,2048,256,output_bits=32,score_heads=4,
                                    kv_hidden_size=128,execution_phase='prefill')
    estimate = estimate_gpu_fused_attention(gpu,hbm,workload)
    stages = [p for p in estimate.phases if 'kernel_model' in p.metadata]
    def task(index, invocation):
        phase = stages[index]
        audit = phase.metadata['kernel_model']
        accesses = attention_stream_k_accesses(audit, invocation, fixup=bool(index))
        item = TaskSpec(str(index),'r',phase.name,TaskCategory.COMPUTE,
                        demands=phase.demands,dependencies=('0',) if index else (),
                        metadata={'phase_metadata':phase.metadata})
        return attach_l2_contract(item,gpu=gpu,hbm=hbm,memory_resource=hbm.resource_id,
            cache_resource=level.resource_id,owner='gpu.l2',accesses=accesses)
    events = drain(UnifiedEventKernel.from_closed_graph((task(0,'call1'),task(1,'call1'))))
    assert events[1].task.metadata['l2_execution']['hbm_read_bytes'] == 0
    assert events[1].task.metadata['l2_execution']['hit_lines'] > 0
    isolated = drain(UnifiedEventKernel.from_closed_graph((task(0,'call1'),task(1,'call2'))))
    assert isolated[1].task.metadata['l2_execution']['hbm_read_bytes'] > 0

@pytest.mark.parametrize('queries', [33, 64])
def test_planner_binds_streamk_scratch_ranges_automatically(queries):
    import heterollm_sim.planner as planner
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.communication import TopologyRouter
    scenario = build_reference_scenario()
    groups = {kind:dict(values) for kind,values in scenario.component_profiles.items()}
    descriptor = capability(operator='attention', phase='prefill', weight_formats=('fp16',),
        output_bits=32, internal_dtype='fp16', compute_primitive='tensor',
        attention_stream_k=True, registers_per_thread=64)
    for key,gpu in groups['gpu'].items():
        groups['gpu'][key] = replace(gpu,kernel_model=profile(descriptor,stateful_l2=True))
    scenario = replace(scenario,component_profiles=groups)
    with planner._compilation_scope(scenario):
        plan = planner._parallel_plan(scenario)
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._add_rank_fused_attention(builder,scenario,TopologyRouter(scenario.hardware),plan,plan.ranks[0],
            FusedAttentionWorkload(queries,2048,256,output_bits=32,score_heads=4,kv_hidden_size=128,execution_phase='prefill'),
            'probe',())
    tasks = [t for t in builder.tasks if t.metadata.get('buffer_accesses')]
    assert len(tasks) == 2
    writer = {a['buffer_id'] for a in tasks[0].metadata['stateful_l2']['accesses'] if a['operation']=='write'}
    reader = {a['buffer_id'] for a in tasks[1].metadata['stateful_l2']['accesses'] if a['operation']=='read'}
    assert writer == reader
    assert tasks[1].metadata['kernel_prediction']['kernel_family'] == 'flash_attn_stream_k_fixup_uniform'


@pytest.mark.parametrize('queries', [1, 31, 33, 63, 65])
def test_streamk_tail_query_ranges_use_padded_block_stride(queries):
    from heterollm_sim.kernel_memory import attention_stream_k_accesses
    descriptor = capability(operator='attention',phase='prefill',weight_formats=('fp16',),
        output_bits=32,internal_dtype='fp16',compute_primitive='tensor',
        attention_stream_k=True,registers_per_thread=64)
    gpu = replace(gpu_profile(),tensor_core=replace(gpu_profile().tensor_core, sm_count=12),kernel_model=profile(descriptor))
    workload = FusedAttentionWorkload(queries,2048,256,output_bits=32,score_heads=4,
                                    kv_hidden_size=128,execution_phase='prefill')
    estimate = estimate_gpu_fused_attention(gpu,HBMProfile(1000),workload)
    stages = [p.metadata['kernel_model'] for p in estimate.phases if 'kernel_model' in p.metadata]
    assert len(stages) == 2
    writes = attention_stream_k_accesses(stages[0],'call')
    reads = attention_stream_k_accesses(stages[1],'call',fixup=True)
    scratch_reads = [a for a in reads if a.buffer_id.endswith(':scratch')]
    scratch_writes = [a for a in writes if a.buffer_id.endswith(':scratch')]
    assert sum(a.size_bytes for a in scratch_reads) == stages[0]['scratch_traffic']['scratch_read_bytes']
    assert sum(a.size_bytes for a in scratch_reads) < sum(a.size_bytes for a in scratch_writes)
    for access in scratch_reads:
        assert any(w.offset_bytes <= access.offset_bytes and
                   access.offset_bytes + access.size_bytes <= w.offset_bytes + w.size_bytes
                   for w in scratch_writes)
    # Tiled query index is the fastest dimension, as in the CUDA decoder.
    tile = stages[0]['stream_k']['query_tile']
    heads = stages[0]['stream_k']['heads_per_tile']
    expected_columns = min(queries,tile) * heads
    assert scratch_reads[0].size_bytes == expected_columns * 8



def test_streamk_nonuniform_partition_still_falls_back():
    descriptor = capability(operator='attention',phase='prefill',weight_formats=('fp16',),
        output_bits=32,internal_dtype='fp16',compute_primitive='tensor',
        attention_stream_k=True,registers_per_thread=64)
    gpu = replace(gpu_profile(),kernel_model=profile(descriptor))
    workload = FusedAttentionWorkload(65,2048,256,output_bits=32,score_heads=4,
                                    kv_hidden_size=128,execution_phase='prefill')
    estimate = estimate_gpu_fused_attention(gpu,HBMProfile(1000),workload)
    assert estimate.metadata['model'] != 'kernel_aware'
    assert 'gpu_attention_stream_k_fixup' not in estimate.phase_names

@pytest.mark.parametrize('route', ['plain','cached','trusted'])
def test_online_replay_rebinds_temporary_l2_buffers_not_persistent_buffers(route):
    from heterollm_sim.serving import _OnlineRuntime, _ExecutionStageReplayLayout
    from heterollm_sim.contracts import _PreparedExecutionTask, _PreparedExecutionStage
    from copy import deepcopy
    tasks = (l2_task('writer','@invocation:scratch',operation='write'),
             l2_task('reader','@invocation:scratch',('writer',)))
    rows = tuple(_PreparedExecutionTask(t.task_id,t.dependencies,('r',),t.demands,metadata=t.metadata) for t in tasks)
    stages = (_PreparedExecutionStage('stage',0,(),('r',),'gpu',100,rows),)
    before = deepcopy(stages)
    layout = _ExecutionStageReplayLayout.compile(stages)
    def instantiate(namespace):
        if route == 'plain':
            return _OnlineRuntime._stage_task_specs(stages,{'stage':0},namespace)[0]
        method = layout.instantiate if route == 'cached' else layout.instantiate_trusted
        return method(stages,{'stage':0},namespace=namespace)
    a,b = instantiate('batch1'),instantiate('batch2')
    def identity(spec):return spec.metadata['stateful_l2']['accesses'][0]['buffer_id']
    assert identity(a[0]) == identity(a[1])
    assert identity(a[0]) != identity(b[0])
    assert stages == before
    # Decode replays share persistent weight allocations, but not temporaries.
    from heterollm_sim.kernel_memory import bind_l2_invocation
    weight = l2_task('weight','gpu:weights:projection')
    assert bind_l2_invocation(weight.metadata,'batch1') is weight.metadata
    kernel = UnifiedEventKernel.from_closed_graph(a)
    events = drain(kernel)
    assert events[1].task.metadata['l2_execution']['hbm_read_bytes'] == 0
    kernel.submit(b)
    events = drain(kernel)
    assert events[0].task.metadata['l2_execution']['miss_lines'] == 2

def test_calibration_domain_distance_is_not_nearest_sample_distance():
    samples = tuple(KernelSample(m,n,k,100,100,1,10,'test')
                    for m,n,k in product((1,4),(1024,4096),(1024,4096)))
    kernel = capability(samples=samples)
    _,_,outside = performance_surface(kernel,(8,2048,2048),100)
    assert outside['distance_to_calibration_domain'] == pytest.approx(1)
    assert outside['nearest_sample_distance'] == pytest.approx(3**.5)
    _,_,inside = performance_surface(kernel,(2,2048,2048),100)
    assert inside['distance_to_calibration_domain'] == 0
    assert inside['nearest_sample_distance'] > 0
    _,_,hole = performance_surface(replace(kernel,samples=samples[:-1]),(2,2048,2048),100)
    assert hole['distance_to_calibration_domain'] is None
    assert hole['reason'] == 'joint_cell_not_measured'


def test_blackwell_level2_surface_is_opt_in_and_source_bound():
    from heterollm_sim.kernel_model import llama_blackwell_analytical_profile

    analytical = llama_blackwell_analytical_profile(
        'nvidia-rtx-5080', 'runtime', calibrated=False
    )
    calibrated = llama_blackwell_analytical_profile(
        'nvidia-rtx-5080', 'runtime', calibrated=True
    )
    assert sum(len(kernel.samples) for kernel in analytical.kernels) == 0
    assert sum(len(kernel.samples) for kernel in calibrated.kernels) > 0
    assert calibrated.calibration_hardware_id == 'nvidia-rtx-5080'
    assert calibrated.calibration_runtime_sha256 is not None

    signature = 'mmvq:type=12:m=1:fusion=0:small_k=0:halve_iters=0:warps=4:rows=1'
    selected = calibrated.dispatch(
        shape=(1, 1536, 3072), formats=('q4_k',), dtype='fp16',
        phase='decode', output_bits=32, accumulator_bits=32, epilogue='',
        dispatch_signature=signature,
    )
    assert selected is not None
    assert selected.calibration_source_bound is True
    assert selected.dispatch_signature == signature
    assert all(sample.dispatch_signature == signature for sample in selected.samples)


def test_blackwell_level2_prefill_mmq_surface_uses_runtime_and_resource_binding():
    from dataclasses import replace
    from heterollm_sim.kernel_model import llama_blackwell_analytical_profile
    from heterollm_sim.mmq_work import derive_mmq_work

    profile_model = llama_blackwell_analytical_profile(
        'nvidia-rtx-5080', 'runtime', calibrated=True
    )
    base_gpu = gpu_profile()
    base_gpu = replace(
        base_gpu,
        tensor_core=replace(base_gpu.tensor_core, sm_count=84),
    )
    source = derive_mmq_work(
        m=512, n=3072, k=5120, weight_format='Q4_K', sm_count=84,
        shared_memory_per_block=101376,
        runtime_binary_sha256=profile_model.calibration_runtime_sha256,
    )
    workload = GemmWorkload(
        512, 5120, 3072, activation_bits=32, weight_bits=4,
        output_bits=32, accumulator_bits=32, packed_weight_formats=('q4_k',),
        weight_storage_bytes=source.logical_weight_bytes,
        activation_storage_bytes=source.consumer_unique_bytes,
        output_storage_bytes=source.native_output_bytes, mmq_work=source,
        cache_protocol='cold_streaming', execution_phase='prefill',
        activation_dtype='fp32',
    )
    gpu = replace(base_gpu, kernel_model=profile_model)
    estimate = estimate_gpu_gemm(gpu, HBMProfile(1000), workload)
    prediction = estimate.metadata['prediction']
    assert prediction['model'] == 'calibrated_analytical'
    assert prediction['reason'] == 'joint_grid_interpolation'
    assert estimate.metadata['kernel_model']['warps_per_cta'] == 8
    assert estimate.metadata['kernel_model']['shared_memory_per_cta'] == 58880

    mismatched = replace(source, runtime_binary_sha256='a' * 64)
    fallback = replace(workload, mmq_work=mismatched,
                       activation_storage_bytes=mismatched.consumer_unique_bytes,
                       weight_storage_bytes=mismatched.logical_weight_bytes,
                       output_storage_bytes=mismatched.native_output_bytes)
    rejected = estimate_gpu_gemm(gpu, HBMProfile(1000), fallback)
    assert rejected.metadata['prediction']['reason'] == 'source_calibration_binding_mismatch'
    assert rejected.metadata['prediction']['model'] == 'analytical'

def test_invalid_l2_batch_does_not_disappear_from_event_ready_queue():
    from copy import deepcopy
    task = l2_task('bad','w')
    metadata = deepcopy(task.metadata)
    metadata['stateful_l2']['accesses'] = (
        {'buffer_id':'w','offset_bytes':0,'size_bytes':64,'operation':'read','buffer_size_bytes':64},
        {'buffer_id':'w','offset_bytes':64,'size_bytes':64,'operation':'read'})
    task = replace(task,metadata=metadata)
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    for _ in range(2):
        with pytest.raises(ValueError,match='remembered'):
            kernel.step()
        assert kernel.has_active_tasks
        assert kernel.peek_ready_key() is not None
        assert kernel._l2_states == {}

@pytest.mark.parametrize('causal', [False, True])
def test_streamk_compute_uses_selected_kernel_tiles_not_generic_defaults(causal):
    descriptor = capability(operator='attention',phase='prefill',weight_formats=('fp16',),
        output_bits=32,internal_dtype='fp16',compute_primitive='tensor',
        attention_stream_k=True,registers_per_thread=64,attention_query_tile=32)
    gpu = replace(gpu_profile(),kernel_model=profile(descriptor))
    workload = FusedAttentionWorkload(1,2048,256,output_bits=32,score_heads=4,
        kv_hidden_size=128,execution_phase='prefill',query_tile_tokens=128,
        causal_query_positions=(2047,) if causal else ())
    estimate = estimate_gpu_fused_attention(gpu,HBMProfile(1000),workload)
    tile = estimate.metadata['kernel_model']['attention_tile_work']
    assert tile['executed_pairs'] == 32 * 2048
    assert tile['useful_pairs'] == 2048
    assert tile['query_tile_tokens'] == 32
    other = estimate_gpu_fused_attention(gpu,HBMProfile(1000),replace(workload,query_tile_tokens=16))
    assert other.service_ns == estimate.service_ns

def test_source_mmvq_geometry_overrides_generic_descriptor_residency():
    from test_mmvq_work import work as source_work
    source = source_work(m=8,fmt='Q8_0')
    descriptor = capability(weight_formats=('q8_0',),kernel_family='cuda_mmvq_q8_0',
                            output_bits=32,activation_dtype='fp32',warps_per_cta=8)
    gpu = replace(gpu_profile(),kernel_model=profile(descriptor))
    workload = GemmWorkload(source.m,source.n,source.k,activation_bits=32,weight_bits=8,
        output_bits=32,packed_weight_formats=('q8_0',),execution_phase='decode',
        activation_storage_bytes=source.consumer_q8_1_unique_bytes,mmvq_work=source)
    estimate = estimate_gpu_gemm(gpu,HBMProfile(1000),workload)
    audit = estimate.metadata['kernel_model']
    assert audit['warps_per_cta'] == source.warps_per_cta == 2
    assert audit['shared_memory_per_cta'] == source.reduction_shared_array_bytes
    assert audit['cta_count'] == source.cta_count
    assert audit['residency_source'] == 'mmvq_source_warps_shared_lower_bound_registers_descriptor'
    # A larger explicitly declared static allocation must not be discarded.
    gpu = replace(gpu,kernel_model=profile(replace(descriptor,shared_memory_per_cta=16384)))
    assert estimate_gpu_gemm(gpu,HBMProfile(1000),workload).metadata['kernel_model']['shared_memory_per_cta'] == 16384


def test_column_only_native_resources_are_rejected():
    with pytest.raises(ValueError, match='specialization binding'):
        capability(native_resource_variants=((1,53,1408),))


@pytest.mark.parametrize('m,regs,resident', [(1,53,9),(2,80,6),(4,120,4)])
def test_native_resources_bound_to_binary_shape_and_device(m,regs,resident):
    from test_mmvq_work import contract
    from heterollm_sim.mmvq_work import derive_mmvq_work
    from heterollm_sim.native_kernel_resources import BINARY_SHA256, observed_mmvq_resources
    from heterollm_sim.kernel_model import llama_blackwell_analytical_profile
    source=derive_mmvq_work(m=m,n=4096,k=4096,weight_format='Q4_K',
        contract=contract(runtime_binary_sha256=BINARY_SHA256),allow_k_formats=True)
    assert observed_mmvq_resources(source,hardware_id='nvidia-rtx-5080')['registers']==regs
    assert observed_mmvq_resources(replace(source,runtime_binary_sha256='a'*64),hardware_id='nvidia-rtx-5080') is None
    wider=derive_mmvq_work(m=m,n=2048,k=8192,weight_format='Q4_K',
        contract=contract(runtime_binary_sha256=BINARY_SHA256),allow_k_formats=True)
    assert observed_mmvq_resources(wider,hardware_id='nvidia-rtx-5080')['registers']==regs
    assert observed_mmvq_resources(replace(source,k=8192),hardware_id='nvidia-rtx-5080') is None
    assert observed_mmvq_resources(replace(source,small_k=not source.small_k),hardware_id='nvidia-rtx-5080') is None
    assert observed_mmvq_resources(source,hardware_id='another-gpu') is None
    gpu=replace(gpu_profile(),kernel_model=llama_blackwell_analytical_profile('nvidia-rtx-5080','test'))
    workload=GemmWorkload(m,4096,4096,activation_bits=32,weight_bits=4,output_bits=32,
        packed_weight_formats=('q4_k',),execution_phase='decode',mmvq_work=source,
        activation_storage_bytes=source.consumer_q8_1_unique_bytes)
    audit=estimate_gpu_gemm(gpu,HBMProfile(1000),workload).metadata['kernel_model']
    assert audit['occupancy']['resident_ctas']==resident
    assert audit['occupancy']['source']=='cuda_occupancy_api'
    assert audit['occupancy']['native_resources']['registers']==regs
