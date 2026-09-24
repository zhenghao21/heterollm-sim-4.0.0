from dataclasses import replace
import pytest
from heterollm_sim.cost_models import DigitalSramCimProfile, GemmWorkload, estimate_cim_gemm
from heterollm_sim.config import _cim_profile_from_dict
from heterollm_sim.serde import to_primitive

def profile(**changes):
    p = DigitalSramCimProfile(arithmetic_mode='fp16_fp32_analytical',
        float_cycles_per_eval=2, float_accumulator_outputs_per_cycle=8,
        float_contract_basis='unvalidated test assumption', p_k=4, p_n=4,
        weight_conversion_mode='packed_to_fp16_cold', weight_decode_elements_per_ns=2,
        conversion_scratch_capacity_bytes=4096, weight_capacity_bytes=4096,
        activation_fp32_to_fp16_elements_per_ns=3,
        conversion_contract_basis='unvalidated test converter assumption',
        conversion_read_energy_pj_per_byte=2, conversion_write_energy_pj_per_byte=3)
    return replace(p, **changes)

def workload(**changes):
    return replace(GemmWorkload(m=2,k=5,n=7,activation_bits=16,weight_bits=4,
        cim_arithmetic='fp16',packed_weight_formats=('IQ4_XS',),
        weight_storage_bytes=24,weight_metadata_bytes=6,
        packed_weight_transform_operations=35), **changes)

@pytest.mark.parametrize('fmt',['IQ3_S','IQ4_XS'])
def test_conversion_bytes_capacity_energy_and_lock(fmt):
    p=profile(); w=workload(packed_weight_formats=(fmt,)); e=estimate_cim_gemm(p,w)
    a=e.metadata['weight_conversion']; phases={v['name']:v for v in a['ordered_phases']}
    assert a['packed_read_bytes']==30 and a['dense_padded_bytes']==128
    assert a['scratch_peak_bytes']==206 and a['conversions_per_invocation']==1
    assert phases['weight_decode']['energy_pj']==30*2+128*3
    assert phases['dense_weight_program']['bytes_moved']==128
    assert phases['dense_weight_program']['energy_pj']==128*2
    assert len(e.phases)==1
    assert next(d for d in e.phases[0].demands if d.resource_id==p.load_resource_id).service_ns==e.service_ns
    assert e.service_ns==sum(v['service_ns'] for v in a['ordered_phases'])
    assert _cim_profile_from_dict(to_primitive(p))==p

@pytest.mark.parametrize('changes,match',[
    ({'conversion_scratch_capacity_bytes':157},'scratch'),
    ({'weight_capacity_bytes':127},'capacity'),
    ({'weight_conversion_mode':'disabled'},'packed weights')])
def test_capacity_and_default_rejection(changes,match):
    with pytest.raises(ValueError,match=match): estimate_cim_gemm(profile(**changes),workload())

def test_no_free_residency_and_unsupported():
    with pytest.raises(ValueError,match='warm'): estimate_cim_gemm(profile(),workload(),True)
    with pytest.raises(ValueError,match='only IQ'): estimate_cim_gemm(profile(),workload(packed_weight_formats=('Q4_0',)))
    with pytest.raises(ValueError,match='actual packed'): estimate_cim_gemm(profile(),workload(weight_storage_bytes=None))
    a=estimate_cim_gemm(profile(),workload()); b=estimate_cim_gemm(profile(),workload())
    assert a==b and a.service_ns>0

def test_fp32_requires_explicit_conversion_and_scratch():
    w=workload(activation_storage_bytes=40)
    with pytest.raises(ValueError,match='FP32 input'): estimate_cim_gemm(profile(activation_fp32_to_fp16_elements_per_ns=0),w)
    a=estimate_cim_gemm(profile(),w).metadata['weight_conversion']
    assert a['fp32_input_conversion'] and a['scratch_peak_bytes']==246
    assert 'activation_fp32_to_fp16' in [p['name'] for p in a['ordered_phases']]

def test_real_qwen_cold_conversion_online_and_component_partition():
    from tools.qwen38_memory_scenario import load_model, build_scenario
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.control_plane_planner import _derive_requirements, _cim_rank_padded_weight_bytes, _requirement_execution_ranks
    model,_=load_model()
    s=build_scenario(model,'dual_dram_cim_converted',prompt_tokens=2,output_tokens=2)
    result=run_scenario(s,retention_policy='aggregate')
    rows=[b.cost.metadata['cim_weight_conversion'] for b in result.serving.batches]
    assert all(r['conversions']>0 and r['packed_read_bytes']>0 for r in rows)
    from tools.qwen38_memory_scenario import compact_observation
    from heterollm_sim.reporting import report_dict
    observation = compact_observation(result, report_dict(
        result, visualization_limit=1, visualization_memory_segment_limit=1))
    backing = "layer-000.mlp_weights#cold_backing"
    assert backing in observation["storage_distribution"]["dram0"]["tensor_ids"]
    assert observation["placement"]["rank_weight_shards"][backing][0]["physical_bytes"] == 123_944_960
    assert observation["storage_distribution"]["cim0"]["persistent_weight_shard_bytes"] == 0


    requirement=next(r for r in _derive_requirements(s) if r.tensor_id=='layer-000.mlp_weights')
    rank=_requirement_execution_ranks(s,requirement,(s.hardware.get_component('soc0'),))[0]
    bad=replace(s, hardware=replace(s.hardware,components=tuple(
        replace(c,capacity_bytes=512*1024*1024) if c.component_id=='cim0' else c for c in s.hardware.components)))
    with pytest.raises(ValueError,match='component capacity'):
        _cim_rank_padded_weight_bytes(bad,requirement,rank,'cim0')
    multi=replace(s,workload=replace(s.workload,scheduler=replace(s.workload.scheduler,max_num_seqs=2)))
    assert _cim_rank_padded_weight_bytes(multi,requirement,rank,'cim0') > 0


def test_slow_converter_cost_and_replication_storage():
    p=profile(max_m_replication=2);w=workload(m=8)
    e=estimate_cim_gemm(p,w);a=e.metadata['weight_conversion']
    assert a['array_replication']==2 and a['array_storage_bytes']==256
    assert a['scratch_peak_bytes']==350  # includes input/output; dense scratch is not replicated
    assert estimate_cim_gemm(replace(p,weight_decode_elements_per_ns=0.01),w).service_ns>e.service_ns


@pytest.mark.parametrize('batch', [2, 4])
def test_real_qwen_multi_request_conversion(batch):
    from tools.qwen38_memory_scenario import load_model, build_scenario
    from heterollm_sim.reporting import run_scenario
    model, _ = load_model()
    s = build_scenario(model, 'dual_dram_cim_converted', prompt_tokens=2, output_tokens=2, batch=batch)
    result = run_scenario(s, retention_policy='aggregate')
    rows = [b.cost.metadata['cim_weight_conversion'] for b in result.serving.batches]
    assert rows and all(row['conversions'] > 0 for row in rows)
    assert all(row['incoming_transfer_safety'] == 'atomic_call_includes_incoming_transfers' for row in rows)


def test_atomic_lifetime_includes_transfers_and_serializes_requests():
    from heterollm_sim.planner import _TaskBuilder, _coalesce_cim_conversion_lifetime
    from heterollm_sim.ir import RequestSpec
    from heterollm_sim.contracts import ResourceDemand, TaskCategory
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from heterollm_sim.architecture_presets import get_architecture_preset
    hardware = get_architecture_preset('soc-2x-dram-sram-cim').hardware
    tasks = []
    for request_id in ('a', 'b'):
        builder = _TaskBuilder(RequestSpec(request_id, 0, 2, 2))
        prior = ()
        estimate = estimate_cim_gemm(profile(), workload())
        for name, demands, metadata in (
            ('activation_in', (ResourceDemand('noc.input', 3, bytes_moved=20),), {}),
            ('weight_in', (ResourceDemand('dram.read', 5, bytes_moved=30),), {}),
            ('convert_array', estimate.phases[0].demands, {'phase': 'weight_load', 'phase_metadata': estimate.phases[0].metadata}),
            ('output_out', (ResourceDemand('noc.output', 7, bytes_moved=28),), {})):
            last = builder.add(name, TaskCategory.COMPUTE, demands, dependencies=prior, metadata=metadata)
            prior = (last,)
        builder.record_rank_value(last, 0, 'soc0')
        original_ids = {t.task_id for t in builder.tasks}
        original_bytes = sum(d.bytes_moved for t in builder.tasks for d in t.demands)
        _coalesce_cim_conversion_lifetime(builder, 0, last, profile(), hardware)
        assert builder.rank_value_component((last,), 0) == 'soc0'
        assert original_ids <= {t.task_id for t in builder.tasks}
        assert sum(d.bytes_moved for t in builder.tasks for d in t.demands) == original_bytes
        assert len(builder.tasks) == 5
        assert builder.previous == last
        assert builder.tasks[-1].task_id == last
        assert all(not t.demands for t in builder.tasks[1:])
        task = builder.tasks[0]
        audit = task.metadata['cim_scratch_lifecycle']
        assert [row['name'] for row in audit['ordered_tasks']] == ['activation_in', 'weight_in', 'convert_array', 'output_out']
        assert audit['acquire'] == 'before_incoming_activation_and_weight'
        assert audit['release'] == 'after_output_transfer_last_consumer'
        assert task.dependencies == ()
        assert all(d.service_ns == estimate.service_ns + 15 for d in task.demands)
        tasks.append(task)
    for incremental in (False, True):
        kernel = UnifiedEventKernel() if incremental else UnifiedEventKernel.from_closed_graph(tasks)
        if incremental:
            for task in tasks:
                kernel.submit((task,))
        events = []
        while kernel.has_active_tasks:
            events.append(kernel.step())
        kernel.assert_drained()
        assert events[1].start_ns >= events[0].end_ns
        assert events[1].queue_wait_ns > 0
        assert audit['scratch_resource_id'] in events[1].resource_predecessors


def test_scratch_precheck_failure_does_not_poison_next_call():
    with pytest.raises(ValueError, match='scratch'):
        estimate_cim_gemm(profile(conversion_scratch_capacity_bytes=205), workload())
    assert estimate_cim_gemm(profile(conversion_scratch_capacity_bytes=206), workload()).metadata['weight_conversion']['scratch_peak_bytes'] == 206

def test_real_static_two_requests_have_disjoint_scratch_intervals_and_retry():
    from tools.qwen38_memory_scenario import load_model, build_scenario
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.planner import compile_scenario
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from heterollm_sim.cost_models import DigitalSramCimProfile
    model, _ = load_model()
    scenario = build_scenario(model, 'dual_dram_cim_converted', prompt_tokens=2, output_tokens=2, batch=2)
    mapped = bootstrap_control_plane(scenario).scenario
    bad = replace(scenario, component_profiles={**scenario.component_profiles,
        'cim': {key: replace(value, conversion_scratch_capacity_bytes=1)
                for key, value in scenario.component_profiles['cim'].items()}})
    with pytest.raises(ValueError, match='scratch'):
        bootstrap_control_plane(bad)
    schedule = compile_scenario(mapped)
    atomic = {t.task_id: t for t in schedule.tasks if t.metadata.get('atomic_cim_conversion')}
    assert len({t.request_id for t in atomic.values()}) == 2
    task_ids = {t.task_id for t in schedule.tasks}
    assert all(set(t.dependencies) <= task_ids for t in schedule.tasks)
    markers = [t for t in schedule.tasks if t.metadata.get('cim_atomic_parent')]
    assert any(t.metadata.get('event_kind') == 'model_weight_access' for t in markers)
    assert any(t.metadata.get('event_kind') == 'model_weight_read' for t in markers)
    assert all(not t.demands for t in markers)
    kernel = UnifiedEventKernel.from_closed_graph(schedule.tasks)
    intervals = []
    while kernel.has_active_tasks:
        event = kernel.step()
        if event.task.task_id in atomic:
            intervals.append(event)
    kernel.assert_drained()
    intervals.sort(key=lambda event: event.start_ns)
    assert intervals
    assert all(left.end_ns <= right.start_ns for left, right in zip(intervals, intervals[1:]))
    assert any(event.queue_wait_ns > 0 for event in intervals)
    # Failed compilation does not change scenario/profile or persistent reserve state.
    assert compile_scenario(mapped).tasks == schedule.tasks



@pytest.fixture(scope="module")
def packed_backing_pair():
    from tools.qwen38_memory_scenario import load_model, build_scenario
    from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement

    model, _ = load_model()
    scenario = build_scenario(model, "dual_dram_cim_converted", prompt_tokens=2, output_tokens=2)
    policy = PlacementPolicy(**scenario.placement.metadata["control_plane"]["policy"]["options"])
    baseline_policy = replace(policy, weight_tensor_targets={
        **policy.weight_tensor_targets, "layer-000.mlp_weights": "dram0",
    })
    baseline = plan_runtime_placement(scenario, baseline_policy)
    cold = plan_runtime_placement(scenario, policy)
    assert baseline.fully_placed, baseline.unplaced
    assert cold.fully_placed, cold.unplaced
    return scenario, policy, baseline, cold


def test_cold_backing_preserves_exact_packed_storage_and_replanning(packed_backing_pair):
    from heterollm_sim.control_plane_planner import plan_runtime_placement

    scenario, policy, baseline, cold = packed_backing_pair
    tensor = "layer-000.mlp_weights"
    backing = tensor + "#cold_backing"
    before = baseline.placement.metadata["control_plane"]["decision"]
    after = cold.placement.metadata["control_plane"]["decision"]
    # IQ3_S gate/up plus IQ4_XS down, including block metadata, not the
    # nominal group allocation (135450366) or padded FP16 array (534773760).
    packed = 76_595_200 + 47_349_760
    assert sum(s["physical_bytes"] for s in before["rank_weight_shards"][tensor]) == packed
    assert cold.placement.tensor_bytes[backing] == packed
    assert cold.placement.tensor_to_component[backing] == "dram0"
    assert tensor not in cold.placement.tensor_bytes
    detail = after["weight_tensor_details"][tensor]
    assert detail["backing_storage_tensor_id"] == backing
    assert detail["total_physical_bytes"] == 0
    assert after["cim_total_physical_bytes"][tensor] == 0
    assert all(s["physical_bytes"] == 0 for s in after["rank_weight_shards"][tensor])

    def storage_totals(decision):
        # Same input fields as compact_observation.storage_distribution.
        totals = {}
        for shards in decision["rank_weight_shards"].values():
            for shard in shards:
                component = shard["storage_component_id"]
                totals[component] = totals.get(component, 0) + shard["physical_bytes"]
        return {key: value for key, value in totals.items() if value}

    assert storage_totals(before) == storage_totals(after)
    replanned = plan_runtime_placement(cold.apply(scenario), policy)
    assert replanned.fully_placed, replanned.unplaced
    again = replanned.placement.metadata["control_plane"]["decision"]
    assert storage_totals(again) == storage_totals(after)
    assert replanned.placement.tensor_bytes == cold.placement.tensor_bytes


def test_cold_backing_enters_shared_solver_capacity_admission(packed_backing_pair):
    from heterollm_sim import control_plane_planner as cp
    from heterollm_sim.communication import TopologyRouter

    scenario, policy, _, _ = packed_backing_pair
    requirement = next(r for r in cp._derive_requirements(scenario)
                       if r.tensor_id == "layer-000.mlp_weights")
    requirement = replace(requirement, fixed_component="cim0")
    candidates, rejected = cp._candidates_for_requirement(
        scenario, policy, requirement, 0, TopologyRouter(scenario.hardware),
        cp._component_capacities(scenario), {}, cp._mapping_controls(scenario, []),
    )
    cold = next(c for c in candidates if c.component_id == "cim0")
    assert cold.usage == (("dram0", 123_944_960),), rejected
    # Both individually fit, but combined they exceed capacity by one byte.
    # This exercises the actual generated candidate, not a report-only ledger.
    other = cp._Candidate(1, "soc0", "dram0", 0.0, (("dram0", 64),))
    capacities = {"dram0": 123_944_960 + 64}
    fit = cp._solve_builtin(((cold,), (other,)), capacities, {}, 1.0)
    assert all(c is not None for c in fit.assignment)
    capacities["dram0"] -= 1
    rejected = cp._solve_builtin(((cold,), (other,)), capacities, {}, 1.0)
    assert any(c is None for c in rejected.assignment)
    assert not cp._assignment_fits((cold, other), capacities, {})
    assert not cp._fits(cold, {"dram0": 1}, {"dram0": 123_944_960})
    assert cp._fits(cold, {}, {"dram0": 123_944_960, "cim0": 0})


def test_cold_backing_rejects_undersized_dram_on_mapped_validation(packed_backing_pair):
    from heterollm_sim.planner import validate_scenario

    scenario, _, _, cold = packed_backing_pair
    mapped = cold.apply(scenario)
    decision = cold.placement.metadata["control_plane"]["decision"]
    packed_usage = sum(
        shard["physical_bytes"]
        for shards in decision["rank_weight_shards"].values()
        for shard in shards if shard["storage_component_id"] == "dram0"
    )
    assert packed_usage >= 123_944_960
    # Without the backing entry the remaining weights fit this limit.
    low = replace(mapped, hardware=replace(mapped.hardware, components=tuple(
        replace(c, capacity_bytes=packed_usage - 1) if c.component_id == "dram0" else c
        for c in mapped.hardware.components
    )))
    validation = validate_scenario(low)
    assert (
        f"declared tensors require {packed_usage} bytes on dram0, capacity is {packed_usage - 1}"
        in validation.errors_en
    ), validation.errors_en
