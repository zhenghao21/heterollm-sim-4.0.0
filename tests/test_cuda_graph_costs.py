"""Graph runtime pricing must preserve physical work and async ordering."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cuda_graph_costs import apply_cuda_graph_runtime_costs
from heterollm_sim.cuda_graph_lifecycle import CudaGraphInvocation, CudaGraphRuntime, bind_cuda_graph_tasks
from heterollm_sim.runtime_residual import RuntimeStructureMeasurements, PHASES, DEVICE_PHASES


class IndependentCosts:
    evidence = 'synthetic-test'
    qualified = True
    validation_relative_error = 0.01

    def costs(self, topology, node_count, *, phases=None):
        assert topology == 'chain' and node_count == 2
        return {'ordinary_submit': 100, 'capture': 20, 'instantiate': 30,
                'update': 40, 'first_launch_submit': 50, 'replay_submit': 10}


def experimental_measurements():
    timings = {phase: [900.0, 1000.0, 1100.0] for phase in PHASES + DEVICE_PHASES}
    summary = {phase: {'median_ns': 1000.0, 'repeat_mad_ns': 100.0, 'minimum_ns': 900.0,
                      'maximum_ns': 1100.0, 'repeat_count': 3} for phase in timings}
    return RuntimeStructureMeasurements(hardware_id='gpu', runtime_id='cuda', architecture='sm_120',
        device='gpu', cc='sm_120', driver_version='driver', runtime_version='runtime', cpu_id='cpu', os_id='os',
        evidence='independent test data', limitations=('Synthetic kernel parameters.',),
        samples=({'topology': 'chain', 'node_count': 2, 'source_structures': [{'source_revision': 'source'}],
                  'timings_ns': timings, 'phase_measurements': summary},))


def bound_tasks(captured):
    invocation = CudaGraphInvocation('cuda:0', 'first-node', 0, ((1,),), True, 'compatible')
    runtime = CudaGraphRuntime()
    transition = runtime.prepare(invocation, host_time_us=0)
    runtime.commit(transition)
    if captured:
        transition = runtime.prepare(invocation, host_time_us=0)
    def task(ident, phase, deps, service, byte_count=0):
        return TaskSpec(ident, 'request', ident, TaskCategory.COMPUTE, tuple(deps),
            (ResourceDemand('gpu0.frontend' if phase == 'kernel_launch' else 'gddr0', service,
                            bytes_moved=byte_count),),
            metadata={'phase': phase, 'target_component': 'gpu0'})
    tasks = (task('launch1', 'kernel_launch', (), 3),
             task('body1', 'gpu_compute', ('launch1',), 1000, 2048),
             task('launch2', 'kernel_launch', ('body1',), 3),
             task('body2', 'gpu_compute', ('launch2',), 1000, 4096))
    tasks = bind_cuda_graph_tasks(tasks, transition, member_task_ids=tuple(t.task_id for t in tasks))
    return (replace(tasks[0], metadata={**tasks[0].metadata,
            'cuda_graph_node_count': 2, 'cuda_graph_topology': 'chain'}), *tasks[1:])


@pytest.mark.parametrize('captured', [False, True])
def test_independent_host_costs_preserve_all_device_work(captured):
    before = bound_tasks(captured)
    result = apply_cuda_graph_runtime_costs(before, SimpleNamespace(
        runtime_calibration=IndependentCosts(), runtime_host_resource_id='cpu0.cuda_submission',
        graph_enabled=captured))
    after = {t.task_id: t for t in result}
    assert all(after[t.task_id].demands == t.demands for t in before)
    assert sum(d.bytes_moved for t in result for d in t.demands) == 6144
    hosts = [t for t in result if t.task_id not in {b.task_id for b in before}]
    assert all(t.demands[0].resource_id == 'cpu0.cuda_submission' for t in hosts)
    assert sum(t.demands[0].service_ns for t in hosts) == (140 if captured else 100)
    if not captured:
        # The second CPU enqueue may overlap body1. Device launch2 still
        # respects the original stream order and its own enqueue completion.
        assert hosts[1].dependencies == (hosts[0].task_id,)
        assert 'body1' not in hosts[1].dependencies
        assert set(after['launch2'].dependencies) == {'body1', hosts[1].task_id}
    else:
        assert hosts[-1].task_id in after['launch1'].dependencies
        assert hosts[-1].task_id in after['launch2'].dependencies


def test_graph_mode_mismatch_and_unbound_launch_fail_closed():
    profile = SimpleNamespace(runtime_calibration=IndependentCosts(),
                              runtime_host_resource_id='cpu0.cuda_submission', graph_enabled=False)
    with pytest.raises(ValueError, match='contradicts disabled'):
        apply_cuda_graph_runtime_costs(bound_tasks(True), profile)
    tasks = bound_tasks(False)
    extra = replace(tasks[0], task_id='unbound', metadata={'phase': 'kernel_launch'})
    with pytest.raises(ValueError, match='no lifecycle owner'):
        apply_cuda_graph_runtime_costs((*tasks, extra), profile)


@pytest.mark.parametrize('root_kind', ['device_memory', 'marker'])
@pytest.mark.parametrize('captured', [False, True])
def test_independent_device_body_roots_wait_for_graph_submission_only(root_kind, captured):
    from heterollm_sim.event_kernel import UnifiedEventKernel
    tasks = bound_tasks(captured)
    # A complete device body can contain a root that is not a kernel launch,
    # such as a copy or a cost-free dependency marker.
    root = replace(tasks[1], dependencies=())
    if root_kind == 'marker':
        root = replace(root, category=TaskCategory.SYNCHRONIZATION, demands=())
    tasks = (tasks[0], root, *tasks[2:])
    result = apply_cuda_graph_runtime_costs(tasks, SimpleNamespace(
        runtime_calibration=IndependentCosts(), runtime_host_resource_id='cpu0.cuda_submission',
        graph_enabled=captured))
    kernel = UnifiedEventKernel.from_closed_graph(result)
    events = {}
    while (event := kernel.step()) is not None:
        events[event.task.task_id] = event
    kernel.assert_drained()
    hosts = [event for ident, event in events.items() if '.cuda_host.' in ident]
    if captured:
        assert events[root.task_id].start_ns >= max(event.end_ns for event in hosts)
    else:
        assert events[root.task_id].start_ns == 0
        # Ordinary enqueues remain independent of the preceding GPU body.
        assert hosts[1].start_ns == hosts[0].end_ns


def test_missing_phase_cost_cannot_disappear_as_zero():
    tasks = bound_tasks(True)
    tasks = (replace(tasks[0], metadata={**tasks[0].metadata,
             'cuda_graph_lifecycle_events': ('destroy_executable', 'replay_submit')}), *tasks[1:])
    with pytest.raises(ValueError, match='no independent measured cost'):
        apply_cuda_graph_runtime_costs(tasks, SimpleNamespace(runtime_calibration=IndependentCosts(),
            runtime_host_resource_id='cpu0.cuda_submission', graph_enabled=True))


@pytest.mark.parametrize('captured', [False, True])
def test_explicit_experiment_preserves_physical_work_and_discloses_uncertainty(captured):
    before = bound_tasks(captured)
    profile = SimpleNamespace(runtime_calibration=experimental_measurements(),
        runtime_host_resource_id='cpu0.cuda_submission', graph_enabled=captured,
        runtime_measurement_mode='experimental_exact_structure')
    result = apply_cuda_graph_runtime_costs(before, profile)
    after = {task.task_id: task for task in result}
    assert all(after[task.task_id].demands == task.demands for task in before)
    host_tasks = [task for task in result if '.cuda_host.' in task.task_id]
    assert sum(demand.service_ns for task in host_tasks for demand in task.demands) == (4000 if captured else 1000)
    for task in host_tasks:
        audit = task.metadata['cuda_runtime_cost']
        assert audit['prediction_qualified'] is False
        assert audit['runtime_measurement_mode'] == 'experimental_exact_structure'
        if audit['schema'] == 'cuda-runtime-cost/v1':
            assert audit['independent_measurement_uncertainty']
            assert audit['physical_body_costs_preserved'] is True
        else:
            assert audit['schema'] == 'cuda-runtime-cost-ref/v1'
            assert audit['audit_id'] == host_tasks[0].metadata['cuda_runtime_cost']['audit_id']
    profile.runtime_measurement_mode = 'qualified'
    with pytest.raises(ValueError, match='explicit experimental'):
        apply_cuda_graph_runtime_costs(before, profile)


def test_old_graph_destruction_and_failed_update_use_old_new_measurements():
    import copy
    evidence = experimental_measurements()
    old_sample = copy.deepcopy(evidence.samples[0])
    old_sample['node_count'] = 3
    for phase in old_sample['timings_ns']:
        old_sample['timings_ns'][phase] = [2900.0, 3000.0, 3100.0]
        old_sample['phase_measurements'][phase] = {'median_ns': 3000.0, 'repeat_mad_ns': 100.0,
            'minimum_ns': 2900.0, 'maximum_ns': 3100.0, 'repeat_count': 3}
    old = {'topology': 'chain', 'node_count': 3}
    new = {'topology': 'chain', 'node_count': 2}
    evidence = replace(evidence, samples=(*evidence.samples, old_sample), update_pairs=({
        'old': old, 'new': new, 'phase': 'update_failure', 'repeated_ns': [70.0, 75.0, 80.0],
        'measurement': {'median_ns': 75.0, 'repeat_mad_ns': 5.0, 'minimum_ns': 70.0,
                        'maximum_ns': 80.0, 'repeat_count': 3}},))
    tasks = bound_tasks(True)
    events = ('destroy_graph', 'update_failure', 'destroy_exec', 'instantiate', 'first_launch_submit')
    bindings = {event: old if event.startswith('destroy') else new for event in events}
    bindings['update_failure'] = {**new, 'previous': old, 'current': new}
    tasks = (replace(tasks[0], metadata={**tasks[0].metadata,
        'cuda_graph_lifecycle_events': events, 'cuda_graph_phase_structures': bindings}), *tasks[1:])
    profile = SimpleNamespace(runtime_calibration=evidence, graph_enabled=True,
        runtime_host_resource_id='cpu0.cuda_submission', runtime_measurement_mode='experimental_exact_structure')
    result = apply_cuda_graph_runtime_costs(tasks, profile)
    host_tasks = [task for task in result if '.cuda_host.' in task.task_id]
    assert sum(d.service_ns for task in host_tasks for d in task.demands) == 8075
    bad = copy.deepcopy(bindings)
    bad['update_failure']['previous'] = new
    tasks = (replace(tasks[0], metadata={**tasks[0].metadata, 'cuda_graph_phase_structures': bad}), *tasks[1:])
    with pytest.raises(ValueError, match='exact independently measured old/new'):
        apply_cuda_graph_runtime_costs(tasks, profile)


def test_structural_audit_is_shared_once_without_changing_tasks_or_demands():
    import json
    from heterollm_sim.serde import to_primitive
    tasks = bound_tasks(False)
    large_topology = 'typed_chain/v1:' + 'kernel,' * 5000
    evidence = experimental_measurements()
    sample = {**evidence.samples[0], 'topology': large_topology}
    evidence = replace(evidence, samples=(sample,))
    tasks = (replace(tasks[0], metadata={**tasks[0].metadata, 'cuda_graph_topology': large_topology}), *tasks[1:])
    result = apply_cuda_graph_runtime_costs(tasks, SimpleNamespace(runtime_calibration=evidence,
        graph_enabled=False, runtime_host_resource_id='cpu0.cuda_submission', runtime_measurement_mode='experimental_exact_structure'))
    costs = [task.metadata['cuda_runtime_cost'] for task in result if 'cuda_runtime_cost' in task.metadata]
    full = [cost for cost in costs if cost['schema'] == 'cuda-runtime-cost/v1']
    assert len(full) == 1
    assert all(cost['audit_id'] == full[0]['audit_id'] for cost in costs)
    assert all(cost is costs[1] for cost in costs[1:])
    # Compare the exact previous serialized representation: same resulting task
    # DAG and demand objects, with the full audit repeated at every reference.
    expanded = tuple(replace(task, metadata={**task.metadata, 'cuda_runtime_cost': full[0]})
        if 'cuda_runtime_cost' in task.metadata else task for task in result)
    assert [(task.task_id, task.dependencies, task.demands) for task in result] == [
        (task.task_id, task.dependencies, task.demands) for task in expanded]
    compact_bytes = len(json.dumps(to_primitive(result)))
    expanded_bytes = len(json.dumps(to_primitive(expanded)))
    assert compact_bytes < expanded_bytes * 0.6
    from heterollm_sim.event_kernel import UnifiedEventKernel
    def realized(graph):
        kernel = UnifiedEventKernel.from_closed_graph(graph)
        events = []
        while (event := kernel.step()) is not None:
            events.append((event.task.task_id, event.start_ns, event.end_ns, event.demands))
        kernel.assert_drained()
        return events
    assert realized(result) == realized(expanded)


def test_run_registry_removes_typed_topology_strings_from_realized_tasks():
    tasks = bound_tasks(False)
    evidence = experimental_measurements()
    large_topology = 'typed_chain/v1:' + 'kernel,' * 5000
    evidence = replace(evidence, samples=({**evidence.samples[0], 'topology': large_topology},))
    descriptor = {'structure_id': 'cuda-structure-1', 'topology': large_topology, 'node_count': 2}
    tasks = (replace(tasks[0], metadata={**tasks[0].metadata, 'cuda_graph_topology': large_topology,
        'cuda_graph_structure_id': 'cuda-structure-1', 'cuda_graph_phase_structures': {'ordinary_submit': descriptor}}), *tasks[1:])
    result = apply_cuda_graph_runtime_costs(tasks, SimpleNamespace(runtime_calibration=evidence,
        graph_enabled=False, runtime_host_resource_id='cpu0.cuda_submission', runtime_measurement_mode='experimental_exact_structure'))
    import json
    from heterollm_sim.serde import to_primitive
    encoded = json.dumps(to_primitive(result))
    assert large_topology not in encoded
    assert 'cuda-structure-1' in encoded
    assert 'cuda_graph_topology' not in next(task for task in result if task.task_id == tasks[0].task_id).metadata


@pytest.mark.parametrize('captured', [False, True])
def test_online_stage_compaction_preserves_cpu_enqueue_gpu_overlap(captured):
    from heterollm_sim import planner
    from heterollm_sim.event_kernel import UnifiedEventKernel
    from heterollm_sim.reference import build_llama_default_scenario
    from heterollm_sim.scalable_serving import execute_cost_schedule
    from heterollm_sim.serving import _OnlineRuntime
    scenario = build_llama_default_scenario()
    original = bound_tasks(captured)
    original = tuple(replace(task, metadata={**task.metadata, 'layer_id': 'layer0',
        'operator_invocation_group_id': 'group0'}) for task in original)
    tasks = apply_cuda_graph_runtime_costs(original, SimpleNamespace(runtime_calibration=IndependentCosts(),
        runtime_host_resource_id='cpu0.cuda_submission', graph_enabled=captured))
    schedule = SimpleNamespace(tasks=tasks, resource_capacities={}, resource_owners={})
    direct = execute_cost_schedule(schedule, retain_task_metadata=False)
    rows, error = planner._compact_execution_stages(scenario, direct.execution_records,
        ({'group_id': 'group0', 'request_ids': ('request',)},))
    assert error is None
    stages = planner._trusted_execution_stages(rows)
    assert stages is not None
    staged, _ = _OnlineRuntime._stage_task_specs(stages,
        {stage.stage_id: 0 for stage in stages}, 'online-test')
    kernel = UnifiedEventKernel.from_closed_graph(staged)
    realized = []
    while (event := kernel.step()) is not None:
        realized.append(event)
    kernel.assert_drained()
    assert kernel.makespan_ns == pytest.approx(direct.makespan_ns)
    assert sum(d.bytes_moved for event in realized for d in event.demands) == 6144
    if not captured:
        host2 = next(event for event in realized if '.cuda_host.1.' in event.task.name)
        body1 = next(event for event in realized if event.task.name == 'body1')
        assert host2.start_ns < body1.end_ns
        assert host2.end_ns > body1.start_ns
        assert kernel.makespan_ns == 2056


def measured_driver_case(captured):
    """Use the real local host/CP lowering around a bounded device body."""
    import json
    from pathlib import Path
    from heterollm_sim import planner
    from heterollm_sim.config import scenario_from_dict
    path = Path(__file__).parents[1] / 'docs/frontend_native_validation_2026-10-07/scenario_qwen3_0_6b_f16_512_128.json'
    scenario = scenario_from_dict(json.loads(path.read_text(encoding='utf-8')))
    context = planner.CompilationContext(scenario)
    with planner._compilation_scope(scenario, context):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        terminal = planner._add_physical_invocation_frontend(builder, scenario,
            planner._parallel_plan(scenario), (), name='frontend', request_count=1,
            token_count=1, invocation_count=1, invocation_family='target_operator',
            orchestration_stage='gpu_consumer_frontend', invocation_group_ids=('group0',))
    device = tuple(replace(task,
        dependencies=(terminal,) if task.task_id == 'launch1' else task.dependencies,
        metadata={**task.metadata, 'operator_invocation_group_id': 'group0', 'layer_id': 'layer0'})
        for task in bound_tasks(captured))
    profile = SimpleNamespace(runtime_calibration=IndependentCosts(),
        runtime_host_resource_id='cpu0.cuda_submission', graph_enabled=captured)
    frontend = tuple(replace(task, metadata={**task.metadata,
        'gpu_consumer_component_id': 'gpu0', 'gpu_consumer_request_ids': ('request',),
        'gpu_consumer_frontend_basis': 'lowered_operator_execution_device'}) for task in builder.tasks)
    return scenario, context, frontend, device, profile


@pytest.mark.parametrize('captured', [False, True])
def test_planner_replaces_driver_owner_once_preserving_boundaries_and_all_device_work(captured, monkeypatch):
    from heterollm_sim import planner
    from heterollm_sim.event_kernel import UnifiedEventKernel
    scenario, context, frontend, device, profile = measured_driver_case(captured)
    monkeypatch.setattr(planner, '_gpu_profiles', lambda *args: (SimpleNamespace(kernel_model=profile), None))
    old = (*frontend, *apply_cuda_graph_runtime_costs(device, profile))
    with planner._compilation_scope(scenario, context):
        fixed = planner._apply_planned_graph_launches((*frontend, *device))
    old_by_id, fixed_by_id = ({task.task_id: task for task in rows} for rows in (old, fixed))
    submit = next(task for task in frontend if task.metadata.get('runtime_phase') == 'driver_submit')
    replacement = fixed_by_id[submit.task_id]
    assert submit.demands[0].service_ns == 250
    assert not replacement.demands
    assert replacement.dependencies == submit.dependencies
    assert replacement.metadata['cost_owner_replacement']['removed_service_ns'] == 250
    assert replacement.metadata['cost_owner_replacement']['physical_invocation_group_id'] == 'group0'
    assert set(old_by_id) == set(fixed_by_id)
    assert all(task == fixed_by_id[task.task_id] for task in old if task.task_id != submit.task_id)
    assert sum(d.bytes_moved for task in old for d in task.demands) == sum(
        d.bytes_moved for task in fixed for d in task.demands)
    def run(tasks):
        kernel = UnifiedEventKernel.from_closed_graph(tasks)
        events = {}
        while (event := kernel.step()) is not None:
            events[event.task.task_id] = event
        kernel.assert_drained()
        return kernel.makespan_ns, events
    old_end, old_events = run(old)
    new_end, new_events = run(fixed)
    assert old_end - new_end == pytest.approx(250)
    for task in device:
        assert old_events[task.task_id].demands == new_events[task.task_id].demands
        assert old_events[task.task_id].start_ns - new_events[task.task_id].start_ns == pytest.approx(250)
    from heterollm_sim.scalable_serving import execute_cost_schedule
    from heterollm_sim.serving import _OnlineRuntime
    direct = execute_cost_schedule(SimpleNamespace(tasks=fixed, resource_capacities={}, resource_owners={}),
                                   retain_task_metadata=False)
    rows, error = planner._compact_execution_stages(scenario, direct.execution_records,
        ({'group_id': 'group0', 'request_ids': ('request',)},))
    assert error is None
    stages = planner._trusted_execution_stages(rows)
    staged, _ = _OnlineRuntime._stage_task_specs(stages,
        {stage.stage_id: 0 for stage in stages}, 'driver-owner-integration')
    staged_end, _ = run(staged)
    assert staged_end == pytest.approx(new_end)


@pytest.mark.parametrize('mutation', ['group', 'multiple_groups', 'no_host_cost', 'wrong_host_group',
                                    'different_device', 'inactive_device', 'nonancestor', 'resource', 'bytes', 'duplicate'])
def test_driver_ownership_mismatch_is_rejected_without_removing_device_work(mutation):
    from heterollm_sim.cuda_graph_costs import replace_measured_driver_submission_costs
    scenario, _, frontend, device, profile = measured_driver_case(True)
    tasks = list((*frontend, *apply_cuda_graph_runtime_costs(device, profile)))
    index = next(i for i, task in enumerate(tasks) if task.metadata.get('runtime_phase') == 'driver_submit')
    submit = tasks[index]
    if mutation == 'group':
        tasks[index] = replace(submit, metadata={**submit.metadata, 'physical_invocation_group_ids': ('other',)})
    elif mutation == 'multiple_groups':
        tasks[index] = replace(submit, metadata={**submit.metadata, 'physical_invocation_group_ids': ('group0', 'other')})
    elif mutation == 'no_host_cost':
        tasks = [task for task in tasks if not task.metadata.get('phase', '').startswith('cuda_runtime_')]
    elif mutation == 'wrong_host_group':
        tasks = [replace(task, metadata={**task.metadata, 'operator_invocation_group_id': 'other'})
                 if task.metadata.get('phase', '').startswith('cuda_runtime_') else task for task in tasks]
    elif mutation in ('different_device', 'inactive_device'):
        tasks[index] = replace(submit, metadata={**submit.metadata, 'target_component': 'gpu1'})
    elif mutation == 'nonancestor':
        tasks[index] = replace(submit, task_id='unrelated-boundary')
    elif mutation in ('resource', 'bytes'):
        change = {'resource_id': 'gpu0.tensor'} if mutation == 'resource' else {'bytes_moved': 1024}
        tasks[index] = replace(submit, demands=(replace(submit.demands[0], **change),))
    elif mutation == 'duplicate':
        tasks.append(replace(submit, task_id='duplicate-submission'))
    with pytest.raises(ValueError, match='CUDA driver submission'):
        profiles = {'gpu0': profile} if mutation == 'inactive_device' else {'gpu0': profile, 'gpu1': profile}
        replace_measured_driver_submission_costs(tasks, profiles,
                                                scenario.host_orchestration_profile.submission_resource_id)


def test_driver_owner_replacement_does_not_remove_other_submissions():
    from heterollm_sim.cuda_graph_costs import replace_measured_driver_submission_costs
    scenario, _, frontend, device, profile = measured_driver_case(False)
    submit = next(task for task in frontend if task.metadata.get('runtime_phase') == 'driver_submit')
    other_device = replace(submit, task_id='other-device-submit',
                           metadata={**submit.metadata, 'target_component': 'gpu1'})
    other_kind = replace(submit, task_id='other-submit-kind',
                         metadata={**submit.metadata, 'event_kind': 'other_host_submit'})
    priced = (*frontend, *apply_cuda_graph_runtime_costs(device, profile), other_device, other_kind)
    fixed = replace_measured_driver_submission_costs(priced, {'gpu0': profile},
                                                    scenario.host_orchestration_profile.submission_resource_id)
    assert other_device in fixed and other_kind in fixed
    assert replace_measured_driver_submission_costs(priced, {}, 'unused') == priced


@pytest.mark.parametrize('captured', [False, True])
def test_real_qwen_layer_keeps_physical_service_when_driver_owner_changes(captured, monkeypatch):
    from heterollm_sim import planner
    from heterollm_sim.event_kernel import UnifiedEventKernel
    scenario, context, frontend, _, profile = measured_driver_case(captured)
    with planner._compilation_scope(scenario, context):
        layer = planner._execution_layers(scenario)[0]
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        planner._compile_parallel_layer_body(builder, scenario, planner._parallel_plan(scenario),
            planner._topology_router(scenario), layer, token_batch=1, context_tokens=513,
            kv_read_tokens=512, kv_append_tokens=1, kv_materialized_tokens=1,
            linear_state_runtime=None, phase='decode', dependencies=(frontend[-1].task_id,))
        device = planner._promote_physical_allocation_extents(builder.tasks)
    device = tuple(replace(task, metadata={**task.metadata,
        'operator_invocation_group_id': 'group0'}) for task in device)
    runtime = CudaGraphRuntime()
    invocation = CudaGraphInvocation('cuda:0', 'first-node', 0, ((1,),), True, 'compatible')
    transition = runtime.prepare(invocation, host_time_us=0)
    if captured:
        runtime.commit(transition)
        transition = runtime.prepare(invocation, host_time_us=0)
    markers = tuple(task.task_id for task in device if not task.demands)
    device = bind_cuda_graph_tasks(device, transition,
        member_task_ids=tuple(task.task_id for task in device),
        device_marker_task_ids=markers, device_memory_component_ids=('gddr0',),
        topology='chain', node_count=2, node_count_basis='independent_test_cost')
    assert any(task.metadata.get('physical_memory_config') for task in device)
    old = (*frontend, *apply_cuda_graph_runtime_costs(device, profile))
    monkeypatch.setattr(planner, '_gpu_profiles', lambda *args: (SimpleNamespace(kernel_model=profile), None))
    with planner._compilation_scope(scenario, context):
        fixed = planner._apply_planned_graph_launches((*frontend, *device))
    def execute(tasks):
        kernel = UnifiedEventKernel.from_closed_graph(tasks,
            resource_capacities=planner._scenario_resource_capacities(scenario),
            resource_owners=planner._scenario_resource_owners(scenario), capture_physical_details=False)
        events = {}
        while (event := kernel.step()) is not None:
            events[event.task.task_id] = event
        kernel.assert_drained()
        return kernel.makespan_ns, events
    old_end, old_events = execute(old)
    fixed_end, fixed_events = execute(fixed)
    # All device demands (including realized DRAM work) remain identical;
    # only the independently measured host driver's former 250 ns disappear.
    maximum_physical_service_difference_ns = 0.0
    for task in device:
        before, after = old_events[task.task_id], fixed_events[task.task_id]
        assert before.demands == after.demands
        # The physical work starts earlier, so absolute controller timestamps
        # move. Service, row/burst counters, bytes and energy must not change.
        timestamps = {'arrival_ns', 'completion_ns', 'resource_intervals', 'service_ns',
                      'resource_last_intervals', 'resource_interval_payloads'}
        physical_before = before.task.metadata.get('physical_execution', {})
        physical_after = after.task.metadata.get('physical_execution', {})
        # Subtracting two shifted absolute double-precision timestamps can
        # change their service difference by a few billionths of a ns.
        maximum_physical_service_difference_ns = max(maximum_physical_service_difference_ns,
            abs(physical_before.get('service_ns', 0) - physical_after.get('service_ns', 0)))
        assert maximum_physical_service_difference_ns < 1e-6
        assert {key: value for key, value in physical_before.items() if key not in timestamps} == {
            key: value for key, value in physical_after.items() if key not in timestamps}
    assert old_end - fixed_end == pytest.approx(250)
    print({'captured': captured, 'physical_device_tasks': len(device),
           'old_makespan_ns': old_end, 'fixed_makespan_ns': fixed_end,
           'removed_host_service_ns': 250, 'physical_demand_difference': 0,
           'maximum_physical_service_roundoff_ns': maximum_physical_service_difference_ns})
