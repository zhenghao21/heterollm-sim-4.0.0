"""Common REF epochs require a closed real native ring and unchanged sources."""
from dataclasses import replace
import math

import numpy as np
import pytest

from heterollm_sim.contracts import (_PreparedExecutionTask, ResourceDemand,
    TaskCategory, TaskSpec)
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.native_collective_refresh_bound import (
    _epoch_source, native_collective_refresh_bound)
from heterollm_sim.native_layer_sampling import _independent_kernel
from heterollm_sim.native_window_producer import (_native_dag_ancestry,
    _native_refresh_source_qualification, _phase, _rank_private_phase_bound)
from test_native_refresh_source import source


def ring(config, owners, *, name="allreduce", dependencies=(), counts=None):
    tasks, previous = [], tuple(dependencies)
    for number in range(2*(len(owners)-1)):
        writes = []
        for index, owner in enumerate(owners):
            target = owners[(index+1) % len(owners)]
            count = 4096 if counts is None else counts[index]
            prefix = f"{name}.round{number:02d}.tx{index:02d}"
            common = {"collective_name": name, "collective_kind": "all_reduce",
                "event_kind": "collective_transfer", "algorithm": "ring"}
            for direction, selected, deps in (("read", owner, previous),
                    ("write", target, (prefix+".link",))):
                metadata = {**common, "memory_direction": direction,
                    "physical_owner": selected, "physical_memory_config": config,
                    "memory_accesses": ({"physical_owner": selected,
                        "resource_id": selected, "operation": direction,
                        "address": number*65536+index*8192, "byte_count": count},)}
                item = _PreparedExecutionTask(prefix+"."+direction, deps, ("r",),
                    (), category=TaskCategory.MEMORY, metadata=metadata)
                if direction == "read":
                    tasks.append(item)
                    tasks.append(_PreparedExecutionTask(prefix+".link", (item.task_id,), ("r",),
                        (ResourceDemand("link."+owner+"."+target, 504.551111),),
                        category=TaskCategory.COMMUNICATION,
                        metadata={"event_kind": "collective_transfer"}))
                else:
                    tasks.append(item)
                    writes.append(item.task_id)
        barrier = _PreparedExecutionTask(f"{name}.round{number:02d}.barrier",
            tuple(writes), ("r",), (), metadata={"event_kind": "collective_barrier"})
        tasks.append(barrier)
        previous = (barrier.task_id,)
    return tuple(tasks)


def fixture(*, ranks=2, arrival=0., counts=None):
    data = source(arrival=arrival)
    runtime, config, physical, core = data
    owners = ("hbm",)+tuple("hbm"+str(index) for index in range(1, ranks))
    for owner in owners[1:]:
        physical.runtime(config, owner)
    return data, {owner: {index} for index, owner in enumerate(owners)}, ring(config, owners, counts=counts)


def inputs(tasks):
    rows, bills = [], {}
    for task in tasks:
        metadata = task.metadata
        accesses = metadata.get("memory_accesses", ())
        if accesses:
            access, = accesses
            values = _phase(metadata["physical_memory_config"], access["byte_count"], access["operation"],
                independent_hbm_channels=True, allow_bounded_clock=True, native_refresh_reachable=True)
            phase, refresh = sum(values), values[1]
            resources = {"@physical-owner:"+access["physical_owner"]}
        else:
            phase, refresh = 0., 0.
            resources = {d.resource_id for d in task.demands}
        rows.append((task, phase, resources))
        bills[task.task_id] = {"refresh_ns": refresh}
    return rows, bills


def bound(data, owners, tasks, **kwargs):
    rows, bills = inputs(tasks)
    ancestry, _ = _native_dag_ancestry(tasks)
    result, proof = native_collective_refresh_bound(rows, bills,
        physical_runtime=data[2], native_runtime=data[0], owner_ranks=owners,
        exact_ancestry=ancestry, rank_bound=_rank_private_phase_bound,
        all_tasks=tasks, **kwargs)
    return result, proof, _rank_private_phase_bound(rows, owner_ranks=owners)[0]


def events(data, tasks, *, compiled=False):
    kernel, _executor = _independent_kernel(data[0]._execution_resource_kernel)
    origin = max(data[0].now, kernel.makespan_ns,
        max(data[2].timeline.ready_ns.values(), default=0.))
    specs = tuple(TaskSpec(task.task_id, "r", task.task_id, task.category,
        task.dependencies, task.demands, earliest_start_ns=origin,
        metadata=task.metadata) for task in tasks)
    if compiled:
        from heterollm_sim.event_kernel import CompiledGraphLayout
        return origin, kernel._drain_prevalidated_compiled(specs, CompiledGraphLayout.compile(specs))
    kernel.submit(specs)
    result = []
    while (event := kernel.step()) is not None:
        result.append(event)
    return origin, result


@pytest.mark.parametrize("ranks", (2, 8))
def test_closed_ring_unions_REF_only_and_does_not_advance_real_controllers(ranks):
    data, owners, tasks = fixture(ranks=ranks, arrival=3900.)
    packed = data[3]._hbm_numeric_state
    before = tuple(array.copy() for array in packed.arrays)
    calendars = {name: tuple(slots) for name, slots in data[2].timeline.lane_available.items()}
    value, proof, original = bound(data, owners, tasks)
    assert proof["qualified"], proof
    collective, = proof["collectives"]
    assert 0 < value < original
    assert collective["row_and_cache_phase_upper_ns"] > 0.
    assert collective["common_REF_phase_upper_ns"] > 0.
    assert collective["collective_phase_upper_ns"] == value
    assert collective["round_count"] == 2*(ranks-1)
    assert not proof["physical_state_advanced"]
    assert data[3]._hbm_numeric_state is packed
    assert all(np.array_equal(old, new) for old, new in zip(before, packed.arrays))
    assert calendars == {name: tuple(slots) for name, slots in data[2].timeline.lane_available.items()}


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize("arrival", (0., 3899.5, 3900., 1e8+.17))
def test_actual_native_heap_dispatches_all_reads_before_links_and_service_is_bounded(compiled, arrival):
    # Unequal read durations and prior native controller histories do not
    # let an early link/write get ahead of a same-release physical read.
    data, owners, tasks = fixture(arrival=arrival, counts=(4096, 8192))
    value, proof, _ = bound(data, owners, tasks)
    assert proof["qualified"], proof
    origin, actual = events(data, tasks, compiled=compiled)
    positions = {event.task.task_id: index for index, event in enumerate(actual)}
    for number in range(2):
        prefix = f"allreduce.round{number:02d}."
        reads = [positions[task.task_id] for task in tasks
            if task.task_id.startswith(prefix) and task.metadata.get("memory_direction") == "read"]
        others = [positions[task.task_id] for task in tasks
            if task.task_id.startswith(prefix) and task.metadata.get("event_kind") == "collective_transfer"
            and task.metadata.get("memory_direction") != "read"]
        assert max(reads) < min(others)
    collective, = proof["collectives"]
    full = collective["complete_native_active_upper_ns"]+collective["common_REF_phase_upper_ns"]
    assert max(event.end_ns for event in actual)-origin <= full
    assert all(event.task.metadata["physical_execution_by_owner"]
        for event in actual if event.task.metadata.get("memory_accesses"))


@pytest.mark.parametrize("change", ("missing_barrier", "extra_wait", "incomplete_round", "shared_link"))
def test_incomplete_or_shared_native_graph_keeps_existing_bound(change):
    data, owners, original = fixture()
    tasks = list(original)
    if change == "missing_barrier":
        tasks = [task for task in tasks if task.task_id != "allreduce.round00.barrier"]
    elif change == "extra_wait":
        tasks[0] = replace(tasks[0], dependencies=("unproven_foreign_wait",))
    elif change == "incomplete_round":
        tasks = [task for task in tasks if ".round01." not in task.task_id]
    else:
        link = tasks[1].demands[0]
        tasks[4] = replace(tasks[4], demands=(link,))
    value, proof, _ = bound(data, owners, tuple(tasks))
    assert math.isinf(value) and not proof["qualified"]
    assert proof["rejections"]


@pytest.mark.parametrize("alias", ("none", "into", "outof"))
@pytest.mark.parametrize("ordered", (False, True))
def test_direct_physical_port_writer_is_rejected_even_when_ordered_or_aliased(alias, ordered):
    data, owners, tasks = fixture()
    physical_id = "hbm:dram:data:0"
    resource = physical_id
    if alias == "into":
        resource = "ordinary.writer"
        data[2].timeline.resource_aliases[resource] = physical_id
    elif alias == "outof":
        resource = "ordinary.writer"
        data[2].timeline.resource_aliases[physical_id] = resource
    writer = _PreparedExecutionTask("direct", (), ("r",), (ResourceDemand(resource, 20000.),))
    if ordered:
        tasks = tuple(replace(task, dependencies=("direct",)) if not task.dependencies else task for task in tasks)
    value, proof, _ = bound(data, owners, (writer, *tasks))
    assert math.isinf(value) and not proof["qualified"]
    assert proof["rejections"][0]["reason"] == "common_REF_unknown_direct_physical_port_writer"


def test_external_same_owner_access_must_precede_all_roots_or_follow_terminal():
    data, owners, tasks = fixture()
    foreign = replace(tasks[0], task_id="foreign", dependencies=(tasks[0].task_id,),
        metadata={key: value for key, value in tasks[0].metadata.items() if key != "collective_name"})
    value, proof, _ = bound(data, owners, (*tasks, foreign))
    assert math.isinf(value)
    assert proof["rejections"][0]["reason"].startswith("common_REF_original_subgraph_not_closed:")


def test_two_collectives_preserve_direct_cross_group_edges_after_contraction():
    data, owners, first = fixture()
    second = ring(data[1], tuple(owners), name="next", dependencies=(first[-1].task_id,))
    value, proof, original = bound(data, owners, (*first, *second))
    assert proof["qualified"] and len(proof["collectives"]) == 2, proof
    assert value < original
    assert value == sum(group["collective_phase_upper_ns"] for group in proof["collectives"])
    rows, bills = inputs((*first, *second))
    ancestry, _ = _native_dag_ancestry((*first, *second))
    contracted = []
    def capture(rows, **kwargs):
        if any(task.task_id.startswith("@native-common-REF") for task, _p, _r in rows):
            contracted.extend(task for task, _p, _r in rows)
        return _rank_private_phase_bound(rows, **kwargs)
    native_collective_refresh_bound(rows, bills, physical_runtime=data[2], native_runtime=data[0],
        owner_ranks=owners, exact_ancestry=ancestry, rank_bound=capture, all_tasks=(*first, *second))
    assert len(contracted) == 2
    assert contracted[1].dependencies == (contracted[0].task_id,)


@pytest.mark.parametrize("change", ("fractional_interval", "grid_delay", "foreign_due", "external_callback"))
def test_common_epoch_requires_exact_native_integer_recurrence(change):
    data, owners, tasks = fixture(arrival=7800.)
    config = data[1]
    if change == "fractional_interval":
        config = replace(config, refresh_interval_ns=3899.5)
    elif change == "grid_delay":
        config = replace(config, refresh_interval_ns=3840.)
    elif change == "foreign_due":
        data[3]._hbm_numeric_state.arrays[6][0, 1] += .125
    else:
        from heterollm_sim.execution_control import ExecutionControl
        data[0].execution_control = ExecutionControl(progress_callback=lambda _event: None)
    with pytest.raises(ValueError):
        _epoch_source(config, "hbm", data[2], data[0])


def test_same_summary_source_cache_is_reused_without_changing_arrays(monkeypatch):
    data, owners, tasks = fixture()
    sources = {(owner, id(data[1])): _native_refresh_source_qualification(data[1], owner, data[2], data[0])
               for owner in owners}
    assert all(item["qualified"] for item in sources.values())
    def forbidden(*args, **kwargs):
        raise AssertionError("duplicate source reread inside same summary")
    monkeypatch.setattr("heterollm_sim.native_window_producer._native_refresh_source_qualification", forbidden)
    value, proof, _ = bound(data, owners, tasks, source_proofs=sources)
    assert proof["qualified"] and math.isfinite(value)


def test_revisiting_a_bank_keeps_all_original_REF_charges():
    data, owners, tasks = fixture(counts=(2*1024**2, 2*1024**2))
    value, proof, _ = bound(data, owners, tasks)
    assert math.isinf(value) and not proof["qualified"]
    assert proof["rejections"][0]["reason"] == "common_REF_unique_bank_common_bulk_graph_unproven"


def test_zero_service_link_demand_still_participates_in_resource_isolation():
    data, owners, original = fixture()
    tasks = list(original)
    tasks[1] = replace(tasks[1], demands=tasks[1].demands+(ResourceDemand("link.hidden", 0.),))
    foreign = _PreparedExecutionTask("foreign", (), ("r",),
        (ResourceDemand("link.hidden", 100000.),))
    value, proof, _ = bound(data, owners, (*tasks, foreign))
    assert math.isinf(value) and not proof["qualified"]
    assert proof["rejections"][0]["reason"].startswith("common_REF_original_subgraph_not_closed:")


def test_native_sibling_host_alias_into_HBM_port_does_not_share_absolute_REF_epoch():
    from heterollm_sim.memory_types import NandConfig
    data, owners, tasks = fixture()
    actual = data[2].runtime(NandConfig(kind="HBF", protocol_mode="generic_nand"), "nand").core.config
    host = actual.metadata["host_resource_id"]
    data[2].timeline.resource_aliases[host] = "hbm:dram:data:0"
    with pytest.raises(ValueError, match="foreign_resource_aliases_HBM_port"):
        _epoch_source(data[1], "hbm", data[2], data[0])
    value, proof, _ = bound(data, owners, tasks)
    assert math.isinf(value) and not proof["qualified"]


def test_real_native_mapper_and_lowerer_ring_qualifies_without_running_a_candidate():
    from exploring_hbf_native_config import build_native_comparison_scenario
    from exploring_hbf_native_pilot import build_pilot_scenario
    from heterollm_sim.control_plane import bootstrap_control_plane
    from heterollm_sim.planner import TopologyAwareBatchCostProvider
    from heterollm_sim.serving import (_OnlineRuntime, compile_serving_plan,
        BatchItem, BatchCohort, _lower_cost)
    from heterollm_sim.native_window_producer import summarize_native_stages

    scenario = build_native_comparison_scenario("llama3_405b", "HBF", "SHORT", tp=2,
        batch_size=1, model_override=build_pilot_scenario("hbm_tp1").model,
        prompt_override=8, output_override=4, warmup_override=0, measurement_override=2)
    bootstrap = bootstrap_control_plane(scenario, capture_physical_details=False)
    scenario = bootstrap.scenario
    runtime = _OnlineRuntime(compile_serving_plan(scenario),
        TopologyAwareBatchCostProvider(scenario, defer_physical_execution=True),
        execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
    request = runtime.plan.requests[0]
    item = BatchItem(request.request_id, "decode", 1, 7, kv_append_tokens=1, kv_materialized_tokens=1,
        main_tokens=1, draft_tokens=0, verifier_tokens=1, committed_tokens=1,
        logit_tokens=1, completion_cursor=8)
    cohort = BatchCohort("native-ring", "decode", 0., (item,))
    cost = _lower_cost(runtime.lowerer, scenario, cohort)
    stages, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
    assert stages and not fallback
    # The production layer sampler summarizes exactly one native layer.
    # Frontend/PD stages execute and pay their carry at its input boundary;
    # raw prepared cross-stage edges are not claimed as in-layer edges.
    stages = tuple(replace(stage, execution_tasks=tuple(task for task in stage.execution_tasks
        if task.metadata.get("layer_id") == "dense0")) for stage in stages
        if any(task.metadata.get("layer_id") == "dense0" for task in stage.execution_tasks))
    physical = bootstrap.kernel.physical_runtime
    before = dict(physical.timeline.ready_ns)
    summary = summarize_native_stages(stages, scenario, physical,
        allow_bounded_clock=True, native_runtime=runtime)
    accepted = [group for stage in summary["native_resource_order_proof"]
        for group in stage["closed_collective_refresh"].get("collectives", ())]
    rejected = [item for stage in summary["native_resource_order_proof"]
        for item in stage["closed_collective_refresh"].get("rejections", ())]
    assert len(accepted) == 2, rejected
    assert all(group["collective_phase_upper_ns"] < group["old_collective_phase_upper_ns"]
               for group in accepted)
    assert dict(physical.timeline.ready_ns) == before
