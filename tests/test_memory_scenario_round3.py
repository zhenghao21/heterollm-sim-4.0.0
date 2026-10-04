"""Round-three event-kernel integration checks for physical memory access."""

from dataclasses import replace

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.dram_core import DramCore
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import AccessRequest, make_ddr_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.reporting import run_scenario


def _config(**kwargs):
    defaults = dict(
        channels=1, banks_per_group=1, rows_per_bank=8, row_bytes=128,
        burst_bytes=64, open_ns=10, close_ns=10, read_latency_ns=5,
        burst_interval_ns=1, lane_bandwidth_gb_s=64,
    )
    defaults.update(kwargs)
    return make_ddr_config(**defaults)


def _task(task_id, address, *, config, owner="dram0", dependency=() , earliest=0.0):
    return TaskSpec(
        task_id=task_id, request_id=task_id, name=task_id,
        category=TaskCategory.MEMORY, dependencies=tuple(dependency),
        earliest_start_ns=earliest,
        metadata={
            "memory_access": {"operation": "read", "address": address, "byte_count": 64,
                              "physical_owner": owner},
            "physical_memory_config": config, "physical_owner": owner,
        },
    )


def _run(tasks):
    kernel = UnifiedEventKernel.from_closed_graph(tuple(tasks))
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
        kernel.release_completed(event.task.task_id)
    return events


def test_kernel_submits_physical_access_and_matches_direct_core():
    config = _config()
    event = _run((_task("r0", 0, config=config),))[0]
    direct = DramCore(config).execute(AccessRequest("r0", "read", 0, 64))
    physical = event.task.metadata["physical_execution"]
    assert event.task.metadata["physical_arrival_ns"] == 0.0
    assert event.end_ns == physical["completion_ns"]
    assert event.end_ns == direct.completion_ns
    assert physical["row_hits"] == direct.counters["row_hits"]
    assert physical["row_misses"] == direct.counters["row_misses"]


def test_same_row_hit_and_new_row_conflict_are_real_kernel_state():
    config = _config()
    first, hit, conflict = _run((
        _task("a", 0, config=config),
        _task("b", 64, config=config),
        _task("c", 128, config=config),
    ))
    assert first.task.metadata["physical_execution"]["row_misses"] == 1
    assert hit.task.metadata["physical_execution"]["row_hits"] >= 1
    assert conflict.task.metadata["physical_execution"]["row_conflicts"] == 1
    assert conflict.task.metadata["physical_completion_ns"] > hit.task.metadata["physical_completion_ns"]


def test_independent_channels_can_arrive_together_and_dependency_delays_child():
    config = _config(channels=2, interleave_bytes=64)
    first, second = _run((
        _task("ch0", 0, config=config),
        _task("ch1", 64, config=config),
    ))
    assert first.task.metadata["physical_arrival_ns"] == second.task.metadata["physical_arrival_ns"] == 0.0
    assert first.task.metadata["physical_execution"]["completion_ns"] == first.end_ns
    assert second.task.metadata["physical_execution"]["completion_ns"] == second.end_ns
    direct = DramCore(config).execute_batch((
        AccessRequest("ch0", "read", 0, 64, arrival_ns=0),
        AccessRequest("ch1", "read", 64, 64, arrival_ns=0),
    ))
    assert sorted(event.end_ns for event in (first, second)) == sorted(
        result.completion_ns for result in direct
    )

    parent, child = _run((
        _task("parent", 0, config=config),
        _task("child", 64, config=config, dependency=("parent",)),
    ))
    assert child.task.metadata["physical_arrival_ns"] >= parent.end_ns


def test_exact_and_split_submission_have_identical_physical_results():
    config = _config()
    exact = _run((_task("a", 0, config=config), _task("b", 128, config=config)))
    split_kernel = UnifiedEventKernel.from_closed_graph((_task("a", 0, config=config),))
    first = split_kernel.step(); assert first is not None
    split_kernel.release_completed("a")
    split_kernel.add_tasks((_task("b", 128, config=config),))
    second = split_kernel.step(); assert second is not None
    assert [e.task.metadata["physical_execution"]["completion_ns"] for e in exact] == [
        first.task.metadata["physical_execution"]["completion_ns"],
        second.task.metadata["physical_execution"]["completion_ns"],
    ]


def test_two_kernels_isolate_owner_row_state():
    config = _config()
    left = _run((_task("left", 0, config=config, owner="same"), _task("left2", 64, config=config, owner="same")))
    right = _run((_task("right", 0, config=config, owner="same"), _task("right2", 64, config=config, owner="same")))
    assert left[0].task.metadata["physical_execution"]["row_misses"] == 1
    assert right[0].task.metadata["physical_execution"]["row_misses"] == 1
    assert left[1].task.metadata["physical_execution"]["row_hits"] == right[1].task.metadata["physical_execution"]["row_hits"]


def test_static_schedule_exact_and_streaming_retention_have_same_makespan():
    scenario = build_reference_scenario()
    scenario = replace(scenario, workload=replace(
        scenario.workload, scheduler=replace(scenario.workload.scheduler, mode="static")
    ))
    exact = run_scenario(scenario, retention_policy="exact")
    streaming = run_scenario(scenario, retention_policy="streaming")
    assert exact.trace.makespan_ns == streaming.trace.makespan_ns
    assert exact.metrics.makespan_ns == streaming.metrics.makespan_ns
