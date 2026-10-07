"""Actual event reduction stays consistent with the kernel and closed path."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import KernelEvent, UnifiedEventKernel
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.planner import _resource_busy_by_direction, _summarize_dram_task_traffic
from heterollm_sim.scalable_serving import execute_cost_schedule
from test_dram_traffic_summary import _config


def task(name, resource, duration, *, deps=(), metadata=None, earliest=0):
    return TaskSpec(name, "r", name, TaskCategory.COMPUTE, dependencies=deps,
        demands=(ResourceDemand(resource, duration, bytes_moved=3, energy_pj=duration * 0.3),),
        earliest_start_ns=earliest, metadata=metadata or {})


def observed(tasks, *, capacities=None, owners=None, origin=0.0):
    kernel = UnifiedEventKernel.from_closed_graph(tasks, resource_capacities=capacities or {},
                                                  resource_owners=owners or {})
    accumulator = LiveCohortMetrics(resource_owners=owners, origin_ns=origin)
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
        accumulator.observe(event)
    return accumulator, kernel, events


@pytest.mark.parametrize("capacity", [1, 2])
def test_closed_schedule_matches_every_summary_field_and_resource_busy(capacity):
    tasks = (task("a.qkv", "logical-a", 10.2, metadata={"analytical_ops": 17, "operator_id": "qkv"}),
             task("b.qkv", "logical-b", 7.3, metadata={"model_operator_id": "model-qkv"}),
             task("c", "cpu", 2.1, metadata={"event_kind": "linear_state_read", "bytes": 71}),
             task("d", "logical-a", 0.0, deps=("a.qkv",)),
             task("e", "logical-b", 1.1, deps=("b.qkv", "d"), metadata={"coverage_component": "custom"}))
    owners, capacities = {"logical-a": "physical", "logical-b": "physical"}, {"physical": capacity}
    accumulator, kernel, _ = observed(tasks, capacities=capacities, owners=owners)
    expected = execute_cost_schedule(SimpleNamespace(tasks=tasks, resource_capacities=capacities,
        resource_owners=owners), retain_task_metadata=False)
    actual = accumulator.summary()
    # Original task retention is for preview replay, not part of live facts.
    assert replace(actual, execution_records=()) == replace(expected, execution_records=())
    assert actual.resource_busy_ns == kernel.resource_busy_ns
    assert actual.linear_state_bytes["read"] == 71
    assert actual.coverage["full_attention"]["operator_ids"] == ["model-qkv", "qkv"]
    assert all(record.original_task is None for record in actual.execution_records)


def test_physical_events_use_actual_core_traffic_and_compact_records():
    physical = TaskSpec("physical", "r", "memory", TaskCategory.MEMORY,
        metadata={"physical_memory_config": _config(), "physical_owner": "mem",
                  "physical_energy_pj_per_byte": 2.0,
                  "memory_accesses": ({"operation": "read", "address": 1, "byte_count": 70,
                                       "physical_owner": "mem"},
                                      {"operation": "write", "address": 257, "byte_count": 3,
                                       "physical_owner": "mem"})})
    accumulator, kernel, events = observed((physical,))
    metadata = accumulator.metadata()
    expected = _summarize_dram_task_traffic(tuple(event.task for event in events))
    assert metadata["dram_traffic"] == expected
    assert metadata["dram_traffic"]["logical_read_bytes"] == 70
    assert metadata["dram_traffic"]["physical_read_bytes"] == 128
    assert metadata["dram_traffic"]["physical_write_bytes"] == 64
    assert metadata["resource_busy_ns"] == kernel.resource_busy_ns
    assert accumulator.summary().energy_pj == sum(d.energy_pj for event in events for d in event.demands)
    record = accumulator.summary().execution_records[0]
    assert "memory_accesses" not in record.metadata
    assert "resource_interval_payloads" not in record.metadata["physical_execution"]
    assert metadata["physical_execution_scope"] == "persistent_live_kernel"
    assert metadata["cost_duration_semantics"] == "live_cohort_span"
    assert "energy_pj" not in metadata and "duration_ns" not in metadata


def test_actual_nand_read_write_pages_survive_live_and_online_reporting():
    from dataclasses import asdict
    from heterollm_sim.memory_types import NandConfig
    from heterollm_sim.reporting import _sum_batch_storage_traffic

    config = NandConfig(kind="HBF", page_bytes=4096, host_granularity_bytes=4096)
    physical = TaskSpec("nand", "r", "nand", TaskCategory.MEMORY, metadata={
        "physical_memory_config": asdict(config), "physical_owner": "hbf",
        "physical_energy_pj_per_byte": 12.0,
        "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 4096, "physical_owner": "hbf"},
                            {"operation": "write", "address": 4096, "byte_count": 4096, "physical_owner": "hbf"})})
    accumulator, _, events = observed((physical,))
    metadata = accumulator.metadata()
    summary = _sum_batch_storage_traffic(SimpleNamespace(serving=SimpleNamespace(
        batches=(SimpleNamespace(cost=SimpleNamespace(metadata=metadata)),))))
    assert summary["logical_read_bytes"] == summary["logical_write_bytes"] == 4096
    assert summary["logical_bytes"] == 8192
    assert summary["physical_read_bytes"] == summary["physical_write_bytes"] == 4096
    assert summary["pages_read"] == summary["pages_programmed"] == 1
    assert summary["energy_pj"] == sum(d.energy_pj for event in events for d in event.demands)
    assert summary["energy_pj"] == 8192 * 12
    for key in ("pages_read", "pages_programmed", "logical_read_bytes", "logical_write_bytes"):
        assert summary["metric_availability"][key]["status"] == "recorded"
    for key in ("read_operations", "program_operations", "pages_touched", "media_waves",
                "host_queue_wait_ns", "media_queue_wait_ns", "rmw_read_operations"):
        assert summary[key] is None
        assert summary["metric_availability"][key]["status"] == "not_recorded"


def test_span_origin_host_metrics_and_direction_use_real_events():
    tasks = (task("host", "cpu", 2, earliest=100, metadata={
        "orchestration_stage": "host_prefix", "submission_count": 2}),
        task("opaque", "device", 7, deps=("host",), metadata={"memory_direction": "write"}),
        task("suffix", "cpu", 3, deps=("opaque",), metadata={"orchestration_stage": "host_suffix"}))
    accumulator, _, events = observed(tasks, origin=None)
    assert accumulator.summary().makespan_ns == 12
    assert accumulator.summary().execution_records[0].start_ns == 100
    metadata = accumulator.metadata()
    assert metadata["execution_stage_makespan_ns"] == 12
    assert metadata["host_orchestration_ns"] == 5
    assert metadata["host_orchestration_task_count"] == 2
    assert metadata["host_submission_count"] == 2
    assert metadata["device_execution_ns"] == 7
    assert metadata["resource_busy_by_direction_ns"] == _resource_busy_by_direction(tuple(e.task for e in events))
    assert metadata["resource_busy_by_direction_ns"]["write"]["device"] == 7
    explicit = LiveCohortMetrics(origin_ns=50)
    for event in events:
        explicit.observe(event)
    assert explicit.summary().makespan_ns == 62


def test_external_cohort_resource_predecessor_does_not_reuse_older_local_path():
    accumulator = LiveCohortMetrics()
    first = task("a0", "shared", 3)
    last = task("a1", "shared", 2)
    accumulator.observe(KernelEvent(first, 0, 3, 0, 0, first.demands, {}))
    accumulator.observe(KernelEvent(last, 10, 12, 0, 0, last.demands,
        {"shared": {"task_id": "other-cohort", "resource_id": "shared"}}))
    assert accumulator.summary().makespan_ns == 12
    assert accumulator.summary().resource_busy_ns == {"shared": 5}
    assert accumulator.summary().critical_path_category_ns == {TaskCategory.COMPUTE: 2}


def test_separate_cohort_observers_report_shared_kernel_warm_row_state():
    first = TaskSpec("cohort-a", "r", "read", TaskCategory.MEMORY,
        metadata={"physical_memory_config": _config(), "physical_owner": "mem",
                  "memory_access": {"operation": "read", "address": 0, "byte_count": 64,
                                    "physical_owner": "mem"}})
    second = replace(first, task_id="cohort-b", dependencies=("cohort-a",))
    kernel = UnifiedEventKernel()
    kernel.submit((first,))
    first_event = kernel.step()
    kernel.submit((second,))
    second_event = kernel.step()
    cold, warm = LiveCohortMetrics(), LiveCohortMetrics()
    cold.observe(first_event)
    warm.observe(second_event)
    assert cold.metadata()["dram_traffic"]["row_misses"] == 1
    assert warm.metadata()["dram_traffic"]["row_hits"] == 1
    assert warm.metadata()["dram_traffic"]["row_misses"] == 0
    assert warm.summary().makespan_ns == second_event.end_ns - second_event.start_ns
    assert warm.metadata()["dram_traffic"] == _summarize_dram_task_traffic((second_event.task,))


def test_empty_and_repeated_observation_contracts():
    accumulator = LiveCohortMetrics()
    assert accumulator.summary().makespan_ns == accumulator.summary().task_count == 0
    metadata = accumulator.metadata()
    assert metadata["dram_traffic"]["task_count"] == 0
    assert metadata["storage_traffic"]["task_count"] == 0
    kernel = UnifiedEventKernel.from_closed_graph((task("once", "r", 1),))
    event = kernel.step()
    accumulator.observe(event)
    assert accumulator.summary().task_count == 1  # invalidates the empty cache
    with pytest.raises(ValueError, match="duplicate"):
        accumulator.observe(event)


@pytest.mark.parametrize("origin", [True, -1, float("nan"), float("inf")])
def test_invalid_explicit_origin_is_rejected(origin):
    with pytest.raises(ValueError, match="origin"):
        LiveCohortMetrics(origin_ns=origin)


def test_explicit_origin_cannot_hide_observed_work():
    accumulator, _, _ = observed((task("a", "r", 1),), origin=2)
    with pytest.raises(ValueError, match="later than"):
        accumulator.summary()
