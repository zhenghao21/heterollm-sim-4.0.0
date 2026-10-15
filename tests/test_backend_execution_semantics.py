"""Cross-backend scheduler and accounting invariants."""
from dataclasses import replace
from random import Random
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, RunManifest, TaskCategory, TaskSpec, RetentionPolicy
from heterollm_sim.engine import ScheduleIR, simulate_schedule
from heterollm_sim.event_kernel import UnifiedEventKernel, CompiledGraphExecutor
from heterollm_sim.data_motion import LinkService, PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.memory_types import make_hbm_config, make_hbf_config
from heterollm_sim.metrics import summarize_metrics
from heterollm_sim.scalable_serving import execute_cost_schedule
from heterollm_sim.schema_v4 import SIMULATION_SCHEMA_VERSION
from heterollm_sim.streaming_des import _CriticalPathAccumulator, _ResultAccumulator, _merge_storage_traffic, _merge_dram_traffic
from heterollm_sim.planner import _summarize_nand_task_traffic
from heterollm_sim.reporting import _sum_batch_storage_traffic, _sum_batch_dram_traffic


def manifest():
    return RunManifest(schema_version=SIMULATION_SCHEMA_VERSION, run_id="audit", random_seed=0,
                       simulator_version="audit", model_name="audit", hardware_name="audit", workload_name="audit")


def core_schedule(lanes=1, limit=100):
    config = make_hbm_config(capacity_bytes=1048576, channels=lanes,
                             pseudo_channels_per_channel=1, max_expanded_segments=limit)
    tasks = (TaskSpec("a-core", "r", "core read", TaskCategory.MEMORY,
        metadata={"physical_memory_config": config, "physical_owner": "memory",
                  "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 64*lanes},)}),
        *(TaskSpec("z-compute"+str(i), "r", "wire consumer", TaskCategory.COMPUTE,
                   demands=(ResourceDemand("wire", 100.+i),)) for i in range(lanes)))
    return ScheduleIR(manifest(), tasks, resource_capacities={"wire": lanes},
                      resource_owners={f"memory:dram:data:{i}": "wire" for i in range(lanes)})


@pytest.mark.parametrize("lanes", [1, 2])
@pytest.mark.parametrize("limit", [1, 100])
@pytest.mark.parametrize("details", [False, True])
def test_physical_release_point_drives_every_critical_path_reducer(lanes, limit, details):
    schedule = core_schedule(lanes, limit)
    cost = execute_cost_schedule(schedule, retain_task_metadata=details)
    trace = simulate_schedule(schedule)
    kernel = UnifiedEventKernel.from_closed_graph(schedule.tasks,
        resource_capacities=schedule.resource_capacities, resource_owners=schedule.resource_owners,
        capture_physical_details=details)
    live, streaming = LiveCohortMetrics(), _CriticalPathAccumulator()
    while kernel.has_active_tasks:
        event = kernel.step()
        live.observe(event)
        streaming.observe(event.task, start_ns=event.start_ns, end_ns=event.end_ns,
                          demands=event.demands, resource_predecessors=event.resource_predecessors,
                          resource_lanes=event.resource_lanes)
    expected = {TaskCategory.MEMORY: 30.5, TaskCategory.COMPUTE: 100.+lanes-1}
    assert trace.makespan_ns == pytest.approx(sum(expected.values()))
    for categories in (summarize_metrics(trace).critical_path_category_ns,
                       cost.critical_path_category_ns, live.summary().critical_path_category_ns,
                       streaming.global_categories()):
        assert categories == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(6))
def test_compiled_live_and_incremental_reductions_agree_on_causal_multilane_graphs(seed):
    rng = Random(seed)
    tasks = []
    for i in range(24):
        dependencies = tuple(t.task_id for t in rng.sample(tasks, min(len(tasks), rng.randrange(3))))
        demands = tuple(ResourceDemand(r, rng.choice([0., .125, 2.5, 10.]),
                                       bytes_moved=i*7)
                        for r in rng.sample(["a", "b", "c"], rng.randrange(4)))
        tasks.append(TaskSpec(str(i).zfill(2), "r", "work", rng.choice(list(TaskCategory)),
                              dependencies=dependencies, demands=demands,
                              earliest_start_ns=rng.choice([0., 5., 20.])))
    capacities = {"a": 2, "b": 1, "c": 3}
    schedule = ScheduleIR(manifest(), tuple(tasks), resource_capacities=capacities)
    reference = simulate_schedule(schedule)
    compiled = CompiledGraphExecutor()
    for scale in (1., 2.):
        scaled = replace(schedule, tasks=tuple(replace(t, demands=tuple(
            replace(d, service_ns=d.service_ns*scale) for d in t.demands)) for t in tasks))
        actual = execute_cost_schedule(scaled, compiled_executor=compiled)
        direct = simulate_schedule(scaled)
        assert actual.makespan_ns == direct.makespan_ns
        assert actual.resource_busy_ns == direct.resource_busy_ns
        assert actual.critical_path_category_ns == summarize_metrics(direct).critical_path_category_ns
    assert compiled.stats["hits"] == 1
    kernel = UnifiedEventKernel.from_closed_graph(tasks, resource_capacities=capacities)
    live, incremental = LiveCohortMetrics(), _CriticalPathAccumulator()
    incremental.register_chunk(SimpleNamespace(tasks=tasks, terminal_task_id=tasks[-1].task_id))
    while kernel.has_active_tasks:
        event = kernel.step()
        live.observe(event)
        incremental.observe(event.task, start_ns=event.start_ns, end_ns=event.end_ns,
                            demands=event.demands, resource_predecessors=event.resource_predecessors,
                            resource_lanes=event.resource_lanes)
    expected = summarize_metrics(reference).critical_path_category_ns
    assert live.summary().critical_path_category_ns == expected
    assert incremental.global_categories() == expected


def test_zero_byte_link_bill_is_a_noop():
    link = LinkService("link", "link", "wire", 10., latency_ns=100.)
    assert link.demands(0) == ()
    assert link.bill(0) == 0.
    assert link.bill(100) == 110.


def test_resolved_core_combined_counters_keep_integer_types_and_fractional_time():
    task = replace(core_schedule().tasks[0], metadata={
        **core_schedule().tasks[0].metadata,
        "memory_accesses": (
            {"operation": "read", "address": 0, "byte_count": 64},
            {"operation": "read", "address": 64, "byte_count": 64},
        )})
    result = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    combined = next(iter(result.metadata["physical_execution_by_owner"].values()))
    for key, value in (("physical_read_bytes", 128), ("burst_count", 2)):
        assert combined[key] == value
        assert type(combined[key]) is int
    assert isinstance(combined["queue_wait_ns"], float)


@pytest.mark.parametrize("details", [False, True])
def test_overlapping_core_envelopes_do_not_add_the_same_internal_wait_twice(details):
    original = core_schedule().tasks[0]
    schedule = replace(core_schedule(), tasks=(original, replace(original, task_id="b-core")))
    trace = simulate_schedule(schedule)
    cost = execute_cost_schedule(schedule, retain_task_metadata=details)
    kernel = UnifiedEventKernel.from_closed_graph(schedule.tasks,
        resource_owners=schedule.resource_owners, capture_physical_details=details)
    live, incremental = LiveCohortMetrics(), _CriticalPathAccumulator()
    while kernel.has_active_tasks:
        event = kernel.step()
        live.observe(event)
        incremental.observe(event.task, start_ns=event.start_ns, end_ns=event.end_ns,
                            demands=event.demands, resource_predecessors=event.resource_predecessors,
                            resource_lanes=event.resource_lanes)
    assert all(t.start_ns == 0 for t in trace.tasks)
    for categories in (summarize_metrics(trace).critical_path_category_ns, cost.critical_path_category_ns,
                       live.summary().critical_path_category_ns, incremental.global_categories()):
        assert categories == {TaskCategory.MEMORY: trace.makespan_ns}


def test_streaming_nand_ledger_preserves_v2_counts_observations_and_unknown_fields():
    runtime = PhysicalRuntimeContext()
    config = make_hbf_config(capacity_bytes=1048576, detailed_observability=True)
    ledgers = []
    for index, operation in enumerate(("read", "write")):
        task = TaskSpec(str(index), "r", operation, TaskCategory.MEMORY,
            metadata={"physical_memory_config": config, "physical_owner": "nand",
                      "memory_accesses": ({"operation": operation, "address": 0, "byte_count": 100},)})
        ledgers.append(_summarize_nand_task_traffic([resolve_physical_task(task, runtime, 0.)]))
    total = {}
    for ledger in ledgers:
        _merge_storage_traffic(total, ledger)
    online = _sum_batch_storage_traffic(SimpleNamespace(serving=SimpleNamespace(runtime_kernel_metrics={}, batches=[
        SimpleNamespace(cost=SimpleNamespace(metadata={"storage_traffic": ledger})) for ledger in ledgers])))
    for key in ("logical_read_bytes", "logical_write_bytes", "internal_transfer_bytes", "pages_read",
                "pages_programmed", "host_outstanding_at_admission_peak", "page_access_payload_utilization"):
        assert total[key] == online[key]
    assert total["nand_observations"] == online["nand_observations"]
    assert total["metric_availability"]["nand_read_queue_depth"]["status"] == "recorded"
    assert total["media_waves"] is None
    _merge_storage_traffic(total, {"task_count": 1})
    assert total["internal_transfer_bytes"] is None
    assert total["metric_availability"]["internal_transfer_bytes"]["status"] == "partial"
    assert total["metric_availability"]["nand_read_queue_depth"]["status"] == "partial"


@pytest.mark.parametrize("role", [None, "construction_preview", "runtime_dispatch"])
def test_chunk_registration_does_not_bill_an_unexecuted_media_preview(role):
    task = TaskSpec("read", "r", "read", TaskCategory.MEMORY,
        metadata={"physical_memory_config": make_hbf_config(capacity_bytes=1048576),
                  "physical_owner": "nand",
                  "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 100},)})
    preview = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    metadata = dict(preview.metadata)
    metadata.pop("physical_execution_role", None)
    if role is not None:
        metadata["physical_execution_role"] = role
    authored = replace(preview, metadata=metadata)
    accumulator = _ResultAccumulator(RetentionPolicy.EXACT, 100)
    accumulator.register_chunk(SimpleNamespace(tasks=(authored,), terminal_task_id=authored.task_id))
    assert not accumulator.storage_traffic
    kernel = UnifiedEventKernel.from_closed_graph((authored,))
    event = kernel.step()
    accumulator.observe(event.task, start_ns=event.start_ns, end_ns=event.end_ns,
        dependency_ready_ns=event.dependency_ready_ns, effective_ready_ns=event.effective_ready_ns,
        demands=event.demands, resource_predecessors=event.resource_predecessors,
        resource_lanes=event.resource_lanes)
    expected = _summarize_nand_task_traffic((event.task,))
    assert accumulator.storage_traffic["task_count"] == expected["task_count"] == 1
    assert accumulator.storage_traffic["resource_totals"] == expected["resource_totals"]


@pytest.mark.parametrize("medium", ["dram", "nand"])
def test_resource_owner_is_not_misattributed_across_batch_or_stream_boundaries(medium):
    ledgers = [{"task_count": 1, "resource_totals": {"wire": {
        "owner": owner, "bytes_moved": 64, "service_ns": 2.}}}
        for owner in ("a", "b")]
    merge = _merge_dram_traffic if medium == "dram" else _merge_storage_traffic
    total = {}
    for ledger in ledgers:
        merge(total, ledger)
    key = "dram_traffic" if medium == "dram" else "storage_traffic"
    report = _sum_batch_dram_traffic if medium == "dram" else _sum_batch_storage_traffic
    online = report(SimpleNamespace(serving=SimpleNamespace(runtime_kernel_metrics={}, batches=[
        SimpleNamespace(cost=SimpleNamespace(metadata={key: ledger})) for ledger in ledgers])))
    for result in (total, online):
        assert result["resource_totals"]["wire"] == {
            "owner": "wire", "bytes_moved": 128, "service_ns": 4.}
    if medium == "dram":
        assert total["refresh_wait_ns"] is None
        assert total["metric_availability"]["refresh_wait_ns"]["status"] == "not_recorded"
