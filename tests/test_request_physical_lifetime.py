from dataclasses import asdict

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import DramConfig, MemoryKind


def task(buffer_id, request_owners=()):
    config = DramConfig(kind=MemoryKind.GDDR, channels=1, data_lanes=2,
                        banks_per_group=2, rows_per_bank=32, row_bytes=256,
                        capacity_bytes=32768, interface_bandwidth_gb_s=64)
    return TaskSpec(
        task_id=buffer_id, request_id="actual-request", name=buffer_id,
        category=TaskCategory.MEMORY,
        demands=(ResourceDemand("dram0", 0, bytes_moved=64),),
        metadata={
            "physical_memory_config": asdict(config), "physical_owner": "dram0",
            "physical_energy_pj_per_byte": 2.0,
            "physical_allocation_scope": "closed_cohort",
            "persistent_request_buffers": {owner: [buffer_id] for owner in request_owners},
            "memory_accesses": ({"operation": "read", "buffer_id": buffer_id,
                "byte_count": 64, "offset_bytes": 0, "allocation_size_bytes": 128,
                "allocation_generation": 0, "physical_owner": "dram0", "resource_id": "dram0",
                "address_source": "stable_buffer_tensor_offset"},),
        },
    )


def drain(kernel):
    while kernel.has_active_tasks:
        assert kernel.step() is not None


def test_request_buffers_survive_cohort_and_release_without_resident_weights():
    kernel = UnifiedEventKernel.from_closed_graph((task("kv:a", ("a",)), task("weight")))
    with pytest.raises(ValueError, match="before their tasks complete"):
        kernel.release_request_physical_allocations("a")
    drain(kernel)
    allocator = kernel.physical_runtime.allocators["dram0"]
    assert allocator.get_allocation("kv:a", 0) is not None
    assert allocator.get_allocation("weight", 0) is not None
    kernel.release_request_physical_allocations("a")
    assert allocator.get_allocation("kv:a", 0) is None
    assert allocator.get_allocation("weight", 0) is not None
    assert not kernel._request_physical_buffers
    kernel.release_request_physical_allocations("a")


def test_explicit_shared_buffer_lives_until_last_request_finishes():
    kernel = UnifiedEventKernel.from_closed_graph((task("shared", ("a", "b")),))
    drain(kernel)
    allocator = kernel.physical_runtime.allocators["dram0"]
    kernel.release_request_physical_allocations("a")
    assert allocator.get_allocation("shared", 0) is not None
    kernel.release_request_physical_allocations("b")
    assert allocator.get_allocation("shared", 0) is None


def test_live_stage_replay_indexes_request_lifetime_and_keeps_state():
    from heterollm_sim.contracts import _PreparedExecutionStage, _PreparedExecutionTask
    from heterollm_sim.serving import _OnlineRuntime

    class Observer:
        def __init__(self):
            self.events = []

        def observe(self, event):
            self.events.append(event)

    raw = task("kv:a", ("a",))
    prepared = _PreparedExecutionTask(raw.task_id, (), ("a",), raw.demands,
                                     category=raw.category, metadata=raw.metadata)
    stage = _PreparedExecutionStage("stage", 0, (), ("a",), "gpu", 0, (prepared,))
    kernel = UnifiedEventKernel(capture_physical_details=False)
    observer = Observer()
    for index in range(2):
        _OnlineRuntime._replay_execution_stage_tasks(
            kernel, (stage,), {"stage": kernel.makespan_ns}, "live." + str(index),
            live_metrics=observer,
        )
        allocator = kernel.physical_runtime.allocators["dram0"]
        assert allocator.get_allocation("kv:a", 0) is not None
    assert len(observer.events) == 2
    assert all(event.task.metadata.get("physical_execution") for event in observer.events)
    kernel.release_request_physical_allocations("a")
    assert allocator.get_allocation("kv:a", 0) is None


def test_live_stage_replay_checks_cancellation_during_physical_execution():
    from heterollm_sim.contracts import _PreparedExecutionStage, _PreparedExecutionTask
    from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
    from heterollm_sim.serving import _OnlineRuntime

    prepared = []
    for index in range(300):
        raw = task("kv:" + str(index), ("a",))
        prepared.append(_PreparedExecutionTask(raw.task_id, (), ("a",), raw.demands,
                        category=raw.category, metadata=raw.metadata))
    stage = _PreparedExecutionStage("stage", 0, (), ("a",), "gpu", 0, tuple(prepared))
    kernel = UnifiedEventKernel(capture_physical_details=False)
    checks = []

    def cancel_after_first_poll():
        checks.append(True)
        return len(checks) > 1

    control = ExecutionControl(cancellation_callback=cancel_after_first_poll)
    with pytest.raises(ExecutionCancelledError):
        _OnlineRuntime._replay_execution_stage_tasks(
            kernel, (stage,), {"stage": 0.0}, "live", execution_control=control,
        )
    assert kernel.completed_count == 256
    assert kernel.has_active_tasks


def test_serialized_stage_keeps_unresolved_physical_work_and_category():
    import json
    from heterollm_sim.serving import _execution_stages_from_metadata, _OnlineRuntime
    from heterollm_sim.live_execution_metrics import LiveCohortMetrics

    raw = task("kv:a", ("a",))
    payload = {"execution_stage_source": "executed_task_dag_kernel_timeline",
        "execution_stages": [{"stage_id": "stage", "component_id": "gpu",
            "service_ns": 1.0, "request_ids": ["a"], "execution_tasks": [{
                "task_id": raw.task_id, "category": raw.category.value,
                "request_ids": ["a"], "resource_demands": [asdict(d) for d in raw.demands],
                "metadata": raw.metadata}]}]}
    stages, error = _execution_stages_from_metadata(json.loads(json.dumps(payload)))
    assert error is None
    assert stages[0].execution_tasks[0].category == TaskCategory.MEMORY
    assert stages[0].execution_tasks[0].metadata["memory_accesses"]
    assert stages[0].execution_tasks[0].demands[0].service_ns == 0
    kernel = UnifiedEventKernel(capture_physical_details=False)
    metrics = LiveCohortMetrics()
    _OnlineRuntime._replay_execution_stage_tasks(kernel, stages, {"stage": 0.0},
                                               "live", live_metrics=metrics)
    actual = metrics.summary()
    assert actual.makespan_ns > 0
    assert actual.energy_pj == 128
    assert TaskCategory.MEMORY in actual.category_time_ns
    assert actual.execution_records[0].metadata["physical_execution"]
