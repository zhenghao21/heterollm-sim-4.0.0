"""Residency and live-state readers use the current producer contracts."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.execution_control import ExecutionControl, ExecutionProgress
from heterollm_sim.ir import ComponentSpec, HardwareSpec
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.memory_types import make_hbm_config
from heterollm_sim.residency import AllocationResidencyManager, PhysicalPool
from heterollm_sim.workspace_memory import _device_workspace_reservation


@pytest.mark.parametrize("with_backing", [False, True])
def test_residency_rejects_old_home_input_before_registering(with_backing):
    manager = AllocationResidencyManager("hbm", 256)
    arguments = {"home": "old"}
    if with_backing:
        arguments["backing"] = "hbf"
    with pytest.raises(TypeError, match="home"):
        manager.register("data", 128, **arguments)
    assert not manager.allocations and manager.resident_bytes == 0


@pytest.mark.parametrize("injected_pool", [False, True])
def test_residency_current_views_and_explicit_leases_keep_one_owner(injected_pool):
    manager = (AllocationResidencyManager(PhysicalPool("hbm", 256, page_size_bytes=64))
               if injected_pool else AllocationResidencyManager("hbm", 256, page_size_bytes=64))
    owner = manager.register("data", 128, backing="hbf", resident_bytes=128, dirty=True)
    view = manager.create_view("slice", "data", offset_bytes=64, size_bytes=64)
    lease = manager.acquire_lease("consumer-lease", "slice", "consumer")
    assert owner.allocation_id == lease.owner_id == view.owner_id == "data"
    assert owner.backing == "hbf" and owner.physical_owner == "hbm"
    assert manager.resident_bytes == 128 and view.capacity_bytes == 0
    manager.release_lease(lease.lease_id)
    manager.release_view(view.view_id)
    events = manager.evict("data", 128)
    assert sum(event.dirty_writeback_bytes for event in events) == 128
    assert any("consumer" in event.consumer_task_ids for event in events)
    for target, obsolete in ((owner, "home"), (owner, "owner_id"),
                             (view, "physical_owner"), (manager, "lease")):
        assert not hasattr(target, obsolete)
    manager.assert_consistent()


def test_workspace_maps_current_component_owner_not_retired_top_level_field(monkeypatch):
    from heterollm_sim import planner
    config = make_hbm_config(capacity_bytes=4096)
    component = ComponentSpec("memory", "hbm", capacity_bytes=4096, metadata={
        "physical_memory_config": config, "memory_service_owner": "canonical.memory",
        "physical_owner": "obsolete.memory",
    })
    task = TaskSpec("read", "request", "read", TaskCategory.MEMORY, metadata={
        "physical_memory_config": config, "physical_owner": "canonical.memory",
        "physical_allocations": ({"buffer_id": "temporary", "generation": 3, "size_bytes": 64},),
    })
    scenario = SimpleNamespace(llama_cpp_config=SimpleNamespace(ubatch=1, batch=1, parallel=1),
                               hardware=HardwareSpec("hardware", components=(component,), links=()))
    monkeypatch.setattr(planner, "_mtp_configuration", lambda _: None)
    monkeypatch.setattr(planner, "materialize_requests", lambda _: (
        SimpleNamespace(request_id="request", prompt_tokens=1, output_tokens=1),))
    monkeypatch.setattr(planner, "_compilation_scope", lambda _: nullcontext())
    monkeypatch.setattr(planner, "_lower_serving_cohort", lambda *_: SimpleNamespace(
        schedule=SimpleNamespace(tasks=(task,))))
    monkeypatch.setattr(planner, "_promote_physical_allocation_extents", lambda tasks: tasks)
    result = _device_workspace_reservation(scenario)
    assert result["bytes_by_component"] == {"memory": 64}


def kernel_events(lanes):
    tasks = tuple(TaskSpec(name, "request", name, TaskCategory.COMPUTE,
        demands=(ResourceDemand("wire", duration),)) for name, duration in (("a", 5.), ("b", 7.), ("c", 11.)))
    kernel = UnifiedEventKernel(resource_capacities={"wire": lanes})
    kernel.submit(tasks)
    events = []
    while kernel.has_active_tasks:
        events.append(kernel.step())
    return tuple(events)


@pytest.mark.parametrize("lanes,expected", [(1, 23.), (2, 16.)])
def test_live_metrics_keep_current_optional_lane_shape(lanes, expected):
    collector = LiveCohortMetrics()
    events = kernel_events(lanes)
    for event in events:
        collector.observe(event)
    predecessor = events[-1].resource_predecessors["wire"]
    assert ("lane" in predecessor) == (lanes > 1)
    result = collector.summary()
    assert result.makespan_ns == expected
    assert result.critical_path_category_ns == {TaskCategory.COMPUTE: expected}


@pytest.mark.parametrize("missing", ["task_id", "resource_id", "end_ns"])
def test_live_metrics_do_not_fabricate_missing_predecessor_identity_or_time(missing):
    collector = LiveCohortMetrics()
    first, second, last = kernel_events(1)
    collector.observe(first)
    collector.observe(second)
    predecessor = dict(last.resource_predecessors["wire"])
    predecessor.pop(missing)
    with pytest.raises(KeyError, match=missing):
        collector.observe(replace(last, resource_predecessors={"wire": predecessor}))


def test_execution_control_keeps_optional_progress_callback_and_typed_units():
    ExecutionControl().report("schedule", 0, 1)
    events = []
    control = ExecutionControl(progress_callback=events.append)
    control.report("schedule", 1, 2)
    control.report("custom", 3, unit="pages")
    assert all(isinstance(event, ExecutionProgress) for event in events)
    assert [(event.completed, event.unit) for event in events] == [(1, "schedule_tasks"), (3, "pages")]
