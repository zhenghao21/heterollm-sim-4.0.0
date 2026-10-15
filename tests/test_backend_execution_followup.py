"""Execution interactions discovered in the full-backend follow-up audit."""
from dataclasses import asdict, replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("operation", ["read", "write"])
def test_nand_internal_stages_overlap_busy_host_wire(as_mapping, alias, operation):
    config = make_hbf_config(capacity_bytes=1048576)
    config = replace(config, metadata={**config.metadata, "host_resource_id": "wire"})
    owners = {"wire": "shared"} if alias else {}
    task = TaskSpec("z-io", "r", "io", TaskCategory.MEMORY, metadata={
        "physical_memory_config": asdict(config) if as_mapping else config,
        "physical_owner": "nand",
        "memory_accesses": ({"operation": operation, "address": 0, "byte_count": 64},)})
    blocking = TaskSpec("a-wire", "r", "other transfer", TaskCategory.MEMORY,
                        demands=(ResourceDemand("wire", 10000.),))
    kernel = UnifiedEventKernel.from_closed_graph((blocking, task), resource_owners=owners)
    first, actual = kernel.step(), kernel.step()
    assert first.task.task_id == blocking.task_id
    reference = PhysicalRuntimeContext()
    reference.timeline.resource_aliases = owners
    reference.timeline.lane_available["shared" if alias else "wire"] = [10000.]
    expected = resolve_physical_task(task, reference, 0.)
    assert actual.start_ns == 0.
    assert actual.end_ns == pytest.approx(expected.metadata["physical_completion_ns"])
    host_intervals = [stage for stage in actual.task.metadata["physical_resource_intervals"]
                      if stage.resource_id == ("shared" if alias else "wire")]
    assert host_intervals and all(stage.start_ns >= first.end_ns for stage in host_intervals)
    if operation == "read":
        # Array sensing proceeds during the unrelated transfer, rather than
        # adding the complete read service after the wire becomes available.
        assert actual.end_ns < 10100.


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("generation", [0, 3])
def test_pool_owned_pages_are_borrowed_not_reclaimed_by_cohort(external, generation):
    config = make_hbm_config(capacity_bytes=1048576)
    key = ("memory", "pool-page", generation)
    task = TaskSpec("read", "cohort-0", "page read", TaskCategory.MEMORY, metadata={
        "physical_memory_config": config, "physical_owner": "memory",
        "physical_external_allocations": (key,) if external else (),
        "persistent_request_buffers": {"request": ("pool-page",)},
        "memory_accesses": ({"operation": "read", "buffer_id": "pool-page",
            "allocation_generation": generation, "byte_count": 64,
            "allocation_size_bytes": 4096},)})
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    kernel.physical_runtime.runtime(config, "memory")
    allocator = kernel.physical_runtime.allocators["memory"]
    allocator.allocate("pool-page", 4096, generation)
    kernel.step()
    assert (allocator.get_allocation("pool-page", generation) is not None) == (external or generation == 0)
    kernel.release_request_physical_allocations("request")
    assert (allocator.get_allocation("pool-page", generation) is not None) == external
    assert not kernel._external_physical_allocation_keys
    assert not kernel._physical_allocation_uses
    if external:
        allocator.release("pool-page", generation)
        assert allocator.get_allocation("pool-page", generation) is None


@pytest.mark.parametrize("generation", [0, 3])
def test_json_pool_keys_do_not_duplicate_workspace_capacity(generation):
    from heterollm_sim.workspace_memory import physical_workspace_bound

    task = TaskSpec("read", "cohort", "page read", TaskCategory.MEMORY, metadata={
        "physical_memory_config": asdict(make_hbm_config(capacity_bytes=1048576)),
        "physical_owner": "memory",
        "physical_external_allocations": [["memory", "pool-page", generation]],
        "memory_accesses": ({"operation": "read", "buffer_id": "pool-page",
            "allocation_generation": generation, "byte_count": 64,
            "allocation_size_bytes": 4096},)})
    bound = physical_workspace_bound((task,))
    assert bound["bytes_by_owner"] == {}
    assert bound["resident_bytes_by_owner"] == {}
    assert bound["buffer_count"] == 0


@pytest.mark.parametrize("bandwidth", [.01, 1000.])
@pytest.mark.parametrize("explicit_dag", [False, True])
def test_serving_contention_does_not_reintroduce_hbm_bandwidth_or_serialize_dag(bandwidth, explicit_dag):
    from types import SimpleNamespace as NS
    from heterollm_sim.serving import _OnlineRuntime, BatchCost

    runner = object.__new__(_OnlineRuntime)
    runner.resource_policy = NS(enabled=True, include_custom_lowerers=True)
    runner._ensure_resource_profiles = lambda: None
    runner._resource_snapshot = lambda items: {}
    runner._gpu_profile = NS(attainable_tops=1.)
    runner._hbm_profile = NS(effective_bandwidth_gb_s=bandwidth)
    runner._resource_kv_host_backed_pages = runner._resource_state_host_backed_allocations = 0
    runner.residency_manager = None
    runner.plan = NS(kv_policy=NS(bytes_per_page=4096, offload_component=None),
        linear_state_policy=NS(bytes_per_request=0, offload_component=None))
    runner._page_transfer_cost = lambda *args: (0., "none")
    metadata = {"model": "topology_aware_parallel_lowering",
        "host_orchestration_ns": 10., "device_execution_ns": 10.,
        "resource_demand": {"gpu_operations": 0, "hbm_bytes": 1000000}}
    if explicit_dag:
        metadata.update(execution_stage_source="executed_task_dag_kernel_timeline",
                        execution_time_semantics="overlapping_task_interval_unions")
    cost = BatchCost(10. if explicit_dag else 20., metadata)
    actual = runner._apply_resource_contention(NS(kind="decode", items=()), cost)
    assert actual.duration_ns == cost.duration_ns
    assert actual.metadata["device_execution_ns"] == 10.
    assert actual.metadata["resource_contention_memory_cost_source"] == "lowerer_memory_execution"
    assert "resource_contention_hbm_demand_ns" not in actual.metadata
