"""HBM cache and physical lowering must retain identical allocation contracts."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.data_motion import lower_physical_allocations
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.kernel_model import KernelModelProfile
from heterollm_sim.planner import (
    _attach_gddr_physical_task, _attach_planned_l2, _compilation_scope,
    _parallel_plan, _rank_memory_resource, compile_scenario,
)
from heterollm_sim.control_plane_planner import plan_runtime_placement
from test_backend_lowering_followup import configured_scenario


def scenario_with_l2():
    scenario = configured_scenario(tp=1)
    gpu = scenario.resolve_component_profile("gpu0")
    profile = replace(gpu, kernel_model=KernelModelProfile(
        "device", "runtime", "architecture", stateful_l2=True))
    return replace(scenario, component_profiles={**scenario.component_profiles,
        "gpu": {**scenario.component_profiles["gpu"], "legacy-gpu": profile}})


def lower_task(scenario, *, name="test", request="cohort-3", metadata=None,
               cost=None, dependencies=()):
    with _compilation_scope(scenario):
        rank = _parallel_plan(scenario).ranks[0]
        memory = _rank_memory_resource(scenario, rank)
        cost = cost or {"activation_bytes": 64, "weight_bytes": 64, "output_bytes": 64}
        total = sum(cost.values())
        task = TaskSpec(name, request, name, TaskCategory.COMPUTE,
            dependencies=dependencies, demands=(ResourceDemand(memory, 1., total),),
            metadata={"target_component": "gpu0", "rank": 0, "tp_rank": 0, "pp_rank": 0,
                "phase": "gpu_gemm", "operator_id": "audit", "cost_model": cost,
                **(metadata or {})})
        physical = _attach_gddr_physical_task(task, scenario)
        return lower_physical_allocations(_attach_planned_l2(physical))


@pytest.mark.parametrize("named", [False, True])
def test_l2_preserves_backing_buffer_identity_generation_and_extent(named):
    metadata = {"input_generation": 7, "weight_generation": 5, "output_generation": 9,
                "input_buffer_size_bytes": 128, "weight_buffer_size_bytes": 256,
                "output_buffer_size_bytes": 192}
    if named:
        metadata.update(input_buffer_id="input", weight_buffer_id="weights", output_buffer_id="output")
    task = lower_task(scenario_with_l2(), metadata=metadata)
    backing = {(row["buffer_id"], row["operation"], row["allocation_generation"],
                row["allocation_size_bytes"]) for row in task.metadata["memory_accesses"]}
    cached = {(row["buffer_id"], row["operation"], row["allocation_generation"],
               row["buffer_size_bytes"]) for row in task.metadata["stateful_l2"]["accesses"]}
    assert cached == backing
    event = UnifiedEventKernel.from_closed_graph((task,)).step()
    actual = {(row["buffer_id"], row["allocation_generation"])
              for row in event.task.metadata["l2_execution"]["accesses"]}
    assert actual == {(row[0], row[2]) for row in backing}


def test_weight_version_change_misses_l2_instead_of_reusing_old_generation():
    scenario = scenario_with_l2()
    old = lower_task(scenario, name="old", request="static-request",
        metadata={"weight_tensor_id": "weight", "weight_generation": 4},
        cost={"activation_bytes": 0, "weight_bytes": 64, "output_bytes": 0})
    new = lower_task(scenario, name="new", request="static-request",
        metadata={"weight_tensor_id": "weight", "weight_generation": 5},
        cost={"activation_bytes": 0, "weight_bytes": 64, "output_bytes": 0}, dependencies=("old",))
    kernel = UnifiedEventKernel.from_closed_graph((old, new))
    first, second = kernel.step(), kernel.step()
    assert first.task.metadata["l2_execution"]["miss_lines"] == 1
    assert second.task.metadata["l2_execution"]["miss_lines"] == 1
    assert second.task.metadata["l2_execution"]["hbm_read_bytes"] == (
        scenario.resolve_component_profile("gpu0").cache_hierarchy.levels[-1].line_bytes)
    assert not kernel._pending_physical_releases
    allocations = [allocation for allocator in kernel.physical_runtime.allocators.values()
                   for allocation in allocator.allocations()]
    assert {allocation.generation for allocation in allocations} == {4, 5}
    assert {allocation.allocation_arena for allocation in allocations} == {"resident"}


def test_explicit_cache_views_use_directional_generation_contract():
    task = lower_task(scenario_with_l2(), cost={"read_bytes": 64, "write_bytes": 64},
        metadata={"input_generation": 7, "output_generation": 9,
            "buffer_accesses": (
                {"buffer_id": "input", "operation": "read", "offset_bytes": 0, "size_bytes": 64},
                {"buffer_id": "output", "operation": "write", "offset_bytes": 0, "size_bytes": 64})})
    assert [(row["operation"], row["allocation_generation"])
            for row in task.metadata["stateful_l2"]["accesses"]] == [("read", 7), ("write", 9)]


def test_explicit_physical_views_remain_the_l2_backing_identity():
    task = lower_task(scenario_with_l2(), cost={"read_bytes": 64, "write_bytes": 64},
        metadata={"persistent_buffer_ids": ("actual-input",), "memory_accesses": (
            {"buffer_id": "actual-input", "operation": "read", "offset_bytes": 0,
             "byte_count": 64, "allocation_generation": 11, "allocation_size_bytes": 64},
            {"buffer_id": "actual-output", "operation": "write", "offset_bytes": 0,
             "byte_count": 64, "allocation_generation": 13, "allocation_size_bytes": 64})})
    assert [(row["buffer_id"], row["allocation_generation"])
            for row in task.metadata["stateful_l2"]["accesses"]] == [("actual-input", 11), ("actual-output", 13)]
    assert task.metadata["memory_accesses"][0]["allocation_arena"] == "resident"
    event = UnifiedEventKernel.from_closed_graph((task,)).step()
    assert {row["buffer_id"] for row in event.task.metadata["l2_execution"]["accesses"]} == {
        "actual-input", "actual-output"}


@pytest.mark.parametrize("owners", [("custom-hbm",), ("board0.hbm", "board1.hbm")])
def test_custom_rank_memory_owners_still_lower_to_dram_core(owners):
    scenario = configured_scenario(tp=len(owners))
    names = {"hbm" + str(rank): owner for rank, owner in enumerate(owners)}
    hardware = replace(scenario.hardware, components=tuple(
        replace(component, metadata={**component.metadata, "memory_service_owner": names[component.component_id]})
        if component.component_id in names else component
        for component in scenario.hardware.components))
    scenario = replace(scenario, hardware=hardware)
    scenario = plan_runtime_placement(scenario).apply(scenario)
    tasks = compile_scenario(scenario).tasks
    qkv = [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
           and task.metadata.get("projection_id") == "attention.qkv"]
    assert len(qkv) == len(owners)
    for task in qkv:
        owner = owners[task.metadata["rank"]]
        assert task.metadata.get("physical_owner") == owner
        assert task.metadata["physical_memory_config"] == scenario.hardware.get_component(
            "hbm" + str(task.metadata["rank"])).metadata["physical_memory_config"]
        assert {row["physical_owner"] for row in task.metadata["memory_accesses"]} == {owner}
        event = UnifiedEventKernel.from_closed_graph((replace(task, dependencies=()),)).step()
        assert next(iter(event.task.metadata["physical_execution_by_owner"].values()))["physical_bytes"] > 0
        assert owner not in {d.resource_id for d in event.demands}
        assert event.task.metadata["physical_completion_ns"] > event.start_ns


def test_unlowered_legacy_rank_prefix_is_rejected():
    scenario = configured_scenario(tp=1)
    hardware = replace(scenario.hardware, components=tuple(
        replace(component, metadata={**component.metadata, "memory_service_owner": "board.hbm"})
        if component.component_id == "hbm0" else component
        for component in scenario.hardware.components))
    scenario = replace(scenario, hardware=hardware)
    task = TaskSpec("bad", "r", "bad", TaskCategory.MEMORY,
        demands=(ResourceDemand("hbm0.hbm", 1., 64),),
        metadata={"target_component": "gpu0", "operator_id": "audit",
                  "cost_model": {"read_bytes": 64, "write_bytes": 0}})
    with _compilation_scope(scenario), pytest.raises(ValueError, match="aggregate-cost fallback is disabled"):
        _attach_gddr_physical_task(task, scenario)
