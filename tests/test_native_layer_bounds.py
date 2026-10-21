"""Only native invariant work and actual serving endpoint rules tighten bounds."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.memory_types import SramConfig
from heterollm_sim.native_layer_bounds import native_layer_boundary_bounds


def fixture():
    config = SramConfig(4096, 64, 64., 64., 64., 0., 0., 1, 16)
    launch = TaskSpec("launch", "r", "launch", TaskCategory.COMPUTE,
        demands=(ResourceDemand("gpu.frontend", 5.),),
        metadata={"phase": "kernel_launch", "cost_model": True})
    read = TaskSpec("read", "r", "read", TaskCategory.MEMORY, dependencies=("launch",),
        metadata={"weight_tensor_id": "w", "weight_lifecycle_mode": "preloaded_resident",
            "physical_owner": "memory", "physical_memory_config": config,
            "memory_accesses": ({"buffer_id": "tensor:w", "operation": "read",
                "byte_count": 128, "allocation_generation": 0},)})
    tail = TaskSpec("tail", "r", "tail", TaskCategory.POLICY, dependencies=("read",),
        demands=(ResourceDemand("cpu", 999.),))
    stages = (NS(stage_id="body", dependencies=(), execution_tasks=(launch, read),
                 causal_invocation_group_ids=("g",)),
              NS(stage_id="tail", dependencies=("body",), execution_tasks=(tail,),
                 causal_invocation_group_ids=("g",)))
    scenario = NS(model=NS(_execution_view=NS(layer_instances=(NS(layer=NS(is_moe=False)),))),
        workload=NS(scheduler=NS(max_num_seqs=1)))
    cost = NS(metadata={"item_terminal_tasks": ({"item_index": 0, "terminal_task_id": "logical-marker",
        "operator_invocation_group_ids": ("g",)},)})
    return scenario, cost, stages


def test_invariant_launch_and_resident_weight_path_excludes_variable_arithmetic():
    scenario, cost, stages = fixture()
    result = native_layer_boundary_bounds(scenario, cost, stages)
    assert result["batch_invariant_native_path_floor_ns"] == 7.
    assert result["batch_invariant_launch_task_count"] == result["batch_invariant_weight_read_count"] == 1
    assert result["visible_commit_lag_upper_ns"] == 0.
    assert result["visible_commit_boundary_proof"]["qualified"]


@pytest.mark.parametrize("change,expected", [("variable_launch", 2.), ("unknown_launch", 2.),
    ("unloaded_weight", 5.), ("variable_weight_generation", 5.), ("moe", 0.)])
def test_unknown_or_batch_dependent_cost_never_becomes_invariant_work(change, expected):
    scenario, cost, stages = fixture()
    launch, read = stages[0].execution_tasks
    if change == "variable_launch":
        launch = replace(launch, demands=(ResourceDemand("gpu.frontend", 5., work_units=1.),))
    elif change == "unknown_launch":
        launch = replace(launch, metadata={"phase": "kernel_launch"})
    elif change == "unloaded_weight":
        read = replace(read, metadata={**read.metadata, "weight_lifecycle_mode": "on_demand"})
    elif change == "variable_weight_generation":
        read = replace(read, metadata={**read.metadata, "memory_accesses": (
            {**read.metadata["memory_accesses"][0], "allocation_generation": 1},)})
    else:
        scenario.model._execution_view.layer_instances[0].layer.is_moe = True
    stages[0].execution_tasks = (launch, read)
    assert native_layer_boundary_bounds(scenario, cost, stages)["batch_invariant_native_path_floor_ns"] == expected


@pytest.mark.parametrize("change", ("early_terminal", "uncovered_tail", "missing_item", "unknown_group"))
def test_token_lag_zero_requires_the_exact_native_terminal_fallback_covering_every_sink(change):
    scenario, cost, stages = fixture()
    if change == "early_terminal":
        cost.metadata["item_terminal_tasks"][0]["terminal_task_id"] = "read"
    elif change == "uncovered_tail":
        stages[-1].causal_invocation_group_ids = ()
    elif change == "missing_item":
        scenario.workload.scheduler.max_num_seqs = 2
    else:
        cost.metadata["item_terminal_tasks"][0]["operator_invocation_group_ids"] = ("unknown",)
    assert native_layer_boundary_bounds(scenario, cost, stages)["visible_commit_lag_upper_ns"] is None


def test_real_tp2_b8_terminal_fallback_and_dense_launch_floor_use_existing_native_graph():
    from test_native_window_lifecycle import capture_first_native_query, lifecycle_scenario

    runtime, prepared = capture_first_native_query(lifecycle_scenario(8))
    result = native_layer_boundary_bounds(runtime.plan.scenario, prepared["cost"], prepared["stages"])
    assert result["batch_invariant_native_path_floor_ns"] > 0.
    assert result["batch_invariant_launch_task_count"] == 56
    assert result["visible_commit_lag_upper_ns"] == 0.
    assert runtime.batches == []
    assert not any(event.event_type == "tokens_committed" for event in runtime.events)
