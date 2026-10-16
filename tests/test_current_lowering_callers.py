"""Runtime weight placement has one physical allocation ledger."""
from dataclasses import replace

import pytest

from heterollm_sim.compiler_ir import CanonicalizationError, compile_canonical_scenario
from heterollm_sim.control_plane_planner import (
    PlacementPolicy, _base_capacity_usage, plan_runtime_placement,
)
from heterollm_sim.planner import (
    _unallocated_component_capacity, compile_scenario, validate_scenario,
)
from heterollm_sim.reference import build_reference_scenario
from test_backend_lowering_followup import configured_scenario


def shard_rows(scenario):
    return scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]


def test_actual_tp_placement_compiles_and_accounts_each_shard_without_scalar_copy():
    scenario = configured_scenario(tp=2)
    rows = shard_rows(scenario)
    assert rows
    assert not set(rows).intersection(scenario.placement.tensor_bytes)
    decision = scenario.placement.metadata["control_plane"]["decision"]
    assert not set(rows).intersection(decision["physical_tensor_bytes"])
    assert all("placement_bytes" not in decision["weight_tensor_details"][tensor]
               for tensor in rows)
    canonical = compile_canonical_scenario(scenario)
    for plan in canonical.placement_plan.tensor_plans:
        if plan.tensor_id in rows:
            assert sum(replica.physical_bytes for replica in plan.replicas) == sum(
                row["physical_bytes"] for row in rows[plan.tensor_id])
    expected = {}
    for shards in rows.values():
        for shard in shards:
            owner = shard["storage_component_id"]
            expected[owner] = expected.get(owner, 0) + shard["physical_bytes"]
    assert _base_capacity_usage(scenario, ()) == expected
    for owner, occupied in expected.items():
        assert _unallocated_component_capacity(scenario, owner) == max(
            0, scenario.hardware.component_map()[owner].capacity_bytes - occupied)
    assert compile_scenario(scenario).tasks


def test_representative_tensor_bytes_are_rejected_by_validation_and_canonical_compiler():
    scenario = configured_scenario(tp=2)
    tensor, shards = next(iter(shard_rows(scenario).items()))
    placement = replace(scenario.placement, tensor_bytes={
        **scenario.placement.tensor_bytes, tensor: shards[0]["physical_bytes"],
    })
    obsolete = replace(scenario, placement=placement)
    report = validate_scenario(obsolete)
    assert any("representative tensor_bytes" in error for error in report.errors)
    with pytest.raises(CanonicalizationError, match="representative tensor_bytes"):
        compile_canonical_scenario(obsolete)


def test_cold_cim_shared_backing_is_compiled_from_the_current_shard_ledger():
    scenario = replace(build_reference_scenario(), weights_resident=False)
    placement = plan_runtime_placement(scenario, PlacementPolicy(
        allow_cold_cim_streaming=True, operator_targets={"dense0.mlp": "cim0"}))
    assert placement.fully_placed
    scenario = placement.apply(scenario)
    rows = shard_rows(scenario)
    backing_ids = {tensor for tensor in rows if tensor.endswith("#cold_backing")}
    assert backing_ids
    assert not set(rows).intersection(scenario.placement.tensor_bytes)
    canonical = compile_canonical_scenario(scenario)
    plans = {plan.tensor_id: plan for plan in canonical.placement_plan.tensor_plans}
    for tensor in backing_ids:
        row, = rows[tensor]
        assert (row["shard_kind"], row["shard_index"], row["shard_count"]) == ("packed_backing", 0, 1)
        plan = plans[tensor]
        assert plan.logical_bytes == row["logical_bytes"]
        replica, = plan.replicas
        assert replica.physical_bytes == row["physical_bytes"]
        assert replica.component_id == row["storage_component_id"]
    assert not validate_scenario(scenario).errors


@pytest.mark.parametrize("native_gather", [False, True])
def test_remote_embedding_keeps_one_table_allocation_across_cohorts(tmp_path, native_gather):
    from hbf_hbm_ablation_paths import build_path_scenario
    from heterollm_sim.config import model_from_dict
    from heterollm_sim.data_motion import PhysicalRuntimeContext, register_physical_allocations
    from heterollm_sim.gguf_model_catalog import GGUFModelCatalog
    from heterollm_sim.llama_tensor_storage import apply_llama_tensor_storage_contract
    from heterollm_sim.planner import compile_serving_cohort_schedule
    from heterollm_sim.serving import BatchCohort, BatchItem

    scenario, _ = build_path_scenario("direct", prompt=32, output=12)
    prefill_rows, row_bytes = 32, 256
    if native_gather:
        payload = GGUFModelCatalog(tmp_path).derive(
            "qwen3-0_6b", "embedding-allocation", {"layer_count": 1}, save=False)["model"]
        model = model_from_dict(payload)
        placement = replace(scenario.placement, model_name=model.name,
            op_to_component={}, tensor_to_component={}, tensor_bytes={},
            parallel=replace(scenario.placement.parallel, layer_to_stage={}),
            metadata={"llama_backend_memory": scenario.placement.metadata["llama_backend_memory"]})
        scenario = replace(scenario, model=model, placement=placement,
                           workload=replace(scenario.workload, mtp=None))
        scenario = apply_llama_tensor_storage_contract(
            scenario, model.metadata["llama_cpp_tensor_storage_contract"])
        policy = PlacementPolicy(weight_tensor_targets={"embedding_weights": "hbf_weights"},
                                 operator_targets={"embedding": "gpu0"})
        scenario = plan_runtime_placement(scenario, policy).apply(scenario)
        prefill_rows, row_bytes = 2, 2048
    else:
        scenario = plan_runtime_placement(scenario).apply(scenario)
    assert "embedding_weights" not in scenario.placement.tensor_bytes
    shard, = shard_rows(scenario)["embedding_weights"]
    capacity = shard["physical_bytes"]
    runtime = PhysicalRuntimeContext(capture_details=False)
    first_allocation = None
    for phase, rows, context in (("prefill", prefill_rows, 0), ("decode", 1, prefill_rows)):
        item = BatchItem("embedding-request", phase, rows, context, main_tokens=rows,
            draft_tokens=0, verifier_tokens=rows, committed_tokens=rows, logit_tokens=1)
        schedule = compile_serving_cohort_schedule(
            scenario, BatchCohort(phase, phase, 0., (item,)))
        task, = (task for task in schedule.tasks if task.metadata.get("event_kind") == "embedding"
                 and task.metadata.get("phase") == "gpu_memory")
        accesses = [row for row in task.metadata["memory_accesses"]
                    if "embedding_weights" in row["buffer_id"]]
        declaration, = (row for row in task.metadata["physical_allocations"]
                        if "embedding_weights" in row["buffer_id"])
        assert declaration["size_bytes"] == capacity
        assert sum(row["byte_count"] for row in accesses) == rows * row_bytes
        assert all(row["allocation_size_bytes"] == capacity for row in accesses)
        register_physical_allocations(task, runtime)
        allocation = runtime.allocators[declaration["physical_owner"]].get_allocation(
            declaration["buffer_id"], declaration["generation"])
        assert allocation.size_bytes == capacity
        if first_allocation is None:
            first_allocation = allocation
        else:
            assert allocation == first_allocation
        if native_gather:
            assert task.metadata["native_get_rows_storage"]["qualified"]
            assert task.metadata["lookup_read_bytes"] == rows * 2048
            assert task.metadata["lookup_index_read_bytes"] == rows * 4
            assert task.metadata["lookup_write_bytes"] == rows * 1024 * 4


@pytest.mark.parametrize("table_bytes", [524288, 524289])
def test_embedding_allocation_capacity_is_the_rank_shard_not_the_whole_table(table_bytes):
    scenario = configured_scenario(tp=2)
    graph = replace(scenario.model.graph, tensors=tuple(
        replace(tensor, logical_bytes=table_bytes)
        if tensor.tensor_id == "embedding_weights" else tensor
        for tensor in scenario.model.graph.tensors))
    scenario = replace(scenario, model=replace(scenario.model, graph=graph),
        placement=replace(scenario.placement, metadata={**scenario.placement.metadata,
            "llama_backend_memory": {"hbf0": {"device_ids": ["gpu0", "gpu1"],
                "buffer_kind": "HBF", "access": "direct"}}}))
    scenario = plan_runtime_placement(scenario, PlacementPolicy(
        weight_tensor_targets={"embedding_weights": "hbf0"})).apply(scenario)
    capacities = {row["rank_id"]: row["physical_bytes"]
                  for row in shard_rows(scenario)["embedding_weights"]}
    assert capacities == {0: (table_bytes + 1) // 2, 1: table_bytes // 2}
    tasks = [task for task in compile_scenario(scenario).tasks
             if task.metadata.get("event_kind") == "embedding"
             and task.metadata.get("phase") == "gpu_memory"]
    assert {task.metadata["rank"] for task in tasks} == set(capacities)
    for task in tasks:
        capacity = capacities[task.metadata["rank"]]
        assert task.metadata["weight_buffer_size_bytes"] == capacity
        declaration, = (row for row in task.metadata["physical_allocations"]
                        if "embedding_weights" in row["buffer_id"])
        assert declaration["size_bytes"] == capacity
        accesses = [row for row in task.metadata["memory_accesses"]
                    if "embedding_weights" in row["buffer_id"]]
        assert accesses
        assert all(row["allocation_size_bytes"] == capacity for row in accesses)
        assert sum(row["byte_count"] for row in accesses) == 256
