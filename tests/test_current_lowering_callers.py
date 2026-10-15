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
