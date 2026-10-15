"""Configuration and compiler consumers accept only current component fields."""
from copy import deepcopy

import pytest

from heterollm_sim.compiler_ir import (
    CanonicalizationError, _tensor_plan_from_rank_shards, compile_canonical_scenario,
)
from heterollm_sim.component_presets import materialize_component_payload
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.ir import ComponentSpec, normalize_component_kind
from heterollm_sim.parallel import LogicalRank
from heterollm_sim.reference import build_reference_scenario


@pytest.mark.parametrize("kind", ["high_i_o_ssd", "highio_ssd", "high_i/o_ssd"])
def test_high_io_ssd_does_not_reinterpret_retired_kind_spellings(kind):
    assert normalize_component_kind(kind) == kind
    with pytest.raises(ValueError, match="kind must be high_io_ssd"):
        ComponentSpec("ssd", kind)
    with pytest.raises(ValueError, match="kind must be high_io_ssd"):
        hardware_from_dict({"name": "old-ssd", "components": [
            {"component_id": "ssd", "kind": kind}], "links": []})


def test_current_ssd_component_and_high_io_variant_keep_the_physical_model():
    payload = materialize_component_payload("ymtc-zhitai-ti-pro9100")
    for kind in ("ssd", "high_io_ssd"):
        component = deepcopy(payload)
        component["kind"] = kind
        restored = hardware_from_dict({"name": "ssd", "components": [component], "links": []})
        storage, = restored.components
        assert storage.normalized_kind == kind
        assert storage.is_storage and storage.memory_class == "offload"
        assert storage.metadata["physical_memory_config"] == payload["metadata"]["physical_memory_config"]
    assert normalize_component_kind(" High-IO SSD ") == "high_io_ssd"


def _compile_shard_row(row):
    return _tensor_plan_from_rank_shards(
        build_reference_scenario(), "test-weight", (row,), 256,
        (LogicalRank(0, "gpu0", 0, 0, 0, memory_component_id="hbm0"),),
        (0,), False,
    )


def test_compiler_reads_current_rank_shard_storage_component():
    plan = _compile_shard_row({"rank_id": 0, "storage_component_id": "hbm0",
                              "logical_bytes": 256, "physical_bytes": 320})
    replica, = plan.replicas
    assert (replica.rank_id, replica.component_id, replica.physical_bytes) == (0, "hbm0", 320)
    assert plan.logical_bytes == 256


@pytest.mark.parametrize("old,current,value", [
    ("rank", "rank_id", 0), ("component_id", "storage_component_id", "hbm0"),
])
@pytest.mark.parametrize("keep_current", [False, True])
def test_compiler_refuses_retired_rank_shard_fields(old, current, value, keep_current):
    row = {"rank_id": 0, "storage_component_id": "hbm0",
           "logical_bytes": 256, "physical_bytes": 256, old: value}
    if not keep_current:
        row.pop(current)
    with pytest.raises(CanonicalizationError, match="retired fields: " + old):
        _compile_shard_row(row)


def test_current_placement_producer_compiles_without_duplicate_shard_fields():
    scenario = build_reference_scenario()
    decision = plan_runtime_placement(scenario)
    placed = decision.apply(scenario)
    rows = placed.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]
    assert rows
    assert all("rank_id" in row and "storage_component_id" in row
               and "rank" not in row and "component_id" not in row
               for shards in rows.values() for row in shards)
    canonical = compile_canonical_scenario(placed)
    for plan in canonical.placement_plan.tensor_plans:
        if plan.tensor_id in rows:
            assert {replica.component_id for replica in plan.replicas} == {
                row["storage_component_id"] for row in rows[plan.tensor_id]}
