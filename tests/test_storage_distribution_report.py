"""Per-medium reporting must retain placement and physical traffic identities."""
from tools.qwen38_memory_scenario import load_model, build_scenario, execute_scenario


def test_real_qwen_reports_kv_weight_and_linear_state_owners_separately():
    model, _ = load_model()
    observed = execute_scenario(build_scenario(model, "hbf_active_kv"))
    stores = observed["storage_distribution"]
    assert stores["hbf0"]["kv_default_owner"]
    assert not stores["hbf0"]["linear_state_default_owner"]
    assert stores["hbm2"]["linear_state_default_owner"]
    assert stores["hbf0"]["resource_bytes"]["component.hbf0.write"] > 0
    for component, row in stores.items():
        assert row["persistent_weight_shard_bytes"] <= row["capacity_bytes"]
        for tensor in row["tensor_ids"]:
            assert observed["placement"]["tensor_to_component"][tensor] == component
    expected = sum(s["physical_bytes"] for shards in observed["placement"]["rank_weight_shards"].values() for s in shards)
    assert sum(row["persistent_weight_shard_bytes"] for row in stores.values()) == expected
