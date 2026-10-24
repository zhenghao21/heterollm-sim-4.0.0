"""Native comparison inputs for EP attention data parallelism (one DP replica)."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
from exploring_hbf_native_config import (
    build_native_comparison_scenario, native_capacity_batch, native_import_pool, validate_native_layout,
)
from exploring_hbf_native_models import build_native_paper_model
from exploring_hbf_paper_config import hardware_by_id
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.ir import LayerSpec, model_graph_execution_layers
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.serving import compile_serving_plan
from test_native_maverick_semantics import miniature_maverick

SMALL = dict(prompt_override=7, output_override=4, warmup_override=3, measurement_override=6)


def moe_scenario(*, batch_size=4, tp=2, pp=1, ep=2, hardware="HBM4"):
    return build_native_comparison_scenario("miniature-functional", hardware, "SHORT", batch_size=batch_size,
        tp=tp, pp=pp, ep=ep, model_override=miniature_maverick(prompt=7).model, **SMALL)


def layers_of(model_id):
    return model_graph_execution_layers(build_native_paper_model(model_id).graph)


def test_layout_validation_follows_batch_and_model_graph():
    dense, maverick = layers_of("llama3_405b"), layers_of("llama4_maverick")
    validate_native_layout(dense, 8, 1)
    validate_native_layout(maverick, 8, 8)
    validate_native_layout(maverick, 16, 16)
    with pytest.raises(ValueError, match="ep > 1 requires a model with MoE layers"):
        validate_native_layout(dense, 8, 2)
    with pytest.raises(ValueError, match="batch_size 6 must be divisible by ep 4"):
        validate_native_layout(maverick, 6, 4)
    experts = [layer for layer in maverick if layer.is_moe][0].num_experts
    assert experts == 128
    odd = tuple(LayerSpec(layer.layer_id, layer.kind, layer.hidden_size, layer.intermediate_size, layer.attention_heads,
        kv_heads=layer.kv_heads, num_experts=6 if layer.is_moe else 1, dtype=layer.dtype) for layer in maverick[:4])
    with pytest.raises(ValueError, match="num_experts must be divisible by ep 4; got \\[6\\]"):
        validate_native_layout(odd, 8, 4)
    validate_native_layout(odd, 8, 2)


def test_builder_rejects_unsplittable_ep_before_building_any_hardware():
    with pytest.raises(ValueError, match="ep > 1 requires a model with MoE layers"):
        build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", batch_size=8, tp=4, ep=2)
    with pytest.raises(ValueError, match="batch_size 6 must be divisible by ep 4"):
        build_native_comparison_scenario("llama4_maverick", "HBM4", "SHORT", batch_size=6, tp=2, ep=4)
    with pytest.raises(ValueError, match="num_experts must be divisible by ep 3"):
        build_native_comparison_scenario("llama4_maverick", "HBM4", "SHORT", batch_size=6, tp=2, ep=3)
    with pytest.raises(ValueError, match="1..16 ranks"):
        build_native_comparison_scenario("llama4_maverick", "HBM4", "SHORT", batch_size=32, tp=8, ep=4)


@pytest.mark.parametrize("hardware", ["HBM4", "HBF+"])
def test_ep_replica_has_group_ordered_ranks_private_stores_and_group_divisible_pools(hardware):
    scenario = moe_scenario(batch_size=4, tp=2, ep=2, hardware=hardware)
    parallel = scenario.placement.parallel
    assert (parallel.tp_degree, parallel.pp_degree, parallel.ep_degree) == (2, 1, 2)
    plan = build_parallel_plan(scenario)
    assert [(rank.rank, rank.tp_rank, rank.pp_rank, rank.ep_rank) for rank in plan.ranks] == [
        (0, 0, 0, 0), (1, 1, 0, 0), (2, 0, 0, 1), (3, 1, 0, 1)]
    assert [[rank.rank for rank in plan.tp_group(0, group)] for group in range(2)] == [[0, 1], [2, 3]]
    assert [[rank.rank for rank in plan.ep_group(0, tp)] for tp in range(2)] == [[0, 2], [1, 3]]
    stores = scenario.placement.metadata["rank_weight_components"]
    assert scenario.placement.metadata["rank_kv_components"] == stores
    assert len(set(stores.values())) == 4
    assert scenario.workload.scheduler.max_num_seqs == 4
    ingress = scenario.workload.metadata["external_kv_ingress"]
    assert ingress["max_inflight_requests"] % 2 == 0 and ingress["max_inflight_requests"] >= 2
    assert any("Attention data parallelism over 2 EP groups" in text for text in scenario.assumptions)
    assert "TP2PP1EP2" in scenario.name


def test_ep1_scenario_has_no_ep_assumption_and_keeps_the_single_group_layout():
    scenario = moe_scenario(batch_size=3, tp=2, ep=1)
    assert not any("Attention data parallelism" in text for text in scenario.assumptions)
    assert any("(1 group(s), up to" in text for text in scenario.assumptions)
    assert [rank.ep_rank for rank in build_parallel_plan(scenario).ranks] == [0, 0]


def test_seed_ages_keep_the_centered_formula_and_seeds_precede_fresh_requests_for_ep():
    scenario = moe_scenario(batch_size=4, tp=2, ep=2)
    requests = scenario.workload.requests
    assert len(requests) == 16
    seeds = [request for request in requests if request.metadata["warmup_seed"]]
    assert [request.request_id for request in seeds] == [request.request_id for request in requests[:4]]
    output = 4
    assert [request.prompt_tokens - 7 for request in seeds] == [(2 * index + 1) * output // 8 for index in range(4)]


def test_import_pool_is_per_ep_group_and_uses_the_group_batch_for_the_slot_count():
    layers, hbf = layers_of("llama4_maverick"), hardware_by_id("HBF")
    # LONG: 103.5K prompt, 1100 outputs; one import spans 2/3/4 steps-worth of pool at tp 8/2/1.
    call = lambda batch, tp, pp, ep: native_import_pool(layers, 103500, 1100, batch, tp, pp, ep, hbf)
    assert (call(1, 8, 1, 1), call(1, 2, 1, 1), call(1, 1, 1, 1)) == (2, 3, 4)
    # B=2400 frees ceil(2400/1100)=3 slots per step on one group, but ceil(1200/1100)=2
    # on each of two groups and ceil(600/1100)=1 on each of four.
    assert call(2400, 8, 1, 1) == 3 * 2 and call(2400, 8, 1, 2) == 2 * 2 * 2 and call(2400, 8, 1, 4) == 4 * 1 * 2
    # A request's KV sits on its group's tp*pp ranks only: ep never divides the import time.
    assert call(2, 2, 1, 2) == 2 * 3 and call(2, 1, 2, 2) == 2 * 3 and call(2, 1, 1, 2) == 2 * 4
    for ep in (1, 2, 4, 8):
        assert call(8 * ep, 2, 1, ep) % ep == 0
    with pytest.raises(ValueError, match="divisible by ep"):
        call(6, 8, 1, 4)


def test_capacity_batch_fits_seeds_and_the_pool_in_the_requests_a_group_held():
    layers, hbf = layers_of("llama4_maverick"), hardware_by_id("HBF")
    pool = lambda batch: native_import_pool(layers, 103500, 1100, batch, 8, 1, 2, hbf)
    batch = native_capacity_batch(1000, 2400, pool(2400), 1100, 2)
    # Each group holds its B/ep seeds plus its pool share, and one more pair would not fit.
    assert batch == 1996 and batch // 2 + pool(batch) // 2 <= 1000 < (batch + 2) // 2 + pool(batch + 2) // 2
    assert native_capacity_batch(2, 2400, pool(2400), 1100, 2) == 0


def test_built_pool_matches_the_helper_for_the_scenario_it_describes():
    scenario = moe_scenario(batch_size=4, tp=2, ep=2, hardware="HBF")
    layers, hbf = model_graph_execution_layers(scenario.model.graph), hardware_by_id("HBF")
    expected = native_import_pool(layers, 7, 4, 4, 2, 1, 2, hbf)
    assert scenario.workload.metadata["external_kv_ingress"]["max_inflight_requests"] == expected
    assert expected % 2 == 0 and expected == 2 * native_import_pool(layers, 7, 4, 2, 2, 1, 1, hbf)


def test_dense_weights_are_replicated_on_every_rank_and_ep2_places_under_the_control_plane():
    scenario = moe_scenario(batch_size=4, tp=2, ep=2)
    decision = plan_runtime_placement(scenario)
    assert decision.fully_placed, [(item.item_id, item.reason) for item in decision.unplaced]
    shards = decision.apply(scenario).placement.metadata["control_plane"]["decision"]["rank_weight_shards"]
    for name in ("dense0.attention_weights", "dense0.mlp_weights", "embedding_weights"):
        assert sorted(row["rank_id"] for row in shards[name]) == [0, 1, 2, 3]
    for rows in shards.values():
        for row in rows:
            assert row["storage_component_id"] == f"hbm{row['rank_id']}"


def test_ep2_scenario_compiles_a_serving_plan_with_group_divisible_slots():
    plan = compile_serving_plan(moe_scenario(batch_size=4, tp=2, ep=2))
    assert plan.scheduler.max_num_seqs == 4
    assert sum(request.metadata["warmup_seed"] for request in plan.requests) == 4
