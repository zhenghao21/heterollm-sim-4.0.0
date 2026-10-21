"""Fixed chunks and Maverick routing are real native graph/physical operands."""
from dataclasses import replace
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_native_pilot import build_pilot_scenario
from heterollm_sim.cost_models import FusedAttentionWorkload
from heterollm_sim.ir import LayerSpec, ModelSpec, RequestSpec, build_model_graph_from_layer_specs
from heterollm_sim.kernel_model import attention_tile_work
from heterollm_sim.planner import compile_serving_cohort_schedule
from heterollm_sim.serving import BatchCohort, BatchItem


def miniature_maverick(*, prompt=3, temperature=False):
    base = build_pilot_scenario("hbm_tp1")
    layers = []
    for index in range(4):
        moe, full = (index + 1) % 2 == 0, (index + 1) % 4 == 0
        ffn = 3 * 64 * (64 if moe else 128) * (5 if moe else 1) + (64 * 4 if moe else 0)
        layers.append(LayerSpec(f"dense{index}", "moe" if moe else "dense", 64,
            64 if moe else 128, 4, kv_heads=2, attention_head_dim=16,
            sequence_mixer="full_attention" if full else "chunked_attention",
            attention_chunk_size=0 if full else 4, use_rope=not full,
            attention_temperature_tuning=temperature and full,
            num_experts=4 if moe else 1, experts_per_token=1,
            shared_expert_intermediate_size=64 if moe else 0, dtype="bf16",
            weight_bytes=2 * (2 * 64 * 64 + 2 * 64 * 32 + ffn),
            metadata={"moe_routing": {"gating": "sigmoid", "normalize_selected_weights": False,
                "weight_scale": 1.0, "weight_application": "input", "selection_before_gating": True}}
                if moe else {}))
    model = ModelSpec(name="native-maverick-miniature", graph=build_model_graph_from_layer_specs(
        "native-maverick-miniature", layers, architecture="llama4", vocabulary_size=128,
        max_sequence_length=16384, embedding_weight_bytes=64 * 128 * 2,
        output_weight_bytes=64 * 128 * 2, output_head_dtype="bf16"))
    return replace(base, model=model, fusion_policy=replace(base.fusion_policy, qkv_rope=False),
        placement=replace(base.placement, model_name=model.name,
            parallel=replace(base.placement.parallel, layer_to_stage={layer.layer_id: 0 for layer in layers})),
        workload=replace(base.workload, requests=(RequestSpec("native-mav", 0.0, prompt, 5),), request_count=1,
            scheduler=replace(base.workload.scheduler, max_num_seqs=1, prefill_chunk_tokens=4)))


def decode_graph(context, *, temperature=False):
    scenario = miniature_maverick(prompt=max(3, context), temperature=temperature)
    cohort = BatchCohort("maverick-decode", "decode", 0.0,
        (BatchItem("native-mav", "decode", 1, context, main_tokens=1, draft_tokens=0,
                   verifier_tokens=0, committed_tokens=1, logit_tokens=1),))
    return scenario, compile_serving_cohort_schedule(scenario, cohort)


def test_fixed_chunk_is_not_sliding_window_at_boundary():
    layer = LayerSpec("chunk", "dense", 64, 128, 4, sequence_mixer="chunked_attention", attention_chunk_size=8192)
    assert layer.attention_live_range(8191) == (0, 8191)
    assert layer.attention_live_range(8192) == (0, 8192)
    assert layer.attention_live_range(8193) == (8192, 8193)
    assert layer.attention_kv_start(16384) == 16384
    with pytest.raises(ValueError, match="positive attention_chunk_size"):
        replace(layer, attention_chunk_size=0)
    with pytest.raises(ValueError, match="NoPE"):
        replace(layer, attention_temperature_tuning=True)


def test_attention_cost_uses_independent_true_query_masks():
    work = FusedAttentionWorkload(batch_tokens=3, context_tokens=4, hidden_size=64,
        score_heads=4, query_context_lengths=(4, 1, 3), query_group_sizes=(1, 1, 1))
    assert work.score_elements == 4 * 8
    assert work.tensor_operations == 4 * 64 * 8
    tile = attention_tile_work(work)
    assert tile["useful_pairs"] == 8
    assert tile["independent_query_groups"] == 3
    assert tile["executed_pairs"] == 3 * 64 * 64


def test_native_graph_has_real_nope_and_sigmoid_before_expert_input_weight():
    _, graph = decode_graph(4)
    tasks = graph.tasks
    assert not any(task.metadata.get("layer_id") == "dense3" and task.metadata.get("event_kind") in
                   {"rope", "rope_table_generate"} for task in tasks)
    assert any(task.metadata.get("layer_id") == "dense0" and task.metadata.get("event_kind") == "rope" for task in tasks)
    assert any(task.metadata.get("event_kind") == "router_sigmoid" for task in tasks)
    assert any(task.metadata.get("event_kind") == "expert_input_weight" for task in tasks)
    assert not any(task.metadata.get("event_kind") in {"router_softmax_reduce", "router_softmax_normalize", "expert_route_weight"}
                   for task in tasks)
    visibility = [task.metadata for task in tasks if task.metadata.get("layer_id") == "dense0"
                  and task.metadata.get("generic_kv_storage")]
    assert visibility
    contracts = [row["generic_kv_storage"] for row in visibility]
    owner = contracts[0]["owners"][0]
    assert owner["storage_policy"] == "fixed_chunk_ring"
    assert owner["capacity_tokens"] == 4
    assert owner["absolute_write_positions"] == (4,)
    assert owner["write_positions"] == (0,)
    assert owner["logical_read_ranges"] == ((4, 1),)
    assert owner["read_ranges"] == [(0, 1)]


def test_temperature_scales_q_on_nope_and_pays_real_primitive():
    _, graph = decode_graph(8192, temperature=True)
    scaled = [task for task in graph.tasks if task.metadata.get("event_kind") == "attention_temperature"]
    assert scaled
    assert all(task.metadata["layer_id"] == "dense3" for task in scaled)
    assert scaled[0].metadata["query_scales"] == (1.0 + 0.1 * math.log(2),)
    assert any(demand.service_ns > 0 for task in scaled for demand in task.demands)


def test_unsplit_physical_prefill_cannot_overwrite_previous_chunk():
    scenario = miniature_maverick(prompt=5)
    cohort = BatchCohort("bad-crossing", "prefill", 0.0,
        (BatchItem("native-mav", "prefill", 2, 3, main_tokens=2, draft_tokens=0,
                   verifier_tokens=0, committed_tokens=0, logit_tokens=1),))
    with pytest.raises(ValueError, match="split.*chunk boundary"):
        compile_serving_cohort_schedule(scenario, cohort)


def test_native_maverick_miniature_physically_runs_across_chunk_boundary():
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.serving import RequestStatus
    result = run_scenario(miniature_maverick())
    metric, = result.serving.request_metrics.values()
    assert metric.status is RequestStatus.FINISHED
    assert metric.visible_output_tokens == 5
    assert metric.tpot_ns > 0
