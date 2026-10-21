"""Architectural invariants for the operation-event paper comparison."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heterollm_sim.paper_models import (
    LLAMA3_405B, LLAMA4_MAVERICK, PaperPrecision, expected_experts_touched,
    get_paper_model, verify_hf_geometry,
)


def test_meta_405b_tensor_count_uses_official_mp8_kv_not_mp16_duplicate():
    model = LLAMA3_405B
    assert (model.layers, model.hidden_size, model.query_heads, model.kv_heads) == (126, 16384, 128, 8)
    assert model.intermediate_size == 53248
    assert model.total_parameters == 405_853_388_800
    assert model.active_parameters == model.total_parameters
    assert not model.tie_word_embeddings
    assert model.kv_bytes_per_token() == 126 * 2 * 8 * 128 * 2


def test_maverick_dense_moe_and_fixed_chunks_have_distinct_schedules():
    model = LLAMA4_MAVERICK
    assert sum(model.is_moe(i) for i in range(48)) == 24
    assert [model.is_moe(i) for i in range(4)] == [False, True, False, True]
    assert [model.is_full_attention(i) for i in range(8)] == [False, False, False, True] * 2
    assert model.routed_experts == 128 and model.shared_experts == 1 and model.top_k == 1
    assert 400e9 < model.total_parameters < 402e9
    assert 16e9 < model.active_parameters < 18e9
    metadata = model.metadata()
    assert metadata["checkpoint_geometry_verified"] is False
    assert "gated" in metadata["geometry_verification"]
    assert "Official Meta torchtune llama4_maverick_17b_128e" in metadata["geometry_verification"]
    assert any("meta-pytorch/torchtune" in source and "_model_builders.py" in source for source in metadata["sources"])
    assert any("meta-pytorch/torchtune" in source and "_component_builders.py" in source for source in metadata["sources"])
    assert metadata["execution_backend"] == "paper_operation_event_model_not_native_ir"
    assert "not a dtype reported by the paper" in metadata["precision"]["basis"]


def test_paper_746_gb_is_a_labeled_unit_sensitivity_not_a_weight_override():
    model = LLAMA4_MAVERICK
    metadata = model.metadata()
    sensitivity = metadata["paper_capacity_unit_sensitivity"]
    assert sensitivity["reported_value"] == 746
    assert sensitivity["explicit_options"] == {"decimal_GB_bytes": 746_000_000_000,
                                               "binary_GiB_bytes": 746 * 1024**3}
    assert sensitivity["default_weight_bytes_changed"] is False
    assert metadata["text_weight_bytes"] == model.stage_weight_bytes() == 801_423_697_920
    assert metadata["text_weight_gib"] == pytest.approx(746.3839817047119)
    assert metadata["text_weight_decimal_gb"] == pytest.approx(801.42369792)
    assert model.metadata(PaperPrecision(weight_dtype="FP32"))["text_weight_bytes"] == 801_423_697_920 * 2
    assert "paper_capacity_unit_sensitivity" not in LLAMA3_405B.metadata()


@pytest.mark.parametrize("context,attended,read_tokens", [(0, 1, 0), (8191, 8192, 8191),
                                                        (8192, 1, 0), (8193, 2, 1),
                                                        (103500, 5197, 5196)])
def test_chunked_attention_wraps_aligned_chunks_instead_of_sliding(context, attended, read_tokens):
    work = LLAMA4_MAVERICK.layer_work(0, 4, context, tp=2)
    assert work.attention_kind == "chunked" and work.attended_tokens == attended
    assert work.kv_write_bytes == 2 * 4 * 4 * 128 * 2
    assert work.kv_read_bytes == read_tokens * work.kv_write_bytes
    assert work.kv_resident_bytes == attended * work.kv_write_bytes
    full = LLAMA4_MAVERICK.layer_work(3, 4, context, tp=2)
    assert full.attended_tokens == context + 1
    assert full.kv_read_bytes == context * full.kv_write_bytes


def test_attention_flops_use_query_heads_but_cache_uses_gqa_heads():
    work = LLAMA3_405B.layer_work(0, 2, 100, tp=8)
    assert work.local_query_heads == 16 and work.local_kv_heads == 1
    assert work.attention_flops == 4 * 2 * 16 * 128 * 101
    assert work.kv_read_bytes == 100 * 2 * 2 * 1 * 128 * 2


def test_tp_beyond_kv_count_replicates_heads_and_weights():
    model = LLAMA3_405B
    w8 = model.layer_work(0, 8, 10, tp=8)
    w16 = model.layer_work(0, 8, 10, tp=16)
    assert w16.local_kv_heads == w8.local_kv_heads == 1
    assert w16.kv_read_bytes == w8.kv_read_bytes
    # Q/O halve while K/V are replicated across pairs of TP ranks.
    kv_weights = 2 * model.hidden_size * 128 * 2
    assert w16.attention_weight_bytes == (w8.attention_weight_bytes - kv_weights) // 2 + kv_weights


def test_moe_batch_reads_distinct_experts_and_computes_selected_tokens_only():
    model = LLAMA4_MAVERICK
    one = model.layer_work(1, 1, 100)
    many = model.layer_work(1, 128, 100)
    assert one.expected_local_experts_touched == pytest.approx(1)
    assert one.routed_weight_bytes == pytest.approx(model.expert_parameters * 2)
    assert 80 < many.expected_local_experts_touched < 82
    assert many.routed_weight_bytes < model.expert_parameters * model.routed_experts * 2
    assert many.routed_flops == one.routed_flops * 128
    assert many.shared_flops == one.shared_flops * 128
    assert many.shared_weight_bytes == one.shared_weight_bytes
    assert one.router_weight_bytes == 5120 * 128 * 2
    assert many.weights_read_bytes < one.weights_read_bytes * 128


def test_ep_shards_only_experts_and_tokens_not_attention_or_shared_weights():
    model = LLAMA4_MAVERICK
    ep1 = model.layer_work(1, 128, 100, tp=2)
    ep4 = model.layer_work(1, 128, 100, tp=2, ep=4)
    assert ep4.local_batch_size == 32
    assert ep4.attention_weight_bytes == ep1.attention_weight_bytes
    assert ep4.shared_weight_bytes == ep1.shared_weight_bytes
    assert ep4.router_weight_bytes == ep1.router_weight_bytes
    assert ep4.routed_weight_bytes == pytest.approx(ep1.routed_weight_bytes / 4)
    assert ep4.compute_flops == pytest.approx(ep1.compute_flops / 4)
    assert ep4.kv_read_bytes == ep1.kv_read_bytes // 4
    assert ep1.expert_dispatch_bytes == 0
    assert ep4.expert_dispatch_bytes == 2 * 32 * 5120 * 2 * 3 / 4
    # Residency includes every local expert, not only those touched this step.
    assert model.stage_weight_bytes(tp=2, ep=4) > sum(
        model.layer_work(i, 128, 100, tp=2, ep=4).weights_read_bytes for i in range(48))


def test_pp_partitions_layers_and_does_not_divide_a_layer():
    model = LLAMA3_405B
    partitions = [list(model.layer_indices(4, stage)) for stage in range(4)]
    assert sum(partitions, []) == list(range(126))
    assert max(map(len, partitions)) - min(map(len, partitions)) == 1
    assert sum(model.stage_weight_bytes(pp=4, stage=stage) for stage in range(4)) == model.total_parameters * 2


def test_tied_embeddings_count_one_tensor_but_separate_pp_stages_need_copies():
    model = replace(LLAMA3_405B, tie_word_embeddings=True)
    table_bytes = model.hidden_size * model.vocabulary_size * 2
    assert LLAMA3_405B.total_parameters - model.total_parameters == table_bytes // 2
    assert model.stage_weight_bytes() == model.total_parameters * 2
    assert sum(model.stage_weight_bytes(pp=2, stage=stage) for stage in range(2)) == model.stage_weight_bytes() + table_bytes


def test_workspace_lifetimes_reuse_attention_and_ffn_storage():
    work = LLAMA4_MAVERICK.layer_work(1, 128, 5900, tp=4, ep=2)
    sizes = dict(work.live_buffers)
    assert work.intermediate_live_bytes < sum(sizes.values())
    common = sizes["residual"] + sizes["normalized_input"] + sizes["attention_output"]
    assert work.intermediate_live_bytes >= common + sizes["ffn_gate_up"] + sizes["router_logits"]
    assert all("score_matrix" not in key for key in sizes)


def test_endpoints_read_full_head_but_only_embedding_lookup_rows():
    work = LLAMA3_405B.endpoint_work(1, tp=8)
    assert work["embedding_read_bytes"] == 16384 * 2 // 8
    assert work["head_read_bytes"] == 16384 * 128256 * 2 // 8
    assert work["head_flops"] == 2 * 16384 * 128256 // 8


@pytest.mark.parametrize("kwargs", [{"weight_dtype": "FP8"}, {"kv_dtype": "INT8"},
                                    {"activation_dtype": "INT4"}, {"accumulator_dtype": "BF16"}])
def test_quantization_cannot_silently_omit_scale_traffic(kwargs):
    with pytest.raises(ValueError):
        PaperPrecision(**kwargs)


def test_explicit_fp32_changes_memory_but_not_architectural_matrix_flops():
    a = LLAMA3_405B.layer_work(0, 8, 1024)
    b = LLAMA3_405B.layer_work(0, 8, 1024, precision=PaperPrecision("FP32", "FP32", "FP32"))
    assert b.weights_read_bytes == a.weights_read_bytes * 2
    assert b.kv_read_bytes == a.kv_read_bytes * 2
    assert b.compute_flops == a.compute_flops


@pytest.mark.parametrize("tp,ep", [(3, 1), (16, 1), (2, 3), (0, 1), (1, 0)])
def test_maverick_invalid_parallelism_is_rejected(tp, ep):
    with pytest.raises(ValueError):
        LLAMA4_MAVERICK.validate_parallelism(tp, ep)


@pytest.mark.parametrize("batch,context", [(0, 1), (3, 1), (2, -1), (2, 1048576)])
def test_invalid_batch_or_context_is_not_rounded(batch, context):
    with pytest.raises(ValueError):
        LLAMA4_MAVERICK.layer_work(1, batch, context, ep=2)


def test_expected_expert_occupancy_has_topk_and_zero_boundaries():
    assert expected_experts_touched(0, 128) == 0
    assert expected_experts_touched(1, 128, 3) == pytest.approx(3)
    assert expected_experts_touched(100000, 128) == 128
    assert expected_experts_touched(1, 128, 128) == 128


def _maverick_config():
    m = LLAMA4_MAVERICK
    return {"tie_word_embeddings": False, "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 48, "intermediate_size": 8192,
        "intermediate_size_mlp": 16384, "num_attention_heads": 40,
        "num_key_value_heads": 8, "vocab_size": 202048, "max_position_embeddings": 1048576,
        "num_local_experts": 128, "num_experts_per_tok": 1, "attention_chunk_size": 8192,
        "interleave_moe_layer_step": 2,
        "layer_types": ["full_attention" if m.is_full_attention(i) else "chunked_attention" for i in range(48)]}}


def test_user_checkpoint_config_must_explicitly_match_maverick_not_scout_defaults():
    config = _maverick_config()
    verify_hf_geometry(LLAMA4_MAVERICK, config)
    config["text_config"]["num_local_experts"] = 16
    with pytest.raises(ValueError, match="num_local_experts"):
        verify_hf_geometry(LLAMA4_MAVERICK, config)


@pytest.mark.parametrize("field", ["interleave_moe_layer_step", "layer_types", "num_key_value_heads"])
def test_missing_geometry_never_falls_back_to_generic_transformers_defaults(field):
    config = _maverick_config()
    del config["text_config"][field]
    with pytest.raises(ValueError):
        verify_hf_geometry(LLAMA4_MAVERICK, config)


def test_model_lookup_has_no_silent_fallback():
    assert get_paper_model("llama3_405b") is LLAMA3_405B
    assert get_paper_model("llama4_maverick") is LLAMA4_MAVERICK
    with pytest.raises(ValueError):
        get_paper_model("llama4-scout")
