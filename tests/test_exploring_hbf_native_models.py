"""Paper inputs are authored through existing native graph/compiler contracts."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from exploring_hbf_native_models import (
    build_native_paper_model, native_paper_model_catalog, native_paper_model_metadata,
)
from heterollm_sim.compiler_ir import _build_model_graph
from heterollm_sim.config import model_from_dict
from heterollm_sim.ir import model_graph_execution_layers
from heterollm_sim.model_presets import LayerPattern, UnsupportedPresetError, _layer_weight_bytes
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive


@pytest.mark.parametrize("gated_mlp", [False, True])
@pytest.mark.parametrize("shared_gate", [False, True])
def test_shared_expert_matrix_shape_and_optional_router_gate_are_independent(gated_mlp, shared_gate):
    h, i, shared, experts = 64, 96, 80, 8
    pattern = LayerPattern(repeat=1, kind="moe", hidden_size=h, intermediate_size=i,
        attention_heads=4, kv_heads=2, attention_head_dim=16, num_experts=experts,
        experts_per_token=1, shared_expert_intermediate_size=shared,
        shared_expert_gate=shared_gate, gated_mlp=gated_mlp)
    projections = 3 if gated_mlp else 2
    attention = 2 * h * (4 * 16) + 2 * h * (2 * 16)
    expected = attention + projections * h * i * experts + h * experts
    expected += projections * h * shared + (h if shared_gate else 0)
    assert _layer_weight_bytes(pattern) == 2 * expected


def test_405b_uses_official_bf16_kv8_native_graph_without_overwriting_gguf_preset():
    model = build_native_paper_model("llama3_405b")
    layers = model_graph_execution_layers(model.graph)
    assert len(layers) == 126
    assert all((layer.hidden_size, layer.intermediate_size, layer.attention_heads,
                layer.effective_kv_heads, layer.effective_attention_head_dim) ==
               (16384, 53248, 128, 8, 128) for layer in layers)
    assert all(layer.dtype == "bf16" and layer.quantization is None for layer in layers)
    assert model.max_sequence_length == 131072
    assert model.vocabulary_size == 128256
    assert model.graph.attributes["tie_word_embeddings"] is False
    assert model.output_weight_bytes == model.embedding_weight_bytes == 4_202_692_608
    assert model.total_declared_weight_bytes == model.metadata["native_matrix_weight_bytes"]
    assert model.metadata["undeclared_rmsnorm_affine_bytes"] == 8_290_304
    assert model.metadata["official_tensor_geometry_weight_bytes"] == 811_706_777_600
    tensors = {tensor.tensor_id: tensor for tensor in model.graph.tensors}
    assert tensors["logits"].dtype == "bf16"
    assert not tensors["lm_head_weights"].attributes.get("storage_id")
    assert model.metadata["paper_side_backend_used"] is False


def test_405b_roundtrips_and_expands_with_native_canonical_graph_builder():
    model = build_native_paper_model("llama3_405b")
    primitive = to_primitive(model)
    restored = model_from_dict(json.loads(json.dumps(primitive)))
    assert to_primitive(restored) == primitive
    scenario = replace(build_reference_scenario(), model=restored)
    compiled = _build_model_graph(scenario, layer_stages={
        layer.layer_id: 0 for layer in model_graph_execution_layers(model.graph)})
    assert sum(operator.op_kind == "full_attention" for operator in compiled.operators) == 126
    assert sum(operator.op_kind == "dense_mlp" for operator in compiled.operators) == 126
    assert any(operator.op_kind == "lm_head" for operator in compiled.operators)
    assert {tensor.tensor_id: tensor for tensor in compiled.tensors}["logits"].dtype == "bf16"


def test_maverick_uses_native_fixed_chunk_nope_and_sigmoid_input_contracts():
    metadata = native_paper_model_metadata("llama4_maverick")
    assert metadata["native_executable"] is True
    assert (metadata["routed_experts"], metadata["moe_layer_step"],
            metadata["attention_chunk_size"]) == (128, 2, 8192)
    model = build_native_paper_model("llama4_maverick")
    layers = model_graph_execution_layers(model.graph)
    assert len(layers) == 48
    assert sum(layer.is_moe for layer in layers) == 24
    assert sum(bool(layer.attention_chunk_size) for layer in layers) == 36
    assert sum(not layer.use_rope for layer in layers) == 12
    assert all(layer.attention_chunk_size == (0 if (i + 1) % 4 == 0 else 8192)
               for i, layer in enumerate(layers))
    assert all(layer.metadata["moe_routing"]["weight_application"] == "input" for layer in layers if layer.is_moe)
    assert model.metadata["official_tensor_geometry_weight_bytes"] == 801_423_697_920
    assert model_from_dict(json.loads(json.dumps(to_primitive(model)))).num_layers == 48


def test_catalog_metadata_is_not_mutable_model_state_and_unknown_model_fails():
    catalog = native_paper_model_catalog()
    catalog["llama3_405b"]["kv_heads"] = 16
    assert native_paper_model_metadata("llama3_405b")["kv_heads"] == 8
    with pytest.raises(ValueError, match="未知论文模型"):
        build_native_paper_model("scout_as_maverick")
