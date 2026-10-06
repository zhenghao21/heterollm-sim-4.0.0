from dataclasses import replace

from heterollm_sim.compiler_ir import compile_canonical_scenario
from heterollm_sim.ir import (
    MTPBranchSpec,
    ModelSpec,
    build_model_graph_from_layer_specs,
    model_graph_execution_view,
)
from heterollm_sim.reference import build_reference_scenario


def _scenario_with_output_contract(*, tied: bool):
    scenario = build_reference_scenario()
    source_view = model_graph_execution_view(scenario.model.graph)
    graph = build_model_graph_from_layer_specs(
        scenario.model.name,
        tuple(item.layer for item in source_view.layer_instances),
        architecture=source_view.architecture,
        vocabulary_size=source_view.vocabulary_size,
        max_sequence_length=source_view.max_sequence_length,
        embedding_weight_bytes=source_view.embedding_weight_bytes,
        output_weight_bytes=source_view.output_weight_bytes,
        tie_word_embeddings=tied,
        output_head_dtype="fp32",
        mtp=MTPBranchSpec(
            prediction_layers=1,
            auxiliary_head=True,
            prediction_layer_weight_bytes=source_view.mtp_descriptors[0].weight_bytes,
            auxiliary_head_weight_bytes=source_view.mtp_descriptors[1].weight_bytes,
        ),
    )
    return replace(scenario, model=ModelSpec(name=scenario.model.name, graph=graph))


def test_canonical_projection_preserves_independent_fp32_lm_head_contract():
    canonical = compile_canonical_scenario(
        _scenario_with_output_contract(tied=False)
    )
    lm_head = next(
        operator for operator in canonical.model.operators
        if operator.operator_id == "lm_head"
    )
    assert lm_head.weight_tensor_ids == ("lm_head_weights",)
    assert lm_head.output_tensor_ids == ("logits",)
    tensors = {tensor.tensor_id: tensor for tensor in canonical.model.tensors}
    assert tensors["lm_head_weights"].attributes.get("storage_id") is None
    assert tensors["lm_head_weights"].dtype == "int8"
    assert tensors["lm_head_weights"].logical_bytes == 16_384_000
    assert tensors["logits"].dtype == "fp32"
    assert canonical.model.attributes["execution_projection"]["lossless"] is True


def test_canonical_projection_preserves_tied_lm_head_storage_alias():
    canonical = compile_canonical_scenario(
        _scenario_with_output_contract(tied=True)
    )
    lm_head = next(
        operator for operator in canonical.model.operators
        if operator.operator_id == "lm_head"
    )
    tensors = {tensor.tensor_id: tensor for tensor in canonical.model.tensors}
    assert lm_head.weight_tensor_ids == ("lm_head_weights",)
    assert tensors["lm_head_weights"].attributes["storage_id"] == "embedding_weights"
    assert tensors["logits"].dtype == "fp32"
    assert canonical.model.attributes["execution_projection"]["lossless"] is True
