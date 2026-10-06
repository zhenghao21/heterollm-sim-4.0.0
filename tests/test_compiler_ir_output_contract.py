from dataclasses import replace

import pytest

from heterollm_sim.compiler_ir import compile_canonical_scenario
from heterollm_sim.ir import (
    MTPBranchSpec,
    ModelSpec,
    build_model_graph_from_layer_specs,
    model_graph_execution_view,
)
from heterollm_sim.reference import build_reference_scenario


_PLACEMENT_ENTRYPOINTS = (
    "tensor_bytes",
    "tensor_to_component",
    "weight_tensor_details",
    "rank_weight_shards",
)


def _with_lm_head_placement_entrypoint(scenario, entrypoint):
    placement = scenario.placement
    if entrypoint == "tensor_bytes":
        placement = replace(
            placement,
            tensor_bytes={"lm_head_weights": 16_384_000},
        )
    elif entrypoint == "tensor_to_component":
        placement = replace(
            placement,
            tensor_to_component={"lm_head_weights": "hbm0"},
        )
    elif entrypoint == "weight_tensor_details":
        placement = replace(
            placement,
            metadata={
                "control_plane": {
                    "decision": {
                        "weight_tensor_details": {
                            "lm_head_weights": {
                                "logical_bytes": 16_384_000,
                                "component_id": "hbm0",
                            }
                        }
                    }
                }
            },
        )
    elif entrypoint == "rank_weight_shards":
        placement = replace(
            placement,
            metadata={
                "control_plane": {
                    "decision": {
                        "rank_weight_shards": {
                            "lm_head_weights": [
                                {
                                    "rank_id": 0,
                                    "component_id": "gpu0",
                                    "logical_bytes": 16_384_000,
                                    "physical_bytes": 16_384_000,
                                }
                            ]
                        }
                    }
                }
            },
        )
    else:  # pragma: no cover - kept close to the parameter contract
        raise AssertionError("unknown placement entrypoint: {}".format(entrypoint))
    return replace(scenario, placement=placement)


def _scenario_with_output_contract(*, tied: bool):
    scenario = build_reference_scenario()
    source_view = model_graph_execution_view(scenario.model.graph)
    # Keep the acceptance fixture focused on the audit contract: an FP16
    # backbone with an explicitly FP32 logits tensor.  The measured byte
    # declarations remain intact and are intentionally not recomputed.
    layers = tuple(
        replace(item.layer, dtype="fp16", quantization=None)
        for item in source_view.layer_instances
    )
    graph = build_model_graph_from_layer_specs(
        scenario.model.name,
        layers,
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
    assert tensors["lm_head_weights"].dtype == "fp16"
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


@pytest.mark.parametrize("entrypoint", _PLACEMENT_ENTRYPOINTS)
@pytest.mark.parametrize("tied", (False, True))
def test_lm_head_placement_extensions_preserve_lossless_projection(
    entrypoint, tied
):
    baseline = compile_canonical_scenario(
        _scenario_with_output_contract(tied=tied)
    )
    canonical = compile_canonical_scenario(
        _with_lm_head_placement_entrypoint(
            _scenario_with_output_contract(tied=tied), entrypoint
        )
    )

    assert canonical.model.attributes["execution_projection"]["lossless"] is True
    baseline_lm_head = next(
        operator
        for operator in baseline.model.operators
        if operator.operator_id == "lm_head"
    )
    lm_head = next(
        operator
        for operator in canonical.model.operators
        if operator.operator_id == "lm_head"
    )
    assert lm_head.input_tensor_ids == baseline_lm_head.input_tensor_ids
    assert lm_head.output_tensor_ids == baseline_lm_head.output_tensor_ids
    assert lm_head.weight_tensor_ids == baseline_lm_head.weight_tensor_ids

    baseline_tensors = {
        tensor.tensor_id: tensor for tensor in baseline.model.tensors
    }
    tensors = {tensor.tensor_id: tensor for tensor in canonical.model.tensors}
    for tensor_id in ("lm_head_weights", "logits"):
        projected = tensors[tensor_id]
        source = baseline_tensors[tensor_id]
        assert projected.role == source.role
        assert projected.logical_bytes == source.logical_bytes
        assert projected.dtype == source.dtype
        assert projected.shape == source.shape
        assert projected.layout == source.layout
        # Placement extensions are compiler metadata; they must not alter
        # execution tensor semantics used by the authoritative contract.
        projected_attributes = dict(projected.attributes)
        projected_attributes.pop("placement_extension_tensor", None)
        assert projected_attributes == source.attributes


@pytest.mark.parametrize(
    ("entrypoint", "expected_lossless"),
    (
        ("tensor_to_component", True),
        ("rank_weight_shards", True),
        ("tensor_bytes", False),
        ("weight_tensor_details", False),
    ),
)
def test_none_source_bytes_use_contract_derivation_for_lm_head_extensions(
    entrypoint, expected_lossless
):
    # The reference graph leaves lm_head_weights.logical_bytes unspecified.
    # Component-only placement keeps that omission equivalent, while an
    # explicit byte declaration remains observable and must not be treated as
    # lossless without an authoritative source byte contract.
    scenario = build_reference_scenario()
    scenario = _with_lm_head_placement_entrypoint(scenario, entrypoint)
    canonical = compile_canonical_scenario(scenario)

    assert (
        canonical.model.attributes["execution_projection"]["lossless"]
        is expected_lossless
    )
    lm_head = next(
        operator
        for operator in canonical.model.operators
        if operator.operator_id == "lm_head"
    )
    assert lm_head.output_tensor_ids == ("logits",)
    tensors = {tensor.tensor_id: tensor for tensor in canonical.model.tensors}
    assert tensors["lm_head_weights"].dtype == "int8"
    assert tensors["lm_head_weights"].shape == (512, "V")
