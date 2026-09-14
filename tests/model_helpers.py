from __future__ import annotations

from typing import Mapping, Sequence

from heterollm_sim.ir import (
    SCHEMA_VERSION,
    LayerSpec,
    MTPBranchSpec,
    ModelSpec,
    build_model_graph_from_layer_specs,
    model_graph_execution_view,
    model_graph_execution_layers,
)


def model_from_layer_specs(
    name: str,
    layers: Sequence[LayerSpec],
    *,
    vocabulary_size: int = 0,
    max_sequence_length: int = 0,
    embedding_weight_bytes: int = 0,
    mtp: MTPBranchSpec | None = None,
    text_backbone_only: bool = True,
    supported_modalities: tuple[str, ...] = ("text",),
    excluded_subgraphs: tuple[str, ...] = (),
    architecture: str = "transformer",
    metadata: Mapping[str, object] | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> ModelSpec:
    graph = build_model_graph_from_layer_specs(
        name,
        layers,
        architecture=architecture,
        vocabulary_size=vocabulary_size,
        max_sequence_length=max_sequence_length,
        embedding_weight_bytes=embedding_weight_bytes,
        metadata=metadata,
        mtp=mtp,
    )
    return ModelSpec(
        name=name,
        graph=graph,
        text_backbone_only=text_backbone_only,
        supported_modalities=supported_modalities,
        excluded_subgraphs=excluded_subgraphs,
        metadata={} if metadata is None else metadata,
        schema_version=schema_version,
    )


def execution_layers(model: ModelSpec) -> tuple[LayerSpec, ...]:
    return model_graph_execution_layers(
        model.graph,
        schema_version=model.schema_version,
    )


def replace_model_layer_specs(
    model: ModelSpec,
    layers: Sequence[LayerSpec],
) -> ModelSpec:
    view = model_graph_execution_view(
        model.graph,
        schema_version=model.schema_version,
    )
    predictions = tuple(
        item for item in view.mtp_descriptors if item.prediction_index is not None
    )
    auxiliaries = tuple(
        item for item in view.mtp_descriptors if item.prediction_index is None
    )
    mtp = MTPBranchSpec(
        prediction_layers=len(predictions),
        auxiliary_head=bool(auxiliaries),
        prediction_layer_weight_bytes=(
            predictions[0].weight_bytes if predictions else 0
        ),
        auxiliary_head_weight_bytes=(
            auxiliaries[0].weight_bytes if auxiliaries else 0
        ),
    )
    return model_from_layer_specs(
        model.name,
        layers,
        vocabulary_size=model.vocabulary_size,
        max_sequence_length=model.max_sequence_length,
        embedding_weight_bytes=model.embedding_weight_bytes,
        mtp=mtp,
        text_backbone_only=model.text_backbone_only,
        supported_modalities=model.supported_modalities,
        excluded_subgraphs=model.excluded_subgraphs,
        architecture=model.architecture,
        metadata=model.metadata,
        schema_version=model.schema_version,
    )
