"""Pure GGUF projection and attention-execution descriptor contracts.

This module deliberately has no planner, scenario, placement, or task-graph
dependency.  Importers may memoize these immutable results at their own
compilation boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA = "heterollm.weight-projections/v1"
ATTENTION_EXECUTION_DESCRIPTOR_SCHEMA = "heterollm.attention-execution/v1"


@dataclass(frozen=True)
class ArtifactQuantizationSpec:
    name: str
    block_size: int
    payload_bytes: int
    metadata_bytes: int
    compute_weight_bits: int = 4
    dequant_operations_per_weight: int = 1


ARTIFACT_QUANTIZATION_REGISTRY: Mapping[str, ArtifactQuantizationSpec] = {
    # Importance-matrix formats used by Qwen3.8 GGUFs.  Payload/metadata are
    # split according to ggml type traits (their sum matches GGUF block bytes).
    "IQ3_XXS": ArtifactQuantizationSpec("IQ3_XXS", 256, 96, 2, compute_weight_bits=3),
    "IQ3_S": ArtifactQuantizationSpec("IQ3_S", 256, 96, 14, compute_weight_bits=3),
    "IQ4_NL": ArtifactQuantizationSpec("IQ4_NL", 32, 16, 2),
    "IQ4_XS": ArtifactQuantizationSpec("IQ4_XS", 256, 128, 8),
    "Q4_0": ArtifactQuantizationSpec("Q4_0", 32, 16, 2),
    "Q4_K": ArtifactQuantizationSpec("Q4_K", 256, 128, 16),
    "Q5_0": ArtifactQuantizationSpec(
        "Q5_0", 32, 20, 2, compute_weight_bits=5
    ),
    "Q5_K": ArtifactQuantizationSpec(
        "Q5_K", 256, 160, 16, compute_weight_bits=5
    ),
    "Q6_K": ArtifactQuantizationSpec(
        "Q6_K", 256, 192, 18, compute_weight_bits=6
    ),
    "Q8_0": ArtifactQuantizationSpec(
        "Q8_0", 32, 32, 2, compute_weight_bits=8
    ),
}


def _artifact_text(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())


def _artifact_quantization_aliases() -> Dict[str, str]:
    return {_artifact_text(name): name for name in ARTIFACT_QUANTIZATION_REGISTRY}


_ARTIFACT_QUANTIZATION_ALIASES = _artifact_quantization_aliases()


@dataclass(frozen=True)
class ProjectionSegment:
    segment_id: str
    physical_tensor_name: str
    k: int
    n: int
    artifact_spec: ArtifactQuantizationSpec
    physical_bytes: int
    tp_shard_axis: str


@dataclass(frozen=True)
class MaterializedProjectionSegment:
    segment: ProjectionSegment
    local_k: int
    local_n: int
    local_payload_bytes: int
    local_metadata_bytes: int
    local_block_count: int

    @property
    def local_physical_bytes(self) -> int:
        return self.local_payload_bytes + self.local_metadata_bytes

    def audit_metadata(self) -> Dict[str, object]:
        spec = self.segment.artifact_spec
        return {
            "segment_id": self.segment.segment_id,
            "physical_tensor_name": self.segment.physical_tensor_name,
            "format": spec.name,
            "global_k": self.segment.k,
            "global_n": self.segment.n,
            "local_k": self.local_k,
            "local_n": self.local_n,
            "tp_shard_axis": self.segment.tp_shard_axis,
            "physical_bytes": self.segment.physical_bytes,
            "local_payload_bytes": self.local_payload_bytes,
            "local_metadata_bytes": self.local_metadata_bytes,
            "local_physical_bytes": self.local_physical_bytes,
            "local_block_count": self.local_block_count,
            "block_size": spec.block_size,
            "payload_bytes_per_block": spec.payload_bytes,
            "metadata_bytes_per_block": spec.metadata_bytes,
            "compute_weight_bits": spec.compute_weight_bits,
        }


@dataclass(frozen=True)
class MaterializedWeightProjection:
    projection_id: str
    k: int
    n: int
    weight_storage_bytes: int
    weight_metadata_bytes: int
    weight_bits: int
    fused_dequant_operations: int
    segments: Tuple[MaterializedProjectionSegment, ...]
    full_physical_bytes: int

    @property
    def segment_audit(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(segment.audit_metadata() for segment in self.segments)

    def audit_metadata(self) -> Dict[str, object]:
        segment_audit = self.segment_audit
        formats = tuple(str(segment["format"]) for segment in segment_audit)
        unique_formats = tuple(dict.fromkeys(formats))
        artifact_label = (
            unique_formats[0]
            if len(unique_formats) == 1
            else "+".join(unique_formats)
        )
        return {
            "weight_projection_descriptor_applied": True,
            "weight_projection_descriptor_schema_version": (
                WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA
            ),
            "projection_id": self.projection_id,
            "projection_segment_count": len(segment_audit),
            "projection_segments": segment_audit,
            "projection_full_physical_bytes": self.full_physical_bytes,
            "projection_local_physical_bytes": (
                self.weight_storage_bytes + self.weight_metadata_bytes
            ),
            "artifact_quantization_applied": True,
            "artifact_quantization": artifact_label,
            "physical_weight_storage_bytes": self.weight_storage_bytes,
            "physical_weight_metadata_bytes": self.weight_metadata_bytes,
            "artifact_packed_bytes": self.weight_storage_bytes,
            "artifact_metadata_bytes": self.weight_metadata_bytes,
            "weight_compute_bits": self.weight_bits,
            "dequant_execution_model": "fused_quantized_dot",
            "fused_dequant_operations": self.fused_dequant_operations,
            "dequant_operations": self.fused_dequant_operations,
            "dequant_transcendental_operations": 0,
            "dequant_output_elements": 0,
            "dequant_operations_basis": (
                "projection_descriptor_block_layout_per_invocation"
            ),
            "dequant_operations_evidence": (
                "explicit_weight_projection_descriptor"
            ),
        }


@dataclass(frozen=True)
class AttentionExecutionDescriptor:
    query_heads: int
    kv_heads: int
    head_dim: int
    query_width: int
    gate_width: int
    q_projection_width: int
    rotary_dim: int
    qk_scale: float
    qk_norm: bool
    gate_activation: str


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(path))
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be non-empty text".format(path))
    return value.strip()


def canonical_artifact_quantization(value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("artifact quantization must be text")
    return _ARTIFACT_QUANTIZATION_ALIASES.get(_artifact_text(value))


def resolve_weight_projection(
    metadata: Mapping[str, object],
    projection_id: str,
) -> Optional[Tuple[ProjectionSegment, ...]]:
    """Resolve one semantic projection by direct keys only."""

    root = metadata.get("weight_projection_descriptors")
    if root is None:
        return None
    if not isinstance(root, Mapping):
        raise ValueError("weight_projection_descriptors must be a mapping")
    if root.get("schema_version") != WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA:
        raise ValueError(
            "weight_projection_descriptors.schema_version must be exactly {}"
            .format(WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA)
        )
    projections = root.get("projections")
    if not isinstance(projections, Mapping):
        raise ValueError(
            "weight_projection_descriptors.projections must be a mapping"
        )
    projection = projections.get(projection_id)
    if not isinstance(projection, Mapping):
        raise ValueError(
            "weight projection descriptor is missing {}".format(projection_id)
        )
    raw_segments = projection.get("segments")
    if not isinstance(raw_segments, (list, tuple)) or not raw_segments:
        raise ValueError(
            "weight projection {}.segments must be a non-empty sequence"
            .format(projection_id)
        )

    segments: List[ProjectionSegment] = []
    segment_ids: Set[str] = set()
    for index, raw_segment in enumerate(raw_segments):
        path = "weight projection {}.segments[{}]".format(
            projection_id, index
        )
        if not isinstance(raw_segment, Mapping):
            raise ValueError("{} must be a mapping".format(path))
        segment_id = _nonempty_text(
            raw_segment.get("segment_id"), path + ".segment_id"
        )
        if segment_id in segment_ids:
            raise ValueError(
                "weight projection {} has duplicate segment_id {}".format(
                    projection_id, segment_id
                )
            )
        segment_ids.add(segment_id)
        physical_tensor_name = _nonempty_text(
            raw_segment.get("physical_tensor_name"),
            path + ".physical_tensor_name",
        )
        k = _positive_int(raw_segment.get("k"), path + ".k")
        n = _positive_int(raw_segment.get("n"), path + ".n")
        format_name = _nonempty_text(
            raw_segment.get("format"), path + ".format"
        )
        canonical_format = canonical_artifact_quantization(format_name)
        artifact_spec = (
            ARTIFACT_QUANTIZATION_REGISTRY.get(canonical_format)
            if canonical_format is not None
            else None
        )
        if artifact_spec is None:
            raise ValueError(
                "{} has unsupported format {}".format(path, format_name)
            )
        physical_bytes = _positive_int(
            raw_segment.get("physical_bytes"), path + ".physical_bytes"
        )
        expected_physical_bytes = (
            n
            * int(math.ceil(k / float(artifact_spec.block_size)))
            * (artifact_spec.payload_bytes + artifact_spec.metadata_bytes)
        )
        if physical_bytes != expected_physical_bytes:
            raise ValueError(
                "{}.physical_bytes {} does not match {} block contract {}"
                .format(
                    path,
                    physical_bytes,
                    artifact_spec.name,
                    expected_physical_bytes,
                )
            )
        shard_axis = _nonempty_text(
            raw_segment.get("tp_shard_axis"), path + ".tp_shard_axis"
        ).casefold()
        if shard_axis not in {"n", "k", "replicated"}:
            raise ValueError(
                "{}.tp_shard_axis must be n, k, or replicated".format(path)
            )
        segments.append(
            ProjectionSegment(
                segment_id,
                physical_tensor_name,
                k,
                n,
                artifact_spec,
                physical_bytes,
                shard_axis,
            )
        )
    if len({segment.k for segment in segments}) != 1:
        raise ValueError(
            "weight projection {} fused segments must share K".format(
                projection_id
            )
        )
    return tuple(segments)


def _local_extent(
    size: int,
    degree: int,
    rank: int,
    allow_padding: bool,
) -> int:
    if size % degree and not allow_padding:
        raise ValueError(
            "dimension {} is not divisible by degree {}".format(size, degree)
        )
    return int(math.ceil(size / float(degree)))


def materialize_weight_projection(
    metadata: Mapping[str, object],
    projection_id: str,
    *,
    tp_degree: int = 1,
    tp_rank: int = 0,
    allow_padding: bool = True,
) -> Optional[MaterializedWeightProjection]:
    segments = resolve_weight_projection(metadata, projection_id)
    if segments is None:
        return None
    if (
        isinstance(tp_degree, bool)
        or not isinstance(tp_degree, int)
        or tp_degree <= 0
        or isinstance(tp_rank, bool)
        or not isinstance(tp_rank, int)
        or not 0 <= tp_rank < tp_degree
    ):
        raise ValueError("invalid projection TP degree or rank")

    local_k_values: Set[int] = set()
    local_n_total = 0
    storage_bytes = 0
    metadata_bytes = 0
    dequant_operations = 0
    materialized: List[MaterializedProjectionSegment] = []
    for segment in segments:
        local_k = segment.k
        local_n = segment.n
        if segment.tp_shard_axis == "k":
            local_k = _local_extent(
                segment.k, tp_degree, tp_rank, allow_padding
            )
        elif segment.tp_shard_axis == "n":
            local_n = _local_extent(
                segment.n, tp_degree, tp_rank, allow_padding
            )
        block_count = local_n * int(
            math.ceil(local_k / float(segment.artifact_spec.block_size))
        )
        local_payload_bytes = block_count * segment.artifact_spec.payload_bytes
        local_metadata_bytes = (
            block_count * segment.artifact_spec.metadata_bytes
        )
        local_k_values.add(local_k)
        local_n_total += local_n
        storage_bytes += local_payload_bytes
        metadata_bytes += local_metadata_bytes
        dequant_operations += (
            block_count
            * segment.artifact_spec.block_size
            * segment.artifact_spec.dequant_operations_per_weight
        )
        materialized.append(
            MaterializedProjectionSegment(
                segment,
                local_k,
                local_n,
                local_payload_bytes,
                local_metadata_bytes,
                block_count,
            )
        )
    if len(local_k_values) != 1:
        raise ValueError(
            "weight projection {} TP-local fused segments must share K"
            .format(projection_id)
        )
    return MaterializedWeightProjection(
        projection_id=projection_id,
        k=next(iter(local_k_values)),
        n=local_n_total,
        weight_storage_bytes=storage_bytes,
        weight_metadata_bytes=metadata_bytes,
        weight_bits=max(
            segment.artifact_spec.compute_weight_bits for segment in segments
        ),
        fused_dequant_operations=dequant_operations,
        segments=tuple(materialized),
        full_physical_bytes=sum(segment.physical_bytes for segment in segments),
    )


def resolve_attention_execution_descriptor(
    metadata: Mapping[str, object],
    *,
    attention_heads: int,
    kv_heads: int,
    head_dim: int,
    hidden_size: int,
) -> Optional[AttentionExecutionDescriptor]:
    raw = metadata.get("attention_execution_descriptor")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("attention_execution_descriptor must be a mapping")
    if raw.get("schema_version") != ATTENTION_EXECUTION_DESCRIPTOR_SCHEMA:
        raise ValueError(
            "attention_execution_descriptor.schema_version must be exactly {}"
            .format(ATTENTION_EXECUTION_DESCRIPTOR_SCHEMA)
        )
    query_heads = _positive_int(
        raw.get("query_heads"), "attention_execution_descriptor.query_heads"
    )
    descriptor_kv_heads = _positive_int(
        raw.get("kv_heads"), "attention_execution_descriptor.kv_heads"
    )
    descriptor_head_dim = _positive_int(
        raw.get("head_dim"), "attention_execution_descriptor.head_dim"
    )
    query_width = _positive_int(
        raw.get("query_width"), "attention_execution_descriptor.query_width"
    )
    gate_width = _positive_int(
        raw.get("gate_width"), "attention_execution_descriptor.gate_width"
    )
    q_projection_width = _positive_int(
        raw.get("q_projection_width"),
        "attention_execution_descriptor.q_projection_width",
    )
    rotary_dim = _positive_int(
        raw.get("rotary_dim"), "attention_execution_descriptor.rotary_dim"
    )
    qk_scale_raw = raw.get("qk_scale")
    if (
        isinstance(qk_scale_raw, bool)
        or not isinstance(qk_scale_raw, (int, float))
        or not math.isfinite(float(qk_scale_raw))
        or float(qk_scale_raw) <= 0.0
    ):
        raise ValueError(
            "attention_execution_descriptor.qk_scale must be positive and finite"
        )
    qk_norm = raw.get("qk_norm")
    if not isinstance(qk_norm, bool):
        raise ValueError(
            "attention_execution_descriptor.qk_norm must be boolean"
        )
    gate_activation = _nonempty_text(
        raw.get("gate_activation"),
        "attention_execution_descriptor.gate_activation",
    ).casefold()
    if gate_activation != "sigmoid":
        raise ValueError(
            "attention_execution_descriptor.gate_activation must be sigmoid"
        )
    if query_heads != attention_heads:
        raise ValueError("attention execution query_heads mismatch")
    if descriptor_kv_heads != kv_heads:
        raise ValueError("attention execution kv_heads mismatch")
    if descriptor_head_dim != head_dim:
        raise ValueError("attention execution head_dim mismatch")
    if query_width != query_heads * descriptor_head_dim:
        raise ValueError("attention execution query_width must equal heads*dim")
    if gate_width != query_width:
        raise ValueError(
            "attention execution gate_width must equal query_width for gating"
        )
    if q_projection_width != query_width + gate_width:
        raise ValueError(
            "attention execution q_projection_width must equal query+gate"
        )
    if query_heads % descriptor_kv_heads:
        raise ValueError("attention execution query_heads must divide by kv_heads")
    if rotary_dim > descriptor_head_dim or rotary_dim % 2:
        raise ValueError(
            "attention execution rotary_dim must be even and <= head_dim"
        )
    descriptor = AttentionExecutionDescriptor(
        query_heads,
        descriptor_kv_heads,
        descriptor_head_dim,
        query_width,
        gate_width,
        q_projection_width,
        rotary_dim,
        float(qk_scale_raw),
        qk_norm,
        gate_activation,
    )
    _validate_attention_projection_geometry(metadata, descriptor, hidden_size)
    return descriptor


def _validate_attention_projection_geometry(
    metadata: Mapping[str, object],
    descriptor: AttentionExecutionDescriptor,
    hidden_size: int,
) -> None:
    qkv_segments = resolve_weight_projection(metadata, "attention.qkv")
    output_segments = resolve_weight_projection(metadata, "attention.output")
    if qkv_segments is None or output_segments is None:
        raise ValueError(
            "attention execution descriptor requires weight projection descriptors"
        )
    qkv_by_id = {segment.segment_id: segment for segment in qkv_segments}
    if set(qkv_by_id) != {"q", "k", "v"}:
        raise ValueError(
            "attention.qkv must contain exactly q, k, and v segments"
        )
    kv_width = descriptor.kv_heads * descriptor.head_dim
    expected = {
        "q": (hidden_size, descriptor.q_projection_width),
        "k": (hidden_size, kv_width),
        "v": (hidden_size, kv_width),
    }
    for segment_id, (expected_k, expected_n) in expected.items():
        segment = qkv_by_id[segment_id]
        if (segment.k, segment.n) != (expected_k, expected_n):
            raise ValueError(
                "attention.qkv {} geometry must be [{}, {}]".format(
                    segment_id, expected_k, expected_n
                )
            )
        if segment.tp_shard_axis != "n":
            raise ValueError(
                "attention.qkv {} must shard on n".format(segment_id)
            )
    if len(output_segments) != 1:
        raise ValueError("attention.output must contain exactly one segment")
    output = output_segments[0]
    if (output.k, output.n) != (descriptor.query_width, hidden_size):
        raise ValueError(
            "attention.output geometry must be [{}, {}]".format(
                descriptor.query_width, hidden_size
            )
        )
    if output.tp_shard_axis != "k":
        raise ValueError("attention.output must shard on k")


__all__ = [
    "ARTIFACT_QUANTIZATION_REGISTRY",
    "ATTENTION_EXECUTION_DESCRIPTOR_SCHEMA",
    "ArtifactQuantizationSpec",
    "AttentionExecutionDescriptor",
    "MaterializedProjectionSegment",
    "MaterializedWeightProjection",
    "ProjectionSegment",
    "WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA",
    "canonical_artifact_quantization",
    "materialize_weight_projection",
    "resolve_attention_execution_descriptor",
    "resolve_weight_projection",
]
