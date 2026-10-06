"""Offline model preset catalog built from public model configurations.

The catalog intentionally stores layer patterns rather than thousands of expanded
layer dictionaries.  ``materialize_model_payload`` expands a supported preset into
the ordinary :class:`ModelSpec` JSON shape used by the rest of the simulator.

This module performs no network or filesystem I/O at import or request time.
Upstream repository names and revisions are provenance, not runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .ir import (
    SCHEMA_VERSION,
    LayerSpec,
    LinearAttentionSpec,
    MTPBranchSpec,
    ModelSpec,
    build_model_graph_from_layer_specs,
)
from .serde import to_primitive
from .schema_v1 import ModelGraph, OperatorNode, OperatorPort, TensorValue
from .model_presets_shas import SOURCE_SHAS
from .precision import weight_storage_bits


EXACT = "exact"
APPROXIMATION = "analytical_approximation"
OUT_OF_DOMAIN = "out_of_domain"
SUPPORT_LEVELS = frozenset({EXACT, APPROXIMATION, OUT_OF_DOMAIN})
CATALOG_VERSION = "0.6"
CATALOG_CUTOFF_AT = "2026-10-06T00:00:00Z"

DIAGRAM_VERIFIED = "diagram_verified"
CONFIG_VERIFIED_NO_OFFICIAL_DIAGRAM = "config_verified_no_official_diagram"
UNPINNED_SOURCE = "unpinned_source"
GATED_CONFIG = "gated_config"
METADATA_ONLY_UNSUPPORTED_IR = "metadata_only_unsupported_ir"
ARCHITECTURE_EVIDENCE_STATUSES = frozenset(
    {
        DIAGRAM_VERIFIED,
        CONFIG_VERIFIED_NO_OFFICIAL_DIAGRAM,
        UNPINNED_SOURCE,
        GATED_CONFIG,
        METADATA_ONLY_UNSUPPORTED_IR,
    }
)


class UnsupportedPresetError(ValueError):
    """Raised when an out-of-domain catalog entry cannot be materialized."""


@dataclass(frozen=True)
class LinearAttentionPattern:
    """Config-only geometry for one supported linear-attention mixer."""

    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int = 1
    state_dtype: str = "fp32"
    output_gate: bool = True
    gate_activation: str = "silu"


@dataclass(frozen=True)
class LayerPattern:
    """A consecutive run of decoder layers sharing one public config shape."""

    repeat: int
    kind: str
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    num_experts: int = 1
    experts_per_token: int = 1
    dtype: str = "bf16"
    label: str = "decoder"
    sequence_mixer: str = "full_attention"
    linear_attention: Optional[LinearAttentionPattern] = None
    shared_expert_intermediate_size: int = 0
    shared_expert_gate: bool = False
    attention_head_dim: int = 0
    quantization: Optional[str] = None
    gated_mlp: bool = True


@dataclass(frozen=True)
class ArchitectureEvidence:
    """Serializable audit evidence for a bundled or imported architecture."""

    status: str
    source_type: str
    source_url: str
    config_url: str = ""
    notes: str = ""
    uncertainty: str = ""


@dataclass(frozen=True)
class PresetDefinition:
    preset_id: str
    name: str
    family: str
    parameter_scale: str
    source_repo: str
    source_revision: str
    license: str
    openness: str
    support_level: str
    notes: str
    vocabulary_size: int
    max_sequence_length: int
    patterns: Tuple[LayerPattern, ...]
    # The HF config contract is explicit: false means an independent LM head
    # tensor, while true means the head aliases the input embedding storage.
    tie_word_embeddings: bool = False
    architecture: str = "decoder_only_transformer"
    source_sha: Optional[str] = None
    config_hash: Optional[str] = None
    source: str = "bundled"
    access: str = "public"
    commercial_use: str = "allowed"
    modalities: Tuple[str, ...] = ("text",)
    supported_modalities: Tuple[str, ...] = ("text",)
    unsupported_subgraphs: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    coverage: str = "full_language_model"
    variants: Tuple[str, ...] = ()
    model_kind_override: Optional[str] = None
    layer_count_override: Optional[int] = None
    mtp: MTPBranchSpec = field(default_factory=MTPBranchSpec)
    text_backbone_only: bool = False
    architecture_evidence: Optional[ArchitectureEvidence] = None

    @property
    def layer_count(self) -> int:
        if self.layer_count_override is not None:
            return self.layer_count_override
        return sum(pattern.repeat for pattern in self.patterns)

    @property
    def model_kind(self) -> str:
        if self.model_kind_override is not None:
            return self.model_kind_override
        return "moe" if any(pattern.kind == "moe" for pattern in self.patterns) else "dense"


# Policy aliases prevent a compact catalog from accidentally labelling gated or
# bespoke-license weights as OSI open source.
_POLICIES: Mapping[str, Tuple[str, str, str]] = {
    "apache": ("Apache-2.0", "open_source", "allowed"),
    "llama": ("Llama Community License", "open_weight", "conditional"),
}


def _pattern(
    repeat: int,
    kind: str,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    experts: int = 1,
    top_k: int = 1,
    *,
    dtype: str = "bf16",
    label: str = "decoder",
    sequence_mixer: str = "full_attention",
    linear_attention: Optional[LinearAttentionPattern] = None,
    shared_expert_intermediate_size: int = 0,
    shared_expert_gate: bool = False,
    attention_head_dim: int = 0,
    quantization: Optional[str] = None,
    gated_mlp: bool = True,
) -> LayerPattern:
    return LayerPattern(
        repeat=repeat,
        kind=kind,
        hidden_size=hidden,
        intermediate_size=intermediate,
        attention_heads=heads,
        kv_heads=kv_heads,
        num_experts=experts,
        experts_per_token=top_k,
        dtype=dtype,
        label=label,
        sequence_mixer=sequence_mixer,
        linear_attention=linear_attention,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        shared_expert_gate=shared_expert_gate,
        attention_head_dim=attention_head_dim,
        quantization=quantization,
        gated_mlp=gated_mlp,
    )


def _definition(
    preset_id: str,
    name: str,
    family: str,
    scale: str,
    repo: str,
    policy: str,
    vocabulary_size: int,
    max_sequence_length: int,
    patterns: Iterable[LayerPattern],
    *,
    tie_word_embeddings: bool = False,
    support: str = EXACT,
    notes: str = "Public config.json fields; bf16 storage bytes are derived analytically.",
    revision: str = "main",
    architecture: str = "decoder_only_transformer",
    source_sha: Optional[str] = None,
    config_hash: Optional[str] = None,
    source: str = "bundled",
    access: str = "public",
    commercial_use: Optional[str] = None,
    modalities: Sequence[str] = ("text",),
    supported_modalities: Sequence[str] = ("text",),
    unsupported_subgraphs: Sequence[str] = (),
    limitations: Sequence[str] = (),
    coverage: str = "full_language_model",
    variants: Sequence[str] = (),
    model_kind_override: Optional[str] = None,
    layer_count_override: Optional[int] = None,
    mtp: Optional[MTPBranchSpec] = None,
    text_backbone_only: bool = False,
    architecture_evidence: Optional[ArchitectureEvidence] = None,
) -> PresetDefinition:
    license_name, openness, policy_commercial_use = _POLICIES[policy]
    return PresetDefinition(
        preset_id=preset_id,
        name=name,
        family=family,
        parameter_scale=scale,
        source_repo=repo,
        source_revision=revision,
        source_sha=source_sha if source_sha is not None else SOURCE_SHAS.get(repo),
        config_hash=config_hash,
        source=source,
        access=access,
        commercial_use=commercial_use or policy_commercial_use,
        license=license_name,
        openness=openness,
        support_level=support,
        notes=notes,
        vocabulary_size=vocabulary_size,
        max_sequence_length=max_sequence_length,
        tie_word_embeddings=bool(tie_word_embeddings),
        patterns=tuple(patterns),
        architecture=architecture,
        modalities=tuple(modalities),
        supported_modalities=tuple(supported_modalities),
        unsupported_subgraphs=tuple(unsupported_subgraphs),
        limitations=tuple(limitations),
        coverage=coverage,
        variants=tuple(variants),
        model_kind_override=model_kind_override,
        layer_count_override=layer_count_override,
        mtp=mtp or MTPBranchSpec(),
        text_backbone_only=text_backbone_only,
        architecture_evidence=architecture_evidence,
    )


def _dense(
    preset_id: str,
    name: str,
    family: str,
    scale: str,
    repo: str,
    layers: int,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    vocabulary_size: int,
    max_sequence_length: int,
    policy: str = "apache",
    gated_mlp: Optional[bool] = None,
    attention_head_dim: int = 0,
    **kwargs: Any,
) -> PresetDefinition:
    if gated_mlp is None:
        gated_mlp = True
    return _definition(
        preset_id, name, family, scale, repo, policy,
        vocabulary_size, max_sequence_length,
        (_pattern(layers, "dense", hidden, intermediate, heads, kv_heads,
                  attention_head_dim=attention_head_dim,
                  gated_mlp=bool(gated_mlp)),),
        **kwargs,
    )


def _moe(
    preset_id: str,
    name: str,
    family: str,
    scale: str,
    repo: str,
    layers: int,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    experts: int,
    top_k: int,
    vocabulary_size: int,
    max_sequence_length: int,
    policy: str = "apache",
    attention_head_dim: int = 0,
    **kwargs: Any,
) -> PresetDefinition:
    return _definition(
        preset_id, name, family, scale, repo, policy,
        vocabulary_size, max_sequence_length,
        (_pattern(layers, "moe", hidden, intermediate, heads, kv_heads, experts, top_k,
                  attention_head_dim=attention_head_dim),),
        **kwargs,
    )










# Each row is one distinct base architecture/scale configuration.  Chat/instruct
# aliases with an identical config are deliberately excluded.
_PRESETS: Tuple[PresetDefinition, ...] = (
    # Qwen2.5 official text-only causal models.  Values come from the
    # corresponding official Hugging Face config.json files.
    _dense(
        "qwen2_5-0_5b", "Qwen2.5-0.5B", "Qwen2.5", "0.5B",
        "Qwen/Qwen2.5-0.5B", 24, 896, 4864, 14, 2, 151936, 32768,
        policy="apache", architecture="qwen2", attention_head_dim=64,
    ),
    _dense(
        "qwen2_5-1_5b", "Qwen2.5-1.5B", "Qwen2.5", "1.5B",
        "Qwen/Qwen2.5-1.5B", 28, 1536, 8960, 12, 2, 151936, 32768,
        policy="apache", architecture="qwen2", attention_head_dim=128,
    ),
    _dense(
        "qwen2_5-3b", "Qwen2.5-3B", "Qwen2.5", "3B",
        "Qwen/Qwen2.5-3B", 36, 2048, 11008, 16, 2, 151936, 32768,
        policy="apache", architecture="qwen2", attention_head_dim=128,
    ),
    _dense(
        "qwen2_5-7b", "Qwen2.5-7B", "Qwen2.5", "7B",
        "Qwen/Qwen2.5-7B", 28, 3584, 18944, 28, 4, 152064, 131072,
        policy="apache", architecture="qwen2", attention_head_dim=128,
    ),
    _dense(
        "qwen2_5-14b", "Qwen2.5-14B", "Qwen2.5", "14B",
        "Qwen/Qwen2.5-14B", 48, 5120, 13824, 40, 8, 152064, 131072,
        policy="apache", architecture="qwen2", attention_head_dim=128,
    ),
    _dense(
        "qwen2_5-32b", "Qwen2.5-32B", "Qwen2.5", "32B",
        "Qwen/Qwen2.5-32B", 64, 5120, 27648, 40, 8, 152064, 131072,
        policy="apache", architecture="qwen2", attention_head_dim=128,
    ),
    _dense(
        "qwen2_5-72b", "Qwen2.5-72B", "Qwen2.5", "72B",
        "Qwen/Qwen2.5-72B", 80, 8192, 29568, 64, 8, 152064, 131072,
        policy="apache", architecture="qwen2", attention_head_dim=128,
    ),

    # Qwen3 official text-only causal models.
    _dense(
        "qwen3-0_6b", "Qwen3-0.6B", "Qwen3", "0.6B",
        "Qwen/Qwen3-0.6B", 28, 1024, 3072, 16, 8, 151936, 40960,
        policy="apache", architecture="qwen3", attention_head_dim=128,
    ),
    _dense(
        "qwen3-1_7b", "Qwen3-1.7B", "Qwen3", "1.7B",
        "Qwen/Qwen3-1.7B", 28, 2048, 6144, 16, 8, 151936, 40960,
        policy="apache", architecture="qwen3", attention_head_dim=128,
    ),
    _dense(
        "qwen3-4b", "Qwen3-4B", "Qwen3", "4B",
        "Qwen/Qwen3-4B", 36, 2560, 9728, 32, 8, 151936, 40960,
        policy="apache", architecture="qwen3", attention_head_dim=128,
    ),
    _dense(
        "qwen3-8b", "Qwen3-8B", "Qwen3", "8B",
        "Qwen/Qwen3-8B", 36, 4096, 12288, 32, 8, 151936, 40960,
        policy="apache", architecture="qwen3", attention_head_dim=128,
    ),
    _dense(
        "qwen3-14b", "Qwen3-14B", "Qwen3", "14B",
        "Qwen/Qwen3-14B", 40, 5120, 17408, 40, 8, 151936, 40960,
        policy="apache", architecture="qwen3", attention_head_dim=128,
    ),
    _dense(
        "qwen3-32b", "Qwen3-32B", "Qwen3", "32B",
        "Qwen/Qwen3-32B", 64, 5120, 25600, 64, 8, 151936, 40960,
        policy="apache", architecture="qwen3", attention_head_dim=128,
    ),
    _moe(
        "qwen3-30b-a3b", "Qwen3-30B-A3B", "Qwen3", "30B/A3B",
        "Qwen/Qwen3-30B-A3B-Base", 48, 2048, 768, 32, 4, 128, 8,
        151936, 32768, policy="apache", support=APPROXIMATION,
        attention_head_dim=128,
        architecture="qwen3",
        notes="官方 config.json 的 MoE 几何；当前 IR 保留专家数量和 Top-K，专家路由字节数按分析模型估算。",
    ),
    _moe(
        "qwen3-235b-a22b", "Qwen3-235B-A22B", "Qwen3", "235B/A22B",
        "Qwen/Qwen3-235B-A22B", 94, 4096, 1536, 64, 4, 128, 8,
        151936, 40960, policy="apache", support=APPROXIMATION,
        attention_head_dim=128,
        architecture="qwen3",
        notes="官方 config.json 的 MoE 几何；当前 IR 保留专家数量和 Top-K，专家路由字节数按分析模型估算。",
    ),

    # Meta Llama official text-only models.  The official repositories are
    # gated and use the Llama Community License; dimensions remain usable for
    # graph simulation without downloading weights.
    _dense(
        "llama3_2-1b", "Llama 3.2 1B", "Llama3.2", "1B",
        "meta-llama/Llama-3.2-1B", 16, 2048, 8192, 32, 8, 128256, 131072,
        policy="llama", architecture="llama", access="gated", attention_head_dim=64,
        source_sha="4e20de362430cd3b72f300e6b0f18e50e7166e08",
        notes="官方 Meta Llama 配置；仓库 gated，仿真仅使用公开架构元数据，不加载权重。",
    ),
    _dense(
        "llama3_2-3b", "Llama 3.2 3B", "Llama3.2", "3B",
        "meta-llama/Llama-3.2-3B", 28, 3072, 8192, 24, 8, 128256, 131072,
        policy="llama", architecture="llama", access="gated", attention_head_dim=128,
        source_sha="13afe5124825b4f3751f836b40dafda64c1ed062",
        notes="官方 Meta Llama 配置；仓库 gated，仿真仅使用公开架构元数据，不加载权重。",
    ),
    _dense(
        "llama3_1-8b", "Llama 3.1 8B", "Llama3.1", "8B",
        "meta-llama/Llama-3.1-8B", 32, 4096, 14336, 32, 8, 128256, 131072,
        policy="llama", architecture="llama", access="gated", attention_head_dim=128,
        source_sha="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        notes="官方 Meta Llama 配置；仓库 gated，仿真仅使用公开架构元数据，不加载权重。",
    ),
    _dense(
        "llama3_1-70b", "Llama 3.1 70B", "Llama3.1", "70B",
        "meta-llama/Llama-3.1-70B", 80, 8192, 28672, 64, 8, 128256, 131072,
        policy="llama", architecture="llama", access="gated", attention_head_dim=128,
        source_sha="349b2ddb53ce8f2849a6c168a81980ab25258dac",
        notes="官方 Meta Llama 配置；仓库 gated，仿真仅使用公开架构元数据，不加载权重。",
    ),
    _dense(
        "llama3_1-405b", "Llama 3.1 405B", "Llama3.1", "405B",
        "meta-llama/Llama-3.1-405B", 126, 16384, 53248, 128, 8, 128256, 131072,
        policy="llama", architecture="llama", access="gated", attention_head_dim=128,
        source_sha="b906e4dc842aa489c962f9db26554dcfdde901fe",
        notes="官方 Meta Llama 配置；仓库 gated，仿真仅使用公开架构元数据，不加载权重。",
    ),
    _dense(
        "llama3_3-70b", "Llama 3.3 70B Instruct", "Llama3.3", "70B",
        "meta-llama/Llama-3.3-70B-Instruct", 80, 8192, 28672, 64, 8, 128256, 131072,
        policy="llama", architecture="llama", access="gated", attention_head_dim=128,
        source_sha="6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        notes="官方 Meta Llama 配置；仓库 gated，仿真仅使用公开架构元数据，不加载权重。",
    ),
)



_BY_ID: Mapping[str, PresetDefinition] = {item.preset_id: item for item in _PRESETS}
if len(_BY_ID) != len(_PRESETS):  # import-time invariant, never user input
    raise RuntimeError("duplicate model preset id")


def _architecture_config_url(definition: PresetDefinition) -> str:
    ref = definition.source_sha or definition.source_revision or "main"
    return "https://huggingface.co/{}/blob/{}/config.json".format(
        definition.source_repo,
        ref,
    )


# These are intentionally preset-level (rather than family-level) promotions:
# the reviewed figure often covers one scale or variant only.  ``config_url``
# is filled from the preset's pinned revision when the evidence is resolved.
DIAGRAM_OVERRIDES: Mapping[str, ArchitectureEvidence] = {}
CONSERVATIVE_DIAGRAM_OVERRIDES: Mapping[str, ArchitectureEvidence] = {}


def _resolved_architecture_evidence(
    definition: PresetDefinition,
    evidence: ArchitectureEvidence,
) -> ArchitectureEvidence:
    """Attach the preset's fixed HF config URL to paper/vendor evidence."""

    return replace(evidence, config_url=_architecture_config_url(definition))


def _architecture_evidence(definition: PresetDefinition) -> ArchitectureEvidence:
    config_url = _architecture_config_url(definition)
    if definition.access == "gated":
        conservative = CONSERVATIVE_DIAGRAM_OVERRIDES.get(definition.preset_id)
        if conservative is not None:
            return _resolved_architecture_evidence(
                definition,
                replace(conservative, status=GATED_CONFIG),
            )
        return ArchitectureEvidence(
            status=GATED_CONFIG,
            source_type="official_huggingface_gated_config",
            source_url=config_url,
            config_url=config_url,
            notes=(
                "Repository metadata resolved, but anonymous config access was gated "
                "during the catalog audit; architecture dimensions remain a bundled "
                "snapshot and are not claimed as anonymously re-readable."
            ),
            uncertainty="Gated upstream config access; revalidation requires credentials.",
        )
    if definition.source_sha is None:
        return ArchitectureEvidence(
            status=UNPINNED_SOURCE,
            source_type="bundled_config_snapshot",
            source_url=config_url,
            config_url=config_url,
            notes=(
                "The catalog could not anonymously resolve and pin the upstream "
                "repository revision; dimensions are retained from the bundled "
                "config snapshot."
            ),
            uncertainty="No pinned source commit is available in the offline catalog.",
        )
    if definition.support_level == OUT_OF_DOMAIN or definition.coverage == "metadata_only":
        conservative = CONSERVATIVE_DIAGRAM_OVERRIDES.get(definition.preset_id)
        if conservative is not None:
            return _resolved_architecture_evidence(definition, conservative)
        unsupported = ", ".join(definition.unsupported_subgraphs) or definition.architecture
        return ArchitectureEvidence(
            status=METADATA_ONLY_UNSUPPORTED_IR,
            source_type="official_pinned_config",
            source_url=config_url,
            config_url=config_url,
            notes=(
                "Official pinned config metadata is retained, but the current ModelSpec "
                "IR cannot faithfully materialize: {}."
            ).format(unsupported),
            uncertainty="Display-only preset; no executable topology is generated.",
        )
    if definition.architecture_evidence is not None:
        return _resolved_architecture_evidence(
            definition,
            definition.architecture_evidence,
        )
    diagram = DIAGRAM_OVERRIDES.get(definition.preset_id)
    if diagram is not None:
        return _resolved_architecture_evidence(definition, diagram)
    return ArchitectureEvidence(
        status=CONFIG_VERIFIED_NO_OFFICIAL_DIAGRAM,
        source_type="official_pinned_config",
        source_url=config_url,
        config_url=config_url,
        notes=(
            "Layer counts and tensor dimensions are grounded in the official config "
            "at the pinned commit; no separate official architecture diagram is "
            "asserted by this catalog entry."
        ),
        uncertainty="Architecture graph is generated from config fields, not a manually traced paper figure.",
    )


def _architecture_evidence_metadata(definition: PresetDefinition) -> Dict[str, Any]:
    evidence = _architecture_evidence(definition)
    if evidence.status not in ARCHITECTURE_EVIDENCE_STATUSES:
        raise RuntimeError("unknown architecture evidence status: {}".format(evidence.status))
    return to_primitive(evidence)


def _metadata(definition: PresetDefinition) -> Dict[str, Any]:
    """Return a fresh, lightweight and JSON-compatible metadata mapping."""

    return {
        "id": definition.preset_id,
        "name": definition.name,
        "family": definition.family,
        "parameter_scale": definition.parameter_scale,
        "architecture": definition.architecture,
        "model_kind": definition.model_kind,
        "layer_count": definition.layer_count,
        "vocabulary_size": definition.vocabulary_size,
        "max_sequence_length": definition.max_sequence_length,
        "tie_word_embeddings": definition.tie_word_embeddings,
        "source_repo": definition.source_repo,
        "source_revision": definition.source_revision,
        "source_sha": definition.source_sha,
        "config_hash": definition.config_hash,
        "provenance_status": _provenance_status(definition),
        "source": definition.source,
        "license": definition.license,
        "openness": definition.openness,
        "access": definition.access,
        "commercial_use": definition.commercial_use,
        "support_level": definition.support_level,
        "generation_allowed": (
            definition.support_level != OUT_OF_DOMAIN
            and bool(definition.patterns)
            and definition.layer_count > 0
        ),
        "modalities": list(definition.modalities),
        "supported_modalities": list(definition.supported_modalities),
        "unsupported_subgraphs": list(definition.unsupported_subgraphs),
        "limitations": list(_metadata_limitations(definition)),
        "architecture_evidence": _architecture_evidence_metadata(definition),
        "coverage": definition.coverage,
        "variants": list(definition.variants),
        "notes": definition.notes,
    }


def model_preset_metadata(definition: PresetDefinition) -> Dict[str, Any]:
    """Public metadata builder used by bundled and user-imported catalogs."""

    return _metadata(definition)


def _provenance_status(definition: PresetDefinition) -> str:
    if definition.source_sha and definition.config_hash:
        return "pinned_commit_and_config"
    if definition.source_sha:
        return "pinned_commit"
    if definition.access == "requires_authentication":
        return "unverified_requires_authentication"
    return "unverified_revision"


def _metadata_limitations(definition: PresetDefinition) -> Tuple[str, ...]:
    limitations = list(definition.limitations)
    if (
        definition.source_sha is None
        and definition.access == "requires_authentication"
    ):
        limitations.append(
            "Source provenance / 来源验证：the upstream revision cannot be anonymously resolved and pinned; authentication is required. This entry uses the bundled config snapshot, keeps source_sha empty, and is not claimed as a verified pinned revision. / 无法匿名验证并固定上游 revision，访问需要认证；当前使用内置 config 快照，source_sha 保持为空，不声明为已验证固定版本。"
        )
    return tuple(limitations)


def list_model_presets() -> Tuple[Dict[str, Any], ...]:
    """List stable metadata ordered by URL-safe preset ID."""

    return tuple(_metadata(item) for item in sorted(_PRESETS, key=lambda item: item.preset_id))


def get_model_preset(preset_id: str) -> PresetDefinition:
    """Return an immutable catalog definition, raising ``KeyError`` if absent."""

    return _BY_ID[preset_id]


def _layer_weight_bytes(pattern: LayerPattern) -> int:
    """Estimate resident weights from public shape and precision declarations.

    Q/K/V/O includes grouped-query K/V width.  Gated MLPs use three matrices;
    for MoE the expert matrices are resident even though only top-k are active.
    The estimate is intentionally disclosed in payload metadata.
    """

    storage_bits = _pattern_weight_storage_bits(pattern)
    if pattern.sequence_mixer == "linear_attention" and pattern.linear_attention:
        linear = pattern.linear_attention
        query_width = linear.key_heads * linear.key_head_dim
        key_width = query_width
        value_width = linear.value_heads * linear.value_head_dim
        attention_elements = pattern.hidden_size * (
            query_width + key_width + value_width
        )
        attention_elements += value_width * pattern.hidden_size
        if linear.output_gate:
            attention_elements += pattern.hidden_size * value_width
    else:
        head_dim = pattern.attention_head_dim or (
            pattern.hidden_size // pattern.attention_heads
        )
        query_width = pattern.attention_heads * head_dim
        kv_width = pattern.kv_heads * head_dim
        attention_elements = 2 * pattern.hidden_size * query_width
        attention_elements += 2 * pattern.hidden_size * kv_width
    mlp_elements = (3 if pattern.gated_mlp else 2) * pattern.hidden_size * pattern.intermediate_size
    if pattern.kind == "moe":
        mlp_elements *= pattern.num_experts
        mlp_elements += pattern.hidden_size * pattern.num_experts
        if pattern.shared_expert_intermediate_size:
            mlp_elements += (
                (3 if pattern.shared_expert_gate else 2)
                * pattern.hidden_size
                * pattern.shared_expert_intermediate_size
            )
            if pattern.shared_expert_gate:
                mlp_elements += pattern.hidden_size
    elements = attention_elements + mlp_elements
    return (elements * storage_bits + 7) // 8


def _pattern_weight_storage_bits(pattern: LayerPattern) -> int:
    return weight_storage_bits(
        pattern.dtype,
        pattern.quantization,
        unsupported_dtype_message=(
            "unsupported preset dtype {}".format(pattern.dtype)
        ),
        unsupported_quantization_message=(
            "unsupported preset quantization {}".format(
                pattern.quantization
            )
        ),
    )


def _weight_bytes_method(pattern: LayerPattern, *, embedding: bool) -> str:
    if pattern.dtype == "bf16" and not pattern.quantization:
        return (
            "bf16_input_embedding_estimate"
            if embedding
            else "bf16_matrix_shape_estimate"
        )
    return (
        "declared_precision_input_embedding_estimate"
        if embedding
        else "declared_precision_matrix_shape_estimate"
    )


def materialize_model_payload(preset_id: str) -> Dict[str, Any]:
    """Expand one supported preset to the standard ``ModelSpec`` payload shape."""

    definition = get_model_preset(preset_id)
    return materialize_preset_definition(definition)


def materialize_preset_definition(definition: PresetDefinition) -> Dict[str, Any]:
    """Expand a supported bundled or imported definition into ModelSpec JSON."""

    if definition.support_level == OUT_OF_DOMAIN:
        raise UnsupportedPresetError(
            "模型预设 {!r} 超出当前可执行组件域，只能查看结构图，不能映射或仿真".format(definition.preset_id)
        )
    if not definition.patterns:
        raise UnsupportedPresetError(
            "模型预设 {!r} 没有可生成的层模式".format(definition.preset_id)
        )

    layers = []
    layer_index = 0
    for pattern_index, pattern in enumerate(definition.patterns):
        for offset in range(pattern.repeat):
            layer_payload: Dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "layer_id": "layer-{:03d}".format(layer_index),
                    "kind": pattern.kind,
                    "hidden_size": pattern.hidden_size,
                    "intermediate_size": pattern.intermediate_size,
                    "attention_heads": pattern.attention_heads,
                    "kv_heads": pattern.kv_heads,
                    "attention_head_dim": pattern.attention_head_dim,
                    "sequence_mixer": pattern.sequence_mixer,
                    "num_experts": pattern.num_experts,
                    "experts_per_token": pattern.experts_per_token,
                    "shared_expert_intermediate_size": pattern.shared_expert_intermediate_size,
                    "shared_expert_gate": pattern.shared_expert_gate,
                    "gated_mlp": pattern.gated_mlp,
                    "dtype": pattern.dtype,
                    "quantization": pattern.quantization,
                    "weight_bytes": _layer_weight_bytes(pattern),
                    "metadata": {
                        "preset_pattern": pattern.label,
                        "pattern_index": pattern_index,
                        "pattern_offset": offset,
                        "weight_bytes_method": _weight_bytes_method(
                            pattern, embedding=False
                        ),
                        "weight_storage_bits": (
                            _pattern_weight_storage_bits(pattern)
                        ),
                    },
                }
            if pattern.linear_attention is not None:
                linear = pattern.linear_attention
                layer_payload["linear_attention"] = {
                    "key_heads": linear.key_heads,
                    "value_heads": linear.value_heads,
                    "key_head_dim": linear.key_head_dim,
                    "value_head_dim": linear.value_head_dim,
                    "conv_kernel_size": linear.conv_kernel_size,
                    "state_dtype": linear.state_dtype,
                    "output_gate": linear.output_gate,
                    "gate_activation": linear.gate_activation,
                }
            layers.append(layer_payload)
            layer_index += 1

    hidden_size = definition.patterns[0].hidden_size
    embedding_weight_bytes = (
        definition.vocabulary_size
        * hidden_size
        * _pattern_weight_storage_bits(definition.patterns[0])
        + 7
    ) // 8
    # Keep the logical LM-head matrix visible even when it aliases the
    # embedding storage.  The graph builder records the alias and de-duplicates
    # resident bytes; dropping this value would make output_weight_bytes mean
    # "unknown" rather than "shared".
    output_weight_bytes = embedding_weight_bytes
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": definition.name,
        "architecture": definition.architecture,
        "vocabulary_size": definition.vocabulary_size,
        "max_sequence_length": definition.max_sequence_length,
        "embedding_weight_bytes": embedding_weight_bytes,
        "text_backbone_only": definition.text_backbone_only,
        "supported_modalities": list(definition.supported_modalities),
        "excluded_subgraphs": list(definition.unsupported_subgraphs),
        "metadata": {
            "preset_id": definition.preset_id,
            "family": definition.family,
            "parameter_scale": definition.parameter_scale,
            "model_kind": definition.model_kind,
            "source_repo": definition.source_repo,
            "source_revision": definition.source_revision,
            "source_sha": definition.source_sha,
            "config_hash": definition.config_hash,
            "provenance_status": _provenance_status(definition),
            "source": definition.source,
            "license": definition.license,
            "openness": definition.openness,
            "access": definition.access,
            "commercial_use": definition.commercial_use,
            "support_level": definition.support_level,
            "coverage": definition.coverage,
            "limitations": list(_metadata_limitations(definition)),
            "architecture_evidence": _architecture_evidence_metadata(definition),
            "notes": definition.notes,
            "embedding_weight_bytes_method": _weight_bytes_method(
                definition.patterns[0], embedding=True
            ),
            "embedding_weight_storage_bits": (
                _pattern_weight_storage_bits(definition.patterns[0])
            ),
            "output_weight_bytes": output_weight_bytes,
            "output_weight_storage_bits": _pattern_weight_storage_bits(definition.patterns[0]),
            "tie_word_embeddings": definition.tie_word_embeddings,
        },
    }
    layer_specs = tuple(
        LayerSpec(
            layer_id=str(layer["layer_id"]),
            kind=str(layer["kind"]),
            hidden_size=int(layer["hidden_size"]),
            intermediate_size=int(layer["intermediate_size"]),
            attention_heads=int(layer["attention_heads"]),
            kv_heads=int(layer["kv_heads"]),
            attention_head_dim=int(layer["attention_head_dim"]),
            sequence_mixer=str(layer["sequence_mixer"]),
            linear_attention=(
                LinearAttentionSpec(**dict(layer["linear_attention"]))
                if layer.get("linear_attention") is not None
                else None
            ),
            num_experts=int(layer["num_experts"]),
            experts_per_token=int(layer["experts_per_token"]),
            shared_expert_intermediate_size=int(
                layer["shared_expert_intermediate_size"]
            ),
            shared_expert_gate=bool(layer["shared_expert_gate"]),
            gated_mlp=bool(layer.get("gated_mlp", True)),
            dtype=str(layer["dtype"]),
            quantization=layer["quantization"],
            weight_bytes=int(layer["weight_bytes"]),
            metadata=dict(layer["metadata"]),
            schema_version=SCHEMA_VERSION,
        )
        for layer in layers
    )
    graph = build_model_graph_from_layer_specs(
        definition.name,
        layer_specs,
        architecture=definition.architecture,
        vocabulary_size=definition.vocabulary_size,
        max_sequence_length=definition.max_sequence_length,
        embedding_weight_bytes=embedding_weight_bytes,
        output_weight_bytes=output_weight_bytes,
        tie_word_embeddings=definition.tie_word_embeddings,
        metadata=payload["metadata"],
        mtp=MTPBranchSpec(
            prediction_layers=definition.mtp.prediction_layers,
            auxiliary_head=definition.mtp.auxiliary_head,
            # Zero is intentional: the graph builder derives the exact
            # hidden×hidden / hidden×vocabulary matrix bytes from the last
            # layer's dtype and quantization.  Substituting a whole decoder
            # layer estimate here would overstate MTP capacity by orders of
            # magnitude.
            prediction_layer_weight_bytes=(
                definition.mtp.prediction_layer_weight_bytes
            ),
            auxiliary_head_weight_bytes=(
                definition.mtp.auxiliary_head_weight_bytes
            ),
        ),
    )
    return to_primitive(
        ModelSpec(
            name=definition.name,
            graph=graph,
            text_backbone_only=definition.text_backbone_only,
            supported_modalities=tuple(definition.supported_modalities),
            excluded_subgraphs=tuple(definition.unsupported_subgraphs),
            metadata=payload["metadata"],
        )
    )


def _display_only_graph(definition: PresetDefinition) -> ModelGraph:
    """Materialize a non-executable skeleton for out-of-domain presets."""

    dtype = "unknown"
    shape = ("B", "T", "H")
    unsupported = tuple(definition.unsupported_subgraphs) or (definition.architecture,)
    operators = (
        OperatorNode(
            operator_id="input",
            op_kind="model_input",
            sequence_index=0,
            output_tensor_ids=("input.hidden",),
            ports=(OperatorPort("out0", "output", "input.hidden", dtype, shape),),
        ),
        OperatorNode(
            operator_id="unsupported-backbone",
            op_kind="unsupported_component_group",
            sequence_index=1,
            input_tensor_ids=("input.hidden",),
            output_tensor_ids=("output.hidden",),
            ports=(
                OperatorPort("in0", "input", "input.hidden", dtype, shape),
                OperatorPort("out0", "output", "output.hidden", dtype, shape),
            ),
            parameters={"components": list(unsupported), "layer_count": definition.layer_count},
            attributes={"unsupported": True},
        ),
        OperatorNode(
            operator_id="output",
            op_kind="model_output",
            sequence_index=2,
            input_tensor_ids=("output.hidden",),
            ports=(OperatorPort("in0", "input", "output.hidden", dtype, shape),),
        ),
    )
    tensors = (
        TensorValue("input.hidden", "input", producer_operator_id="input", consumer_operator_ids=("unsupported-backbone",), dtype=dtype, shape=shape),
        TensorValue("output.hidden", "output", producer_operator_id="unsupported-backbone", consumer_operator_ids=("output",), dtype=dtype, shape=shape),
    )
    return ModelGraph(
        graph_id=definition.name,
        operators=operators,
        tensors=tensors,
        executable=False,
        attributes={
            "authoritative": True,
            "architecture": definition.architecture,
            "support_level": definition.support_level,
            "unsupported_subgraphs": list(unsupported),
            "limitations": list(_metadata_limitations(definition)),
            "architecture_evidence": _architecture_evidence_metadata(definition),
            "ui": {"collapsed_groups": ["unsupported-backbone"]},
        },
    )


def model_preset_detail(preset_id: str) -> Dict[str, Any]:
    """Build the API detail object; out-of-domain entries remain display-only."""

    definition = get_model_preset(preset_id)
    return model_preset_definition_detail(definition)


def model_preset_definition_detail(definition: PresetDefinition) -> Dict[str, Any]:
    """Build API detail for any definition, preserving fail-closed OOD entries."""

    model = None
    if _metadata(definition)["generation_allowed"]:
        model = materialize_preset_definition(definition)
        graph = model["graph"]
    else:
        graph = to_primitive(_display_only_graph(definition))
    return {"preset": _metadata(definition), "model": model, "graph": graph}


__all__ = [
    "APPROXIMATION",
    "ARCHITECTURE_EVIDENCE_STATUSES",
    "CATALOG_CUTOFF_AT",
    "CATALOG_VERSION",
    "CONFIG_VERIFIED_NO_OFFICIAL_DIAGRAM",
    "CONSERVATIVE_DIAGRAM_OVERRIDES",
    "DIAGRAM_OVERRIDES",
    "DIAGRAM_VERIFIED",
    "EXACT",
    "GATED_CONFIG",
    "METADATA_ONLY_UNSUPPORTED_IR",
    "OUT_OF_DOMAIN",
    "SUPPORT_LEVELS",
    "UNPINNED_SOURCE",
    "ArchitectureEvidence",
    "UnsupportedPresetError",
    "LayerPattern",
    "LinearAttentionPattern",
    "PresetDefinition",
    "get_model_preset",
    "list_model_presets",
    "materialize_model_payload",
    "materialize_preset_definition",
    "model_preset_detail",
    "model_preset_definition_detail",
    "model_preset_metadata",
]
