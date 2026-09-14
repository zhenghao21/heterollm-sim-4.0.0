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
CATALOG_VERSION = "0.5"
CATALOG_CUTOFF_AT = "2026-08-25T00:00:00Z"

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
    "mit": ("MIT", "open_source", "allowed"),
    "qwen": ("Tongyi Qianwen Research License", "open_weight", "conditional"),
    "qwen38": ("Qwen3.8-Max License", "open_weight", "conditional"),
    "deepseek": ("DeepSeek Model License", "open_weight", "allowed"),
    "kimi_modified_mit": ("Modified MIT License", "open_weight", "conditional"),
    "kimi_k3": ("Kimi K3 License", "open_weight", "conditional"),
    "microsoft": ("Microsoft Research License", "open_weight", "prohibited"),
    "bigscience": ("BigScience RAIL 1.0", "open_weight", "allowed"),
    "falcon": ("Falcon LLM License (model-specific version)", "open_weight", "conditional"),
    "internlm": ("InternLM Model License", "open_weight", "conditional"),
    "yi": ("Yi Series Models Community License", "open_weight", "conditional"),
    "baichuan": ("Baichuan Model License", "open_weight", "conditional"),
    "glm": ("GLM-4 Model License", "open_weight", "conditional"),
    "bigcode": ("BigCode OpenRAIL-M", "open_weight", "allowed"),
    "mistral": ("Mistral AI Research License", "open_weight", "prohibited"),
    "dbrx": ("DBRX License", "open_weight", "conditional"),
    "opt": ("OPT-175B License Agreement", "open_weight", "prohibited"),
    "cc_by_sa": ("CC-BY-SA-4.0", "open_weight", "allowed"),
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
    **kwargs: Any,
) -> PresetDefinition:
    if gated_mlp is None:
        legacy_gelu_family = family in {"BLOOM", "Pythia", "OPT", "MPT"}
        legacy_gelu_family = legacy_gelu_family or (
            family == "Falcon" and not preset_id.startswith("falcon3-")
        )
        gated_mlp = not legacy_gelu_family
    return _definition(
        preset_id, name, family, scale, repo, policy,
        vocabulary_size, max_sequence_length,
        (_pattern(layers, "dense", hidden, intermediate, heads, kv_heads, gated_mlp=bool(gated_mlp)),),
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
    **kwargs: Any,
) -> PresetDefinition:
    return _definition(
        preset_id, name, family, scale, repo, policy,
        vocabulary_size, max_sequence_length,
        (_pattern(layers, "moe", hidden, intermediate, heads, kv_heads, experts, top_k),),
        **kwargs,
    )


def _qwen_hybrid_patterns(
    layers: int,
    kind: str,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    linear_value_heads: int,
    *,
    experts: int = 1,
    top_k: int = 1,
    shared_expert_intermediate_size: int = 0,
) -> Tuple[LayerPattern, ...]:
    """Build the official three-linear/one-full ordered Qwen3.5 pattern."""

    linear = LinearAttentionPattern(
        key_heads=16,
        value_heads=linear_value_heads,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_size=4,
        state_dtype="fp32",
        output_gate=True,
        gate_activation="silu",
    )
    patterns = []
    remaining = layers
    while remaining:
        linear_count = min(3, remaining)
        patterns.append(
            _pattern(
                linear_count,
                kind,
                hidden,
                intermediate,
                heads,
                kv_heads,
                experts,
                top_k,
                label="linear_attention_block",
                sequence_mixer="linear_attention",
                linear_attention=linear,
                shared_expert_intermediate_size=shared_expert_intermediate_size,
                shared_expert_gate=shared_expert_intermediate_size > 0,
                attention_head_dim=256,
            )
        )
        remaining -= linear_count
        if remaining:
            patterns.append(
                _pattern(
                    1,
                    kind,
                    hidden,
                    intermediate,
                    heads,
                    kv_heads,
                    experts,
                    top_k,
                    label="full_attention_block",
                    shared_expert_intermediate_size=shared_expert_intermediate_size,
                    shared_expert_gate=shared_expert_intermediate_size > 0,
                    attention_head_dim=256,
                )
            )
            remaining -= 1
    return tuple(patterns)


def _qwen35(
    preset_id: str,
    name: str,
    family: str,
    scale: str,
    repo: str,
    source_sha: str,
    config_hash: str,
    layers: int,
    hidden: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    linear_value_heads: int,
    *,
    experts: int = 1,
    top_k: int = 1,
    shared_expert_intermediate_size: int = 0,
    variants: Sequence[str] = (),
    vision: bool = True,
    policy: str = "apache",
) -> PresetDefinition:
    kind = "moe" if experts > 1 else "dense"
    patterns = _qwen_hybrid_patterns(
        layers,
        kind,
        hidden,
        intermediate,
        heads,
        kv_heads,
        linear_value_heads,
        experts=experts,
        top_k=top_k,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
    )
    limitations = (
        "Layer order, linear-attention state geometry, routed/shared experts, and MTP are retained; declared weight bytes remain analytical estimates.",
        "The public full-attention head_dim=256 is retained as explicit LayerSpec geometry.",
    )
    unsupported = ()
    modalities = ("text",)
    coverage = "full_language_model"
    if vision:
        modalities = ("text", "image", "video")
        unsupported = ("vision_encoder", "multimodal_projector")
        coverage = "text_backbone_only"
        limitations += (
            "The official vision encoder and multimodal projector are excluded; only the ordered text backbone is materialized.",
        )
    return _definition(
        preset_id,
        name,
        family,
        scale,
        repo,
        policy,
        248320,
        262144,
        patterns,
        support=APPROXIMATION,
        notes="V4 typed-graph hybrid text-backbone representation from the pinned official config.json.",
        architecture="qwen3_5_hybrid_transformer",
        source_sha=source_sha,
        config_hash=config_hash,
        modalities=modalities,
        supported_modalities=("text",),
        unsupported_subgraphs=unsupported,
        limitations=limitations,
        coverage=coverage,
        variants=variants,
        mtp=MTPBranchSpec(prediction_layers=1, auxiliary_head=True),
        text_backbone_only=vision,
    )


_SHARED_EXPERT_NOTE = (
    "Analytical approximation: public routed-expert dimensions are retained, but "
    "the IR cannot separately represent shared experts."
)
_MLA_NOTE = (
    "Analytical approximation: public layer and routed-expert dimensions are retained; "
    "MLA compression and shared experts are not representable in the current IR."
)
_KIMI_K2_LIMITATIONS = (
    "解析近似：Kimi K2 的 MLA 潜在 KV 几何参数 "
    "(q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, "
    "qk_rope_head_dim=64, v_head_dim=128) 会折算到当前全注意力层（Full-Attention LayerSpec）。",
    "保留公开的 Dense 前缀与路由 MoE 维度；路由打分细节仅作为元数据（Metadata Only）记录。",
)
_KIMI_K2_VISION_LIMITATION = (
    "不生成官方视觉塔、视频塔与多模态投影器（Multimodal Projector）；仅实例化文本主干。"
)
_KIMI_K3_LIMITATIONS = (
    "仅元数据：Kimi K3 的 KDA 层（text_config.linear_attn_config）超出当前序列混合器建模范围。",
    "仅元数据：AttnRes 块（attn_res_block_size=12）无法由当前 LayerSpec 表达。",
    "仅元数据：不生成官方视觉/视频塔与多模态投影器，也不会生成可运行仿真模型。",
)
_DEEPSEEK_V4_LIMITATIONS = (
    "仅元数据：DeepSeek V4 的哈希注意力（Hash Attention）、上下文压缩比例、"
    "滑动窗口注意力与特殊注意力超出当前 LayerSpec 和服务 IR 的表达范围。",
    "总参数规模保持未公开（Unspecified）；不会把官方配置或混合精度权重字节数误当作总参数量。",
)


def _kimi_k2(
    preset_id: str,
    name: str,
    family: str,
    scale: str,
    repo: str,
    source_sha: str,
    config_hash: str,
    max_sequence_length: int,
    *,
    text_backbone_only: bool = False,
) -> PresetDefinition:
    unsupported: Tuple[str, ...] = ()
    modalities: Tuple[str, ...] = ("text",)
    coverage = "full_language_model"
    limitations = _KIMI_K2_LIMITATIONS
    if text_backbone_only:
        modalities = ("text", "image", "video")
        unsupported = ("vision_encoder", "video_encoder", "multimodal_projector")
        coverage = "text_backbone_only"
        limitations = limitations + (_KIMI_K2_VISION_LIMITATION,)
    return _definition(
        preset_id,
        name,
        family,
        scale,
        repo,
        "kimi_modified_mit",
        163840,
        max_sequence_length,
        (
            _pattern(1, "dense", 7168, 18432, 64, 64, label="dense_prefix"),
            _pattern(
                60,
                "moe",
                7168,
                2048,
                64,
                64,
                384,
                8,
                label="routed_moe",
                shared_expert_intermediate_size=18432,
            ),
        ),
        support=APPROXIMATION,
        notes=(
            "官方固定配置的文本主干解析近似：61 层、隐藏维度 7168、64 个注意力/KV 头；"
            "1 层 Dense 前缀，随后 60 层 384 路由专家 + 1 共享专家、Top-8 MoE。"
        ),
        architecture="kimi_k2_mla_moe_transformer",
        source_sha=source_sha,
        config_hash=config_hash,
        modalities=modalities,
        supported_modalities=("text",),
        unsupported_subgraphs=unsupported,
        limitations=limitations,
        coverage=coverage,
        text_backbone_only=text_backbone_only,
    )


# Each row is one distinct base architecture/scale configuration.  Chat/instruct
# aliases with an identical config are deliberately excluded.
_PRESETS: Tuple[PresetDefinition, ...] = (
    # Qwen and Qwen 1.5
    _dense("qwen-1_8b", "Qwen-1.8B", "Qwen", "1.8B", "Qwen/Qwen-1_8B", 24, 2048, 11008, 16, 16, 151936, 8192, "qwen"),
    _dense("qwen-7b", "Qwen-7B", "Qwen", "7B", "Qwen/Qwen-7B", 32, 4096, 22016, 32, 32, 151936, 32768, "qwen"),
    _dense("qwen-14b", "Qwen-14B", "Qwen", "14B", "Qwen/Qwen-14B", 40, 5120, 27392, 40, 40, 152064, 8192, "qwen"),
    _dense("qwen1_5-0_5b", "Qwen1.5-0.5B", "Qwen1.5", "0.5B", "Qwen/Qwen1.5-0.5B", 24, 1024, 2816, 16, 16, 151936, 32768, "qwen"),
    _dense("qwen1_5-1_8b", "Qwen1.5-1.8B", "Qwen1.5", "1.8B", "Qwen/Qwen1.5-1.8B", 24, 2048, 5504, 16, 16, 151936, 32768, "qwen"),
    _dense("qwen1_5-4b", "Qwen1.5-4B", "Qwen1.5", "4B", "Qwen/Qwen1.5-4B", 40, 2560, 6912, 20, 20, 151936, 32768, "qwen"),
    _dense("qwen1_5-7b", "Qwen1.5-7B", "Qwen1.5", "7B", "Qwen/Qwen1.5-7B", 32, 4096, 11008, 32, 32, 151936, 32768, "qwen"),
    _dense("qwen1_5-14b", "Qwen1.5-14B", "Qwen1.5", "14B", "Qwen/Qwen1.5-14B", 40, 5120, 13696, 40, 40, 152064, 32768, "qwen"),
    _dense("qwen1_5-32b", "Qwen1.5-32B", "Qwen1.5", "32B", "Qwen/Qwen1.5-32B", 64, 5120, 27392, 40, 8, 152064, 32768, "qwen"),
    _dense("qwen1_5-72b", "Qwen1.5-72B", "Qwen1.5", "72B", "Qwen/Qwen1.5-72B", 80, 8192, 24576, 64, 64, 152064, 32768, "qwen"),
    _dense("qwen1_5-110b", "Qwen1.5-110B", "Qwen1.5", "110B", "Qwen/Qwen1.5-110B", 80, 8192, 49152, 64, 8, 152064, 32768, "qwen"),
    # Qwen2 / Qwen2.5 / Qwen3
    _moe("qwen2-57b-a14b", "Qwen2-57B-A14B", "Qwen2", "57B/A14B", "Qwen/Qwen2-57B-A14B", 28, 3584, 2560, 28, 4, 64, 8, 151936, 131072, support=APPROXIMATION, notes=_SHARED_EXPERT_NOTE),
    _dense("qwen2_5-0_5b", "Qwen2.5-0.5B", "Qwen2.5", "0.5B", "Qwen/Qwen2.5-0.5B", 24, 896, 4864, 14, 2, 151936, 32768),
    _dense("qwen2_5-1_5b", "Qwen2.5-1.5B", "Qwen2.5", "1.5B", "Qwen/Qwen2.5-1.5B", 28, 1536, 8960, 12, 2, 151936, 131072),
    _dense("qwen2_5-3b", "Qwen2.5-3B", "Qwen2.5", "3B", "Qwen/Qwen2.5-3B", 36, 2048, 11008, 16, 2, 151936, 32768, "qwen"),
    _dense("qwen2_5-7b", "Qwen2.5-7B", "Qwen2.5", "7B", "Qwen/Qwen2.5-7B", 28, 3584, 18944, 28, 4, 152064, 131072),
    _dense("qwen2_5-14b", "Qwen2.5-14B", "Qwen2.5", "14B", "Qwen/Qwen2.5-14B", 48, 5120, 13824, 40, 8, 152064, 131072),
    _dense("qwen2_5-32b", "Qwen2.5-32B", "Qwen2.5", "32B", "Qwen/Qwen2.5-32B", 64, 5120, 27648, 40, 8, 152064, 131072),
    _dense("qwen2_5-72b", "Qwen2.5-72B", "Qwen2.5", "72B", "Qwen/Qwen2.5-72B", 80, 8192, 29568, 64, 8, 152064, 131072, "qwen"),
    _dense("qwen3-0_6b", "Qwen3-0.6B", "Qwen3", "0.6B", "Qwen/Qwen3-0.6B-Base", 28, 1024, 3072, 16, 8, 151936, 32768),
    _dense("qwen3-1_7b", "Qwen3-1.7B", "Qwen3", "1.7B", "Qwen/Qwen3-1.7B-Base", 28, 2048, 6144, 16, 8, 151936, 32768),
    _dense("qwen3-4b", "Qwen3-4B", "Qwen3", "4B", "Qwen/Qwen3-4B-Base", 36, 2560, 9728, 32, 8, 151936, 32768),
    _dense("qwen3-8b", "Qwen3-8B", "Qwen3", "8B", "Qwen/Qwen3-8B-Base", 36, 4096, 12288, 32, 8, 151936, 32768),
    _dense("qwen3-14b", "Qwen3-14B", "Qwen3", "14B", "Qwen/Qwen3-14B-Base", 40, 5120, 17408, 40, 8, 151936, 32768),
    _dense("qwen3-32b", "Qwen3-32B", "Qwen3", "32B", "Qwen/Qwen3-32B", 64, 5120, 25600, 64, 8, 151936, 40960),
    _moe("qwen3-30b-a3b", "Qwen3-30B-A3B", "Qwen3", "30B/A3B", "Qwen/Qwen3-30B-A3B-Base", 48, 2048, 768, 32, 4, 128, 8, 151936, 32768, support=APPROXIMATION, notes=_SHARED_EXPERT_NOTE),
    _moe("qwen3-235b-a22b", "Qwen3-235B-A22B", "Qwen3", "235B/A22B", "Qwen/Qwen3-235B-A22B", 94, 4096, 1536, 64, 4, 128, 8, 151936, 40960, support=APPROXIMATION, notes=_SHARED_EXPERT_NOTE),
    _moe("qwen3-coder-480b-a35b", "Qwen3-Coder-480B-A35B", "Qwen3-Coder", "480B/A35B", "Qwen/Qwen3-Coder-480B-A35B-Instruct", 62, 6144, 2560, 96, 8, 160, 8, 151936, 262144, support=APPROXIMATION, notes=_SHARED_EXPERT_NOTE),
    _definition("qwen3-next-80b-a3b", "Qwen3-Next-80B-A3B", "Qwen3-Next", "80B/A3B", "Qwen/Qwen3-Next-80B-A3B-Instruct", "apache", 151936, 262144, (), support=OUT_OF_DOMAIN, architecture="hybrid_gated_delta_net_transformer", config_hash="2d483c7cabad7c8704478ed4038fa7e7b2eff840bc00a118eccbe38e2b488303", model_kind_override="moe", layer_count_override=48, coverage="metadata_only", unsupported_subgraphs=("gated_delta_net",), limitations=("Hybrid Gated DeltaNet/attention blocks are not representable in the current LayerSpec.",), notes="Metadata-only pinned canonical config; hybrid mixers are not coerced to full attention."),
    # Qwen3.5 / Qwen3.8 official canonical configs. Base, post-trained,
    # quantized, GGUF, MLX and mirrors with the same architecture belong in
    # ``variants`` rather than becoming duplicate presets.
    _qwen35("qwen3_5-0_8b", "Qwen3.5-0.8B", "Qwen3.5", "0.8B", "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68", "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204", 24, 1024, 3584, 8, 2, 16, variants=("Qwen/Qwen3.5-0.8B",)),
    _qwen35("qwen3_5-2b", "Qwen3.5-2B", "Qwen3.5", "2B", "Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c", "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4", 24, 2048, 6144, 8, 2, 16, variants=("Qwen/Qwen3.5-2B",)),
    _qwen35("qwen3_5-4b", "Qwen3.5-4B", "Qwen3.5", "4B", "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b", "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670", 32, 2560, 9216, 16, 4, 32, variants=("Qwen/Qwen3.5-4B",)),
    _qwen35("qwen3_5-9b", "Qwen3.5-9B", "Qwen3.5", "9B", "Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404", "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05", 32, 4096, 12288, 16, 4, 32, variants=("Qwen/Qwen3.5-9B",)),
    _qwen35("qwen3_5-27b", "Qwen3.5-27B", "Qwen3.5", "27B", "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c", 64, 5120, 17408, 24, 4, 48, variants=("Qwen/Qwen3.5-27B-FP8", "Qwen/Qwen3.5-27B-GPTQ-Int4")),
    _qwen35("qwen3_5-35b-a3b", "Qwen3.5-35B-A3B", "Qwen3.5", "35B/A3B", "Qwen/Qwen3.5-35B-A3B-Base", "0f0813072d2358973511097385626f21fcb6d422", "5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944", 40, 2048, 512, 16, 2, 32, experts=256, top_k=8, shared_expert_intermediate_size=512, variants=("Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B-FP8", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4")),
    _qwen35("qwen3_5-122b-a10b", "Qwen3.5-122B-A10B", "Qwen3.5", "122B/A10B", "Qwen/Qwen3.5-122B-A10B", "dc4d348443bc740c68e2d77492492c11606384d5", "af07d0423658865afb52bd0d6fd3c2cc45d988425b76193467fd19129655cecd", 48, 3072, 1024, 32, 2, 64, experts=256, top_k=8, shared_expert_intermediate_size=1024, variants=("Qwen/Qwen3.5-122B-A10B-FP8", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")),
    _qwen35("qwen3_5-397b-a17b", "Qwen3.5-397B-A17B", "Qwen3.5", "397B/A17B", "Qwen/Qwen3.5-397B-A17B", "8472618112abcbd45acbcdc58436aff4233c23f7", "3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da", 60, 4096, 1024, 32, 2, 64, experts=512, top_k=10, shared_expert_intermediate_size=1024, variants=("Qwen/Qwen3.5-397B-A17B-FP8", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4")),
    _qwen35("qwen3_8-27b", "Qwen3.8-27B", "Qwen3.8", "27B", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab", 64, 5120, 17408, 24, 4, 48, variants=("Qwen/Qwen3.8-27B-FP8",)),
    _qwen35("qwen3_8-2_4t-a95b", "Qwen3.8-2.4T-A95B", "Qwen3.8", "2.4T/A95B", "Qwen/Qwen3.8-2.4T-A95B", "207bd685a7e3696cfaff12ded7c6a7ea0f88c996", "4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987", 92, 8192, 2048, 64, 4, 128, experts=512, top_k=10, shared_expert_intermediate_size=2048, variants=("Qwen/Qwen3.8-2.4T-A95B-FP8",), vision=False, policy="qwen38"),

    # Kimi official configs. K2 text backbones are represented as transparent
    # analytical MLA/MoE approximations; K3 remains fail-closed metadata.
    _kimi_k2("kimi-k2-instruct", "Kimi K2-Instruct", "Kimi K2", "1T/A32B", "moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50", "8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439", 131072),
    _kimi_k2("kimi-k2-instruct-0905", "Kimi K2-Instruct-0905", "Kimi K2", "1T/A32B", "moonshotai/Kimi-K2-Instruct-0905", "ac6c49f04883bd0a0598b790693a72061c676629", "4cd10b0f3cb1c1dfdbfcdd2f303057500f570b24f671ce5eb7650b6a511807cc", 262144),
    _kimi_k2("kimi-k2_5", "Kimi-K2.5", "Kimi K2.5", "1T/A32B", "moonshotai/Kimi-K2.5", "4d01dfe0332d63057c186e0b262165819efb6611", "acd5bb01a16f64b309599cd6ed196be056f613c99d6bc9300692b82cd10882f6", 262144, text_backbone_only=True),
    _definition("kimi-k3", "Kimi-K3", "Kimi K3", "unspecified", "moonshotai/Kimi-K3", "kimi_k3", 163840, 1048576, (), support=OUT_OF_DOMAIN, architecture="kimi_k3_kda_attnres_multimodal", source_sha="a590ce090cb049c93a33dfe8c208ec652aa20503", config_hash="9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213", model_kind_override="moe", layer_count_override=93, modalities=("text", "image", "video"), supported_modalities=("text",), unsupported_subgraphs=("kda_linear_attention", "attention_residual_blocks", "vision_encoder", "video_encoder", "multimodal_projector"), limitations=_KIMI_K3_LIMITATIONS, coverage="metadata_only", notes="官方固定配置摘要：93 层、隐藏维度 7168、96 个注意力/KV 头、896 个路由专家、Top-16、2 个共享专家；69 个 KDA 层与 24 个全注意力层，原生图像/视频及 1M 上下文。MXFP4 仅覆盖部分权重，不能视为全模型统一量化。", text_backbone_only=True),

    # Mistral and Mixtral
    _dense("mistral-7b-v0_1", "Mistral-7B-v0.1", "Mistral", "7B", "mistralai/Mistral-7B-v0.1", 32, 4096, 14336, 32, 8, 32000, 32768),
    _dense("mistral-nemo-12b", "Mistral-Nemo-Base-2407", "Mistral", "12B", "mistralai/Mistral-Nemo-Base-2407", 40, 5120, 14336, 32, 8, 131072, 131072),
    _dense("codestral-22b-v0_1", "Codestral-22B-v0.1", "Codestral", "22B", "mistralai/Codestral-22B-v0.1", 56, 6144, 16384, 48, 8, 32768, 32768, "mistral"),
    _dense("mistral-small-24b", "Mistral-Small-24B-Base-2501", "Mistral", "24B", "mistralai/Mistral-Small-24B-Base-2501", 40, 5120, 32768, 32, 8, 131072, 32768),
    _dense("mistral-large-123b", "Mistral-Large-Instruct-2407", "Mistral", "123B", "mistralai/Mistral-Large-Instruct-2407", 88, 12288, 28672, 96, 8, 32768, 131072, "mistral", access="gated"),
    _dense("ministral-3b", "Ministral-3-3B-Base-2512", "Ministral", "3B", "mistralai/Ministral-3-3B-Base-2512", 26, 3072, 9216, 32, 8, 131072, 262144, "apache", support=APPROXIMATION, notes="Analytical approximation: the public text decoder config is retained; the Pixtral vision tower is outside ModelSpec."),
    _dense("ministral-8b", "Ministral-8B-Instruct-2410", "Ministral", "8B", "mistralai/Ministral-8B-Instruct-2410", 36, 4096, 12288, 32, 8, 131072, 32768, "mistral"),
    _moe("mixtral-8x7b-v0_1", "Mixtral-8x7B-v0.1", "Mixtral", "8x7B", "mistralai/Mixtral-8x7B-v0.1", 32, 4096, 14336, 32, 8, 8, 2, 32000, 32768),
    _moe("mixtral-8x22b-v0_1", "Mixtral-8x22B-v0.1", "Mixtral", "8x22B", "mistralai/Mixtral-8x22B-v0.1", 56, 6144, 16384, 48, 8, 8, 2, 32000, 65536),

    # DeepSeek: MLA/shared-expert generations are deliberately approximations.
    _dense("deepseek-coder-1_3b", "DeepSeek-Coder-1.3B-Base", "DeepSeek-Coder", "1.3B", "deepseek-ai/deepseek-coder-1.3b-base", 24, 2048, 5504, 16, 16, 32256, 16384, "deepseek"),
    _dense("deepseek-coder-6_7b", "DeepSeek-Coder-6.7B-Base", "DeepSeek-Coder", "6.7B", "deepseek-ai/deepseek-coder-6.7b-base", 32, 4096, 11008, 32, 32, 32256, 16384, "deepseek"),
    _dense("deepseek-coder-33b", "DeepSeek-Coder-33B-Base", "DeepSeek-Coder", "33B", "deepseek-ai/deepseek-coder-33b-base", 62, 7168, 19200, 56, 8, 32256, 16384, "deepseek"),
    _dense("deepseek-llm-7b", "DeepSeek-LLM-7B-Base", "DeepSeek-LLM", "7B", "deepseek-ai/deepseek-llm-7b-base", 30, 4096, 11008, 32, 32, 102400, 4096, "deepseek"),
    _dense("deepseek-llm-67b", "DeepSeek-LLM-67B-Base", "DeepSeek-LLM", "67B", "deepseek-ai/deepseek-llm-67b-base", 95, 8192, 22016, 64, 8, 102400, 4096, "deepseek"),
    _definition("deepseek-moe-16b", "DeepSeek-MoE-16B-Base", "DeepSeek-MoE", "16B/A2.8B", "deepseek-ai/deepseek-moe-16b-base", "deepseek", 102400, 4096, (_pattern(1, "dense", 2048, 10944, 16, 16, label="dense_prefix"), _pattern(27, "moe", 2048, 1408, 16, 16, 64, 6, label="routed_moe")), support=APPROXIMATION, notes=_SHARED_EXPERT_NOTE),
    _definition("deepseek-v2-lite", "DeepSeek-V2-Lite", "DeepSeek-V2", "16B/A2.4B", "deepseek-ai/DeepSeek-V2-Lite", "deepseek", 102400, 163840, (_pattern(1, "dense", 2048, 10944, 16, 16, label="dense_prefix"), _pattern(26, "moe", 2048, 1408, 16, 16, 64, 6, label="routed_moe")), support=APPROXIMATION, notes=_MLA_NOTE),
    _definition("deepseek-v2-236b", "DeepSeek-V2", "DeepSeek-V2", "236B/A21B", "deepseek-ai/DeepSeek-V2", "deepseek", 102400, 163840, (_pattern(1, "dense", 5120, 12288, 128, 128, label="dense_prefix"), _pattern(59, "moe", 5120, 1536, 128, 128, 160, 6, label="routed_moe")), support=APPROXIMATION, notes=_MLA_NOTE),
    _definition("deepseek-v3-671b", "DeepSeek-V3-Base", "DeepSeek-V3", "671B/A37B", "deepseek-ai/DeepSeek-V3-Base", "mit", 129280, 163840, (_pattern(3, "dense", 7168, 18432, 128, 128, label="dense_prefix"), _pattern(58, "moe", 7168, 2048, 128, 128, 256, 8, label="routed_moe")), support=APPROXIMATION, notes=_MLA_NOTE),
    _definition("deepseek-v4-pro-base", "DeepSeek-V4-Pro-Base", "DeepSeek-V4", "unspecified", "deepseek-ai/DeepSeek-V4-Pro-Base", "mit", 129280, 1048576, (), support=OUT_OF_DOMAIN, architecture="deepseek_v4_hash_compressed_attention_moe", source_sha="98730c030fbdbaca4950788280a35c4642b208a9", config_hash="718c188325de5ec90885a2782990b6e49fe773ef701c3aed03c55c8c21b99059", model_kind_override="moe", layer_count_override=61, coverage="metadata_only", unsupported_subgraphs=("hash_attention", "context_compression", "sliding_window_attention", "special_attention"), limitations=_DEEPSEEK_V4_LIMITATIONS, notes="官方固定配置摘要：61 层、隐藏维度 7168、128 个注意力头、1 个 KV 头、384 个路由专家 + 1 个共享专家、Top-6；BF16 主计算与 FP8 专家，含 1 个 MTP 层、3 个 Hash 层、128 滑动窗口及 1M 上下文。"),

    # Microsoft Phi
    _dense("phi-1", "Phi-1", "Phi", "1.3B", "microsoft/phi-1", 24, 2048, 8192, 32, 32, 51200, 2048, "microsoft"),
    _dense("phi-2", "Phi-2", "Phi", "2.7B", "microsoft/phi-2", 32, 2560, 10240, 32, 32, 51200, 2048, "microsoft"),
    _dense("phi-3-mini-128k", "Phi-3-Mini-128K-Instruct", "Phi-3", "3.8B", "microsoft/Phi-3-mini-128k-instruct", 32, 3072, 8192, 32, 32, 32064, 131072, "mit"),
    _dense(
        "phi-3-small-128k",
        "Phi-3-Small-128K-Instruct",
        "Phi-3",
        "7B",
        "microsoft/Phi-3-small-128k-instruct",
        32,
        4096,
        14336,
        32,
        8,
        100352,
        131072,
        "mit",
        support=OUT_OF_DOMAIN,
        architecture="phi_3_block_sparse_attention",
        unsupported_subgraphs=("block_sparse_attention",),
        limitations=(
            "Phi-3-Small uses block-sparse attention (local plus remote/vertical blocks), "
            "which is not faithfully representable by the current full_attention LayerSpec.",
        ),
        coverage="metadata_only",
        notes=(
            "Metadata-only pinned canonical config; Phi-3-Small block-sparse attention "
            "is not coerced to full attention."
        ),
    ),
    _dense("phi-3-medium-128k", "Phi-3-Medium-128K-Instruct", "Phi-3", "14B", "microsoft/Phi-3-medium-128k-instruct", 40, 5120, 17920, 40, 10, 32064, 131072, "mit"),
    _dense("phi-4", "Phi-4", "Phi-4", "14B", "microsoft/phi-4", 40, 5120, 17920, 40, 10, 100352, 16384, "mit"),
    _dense("phi-4-mini", "Phi-4-Mini-Instruct", "Phi-4", "3.8B", "microsoft/Phi-4-mini-instruct", 32, 3072, 8192, 24, 8, 200064, 131072, "mit"),

    # OLMo
    _dense("olmo-1b", "OLMo-1B", "OLMo", "1B", "allenai/OLMo-1B", 16, 2048, 8192, 16, 16, 50280, 2048),
    _dense("olmo-7b", "OLMo-7B", "OLMo", "7B", "allenai/OLMo-7B", 32, 4096, 11008, 32, 32, 50280, 2048),
    _dense("olmo2-1b", "OLMo-2-0425-1B", "OLMo2", "1B", "allenai/OLMo-2-0425-1B", 16, 2048, 8192, 16, 16, 100352, 4096),
    _dense("olmo2-7b", "OLMo-2-1124-7B", "OLMo2", "7B", "allenai/OLMo-2-1124-7B", 32, 4096, 11008, 32, 32, 100352, 4096),
    _dense("olmo2-13b", "OLMo-2-1124-13B", "OLMo2", "13B", "allenai/OLMo-2-1124-13B", 40, 5120, 13824, 40, 40, 100352, 4096),
    _dense("olmo2-32b", "OLMo-2-0325-32B", "OLMo2", "32B", "allenai/OLMo-2-0325-32B", 64, 5120, 27648, 40, 8, 100352, 4096),

    # BLOOM
    _dense("bloom-560m", "BLOOM-560M", "BLOOM", "0.56B", "bigscience/bloom-560m", 24, 1024, 4096, 16, 16, 250880, 2048, "bigscience"),
    _dense("bloom-1b1", "BLOOM-1.1B", "BLOOM", "1.1B", "bigscience/bloom-1b1", 24, 1536, 6144, 16, 16, 250880, 2048, "bigscience"),
    _dense("bloom-1b7", "BLOOM-1.7B", "BLOOM", "1.7B", "bigscience/bloom-1b7", 24, 2048, 8192, 16, 16, 250880, 4096, "bigscience"),
    _dense("bloom-3b", "BLOOM-3B", "BLOOM", "3B", "bigscience/bloom-3b", 30, 2560, 10240, 32, 32, 250880, 2048, "bigscience"),
    _dense("bloom-7b1", "BLOOM-7.1B", "BLOOM", "7.1B", "bigscience/bloom-7b1", 30, 4096, 16384, 32, 32, 250880, 2048, "bigscience"),
    _dense("bloom-176b", "BLOOM-176B", "BLOOM", "176B", "bigscience/bloom", 70, 14336, 57344, 112, 112, 250880, 2048, "bigscience"),

    # Pythia
    _dense("pythia-70m", "Pythia-70M-Deduped", "Pythia", "70M", "EleutherAI/pythia-70m-deduped", 6, 512, 2048, 8, 8, 50304, 2048),
    _dense("pythia-160m", "Pythia-160M-Deduped", "Pythia", "160M", "EleutherAI/pythia-160m-deduped", 12, 768, 3072, 12, 12, 50304, 2048),
    _dense("pythia-410m", "Pythia-410M-Deduped", "Pythia", "410M", "EleutherAI/pythia-410m-deduped", 24, 1024, 4096, 16, 16, 50304, 2048),
    _dense("pythia-1b", "Pythia-1B-Deduped", "Pythia", "1B", "EleutherAI/pythia-1b-deduped", 16, 2048, 8192, 8, 8, 50304, 2048),
    _dense("pythia-1_4b", "Pythia-1.4B-Deduped", "Pythia", "1.4B", "EleutherAI/pythia-1.4b-deduped", 24, 2048, 8192, 16, 16, 50304, 2048),
    _dense("pythia-2_8b", "Pythia-2.8B-Deduped", "Pythia", "2.8B", "EleutherAI/pythia-2.8b-deduped", 32, 2560, 10240, 32, 32, 50304, 2048),
    _dense("pythia-6_9b", "Pythia-6.9B-Deduped", "Pythia", "6.9B", "EleutherAI/pythia-6.9b-deduped", 32, 4096, 16384, 32, 32, 50432, 2048),
    _dense("pythia-12b", "Pythia-12B-Deduped", "Pythia", "12B", "EleutherAI/pythia-12b-deduped", 36, 5120, 20480, 40, 40, 50688, 2048),

    # Falcon. Falcon-H1 is listed but never coerced into transformer layers.
    _dense("falcon-rw-1b", "Falcon-RW-1B", "Falcon", "1B", "tiiuae/falcon-rw-1b", 24, 2048, 8192, 32, 32, 50304, 2048),
    _dense("falcon-7b", "Falcon-7B", "Falcon", "7B", "tiiuae/falcon-7b", 32, 4544, 18176, 71, 1, 65024, 2048),
    _dense("falcon-11b", "Falcon-11B", "Falcon", "11B", "tiiuae/falcon-11B", 60, 4096, 16384, 32, 8, 65024, 8192, "falcon"),
    _dense("falcon-40b", "Falcon-40B", "Falcon", "40B", "tiiuae/falcon-40b", 60, 8192, 32768, 128, 8, 65024, 2048),
    _dense("falcon-180b", "Falcon-180B", "Falcon", "180B", "tiiuae/falcon-180B", 80, 14848, 59392, 232, 8, 65024, 2048, "falcon", access="gated"),
    _dense("falcon3-1b", "Falcon3-1B-Base", "Falcon3", "1B", "tiiuae/Falcon3-1B-Base", 18, 2048, 8192, 8, 4, 131072, 4096, "falcon"),
    _dense("falcon3-3b", "Falcon3-3B-Base", "Falcon3", "3B", "tiiuae/Falcon3-3B-Base", 22, 3072, 9216, 12, 4, 131072, 32768, "falcon"),
    _dense("falcon3-7b", "Falcon3-7B-Base", "Falcon3", "7B", "tiiuae/Falcon3-7B-Base", 28, 3072, 23040, 12, 4, 131072, 32768, "falcon"),
    _dense("falcon3-10b", "Falcon3-10B-Base", "Falcon3", "10B", "tiiuae/Falcon3-10B-Base", 40, 3072, 23040, 12, 4, 131072, 32768, "falcon"),
    _definition("falcon-h1-34b", "Falcon-H1-34B-Base", "Falcon-H1", "34B", "tiiuae/Falcon-H1-34B-Base", "falcon", 261120, 262144, (), support=OUT_OF_DOMAIN, architecture="hybrid_mamba_transformer", config_hash="156d23cf8c31a19d1951fb530ce703441b8dba7002af7f6185d26a9d191b1a2f", model_kind_override="dense", layer_count_override=72, coverage="metadata_only", unsupported_subgraphs=("mamba2_state_space_mixer",), limitations=("Hybrid Mamba-2/attention blocks are outside the current supported mixer vocabulary.",), notes="Metadata-only pinned canonical config; Mamba-2 blocks are not coerced to attention."),

    # InternLM
    _dense("internlm-7b", "InternLM-7B", "InternLM", "7B", "internlm/internlm-7b", 32, 4096, 11008, 32, 32, 103168, 2048, "internlm"),
    _dense("internlm-20b", "InternLM-20B", "InternLM", "20B", "internlm/internlm-20b", 60, 5120, 13824, 40, 40, 103168, 4096, "internlm"),
    _dense("internlm2-1_8b", "InternLM2-1.8B", "InternLM2", "1.8B", "internlm/internlm2-1_8b", 24, 2048, 8192, 16, 8, 92544, 32768, "internlm"),
    _dense("internlm2-7b", "InternLM2-7B", "InternLM2", "7B", "internlm/internlm2-7b", 32, 4096, 14336, 32, 8, 92544, 32768, "internlm"),
    _dense("internlm2-20b", "InternLM2-20B", "InternLM2", "20B", "internlm/internlm2-20b", 48, 6144, 16384, 48, 8, 92544, 32768, "internlm"),
    _dense("internlm2_5-7b", "InternLM2.5-7B", "InternLM2.5", "7B", "internlm/internlm2_5-7b", 32, 4096, 14336, 32, 8, 92544, 262144, "internlm"),
    _dense("internlm3-8b", "InternLM3-8B-Instruct", "InternLM3", "8B", "internlm/internlm3-8b-instruct", 48, 4096, 10240, 32, 2, 128512, 32768, "internlm"),

    # Yi, Baichuan, and GLM use bespoke weight licenses.
    _dense("yi-6b", "Yi-6B", "Yi", "6B", "01-ai/Yi-6B", 32, 4096, 11008, 32, 4, 64000, 4096, "yi"),
    _dense("yi-9b", "Yi-9B", "Yi", "9B", "01-ai/Yi-9B", 48, 4096, 11008, 32, 4, 64000, 4096, "yi"),
    _dense("yi-34b", "Yi-34B", "Yi", "34B", "01-ai/Yi-34B", 60, 7168, 20480, 56, 8, 64000, 4096, "yi"),
    _dense("baichuan-7b", "Baichuan-7B", "Baichuan", "7B", "baichuan-inc/Baichuan-7B", 32, 4096, 11008, 32, 32, 64000, 4096, "baichuan"),
    _dense("baichuan-13b", "Baichuan-13B-Base", "Baichuan", "13B", "baichuan-inc/Baichuan-13B-Base", 40, 5120, 13696, 40, 40, 64000, 4096, "baichuan"),
    _dense("baichuan2-7b", "Baichuan2-7B-Base", "Baichuan2", "7B", "baichuan-inc/Baichuan2-7B-Base", 32, 4096, 11008, 32, 32, 125696, 4096, "baichuan"),
    _dense("baichuan2-13b", "Baichuan2-13B-Base", "Baichuan2", "13B", "baichuan-inc/Baichuan2-13B-Base", 40, 5120, 13696, 40, 40, 125696, 4096, "baichuan"),
    _dense("chatglm-6b", "ChatGLM-6B", "GLM", "6B", "THUDM/chatglm-6b", 28, 4096, 16384, 32, 32, 130528, 2048, "glm"),
    _dense("chatglm2-6b", "ChatGLM2-6B", "GLM", "6B", "THUDM/chatglm2-6b", 28, 4096, 13696, 32, 2, 65024, 32768, "glm"),
    _dense("glm4-9b", "GLM-4-9B", "GLM-4", "9B", "THUDM/glm-4-9b", 40, 4096, 13696, 32, 2, 151552, 8192, "glm"),
    _dense("glm4-32b", "GLM-4-32B-0414", "GLM-4", "32B", "THUDM/GLM-4-32B-0414", 61, 6144, 32768, 48, 2, 151552, 131072, "glm", access="requires_authentication"),

    # SmolLM
    _dense("smollm-135m", "SmolLM-135M", "SmolLM", "135M", "HuggingFaceTB/SmolLM-135M", 30, 576, 1536, 9, 3, 49152, 2048),
    _dense("smollm-360m", "SmolLM-360M", "SmolLM", "360M", "HuggingFaceTB/SmolLM-360M", 32, 960, 2560, 15, 5, 49152, 2048),
    _dense("smollm-1_7b", "SmolLM-1.7B", "SmolLM", "1.7B", "HuggingFaceTB/SmolLM-1.7B", 24, 2048, 8192, 32, 32, 49152, 2048),
    _dense("smollm2-135m", "SmolLM2-135M", "SmolLM2", "135M", "HuggingFaceTB/SmolLM2-135M", 30, 576, 1536, 9, 3, 49152, 8192),
    _dense("smollm2-360m", "SmolLM2-360M", "SmolLM2", "360M", "HuggingFaceTB/SmolLM2-360M", 32, 960, 2560, 15, 5, 49152, 8192),
    _dense("smollm2-1_7b", "SmolLM2-1.7B", "SmolLM2", "1.7B", "HuggingFaceTB/SmolLM2-1.7B", 24, 2048, 8192, 32, 32, 49152, 8192),

    # Code and long-context architectures
    _dense("starcoder2-3b", "StarCoder2-3B", "StarCoder2", "3B", "bigcode/starcoder2-3b", 30, 3072, 12288, 24, 2, 49152, 16384, "bigcode"),
    _dense("starcoder2-7b", "StarCoder2-7B", "StarCoder2", "7B", "bigcode/starcoder2-7b", 32, 4608, 18432, 36, 4, 49152, 16384, "bigcode"),
    _dense("starcoder2-15b", "StarCoder2-15B", "StarCoder2", "15B", "bigcode/starcoder2-15b", 40, 6144, 24576, 48, 4, 49152, 16384, "bigcode"),
    _dense("mpt-1b-redpajama", "MPT-1B-RedPajama-200B", "MPT", "1B", "mosaicml/mpt-1b-redpajama-200b", 24, 2048, 8192, 16, 16, 50368, 2048, access="requires_authentication"),
    _dense("mpt-7b", "MPT-7B", "MPT", "7B", "mosaicml/mpt-7b", 32, 4096, 16384, 32, 32, 50432, 2048, access="requires_authentication"),
    _dense("mpt-30b", "MPT-30B", "MPT", "30B", "mosaicml/mpt-30b", 48, 7168, 28672, 64, 8, 50432, 8192, access="requires_authentication"),

    # Additional canonical publisher configs that are publicly readable without
    # credentials. Gated Llama/Gemma/Command configs are intentionally not
    # reconstructed from mirrors; users can explicitly import them after access.
    _dense("opt-125m", "OPT-125M", "OPT", "125M", "facebook/opt-125m", 12, 768, 3072, 12, 12, 50272, 2048, "opt", config_hash="54162c9cd0377110df1859dfabebb106fc0988c8aadb88207ce1ee0d34985df6"),
    _dense("opt-1_3b", "OPT-1.3B", "OPT", "1.3B", "facebook/opt-1.3b", 24, 2048, 8192, 32, 32, 50272, 2048, "opt", config_hash="f1aaceefb7d34d346e9b0400d95fdb4970567bcbe1d4aba1068c5402099b644f"),
    _dense("opt-6_7b", "OPT-6.7B", "OPT", "6.7B", "facebook/opt-6.7b", 32, 4096, 16384, 32, 32, 50272, 2048, "opt", config_hash="05bcf525449fe484fd741ec49620e96e74992c13e46d83327165645a38666ce6"),
    _dense("opt-66b", "OPT-66B", "OPT", "66B", "facebook/opt-66b", 64, 9216, 36864, 72, 72, 50272, 2048, "opt", config_hash="d759b638611af8e08c3895fcef5669f0b575caad0252e96e6f724547a06dfa71"),
    _dense("gpt-neox-20b", "GPT-NeoX-20B", "GPT-NeoX", "20B", "EleutherAI/gpt-neox-20b", 44, 6144, 24576, 64, 64, 50432, 2048, config_hash="0c13687d6cfccbe23d0cc980a31605c180e3e93145bfc0192468f0e37cc418be"),
    _definition("gpt-j-6b", "GPT-J-6B", "GPT-J", "6B", "EleutherAI/gpt-j-6b", "apache", 50400, 2048, (), support=OUT_OF_DOMAIN, architecture="gptj_decoder", config_hash="9328fc6e157f344fbbcd8c605a876c1fe2f7441a40a00c3046f4e8f99ab619e6", model_kind_override="dense", layer_count_override=28, coverage="metadata_only", limitations=("The public config omits an explicit intermediate width; config-only import remains fail-closed.",), notes="Metadata-only pinned canonical config; no architecture defaults were invented."),
    _definition("gpt-neo-2_7b", "GPT-Neo-2.7B", "GPT-Neo", "2.7B", "EleutherAI/gpt-neo-2.7B", "mit", 50257, 2048, (), support=OUT_OF_DOMAIN, architecture="gpt_neo_local_global_decoder", config_hash="dfdbaa26ba0351210a2f3b72a1d475ebc989a29c43aa5792aec1466ed1e8ee68", model_kind_override="dense", layer_count_override=32, coverage="metadata_only", unsupported_subgraphs=("local_attention",), limitations=("Alternating local/global attention and the omitted config intermediate width are not representable without inventing defaults.",), notes="Metadata-only pinned canonical config; local attention is not coerced to full attention."),
    _dense("granite-3_3-8b", "Granite-3.3-8B-Base", "Granite", "8B", "ibm-granite/granite-3.3-8b-base", 40, 4096, 12800, 32, 8, 49152, 131072, config_hash="cc89459d265a9703c36085177ccaa645e4d0cdcd27407fc01a0543a8da311df3"),
    _dense("xglm-564m", "XGLM-564M", "XGLM", "564M", "facebook/xglm-564M", 24, 1024, 4096, 16, 16, 256008, 2048, "mit", config_hash="7104a9e867c4ee6d546360d0a6bf4284f0e9e3e1f2961fd9c6a236a27e123fdc"),
    _dense("xglm-1_7b", "XGLM-1.7B", "XGLM", "1.7B", "facebook/xglm-1.7B", 24, 2048, 8192, 16, 16, 256008, 2048, "mit", config_hash="fe61cff641e8d694bc5296034b125a48f9979b897a47f830ce777baea89f1c6a"),
    _dense("xglm-2_9b", "XGLM-2.9B", "XGLM", "2.9B", "facebook/xglm-2.9B", 48, 2048, 8192, 16, 16, 256008, 2048, "mit", config_hash="e619538e403c00fbfea67231fb0be96e60dbe0fcc384554278d0fb5450ac8935"),
    _dense("xglm-4_5b", "XGLM-4.5B", "XGLM", "4.5B", "facebook/xglm-4.5B", 48, 2048, 16384, 16, 16, 256008, 2048, "mit", config_hash="26275cc16efb0baecb62d171b07c09901bbb059d8cf7a18ce0d11297722e772f"),
    _dense("xglm-7_5b", "XGLM-7.5B", "XGLM", "7.5B", "facebook/xglm-7.5B", 32, 4096, 16384, 32, 32, 256008, 2048, "mit", config_hash="a66dfbb6e152f420d509dcc38c2264281bb49c6b3e7be6320eb5322ea6534077"),
    _dense("stablelm-alpha-3b", "StableLM-Base-Alpha-3B", "StableLM", "3B", "stabilityai/stablelm-base-alpha-3b", 16, 4096, 16384, 32, 32, 50688, 4096, "cc_by_sa", config_hash="75430948577d00d26a3a5da76e574436d18a9c877e5f22fbb7be0148a7245f30"),
    _dense("stablelm-alpha-7b", "StableLM-Base-Alpha-7B", "StableLM", "7B", "stabilityai/stablelm-base-alpha-7b", 16, 6144, 24576, 48, 48, 50432, 4096, "cc_by_sa", config_hash="c2e7a57851a68db99e1d687e1ac0d8343335db8386adf1773875da4f42fd1084"),
    _definition("falcon-mamba-7b", "Falcon-Mamba-7B", "Falcon-Mamba", "7B", "tiiuae/falcon-mamba-7b", "falcon", 65024, 0, (), support=OUT_OF_DOMAIN, architecture="mamba_decoder", config_hash="08ec4cda33510a27a3a8cd760d27086bbab0c27a3773583f17f9b61bb8016298", model_kind_override="state_space", layer_count_override=64, coverage="metadata_only", unsupported_subgraphs=("mamba_state_space_mixer",), limitations=("Pure Mamba state-space blocks are outside the transformer/linear-attention LayerSpec domain.",), notes="Metadata-only pinned canonical config; Mamba blocks are not mislabeled as attention."),

    # Large MoE systems
    _moe("dbrx-132b-a36b", "DBRX-Base", "DBRX", "132B/A36B", "databricks/dbrx-base", 40, 6144, 10752, 48, 8, 16, 4, 100352, 32768, "dbrx", access="requires_authentication"),
    _definition("snowflake-arctic-480b", "Snowflake-Arctic", "Arctic", "480B/A17B", "Snowflake/snowflake-arctic-base", "apache", 32000, 4096, (_pattern(1, "dense", 7168, 4864, 56, 56, label="dense_prefix"), _pattern(34, "moe", 7168, 4864, 56, 56, 128, 2, label="routed_moe")), support=APPROXIMATION, notes="Analytical approximation: public routed experts are represented; Arctic's residual dense path is folded into each MoE layer estimate."),
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
DIAGRAM_OVERRIDES: Mapping[str, ArchitectureEvidence] = {
    "snowflake-arctic-480b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_vendor_architecture_diagram",
        source_url=(
            "https://www.snowflake.com/en/blog/"
            "arctic-open-efficient-foundation-language-models-snowflake/"
        ),
        notes=(
            "Snowflake (2024-04-24), Fig. 2: dense transformer plus residual "
            "128x3.66B MoE MLP with Top-2 routing."
        ),
        uncertainty=(
            "The current 1-dense/34-MoE representation folds the residual dense "
            "path into each MoE layer; it remains an analytical approximation."
        ),
    ),
    "bloom-560m": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2211.05100",
        notes=(
            "BLOOM paper (2022), Fig. 5, 'The BLOOM architecture': full decoder "
            "stack with attention, MLP, residual, layer normalization, and ALiBi."
        ),
        uncertainty=(
            "ALiBi and normalization ordering are not separate fields in the "
            "current LayerSpec; the figure verifies the dense decoder core only."
        ),
    ),
    "bloom-1b1": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2211.05100",
        notes=(
            "BLOOM paper (2022), Fig. 5, 'The BLOOM architecture'; this preset "
            "uses the same family decoder topology at the pinned 1.1B scale."
        ),
        uncertainty=(
            "ALiBi and normalization ordering are not separately represented by "
            "the current LayerSpec."
        ),
    ),
    "bloom-1b7": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2211.05100",
        notes=(
            "BLOOM paper (2022), Fig. 5, 'The BLOOM architecture'; this preset "
            "uses the same family decoder topology at the pinned 1.7B scale."
        ),
        uncertainty=(
            "ALiBi and normalization ordering are not separately represented by "
            "the current LayerSpec."
        ),
    ),
    "bloom-3b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2211.05100",
        notes=(
            "BLOOM paper (2022), Fig. 5, 'The BLOOM architecture'; this preset "
            "uses the same family decoder topology at the pinned 3B scale."
        ),
        uncertainty=(
            "ALiBi and normalization ordering are not separately represented by "
            "the current LayerSpec."
        ),
    ),
    "bloom-7b1": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2211.05100",
        notes=(
            "BLOOM paper (2022), Fig. 5, 'The BLOOM architecture'; this preset "
            "uses the same family decoder topology at the pinned 7.1B scale."
        ),
        uncertainty=(
            "ALiBi and normalization ordering are not separately represented by "
            "the current LayerSpec."
        ),
    ),
    "bloom-176b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2211.05100",
        notes=(
            "BLOOM paper (2022), Fig. 5, 'The BLOOM architecture': full decoder "
            "topology for the 176B family reference."
        ),
        uncertainty=(
            "ALiBi and normalization ordering are not separate fields in the "
            "current LayerSpec; dimensions remain config-grounded."
        ),
    ),
    "deepseek-moe-16b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2401.06066",
        notes=(
            "DeepSeekMoE (2024), Fig. 2: fine-grained routed experts and shared "
            "expert isolation; the config remains authoritative for the 1+27 layer schedule."
        ),
        uncertainty=(
            "The current IR folds the separate shared expert into the analytical "
            "MoE estimate; the figure verifies the MoE primitive, not every dimension."
        ),
    ),
    "deepseek-v2-lite": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2405.04434",
        notes=(
            "DeepSeek-V2 (2024-05), Fig. 2/3: Multi-head Latent Attention (MLA) "
            "combined with DeepSeekMoE and its KV latent-compression path."
        ),
        uncertainty=(
            "The current full-attention/shared-expert handling is an analytical "
            "approximation; MLA compression is not representable in LayerSpec."
        ),
    ),
    "deepseek-v2-236b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2405.04434",
        notes=(
            "DeepSeek-V2 (2024-05), Fig. 2/3: MLA plus DeepSeekMoE; the pinned "
            "config supplies this 236B variant's layer and expert dimensions."
        ),
        uncertainty=(
            "The current full-attention/shared-expert handling is an analytical "
            "approximation; MLA compression is not representable in LayerSpec."
        ),
    ),
    "deepseek-v3-671b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2412.19437",
        notes=(
            "DeepSeek-V3 (2024-12), Fig. 2/3: MLA, DeepSeekMoE, and the explicit "
            "multi-token prediction (MTP) component."
        ),
        uncertainty=(
            "The current graph omits MLA internals, the shared expert, and the MTP "
            "auxiliary prediction graph; the bundled layer schedule is analytical."
        ),
    ),
    "falcon-7b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2311.16867",
        notes=(
            "Falcon Series (2023-11), Fig. 5, §3.4: parallel attention and MLP "
            "paths with the report's MQA/GQA discussion."
        ),
        uncertainty=(
            "The generic dense LayerSpec does not encode Falcon's parallel rather "
            "than serial attention/MLP ordering."
        ),
    ),
    "falcon-40b": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2311.16867",
        notes=(
            "Falcon Series (2023-11), Fig. 5, §3.4: parallel attention and MLP "
            "paths; Table 16 covers the 40B model shape."
        ),
        uncertainty=(
            "The generic dense LayerSpec does not encode Falcon's parallel rather "
            "than serial attention/MLP ordering."
        ),
    ),
    "mistral-7b-v0_1": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2310.06825",
        notes=(
            "Mistral 7B (2023-10), Fig. 1–3: Sliding Window Attention and the "
            "rolling-cache/prefill behavior; the config declares sliding_window=4096."
        ),
        uncertainty=(
            "The current full_attention mixer does not faithfully represent Sliding "
            "Window Attention; this evidence is limited to the 7B preset."
        ),
    ),
    "mixtral-8x7b-v0_1": ArchitectureEvidence(
        status=DIAGRAM_VERIFIED,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2401.04088",
        notes=(
            "Mixtral (2024-01), Fig. 1: each token is routed to 2 of 8 experts "
            "and their weighted outputs are combined."
        ),
        uncertainty=(
            "The figure covers Mixtral-8x7B only; the pinned config supplies exact "
            "dimensions, and Mixtral-8x22B remains config-only."
        ),
    ),
}


# A reviewed diagram can strengthen provenance without making an unsupported
# or gated preset executable.  These entries deliberately retain the safer
# status and are resolved before any generic config-only fallback.
CONSERVATIVE_DIAGRAM_OVERRIDES: Mapping[str, ArchitectureEvidence] = {
    "falcon-180b": ArchitectureEvidence(
        status=GATED_CONFIG,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2311.16867",
        notes=(
            "Falcon Series (2023-11), Fig. 5 is corroborative for the parallel "
            "attention/MLP family topology; the Falcon-180B config remains gated."
        ),
        uncertainty=(
            "Anonymous upstream config access is gated, so this figure cannot "
            "upgrade the preset beyond gated_config."
        ),
    ),
    "falcon-h1-34b": ArchitectureEvidence(
        status=METADATA_ONLY_UNSUPPORTED_IR,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2507.22448",
        notes=(
            "Falcon-H1 (2025-07), Fig. 1: parallel SSM and attention outputs are "
            "concatenated and projected; the mixer is outside the current IR."
        ),
        uncertainty=(
            "The Mamba-2/state-space mixer is outside the current LayerSpec; the "
            "diagram confirms the fail-closed metadata-only status."
        ),
    ),
    "kimi-k3": ArchitectureEvidence(
        status=METADATA_ONLY_UNSUPPORTED_IR,
        source_type="official_technical_report_diagram",
        source_url="https://arxiv.org/html/2607.24653",
        notes=(
            "Kimi K3 (2026-07), Fig. 2: KDA, Gated-MLA, LatentMoE, AttnRes, and "
            "the native vision path are shown; these subgraphs are outside the current IR."
        ),
        uncertainty=(
            "KDA, attention-residual blocks, LatentMoE internals, and multimodal "
            "subgraphs are outside the current IR; no executable topology is generated."
        ),
    ),
    "phi-3-small-128k": ArchitectureEvidence(
        status=METADATA_ONLY_UNSUPPORTED_IR,
        source_type="official_technical_report_figure",
        source_url="https://arxiv.org/pdf/2404.14219",
        notes=(
            "Phi-3 Technical Report (2024), Figure 1, PDF p.3: Phi-3-Small uses "
            "local and vertical/remote block-sparse attention, outside the current IR."
        ),
        uncertainty=(
            "The figure covers Phi-3-Small only; block_sparse_attention is not "
            "faithfully represented by the current full_attention LayerSpec."
        ),
    ),
    "qwen3-next-80b-a3b": ArchitectureEvidence(
        status=METADATA_ONLY_UNSUPPORTED_IR,
        source_type="official_pinned_model_card_architecture_diagram",
        source_url=(
            "https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/"
            "blob/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7/README.md"
        ),
        notes=(
            "Official README (2025-09-09), embedded 'Qwen3-Next Model Architecture' "
            "diagram: 12 x [3 x (Gated DeltaNet -> MoE), 1 x (Gated Attention -> MoE)]; "
            "Gated DeltaNet is outside the current IR."
        ),
        uncertainty=(
            "The diagram confirms the topology, but Gated DeltaNet/Delta Rule is "
            "outside the current LayerSpec mixer vocabulary; keep metadata-only."
        ),
    ),
}


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
