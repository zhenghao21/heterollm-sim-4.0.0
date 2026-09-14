"""Versioned, dependency-free input IR for the simulator.

The objects in this module describe intent.  They deliberately do not contain
runtime state or elapsed-time estimates; those belong to ScheduleIR and the
event engine contracts in :mod:`heterollm_sim.contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .precision import dtype_bits, weight_storage_bits
from .schema_v4 import AUTHORING_SCHEMA_VERSION

if TYPE_CHECKING:
    from .schema_v1 import ModelGraph, OperatorNode, TensorValue


SCHEMA_VERSION = AUTHORING_SCHEMA_VERSION


# Component-kind vocabulary shared by topology and planning.  HBF is NAND
# flash and must never be folded into the HBM/DRAM active-memory class.  The
# normalization below only canonicalizes punctuation and equivalent spelling.
ACTIVE_MEMORY_COMPONENT_KINDS = frozenset(
    {
        "hbm",
        "hbm_stack",
        "dram",
        "ddr",
        "ddr_memory",
        "cxl_memory",
        "host_memory",
        "memory",
        "sram",
        "shared_memory",
    }
)
OFFLOAD_STORAGE_COMPONENT_KINDS = frozenset(
    {"hbf", "ssd", "high_io_ssd"}
)
STORAGE_COMPONENT_KINDS = (
    ACTIVE_MEMORY_COMPONENT_KINDS | OFFLOAD_STORAGE_COMPONENT_KINDS
)


def normalize_component_kind(value: str) -> str:
    """Return the canonical V4 spelling for a component kind."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "high_i_o_ssd": "high_io_ssd",
        "highio_ssd": "high_io_ssd",
        "high_i/o_ssd": "high_io_ssd",
    }
    return aliases.get(normalized, normalized)


def _require_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must not be empty".format(field_name))


def _require_schema_version(value: str, field_name: str = "schema_version") -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(
            "{} must be exactly {}; got {}".format(
                field_name, SCHEMA_VERSION, value
            )
        )


def _require_int(value: int, field_name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(field_name, minimum))


def _require_number(value: float, field_name: str, *, minimum: float = 0.0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < minimum
    ):
        raise ValueError("{} must be a number >= {}".format(field_name, minimum))


def _require_optional_name(value: Optional[str], field_name: str) -> None:
    if value is not None:
        _require_name(value, field_name)


def _require_mapping(value: Mapping[str, Any], field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(field_name))


def _matrix_storage_bytes(
    rows: int,
    columns: int,
    dtype_name: str,
    quantization: Optional[str] = None,
) -> int:
    bits = rows * columns * weight_storage_bits(
        dtype_name,
        quantization,
        unsupported_dtype_message=(
            "不支持的数据类型 {}，无法推导 MTP 权重 logical_bytes".format(
                dtype_name
            )
        ),
        unsupported_quantization_message=(
            "不支持的权重量化 {}，无法推导 MTP 权重 logical_bytes".format(
                quantization
            )
        ),
    )
    return (bits + 7) // 8


def _require_tuple(value: Tuple[Any, ...], field_name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError("{} must be a tuple".format(field_name))


@dataclass(frozen=True)
class LinearAttentionSpec:
    """Public geometry for a recurrent/scan based linear sequence mixer.

    The state matrix is represented per value head.  When key and value head
    counts differ, value heads are grouped over key heads; consequently the
    value-head count must be divisible by the key-head count.  The local
    convolution state covers the projected Q/K/V channels and stores
    ``conv_kernel_size - 1`` historical elements per channel.
    """

    key_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_size: int = 1
    state_dtype: str = "fp32"
    output_gate: bool = True
    gate_activation: str = "silu"

    def __post_init__(self) -> None:
        _require_int(self.key_heads, "key_heads", minimum=1)
        _require_int(self.value_heads, "value_heads", minimum=1)
        _require_int(self.key_head_dim, "key_head_dim", minimum=1)
        _require_int(self.value_head_dim, "value_head_dim", minimum=1)
        _require_int(self.conv_kernel_size, "conv_kernel_size", minimum=1)
        _require_name(self.state_dtype, "state_dtype")
        _require_name(self.gate_activation, "gate_activation")
        if not isinstance(self.output_gate, bool):
            raise ValueError("output_gate must be boolean")
        if self.value_heads % self.key_heads:
            raise ValueError("value_heads must be divisible by key_heads")

    @property
    def query_width(self) -> int:
        return self.key_heads * self.key_head_dim

    @property
    def key_width(self) -> int:
        return self.key_heads * self.key_head_dim

    @property
    def value_width(self) -> int:
        return self.value_heads * self.value_head_dim

    @property
    def recurrent_state_elements(self) -> int:
        return self.value_heads * self.key_head_dim * self.value_head_dim

    @property
    def convolution_state_elements(self) -> int:
        projected_width = self.query_width + self.key_width + self.value_width
        return projected_width * (self.conv_kernel_size - 1)


@dataclass(frozen=True)
class LayerSpec:
    """One ordered text-backbone block.

    ``kind`` describes the feed-forward path (``dense`` or ``moe``) and is
    orthogonal to ``sequence_mixer`` (full or linear attention).
    """

    layer_id: str
    kind: str
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int = 0
    attention_head_dim: int = 0
    sequence_mixer: str = "full_attention"
    linear_attention: Optional[LinearAttentionSpec] = None
    num_experts: int = 1
    experts_per_token: int = 1
    shared_expert_intermediate_size: int = 0
    shared_expert_gate: bool = False
    dtype: str = "fp16"
    quantization: Optional[str] = None
    gated_mlp: bool = True
    weight_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.layer_id, "layer_id")
        _require_name(self.kind, "kind")
        _require_name(self.dtype, "dtype")
        _require_name(self.sequence_mixer, "sequence_mixer")
        _require_schema_version(self.schema_version)
        _require_mapping(self.metadata, "metadata")
        _require_int(self.hidden_size, "hidden_size", minimum=1)
        _require_int(self.intermediate_size, "intermediate_size", minimum=1)
        _require_int(self.attention_heads, "attention_heads", minimum=1)
        _require_int(self.kv_heads, "kv_heads")
        _require_int(self.attention_head_dim, "attention_head_dim")
        _require_int(self.num_experts, "num_experts", minimum=1)
        _require_int(self.experts_per_token, "experts_per_token", minimum=1)
        _require_int(
            self.shared_expert_intermediate_size,
            "shared_expert_intermediate_size",
        )
        _require_int(self.weight_bytes, "weight_bytes")
        _require_optional_name(self.quantization, "quantization")
        if not isinstance(self.shared_expert_gate, bool):
            raise ValueError("shared_expert_gate must be boolean")
        if not isinstance(self.gated_mlp, bool):
            raise ValueError("gated_mlp must be boolean")
        if self.kv_heads < 0 or self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads must be in [0, attention_heads]")
        if self.attention_head_dim < 0:
            raise ValueError("attention_head_dim must be non-negative")
        if not 1 <= self.experts_per_token <= self.num_experts:
            raise ValueError("experts_per_token must be in [1, num_experts]")
        normalized_kind = self.kind.strip().lower().replace("-", "_")
        if normalized_kind not in {"dense", "moe"}:
            raise ValueError("kind must be dense or moe")
        normalized_mixer = self.sequence_mixer.strip().lower().replace("-", "_")
        if normalized_mixer not in {"full_attention", "linear_attention"}:
            raise ValueError(
                "sequence_mixer must be full_attention or linear_attention"
            )
        if normalized_mixer == "linear_attention":
            if not isinstance(self.linear_attention, LinearAttentionSpec):
                raise ValueError(
                    "linear_attention mixer requires a LinearAttentionSpec"
                )
        elif self.linear_attention is not None:
            raise ValueError(
                "linear_attention geometry is only valid for linear_attention mixer"
            )
        if normalized_kind == "dense" and (
            self.num_experts != 1 or self.experts_per_token != 1
        ):
            raise ValueError("dense layers must use one expert")
        if normalized_kind == "moe" and self.num_experts < 2:
            raise ValueError("MoE layers must define at least two experts")
        if normalized_kind == "dense" and self.shared_expert_intermediate_size:
            raise ValueError("dense layers cannot define a shared expert")
        if self.shared_expert_gate and not self.shared_expert_intermediate_size:
            raise ValueError(
                "shared_expert_gate requires shared_expert_intermediate_size"
            )

    @property
    def effective_kv_heads(self) -> int:
        return self.kv_heads or self.attention_heads

    @property
    def effective_attention_head_dim(self) -> int:
        """Return explicit public geometry or the hidden/head derivation."""

        if self.attention_head_dim:
            return self.attention_head_dim
        return int(math.ceil(self.hidden_size / float(self.attention_heads)))

    @property
    def is_moe(self) -> bool:
        return self.kind.strip().lower().replace("-", "_") == "moe"

    @property
    def is_linear_attention(self) -> bool:
        return (
            self.sequence_mixer.strip().lower().replace("-", "_")
            == "linear_attention"
        )

    @property
    def has_shared_expert(self) -> bool:
        return self.is_moe and self.shared_expert_intermediate_size > 0


@dataclass(frozen=True, kw_only=True)
class MTPBranchSpec:
    """Typed authoring input for building an explicit MTP graph branch."""

    prediction_layers: int = 0
    auxiliary_head: bool = False
    prediction_layer_weight_bytes: int = 0
    auxiliary_head_weight_bytes: int = 0


@dataclass(frozen=True)
class GraphLayerExecutionDescriptor:
    """One executable layer expanded from a compact typed layer group."""

    group_operator_id: str
    layer_id: str
    repeat_index: int
    layer: LayerSpec


@dataclass(frozen=True)
class MTPExecutionDescriptor:
    """Read-only graph-native descriptor for one typed MTP operator."""

    operator: "OperatorNode"
    input_tensor: "TensorValue"
    output_tensor: "TensorValue"
    weight_tensor: "TensorValue"
    prediction_index: Optional[int]
    hidden_size: int
    vocabulary_size: int
    weight_bytes: int


@dataclass(frozen=True)
class ModelGraphExecutionView:
    """Validated immutable graph view for mapping and planning consumers."""

    operators: Tuple["OperatorNode", ...]
    tensors: Tuple["TensorValue", ...]
    layer_instances: Tuple[GraphLayerExecutionDescriptor, ...]
    mtp_descriptors: Tuple[MTPExecutionDescriptor, ...]
    architecture: str
    vocabulary_size: int
    max_sequence_length: int
    embedding_weight_bytes: int
    output_weight_bytes: int


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """V4 graph-native model contract.

    ``graph`` is the sole executable definition.  Ordered layer cost geometry
    and typed MTP descriptors are derived through ``model_graph_execution_view``;
    they are deliberately not mirrored as constructor fields.
    """

    name: str
    graph: "ModelGraph"
    text_backbone_only: bool = True
    supported_modalities: Tuple[str, ...] = ("text",)
    excluded_subgraphs: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.name, "name")
        _require_schema_version(self.schema_version)
        _require_mapping(self.metadata, "metadata")
        _require_tuple(self.supported_modalities, "supported_modalities")
        _require_tuple(self.excluded_subgraphs, "excluded_subgraphs")
        from .schema_v1 import ModelGraph

        if not isinstance(self.graph, ModelGraph):
            raise ValueError("graph must be a ModelGraph")
        if not self.graph.executable:
            raise ValueError("不可执行的模型图不能用于映射或仿真")
        execution_view = self._execution_view
        if not execution_view.layer_instances:
            raise ValueError("model.graph must contain executable layer_group instances")
        layer_ids = [item.layer_id for item in execution_view.layer_instances]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("layer_id values must be unique within a model")
        has_mtp_aux_head = any(
            item.prediction_index is None
            for item in execution_view.mtp_descriptors
        )
        if has_mtp_aux_head and execution_view.vocabulary_size <= 0:
            raise ValueError("mtp_aux_head requires vocabulary_size")
        if not isinstance(self.text_backbone_only, bool):
            raise ValueError("text_backbone_only must be boolean")
        if not self.supported_modalities:
            raise ValueError("supported_modalities must not be empty")
        for modality in self.supported_modalities:
            _require_name(modality, "supported_modalities value")
        for subgraph in self.excluded_subgraphs:
            _require_name(subgraph, "excluded_subgraphs value")
        normalized_modalities = tuple(
            str(item).strip().lower().replace("-", "_")
            for item in self.supported_modalities
        )
        if len(normalized_modalities) != len(set(normalized_modalities)):
            raise ValueError("supported_modalities values must be unique")
        if len(self.excluded_subgraphs) != len(set(self.excluded_subgraphs)):
            raise ValueError("excluded_subgraphs values must be unique")
        object.__setattr__(self, "supported_modalities", normalized_modalities)

    @cached_property
    def _execution_view(self) -> ModelGraphExecutionView:
        """Validate and project this immutable model graph once per model.

        ``ModelSpec`` and ``ModelGraph`` are frozen value objects.  Mapping,
        planning, and reporting read the same graph-derived view repeatedly;
        rebuilding the lossless coverage projection for every property access
        adds substantial pure-Python work without changing execution state.
        ``cached_property`` keeps the derived value outside dataclass fields,
        so canonical serialization and equality remain unchanged.
        """

        return model_graph_execution_view(
            self.graph,
            schema_version=self.schema_version,
        )

    @property
    def num_layers(self) -> int:
        return len(self._execution_view.layer_instances)

    @property
    def architecture(self) -> str:
        """Return graph-declared architecture metadata (read-only)."""

        return self._execution_view.architecture

    @property
    def vocabulary_size(self) -> int:
        """Return the vocabulary resolved from graph symbols/operators."""

        return self._execution_view.vocabulary_size

    @property
    def max_sequence_length(self) -> int:
        """Return the graph-native input sequence bound."""

        return self._execution_view.max_sequence_length

    @property
    def embedding_weight_bytes(self) -> int:
        """Return embedding storage declared or derivable from its tensor."""

        return self._execution_view.embedding_weight_bytes

    @property
    def output_weight_bytes(self) -> int:
        """Return untied LM-head storage, or zero when tied to embeddings."""

        return self._execution_view.output_weight_bytes

    @property
    def total_declared_weight_bytes(self) -> int:
        execution_view = self._execution_view
        layer_bytes = sum(
            item.layer.weight_bytes for item in execution_view.layer_instances
        )
        mtp_bytes = sum(item.weight_bytes for item in execution_view.mtp_descriptors)
        return (
            execution_view.embedding_weight_bytes
            + execution_view.output_weight_bytes
            + layer_bytes
            + mtp_bytes
        )


def _layer_graph_template(layer: LayerSpec) -> Dict[str, Any]:
    linear = None
    if layer.linear_attention is not None:
        linear = {
            "key_heads": layer.linear_attention.key_heads,
            "value_heads": layer.linear_attention.value_heads,
            "key_head_dim": layer.linear_attention.key_head_dim,
            "value_head_dim": layer.linear_attention.value_head_dim,
            "conv_kernel_size": layer.linear_attention.conv_kernel_size,
            "state_dtype": layer.linear_attention.state_dtype,
            "output_gate": layer.linear_attention.output_gate,
            "gate_activation": layer.linear_attention.gate_activation,
        }
    return {
        "kind": layer.kind,
        "hidden_size": layer.hidden_size,
        "intermediate_size": layer.intermediate_size,
        "attention_heads": layer.attention_heads,
        "kv_heads": layer.kv_heads,
        "attention_head_dim": layer.attention_head_dim,
        "sequence_mixer": layer.sequence_mixer,
        "linear_attention": linear,
        "num_experts": layer.num_experts,
        "experts_per_token": layer.experts_per_token,
        "shared_expert_intermediate_size": layer.shared_expert_intermediate_size,
        "shared_expert_gate": layer.shared_expert_gate,
        "gated_mlp": layer.gated_mlp,
        "dtype": layer.dtype,
        "quantization": layer.quantization,
        "weight_bytes": layer.weight_bytes,
    }


def _layer_groups(layers: Sequence[LayerSpec]) -> Tuple[Tuple[LayerSpec, ...], ...]:
    groups: List[List[LayerSpec]] = []
    previous: Optional[Dict[str, Any]] = None
    for layer in layers:
        template = _layer_graph_template(layer)
        pattern_index = layer.metadata.get("pattern_index")
        key = {"template": template, "pattern_index": pattern_index}
        if groups and key == previous:
            groups[-1].append(layer)
        else:
            groups.append([layer])
            previous = key
    return tuple(tuple(group) for group in groups)


def build_model_graph_from_layer_specs(
    name: str,
    layers: Sequence[LayerSpec],
    *,
    architecture: str = "transformer",
    vocabulary_size: int = 0,
    max_sequence_length: int = 0,
    embedding_weight_bytes: int = 0,
    output_weight_bytes: int = 0,
    output_head_dtype: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    mtp: Optional[MTPBranchSpec] = None,
) -> "ModelGraph":
    """Deterministically materialize a compact, editable component graph.

    Consecutive identical blocks are represented by one ``layer_group`` with a
    repeat count.  Compilation still expands the projected layers and therefore
    preserves the established per-layer operator IDs used by placement plans.
    """

    from .schema_v1 import ModelGraph, OperatorNode, OperatorPort, TensorValue

    if not layers:
        raise ValueError("无法从空 layers 生成模型图")
    mtp = mtp or MTPBranchSpec()
    if not isinstance(mtp, MTPBranchSpec):
        raise ValueError("mtp must be an MTPBranchSpec")
    _require_int(vocabulary_size, "vocabulary_size")
    _require_int(max_sequence_length, "max_sequence_length")
    _require_int(embedding_weight_bytes, "embedding_weight_bytes")
    _require_int(output_weight_bytes, "output_weight_bytes")
    if output_head_dtype is not None:
        _require_name(output_head_dtype, "output_head_dtype")
    _require_int(mtp.prediction_layers, "mtp.prediction_layers")
    _require_int(
        mtp.prediction_layer_weight_bytes,
        "mtp.prediction_layer_weight_bytes",
    )
    _require_int(mtp.auxiliary_head_weight_bytes, "mtp.auxiliary_head_weight_bytes")
    if not isinstance(mtp.auxiliary_head, bool):
        raise ValueError("mtp.auxiliary_head must be boolean")
    if mtp.auxiliary_head and vocabulary_size <= 0:
        raise ValueError("mtp.auxiliary_head requires vocabulary_size")
    if mtp.prediction_layers == 0 and mtp.prediction_layer_weight_bytes:
        raise ValueError(
            "mtp.prediction_layer_weight_bytes requires mtp.prediction_layers"
        )
    if not mtp.auxiliary_head and mtp.auxiliary_head_weight_bytes:
        raise ValueError("mtp.auxiliary_head_weight_bytes requires mtp.auxiliary_head")
    operators: List[OperatorNode] = []
    tensor_records: Dict[str, Dict[str, Any]] = {}
    sequence_index = 0

    def tensor(
        tensor_id: str,
        role: str,
        dtype: str,
        shape: Tuple[Any, ...],
        *,
        producer: Optional[str] = None,
        consumer: Optional[str] = None,
        logical_bytes: Optional[int] = None,
    ) -> None:
        record = tensor_records.setdefault(tensor_id, {
            "role": role, "dtype": dtype, "shape": shape, "layout": "logical",
            "producer": producer, "consumers": [], "logical_bytes": logical_bytes,
        })
        actual = (record["dtype"], record["shape"], record["layout"])
        expected = (dtype, shape, "logical")
        if actual != expected:
            raise ValueError("张量 {} 维度不匹配：期望 {}，实际 {}".format(tensor_id, expected, actual))
        if producer is not None:
            if record["producer"] not in {None, producer}:
                raise ValueError("张量 {} 不能有多个生产组件".format(tensor_id))
            record["producer"] = producer
        if consumer is not None and consumer not in record["consumers"]:
            record["consumers"].append(consumer)

    def add_operator(
        operator_id: str,
        op_kind: str,
        *,
        inputs: Sequence[str] = (),
        outputs: Sequence[Tuple[str, str, Tuple[Any, ...]]] = (),
        weights: Sequence[Tuple[str, str, Tuple[Any, ...], Optional[int]]] = (),
        parameters: Optional[Mapping[str, Any]] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        layer_id: Optional[str] = None,
    ) -> None:
        nonlocal sequence_index
        ports: List[OperatorPort] = []
        for port_index, tensor_id in enumerate(inputs):
            record = tensor_records[tensor_id]
            tensor(tensor_id, record["role"], record["dtype"], record["shape"], consumer=operator_id)
            ports.append(OperatorPort("in{}".format(port_index), "input", tensor_id, record["dtype"], record["shape"], record["layout"]))
        output_ids = []
        for port_index, (tensor_id, dtype, shape) in enumerate(outputs):
            tensor(tensor_id, "activation", dtype, shape, producer=operator_id)
            output_ids.append(tensor_id)
            ports.append(OperatorPort("out{}".format(port_index), "output", tensor_id, dtype, shape))
        weight_ids = []
        for port_index, (tensor_id, dtype, shape, logical_bytes) in enumerate(weights):
            tensor(tensor_id, "weight", dtype, shape, consumer=operator_id, logical_bytes=logical_bytes)
            weight_ids.append(tensor_id)
            ports.append(OperatorPort("weight{}".format(port_index), "weight", tensor_id, dtype, shape))
        operators.append(OperatorNode(
            operator_id=operator_id,
            op_kind=op_kind,
            sequence_index=sequence_index,
            layer_id=layer_id,
            input_tensor_ids=tuple(inputs),
            output_tensor_ids=tuple(output_ids),
            weight_tensor_ids=tuple(weight_ids),
            ports=tuple(ports),
            parameters=dict(parameters or {}),
            attributes=dict(attributes or {}),
        ))
        sequence_index += 1

    first = layers[0]
    hidden = first.hidden_size
    dtype = first.dtype
    resolved_output_head_dtype = output_head_dtype or dtype
    add_operator(
        "input",
        "model_input",
        outputs=(("input.tokens", "int64", ("B", "T")),),
        parameters={"max_sequence_length": max_sequence_length},
    )
    tensor_records["input.tokens"]["role"] = "input"
    embedding_bytes = embedding_weight_bytes or None
    add_operator(
        "embedding", "embedding", inputs=("input.tokens",),
        outputs=(("embedding.output", dtype, ("B", "T", hidden)),),
        weights=(("embedding_weights", dtype, ("V", hidden), embedding_bytes),),
        parameters={"vocabulary_size": vocabulary_size, "hidden_size": hidden},
    )
    previous = "embedding.output"

    for group_index, group in enumerate(_layer_groups(layers)):
        base = group[0]
        template = _layer_graph_template(base)
        group_id = "block-group-{:03d}".format(group_index)
        layer_ids = [item.layer_id for item in group]
        overrides = {item.layer_id: {"metadata": dict(item.metadata), "schema_version": item.schema_version} for item in group}
        add_operator(
            group_id, "layer_group",
            parameters={
                "repeat": len(group), "layer_ids": layer_ids,
                "hidden_size": base.hidden_size, "dtype": base.dtype,
                "weight_bytes": base.weight_bytes, "overrides": overrides,
            },
            attributes={"collapsed": True, "layer_ids": layer_ids},
        )
        parent = {"parent_group_id": group_id, "layer_ids": layer_ids}
        hidden = base.hidden_size
        dtype = base.dtype
        norm1 = group_id + ".norm1.output"
        add_operator(group_id + ".norm1", "rms_norm", inputs=(previous,), outputs=((norm1, dtype, ("B", "T", hidden)),), attributes=parent)
        mixer = group_id + ".mixer.output"
        mixer_kind = "linear_attention" if base.is_linear_attention else "attention"
        add_operator(
            group_id + "." + mixer_kind, mixer_kind, inputs=(norm1,),
            outputs=((mixer, dtype, ("B", "T", hidden)),),
            weights=((group_id + ".attention_weights", dtype, (hidden, hidden), None),),
            parameters={
                "sequence_mixer": base.sequence_mixer,
                "attention_heads": base.attention_heads,
                "kv_heads": base.kv_heads,
                "attention_head_dim": base.attention_head_dim,
                "linear_attention": template["linear_attention"],
            }, attributes=parent,
        )
        residual1 = group_id + ".residual1.output"
        add_operator(group_id + ".residual1", "residual_add", inputs=(previous, mixer), outputs=((residual1, dtype, ("B", "T", hidden)),), attributes=parent)
        norm2 = group_id + ".norm2.output"
        add_operator(group_id + ".norm2", "rms_norm", inputs=(residual1,), outputs=((norm2, dtype, ("B", "T", hidden)),), attributes=parent)
        ff_output = group_id + ".ff.output"
        ff_parameters = {
            "kind": base.kind,
            "intermediate_size": base.intermediate_size,
            "num_experts": base.num_experts,
            "experts_per_token": base.experts_per_token,
            "shared_expert_intermediate_size": base.shared_expert_intermediate_size,
            "shared_expert_gate": base.shared_expert_gate,
            "gated_mlp": base.gated_mlp,
            "quantization": base.quantization,
        }
        if not base.is_moe:
            add_operator(
                group_id + ".mlp", "dense_mlp", inputs=(norm2,),
                outputs=((ff_output, dtype, ("B", "T", hidden)),),
                weights=((group_id + ".mlp_weights", dtype, (hidden, base.intermediate_size), base.weight_bytes),),
                parameters=ff_parameters, attributes=parent,
            )
        else:
            route = group_id + ".router.output"
            add_operator(
                group_id + ".router", "moe_router", inputs=(norm2,),
                outputs=((route, dtype, ("B", "T", base.experts_per_token)),),
                weights=((group_id + ".router_weights", dtype, (hidden, base.num_experts), None),),
                parameters=ff_parameters, attributes=parent,
            )
            expert = group_id + ".experts.output"
            add_operator(
                group_id + ".experts", "moe_experts", inputs=(norm2, route),
                outputs=((expert, dtype, ("B", "T", hidden)),),
                weights=((group_id + ".expert_weights", dtype, (base.num_experts, hidden, base.intermediate_size), base.weight_bytes),),
                parameters=ff_parameters, attributes=parent,
            )
            combine_inputs = [expert]
            if base.has_shared_expert:
                shared = group_id + ".shared_expert.output"
                add_operator(
                    group_id + ".shared_expert", "shared_expert", inputs=(norm2,),
                    outputs=((shared, dtype, ("B", "T", hidden)),),
                    weights=((group_id + ".shared_expert_weights", dtype, (hidden, base.shared_expert_intermediate_size), None),),
                    parameters=ff_parameters, attributes=parent,
                )
                combine_inputs.append(shared)
            add_operator(group_id + ".moe_combine", "moe_combine", inputs=tuple(combine_inputs), outputs=((ff_output, dtype, ("B", "T", hidden)),), parameters=ff_parameters, attributes=parent)
        output = group_id + ".output"
        add_operator(group_id + ".residual2", "residual_add", inputs=(residual1, ff_output), outputs=((output, dtype, ("B", "T", hidden)),), attributes=parent)
        previous = output

    mtp_branch_source = previous
    add_operator("final_norm", "rms_norm", inputs=(previous,), outputs=(("final_norm.output", dtype, ("B", "T", hidden)),))
    output_head_bytes = output_weight_bytes or None
    add_operator(
        "lm_head", "lm_head", inputs=("final_norm.output",),
        outputs=(("logits", resolved_output_head_dtype, ("B", "T", "V")),),
        weights=(("lm_head_weights", dtype, (hidden, "V"), output_head_bytes),),
        parameters={"vocabulary_size": vocabulary_size},
    )
    add_operator("output", "model_output", inputs=("logits",), parameters={"kind": "logits"})

    resolved_prediction_weight_bytes = mtp.prediction_layer_weight_bytes
    if mtp.prediction_layers and resolved_prediction_weight_bytes == 0:
        resolved_prediction_weight_bytes = _matrix_storage_bytes(
            hidden, hidden, dtype, layers[-1].quantization
        )
    resolved_aux_head_weight_bytes = mtp.auxiliary_head_weight_bytes
    if mtp.auxiliary_head and resolved_aux_head_weight_bytes == 0:
        resolved_aux_head_weight_bytes = _matrix_storage_bytes(
            hidden, vocabulary_size, dtype, layers[-1].quantization
        )
    mtp_previous = mtp_branch_source
    mtp_attributes = {
        "branch": "mtp",
        "source_tensor_id": mtp_branch_source,
    }
    for prediction_index in range(mtp.prediction_layers):
        prefix = "mtp.prediction_layer.{:03d}".format(prediction_index)
        output_tensor_id = prefix + ".output"
        add_operator(
            prefix,
            "mtp_prediction_layer",
            inputs=(mtp_previous,),
            outputs=((output_tensor_id, dtype, ("B", "T", hidden)),),
            weights=((
                prefix + ".weights",
                dtype,
                (hidden, hidden),
                resolved_prediction_weight_bytes,
            ),),
            parameters={
                "prediction_index": prediction_index,
                "hidden_size": hidden,
                "weight_bytes": resolved_prediction_weight_bytes,
            },
            attributes=mtp_attributes,
        )
        mtp_previous = output_tensor_id
    if mtp.auxiliary_head:
        add_operator(
            "mtp.aux_head",
            "mtp_aux_head",
            inputs=(mtp_previous,),
            outputs=((
                "mtp.proposal_logits",
                resolved_output_head_dtype,
                ("B", "T", "V"),
            ),),
            weights=((
                "mtp.aux_head.weights",
                dtype,
                (hidden, "V"),
                resolved_aux_head_weight_bytes,
            ),),
            parameters={
                "vocabulary_size": vocabulary_size,
                "hidden_size": hidden,
                "weight_bytes": resolved_aux_head_weight_bytes,
            },
            attributes=mtp_attributes,
        )
    tensors = tuple(
        TensorValue(
            tensor_id=tensor_id,
            role=record["role"],
            logical_bytes=record["logical_bytes"],
            producer_operator_id=record["producer"],
            consumer_operator_ids=tuple(record["consumers"]),
            dtype=record["dtype"],
            shape=record["shape"],
            layout=record["layout"],
        )
        for tensor_id, record in sorted(tensor_records.items())
    )
    return ModelGraph(
        graph_id=name,
        operators=tuple(operators),
        tensors=tensors,
        attributes={
            "authoritative": True,
            "derivation": "layer_specs",
            "architecture": architecture,
            "symbols": {"B": "batch", "T": "sequence", "V": vocabulary_size},
            "max_sequence_length": max_sequence_length,
            "metadata": dict(metadata or {}),
            "ui": {"collapsed_groups": [item.operator_id for item in operators if item.op_kind == "layer_group"]},
        },
    )


_LAYER_OVERRIDE_KEYS = frozenset(
    {
        "kind",
        "hidden_size",
        "intermediate_size",
        "attention_heads",
        "kv_heads",
        "attention_head_dim",
        "sequence_mixer",
        "linear_attention",
        "num_experts",
        "experts_per_token",
        "shared_expert_intermediate_size",
        "shared_expert_gate",
        "dtype",
        "quantization",
        "weight_bytes",
        "metadata",
        "schema_version",
    }
)


def _model_graph_groups(graph: "ModelGraph") -> Tuple[List[Any], Dict[str, List[Any]]]:
    by_parent: Dict[str, List[Any]] = {}
    groups = []
    for operator in graph.operators:
        if operator.op_kind == "layer_group":
            groups.append(operator)
        parent = operator.attributes.get("parent_group_id")
        if parent:
            by_parent.setdefault(str(parent), []).append(operator)
    groups.sort(key=lambda item: item.sequence_index)
    if not groups:
        raise ValueError("模型图缺少 layer_group，无法生成 layers 执行投影")
    return groups, by_parent


def _layer_from_model_graph_group(
    group: Any,
    children: Sequence[Any],
    layer_id: str,
    override: Mapping[str, Any],
    *,
    schema_version: str,
) -> LayerSpec:
    params = dict(group.parameters)
    mixer = next(
        (
            item
            for item in children
            if item.op_kind in {"attention", "linear_attention"}
        ),
        None,
    )
    ff = next(
        (
            item
            for item in children
            if item.op_kind in {"dense_mlp", "moe_router", "moe_experts"}
        ),
        None,
    )
    if mixer is None or ff is None:
        raise ValueError(
            "组件组 {} 缺少注意力或前馈组件".format(group.operator_id)
        )
    mixer_params = dict(mixer.parameters)
    ff_params = dict(ff.parameters)
    linear_raw = override.get(
        "linear_attention", mixer_params.get("linear_attention")
    )
    linear = None
    if linear_raw is not None:
        if not isinstance(linear_raw, Mapping):
            raise ValueError(
                "层 {} 的 linear_attention 必须为对象".format(layer_id)
            )
        linear = LinearAttentionSpec(**dict(linear_raw))
    return LayerSpec(
        layer_id=str(layer_id),
        kind=str(override.get("kind", ff_params.get("kind", "dense"))),
        hidden_size=int(
            override.get("hidden_size", params.get("hidden_size", 0))
        ),
        intermediate_size=int(
            override.get(
                "intermediate_size", ff_params.get("intermediate_size", 0)
            )
        ),
        attention_heads=int(
            override.get(
                "attention_heads", mixer_params.get("attention_heads", 0)
            )
        ),
        kv_heads=int(
            override.get("kv_heads", mixer_params.get("kv_heads", 0))
        ),
        attention_head_dim=int(
            override.get(
                "attention_head_dim",
                mixer_params.get("attention_head_dim", 0),
            )
        ),
        sequence_mixer=str(
            override.get(
                "sequence_mixer",
                mixer_params.get("sequence_mixer", "full_attention"),
            )
        ),
        linear_attention=linear,
        num_experts=int(
            override.get("num_experts", ff_params.get("num_experts", 1))
        ),
        experts_per_token=int(
            override.get(
                "experts_per_token", ff_params.get("experts_per_token", 1)
            )
        ),
        shared_expert_intermediate_size=int(
            override.get(
                "shared_expert_intermediate_size",
                ff_params.get("shared_expert_intermediate_size", 0),
            )
        ),
        shared_expert_gate=bool(
            override.get(
                "shared_expert_gate",
                ff_params.get("shared_expert_gate", False),
            )
        ),
        gated_mlp=bool(
            override.get("gated_mlp", ff_params.get("gated_mlp", True))
        ),
        dtype=str(override.get("dtype", params.get("dtype", "fp16"))),
        quantization=override.get(
            "quantization", ff_params.get("quantization")
        ),
        weight_bytes=int(
            override.get("weight_bytes", params.get("weight_bytes", 0))
        ),
        metadata=dict(override.get("metadata", {})),
        schema_version=str(override.get("schema_version", schema_version)),
    )


def _project_model_graph_layers_unchecked(
    graph: "ModelGraph", *, schema_version: str
) -> Tuple[LayerSpec, ...]:
    groups, by_parent = _model_graph_groups(graph)
    result = []
    for group in groups:
        params = dict(group.parameters)
        layer_ids = params.get("layer_ids", group.attributes.get("layer_ids", []))
        if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
            raise ValueError("组件组 {} 缺少 layer_ids".format(group.operator_id))
        repeat = params.get("repeat", len(layer_ids))
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError("组件组 {} 的 repeat 必须为正整数".format(group.operator_id))
        if repeat != len(layer_ids):
            raise ValueError("组件组 {} 的 repeat 与 layer_ids 数量不一致".format(group.operator_id))
        children = by_parent.get(group.operator_id, [])
        overrides = params.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("组件组 {} 的 overrides 必须为对象".format(group.operator_id))
        layer_id_strings = tuple(str(item) for item in layer_ids)
        if set(str(item) for item in overrides) != set(layer_id_strings):
            raise ValueError(
                "组件组 {} 的 overrides 必须与 layer_ids 一一对应".format(
                    group.operator_id
                )
            )
        for layer_id in layer_ids:
            override = overrides.get(str(layer_id), {})
            if not isinstance(override, Mapping):
                raise ValueError("层 {} 的 override 必须为对象".format(layer_id))
            unknown = sorted(set(str(item) for item in override) - _LAYER_OVERRIDE_KEYS)
            if unknown:
                raise ValueError(
                    "层 {} 的 override 含未知参数：{}；执行投影无法无损覆盖，已拒绝静默降级".format(
                        layer_id, "、".join(unknown)
                    )
                )
            result.append(
                _layer_from_model_graph_group(
                    group,
                    children,
                    str(layer_id),
                    override,
                    schema_version=schema_version,
                )
            )
    layer_ids = [item.layer_id for item in result]
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("模型图的 layer_id 不能重复")
    return tuple(result)


def _model_graph_mtp_descriptors(
    graph: "ModelGraph",
) -> Tuple[MTPExecutionDescriptor, ...]:
    tensors = {item.tensor_id: item for item in graph.tensors}
    predictions = sorted(
        (
            item
            for item in graph.operators
            if item.op_kind == "mtp_prediction_layer"
        ),
        key=lambda item: (item.sequence_index, item.operator_id),
    )
    aux_heads = [
        item for item in graph.operators if item.op_kind == "mtp_aux_head"
    ]
    if len(aux_heads) > 1:
        raise ValueError("模型图最多只能包含一个标准 mtp_aux_head 组件")
    if not predictions and not aux_heads:
        return ()

    final_norms = [
        item for item in graph.operators if item.operator_id == "final_norm"
    ]
    if len(final_norms) != 1 or len(final_norms[0].input_tensor_ids) != 1:
        raise ValueError(
            "标准 typed MTP 旁路必须从 final_norm 之前的 decoder hidden 分叉；执行投影已拒绝静默降级"
        )
    branch_source = final_norms[0].input_tensor_ids[0]
    previous = branch_source
    result: List[MTPExecutionDescriptor] = []

    def references(operator: Any) -> Tuple[Any, Any, Any]:
        if (
            len(operator.input_tensor_ids) != 1
            or len(operator.output_tensor_ids) != 1
            or len(operator.weight_tensor_ids) != 1
        ):
            raise ValueError(
                "MTP 组件 {} 必须各声明一个 input/output/weight typed tensor".format(
                    operator.operator_id
                )
            )
        return (
            tensors[operator.input_tensor_ids[0]],
            tensors[operator.output_tensor_ids[0]],
            tensors[operator.weight_tensor_ids[0]],
        )

    for prediction_index, operator in enumerate(predictions):
        prefix = "mtp.prediction_layer.{:03d}".format(prediction_index)
        expected_refs = (
            previous,
            prefix + ".output",
            prefix + ".weights",
        )
        actual_refs = (
            operator.input_tensor_ids[0] if operator.input_tensor_ids else "",
            operator.output_tensor_ids[0] if operator.output_tensor_ids else "",
            operator.weight_tensor_ids[0] if operator.weight_tensor_ids else "",
        )
        if operator.operator_id != prefix or actual_refs != expected_refs:
            raise ValueError(
                "标准 typed MTP 预测层 ID/张量链不一致；期望 {}，已拒绝静默降级".format(
                    prefix
                )
            )
        input_tensor, output_tensor, weight_tensor = references(operator)
        if weight_tensor.logical_bytes is None:
            raise ValueError(
                "MTP 权重张量 {} 必须显式声明 logical_bytes".format(
                    weight_tensor.tensor_id
                )
            )
        params = dict(operator.parameters)
        if params.get("prediction_index") != prediction_index:
            raise ValueError(
                "MTP 组件 {} 的 prediction_index 与稳定 ID 不一致".format(
                    operator.operator_id
                )
            )
        hidden_size = params.get("hidden_size", 0)
        _require_int(hidden_size, "MTP hidden_size", minimum=1)
        result.append(
            MTPExecutionDescriptor(
                operator=operator,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                weight_tensor=weight_tensor,
                prediction_index=prediction_index,
                hidden_size=hidden_size,
                vocabulary_size=0,
                weight_bytes=weight_tensor.logical_bytes,
            )
        )
        previous = output_tensor.tensor_id

    if aux_heads:
        operator = aux_heads[0]
        expected_refs = (
            previous,
            "mtp.proposal_logits",
            "mtp.aux_head.weights",
        )
        actual_refs = (
            operator.input_tensor_ids[0] if operator.input_tensor_ids else "",
            operator.output_tensor_ids[0] if operator.output_tensor_ids else "",
            operator.weight_tensor_ids[0] if operator.weight_tensor_ids else "",
        )
        if operator.operator_id != "mtp.aux_head" or actual_refs != expected_refs:
            raise ValueError(
                "标准 typed MTP aux head ID/张量链不一致，已拒绝静默降级"
            )
        input_tensor, output_tensor, weight_tensor = references(operator)
        if weight_tensor.logical_bytes is None:
            raise ValueError(
                "MTP 权重张量 {} 必须显式声明 logical_bytes".format(
                    weight_tensor.tensor_id
                )
            )
        params = dict(operator.parameters)
        hidden_size = params.get("hidden_size", 0)
        vocabulary_size = params.get("vocabulary_size", 0)
        _require_int(hidden_size, "MTP hidden_size", minimum=1)
        _require_int(vocabulary_size, "MTP vocabulary_size", minimum=1)
        result.append(
            MTPExecutionDescriptor(
                operator=operator,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                weight_tensor=weight_tensor,
                prediction_index=None,
                hidden_size=hidden_size,
                vocabulary_size=vocabulary_size,
                weight_bytes=weight_tensor.logical_bytes,
            )
        )
    return tuple(result)


def _model_graph_mtp_summary(graph: "ModelGraph") -> MTPBranchSpec:
    """Summarize the standard typed MTP branch for graph verification."""

    descriptors = _model_graph_mtp_descriptors(graph)
    predictions = tuple(
        item for item in descriptors if item.prediction_index is not None
    )
    aux = tuple(item for item in descriptors if item.prediction_index is None)
    prediction_bytes = {item.weight_bytes for item in predictions}
    if len(prediction_bytes) > 1:
        raise ValueError(
            "typed MTP 各预测层 logical_bytes 不一致，无法形成一致的执行摘要"
        )
    return MTPBranchSpec(
        prediction_layers=len(predictions),
        auxiliary_head=bool(aux),
        prediction_layer_weight_bytes=(
            next(iter(prediction_bytes)) if prediction_bytes else 0
        ),
        auxiliary_head_weight_bytes=aux[0].weight_bytes if aux else 0,
    )


def _model_graph_authoring_summary(
    graph: "ModelGraph",
) -> Tuple[str, int, int, int, int, str]:
    """Resolve graph-owned model metadata and reject conflicting mirrors.

    The standard graph intentionally repeats a few values at the points where
    they are consumed (for example ``V`` on embedding and LM-head operators).
    Those repetitions are graph-internal contracts, not ModelSpec fields.  A
    disagreement is rejected so no consumer can silently choose a different
    source of truth.
    """

    attributes = graph.attributes
    architecture = attributes.get("architecture")
    _require_name(architecture, "model.graph attributes.architecture")

    symbols = attributes.get("symbols")
    if not isinstance(symbols, Mapping):
        raise ValueError("model.graph attributes.symbols must be a mapping")
    vocabulary_size = symbols.get("V", 0)
    _require_int(vocabulary_size, "model.graph symbols.V")

    max_sequence_length = attributes.get("max_sequence_length", 0)
    _require_int(
        max_sequence_length,
        "model.graph attributes.max_sequence_length",
    )

    vocabulary_mirrors: List[Tuple[str, Any]] = []
    sequence_mirrors: List[Tuple[str, Any]] = []
    for operator in graph.operators:
        parameters = operator.parameters
        if operator.op_kind in {"embedding", "lm_head", "mtp_aux_head"}:
            if "vocabulary_size" in parameters:
                vocabulary_mirrors.append(
                    (operator.operator_id, parameters["vocabulary_size"])
                )
        if operator.op_kind == "model_input" and "max_sequence_length" in parameters:
            sequence_mirrors.append(
                (operator.operator_id, parameters["max_sequence_length"])
            )
    for operator_id, value in vocabulary_mirrors:
        _require_int(
            value,
            "model.graph operator {} vocabulary_size".format(operator_id),
        )
        if value != vocabulary_size:
            raise ValueError(
                "model.graph vocabulary_size conflict: symbols.V={} but {} declares {}".format(
                    vocabulary_size, operator_id, value
                )
            )
    for operator_id, value in sequence_mirrors:
        _require_int(
            value,
            "model.graph operator {} max_sequence_length".format(operator_id),
        )
        if value != max_sequence_length:
            raise ValueError(
                "model.graph max_sequence_length conflict: attributes={} but {} declares {}".format(
                    max_sequence_length, operator_id, value
                )
            )

    embedding_operators = [
        item for item in graph.operators if item.op_kind == "embedding"
    ]
    if len(embedding_operators) != 1:
        raise ValueError("model.graph must contain exactly one embedding operator")
    embedding_operator = embedding_operators[0]
    if len(embedding_operator.weight_tensor_ids) != 1:
        raise ValueError("model.graph embedding must declare exactly one weight tensor")
    tensor_map = {item.tensor_id: item for item in graph.tensors}
    embedding_tensor = tensor_map[embedding_operator.weight_tensor_ids[0]]
    embedding_weight_bytes = embedding_tensor.logical_bytes
    if embedding_weight_bytes is None:
        dimensions: List[int] = []
        for raw_dimension in embedding_tensor.shape:
            dimension = raw_dimension
            if isinstance(dimension, str):
                dimension = symbols.get(dimension)
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
            ):
                raise ValueError(
                    "model.graph embedding weight bytes are not derivable; declare tensor logical_bytes"
                )
            dimensions.append(dimension)
        element_count = math.prod(dimensions)
        embedding_weight_bytes = (
            element_count
            * dtype_bits(
                embedding_tensor.dtype,
                unsupported_message=(
                    "不支持的数据类型 {}，无法推导 embedding 权重 logical_bytes".format(
                        embedding_tensor.dtype
                    )
                ),
            )
            + 7
        ) // 8
    _require_int(
        embedding_weight_bytes,
        "model.graph embedding logical_bytes",
    )
    lm_head_operators = [
        item for item in graph.operators if item.op_kind == "lm_head"
    ]
    if len(lm_head_operators) != 1:
        raise ValueError("model.graph must contain exactly one lm_head operator")
    lm_head_operator = lm_head_operators[0]
    if len(lm_head_operator.output_tensor_ids) != 1:
        raise ValueError("model.graph lm_head must declare exactly one output tensor")
    if len(lm_head_operator.weight_tensor_ids) != 1:
        raise ValueError("model.graph lm_head must declare exactly one weight tensor")
    output_weight_tensor = tensor_map[lm_head_operator.weight_tensor_ids[0]]
    output_weight_bytes = output_weight_tensor.logical_bytes or 0
    _require_int(output_weight_bytes, "model.graph lm_head logical_bytes")
    output_head_tensor = tensor_map[lm_head_operator.output_tensor_ids[0]]
    _require_name(output_head_tensor.dtype, "model.graph lm_head output dtype")
    return (
        str(architecture),
        vocabulary_size,
        max_sequence_length,
        embedding_weight_bytes,
        output_weight_bytes,
        output_head_tensor.dtype,
    )


def _without_graph_provenance(value: Any) -> Any:
    """Drop only schema provenance fields, never user execution parameters.

    ``provenance`` is also a valid arbitrary key inside operator parameters or
    attributes.  Recursively deleting every mapping key with that spelling
    would let an unknown execution parameter evade the coverage gate and the
    mapping fingerprint.  The canonical schema stores provenance only on the
    graph and its top-level operator/tensor/transform records, so removal is
    intentionally limited to those structural positions.
    """

    if not isinstance(value, Mapping):
        raise TypeError("模型图执行 payload 必须是 mapping")
    payload = {str(key): item for key, item in value.items()}
    payload.pop("provenance", None)
    for collection_name in ("operators", "tensors", "transforms"):
        collection = payload.get(collection_name)
        if not isinstance(collection, (list, tuple)):
            continue
        normalized = []
        for item in collection:
            if isinstance(item, Mapping):
                record = {str(key): nested for key, nested in item.items()}
                record.pop("provenance", None)
                normalized.append(record)
            else:
                normalized.append(item)
        payload[collection_name] = normalized
    return payload


def _derived_tensor_logical_bytes(
    tensor: Mapping[str, Any], symbols: Mapping[str, Any]
) -> Optional[int]:
    """Return bytes fixed entirely by a tensor's dtype/shape contract.

    A concrete ``logical_bytes`` equal to this value is only a redundant
    serialization of the typed contract.  Unresolved symbols and unknown
    dtypes deliberately return ``None`` so the coverage gate continues to
    compare their explicit byte declarations fail-closed.
    """

    dtype = tensor.get("dtype")
    shape = tensor.get("shape")
    if not isinstance(dtype, str) or not isinstance(shape, (list, tuple)):
        return None
    try:
        storage_bits = dtype_bits(
            dtype,
            unsupported_message=(
                "不支持的数据类型 {}，无法推导 MTP 权重 logical_bytes".format(
                    dtype
                )
            ),
        )
    except ValueError:
        return None
    element_count = 1
    for raw_dimension in shape:
        dimension = raw_dimension
        seen = set()
        while (
            isinstance(dimension, str)
            and dimension in symbols
            and dimension not in seen
        ):
            seen.add(dimension)
            dimension = symbols[dimension]
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
        ):
            return None
        element_count *= dimension
    return (element_count * storage_bits + 7) // 8


def _model_graph_execution_payload(graph: "ModelGraph") -> Dict[str, Any]:
    from .serde import to_primitive

    payload = _without_graph_provenance(to_primitive(graph))
    attributes = payload.get("attributes")
    if isinstance(attributes, dict):
        attributes.pop("ui", None)
    operators = payload.get("operators")
    if isinstance(operators, list):
        payload["operators"] = sorted(
            operators,
            key=lambda item: (
                item.get("sequence_index", 0) if isinstance(item, Mapping) else 0,
                str(item.get("operator_id", "")) if isinstance(item, Mapping) else "",
                str(item.get("op_kind", "")) if isinstance(item, Mapping) else "",
            ),
        )
        # ``gated_mlp`` was added after the fixed validation graphs were
        # authored.  Its schema default is true, so an explicit true and an
        # omitted value describe the same execution graph.  Keep false
        # visible: it changes the projected GEMM shape and must remain part of
        # the coverage contract.
        for operator in payload["operators"]:
            if not isinstance(operator, dict):
                continue
            parameters = operator.get("parameters")
            if isinstance(parameters, dict) and parameters.get("gated_mlp") is True:
                parameters.pop("gated_mlp")
    tensors = payload.get("tensors")
    if isinstance(tensors, list):
        symbols = {}
        if isinstance(attributes, Mapping):
            raw_symbols = attributes.get("symbols")
            if isinstance(raw_symbols, Mapping):
                symbols = raw_symbols
        normalized_tensors = []
        for tensor in tensors:
            if isinstance(tensor, Mapping):
                next_tensor = dict(tensor)
                if (
                    next_tensor.get("tensor_id") == "lm_head_weights"
                    and next_tensor.get("logical_bytes") == 0
                ):
                    next_tensor["logical_bytes"] = None
                derived_logical_bytes = _derived_tensor_logical_bytes(
                    next_tensor, symbols
                )
                if (
                    derived_logical_bytes is not None
                    and next_tensor.get("logical_bytes")
                    == derived_logical_bytes
                ):
                    next_tensor["logical_bytes"] = None
                consumers = next_tensor.get("consumer_operator_ids")
                if isinstance(consumers, list):
                    next_tensor["consumer_operator_ids"] = sorted(
                        consumers, key=lambda item: str(item)
                    )
                normalized_tensors.append(next_tensor)
            else:
                normalized_tensors.append(tensor)
        payload["tensors"] = sorted(
            normalized_tensors,
            key=lambda item: (
                str(item.get("tensor_id", "")) if isinstance(item, Mapping) else ""
            ),
        )
    return payload


def _first_graph_difference(expected: Any, actual: Any, path: str = "graph") -> str:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        missing = sorted(expected_keys - actual_keys)
        if missing:
            return "{} 缺少字段 {}".format(path, "、".join(missing))
        extra = sorted(actual_keys - expected_keys)
        if extra:
            return "{} 含未知字段 {}".format(path, "、".join(extra))
        for key in sorted(expected_keys):
            difference = _first_graph_difference(
                expected[key], actual[key], "{}.{}".format(path, key)
            )
            if difference:
                return difference
        return ""
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return "{} 数量不一致（期望 {}，实际 {}）".format(
                path, len(expected), len(actual)
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            difference = _first_graph_difference(
                expected_item,
                actual_item,
                "{}[{}]".format(path, index),
            )
            if difference:
                return difference
        return ""
    if expected != actual:
        return "{} 不一致（期望 {!r}，实际 {!r}）".format(
            path, expected, actual
        )
    return ""


def _assert_execution_projection_coverage(
    graph: "ModelGraph",
    projected_layers: Sequence[LayerSpec],
    *,
    schema_version: str,
) -> None:
    """Fail closed unless the graph is exactly the supported layer macro.

    The current planner executes ordered ``LayerSpec`` values.  This check
    therefore treats the standard graph emitted by
    ``build_model_graph_from_layer_specs``
    as the complete execution surface. Per-layer overrides are retained,
    while UI-only graph attributes and provenance never affect coverage.
    """

    if graph.transforms:
        raise ValueError(
            "模型图包含显式 Transform；layers 执行投影无法执行，已拒绝静默降级"
        )
    groups, by_parent = _model_graph_groups(graph)
    base_layers: List[LayerSpec] = []
    for group_index, group in enumerate(groups):
        params = dict(group.parameters)
        layer_ids = params.get(
            "layer_ids", group.attributes.get("layer_ids", [])
        )
        children = by_parent.get(group.operator_id, [])
        marker = "__projection_group_{}".format(group_index)
        for layer_id in layer_ids:
            base_layers.append(
                _layer_from_model_graph_group(
                    group,
                    children,
                    str(layer_id),
                    {"metadata": {"pattern_index": marker}},
                    schema_version=schema_version,
                )
            )
    attributes = dict(graph.attributes)
    symbols = attributes.get("symbols", {})
    if not isinstance(symbols, Mapping):
        raise ValueError(
            "模型图 attributes.symbols 必须为对象；执行投影已拒绝静默降级"
        )
    metadata = attributes.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "模型图 attributes.metadata 必须为对象；执行投影已拒绝静默降级"
        )
    mtp_projection = _model_graph_mtp_summary(graph)
    (
        architecture,
        vocabulary_size,
        max_sequence_length,
        embedding_weight_bytes,
        output_weight_bytes,
        output_head_dtype,
    ) = _model_graph_authoring_summary(graph)
    expected = build_model_graph_from_layer_specs(
        graph.graph_id,
        base_layers,
        architecture=architecture,
        vocabulary_size=vocabulary_size,
        max_sequence_length=max_sequence_length,
        embedding_weight_bytes=embedding_weight_bytes,
        output_weight_bytes=output_weight_bytes,
        output_head_dtype=output_head_dtype,
        metadata=metadata,
        mtp=mtp_projection,
    )
    expected_payload = _model_graph_execution_payload(expected)
    actual_payload = _model_graph_execution_payload(graph)
    expected_groups = [
        item
        for item in expected_payload["operators"]
        if item.get("op_kind") == "layer_group"
    ]
    actual_groups = [
        item
        for item in actual_payload["operators"]
        if item.get("op_kind") == "layer_group"
    ]
    if len(expected_groups) == len(actual_groups):
        for expected_group, actual_group in zip(
            expected_groups, actual_groups
        ):
            expected_group["parameters"]["overrides"] = actual_group.get(
                "parameters", {}
            ).get("overrides")
    difference = _first_graph_difference(expected_payload, actual_payload)
    if difference:
        raise ValueError(
            "模型图无法被 layers 执行投影无损覆盖：{}；可能存在额外/缺失组件、重连、额外边、未知参数或端口/张量 dtype/shape/layout 合同差异，已拒绝静默降级".format(
                difference
            )
        )
    if tuple(item.layer_id for item in base_layers) != tuple(
        item.layer_id for item in projected_layers
    ):
        raise ValueError(
            "模型图 layer_group 展开顺序不稳定；执行投影已拒绝静默降级"
        )


def model_graph_execution_layers(
    graph: "ModelGraph", *, schema_version: str = SCHEMA_VERSION
) -> Tuple[LayerSpec, ...]:
    """Project a fully covered authoring graph into ordered execution layers.

    Projection is also the centralized execution gate: a graph
    is rejected unless every executable detail is represented losslessly by
    the existing standard Transformer ``layer_group``/``layers`` lowering.
    """

    if not graph.executable:
        raise ValueError("不可执行的模型图不能投影为 layers")
    projected = _project_model_graph_layers_unchecked(
        graph, schema_version=schema_version
    )
    _assert_execution_projection_coverage(
        graph, projected, schema_version=schema_version
    )
    return projected


def model_graph_execution_view(
    graph: "ModelGraph", *, schema_version: str = SCHEMA_VERSION
) -> ModelGraphExecutionView:
    """Return a stable graph-native execution view after fail-closed coverage.

    Operators remain the authoritative typed nodes.  ``layer_instances`` only
    supplies the cost geometry for each real layer expanded from a
    compact ``layer_group``; it is not a second executable definition.
    """

    # ``ModelGraph`` is a frozen value object and execution projection is a
    # pure function of that graph plus the requested schema version.  A
    # single scenario is wrapped in several short-lived CompilationContext
    # instances during validation, mapping, planning, and reporting.  Cache
    # the validated view on the graph itself so those contexts do not each
    # rebuild and serialize the same large authoring graph.  The attribute is
    # not a dataclass field, hence canonical serialization and equality are
    # unchanged; ``dataclasses.replace`` creates a fresh graph without the
    # cache, which is the required invalidation boundary.
    # ``ModelGraph`` is frozen, but its compatibility-facing nested mappings
    # are intentionally mutable.  Guard the object-local cache with the exact
    # executable payload so a post-parse mutation can never reuse a stale
    # validated projection.  Building this normalized payload is materially
    # cheaper than re-projecting and re-validating a large repeated graph.
    from .serde import stable_hash

    cache_guard = stable_hash(_model_graph_execution_payload(graph))
    cache = getattr(graph, "_execution_view_cache", None)
    if isinstance(cache, dict):
        cached = cache.get(schema_version)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] == cache_guard
            and isinstance(cached[1], ModelGraphExecutionView)
        ):
            return cached[1]

    if not graph.executable:
        raise ValueError("不可执行的模型图不能生成执行视图")
    projected = _project_model_graph_layers_unchecked(
        graph, schema_version=schema_version
    )
    _assert_execution_projection_coverage(
        graph, projected, schema_version=schema_version
    )

    operator_map = {item.operator_id: item for item in graph.operators}
    adjacency: Dict[str, set] = {
        item.operator_id: set() for item in graph.operators
    }
    indegree = {item.operator_id: 0 for item in graph.operators}
    for tensor in graph.tensors:
        producer = tensor.producer_operator_id
        if producer is None:
            continue
        for consumer in tensor.consumer_operator_ids:
            if consumer == producer or consumer in adjacency[producer]:
                continue
            adjacency[producer].add(consumer)
            indegree[consumer] += 1

    def operator_key(operator_id: str) -> Tuple[int, str]:
        operator = operator_map[operator_id]
        return operator.sequence_index, operator.operator_id

    ready = sorted(
        (operator_id for operator_id, degree in indegree.items() if degree == 0),
        key=operator_key,
    )
    ordered = []
    while ready:
        operator_id = ready.pop(0)
        ordered.append(operator_map[operator_id])
        for consumer in sorted(adjacency[operator_id], key=operator_key):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                ready.append(consumer)
                ready.sort(key=operator_key)
    if len(ordered) != len(graph.operators):
        raise ValueError("模型组件图必须是 DAG，无法生成稳定执行视图")

    groups, _ = _model_graph_groups(graph)
    layer_instances: List[GraphLayerExecutionDescriptor] = []
    projected_index = 0
    for group in groups:
        params = dict(group.parameters)
        layer_ids = params.get(
            "layer_ids", group.attributes.get("layer_ids", [])
        )
        for repeat_index, layer_id in enumerate(layer_ids):
            layer = projected[projected_index]
            layer_instances.append(
                GraphLayerExecutionDescriptor(
                    group_operator_id=group.operator_id,
                    layer_id=str(layer_id),
                    repeat_index=repeat_index,
                    layer=layer,
                )
            )
            projected_index += 1

    (
        architecture,
        vocabulary_size,
        max_sequence_length,
        embedding_weight_bytes,
        output_weight_bytes,
        _output_head_dtype,
    ) = _model_graph_authoring_summary(graph)
    execution_view = ModelGraphExecutionView(
        operators=tuple(ordered),
        tensors=tuple(sorted(graph.tensors, key=lambda item: item.tensor_id)),
        layer_instances=tuple(layer_instances),
        mtp_descriptors=_model_graph_mtp_descriptors(graph),
        architecture=architecture,
        vocabulary_size=vocabulary_size,
        max_sequence_length=max_sequence_length,
        embedding_weight_bytes=embedding_weight_bytes,
        output_weight_bytes=output_weight_bytes,
    )
    if not isinstance(cache, dict):
        cache = {}
        object.__setattr__(graph, "_execution_view_cache", cache)
    cache[schema_version] = (cache_guard, execution_view)
    return execution_view


def model_graph_execution_digest(
    graph: "ModelGraph", *, schema_version: str = SCHEMA_VERSION
) -> str:
    """Hash the validated execution semantics, excluding UI and provenance."""

    from .serde import stable_hash

    model_graph_execution_layers(graph, schema_version=schema_version)
    return stable_hash(_model_graph_execution_payload(graph))


@dataclass(frozen=True)
class PortSpec:
    """A typed component port.

    ``version`` is the maximum protocol version supported by the port.
    ``payload`` is primarily used for UCIe (for example ``streaming`` or
    ``cxl``), but is left generic for protocol extensions.
    """

    port_id: str
    protocol: str
    role: str
    direction: str = "bidirectional"
    version: str = "1.0"
    lanes: int = 1
    bandwidth_gbps: float = 0.0
    max_links: int = 1
    payload: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.port_id, "port_id")
        _require_name(self.protocol, "protocol")
        _require_name(self.role, "role")
        _require_name(self.direction, "direction")
        _require_name(self.version, "version")
        _require_optional_name(self.payload, "payload")
        _require_schema_version(self.schema_version)
        _require_int(self.lanes, "lanes", minimum=1)
        _require_number(self.bandwidth_gbps, "bandwidth_gbps")
        _require_int(self.max_links, "max_links", minimum=1)
        _require_mapping(self.metadata, "metadata")
        if self.direction not in {"input", "output", "bidirectional"}:
            raise ValueError("direction must be input, output, or bidirectional")


@dataclass(frozen=True)
class ComponentSpec:
    """A compute, memory, fabric, or bridge component in HardwareIR."""

    component_id: str
    kind: str
    cost_profile_id: Optional[str] = field(default=None, kw_only=True)
    ports: Tuple[PortSpec, ...] = ()
    package_id: str = ""
    die_id: str = ""
    capacity_bytes: int = 0
    peak_ops_per_s: float = 0.0
    read_bandwidth_gbps: float = 0.0
    write_bandwidth_gbps: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.component_id, "component_id")
        _require_name(self.kind, "kind")
        _require_optional_name(self.cost_profile_id, "cost_profile_id")
        _require_tuple(self.ports, "ports")
        if not all(isinstance(port, PortSpec) for port in self.ports):
            raise ValueError("ports must contain PortSpec values")
        port_ids = [port.port_id for port in self.ports]
        if len(port_ids) != len(set(port_ids)):
            raise ValueError("port_id values must be unique within a component")
        # Negative capacity remains representable so topology validation can
        # return a structured ``negative_capacity`` diagnostic.
        if isinstance(self.capacity_bytes, bool) or not isinstance(self.capacity_bytes, int):
            raise ValueError("capacity_bytes must be an integer")
        _require_number(self.peak_ops_per_s, "peak_ops_per_s")
        _require_number(self.read_bandwidth_gbps, "read_bandwidth_gbps")
        _require_number(self.write_bandwidth_gbps, "write_bandwidth_gbps")
        _require_mapping(self.metadata, "metadata")
        _require_schema_version(self.schema_version)
        if self.package_id:
            _require_name(self.package_id, "package_id")
        if self.die_id:
            _require_name(self.die_id, "die_id")

    def port_map(self) -> Dict[str, PortSpec]:
        return {port.port_id: port for port in self.ports}

    @property
    def normalized_kind(self) -> str:
        return normalize_component_kind(self.kind)

    @property
    def memory_class(self) -> Optional[str]:
        """Classify memory-like components as ``active`` or ``offload``."""

        if self.normalized_kind in ACTIVE_MEMORY_COMPONENT_KINDS:
            return "active"
        if self.normalized_kind in OFFLOAD_STORAGE_COMPONENT_KINDS:
            return "offload"
        return None

    @property
    def is_active_memory(self) -> bool:
        return self.memory_class == "active"

    @property
    def is_writable(self) -> bool:
        """Return metadata-declared writability, independent of bandwidth."""

        return (
            self.metadata.get("read_only", False) is not True
            and self.metadata.get("writable", True) is not False
        )

    @property
    def is_storage(self) -> bool:
        return self.normalized_kind in STORAGE_COMPONENT_KINDS


@dataclass(frozen=True)
class LinkSpec:
    """A negotiated connection between two component ports."""

    link_id: str
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    protocol: str
    version: str = "1.0"
    lanes: int = 1
    bandwidth_gbps: float = 0.0
    latency_ns: float = 0.0
    bidirectional: bool = True
    payload: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "link_id",
            "source_component",
            "source_port",
            "target_component",
            "target_port",
            "protocol",
            "version",
        ):
            _require_name(getattr(self, field_name), field_name)
        _require_schema_version(self.schema_version)
        _require_optional_name(self.payload, "payload")
        _require_int(self.lanes, "lanes", minimum=1)
        _require_number(self.bandwidth_gbps, "bandwidth_gbps")
        _require_number(self.latency_ns, "latency_ns")
        if not isinstance(self.bidirectional, bool):
            raise ValueError("bidirectional must be boolean")
        _require_mapping(self.metadata, "metadata")
        if (self.source_component, self.source_port) == (
            self.target_component,
            self.target_port,
        ):
            raise ValueError("link endpoints must be distinct")

    @property
    def endpoints(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        return (
            (self.source_component, self.source_port),
            (self.target_component, self.target_port),
        )


@dataclass(frozen=True)
class HardwareSpec:
    """A component graph and its physical/protocol links."""

    name: str
    components: Tuple[ComponentSpec, ...]
    links: Tuple[LinkSpec, ...]
    require_connected: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.name, "name")
        _require_tuple(self.components, "components")
        _require_tuple(self.links, "links")
        if not self.components:
            raise ValueError("components must not be empty")
        if not all(isinstance(component, ComponentSpec) for component in self.components):
            raise ValueError("components must contain ComponentSpec values")
        if not all(isinstance(link, LinkSpec) for link in self.links):
            raise ValueError("links must contain LinkSpec values")
        component_ids = [component.component_id for component in self.components]
        link_ids = [link.link_id for link in self.links]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component_id values must be unique within hardware")
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("link_id values must be unique within hardware")
        if not isinstance(self.require_connected, bool):
            raise ValueError("require_connected must be boolean")
        _require_mapping(self.metadata, "metadata")
        _require_schema_version(self.schema_version)

    def component_map(self) -> Dict[str, ComponentSpec]:
        return {component.component_id: component for component in self.components}

    def get_component(self, component_id: str) -> ComponentSpec:
        try:
            return self.component_map()[component_id]
        except KeyError:
            raise KeyError("unknown component_id: {}".format(component_id))

    def get_port(self, component_id: str, port_id: str) -> PortSpec:
        component = self.get_component(component_id)
        try:
            return component.port_map()[port_id]
        except KeyError:
            raise KeyError("unknown port: {}.{}".format(component_id, port_id))


@dataclass(frozen=True)
class RankMappingSpec:
    """Placement of one logical parallel rank on hardware components."""

    rank: int
    component_id: str
    tp_rank: int
    pp_rank: int
    ep_rank: int
    memory_component_id: Optional[str] = None
    cim_component_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_int(self.rank, "rank")
        _require_name(self.component_id, "component_id")
        _require_int(self.tp_rank, "tp_rank")
        _require_int(self.pp_rank, "pp_rank")
        _require_int(self.ep_rank, "ep_rank")
        _require_optional_name(self.memory_component_id, "memory_component_id")
        _require_optional_name(self.cim_component_id, "cim_component_id")


@dataclass(frozen=True)
class ParallelSpec:
    """Inference parallelism and logical-rank mapping."""

    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    rank_mapping: Tuple[RankMappingSpec, ...] = ()
    layer_to_stage: Mapping[str, int] = field(default_factory=dict)
    collective_algorithm: str = "auto"
    routing_policy: str = "lowest_latency"
    allow_padding: bool = True

    def __post_init__(self) -> None:
        _require_int(self.tp_degree, "tp_degree", minimum=1)
        _require_int(self.pp_degree, "pp_degree", minimum=1)
        _require_int(self.ep_degree, "ep_degree", minimum=1)
        _require_tuple(self.rank_mapping, "rank_mapping")
        _require_mapping(self.layer_to_stage, "layer_to_stage")
        _require_name(self.collective_algorithm, "collective_algorithm")
        _require_name(self.routing_policy, "routing_policy")
        if self.routing_policy != "lowest_latency":
            raise ValueError("routing_policy must be lowest_latency")
        if not isinstance(self.allow_padding, bool):
            raise ValueError("allow_padding must be boolean")
        if not all(isinstance(mapping, RankMappingSpec) for mapping in self.rank_mapping):
            raise ValueError("rank_mapping must contain RankMappingSpec values")
        if self.rank_mapping:
            if len(self.rank_mapping) != self.world_size:
                raise ValueError("rank_mapping must contain exactly world_size entries")
            ranks = [mapping.rank for mapping in self.rank_mapping]
            coordinates = [
                (mapping.tp_rank, mapping.pp_rank, mapping.ep_rank)
                for mapping in self.rank_mapping
            ]
            if set(ranks) != set(range(self.world_size)):
                raise ValueError("rank_mapping ranks must cover [0, world_size)")
            if len(coordinates) != len(set(coordinates)):
                raise ValueError("parallel rank coordinates must be unique")
            for mapping in self.rank_mapping:
                if mapping.tp_rank >= self.tp_degree:
                    raise ValueError("tp_rank must be less than tp_degree")
                if mapping.pp_rank >= self.pp_degree:
                    raise ValueError("pp_rank must be less than pp_degree")
                if mapping.ep_rank >= self.ep_degree:
                    raise ValueError("ep_rank must be less than ep_degree")
        for layer_id, stage in self.layer_to_stage.items():
            _require_name(layer_id, "layer_to_stage key")
            _require_int(stage, "layer_to_stage value")
            if stage >= self.pp_degree:
                raise ValueError("layer_to_stage values must be less than pp_degree")

    @property
    def world_size(self) -> int:
        return self.tp_degree * self.pp_degree * self.ep_degree


@dataclass(frozen=True)
class KVCachePolicy:
    """KV-cache paging, placement, offload, and preemption policy."""

    cache_component: Optional[str] = None
    offload_component: Optional[str] = None
    tokens_per_page: int = 16
    dtype: Optional[str] = None
    offload_ratio: float = 1.0
    allocation_policy: str = "lazy"
    preemption_mode: str = "auto"
    prefetch_distance: int = 0

    def __post_init__(self) -> None:
        _require_optional_name(self.cache_component, "cache_component")
        _require_optional_name(self.offload_component, "offload_component")
        _require_optional_name(self.dtype, "dtype")
        _require_int(self.tokens_per_page, "tokens_per_page", minimum=1)
        _require_number(self.offload_ratio, "offload_ratio")
        if self.offload_ratio > 1.0:
            raise ValueError("offload_ratio must be in [0, 1]")
        _require_name(self.allocation_policy, "allocation_policy")
        _require_name(self.preemption_mode, "preemption_mode")
        _require_int(self.prefetch_distance, "prefetch_distance")
        if self.allocation_policy not in {"lazy", "eager"}:
            raise ValueError("allocation_policy must be lazy or eager")
        if self.preemption_mode not in {"auto", "swap", "recompute"}:
            raise ValueError("preemption_mode must be auto, swap, or recompute")

@dataclass(frozen=True)
class MTPPolicy:
    """Multi-token-prediction proposal and acceptance model."""

    method: str = "head_based"
    # Maximum number of draft tokens.  The verifier width is 1 + this value
    # because every MTP round also contains one guaranteed main-head token.
    candidate_tokens: int = 4
    min_draft_tokens: int = 0
    continuation_threshold: Optional[float] = None
    proposal_length_model: str = "max"
    expected_draft_tokens_per_round: Optional[float] = None
    draft_length_trace: Tuple[int, ...] = ()
    acceptance_model: str = "expected"
    acceptance_rate: Optional[float] = None
    proposal_cost_scale: float = 0.15
    acceptance_trace: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _require_name(self.method, "method")
        _require_int(self.candidate_tokens, "candidate_tokens", minimum=1)
        _require_int(self.min_draft_tokens, "min_draft_tokens")
        if self.min_draft_tokens > self.candidate_tokens:
            raise ValueError(
                "min_draft_tokens must not exceed candidate_tokens"
            )
        if self.continuation_threshold is not None:
            _require_number(
                self.continuation_threshold, "continuation_threshold"
            )
            if self.continuation_threshold > 1.0:
                raise ValueError("continuation_threshold must be in [0, 1]")
        _require_name(self.proposal_length_model, "proposal_length_model")
        normalized_proposal_model = self.proposal_length_model.strip().lower()
        if normalized_proposal_model not in {"max", "expected_mean", "trace"}:
            raise ValueError(
                "proposal_length_model must be max, expected_mean, or trace"
            )
        if self.expected_draft_tokens_per_round is not None:
            _require_number(
                self.expected_draft_tokens_per_round,
                "expected_draft_tokens_per_round",
            )
            if not (
                self.min_draft_tokens
                <= self.expected_draft_tokens_per_round
                <= self.candidate_tokens
            ):
                raise ValueError(
                    "expected_draft_tokens_per_round must be between "
                    "min_draft_tokens and candidate_tokens"
                )
        if (
            normalized_proposal_model == "expected_mean"
            and self.expected_draft_tokens_per_round is None
        ):
            raise ValueError(
                "proposal_length_model='expected_mean' requires "
                "expected_draft_tokens_per_round"
            )
        _require_tuple(self.draft_length_trace, "draft_length_trace")
        for value in self.draft_length_trace:
            _require_int(value, "draft_length_trace value")
            if not self.min_draft_tokens <= value <= self.candidate_tokens:
                raise ValueError(
                    "draft_length_trace values must be between "
                    "min_draft_tokens and candidate_tokens"
                )
        if normalized_proposal_model == "trace" and not self.draft_length_trace:
            raise ValueError(
                "proposal_length_model='trace' requires draft_length_trace"
            )
        _require_name(self.acceptance_model, "acceptance_model")
        normalized_acceptance_model = self.acceptance_model.strip().lower()
        if normalized_acceptance_model not in {
            "expected",
            "expected_prefix",
            "trace",
        }:
            raise ValueError(
                "acceptance_model must be expected, expected_prefix, or trace"
            )
        if self.acceptance_rate is not None:
            _require_number(self.acceptance_rate, "acceptance_rate")
            if self.acceptance_rate > 1.0:
                raise ValueError("acceptance_rate must be in [0, 1]")
        _require_number(self.proposal_cost_scale, "proposal_cost_scale")
        if self.proposal_cost_scale == 0:
            raise ValueError("proposal_cost_scale must be positive")
        _require_tuple(self.acceptance_trace, "acceptance_trace")
        for value in self.acceptance_trace:
            _require_number(value, "acceptance_trace value")
            if value > 1.0:
                raise ValueError("acceptance_trace values must be in [0, 1]")
        if normalized_acceptance_model == "trace" and not self.acceptance_trace:
            raise ValueError("acceptance_model='trace' requires acceptance_trace")

    @property
    def enabled(self) -> bool:
        return self.method.strip().lower() not in {
            "",
            "none",
            "disabled",
            "off",
        }


@dataclass(frozen=True)
class SchedulerSpec:
    """Static or continuous inference-request scheduling policy."""

    mode: str = "static"
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 2048
    # Logical scheduler packing and one hardware graph microbatch are
    # separate limits.  ``None`` preserves the historical contract by using
    # ``max_num_batched_tokens`` as the physical limit at compile time.
    max_num_ubatch_tokens: Optional[int] = None
    prefill_chunk_tokens: int = 512
    mixed_phase_batching: bool = False
    policy: str = "decode_first"
    phase_candidate_order: str = "least_recently_served"
    starvation_ns: float = 5_000_000
    preemption_enabled: bool = True
    preemption_granularity: str = "boundary"
    preemption_policy: str = "auto"
    slo_ttft_ns: Optional[float] = None
    slo_tbt_ns: Optional[float] = None
    # Optional backend stops measured backward from the end of a fresh prompt.
    # This partitions work; it does not charge checkpoint time or bytes.
    prefill_stop_offsets: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_name(self.mode, "mode")
        if self.mode not in {"static", "continuous"}:
            raise ValueError("mode must be static or continuous")
        _require_int(self.max_num_seqs, "max_num_seqs", minimum=1)
        _require_int(self.max_num_batched_tokens, "max_num_batched_tokens", minimum=1)
        if self.max_num_ubatch_tokens is not None:
            _require_int(
                self.max_num_ubatch_tokens,
                "max_num_ubatch_tokens",
                minimum=1,
            )
        _require_int(self.prefill_chunk_tokens, "prefill_chunk_tokens", minimum=1)
        if not isinstance(self.prefill_stop_offsets, tuple):
            raise ValueError("prefill_stop_offsets must be a tuple")
        for offset in self.prefill_stop_offsets:
            _require_int(offset, "prefill_stop_offsets value", minimum=1)
        if len(set(self.prefill_stop_offsets)) != len(self.prefill_stop_offsets):
            raise ValueError("prefill_stop_offsets must be unique")
        if not isinstance(self.mixed_phase_batching, bool):
            raise ValueError("mixed_phase_batching must be boolean")
        _require_name(self.policy, "policy")
        _require_name(self.phase_candidate_order, "phase_candidate_order")
        _require_number(self.starvation_ns, "starvation_ns")
        if self.starvation_ns == 0:
            raise ValueError("starvation_ns must be positive")
        if not isinstance(self.preemption_enabled, bool):
            raise ValueError("preemption_enabled must be boolean")
        _require_name(self.preemption_granularity, "preemption_granularity")
        _require_name(self.preemption_policy, "preemption_policy")
        if self.policy not in {"decode_first", "decode_first_aging"}:
            raise ValueError("unsupported scheduler policy")
        if self.phase_candidate_order not in {
            "least_recently_served",
            "stable_admission",
        }:
            raise ValueError("unsupported scheduler phase_candidate_order")
        for field_name in ("slo_ttft_ns", "slo_tbt_ns"):
            value = getattr(self, field_name)
            if value is not None:
                _require_number(value, field_name)
                if value == 0:
                    raise ValueError("{} must be positive".format(field_name))
        if self.preemption_granularity != "boundary":
            raise ValueError(
                "preemption_granularity must be boundary; running kernels and "
                "collectives cannot be interrupted"
            )
        if self.preemption_policy not in {"auto", "swap", "recompute"}:
            raise ValueError("preemption_policy must be auto, swap, or recompute")


@dataclass(frozen=True)
class PlacementSpec:
    """Placement and parallelism decisions made before task lowering."""

    model_name: str
    hardware_name: str
    op_to_component: Mapping[str, str] = field(default_factory=dict)
    tensor_to_component: Mapping[str, str] = field(default_factory=dict)
    tensor_bytes: Mapping[str, int] = field(default_factory=dict)
    parallel: ParallelSpec = field(default_factory=ParallelSpec)
    kv_policy: KVCachePolicy = field(default_factory=KVCachePolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.model_name, "model_name")
        _require_name(self.hardware_name, "hardware_name")
        _require_schema_version(self.schema_version)
        _require_mapping(self.op_to_component, "op_to_component")
        _require_mapping(self.tensor_to_component, "tensor_to_component")
        _require_mapping(self.tensor_bytes, "tensor_bytes")
        _require_mapping(self.metadata, "metadata")
        if not isinstance(self.parallel, ParallelSpec):
            raise ValueError("parallel must be a ParallelSpec")
        if not isinstance(self.kv_policy, KVCachePolicy):
            raise ValueError("kv_policy must be a KVCachePolicy")
        for mapping_name, mapping in (
            ("op_to_component", self.op_to_component),
            ("tensor_to_component", self.tensor_to_component),
        ):
            for key, value in mapping.items():
                _require_name(key, "{} key".format(mapping_name))
                _require_name(value, "{} value".format(mapping_name))
        if "mtp" in self.op_to_component:
            raise ValueError(
                "op_to_component.mtp is not part of the V4 contract; map each "
                "typed MTP operator_id explicitly"
            )
        if "mtp_weights" in self.tensor_to_component:
            raise ValueError(
                "tensor_to_component.mtp_weights is not part of the V4 contract; "
                "map each typed MTP weight tensor explicitly"
            )
        for tensor_id, size in self.tensor_bytes.items():
            _require_name(tensor_id, "tensor_bytes key")
            _require_int(size, "tensor_bytes value")
        if "mtp_weights" in self.tensor_bytes:
            raise ValueError(
                "tensor_bytes.mtp_weights is not part of the V4 contract; "
                "declare each typed MTP weight tensor explicitly"
            )

@dataclass(frozen=True)
class RequestSpec:
    """One request in a trace-driven workload."""

    request_id: str
    arrival_ns: float
    prompt_tokens: int
    output_tokens: int
    priority: int = 0
    deadline_ns: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.request_id, "request_id")
        _require_schema_version(self.schema_version)
        _require_number(self.arrival_ns, "arrival_ns")
        _require_int(self.prompt_tokens, "prompt_tokens")
        _require_int(self.output_tokens, "output_tokens")
        _require_int(self.priority, "priority")
        _require_mapping(self.metadata, "metadata")
        if self.deadline_ns is not None:
            _require_number(self.deadline_ns, "deadline_ns")
            if self.deadline_ns < self.arrival_ns:
                raise ValueError("deadline_ns must not precede arrival_ns")


@dataclass(frozen=True)
class WorkloadSpec:
    """Trace-driven or synthetic inference workload description."""

    name: str
    requests: Tuple[RequestSpec, ...] = ()
    request_count: int = 1
    prompt_tokens: int = 0
    output_tokens: int = 0
    arrival_rate_rps: float = 0.0
    random_seed: int = 0
    scheduler: SchedulerSpec = field(default_factory=SchedulerSpec)
    mtp: Optional[MTPPolicy] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_name(self.name, "name")
        _require_schema_version(self.schema_version)
        _require_tuple(self.requests, "requests")
        if not all(isinstance(request, RequestSpec) for request in self.requests):
            raise ValueError("requests must contain RequestSpec values")
        _require_int(self.request_count, "request_count")
        _require_int(self.prompt_tokens, "prompt_tokens")
        _require_int(self.output_tokens, "output_tokens")
        _require_number(self.arrival_rate_rps, "arrival_rate_rps")
        _require_int(self.random_seed, "random_seed")
        _require_mapping(self.metadata, "metadata")
        if not isinstance(self.scheduler, SchedulerSpec):
            raise ValueError("scheduler must be a SchedulerSpec")
        if self.mtp is not None:
            if not isinstance(self.mtp, MTPPolicy):
                raise ValueError("mtp must be an MTPPolicy")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_id values must be unique within a workload")

    @property
    def effective_request_count(self) -> int:
        return len(self.requests) if self.requests else self.request_count

__all__ = [
    "ACTIVE_MEMORY_COMPONENT_KINDS",
    "OFFLOAD_STORAGE_COMPONENT_KINDS",
    "SCHEMA_VERSION",
    "STORAGE_COMPONENT_KINDS",
    "ComponentSpec",
    "HardwareSpec",
    "LayerSpec",
    "LinearAttentionSpec",
    "LinkSpec",
    "KVCachePolicy",
    "GraphLayerExecutionDescriptor",
    "MTPBranchSpec",
    "MTPExecutionDescriptor",
    "ModelGraphExecutionView",
    "MTPPolicy",
    "ModelSpec",
    "model_graph_execution_digest",
    "model_graph_execution_view",
    "build_model_graph_from_layer_specs",
    "model_graph_execution_layers",
    "ParallelSpec",
    "PlacementSpec",
    "PortSpec",
    "RankMappingSpec",
    "RequestSpec",
    "SchedulerSpec",
    "WorkloadSpec",
    "normalize_component_kind",
]
