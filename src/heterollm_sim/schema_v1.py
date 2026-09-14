"""Canonical, compiler-facing schema for HeteroLLM Simulator 1.1.

The :mod:`heterollm_sim.ir` objects form the V4 authoring schema. This module is
the independent, normalized boundary between authoring inputs and lowering
passes. It intentionally has no dependency on the authoring schema so future
frontends can produce the canonical representation directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import math
import re
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

from .serde import canonical_json, stable_hash, to_primitive


SCHEMA_V1_VERSION = "1.1"
MODEL_GRAPH_TRANSFORM_KINDS = ("reshape", "transpose", "concat", "split", "cast", "broadcast")
_MODEL_GRAPH_UNREPRESENTABLE_TRANSFORMS = {"concat", "split"}
_DTYPE_ALIASES = {
    "float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float32": "fp32",
    "float": "fp32",
    "fp32": "fp32",
    "float64": "fp64",
    "double": "fp64",
    "fp64": "fp64",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
    "boolean": "bool",
}


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(label))


def _optional_name(value: Optional[str], label: str) -> None:
    if value is not None:
        _name(value, label)


def _integer(value: int, label: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(label, minimum))


def _number(value: Real, label: str, minimum: float = 0.0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError("{} must be a number >= {}".format(label, minimum))


def _tuple(value: Tuple[Any, ...], label: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError("{} must be a tuple".format(label))


def _mapping(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a mapping".format(label))


def _unique(values: Tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("{} must be unique".format(label))


def _shape(value: Tuple[Any, ...], label: str) -> None:
    _tuple(value, label)
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, (int, str)):
            raise ValueError("{} dimensions must be positive integers or symbols".format(label))
        if isinstance(dimension, int) and dimension < 1:
            raise ValueError("{} integer dimensions must be >= 1".format(label))
        if isinstance(dimension, str) and not dimension.strip():
            raise ValueError("{} symbolic dimensions must not be empty".format(label))


def _canonical_dtype(value: Any) -> str:
    text = str(value if value is not None else "unknown").strip().lower().replace("-", "_").replace(" ", "_") or "unknown"
    return _DTYPE_ALIASES.get(text, text)


def _canonical_layout(value: Any) -> str:
    return str(value if value is not None else "logical").strip().lower().replace("-", "_").replace(" ", "_") or "logical"


def _normalized_dimension(value: Any, label: str, *, allow_inferred: bool = False) -> Any:
    if isinstance(value, bool):
        raise ValueError("{} 的维度不能是布尔值".format(label))
    if isinstance(value, int):
        if allow_inferred and value == -1:
            return value
        if value >= 1:
            return value
        raise ValueError("{} 的整数维度必须 >= 1".format(label))
    if isinstance(value, Real):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return _normalized_dimension(int(number), label, allow_inferred=allow_inferred)
        raise ValueError("{} 的数值维度必须是整数".format(label))
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError("{} 的符号维度不能为空".format(label))
    if re.fullmatch(r"-?\d+", text):
        return _normalized_dimension(int(text), label, allow_inferred=allow_inferred)
    return text


def _normalized_shape(value: Any, label: str, *, allow_inferred: bool = False) -> Tuple[Any, ...]:
    if isinstance(value, str):
        raw = [item for item in re.split(r"[×x,\s]+", value) if item]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError("{} 必须是 shape 数组或字符串".format(label))
    return tuple(_normalized_dimension(item, label, allow_inferred=allow_inferred) for item in raw)


def _contract(tensor: "TensorValue") -> Tuple[str, Tuple[Any, ...], str]:
    return (
        _canonical_dtype(tensor.dtype),
        _normalized_shape(tensor.shape, "张量 {} shape".format(tensor.tensor_id)),
        _canonical_layout(tensor.layout),
    )


def _shape_text(shape: Tuple[Any, ...]) -> str:
    return " × ".join(str(item) for item in shape) if shape else "标量"


def _contract_text(contract: Tuple[str, Tuple[Any, ...], str]) -> str:
    return "{} [{}] / {}".format(contract[0], _shape_text(contract[1]), contract[2])


def _attribute_value(attributes: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in attributes:
            return attributes[name]
    return None


def _is_wildcard(value: Any) -> bool:
    return value is None or value == "" or value == "*" or str(value).lower() == "unknown"


def _is_symbol(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is not None and not _is_wildcard(value)


def _resolve_binding(value: Any, bindings: Dict[str, Any]) -> Any:
    current = value
    seen = set()
    while _is_symbol(current) and current in bindings and current not in seen:
        seen.add(current)
        current = bindings[current]
    return current


def _bind_dimension(expected: Any, actual: Any, bindings: Dict[str, Any]) -> Tuple[bool, Any]:
    left = _resolve_binding(expected, bindings)
    right = _resolve_binding(actual, bindings)
    if _is_wildcard(left) or _is_wildcard(right):
        return True, right if _is_wildcard(left) else left
    if left == right:
        return True, left
    if _is_symbol(left):
        bindings[left] = right
        return True, right
    if _is_symbol(right):
        bindings[right] = left
        return True, left
    return False, left


def _numeric_product(shape: Tuple[Any, ...]) -> Optional[int]:
    if not all(isinstance(item, int) and item >= 0 for item in shape):
        return None
    total = 1
    for item in shape:
        total *= item
    return total


def _integer_sequence(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("{} 必须是整数数组".format(label))
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("{} 不能包含布尔值".format(label))
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, Real) and math.isfinite(float(item)) and float(item).is_integer():
            result.append(int(item))
        elif isinstance(item, str) and re.fullmatch(r"-?\d+", item.strip()):
            result.append(int(item.strip()))
        else:
            raise ValueError("{} 只能包含整数".format(label))
    return tuple(result)


@dataclass(frozen=True)
class Origin:
    """One traceable source used to derive a canonical object."""

    source_kind: str
    source_id: str
    source_schema_version: Optional[str] = None
    transform: str = "identity"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.source_kind, "source_kind")
        _name(self.source_id, "source_id")
        _optional_name(self.source_schema_version, "source_schema_version")
        _name(self.transform, "transform")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class HardwarePort:
    port_id: str
    protocol: str
    role: str
    direction: str
    bandwidth_gbps: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in ("port_id", "protocol", "role", "direction"):
            _name(getattr(self, label), label)
        _number(self.bandwidth_gbps, "bandwidth_gbps")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class HardwareNode:
    node_id: str
    kind: str
    ports: Tuple[HardwarePort, ...] = ()
    capacity_bytes: int = 0
    peak_ops_per_s: float = 0.0
    read_bandwidth_gbps: float = 0.0
    write_bandwidth_gbps: float = 0.0
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.node_id, "node_id")
        _name(self.kind, "kind")
        _tuple(self.ports, "ports")
        if not all(isinstance(item, HardwarePort) for item in self.ports):
            raise ValueError("ports must contain HardwarePort values")
        _unique(tuple(item.port_id for item in self.ports), "port ids within node")
        if isinstance(self.capacity_bytes, bool) or not isinstance(self.capacity_bytes, int):
            raise ValueError("capacity_bytes must be an integer")
        for label in ("peak_ops_per_s", "read_bandwidth_gbps", "write_bandwidth_gbps"):
            _number(getattr(self, label), label)
        _mapping(self.attributes, "attributes")
        _tuple(self.provenance, "provenance")


@dataclass(frozen=True)
class HardwareLink:
    link_id: str
    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str
    protocol: str
    bandwidth_gbps: float = 0.0
    latency_ns: float = 0.0
    bidirectional: bool = True
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        for label in (
            "link_id",
            "source_node_id",
            "source_port_id",
            "target_node_id",
            "target_port_id",
            "protocol",
        ):
            _name(getattr(self, label), label)
        _number(self.bandwidth_gbps, "bandwidth_gbps")
        _number(self.latency_ns, "latency_ns")
        if not isinstance(self.bidirectional, bool):
            raise ValueError("bidirectional must be boolean")
        _mapping(self.attributes, "attributes")
        _tuple(self.provenance, "provenance")


@dataclass(frozen=True)
class HardwareGraph:
    graph_id: str
    nodes: Tuple[HardwareNode, ...]
    links: Tuple[HardwareLink, ...] = ()
    require_connected: bool = True
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.graph_id, "graph_id")
        _tuple(self.nodes, "nodes")
        _tuple(self.links, "links")
        if not self.nodes or not all(isinstance(item, HardwareNode) for item in self.nodes):
            raise ValueError("nodes must contain HardwareNode values")
        if not all(isinstance(item, HardwareLink) for item in self.links):
            raise ValueError("links must contain HardwareLink values")
        _unique(tuple(item.node_id for item in self.nodes), "hardware node ids")
        _unique(tuple(item.link_id for item in self.links), "hardware link ids")
        if not isinstance(self.require_connected, bool):
            raise ValueError("require_connected must be boolean")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class TensorValue:
    tensor_id: str
    role: str
    logical_bytes: Optional[int] = None
    producer_operator_id: Optional[str] = None
    consumer_operator_ids: Tuple[str, ...] = ()
    dtype: str = "unknown"
    shape: Tuple[Any, ...] = ()
    layout: str = "logical"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.tensor_id, "tensor_id")
        _name(self.role, "role")
        if self.logical_bytes is not None:
            _integer(self.logical_bytes, "logical_bytes")
        _optional_name(self.producer_operator_id, "producer_operator_id")
        _tuple(self.consumer_operator_ids, "consumer_operator_ids")
        _unique(self.consumer_operator_ids, "consumer_operator_ids")
        _name(self.dtype, "dtype")
        _shape(self.shape, "shape")
        _name(self.layout, "layout")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class OperatorPort:
    """A typed authoring port bound to one logical tensor."""

    port_id: str
    direction: str
    tensor_id: str
    dtype: str
    shape: Tuple[Any, ...]
    layout: str = "logical"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.port_id, "port_id")
        if self.direction not in {"input", "output", "weight"}:
            raise ValueError("direction must be input, output, or weight")
        _name(self.tensor_id, "tensor_id")
        _name(self.dtype, "dtype")
        _shape(self.shape, "shape")
        _name(self.layout, "layout")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class TensorTransform:
    """An explicit, user-visible tensor contract conversion."""

    transform_id: str
    kind: str
    input_tensor_id: str
    output_tensor_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.transform_id, "transform_id")
        _name(self.kind, "kind")
        if self.kind not in MODEL_GRAPH_TRANSFORM_KINDS:
            raise ValueError("显式变换 {} 使用了不支持的类型 {}；仅支持 {}".format(
                self.transform_id,
                self.kind,
                "、".join(MODEL_GRAPH_TRANSFORM_KINDS),
            ))
        _name(self.input_tensor_id, "input_tensor_id")
        _name(self.output_tensor_id, "output_tensor_id")
        if self.input_tensor_id == self.output_tensor_id:
            raise ValueError("显式变换的输入和输出张量不能相同")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class OperatorNode:
    operator_id: str
    op_kind: str
    sequence_index: int
    layer_id: Optional[str] = None
    input_tensor_ids: Tuple[str, ...] = ()
    output_tensor_ids: Tuple[str, ...] = ()
    weight_tensor_ids: Tuple[str, ...] = ()
    ports: Tuple[OperatorPort, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.operator_id, "operator_id")
        _name(self.op_kind, "op_kind")
        _integer(self.sequence_index, "sequence_index")
        _optional_name(self.layer_id, "layer_id")
        for label in ("input_tensor_ids", "output_tensor_ids", "weight_tensor_ids"):
            value = getattr(self, label)
            _tuple(value, label)
            _unique(value, label)
        _tuple(self.ports, "ports")
        if not all(isinstance(item, OperatorPort) for item in self.ports):
            raise ValueError("ports must contain OperatorPort values")
        _unique(tuple(item.port_id for item in self.ports), "operator port ids")
        _mapping(self.parameters, "parameters")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class SubOperator:
    """One runtime primitive owned by an authoritative authoring operator."""

    sub_operator_id: str
    parent_operator_id: str
    operator_class: str
    expanded_operator_id: Optional[str] = None
    layer_id: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.sub_operator_id, "sub_operator_id")
        _name(self.parent_operator_id, "parent_operator_id")
        _name(self.operator_class, "operator_class")
        _optional_name(self.expanded_operator_id, "expanded_operator_id")
        _optional_name(self.layer_id, "layer_id")
        _mapping(self.attributes, "attributes")
        _tuple(self.provenance, "provenance")
        if not all(isinstance(item, Origin) for item in self.provenance):
            raise ValueError("provenance must contain Origin values")


@dataclass(frozen=True)
class ModelGraph:
    graph_id: str
    operators: Tuple[OperatorNode, ...]
    tensors: Tuple[TensorValue, ...]
    source_operators: Tuple[OperatorNode, ...] = ()
    sub_operators: Tuple[SubOperator, ...] = ()
    transforms: Tuple[TensorTransform, ...] = ()
    executable: bool = True
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.graph_id, "graph_id")
        _tuple(self.operators, "operators")
        _tuple(self.tensors, "tensors")
        if not self.operators or not all(isinstance(item, OperatorNode) for item in self.operators):
            raise ValueError("operators must contain OperatorNode values")
        if not all(isinstance(item, TensorValue) for item in self.tensors):
            raise ValueError("tensors must contain TensorValue values")
        _tuple(self.source_operators, "source_operators")
        if not all(isinstance(item, OperatorNode) for item in self.source_operators):
            raise ValueError("source_operators must contain OperatorNode values")
        _tuple(self.sub_operators, "sub_operators")
        if not all(isinstance(item, SubOperator) for item in self.sub_operators):
            raise ValueError("sub_operators must contain SubOperator values")
        _tuple(self.transforms, "transforms")
        if not all(isinstance(item, TensorTransform) for item in self.transforms):
            raise ValueError("transforms must contain TensorTransform values")
        if not isinstance(self.executable, bool):
            raise ValueError("executable must be boolean")
        _unique(tuple(item.operator_id for item in self.operators), "operator ids")
        _unique(tuple(item.operator_id for item in self.source_operators), "source operator ids")
        _unique(tuple(item.sub_operator_id for item in self.sub_operators), "sub-operator ids")
        _unique(tuple(item.tensor_id for item in self.tensors), "model tensor ids")
        _unique(tuple(item.transform_id for item in self.transforms), "transform ids")
        _mapping(self.attributes, "attributes")
        source_ids = {item.operator_id for item in self.source_operators}
        expanded_ids = {item.operator_id for item in self.operators}
        if self.sub_operators and not source_ids:
            raise ValueError("sub_operators require an authoritative source_operators registry")
        for sub_operator in self.sub_operators:
            if sub_operator.parent_operator_id not in source_ids:
                raise ValueError(
                    "sub-operator {} references unknown source parent {}".format(
                        sub_operator.sub_operator_id,
                        sub_operator.parent_operator_id,
                    )
                )
            if (
                sub_operator.expanded_operator_id is not None
                and sub_operator.expanded_operator_id not in expanded_ids
            ):
                raise ValueError(
                    "sub-operator {} references unknown expanded operator {}".format(
                        sub_operator.sub_operator_id,
                        sub_operator.expanded_operator_id,
                    )
                )
        if source_ids:
            for operator in self.operators:
                authoring_id = operator.attributes.get("authoring_operator_id")
                if authoring_id is not None and str(authoring_id) not in source_ids:
                    raise ValueError(
                        "expanded operator {} references unknown authoring operator {}".format(
                            operator.operator_id,
                            authoring_id,
                        )
                    )
            expanded_by_id = {item.operator_id: item for item in self.operators}
            for sub_operator in self.sub_operators:
                if sub_operator.expanded_operator_id is None:
                    continue
                expanded = expanded_by_id[sub_operator.expanded_operator_id]
                authoring_id = expanded.attributes.get("authoring_operator_id")
                if authoring_id != sub_operator.parent_operator_id:
                    raise ValueError(
                        "sub-operator {} parent {} disagrees with expanded operator {} authoring parent {}".format(
                            sub_operator.sub_operator_id,
                            sub_operator.parent_operator_id,
                            expanded.operator_id,
                            authoring_id,
                        )
                    )
        _validate_model_graph_contracts(self)


def _validate_model_graph_contracts(graph: ModelGraph) -> None:
    tensors = {item.tensor_id: item for item in graph.tensors}
    operators = {item.operator_id: item for item in graph.operators}
    expected_producers: Dict[str, str] = {}
    expected_consumers: Dict[str, set] = {item.tensor_id: set() for item in graph.tensors}
    for operator in graph.operators:
        declared = {
            "input": operator.input_tensor_ids,
            "output": operator.output_tensor_ids,
            "weight": operator.weight_tensor_ids,
        }
        port_refs = {
            direction: tuple(port.tensor_id for port in operator.ports if port.direction == direction)
            for direction in ("input", "output", "weight")
        }
        for direction in ("input", "output", "weight"):
            if port_refs[direction] != declared[direction]:
                raise ValueError(
                    "组件 {} 的 {} 张量列表与端口定义不一致：端口为 [{}]，列表为 [{}]".format(
                        operator.operator_id,
                        direction,
                        ", ".join(port_refs[direction]),
                        ", ".join(declared[direction]),
                    )
                )
        for port in operator.ports:
            tensor = tensors.get(port.tensor_id)
            if tensor is None:
                raise ValueError("组件 {} 的端口 {} 引用了不存在的张量 {}".format(operator.operator_id, port.port_id, port.tensor_id))
            if (port.dtype, port.shape, port.layout) != (tensor.dtype, tensor.shape, tensor.layout):
                raise ValueError(
                    "组件 {} 端口 {} 维度/类型不匹配：期望 {} {} {}，实际 {} {} {}".format(
                        operator.operator_id, port.port_id,
                        port.dtype, list(port.shape), port.layout,
                        tensor.dtype, list(tensor.shape), tensor.layout,
                    )
                )
        for tensor_id in operator.input_tensor_ids + operator.output_tensor_ids + operator.weight_tensor_ids:
            if tensor_id not in tensors:
                raise ValueError("组件 {} 引用了不存在的张量 {}".format(operator.operator_id, tensor_id))
        for tensor_id in operator.output_tensor_ids:
            producer = expected_producers.get(tensor_id)
            if producer is not None and producer != operator.operator_id:
                raise ValueError("张量 {} 不能同时由组件 {} 和 {} 生产".format(tensor_id, producer, operator.operator_id))
            expected_producers[tensor_id] = operator.operator_id
        for tensor_id in operator.input_tensor_ids + operator.weight_tensor_ids:
            expected_consumers.setdefault(tensor_id, set()).add(operator.operator_id)
    for tensor in graph.tensors:
        if tensor.producer_operator_id is not None and tensor.producer_operator_id not in operators:
            raise ValueError("张量 {} 的生产组件不存在".format(tensor.tensor_id))
        missing = [item for item in tensor.consumer_operator_ids if item not in operators]
        if missing:
            raise ValueError("张量 {} 的消费组件不存在：{}".format(tensor.tensor_id, ", ".join(missing)))
        expected_producer = expected_producers.get(tensor.tensor_id)
        if tensor.producer_operator_id != expected_producer:
            if expected_producer is None:
                raise ValueError("张量 {} 的生产组件索引已过期：{} 没有匹配的 output 端口".format(
                    tensor.tensor_id,
                    tensor.producer_operator_id,
                ))
            if tensor.producer_operator_id is None:
                raise ValueError("张量 {} 缺少生产组件索引：{} 的 output 端口声明了该张量".format(
                    tensor.tensor_id,
                    expected_producer,
                ))
            raise ValueError("张量 {} 的生产组件索引不一致：期望 {}，实际 {}".format(
                tensor.tensor_id,
                expected_producer,
                tensor.producer_operator_id,
            ))
        expected = expected_consumers.get(tensor.tensor_id, set())
        actual = set(tensor.consumer_operator_ids)
        stale = sorted(actual - expected)
        if stale:
            raise ValueError("张量 {} 的消费组件索引已过期：{} 没有匹配的 input/weight 端口".format(
                tensor.tensor_id,
                ", ".join(stale),
            ))
        absent = sorted(expected - actual)
        if absent:
            raise ValueError("张量 {} 缺少消费组件索引：{} 的 input/weight 端口声明了该张量".format(
                tensor.tensor_id,
                ", ".join(absent),
            ))
    for transform in graph.transforms:
        _validate_tensor_transform_contract(transform, tensors)
    _validate_typed_mtp_contracts(graph, tensors)
    _validate_model_graph_dag(graph, tensors)


def _validate_typed_mtp_contracts(
    graph: ModelGraph, tensors: Mapping[str, TensorValue]
) -> None:
    for operator in graph.operators:
        if operator.op_kind not in {"mtp_prediction_layer", "mtp_aux_head"}:
            continue
        if (
            len(operator.input_tensor_ids) != 1
            or len(operator.output_tensor_ids) != 1
            or len(operator.weight_tensor_ids) != 1
        ):
            raise ValueError(
                "typed MTP 组件 {} 必须各声明一个 input/output/weight 张量".format(
                    operator.operator_id
                )
            )
        input_tensor = tensors[operator.input_tensor_ids[0]]
        output_tensor = tensors[operator.output_tensor_ids[0]]
        weight_tensor = tensors[operator.weight_tensor_ids[0]]
        if weight_tensor.role != "weight" or weight_tensor.logical_bytes is None:
            raise ValueError(
                "typed MTP 权重张量 {} 必须使用 weight role 并显式声明 logical_bytes".format(
                    weight_tensor.tensor_id
                )
            )
        parameters = operator.parameters
        hidden_size = parameters.get("hidden_size")
        weight_bytes = parameters.get("weight_bytes")
        _integer(hidden_size, "typed MTP hidden_size", minimum=1)
        _integer(weight_bytes, "typed MTP weight_bytes")
        if weight_bytes != weight_tensor.logical_bytes:
            raise ValueError(
                "typed MTP 组件 {} 的 weight_bytes 与权重张量 logical_bytes 不一致".format(
                    operator.operator_id
                )
            )
        hidden_contract = (
            input_tensor.dtype,
            ("B", "T", hidden_size),
            input_tensor.layout,
        )
        if _contract(input_tensor) != hidden_contract:
            raise ValueError(
                "typed MTP 组件 {} 输入合同不匹配：期望 {}，实际 {}".format(
                    operator.operator_id,
                    _contract_text(hidden_contract),
                    _contract_text(_contract(input_tensor)),
                )
            )
        if operator.op_kind == "mtp_prediction_layer":
            prediction_index = parameters.get("prediction_index")
            _integer(
                prediction_index,
                "typed MTP prediction_index",
            )
            expected_output = hidden_contract
            expected_weight = (
                input_tensor.dtype,
                (hidden_size, hidden_size),
                input_tensor.layout,
            )
        else:
            vocabulary_size = parameters.get("vocabulary_size")
            _integer(
                vocabulary_size,
                "typed MTP vocabulary_size",
                minimum=1,
            )
            expected_output = (
                _canonical_dtype(output_tensor.dtype),
                ("B", "T", "V"),
                input_tensor.layout,
            )
            expected_weight = (
                input_tensor.dtype,
                (hidden_size, "V"),
                input_tensor.layout,
            )
        for label, tensor, expected in (
            ("输出", output_tensor, expected_output),
            ("权重", weight_tensor, expected_weight),
        ):
            actual = _contract(tensor)
            if actual != expected:
                raise ValueError(
                    "typed MTP 组件 {} {}合同不匹配：期望 {}，实际 {}".format(
                        operator.operator_id,
                        label,
                        _contract_text(expected),
                        _contract_text(actual),
                    )
                )


def _validate_tensor_transform_contract(transform: TensorTransform, tensors: Mapping[str, TensorValue]) -> None:
    if transform.input_tensor_id not in tensors or transform.output_tensor_id not in tensors:
        raise ValueError("显式变换 {} 引用了不存在的张量".format(transform.transform_id))
    hidden_refs = {"input_tensor_ids", "output_tensor_ids", "inputs", "outputs"}.intersection(transform.attributes)
    if hidden_refs:
        raise ValueError("显式变换 {} 不能在 attributes 中暗藏张量引用：{}".format(
            transform.transform_id,
            "、".join(sorted(hidden_refs)),
        ))
    if transform.kind in _MODEL_GRAPH_UNREPRESENTABLE_TRANSFORMS:
        raise ValueError(
            "显式变换 {} 的 {} 需要多输入或多输出；当前 schema 只有单数 input_tensor_id/output_tensor_id，无法完整表达，已拒绝".format(
                transform.transform_id,
                transform.kind,
            )
        )
    input_contract = _contract(tensors[transform.input_tensor_id])
    expected_output = _infer_single_tensor_transform_output(transform, input_contract)
    actual_output = _contract(tensors[transform.output_tensor_id])
    if actual_output != expected_output:
        raise ValueError("显式变换 {} 输出张量 {} 合同不匹配：期望 {}，实际 {}".format(
            transform.transform_id,
            transform.output_tensor_id,
            _contract_text(expected_output),
            _contract_text(actual_output),
        ))


def _infer_single_tensor_transform_output(
    transform: TensorTransform,
    input_contract: Tuple[str, Tuple[Any, ...], str],
) -> Tuple[str, Tuple[Any, ...], str]:
    dtype, shape, layout = input_contract
    attributes = transform.attributes
    if transform.kind == "reshape":
        raw = _attribute_value(attributes, "shape", "target_shape")
        if raw is None:
            raise ValueError("显式变换 {} 的 reshape 必须声明目标 shape".format(transform.transform_id))
        target = list(_normalized_shape(raw, "显式变换 {} reshape 目标 shape".format(transform.transform_id), allow_inferred=True))
        inferred = [index for index, item in enumerate(target) if item == -1]
        if len(inferred) > 1:
            raise ValueError("显式变换 {} 的 reshape 最多只能包含一个 -1 推导维度".format(transform.transform_id))
        input_count = _numeric_product(shape)
        if inferred:
            if input_count is None:
                raise ValueError("显式变换 {} 的 reshape 含有 -1，但输入元素数量含符号维度，无法后端推导".format(transform.transform_id))
            known = tuple(item for index, item in enumerate(target) if index != inferred[0])
            known_count = _numeric_product(known)
            if not known_count or input_count % known_count:
                raise ValueError("显式变换 {} 的 reshape 元素数量不匹配：期望可整除 {}，实际 {}".format(
                    transform.transform_id,
                    known_count,
                    input_count,
                ))
            target[inferred[0]] = input_count // known_count
        target_shape = tuple(target)
        target_count = _numeric_product(target_shape)
        if input_count is not None and target_count is not None and input_count != target_count:
            raise ValueError("显式变换 {} 的 reshape 元素数量不匹配：期望 {}，实际 {}".format(
                transform.transform_id,
                target_count,
                input_count,
            ))
        return dtype, target_shape, layout
    if transform.kind == "transpose":
        raw = _attribute_value(attributes, "permutation", "perm")
        rank = len(shape)
        permutation = tuple(reversed(range(rank))) if raw is None else _integer_sequence(raw, "显式变换 {} transpose permutation".format(transform.transform_id))
        if len(permutation) != rank or len(set(permutation)) != rank or any(item < 0 or item >= rank for item in permutation):
            raise ValueError("显式变换 {} 的 transpose permutation 必须是 0..{} 的完整排列".format(
                transform.transform_id,
                max(0, rank - 1),
            ))
        output_layout = _canonical_layout(attributes.get("layout") or "{}_transposed".format(layout))
        return dtype, tuple(shape[index] for index in permutation), output_layout
    if transform.kind == "cast":
        raw = _attribute_value(attributes, "dtype", "to_dtype")
        output_dtype = _canonical_dtype(raw)
        if _is_wildcard(output_dtype):
            raise ValueError("显式变换 {} 的 cast 必须声明目标 dtype".format(transform.transform_id))
        return output_dtype, shape, layout
    if transform.kind == "broadcast":
        raw = _attribute_value(attributes, "shape", "target_shape")
        if raw is None:
            raise ValueError("显式变换 {} 的 broadcast 必须声明目标 shape".format(transform.transform_id))
        target = _normalized_shape(raw, "显式变换 {} broadcast 目标 shape".format(transform.transform_id))
        if len(target) < len(shape):
            raise ValueError("显式变换 {} 的 broadcast 目标维度必须不少于输入：期望至少 {} 维，实际 {} 维".format(
                transform.transform_id,
                len(shape),
                len(target),
            ))
        padded = tuple(1 for _ in range(len(target) - len(shape))) + shape
        bindings: Dict[str, Any] = {}
        for index, (expected, actual) in enumerate(zip(target, padded), start=1):
            if actual == 1:
                continue
            ok, _ = _bind_dimension(expected, actual, bindings)
            if not ok:
                raise ValueError("显式变换 {} 的 broadcast 第 {} 维不匹配：期望 {}，实际 {}".format(
                    transform.transform_id,
                    index,
                    _shape_text(target),
                    _shape_text(shape),
                ))
        return dtype, tuple(_resolve_binding(item, bindings) for item in target), layout
    raise ValueError("显式变换 {} 使用了不支持的类型 {}".format(transform.transform_id, transform.kind))


def _validate_model_graph_dag(graph: ModelGraph, tensors: Mapping[str, TensorValue]) -> None:
    nodes = []
    node_set = set()

    def add_node(node_id: str) -> None:
        if node_id not in node_set:
            node_set.add(node_id)
            nodes.append(node_id)

    for operator in graph.operators:
        add_node("op:" + operator.operator_id)
    for tensor in graph.tensors:
        add_node("tensor:" + tensor.tensor_id)

    adjacency = {node_id: set() for node_id in nodes}

    def add_edge(source: str, target: str) -> None:
        add_node(source)
        add_node(target)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())

    for tensor in graph.tensors:
        tensor_node = "tensor:" + tensor.tensor_id
        if tensor.producer_operator_id is not None:
            add_edge("op:" + tensor.producer_operator_id, tensor_node)
        for consumer in tensor.consumer_operator_ids:
            add_edge(tensor_node, "op:" + consumer)
    for transform in graph.transforms:
        input_tensor = tensors[transform.input_tensor_id]
        output_tensor = tensors[transform.output_tensor_id]
        add_edge("tensor:" + input_tensor.tensor_id, "tensor:" + output_tensor.tensor_id)

    indegree = {node_id: 0 for node_id in nodes}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1
    order = {node_id: index for index, node_id in enumerate(nodes)}
    queue = sorted((node_id for node_id in nodes if indegree.get(node_id, 0) == 0), key=order.get)
    visited = []
    while queue:
        current = queue.pop(0)
        visited.append(current)
        for target in sorted(adjacency.get(current, ()), key=order.get):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort(key=order.get)
    if len(visited) != len(nodes):
        visited_set = set(visited)
        cycle_nodes = [node_id for node_id in nodes if node_id not in visited_set]
        labels = [node_id[3:] for node_id in cycle_nodes if node_id.startswith("op:")]
        if not labels:
            labels = [node_id.split(":", 1)[1] for node_id in cycle_nodes]
        raise ValueError("模型组件图必须是 DAG；检测到环：{}。".format("、".join(labels)))


@dataclass(frozen=True)
class RequestNode:
    request_id: str
    arrival_ns: float
    prompt_tokens: int
    output_tokens: int
    priority: int = 0
    deadline_ns: Optional[float] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.request_id, "request_id")
        _number(self.arrival_ns, "arrival_ns")
        _integer(self.prompt_tokens, "prompt_tokens")
        _integer(self.output_tokens, "output_tokens")
        _integer(self.priority, "priority")
        if self.deadline_ns is not None:
            _number(self.deadline_ns, "deadline_ns")
            if self.deadline_ns < self.arrival_ns:
                raise ValueError("deadline_ns must not precede arrival_ns")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class WorkloadGraph:
    graph_id: str
    requests: Tuple[RequestNode, ...]
    scheduler: Mapping[str, Any] = field(default_factory=dict)
    policy: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.graph_id, "graph_id")
        _tuple(self.requests, "requests")
        if not all(isinstance(item, RequestNode) for item in self.requests):
            raise ValueError("requests must contain RequestNode values")
        _unique(tuple(item.request_id for item in self.requests), "request ids")
        for label in ("scheduler", "policy", "attributes"):
            _mapping(getattr(self, label), label)


@dataclass(frozen=True)
class RankPlan:
    rank_id: int
    tp_rank: int
    pp_rank: int
    ep_rank: int
    compute_node_id: str
    memory_node_id: Optional[str] = None
    cim_node_id: Optional[str] = None
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        for label in ("rank_id", "tp_rank", "pp_rank", "ep_rank"):
            _integer(getattr(self, label), label)
        _name(self.compute_node_id, "compute_node_id")
        _optional_name(self.memory_node_id, "memory_node_id")
        _optional_name(self.cim_node_id, "cim_node_id")


@dataclass(frozen=True)
class StageAssignment:
    layer_id: str
    stage_id: int
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.layer_id, "layer_id")
        _integer(self.stage_id, "stage_id")


@dataclass(frozen=True)
class ParallelPlan:
    tp_degree: int
    pp_degree: int
    ep_degree: int
    ranks: Tuple[RankPlan, ...]
    layer_stages: Tuple[StageAssignment, ...]
    collective_algorithm: str = "auto"
    routing_policy: str = "lowest_latency"
    allow_padding: bool = True
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        for label in ("tp_degree", "pp_degree", "ep_degree"):
            _integer(getattr(self, label), label, minimum=1)
        _tuple(self.ranks, "ranks")
        _tuple(self.layer_stages, "layer_stages")
        if not all(isinstance(item, RankPlan) for item in self.ranks):
            raise ValueError("ranks must contain RankPlan values")
        if not all(isinstance(item, StageAssignment) for item in self.layer_stages):
            raise ValueError("layer_stages must contain StageAssignment values")
        _name(self.collective_algorithm, "collective_algorithm")
        _name(self.routing_policy, "routing_policy")
        if not isinstance(self.allow_padding, bool):
            raise ValueError("allow_padding must be boolean")

    @property
    def world_size(self) -> int:
        return self.tp_degree * self.pp_degree * self.ep_degree


@dataclass(frozen=True)
class OperatorTarget:
    operator_id: str
    rank_id: int
    component_id: str
    source_key: str
    derivation: str
    cost_profile_kind: Optional[str] = None
    cost_profile_id: Optional[str] = None
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.operator_id, "operator_id")
        _integer(self.rank_id, "rank_id")
        _name(self.component_id, "component_id")
        _name(self.source_key, "source_key")
        _name(self.derivation, "derivation")
        _optional_name(self.cost_profile_kind, "cost_profile_kind")
        _optional_name(self.cost_profile_id, "cost_profile_id")
        if (self.cost_profile_kind is None) != (self.cost_profile_id is None):
            raise ValueError(
                "cost_profile_kind and cost_profile_id must be set together"
            )
        _tuple(self.provenance, "provenance")
        if not all(isinstance(item, Origin) for item in self.provenance):
            raise ValueError("provenance must contain Origin values")


@dataclass(frozen=True)
class SubOperatorTarget:
    """One rank-aware placement for a formal runtime primitive."""

    sub_operator_id: str
    rank_id: int
    component_id: str
    source_key: str
    derivation: str
    cost_profile_kind: Optional[str] = None
    cost_profile_id: Optional[str] = None
    parent_fallback_operator_id: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.sub_operator_id, "sub_operator_id")
        _integer(self.rank_id, "rank_id")
        _name(self.component_id, "component_id")
        _name(self.source_key, "source_key")
        _name(self.derivation, "derivation")
        _optional_name(self.cost_profile_kind, "cost_profile_kind")
        _optional_name(self.cost_profile_id, "cost_profile_id")
        if (self.cost_profile_kind is None) != (self.cost_profile_id is None):
            raise ValueError(
                "cost_profile_kind and cost_profile_id must be set together"
            )
        _optional_name(
            self.parent_fallback_operator_id,
            "parent_fallback_operator_id",
        )
        _mapping(self.attributes, "attributes")
        _tuple(self.provenance, "provenance")
        if not all(isinstance(item, Origin) for item in self.provenance):
            raise ValueError("provenance must contain Origin values")


@dataclass(frozen=True)
class TensorShard:
    shard_id: str
    tensor_id: str
    shard_index: int
    shard_count: int
    logical_bytes: Optional[int] = None
    axis: Optional[int] = None
    derivation: str = "explicit_whole_tensor"
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.shard_id, "shard_id")
        _name(self.tensor_id, "tensor_id")
        _integer(self.shard_index, "shard_index")
        _integer(self.shard_count, "shard_count", minimum=1)
        if self.shard_index >= self.shard_count:
            raise ValueError("shard_index must be less than shard_count")
        if self.logical_bytes is not None:
            _integer(self.logical_bytes, "logical_bytes")
        if self.axis is not None:
            _integer(self.axis, "axis")
        _name(self.derivation, "derivation")


@dataclass(frozen=True)
class TensorReplica:
    replica_id: str
    tensor_id: str
    shard_id: str
    component_id: str
    rank_id: Optional[int] = None
    physical_bytes: Optional[int] = None
    residency: str = "unspecified"
    derivation: str = "authoring_explicit"
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        for label in ("replica_id", "tensor_id", "shard_id", "component_id"):
            _name(getattr(self, label), label)
        if self.rank_id is not None:
            _integer(self.rank_id, "rank_id")
        if self.physical_bytes is not None:
            _integer(self.physical_bytes, "physical_bytes")
        _name(self.residency, "residency")
        _name(self.derivation, "derivation")


@dataclass(frozen=True)
class TensorPlan:
    tensor_id: str
    logical_bytes: Optional[int]
    shards: Tuple[TensorShard, ...]
    replicas: Tuple[TensorReplica, ...]
    source_tensor_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _name(self.tensor_id, "tensor_id")
        if self.logical_bytes is not None:
            _integer(self.logical_bytes, "logical_bytes")
        _tuple(self.shards, "shards")
        _tuple(self.replicas, "replicas")
        if not all(isinstance(item, TensorShard) for item in self.shards):
            raise ValueError("shards must contain TensorShard values")
        if not all(isinstance(item, TensorReplica) for item in self.replicas):
            raise ValueError("replicas must contain TensorReplica values")
        _unique(tuple(item.shard_id for item in self.shards), "shard ids within tensor")
        _unique(tuple(item.replica_id for item in self.replicas), "replica ids within tensor")
        _name(self.source_tensor_id, "source_tensor_id")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class PlacementPlan:
    operator_targets: Tuple[OperatorTarget, ...]
    tensor_plans: Tuple[TensorPlan, ...]
    sub_operator_targets: Tuple[SubOperatorTarget, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: Tuple[Origin, ...] = ()

    def __post_init__(self) -> None:
        _tuple(self.operator_targets, "operator_targets")
        _tuple(self.tensor_plans, "tensor_plans")
        _tuple(self.sub_operator_targets, "sub_operator_targets")
        if not all(isinstance(item, OperatorTarget) for item in self.operator_targets):
            raise ValueError("operator_targets must contain OperatorTarget values")
        if not all(
            isinstance(item, SubOperatorTarget)
            for item in self.sub_operator_targets
        ):
            raise ValueError(
                "sub_operator_targets must contain SubOperatorTarget values"
            )
        if not all(isinstance(item, TensorPlan) for item in self.tensor_plans):
            raise ValueError("tensor_plans must contain TensorPlan values")
        pairs = tuple((item.operator_id, item.rank_id) for item in self.operator_targets)
        if len(pairs) != len(set(pairs)):
            raise ValueError("operator targets must be unique per operator and rank")
        sub_pairs = tuple(
            (item.sub_operator_id, item.rank_id)
            for item in self.sub_operator_targets
        )
        if len(sub_pairs) != len(set(sub_pairs)):
            raise ValueError(
                "sub-operator targets must be unique per sub-operator and rank"
            )
        _unique(tuple(item.tensor_id for item in self.tensor_plans), "tensor plan ids")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class CompilationDiagnostic:
    severity: str
    code: str
    message: str
    source_ref: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError("severity must be info, warning, or error")
        _name(self.code, "code")
        _name(self.message, "message")
        _optional_name(self.source_ref, "source_ref")
        _mapping(self.attributes, "attributes")


@dataclass(frozen=True)
class CompilationStage:
    stage_id: str
    status: str
    input_refs: Tuple[str, ...] = ()
    output_refs: Tuple[str, ...] = ()
    diagnostics: Tuple[CompilationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _name(self.stage_id, "stage_id")
        if self.status not in {"completed", "partial", "failed", "skipped"}:
            raise ValueError("unsupported compilation stage status")
        _tuple(self.input_refs, "input_refs")
        _tuple(self.output_refs, "output_refs")
        _tuple(self.diagnostics, "diagnostics")


@dataclass(frozen=True)
class CompilationRecord:
    compiler_id: str
    compiler_version: str
    stages: Tuple[CompilationStage, ...]
    source_digest: str
    diagnostics: Tuple[CompilationDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _name(self.compiler_id, "compiler_id")
        _name(self.compiler_version, "compiler_version")
        _tuple(self.stages, "stages")
        if not all(isinstance(item, CompilationStage) for item in self.stages):
            raise ValueError("stages must contain CompilationStage values")
        _unique(tuple(item.stage_id for item in self.stages), "compilation stage ids")
        _name(self.source_digest, "source_digest")
        _tuple(self.diagnostics, "diagnostics")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


class CanonicalValidationError(ValueError):
    def __init__(self, issues: Tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        super().__init__("canonical scenario is invalid:\n- " + "\n- ".join(
            "{}: {}".format(issue.path, issue.message) for issue in issues
        ))


@dataclass(frozen=True)
class CanonicalScenario:
    scenario_id: str
    hardware: HardwareGraph
    model: ModelGraph
    workload: WorkloadGraph
    parallel_plan: ParallelPlan
    placement_plan: PlacementPlan
    compilation: CompilationRecord
    provenance: Tuple[Origin, ...]
    assumptions: Tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_V1_VERSION

    def __post_init__(self) -> None:
        _name(self.scenario_id, "scenario_id")
        if self.schema_version != SCHEMA_V1_VERSION:
            raise ValueError("schema_version must be {}".format(SCHEMA_V1_VERSION))
        _tuple(self.provenance, "provenance")
        _tuple(self.assumptions, "assumptions")
        _mapping(self.attributes, "attributes")
        issues = validate_canonical_scenario(self)
        if any(issue.severity == "error" for issue in issues):
            raise CanonicalValidationError(issues)

    def to_dict(self) -> Dict[str, Any]:
        return to_primitive(self)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return canonical_json(self, indent=indent)

    @property
    def digest(self) -> str:
        return stable_hash(self)

    @property
    def sub_operators(self) -> Tuple[SubOperator, ...]:
        """Expose the model-owned runtime primitive registry at scenario scope."""

        return self.model.sub_operators

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CanonicalScenario":
        return canonical_scenario_from_dict(data)

    @classmethod
    def from_json(cls, payload: str) -> "CanonicalScenario":
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("canonical scenario JSON must contain an object")
        return canonical_scenario_from_dict(data)


def validate_canonical_scenario(scenario: CanonicalScenario) -> Tuple[ValidationIssue, ...]:
    """Validate cross-graph references and parallel/placement invariants."""

    issues = []

    def error(code: str, path: str, message: str) -> None:
        issues.append(ValidationIssue("error", code, path, message))

    def warning(code: str, path: str, message: str) -> None:
        issues.append(ValidationIssue("warning", code, path, message))

    nodes = {item.node_id: item for item in scenario.hardware.nodes}
    profiles_raw = scenario.attributes.get("profiles", {})
    profiles = profiles_raw if isinstance(profiles_raw, Mapping) else {}
    registries_raw = profiles.get("components", {})
    component_profile_registries = (
        registries_raw if isinstance(registries_raw, Mapping) else {}
    )
    bindings_raw = profiles.get("component_bindings", {})
    component_bindings = (
        bindings_raw if isinstance(bindings_raw, Mapping) else {}
    )
    for index, link in enumerate(scenario.hardware.links):
        for side, node_id, port_id in (
            ("source", link.source_node_id, link.source_port_id),
            ("target", link.target_node_id, link.target_port_id),
        ):
            node = nodes.get(node_id)
            if node is None:
                error("unknown_hardware_node", "hardware.links[{}].{}".format(index, side), node_id)
            elif port_id not in {port.port_id for port in node.ports}:
                error("unknown_hardware_port", "hardware.links[{}].{}_port_id".format(index, side), port_id)

    operators = {item.operator_id: item for item in scenario.model.operators}
    tensors = {item.tensor_id: item for item in scenario.model.tensors}
    for index, operator in enumerate(scenario.model.operators):
        for tensor_id in operator.input_tensor_ids + operator.output_tensor_ids + operator.weight_tensor_ids:
            if tensor_id not in tensors:
                error("unknown_model_tensor", "model.operators[{}]".format(index), tensor_id)
    for index, tensor in enumerate(scenario.model.tensors):
        if tensor.producer_operator_id is not None and tensor.producer_operator_id not in operators:
            error("unknown_tensor_producer", "model.tensors[{}].producer_operator_id".format(index), tensor.producer_operator_id)
        for consumer in tensor.consumer_operator_ids:
            if consumer not in operators:
                error("unknown_tensor_consumer", "model.tensors[{}].consumer_operator_ids".format(index), consumer)

    plan = scenario.parallel_plan
    if len(plan.ranks) != plan.world_size:
        error("parallel_world_size", "parallel_plan.ranks", "expected {} ranks".format(plan.world_size))
    rank_ids = {rank.rank_id for rank in plan.ranks}
    if rank_ids != set(range(plan.world_size)):
        error("parallel_rank_ids", "parallel_plan.ranks", "rank ids must cover [0, world_size)")
    coordinates = {(rank.tp_rank, rank.pp_rank, rank.ep_rank) for rank in plan.ranks}
    expected_coordinates = {
        (tp, pp, ep)
        for tp in range(plan.tp_degree)
        for pp in range(plan.pp_degree)
        for ep in range(plan.ep_degree)
    }
    if coordinates != expected_coordinates:
        error("parallel_coordinates", "parallel_plan.ranks", "rank coordinates must cover the TP/PP/EP product")
    for index, rank in enumerate(plan.ranks):
        for label, node_id in (
            ("compute_node_id", rank.compute_node_id),
            ("memory_node_id", rank.memory_node_id),
            ("cim_node_id", rank.cim_node_id),
        ):
            if node_id is not None and node_id not in nodes:
                error("unknown_rank_node", "parallel_plan.ranks[{}].{}".format(index, label), node_id)

    layer_ids = {
        sub_operator.layer_id
        for sub_operator in scenario.model.sub_operators
        if sub_operator.layer_id is not None
    }
    if not layer_ids:
        layer_ids = {
            operator.layer_id
            for operator in operators.values()
            if operator.layer_id is not None
        }
    assignments = {item.layer_id: item.stage_id for item in plan.layer_stages}
    if set(assignments) != layer_ids:
        error("layer_stage_coverage", "parallel_plan.layer_stages", "assignments must cover all model layer ids")
    for layer_id, stage in assignments.items():
        if stage >= plan.pp_degree:
            error("invalid_stage", "parallel_plan.layer_stages.{}".format(layer_id), str(stage))

    source_operators = {
        item.operator_id: item for item in scenario.model.source_operators
    }
    sub_operators = {
        item.sub_operator_id: item for item in scenario.model.sub_operators
    }
    ranks_by_id = {item.rank_id: item for item in plan.ranks}
    ranks_by_stage = {
        stage: {rank.rank_id for rank in plan.ranks if rank.pp_rank == stage}
        for stage in range(plan.pp_degree)
    }

    def validate_target_component(path: str, component_id: str) -> None:
        if component_id not in nodes:
            error("unknown_target_component", path, component_id)
            return
        target_kind = nodes[component_id].kind
        if (
            target_kind not in {"gpu", "cpu"}
            and "cim" not in target_kind
            and "compute_in_memory" not in target_kind
        ):
            error(
                "invalid_target_component_kind",
                path,
                "operator targets must be GPU, CPU, or CIM nodes, not {}".format(
                    target_kind
                ),
            )

    def validate_target_profile(path: str, target: Any) -> None:
        node = nodes.get(target.component_id)
        if node is None:
            return
        normalized_kind = node.kind.strip().lower().replace("-", "_")
        if normalized_kind in {"gpu", "cpu"}:
            expected_kind = normalized_kind
        elif "cim" in normalized_kind or "compute_in_memory" in normalized_kind:
            expected_kind = "cim"
        else:
            # Canonical schemas may eventually place auxiliary, non-cost
            # operations on other node types.  The current executable target
            # validator rejects those types independently.
            return
        node_profile_kind = node.attributes.get("cost_profile_kind")
        node_profile_id = node.attributes.get("cost_profile_id")
        if node_profile_kind != expected_kind or not isinstance(
            node_profile_id, str
        ) or not node_profile_id.strip():
            error(
                "target_component_profile_missing",
                path + ".component_id",
                "target component {} must bind one {} cost profile".format(
                    target.component_id,
                    expected_kind,
                ),
            )
            return
        if target.cost_profile_kind != expected_kind:
            error(
                "target_cost_profile_kind_mismatch",
                path + ".cost_profile_kind",
                "expected {}, got {}".format(
                    expected_kind,
                    target.cost_profile_kind,
                ),
            )
        if target.cost_profile_id != node_profile_id:
            error(
                "target_cost_profile_id_mismatch",
                path + ".cost_profile_id",
                "target component {} binds {}, got {}".format(
                    target.component_id,
                    node_profile_id,
                    target.cost_profile_id,
                ),
            )
        registry = component_profile_registries.get(expected_kind)
        if not isinstance(registry, Mapping) or node_profile_id not in registry:
            error(
                "unknown_target_cost_profile",
                "attributes.profiles.components.{}".format(expected_kind),
                "target component {} binds unknown profile {}".format(
                    target.component_id,
                    node_profile_id,
                ),
            )
        binding = component_bindings.get(target.component_id)
        if binding is None:
            error(
                "component_profile_binding_missing",
                "attributes.profiles.component_bindings.{}".format(
                    target.component_id
                ),
                "target component requires an explicit profile binding",
            )
        elif not isinstance(binding, Mapping):
            error(
                "invalid_component_profile_binding",
                "attributes.profiles.component_bindings.{}".format(
                    target.component_id
                ),
                "component profile binding must be an object",
            )
        elif (
            binding.get("profile_kind") != expected_kind
            or binding.get("cost_profile_id") != node_profile_id
        ):
            error(
                "component_profile_binding_mismatch",
                "attributes.profiles.component_bindings.{}".format(
                    target.component_id
                ),
                "expected {}/{} from HardwareNode".format(
                    expected_kind,
                    node_profile_id,
                ),
            )

    for index, target in enumerate(scenario.placement_plan.operator_targets):
        path = "placement_plan.operator_targets[{}]".format(index)
        if target.operator_id not in source_operators:
            error("unknown_target_operator", "placement_plan.operator_targets[{}].operator_id".format(index), target.operator_id)
        if target.rank_id not in rank_ids:
            error("unknown_target_rank", "placement_plan.operator_targets[{}].rank_id".format(index), str(target.rank_id))
        validate_target_component(path + ".component_id", target.component_id)
        validate_target_profile(path, target)
        operator = source_operators.get(target.operator_id)
        rank = ranks_by_id.get(target.rank_id)
        if operator is not None and rank is not None:
            parent_layer_ids = tuple(
                str(item)
                for item in operator.attributes.get("layer_ids", ())
            )
            if target.operator_id == "embedding":
                allowed_stages = {0}
            elif target.operator_id == "lm_head" or operator.op_kind.startswith("mtp_"):
                allowed_stages = {plan.pp_degree - 1}
            else:
                allowed_stages = {
                    assignments[layer_id]
                    for layer_id in parent_layer_ids
                    if layer_id in assignments
                }
            if allowed_stages and rank.pp_rank not in allowed_stages:
                error(
                    "operator_target_stage_mismatch",
                    path + ".rank_id",
                    "rank PP stage {} does not own parent operator stages {}".format(
                        rank.pp_rank,
                        sorted(allowed_stages),
                    ),
                )

    sub_target_pairs = set()
    for index, target in enumerate(
        scenario.placement_plan.sub_operator_targets
    ):
        path = "placement_plan.sub_operator_targets[{}]".format(index)
        sub_operator = sub_operators.get(target.sub_operator_id)
        if sub_operator is None:
            error(
                "unknown_target_sub_operator",
                path + ".sub_operator_id",
                target.sub_operator_id,
            )
        if target.rank_id not in rank_ids:
            error("unknown_target_rank", path + ".rank_id", str(target.rank_id))
        validate_target_component(path + ".component_id", target.component_id)
        validate_target_profile(path, target)
        target_node = nodes.get(target.component_id)
        if sub_operator is not None and target_node is not None:
            target_kind = target_node.kind
            target_is_cim = (
                "cim" in target_kind
                or "compute_in_memory" in target_kind
            )
            if sub_operator.operator_class != "gemm" and target_is_cim:
                error(
                    "non_gemm_cim_target_unsupported",
                    path + ".component_id",
                    (
                        "operator_id={}; operator_class={}; "
                        "requested_target={}; resolved_target={}; "
                        "resolution_applied=false"
                    ).format(
                        sub_operator.sub_operator_id,
                        sub_operator.operator_class,
                        target.attributes.get(
                            "requested_target",
                            target.component_id,
                        ),
                        target.component_id,
                    ),
                )
        if target.parent_fallback_operator_id is not None:
            if sub_operator is not None and (
                target.parent_fallback_operator_id
                != sub_operator.parent_operator_id
            ):
                error(
                    "sub_operator_parent_fallback_mismatch",
                    path + ".parent_fallback_operator_id",
                    "expected {}, got {}".format(
                        sub_operator.parent_operator_id,
                        target.parent_fallback_operator_id,
                    ),
                )
            if not any(
                parent.operator_id == target.parent_fallback_operator_id
                and parent.rank_id == target.rank_id
                for parent in scenario.placement_plan.operator_targets
            ):
                error(
                    "missing_parent_fallback_target",
                    path + ".parent_fallback_operator_id",
                    "no matching operator target exists for rank {}".format(
                        target.rank_id
                    ),
                )
        rank = ranks_by_id.get(target.rank_id)
        if sub_operator is not None and rank is not None:
            stage_raw = sub_operator.attributes.get("stage_id")
            if stage_raw is None and sub_operator.layer_id in assignments:
                stage_raw = assignments[sub_operator.layer_id]
            if stage_raw is not None and (
                isinstance(stage_raw, bool)
                or not isinstance(stage_raw, int)
                or stage_raw < 0
                or stage_raw >= plan.pp_degree
            ):
                error(
                    "invalid_sub_operator_stage",
                    "model.sub_operators[{}].attributes.stage_id".format(
                        target.sub_operator_id
                    ),
                    str(stage_raw),
                )
            elif stage_raw is not None and rank.pp_rank != stage_raw:
                error(
                    "sub_operator_target_stage_mismatch",
                    path + ".rank_id",
                    "rank PP stage {} does not own sub-operator stage {}".format(
                        rank.pp_rank,
                        stage_raw,
                    ),
                )
        sub_target_pairs.add((target.sub_operator_id, target.rank_id))

    for sub_operator in scenario.model.sub_operators:
        stage_raw = sub_operator.attributes.get("stage_id")
        if stage_raw is None and sub_operator.layer_id in assignments:
            stage_raw = assignments[sub_operator.layer_id]
        if stage_raw is None:
            error(
                "sub_operator_stage_missing",
                "model.sub_operators.{}".format(sub_operator.sub_operator_id),
                "sub-operator placement coverage requires an explicit stage",
            )
            continue
        expected_ranks = ranks_by_stage.get(stage_raw, set())
        actual_ranks = {
            rank_id
            for sub_operator_id, rank_id in sub_target_pairs
            if sub_operator_id == sub_operator.sub_operator_id
        }
        missing = sorted(expected_ranks - actual_ranks)
        if missing:
            error(
                "sub_operator_target_coverage_incomplete",
                "placement_plan.sub_operator_targets",
                "{} is missing ranks {}".format(
                    sub_operator.sub_operator_id,
                    missing,
                ),
            )

    for plan_index, tensor_plan in enumerate(scenario.placement_plan.tensor_plans):
        if tensor_plan.tensor_id not in tensors:
            error("unknown_placement_tensor", "placement_plan.tensor_plans[{}].tensor_id".format(plan_index), tensor_plan.tensor_id)
        shard_ids = {item.shard_id for item in tensor_plan.shards}
        for shard_index, shard in enumerate(tensor_plan.shards):
            if shard.tensor_id != tensor_plan.tensor_id:
                error("shard_tensor_mismatch", "placement_plan.tensor_plans[{}].shards[{}]".format(plan_index, shard_index), shard.tensor_id)
        for replica_index, replica in enumerate(tensor_plan.replicas):
            path = "placement_plan.tensor_plans[{}].replicas[{}]".format(plan_index, replica_index)
            if replica.tensor_id != tensor_plan.tensor_id:
                error("replica_tensor_mismatch", path, replica.tensor_id)
            if replica.shard_id not in shard_ids:
                error("unknown_replica_shard", path + ".shard_id", replica.shard_id)
            if replica.component_id not in nodes:
                error("unknown_replica_component", path + ".component_id", replica.component_id)
            if replica.rank_id is not None and replica.rank_id not in rank_ids:
                error("unknown_replica_rank", path + ".rank_id", str(replica.rank_id))
    return tuple(issues)


T = TypeVar("T")


def _object(data: Any, label: str) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError("{} must be an object".format(label))
    return dict(data)


def _construct(cls: Type[T], data: Any, **nested: Tuple[Type[Any], bool]) -> T:
    """Strictly construct one schema dataclass.

    ``nested`` maps field names to ``(child_type, is_tuple)``.  Primitive tuple
    fields are normalized below so a JSON round trip recreates immutable IR.
    """

    values = _object(data, cls.__name__)
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("{} contains unknown fields: {}".format(cls.__name__, ", ".join(unknown)))
    for name, (child_type, is_tuple) in nested.items():
        if name not in values:
            continue
        if is_tuple:
            raw = values[name]
            if not isinstance(raw, (list, tuple)):
                raise ValueError("{}.{} must be an array".format(cls.__name__, name))
            values[name] = tuple(_construct(child_type, item, **_NESTED.get(child_type, {})) for item in raw)
        else:
            values[name] = _construct(child_type, values[name], **_NESTED.get(child_type, {}))
    tuple_fields = {
        "provenance",
        "consumer_operator_ids",
        "input_tensor_ids",
        "output_tensor_ids",
        "weight_tensor_ids",
        "input_refs",
        "output_refs",
        "assumptions",
        "shape",
    }
    for name in tuple_fields.intersection(values):
        if name == "provenance" and "provenance" in nested:
            continue
        raw = values[name]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("{}.{} must be an array".format(cls.__name__, name))
        values[name] = tuple(raw)
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError("invalid {}: {}".format(cls.__name__, exc)) from exc


_NESTED: Dict[Type[Any], Dict[str, Tuple[Type[Any], bool]]] = {
    HardwareNode: {"ports": (HardwarePort, True), "provenance": (Origin, True)},
    HardwareLink: {"provenance": (Origin, True)},
    HardwareGraph: {"nodes": (HardwareNode, True), "links": (HardwareLink, True), "provenance": (Origin, True)},
    TensorValue: {"provenance": (Origin, True)},
    OperatorNode: {"ports": (OperatorPort, True), "provenance": (Origin, True)},
    SubOperator: {"provenance": (Origin, True)},
    ModelGraph: {
        "operators": (OperatorNode, True),
        "source_operators": (OperatorNode, True),
        "sub_operators": (SubOperator, True),
        "tensors": (TensorValue, True),
        "transforms": (TensorTransform, True),
        "provenance": (Origin, True),
    },
    RequestNode: {"provenance": (Origin, True)},
    WorkloadGraph: {"requests": (RequestNode, True), "provenance": (Origin, True)},
    RankPlan: {"provenance": (Origin, True)},
    StageAssignment: {"provenance": (Origin, True)},
    ParallelPlan: {"ranks": (RankPlan, True), "layer_stages": (StageAssignment, True), "provenance": (Origin, True)},
    OperatorTarget: {"provenance": (Origin, True)},
    SubOperatorTarget: {"provenance": (Origin, True)},
    TensorShard: {"provenance": (Origin, True)},
    TensorReplica: {"provenance": (Origin, True)},
    TensorPlan: {"shards": (TensorShard, True), "replicas": (TensorReplica, True), "provenance": (Origin, True)},
    PlacementPlan: {
        "operator_targets": (OperatorTarget, True),
        "sub_operator_targets": (SubOperatorTarget, True),
        "tensor_plans": (TensorPlan, True),
        "provenance": (Origin, True),
    },
    CompilationStage: {"diagnostics": (CompilationDiagnostic, True)},
    CompilationRecord: {"stages": (CompilationStage, True), "diagnostics": (CompilationDiagnostic, True)},
    CanonicalScenario: {
        "hardware": (HardwareGraph, False),
        "model": (ModelGraph, False),
        "workload": (WorkloadGraph, False),
        "parallel_plan": (ParallelPlan, False),
        "placement_plan": (PlacementPlan, False),
        "compilation": (CompilationRecord, False),
        "provenance": (Origin, True),
    },
}


def canonical_scenario_from_dict(data: Mapping[str, Any]) -> CanonicalScenario:
    return _construct(CanonicalScenario, data, **_NESTED[CanonicalScenario])


def model_graph_from_dict(data: Mapping[str, Any]) -> ModelGraph:
    """Strictly parse an authoring/canonical model graph JSON object."""

    return _construct(ModelGraph, data, **_NESTED[ModelGraph])


__all__ = [
    "SCHEMA_V1_VERSION",
    "CanonicalScenario",
    "CanonicalValidationError",
    "CompilationDiagnostic",
    "CompilationRecord",
    "CompilationStage",
    "HardwareGraph",
    "HardwareLink",
    "HardwareNode",
    "HardwarePort",
    "ModelGraph",
    "OperatorPort",
    "OperatorNode",
    "OperatorTarget",
    "SubOperator",
    "SubOperatorTarget",
    "Origin",
    "ParallelPlan",
    "PlacementPlan",
    "RankPlan",
    "RequestNode",
    "StageAssignment",
    "TensorPlan",
    "TensorReplica",
    "TensorShard",
    "TensorValue",
    "TensorTransform",
    "ValidationIssue",
    "WorkloadGraph",
    "canonical_scenario_from_dict",
    "model_graph_from_dict",
    "validate_canonical_scenario",
]
