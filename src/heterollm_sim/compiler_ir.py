"""V4 authoring-schema compiler for canonical schema 1.1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import ScenarioConfig, normalize_cost_profile_kind
from .ir import (
    ACTIVE_MEMORY_COMPONENT_KINDS,
    STORAGE_COMPONENT_KINDS,
    LayerSpec,
    model_graph_execution_digest,
    model_graph_execution_view,
    normalize_component_kind,
)
from .parallel import LogicalRank, build_parallel_plan
from .schema_v1 import (
    SCHEMA_V1_VERSION,
    CanonicalScenario,
    CompilationDiagnostic,
    CompilationRecord,
    CompilationStage,
    HardwareGraph,
    HardwareLink,
    HardwareNode,
    HardwarePort,
    ModelGraph,
    OperatorNode,
    OperatorPort,
    OperatorTarget,
    Origin,
    ParallelPlan,
    PlacementPlan,
    RankPlan,
    RequestNode,
    StageAssignment,
    SubOperator,
    SubOperatorTarget,
    TensorPlan,
    TensorReplica,
    TensorShard,
    TensorValue,
    WorkloadGraph,
)
from .serde import stable_hash, to_primitive


COMPILER_ID = "heterollm-v4-canonical-compiler"
COMPILER_VERSION = "4.0.0"


class CompilationPhase(str, Enum):
    READ_SOURCE = "read_source"
    BUILD_GRAPHS = "build_graphs"
    PLAN_PARALLELISM = "plan_parallelism"
    PLAN_PLACEMENT = "plan_placement"
    VALIDATE_CANONICAL = "validate_canonical"


class CanonicalizationError(ValueError):
    def __init__(
        self,
        phase: CompilationPhase,
        message: str,
        source_ref: str = "",
        diagnostics: Sequence[CompilationDiagnostic] = (),
    ) -> None:
        self.phase = phase
        self.source_ref = source_ref
        self.diagnostics = tuple(diagnostics)
        super().__init__("{}{}: {}".format(
            phase.value,
            " ({})".format(source_ref) if source_ref else "",
            message,
        ))


@dataclass(frozen=True)
class CompilerOptions:
    """Canonical compilation controls."""

    expand_synthetic_requests: bool = False
    include_unplaced_weight_tensors: bool = True


def _source_operator_registry(scenario: ScenarioConfig) -> Tuple[OperatorNode, ...]:
    """Return the immutable, authoritative authoring operator registry."""

    operators = tuple(scenario.model.graph.operators)
    ids = tuple(item.operator_id for item in operators)
    if len(ids) != len(set(ids)):
        raise CanonicalizationError(
            CompilationPhase.BUILD_GRAPHS,
            "authoritative model.graph operator ids must be unique",
            "ScenarioConfig.model.graph.operators",
        )
    return operators


def _layer_group_registry(execution_view: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for descriptor in execution_view.layer_instances:
        layer_id = str(descriptor.layer_id)
        group_id = str(descriptor.group_operator_id)
        previous = result.get(layer_id)
        if previous is not None and previous != group_id:
            raise CanonicalizationError(
                CompilationPhase.BUILD_GRAPHS,
                "layer {} belongs to multiple authoring groups".format(layer_id),
                "ScenarioConfig.model.graph.operators",
            )
        result[layer_id] = group_id
    return result


def _source_child_id(
    source_ids: set,
    layer_groups: Mapping[str, str],
    layer_id: str,
    suffix: str,
    *,
    fallback_suffix: Optional[str] = None,
) -> str:
    group_id = layer_groups.get(layer_id)
    if group_id is None:
        raise CanonicalizationError(
            CompilationPhase.BUILD_GRAPHS,
            "runtime layer {} has no authoritative authoring group".format(
                layer_id
            ),
            "ScenarioConfig.model.graph",
        )
    candidate = "{}.{}".format(group_id, suffix)
    if candidate in source_ids:
        return candidate
    if fallback_suffix is not None:
        fallback = "{}.{}".format(group_id, fallback_suffix)
        if fallback in source_ids:
            return fallback
    raise CanonicalizationError(
        CompilationPhase.BUILD_GRAPHS,
        "runtime primitive parent {} is absent from authoritative model.graph".format(
            candidate
        ),
        "ScenarioConfig.model.graph.operators",
    )


def _runtime_parent_suffix(relative_id: str) -> Tuple[str, Optional[str]]:
    if relative_id.startswith("input_norm."):
        return "norm1", None
    if relative_id.startswith("post_attention_norm."):
        return "norm2", None
    if relative_id in {"attention.residual", "linear_attention.residual"}:
        return "residual1", None
    if relative_id == "mlp.residual" or relative_id == "moe.residual":
        return "residual2", None
    if relative_id == "shared_expert.gate_apply":
        return "shared_expert_gate", "shared_expert"
    if relative_id == "shared_expert_gate":
        return "shared_expert_gate", "shared_expert"
    for prefix, suffix in (
        ("linear_attention", "linear_attention"),
        ("attention", "attention"),
        ("shared_expert", "shared_expert"),
        ("experts", "experts"),
        ("router", "router"),
        ("mlp", "mlp"),
    ):
        if relative_id == prefix or relative_id.startswith(prefix + "."):
            return suffix, None
    raise CanonicalizationError(
        CompilationPhase.BUILD_GRAPHS,
        "runtime primitive id {} has no canonical parent rule".format(
            relative_id
        ),
        "runtime_primitive_registry",
    )


def _runtime_expanded_operator_id(
    sub_operator_id: str,
    layer_id: Optional[str],
    expanded_ids: set,
) -> Optional[str]:
    if sub_operator_id in expanded_ids:
        return sub_operator_id
    if layer_id is None or not sub_operator_id.startswith(layer_id + "."):
        return None
    relative = sub_operator_id[len(layer_id) + 1 :]
    if relative.startswith("input_norm.") or relative.startswith(
        "post_attention_norm."
    ):
        return None
    if relative in {
        "attention.residual",
        "linear_attention.residual",
        "mlp.residual",
        "moe.residual",
    }:
        return None
    for prefix, expanded_suffix in (
        ("linear_attention", "linear_attention"),
        ("attention", "attention"),
        ("shared_expert_gate", "shared_expert_gate"),
        ("shared_expert", "shared_expert"),
        ("experts", "experts"),
        ("router", "router"),
        ("mlp", "mlp"),
    ):
        if relative == prefix or relative.startswith(prefix + "."):
            candidate = "{}.{}".format(layer_id, expanded_suffix)
            return candidate if candidate in expanded_ids else None
    return None


def _origin(
    scenario: ScenarioConfig,
    source_kind: str,
    source_id: str,
    transform: str,
    **attributes: Any
) -> Origin:
    return Origin(
        source_kind=source_kind,
        source_id=source_id,
        source_schema_version=scenario.schema_version,
        transform=transform,
        attributes=attributes,
    )


def _metadata(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _control_plane_decision(scenario: ScenarioConfig) -> Dict[str, Any]:
    control_plane = _metadata(
        scenario.placement.metadata.get("control_plane", {})
    )
    return _metadata(control_plane.get("decision", {}))


def _build_sub_operator_registry(
    scenario: ScenarioConfig,
    execution_view: Any,
    source_operators: Sequence[OperatorNode],
    expanded_operators: Sequence[OperatorNode],
    layer_stages: Mapping[str, int],
    pp_degree: int,
) -> Tuple[SubOperator, ...]:
    """Project mapper/runtime primitives onto authoritative graph parents."""

    # The mapper owns the exact runtime primitive vocabulary and operator
    # classes.  Consuming its immutable requirements here prevents the
    # canonical compiler from maintaining a second, drifting alias table.
    from .control_plane_planner import _derive_requirements

    source_ids = {item.operator_id for item in source_operators}
    expanded_by_id = {item.operator_id: item for item in expanded_operators}
    expanded_ids = set(expanded_by_id)
    layer_groups = _layer_group_registry(execution_view)
    requirements = tuple(
        item
        for item in _derive_requirements(scenario, execution_view)
        if not item.state_tensor
    )
    result = []
    seen = set()
    for requirement in requirements:
        sub_operator_id = str(requirement.item_id)
        if sub_operator_id in seen:
            raise CanonicalizationError(
                CompilationPhase.BUILD_GRAPHS,
                "duplicate runtime primitive id {}".format(sub_operator_id),
                "runtime_primitive_registry",
            )
        seen.add(sub_operator_id)
        requirement_layer_id = (
            str(requirement.layer.layer_id)
            if requirement.layer is not None
            else None
        )
        layer_id = (
            requirement_layer_id
            if requirement_layer_id is not None
            and sub_operator_id.startswith(requirement_layer_id + ".")
            else None
        )
        if layer_id is not None:
            relative_id = sub_operator_id[len(layer_id) + 1 :]
            suffix, fallback_suffix = _runtime_parent_suffix(relative_id)
            parent_operator_id = _source_child_id(
                source_ids,
                layer_groups,
                layer_id,
                suffix,
                fallback_suffix=fallback_suffix,
            )
            parent_rule = "layer_group_child:{}".format(suffix)
        elif sub_operator_id in source_ids:
            parent_operator_id = sub_operator_id
            parent_rule = "exact_authoring_operator_id"
        else:
            raise CanonicalizationError(
                CompilationPhase.BUILD_GRAPHS,
                "runtime primitive {} has no authoritative authoring parent".format(
                    sub_operator_id
                ),
                "ScenarioConfig.model.graph.operators",
            )
        expanded_operator_id = _runtime_expanded_operator_id(
            sub_operator_id,
            layer_id,
            expanded_ids,
        )
        if expanded_operator_id is not None:
            expanded_parent = expanded_by_id[
                expanded_operator_id
            ].attributes.get("authoring_operator_id")
            if expanded_parent != parent_operator_id:
                raise CanonicalizationError(
                    CompilationPhase.BUILD_GRAPHS,
                    "expanded operator {} resolves to {}, but runtime primitive {} resolves to {}".format(
                        expanded_operator_id,
                        expanded_parent,
                        sub_operator_id,
                        parent_operator_id,
                    ),
                    "ModelGraph.operators",
                )
        if sub_operator_id == "embedding":
            stage_id = 0
        elif sub_operator_id == "lm_head" or sub_operator_id.startswith(
            "mtp."
        ):
            stage_id = pp_degree - 1
        elif layer_id is not None and layer_id in layer_stages:
            stage_id = int(layer_stages[layer_id])
        else:
            raise CanonicalizationError(
                CompilationPhase.BUILD_GRAPHS,
                "runtime primitive {} has no pipeline stage".format(
                    sub_operator_id
                ),
                "PlacementSpec.parallel.layer_to_stage",
            )
        result.append(
            SubOperator(
                sub_operator_id=sub_operator_id,
                parent_operator_id=parent_operator_id,
                operator_class=requirement.operator_class.value,
                expanded_operator_id=expanded_operator_id,
                layer_id=layer_id,
                attributes={
                    "runtime_kind": requirement.kind,
                    "mapping_key": requirement.mapping_key,
                    "tensor_id": requirement.tensor_id,
                    "stage_id": stage_id,
                    "parent_relation": parent_rule,
                    "source_schema_version": scenario.model.schema_version,
                },
                provenance=(
                    _origin(
                        scenario,
                        "runtime_primitive_registry",
                        sub_operator_id,
                        "bind_runtime_primitive_to_authoring_operator",
                        parent_operator_id=parent_operator_id,
                        expanded_operator_id=expanded_operator_id,
                    ),
                ),
            )
        )
    return tuple(sorted(result, key=lambda item: item.sub_operator_id))


def _build_hardware_graph(scenario: ScenarioConfig) -> HardwareGraph:
    nodes = []
    for component in sorted(scenario.hardware.components, key=lambda item: item.component_id):
        ports = tuple(
            HardwarePort(
                port_id=port.port_id,
                protocol=port.protocol,
                role=port.role,
                direction=port.direction,
                bandwidth_gbps=float(port.bandwidth_gbps),
                attributes={
                    "version": port.version,
                    "lanes": port.lanes,
                    "max_links": port.max_links,
                    "payload": port.payload,
                    "metadata": dict(port.metadata),
                    "source_schema_version": port.schema_version,
                },
            )
            for port in sorted(component.ports, key=lambda item: item.port_id)
        )
        nodes.append(
            HardwareNode(
                node_id=component.component_id,
                kind=normalize_component_kind(component.kind),
                ports=ports,
                capacity_bytes=component.capacity_bytes,
                peak_ops_per_s=float(component.peak_ops_per_s),
                read_bandwidth_gbps=float(component.read_bandwidth_gbps),
                write_bandwidth_gbps=float(component.write_bandwidth_gbps),
                attributes={
                    "declared_kind": component.kind,
                    "cost_profile_id": component.cost_profile_id,
                    "cost_profile_kind": normalize_cost_profile_kind(
                        component.normalized_kind
                    ),
                    "package_id": component.package_id,
                    "die_id": component.die_id,
                    "metadata": dict(component.metadata),
                    "source_schema_version": component.schema_version,
                },
                provenance=(
                    _origin(
                        scenario,
                        "HardwareSpec.component",
                        component.component_id,
                        "normalize_component",
                    ),
                ),
            )
        )
    links = tuple(
        HardwareLink(
            link_id=link.link_id,
            source_node_id=link.source_component,
            source_port_id=link.source_port,
            target_node_id=link.target_component,
            target_port_id=link.target_port,
            protocol=link.protocol,
            bandwidth_gbps=float(link.bandwidth_gbps),
            latency_ns=float(link.latency_ns),
            bidirectional=link.bidirectional,
            attributes={
                "version": link.version,
                "lanes": link.lanes,
                "payload": link.payload,
                "metadata": dict(link.metadata),
                "source_schema_version": link.schema_version,
            },
            provenance=(
                _origin(scenario, "HardwareSpec.link", link.link_id, "normalize_link"),
            ),
        )
        for link in sorted(scenario.hardware.links, key=lambda item: item.link_id)
    )
    return HardwareGraph(
        graph_id=scenario.hardware.name,
        nodes=tuple(nodes),
        links=links,
        require_connected=scenario.hardware.require_connected,
        attributes={
            "metadata": dict(scenario.hardware.metadata),
            "source_schema_version": scenario.hardware.schema_version,
        },
        provenance=(
            _origin(
                scenario,
                "ScenarioConfig.hardware",
                scenario.hardware.name,
                "normalize_hardware_graph",
            ),
        ),
    )


def _weight_tensor_id(layer: LayerSpec, group: str) -> Optional[str]:
    if group == "full_attention":
        return "{}.attention_weights".format(layer.layer_id)
    if group == "linear_attention":
        return "{}.linear_attention_weights".format(layer.layer_id)
    if group == "mlp":
        return "{}.mlp_weights".format(layer.layer_id)
    if group == "router":
        return "{}.router_weights".format(layer.layer_id)
    if group == "experts":
        return "{}.expert_weights".format(layer.layer_id)
    if group == "shared_expert":
        return "{}.shared_expert_weights".format(layer.layer_id)
    if group == "shared_expert_gate":
        return "{}.shared_expert_gate_weights".format(layer.layer_id)
    return None


def _validated_model_execution_layers(
    scenario: ScenarioConfig, phase: CompilationPhase
) -> Tuple[LayerSpec, ...]:
    try:
        execution_view = model_graph_execution_view(
            scenario.model.graph,
            schema_version=scenario.model.schema_version,
        )
        execution_layers = tuple(
            item.layer for item in execution_view.layer_instances
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise CanonicalizationError(
            phase,
            "权威模型图未通过执行覆盖门禁：{}".format(exc),
            "ScenarioConfig.model.graph",
        )
    return execution_layers


def _build_model_graph(
    scenario: ScenarioConfig,
    execution_layers: Optional[Tuple[LayerSpec, ...]] = None,
    *,
    layer_stages: Optional[Mapping[str, int]] = None,
    pp_degree: int = 1,
) -> ModelGraph:
    if execution_layers is None:
        execution_layers = _validated_model_execution_layers(
            scenario, CompilationPhase.BUILD_GRAPHS
        )
    authoring_graph = to_primitive(scenario.model.graph)
    authoring_graph_digest = model_graph_execution_digest(
        scenario.model.graph,
        schema_version=scenario.model.schema_version,
    )
    execution_view = model_graph_execution_view(
        scenario.model.graph,
        schema_version=scenario.model.schema_version,
    )
    source_operators = _source_operator_registry(scenario)
    source_ids = {item.operator_id for item in source_operators}
    layer_groups = _layer_group_registry(execution_view)
    resolved_layer_stages = dict(layer_stages or {})
    operators: List[OperatorNode] = []
    tensor_records: Dict[str, Dict[str, Any]] = {}
    sequence_index = 0

    def ensure_tensor(
        tensor_id: str,
        role: str,
        *,
        logical_bytes: Optional[int] = None,
        producer: Optional[str] = None,
        consumer: Optional[str] = None,
        source_id: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        dtype: Optional[str] = None,
        shape: Optional[Tuple[Any, ...]] = None,
        layout: Optional[str] = None,
    ) -> None:
        record = tensor_records.setdefault(
            tensor_id,
            {
                "role": role,
                "logical_bytes": logical_bytes,
                "producer": producer,
                "consumers": [],
                "attributes": dict(attributes or {}),
                "source_id": source_id or tensor_id,
                "dtype": dtype or "unknown",
                "shape": shape or (),
                "layout": layout or "logical",
            },
        )
        expected = (dtype, shape, layout)
        actual = (record["dtype"], record["shape"], record["layout"])
        for index, value in enumerate(expected):
            if value is not None and actual[index] not in {value, "unknown", ()}:
                raise CanonicalizationError(
                    CompilationPhase.BUILD_GRAPHS,
                    "张量 {} 维度/类型不匹配：期望 {}，实际 {}".format(tensor_id, expected, actual),
                    tensor_id,
                )
        if dtype is not None and record["dtype"] == "unknown":
            record["dtype"] = dtype
        if shape is not None and not record["shape"]:
            record["shape"] = shape
        if layout is not None:
            record["layout"] = layout
        if record["logical_bytes"] is None and logical_bytes is not None:
            record["logical_bytes"] = logical_bytes
        if record["producer"] is None and producer is not None:
            record["producer"] = producer
        if consumer is not None and consumer not in record["consumers"]:
            record["consumers"].append(consumer)
        record["attributes"].update(dict(attributes or {}))

    def add_operator(
        operator_id: str,
        op_kind: str,
        *,
        layer: Optional[LayerSpec],
        inputs: Sequence[str],
        outputs: Sequence[str],
        weights: Sequence[str] = (),
        placement_group: str,
        attributes: Optional[Mapping[str, Any]] = None,
        authoring_operator_id: Optional[str] = None,
    ) -> None:
        nonlocal sequence_index
        source_id = layer.layer_id if layer is not None else operator_id
        op_attributes = dict(attributes or {})
        op_attributes["placement_group"] = placement_group
        resolved_authoring_id = authoring_operator_id or operator_id
        if resolved_authoring_id not in source_ids:
            raise CanonicalizationError(
                CompilationPhase.BUILD_GRAPHS,
                "expanded operator {} has no authoritative authoring operator {}".format(
                    operator_id,
                    resolved_authoring_id,
                ),
                "ScenarioConfig.model.graph.operators",
            )
        op_attributes["authoring_operator_id"] = resolved_authoring_id
        if layer is not None:
            op_attributes.update(
                {
                    "kind": layer.kind,
                    "hidden_size": layer.hidden_size,
                    "intermediate_size": layer.intermediate_size,
                    "attention_heads": layer.attention_heads,
                    "kv_heads": layer.effective_kv_heads,
                    "dtype": layer.dtype,
                    "quantization": layer.quantization,
                    "source_metadata": dict(layer.metadata),
                }
            )
        activation_dtype = layer.dtype if layer is not None else None
        activation_shape = ("B", "T", layer.hidden_size) if layer is not None else None
        for tensor_id in inputs:
            ensure_tensor(tensor_id, "activation", consumer=operator_id, dtype=activation_dtype, shape=activation_shape)
        for tensor_id in outputs:
            ensure_tensor(tensor_id, "activation", producer=operator_id, dtype=activation_dtype, shape=activation_shape)
        for tensor_id in weights:
            ensure_tensor(tensor_id, "weight", consumer=operator_id, dtype=activation_dtype)
        ports = []
        for direction, tensor_ids in (("input", inputs), ("output", outputs), ("weight", weights)):
            for port_index, tensor_id in enumerate(tensor_ids):
                record = tensor_records[tensor_id]
                ports.append(OperatorPort(
                    port_id="{}.{}".format(direction, port_index),
                    direction=direction,
                    tensor_id=tensor_id,
                    dtype=record["dtype"],
                    shape=record["shape"],
                    layout=record["layout"],
                ))
        operators.append(
            OperatorNode(
                operator_id=operator_id,
                op_kind=op_kind,
                sequence_index=sequence_index,
                layer_id=layer.layer_id if layer is not None else None,
                input_tensor_ids=tuple(inputs),
                output_tensor_ids=tuple(outputs),
                weight_tensor_ids=tuple(weights),
                ports=tuple(ports),
                parameters=dict(op_attributes),
                attributes=op_attributes,
                provenance=(
                    _origin(
                        scenario,
                        "ModelSpec.graph",
                        scenario.model.graph.graph_id,
                        "expand_validated_graph_projection_operator",
                        operator_id=operator_id,
                        projected_source_id=source_id,
                        authoring_graph_digest=authoring_graph_digest,
                    ),
                ),
            )
        )
        sequence_index += 1
    first_layer = execution_layers[0]
    ensure_tensor("input.tokens", "input", dtype="int64", shape=("B", "T"))
    embedding_bytes = scenario.model.embedding_weight_bytes or None
    ensure_tensor("embedding_weights", "weight", logical_bytes=embedding_bytes, dtype=first_layer.dtype, shape=("V", first_layer.hidden_size))
    ensure_tensor("embedding.output", "activation", dtype=first_layer.dtype, shape=("B", "T", first_layer.hidden_size))
    add_operator(
        "embedding",
        "embedding",
        layer=None,
        inputs=("input.tokens",),
        outputs=("embedding.output",),
        weights=("embedding_weights",),
        placement_group="embedding",
        attributes={"vocabulary_size": scenario.model.vocabulary_size},
    )
    previous = "embedding.output"

    for layer in execution_layers:
        mixer_op_kind = "linear_attention" if layer.is_linear_attention else "full_attention"
        mixer_group = "linear_attention" if layer.is_linear_attention else "attention"
        mixer_id = "{}.{}".format(layer.layer_id, mixer_group)
        mixer_output = "{}.mixer.output".format(layer.layer_id)
        mixer_weight = _weight_tensor_id(layer, mixer_op_kind)
        add_operator(
            mixer_id,
            mixer_op_kind,
            layer=layer,
            inputs=(previous,),
            outputs=(mixer_output,),
            weights=(mixer_weight,) if mixer_weight else (),
            placement_group=mixer_group,
            attributes={
                "sequence_mixer": layer.sequence_mixer,
                "linear_attention": to_primitive(layer.linear_attention),
            },
            authoring_operator_id=_source_child_id(
                source_ids,
                layer_groups,
                layer.layer_id,
                mixer_group,
            ),
        )
        if not layer.is_moe:
            output = "{}.output".format(layer.layer_id)
            weight_id = _weight_tensor_id(layer, "mlp")
            add_operator(
                "{}.mlp".format(layer.layer_id),
                "dense_mlp",
                layer=layer,
                inputs=(mixer_output,),
                outputs=(output,),
                weights=(weight_id,) if weight_id else (),
                placement_group="mlp",
                authoring_operator_id=_source_child_id(
                    source_ids,
                    layer_groups,
                    layer.layer_id,
                    "mlp",
                ),
            )
            previous = output
            continue

        router_output = "{}.router.output".format(layer.layer_id)
        router_weight = _weight_tensor_id(layer, "router")
        add_operator(
            "{}.router".format(layer.layer_id),
            "moe_router",
            layer=layer,
            inputs=(mixer_output,),
            outputs=(router_output,),
            weights=(router_weight,) if router_weight else (),
            placement_group="router",
            attributes={
                "num_experts": layer.num_experts,
                "experts_per_token": layer.experts_per_token,
            },
            authoring_operator_id=_source_child_id(
                source_ids,
                layer_groups,
                layer.layer_id,
                "router",
            ),
        )
        expert_output = "{}.experts.output".format(layer.layer_id)
        expert_weight = _weight_tensor_id(layer, "experts")
        add_operator(
            "{}.experts".format(layer.layer_id),
            "moe_experts",
            layer=layer,
            inputs=(router_output,),
            outputs=(expert_output,),
            weights=(expert_weight,) if expert_weight else (),
            placement_group="experts",
            authoring_operator_id=_source_child_id(
                source_ids,
                layer_groups,
                layer.layer_id,
                "experts",
            ),
        )
        combine_inputs = [expert_output]
        if layer.has_shared_expert:
            shared_output = "{}.shared_expert.output".format(layer.layer_id)
            shared_weight = _weight_tensor_id(layer, "shared_expert")
            add_operator(
                "{}.shared_expert".format(layer.layer_id),
                "shared_expert",
                layer=layer,
                inputs=(mixer_output,),
                outputs=(shared_output,),
                weights=(shared_weight,) if shared_weight else (),
                placement_group="shared_expert",
                authoring_operator_id=_source_child_id(
                    source_ids,
                    layer_groups,
                    layer.layer_id,
                    "shared_expert",
                ),
            )
            combine_inputs.append(shared_output)
            if layer.shared_expert_gate:
                gate_output = "{}.shared_expert_gate.output".format(layer.layer_id)
                gate_weight = _weight_tensor_id(layer, "shared_expert_gate")
                add_operator(
                    "{}.shared_expert_gate".format(layer.layer_id),
                    "shared_expert_gate",
                    layer=layer,
                    inputs=(mixer_output,),
                    outputs=(gate_output,),
                    weights=(gate_weight,) if gate_weight else (),
                    placement_group="shared_expert_gate",
                    authoring_operator_id=_source_child_id(
                        source_ids,
                        layer_groups,
                        layer.layer_id,
                        "shared_expert_gate",
                        fallback_suffix="shared_expert",
                    ),
                )
                combine_inputs.append(gate_output)
        output = "{}.output".format(layer.layer_id)
        add_operator(
            "{}.moe_combine".format(layer.layer_id),
            "moe_combine",
            layer=layer,
            inputs=tuple(combine_inputs),
            outputs=(output,),
            placement_group="moe_combine",
            authoring_operator_id=_source_child_id(
                source_ids,
                layer_groups,
                layer.layer_id,
                "moe_combine",
            ),
        )
        previous = output

    ensure_tensor("logits", "output", dtype=execution_layers[-1].dtype, shape=("B", "T", "V"))
    add_operator(
        "lm_head",
        "lm_head",
        layer=None,
        inputs=(previous,),
        outputs=("logits",),
        weights=("embedding_weights",),
        placement_group="lm_head",
        attributes={"vocabulary_size": scenario.model.vocabulary_size},
    )

    # The canonical execution graph expands validated backbone layer groups and
    # copies every typed MTP branch by operator/tensor identity.
    for descriptor in execution_view.mtp_descriptors:
        source_operator = descriptor.operator
        for tensor in (
            descriptor.input_tensor,
            descriptor.output_tensor,
            descriptor.weight_tensor,
        ):
            ensure_tensor(
                tensor.tensor_id,
                tensor.role,
                logical_bytes=tensor.logical_bytes,
                source_id=tensor.tensor_id,
                attributes={
                    **dict(tensor.attributes),
                    "authoring_graph_tensor": True,
                },
                dtype=tensor.dtype,
                shape=tensor.shape,
                layout=tensor.layout,
            )
        add_operator(
            source_operator.operator_id,
            source_operator.op_kind,
            layer=None,
            inputs=source_operator.input_tensor_ids,
            outputs=source_operator.output_tensor_ids,
            weights=source_operator.weight_tensor_ids,
            placement_group=source_operator.operator_id,
            attributes={
                **dict(source_operator.parameters),
                "source_attributes": dict(source_operator.attributes),
                "authoring_sequence_index": source_operator.sequence_index,
                "graph_native": True,
            },
        )

    decision = _control_plane_decision(scenario)
    weight_details = _metadata(decision.get("weight_tensor_details", {}))
    tensor_ids = set(scenario.placement.tensor_to_component)
    tensor_ids.update(scenario.placement.tensor_bytes)
    tensor_ids.update(weight_details)
    rank_shards = decision.get("rank_weight_shards", {})
    if isinstance(rank_shards, Mapping):
        tensor_ids.update(str(item) for item in rank_shards)
    for tensor_id in sorted(str(item) for item in tensor_ids):
        detail = _metadata(weight_details.get(tensor_id, {}))
        logical_raw = detail.get("logical_bytes", scenario.placement.tensor_bytes.get(tensor_id))
        logical_bytes = _optional_nonnegative_int(logical_raw, "tensor logical bytes", permissive=True)
        role = "weight" if "weight" in tensor_id.lower() else "state"
        ensure_tensor(
            tensor_id,
            role,
            logical_bytes=logical_bytes,
            attributes={"placement_extension_tensor": True},
        )

    tensors = tuple(
        TensorValue(
            tensor_id=tensor_id,
            role=str(record["role"]),
            logical_bytes=record["logical_bytes"],
            producer_operator_id=record["producer"],
            consumer_operator_ids=tuple(record["consumers"]),
            attributes=dict(record["attributes"]),
            dtype=record["dtype"],
            shape=record["shape"],
            layout=record["layout"],
            provenance=(
                _origin(
                    scenario,
                    "ModelSpec.graph",
                    scenario.model.graph.graph_id,
                    "expand_validated_graph_projection_tensor",
                    tensor_id=tensor_id,
                    projected_source_id=str(record["source_id"]),
                    placement_extension=bool(
                        record["attributes"].get("placement_extension_tensor")
                    ),
                    authoring_graph_digest=authoring_graph_digest,
                ),
            ),
        )
        for tensor_id, record in sorted(tensor_records.items())
    )
    sub_operators = _build_sub_operator_registry(
        scenario,
        execution_view,
        source_operators,
        operators,
        resolved_layer_stages,
        pp_degree,
    )
    return ModelGraph(
        graph_id=scenario.model.name,
        operators=tuple(operators),
        tensors=tensors,
        source_operators=source_operators,
        sub_operators=sub_operators,
        attributes={
            "architecture": scenario.model.architecture,
            "vocabulary_size": scenario.model.vocabulary_size,
            "max_sequence_length": scenario.model.max_sequence_length,
            "mtp_operator_ids": [
                descriptor.operator.operator_id
                for descriptor in execution_view.mtp_descriptors
            ],
            "mtp_weight_tensor_ids": [
                descriptor.weight_tensor.tensor_id
                for descriptor in execution_view.mtp_descriptors
            ],
            "text_backbone_only": scenario.model.text_backbone_only,
            "supported_modalities": list(scenario.model.supported_modalities),
            "excluded_subgraphs": list(scenario.model.excluded_subgraphs),
            "metadata": dict(scenario.model.metadata),
            "source_schema_version": scenario.model.schema_version,
            "authoring_graph": authoring_graph,
            "authoring_graph_digest": authoring_graph_digest,
            "execution_projection": {
                "validated": True,
                "lossless": True,
                "source": "ModelSpec.graph",
                "target": "expanded_per_layer_execution_graph",
            },
        },
        provenance=(
            _origin(
                scenario,
                "ModelSpec.graph",
                scenario.model.graph.graph_id,
                "project_validated_authoring_graph_to_expanded_execution_graph",
                authoring_graph_digest=authoring_graph_digest,
                projection="lossless_standard_transformer_layers",
            ),
        ),
    )


def _build_workload_graph(
    scenario: ScenarioConfig, options: CompilerOptions
) -> WorkloadGraph:
    requests = []
    if scenario.workload.requests:
        source_requests = tuple(scenario.workload.requests)
        transform = "normalize_explicit_request"
    elif options.expand_synthetic_requests:
        interval_ns = (
            1_000_000_000.0 / scenario.workload.arrival_rate_rps
            if scenario.workload.arrival_rate_rps > 0
            else 0.0
        )
        source_requests = (
            {
                "request_id": "request-{:04d}".format(index),
                "arrival_ns": index * interval_ns,
                "prompt_tokens": scenario.workload.prompt_tokens,
                "output_tokens": scenario.workload.output_tokens,
                "priority": 0,
                "deadline_ns": None,
                "metadata": {},
            }
            for index in range(scenario.workload.request_count)
        )
        transform = "expand_synthetic_request"
    else:
        source_requests = ()
        transform = "preserve_synthetic_template"
    for raw in source_requests:
        getter = raw.get if isinstance(raw, Mapping) else lambda name, default=None: getattr(raw, name, default)
        request_id = str(getter("request_id", ""))
        requests.append(
            RequestNode(
                request_id=request_id,
                arrival_ns=float(getter("arrival_ns", 0.0)),
                prompt_tokens=int(getter("prompt_tokens", 0)),
                output_tokens=int(getter("output_tokens", 0)),
                priority=int(getter("priority", 0)),
                deadline_ns=(
                    float(getter("deadline_ns"))
                    if getter("deadline_ns") is not None
                    else None
                ),
                attributes=dict(getter("metadata", {}) or {}),
                provenance=(
                    _origin(
                        scenario,
                        "WorkloadSpec.request",
                        request_id,
                        transform,
                    ),
                ),
            )
        )
    workload = scenario.workload
    return WorkloadGraph(
        graph_id=workload.name,
        requests=tuple(requests),
        scheduler=to_primitive(workload.scheduler),
        policy={"mtp": to_primitive(workload.mtp)},
        attributes={
            "synthetic_template": {
                "request_count": workload.request_count,
                "prompt_tokens": workload.prompt_tokens,
                "output_tokens": workload.output_tokens,
                "arrival_rate_rps": workload.arrival_rate_rps,
                "expanded": bool(
                    workload.requests or options.expand_synthetic_requests
                ),
            },
            "random_seed": workload.random_seed,
            "metadata": dict(workload.metadata),
            "source_schema_version": workload.schema_version,
        },
        provenance=(
            _origin(
                scenario,
                "ScenarioConfig.workload",
                workload.name,
                "normalize_workload_graph",
            ),
        ),
    )


def _build_parallel_plan(scenario: ScenarioConfig) -> Tuple[ParallelPlan, Any]:
    try:
        runtime_plan = build_parallel_plan(scenario)
    except ValueError as exc:
        raise CanonicalizationError(
            CompilationPhase.PLAN_PARALLELISM,
            str(exc),
            "placement.parallel",
        ) from exc
    ranks = tuple(
        RankPlan(
            rank_id=rank.rank,
            tp_rank=rank.tp_rank,
            pp_rank=rank.pp_rank,
            ep_rank=rank.ep_rank,
            compute_node_id=rank.component_id,
            memory_node_id=rank.memory_component_id,
            cim_node_id=rank.cim_component_id,
            provenance=(
                _origin(
                    scenario,
                    "ParallelSpec.rank_mapping",
                    str(rank.rank),
                    "normalize_logical_rank",
                ),
            ),
        )
        for rank in runtime_plan.ranks
    )
    execution_view = model_graph_execution_view(
        scenario.model.graph,
        schema_version=scenario.model.schema_version,
    )
    stages = tuple(
        StageAssignment(
            layer_id=layer.layer_id,
            stage_id=int(runtime_plan.layer_to_stage[layer.layer_id]),
            provenance=(
                _origin(
                    scenario,
                    "ParallelSpec.layer_to_stage",
                    layer.layer_id,
                    "normalize_pipeline_stage",
                ),
            ),
        )
        for layer in (
            item.layer for item in execution_view.layer_instances
        )
    )
    return (
        ParallelPlan(
            tp_degree=runtime_plan.tp_degree,
            pp_degree=runtime_plan.pp_degree,
            ep_degree=runtime_plan.ep_degree,
            ranks=ranks,
            layer_stages=stages,
            collective_algorithm=runtime_plan.collective_algorithm,
            routing_policy=runtime_plan.routing_policy,
            allow_padding=runtime_plan.allow_padding,
            provenance=(
                _origin(
                    scenario,
                    "PlacementSpec.parallel",
                    scenario.placement.model_name,
                    "materialize_parallel_plan",
                ),
            ),
        ),
        runtime_plan,
    )


def _operator_stage(operator: OperatorNode, layer_stages: Mapping[str, int], pp_degree: int) -> int:
    if operator.operator_id == "embedding":
        return 0
    if operator.operator_id == "lm_head":
        return pp_degree - 1
    if operator.op_kind in {"mtp_prediction_layer", "mtp_aux_head"}:
        return pp_degree - 1
    if operator.layer_id is None:
        return 0
    return int(layer_stages[operator.layer_id])


def _rank_for_component(
    component_id: str, ranks: Sequence[LogicalRank]
) -> Optional[int]:
    matches = {
        rank.rank
        for rank in ranks
        if component_id in {
            rank.component_id,
            rank.memory_component_id,
            rank.cim_component_id,
        }
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _target_cost_profile_binding(
    scenario: ScenarioConfig,
    component_id: str,
    source_ref: str,
) -> Tuple[str, str]:
    """Resolve the formal target profile from the actual target component."""

    component = scenario.hardware.component_map().get(component_id)
    if component is None:
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "operator target references unknown component {}".format(
                component_id
            ),
            source_ref,
        )
    profile_kind = normalize_cost_profile_kind(component.normalized_kind)
    if profile_kind is None or component.cost_profile_id is None:
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "operator target component {} has no typed cost-profile binding".format(
                component_id
            ),
            source_ref,
        )
    try:
        scenario.resolve_component_profile(component)
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "operator target component {} has an invalid cost-profile binding: {}".format(
                component_id,
                exc,
            ),
            source_ref,
        ) from exc
    return profile_kind, component.cost_profile_id


def _targets_from_explicit_entries(
    scenario: ScenarioConfig,
    operator: OperatorNode,
    entries: Sequence[Any],
    eligible_ranks: Sequence[LogicalRank],
    diagnostics: List[CompilationDiagnostic],
    *,
    operator_class: Optional[str] = None,
) -> Tuple[OperatorTarget, ...]:
    targets = []
    eligible_by_id = {rank.rank: rank for rank in eligible_ranks}
    seen_ranks = set()
    components = scenario.hardware.component_map()

    def error(
        code: str,
        message: str,
        source_ref: str,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        diagnostics.append(
            CompilationDiagnostic(
                severity="error",
                code=code,
                message=message,
                source_ref=source_ref,
                attributes=dict(attributes or {}),
            )
        )

    for index, raw in enumerate(entries):
        source_ref = "placement.metadata.control_plane.decision.operator_execution_targets.{}[{}]".format(operator.operator_id, index)
        if isinstance(raw, Mapping):
            component_raw = raw.get("component_id")
            component_id = str(component_raw) if component_raw not in (None, "") else ""
            rank_raw = raw.get("rank_id")
            rank_id = (
                _optional_nonnegative_int(rank_raw, "operator target rank_id")
                if rank_raw is not None
                else None
            )
        else:
            component_id = ""
            rank_id = None
        if not component_id or rank_id is None or rank_id not in eligible_by_id:
            error(
                "ambiguous_operator_execution_target",
                "operator execution metadata must identify a component and one eligible rank; no target was guessed",
                source_ref,
            )
            continue
        rank = eligible_by_id[rank_id]
        if rank_id in seen_ranks:
            error(
                "duplicate_operator_rank_target",
                "operator {} has more than one explicit target for rank {}".format(
                    operator.operator_id, rank_id
                ),
                source_ref,
            )
            continue
        component = components.get(component_id)
        if component is None:
            error(
                "unknown_operator_target_component",
                "operator {} rank {} references unknown component {}".format(
                    operator.operator_id, rank_id, component_id
                ),
                source_ref,
            )
            continue
        kind = normalize_component_kind(component.kind)
        is_cim = "cim" in kind or "compute_in_memory" in kind
        if kind not in {"gpu", "cpu"} and not is_cim:
            error(
                "invalid_operator_target_component",
                "operator targets must be GPU, CPU, or CIM components, not {}".format(
                    kind
                ),
                source_ref,
            )
            continue
        if operator_class is not None and operator_class != "gemm" and is_cim:
            failure = {
                "operator_id": operator.operator_id,
                "operator_class": operator_class,
                "requested_target": component_id,
                "resolved_target": component_id,
                "resolution_applied": False,
            }
            error(
                "non_gemm_cim_target_unsupported",
                (
                    "operator_id={operator_id}; "
                    "operator_class={operator_class}; "
                    "requested_target={requested_target}; "
                    "resolved_target={resolved_target}; "
                    "resolution_applied=false"
                ).format(**failure),
                source_ref,
                failure,
            )
            continue
        if kind == "gpu" and component_id != rank.component_id:
            error(
                "operator_rank_compute_mismatch",
                "rank {} executes on {}, not {}".format(
                    rank_id, rank.component_id, component_id
                ),
                source_ref,
            )
            continue
        if is_cim and rank.cim_component_id and component_id != rank.cim_component_id:
            error(
                "operator_rank_cim_mismatch",
                "rank {} is bound to CIM {}, not {}".format(
                    rank_id, rank.cim_component_id, component_id
                ),
                source_ref,
            )
            continue
        if isinstance(raw, Mapping):
            coordinates = (
                ("tp_rank", rank.tp_rank),
                ("pp_rank", rank.pp_rank),
                ("ep_rank", rank.ep_rank),
            )
            mismatch = next(
                (
                    (
                        name,
                        _optional_nonnegative_int(
                            raw[name], "operator target {}".format(name)
                        ),
                        expected,
                    )
                    for name, expected in coordinates
                    if raw.get(name) is not None
                    and _optional_nonnegative_int(
                        raw[name], "operator target {}".format(name)
                    )
                    != expected
                ),
                None,
            )
            compute_raw = raw.get("compute_component_id")
            if mismatch is not None:
                error(
                    "operator_rank_coordinate_mismatch",
                    "rank {} has {}={}, not {}".format(
                        rank_id, mismatch[0], mismatch[1], mismatch[2]
                    ),
                    source_ref,
                )
                continue
            if compute_raw not in (None, "") and str(compute_raw) != rank.component_id:
                error(
                    "operator_rank_compute_mismatch",
                    "rank {} compute component is {}, not {}".format(
                        rank_id, rank.component_id, compute_raw
                    ),
                    source_ref,
                )
                continue
        seen_ranks.add(rank_id)
        cost_profile_kind, cost_profile_id = _target_cost_profile_binding(
            scenario,
            component_id,
            source_ref,
        )
        targets.append(
            OperatorTarget(
                operator_id=operator.operator_id,
                rank_id=rank_id,
                component_id=component_id,
                source_key=source_ref,
                derivation="control_plane.decision.operator_execution_targets",
                cost_profile_kind=cost_profile_kind,
                cost_profile_id=cost_profile_id,
                provenance=(
                    _origin(
                        scenario,
                        "PlacementSpec.metadata.control_plane.decision",
                        source_ref,
                        "consume_rank_aware_operator_target",
                    ),
                ),
            )
        )
    return tuple(targets)


def _resolved_rank_target_component(
    scenario: ScenarioConfig,
    configured: Optional[str],
    rank: LogicalRank,
) -> Tuple[str, str]:
    components = scenario.hardware.component_map()
    configured_component = (
        components.get(configured) if configured is not None else None
    )
    configured_kind = (
        normalize_component_kind(configured_component.kind)
        if configured_component is not None
        else None
    )
    configured_is_cim = (
        configured_component is not None
        and (
            "cim" in configured_kind
            or "compute_in_memory" in configured_kind
        )
    )
    if configured is not None and configured_component is None:
        return configured, "invalid_authoring_op_mapping_preserved"
    if configured_is_cim:
        return (
            rank.cim_component_id or configured,
            "authoring_op_mapping_rank_cim",
        )
    if configured_kind == "cpu":
        return configured, "authoring_op_mapping_rank_cpu"
    if configured_component is not None and configured_kind != "gpu":
        return configured, "invalid_authoring_op_target_kind_preserved"
    return (
        rank.component_id,
        (
            "authoring_op_mapping_rank_compute"
            if configured is not None
            else "parallel_rank_default"
        ),
    )


def _sub_targets_from_explicit_entries(
    scenario: ScenarioConfig,
    sub_operator: SubOperator,
    parent_operator: OperatorNode,
    entries: Sequence[Any],
    eligible_ranks: Sequence[LogicalRank],
    diagnostics: List[CompilationDiagnostic],
) -> Tuple[SubOperatorTarget, ...]:
    probe = replace(parent_operator, operator_id=sub_operator.sub_operator_id)
    parsed = _targets_from_explicit_entries(
        scenario,
        probe,
        entries,
        eligible_ranks,
        diagnostics,
        operator_class=sub_operator.operator_class,
    )
    return tuple(
        SubOperatorTarget(
            sub_operator_id=sub_operator.sub_operator_id,
            rank_id=item.rank_id,
            component_id=item.component_id,
            source_key=item.source_key,
            derivation="control_plane.decision.operator_execution_targets",
            cost_profile_kind=item.cost_profile_kind,
            cost_profile_id=item.cost_profile_id,
            attributes={
                "requested_target": item.component_id,
                "resolved_target": item.component_id,
                "resolution_applied": False,
            },
            provenance=(
                _origin(
                    scenario,
                    "PlacementSpec.metadata.control_plane.decision",
                    item.source_key,
                    "consume_exact_sub_operator_target",
                    parent_operator_id=sub_operator.parent_operator_id,
                ),
            ),
        )
        for item in parsed
    )


def _sub_targets_from_exact_placement_key(
    scenario: ScenarioConfig,
    sub_operator: SubOperator,
    eligible_ranks: Sequence[LogicalRank],
    *,
    placement_key: Optional[str],
    parent_fallback: bool,
) -> Tuple[SubOperatorTarget, ...]:
    configured = (
        str(scenario.placement.op_to_component[placement_key])
        if placement_key is not None
        else None
    )
    source_key = placement_key or "parallel.rank_mapping"
    targets = []
    for rank in eligible_ranks:
        component_id, derivation = _resolved_rank_target_component(
            scenario,
            configured,
            rank,
        )
        if parent_fallback:
            derivation = "explicit_parent_operator_fallback"
        cost_profile_kind, cost_profile_id = _target_cost_profile_binding(
            scenario,
            component_id,
            source_key,
        )
        targets.append(
            SubOperatorTarget(
                sub_operator_id=sub_operator.sub_operator_id,
                rank_id=rank.rank,
                component_id=component_id,
                source_key=source_key,
                derivation=derivation,
                cost_profile_kind=cost_profile_kind,
                cost_profile_id=cost_profile_id,
                parent_fallback_operator_id=(
                    sub_operator.parent_operator_id
                    if parent_fallback
                    else None
                ),
                attributes={
                    "requested_target": configured or rank.component_id,
                    "resolved_target": component_id,
                    "resolution_applied": False,
                },
                provenance=(
                    _origin(
                        scenario,
                        (
                            "PlacementSpec.op_to_component"
                            if placement_key is not None
                            else "ParallelSpec.rank_mapping"
                        ),
                        source_key,
                        (
                            "apply_explicit_parent_operator_fallback"
                            if parent_fallback
                            else "place_exact_runtime_sub_operator"
                        ),
                        parent_operator_id=sub_operator.parent_operator_id,
                        configured_component_id=configured,
                    ),
                ),
            )
        )
    return tuple(targets)


def _optional_nonnegative_int(value: Any, label: str, *, permissive: bool = False) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        if permissive:
            return None
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "{} must be an integer".format(label),
        )
    try:
        converted = int(value)
    except (TypeError, ValueError):
        if permissive:
            return None
        raise CanonicalizationError(CompilationPhase.PLAN_PLACEMENT, "{} must be an integer".format(label))
    if converted < 0:
        if permissive:
            return None
        raise CanonicalizationError(CompilationPhase.PLAN_PLACEMENT, "{} must be non-negative".format(label))
    return converted


def _rank_shard_entries(rank_weight_shards: Any, tensor_id: str) -> Optional[Sequence[Any]]:
    if not isinstance(rank_weight_shards, Mapping):
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "rank_weight_shards must be a mapping",
            "placement.metadata.control_plane.decision.rank_weight_shards",
        )
    if tensor_id not in rank_weight_shards:
        return None
    raw = rank_weight_shards[tensor_id]
    if not isinstance(raw, (list, tuple)):
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "rank_weight_shards values must be arrays",
            "placement.metadata.control_plane.decision.rank_weight_shards.{}".format(
                tensor_id
            ),
        )
    return raw


def _tensor_plan_from_rank_shards(
    scenario: ScenarioConfig,
    tensor_id: str,
    entries: Sequence[Any],
    logical_bytes: Optional[int],
    runtime_ranks: Sequence[LogicalRank],
    expected_rank_ids: Sequence[int],
    expert_sharded: bool,
    *,
    source_name: str = "rank_weight_shards",
    default_residency: str = "rank_weight_shard",
) -> TensorPlan:
    shard_map: Dict[str, TensorShard] = {}
    replicas = []
    rank_by_id = {rank.rank: rank for rank in runtime_ranks}
    expected_ids = set(expected_rank_ids)
    seen_rank_ids = set()
    components = scenario.hardware.component_map()
    parallel = scenario.placement.parallel
    expected_shard_count = (
        parallel.tp_degree * parallel.ep_degree
        if expert_sharded
        else parallel.tp_degree
    )
    for index, raw in enumerate(entries):
        source_ref = "placement.metadata.control_plane.decision.{}.{}[{}]".format(
            source_name, tensor_id, index
        )
        if not isinstance(raw, Mapping):
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank_weight_shards entries must be objects",
                source_ref,
            )
        component_raw = raw.get("component_id")
        rank_raw = raw.get("rank_id")
        shard_index_raw = raw.get("shard_index")
        shard_count_raw = raw.get("shard_count")
        if component_raw in (None, "") or rank_raw is None:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank_weight_shards entries require rank_id and component_id",
                source_ref,
            )
        rank_id = _optional_nonnegative_int(rank_raw, "rank_id")
        assert rank_id is not None
        rank = rank_by_id.get(rank_id)
        if rank is None:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank_weight_shards references unknown rank {}".format(rank_id),
                source_ref,
            )
        if expected_ids and rank_id not in expected_ids:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank {} is outside the PP stage(s) that consume {}".format(
                    rank_id, tensor_id
                ),
                source_ref,
            )
        if rank_id in seen_rank_ids:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank {} has more than one replica entry for {}".format(
                    rank_id, tensor_id
                ),
                source_ref,
            )
        seen_rank_ids.add(rank_id)
        for coordinate_name, expected_coordinate in (
            ("tp_rank", rank.tp_rank),
            ("pp_rank", rank.pp_rank),
            ("ep_rank", rank.ep_rank),
        ):
            coordinate_raw = raw.get(coordinate_name)
            coordinate_value = (
                _optional_nonnegative_int(
                    coordinate_raw, "rank shard {}".format(coordinate_name)
                )
                if coordinate_raw is not None
                else None
            )
            if coordinate_value is not None and coordinate_value != expected_coordinate:
                raise CanonicalizationError(
                    CompilationPhase.PLAN_PLACEMENT,
                    "rank {} has {}={}, not {}".format(
                        rank_id,
                        coordinate_name,
                        expected_coordinate,
                        coordinate_value,
                    ),
                    source_ref,
                )
        compute_raw = raw.get("compute_component_id")
        if compute_raw not in (None, "") and str(compute_raw) != rank.component_id:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank {} compute component is {}, not {}".format(
                    rank_id, rank.component_id, compute_raw
                ),
                source_ref,
            )
        component_id = str(component_raw)
        component = components.get(component_id)
        if component is None:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank {} shard references unknown component {}".format(
                    rank_id, component_id
                ),
                source_ref,
            )
        component_kind = normalize_component_kind(component.kind)
        is_cim = "cim" in component_kind or "compute_in_memory" in component_kind
        if (
            component_kind not in STORAGE_COMPONENT_KINDS
            and component_kind not in ACTIVE_MEMORY_COMPONENT_KINDS
            and not is_cim
            and not (component_kind == "gpu" and component_id == rank.component_id)
        ):
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank {} shard component {} is not active memory, storage, its GPU, or CIM".format(
                    rank_id, component_id
                ),
                source_ref,
            )
        expected_shard_index = (
            rank.ep_rank * parallel.tp_degree + rank.tp_rank
            if expert_sharded
            else rank.tp_rank
        )
        shard_index = (
            _optional_nonnegative_int(shard_index_raw, "shard_index")
            if shard_index_raw is not None
            else expected_shard_index
        )
        shard_count = (
            _optional_nonnegative_int(shard_count_raw, "shard_count")
            if shard_count_raw is not None
            else expected_shard_count
        )
        assert shard_index is not None and shard_count is not None
        if shard_index != expected_shard_index or shard_count != expected_shard_count:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "rank {} must describe shard {}/{} for {} policy, got {}/{}".format(
                    rank_id,
                    expected_shard_index,
                    expected_shard_count,
                    "TPxEP expert" if expert_sharded else "TP with EP replication",
                    shard_index,
                    shard_count,
                ),
                source_ref,
            )
        shard_kind = str(raw.get("shard_kind", ""))
        allowed_kinds = (
            {"", "tp_ep_expert_shard"}
            if expert_sharded
            else {"", "tp_shard", "tp_shard_ep_replica"}
        )
        if shard_kind not in allowed_kinds:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "shard_kind {} conflicts with {} tensor policy".format(
                    shard_kind, "expert" if expert_sharded else "dense"
                ),
                source_ref,
            )
        shard_id = str(raw.get("shard_id", "{}#shard-{:04d}".format(tensor_id, shard_index)))
        shard_logical_bytes = _optional_nonnegative_int(
            raw.get("logical_bytes"), "shard logical_bytes"
        )
        replica_physical_bytes = _optional_nonnegative_int(
            raw.get("physical_bytes"), "replica physical_bytes"
        )
        shard = TensorShard(
            shard_id=shard_id,
            tensor_id=tensor_id,
            shard_index=shard_index,
            shard_count=shard_count,
            logical_bytes=shard_logical_bytes,
            axis=(int(raw["axis"]) if raw.get("axis") is not None else None),
            derivation="control_plane.decision.{}".format(source_name),
            provenance=(
                _origin(scenario, "PlacementSpec.metadata.control_plane.decision", source_ref, "consume_explicit_rank_shard"),
            ),
        )
        previous = shard_map.get(shard_id)
        if previous is not None:
            logical_signature = (
                shard.tensor_id,
                shard.shard_index,
                shard.shard_count,
                shard.logical_bytes,
                shard.axis,
                shard.derivation,
            )
            previous_signature = (
                previous.tensor_id,
                previous.shard_index,
                previous.shard_count,
                previous.logical_bytes,
                previous.axis,
                previous.derivation,
            )
            if previous_signature != logical_signature:
                raise CanonicalizationError(
                    CompilationPhase.PLAN_PLACEMENT,
                    "replicas sharing shard_id must describe the same logical shard",
                    source_ref,
                )
        else:
            shard_map[shard_id] = shard
        replicas.append(
            TensorReplica(
                replica_id=str(raw.get("replica_id", "{}#replica-{:04d}".format(shard_id, index))),
                tensor_id=tensor_id,
                shard_id=shard_id,
                component_id=component_id,
                rank_id=rank_id,
                physical_bytes=replica_physical_bytes,
                residency=str(raw.get("residency", default_residency)),
                derivation="control_plane.decision.{}".format(source_name),
                provenance=(
                    _origin(scenario, "PlacementSpec.metadata.control_plane.decision", source_ref, "consume_explicit_tensor_replica"),
                ),
            )
        )
    if expected_ids and seen_rank_ids != expected_ids:
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "rank-aware shards for {} cover ranks {}, expected {}".format(
                tensor_id, sorted(seen_rank_ids), sorted(expected_ids)
            ),
            "placement.metadata.control_plane.decision.{}.{}".format(
                source_name, tensor_id
            ),
        )
    if {shard.shard_index for shard in shard_map.values()} != set(
        range(expected_shard_count)
    ):
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "rank-aware shards for {} must cover every logical shard index in [0, {})".format(
                tensor_id, expected_shard_count
            ),
            "placement.metadata.control_plane.decision.{}.{}".format(
                source_name, tensor_id
            ),
        )
    if logical_bytes is not None:
        shard_logical_total = sum(
            int(shard.logical_bytes or 0) for shard in shard_map.values()
        )
        if shard_logical_total != logical_bytes:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "unique logical shards for {} total {} bytes, expected {}".format(
                    tensor_id, shard_logical_total, logical_bytes
                ),
                "placement.metadata.control_plane.decision.{}.{}".format(
                    source_name, tensor_id
                ),
            )
    return TensorPlan(
        tensor_id=tensor_id,
        logical_bytes=logical_bytes,
        shards=tuple(sorted(shard_map.values(), key=lambda item: (item.shard_index, item.shard_id))),
        replicas=tuple(sorted(replicas, key=lambda item: item.replica_id)),
        source_tensor_id=tensor_id,
        attributes={"source": source_name},
        provenance=(
            _origin(scenario, "PlacementSpec.metadata.control_plane.decision", tensor_id, "normalize_rank_weight_shards"),
        ),
    )


def _tensor_plan_from_authoring_placement(
    scenario: ScenarioConfig,
    tensor_id: str,
    detail: Mapping[str, Any],
    runtime_ranks: Sequence[LogicalRank],
    include_unplaced: bool,
) -> Optional[TensorPlan]:
    placement = scenario.placement
    logical_raw = detail.get(
        "logical_bytes",
        placement.tensor_bytes.get(tensor_id),
    )
    logical_bytes = _optional_nonnegative_int(logical_raw, "tensor logical_bytes", permissive=True)
    component_ids_raw = detail.get("replica_component_ids")
    if isinstance(component_ids_raw, (list, tuple)):
        component_ids = tuple(str(item) for item in component_ids_raw if str(item))
    else:
        component_ids = ()
    authoring_component = placement.tensor_to_component.get(tensor_id)
    backing_component = detail.get("backing_component_id")
    if not component_ids and authoring_component:
        component_ids = (str(authoring_component),)
    if not component_ids and backing_component:
        component_ids = (str(backing_component),)
    if not component_ids and not include_unplaced:
        return None
    shard = TensorShard(
        shard_id="{}#whole".format(tensor_id),
        tensor_id=tensor_id,
        shard_index=0,
        shard_count=1,
        logical_bytes=logical_bytes,
        derivation=(
            "authoring_whole_tensor_no_shard_evidence"
            if component_ids
            else "unresolved_no_shard_evidence"
        ),
        provenance=(
            _origin(
                scenario,
                "PlacementSpec.tensor",
                tensor_id,
                "preserve_whole_tensor_without_inferred_sharding",
            ),
        ),
    )
    padded = _optional_nonnegative_int(detail.get("padded_bytes_per_replica"), "padded_bytes_per_replica", permissive=True)
    placement_bytes = _optional_nonnegative_int(detail.get("placement_bytes"), "placement_bytes", permissive=True)
    total = _optional_nonnegative_int(detail.get("total_physical_bytes"), "total_physical_bytes", permissive=True)
    replicas = []
    for index, component_id in enumerate(component_ids):
        physical_bytes = padded or placement_bytes
        if physical_bytes is None and total is not None and len(component_ids) == 1:
            physical_bytes = total
        replicas.append(
            TensorReplica(
                replica_id="{}#replica-{:04d}".format(tensor_id, index),
                tensor_id=tensor_id,
                shard_id=shard.shard_id,
                component_id=component_id,
                rank_id=_rank_for_component(component_id, runtime_ranks),
                physical_bytes=physical_bytes,
                residency=str(detail.get("residency", "authoring_explicit")),
                derivation=(
                    "control_plane.decision.weight_tensor_details"
                    if detail
                    else "authoring_tensor_to_component"
                ),
                provenance=(
                    _origin(
                        scenario,
                        "PlacementSpec.metadata.control_plane.decision.weight_tensor_details" if detail else "PlacementSpec.tensor_to_component",
                        tensor_id,
                        "normalize_tensor_replica_without_inferred_rank",
                    ),
                ),
            )
        )
    return TensorPlan(
        tensor_id=tensor_id,
        logical_bytes=logical_bytes,
        shards=(shard,),
        replicas=tuple(replicas),
        source_tensor_id=tensor_id,
        attributes={
            "backing_tensor_id": detail.get("backing_tensor_id"),
            "backing_component_id": backing_component,
            "total_physical_bytes": total,
            "rank_assignment_known": all(item.rank_id is not None for item in replicas),
        },
        provenance=(
            _origin(scenario, "PlacementSpec.tensor", tensor_id, "normalize_tensor_plan"),
        ),
    )


def _validate_v4_control_plane_decision(decision: Mapping[str, Any]) -> None:
    for removed_key in ("execution_component_ids", "decisions"):
        if removed_key in decision:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "control_plane.decision.{} is not part of the V4 contract".format(
                    removed_key
                ),
                "placement.metadata.control_plane.decision.{}".format(removed_key),
            )
    operator_targets = decision.get("operator_execution_targets", {})
    if isinstance(operator_targets, Mapping) and "mtp" in operator_targets:
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "aggregate MTP operator targets are not part of the V4 contract",
            "placement.metadata.control_plane.decision.operator_execution_targets.mtp",
        )
    rank_weight_shards = decision.get("rank_weight_shards", {})
    if isinstance(rank_weight_shards, Mapping) and "mtp_weights" in rank_weight_shards:
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "aggregate MTP weight shards are not part of the V4 contract",
            "placement.metadata.control_plane.decision.rank_weight_shards.mtp_weights",
        )
    weight_details = decision.get("weight_tensor_details", {})
    if not isinstance(weight_details, Mapping):
        return
    for tensor_id, raw_detail in weight_details.items():
        if str(tensor_id) == "mtp_weights":
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "aggregate MTP weight details are not part of the V4 contract",
                "placement.metadata.control_plane.decision.weight_tensor_details.mtp_weights",
            )
        if isinstance(raw_detail, Mapping) and "rank_shards" in raw_detail:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "weight_tensor_details.rank_shards is not part of V4; use "
                "control_plane.decision.rank_weight_shards",
                "placement.metadata.control_plane.decision.weight_tensor_details.{}.rank_shards".format(
                    tensor_id
                ),
            )


def _build_placement_plan(
    scenario: ScenarioConfig,
    model: ModelGraph,
    canonical_parallel: ParallelPlan,
    runtime_plan: Any,
    options: CompilerOptions,
) -> Tuple[PlacementPlan, Tuple[CompilationDiagnostic, ...]]:
    control_plane_raw = scenario.placement.metadata.get("control_plane", {})
    if not isinstance(control_plane_raw, Mapping):
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "control_plane must be a mapping",
            "placement.metadata.control_plane",
        )
    decision_raw = control_plane_raw.get("decision", {})
    if not isinstance(decision_raw, Mapping):
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "control_plane.decision must be a mapping",
            "placement.metadata.control_plane.decision",
        )
    decision = dict(decision_raw)
    _validate_v4_control_plane_decision(decision)
    diagnostics: List[CompilationDiagnostic] = []
    layer_stages = {item.layer_id: item.stage_id for item in canonical_parallel.layer_stages}
    direct_raw = decision.get("operator_execution_targets")
    if direct_raw is not None and not isinstance(direct_raw, Mapping):
        raise CanonicalizationError(
            CompilationPhase.PLAN_PLACEMENT,
            "operator_execution_targets must be a mapping",
            "placement.metadata.control_plane.decision.operator_execution_targets",
        )
    direct_targets = dict(direct_raw) if isinstance(direct_raw, Mapping) else None
    source_operators = {
        item.operator_id: item for item in model.source_operators
    }
    operator_targets_by_pair: Dict[Tuple[str, int], OperatorTarget] = {}
    sub_operator_targets = []
    for sub_operator in model.sub_operators:
        stage = int(sub_operator.attributes["stage_id"])
        eligible = tuple(
            rank for rank in runtime_plan.ranks if rank.pp_rank == stage
        )
        parent_operator = source_operators.get(
            sub_operator.parent_operator_id
        )
        if parent_operator is None:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "sub-operator {} references unknown authoritative parent {}".format(
                    sub_operator.sub_operator_id,
                    sub_operator.parent_operator_id,
                ),
                "model.sub_operators",
            )
        entries = (
            direct_targets.get(sub_operator.sub_operator_id)
            if direct_targets is not None
            else None
        )
        if entries is not None:
            if not isinstance(entries, (list, tuple)):
                raise CanonicalizationError(
                    CompilationPhase.PLAN_PLACEMENT,
                    "operator_execution_targets values must be arrays",
                    "placement.metadata.control_plane.decision.operator_execution_targets.{}".format(
                        sub_operator.sub_operator_id
                    ),
                )
            explicit_targets = _sub_targets_from_explicit_entries(
                scenario,
                sub_operator,
                parent_operator,
                entries,
                eligible,
                diagnostics,
            )
            sub_operator_targets.extend(explicit_targets)
            covered = {target.rank_id for target in explicit_targets}
            if covered != {rank.rank for rank in eligible}:
                diagnostics.append(
                    CompilationDiagnostic(
                        severity="error",
                        code="sub_operator_rank_coverage_incomplete",
                        message="exact rank-aware sub-operator targets for {} cover ranks {}, expected {}; fallback was not applied".format(
                            sub_operator.sub_operator_id,
                            sorted(covered),
                            sorted(rank.rank for rank in eligible),
                        ),
                        source_ref="placement.metadata.control_plane.decision.operator_execution_targets.{}".format(
                            sub_operator.sub_operator_id
                        ),
                    )
                )
            continue
        exact_key = (
            sub_operator.sub_operator_id
            if sub_operator.sub_operator_id
            in scenario.placement.op_to_component
            else None
        )
        parent_key = (
            sub_operator.parent_operator_id
            if sub_operator.parent_operator_id
            in scenario.placement.op_to_component
            else None
        )
        if exact_key is not None:
            sub_operator_targets.extend(
                _sub_targets_from_exact_placement_key(
                    scenario,
                    sub_operator,
                    eligible,
                    placement_key=exact_key,
                    parent_fallback=False,
                )
            )
            continue
        if parent_key is not None:
            parent_configured = str(
                scenario.placement.op_to_component[parent_key]
            )
            for rank in eligible:
                pair = (parent_key, rank.rank)
                if pair not in operator_targets_by_pair:
                    component_id, derivation = _resolved_rank_target_component(
                        scenario,
                        parent_configured,
                        rank,
                    )
                    cost_profile_kind, cost_profile_id = (
                        _target_cost_profile_binding(
                            scenario,
                            component_id,
                            parent_key,
                        )
                    )
                    operator_targets_by_pair[pair] = OperatorTarget(
                        operator_id=parent_key,
                        rank_id=rank.rank,
                        component_id=component_id,
                        source_key=parent_key,
                        derivation=derivation,
                        cost_profile_kind=cost_profile_kind,
                        cost_profile_id=cost_profile_id,
                        provenance=(
                            _origin(
                                scenario,
                                "PlacementSpec.op_to_component",
                                parent_key,
                                "materialize_explicit_parent_fallback_target",
                            ),
                        ),
                    )
            sub_operator_targets.extend(
                _sub_targets_from_exact_placement_key(
                    scenario,
                    sub_operator,
                    eligible,
                    placement_key=parent_key,
                    parent_fallback=True,
                )
            )
            continue
        if direct_targets is not None:
            diagnostics.append(
                CompilationDiagnostic(
                    severity="error",
                    code="sub_operator_target_missing",
                    message="control-plane placement has no exact target for {} and no explicit parent fallback {}".format(
                        sub_operator.sub_operator_id,
                        sub_operator.parent_operator_id,
                    ),
                    source_ref="placement.metadata.control_plane.decision.operator_execution_targets",
                )
            )
            continue
        sub_operator_targets.extend(
            _sub_targets_from_exact_placement_key(
                scenario,
                sub_operator,
                eligible,
                placement_key=None,
                parent_fallback=False,
            )
        )

    sub_operators_by_id = {
        item.sub_operator_id: item for item in model.sub_operators
    }
    components = scenario.hardware.component_map()
    for target in sub_operator_targets:
        sub_operator = sub_operators_by_id[target.sub_operator_id]
        component = components.get(target.component_id)
        resolved_kind = (
            normalize_component_kind(component.kind)
            if component is not None
            else "unknown"
        )
        resolved_is_cim = (
            "cim" in resolved_kind
            or "compute_in_memory" in resolved_kind
        )
        if sub_operator.operator_class != "gemm" and resolved_is_cim:
            requested_target = str(
                target.attributes.get(
                    "requested_target",
                    target.component_id,
                )
            )
            diagnostics.append(
                CompilationDiagnostic(
                    severity="error",
                    code="non_gemm_cim_target_unsupported",
                    message=(
                        "operator_id={}; operator_class={}; "
                        "requested_target={}; resolved_target={}; "
                        "resolution_applied=false"
                    ).format(
                        sub_operator.sub_operator_id,
                        sub_operator.operator_class,
                        requested_target,
                        target.component_id,
                    ),
                    source_ref=target.source_key,
                    attributes={
                        "operator_id": sub_operator.sub_operator_id,
                        "operator_class": sub_operator.operator_class,
                        "requested_target": requested_target,
                        "resolved_target": target.component_id,
                        "resolution_applied": False,
                    },
                )
            )

    weight_details = _metadata(decision.get("weight_tensor_details", {}))
    rank_weight_shards = decision.get("rank_weight_shards", {})
    tensor_ids = set(scenario.placement.tensor_to_component)
    tensor_ids.update(scenario.placement.tensor_bytes)
    tensor_ids.update(str(item) for item in weight_details)
    if isinstance(rank_weight_shards, Mapping):
        tensor_ids.update(str(item) for item in rank_weight_shards)
    if options.include_unplaced_weight_tensors:
        tensor_ids.update(
            tensor.tensor_id for tensor in model.tensors if tensor.role == "weight"
        )
    tensor_plans = []
    model_tensors = {tensor.tensor_id: tensor for tensor in model.tensors}
    tensor_consumers: Dict[str, List[OperatorNode]] = {}
    for operator in model.operators:
        for weight_tensor_id in operator.weight_tensor_ids:
            tensor_consumers.setdefault(weight_tensor_id, []).append(operator)
    for tensor_id in sorted(tensor_ids):
        model_tensor = model_tensors.get(tensor_id)
        detail = _metadata(weight_details.get(tensor_id, {}))
        logical_raw = detail.get(
            "logical_bytes",
            scenario.placement.tensor_bytes.get(
                tensor_id,
                model_tensor.logical_bytes if model_tensor is not None else None,
            ),
        )
        logical_bytes = _optional_nonnegative_int(logical_raw, "tensor logical_bytes", permissive=True)
        entries = _rank_shard_entries(rank_weight_shards, tensor_id)
        shard_source = "rank_weight_shards"
        if entries is not None:
            consumers = tensor_consumers.get(tensor_id, [])
            expected_rank_ids = set()
            for consumer in consumers:
                consumer_stage = _operator_stage(
                    consumer, layer_stages, canonical_parallel.pp_degree
                )
                expected_rank_ids.update(
                    rank.rank
                    for rank in runtime_plan.ranks
                    if rank.pp_rank == consumer_stage
                )
            expert_sharded = any(
                str(consumer.attributes.get("placement_group", ""))
                == "experts"
                or consumer.op_kind == "moe_experts"
                for consumer in consumers
            )
            tensor_plans.append(
                _tensor_plan_from_rank_shards(
                    scenario,
                    tensor_id,
                    entries,
                    logical_bytes,
                    runtime_plan.ranks,
                    tuple(sorted(expected_rank_ids)),
                    expert_sharded,
                    source_name=shard_source,
                    default_residency=str(
                        detail.get("residency", "rank_sharded_storage")
                    ),
                )
            )
            continue
        authoring_plan = _tensor_plan_from_authoring_placement(
            scenario,
            tensor_id,
            detail,
            runtime_plan.ranks,
            options.include_unplaced_weight_tensors,
        )
        if authoring_plan is not None:
            tensor_plans.append(authoring_plan)
            if not authoring_plan.replicas and model_tensor is not None and model_tensor.role == "weight":
                diagnostics.append(
                    CompilationDiagnostic(
                        severity="warning",
                        code="weight_tensor_unplaced",
                        message="{} has no explicit physical replica; the compiler did not infer one".format(tensor_id),
                        source_ref="placement.tensor_to_component.{}".format(tensor_id),
                    )
                )
    return (
        PlacementPlan(
            operator_targets=tuple(
                sorted(
                    operator_targets_by_pair.values(),
                    key=lambda item: (item.operator_id, item.rank_id),
                )
            ),
            tensor_plans=tuple(tensor_plans),
            sub_operator_targets=tuple(
                sorted(
                    sub_operator_targets,
                    key=lambda item: (
                        item.sub_operator_id,
                        item.rank_id,
                    ),
                )
            ),
            attributes={
                "model_name": scenario.placement.model_name,
                "hardware_name": scenario.placement.hardware_name,
                "weights_resident": scenario.weights_resident,
                "kv_policy": to_primitive(scenario.placement.kv_policy),
                "metadata": dict(scenario.placement.metadata),
                "source_schema_version": scenario.placement.schema_version,
            },
            provenance=(
                _origin(
                    scenario,
                    "ScenarioConfig.placement",
                    scenario.placement.model_name,
                    "normalize_rank_aware_placement",
                ),
            ),
        ),
        tuple(diagnostics),
    )


class ScenarioCompilerV1:
    """Compile V4 ``ScenarioConfig`` values to canonical schema 1.1."""

    def __init__(self, options: Optional[CompilerOptions] = None) -> None:
        self.options = options or CompilerOptions()

    def compile(self, scenario: ScenarioConfig) -> CanonicalScenario:
        if not isinstance(scenario, ScenarioConfig):
            raise CanonicalizationError(
                CompilationPhase.READ_SOURCE,
                "source must be a ScenarioConfig",
            )
        execution_layers = _validated_model_execution_layers(
            scenario, CompilationPhase.READ_SOURCE
        )
        if scenario.placement.model_name != scenario.model.name:
            raise CanonicalizationError(
                CompilationPhase.READ_SOURCE,
                "placement.model_name does not match model.name",
                "placement.model_name",
            )
        if scenario.placement.hardware_name != scenario.hardware.name:
            raise CanonicalizationError(
                CompilationPhase.READ_SOURCE,
                "placement.hardware_name does not match hardware.name",
                "placement.hardware_name",
            )
        canonical_parallel, runtime_plan = _build_parallel_plan(scenario)
        layer_stages = {
            item.layer_id: item.stage_id
            for item in canonical_parallel.layer_stages
        }
        hardware = _build_hardware_graph(scenario)
        model = _build_model_graph(
            scenario,
            execution_layers,
            layer_stages=layer_stages,
            pp_degree=canonical_parallel.pp_degree,
        )
        workload = _build_workload_graph(scenario, self.options)
        placement, diagnostics = _build_placement_plan(
            scenario,
            model,
            canonical_parallel,
            runtime_plan,
            self.options,
        )
        placement_errors = tuple(
            item for item in diagnostics if item.severity == "error"
        )
        if placement_errors:
            raise CanonicalizationError(
                CompilationPhase.PLAN_PLACEMENT,
                "; ".join(
                    "{}: {}".format(item.code, item.message)
                    for item in placement_errors
                ),
                placement_errors[0].source_ref,
                diagnostics=placement_errors,
            )
        placement_status = (
            "partial"
            if any(item.severity == "error" for item in diagnostics)
            else "completed"
        )
        source_digest = stable_hash(scenario)
        stages = (
            CompilationStage(
                stage_id=CompilationPhase.READ_SOURCE.value,
                status="completed",
                input_refs=("ScenarioConfig",),
                output_refs=("v4_authoring_snapshot",),
            ),
            CompilationStage(
                stage_id=CompilationPhase.BUILD_GRAPHS.value,
                status="completed",
                input_refs=("ScenarioConfig.hardware", "ScenarioConfig.model", "ScenarioConfig.workload"),
                output_refs=("HardwareGraph", "ModelGraph", "WorkloadGraph"),
            ),
            CompilationStage(
                stage_id=CompilationPhase.PLAN_PARALLELISM.value,
                status="completed",
                input_refs=("PlacementSpec.parallel",),
                output_refs=("ParallelPlan",),
            ),
            CompilationStage(
                stage_id=CompilationPhase.PLAN_PLACEMENT.value,
                status=placement_status,
                input_refs=("PlacementSpec", "ParallelPlan", "ModelGraph"),
                output_refs=("PlacementPlan",),
                diagnostics=diagnostics,
            ),
            CompilationStage(
                stage_id=CompilationPhase.VALIDATE_CANONICAL.value,
                status="completed" if placement_status == "completed" else "partial",
                input_refs=("CanonicalScenario",),
                output_refs=("schema:1.1",),
            ),
        )
        provenance = (
            _origin(
                scenario,
                "ScenarioConfig",
                scenario.name,
                "compile_v4_authoring_to_canonical_v1_1",
                source_digest=source_digest,
                source_section_versions={
                    "hardware": scenario.hardware.schema_version,
                    "model": scenario.model.schema_version,
                    "placement": scenario.placement.schema_version,
                    "workload": scenario.workload.schema_version,
                },
            ),
        )
        return CanonicalScenario(
            scenario_id=scenario.name,
            hardware=hardware,
            model=model,
            workload=workload,
            parallel_plan=canonical_parallel,
            placement_plan=placement,
            compilation=CompilationRecord(
                compiler_id=COMPILER_ID,
                compiler_version=COMPILER_VERSION,
                stages=stages,
                source_digest=source_digest,
                diagnostics=diagnostics,
            ),
            provenance=provenance,
            assumptions=tuple(scenario.assumptions),
            attributes={
                "source_schema_version": scenario.schema_version,
                "canonical_schema_version": SCHEMA_V1_VERSION,
                "weights_resident": scenario.weights_resident,
                "profiles": {
                    "components": to_primitive(
                        scenario.component_profiles
                    ),
                    "component_bindings": {
                        component.component_id: {
                            "profile_kind": normalize_cost_profile_kind(
                                component.normalized_kind
                            ),
                            "cost_profile_id": component.cost_profile_id,
                        }
                        for component in scenario.hardware.components
                        if component.cost_profile_id is not None
                    },
                    "host_orchestration": to_primitive(
                        scenario.host_orchestration_profile
                    ),
                    "fusion": to_primitive(scenario.fusion_policy),
                    "cim_interconnect": to_primitive(scenario.cim_interconnect),
                    "runtime": to_primitive(scenario.runtime_profile),
                    "host_output": to_primitive(
                        scenario.host_output_contract
                    ),
                    "sampling": to_primitive(scenario.sampling_policy),
                    "llama_cpp": to_primitive(scenario.llama_cpp_config),
                    "llama_cpp_fingerprint": (
                        scenario.llama_cpp_config.fingerprint
                        if scenario.llama_cpp_config is not None else None
                    ),
                },
            },
        )


def compile_canonical_scenario(
    scenario: ScenarioConfig, *, options: Optional[CompilerOptions] = None
) -> CanonicalScenario:
    """Compile one V4 authoring scenario to canonical schema 1.1."""

    return ScenarioCompilerV1(options).compile(scenario)


__all__ = [
    "COMPILER_ID",
    "COMPILER_VERSION",
    "CanonicalizationError",
    "CompilationPhase",
    "CompilerOptions",
    "ScenarioCompilerV1",
    "compile_canonical_scenario",
]
