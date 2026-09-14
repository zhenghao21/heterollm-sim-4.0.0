"""Internal deterministic runtime control-plane placement planning.

The control plane deliberately searches only placement.  Hardware topology,
parallel degrees/rank mappings, and KV/offload policy are immutable deployment
inputs.  Workload is preserved unchanged but request/scheduler/workload-MTP
fields do not influence placement or its fingerprint.  Search objectives use a
disclosed analytical service-time surrogate;
the returned placement is subsequently checked by the normal scenario
validator, so a surrogate optimum is never presented as a simulated result.

The standard-library heuristic is always available.  ``mode="optimal"`` uses
either the deterministic built-in branch-and-bound solver or an optional
OR-Tools CP-SAT adapter.  Capacity and topology reachability are hard
constraints in both paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
from itertools import product
import math
import time
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .communication import TopologyRouter
from .config import ScenarioConfig
from .contracts import OperatorClass
from .cost_models import (
    CPUProfile,
    DigitalSramCimProfile,
    ElementwiseWorkload,
    FusedAttentionWorkload,
    GemmWorkload,
    GPUProfile,
    HBMProfile,
    HostMemoryProfile,
    MemoryWorkload,
    ReductionWorkload,
    estimate_cim_gemm,
    estimate_cpu_elementwise,
    estimate_cpu_gemm,
    estimate_cpu_memory,
    estimate_cpu_reduction,
    estimate_gpu_elementwise,
    estimate_gpu_gemm,
    estimate_gpu_memory,
    estimate_gpu_reduction,
)
from .ir import (
    ACTIVE_MEMORY_COMPONENT_KINDS,
    OFFLOAD_STORAGE_COMPONENT_KINDS,
    STORAGE_COMPONENT_KINDS,
    ComponentSpec,
    LayerSpec,
    ModelGraphExecutionView,
    PlacementSpec,
    MTPExecutionDescriptor,
    model_graph_execution_view,
    normalize_component_kind,
)
from .control_plane_state import (
    MAPPING_FINGERPRINT_ALGORITHM,
    MAPPING_FINGERPRINT_SCHEMA,
    mapping_input_fingerprint,
)
from .parallel import ParallelPlan, build_parallel_plan
from .precision import layer_precision_bits
from .projection_descriptors import (
    materialize_weight_projection,
    resolve_attention_execution_descriptor,
)
from .planner import non_gemm_cim_mapping_diagnostics, validate_scenario
from .serde import to_primitive


_MODES = frozenset({"heuristic", "optimal"})
_OBJECTIVES = frozenset({"balanced", "ttft", "tpot", "throughput"})
_SOLVERS = frozenset({"auto", "builtin", "ortools"})
_INFINITE_CAPACITY = (1 << 62) - 1
_COST_QUANTIZATION_SCALE = 1000
_COLD_CIM_BACKING_COMPONENT_KINDS = (
    OFFLOAD_STORAGE_COMPONENT_KINDS
    | frozenset({"host_memory", "ddr", "dram", "ddr_memory", "memory"})
)


@dataclass(frozen=True)
class PlacementPolicy:
    """Controls runtime placement without changing topology or parallelism."""

    mode: str = "heuristic"
    objective: str = "balanced"
    time_limit_s: float = 60.0
    solver: str = "auto"
    allow_cold_cim_streaming: bool = False
    design_prefill_tokens: int = 2048
    design_decode_batch_size: int = 1
    design_throughput_tokens: int = 4096
    gpu_loadable_layers: Optional[int] = None
    gpu_loadable_order: str = "tail"
    tied_weight_runtime_copies: bool = False

    def __post_init__(self) -> None:
        mode = _normalized(self.mode)
        objective = _normalized(self.objective)
        solver = _normalized(self.solver)
        gpu_loadable_order = _normalized(self.gpu_loadable_order)
        if mode not in _MODES:
            raise ValueError("mode 必须为 heuristic 或 optimal")
        if objective not in _OBJECTIVES:
            raise ValueError(
                "objective 必须为 balanced、ttft、tpot 或 throughput"
            )
        if solver not in _SOLVERS:
            raise ValueError("solver 必须为 auto、builtin 或 ortools")
        if gpu_loadable_order not in {"tail", "llama_tail"}:
            raise ValueError("gpu_loadable_order 目前必须为 tail 或 llama_tail")
        if (
            self.gpu_loadable_layers is not None
            and (
                isinstance(self.gpu_loadable_layers, bool)
                or not isinstance(self.gpu_loadable_layers, int)
                or self.gpu_loadable_layers < 0
            )
        ):
            raise ValueError("gpu_loadable_layers 必须是非负整数或 None")
        if not isinstance(self.tied_weight_runtime_copies, bool):
            raise ValueError("tied_weight_runtime_copies 必须是 bool")
        if (
            isinstance(self.time_limit_s, bool)
            or not isinstance(self.time_limit_s, (int, float))
            or not math.isfinite(float(self.time_limit_s))
            or float(self.time_limit_s) <= 0.0
        ):
            raise ValueError("time_limit_s 必须是有限正数")
        if not isinstance(self.allow_cold_cim_streaming, bool):
            raise ValueError("allow_cold_cim_streaming 必须是布尔值")
        for name in (
            "design_prefill_tokens",
            "design_decode_batch_size",
            "design_throughput_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} 必须是正整数".format(name))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "gpu_loadable_order", gpu_loadable_order)
        object.__setattr__(self, "time_limit_s", float(self.time_limit_s))


@dataclass(frozen=True)
class PlacementAction:
    """One derived operator or standalone-state placement decision."""

    item_id: str
    kind: str
    component_id: str
    tensor_id: Optional[str] = None
    tensor_component_id: Optional[str] = None
    tensor_bytes: int = 0
    padded_weight_bytes: int = 0
    cim_eligible: bool = False
    mapping_key: Optional[str] = None
    analytical_cost: float = 0.0
    reason: str = ""
    execution_component_ids: Tuple[str, ...] = ()
    physical_tensor_component_ids: Tuple[str, ...] = ()
    rank_tensor_shards: Tuple["RuntimeTensorShard", ...] = ()
    rank_execution_targets: Tuple["RuntimeExecutionTarget", ...] = ()


@dataclass(frozen=True)
class RuntimeExecutionTarget:
    """One operator target bound to an exact logical TP/PP/EP rank."""

    rank_id: int
    tp_rank: int
    pp_rank: int
    ep_rank: int
    compute_component_id: str
    component_id: str


@dataclass(frozen=True)
class RuntimeTensorShard:
    """One rank-local physical view of a logical weight tensor.

    ``logical_bytes`` is the unpadded portion owned by the logical rank while
    ``physical_bytes`` is the capacity charged to its selected storage.  Dense
    weights are TP-sharded and replicated across EP groups; expert weights are
    sharded across both TP and EP ranks.
    """

    rank: int
    tp_rank: int
    pp_rank: int
    ep_rank: int
    compute_component_id: str
    storage_component_id: str
    logical_bytes: int
    physical_bytes: int
    shard_kind: str


@dataclass(frozen=True)
class UnplacedRequirement:
    """A requirement that could not satisfy hard capacity/reachability rules."""

    item_id: str
    kind: str
    tensor_id: Optional[str]
    required_bytes: int
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementDecision:
    """Internal runtime control-plane placement decision."""

    placement: PlacementSpec
    decisions: Tuple[PlacementAction, ...]
    reasons: Tuple[str, ...]
    warnings: Tuple[str, ...]
    information: Tuple[str, ...]
    unplaced: Tuple[UnplacedRequirement, ...]
    fully_placed: bool
    status: str
    optimality_proven: bool
    objective: str
    objective_value: Optional[float]
    lower_bound: Optional[float]
    gap: Optional[float]
    elapsed_s: float
    solver: str
    weights_resident: bool
    input_fingerprint: str
    current_input_fingerprint: str
    mapping_stale: bool = False

    @property
    def placement_patch(self) -> Dict[str, Any]:
        """Return fields needed to apply the mapping to a scenario."""

        return {
            "op_to_component": dict(self.placement.op_to_component),
            "tensor_to_component": dict(self.placement.tensor_to_component),
            "tensor_bytes": dict(self.placement.tensor_bytes),
            "metadata": dict(self.placement.metadata),
        }

    def apply(self, scenario: ScenarioConfig) -> ScenarioConfig:
        """Return a scenario with only ``placement`` replaced."""

        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario 必须是 ScenarioConfig")
        return replace(scenario, placement=self.placement)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "status": self.status,
            "fully_placed": self.fully_placed,
            "optimality_proven": self.optimality_proven,
            "objective": self.objective,
            "objective_value": self.objective_value,
            "lower_bound": self.lower_bound,
            "gap": self.gap,
            "elapsed_s": self.elapsed_s,
            "solver": self.solver,
            "weights_resident": self.weights_resident,
            "fingerprint_algorithm": MAPPING_FINGERPRINT_ALGORITHM,
            "input_fingerprint": self.input_fingerprint,
            "current_input_fingerprint": self.current_input_fingerprint,
            "mapping_stale": self.mapping_stale,
            "placement": to_primitive(self.placement),
            "placement_patch": to_primitive(self.placement_patch),
            "decisions": to_primitive(self.decisions),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "information": list(self.information),
            "unplaced": to_primitive(self.unplaced),
            "surrogate": {
                "name": "analytical-control-plane-placement-service-v4-fusion-aware-quantized",
                "is_full_simulation": False,
                "cost_quantization_scale": _COST_QUANTIZATION_SCALE,
                "cost_unit": "1/1000 analytical nanosecond",
                "unplaced_penalty_rule": (
                    "1 + 每个求解单元的最大量化候选成本之和；"
                    "融合单元按缺失的原始需求数倍增"
                ),
                "optimality_scope": "仅限量化后的分析型放置目标",
                "description": (
                    "按字典序优先最大化已放置需求数量，再在固定 stage/rank 执行集上，"
                    "最小化量化后的 GPU/CIM 分析服务成本、经路由的权重 IO 成本，"
                    "并对满足同 rank GPU 与 SRAM 门限的联合候选计入保守的融合启动收益。"
                ),
            },
        }


@dataclass(frozen=True)
class _Matrix:
    k: int
    n: int
    resident_count: int = 1
    compute_count: int = 1
    weight_storage_bytes: Optional[int] = None
    weight_metadata_bytes: int = 0
    weight_bits: Optional[int] = None
    packed_weight_formats: Tuple[str, ...] = ()
    packed_weight_transform_operations: int = 0
    projection_id: Optional[str] = None
    descriptor_audit: Mapping[str, Any] = field(default_factory=dict)
    declared_allocation_bytes: Optional[int] = None


@dataclass(frozen=True)
class _Requirement:
    item_id: str
    kind: str
    mapping_key: Optional[str]
    tensor_id: Optional[str]
    tensor_bytes: int
    matrices: Tuple[_Matrix, ...] = ()
    layer: Optional[LayerSpec] = None
    cim_eligible: bool = False
    state_tensor: bool = False
    fixed_component: Optional[str] = None
    logical_alias: Optional[str] = None
    operator_class: OperatorClass = OperatorClass.GEMM
    elements_per_token: int = 0
    operation_elements_per_token: int = 0
    output_elements_per_token: int = 1
    operations_per_element: int = 1
    fixed_operations_per_token: int = 0
    transcendental_operations_per_element: int = 0
    fixed_transcendental_operations_per_token: int = 0
    input_count: int = 1
    read_bytes_per_token: int = 0
    write_bytes_per_token: int = 0
    dynamic_rhs: bool = False
    context_scaled_elements: bool = False
    attention_dynamic_kind: Optional[str] = None
    attention_query_width: int = 0
    attention_kv_width: int = 0
    attention_score_heads: int = 0


@dataclass(frozen=True)
class _Candidate:
    requirement_index: int
    component_id: str
    tensor_component_id: Optional[str]
    cost: float
    usage: Tuple[Tuple[str, int], ...]
    padded_weight_bytes: int = 0
    reason: str = ""
    execution_component_ids: Tuple[str, ...] = ()
    physical_tensor_component_ids: Tuple[str, ...] = ()
    rank_tensor_shards: Tuple[RuntimeTensorShard, ...] = ()
    rank_execution_targets: Tuple[RuntimeExecutionTarget, ...] = ()
    streaming_backing_component_id: Optional[str] = None
    member_candidates: Tuple[Optional["_Candidate"], ...] = ()
    missing_count: int = 0
    choice_signature: str = ""
    fusion_credit: float = 0.0

    @property
    def signature(self) -> Tuple[str, str]:
        return (
            self.choice_signature or self.component_id,
            self.tensor_component_id or "",
        )


@dataclass(frozen=True)
class _FusionOpportunity:
    opportunity_id: str
    group: str
    variant: str
    layer_id: str
    member_indices: Tuple[int, ...]
    member_op_keys: Tuple[str, ...]
    working_set_bytes: int
    eliminated_launches: int


@dataclass(frozen=True)
class _SolveUnit:
    requirement_indices: Tuple[int, ...]
    candidates: Tuple[_Candidate, ...]


@dataclass(frozen=True)
class _ExecutionRank:
    rank: int
    tp_rank: int
    pp_rank: int
    ep_rank: int
    component_id: str
    memory_component_id: Optional[str] = None
    cim_component_id: Optional[str] = None


@dataclass(frozen=True)
class _SolveResult:
    assignment: Tuple[Optional[_Candidate], ...]
    completed: bool
    objective_value: float
    lower_bound: Optional[float]
    solver: str


@dataclass(frozen=True)
class _MappingControls:
    previous_generated_op_keys: FrozenSet[str]
    previous_generated_tensor_ids: FrozenSet[str]


@dataclass(frozen=True)
class _MappingRunContext:
    """Validated graph/parallel state shared by one complete mapping run."""

    execution_view: ModelGraphExecutionView
    parallel_plan: ParallelPlan
    deadline: float
    load_policy_targets: Mapping[str, str] = field(default_factory=dict)


class _MappingDeadlineExceeded(RuntimeError):
    """Internal cooperative stop used while generating placement candidates."""


def _run_parallel_plan(
    scenario: ScenarioConfig,
    run_context: Optional[_MappingRunContext] = None,
) -> ParallelPlan:
    return (
        run_context.parallel_plan
        if run_context is not None
        else build_parallel_plan(scenario)
    )


def _run_execution_view(
    scenario: ScenarioConfig,
    run_context: Optional[_MappingRunContext] = None,
) -> ModelGraphExecutionView:
    return (
        run_context.execution_view
        if run_context is not None
        else model_graph_execution_view(
            scenario.model.graph,
            schema_version=scenario.model.schema_version,
        )
    )


def _check_mapping_deadline(
    run_context: Optional[_MappingRunContext],
) -> None:
    if run_context is not None and time.monotonic() >= run_context.deadline:
        raise _MappingDeadlineExceeded


def _authored_policy_options(scenario: ScenarioConfig) -> Dict[str, Any]:
    """Read the immutable load settings consumed by a default CPU plan."""

    control_plane = scenario.placement.metadata.get("control_plane", {})
    if control_plane is None:
        return {}
    if not isinstance(control_plane, Mapping):
        raise ValueError("placement.metadata.control_plane 必须是 mapping")
    policy = control_plane.get("policy", {})
    if policy is None:
        return {}
    if not isinstance(policy, Mapping):
        raise ValueError("placement.metadata.control_plane.policy 必须是 mapping")
    options = policy.get("options", {})
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ValueError(
            "placement.metadata.control_plane.policy.options 必须是 mapping"
        )
    return dict(options)


def _model_contracts(scenario: ScenarioConfig) -> Tuple[Mapping[str, Any], ...]:
    metadata = scenario.model.metadata
    if not isinstance(metadata, Mapping):
        return ()
    contracts: List[Mapping[str, Any]] = []
    for key in ("gguf", "runtime_cost_contract"):
        value = metadata.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError("model.metadata.{} 必须是 mapping".format(key))
        contracts.append(value)
    return tuple(contracts)


def _model_weight_aliases(scenario: ScenarioConfig) -> Dict[str, str]:
    """Return canonical, fail-closed physical ownership aliases."""

    aliases: Dict[str, str] = {}
    for contract in _model_contracts(scenario):
        raw = contract.get("weight_aliases", {})
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError("model weight_aliases 必须是 mapping")
        for raw_alias, raw_owner in raw.items():
            if (
                not isinstance(raw_alias, str)
                or not raw_alias.strip()
                or not isinstance(raw_owner, str)
                or not raw_owner.strip()
            ):
                raise ValueError("model weight_aliases 必须映射非空字符串")
            alias = raw_alias.strip()
            owner = raw_owner.strip()
            previous = aliases.get(alias)
            if previous is not None and previous != owner:
                raise ValueError(
                    "model weight_aliases 对 {} 声明了冲突物理 owner".format(
                        alias
                    )
                )
            aliases[alias] = owner
    if scenario.model.output_weight_bytes <= 0:
        aliases.setdefault("lm_head_weights", "embedding_weights")

    byte_evidence: Dict[str, set[int]] = {}
    concrete_owners = set()
    for tensor in scenario.model.graph.tensors:
        if str(tensor.role) != "weight":
            continue
        concrete_owners.add(tensor.tensor_id)
        values = []
        if tensor.logical_bytes is not None:
            values.append(tensor.logical_bytes)
        if isinstance(tensor.attributes, Mapping):
            for key in ("physical_bytes", "allocation_bytes", "storage_bytes"):
                value = tensor.attributes.get(key)
                if value is not None:
                    values.append(value)
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "model graph weight byte evidence must be a non-negative integer"
                )
            if value > 0:
                byte_evidence.setdefault(tensor.tensor_id, set()).add(value)
    embedding_bytes = scenario.model.embedding_weight_bytes
    if embedding_bytes is not None:
        if (
            isinstance(embedding_bytes, bool)
            or not isinstance(embedding_bytes, int)
            or embedding_bytes < 0
        ):
            raise ValueError(
                "model embedding_weight_bytes must be a non-negative integer"
            )
        if embedding_bytes > 0:
            byte_evidence.setdefault("embedding_weights", set()).add(
                embedding_bytes
            )

    for contract in _model_contracts(scenario):
        physical_layout = contract.get("physical_layout", {})
        if physical_layout is None:
            continue
        if not isinstance(physical_layout, Mapping):
            raise ValueError("model physical_layout must be a mapping")
        tensor_evidence = physical_layout.get("tensor_evidence", {})
        if tensor_evidence is None:
            continue
        if not isinstance(tensor_evidence, Mapping):
            raise ValueError(
                "model physical_layout.tensor_evidence must be a mapping"
            )
        for raw_tensor_id, raw_details in tensor_evidence.items():
            tensor_id = str(raw_tensor_id).strip()
            if not tensor_id or not isinstance(raw_details, Mapping):
                raise ValueError(
                    "model tensor_evidence entries must contain tensor mappings"
                )
            concrete_owners.add(tensor_id)
            value = raw_details.get("physical_bytes")
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "model tensor_evidence physical_bytes must be a non-negative integer"
                )
            if value > 0:
                byte_evidence.setdefault(tensor_id, set()).add(value)

    for tensor_id, values in byte_evidence.items():
        if len(values) > 1:
            raise ValueError(
                "model weight byte evidence conflicts for {}".format(tensor_id)
            )

    canonical: Dict[str, str] = {}
    visiting = set()

    def resolve(alias: str) -> str:
        existing = canonical.get(alias)
        if existing is not None:
            return existing
        if alias in visiting:
            raise ValueError("model weight_aliases contains a cycle at {}".format(alias))
        visiting.add(alias)
        owner = aliases[alias]
        terminal = resolve(owner) if owner in aliases else owner
        visiting.remove(alias)
        if terminal not in concrete_owners:
            raise ValueError(
                "model weight alias {} resolves to missing physical owner {}".format(
                    alias, terminal
                )
            )
        alias_values = byte_evidence.get(alias, set())
        owner_values = byte_evidence.get(terminal, set())
        if alias_values and owner_values and alias_values != owner_values:
            raise ValueError(
                "model weight alias {} byte evidence does not match owner {}".format(
                    alias, terminal
                )
            )
        canonical[alias] = terminal
        return terminal

    for alias in sorted(aliases):
        resolve(alias)
    return canonical


def _physical_only_weight_artifacts(
    scenario: ScenarioConfig,
    owned_tensor_ids: FrozenSet[str],
    aliases: Mapping[str, str],
) -> Tuple[Tuple[str, int], ...]:
    """Materialize GGUF tensors absent from the executable graph as weights."""

    evidence: Dict[str, int] = {}
    non_executable_totals: set[int] = set()
    for contract in _model_contracts(scenario):
        raw_non_executable = contract.get("non_executable_tensor_bytes")
        if raw_non_executable is not None:
            if (
                isinstance(raw_non_executable, bool)
                or not isinstance(raw_non_executable, int)
                or raw_non_executable < 0
            ):
                raise ValueError(
                    "model non_executable_tensor_bytes 必须是非负整数"
                )
            non_executable_totals.add(raw_non_executable)
        physical_layout = contract.get("physical_layout", {})
        if physical_layout is None:
            continue
        if not isinstance(physical_layout, Mapping):
            raise ValueError("model physical_layout 必须是 mapping")
        tensor_evidence = physical_layout.get("tensor_evidence", {})
        if tensor_evidence is None:
            continue
        if not isinstance(tensor_evidence, Mapping):
            raise ValueError("model physical_layout.tensor_evidence 必须是 mapping")
        for raw_tensor_id, raw_details in tensor_evidence.items():
            tensor_id = str(raw_tensor_id).strip()
            if not tensor_id or not isinstance(raw_details, Mapping):
                raise ValueError("model tensor_evidence 条目必须包含张量 mapping")
            raw_bytes = raw_details.get("physical_bytes")
            if raw_bytes is None:
                continue
            if (
                isinstance(raw_bytes, bool)
                or not isinstance(raw_bytes, int)
                or raw_bytes <= 0
            ):
                raise ValueError("model tensor_evidence physical_bytes 必须是正整数")
            previous = evidence.get(tensor_id)
            if previous is not None and previous != raw_bytes:
                raise ValueError(
                    "model tensor_evidence 对 {} 声明了冲突字节数".format(
                        tensor_id
                    )
                )
            evidence[tensor_id] = raw_bytes
    if len(non_executable_totals) > 1:
        raise ValueError("model non_executable_tensor_bytes 声明冲突")
    artifacts = tuple(
        sorted(
            (tensor_id, byte_count)
            for tensor_id, byte_count in evidence.items()
            if tensor_id not in owned_tensor_ids and tensor_id not in aliases
        )
    )
    if non_executable_totals:
        expected = next(iter(non_executable_totals))
        actual = sum(byte_count for _, byte_count in artifacts)
        if actual != expected:
            raise ValueError(
                "physical-only GGUF weight bytes {} 与 non_executable_tensor_bytes {} 不一致"
                .format(actual, expected)
            )
    return artifacts


def _load_policy_targets(
    options: PlacementPolicy,
    requirements: Sequence[_Requirement],
) -> Dict[str, str]:
    """Resolve llama-style output/MTP/target-tail load units to CPU/GPU."""

    count = options.gpu_loadable_layers
    if count is None:
        return {}
    output_owner_tensor_ids = {
        str(item.logical_alias)
        for item in requirements
        if item.kind == "lm_head" and item.logical_alias
    }
    output_requirements = {
        item.item_id
        for item in requirements
        if (
            item.kind in {"lm_head", "mtp_aux_head", "weight_artifact"}
            or (
                item.kind != "embedding"
                and item.tensor_id in output_owner_tensor_ids
            )
        )
    }
    mtp_prediction_ids = tuple(
        sorted(
            item.item_id
            for item in requirements
            if item.kind == "mtp_prediction_layer"
        )
    )
    layer_ids: List[str] = []
    for item in requirements:
        if item.layer is None or item.item_id.startswith("mtp."):
            continue
        layer_id = item.layer.layer_id
        if item.item_id.startswith(layer_id + ".") and layer_id not in layer_ids:
            layer_ids.append(layer_id)
    available = (
        (1 if output_requirements else 0)
        + len(mtp_prediction_ids)
        + len(layer_ids)
    )
    if count > available:
        raise ValueError(
            "gpu_loadable_layers={} 超过模型可加载单元 {}".format(
                count, available
            )
        )
    remaining = count
    # Legacy ``tail`` counts output/MTP units before transformer blocks.
    # ``llama_tail`` mirrors llama.cpp --n-gpu-layers semantics.  llama.cpp
    # counts the output layer in the offload budget (25/25 for a 24-layer
    # model), then places the remaining budget on the final transformer
    # blocks.  Keeping the output layer on CPU here would make a GPU lm_head
    # repeatedly pull its 144 MB weight from host memory during decode.
    if options.gpu_loadable_order == "llama_tail":
        output_on_gpu = bool(output_requirements and remaining > 0)
        if output_on_gpu:
            remaining -= 1
        selected_layers = set(layer_ids[len(layer_ids) - min(remaining, len(layer_ids)) :]) if remaining else set()
        selected_mtp_ids = set()
    else:
        output_on_gpu = bool(output_requirements and remaining > 0)
        if output_on_gpu:
            remaining -= 1
        selected_mtp = min(remaining, len(mtp_prediction_ids))
        remaining -= selected_mtp
        selected_mtp_ids = set(mtp_prediction_ids[:selected_mtp])
        selected_layers = set(layer_ids[len(layer_ids) - remaining :]) if remaining else set()
    targets: Dict[str, str] = {}
    for item in requirements:
        if item.state_tensor:
            continue
        if item.kind == "embedding":
            targets[item.item_id] = "cpu"
        elif item.item_id in output_requirements:
            targets[item.item_id] = "gpu" if output_on_gpu else "cpu"
        elif item.kind == "mtp_prediction_layer":
            targets[item.item_id] = (
                "gpu" if item.item_id in selected_mtp_ids else "cpu"
            )
        elif item.layer is not None and item.layer.layer_id in layer_ids:
            targets[item.item_id] = (
                "gpu" if item.layer.layer_id in selected_layers else "cpu"
            )
    return targets


def _split_tied_weight_runtime_copies(
    scenario: ScenarioConfig,
    options: PlacementPolicy,
) -> bool:
    """Gate the explicit warm CPU-input/GPU-output tied-weight layout."""

    if (
        not options.tied_weight_runtime_copies
        or options.gpu_loadable_layers is None
        or options.gpu_loadable_layers <= 0
        or scenario.model.vocabulary_size <= 0
        or (scenario.model.embedding_weight_bytes or 0) <= 0
        or _model_weight_aliases(scenario).get("lm_head_weights")
        != "embedding_weights"
    ):
        return False
    component_kinds = {
        _kind(component) for component in scenario.hardware.components
    }
    if not {"cpu", "gpu"}.issubset(component_kinds):
        return False
    if not scenario.weights_resident:
        # ponytail: warm-only until cold initialization is a first-class event.
        raise ValueError(
            "tied_weight_runtime_copies 目前只支持 weights_resident=true；"
            "冷态跨设备副本的初始化/复制生命周期尚未建模"
        )
    return True


def _logical_alias_reason(requirement: _Requirement) -> str:
    if not requirement.logical_alias:
        return ""
    if (
        requirement.kind == "lm_head"
        and requirement.tensor_id == "lm_head_weights"
        and requirement.logical_alias == "embedding_weights"
    ):
        return (
            "；文件权重与 embedding_weights 共享，跨设备运行时副本单独计容量"
        )
    return "；lm_head_weights 与 {} 共享权重，不重复计容量".format(
        requirement.logical_alias
    )


def plan_runtime_placement(
    scenario: ScenarioConfig,
    policy: Optional[PlacementPolicy] = None,
) -> PlacementDecision:
    """Derive and place operators/tensors for ``scenario``.

    The input scenario is never mutated.  Existing topology, parallel/KV
    policy, offload fields and workload are preserved in the returned
    :class:`PlacementSpec` or scenario produced by ``result.apply``.
    """

    started = time.monotonic()
    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario 必须是 ScenarioConfig")
    _validate_v4_planner_boundary(scenario)
    if policy is None:
        options = PlacementPolicy(**_authored_policy_options(scenario))
    elif isinstance(policy, Mapping):
        options = PlacementPolicy(**dict(policy))
    elif isinstance(policy, PlacementPolicy):
        options = policy
    else:
        raise TypeError("policy 必须是 PlacementPolicy、mapping 或 None")

    # ModelSpec is frozen, but its nested graph mappings remain mutable.  Run
    # the centralized fail-closed graph gate again before requirement derivation.
    execution_view = model_graph_execution_view(
        scenario.model.graph,
        schema_version=scenario.model.schema_version,
    )
    parallel_plan = build_parallel_plan(scenario, execution_view)
    # Candidate generation is part of the advertised total time limit.  Keep
    # the historical sub-microsecond no-search contract: tests and callers use
    # it to request an incumbent without making timer resolution observable.
    candidate_deadline = (
        math.inf
        if options.time_limit_s < 1.0e-6
        else started + options.time_limit_s
    )
    run_context = _MappingRunContext(
        execution_view=execution_view,
        parallel_plan=parallel_plan,
        deadline=candidate_deadline,
    )

    warnings: List[str] = []
    information: List[str] = []
    controls = _mapping_controls(scenario, warnings)
    reasons = [
        "搜索过程中保持拓扑、TP/PP/EP rank 映射、KV/卸载策略和工作负载不变。",
        (
            "目标函数使用 analytical-control-plane-placement-service-v4-fusion-aware-quantized，"
            "而不是 TTFT/TPOT 事件仿真；最终指标请以放置后场景的校验和仿真结果为准。"
        ),
        (
            "融合组以联合候选参与搜索；只有全部成员位于同一 rank GPU 且设计工作集"
            "不超过有效 SRAM 门限时，才计入保守的 kernel-launch 融合收益。"
        ),
    ]
    split_tied_runtime_copies = _split_tied_weight_runtime_copies(
        scenario, options
    )
    requirements = _derive_requirements(
        scenario,
        execution_view,
        split_tied_runtime_copies=split_tied_runtime_copies,
    )
    run_context = replace(
        run_context,
        load_policy_targets=_load_policy_targets(options, requirements),
    )
    mapping_diagnostics = non_gemm_cim_mapping_diagnostics(scenario)
    blocked_by_operator = {
        str(diagnostic.get("operator_id")): diagnostic
        for diagnostic in mapping_diagnostics
        if diagnostic.get("operator_id") is not None
    }
    blocked_mapping_keys = {
        str(diagnostic.get("requested_mapping_key"))
        for diagnostic in mapping_diagnostics
        if diagnostic.get("requested_mapping_key") is not None
    }
    router = TopologyRouter(scenario.hardware)
    capacities = _component_capacities(scenario)
    for component in scenario.hardware.components:
        if (
            component.capacity_bytes == 0
            and _kind(component) in OFFLOAD_STORAGE_COMPONENT_KINDS
        ):
            warnings.append(
                "卸载存储 {} 的 capacity_bytes=0（容量未知）；控制平面不会将其作为无限容量，"
                "请先声明正容量".format(component.component_id)
            )
        elif (
            component.capacity_bytes == 0
            and _kind(component) in ACTIVE_MEMORY_COMPONENT_KINDS
        ):
            warnings.append(
                "V4 控制平面将活动内存 {} 的 capacity_bytes=0（容量未知）"
                "按不可用处理；不会选择该组件，请先声明正容量".format(
                    component.component_id
                )
            )

    variable_tensor_ids = {
        requirement.tensor_id
        for requirement in requirements
        if requirement.tensor_id
    }
    variable_tensor_ids.update(
        controls.previous_generated_tensor_ids
    )
    base_usage = _base_capacity_usage(scenario, variable_tensor_ids)
    for component_id, used in sorted(base_usage.items()):
        limit = capacities.get(component_id, _INFINITE_CAPACITY)
        if used > limit:
            warnings.append(
                "现有张量在 {} 上占用 {} 字节，超过有效容量 {}"
                .format(component_id, used, limit)
            )

    candidate_lists: List[Tuple[_Candidate, ...]] = []
    rejection_reasons: List[Tuple[str, ...]] = []
    candidate_generation_timed_out = False
    timeout_reason = (
        "候选生成超过控制平面的全程 time_limit_s；其余需求未继续展开，"
        "已返回可用的部分结果"
    )
    for index, requirement in enumerate(requirements):
        blocked_diagnostic = blocked_by_operator.get(requirement.item_id)
        if blocked_diagnostic is None and requirement.mapping_key is not None:
            blocked_diagnostic = blocked_by_operator.get(
                requirement.mapping_key
            )
        if blocked_diagnostic is not None:
            candidate_lists.append(())
            rejection_reasons.append(
                (str(blocked_diagnostic.get("message_en", "non-GEMM CIM target")),)
            )
            continue
        try:
            _check_mapping_deadline(run_context)
            candidates, rejected = _candidates_for_requirement(
                scenario,
                options,
                requirement,
                index,
                router,
                capacities,
                base_usage,
                controls,
                run_context=run_context,
            )
        except _MappingDeadlineExceeded:
            candidate_generation_timed_out = True
            remaining = len(requirements) - index
            candidate_lists.extend([()] * remaining)
            rejection_reasons.extend([(timeout_reason,)] * remaining)
            warnings.append(timeout_reason)
            break
        candidate_lists.append(candidates)
        rejection_reasons.append(rejected)

    fusion_opportunities = _derive_fusion_opportunities(
        scenario,
        options,
        requirements,
        run_context=run_context,
    )
    solve_units = _build_fusion_solve_units(
        scenario,
        requirements,
        candidate_lists,
        fusion_opportunities,
    )
    solve_candidate_lists = tuple(unit.candidates for unit in solve_units)
    solve_missing_weights = tuple(
        len(unit.requirement_indices) for unit in solve_units
    )

    if options.mode == "heuristic":
        assignment = _heuristic_assignment(
            solve_candidate_lists, capacities, base_usage
        )
        penalty_units = _unplaced_penalty_units(solve_candidate_lists)
        objective_value = _quantized_penalized_cost(
            assignment,
            penalty_units,
            solve_missing_weights,
        ) / float(_COST_QUANTIZATION_SCALE)
        solve = _SolveResult(
            assignment=assignment,
            completed=False,
            objective_value=objective_value,
            lower_bound=None,
            solver="heuristic",
        )
    else:
        solve = _solve_optimal(
            solve_candidate_lists,
            capacities,
            base_usage,
            options,
            started,
            warnings,
            missing_weights=solve_missing_weights,
        )
    solve = replace(
        solve,
        assignment=_expand_solve_assignment(
            solve.assignment,
            solve_units,
            len(requirements),
        ),
    )

    op_mapping = dict(scenario.placement.op_to_component)
    tensor_mapping = dict(scenario.placement.tensor_to_component)
    tensor_bytes = dict(scenario.placement.tensor_bytes)
    for key in controls.previous_generated_op_keys:
        if key not in blocked_mapping_keys:
            op_mapping.pop(key, None)
    for tensor_id in controls.previous_generated_tensor_ids:
        tensor_mapping.pop(tensor_id, None)
        tensor_bytes.pop(tensor_id, None)
    for requirement in requirements:
        if requirement.mapping_key:
            if (
                requirement.mapping_key not in blocked_mapping_keys
                and requirement.item_id not in blocked_by_operator
            ):
                op_mapping.pop(requirement.mapping_key, None)
        if requirement.tensor_id:
            tensor_mapping.pop(requirement.tensor_id, None)
            tensor_bytes.pop(requirement.tensor_id, None)

    # Mapping is placement-only.  Residency is an immutable scenario input;
    # ``allow_cold_cim_streaming`` merely permits candidates for an already
    # cold scenario and never rewrites that top-level policy.
    weights_resident = scenario.weights_resident
    decisions: List[PlacementAction] = []
    unplaced: List[UnplacedRequirement] = []
    generated_op_keys: List[str] = []
    generated_tensor_ids: List[str] = []
    cold_cim_backing_components: Dict[str, str] = {}
    for index, (requirement, candidate) in enumerate(
        zip(requirements, solve.assignment)
    ):
        if candidate is None:
            rejected = rejection_reasons[index]
            blocked_diagnostic = blocked_by_operator.get(requirement.item_id)
            if blocked_diagnostic is None and requirement.mapping_key is not None:
                blocked_diagnostic = blocked_by_operator.get(
                    requirement.mapping_key
                )
            reason = (
                "总容量约束排除了所有可行候选项"
                if candidate_lists[index]
                else "; ".join(rejected[:6])
                if rejected
                else "没有可用的放置候选项"
            )
            unplaced.append(
                UnplacedRequirement(
                    item_id=requirement.item_id,
                    kind=requirement.kind,
                    tensor_id=requirement.tensor_id,
                    required_bytes=requirement.tensor_bytes,
                    reason=reason,
                    details=dict(blocked_diagnostic or {}),
                )
            )
            # Fixed state policy is user intent, even when infeasible.  Keep
            # it visible with the computed byte requirement; never silently
            # replace it with a feasible different component.
            if requirement.state_tensor and requirement.fixed_component:
                assert requirement.tensor_id is not None
                tensor_mapping[requirement.tensor_id] = requirement.fixed_component
                tensor_bytes.pop(requirement.tensor_id, None)
            continue
        if requirement.mapping_key:
            op_mapping[requirement.mapping_key] = candidate.component_id
            generated_op_keys.append(requirement.mapping_key)
        if requirement.tensor_id and candidate.tensor_component_id:
            tensor_mapping[requirement.tensor_id] = candidate.tensor_component_id
            if requirement.state_tensor:
                tensor_bytes.pop(requirement.tensor_id, None)
            else:
                placement_bytes = requirement.tensor_bytes
                if candidate.padded_weight_bytes and weights_resident:
                    placement_bytes = candidate.padded_weight_bytes
                elif candidate.rank_tensor_shards:
                    placement_bytes = sum(
                        shard.physical_bytes
                        for shard in candidate.rank_tensor_shards
                        if shard.storage_component_id
                        == candidate.tensor_component_id
                    )
                tensor_bytes[requirement.tensor_id] = placement_bytes
            generated_tensor_ids.append(requirement.tensor_id)
            if candidate.streaming_backing_component_id:
                cold_cim_backing_components[requirement.tensor_id] = (
                    candidate.streaming_backing_component_id
                )
        decisions.append(
            PlacementAction(
                item_id=requirement.item_id,
                kind=requirement.kind,
                component_id=candidate.component_id,
                tensor_id=requirement.tensor_id,
                tensor_component_id=candidate.tensor_component_id,
                tensor_bytes=requirement.tensor_bytes,
                padded_weight_bytes=candidate.padded_weight_bytes,
                cim_eligible=requirement.cim_eligible,
                mapping_key=requirement.mapping_key,
                analytical_cost=candidate.cost,
                reason=candidate.reason,
                execution_component_ids=candidate.execution_component_ids,
                physical_tensor_component_ids=(
                    candidate.physical_tensor_component_ids
                ),
                rank_tensor_shards=candidate.rank_tensor_shards,
                rank_execution_targets=candidate.rank_execution_targets,
            )
        )

    logical_views: Dict[str, int] = {}
    cold_cim_streaming_tensors: Dict[str, int] = {}
    resident_replicas: Dict[str, List[str]] = {}
    padded_tensor_bytes: Dict[str, int] = {}
    cim_total_physical_bytes: Dict[str, int] = {}
    if options.allow_cold_cim_streaming and not scenario.weights_resident:
        component_map = scenario.hardware.component_map()
        for decision in decisions:
            if (
                decision.tensor_id
                and decision.padded_weight_bytes
                and _is_cim(component_map[decision.component_id])
            ):
                cold_cim_streaming_tensors[decision.tensor_id] = decision.tensor_bytes
                tensor_bytes.pop(decision.tensor_id, None)
    for decision in decisions:
        if decision.padded_weight_bytes and decision.tensor_id:
            targets = decision.physical_tensor_component_ids or (
                decision.component_id,
            )
            resident_replicas[decision.tensor_id] = list(targets)
            padded_tensor_bytes[decision.tensor_id] = (
                decision.padded_weight_bytes
            )
            cim_total_physical_bytes[decision.tensor_id] = (
                (
                    sum(
                        shard.physical_bytes
                        for shard in decision.rank_tensor_shards
                    )
                    if decision.rank_tensor_shards
                    else decision.padded_weight_bytes * len(targets)
                )
                if weights_resident
                else 0
            )
    derived_tensor_bytes = {
        decision.tensor_id: decision.tensor_bytes
        for decision in decisions
        if decision.tensor_id
    }
    physical_tensor_bytes = {
        tensor_id: tensor_bytes[tensor_id]
        for tensor_id in derived_tensor_bytes
        if tensor_id in tensor_bytes
    }
    logical_aliases = _model_weight_aliases(scenario)
    weight_tensor_details: Dict[str, Dict[str, Any]] = {}
    for decision in decisions:
        tensor_id = decision.tensor_id
        if not tensor_id or "weight" not in tensor_id:
            continue
        replicas = list(decision.physical_tensor_component_ids)
        is_cim_resident = bool(decision.padded_weight_bytes)
        if tensor_id in logical_views:
            residency = "aggregate_backing_logical_view"
        elif tensor_id in cold_cim_streaming_tensors:
            residency = "cold_cim_transient"
        elif is_cim_resident:
            residency = "warm_cim_resident"
        elif decision.rank_tensor_shards:
            residency = "rank_sharded_storage"
        else:
            residency = "resident_storage"
        rank_total_physical_bytes = sum(
            shard.physical_bytes for shard in decision.rank_tensor_shards
        )
        if tensor_id in cold_cim_streaming_tensors:
            total_physical_bytes = 0
        elif is_cim_resident:
            total_physical_bytes = cim_total_physical_bytes.get(tensor_id)
        elif decision.rank_tensor_shards:
            total_physical_bytes = rank_total_physical_bytes
        else:
            total_physical_bytes = physical_tensor_bytes.get(tensor_id)
        detail = {
            "logical_bytes": decision.tensor_bytes,
            "placement_bytes": physical_tensor_bytes.get(tensor_id),
            "padded_bytes_per_replica": (
                decision.padded_weight_bytes or None
            ),
            "replica_component_ids": replicas,
            "total_physical_bytes": total_physical_bytes,
            "backing_tensor_id": (
                tensor_id
                if tensor_id in cold_cim_streaming_tensors
                else None
            ),
            "backing_component_id": cold_cim_backing_components.get(tensor_id),
            "residency": residency,
            "shard_policy": (
                "tp_ep_expert_shard"
                if decision.kind == "experts"
                and decision.rank_tensor_shards
                else (
                    "tp_shard_with_ep_replication"
                    if decision.rank_tensor_shards
                    else None
                )
            ),
        }
        if split_tied_runtime_copies and decision.kind in {
            "embedding",
            "lm_head",
        }:
            detail.update(
                {
                    "file_owner_tensor_id": "embedding_weights",
                    "runtime_copy_role": (
                        "input_embedding"
                        if decision.kind == "embedding"
                        else "output_head"
                    ),
                    "runtime_copy_of": (
                        "embedding_weights"
                        if decision.kind == "lm_head"
                        else None
                    ),
                }
            )
        weight_tensor_details[tensor_id] = detail
    logical_weight_ids = {
        decision.tensor_id
        for decision in decisions
        if (
            decision.tensor_id
            and "weight" in decision.tensor_id
            and decision.kind != "weight_artifact"
        )
    }
    if split_tied_runtime_copies:
        logical_bytes_by_owner: Dict[str, int] = {}
        for tensor_id in logical_weight_ids:
            owner = logical_aliases.get(tensor_id, tensor_id)
            logical_bytes_by_owner[owner] = max(
                logical_bytes_by_owner.get(owner, 0),
                derived_tensor_bytes[tensor_id],
            )
        derived_logical_model_weight_bytes = sum(
            logical_bytes_by_owner.values()
        )
    else:
        derived_logical_model_weight_bytes = sum(
            derived_tensor_bytes[tensor_id] for tensor_id in logical_weight_ids
        )
    placement_metadata = dict(scenario.placement.metadata)
    options_payload = to_primitive(options)
    if not options.tied_weight_runtime_copies:
        options_payload.pop("tied_weight_runtime_copies", None)
    rank_weight_shards: Dict[str, List[Dict[str, Any]]] = {}
    for decision in decisions:
        if not decision.tensor_id or not decision.rank_tensor_shards:
            continue
        entries: List[Dict[str, Any]] = []
        residency = weight_tensor_details.get(decision.tensor_id, {}).get(
            "residency", "rank_sharded_storage"
        )
        for index, shard in enumerate(decision.rank_tensor_shards):
            if shard.shard_kind == "tp_ep_expert_shard":
                shard_index = shard.ep_rank * parallel_plan.tp_degree + shard.tp_rank
                shard_count = parallel_plan.tp_degree * parallel_plan.ep_degree
            else:
                shard_index = shard.tp_rank
                shard_count = parallel_plan.tp_degree
            shard_id = "{}#shard-{:04d}".format(
                decision.tensor_id, shard_index
            )
            entries.append(
                {
                    "rank_id": shard.rank,
                    "tp_rank": shard.tp_rank,
                    "pp_rank": shard.pp_rank,
                    "ep_rank": shard.ep_rank,
                    "compute_component_id": shard.compute_component_id,
                    "component_id": shard.storage_component_id,
                    "storage_component_id": shard.storage_component_id,
                    "shard_id": shard_id,
                    "replica_id": "{}#rank-{:04d}".format(
                        shard_id, shard.rank
                    ),
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "logical_bytes": shard.logical_bytes,
                    "physical_bytes": (
                        0
                        if residency == "cold_cim_transient"
                        else shard.physical_bytes
                    ),
                    "shard_kind": shard.shard_kind,
                    "residency": residency,
                }
            )
        rank_weight_shards[decision.tensor_id] = entries
    policy_metadata = {
        "options": options_payload,
    }
    evidence_metadata = {
        "fingerprint_algorithm": MAPPING_FINGERPRINT_ALGORITHM,
        "fingerprint_schema": MAPPING_FINGERPRINT_SCHEMA,
        "surrogate": "analytical-control-plane-placement-service-v4-fusion-aware-quantized",
        "cost_quantization_scale": _COST_QUANTIZATION_SCALE,
        "weight_semantics": (
            "cold CIM backing is derived from declared host/offload topology "
            "and recorded per logical tensor without materializing an aggregate "
            "model_weights placement; warm CIM PlacementSpec.tensor_bytes are "
            "padded bytes per replica; "
            "rank_shards charge every rank-local physical shard and keep only "
            "one representative PlacementSpec entry for execution lowering; "
            "main lm_head is a logical alias of embedding_weights"
            + (
                "; explicit warm tied runtime copies retain one file owner and "
                "charge CPU-input/GPU-output capacity separately"
                if split_tied_runtime_copies
                else ""
            )
        ),
    }
    decision_metadata = {
        "operator_execution_targets": {
            decision.mapping_key or decision.item_id: to_primitive(
                decision.rank_execution_targets
            )
            for decision in decisions
            if decision.rank_execution_targets
        },
        "rank_weight_shards": rank_weight_shards,
        "logical_weight_views": logical_views,
        "logical_weight_aliases": logical_aliases,
        "cold_cim_streaming_tensors": cold_cim_streaming_tensors,
        "resident_cim_replicas": resident_replicas,
        "derived_tensor_bytes": derived_tensor_bytes,
        "physical_tensor_bytes": physical_tensor_bytes,
        "padded_tensor_bytes": padded_tensor_bytes,
        "cim_total_physical_bytes": cim_total_physical_bytes,
        "weight_tensor_details": weight_tensor_details,
        "derived_logical_model_weight_bytes": (
            derived_logical_model_weight_bytes
        ),
        "declared_model_weight_bytes": (
            scenario.model.total_declared_weight_bytes
        ),
        "generated_op_keys": sorted(set(generated_op_keys)),
        "generated_tensor_ids": sorted(set(generated_tensor_ids)),
        "fusion_analysis": _selected_fusion_analysis(
            scenario,
            requirements,
            solve.assignment,
            fusion_opportunities,
        ),
    }
    control_plane_metadata = {
        "policy": policy_metadata,
        "decision": decision_metadata,
        "evidence": evidence_metadata,
    }
    placement_metadata["control_plane"] = control_plane_metadata
    placement_metadata["logical_weight_aliases"] = dict(logical_aliases)
    placement_metadata.pop("auto_mapping", None)
    candidate_placement = replace(
        scenario.placement,
        op_to_component=op_mapping,
        tensor_to_component=tensor_mapping,
        tensor_bytes=tensor_bytes,
        metadata=placement_metadata,
    )
    # Fingerprint the scenario clients will actually obtain after merging the
    # placement.  Generated byte ledgers remain excluded by the projection,
    # while newly selected fixed state components are now represented.
    input_fingerprint = mapping_input_fingerprint(
        replace(scenario, placement=candidate_placement),
        options=options_payload,
    )
    evidence_metadata["input_fingerprint"] = input_fingerprint
    placement_metadata["control_plane"] = control_plane_metadata
    placement = replace(candidate_placement, metadata=placement_metadata)
    mapped_scenario = replace(scenario, placement=placement)
    # Placement validation uses an empty workload.  Runtime state admission is
    # intentionally deferred to compile/simulation and must not turn a valid
    # mapping into a different result merely because requests changed.
    validation_workload = replace(
        scenario.workload,
        requests=(),
        request_count=1,
        prompt_tokens=1,
        output_tokens=0,
        arrival_rate_rps=0.0,
        scheduler=replace(scenario.workload.scheduler, max_num_seqs=1),
    )
    validation = validate_scenario(
        replace(mapped_scenario, workload=validation_workload)
    )
    warnings.extend(validation.warnings)
    information.extend(validation.information)
    if validation.errors:
        for validation_error in validation.errors:
            unplaced.append(
                UnplacedRequirement(
                    item_id="scenario_validation",
                    kind="validation",
                    tensor_id=None,
                    required_bytes=0,
                    reason=validation_error,
                    details={
                        "diagnostics": [
                            dict(diagnostic)
                            for diagnostic in validation.diagnostics
                        ]
                    },
                )
            )

    fully_placed = not unplaced
    timed_out = candidate_generation_timed_out or (
        options.mode == "optimal" and not solve.completed
    )
    if fully_placed and options.mode == "optimal" and solve.completed:
        status = "optimal"
        optimality_proven = True
    elif timed_out:
        status = "feasible_timeout" if fully_placed else "partial_timeout"
        optimality_proven = False
    elif fully_placed:
        status = "feasible"
        optimality_proven = False
    else:
        status = "partial"
        optimality_proven = False

    gap = (
        _relative_gap(solve.objective_value, solve.lower_bound)
        if fully_placed
        else None
    )
    elapsed = time.monotonic() - started
    decision_metadata.update(
        {
            "fully_placed": fully_placed,
            "status": status,
            "optimality_proven": optimality_proven,
            "objective": options.objective,
            "objective_value": solve.objective_value,
            "lower_bound": solve.lower_bound,
            "gap": gap,
            "solver": solve.solver,
            "unplaced": [to_primitive(item) for item in unplaced],
        }
    )
    placement_metadata["control_plane"] = control_plane_metadata
    placement = replace(placement, metadata=placement_metadata)
    return PlacementDecision(
        placement=placement,
        decisions=tuple(decisions),
        reasons=tuple(reasons),
        warnings=tuple(_deduplicate(warnings)),
        information=tuple(_deduplicate(information)),
        unplaced=tuple(unplaced),
        fully_placed=fully_placed,
        status=status,
        optimality_proven=optimality_proven,
        objective=options.objective,
        objective_value=solve.objective_value,
        lower_bound=solve.lower_bound,
        gap=gap,
        elapsed_s=elapsed,
        solver=solve.solver,
        weights_resident=weights_resident,
        input_fingerprint=input_fingerprint,
        current_input_fingerprint=input_fingerprint,
        mapping_stale=False,
    )


def _projection_matrix(
    layer: LayerSpec,
    projection_id: str,
    fallback: _Matrix,
) -> _Matrix:
    """Return descriptor-backed geometry/storage or the legacy matrix.

    Projection descriptors describe the GGUF backing representation.  Their
    byte counts are intentionally kept on the matrix instead of replacing the
    logical owner tensor bytes used by the placement capacity ledger.
    """

    projection = materialize_weight_projection(
        layer.metadata,
        projection_id,
    )
    if projection is None:
        return fallback
    return _Matrix(
        k=projection.k,
        n=projection.n,
        resident_count=fallback.resident_count,
        compute_count=fallback.compute_count,
        weight_storage_bytes=projection.weight_storage_bytes,
        weight_metadata_bytes=projection.weight_metadata_bytes,
        weight_bits=projection.weight_bits,
        packed_weight_formats=tuple(
            dict.fromkeys(
                segment.segment.artifact_spec.name
                for segment in projection.segments
            )
        ),
        packed_weight_transform_operations=(
            projection.fused_dequant_operations
        ),
        projection_id=projection.projection_id,
        descriptor_audit=projection.audit_metadata(),
        declared_allocation_bytes=_matrix_bytes(
            fallback.k, fallback.n, _weight_bits(layer)
        ),
    )


def _matrix_declared_allocation_bytes(
    matrix: _Matrix, fallback_weight_bits: int
) -> int:
    if matrix.declared_allocation_bytes is not None:
        return matrix.declared_allocation_bytes
    if matrix.weight_storage_bytes is not None:
        return matrix.weight_storage_bytes + matrix.weight_metadata_bytes
    return _matrix_bytes(matrix.k, matrix.n, fallback_weight_bits)


def _derive_requirements(
    scenario: ScenarioConfig,
    execution_view: Optional[ModelGraphExecutionView] = None,
    *,
    split_tied_runtime_copies: bool = False,
) -> Tuple[_Requirement, ...]:
    requirements: List[_Requirement] = []
    model = scenario.model
    execution_view = execution_view or model_graph_execution_view(
        model.graph, schema_version=model.schema_version
    )
    execution_layers = tuple(
        item.layer for item in execution_view.layer_instances
    )
    first_layer = execution_layers[0]
    last_layer = execution_layers[-1]
    logical_weight_aliases = _model_weight_aliases(scenario)
    lm_head_physical_owner = logical_weight_aliases.get(
        "lm_head_weights",
        "lm_head_weights",
    )
    untied_lm_head = lm_head_physical_owner != "embedding_weights"
    embedding_bytes = model.embedding_weight_bytes or _matrix_bytes(
        first_layer.hidden_size,
        max(1, model.vocabulary_size),
        _weight_bits(first_layer),
    )
    output_head_bytes = model.output_weight_bytes
    if model.vocabulary_size > 0 or embedding_bytes > 0:
        requirements.append(
            _Requirement(
                item_id="embedding",
                kind="embedding",
                mapping_key="embedding",
                tensor_id=(
                    "embedding_weights"
                    if (
                        model.vocabulary_size <= 0
                        or untied_lm_head
                        or split_tied_runtime_copies
                    )
                    else None
                ),
                tensor_bytes=(
                    embedding_bytes
                    if (
                        model.vocabulary_size <= 0
                        or untied_lm_head
                        or split_tied_runtime_copies
                    )
                    else 0
                ),
                layer=first_layer,
                cim_eligible=False,
                operator_class=OperatorClass.MEMORY,
                read_bytes_per_token=max(1, first_layer.hidden_size * 2),
            )
        )

    for layer in execution_layers:
        h = layer.hidden_size
        head_dim = _attention_head_dim(layer)
        q_width = layer.attention_heads * head_dim
        kv_width = layer.effective_kv_heads * head_dim
        attention_descriptor = None
        if not layer.is_linear_attention:
            attention_descriptor = resolve_attention_execution_descriptor(
                layer.metadata,
                attention_heads=layer.attention_heads,
                kv_heads=layer.effective_kv_heads,
                head_dim=head_dim,
                hidden_size=h,
            )
            if attention_descriptor is not None:
                head_dim = attention_descriptor.head_dim
                q_width = attention_descriptor.query_width
                kv_width = attention_descriptor.kv_heads * head_dim
        if layer.is_linear_attention and layer.linear_attention is not None:
            geometry = layer.linear_attention
            mixer_matrices = [
                _projection_matrix(
                    layer,
                    "linear_attention.qkv",
                    _Matrix(
                        h,
                        geometry.query_width
                        + geometry.key_width
                        + geometry.value_width,
                    ),
                ),
                _projection_matrix(
                    layer,
                    "linear_attention.output",
                    _Matrix(geometry.value_width, h),
                ),
            ]
            if geometry.output_gate:
                mixer_matrices.append(
                    _projection_matrix(
                        layer,
                        "linear_attention.output_gate",
                        _Matrix(h, geometry.value_width),
                    )
                )
            mixer_kind = "linear_attention"
            mixer_key = "{}.linear_attention".format(layer.layer_id)
        else:
            mixer_matrices = [
                _projection_matrix(
                    layer,
                    "attention.qkv",
                    _Matrix(h, q_width + 2 * kv_width),
                ),
                _projection_matrix(
                    layer,
                    "attention.output",
                    _Matrix(q_width, h),
                ),
            ]
            mixer_kind = "attention"
            mixer_key = "{}.attention".format(layer.layer_id)

        group_specs: List[Tuple[str, str, Optional[str], str, List[_Matrix], bool]] = [
            (
                "{}.{}".format(layer.layer_id, mixer_kind),
                mixer_kind,
                mixer_key,
                "{}.{}_weights".format(layer.layer_id, mixer_kind),
                mixer_matrices,
                True,
            )
        ]
        if layer.is_moe:
            expert_matrices = [
                _Matrix(
                    h,
                    2 * layer.intermediate_size,
                    layer.num_experts,
                    layer.experts_per_token,
                ),
                _Matrix(
                    layer.intermediate_size,
                    h,
                    layer.num_experts,
                    layer.experts_per_token,
                ),
            ]
            group_specs.extend(
                [
                    (
                        "{}.experts".format(layer.layer_id),
                        "experts",
                        "{}.experts".format(layer.layer_id),
                        "{}.expert_weights".format(layer.layer_id),
                        expert_matrices,
                        True,
                    ),
                    (
                        "{}.router".format(layer.layer_id),
                        "router",
                        "{}.router".format(layer.layer_id),
                        "{}.router_weights".format(layer.layer_id),
                        [_Matrix(h, layer.num_experts)],
                        True,
                    ),
                ]
            )
            if layer.has_shared_expert:
                shared = layer.shared_expert_intermediate_size
                group_specs.append(
                    (
                        "{}.shared_expert".format(layer.layer_id),
                        "shared_expert",
                        "{}.shared_expert".format(layer.layer_id),
                        "{}.shared_expert_weights".format(layer.layer_id),
                        [_Matrix(h, 2 * shared), _Matrix(shared, h)],
                        True,
                    )
                )
                if layer.shared_expert_gate:
                    group_specs.append(
                        (
                            "{}.shared_expert_gate".format(layer.layer_id),
                            "shared_expert_gate",
                            "{}.shared_expert_gate".format(layer.layer_id),
                            "{}.shared_expert_gate_weights".format(layer.layer_id),
                            [_Matrix(h, 1)],
                            True,
                        )
                    )
        else:
            group_specs.append(
                (
                    "{}.mlp".format(layer.layer_id),
                    "mlp",
                    "{}.mlp".format(layer.layer_id),
                    "{}.mlp_weights".format(layer.layer_id),
                    [
                        _projection_matrix(
                            layer,
                            "mlp.up_gate",
                            _Matrix(h, 2 * layer.intermediate_size),
                        ),
                        _projection_matrix(
                            layer,
                            "mlp.down",
                            _Matrix(layer.intermediate_size, h),
                        ),
                    ],
                    True,
                )
            )

        raw_bytes = [
            sum(
                _matrix_declared_allocation_bytes(
                    matrix, _weight_bits(layer)
                )
                * matrix.resident_count
                for matrix in spec[4]
            )
            for spec in group_specs
        ]
        declared_bytes = _allocate_declared_bytes(raw_bytes, layer.weight_bytes)
        for spec, byte_count in zip(group_specs, declared_bytes):
            item_id, kind, key, tensor_id, matrices, eligible = spec
            requirements.append(
                _Requirement(
                    item_id=item_id,
                    kind=kind,
                    mapping_key=key,
                    tensor_id=tensor_id,
                    tensor_bytes=byte_count,
                    matrices=tuple(matrices),
                    layer=layer,
                    cim_eligible=eligible,
                )
            )

        def add_primitive(
            suffix: str,
            kind: str,
            operator_class: OperatorClass,
            *,
            elements_per_token: int,
            operation_elements_per_token: int = 0,
            output_elements_per_token: int = 1,
            operations_per_element: int = 1,
            fixed_operations_per_token: int = 0,
            transcendental_operations_per_element: int = 0,
            fixed_transcendental_operations_per_token: int = 0,
            input_count: int = 1,
            read_bytes_per_token: int = 0,
            write_bytes_per_token: int = 0,
            context_scaled_elements: bool = False,
        ) -> None:
            item_id = "{}.{}".format(layer.layer_id, suffix)
            requirements.append(
                _Requirement(
                    item_id=item_id,
                    kind=kind,
                    mapping_key=item_id,
                    tensor_id=None,
                    tensor_bytes=0,
                    layer=layer,
                    operator_class=operator_class,
                    elements_per_token=max(1, elements_per_token),
                    operation_elements_per_token=max(
                        0, operation_elements_per_token
                    ),
                    output_elements_per_token=max(
                        1, output_elements_per_token
                    ),
                    operations_per_element=max(1, operations_per_element),
                    fixed_operations_per_token=max(
                        0, fixed_operations_per_token
                    ),
                    transcendental_operations_per_element=max(
                        0, transcendental_operations_per_element
                    ),
                    fixed_transcendental_operations_per_token=max(
                        0, fixed_transcendental_operations_per_token
                    ),
                    input_count=max(1, input_count),
                    read_bytes_per_token=max(0, read_bytes_per_token),
                    write_bytes_per_token=max(0, write_bytes_per_token),
                    context_scaled_elements=context_scaled_elements,
                )
            )

        for norm_prefix in ("input_norm", "post_attention_norm"):
            add_primitive(
                norm_prefix + ".reduce",
                "norm_reduce",
                OperatorClass.REDUCTION,
                elements_per_token=h,
                output_elements_per_token=1,
                operations_per_element=2,
            )
            add_primitive(
                norm_prefix + ".apply",
                "norm_apply",
                OperatorClass.ELEMENTWISE,
                elements_per_token=h,
                operations_per_element=4,
                input_count=2,
            )

        if layer.is_linear_attention and layer.linear_attention is not None:
            geometry = layer.linear_attention
            projected_width = (
                geometry.query_width
                + geometry.key_width
                + geometry.value_width
            )
            add_primitive(
                "linear_attention.local_conv",
                "linear_local_conv",
                OperatorClass.REDUCTION,
                elements_per_token=(
                    projected_width * geometry.conv_kernel_size
                ),
                output_elements_per_token=projected_width,
                operations_per_element=2,
            )
            add_primitive(
                "linear_attention.state_update",
                "linear_state_update",
                OperatorClass.ELEMENTWISE,
                elements_per_token=max(
                    1, geometry.recurrent_state_elements
                ),
                operations_per_element=6,
                input_count=2,
            )
            add_primitive(
                "linear_attention.gate_norm.reduce",
                "linear_gate_norm_reduce",
                OperatorClass.REDUCTION,
                elements_per_token=geometry.value_width,
                output_elements_per_token=1,
                operations_per_element=2,
            )
            add_primitive(
                "linear_attention.gate_norm.apply",
                "linear_gate_norm_apply",
                OperatorClass.ELEMENTWISE,
                elements_per_token=geometry.value_width,
                operations_per_element=8,
                input_count=2 if geometry.output_gate else 1,
            )
            add_primitive(
                "linear_attention.residual",
                "residual",
                OperatorClass.ELEMENTWISE,
                elements_per_token=h,
                input_count=2,
            )
        else:
            if attention_descriptor is not None and attention_descriptor.qk_norm:
                for prefix, width, groups in (
                    ("q", q_width, attention_descriptor.query_heads),
                    ("k", kv_width, attention_descriptor.kv_heads),
                ):
                    add_primitive(
                        "attention.{}_norm.reduce".format(prefix),
                        "{}_norm_reduce".format(prefix),
                        OperatorClass.REDUCTION,
                        elements_per_token=width,
                        output_elements_per_token=groups,
                        operations_per_element=2,
                        fixed_operations_per_token=groups,
                    )
                    add_primitive(
                        "attention.{}_norm.apply".format(prefix),
                        "{}_norm_apply".format(prefix),
                        OperatorClass.ELEMENTWISE,
                        elements_per_token=width,
                        operations_per_element=2,
                        fixed_transcendental_operations_per_token=groups,
                        input_count=2,
                    )
            rope_compute_elements = (
                (
                    attention_descriptor.query_heads
                    + attention_descriptor.kv_heads
                )
                * attention_descriptor.rotary_dim
                if attention_descriptor is not None
                else 0
            )
            rope_traffic_elements = q_width + kv_width
            add_primitive(
                "attention.rope",
                "rope",
                OperatorClass.ELEMENTWISE,
                elements_per_token=rope_traffic_elements,
                operation_elements_per_token=rope_compute_elements,
                operations_per_element=3,
                input_count=3 if attention_descriptor is not None else 1,
                read_bytes_per_token=(
                    _storage_bytes(
                        rope_traffic_elements + 2 * rope_compute_elements,
                        _activation_bits(layer),
                    )
                    if attention_descriptor is not None
                    else 0
                ),
                write_bytes_per_token=(
                    _storage_bytes(
                        rope_traffic_elements, _activation_bits(layer)
                    )
                    if attention_descriptor is not None
                    else 0
                ),
            )
            for suffix in ("attention.qk", "attention.pv"):
                item_id = "{}.{}".format(layer.layer_id, suffix)
                dynamic_kind = suffix.rsplit(".", 1)[-1]
                requirements.append(
                    _Requirement(
                        item_id=item_id,
                        kind=suffix.rsplit(".", 1)[-1],
                        mapping_key=item_id,
                        tensor_id=None,
                        tensor_bytes=0,
                        matrices=(
                            (
                                _Matrix(q_width, 1)
                                if dynamic_kind == "qk"
                                else _Matrix(1, q_width)
                            )
                            if attention_descriptor is not None
                            else _Matrix(h, h)
                        ,),
                        layer=layer,
                        cim_eligible=False,
                        operator_class=OperatorClass.GEMM,
                        dynamic_rhs=True,
                        attention_dynamic_kind=(
                            dynamic_kind
                            if attention_descriptor is not None
                            else None
                        ),
                        attention_query_width=(
                            q_width if attention_descriptor is not None else 0
                        ),
                        attention_kv_width=(
                            kv_width if attention_descriptor is not None else 0
                        ),
                        attention_score_heads=(
                            attention_descriptor.query_heads
                            if attention_descriptor is not None
                            else 0
                        ),
                    )
                )
            if attention_descriptor is not None:
                add_primitive(
                    "attention.qk_scale",
                    "qk_scale",
                    OperatorClass.ELEMENTWISE,
                    elements_per_token=attention_descriptor.query_heads,
                    operations_per_element=1,
                    context_scaled_elements=True,
                )
            score_elements_per_token = (
                attention_descriptor.query_heads
                if attention_descriptor is not None
                else h
            )
            add_primitive(
                "attention.softmax.reduce",
                "softmax_reduce",
                OperatorClass.REDUCTION,
                elements_per_token=score_elements_per_token,
                output_elements_per_token=1,
                operations_per_element=2,
                context_scaled_elements=attention_descriptor is not None,
            )
            add_primitive(
                "attention.softmax.normalize",
                "softmax_normalize",
                OperatorClass.ELEMENTWISE,
                elements_per_token=score_elements_per_token,
                operations_per_element=(
                    2 if attention_descriptor is not None else 5
                ),
                transcendental_operations_per_element=(
                    1 if attention_descriptor is not None else 0
                ),
                input_count=2,
                context_scaled_elements=attention_descriptor is not None,
            )
            if attention_descriptor is not None:
                add_primitive(
                    "attention.gate",
                    "attention_gate",
                    OperatorClass.ELEMENTWISE,
                    elements_per_token=attention_descriptor.gate_width,
                    operations_per_element=3,
                    transcendental_operations_per_element=1,
                    input_count=2,
                )
            add_primitive(
                "attention.residual",
                "residual",
                OperatorClass.ELEMENTWISE,
                elements_per_token=h,
                input_count=2,
            )

        if layer.is_moe:
            add_primitive(
                "router.softmax.reduce",
                "router_softmax_reduce",
                OperatorClass.REDUCTION,
                elements_per_token=layer.num_experts,
                output_elements_per_token=1,
                operations_per_element=2,
            )
            add_primitive(
                "router.softmax.normalize",
                "router_softmax_normalize",
                OperatorClass.ELEMENTWISE,
                elements_per_token=layer.num_experts,
                operations_per_element=5,
                input_count=2,
            )
            add_primitive(
                "router.topk",
                "router_topk",
                OperatorClass.REDUCTION,
                elements_per_token=layer.num_experts,
                output_elements_per_token=layer.experts_per_token,
            )
            add_primitive(
                "experts.activation",
                "expert_activation",
                OperatorClass.ELEMENTWISE,
                elements_per_token=2 * layer.intermediate_size,
                operations_per_element=8,
            )
            if layer.has_shared_expert:
                add_primitive(
                    "shared_expert.activation",
                    "shared_expert_activation",
                    OperatorClass.ELEMENTWISE,
                    elements_per_token=(
                        2 * layer.shared_expert_intermediate_size
                    ),
                    operations_per_element=8,
                )
                if layer.shared_expert_gate:
                    add_primitive(
                        "shared_expert.gate_apply",
                        "shared_expert_gate_apply",
                        OperatorClass.ELEMENTWISE,
                        elements_per_token=h,
                        operations_per_element=2,
                        input_count=2,
                    )
            add_primitive(
                "moe.residual",
                "residual",
                OperatorClass.ELEMENTWISE,
                elements_per_token=h,
                input_count=3 if layer.has_shared_expert else 2,
            )
        else:
            add_primitive(
                "mlp.activation",
                "mlp_activation",
                OperatorClass.ELEMENTWISE,
                elements_per_token=2 * layer.intermediate_size,
                operations_per_element=8,
            )
            add_primitive(
                "mlp.residual",
                "residual",
                OperatorClass.ELEMENTWISE,
                elements_per_token=h,
                input_count=2,
            )

    if model.vocabulary_size > 0:
        lm_head_owns_tensor = untied_lm_head and output_head_bytes > 0
        requirements.append(
            _Requirement(
                item_id="lm_head",
                kind="lm_head",
                mapping_key="lm_head",
                tensor_id=(
                    "lm_head_weights"
                    if split_tied_runtime_copies
                    else (
                        lm_head_physical_owner
                        if lm_head_owns_tensor
                        else "embedding_weights"
                        if not untied_lm_head
                        else None
                    )
                ),
                tensor_bytes=(
                    embedding_bytes
                    if split_tied_runtime_copies
                    else (
                        output_head_bytes
                        if lm_head_owns_tensor
                        else embedding_bytes
                        if not untied_lm_head
                        else 0
                    )
                ),
                matrices=(_Matrix(last_layer.hidden_size, model.vocabulary_size),),
                layer=last_layer,
                cim_eligible=True,
                logical_alias=(
                    lm_head_physical_owner
                    if lm_head_physical_owner != "lm_head_weights"
                    else None
                ),
            )
        )

    # MTP is a real typed branch in the authoritative graph.  Each operator
    # owns one independently placeable weight tensor; the removed
    # ``mtp_weights`` aggregate is deliberately not materialized here.
    for descriptor in execution_view.mtp_descriptors:
        requirements.append(
            _mtp_requirement(descriptor, last_layer)
        )

    owned_weight_tensors = frozenset(
        str(requirement.tensor_id)
        for requirement in requirements
        if requirement.tensor_id and requirement.tensor_bytes > 0
    )
    for tensor_id, byte_count in _physical_only_weight_artifacts(
        scenario,
        owned_weight_tensors,
        logical_weight_aliases,
    ):
        requirements.append(
            _Requirement(
                item_id=tensor_id,
                kind="weight_artifact",
                mapping_key=None,
                tensor_id=tensor_id,
                tensor_bytes=byte_count,
                layer=last_layer,
                cim_eligible=False,
                operator_class=OperatorClass.MEMORY,
            )
        )

    state_sizes = _state_tensor_sizes(scenario, execution_view)
    for tensor_id, byte_count in state_sizes:
        # V4 tensor_to_component entries are generated outputs, not authoring
        # locks.  Replanning must be free to relocate every state tensor after
        # a topology/capacity change.  Only the public KV policy carries an
        # explicit fixed state target.
        configured = (
            scenario.placement.kv_policy.cache_component
            if tensor_id == "kv_cache"
            else None
        )
        requirements.append(
            _Requirement(
                item_id=tensor_id,
                kind="state_tensor",
                mapping_key=None,
                tensor_id=tensor_id,
                # Automatic mapping owns only the state location/policy.  The
                # serving planner derives live bytes from admitted requests.
                tensor_bytes=int(byte_count),
                state_tensor=True,
                fixed_component=str(configured) if configured else None,
                operator_class=OperatorClass.MEMORY,
            )
        )
    return tuple(requirements)


def _state_tensor_sizes(
    scenario: ScenarioConfig,
    execution_view: Optional[ModelGraphExecutionView] = None,
) -> Tuple[Tuple[str, int], ...]:
    """Return workload-independent state placement requirements.

    A zero byte count is intentional: KV cache and recurrent linear-attention
    state are runtime allocations whose live size depends on admission and
    paging.  Baking the current request set into placement would make a
    workload edit silently change placement and stale fingerprints.
    """

    result: List[Tuple[str, int]] = []
    execution_view = execution_view or model_graph_execution_view(
        scenario.model.graph,
        schema_version=scenario.model.schema_version,
    )
    layers = tuple(item.layer for item in execution_view.layer_instances)
    if any(not layer.is_linear_attention for layer in layers):
        result.append(("kv_cache", 0))
    if any(layer.is_linear_attention for layer in layers):
        result.append(("linear_state", 0))
    return tuple(result)


def _mtp_requirement(
    descriptor: MTPExecutionDescriptor, layer: LayerSpec
) -> _Requirement:
    """Translate one typed MTP graph operator without aggregate aliases."""

    operator = descriptor.operator
    if operator.op_kind == "mtp_prediction_layer":
        matrix = _Matrix(descriptor.hidden_size, descriptor.hidden_size)
    elif operator.op_kind == "mtp_aux_head":
        matrix = _Matrix(
            descriptor.hidden_size, descriptor.vocabulary_size
        )
    else:  # The execution view is fail-closed; keep this defensive boundary.
        raise ValueError(
            "不支持的 typed MTP 组件类型 {}".format(operator.op_kind)
        )
    return _Requirement(
        item_id=operator.operator_id,
        kind=operator.op_kind,
        mapping_key=operator.operator_id,
        tensor_id=descriptor.weight_tensor.tensor_id,
        tensor_bytes=int(descriptor.weight_bytes),
        matrices=(matrix,),
        layer=layer,
        cim_eligible=True,
    )


def _requirement_execution_ranks(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    gpus: Sequence[ComponentSpec],
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[_ExecutionRank, ...]:
    """Resolve the immutable logical ranks that execute one requirement.

    The authoritative parallel planner is used for both explicit and implicit
    rank mappings so cost-weighted PP stage assignment cannot drift from the
    schedule compiler.
    """

    plan = _run_parallel_plan(scenario, run_context)
    stage = _requirement_stage(scenario, requirement, run_context)
    selected = plan.ranks if stage is None else plan.ranks_for_stage(stage)
    return tuple(
        _ExecutionRank(
            rank=int(rank.rank),
            tp_rank=int(rank.tp_rank),
            pp_rank=int(rank.pp_rank),
            ep_rank=int(rank.ep_rank),
            component_id=str(rank.component_id),
            memory_component_id=(
                str(rank.memory_component_id)
                if rank.memory_component_id
                else None
            ),
            cim_component_id=(
                str(rank.cim_component_id) if rank.cim_component_id else None
            ),
        )
        for rank in selected
    )


def _requirement_weight_ranks(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    gpus: Sequence[ComponentSpec],
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[_ExecutionRank, ...]:
    """Return ranks that need physical access to a requirement's weights."""

    ranks = list(
        _requirement_execution_ranks(
            scenario, requirement, gpus, run_context
        )
    )
    # Tied embeddings are consumed on PP stage zero and by lm_head on the final
    # stage.  PP therefore creates another replica of each TP/EP logical shard.
    if (
        requirement.logical_alias == "embedding_weights"
        and requirement.tensor_id == requirement.logical_alias
        and _run_parallel_plan(scenario, run_context).pp_degree > 1
    ):
        embedding_requirement = replace(
            requirement,
            kind="embedding",
            layer=_run_execution_view(
                scenario, run_context
            ).layer_instances[0].layer,
            logical_alias=None,
        )
        ranks.extend(
            _requirement_execution_ranks(
                scenario, embedding_requirement, gpus, run_context
            )
        )
    return tuple(
        sorted(
            {rank.rank: rank for rank in ranks}.values(),
            key=lambda rank: rank.rank,
        )
    )


def _rank_execution_targets(
    ranks: Sequence[_ExecutionRank],
    configured_component_id: str,
    *,
    cim: bool,
    fixed_compute: bool = False,
) -> Tuple[RuntimeExecutionTarget, ...]:
    return tuple(
        RuntimeExecutionTarget(
            rank_id=rank.rank,
            tp_rank=rank.tp_rank,
            pp_rank=rank.pp_rank,
            ep_rank=rank.ep_rank,
            compute_component_id=rank.component_id,
            component_id=(
                rank.cim_component_id or configured_component_id
                if cim
                else configured_component_id
                if fixed_compute
                else rank.component_id
            ),
        )
        for rank in ranks
    )


def _rank_local_store_pools(
    scenario: ScenarioConfig,
    router: TopologyRouter,
    ranks: Sequence[_ExecutionRank],
    stores: Sequence[ComponentSpec],
    shard_bytes: int,
) -> Tuple[Tuple[ComponentSpec, ...], ...]:
    """Return topology-grounded storage candidates for every logical rank."""

    components = scenario.hardware.component_map()
    adjacent: Dict[str, set[str]] = {}
    for link in scenario.hardware.links:
        adjacent.setdefault(link.source_component, set()).add(
            link.target_component
        )
        adjacent.setdefault(link.target_component, set()).add(
            link.source_component
        )

    result: List[Tuple[ComponentSpec, ...]] = []
    for rank in ranks:
        if rank.memory_component_id:
            explicit = components.get(rank.memory_component_id)
            if (
                explicit is not None
                and _kind(explicit) in ACTIVE_MEMORY_COMPONENT_KINDS
                and _writable(explicit)
            ):
                try:
                    _route_cost(
                        router,
                        explicit.component_id,
                        rank.component_id,
                        max(1, shard_bytes),
                    )
                except ValueError:
                    result.append(())
                else:
                    result.append((explicit,))
                continue
            result.append(())
            continue

        reachable: List[Tuple[int, float, str, ComponentSpec]] = []
        direct_ids = adjacent.get(rank.component_id, set())
        for store in stores:
            kind = _kind(store)
            if kind not in ACTIVE_MEMORY_COMPONENT_KINDS or not _writable(store):
                continue
            try:
                route_cost = _route_cost(
                    router,
                    store.component_id,
                    rank.component_id,
                    max(1, shard_bytes),
                )
            except ValueError:
                continue
            locality = 0 if store.component_id in direct_ids else 1
            reachable.append(
                (locality, route_cost, store.component_id, store)
            )
        reachable.sort(key=lambda item: (item[0], item[1], item[2]))
        if not reachable:
            result.append(())
            continue
        best_locality = reachable[0][0]
        # Keep the full best locality class.  This naturally exposes all HBM
        # stacks directly attached to a GPU while excluding remote HBM until a
        # local pool is genuinely unavailable.
        result.append(
            tuple(item[3] for item in reachable if item[0] == best_locality)
        )
    return tuple(result)


def _rank_local_weight_candidates(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    requirement_index: int,
    router: TopologyRouter,
    ranks: Sequence[_ExecutionRank],
    stores: Sequence[ComponentSpec],
) -> Tuple[
    Tuple[
        Tuple[RuntimeTensorShard, ...],
        Tuple[Tuple[str, int], ...],
        float,
    ],
    ...,
]:
    """Build a bounded set of deterministic rank-local weight stripes."""

    if not ranks or requirement.tensor_bytes <= 0:
        return ()
    parallel = scenario.placement.parallel
    tp_degree = max(1, parallel.tp_degree)
    ep_degree = max(1, parallel.ep_degree)
    shard_degree = tp_degree * ep_degree if requirement.kind == "experts" else tp_degree
    nominal_shard_bytes = _ceil_div(requirement.tensor_bytes, shard_degree)
    pools = _rank_local_store_pools(
        scenario, router, ranks, stores, nominal_shard_bytes
    )
    if any(not pool for pool in pools):
        return ()
    alternatives = max(len(pool) for pool in pools)
    candidates = []
    for alternative in range(alternatives):
        shards: List[RuntimeTensorShard] = []
        usage: Dict[str, int] = {}
        route_cost = 0.0
        for rank, pool in zip(ranks, pools):
            if requirement.kind == "experts":
                shard_index = rank.ep_rank * tp_degree + rank.tp_rank
                shard_kind = "tp_ep_expert_shard"
            else:
                shard_index = rank.tp_rank
                shard_kind = (
                    "tp_shard_ep_replica" if ep_degree > 1 else "tp_shard"
                )
            logical_bytes = requirement.tensor_bytes // shard_degree
            if shard_index < requirement.tensor_bytes % shard_degree:
                logical_bytes += 1
            # Each EP group owns another dense TP replica.  Expert shards use
            # a unique TP×EP index and therefore are not replicated here.
            physical_bytes = logical_bytes
            primary_offset = requirement_index % len(pool)
            store = pool[(primary_offset + alternative) % len(pool)]
            shard = RuntimeTensorShard(
                rank=rank.rank,
                tp_rank=rank.tp_rank,
                pp_rank=rank.pp_rank,
                ep_rank=rank.ep_rank,
                compute_component_id=rank.component_id,
                storage_component_id=store.component_id,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes,
                shard_kind=shard_kind,
            )
            shards.append(shard)
            usage[store.component_id] = (
                usage.get(store.component_id, 0) + physical_bytes
            )
            if store.component_id != rank.component_id:
                route_cost += _route_cost(
                    router,
                    store.component_id,
                    rank.component_id,
                    max(1, physical_bytes),
                )
        # A one-unit quantized preference preserves stable round-robin stripes
        # while still allowing capacity constraints to select a fallback.
        route_cost += alternative / float(_COST_QUANTIZATION_SCALE)
        candidates.append(
            (
                tuple(shards),
                tuple(sorted(usage.items())),
                route_cost,
            )
        )
    return tuple(candidates)


def _rank_shard_geometry(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    rank: _ExecutionRank,
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[int, int, str]:
    parallel = _run_parallel_plan(scenario, run_context)
    if requirement.kind == "experts":
        shard_count = parallel.tp_degree * parallel.ep_degree
        shard_index = rank.ep_rank * parallel.tp_degree + rank.tp_rank
        shard_kind = "tp_ep_expert_shard"
    else:
        shard_count = parallel.tp_degree
        shard_index = rank.tp_rank
        shard_kind = (
            "tp_shard_ep_replica"
            if parallel.ep_degree > 1
            else "tp_shard"
        )
    return shard_index, shard_count, shard_kind


def _logical_rank_shard_bytes(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    rank: _ExecutionRank,
    run_context: Optional[_MappingRunContext] = None,
) -> int:
    shard_index, shard_count, _ = _rank_shard_geometry(
        scenario, requirement, rank, run_context
    )
    result = requirement.tensor_bytes // shard_count
    if shard_index < requirement.tensor_bytes % shard_count:
        result += 1
    return result


def _cim_rank_padded_weight_bytes(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    rank: _ExecutionRank,
    cim_component_id: str,
    run_context: Optional[_MappingRunContext] = None,
) -> int:
    """Return the physical CIM bytes for one TP/EP rank-local shard."""

    profile = scenario.resolve_component_profile(
        cim_component_id, DigitalSramCimProfile
    )
    layer = requirement.layer
    if layer is None:
        raise ValueError("缺少 CIM profile 或层定义")
    activation_bits = _activation_bits(layer)
    weight_bits = _weight_bits(layer)
    if activation_bits not in profile.supported_activation_bits:
        raise ValueError("不支持的 CIM 激活位宽 {}".format(activation_bits))
    if weight_bits not in profile.supported_weight_bits:
        raise ValueError("不支持的 CIM 权重位宽 {}".format(weight_bits))
    parallel = _run_parallel_plan(scenario, run_context)
    total = 0
    for matrix in requirement.matrices:
        local_n = _ceil_div(matrix.n, parallel.tp_degree)
        resident_count = matrix.resident_count
        if requirement.kind == "experts":
            resident_count = matrix.resident_count // parallel.ep_degree
            if rank.ep_rank < matrix.resident_count % parallel.ep_degree:
                resident_count += 1
        if resident_count <= 0:
            continue
        required_accumulator = (
            activation_bits
            + weight_bits
            + int(math.ceil(math.log(max(1, matrix.k), 2)))
            + profile.accumulator_guard_bits
        )
        if min(32, profile.accumulator_bits) < required_accumulator:
            raise ValueError(
                "累加器位宽 {} 小于所需位宽 {}".format(
                    min(32, profile.accumulator_bits), required_accumulator
                )
            )
        padded_elements = (
            _ceil_div(matrix.k, profile.p_k)
            * profile.p_k
            * _ceil_div(local_n, profile.p_n)
            * profile.p_n
        )
        matrix_bytes = _storage_bytes(padded_elements, weight_bits)
        if matrix_bytes > profile.weight_capacity_bytes:
            raise ValueError(
                "单个 rank-local 填充矩阵需要 {} 字节，但 profile 容量只有 {} 字节".format(
                    matrix_bytes, profile.weight_capacity_bytes
                )
            )
        total += matrix_bytes * resident_count
    return max(
        total,
        _logical_rank_shard_bytes(
            scenario, requirement, rank, run_context
        ),
    )


def _cim_rank_weight_shards(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    ranks: Sequence[_ExecutionRank],
    configured_cim_id: str,
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[Tuple[RuntimeTensorShard, ...], Tuple[Tuple[str, int], ...]]:
    components = scenario.hardware.component_map()
    shards: List[RuntimeTensorShard] = []
    usage: Dict[str, int] = {}
    for rank in ranks:
        target_id = rank.cim_component_id or configured_cim_id
        target = components.get(target_id)
        if target is None or not _is_cim(target):
            raise ValueError(
                "rank {} 的 CIM 目标 {} 不存在或不是 CIM".format(
                    rank.rank, target_id
                )
            )
        shard_index, _, shard_kind = _rank_shard_geometry(
            scenario, requirement, rank, run_context
        )
        logical_bytes = _logical_rank_shard_bytes(
            scenario, requirement, rank, run_context
        )
        physical_bytes = _cim_rank_padded_weight_bytes(
            scenario,
            requirement,
            rank,
            target_id,
            run_context,
        )
        shards.append(
            RuntimeTensorShard(
                rank=rank.rank,
                tp_rank=rank.tp_rank,
                pp_rank=rank.pp_rank,
                ep_rank=rank.ep_rank,
                compute_component_id=rank.component_id,
                storage_component_id=target_id,
                logical_bytes=logical_bytes,
                physical_bytes=physical_bytes,
                shard_kind=shard_kind,
            )
        )
        usage[target_id] = usage.get(target_id, 0) + physical_bytes
    return tuple(shards), tuple(sorted(usage.items()))


def _candidates_for_requirement(
    scenario: ScenarioConfig,
    options: PlacementPolicy,
    requirement: _Requirement,
    index: int,
    router: TopologyRouter,
    capacities: Mapping[str, int],
    base_usage: Mapping[str, int],
    controls: _MappingControls,
    *,
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[Tuple[_Candidate, ...], Tuple[str, ...]]:
    _check_mapping_deadline(run_context)
    components = scenario.hardware.component_map()
    gpus = tuple(
        sorted(
            (
                component
                for component in components.values()
                if _kind(component) == "gpu"
            ),
            key=lambda component: component.component_id,
        )
    )
    cpus = tuple(
        sorted(
            (
                component
                for component in components.values()
                if _kind(component) == "cpu"
            ),
            key=lambda component: component.component_id,
        )
    )
    cims = tuple(
        sorted(
            (component for component in components.values() if _is_cim(component)),
            key=lambda component: component.component_id,
        )
    )
    stores = tuple(
        sorted(
            (
                component
                for component in components.values()
                if _kind(component) in STORAGE_COMPONENT_KINDS
                or _is_cim(component)
            ),
            key=lambda component: component.component_id,
        )
    )
    rejected: List[str] = []
    candidates: List[_Candidate] = []
    load_policy_target = (
        run_context.load_policy_targets.get(requirement.item_id)
        if run_context is not None
        else None
    )
    if requirement.state_tensor:
        required_target = requirement.fixed_component
        state_stores = tuple(
            store
            for store in stores
            if required_target is None or store.component_id == required_target
        )
        if required_target and not state_stores:
            return (), (
                "固定状态目标 {} 不是内存或存储组件".format(
                    required_target
                ),
            )
        execution_gpus = _requirement_gpu_components(
            scenario, requirement, gpus, run_context
        )
        for store in state_stores:
            _check_mapping_deadline(run_context)
            if _kind(store) not in ACTIVE_MEMORY_COMPONENT_KINDS:
                rejected.append(
                    "{} 不是活动内存".format(store.component_id)
                )
                continue
            if not _writable(store):
                rejected.append("{} 不是可写的活动内存".format(store.component_id))
                continue
            usage = requirement.tensor_bytes
            if base_usage.get(store.component_id, 0) + usage > capacities[store.component_id]:
                if (
                    store.capacity_bytes <= 0
                    and _kind(store) in ACTIVE_MEMORY_COMPONENT_KINDS
                ):
                    rejected.append(
                        "V4 控制平面将活动内存 {} 的 capacity_bytes=0（容量未知）"
                        "按不可用处理；无法放置 {} 字节的状态".format(
                            store.component_id, usage
                        )
                    )
                else:
                    rejected.append(
                        "{} 缺少 {} 字节的状态容量".format(
                            store.component_id, usage
                        )
                    )
                continue
            if not _state_reachable(router, store.component_id, execution_gpus):
                rejected.append("{} 与 rank GPU 之间没有双向路由".format(store.component_id))
                continue
            candidates.append(
                _Candidate(
                    index,
                    store.component_id,
                    store.component_id,
                    _state_route_cost(
                        router, store.component_id, execution_gpus, usage
                    ),
                    ((store.component_id, usage),),
                    reason=(
                        "固定的可写活动内存状态已通过所有必需 rank GPU 读写路由校验"
                        if required_target
                        else "可写活动内存状态具备所有必需的 rank GPU 读写路由"
                    ),
                    execution_component_ids=tuple(
                        gpu.component_id for gpu in execution_gpus
                    ),
                    physical_tensor_component_ids=(store.component_id,),
                )
            )
        return tuple(sorted(candidates, key=_candidate_sort_key)), tuple(_deduplicate(rejected))

    if not gpus:
        return (), ("硬件中没有具备计算能力的 GPU",)

    execution_gpus = _requirement_gpu_components(
        scenario, requirement, gpus, run_context
    )
    execution_ranks = _requirement_execution_ranks(
        scenario, requirement, gpus, run_context
    )
    weight_ranks = (
        _requirement_weight_ranks(
            scenario, requirement, gpus, run_context
        )
        if requirement.tensor_id and requirement.tensor_bytes > 0
        else execution_ranks
    )
    gpu_selector: Optional[ComponentSpec] = execution_gpus[
        index % len(execution_gpus)
    ]
    if load_policy_target == "cpu":
        compute_candidates = cpus
        if not compute_candidates:
            rejected.append("gpu_loadable_layers 要求 CPU 放置，但硬件没有 CPU")
    elif load_policy_target == "gpu":
        compute_candidates = (
            (gpu_selector,) if gpu_selector is not None else ()
        )
    else:
        compute_candidates = (
            ((gpu_selector,) if gpu_selector is not None else ())
            + (cpus if requirement.mapping_key else ())
        )
    if (
        requirement.cim_eligible
        and requirement.operator_class == OperatorClass.GEMM
        and load_policy_target is None
    ):
        if not cims:
            rejected.append("缺少 CIM profile")
        elif scenario.cim_interconnect is None:
            rejected.append("缺少 CIM interconnect profile")
        else:
            compute_candidates += cims

    for compute in compute_candidates:
        _check_mapping_deadline(run_context)
        is_cim = _is_cim(compute)
        padded_bytes = 0
        if is_cim:
            if not requirement.cim_eligible or requirement.layer is None:
                continue
            if (
                not scenario.weights_resident
                and not options.allow_cold_cim_streaming
            ):
                rejected.append(
                    "冷权重不会隐式远端读取；必须显式启用 allow_cold_cim_streaming"
                )
                continue
            try:
                rank_shards, resident_usage = _cim_rank_weight_shards(
                    scenario,
                    requirement,
                    weight_ranks,
                    compute.component_id,
                    run_context,
                )
            except ValueError as exc:
                rejected.append("{}：{}".format(compute.component_id, exc))
                continue
            padded_bytes = max(
                (shard.physical_bytes for shard in rank_shards),
                default=0,
            )
            physical_targets = tuple(
                dict.fromkeys(
                    shard.storage_component_id for shard in rank_shards
                )
            )
            cold_cim_streaming = (
                options.allow_cold_cim_streaming
                and not scenario.weights_resident
            )
            capacity_failure = next(
                (
                    (target_id, byte_count)
                    for target_id, byte_count in resident_usage
                    if not cold_cim_streaming
                    and base_usage.get(target_id, 0) + byte_count
                    > capacities.get(target_id, 0)
                ),
                None,
            )
            if capacity_failure is not None:
                failed_component = components[capacity_failure[0]]
                if (
                    failed_component.capacity_bytes <= 0
                    and _kind(failed_component) in ACTIVE_MEMORY_COMPONENT_KINDS
                ):
                    rejected.append(
                        "V4 控制平面将活动内存 {} 的 capacity_bytes=0（容量未知）"
                        "按不可用处理；无法放置 {} 字节的填充权重".format(
                            capacity_failure[0], capacity_failure[1]
                        )
                    )
                else:
                    rejected.append(
                        "{} 上填充后的权重 {} 字节超过有效容量 {}".format(
                            capacity_failure[0],
                            capacity_failure[1],
                            capacities.get(capacity_failure[0], 0),
                        )
                    )
                continue
            unreachable_target = next(
                (
                    target_id
                    for target_id in physical_targets
                    if not _cim_activation_reachable(
                        scenario,
                        requirement,
                        router,
                        target_id,
                        gpus,
                        run_context,
                    )
                ),
                None,
            )
            if unreachable_target is not None:
                rejected.append(
                    "{} 与 rank GPU 之间没有激活值/输出路由".format(
                        unreachable_target
                    )
                )
                continue
            if cold_cim_streaming:
                backing = _select_cold_cim_backing_component(
                    scenario,
                    requirement,
                    compute.component_id,
                    router,
                    physical_targets,
                    capacities,
                    base_usage,
                    run_context,
                )
                if backing is None:
                    rejected.append(
                        "CIM 冷流式加载需要一个具有正容量、可容纳完整模型权重且"
                        "能到达所有物理 CIM 目标的 host/offload 后备组件"
                    )
                    continue
            else:
                backing = None
            usage = () if cold_cim_streaming else resident_usage
            cost = _operator_cost(
                scenario,
                options.objective,
                requirement,
                compute,
                compute,
                router,
                cold_cim_streaming=cold_cim_streaming,
                cold_cim_backing_component_id=(
                    backing.component_id if backing is not None else None
                ),
                placement_policy=options,
                run_context=run_context,
            )
            candidates.append(
                _Candidate(
                    index,
                    compute.component_id,
                    compute.component_id if requirement.tensor_id else None,
                    cost,
                    usage,
                    padded_weight_bytes=padded_bytes,
                    reason=(
                        (
                            "CIM GEMM 使用经路由的冷加载和临时阵列驻留"
                            if not usage and requirement.tensor_id
                            else "符合条件的 GEMM 权重按 TP/EP rank 分片并常驻于对应 CIM 目标"
                        )
                        + _logical_alias_reason(requirement)
                    ),
                    execution_component_ids=tuple(
                        dict.fromkeys(
                            rank.component_id for rank in execution_ranks
                        )
                    ),
                    physical_tensor_component_ids=physical_targets,
                    rank_tensor_shards=rank_shards,
                    rank_execution_targets=_rank_execution_targets(
                        execution_ranks, compute.component_id, cim=True
                    ),
                    streaming_backing_component_id=(
                        backing.component_id if backing is not None else None
                    ),
                )
            )
            continue

        if requirement.tensor_id is None or requirement.tensor_bytes <= 0:
            try:
                cost = _operator_cost(
                    scenario,
                    options.objective,
                    requirement,
                    compute,
                    None,
                    router,
                    placement_policy=options,
                    run_context=run_context,
                )
            except ValueError as exc:
                rejected.append(
                    "{} 缺少完整执行/host orchestration 路由：{}".format(
                        compute.component_id, exc
                    )
                )
                continue
            candidates.append(
                _Candidate(
                    index,
                    compute.component_id,
                    None,
                    cost,
                    (),
                    reason=(
                        "{} 原语按执行、host/local-memory、rank 搬运与共享设备排队代理成本选择 {}"
                        .format(
                            requirement.operator_class.value,
                            compute.component_id,
                        )
                    ),
                    execution_component_ids=tuple(
                        gpu.component_id for gpu in execution_gpus
                    ),
                    rank_execution_targets=_rank_execution_targets(
                        execution_ranks,
                        compute.component_id,
                        cim=False,
                        fixed_compute=_kind(compute) == "cpu",
                    ),
                )
            )
            continue
        if requirement.tensor_id is not None:
            placement_weight_ranks = weight_ranks
            if _kind(compute) == "cpu" and load_policy_target == "cpu":
                placement_weight_ranks = tuple(
                    replace(
                        rank,
                        component_id=compute.component_id,
                        memory_component_id=None,
                        cim_component_id=None,
                    )
                    for rank in weight_ranks
                )
            local_stores = tuple(
                store for store in stores if not _is_cim(store)
            )
            local_options = _rank_local_weight_candidates(
                scenario,
                requirement,
                index,
                router,
                placement_weight_ranks,
                local_stores,
            )
            for shards, shard_usage, route_cost in local_options:
                _check_mapping_deadline(run_context)
                if _kind(compute) == "cpu" and load_policy_target == "cpu":
                    rank_component_ids = {
                        rank.rank: rank.component_id for rank in weight_ranks
                    }
                    shards = tuple(
                        replace(
                            shard,
                            compute_component_id=rank_component_ids[shard.rank],
                        )
                        for shard in shards
                    )
                capacity_failure = next(
                    (
                        (component_id, byte_count)
                        for component_id, byte_count in shard_usage
                        if base_usage.get(component_id, 0) + byte_count
                        > capacities.get(component_id, 0)
                    ),
                    None,
                )
                if capacity_failure is not None:
                    failed_component = components[capacity_failure[0]]
                    if (
                        _kind(failed_component)
                        in OFFLOAD_STORAGE_COMPONENT_KINDS
                        and failed_component.capacity_bytes <= 0
                    ):
                        rejected.append(
                            "卸载存储 {} 的 capacity_bytes=0（容量未知）；请声明可容纳 {} 字节 "
                            "rank-local 权重分片的正容量".format(
                                capacity_failure[0], capacity_failure[1]
                            )
                        )
                    elif (
                        _kind(failed_component)
                        in ACTIVE_MEMORY_COMPONENT_KINDS
                        and failed_component.capacity_bytes <= 0
                    ):
                        rejected.append(
                            "V4 控制平面将活动内存 {} 的 capacity_bytes=0（容量未知）"
                            "按不可用处理；无法放置 {} 字节的 rank-local 权重分片".format(
                                capacity_failure[0], capacity_failure[1]
                            )
                        )
                    else:
                        rejected.append(
                            "{} 缺少 {} 字节的 rank-local 权重分片容量".format(
                                capacity_failure[0], capacity_failure[1]
                            )
                        )
                    continue
                representative = next(
                    (
                        shard.storage_component_id
                        for shard in shards
                        if shard.compute_component_id == compute.component_id
                    ),
                    shards[0].storage_component_id,
                )
                physical_targets = tuple(
                    dict.fromkeys(
                        shard.storage_component_id for shard in shards
                    )
                )
                effective_route_cost = route_cost
                if _kind(compute) == "cpu":
                    effective_route_cost = sum(
                        _route_cost(
                            router,
                            shard.storage_component_id,
                            compute.component_id,
                            max(1, shard.physical_bytes),
                        )
                        for shard in shards
                        if shard.storage_component_id
                        != compute.component_id
                    )
                candidates.append(
                    _Candidate(
                        index,
                        compute.component_id,
                        representative,
                        _operator_cost(
                            scenario,
                            options.objective,
                            requirement,
                            compute,
                            None,
                            router,
                            placement_policy=options,
                            run_context=run_context,
                        )
                        + effective_route_cost,
                        shard_usage,
                        reason=(
                            "分类 CPU/GPU 候选保持固定 stage/rank 执行集并使用当前拓扑中的 rank-local 权重目标；"
                            "冷场景由运行时控制面从 NVMe/page cache 装载，TP 按张量分片，"
                            "EP 对 Dense 权重复制并对专家权重分片"
                            + _logical_alias_reason(requirement)
                        ),
                        execution_component_ids=tuple(
                            dict.fromkeys(
                                rank.component_id for rank in execution_ranks
                            )
                        ),
                        physical_tensor_component_ids=physical_targets,
                        rank_tensor_shards=shards,
                        rank_execution_targets=_rank_execution_targets(
                            execution_ranks,
                            compute.component_id,
                            cim=False,
                            fixed_compute=_kind(compute) == "cpu",
                        ),
                    )
                )
            if not local_options:
                rejected.append(
                    "所有执行 rank 都必须具有可达的活动本地权重存储；聚合 model_weights 不作为常驻远端读取后备"
                )
            # Every local option is either a complete rank stripe or a
            # recorded capacity rejection.  Do not silently fall back to one
            # remote aggregate tensor, which recreates the old TP bug.
            continue
    return tuple(sorted(candidates, key=_candidate_sort_key)), tuple(_deduplicate(rejected))


def _operator_cost(
    scenario: ScenarioConfig,
    objective: str,
    requirement: _Requirement,
    compute: ComponentSpec,
    store: Optional[ComponentSpec],
    router: TopologyRouter,
    *,
    cold_cim_streaming: bool = False,
    cold_cim_backing_component_id: Optional[str] = None,
    placement_policy: Optional[PlacementPolicy] = None,
    run_context: Optional[_MappingRunContext] = None,
) -> float:
    _check_mapping_deadline(run_context)
    layer = requirement.layer
    if requirement.kind == "weight_artifact":
        return 0.0
    if layer is None:
        return 0.0
    if requirement.operator_class == OperatorClass.GEMM and not requirement.matrices:
        return 0.0
    prefill_m, decode_m, throughput_m = _objective_batch_sizes(placement_policy)
    all_gpus = tuple(
        component
        for component in scenario.hardware.components
        if _kind(component) == "gpu"
    )
    execution_gpus = _requirement_gpu_components(
        scenario, requirement, all_gpus, run_context
    )
    execution_ranks = _requirement_execution_ranks(
        scenario, requirement, all_gpus, run_context
    )

    def nearest_profile_component_id(
        source_component_id: str,
        component_kind: str,
    ) -> str:
        ranked: List[Tuple[float, str]] = []
        for component in scenario.hardware.components:
            if _kind(component) != component_kind:
                continue
            try:
                scenario.resolve_component_profile(component)
                cost = _route_cost(
                    router,
                    source_component_id,
                    component.component_id,
                    1,
                ) + _route_cost(
                    router,
                    component.component_id,
                    source_component_id,
                    1,
                )
            except (KeyError, TypeError, ValueError):
                continue
            ranked.append((cost, component.component_id))
        if not ranked:
            raise ValueError(
                "组件 {} 没有可达的 {} profile 目标".format(
                    source_component_id, component_kind
                )
            )
        return min(ranked)[1]

    def cpu_profiles(
        cpu_component_id: str,
    ) -> Tuple[CPUProfile, HostMemoryProfile]:
        cpu_profile = scenario.resolve_component_profile(
            cpu_component_id, CPUProfile
        )
        memory_component_id = nearest_profile_component_id(
            cpu_component_id, "host_memory"
        )
        memory_profile = scenario.resolve_component_profile(
            memory_component_id, HostMemoryProfile
        )
        return cpu_profile, memory_profile

    def gpu_profiles(
        gpu_component_id: str,
        memory_component_id: Optional[str] = None,
    ) -> Tuple[GPUProfile, HBMProfile]:
        gpu_profile = scenario.resolve_component_profile(
            gpu_component_id, GPUProfile
        )
        selected_memory = memory_component_id
        if selected_memory is not None:
            memory_component = scenario.hardware.get_component(
                selected_memory
            )
            if _kind(memory_component) != "hbm":
                selected_memory = None
        if selected_memory is None:
            selected_memory = nearest_profile_component_id(
                gpu_component_id, "hbm"
            )
        memory_profile = scenario.resolve_component_profile(
            selected_memory, HBMProfile
        )
        return gpu_profile, memory_profile

    def typed_workload(m: int) -> object:
        activation_bits = _activation_bits(layer)
        context_tokens = max(
            1,
            max(1, m),
            prefill_m,
        )
        input_elements = max(
            1,
            max(1, m)
            * requirement.elements_per_token
            * (context_tokens if requirement.context_scaled_elements else 1),
        )
        if requirement.operator_class == OperatorClass.ELEMENTWISE:
            operation_elements = max(
                1,
                max(1, m)
                * (
                    requirement.operation_elements_per_token
                    or requirement.elements_per_token
                ),
            )
            explicit_traffic = (
                operation_elements != input_elements
            )
            return ElementwiseWorkload(
                elements=operation_elements,
                operations_per_element=requirement.operations_per_element,
                fixed_operations=(
                    max(1, m) * requirement.fixed_operations_per_token
                ),
                input_count=requirement.input_count,
                input_bits=activation_bits,
                output_bits=activation_bits,
                transcendental_ops_per_element=(
                    requirement.transcendental_operations_per_element
                ),
                fixed_transcendental_operations=(
                    max(1, m)
                    * requirement.fixed_transcendental_operations_per_token
                ),
                read_storage_bytes=(
                    max(1, m) * requirement.read_bytes_per_token
                    if requirement.read_bytes_per_token
                    else (
                        _storage_bytes(
                            input_elements * requirement.input_count,
                            activation_bits,
                        )
                        if explicit_traffic
                        else None
                    )
                ),
                write_storage_bytes=(
                    max(1, m) * requirement.write_bytes_per_token
                    if requirement.write_bytes_per_token
                    else (
                        _storage_bytes(input_elements, activation_bits)
                        if explicit_traffic
                        else None
                    )
                ),
                name=requirement.item_id,
            )
        if requirement.operator_class == OperatorClass.REDUCTION:
            output_elements = min(
                input_elements,
                max(
                    1,
                    max(1, m)
                    * requirement.output_elements_per_token,
                ),
            )
            return ReductionWorkload(
                input_elements=input_elements,
                output_elements=output_elements,
                operations_per_combine=(
                    requirement.operations_per_element
                ),
                fixed_operations=(
                    max(1, m) * requirement.fixed_operations_per_token
                ),
                input_bits=activation_bits,
                output_bits=max(16, activation_bits),
                name=requirement.item_id,
            )
        if requirement.operator_class == OperatorClass.MEMORY:
            return MemoryWorkload(
                read_bytes=max(
                    0, max(1, m) * requirement.read_bytes_per_token
                ),
                write_bytes=max(
                    0, max(1, m) * requirement.write_bytes_per_token
                ),
                name=requirement.item_id,
            )
        raise ValueError(
            "不支持的 typed operator class {}".format(
                requirement.operator_class.value
            )
        )

    def non_gemm_service(m: int) -> float:
        workload = typed_workload(m)
        if _is_cim(compute):
            raise ValueError("CIM 只支持 GEMM 原语")
        if _kind(compute) == "cpu":
            cpu_profile, host_memory_profile = cpu_profiles(
                compute.component_id
            )
            estimators = {
                OperatorClass.ELEMENTWISE: estimate_cpu_elementwise,
                OperatorClass.REDUCTION: estimate_cpu_reduction,
                OperatorClass.MEMORY: estimate_cpu_memory,
            }
            estimate = estimators[requirement.operator_class](
                cpu_profile, host_memory_profile, workload
            )
            total = estimate.service_ns * max(1, len(execution_gpus))
        else:
            estimators = {
                OperatorClass.ELEMENTWISE: estimate_gpu_elementwise,
                OperatorClass.REDUCTION: estimate_gpu_reduction,
                OperatorClass.MEMORY: estimate_gpu_memory,
            }
            rank_targets = execution_ranks or (
                _ExecutionRank(
                    rank=0,
                    tp_rank=0,
                    pp_rank=0,
                    ep_rank=0,
                    component_id=compute.component_id,
                ),
            )
            total = 0.0
            for rank_target in rank_targets:
                gpu_profile, hbm_profile = gpu_profiles(
                    rank_target.component_id,
                    rank_target.memory_component_id,
                )
                total += estimators[requirement.operator_class](
                    gpu_profile, hbm_profile, workload
                ).service_ns
        # One CPU candidate is shared by all logical ranks, so summing rank
        # service is the deterministic queueing surrogate. GPU candidates use
        # each rank target's explicitly bound GPU/HBM profiles.
        if _kind(compute) == "cpu":
            read_bytes = int(getattr(workload, "read_bytes", 0))
            write_bytes = int(getattr(workload, "write_bytes", 0))
            for execution_gpu in execution_gpus:
                if execution_gpu.component_id == compute.component_id:
                    continue
                total += _route_cost(
                    router,
                    execution_gpu.component_id,
                    compute.component_id,
                    max(1, read_bytes),
                )
                total += _route_cost(
                    router,
                    compute.component_id,
                    execution_gpu.component_id,
                    max(1, write_bytes),
                )
        return total

    def service(m: int) -> float:
        _check_mapping_deadline(run_context)
        if requirement.operator_class != OperatorClass.GEMM:
            return non_gemm_service(m)
        total = 0.0
        for matrix in requirement.matrices:
            _check_mapping_deadline(run_context)
            count = max(1, matrix.compute_count)
            workload_k = matrix.k
            workload_n = matrix.n
            activation_storage_bytes = None
            output_storage_bytes = None
            descriptor_backing = (
                matrix.weight_storage_bytes is not None
                and not requirement.dynamic_rhs
                and not _is_cim(compute)
            )
            weight_storage_bytes = (
                matrix.weight_storage_bytes
                if descriptor_backing
                else None
            )
            weight_metadata_bytes = (
                matrix.weight_metadata_bytes
                if descriptor_backing
                else 0
            )
            if requirement.attention_dynamic_kind is not None:
                context_tokens = max(1, prefill_m, max(1, m))
                score_elements = (
                    max(1, m)
                    * requirement.attention_score_heads
                    * context_tokens
                )
                kv_operand_bytes = _storage_bytes(
                    context_tokens * requirement.attention_kv_width,
                    _activation_bits(layer),
                )
                weight_storage_bytes = kv_operand_bytes
                if requirement.attention_dynamic_kind == "qk":
                    workload_k = requirement.attention_query_width
                    workload_n = context_tokens
                    output_storage_bytes = _storage_bytes(
                        score_elements, _activation_bits(layer)
                    )
                else:
                    workload_k = context_tokens
                    workload_n = requirement.attention_query_width
                    activation_storage_bytes = _storage_bytes(
                        score_elements, _activation_bits(layer)
                    )
            workload = GemmWorkload(
                m=max(1, m),
                k=workload_k,
                n=workload_n,
                activation_bits=_activation_bits(layer),
                weight_bits=(
                    _activation_bits(layer)
                    if requirement.dynamic_rhs
                    else (
                        matrix.weight_bits
                        if descriptor_backing and matrix.weight_bits is not None
                        else _weight_bits(layer)
                    )
                ),
                output_bits=max(16, _activation_bits(layer)),
                accumulator_bits=32,
                packed_weight_formats=(
                    matrix.packed_weight_formats
                    if descriptor_backing
                    else ()
                ),
                packed_weight_transform_operations=(
                    matrix.packed_weight_transform_operations
                    if descriptor_backing
                    else 0
                ),
                weight_storage_bytes=weight_storage_bytes,
                weight_metadata_bytes=weight_metadata_bytes,
                activation_storage_bytes=activation_storage_bytes,
                output_storage_bytes=output_storage_bytes,
                name=matrix.projection_id or requirement.item_id,
            )
            if _is_cim(compute):
                cim_profile = scenario.resolve_component_profile(
                    compute, DigitalSramCimProfile
                )
                total += estimate_cim_gemm(
                    cim_profile,
                    workload,
                    weights_resident=not cold_cim_streaming,
                ).service_ns * count
            elif _kind(compute) == "cpu":
                cpu_profile, host_memory_profile = cpu_profiles(
                    compute.component_id
                )
                total += (
                    estimate_cpu_gemm(
                        cpu_profile, host_memory_profile, workload
                    ).service_ns
                    * count
                    * max(1, len(execution_gpus))
                )
            else:
                rank_targets = execution_ranks or (
                    _ExecutionRank(
                        rank=0,
                        tp_rank=0,
                        pp_rank=0,
                        ep_rank=0,
                        component_id=compute.component_id,
                    ),
                )
                for rank_target in rank_targets:
                    gpu_profile, hbm_profile = gpu_profiles(
                        rank_target.component_id,
                        rank_target.memory_component_id,
                    )
                    total += (
                        estimate_gpu_gemm(
                            gpu_profile,
                            hbm_profile,
                            workload,
                        ).service_ns
                        * count
                    )
        if store is not None and requirement.tensor_bytes > 0 and not _is_cim(compute):
            destinations = (
                (compute.component_id,)
                if _kind(compute) == "cpu"
                else tuple(gpu.component_id for gpu in execution_gpus)
            )
            for destination in destinations:
                if store.component_id != destination:
                    total += _route_cost(
                        router,
                        store.component_id,
                        destination,
                        requirement.tensor_bytes,
                    )
        if _kind(compute) == "cpu":
            attention_context_tokens = max(1, prefill_m, max(1, m))
            if requirement.attention_dynamic_kind == "qk":
                activation_bytes = _storage_bytes(
                    max(1, m) * requirement.attention_query_width,
                    _activation_bits(layer),
                )
                output_bytes = _storage_bytes(
                    max(1, m)
                    * requirement.attention_score_heads
                    * attention_context_tokens,
                    _activation_bits(layer),
                )
            elif requirement.attention_dynamic_kind == "pv":
                activation_bytes = _storage_bytes(
                    max(1, m)
                    * requirement.attention_score_heads
                    * attention_context_tokens,
                    _activation_bits(layer),
                )
                output_bytes = _storage_bytes(
                    max(1, m) * requirement.attention_query_width,
                    _activation_bits(layer),
                )
            else:
                activation_bytes = _storage_bytes(
                    max(1, m) * layer.hidden_size,
                    _activation_bits(layer),
                )
                output_bytes = activation_bytes
            dynamic_rhs_bytes = (
                (
                    _storage_bytes(
                        attention_context_tokens
                        * requirement.attention_kv_width,
                        _activation_bits(layer),
                    )
                    if requirement.attention_dynamic_kind is not None
                    else sum(
                        _storage_bytes(
                            matrix.k * matrix.n,
                            _activation_bits(layer),
                        )
                        * max(1, matrix.compute_count)
                        for matrix in requirement.matrices
                    )
                )
                if requirement.dynamic_rhs
                else 0
            )
            for execution_gpu in execution_gpus:
                total += _route_cost(
                    router,
                    execution_gpu.component_id,
                    compute.component_id,
                    activation_bytes,
                )
                if dynamic_rhs_bytes:
                    total += _route_cost(
                        router,
                        execution_gpu.component_id,
                        compute.component_id,
                        dynamic_rhs_bytes,
                    )
                total += _route_cost(
                    router,
                    compute.component_id,
                    execution_gpu.component_id,
                    output_bytes,
                )
        if _is_cim(compute):
            physical_targets = _cim_target_ids(
                scenario, requirement, compute.component_id, run_context
            )
            if cold_cim_streaming and requirement.tensor_bytes > 0:
                if not cold_cim_backing_component_id:
                    raise ValueError(
                        "cold CIM streaming cost requires a topology-derived backing component"
                    )
                target_bytes = _cold_cim_streaming_bytes_by_target(
                    scenario,
                    requirement,
                    compute.component_id,
                    physical_targets,
                    run_context,
                )
                for target_id, byte_count in target_bytes.items():
                    if cold_cim_backing_component_id != target_id:
                        total += _route_cost(
                            router,
                            cold_cim_backing_component_id,
                            target_id,
                            byte_count,
                        )
            for gpu_id, cim_id in _cim_execution_pairs(
                scenario,
                requirement,
                physical_targets,
                tuple(gpu.component_id for gpu in all_gpus),
                run_context,
            ):
                activation_bytes = _storage_bytes(
                    max(1, m) * layer.hidden_size,
                    _activation_bits(layer),
                )
                total += _route_cost(
                    router, gpu_id, cim_id, activation_bytes
                )
                total += _route_cost(
                    router, cim_id, gpu_id, activation_bytes
                )
        return total

    if objective == "ttft":
        return service(prefill_m)
    if objective == "tpot":
        return service(decode_m)
    if objective == "throughput":
        return service(throughput_m)
    return 0.5 * service(prefill_m) + 0.5 * service(decode_m)


def _objective_batch_sizes(
    options: Optional[PlacementPolicy] = None,
) -> Tuple[int, int, int]:
    """Return an explicit workload-independent analytical design point."""

    design = options or PlacementPolicy()
    return (
        design.design_prefill_tokens,
        design.design_decode_batch_size,
        design.design_throughput_tokens,
    )


def _fusion_design_tokens(options: PlacementPolicy) -> int:
    prefill_m, decode_m, throughput_m = _objective_batch_sizes(options)
    if options.objective == "ttft":
        return prefill_m
    if options.objective == "tpot":
        return decode_m
    if options.objective == "throughput":
        return throughput_m
    return max(prefill_m, decode_m)


def _derive_fusion_opportunities(
    scenario: ScenarioConfig,
    options: PlacementPolicy,
    requirements: Sequence[_Requirement],
    *,
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[_FusionOpportunity, ...]:
    """Return disjoint, shape-derived fusion hyperedges.

    The normal solver remains additive by collapsing each hyperedge into one
    solve unit later.  This function intentionally depends only on mapping
    design points, never on the runtime request stream.
    """

    index_by_id = {
        requirement.item_id: index
        for index, requirement in enumerate(requirements)
    }
    layer_list: List[LayerSpec] = []
    seen_layer_ids: set[str] = set()
    for requirement in requirements:
        if (
            requirement.layer is not None
            and requirement.layer.layer_id not in seen_layer_ids
        ):
            seen_layer_ids.add(requirement.layer.layer_id)
            layer_list.append(requirement.layer)
    layers = tuple(layer_list)
    plan = _run_parallel_plan(scenario, run_context)
    tokens = _fusion_design_tokens(options)
    opportunities: List[_FusionOpportunity] = []
    occupied: set[int] = set()

    def add(
        layer: LayerSpec,
        group: str,
        variant: str,
        item_ids: Sequence[str],
        working_set_bytes: int,
        eliminated_launches: int,
    ) -> None:
        if any(item_id not in index_by_id for item_id in item_ids):
            return
        member_indices = tuple(index_by_id[item_id] for item_id in item_ids)
        overlap = occupied.intersection(member_indices)
        if overlap:
            raise ValueError(
                "fusion opportunity requirements overlap: {}".format(
                    ", ".join(str(index) for index in sorted(overlap))
                )
            )
        occupied.update(member_indices)
        member_op_keys = tuple(
            requirements[index].mapping_key or requirements[index].item_id
            for index in member_indices
        )
        opportunities.append(
            _FusionOpportunity(
                opportunity_id="{}:{}:{}".format(
                    layer.layer_id, group, variant
                ),
                group=group,
                variant=variant,
                layer_id=layer.layer_id,
                member_indices=member_indices,
                member_op_keys=member_op_keys,
                working_set_bytes=max(1, int(working_set_bytes)),
                eliminated_launches=max(1, int(eliminated_launches)),
            )
        )

    for layer in layers:
        bits = _activation_bits(layer)
        tp_degree = max(1, plan.tp_degree)
        hidden_local = _ceil_div(layer.hidden_size, tp_degree)
        hidden_elements = max(1, tokens * layer.hidden_size)
        if layer.is_linear_attention:
            residual_key = "{}.linear_attention.residual".format(
                layer.layer_id
            )
        else:
            head_dim = _attention_head_dim(layer)
            q_width = layer.attention_heads * head_dim
            kv_width = layer.effective_kv_heads * head_dim
            attention_descriptor = resolve_attention_execution_descriptor(
                layer.metadata,
                attention_heads=layer.attention_heads,
                kv_heads=layer.effective_kv_heads,
                head_dim=head_dim,
                hidden_size=layer.hidden_size,
            )
            q_projection_width = q_width
            if attention_descriptor is not None:
                q_width = attention_descriptor.query_width
                kv_width = (
                    attention_descriptor.kv_heads
                    * attention_descriptor.head_dim
                )
                q_projection_width = attention_descriptor.q_projection_width
            qkv_local = _ceil_div(
                q_projection_width + 2 * kv_width, tp_degree
            )
            rope_local = _ceil_div(q_width + kv_width, tp_degree)
            qkv_elements = max(1, tokens * qkv_local)
            rope_elements = max(1, tokens * rope_local)
            if attention_descriptor is None or not attention_descriptor.qk_norm:
                add(
                    layer,
                    "qkv_rope",
                    "attention",
                    (
                        "{}.attention".format(layer.layer_id),
                        "{}.attention.rope".format(layer.layer_id),
                    ),
                    _storage_bytes(qkv_elements, bits)
                    + _storage_bytes(2 * rope_elements, bits),
                    1,
                )
            flash_workload = FusedAttentionWorkload(
                batch_tokens=max(1, tokens),
                context_tokens=max(
                    1, options.design_prefill_tokens, tokens
                ),
                hidden_size=(
                    max(1, _ceil_div(q_width, tp_degree))
                    if attention_descriptor is not None
                    else max(1, hidden_local)
                ),
                input_bits=bits,
                output_bits=bits,
                kv_hidden_size=max(1, _ceil_div(kv_width, tp_degree)),
                kv_input_bits=bits,
                score_heads=(
                    max(
                        1,
                        _ceil_div(
                            attention_descriptor.query_heads, tp_degree
                        ),
                    )
                    if attention_descriptor is not None
                    else 1
                ),
                qk_scale=(
                    attention_descriptor.qk_scale
                    if attention_descriptor is not None
                    else None
                ),
                name="control_plane_flash_attention",
            )
            flash_member_ids = [
                "{}.attention.qk".format(layer.layer_id),
            ]
            if attention_descriptor is not None:
                flash_member_ids.append(
                    "{}.attention.qk_scale".format(layer.layer_id)
                )
            flash_member_ids.extend(
                [
                    "{}.attention.softmax.reduce".format(layer.layer_id),
                    "{}.attention.softmax.normalize".format(layer.layer_id),
                    "{}.attention.pv".format(layer.layer_id),
                ]
            )
            add(
                layer,
                "flash_attention",
                "attention",
                tuple(flash_member_ids),
                flash_workload.onchip_working_set_bytes,
                len(flash_member_ids) - 1,
            )
            residual_key = "{}.attention.residual".format(layer.layer_id)

        if layer.is_moe:
            activation_elements = max(
                1, tokens * _ceil_div(2 * layer.intermediate_size, tp_degree)
            )
            add(
                layer,
                "gemm_epilogue_activation",
                "experts",
                (
                    "{}.experts".format(layer.layer_id),
                    "{}.experts.activation".format(layer.layer_id),
                ),
                _storage_bytes(activation_elements, bits),
                1,
            )
            if layer.has_shared_expert:
                shared_elements = max(
                    1,
                    tokens
                    * _ceil_div(
                        2 * layer.shared_expert_intermediate_size,
                        tp_degree,
                    ),
                )
                add(
                    layer,
                    "gemm_epilogue_activation",
                    "shared_expert",
                    (
                        "{}.shared_expert".format(layer.layer_id),
                        "{}.shared_expert.activation".format(layer.layer_id),
                    ),
                    _storage_bytes(shared_elements, bits),
                    1,
                )
        else:
            activation_elements = max(
                1, tokens * _ceil_div(2 * layer.intermediate_size, tp_degree)
            )
            add(
                layer,
                "gemm_epilogue_activation",
                "dense_mlp",
                (
                    "{}.mlp".format(layer.layer_id),
                    "{}.mlp.activation".format(layer.layer_id),
                ),
                _storage_bytes(activation_elements, bits),
                1,
            )

        add(
            layer,
            "residual_norm",
            "linear_attention" if layer.is_linear_attention else "attention",
            (
                residual_key,
                "{}.post_attention_norm.reduce".format(layer.layer_id),
                "{}.post_attention_norm.apply".format(layer.layer_id),
            ),
            _storage_bytes(3 * hidden_elements, bits),
            2,
        )
    return tuple(opportunities)


def _fusion_effective_sram_limit(
    scenario: ScenarioConfig,
    target_component_id: str,
) -> int:
    component = scenario.hardware.get_component(target_component_id)
    if _kind(component) != "gpu":
        return 0
    profile = scenario.resolve_component_profile(
        target_component_id, GPUProfile
    )
    hardware_limit = max(
        (
            int(level.capacity_bytes)
            for level in profile.cache_hierarchy.levels
        ),
        default=0,
    )
    configured_limit = int(
        scenario.fusion_policy.max_fused_working_set_bytes
    )
    return (
        min(hardware_limit, configured_limit)
        if configured_limit > 0
        else hardware_limit
    )


def _fusion_combo_analysis(
    scenario: ScenarioConfig,
    opportunity: _FusionOpportunity,
    member_candidates: Sequence[Optional[_Candidate]],
) -> Tuple[Dict[str, Any], ...]:
    selected = tuple(member_candidates)
    target_maps: List[Dict[int, RuntimeExecutionTarget]] = []
    for candidate in selected:
        if candidate is None:
            target_maps.append({})
            continue
        targets = {
            target.rank_id: target
            for target in candidate.rank_execution_targets
        }
        if not targets:
            targets = {
                0: RuntimeExecutionTarget(
                    rank_id=0,
                    tp_rank=0,
                    pp_rank=0,
                    ep_rank=0,
                    compute_component_id=candidate.component_id,
                    component_id=candidate.component_id,
                )
            }
        target_maps.append(targets)
    rank_ids = sorted(
        {
            rank_id
            for targets in target_maps
            for rank_id in targets
        }
        or {0}
    )
    policy_enabled = bool(
        getattr(scenario.fusion_policy, opportunity.group)
    )
    records: List[Dict[str, Any]] = []
    for rank_id in rank_ids:
        rank_targets = [targets.get(rank_id) for targets in target_maps]
        complete = all(candidate is not None for candidate in selected) and all(
            target is not None for target in rank_targets
        )
        selected_targets = {
            key: (
                target.component_id if target is not None else None
            )
            for key, target in zip(
                opportunity.member_op_keys, rank_targets
            )
        }
        concrete_targets = [
            target for target in rank_targets if target is not None
        ]
        target_ids = {
            target.component_id for target in concrete_targets
        }
        target_component_id = (
            concrete_targets[0].component_id
            if concrete_targets
            else None
        )
        anchor_target = rank_targets[0] if rank_targets else None
        anchor_is_gpu = bool(
            anchor_target is not None
            and _kind(
                scenario.hardware.get_component(
                    anchor_target.component_id
                )
            )
            == "gpu"
        )
        same_rank_gpu = bool(complete and len(target_ids) == 1)
        if same_rank_gpu:
            assert target_component_id is not None
            component = scenario.hardware.get_component(
                target_component_id
            )
            same_rank_gpu = _kind(component) == "gpu" and all(
                target.component_id == target.compute_component_id
                for target in concrete_targets
            )
        sram_limit = (
            _fusion_effective_sram_limit(
                scenario, target_component_id
            )
            if target_component_id is not None
            and _kind(
                scenario.hardware.get_component(target_component_id)
            )
            == "gpu"
            else 0
        )
        fits = opportunity.working_set_bytes <= sram_limit
        active = policy_enabled and same_rank_gpu and fits
        if not policy_enabled:
            decision = "disabled_by_policy"
        elif not complete:
            decision = "fusion_member_unplaced"
        elif not same_rank_gpu:
            decision = (
                "fusion_members_not_on_same_rank_gpu"
                if anchor_is_gpu
                else "target_is_not_gpu"
            )
        elif not fits:
            decision = "working_set_exceeds_sram_limit"
        else:
            decision = "enabled_same_gpu_sram_resident"
        launch_credit = 0.0
        if active and target_component_id is not None:
            profile = scenario.resolve_component_profile(
                target_component_id, GPUProfile
            )
            launch_credit = (
                opportunity.eliminated_launches
                * float(profile.kernel_launch_ns)
            )
        records.append(
            {
                "opportunity_id": opportunity.opportunity_id,
                "fusion_group": opportunity.group,
                "variant": opportunity.variant,
                "layer_id": opportunity.layer_id,
                "rank_id": rank_id,
                "fusion_enabled": active,
                "fusion_decision": decision,
                "fusion_target_component": target_component_id,
                "fusion_working_set_bytes": (
                    opportunity.working_set_bytes
                ),
                "fusion_sram_limit_bytes": sram_limit,
                "co_located_op_keys": list(
                    opportunity.member_op_keys
                ),
                "selected_member_targets": selected_targets,
                "colocation_required": active,
                "eliminated_kernel_launches": (
                    opportunity.eliminated_launches if active else 0
                ),
                "fusion_credit_ns": launch_credit,
                "cost_model": "conservative_avoided_kernel_launch",
            }
        )
    return tuple(records)


def _bundle_candidate(
    scenario: ScenarioConfig,
    opportunity: _FusionOpportunity,
    member_candidates: Sequence[Optional[_Candidate]],
) -> _Candidate:
    selected = tuple(member_candidates)
    usage_by_component: Dict[str, int] = {}
    cost = 0.0
    for candidate in selected:
        if candidate is None:
            continue
        cost += candidate.cost
        for component_id, amount in candidate.usage:
            usage_by_component[component_id] = (
                usage_by_component.get(component_id, 0) + amount
            )
    analysis = _fusion_combo_analysis(
        scenario, opportunity, selected
    )
    fusion_credit = sum(
        float(record["fusion_credit_ns"]) for record in analysis
    )
    first = next(candidate for candidate in selected if candidate is not None)
    signature_parts = []
    for index, candidate in zip(opportunity.member_indices, selected):
        if candidate is None:
            signature_parts.append("{}:~".format(index))
        else:
            signature_parts.append(
                "{}:{}/{}".format(
                    index,
                    candidate.component_id,
                    candidate.tensor_component_id or "",
                )
            )
    return _Candidate(
        requirement_index=min(opportunity.member_indices),
        component_id=first.component_id,
        tensor_component_id=first.tensor_component_id,
        cost=max(0.0, cost - fusion_credit),
        usage=tuple(sorted(usage_by_component.items())),
        reason=(
            "fusion bundle {} 联合比较融合与拆分放置；仅扣除满足 policy、"
            "同 rank GPU 和 SRAM 门限时可避免的 kernel launch"
        ).format(opportunity.opportunity_id),
        rank_execution_targets=first.rank_execution_targets,
        member_candidates=selected,
        missing_count=sum(
            candidate is None for candidate in selected
        ),
        choice_signature="|".join(signature_parts),
        fusion_credit=fusion_credit,
    )


def _solve_candidate_sort_key(
    candidate: _Candidate,
) -> Tuple[int, int, float, str, str]:
    return (
        candidate.missing_count,
        _candidate_cost_units(candidate),
        candidate.cost,
        candidate.choice_signature or candidate.component_id,
        candidate.tensor_component_id or "",
    )


def _opportunity_can_fit_any_gpu(
    scenario: ScenarioConfig,
    opportunity: _FusionOpportunity,
) -> bool:
    for component in scenario.hardware.components:
        if _kind(component) != "gpu":
            continue
        try:
            limit = _fusion_effective_sram_limit(
                scenario, component.component_id
            )
        except (KeyError, TypeError, ValueError):
            continue
        if opportunity.working_set_bytes <= limit:
            return True
    return False


def _candidate_can_join_gpu_fusion(
    scenario: ScenarioConfig,
    candidate: _Candidate,
) -> bool:
    targets = candidate.rank_execution_targets
    if not targets:
        try:
            return _kind(
                scenario.hardware.get_component(candidate.component_id)
            ) == "gpu"
        except KeyError:
            return False
    return all(
        target.component_id == target.compute_component_id
        and _kind(
            scenario.hardware.get_component(target.component_id)
        )
        == "gpu"
        for target in targets
    )


def _fusion_member_candidates(
    scenario: ScenarioConfig,
    candidates: Sequence[_Candidate],
) -> Tuple[_Candidate, ...]:
    """Drop dominated non-fusible choices for capacity-free primitives."""

    if len(candidates) <= 1 or any(
        candidate.usage
        or candidate.tensor_component_id is not None
        or candidate.rank_tensor_shards
        for candidate in candidates
    ):
        return tuple(candidates)
    cheapest = min(candidates, key=_candidate_sort_key)
    return tuple(
        candidate
        for candidate in candidates
        if candidate is cheapest
        or _candidate_can_join_gpu_fusion(scenario, candidate)
    )


def _build_fusion_solve_units(
    scenario: ScenarioConfig,
    requirements: Sequence[_Requirement],
    candidate_lists: Sequence[Tuple[_Candidate, ...]],
    opportunities: Sequence[_FusionOpportunity],
) -> Tuple[_SolveUnit, ...]:
    del requirements
    enabled = tuple(
        opportunity
        for opportunity in opportunities
        if bool(getattr(scenario.fusion_policy, opportunity.group))
        and _opportunity_can_fit_any_gpu(scenario, opportunity)
    )
    by_first = {
        min(opportunity.member_indices): opportunity
        for opportunity in enabled
    }
    bundled_indices = {
        index
        for opportunity in enabled
        for index in opportunity.member_indices
    }
    units: List[_SolveUnit] = []
    for index, candidates in enumerate(candidate_lists):
        opportunity = by_first.get(index)
        if opportunity is not None:
            member_options = tuple(
                (None,)
                + _fusion_member_candidates(
                    scenario, candidate_lists[member_index]
                )
                for member_index in opportunity.member_indices
            )
            bundled: List[_Candidate] = []
            for choice in product(*member_options):
                if all(candidate is None for candidate in choice):
                    continue
                bundled.append(
                    _bundle_candidate(scenario, opportunity, choice)
                )
            units.append(
                _SolveUnit(
                    requirement_indices=opportunity.member_indices,
                    candidates=tuple(
                        sorted(bundled, key=_solve_candidate_sort_key)
                    ),
                )
            )
            continue
        if index in bundled_indices:
            continue
        units.append(
            _SolveUnit(
                requirement_indices=(index,),
                candidates=candidates,
            )
        )
    return tuple(units)


def _expand_solve_assignment(
    assignment: Sequence[Optional[_Candidate]],
    solve_units: Sequence[_SolveUnit],
    requirement_count: int,
) -> Tuple[Optional[_Candidate], ...]:
    expanded: List[Optional[_Candidate]] = [None] * requirement_count
    for candidate, unit in zip(assignment, solve_units):
        if candidate is None:
            continue
        if candidate.member_candidates:
            for requirement_index, member in zip(
                unit.requirement_indices, candidate.member_candidates
            ):
                expanded[requirement_index] = member
        else:
            expanded[unit.requirement_indices[0]] = candidate
    return tuple(expanded)


def _selected_fusion_analysis(
    scenario: ScenarioConfig,
    requirements: Sequence[_Requirement],
    assignment: Sequence[Optional[_Candidate]],
    opportunities: Sequence[_FusionOpportunity],
) -> List[Dict[str, Any]]:
    del requirements  # member indices already bind the authoritative order
    records: List[Dict[str, Any]] = []
    for opportunity in opportunities:
        selected = tuple(
            assignment[index] for index in opportunity.member_indices
        )
        records.extend(
            _fusion_combo_analysis(scenario, opportunity, selected)
        )
    return records


def _solve_optimal(
    candidate_lists: Sequence[Tuple[_Candidate, ...]],
    capacities: Mapping[str, int],
    base_usage: Mapping[str, int],
    options: PlacementPolicy,
    started: float,
    warnings: List[str],
    *,
    missing_weights: Optional[Sequence[int]] = None,
) -> _SolveResult:
    remaining = max(0.0, options.time_limit_s - (time.monotonic() - started))
    if options.time_limit_s < 1.0e-6:
        # Sub-microsecond limits are below the portable timer/scheduler
        # resolution.  Treat them as an explicit no-search request instead of
        # occasionally claiming optimality when two monotonic reads coincide.
        remaining = 0.0
    if options.solver in {"auto", "ortools"}:
        try:
            importlib.import_module("ortools.sat.python.cp_model")
        except ImportError:
            warnings.append(
                "OR-Tools 不可用；已回退到确定性的内置分支定界求解器。"
            )
        else:
            try:
                return _solve_ortools(
                    candidate_lists,
                    capacities,
                    base_usage,
                    remaining,
                    missing_weights=missing_weights,
                )
            except Exception as exc:  # pragma: no cover - optional adapter guard
                warnings.append(
                    "OR-Tools 适配器运行失败；已回退到内置分支定界求解器。"
                )
    return _solve_builtin(
        candidate_lists,
        capacities,
        base_usage,
        remaining,
        missing_weights=missing_weights,
    )


def _heuristic_assignment(
    candidate_lists: Sequence[Tuple[_Candidate, ...]],
    capacities: Mapping[str, int],
    base_usage: Mapping[str, int],
) -> Tuple[Optional[_Candidate], ...]:
    used = dict(base_usage)
    assignment: List[Optional[_Candidate]] = [None] * len(candidate_lists)
    order = sorted(
        range(len(candidate_lists)),
        key=lambda index: (
            len(candidate_lists[index]) or _INFINITE_CAPACITY,
            -max(
                (sum(value for _, value in candidate.usage) for candidate in candidate_lists[index]),
                default=0,
            ),
            index,
        ),
    )
    for index in order:
        for candidate in candidate_lists[index]:
            if _fits(candidate, used, capacities):
                assignment[index] = candidate
                _consume(candidate, used)
                break
    return tuple(assignment)


def _solve_builtin(
    candidate_lists: Sequence[Tuple[_Candidate, ...]],
    capacities: Mapping[str, int],
    base_usage: Mapping[str, int],
    time_limit_s: float,
    *,
    missing_weights: Optional[Sequence[int]] = None,
) -> _SolveResult:
    normalized_missing_weights = _normalized_missing_weights(
        candidate_lists, missing_weights
    )
    heuristic = _heuristic_assignment(candidate_lists, capacities, base_usage)
    penalty_units = _unplaced_penalty_units(candidate_lists)
    incumbent = _quantized_penalized_cost(
        heuristic, penalty_units, normalized_missing_weights
    )
    best = list(heuristic)
    root_lower_bound = sum(
        min(
            (
                _candidate_objective_units(candidate, penalty_units)
                for candidate in candidates
            ),
            default=(
                penalty_units * normalized_missing_weights[index]
            ),
        )
        for index, candidates in enumerate(candidate_lists)
    )
    if time_limit_s <= 0.0:
        return _SolveResult(
            tuple(heuristic),
            False,
            incumbent / float(_COST_QUANTIZATION_SCALE),
            root_lower_bound / float(_COST_QUANTIZATION_SCALE),
            "builtin",
        )
    # If the independent lower bound is capacity-feasible, it is an immediate
    # deterministic certificate and avoids exponential work on large models.
    cheapest = tuple(candidates[0] if candidates else None for candidates in candidate_lists)
    if _assignment_fits(cheapest, capacities, base_usage):
        return _SolveResult(
            cheapest,
            True,
            _quantized_penalized_cost(
                cheapest, penalty_units, normalized_missing_weights
            )
            / float(_COST_QUANTIZATION_SCALE),
            _quantized_penalized_cost(
                cheapest, penalty_units, normalized_missing_weights
            )
            / float(_COST_QUANTIZATION_SCALE),
            "builtin",
        )

    deadline = time.monotonic() + max(0.0, time_limit_s)
    order = sorted(
        range(len(candidate_lists)),
        key=lambda index: (
            len(candidate_lists[index]) or _INFINITE_CAPACITY,
            -(
                _candidate_objective_units(
                    candidate_lists[index][1], penalty_units
                )
                - _candidate_objective_units(
                    candidate_lists[index][0], penalty_units
                )
                if len(candidate_lists[index]) > 1
                else (
                    penalty_units * normalized_missing_weights[index]
                )
            ),
            index,
        ),
    )
    suffix_lb = [0] * (len(order) + 1)
    for position in range(len(order) - 1, -1, -1):
        item_index = order[position]
        candidates = candidate_lists[item_index]
        suffix_lb[position] = suffix_lb[position + 1] + min(
            (
                _candidate_objective_units(candidate, penalty_units)
                for candidate in candidates
            ),
            default=(
                penalty_units * normalized_missing_weights[item_index]
            ),
        )
    used = dict(base_usage)
    current: List[Optional[_Candidate]] = [None] * len(candidate_lists)
    timed_out = False

    def visit(position: int, cost: int) -> None:
        nonlocal incumbent, best, timed_out
        if timed_out:
            return
        if time.monotonic() >= deadline:
            timed_out = True
            return
        if cost + suffix_lb[position] >= incumbent:
            return
        if position == len(order):
            signature = _assignment_signature(current)
            if cost < incumbent or (
                cost == incumbent
                and signature < _assignment_signature(best)
            ):
                incumbent = cost
                best = list(current)
            return
        requirement_index = order[position]
        for candidate in candidate_lists[requirement_index]:
            if not _fits(candidate, used, capacities):
                continue
            current[requirement_index] = candidate
            _consume(candidate, used)
            visit(
                position + 1,
                cost
                + _candidate_objective_units(candidate, penalty_units),
            )
            _release(candidate, used)
            current[requirement_index] = None
        # Explicit unplaced branch gives a best partial mapping if capacities
        # make complete placement impossible.
        visit(
            position + 1,
            cost
            + penalty_units
            * normalized_missing_weights[requirement_index],
        )

    visit(0, 0)
    assignment = tuple(best)
    final_lower_bound = incumbent if not timed_out else root_lower_bound
    return _SolveResult(
        assignment,
        not timed_out,
        incumbent / float(_COST_QUANTIZATION_SCALE),
        final_lower_bound / float(_COST_QUANTIZATION_SCALE),
        "builtin",
    )


def _solve_ortools(
    candidate_lists: Sequence[Tuple[_Candidate, ...]],
    capacities: Mapping[str, int],
    base_usage: Mapping[str, int],
    time_limit_s: float,
    *,
    missing_weights: Optional[Sequence[int]] = None,
) -> _SolveResult:
    cp_model = importlib.import_module("ortools.sat.python.cp_model")
    model = cp_model.CpModel()
    normalized_missing_weights = _normalized_missing_weights(
        candidate_lists, missing_weights
    )
    penalty_units = _unplaced_penalty_units(candidate_lists)
    variables: List[List[Any]] = []
    unplaced_variables: List[Any] = []
    objective_terms = []
    for index, candidates in enumerate(candidate_lists):
        row = [model.NewBoolVar("x_{}_{}".format(index, offset)) for offset in range(len(candidates))]
        missing = model.NewBoolVar("u_{}".format(index))
        model.Add(sum(row) + missing == 1)
        variables.append(row)
        unplaced_variables.append(missing)
        for variable, candidate in zip(row, candidates):
            objective_terms.append(
                (
                    _candidate_objective_units(candidate, penalty_units),
                    variable,
                )
            )
        objective_terms.append(
            (penalty_units * normalized_missing_weights[index], missing)
        )
    for component_id, limit in capacities.items():
        if limit >= _INFINITE_CAPACITY:
            continue
        terms = []
        for candidates, row in zip(candidate_lists, variables):
            for candidate, variable in zip(candidates, row):
                amount = dict(candidate.usage).get(component_id, 0)
                if amount:
                    terms.append(amount * variable)
        model.Add(sum(terms) + base_usage.get(component_id, 0) <= limit)
    model.Minimize(sum(coefficient * variable for coefficient, variable in objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1e-9, time_limit_s)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    heuristic = _heuristic_assignment(candidate_lists, capacities, base_usage)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _SolveResult(
            heuristic,
            False,
            _quantized_penalized_cost(
                heuristic, penalty_units, normalized_missing_weights
            )
            / float(_COST_QUANTIZATION_SCALE),
            None,
            "ortools",
        )
    assignment: List[Optional[_Candidate]] = []
    for candidates, row in zip(candidate_lists, variables):
        selected = None
        for candidate, variable in zip(candidates, row):
            if solver.Value(variable):
                selected = candidate
                break
        assignment.append(selected)
    solver_assignment = tuple(assignment)
    solver_cost_units = _quantized_penalized_cost(
        solver_assignment, penalty_units, normalized_missing_weights
    )
    heuristic_cost_units = _quantized_penalized_cost(
        heuristic, penalty_units, normalized_missing_weights
    )
    if heuristic_cost_units < solver_cost_units or (
        heuristic_cost_units == solver_cost_units
        and _assignment_signature(heuristic)
        < _assignment_signature(solver_assignment)
    ):
        selected_assignment = heuristic
        selected_cost_units = heuristic_cost_units
    else:
        selected_assignment = solver_assignment
        selected_cost_units = solver_cost_units
    completed = status == cp_model.OPTIMAL
    objective_value = (
        selected_cost_units / float(_COST_QUANTIZATION_SCALE)
    )
    bound = (
        objective_value
        if completed
        else min(
            objective_value,
            float(solver.BestObjectiveBound())
            / _COST_QUANTIZATION_SCALE,
        )
    )
    return _SolveResult(
        selected_assignment,
        completed,
        objective_value,
        bound,
        "ortools",
    )


def _component_capacities(scenario: ScenarioConfig) -> Dict[str, int]:
    capacities: Dict[str, int] = {}
    for component in scenario.hardware.components:
        physical = component.capacity_bytes
        if physical <= 0:
            physical = (
                0
                if _kind(component)
                in (OFFLOAD_STORAGE_COMPONENT_KINDS | ACTIVE_MEMORY_COMPONENT_KINDS)
                else _INFINITE_CAPACITY
            )
        if _is_cim(component):
            cim_profile = scenario.resolve_component_profile(
                component, DigitalSramCimProfile
            )
            physical = min(physical, cim_profile.weight_capacity_bytes)
        capacities[component.component_id] = physical
    return capacities


def _base_capacity_usage(
    scenario: ScenarioConfig,
    excluded_tensor_ids: Iterable[Optional[str]],
) -> Dict[str, int]:
    excluded = set(excluded_tensor_ids)
    usage: Dict[str, int] = {}
    for tensor_id, byte_count in scenario.placement.tensor_bytes.items():
        if tensor_id in excluded:
            continue
        component_id = scenario.placement.tensor_to_component.get(tensor_id)
        if component_id:
            usage[str(component_id)] = usage.get(str(component_id), 0) + int(byte_count)
    return usage


def _attention_head_dim(layer: LayerSpec) -> int:
    explicit = getattr(layer, "attention_head_dim", 0)
    if not explicit:
        explicit = layer.metadata.get("attention_head_dim", 0)
    try:
        value = int(explicit)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    return int(math.ceil(layer.hidden_size / float(layer.attention_heads)))


def _activation_bits(layer: LayerSpec) -> int:
    return _layer_precision_bits(layer)[0]


def _weight_bits(layer: LayerSpec) -> int:
    return _layer_precision_bits(layer)[1]


def _layer_precision_bits(layer: LayerSpec) -> Tuple[int, int]:
    return layer_precision_bits(
        layer.dtype,
        layer.quantization,
        unsupported_dtype_message="不支持的数据类型 {}".format(layer.dtype),
    )


def _allocate_declared_bytes(raw: Sequence[int], declared: int) -> Tuple[int, ...]:
    if not raw:
        return ()
    if declared <= 0 or sum(raw) <= 0:
        return tuple(raw)
    total = sum(raw)
    allocated = [declared * value // total for value in raw]
    remainder = declared - sum(allocated)
    # Stable largest-remainder allocation keeps the exact ModelSpec total.
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(declared * raw[index] % total), index),
    )
    for index in order[:remainder]:
        allocated[index] += 1
    return tuple(allocated)


def _matrix_bytes(k: int, n: int, bits: int) -> int:
    return _storage_bytes(k * n, bits)


def _storage_bytes(elements: int, bits: int) -> int:
    return _ceil_div(elements * bits, 8)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _kind(component: ComponentSpec) -> str:
    return normalize_component_kind(component.kind)


def _is_cim(component: ComponentSpec) -> bool:
    return "cim" in _kind(component)


def _writable(component: ComponentSpec) -> bool:
    return component.is_writable


def _validate_v4_planner_boundary(scenario: ScenarioConfig) -> None:
    """Reject retired/manual authoring state even when callers bypass parsing."""

    metadata = scenario.placement.metadata
    if "auto_mapping" in metadata:
        raise ValueError(
            "V4 planner does not accept retired placement.metadata.auto_mapping"
        )
    raw_control_plane = metadata.get("control_plane", {})
    if not isinstance(raw_control_plane, Mapping):
        raise ValueError("placement.metadata.control_plane must be a mapping")
    raw_policy = raw_control_plane.get("policy", {})
    raw_decision = raw_control_plane.get("decision", {})
    raw_evidence = raw_control_plane.get("evidence", {})
    if not isinstance(raw_policy, Mapping):
        raise ValueError("placement.metadata.control_plane.policy must be a mapping")
    if not isinstance(raw_decision, Mapping):
        raise ValueError("placement.metadata.control_plane.decision must be a mapping")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("placement.metadata.control_plane.evidence must be a mapping")

    retired = tuple(
        field_name
        for field_name in ("locked_op_keys", "locked_tensor_ids")
        if field_name in raw_policy
    )
    if retired:
        raise ValueError(
            "V4 planner does not accept retired manual control-plane locks: {}"
            .format(", ".join(retired))
        )
    unknown_policy = sorted(set(raw_policy) - {"options"})
    if unknown_policy:
        raise ValueError(
            "placement.metadata.control_plane.policy has unknown fields: {}"
            .format(", ".join(unknown_policy))
        )
    raw_options = raw_policy.get("options", {})
    if not isinstance(raw_options, Mapping):
        raise ValueError(
            "placement.metadata.control_plane.policy.options must be a mapping"
        )
    public_options = {
        "mode",
        "objective",
        "time_limit_s",
        "solver",
        "allow_cold_cim_streaming",
        "design_prefill_tokens",
        "design_decode_batch_size",
        "design_throughput_tokens",
        "gpu_loadable_layers",
        "gpu_loadable_order",
        "tied_weight_runtime_copies",
    }
    unknown_options = sorted(set(raw_options) - public_options)
    if unknown_options:
        raise ValueError(
            "placement.metadata.control_plane.policy.options has unknown fields: {}"
            .format(", ".join(unknown_options))
        )

    def string_set(source: Mapping[str, Any], key: str) -> FrozenSet[str]:
        raw = source.get(key, ())
        if raw is None:
            return frozenset()
        if not isinstance(raw, (list, tuple, set, frozenset)) or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            raise ValueError(
                "placement.metadata.control_plane.decision.{} must be a string array"
                .format(key)
            )
        return frozenset(item.strip() for item in raw)

    generated_ops = string_set(raw_decision, "generated_op_keys")
    generated_tensors = string_set(raw_decision, "generated_tensor_ids")
    derived_byte_ids = set()
    for key in ("derived_tensor_bytes", "physical_tensor_bytes", "padded_tensor_bytes"):
        raw = raw_decision.get(key, {})
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(
                "placement.metadata.control_plane.decision.{} must be a mapping"
                .format(key)
            )
        if any(
            not isinstance(tensor_id, str)
            or not tensor_id.strip()
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            for tensor_id, byte_count in raw.items()
        ):
            raise ValueError(
                "placement.metadata.control_plane.decision.{} must map non-empty "
                "tensor ids to nonnegative integer bytes".format(key)
            )
        derived_byte_ids.update(tensor_id.strip() for tensor_id in raw)

    placement = scenario.placement
    has_materialized_state = bool(
        placement.op_to_component
        or placement.tensor_to_component
        or placement.tensor_bytes
    )
    if not has_materialized_state:
        return
    stored_fingerprint = raw_evidence.get("input_fingerprint")
    stored_algorithm = raw_evidence.get("fingerprint_algorithm")
    stored_schema = raw_evidence.get("fingerprint_schema")
    if (
        stored_algorithm != MAPPING_FINGERPRINT_ALGORITHM
        or stored_schema != MAPPING_FINGERPRINT_SCHEMA
        or not isinstance(stored_fingerprint, str)
        or len(stored_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in stored_fingerprint)
    ):
        raise ValueError(
            "V4 planner accepts materialized placement only with prior control-plane fingerprint evidence"
        )

    unexpected_ops = sorted(set(placement.op_to_component) - generated_ops)
    unexpected_tensor_targets = sorted(
        set(placement.tensor_to_component) - generated_tensors
    )
    allowed_byte_ids = generated_tensors | frozenset(derived_byte_ids)
    unexpected_tensor_bytes = sorted(set(placement.tensor_bytes) - allowed_byte_ids)
    if unexpected_ops or unexpected_tensor_targets or unexpected_tensor_bytes:
        details = []
        if unexpected_ops:
            details.append("op_to_component=" + ",".join(unexpected_ops))
        if unexpected_tensor_targets:
            details.append(
                "tensor_to_component=" + ",".join(unexpected_tensor_targets)
            )
        if unexpected_tensor_bytes:
            details.append("tensor_bytes=" + ",".join(unexpected_tensor_bytes))
        raise ValueError(
            "V4 planner does not accept manual placement maps: {}".format(
                "; ".join(details)
            )
        )


def _mapping_controls(
    scenario: ScenarioConfig, warnings: List[str]
) -> _MappingControls:
    raw_control_plane = scenario.placement.metadata.get("control_plane", {})
    if not isinstance(raw_control_plane, Mapping):
        warnings.append(
            "placement.metadata.control_plane 不是 mapping；已忽略控制平面约定"
        )
        raw_control_plane = {}
    raw_policy = raw_control_plane.get("policy", {})
    raw_decision = raw_control_plane.get("decision", {})
    if not isinstance(raw_policy, Mapping):
        warnings.append(
            "placement.metadata.control_plane.policy 不是 mapping；已忽略放置策略"
        )
        raw_policy = {}
    if not isinstance(raw_decision, Mapping):
        warnings.append(
            "placement.metadata.control_plane.decision 不是 mapping；已忽略决策台账"
        )
        raw_decision = {}

    def string_set(
        source: Mapping[str, Any], key: str, section: str
    ) -> FrozenSet[str]:
        raw = source.get(key, ())
        if raw is None:
            return frozenset()
        if isinstance(raw, (list, tuple, set, frozenset)) and all(
            isinstance(item, str) and item.strip() for item in raw
        ):
            return frozenset(str(item) for item in raw)
        warnings.append(
            "placement.metadata.control_plane.{}.{} 必须是非空字符串数组；已忽略该字段"
            .format(section, key)
        )
        return frozenset()

    previous_generated_op_keys = string_set(
        raw_decision, "generated_op_keys", "decision"
    )
    previous_generated_tensor_ids = string_set(
        raw_decision, "generated_tensor_ids", "decision"
    )
    return _MappingControls(
        previous_generated_op_keys=previous_generated_op_keys,
        previous_generated_tensor_ids=previous_generated_tensor_ids,
    )


def _requirement_stage(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    run_context: Optional[_MappingRunContext] = None,
) -> Optional[int]:
    plan = _run_parallel_plan(scenario, run_context)
    if requirement.state_tensor:
        return None
    if requirement.kind in {
        "lm_head",
        "mtp_prediction_layer",
        "mtp_aux_head",
    }:
        return plan.pp_degree - 1
    if requirement.layer is None:
        return 0
    return plan.stage_for_layer(requirement.layer)


def _requirement_gpu_components(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    gpus: Sequence[ComponentSpec],
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[ComponentSpec, ...]:
    component_map = scenario.hardware.component_map()
    ranks = _requirement_execution_ranks(
        scenario, requirement, gpus, run_context
    )
    selected = tuple(
        component_map[component_id]
        for component_id in dict.fromkeys(
            rank.component_id for rank in ranks
        )
        if component_id in component_map
        and _kind(component_map[component_id]) == "gpu"
    )
    return selected or tuple(gpus)


def _cim_execution_pairs(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    cim_ids: Sequence[str],
    fallback_gpu_ids: Sequence[str],
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[Tuple[str, str], ...]:
    components = scenario.hardware.component_map()
    gpus = tuple(
        components[gpu_id]
        for gpu_id in fallback_gpu_ids
        if gpu_id in components and _kind(components[gpu_id]) == "gpu"
    )
    selected_ranks = _requirement_execution_ranks(
        scenario, requirement, gpus, run_context
    )
    pairs = []
    for rank in selected_ranks:
        target = rank.cim_component_id
        if target and target in cim_ids:
            pairs.append((rank.component_id, target))
    if pairs:
        return tuple(dict.fromkeys(pairs))
    selected_gpu_ids = tuple(rank.component_id for rank in selected_ranks)
    if not cim_ids or not selected_gpu_ids:
        return ()
    if len(cim_ids) == 1:
        return tuple((gpu_id, cim_ids[0]) for gpu_id in selected_gpu_ids)
    return tuple(
        (gpu_id, cim_ids[min(index, len(cim_ids) - 1)])
        for index, gpu_id in enumerate(selected_gpu_ids)
    )


def _has_route(router: TopologyRouter, source: str, target: str) -> bool:
    try:
        router.route(source, target, 1)
        return True
    except ValueError:
        return False


def _state_reachable(
    router: TopologyRouter,
    store_id: str,
    gpus: Sequence[ComponentSpec],
) -> bool:
    return bool(gpus) and all(
        _has_route(router, store_id, gpu.component_id)
        and _has_route(router, gpu.component_id, store_id)
        for gpu in gpus
    )


def _cim_activation_reachable(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    router: TopologyRouter,
    cim_id: str,
    gpus: Sequence[ComponentSpec],
    run_context: Optional[_MappingRunContext] = None,
) -> bool:
    ranks = _requirement_execution_ranks(
        scenario, requirement, gpus, run_context
    )
    associated = tuple(
        rank.component_id
        for rank in ranks
        if rank.cim_component_id == cim_id
    )
    gpu_ids = associated or tuple(rank.component_id for rank in ranks)
    return bool(gpu_ids) and all(
        _has_route(router, gpu_id, cim_id)
        and _has_route(router, cim_id, gpu_id)
        for gpu_id in gpu_ids
    )


def _cim_target_ids(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    configured_cim_id: str,
    run_context: Optional[_MappingRunContext] = None,
) -> Tuple[str, ...]:
    """Return physical CIMs used by the current fixed parallel rank mapping."""

    gpus = tuple(
        component
        for component in scenario.hardware.components
        if _kind(component) == "gpu"
    )
    ranks = _requirement_execution_ranks(
        scenario, requirement, gpus, run_context
    )
    targets = [rank.cim_component_id or configured_cim_id for rank in ranks]
    return tuple(dict.fromkeys(targets or [configured_cim_id]))


def _cold_cim_streaming_bytes_by_target(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    configured_cim_id: str,
    physical_targets: Sequence[str],
    run_context: Optional[_MappingRunContext] = None,
) -> Dict[str, int]:
    gpus = tuple(
        component
        for component in scenario.hardware.components
        if _kind(component) == "gpu"
    )
    target_bytes: Dict[str, int] = {}
    for rank in _requirement_weight_ranks(
        scenario, requirement, gpus, run_context
    ):
        target_id = rank.cim_component_id or configured_cim_id
        target_bytes[target_id] = (
            target_bytes.get(target_id, 0)
            + _logical_rank_shard_bytes(
                scenario,
                requirement,
                rank,
                run_context,
            )
        )
    for target_id in physical_targets:
        target_bytes.setdefault(target_id, requirement.tensor_bytes)
    return target_bytes


def _select_cold_cim_backing_component(
    scenario: ScenarioConfig,
    requirement: _Requirement,
    configured_cim_id: str,
    router: TopologyRouter,
    physical_targets: Sequence[str],
    capacities: Mapping[str, int],
    base_usage: Mapping[str, int],
    run_context: Optional[_MappingRunContext] = None,
) -> Optional[ComponentSpec]:
    """Choose a declared backing source without accepting an authored map."""

    if not physical_targets:
        return None
    required_backing_bytes = max(
        requirement.tensor_bytes,
        scenario.model.total_declared_weight_bytes,
    )
    target_bytes = _cold_cim_streaming_bytes_by_target(
        scenario,
        requirement,
        configured_cim_id,
        physical_targets,
        run_context,
    )
    ranked: List[Tuple[float, str, ComponentSpec]] = []
    for component in scenario.hardware.components:
        component_id = component.component_id
        if _kind(component) not in _COLD_CIM_BACKING_COMPONENT_KINDS:
            continue
        if component.capacity_bytes <= 0:
            continue
        if (
            base_usage.get(component_id, 0) + required_backing_bytes
            > capacities.get(component_id, 0)
        ):
            continue
        if any(
            not _has_route(router, component_id, target_id)
            for target_id in physical_targets
        ):
            continue
        route_cost = sum(
            _route_cost(router, component_id, target_id, byte_count)
            for target_id, byte_count in target_bytes.items()
            if component_id != target_id
        )
        ranked.append((route_cost, component_id, component))
    if not ranked:
        return None
    return min(ranked, key=lambda item: (item[0], item[1]))[2]


def _route_cost(
    router: TopologyRouter,
    source: str,
    target: str,
    byte_count: int,
) -> float:
    return sum(
        hop.transfer_ns(byte_count)
        for hop in router.route(source, target, max(0, byte_count))
    )


def _state_route_cost(
    router: TopologyRouter,
    store_id: str,
    gpus: Sequence[ComponentSpec],
    byte_count: int,
) -> float:
    if not gpus:
        return 0.0
    return sum(
        _route_cost(router, store_id, gpu.component_id, byte_count)
        + _route_cost(router, gpu.component_id, store_id, byte_count)
        for gpu in gpus
    )


def _fits(
    candidate: _Candidate,
    used: Mapping[str, int],
    capacities: Mapping[str, int],
) -> bool:
    return all(
        used.get(component_id, 0) + amount
        <= capacities.get(component_id, _INFINITE_CAPACITY)
        for component_id, amount in candidate.usage
    )


def _consume(candidate: _Candidate, used: Dict[str, int]) -> None:
    for component_id, amount in candidate.usage:
        used[component_id] = used.get(component_id, 0) + amount


def _release(candidate: _Candidate, used: Dict[str, int]) -> None:
    for component_id, amount in candidate.usage:
        used[component_id] = used.get(component_id, 0) - amount


def _assignment_fits(
    assignment: Sequence[Optional[_Candidate]],
    capacities: Mapping[str, int],
    base_usage: Mapping[str, int],
) -> bool:
    if any(candidate is None for candidate in assignment):
        return False
    used = dict(base_usage)
    for candidate in assignment:
        assert candidate is not None
        if not _fits(candidate, used, capacities):
            return False
        _consume(candidate, used)
    return True


def _candidate_cost_units(candidate: _Candidate) -> int:
    return max(
        0, int(round(candidate.cost * _COST_QUANTIZATION_SCALE))
    )


def _candidate_objective_units(
    candidate: _Candidate, penalty_units: int
) -> int:
    return (
        _candidate_cost_units(candidate)
        + max(0, int(candidate.missing_count)) * penalty_units
    )


def _normalized_missing_weights(
    candidate_lists: Sequence[Tuple[_Candidate, ...]],
    missing_weights: Optional[Sequence[int]],
) -> Tuple[int, ...]:
    if missing_weights is None:
        return (1,) * len(candidate_lists)
    normalized = tuple(int(weight) for weight in missing_weights)
    if len(normalized) != len(candidate_lists) or any(
        weight <= 0 for weight in normalized
    ):
        raise ValueError(
            "missing_weights 必须与候选列表等长且全部为正整数"
        )
    return normalized


def _unplaced_penalty_units(
    candidate_lists: Sequence[Tuple[_Candidate, ...]],
) -> int:
    return max(
        1,
        1
        + sum(
            max(
                (_candidate_cost_units(candidate) for candidate in candidates),
                default=0,
            )
            for candidates in candidate_lists
        ),
    )


def _quantized_penalized_cost(
    assignment: Sequence[Optional[_Candidate]],
    penalty_units: int,
    missing_weights: Optional[Sequence[int]] = None,
) -> int:
    normalized_missing_weights = (
        (1,) * len(assignment)
        if missing_weights is None
        else tuple(int(weight) for weight in missing_weights)
    )
    if len(normalized_missing_weights) != len(assignment):
        raise ValueError("missing_weights 必须与 assignment 等长")
    return sum(
        (
            _candidate_objective_units(candidate, penalty_units)
            if candidate is not None
            else penalty_units * normalized_missing_weights[index]
        )
        for index, candidate in enumerate(assignment)
    )


def _assignment_signature(
    assignment: Sequence[Optional[_Candidate]],
) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        candidate.signature if candidate is not None else ("~", "~")
        for candidate in assignment
    )


def _candidate_sort_key(candidate: _Candidate) -> Tuple[int, float, str, str]:
    return (
        _candidate_cost_units(candidate),
        candidate.cost,
        candidate.component_id,
        candidate.tensor_component_id or "",
    )


def _relative_gap(value: float, lower_bound: Optional[float]) -> Optional[float]:
    if lower_bound is None:
        return None
    if not math.isfinite(value) or not math.isfinite(lower_bound):
        return None
    return max(0.0, value - lower_bound) / max(abs(value), 1e-12)


def _normalized(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_")


def _deduplicate(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values))


__all__ = [
    "PlacementPolicy",
    "PlacementDecision",
    "PlacementAction",
    "RuntimeTensorShard",
    "UnplacedRequirement",
    "plan_runtime_placement",
]
