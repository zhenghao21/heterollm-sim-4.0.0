"""Public analytical screening of architecture and placement candidates.

The scan in this module is intentionally independent from the ordered event
simulator.  It builds representative model GEMMs from the current
``ScenarioConfig``, combines them with the physical compute components and
logical ranks that really exist in that scenario, and delegates the numeric
roofline columns to :func:`evaluate_batched_gemm` in one batch.

The result is a JSON-safe dictionary for API/UI consumers.  It is a screening
view, not a claim that the returned candidates form a simultaneously feasible
schedule: contention, collectives, admission, preemption and ordered events
remain the responsibility of the normal simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .batched_accelerator import (
    BatchedGemmInput,
    _BACKENDS,
    evaluate_batched_gemm,
)
from .communication import TopologyRouter
from .config import ScenarioConfig, normalize_cost_profile_kind
from .cost_models import (
    DigitalSramCimProfile,
    GPUProfile,
    HBMProfile,
    HostMemoryProfile,
)
from .ir import (
    ComponentSpec,
    LayerSpec,
    model_graph_execution_view,
    normalize_component_kind,
)
from .parallel import LogicalRank, ParallelPlan, build_parallel_plan, shard_extent
from .precision import layer_precision_bits


ANALYSIS_KIND = "analytical_architecture_scan"


@dataclass(frozen=True)
class BatchedGemmCandidate:
    """One explainable row submitted to the vectorized GEMM evaluator."""

    candidate_id: str
    operator_id: str
    operator_kind: str
    layer_id: str
    placement_key: str
    current_placement_component_id: Optional[str]
    component_id: str
    component_kind: str
    capability_basis: str
    logical_rank: Optional[int]
    tp_rank: Optional[int]
    pp_rank: Optional[int]
    ep_rank: Optional[int]
    rank_component_id: Optional[str]
    rank_memory_component_id: Optional[str]
    rank_cim_component_id: Optional[str]
    m: int
    k: int
    n: int
    activation_bits: int
    weight_bits: int
    output_bits: int
    peak_tops: float
    memory_bandwidth_gb_s: float
    compute_efficiency: float
    memory_efficiency: float
    kernel_launch_ns: float
    communication_bytes: int = 0
    communication_bandwidth_gbps: float = 0.0
    communication_latency_ns: float = 0.0
    communication_path: Tuple[str, ...] = ()

    @property
    def is_current_placement(self) -> bool:
        return self.current_placement_component_id == self.component_id


@dataclass(frozen=True)
class _OperatorTemplate:
    operator_id: str
    operator_kind: str
    layer: LayerSpec
    placement_key: str
    m: int
    k: int
    n_global: int
    shard_axis: str
    stage: int
    rank_scope: str
    activation_bits: int
    weight_bits: int
    output_bits: int


@dataclass(frozen=True)
class _ComponentCapability:
    component: ComponentSpec
    eligible: bool
    capability_basis: str
    peak_tops: float
    memory_bandwidth_gb_s: float
    reason: str = ""


def scan_architecture_candidates(
    scenario: ScenarioConfig,
    *,
    backend: str = "auto",
    top_n: int = 20,
) -> Dict[str, Any]:
    """Return a stable, JSON-safe analytical scan for ``scenario``.

    ``backend`` is passed through to :func:`evaluate_batched_gemm`; CuPy stays
    optional and lazily loaded there.  ``top_n`` limits only the returned
    ranked rows, not the batch that is evaluated.
    """

    _validate_public_inputs(scenario, backend, top_n)
    diagnostics: List[str] = [
        "本结果是 analytical scan（分析型架构候选扫描），不是有序事件仿真；不包含并发争用、调度、抢占或集体通信时序。"
    ]

    router = TopologyRouter(scenario.hardware)
    capabilities = _component_capabilities(scenario)
    for capability in capabilities:
        if not capability.eligible:
            diagnostics.append(
                "已跳过组件 {}（{}）：{}。".format(
                    capability.component.component_id,
                    capability.component.normalized_kind,
                    capability.reason,
                )
            )

    plan: Optional[ParallelPlan]
    try:
        plan = build_parallel_plan(scenario)
    except ValueError as exc:
        plan = None
        diagnostics.append(
            "当前并行/rank 配置无法形成有效 ParallelPlan（{}）；本次退化为不带逻辑 rank 的组件级扫描。".format(
                exc
            )
        )

    operator_diagnostics: List[str] = []
    operators = _representative_operators(scenario, plan, operator_diagnostics)
    diagnostics.extend(operator_diagnostics)
    candidates, candidate_diagnostics = _build_candidates(
        scenario, plan, operators, capabilities, router
    )
    diagnostics.extend(candidate_diagnostics)

    component_rows = [_component_row(item) for item in capabilities]
    operator_rows = [_operator_row(item) for item in operators]
    rank_rows = _rank_rows(plan)
    eligible_count = sum(1 for item in capabilities if item.eligible)

    if not candidates:
        diagnostics.append(
            "没有可评估候选：请检查计算组件能力、内存带宽、CIM profile、算子精度和拓扑可达性。"
        )
        return _result_payload(
            scenario=scenario,
            backend_requested=backend.strip().lower(),
            backend_used="not_run",
            diagnostics=diagnostics,
            capabilities=capabilities,
            component_rows=component_rows,
            operator_rows=operator_rows,
            rank_rows=rank_rows,
            candidate_count=0,
            eligible_count=eligible_count,
            top_results=[],
        )

    # Candidate ids are the final deterministic tie breaker.  The backend's
    # stable sort therefore has identical semantics on NumPy and CuPy.
    candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    batch = _candidate_batch(candidates)
    evaluated = evaluate_batched_gemm(batch, backend=backend)
    diagnostics.extend(evaluated.diagnostics)

    top_results = []
    for index in evaluated.sort_order[: min(top_n, len(candidates))]:
        row_index = int(index)
        candidate = candidates[row_index]
        top_results.append(
            _candidate_result_row(
                candidate,
                position=int(evaluated.rank[row_index]) + 1,
                operations=float(evaluated.operations[row_index]),
                activation_bytes=float(evaluated.activation_bytes[row_index]),
                weight_bytes=float(evaluated.weight_bytes[row_index]),
                output_bytes=float(evaluated.output_bytes[row_index]),
                gemm_io_bytes=float(evaluated.gemm_io_bytes[row_index]),
                compute_ns=float(evaluated.compute_ns[row_index]),
                memory_ns=float(evaluated.memory_ns[row_index]),
                roofline_ns=float(evaluated.roofline_ns[row_index]),
                communication_ns=float(evaluated.communication_ns[row_index]),
                total_ns=float(evaluated.total_ns[row_index]),
                compute_utilization=float(
                    evaluated.compute_utilization[row_index]
                ),
                bound=str(evaluated.bound[row_index]),
            )
        )

    return _result_payload(
        scenario=scenario,
        backend_requested=evaluated.backend_requested,
        backend_used=evaluated.backend_used,
        diagnostics=diagnostics,
        capabilities=capabilities,
        component_rows=component_rows,
        operator_rows=operator_rows,
        rank_rows=rank_rows,
        candidate_count=len(candidates),
        eligible_count=eligible_count,
        top_results=top_results,
    )


def build_batched_gemm_candidates(
    scenario: ScenarioConfig,
) -> Tuple[BatchedGemmCandidate, ...]:
    """Build the stable candidate rows without executing the numeric backend."""

    if not isinstance(scenario, ScenarioConfig):
        raise ValueError("scenario 必须是 ScenarioConfig")
    router = TopologyRouter(scenario.hardware)
    capabilities = _component_capabilities(scenario)
    try:
        plan: Optional[ParallelPlan] = build_parallel_plan(scenario)
    except ValueError:
        plan = None
    operators = _representative_operators(scenario, plan, [])
    candidates, _ = _build_candidates(
        scenario, plan, operators, capabilities, router
    )
    return tuple(sorted(candidates, key=lambda item: item.candidate_id))


def _validate_public_inputs(
    scenario: ScenarioConfig, backend: str, top_n: int
) -> None:
    if not isinstance(scenario, ScenarioConfig):
        raise ValueError("scenario 必须是 ScenarioConfig")
    if not isinstance(backend, str) or backend.strip().lower() not in _BACKENDS:
        raise ValueError("backend 必须是 auto、numpy 或 cupy")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n 必须是大于 0 的整数")


def _component_capabilities(
    scenario: ScenarioConfig,
) -> Tuple[_ComponentCapability, ...]:
    rows = []
    for component in sorted(
        scenario.hardware.components, key=lambda item: item.component_id
    ):
        kind = normalize_component_kind(component.kind)
        if kind == "gpu":
            peak_tops = float(component.peak_ops_per_s) / 1.0e12
            bandwidth = _gpu_memory_bandwidth_gb_s(scenario, component)
            if peak_tops <= 0.0:
                rows.append(
                    _ComponentCapability(
                        component, False, "component.peak_ops_per_s", 0.0, bandwidth,
                        "GPU 未声明正数 peak_ops_per_s，不能虚构其 GEMM 算力",
                    )
                )
            elif bandwidth <= 0.0:
                rows.append(
                    _ComponentCapability(
                        component, False, "physical active-memory links", peak_tops, 0.0,
                        "GPU 没有声明本地读带宽，也没有连接带正带宽的活跃内存",
                    )
                )
            else:
                rows.append(
                    _ComponentCapability(
                        component,
                        True,
                        "component.peak_ops_per_s + physical active-memory links",
                        peak_tops,
                        bandwidth,
                    )
                )
            continue

        if _is_cim(component):
            profile = scenario.resolve_component_profile(
                component, DigitalSramCimProfile
            )
            capacity = _cim_capacity_bytes(scenario, component)
            if capacity <= 0:
                rows.append(
                    _ComponentCapability(
                        component, False, "component/profile capacity", 0.0, 0.0,
                        "CIM 没有正数权重容量",
                    )
                )
                continue
            bandwidth = min(
                float(profile.activation_bandwidth_gb_s),
                float(profile.output_bandwidth_gb_s),
                float(profile.load_bandwidth_gb_s),
            )
            rows.append(
                _ComponentCapability(
                    component,
                    True,
                    "component-bound DigitalSramCimProfile geometry",
                    0.0,
                    bandwidth,
                )
            )
            continue

        rows.append(
            _ComponentCapability(
                component,
                False,
                "component.kind",
                0.0,
                0.0,
                "该组件类型不是项目成本模型支持的 GPU 或数字 SRAM-CIM 执行器",
            )
        )
    return tuple(rows)


def _gpu_memory_bandwidth_gb_s(
    scenario: ScenarioConfig, gpu: ComponentSpec
) -> float:
    components = scenario.hardware.component_map()
    attached_gbps = 0.0
    seen_memory = set()
    for link in sorted(scenario.hardware.links, key=lambda item: item.link_id):
        if link.source_component == gpu.component_id:
            memory_id = link.target_component
            gpu_port_id, memory_port_id = link.source_port, link.target_port
        elif link.target_component == gpu.component_id:
            memory_id = link.source_component
            gpu_port_id, memory_port_id = link.target_port, link.source_port
        else:
            continue
        memory = components.get(memory_id)
        if memory is None or not memory.is_active_memory or memory_id in seen_memory:
            continue
        bandwidth = _link_bandwidth_gbps(
            scenario, link, gpu.component_id, gpu_port_id, memory_id, memory_port_id
        )
        if float(memory.read_bandwidth_gbps) > 0.0:
            bandwidth = min(bandwidth, float(memory.read_bandwidth_gbps))
        if bandwidth > 0.0:
            attached_gbps += bandwidth
            seen_memory.add(memory_id)
    if attached_gbps > 0.0:
        return attached_gbps / 8.0
    # Some component presets explicitly carry an aggregate local-memory
    # bandwidth on the GPU itself.  Use it only as a declared fallback.
    return max(0.0, float(gpu.read_bandwidth_gbps)) / 8.0


def _gpu_memory_efficiency(
    scenario: ScenarioConfig, gpu: ComponentSpec
) -> float:
    """Return the bandwidth-weighted efficiency of this GPU's real memories."""

    components = scenario.hardware.component_map()
    raw_bandwidth = 0.0
    effective_bandwidth = 0.0
    seen_memory = set()
    for link in sorted(scenario.hardware.links, key=lambda item: item.link_id):
        if link.source_component == gpu.component_id:
            memory_id = link.target_component
            gpu_port_id, memory_port_id = link.source_port, link.target_port
        elif link.target_component == gpu.component_id:
            memory_id = link.source_component
            gpu_port_id, memory_port_id = link.target_port, link.source_port
        else:
            continue
        memory = components.get(memory_id)
        if memory is None or not memory.is_active_memory or memory_id in seen_memory:
            continue
        bandwidth = _link_bandwidth_gbps(
            scenario, link, gpu.component_id, gpu_port_id, memory_id, memory_port_id
        )
        if float(memory.read_bandwidth_gbps) > 0.0:
            bandwidth = min(bandwidth, float(memory.read_bandwidth_gbps))
        if bandwidth <= 0.0:
            continue
        profile_kind = normalize_cost_profile_kind(memory.normalized_kind)
        if profile_kind == "hbm":
            profile = scenario.resolve_component_profile(memory, HBMProfile)
            efficiency = float(profile.efficiency)
        elif profile_kind == "host_memory":
            profile = scenario.resolve_component_profile(memory, HostMemoryProfile)
            efficiency = float(profile.efficiency)
        else:
            efficiency = 1.0
        raw_bandwidth += bandwidth
        effective_bandwidth += bandwidth * efficiency
        seen_memory.add(memory_id)
    if raw_bandwidth <= 0.0:
        return 1.0
    return effective_bandwidth / raw_bandwidth


def _link_bandwidth_gbps(
    scenario: ScenarioConfig,
    link: Any,
    source_id: str,
    source_port_id: str,
    target_id: str,
    target_port_id: str,
) -> float:
    if float(link.bandwidth_gbps) > 0.0:
        return float(link.bandwidth_gbps)
    positive = [
        float(value)
        for value in (
            scenario.hardware.get_port(source_id, source_port_id).bandwidth_gbps,
            scenario.hardware.get_port(target_id, target_port_id).bandwidth_gbps,
        )
        if float(value) > 0.0
    ]
    return min(positive) if positive else 0.0


def _representative_operators(
    scenario: ScenarioConfig,
    plan: Optional[ParallelPlan],
    diagnostics: List[str],
) -> Tuple[_OperatorTemplate, ...]:
    batch_tokens = _representative_batch_tokens(scenario)
    parallel = scenario.placement.parallel
    tp_degree = plan.tp_degree if plan is not None else parallel.tp_degree
    pp_degree = plan.pp_degree if plan is not None else parallel.pp_degree
    allow_padding = plan.allow_padding if plan is not None else parallel.allow_padding
    operators: List[_OperatorTemplate] = []
    execution_view = model_graph_execution_view(
        scenario.model.graph,
        schema_version=scenario.model.schema_version,
    )
    for layer_index, descriptor in enumerate(execution_view.layer_instances):
        layer = descriptor.layer
        try:
            activation_bits, weight_bits = _layer_precision_bits(layer)
        except ValueError as exc:
            diagnostics.append(
                "已跳过层 {} 的代表性 GEMM：{}。".format(layer.layer_id, exc)
            )
            continue
        stage = (
            plan.stage_for_layer(layer)
            if plan is not None
            else parallel.layer_to_stage.get(
                layer.layer_id,
                min(
                    pp_degree - 1,
                    layer_index * pp_degree // max(1, scenario.model.num_layers),
                ),
            )
        )
        if layer.is_linear_attention and layer.linear_attention is not None:
            geometry = layer.linear_attention
            projection_width = (
                geometry.query_width
                + geometry.key_width
                + geometry.value_width
            )
            attention_operators = [
                _OperatorTemplate(
                    "{}.linear_qkv_projection".format(layer.layer_id),
                    "linear_qkv_projection",
                    layer,
                    "{}.attention".format(layer.layer_id),
                    batch_tokens,
                    layer.hidden_size,
                    projection_width,
                    "n",
                    int(stage),
                    "tp",
                    activation_bits,
                    weight_bits,
                    max(16, activation_bits),
                )
            ]
            if geometry.output_gate:
                attention_operators.append(
                    _OperatorTemplate(
                        "{}.linear_output_gate_projection".format(
                            layer.layer_id
                        ),
                        "linear_output_gate_projection",
                        layer,
                        "{}.attention".format(layer.layer_id),
                        batch_tokens,
                        layer.hidden_size,
                        geometry.value_width,
                        "n",
                        int(stage),
                        "tp",
                        activation_bits,
                        weight_bits,
                        max(16, activation_bits),
                    )
                )
            attention_operators.append(
                _OperatorTemplate(
                    "{}.linear_output_projection".format(layer.layer_id),
                    "linear_output_projection",
                    layer,
                    "{}.attention".format(layer.layer_id),
                    batch_tokens,
                    _shard_size(
                        geometry.value_width, tp_degree, allow_padding
                    ),
                    layer.hidden_size,
                    "none",
                    int(stage),
                    "tp",
                    activation_bits,
                    weight_bits,
                    max(16, activation_bits),
                )
            )
            operators.extend(attention_operators)
        else:
            head_dim = layer.effective_attention_head_dim
            qkv_width = (
                layer.hidden_size + 2 * layer.effective_kv_heads * head_dim
            )
            operators.extend(
                (
                    _OperatorTemplate(
                        "{}.attention_qkv".format(layer.layer_id),
                        "attention_qkv",
                        layer,
                        "{}.attention".format(layer.layer_id),
                        batch_tokens,
                        layer.hidden_size,
                        qkv_width,
                        "n",
                        int(stage),
                        "tp",
                        activation_bits,
                        weight_bits,
                        max(16, activation_bits),
                    ),
                    _OperatorTemplate(
                        "{}.attention_output".format(layer.layer_id),
                        "attention_output",
                        layer,
                        "{}.attention".format(layer.layer_id),
                        batch_tokens,
                        _shard_size(
                            layer.hidden_size, tp_degree, allow_padding
                        ),
                        layer.hidden_size,
                        "none",
                        int(stage),
                        "tp",
                        activation_bits,
                        weight_bits,
                        max(16, activation_bits),
                    ),
                )
            )
        if layer.is_moe:
            expert_tokens = max(
                1,
                int(
                    math.ceil(
                        batch_tokens
                        * layer.experts_per_token
                        / float(layer.num_experts)
                    )
                ),
            )
            local_intermediate = _shard_size(
                layer.intermediate_size, tp_degree, allow_padding
            )
            operators.extend(
                (
                    _OperatorTemplate(
                        "{}.moe_expert_up".format(layer.layer_id),
                        "moe_expert_up",
                        layer,
                        "{}.experts".format(layer.layer_id),
                        expert_tokens,
                        layer.hidden_size,
                        2 * local_intermediate,
                        "none",
                        int(stage),
                        "all_stage",
                        activation_bits,
                        weight_bits,
                        max(16, activation_bits),
                    ),
                    _OperatorTemplate(
                        "{}.moe_expert_down".format(layer.layer_id),
                        "moe_expert_down",
                        layer,
                        "{}.experts".format(layer.layer_id),
                        expert_tokens,
                        local_intermediate,
                        layer.hidden_size,
                        "none",
                        int(stage),
                        "all_stage",
                        activation_bits,
                        weight_bits,
                        max(16, activation_bits),
                    ),
                )
            )
        else:
            local_intermediate = _shard_size(
                layer.intermediate_size, tp_degree, allow_padding
            )
            operators.extend(
                (
                    _OperatorTemplate(
                        "{}.mlp_up_gate".format(layer.layer_id),
                        "dense_mlp_up",
                        layer,
                        "{}.mlp".format(layer.layer_id),
                        batch_tokens,
                        layer.hidden_size,
                        2 * local_intermediate,
                        "none",
                        int(stage),
                        "tp",
                        activation_bits,
                        weight_bits,
                        max(16, activation_bits),
                    ),
                    _OperatorTemplate(
                        "{}.mlp_down".format(layer.layer_id),
                        "dense_mlp_down",
                        layer,
                        "{}.mlp".format(layer.layer_id),
                        batch_tokens,
                        local_intermediate,
                        layer.hidden_size,
                        "none",
                        int(stage),
                        "tp",
                        activation_bits,
                        weight_bits,
                        max(16, activation_bits),
                    ),
                )
            )
    diagnostics.append(
        "代表性 GEMM 的 M 取当前工作负载最大序列批量 {}；MoE 专家 M 按均匀路由的每专家 token 数计算。".format(
            batch_tokens
        )
    )
    return tuple(operators)


def _representative_batch_tokens(scenario: ScenarioConfig) -> int:
    return max(1, int(scenario.workload.scheduler.max_num_seqs))


def _shard_size(global_size: int, degree: int, allow_padding: bool) -> int:
    try:
        return shard_extent(
            global_size, degree, 0, allow_padding=allow_padding
        ).local_size
    except ValueError:
        # The invalid no-padding configuration is reported by ParallelPlan.
        # A component-level fallback still needs a positive representative
        # shape, so use the mathematical shard ceiling and label it analytical.
        return int(math.ceil(global_size / float(max(1, degree))))


def _layer_precision_bits(layer: LayerSpec) -> Tuple[int, int]:
    return layer_precision_bits(
        layer.dtype,
        layer.quantization,
        unsupported_dtype_message=(
            "不支持的数据类型 {}，不能猜测位宽".format(layer.dtype)
        ),
    )


def _build_candidates(
    scenario: ScenarioConfig,
    plan: Optional[ParallelPlan],
    operators: Sequence[_OperatorTemplate],
    capabilities: Sequence[_ComponentCapability],
    router: TopologyRouter,
) -> Tuple[Tuple[BatchedGemmCandidate, ...], Tuple[str, ...]]:
    candidates: List[BatchedGemmCandidate] = []
    diagnostics: List[str] = []
    eligible = tuple(item for item in capabilities if item.eligible)
    rejected_capacity = 0
    rejected_precision = 0
    rejected_route = 0

    for operator in operators:
        ranks: Tuple[Optional[LogicalRank], ...]
        if plan is None:
            ranks = (None,)
        elif operator.rank_scope == "all_stage":
            ranks = tuple(
                rank
                for rank in plan.ranks_for_stage(operator.stage)
                if not operator.layer.is_moe
                or _local_expert_count(operator.layer, plan, rank) > 0
            )
        else:
            ranks = tuple(plan.tp_group(operator.stage, 0))
        for rank in ranks:
            n = operator.n_global
            if operator.shard_axis == "n":
                degree = plan.tp_degree if plan is not None else scenario.placement.parallel.tp_degree
                allow_padding = plan.allow_padding if plan is not None else True
                n = _shard_size(operator.n_global, degree, allow_padding)
            for capability in eligible:
                component = capability.component
                route_values = _communication_values(
                    router,
                    rank.component_id if rank is not None else None,
                    component.component_id,
                    operator.m,
                    operator.k,
                    n,
                    operator.activation_bits,
                    operator.output_bits,
                    plan.routing_policy if plan is not None else "lowest_latency",
                )
                if route_values is None:
                    rejected_route += 1
                    continue
                communication_bytes, communication_bandwidth, communication_latency, path = route_values
                peak_tops = capability.peak_tops
                memory_bandwidth = capability.memory_bandwidth_gb_s
                compute_efficiency = 1.0
                memory_efficiency = 1.0
                kernel_launch_ns = 0.0
                capability_basis = capability.capability_basis

                if _is_cim(component):
                    profile = scenario.resolve_component_profile(
                        component, DigitalSramCimProfile
                    )
                    required_accumulator_bits = (
                        operator.activation_bits
                        + operator.weight_bits
                        + int(math.ceil(math.log2(operator.k)))
                        + int(profile.accumulator_guard_bits)
                    )
                    if min(32, int(profile.accumulator_bits)) < required_accumulator_bits:
                        rejected_precision += 1
                        continue
                    cim_values = _cim_candidate_values(
                        scenario,
                        component,
                        operator.m,
                        operator.k,
                        n,
                        operator.activation_bits,
                        operator.weight_bits,
                        operator.output_bits,
                    )
                    if cim_values is None:
                        if (
                            operator.activation_bits not in profile.supported_activation_bits
                            or operator.weight_bits not in profile.supported_weight_bits
                        ):
                            rejected_precision += 1
                        else:
                            rejected_capacity += 1
                        continue
                    peak_tops, memory_bandwidth, kernel_launch_ns = cim_values
                    compute_efficiency = 1.0
                    memory_efficiency = 1.0
                    capability_basis += " + candidate-specific array waves"
                else:
                    gpu_profile = scenario.resolve_component_profile(
                        component, GPUProfile
                    )
                    compute_efficiency = float(
                        gpu_profile.attainable_efficiency
                    )
                    memory_efficiency = _gpu_memory_efficiency(
                        scenario, component
                    )
                    kernel_launch_ns = float(gpu_profile.kernel_launch_ns)

                rank_label = (
                    "rank{:04d}".format(rank.rank)
                    if rank is not None
                    else "rank-none"
                )
                candidate_id = "{}|{}|{}".format(
                    operator.operator_id, rank_label, component.component_id
                )
                current_target = scenario.placement.op_to_component.get(
                    operator.placement_key
                )
                candidates.append(
                    BatchedGemmCandidate(
                        candidate_id=candidate_id,
                        operator_id=operator.operator_id,
                        operator_kind=operator.operator_kind,
                        layer_id=operator.layer.layer_id,
                        placement_key=operator.placement_key,
                        current_placement_component_id=current_target,
                        component_id=component.component_id,
                        component_kind=component.normalized_kind,
                        capability_basis=capability_basis,
                        logical_rank=rank.rank if rank is not None else None,
                        tp_rank=rank.tp_rank if rank is not None else None,
                        pp_rank=rank.pp_rank if rank is not None else None,
                        ep_rank=rank.ep_rank if rank is not None else None,
                        rank_component_id=(
                            rank.component_id if rank is not None else None
                        ),
                        rank_memory_component_id=(
                            rank.memory_component_id if rank is not None else None
                        ),
                        rank_cim_component_id=(
                            rank.cim_component_id if rank is not None else None
                        ),
                        m=operator.m,
                        k=operator.k,
                        n=n,
                        activation_bits=operator.activation_bits,
                        weight_bits=operator.weight_bits,
                        output_bits=operator.output_bits,
                        peak_tops=peak_tops,
                        memory_bandwidth_gb_s=memory_bandwidth,
                        compute_efficiency=compute_efficiency,
                        memory_efficiency=memory_efficiency,
                        kernel_launch_ns=kernel_launch_ns,
                        communication_bytes=communication_bytes,
                        communication_bandwidth_gbps=communication_bandwidth,
                        communication_latency_ns=communication_latency,
                        communication_path=path,
                    )
                )

    if rejected_capacity:
        diagnostics.append(
            "已跳过 {} 个 CIM 候选：代表算子的填充后权重超过组件/profile 的真实容量。".format(
                rejected_capacity
            )
        )
    if rejected_precision:
        diagnostics.append(
            "已跳过 {} 个 CIM 候选：profile 未声明支持该激活/权重位宽。".format(
                rejected_precision
            )
        )
    if rejected_route:
        diagnostics.append(
            "已跳过 {} 个候选：执行组件与对应 rank GPU 在当前拓扑上不可达。".format(
                rejected_route
            )
        )
    return tuple(candidates), tuple(diagnostics)


def _communication_values(
    router: TopologyRouter,
    source_id: Optional[str],
    target_id: str,
    m: int,
    k: int,
    n: int,
    activation_bits: int,
    output_bits: int,
    policy: str,
) -> Optional[Tuple[int, float, float, Tuple[str, ...]]]:
    if source_id is None or source_id == target_id:
        return (0, 0.0, 0.0, ())
    activation_bytes = int(math.ceil(m * k * activation_bits / 8.0))
    output_bytes = int(math.ceil(m * n * output_bits / 8.0))
    byte_count = activation_bytes + output_bytes
    try:
        forward = router.route(
            source_id, target_id, activation_bytes, policy=policy
        )
        reverse = router.route(target_id, source_id, output_bytes, policy=policy)
    except ValueError:
        return None
    hops = tuple(forward) + tuple(reverse)
    if not hops:
        return (0, 0.0, 0.0, ())
    bandwidth = min(float(hop.bandwidth_gbps) for hop in hops)
    # The batched formula applies one transfer term.  Summed physical hop
    # latency plus a bottleneck bandwidth is the conservative aggregation used
    # for this independent scan; it is not link contention simulation.
    latency = sum(float(hop.latency_ns) for hop in hops)
    path = tuple(hop.link_id for hop in hops)
    return byte_count, bandwidth, latency, path


def _cim_candidate_values(
    scenario: ScenarioConfig,
    component: ComponentSpec,
    m: int,
    k: int,
    n: int,
    activation_bits: int,
    weight_bits: int,
    output_bits: int,
) -> Optional[Tuple[float, float, float]]:
    profile = scenario.resolve_component_profile(
        component, DigitalSramCimProfile
    )
    if (
        activation_bits not in profile.supported_activation_bits
        or weight_bits not in profile.supported_weight_bits
    ):
        return None
    required_accumulator_bits = (
        activation_bits
        + weight_bits
        + int(math.ceil(math.log2(k)))
        + int(profile.accumulator_guard_bits)
    )
    if min(32, int(profile.accumulator_bits)) < required_accumulator_bits:
        return None
    n_m = _ceil_div(m, profile.p_m)
    n_k = _ceil_div(k, profile.p_k)
    n_n = _ceil_div(n, profile.p_n)
    padded_weight_elements = n_k * profile.p_k * n_n * profile.p_n
    padded_weight_bytes = int(math.ceil(padded_weight_elements * weight_bits / 8.0))
    if padded_weight_bytes > _cim_capacity_bytes(scenario, component):
        return None
    placements = n_k * n_n
    replication_by_arrays = max(1, profile.array_count // placements)
    replication_by_capacity = max(
        1, _cim_capacity_bytes(scenario, component) // padded_weight_bytes
    )
    replication = min(
        profile.max_m_replication,
        n_m,
        replication_by_arrays,
        replication_by_capacity,
    )
    active_arrays = min(profile.array_count, placements * replication)
    bit_slices = (
        _ceil_div(activation_bits, profile.input_parallel_bits)
        * _ceil_div(weight_bits, profile.weight_parallel_bits)
    )
    block_waves = _ceil_div(n_m * placements, active_arrays)
    array_service_ns = (
        block_waves * bit_slices * profile.cycles_per_eval
    ) / profile.frequency_ghz
    operations = 2.0 * m * k * n
    peak_tops = operations / (array_service_ns * 1000.0)
    bandwidths = [
        float(profile.activation_bandwidth_gb_s),
        float(profile.output_bandwidth_gb_s),
    ]
    if not scenario.weights_resident:
        bandwidths.append(float(profile.load_bandwidth_gb_s))
    memory_bandwidth = min(bandwidths)
    launch_ns = float(profile.peripheral_latency_ns)
    if not scenario.weights_resident:
        launch_ns += float(profile.load_latency_ns)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (peak_tops, memory_bandwidth)
    ):
        return None
    return peak_tops, memory_bandwidth, max(0.0, launch_ns)


def _cim_capacity_bytes(
    scenario: ScenarioConfig, component: ComponentSpec
) -> int:
    profile = scenario.resolve_component_profile(
        component, DigitalSramCimProfile
    )
    values = [
        int(value)
        for value in (
            component.capacity_bytes,
            profile.weight_capacity_bytes,
        )
        if int(value) > 0
    ]
    return min(values) if values else 0


def _candidate_batch(
    candidates: Sequence[BatchedGemmCandidate],
) -> BatchedGemmInput:
    return BatchedGemmInput(
        m=[item.m for item in candidates],
        k=[item.k for item in candidates],
        n=[item.n for item in candidates],
        peak_tops=[item.peak_tops for item in candidates],
        memory_bandwidth_gb_s=[
            item.memory_bandwidth_gb_s for item in candidates
        ],
        activation_bits=[item.activation_bits for item in candidates],
        weight_bits=[item.weight_bits for item in candidates],
        output_bits=[item.output_bits for item in candidates],
        compute_efficiency=[item.compute_efficiency for item in candidates],
        memory_efficiency=[item.memory_efficiency for item in candidates],
        kernel_launch_ns=[item.kernel_launch_ns for item in candidates],
        communication_bytes=[item.communication_bytes for item in candidates],
        communication_bandwidth_gbps=[
            item.communication_bandwidth_gbps for item in candidates
        ],
        communication_latency_ns=[
            item.communication_latency_ns for item in candidates
        ],
        candidate_ids=tuple(item.candidate_id for item in candidates),
    )


def _local_expert_count(
    layer: LayerSpec, plan: ParallelPlan, rank: LogicalRank
) -> int:
    base, extra = divmod(layer.num_experts, plan.ep_degree)
    return base + (1 if rank.ep_rank < extra else 0)


def _result_payload(
    *,
    scenario: ScenarioConfig,
    backend_requested: str,
    backend_used: str,
    diagnostics: Sequence[str],
    capabilities: Sequence[_ComponentCapability],
    component_rows: Sequence[Dict[str, Any]],
    operator_rows: Sequence[Dict[str, Any]],
    rank_rows: Sequence[Dict[str, Any]],
    candidate_count: int,
    eligible_count: int,
    top_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    counts = {
        "candidates": int(candidate_count),
        "components_total": len(capabilities),
        "components_eligible": int(eligible_count),
        "components_skipped": len(capabilities) - int(eligible_count),
        "operators": len(operator_rows),
        "ranks": len(rank_rows),
        "top_results": len(top_results),
    }
    return {
        "analysis_kind": ANALYSIS_KIND,
        "is_event_simulation": False,
        "notice": "分析型候选扫描（analytical scan），不是事件仿真或完整部署计划。",
        "scenario_name": scenario.name,
        "hardware_name": scenario.hardware.name,
        "model_name": scenario.model.name,
        "backend_requested": backend_requested,
        "backend_used": backend_used,
        "backend": {
            "requested": backend_requested,
            "used": backend_used,
        },
        "counts": counts,
        "statistics": dict(counts),
        "diagnostics": list(diagnostics),
        "components": list(component_rows),
        "operators": list(operator_rows),
        "rank_placement": list(rank_rows),
        "top_results": list(top_results),
    }


def _component_row(capability: _ComponentCapability) -> Dict[str, Any]:
    component = capability.component
    return {
        "component_id": component.component_id,
        "kind": component.normalized_kind,
        "eligible": capability.eligible,
        "capability_basis": capability.capability_basis,
        "peak_tops": float(capability.peak_tops),
        "memory_bandwidth_gb_s": float(capability.memory_bandwidth_gb_s),
        "capacity_bytes": int(component.capacity_bytes),
        "reason": capability.reason or None,
    }


def _operator_row(operator: _OperatorTemplate) -> Dict[str, Any]:
    return {
        "operator_id": operator.operator_id,
        "operator_kind": operator.operator_kind,
        "layer_id": operator.layer.layer_id,
        "placement_key": operator.placement_key,
        "stage": operator.stage,
        "rank_scope": operator.rank_scope,
        "m": operator.m,
        "k": operator.k,
        "n_global": operator.n_global,
        "shard_axis": operator.shard_axis,
        "activation_bits": operator.activation_bits,
        "weight_bits": operator.weight_bits,
        "output_bits": operator.output_bits,
    }


def _rank_rows(plan: Optional[ParallelPlan]) -> List[Dict[str, Any]]:
    if plan is None:
        return []
    return [
        {
            "rank": rank.rank,
            "component_id": rank.component_id,
            "tp_rank": rank.tp_rank,
            "pp_rank": rank.pp_rank,
            "ep_rank": rank.ep_rank,
            "memory_component_id": rank.memory_component_id,
            "cim_component_id": rank.cim_component_id,
        }
        for rank in sorted(plan.ranks, key=lambda item: item.rank)
    ]


def _candidate_result_row(
    candidate: BatchedGemmCandidate, **metrics: Any
) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "position": int(metrics["position"]),
        "operator_id": candidate.operator_id,
        "operator_kind": candidate.operator_kind,
        "layer_id": candidate.layer_id,
        "placement_key": candidate.placement_key,
        "current_placement_component_id": candidate.current_placement_component_id,
        "is_current_placement": candidate.is_current_placement,
        "component_id": candidate.component_id,
        "component_kind": candidate.component_kind,
        "capability_basis": candidate.capability_basis,
        "logical_rank": candidate.logical_rank,
        "tp_rank": candidate.tp_rank,
        "pp_rank": candidate.pp_rank,
        "ep_rank": candidate.ep_rank,
        "rank_component_id": candidate.rank_component_id,
        "rank_memory_component_id": candidate.rank_memory_component_id,
        "rank_cim_component_id": candidate.rank_cim_component_id,
        "m": candidate.m,
        "k": candidate.k,
        "n": candidate.n,
        "activation_bits": candidate.activation_bits,
        "weight_bits": candidate.weight_bits,
        "output_bits": candidate.output_bits,
        "peak_tops": float(candidate.peak_tops),
        "memory_bandwidth_gb_s": float(candidate.memory_bandwidth_gb_s),
        "communication_bytes": candidate.communication_bytes,
        "communication_bandwidth_gbps": float(
            candidate.communication_bandwidth_gbps
        ),
        "communication_latency_ns": float(candidate.communication_latency_ns),
        "communication_path": list(candidate.communication_path),
        "operations": float(metrics["operations"]),
        "activation_bytes": float(metrics["activation_bytes"]),
        "weight_bytes": float(metrics["weight_bytes"]),
        "output_bytes": float(metrics["output_bytes"]),
        "gemm_io_bytes": float(metrics["gemm_io_bytes"]),
        "compute_ns": float(metrics["compute_ns"]),
        "memory_ns": float(metrics["memory_ns"]),
        "roofline_ns": float(metrics["roofline_ns"]),
        "communication_ns": float(metrics["communication_ns"]),
        "total_ns": float(metrics["total_ns"]),
        "compute_utilization": float(metrics["compute_utilization"]),
        "bound": str(metrics["bound"]),
    }


def _is_cim(component: ComponentSpec) -> bool:
    kind = normalize_component_kind(component.kind)
    return "cim" in kind or "compute_in_memory" in kind


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


__all__ = (
    "ANALYSIS_KIND",
    "BatchedGemmCandidate",
    "build_batched_gemm_candidates",
    "scan_architecture_candidates",
)
