"""Human- and machine-readable simulation reports."""

from __future__ import annotations

import math
from dataclasses import replace
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import __version__
from .config import ScenarioConfig, normalize_cost_profile_kind
from .cost_models import (
    CPUProfile,
    DigitalSramCimProfile,
    GPUProfile,
    HBMProfile,
    HostMemoryProfile,
)
from .contracts import (
    ANALYTICAL_MODEL_VERSION,
    COMPONENT_TIMESERIES_SCHEMA_VERSION,
    ChangePointInterval,
    DEFAULT_COMPONENT_TIMESERIES_POINT_LIMIT,
    DEFAULT_VISUALIZATION_EVENT_LIMIT,
    DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT,
    EvidenceStatus,
    RunManifest,
    SeriesQuality,
    SIMULATION_SCHEMA_VERSION,
    RetentionPolicy,
    SimulationTrace,
    TaskCategory,
    TraceFidelity,
    VISUALIZATION_SCHEMA_VERSION,
    VisualizationTraceOptions,
)
from .control_plane import ControlPlaneBootstrap, bootstrap_control_plane
from .control_plane_state import control_plane_decision
from .execution_control import ExecutionControl
from .engine import simulate_schedule
from .metrics import (
    MetricsSummary,
    bound_change_point_intervals,
)
from .planner import (
    TopologyAwareBatchCostProvider,
    _compilation_scope,
    _execution_view,
    _host_output_contract_projection,
    _parallel_plan,
    _topology_router,
    compile_serving_cohort_schedule,
    compile_streaming_scenario,
    localized_manifest_assumptions,
    validate_scenario,
)
from .ir import (
    ACTIVE_MEMORY_COMPONENT_KINDS,
    OFFLOAD_STORAGE_COMPONENT_KINDS,
)
from .serde import to_primitive
from .serde import stable_hash
from .serving import (
    BatchCohort,
    RequestStatus,
    ServingResult,
    compile_serving_plan,
    simulate_online,
)
from .residency import AllocationResidencyManager
from .streaming_des import (
    ScheduleExecutionResult,
    execute_incremental_schedule,
)


_SCALABLE_REQUEST_DETAIL_LIMIT = 1_000
_SCALABLE_BATCH_HISTORY_LIMIT = 2_000
_SCALABLE_SCHEDULER_EVENT_LIMIT = 4_000
_SCALABLE_BATCH_ITEM_LIMIT = 32


@dataclass(frozen=True)
class ScenarioResult:
    scenario: ScenarioConfig
    execution: ScheduleExecutionResult
    validation_warnings: Tuple[str, ...]
    retention_policy: str
    validation_information: Tuple[str, ...] = ()

    @property
    def trace(self) -> SimulationTrace:
        return self.execution.trace

    @property
    def metrics(self) -> MetricsSummary:
        return self.execution.metrics


@dataclass(frozen=True)
class OnlineScenarioResult:
    scenario: ScenarioConfig
    serving: ServingResult
    manifest: RunManifest
    validation_warnings: Tuple[str, ...]
    retention_policy: str
    validation_information: Tuple[str, ...] = ()

    @property
    def trace(self) -> ServingResult:
        return self.serving

    @property
    def metrics(self) -> ServingResult:
        return self.serving


RunResult = Union[ScenarioResult, OnlineScenarioResult]
StaticRunResult = ScenarioResult


@dataclass(frozen=True)
class _OnlineReportCore:
    """Shared aggregate projection used by full and summary-only reports."""

    summary: Dict[str, Any]
    requests: Dict[str, Any]
    resource_utilization: Dict[str, float]
    measurement_windows_ns: Dict[str, List[float]]
    model_coverage: Dict[str, Any]


def _retention_policy(retention_policy: str) -> RetentionPolicy:
    if not isinstance(retention_policy, str):
        raise TypeError("retention_policy 必须是 exact、streaming 或 aggregate")
    try:
        return RetentionPolicy(retention_policy)
    except ValueError as exc:
        raise ValueError(
            "retention_policy 必须是 exact、streaming 或 aggregate"
        ) from exc


def run_scenario(
    scenario: ScenarioConfig,
    *,
    retention_policy: Optional[str] = None,
    control: Optional[ExecutionControl] = None,
    residency_manager: Optional[AllocationResidencyManager] = None,
    batch_lowerer: Optional[TopologyAwareBatchCostProvider] = None,
) -> RunResult:
    """Plan, validate, and execute through the V4 unified event kernel."""

    scheduler = getattr(scenario.workload, "scheduler", None)
    scheduler_mode = (
        str(getattr(scheduler, "mode", "static")) if scheduler else "static"
    )
    selected_policy = (
        RetentionPolicy.AGGREGATE
        if retention_policy is None and scheduler_mode == "continuous"
        else RetentionPolicy.EXACT
        if retention_policy is None
        else _retention_policy(retention_policy)
    )
    if (
        scheduler_mode == "continuous"
        and selected_policy is not RetentionPolicy.AGGREGATE
    ):
        raise ValueError(
            "continuous 调度仅支持 aggregate retention_policy；"
            "批次外层不会伪装成 exact 或 streaming 任务轨迹"
        )
    execution_control = control or ExecutionControl()
    execution_control.raise_if_cancelled()
    execution_control.report(
        "control_plane",
        0,
        1,
        message="正在执行 CPU 动态放置与运行时初始化",
    )
    bootstrap = bootstrap_control_plane(scenario)
    mapped_scenario = bootstrap.scenario
    if (
        batch_lowerer is not None
        and mapped_scenario is not scenario
        and isinstance(batch_lowerer, TopologyAwareBatchCostProvider)
    ):
        batch_lowerer._rebind_control_plane_successor(
            scenario,
            mapped_scenario,
        )
    execution_control.report(
        "control_plane",
        1,
        1,
        message="CPU 动态放置与运行时初始化完成",
        simulated_time_ns=bootstrap.makespan_ns,
        metadata={
            "completed_control_tasks": bootstrap.result.completed_count,
            "placement_status": bootstrap.decision.status,
        },
    )
    with _compilation_scope(mapped_scenario):
        return _run_scenario_in_context(
            mapped_scenario,
            retention_policy=selected_policy,
            control=execution_control,
            residency_manager=residency_manager,
            batch_lowerer=batch_lowerer,
            control_plane=bootstrap,
        )


def _run_scenario_in_context(
    scenario: ScenarioConfig,
    *,
    retention_policy: RetentionPolicy,
    control: Optional[ExecutionControl] = None,
    residency_manager: Optional[AllocationResidencyManager] = None,
    batch_lowerer: Optional[TopologyAwareBatchCostProvider] = None,
    control_plane: ControlPlaneBootstrap,
) -> RunResult:
    execution_control = control or ExecutionControl()
    execution_control.raise_if_cancelled()
    execution_control.report("validation", 0, 1, message="正在校验场景")
    validation = validate_scenario(scenario)
    validation.raise_for_errors()
    execution_control.report("validation", 1, 1, message="场景校验完成")
    scheduler = getattr(scenario.workload, "scheduler", None)
    scheduler_mode = (
        str(getattr(scheduler, "mode", "static")) if scheduler else "static"
    )
    if scheduler_mode == "continuous":
        active_batch_lowerer = batch_lowerer or TopologyAwareBatchCostProvider(
            scenario,
            execution_control=execution_control,
        )
        serving = simulate_online(
            scenario,
            batch_lowerer=active_batch_lowerer,
            execution_control=execution_control,
            residency_manager=residency_manager,
            execution_kernel=control_plane.kernel,
            runtime_origin_ns=control_plane.makespan_ns,
        )
        manifest_assumptions = (
            scenario.assumptions + validation.warnings + validation.information
        )
        manifest_assumptions_zh, manifest_assumptions_en = localized_manifest_assumptions(
            manifest_assumptions,
            assumptions_en=(
                scenario.assumptions
                + validation.warnings_en
                + validation.information_en
            ),
        )
        manifest = RunManifest(
            schema_version=SIMULATION_SCHEMA_VERSION,
            run_id=stable_hash(scenario)[:16],
            random_seed=scenario.workload.random_seed,
            simulator_version=__version__,
            model_name=scenario.model.name,
            hardware_name=scenario.hardware.name,
            workload_name=scenario.workload.name,
            calibration_version=ANALYTICAL_MODEL_VERSION,
            evidence=EvidenceStatus.ANALYTICAL,
            assumptions=manifest_assumptions,
            assumptions_zh=manifest_assumptions_zh,
            assumptions_en=manifest_assumptions_en,
            metadata={
                "scenario_name": scenario.name,
                "execution_mode": "continuous_batching",
                "retention_policy": retention_policy.value,
                "control_plane": {
                    "runtime": "dynamic_hardware_dag",
                    "makespan_ns": control_plane.makespan_ns,
                    "completed_task_count": control_plane.result.completed_count,
                    "placement_status": control_plane.decision.status,
                    "placement_fingerprint": control_plane.decision.input_fingerprint,
                },
            },
        )
        return OnlineScenarioResult(
            scenario=scenario,
            serving=serving,
            manifest=manifest,
            validation_warnings=validation.warnings,
            retention_policy=retention_policy.value,
            validation_information=validation.information,
        )

    execution_control.raise_if_cancelled()
    execution_control.report("compilation", 0, 1, message="正在编译统一执行计划")
    schedule = compile_streaming_scenario(scenario)
    execution_control.report(
        "compilation",
        1,
        1,
        message="统一执行计划编译完成",
        metadata={"retention_policy": retention_policy.value},
    )
    execution = execute_incremental_schedule(
        schedule,
        retention_policy=retention_policy,
        control=execution_control,
        execution_kernel=control_plane.kernel,
        runtime_origin_ns=control_plane.makespan_ns,
    )
    return ScenarioResult(
        scenario=scenario,
        execution=execution,
        validation_warnings=validation.warnings,
        retention_policy=retention_policy.value,
        validation_information=validation.information,
    )


def gpu_baseline_scenario(scenario: ScenarioConfig) -> ScenarioConfig:
    gpu_components = [
        component
        for component in scenario.hardware.components
        if component.kind.strip().lower().replace("-", "_") == "gpu"
    ]
    if not gpu_components:
        raise ValueError("缺少 GPU 组件，无法构建纯 GPU 基线。")
    gpu_id = gpu_components[0].component_id
    hbm_components = [
        component
        for component in scenario.hardware.components
        if component.kind.strip().lower().replace("-", "_") in {"hbm", "hbm_stack"}
    ]
    hbm_id = hbm_components[0].component_id if hbm_components else gpu_id
    component_map = scenario.hardware.component_map()
    tensor_mapping = {
        name: (
            hbm_id
            if component_id in component_map
            and "cim" in component_map[component_id].kind.strip().lower()
            else component_id
        )
        for name, component_id in scenario.placement.tensor_to_component.items()
    }
    placement = replace(
        scenario.placement,
        op_to_component={
            name: gpu_id for name in scenario.placement.op_to_component
        },
        tensor_to_component=tensor_mapping,
    )
    return replace(
        scenario,
        name=scenario.name + "-gpu-baseline",
        placement=placement,
        weights_resident=False,
        assumptions=scenario.assumptions
        + ("纯 GPU 基线会把所有已配置的 GEMM 组映射到参考 GPU。",),
    )


def compare_with_gpu_baseline(scenario: ScenarioConfig) -> Dict[str, Any]:
    candidate = run_scenario(scenario)
    baseline = run_scenario(gpu_baseline_scenario(scenario))
    candidate_report = report_dict(candidate)
    baseline_report = report_dict(baseline)
    candidate_makespan = float(candidate_report["summary"]["makespan_ns"])
    baseline_makespan = float(baseline_report["summary"]["makespan_ns"])
    candidate_tokens = float(
        candidate_report["summary"]["throughput"]["visible_output_tokens_per_s"]
    )
    baseline_tokens = float(
        baseline_report["summary"]["throughput"]["visible_output_tokens_per_s"]
    )
    candidate_energy = candidate_report["summary"]["total_energy_pj"]
    baseline_energy = baseline_report["summary"]["total_energy_pj"]
    return {
        "candidate": candidate_report,
        "gpu_baseline": baseline_report,
        "comparison": {
            "latency_speedup": (
                baseline_makespan / candidate_makespan if candidate_makespan > 0 else None
            ),
            "throughput_speedup": (
                candidate_tokens / baseline_tokens if baseline_tokens > 0 else None
            ),
            "energy_ratio": (
                candidate_energy / baseline_energy if baseline_energy > 0 else None
            ),
            "candidate_makespan_ns": candidate_makespan,
            "baseline_makespan_ns": baseline_makespan,
        },
    }


def format_comparison(scenario: ScenarioConfig) -> str:
    data = compare_with_gpu_baseline(scenario)
    comparison = data["comparison"]
    return "\n".join(
        (
            "候选场景：{}".format(data["candidate"]["scenario"]),
            "纯 GPU 基线：{}".format(data["gpu_baseline"]["scenario"]),
            "候选场景总时长：{:.3f} ms".format(
                comparison["candidate_makespan_ns"] / 1_000_000.0
            ),
            "基线总时长：{:.3f} ms".format(
                comparison["baseline_makespan_ns"] / 1_000_000.0
            ),
            "延迟加速比：{}".format(
                "NA"
                if comparison["latency_speedup"] is None
                else "{:.3f}x".format(comparison["latency_speedup"])
            ),
            "吞吐加速比：{}".format(
                "NA"
                if comparison["throughput_speedup"] is None
                else "{:.3f}x".format(comparison["throughput_speedup"])
            ),
            "能耗比：{}".format(
                "NA"
                if comparison["energy_ratio"] is None
                else "{:.3f}x".format(comparison["energy_ratio"])
            ),
            "证据等级：解析估算；用于架构决策前应使用实测数据校准性能配置。",
        )
    )


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    data = tuple(values)
    return {
        "p50": _percentile(data, 0.50),
        "p95": _percentile(data, 0.95),
        "p99": _percentile(data, 0.99),
    }


def _online_token_times(result: OnlineScenarioResult) -> Dict[str, List[float]]:
    token_times: Dict[str, List[float]] = {
        request_id: [] for request_id in result.serving.request_metrics
    }
    for event in result.serving.events:
        if event.event_type != "tokens_committed" or not event.request_id:
            continue
        visible = max(0, int(event.details.get("visible_tokens", 0)))
        token_times.setdefault(event.request_id, []).extend(
            [float(event.timestamp_ns)] * visible
        )
    return token_times


def _sum_batch_metadata(
    result: OnlineScenarioResult, key: str
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for batch in result.serving.batches:
        raw = batch.cost.metadata.get(key, {})
        if not isinstance(raw, Mapping):
            continue
        for name, value in raw.items():
            try:
                totals[str(name)] = totals.get(str(name), 0.0) + float(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(totals.items()))


def _sum_batch_coverage(
    result: OnlineScenarioResult,
) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = {}
    for batch in result.serving.batches:
        raw = batch.cost.metadata.get("coverage", {})
        if not isinstance(raw, Mapping):
            continue
        for component, metrics in raw.items():
            if not isinstance(metrics, Mapping):
                continue
            row = totals.setdefault(str(component), {})
            for name, value in metrics.items():
                try:
                    row[str(name)] = row.get(str(name), 0.0) + float(value)
                except (TypeError, ValueError):
                    continue
    return {
        component: dict(sorted(metrics.items()))
        for component, metrics in sorted(totals.items())
    }


def _sum_batch_coverage_references(
    result: OnlineScenarioResult,
) -> Dict[str, Dict[str, List[str]]]:
    rows: Dict[str, Dict[str, set]] = {}
    for batch in result.serving.batches:
        raw = batch.cost.metadata.get("coverage_references", {})
        if not isinstance(raw, Mapping):
            continue
        for component, references in raw.items():
            if not isinstance(references, Mapping):
                continue
            row = rows.setdefault(
                str(component), {"operator_ids": set(), "tensor_ids": set()}
            )
            for key in ("operator_ids", "tensor_ids"):
                values = references.get(key, ())
                if isinstance(values, (list, tuple, set, frozenset)):
                    row[key].update(str(item) for item in values if item)
    return {
        component: {
            key: sorted(values) for key, values in references.items()
        }
        for component, references in sorted(rows.items())
    }


def _model_coverage(scenario: ScenarioConfig) -> Dict[str, Any]:
    execution_view = _execution_view(scenario)
    layers = tuple(item.layer for item in execution_view.layer_instances)
    mtp_descriptors = execution_view.mtp_descriptors
    linear_state_bytes = 0
    for layer in layers:
        geometry = layer.linear_attention
        if not layer.is_linear_attention or geometry is None:
            continue
        normalized = geometry.state_dtype.lower().replace("-", "").replace("_", "")
        state_bits = {
            "fp32": 32,
            "float32": 32,
            "fp16": 16,
            "float16": 16,
            "bf16": 16,
            "bfloat16": 16,
            "fp8": 8,
            "float8": 8,
            "int8": 8,
            "uint8": 8,
            "int4": 4,
            "uint4": 4,
        }[normalized]
        linear_state_bytes += int(
            math.ceil(
                (
                    geometry.recurrent_state_elements
                    + geometry.convolution_state_elements
                )
                * state_bits
                / 8.0
            )
        )
    return {
        "ordered_text_backbone_layers": len(layers),
        "full_attention_layers": sum(
            not layer.is_linear_attention for layer in layers
        ),
        "linear_attention_layers": sum(
            layer.is_linear_attention for layer in layers
        ),
        "dense_ffn_layers": sum(not layer.is_moe for layer in layers),
        "routed_moe_layers": sum(layer.is_moe for layer in layers),
        "shared_expert_layers": sum(layer.has_shared_expert for layer in layers),
        "routed_experts_total": sum(
            layer.num_experts for layer in layers if layer.is_moe
        ),
        "shared_expert_intermediate_total": sum(
            layer.shared_expert_intermediate_size
            for layer in layers
            if layer.has_shared_expert
        ),
        "linear_state_bytes_per_request": linear_state_bytes,
        "mtp": {
            "prediction_layers": sum(
                descriptor.operator.op_kind == "mtp_prediction_layer"
                for descriptor in mtp_descriptors
            ),
            "auxiliary_head": any(
                descriptor.operator.op_kind == "mtp_aux_head"
                for descriptor in mtp_descriptors
            ),
            "declared_weight_bytes": sum(
                descriptor.weight_bytes for descriptor in mtp_descriptors
            ),
            "operator_ids": [
                descriptor.operator.operator_id
                for descriptor in mtp_descriptors
            ],
            "weight_tensor_ids": [
                descriptor.weight_tensor.tensor_id
                for descriptor in mtp_descriptors
            ],
            "weight_bytes_by_tensor": {
                descriptor.weight_tensor.tensor_id: descriptor.weight_bytes
                for descriptor in mtp_descriptors
            },
        },
        "text_backbone_only": scenario.model.text_backbone_only,
        "supported_modalities": list(scenario.model.supported_modalities),
        "excluded_subgraphs": list(scenario.model.excluded_subgraphs),
    }


def _integer_or_none(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _non_negative_int(value: Any, default: int = 0) -> int:
    parsed = _integer_or_none(value)
    return max(0, parsed) if parsed is not None else default


def _tensor_kind(tensor_id: str) -> str:
    normalized = tensor_id.lower()
    if "weight" in normalized or "expert" in normalized:
        return "weight"
    if "linear_state" in normalized:
        return "linear_state"
    if "kv" in normalized and "cache" in normalized:
        return "kv_cache"
    if "state" in normalized:
        return "state"
    return "tensor"


def _task_transfer_bytes(task: Any) -> int:
    metadata_bytes = _integer_or_none(task.metadata.get("bytes"))
    if metadata_bytes is not None:
        return max(0, metadata_bytes)
    return max(
        (
            max(0, int(interval.bytes_moved))
            for interval in task.resource_intervals
        ),
        default=0,
    )


def _state_tensor_id(task: Any) -> Optional[str]:
    event_kind = str(task.metadata.get("event_kind", ""))
    if not event_kind.startswith("linear_state_"):
        return None
    layer_id = task.metadata.get("layer_id")
    rank = task.metadata.get("rank")
    if layer_id is None or rank is None:
        return "linear_state"
    return "linear_state:{}:{}:rank-{}".format(
        task.request_id, layer_id, rank
    )


def _memory_layout(
    result: RunResult,
    *,
    segment_limit: int,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, Optional[int]], Dict[str, Any]]]:
    """Build deterministic component-local logical byte ranges.

    Offsets are simulator bookkeeping only.  They deliberately do not claim
    JEDEC channels, banks, rows, columns, or device physical addresses.
    """

    scenario = result.scenario
    placement = scenario.placement
    parallel = placement.parallel
    raw_segments: List[Dict[str, Any]] = []
    represented_weight_tensors = set()
    metadata = placement.metadata
    decision = control_plane_decision(scenario)
    aliases_raw = decision.get("logical_weight_aliases", {})
    aliases = aliases_raw if isinstance(aliases_raw, Mapping) else {}
    details_raw = decision.get("weight_tensor_details", {})
    weight_details = details_raw if isinstance(details_raw, Mapping) else {}
    rank_shards_raw = decision.get("rank_weight_shards", {})
    rank_weight_shards = (
        rank_shards_raw if isinstance(rank_shards_raw, Mapping) else {}
    )

    for tensor_key in sorted(
        {str(key) for key in weight_details}
        | {str(key) for key in rank_weight_shards}
    ):
        detail_raw = weight_details.get(tensor_key, {})
        if not isinstance(detail_raw, Mapping):
            detail_raw = {}
        logical_id = str(detail_raw.get("logical_tensor_id") or tensor_key)
        logical_bytes = _non_negative_int(
            detail_raw.get(
                "logical_bytes",
                placement.tensor_bytes.get(tensor_key, 0),
            )
        )
        shards_raw = rank_weight_shards.get(tensor_key, ())
        shards = (
            [item for item in shards_raw if isinstance(item, Mapping)]
            if isinstance(shards_raw, (list, tuple))
            else []
        )
        shards.sort(
            key=lambda item: (
                _non_negative_int(item.get("rank_id", item.get("rank"))),
                str(item.get("storage_component_id", "")),
                str(item.get("compute_component_id", "")),
            )
        )
        if shards:
            fallback_count = max(1, len(shards))
            fallback_length = int(math.ceil(logical_bytes / float(fallback_count))) if logical_bytes else 0
            for physical_index, shard in enumerate(shards):
                component_id = str(shard.get("storage_component_id", ""))
                length_bytes = _non_negative_int(
                    shard.get("physical_bytes"), fallback_length
                )
                if not component_id or length_bytes <= 0:
                    continue
                rank = _integer_or_none(shard.get("rank_id", shard.get("rank")))
                tp_rank = _non_negative_int(shard.get("tp_rank"))
                pp_rank = _non_negative_int(shard.get("pp_rank"))
                ep_rank = _non_negative_int(shard.get("ep_rank"))
                shard_kind = str(shard.get("shard_kind", "tp_shard"))
                if "tp_ep" in shard_kind:
                    shard_index = _non_negative_int(
                        shard.get("shard_index"),
                        ep_rank * max(1, parallel.tp_degree) + tp_rank,
                    )
                    shard_count = max(
                        1,
                        _non_negative_int(
                            shard.get("shard_count"),
                            parallel.tp_degree * parallel.ep_degree,
                        ),
                    )
                    replica_index = 0
                    replica_count = 1
                else:
                    shard_index = _non_negative_int(
                        shard.get("shard_index"),
                        tp_rank,
                    )
                    shard_count = max(
                        1,
                        _non_negative_int(
                            shard.get("shard_count"),
                            parallel.tp_degree,
                        ),
                    )
                    replica_index = ep_rank if "ep_replica" in shard_kind else 0
                    replica_count = (
                        max(1, parallel.ep_degree)
                        if "ep_replica" in shard_kind
                        else 1
                    )
                raw_segments.append(
                    {
                        "logical_id": logical_id,
                        "physical_id": "{}:rank-{}".format(
                            logical_id,
                            rank if rank is not None else physical_index,
                        ),
                        "component_id": component_id,
                        "length_bytes": length_bytes,
                        "allocation_kind": "weight",
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                        "replica_index": replica_index,
                        "replica_count": replica_count,
                        "rank": rank,
                        "tp_rank": tp_rank,
                        "pp_rank": pp_rank,
                        "ep_rank": ep_rank,
                        "shard_kind": shard_kind,
                        "logical_length_bytes": logical_bytes or None,
                        "compute_component_id": shard.get("compute_component_id"),
                        "semantics": "physical_weight_shard",
                    }
                )
            represented_weight_tensors.add(logical_id)
            continue

        replicas_raw = detail_raw.get("replica_component_ids", ())
        replicas = (
            sorted({str(item) for item in replicas_raw if item})
            if isinstance(replicas_raw, (list, tuple))
            else []
        )
        if replicas:
            replica_bytes = _non_negative_int(
                detail_raw.get("padded_bytes_per_replica"), logical_bytes
            )
            for replica_index, component_id in enumerate(replicas):
                if replica_bytes <= 0:
                    continue
                raw_segments.append(
                    {
                        "logical_id": logical_id,
                        "physical_id": "{}:replica-{}".format(
                            logical_id, replica_index
                        ),
                        "component_id": component_id,
                        "length_bytes": replica_bytes,
                        "allocation_kind": "weight",
                        "shard_index": 0,
                        "shard_count": 1,
                        "replica_index": replica_index,
                        "replica_count": len(replicas),
                        "rank": None,
                        "logical_length_bytes": logical_bytes or None,
                        "semantics": "physical_weight_replica",
                    }
                )
            represented_weight_tensors.add(logical_id)

    runtime_state_segments: Dict[Tuple[str, str, Optional[int]], Dict[str, Any]] = {}
    if not isinstance(result, OnlineScenarioResult):
        default_state_component = placement.tensor_to_component.get("linear_state")
        tp_degree = max(1, placement.parallel.tp_degree)
        for task in result.trace.tasks:
            physical_id = _state_tensor_id(task)
            if physical_id is None:
                continue
            layer_id = str(task.metadata.get("layer_id", "unknown"))
            rank = _integer_or_none(task.metadata.get("rank"))
            component_id = str(default_state_component or "")
            if not component_id:
                event_kind = str(task.metadata.get("event_kind", ""))
                if event_kind in {"linear_state_write", "linear_state_prefetch"}:
                    component_id = str(task.metadata.get("target_component", ""))
                elif event_kind in {"linear_state_read", "linear_state_offload"}:
                    component_id = str(task.metadata.get("source_component", ""))
            if not component_id:
                component_id = str(
                    task.metadata.get("component_id")
                    or task.metadata.get("target_component")
                    or task.metadata.get("source_component")
                    or ""
                )
            length_bytes = _task_transfer_bytes(task)
            if not component_id or length_bytes <= 0:
                continue
            key = (physical_id, component_id, rank)
            current = runtime_state_segments.get(key)
            if current is None or length_bytes > current["length_bytes"]:
                runtime_state_segments[key] = {
                    "logical_id": "linear_state:{}:{}".format(
                        task.request_id, layer_id
                    ),
                    "physical_id": physical_id,
                    "component_id": component_id,
                    "length_bytes": length_bytes,
                    "allocation_kind": "linear_state",
                    "shard_index": (
                        _non_negative_int(task.metadata.get("tp_rank"), rank or 0)
                        % tp_degree
                    ),
                    "shard_count": tp_degree,
                    "rank": rank,
                    "request_id": task.request_id,
                    "layer_id": layer_id,
                    "semantics": "realized_request_state_shard",
                }
        raw_segments.extend(runtime_state_segments.values())
    else:
        peak_state_bytes = max(0, int(result.serving.linear_state_metrics.peak_used_bytes))
        state_component = result.serving.plan.linear_state_policy.cache_component
        if state_component and peak_state_bytes > 0:
            raw_segments.append(
                {
                    "logical_id": "linear_state:runtime_peak",
                    "physical_id": "linear_state:runtime_peak",
                    "component_id": str(state_component),
                    "length_bytes": peak_state_bytes,
                    "allocation_kind": "linear_state",
                    "shard_index": 0,
                    "shard_count": max(1, placement.parallel.tp_degree),
                    "rank": None,
                    "semantics": "aggregate_runtime_peak",
                }
            )

    has_realized_linear_state = bool(runtime_state_segments) or isinstance(
        result, OnlineScenarioResult
    )
    for tensor_id in sorted(placement.tensor_to_component):
        component_id = str(placement.tensor_to_component[tensor_id])
        length_bytes = _non_negative_int(placement.tensor_bytes.get(tensor_id))
        if not component_id or length_bytes <= 0:
            continue
        logical_id = str(aliases.get(tensor_id, tensor_id))
        if logical_id in represented_weight_tensors:
            continue
        if tensor_id == "linear_state" and has_realized_linear_state:
            continue
        raw_segments.append(
            {
                "logical_id": logical_id,
                "physical_id": str(tensor_id),
                "component_id": component_id,
                "length_bytes": length_bytes,
                "allocation_kind": _tensor_kind(str(tensor_id)),
                "shard_index": 0,
                "shard_count": 1,
                "rank": None,
                "semantics": "declared_placement_allocation",
            }
        )

    raw_segments.sort(
        key=lambda item: (
            str(item["component_id"]),
            str(item["allocation_kind"]),
            str(item["logical_id"]),
            _non_negative_int(item.get("shard_index")),
            str(item["physical_id"]),
        )
    )
    offsets: Dict[str, int] = {}
    component_totals: Dict[str, int] = {
        component.component_id: 0 for component in scenario.hardware.components
    }
    lookup: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
    for segment in raw_segments:
        component_id = str(segment["component_id"])
        offset_bytes = offsets.get(component_id, 0)
        segment["offset_bytes"] = offset_bytes
        offsets[component_id] = offset_bytes + int(segment["length_bytes"])
        component_totals[component_id] = offsets[component_id]
        rank = _integer_or_none(segment.get("rank"))
        for tensor_id in (segment["logical_id"], segment["physical_id"]):
            lookup.setdefault((str(tensor_id), rank), segment)
            lookup.setdefault((str(tensor_id), None), segment)

    returned = raw_segments[:segment_limit]
    components: Dict[str, List[Dict[str, Any]]] = {
        component.component_id: [] for component in scenario.hardware.components
    }
    for segment in returned:
        components.setdefault(str(segment["component_id"]), []).append(segment)
    truncated = len(returned) < len(raw_segments)
    layout = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "address_space": "component_local_logical_bytes",
        "physical_addressing": "not_modeled",
        "not_jedec_addressing": True,
        "components": dict(sorted(components.items())),
        "component_totals_bytes": dict(sorted(component_totals.items())),
        "segment_count": len(raw_segments),
        "returned_segment_count": len(returned),
        "segment_limit": segment_limit,
        "truncated": truncated,
    }
    return layout, lookup


def _link_protocols(scenario: ScenarioConfig) -> Dict[str, str]:
    return {
        link.link_id: link.protocol for link in scenario.hardware.links
    }


def _trace_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): to_primitive(value)
        for key, value in metadata.items()
        if not str(key).startswith("_engine_")
    }


def _rank_payload(
    metadata: Mapping[str, Any],
    rank_components: Optional[Mapping[int, Any]] = None,
) -> Dict[str, Any]:
    rank = _integer_or_none(metadata.get("rank"))
    rank_detail = (
        rank_components.get(rank)
        if rank is not None and rank_components is not None
        else None
    )
    component = metadata.get("compute_component_id")
    if component is None and rank_detail is not None:
        component = getattr(rank_detail, "component_id", rank_detail)
    tp_rank = _integer_or_none(metadata.get("tp_rank"))
    pp_rank = _integer_or_none(metadata.get("pp_rank"))
    ep_rank = _integer_or_none(metadata.get("ep_rank"))
    return {
        "rank": rank,
        "tp_rank": (
            tp_rank
            if tp_rank is not None
            else _integer_or_none(getattr(rank_detail, "tp_rank", None))
        ),
        "pp_rank": (
            pp_rank
            if pp_rank is not None
            else _integer_or_none(getattr(rank_detail, "pp_rank", None))
        ),
        "ep_rank": (
            ep_rank
            if ep_rank is not None
            else _integer_or_none(getattr(rank_detail, "ep_rank", None))
        ),
        "component_id": str(component) if component else None,
    }


def _matched_segment(
    lookup: Mapping[Tuple[str, Optional[int]], Dict[str, Any]],
    *,
    logical_id: Optional[str],
    physical_id: Optional[str],
    rank: Optional[int],
) -> Optional[Dict[str, Any]]:
    for tensor_id in (logical_id, physical_id):
        if not tensor_id:
            continue
        for key in ((tensor_id, rank), (tensor_id, None)):
            segment = lookup.get(key)
            if segment is not None:
                return segment
    return None


def _task_tensor_payload(
    task: Any,
    lookup: Mapping[Tuple[str, Optional[int]], Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = task.metadata
    rank = _integer_or_none(metadata.get("rank"))
    state_physical_id = _state_tensor_id(task)
    event_kind = str(metadata.get("event_kind", ""))
    if state_physical_id is not None:
        layer_id = str(metadata.get("layer_id", "unknown"))
        logical_id: Optional[str] = "linear_state:{}:{}".format(
            task.request_id, layer_id
        )
        physical_id: Optional[str] = state_physical_id
    elif event_kind.startswith("kv_"):
        logical_id = "kv_cache"
        physical_id = "kv_cache"
    else:
        logical_id = (
            metadata.get("logical_tensor_id")
            or metadata.get("logical_weight_tensor")
            or metadata.get("tensor")
        )
        physical_id = (
            metadata.get("physical_tensor_id") or metadata.get("tensor")
        )
        logical_id = str(logical_id) if logical_id is not None else None
        physical_id = str(physical_id) if physical_id is not None else None
    segment = _matched_segment(
        lookup,
        logical_id=logical_id,
        physical_id=physical_id,
        rank=rank,
    )
    component_id = (
        str(segment["component_id"])
        if segment is not None
        else metadata.get("weight_source_component")
        or metadata.get("component_id")
        or metadata.get("source_component")
        or metadata.get("target_component")
    )
    length_bytes = _task_transfer_bytes(task)
    if length_bytes <= 0 and segment is not None:
        length_bytes = int(segment["length_bytes"])
    return {
        "logical_id": logical_id,
        "physical_id": physical_id,
        "component_id": str(component_id) if component_id else None,
        "offset_bytes": (
            int(segment["offset_bytes"]) if segment is not None else None
        ),
        "length_bytes": length_bytes,
        "shard_index": (
            _integer_or_none(segment.get("shard_index"))
            if segment is not None
            else None
        ),
        "shard_count": (
            _integer_or_none(segment.get("shard_count"))
            if segment is not None
            else None
        ),
    }


def _task_event(
    task: Any,
    *,
    lookup: Mapping[Tuple[str, Optional[int]], Dict[str, Any]],
    protocols: Mapping[str, str],
    scenario: Optional[ScenarioConfig] = None,
    rank_components: Optional[Mapping[int, Any]] = None,
    batch_id: Optional[str] = None,
    time_offset_ns: float = 0.0,
) -> Dict[str, Any]:
    """Serialize one task using the stable replay transfer contract.

    ``transfer.kind`` is always ``data`` or ``instruction``.  Exact link
    phases retain their serviced hop interval.  Instruction submissions have
    no modeled link service, so their route is topology identity only; when
    the declared endpoints are not routable, the endpoints remain visible,
    ``hops`` is empty, and ``route_status`` is ``unavailable``.  Same-component
    and resource-only tasks are not emitted as cross-component transfers.
    """

    metadata = task.metadata
    raw_resource_directions = metadata.get("resource_directions", {})
    resource_directions = (
        raw_resource_directions
        if isinstance(raw_resource_directions, Mapping)
        else {}
    )
    resources = []
    for interval in task.resource_intervals:
        resource = {
            "resource_id": interval.resource_id,
            "start_ns": interval.start_ns + time_offset_ns,
            "end_ns": interval.end_ns + time_offset_ns,
            "bytes": interval.bytes_moved,
            "energy_pj": interval.energy_pj,
            "interval_semantics": "exact_exclusive_service",
        }
        direction = resource_directions.get(interval.resource_id)
        if direction is not None and str(direction).strip():
            resource["direction"] = str(direction)
        resources.append(resource)
    source = (
        metadata.get("source_component")
        or metadata.get("weight_source_component")
    )
    target = (
        metadata.get("target_component")
        or metadata.get("weight_target_component")
    )
    transfer_kind_raw = str(metadata.get("transfer_kind", "data")).lower()
    transfer_kind = (
        transfer_kind_raw
        if transfer_kind_raw in {"data", "instruction"}
        else "data"
    )
    link_id = metadata.get("link_id")
    hops: List[Dict[str, Any]] = []
    cross_component = bool(source and target and source != target)
    route_hops_raw = metadata.get("route_hops", ())
    route_hops = (
        tuple(route_hops_raw)
        if isinstance(route_hops_raw, Sequence)
        and not isinstance(route_hops_raw, (str, bytes))
        else ()
    )
    link_ids_raw = metadata.get("link_ids", ())
    link_ids = (
        tuple(str(item) for item in link_ids_raw)
        if isinstance(link_ids_raw, Sequence)
        and not isinstance(link_ids_raw, (str, bytes))
        else ()
    )
    if route_hops and cross_component:
        for raw_hop in route_hops:
            if not isinstance(raw_hop, Mapping):
                continue
            route_link_id = str(raw_hop.get("link_id", ""))
            if not route_link_id:
                continue
            route_resource_id = str(
                raw_hop.get("resource_id", "link.{}".format(route_link_id))
            )
            hop_interval = next(
                (
                    interval
                    for interval in task.resource_intervals
                    if interval.resource_id == route_resource_id
                ),
                None,
            )
            hops.append(
                {
                    "link_id": route_link_id,
                    "protocol": raw_hop.get("protocol")
                    or protocols.get(route_link_id),
                    "source_component": raw_hop.get("source_component"),
                    "target_component": raw_hop.get("target_component"),
                    "resource_id": route_resource_id,
                    "start_ns": (
                        hop_interval.start_ns + time_offset_ns
                        if hop_interval is not None
                        else None
                    ),
                    "end_ns": (
                        hop_interval.end_ns + time_offset_ns
                        if hop_interval is not None
                        else None
                    ),
                    "interval_semantics": (
                        "exact_route_hop"
                        if hop_interval is not None
                        else "topology_route_identity_without_hop_service"
                    ),
                }
            )
    elif link_ids and cross_component:
        for route_link_id in link_ids:
            hop_interval = next(
                (
                    interval
                    for interval in task.resource_intervals
                    if interval.resource_id == "link.{}".format(route_link_id)
                    or interval.resource_id.startswith(
                        "link.{}.".format(route_link_id)
                    )
                ),
                None,
            )
            hops.append(
                {
                    "link_id": route_link_id,
                    "protocol": protocols.get(route_link_id),
                    "source_component": (
                        str(source) if len(link_ids) == 1 else None
                    ),
                    "target_component": (
                        str(target) if len(link_ids) == 1 else None
                    ),
                    "resource_id": (
                        hop_interval.resource_id
                        if hop_interval is not None
                        else "link.{}".format(route_link_id)
                    ),
                    "start_ns": (
                        hop_interval.start_ns + time_offset_ns
                        if hop_interval is not None
                        else None
                    ),
                    "end_ns": (
                        hop_interval.end_ns + time_offset_ns
                        if hop_interval is not None
                        else None
                    ),
                    "interval_semantics": (
                        "exact_route_hop"
                        if hop_interval is not None
                        else "topology_route_identity_without_hop_service"
                    ),
                }
            )
    elif link_id is not None and cross_component:
        hop_interval = next(
            (
                interval
                for interval in task.resource_intervals
                if interval.resource_id.startswith("link.")
            ),
            None,
        )
        hops.append(
            {
                "link_id": str(link_id),
                "protocol": protocols.get(str(link_id)),
                "source_component": str(source) if source else None,
                "target_component": str(target) if target else None,
                "resource_id": (
                    hop_interval.resource_id if hop_interval is not None else None
                ),
                "start_ns": (
                    hop_interval.start_ns + time_offset_ns
                    if hop_interval is not None
                    else None
                ),
                "end_ns": (
                    hop_interval.end_ns + time_offset_ns
                    if hop_interval is not None
                    else None
                ),
                "interval_semantics": (
                    "exact_route_hop"
                    if hop_interval is not None
                    else "topology_route_identity_without_hop_service"
                ),
            }
        )
    instruction_route_status: Optional[str] = None
    if transfer_kind == "instruction" and cross_component:
        if hops and all(
            hop["interval_semantics"] == "exact_route_hop" for hop in hops
        ):
            instruction_route_status = "exact_hop_timing"
        elif hops:
            instruction_route_status = "topology_identity_only"
        elif scenario is not None:
            hops = [
                {
                    **hop,
                    "interval_semantics": (
                        "topology_route_identity_without_hop_service"
                    ),
                }
                for hop in _aggregate_route_hops(
                    scenario,
                    str(source),
                    str(target),
                    _task_transfer_bytes(task),
                )
            ]
            instruction_route_status = (
                "topology_identity_only" if hops else "unavailable"
            )
        else:
            instruction_route_status = "unavailable"
    transfer = None
    if cross_component:
        transfer = {
            "kind": transfer_kind,
            "source_component": str(source) if source else None,
            "target_component": str(target) if target else None,
            "bytes": _task_transfer_bytes(task),
            "hops": hops,
        }
        if instruction_route_status is not None:
            transfer["route_status"] = instruction_route_status
    marker = getattr(task, "marker", None)
    marker_value = getattr(marker, "value", None)
    event_kind = metadata.get("event_kind") or marker_value or task.category.value
    phase = metadata.get("phase") or event_kind
    return {
        "event_id": "task:{}".format(task.task_id),
        "task_id": task.task_id,
        "name": task.name,
        "request_id": task.request_id,
        "batch_id": batch_id or metadata.get("batch_id") or metadata.get("cohort_id"),
        "event_kind": str(event_kind),
        "phase": str(phase) if phase is not None else None,
        "operator_id": str(
            metadata.get("operator_id") or metadata.get("op_name") or task.name
        ),
        "layer_id": (
            str(metadata.get("layer_id"))
            if metadata.get("layer_id") is not None
            else None
        ),
        "rank": _rank_payload(metadata, rank_components),
        "start_ns": task.start_ns + time_offset_ns,
        "end_ns": task.end_ns + time_offset_ns,
        "category": task.category.value,
        "marker": marker_value,
        "token_index": task.token_index,
        "tensor": _task_tensor_payload(task, lookup),
        "transfer": transfer,
        "resources": resources,
        "metadata": _trace_metadata(metadata),
    }


def _aggregate_route_hops(
    scenario: ScenarioConfig,
    source: Optional[str],
    target: Optional[str],
    byte_count: int,
) -> List[Dict[str, Any]]:
    if not source or not target or source == target or byte_count < 0:
        return []
    protocols = _link_protocols(scenario)
    try:
        route = _topology_router(scenario).route(
            source,
            target,
            byte_count,
            policy=scenario.placement.parallel.routing_policy,
        )
    except ValueError:
        return []
    return [
        {
            "link_id": hop.link_id,
            "protocol": protocols.get(hop.link_id),
            "source_component": hop.source_component,
            "target_component": hop.target_component,
            "resource_id": hop.resource_id,
            "start_ns": None,
            "end_ns": None,
            "interval_semantics": "aggregate_route_without_hop_timing",
        }
        for hop in route
    ]


def _batch_event(
    result: OnlineScenarioResult,
    batch: Any,
    *,
    lookup: Mapping[Tuple[str, Optional[int]], Dict[str, Any]],
) -> Dict[str, Any]:
    metadata = batch.cost.metadata
    source_raw = metadata.get("source_component")
    target_raw = metadata.get("target_component")
    source = str(source_raw) if source_raw else None
    target = str(target_raw) if target_raw else None
    resource_accounted_bytes = _non_negative_int(
        metadata.get("resource_accounted_bytes")
    )
    explicit_transfer_bytes = _integer_or_none(metadata.get("transfer_bytes"))
    byte_count = (
        max(0, explicit_transfer_bytes)
        if explicit_transfer_bytes is not None
        else 0
    )
    resource_busy_raw = metadata.get("resource_busy_ns", {})
    resource_busy = resource_busy_raw if isinstance(resource_busy_raw, Mapping) else {}
    resources = [
        {
            "resource_id": str(resource_id),
            "start_ns": batch.start_ns,
            "end_ns": batch.end_ns,
            "busy_ns": float(busy_ns),
            "bytes": None,
            "energy_pj": None,
            "interval_semantics": "aggregate_busy_within_batch_envelope",
        }
        for resource_id, busy_ns in sorted(resource_busy.items())
    ]
    if batch.kind.startswith("linear_state_"):
        tensor_id: Optional[str] = "linear_state:runtime_peak"
    elif batch.kind.startswith("kv_"):
        tensor_id = "kv_cache"
    else:
        tensor_id = None
    segment = _matched_segment(
        lookup,
        logical_id=tensor_id,
        physical_id=tensor_id,
        rank=None,
    )
    transfer = None
    if source and target:
        transfer = {
            "kind": "data",
            "source_component": source,
            "target_component": target,
            "bytes": byte_count,
            "hops": _aggregate_route_hops(
                result.scenario, source, target, byte_count
            ),
        }
    selected_items = list(batch.items[:_SCALABLE_BATCH_ITEM_LIMIT])
    selected_request_ids = list(
        batch.request_ids[:_SCALABLE_BATCH_ITEM_LIMIT]
    )
    return {
        "event_id": "batch:{}".format(batch.cohort_id),
        "task_id": None,
        "name": batch.kind,
        "request_id": batch.request_ids[0] if len(batch.request_ids) == 1 else None,
        "request_ids": selected_request_ids,
        "request_ids_truncated": len(selected_request_ids) < len(batch.request_ids),
        "batch_id": batch.cohort_id,
        "event_kind": "batch",
        "phase": batch.kind,
        "operator_id": None,
        "rank": {
            "rank": None,
            "tp_rank": None,
            "pp_rank": None,
            "ep_rank": None,
            "component_id": None,
        },
        "start_ns": batch.start_ns,
        "end_ns": batch.end_ns,
        "category": (
            "communication"
            if batch.kind.startswith(("kv_swap_", "linear_state_swap_"))
            else "batch"
        ),
        "marker": None,
        "token_index": None,
        "tensor": {
            "logical_id": tensor_id,
            "physical_id": tensor_id,
            "component_id": (
                str(segment["component_id"]) if segment is not None else source
            ),
            "offset_bytes": (
                int(segment["offset_bytes"]) if segment is not None else None
            ),
            "length_bytes": byte_count,
            "shard_index": (
                _integer_or_none(segment.get("shard_index"))
                if segment is not None
                else None
            ),
            "shard_count": (
                _integer_or_none(segment.get("shard_count"))
                if segment is not None
                else None
            ),
        },
        "transfer": transfer,
        "resource_accounted_bytes": resource_accounted_bytes,
        "resources": resources,
        "trace_available": True,
        "trace_event_count": _non_negative_int(metadata.get("task_count")),
        "representative_items": [to_primitive(item) for item in selected_items],
        "representative_items_truncated": len(selected_items) < len(batch.items),
        "detail_semantics": "aggregate_batch_with_selected_items",
        "metadata": _trace_metadata(metadata),
    }


def _bounded_batch_history_row(batch: Any) -> Dict[str, Any]:
    row = to_primitive(batch)
    request_ids = list(batch.request_ids[:_SCALABLE_BATCH_ITEM_LIMIT])
    items = list(batch.items[:_SCALABLE_BATCH_ITEM_LIMIT])
    row["request_ids"] = request_ids
    row["items"] = [to_primitive(item) for item in items]
    row["detail_limit"] = _SCALABLE_BATCH_ITEM_LIMIT
    row["request_ids_truncated"] = len(request_ids) < len(batch.request_ids)
    row["items_truncated"] = len(items) < len(batch.items)
    row["detail_semantics"] = "selected_batch_items"
    row["trace_available"] = True
    row["trace_event_count"] = _non_negative_int(
        batch.cost.metadata.get("task_count")
    )
    return row


def _bounded_scheduler_event_row(event: Any) -> Dict[str, Any]:
    row = to_primitive(event)
    details = dict(row.get("details", {}))
    request_ids = details.get("request_ids")
    if isinstance(request_ids, list):
        details["request_ids"] = request_ids[:_SCALABLE_BATCH_ITEM_LIMIT]
        details["request_ids_truncated"] = (
            len(details["request_ids"]) < len(request_ids)
        )
        details["request_id_limit"] = _SCALABLE_BATCH_ITEM_LIMIT
    row["details"] = details
    return row


def _batch_trace_index(result: OnlineScenarioResult) -> Dict[str, Any]:
    batches = result.serving.batches[:_SCALABLE_BATCH_HISTORY_LIMIT]
    rows = [
        {
            "batch_id": batch.cohort_id,
            "kind": batch.kind,
            "start_ns": batch.start_ns,
            "end_ns": batch.end_ns,
            "trace_available": True,
            "event_count": _non_negative_int(
                batch.cost.metadata.get("task_count")
            ),
        }
        for batch in batches
    ]
    return {
        "available": True,
        "granularity": "task",
        "trace_source": "exact_cohort_replay",
        "fidelity": TraceFidelity.REPRESENTATIVE.value,
        "total": len(result.serving.batches),
        "returned": len(rows),
        "limit": _SCALABLE_BATCH_HISTORY_LIMIT,
        "truncated": len(rows) < len(result.serving.batches),
        "batches": rows,
    }


def _arrival_process_semantics(scenario: ScenarioConfig) -> Dict[str, Any]:
    """Describe how request arrival timestamps in a report were produced."""

    workload = scenario.workload
    if workload.requests:
        return {
            "kind": "explicit_requests",
            "source": "workload.requests[*].arrival_ns",
            "deterministic": True,
            "arrival_rate_rps": float(workload.arrival_rate_rps),
            "interval_ns": None,
            "formula": None,
            "random_seed_affects_arrivals": False,
            "random_seed_effect": "does_not_affect_explicit_arrival_timestamps",
        }
    rate = float(workload.arrival_rate_rps)
    if rate > 0.0:
        interval_ns = 1_000_000_000.0 / rate
        return {
            "kind": "deterministic_fixed_interval",
            "source": "synthetic_workload",
            "deterministic": True,
            "arrival_rate_rps": rate,
            "interval_ns": interval_ns,
            "formula": "interval_ns = 1e9 / arrival_rate_rps",
            "random_seed_affects_arrivals": False,
            "random_seed_effect": "does_not_affect_deterministic_arrivals",
        }
    return {
        "kind": "simultaneous_synthetic",
        "source": "synthetic_workload",
        "deterministic": True,
        "arrival_rate_rps": rate,
        "interval_ns": 0.0,
        "formula": "all synthetic arrivals use arrival_ns=0 when arrival_rate_rps=0",
        "random_seed_affects_arrivals": False,
        "random_seed_effect": "does_not_affect_deterministic_arrivals",
    }


def _prefetch_semantics(scenario: ScenarioConfig) -> Dict[str, Any]:
    distance = int(scenario.placement.kv_policy.prefetch_distance)
    contract = "must_be_zero"
    description = (
        "V4 placement requires prefetch_distance=0; proactive KV lookahead "
        "is not modeled"
    )
    return {
        "value": distance,
        "modeled": False,
        "contract": contract,
        "description": description,
    }


def _sequence_length_semantics(scenario: ScenarioConfig) -> Dict[str, Any]:
    return {
        "max_sequence_length": int(
            getattr(scenario.model, "max_sequence_length", 0) or 0
        ),
        "request_validation_rule": (
            "prompt_tokens + output_tokens <= max_sequence_length"
        ),
        "live_kv_rule": "prompt_tokens + max(output_tokens - 1, 0)",
        "live_kv_is_not_request_validation_rule": True,
    }


def _deadline_semantics(
    scenario: ScenarioConfig,
    *,
    online: bool,
) -> Dict[str, Any]:
    """Expose whether deadline metadata affects scheduling or only reporting."""

    configured = any(
        getattr(request, "deadline_ns", None) is not None
        for request in scenario.workload.requests
    )
    return {
        "configured": configured,
        "scheduler_target": bool(online and configured),
        "used_for_ordering": bool(online and configured),
        "used_for_completion_evaluation": bool(online and configured),
        "completion_rule": (
            "finished request meets deadline when finish_ns <= deadline_ns"
            if online and configured
            else None
        ),
        "static_behavior": (
            "deadline_ns is not consumed by static scheduling"
            if not online
            else None
        ),
    }


def _slo_semantics(
    scenario: ScenarioConfig,
    *,
    online: bool,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Expose SLO thresholds as evaluation filters, never scheduler targets."""

    policy = scheduler
    if policy is None:
        policy = getattr(scenario.workload, "scheduler", None)
    ttft = getattr(policy, "slo_ttft_ns", None) if policy is not None else None
    tbt = getattr(policy, "slo_tbt_ns", None) if policy is not None else None
    configured = ttft is not None or tbt is not None
    return {
        "configured": configured,
        "scheduler_target": False,
        "role": "evaluation_filter_only",
        "slo_ttft_ns": float(ttft) if ttft is not None else None,
        "slo_tbt_ns": float(tbt) if tbt is not None else None,
        "applied_to_goodput": bool(online),
        "goodput_filter": (
            "completed requests satisfying deadline, TTFT, and TBT filters"
            if online
            else "not computed by static report"
        ),
        "goodput_denominator_window": "active" if online else None,
    }


def _trace_semantics(result: RunResult) -> Dict[str, Any]:
    """Describe complete aggregates separately from retained event rows."""

    if isinstance(result, OnlineScenarioResult):
        return {
            "fidelity": TraceFidelity.AGGREGATE.value,
            "aggregate_metrics": "exact_complete_online_batch_aggregation",
            "aggregate_metrics_complete": True,
            "task_trace": "batch_envelope_and_scheduler_events",
            "task_trace_complete": False,
            "task_trace_source": "batch_level_data",
            "cohort_replay": {
                "available": True,
                "granularity": "task",
                "scope": "one_realized_batch_at_a_time",
                "lazy": True,
            },
        }

    execution = result.execution
    retention_policy = execution.retention_policy
    if retention_policy is RetentionPolicy.EXACT:
        task_trace = "complete_task_trace"
        task_trace_complete = True
        task_trace_source = "unified_event_kernel"
        fidelity = TraceFidelity.EXACT
    elif retention_policy is RetentionPolicy.STREAMING:
        task_trace = "bounded_retained_tasks"
        task_trace_complete = False
        task_trace_source = "unified_event_kernel_head_tail"
        fidelity = TraceFidelity.REPRESENTATIVE
    else:
        task_trace = "not_retained"
        task_trace_complete = False
        task_trace_source = "unified_event_kernel_aggregates"
        fidelity = TraceFidelity.AGGREGATE
    return {
        "fidelity": fidelity.value,
        "aggregate_metrics": "exact_complete_static_aggregate",
        "aggregate_metrics_complete": True,
        "task_trace": task_trace,
        "task_trace_complete": task_trace_complete,
        "task_trace_source": task_trace_source,
        "retained_task_limit": execution.retained_task_limit,
        "total_task_count": int(execution.task_count),
        "retained_task_count": len(result.trace.tasks),
    }


def _static_result_semantics(result: ScenarioResult) -> Dict[str, Any]:
    makespan = float(result.trace.makespan_ns)
    return {
        "execution_mode": "static",
        "arrival_process": _arrival_process_semantics(result.scenario),
        "kv_prefetch": _prefetch_semantics(result.scenario),
        "sequence_length": _sequence_length_semantics(result.scenario),
        "slo": _slo_semantics(result.scenario, online=False),
        "deadline": _deadline_semantics(result.scenario, online=False),
        "trace": _trace_semantics(result),
        "throughput": {
            "primary_window": "makespan",
            "primary_window_ns": [0.0, makespan],
            "primary_denominator": "makespan_ns",
            "primary_denominator_ns": makespan,
            "window_semantics": "static throughput uses the complete makespan window",
        },
    }


def _online_result_semantics(
    result: OnlineScenarioResult,
    *,
    windows_ns: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    makespan = float(result.serving.makespan_ns)
    return {
        "execution_mode": "continuous_batching",
        "arrival_process": _arrival_process_semantics(result.scenario),
        "kv_prefetch": _prefetch_semantics(result.scenario),
        "sequence_length": _sequence_length_semantics(result.scenario),
        "slo": _slo_semantics(
            result.scenario, online=True, scheduler=result.serving.plan.scheduler
        ),
        "deadline": _deadline_semantics(result.scenario, online=True),
        "trace": _trace_semantics(result),
        "throughput": {
            "primary_window": "wall_clock",
            "primary_window_ns": list(windows_ns["wall_clock"]),
            "primary_denominator": "wall_clock_duration_ns",
            "primary_denominator_ns": makespan,
            "windows_ns": {
                name: list(bounds) for name, bounds in windows_ns.items()
            },
            "rate_denominators": {
                "requests_per_s": "wall_clock",
                "visible_output_tokens_per_s": "wall_clock",
                "active_requests_per_s": "active",
                "active_visible_tokens_per_s": "active",
                "steady_requests_per_s": "steady",
                "steady_visible_tokens_per_s": "steady",
                "goodput.requests_per_s": "active",
                "goodput.visible_output_tokens_per_s": "active",
            },
        },
    }


def _visualization_payload(
    result: RunResult,
    options: VisualizationTraceOptions,
) -> Dict[str, Any]:
    memory_layout, lookup = _memory_layout(
        result,
        segment_limit=options.memory_segment_limit,
    )
    limitations = [
        "内存偏移是组件本地的确定性逻辑字节区间；未建模 JEDEC bank/row/column 物理寻址。"
    ]
    start = options.event_offset
    representative_truncated = False
    logical_event_total: Optional[int] = None
    if isinstance(result, OnlineScenarioResult):
        total_events = len(result.serving.batches)
        selected = result.serving.batches[start : start + options.event_limit]
        events = [
            _batch_event(result, batch, lookup=lookup) for batch in selected
        ]
        fidelity = TraceFidelity.AGGREGATE
        granularity = "batch"
        trace_source = "online_batch_envelope"
        makespan_ns = float(result.serving.makespan_ns)
        limitations.extend(
            [
                "可扩展在线推理仅公开已实现的批次包络和聚合资源计数，无法据此反推算子执行区间。",
                "每个批次最多序列化 {} 个代表性条目。".format(
                    _SCALABLE_BATCH_ITEM_LIMIT
                ),
                "聚合路由 hop 保留拓扑、链路和协议标识，但不声称逐 hop 的开始/结束时间。",
                "KV read 仅表示外部持久化 KV Cache 读取；prefill/recompute 的历史 prompt 注意力仍计入 causal pairs、算力及普通 attention/activation 流量。",
            ]
        )
        batch_trace_index: Optional[Dict[str, Any]] = _batch_trace_index(result)
    elif isinstance(result, ScenarioResult):
        total_events = len(result.trace.tasks)
        logical_event_total = result.execution.task_count
        selected = result.trace.tasks[start : start + options.event_limit]
        protocols = _link_protocols(result.scenario)
        events = [
            _task_event(
                task,
                lookup=lookup,
                protocols=protocols,
                scenario=result.scenario,
            )
            for task in selected
        ]
        fidelity = {
            RetentionPolicy.EXACT: TraceFidelity.EXACT,
            RetentionPolicy.STREAMING: TraceFidelity.REPRESENTATIVE,
            RetentionPolicy.AGGREGATE: TraceFidelity.AGGREGATE,
        }[result.execution.retention_policy]
        granularity = "task"
        trace_source = {
            RetentionPolicy.EXACT: "unified_event_kernel",
            RetentionPolicy.STREAMING: "unified_event_kernel_head_tail",
            RetentionPolicy.AGGREGATE: "unified_event_kernel_aggregates",
        }[result.execution.retention_policy]
        makespan_ns = float(result.trace.makespan_ns)
        representative_truncated = logical_event_total > total_events
        if result.execution.retention_policy is not RetentionPolicy.EXACT:
            limitations.extend(
                [
                    "保留策略未保存完整任务轨迹；未保留的任务不支持分页回放。",
                    "汇总时延、吞吐、关键路径、资源利用率、能耗和流量来自完整执行。",
                ]
            )
        batch_trace_index = None

    returned = len(events)
    has_more = start + returned < total_events
    event_page_truncated = start > 0 or has_more
    if event_page_truncated:
        limitations.append(
            "事件数组采用分页返回；当前页中的事件保留其声明的建模精度。"
        )
    if memory_layout["truncated"]:
        limitations.append(
            "逻辑内存分段列表已达到响应数量上限。"
        )
    payload = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "fidelity": fidelity.value,
        "granularity": granularity,
        "trace_source": trace_source,
        "truncated": (
            representative_truncated
            or event_page_truncated
            or memory_layout["truncated"]
        ),
        "limitations": limitations,
        "time_range_ns": [0.0, makespan_ns],
        "events": events,
        "memory_layout": memory_layout,
        "pagination": {
            "offset": start,
            "limit": options.event_limit,
            "returned": returned,
            "total": total_events,
            "has_more": has_more,
            "next_offset": start + returned if has_more else None,
        },
    }
    if logical_event_total is not None:
        payload["logical_event_total"] = logical_event_total
        payload["retained_event_total"] = total_events
    if batch_trace_index is not None:
        payload["batch_trace_index"] = batch_trace_index
    return payload


def replay_online_batch_trace(
    result: OnlineScenarioResult,
    batch_id: str,
) -> Dict[str, Any]:
    """Materialize all exact planner events for one realized online batch."""

    if not isinstance(result, OnlineScenarioResult):
        raise TypeError("result 必须是在线连续批处理结果")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id 必须是非空字符串")
    batch = next(
        (
            candidate
            for candidate in result.serving.batches
            if candidate.cohort_id == batch_id
        ),
        None,
    )
    if batch is None:
        raise KeyError(batch_id)

    host_duration = max(
        0.0,
        float(batch.cost.metadata.get("host_orchestration_ns", 0.0)),
    )
    host_start = float(
        batch.cost.metadata.get("host_start_ns", batch.start_ns)
    )
    host_end = float(
        batch.cost.metadata.get("host_end_ns", host_start + host_duration)
    )
    device_start = float(
        batch.cost.metadata.get("device_start_ns", batch.start_ns)
    )
    device_end = float(
        batch.cost.metadata.get("device_end_ns", batch.end_ns)
    )
    scheduled_gap = max(0.0, device_start - host_end)
    scope_start = host_start if host_duration > 0.0 else device_start

    cohort = BatchCohort(
        cohort_id=batch.cohort_id,
        kind=batch.kind,
        start_ns=batch.start_ns,
        items=batch.items,
        proposal_cost_scale=batch.proposal_cost_scale,
        metadata=batch.metadata,
    )
    trace = simulate_schedule(
        compile_serving_cohort_schedule(result.scenario, cohort)
    )
    _layout, lookup = _memory_layout(
        result,
        segment_limit=DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT,
    )
    protocols = _link_protocols(result.scenario)
    rank_components = {
        rank.rank: rank
        for rank in _parallel_plan(result.scenario).ranks
    }
    events = []
    for task in trace.tasks:
        is_host_prefix = (
            task.metadata.get("orchestration_stage") == "host_prefix"
        )
        if host_duration <= 0.0:
            time_offset = device_start
        elif is_host_prefix:
            time_offset = host_start
        else:
            # The isolated cohort graph is host->device sequential.  During
            # online execution the host prefix may overlap the previous GPU
            # batch, so anchor device work at its realized device start.
            time_offset = device_start - host_duration
        events.append(
            _task_event(
                task,
                lookup=lookup,
                protocols=protocols,
                scenario=result.scenario,
                rank_components=rank_components,
                batch_id=batch.cohort_id,
                time_offset_ns=time_offset,
            )
        )
    limitations = [
        "这是代表性批次 lowering 后任务图的精确回放；不会重新执行调度准入，也不会改变聚合运行指标。",
        "长短不齐的请求上下文继续沿用在线 lowerer 的加权平均批次语义。",
        "KV read 仅表示外部持久化 KV Cache 读取；prefill/recompute 的历史 prompt 注意力仍保留在 causal pairs、算力及普通 attention/activation 流量中。",
        "内存区间是组件本地的确定性逻辑字节区间，不是 JEDEC 物理地址。",
        "transfer.kind 固定为 data 或 instruction；instruction 的拓扑 hop 仅表示路由身份，不声称链路服务时间，无法路由时保留端点并返回空 hops。",
        "仅当后台任务仍保留在管理器的有界完成历史中时，才能读取回放。",
    ]
    replay_duration = float(trace.makespan_ns)
    model_duration = float(batch.cost.duration_ns)
    realized_duration = max(0.0, device_end - scope_start)
    if not math.isclose(
        replay_duration,
        model_duration,
        rel_tol=2.0e-10,
        abs_tol=1.0e-6,
    ):
        limitations.append(
            "代表性任务图与运行时采用的批次成本不完全一致；回放事件仅用于解释该批次。"
        )
    return {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "granularity": "task",
        "trace_source": "exact_cohort_replay",
        "fidelity": TraceFidelity.REPRESENTATIVE.value,
        "scope": {
            "batch_id": batch.cohort_id,
            "kind": batch.kind,
            "start_ns": scope_start,
            "end_ns": device_end,
            "host_start_ns": host_start,
            "host_end_ns": host_end,
            "device_start_ns": device_start,
            "device_end_ns": device_end,
            "scheduled_gap_ns": scheduled_gap,
            "realized_duration_ns": realized_duration,
            "replay_duration_ns": replay_duration,
            "model_duration_ns": model_duration,
        },
        "events": events,
        "limitations": limitations,
    }


def page_online_batch_trace(
    replay: Mapping[str, Any],
    *,
    offset: int = 0,
    limit: int = 5_000,
) -> Dict[str, Any]:
    """Slice a previously materialized replay without recompiling its batch."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset 必须是非负整数")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > 5_000
    ):
        raise ValueError("limit 必须是 1 到 5000 之间的整数")
    raw_events = replay.get("events", ())
    events = list(raw_events) if isinstance(raw_events, (list, tuple)) else []
    total = len(events)
    selected = events[offset : offset + limit]
    returned = len(selected)
    has_more = offset + returned < total
    limitations_raw = replay.get("limitations", ())
    limitations = (
        list(limitations_raw)
        if isinstance(limitations_raw, (list, tuple))
        else []
    )
    if offset > 0 or has_more:
        limitations.append("任务事件数组采用分页返回。")
    payload = {
        str(key): value
        for key, value in replay.items()
        if key not in {"events", "pagination", "limitations"}
    }
    payload.update(
        {
            "events": selected,
            "pagination": {
                "offset": offset,
                "limit": limit,
                "total": total,
                "returned": returned,
                "has_more": has_more,
                "next_offset": offset + returned if has_more else None,
            },
            "limitations": limitations,
        }
    )
    return payload


def online_batch_trace_page(
    result: OnlineScenarioResult,
    batch_id: str,
    *,
    offset: int = 0,
    limit: int = 5_000,
) -> Dict[str, Any]:
    """Replay and page the exact planner tasks for one realized online batch."""

    return page_online_batch_trace(
        replay_online_batch_trace(result, batch_id),
        offset=offset,
        limit=limit,
    )


_COMPONENT_METRIC_LABELS_CN = {
    "busy_fraction": "组件忙碌占比",
    "modeled_compute_utilization": "建模计算利用率",
    "weight_residency_bytes": "权重驻留量",
    "kv_cache_residency_bytes": "KV 缓存驻留量",
    "linear_state_residency_bytes": "线性状态驻留量",
    "activation_residency_bytes": "激活驻留量",
    "temporary_residency_bytes": "临时缓冲驻留量",
    "memory_read_bandwidth_utilization": "内存读取带宽利用率",
    "memory_write_bandwidth_utilization": "内存写入带宽利用率",
    "storage_occupancy_bytes": "存储占用量",
    "storage_io_utilization": "存储 I/O 活跃度",
    "storage_read_bandwidth_utilization": "存储介质读取带宽利用率",
    "storage_write_bandwidth_utilization": "存储介质写入带宽利用率",
    "dma_engine_utilization": "DMA 引擎利用率",
    "fabric_bandwidth_utilization": "互连结构带宽利用率",
    "link_bandwidth_utilization": "链路带宽利用率",
}


def _finite_non_negative(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return result if math.isfinite(result) and result > 0.0 else 0.0


def _capacity_payload(
    value: Any, unit: str, source: str
) -> Optional[Dict[str, Any]]:
    finite = _finite_non_negative(value)
    if finite <= 0.0:
        return None
    return {"value": finite, "unit": unit, "source": source}


def _sweep_change_points(
    samples: Iterable[Tuple[float, float, float]],
    makespan_ns: float,
    *,
    clamp_ratio: bool,
) -> Tuple[ChangePointInterval, ...]:
    """Turn additive interval samples into contiguous change-point rows."""

    end_of_run = _finite_non_negative(makespan_ns)
    if end_of_run <= 0.0:
        return ()
    deltas: Dict[float, float] = {0.0: 0.0, end_of_run: 0.0}
    for raw_start, raw_end, raw_value in samples:
        start = _finite_non_negative(raw_start)
        end = _finite_non_negative(raw_end)
        value = _finite_non_negative(raw_value)
        start = min(start, end_of_run)
        end = min(end, end_of_run)
        if end <= start or value <= 0.0:
            continue
        deltas[start] = deltas.get(start, 0.0) + value
        deltas[end] = deltas.get(end, 0.0) - value
    points: List[ChangePointInterval] = []
    level = 0.0
    previous = 0.0
    for timestamp in sorted(deltas):
        if timestamp > previous:
            value = min(1.0, max(0.0, level)) if clamp_ratio else max(0.0, level)
            points.append(ChangePointInterval(previous, timestamp, value))
        level += deltas[timestamp]
        if abs(level) < 1.0e-12:
            level = 0.0
        previous = timestamp
    return tuple(points)


def _series_payload(
    *,
    series_id: str,
    metric: str,
    unit: str,
    intervals: Iterable[ChangePointInterval] = (),
    fidelity: TraceFidelity,
    quality: SeriesQuality,
    capacity: Optional[Dict[str, Any]] = None,
    allocation_kind: Optional[str] = None,
    rank: Optional[Dict[str, Any]] = None,
    channel_id: Optional[str] = None,
    scope: str = "component",
    note_cn: Optional[str] = None,
    clamp_ratio: bool = False,
) -> Dict[str, Any]:
    bounded = bound_change_point_intervals(
        intervals,
        point_limit=DEFAULT_COMPONENT_TIMESERIES_POINT_LIMIT,
        clamp_ratio=clamp_ratio,
    )
    payload: Dict[str, Any] = {
        "series_id": series_id,
        "label_cn": _COMPONENT_METRIC_LABELS_CN[metric],
        "metric": metric,
        "unit": unit,
        "scope": scope,
        "rank": rank,
        "channel_id": channel_id,
        "allocation_kind": allocation_kind,
        "fidelity": fidelity.value,
        "quality": quality.value,
        "capacity": capacity,
        "points": [
            {
                "start_ns": point.start_ns,
                "end_ns": point.end_ns,
                "value": point.value,
            }
            for point in bounded.points
        ],
        "raw_point_count": bounded.raw_point_count,
        "point_count": bounded.point_count,
        "point_limit": bounded.point_limit,
        "merged": bounded.merged,
    }
    if note_cn:
        payload["note_cn"] = note_cn
    return payload


def _constant_intervals(
    makespan_ns: float, value: Any, *, include_zero: bool = False
) -> Tuple[ChangePointInterval, ...]:
    end = _finite_non_negative(makespan_ns)
    finite = _finite_non_negative(value)
    if end <= 0.0 or (finite <= 0.0 and not include_zero):
        return ()
    return (ChangePointInterval(0.0, end, finite),)


def _rank_drilldown(scenario: ScenarioConfig) -> Dict[str, List[Dict[str, int]]]:
    rows: Dict[str, List[Dict[str, int]]] = {
        component.component_id: [] for component in scenario.hardware.components
    }
    try:
        ranks = _parallel_plan(scenario).ranks
    except (KeyError, TypeError, ValueError):
        ranks = ()
    for rank in ranks:
        detail = {
            "rank": int(rank.rank),
            "tp_rank": int(rank.tp_rank),
            "pp_rank": int(rank.pp_rank),
            "ep_rank": int(rank.ep_rank),
        }
        owners = {
            str(owner)
            for owner in (
                rank.component_id,
                rank.memory_component_id,
                rank.cim_component_id,
            )
            if owner
        }
        for component_id in owners:
            rows.setdefault(component_id, []).append(dict(detail))
    for component_id in rows:
        rows[component_id].sort(
            key=lambda item: (
                item["rank"], item["tp_rank"], item["pp_rank"], item["ep_rank"]
            )
        )
    return rows


def _rank_owner_maps(scenario: ScenarioConfig) -> Dict[int, Dict[str, Optional[str]]]:
    try:
        ranks = _parallel_plan(scenario).ranks
    except (KeyError, TypeError, ValueError):
        ranks = ()
    return {
        int(rank.rank): {
            "compute": str(rank.component_id),
            "memory": (
                str(rank.memory_component_id)
                if rank.memory_component_id
                else None
            ),
            "cim": str(rank.cim_component_id) if rank.cim_component_id else None,
        }
        for rank in ranks
    }


def _component_capacities(
    scenario: ScenarioConfig, component: Any
) -> Dict[str, float]:
    kind = component.normalized_kind
    memory_bytes = _finite_non_negative(component.capacity_bytes)
    peak_ops = _finite_non_negative(component.peak_ops_per_s)
    read_bandwidth = _finite_non_negative(component.read_bandwidth_gbps) * 1.0e9 / 8.0
    write_bandwidth = _finite_non_negative(component.write_bandwidth_gbps) * 1.0e9 / 8.0
    dma_bandwidth = (
        _finite_non_negative(component.metadata.get("dma_bandwidth_gbps", 0.0))
        * 1.0e9
        / 8.0
    )

    profile_kind = normalize_cost_profile_kind(kind)
    if kind == "cpu":
        cpu_profile = scenario.resolve_component_profile(component, CPUProfile)
        efficiency = _finite_non_negative(
            cpu_profile.attainable_efficiency
        )
        class_rates = (
            _finite_non_negative(cpu_profile.gemm_gops),
            _finite_non_negative(cpu_profile.elementwise_gops),
            _finite_non_negative(cpu_profile.reduction_gops),
        )
        profile_peak = max(class_rates, default=0.0) * efficiency * 1.0e9
        if peak_ops <= 0.0:
            peak_ops = profile_peak
        attached_bandwidth = _attached_host_memory_bandwidth_bytes_per_s(
            scenario, component
        )
        if read_bandwidth <= 0.0:
            read_bandwidth = attached_bandwidth
        if write_bandwidth <= 0.0:
            write_bandwidth = attached_bandwidth

    if profile_kind in {"hbm", "host_memory"}:
        if profile_kind == "hbm":
            memory_profile = scenario.resolve_component_profile(
                component, HBMProfile
            )
        else:
            memory_profile = scenario.resolve_component_profile(
                component, HostMemoryProfile
            )
        profile_bandwidth = _finite_non_negative(
            memory_profile.effective_bandwidth_gb_s
        )
        profile_bytes_s = profile_bandwidth * 1.0e9
        if read_bandwidth <= 0.0:
            read_bandwidth = profile_bytes_s
        if write_bandwidth <= 0.0:
            write_bandwidth = profile_bytes_s

    if profile_kind == "cim":
        profile = scenario.resolve_component_profile(
            component, DigitalSramCimProfile
        )
        if peak_ops <= 0.0:
            peak_ops = (
                2.0
                * _finite_non_negative(profile.array_count)
                * _finite_non_negative(profile.p_m)
                * _finite_non_negative(profile.p_k)
                * _finite_non_negative(profile.p_n)
                * _finite_non_negative(profile.frequency_ghz)
                * 1.0e9
                / max(1.0, _finite_non_negative(profile.cycles_per_eval))
            )
        cim_bandwidth = max(
            _finite_non_negative(profile.load_bandwidth_gb_s),
            _finite_non_negative(profile.activation_bandwidth_gb_s),
            _finite_non_negative(profile.output_bandwidth_gb_s),
            _finite_non_negative(profile.noc_bandwidth_gb_s),
        ) * 1.0e9
        if read_bandwidth <= 0.0:
            read_bandwidth = cim_bandwidth
        if write_bandwidth <= 0.0:
            write_bandwidth = cim_bandwidth
    return {
        "memory_bytes": memory_bytes,
        "peak_ops_per_s": peak_ops,
        "read_bandwidth_bytes_per_s": read_bandwidth,
        "write_bandwidth_bytes_per_s": write_bandwidth,
        "dma_bandwidth_bytes_per_s": dma_bandwidth,
    }


def _attached_host_memory_bandwidth_bytes_per_s(
    scenario: ScenarioConfig, cpu_component: Any
) -> float:
    """Aggregate only the host-memory components physically attached to a CPU."""

    components = scenario.hardware.component_map()
    attached = set()
    total = 0.0
    for link in scenario.hardware.links:
        if link.source_component == cpu_component.component_id:
            memory_id = link.target_component
        elif link.target_component == cpu_component.component_id:
            memory_id = link.source_component
        else:
            continue
        if memory_id in attached:
            continue
        memory = components.get(memory_id)
        if (
            memory is None
            or normalize_cost_profile_kind(memory.normalized_kind)
            != "host_memory"
        ):
            continue
        profile = scenario.resolve_component_profile(
            memory, HostMemoryProfile
        )
        total += _finite_non_negative(profile.effective_bandwidth_gb_s) * 1.0e9
        attached.add(memory_id)
    return total


def _task_operator_class(task: Any) -> str:
    direct = str(task.metadata.get("operator_class", "")).strip().lower()
    if direct:
        return direct
    cost_model = task.metadata.get("cost_model", {})
    if isinstance(cost_model, Mapping):
        return str(cost_model.get("operator_class", "")).strip().lower()
    return ""


def _operator_peak_ops_per_s(
    scenario: ScenarioConfig,
    component: Any,
    task: Any,
    fallback: float,
) -> float:
    """Return the throughput contract for this exact operator primitive."""

    operator_class = _task_operator_class(task)
    kind = component.normalized_kind
    if kind == "cpu":
        profile = scenario.resolve_component_profile(component, CPUProfile)
        field = {
            "gemm": "gemm_gops",
            "elementwise": "elementwise_gops",
            "reduction": "reduction_gops",
        }.get(operator_class)
        if field:
            return (
                _finite_non_negative(getattr(profile, field, 0.0))
                * _finite_non_negative(profile.attainable_efficiency)
                * 1.0e9
            )
    if kind == "gpu":
        profile = scenario.resolve_component_profile(component, GPUProfile)
        if operator_class == "gemm":
            return (
                _finite_non_negative(getattr(profile, "attainable_tops", 0.0))
                * 1.0e12
            )
        field = {
            "elementwise": "elementwise_gops",
            "reduction": "reduction_gops",
        }.get(operator_class)
        if field:
            return _finite_non_negative(getattr(profile, field, 0.0)) * 1.0e9
    return _finite_non_negative(fallback)


def _is_dma_resource(resource_id: str, task: Optional[Any] = None) -> bool:
    normalized = str(resource_id).lower()
    if normalized.endswith(".dma") or ".dma_" in normalized:
        return True
    if task is None:
        return False
    return str(task.metadata.get("event_kind", "")).lower() == "dma"


def _resource_link_id(scenario: ScenarioConfig, resource_id: str) -> Optional[str]:
    for link in sorted(scenario.hardware.links, key=lambda item: (-len(item.link_id), item.link_id)):
        prefix = "link.{}".format(link.link_id)
        if resource_id == prefix or resource_id.startswith(prefix + "."):
            return link.link_id
    return None


def _resource_component_id(
    scenario: ScenarioConfig,
    resource_id: str,
    metadata: Mapping[str, Any],
    rank_owners: Mapping[int, Mapping[str, Optional[str]]],
) -> Optional[str]:
    if _resource_link_id(scenario, resource_id) is not None:
        return None
    rank = _integer_or_none(metadata.get("rank"))
    owner = rank_owners.get(rank, {}) if rank is not None else {}
    memory_owner = owner.get("memory")
    if memory_owner:
        memory_component = scenario.hardware.component_map().get(memory_owner)
        if (
            memory_component is not None
            and resource_id
            == _component_memory_resource_id(scenario, memory_component)
        ):
            return str(memory_owner)
    unique_memory = sorted(
        {
            str(row["memory"])
            for row in rank_owners.values()
            if row.get("memory")
        }
    )
    matching_memory = [
        component_id
        for component_id in unique_memory
        if resource_id
        == _component_memory_resource_id(
            scenario, scenario.hardware.get_component(component_id)
        )
    ]
    if len(matching_memory) == 1:
        return matching_memory[0]
    if len(matching_memory) > 1:
        # A shared resource_id does not prove which physical memory owned an
        # aggregate interval.  Leave it unassigned instead of attributing the
        # load to the first rank memory or to a component-id prefix.
        return None
    component_ids = sorted(
        (component.component_id for component in scenario.hardware.components),
        key=lambda value: (-len(value), value),
    )
    for component_id in component_ids:
        if (
            resource_id == component_id
            or resource_id.startswith(component_id + ".")
            or resource_id.startswith("component.{}.".format(component_id))
        ):
            return component_id
    for key in (
        "component_id",
        "target_component",
        "compute_component_id",
        "source_component",
    ):
        component_id = metadata.get(key)
        if component_id in scenario.hardware.component_map():
            return str(component_id)
    return None


def _component_memory_resource_id(
    scenario: ScenarioConfig, component: Any
) -> Optional[str]:
    profile_kind = normalize_cost_profile_kind(component.normalized_kind)
    if profile_kind == "hbm":
        profile = scenario.resolve_component_profile(component, HBMProfile)
    elif profile_kind == "host_memory":
        profile = scenario.resolve_component_profile(
            component, HostMemoryProfile
        )
    else:
        return None
    return str(profile.resource_id)


def _is_compute_resource(resource_id: str) -> bool:
    normalized = resource_id.lower()
    return any(
        marker in normalized
        for marker in (".compute", ".array", ".accumulator", ".peripheral")
    )


def _memory_direction(task: Any, resource_id: str) -> str:
    resource_directions = task.metadata.get("resource_directions", {})
    if isinstance(resource_directions, Mapping):
        resource_direction = str(
            resource_directions.get(resource_id, "")
        ).lower()
        if resource_direction in {"read", "write"}:
            return resource_direction
    explicit = str(task.metadata.get("memory_direction", "")).lower()
    if explicit in {"read", "write"}:
        return explicit
    event_kind = str(task.metadata.get("event_kind", "")).lower()
    if event_kind in {
        "kv_append",
        "kv_swap_in",
        "linear_state_write",
        "linear_state_swap_in",
    }:
        return "write"
    if event_kind in {
        "kv_read",
        "kv_swap_out",
        "linear_state_read",
        "linear_state_swap_out",
    }:
        return "read"
    text = " ".join(
        str(value).lower()
        for value in (
            resource_id,
            task.metadata.get("event_kind", ""),
            task.metadata.get("phase", ""),
            task.name,
        )
    )
    return "write" if any(token in text for token in ("write", "store", "output")) else "read"


def _finite_non_negative_bytes_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _metadata_write_bytes(task: Any, total_bytes: float) -> Optional[float]:
    metadata = task.metadata
    if "modeled_memory_write_bytes" in metadata:
        return min(
            total_bytes,
            _finite_non_negative(metadata.get("modeled_memory_write_bytes")),
        )

    output_bytes = _finite_non_negative_bytes_or_none(
        metadata.get("output_bytes")
    )
    if output_bytes is not None:
        return min(total_bytes, output_bytes)

    cost_model = metadata.get("cost_model", {})
    if isinstance(cost_model, Mapping):
        write_bytes = _finite_non_negative_bytes_or_none(
            cost_model.get("write_bytes")
        )
    else:
        write_bytes = _finite_non_negative_bytes_or_none(
            getattr(cost_model, "write_bytes", None)
        )
    if write_bytes is None:
        return None
    return min(total_bytes, write_bytes)


def _memory_direction_rows(
    task: Any, resource_id: str, total_bytes: float
) -> List[Tuple[str, float]]:
    resource_directions = task.metadata.get("resource_directions", {})
    if isinstance(resource_directions, Mapping):
        resource_direction = str(
            resource_directions.get(resource_id, "")
        ).lower()
        if resource_direction in {"read", "write"}:
            return [(resource_direction, total_bytes)]
    write_bytes = _metadata_write_bytes(task, total_bytes)
    if write_bytes is None:
        return [(_memory_direction(task, resource_id), total_bytes)]
    read_bytes = max(0.0, total_bytes - write_bytes)
    rows: List[Tuple[str, float]] = []
    if write_bytes > 0.0:
        rows.append(("write", write_bytes))
    if read_bytes > 0.0:
        rows.append(("read", read_bytes))
    return rows


def _empty_component_samples(
    scenario: ScenarioConfig,
) -> Dict[str, Dict[str, List[Tuple[float, float, float]]]]:
    return {
        component.component_id: {}
        for component in scenario.hardware.components
    }


def _append_sample(
    samples: Dict[str, Dict[str, List[Tuple[float, float, float]]]],
    owner_id: str,
    metric: str,
    start_ns: float,
    end_ns: float,
    value: float,
) -> None:
    if not math.isfinite(float(value)) or value <= 0.0 or end_ns <= start_ns:
        return
    samples.setdefault(owner_id, {}).setdefault(metric, []).append(
        (float(start_ns), float(end_ns), float(value))
    )


def _static_activity_samples(
    result: StaticRunResult,
) -> Tuple[
    Dict[str, Dict[str, List[Tuple[float, float, float]]]],
    Dict[str, List[Tuple[float, float, float]]],
    Dict[str, Dict[str, set]],
]:
    scenario = result.scenario
    component_map = scenario.hardware.component_map()
    rank_owners = _rank_owner_maps(scenario)
    capacities = {
        component_id: _component_capacities(scenario, component)
        for component_id, component in component_map.items()
    }
    component_samples = _empty_component_samples(scenario)
    link_samples: Dict[str, List[Tuple[float, float, float]]] = {
        link.link_id: [] for link in scenario.hardware.links
    }
    channels: Dict[str, Dict[str, set]] = {
        component_id: {} for component_id in component_map
    }

    for task in result.trace.tasks:
        mapped: List[Tuple[Any, Optional[str], Optional[str]]] = []
        rank = _integer_or_none(task.metadata.get("rank"))
        for interval in task.resource_intervals:
            resource_id = str(interval.resource_id)
            link_id = _resource_link_id(scenario, resource_id)
            component_id = _resource_component_id(
                scenario, resource_id, task.metadata, rank_owners
            )
            mapped.append((interval, component_id, link_id))
            duration_ns = float(interval.end_ns - interval.start_ns)
            if duration_ns <= 0.0:
                continue
            if link_id is not None:
                link = next(
                    item for item in scenario.hardware.links if item.link_id == link_id
                )
                bandwidth = _finite_non_negative(link.bandwidth_gbps) * 1.0e9 / 8.0
                value = (
                    float(interval.bytes_moved) / (duration_ns / 1.0e9) / bandwidth
                    if interval.bytes_moved > 0 and bandwidth > 0.0
                    else 0.0
                )
                if value > 0.0:
                    link_samples.setdefault(link_id, []).append(
                        (interval.start_ns, interval.end_ns, value)
                    )
                continue
            if component_id is None:
                continue
            channel = channels.setdefault(component_id, {}).setdefault(
                resource_id, set()
            )
            if rank is not None:
                channel.add(rank)
            _append_sample(
                component_samples,
                component_id,
                "busy_fraction",
                interval.start_ns,
                interval.end_ns,
                1.0,
            )
            component = component_map[component_id]
            kind = component.normalized_kind
            byte_count = _finite_non_negative(interval.bytes_moved)
            if byte_count > 0.0:
                direction_rows = _memory_direction_rows(
                    task, resource_id, byte_count
                )
                if kind in ACTIVE_MEMORY_COMPONENT_KINDS:
                    for direction, directional_bytes in direction_rows:
                        capacity_key = "{}_bandwidth_bytes_per_s".format(direction)
                        bandwidth = capacities[component_id][capacity_key]
                        utilization = (
                            directional_bytes / (duration_ns / 1.0e9) / bandwidth
                            if bandwidth > 0.0
                            else 0.0
                        )
                        _append_sample(
                            component_samples,
                            component_id,
                            "memory_{}_bandwidth_utilization".format(direction),
                            interval.start_ns,
                            interval.end_ns,
                            utilization,
                        )
                if kind in OFFLOAD_STORAGE_COMPONENT_KINDS:
                    dma_resource = _is_dma_resource(resource_id, task)
                    if dma_resource:
                        bandwidth = capacities[component_id][
                            "dma_bandwidth_bytes_per_s"
                        ]
                        utilization = (
                            byte_count / (duration_ns / 1.0e9) / bandwidth
                            if bandwidth > 0.0
                            else 1.0
                        )
                        _append_sample(
                            component_samples,
                            component_id,
                            "dma_engine_utilization",
                            interval.start_ns,
                            interval.end_ns,
                            utilization,
                        )
                    else:
                        for direction, directional_bytes in direction_rows:
                            bandwidth = capacities[component_id][
                                "{}_bandwidth_bytes_per_s".format(direction)
                            ]
                            utilization = (
                                directional_bytes
                                / (duration_ns / 1.0e9)
                                / bandwidth
                                if bandwidth > 0.0
                                else 0.0
                            )
                            _append_sample(
                                component_samples,
                                component_id,
                                "storage_{}_bandwidth_utilization".format(
                                    direction
                                ),
                                interval.start_ns,
                                interval.end_ns,
                                utilization,
                            )
                    aggregate_bandwidth = (
                        capacities[component_id]["dma_bandwidth_bytes_per_s"]
                        if dma_resource
                        else max(
                            capacities[component_id][
                                "read_bandwidth_bytes_per_s"
                            ],
                            capacities[component_id][
                                "write_bandwidth_bytes_per_s"
                            ],
                        )
                    )
                    aggregate_utilization = (
                        byte_count
                        / (duration_ns / 1.0e9)
                        / aggregate_bandwidth
                        if aggregate_bandwidth > 0.0
                        else 1.0
                    )
                    _append_sample(
                        component_samples,
                        component_id,
                        "storage_io_utilization",
                        interval.start_ns,
                        interval.end_ns,
                        aggregate_utilization,
                    )
            if kind in {"fabric", "switch", "bridge", "interconnect"}:
                _append_sample(
                    component_samples,
                    component_id,
                    "fabric_bandwidth_utilization",
                    interval.start_ns,
                    interval.end_ns,
                    1.0,
                )

        analytical_ops = _finite_non_negative(task.metadata.get("analytical_ops"))
        if analytical_ops <= 0.0 or task.category not in {
            TaskCategory.COMPUTE,
            TaskCategory.CIM,
        }:
            continue
        compute_rows = [
            (interval, component_id)
            for interval, component_id, link_id in mapped
            if link_id is None
            and component_id is not None
            and _is_compute_resource(str(interval.resource_id))
        ]
        if not compute_rows:
            compute_rows = [
                (interval, component_id)
                for interval, component_id, link_id in mapped
                if link_id is None
                and component_id is not None
                and component_map[component_id].normalized_kind
                not in ACTIVE_MEMORY_COMPONENT_KINDS
                and component_map[component_id].normalized_kind
                not in OFFLOAD_STORAGE_COMPONENT_KINDS
            ][:1]
        share_count = max(1, len(compute_rows))
        for interval, component_id in compute_rows:
            duration_ns = float(interval.end_ns - interval.start_ns)
            peak_ops = _operator_peak_ops_per_s(
                scenario,
                component_map[component_id],
                task,
                capacities[component_id]["peak_ops_per_s"],
            )
            if duration_ns <= 0.0 or peak_ops <= 0.0:
                continue
            utilization = (
                analytical_ops / share_count / (duration_ns / 1.0e9) / peak_ops
            )
            _append_sample(
                component_samples,
                component_id,
                "modeled_compute_utilization",
                interval.start_ns,
                interval.end_ns,
                utilization,
            )
    return component_samples, link_samples, channels


def _online_activity_samples(
    result: OnlineScenarioResult,
) -> Tuple[
    Dict[str, Dict[str, List[Tuple[float, float, float]]]],
    Dict[str, List[Tuple[float, float, float]]],
    Dict[str, Dict[str, set]],
]:
    scenario = result.scenario
    component_map = scenario.hardware.component_map()
    rank_owners = _rank_owner_maps(scenario)
    capacities = {
        component_id: _component_capacities(scenario, component)
        for component_id, component in component_map.items()
    }
    component_samples = _empty_component_samples(scenario)
    link_samples: Dict[str, List[Tuple[float, float, float]]] = {
        link.link_id: [] for link in scenario.hardware.links
    }
    channels: Dict[str, Dict[str, set]] = {
        component_id: {} for component_id in component_map
    }
    for batch in result.serving.batches:
        duration_ns = float(batch.end_ns - batch.start_ns)
        if duration_ns <= 0.0:
            continue
        raw_busy = batch.cost.metadata.get("resource_busy_ns", {})
        resource_busy = raw_busy if isinstance(raw_busy, Mapping) else {}
        raw_directional = batch.cost.metadata.get(
            "resource_busy_by_direction_ns", {}
        )
        directional = (
            raw_directional if isinstance(raw_directional, Mapping) else {}
        )
        directional_read = (
            directional.get("read", {})
            if isinstance(directional.get("read", {}), Mapping)
            else {}
        )
        directional_write = (
            directional.get("write", {})
            if isinstance(directional.get("write", {}), Mapping)
            else {}
        )
        compute_rows: List[Tuple[str, float]] = []
        for raw_resource_id, raw_busy_ns in sorted(resource_busy.items()):
            resource_id = str(raw_resource_id)
            busy_ns = min(duration_ns, _finite_non_negative(raw_busy_ns))
            if busy_ns <= 0.0:
                continue
            value = busy_ns / duration_ns
            link_id = _resource_link_id(scenario, resource_id)
            if link_id is not None:
                link_samples.setdefault(link_id, []).append(
                    (batch.start_ns, batch.end_ns, value)
                )
                continue
            component_id = _resource_component_id(
                scenario, resource_id, {}, rank_owners
            )
            if component_id is None:
                continue
            channels.setdefault(component_id, {}).setdefault(resource_id, set())
            _append_sample(
                component_samples,
                component_id,
                "busy_fraction",
                batch.start_ns,
                batch.end_ns,
                value,
            )
            kind = component_map[component_id].normalized_kind
            if _is_compute_resource(resource_id):
                compute_rows.append((component_id, busy_ns))
            elif kind in ACTIVE_MEMORY_COMPONENT_KINDS:
                read_busy = min(
                    duration_ns,
                    _finite_non_negative(directional_read.get(resource_id, 0.0)),
                )
                write_busy = min(
                    duration_ns,
                    _finite_non_negative(directional_write.get(resource_id, 0.0)),
                )
                if read_busy <= 0.0 and write_busy <= 0.0:
                    read_busy = busy_ns
                for direction, direction_busy in (
                    ("read", read_busy),
                    ("write", write_busy),
                ):
                    _append_sample(
                        component_samples,
                        component_id,
                        "memory_{}_bandwidth_utilization".format(direction),
                        batch.start_ns,
                        batch.end_ns,
                        direction_busy / duration_ns,
                    )
            elif kind in OFFLOAD_STORAGE_COMPONENT_KINDS:
                dma_resource = _is_dma_resource(resource_id)
                if dma_resource:
                    _append_sample(
                        component_samples,
                        component_id,
                        "dma_engine_utilization",
                        batch.start_ns,
                        batch.end_ns,
                        value,
                    )
                else:
                    read_busy = min(
                        duration_ns,
                        _finite_non_negative(
                            directional_read.get(resource_id, 0.0)
                        ),
                    )
                    write_busy = min(
                        duration_ns,
                        _finite_non_negative(
                            directional_write.get(resource_id, 0.0)
                        ),
                    )
                    if read_busy <= 0.0 and write_busy <= 0.0:
                        direction = (
                            "write"
                            if resource_id.lower().endswith(".write")
                            else "read"
                        )
                        if direction == "write":
                            write_busy = busy_ns
                        else:
                            read_busy = busy_ns
                    for direction, direction_busy in (
                        ("read", read_busy),
                        ("write", write_busy),
                    ):
                        _append_sample(
                            component_samples,
                            component_id,
                            "storage_{}_bandwidth_utilization".format(
                                direction
                            ),
                            batch.start_ns,
                            batch.end_ns,
                            direction_busy / duration_ns,
                        )
                _append_sample(
                    component_samples,
                    component_id,
                    "storage_io_utilization",
                    batch.start_ns,
                    batch.end_ns,
                    value,
                )
            elif kind in {"fabric", "switch", "bridge", "interconnect"}:
                _append_sample(
                    component_samples,
                    component_id,
                    "fabric_bandwidth_utilization",
                    batch.start_ns,
                    batch.end_ns,
                    value,
                )

        coverage_raw = batch.cost.metadata.get("coverage", {})
        coverage = coverage_raw if isinstance(coverage_raw, Mapping) else {}
        operations = sum(
            _finite_non_negative(row.get("operations"))
            for row in coverage.values()
            if isinstance(row, Mapping)
        )
        total_compute_busy = sum(row[1] for row in compute_rows)
        if operations <= 0.0 or total_compute_busy <= 0.0:
            continue
        for component_id, busy_ns in compute_rows:
            peak_ops = capacities[component_id]["peak_ops_per_s"]
            if peak_ops <= 0.0:
                continue
            assigned_ops = operations * busy_ns / total_compute_busy
            utilization = assigned_ops / (duration_ns / 1.0e9) / peak_ops
            _append_sample(
                component_samples,
                component_id,
                "modeled_compute_utilization",
                batch.start_ns,
                batch.end_ns,
                utilization,
            )
    return component_samples, link_samples, channels


def _declared_weight_residency(result: RunResult) -> Dict[str, int]:
    _layout, lookup = _memory_layout(result, segment_limit=1)
    totals: Dict[str, int] = {}
    seen = set()
    for segment in lookup.values():
        marker = id(segment)
        if marker in seen or segment.get("allocation_kind") != "weight":
            continue
        seen.add(marker)
        component_id = str(segment.get("component_id", ""))
        length_bytes = _non_negative_int(segment.get("length_bytes"))
        if component_id and length_bytes > 0:
            totals[component_id] = totals.get(component_id, 0) + length_bytes
    return dict(sorted(totals.items()))


def _estimated_static_runtime_residency(
    result: StaticRunResult,
) -> Dict[str, Dict[str, float]]:
    scenario = result.scenario
    rows: Dict[str, Dict[str, float]] = {}
    try:
        plan = compile_serving_plan(scenario)
    except (KeyError, TypeError, ValueError):
        return rows
    workload = scenario.workload
    if workload.requests:
        request_token_counts = [
            max(0, int(request.prompt_tokens))
            + max(0, int(request.output_tokens) - 1)
            for request in workload.requests
        ]
        request_count = len(request_token_counts)
        pages = sum(
            int(math.ceil(tokens / float(plan.kv_policy.tokens_per_page)))
            for tokens in request_token_counts
            if tokens > 0
        )
    else:
        request_count = max(0, int(workload.effective_request_count))
        tokens_per_request = max(
            0, int(workload.prompt_tokens)
        ) + max(
            0, int(workload.output_tokens) - 1
        )
        pages = request_count * (
            int(
                math.ceil(
                    tokens_per_request / float(plan.kv_policy.tokens_per_page)
                )
            )
            if tokens_per_request > 0
            else 0
        )
    if plan.kv_policy.cache_component:
        peak_bytes = min(
            max(0, int(plan.kv_policy.capacity_bytes)),
            pages * max(0, int(plan.kv_policy.bytes_per_page)),
        )
        rows.setdefault(str(plan.kv_policy.cache_component), {})[
            "kv_cache_residency_bytes"
        ] = float(peak_bytes)
    if plan.linear_state_policy.cache_component:
        peak_state = min(
            max(0, int(plan.linear_state_policy.capacity_bytes)),
            request_count
            * max(0, int(plan.linear_state_policy.bytes_per_request)),
        )
        rows.setdefault(str(plan.linear_state_policy.cache_component), {})[
            "linear_state_residency_bytes"
        ] = float(peak_state)
    return rows


def _online_runtime_residency(
    result: OnlineScenarioResult,
) -> Dict[str, Dict[str, float]]:
    serving = result.serving
    rows: Dict[str, Dict[str, float]] = {}
    kv_policy = serving.plan.kv_policy
    state_policy = serving.plan.linear_state_policy
    if kv_policy.cache_component:
        rows.setdefault(str(kv_policy.cache_component), {})[
            "kv_cache_residency_bytes"
        ] = float(max(0, int(serving.kv_metrics.peak_used_bytes)))
    if kv_policy.offload_component:
        rows.setdefault(str(kv_policy.offload_component), {})[
            "kv_cache_residency_bytes"
        ] = float(max(0, int(serving.kv_metrics.offload_peak_bytes)))
    if state_policy.cache_component:
        rows.setdefault(str(state_policy.cache_component), {})[
            "linear_state_residency_bytes"
        ] = float(max(0, int(serving.linear_state_metrics.peak_used_bytes)))
    if state_policy.offload_component:
        rows.setdefault(str(state_policy.offload_component), {})[
            "linear_state_residency_bytes"
        ] = float(max(0, int(serving.linear_state_metrics.offload_peak_bytes)))
    return rows


def _component_timeseries(result: RunResult) -> Dict[str, Any]:
    """Build the bounded public component-capacity and interval contract."""

    scenario = result.scenario
    online = isinstance(result, OnlineScenarioResult)
    retained_static = (
        isinstance(result, ScenarioResult)
        and result.execution.retention_policy is not RetentionPolicy.EXACT
    )
    makespan_ns = (
        float(result.serving.makespan_ns)
        if online
        else float(result.trace.makespan_ns)
    )
    if online:
        component_samples, link_samples, channel_rows = _online_activity_samples(result)
        runtime_residency = _online_runtime_residency(result)
        activity_fidelity = TraceFidelity.AGGREGATE
        top_fidelity = TraceFidelity.AGGREGATE
    else:
        component_samples, link_samples, channel_rows = _static_activity_samples(result)
        runtime_residency = _estimated_static_runtime_residency(result)
        activity_fidelity = {
            RetentionPolicy.EXACT: TraceFidelity.EXACT,
            RetentionPolicy.STREAMING: TraceFidelity.REPRESENTATIVE,
            RetentionPolicy.AGGREGATE: TraceFidelity.AGGREGATE,
        }[result.execution.retention_policy]
        top_fidelity = activity_fidelity
    weight_residency = _declared_weight_residency(result)
    rank_rows = _rank_drilldown(scenario)
    component_payloads: List[Dict[str, Any]] = []

    ratio_capacity = _capacity_payload(1.0, "ratio", "归一化上限")
    for component in sorted(
        scenario.hardware.components, key=lambda item: item.component_id
    ):
        component_id = component.component_id
        kind = component.normalized_kind
        capacities = _component_capacities(scenario, component)
        series: List[Dict[str, Any]] = []
        metric_samples = dict(component_samples.get(component_id, {}))
        is_compute_component = (
            capacities["peak_ops_per_s"] > 0.0
            or kind in {"gpu", "cpu", "accelerator"}
            or "cim" in kind
        )
        if is_compute_component:
            metric_samples.setdefault("busy_fraction", [])
            metric_samples.setdefault("modeled_compute_utilization", [])
        if kind in ACTIVE_MEMORY_COMPONENT_KINDS:
            metric_samples.setdefault("memory_read_bandwidth_utilization", [])
            metric_samples.setdefault("memory_write_bandwidth_utilization", [])
        if kind in OFFLOAD_STORAGE_COMPONENT_KINDS:
            metric_samples.setdefault("storage_io_utilization", [])
            metric_samples.setdefault("storage_read_bandwidth_utilization", [])
            metric_samples.setdefault("storage_write_bandwidth_utilization", [])
            metric_samples.setdefault("dma_engine_utilization", [])
        if kind in {"fabric", "switch", "bridge", "interconnect"}:
            metric_samples.setdefault("fabric_bandwidth_utilization", [])
        for metric, samples in sorted(metric_samples.items()):
            intervals = _sweep_change_points(
                samples, makespan_ns, clamp_ratio=True
            )
            if metric == "modeled_compute_utilization":
                capacity = _capacity_payload(
                    capacities["peak_ops_per_s"],
                    "ops_per_s",
                    "HardwareIR 或 CIM profile",
                )
            elif metric == "memory_read_bandwidth_utilization":
                capacity = _capacity_payload(
                    capacities["read_bandwidth_bytes_per_s"],
                    "bytes_per_s",
                    "HardwareIR 或内存 profile",
                )
            elif metric == "memory_write_bandwidth_utilization":
                capacity = _capacity_payload(
                    capacities["write_bandwidth_bytes_per_s"],
                    "bytes_per_s",
                    "HardwareIR 或内存 profile",
                )
            elif metric == "storage_read_bandwidth_utilization":
                capacity = _capacity_payload(
                    capacities["read_bandwidth_bytes_per_s"],
                    "bytes_per_s",
                    "HardwareIR 存储介质读取端口",
                )
            elif metric == "storage_write_bandwidth_utilization":
                capacity = _capacity_payload(
                    capacities["write_bandwidth_bytes_per_s"],
                    "bytes_per_s",
                    "HardwareIR 存储介质写入端口",
                )
            elif metric == "dma_engine_utilization":
                capacity = _capacity_payload(
                    capacities["dma_bandwidth_bytes_per_s"],
                    "bytes_per_s",
                    "HardwareIR DMA profile",
                )
            elif metric in {"storage_io_utilization", "fabric_bandwidth_utilization"}:
                capacity = _capacity_payload(
                    max(
                        capacities["read_bandwidth_bytes_per_s"],
                        capacities["write_bandwidth_bytes_per_s"],
                    ),
                    "bytes_per_s",
                    "HardwareIR",
                )
            else:
                capacity = ratio_capacity
            series.append(
                _series_payload(
                    series_id="{}:{}".format(component_id, metric),
                    metric=metric,
                    unit="ratio",
                    intervals=intervals,
                    fidelity=activity_fidelity,
                    quality=SeriesQuality.MODELED,
                    capacity=capacity,
                    clamp_ratio=True,
                    note_cn=(
                        "区间只来自当前策略保留的静态任务；精确总体利用率见摘要。"
                        if retained_static
                        else (
                            "区间来自静态离散事件仿真的精确资源变点。"
                            if not online
                            else "区间表示批次包络内的聚合忙时，不能还原为逐算子轨迹。"
                        )
                    ),
                )
            )

        resident_values: Dict[str, float] = {}
        if component_id in weight_residency:
            resident_values["weight_residency_bytes"] = float(
                weight_residency[component_id]
            )
        resident_values.update(runtime_residency.get(component_id, {}))
        for metric in (
            "weight_residency_bytes",
            "kv_cache_residency_bytes",
            "linear_state_residency_bytes",
        ):
            if metric not in resident_values:
                continue
            is_weight = metric == "weight_residency_bytes"
            series.append(
                _series_payload(
                    series_id="{}:{}".format(component_id, metric),
                    metric=metric,
                    unit="bytes",
                    intervals=_constant_intervals(
                        makespan_ns,
                        resident_values[metric],
                        include_zero=not is_weight,
                    ),
                    fidelity=(
                        TraceFidelity.AGGREGATE
                        if online or not is_weight
                        else TraceFidelity.EXACT
                    ),
                    quality=(
                        SeriesQuality.DECLARED
                        if is_weight
                        else SeriesQuality.ESTIMATED
                    ),
                    capacity=_capacity_payload(
                        capacities["memory_bytes"], "bytes", "HardwareIR"
                    ),
                    allocation_kind={
                        "weight_residency_bytes": "weight",
                        "kv_cache_residency_bytes": "kv_cache",
                        "linear_state_residency_bytes": "linear_state",
                    }[metric],
                    note_cn=(
                        "权重驻留量来自已声明的物理分片或副本布局。"
                        if is_weight
                        else "仅提供运行峰值或保守峰值，未伪造完整分配生命周期。"
                    ),
                )
            )

        if kind in OFFLOAD_STORAGE_COMPONENT_KINDS:
            occupancy = sum(resident_values.values())
            series.append(
                _series_payload(
                    series_id="{}:storage_occupancy_bytes".format(component_id),
                    metric="storage_occupancy_bytes",
                    unit="bytes",
                    intervals=_constant_intervals(
                        makespan_ns, occupancy, include_zero=True
                    ),
                    fidelity=TraceFidelity.AGGREGATE,
                    quality=SeriesQuality.ESTIMATED,
                    capacity=_capacity_payload(
                        capacities["memory_bytes"], "bytes", "HardwareIR"
                    ),
                    allocation_kind="known_resident_total",
                    note_cn="占用量只汇总已知权重、KV 缓存和线性状态，不包含未知临时数据。",
                )
            )

        if (
            capacities["memory_bytes"] > 0.0
            or kind in ACTIVE_MEMORY_COMPONENT_KINDS
            or kind in OFFLOAD_STORAGE_COMPONENT_KINDS
            or kind in {"gpu", "cpu", "cim", "digital_sram_cim"}
        ):
            for metric, allocation_kind in (
                ("activation_residency_bytes", "activation"),
                ("temporary_residency_bytes", "temporary"),
            ):
                series.append(
                    _series_payload(
                        series_id="{}:{}".format(component_id, metric),
                        metric=metric,
                        unit="bytes",
                        fidelity=TraceFidelity.AGGREGATE,
                        quality=SeriesQuality.UNKNOWN,
                        capacity=_capacity_payload(
                            capacities["memory_bytes"], "bytes", "HardwareIR"
                        ),
                        allocation_kind=allocation_kind,
                        note_cn="当前后端没有完整分配与释放生命周期，因此不生成数值曲线。",
                    )
                )
        channels = [
            {
                "channel_id": resource_id,
                "resource_ids": [resource_id],
                "ranks": sorted(int(rank) for rank in ranks),
            }
            for resource_id, ranks in sorted(
                channel_rows.get(component_id, {}).items()
            )
        ]
        component_payloads.append(
            {
                "component_id": component_id,
                "component_kind": kind,
                "capacities": capacities,
                "drilldown": {
                    "ranks": rank_rows.get(component_id, []),
                    "channels": channels,
                },
                "series": sorted(series, key=lambda item: item["series_id"]),
            }
        )

    link_payloads: List[Dict[str, Any]] = []
    for link in sorted(scenario.hardware.links, key=lambda item: item.link_id):
        bandwidth = _finite_non_negative(link.bandwidth_gbps) * 1.0e9 / 8.0
        intervals = _sweep_change_points(
            link_samples.get(link.link_id, ()), makespan_ns, clamp_ratio=True
        )
        link_payloads.append(
            {
                "link_id": link.link_id,
                "source_component": link.source_component,
                "target_component": link.target_component,
                "protocol": link.protocol,
                "capacities": {"bandwidth_bytes_per_s": bandwidth},
                "series": [
                    _series_payload(
                        series_id="{}:link_bandwidth_utilization".format(
                            link.link_id
                        ),
                        metric="link_bandwidth_utilization",
                        unit="ratio",
                        intervals=intervals,
                        fidelity=activity_fidelity,
                        quality=SeriesQuality.MODELED,
                        capacity=_capacity_payload(
                            bandwidth, "bytes_per_s", "HardwareIR"
                        ),
                        scope="link",
                        clamp_ratio=True,
                        note_cn=(
                            "区间只来自当前策略保留的静态链路任务。"
                            if retained_static
                            else (
                                "区间来自静态离散事件仿真的精确链路变点。"
                                if not online
                                else "区间表示批次包络内的链路聚合忙时。"
                            )
                        ),
                    )
                ],
            }
        )
    return {
        "schema_version": COMPONENT_TIMESERIES_SCHEMA_VERSION,
        "time_unit": "ns",
        "execution_mode": "continuous_batching" if online else "static",
        "fidelity": top_fidelity.value,
        "fidelity_description_cn": (
            "当前静态保留策略未保存完整资源变点；精确总体指标不依赖这些样本。"
            if retained_static
            else (
                "静态活动曲线使用离散事件仿真的精确资源变点；驻留估算会在各曲线上单独标注。"
                if not online
                else "连续批处理曲线是批次包络聚合或代表性峰值，不能还原为逐算子时间线。"
            )
        ),
        "point_limit": DEFAULT_COMPONENT_TIMESERIES_POINT_LIMIT,
        "components": component_payloads,
        "links": link_payloads,
    }


def _is_mtp_marker(value: object) -> bool:
    return str(value).strip().lower() == "mtp"


def _metadata_marks_mtp(metadata: Mapping[str, Any]) -> bool:
    return any(
        _is_mtp_marker(metadata.get(key))
        for key in (
            "phase",
            "kind",
            "batch_kind",
            "cohort_kind",
            "item_phase",
        )
    )


def _item_has_mtp_candidate_work(item: object) -> bool:
    draft_tokens = getattr(item, "draft_tokens", None)
    if draft_tokens is not None:
        return _non_negative_int(draft_tokens) > 0
    verifier_tokens = getattr(item, "verifier_tokens", None)
    if verifier_tokens is None:
        verifier_tokens = (
            getattr(item, "proposed_tokens", 0)
            or getattr(item, "token_count", 0)
        )
    main_tokens = getattr(item, "main_tokens", None)
    if main_tokens is None:
        main_tokens = 1 if _is_mtp_marker(getattr(item, "phase", None)) else 0
    return _non_negative_int(verifier_tokens) > _non_negative_int(main_tokens)


def _event_has_mtp_candidate_work(details: Mapping[str, Any]) -> bool:
    if (
        _non_negative_int(details.get("draft_tokens")) > 0
        or _non_negative_int(details.get("accepted_draft_tokens")) > 0
        or _non_negative_int(details.get("rejected_draft_tokens")) > 0
    ):
        return True
    verifier_tokens = details.get("verifier_tokens", details.get("proposed", 0))
    return _non_negative_int(verifier_tokens) > _non_negative_int(
        details.get("main_tokens")
    )


def _online_mtp_token_events(serving: ServingResult) -> Tuple[Any, ...]:
    mtp_cohort_ids: set[str] = set()
    mixed_mtp_cohort_ids: set[str] = set()
    mtp_item_keys: set[Tuple[str, str]] = set()

    for batch in serving.batches:
        cohort_id = str(batch.cohort_id)
        items = tuple(getattr(batch, "items", ()) or ())
        generation_items = tuple(
            item
            for item in items
            if str(getattr(item, "phase", batch.kind)) not in {
                "prefill",
                "recompute",
            }
        )
        phase_mtp_items = tuple(
            item
            for item in generation_items
            if _is_mtp_marker(getattr(item, "phase", None))
        )
        mtp_items = tuple(
            item for item in phase_mtp_items if _item_has_mtp_candidate_work(item)
        )
        for item in mtp_items:
            mtp_item_keys.add((cohort_id, str(getattr(item, "request_id", ""))))
        if mtp_items:
            # Whole-cohort attribution is safe only when every item is real
            # MTP candidate work.  Comparing with generation_items used to
            # discard prefill/recompute items first, so a mixed cohort looked
            # pure and its final-prefill token leaked into the MTP summary.
            if len(mtp_items) == len(items):
                mtp_cohort_ids.add(cohort_id)
            else:
                mixed_mtp_cohort_ids.add(cohort_id)
            continue
        if phase_mtp_items:
            mixed_mtp_cohort_ids.add(cohort_id)
            continue
        raw_metadata = getattr(batch, "metadata", {})
        batch_metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        raw_cost_metadata = getattr(batch.cost, "metadata", {})
        cost_metadata = (
            raw_cost_metadata if isinstance(raw_cost_metadata, Mapping) else {}
        )
        if (
            _is_mtp_marker(getattr(batch, "kind", None))
            or _metadata_marks_mtp(batch_metadata)
            or _metadata_marks_mtp(cost_metadata)
        ):
            mtp_cohort_ids.add(cohort_id)

    for event in serving.events:
        details = event.details if isinstance(event.details, Mapping) else {}
        if (
            event.cohort_id is not None
            and _metadata_marks_mtp(details)
            and _event_has_mtp_candidate_work(details)
        ):
            mtp_cohort_ids.add(str(event.cohort_id))

    def is_mtp_token_event(event: Any) -> bool:
        if event.event_type != "tokens_committed" or event.cohort_id is None:
            return False
        cohort_id = str(event.cohort_id)
        details = event.details if isinstance(event.details, Mapping) else {}
        if str(details.get("source", "")).strip().lower() in {
            "prefill",
            "recompute",
        }:
            return False
        if (cohort_id, str(event.request_id)) in mtp_item_keys:
            return True
        if _metadata_marks_mtp(details) and _event_has_mtp_candidate_work(
            details
        ):
            return True
        return (
            cohort_id in mtp_cohort_ids
            and cohort_id not in mixed_mtp_cohort_ids
        )

    return tuple(event for event in serving.events if is_mtp_token_event(event))


_PREFILL_SERVICE_PHASES = {"prefill", "recompute"}
_GENERATION_SERVICE_EXCLUDED_PHASES = _PREFILL_SERVICE_PHASES | {
    "kv_swap_out",
    "kv_swap_in",
    "linear_state_swap_out",
    "linear_state_swap_in",
}


def _sim_subtarget_ms(value_ns: Optional[float]) -> Optional[float]:
    return None if value_ns is None else float(value_ns) / 1_000_000.0


def _sim_subtarget_mean_ms(
    total_ns: Optional[float], count: int
) -> Optional[float]:
    if total_ns is None or count <= 0:
        return None
    return float(total_ns) / float(count) / 1_000_000.0


def _sim_subtarget_batch_metadata(batch: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    raw_batch_metadata = getattr(batch, "metadata", {})
    if isinstance(raw_batch_metadata, Mapping):
        metadata.update(raw_batch_metadata)
    raw_cost_metadata = getattr(getattr(batch, "cost", None), "metadata", {})
    if isinstance(raw_cost_metadata, Mapping):
        metadata.update(raw_cost_metadata)
    return metadata


def _sim_subtarget_batch_wall_ns(batch: Any) -> Optional[float]:
    try:
        start_ns = float(getattr(batch, "start_ns"))
        end_ns = float(getattr(batch, "end_ns"))
    except (TypeError, ValueError, OverflowError):
        return None
    duration_ns = end_ns - start_ns
    if not math.isfinite(duration_ns) or duration_ns < 0.0:
        return None
    return duration_ns


def _sim_subtarget_batch_interval_ns(
    batch: Any,
) -> Optional[Tuple[float, float]]:
    """Return a finite batch wall interval for report-only projections."""

    try:
        start_ns = float(getattr(batch, "start_ns"))
        end_ns = float(getattr(batch, "end_ns"))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(start_ns)
        or not math.isfinite(end_ns)
        or end_ns < start_ns
    ):
        return None
    return start_ns, end_ns


def _sim_subtarget_batch_phases(batch: Any) -> set[str]:
    phases = {
        str(getattr(item, "phase", "")).strip().lower()
        for item in tuple(getattr(batch, "items", ()) or ())
        if str(getattr(item, "phase", "")).strip()
    }
    if not phases:
        kind = str(getattr(batch, "kind", "")).strip().lower()
        if kind:
            phases.add(kind)
    return phases


def _sim_subtarget_batch_is_mixed(batch: Any) -> bool:
    phases = _sim_subtarget_batch_phases(batch)
    return len(phases) > 1 or str(getattr(batch, "kind", "")).lower() == "mixed"


def _online_subtarget_request_rows(
    serving: ServingResult,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    request_rows: Dict[str, Dict[str, Any]] = {
        request_id: {
            "prefill_service_ns": 0.0,
            "prefill_service_additive_ns": 0.0,
            "mixed_prefill_service_ns": 0.0,
            "mixed_prefill_service_additive_ns": 0.0,
            "prefill_wall_union_ns": 0.0,
            "mixed_prefill_wall_union_ns": 0.0,
            "initial_queue_delay_ns": None,
            "ttft_residual_ns": None,
            "ttft_residual_additive_ns": None,
        }
        for request_id in serving.request_metrics
    }
    prefill_intervals: Dict[str, List[Tuple[float, float]]] = {
        request_id: [] for request_id in serving.request_metrics
    }
    mixed_prefill_intervals: Dict[str, List[Tuple[float, float]]] = {
        request_id: [] for request_id in serving.request_metrics
    }
    batch_item_coverage_complete = True
    batch_wall_coverage_complete = True
    truncated_batches = 0
    malformed_batches = 0

    for batch in serving.batches:
        metadata = _sim_subtarget_batch_metadata(batch)
        if (
            metadata.get("request_ids_truncated")
            or metadata.get("items_truncated")
            or metadata.get("subtarget_items_truncated")
        ):
            batch_item_coverage_complete = False
            truncated_batches += 1
            continue
        wall_ns = _sim_subtarget_batch_wall_ns(batch)
        if wall_ns is None:
            batch_wall_coverage_complete = False
            malformed_batches += 1
            continue
        interval = _sim_subtarget_batch_interval_ns(batch)
        items = tuple(getattr(batch, "items", ()) or ())
        declared_request_ids = {
            str(request_id)
            for request_id in tuple(getattr(batch, "request_ids", ()) or ())
            if str(request_id)
        }
        item_request_ids = {
            str(getattr(item, "request_id", ""))
            for item in items
            if str(getattr(item, "request_id", ""))
        }
        if declared_request_ids and not declared_request_ids <= item_request_ids:
            batch_item_coverage_complete = False
            malformed_batches += 1
            continue
        if declared_request_ids and not items:
            batch_item_coverage_complete = False
            malformed_batches += 1
            continue
        mixed = _sim_subtarget_batch_is_mixed(batch)
        serviced_request_ids = {
            str(getattr(item, "request_id", ""))
            for item in items
            if str(getattr(item, "phase", "")).strip().lower()
            in _PREFILL_SERVICE_PHASES
        }
        for request_id in serviced_request_ids:
            if request_id not in request_rows:
                continue
            additive_service_ns = (
                float(request_rows[request_id]["prefill_service_ns"] or 0.0)
                + wall_ns
            )
            request_rows[request_id]["prefill_service_ns"] = additive_service_ns
            request_rows[request_id][
                "prefill_service_additive_ns"
            ] = additive_service_ns
            if interval is not None:
                prefill_intervals[request_id].append(interval)
            if mixed:
                mixed_additive_service_ns = (
                    float(
                        request_rows[request_id][
                            "mixed_prefill_service_ns"
                        ]
                        or 0.0
                    )
                    + wall_ns
                )
                request_rows[request_id][
                    "mixed_prefill_service_ns"
                ] = mixed_additive_service_ns
                request_rows[request_id][
                    "mixed_prefill_service_additive_ns"
                ] = mixed_additive_service_ns
                if interval is not None:
                    mixed_prefill_intervals[request_id].append(interval)

    if not batch_item_coverage_complete or not batch_wall_coverage_complete:
        for row in request_rows.values():
            row["prefill_service_ns"] = None
            row["prefill_service_additive_ns"] = None
            row["mixed_prefill_service_ns"] = None
            row["mixed_prefill_service_additive_ns"] = None
            row["prefill_wall_union_ns"] = None
            row["mixed_prefill_wall_union_ns"] = None
    else:
        for request_id, row in request_rows.items():
            row["prefill_wall_union_ns"] = _union_interval_duration_ns(
                prefill_intervals.get(request_id, ())
            )
            row["mixed_prefill_wall_union_ns"] = _union_interval_duration_ns(
                mixed_prefill_intervals.get(request_id, ())
            )

    negative_residual_count = 0
    for request_id, metric in serving.request_metrics.items():
        row = request_rows.setdefault(
            request_id,
            {
                "prefill_service_ns": None,
                "prefill_service_additive_ns": None,
                "mixed_prefill_service_ns": None,
                "mixed_prefill_service_additive_ns": None,
                "prefill_wall_union_ns": None,
                "mixed_prefill_wall_union_ns": None,
                "initial_queue_delay_ns": None,
                "ttft_residual_ns": None,
                "ttft_residual_additive_ns": None,
            },
        )
        queue_delay = getattr(metric, "queue_delay_ns", None)
        if queue_delay is None and getattr(metric, "start_ns", None) is not None:
            queue_delay = float(metric.start_ns) - float(metric.arrival_ns)
        row["initial_queue_delay_ns"] = (
            None if queue_delay is None else float(queue_delay)
        )
        prefill_service_ns = row.get("prefill_service_ns")
        prefill_wall_union_ns = row.get("prefill_wall_union_ns")
        ttft_ns = getattr(metric, "ttft_ns", None)
        if prefill_wall_union_ns is None or ttft_ns is None:
            row["ttft_residual_ns"] = None
        else:
            # The formal residual follows the same wall-union prompt-eval
            # proxy used by flat local comparison rows.  Keep the old
            # additive residual separately for overlap diagnostics.
            residual_ns = float(ttft_ns) - float(prefill_wall_union_ns)
            row["ttft_residual_ns"] = residual_ns
            if residual_ns < 0.0:
                negative_residual_count += 1
        if prefill_service_ns is None or ttft_ns is None:
            row["ttft_residual_additive_ns"] = None
        else:
            row["ttft_residual_additive_ns"] = (
                float(ttft_ns) - float(prefill_service_ns)
            )

    return request_rows, {
        "batch_item_coverage_complete": batch_item_coverage_complete,
        "batch_wall_coverage_complete": batch_wall_coverage_complete,
        "truncated_batches": truncated_batches,
        "malformed_batches": malformed_batches,
        "negative_ttft_residual_count": negative_residual_count,
        "negative_ttft_residual_additive_count": sum(
            1
            for row in request_rows.values()
            if row.get("ttft_residual_additive_ns") is not None
            and row["ttft_residual_additive_ns"] < 0.0
        ),
    }


def _online_generation_subtargets(
    serving: ServingResult,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    batch_by_cohort: Dict[str, Any] = {}
    duplicate_cohort_ids = 0
    batch_wall_coverage_complete = True
    batch_item_coverage_complete = True
    for batch in serving.batches:
        cohort_id = str(getattr(batch, "cohort_id", ""))
        if not cohort_id:
            continue
        if cohort_id in batch_by_cohort:
            duplicate_cohort_ids += 1
        batch_by_cohort[cohort_id] = batch
        metadata = _sim_subtarget_batch_metadata(batch)
        if (
            metadata.get("request_ids_truncated")
            or metadata.get("items_truncated")
            or metadata.get("subtarget_items_truncated")
        ):
            batch_item_coverage_complete = False
        if _sim_subtarget_batch_wall_ns(batch) is None:
            batch_wall_coverage_complete = False

    visible_by_cohort: Dict[str, int] = {}
    token_event_coverage_complete = True
    token_events = 0
    malformed_token_events = 0
    for event in serving.events:
        if getattr(event, "event_type", None) != "tokens_committed":
            continue
        token_events += 1
        details = event.details if isinstance(event.details, Mapping) else None
        if details is None or "visible_tokens" not in details:
            token_event_coverage_complete = False
            malformed_token_events += 1
            continue
        visible_tokens = _non_negative_int(details.get("visible_tokens"))
        if visible_tokens <= 0:
            continue
        if event.cohort_id is None:
            token_event_coverage_complete = False
            malformed_token_events += 1
            continue
        cohort_id = str(event.cohort_id)
        if cohort_id not in batch_by_cohort:
            token_event_coverage_complete = False
            malformed_token_events += 1
            continue
        visible_by_cohort[cohort_id] = (
            visible_by_cohort.get(cohort_id, 0) + visible_tokens
        )

    generation_coverage_complete = (
        batch_wall_coverage_complete
        and batch_item_coverage_complete
        and token_event_coverage_complete
        and duplicate_cohort_ids == 0
    )
    visible_output_tokens = sum(visible_by_cohort.values())
    generation_wall_ns: Optional[float] = 0.0
    mixed_generation_wall_ns: Optional[float] = 0.0
    visible_cohort_count = 0
    if generation_coverage_complete:
        for cohort_id in sorted(visible_by_cohort):
            batch = batch_by_cohort[cohort_id]
            phases = _sim_subtarget_batch_phases(batch)
            if phases and phases <= _GENERATION_SERVICE_EXCLUDED_PHASES:
                continue
            wall_ns = _sim_subtarget_batch_wall_ns(batch)
            if wall_ns is None:
                generation_coverage_complete = False
                break
            visible_cohort_count += 1
            generation_wall_ns = float(generation_wall_ns or 0.0) + wall_ns
            if _sim_subtarget_batch_is_mixed(batch):
                mixed_generation_wall_ns = (
                    float(mixed_generation_wall_ns or 0.0) + wall_ns
                )
    if not generation_coverage_complete:
        generation_wall_ns = None
        mixed_generation_wall_ns = None
    eval_ms_per_token = (
        _sim_subtarget_ms(generation_wall_ns) / visible_output_tokens
        if generation_wall_ns is not None and visible_output_tokens > 0
        else None
    )
    return {
        "generation_wall_ns": generation_wall_ns,
        "generation_wall_ms": _sim_subtarget_ms(generation_wall_ns),
        "mixed_generation_wall_ns": mixed_generation_wall_ns,
        "mixed_generation_wall_ms": _sim_subtarget_ms(
            mixed_generation_wall_ns
        ),
        "visible_output_tokens": (
            visible_output_tokens if generation_coverage_complete else None
        ),
        "eval_ms_per_token": eval_ms_per_token,
        "wall_counting_semantics": (
            "each batch with positive visible generation tokens contributes "
            "its wall interval once per cohort"
        ),
        "visible_output_token_source": (
            "tokens_committed.details.visible_tokens grouped by cohort_id"
        ),
    }, {
        "generation_coverage_complete": generation_coverage_complete,
        "token_event_coverage_complete": token_event_coverage_complete,
        "batch_wall_coverage_complete": batch_wall_coverage_complete,
        "batch_item_coverage_complete": batch_item_coverage_complete,
        "duplicate_cohort_ids": duplicate_cohort_ids,
        "token_events": token_events,
        "malformed_token_events": malformed_token_events,
        "visible_generation_cohorts": visible_cohort_count,
    }


def _online_mtp_draft_subtargets(serving: ServingResult) -> Dict[str, Any]:
    draft_tokens = 0
    accepted_draft_tokens = 0
    malformed_token_events = 0
    for event in serving.events:
        if getattr(event, "event_type", None) != "tokens_committed":
            continue
        details = event.details if isinstance(event.details, Mapping) else None
        if details is None:
            malformed_token_events += 1
            continue
        draft_tokens += _non_negative_int(details.get("draft_tokens", 0))
        accepted_draft_tokens += _non_negative_int(
            details.get("accepted_draft_tokens", 0)
        )
    rejected_draft_tokens = draft_tokens - accepted_draft_tokens
    valid = rejected_draft_tokens >= 0 and malformed_token_events == 0
    draft_acceptance_rate = (
        accepted_draft_tokens / draft_tokens
        if valid and draft_tokens > 0
        else None
    )
    return {
        "draft_tokens": draft_tokens if valid else None,
        "accepted_draft_tokens": accepted_draft_tokens if valid else None,
        "rejected_draft_tokens": rejected_draft_tokens if valid else None,
        "draft_acceptance_rate": draft_acceptance_rate,
        "draft_acceptance_rate_semantics": (
            "accepted_draft_tokens / draft_tokens; main tokens are excluded"
        ),
        "source": "tokens_committed.details.draft_tokens and accepted_draft_tokens",
        "valid": valid,
        "malformed_token_events": malformed_token_events,
    }


def _union_interval_duration_ns(
    intervals: Iterable[Tuple[float, float]]
) -> float:
    ordered = sorted(
        (start, end)
        for start, end in intervals
        if math.isfinite(start) and math.isfinite(end) and end > start
    )
    total = 0.0
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    for start, end in ordered:
        if current_start is None or current_end is None:
            current_start, current_end = start, end
            continue
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def _online_gpu_active_proxy(result: OnlineScenarioResult) -> Dict[str, Any]:
    serving = result.serving
    makespan_ns = float(serving.makespan_ns)
    gpu_component_ids = {
        component.component_id
        for component in result.scenario.hardware.components
        if component.normalized_kind == "gpu"
    }
    if not gpu_component_ids or makespan_ns <= 0.0:
        return {
            "active_ratio": None,
            "active_ns": None,
            "measurement_wall_ns": makespan_ns if makespan_ns > 0.0 else None,
            "coverage": "unavailable",
            "proxy": True,
            "guardrail": (
                "GPU active proxy requires a positive measurement window and "
                "declared GPU components"
            ),
        }

    intervals: List[Tuple[float, float]] = []
    incomplete_batches = 0
    compute_batch_kinds = {"prefill", "decode", "mtp", "mixed", "recompute"}
    for batch in serving.batches:
        wall_ns = _sim_subtarget_batch_wall_ns(batch)
        if wall_ns is None or wall_ns <= 0.0:
            continue
        if str(getattr(batch, "kind", "")).lower() not in compute_batch_kinds:
            continue
        metadata = _sim_subtarget_batch_metadata(batch)
        raw_schedule = metadata.get("execution_stage_schedule")
        if (
            metadata.get("execution_stage_schedule_mode") != "resource_dag"
            or isinstance(raw_schedule, (str, bytes, Mapping))
            or not isinstance(raw_schedule, Sequence)
        ):
            incomplete_batches += 1
            continue
        for stage in raw_schedule:
            if not isinstance(stage, Mapping):
                incomplete_batches += 1
                continue
            if str(stage.get("component_id", "")) not in gpu_component_ids:
                continue
            try:
                start_ns = float(stage["start_ns"])
                end_ns = float(stage["end_ns"])
            except (KeyError, TypeError, ValueError, OverflowError):
                incomplete_batches += 1
                continue
            if not math.isfinite(start_ns) or not math.isfinite(end_ns):
                incomplete_batches += 1
                continue
            intervals.append((start_ns, end_ns))

    if incomplete_batches:
        return {
            "active_ratio": None,
            "active_ns": None,
            "measurement_wall_ns": makespan_ns,
            "coverage": "incomplete",
            "proxy": True,
            "guardrail": (
                "stage schedule coverage is incomplete; GPU active proxy "
                "fails closed instead of estimating from partial rows"
            ),
            "incomplete_batch_count": incomplete_batches,
        }
    active_ns = _union_interval_duration_ns(intervals)
    return {
        "active_ratio": active_ns / makespan_ns,
        "active_ns": active_ns,
        "measurement_wall_ns": makespan_ns,
        "coverage": "complete",
        "proxy": True,
        "semantics": (
            "union of GPU execution_stage_schedule intervals divided by "
            "wall-clock measurement window"
        ),
    }


_SIM_HARDWARE_LEDGER_DOMAIN_LIMIT = 128
_SIM_HARDWARE_TRANSFER_KINDS = frozenset(
    {
        "kv_swap_out",
        "kv_swap_in",
        "linear_state_swap_out",
        "linear_state_swap_in",
    }
)


def _sim_hardware_non_negative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0 or numeric != float(parsed):
        return None
    return parsed


def _sim_hardware_non_negative_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _sim_hardware_sequence_count(
    metadata: Mapping[str, Any], key: str
) -> Optional[int]:
    if key not in metadata:
        return None
    raw = metadata.get(key)
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
        return None
    if any(not isinstance(item, Mapping) for item in raw):
        return None
    return len(raw)


def _sim_hardware_mtp_rows(batch: Any) -> Tuple[Dict[str, int], bool]:
    totals = {
        "main_rows": 0,
        "draft_rows": 0,
        "committed_rows": 0,
        "rejected_rows": 0,
    }
    mtp_items = tuple(
        item
        for item in tuple(getattr(batch, "items", ()) or ())
        if str(getattr(item, "phase", "")).strip().lower() == "mtp"
    )
    for item in mtp_items:
        values = []
        for field in ("main_tokens", "draft_tokens", "committed_tokens"):
            value = _sim_hardware_non_negative_int(getattr(item, field, None))
            values.append(value)
        if any(value is None for value in values):
            return totals, False
        main, draft, committed = (int(value) for value in values)
        accepted_draft = committed - main
        if accepted_draft < 0 or accepted_draft > draft:
            return totals, False
        totals["main_rows"] += main
        totals["draft_rows"] += draft
        totals["committed_rows"] += committed
        totals["rejected_rows"] += draft - accepted_draft
    return totals, True


def _sim_hardware_owner_rows(
    metadata: Mapping[str, Any],
) -> Tuple[Optional[Tuple[Mapping[str, Any], ...]], bool]:
    if "owner_residency_transfer_batches" not in metadata:
        return None, False
    raw = metadata.get("owner_residency_transfer_batches")
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
        return None, False
    if any(not isinstance(row, Mapping) for row in raw):
        return None, False
    return tuple(raw), True


def _sim_hardware_controller_fact(
    metadata: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    raw_ledger = metadata.get("runtime_controller_ledger")
    if not isinstance(raw_ledger, Mapping):
        return None
    if (
        raw_ledger.get("schema_version")
        != "heterollm.runtime-controller-ledger/v1"
    ):
        return None
    stage_count = _sim_hardware_non_negative_int(
        metadata.get("runtime_controller_stage_count")
    )
    task_count = _sim_hardware_non_negative_int(raw_ledger.get("task_count"))
    raw_domains = raw_ledger.get("domains")
    if stage_count is None or task_count is None:
        return None
    if isinstance(raw_domains, (str, bytes, Mapping)) or not isinstance(
        raw_domains, Sequence
    ):
        return None
    domain_service_ns: Dict[str, float] = {}
    observed_bytes = 0.0
    for row in raw_domains:
        if not isinstance(row, Mapping):
            return None
        if "service_domain" not in row or "service_ns" not in row:
            return None
        service_ns = _sim_hardware_non_negative_number(row.get("service_ns"))
        byte_count = _sim_hardware_non_negative_int(
            row.get("observed_bytes")
        )
        if service_ns is None or byte_count is None:
            return None
        domain = str(row.get("service_domain")).strip()
        if not domain:
            return None
        domain_service_ns[domain] = (
            domain_service_ns.get(domain, 0.0) + service_ns
        )
        observed_bytes += byte_count
    return {
        "stage_count": stage_count,
        "task_count": task_count,
        "domain_service_ns": domain_service_ns,
        "domain_service_ns_total": sum(domain_service_ns.values()),
        "observed_bytes": observed_bytes,
    }


def _sim_hardware_phase_ledger(
    batches: Sequence[Any],
    *,
    owner_residency_metrics: Optional[Any] = None,
) -> Dict[str, Any]:
    """Aggregate the bounded current-V4 hardware phase ledger."""

    batch_rows = tuple(batches or ())
    metadata_rows = tuple(_sim_subtarget_batch_metadata(batch) for batch in batch_rows)
    owner_enabled = bool(
        getattr(owner_residency_metrics, "enabled", False)
        or any(row.get("owner_residency_enabled") is True for row in metadata_rows)
    )

    cohort_ids = []
    cohort_missing = False
    physical_total = 0
    physical_complete = True
    group_totals = {
        "operator_invocation_group_count": 0,
        "target_invocation_group_count": 0,
        "mtp_proposer_invocation_group_count": 0,
        "mtp_draft_catchup_invocation_group_count": 0,
    }
    groups_complete = True
    rows_total = {
        "main_rows": 0,
        "draft_rows": 0,
        "committed_rows": 0,
        "rejected_rows": 0,
    }
    rows_complete = True
    by_kind: Dict[str, Dict[str, Any]] = {}

    owner_int_fields = {
        "migration_event_count": "owner_residency_migration_event_count",
        "fault_batch_count": "owner_residency_fault_batch_count",
        "clean_eviction_batch_count": "owner_residency_clean_eviction_batch_count",
        "page_in_bytes": "owner_residency_page_in_bytes",
        "page_out_bytes": "owner_residency_page_out_bytes",
        "clean_discard_bytes": "owner_residency_clean_discard_bytes",
        "dirty_writeback_bytes": "owner_residency_dirty_writeback_bytes",
        "causal_transfer_stage_count": "owner_residency_causal_transfer_stage_count",
        "causal_violation_count": "owner_residency_causal_violation_count",
        "unattributed_positive_batch_count": (
            "owner_residency_unattributed_positive_batch_count"
        ),
        "unattributed_positive_bytes": (
            "owner_residency_unattributed_positive_bytes"
        ),
        "unattributed_page_in_bytes": "owner_residency_unattributed_page_in_bytes",
    }
    owner_number_fields = {
        "page_in_ns": "owner_residency_page_in_ns",
        "dirty_writeback_ns": "owner_residency_dirty_writeback_ns",
        "clean_discard_ns": "owner_residency_clean_discard_ns",
        "transfer_service_ns": "owner_residency_transfer_ns",
        "causally_placed_transfer_ns": (
            "owner_residency_causally_placed_transfer_ns"
        ),
        "tail_transfer_ns": "owner_residency_tail_transfer_ns",
        "unattributed_positive_service_ns": (
            "owner_residency_unattributed_positive_service_ns"
        ),
        "unattributed_page_in_service_ns": (
            "owner_residency_unattributed_page_in_service_ns"
        ),
    }
    owner_totals: Dict[str, Union[int, float]] = {
        key: 0 for key in (*owner_int_fields, *owner_number_fields)
    }
    owner_complete = True
    eligible_compute_batch_count = 0
    transfer_service_batch_count = 0
    transfer_bytes = 0
    causal_degraded = False
    causal_complete = True

    controller_facts = []
    controller_complete = True

    for batch, metadata in zip(batch_rows, metadata_rows):
        cohort_id = str(getattr(batch, "cohort_id", "")).strip()
        if cohort_id:
            cohort_ids.append(cohort_id)
        else:
            cohort_missing = True

        kind = str(getattr(batch, "kind", "")).strip().lower() or "unknown"
        is_transfer = kind in _SIM_HARDWARE_TRANSFER_KINDS
        kind_row = by_kind.setdefault(
            kind, {"batch_count": 0, "physical_rows": 0}
        )
        kind_row["batch_count"] += 1

        physical = (
            0
            if is_transfer
            else _sim_hardware_non_negative_int(
                metadata.get("physical_batch_rows")
            )
        )
        if physical is None:
            physical_complete = False
            kind_row["physical_rows"] = None
        elif kind_row["physical_rows"] is not None:
            physical_total += physical
            kind_row["physical_rows"] += physical

        group_counts = (
            {key: 0 for key in group_totals}
            if is_transfer
            else {
                "operator_invocation_group_count": _sim_hardware_sequence_count(
                    metadata, "operator_invocation_groups"
                ),
                "target_invocation_group_count": _sim_hardware_non_negative_int(
                    metadata.get("target_backbone_invocation_count")
                ),
                "mtp_proposer_invocation_group_count": _sim_hardware_sequence_count(
                    metadata, "mtp_proposer_invocation_groups"
                ),
                "mtp_draft_catchup_invocation_group_count": (
                    _sim_hardware_sequence_count(
                        metadata, "mtp_draft_catchup_invocation_groups"
                    )
                ),
            }
        )
        if any(value is None for value in group_counts.values()):
            groups_complete = False
        else:
            for key, value in group_counts.items():
                group_totals[key] += int(value)

        item_rows, item_rows_complete = _sim_hardware_mtp_rows(batch)
        if not item_rows_complete:
            rows_complete = False
        else:
            for key, value in item_rows.items():
                rows_total[key] += value

        if not is_transfer:
            controller_fact = _sim_hardware_controller_fact(metadata)
            if controller_fact is None:
                controller_complete = False
            else:
                controller_facts.append(controller_fact)

        if is_transfer:
            continue
        eligible_compute_batch_count += 1
        if not owner_enabled:
            continue
        batch_owner_enabled = metadata.get("owner_residency_enabled")
        if batch_owner_enabled is False:
            continue
        if batch_owner_enabled is not True:
            owner_complete = False
            causal_complete = False
            continue
        for output_key, metadata_key in owner_int_fields.items():
            value = _sim_hardware_non_negative_int(metadata.get(metadata_key))
            if value is None:
                owner_complete = False
            else:
                owner_totals[output_key] += value
        for output_key, metadata_key in owner_number_fields.items():
            value = _sim_hardware_non_negative_number(metadata.get(metadata_key))
            if value is None:
                owner_complete = False
            else:
                owner_totals[output_key] += value
        degraded = metadata.get("owner_residency_causality_degraded")
        if not isinstance(degraded, bool):
            owner_complete = False
        elif degraded:
            causal_degraded = True
        rows, rows_valid = _sim_hardware_owner_rows(metadata)
        if not rows_valid:
            owner_complete = False
            causal_complete = False
            continue
        transfer_service_batch_count += len(rows)
        for row in rows:
            byte_count = _sim_hardware_non_negative_int(row.get("byte_count"))
            if byte_count is None:
                owner_complete = False
                causal_complete = False
            else:
                transfer_bytes += byte_count

    if not batch_rows:
        controller_complete = True
    if controller_complete:
        controller_stage_count = sum(fact["stage_count"] for fact in controller_facts)
        controller_task_count = sum(fact["task_count"] for fact in controller_facts)
        controller_service_ns = sum(
            fact["domain_service_ns_total"] for fact in controller_facts
        )
        controller_observed_bytes = sum(
            fact["observed_bytes"] for fact in controller_facts
        )
        all_controller_domains: Dict[str, float] = {}
        for fact in controller_facts:
            for domain, value in fact["domain_service_ns"].items():
                all_controller_domains[domain] = (
                    all_controller_domains.get(domain, 0.0) + value
                )
        domain_count = len(all_controller_domains)
        controller_domains = dict(
            sorted(all_controller_domains.items())[
                :_SIM_HARDWARE_LEDGER_DOMAIN_LIMIT
            ]
        )
        dropped_domain_count = max(
            0, domain_count - _SIM_HARDWARE_LEDGER_DOMAIN_LIMIT
        )
        domains_truncated = dropped_domain_count > 0
        controller_status = "incomplete" if domains_truncated else "complete"
    else:
        controller_stage_count = controller_task_count = None
        controller_service_ns = controller_observed_bytes = None
        controller_domains = {}
        domain_count = dropped_domain_count = None
        domains_truncated = None
        controller_status = "incomplete"

    owner_status = (
        "incomplete"
        if owner_enabled and eligible_compute_batch_count and not owner_complete
        else "complete"
    )
    if not owner_complete:
        causal_complete = False
    if owner_totals["tail_transfer_ns"] > 0 or owner_totals["causal_violation_count"] > 0:
        causal_degraded = True
    if owner_totals["unattributed_positive_batch_count"] > 0:
        causal_degraded = True
    causal_coverage = (
        "incomplete"
        if not causal_complete or owner_status == "incomplete"
        else "degraded"
        if causal_degraded
        else "complete"
    )

    physical_rows = physical_total if physical_complete else None
    counts = {
        "cohort_count": len(set(cohort_ids)) if not cohort_missing else None,
        "batch_count": len(batch_rows),
        "physical_rows": physical_rows,
        "by_cohort_kind": {},
        **(
            group_totals
            if groups_complete
            else {key: None for key in group_totals}
        ),
    }
    for kind, row in sorted(by_kind.items()):
        counts["by_cohort_kind"][kind] = row

    row_payload = (
        dict(rows_total)
        if rows_complete
        else {key: None for key in rows_total}
    )
    owner_payload_keys = (
        *owner_totals,
        "service_batch_count",
        "transfer_bytes",
        "transfer_service_ms",
    )
    owner_payload: Dict[str, Any] = {
        key: None for key in owner_payload_keys
    }
    if owner_status != "incomplete":
        owner_payload.update(owner_totals)
        owner_payload.update(
            {
                "service_batch_count": transfer_service_batch_count,
                "transfer_bytes": transfer_bytes,
            }
        )
    owner_payload["transfer_service_ms"] = (
        owner_payload["transfer_service_ns"] / 1_000_000.0
        if owner_payload["transfer_service_ns"] is not None
        else None
    )
    owner_payload["coverage"] = owner_status

    controller_payload = {
        "stage_count": controller_stage_count,
        "task_count": controller_task_count,
        "domain_service_ns": controller_service_ns,
        "domain_count": domain_count,
        "domain_service_ns_by_domain": controller_domains,
        "domain_service_ns_by_domain_truncated": domains_truncated,
        "dropped_domain_count": dropped_domain_count,
        "observed_bytes": controller_observed_bytes,
        "coverage": controller_status,
    }
    causal_payload = {
        "coverage": causal_coverage,
        "causality_degraded": (
            causal_degraded if causal_coverage != "incomplete" else None
        ),
        "causally_placed_transfer_ns": (
            owner_payload["causally_placed_transfer_ns"]
            if causal_coverage != "incomplete"
            else None
        ),
        "tail_transfer_ns": (
            owner_payload["tail_transfer_ns"]
            if causal_coverage != "incomplete"
            else None
        ),
        "causal_violation_count": (
            owner_payload["causal_violation_count"]
            if causal_coverage != "incomplete"
            else None
        ),
        "unattributed_positive_batch_count": (
            owner_payload["unattributed_positive_batch_count"]
            if causal_coverage != "incomplete"
            else None
        ),
        "unattributed_positive_bytes": (
            owner_payload["unattributed_positive_bytes"]
            if causal_coverage != "incomplete"
            else None
        ),
        "unattributed_positive_service_ns": (
            owner_payload["unattributed_positive_service_ns"]
            if causal_coverage != "incomplete"
            else None
        ),
    }

    coverage = {
        "status": (
            "incomplete"
            if (
                cohort_missing
                or not physical_complete
                or not groups_complete
                or not rows_complete
                or controller_status == "incomplete"
                or owner_status == "incomplete"
                or causal_coverage == "incomplete"
            )
            else "degraded"
            if (
                causal_coverage == "degraded"
                or controller_status == "degraded"
            )
            else "complete"
        ),
        "cohorts": "incomplete" if cohort_missing else "complete",
        "physical_rows": "complete" if physical_complete else "incomplete",
        "invocation_groups": "complete" if groups_complete else "incomplete",
        "rows": "complete" if rows_complete else "incomplete",
        "owner_residency": owner_status,
        "runtime_controller": controller_status,
        "causal": causal_coverage,
    }
    return {
        "schema_version": "heterollm.hardware-phase-ledger/v1",
        "source": "ServingBatch.cost.metadata + ServingBatch.items",
        "coverage": coverage,
        "counts": counts,
        "rows": row_payload,
        "owner_residency": owner_payload,
        "runtime_controller": controller_payload,
        "causal_quality": causal_payload,
        "guardrails": (
            "aggregate current-V4 facts only; timeline details are omitted",
            "controller observed bytes may overlap bulk traffic and remain separately labeled",
            "owner transfer rows count service batches and bytes; batch totals own duration",
        ),
    }


def _online_simulated_subtargets(
    result: OnlineScenarioResult,
    request_subtargets: Mapping[str, Mapping[str, Optional[float]]],
    request_diagnostics: Mapping[str, Any],
    generation: Mapping[str, Any],
    generation_diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    serving = result.serving
    completed = [
        metric
        for metric in serving.request_metrics.values()
        if metric.status == RequestStatus.FINISHED
    ]
    ttft_values = [
        float(metric.ttft_ns)
        for metric in completed
        if metric.ttft_ns is not None
    ]
    tpot_values = [
        float(metric.tpot_ns)
        for metric in completed
        if metric.tpot_ns is not None
    ]
    e2e_values = [
        float(metric.finish_ns) - float(metric.arrival_ns)
        for metric in completed
        if metric.finish_ns is not None
    ]
    prefill_values = [
        float(row["prefill_service_ns"])
        for row in request_subtargets.values()
        if row.get("prefill_service_ns") is not None
    ]
    mixed_prefill_values = [
        float(row["mixed_prefill_service_ns"])
        for row in request_subtargets.values()
        if row.get("mixed_prefill_service_ns") is not None
    ]
    prefill_wall_union_values = [
        float(row["prefill_wall_union_ns"])
        for row in request_subtargets.values()
        if row.get("prefill_wall_union_ns") is not None
    ]
    mixed_prefill_wall_union_values = [
        float(row["mixed_prefill_wall_union_ns"])
        for row in request_subtargets.values()
        if row.get("mixed_prefill_wall_union_ns") is not None
    ]
    queue_values = [
        float(row["initial_queue_delay_ns"])
        for row in request_subtargets.values()
        if row.get("initial_queue_delay_ns") is not None
    ]
    residual_values = [
        float(row["ttft_residual_ns"])
        for row in request_subtargets.values()
        if row.get("ttft_residual_ns") is not None
    ]
    additive_residual_values = [
        float(row["ttft_residual_additive_ns"])
        for row in request_subtargets.values()
        if row.get("ttft_residual_additive_ns") is not None
    ]
    negative_residuals = [
        value for value in residual_values if value < 0.0
    ]
    residual_sum = None if negative_residuals else sum(residual_values)
    negative_additive_residuals = [
        value for value in additive_residual_values if value < 0.0
    ]
    # Additive residuals are retained as diagnostics even when overlap makes
    # them negative; they are not used as the formal quality gate.
    additive_residual_sum = sum(additive_residual_values)
    gpu_active = _online_gpu_active_proxy(result)
    owner_residency = serving.owner_residency_metrics
    hardware_phase_ledger = _sim_hardware_phase_ledger(
        serving.batches,
        owner_residency_metrics=owner_residency,
    )
    vram_peak_bytes = (
        max(0, int(owner_residency.peak_resident_bytes))
        if owner_residency.enabled
        else None
    )
    vram_interval_baseline_bytes = (
        max(0, int(owner_residency.interval_baseline_resident_bytes))
        if owner_residency.enabled
        else None
    )
    vram_interval_peak_bytes = (
        max(0, int(owner_residency.interval_peak_resident_bytes))
        if owner_residency.enabled
        else None
    )
    vram_interval_delta_bytes = (
        max(0, int(owner_residency.interval_resident_delta_bytes))
        if owner_residency.enabled
        else None
    )
    makespan_ns = float(serving.makespan_ns)
    total_energy_pj = sum(batch.cost.energy_pj for batch in serving.batches)
    modeled_power_mean_w = (
        total_energy_pj / makespan_ns * 1.0e-3
        if makespan_ns > 0.0
        else None
    )
    quality_available = (
        bool(request_diagnostics.get("batch_item_coverage_complete"))
        and bool(request_diagnostics.get("batch_wall_coverage_complete"))
        and bool(generation_diagnostics.get("generation_coverage_complete"))
        and not negative_residuals
    )
    guardrails = [
        "prefill_service_ns is an additive diagnostic: it counts full batch wall for every request with prefill or recompute work and may exceed TTFT when chunks overlap",
        "prefill_wall_union_ns is the union of each request's prefill/recompute batch wall intervals; overlapping chunks are counted once; it is the key-path source for flat prefill_service_ms/prefill_tps",
        "mixed_prefill_wall_union_ns is the same interval-union metric restricted to mixed batches",
        "flat prefill_service_ms and prefill_tps use prefill_wall_union_mean_ms; flat prefill_service_additive_ms preserves the additive service diagnostic",
        "initial_queue_delay_ns is ServingRequestMetrics.queue_delay_ns/start_ns minus arrival_ns, not TTFT residual",
        "ttft_residual_ns is TTFT minus prefill_wall_union_ns and is the formal prompt-eval residual; ttft_residual_additive_ns remains the additive diagnostic",
        "generation_wall_ns counts each visible-token batch once, even when multiple requests commit tokens",
        "MTP draft_acceptance_rate uses accepted_draft_tokens / draft_tokens and excludes main tokens",
        "vram.session_peak_mib is an owner_residency proxy, not device telemetry",
        "vram interval fields use the physical-total residency baseline captured at replay start and the peak observed before temporary workspace release",
        "modeled_total_dynamic_power_mean_w is analytical dynamic energy over wall time, not GPU board power",
    ]
    if not request_diagnostics.get("batch_item_coverage_complete"):
        guardrails.append("batch item coverage is incomplete; request service subtargets fail closed")
    if not generation_diagnostics.get("generation_coverage_complete"):
        guardrails.append("generation event or batch coverage is incomplete; generation subtargets fail closed")
    if negative_residuals:
        guardrails.append("negative TTFT residuals are exposed per request but excluded from aggregate residual means")
    if negative_additive_residuals:
        guardrails.append(
            "negative additive TTFT residuals are diagnostic overlap artifacts "
            "and do not invalidate the wall-union prompt-eval target"
        )
    if gpu_active.get("coverage") != "complete":
        guardrails.append(str(gpu_active.get("guardrail")))

    latency = {
        "request_entry_count": len(serving.request_metrics),
        "completed_request_entry_count": len(completed),
        "ttft_count": len(ttft_values),
        "ttft_sum_ns": sum(ttft_values),
        "ttft_mean_ms": _sim_subtarget_mean_ms(sum(ttft_values), len(ttft_values)),
        "tpot_count": len(tpot_values),
        "tpot_sum_ns": sum(tpot_values),
        "tpot_mean_ms": _sim_subtarget_mean_ms(sum(tpot_values), len(tpot_values)),
        "e2e_count": len(e2e_values),
        "e2e_sum_ns": sum(e2e_values),
        "e2e_mean_ms": _sim_subtarget_mean_ms(sum(e2e_values), len(e2e_values)),
        "prefill_service_count": len(prefill_values),
        "prefill_service_sum_ns": (
            sum(prefill_values)
            if request_diagnostics.get("batch_item_coverage_complete")
            and request_diagnostics.get("batch_wall_coverage_complete")
            else None
        ),
        "prefill_service_mean_ms": _sim_subtarget_mean_ms(
            (
                sum(prefill_values)
                if request_diagnostics.get("batch_item_coverage_complete")
                and request_diagnostics.get("batch_wall_coverage_complete")
                else None
            ),
            len(prefill_values),
        ),
        "mixed_prefill_service_count": len(mixed_prefill_values),
        "mixed_prefill_service_sum_ns": (
            sum(mixed_prefill_values)
            if request_diagnostics.get("batch_item_coverage_complete")
            and request_diagnostics.get("batch_wall_coverage_complete")
            else None
        ),
        "mixed_prefill_service_mean_ms": _sim_subtarget_mean_ms(
            (
                sum(mixed_prefill_values)
                if request_diagnostics.get("batch_item_coverage_complete")
                and request_diagnostics.get("batch_wall_coverage_complete")
                else None
            ),
            len(mixed_prefill_values),
        ),
        "prefill_wall_union_count": len(prefill_wall_union_values),
        "prefill_wall_union_sum_ns": (
            sum(prefill_wall_union_values)
            if request_diagnostics.get("batch_item_coverage_complete")
            and request_diagnostics.get("batch_wall_coverage_complete")
            else None
        ),
        "prefill_wall_union_mean_ms": _sim_subtarget_mean_ms(
            (
                sum(prefill_wall_union_values)
                if request_diagnostics.get("batch_item_coverage_complete")
                and request_diagnostics.get("batch_wall_coverage_complete")
                else None
            ),
            len(prefill_wall_union_values),
        ),
        "mixed_prefill_wall_union_count": len(
            mixed_prefill_wall_union_values
        ),
        "mixed_prefill_wall_union_sum_ns": (
            sum(mixed_prefill_wall_union_values)
            if request_diagnostics.get("batch_item_coverage_complete")
            and request_diagnostics.get("batch_wall_coverage_complete")
            else None
        ),
        "mixed_prefill_wall_union_mean_ms": _sim_subtarget_mean_ms(
            (
                sum(mixed_prefill_wall_union_values)
                if request_diagnostics.get("batch_item_coverage_complete")
                and request_diagnostics.get("batch_wall_coverage_complete")
                else None
            ),
            len(mixed_prefill_wall_union_values),
        ),
        "initial_queue_delay_count": len(queue_values),
        "initial_queue_delay_sum_ns": sum(queue_values),
        "initial_queue_delay_mean_ms": _sim_subtarget_mean_ms(
            sum(queue_values), len(queue_values)
        ),
        "ttft_residual_count": len(residual_values),
        "ttft_residual_sum_ns": residual_sum,
        "ttft_residual_mean_ms": _sim_subtarget_mean_ms(
            residual_sum, len(residual_values)
        ),
        "negative_ttft_residual_count": len(negative_residuals),
        "ttft_residual_additive_count": len(additive_residual_values),
        "ttft_residual_additive_sum_ns": additive_residual_sum,
        "ttft_residual_additive_mean_ms": _sim_subtarget_mean_ms(
            additive_residual_sum, len(additive_residual_values)
        ),
        "negative_ttft_residual_additive_count": len(
            negative_additive_residuals
        ),
    }
    return {
        "schema_version": "heterollm.simulated-subtargets/v1",
        "source": "ServingResult.batches/items/request_metrics/events",
        "quality": {
            "available": quality_available,
            "status": "complete" if quality_available else "incomplete",
            "batch_item_coverage": (
                "complete"
                if request_diagnostics.get("batch_item_coverage_complete")
                else "incomplete"
            ),
            "token_event_coverage": (
                "complete"
                if generation_diagnostics.get("token_event_coverage_complete")
                else "incomplete"
            ),
            "gpu_active_proxy_coverage": gpu_active.get("coverage"),
        },
        "guardrails": guardrails,
        "counts": {
            "request_count": len(serving.request_metrics),
            "completed_requests": len(completed),
            "batch_count": len(serving.batches),
            "batch_item_count": sum(
                len(tuple(getattr(batch, "items", ()) or ()))
                for batch in serving.batches
            ),
            "tokens_committed_events": generation_diagnostics.get("token_events", 0),
            "visible_generation_cohorts": generation_diagnostics.get(
                "visible_generation_cohorts", 0
            ),
            "truncated_batches": request_diagnostics.get("truncated_batches", 0),
            "malformed_batches": request_diagnostics.get("malformed_batches", 0),
            "malformed_token_events": generation_diagnostics.get(
                "malformed_token_events", 0
            ),
        },
        "latency": latency,
        "prefill": {
            "service_ns": _percentiles(prefill_values),
            "service_mean_ms": latency["prefill_service_mean_ms"],
            "service_additive_mean_ms": latency["prefill_service_mean_ms"],
            "mixed_service_mean_ms": latency["mixed_prefill_service_mean_ms"],
            "mixed_service_additive_mean_ms": latency[
                "mixed_prefill_service_mean_ms"
            ],
            "wall_union_ns": _percentiles(prefill_wall_union_values),
            "wall_union_mean_ms": latency["prefill_wall_union_mean_ms"],
            "mixed_wall_union_mean_ms": latency[
                "mixed_prefill_wall_union_mean_ms"
            ],
            "service_semantics": (
                "additive sum of full batch wall per request where that "
                "request has prefill or recompute work; diagnostic only"
            ),
            "service_additive_semantics": (
                "same additive full-batch wall diagnostic exposed explicitly "
                "to distinguish it from the flat wall-union target"
            ),
            "wall_union_semantics": (
                "union of each request's prefill/recompute batch wall "
                "intervals; overlapping chunks are counted once; this is "
                "the flat prefill_service_ms/prefill_tps source"
            ),
            "mixed_wall_union_semantics": (
                "union of each request's mixed prefill/recompute batch wall "
                "intervals; overlapping chunks are counted once"
            ),
        },
        "generation": dict(generation),
        "mtp": _online_mtp_draft_subtargets(serving),
        "hardware_phase_ledger": hardware_phase_ledger,
        "vram": {
            "session_peak_mib": (
                vram_peak_bytes / float(2**20)
                if vram_peak_bytes is not None
                else None
            ),
            "interval_baseline_resident_mib": (
                vram_interval_baseline_bytes / float(2**20)
                if vram_interval_baseline_bytes is not None
                else None
            ),
            "interval_peak_resident_mib": (
                vram_interval_peak_bytes / float(2**20)
                if vram_interval_peak_bytes is not None
                else None
            ),
            "interval_resident_delta_mib": (
                vram_interval_delta_bytes / float(2**20)
                if vram_interval_delta_bytes is not None
                else None
            ),
            "source": "owner_residency_metrics.peak_resident_bytes",
            "proxy": True,
            "semantics": (
                "owner_residency peak; local replay aggregates this across "
                "constructor, warmup, and all measured replays"
            ),
            "interval_semantics": (
                "physical-total resident bytes at this replay runtime start; "
                "peak sampled during the replay before temporary allocations "
                "are released; delta is peak minus baseline"
            ),
            "guardrail": "not nvidia-smi telemetry and not GPU board memory accounting",
        },
        "gpu_active_proxy": gpu_active,
        "power": {
            "modeled_total_dynamic_power_mean_w": modeled_power_mean_w,
            "source": "sum(batch.cost.energy_pj) / serving.makespan_ns",
            "guardrail": (
                "modeled dynamic workload power only; not GPU board power"
            ),
        },
    }


def _online_report_core(
    result: OnlineScenarioResult,
) -> _OnlineReportCore:
    serving = result.serving
    token_times = _online_token_times(result)
    (
        subtarget_request_rows,
        subtarget_request_diagnostics,
    ) = _online_subtarget_request_rows(serving)
    (
        generation_subtargets,
        generation_diagnostics,
    ) = _online_generation_subtargets(serving)
    request_rows: Dict[str, Any] = {}
    all_ttft: List[float] = []
    all_tbt: List[float] = []
    all_tpot: List[float] = []
    all_e2e: List[float] = []
    committed_total = 0
    completed = 0
    good_requests = 0
    good_tokens = 0
    scheduler = serving.plan.scheduler
    ordered_request_ids = sorted(serving.request_metrics)
    if len(ordered_request_ids) <= _SCALABLE_REQUEST_DETAIL_LIMIT:
        selected_request_ids = set(ordered_request_ids)
    else:
        leading = _SCALABLE_REQUEST_DETAIL_LIMIT // 2
        trailing = _SCALABLE_REQUEST_DETAIL_LIMIT - leading
        selected_request_ids = set(
            ordered_request_ids[:leading] + ordered_request_ids[-trailing:]
        )
    for request_id in ordered_request_ids:
        metric = serving.request_metrics[request_id]
        times = token_times.get(request_id, [])
        tbt = [times[index] - times[index - 1] for index in range(1, len(times))]
        e2e = (
            metric.finish_ns - metric.arrival_ns
            if metric.finish_ns is not None
            else None
        )
        row = {
            "request_id": request_id,
            "status": metric.status.value,
            "reason": metric.rejection_reason,
            "rejection_reason": metric.rejection_reason,
            "arrival_ns": metric.arrival_ns,
            "first_token_ns": metric.first_token_ns,
            "done_ns": metric.finish_ns,
            "ttft_ns": metric.ttft_ns,
            "tbt_ns": tbt,
            "tpot_ns": metric.tpot_ns,
            "e2e_ns": e2e,
            "visible_output_tokens": metric.visible_output_tokens,
            "proposed_tokens": metric.proposed_tokens,
            "accepted_tokens": metric.accepted_tokens,
            "rejected_tokens": max(
                0, metric.proposed_tokens - metric.accepted_tokens
            ),
            "queue_delay_ns": metric.queue_delay_ns,
            "initial_queue_delay_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("initial_queue_delay_ns"),
            "prefill_service_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("prefill_service_ns"),
            "prefill_service_additive_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("prefill_service_additive_ns"),
            "prefill_wall_union_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("prefill_wall_union_ns"),
            "ttft_residual_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("ttft_residual_ns"),
            "ttft_residual_additive_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("ttft_residual_additive_ns"),
            "mixed_prefill_service_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("mixed_prefill_service_ns"),
            "mixed_prefill_service_additive_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("mixed_prefill_service_additive_ns"),
            "mixed_prefill_wall_union_ns": subtarget_request_rows.get(
                request_id, {}
            ).get("mixed_prefill_wall_union_ns"),
            "preemptions": metric.preemptions,
            "swaps": metric.swaps,
            "recomputes": metric.recomputes,
            "deadline_met": metric.deadline_met,
            "category_time_ns": {},
            "critical_path_category_ns": {},
        }
        if request_id in selected_request_ids:
            request_rows[request_id] = row
        committed_total += metric.visible_output_tokens
        if metric.status == RequestStatus.FINISHED:
            completed += 1
            if metric.ttft_ns is not None:
                all_ttft.append(metric.ttft_ns)
            all_tbt.extend(tbt)
            if metric.tpot_ns is not None:
                all_tpot.append(metric.tpot_ns)
            if e2e is not None:
                all_e2e.append(e2e)
            meets = metric.deadline_met is not False
            if scheduler.slo_ttft_ns is not None:
                meets = meets and metric.ttft_ns is not None and metric.ttft_ns <= scheduler.slo_ttft_ns
            if scheduler.slo_tbt_ns is not None:
                meets = meets and (not tbt or max(tbt) <= scheduler.slo_tbt_ns)
            if meets:
                good_requests += 1
                good_tokens += metric.visible_output_tokens

    mtp_events = _online_mtp_token_events(serving)
    mtp_proposed_total = sum(
        max(0, int(event.details.get("proposed", 0)))
        for event in mtp_events
    )
    mtp_accepted_total = sum(
        max(0, int(event.details.get("accepted", 0)))
        for event in mtp_events
    )
    mtp_committed_total = sum(
        max(0, int(event.details.get("visible_tokens", 0)))
        for event in mtp_events
    )

    makespan = float(serving.makespan_ns)
    wall_seconds = makespan / 1_000_000_000.0 if makespan > 0 else 0.0
    arrivals = [metric.arrival_ns for metric in serving.request_metrics.values()]
    finishes = [
        metric.finish_ns
        for metric in serving.request_metrics.values()
        if metric.finish_ns is not None
    ]
    active_start = min(arrivals) if arrivals else 0.0
    active_end = max(finishes) if finishes else active_start
    active_ns = max(0.0, active_end - active_start)
    active_seconds = active_ns / 1_000_000_000.0 if active_ns > 0 else 0.0

    if len(serving.batches) >= 3:
        steady_start = serving.batches[0].end_ns
        steady_end = serving.batches[-1].start_ns
        if steady_end <= steady_start:
            steady_start, steady_end = active_start, active_end
    else:
        steady_start, steady_end = active_start, active_end
    steady_tokens = sum(
        max(0, int(event.details.get("visible_tokens", 0)))
        for event in serving.events
        if event.event_type == "tokens_committed"
        and steady_start <= event.timestamp_ns <= steady_end
    )
    steady_requests = sum(
        1
        for metric in serving.request_metrics.values()
        if metric.finish_ns is not None and steady_start <= metric.finish_ns <= steady_end
    )
    steady_seconds = max(0.0, steady_end - steady_start) / 1_000_000_000.0

    resource_busy = _sum_batch_metadata(result, "resource_busy_ns")
    utilization = {
        name: (busy / makespan if makespan > 0 else 0.0)
        for name, busy in resource_busy.items()
    }
    bottleneck = (
        max(utilization.items(), key=lambda item: (item[1], item[0]))
        if utilization
        else None
    )
    total_energy = sum(batch.cost.energy_pj for batch in serving.batches)
    total_bytes = sum(
        float(batch.cost.metadata.get("resource_accounted_bytes", 0.0))
        for batch in serving.batches
    )
    task_count = sum(
        int(batch.cost.metadata.get("task_count", 0))
        for batch in serving.batches
    )
    parallel = result.scenario.placement.parallel
    model_coverage = _model_coverage(result.scenario)
    mtp_model = model_coverage["mtp"]
    measurement_windows_ns = {
        "wall_clock": [0.0, makespan],
        "active": [active_start, active_end],
        "steady": [steady_start, steady_end],
    }
    simulated_subtargets = _online_simulated_subtargets(
        result,
        subtarget_request_rows,
        subtarget_request_diagnostics,
        generation_subtargets,
        generation_diagnostics,
    )
    summary = {
        "makespan_ns": makespan,
        "task_count": task_count,
        **_host_output_contract_projection(result.scenario),
        "batch_count": len(serving.batches),
        "completed_requests": completed,
        "rejected_requests": sum(
            metric.status == RequestStatus.REJECTED
            for metric in serving.request_metrics.values()
        ),
        "total_energy_pj": total_energy,
        "resource_accounted_bytes": total_bytes,
        "throughput": {
            "requests_per_s": completed / wall_seconds if wall_seconds else 0.0,
            "visible_output_tokens_per_s": (
                committed_total / wall_seconds if wall_seconds else 0.0
            ),
            "active_requests_per_s": (
                completed / active_seconds if active_seconds else 0.0
            ),
            "active_visible_tokens_per_s": (
                committed_total / active_seconds if active_seconds else 0.0
            ),
            "steady_requests_per_s": (
                steady_requests / steady_seconds if steady_seconds else 0.0
            ),
            "steady_visible_tokens_per_s": (
                steady_tokens / steady_seconds if steady_seconds else 0.0
            ),
        },
        "goodput": {
            "requests_per_s": (
                good_requests / active_seconds if active_seconds else 0.0
            ),
            "visible_output_tokens_per_s": (
                good_tokens / active_seconds if active_seconds else 0.0
            ),
            "qualified_requests": good_requests,
        },
        "measurement_windows_ns": measurement_windows_ns,
        "ttft_ns": _percentiles(all_ttft),
        "tbt_ns": _percentiles(all_tbt),
        "tpot_ns": _percentiles(all_tpot),
        "e2e_ns": _percentiles(all_e2e),
        "bottleneck_resource": (
            {"resource_id": bottleneck[0], "utilization": bottleneck[1]}
            if bottleneck
            else None
        ),
        "mtp": {
            "enabled": serving.plan.mtp.enabled,
            "prediction_layers": mtp_model["prediction_layers"],
            "auxiliary_head": mtp_model["auxiliary_head"],
            "declared_weight_bytes": mtp_model["declared_weight_bytes"],
            "operator_ids": mtp_model["operator_ids"],
            "weight_tensor_ids": mtp_model["weight_tensor_ids"],
            "method": serving.plan.mtp.method,
            "candidate_tokens": serving.plan.mtp.candidate_tokens,
            "candidate_tokens_semantics": "max_draft_tokens",
            "verifier_width_at_max": 1 + serving.plan.mtp.candidate_tokens,
            "min_draft_tokens": serving.plan.mtp.min_draft_tokens,
            "continuation_threshold": serving.plan.mtp.continuation_threshold,
            "proposal_length_model": serving.plan.mtp.proposal_length_model,
            "expected_draft_tokens_per_round": (
                serving.plan.mtp.expected_draft_tokens_per_round
            ),
            "draft_length_trace": list(serving.plan.mtp.draft_length_trace),
            "proposed_tokens": mtp_proposed_total,
            "accepted_tokens": mtp_accepted_total,
            "committed_tokens": mtp_committed_total,
            "rejected_tokens": max(
                0, mtp_proposed_total - mtp_accepted_total
            ),
            "effective_acceptance_rate": (
                mtp_accepted_total / mtp_proposed_total
                if mtp_proposed_total
                else None
            ),
        },
        "parallel": {
            "tp_degree": parallel.tp_degree,
            "pp_degree": parallel.pp_degree,
            "ep_degree": parallel.ep_degree,
            "world_size": parallel.world_size,
            "collective_algorithm": parallel.collective_algorithm,
            "routing_policy": parallel.routing_policy,
        },
        "simulated_subtargets": simulated_subtargets,
    }
    if serving.prompt_cache_save_timing:
        summary["prompt_cache_save_timing"] = dict(serving.prompt_cache_save_timing)
    return _OnlineReportCore(
        summary=summary,
        requests=request_rows,
        resource_utilization=dict(sorted(utilization.items())),
        measurement_windows_ns=measurement_windows_ns,
        model_coverage=model_coverage,
    )


def _online_report_dict(
    result: OnlineScenarioResult,
    options: VisualizationTraceOptions,
) -> Dict[str, Any]:
    core = _online_report_core(result)
    serving = result.serving
    category_time = _sum_batch_metadata(result, "category_time_ns")
    critical_time = _sum_batch_metadata(result, "critical_path_category_ns")
    analytical_coverage = {
        "evidence": "analytical",
        "calibration_version": ANALYTICAL_MODEL_VERSION,
        "model": core.model_coverage,
        "runtime": _sum_batch_coverage(result),
        "runtime_references": _sum_batch_coverage_references(result),
    }
    batch_history = [
        _bounded_batch_history_row(batch)
        for batch in serving.batches[:_SCALABLE_BATCH_HISTORY_LIMIT]
    ]
    scheduler_events = [
        _bounded_scheduler_event_row(event)
        for event in serving.events[:_SCALABLE_SCHEDULER_EVENT_LIMIT]
    ]
    result_semantics = _online_result_semantics(
        result, windows_ns=core.measurement_windows_ns
    )
    visualization = _visualization_payload(result, options)
    component_timeseries = _component_timeseries(result)
    return {
        "scenario": result.scenario.name,
        "manifest": to_primitive(result.manifest),
        "execution_mode": "continuous_batching",
        "retention_policy": result.retention_policy,
        "trace_fidelity": TraceFidelity.AGGREGATE.value,
        "result_semantics": result_semantics,
        "measurement_semantics": result_semantics,
        "summary": core.summary,
        "requests": core.requests,
        "resource_utilization": core.resource_utilization,
        "category_time_ns": category_time,
        "critical_path_category_ns": critical_time,
        "scheduler": to_primitive(serving.scheduler_metrics),
        "owner_residency": to_primitive(serving.owner_residency_metrics),
        "kv_cache": {
            **to_primitive(serving.kv_metrics),
            "prefetch_distance_modeled": False,
            "prefetch_distance_semantics": _prefetch_semantics(result.scenario),
            "modeling_limits": [
                "offload_ratio bounds pressure-triggered migratable capacity; it is not a per-token mirror ratio",
                _prefetch_semantics(result.scenario)["description"],
                "logical KV traffic counters are not added to resource-accounted bytes when attention kernels already include local memory traffic",
                "KV read means external persisted-cache access only; prefill/recompute historical-prompt attention remains modeled as causal compute and ordinary attention/activation traffic",
            ],
        },
        "linear_state": to_primitive(serving.linear_state_metrics),
        "analytical_coverage": analytical_coverage,
        "batch_history": batch_history,
        "batch_trace_index": visualization["batch_trace_index"],
        "scheduler_events": scheduler_events,
        "response_limits": {
            "request_details": {
                "total": len(serving.request_metrics),
                "returned": len(core.requests),
                "limit": _SCALABLE_REQUEST_DETAIL_LIMIT,
                "truncated": len(core.requests) < len(serving.request_metrics),
                "selection": "lexicographic_head_and_tail",
            },
            "batch_history": {
                "total": len(serving.batches),
                "returned": len(batch_history),
                "limit": _SCALABLE_BATCH_HISTORY_LIMIT,
                "truncated": len(batch_history) < len(serving.batches),
            },
            "scheduler_events": {
                "total": len(serving.events),
                "returned": len(scheduler_events),
                "limit": _SCALABLE_SCHEDULER_EVENT_LIMIT,
                "truncated": len(scheduler_events) < len(serving.events),
            },
        },
        "visualization": visualization,
        "component_timeseries": component_timeseries,
        "validation_warnings": list(result.validation_warnings),
        "validation_information": list(result.validation_information),
    }


def _static_kv_report(result: StaticRunResult) -> Dict[str, Any]:
    """Build logical/physical KV counters for the static task graph."""

    plan = compile_serving_plan(result.scenario)
    execution = result.execution
    event_counts = dict(execution.kv_event_counts)
    logical_event_bytes = dict(execution.kv_logical_event_bytes)
    physical_event_bytes = dict(execution.kv_physical_event_bytes)
    phase_bytes = dict(execution.kv_phase_bytes)
    workload = result.scenario.workload
    if workload.requests:
        live_tokens = [
            max(0, int(request.prompt_tokens))
            + max(0, int(request.output_tokens) - 1)
            for request in workload.requests
        ]
        peak_pages = sum(
            int(math.ceil(tokens / float(plan.kv_policy.tokens_per_page)))
            for tokens in live_tokens
            if tokens > 0 and plan.kv_policy.tokens_per_page > 0
        )
        max_live_tokens = max(live_tokens, default=0)
    else:
        request_count = max(0, int(workload.effective_request_count))
        max_live_tokens = (
            max(0, int(workload.prompt_tokens))
            + max(0, int(workload.output_tokens) - 1)
            if request_count > 0
            else 0
        )
        pages_per_request = (
            int(
                math.ceil(
                    max_live_tokens / float(plan.kv_policy.tokens_per_page)
                )
            )
            if max_live_tokens > 0 and plan.kv_policy.tokens_per_page > 0
            else 0
        )
        peak_pages = request_count * pages_per_request
    peak_pages = min(peak_pages, max(0, int(plan.kv_policy.capacity_pages)))

    def phase_value(phase: str, event: str, semantics: str) -> int:
        return phase_bytes.get((phase, event, semantics), 0)

    return {
        "mode": "static",
        "policy": to_primitive(result.scenario.placement.kv_policy),
        "tokens_per_page": int(plan.kv_policy.tokens_per_page),
        "bytes_per_page": int(plan.kv_policy.bytes_per_page),
        "logical_bytes_per_token": int(plan.kv_policy.logical_bytes_per_token),
        "physical_bytes_per_token": int(plan.kv_policy.bytes_per_page)
        // max(1, int(plan.kv_policy.tokens_per_page)),
        "capacity_pages": int(plan.kv_policy.capacity_pages),
        "capacity_bytes": int(plan.kv_policy.capacity_bytes),
        "peak_used_pages": peak_pages,
        "peak_used_bytes": peak_pages * int(plan.kv_policy.bytes_per_page),
        "peak_semantics": "conservative_all_materialized_requests_prompt_plus_output_minus_one",
        "max_live_tokens_per_request": max_live_tokens,
        "event_task_counts": dict(sorted(event_counts.items())),
        "logical_event_bytes": dict(sorted(logical_event_bytes.items())),
        "physical_event_bytes": dict(sorted(physical_event_bytes.items())),
        "logical_prefill_read_bytes": phase_value("prefill", "kv_read", "logical"),
        "logical_prefill_write_bytes": phase_value("prefill", "kv_append", "logical"),
        "logical_decode_read_bytes": phase_value("decode", "kv_read", "logical"),
        "logical_decode_write_bytes": phase_value("decode", "kv_append", "logical"),
        "logical_decode_append_bytes": phase_value("decode", "kv_append", "logical"),
        "physical_prefill_read_bytes": phase_value("prefill", "kv_read", "physical"),
        "physical_prefill_write_bytes": phase_value("prefill", "kv_append", "physical"),
        "physical_decode_read_bytes": phase_value("decode", "kv_read", "physical"),
        "physical_decode_write_bytes": phase_value("decode", "kv_append", "physical"),
        "physical_decode_append_bytes": phase_value("decode", "kv_append", "physical"),
        "prefetch_events": event_counts.get("kv_prefetch", 0),
        "prefetch_bytes": physical_event_bytes.get("kv_prefetch", 0),
        "offload_events": event_counts.get("kv_offload", 0),
        "offload_bytes": physical_event_bytes.get("kv_offload", 0),
        "logical_offload_bytes": logical_event_bytes.get("kv_offload", 0),
        "physical_offload_bytes": physical_event_bytes.get("kv_offload", 0),
        "migration_events": event_counts.get("kv_prefetch", 0)
        + event_counts.get("kv_offload", 0),
        "migration_bytes": physical_event_bytes.get("kv_prefetch", 0)
        + physical_event_bytes.get("kv_offload", 0),
        "logical_migration_bytes": logical_event_bytes.get("kv_prefetch", 0)
        + logical_event_bytes.get("kv_offload", 0),
        "prefetch_distance_modeled": False,
        "prefetch_distance_semantics": _prefetch_semantics(result.scenario),
        "traffic_semantics": "logical_kv_traffic_separate_from_resource_accounting",
        "modeling_limits": [
            "offload_ratio bounds pressure-triggered migratable capacity; it is not a per-token mirror ratio",
            _prefetch_semantics(result.scenario)["description"],
            "static peak is conservative across all materialized requests rather than a reconstructed allocation timeline",
        ],
    }


def online_summary_dict(result: OnlineScenarioResult) -> Dict[str, Any]:
    """Serialize only the shared online summary and bounded request rows."""

    if not isinstance(result, OnlineScenarioResult):
        raise TypeError("online_summary_dict requires an OnlineScenarioResult")
    with _compilation_scope(result.scenario):
        core = _online_report_core(result)
    return {"summary": core.summary, "requests": core.requests}


def report_dict(
    result: RunResult,
    *,
    visualization_offset: int = 0,
    visualization_limit: int = DEFAULT_VISUALIZATION_EVENT_LIMIT,
    visualization_memory_segment_limit: int = DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT,
) -> Dict[str, Any]:
    """Serialize a bounded report and replay-oriented visualization page."""

    with _compilation_scope(result.scenario):
        return _report_dict_in_context(
            result,
            visualization_offset=visualization_offset,
            visualization_limit=visualization_limit,
            visualization_memory_segment_limit=visualization_memory_segment_limit,
        )


def _report_dict_in_context(
    result: RunResult,
    *,
    visualization_offset: int = 0,
    visualization_limit: int = DEFAULT_VISUALIZATION_EVENT_LIMIT,
    visualization_memory_segment_limit: int = DEFAULT_VISUALIZATION_MEMORY_SEGMENT_LIMIT,
) -> Dict[str, Any]:
    options = VisualizationTraceOptions(
        event_offset=visualization_offset,
        event_limit=visualization_limit,
        memory_segment_limit=visualization_memory_segment_limit,
    )
    if isinstance(result, OnlineScenarioResult):
        return _online_report_dict(result, options)
    execution = result.execution
    request_rows: Dict[str, Any] = {}
    all_tbt: List[float] = []
    all_tpot: List[float] = []
    ttft: List[float] = []
    e2e: List[float] = []
    ordered_request_ids = sorted(result.metrics.request_metrics)
    if (
        execution.retention_policy is not RetentionPolicy.EXACT
        and len(ordered_request_ids) > _SCALABLE_REQUEST_DETAIL_LIMIT
    ):
        leading = _SCALABLE_REQUEST_DETAIL_LIMIT // 2
        trailing = _SCALABLE_REQUEST_DETAIL_LIMIT - leading
        selected_request_ids = set(
            ordered_request_ids[:leading] + ordered_request_ids[-trailing:]
        )
    else:
        selected_request_ids = set(ordered_request_ids)
    for request_id in ordered_request_ids:
        item = result.metrics.request_metrics[request_id]
        if request_id in selected_request_ids:
            request_rows[request_id] = to_primitive(item)
        if item.ttft_ns is not None:
            ttft.append(item.ttft_ns)
        if item.e2e_ns is not None:
            e2e.append(item.e2e_ns)
        if item.tpot_ns is not None:
            all_tpot.append(item.tpot_ns)
        all_tbt.extend(item.tbt_ns)

    energy_pj = execution.total_energy_pj
    moved_bytes = execution.resource_accounted_bytes
    utilization = dict(result.metrics.resource_utilization)
    bottleneck = max(utilization.items(), key=lambda item: (item[1], item[0])) if utilization else None
    parallel = result.scenario.placement.parallel
    mtp_policy = result.scenario.workload.mtp
    proposed_tokens = execution.proposed_tokens
    accepted_tokens = execution.accepted_tokens
    committed_tokens = accepted_tokens
    state_event_counts = dict(execution.state_event_counts)
    state_event_bytes = dict(execution.state_event_bytes)

    model_coverage = _model_coverage(result.scenario)
    state_bytes_per_request = int(
        model_coverage["linear_state_bytes_per_request"]
    )
    static_peak_state_bytes = (
        state_bytes_per_request
        * result.scenario.workload.effective_request_count
    )
    analytical_coverage = {
        "evidence": "analytical",
        "calibration_version": ANALYTICAL_MODEL_VERSION,
        "model": model_coverage,
        "runtime": execution.analytical_coverage,
    }
    mtp_model = analytical_coverage["model"]["mtp"]
    result_semantics = _static_result_semantics(result)

    return {
        "scenario": result.scenario.name,
        "manifest": to_primitive(result.trace.manifest),
        "execution_mode": "static",
        "retention_policy": result.retention_policy,
        "result_semantics": result_semantics,
        "measurement_semantics": result_semantics,
        "trace_fidelity": {
            RetentionPolicy.EXACT: TraceFidelity.EXACT.value,
            RetentionPolicy.STREAMING: TraceFidelity.REPRESENTATIVE.value,
            RetentionPolicy.AGGREGATE: TraceFidelity.AGGREGATE.value,
        }[execution.retention_policy],
        "summary": {
            "makespan_ns": result.trace.makespan_ns,
            "task_count": execution.task_count,
            **_host_output_contract_projection(result.scenario),
            "total_energy_pj": energy_pj,
            "resource_accounted_bytes": moved_bytes,
            "throughput": dict(result.metrics.throughput),
            "ttft_ns": _percentiles(ttft),
            "tbt_ns": _percentiles(all_tbt),
            "tpot_ns": _percentiles(all_tpot),
            "e2e_ns": _percentiles(e2e),
            "bottleneck_resource": (
                {"resource_id": bottleneck[0], "utilization": bottleneck[1]}
                if bottleneck
                else None
            ),
            "mtp": {
                "enabled": bool(mtp_policy and mtp_policy.enabled),
                "prediction_layers": mtp_model["prediction_layers"],
                "auxiliary_head": mtp_model["auxiliary_head"],
                "declared_weight_bytes": mtp_model["declared_weight_bytes"],
                "operator_ids": mtp_model["operator_ids"],
                "weight_tensor_ids": mtp_model["weight_tensor_ids"],
                "method": getattr(mtp_policy, "method", "disabled"),
                "candidate_tokens": int(
                    getattr(mtp_policy, "candidate_tokens", 1)
                ),
                "candidate_tokens_semantics": "max_draft_tokens",
                "verifier_width_at_max": 1
                + int(getattr(mtp_policy, "candidate_tokens", 1)),
                "min_draft_tokens": int(
                    getattr(mtp_policy, "min_draft_tokens", 0)
                ),
                "continuation_threshold": getattr(
                    mtp_policy, "continuation_threshold", None
                ),
                "proposal_length_model": getattr(
                    mtp_policy, "proposal_length_model", "max"
                ),
                "expected_draft_tokens_per_round": getattr(
                    mtp_policy, "expected_draft_tokens_per_round", None
                ),
                "draft_length_trace": list(
                    getattr(mtp_policy, "draft_length_trace", ())
                ),
                "proposed_tokens": proposed_tokens,
                "accepted_tokens": accepted_tokens,
                "committed_tokens": committed_tokens,
                "rejected_tokens": max(0, proposed_tokens - accepted_tokens),
                "effective_acceptance_rate": (
                    accepted_tokens / proposed_tokens if proposed_tokens else None
                ),
            },
            "parallel": {
                "tp_degree": parallel.tp_degree,
                "pp_degree": parallel.pp_degree,
                "ep_degree": parallel.ep_degree,
                "world_size": parallel.world_size,
                "collective_algorithm": parallel.collective_algorithm,
                "routing_policy": parallel.routing_policy,
            },
        },
        "requests": request_rows,
        "resource_utilization": utilization,
        "category_time_ns": {
            category.value: value
            for category, value in result.metrics.category_time_ns.items()
        },
        "critical_path_category_ns": {
            category.value: value
            for category, value in result.metrics.critical_path_category_ns.items()
        },
        "kv_cache": _static_kv_report(result),
        "linear_state": {
            "mode": "static",
            "cache_component": result.scenario.placement.tensor_to_component.get(
                "linear_state"
            ),
            "offload_component": result.scenario.placement.tensor_to_component.get(
                "linear_state_offload"
            ),
            "bytes_per_request": state_bytes_per_request,
            "peak_used_bytes": static_peak_state_bytes,
            "peak_semantics": "conservative_all_materialized_requests",
            "event_task_counts": dict(sorted(state_event_counts.items())),
            "event_bytes": dict(sorted(state_event_bytes.items())),
        },
        "analytical_coverage": analytical_coverage,
        "visualization": _visualization_payload(result, options),
        "component_timeseries": _component_timeseries(result),
        "report_limits": {
            "requests": {
                "total": len(ordered_request_ids),
                "returned": len(request_rows),
                "limit": (
                    _SCALABLE_REQUEST_DETAIL_LIMIT
                    if execution.retention_policy is not RetentionPolicy.EXACT
                    else None
                ),
                "truncated": len(request_rows) < len(ordered_request_ids),
            },
            "retained_tasks": {
                "total": execution.task_count,
                "returned": len(result.trace.tasks),
                "limit": execution.retained_task_limit,
                "truncated": len(result.trace.tasks) < execution.task_count,
            },
        },
        "validation_warnings": list(result.validation_warnings),
        "validation_information": list(result.validation_information),
    }


def format_report(result: RunResult) -> str:
    data = report_dict(result)
    summary = data["summary"]
    manifest = data["manifest"]
    lines = [
        "场景：{}".format(result.scenario.name),
        "运行标识：{}".format(manifest["run_id"]),
        "证据等级：{}".format(manifest["evidence"]),
        "执行模式：{}".format(data.get("execution_mode", "static")),
        "事件保留策略：{}".format(data.get("retention_policy", "exact")),
        "任务数：{}".format(summary["task_count"]),
        "总时长：{:.3f} ms".format(summary["makespan_ns"] / 1_000_000.0),
    ]
    throughput = summary["throughput"]
    lines.append(
        "吞吐：{:.3f} 请求/秒，{:.3f} 可见 Token/秒".format(
            throughput["requests_per_s"],
            throughput["visible_output_tokens_per_s"],
        )
    )
    for request_id, request in data["requests"].items():
        lines.append(
            "{}：TTFT={} ms，TPOT={} ms，E2E={} ms，Token 数={}".format(
                request_id,
                _fmt_ms(request.get("ttft_ns")),
                _fmt_ms(request.get("tpot_ns")),
                _fmt_ms(request.get("e2e_ns")),
                request.get("visible_output_tokens", 0),
            )
        )
    bottleneck = summary["bottleneck_resource"]
    if bottleneck:
        lines.append(
            "最高利用率资源：{}（{:.1%}）".format(
                bottleneck["resource_id"], bottleneck["utilization"]
            )
        )
    lines.append("资源利用率：")
    for resource_id, value in sorted(
        data["resource_utilization"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append("  {:28s} {:8.2%}".format(resource_id, value))
    if result.validation_warnings:
        lines.append("警告：")
        lines.extend("  - {}".format(item) for item in result.validation_warnings)
    if result.validation_information:
        lines.append("信息：")
        lines.extend("  - {}".format(item) for item in result.validation_information)
    return "\n".join(lines)


def _fmt_ms(value_ns: Optional[float]) -> str:
    return "NA" if value_ns is None else "{:.3f}".format(value_ns / 1_000_000.0)


__all__ = [
    "OnlineScenarioResult",
    "RunResult",
    "ScenarioResult",
    "compare_with_gpu_baseline",
    "format_comparison",
    "format_report",
    "gpu_baseline_scenario",
    "online_batch_trace_page",
    "online_summary_dict",
    "page_online_batch_trace",
    "replay_online_batch_trace",
    "report_dict",
    "run_scenario",
]
