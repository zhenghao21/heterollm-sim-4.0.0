"""Bounded samples of the existing native continuous-serving backend.

The live runtime stops at a real cohort boundary. A window projection is a
separate value; neither the native requests nor any physical calendar advance
through its predicted suffix. Rejected forecasts never run a detailed suffix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping

from .execution_control import ExecutionControl, NativeFastWallBudgetError
from .planner import _compilation_scope
from .serving import _OnlineRuntime, compile_serving_plan


@dataclass(frozen=True)
class NativeWindowForecast:
    cutoff_ns: float
    sampled_decode_cohorts: int
    predicted_decode_cohorts: int
    sampled_visible_output_tokens: int
    predicted_visible_output_tokens: int
    original_window: Mapping[str, Any]
    measurement_window: Mapping[str, Any]
    metrics: Mapping[str, Any]
    metric_bounds: Mapping[str, Any]
    request_forecasts: tuple[Mapping[str, Any], ...]
    error_budget: Mapping[str, Any]
    sampling_unit: str = "cohorts"

    def __post_init__(self):
        if not isinstance(self.sampling_unit, str) or self.sampling_unit not in {"cohorts", "layer_windows"}:
            raise ValueError("native window forecast requires an explicit supported sampling unit")
        if self.sampling_unit == "layer_windows" and (self.sampled_decode_cohorts or self.sampled_visible_output_tokens):
            raise ValueError("native layer windows cannot claim completed Decode cohorts or visible output Tokens")

    def to_metadata(self) -> dict:
        return {"schema": "native-continuous-window-forecast/v1", **asdict(self),
                "controller_state": ("discarded_after_boundary_conditioned_layer_windows"
                    if self.sampling_unit == "layer_windows" else
                    "discarded_at_sample_cutoff_not_advanced_to_forecast_end"),
                "forecast_has_physical_trace": False}


@dataclass(frozen=True)
class NativeLayerWindowSample:
    """One completely executed physical layer window, not a complete Token.

    Times and transaction observations refer only to these real tasks. Any
    imported KV or boundary initialization is accounted separately; no
    request/cohort completion can be inferred from this record.
    """
    window_id: str
    role: str
    cohort_id: str
    cohort_kind: str
    decode_step: int
    layer_ids: tuple[str, ...]
    rank_ids: tuple[int, ...]
    context_tokens_by_request: Mapping[str, int]
    executed_task_count: int
    start_ns: float
    end_ns: float
    wall_seconds: float
    physical_observations: Mapping[str, Any]

    def __post_init__(self):
        if (type(self.window_id) is not str or not self.window_id
                or type(self.cohort_id) is not str or not self.cohort_id
                or type(self.role) is not str or self.role not in {"calibration", "validation", "exhaustive"}):
            raise ValueError("native layer sample requires an identity and calibration, validation or exhaustive role")
        if type(self.cohort_kind) is not str or self.cohort_kind not in {"prefill", "decode"}:
            raise ValueError("native layer sample requires an ordinary prefill or decode source cohort")
        if type(self.decode_step) is not int or self.decode_step < 0:
            raise ValueError("native layer sample decode_step must be a nonnegative integer")
        if (not isinstance(self.layer_ids, tuple) or not self.layer_ids
                or any(type(value) is not str or not value for value in self.layer_ids)
                or len(set(self.layer_ids)) != len(self.layer_ids)):
            raise ValueError("native layer sample requires distinct native layer ids")
        if (not isinstance(self.rank_ids, tuple) or not self.rank_ids
                or any(type(value) is not int or value < 0 for value in self.rank_ids)
                or len(set(self.rank_ids)) != len(self.rank_ids)):
            raise ValueError("native layer sample requires distinct nonnegative rank ids")
        if (not isinstance(self.context_tokens_by_request, Mapping) or not self.context_tokens_by_request
                or any(type(key) is not str or not key or type(value) is not int or value < 0
                       for key, value in self.context_tokens_by_request.items())):
            raise ValueError("native layer sample requires explicit request context positions")
        if type(self.executed_task_count) is not int or self.executed_task_count < 1:
            raise ValueError("native layer sample requires a positive executed task count")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in (self.start_ns, self.end_ns, self.wall_seconds))
                or self.start_ns < 0 or self.end_ns < self.start_ns or self.wall_seconds < 0):
            raise ValueError("native layer sample requires finite ordered physical times and wall duration")
        if not isinstance(self.physical_observations, Mapping):
            raise ValueError("native layer sample observations must be a mapping")

    def to_metadata(self):
        return {"schema": "native-physical-layer-window-sample/v1", **asdict(self),
            "window_fully_executed": True, "complete_native_cohort": False,
            "visible_output_tokens_observed": 0,
            "validation_semantics": ("all_native_members_of_this_layer_class_physically_executed"
                if self.role == "exhaustive" else "disjoint_window_holdout" if self.role == "validation"
                else "calibration_window_only"),
            "state_scope": "boundary_conditioned_native_layer_window_only",
            "physical_trace_scope": "only_executed_window_tasks"}


class _SampleWallLimit(RuntimeError):
    pass


def _sample_physical_observations(metadata):
    """Keep budget observations bounded independently of physical bank count.

    The original BatchCost and physical prefix retain their complete bill.
    Forecast records need scalar traffic totals, not another per-resource
    ledger repeated in every sampling attempt and budget diagnostic.
    """
    observations = {}
    for name in ("storage_traffic", "dram_traffic", "sram_traffic"):
        traffic = metadata.get(name)
        if isinstance(traffic, Mapping):
            observations[name] = {key: value for key, value in traffic.items()
                if value is None or type(value) in (str, bool, int, float)}
    for name in ("storage_gc_observations", "task_count", "physical_execution",
                 "native_fast_budget_observations"):
        if name in metadata:
            observations[name] = metadata[name]
    return observations


def cohort_samples(runtime) -> tuple[dict, ...]:
    """Record only committed native cohorts and real visible-token events."""
    visible, commits = {}, {}
    for event in runtime.events:
        if event.event_type == "tokens_committed":
            count = event.details.get("visible_tokens", 0)
            if type(count) is not int or count < 0:
                raise ValueError("native Token commits require nonnegative integer visible counts")
            visible[event.cohort_id] = visible.get(event.cohort_id, 0) + count
            commits.setdefault(event.cohort_id, []).append({
                "request_id": event.request_id, "timestamp_ns": event.timestamp_ns,
                "visible_tokens": count, "committed": event.details.get("committed"),
                "source": event.details.get("source")})
    records, previous = [], float(runtime._runtime_origin_ns)
    for index, batch in enumerate(runtime.batches, 1):
        end, start = float(batch.end_ns), float(batch.start_ns)
        if not all(math.isfinite(value) for value in (start, end, previous)) or end < start or end < previous:
            raise ValueError("native sample cohorts require ordered finite physical boundaries")
        records.append({
            "cohort_index": index, "cohort_id": batch.cohort_id, "kind": batch.kind,
            "request_ids": list(batch.request_ids), "start_ns": start, "end_ns": end,
            "commit_interval_ns": end - previous, "execution_ns": end - start,
            "interval_semantics": "cohort_end_to_previous_cohort_end_not_request_TPOT",
            "boundary_gap_ns": max(0., start - previous),
            "visible_output_tokens": visible.get(batch.cohort_id, 0),
            "visible_commits": commits.get(batch.cohort_id, []),
            "context_tokens": [{"request_id": item.request_id,
                "context_tokens": item.context_tokens, "committed_tokens": item.committed_tokens,
                "phase": item.phase, "token_count": item.token_count} for item in batch.items],
            "physical_observations": _sample_physical_observations(batch.cost.metadata),
        })
        previous = end
    return tuple(records)


def _eligibility_reasons(scenario, plan, *, custom_lowerer, shared_residency):
    reasons = []
    ingress = scenario.workload.metadata.get("external_kv_ingress")
    if not isinstance(ingress, Mapping):
        reasons.append("native_window_requires_external_prefill_ingress")
    elif (type(ingress.get("warmup_decode_cohorts")) is not int
            or type(ingress.get("measurement_decode_cohorts")) is not int
            or ingress["warmup_decode_cohorts"] < 0 or ingress["measurement_decode_cohorts"] < 1):
        reasons.append("native_window_requires_explicit_positive_measurement_window")
    if custom_lowerer:
        reasons.append("native_window_requires_standard_native_lowerer")
    if shared_residency:
        reasons.append("native_window_shared_residency_not_bounded")
    if plan.mtp.enabled:
        reasons.append("native_window_speculative_decode_not_bounded")
    if plan.scheduler.preemption_enabled:
        reasons.append("native_window_preemption_not_bounded")
    if plan.kv_policy.offload_component or plan.kv_policy.prefetch_distance:
        reasons.append("native_window_offload_or_prefetch_not_bounded")
    return tuple(reasons)


def _checked_budget(budget):
    if not isinstance(budget, Mapping):
        raise TypeError("native window budget must return a mapping")
    if budget.get("accepted") is not True:
        return budget
    invalid = []
    metrics = budget.get("metrics")
    if not isinstance(metrics, Mapping):
        invalid.append("native_window_accepted_budget_metrics_missing")
    else:
        for name in ("cluster_tokens_per_second", "tokens_per_second_per_gpu",
                     "mean_complete_request_tpot_ns", "mean_complete_request_ingress_ttft_ns",
                     "measurement_duration_ns"):
            value = metrics.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                invalid.append("native_window_accepted_budget_invalid_" + name)
        for name in ("complete_fresh_requests", "measurement_visible_output_tokens"):
            if type(metrics.get(name)) is not int or metrics[name] < 1:
                invalid.append("native_window_accepted_budget_invalid_" + name)
    predicted = budget.get("predicted_visible_output_tokens")
    if type(predicted) is not int or predicted < 0:
        invalid.append("native_window_accepted_budget_invalid_predicted_visible_output_tokens")
    for name in ("window", "metric_bounds", "error_budget"):
        if not isinstance(budget.get(name), Mapping):
            invalid.append("native_window_accepted_budget_missing_" + name)
    if invalid:
        return {**budget, "accepted": False, "can_expand_sampling": False,
                "reasons": [*budget.get("reasons", ()), *invalid]}
    return budget


def simulate_native_window(scenario, *, batch_lowerer, execution_control=None,
        residency_manager=None, execution_kernel=None, runtime_origin_ns=0.,
        custom_lowerer=False, paired_reference=None):
    from .fast_simulation import FastSimulationOutcome
    from .native_window_budget import evaluate_native_window

    options = scenario.simulation.native_window
    if options is None:
        raise ValueError("native window sampling requires explicit simulation.native_window options")
    control = execution_control or ExecutionControl()
    started = time.perf_counter()
    deadline = started + options.max_wall_seconds
    diagnostics = {
        "method": "native_continuous_physical_cohort_window/v1",
        "requested_mode": "fast", "effective_mode": "forecast_rejected",
        "target_relative_error": options.target_relative_error,
        "validation_status": "conditional_budget_not_full_window_validation",
        "accuracy_target_is_guarantee": False,
        "native_window_prefix_only": True,
        "physical_tail_transactions_recorded": False, "future_physical_state_advanced": False,
        "physical_runtime_resumable_from_forecast": False,
        "sampling_options": asdict(options), "sampling_attempts": [],
        "sampled_decode_cohorts": 0, "predicted_decode_cohorts": 0,
        "fallback_reasons": [],
    }

    def cancel():
        # The user's pause/cancel always wins over our wall-clock budget.
        control.raise_if_cancelled()
        if time.perf_counter() >= deadline:
            raise _SampleWallLimit("native cohort sample exceeded its wall-clock budget")
        return False

    if type(control) is ExecutionControl and control.cancellation_callback is None:
        sampling_control = ExecutionControl(progress_callback=control.progress_callback,
            progress_queue=control.progress_queue, stop_request_path=control.stop_request_path,
            deadline_monotonic=min(deadline, control.deadline_monotonic)
                if control.deadline_monotonic is not None else deadline)
    else:
        sampling_control = ExecutionControl(progress_callback=control.progress_callback,
            progress_queue=control.progress_queue, cancellation_callback=cancel)
    lowerer_context = getattr(batch_lowerer, "_compilation_context", None)
    if getattr(lowerer_context, "scenario", None) is not scenario:
        lowerer_context = None
    with _compilation_scope(scenario, lowerer_context):
        control.raise_if_cancelled()
        plan = compile_serving_plan(scenario)
        reasons = _eligibility_reasons(scenario, plan, custom_lowerer=custom_lowerer,
                                      shared_residency=residency_manager is not None)
        if reasons:
            diagnostics.update(fallback_reasons=list(reasons),
                rejection_reasons=list(reasons), wall_seconds=time.perf_counter() - started)
            return FastSimulationOutcome(diagnostics=diagnostics)
        try:
            sampling_control.raise_if_cancelled()
            runtime = _OnlineRuntime(plan, batch_lowerer, execution_control=sampling_control,
                residency_manager=residency_manager, execution_kernel=execution_kernel,
                runtime_origin_ns=runtime_origin_ns)
        except (_SampleWallLimit, NativeFastWallBudgetError):
            control.raise_if_cancelled()
            diagnostics.update(fallback_reasons=["native_sample_wall_budget_exhausted"],
                rejection_reasons=["native_sample_wall_budget_exhausted"],
                sample_cutoff_kind="native_initialization_aborted",
                partial_native_initialization_discarded=True,
                wall_seconds=time.perf_counter() - started)
            return FastSimulationOutcome(diagnostics=diagnostics)
        target = options.sample_cohorts
        while True:
            try:
                sampling_control.raise_if_cancelled()
                control.report("native_fast_sampling", runtime._completed_decode_steps, target,
                    unit="serving_batches", simulated_time_ns=runtime.now,
                    message="正在执行完整原生批次样本；后续窗口尚未预测")
                runtime.run_until_decode_steps(target, strict_cohort_cutoff=True)
            except (_SampleWallLimit, NativeFastWallBudgetError):
                control.raise_if_cancelled()
                diagnostics.update(fallback_reasons=["native_sample_wall_budget_exhausted"],
                    rejection_reasons=["native_sample_wall_budget_exhausted"],
                    sampled_decode_cohorts=runtime._completed_decode_steps,
                    sample_cutoff_kind="cooperative_abort_may_contain_partial_cohort",
                    partial_physical_cohort_discarded=True,
                    sample_records=list(cohort_samples(runtime)),
                    wall_seconds=time.perf_counter() - started)
                return FastSimulationOutcome(diagnostics=diagnostics)
            samples = cohort_samples(runtime)
            budget = _checked_budget(evaluate_native_window(runtime, samples, options,
                                                            paired_reference=paired_reference))
            control.raise_if_cancelled()
            if time.perf_counter() >= deadline:
                budget = {**budget, "accepted": False, "can_expand_sampling": False,
                    "reasons": [*budget.get("reasons", ()), "native_projection_wall_budget_exhausted"]}
            diagnostics["sampling_attempts"].append({"sampled_decode_cohorts": runtime._completed_decode_steps,
                                                     "budget": budget})
            if (budget.get("accepted") is not True and budget.get("can_expand_sampling") is True
                    and target < options.max_sample_cohorts
                    and not runtime._decode_window.get("complete")):
                target = min(options.max_sample_cohorts, target + options.validation_cohorts)
                continue
            break

        sampled = runtime._result()
        control.raise_if_cancelled()
        # Final native result materialization can itself be significant for a
        # large prefix. It belongs to the candidate budget too, not just the
        # physical execution and projection calculation above.
        if time.perf_counter() >= deadline:
            budget = {**budget, "accepted": False, "can_expand_sampling": False,
                "reasons": [*budget.get("reasons", ()), "native_result_wall_budget_exhausted"]}
        visible = sum(row["visible_output_tokens"] for row in samples)
        diagnostics.update(sampled_decode_cohorts=runtime._completed_decode_steps,
            sampled_visible_output_tokens=visible, sample_records=list(samples),
            physical_sample_cutoff_ns=runtime.now,
            cutoff_kind="complete_native_cohort_boundary",
            sampled_window_complete=sampled.decode_measurement.get("complete") is True,
            cutoff_requests_finished=sum(str(getattr(row.status, "value", row.status)).lower() == "finished"
                                         for row in sampled.request_states.values()),
            error_budget=budget.get("error_budget", {}), semantic_guards=budget.get("semantic_guards", {}),
            rejection_reasons=list(budget.get("reasons", ())),
            fallback_reasons=list(budget.get("reasons", ())),
            wall_seconds=time.perf_counter() - started)
        if budget.get("accepted") is not True:
            control.report("native_fast_rejected", runtime._completed_decode_steps,
                simulated_time_ns=runtime.now, unit="serving_batches",
                message="原生窗口预测未通过预算；保留真实样本，不执行详细尾段",
                metadata={"approximation": diagnostics})
            return FastSimulationOutcome(sampled=sampled, diagnostics=diagnostics)
        ingress = scenario.workload.metadata["external_kv_ingress"]
        original_window = {key: ingress[key] for key in (
            "warmup_decode_cohorts", "measurement_decode_cohorts", "require_full_batch") if key in ingress}
        remaining = max(0, ingress["warmup_decode_cohorts"] + ingress["measurement_decode_cohorts"]
                        - runtime._completed_decode_steps)
        forecast = NativeWindowForecast(runtime.now, runtime._completed_decode_steps, remaining,
            visible, int(budget.get("predicted_visible_output_tokens", 0)), original_window,
            dict(budget.get("window", {})), dict(budget["metrics"]), dict(budget["metric_bounds"]),
            tuple(budget.get("request_forecasts", ())), dict(budget["error_budget"]))
        diagnostics.update(effective_mode="fast", fallback_reasons=[], rejection_reasons=[],
            predicted_decode_cohorts=remaining,
            predicted_visible_output_tokens=forecast.predicted_visible_output_tokens)
        control.report("native_fast_forecast_complete", runtime._completed_decode_steps,
            unit="serving_batches", simulated_time_ns=runtime.now,
            message="原生样本与预测窗口已分别保存；物理状态停留在采样截止点",
            metadata={"approximation": diagnostics})
        control.raise_if_cancelled()
        if time.perf_counter() >= deadline:
            reasons = ["native_publication_wall_budget_exhausted"]
            diagnostics.update(effective_mode="forecast_rejected", fallback_reasons=reasons,
                rejection_reasons=reasons, predicted_decode_cohorts=0,
                predicted_visible_output_tokens=0, wall_seconds=time.perf_counter() - started)
            control.report("native_fast_rejected", runtime._completed_decode_steps,
                simulated_time_ns=runtime.now, unit="serving_batches",
                message="原生候选发布超过时间预算；保留真实样本，不执行详细尾段",
                metadata={"approximation": diagnostics})
            return FastSimulationOutcome(sampled=sampled, diagnostics=diagnostics)
        diagnostics["wall_seconds"] = time.perf_counter() - started
        return FastSimulationOutcome(native_window_forecast=forecast, sampled=sampled,
                                     diagnostics=diagnostics)


__all__ = ["NativeWindowForecast", "NativeLayerWindowSample", "cohort_samples", "simulate_native_window"]
