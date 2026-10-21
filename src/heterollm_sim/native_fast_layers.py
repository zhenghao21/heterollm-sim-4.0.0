"""Orchestrate isolated native layer windows without committing a cohort.

Paid PD/admission and ordinary native IR preparation are shared with serving.
The layer executor observes independent, boundary-conditioned physical graphs;
its unexecuted layers never become physical events or resumable Token state.
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import time

from .execution_control import ExecutionControl, _NativeLayerSamplingCutoff, NativeFastWallBudgetError
from .native_fast_window import (NativeLayerWindowSample, NativeWindowForecast,
                                 _checked_budget, _eligibility_reasons)
from .planner import _compilation_scope
from .serving import _OnlineRuntime, compile_serving_plan


class _LayerWallLimit(RuntimeError):
    pass


_EXPANDABLE_BUDGET_REASONS = frozenset((
    "native_layer_conditional_single_or_independent_pair_budget_exceeded",
    "native_layer_window_conditional_single_or_independent_pair_budget_exceeded",
    "native_layer_multislot_conditional_single_or_independent_pair_budget_exceeded",
    *("native_window_directional_pair_error_exceeds_target:"+metric for metric in (
        "mean_complete_request_tpot_ns", "mean_complete_request_ingress_ttft_ns",
        "cluster_tokens_per_second")),
))


def _can_expand_layer_budget(budget):
    reasons = budget.get("reasons", ())
    return (budget.get("accepted") is not True and bool(reasons)
            and all(reason in _EXPANDABLE_BUDGET_REASONS for reason in reasons))


def _typed_samples(records):
    names = tuple(field.name for field in fields(NativeLayerWindowSample))
    return tuple(record if isinstance(record, NativeLayerWindowSample) else
                 NativeLayerWindowSample(**{name: record[name] for name in names})
                 for record in records)


def simulate_native_layer_windows(scenario, *, batch_lowerer, execution_control=None,
        residency_manager=None, execution_kernel=None, runtime_origin_ns=0.,
        custom_lowerer=False, paired_reference=None):
    from .fast_simulation import FastSimulationOutcome
    from .native_layer_sampling import (prepare_native_layer_windows,
        sample_native_layer_windows, NativeLayerSamplingGuardError)
    from .native_layer_budget import evaluate_native_layer_window

    options = scenario.simulation.native_window
    if options is None or options.sampling_unit != "layer_windows":
        raise ValueError("native layer sampling requires explicit layer_windows configuration")
    started = time.perf_counter()
    deadline = started + options.max_wall_seconds
    control = execution_control or ExecutionControl()
    diagnostics = {"method": "native_boundary_conditioned_layer_windows/v1",
        "requested_mode": "fast", "effective_mode": "forecast_rejected",
        "sampling_unit": "layer_windows", "sampling_options": asdict(options),
        "target_relative_error": options.target_relative_error,
        "validation_status": "conditional_budget_not_full_window_validation",
        "accuracy_target_is_guarantee": False,
        "physical_tail_transactions_recorded": False,
        "future_physical_state_advanced": False,
        "physical_runtime_resumable_from_forecast": False,
        "sampled_decode_cohorts": 0, "sampled_visible_output_tokens": 0,
        "predicted_decode_cohorts": 0, "rejection_reasons": [], "fallback_reasons": []}

    def cancel():
        control.raise_if_cancelled()
        if time.perf_counter() >= deadline:
            raise _LayerWallLimit("native layer windows exceeded the candidate time budget")
        return False

    if type(control) is ExecutionControl and control.cancellation_callback is None:
        # The native file/deadline checks cannot edit a physical calendar.
        # Keep them native so ordinary progress and stop support do not force
        # every PD phase into the unknown-callback rollback/publication path.
        layer_control = ExecutionControl(progress_callback=control.progress_callback,
            progress_queue=control.progress_queue, stop_request_path=control.stop_request_path,
            deadline_monotonic=min(deadline, control.deadline_monotonic)
                if control.deadline_monotonic is not None else deadline)
    else:
        layer_control = ExecutionControl(progress_callback=control.progress_callback,
            progress_queue=control.progress_queue, cancellation_callback=cancel)
    samples, payload, prepared, runtime = (), {}, {}, None
    attempts = []
    diagnostics["layer_sampling_attempts"] = attempts
    current_options = options
    attempt_started = None

    def preserve_attempt(budget, state):
        nonlocal attempt_started
        if attempt_started is None:
            return
        elapsed = time.perf_counter()-attempt_started
        attempts.append({"attempt_index": len(attempts),
            "layer_window_size": current_options.layer_window_size,
            "status": state, "wall_seconds": elapsed,
            "candidate_elapsed_wall_seconds": time.perf_counter()-started,
            "complete_physical_layer_windows": len(samples),
            "executed_physical_layers": sum(len(row.layer_ids) for row in samples),
            "physical_layer_samples": [row.to_metadata() for row in samples],
            "budget": budget, "prepared_source_IR_reused": True,
            "non_layer_samples_reused": payload.get("non_layer_samples_reused", False),
            "calibration_scope": "only_this_attempt; overlapping_earlier_attempts_not_independent_holdout",
            "physical_complete_cohorts": 0, "physical_visible_output_tokens": 0})
        attempt_started = None
    context = getattr(batch_lowerer, "_compilation_context", None)
    with _compilation_scope(scenario, context):
        layer_control.raise_if_cancelled()
        plan = compile_serving_plan(scenario)
        reasons = _eligibility_reasons(scenario, plan, custom_lowerer=custom_lowerer,
                                      shared_residency=residency_manager is not None)
        if reasons:
            diagnostics.update(rejection_reasons=list(reasons), fallback_reasons=list(reasons),
                               wall_seconds=time.perf_counter()-started)
            return FastSimulationOutcome(diagnostics=diagnostics)
        try:
            runtime = _OnlineRuntime(plan, batch_lowerer, execution_control=layer_control,
                residency_manager=residency_manager, execution_kernel=execution_kernel,
                runtime_origin_ns=runtime_origin_ns)

            def sample(active, cohort, cost, stages, request_ready_ns):
                layer_control.raise_if_cancelled()
                prepared.update(cohort=cohort, cost=cost, stages=stages,
                    request_ready_ns=request_ready_ns,
                    sampling=prepare_native_layer_windows(active, cohort, cost,
                        stages, request_ready_ns, layer_control))
                layer_control.raise_if_cancelled()
                raise _NativeLayerSamplingCutoff()

            runtime._native_layer_sampler = sample
            try:
                runtime.run_until_decode_steps(1, strict_cohort_cutoff=True)
            except _NativeLayerSamplingCutoff:
                pass
            else:
                raise RuntimeError("native layer sampling unexpectedly committed an ordinary cohort")
            maximum = options.max_layer_window_size or options.layer_window_size
            while True:
                layer_control.raise_if_cancelled()
                attempt_started = time.perf_counter()
                payload, samples = {}, ()
                runtime._native_layer_sample_records = []
                runtime._native_layer_sampling_attempt = {
                    "attempt_index": len(attempts),
                    "layer_window_size": current_options.layer_window_size,
                    "previous_completed_windows": sum(row["complete_physical_layer_windows"] for row in attempts),
                    "previous_executed_tasks": sum(sample["executed_task_count"]
                        for row in attempts for sample in row["physical_layer_samples"])}
                layer_control.report("native_layer_sampling_attempt", len(attempts),
                    message="开始连续原生层窗口采样，沿用已完成的数据入站和总计时",
                    metadata={"layer_sampling_attempt": runtime._native_layer_sampling_attempt})
                layer_control.raise_if_cancelled()
                payload = sample_native_layer_windows(runtime, prepared["cohort"],
                    prepared["cost"], prepared["stages"], prepared["request_ready_ns"],
                    current_options, layer_control, prepared=prepared["sampling"])
                if getattr(plan.scheduler, "max_num_seqs", 1) > 1:
                    from .native_layer_anchors import add_native_seed_shape_anchor
                    payload = add_native_seed_shape_anchor(runtime, prepared["cohort"],
                        prepared["cost"], prepared["stages"], prepared["request_ready_ns"],
                        payload, current_options, layer_control)
                samples = _typed_samples(payload.get("records", ()))
                layer_control.raise_if_cancelled()
                budget = _checked_budget(evaluate_native_layer_window(runtime,
                    prepared["cohort"], prepared["cost"], prepared["stages"], payload,
                    current_options, paired_reference=paired_reference))
                layer_control.raise_if_cancelled()
                preserve_attempt(budget, "accepted" if budget.get("accepted") else "budget_rejected")
                if (budget.get("accepted") is True or not _can_expand_layer_budget(budget)
                        or current_options.layer_window_size >= maximum):
                    break
                current_options = replace(current_options,
                    layer_window_size=min(maximum, current_options.layer_window_size*2))
        except (_LayerWallLimit, NativeFastWallBudgetError):
            if runtime is not None:
                samples = _typed_samples(getattr(runtime, "_native_layer_sample_records", ()))
            budget = {"accepted": False, "reasons": ["native_layer_sample_wall_budget_exhausted"]}
            preserve_attempt(budget, "wall_budget_exhausted")
        except NativeLayerSamplingGuardError as error:
            if runtime is not None:
                samples = _typed_samples(getattr(runtime, "_native_layer_sample_records", ()))
            budget = {"accepted": False, "reasons": [str(error)]}
            preserve_attempt(budget, "sampling_guard_rejected")
        except Exception as error:
            if runtime is not None:
                samples = _typed_samples(getattr(runtime, "_native_layer_sample_records", ()))
            preserve_attempt({"accepted": False, "reasons": [str(error)]}, "execution_stopped")
            error.native_layer_sampling_attempts = attempts
            raise

        if runtime is not None:
            diagnostics["physical_ingress_accounting"] = {**runtime._external_ingress_report(),
                "first_output_status": "not_committed_by_isolated_layer_samples"}
            diagnostics["physical_boundary_accounting"] = {
                "non_layer_samples": payload.get("non_layer_samples", ()),
                "layer_composition": payload.get("layer_composition", {}),
                "seed_shape_anchor_boundaries": [{"batch_size": anchor["batch_size"],
                    **{key: anchor["sample_payload"].get(key) for key in (
                        "non_layer_samples", "layer_composition", "source_cutoff_ns")}}
                    for anchor in payload.get("batch_shape_anchors", ())],
                "full_cohort_committed": False,
                "visible_output_tokens": 0,
            }
            diagnostics["physical_admission_cutoff_ns"] = runtime.now
        reasons = list(budget.get("reasons", ()))
        diagnostics.update(error_budget=budget.get("error_budget", {}),
            semantic_guards=budget.get("semantic_guards", {}),
            rejection_reasons=reasons, fallback_reasons=reasons,
            physical_layer_window_count=len(samples),
            physical_layer_windows_all_attempts=sum(row["complete_physical_layer_windows"] for row in attempts),
            final_layer_window_size=(attempts[-1]["layer_window_size"] if attempts
                                     else options.layer_window_size),
            physical_complete_cohort_count=0, physical_visible_output_token_count=0,
            wall_seconds=time.perf_counter()-started)
        try:
            control.raise_if_cancelled()
        except NativeFastWallBudgetError:
            # Preserve finished samples when the outer deadline precedes
            # this local sampler deadline. A user pause remains distinct.
            budget = {**budget, "accepted": False}
            diagnostics.update(rejection_reasons=[*reasons, "native_total_candidate_wall_budget_exhausted"],
                fallback_reasons=[*reasons, "native_total_candidate_wall_budget_exhausted"])
        if time.perf_counter() >= deadline:
            budget = {**budget, "accepted": False}
            diagnostics.update(rejection_reasons=[*reasons, "native_layer_result_wall_budget_exhausted"],
                fallback_reasons=[*reasons, "native_layer_result_wall_budget_exhausted"])
        if budget.get("accepted") is not True:
            control.report("native_fast_rejected", len(samples), unit="work_units",
                message="原生层样本已保留，窗口预测未通过预算；不执行详细尾段")
            return FastSimulationOutcome(diagnostics=diagnostics, layer_window_samples=samples)
        ingress = getattr(runtime, "_external_ingress", None) or scenario.workload.metadata["external_kv_ingress"]
        count = ingress["warmup_decode_cohorts"]+ingress["measurement_decode_cohorts"]
        forecast = NativeWindowForecast(runtime.now, 0, count, 0,
            budget["predicted_visible_output_tokens"],
            {"warmup_decode_cohorts": ingress["warmup_decode_cohorts"],
             "measurement_decode_cohorts": ingress["measurement_decode_cohorts"],
             "require_full_batch": ingress.get("require_full_batch", True)},
            budget["window"], budget["metrics"], budget["metric_bounds"],
            tuple(budget.get("request_forecasts", ())), budget["error_budget"],
            sampling_unit="layer_windows")
        diagnostics.update(effective_mode="fast", predicted_decode_cohorts=count,
            predicted_visible_output_tokens=forecast.predicted_visible_output_tokens)
        control.report("native_layer_forecast_complete", len(samples), unit="work_units",
            message="原生层窗口和预测结果分别保存；没有提交完整批次或Token")
        outer_expired = False
        try:
            control.raise_if_cancelled()
        except NativeFastWallBudgetError:
            outer_expired = True
        if outer_expired or time.perf_counter() >= deadline:
            diagnostics.update(effective_mode="forecast_rejected",
                rejection_reasons=["native_layer_publication_wall_budget_exhausted"],
                fallback_reasons=["native_layer_publication_wall_budget_exhausted"],
                predicted_decode_cohorts=0, predicted_visible_output_tokens=0,
                wall_seconds=time.perf_counter()-started)
            return FastSimulationOutcome(diagnostics=diagnostics, layer_window_samples=samples)
        diagnostics["wall_seconds"] = time.perf_counter()-started
        return FastSimulationOutcome(diagnostics=diagnostics,
            native_window_forecast=forecast, layer_window_samples=samples)


__all__ = ["simulate_native_layer_windows"]
