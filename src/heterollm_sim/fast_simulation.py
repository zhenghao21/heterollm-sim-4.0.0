"""Opt-in prefix-calibrated Decode forecasting.

The original full request is reserved and its contiguous prefix executes on
the detailed kernel.  The unexecuted suffix is a forecast, never a replay of
fabricated physical transactions.  No caller may reuse its cutoff controller
state as if that state had advanced to the forecast endpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from statistics import fmean
from typing import Any, Mapping

from .execution_control import ExecutionControl
from .fast_budget import PairedTimingReference, paired_timing_decision, validate_paired_reference
from .fast_state import hbf_kv_page_period, project_fast_tail
from .planner import _compilation_scope, _execution_layers, _parallel_plan
from .precision import canonical_dtype
from .physical_contract import require_physical_memory_config
from .serving import (
    ServingResult, _OnlineRuntime, compile_serving_plan,
)


@dataclass(frozen=True)
class DecodeSample:
    step: int
    context_tokens: int
    duration_ns: float


@dataclass(frozen=True)
class SamplingWindow:
    warmup: int
    training: int
    holdout: int
    period: int

    @property
    def total(self) -> int:
        return self.warmup + self.training + self.holdout


def sampling_window(options, period: int) -> SamplingWindow:
    """Cover complete flush cycles without forcing an aligned start phase.

    Configured window lengths select fixed sampling. Adaptive sampling uses
    smaller geometry-derived windows, retaining the configured total cap.
    Training is an even number of periods so its halves are comparable.
    """
    if period < 1:
        raise ValueError("Decode page period must be positive")
    if options.sampling_strategy == "fixed":
        return SamplingWindow(options.warmup_tokens, options.calibration_tokens,
                              options.validation_tokens, period)
    def rounded(value, multiple):
        return ((value + multiple - 1) // multiple) * multiple
    training = rounded(max(2 * period, min(options.calibration_tokens, 8)), 2 * period)
    holdout = rounded(max(period, min(options.validation_tokens, 4)), period)
    return SamplingWindow(min(options.warmup_tokens, 4), training, holdout, period)


def expanded_sampling_window(window: SamplingWindow, cap: int) -> SamplingWindow | None:
    """Extend both independent windows; retain the original warmup boundary."""
    candidate = SamplingWindow(window.warmup, window.training + 2 * window.period,
                               window.holdout + window.period, window.period)
    return candidate if candidate.total <= cap else None


def maintenance_recalibration(runtime, sampling: SamplingWindow, options):
    """Select a clean calibration window from observed media events only.

    A Decode cohort can release buffers after its visible commit. Its erase
    work may therefore affect the next commit duration as well. Both cohorts
    remain in the exact prefix; excluding them from the steady-cost fit does
    not remove their latency or the projected future erase service budget.
    """
    decode = [batch for batch in runtime.batches if batch.kind == "decode"]
    erases = [(step, int(batch.cost.metadata.get("storage_traffic", {}).get(
        "erase_operations", 0) or 0)) for step, batch in enumerate(decode, 1)]
    erases = [(step, count) for step, count in erases if count and step >= sampling.warmup]
    if not erases:
        return None
    last_step, last_count = erases[-1]
    cleanup_step = last_step + 1
    warmup = ((cleanup_step + sampling.period - 1) // sampling.period) * sampling.period
    if warmup <= sampling.warmup:
        return None
    fresh = sampling_window(options, sampling.period)
    candidate = SamplingWindow(warmup, fresh.training, fresh.holdout, sampling.period)
    # Preserve the physical reset boundary if a late rejection already has
    # more samples than the minimum new window. Never rewind the runtime or
    # report a warmup that differs from the actual final window slice.
    while candidate.total < len(decode):
        candidate = SamplingWindow(candidate.warmup,
            candidate.training + 2 * candidate.period, candidate.holdout, candidate.period)
    record = {
        "cause": "observed_ordinary_page_zero_erase_and_following_cohort_cleanup",
        "selection_inputs": ["observed_decode_cohort_erase_operations", "hbf_kv_page_period_tokens"],
        "sampled_steps_at_reset": len(decode),
        "old_warmup": sampling.warmup, "new_warmup": candidate.warmup,
        "old_sampling_window": asdict(sampling), "new_sampling_window": asdict(candidate),
        "erase_cohorts": [{"decode_step": step, "erase_operations": count} for step, count in erases],
        "last_erase_decode_step": last_step, "last_erase_operations": last_count,
        "cleanup_through_decode_step": cleanup_step,
        "aligned_cleanup_through_decode_step": warmup,
        "next_target_decode_steps": candidate.total,
        "within_sample_budget": candidate.total <= options.max_sample_tokens,
        "exact_prefix_samples_and_latency_preserved": True,
        "future_erase_service_budget_preserved": True,
        "semantics": "exclude_observed_maintenance_from_baseline_fit_only_not_from_execution_cost",
    }
    return candidate, record


@dataclass(frozen=True)
class ForecastSegment:
    first_decode_step: int
    decode_steps: int
    first_context_tokens: int
    last_context_tokens: int
    duration_ns: float
    model_lower_ns: float
    model_upper_ns: float


@dataclass(frozen=True)
class DecodeForecast:
    request_id: str
    arrival_ns: float
    first_token_ns: float
    cutoff_ns: float
    finish_ns: float
    runtime_origin_ns: float
    prompt_tokens: int
    output_tokens: int
    sampled_decode_steps: int
    predicted_decode_steps: int
    ttft_ns: float
    tpot_ns: float
    segments: tuple[ForecastSegment, ...]
    samples: tuple[DecodeSample, ...]
    storage_projection: Mapping[str, Any]


@dataclass(frozen=True)
class FastSimulationOutcome:
    # Exactly one is populated: a completed detailed result or an independent
    # forecast.  A forecast deliberately has no ServingResult for its suffix.
    detailed: ServingResult | None = None
    forecast: DecodeForecast | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def model_envelope(base_ns: float, anchor_context: float,
                   first_context: int, steps: int) -> tuple[float, float, float]:
    """Integrate the declared T(L)=A+B*L, A,B>=0 research envelope.

    A short prefix cannot identify B reliably at a 64K context.  Use the
    harmonic midpoint of the constant-cost and proportional-cost endpoints
    instead of extrapolating an ill-conditioned fitted slope.  The bounds
    are conditional model bounds, not confidence intervals or hardware laws.
    """
    if steps <= 0:
        return 0.0, 0.0, 0.0
    if (not math.isfinite(base_ns) or base_ns <= 0
            or not math.isfinite(anchor_context) or anchor_context <= 0):
        raise ValueError("forecast requires a positive finite cost and context")
    mean_context = first_context + (steps - 1) / 2.0
    lower = steps * base_ns
    upper = lower * max(1.0, mean_context / anchor_context)
    midpoint = lower * (2.0 / (1.0 + lower / upper))
    return midpoint, lower, upper


def samples_from_runtime(runtime: _OnlineRuntime) -> tuple[DecodeSample, ...]:
    """Use visible Token commit times, including real inter-cohort gaps."""
    state = runtime._state_values[0]
    previous = state.first_token_ns
    if previous is None:
        return ()
    samples = []
    for event in runtime.events:
        if (event.event_type != "tokens_committed"
                or event.request_id != state.spec.request_id
                or event.details.get("source") == "prefill"):
            continue
        step = int(event.details["committed"]) - 1
        if event.details.get("visible_tokens") != 1:
            raise ValueError("fast Decode requires one visible Token per step")
        # Attention length after this step's input was appended to K/V.
        samples.append(DecodeSample(step, state.spec.prompt_tokens + step,
                                    event.timestamp_ns - previous))
        previous = event.timestamp_ns
    return tuple(samples)


def forecast_segments(base: float, anchor: float, prompt: int,
                      sampled_steps: int, total_steps: int, segment_tokens: int):
    segments = []
    step = sampled_steps + 1
    while step <= total_steps:
        count = min(segment_tokens, total_steps - step + 1)
        first_context = prompt + step
        duration, lower, upper = model_envelope(base, anchor, first_context, count)
        segments.append(ForecastSegment(step, count, first_context,
            first_context + count - 1, duration, lower, upper))
        step += count
    return tuple(segments)


def _eligibility_reasons(scenario, plan, *, shared_residency: bool,
                         custom_lowerer: bool) -> tuple[str, ...]:
    reasons = []
    if len(plan.requests) != 1:
        reasons.append("fast_mode_requires_single_request")
    elif plan.requests[0].prompt_tokens <= 0:
        reasons.append("fast_mode_requires_nonempty_prompt")
    if _parallel_plan(scenario).world_size != 1:
        reasons.append("fast_mode_requires_single_rank")
    if plan.mtp.enabled:
        reasons.append("speculative_decode_requires_detailed_mode")
    if plan.kv_policy.layout_mode != "legacy_single":
        reasons.append("fast_mode_requires_contiguous_generic_kv")
    if plan.kv_policy.offload_component or plan.kv_policy.prefetch_distance:
        reasons.append("kv_offload_or_prefetch_requires_detailed_mode")
    layers = _execution_layers(scenario)
    if any(layer.is_linear_attention for layer in layers):
        reasons.append("linear_attention_requires_detailed_mode")
    if any(layer.is_moe or layer.kind.strip().lower() != "dense" for layer in layers):
        reasons.append("fast_mode_requires_dense_layers")
    if any(canonical_dtype(layer.dtype) != "bf16" or layer.quantization for layer in layers):
        reasons.append("fast_mode_requires_bf16_unquantized_layers")
    if plan.kv_policy.dtype and canonical_dtype(plan.kv_policy.dtype) != "bf16":
        reasons.append("fast_mode_requires_bf16_kv")
    if not scenario.weights_resident:
        reasons.append("fast_mode_requires_resident_weights")
    if not scenario.fusion_policy.flash_attention:
        reasons.append("fast_mode_requires_flash_attention")
    hardware = scenario.hardware.component_map()
    gpus = [item for item in hardware.values() if item.normalized_kind == "gpu"]
    if len(gpus) != 1 or not scenario.placement.op_to_component or any(
            hardware[target].normalized_kind != "gpu"
            for target in scenario.placement.op_to_component.values()):
        reasons.append("fast_mode_requires_one_gpu_for_model_operators")
    cache = hardware.get(plan.kv_policy.cache_component)
    if (cache is None or cache.normalized_kind not in {"hbm", "hbm_stack", "hbf"}
            or cache.metadata.get("physical_memory_config") is None):
        reasons.append("fast_mode_requires_physical_hbm_or_hbf")
    elif (cache.normalized_kind == "hbf" and
          getattr(require_physical_memory_config(cache), "protocol_mode", None) != "ocp_hbf_0_7"):
        reasons.append("fast_mode_requires_ocp_hbf_protocol")
    for profile in scenario.component_profiles.get("gpu", {}).values():
        kernel_model = getattr(profile, "kernel_model", None)
        if kernel_model is not None and getattr(kernel_model, "stateful_l2", False):
            reasons.append("stateful_gpu_cache_requires_detailed_mode")
    if scenario.llama_cpp_config is not None or scenario.workload.metadata.get("llama_cpp_runtime"):
        reasons.append("llama_runtime_requires_detailed_mode")
    if shared_residency:
        reasons.append("shared_residency_requires_detailed_mode")
    if custom_lowerer:
        reasons.append("custom_batch_lowerer_requires_detailed_mode")
    metadata = scenario.workload.metadata
    # These features add discontinuities or completion-side work which a
    # single ordinary Decode prefix cannot establish.
    if metadata.get("serving_runtime") or metadata.get("agent_lifecycle"):
        reasons.append("stateful_serving_extension_requires_detailed_mode")
    if metadata.get("prompt_cache") or metadata.get("agent_sessions"):
        reasons.append("retained_or_agent_kv_requires_detailed_mode")
    if len(plan.requests) == 1 and plan.requests[0].metadata:
        reasons.append("request_extensions_require_detailed_mode")
    return tuple(reasons)


def _calibration(samples: tuple[DecodeSample, ...], options,
                 sampling: SamplingWindow | None = None) -> dict[str, Any]:
    sampling = sampling or sampling_window(options, 1)
    count = sampling.training + sampling.holdout
    window = samples[-count:]
    training = window[:sampling.training]
    holdout = window[sampling.training:]
    if len(holdout) != sampling.holdout or any(
            not math.isfinite(sample.duration_ns) or sample.duration_ns <= 0
            for sample in window):
        return {"accepted": False, "reason": "invalid_decode_calibration"}
    base = fmean(sample.duration_ns for sample in training)
    anchor = fmean(sample.context_tokens for sample in training)
    estimated, _, _ = model_envelope(base, anchor, holdout[0].context_tokens, len(holdout))
    observed = math.fsum(sample.duration_ns for sample in holdout)
    error = abs(estimated - observed) / observed
    # A second check guards an obvious local phase change.  It is still a
    # prefix check, not a validation of the distant unexecuted suffix.
    split = len(training) // 2
    early = fmean(item.duration_ns for item in training[:split])
    late = fmean(item.duration_ns for item in training[split:])
    drift = abs(early - late) / max(early, late)
    return {"accepted": True, "base_ns": fmean(item.duration_ns for item in window),
            "anchor_context": fmean(item.context_tokens for item in window),
            "prefix_validation_relative_error": error,
            "calibration_half_window_relative_drift": drift,
            "training_mean_ns": base, "training_anchor_context": anchor,
            "training_early_mean_ns": early, "training_late_mean_ns": late,
            "holdout_observed_ns": observed, "holdout_estimated_ns": estimated,
            "training_steps": len(training), "holdout_steps": len(holdout),
            "training_first_decode_step": training[0].step,
            "training_last_decode_step": training[-1].step,
            "holdout_first_decode_step": holdout[0].step,
            "holdout_last_decode_step": holdout[-1].step,
            "refit_first_decode_step": window[0].step,
            "refit_last_decode_step": window[-1].step,
            "holdout_disjoint_from_training": True,
            "final_refit_includes_holdout": True}


def conditional_error_budget(lower, upper, estimate, local_error, maintenance):
    """Record each conditional interval; the terms are not additive errors."""
    prefix = maintenance["prefix_erase_bias_guard_ns"]
    future = maintenance["future_erase_serial_guard_ns"]

    def interval(lo, hi):
        if lo <= 0 or hi <= 0 or not all(math.isfinite(x) for x in (lo, hi, estimate)):
            return {"lower_ns": None, "upper_ns": None,
                    "overestimate_relative_error": None, "underestimate_relative_error": None,
                    "relative_error_budget": None}
        over, under = estimate / lo - 1, 1 - estimate / hi
        return {"lower_ns": lo, "upper_ns": hi,
                "overestimate_relative_error": over, "underestimate_relative_error": under,
                "relative_error_budget": max(over, under)}

    model = interval(lower, upper)
    media = interval(lower - prefix, upper + future)
    valid_local = math.isfinite(local_error) and 0 <= local_error < 1
    local = interval(lower / (1 + local_error), upper / (1 - local_error)) if valid_local else interval(0., 0.)
    combined = (interval((lower - prefix) / (1 + local_error),
                         (upper + future) / (1 - local_error))
                if valid_local else interval(0., 0.))
    reason = ("prefix_erase_budget_exhausts_lower_timing_bound" if lower - prefix <= 0 else
              "local_calibration_error_not_bounded_below_one" if not valid_local else None)
    return {"conditional_relative_error_budget": combined["relative_error_budget"],
        "budget_unavailable_reason": reason,
        "conditional_tail_lower_ns": combined["lower_ns"],
        "conditional_tail_upper_ns": combined["upper_ns"],
        "error_budget_breakdown": {"semantics": "conditional_interval_bounds_not_additive_percentages",
            "forecast_tail_ns": estimate, "local_relative_error": local_error if valid_local else None,
            "prefix_erase_bias_guard_ns": prefix, "future_erase_serial_guard_ns": future,
            "legacy_future_erase_serial_guard_ns": maintenance.get("legacy_future_erase_serial_guard_ns", future),
            "future_erase_upper_bound_methods": maintenance.get("future_erase_upper_bound_methods", []),
            "model_only": model, "model_with_local_error": local,
            "model_with_prefix_erase_bias": interval(lower - prefix, upper),
            "model_with_future_erase": interval(lower, upper + future),
            "model_with_maintenance": media, "combined": combined}}


def maintenance_error_envelope(runtime, projection, sampling: SamplingWindow,
                               remaining: int, context_scale: float) -> dict[str, Any]:
    """Bound possible erase bias without adding fabricated maintenance time.

    A prefix already includes some erase stalls. For the lower timing bound
    remove their entire serial service budget; for the upper timing bound
    add every projected ordinary page-0 erase. These are deliberately
    conservative conditional service bounds, not measured future latency.
    """
    if not projection.hbf_devices:
        return {"prefix_erase_bias_guard_ns": 0., "future_erase_serial_guard_ns": 0.,
                "legacy_future_erase_serial_guard_ns": 0., "future_erase_upper_bound_methods": [],
                "sampled_erase_operations": 0, "erase_observations_available": True}
    count = sampling.training + sampling.holdout
    indexes = [index for index, batch in enumerate(runtime.batches) if batch.kind == "decode"]
    selected = indexes[-count:]
    # Visible commits precede the cohort's kernel cleanup. A duration from
    # commit N-1 to commit N therefore includes cleanup from the preceding
    # cohort (possibly the final Prefill when warmup is zero). Charge all its
    # erases as well, rather than silently attributing that boundary to zero.
    has_previous = bool(selected) and selected[0] > 0
    previous = runtime.batches[selected[0] - 1] if has_previous else None
    batches = ([previous] if previous is not None else []) + [runtime.batches[index] for index in selected]
    traffic = [batch.cost.metadata.get("storage_traffic", {}) for batch in batches]
    available = len(selected) == count and has_previous and all("erase_operations" in row for row in traffic)
    erases = sum(int(row.get("erase_operations", 0) or 0) for row in traffic)
    previous_erases = int(traffic[0].get("erase_operations", 0) or 0) if has_previous else 0
    # Aggregate observations may cover multiple owners. Charging all observed
    # erases to each owner is conservative, as is the media-write guard.
    serial_per_erase = math.fsum(float(row["block_erase_ns"])
                                for row in projection.hbf_devices)
    prefix_bias = erases * serial_per_erase * remaining / count * context_scale
    future = math.fsum(float(row["ordinary_erase_serial_time_guard_ns"])
                       for row in projection.hbf_devices)
    legacy_future = math.fsum(float(row.get("legacy_ordinary_erase_serial_time_guard_ns",
        row["ordinary_erase_serial_time_guard_ns"])) for row in projection.hbf_devices)
    return {"prefix_erase_bias_guard_ns": prefix_bias,
            "future_erase_serial_guard_ns": future,
            "legacy_future_erase_serial_guard_ns": legacy_future,
            "future_erase_upper_bound_methods": [row.get("ordinary_erase_upper_bound_method",
                "serial_service_upper_bound") for row in projection.hbf_devices],
            "sampled_erase_operations": erases,
            "selected_cohort_erase_operations": erases - previous_erases,
            "previous_cohort_erase_operations": previous_erases,
            "erase_guard_batches": len(batches),
            "erase_guard_sample_steps": count,
            "erase_guard_remaining_steps": remaining,
            "erase_guard_context_scale": context_scale,
            "erase_observations_available": available,
            "erase_guard_semantics": "serial_service_upper_bounds_not_predicted_stall_time"}


def _candidate_memory_kind(scenario, plan) -> str | None:
    hardware = getattr(scenario, "hardware", None)
    kv_policy = getattr(plan, "kv_policy", None)
    if hardware is None or kv_policy is None:
        return None
    component = hardware.component_map().get(kv_policy.cache_component)
    kind = getattr(component, "normalized_kind", None)
    return "hbm" if kind in {"hbm", "hbm_stack"} else "hbf" if kind == "hbf" else None


def simulate_fast(scenario, *, batch_lowerer, execution_control=None,
                  residency_manager=None, execution_kernel=None,
                  runtime_origin_ns=0.0, custom_lowerer=False,
                  paired_reference: PairedTimingReference | None = None) -> FastSimulationOutcome:
    """Execute a real prefix and forecast only inside the declared safe domain."""
    control = execution_control or ExecutionControl()
    options = scenario.simulation.fast_decode
    if paired_reference is not None and not isinstance(paired_reference, PairedTimingReference):
        raise ValueError("paired_reference must be a PairedTimingReference")
    diagnostics: dict[str, Any] = {
        "method": "maintenance_aware_adaptive_prefix_directional_paired_envelope/v4",
        "requested_mode": "fast", "effective_mode": "detailed",
        "target_relative_error": options.target_relative_error,
        "validation_status": "not_validated_against_full_detailed",
        "physical_tail_transactions_recorded": False,
        "future_physical_state_advanced": False,
        "sample_context_semantics": "attention_context_tokens_after_current_input_append",
        "accuracy_target_is_guarantee": False,
        "sampling_strategy": options.sampling_strategy,
        "pairwise_ratio_target_relative_error": options.target_relative_error,
        "per_storage_conditional_error_target": (options.target_relative_error if paired_reference
            else options.target_relative_error / (2 + options.target_relative_error)),
        "independent_symmetric_conditional_error_target": options.target_relative_error / (2 + options.target_relative_error),
        "paired_reference": asdict(paired_reference) if paired_reference else None,
        "sampling_attempts": [],
        "sampling_resets": [],
        "model_assumptions": [
            "ordinary single-request Decode with fixed resident weights and dense attention",
            "steady mean Token cost lies in a nonnegative affine context-growth envelope",
            "ordinary OCP page-zero erases are bounded by serial service budgets; GC and block reuse remain excluded",
            "after observed ordinary erases, a new steady-cost fit may exclude those already timed prefix cohorts and their cleanup; future erase guards remain unchanged",
            "no unseen cache, scheduling, pressure-flush or allocation regime change",
            "whole-Decode bounds include the exact timed prefix; a matched BF16 request uses directional HBF/HBM interval arithmetic with both single-point and ratio errors bounded by target",
            "without a matched reference, per-storage conditional bounds retain target/(2+target) as a sufficient ratio condition",
            "prefix holdout error does not establish full-suffix accuracy",
        ],
    }
    lowerer_context = getattr(batch_lowerer, "_compilation_context", None)
    if getattr(lowerer_context, "scenario", None) is not scenario:
        lowerer_context = None
    with _compilation_scope(scenario, lowerer_context):
        plan = compile_serving_plan(scenario)
        memory_kind = _candidate_memory_kind(scenario, plan)
        if paired_reference is not None:
            reference_domain_reasons = _eligibility_reasons(scenario, plan,
                shared_residency=residency_manager is not None, custom_lowerer=custom_lowerer)
            if reference_domain_reasons:
                raise ValueError("paired reference candidate is outside the BF16 fast domain: "
                                 + ", ".join(reference_domain_reasons))
            request = plan.requests[0]
            validate_paired_reference(paired_reference, model_name=scenario.model.name,
                prompt_tokens=request.prompt_tokens, output_tokens=request.output_tokens,
                memory_kind=memory_kind, target_relative_error=options.target_relative_error)
        diagnostics["paired_budget_decision"] = paired_timing_decision(
            decode_estimate_ns=0., decode_lower_ns=None, decode_upper_ns=None,
            target_relative_error=options.target_relative_error,
            candidate_memory_kind=memory_kind, reference=paired_reference)
        runtime = _OnlineRuntime(plan, batch_lowerer, execution_control=control,
            residency_manager=residency_manager, execution_kernel=execution_kernel,
            runtime_origin_ns=runtime_origin_ns)

        def recorded_diagnostics():
            # Later calibrations append to these lists. Other nested
            # diagnostic values are replaced, not mutated, between attempts.
            return {**diagnostics,
                    "sampling_attempts": list(diagnostics["sampling_attempts"]),
                    "sampling_resets": list(diagnostics["sampling_resets"])}

        def detailed(*reasons):
            diagnostics.update(effective_mode="forecast_rejected", fallback_reasons=list(reasons),
                sampled_decode_steps=runtime._completed_decode_steps, predicted_decode_steps=0,
                validation_status="forecast_rejected_before_detailed_fallback")
            control.report("fast_mode_fallback", runtime._completed_decode_steps,
                message="快速模式转为详细执行：" + ", ".join(reasons),
                simulated_time_ns=runtime.now, metadata={"fallback_reasons": list(reasons),
                    "approximation": recorded_diagnostics()})
            result = runtime.run()
            diagnostics.update(effective_mode="detailed", fallback_reasons=list(reasons),
                sampled_decode_steps=sum(batch.kind == "decode" for batch in result.batches),
                predicted_decode_steps=0, validation_status="detailed_execution",
                physical_tail_transactions_recorded=True,
                future_physical_state_advanced=True)
            return FastSimulationOutcome(detailed=result, diagnostics=dict(diagnostics))

        reasons = _eligibility_reasons(scenario, plan,
            shared_residency=residency_manager is not None, custom_lowerer=custom_lowerer)
        if getattr(runtime, "residency_manager", None) is not None:
            reasons += ("runtime_residency_requires_detailed_mode",)
        if (getattr(getattr(runtime, "prompt_cache", None), "enabled", False)
                or getattr(runtime, "_retained_kv_state", None) is not None
                or getattr(runtime, "_agent_config", None)):
            reasons += ("retained_or_agent_runtime_requires_detailed_mode",)
        if reasons:
            if paired_reference is not None:
                raise ValueError("paired reference candidate is outside the fast runtime domain: "
                                 + ", ".join(reasons))
            return detailed(*reasons)
        request = plan.requests[0]
        total_steps = max(0, request.output_tokens - 1)
        page_period = hbf_kv_page_period(scenario)
        sampling = sampling_window(options, page_period)
        diagnostics["hbf_kv_page_period_tokens"] = page_period
        diagnostics["initial_sampling_window"] = asdict(sampling)
        initial_sample = sampling.total
        if initial_sample > options.max_sample_tokens:
            return detailed("calibration_does_not_cover_hbf_kv_page_period")
        split = sampling.training // 2
        if (min(sampling.training, sampling.holdout) < page_period
                or sampling.holdout % page_period
                or split % page_period or (sampling.training - split) % page_period):
            return detailed("calibration_does_not_cover_hbf_kv_page_period")
        if total_steps <= initial_sample:
            return detailed("output_shorter_than_calibration_budget")
        growth = total_steps / max(1, request.prompt_tokens)
        diagnostics["request_context_growth"] = growth
        if growth > options.max_context_growth:
            return detailed("context_growth_exceeds_fast_domain")
        target = initial_sample
        while True:
            control.raise_if_cancelled()
            if runtime.run_until_decode_steps(target):
                return detailed("request_completed_during_calibration")
            samples = samples_from_runtime(runtime)
            diagnostics["calibration_samples"] = [
                {**asdict(sample), "duration_ns": sample.duration_ns if math.isfinite(sample.duration_ns) else None}
                for sample in samples]
            decode_batches = [batch for batch in runtime.batches if batch.kind == "decode"]
            cycles = []
            for start in range(0, len(samples), page_period):
                cycle = samples[start:start + page_period]
                traffic = [batch.cost.metadata.get("storage_traffic", {})
                           for batch in decode_batches[start:start + page_period]]
                cycles.append({"first_step": cycle[0].step, "steps": len(cycle),
                    "duration_ns": math.fsum(sample.duration_ns for sample in cycle)
                        if all(math.isfinite(sample.duration_ns) for sample in cycle) else None,
                    "cohort_erase_operations": sum(int(row.get("erase_operations", 0) or 0) for row in traffic),
                    "cohort_program_pages": sum(int(row.get("pages_programmed", 0) or 0) for row in traffic),
                    "semantics": "commit_durations_and_cohort_media_counts_have_different_cleanup_boundaries"})
            diagnostics["calibration_cycles"] = cycles
            calibration = _calibration(samples, options, sampling)
            diagnostics.update(calibration)
            projection = project_fast_tail(runtime, detailed_decode_steps=len(samples))
            diagnostics["storage_projection"] = projection.to_metadata()
            if not projection.safe:
                return detailed(*projection.safety_reasons)
            remaining = total_steps - len(samples)
            if remaining <= 0:
                return detailed("no_remaining_decode_steps")
            if calibration["accepted"]:
                base, anchor = calibration["base_ns"], calibration["anchor_context"]
                segments = forecast_segments(base, anchor, request.prompt_tokens,
                    len(samples), total_steps, options.segment_tokens)
                lower = math.fsum(segment.model_lower_ns for segment in segments)
                upper = math.fsum(segment.model_upper_ns for segment in segments)
                estimate = math.fsum(segment.duration_ns for segment in segments)
                half_width = max(estimate / lower - 1, 1 - estimate / upper)
                local_error = max(calibration["prefix_validation_relative_error"],
                                  calibration["calibration_half_window_relative_drift"])
                maintenance = maintenance_error_envelope(runtime, projection, sampling,
                    remaining, max(1., (request.prompt_tokens + total_steps) / anchor))
                diagnostics.update(maintenance)
                if not maintenance["erase_observations_available"]:
                    return detailed("hbf_erase_observations_unavailable")
                budget = conditional_error_budget(lower, upper, estimate, local_error, maintenance)
                error_budget = budget["conditional_relative_error_budget"]
                exact_prefix = math.fsum(sample.duration_ns for sample in samples)
                tail_lower, tail_upper = budget["conditional_tail_lower_ns"], budget["conditional_tail_upper_ns"]
                whole_lower = exact_prefix + tail_lower if tail_lower is not None else None
                whole_upper = exact_prefix + tail_upper if tail_upper is not None else None
                decision = paired_timing_decision(decode_estimate_ns=exact_prefix + estimate,
                    decode_lower_ns=whole_lower, decode_upper_ns=whole_upper,
                    target_relative_error=options.target_relative_error,
                    candidate_memory_kind=memory_kind, reference=paired_reference)
                diagnostics.update(model_relative_half_width=half_width, **budget,
                    exact_sampled_decode_ns=exact_prefix,
                    decode_forecast_estimate_ns=exact_prefix + estimate,
                    decode_conditional_lower_ns=whole_lower, decode_conditional_upper_ns=whole_upper,
                    decode_conditional_relative_error_budget=decision["candidate_relative_error_budget"],
                    paired_budget_decision=decision)
                attempt = {
                    "sampled_steps": len(samples), **asdict(sampling),
                    **calibration, **maintenance, **budget,
                    "acceptance_target_relative_error": diagnostics["per_storage_conditional_error_target"],
                    "paired_budget_decision": decision,
                    "forecast_accepted": decision["accepted"]}
                diagnostics["sampling_attempts"].append(attempt)
                control.report("fast_calibration", len(samples), total_steps,
                    message="Decode 采样误差预算分项", simulated_time_ns=runtime.now,
                    metadata={"approximation": recorded_diagnostics(), "calibration_attempt": attempt})
                if decision["accepted"]:
                    break
                if options.sampling_strategy == "adaptive" and projection.hbf_devices:
                    reset = maintenance_recalibration(runtime, sampling, options)
                    if reset is not None:
                        candidate, reset_record = reset
                        diagnostics["sampling_resets"].append(reset_record)
                        control.report("fast_recalibration", len(samples), total_steps,
                            message="按已执行擦除和清理边界重新采集独立校准窗口",
                            simulated_time_ns=runtime.now,
                            metadata={"approximation": recorded_diagnostics(),
                                      "sampling_reset": reset_record})
                        if not reset_record["within_sample_budget"]:
                            return detailed("maintenance_recalibration_exceeds_sample_budget")
                        if candidate.total >= total_steps:
                            return detailed("output_shorter_than_calibration_budget")
                        sampling, target = candidate, candidate.total
                        continue
            if target >= options.max_sample_tokens:
                return detailed("prefix_calibration_exceeds_error_budget")
            if options.sampling_strategy == "adaptive":
                expanded = expanded_sampling_window(sampling, options.max_sample_tokens)
                if expanded is None:
                    return detailed("prefix_calibration_exceeds_error_budget")
                sampling = expanded
                target = min(total_steps, sampling.total)
            else:
                target = min(options.max_sample_tokens, total_steps,
                             target + options.validation_tokens)

        state = runtime._state_values[0]
        control.raise_if_cancelled()
        # The cutoff is the last visible Token boundary.  Device cleanup and
        # sampled cohort bookkeeping after that boundary are not added to TPOT.
        commits = [event.timestamp_ns for event in runtime.events
            if event.event_type == "tokens_committed" and event.request_id == request.request_id]
        cutoff = commits[-1]
        finish = cutoff + math.fsum(segment.duration_ns for segment in segments)
        first = state.first_token_ns
        assert first is not None
        forecast = DecodeForecast(request.request_id, request.arrival_ns, first,
            cutoff, finish, runtime_origin_ns, request.prompt_tokens, request.output_tokens,
            len(samples), remaining, first - request.arrival_ns,
            (finish - first) / total_steps, tuple(segments), samples,
            projection.to_metadata())
        diagnostics.update(effective_mode="fast", fallback_reasons=[],
            sampled_decode_steps=len(samples), predicted_decode_steps=remaining,
            detailed_prefill_batches=sum(batch.kind == "prefill" for batch in runtime.batches),
            detailed_prefix_task_count=sum(int(batch.cost.metadata.get("task_count", 0))
                                           for batch in runtime.batches),
            forecast_segments=[asdict(segment) for segment in segments],
            detailed_physical_cutoff_ns=runtime.now)
        control.report("fast_forecast_complete", total_steps, total_steps,
            message="快速仿真完成：Prefill 与 Decode 前缀为详细执行，尾段为预测",
            simulated_time_ns=finish,
            metadata={"sampled_decode_steps": len(samples), "predicted_decode_steps": remaining})
        return FastSimulationOutcome(forecast=forecast, diagnostics=diagnostics)
