"""Native Prefill pass of one request, from native layer samples.

The Prefill node is a replica with the Decode replica's tp/pp/ep layout. A
request is prefilled alone the moment it arrives, in one forward pass over
its first P-1 prompt tokens; the Decode node imports that KV and computes
the final prompt token with the first output token. The pass lasts from its
cohort boundary to the completion of its last layer, where the KV export
starts; the output head is not part of it.

Sampled: on independent copies of the boundary, a calibration run (the
prefix and the first layers), a holdout run (the last layers, later in
time) and the entry once more at the holdout origin. Each run takes at least
its configured layer count and as many more as it needs to step every layer
class. The pass is the entry
plus one class step per later layer, composed and widened exactly as the
stationary Decode step (``native_stationary.compose_layers``) without its
exit; unsampled layers take their class's step under
SAME_CLASS_ADDRESS_ASSUMPTION. Full: one run of the prefix and every layer,
read directly and widened only by its boundary row-state bound.
"""
from __future__ import annotations

from dataclasses import asdict
import time

from .execution_control import ExecutionControl, NativeFastWallBudgetError, _NativeLayerSamplingCutoff
from .native_fast_window import _native_runtime_reasons
from .native_layer_sampling import NativeLayerSamplingGuardError
from .native_stationary import (HOLDOUT_SHIFT_NS, SAME_CLASS_ADDRESS_ASSUMPTION, _completions, _interval,
                                _within, cohort_layers, compose_layers, deadline_control, run_window,
                                window_terms)
from .planner import _compilation_scope
from .serving import _OnlineRuntime, compile_serving_plan


def _eligibility_reasons(scenario, plan, *, custom_lowerer, shared_residency):
    reasons = []
    if scenario.workload.metadata.get("external_kv_ingress") is not None:
        reasons.append("native_prefill_pass_has_no_external_ingress")
    if len(scenario.workload.requests) != 1:
        reasons.append("native_prefill_pass_requires_one_request")
    return (*reasons, *_native_runtime_reasons(plan, custom_lowerer=custom_lowerer,
                                               shared_residency=shared_residency))


def prepare_prefill(runtime, cohort, stages, request_ready_ns, control):
    """The request's whole-prompt forward pass on the unchanged source."""
    started = time.perf_counter()
    control.raise_if_cancelled()
    item = cohort.items[0] if cohort.kind == "prefill" and len(cohort.items) == 1 else None
    if (item is None or item.phase != "prefill" or item.context_tokens != 0
            or item.token_count != runtime.states[item.request_id].spec.prompt_tokens):
        raise NativeLayerSamplingGuardError("native_prefill_pass_requires_one_whole_prompt_forward")
    prepared = cohort_layers(runtime, cohort, stages, request_ready_ns, "native-prefill-sample.")
    return {**prepared, "prompt_tokens": item.token_count,
            "preparation_wall_seconds": time.perf_counter() - started}


def _layer_counts(classes, calibration, holdout):
    """Fewest layers from the stack's start and end, at least ``calibration`` and
    ``holdout``, whose steps (every run layer after the first) reach every class."""
    wanted, count = set(classes), len(classes)

    def covering(minimum, stepped):
        return next((n for n in range(minimum, count + 1) if set(stepped(n)) == wanted), count + 1)

    return (covering(calibration, lambda n: classes[1:n]),
            covering(holdout, lambda n: classes[count - n + 1:]))


def sample_prefill(prepared, control, options, records, *, holdout_shift_ns=HOLDOUT_SHIFT_NS):
    """Run the pass's native samples; the source kernel never advances."""
    started = time.perf_counter()
    layer_ids = [row["layer_id"] for row in prepared["inventory"]]
    prefix = prepared["boundary"].get("before_first_layer", [])
    first = prepared["origin"]

    def run(role, ids, groups, origin, tag, entry):
        return run_window(prepared, control, records, role, ids, groups, origin, "prefill-" + tag,
                          row_terms=window_terms(prepared, ids, entry=entry, exit=False))

    if options.full_pass:
        runs = {"full": run("prefill_pass", layer_ids, (prefix,), first, "full", True)}
    else:
        calibration, holdout = _layer_counts([row["class_key"] for row in prepared["inventory"]],
                                             options.calibration_layers, options.holdout_layers)
        if calibration + holdout > len(layer_ids):
            raise NativeLayerSamplingGuardError("native_prefill_stack_too_short_for_disjoint_runs")
        head, tail, second = layer_ids[:calibration], layer_ids[-holdout:], first + holdout_shift_ns
        runs = {"calibration": run("calibration", head, (prefix,), first, "calibration", True),
                "holdout": run("holdout", tail, (), second, "holdout", False),
                "entry_at_holdout_origin": run("entry", head[:1], (prefix,), second, "entry-second", True)}
    return {"runs": runs, "layer_count": len(layer_ids), "holdout_shift_ns": holdout_shift_ns,
            "layers": {role: len(run["layer_ids"]) for role, run in runs.items()},
            "wall_seconds": time.perf_counter() - started}


def evaluate_prefill(prepared, samples, options):
    """The pass's interval within the independent per-metric budget."""
    target = options.target_relative_error
    limit = target / (2 + target)
    runs = samples["runs"]
    budget = {"target_relative_error": target, "effective_independent_target": limit,
              "prompt_tokens": prepared["prompt_tokens"]}
    result = {"accepted": False, "reasons": [], "metrics": {}, "metric_bounds": {}, "error_budget": budget}
    try:
        if options.full_pass:
            run = runs["full"]
            times = _completions(run, run["layer_ids"])
            value = times[-1] - run["origin_ns"]
            width = sum(run["first_access_ns"].values())
            lower, upper = value - width, value + width
            # Every layer's own step, for checking sampled class intervals.
            budget.update(basis="one_native_run_of_prefix_and_every_layer; boundary_first_access_row_state_bounds",
                          observed_pass_ns=value, first_access_ns=run["first_access_ns"], assumptions=[],
                          entry_with_prefix_ns=times[0] - run["origin_ns"],
                          layer_steps_ns=dict(zip(run["layer_ids"][1:], (b - a for a, b in zip(times, times[1:])))))
        else:
            lower, upper, parts = compose_layers(prepared["inventory"], runs, exit=False)
            budget.update(basis="calibration_holdout_layer_hull_widened_by_drift; in_situ_entry_two_origins; "
                                "boundary_first_access_row_state_bounds",
                          composition=parts, assumptions=[SAME_CLASS_ADDRESS_ASSUMPTION])
    except ValueError as error:
        result["reasons"].append(str(error))
        return result
    if lower <= 0:
        result["reasons"].append("native_prefill_pass_lower_bound_not_positive")
        return result
    row = _interval(lower, upper)
    result["metric_bounds"] = {"prefill_pass_ns": row}
    if not _within(row, limit):
        result["reasons"].append("native_prefill_independent_budget_exceeded")
        return result
    result.update(accepted=True, metrics={"prefill_pass_ns": row["estimate"]})
    return result


def simulate_native_prefill_pass(scenario, *, batch_lowerer, execution_control=None,
                                 residency_manager=None, execution_kernel=None, runtime_origin_ns=0.,
                                 custom_lowerer=False):
    """Sample the request's Prefill cohort boundary natively and bound the pass."""
    from .fast_simulation import FastSimulationOutcome

    options = scenario.simulation.native_prefill
    started = time.perf_counter()
    deadline = None if options.max_wall_seconds is None else started + options.max_wall_seconds
    control = execution_control or ExecutionControl()
    sample_control = deadline_control(control, deadline)
    diagnostics = {"method": "native_prefill_pass/v1", "requested_mode": "fast",
        "effective_mode": "forecast_rejected", "sampling_options": asdict(options),
        "target_relative_error": options.target_relative_error,
        "accuracy_target_is_guarantee": False, "future_physical_state_advanced": False,
        "rejection_reasons": [], "fallback_reasons": [],
        "prefill_pass": {"accepted": False, "metrics": {}, "metric_bounds": {}, "error_budget": {}}}
    records, prepared, samples = [], {}, {}
    context = getattr(batch_lowerer, "_compilation_context", None)
    with _compilation_scope(scenario, context):
        sample_control.raise_if_cancelled()
        plan = compile_serving_plan(scenario)
        reasons = _eligibility_reasons(scenario, plan, custom_lowerer=custom_lowerer,
                                       shared_residency=residency_manager is not None)
        if reasons:
            diagnostics.update(rejection_reasons=list(reasons), fallback_reasons=list(reasons),
                               wall_seconds=time.perf_counter() - started)
            return FastSimulationOutcome(diagnostics=diagnostics)
        try:
            runtime = _OnlineRuntime(plan, batch_lowerer, execution_control=sample_control,
                residency_manager=residency_manager, execution_kernel=execution_kernel,
                runtime_origin_ns=runtime_origin_ns)

            def sampler(active, cohort, cost, stages, request_ready_ns):
                prepared.update(prepare_prefill(active, cohort, stages, request_ready_ns, sample_control))
                raise _NativeLayerSamplingCutoff()

            runtime._native_layer_sampler = sampler
            try:
                runtime.run_until_decode_steps(1, strict_cohort_cutoff=True)
            except _NativeLayerSamplingCutoff:
                pass
            else:
                raise RuntimeError("native Prefill sampling unexpectedly committed an ordinary cohort")
            samples = sample_prefill(prepared, sample_control, options, records)
            budget = evaluate_prefill(prepared, samples, options)
        except NativeFastWallBudgetError:
            budget = {"accepted": False, "reasons": ["native_prefill_sample_wall_budget_exhausted"]}
        except NativeLayerSamplingGuardError as error:
            budget = {"accepted": False, "reasons": [str(error)]}
        layer_samples = tuple(records)
        reasons = list(budget.get("reasons", ()))
        expired = deadline is not None and time.perf_counter() >= deadline
        try:
            control.raise_if_cancelled()
        except NativeFastWallBudgetError:
            expired = True
        if expired:
            reasons.append("native_total_candidate_wall_budget_exhausted")
        accepted = budget.get("accepted") is True and not expired
        diagnostics.update(effective_mode="fast" if accepted else "forecast_rejected",
            rejection_reasons=reasons, fallback_reasons=reasons,
            prefill_pass={"accepted": accepted, "metrics": budget.get("metrics", {}) if accepted else {},
                          "metric_bounds": budget.get("metric_bounds", {}),
                          "error_budget": budget.get("error_budget", {}),
                          "boundary_origin_ns": prepared.get("origin"),
                          "layer_count": samples.get("layer_count"), "run_layers": samples.get("layers"),
                          "sample_wall_seconds": samples.get("wall_seconds")},
            preparation_wall_seconds=prepared.get("preparation_wall_seconds"),
            physical_layer_window_count=len(layer_samples), wall_seconds=time.perf_counter() - started)
        control.report("native_prefill_pass_complete" if accepted else "native_fast_rejected",
            len(layer_samples), unit="work_units",
            message="原生Prefill层样本已保存" + ("，Prefill时长区间已通过预算" if accepted else "，未通过预算"))
        return FastSimulationOutcome(diagnostics=diagnostics, layer_window_samples=layer_samples)


__all__ = ["prepare_prefill", "sample_prefill", "evaluate_prefill", "simulate_native_prefill_pass"]
