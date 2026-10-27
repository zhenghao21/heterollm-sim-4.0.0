"""Native Prefill pass of one request, from native layer samples.

The Prefill node is a replica with the Decode replica's tp/pp/ep layout. A
request is prefilled alone the moment it arrives, over its first P-1 prompt
tokens; model-required chunk boundaries remain in force. A multi-chunk pass
executes entirely through the ordinary native runtime, including its real
cross-chunk preparation overlap. Its end is observed at the last layer,
before the final output head. A single-chunk pass retains layer sampling.
The Decode node imports that KV and computes
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
from .native_layer_sampling import NativeLayerSamplingGuardError, _layer_counts
from .native_stationary import (HOLDOUT_SHIFT_NS, SAME_CLASS_ADDRESS_ASSUMPTION, _completions, _interval,
                                _within, cohort_layers, cohort_origin, compose_layers, deadline_control, run_window,
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


def _prefill_item(runtime, cohort):
    """One native chunk at this request's actually committed cursor."""
    item = cohort.items[0] if cohort.kind == "prefill" and len(cohort.items) == 1 else None
    state = runtime.states.get(item.request_id) if item is not None else None
    if (state is None or item.phase != "prefill" or item.token_count <= 0
            or item.context_tokens != state.prefill_cursor
            or item.context_tokens + item.token_count > state.spec.prompt_tokens):
        raise NativeLayerSamplingGuardError("native_prefill_pass_requires_one_contiguous_prompt_forward")
    return item, state


def prepare_prefill(runtime, cohort, stages, request_ready_ns, control):
    """A whole-prompt single cohort on the unchanged source."""
    started = time.perf_counter()
    control.raise_if_cancelled()
    item, state = _prefill_item(runtime, cohort)
    if item.context_tokens or item.token_count != state.spec.prompt_tokens:
        raise NativeLayerSamplingGuardError("native_prefill_pass_requires_one_whole_prompt_forward")
    prepared = cohort_layers(runtime, cohort, stages, request_ready_ns, "native-prefill-sample.")
    return {**prepared, "prompt_tokens": state.spec.prompt_tokens,
            "preparation_wall_seconds": time.perf_counter() - started}


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
    diagnostics = {"method": "native_prefill_pass/v2", "requested_mode": "fast",
        "effective_mode": "forecast_rejected", "sampling_options": asdict(options),
        "target_relative_error": options.target_relative_error,
        "accuracy_target_is_guarantee": False, "future_physical_state_advanced": False,
        "rejection_reasons": [], "fallback_reasons": [],
        "prefill_pass": {"accepted": False, "metrics": {}, "metric_bounds": {}, "error_budget": {}}}
    records, prepared, samples, native_chunks = [], {}, {}, []
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
        runtime = None
        try:
            runtime = _OnlineRuntime(plan, batch_lowerer, execution_control=sample_control,
                residency_manager=residency_manager, execution_kernel=execution_kernel,
                runtime_origin_ns=runtime_origin_ns)

            from .ir import model_graph_execution_layers
            last_layer = model_graph_execution_layers(scenario.model.graph)[-1].layer_id
            pass_origin_ns, native_chunks_mode = None, False

            def sampler(active, cohort, cost, stages, request_ready_ns):
                nonlocal pass_origin_ns, native_chunks_mode
                item, state = _prefill_item(active, cohort)
                if pass_origin_ns is None:
                    if item.context_tokens:
                        raise NativeLayerSamplingGuardError("native_prefill_pass_requires_prompt_start")
                    pass_origin_ns = cohort_origin(active, request_ready_ns)[0]
                    native_chunks_mode = (item.token_count < state.spec.prompt_tokens
                                          or scenario.placement.parallel.pp_degree > 1)
                if native_chunks_mode:
                    # Sampling a later chunk loses its overlap with earlier
                    # chunks. PP also contains inter-layer transfers that the
                    # serial-layer composition cannot represent. Execute both
                    # cases on the complete live runtime, even for one chunk.
                    return False
                prepared.update(prepare_prefill(active, cohort, stages, request_ready_ns, sample_control))
                raise _NativeLayerSamplingCutoff()

            def completed_chunk(active, cohort, metrics):
                if not native_chunks_mode:
                    return
                item, _state = _prefill_item(active, cohort)
                ends = metrics.layer_completion_ns()
                if last_layer not in ends:
                    raise NativeLayerSamplingGuardError("native_prefill_chunk_last_layer_not_observed")
                native_chunks.append({"context_tokens": item.context_tokens, "prompt_tokens": item.token_count,
                                      "last_layer_end_ns": ends[last_layer]})

            runtime._native_layer_sampler = sampler
            runtime._native_prefill_cohort_completed = completed_chunk
            try:
                runtime.run_until_decode_steps(1, strict_cohort_cutoff=True)
            except _NativeLayerSamplingCutoff:
                pass
            else:
                if not native_chunks_mode:
                    raise RuntimeError("native Prefill sampling unexpectedly committed an ordinary cohort")
            if native_chunks_mode:
                state = next(iter(runtime.states.values()))
                if (not native_chunks or state.prefill_cursor != state.spec.prompt_tokens
                        or sum(chunk["prompt_tokens"] for chunk in native_chunks) != state.spec.prompt_tokens):
                    raise NativeLayerSamplingGuardError("native_prefill_complete_prompt_not_observed")
                value = native_chunks[-1]["last_layer_end_ns"] - pass_origin_ns
                if value <= 0:
                    raise NativeLayerSamplingGuardError("native_prefill_pass_lower_bound_not_positive")
                bounds = {"estimate": value, "lower": value, "upper": value, "relative_error_budget": 0.}
                budget = {"accepted": True, "reasons": [], "metrics": {"prefill_pass_ns": value},
                    "metric_bounds": {"prefill_pass_ns": bounds},
                    "error_budget": {"basis": "complete_native_prompt_chunks; observed_last_layer_completion",
                        "observed_pass_ns": value, "prompt_tokens": state.spec.prompt_tokens,
                        "native_chunk_count": len(native_chunks), "chunks": native_chunks,
                        "pass_origin_ns": pass_origin_ns, "assumptions": []}}
            else:
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
        if native_chunks and runtime is not None:
            diagnostics.update(physical_complete_cohort_count=len(runtime.batches),
                physical_visible_output_token_count=sum(event.details.get("visible_tokens", 0)
                    for event in runtime.events if event.event_type == "tokens_committed"))
        diagnostics.update(effective_mode="fast" if accepted else "forecast_rejected",
            rejection_reasons=reasons, fallback_reasons=reasons,
            prefill_pass={"accepted": accepted, "metrics": budget.get("metrics", {}) if accepted else {},
                          "metric_bounds": budget.get("metric_bounds", {}),
                          "error_budget": budget.get("error_budget", {}),
                          "boundary_origin_ns": prepared.get("origin"),
                          "execution_mode": "complete_native_chunks" if native_chunks else "native_layer_samples",
                          "native_chunk_count": len(native_chunks),
                          "layer_count": samples.get("layer_count"), "run_layers": samples.get("layers"),
                          "sample_wall_seconds": samples.get("wall_seconds")},
            preparation_wall_seconds=prepared.get("preparation_wall_seconds"),
            physical_layer_window_count=len(layer_samples), wall_seconds=time.perf_counter() - started)
        control.report("native_prefill_pass_complete" if accepted else "native_fast_rejected",
            len(layer_samples), unit="work_units",
            message=("原生Prefill完整前向已执行" if native_chunks else "原生Prefill层样本已保存")
                    + ("，Prefill时长区间已通过预算" if accepted else "，未通过预算"))
        return FastSimulationOutcome(diagnostics=diagnostics, layer_window_samples=layer_samples)


__all__ = ["prepare_prefill", "sample_prefill", "evaluate_prefill", "simulate_native_prefill_pass"]
