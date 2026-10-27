"""Real native PP samples and a separate conditional window forecast.

The sampler executes every layer, microbatch and activation transfer of its
physical cohorts. Only a structurally verified staggered preloaded seed can
represent later boundaries without first replaying the requested warmup.
"""
from __future__ import annotations

import time
from dataclasses import replace
from copy import copy

from .execution_control import ExecutionControl, NativeFastWallBudgetError, _NativeLayerSamplingCutoff
from .native_fast_window import _eligibility_reasons
from .native_layer_sampling import NativeLayerSamplingGuardError
from .native_stationary import (_import_pool_occupancy, _preloaded_seed_admission,
                                _query_decode_guard, deadline_control)
from .planner import _compilation_scope
from .serving import RequestStatus, _OnlineRuntime, _max_live_kv_tokens, compile_serving_plan
from .native_pipeline_forecast import compose_pipeline_window, samples_from_execution_schedule
from .native_pipeline_sampling import (PipelineStageSamplingGuardError,
    select_representative_pipeline_tasks, stage_samples_from_window)
from .native_layer_sampling import _execute_window


def simulate_native_pipeline_forecast_window(scenario, *, batch_lowerer, execution_control=None,
                                             residency_manager=None, execution_kernel=None,
                                             runtime_origin_ns=0., custom_lowerer=False):
    """Run a small real PP prefix and compose the remaining window conditionally."""
    from .fast_simulation import FastSimulationOutcome
    started = time.perf_counter()
    options = scenario.simulation.native_window
    control = deadline_control(execution_control or ExecutionControl(), started + options.max_wall_seconds)
    diagnostics = {"method":"native_pipeline_stage_forecast/v2", "requested_mode":"fast",
        "effective_mode":"forecast_rejected", "target_relative_error":options.target_relative_error,
        "forecast_used":True, "rejection_reasons":[], "fallback_reasons":[],
        "physical_complete_cohort_count":0, "physical_visible_output_token_count":0,
        "pipeline":{"stages":scenario.placement.parallel.pp_degree,
                    "microbatches":scenario.placement.parallel.pipeline_microbatches,
                    "schedule":"cohort_flush_stage_fifo"}}
    runtime = None
    stage_sample_rows = []
    with _compilation_scope(scenario, getattr(batch_lowerer, "_compilation_context", None)):
        try:
            plan = compile_serving_plan(scenario)
            reasons = _eligibility_reasons(scenario, plan, custom_lowerer=custom_lowerer,
                                           shared_residency=residency_manager is not None)
            if reasons:
                diagnostics["rejection_reasons"] = list(reasons)
                return FastSimulationOutcome(diagnostics=diagnostics)
            runtime = _OnlineRuntime(plan, batch_lowerer, execution_control=control,
                residency_manager=residency_manager, execution_kernel=execution_kernel,
                runtime_origin_ns=runtime_origin_ns)
            runtime._native_preload_admission_guard = lambda active: _preloaded_seed_admission(active, diagnostics)
            def first_cohort_admission(active, cohort, _cost, _stages, _ready):
                active._native_layer_sampler = None
                _first_cohort_pool_admission(active, cohort, diagnostics)
                if _representative_seed_boundary(active, cohort):
                    try:
                        guard = _query_decode_guard(active, cohort, _stages)
                    except NativeLayerSamplingGuardError as error:
                        diagnostics["seed_boundary_rejection"] = str(error)
                    else:
                        # The IR guard bounds changed nonphysical service, rather
                        # than proving it equal. Keep the ordinary warmup whenever
                        # that service is not exactly the same.
                        diagnostics["seed_query_decode_guard_ns"] = guard
                        diagnostics["representative_seed_boundary"] = guard == 0.
                        if options.pipeline_stage_sampling and guard == 0.:
                            if sample_decode < 1:
                                raise PipelineStageSamplingGuardError(
                                    "native_pipeline_stage_sampling_requires_decode_samples")
                            ready = {stage.stage_id: active.now for stage in _stages}
                            specs, _ = active._stage_task_specs(
                                _stages, ready, "native-pipeline-stage-sample." + cohort.cohort_id)
                            selected, descriptors = select_representative_pipeline_tasks(_stages, specs)
                            imports = (active._external_executor, tuple(active._external_pending)) \
                                if active._external_executor is not None and active._external_pending else None
                            for sample_index in range(sample_decode):
                                row = _execute_window(
                                    active._execution_resource_kernel, specs, selected,
                                    origin_ns=active.now + sample_index * 1.0e6,
                                    control=active.execution_control,
                                    sample_id="native-pipeline-stage-{}".format(sample_index),
                                    concurrent_imports=imports)
                                stage_sample_rows.extend(stage_samples_from_window(
                                    row, descriptors, cohort=sample_index))
                            diagnostics.update(
                                pipeline_stage_sampling=True,
                                stage_sample_task_count=len(selected),
                                stage_sample_total_task_count=len(specs),
                                physical_complete_cohort_count=0,
                                physical_visible_output_token_count=0)
                            raise _NativeLayerSamplingCutoff()
                return False
            runtime._native_layer_sampler = first_cohort_admission
            ingress = scenario.workload.metadata.get("external_kv_ingress", {})
            warmup = int(ingress.get("warmup_decode_cohorts", 0)); measured = int(ingress.get("measurement_decode_cohorts", 0))
            sample_decode = min(options.pipeline_sample_decode_cohorts, measured)
            if sample_decode < 1: raise NativeLayerSamplingGuardError("native_pipeline_forecast_requires_decode_samples")
            # The first call discovers whether the real seed boundary qualifies.
            # A generic pipeline keeps its original physical warmup; a qualified
            # seed already has the declared steady-state context distribution.
            runtime.run_until_decode_steps(1, strict_cohort_cutoff=True)
            # The first decode after a preloaded boundary still carries the
            # boundary's activation handoff.  Run one extra cohort and use the
            # trailing samples so that the forecast is built from the steady
            # FIFO service rather than that one-time handoff.
            physical_target = (sample_decode + 2 if diagnostics.get("representative_seed_boundary")
                               else warmup + sample_decode)
            runtime.run_until_decode_steps(physical_target, strict_cohort_cutoff=True)
            sampled = runtime._result()
            rows, prefixes, suffixes = [], [], []
            decode_batches = [b for b in runtime.batches if b.kind != "prefill"]
            if len(decode_batches) < physical_target:
                raise NativeLayerSamplingGuardError("native_pipeline_decode_sample_prefix_incomplete")
            for index, batch in enumerate(decode_batches[-sample_decode:]):
                schedule = batch.cost.metadata.get("execution_stage_schedule", ())
                selected = samples_from_execution_schedule(schedule, cohort=index)
                if not selected:
                    raise NativeLayerSamplingGuardError("native_pipeline_stage_samples_unavailable")
                rows.extend(selected)
                prefixes.append(max(0., min(row.start_ns for row in selected) - batch.start_ns))
                suffixes.append(max(0., batch.end_ns - max(row.end_ns for row in selected)))
            forecast = compose_pipeline_window(rows, sampled_decode_cohorts=sample_decode,
                predicted_decode_cohorts=measured, batch_size=plan.scheduler.max_num_seqs,
                output_tokens=max(2, max((s.spec.output_tokens for s in runtime._state_values), default=2)),
                cutoff_ns=runtime.now, target_relative_error=options.target_relative_error,
                warmup_decode_cohorts=warmup, physical_prefix_decode_cohorts=len(decode_batches),
                cohort_prefix_samples_ns=prefixes, cohort_suffix_samples_ns=suffixes)
            physical_visible = sum(e.details.get("visible_tokens", 0) for e in runtime.events if e.event_type == "tokens_committed")
            forecast = replace(forecast, sampled_visible_output_tokens=physical_visible)
            diagnostics.update(effective_mode="fast", predicted_decode_cohorts=forecast.predicted_decode_cohorts,
                predicted_visible_output_tokens=forecast.predicted_visible_output_tokens,
                physical_complete_cohort_count=len(runtime.batches),
                physical_visible_output_token_count=physical_visible,
                physical_warmup_decode_cohorts=min(len(decode_batches), warmup),
                physical_sample_decode_cohorts=sample_decode,
                skipped_physical_warmup_decode_cohorts=max(0, warmup - len(decode_batches)),
                pipeline_tp_reuse=getattr(runtime, "_native_pipeline_tp_reuse", None),
                wall_seconds=time.perf_counter()-started)
            return FastSimulationOutcome(native_pipeline_forecast=forecast, sampled=sampled, diagnostics=diagnostics)
        except _NativeLayerSamplingCutoff:
            if not stage_sample_rows:
                raise NativeLayerSamplingGuardError("native_pipeline_stage_sampling_empty")
            output_tokens = max(2, max((s.spec.output_tokens for s in runtime._state_values), default=2))
            forecast = compose_pipeline_window(
                stage_sample_rows, sampled_decode_cohorts=sample_decode,
                predicted_decode_cohorts=measured, batch_size=plan.scheduler.max_num_seqs,
                output_tokens=output_tokens, cutoff_ns=runtime.now,
                target_relative_error=options.target_relative_error,
                warmup_decode_cohorts=warmup, physical_prefix_decode_cohorts=0,
                allow_zero_physical_prefix=True)
            diagnostics.update(effective_mode="fast", predicted_decode_cohorts=forecast.predicted_decode_cohorts,
                predicted_visible_output_tokens=forecast.predicted_visible_output_tokens,
                physical_warmup_decode_cohorts=0,
                physical_sample_decode_cohorts=0,
                skipped_physical_warmup_decode_cohorts=warmup,
                wall_seconds=time.perf_counter()-started)
            return FastSimulationOutcome(native_pipeline_forecast=forecast, sampled=None,
                                         diagnostics=diagnostics)
        except (NativeFastWallBudgetError, NativeLayerSamplingGuardError,
                PipelineStageSamplingGuardError, ValueError) as error:
            diagnostics["rejection_reasons"] = [str(error)]
            diagnostics["fallback_reasons"] = [str(error)]
            if runtime is not None:
                diagnostics["physical_complete_cohort_count"] = len(runtime.batches)
                diagnostics["physical_visible_output_token_count"] = sum(
                    e.details.get("visible_tokens", 0) for e in runtime.events
                    if e.event_type == "tokens_committed")
                diagnostics["physical_ingress_accounting"] = runtime._external_ingress_report()
            diagnostics["wall_seconds"] = time.perf_counter()-started
            return FastSimulationOutcome(diagnostics=diagnostics)


def _representative_seed_boundary(runtime, cohort):
    """Recognize the already approved centered-age seed construction exactly."""
    if not runtime._external_ingress.get("preload_warmup_seed_kv", False):
        return False
    batch = runtime.plan.scheduler.max_num_seqs
    seeds = [state for state in runtime._state_values if state.spec.metadata.get("warmup_seed", False)]
    fresh = [state for state in runtime._state_values if not state.spec.metadata.get("warmup_seed", False)]
    if (len(seeds) != batch or runtime._external_preloaded_requests != batch or not fresh
            or len(cohort.items) != batch or set(cohort.request_ids) != {s.spec.request_id for s in seeds}
            or any(s.spec.arrival_ns > runtime.now for s in (*seeds, *fresh))):
        return False
    shapes = {(state.spec.prompt_tokens, state.spec.output_tokens) for state in fresh}
    if len(shapes) != 1:
        return False
    prompt, output = next(iter(shapes))
    if output < 2:
        return False
    ages = sorted(state.spec.prompt_tokens - prompt for state in seeds)
    expected = [(2 * index + 1) * output // (2 * batch) for index in range(batch)]
    return (ages == expected and all(state.spec.prompt_tokens + state.spec.output_tokens == prompt + output
                                    for state in seeds))


def _first_cohort_pool_admission(runtime, cohort, diagnostics):
    """Reject a pool shortfall only when the native KV ledger proves it.

    This observes the first full preloaded-seed cohort, after ordinary import
    admission has run. In-flight imports count as occupied pool entries: a
    slow paid transfer is not a capacity rejection. Future arrivals, unequal
    request reservations and other serving contracts retain full execution.
    """
    config = runtime._external_ingress
    if not config.get("preload_warmup_seed_kv", False):
        return
    batch, degree = runtime.plan.scheduler.max_num_seqs, runtime._ep_degree
    pool = config.get("max_inflight_requests", 2)
    if type(pool) is not int or pool < degree or pool % degree:
        return
    seeds, fresh = [], []
    for state in runtime._state_values:
        (seeds if state.spec.metadata.get("warmup_seed", False) else fresh).append(state)
    seed_ids = {state.spec.request_id for state in seeds}
    if (len(seeds) != batch or runtime._external_preloaded_requests != batch
            or len(cohort.items) != batch or set(cohort.request_ids) != seed_ids
            or len(fresh) < pool
            or any(state.spec.arrival_ns > runtime.now for state in (*seeds, *fresh))
            or any(state.status != RequestStatus.WAITING for state in fresh)
            or any(state.spec.request_id not in runtime._external_initialized for state in seeds)):
        return
    reservations = {_max_live_kv_tokens(state.spec.prompt_tokens, state.spec.output_tokens)
                    for state in (*seeds, *fresh)}
    if len(reservations) != 1:
        return
    if any(type(state.ep_group) is not int or not 0 <= state.ep_group < degree for state in seeds):
        return
    occupancy = _import_pool_occupancy(runtime, degree)
    target = pool // degree
    deficient = [group for group, count in enumerate(occupancy) if count < target]
    if not deficient:
        return
    held_ids = runtime._external_initialized | set(runtime._external_pending) | runtime._external_ready
    candidates = [state for state in fresh if state.spec.request_id not in held_ids]
    pages = runtime.ledger.pages_for_tokens(next(iter(reservations)))
    for group in deficient:
        eligible = [state for state in candidates if state.ep_group in (None, group)]
        if len(eligible) < target - occupancy[group]:
            return
        for state in eligible:
            # A request without a group cannot be queried against an EP
            # ledger. Bind a private probe without invoking the live state's
            # trace callback or changing scheduler/group accounting.
            probe = copy(state)
            object.__setattr__(probe, "ep_group", group)
            if runtime.ledger.can_resize(probe, pages):
                return
    held = list(occupancy)
    for state in seeds:
        held[state.ep_group] += 1
    diagnostics.update(physical_admission_cutoff_ns=runtime.now,
                       kv_requests_held_by_group=held)
    raise NativeLayerSamplingGuardError("native_pipeline_import_pool_not_full")
