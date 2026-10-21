"""Conditional composition of real native layer-window samples.

Sampling produces independent native-kernel traces, not a full cohort
prefix. This module composes values only. It cannot commit a Token, resume
the sampled kernel, or turn an omitted task into a physical transaction.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
from statistics import fmean
from typing import Mapping


def _finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _midpoint(lower, upper):
    return lower+((upper-lower)/upper)*(lower/(1+lower/upper))


def _interval(lower, upper):
    if lower == 0:
        return {"estimate": upper/2, "lower": 0., "upper": upper,
            "relative_error_budget": None}
    estimate = _midpoint(lower, upper)
    return {"estimate": estimate, "lower": lower, "upper": upper,
        "relative_error_budget": max(estimate/lower-1., 1.-estimate/upper)}


def _within(interval, target):
    value, lower, upper = (Fraction(interval[key]) for key in ("estimate", "lower", "upper"))
    target = Fraction(target)
    return value <= (1+target)*lower and value >= (1-target)*upper


def _class_key(row):
    # Native LayerSpec semantics are carried as a primitive tuple/list/dict.
    # A stable in-process key is enough; no hash/freeze protocol is needed.
    return repr(row["class_key"])


def _phase(summary):
    if (not isinstance(summary, Mapping)
            or summary.get("schema") != "native-window-ir-observations/v1"
            or summary.get("unsupported") or summary.get("stateful_gpu_cache")):
        raise ValueError("native_layer_physical_phase_summary_unavailable")
    value = summary.get("ordered_resource_phase_per_cohort_ns")
    if not _finite(value):
        raise ValueError("native_layer_row_cache_refresh_phase_bound_unavailable")
    # Refresh is already included in this dependency-aware phase path.
    # Never replace the active-IO REF feedback by one tRFC per owner.
    return value


def _issue_order_fixed(summary):
    # Every phase guard is an excess over the best-case reference D0. With a
    # fixed native issue order the duration is also monotone, so a sample lies
    # in [D0, D0+G]. Without that proof keep the symmetric +-(Gc+Gt) rule.
    proof = summary.get("native_issue_order_proof") if isinstance(summary, Mapping) else None
    return isinstance(proof, Mapping) and proof.get("qualified") is True


def _hbm_clock_limit(summaries):
    limits = []
    for summary in summaries:
        value = summary.get("hbm_clock_domain_limit_ns")
        if value is not None:
            if not _finite(value, positive=True):
                raise ValueError("native_layer_hbm_clock_domain_invalid")
            limits.append(value)
    return min(limits, default=None)


def _close_hbm_clock_domain(result, layer_budgets, forecast_upper_end_ns, *, phase_proofs=()):
    """Close conditional CK profiles over the entire native lifecycle.

    The restricted per-IO bound is valid only while the absolute native
    clock remains in its proven domain. Summing all forecast event uppers,
    every future PD import, carry and cleanup proves by induction that no
    event leaves that domain. A short measurement window alone is not enough.
    Independent physical samples must also lie within the same domain.
    """
    budgets = tuple(layer_budgets)
    limit = _hbm_clock_limit((*budgets, *phase_proofs))
    if limit is None:
        return
    ends = [row.get("physical_sample_completion_upper_ns") for row in budgets
            if row.get("hbm_clock_domain_limit_ns") is not None]
    qualified = (_finite(forecast_upper_end_ns) and forecast_upper_end_ns <= limit
                 and all(_finite(value) and value <= limit for value in ends))
    result["error_budget"]["native_hbm_clock_domain"] = {
        "qualified": qualified, "absolute_clock_limit_ns": limit,
        "forecast_full_lifecycle_upper_end_ns": forecast_upper_end_ns,
        "physical_sample_upper_end_ns": max(ends, default=None) if all(_finite(value) for value in ends) else None,
        "scope": "all_native_warmup_measurement_and_cleanup_events_plus_all_future_PD_and_carry; independent_samples_also_checked",
        "basis": "conditional_native_CK_profile_closed_by_full_lifecycle_upper_bound_induction"}
    if not qualified:
        result["reasons"].append("native_layer_full_lifecycle_exceeds_proven_hbm_clock_domain")


def _sample_duration(row):
    if (row.get("sample_state") != "independent_native_kernel_copy_not_full_cohort_prefix"
            or row.get("visible_output_tokens", 0) != 0
            or row.get("full_cohort_committed", False)
            or not _finite(row.get("start_ns")) or not _finite(row.get("end_ns"))
            or not _finite(row.get("duration_ns"), positive=True)
            or row["end_ns"] < row["start_ns"]
            or row["end_ns"]-row["start_ns"] != row["duration_ns"]):
        raise ValueError("native_layer_sample_duration_or_state_invalid")
    return row["duration_ns"]


def evaluate_native_layer_windows(samples, *, target_relative_error=.10,
                                  independent_sufficient=True):
    """Require independent same-class holdout and all composition guards.

    The error is conditional on native same-class service continuation.
    Physical address/row/cache/refresh and boundary phases are independent
    explicit interval terms, not evidence inferred from a small residual.
    """
    if not _finite(target_relative_error, positive=True) or target_relative_error > .1:
        raise ValueError("native layer target must be in (0,.1]")
    result = {"accepted": False, "reasons": [], "cohort_interval": None,
        "layer_forecasts": [], "error_budget": {"conditional_only": True,
            "target_relative_error": target_relative_error,
            "effective_independent_target": (target_relative_error/(2+target_relative_error)
                                             if independent_sufficient else target_relative_error),
            "physical_state_advanced": False, "forecast_has_physical_trace": False,
            "same_class_assumption": "fixed native LayerSpec semantics, batch/context shape, IO geometry and operator multiset; disjoint layer holdout bounds declared continuation residual"},
        "physical_sampled_layer_count": 0, "forecast_layer_count": 0,
        "physical_visible_output_tokens": 0}
    reasons, budget = result["reasons"], result["error_budget"]
    if not isinstance(samples, Mapping) or samples.get("schema") != "native-layer-physical-samples/v1":
        reasons.append("native_layer_samples_contract_invalid")
        return result
    composition = samples.get("layer_composition", {})
    if (composition.get("qualified") is not True
            or not composition.get("basis")
            or composition.get("serial_layer_boundaries_proven") is not True):
        reasons.append("native_layer_serial_composition_not_proven")
    if (samples.get("physical_visible_output_tokens", 0) != 0
            or samples.get("visible_output_tokens", 0) != 0
            or samples.get("full_cohorts_committed", 0) != 0):
        reasons.append("native_layer_samples_must_not_commit_full_cohort_tokens")
    inventory = tuple(samples.get("layer_inventory", ()))
    ids = [row.get("layer_id") for row in inventory]
    if not inventory or any(not isinstance(identity, str) or not identity for identity in ids) or len(set(ids)) != len(ids):
        reasons.append("native_layer_inventory_invalid")
    if reasons:
        return result
    by_class, by_id = defaultdict(list), {row["layer_id"]: row for row in inventory}
    for row in inventory:
        if "class_key" not in row or not isinstance(row.get("signature"), Mapping):
            reasons.append("native_layer_native_semantic_class_missing")
            continue
        by_class[_class_key(row)].append(row)
    windows = tuple(samples.get("layer_window_samples", ()))
    grouped, selected_ids, sample_ids = defaultdict(list), set(), set()
    try:
        for row in windows:
            duration = _sample_duration(row)
            identity = row.get("sample_id")
            layer_ids = tuple(row.get("layer_ids", ()))
            if (not isinstance(identity, str) or identity in sample_ids or not layer_ids
                    or len(set(layer_ids)) != len(layer_ids) or any(layer not in by_id for layer in layer_ids)):
                raise ValueError("native_layer_window_identities_invalid")
            if selected_ids.intersection(layer_ids):
                raise ValueError("native_layer_calibration_and_holdout_layers_overlap")
            sample_ids.add(identity); selected_ids.update(layer_ids)
            keys = {_class_key(by_id[layer]) for layer in layer_ids}
            if len(keys) != 1 or _class_key(row) not in keys:
                raise ValueError("native_layer_window_crosses_semantic_classes")
            if row.get("role") not in {"calibration", "validation", "exhaustive"}:
                raise ValueError("native_layer_sample_role_invalid")
            _phase(row.get("native_ir_summary"))
            if row.get("input_boundary", {}).get("qualified") is not True:
                raise ValueError("native_layer_omitted_input_boundary_not_bounded")
            if row.get("output_boundary", {}).get("qualified") is not True:
                raise ValueError("native_layer_output_boundary_not_bounded")
            if not _finite(row.get("input_boundary", {}).get("initial_carry_upper_ns")):
                raise ValueError("native_layer_source_carry_bound_unavailable")
            grouped[next(iter(keys))].append((row, duration/len(layer_ids)))
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(str(error))
    class_intervals, class_rows = {}, []
    try:
        for key, layers in by_class.items():
            rows = grouped.get(key, ())
            train = [(row, value) for row, value in rows if row["role"] == "calibration"]
            holdout = [(row, value) for row, value in rows if row["role"] == "validation"]
            # A larger requested window can exhaust a class with more than
            # three layers. Complete physical coverage, not class size,
            # removes the need to infer unseen same-class layers. These
            # independent source copies still pay every phase guard below.
            class_ids = {row["layer_id"] for row in layers}
            covered_ids = {identity for row, _value in rows for identity in row["layer_ids"]}
            exhaustive_rows = [row for row, _value in rows if row["role"] == "exhaustive"]
            exhaustive = bool(rows) and len(exhaustive_rows) == len(rows) and covered_ids == class_ids
            if exhaustive_rows and not exhaustive:
                raise ValueError("native_layer_exhaustive_class_coverage_or_roles_invalid")
            if not rows or (not exhaustive and (len(train) < 2 or len(holdout) < 2)):
                raise ValueError("native_layer_disjoint_semantic_class_samples_insufficient")
            fit = train if train else list(rows)
            base = fmean(value for _row, value in fit)
            local = max((abs(value-base)/base for _row, value in (holdout or rows)), default=0.)
            local = max(local, (max(value for _row, value in fit)-min(value for _row, value in fit))/base)
            if not _finite(local) or local >= 1:
                raise ValueError("native_layer_class_calibration_unbounded")
            lo, hi = base/(1+local), base/(1-local)
            prefix_phase = max(_phase(row["native_ir_summary"])/len(row["layer_ids"])
                               for row, _value in fit)
            # First-task start already excludes the entry wait. Residual
            # source work can still overlap later selected tasks, and may
            # only inflate that independent calibration duration. Remove
            # its possible bias from the lower service bound; it is never
            # multiplied into the predicted nominal layer service.
            prefix_carry = max(max(0., row["input_boundary"]["initial_carry_upper_ns"]
                -row.get("boundary_wait_ns", 0.))/len(row["layer_ids"])
                for row, _value in fit)
            one_sided = all(_issue_order_fixed(row["native_ir_summary"]) for row, _value in rows)
            class_intervals[key] = (lo, hi, prefix_phase, prefix_carry, one_sided)
            class_rows.append({"class_key": layers[0]["class_key"], "layer_count": len(layers),
                "calibration_windows": len(train), "holdout_windows": len(holdout),
                "disjoint_layer_ids": True, "exhaustive_class": exhaustive,
                "issue_order_proven_for_all_windows": one_sided,
                "base_service_per_layer_ns": base, "holdout_and_drift_relative_error": local,
                "calibration_initial_state_bias_guard_per_layer_ns": prefix_phase,
                "calibration_source_carry_bias_guard_per_layer_ns": prefix_carry})
    except (KeyError, ValueError) as error:
        reasons.append(str(error))
    if reasons:
        budget["classes"] = class_rows
        return result
    budget["classes"] = class_rows
    budget["composition"] = composition
    lower, upper, phase_total = 0., 0., 0.
    one_sided_layers, symmetric_components = 0, []
    try:
        for layer in inventory:
            lo, hi, prefix_phase, prefix_carry, class_one_sided = class_intervals[_class_key(layer)]
            phase = _phase(layer.get("native_ir_summary"))
            # Different sampled layers begin at independent copies of the
            # source state. Even selected IDs require the full-composition
            # phase guard; their timings are not claimed exact in situ.
            combined_phase = prefix_phase+phase
            one_sided = class_one_sided and _issue_order_fixed(layer.get("native_ir_summary"))
            if one_sided:
                # Calibration S lies in [D0, D0+Gc] and the target T in
                # [D0', D0'+Gt]; the class drift term already covers D0'-D0.
                lo, hi = max(0., lo-prefix_phase-prefix_carry), hi+phase
                one_sided_layers += 1
            else:
                lo, hi = max(0., lo-combined_phase-prefix_carry), hi+combined_phase
                symmetric_components.append(layer["layer_id"])
            lower += lo; upper += hi; phase_total += combined_phase
            result["layer_forecasts"].append({"layer_id": layer["layer_id"],
                "sampled_in_independent_kernel": layer["layer_id"] in selected_ids,
                "forecast_only_in_full_cohort": True, "interval": _interval(lo, hi),
                "physical_phase_guard_ns": phase,
                "calibration_initial_state_bias_guard_ns": prefix_phase,
                "calibration_source_carry_lower_bias_guard_ns": prefix_carry,
                "one_sided_guard_composition": one_sided})
        non_layers = tuple(samples.get("non_layer_samples", ()))
        if not non_layers or samples.get("non_layer_coverage_complete") is not True:
            raise ValueError("native_layer_embedding_head_host_coverage_incomplete")
        non_layer_ns = non_layer_phase = 0.
        one_sided_non_layers = 0
        for row in non_layers:
            duration, phase = _sample_duration(row), _phase(row.get("native_ir_summary"))
            if row.get("input_boundary", {}).get("qualified") is not True or row.get("output_boundary", {}).get("qualified") is not True:
                raise ValueError("native_nonlayer_boundary_not_bounded")
            carry = row.get("input_boundary", {}).get("initial_carry_upper_ns")
            if not _finite(carry):
                raise ValueError("native_nonlayer_source_carry_bound_unavailable")
            carry = max(0., carry-row.get("boundary_wait_ns", 0.))
            # The sample and its in-situ target run the same IR: one guard
            # per side when monotone, otherwise both guards on each side.
            if _issue_order_fixed(row.get("native_ir_summary")):
                one_sided_non_layers += 1
            else:
                phase *= 2
                symmetric_components.append(row.get("sample_id"))
            lower += max(0., duration-phase-carry); upper += duration+phase
            non_layer_ns += duration; non_layer_phase += phase
        boundary = composition.get("boundary_guard_ns")
        if not _finite(boundary):
            raise ValueError("native_layer_cross_boundary_service_guard_unavailable")
        source_carry = composition.get("source_pending_carry_upper_ns")
        if not _finite(source_carry):
            raise ValueError("native_layer_full_cohort_source_carry_bound_unavailable")
        if lower <= boundary:
            raise ValueError("native_layer_boundary_exhausts_cohort_lower_bound")
        lower -= boundary; upper += boundary+source_carry
    except (KeyError, ValueError) as error:
        reasons.append(str(error))
        budget.update(full_cohort_physical_phase_guard_ns=phase_total,
            partial_full_cohort_lower_ns=lower, partial_full_cohort_upper_ns=upper)
        return result
    interval = _interval(lower, upper)
    limit = target_relative_error/(2+target_relative_error) if independent_sufficient else target_relative_error
    if not _within(interval, limit):
        reasons.append("native_layer_conditional_single_or_independent_pair_budget_exceeded")
    budget.update(classes=class_rows, full_cohort_physical_phase_guard_ns=phase_total,
        guard_composition={"one_sided_layers": one_sided_layers,
            "one_sided_non_layer_samples": one_sided_non_layers,
            "symmetric_fallback_components": symmetric_components,
            "basis": "phase_guards_are_excess_over_best_case_reference; one_sided_[S-Gc,S+Gt]_only_with_native_issue_order_proof_for_target_and_all_class_windows; else_symmetric_+-(Gc+Gt)"},
        embedding_head_host_sample_service_ns=non_layer_ns,
        # Upper-side guard actually applied to the embedding/head/host samples.
        embedding_head_host_phase_guard_ns=non_layer_phase,
        cross_boundary_guard_ns=boundary,
        full_cohort_source_pending_carry_upper_ns=source_carry,
        composition=composition, interval_semantics="conditional_full_cohort_values_not_a_physical_trace")
    try:
        budget["hbm_clock_domain_limit_ns"] = _hbm_clock_limit(
            row.get("native_ir_summary", {}) for row in (*inventory, *windows, *non_layers))
    except ValueError as error:
        reasons.append(str(error))
    budget["physical_sample_completion_upper_ns"] = max(row["end_ns"] for row in (*windows, *non_layers))
    result.update(accepted=not reasons, cohort_interval=interval,
        physical_sampled_layer_count=len(selected_ids), forecast_layer_count=len(inventory)-len(selected_ids))
    return result


def _query_decode_contract(runtime, cohort, stages, *, query_cost=None):
    """Compare one final-query IR to Decode at the identical native context."""
    from dataclasses import replace
    from .serving import _lower_cost

    if not cohort.items or cohort.kind != "prefill" or any(item.token_count != 1 for item in cohort.items):
        raise ValueError("native_layer_window_requires_final_prompt_query_batch")
    items = tuple(replace(original, phase="decode", token_count=1, main_tokens=1,
        draft_tokens=0, verifier_tokens=1, committed_tokens=1,
        kv_append_tokens=1, kv_materialized_tokens=1, logit_tokens=1, completion_cursor=2)
        for original in cohort.items)
    projected = replace(cohort, cohort_id=cohort.cohort_id+".native-layer-decode-proof",
                        kind="decode", items=items)
    runtime.execution_control.raise_if_cancelled()
    cost = _lower_cost(runtime.lowerer, runtime.plan.scenario, projected)
    candidate, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
    if fallback or not candidate:
        raise ValueError("native_layer_decode_ir_stage_contract_unavailable")
    candidate = runtime._with_gpu_controller_stages(candidate, projected)
    runtime.execution_control.raise_if_cancelled()

    def describe(stage_rows):
        tasks = [task for stage in stage_rows for task in stage.execution_tasks]
        positions = {task.task_id: index for index, task in enumerate(tasks)}
        rows = []
        for task in tasks:
            md = task.metadata
            accesses = tuple((access.get("physical_owner", md.get("physical_owner")),
                access.get("operation"), access.get("byte_count"),
                access.get("allocation_generation", 0) == 0,
                str(access.get("buffer_id", "")).startswith("kv:"))
                for access in md.get("memory_accesses", ()))
            demands = tuple((row.resource_id, row.service_ns, row.bytes_moved, row.work_units)
                            for row in task.demands)
            rows.append((tuple(positions.get(dep, -1) for dep in task.dependencies),
                str(task.category), str(md.get("operator_class", "")),
                md.get("layer_id"), md.get("rank"), md.get("expert_index"),
                md.get("expert_tokens"), accesses, demands))
        return rows

    left, right = describe(stages), describe(candidate)
    if len(left) != len(right):
        raise ValueError("native_layer_query_decode_task_counts_differ")
    delta = 0.
    for before, after in zip(left, right):
        # Demand service differences remain charged as a serial Lipschitz
        # guard. Changed physical operations/bytes or resource graph shape
        # need an additional real sample instead of an analytical shortcut.
        if before[:-1] != after[:-1]:
            raise ValueError("native_layer_query_decode_native_graph_or_io_shape_differs")
        bd, ad = before[-1], after[-1]
        if len(bd) != len(ad) or any((x[0], x[2], x[3]) != (y[0], y[2], y[3])
                                   for x, y in zip(bd, ad)):
            raise ValueError("native_layer_query_decode_resource_work_shape_differs")
        delta += sum(abs(x[1]-y[1]) for x, y in zip(bd, ad))
    future_phase = _future_context_phase_contract(runtime, projected, candidate)
    future_query_phase = _future_context_phase_contract(runtime, cohort, stages)
    from .native_layer_bounds import native_layer_boundary_bounds
    decode_boundary = native_layer_boundary_bounds(runtime.plan.scenario, cost, candidate)
    query_boundary = (native_layer_boundary_bounds(runtime.plan.scenario, query_cost, stages)
                      if query_cost is not None else {})
    lag = (0. if query_boundary.get("visible_commit_lag_upper_ns") == 0.
           and decode_boundary.get("visible_commit_lag_upper_ns") == 0. else None)
    return {"qualified": True, "basis": "same_native_lowerer_Decode_IR_at_identical_query_context; graph_IO_and_work_shapes_equal",
        "task_count": len(left), "nonphysical_service_difference_serial_guard_ns": delta,
        "future_context_phase_contract": future_phase,
        "future_query_context_phase_contract": future_query_phase,
        "query_native_boundary": query_boundary, "decode_native_boundary": decode_boundary,
        "visible_commit_lag_upper_ns": lag,
        "batch_invariant_native_path_floor_ns": min(query_boundary.get("batch_invariant_native_path_floor_ns", 0.),
                                                   decode_boundary.get("batch_invariant_native_path_floor_ns", 0.)),
        "physical_decode_executed": False}


def _future_context_phase_contract(runtime, decode_cohort, anchor_stages):
    """Read native IR at full/chunk corners; preserve discrete phase jumps."""
    from dataclasses import replace
    from .serving import _lower_cost
    from .physical_contract import parse_execution_physical_config
    from .native_window_producer import (_hbm_channels_independent, _nand_cache_release_phase,
        _native_cold_ref_service, _native_refresh_source_qualification,
        _phase as physical_phase)
    from .memory_types import DramConfig, NandConfig

    requests = runtime.plan.requests
    lower = min(row.prompt_tokens-1 for row in requests)
    upper = max((row.prompt_tokens-1 if decode_cohort.kind == "prefill"
                 else row.prompt_tokens+row.output_tokens-2) for row in requests)
    chunks = {instance.layer.attention_chunk_size for instance
              in runtime.plan.scenario.model._execution_view.layer_instances
              if instance.layer.attention_chunk_size}
    corners = {lower, upper}
    for chunk in chunks:
        end = ((upper+1)//chunk)*chunk-1
        if end >= lower:
            corners.add(end)
        reset = (upper//chunk)*chunk
        if reset >= lower:
            corners.add(reset)
    fixed_context = len(corners) == 1 and all(item.context_tokens == lower for item in decode_cohort.items)
    physical = runtime._execution_resource_kernel.physical_runtime
    parsed, clock_limits, gpu_profiles = {}, [], {}

    # These native formulas use fixed positive throughput/profile constants,
    # nonnegative linear work, integer ceil MMA tiles/waves and ceil(log2)
    # reduction depth. All are monotone in effective attention context. A
    # measured kernel surface, source/quantized dispatch or unknown role has
    # no such endpoint proof and is rejected instead of inferred from two
    # equal endpoint observations.
    from .planner import TopologyAwareBatchCostProvider, _gpu_profiles
    from .native_window_producer import _context_sensitive
    if type(runtime.lowerer) is not TopologyAwareBatchCostProvider:
        raise ValueError("native_layer_future_nonphysical_unknown_lowerer")
    if any(runtime.plan.scenario.placement.metadata.get(key) for key in (
            "native_calibration_apply_stage", "native_calibration_apply_memory",
            "native_calibration_apply_launch")):
        raise ValueError("native_layer_future_nonphysical_calibrated_model_not_bounded")
    layers = {instance.layer.layer_id: instance.layer
              for instance in runtime.plan.scenario.model._execution_view.layer_instances}
    mmu_proof = {}

    def nonphysical(row):
        physical_ids = {access.get("resource_id") for access in row.metadata.get("memory_accesses", ())}
        terms = {}
        seen = set()
        for demand in row.demands:
            if demand.resource_id in seen:
                # Native resource ordering for duplicates cannot be
                # represented by overwriting one entry in a dictionary.
                raise ValueError("native_layer_future_duplicate_resource_demand")
            seen.add(demand.resource_id)
            if demand.resource_id not in physical_ids:
                terms[demand.resource_id] = demand
        return terms

    def context_sensitive(md):
        return _context_sensitive(md) or ".attention." in str(md.get("operator_id", ""))

    def qualify_physical_context_traffic(row):
        md = row.metadata
        if (fixed_context or not context_sensitive(md)
                or not any(access.get("operation") == "read"
                           for access in md.get("memory_accesses", ()))):
            return
        model = md.get("cost_model")
        if not isinstance(model, Mapping):
            model = md.get("native_cost_model")
        if not isinstance(model, Mapping):
            # Native transfer/endpoint geometry is separately qualified by
            # the controller contract. It has no GPU aggregate hit curve.
            return
        cache = model.get("cache")
        if not isinstance(cache, Mapping):
            raise ValueError("native_layer_future_physical_cache_source_not_bounded")
        target = md.get("target_component")
        if target not in gpu_profiles:
            gpu_profiles[target] = _gpu_profiles(runtime.plan.scenario, target)[0]
        if not gpu_profiles[target].cache_hierarchy.levels:
            return
        reuse, streaming = cache.get("reuse_factor"), cache.get("streaming_fraction")
        if (cache.get("cache_model") != "v4_directional_closed_interval"
                or cache.get("cross_invocation_cache_state") is not False
                or not _finite(reuse, positive=True) or not _finite(streaming)
                or streaming > 1):
            raise ValueError("native_layer_future_physical_cache_source_not_bounded")
        if reuse > 1 and streaming < 1:
            # A capacity crossing may grow backing reads faster than context:
            # R-C/2 at reuse=2 triples misses when R goes from C to 2C.
            # Full-payload cache/initial-state bounds do not cover that bulk
            # difference. Do not silently reuse the affine continuation.
            raise ValueError("native_layer_future_physical_cache_bulk_not_affine")

    def context_envelope(row):
        """Per-demand native upper/lower over a proven context segment.

        Cache lookup/reuse fractions need not be monotone. Bound every cache
        by the full logical payload through that level, and use zero below;
        this retains integer line/port/wave latencies without assuming hits.
        It creates no cache access or physical execution.
        """
        md, terms = row.metadata, nonphysical(row)
        if md.get("event_kind") == "gpu_mmu_tlb_batch":
            bound = mmu_proof.get("task_bounds", {}).get(row.task_id)
            if not bound or set(terms) != {bound["resource_id"]}:
                raise ValueError("native_layer_future_MMU_native_contract_unavailable")
            constant = terms[bound["resource_id"]].service_ns
            return {bound["resource_id"]: ((constant, constant) if fixed_context
                    else (bound["lower_ns"], bound["upper_ns"]))}, {
                "native_role": "gpu_mmu_tlb_batch", "basis": mmu_proof["basis"],
                "conditional_only": True}
        if not terms or not context_sensitive(md):
            return {key: (value.service_ns, value.service_ns) for key, value in terms.items()}, None
        model = md.get("cost_model")
        if not isinstance(model, Mapping):
            model = md.get("native_cost_model")
        kind = model.get("model") if isinstance(model, Mapping) else None
        role = md.get("event_kind", "")
        operator = str(md.get("operator_id", ""))
        layer = layers.get(md.get("layer_id"))
        if (layer is None or layer.quantization is not None
                or layer.dtype.lower() not in {"bf16", "fp16", "fp32"}):
            raise ValueError("native_layer_future_nonphysical_native_role_not_bounded")
        supported = ((kind == "gpu_fused_attention_v3" and role == "fused_attention")
            or (kind == "gpu_memory_roofline" and role == "kv_materialize"
                and operator == layer.layer_id+".attention.kv_materialize")
            or (kind in {"gpu_reduction_roofline", "gpu_elementwise_roofline"}
                and role in {"softmax_reduce", "softmax_normalize", "attention_qk_scale",
                    "rope", "rope_table_generate", "attention_temperature",
                    "attention_q_norm_reduce", "attention_q_norm_apply",
                    "attention_k_norm_reduce", "attention_k_norm_apply",
                    "attention_gate", "attention_residual", "attention_context_contiguous"})
            or (kind == "gpu_hbm_roofline"
                and operator in {layer.layer_id+".attention.qk", layer.layer_id+".attention.pv"}))
        if not supported:
            raise ValueError("native_layer_future_nonphysical_native_role_not_bounded:"
                +repr((md.get("layer_id"), kind, role, operator)))
        if role == "kv_materialize" and (
                md.get("timing_completeness") != "complete_copy"
                or md.get("conversion_instructions_priced") is not True
                or md.get("cache_input_bits") != md.get("cache_output_bits")
                or md.get("cache_input_bits") not in (16, 32)
                or md.get("cache_artifact_format") is not None
                or md.get("unpriced_terms")):
            raise ValueError("native_layer_future_KV_materialization_conversion_not_bounded")
        if (model.get("tensor_dtype", "bf16") not in {"bf16", "fp16", "fp32"}
                or model.get("quantized_weight_path") or model.get("source_partial_service_ns", 0.)
                or model.get("quantized_shape_efficiency") or model.get("mmvq_work")
                or model.get("mmq_work") or md.get("source_f32_kv")):
            raise ValueError("native_layer_future_nonphysical_quantized_or_source_path_not_bounded")
        target = md.get("target_component")
        if target not in gpu_profiles:
            gpu_profiles[target] = _gpu_profiles(runtime.plan.scenario, target)[0]
        gpu = gpu_profiles[target]
        if gpu.kernel_model is not None or model.get("profile") != gpu.name:
            raise ValueError("native_layer_future_nonphysical_dynamic_profile_not_bounded")
        if md.get("phase") == "kernel_launch":
            expected = gpu.launch_resource_id.split(".", 1)[-1]
            if (len(terms) != 1 or any(
                    key.split(".", 1)[-1] != expected
                    or demand.bytes_moved != 0 or demand.work_units != 0
                    or demand.service_ns != gpu.kernel_launch_ns
                    for key, demand in terms.items())):
                raise ValueError("native_layer_future_nonphysical_launch_not_native_constant")
            return {key: (d.service_ns, d.service_ns) for key, d in terms.items()}, None
        cache = model.get("cache", {})
        read, write = cache.get("logical_read_bytes"), cache.get("logical_write_bytes")
        if (not _finite(read) or not _finite(write)
                or (cache.get("cross_invocation_cache_state") is not False
                    and cache.get("cache_model") != "none")):
            raise ValueError("native_layer_future_nonphysical_cache_contract_not_bounded")
        cache_levels = {level.resource_id: level for level in gpu.cache_hierarchy.levels}
        # Fused attention and KV materialization are lowered with the fixed
        # constants reuse=1, streaming=1 at every context. The closed-interval
        # cache then has no hits: each level serves the whole logical read
        # stream plus allocated writes, i.e. its full-payload formula exactly.
        # That formula is monotone in the payload, which is monotone in
        # context, so its corner minimum is a lower bound on the segment.
        no_reuse = ((kind == "gpu_fused_attention_v3" or kind == "gpu_memory_roofline")
            and cache.get("cache_model") == "v4_directional_closed_interval"
            and cache.get("reuse_factor") == 1.0 and cache.get("streaming_fraction") == 1.0
            and type(read) is int and type(write) is int)
        allocated = write if gpu.cache_hierarchy.write_allocate else 0

        def level_service(level, reads, writes):
            waves = max(-(-(-(-reads//level.line_bytes))//(level.banks*level.read_ports*level.max_outstanding)),
                -(-(-(-writes//level.line_bytes))//(level.banks*level.write_ports*level.max_outstanding)))
            return max((reads+writes)/level.bandwidth_gb_s, waves*level.hit_latency_ns)

        # Native lowering namespaces profile resource IDs to the actual GPU.
        def bound(key, demand):
            if fixed_context:
                # The qualified native, stateless formula sees identical
                # inputs. Physical row/REF state remains separately charged.
                return demand.service_ns, demand.service_ns
            local = key.split(".", 1)[-1]
            level = next((level for identity, level in cache_levels.items()
                          if identity == key or identity.split(".", 1)[-1] == local), None)
            if level is None:
                return demand.service_ns, demand.service_ns
            reads = math.ceil(read/level.line_bytes)
            writes = math.ceil(write/level.line_bytes)
            waves = max(math.ceil(reads/(level.banks*level.read_ports*level.max_outstanding)),
                math.ceil(writes/(level.banks*level.write_ports*level.max_outstanding)))
            lower = level_service(level, read, allocated) if no_reuse else 0.
            if lower > demand.service_ns:
                # The native demand is not this closed-interval stream.
                lower = 0.
            return lower, max((read+write)/level.bandwidth_gb_s, waves*level.hit_latency_ns)
        return {key: bound(key, demand) for key, demand in terms.items()}, {
            "cost_model": kind, "native_role": role or operator,
            "basis": "fixed_native_positive_throughput_max_linear_work_ceil_MMA_tiles_waves_and_log_depth; cache_full_payload_upper; no_reuse_closed_interval_payload_lower" if no_reuse
                else "fixed_native_positive_throughput_max_linear_work_ceil_MMA_tiles_waves_and_log_depth; cache_full_payload_upper_zero_lower",
            "conditional_only": True}

    # This contract only reads IR; no physical state advances while it runs.
    # Source proofs and per-geometry phases are therefore invariant per owner
    # and per (config, size, direction), as in summarize_native_stages.
    source_proofs, phases = {}, {}

    def phase(access, metadata):
        owner = access.get("physical_owner", metadata.get("physical_owner"))
        raw = metadata.get("physical_memory_configs", {}).get(owner, metadata.get("physical_memory_config"))
        if id(raw) not in parsed:
            parsed[id(raw)] = parse_execution_physical_config(raw)
        config = parsed[id(raw)]
        count, operation = access["byte_count"], access["operation"]
        reachable = False
        if isinstance(config, DramConfig) and config.control_mode == "jedec_hbm":
            key = (owner, id(config))
            if key not in source_proofs:
                source_proofs[key] = _native_refresh_source_qualification(config, owner, physical, runtime)
            source_proof = source_proofs[key]
            if source_proof["qualified"] is not True:
                raise ValueError("native_layer_future_HBM_source_not_closed:" + source_proof["reason"])
            reachable = True
        key = (id(config), count, operation, reachable)
        if key not in phases:
            independent = (isinstance(config, DramConfig)
                           and _hbm_channels_independent(config, physical))
            row, refresh, cache = physical_phase(config, count, operation,
                independent_hbm_channels=independent, allow_bounded_clock=True,
                native_refresh_reachable=reachable)
            cold = (_native_cold_ref_service(config, count, operation, independent_channels=True,
                    native_refresh_reachable=reachable)
                    if independent else None)
            if cold is not None:
                clock_limits.append(cold["hbm_clock_domain_limit_ns"])
            if operation == "read" and isinstance(config, NandConfig):
                release = _nand_cache_release_phase(config, count, physical)
                if release is not None:
                    cache = min(cache, release)
            phases[key] = row+refresh+cache
        return phases[key]

    anchor = [task for stage in anchor_stages for task in stage.execution_tasks]
    before_demands = [nonphysical(task) for task in anchor]
    before_phases = [[phase(access, task.metadata) for access in task.metadata.get("memory_accesses", ())]
                     for task in anchor]
    per_access_extra = [[0. for _phase in phases] for phases in before_phases]
    anchor_positions = {task.task_id: index for index, task in enumerate(anchor)}
    from .native_layer_bounds import native_mmu_context_bounds
    mmu_payload_maxima = {}
    mmu_proof = native_mmu_context_bounds(runtime, decode_cohort, anchor_stages,
        payload_maxima=mmu_payload_maxima)
    for task in anchor:
        qualify_physical_context_traffic(task)
    # Qualification is required even for equal endpoint services. Otherwise
    # an unknown cost function could hide a larger interior value.
    anchor_envelopes = [context_envelope(task)[0] for task in anchor]
    nonphysical_upper = [[max(d.service_ns, envelope[key][1]) for key, d in values.items()]
                        for values, envelope in zip(before_demands, anchor_envelopes)]
    nonphysical_lower = [[min(d.service_ns, envelope[key][0]) for key, d in values.items()]
                        for values, envelope in zip(before_demands, anchor_envelopes)]
    demand_proofs = []
    proofs = []
    for context in sorted(corners):
        runtime.execution_control.raise_if_cancelled()
        projected = replace(decode_cohort,
            cohort_id=decode_cohort.cohort_id+".context-corner-"+str(context),
            items=tuple(replace(item, context_tokens=context) for item in decode_cohort.items))
        cost = _lower_cost(runtime.lowerer, runtime.plan.scenario, projected)
        stages, fallback = runtime._execution_stage_metadata_cache.resolve(cost.metadata)
        if fallback or not stages:
            raise ValueError("native_layer_future_context_IR_unavailable")
        stages = runtime._with_gpu_controller_stages(stages, projected)
        mmu_proof = native_mmu_context_bounds(runtime, projected, stages,
            payload_maxima=mmu_payload_maxima)
        after = [task for stage in stages for task in stage.execution_tasks]
        if len(anchor) != len(after):
            raise ValueError("native_layer_future_context_changes_native_task_population")
        after_positions = {task.task_id: index for index, task in enumerate(after)}
        extra = 0.
        for index, (a, b) in enumerate(zip(anchor, after)):
            qualify_physical_context_traffic(b)
            aa, ba = tuple(a.metadata.get("memory_accesses", ())), tuple(b.metadata.get("memory_accesses", ()))
            if (a.category != b.category or a.metadata.get("operator_class") != b.metadata.get("operator_class")
                    or a.metadata.get("layer_id") != b.metadata.get("layer_id")
                    or a.metadata.get("rank") != b.metadata.get("rank") or len(aa) != len(ba)
                    or tuple(anchor_positions.get(dep, -1) for dep in a.dependencies)
                       != tuple(after_positions.get(dep, -1) for dep in b.dependencies)):
                raise ValueError("native_layer_future_context_changes_native_access_graph")
            original_demands, future_demands = before_demands[index], nonphysical(b)
            if original_demands.keys() != future_demands.keys():
                raise ValueError("native_layer_future_context_changes_nonphysical_resource_graph")
            envelope, operator_proof = context_envelope(b)
            if (context_sensitive(a.metadata) != context_sensitive(b.metadata)
                    or a.metadata.get("event_kind") != b.metadata.get("event_kind")):
                raise ValueError("native_layer_future_context_changes_native_cost_role")
            for position, (resource, original) in enumerate(original_demands.items()):
                value = future_demands[resource]
                if not all(_finite(number) for number in (value.service_ns, value.bytes_moved, value.work_units)):
                    raise ValueError("native_layer_future_nonphysical_demand_invalid")
                if operator_proof is None and (value.service_ns, value.bytes_moved, value.work_units) != (
                        original.service_ns, original.bytes_moved, original.work_units):
                    raise ValueError("native_layer_future_nonphysical_unknown_changed_task:"
                        +repr((a.metadata.get("operator_id"), a.metadata.get("event_kind"),
                               a.metadata.get("phase"), resource,
                               (original.service_ns, original.bytes_moved, original.work_units),
                               (value.service_ns, value.bytes_moved, value.work_units))))
                lo, hi = envelope[resource]
                nonphysical_upper[index][position] = max(nonphysical_upper[index][position], hi)
                nonphysical_lower[index][position] = min(nonphysical_lower[index][position], lo)
                if hi > original.service_ns or lo < original.service_ns:
                    demand_proofs.append({"task_position": index, "task_id": a.task_id,
                        "rank": a.metadata.get("rank"), "context_tokens": context,
                        "resource_id": resource, "anchor_service_ns": original.service_ns,
                        "native_service_ns": value.service_ns, "envelope_lower_ns": lo,
                        "envelope_upper_ns": hi, **(operator_proof or {})})
            for access_index, (old, new) in enumerate(zip(aa, ba)):
                if (old.get("physical_owner", a.metadata.get("physical_owner")), old["operation"]) != (
                        new.get("physical_owner", b.metadata.get("physical_owner")), new["operation"]):
                    raise ValueError("native_layer_future_context_changes_memory_direction_or_owner")
                change = max(0., phase(new, b.metadata)-before_phases[index][access_index])
                extra += change
                per_access_extra[index][access_index] = max(per_access_extra[index][access_index], change)
        proofs.append({"context_tokens": context, "native_task_count": len(after),
            "initial_state_phase_growth_serial_upper_ns": extra})
    increase = math.fsum(max(0., upper-original.service_ns)
        for values, uppers in zip(before_demands, nonphysical_upper)
        for original, upper in zip(values.values(), uppers))
    decrease = math.fsum(max(0., original.service_ns-lower_bound)
        for values, lowers in zip(before_demands, nonphysical_lower)
        for original, lower_bound in zip(values.values(), lowers))
    return {"qualified": True, "cohort_kind": decode_cohort.kind,
        "basis": "native_Query_or_Decode_IR_known_monotone_attention_formulas_at_full_and_chunk_extrema; serial_nonphysical_delta_and_memory_phase_growth",
        "corners": proofs, "future_context_phase_growth_serial_upper_ns": math.fsum(
            value for row in per_access_extra for value in row),
        "future_nonphysical_service_growth_serial_upper_ns": increase,
        "future_nonphysical_service_decrease_serial_upper_ns": decrease,
        "identical_native_context_inputs": fixed_context,
        "nonphysical_demand_growth_terms": demand_proofs,
        "monotonicity_scope": "fixed_native_unquantized_GPU_attention_roles_and_profiles; full_context_or_each_chunk_effective_context; unknown_roles_and_measured_surfaces_rejected",
        "growth_accounting": "full_positive_per_demand_delta_added_once_after_affine_envelope; no_unproven_ratio_credit",
        "hbm_clock_domain_limit_ns": min(clock_limits, default=None),
        "physical_state_advanced": False}


def evaluate_native_layer_window(runtime, cohort, cost, stages, sample_payload, options,
                                 paired_reference=None):
    """Project a native serving window from independent layer samples.

    The current final-query identity and paid PD are real. All complete
    cohort times and all future visible commits below are forecasts; none
    is inserted into the live runtime's batches, requests or events.
    """
    from .native_window_budget import (_domain, _memory_kind, _paired_decision,
                                      single_slot_skeleton)
    from .native_window_producer import (_allocation_evidence, _native_cores_supported,
                                        _native_pd_evidence, native_context_features,
                                        summarize_native_stages)

    result = {"accepted": False, "reasons": [], "metrics": {}, "metric_bounds": {},
        "window": {}, "request_forecasts": [], "error_budget": {"conditional_only": True},
        "semantic_guards": {"physical_state_advanced": False,
            "forecast_has_physical_trace": False, "complete_fresh_request_definition_preserved": True,
            "physical_samples_are_layers_not_complete_cohorts": True},
        "predicted_visible_output_tokens": 0, "can_expand_sampling": False}
    reasons, plan, scenario = result["reasons"], runtime.plan, runtime.plan.scenario
    target = options.target_relative_error
    layers = [instance.layer for instance in scenario.model._execution_view.layer_instances]
    if (not _native_cores_supported(runtime._execution_resource_kernel.physical_runtime)
            or any(layer.is_linear_attention for layer in layers)
            or any(layer.is_moe and layer.metadata.get("moe_routing", {}).get("load_model")
                   != "deterministic_balanced_histogram" for layer in layers)):
        reasons.append("native_layer_custom_core_or_operator_continuation_not_bounded")
        return result
    if (cohort.kind != "prefill" or not cohort.items
            or any(not runtime.states[item.request_id].spec.metadata.get("warmup_seed", False)
                   for item in cohort.items)):
        reasons.append("native_layer_first_selected_query_is_not_the_authored_seed")
        return result
    ingress = runtime._external_ingress
    if (plan.scheduler.preemption_enabled or plan.scheduler.mixed_phase_batching
            or any(row.arrival_ns != plan.requests[0].arrival_ns or row.priority != 0
                   or row.deadline_ns is not None for row in plan.requests)
            or (plan.scheduler.max_num_seqs == 1
                and len({(row.prompt_tokens, row.output_tokens) for row in plan.requests}) != 1)):
        reasons.append("native_layer_window_scheduler_or_request_shape_not_bounded")
        return result
    layer = evaluate_native_layer_windows(sample_payload, target_relative_error=target,
        independent_sufficient=False)
    result["error_budget"]["native_layer_sampling"] = layer
    if not layer["accepted"]:
        reasons.extend(layer["reasons"])
        return result
    try:
        shape_proof = _query_decode_contract(runtime, cohort, stages, query_cost=cost)
        whole = cost.metadata.get("native_fast_budget_observations")
        if not whole:
            whole = summarize_native_stages(stages, scenario,
                runtime._execution_resource_kernel.physical_runtime,
                native_runtime=runtime)
        resident = [declaration for stage in stages for task in stage.execution_tasks
            for declaration in task.metadata.get("physical_allocations", ())
            if declaration.get("generation", 0) == 0 and declaration.get("allocation_arena") == "resident"]
        capacity = _allocation_evidence(runtime, (whole,), resident_declarations=resident)
        if not capacity.get("qualified"):
            raise ValueError("native_layer_future_physical_allocation_bound_unavailable")
    except ValueError as error:
        reasons.append(str(error))
        return result
    result["error_budget"].update(query_decode_native_IR=shape_proof, capacity=capacity)
    if plan.scheduler.max_num_seqs > 1:
        from .native_multislot_budget import evaluate_multislot_layer_window
        multi = evaluate_multislot_layer_window(runtime, cohort, whole, layer,
            shape_proof, options, paired_reference=paired_reference,
            seed_shape_anchors=sample_payload.get("batch_shape_anchors", ()))
        multi.setdefault("error_budget", {}).update(native_layer_sampling=layer,
            query_decode_native_IR=shape_proof, capacity=capacity)
        return multi
    skeleton = single_slot_skeleton(plan.requests, ingress["warmup_decode_cohorts"],
                                    ingress["measurement_decode_cohorts"])
    raw = layer["cohort_interval"]
    delta = shape_proof["nonphysical_service_difference_serial_guard_ns"]
    context_phase_growth = shape_proof["future_context_phase_contract"]["future_context_phase_growth_serial_upper_ns"]
    nonphysical_growth = shape_proof["future_context_phase_contract"]["future_nonphysical_service_growth_serial_upper_ns"]
    nonphysical_decrease = shape_proof["future_context_phase_contract"]["future_nonphysical_service_decrease_serial_upper_ns"]
    query_context = shape_proof["future_query_context_phase_contract"]
    query_phase = query_context["future_context_phase_growth_serial_upper_ns"]
    query_increase = query_context["future_nonphysical_service_growth_serial_upper_ns"]
    query_decrease = query_context["future_nonphysical_service_decrease_serial_upper_ns"]
    query_lo, query_hi = raw["lower"], raw["upper"]
    decode_lo, decode_hi = query_lo-delta, query_hi+delta
    if decode_lo <= 0:
        reasons.append("native_layer_query_decode_difference_exhausts_lower_bound")
        return result
    anchors = native_context_features(scenario, [cohort.items[0].context_tokens])
    floor = whole.get("fixed_path_floor_ns", 0.)
    if not _finite(floor) or floor > decode_hi:
        reasons.append("native_layer_fixed_native_path_floor_inconsistent")
        return result
    pd = _native_pd_evidence(runtime)
    if not pd["qualified"]:
        reasons.append(pd["reason"])
        return result
    timeline = runtime._execution_resource_kernel.physical_runtime.timeline
    carry = max(0., max(timeline.ready_ns.values(), default=runtime.now)-runtime.now)
    # Only work downstream of the last layer can separate visible output
    # from cohort completion. Prefix embedding/controller work is already
    # in service, and charging it as commit lag would create a false TPOT
    # rejection. A suffix's complete independent native duration plus both
    # controller phases is a conservative cleanup upper, including head.
    suffix = [row for row in sample_payload["non_layer_samples"]
              if row.get("position") == "after_last_layer"]
    if not suffix:
        reasons.append("native_layer_visible_commit_suffix_not_covered")
        return result
    cleanup = shape_proof["visible_commit_lag_upper_ns"]
    if cleanup is None:
        cleanup = math.fsum(_sample_duration(row)+(1 if _issue_order_fixed(row["native_ir_summary"]) else 2)
                           *_phase(row["native_ir_summary"]) for row in suffix)
    if cleanup >= decode_lo:
        reasons.append("native_layer_visible_commit_cleanup_may_cross_cohort_boundary")
        return result
    services, times, current = [], [], (runtime.now, runtime.now)
    for event_index, event in enumerate(skeleton["events"]):
        if event["kind"] == "prefill":
            service = (query_lo, query_hi)
            if event_index:
                service = (max(floor, service[0]-query_phase-query_decrease),
                           service[1]+query_phase+query_increase)
        else:
            coordinates = native_context_features(scenario, [event["context_tokens"]])
            ratios = [value/anchor for value, anchor in zip(coordinates, anchors)]
            service = (max(floor, floor+max(0., decode_lo-floor)*min([1., *ratios])),
                       floor+(decode_hi-floor)*max([1., *ratios]))
            service = (max(floor, service[0]-context_phase_growth-nonphysical_decrease),
                       service[1]+context_phase_growth+nonphysical_growth)
        services.append(service)
        current = (current[0]+service[0], current[1]+service[1])
        times.append(current)
    first_window = skeleton["start_event"]+1
    # Future PD imports are placed on the skeleton and charged with the
    # measured k=1 interference; no per-import worst-case span is added.
    from .native_pd_interference import interference_model, single_slot_import_timeline
    model = interference_model(sample_payload.get("pd_interference"))
    imports = single_slot_import_timeline(runtime, skeleton, pd, services, model)
    if not imports["qualified"]:
        result["error_budget"]["future_PD_measured_interference"] = {"model": model, "timeline": imports}
        reasons.append(imports["reason"])
        return result
    span_extra = imports.pop("span_extra")
    clock_origin = max(runtime.now, sample_payload["layer_composition"].get("sample_boundary_origin_ns", runtime.now))
    _close_hbm_clock_domain(result, (layer["error_budget"],), math.nextafter(
        math.fsum((clock_origin, *(row[1] for row in services), imports["total_upper_ns"], carry, cleanup)), math.inf),
        phase_proofs=(shape_proof["future_context_phase_contract"], query_context))
    window_extra = span_extra(first_window, len(services))
    duration_lo = math.fsum(row[0] for row in services[first_window:])+window_extra[0]
    duration_hi = math.fsum(row[1] for row in services[first_window:])+window_extra[1]+carry
    request_extras = []
    tpot_lo, tpot_hi, ttft_lo, ttft_hi = [], [], [], []
    fresh_ids = [spec.request_id for spec in plan.requests if not spec.metadata.get("warmup_seed", False)]
    for request in skeleton["complete_fresh"]:
        first, last = request["first_event"], request["last_event"]
        life = services[first+1:last+1]
        life_extra, ingress_extra = span_extra(first+1, last+1), span_extra(0, first+1)
        request_extras.append({"first_event": first, "life_extra_ns": _interval(*life_extra),
                               "ingress_extra_ns": _interval(*ingress_extra)})
        lo = (math.fsum(row[0] for row in life)-cleanup+life_extra[0])/(request["output_tokens"]-1)
        hi = (math.fsum(row[1] for row in life)+cleanup+life_extra[1]+carry)/(request["output_tokens"]-1)
        flo = times[first][0]-cleanup-request["arrival_ns"]+ingress_extra[0]
        fhi = times[first][1]+ingress_extra[1]+carry-request["arrival_ns"]
        if min(lo, flo) <= 0:
            reasons.append("native_layer_cleanup_or_PD_exhausts_fresh_lower_bound")
            return result
        tpot_lo.append(lo); tpot_hi.append(hi); ttft_lo.append(flo); ttft_hi.append(fhi)
        result["request_forecasts"].append({"request_id": None,
            "logical_fifo_slot_id": request["request_id"], "possible_same_shape_request_ids": fresh_ids,
            "per_request_label_order_proven": False, "forecast_only": True,
            "tpot_ns": _interval(lo, hi), "ingress_ttft_ns": _interval(flo, fhi)})
    if not tpot_lo:
        reasons.append("native_layer_window_has_no_complete_fresh_request")
        return result
    tokens = skeleton["visible_output_tokens"]
    duration = _interval(duration_lo, duration_hi)
    throughput = _interval(tokens*1e9/duration_hi, tokens*1e9/duration_lo)
    bounds = {"measurement_duration_ns": duration, "cluster_tokens_per_second": throughput,
        "mean_complete_request_tpot_ns": _interval(fmean(tpot_lo), fmean(tpot_hi)),
        "mean_complete_request_ingress_ttft_ns": _interval(fmean(ttft_lo), fmean(ttft_hi))}
    limit = target/(2+target) if paired_reference is None else target
    if any(not _within(row, limit) for row in bounds.values()):
        reasons.append("native_layer_window_conditional_single_or_independent_pair_budget_exceeded")
    pair = _paired_decision(bounds, paired_reference, runtime, target)
    if not pair["accepted"]:
        reasons.extend(pair["reasons"])
    result.update(metric_bounds=bounds, predicted_visible_output_tokens=len(skeleton["events"]),
        window={"warmup_decode_cohorts": ingress["warmup_decode_cohorts"],
            "requested_decode_cohorts": ingress["measurement_decode_cohorts"],
            "measurement_visible_output_tokens": tokens, "complete_fresh_requests": len(tpot_lo),
            "forecast_only": True, "physical_complete_cohort_samples": 0})
    result["error_budget"].update(context_feature_anchors=anchors,
        native_fixed_path_floor_ns=floor,
        future_context_phase_growth_per_event_upper_ns=context_phase_growth,
        future_nonphysical_service_growth_per_event_upper_ns=nonphysical_growth,
        future_nonphysical_service_decrease_per_event_upper_ns=nonphysical_decrease,
        future_query_context_phase_growth_per_event_upper_ns=query_phase,
        future_query_nonphysical_service_growth_per_event_upper_ns=query_increase,
        future_query_nonphysical_service_decrease_per_event_upper_ns=query_decrease,
        future_PD_outstanding_request_ids=[row["request_id"] for row in imports["imports"]],
        future_PD_measured_interference={"model": model, "timeline": imports,
            "window_extra_ns": _interval(*window_extra), "request_extras": request_extras},
        native_PD_geometry_and_phase_budget=pd,
        cutoff_carry_in_upper_ns=carry, visible_commit_cleanup_upper_ns=cleanup,
        per_point_target=target, effective_independent_target=limit,
        paired_budget_decision=pair, conditional_only=True)
    if not reasons:
        world = scenario.placement.parallel.tp_degree*scenario.placement.parallel.pp_degree*scenario.placement.parallel.ep_degree
        result.update(accepted=True, metrics={"cluster_tokens_per_second": throughput["estimate"],
            "tokens_per_second_per_gpu": throughput["estimate"]/world,
            "mean_complete_request_tpot_ns": bounds["mean_complete_request_tpot_ns"]["estimate"],
            "mean_complete_request_ingress_ttft_ns": bounds["mean_complete_request_ingress_ttft_ns"]["estimate"],
            "complete_fresh_requests": len(tpot_lo), "measurement_visible_output_tokens": tokens,
            "measurement_duration_ns": duration["estimate"]})
        result["error_budget"]["paired_reference"] = {"schema": "native-window-paired-reference/v1",
            "domain": _domain(runtime), "memory_kind": _memory_kind(runtime),
            "source_id": scenario.name, "target_relative_error": target,
            "metric_bounds": bounds, "accepted": True}
    return result


__all__ = ["evaluate_native_layer_windows", "evaluate_native_layer_window"]
