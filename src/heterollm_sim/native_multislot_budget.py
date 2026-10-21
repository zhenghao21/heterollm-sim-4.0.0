"""Conditional batch-window intervals over the original native lifecycle.

The original selector supplies batch identities. This module only evaluates
timing intervals; it cannot execute, resume or advance a physical runtime.
"""
from __future__ import annotations

from statistics import fmean
import math


def _layer_pd_exposure(runtime, skeleton, pd):
    """Bound paid PD separately at query admission and inside a fresh wave.

    During a proven full wave every same-shape request retires together.
    No waiting request can enter a slot until that last Decode boundary.
    The original importer's pending+ready cap therefore allows at most its
    explicit inflight limit of foreign imports during that wave. Imports
    needed to construct the wave are already complete before its query.
    Unknown/partial waves keep the original whole-future serial bound.
    """
    ingress, plan = runtime._external_ingress, runtime.plan
    limit = ingress.get("max_inflight_requests", 2)
    rows, events = tuple(pd["future_requests"]), skeleton["events"]
    global_upper = pd["future_PD_serial_upper_ns"]
    result = {"qualified": False, "measurement_upper_ns": global_upper,
        "per_request": {}, "fallback": "whole_future_native_PD_serial_upper",
        "future_request_count": len(rows), "inflight_limit": limit}
    if type(limit) is not int or limit < 1:
        return result
    specs = {row.request_id: row for row in plan.requests}
    fresh = {key for key, row in specs.items() if not row.metadata.get("warmup_seed", False)}
    start, batch = skeleton["start_event"]+1, plan.scheduler.max_num_seqs
    if (any(set(event["request_ids"]) & fresh for event in events[:start])
            or not rows or any(row["request_id"] not in fresh for row in rows)):
        return result
    maximum = max(row["native_import_upper_ns"] for row in rows)
    index, wave, request_guards = start, 0, {}
    while index < len(events):
        query = events[index]
        ids = set(query["request_ids"])
        if (query["kind"] != "prefill" or len(ids) != batch or not ids <= fresh
                or len({specs[key].output_tokens for key in ids}) != 1):
            return result
        output = specs[next(iter(ids))].output_tokens
        last = index+output-1
        if last >= len(events):
            return result
        if any(event["kind"] != "decode" or set(event["request_ids"]) != ids
               for event in events[index+1:last+1]):
            return result
        if any(commit["committed"] != output for commit in events[last]["visible_commits"]):
            return result
        wave += 1
        prefix_count = min(len(rows), wave*batch+limit)
        foreign_count = min(limit, max(0, len(fresh)-wave*batch))
        for key in ids:
            request_guards[key] = {"query_admission_and_prefix_interference_upper_ns": prefix_count*maximum,
                "within_request_PD_interference_upper_ns": foreign_count*maximum,
                "prefix_import_count_upper": prefix_count,
                "within_wave_import_count_upper": foreign_count, "wave_index": wave}
        index = last+1
    if not wave:
        return result
    return {**result, "qualified": True, "fallback": None,
        "basis": "native_full_slot_gate_equal_fresh_output_wave_and_pending_plus_ready_import_cap",
        "full_fresh_waves": wave, "per_request": request_guards,
        "measurement_import_count_upper": min(len(rows), wave*batch+limit),
        "measurement_upper_ns": min(len(rows), wave*batch+limit)*maximum,
        "per_import_upper_ns": maximum,
        "PD_cost_not_removed": True, "per_request_PD_label_order_proven": False}


def evaluate_native_multislot(runtime, records, options, paired_reference=None, *, budget_evidence=None):
    from .native_window_budget import (_domain, _finite, _interval, _memory_kind,
        _midpoint, _paired_decision, _within, collect_native_budget_evidence)
    from .native_window_lifecycle import native_lifecycle_skeleton
    from .native_window_producer import native_context_features

    result = {"accepted": False, "reasons": [], "metrics": {}, "metric_bounds": {},
        "window": {}, "request_forecasts": [], "error_budget": {}, "semantic_guards": {
            "physical_state_advanced": False, "forecast_has_physical_trace": False,
            "complete_fresh_request_definition_preserved": True},
        "predicted_visible_output_tokens": 0, "can_expand_sampling": False}
    reasons, plan = result["reasons"], runtime.plan
    target, batch = options.target_relative_error, plan.scheduler.max_num_seqs
    from .serving import _OnlineRuntime
    if type(runtime) is not _OnlineRuntime:
        reasons.append("native_multislot_requires_live_native_runtime")
        return result
    try:
        skeleton = native_lifecycle_skeleton(runtime, sample_records=records)
    except ValueError as error:
        reasons.append(str(error))
        return result
    washout = skeleton["seed_washout"]
    result["error_budget"]["native_lifecycle"] = {key: skeleton[key] for key in (
        "lifecycle_contract", "future_admission_order_proven",
        "future_admission_order_bound_required", "seed_washout")}
    if not washout["qualified"]:
        reasons.append("native_multislot_seed_washout_not_proven")
        return result
    ingress = runtime._external_ingress
    seed_end = runtime._completed_decode_steps+washout["seed_decode_cohort_upper"]
    if seed_end > ingress["warmup_decode_cohorts"]:
        reasons.append("native_multislot_seed_transient_crosses_measurement_start")
        return result
    events = skeleton["events"]
    start = skeleton["start_event"]
    if any(len(row["request_ids"]) != batch for row in events[start+1:]):
        reasons.append("native_multislot_measurement_full_wave_shapes_not_proven")
        return result
    decode = [row for row in records if row["kind"] == "decode" and len(row["request_ids"]) == batch]
    count = options.calibration_cohorts+options.validation_cohorts
    if len(decode) < options.warmup_cohorts+count:
        reasons.append("native_multislot_disjoint_full_batch_samples_insufficient")
        result["can_expand_sampling"] = True
        return result
    fit = decode[-count:]
    training = options.calibration_cohorts
    if any(not _finite(row["execution_ns"], positive=True) for row in fit):
        reasons.append("native_multislot_sample_duration_invalid")
        return result
    contexts = [sum(item["context_tokens"] for item in row["context_tokens"]) for row in fit]
    base, anchor = fmean(row["execution_ns"] for row in fit[:training]), fmean(contexts[:training])
    if not _finite(anchor, positive=True):
        reasons.append("native_multislot_sample_context_invalid")
        return result
    features = [native_context_features(plan.scenario, [item["context_tokens"]
                for item in row["context_tokens"]]) for row in fit]
    feature_anchors = tuple(fmean(row[index] for row in features[:training])
                            for index in range(len(features[0])))
    if not feature_anchors or any(not _finite(value, positive=True) for value in feature_anchors):
        reasons.append("native_multislot_context_feature_anchor_invalid")
        return result
    def ratios(values, anchors=feature_anchors):
        return [value/anchor for value, anchor in zip(values, anchors)]
    predicted = sum(_midpoint(base*min(1., *ratios(row)), base*max(1., *ratios(row)))
                    for row in features[training:])
    measured = sum(row["execution_ns"] for row in fit[training:])
    half = training//2
    drift = abs(fmean(row["execution_ns"] for row in fit[:half])-
        fmean(row["execution_ns"] for row in fit[half:training]))/base
    local = max(abs(predicted-measured)/measured, drift)
    result["error_budget"]["calibration"] = {"training_cohorts": training,
        "validation_cohorts": options.validation_cohorts, "holdout_disjoint": True,
        "full_batch_shape": batch, "holdout_relative_error": local,
        "training_base_ns": base, "anchor_context_sum": anchor,
        "native_layer_context_feature_anchors": feature_anchors}
    evidence = budget_evidence or collect_native_budget_evidence(runtime, records, options)
    result["error_budget"]["native_evidence"] = evidence
    capacity, steady = evidence.get("capacity", {}), evidence.get("steady_contract", {})
    if not capacity.get("qualified") or not capacity.get("physical_workspace_and_allocator_fragmentation_covered"):
        reasons.append("native_multislot_future_capacity_or_fragmentation_not_bounded")
    if not steady.get("qualified"):
        reasons.append("native_multislot_future_service_envelope_unavailable")
    lower_guard = upper_guard = 0.
    for name in ("pd_service_and_interference", "cache_and_page_phase", "refresh_and_row_state",
                 "allocation_reuse", "cutoff_carry_in", "prefill_query"):
        row = evidence.get("future_guards", {}).get(name, {})
        if (row.get("qualified") is not True or not row.get("basis")
                or not _finite(row.get("lower_guard_ns")) or not _finite(row.get("upper_guard_ns"))):
            reasons.append("native_multislot_future_"+name+"_bound_unavailable")
        else:
            lower_guard += row["lower_guard_ns"]
            upper_guard += row["upper_guard_ns"]
    lag = evidence.get("visible_commit_lag_ns", {})
    if (not lag.get("qualified") or not _finite(lag.get("lower_ns"))
            or not _finite(lag.get("upper_ns")) or lag["lower_ns"] > lag["upper_ns"]):
        reasons.append("native_multislot_future_visible_commit_cleanup_unavailable")
    floor = evidence.get("constant_path_floor_ns", 0.)
    # The B-sized fixed path can include B-sized compute. It is not a floor
    # for an unobserved smaller batch. Only an explicitly batch-invariant
    # native weight-path floor can bound the seed transient below.
    transient_floor = evidence.get("batch_invariant_path_floor_ns", 0.)
    if transient_floor and (evidence.get("batch_invariant_path_floor_proof") !=
            "native_preloaded_generation_zero_tensor_read_bytes_over_directional_bandwidth_only"
            or evidence.get("batch_invariant_weight_floor_applicable") is not True):
        reasons.append("native_multislot_unsampled_shape_weight_floor_not_proven")
    if (not _finite(floor) or not _finite(transient_floor) or floor > base
            or transient_floor > floor or local >= 1):
        reasons.append("native_multislot_constant_work_or_calibration_invalid")
    if reasons:
        return result
    prefill = [row for row in records if row["kind"] == "prefill" and len(row["request_ids"]) == batch]
    if not prefill:
        reasons.append("native_multislot_full_query_sample_unavailable")
        return result
    query = fmean(row["execution_ns"] for row in prefill)
    query_features = [native_context_features(plan.scenario, [item["context_tokens"]
                      for item in row["context_tokens"]]) for row in prefill]
    query_anchors = tuple(fmean(row[index] for row in query_features)
                          for index in range(len(query_features[0])))
    phase = evidence.get("event_guards", {})
    for kind in ("decode", "prefill"):
        row = phase.get(kind, {})
        if not _finite(row.get("lower_ns")) or not _finite(row.get("upper_ns")):
            reasons.append("native_multislot_event_phase_unavailable:"+kind)
    carry = phase.get("cutoff_carry_in_upper_ns", 0.)
    if not _finite(carry):
        reasons.append("native_multislot_cutoff_carry_in_invalid")
    if reasons:
        return result
    now_lo = now_hi = float(runtime._runtime_origin_ns)
    times, services, commit_lags = [], [], []
    for index, event in enumerate(events):
        lags = {}
        if index < len(records):
            record = records[index]
            service = (record["end_ns"]-now_lo, record["end_ns"]-now_hi)
            if service[0] < 0:
                reasons.append("native_multislot_real_prefix_order_invalid")
                return result
            for commit in record["visible_commits"]:
                stamp = commit.get("timestamp_ns")
                if not _finite(stamp) or not now_lo < stamp <= record["end_ns"]:
                    reasons.append("native_multislot_actual_commit_boundary_invalid")
                    return result
                lags[commit["request_id"]] = (record["end_ns"]-stamp,)*2
            now_lo = now_hi = record["end_ns"]
        else:
            coordinates = native_context_features(plan.scenario,
                [item["context_tokens"] for item in event["items"]])
            if event["kind"] == "prefill":
                change = ratios(coordinates, query_anchors)
                lo, hi = query*min(1., *change), query*max(1., *change)
            else:
                change = ratios(coordinates)
                if len(event["request_ids"]) == batch:
                    lo = floor+(base-floor)*min(1., *change)
                else:
                    lo = transient_floor
                hi = floor+(base-floor)*max(1., *change)
            guard = phase[event["kind"]]
            service = (max(0., lo/(1+local)-guard["lower_ns"]), hi/(1-local)+guard["upper_ns"]+
                (carry if index == len(records) else 0.))
            if lag["upper_ns"] >= service[0] and len(event["request_ids"]) == batch:
                reasons.append("native_multislot_visible_commit_can_cross_full_cohort")
                return result
            for commit in event["visible_commits"]:
                lags[commit["request_id"]] = (lag["lower_ns"], lag["upper_ns"])
            now_lo += service[0]
            now_hi += service[1]
        services.append(service)
        times.append((now_lo, now_hi))
        commit_lags.append(lags)
    dlo = math.fsum(row[0] for row in services[start+1:])-lower_guard
    dhi = math.fsum(row[1] for row in services[start+1:])+upper_guard
    if dlo <= 0:
        reasons.append("native_multislot_future_guard_exhausts_measurement")
        return result
    duration = _midpoint(dlo, dhi)
    tlo, thi, flo, fhi, requests = [], [], [], [], []
    for request in skeleton["complete_fresh"]:
        first_index, last_index, request_id = request["first_event"], request["last_event"], request["request_id"]
        first, last = times[first_index], times[last_index]
        flag, llag = commit_lags[first_index][request_id], commit_lags[last_index][request_id]
        lifetime = services[first_index+1:last_index+1]
        denominator = request["output_tokens"]-1
        lo = (math.fsum(row[0] for row in lifetime)-llag[1]+flag[0]-lower_guard)/denominator
        hi = (math.fsum(row[1] for row in lifetime)-llag[0]+flag[1]+upper_guard)/denominator
        fl, fh = first[0]-flag[1]-request["arrival_ns"]-lower_guard, first[1]-flag[0]-request["arrival_ns"]+upper_guard
        if min(lo, fl) <= 0:
            reasons.append("native_multislot_transient_or_PD_exhausts_fresh_lower_bound")
            return result
        tlo.append(lo); thi.append(hi); flo.append(fl); fhi.append(fh)
        requests.append({**request, "forecast_only": True,
            "request_id_semantics": "canonical_equal_shape_forecast_slot_not_future_PD_identity",
            "tpot_ns": _interval(_midpoint(lo, hi), lo, hi),
            "ingress_ttft_ns": _interval(_midpoint(fl, fh), fl, fh)})
    if not requests:
        reasons.append("native_multislot_no_complete_fresh_request")
        return result
    lo, hi, fl, fh = fmean(tlo), fmean(thi), fmean(flo), fmean(fhi)
    tokens = skeleton["visible_output_tokens"]
    bounds = {"mean_complete_request_tpot_ns": _interval(_midpoint(lo, hi), lo, hi),
        "mean_complete_request_ingress_ttft_ns": _interval(_midpoint(fl, fh), fl, fh),
        "cluster_tokens_per_second": _interval(tokens*1e9/duration, tokens*1e9/dhi, tokens*1e9/dlo),
        "measurement_duration_ns": _interval(duration, dlo, dhi)}
    limit = target/(2+target) if paired_reference is None else target
    if any(not _within(row, limit) for row in bounds.values()):
        reasons.append("native_multislot_single_point_or_independent_pair_budget_exceeded")
    paired = _paired_decision(bounds, paired_reference, runtime, target)
    reasons.extend(paired.get("reasons", ()))
    result.update(metric_bounds=bounds, request_forecasts=requests,
        predicted_visible_output_tokens=max(0, sum(row["visible_output_tokens"] for row in events)-
            sum(row["visible_output_tokens"] for row in records)),
        window={"warmup_decode_cohorts": ingress["warmup_decode_cohorts"],
            "requested_decode_cohorts": ingress["measurement_decode_cohorts"],
            "measurement_visible_output_tokens": tokens, "complete_fresh_requests": len(requests),
            "forecast_only": True}, error_budget={**result["error_budget"], "conditional_only": True,
            "lower_guard_ns": lower_guard, "upper_guard_ns": upper_guard,
            "per_point_target": target, "effective_independent_target": limit,
            "paired_budget_decision": paired})
    if not reasons:
        world = plan.scenario.placement.parallel.tp_degree*plan.scenario.placement.parallel.pp_degree*plan.scenario.placement.parallel.ep_degree
        result.update(accepted=True, metrics={"cluster_tokens_per_second": bounds["cluster_tokens_per_second"]["estimate"],
            "tokens_per_second_per_gpu": bounds["cluster_tokens_per_second"]["estimate"]/world,
            "mean_complete_request_tpot_ns": bounds["mean_complete_request_tpot_ns"]["estimate"],
            "mean_complete_request_ingress_ttft_ns": bounds["mean_complete_request_ingress_ttft_ns"]["estimate"],
            "complete_fresh_requests": len(requests), "measurement_visible_output_tokens": tokens,
            "measurement_duration_ns": duration})
        result["error_budget"]["paired_reference"] = {"schema": "native-window-paired-reference/v1",
            "domain": _domain(runtime), "memory_kind": _memory_kind(runtime),
            "source_id": plan.scenario.name, "target_relative_error": target,
            "metric_bounds": bounds, "accepted": True}
    return result


def evaluate_multislot_layer_window(runtime, cohort, whole, layer, shape_proof, options,
                                    paired_reference=None, *, seed_shape_anchors=()):
    """Compose value intervals without claiming a physically completed batch.

    The actual native pre-query admission state supplies the lifecycle.
    Only the private logical selector commits a *forecast* first query;
    paid PD, unsampled seed batch shapes and future request labels remain
    explicit independent bounds. Layer samples never become native batches.
    """
    from .native_layer_budget import _interval, _close_hbm_clock_domain
    from .native_window_budget import _domain, _finite, _memory_kind, _paired_decision, _within
    from .native_window_lifecycle import native_layer_lifecycle_skeleton
    from .native_window_producer import _native_pd_evidence, native_context_features

    result = {"accepted": False, "reasons": [], "metrics": {}, "metric_bounds": {},
        "window": {}, "request_forecasts": [], "error_budget": {"conditional_only": True,
            "native_layer_sampling": layer}, "semantic_guards": {
            "physical_state_advanced": False, "forecast_has_physical_trace": False,
            "complete_fresh_request_definition_preserved": True,
            "physical_samples_are_layers_not_complete_cohorts": True},
        "predicted_visible_output_tokens": 0, "can_expand_sampling": False}
    reasons, plan, scenario = result["reasons"], runtime.plan, runtime.plan.scenario
    target, batch, ingress = options.target_relative_error, plan.scheduler.max_num_seqs, runtime._external_ingress
    try:
        skeleton = native_layer_lifecycle_skeleton(runtime, cohort)
    except ValueError as error:
        reasons.append(str(error))
        return result
    washout = skeleton["seed_washout"]
    result["error_budget"].update(native_lifecycle={key: skeleton[key] for key in (
        "lifecycle_source", "lifecycle_contract", "seed_washout",
        "future_admission_order_bound_required", "physical_complete_cohort_samples")},
        query_decode_native_IR=shape_proof)
    if not washout["qualified"]:
        reasons.append("native_layer_multislot_seed_washout_not_proven")
        return result
    if washout["seed_decode_cohort_upper"] > ingress["warmup_decode_cohorts"]:
        reasons.append("native_layer_multislot_seed_transient_crosses_measurement")
        return result
    events, first_window = skeleton["events"], skeleton["start_event"]+1
    if any(len(row["request_ids"]) != batch for row in events[first_window:]):
        reasons.append("native_layer_multislot_measurement_full_wave_not_proven")
        return result
    raw, delta = layer["cohort_interval"], shape_proof["nonphysical_service_difference_serial_guard_ns"]
    decode_contract = shape_proof["future_context_phase_contract"]
    query_contract = shape_proof["future_query_context_phase_contract"]
    context_phase_growth = decode_contract[
        "future_context_phase_growth_serial_upper_ns"]
    nonphysical_growth = decode_contract[
        "future_nonphysical_service_growth_serial_upper_ns"]
    nonphysical_decrease = decode_contract[
        "future_nonphysical_service_decrease_serial_upper_ns"]
    query_phase_growth = query_contract["future_context_phase_growth_serial_upper_ns"]
    query_nonphysical_growth = query_contract["future_nonphysical_service_growth_serial_upper_ns"]
    query_nonphysical_decrease = query_contract["future_nonphysical_service_decrease_serial_upper_ns"]
    if (decode_contract.get("qualified") is not True
            or query_contract.get("qualified") is not True
            or any(not _finite(value) for value in (
                context_phase_growth, nonphysical_growth, nonphysical_decrease,
                query_phase_growth, query_nonphysical_growth, query_nonphysical_decrease))):
        reasons.append("native_layer_multislot_future_context_phase_growth_invalid")
        return result
    qlo, qhi = raw["lower"], raw["upper"]
    dlo, dhi = qlo-delta, qhi+delta
    if dlo <= 0:
        reasons.append("native_layer_multislot_query_decode_difference_exhausts_lower")
        return result
    anchors = native_context_features(scenario, [item.context_tokens for item in cohort.items])
    if not anchors or any(not _finite(value, positive=True) for value in anchors):
        reasons.append("native_layer_multislot_context_feature_anchor_invalid")
        return result
    floor = whole.get("fixed_path_floor_ns", 0.)
    moe = any(instance.layer.is_moe for instance in scenario.model._execution_view.layer_instances)
    weight_floor = 0. if moe else whole.get("batch_invariant_weight_read_floor_ns", 0.)
    transient = weight_floor
    invariant = shape_proof.get("batch_invariant_native_path_floor_ns", 0.)
    boundaries = [shape_proof.get(key, {}) for key in ("query_native_boundary", "decode_native_boundary")]
    if (not _finite(invariant) or invariant and any(row.get("batch_invariant_native_path_floor_basis") !=
            "native_dense_fixed_operator_graph_zero_work_launch_and_preloaded_weight_read_DAG"
            for row in boundaries)):
        reasons.append("native_layer_multislot_batch_invariant_native_path_floor_invalid")
        return result
    transient, floor = max(transient, invariant), max(floor, invariant)
    if (not _finite(floor) or floor > dhi or not _finite(transient) or transient > floor
            or weight_floor and whole.get("batch_invariant_weight_floor_basis") !=
            "native_preloaded_generation_zero_tensor_read_bytes_over_directional_bandwidth_only"):
        reasons.append("native_layer_multislot_unsampled_shape_fixed_work_floor_invalid")
        return result
    pd = _native_pd_evidence(runtime)
    if not pd["qualified"]:
        reasons.append(pd["reason"])
        return result
    timeline = runtime._execution_resource_kernel.physical_runtime.timeline
    carry = max(0., max(timeline.ready_ns.values(), default=runtime.now)-runtime.now)
    exposure = _layer_pd_exposure(runtime, skeleton, pd)
    guard = exposure["measurement_upper_ns"]+carry
    cleanup = layer["error_budget"]["embedding_head_host_sample_service_ns"]+layer[
        "error_budget"]["embedding_head_host_phase_guard_ns"]
    terminal_lag = shape_proof.get("visible_commit_lag_upper_ns")
    if (terminal_lag == 0. and all(row.get("visible_commit_boundary_proof", {}).get("qualified") is True
                                 for row in boundaries)):
        cleanup = 0.
    if cleanup >= dlo:
        reasons.append("native_layer_multislot_cleanup_can_cross_full_batch")
        return result
    anchor_lower, anchor_features, anchor_request = None, None, None
    anchor_bound = None
    if seed_shape_anchors:
        from .native_layer_budget import evaluate_native_layer_windows
        if len(seed_shape_anchors) != 1 or seed_shape_anchors[0].get("batch_size") != 1:
            reasons.append("native_layer_multislot_seed_shape_anchor_contract_invalid")
            return result
        anchor = seed_shape_anchors[0]
        anchor_proof = anchor.get("query_decode_native_IR", {})
        anchor_bound = evaluate_native_layer_windows(anchor["sample_payload"], target_relative_error=target,
                                                     independent_sufficient=False)
        result["error_budget"]["native_seed_shape_anchor"] = {
            "batch_size": 1, "request_id": anchor.get("request_id"),
            "context_tokens": anchor.get("context_tokens"),
            "layer_budget": anchor_bound, "query_decode_native_IR": anchor_proof,
            "native_monotone_work_contract": anchor.get("native_monotone_work_contract"),
            "physical_complete_cohort_count": 0, "physical_visible_output_token_count": 0}
        if (not anchor_bound["accepted"] or anchor_proof.get("qualified") is not True
                or anchor.get("native_monotone_work_contract", {}).get("qualified") is not True):
            reasons.append("native_layer_multislot_seed_shape_anchor_not_qualified")
            reasons.extend(anchor_bound["reasons"])
            return result
        phase_growth = anchor_proof["future_context_phase_contract"][
            "future_context_phase_growth_serial_upper_ns"]
        anchor_lower = (anchor_bound["cohort_interval"]["lower"]
                        -anchor_proof["nonphysical_service_difference_serial_guard_ns"]-phase_growth
                        -anchor_proof["future_context_phase_contract"][
                            "future_nonphysical_service_decrease_serial_upper_ns"])
        anchor_features = native_context_features(scenario, [anchor["context_tokens"]])
        anchor_request = anchor["request_id"]
        if (not _finite(anchor_lower, positive=True) or not anchor_features
                or any(not _finite(value, positive=True) for value in anchor_features)):
            reasons.append("native_layer_multislot_seed_shape_anchor_lower_invalid")
            return result
        if anchor_lower > dhi:
            reasons.append("native_layer_multislot_seed_shape_anchor_service_exceeds_full_batch_upper")
            return result
    services, times, now = [], [], (runtime.now, runtime.now)
    for event_index, event in enumerate(events):
        features = native_context_features(scenario, [item["context_tokens"] for item in event["items"]])
        ratios = [value/anchor for value, anchor in zip(features, anchors)]
        if event["kind"] == "prefill":
            service = (qlo*min(1., *ratios), qhi*max(1., *ratios))
        elif len(event["request_ids"]) == batch:
            service = (floor+max(0., dlo-floor)*min(1., *ratios),
                       floor+(dhi-floor)*max(1., *ratios))
        else:
            # No complete smaller-batch physical sample exists. Only the
            # proven batch-invariant dense weight path survives below;
            # the full-B nonnegative-work envelope remains above. MoE
            # active expert subgraphs have no such dense floor and use 0.
            lo = transient
            if anchor_lower is not None:
                if anchor_request not in event["request_ids"]:
                    reasons.append("native_layer_multislot_seed_shape_anchor_is_not_in_remaining_seed_batch")
                    return result
                anchor_ratios = [value/feature for value, feature in zip(features, anchor_features)]
                lo = max(lo, anchor_lower*min(1., *anchor_ratios))
            service = (lo, floor+(dhi-floor)*max(1., *ratios))
        # Fresh final-query batches reset the aged seed contexts and can
        # cross the same native reduction/tile/cache discontinuities as
        # Decode. Their bound comes from their own real Query IR corners,
        # not the Query/Decode difference at the sample anchor alone.
        if event_index:
            phase_growth, growth, decrease = (
                (query_phase_growth, query_nonphysical_growth, query_nonphysical_decrease)
                if event["kind"] == "prefill" else
                (context_phase_growth, nonphysical_growth, nonphysical_decrease))
            invariant_floor = floor if len(event["request_ids"]) == batch else transient
            service = (max(invariant_floor, service[0]-phase_growth-decrease),
                       service[1]+phase_growth+growth)
        services.append(service)
        now = (now[0]+service[0], now[1]+service[1])
        times.append(now)
    clock_budgets = (layer["error_budget"], *((anchor_bound["error_budget"],) if anchor_bound is not None else ()))
    clock_origin = max(runtime.now, *(row.get("composition", {}).get("sample_boundary_origin_ns", runtime.now)
                                     for row in clock_budgets))
    _close_hbm_clock_domain(result, clock_budgets,
        math.nextafter(math.fsum((clock_origin, *(row[1] for row in services),
            pd["future_PD_serial_upper_ns"], carry, cleanup)), math.inf),
        phase_proofs=(decode_contract, query_contract,
            *((anchor_proof["future_context_phase_contract"],) if anchor_bound is not None else ())))
    duration_lo = math.fsum(row[0] for row in services[first_window:])
    duration_hi = math.fsum(row[1] for row in services[first_window:])+guard
    if duration_lo <= 0:
        reasons.append("native_layer_multislot_measurement_lower_unavailable")
        return result
    fresh_ids = [state.spec.request_id for state in runtime._state_values
                 if not state.spec.metadata.get("warmup_seed", False)]
    tlo, thi, flo, fhi = [], [], [], []
    for request in skeleton["complete_fresh"]:
        first, last = request["first_event"], request["last_event"]
        lifetime = services[first+1:last+1]
        denominator = request["output_tokens"]-1
        request_guard = exposure["per_request"].get(request["request_id"], {})
        lifetime_guard = request_guard.get("within_request_PD_interference_upper_ns",
                                          pd["future_PD_serial_upper_ns"])+carry
        first_guard = request_guard.get("query_admission_and_prefix_interference_upper_ns",
                                       pd["future_PD_serial_upper_ns"])+carry
        lo, hi = ((math.fsum(row[0] for row in lifetime)-cleanup)/denominator,
                  (math.fsum(row[1] for row in lifetime)+cleanup+lifetime_guard)/denominator)
        fl, fh = times[first][0]-cleanup-request["arrival_ns"], times[first][1]+first_guard-request["arrival_ns"]
        if min(lo, fl) <= 0:
            reasons.append("native_layer_multislot_transient_or_cleanup_exhausts_request_lower")
            return result
        tlo.append(lo); thi.append(hi); flo.append(fl); fhi.append(fh)
        result["request_forecasts"].append({"request_id": None,
            "logical_fifo_slot_id": request["request_id"], "possible_same_shape_request_ids": fresh_ids,
            "per_request_label_order_proven": False, "forecast_only": True,
            "tpot_ns": _interval(lo, hi), "ingress_ttft_ns": _interval(fl, fh)})
    if not tlo:
        reasons.append("native_layer_multislot_no_complete_fresh_request")
        return result
    tokens = skeleton["visible_output_tokens"]
    duration = _interval(duration_lo, duration_hi)
    throughput = _interval(tokens*1e9/duration_hi, tokens*1e9/duration_lo)
    bounds = {"measurement_duration_ns": duration, "cluster_tokens_per_second": throughput,
        "mean_complete_request_tpot_ns": _interval(fmean(tlo), fmean(thi)),
        "mean_complete_request_ingress_ttft_ns": _interval(fmean(flo), fmean(fhi))}
    limit = target/(2+target) if paired_reference is None else target
    if any(not _within(row, limit) for row in bounds.values()):
        reasons.append("native_layer_multislot_conditional_single_or_independent_pair_budget_exceeded")
    pair = _paired_decision(bounds, paired_reference, runtime, target)
    reasons.extend(pair.get("reasons", ()))
    result.update(metric_bounds=bounds,
        predicted_visible_output_tokens=sum(row["visible_output_tokens"] for row in events),
        window={"warmup_decode_cohorts": ingress["warmup_decode_cohorts"],
            "requested_decode_cohorts": ingress["measurement_decode_cohorts"],
            "measurement_visible_output_tokens": tokens, "complete_fresh_requests": len(tlo),
            "forecast_only": True, "physical_complete_cohort_samples": 0})
    result["error_budget"].update(context_feature_anchors=anchors,
        native_full_batch_fixed_path_floor_ns=floor,
        unsampled_seed_shape_weight_read_floor_ns=weight_floor,
        unsampled_seed_shape_lower_floor_ns=transient,
        unsampled_seed_native_invariant_floor_ns=invariant,
        future_context_phase_growth_per_event_upper_ns=context_phase_growth,
        future_nonphysical_service_growth_per_event_upper_ns=nonphysical_growth,
        future_nonphysical_service_decrease_per_event_upper_ns=nonphysical_decrease,
        future_query_context_phase_growth_per_event_upper_ns=query_phase_growth,
        future_query_nonphysical_service_growth_per_event_upper_ns=query_nonphysical_growth,
        future_query_nonphysical_service_decrease_per_event_upper_ns=query_nonphysical_decrease,
        seed_transient_charged_to_ingress_ttft=True,
        native_PD_geometry_and_phase_budget=pd,
        native_wave_PD_exposure=exposure,
        future_PD_serial_upper_ns=pd["future_PD_serial_upper_ns"],
        cutoff_carry_in_upper_ns=carry, visible_commit_cleanup_upper_ns=cleanup,
        per_point_target=target, effective_independent_target=limit, paired_budget_decision=pair)
    if not reasons:
        world = scenario.placement.parallel.tp_degree*scenario.placement.parallel.pp_degree*scenario.placement.parallel.ep_degree
        result.update(accepted=True, metrics={"cluster_tokens_per_second": throughput["estimate"],
            "tokens_per_second_per_gpu": throughput["estimate"]/world,
            "mean_complete_request_tpot_ns": bounds["mean_complete_request_tpot_ns"]["estimate"],
            "mean_complete_request_ingress_ttft_ns": bounds["mean_complete_request_ingress_ttft_ns"]["estimate"],
            "complete_fresh_requests": len(tlo), "measurement_visible_output_tokens": tokens,
            "measurement_duration_ns": duration["estimate"]})
        result["error_budget"]["paired_reference"] = {"schema": "native-window-paired-reference/v1",
            "domain": _domain(runtime), "memory_kind": _memory_kind(runtime),
            "source_id": scenario.name, "target_relative_error": target,
            "metric_bounds": bounds, "accepted": True}
    return result


__all__ = ["evaluate_native_multislot", "evaluate_multislot_layer_window"]
