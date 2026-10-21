"""Read-only conditional budgets for a native continuous-serving forecast.

The physical runtime remains at its real cutoff. Logical Token positions
below describe a forecast, never completed native requests or Core events.
"""
from __future__ import annotations

from fractions import Fraction
import math
from statistics import fmean
from typing import Mapping

from .fast_budget import timing_interval


def _finite(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _interval(estimate, lower, upper):
    row = timing_interval(estimate, lower, upper)
    return {key.removesuffix("_ns"): value for key, value in row.items()}


def _within(row, target):
    e, lo, hi = (Fraction(row[key]) for key in ("estimate", "lower", "upper"))
    limit = Fraction(target)
    return e <= (1 + limit) * lo and e >= (1 - limit) * hi


def _midpoint(lower, upper):
    return lower + ((upper-lower)/upper)*(lower/(1+lower/upper))


def _domain(runtime):
    from .serde import to_primitive

    plan, scenario = runtime.plan, runtime.plan.scenario
    parallel = scenario.placement.parallel
    return {"model_name": scenario.model.name, "batch_size": plan.scheduler.max_num_seqs,
        "model_semantics": to_primitive(scenario.model),
        "tp": parallel.tp_degree, "pp": parallel.pp_degree, "ep": parallel.ep_degree,
        "kv_dtype": plan.kv_policy.dtype,
        "scheduler_semantics": to_primitive(plan.scheduler),
        "requests": [[row.prompt_tokens, row.output_tokens, row.arrival_ns,
                      bool(row.metadata.get("warmup_seed", False))] for row in plan.requests],
        "warmup_decode_cohorts": runtime._external_ingress["warmup_decode_cohorts"],
        "measurement_decode_cohorts": runtime._external_ingress["measurement_decode_cohorts"],
        "metrics_semantics": "complete_fresh_equal_weight_tpot_and_all_visible_window_tokens/v1"}


def _memory_kind(runtime):
    scenario = runtime.plan.scenario
    if hasattr(scenario, "placement") and hasattr(scenario.placement, "metadata"):
        stores = set(scenario.placement.metadata.get("rank_kv_components", {}).values())
        stores.update(scenario.placement.metadata.get("rank_weight_components", {}).values())
        if stores:
            return "hbf" if any(scenario.hardware.get_component(component).normalized_kind == "hbf"
                                for component in stores) else "hbm"
    used = {owner for batch in getattr(runtime, "batches", ())
            for owner in batch.cost.metadata.get("native_fast_budget_observations", {}).get("owner_accesses", {})}
    return "hbf" if any(getattr(active.core.config, "protocol_mode", None) == "generic_nand"
        for owner, active in runtime._execution_resource_kernel.physical_runtime.runtimes.items()
        if not used or owner in used) else "hbm"


def _paired_decision(bounds, reference, runtime, target):
    if reference is None:
        return {"accepted": True, "strategy": "independent_symmetric_sufficient_bound",
                "target_relative_error": target, "ratios": {}}
    reasons, ratios = [], {}
    candidate_kind = _memory_kind(runtime)
    if (reference.get("schema") != "native-window-paired-reference/v1"
            or reference.get("domain") != _domain(runtime)
            or reference.get("memory_kind") not in {"hbm", "hbf"}
            or reference.get("memory_kind") == candidate_kind
            or reference.get("target_relative_error") != target
            or reference.get("accepted") is not True
            or not reference.get("source_id")):
        return {"accepted": False, "reasons": ["native_window_paired_reference_domain_mismatch"]}
    for name in ("mean_complete_request_tpot_ns", "mean_complete_request_ingress_ttft_ns",
                 "cluster_tokens_per_second"):
        other = reference.get("metric_bounds", {}).get(name)
        if (not isinstance(other, Mapping) or not all(_finite(other.get(key), positive=True)
                for key in ("estimate", "lower", "upper"))
                or not other["lower"] <= other["estimate"] <= other["upper"]
                or not _within(other, target)):
            reasons.append("native_window_paired_reference_metric_invalid:"+name)
            continue
        numerator, denominator = ((bounds[name], other) if candidate_kind == "hbf" else (other, bounds[name]))
        ratio = _interval(numerator["estimate"]/denominator["estimate"],
            numerator["lower"]/denominator["upper"], numerator["upper"]/denominator["lower"])
        ne, nl, nu = (Fraction(numerator[key]) for key in ("estimate", "lower", "upper"))
        de, dl, du = (Fraction(denominator[key]) for key in ("estimate", "lower", "upper"))
        limit = Fraction(target)
        accepted = ne*du <= (1+limit)*nl*de and ne*dl >= (1-limit)*nu*de
        ratios[name] = {**ratio, "direction": "hbf/hbm", "within_target": accepted}
        if not accepted:
            reasons.append("native_window_directional_pair_error_exceeds_target:"+name)
    return {"accepted": not reasons, "reasons": reasons, "strategy": "matched_native_window_directional_pair",
            "target_relative_error": target, "ratios": ratios, "source_id": reference.get("source_id")}


def single_slot_skeleton(requests, warmup, measurement):
    """Token-count identity for the B=1 native full-output FIFO domain."""
    stop = warmup + measurement
    events, requests_out, decode = [], [], 0
    for spec in requests:
        if decode >= stop:
            break
        if spec.output_tokens < 2:
            raise ValueError("native_window_requires_at_least_two_outputs")
        first = len(events)
        events.append({"kind": "prefill", "request_id": spec.request_id,
                       "after_decode_step": decode, "context_tokens": spec.prompt_tokens-1})
        for local in range(1, spec.output_tokens):
            if decode >= stop:
                break
            decode += 1
            events.append({"kind": "decode", "request_id": spec.request_id,
                           "after_decode_step": decode,
                           "context_tokens": spec.prompt_tokens+local-1})
        complete = decode == events[first]["after_decode_step"] + spec.output_tokens-1
        requests_out.append({"request_id": spec.request_id, "first_event": first,
            "last_event": len(events)-1, "output_tokens": spec.output_tokens,
            "complete": complete, "warmup_seed": bool(spec.metadata.get("warmup_seed", False)),
            "arrival_ns": spec.arrival_ns})
    if decode != stop:
        raise ValueError("native_window_request_supply_does_not_cover_authored_window")
    start_event = next((index for index, row in enumerate(events)
                        if row["kind"] == "decode" and row["after_decode_step"] == warmup), -1)
    if warmup == 0:
        raise ValueError("native_window_zero_warmup_boundary_not_qualified")
    complete = tuple(row for row in requests_out if row["complete"] and not row["warmup_seed"]
                     and start_event < row["first_event"] <= row["last_event"] < len(events))
    return {"events": tuple(events), "requests": tuple(requests_out),
            "complete_fresh": complete, "start_event": start_event,
            "visible_output_tokens": len(events)-start_event-1, "decode_cohorts": stop}


def collect_native_budget_evidence(runtime, sample_records, options):
    """Expose what the live native prefix proves; missing bounds stay missing.

    This producer deliberately does not call a lowerer, preview physical IO,
    or convert a measured prefix average into an unseen-state certificate.
    """
    from .native_window_producer import native_continuation_evidence

    produced = native_continuation_evidence(runtime, sample_records, options)
    if produced is not None:
        return produced
    policy, ledger = runtime.plan.kv_policy, runtime.ledger
    pages = sum(ledger.pages_for_tokens(spec.prompt_tokens+spec.output_tokens-1)
                for spec in runtime.plan.requests)
    capacity = pages <= policy.capacity_pages
    ingress = runtime._external_ingress
    observations = [row.get("physical_observations", {}) for row in sample_records]
    kinds = set()
    for active in runtime._execution_resource_kernel.physical_runtime.runtimes.values():
        config = active.core.config
        kinds.add((getattr(config.kind, "value", str(config.kind)),
                   getattr(config, "protocol_mode", getattr(config, "control_mode", None))))
    generic = any(protocol == "generic_nand" for _kind, protocol in kinds)
    return {"schema": "native-window-budget-evidence/v1",
        "capacity": {"qualified": capacity,
            "basis": "all_authored_requests_simultaneous_logical_kv_reservation_upper_bound",
            "required_pages": pages, "capacity_pages": policy.capacity_pages,
            "physical_workspace_and_allocator_fragmentation_covered": False},
        "steady_contract": {"qualified": False,
            "reason": "native_future_context_topology_and_service_envelope_not_yet_proven"},
        "future_guards": {name: {"qualified": False, "lower_guard_ns": None,
            "upper_guard_ns": None, "basis": "native_projection_unavailable"} for name in (
                "pd_service_and_interference", "cache_and_page_phase", "refresh_and_row_state",
                "allocation_reuse", "cutoff_carry_in", "prefill_query")},
        "constant_path_floor_ns": 0., "constant_path_floor_proof": "zero_is_universal_lower_bound",
        "visible_commit_lag_ns": {"qualified": False,
            "reason": "future_visible_commit_to_cohort_cleanup_bound_unavailable"},
        "observed_physical_memory_domains": sorted(kinds, key=str),
        "generic_nand_gc": {"applicable": False if generic else None,
            "basis": "generic_nand_has_no_ftl_or_implicit_gc" if generic else "not_a_generic_nand_domain"},
        "sampled_physical_observations": observations,
        "external_pending_requests": len(runtime._external_pending),
        "external_imported_requests": runtime._external_imported_requests,
        "pd_configuration": dict(ingress),
        "physical_state_advanced": False}


def evaluate_native_window(runtime, sample_records, options, paired_reference=None, *, budget_evidence=None):
    from .native_window_producer import native_context_features

    target = options.target_relative_error
    if not _finite(target, positive=True) or target > .1:
        raise ValueError("native window target must be in (0,.1]")
    result = {"accepted": False, "reasons": [], "metrics": {}, "metric_bounds": {},
        "window": {}, "request_forecasts": [], "error_budget": {}, "semantic_guards": {
            "physical_state_advanced": False, "forecast_has_physical_trace": False,
            "complete_fresh_request_definition_preserved": True},
        "predicted_visible_output_tokens": 0, "can_expand_sampling": False}
    reasons = result["reasons"]
    scheduler, plan = runtime.plan.scheduler, runtime.plan
    ingress = runtime._external_ingress
    if scheduler.max_num_seqs != 1:
        from .native_multislot_budget import evaluate_native_multislot
        return evaluate_native_multislot(runtime, sample_records, options,
            paired_reference, budget_evidence=budget_evidence)
    if not ingress.get("require_full_batch", True):
        reasons.append("native_window_requires_full_batch_admission")
    if scheduler.preemption_enabled or scheduler.mixed_phase_batching:
        reasons.append("native_window_scheduler_regime_not_bounded")
    if any(spec.arrival_ns != plan.requests[0].arrival_ns or spec.priority != 0
           or spec.deadline_ns is not None for spec in plan.requests):
        reasons.append("native_window_requires_same_arrival_fifo_requests")
    if paired_reference is not None and (not isinstance(paired_reference, Mapping)
            or paired_reference.get("schema") != "native-window-paired-reference/v1"):
        reasons.append("native_window_reference_contract_mismatch")
    if reasons:
        return result
    try:
        skeleton = single_slot_skeleton(plan.requests, ingress["warmup_decode_cohorts"],
                                        ingress["measurement_decode_cohorts"])
    except ValueError as error:
        reasons.append(str(error))
        return result
    observed = tuple(sample_records)
    events = skeleton["events"]
    if len(observed) >= len(events):
        reasons.append("native_window_already_physically_complete")
        return result
    if any(row["kind"] != expected["kind"] or tuple(row["request_ids"]) != (expected["request_id"],)
           for row, expected in zip(observed, events)):
        reasons.append("native_prefix_disagrees_with_single_slot_request_lifecycle")
        return result
    decode = [row for row in observed if row["kind"] == "decode"]
    warmup, training, validation = options.warmup_cohorts, options.calibration_cohorts, options.validation_cohorts
    count = training+validation
    if len(decode) < warmup+count:
        reasons.append("native_complete_cohort_samples_insufficient")
        result["can_expand_sampling"] = True
        return result
    fit = decode[-count:]
    if any(not _finite(row["execution_ns"], positive=True) for row in fit):
        reasons.append("native_cohort_duration_invalid")
        return result
    contexts = [sum(item["context_tokens"] for item in row["context_tokens"]) for row in fit]
    if any(not _finite(value, positive=True) for value in contexts):
        reasons.append("native_context_samples_invalid")
        return result
    base = fmean(row["execution_ns"] for row in fit[:training])
    anchor = fmean(contexts[:training])
    feature_rows = [native_context_features(plan.scenario, [item["context_tokens"]
                    for item in row["context_tokens"]]) for row in fit]
    anchors = tuple(fmean(values[index] for values in feature_rows[:training])
                    for index in range(len(feature_rows[0])))
    if not anchors or any(not _finite(value, positive=True) for value in anchors):
        reasons.append("native_context_feature_anchor_invalid")
        return result
    def feature_ratios(features):
        return [value/anchor for value, anchor in zip(features, anchors)]
    predicted = sum(_midpoint(base*min(1., *feature_ratios(features)),
                             base*max(1., *feature_ratios(features)))
                    for features in feature_rows[training:])
    measured = sum(row["execution_ns"] for row in fit[training:])
    local = abs(predicted-measured)/measured
    half = training//2
    drift = (abs(fmean(row["execution_ns"] for row in fit[:half])
                 - fmean(row["execution_ns"] for row in fit[half:training]))/base) if half else 0.
    local = max(local, drift)
    result["error_budget"]["calibration"] = {"training_cohorts": training,
        "validation_cohorts": validation, "holdout_disjoint": True,
        "holdout_relative_error": local, "training_base_ns": base, "anchor_context": anchor}
    result["error_budget"]["calibration"]["native_context_feature_anchors"] = list(anchors)
    evidence = budget_evidence or collect_native_budget_evidence(runtime, observed, options)
    result["error_budget"]["native_evidence"] = evidence
    if not evidence.get("capacity", {}).get("qualified"):
        reasons.append("native_future_capacity_upper_bound_unavailable")
    if evidence.get("capacity", {}).get("physical_workspace_and_allocator_fragmentation_covered") is not True:
        reasons.append("native_future_physical_allocation_bound_unavailable")
    if evidence.get("steady_contract", {}).get("qualified") is not True:
        reasons.append("native_future_service_envelope_unavailable")
    guards = evidence.get("future_guards", {})
    lower_guard = upper_guard = 0.
    for name in ("pd_service_and_interference", "cache_and_page_phase", "refresh_and_row_state",
                 "allocation_reuse", "cutoff_carry_in", "prefill_query"):
        row = guards.get(name, {})
        if (row.get("qualified") is not True or not row.get("basis")
                or not _finite(row.get("lower_guard_ns")) or not _finite(row.get("upper_guard_ns"))):
            reasons.append("native_future_"+name+"_bound_unavailable")
        else:
            lower_guard += row["lower_guard_ns"]
            upper_guard += row["upper_guard_ns"]
    lag = evidence.get("visible_commit_lag_ns", {})
    if (lag.get("qualified") is not True or not lag.get("basis")
            or not _finite(lag.get("lower_ns")) or not _finite(lag.get("upper_ns"))
            or lag["lower_ns"] > lag["upper_ns"]):
        reasons.append("native_future_visible_commit_cleanup_bound_unavailable")
    if reasons:
        return result
    floor = evidence.get("constant_path_floor_ns", 0.)
    if (not _finite(floor) or floor > base or (floor and
            evidence.get("constant_path_floor_proof") != "all_paths_constant_floor")):
        reasons.append("native_constant_path_floor_not_proven")
        return result
    if local >= 1:
        reasons.append("native_local_calibration_error_unbounded")
        return result
    prefill = [row for row in observed if row["kind"] == "prefill"]
    if not prefill:
        reasons.append("native_first_query_sample_unavailable")
        return result
    query = fmean(row["execution_ns"] for row in prefill)
    event_guards = evidence.get("event_guards", {})
    for kind in ("decode", "prefill"):
        guard = event_guards.get(kind, {"lower_ns": 0., "upper_ns": 0.})
        if (not isinstance(guard, Mapping) or not _finite(guard.get("lower_ns"))
                or not _finite(guard.get("upper_ns"))):
            reasons.append("native_future_event_guard_invalid:"+kind)
    carry = event_guards.get("cutoff_carry_in_upper_ns", 0.)
    if not _finite(carry):
        reasons.append("native_future_cutoff_carry_in_bound_invalid")
    if reasons:
        return result
    origin = float(getattr(runtime, "_runtime_origin_ns", 0.))
    times, services, lags, current_lo, current_hi = [], [], [], origin, origin
    for index, event in enumerate(events):
        if index < len(observed):
            record = observed[index]
            commits = record.get("visible_commits", ())
            if (len(commits) != 1 or commits[0].get("request_id") != event["request_id"]
                    or commits[0].get("visible_tokens") != 1
                    or not _finite(commits[0].get("timestamp_ns"))
                    or not current_lo < commits[0]["timestamp_ns"] <= record["end_ns"]):
                reasons.append("native_sample_visible_commit_boundary_unavailable")
                return result
            services.append((record["end_ns"]-current_lo, record["end_ns"]-current_hi))
            cleanup = record["end_ns"]-commits[0]["timestamp_ns"]
            lags.append((cleanup, cleanup))
            current_lo = current_hi = observed[index]["end_ns"]
        else:
            if event["kind"] == "prefill":
                lo, hi = query, query
            else:
                ratios = feature_ratios(native_context_features(plan.scenario, [event["context_tokens"]]))
                lo = floor+(base-floor)*min(1., *ratios)
                hi = floor+(base-floor)*max(1., *ratios)
            event_guard = event_guards.get(event["kind"], {"lower_ns": 0., "upper_ns": 0.})
            service = (lo/(1+local)-event_guard["lower_ns"],
                       hi/(1-local)+event_guard["upper_ns"]+(carry if index == len(observed) else 0.))
            if service[0] <= 0:
                reasons.append("native_future_event_phase_exhausts_service_lower_bound")
                return result
            if lag["upper_ns"] >= service[0]:
                reasons.append("native_visible_commit_may_cross_cohort_boundary")
                return result
            services.append(service)
            lags.append((lag["lower_ns"], lag["upper_ns"]))
            current_lo += service[0]
            current_hi += service[1]
        times.append((current_lo, current_hi))
    start = times[skeleton["start_event"]]
    end = times[-1]
    duration_lo = math.fsum(row[0] for row in services[skeleton["start_event"]+1:])-lower_guard
    duration_hi = math.fsum(row[1] for row in services[skeleton["start_event"]+1:])+upper_guard
    if duration_lo <= 0:
        reasons.append("native_future_guard_exhausts_measurement_lower_bound")
        return result
    duration = _midpoint(duration_lo, duration_hi)
    request_rows, tpot_lo, tpot_hi, ttft_lo, ttft_hi = [], [], [], [], []
    for request in skeleton["complete_fresh"]:
        first, last = times[request["first_event"]], times[request["last_event"]]
        first_lag, last_lag = lags[request["first_event"]], lags[request["last_event"]]
        lifetime = services[request["first_event"]+1:request["last_event"]+1]
        lo = (math.fsum(row[0] for row in lifetime)-last_lag[1]+first_lag[0]-lower_guard)/(request["output_tokens"]-1)
        hi = (math.fsum(row[1] for row in lifetime)-last_lag[0]+first_lag[1]+upper_guard)/(request["output_tokens"]-1)
        flo = first[0]-first_lag[1]-request["arrival_ns"]-lower_guard
        fhi = first[1]-first_lag[0]-request["arrival_ns"]+upper_guard
        if min(lo, flo) <= 0:
            reasons.append("native_future_guard_exhausts_fresh_request_lower_bound")
            return result
        tpot_lo.append(lo); tpot_hi.append(hi); ttft_lo.append(flo); ttft_hi.append(fhi)
        label_permutation = evidence.get("steady_contract", {}).get("future_fresh_request_labels_may_permute") is True
        identity = ({"request_id": None, "logical_fifo_slot_id": request["request_id"],
            "possible_same_shape_request_ids": [spec.request_id for spec in plan.requests
                if not spec.metadata.get("warmup_seed", False)],
            "per_request_label_order_proven": False} if label_permutation else {})
        request_rows.append({**request, **identity, "forecast_only": True,
            "tpot_ns": _interval(_midpoint(lo, hi), lo, hi),
            "ingress_ttft_ns": _interval(_midpoint(flo, fhi), flo, fhi)})
    if not request_rows:
        reasons.append("native_window_has_no_complete_fresh_request")
        return result
    tlo, thi, flo, fhi = fmean(tpot_lo), fmean(tpot_hi), fmean(ttft_lo), fmean(ttft_hi)
    token_count = skeleton["visible_output_tokens"]
    tpot = _interval(_midpoint(tlo, thi), tlo, thi)
    ttft = _interval(_midpoint(flo, fhi), flo, fhi)
    throughput = _interval(token_count*1e9/duration, token_count*1e9/duration_hi,
                           token_count*1e9/duration_lo)
    bounds = {"mean_complete_request_tpot_ns": tpot,
        "mean_complete_request_ingress_ttft_ns": ttft, "cluster_tokens_per_second": throughput,
        "measurement_duration_ns": _interval(duration, duration_lo, duration_hi)}
    limit = target/(2+target) if paired_reference is None else target
    if any(not _within(row, limit) for row in bounds.values()):
        reasons.append("native_window_single_point_or_independent_pair_budget_exceeded")
    paired = _paired_decision(bounds, paired_reference, runtime, target)
    if not paired["accepted"]:
        reasons.extend(paired["reasons"])
    sampled_visible = sum(row.get("visible_output_tokens", 0) for row in observed)
    result.update(metric_bounds=bounds, request_forecasts=request_rows,
        predicted_visible_output_tokens=max(0, len(events)-sampled_visible),
        window={"warmup_decode_cohorts": ingress["warmup_decode_cohorts"],
            "requested_decode_cohorts": ingress["measurement_decode_cohorts"],
            "measurement_visible_output_tokens": token_count,
            "complete_fresh_requests": len(request_rows), "forecast_only": True},
        error_budget={**result["error_budget"], "conditional_only": True,
            "lower_guard_ns": lower_guard, "upper_guard_ns": upper_guard,
            "per_point_target": target, "effective_independent_target": limit,
            "paired_budget_decision": paired})
    if not reasons:
        world = plan.scenario.placement.parallel.tp_degree*plan.scenario.placement.parallel.pp_degree*plan.scenario.placement.parallel.ep_degree
        result.update(accepted=True, metrics={"cluster_tokens_per_second": throughput["estimate"],
            "tokens_per_second_per_gpu": throughput["estimate"]/world,
            "mean_complete_request_tpot_ns": tpot["estimate"],
            "mean_complete_request_ingress_ttft_ns": ttft["estimate"],
            "complete_fresh_requests": len(request_rows),
            "measurement_visible_output_tokens": token_count, "measurement_duration_ns": duration})
        result["error_budget"]["paired_reference"] = {"schema": "native-window-paired-reference/v1",
            "domain": _domain(runtime), "memory_kind": _memory_kind(runtime),
            "source_id": plan.scenario.name, "target_relative_error": target,
            "metric_bounds": bounds, "accepted": True}
    return result


__all__ = ["evaluate_native_window", "collect_native_budget_evidence", "single_slot_skeleton"]
