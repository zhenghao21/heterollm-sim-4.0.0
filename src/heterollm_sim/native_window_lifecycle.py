"""Logical cohort identities selected by the original native scheduler.

This is an un-timed forecast skeleton. It has no Core, physical calendar,
ServingResult or fabricated completion timestamp. The native selector and
BatchItem builders are reused directly; capacity and PD timing must be
qualified separately before a caller may use this lifecycle for a forecast.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from .serving import (_OnlineRuntime, RequestStatus,
                      _batch_item_committed_tokens, _serving_resource_policy)


_NATIVE_SELECTION_METHODS = tuple((name, getattr(_OnlineRuntime, name)) for name in (
    "_fill_slots", "_select_cohort", "_decode_cohort", "_prefill_cohort", "_rank", "_service_rank"))


def native_admission_washout(runtime):
    """Prove full fresh-wave shapes after already-active seeds wash out.

    Request labels in a later equal-shape wave may be permuted by paid PD.
    No per-request TTFT order is claimed. The proof is only that existing
    smaller-id seed Decode items prevent new queries until those seeds exit;
    the full-slot gate then admits one identical fresh query wave.
    """
    from .serving import _TERMINAL_STATUSES

    plan, scheduler, reasons = runtime.plan, runtime.plan.scheduler, []
    resource = _serving_resource_policy(plan.scenario)
    states = tuple(runtime._state_values)
    seeds = [state for state in states if state.spec.metadata.get("warmup_seed", False)]
    fresh = [state for state in states if not state.spec.metadata.get("warmup_seed", False)]
    live_seed = [state for state in seeds if state.status not in _TERMINAL_STATUSES]
    if (scheduler.policy.strip().lower() != "decode_first" or scheduler.preemption_enabled
            or scheduler.mixed_phase_batching or plan.mtp.enabled
            or any(getattr(_OnlineRuntime, name) is not original
                   for name, original in _NATIVE_SELECTION_METHODS)):
        reasons.append("time_dependent_or_mixed_native_selection")
    if (resource.enabled or resource.prompt_cache is not None and resource.prompt_cache.enabled
            or plan.linear_state_policy.bytes_per_request or plan.kv_policy.offload_component
            or plan.kv_policy.prefetch_distance
            or plan.scenario.workload.metadata.get("agent_lifecycle")):
        reasons.append("resource_or_retained_state_can_change_fresh_admission")
    if (not runtime._external_ingress.get("require_full_batch", True)
            or scheduler.max_num_batched_tokens < scheduler.max_num_seqs
            or scheduler.prefill_chunk_tokens != 1):
        reasons.append("fresh_query_cannot_form_full_single_token_wave")
    if any(state.status is not RequestStatus.RUNNING
            or state.prefill_cursor != state.spec.prompt_tokens or state.committed < 1
            for state in live_seed):
        reasons.append("warmup_seed_still_waiting_or_prefilling")
    if seeds and fresh and max(state.spec.request_id for state in seeds) >= min(
            state.spec.request_id for state in fresh):
        reasons.append("seed_ids_do_not_precede_fresh_query_ids")
    if (not fresh or len({(state.spec.prompt_tokens, state.spec.output_tokens) for state in fresh}) != 1
            or any(state.committed != 0 or state.status in _TERMINAL_STATUSES
                   or state.prefill_cursor not in (0, state.spec.prompt_tokens-1)
                   for state in fresh)):
        reasons.append("fresh_wave_has_unequal_shape_or_already_served_items")
    if any(state.spec.priority != 0 or state.spec.deadline_ns is not None
            or state.spec.arrival_ns != states[0].spec.arrival_ns for state in states):
        reasons.append("time_priority_can_promote_fresh_query")
    if not live_seed:
        reasons.append("no_live_seed_washout_boundary")
    ingress_limit = runtime._external_ingress.get("max_inflight_requests", 2)
    # Every seed is already Decode-eligible, has an id before every fresh
    # query, and the token/sequence budget covers the entire running set.
    # The original selector therefore advances every remaining seed once
    # in each selected Decode cohort, until the longest one retires.
    bound = max((max(0, state.spec.output_tokens-state.committed) for state in live_seed), default=0)
    return {"qualified": not reasons, "reasons": reasons,
        "basis": "native_decode_first_id_order_full_slot_gate_and_equal_unserved_fresh_wave",
        "active_seed_request_ids": [state.spec.request_id for state in live_seed],
        "seed_decode_cohort_upper": bound,
        "fresh_batch_size": scheduler.max_num_seqs,
        "fresh_wave_query_tokens": scheduler.max_num_seqs,
        "future_pd_max_inflight_requests": ingress_limit,
        "per_request_label_order_proven": False,
        "forecast_requires_per_request_PD_and_batch_shape_timing_bounds": True}


class _LogicalSelection:
    # These are the same phase ranking, admission order and BatchItem packing
    # functions used by the live runtime, not a second batching algorithm.
    for _name, _method in _NATIVE_SELECTION_METHODS:
        locals()[_name] = _method
    del _name, _method

    def __init__(self, source):
        plan = source.plan
        self.plan, self.now, self.events = plan, 0., []
        self.batches = [None]*len(source.batches)
        self._state_values = tuple(replace(state,
            kv_cache_range_tokens=list(state.kv_cache_range_tokens)) for state in source._state_values)
        for state in self._state_values:
            if state.spec.request_id not in source._external_initialized:
                state.prefill_cursor = state.spec.prompt_tokens-1
        self.states = {state.spec.request_id: state for state in self._state_values}
        self._request_count = len(self._state_values)
        self._running_count = source._running_count
        self._admission_sequence, self._service_sequence = source._admission_sequence, source._service_sequence
        self._free_slot_ids = list(source._free_slot_ids)
        self._agent_paused = {}
        self._external_executor = object()
        self._external_initialized = set(self.states)
        self._external_ready = set(self.states)
        self._external_ingress = {**source._external_ingress, "require_full_batch": True}
        self._kv_scan_enabled = False
        self.resource_policy = SimpleNamespace(enabled=False)
        self.prompt_cache = SimpleNamespace(ensure_capacity=lambda _size: True,
            policy=SimpleNamespace(save_implementation="disabled"))
        self.state_ledger = SimpleNamespace(restore=lambda _state: True)
        quantum = plan.kv_policy.tokens_per_page
        self.ledger = SimpleNamespace(pages_for_tokens=lambda n: max(0, n+quantum-1)//quantum,
            resize=lambda state, pages: self._reserve_with_pressure(state, pages))

    def _save_pending_prompt_cache_states(self):
        pass

    def _eager_slot_available(self, _count):
        return True

    def _record_llama_engine_start(self, _state):
        pass

    def _reserve_with_pressure(self, state, pages, _protected=()):
        # A separate physical/allocation envelope must prove this assumption.
        # This is a logical reservation; no address or Core is allocated.
        state.kv_pages = pages
        return True

    def _set_status(self, state, status):
        if state.status == status:
            return
        if state.status is RequestStatus.RUNNING:
            self._running_count -= 1
            self._free_slot_ids.append(state.slot_id)
            self._free_slot_ids.sort()
            state.slot_id = None
        if status is RequestStatus.RUNNING:
            state.slot_id = self._free_slot_ids.pop(0)
            self._running_count += 1
        state.status = status

    def commit(self, cohort):
        commits = []
        for item in cohort.items:
            state = self.states[item.request_id]
            if item.phase == "prefill":
                state.prefill_cursor += item.token_count
                visible = int(state.prefill_cursor >= state.spec.prompt_tokens and state.committed == 0)
            else:
                visible = min(_batch_item_committed_tokens(item), state.spec.output_tokens-state.committed)
            state.committed += visible
            if state.committed >= state.spec.output_tokens:
                self._set_status(state, RequestStatus.FINISHED)
            self._service_sequence += 1
            state.last_service_sequence = self._service_sequence
            state.queued_since_ns = self.now
            if visible:
                commits.append({"request_id": item.request_id,
                    "visible_tokens": visible, "committed": state.committed})
        self.batches.append(cohort)
        # Native admission emits logical events internally. No such untimed
        # event is exposed as an executed ServingEvent by this helper.
        self.events.clear()
        return tuple(commits)


def native_lifecycle_skeleton(runtime, warmup=None, measurement=None, *, sample_records=None,
                              _initial_forecast_cohort=None):
    """Select future batch identities under an explicit, un-timed PD gate.

    PD readiness is assumed before selection when a slot is free. It is not
    assumed free: callers must bound each corresponding admission gap. This
    domain is time-independent ``decode_first`` FIFO; aging, other arrivals,
    resource pressure or cache state can change selection and are rejected.
    """
    from .native_fast_window import cohort_samples

    plan, scheduler = runtime.plan, runtime.plan.scheduler
    ingress = runtime._external_ingress
    warmup = ingress["warmup_decode_cohorts"] if warmup is None else warmup
    measurement = ingress["measurement_decode_cohorts"] if measurement is None else measurement
    resource = _serving_resource_policy(plan.scenario)
    if (any(getattr(_OnlineRuntime, name) is not original for name, original in _NATIVE_SELECTION_METHODS)
            or scheduler.policy.strip().lower() != "decode_first"
            or scheduler.preemption_enabled or scheduler.mixed_phase_batching
            or plan.mtp.enabled or resource.enabled
            or resource.prompt_cache is not None and resource.prompt_cache.enabled
            or plan.linear_state_policy.bytes_per_request
            or plan.kv_policy.offload_component or plan.kv_policy.prefetch_distance
            or plan.scenario.workload.metadata.get("agent_lifecycle")):
        raise ValueError("native_lifecycle_requires_closed_time_independent_native_scheduler")
    if (not plan.requests or any(spec.prompt_tokens < 1 or spec.output_tokens < 2
            or spec.arrival_ns != plan.requests[0].arrival_ns or spec.priority != 0
            or spec.deadline_ns is not None for spec in plan.requests)):
        raise ValueError("native_lifecycle_requires_same_arrival_plain_fifo_requests")
    if type(warmup) is not int or warmup < 1 or type(measurement) is not int or measurement < 1:
        raise ValueError("native_lifecycle_requires_positive_explicit_decode_window")
    future = [spec for spec in plan.requests if spec.request_id not in runtime._external_initialized]
    if (any(spec.metadata.get("warmup_seed", False) for spec in future)
            or len({(spec.prompt_tokens, spec.output_tokens) for spec in future}) > 1):
        # Paid PD can complete out of FIFO order for different histories.
        # We use observed admission state, never invent the order of an
        # unimported aged seed versus a shorter fresh request.
        raise ValueError("native_lifecycle_heterogeneous_unimported_requests_not_ordered")
    records = tuple(cohort_samples(runtime) if sample_records is None else sample_records)
    if len(records) != len(runtime.batches):
        raise ValueError("native_lifecycle_requires_complete_real_prefix_records")
    logical = _LogicalSelection(runtime)
    stop, decode, events, by_request, start_event = warmup+measurement, 0, [], {}, None

    def append_event(row):
        nonlocal decode, start_event
        if row["kind"] == "decode":
            decode += 1
        index = len(events)
        if decode == warmup and row["kind"] == "decode":
            start_event = index
        row = {**row, "after_decode_step": decode}
        events.append(row)
        for commit in row["visible_commits"]:
            request_id = commit["request_id"]
            spec = logical.states[request_id].spec
            request = by_request.setdefault(request_id, {"request_id": request_id,
                "first_event": index, "output_tokens": spec.output_tokens,
                "warmup_seed": bool(spec.metadata.get("warmup_seed", False)),
                "arrival_ns": spec.arrival_ns})
            request.update(last_event=index, complete=commit["committed"] >= spec.output_tokens)

    for record in records:
        append_event({"kind": record["kind"], "cohort_id": record["cohort_id"],
            "request_ids": tuple(record["request_ids"]),
            "context_tokens": sum(item["context_tokens"] for item in record["context_tokens"]),
            "items": tuple(dict(item) for item in record["context_tokens"]),
            "visible_commits": tuple({key: item.get(key) for key in
                ("request_id", "visible_tokens", "committed")} for item in record["visible_commits"]),
            "visible_output_tokens": record["visible_output_tokens"], "observed_prefix": True})
    if _initial_forecast_cohort is not None:
        initial = _initial_forecast_cohort
        if (records or runtime.batches or initial.kind != "prefill"
                or len(initial.items) != scheduler.max_num_seqs
                or any(item.phase != "prefill" or item.token_count != 1
                    or runtime.states[item.request_id].status is not RequestStatus.RUNNING
                    or runtime.states[item.request_id].prefill_cursor != runtime.states[item.request_id].spec.prompt_tokens-1
                    or runtime.states[item.request_id].committed != 0 for item in initial.items)):
            raise ValueError("native_layer_lifecycle_requires_real_first_full_final_query_state")
        commits = logical.commit(initial)
        append_event({"kind": initial.kind, "cohort_id": initial.cohort_id,
            "request_ids": tuple(initial.request_ids),
            "context_tokens": sum(item.context_tokens for item in initial.items),
            "items": tuple({"request_id": item.request_id, "phase": item.phase,
                "token_count": item.token_count, "context_tokens": item.context_tokens,
                "committed_tokens": item.committed_tokens} for item in initial.items),
            "visible_commits": commits, "visible_output_tokens": sum(row["visible_tokens"] for row in commits),
            "observed_prefix": False, "forecast_only": True})
    washout = native_admission_washout(logical if _initial_forecast_cohort is not None else runtime)
    while decode < stop:
        logical._fill_slots()
        cohort = logical._select_cohort()
        if cohort is None:
            raise ValueError("native_lifecycle_request_supply_cannot_cover_window")
        commits = logical.commit(cohort)
        append_event({"kind": cohort.kind, "cohort_id": cohort.cohort_id,
            "request_ids": tuple(cohort.request_ids),
            "context_tokens": sum(item.context_tokens for item in cohort.items),
            "items": tuple({"request_id": item.request_id, "phase": item.phase,
                "token_count": item.token_count, "context_tokens": item.context_tokens,
                "committed_tokens": item.committed_tokens} for item in cohort.items),
            "visible_commits": commits,
            "visible_output_tokens": sum(row["visible_tokens"] for row in commits)})
    if start_event is None:
        raise ValueError("native_lifecycle_warmup_boundary_unavailable")
    requests = tuple(by_request.values())
    return {"events": tuple(events), "requests": requests,
        "complete_fresh": tuple(row for row in requests if row["complete"]
            and not row["warmup_seed"] and row["first_event"] > start_event),
        "start_event": start_event,
        "visible_output_tokens": sum(row["visible_output_tokens"] for row in events[start_event+1:]),
        "decode_cohorts": decode,
        "lifecycle_contract": "native_selector_with_capacity_proven_and_PD_ready_before_slot_selection",
        "future_request_id_semantics": "canonical_FIFO_labels_for_equal_shape_fresh_PD_requests",
        "future_admission_order_proven": not future,
        "future_admission_order_bound_required": bool(future),
        "fresh_wave_shapes_qualified_after_seed_washout": washout["qualified"],
        "seed_washout": washout,
        "physical_state_advanced": False, "timestamps_generated": False}


def native_layer_lifecycle_skeleton(runtime, cohort):
    """Forecast a first query commit in a private logical selector only.

    A layer sampler has not physically committed this query. The explicit
    entry point prevents that value-only commit from being mistaken for a
    sampled ServingResult or a visible Token observed on the live runtime.
    """
    result = native_lifecycle_skeleton(runtime, sample_records=(), _initial_forecast_cohort=cohort)
    return {**result, "lifecycle_source": "real_prequery_admission_and_forecast_query_commit",
        "physical_complete_cohort_samples": 0, "physical_visible_output_tokens": 0}


__all__ = ["native_lifecycle_skeleton", "native_layer_lifecycle_skeleton", "native_admission_washout"]
