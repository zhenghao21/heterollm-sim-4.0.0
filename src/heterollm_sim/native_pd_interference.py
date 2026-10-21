"""Measured PD import interference and the B=1 future import timeline.

A PD import only adds work to the shared memory path: while one import is
active, quiet native work W takes W*(1+s), and an import of B bytes stays
active for T = B/r. Both the slowdown s and the contended import rate r come
from paired quiet/concurrent native samples of the real in-flight import
(``native_layer_sampling.sample_native_pd_interference``). No per-import
worst-case span is used; anything the samples do not cover fails closed.
"""
from __future__ import annotations

from collections import deque
import math


def _fail(reason, **evidence):
    return {"qualified": False, "reason": reason, **evidence}


def _pair_import(row, continued):
    observed = row["concurrent"]["concurrent_imports"]
    if (observed is None or observed["request_ids"] != [continued] or observed["final_reached"]
            or row["quiet"]["task_count"] != row["concurrent"]["task_count"]):
        return None
    return observed


def interference_model(samples):
    """Slowdown and contended-rate intervals from calibration/holdout pairs.

    Per layer window, s is the completion slope after the first layer (the
    lock-in transient) relative to the quiet slope. A layer's excess is
    quantized by how many import writes land in it, so each window carries
    Q = (max - min per-layer excess) / quiet span. Decode and the import
    phase-lock, and different start phases can lock into different stable
    patterns, so calibration and holdout may differ beyond Q. The interval
    is the hull of s +- Q, widened on both sides by that calibration/holdout
    drift as the margin for a pattern neither window reached. Non-layer
    groups enter with their quiet weight and the largest observed
    single-layer excess as their quantum.
    """
    if not samples or not samples.get("qualified"):
        return _fail((samples or {}).get("reason") or "native_PD_interference_samples_unavailable")
    continued = samples["continued_request_id"]
    windows = []
    for row in samples["layer_windows"]:
        observed = _pair_import(row, continued)
        if observed is None:
            return _fail("native_PD_interference_window_not_single_unfinished_import")
        quiet, busy = row["quiet"]["layer_completion_ns"], row["concurrent"]["layer_completion_ns"]
        q = [quiet[layer]-row["origin_ns"] for layer in row["layer_ids"]]
        c = [busy[layer]-row["origin_ns"] for layer in row["layer_ids"]]
        if (len(q) < 2 or not all(map(math.isfinite, (*q, *c)))
                or any(b <= a for a, b in zip(q, q[1:])) or any(b <= a for a, b in zip(c, c[1:]))):
            return _fail("native_PD_interference_layer_completions_invalid")
        if observed["write_phases_in_window"] < 2:
            return _fail("native_PD_interference_window_has_too_few_import_writes")
        span = q[-1]-q[0]
        excess = [(c1-c0)-(q1-q0) for q0, q1, c0, c1 in zip(q, q[1:], c, c[1:])]
        busy_span = row["concurrent"]["end_ns"]-row["concurrent"]["start_ns"]
        windows.append({"role": row["role"], "layers": len(q),
            "slowdown": (c[-1]-c[0])/span-1., "quantization": (max(excess)-min(excess))/span,
            "transient_ns": c[0]-q[0], "max_layer_excess_ns": max(excess),
            "quiet_per_layer_ns": span/(len(q)-1),
            "write_rate_bytes_per_ns": observed["written_bytes_in_window"]/busy_span,
            "writes": observed["write_phases_in_window"]})
    roles = {row["role"]: row for row in windows}
    if set(roles) != {"calibration", "holdout"} or len(windows) != 2:
        return _fail("native_PD_interference_requires_calibration_and_holdout")
    calibration, holdout = roles["calibration"], roles["holdout"]
    drift = abs(calibration["slowdown"]-holdout["slowdown"])
    quantum = max(max(row["max_layer_excess_ns"], row["transient_ns"]) for row in windows)
    layer_lo = min(row["slowdown"]-row["quantization"] for row in windows)-drift
    layer_hi = max(row["slowdown"]+row["quantization"] for row in windows)+drift
    per_layer = max(row["quiet_per_layer_ns"] for row in windows)
    weight = samples["layer_count"]*per_layer
    total_lo, total_hi, total_weight, non_layer = weight*layer_lo, weight*layer_hi, weight, []
    for row in samples["non_layer_windows"]:
        if _pair_import(row, continued) is None:
            return _fail("native_PD_interference_window_not_single_unfinished_import")
        quiet, busy = row["quiet"]["elapsed_from_boundary_ns"], row["concurrent"]["elapsed_from_boundary_ns"]
        if not (quiet > 0 and math.isfinite(busy)):
            return _fail("native_PD_interference_non_layer_window_invalid")
        slowdown, band = busy/quiet-1., quantum/quiet
        non_layer.append({"position": row["position"], "quiet_ns": quiet,
                          "slowdown": slowdown, "quantization": band})
        total_lo += quiet*(slowdown-band); total_hi += quiet*(slowdown+band); total_weight += quiet
    lower, upper = total_lo/total_weight, total_hi/total_weight
    rate_lo = min(row["write_rate_bytes_per_ns"]*(1.-1./row["writes"]) for row in windows)
    rate_hi = max(row["write_rate_bytes_per_ns"]*(1.+1./row["writes"]) for row in windows)
    if not (lower > -1. and rate_lo > 0. and math.isfinite(upper) and math.isfinite(rate_hi)):
        return _fail("native_PD_interference_interval_invalid")
    return {"qualified": True, "reason": None, "continued_request_id": continued,
        "slowdown": {"lower": lower, "upper": upper},
        "layer_slowdown": {"lower": layer_lo, "upper": layer_hi},
        "calibration_holdout_drift": drift,
        "write_rate_bytes_per_ns": {"lower": rate_lo, "upper": rate_hi},
        "quantum_ns": quantum, "windows": windows, "non_layer": non_layer,
        "basis": "paired_same_boundary_quiet_vs_single_real_PD_import; post_transient_layer_slope; write_count_quantization; calibration_holdout_hull_widened_by_drift"}


def single_slot_import_timeline(runtime, skeleton, pd, services, model):
    """Anchor every future import on the B=1 skeleton and bound its extra time.

    serving._start_external_imports runs once per boundary after admission
    and starts WAITING requests in state order while pending+ready < limit.
    A completion only moves pending to ready, so starts depend on admissions
    alone and each import has an exact boundary anchor. An admission must be
    proven not to wait for its import. A single active import adds
    T*s/(1+s) with T in [B/r_hi, (B+chunk)/r_lo + initial-state phase],
    widened by one measured layer quantum at each end. An import that may
    overlap another is charged as a k=2 tail: one quantum per chunk, and its
    whole span extends the other import.
    """
    future = {row["request_id"]: row for row in pd["future_requests"]}
    if not future:
        return {"qualified": True, "reason": None, "imports": [], "total_upper_ns": 0.,
                "span_extra": lambda first, last: (0., 0.), "basis": "no_future_PD_import"}
    if not model.get("qualified"):
        return _fail(model.get("reason") or "native_PD_interference_unavailable")
    config, executor = runtime._external_ingress, runtime._external_executor
    limit = config.get("max_inflight_requests", 2)
    if runtime.plan.scheduler.max_num_seqs != 1 or type(limit) is not int or limit < 1:
        return _fail("native_PD_timeline_requires_single_slot_and_integer_inflight_limit")
    chunk = min(config.get("chunk_bytes", 1024**2), config.get("buffer_bytes", 64*1024**2)//limit)
    pending, ready = list(runtime._external_pending), set(runtime._external_ready)
    states = {state.get("request_id"): state for state in executor._async_plan_states.values()
              if not state.get("done") and "destinations" in state}
    if not set(pending) <= set(states) or not set(pending) <= set(future):
        return _fail("native_PD_in_flight_plan_state_unavailable")
    from .native_layer_sampling import remaining_import_bytes
    order = [state.spec.request_id for state in runtime._state_values]
    waiting = deque(identity for identity in order
                    if identity in future and identity not in pending and identity not in ready)
    # The start rule also needs ledger room. With equal request shapes, KV
    # released at finish and nothing retained, a later start never holds
    # more requests than the slots+limit already held together now.
    if waiting and (sum(state.kv_pages > 0 for state in runtime._state_values) < 1+limit
                    or getattr(runtime, "_retained_kv_state", None) is not None
                    or runtime._prompt_cache_save_enabled()
                    or len({(spec.prompt_tokens, spec.output_tokens) for spec in runtime.plan.requests}) != 1):
        return _fail("native_PD_future_import_capacity_not_proven")
    s_lo, s_hi = model["slowdown"]["lower"], model["slowdown"]["upper"]
    r_lo, r_hi = model["write_rate_bytes_per_ns"]["lower"], model["write_rate_bytes_per_ns"]["upper"]
    quantum = model["quantum_ns"]
    lo_prefix, hi_prefix = [0.], [0.]
    for lower, upper in services:
        lo_prefix.append(lo_prefix[-1]+lower); hi_prefix.append(hi_prefix[-1]+upper)
    elapsed_lo = lambda a, b: lo_prefix[b]-lo_prefix[a]
    work_hi = lambda a, b: hi_prefix[b]-hi_prefix[a]
    imports = {}

    def start(identity, anchor, size, in_flight=0):
        # A live cursor advances only after a chunk's last phase, so the
        # remaining bytes include the chunk in flight, which may be partly done.
        chunks = math.ceil(size/chunk)
        source = future[identity]
        phase = source["initial_state_phase_serial_upper_ns"]*chunks/source["chunk_count"]
        imports[identity] = {"request_id": identity, "anchor_boundary": anchor, "bytes": size,
            "chunks": chunks, "span_lower_ns": (size-in_flight)/r_hi,
            "span_upper_ns": size/r_lo+phase, "initial_state_phase_upper_ns": phase}

    for identity in pending:
        state = states[identity]
        start(identity, 0, remaining_import_bytes(state), state["amount"] if state["actions"] else 0)
    held, admissions = set(pending) | ready, []
    for index, event in enumerate(skeleton["events"]):
        if index == 0 or event["kind"] != "prefill":
            continue
        identity = event["request_id"]
        if identity not in imports and identity not in ready:
            return _fail("native_PD_admission_import_not_started")
        admissions.append((identity, index))
        held.discard(identity)
        while len(held) < limit and waiting:
            follower = waiting.popleft()
            start(follower, index, future[follower]["logical_bytes"])
            held.add(follower)
    rows = sorted(imports.values(), key=lambda row: (row["anchor_boundary"], row["request_id"]))
    for row in rows:
        row.update(tail=False, extension_ns=0.)
    # Two imports that may be active together are outside the k=1 samples.
    for _ in range(len(rows)):
        changed = False
        for left_index, left in enumerate(rows):
            for right in rows[left_index+1:]:
                if elapsed_lo(left["anchor_boundary"], right["anchor_boundary"]) >= left["span_upper_ns"]:
                    continue
                tail, other = sorted((left, right), key=lambda row: (row["span_upper_ns"], row["request_id"]))
                if not tail["tail"]:
                    tail["tail"] = changed = True
                extension = max(other["extension_ns"], tail["span_upper_ns"])
                if extension != other["extension_ns"]:
                    other["span_upper_ns"] += extension-other["extension_ns"]
                    other["extension_ns"], changed = extension, True
        if not changed:
            break
    for identity, index in admissions:
        row = imports.get(identity)
        if row is not None and elapsed_lo(row["anchor_boundary"], index) < row["span_upper_ns"]:
            return _fail("native_PD_admission_may_wait_for_import")
    total = 0.
    for row in rows:
        if row["tail"]:
            row["extra_ns"] = (0., row["chunks"]*quantum)
        else:
            row["extra_ns"] = (max(0., row["span_lower_ns"]*s_lo/(1.+s_lo)-2.*quantum),
                               row["span_upper_ns"]*s_hi/(1.+s_hi)+2.*quantum)
        total += row["extra_ns"][1]

    def contribution(row, first, last):
        anchor = row["anchor_boundary"]
        lower, upper = row["extra_ns"]
        if anchor >= last or (anchor < first and elapsed_lo(anchor, first) >= row["span_upper_ns"]):
            return 0., 0.
        if row["tail"]:
            return 0., upper
        # Extra accrues only while quiet work runs under the import: work W
        # gains at most W*s_hi plus one quantum at a cut.
        before = 0. if anchor >= first else min(upper, work_hi(anchor, first)*max(0., s_hi)+quantum)
        inside_hi = min(upper, work_hi(max(anchor, first), last)*max(0., s_hi)+quantum)
        ends_inside = elapsed_lo(anchor, last) >= row["span_upper_ns"]
        return (max(0., lower-before) if ends_inside else 0.), inside_hi

    def span_extra(first, last):
        parts = [contribution(row, first, last) for row in rows]
        return math.fsum(row[0] for row in parts), math.fsum(row[1] for row in parts)

    return {"qualified": True, "reason": None, "imports": rows, "total_upper_ns": total,
        "inflight_limit": limit, "chunk_bytes": chunk, "span_extra": span_extra,
        "basis": "admission_only_import_starts_on_B1_skeleton; no_admission_wait_proven; k1_measured_slowdown_over_contended_span; k2_overlap_charged_as_tail"}


__all__ = ["interference_model", "single_slot_import_timeline"]
