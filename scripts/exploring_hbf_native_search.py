"""Paper-style deployment search over native fast candidates.

Per case the bypass screen (``exploring_hbf_search.search_candidates``) ranks
every legal layout and brackets its batch. The fixed ``max_batch_under_slo``
mode measures the two best selected layouts without an artificial point or
attempt cap, stopping at the SLO bracket; the separate ``throughput`` mode
keeps its own point budget. The TPOT upper bound is interpolated to the 100 ms SLO, and one
native acceptance run at the best layout's interpolated batch settles the
point. Bypass values only place batch points; every reported number is a
native candidate row.

A layout whose 100 ms batch lies beyond what a native candidate can run (its
KV capacity, wall cap, import speed, or memory projected from its measured
points) stops at its largest measured point and is marked limited; it is never
extrapolated. The bypass KV capacity bounds the points until a native point
reaches it; past it native capacity rejections do. A KV capacity rejection
reports the largest batch the native KV holds, and the next point goes
straight there. A refill-not-ready rejection is also a too-big point: the
existing halving/bisection path is reused, and its final limit is labelled
native_import_limited. A failed acceptance falls back to the best measured
passing point.

The plan is a pure function of the screen and the saved candidate rows, so a
stopped search resumes by replaying it.
"""
from __future__ import annotations

from dataclasses import dataclass

TOP_LAYOUTS = 2
# Measured batch points per layout, and native runs per layout including
# attempts that were too big to measure.
MAX_POINTS = 6
MAX_RUNS = 10
# Native cost grows with the batch (405B B=1024 needs ~1540 s); the first
# point stays below this.
FIRST_POINT_MAX_BATCH = 512
# A planned point's wall, projected linearly in B from the largest measured
# point, stays within this share of the per-worker candidate cap. The
# measured wall counts as it is: concurrent workers slow a candidate far less
# than their count (405B TP8 B=512: about 1.07x beside one other), and the
# k-fold cap of a candidate started beside others only absorbs that slowdown.
# Its peak memory, projected the same way, stays within the memory budget.
WALL_FILL = .9
# A new point differs from every measured batch by at least this share (and a
# quantum) and repeats no tried one; a limit closer than that ends the layout.
MIN_STEP = .1
THROUGHPUT_SETTINGS = {"top_layouts": TOP_LAYOUTS, "max_points_per_layout": MAX_POINTS, "max_runs_per_layout": MAX_RUNS,
            "first_point_max_batch": FIRST_POINT_MAX_BATCH, "wall_fill": WALL_FILL, "min_step": MIN_STEP,
            "native_layouts": "legal DP/TP/PP/EP and pipeline microbatch layouts; PP/microbatch candidates execute complete native windows",
            "interpolation": "linear TPOT upper bound between the adjacent measured points bracketing the SLO",
            "acceptance": "one fast native run at the best layout's interpolated batch; falls back to the best measured passing point",
            "beyond_native_reach": "largest measured point, marked limited, never extrapolated",
            "capacity": "bypass KV capacity until a native point reaches it, then native capacity rejections; a KV rejection's reported native capacity batch is measured next",
            "import_speed": "native_stationary_refill_not_always_ready is a too-big point; halving/bisection is reused and the limit is native_import_limited",
            "wall_projection": "measured wall without the Prefill pass, as measured, against the per-worker 1800s projection; no concurrency divisor"}
# The max-Batch mode is a separate fixed search contract. Layout scope is
# supplied independently, so the same mode can search all layouts or only PP.
MAX_BATCH_SETTINGS = {**THROUGHPUT_SETTINGS,
                       "search_mode": "max_batch_under_slo",
                       "objective": "maximum_batch_under_slo",
                       "max_points_per_layout": None,
                       "max_runs_per_layout": None,
                       "stopping": "SLO upper-bound bracket; physical capacity, memory and wall limits remain hard limits"}
SETTINGS = MAX_BATCH_SETTINGS


@dataclass(frozen=True)
class SearchPolicy:
    name: str
    settings: dict
    max_points: int | None
    max_runs: int | None
    objective: str


THROUGHPUT_POLICY = SearchPolicy("throughput", THROUGHPUT_SETTINGS, MAX_POINTS, MAX_RUNS,
                            "per_gpu_throughput")
MAX_BATCH_POLICY = SearchPolicy("max_batch_under_slo", MAX_BATCH_SETTINGS, None, None,
                                "maximum_batch_under_slo")
SEARCH_POLICIES = {THROUGHPUT_POLICY.name: THROUGHPUT_POLICY, MAX_BATCH_POLICY.name: MAX_BATCH_POLICY}
DEFAULT_SEARCH_MODE = MAX_BATCH_POLICY.name


def search_policy(mode=DEFAULT_SEARCH_MODE):
    try:
        return SEARCH_POLICIES[mode]
    except KeyError as error:
        raise ValueError("unknown search mode: " + str(mode)) from error
# Rejections that say the batch is too big for a native candidate: its wall
# cap, or its KV capacity.
_WALL_TOO_BIG = ("wall_budget",)
CAPACITY_REJECTIONS = ("not_a_full_batch", "import_pool_not_full")
IMPORT_REJECTIONS = ("native_stationary_refill_not_always_ready", "native_pipeline_refill_not_always_ready")
# These are forecast-quality failures rather than physical limits.  A later
# scan may use a better sampler or a repaired transfer/ingress estimator, so
# an old terminal row must be allowed one fresh native attempt.  Keep
# capacity, wall-clock, import-readiness and DAG-shape failures out of this
# set: those rows are useful evidence of a real upper bound or an invalid
# layout and must remain reusable as-is.
RETRYABLE_FORECAST_REJECTIONS = frozenset({
    "native_pipeline_independent_budget_exceeded",
    "native_pipeline_activation_transfer_sample_missing",
    "native_pipeline_ingress_ttft_unavailable",
    "native_pipeline_forecast_error_budget_exceeded",
    "native_prefill_independent_budget_exceeded",
    "native_prefill_pass_lower_bound_not_positive",
})
# A quality rejection is not evidence that the batch is physically too large.
# After the one cache retry, probe a few larger points so a layout can reach a
# stable sample at a useful batch.  The cap prevents a quality-only layout
# from becoming an unbounded scan; physical wall/capacity limits still stop
# through the ordinary ``too_big`` path.
QUALITY_REJECTIONS = frozenset({
    "native_pipeline_independent_budget_exceeded",
    "native_pipeline_forecast_error_budget_exceeded",
})
MAX_QUALITY_PROBES = 3


def is_pipeline_layout(layout):
    """Whether a layout uses the native PP/microbatch execution path."""
    return layout.get("pp", 1) > 1 or layout.get("microbatches", 1) > 1


def forecast_retryable(row):
    """Whether a saved forecast rejection should be sampled again once.

    The search cache intentionally retains terminal rows.  Budget and
    estimator-quality rejections are the exception: they do not establish a
    batch limit, and reusing them would prevent a repaired/relaxed forecast
    from ever reaching the next, larger batch point.
    """
    return (row.get("status") == "forecast_rejected"
            and row.get("rejection_reason") in RETRYABLE_FORECAST_REJECTIONS)


def screen_layouts(rows, buildable, top=None, *, policy=None, layout_scope="all", pp_only=None):
    """The ``top`` best bypass layouts a native candidate can run, and those skipped above them.

    PP and pipeline microbatches are native execution dimensions. The bypass
    only ranks them; complete native windows settle their measured metrics.
    ``buildable(layout, quantum)`` returns why the native layout cannot be
    built, else None.
    """
    policy = policy or MAX_BATCH_POLICY
    top = policy.settings.get("top_layouts", TOP_LAYOUTS) if top is None else top
    if pp_only is not None:
        layout_scope = "pipeline" if pp_only else "all"
    chosen, skipped, seen = [], [], set()
    for row in rows:
        p = row["parallelism"]
        layout = {key: p[key] for key in ("dp", "tp", "pp", "ep")}
        if p["microbatches"] != 1:
            layout["microbatches"] = p["microbatches"]
        if layout_scope == "pipeline" and not is_pipeline_layout(layout):
            continue
        key = tuple(layout.values())
        if key in seen:
            continue
        seen.add(key)
        reason = None if row["status"] == "screened_candidate" else "bypass_" + row["status"]
        reason = reason or buildable(layout, row["batch_quantum"])
        if reason:
            skipped.append({"layout": layout, "reason": reason})
            continue
        trials = [(trial["batch_size"], trial["tpot_seconds"] * 1e9) for trial in row["trials"]]
        chosen.append({"layout": layout, "quantum": row["batch_quantum"], "bypass": {
            "candidate_batch": row["candidate_batch"], "capacity_max_batch": row["capacity_max_batch"],
            "tpot_ns": row["screen_tpot_seconds"] * 1e9, "per_gpu_tokens_s": row["screen_per_gpu_tokens_s"],
            "tpot_line_ns": fit_line(trials)}})
        if len(chosen) == top:
            break
    return chosen, skipped


def fit_line(points):
    """Least-squares (intercept, slope) through (batch, value) points."""
    count = len(points)
    mean_b = sum(b for b, _ in points) / count
    mean_v = sum(v for _, v in points) / count
    spread = sum((b - mean_b) ** 2 for b, _ in points)
    slope = sum((b - mean_b) * (v - mean_v) for b, v in points) / spread if spread else 0.
    return mean_v - slope * mean_b, slope


def classify(row):
    """measured / quality_rejected / too_big / unusable / failed / pending."""
    status = row.get("status")
    if status in ("completed", "slo_rejected"):
        bounds = ((row.get("diagnostics") or {}).get("metric_bounds") or {}).get("mean_complete_request_tpot_ns")
        return "measured" if bounds else "unusable"
    if status == "capacity_rejected":
        return "too_big"
    if status == "forecast_rejected":
        reason = str(row.get("rejection_reason", ""))
        if reason in QUALITY_REJECTIONS:
            return "quality_rejected"
        return "too_big" if any(marker in reason for marker in _WALL_TOO_BIG + CAPACITY_REJECTIONS + IMPORT_REJECTIONS) else "unusable"
    return "failed" if status == "failed" else "pending"


def _point(batch, row):
    bounds = row["diagnostics"]["metric_bounds"]["mean_complete_request_tpot_ns"]
    # A measured-pool re-run projects from that attempt alone.
    wall = (row["native_retry_wall_seconds"] if "native_retry_wall_seconds" in row else
            max(0., (row.get("wall_seconds") or 0.) - (row.get("prefill_wall_seconds") or 0.)))
    return {"batch": batch, "tpot_upper_ns": bounds["upper"], "tpot_ns": bounds["estimate"],
            "status": row["status"], "native_wall_seconds": wall,
            "peak_memory_bytes": row.get("peak_memory_bytes")}


def _bracket(points, slo_ns):
    """Adjacent measured points whose TPOT upper bounds straddle the SLO, else None."""
    ordered = sorted(points, key=lambda point: point["batch"])
    for low, high in zip(ordered, ordered[1:]):
        if low["tpot_upper_ns"] <= slo_ns < high["tpot_upper_ns"]:
            return low, high
    return None


def _toward(line, slo_ns):
    """Batch where ``line`` reaches the SLO, or None without a rising line."""
    intercept, slope = line
    return (slo_ns - intercept) / slope if slope > 0 else None


def plan_layout(entry, lookup, slo_ns, wall_cap_seconds, memory_budget_bytes=None, *, policy=None):
    """Replay one layout's batch points; the next batch to run, or the layout's outcome.

    ``lookup(batch)`` returns the saved row at that batch, or None;
    ``memory_budget_bytes`` (None: unbounded) bounds a candidate's peak memory.
    """
    policy = policy or MAX_BATCH_POLICY
    caps = (wall_cap_seconds, memory_budget_bytes)
    q = entry["quantum"]

    def rounded(batch):
        return max(q, int(batch // q) * q)

    bypass = entry["bypass"]
    batch = rounded(min(bypass["candidate_batch"] / 2, FIRST_POINT_MAX_BATCH))
    runs = []
    while True:
        row = lookup(batch)
        kind = "pending" if row is None else classify(row)
        if kind == "pending":
            return {"next": batch, "runs": runs}
        runs.append({"batch": batch, "kind": kind, "status": row["status"],
                     "reason": row.get("rejection_reason"), "capacity": row.get("native_capacity_max_batch")})
        if kind == "failed":
            return {"runs": runs, "outcome": {"kind": "failed"}}
        points = [_point(run["batch"], lookup(run["batch"])) for run in runs if run["kind"] == "measured"]
        batch = _following(entry, runs, points, slo_ns, caps, rounded, policy=policy)
        if batch is None:
            return {"runs": runs, "points": points,
                    "outcome": _outcome(entry, runs, points, slo_ns, caps, rounded, policy=policy)}


def _limit(entry, runs, points, caps, rounded):
    """Largest batch a further native point may take, and what binds it."""
    wall_cap_seconds, memory_budget_bytes = caps
    limits = [(float("inf"), None)]
    big = [run for run in runs if run["kind"] == "too_big"]
    if big:
        smallest = min(big, key=lambda run: run["batch"])
        reason = str(smallest["reason"])
        wall = any(marker in reason for marker in _WALL_TOO_BIG)
        import_limited = any(marker in reason for marker in IMPORT_REJECTIONS)
        label = ("native_wall_limited" if wall else "native_import_limited"
                 if import_limited else "native_capacity_limited")
        limits.append((smallest["batch"] - 1, label))
    known = _native_capacities(runs)
    if known:
        limits.append((min(known), "native_capacity_limited"))
    capacity = entry["bypass"]["capacity_max_batch"]
    if not any(point["batch"] >= capacity for point in points):
        limits.append((capacity, "native_capacity_limited"))
    if points:
        largest = max(points, key=lambda point: point["batch"])
        if largest["native_wall_seconds"] > 0:
            limits.append((rounded(largest["batch"] * WALL_FILL * wall_cap_seconds / largest["native_wall_seconds"]),
                           "native_wall_limited"))
        peaks = [point for point in points if point["peak_memory_bytes"]]
        if memory_budget_bytes is not None and peaks:
            heaviest = max(peaks, key=lambda point: point["batch"])
            limits.append((rounded(heaviest["batch"] * memory_budget_bytes / heaviest["peak_memory_bytes"]),
                           "native_memory_limited"))
    return min(limits, key=lambda item: item[0])


def _following(entry, runs, points, slo_ns, caps, rounded, *, policy=MAX_BATCH_POLICY):
    """The next batch of a layout's search, or None when it is finished."""
    q = entry["quantum"]
    if _bracket(points, slo_ns):
        return None
    if ((policy.max_runs is not None and len(runs) >= policy.max_runs)
            or (policy.max_points is not None and len(points) >= policy.max_points)):
        return None
    last, known, tried = runs[-1], _native_capacities(runs), [run["batch"] for run in runs]
    if not points:
        if last["kind"] == "quality_rejected":
            # Forecast quality is independent of physical batch capacity. A
            # larger probe may have a steadier stage/transfer sample and can
            # then enter the ordinary measured-point search. Keep the probe
            # count finite and never cross the bypass physical capacity.
            quality_probes = sum(run["kind"] == "quality_rejected" for run in runs)
            if quality_probes >= MAX_QUALITY_PROBES:
                return None
            capacity = entry["bypass"]["capacity_max_batch"]
            target = max(q, last["batch"] * 2)
            if capacity is not None:
                target = min(target, capacity)
            batch = rounded(target)
            return batch if batch > last["batch"] and batch not in tried else None
        # Too big: a reported native capacity goes there directly; wall and
        # import-speed rejections use the existing halving path. An unusable
        # rejection ends the layout.
        if last["kind"] != "too_big" or last["batch"] <= q:
            return None
        batch = min(known) if known else rounded(last["batch"] / 2)
        return batch if batch >= q and batch not in tried else None
    limit, _reason = _limit(entry, runs, points, caps, rounded)
    if len(points) == 1:
        # The native/bypass TPOT ratio at the measured point scales the
        # bypass TPOT line to the SLO.
        point, line = points[0], entry["bypass"]["tpot_line_ns"]
        bypass_tpot = line[0] + line[1] * point["batch"]
        target = (_toward((line[0] * point["tpot_upper_ns"] / bypass_tpot,
                           line[1] * point["tpot_upper_ns"] / bypass_tpot), slo_ns) if bypass_tpot > 0 else None)
    else:
        ordered = sorted(points, key=lambda point: point["batch"])
        below = all(point["tpot_upper_ns"] <= slo_ns for point in ordered)
        pair = ordered[-2:] if below else ordered[:2]
        target = _toward(fit_line([(point["batch"], point["tpot_upper_ns"]) for point in pair]), slo_ns)
    below = all(point["tpot_upper_ns"] <= slo_ns for point in points)
    if target is None:
        target = max(point["batch"] for point in points) * 2 if below else q
    big = [run["batch"] for run in runs if run["kind"] == "too_big"]
    if below and big and target >= min(big) and not known:
        # Bisect between the largest measured point and the smallest too-big
        # batch; a reported native capacity is the limit itself.
        target = (max(point["batch"] for point in points) + min(big)) / 2
    batch = rounded(min(target, limit))
    # Spacing applies between measured points; a too-big batch only bounds
    # the limit, so the next point may sit right below it.
    measured = [point["batch"] for point in points]
    nearest = min(measured, key=lambda value: abs(value - batch))
    if abs(batch - nearest) < _step(nearest, q):
        # Too close to a measured batch: one step from it toward the SLO.
        batch = nearest + _step(nearest, q) if below else nearest - _step(nearest, q)
        batch = rounded(batch) if below else int(batch // q) * q
    if (batch < q or batch > limit or batch in tried
            or any(abs(batch - value) < _step(value, q) for value in measured)):
        return None
    return batch


def _native_capacities(runs):
    """Native capacity batches reported by KV capacity rejections."""
    return [run["capacity"] for run in runs if run["capacity"] is not None]


def _step(batch, q):
    return max(q, MIN_STEP * batch)


def _outcome(entry, runs, points, slo_ns, caps, rounded, *, policy=MAX_BATCH_POLICY):
    q = entry["quantum"]
    pair = _bracket(points, slo_ns)
    if pair:
        low, high = pair
        share = (slo_ns - low["tpot_upper_ns"]) / (high["tpot_upper_ns"] - low["tpot_upper_ns"])
        batch = rounded(low["batch"] + share * (high["batch"] - low["batch"]))
        tpot = low["tpot_ns"] + (high["tpot_ns"] - low["tpot_ns"]) * (batch - low["batch"]) / (high["batch"] - low["batch"])
        return {"kind": "interpolated", "batch": batch, "bracket": [low["batch"], high["batch"]],
                "predicted_tpot_ns": tpot}
    if not points:
        return {"kind": "no_native_measurement", "reason": runs[-1]["reason"] if runs else None}
    if all(point["tpot_upper_ns"] <= slo_ns for point in points):
        largest = max(points, key=lambda point: point["batch"])
        limit, reason = _limit(entry, runs, points, caps, rounded)
        # Before the run budget is spent a layout stops only where its limit
        # leaves no batch a step from every tried one.
        spent = ((policy.max_runs is not None and len(runs) >= policy.max_runs)
                 or (policy.max_points is not None and len(points) >= policy.max_points))
        if reason is None or (spent and limit - largest["batch"] >= _step(largest["batch"], q)):
            reason = "points_exhausted" if spent else "slo_not_reached"
        return {"kind": "limited", "reason": reason, "batch": largest["batch"]}
    if min(point["batch"] for point in points) == q:
        return {"kind": "slo_unreachable", "batch": q}
    return {"kind": "no_passing_point"}


def plan_case(entries, lookups, slo_ns, wall_cap_seconds, replica_gpus, memory_budget_bytes=None, *, policy=None):
    """Every layout's plan, the acceptance run, and the case's next native runs.

    ``lookups[i]`` reads the saved rows of ``entries[i]``'s layout and
    ``replica_gpus[i]`` its GPUs per replica.
    """
    policy = policy or MAX_BATCH_POLICY
    layouts = [{**entry, **plan_layout(entry, lookup, slo_ns, wall_cap_seconds, memory_budget_bytes,
                                        policy=policy)}
               for entry, lookup in zip(entries, lookups)]
    pending = [(plan["layout"], plan["next"]) for plan in layouts if "next" in plan]
    result = {"layouts": layouts, "next": pending, "acceptance": None}
    if pending or any(plan["outcome"]["kind"] == "failed" for plan in layouts):
        return result
    options = []
    max_batch = policy.objective == "maximum_batch_under_slo"
    for index, plan in enumerate(layouts):
        outcome = plan["outcome"]
        if outcome["kind"] == "interpolated":
            rate = outcome["batch"] * 1e9 / (outcome["predicted_tpot_ns"] * replica_gpus[index])
        elif outcome["kind"] == "limited":
            point = next(point for point in plan["points"] if point["batch"] == outcome["batch"])
            rate = point["batch"] * 1e9 / (point["tpot_ns"] * replica_gpus[index])
        else:
            continue
        # Max-Batch mode prefers the largest batch that remains within the SLO.
        # Throughput is a deterministic tie-breaker.
        options.append(((outcome["batch"], rate) if max_batch else (rate,), index))
    if not options:
        return result
    _objective, index = max(options)
    plan, outcome = layouts[index], layouts[index]["outcome"]
    if outcome["kind"] == "interpolated":
        rate = outcome["batch"] * 1e9 / (outcome["predicted_tpot_ns"] * replica_gpus[index])
    else:
        point = next(point for point in plan["points"] if point["batch"] == outcome["batch"])
        rate = point["batch"] * 1e9 / (point["tpot_ns"] * replica_gpus[index])
    acceptance = {"layout": plan["layout"], "batch": outcome["batch"], "kind": outcome["kind"],
                  "reason": outcome.get("reason"), "predicted_tokens_per_second_per_gpu": rate}
    if outcome["kind"] == "interpolated":
        row = lookups[index](outcome["batch"])
        if row is None or classify(row) == "pending":
            result["next"] = [(plan["layout"], outcome["batch"])]
        else:
            acceptance["status"] = row["status"]
    result["acceptance"] = acceptance
    return result


def search_result(plan, selected_batch, selected_layout):
    """Label of the case's selected point under its search plan."""
    acceptance = plan["acceptance"]
    if acceptance is None:
        return "best_measured_point"
    if acceptance["kind"] == "limited":
        return acceptance["reason"] if (acceptance["layout"] == selected_layout
                                        and acceptance["batch"] == selected_batch) else "best_measured_point"
    if acceptance.get("status") != "completed":
        return "acceptance_failed_fallback"
    return ("slo_accepted" if acceptance["layout"] == selected_layout and acceptance["batch"] == selected_batch
            else "best_measured_point")
