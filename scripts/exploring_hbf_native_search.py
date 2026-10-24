"""Paper-style deployment search over native fast candidates.

Per case the bypass screen (``exploring_hbf_search.search_candidates``) ranks
every legal layout and brackets its batch. The two best layouts a native
stationary candidate can run are measured natively at two or three batch
points, the TPOT upper bound is interpolated to the 100 ms SLO, and one
native acceptance run at the best layout's interpolated batch settles the
point. Bypass values only place batch points; every reported number is a
native candidate row.

A layout whose 100 ms batch lies beyond what a native candidate can run (its
KV capacity, or the wall cap or memory projected from its measured points)
stops at its largest measured point and is marked limited; it is never
extrapolated. The bypass KV capacity bounds the points until a native point
reaches it; past it native capacity rejections do. A KV capacity rejection
reports the largest batch the native KV holds, and the next point goes
straight there. A failed acceptance falls back to the best measured passing
point.

The plan is a pure function of the screen and the saved candidate rows, so a
stopped search resumes by replaying it.
"""
from __future__ import annotations

TOP_LAYOUTS = 2
# Measured batch points per layout, and native runs per layout including
# attempts that were too big to measure.
MAX_POINTS = 3
MAX_RUNS = 5
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
SETTINGS = {"top_layouts": TOP_LAYOUTS, "max_points_per_layout": MAX_POINTS, "max_runs_per_layout": MAX_RUNS,
            "first_point_max_batch": FIRST_POINT_MAX_BATCH, "wall_fill": WALL_FILL, "min_step": MIN_STEP,
            "native_layouts": "bypass rows without microbatching and with pp == 1 that the native layout accepts",
            "interpolation": "linear TPOT upper bound between the adjacent measured points bracketing the SLO",
            "acceptance": "one fast native run at the best layout's interpolated batch; falls back to the best measured passing point",
            "beyond_native_reach": "largest measured point, marked limited, never extrapolated",
            "capacity": "bypass KV capacity until a native point reaches it, then native capacity rejections; a KV rejection's reported native capacity batch is measured next",
            "wall_projection": "measured wall without the Prefill pass, as measured, against the per-worker cap; the k-fold cap of k concurrent workers only absorbs their slowdown"}
# Rejections that say the batch is too big for a native candidate: its wall
# cap, or its KV capacity.
_WALL_TOO_BIG = ("wall_budget",)
CAPACITY_REJECTIONS = ("not_a_full_batch", "import_pool_not_full")


def screen_layouts(rows, buildable, top=TOP_LAYOUTS):
    """The ``top`` best bypass layouts a native candidate can run, and those skipped above them.

    Native serving has no GPipe microbatching, so only unpartitioned bypass
    rows rank, and the stationary model does not compose pipeline stages.
    ``buildable(layout, quantum)`` returns why the native layout cannot be
    built, else None.
    """
    chosen, skipped, seen = [], [], set()
    for row in rows:
        p = row["parallelism"]
        layout = {key: p[key] for key in ("dp", "tp", "pp", "ep")}
        key = tuple(layout.values())
        if p["microbatches"] != 1 or key in seen:
            continue
        seen.add(key)
        reason = None if row["status"] == "screened_candidate" else "bypass_" + row["status"]
        if reason is None and p["pp"] > 1:
            reason = "native_stationary_does_not_compose_pipeline_stages"
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
    """measured / too_big / unusable / failed / pending for one saved candidate row."""
    status = row.get("status")
    if status in ("completed", "slo_rejected"):
        bounds = ((row.get("diagnostics") or {}).get("metric_bounds") or {}).get("mean_complete_request_tpot_ns")
        return "measured" if bounds else "unusable"
    if status == "capacity_rejected":
        return "too_big"
    if status == "forecast_rejected":
        reason = str(row.get("rejection_reason", ""))
        return "too_big" if any(marker in reason for marker in _WALL_TOO_BIG + CAPACITY_REJECTIONS) else "unusable"
    return "failed" if status == "failed" else "pending"


def _point(batch, row):
    bounds = row["diagnostics"]["metric_bounds"]["mean_complete_request_tpot_ns"]
    wall = max(0., (row.get("wall_seconds") or 0.) - (row.get("prefill_wall_seconds") or 0.))
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


def plan_layout(entry, lookup, slo_ns, wall_cap_seconds, memory_budget_bytes=None):
    """Replay one layout's batch points; the next batch to run, or the layout's outcome.

    ``lookup(batch)`` returns the saved row at that batch, or None;
    ``memory_budget_bytes`` (None: unbounded) bounds a candidate's peak memory.
    """
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
        batch = _following(entry, runs, points, slo_ns, caps, rounded)
        if batch is None:
            return {"runs": runs, "points": points, "outcome": _outcome(entry, runs, points, slo_ns, caps, rounded)}


def _limit(entry, runs, points, caps, rounded):
    """Largest batch a further native point may take, and what binds it."""
    wall_cap_seconds, memory_budget_bytes = caps
    limits = [(float("inf"), None)]
    big = [run for run in runs if run["kind"] == "too_big"]
    if big:
        smallest = min(big, key=lambda run: run["batch"])
        wall = any(marker in str(smallest["reason"]) for marker in _WALL_TOO_BIG)
        limits.append((smallest["batch"] - 1, "native_wall_limited" if wall else "native_capacity_limited"))
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


def _following(entry, runs, points, slo_ns, caps, rounded):
    """The next batch of a layout's search, or None when it is finished."""
    q = entry["quantum"]
    if len(runs) >= MAX_RUNS or len(points) >= MAX_POINTS or _bracket(points, slo_ns):
        return None
    last, known, tried = runs[-1], _native_capacities(runs), [run["batch"] for run in runs]
    if not points:
        # Too big: the native capacity it reported, else half; an unusable
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


def _outcome(entry, runs, points, slo_ns, caps, rounded):
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
        spent = len(runs) >= MAX_RUNS or len(points) >= MAX_POINTS
        if reason is None or (spent and limit - largest["batch"] >= _step(largest["batch"], q)):
            reason = "points_exhausted"
        return {"kind": "limited", "reason": reason, "batch": largest["batch"]}
    if min(point["batch"] for point in points) == q:
        return {"kind": "slo_unreachable", "batch": q}
    return {"kind": "no_passing_point"}


def plan_case(entries, lookups, slo_ns, wall_cap_seconds, replica_gpus, memory_budget_bytes=None):
    """Every layout's plan, the acceptance run, and the case's next native runs.

    ``lookups[i]`` reads the saved rows of ``entries[i]``'s layout and
    ``replica_gpus[i]`` its GPUs per replica.
    """
    layouts = [{**entry, **plan_layout(entry, lookup, slo_ns, wall_cap_seconds, memory_budget_bytes)}
               for entry, lookup in zip(entries, lookups)]
    pending = [(plan["layout"], plan["next"]) for plan in layouts if "next" in plan]
    result = {"layouts": layouts, "next": pending, "acceptance": None}
    if pending or any(plan["outcome"]["kind"] == "failed" for plan in layouts):
        return result
    options = []
    for index, plan in enumerate(layouts):
        outcome = plan["outcome"]
        if outcome["kind"] == "interpolated":
            rate = outcome["batch"] * 1e9 / (outcome["predicted_tpot_ns"] * replica_gpus[index])
        elif outcome["kind"] == "limited":
            point = next(point for point in plan["points"] if point["batch"] == outcome["batch"])
            rate = point["batch"] * 1e9 / (point["tpot_ns"] * replica_gpus[index])
        else:
            continue
        options.append((rate, index))
    if not options:
        return result
    rate, index = max(options)
    plan, outcome = layouts[index], layouts[index]["outcome"]
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
