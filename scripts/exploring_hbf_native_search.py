"""Paper-style deployment search over native fast candidates.

Per case the bypass screen (``exploring_hbf_search.search_candidates``) ranks
every legal layout and brackets its batch. The search maximises per-GPU
throughput under the 100 ms TPOT SLO; every layout of a case uses the same
GPUs, so per-GPU throughput is cluster throughput and does not favour fewer
DP replicas. A layout's batch search stops at the SLO bracket without a point
or attempt cap; its TPOT upper bound is interpolated to the SLO, and a
native acceptance run at the interpolated batch of the best layout, and of
every layout predicted within ACCEPTANCE_MARGIN of it, settles the point. Bypass values only place batch points and bound layouts; every
reported number is a native candidate row.

Layouts are searched by branch and bound in the order of their bypass rate
bound (the per-GPU rate the bypass TPOT line allows under the SLO; native
TPOT lies above that line). The first layout (the anchor; the next one while
an anchor ends without a passing point) is searched from its default first
point. Every later layout first runs at the batch where passing the SLO would
match the anchor's best passing rate, and is pruned once its bypass bound or a
point above the SLO shows it cannot beat the best passing point the case has
measured (the incumbent).

A layout whose 100 ms batch lies beyond what a native candidate can run (its
KV capacity, wall cap, import speed, or memory projected from its measured
points) stops at its largest measured point and is marked limited; it is never
extrapolated. The bypass KV capacity bounds the points until a native point
reaches it; past it native capacity rejections do. A KV capacity rejection
reports the largest batch the native KV holds, and the next point goes
straight there. A refill-not-ready rejection is also a too-big point: the
existing halving/bisection path is reused, and its final limit is labelled
native_import_limited. So is a forecast interval wider than its budget, labelled
native_interval_limited: the width does not grow steadily with the batch, so that
limit is a lower bound. A target at or past a too-big batch bisects under it from
the largest passing point below it, so a search never walks down one batch
quantum at a time. An acceptance run that proves too big joins these points; any
other failed acceptance falls back to the best measured passing point.

Once that is settled, a second branch and bound finds the paper's Fig. 3
metric, the largest per-GPU batch any layout runs within the SLO: the same
per-layout batch search, pruned when a layout cannot pass above the largest
passing per-GPU batch measured so far, with one acceptance run.

The plan is a pure function of the screen and the saved candidate rows, so a
stopped search resumes by replaying it; the incumbent only grows with saved
rows, so a replay prunes the same layouts or more.
"""
from __future__ import annotations

import math

# The native search is an exhaustive finite-layout search. Bypass screening
# ranks rows and supplies batch brackets; layouts leave only by a bound.
TOP_LAYOUTS = None
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
# Below a point above the SLO a layout's per-GPU rate may exceed that point's
# by this share (measured adjacent points dropped by at most 7.5%).
RATE_MARGIN = .1
# Native TPOT is taken to be at least this share of the bypass TPOT line when
# a layout is pruned by its bypass bound (791 native points of the 20261028
# scan: 0.996x and above, no layout's best rate above this bound; 174 points on
# the current memory physics: 1.016x and above).
BYPASS_TPOT_FLOOR = .95
# Beyond a measured point below the SLO, native TPOT is taken to grow at
# least this share of the bypass TPOT slope, times the case's least
# native/bypass slope ratio between measured points MIN_STEP apart when that
# is below 1 (PP>1 intercepts measured 1.8-3.8x the bypass line but slopes
# 1.05-1.35x; Maverick HBF PP1 slopes fell to 0.40x at large batches).
SLOPE_FLOOR = .8
# Every interpolated outcome whose predicted rate is within this share of the
# best outcome's gets an acceptance run. The linear TPOT line through the
# bracket never underestimated TPOT in 50 acceptance runs (scans 20261103 to
# 20261111: measured/predicted rate 0.998x to 1.127x, Maverick HBF MID 1.09x
# and 1.127x), so a layout predicted just below the best can still win.
ACCEPTANCE_MARGIN = .15
OBJECTIVE = "per_gpu_throughput_under_slo"
SETTINGS = {"search_mode": "throughput_under_slo", "objective": OBJECTIVE, "top_layouts": TOP_LAYOUTS,
            "max_points_per_layout": None, "max_runs_per_layout": None,
            "first_point_max_batch": FIRST_POINT_MAX_BATCH, "wall_fill": WALL_FILL, "min_step": MIN_STEP,
            "rate_margin": RATE_MARGIN, "bypass_tpot_floor": BYPASS_TPOT_FLOOR, "slope_floor": SLOPE_FLOOR,
            "native_layouts": "legal DP/TP/PP/EP and pipeline microbatch layouts",
            "interpolation": "linear TPOT upper bound between the adjacent measured points bracketing the SLO",
            "acceptance_margin": ACCEPTANCE_MARGIN,
            "acceptance": ("one fast native run at the interpolated batch of the best layout and of every layout predicted "
                           "within acceptance_margin of it; one above the SLO joins the bracket "
                           "and the batch is interpolated again, settling at the bracket's passing end within a step of a "
                           "measured batch; one that proves too big bisects under it; falls back to the best measured "
                           "passing point"),
            "no_rising_line": "double the largest point below the SLO, or halve the smallest above it",
            "beyond_native_reach": "largest measured point, marked limited, never extrapolated",
            "capacity": "bypass KV capacity until a native point reaches it, then native capacity rejections; a KV rejection's reported native capacity batch is measured next",
            "import_speed": "native_stationary_refill_not_always_ready is a too-big point; halving/bisection is reused and the limit is native_import_limited",
            "interval_width": "native_stationary_independent_budget_exceeded is a too-big point; halving/bisection is reused and the limit is native_interval_limited",
            "too_big": ("a target at or past the smallest too-big batch bisects from the largest passing point below it, "
                        "after any reported native capacity below that batch has been tried"),
            "wall_projection": "measured wall without the Prefill pass, as measured, against the candidate wall budget; no concurrency divisor",
            "wall_budget": ("6 h per candidate and Prefill pass, a guard against a hung worker that must not decide a result: "
                            "a budget change reuses saved rows, and a row killed under a smaller budget runs again"),
            "stopping": "SLO upper-bound bracket; physical capacity, memory and wall limits remain hard limits",
            "pruning": ("branch and bound in bypass rate-bound order: the anchor searches from its default first point; "
                        "later layouts first run where passing the SLO matches the anchor's best passing rate and are pruned "
                        "when the bypass rate bound (below any native limit, divided by bypass_tpot_floor) or a point above the SLO "
                        "(rate x (1 + rate_margin)) "
                        "falls below the case's best measured passing rate, or when a point below the SLO cannot reach it "
                        "with TPOT growing at slope_floor x min(1, the case's least native/bypass slope ratio from a point within the SLO) "
                        "x the bypass slope"),
            "max_batch": ("paper Fig. 3: after the throughput search, the largest per-GPU batch within the SLO by the "
                          "same branch and bound per layout (bypass batch bound with bypass_tpot_floor, a point above "
                          "the SLO times 1 + rate_margin, the slope floor), one acceptance run at the best layout's "
                          "interpolated batch, falling back to the largest measured passing point")}
# Rejections that say the batch is too big for a native candidate: its wall
# cap, or its KV capacity.
_WALL_TOO_BIG = ("wall_budget",)
CAPACITY_REJECTIONS = ("kv_block_peak_exceeds_capacity", "not_a_full_batch", "import_pool_not_full")
IMPORT_REJECTIONS = ("native_stationary_refill_not_always_ready",)
INTERVAL_REJECTIONS = ("native_stationary_independent_budget_exceeded",)
TOO_BIG_REJECTIONS = _WALL_TOO_BIG + CAPACITY_REJECTIONS + IMPORT_REJECTIONS + INTERVAL_REJECTIONS


def is_pipeline_layout(layout):
    """Whether a layout uses the native PP/microbatch execution path."""
    return layout.get("pp", 1) > 1 or layout.get("microbatches", 1) > 1


def screen_layouts(rows, buildable, top=TOP_LAYOUTS, *, layout_scope="all", pp_only=None):
    """The ``top`` best bypass layouts a native candidate can run, and those skipped above them.

    PP and pipeline microbatches are native execution dimensions. The bypass
    only ranks them; complete native windows settle their measured metrics.
    ``buildable(layout, quantum)`` returns why the native layout cannot be
    built, else None.
    """
    # ``all_legal`` is an explicit exhaustive mode.  The ordinary ``all``
    # scope keeps the historical top-layout screen so existing scans remain
    # bounded; exhaustive mode retains every buildable legal row.
    if layout_scope == "all_legal":
        top = None
    if pp_only is not None:
        layout_scope = "pipeline" if pp_only else "all"
    chosen, skipped, seen = [], [], set()
    for row in rows:
        p = row["parallelism"]
        layout = {key: p[key] for key in ("dp", "tp", "pp", "ep")}
        # Single-stage microbatching is not a separate deployment dimension in
        # this study. Normalize legacy PP1_M>1 rows to canonical PP1_M1;
        # real pipeline layouts retain their explicit microbatch degree.
        microbatches = 1 if p["pp"] == 1 else p["microbatches"]
        if microbatches != 1:
            layout["microbatches"] = microbatches
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
        if top is not None and len(chosen) == top:
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
        # The Prefill pass does not depend on the batch, so a smaller one cannot rescue it.
        if (row.get("process_watchdog") or {}).get("phase") == "prefill":
            return "unusable"
        reason = str(row.get("rejection_reason", ""))
        return "too_big" if any(marker in reason for marker in TOO_BIG_REJECTIONS) else "unusable"
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


def _quantum(entry):
    """The batch quantum: the screen's, and every EP group times pipeline microbatch."""
    # Older screen caches recorded only EP as the quantum. Native PP also
    # requires the per-replica batch to split across every pipeline
    # microbatch, or a probe creates an invalid candidate (B449 for EP1/M2).
    layout = entry.get("layout") or {}
    return max(int(entry["quantum"]), int(layout.get("ep", 1)) * int(layout.get("microbatches", 1)))


def _rounded(batch, q):
    return max(q, int(batch // q) * q)


def default_first_batch(entry):
    """An anchor's first point: half the bypass batch, at most FIRST_POINT_MAX_BATCH."""
    return _rounded(min(entry["bypass"]["candidate_batch"] / 2, FIRST_POINT_MAX_BATCH), _quantum(entry))


def plan_layout(entry, lookup, slo_ns, wall_cap_seconds, memory_budget_bytes=None, *, first=None, prune=None):
    """Replay one layout's batch points; the next batch to run, or the layout's outcome.

    ``lookup(batch)`` returns the saved row at that batch, or None;
    ``memory_budget_bytes`` (None: unbounded) bounds a candidate's peak memory.
    ``first`` (default ``default_first_batch``) is the first point;
    ``prune(runs, points)`` names why the layout cannot beat the incumbent, else None.
    """
    caps = (wall_cap_seconds, memory_budget_bytes)
    q = _quantum(entry)

    def rounded(batch):
        return _rounded(batch, q)

    batch = default_first_batch(entry) if first is None else first
    runs, points, refining = [], [], False
    while True:
        reason = prune and prune(runs, points)
        if reason:
            return {"runs": runs, "points": points, "outcome": {"kind": "pruned", "reason": reason}}
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
            outcome = _outcome(entry, runs, points, slo_ns, caps, rounded)
            batch = _refinement(outcome, lookup, points, slo_ns, q)
            if batch is None and refining:
                outcome = _refined_outcome(outcome, points, q)
            if batch is None:
                return {"runs": runs, "points": points, "outcome": outcome}
            refining = True


def _refinement(outcome, lookup, points, slo_ns, q):
    """The interpolated batch again, as a point, when its acceptance run exceeded the SLO or proved too big."""
    # A failed acceptance is a point above the SLO inside the bracket: it
    # narrows the bracket and the batch is interpolated again. A too-big one
    # caps the bracket, and the search bisects under it.
    if outcome["kind"] != "interpolated" or outcome["batch"] in [point["batch"] for point in points]:
        return None
    row = lookup(outcome["batch"])
    if row is None:
        return None
    kind = classify(row)
    if kind == "too_big":
        return outcome["batch"]
    if kind != "measured":
        return None
    return outcome["batch"] if _point(outcome["batch"], row)["tpot_upper_ns"] > slo_ns else None


def _refined_outcome(outcome, points, q):
    """A re-interpolated batch closer than a step to a measured one settles at the bracket's passing end."""
    if outcome["kind"] != "interpolated":
        return outcome
    if all(abs(outcome["batch"] - point["batch"]) >= _step(point["batch"], q) for point in points):
        return outcome
    low = next(point for point in points if point["batch"] == outcome["bracket"][0])
    return {**outcome, "batch": low["batch"], "predicted_tpot_ns": low["tpot_ns"], "settled": "bracket_passing_end"}


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
        interval_limited = any(marker in reason for marker in INTERVAL_REJECTIONS)
        label = ("native_wall_limited" if wall else "native_import_limited" if import_limited
                 else "native_interval_limited" if interval_limited else "native_capacity_limited")
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


def _smallest_too_big(runs):
    return min((run["batch"] for run in runs if run["kind"] == "too_big"), default=None)


def _interpolated(pair, slo_ns, rounded):
    """The batch where the TPOT upper bound's line through a bracket reaches the SLO."""
    low, high = pair
    share = (slo_ns - low["tpot_upper_ns"]) / (high["tpot_upper_ns"] - low["tpot_upper_ns"])
    return rounded(low["batch"] + share * (high["batch"] - low["batch"]))


def _following(entry, runs, points, slo_ns, caps, rounded):
    """The next batch of a layout's search, or None when it is finished."""
    q = _quantum(entry)
    last, known, tried = runs[-1], _native_capacities(runs), [run["batch"] for run in runs]
    big = _smallest_too_big(runs)
    pair = _bracket(points, slo_ns)
    if pair:
        if big is None or _interpolated(pair, slo_ns, rounded) < big:
            return None
        # The SLO lies at or past a too-big batch inside the bracket.
        low = pair[0]["batch"]
        batch = rounded((low + big) / 2)
        return batch if batch - low >= _step(low, q) and batch not in tried else None
    if not points:
        # Too big: a reported native capacity below it goes there directly,
        # down to the batch quantum; otherwise halve. An unusable rejection
        # ends the layout.
        if last["kind"] != "too_big" or last["batch"] <= q:
            return None
        batch = min(known) // q * q if known and min(known) < last["batch"] else rounded(last["batch"] / 2)
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
        # No rising line: double past points below the SLO, halve below
        # points above it (native TPOT need not rise between close batches).
        target = (max(point["batch"] for point in points) * 2 if below
                  else min(point["batch"] for point in points) / 2)
    if big is not None and target >= big and not (known and min(known) < big):
        # Bisect between the largest passing point below the smallest too-big
        # batch and that batch; a reported native capacity not yet tried is
        # the limit itself.
        target = (max((point["batch"] for point in points
                       if point["tpot_upper_ns"] <= slo_ns and point["batch"] < big), default=0) + big) / 2
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
    pair = _bracket(points, slo_ns)
    if pair:
        low, high = pair
        batch = _interpolated(pair, slo_ns, rounded)
        big = _smallest_too_big(runs)
        if big is not None and batch >= big:
            # No room left to bisect under a too-big batch inside the bracket.
            _limit_batch, reason = _limit(entry, runs, points, caps, rounded)
            return {"kind": "limited", "reason": reason or "slo_not_reached", "batch": low["batch"]}
        tpot = low["tpot_ns"] + (high["tpot_ns"] - low["tpot_ns"]) * (batch - low["batch"]) / (high["batch"] - low["batch"])
        return {"kind": "interpolated", "batch": batch, "bracket": [low["batch"], high["batch"]],
                "predicted_tpot_ns": tpot}
    if not points:
        return {"kind": "no_native_measurement", "reason": runs[-1]["reason"] if runs else None}
    if all(point["tpot_upper_ns"] <= slo_ns for point in points):
        largest = max(points, key=lambda point: point["batch"])
        limit, reason = _limit(entry, runs, points, caps, rounded)
        # A limit at least a step away did not end the search: the SLO is within
        # one step. The smallest limit is not the cause just because it is smallest.
        if limit - largest["batch"] >= _step(largest["batch"], _quantum(entry)) \
                and not any(run["kind"] == "too_big" for run in runs):
            reason = "native_slo_within_step"
        return {"kind": "limited", "reason": reason or "slo_not_reached", "batch": largest["batch"]}
    q = _quantum(entry)
    if min(point["batch"] for point in points) == q:
        return {"kind": "slo_unreachable", "batch": q}
    return {"kind": "no_passing_point"}


def bypass_rate_bound(entry, gpus, slo_ns, limit=None):
    """Highest per-GPU rate the layout's bypass TPOT line allows under the SLO and below ``limit``.

    Native TPOT lies above the bypass line (367 native points: 1.03x to 7.8x),
    and along the line the rate grows with the batch, so no native batch under
    the SLO runs faster; pruning still allows BYPASS_TPOT_FLOOR of the line. A
    line that does not rise from a positive intercept bounds nothing.
    """
    intercept, slope = entry["bypass"]["tpot_line_ns"]
    if intercept <= 0 or slope <= 0:
        return math.inf
    batch = (slo_ns - intercept) / slope
    if limit is not None:
        batch = min(batch, limit)
    return batch * 1e9 / (gpus * (intercept + slope * batch)) if batch > 0 else 0.


def _point_rate(point, gpus):
    return point["batch"] * 1e9 / (point["tpot_ns"] * gpus)


def _passing_rate(plan, gpus, slo_ns):
    """The best per-GPU rate among a layout's measured points within the SLO, else 0."""
    return max([_point_rate(point, gpus) for point in plan.get("points") or ()
                if point["tpot_upper_ns"] <= slo_ns], default=0.)


def slope_floor(plans, slo_ns):
    """SLOPE_FLOOR times the least native/bypass TPOT slope ratio from a measured point of
    ``plans`` within the SLO to one at least MIN_STEP larger, if below 1; None without such a pair."""
    # The bound extrapolates only from points within the SLO, so only pairs
    # starting there calibrate it: native TPOT can fall between two batches
    # above the SLO (Maverick HBF+ MID EP8: 130.9 ms at B4160, 113.8 ms at B4768).
    ratios = []
    for plan in plans:
        slope = plan["bypass"]["tpot_line_ns"][1]
        points = sorted(plan.get("points") or (), key=lambda point: point["batch"])
        if slope > 0:
            ratios += [(high["tpot_ns"] - low["tpot_ns"]) / (high["batch"] - low["batch"]) / slope
                       for index, low in enumerate(points) for high in points[index + 1:]
                       if low["tpot_upper_ns"] <= slo_ns and high["batch"] >= (1 + MIN_STEP) * low["batch"]]
    return SLOPE_FLOOR * min(1., *ratios) if ratios else None


def slope_rate_bound(entry, point, gpus, slo_ns, floor):
    """Highest per-GPU rate under the SLO when TPOT grows beyond ``point`` at least ``floor`` times
    the bypass TPOT slope; below the point the rate stays within RATE_MARGIN of the point's."""
    slope = floor * entry["bypass"]["tpot_line_ns"][1]
    if slope <= 0:
        return math.inf
    # Along that floor line the rate peaks where it reaches the SLO, or at
    # the point itself when the line has no positive intercept.
    batch = point["batch"] + (slo_ns - point["tpot_ns"]) / slope
    return max(batch * 1e9 / (gpus * slo_ns), _point_rate(point, gpus) * (1 + RATE_MARGIN))


def _pruner(entry, gpus, slo_ns, incumbent, floor):
    """Why a layout cannot beat ``incumbent`` given its runs and points, else None.

    ``floor`` (None: no slope bound) is the case's ``slope_floor``.
    """
    def prune(runs, points):
        limits = _native_capacities(runs) + [run["batch"] - 1 for run in runs if run["kind"] == "too_big"]
        if bypass_rate_bound(entry, gpus, slo_ns, min(limits, default=None)) / BYPASS_TPOT_FLOOR < incumbent:
            return "bypass_rate_bound_below_incumbent"
        # Every batch under the SLO lies below such a point (TPOT grows with
        # the batch), where the rate is at most RATE_MARGIN above the point's.
        if any(point["tpot_upper_ns"] > slo_ns and _point_rate(point, gpus) * (1 + RATE_MARGIN) < incumbent
               for point in points):
            return "native_point_above_slo_below_incumbent"
        if floor is not None and any(point["tpot_upper_ns"] <= slo_ns
                                     and slope_rate_bound(entry, point, gpus, slo_ns, floor) < incumbent
                                     for point in points):
            return "native_slope_bound_below_incumbent"
        return None
    return prune


def bypass_batch_bound(entry, slo_ns, limit=None):
    """Largest batch under the SLO when native TPOT is at least BYPASS_TPOT_FLOOR of the
    layout's bypass TPOT line, and at most ``limit``; a line that does not rise from a positive
    intercept bounds nothing."""
    intercept, slope = entry["bypass"]["tpot_line_ns"]
    batch = (slo_ns / BYPASS_TPOT_FLOOR - intercept) / slope if intercept > 0 and slope > 0 else math.inf
    return batch if limit is None else min(batch, limit)


def slope_batch_bound(entry, point, slo_ns, floor):
    """Largest batch under the SLO when TPOT grows beyond ``point`` at least ``floor`` times the bypass slope."""
    slope = floor * entry["bypass"]["tpot_line_ns"][1]
    return point["batch"] + (slo_ns - point["tpot_ns"]) / slope if slope > 0 else math.inf


def _passing_batch(plan, gpus, slo_ns):
    """The largest per-GPU batch among a layout's measured points within the SLO, else 0."""
    return max([point["batch"] / gpus for point in plan.get("points") or ()
                if point["tpot_upper_ns"] <= slo_ns], default=0.)


def _batch_pruner(entry, gpus, slo_ns, incumbent, floor):
    """Why a layout cannot pass the SLO above ``incumbent`` requests per GPU, else None."""
    def prune(runs, points):
        limits = _native_capacities(runs) + [run["batch"] - 1 for run in runs if run["kind"] == "too_big"]
        capacity = entry["bypass"]["capacity_max_batch"]
        if not any(point["batch"] >= capacity for point in points):
            # The search measures no batch above the bypass KV capacity
            # until a native point reaches it (``_limit``).
            limits.append(capacity)
        if bypass_batch_bound(entry, slo_ns, min(limits)) / gpus <= incumbent:
            return "bypass_batch_bound_not_above_incumbent"
        if any(point["tpot_upper_ns"] > slo_ns and point["batch"] * (1 + RATE_MARGIN) / gpus <= incumbent
               for point in points):
            return "native_point_above_slo_not_above_incumbent"
        if floor is not None and any(point["tpot_upper_ns"] <= slo_ns
                                     and slope_batch_bound(entry, point, slo_ns, floor) / gpus <= incumbent
                                     for point in points):
            return "native_slope_bound_not_above_incumbent"
        return None
    return prune


def _max_batch(entries, plan, firsts, lookup, replica_gpus, slo_ns, measured):
    """Paper Fig. 3: the largest per-GPU batch a layout runs within the SLO, by branch and bound.

    Runs after the throughput search; ``measured`` are its layout plans, whose
    passing points seed the incumbent. Each layout replays its saved rows from
    the same first point, pruned once it cannot pass the SLO above the
    incumbent; the best layout's interpolated batch gets one acceptance run.
    """
    count = len(entries)

    def incumbent_of(plans):
        return max([_passing_batch(plan_, gpus, slo_ns) for plan_, gpus in zip(plans, replica_gpus)], default=0.)

    def planned(incumbent, floor, keep=None):
        return [keep[index] if keep is not None and index == holder else
                plan(index, first=firsts[index],
                     prune=_batch_pruner(entries[index], replica_gpus[index], slo_ns, incumbent, floor))
                for index in range(count)]

    holder = None
    incumbent = incumbent_of(measured)
    layouts = planned(incumbent, slope_floor(measured, slo_ns))
    # The best passing batch any layout reaches then prunes every other layout.
    batches = [_passing_batch(plan_, gpus, slo_ns) for plan_, gpus in zip(layouts, replica_gpus)]
    if max(batches, default=0.) > incumbent:
        incumbent = max(batches)
        holder = batches.index(incumbent)
        layouts = planned(incumbent, slope_floor([*measured, *layouts], slo_ns), layouts)
    result = {"layouts": [{key: plan_.get(key) for key in ("layout", "runs", "points", "outcome", "next")}
                          for plan_ in layouts],
              "next": [(plan_["layout"], plan_["next"]) for plan_ in layouts if "next" in plan_],
              "incumbent_batch_per_gpu": incumbent, "acceptance": None, "selected": None}
    if result["next"] or any(plan_["outcome"]["kind"] == "failed" for plan_ in layouts):
        return result
    # A passing point on a layout the bound pruned can exceed every remaining outcome.
    options = [(plan_["outcome"]["batch"] / replica_gpus[index], index) for index, plan_ in enumerate(layouts)
               if plan_["outcome"]["kind"] in ("interpolated", "limited")
               and plan_["outcome"]["batch"] / replica_gpus[index] >= incumbent]
    if options:
        per_gpu, index = max(options)
        chosen, outcome = layouts[index], layouts[index]["outcome"]
        acceptance = {"layout": chosen["layout"], "batch": outcome["batch"], "batch_per_gpu": per_gpu,
                      "kind": outcome["kind"], "reason": outcome.get("reason"), "settled": outcome.get("settled")}
        if outcome["kind"] == "interpolated":
            row = lookup(index)(outcome["batch"])
            if row is None or classify(row) == "pending":
                result["next"] = [(chosen["layout"], outcome["batch"])]
                result["acceptance"] = acceptance
                return result
            acceptance["status"] = row["status"]
        result["acceptance"] = acceptance
        if outcome["kind"] == "limited" or acceptance["status"] == "completed":
            label = (outcome.get("reason") if outcome["kind"] == "limited"
                     else outcome.get("settled") or "slo_accepted")
            result["selected"] = {"layout": chosen["layout"], "batch": outcome["batch"],
                                  "batch_per_gpu": per_gpu, "result": label}
            return result
    # No such outcome, or a failed acceptance: the largest measured passing point.
    best = max(((point["batch"] / replica_gpus[index], index, point["batch"])
                for plans in (measured, layouts) for index, plan_ in enumerate(plans)
                for point in plan_.get("points") or () if point["tpot_upper_ns"] <= slo_ns), default=None)
    if best is not None:
        result["selected"] = {"layout": entries[best[1]]["layout"], "batch": best[2], "batch_per_gpu": best[0],
                              "result": "best_measured_point"}
    return result


def _outcome_rate(plan, gpus):
    outcome = plan["outcome"]
    if outcome["kind"] == "interpolated":
        return outcome["batch"] * 1e9 / (outcome["predicted_tpot_ns"] * gpus)
    point = next(point for point in plan["points"] if point["batch"] == outcome["batch"])
    return _point_rate(point, gpus)


def plan_case(entries, lookups, slo_ns, wall_cap_seconds, replica_gpus, memory_budget_bytes=None):
    """Every layout's plan (branch and bound), the acceptance run, and the case's next native runs.

    ``lookups[i]`` reads the saved rows of ``entries[i]``'s layout and
    ``replica_gpus[i]`` its GPUs per replica.
    """
    count = len(entries)
    cached = [{} for _ in range(count)]

    def lookup(index):
        def read(batch):
            if batch not in cached[index]:
                cached[index][batch] = lookups[index](batch)
            return cached[index][batch]
        return read

    def plan(index, **options):
        return {**entries[index], **plan_layout(entries[index], lookup(index), slo_ns, wall_cap_seconds,
                                                memory_budget_bytes, **options)}

    bounds = [bypass_rate_bound(entry, gpus, slo_ns) for entry, gpus in zip(entries, replica_gpus)]
    order = sorted(range(count), key=lambda index: -bounds[index])
    layouts, anchor_rate, settled = [None] * count, 0., 0
    for index in order:
        layouts[index] = plan(index)
        settled += 1
        anchor_rate = _passing_rate(layouts[index], replica_gpus[index], slo_ns)
        if "next" in layouts[index] or anchor_rate > 0:
            break
    later = order[settled:]
    if count and "next" in layouts[order[settled - 1]]:
        # Later layouts wait for the anchor's rate.
        for index in later:
            layouts[index] = {**entries[index], "runs": [], "waiting": "anchor"}
        later = []

    def first(index):
        # Where passing the SLO matches the anchor's rate, within the first
        # point's wall cap (a point above the SLO below it prunes even more
        # surely); a layout that ran only its default first point before an
        # anchor existed replays it.
        q = _quantum(entries[index])
        threshold = min(q * math.ceil(anchor_rate * replica_gpus[index] * slo_ns / 1e9 / q),
                        _rounded(FIRST_POINT_MAX_BATCH, q))
        default = default_first_batch(entries[index])
        if lookup(index)(threshold) is None and lookup(index)(default) is not None:
            return default
        return threshold

    # The anchor's rate and slope floor prune on a first pass; the best
    # passing rate any layout reaches, with the slope floor of every measured
    # layout, then prunes every later layout but the one that holds it.
    floor = slope_floor([layouts[index] for index in order[:settled]], slo_ns)
    for index in later:
        layouts[index] = plan(index, first=first(index),
                              prune=_pruner(entries[index], replica_gpus[index], slo_ns, anchor_rate, floor))
    rates = [_passing_rate(plan_, gpus, slo_ns) for plan_, gpus in zip(layouts, replica_gpus)]
    incumbent = max(rates, default=0.)
    holder = rates.index(incumbent) if count else None
    floor = slope_floor(layouts, slo_ns)
    for index in later:
        if index != holder:
            layouts[index] = plan(index, first=first(index),
                                  prune=_pruner(entries[index], replica_gpus[index], slo_ns, incumbent, floor))
        if layouts[index].get("outcome", {}).get("kind") == "pruned":
            layouts[index]["outcome"]["incumbent_tokens_per_second_per_gpu"] = incumbent
    pending = [(plan_["layout"], plan_["next"]) for plan_ in layouts if "next" in plan_]
    result = {"layouts": layouts, "next": pending, "acceptance": None, "acceptances": [], "max_batch": None,
              "bounds": {"order": order,
                         # None: a bypass line that bounds nothing.
                         "bypass_rate_bounds": [None if math.isinf(bound) else bound for bound in bounds],
                         "anchor_tokens_per_second_per_gpu": anchor_rate,
                         "incumbent_tokens_per_second_per_gpu": incumbent,
                         "slope_floor": floor}}
    if pending or any("waiting" in plan_ or plan_["outcome"]["kind"] == "failed" for plan_ in layouts):
        return result
    options = [(_outcome_rate(plan_, replica_gpus[index]), index) for index, plan_ in enumerate(layouts)
               if plan_["outcome"]["kind"] in ("interpolated", "limited")]
    if not options:
        return result
    best = max(options)[0]
    # Every interpolated outcome within ACCEPTANCE_MARGIN of the best one gets
    # its acceptance run: the prediction only ever underestimates the rate.
    acceptances = []
    for rate, index in sorted(options, reverse=True):
        outcome = layouts[index]["outcome"]
        if (rate, index) != max(options) and (outcome["kind"] != "interpolated"
                                              or rate < (1 - ACCEPTANCE_MARGIN) * best):
            continue
        acceptance = {"layout": layouts[index]["layout"], "batch": outcome["batch"], "kind": outcome["kind"],
                      "reason": outcome.get("reason"), "predicted_tokens_per_second_per_gpu": rate,
                      "settled": outcome.get("settled")}
        if outcome["kind"] == "interpolated":
            row = lookup(index)(outcome["batch"])
            if row is None or classify(row) == "pending":
                result["next"].append((layouts[index]["layout"], outcome["batch"]))
            else:
                acceptance["status"] = row["status"]
        acceptances.append(acceptance)
    result["acceptance"], result["acceptances"] = acceptances[0], acceptances
    if not result["next"]:
        # The throughput search is settled; the largest batch is searched next.
        firsts = [first(index) if index in later else None for index in range(count)]
        result["max_batch"] = _max_batch(entries, plan, firsts, lookup, replica_gpus, slo_ns, layouts)
        result["next"] = result["max_batch"]["next"]
    return result


def search_result(plan, selected_batch, selected_layout):
    """Label of the case's selected point under its search plan."""
    acceptances = plan["acceptances"]
    if not acceptances:
        return "best_measured_point"
    match = next((acceptance for acceptance in acceptances
                  if acceptance["layout"] == selected_layout and acceptance["batch"] == selected_batch), None)
    if match is None:
        best = acceptances[0]
        return ("acceptance_failed_fallback" if best["kind"] == "interpolated" and best.get("status") != "completed"
                else "best_measured_point")
    if match["kind"] == "limited":
        return match["reason"]
    return match["settled"] or "slo_accepted"
