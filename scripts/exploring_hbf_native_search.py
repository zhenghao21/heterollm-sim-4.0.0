"""Paper-style deployment search over native fast candidates.

Per case the bypass screen (``exploring_hbf_search.search_candidates``) ranks
every legal layout and brackets its batch. The search maximises per-GPU
throughput under the 100 ms TPOT SLO; every layout of a case uses the same
GPUs, so per-GPU throughput is cluster throughput and does not favour fewer
DP replicas. A layout's batch search stops at the SLO bracket without a point
or attempt cap; its TPOT upper bound is interpolated to the SLO, and one
native acceptance run at the best layout's interpolated batch settles the
point. Bypass values only place batch points and bound layouts; every
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
native_import_limited. A failed acceptance falls back to the best measured
passing point.

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
# a layout is pruned by its bypass bound (367 native points: 1.03x to 7.8x;
# Maverick PP>1 1.83x and above).
BYPASS_TPOT_FLOOR = .8
# Beyond a measured point below the SLO, native TPOT is taken to grow at
# least this share of the bypass TPOT slope, times the case's least
# native/bypass slope ratio between measured points MIN_STEP apart when that
# is below 1 (PP>1 intercepts measured 1.8-3.8x the bypass line but slopes
# 1.05-1.35x; Maverick HBF PP1 slopes fell to 0.40x at large batches).
SLOPE_FLOOR = BYPASS_TPOT_FLOOR
OBJECTIVE = "per_gpu_throughput_under_slo"
SETTINGS = {"search_mode": "throughput_under_slo", "objective": OBJECTIVE, "top_layouts": TOP_LAYOUTS,
            "max_points_per_layout": None, "max_runs_per_layout": None,
            "first_point_max_batch": FIRST_POINT_MAX_BATCH, "wall_fill": WALL_FILL, "min_step": MIN_STEP,
            "rate_margin": RATE_MARGIN, "bypass_tpot_floor": BYPASS_TPOT_FLOOR, "slope_floor": SLOPE_FLOOR,
            "native_layouts": "legal DP/TP/PP/EP and pipeline microbatch layouts",
            "interpolation": "linear TPOT upper bound between the adjacent measured points bracketing the SLO",
            "acceptance": "one fast native run at the best layout's interpolated batch; falls back to the best measured passing point",
            "beyond_native_reach": "largest measured point, marked limited, never extrapolated",
            "capacity": "bypass KV capacity until a native point reaches it, then native capacity rejections; a KV rejection's reported native capacity batch is measured next",
            "import_speed": "native_stationary_refill_not_always_ready is a too-big point; halving/bisection is reused and the limit is native_import_limited",
            "wall_projection": "measured wall without the Prefill pass, as measured, against the per-worker 1800s projection; no concurrency divisor",
            "stopping": "SLO upper-bound bracket; physical capacity, memory and wall limits remain hard limits",
            "pruning": ("branch and bound in bypass rate-bound order: the anchor searches from its default first point; "
                        "later layouts first run where passing the SLO matches the anchor's best passing rate and are pruned "
                        "when the bypass rate bound (below any native limit, divided by bypass_tpot_floor) or a point above the SLO "
                        "(rate x (1 + rate_margin)) "
                        "falls below the case's best measured passing rate, or when a point below the SLO cannot reach it "
                        "with TPOT growing at slope_floor x min(1, the case's least native/bypass slope ratio) "
                        "x the bypass slope")}
# Rejections that say the batch is too big for a native candidate: its wall
# cap, or its KV capacity.
_WALL_TOO_BIG = ("wall_budget",)
CAPACITY_REJECTIONS = ("not_a_full_batch", "import_pool_not_full")
IMPORT_REJECTIONS = ("native_stationary_refill_not_always_ready",)


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
        reason = str(row.get("rejection_reason", ""))
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
    runs, points = [], []
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
            return {"runs": runs, "points": points,
                    "outcome": _outcome(entry, runs, points, slo_ns, caps, rounded)}


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


def _following(entry, runs, points, slo_ns, caps, rounded):
    """The next batch of a layout's search, or None when it is finished."""
    q = _quantum(entry)
    if _bracket(points, slo_ns):
        return None
    last, known, tried = runs[-1], _native_capacities(runs), [run["batch"] for run in runs]
    if not points:
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


def _outcome(entry, runs, points, slo_ns, caps, rounded):
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
        _limit_batch, reason = _limit(entry, runs, points, caps, rounded)
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


def slope_floor(plans):
    """SLOPE_FLOOR times the least native/bypass TPOT slope ratio between measured points of
    ``plans`` at least MIN_STEP apart, if below 1; None without such a pair."""
    ratios = []
    for plan in plans:
        slope = plan["bypass"]["tpot_line_ns"][1]
        points = sorted(plan.get("points") or (), key=lambda point: point["batch"])
        if slope > 0:
            ratios += [(high["tpot_ns"] - low["tpot_ns"]) / (high["batch"] - low["batch"]) / slope
                       for index, low in enumerate(points) for high in points[index + 1:]
                       if high["batch"] >= (1 + MIN_STEP) * low["batch"]]
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
    floor = slope_floor([layouts[index] for index in order[:settled]])
    for index in later:
        layouts[index] = plan(index, first=first(index),
                              prune=_pruner(entries[index], replica_gpus[index], slo_ns, anchor_rate, floor))
    rates = [_passing_rate(plan_, gpus, slo_ns) for plan_, gpus in zip(layouts, replica_gpus)]
    incumbent = max(rates, default=0.)
    holder = rates.index(incumbent) if count else None
    floor = slope_floor(layouts)
    for index in later:
        if index != holder:
            layouts[index] = plan(index, first=first(index),
                                  prune=_pruner(entries[index], replica_gpus[index], slo_ns, incumbent, floor))
        if layouts[index].get("outcome", {}).get("kind") == "pruned":
            layouts[index]["outcome"]["incumbent_tokens_per_second_per_gpu"] = incumbent
    pending = [(plan_["layout"], plan_["next"]) for plan_ in layouts if "next" in plan_]
    result = {"layouts": layouts, "next": pending, "acceptance": None,
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
    rate, index = max(options)
    chosen, outcome = layouts[index], layouts[index]["outcome"]
    acceptance = {"layout": chosen["layout"], "batch": outcome["batch"], "kind": outcome["kind"],
                  "reason": outcome.get("reason"), "predicted_tokens_per_second_per_gpu": rate}
    if outcome["kind"] == "interpolated":
        row = lookup(index)(outcome["batch"])
        if row is None or classify(row) == "pending":
            result["next"] = [(chosen["layout"], outcome["batch"])]
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
