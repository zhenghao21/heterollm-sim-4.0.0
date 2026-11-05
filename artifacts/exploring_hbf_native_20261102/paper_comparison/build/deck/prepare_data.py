"""Collect every number the lecture deck shows into deck_data.json (scan rows, paper readings, what-if runs)."""
import json
import math
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[4]
SCAN = ROOT / "artifacts/exploring_hbf_native_20261102"
SPLIT = ROOT / "tmp/wall_study_20261103/split_maverick"
PREVIOUS = ROOT / "artifacts/exploring_hbf_native_20261028_pp_stationary/paper_comparison/data.json"
HW = ("HBM4", "HBF", "HBF+", "CONV", "CONV+")
WL = ("SHORT", "MID", "LONG")
MODELS = ("llama3_405b", "llama4_maverick")
WEIGHTS = {"llama3_405b": 811.7e9, "llama4_maverick": 801e9}


def geomean(values):
    values = list(values)
    return math.exp(sum(map(math.log, values)) / len(values))


def bounds_of(row):
    metrics = row.get("metrics") or {}
    rejected = ((row.get("rejected_metric_bounds") or {}).get("metrics") or {}).get("mean_complete_request_tpot_ns") or {}
    measured = ((row.get("diagnostics") or {}).get("metric_bounds") or {}).get("mean_complete_request_tpot_ns") or {}
    tpot = metrics.get("mean_complete_request_tpot_ns") or rejected.get("estimate")
    return tpot, measured.get("lower") or rejected.get("lower") or tpot, measured.get("upper") or rejected.get("upper") or tpot


data = json.loads((SCAN / "paper_comparison/data.json").read_text(encoding="utf-8"))
rows = data["rows"]
key = {(r["model"], r["workload"], r["hardware"]): r for r in rows}

# Fig. 3 reading: the largest measured per-GPU batch within the SLO among all finished rows of a case,
# including the acceptance runs of the max-batch search (the HTML report only used throughput-search points).
fig3 = {}
for path in (SCAN / "summaries").glob("*.json"):
    row = json.loads(path.read_text(encoding="utf-8"))
    if "parameters" not in row or row.get("status") != "completed":
        continue
    parts = path.stem.split("__")
    case = (parts[1], parts[3], parts[2].replace("_plus", "+"))
    tpot, _, upper = bounds_of(row)
    if tpot is None or upper > 100e6:
        continue
    p = row["parameters"]
    per_gpu = p["batch_size"] * p["dp"] / 8
    if per_gpu > fig3.get(case, (0,))[0]:
        fig3[case] = (per_gpu, f"DP{p['dp']}/TP{p['tp']}/PP{p['pp']}/EP{p['ep']}", p["batch_size"], tpot / 1e6)

for r in rows:
    case = (r["model"], r["workload"], r["hardware"])
    kv_per_request = r["native_context_bytes_per_gpu"] / r["native_batch_per_gpu"]
    nominal = r["nominal_tb_s"] * 1e12
    single_copy = WEIGHTS[r["model"]] / 8
    paper_kv = r["paper_batch_per_gpu"] * kv_per_request
    paper_total = single_copy + paper_kv
    native_total = r["native_step_bytes_per_gpu"]
    native_kv = r["native_context_bytes_per_gpu"]
    # Single-copy estimate; the paper's layouts are not published, so this is a lower bound on its utilisation.
    r["paper_util"] = paper_total / (r["paper_tpot_ms"] / 1e3) / nominal
    r["kv_share_native"] = native_kv / native_total
    r["other_bytes_gb"] = (native_total - native_kv) / 1e9
    r["paper_total_gb"] = paper_total / 1e9
    r["fig3_batch"], r["fig3_layout"], r["fig3_b"], r["fig3_tpot"] = fig3[case]
for model in MODELS:
    for wl in WL:
        base = key[model, wl, "HBM4"]["fig3_batch"]
        for hw in HW:
            r = key[model, wl, hw]
            r["fig3_gain"] = r["fig3_batch"] / base


def span(group, field):
    return [min(r[field] for r in group), max(r[field] for r in group)]


def factors(group):
    total_ratio = [r["native_step_bytes_per_gpu"] / (WEIGHTS[r["model"]] / 8 + r["paper_batch_per_gpu"] * r["native_context_bytes_per_gpu"] / r["native_batch_per_gpu"]) for r in group]
    return {"count": len(group), "batch": geomean(r["batch_ratio"] for r in group), "tpot": geomean(r["tpot_ratio"] for r in group),
            "abs": geomean(r["abs_ratio"] for r in group), "batch_span": span(group, "batch_ratio"), "util_span": span(group, "native_bandwidth_fraction"),
            "tpot_ms_span": span(group, "native_tpot_ms"), "kv_share_span": span(group, "kv_share_native"), "tpot_span": span(group, "tpot_ratio"),
            "factor_span": span(group, "layout_byte_factor"), "total_ratio": geomean(total_ratio)}


slo = [r for r in rows if r["paper_bound"] == "slo"]
capacity = [r for r in rows if r["paper_bound"] == "capacity"]
# Mutually exclusive groups of the 23 SLO-bound points, by what limits the native batch.
dp4 = [r for r in slo if r["layout"]["dp"] == 4 and r["model"] == "llama3_405b"]
long_hbf = [r for r in slo if r["workload"] == "LONG" and r["hardware"] in ("HBF", "HBF+")]
conv = [r for r in slo if r["hardware"] in ("CONV", "CONV+")]
maverick_mid_hbf = [r for r in slo if r["model"] == "llama4_maverick" and r["workload"] == "MID" and r["hardware"] in ("HBF", "HBF+")]
maverick_short_hbf = [r for r in slo if r["model"] == "llama4_maverick" and r["workload"] == "SHORT" and r["hardware"] == "HBF"]
named = (("dp4", dp4), ("long_hbf", long_hbf), ("conv", conv), ("maverick_mid_hbf", maverick_mid_hbf), ("maverick_short_hbf", maverick_short_hbf))
assert sum(len(g) for _, g in named) == len(slo) == 23
groups = {name: factors(group) for name, group in named} | {"slo": factors(slo), "capacity": factors(capacity),
                                                           "hbm4_capacity": factors([r for r in capacity if r["hardware"] == "HBM4"])}
group_members = {name: [f"{r['model'][6:11]} {r['workload']} {r['hardware']}" for r in group] for name, group in named}
_x = [math.log(r["native_bandwidth_fraction"]) for r in slo]
_y = [math.log(r["batch_ratio"]) for r in slo]
_mx, _my = sum(_x) / len(_x), sum(_y) / len(_y)
util_batch_correlation = sum((a - _mx) * (b - _my) for a, b in zip(_x, _y)) / math.sqrt(sum((a - _mx) ** 2 for a in _x) * sum((b - _my) ** 2 for b in _y))
per_hardware = {hw: {"abs": geomean(r["abs_ratio"] for r in rows if r["hardware"] == hw), "batch": geomean(r["batch_ratio"] for r in rows if r["hardware"] == hw),
                     "tpot": geomean(r["tpot_ratio"] for r in rows if r["hardware"] == hw)} for hw in HW}

disagreements = []
for model in MODELS:
    for wl in WL:
        for a, b in combinations(HW, 2):
            ra, rb = key[model, wl, a], key[model, wl, b]
            if (ra["paper_norm_tps"] - rb["paper_norm_tps"]) * (ra["native_norm_tps"] - rb["native_norm_tps"]) <= 0:
                disagreements.append(dict(model=model, workload=wl, a=a, b=b, paper_a=ra["paper_norm_tps"], paper_b=rb["paper_norm_tps"],
                                          native_a=ra["native_norm_tps"], native_b=rb["native_norm_tps"]))


def order_with(substitute):
    agree = 0
    for model in MODELS:
        for wl in WL:
            for a, b in combinations(HW, 2):
                ra, rb = key[model, wl, a], key[model, wl, b]
                agree += (ra["paper_norm_tps"] - rb["paper_norm_tps"]) * (substitute.get((model, wl, a), ra["native_tps"]) - substitute.get((model, wl, b), rb["native_tps"])) > 0
    return agree


# Rejected rows whose estimate or optimistic end beats the selected point.
rejected_above = []
estimate_sub, optimistic_sub = {}, {}
wall_max = wall_rejected = 0
for path in (SCAN / "summaries").glob("*.json"):
    row = json.loads(path.read_text(encoding="utf-8"))
    if "parameters" not in row:
        continue
    wall_max = max(wall_max, row.get("wall_seconds") or 0)
    wall_rejected += "wall" in str(row.get("rejection_reason") or "")
    if row.get("status") != "forecast_rejected":
        continue
    parts = path.stem.split("__")
    case = (parts[1], parts[3], parts[2].replace("_plus", "+"))
    tpot, lower, upper = bounds_of(row)
    if tpot is None:
        continue
    p = row["parameters"]
    per_gpu = p["batch_size"] * p["dp"] / 8
    estimate, optimistic, selected = per_gpu / tpot * 1e9, per_gpu / lower * 1e9, key[case]["native_tps"]
    if tpot <= 100e6 and estimate > selected:
        estimate_sub[case] = max(estimate_sub.get(case, 0), estimate)
    if lower <= 100e6 and optimistic > selected:
        optimistic_sub[case] = max(optimistic_sub.get(case, 0), optimistic)
for case, value in estimate_sub.items():
    rejected_above.append(dict(case=case, estimate=value, selected=key[case]["native_tps"], gain=value / key[case]["native_tps"] - 1))

# TPOT against batch on the EP8 layout each investigated Maverick case ended on.
curves = {}
for name, tag, wl in (("mid_hbf", "HBF", "MID"), ("mid_hbf_plus", "HBF_plus", "MID"), ("short_hbf_plus", "HBF_plus", "SHORT")):
    case = json.loads((SCAN / "points" / f"native__llama4_maverick__{tag}__{wl}__G8.json").read_text(encoding="utf-8"))
    layout = next(entry for entry in case["search"]["layouts"] if entry["layout"] == {"dp": 1, "tp": 1, "pp": 1, "ep": 8})
    points = {p["batch"]: dict(batch=p["batch"], status=p["status"], tpot=p["tpot_ns"] / 1e6, lower=p["tpot_ns"] / 1e6, upper=p["tpot_upper_ns"] / 1e6, pd_lower=0, pd_upper=0)
              for p in layout["points"]}
    for path in (SCAN / "summaries").glob(f"native__llama4_maverick__{tag}__{wl}__G8__DP1_TP1_PP1_EP8*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if "parameters" not in row or row.get("status") == "paused":
            continue
        tpot, lower, upper = bounds_of(row)
        batch = row["parameters"]["batch_size"]
        if tpot is None:
            continue
        pd = ((row.get("approximation") or {}).get("error_budget") or {}).get("pd_import_extra_ns") or {}
        points[batch] = dict(batch=batch, status=row["status"], tpot=tpot / 1e6, lower=lower / 1e6, upper=upper / 1e6,
                             pd_lower=(pd.get("lower") or 0) / 1e6, pd_upper=(pd.get("upper") or 0) / 1e6)
    curves[name] = sorted(points.values(), key=lambda p: p["batch"])

whatifs = {}
for label, run in (("current", "A_t256_s40_b6048"), ("big_sram", "B_t256_big_b6048"), ("one_block", "C_t2048_s40_b6048"), ("both", "D_t2048_big_b6048"), ("no_program", "F_t256_big_prog1_b6048")):
    for path in (SPLIT / run).rglob("*.json"):
        if "summaries" not in path.as_posix():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if "parameters" in row:
            tpot, lower, upper = bounds_of(row)
            whatifs[label] = dict(status=row["status"], tpot=tpot / 1e6, lower=lower / 1e6, upper=upper / 1e6, rate=row["parameters"]["batch_size"] / 8 / tpot * 1e9)

previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))["summary"]
output = {
    "rows": rows, "summary": data["summary"], "candidate_counts": data["candidate_counts"], "total_candidates": data["total_candidates"],
    "paper_note": data["paper_note"], "previous": previous, "groups": groups, "group_members": group_members, "per_hardware": per_hardware,
    "all_factors": {"abs": geomean(r["abs_ratio"] for r in rows), "batch": geomean(r["batch_ratio"] for r in rows), "tpot": geomean(r["tpot_ratio"] for r in rows)},
    "disagreements": disagreements, "rejected_above": rejected_above,
    "order_estimate": order_with(estimate_sub), "order_optimistic": order_with(optimistic_sub),
    "wall_max": wall_max, "wall_rejected": wall_rejected, "curves": curves, "whatifs": whatifs,
    "util_batch_correlation": util_batch_correlation,
    "tps_identity_max_residual": max(abs(r["native_batch_per_gpu"] / r["native_tpot_ms"] * 1e3 / r["native_tps"] - 1) for r in rows),
}
(HERE / "deck_data.json").write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")

if "-v" in sys.argv:
    fmt = lambda v: [round(x, 2) for x in v] if isinstance(v, list) else round(v, 3)
    for name, g in groups.items():
        print(name, {k: fmt(v) for k, v in g.items()})
    print(group_members)
    print("util vs batch correlation over SLO points", round(util_batch_correlation, 3))
    print("per hardware", {h: {k: round(v, 3) for k, v in x.items()} for h, x in per_hardware.items()})
    print("all", output["all_factors"])
    print("order est/opt", output["order_estimate"], output["order_optimistic"], "wall", wall_max, wall_rejected)
    print("tps identity residual", output["tps_identity_max_residual"])
    print("rejected above", [(x["case"], round(x["estimate"]), round(x["selected"]), round(x["gain"], 3)) for x in rejected_above])
    print("optimistic", {k: round(v) for k, v in optimistic_sub.items()})
    for r in rows:
        if r["hardware"] == "HBM4" or (r["hardware"] == "CONV+" and r["workload"] == "MID"):
            print(r["model"][6:11], r["workload"], r["hardware"], "U %.2f paperU %.2f tpotR %.2f batchR %.2f U-ratio %.2f bytes %.0f/%.0f GB" % (
                r["native_bandwidth_fraction"], r["paper_util"], r["tpot_ratio"], r["batch_ratio"], r["paper_util"] / r["native_bandwidth_fraction"], r["native_step_bytes_per_gpu"] / 1e9, r["paper_total_gb"]))
    for name, c in curves.items():
        print(name, [(p["batch"], p["status"][:4], round(p["tpot"], 1), round(p["lower"], 1), round(p["upper"], 1), round(p["pd_lower"], 1), round(p["pd_upper"], 1)) for p in c])
    print(whatifs)
