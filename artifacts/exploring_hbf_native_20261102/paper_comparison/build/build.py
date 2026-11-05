"""Build the paper-comparison page from data.json (collect.py + bandwidth.py)."""
import json, math, shutil, sys
from itertools import combinations
from pathlib import Path

here = Path(__file__).parent
out = Path(sys.argv[1])
data = json.loads((here / "data.json").read_text(encoding="utf-8"))
W = {"llama3_405b": 811.7e9, "llama4_maverick": 801e9}
HW = ("HBM4", "HBF", "HBF+", "CONV", "CONV+")
WL = ("SHORT", "MID", "LONG")
for r in data["rows"]:
    r["paper_tpot_ms"] = 1e3 * r["paper_batch_per_gpu"] / r["paper_tps"]
    r["paper_bound"] = "slo" if r["paper_tpot_ms"] >= 90 else "capacity"
    r["abs_ratio"] = r["native_tps"] / r["paper_tps"]
    r["batch_ratio"] = r["native_batch_per_gpu"] / r["paper_batch_per_gpu"]
    r["tpot_ratio"] = r["native_tpot_ms"] / r["paper_tpot_ms"]
    ctx = r["native_context_bytes_per_gpu"] / r["native_batch_per_gpu"]
    r["single_copy_bytes_per_gpu"] = W[r["model"]] / 8 + r["native_batch_per_gpu"] * ctx
    r["layout_byte_factor"] = r["native_step_bytes_per_gpu"] / r["single_copy_bytes_per_gpu"]

rows = data["rows"]
key = {(r["model"], r["workload"], r["hardware"]): r for r in rows}
other = [r for r in rows if r["hardware"] != "HBM4"]
log_ratio = [math.log(r["native_norm_tps"] / r["paper_norm_tps"]) for r in other]
agree = total = 0
for m in W:
    for wl in WL:
        for a, b in combinations(HW, 2):
            pa, pb = key[m, wl, a], key[m, wl, b]
            total += 1
            agree += (pa["paper_norm_tps"] - pb["paper_norm_tps"]) * (pa["native_norm_tps"] - pb["native_norm_tps"]) > 0
slo = [r for r in rows if r["paper_bound"] == "slo"]
cap = [r for r in rows if r["paper_bound"] == "capacity"]
span = lambda xs, f: (min(f(x) for x in xs), max(f(x) for x in xs))
summary = {
    "norm_geomean": math.exp(sum(log_ratio) / len(log_ratio)),
    "norm_mean_diff": sum(r["native_norm_tps"] - r["paper_norm_tps"] for r in other) / len(other),
    "norm_within_015": sum(abs(r["native_norm_tps"] - r["paper_norm_tps"]) <= 0.15 for r in other),
    "norm_count": len(other),
    "order_agree": agree, "order_total": total,
    "abs_ratio": {h: span([r for r in rows if r["hardware"] == h], lambda r: r["abs_ratio"]) for h in HW},
    "abs_ratio_all": span(rows, lambda r: r["abs_ratio"]),
    "slo_count": len(slo), "cap_count": len(cap),
    "slo_tpot_ratio": span(slo, lambda r: r["tpot_ratio"]),
    "slo_batch_ratio": span(slo, lambda r: r["batch_ratio"]),
    "cap_tpot_ratio": span(cap, lambda r: r["tpot_ratio"]),
    "cap_batch_ratio": span(cap, lambda r: r["batch_ratio"]),
    "bw_fraction": {h: span([r for r in rows if r["hardware"] == h], lambda r: r["native_bandwidth_fraction"]) for h in HW},
    "layout_byte_factor": span(rows, lambda r: r["layout_byte_factor"]),
    "candidates": data["total_candidates"],
}
data["summary"] = summary
out.mkdir(parents=True, exist_ok=True)
(out / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
html = (here / "template.html").read_text(encoding="utf-8")
(out / "index.html").write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1))
for r in rows:
    print("%-16s %-5s %-5s bound %-8s abs %.2f batch %.2f tpot %.2f bw %.2f layout_bytes %.2f" % (
        r["model"], r["workload"], r["hardware"], r["paper_bound"], r["abs_ratio"], r["batch_ratio"],
        r["tpot_ratio"], r["native_bandwidth_fraction"], r["layout_byte_factor"]))
