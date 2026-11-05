"""Collect the 20261102 scan's selected points next to the paper's Fig. 3/4 readings (8 GPU)."""
import json, glob, os, sys
from pathlib import Path
ROOT = Path("F:/codex_project/37_LLMsim/heterollm-sim-4.0.0")
D = ROOT / "artifacts/exploring_hbf_native_20261102"
P = ROOT / "tmp/paper_comparison_20261026"
paper = {(r["model"], r["workload"], r["hardware"]): r for r in json.load(open(P / "comparison.json", encoding="utf-8"))["rows"]}
MODELS = {"llama3_405b": "Llama3 405B", "llama4_maverick": "Llama4 Maverick"}

# Paper Fig. 3: the largest measured per-GPU batch within the SLO among every finished row of a case,
# including the acceptance runs of the max-batch search.
largest = {}
for path in (D / "summaries").glob("*.json"):
    row = json.load(open(path, encoding="utf-8"))
    if "parameters" not in row or row.get("status") != "completed":
        continue
    pr = row["parameters"]; parts = path.stem.split("__")
    upper = row["diagnostics"]["metric_bounds"]["mean_complete_request_tpot_ns"]["upper"]
    if upper <= 100e6:
        case = (parts[1], parts[3], parts[2].replace("_plus", "+"))
        largest[case] = max(largest.get(case, 0), pr["batch_size"] * pr["dp"] / 8)

rows = []
for model in MODELS:
    for wl in ("SHORT", "MID", "LONG"):
        for hw in ("HBM4", "HBF", "HBF+", "CONV", "CONV+"):
            tag = hw.replace("+", "_plus")
            p = json.load(open(D / "points" / f"native__{model}__{tag}__{wl}__G8.json", encoding="utf-8"))
            sel = p["selected"]; pr = sel["parameters"]; m = sel["metrics"]; s = p["search"]
            b = sel["diagnostics"]["metric_bounds"]["mean_complete_request_tpot_ns"]
            pp = paper[(model, wl, hw)]
            tested = sum(len(l.get("runs") or []) for l in s["layouts"])
            rows.append({
                "model": model, "model_label": MODELS[model], "workload": wl, "hardware": hw,
                "paper_tps": pp["paper_tps_approx"], "paper_norm_tps": pp["paper_normalized_tps_approx"],
                "paper_batch_per_gpu": pp["paper_max_batch_per_gpu_approx"],
                "native_tps": m["tokens_per_second_per_gpu"],
                "native_tpot_ms": m["mean_complete_request_tpot_ns"] / 1e6, "native_tpot_upper_ms": b["upper"] / 1e6,
                "native_ttft_ms": m.get("mean_complete_request_ttft_ns") and m["mean_complete_request_ttft_ns"] / 1e6,
                "layout": {k: pr[k] for k in ("dp", "tp", "pp", "ep")} | ({"microbatches": pr["microbatches"]} if pr.get("microbatches", 1) > 1 else {}),
                "batch_per_replica": pr["batch_size"], "native_batch_per_gpu": pr["batch_size"] * pr["dp"] / 8,
                "native_max_passing_batch_per_gpu": largest[(model, wl, hw)], "search_result": s["result"],
                "layouts_screened": len(s["layouts"]), "points_run": tested,
                "pruned": sum((l.get("outcome") or {}).get("kind") == "pruned" for l in s["layouts"])})
for model in MODELS:
    for wl in ("SHORT", "MID", "LONG"):
        base = next(r for r in rows if r["model"] == model and r["workload"] == wl and r["hardware"] == "HBM4")
        for r in rows:
            if r["model"] == model and r["workload"] == wl:
                r["native_norm_tps"] = r["native_tps"] / base["native_tps"]
                r["native_norm_batch"] = r["native_batch_per_gpu"] / base["native_batch_per_gpu"]
                r["native_norm_max_batch"] = r["native_max_passing_batch_per_gpu"] / base["native_max_passing_batch_per_gpu"]
                r["paper_norm_batch"] = r["paper_batch_per_gpu"] / base["paper_batch_per_gpu"]
st = json.load(open(D / "status.json", encoding="utf-8"))
out = {"rows": rows, "candidate_counts": st["candidate_counts"], "total_candidates": st["total_candidates"],
       "scan_dir": str(D), "paper_note": json.load(open(P / "comparison.json", encoding="utf-8"))["paper_values"]}
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for r in rows:
    L = r["layout"]
    print("%-15s %-5s %-5s paper %7.0f (%.2f) B/gpu %6.1f | native %7.1f (%.2f) B/gpu %6.1f max %6.1f (%.2f vs %.2f) TPOT %5.1f %-26s %s" % (
        r["model_label"], r["workload"], r["hardware"], r["paper_tps"], r["paper_norm_tps"], r["paper_batch_per_gpu"],
        r["native_tps"], r["native_norm_tps"], r["native_batch_per_gpu"], r["native_max_passing_batch_per_gpu"],
        r["native_norm_max_batch"], r["paper_norm_batch"], r["native_tpot_ms"], r["search_result"],
        "%(dp)d/%(tp)d/%(pp)d/%(ep)d" % L))
