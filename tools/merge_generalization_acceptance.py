"""Merge frozen multi-model acceptance matrix without hiding failures."""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

MODELS = ("qwen25", "qwen35", "qwen38", "tinyllama", "smollm2")
PROMPTS = ("short", "medium", "long")
OUTPUTS = ("short", "medium", "long")
PARALLEL = (1, 2, 4)
REPEATS = 3
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms")

def percentile(values, p):
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals: return None
    pos = (len(vals)-1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] if lo == hi else vals[lo] + (vals[hi]-vals[lo]) * (pos-lo)

def expected_cells():
    out = {}
    for m in MODELS:
        for p in PROMPTS:
            for o in OUTPUTS:
                for n in PARALLEL:
                    for r in range(1, REPEATS+1):
                        cid = f"{m}__{p}__{o}__p{n}__r{r}"
                        out[cid] = (m, p, o, n, r)
    return out

def validate_cell(cell):
    reasons = []
    if cell.get("status") != "valid": reasons.append(f"status={cell.get('status')}")
    checks = cell.get("checks") or {}
    for name in ("schema", "geometry", "tokens", "parallel", "prompt", "model_sha", "binary_sha", "stream"):
        if checks.get(name) is not True: reasons.append(f"check:{name}")
    support = cell.get("parallel_support") or {}
    # A concurrency result is admissible only when both sides explicitly
    # model the requested number of slots.  Missing evidence is a failure,
    # rather than an implicit single-request fallback.
    if support.get("status") != "modeled": reasons.append(f"parallel_support={support.get('status')}")
    for field in ("native_requests", "simulator_requests"):
        if support.get(field) != cell.get("parallel"): reasons.append(f"{field}_count")
    return (not reasons), reasons

def metric_missing_reasons(cell):
    """Return per-metric missing fields without invalidating other metrics.

    TPOT is undefined when a request emits fewer than two output tokens. Such
    a cell remains usable for TTFT/E2E while its TPOT evidence is explicitly
    counted as incomplete.
    """
    out = {}
    for metric in METRICS:
        m = (cell.get("metrics") or {}).get(metric) or {}
        missing = []
        for field in ("native_ms", "simulator_ms", "signed_error_pct", "absolute_error_pct", "absolute_delta_ms"):
            v = m.get(field)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                missing.append(field)
        if missing: out[metric] = missing
    return out

def aggregate(cells):
    result = {"n": len(cells), "metrics": {}}
    for metric in METRICS:
        vals = [(c.get("metrics") or {}).get(metric) or {} for c in cells]
        vals = [v for v in vals if v.get("signed_error_pct") is not None]
        signed = [v["signed_error_pct"] for v in vals]
        absolute = [v["absolute_error_pct"] for v in vals]
        delta = [v["absolute_delta_ms"] for v in vals]
        native = [v["native_ms"] for v in vals]
        simulator = [v["simulator_ms"] for v in vals]
        result["metrics"][metric] = {
            "n": len(vals), "expected_repeats": REPEATS,
            "median_signed_pct": statistics.median(signed) if signed else None,
            "median_abs_pct": statistics.median(absolute) if absolute else None,
            "p90_abs_pct": percentile(absolute, 90),
            "worst_abs_pct": max(absolute) if absolute else None,
            "median_absolute_ms": statistics.median(delta) if delta else None,
            "p90_absolute_ms": percentile(delta, 90),
            "worst_absolute_ms": max(delta) if delta else None,
            "native_median_ms": statistics.median(native) if native else None,
            "simulator_median_ms": statistics.median(simulator) if simulator else None,
        }
    return result

def render_markdown(report):
    lines = ["# 泛化盲测验收表", "",
      f"- 场景 cell：{report['cell_count']}/{report['expected_cells']}；结构有效 {report['valid_cell_count']}；结构无效 {report['invalid_cell_count']}；缺失 {report['missing_cell_count']}；指标证据不完整 {report.get('metric_incomplete_cell_count', 0)}",
      f"- 执行次数：{report['observed_repeated_executions']}/{report['expected_repeated_executions']}（每格目标 {REPEATS} 次）",
      f"- 矩阵完整：`{report['matrix_complete']}`；全部有效：`{report['all_cells_valid']}`", "",
      "| 模型 | Prompt | Output | 并发 | 指标 | n/3 | signed median % | abs median % | abs p90 % | abs worst % | abs ms median | abs ms p90 | abs ms worst |",
      "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(report["aggregation"]):
        mname, prompt, output, parallel = key.split("|")
        for metric in METRICS:
            m = report["aggregation"][key]["metrics"][metric]
            fmt = lambda v: "—" if v is None else f"{v:.3f}"
            row = (mname, prompt, output, parallel, metric, f"{m['n']}/{m['expected_repeats']}",
                   fmt(m["median_signed_pct"]), fmt(m["median_abs_pct"]), fmt(m["p90_abs_pct"]),
                   fmt(m["worst_abs_pct"]), fmt(m["median_absolute_ms"]), fmt(m["p90_absolute_ms"]), fmt(m["worst_absolute_ms"]))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("artifacts/multimodel_next"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/multimodel_next/generalization_acceptance_full_v1.json"))
    ap.add_argument("--markdown-output", type=Path)
    args = ap.parse_args()
    expected = expected_cells(); rows = []; source_files = []; missing_sources = []
    for model in MODELS:
        path = args.input_dir / f"generalization_acceptance_{model}_ctx2048_v1.json"
        source_files.append(path.name)
        if not path.exists(): missing_sources.append(path.name); continue
        try: rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc: missing_sources.append(f"{path.name} (unreadable: {exc})")
    cells = [c for row in rows for c in (row.get("cells") or []) if isinstance(c, dict)]
    by_id = {}; duplicate_ids = []
    for c in cells:
        cid = c.get("cell_id")
        if not cid: duplicate_ids.append("<missing-cell-id>"); continue
        if cid in by_id: duplicate_ids.append(cid); continue
        by_id[cid] = c
    valid, invalid, metric_incomplete = [], [], []
    for cid, c in by_id.items():
        ok, reasons = validate_cell(c)
        if cid in expected:
            em, ep, eo, en, er = expected[cid]
            for field, want in (("model_key", em), ("prompt_band", ep), ("output_band", eo), ("parallel", en), ("repeat", er)):
                if c.get(field) != want:
                    ok = False
                    reasons.append(f"geometry:{field}")
        if ok:
            valid.append(c)
            missing_metrics = metric_missing_reasons(c)
            if missing_metrics:
                metric_incomplete.append({"cell_id": cid, "model_key": c.get("model_key"), "missing": missing_metrics,
                                          "actual_token_counts": c.get("actual_token_counts")})
        else:
            invalid.append({"cell_id": cid, "model_key": c.get("model_key"), "reasons": reasons})
    missing_ids = sorted(set(expected) - set(by_id)); unexpected_ids = sorted(set(by_id) - set(expected))
    groups = {}
    for c in valid:
        key = (c.get("model_key"), c.get("prompt_band"), c.get("output_band"), int(c.get("parallel", -1)))
        groups.setdefault(key, []).append(c)
    aggregation = {}
    for model in MODELS:
        for prompt in PROMPTS:
            for output in OUTPUTS:
                for parallel in PARALLEL:
                    key = (model, prompt, output, parallel)
                    aggregation["|".join(map(str, key))] = aggregate(groups.get(key, []))
    coverage = {}
    identity_evidence = {}
    for model in MODELS:
        ids = {cid for cid, spec in expected.items() if spec[0] == model}; observed = {cid for cid in by_id if cid.startswith(model + "__")}
        vm = sum(1 for c in valid if c.get("model_key") == model); im = sum(1 for c in invalid if c.get("model_key") == model)
        coverage[model] = {"expected_cells": len(ids), "observed_cells": len(observed), "valid_cells": vm, "invalid_cells": im,
          "missing_cells": len(ids-observed), "complete": ids == observed, "expected_repeated_executions": len(ids)}
        model_cells = [c for c in by_id.values() if c.get("model_key") == model]
        def uniques(path):
            vals = []
            for c in model_cells:
                v = c
                for part in path:
                    v = v.get(part) if isinstance(v, dict) else None
                if v is not None and v not in vals: vals.append(v)
            return vals
        configs = [json.loads(x) for x in sorted({json.dumps(c.get("config") or {}, sort_keys=True) for c in model_cells})]
        static_configs = [{k: v for k, v in cfg.items() if k != "parallel"} for cfg in configs]
        static_configs = [json.loads(x) for x in sorted({json.dumps(cfg, sort_keys=True) for cfg in static_configs})]
        runtime_by_parallel = {}
        for c in model_cells:
            fp = (c.get("identity") or {}).get("runtime_fingerprint")
            if fp is not None: runtime_by_parallel.setdefault(str(c.get("parallel")), set()).add(fp)
        identity_evidence[model] = {
            "runtime_fingerprints": uniques(("identity", "runtime_fingerprint")),
            "hardware_fingerprints": uniques(("identity", "hardware_fingerprint")),
            "model_sha256": uniques(("identity", "gguf_sha256")) or uniques(("model_sha256",)),
            "binary_sha256": uniques(("binary_sha256",)),
            "configs": configs,
            "static_configs": static_configs,
            "runtime_fingerprints_by_parallel": {k: sorted(v) for k, v in runtime_by_parallel.items()},
        }
        identity_evidence[model]["runtime_consistent_by_parallel"] = all(len(v) <= 1 for v in runtime_by_parallel.values())
        identity_evidence[model]["consistent"] = identity_evidence[model]["runtime_consistent_by_parallel"] and all(len(identity_evidence[model][k]) <= 1 for k in ("hardware_fingerprints", "model_sha256", "binary_sha256", "static_configs"))
    report = {"schema": "generalization-acceptance-full/v1", "models": list(MODELS), "prompt_bands": list(PROMPTS), "output_bands": list(OUTPUTS), "parallel_values": list(PARALLEL), "repeats": REPEATS,
      "expected_cells": len(expected), "planned_cell_count": len(expected), "expected_repeated_executions": len(expected), "source_files": source_files, "missing_source_files": missing_sources,
      "cell_count": len(by_id), "observed_repeated_executions": len(by_id), "valid_cell_count": len(valid), "invalid_cell_count": len(invalid), "missing_cell_count": len(missing_ids), "missing_cell_ids": missing_ids,
      "unexpected_cell_ids": unexpected_ids, "duplicate_cell_ids": sorted(set(duplicate_ids)), "matrix_complete": not missing_ids and not unexpected_ids and not missing_sources and not duplicate_ids,
      "all_cells_valid": len(valid) == len(expected) and not invalid and not metric_incomplete, "metric_incomplete_cell_count": len(metric_incomplete), "metric_incomplete_cells": metric_incomplete, "identity_evidence": identity_evidence,
      "identity_inconsistent_models": [m for m in MODELS if not identity_evidence[m]["consistent"]], "coverage_by_model": coverage, "invalid_cells": invalid, "aggregation": aggregation,
      "policy": "仅使用通过 schema/geometry/token/identity/stream/并发检查的 cell 计算统计；无效、重复、缺失 cell 保留并单独列出。每格 3 次重复，p90/worst 对重复 cell 的绝对误差计算。",
      "acceptance_reference": "10% absolute relative error is a reporting threshold, not a guarantee."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = args.markdown_output or args.output.with_suffix(".md"); md.parent.mkdir(parents=True, exist_ok=True); md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "markdown": str(md.resolve()), "cells": report["cell_count"], "valid": report["valid_cell_count"], "invalid": report["invalid_cell_count"], "missing": report["missing_cell_count"], "groups": len(aggregation)}, ensure_ascii=False))

if __name__ == "__main__": raise SystemExit(main())
