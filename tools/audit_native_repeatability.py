"""Audit preserved native repeats without executing native or fitting simulator costs.

All captured samples remain in the report.  A +/-5% check is the maximum
absolute deviation from the matched scenario median, never a CV threshold.
Slot/rank matching is diagnostic: rank is engine-start order, not HTTP arrival.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("ttft", "tpot", "e2e")
NAME = re.compile(r"^(?P<model>.+)__(?P<prompt>short|medium|long)__(?P<output>short|medium|long)__p(?P<parallel>\d+)__r(?P<repeat>\d+)\.native_raw\.json$")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def deviation_summary(values: list[float], tolerance_pct: float = 5.0) -> dict[str, Any]:
    """Preserve signed deviations; CV is independent descriptive information."""
    if not values:
        return {"n": 0, "status": "evidence_insufficient", "observed_within_band": False}
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("latencies must be finite and strictly positive")
    median = statistics.median(values)
    deviations = [100.0 * (value / median - 1.0) for value in values]
    absolute = [abs(value) for value in deviations]
    within = sum(value <= tolerance_pct + 1e-10 for value in absolute)
    return {
        "n": len(values), "median_ms": median, "mean_ms": statistics.mean(values),
        "min_ms": min(values), "max_ms": max(values), "values_ms": values,
        "signed_deviation_pct": deviations, "min_signed_deviation_pct": min(deviations),
        "max_signed_deviation_pct": max(deviations), "max_abs_deviation_pct": max(absolute),
        "p90_abs_deviation_pct": percentile(absolute, 0.90),
        "fraction_within_5pct": within / len(values), "within_count": within,
        "cv_pct": 100.0 * statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None,
        "observed_within_band": len(values) > 1 and within == len(values),
        "statistical_guarantee": False,
    }


def timing_record(response: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract only engine timestamps; never fall back to client boundaries."""
    timings = response.get("timings", {})
    times = timings.get("engine_token_times_us", [])
    begin = timings.get("engine_request_begin_us")
    violations: list[str] = []
    expected = response.get("tokens_predicted")
    if timings.get("engine_clock") != "ggml_time_us_monotonic":
        violations.append("unknown_engine_clock")
    if timings.get("engine_timepoints_scope") != "non_speculative_generated_tokens":
        violations.append("unknown_engine_scope")
    if not isinstance(times, list) or not times or not all(isinstance(x, int) and not isinstance(x, bool) for x in times):
        return {}, ["missing_or_noninteger_token_timestamps"]
    if not isinstance(begin, int) or isinstance(begin, bool):
        return {}, ["missing_or_noninteger_engine_start"]
    if timings.get("engine_timepoints_complete") is not True:
        violations.append("timepoints_not_declared_complete")
    if times != sorted(times) or begin > times[0]:
        violations.append("nonmonotonic_timepoints")
    if len(times) != expected or len(times) != timings.get("predicted_n"):
        violations.append("output_token_count_mismatch")
    if timings.get("engine_prompt_last_us") != times[0] or timings.get("engine_last_token_us") != times[-1]:
        violations.append("first_or_last_timestamp_mismatch")
    ttft = (times[0] - begin) / 1000.0
    tpot = (times[-1] - times[0]) / (1000.0 * (len(times) - 1)) if len(times) > 1 else None
    e2e = (times[-1] - begin) / 1000.0
    for key, reference in (("prompt_ms", ttft), ("predicted_ms", (times[-1] - times[0]) / 1000.0)):
        value = timings.get(key)
        # Single-token legacy counters clamp a zero interval to 1 us.
        if len(times) > 1 and (not isinstance(value, (int, float)) or abs(value - reference) > 1e-6):
            violations.append(f"counter_{key}_mismatch")
    if min(ttft, e2e) <= 0 or (tpot is not None and tpot <= 0):
        violations.append("nonpositive_latency")
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e, "begin_us": begin,
            "first_us": times[0], "last_us": times[-1], "token_times_us": times,
            "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
            "output_n": expected, "slot_id": response.get("id_slot")}, violations


def cohort_description(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"status": "evidence_insufficient"}
    first = min(row["first_us"] for row in records)
    begins = [row["begin_us"] for row in records]
    last = max(row["last_us"] for row in records)
    late = sum(row["begin_us"] > first for row in records)
    # Exact split marker: at least one engine request starts after another has
    # produced its first token. Long prompt chunking can cause this intentionally.
    return {"engine_begin_spread_ms": (max(begins) - min(begins)) / 1000.0,
            "engine_batch_makespan_ms": (last - min(begins)) / 1000.0,
            "requests_started_after_first_token": late,
            "split_before_all_engine_starts": late > 0,
            "all_engine_starts_before_first_token": late == 0,
            "distinct_slot_count": len({row["slot_id"] for row in records}),
            "slot_sequence_by_start": [row["slot_id"] for row in sorted(records, key=lambda r: (r["begin_us"], r.get("request_index", 0)))]}


def artifact_identity(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def policy_projection(response: dict[str, Any]) -> dict[str, Any]:
    settings = response.get("generation_settings", {})
    return {key: settings.get(key) for key in ("seed", "temperature", "top_k", "ignore_eos", "stream", "n_predict", "stop", "backend_sampling", "speculative.types")}


def command_config_issues(command: list[str], config: dict[str, Any]) -> list[str]:
    issues = []
    for flag, field in (("-c", "ctx"), ("-np", "parallel"), ("-ngl", "gpu_layers"),
                        ("-b", "batch"), ("-ub", "ubatch"), ("-t", "threads"),
                        ("-tb", "threads_batch"), ("-ctk", "kv_type_k"), ("-ctv", "kv_type_v")):
        if flag not in command or command.index(flag) + 1 >= len(command):
            issues.append(f"missing_command_field:{field}")
        elif str(config.get(field)) != command[command.index(flag) + 1]:
            issues.append(f"command_config_mismatch:{field}")
    return issues


def load_cell(path: Path) -> dict[str, Any]:
    match = NAME.match(path.name)
    if not match:
        raise ValueError(f"unsupported capture name: {path.name}")
    info = match.groupdict()
    comparison_path = path.with_name(path.name.replace(".native_raw.json", ".json"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    comp = json.loads(comparison_path.read_text(encoding="utf-8"))
    source = artifact_identity(path)
    errors: list[str] = []
    config = comp.get("configuration", {})
    if raw.get("command") != comp.get("command"):
        errors.append("raw_comparison_command_mismatch")
    expected_raw_sha = comp.get("evidence", {}).get("raw_native_capture", {}).get("sha256")
    if source["sha256"] != expected_raw_sha:
        errors.append("raw_sha_mismatch")
    errors.extend(command_config_issues(raw.get("command", []), config))
    responses = raw.get("responses", [])
    parallel = int(info["parallel"])
    if len(responses) != parallel or config.get("parallel") != parallel:
        errors.append("request_count_mismatch")
    if len(raw.get("request_boundaries", [])) != parallel:
        errors.append("client_boundary_count_mismatch")
    extracted = comp.get("requests", {}).get("native", [])
    if len(extracted) != parallel:
        errors.append("extractor_request_count_mismatch")
    records = []
    policies = []
    requested = comp.get("request", {}).get("requested_output_tokens")
    prompt_expected = comp.get("token_counts", {}).get("prompt")
    for i, response in enumerate(responses):
        row, violations = timing_record(response)
        errors.extend(f"request_{i}:{reason}" for reason in violations)
        policies.append(policy_projection(response))
        if response.get("truncated") is not False or response.get("stop_type") != "limit":
            errors.append(f"request_{i}:unexpected_finish_policy")
        if response.get("tokens_predicted") != requested:
            errors.append(f"request_{i}:requested_output_mismatch")
        if response.get("tokens_evaluated") != prompt_expected:
            errors.append(f"request_{i}:prompt_count_mismatch")
        if response.get("generation_settings", {}).get("ignore_eos") is not True:
            errors.append(f"request_{i}:ignore_eos_not_enabled")
        if row:
            row["request_index"] = i
            row["client_boundary"] = raw.get("request_boundaries", [])[i] if i < len(raw.get("request_boundaries", [])) else None
            if row["cache_n"] != 0:
                errors.append(f"request_{i}:cache_reuse_or_unknown")
            if i < len(extracted):
                for metric in METRICS:
                    saved = extracted[i].get(f"engine_{metric}_ms")
                    if row[metric] is None:
                        if saved is not None: errors.append(f"request_{i}:single_token_tpot_not_na")
                    elif not isinstance(saved, (int, float)) or abs(saved - row[metric]) > 1e-6:
                        errors.append(f"request_{i}:extractor_{metric}_mismatch")
            records.append(row)
    for rank, record in enumerate(sorted(records, key=lambda r: (r["begin_us"], r["request_index"]))):
        record["engine_start_rank"] = rank
    warm_responses = comp.get("warmup", {}).get("responses", [])
    warm_rows = []
    for i, response in enumerate(warm_responses):
        row, violations = timing_record(response)
        if row:
            row["request_index"] = i
            row["policy"] = policy_projection(response)
            row["stop_type"] = response.get("stop_type")
            row["timing_validation_issues"] = violations
            warm_rows.append(row)
    batch_metrics = {metric: statistics.median([row[metric] for row in records if row[metric] is not None])
                     if records and all(row[metric] is not None for row in records) else None for metric in METRICS}
    warm_same_length = bool(warm_rows) and len(warm_rows) == parallel and all(row["output_n"] == requested for row in warm_rows)
    proof = comp.get("evidence", {}).get("engine_semantic_proof", {})
    return {
        "cell_id": path.name.removesuffix(".native_raw.json"),
        "scenario_id": path.name.rsplit("__r", 1)[0], "model": info["model"],
        "prompt_level": info["prompt"], "output_level": info["output"],
        "parallel": parallel, "repeat": int(info["repeat"]), "records": records,
        "metrics_ms": batch_metrics, "cohort": cohort_description(records),
        "warmup": {"request_count": len(warm_responses), "records": warm_rows,
                   "cohort": cohort_description(warm_rows), "wall_ms": comp.get("warmup", {}).get("wall_ms"),
                   "same_output_length_as_measured": warm_same_length,
                   "all_fixed_eos": bool(warm_rows) and all(row["policy"]["ignore_eos"] is True for row in warm_rows),
                   "all_stream": bool(warm_rows) and all(row["policy"]["stream"] is True for row in warm_rows)},
        "batch_client_wall_ms": raw.get("batch_client_wall_ms"), "actual_policies": policies,
        "configuration": config, "raw_source": source, "comparison_source": artifact_identity(comparison_path),
        "server": comp.get("server"), "captured_utc": raw.get("captured_utc"),
        "identity": comp.get("identity", {}), "runtime_stable": comp.get("evidence", {}).get("runtime_stable"),
        "captured_proof_sha_present": bool(proof.get("captured_sha256")),
        "native_binary_sha256": comp.get("evidence", {}).get("native_binary", {}).get("sha256"),
        "violations": errors, "formula_and_raw_checks_pass": not errors,
        "tokens_cached_field": [r.get("tokens_cached") for r in responses],
    }


def matching_diagnostics(cells: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for cell in cells:
        for record in cell["records"]:
            groups[record[key]].append({**record, "cell_id": cell["cell_id"], "repeat": cell["repeat"]})
    matches = []
    pooled: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for label, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        per_metric = {}
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary = deviation_summary(values)
            per_metric[metric] = summary
            pooled[metric].extend(summary.get("signed_deviation_pct", []))
        repeat_counts = collections.Counter(row["repeat"] for row in rows)
        matches.append({"position": label, "sample_count": len(rows),
                        "repeat_counts": dict(repeat_counts), "one_request_per_repeat": all(n == 1 for n in repeat_counts.values()),
                        "request_refs": [{"cell_id": row["cell_id"], "request_index": row["request_index"], "slot_id": row["slot_id"], "engine_start_rank": row["engine_start_rank"]} for row in rows],
                        "metrics": per_metric})
    return {"matching_key": key, "positions": matches,
            "pooled": {metric: summarize_deviations(values) for metric, values in pooled.items()}}


def summarize_deviations(values: list[float]) -> dict[str, Any]:
    absolute = [abs(x) for x in values]
    return {"n": len(values), "max_abs_deviation_pct": max(absolute) if values else None,
            "p90_abs_deviation_pct": percentile(absolute, 0.9),
            "fraction_within_5pct": sum(x <= 5.0 + 1e-10 for x in absolute) / len(values) if values else None,
            "signed_deviations_pct": values}


def summarize_scenario(cells: list[dict[str, Any]], expected_repeats: int) -> dict[str, Any]:
    cells = sorted(cells, key=lambda x: x["repeat"])
    first = cells[0]
    metrics = {metric: deviation_summary([cell["metrics_ms"][metric] for cell in cells if cell["metrics_ms"][metric] is not None]) for metric in METRICS}
    repeat_complete = len(cells) == expected_repeats and len({c["repeat"] for c in cells}) == expected_repeats
    records_valid = all(c["formula_and_raw_checks_pass"] for c in cells)
    return {"scenario_id": first["scenario_id"], "model": first["model"], "prompt_level": first["prompt_level"],
            "output_level": first["output_level"], "parallel": first["parallel"], "repeat_count": len(cells),
            "repeat_complete": repeat_complete, "metrics": metrics,
            "observed_all_three_within_5pct": repeat_complete and records_valid and all(m["observed_within_band"] for m in metrics.values()),
            "worst_metric_deviation_pct": max(m.get("max_abs_deviation_pct", float("inf")) for m in metrics.values()),
            "raw_checks_pass": records_valid, "cell_ids": [c["cell_id"] for c in cells],
            "split_batches": sum(c["cohort"].get("split_before_all_engine_starts", False) for c in cells),
            "warmup_split_batches": sum(c["warmup"]["cohort"].get("split_before_all_engine_starts", False) for c in cells),
            "batch_client_wall_median_ms": statistics.median(c["batch_client_wall_ms"] for c in cells),
            "engine_batch_makespan_median_ms": statistics.median(c["cohort"]["engine_batch_makespan_ms"] for c in cells),
            "prompt_tokens": sorted({r["prompt_n"] for c in cells for r in c["records"]}),
            "output_tokens": sorted({r["output_n"] for c in cells for r in c["records"]}),
            "slot_matched_requests": matching_diagnostics(cells, "slot_id"),
            "rank_matched_requests": matching_diagnostics(cells, "engine_start_rank")}


def audit(source: Path, expected_repeats: int = 3, expected_runs: int | None = None) -> dict[str, Any]:
    paths = sorted(source.glob("*.native_raw.json"))
    cells = [load_cell(path) for path in paths]
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for cell in cells: groups[cell["scenario_id"]].append(cell)
    scenarios = [summarize_scenario(rows, expected_repeats) for _, rows in sorted(groups.items())]
    by_model = []
    for model in sorted({row["model"] for row in scenarios}):
        rows = [row for row in scenarios if row["model"] == model]
        samples = [c for c in cells if c["model"] == model]
        pooled = {metric: summarize_deviations([d for row in rows for d in row["metrics"][metric].get("signed_deviation_pct", [])]) for metric in METRICS}
        by_model.append({"model": model, "scenario_count": len(rows), "run_count": len(samples),
                         "request_count": sum(len(c["records"]) for c in samples),
                         "observed_all_three_within_5pct_count": sum(r["observed_all_three_within_5pct"] for r in rows),
                         "rank_matched_all_three_within_5pct_count": sum(all(r["rank_matched_requests"]["pooled"][m]["max_abs_deviation_pct"] <= 5 for m in METRICS) for r in rows),
                         "pooled_scenario_matched_batch_deviations": pooled,
                         "split_batches": sum(r["split_batches"] for r in rows), "warmup_split_batches": sum(r["warmup_split_batches"] for r in rows),
                         "warmup_early_eos_requests": sum(r["stop_type"] == "eos" for c in samples for r in c["warmup"]["records"]),
                         "most_variable_scenario": max(rows, key=lambda r: r["worst_metric_deviation_pct"])["scenario_id"],
                         "least_variable_scenario": min(rows, key=lambda r: r["worst_metric_deviation_pct"])["scenario_id"]})
    hashes = collections.Counter(c["raw_source"]["sha256"] for c in cells)
    # Reuse measured wall time only for a transparent experiment-budget estimate.
    # This is not a promised duration: same-shape warmup differs from the old
    # two-token warmup, and process startup/collection overhead is additional.
    compact_qwen38 = ["qwen38__short__medium__p4", "qwen38__medium__medium__p2", "qwen38__long__medium__p1"]
    boundary_qwen38 = ["qwen38__short__short__p1", "qwen38__medium__long__p1", "qwen38__long__long__p4"]
    scenario_map = {s["scenario_id"]: s for s in scenarios}
    repeat_budget = []
    for scenario_id in compact_qwen38 + boundary_qwen38:
        if scenario_id not in scenario_map: continue
        row = scenario_map[scenario_id]
        wall_s = row["batch_client_wall_median_ms"] / 1000.0
        repeat_budget.append({"scenario_id": scenario_id,
                              "tier": "primary" if scenario_id in compact_qwen38 else "boundary_confirmation",
                              "actual_prompt_tokens": row["prompt_tokens"], "actual_output_tokens": row["output_tokens"],
                              "median_native_batch_wall_s": wall_s,
                              "estimated_2_processes_each_8_warmup_20_measured_s": 56.0 * wall_s,
                              "estimate_excludes": "process start, model load, identity capture, preconnection, disk flush; no upper bound"})
    unique_identity = {field: sorted({str(c["identity"].get(field)) for c in cells}) for field in ("hardware_fingerprint", "runtime_fingerprint", "gguf_sha256")}
    policy_sets = collections.Counter(json.dumps(p, sort_keys=True) for c in cells for p in c["actual_policies"])
    warmup_failures = [{"cell_id": c["cell_id"], "warmup": c["warmup"]["cohort"], "actual": c["cohort"],
                        "ttft_ms": c["metrics_ms"]["ttft"],
                        "warmup_outputs": [r["output_n"] for r in c["warmup"]["records"]]}
                       for c in cells if c["warmup"]["cohort"].get("split_before_all_engine_starts")]
    return {"schema": "native-repeatability-audit/v1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": str(source.resolve()), "contract": {
                "boundary": "engine timestamp start -> first/last generated token; TPOT divides N_out-1",
                "batch_value": "median across all requests in one captured repeat",
                "reference": "median of repeat batch values for exactly the same model/shape/concurrency",
                "signed_deviation_formula": "100*(batch_value/reference-1)", "within_band": "every recorded repeated batch deviation is within [-5,+5] percent",
                "cv_formula": "100*sample_stdev(repeat batch values)/mean(repeat batch values)",
                "p90": "linear interpolation quantile of absolute repeat deviations, not request latency P90",
                "invalid_or_slow_samples_dropped": False,
                "slot_rank_note": "engine-start rank is not measured client arrival rank; matching reduces slot-order confounding but does not establish simultaneous arrival",
                "qualification": "three legacy repeats are descriptive screening, not an independent future +/-5% guarantee"},
            "summary": {"run_count": len(cells), "expected_runs": expected_runs,
                        "expected_run_count_pass": expected_runs is None or len(cells) == expected_runs,
                        "scenario_count": len(scenarios), "request_count": sum(len(c["records"]) for c in cells),
                        "formula_and_raw_check_failed_cells": sum(not c["formula_and_raw_checks_pass"] for c in cells),
                        "violations": [{"cell_id": c["cell_id"], "reasons": c["violations"]} for c in cells if c["violations"]],
                        "duplicate_raw_sha": {sha: count for sha, count in hashes.items() if count > 1},
                        "observed_all_three_within_5pct_scenarios": sum(r["observed_all_three_within_5pct"] for r in scenarios)},
            "models": by_model, "scenarios": scenarios, "cells": cells,
            "identity_unique_values": unique_identity,
            "actual_policy_counts": [{"policy": json.loads(policy), "request_count": count} for policy, count in sorted(policy_sets.items())],
            "qwen38_retest_budget": repeat_budget,
            "warmup_split_case_evidence": warmup_failures,
            "limitations": ["All five models are existing development data; this is not new model generalization.",
                            "Raw capture is sufficient to recompute engine timestamps, but historical observer equivalence and capture-time proof integrity remain separate gates.",
                            "Warmup was one non-streaming batch with EOS allowed and requested output 2; fixed-output streaming measurements used different policies.",
                            "tokens_cached is retained cache occupancy at request end; prompt reuse is assessed using cache_n.",
                            "Engine-start split alone is not causal proof of HTTP jitter; long prompt chunking can produce a legitimate staggered engine start."]}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Native 重复性审计（历史五模型，±5%口径）", "",
             "所有慢样本保留；原生数据只读重算。这里检查每次批次中位数相对同场景三次重复中位数的偏差。CV只是附加描述，不能替代±5%判定。三次重复只能初筛，不能证明未来每次运行都在±5%。", "",
             "| 模型 | 三项均在±5%的场景 | TTFT最坏偏差 | TPOT最坏偏差 | E2E最坏偏差 | 实测批次拆分/预热批次拆分 | 预热提前EOS请求 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["models"]:
        values = [f"{row['pooled_scenario_matched_batch_deviations'][m]['max_abs_deviation_pct']:.3f}%" for m in METRICS]
        lines.append(f"| {row['model']} | {row['observed_all_three_within_5pct_count']}/{row['scenario_count']} | " + " | ".join(values) + f" | {row['split_batches']}/{row['warmup_split_batches']} | {row['warmup_early_eos_requests']} |")
    lines += ["", "## 每场景重复", "", "| 场景 | TTFT 最大偏差 / CV | TPOT 最大偏差 / CV | E2E 最大偏差 / CV | 三项±5% | 批次实测墙钟中位秒 |", "|---|---:|---:|---:|---|---:|"]
    for row in report["scenarios"]:
        values = [f"{row['metrics'][m].get('max_abs_deviation_pct',0):.3f}% / {row['metrics'][m].get('cv_pct',0):.3f}%" for m in METRICS]
        lines.append(f"| {row['scenario_id']} | " + " | ".join(values) + f" | {'观测通过' if row['observed_all_three_within_5pct'] else '失败'} | {row['batch_client_wall_median_ms']/1000:.3f} |")
    lines += ["", "逐请求按slot和engine启动顺序匹配的偏差、原始token时间戳、计数/计时公式核对、真实预热与输出策略、SHA均保存在同名JSON。启动顺序不是客户端到达顺序。长prompt的拆分可能来自正确的batch容量/调度，不能全部当成HTTP抖动。", "", "## 限制", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/multimodel_next/p012_shape_screening_v7")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/development/native_repeatability_5pct/legacy_audit.json")
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--expected-runs", type=int, default=135)
    args = parser.parse_args()
    report = audit(args.source, args.expected_repeats, args.expected_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "models": [{k:v for k,v in row.items() if k != "pooled_scenario_matched_batch_deviations"} for row in report["models"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
