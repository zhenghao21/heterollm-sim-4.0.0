"""Run four adaptive candidates after the original eight-run diagnostic ends.

This queue only reads the original results. It never repeats detailed Decode,
resumes the formal scan, or allows a forecast guard to run a detailed suffix.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_fast_mode import compare_case, compare_pair, positive, write_json
from validate_fast_long_context import (
    ARCHITECTURES, CASES as BASELINE_CASES, MODELS, OUTPUT, PROMPT,
    case_name as baseline_case_name, completed, manager_lock, now,
    process_alive, read_json,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = tuple((model, arch, "fast") for model in MODELS for arch in ARCHITECTURES)
GUARD_ERROR = "FastForecastUnavailable"
WALL_SCOPE = ("historical measurements on this machine with the same request, affinity and GC; "
              "not an interleaved ABBA experiment; ambient background load was not controlled")


def case_name(case):
    return baseline_case_name(case) + "_adaptive"


def active_worker_ids(directory, state):
    """Prefer the real interpreter PID; Windows launchers may have another PID."""
    current = state.get("current_case")
    progress = (read_json(directory / "progress" / (current + ".json")) if current else None)
    progress = progress or state.get("worker_progress") or {}
    ids = [progress.get("process_id"), state.get("worker_pid")]
    # An exited manager can leave a real interpreter behind. Its most recent
    # progress remains running even when a launcher PID has already vanished.
    if not current:
        for path in (directory / "progress").glob("*.json"):
            row = read_json(path) or {}
            if row.get("status") in {"starting", "running"}:
                ids.append(row.get("process_id"))
    return tuple(dict.fromkeys(pid for pid in ids if pid and process_alive(pid)))


def baseline_gate(directory):
    state = read_json(directory / "status.json") or {}
    status = state.get("status")
    if (directory / "stop.request").exists() or status in {"paused", "stopping", "failed"}:
        return "blocked", "Original validation stopped: " + str(state.get("reason") or status or "stop.request")
    rows = [read_json(directory / (baseline_case_name(case) + ".json")) for case in BASELINE_CASES]
    count = sum(completed(row, case) for row, case in zip(rows, BASELINE_CASES))
    if status == "completed":
        if count != 8:
            return "blocked", f"Original status says completed but only {count}/8 complete request results exist"
        live = active_worker_ids(directory, state)
        if live:
            return "waiting", "Original worker is still alive: " + ", ".join(map(str, live))
        return "ready", "Original 8/8 complete; no live original worker"
    if status == "running" and state.get("manager_pid") and not process_alive(state["manager_pid"]):
        return "blocked", "Original validation manager exited before completion; restore that queue first"
    if status not in {None, "running", "starting", "waiting"}:
        return "blocked", "Unrecognized original validation state: " + str(status)
    return "waiting", f"Waiting for original validation ({count}/8 complete; state={status or 'missing'})"


def candidate_status(row, case):
    if not row or any(row.get(key) != value for key, value in zip(
            ("model_id", "architecture", "requested_mode"), case)):
        return None
    if row.get("prompt_tokens") != PROMPT or row.get("output_tokens") != OUTPUT:
        return None
    if row.get("error_type") == GUARD_ERROR:
        return "forecast_unavailable"
    strategy = row.get("simulation_config", {}).get("fast_decode", {}).get("sampling_strategy")
    if (completed(row, case) and strategy == "adaptive"
            and row.get("approximation", {}).get("effective_mode") == "fast"
            and row.get("approximation", {}).get("predicted_decode_steps", 0) > 0):
        return "completed"
    return None


def propagate_baseline_stop(directory, baseline):
    """Forward a later user pause without removing or rewriting either marker."""
    original = read_json(baseline / "status.json") or {}
    status = original.get("status")
    marker_exists = (baseline / "stop.request").exists()
    if not marker_exists and status not in {"paused", "stopping"}:
        return None
    reason = "Original validation pause requested: " + str(
        original.get("reason") or ("stop.request" if marker_exists else status))
    try:
        with (directory / "stop.request").open("x", encoding="utf-8") as handle:
            handle.write(reason + "\n")
    except FileExistsError:
        pass
    return reason


def command_for(directory, case):
    model, architecture, _ = case
    name = case_name(case)
    return [sys.executable, "-u", str(ROOT / "scripts" / "benchmark_fast_mode.py"),
        "--model", model, "--architecture", architecture, "--mode", "fast",
        "--sampling-strategy", "adaptive", "--require-forecast",
        "--prompt", str(PROMPT), "--output", str(OUTPUT),
        "--result", str(directory / (name + ".json")),
        "--progress-file", str(directory / "progress" / (name + ".json")),
        "--stop-file", str(directory / "stop.request"), "--report-timing"]


def optimization_comparison(previous_fast, candidate, *, valid):
    """Compare real measured stages, not predicted Decode service time."""
    def measurements(row):
        timing = row.get("timing_breakdown", {})
        approximation = row.get("approximation", {})
        return {"total_wall_seconds": row.get("wall_seconds"),
            "prefill_wall_seconds": timing.get("prefill_cohorts_wall_seconds"),
            "real_decode_wall_seconds": timing.get("decode_cohorts_wall_seconds"),
            "sampled_decode_steps": approximation.get("sampled_decode_steps", row.get("decode_steps")),
            "effective_mode": approximation.get("effective_mode")}
    previous, current = measurements(previous_fast), measurements(candidate)
    valid = valid and all(row["effective_mode"] == "fast" for row in (previous, current))
    ratios = {}
    for field in ("total_wall_seconds", "prefill_wall_seconds", "real_decode_wall_seconds"):
        left, right = previous[field], current[field]
        ratios[field] = left / right if valid and positive(left) and positive(right) else None
    return {"wall_comparison_scope": WALL_SCOPE, "valid_comparison": bool(valid),
        "previous_fast": previous, "adaptive_fast": current,
        "previous_over_adaptive": ratios,
        "real_decode_scope": "measured ordinary Decode cohort intervals only; excludes forecast/reduction after the final cohort",
        "sample_counts_are": "actually executed Decode cohorts; not predicted output tokens"}


def refresh(directory, baseline, state):
    rows = {case: read_json(directory / (case_name(case) + ".json")) for case in CANDIDATES}
    statuses = {case: candidate_status(row, case) for case, row in rows.items()}
    attempted = sum(value is not None for value in statuses.values())
    successes = sum(value == "completed" for value in statuses.values())
    current = state.get("current_case")
    progress = (read_json(directory / "progress" / (current + ".json")) if current else None)
    state = {**state, "updated_at": now(), "total_candidates": 4,
        "attempted_candidates": attempted, "completed_candidates": successes,
        "forecast_unavailable_candidates": attempted - successes,
        "worker_progress": progress, "baseline_directory": str(baseline),
        "wall_comparison_scope": WALL_SCOPE,
        "purpose": "four adaptive fast candidates only; original validation and formal scan untouched"}
    if progress and progress.get("process_id"):
        state["worker_process_id"] = progress["process_id"]
    write_json(directory / "status.json", state)
    comparisons = {}
    per_case = {}
    for case, row in rows.items():
        detailed_case = (case[0], case[1], "detailed")
        reference = read_json(baseline / (baseline_case_name(detailed_case) + ".json"))
        if row and reference and statuses[case]:
            comparison = compare_case(reference, row)
            comparison["wall_comparison_scope"] = WALL_SCOPE
            previous_fast = read_json(baseline / (baseline_case_name(case) + ".json"))
            previous_comparison = compare_case(reference, previous_fast) if previous_fast else None
            previous_wall = (previous_fast or {}).get("wall_seconds")
            current_wall = row.get("wall_seconds")
            comparison["previous_fast_wall_seconds"] = previous_wall
            comparison["previous_fast_over_current_fast"] = (
                previous_wall / current_wall if comparison["valid_comparison"]
                and previous_comparison and previous_comparison["valid_comparison"]
                and positive(previous_wall) and positive(current_wall) else None)
            comparison["optimization_comparison"] = optimization_comparison(
                previous_fast or {}, row, valid=bool(comparison["valid_comparison"]
                    and previous_comparison and previous_comparison["valid_comparison"]))
            per_case[case] = comparison
    for model in MODELS:
        cases = [case for case in CANDIDATES if case[0] == model]
        if all(statuses[case] for case in cases):
            reference_rows = [read_json(baseline / (baseline_case_name((model, arch, "detailed")) + ".json"))
                              for arch in ARCHITECTURES]
            if all(reference_rows):
                comparison = compare_pair([*reference_rows, *(rows[case] for case in cases)])
                comparison["speedup_scope"] = WALL_SCOPE
                for case in cases:
                    if case in per_case:
                        comparison["cases"][case[1]] = per_case[case]
                comparisons[model] = comparison
                write_json(directory / (model + "_comparison.json"), comparison)
    all_within = (all(row["all_within_target"] for row in comparisons.values())
                  if len(comparisons) == 2 else None)
    write_json(directory / "comparison.json", {"attempted_candidates": attempted,
        "completed_candidates": successes, "total_candidates": 4,
        "comparisons": comparisons, "all_within_target": all_within,
        "wall_comparison_scope": WALL_SCOPE})
    lines = ["# 64K / 5K 新版快速模式验证", "",
        f"更新：{state['updated_at']}；状态：{state['status']}；已尝试 {attempted}/4，成功预测 {successes}/4。", "",
        "只运行 0.5B、7B 的 HBM/HBF 共 4 次 adaptive 快速候选；复用原验证完整详细参考。",
        "原 8 次验证按原版本完成后才启动。正式扫描保持暂停，不重复完整详细 Decode。",
        "壁钟加速比使用同机历史测量；请求、绑核、GC 条件须匹配，但后台负载未受控，非 ABBA。", ""]
    if state.get("reason"):
        lines += ["说明：" + str(state["reason"]), ""]
    if current:
        lines += [f"当前：`{current}`；启动器 PID={state.get('worker_pid')}，实际 PID={state.get('worker_process_id')}。"]
    if progress:
        lines += [f"阶段：{progress.get('phase')}；Prefill {progress.get('prefill_steps', 0)}/32 块；"
            f"真实 Decode {progress.get('decode_steps', 0)} 步；已运行 {progress.get('wall_seconds', 0):.1f} 秒。", ""]
    lines += ["| 模型 | 存储 | 状态 | 原详细 s | 原快速 s | 新快速 s | 详细/快速 | 原快速/新快速 | Decode 加速 | TTFT 误差 | TPOT 误差 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    errors = []
    for case in CANDIDATES:
        comparison, row = per_case.get(case), rows[case]
        values = ["待完成"] * 8
        if comparison:
            def display(value, kind):
                if value is None:
                    return "不可比较"
                return f"{value:.3%}" if kind == "error" else f"{value:.3f}" + ("×" if kind == "ratio" else "")
            values = [display(comparison["wall_seconds"]["detailed"], "wall"),
                display(comparison["previous_fast_wall_seconds"], "wall"),
                display(comparison["wall_seconds"]["fast"], "wall"),
                display(comparison["speedup"]["total_wall"], "ratio"),
                display(comparison["previous_fast_over_current_fast"], "ratio"),
                display(comparison["speedup"]["decode_wall"], "ratio")]
            values += [display(comparison["relative_errors"][key], "error") for key in ("ttft_ns", "tpot_ns")]
        label = {"completed": "完成", "forecast_unavailable": "未通过：守卫拒绝预测"}.get(statuses[case], "待完成")
        lines.append("| " + " | ".join([case[0], case[1], label, *values]) + " |")
        if row and row.get("error_type"):
            errors.append(f"`{case_name(case)}`：{row['error_type']}：{row.get('reason', '')}")
    if errors:
        lines += ["", *errors]
    lines += ["", "原快模与新版各阶段实测（同机历史比较；非 ABBA）：", "",
        "| 模型 | 存储 | 原 Prefill s | 新 Prefill s | Prefill 加速 | 原实算 Decode s | 新实算 Decode s | 实算 Decode 加速 | 原/新实算步数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in CANDIDATES:
        optimization = per_case.get(case, {}).get("optimization_comparison")
        values = ["待完成"] * 7
        if optimization:
            previous, current = optimization["previous_fast"], optimization["adaptive_fast"]
            ratios = optimization["previous_over_adaptive"]
            def wall(value):
                return "未记录" if value is None else f"{value:.3f}"
            def ratio(value):
                return "不可比较" if value is None else f"{value:.3f}×"
            values = [wall(previous["prefill_wall_seconds"]), wall(current["prefill_wall_seconds"]),
                ratio(ratios["prefill_wall_seconds"]), wall(previous["real_decode_wall_seconds"]),
                wall(current["real_decode_wall_seconds"]), ratio(ratios["real_decode_wall_seconds"]),
                "/".join("未记录" if row["sampled_decode_steps"] is None
                         else str(row["sampled_decode_steps"]) for row in (previous, current))]
        lines.append("| " + " | ".join([case[0], case[1], *values]) + " |")
    lines += ["", "实算 Decode 耗时只含真实 cohort 区间；上述详细/快速 Decode 加速另含预测与收尾。"]
    lines += ["", "| 模型 | 详细 TPOT HBF/HBM | 新快速 TPOT HBF/HBM | 比值误差 | 全部 10% 检查 |",
        "| --- | ---: | ---: | ---: | --- |"]
    for model in MODELS:
        comparison = comparisons.get(model)
        if comparison is None:
            lines.append(f"| {model} | 待完成 | 待完成 | 待完成 | 待完成 |")
            continue
        ratio = comparison["ratio_comparison"]["tpot_ns"]
        values = ["不可比较" if ratio["hbf_over_hbm"][mode] is None
                  else f"{ratio['hbf_over_hbm'][mode]:.6f}" for mode in ("detailed", "fast")]
        error = ratio["relative_error"]
        lines.append("| " + " | ".join([model, *values,
            "不可比较" if error is None else f"{error:.3%}",
            "通过" if comparison["all_within_target"] else "未通过"]) + " |")
    lines += ["", "误差只针对这 4 个完整请求。守卫拒绝即记未通过，不回退执行 5,119 步详细 Decode。", ""]
    temporary = directory / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(directory / "report.md")
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)
    return state


def run(directory, baseline):
    directory, baseline = directory.resolve(), baseline.resolve()
    if directory.is_relative_to(baseline):
        raise ValueError("Candidate output directory must not write inside the original validation")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "progress"):
        (directory / name).mkdir(exist_ok=True)
    stop = directory / "stop.request"
    state = {"status": "waiting_baseline", "manager_pid": os.getpid(), "started_at": now()}
    with manager_lock(directory):
        previous = read_json(directory / "status.json") or {}
        live = active_worker_ids(directory, previous)
        if live:
            raise RuntimeError("Previous candidate worker is still alive; refusing duplicate execution: "
                               + ", ".join(map(str, live)))
        while True:
            if stop.exists():
                refresh(directory, baseline, {**state, "status": "paused", "reason": "stop.request exists"})
                return 2
            gate, reason = baseline_gate(baseline)
            state.update(reason=reason)
            if gate == "blocked":
                refresh(directory, baseline, {**state, "status": "baseline_stopped"})
                return 2
            if gate == "ready":
                break
            refresh(directory, baseline, state)
            time.sleep(10)
        state.update(status="running", reason=None, baseline_ready_at=now())
        for case in CANDIDATES:
            pause_reason = propagate_baseline_stop(directory, baseline)
            if stop.exists():
                refresh(directory, baseline, {**state, "status": "paused",
                    "reason": pause_reason or "stop.request exists"})
                return 2
            name = case_name(case)
            result_path = directory / (name + ".json")
            if candidate_status(read_json(result_path), case):
                continue
            with (directory / "logs" / (name + ".log")).open("a", encoding="utf-8") as output:
                worker = subprocess.Popen(command_for(directory, case), cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
                state.update(current_case=name, worker_pid=worker.pid,
                    worker_process_id=None, case_started_at=now())
                while worker.poll() is None:
                    pause_reason = propagate_baseline_stop(directory, baseline)
                    if pause_reason:
                        state["reason"] = pause_reason
                    state["status"] = "stopping" if stop.exists() else "running"
                    refresh(directory, baseline, state)
                    time.sleep(10)
            row = read_json(result_path)
            status = candidate_status(row, case)
            pause_reason = propagate_baseline_stop(directory, baseline)
            if stop.exists():
                refresh(directory, baseline, {**state, "status": "paused",
                    "reason": pause_reason or state.get("reason") or "stop.request exists"})
                return 2
            if status == "forecast_unavailable":
                # This is a failed accuracy/coverage point, not permission to
                # spend hours running the detailed fallback.
                refresh(directory, baseline, state)
                continue
            if worker.returncode or status != "completed":
                progress = read_json(directory / "progress" / (name + ".json")) or {}
                reason = (row or {}).get("reason") or progress.get("reason") or (
                    f"worker exited {worker.returncode} without a completed forecast; inspect logs/{name}.log")
                refresh(directory, baseline, {**state, "status": "failed", "reason": reason,
                                              "returncode": worker.returncode})
                return 1
            refresh(directory, baseline, state)
        final_state = refresh(directory, baseline, state)
        comparison = read_json(directory / "comparison.json")
        state.update(status="completed", finished_at=now(), current_case=None,
                     worker_pid=None, worker_process_id=None)
        if final_state["forecast_unavailable_candidates"] or not comparison["all_within_target"]:
            refresh(directory, baseline, {**state, "status": "completed_with_failures",
                "reason": "At least one candidate forecast or complete-request 10% check failed"})
            return 1
        refresh(directory, baseline, state)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
        default=ROOT / "artifacts" / "fast_mode_64k_5k_20261017")
    parser.add_argument("--directory", type=Path,
        default=ROOT / "artifacts" / "fast_mode_v2_20261017")
    args = parser.parse_args()
    raise SystemExit(run(args.directory, args.baseline))
