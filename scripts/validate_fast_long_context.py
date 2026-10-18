"""Run only the eight authorized 64K/5K fast-mode validation requests.

This is an independent diagnostic queue.  It never reads or resumes the
formal 72-point scan, and never substitutes a short detailed reference.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_fast_mode import compare_case, compare_pair, write_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen2_5-0_5b", "qwen2_5-7b")
ARCHITECTURES = ("HBM", "HBF")
PROMPT, OUTPUT = 65536, 5120
CASES = tuple((model, arch, mode) for mode in ("fast", "detailed")
              for model in MODELS for arch in ARCHITECTURES)


def now():
    return datetime.now(timezone.utc).isoformat()


def case_name(case):
    model, arch, mode = case
    return f"{model}_{arch}_P{PROMPT}_D{OUTPUT}_{mode}"


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def completed(row, case):
    return bool(row and all(row.get(key) == value for key, value in zip(
        ("model_id", "architecture", "requested_mode"), case))
        and row.get("prompt_tokens") == PROMPT and row.get("output_tokens") == OUTPUT
        and row.get("visible_output_tokens") == OUTPUT
        and row.get("request_status") in {"finished", "forecast_completed"})


def process_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_bool, ctypes.c_uint]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def manager_lock(directory):
    # The OS releases this lock on exit; no stale PID file can silently
    # authorize a second queue.  The small persistent file is reusable.
    with (directory / "manager.lock").open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def refresh(directory, state):
    rows = {case: row for case in CASES
            if completed(row := read_json(directory / (case_name(case) + ".json")), case)}
    state = {**state, "completed_runs": len(rows), "total_runs": 8, "updated_at": now(),
        "prompt_tokens": PROMPT, "output_tokens": OUTPUT,
        "purpose": "limited fast-mode validation; formal scan remains paused"}
    current = state.get("current_case")
    progress = read_json(directory / "progress" / (current + ".json")) if current else None
    state["worker_progress"] = progress
    write_json(directory / "status.json", state)
    comparisons = {}
    for model in MODELS:
        group = [rows[case] for case in CASES if case[0] == model and case in rows]
        if len(group) == 4:
            comparison = compare_pair(group)
            comparisons[model] = comparison
            write_json(directory / (model + "_comparison.json"), comparison)
    write_json(directory / "comparison.json", {
        "completed_runs": len(rows), "total_runs": 8, "comparisons": comparisons,
        "all_within_target": (all(row["all_within_target"] for row in comparisons.values())
                              if len(comparisons) == 2 else None)})
    lines = ["# 64K / 5K 快速模式完整对照", "",
        f"更新：{state['updated_at']}；状态：{state['status']}；已完成 {len(rows)}/8。", "",
        "Qwen2.5-0.5B / 7B；输入 65,536、输出 5,120；单请求 BF16，B200。",
        "详细参考完整执行 5,119 步 Decode；四个配置分别对照详细／快速模式。",
        "各次独立进程串行运行，不绑单核，不启用 profiler 或逐事务捕获。正式扫描保持暂停。", ""]
    if current:
        lines.append(f"当前：`{current}`；管理 PID={state.get('manager_pid')}，工作 PID={state.get('worker_pid')}。")
    if progress:
        lines += [f"阶段：{progress.get('phase')}；Prefill {progress.get('prefill_steps', 0)}/32 块；"
                  f"真实 Decode {progress.get('decode_steps', 0)}/5119 步；"
                  f"已运行 {progress.get('wall_seconds', 0):.1f} 秒。"]
        rate = progress.get("recent_16_decode_step_mean_seconds")
        if rate:
            lines.append(f"最近真实 Decode 均值 {rate:.3f} 秒/步。")
        if progress.get("fallback_reasons"):
            lines.append("快速模式转详细的原因：" + ", ".join(progress["fallback_reasons"]))
    if state.get("reason"):
        lines += ["", "停止或失败原因：" + str(state["reason"])]
    lines += ["", "| 模型 | 存储 | 详细壁钟 s | 快速壁钟 s | 整请求加速 | Decode 加速 | TTFT 误差 | TPOT 误差 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for arch in ARCHITECTURES:
            detail, fast = rows.get((model, arch, "detailed")), rows.get((model, arch, "fast"))
            times = ["待完成" if row is None else f"{row['wall_seconds']:.3f}" for row in (detail, fast)]
            values = ["待完成"] * 4
            if detail and fast:
                comparison = compare_case(detail, fast)
                values = ["不可比较" if comparison["speedup"][key] is None else f"{comparison['speedup'][key]:.3f}×"
                          for key in ("total_wall", "decode_wall")]
                values += ["不可比较" if comparison["relative_errors"][key] is None else f"{comparison['relative_errors'][key]:.3%}"
                           for key in ("ttft_ns", "tpot_ns")]
            lines.append("| " + " | ".join([model, arch, *times, *values]) + " |")
    lines += ["", "| 模型 | 详细 TPOT HBF/HBM | 快速 TPOT HBF/HBM | 比值误差 | 10% 检查 |",
              "| --- | ---: | ---: | ---: | --- |"]
    for model in MODELS:
        if model not in comparisons:
            lines.append(f"| {model} | 待完成 | 待完成 | 待完成 | 待完成 |")
        else:
            ratio = comparisons[model]["ratio_comparison"]["tpot_ns"]
            lines.append(f"| {model} | {ratio['hbf_over_hbm']['detailed']:.6f} | "
                f"{ratio['hbf_over_hbm']['fast']:.6f} | {ratio['relative_error']:.3%} | "
                f"{'通过' if comparisons[model]['all_within_target'] else '未通过'} |")
    lines += ["", "## 阶段耗时（实际完整测量）", "",
        "| 模型／存储／模式 | 初始化到 serving s | Prefill s | 真实 Decode s | 最后 cohort 后 s | 端点提取 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in CASES:
        if case in rows:
            timing = rows[case].get("timing_breakdown", {})
            values = [timing.get(key) for key in ("setup_to_serving_start_seconds",
                "prefill_cohorts_wall_seconds", "decode_cohorts_wall_seconds",
                "after_last_cohort_seconds", "endpoint_extraction_seconds")]
            lines.append("| " + " | ".join(["/".join(case), *[
                "未记录" if value is None else f"{value:.3f}" for value in values]]) + " |")
    lines += ["", "## 尚未加速的部分", "",
        "- 64K Prefill 的 32 个分块全部详细执行；快速模式仍实算默认 56 步 Decode。",
        "- 5119/56≈91.41，仅为 Decode 工作量减少上限；计入 Prefill 后整请求加速更低。",
        "- 逐 burst／逐页数值执行和状态适配仍占真实采样成本，图绑定及已优化的页归并占比很小。",
        "- 后续候选是按 K/V 凑页周期自适应采样，以及 Prefill 数值/适配优化；本轮先保持当前模式做完整验收。",
        "- 已发现新块 page-0 擦除周期的模型风险；完整对照完成前不宣称所有误差≤10%。",
        "- 本诊断直接提取请求端点，避免构建最多 2000 条大批次报告；不改变仿真结果。", ""]
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


def run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "progress"):
        (directory / name).mkdir(exist_ok=True)
    stop = directory / "stop.request"
    state = {"status": "running", "manager_pid": os.getpid(), "started_at": now()}
    with manager_lock(directory):
        previous = read_json(directory / "status.json") or {}
        if previous.get("status") in {"running", "stopping"} and process_alive(previous.get("worker_pid")):
            raise RuntimeError("Previous diagnostic worker is still alive; refusing duplicate execution")
        previous_case = previous.get("current_case")
        previous_progress = (read_json(directory / "progress" / (previous_case + ".json"))
                             if previous_case else None)
        if (previous_progress and previous_progress.get("status") in {"starting", "running"}
                and process_alive(previous_progress.get("process_id"))):
            raise RuntimeError("Previous retained-runtime child is still alive; refusing duplicate execution")
        refresh(directory, state)
        for case in CASES:
            if stop.exists():
                refresh(directory, {**state, "status": "paused", "reason": "stop.request exists"})
                return 2
            name = case_name(case)
            result = directory / (name + ".json")
            if completed(read_json(result), case):
                continue
            model, arch, mode = case
            command = [sys.executable, str(ROOT / "scripts" / "benchmark_fast_mode.py"),
                "--model", model, "--architecture", arch, "--mode", mode,
                "--prompt", str(PROMPT), "--output", str(OUTPUT), "--result", str(result),
                "--progress-file", str(directory / "progress" / (name + ".json")),
                "--stop-file", str(stop), "--report-timing"]
            with (directory / "logs" / (name + ".log")).open("a", encoding="utf-8") as output:
                worker = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                state.update(current_case=name, worker_pid=worker.pid, case_started_at=now())
                while worker.poll() is None:
                    state["status"] = "stopping" if stop.exists() else "running"
                    refresh(directory, state)
                    time.sleep(10)
            if worker.returncode or not completed(read_json(result), case):
                progress = read_json(directory / "progress" / (name + ".json")) or {}
                state.update(status="paused" if stop.exists() else "failed",
                    returncode=worker.returncode,
                    reason=progress.get("reason", f"worker exited {worker.returncode}; inspect logs/{name}.log"))
                refresh(directory, state)
                return 2 if stop.exists() else 1
            refresh(directory, state)
        state.update(status="completed", finished_at=now(), current_case=None, worker_pid=None)
        refresh(directory, state)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
        default=ROOT / "artifacts" / "fast_mode_64k_5k_20261017")
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.directory.resolve()))
