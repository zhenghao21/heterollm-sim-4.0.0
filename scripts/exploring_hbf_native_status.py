"""Read the native comparison without starting or changing any simulation.

``build_snapshot(directory)`` is also the read-only dashboard data source.
Run this file to publish progress_analysis.json and progress_analysis.md locally.
The manager owns report.md; this module deliberately never writes that file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "artifacts/exploring_hbf_native_20261019"
SCHEMA = "exploring-hbf-native-dashboard/v1"
BEIJING = timezone(timedelta(hours=8))
LABELS = {"running": "运行中", "pending": "等待", "completed": "完成",
          "needs_attention": "需要检查",
          "paused": "暂停", "failed": "失败", "capacity_rejected": "容量拒绝",
          "slo_rejected": "TPOT约束拒绝", "all_candidates_rejected": "候选均被拒绝",
          "forecast_rejected": "预测预算拒绝",
          "completed_with_rejections": "结束（含拒绝）"}
STAGE_LABELS = {"starting": "启动", "validation": "校验", "compilation": "初始编译",
    "serving_cohorts": "批次完成", "serving_complete": "服务窗口完成",
    "serving_boundary": "原生准入及PD入站", "cohort_preparation": "批次状态准备",
    "cohort_lowering": "批次任务图生成", "cohort_scheduling": "资源调度准备",
    "cohort_preview_tasks": "调度前沿预演",
    "cohort_execution": "物理执行准备", "cohort_tasks": "物理任务执行",
    "cohort_commit": "状态及Token提交", "reporting": "结果整理", "completed": "完成"}
STAGE_LABELS.update({"native_fast_sampling": "完整原生批次采样",
    "native_fast_rejected": "窗口预测预算拒绝",
    "native_fast_forecast_complete": "样本及预测窗口已保存",
    "native_layer_tasks": "原生完整层窗口采样",
    "native_layer_sample_completed": "完整层样本已保存",
    "native_layer_forecast_complete": "层样本及预测窗口已保存"})


def _read(path: Path, warnings: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        warnings.append(f"{path.name}暂时无法读取（{type(exc).__name__}），本次不填补缺失数据。")
        return {}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _age_seconds(value: Any, now: datetime) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            return None
        return max(0., (now - timestamp).total_seconds())
    except ValueError:
        return None


def _process_info(pid: Any) -> dict[str, Any]:
    result = {"pid": pid, "alive": None, "cpu_seconds": None, "rss_bytes": None,
              "created_at": None, "observation": "未知"}
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return result
    try:
        import psutil
    except ImportError:
        result["observation"] = "进程观察组件不可用，活性和资源占用未知。"
        return result
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            result["alive"] = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            cpu = process.cpu_times()
            result["cpu_seconds"] = cpu.user + cpu.system
            result["rss_bytes"] = process.memory_info().rss
            result["created_at"] = datetime.fromtimestamp(process.create_time(), BEIJING).isoformat()
        result["observation"] = "当前PID的实际进程；累计CPU时间不能单独证明近期有推进。"
    except psutil.NoSuchProcess:
        result.update(alive=False, observation="该PID当前不存在。")
    except (psutil.AccessDenied, OSError):
        result["observation"] = "无权读取进程，不能据此判断退出或卡住。"
    return result


def _baseline_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(case.get(key) for key in ("model", "workload", "input_tokens", "output_tokens",
        "gpu_count", "weight_dtype", "kv_dtype", "tp", "pp", "ep"))


def _dashboard_case(point: dict[str, Any]) -> dict[str, Any]:
    """Keep online metrics and intervals, leave detailed physical ledgers local."""
    selected = point.get("selected")
    if not selected or not selected.get("native_window_forecast"):
        return point
    forecast = selected["native_window_forecast"]
    keep = ("status", "backend", "runner_contract", "parameters", "candidate_id", "metrics",
            "started_at", "finished_at", "wall_seconds", "forecast_tokens", "sampled_tokens", "fidelity")
    brief = {key: selected[key] for key in keep if key in selected}
    brief["native_window_forecast"] = {key: forecast[key] for key in (
        "schema", "sampling_unit", "cutoff_ns", "sampled_decode_cohorts", "predicted_decode_cohorts",
        "sampled_visible_output_tokens", "predicted_visible_output_tokens", "original_window", "measurement_window", "metrics",
        "metric_bounds") if key in forecast}
    samples = selected.get("physical_layer_samples", ())
    brief["physical_layer_summary"] = {"completed_windows": len(samples),
        "executed_tasks": sum(row.get("executed_task_count", 0) for row in samples),
        "complete_cohorts": 0, "visible_output_tokens": 0}
    return {**point, "selected": brief}


def _completed_results(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for point in cases:
        selected = point.get("selected") or {}
        if point.get("status") != "completed" or selected.get("status") != "completed":
            continue
        if selected.get("backend") != "native_serving_physical":
            continue
        case, metrics = point.get("case", {}), selected.get("metrics") or {}
        tps = _number(metrics.get("tokens_per_second_per_gpu"))
        cluster = _number(metrics.get("cluster_tokens_per_second"))
        tpot = _number(metrics.get("mean_complete_request_tpot_ns"))
        if tps is None or cluster is None or tpot is None or min(tps, cluster, tpot) <= 0:
            continue
        rows.append({"case_id": point.get("case_id", case.get("case_id")),
            "model": case.get("model"), "hardware": case.get("hardware"),
            "workload": case.get("workload"), "input_tokens": case.get("input_tokens"),
            "output_tokens": case.get("output_tokens"), "gpu_count": case.get("gpu_count"),
            "batch_size": selected.get("parameters", {}).get("batch_size"),
            "tokens_per_second_per_gpu": tps, "cluster_tps": cluster,
            "tpot_ms": tpot / 1e6, "throughput_ratio_vs_hbm4": None,
            "tpot_ratio_vs_hbm4": None, "baseline_batch_size": None,
            "measurement_kind": ("conditional_forecast" if selected.get("native_window_forecast") else "physical_window"),
            "metric_bounds": (selected.get("native_window_forecast") or {}).get("metric_bounds"),
            "sampled_tokens": selected.get("sampled_tokens"), "forecast_tokens": selected.get("forecast_tokens", 0),
            "_baseline_key": (_baseline_key(case), selected.get("runner_contract"))})
    baselines = {row["_baseline_key"]: row for row in rows if row["hardware"] == "HBM4"
                 and all(value is not None for value in row["_baseline_key"][0])}
    for row in rows:
        baseline = baselines.get(row["_baseline_key"])
        if baseline:
            row["throughput_ratio_vs_hbm4"] = row["tokens_per_second_per_gpu"] / baseline["tokens_per_second_per_gpu"]
            row["tpot_ratio_vs_hbm4"] = row["tpot_ms"] / baseline["tpot_ms"]
            row["baseline_batch_size"] = baseline["batch_size"]
        del row["_baseline_key"]
    return rows


def _eta(directory: Path, warnings: list[str], override: dict[str, Any] | None) -> dict[str, Any]:
    assessment = override if override is not None else _read(directory / "eta_assessment.json", warnings)
    defaults = {"summary": "尚无完整候选，无法可靠估算整轮剩余时间。",
        "remaining_min_seconds": None, "remaining_max_seconds": None,
        "method": "仅采用完整候选的实际壁钟耗时；当前cohort任务计数不换算为整轮进度。",
        "caveats": ["30组配置共60个批量候选，运行中的任务计数只属于当前cohort。",
                    "不使用微基准、报告更新时间或已模拟纳秒数外推全轮耗时。"]}
    for key in defaults:
        if key in assessment:
            defaults[key] = assessment[key]
    if assessment.get("assessed_at"):
        defaults["assessed_at"] = assessment["assessed_at"]
        age = _age_seconds(assessment["assessed_at"], datetime.now(BEIJING))
        defaults["assessment_age_seconds"] = age
        defaults["assessment_expired"] = age is None or age > 20 * 60
        defaults["current_batch_estimates"] = ([] if defaults["assessment_expired"]
            else assessment.get("current_batch_estimates", []))
    elif isinstance(assessment.get("current_batch_estimates"), list):
        defaults["current_batch_estimates"] = []
    for key in ("remaining_min_seconds", "remaining_max_seconds"):
        value = _number(defaults[key])
        defaults[key] = value if value is not None and value >= 0 else None
    return defaults


def _optimization_snapshot(directory, warnings):
    root = directory / "actual_runtime_profile"
    data = _read(root / "summary.json", warnings)
    for row in data.get("rows", ()):
        if row.get("status") != "running":
            continue
        identity, case = row.get("id"), row.get("case_id")
        if (not isinstance(identity, str) or Path(identity).name != identity or identity in {".", ".."}
                or not isinstance(case, str) or Path(case).name != case or case in {".", ".."}):
            continue
        progress = _read(root / identity / "progress" / (case + ".json"), warnings)
        counters = progress.get("cumulative_progress") or {}
        amount = _number(counters.get("external_submitted_target_write_bytes"))
        row.update(wall_seconds=progress.get("candidate_wall_seconds", row.get("wall_seconds")),
            layer_windows=(progress.get("physical_layer_progress") or {}).get("completed_windows", 0),
            stage=progress.get("stage"), pid=progress.get("pid"),
            submitted_target_write_gib=amount / 2**30 if amount is not None else None,
            complete_imported_requests=counters.get("external_imported_requests"))
        result_path = root / identity / "result.json"
        if result_path.exists():
            result = _read(result_path, warnings)
            layer = result.get("approximation", {}).get("error_budget", {}).get("native_layer_sampling", {})
            row.update(status=result.get("status", "needs_attention"),
                wall_seconds=result.get("wall_seconds", row["wall_seconds"]),
                rejection_reason=result.get("rejection_reason"),
                layer_windows=len(result.get("physical_layer_samples", ())),
                cohort_relative_budget=(layer.get("cohort_interval") or {}).get("relative_error_budget"))
        elif _process_info(row.get("pid")).get("alive") is False:
            row.update(status="needs_attention", observation="诊断进程已退出，尚无终态结果；不自动重启。")
    return data


def build_snapshot(directory: Path | str = DEFAULT_DIRECTORY, *, eta: dict[str, Any] | None = None) -> dict[str, Any]:
    directory = Path(directory)
    now = datetime.now(BEIJING)
    warnings: list[str] = []
    status = _read(directory / "status.json", warnings)
    results = _read(directory / "results.json", warnings)
    plan = _read(directory / "plan.json", warnings)
    inputs = _read(directory / "native_inputs.json", warnings)
    manifest = _read(directory / "run_manifest.json", warnings)
    optimization = _optimization_snapshot(directory, warnings)
    configuration = {key: plan.get(key) for key in ("models", "workloads", "measurement",
        "execution_policy", "deployment_search", "phases", "sources")}
    configuration["hardware"] = {hardware: {key: value for key, value in description.items()
        if key == "per_gpu_table" or key.startswith("native_") or key == "logic_sram_status"}
        for hardware, description in inputs.get("hardware", {}).items() if isinstance(description, dict)}
    configuration["run"] = {key: manifest.get(key) for key in ("runner_contract", "backend",
        "batch_sizes", "max_parallel_configurations", "phases", "forecast_tokens", "simulation_mode", "sampling_unit",
        "fixed_deployment", "selection_scope")}
    cases = results.get("cases", [])
    if not isinstance(cases, list):
        cases = []
        warnings.append("results.json中的配置列表不可读。")
    cases = [_dashboard_case(point) for point in cases]
    status["freshness"] = {"age_seconds": _age_seconds(status.get("updated_at"), now),
                           "manager_process": _process_info(status.get("manager_pid"))}
    status["pause_requested"] = (directory / "stop.request").exists()
    def observed_progress(progress, case_id, launcher_pid=None):
        return {**progress, "case_id": case_id, "launcher_pid": launcher_pid,
            "progress_scope": "current_cohort_tasks_only" if progress.get("stage") == "cohort_tasks" else "current_native_stage_only",
            "cumulative_progress_available": isinstance(progress.get("cumulative_progress"), dict),
            "progress_age_seconds": _age_seconds(progress.get("last_progress_at"), now),
            "heartbeat_age_seconds": _age_seconds(progress.get("updated_at"), now),
            "process": _process_info(progress.get("pid"))}
    running = []
    for active in status.get("running", []):
        case_id = active.get("case_id")
        if not isinstance(case_id, str) or Path(case_id).name != case_id:
            continue
        progress = _read(directory / "progress" / (case_id + ".json"), warnings)
        running.append(observed_progress(progress, case_id, active.get("launcher_pid")))
    active_ids = {row["case_id"] for row in running}
    paused = []
    for path in sorted((directory / "progress").glob("*.json")):
        if path.stem in active_ids:
            continue
        progress = _read(path, warnings)
        if progress.get("state") == "paused":
            paused.append(observed_progress(progress, path.stem))
    point_states = Counter()
    for path in (directory / "points").glob("*.json"):
        point = _read(path, warnings)
        if point.get("status"):
            point_states[point["status"]] += 1
    candidate_states = Counter()
    for path in (directory / "candidates").glob("*.json"):
        candidate = _read(path, warnings)
        if candidate.get("status"):
            candidate_states[candidate["status"]] += 1
    fast = manifest.get("simulation_mode") == "fast"
    notes = ["本轮仅使用原生后端，按9＋6＋15顺序执行，最多两个配置并行。",
        "每组只比较批量1和8，选择通过约束者中吞吐最高的一项，不代表全局最优。快速预测与完整物理结果分别标记。",
        "比值方向固定为当前存储/HBM4；只比较同模型、同负载、同GPU数、精度及采样合同的已完成结果。",
        "当前cohort的completed/total是任务数，不是Token数、请求数或整轮完成百分比。累计Token与批次数独立记录，仅在原生提交完成后增加。",
        "累计Prefill Token只计实际执行的查询Token；PD导入历史KV不冒充Prefill计算。累计输出Token来自真实可见Token提交事件，包含真实首Token。",
        "壁钟分项按原生阶段切换时刻独占累计，包含该阶段未完成的当前区间；它是仿真器运行耗时，不是硬件延迟。旧尝试未记录的累计Token和分项保持未知。",
        ("物理层或批次样本与条件预测窗口分别记录。层样本不计完整批次或Token。10%为条件模型预算，未执行窗口没有物理轨迹，不是全窗口实测误差保证；预测拒绝指标为空，不详细回退。"
            if fast else "本轮有界物理窗口没有预测尾段；10%预测预算未放宽，也不表示已证明与真实硬件误差小于10%。"),
        "旧详细参考和已经取消的5K、10K配置均不重跑。"]
    assessment = _eta(directory, warnings, eta)
    if not eta and not (directory / "eta_assessment.json").exists() and candidate_states:
        assessment["summary"] = "已有候选终态，但不同模型和存储的代表性完整耗时尚不足，无法可靠估算整轮剩余时间。"
    if status.get("state") in {"completed", "completed_with_rejections"}:
        assessment = {"summary": "本轮已结束，无剩余扫描时间。", "remaining_min_seconds": 0.,
            "remaining_max_seconds": 0., "method": "以管理器真实终态为准。", "caveats": []}
    elif status.get("state") == "paused":
        assessment = {"summary": "扫描已暂停；优化并验收前不恢复，不用暂停前的局部速度推算完成时间。",
            "remaining_min_seconds": None, "remaining_max_seconds": None,
            "method": "恢复后的完整真实候选耗时通过验收后再评估。", "caveats": []}
    return {"schema": SCHEMA, "scan_id": directory.name, "observed_at": now.isoformat(),
        "status": status, "cases": cases, "running": running, "paused": paused,
        "configuration": configuration,
        "optimization": optimization if optimization.get("schema") == "native-runtime-diagnostics/v1" else None,
        "completed_results": _completed_results(cases),
        "saved_point_counts": dict(point_states), "saved_candidate_counts": dict(candidate_states),
        "eta": assessment, "notes": notes + warnings}


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def render_markdown(snapshot: dict[str, Any]) -> str:
    status = snapshot["status"]
    counts = status.get("counts", {})
    lines = ["# 原生30点对照进度与已完成结果", "",
        f"观察时间：{snapshot['observed_at']}。状态：{LABELS.get(status.get('state'), status.get('state', '未知'))}。",
        f"已验收完成配置：{counts.get('completed', 0)}/{status.get('total_cases', 30)}；"
        f"已验收完成候选：{snapshot['saved_candidate_counts'].get('completed', 0)}/{status.get('total_candidates', 60)}。",
        "", "## 正在运行", "",
        "|配置|批量|实际worker PID|壁钟小时|当前阶段|当前cohort任务|累计CPU小时|RSS GiB|",
        "|---|---:|---:|---:|---|---:|---:|---:|"]
    for row in snapshot["running"]:
        process = row.get("process", {})
        wall, cpu, rss = (_number(row.get("wall_seconds")), _number(process.get("cpu_seconds")),
                          _number(process.get("rss_bytes")))
        lines.append(f"|{row.get('case_id')}|{row.get('batch_size', '—')}|{row.get('pid', '—')}|"
            f"{_fmt(wall / 3600 if wall is not None else None, 2)}|{STAGE_LABELS.get(row.get('stage'), '原生阶段')}|"
            f"{row.get('completed', '—')}/{row.get('total', '—')}|"
            f"{_fmt(cpu / 3600 if cpu is not None else None, 2)}|"
            f"{_fmt(rss / 2**30 if rss is not None else None, 2)}|")
    observed = snapshot["running"] + snapshot.get("paused", [])
    if snapshot.get("paused"):
        lines += ["", "## 已暂停尝试", ""]
        for row in snapshot["paused"]:
            wall = _number(row.get("wall_seconds"))
            lines.append(f"- {row['case_id']}：批量{row.get('batch_size', '—')}，已暂停；"
                f"运行{_fmt(wall / 3600 if wall is not None else None, 2)}小时，保留该次实际进度，不计为完成候选。")
    lines += ["", "## 累计Token与批次", "",
        "|配置|已完成批次|实际Prefill Token|实际可见输出Token|已完成Decode批次/窗口目标|预热/测量Decode批次|",
        "|---|---:|---:|---:|---:|---:|"]
    for row in observed:
        counter = row.get("cumulative_progress") or {}
        lines.append(f"|{row['case_id']}|{counter.get('cohorts', '未知')}|"
            f"{counter.get('prefill_compute_tokens', '未知')}|{counter.get('visible_output_tokens', '未知')}|"
            f"{counter.get('decode_cohorts', '未知')}/{counter.get('target_decode_cohorts', '未知')}|"
            f"{counter.get('warmup_decode_cohorts', '未知')}/{counter.get('measurement_decode_cohorts', '未知')}|")
    lines += ["", "## 真实壁钟耗时分项", "",
        "|配置|原生阶段|累计秒数|", "|---|---|---:|"]
    for row in observed:
        phases = row.get("wall_breakdown_seconds") or {}
        if not phases:
            lines.append(f"|{row['case_id']}|旧尝试没有记录，保持未知|—|")
        for phase, seconds in phases.items():
            lines.append(f"|{row['case_id']}|{STAGE_LABELS.get(phase, '原生内部阶段')}|{_fmt(seconds)}|")
    lines += ["", "上述任务计数仅属于当前cohort，不代表整个候选的完成比例。累计Token和已完成批次不随下一批任务归零。", "",
        "## 已完成配置", "",
        "|模型|存储|负载|输入/输出Token|选定批量|tokens/s/GPU|集群tokens/s|TPOT ms|吞吐/HBM4|TPOT/HBM4|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in snapshot["completed_results"]:
        lines.append(f"|{row['model']}|{row['hardware']}|{row['workload']}|"
            f"{row['input_tokens']}/{row['output_tokens']}|{row['batch_size']}|"
            f"{_fmt(row['tokens_per_second_per_gpu'])}|{_fmt(row['cluster_tps'])}|"
            f"{_fmt(row['tpot_ms'])}|{_fmt(row['throughput_ratio_vs_hbm4'])}|{_fmt(row['tpot_ratio_vs_hbm4'])}|")
    if not snapshot["completed_results"]:
        lines += ["", "尚无真实完成配置，吞吐、TPOT及相对HBM4比值均不可用；不以局部批次或历史旁路结果填充。"]
    lines += ["", "## 剩余时间", "", snapshot["eta"]["summary"], "", snapshot["eta"]["method"], ""]
    lines += [f"- {text}" for text in snapshot["eta"].get("caveats", [])]
    if snapshot["eta"].get("assessed_at"):
        lines += ["", f"耗时评估时间：{snapshot['eta']['assessed_at']}。"]
        if snapshot["eta"].get("assessment_expired"):
            lines += ["评估已超过20分钟，仅保留当时观察，不显示当前批次的剩余时间预测。"]
        else:
            for estimate in snapshot["eta"].get("current_batch_estimates", []):
                if not isinstance(estimate, dict):
                    continue
                minimum = _number(estimate.get("remaining_min_seconds"))
                maximum = _number(estimate.get("remaining_max_seconds"))
                if minimum is not None and maximum is not None:
                    lines += [f"- {estimate.get('case_id', '运行配置')}：仅当前cohort预计尚需"
                        f"{minimum / 3600:.2f}～{maximum / 3600:.2f}小时；不能当作整个候选或整轮剩余时间。"]
    optimization = snapshot.get("optimization") or {}
    if optimization:
        lines += ["", "## 优化诊断（不计正式结果）", "",
            "|硬件|负载|批量|诊断壁钟秒|结果|层窗口数|整批条件预算|PD已提交目标写GiB|", "|---|---|---:|---:|---|---:|---:|---:|"]
        for row in optimization.get("rows", ()):
            budget = _number(row.get("cohort_relative_budget"))
            lines.append(f"|{row['hardware']}（{row.get('label', '独立诊断')}）|{row['workload']}|{row['batch_size']}|{_fmt(row['wall_seconds'])}|"
                f"{LABELS.get(row['status'], row['status'])}|{row['layer_windows']}|"
                f"{_fmt(100*budget) + '%' if budget is not None else '—'}|{_fmt(row.get('submitted_target_write_gib'))}|")
        lines += ["", *[f"- {text}" for text in optimization.get("findings", ())]]
    lines += ["", "## 统计口径", ""] + [f"- {text}" for text in snapshot["notes"]]
    return "\n".join(lines) + "\n"


def write_snapshot(directory: Path | str = DEFAULT_DIRECTORY, *, eta: dict[str, Any] | None = None) -> dict[str, Any]:
    directory = Path(directory)
    snapshot = build_snapshot(directory, eta=eta)
    for filename, content in (("progress_analysis.json", json.dumps(snapshot, ensure_ascii=False, indent=2)),
                              ("progress_analysis.md", render_markdown(snapshot))):
        directory.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix="." + filename, suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temp, directory / filename)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    snapshot = write_snapshot(args.directory)
    print(json.dumps({"scan_id": snapshot["scan_id"], "observed_at": snapshot["observed_at"],
        "completed_cases": len(snapshot["completed_results"]), "running": len(snapshot["running"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
