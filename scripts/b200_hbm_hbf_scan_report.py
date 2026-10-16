"""Summarize recorded B200 HBM/HBF scan points without running simulations.

Usage: python scripts/b200_hbm_hbf_scan_report.py OUTPUT_DIRECTORY

The directory contains manifest.json and one result row per points/*.json file.
Only completed, positive measurements are plotted. Missing model points break
curves instead of becoming zeros or interpolated observations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


BASE_COLUMNS = (
    "point_id", "model_id", "architecture", "prompt_tokens", "output_tokens",
    "status", "ttft_ns", "tpot_ns", "ttft_s", "tpot_ms", "wall_seconds", "reason",
)
TERMINAL_STATUSES = {"completed", "failed", "capacity_rejected"}
COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#4B3E8A", "#1C7B78", "#8C564B", "#68732D",
)
BEIJING = timezone(timedelta(hours=8))


def _key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (row["model_id"], row["architecture"],
            int(row["prompt_tokens"]), int(row["output_tokens"]))


def _positive_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _markdown(value: Any) -> str:
    return str(_cell(value)).replace("|", "\\|").replace("\n", " ")


def _tokens(value: int) -> str:
    return f"{value // 1024}K" if value % 1024 == 0 else str(value)


def _local_time(value: str | None) -> str:
    if not value:
        return "未记录"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(BEIJING).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    # A Windows reader can briefly be denied while a worker publishes progress.
    # Retry the read; permanent errors still stop the report with their cause.
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return None
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def _latest_log_progress(path: Path) -> dict[str, Any]:
    """Read only the latest attempt, resetting counters when its clock resets."""
    state: dict[str, Any] = {"completed_cohorts": None, "latest": None}
    previous_wall = None
    try:
        stream = path.open(encoding="utf-8-sig", errors="replace")
    except FileNotFoundError:
        return state
    with stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # Tracebacks and an in-flight final line are not progress.
            if not isinstance(event, dict) or "stage" not in event:
                continue
            wall = event.get("wall_seconds")
            reset = (event["stage"] == "control_plane" and event.get("completed") == 0)
            reset |= (isinstance(wall, (int, float)) and previous_wall is not None
                      and wall < previous_wall - 0.001)
            if reset:
                state["completed_cohorts"] = 0
            if isinstance(wall, (int, float)):
                previous_wall = wall
            state["latest"] = event
            if event["stage"] == "serving_cohorts" and isinstance(event.get("completed"), int):
                state["completed_cohorts"] = event["completed"]
    return state


def _running_progress(output_dir: Path, point: dict[str, Any], chunk_tokens: int) -> dict[str, Any]:
    point_id = point["point_id"]
    progress = _read_optional_json(output_dir / "progress" / f"{point_id}.json") or {}
    log_path = output_dir / "logs" / f"{point_id}.log"
    log = _latest_log_progress(log_path)
    # The manager can announce a retry before that worker has replaced its files.
    started = datetime.fromisoformat(point["started_at"]) if point.get("started_at") else None
    if started is not None:
        if progress.get("updated_at") and datetime.fromisoformat(progress["updated_at"]) < started:
            progress = {}
        if log_path.exists() and log_path.stat().st_mtime < started.timestamp():
            log = {"completed_cohorts": None, "latest": None}
    event = progress.get("progress", {})
    latest = log.get("latest") or {}
    if not event or latest.get("wall_seconds", -1) > progress.get("wall_seconds", -1) + 0.02:
        event = latest
    cohorts = log["completed_cohorts"]
    if event.get("stage") == "serving_cohorts":
        cohorts = event.get("completed", cohorts)
    prompt = int(point["prompt_tokens"])
    chunks = math.ceil(prompt / chunk_tokens)
    if cohorts is None:
        phase, prefill, decode = "准备启动", "尚未记录已完成块", "尚未记录"
    else:
        done = min(cohorts, chunks)
        input_done = min(done * chunk_tokens, prompt)
        prefill = f"{done}/{chunks} 块；{input_done:,}/{prompt:,} token（Prefill {input_done / prompt:.1%}）"
        phase = f"Prefill 第 {done + 1}/{chunks} 块" if done < chunks else "Prefill 完成；Decode 阶段"
        decode = "尚未进入" if cohorts < chunks else f"已完成 {cohorts - chunks} 个后续步骤"
    stage = event.get("stage", "")
    message = str(event.get("message", ""))
    if stage == "cohort_tasks":
        scope = "实际物理执行" if "连续物理状态" in message else "准备/估算"
        detail = f"{scope}：当前批次任务 {event.get('completed', '?')}/{event.get('total', '?')}"
    elif stage == "serving_cohorts":
        detail = "上一批次已完成，等待下一批次进度"
    else:
        detail = {"control_plane": "准备控制配置", "validation": "验证配置",
                  "serving_complete": "服务执行完成，正在整理结果"}.get(stage, message or "等待进度记录")
    return {"phase": phase, "prefill": prefill, "decode": decode, "detail": detail,
            "updated_at": _local_time(progress.get("updated_at")), "completed_cohorts": cohorts}


def _atomic_text(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".report-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        for attempt in range(3):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.1)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_inputs(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    models = manifest["models"]
    expected = list(itertools.product(
        [model["model_id"] for model in models], manifest["architectures"],
        manifest["prompts"], manifest["outputs"],
    ))
    order = {key: index for index, key in enumerate(expected)}
    if len(order) != len(expected):
        raise ValueError("Manifest contains duplicate model/architecture/length combinations")
    rows_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for source in sorted((output_dir / "points").glob("*.json")):
        row = json.loads(source.read_text(encoding="utf-8-sig"))
        key = _key(row)
        if key not in order:
            raise ValueError(f"Point is outside the manifest: {source.name}")
        if key in rows_by_key:
            raise ValueError(f"Duplicate point for {key}: {source.name}")
        if row.get("status") == "completed":
            for metric in ("ttft_ns", "tpot_ns"):
                if not _positive_number(row.get(metric)):
                    raise ValueError(f"Completed point has invalid {metric}: {source.name}")
        rows_by_key[key] = row
    rows = sorted(rows_by_key.values(), key=lambda row: order[_key(row)])
    return manifest, rows, len(expected)


def _write_csv(output_dir: Path, completed: list[dict[str, Any]]) -> None:
    nested_columns = {key for row in completed for key, value in row.items()
                      if isinstance(value, (dict, list, tuple))}
    extra_columns = sorted({key for row in completed for key in row}
                           - set(BASE_COLUMNS) - nested_columns)
    columns = [*BASE_COLUMNS, *extra_columns]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in completed:
            values = dict(row, ttft_s=row["ttft_ns"] / 1e9, tpot_ms=row["tpot_ns"] / 1e6)
            writer.writerow({key: values.get(key) for key in columns})


def _plot(output_dir: Path, manifest: dict[str, Any], completed: list[dict[str, Any]], total: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    models = sorted(manifest["models"], key=lambda model: float(model["parameters_b"]))
    x = [float(model["parameters_b"]) for model in models]
    by_key = {_key(row): row for row in completed}
    length_pairs = list(itertools.product(manifest["prompts"], manifest["outputs"]))
    styles = {"HBM": ("-", "o"), "HBF": ("--", "s")}
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#263447", "text.color": "#263447",
        "xtick.color": "#46556A", "ytick.color": "#46556A",
        "axes.edgecolor": "#ADBAC8", "savefig.facecolor": "white",
        "svg.fonttype": "none", "lines.linewidth": 1.7,
    }):
        figure, axes = plt.subplots(1, 2, figsize=(14, 7.6))
        for axis, metric, scale, title, ylabel in (
            (axes[0], "ttft_ns", 1e9, "Time to first token", "TTFT (s)"),
            (axes[1], "tpot_ns", 1e6, "Time per output token", "TPOT (ms/token)"),
        ):
            observed = []
            for pair_index, (prompt_tokens, output_tokens) in enumerate(length_pairs):
                color = COLORS[pair_index % len(COLORS)]
                for architecture in manifest["architectures"]:
                    linestyle, marker = styles[architecture]
                    values = []
                    for model in models:
                        row = by_key.get((model["model_id"], architecture, prompt_tokens, output_tokens))
                        values.append(row[metric] / scale if row else math.nan)
                    observed.extend(value for value in values if math.isfinite(value))
                    axis.plot(x, values, color=color, linestyle=linestyle, marker=marker,
                              markersize=5, markerfacecolor="white" if architecture == "HBF" else color,
                              markeredgewidth=1.2, alpha=0.9)
            axis.set_title(title, fontweight="bold", pad=12)
            axis.set_xlabel("Model parameters (B)", labelpad=9)
            axis.set_ylabel(ylabel, labelpad=9)
            axis.set_xticks(x, [f"{value:g}" for value in x])
            axis.set_yscale("log")
            axis.set_axisbelow(True)
            axis.grid(axis="y", which="major", color="#DDE4EB", linewidth=0.8)
            axis.grid(axis="y", which="minor", color="#EFF2F5", linewidth=0.5)
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.ticklabel_format(axis="y", style="plain")
            if observed and min(observed) == max(observed):
                axis.set_ylim(observed[0] / 1.6, observed[0] * 1.6)
            axis.margins(x=0.06)

        title = f"B200 single-request baseline | {len(completed)}/{total} points completed"
        figure.suptitle(title, fontsize=17, fontweight="bold", y=0.965)
        figure.text(0.5, 0.914, "HBM3E vs. OCP HBF 0.7.0  |  BF16 weights and KV  |  K = 1024 tokens",
                    ha="center", fontsize=10.5, color="#57667B")
        lengths = [Line2D([0], [0], color=COLORS[index % len(COLORS)], linewidth=2.8,
                          label=f"P {_tokens(prompt)} / D {_tokens(decode)}")
                   for index, (prompt, decode) in enumerate(length_pairs)]
        figure.legend(handles=lengths, title="Input (P) / output (D) tokens",
                      loc="lower center", bbox_to_anchor=(0.43, 0.045), ncol=3,
                      frameon=False, columnspacing=2.2, handlelength=2.2, fontsize=9.5,
                      title_fontsize=10)
        architectures = [Line2D([0], [0], color="#34465D", linestyle=styles[name][0],
                                marker=styles[name][1], markerfacecolor="white" if name == "HBF" else "#34465D",
                                label=name)
                         for name in manifest["architectures"]]
        figure.legend(handles=architectures, loc="lower right", bbox_to_anchor=(0.96, 0.095),
                      frameon=False, title="Memory placement", handlelength=3)
        figure.text(0.5, 0.018, "Missing, failed and capacity-rejected points are omitted; gaps are not interpolated.",
                    ha="center", fontsize=9, color="#637187")
        figure.subplots_adjust(left=0.075, right=0.975, bottom=0.30, top=0.84, wspace=0.25)
        figure.savefig(output_dir / "comparison.png", dpi=180)
        figure.savefig(output_dir / "comparison.svg")
        plt.close(figure)


def _parameter_rows(manifest: dict[str, Any]) -> list[tuple[str, Any]]:
    parameters = manifest.get("parameters", {})
    result = []
    labels = {
        "gpu_preset": "GPU 预设", "gpu_compute_peak_bf16_tflops": "GPU BF16 峰值（TFLOPS）",
        "request_count": "请求数", "batch_size": "Batch size",
        "weights_dtype": "权重精度", "kv_dtype": "KV 精度", "activation_dtype": "激活精度",
        "prefill_chunk_tokens": "Prefill 分块（token）", "weights_resident": "权重预驻留",
        "hbm_physical_preset": "HBM 物理预设", "hbm_capacity_bytes": "HBM 容量（byte）",
        "hbm_interface_bandwidth_gb_s": "HBM 接口带宽（GB/s）",
        "hbf_physical_preset": "HBF 物理预设", "host_staging_capacity_bytes": "HOST 暂存容量（byte）",
    }
    for key, label in labels.items():
        if key in parameters:
            value = parameters[key]
            result.append((label, "是" if value is True else "否" if value is False else value))
    for group, labels in (
        ("hbf_parameters", {"capacity_bytes": "HBF 容量（byte）", "array_units": "HBF 阵列单元数",
                            "read_bandwidth_gb_s": "HBF 读取带宽（GB/s）",
                            "write_bandwidth_gb_s": "HBF 写入带宽（GB/s）",
                            "page_bytes": "HBF 编程页（byte）",
                            "read_granularity_bytes": "HBF 读取粒度（byte）",
                            "page_read_ns": "HBF 读页延迟（ns）",
                            "page_program_ns": "HBF 编程延迟（ns）",
                            "block_erase_ns": "HBF 擦除延迟（ns）"}),
        ("host_parameters", {"cpu_cores": "CPU 核数", "cpu_frequency_ghz": "CPU 频率（GHz）",
                             "dram_capacity_bytes": "HOST DRAM 容量（byte）",
                             "dram_bandwidth_gb_s": "HOST DRAM 带宽（GB/s）",
                             "pcie_generation": "PCIe 代际", "pcie_lanes": "PCIe 通道数"}),
    ):
        values = parameters.get(group, {})
        result.extend((label, values[key]) for key, label in labels.items() if key in values)
    return result


def _write_markdown(output_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]],
                    counts: dict[str, int], queue_status: dict[str, Any] | None) -> None:
    model_text = "、".join(f"{model['label']} ({model['parameters_b']:g}B)" for model in manifest["models"])
    lines = [
        "# B200 HBM/HBF 单请求扫描", "",
        f"报告刷新时间：**{datetime.now(BEIJING).strftime('%Y-%m-%d %H:%M:%S')}（北京时间 UTC+08:00）**。", "",
        f"已完成 **{counts['completed']}/{counts['total']}** 个配置点；"
        f"失败 {counts['failed']} 个；容量不满足 {counts['capacity_rejected']} 个；"
        f"尚未完成 {counts['pending']} 个。", "",
    ]
    if queue_status is not None:
        status_label = {"running": "运行中", "completed": "已结束", "paused": "已暂停", "needs_attention": "需要处理"}.get(
            queue_status.get("status"), queue_status.get("status", "未知"))
        lines.extend([f"队列状态：**{_markdown(status_label)}**；"
                      f"队列记录已完成 {queue_status.get('completed', 0)}/{queue_status.get('total', counts['total'])}；"
                      f"等待运行 {queue_status.get('pending', 0)} 个。"
                      f"队列更新时间：{_markdown(_local_time(queue_status.get('updated_at')))}（北京时间）。", ""])
        if queue_status.get("reason"):
            reason = {"stop_requested": "按要求暂停，未自动恢复扫描"}.get(
                queue_status["reason"], queue_status["reason"])
            lines.extend([f"队列说明：{_markdown(reason)}。", ""])
        if queue_status.get("optimization_report"):
            lines.extend([f"仿真器优化结果：[耗时对比与适用范围]({queue_status['optimization_report']})。", ""])
        running = queue_status.get("running", [])
        if running:
            lines.extend(["| 正在运行的模型 | 存储 | 输入/输出 | 开始时间 |",
                          "| --- | --- | --- | --- |"])
            for point in running:
                lines.append(f"| {_markdown(point['model_id'])} | {_markdown(point['architecture'])} | "
                             f"{_tokens(point['prompt_tokens'])}/{_tokens(point['output_tokens'])} | "
                             f"{_markdown(_local_time(point.get('started_at')))} |")
            lines.append("")
            chunk_tokens = int(manifest.get("parameters", {}).get("prefill_chunk_tokens", 2048))
            lines.extend(["| 存储 / 模型 | 当前阶段 | 已完成 Prefill | Decode | 当前批次细节 | 进度记录时间（北京时间） |",
                          "| --- | --- | --- | --- | --- | --- |"])
            for point in running:
                detail = _running_progress(output_dir, point, chunk_tokens)
                lines.append("| " + " | ".join(_markdown(value) for value in (
                    f"{point['architecture']} / {point['model_id']}", detail["phase"], detail["prefill"],
                    detail["decode"], detail["detail"], detail["updated_at"],
                )) + " |")
            lines.extend(["", "Prefill 只在整个批次完成后增加已完成块数。当前批次任务数是局部工作进度，"
                          "不代表整点完成百分比；Decode 后续步骤数也不直接等同于已输出 token 数。"
                          "细粒度记录若没有实际执行标记，则显示为准备/估算。", ""])
    lines.extend([
        "| 参数 | 设置 |", "| --- | --- |",
        f"| 模型 | {_markdown(model_text)} |",
        f"| 存储策略 | {_markdown('、'.join(manifest['architectures']))} |",
        f"| 单请求输入 | {', '.join(_tokens(int(value)) for value in manifest['prompts'])} token |",
        f"| 单请求输出 | {', '.join(_tokens(int(value)) for value in manifest['outputs'])} token |",
        "| K 单位 | 1024 token |",
    ])
    lines.extend(f"| {_markdown(label)} | {_markdown(value)} |" for label, value in _parameter_rows(manifest))
    lines.extend(["", "完整硬件参数、点位清单和指标定义见 [manifest.json](manifest.json)。",
                  "HBF 的器件延迟采用仿真预设值，不代表 OCP 规定的器件性能。", "",
                  "HBM3E 使用标准参考模型；当前 24Gb × 8H 组织未套用只支持 32Gb × 16H 的额外时序包。",
                  "独立 RoPE 每次生成并写入按 Q/K 广播展开的 sin/cos 表，两种部署均计入这项开销。",
                  "超过模型原生上下文的长度仅用于性能外推，不验证输出质量。", "",
                  "数据单位：原始 TTFT/TPOT 为 ns；图中 TTFT 为秒，TPOT 为毫秒/token。", ""])
    if counts["completed"]:
        lines.extend(["![TTFT 和 TPOT 扫描结果](comparison.png)", "",
                      "每个面板用颜色区分 9 种输入/输出长度组合，用实线表示 HBM、虚线表示 HBF。"
                      "横轴为模型参数量，纵轴为对数刻度；未完成和失败点留空。", ""])
    else:
        lines.extend(["当前没有已完成的测量点，尚未生成曲线图。", ""])
    unsuccessful = [row for row in rows if row.get("status") in {"failed", "capacity_rejected"}]
    if unsuccessful:
        lines.extend(["## 未成功的配置", "", "| 模型 | 存储 | 输入/输出 | 状态 | 原因 |", "| --- | --- | --- | --- | --- |"])
        for row in unsuccessful:
            lines.append(f"| {_markdown(row['model_id'])} | {_markdown(row['architecture'])} | "
                         f"{_tokens(row['prompt_tokens'])}/{_tokens(row['output_tokens'])} | "
                         f"{_markdown(row['status'])} | {_markdown(row.get('reason', ''))} |")
        lines.append("")
    lines.extend(["文件：[summary.csv](summary.csv) 仅含已完成点的主指标和其他标量；"
                  "[results.json](results.json) 保留全部结果、详细数据和参数。", ""])
    _atomic_text(output_dir / "report.md", "\n".join(lines))


def _counts(rows: list[dict[str, Any]], total: int) -> dict[str, int]:
    counts = {"total": total, **{status: sum(row.get("status") == status for row in rows)
                                for status in ("completed", "failed", "capacity_rejected")}}
    counts["pending"] = total - sum(counts[status] for status in TERMINAL_STATUSES)
    return counts


def refresh_markdown(output_dir: str | Path) -> dict[str, int]:
    """Refresh live progress only; leave figures, CSV and result JSON untouched."""
    output_dir = Path(output_dir)
    manifest, rows, total = _read_inputs(output_dir)
    queue_status = _read_optional_json(output_dir / "status.json")
    counts = _counts(rows, total)
    _write_markdown(output_dir, manifest, rows, counts, queue_status)
    return counts


def write_report(output_dir: str | Path) -> dict[str, int]:
    """Write CSV, JSON, Markdown and (if measurements exist) both figure formats."""
    output_dir = Path(output_dir)
    manifest, rows, total = _read_inputs(output_dir)
    queue_status = _read_optional_json(output_dir / "status.json")
    completed = [row for row in rows if row.get("status") == "completed"]
    counts = _counts(rows, total)
    _write_csv(output_dir, completed)
    (output_dir / "results.json").write_text(json.dumps(
        {"manifest": manifest, "counts": counts, "queue_status": queue_status, "rows": rows},
        ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n", encoding="utf-8")
    if completed:
        _plot(output_dir, manifest, completed, total)
    else:
        for filename in ("comparison.png", "comparison.svg"):
            (output_dir / filename).unlink(missing_ok=True)
    _write_markdown(output_dir, manifest, rows, counts, queue_status)
    return counts


def watch_report(output_dir: str | Path, interval: float = 60) -> None:
    """Keep a single Markdown refresher alive until the queue stops."""
    output_dir = Path(output_dir)
    if interval <= 0:
        raise ValueError("watch interval must be positive")
    with (output_dir / "report_watch.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if lock.read(1) == b"":
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("A report watcher is already running for this directory") from error
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("A report watcher is already running for this directory") from error
        while True:
            counts = refresh_markdown(output_dir)
            print(json.dumps({"refreshed_at": datetime.now(BEIJING).isoformat(), **counts}), flush=True)
            status = _read_optional_json(output_dir / "status.json") or {}
            if status.get("status") in {"completed", "needs_attention", "paused"}:
                refresh_markdown(output_dir)
                return
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--watch", action="store_true", help="Refresh only report.md until the queue stops")
    parser.add_argument("--interval", type=float, default=60, help="Watch refresh interval in seconds (default: 60)")
    args = parser.parse_args()
    if args.watch:
        watch_report(args.output_dir, args.interval)
    else:
        print(json.dumps(write_report(args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
