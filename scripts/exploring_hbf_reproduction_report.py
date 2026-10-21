"""Render completed paper-comparison results without running any experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("llama3_405b", "llama4_maverick")
WORKLOAD_ORDER = ("SHORT", "MID", "LONG")
HARDWARE_ORDER = ("HBM4", "CONV", "CONV+", "HBF", "HBF+")
GPU_ORDER = (1, 2, 4, 8, 16)
MODEL_NAMES = {"llama3_405b": "Llama 3.1 405B", "llama4_maverick": "Llama 4 Maverick"}
WORKLOAD_NAMES = {"SHORT": "短请求", "MID": "中请求", "LONG": "长请求"}
WORKLOAD_LENGTHS = {"SHORT": (1660, 373), "MID": (5900, 499), "LONG": (103500, 1100)}
TERMINAL = {"completed", "infeasible", "failed", "no_validated_deployment"}
STATUS_NAMES = {"completed": "已完成", "infeasible": "未通过容量或代表事件筛选", "failed": "执行失败",
                "no_validated_deployment": "未找到已验收部署",
                "searching": "搜索中", "validating": "闭环验证中", "invalid_terminal": "结果不完整", "missing": "待运行"}
PAPER_URL = "https://arxiv.org/abs/2608.13868"
PDF_PATH = Path("C:/Users/A/Downloads/HBF_HBM_LLM_2026-10-15/01_HBF/HBF05_2026_Exploring_HBF.pdf")
# Paper values are rounded figure labels, not author raw measurements.
PAPER_HBM4_8GPU = {
    ("llama3_405b", "SHORT"): (194., 3300.),
    ("llama3_405b", "MID"): (61.5, 1800.),
    ("llama3_405b", "LONG"): (3.75, 147.),
    ("llama4_maverick", "SHORT"): (460., 19000.),
    ("llama4_maverick", "MID"): (151.5, 6300.),
    ("llama4_maverick", "LONG"): (31., 1300.),
}
SUMMARY_FIELDS = ("case_id", "variant", "model", "hardware", "workload", "gpus", "screen_slo_seconds", "slo_statistic", "status",
    "batch_per_replica", "batch_per_gpu", "cluster_batch", "per_gpu_tokens_s", "cluster_tokens_s",
    "max_validated_batch_per_gpu", "max_validated_batch_per_replica", "max_validated_batch_parallelism",
    "mean_tpot_ms", "mean_iteration_ms", "max_iteration_ms", "normalized_batch", "normalized_throughput", "dp", "pp", "tp", "ep", "microbatches",
    "measured_steps", "measurement_seconds", "pd_program_completed_bytes", "decode_program_bytes",
    "projected_three_year_average_pec", "measurement_program_bytes_lower_bound", "measurement_program_bytes_upper_bound",
    "projected_three_year_pec_lower_bound", "projected_three_year_pec_upper_bound", "capacity_unit_bytes", "final_deadline_seconds",
    "max_completed_request_elapsed_seconds", "completed_request_elapsed_count", "wall_seconds", "reason")
METRIC_FIELDS = ("batch_per_replica", "batch_per_gpu", "cluster_batch", "per_gpu_tokens_s", "cluster_tokens_s",
    "max_validated_batch_per_gpu", "max_validated_batch_per_replica", "max_validated_batch_parallelism",
    "mean_tpot_ms", "mean_iteration_ms", "max_iteration_ms", "normalized_batch", "normalized_throughput", "dp", "pp", "tp", "ep", "microbatches",
    "measured_steps", "measurement_seconds", "pd_program_completed_bytes", "decode_program_bytes",
    "projected_three_year_average_pec", "measurement_program_bytes_lower_bound", "measurement_program_bytes_upper_bound",
    "projected_three_year_pec_lower_bound", "projected_three_year_pec_upper_bound", "capacity_unit_bytes", "max_completed_request_elapsed_seconds", "completed_request_elapsed_count")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) else None


def _ratio(numerator, denominator):
    n, d = _number(numerator), _number(denominator)
    return n / d if n is not None and d is not None and d > 0 else None


def _relative_error(value, reference):
    ratio = _ratio(value, reference)
    return None if ratio is None else abs(ratio - 1.)


def _case_key(case):
    slo = case.get("slo", .1)
    slo_key = "offline" if slo == "offline" else str(float(slo))
    return (case.get("variant", "primary"), case.get("model"), case.get("hardware"),
            case.get("workload"), int(case.get("gpus", 0)), slo_key, case.get("slo_statistic", "mean_iteration"))


def _is_primary(row):
    return (row["variant"] == "primary" and row.get("screen_slo_seconds") == .1
            and row.get("slo_statistic") == "mean_iteration")


def _slo_statistic(case, selected):
    if case.get("slo") == "offline":
        return "max_completed_request_elapsed"
    if "slo_statistic" in case:
        return case["slo_statistic"] if case["slo_statistic"] in {"mean_iteration", "max_iteration"} else "unknown"
    result = selected.get("result") if isinstance(selected, dict) else None
    # Current primary defaults to a mean interpretation. Legacy data without
    # that measured field has unknown semantics, not an invented average.
    return "mean_iteration" if isinstance(result, dict) and _number(result.get("mean_iteration_ns")) is not None else "unknown"


def _max_validated_batch(record):
    """Largest passed per-GPU batch, independent of throughput selection.

    It is a maximum over the recorded accepted closed-loop validations only;
    untested or rejected batches are never called feasible capacity.
    """
    selected = record.get("selected")
    validations = list(record.get("validations") or [])
    if isinstance(selected, dict):
        validations.append(selected)
    best = None
    for validation in validations:
        result = validation.get("result")
        if validation.get("accepted") is not True or not isinstance(result, dict) or result.get("status") != "completed":
            continue
        parallel = validation.get("parallelism", result.get("parallelism", {}))
        degrees = [parallel.get(key, 1) for key in ("pp", "tp", "ep")]
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in degrees):
            continue
        replicas = degrees[0] * degrees[1] * degrees[2]
        per_replica = _number(validation.get("batch_size"))
        per_gpu = _ratio(per_replica, replicas)
        if per_gpu is not None and per_gpu > 0 and (best is None or per_gpu > best[0]):
            best = per_gpu, per_replica, parallel
    return best


def load_terminal_rows(directory: Path):
    """Ignore searching/validating selected data, even if it looks complete."""
    records = []
    parse_errors = []
    for path in sorted((directory / "points").glob("*.json")):
        try:
            record = _read(path)
            if not isinstance(record, dict) or not isinstance(record.get("case"), dict):
                raise ValueError("point lacks a case object")
            records.append(record)
        except (OSError, ValueError) as error:
            parse_errors.append(f"{path.name}: {type(error).__name__}: {error}")
    rows = []
    for record in records:
        if record.get("status") not in TERMINAL:
            continue
        case, selected = record["case"], record.get("selected")
        row = {key: None for key in SUMMARY_FIELDS}
        row.update(case_id=record.get("case_id", ""), variant=case.get("variant", "primary"),
                   model=case.get("model"), hardware=case.get("hardware"), workload=case.get("workload"),
                   gpus=case.get("gpus"), screen_slo_seconds=record.get("paper_slo_seconds"),
                   slo_statistic=_slo_statistic(case, selected),
                   status=record["status"], wall_seconds=record.get("wall_seconds"),
                   reason=record.get("reason") or record.get("error") or "")
        if row["screen_slo_seconds"] is None:
            output = WORKLOAD_LENGTHS.get(row["workload"], (0, 0))[1]
            row["screen_slo_seconds"] = 86400 / output if case.get("slo") == "offline" and output else case.get("slo", .1)
        if case.get("slo") == "offline":
            row["final_deadline_seconds"] = 86400
        if any(candidate.get("status") == "execution_failed" for candidate in record.get("screening", [])):
            row.update(status="failed", reason="搜索执行失败，不能归为硬件不可行或成功部署。")
            rows.append(row)
            continue
        if record["status"] == "completed":
            result = selected.get("result") if isinstance(selected, dict) else None
            if (not isinstance(result, dict) or selected.get("accepted") is not True
                    or result.get("status") != "completed"):
                row.update(status="invalid_terminal", reason="终态缺少已接受的完整闭环结果；指标不展示。")
            else:
                parallel = selected.get("parallelism", result.get("parallelism", {}))
                replica_gpus = parallel.get("pp", 1) * parallel.get("tp", 1) * parallel.get("ep", 1)
                batch = selected.get("batch_size")
                row.update(batch_per_replica=batch,
                    batch_per_gpu=result.get("per_gpu_batch_size", _ratio(batch, replica_gpus)),
                    cluster_batch=batch * parallel.get("dp", 1) if _number(batch) is not None else None,
                    per_gpu_tokens_s=result.get("per_gpu_decode_tokens_per_second"),
                    cluster_tokens_s=result.get("cluster_decode_tokens_per_second"),
                    mean_tpot_ms=_ratio(result.get("mean_tpot_ns"), 1e6),
                    mean_iteration_ms=_ratio(result.get("mean_iteration_ns"), 1e6),
                    max_iteration_ms=_ratio(result.get("max_iteration_ns"), 1e6),
                    measured_steps=result.get("measured_steps"),
                    measurement_seconds=_ratio(result.get("measurement_end_ns", 0) - result.get("measurement_start_ns", 0), 1e9),
                    pd_program_completed_bytes=result.get("pd_program_completed_bytes"),
                    decode_program_bytes=result.get("decode_program_bytes"),
                    projected_three_year_average_pec=result.get("projected_three_year_average_pec"),
                    measurement_program_bytes_lower_bound=result.get("measurement_program_bytes_lower_bound"),
                    measurement_program_bytes_upper_bound=result.get("measurement_program_bytes_upper_bound"),
                    projected_three_year_pec_lower_bound=result.get("projected_three_year_pec_lower_bound"),
                    projected_three_year_pec_upper_bound=result.get("projected_three_year_pec_upper_bound"),
                    capacity_unit_bytes=result.get("research_assumptions", {}).get("capacity_unit_bytes"),
                    max_completed_request_elapsed_seconds=_ratio(result.get("max_completed_request_elapsed_ns"), 1e9),
                    completed_request_elapsed_count=result.get("completed_request_elapsed_count"))
                largest = _max_validated_batch(record)
                if largest is not None:
                    row.update(max_validated_batch_per_gpu=largest[0], max_validated_batch_per_replica=largest[1],
                               max_validated_batch_parallelism=json.dumps(largest[2], ensure_ascii=False, sort_keys=True))
                for key in ("dp", "pp", "tp", "ep", "microbatches"):
                    row[key] = parallel.get(key, 1)
                # No finished row may accidentally turn a missing metric into zero.
                if any(_number(row[key]) is None or row[key] <= 0 for key in ("batch_per_gpu", "per_gpu_tokens_s", "max_iteration_ms")):
                    row.update(status="invalid_terminal", reason="终态缺少有效批量、吞吐或完整迭代延迟；指标不展示。")
                    for key in METRIC_FIELDS:
                        row[key] = None
                if case.get("slo") == "offline" and (
                        _number(row["max_completed_request_elapsed_seconds"]) is None
                        or row["max_completed_request_elapsed_seconds"] > 86400
                        or row["max_completed_request_elapsed_seconds"] <= 0):
                    row.update(status="invalid_terminal", reason="离线终态缺少完整fresh请求Decode驻留时间，或超过24小时；不能使用单步门槛代替最终验收。")
                    for key in METRIC_FIELDS:
                        row[key] = None
                if row["slo_statistic"] == "mean_iteration" and (
                        _number(row["mean_iteration_ms"]) is None or row["mean_iteration_ms"] <= 0):
                    row.update(status="invalid_terminal", reason="平均迭代验收结果缺少有效mean_iteration_ns；不能用最大值代替平均值。")
                    for key in METRIC_FIELDS:
                        row[key] = None
        rows.append(row)
    baselines = {(row["model"], row["workload"]): row for row in rows
                 if _is_primary(row) and row["gpus"] == 8 and row["hardware"] == "HBM4" and row["status"] == "completed"}
    for row in rows:
        baseline = baselines.get((row["model"], row["workload"])) if _is_primary(row) else None
        if row["status"] == "completed" and baseline:
            row["normalized_batch"] = _ratio(row["max_validated_batch_per_gpu"], baseline["max_validated_batch_per_gpu"])
            row["normalized_throughput"] = _ratio(row["per_gpu_tokens_s"], baseline["per_gpu_tokens_s"])
    return rows, records, parse_errors


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def eight_gpu_comparison(rows):
    indexed = {(row["model"], row["workload"], row["hardware"]): row for row in rows
               if _is_primary(row) and row["gpus"] == 8}
    comparisons = []
    for model in MODEL_ORDER:
        for workload in WORKLOAD_ORDER:
            paper_batch, paper_tps = PAPER_HBM4_8GPU[model, workload]
            for hardware in HARDWARE_ORDER:
                row = indexed.get((model, workload, hardware), {})
                result = dict(model=model, workload=workload, hardware=hardware,
                    status=row.get("status", "missing"), max_validated_batch_per_gpu=row.get("max_validated_batch_per_gpu"),
                    throughput_selected_batch_per_gpu=row.get("batch_per_gpu"),
                    simulated_tokens_s_per_gpu=row.get("per_gpu_tokens_s"),
                    batch_gain_vs_simulated_HBM4=row.get("normalized_batch"),
                    throughput_gain_vs_simulated_HBM4=row.get("normalized_throughput"),
                    paper_HBM4_batch_per_gpu_rounded=paper_batch if hardware == "HBM4" else None,
                    paper_HBM4_tokens_s_per_gpu_rounded=paper_tps if hardware == "HBM4" else None,
                    batch_relative_error_to_rounded=None, throughput_relative_error_to_rounded=None,
                    paper_batch_gain_global_six_case_range=("1.3–4.5" if hardware == "HBF" else "1.7–5.3" if hardware == "HBF+" else None),
                    comparison_scope="批量为全部已通过闭环验证候选中的最大值，非全局最大；吞吐取吞吐最优已验证部署。论文约数与跨六场景范围不是逐柱原始数据。")
                if hardware == "HBM4" and row.get("status") == "completed":
                    result["batch_relative_error_to_rounded"] = _relative_error(row["max_validated_batch_per_gpu"], paper_batch)
                    result["throughput_relative_error_to_rounded"] = _relative_error(row["per_gpu_tokens_s"], paper_tps)
                comparisons.append(result)
    return comparisons


def _plots(directory, rows, *, scope=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import FuncFormatter

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 13
    colors = {"HBM4": "#326ac3", "CONV": "#8b93a5", "CONV+": "#229b81", "HBF": "#e49122", "HBF+": "#cb4367"}
    indexed = {(row["model"], row["workload"], row["hardware"], row["gpus"]): row
               for row in rows if _is_primary(row) and row["status"] == "completed"}
    if scope == "eight_gpu_core":
        for field, name, title, units in (("per_gpu_tokens_s", "throughput", "8 GPU各存储方案的每GPU生成吞吐", "tokens/s/GPU"),
                ("max_validated_batch_per_gpu", "batch", "8 GPU各存储方案的每GPU最大已验证批量", "请求/GPU（非全局最大）")):
            figure, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
            for mi, model in enumerate(MODEL_ORDER):
                for wi, workload in enumerate(WORKLOAD_ORDER):
                    axis = axes[mi, wi]
                    for hi, hardware in enumerate(HARDWARE_ORDER):
                        value = indexed.get((model, workload, hardware, 8), {}).get(field)
                        if value is not None and value > 0:
                            axis.bar(hi, value, color=colors[hardware], width=.7)
                            label = (f"{value:,.3f}".rstrip("0").rstrip(".") if field == "max_validated_batch_per_gpu"
                                     else f"{value:,.0f}" if value >= 100 else f"{value:.2f}")
                            axis.annotate(label,
                                (hi, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=11)
                        else:
                            axis.text(hi, .03, "未验收", transform=axis.get_xaxis_transform(), ha="center", fontsize=11, color="#64748b")
                    axis.set_xticks(range(len(HARDWARE_ORDER)), HARDWARE_ORDER)
                    axis.set_ylim(bottom=0)
                    axis.margins(y=.16)
                    inp, out = WORKLOAD_LENGTHS[workload]
                    axis.set_title(f"{MODEL_NAMES[model]} · {WORKLOAD_NAMES[workload]}\n输入 {inp:,}，输出 {out:,}", fontsize=14)
                    axis.set_ylabel(units)
                    axis.grid(axis="y", alpha=.18)
                    axis.set_axisbelow(True)
                    axis.spines[["top", "right"]].set_visible(False)
            figure.suptitle(title, fontsize=22, y=.995)
            figure.text(.5, .015, "仅8 GPU、平均满批迭代≤100毫秒的核心对照；吞吐取所选部署，批量取已通过闭环验证的最大值；空缺不填零。",
                        ha="center", fontsize=12, color="#475569")
            figure.tight_layout(rect=(0, .045, 1, .95))
            figure.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
            plt.close(figure)
        return
    for field, name, title in (("normalized_throughput", "throughput", "每GPU生成吞吐"),
                               ("normalized_batch", "batch", "每GPU最大已验证批量（非全局最大）")):
        figure, axes = plt.subplots(2, 3, figsize=(17, 10), squeeze=False)
        for mi, model in enumerate(MODEL_ORDER):
            for wi, workload in enumerate(WORKLOAD_ORDER):
                axis = axes[mi, wi]
                present = False
                for hardware in HARDWARE_ORDER:
                    values = [indexed.get((model, workload, hardware, gpu), {}).get(field) for gpu in GPU_ORDER]
                    present |= any(value is not None and value > 0 for value in values)
                    axis.plot(GPU_ORDER, [value if value is not None and value > 0 else float("nan") for value in values],
                              label=hardware, color=colors[hardware], marker="o", linewidth=2, markersize=6)
                axis.set_xscale("log", base=2)
                axis.set_xticks(GPU_ORDER, [str(gpu) for gpu in GPU_ORDER])
                axis.set_xlim(.9, 18)
                if present:
                    axis.set_yscale("log", base=2)
                    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                else:
                    axis.text(.5, .5, "等待平均迭代目标的8 GPU HBM4基准", transform=axis.transAxes,
                              ha="center", va="center", fontsize=13, color="#64748b")
                    axis.set_ylim(0, 2)
                axis.axhline(1, color="#64748b", linestyle="--", linewidth=1, alpha=.65)
                inp, out = WORKLOAD_LENGTHS[workload]
                axis.set_title(f"{MODEL_NAMES[model]} · {WORKLOAD_NAMES[workload]}\n输入 {inp:,}，输出 {out:,}", fontsize=14)
                axis.set_xlabel("GPU数量")
                axis.set_ylabel("相对8 GPU HBM4的比值" + ("（对数）" if present else ""))
                axis.grid(alpha=.18, which="major")
                axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=5, frameon=False, fontsize=14)
        figure.suptitle(f"{title}随GPU数量的变化", fontsize=22, y=.995)
        figure.text(.5, .015, "主实验按平均满批迭代≤100毫秒解释；论文未披露尾分位；空缺不填零；同场景8 GPU HBM4归一。",
                    ha="center", fontsize=12, color="#475569")
        figure.tight_layout(rect=(0, .045, 1, .89))
        figure.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
        plt.close(figure)


def _fmt(value, digits=3):
    return f"{value:,.{digits}f}" if _number(value) is not None else "—"


def _percent(value):
    return f"{value * 100:.2f}%" if _number(value) is not None else "—"


def _markdown(directory, manifest, rows, records, errors, comparisons):
    expected = {_case_key(case) for case in manifest.get("cases", [])}
    terminal = {_case_key(record["case"]) for record in records if record.get("status") in TERMINAL}
    count = len(expected & terminal) if expected else len(terminal)
    complete = bool(expected) and expected <= terminal and not errors and not any(row["status"] == "invalid_terminal" for row in rows)
    counts = Counter(row["status"] for row in rows)
    progress = f"{count}/{len(expected)}" if expected else f"{count}个终态；无完整计划清单"
    verdict = "计划配置已全部成为终态" if complete else "阶段报告：计划尚未全部成为有效终态"
    deferred = manifest.get("deferred_cases", [])
    scope_note = ""
    if deferred:
        deferred_gpus = "、".join(str(value) for value in sorted({case["gpus"] for case in deferred}))
        scope_note = (f"本轮核心对照为{len(expected)}个计划配置。另{len(deferred)}个GPU数量或SLO目标的探索配置未验收，"
                      f"涉及{deferred_gpus} GPU及其他SLO范围；部分初始工作已隔离保留，不纳入对照。它们保存在deferred_cases中，"
                      "只有后续明确请求才运行，不计入本轮完成分母。")
    base = [row for row in comparisons if row["hardware"] == "HBM4" and row["status"] == "completed"]
    lines = ["# Exploring HBF历史旁路模型结果", "",
        "**后端来源更正：以下30组历史实验由独立的 `paper_*` 算子/事件模型产生，没有经过原生 `reporting.run_scenario → simulate_online → planner → UnifiedEventKernel → 物理核`。不能将其称为原生仿真器对论文的验收，也不能用来证明原生后端已支持连续补位、PD、SRAM或Maverick。原始数字保留，仅作历史探索。新的原生实现映射和实验安排见项目文档 `docs/exploring_hbf_native_comparison.md`。**", "",
        f"**{verdict}。** 当前进度 {progress}；成功 {counts['completed']}，未通过筛选 {counts['infeasible']}，未找到已验收部署 {counts['no_validated_deployment']}，执行失败 {counts['failed']}，不完整终态 {counts['invalid_terminal']}。",
        "", "本报告只使用当前目录中真实终态的闭环结果。搜索中或验证中的结果即使已有候选，也不计为完成；失败和不可行配置不填充吞吐数据。此前12个短物理控制实验属于另一项对照，不纳入这里的对照结果。",
        "", scope_note,
        "", "## 结果口径", "", "每个候选先接受容量与中点上下文的单步Decode筛选，再执行完整闭环验证。每个选中结果包含一个输出生命周期的预热、两个输出生命周期的测量，并在请求结束前准备新请求的PD传输和KV写入。选择依据是已验证候选的最大每GPU吞吐；只验证有限个排名靠前的并行方案并按需要缩批，**不证明全局最优，也不把单步筛选误称稳态验收**。",
        "", "论文仅写TPOT≤0.1秒，未公开平均、尾分位或逐步最大值的精确定义。本次主实验采用常用的稳态平均解释：`mean_iteration_ns ≤ 100毫秒`，包含补位等待；**不能断言论文原实现一定采用平均值**。严格`max_iteration_ns`逐步最大目标作为独立控制，仍在CSV中保留最大值，但不混入平均目标主曲线。`mean_tpot_ns`是请求自身相邻输出间隔，不等于系统完整迭代。硬最大值加FIFO会把稀疏整组PD阻塞变成所有请求的SLO失败，这与论文的常数稳态带宽模型不同；改变主实验解释不放松旧快速模式10%预算。",
        "", "离线仅用`24小时 / 总输出长度`作为保守单步筛选门槛，最终必须验证`max_completed_request_elapsed_ns ≤ 24小时`：即新接纳fresh请求从Decode节点接纳到完成的完整驻留时间，包含其他槽位PD造成的阻塞；不包含该请求接纳前的Prefill或自身PD导入时间。作者报告第19页明确全部硬件按规定峰值性能运行，因此主实验矩阵与标量效率均设为1；旧60%效率探索记录单独保留，不混入主实验。BF16和链路启动参数仍是显式研究选择，并非论文披露的完整器件实测参数。",
        "", "`infeasible`仅表示未通过声明的容量或代表事件筛选，不是物理不可行证明；`no_validated_deployment`表示有筛选候选但有限闭环尝试未找到通过部署。任何搜索执行错误均归为`failed`，不能当作硬件不可行或用零吞吐填图。",
        "", "吞吐与批量分别取值：图4吞吐对照取`selected`所选吞吐最优的已验证部署；图3批量对照及batch.png取所有`accepted=true`完整闭环验证中最大的每GPU批量，**不是全局最大批量或硬件容量上限的证明**。二者可能来自不同并行方案，不能用吞吐最优部署的较小批量冒充最大已验证批量。summary.csv中`batch_per_gpu`是吞吐所选批量，`max_validated_batch_per_gpu`是最大已验证批量，并另记其并行方案。",
        "", "归一化公式：`吞吐比 = 本配置所选每GPU tokens/s / 同模型同负载8 GPU HBM4所选每GPU tokens/s`；批量比使用两边各自最大已验证批量。每GPU批量为`副本批量 / (TP × PP × EP)`；DP只增加独立副本数，不再次摊薄每GPU工作量。",
        "", "![每GPU吞吐图](throughput.png)", "", "![每GPU批量图](batch.png)", "", "## 8 GPU HBM4与论文数值对照", "",
        "论文第3页图3、图4的标注经过四舍五入，以下不是作者提供的原始采样值。相对差异为`|本仿真值−论文标注| / 论文标注`。它衡量跨模型差异，不能当作原快速模式的10%误差预算验收。", "",
        "|模型|负载|最大已验证批量/GPU|吞吐所选批量/GPU|论文批量（约）|批量差异|所选吞吐/GPU|论文吞吐（约）|吞吐差异|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODEL_ORDER:
        for workload in WORKLOAD_ORDER:
            row = next(item for item in comparisons if (item["model"], item["workload"], item["hardware"]) == (model, workload, "HBM4"))
            lines.append(f"|{MODEL_NAMES[model]}|{WORKLOAD_NAMES[workload]}|{_fmt(row['max_validated_batch_per_gpu'])}|{_fmt(row['throughput_selected_batch_per_gpu'])}|{_fmt(row['paper_HBM4_batch_per_gpu_rounded'])}|{_percent(row['batch_relative_error_to_rounded'])}|{_fmt(row['simulated_tokens_s_per_gpu'], 1)}|{_fmt(row['paper_HBM4_tokens_s_per_gpu_rounded'], 1)}|{_percent(row['throughput_relative_error_to_rounded'])}|")
    if base:
        for field, label in (("batch_relative_error_to_rounded", "批量"), ("throughput_relative_error_to_rounded", "吞吐")):
            values = [row[field] for row in base if row[field] is not None]
            if values:
                lines += ["", f"已完成的{len(values)}/6组HBM4对照中，{label}相对差异范围为{_percent(min(values))}～{_percent(max(values))}，中位数{_percent(median(values))}。"]
    lines += ["", "以上差异是直接计算结果；未利用论文吞吐值调整计算效率、模型张量尺寸或媒体带宽。模型计算精度、有效算力、块注意力实现和容量单位均可能影响数值；接近个别标注不能证明整篇论文复现。", "", "## HBF容量、批量与吞吐的对应关系", "",
        "论文第3页报告8 GPU下，HBF相对HBM4的批量增益跨六场景为1.3～4.5倍，HBF+为1.7～5.3倍。论文没有提供每根柱的原始值，因此不为各场景编造精确目标，也不对这些范围计算逐柱相对误差。", "",
        "|模型|负载|方案|最大已验证批量/GPU|吞吐所选批量/GPU|相对HBM4最大已验证批量|所选吞吐/GPU|相对HBM4所选吞吐|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in comparisons:
        if row["hardware"] not in {"HBF", "HBF+"}:
            continue
        lines.append(f"|{MODEL_NAMES[row['model']]}|{WORKLOAD_NAMES[row['workload']]}|{row['hardware']}|{_fmt(row['max_validated_batch_per_gpu'])}|{_fmt(row['throughput_selected_batch_per_gpu'])}|{_fmt(row['batch_gain_vs_simulated_HBM4'])}|{_fmt(row['simulated_tokens_s_per_gpu'], 1)}|{_fmt(row['throughput_gain_vs_simulated_HBM4'])}|{STATUS_NAMES.get(row['status'], row['status'])}|")
    for hardware in ("HBF", "HBF+"):
        pairs = [row for row in comparisons if row["hardware"] == hardware and row["status"] == "completed" and row["batch_gain_vs_simulated_HBM4"] is not None]
        if pairs:
            batch_values = [row["batch_gain_vs_simulated_HBM4"] for row in pairs]
            throughput_values = [row["throughput_gain_vs_simulated_HBM4"] for row in pairs]
            lines += ["", f"当前{hardware}有{len(pairs)}/6组有效配对：批量增益{_fmt(min(batch_values))}～{_fmt(max(batch_values))}倍，吞吐增益{_fmt(min(throughput_values))}～{_fmt(max(throughput_values))}倍。"]
    lines += ["", "HBF的容量增加可让更大的批量摊销权重读取，但不保证吞吐按批量同比增加；每序列KV读取随上下文增长，PD补位写入受较低写带宽约束。HBF+删除中间数据HBM后，每堆40 MB SRAM可能先约束工作区；因此容量、并行部署、预取信用与补位停顿需要一起分析。论文研究Decode节点，不能用这里的结果推导完整长Prefill的TTFT。"]
    interpretation = directory / "interpretation.md"
    if complete and interpretation.is_file():
        lines += ["", interpretation.read_text(encoding="utf-8").strip()]
    lines += ["", "## 机制覆盖与仍存在的未知项", "", "|机制|本次执行与边界|", "|---|---|",
        "|模型张量|405B来自Meta官方SKU；Maverick来自Meta torchtune专用builder，包含交替Dense/MoE、Top-1、共享专家及3:1固定块/全注意力。官方HF checkpoint受门控，未下载逐张量核验。|",
        "|私有存储容量|按每GPU和每PP阶段计算TP/EP权重、GQA头复制和工作区；活动KV采用真实错相年龄的paged pool，对各生命周期相位的full/chunk加权占用逐相位求峰值，不再为全部序列同时预留最大长度。有限准备KV池另加，不把所有GPU容量当作一个共享池。|",
        "|HBF读流水|有限阵列准备资源支付tREAD，再经独立数据端口传输；每堆3.13 MB SRAM拆成两个有限读缓冲，填充、消耗和释放分别付费，没有无限阵列准备、无限预取或隐藏HBM溢出。当前MB按十进制解释，3.13 MB=3,130,000字节；论文未消歧的二进制单位应另列敏感性，不自动改字节数。|",
        "|NAND写服务|接收链路单独付费，编程阵列占用为max(tPROG，字节数/论文写带宽)，不把已含编程吞吐的写带宽再叠加一次完整tPROG。|",
        "|中间数据|HBF/CONV使用HBM；HBF+/CONV+使用每堆40 MB逻辑层SRAM，具有显式生命周期与容量。|",
        "|词表输出工作区|当前LM_HEAD在每个TP分片完整保留FP32词表输出，再执行采样归约；这不是论文公开的必选实现。Maverick的202,048词表可能先占满HBF+的320 MB中间SRAM，限制批量。流式词表投影/采样可减少存活工作区，但本轮未实现，也不为追求相同数值而新增。|",
        "|并行与通信|DP、TP、PP、EP及微批逐资源执行；GPU注入/接收、跨节点链路及PD接收共享相应资源日历。|",
        "|PD与主机暂存|采用有限HOST DRAM双缓冲流式导入；64 MiB容量与200 GB/s带宽属于研究补充值，HOST不是新增HBM。每段接收、暂存、读出和GPU写入有真实依赖与缓冲信用。|",
        "|闭环服务|完成请求的槽位保持稳定；新请求KV必须实际完成PD传输和存储写入才可接纳。提前准备窗口由实际PD费用推导，对应准备KV池计入容量；不固定为两步，也不靠修改缓存长度跳过写入。|",
        "|执行粒度|独立的论文批量/算子事件后端；有限缓冲重复转移可精确加速，不调用旧单请求快速预测。它不等同于OCP逐事务FTL仿真。|",
        "|作者方法补充|作者报告第19页明确GPU核心、存储堆栈与互连均按规定峰值性能运行，主实验矩阵与标量效率为1；第20页说明零停顿批处理和更高性能Prefill节点。旧60%效率探索记录单独保留，不能作为当前主实验基线。|",
        "|未披露输入|BF16、SRAM端口、链路启动时延、均匀专家路由等仍为研究补充假设；不把未知量反调到论文曲线。|",
        "|计算侧补充来源|NVIDIA HGX官方Rubin单GPU规格提供dense BF16 4 PFLOPS、FP32向量130 TFLOPS，均为峰值上限且规格仍可能变化；主实验采用作者报告的峰值假设。官网22 TB/s HBM带宽不用于替换论文Table I的12.8 TB/s，也不把官方较新的链路规格代入论文配置。|",
        "|容量单位|主实验明确GB=10⁹字节。Maverick BF16文本权重约801.424 GB=746.384 GiB，与论文746 GB接近只能作为单位歧义线索；GiB须单列敏感性实验，不自动缩减张量。|",
        "|搜索与稳态|全部声明的离散并行布局先筛选；提前准备窗口由实际PD费用推导后，再对对应准备KV池逐策略二分容量边界，复用相同候选的已完成闭环结果。只对有限顶候选完整闭环验收，不证明全局最大吞吐或无限运行周期稳态。|",
        "|GC与耐久性|此后端使用论文带宽抽象，没有虚构FTL、擦除、GC复制或真实磨损分布；原NAND物理后端保持独立。|",
        "", "第4页图5的延迟分解尚不能宣称复现：当前按阶段/微批记录的时间跨度可互相重叠；PD资源服务占用也不是关键路径延迟。不得将这些项相加后归一化为图5柱图。", "", "## 写入完成窗口与三年磨损投影", "",
        "编程量按测量窗口中的Decode写入，加PD导入**完整import组完成**时记账的编程字节；通过窗口起止未完成导入的字节守恒修正，不用仅发起量或仅接纳量代替。一个未完成import组可能已完成部分编程，因此这不是逐页完成时刻的精确计数。启动导入与预热不混入测量窗口。", "",
        "公式：`三年平均P/E投影 = (窗口内Decode编程字节 + 窗口内完成PD编程字节) / 窗口秒数 × (3×365×24×3600) / 副本可用HBF总字节`。DP独立同分布时，分子分母同倍缩放。它是假定均匀磨损的研究投影，不是真实FTL擦除计数、GC写放大预测或器件寿命保证。", "",
        "令A为Decode编程与完整PD组完成字节之和、S/E为窗口起止尚未完成import组的编程字节，则部分编程边界的保守范围为`max(0,A−S) ≤ 实际窗口编程字节 ≤ A+E`。相同时间和容量换算得到下表P/E上下界；组完成口径投影不可冒充逐页精确值。", "",
        "|模型|负载|方案|窗口秒数|PD组完成编程GB|Decode编程GB|组完成P/E投影|P/E下界|P/E上界|", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    pec_rows = [row for row in rows if _is_primary(row) and row["gpus"] == 8 and row["hardware"] in {"HBF", "HBF+", "CONV", "CONV+"} and row["status"] == "completed"]
    for row in sorted(pec_rows, key=lambda item: (MODEL_ORDER.index(item["model"]), WORKLOAD_ORDER.index(item["workload"]), HARDWARE_ORDER.index(item["hardware"]))):
        lines.append(f"|{MODEL_NAMES[row['model']]}|{WORKLOAD_NAMES[row['workload']]}|{row['hardware']}|{_fmt(row['measurement_seconds'], 2)}|{_fmt(_ratio(row['pd_program_completed_bytes'], 1e9))}|{_fmt(_ratio(row['decode_program_bytes'], 1e9))}|{_fmt(row['projected_three_year_average_pec'], 1)}|{_fmt(row['projected_three_year_pec_lower_bound'], 1)}|{_fmt(row['projected_three_year_pec_upper_bound'], 1)}|")
    lines += ["", "## 来源与可复算文件", "",
        f"- [用户提供的论文PDF]({PDF_PATH.as_posix()})：第2页Table I、方法和SRAM组织；第3页图3/图4批量与吞吐；第4页图5与三年P/E讨论。",
        f"- [作者论文主源]({PAPER_URL})。所有HBM4参考数值均为图中约数；HBF/HBF+只提供跨六场景范围。",
        "- [作者方法报告](https://people.inf.ethz.ch/omutlu/pub/HBF-for-LLM-cal26-talk.pdf)：PDF第19页（索引18）明确全部硬件按规定峰值性能运行，第20页（索引19）说明假定高性能Prefill节点以支持零停顿批处理；主实验据此使用效率1。",
        "- [Meta官方Maverick构建](https://raw.githubusercontent.com/meta-pytorch/torchtune/main/torchtune/models/llama4/_model_builders.py)与[算子组件](https://raw.githubusercontent.com/meta-pytorch/torchtune/main/torchtune/models/llama4/_component_builders.py)。",
        "- [NVIDIA HGX官方规格](https://www.nvidia.com/en-gb/data-center/hgx/)：仅补充Rubin dense BF16 4 PFLOPS及FP32 130 TFLOPS算力来源，不用官网22 TB/s替换论文存储带宽。",
        "- `manifest.json`保存实际计划及论文硬件目录；`points/*.json`保存全部终态、筛选、闭环候选、研究假设和资源计数。",
        "- `summary.csv`保存全部终态的绝对指标及主实验归一化值；`8GPUcomparison.csv`保存约数对照、相对差异和HBF增益范围。",
        "- 原40个保留结果、32个取消点及旧12个短控制实验不在本次统计中，未重启旧扫描。"]
    accepted_cases = {row["case_id"] for row in rows if row["status"] == "completed"}
    missing_host = sum(1 for record in records if record.get("case_id") in accepted_cases
                       and not record.get("selected", {}).get("result", {}).get("pd_host_staging"))
    if missing_host:
        lines += ["", f"**机制版本提示：有{missing_host}个成功结果未记录当前PD HOST暂存元数据。** 上述机制表描述当前实现，不能据此认定这些旧结果已包含HOST双缓冲路径；更新后的正式对照应重测或按显式variant分开。"]
    nonprimary = [row for row in rows if not _is_primary(row)]
    if nonprimary:
        lines += ["", f"目录另有{len(nonprimary)}个非主实验终态（不同目标、严格最大值、未知SLO口径或显式variant），保留在summary.csv中，未混入平均100毫秒主曲线和8 GPU约数对照。"]
    if errors:
        lines += ["", "## 未能读取的文件", ""] + [f"- {message}" for message in errors]
    lines += ["", "生成时间：" + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), ""]
    return "\n".join(lines), {"complete": complete, "processed": count,
        "expected": len(expected), "successful": counts["completed"], "infeasible": counts["infeasible"],
        "no_validated_deployment": counts["no_validated_deployment"],
        "failed": counts["failed"], "invalid_terminal": counts["invalid_terminal"], "parse_errors": len(errors)}


def render(directory: Path, *, docs_path: Path | None = ROOT / "docs/exploring_hbf_reproduction.md"):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("result directory does not exist")
    manifest = _read(directory / "manifest.json") if (directory / "manifest.json").exists() else {}
    rows, records, errors = load_terminal_rows(directory)
    _write_csv(directory / "summary.csv", SUMMARY_FIELDS, rows)
    comparisons = eight_gpu_comparison(rows)
    _write_csv(directory / "8GPUcomparison.csv", tuple(comparisons[0]), comparisons)
    _plots(directory, rows, scope=manifest.get("scope"))
    markdown, status = _markdown(directory, manifest, rows, records, errors, comparisons)
    (directory / "report.md").write_text(markdown, encoding="utf-8")
    if docs_path is not None:
        import os
        docs_path = Path(docs_path).resolve()
        image_prefix = Path(os.path.relpath(directory, docs_path.parent)).as_posix()
        docs_markdown = markdown.replace("](throughput.png)", "](" + image_prefix + "/throughput.png)").replace("](batch.png)", "](" + image_prefix + "/batch.png)")
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(docs_markdown, encoding="utf-8")
    return {**status, "terminal_rows": len(rows), "directory": str(directory)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "artifacts/exploring_hbf_reproduction_20261019")
    parser.add_argument("--docs-path", type=Path, default=ROOT / "docs/exploring_hbf_reproduction.md")
    args = parser.parse_args()
    print(json.dumps(render(args.results_dir, docs_path=args.docs_path), ensure_ascii=False))
