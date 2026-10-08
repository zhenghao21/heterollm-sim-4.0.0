"""Report the current frontend GGUF-preset/native pairs without fitting timings.

Each comparison reuses the lifecycle, platform, physical participation and
request-boundary checks of the CUDA Graph experiment. Missing or invalid pairs
stay visible and do not contribute prediction errors or Graph speedups.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re

from analyze_cuda_graph_validation import compare
from render_cuda_graph_validation_report import METRICS, ROOT, graph_speedup, native_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cases(base):
    manifest = read(base / "cases.json")
    cases = manifest["cases"]
    if not cases:
        raise ValueError("case manifest is empty")
    seen = set()
    for case in cases:
        slug = case["case_id"]
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9_]+", slug) or slug in seen:
            raise ValueError("case IDs must be unique safe basenames")
        if not case.get("preset_id") or not case.get("name"):
            raise ValueError("each case requires a preset ID and display name")
        seen.add(slug)
    return cases


def history_changes(current, previous):
    """Separate prediction change and measurement drift; infer no causality."""
    if current.get("status") != "compared" or previous.get("status") != "compared":
        return None
    result = {}
    for key, _ in METRICS:
        new, old = current["metrics"][key], previous["metrics"][key]
        a, b = new["simulation_ns"], old["simulation_ns"]
        n, p = new["native"]["median_ns"], old["native"]["median_ns"]
        result[key] = {
            "previous_simulation_ns": b, "current_simulation_ns": a,
            "simulation_delta_ns": a - b, "simulation_delta_percent": (a / b - 1) * 100,
            "previous_native_median_ns": p, "current_native_median_ns": n,
            "native_delta_percent": (n / p - 1) * 100,
            "previous_signed_error_percent": old["signed_error_percent"],
            "current_signed_error_percent": new["signed_error_percent"],
        }
    return {"causal_attribution": "not established; runtime, costs and source inputs must be checked separately",
            "metrics": result}


def structure_summary(base, cases):
    path = base / "structure_lifecycle_validation.json"
    result = {"status": "missing", "source_file": path.name,
              "scope": "source_lifecycle_predicates_only_not_cuda_node_topology_or_timing",
              "expected_models": len(cases), "qualified_models": 0,
              "matched_invocations": 0, "mismatch_count": 0, "models": []}
    if not path.is_file():
        return result
    record = read(path)
    if record.get("schema") != "heterollm.cuda-graph-source-live-validation/v1":
        raise ValueError("unknown structure lifecycle validation schema")
    for case in cases:
        models = [model for model in record.get("models", []) if model.get("case_id") == case["case_id"]]
        if len(models) > 1:
            raise ValueError("duplicate structure lifecycle model: " + case["case_id"])
        model = models[0] if models else {}
        qualified = (model.get("qualified") is True and model.get("mismatch_count") == 0
                     and isinstance(model.get("dry_call_count"), int) and model["dry_call_count"] > 0
                     and model.get("live_call_count") == model["dry_call_count"]
                     and model.get("scope") == result["scope"]
                     and model.get("native_decisions_used_as_prediction_inputs") is False
                     and record.get("target_llm_latency_used") is False)
        result["models"].append({"case_id": case["case_id"], "qualified": qualified,
                                 "dry_call_count": model.get("dry_call_count"),
                                 "live_call_count": model.get("live_call_count"),
                                 "mismatch_count": model.get("mismatch_count")})
        result["qualified_models"] += qualified
        result["matched_invocations"] += model["dry_call_count"] if qualified else 0
        result["mismatch_count"] += model.get("mismatch_count", 0)
    result["status"] = "complete" if result["qualified_models"] == len(cases) else "incomplete"
    return result


def participation_summary(cases):
    rows = [data["comparison"] for case in cases for data in case["modes"].values()]
    paired = [row for row in rows if row["status"] == "compared"]
    passed = [row for row in paired if row.get("physical_participation", {}).get("status") == "pass"]
    return {"expected_comparisons": len(rows), "paired_comparisons": len(paired),
            "passed_comparisons": len(passed),
            "scope": "completed_paired_reports_only; activity does not establish cost accuracy",
            "cases": [{"case_id": row["case_id"], "graph_mode": row["graph_mode"],
                       "status": row.get("physical_participation", {}).get("status", "missing")}
                      for row in paired]}


def ui_run_summary(base):
    """Keep observer interruption separate from a failed backend simulation."""
    path = base / "ui_runs.json"
    result = {"status": "missing", "source_file": path.name,
              "latest_observer_status_counts": {}, "latest_simulation_status_counts": {},
              "confirmed_simulation_failed_attempts": [], "failed_attempts": [], "observer_interruptions": []}
    if not path.is_file():
        return result
    data = read(path)
    runs = data.get("runs", [])
    history = data.get("failed_attempts", [])
    interruptions = data.get("observer_interruptions", [])
    if not all(isinstance(items, list) for items in (runs, history, interruptions)):
        raise ValueError("UI runs and interruption histories must be arrays")
    fields = ("case_id", "mode", "job_id", "status", "simulation_status", "actual_result_status", "observer_status",
              "started_at", "finished_at", "recorded_at", "error", "reason", "archive_directory",
              "archive_path", "result_path", "recovery", "resolution")

    def brief(record):
        return {key: record[key] for key in fields if key in record}

    def with_result_status(record):
        resolved = dict(record)
        # A saved backend result is stronger than stale/missing observer state.
        # Never infer execution success or failure from the observer's status.
        resolved.pop("actual_result_status", None)
        result_path = record.get("result_path")
        if result_path:
            result_path = Path(result_path)
            if not result_path.is_absolute():
                result_path = base / result_path
            if result_path.is_file():
                job = read(result_path)
                if not record.get("job_id") or job.get("job_id") != record["job_id"]:
                    raise ValueError("saved backend result job_id differs from UI run: " + str(result_path))
                if not isinstance(job.get("status"), str) or not job["status"]:
                    raise ValueError("saved backend result has no status: " + str(result_path))
                resolved["actual_result_status"] = job["status"]
        return resolved

    def simulation_status(record):
        return record.get("actual_result_status", record.get("simulation_status", "unrecorded"))

    runs = [with_result_status(record) for record in runs]
    history = [with_result_status(record) for record in history]
    confirmed = {}
    for record in history + runs:
        if simulation_status(record) == "failed":
            key = record.get("job_id") or (record.get("case_id"), record.get("mode"), record.get("started_at"))
            confirmed[key] = brief(record)
    result.update(status="recorded", updated_at=data.get("updated_at"),
                  latest_observer_status_counts=dict(Counter(item.get("status", "unknown") for item in runs)),
                  latest_simulation_status_counts=dict(Counter(simulation_status(item) for item in runs)),
                  runs=[brief(item) for item in runs],
                  confirmed_simulation_failed_attempts=list(confirmed.values()),
                  failed_attempts=[brief(item) for item in history],
                  observer_interruptions=[brief(item) for item in interruptions])
    return result


def monitoring_summary(base):
    path = base / "execution_monitor.jsonl"
    result = {"status": "missing", "source_file": path.name, "sample_count": 0,
              "observed_span_seconds": None, "first_time_utc": None, "latest_time_utc": None,
              "minimum_available_memory_bytes": None, "peak_service_private_bytes": 0,
              "peak_service_rss_bytes": 0, "peak_service_process_count": 0,
              "latest_case_status_counts": {}, "latest_simulation_status_counts": {},
              "failed_cases_observed": [], "confirmed_simulation_failed_cases_observed": [],
              "trailing_partial_record": False,
              "scope": "sampled_service_processes_and_host_memory; observed span is not total experiment duration"}
    if not path.is_file():
        return result
    failures, simulation_failures = {}, {}
    # The active monitor may be appending its final line while this report reads.
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    result["trailing_partial_record"] = True
                    break
                raise
            timestamp = record["time_utc"]
            datetime.fromisoformat(timestamp)
            if result["first_time_utc"] is None:
                result["first_time_utc"] = timestamp
            result["latest_time_utc"] = timestamp
            result["sample_count"] += 1
            available = record.get("available_memory_bytes")
            minimum = result["minimum_available_memory_bytes"]
            if isinstance(available, int) and available >= 0:
                result["minimum_available_memory_bytes"] = available if minimum is None else min(minimum, available)
            processes = record.get("processes", [])
            for field, output in (("private_bytes", "peak_service_private_bytes"),
                                  ("rss_bytes", "peak_service_rss_bytes")):
                result[output] = max(result[output], sum(process.get(field, 0) for process in processes))
            result["peak_service_process_count"] = max(result["peak_service_process_count"], len(processes))
            result["latest_case_status_counts"] = dict(Counter(case.get("status", "unknown") for case in record.get("cases", [])))
            result["latest_simulation_status_counts"] = dict(Counter(case.get("simulation_status", "unrecorded") for case in record.get("cases", [])))
            for case in record.get("cases", []):
                if case.get("status") == "failed":
                    failures[(case["case_id"], case["mode"])] = {
                        "case_id": case["case_id"], "graph_mode": case["mode"],
                        "last_observed_failure_utc": timestamp, "error": case.get("error"),
                        "simulation_status": case.get("simulation_status", "unrecorded")}
                if case.get("simulation_status") == "failed":
                    simulation_failures[(case["case_id"], case["mode"])] = {
                        "case_id": case["case_id"], "graph_mode": case["mode"],
                        "last_observed_failure_utc": timestamp, "error": case.get("error")}
    if result["sample_count"]:
        result["status"] = "recorded"
        result["observed_span_seconds"] = (datetime.fromisoformat(result["latest_time_utc"])
                                             - datetime.fromisoformat(result["first_time_utc"])).total_seconds()
    result["failed_cases_observed"] = list(failures.values())
    result["confirmed_simulation_failed_cases_observed"] = list(simulation_failures.values())
    return result


def collect(base, historical):
    output = []
    for case in load_cases(base):
        slug = case["case_id"]
        modes = {}
        for mode in ("off", "on"):
            row = compare(base, slug, mode)
            row["preset_id"] = case["preset_id"]
            # A completed imported scenario must still identify the chosen
            # original preset; editing a virtual derivative is not native parity.
            submission = base / f"ui_{slug}_graph_{mode}_submission.json"
            if submission.is_file():
                metadata = read(submission)["scenario"]["model"].get("metadata", {})
                errors = []
                if metadata.get("model_preset_id") != case["preset_id"]:
                    errors.append("submitted model is not the selected original GGUF preset")
                if metadata.get("gguf_preset_changes"):
                    errors.append("modified preset cannot use the original GGUF native timing")
                if errors:
                    row.setdefault("validation_errors", []).extend(errors)
                    row["status"] = "configuration_mismatch"
                    row.pop("metrics", None)
            (base / f"comparison_{slug}_graph_{mode}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            measured_path = base / f"native_{slug}_graph_{mode}.json"
            stats, native_error = None, None
            if measured_path.is_file():
                try:
                    stats = native_metrics(measured_path, mode)
                except (ValueError, KeyError, TypeError) as exc:
                    native_error = str(exc)
            prior_path = historical / f"comparison_{slug}_graph_{mode}.json"
            prior = read(prior_path) if prior_path.is_file() else {}
            modes[mode] = {"comparison": row, "native": stats, "native_error": native_error,
                           "history": history_changes(row, prior)}
        gain = graph_speedup(modes) if all(modes[m]["native"] for m in ("off", "on")) else None
        output.append({**case, "modes": modes, "graph_speedup": gain})
    return {"schema": "heterollm.frontend-gguf-native-validation/v1",
            "completed_comparisons": sum(m["comparison"]["status"] == "compared"
                                         for c in output for m in c["modes"].values()),
            "expected_comparisons": len(output) * 2,
            "target_llm_latency_used_for_calibration": False,
            "generalization_qualified": False, "cases": output,
            "structure_lifecycle_summary": structure_summary(base, output),
            "physical_participation_summary": participation_summary(output),
            "execution_monitor_summary": monitoring_summary(base),
            "ui_run_summary": ui_run_summary(base)}


def render(base, summary):
    rows, gains, histories = [], [], []
    native_completed = 0
    for case in summary["cases"]:
        slug, name = case["case_id"], escape(case["name"])
        for mode, data in case["modes"].items():
            row, native = data["comparison"], data["native"]
            native_completed += native is not None
            valid = row["status"] == "compared"
            cells = []
            for key, _ in METRICS:
                if native:
                    s = native[key]
                    cells.append(f'<td>{s["median_ns"]/1e6:.3f}<small>MAD {s["mad_ns"]/1e6:.3f} ms</small>'
                                 f'<small>{s["min_ns"]/1e6:.3f}–{s["max_ns"]/1e6:.3f} ms</small></td>')
                else:
                    cells.append('<td class="pending">未完成有效实测</td>')
                if valid:
                    p = row["metrics"][key]
                    cells.append(f'<td>{p["simulation_ns"]/1e6:.3f}'
                                 f'<small>绝对误差 {p["absolute_error_ns"]/1e6:.3f} ms</small>'
                                 f'<small>有符号误差 {p["signed_error_percent"]:+.2f}%</small></td>')
                else:
                    cells.append('<td class="pending">未完成有效配对</td>')
            errors = row.get("validation_errors") or row.get("missing_files") or [row.get("error") or row["status"]]
            status = '已配对' if valid else escape('; '.join(str(v) for v in errors))
            rows.append(f'<tr><th>{name}</th><td>{"开启" if mode == "on" else "关闭"}</td>'
                        + ''.join(cells) + f'<td class="status"><a href="comparison_{slug}_graph_{mode}.json">'
                        + status + '</a></td></tr>')
            if data["history"]:
                for key, label in METRICS:
                    h = data["history"]["metrics"][key]
                    histories.append(f'<tr><th>{name}</th><td>{mode}</td><td>{label}</td>'
                        f'<td>{h["simulation_delta_ns"]/1e6:+.4f} ms ({h["simulation_delta_percent"]:+.3f}%)</td>'
                        f'<td>{h["native_delta_percent"]:+.3f}%</td>'
                        f'<td>{h["previous_signed_error_percent"]:+.2f}% → {h["current_signed_error_percent"]:+.2f}%</td></tr>')
        g = case["graph_speedup"]
        if g:
            predicted = g["simulation_e2e_reduction_percent"]
            error = g["signed_reduction_error_percentage_points"]
            gains.append(f'<tr><th>{name}</th><td>{g["native_e2e_reduction_percent"]:.2f}%</td>'
                         + (f'<td>{predicted:+.2f}%</td><td>{error:+.2f} 个百分点</td>'
                            if predicted is not None else '<td>未配对</td><td>未配对</td>') + '</tr>')
    expected = summary["expected_comparisons"]
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GGUF 模型预设 · 前端仿真与 native 对照</title><style>
body{margin:0;padding:28px;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#182638}main{max-width:1500px;margin:auto}h1{font-size:28px}h2{font-size:21px;margin-top:30px}.card{overflow:auto;background:white;border:1px solid #dce4ef;border-radius:12px;padding:16px;margin:20px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #e5ebf3;text-align:left;white-space:nowrap}thead{background:#eaf0f7}small{display:block;font-size:12px;color:#52657a}.status{white-space:normal;min-width:90px;max-width:300px}.pending{color:#8b6117}.note{border-left:4px solid #b8842a;padding-left:14px}a{color:#235cb2}.tags{font-weight:600;color:#284d70}summary{cursor:pointer;font-weight:600;color:#284d70;padding:8px 0}.ranges p{margin:4px 0}</style>
<main><p>2026-10-08 · RTX 5080 / Ryzen 9 9950X3D / GDDR7 / DDR5</p><h1>GGUF 模型预设：前端仿真与 native 对照</h1>
<p>从前端模型预设选择真实 GGUF 模型，加载本机硬件及 llama 调度。无独立控件的运行配置通过高级 JSON，或导出后补充配置再由前端导入；模型执行结构保持与选中预设一致。图结构与独立成本准备完成后，由前端导入、校验并提交。预设目录构图与 native 使用同一份完整 GGUF 权重身份。</p>
<p>统一 512 输入 / 128 输出，单序列，context 768，batch/ubatch 512，16 线程，FP16 KV，Flash Attention 关闭，无提示缓存、无 MTP，ctx-checkpoints=0。native 每组 2 次预热、5 次正式测量；仿真显式执行启动及预热，仅比较正式请求的 engine 边界。单位为 ms；正误差表示仿真高估，MAD 表示重复实测的波动。</p>
<p>本轮修复了 GGUF 构图遗漏 Qwen2.5 Q/K/V 偏置运算声明的问题，并连接偏置后的投影缓冲区依赖；Llama 保留并读取真实 RoPE 频率因子，计入相应读取和除法。Qwen2/Llama 的 F32 中间值与 F16 KV 源路径已补齐。表内仿真来自修复后的前端提交，未根据 native 时延反推补偿系数。</p>
'''
    html += f'<p class="tags">{len(summary["cases"])} 个模型 · {native_completed}/{expected} 组 native 有效实测 · {summary["completed_comparisons"]}/{expected} 组有效配对</p>'
    html += '<div class="ranges"><small>以下为本机本批模型的有符号误差范围，仅统计有效配对；正值为高估，负值为低估。</small>'
    for mode, mode_label in (("off", "关闭"), ("on", "开启")):
        paired = [case["modes"][mode]["comparison"] for case in summary["cases"]
                  if case["modes"][mode]["comparison"]["status"] == "compared"]
        ranges = []
        for (key, _), label in zip(METRICS, ("首 token（TTFT）", "后续每 token（TPOT）", "总耗时（E2E）")):
            values = [row["metrics"][key]["signed_error_percent"] for row in paired]
            if values:
                ranges.append(f'{label} {min(values):+.2f}% ～ {max(values):+.2f}%')
        html += (f'<p><strong>Graph {mode_label} · {len(paired)} 组：</strong>'
                 + ('；'.join(ranges) if paired else '尚无有效配对误差') + '</p>')
    html += '</div><details><summary>查看执行检查与本轮修复</summary>'
    structure = summary["structure_lifecycle_summary"]
    participation = summary["physical_participation_summary"]
    monitor = summary["execution_monitor_summary"]
    ui_runs = summary["ui_run_summary"]
    html += '<h2>结构与执行检查</h2>'
    if structure["status"] != "missing":
        html += (f'<p>源码生命周期条件核验：{structure["qualified_models"]}/{structure["expected_models"]} 个模型通过，'
                 f'通过模型合计 {structure["matched_invocations"]} 次调用，记录中的条件差异共 {structure["mismatch_count"]} 处。'
                 '该检查仅比较源码生命周期条件，不验证 CUDA 节点拓扑或预测时延。'
                 '<a href="structure_lifecycle_validation.json">结构核验记录</a>。</p>')
    else:
        html += '<p class="pending">尚无结构生命周期核验记录。</p>'
    html += (f'<p>已完成有效配对的 {participation["paired_comparisons"]} 组中，'
             f'{participation["passed_comparisons"]} 组通过物理 DRAM 读写与 CPU/GPU 活动检查；'
             f'本轮目标共 {participation["expected_comparisons"]} 组。检查依据实际运行计数，不表示每类成本均已完整或准确建模。'
             '各组参与明细见表格中的配对记录。</p>')
    if monitor["status"] == "recorded":
        status_text = '、'.join(f'{escape(key)} {value}' for key, value in monitor["latest_case_status_counts"].items())
        available = monitor["minimum_available_memory_bytes"]
        memory_text = f'{available / 1024**3:.2f} GiB' if available is not None else '未记录'
        html += (f'<p>运行监控截至 {escape(monitor["latest_time_utc"])}（UTC）：{monitor["sample_count"]} 个采样点，'
                 f'覆盖 {monitor["observed_span_seconds"] / 60:.1f} 分钟；最近观察记录状态：{status_text or "未记录"}。'
                 f'同时监控服务最多 {monitor["peak_service_process_count"]} 个，服务私有内存合计采样峰值 '
                 f'{monitor["peak_service_private_bytes"] / 1024**3:.2f} GiB，主机可用内存采样最低值 {memory_text}。'
                 f'观察记录曾将 {len(monitor["failed_cases_observed"])} 组标为 failed；旧采样没有独立保存后端状态，'
                 '这个数字可能包含 UI/观察器中断，不能直接当作仿真失败数。'
                 '采样覆盖时段不是整个实验耗时，内存采样值也不是连续监测的极值。'
                 '<a href="execution_monitor.jsonl">原始监控</a>。</p>')
    else:
        html += '<p class="pending">尚无完整运行监控采样。</p>'
    if ui_runs["status"] == "recorded":
        simulation_status_text = '、'.join(f'{escape(key)} {value}' for key, value in ui_runs["latest_simulation_status_counts"].items())
        html += (f'<p>执行清单中明确记录的后端状态：{simulation_status_text or "未记录"}；'
                 f'已确认后端失败尝试 {len(ui_runs["confirmed_simulation_failed_attempts"])} 次，'
                 f'另有观察器/UI 中断记录 {len(ui_runs["observer_interruptions"])} 条。'
                 '后端状态优先读取任务编号一致的已保存结果，其次使用明确记录的 simulation_status；'
                 '缺少两者时标为 unrecorded。只有明确的后端 failed 状态计入失败尝试；'
                 '界面或观察器中断不代表仍在运行的后端任务失败。失败尝试和中断历史独立保留，'
                 '当前重试结果见各组配对状态。<a href="ui_runs.json">执行与恢复清单</a>。</p>')
    if (base / "softmax_read_extent_validation.json").is_file():
        html += ('<h2>本轮发现并修复的执行问题</h2>'
                 '<p>Qwen3 14B 的 512 token 输入阶段生成 40 MiB 的 attention score 缓冲区。'
                 'softmax 两遍处理在缓存作用后仍需累计读取 47,028,634 字节；旧地址映射误把多遍累计流量当成单段连续地址，'
                 '因此触发越界检查。修复后按已声明的处理遍次重复访问同一缓冲区，保留原有成本、流量和越界拒绝机制。</p>'
                 '<p>10 个模型各 4 种形状，共 40 个修复前后检查：39 项成本、资源需求和最终物理访问严格相同；'
                 '唯一变化是 Qwen3 14B 的上述输入阶段由越界失败变为可执行。该单层统一事件检查完成 53/53 个任务，'
                 'DRAM 读取为 47,028,672 字节（按突发粒度取整），写入为 41,943,040 字节。'
                 '这些检查说明修复范围，不能代替完整模型仿真或 native 精度比较。'
                 '<a href="softmax_read_extent_validation.json">40 项修复前后检查</a>。</p>'
                 '<p>14B 首次失败的提交、任务及进度结果保存在 '
                 '<a href="failed_attempts/b86b23311849421a9bcbcd6dab52f42e/ui_qwen3_14b_q4_k_m_graph_off_result.json">原失败记录</a>，'
                 '修复后的正式重试是否完成，以本报告的配对状态为准。另一次观察文件写入异常导致部分浏览器页面关闭，'
                 '后端任务继续运行；观察器恢复沿用原任务编号，不能把该 UI 中断误报为后端仿真失败。</p>'
                 '<p>汇总报告仍重复保存逐批执行计划和可视化成本元数据，已完成样本的格式化文件约 150 MiB。'
                 '结果接收期间还发生了无错误日志的 Node 进程退出，目前没有证据确认其退出原因。'
                 '后续使用 Python 串行流式读取原任务结果，以减少同时驻留的报告副本；'
                 '所有任务仍由前端提交，恢复只读取原任务，不重新提交，也不改变预测值。'
                 '恢复后未重新验证各任务在原前端结果页中的渲染。报告重复数据造成的体积问题尚待单独优化。</p>')
    html += '</details><div class="card"><table><thead><tr><th rowspan="2">模型</th><th rowspan="2">Graph</th>'
    html += ''.join(f'<th colspan="2">{label}</th>' for _, label in METRICS)
    html += '<th rowspan="2">配对状态</th></tr><tr>' + '<th>native</th><th>仿真</th>' * 3
    html += '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    html += '<h2>Graph 开启后的总耗时降低比例</h2><p>降低比例 = (关闭耗时 − 开启耗时) / 关闭耗时。预测误差 = 仿真降低比例 − native 降低比例，单位为百分点。开关选项之外，还核对实际 Graph 生命周期事件。</p><div class="card"><table><thead><tr><th>模型</th><th>native</th><th>仿真</th><th>预测误差</th></tr></thead><tbody>' + ''.join(gains) + '</tbody></table></div>'
    html += '<h2>原五模型与历史结果的变化</h2><p class="note">分别列出仿真变化和本轮 native 重测变化。工作区还包含此前的运行时与硬件配置修改，以下差值不能单独归因于 GGUF 预设迁移。历史与当前误差各自使用对应时点的 native 数据。</p><div class="card"><table><thead><tr><th>模型</th><th>Graph</th><th>指标</th><th>仿真变化</th><th>native 中位数变化</th><th>有符号误差变化</th></tr></thead><tbody>' + ''.join(histories) + '</tbody></table></div>'
    html += '<h2>解释范围</h2><p>独立 CUDA 成本来自匹配类型化拓扑的小程序测量，不使用模型端到端实测时延拟合。结构准备调用固定版本的 llama.cpp/CUDA 图构建器，不执行模型推理。native 推理与插桩诊断仅用于最终核验。</p><p class="note">这是实验预测路径。合成 kernel 的 host API 成本代表性、Graph 设备端调度收益仍有限制；embedding 量化转换计算、采样候选与过滤链、完成中断仍有未完整计价项。设备端启动沿用原有分析参数，独立测量的是 host API 成本；不能用当前整体预测残差定量归因于其中某一项。通过配置、生命周期及物理 DRAM 参与检查不等于精度合格，也不证明对其他硬件或未测试模型的泛化性。任何失败或缺失项保留为空，不能由历史结果填补。</p><p><a href="comparison_summary.json">完整比较数据</a> · <a href="cases.json">模型清单</a> · <a href="preparation.json">输入与来源记录</a> · <a href="ui_preparation.json">前端预设配置过程</a></p></main></html>'
    (base / "report.html").write_text(html, encoding="utf-8")
    (base / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return base / "report.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/gguf_preset_native_validation_2026-10-08")
    parser.add_argument("--historical", type=Path, default=ROOT / "docs/cuda_graph_validation_2026-10-08")
    args = parser.parse_args()
    summary = collect(args.output, args.historical)
    print(render(args.output, summary))
    print(f'{summary["completed_comparisons"]}/{summary["expected_comparisons"]} compared')


if __name__ == "__main__":
    main()
