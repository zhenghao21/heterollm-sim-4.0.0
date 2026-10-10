"""Report the October 10 cost-logic rerun only after all results are terminal.

Reuse the existing GGUF/native comparison gates. Read each large job once,
retain only fields those gates inspect, and never execute a simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gzip
from html import escape
import json
import math
from pathlib import Path
import statistics

import analyze_cuda_graph_validation as graph_analysis
import analyze_gguf_preset_validation as gguf_analysis
import render_cuda_graph_validation_report as graph_report
from validate_cuda_graph_dispatch_changes import _input_participation_summary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "docs/cuda_graph_dispatch_validation_2026-10-08"
DEFAULT_OUTPUT = ROOT / "docs/cost_logic_native_validation_2026-10-10"
METRICS = [("engine_ttft_ns", "TTFT 首 token"), ("engine_tpot_ns", "TPOT 后续每 token"),
           ("engine_e2e_ns", "E2E 请求总耗时")]


def read_json(path):
    path = Path(path)
    compressed = Path(str(path) + ".gz")
    source = path if path.is_file() else compressed
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def compact_job(job):
    """Keep every input used by compare(), including all lifecycle predicates."""
    if not isinstance(job.get("report"), dict):
        return job
    report = job["report"]
    lifecycle = graph_analysis.find_lifecycle(report)
    retained = {key: report[key] for key in (
        "summary", "resource_utilization", "category_time_ns", "critical_path_category_ns",
        "analytical_coverage", "measurement_semantics", "requests",
    ) if key in report}
    if lifecycle is not None:
        lifecycle = {key: lifecycle[key] for key in (
            "remaining_compiled_invocations", "event_counts", "transitions",
        ) if key in lifecycle}
        lifecycle["transitions"] = [{key: transition[key] for key in (
            "events", "body_executions", "capture_executes_body", "pricing_ready",
            "unresolved_update_count",
        ) if key in transition} for transition in lifecycle.get("transitions", [])]
        retained["summary"] = {**retained.get("summary", {}), "llama_cuda_graph_lifecycle": lifecycle}
    return {**{key: value for key, value in job.items() if key != "report"}, "report": retained}


@contextmanager
def cached_comparison_reads():
    """Patch reporting readers locally, without changing product/tool sources."""
    cache = {}

    def read(path):
        key = str(Path(path).resolve())
        if key not in cache:
            value = read_json(path)
            cache[key] = compact_job(value) if "_result.json" in Path(path).name else value
        return cache[key]

    modules = (graph_analysis, gguf_analysis, graph_report)
    originals = [module.read for module in modules]
    for module in modules:
        module.read = read
    try:
        yield read
    finally:
        for module, original in zip(modules, originals):
            module.read = original


def metric_error_summary(cases, key, mode=None):
    rows = []
    for case in cases:
        for graph_mode, data in case["modes"].items():
            if mode is not None and graph_mode != mode:
                continue
            current = data["comparison"]
            if current.get("status") != "compared":
                continue
            new = current["metrics"][key]["signed_error_percent"]
            prior = (data.get("history") or {}).get("metrics", {}).get(key)
            rows.append({"case_id": case["case_id"], "graph_mode": graph_mode,
                         "current_signed_error_percent": new,
                         "previous_signed_error_percent": prior["previous_signed_error_percent"] if prior else None})
    matched = [row for row in rows if row["previous_signed_error_percent"] is not None]
    delta = [abs(row["previous_signed_error_percent"]) - abs(row["current_signed_error_percent"]) for row in matched]
    unchanged = [math.isclose(value, 0.0, rel_tol=0, abs_tol=1e-10) for value in delta]
    values = [row["current_signed_error_percent"] for row in rows]
    return {"metric": key, "graph_mode": mode or "all", "current_pairs": len(rows),
            "matched_old_new_pairs": len(matched), "weighting": "equal weight per model and Graph mode",
            "current_mape_percent": statistics.mean(map(abs, values)) if values else None,
            "matched_current_mape_percent": statistics.mean(abs(row["current_signed_error_percent"]) for row in matched) if matched else None,
            "previous_mape_percent": statistics.mean(abs(row["previous_signed_error_percent"]) for row in matched) if matched else None,
            "signed_error_min_percent": min(values) if values else None,
            "signed_error_max_percent": max(values) if values else None,
            "floating_point_unchanged_tolerance_percentage_points": 1e-10,
            "improved_pairs": sum(value > 0 and not same for value, same in zip(delta, unchanged)),
            "worsened_pairs": sum(value < 0 and not same for value, same in zip(delta, unchanged)),
            "unchanged_pairs": sum(unchanged), "rows": rows}


def wall_clock_summary(results):
    starts, finishes, durations, rows = [], [], [], []
    for result in results:
        start, finish = result.get("started_at"), result.get("finished_at")
        row = {key: result.get(key) for key in ("case_id", "graph_mode", "job_id", "status", "started_at", "finished_at")}
        duration = None
        if start is not None and finish is not None:
            a, b = datetime.fromisoformat(start), datetime.fromisoformat(finish)
            if a.tzinfo is None or b.tzinfo is None:
                raise ValueError("job timestamps must include their timezone")
            duration = (b - a).total_seconds()
            if duration < 0:
                raise ValueError("job finished before it started")
            starts.append(a)
            finishes.append(b)
            durations.append(duration)
        rows.append({**row, "wall_seconds": duration})
    return {"scope": "backend started_at to finished_at; 20 concurrent jobs share CPU and memory; not isolated speed benchmark",
            "expected_jobs": len(results), "timed_jobs": len(durations),
            "minimum_wall_seconds": min(durations) if durations else None,
            "maximum_wall_seconds": max(durations) if durations else None,
            "median_wall_seconds": statistics.median(durations) if durations else None,
            "first_started_at": min(starts).isoformat() if starts else None,
            "last_finished_at": max(finishes).isoformat() if finishes else None,
            "observed_execution_span_seconds": (max(finishes) - min(starts)).total_seconds() if starts else None,
            "rows": rows}


def checked_collect(output, baseline):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if output == baseline:
        raise ValueError("rerun must not overwrite baseline")
    provenance = read_json(output / "correction_inputs.json")
    if (provenance.get("native_measurements_reused") is not True
            or provenance.get("scenario_inputs_changed") is not False
            or provenance.get("native_values_used_for_calibration") is True
            or provenance.get("device_dispatch_calibration_installed") is True):
        raise ValueError("rerun provenance is inconsistent")
    cases = gguf_analysis.load_cases(output)
    if len(cases) != 10:
        raise ValueError("this rerun requires ten models")
    timing = read_json(output / "native_timing_manifest.json")
    if timing != read_json(baseline / "native_timing_manifest.json"):
        raise ValueError("reused native measurement manifest differs from baseline")
    with cached_comparison_reads() as read:
        native_reuse, input_reuse, terminal = [], [], []
        for case in cases:
            for mode in ("off", "on"):
                slug = case["case_id"]
                result_path = output / f"ui_{slug}_graph_{mode}_result.json"
                # The shared compare() gate requires the original .json path.
                if not result_path.is_file():
                    raise ValueError("all 20 original JSON results must be saved before reporting: " + result_path.name)
                job = read(result_path)
                if job.get("status") not in {"completed", "failed", "cancelled"}:
                    raise ValueError("nonterminal simulation result: " + result_path.name)
                terminal.append({"case_id": slug, "graph_mode": mode, "job_id": job.get("job_id"),
                                 "status": job.get("status"), "started_at": job.get("started_at"),
                                 "finished_at": job.get("finished_at"), "error": job.get("error")})
                native_name = f"native_{slug}_graph_{mode}.json"
                equal = read(output / native_name) == read(baseline / native_name)
                if not equal:
                    raise ValueError("reused native record differs from baseline: " + native_name)
                if read(output / f"scenario_{slug}_graph_{mode}.json") != read(baseline / f"scenario_{slug}_graph_{mode}.json"):
                    raise ValueError("prepared rerun input differs from baseline: " + slug + " " + mode)
                submission_name = f"ui_{slug}_graph_{mode}_submission.json"
                submission = read(output / submission_name)
                if submission != read(baseline / submission_name):
                    raise ValueError("actual frontend POST differs from baseline: " + submission_name)
                scenario = submission["scenario"]
                metadata = scenario["model"].get("metadata", {})
                if "gguf_preset_changes" not in metadata or metadata["gguf_preset_changes"] != {}:
                    raise ValueError("actual frontend POST must explicitly retain changes={}: " + submission_name)
                measured_id = scenario["workload"]["metadata"]["cuda_graph_comparison_request_id"]
                requests = [request for request in scenario["workload"]["requests"] if request["request_id"] == measured_id]
                if len(requests) != 1 or (requests[0].get("prompt_tokens"), requests[0].get("output_tokens")) != (512, 128):
                    raise ValueError("actual frontend measured request is not 512/128: " + submission_name)
                input_reuse.append({"case_id": slug, "graph_mode": mode,
                                    "same_prepared_scenario": True, "same_actual_frontend_post": True,
                                    "gguf_preset_changes": {}, "measured_request_id": measured_id,
                                    "prompt_tokens": 512, "output_tokens": 128})
                native_reuse.append({"case_id": slug, "graph_mode": mode, "same_native_record": equal})
        summary = gguf_analysis.collect(output, baseline)
    summary.update(schema="heterollm.cost-logic-native-rerun/v1", correction_provenance=provenance,
                   baseline_directory=str(baseline), native_reuse_checks=native_reuse,
                   input_reuse_checks=input_reuse,
                   terminal_results=terminal, backend_status_counts=dict(Counter(row["status"] for row in terminal)))
    summary["gguf_input_participation_summary"] = _input_participation_summary(output, summary["cases"])
    summary["metric_error_summaries"] = [metric_error_summary(summary["cases"], key, mode)
                                         for mode in (None, "off", "on") for key, _ in METRICS]
    summary["native_timing_manifest"] = timing
    summary["wall_clock_summary"] = wall_clock_summary(terminal)
    provenance.update(schema="heterollm.cost-logic-rerun-inputs/v1",
                      scope="Rerun unchanged original GGUF frontend inputs after L2 energy/lookup timing, F16 physical identity and invocation-index lifetime, and CUDA Graph device-root submission dependency fixes; native timings reused without fitting.")
    (output / "correction_inputs.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def number(value, suffix="", signed=False):
    return "—" if value is None else (f"{value:+.2f}" if signed else f"{value:.2f}") + suffix


def render(output, summary):
    rows, gains, aggregate = [], [], []
    for item in summary["metric_error_summaries"]:
        label = dict(METRICS)[item["metric"]]
        graph_label = {"all": "开/关合计", "off": "关闭", "on": "开启"}[item["graph_mode"]]
        aggregate.append(f'<tr><td>{graph_label}</td><td>{label}</td><td>{item["current_pairs"]} / {item["matched_old_new_pairs"]}</td>'
                         f'<td>{number(item["previous_mape_percent"], "%")}</td>'
                         f'<td>{number(item["matched_current_mape_percent"], "%")}</td>'
                         f'<td>{number(item["signed_error_min_percent"], "%", True)} ～ {number(item["signed_error_max_percent"], "%", True)}</td>'
                         f'<td>{item["improved_pairs"]} / {item["worsened_pairs"]} / {item["unchanged_pairs"]}</td></tr>')
    for case in summary["cases"]:
        for mode, data in case["modes"].items():
            row = data["comparison"]
            start = f'<th>{escape(case["name"])}</th><td>{"开启" if mode == "on" else "关闭"}</td>'
            if row["status"] != "compared":
                detail = row.get("error") or "; ".join(map(str, row.get("validation_errors", row.get("missing_files", []))))
                rows.append('<tr>' + start + f'<td colspan="8">{escape(row["status"])} · {escape(str(detail))}</td></tr>')
                continue
            for key, label in METRICS:
                metric = row["metrics"][key]
                old = (data.get("history") or {}).get("metrics", {}).get(key, {})
                reduction = abs(old["previous_signed_error_percent"]) - abs(metric["signed_error_percent"]) if old else None
                rows.append('<tr>' + start + f'<td>{label}</td><td>{metric["native"]["median_ns"]/1e6:.4f}'
                            f'<small>MAD {metric["native"]["mad_ns"]/1e6:.4f}</small></td>'
                            f'<td>{number(old.get("previous_simulation_ns", None) / 1e6 if old else None)}</td>'
                            f'<td>{metric["simulation_ns"]/1e6:.4f}</td>'
                            f'<td>{number(old.get("previous_signed_error_percent"), "%", True)}</td>'
                            f'<td>{number(metric["signed_error_percent"], "%", True)}</td>'
                            f'<td>{number(old.get("simulation_delta_percent"), "%", True)}</td>'
                            f'<td>{number(reduction, " pp", True)}</td></tr>')
        gain = case.get("graph_speedup") or {}
        previous = read_json(Path(summary["baseline_directory"]) / f'comparison_{case["case_id"]}_graph_off.json')
        previous_on = read_json(Path(summary["baseline_directory"]) / f'comparison_{case["case_id"]}_graph_on.json')
        old_gain = ((1 - previous_on["metrics"]["engine_e2e_ns"]["simulation_ns"] / previous["metrics"]["engine_e2e_ns"]["simulation_ns"]) * 100
                    if previous.get("status") == previous_on.get("status") == "compared" else None)
        gain["previous_simulation_e2e_reduction_percent"] = old_gain
        gains.append(f'<tr><th>{escape(case["name"])}</th><td>{number(gain.get("native_e2e_reduction_percent"), "%")}</td>'
                     f'<td>{number(old_gain, "%")}</td><td>{number(gain.get("simulation_e2e_reduction_percent"), "%")}</td>'
                     f'<td>{number(gain.get("signed_reduction_error_percentage_points"), " pp", True)}</td></tr>')
    inputs = summary["gguf_input_participation_summary"]
    timing = summary["native_timing_manifest"]
    status = summary["backend_status_counts"]
    ui = summary["ui_run_summary"]
    wall = summary["wall_clock_summary"]
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>成本逻辑修复后：仿真与 native 对照</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#18263a;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:26px 18px}h1{font-size:27px}h2{font-size:20px;margin-top:28px}.table{overflow:auto;background:white;border:1px solid #dbe3ec;border-radius:9px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:right;border-bottom:1px solid #e3e8ef}th:first-child,td:first-child{text-align:left}thead{background:#e9eff7}tbody tr:nth-child(even){background:#f8fafc}small{display:block;color:#52657a;font-size:12px}.note{border-left:4px solid #a77828;padding-left:14px}a{color:#075ca5}li{margin:8px 0}p{max-width:1180px}@media(max-width:600px){main{padding:16px 10px}h1{font-size:23px}}</style><main>
<p>2026-10-10 · RTX 5080 / Ryzen 9 9950X3D</p><h1>成本逻辑修复后：仿真与 native 对照</h1>
<p>本轮在当前修复版上，使用原 10 个真实 GGUF 预设和原输入，由前端提交 Graph 开/关共 20 组。负载为 512 输入 / 128 输出，单序列，context 768，batch/ubatch 512，16 线程，FP16 KV，Flash Attention、提示缓存、MTP 均关闭，ctx-checkpoints=0。比较正式请求的 engine 边界，排除启动、预热、排队和 HTTP 传输。</p>'''
    html += f'<p>有效配对 {summary["completed_comparisons"]}/{summary["expected_comparisons"]}；后端最终完成 {status.get("completed", 0)} 组、失败 {status.get("failed", 0)} 组、取消 {status.get("cancelled", 0)} 组。'
    html += f'实际提交中原 GGUF 预设 {inputs["original_gguf_presets"]}/20，显式 changes={{}} {inputs["explicit_empty_changes"]}/20；物理 DRAM/CPU/GPU 活动检查 {inputs["physical_participation_passed"]}/20 通过。</p>'
    html += '<p>20 份准备场景及 20 份实际前端 POST（提交内容）分别与基线对应文件逐份相同；每组指定正式请求均为 512/128，核对基于实际提交内容。</p>'
    html += f'<p>native 复用 {escape(timing.get("started_local", "未记录"))} 至 {escape(timing.get("finished_local", "未记录"))}（中国标准时间）的已核实实测，20 份记录逐份与基线相同，未在本轮重测。每组 2 次预热、5 次正式测量；未用模型 native 时延调整成本。</p>'
    html += '<h2>本轮修复范围</h2><ul><li>L2（GPU 二级缓存）：按本次访问字节重算缓存能耗，缓存查询结束后才启动缺失填充及脏数据回写；物理完成时间回填设备占用与保留的成本视图。</li><li>F16 物理身份：融合与拆分投影按同一原始矩阵身份访问，显式内存分配身份优先；常驻与临时缓冲区按声明区分。KV set_rows 的索引改为调用内临时分配，避免误随 KV 常驻及错误复用。</li><li>Graph roots（图中没有内部前驱的任务）：包括复制和零成本标记，均等待主机提交结束；保留原设备依赖顺序。</li></ul>'
    html += '<p>已有预检：Graph 84 项、L2/indices 71 项、projection/direct/identity 40 项、前端 JS 39 项；组之间可重叠，不相加。它们验证修复行为，完整模型的误差由下表另行给出。旧报告中的偏置、RoPE、softmax 等修复不属于本轮新增。多处成本与依赖同时变化，差值不能逐项归因。</p>'
    html += '<p class="note">流程、生命周期和物理参与通过不等于预测准确。当前 Graph 设备调度及独立成本仍有未解决的精度与泛化限制，本批结果只代表本机、这些模型和该负载。</p>'
    html += '<h2>误差汇总</h2><p>有符号误差 = (仿真 / native 中位数 − 1) × 100%，正值为高估。MAPE 为绝对百分比误差的等权平均，每个模型及 Graph 模式各占一份。新旧 MAPE 仅使用两轮都有效的配对；改善按绝对误差下降计数，差值小于 10⁻¹⁰ 个百分点的浮点舍入差异计作相同。pp 表示百分点。</p>'
    html += '<div class="table"><table><thead><tr><th>Graph</th><th>指标</th><th>本轮 / 新旧配对组</th><th>旧 MAPE</th><th>新 MAPE</th><th>新有符号误差范围</th><th>改善 / 变差 / 相同</th></tr></thead><tbody>' + ''.join(aggregate) + '</tbody></table></div>'
    html += '<h2>逐模型：修复前后与同一 native 对照</h2><p>耗时单位 ms；native 显示中位数及 MAD（各次测量与中位数的绝对偏差中位数）。仿真变化 = 新 / 旧 − 1；绝对误差下降为正表示改善。</p>'
    html += '<div class="table"><table><thead><tr><th>模型</th><th>Graph</th><th>指标</th><th>native / MAD</th><th>旧仿真</th><th>新仿真</th><th>旧误差</th><th>新误差</th><th>仿真变化</th><th>绝对误差下降</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    html += '<h2>Graph 的总耗时收益</h2><p>降低比例 = (关闭 − 开启) / 关闭 × 100%。收益误差 = 新仿真降幅 − native 降幅；正值表示高估 Graph 收益。</p><div class="table"><table><thead><tr><th>模型</th><th>native 降幅</th><th>旧仿真降幅</th><th>新仿真降幅</th><th>新收益误差</th></tr></thead><tbody>' + ''.join(gains) + '</tbody></table></div>'
    wall_rows = []
    names = {case["case_id"]: case["name"] for case in summary["cases"]}
    for item in wall["rows"]:
        status_label = {"completed": "完成", "failed": "失败", "cancelled": "取消"}.get(item["status"], str(item["status"]))
        wall_rows.append(f'<tr><th>{escape(names[item["case_id"]])}</th><td>{"开启" if item["graph_mode"] == "on" else "关闭"}</td>'
                         f'<td>{escape(status_label)}</td><td>{escape(str(item["started_at"]))}</td>'
                         f'<td>{escape(str(item["finished_at"]))}</td><td>{number(item["wall_seconds"])}</td></tr>')
    html += f'<h2>本轮实际计算耗时</h2><p>依据各后端任务真实 started_at / finished_at，已记录 {wall["timed_jobs"]}/20 组；'
    html += f'每组墙钟范围 {number(wall["minimum_wall_seconds"])}–{number(wall["maximum_wall_seconds"])} 秒，中位数 {number(wall["median_wall_seconds"])} 秒。'
    html += f'从最早任务开始到最后任务结束共 {number(wall["observed_execution_span_seconds"])} 秒。</p>'
    html += '<p class="note">20 个任务并行运行，竞争同一主机的 CPU 和内存；这些耗时是本轮执行记录，不是单任务独占资源的速度基准，也不是模型预测的请求时延。不能把 20 组墙钟相加当作实验历时；前一轮微性能检查也不等于本轮完整模型耗时。下表时间带有 UTC 时区。</p>'
    html += '<details><summary>查看各组开始、结束和墙钟耗时</summary><div class="table"><table><thead><tr><th>模型</th><th>Graph</th><th>后端状态</th><th>开始时间</th><th>结束时间</th><th>墙钟秒</th></tr></thead><tbody>' + ''.join(wall_rows) + '</tbody></table></div></details>'
    html += f'<h2>执行与查看范围</h2><p>20 组由真实前端导入、校验、提交后关闭页面，随后串行读取已提交任务的结果。最终后端状态按保存结果核对；已记录后端失败尝试 {len(ui.get("confirmed_simulation_failed_attempts", []))} 次、观察器/UI 中断 {len(ui.get("observer_interruptions", []))} 条，两者分别统计。前端结果页的渲染覆盖范围以代表性检查记录为准。</p>'
    if summary.get("temporary_service_cleanup", {}).get("status") == "closed":
        html += '<p>本轮 20 个临时服务（8794–8813）已正常关闭；服务监督器和结果收集器均正常退出，清理后相关监听及服务进程计数为 0。</p>'
    checks = Path(output) / "render_checks/result_page_checks.json"
    rendering = {"status": "not_recorded", "source": "render_checks/result_page_checks.json",
                 "scope": "representative saved-result rendering only; no end-to-end frontend polling claim", "checks": []}
    if checks.is_file():
        record = read_json(checks)
        job_ids = {item["job_id"] for item in summary["terminal_results"]}
        current_checks = [item for item in record.get("checks", []) if item.get("job_id") in job_ids]
        rendering.update(status="recorded", checked_utc=record.get("checked_utc"),
                         unrelated_or_older_records=len(record.get("checks", [])) - len(current_checks),
                         checks=[{key: item.get(key) for key in (
                             "case_id", "mode", "job_id", "status", "report_stale", "page_errors", "viewports", "method")}
                                 for item in current_checks])
        descriptions = []
        for item in current_checks:
            widths = " / ".join(str(view.get("width", "未记录")) + "px" for view in item.get("viewports", []))
            descriptions.append(escape(names.get(item.get("case_id"), str(item.get("case_id"))))
                                + f' · Graph {"开启" if item.get("mode") == "on" else "关闭"} · {escape(widths)}')
        html += f'<p>代表性结果页已记录 {len(current_checks)} 组：' + ('；'.join(descriptions) or '尚无本轮任务编号匹配的记录')
        html += '。检查只覆盖加载原结果后的页面快照，详见 <a href="render_checks/result_page_checks.json">页面检查记录</a>。</p>'
    else:
        html += '<p>本报告生成时未发现代表性结果页检查记录；表格依据后端保存结果。</p>'
    summary["result_rendering_summary"] = rendering
    html += '<p><a href="comparison_summary.json">完整比较数据</a> · <a href="native_timing_manifest.json">native 测量时间与配置</a> · <a href="correction_inputs.json">复跑输入与来源</a> · <a href="ui_runs.json">提交与恢复记录</a> · <a href="../cuda_graph_dispatch_validation_2026-10-08/report.html">旧报告</a></p></main></html>'
    (Path(output) / "report.html").write_text(html, encoding="utf-8")
    (Path(output) / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"completed": summary["completed_comparisons"], "expected": summary["expected_comparisons"],
            "backend_status_counts": status,
            "metric_error_summaries": [{key: value for key, value in item.items() if key != "rows"}
                                       for item in summary["metric_error_summaries"]],
            "wall_clock_summary": {key: value for key, value in wall.items() if key != "rows"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--render-only", action="store_true", help="refresh HTML from comparison_summary.json without re-reading job results")
    args = parser.parse_args()
    summary = read_json(args.output / "comparison_summary.json") if args.render_only else checked_collect(args.output, args.baseline)
    if summary.get("schema") != "heterollm.cost-logic-native-rerun/v1":
        raise ValueError("render-only requires this rerun's completed comparison summary")
    result = render(args.output, summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
