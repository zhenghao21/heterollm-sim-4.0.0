"""Prepare unchanged frontend inputs and compare source-lowering corrections.

The native measurements are explicitly reused from the preceding matched run.
Independent dispatch probes which failed qualification are never calibration
inputs. This script does not submit jobs or claim a complete Graph timing model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import shutil

from analyze_gguf_preset_validation import collect, load_cases
from render_cuda_graph_validation_report import METRICS

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/gguf_preset_native_validation_2026-10-08"
OUTPUT = ROOT / "docs/cuda_graph_dispatch_validation_2026-10-08"


def prepare(output, baseline):
    output, baseline = output.resolve(), baseline.resolve()
    if output == baseline:
        raise ValueError("correction results must not overwrite the baseline")
    cases = load_cases(baseline)
    names = ["cases.json", "preparation.json", "model_manifest.json", "native_timing_manifest.json",
             "structure_lifecycle_validation.json"]
    for case in cases:
        slug = case["case_id"]
        names.append(f"base/scenario_{slug}_512_128.json")
        for mode in ("off", "on"):
            names.extend((f"scenario_{slug}_graph_{mode}.json", f"native_{slug}_graph_{mode}.json",
                          f"native_graph_diagnostic_{slug}_{mode}.json"))
    # Verify every source/destination before starting any writes. No partial
    # preparation silently overwrites previously submitted simulation inputs.
    for name in names:
        source, target = baseline / name, output / name
        if not source.is_file():
            raise ValueError("missing baseline input: " + name)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError("existing correction input differs: " + name)
    for name in names:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(baseline / name, target)
    provenance = {
        "schema": "heterollm.cuda-dispatch-correction-inputs/v1",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(baseline), "model_count": len(cases),
        "scenario_inputs_changed": False, "native_measurements_reused": True,
        "native_measurement_manifest": str(baseline / "native_timing_manifest.json"),
        "native_values_used_for_calibration": False,
        "device_dispatch_calibration_installed": False,
        "scope": "Rerun unchanged browser inputs after source-lowering fixes; independent dispatch probes remain diagnostics.",
    }
    (output / "correction_inputs.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def _optional_record(output, name):
    path = output / name
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def _number(value, *, milliseconds=False, signed=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "—"
    if milliseconds:
        return f"{value / 1e6:.4f}"
    return f"{value:+.2f}%" if signed else f"{value:.3f}%"


def _input_participation_summary(output, cases):
    """Read submitted model identities; never infer them from display names."""
    rows = []
    for case in cases:
        for mode, data in case["modes"].items():
            slug = case.get("case_id")
            submitted = _optional_record(output, f"ui_{slug}_graph_{mode}_submission.json") if slug else None
            metadata = ((submitted or {}).get("scenario", {}).get("model", {}).get("metadata", {}))
            origin = metadata.get("gguf_preset_origin") or {}
            unchanged = metadata.get("gguf_preset_changes") == {} and "gguf_preset_changes" in metadata
            preset_matches = bool(case.get("preset_id")) and metadata.get("model_preset_id") == case["preset_id"]
            source_named = isinstance(origin.get("filename"), str) and bool(origin["filename"].strip())
            comparison = data["comparison"]
            physical = comparison.get("physical_participation", {}).get("status", "missing")
            rows.append({"case_id": slug, "name": case["name"], "graph_mode": mode,
                         "submission_present": submitted is not None,
                         "expected_preset_id": case.get("preset_id"), "submitted_preset_id": metadata.get("model_preset_id"),
                         "gguf_filename": origin.get("filename"), "gguf_preset_changes": metadata.get("gguf_preset_changes"),
                         "original_gguf_preset": bool(preset_matches and source_named and unchanged),
                         "explicit_empty_changes": unchanged,
                         "physical_participation_status": physical,
                         "physical_participation_passed": comparison["status"] == "compared" and physical == "pass"})
    return {"expected_comparisons": len(rows),
            "original_gguf_presets": sum(row["original_gguf_preset"] for row in rows),
            "explicit_empty_changes": sum(row["explicit_empty_changes"] for row in rows),
            "physical_participation_passed": sum(row["physical_participation_passed"] for row in rows),
            "scope": "actual frontend submission identity and explicit changes; physical participation requires a valid completed pair",
            "rows": rows}


def _e2e_error_summary(cases):
    """Equal-weight absolute percentage errors for valid old/new pairs only."""
    paired, expected = [], sum(len(case["modes"]) for case in cases)
    for case in cases:
        for mode, data in case["modes"].items():
            current = data["comparison"]
            previous = (data.get("history") or {}).get("metrics", {}).get("engine_e2e_ns", {})
            if current["status"] != "compared":
                continue
            old = previous.get("previous_signed_error_percent")
            new = current.get("metrics", {}).get("engine_e2e_ns", {}).get("signed_error_percent")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                   for value in (old, new)):
                continue
            paired.append({"case_id": case.get("case_id"), "name": case["name"], "graph_mode": mode,
                           "previous_absolute_error_percent": abs(old), "current_absolute_error_percent": abs(new),
                           "absolute_error_reduction_percentage_points": abs(old) - abs(new)})
    count = len(paired)
    old_mape = sum(row["previous_absolute_error_percent"] for row in paired) / count if count else None
    new_mape = sum(row["current_absolute_error_percent"] for row in paired) / count if count else None
    return {"metric": "engine_e2e_ns", "weighting": "equal_weight_per_model_graph_mode_pair",
            "formula": "mean(abs(simulation / native_median - 1) * 100)",
            "scope": "only valid completed comparisons with finite matched old and new errors; missing entries excluded",
            "expected_pairs": expected, "matched_pairs": count, "excluded_pairs": expected - count,
            "previous_mape_percent": old_mape, "current_mape_percent": new_mape,
            "absolute_error_reduction_percentage_points": old_mape - new_mape if count else None,
            "improved_pairs": sum(row["absolute_error_reduction_percentage_points"] > 0 for row in paired),
            "worsened_pairs": sum(row["absolute_error_reduction_percentage_points"] < 0 for row in paired),
            "unchanged_pairs": sum(row["absolute_error_reduction_percentage_points"] == 0 for row in paired),
            "pairs": paired}


def report(output, baseline):
    output, baseline = Path(output), Path(baseline)
    provenance = json.loads((output / "correction_inputs.json").read_text(encoding="utf-8"))
    if (provenance["native_measurements_reused"] is not True
            or provenance["scenario_inputs_changed"] is not False
            or provenance.get("native_values_used_for_calibration") is True
            or provenance.get("device_dispatch_calibration_installed") is True):
        raise ValueError("comparison provenance is inconsistent")
    summary = collect(output, baseline)
    summary["correction_provenance"] = provenance
    summary["graph_device_dispatch_fix_complete"] = False
    inputs = _input_participation_summary(output, summary["cases"])
    summary["gguf_input_participation_summary"] = inputs
    e2e_summary = _e2e_error_summary(summary["cases"])
    summary["e2e_error_summary"] = e2e_summary
    timing = _optional_record(output, "native_timing_manifest.json")
    independent = _optional_record(output, "independent_dispatch_summary.json")
    coverage = _optional_record(output, "source_dispatch_coverage.json")
    queue_identification = _optional_record(output, "queue_model_probe/queue_model_identification.json")
    quiet = _optional_record(output, "queue_model_probe_quiet/quiet_comparison.json")
    quiet_preflight = _optional_record(output, "queue_model_probe_quiet/preflight.json")
    render_checks = _optional_record(output, "render_checks/result_page_checks.json")
    http_review = _optional_record(output, "http_observer_disconnect_review.json")
    summary["supporting_reports"] = {
        "native_timing": {"source": "native_timing_manifest.json", "status": "available" if timing else "missing",
                          "started_local": (timing or {}).get("started_local"),
                          "finished_local": (timing or {}).get("finished_local")},
        "independent_dispatch": {"source": "independent_dispatch_summary.json",
                                 "status": (independent or {}).get("status", "missing"),
                                 "prediction_qualified": (independent or {}).get("prediction_qualified")},
        "source_ownership": {"source": "source_dispatch_coverage.json", "status": "available" if coverage else "missing",
                             "summary": (coverage or {}).get("summary"),
                             "scope": (coverage or {}).get("scope")},
        "result_rendering": {"source": "render_checks/result_page_checks.json",
                             "status": "recorded" if render_checks else "not_checked",
                             "scope": "representative completed snapshots; rendering only, not end-to-end frontend polling"},
        "queue_model_identification": {"source": "queue_model_probe/queue_model_identification.json",
                                       "status": "available" if queue_identification else "missing",
                                       "prediction_qualified": (queue_identification or {}).get("prediction_qualified"),
                                       "scope": "31 synthetic cases with training/holdout partition declared before measurement"},
        "low_cpu_fixed_model_replication": {"source": "queue_model_probe_quiet/quiet_comparison.json",
                                             "status": "available" if quiet else "missing",
                                             "prediction_qualified": (quiet or {}).get("prediction_qualified"),
                                             "parameter_refit": (quiet or {}).get("parameter_refit"),
                                             "started_beijing": (quiet or {}).get("started_beijing"),
                                             "finished_beijing": (quiet or {}).get("finished_beijing")},
        "http_observer_disconnect": {"source": "http_observer_disconnect_review.json",
                                     "status": "recorded" if http_review else "not_recorded",
                                     "fix_stage": (http_review or {}).get("fix_stage"),
                                     "matrix_rerun_after_http_fix": (http_review or {}).get("matrix_rerun_after_http_fix")},
    }
    rows, gains = [], []
    statuses = {"pending": "待完成", "missing_simulation": "仿真结果待到齐", "missing_native": "缺少 native 实测",
                "running": "运行中", "queued": "排队中", "failed": "失败", "cancelled": "已取消",
                "configuration_mismatch": "配置核对未通过", "invalid_result": "结果无效"}
    for case in summary["cases"]:
        for mode, data in case["modes"].items():
            row = data["comparison"]
            cells = f'<th scope="row">{escape(case["name"])}</th><td>{"开启" if mode == "on" else "关闭"}</td>'
            if row["status"] != "compared":
                status = statuses.get(row["status"], row["status"])
                detail = row.get("error") or "; ".join(map(str, row.get("validation_errors", ())))
                rows.append('<tr>' + cells + '<td colspan="7" class="unavailable">'
                            + escape(status) + f'<small>{escape(row["status"])}'
                            + (" · " + escape(str(detail)) if detail else "") + '</small></td></tr>')
                continue
            for key, label in METRICS:
                metric = row["metrics"][key]
                prior = (data.get("history") or {}).get("metrics", {}).get(key, {})
                error, old_error = metric.get("signed_error_percent"), prior.get("previous_signed_error_percent")
                improvement = (abs(old_error) - abs(error)) if old_error is not None and error is not None else None
                change = f"{improvement:+.2f} pp" if improvement is not None and math.isfinite(improvement) else "—"
                rows.append('<tr>' + cells + f'<td>{label}</td>'
                    f'<td>{_number(metric["native"].get("median_ns"), milliseconds=True)}</td>'
                    f'<td>{_number(prior.get("previous_simulation_ns"), milliseconds=True)}</td>'
                    f'<td>{_number(metric.get("simulation_ns"), milliseconds=True)}</td>'
                    f'<td>{_number(old_error, signed=True)}</td>'
                    f'<td>{_number(error, signed=True)}</td><td>{change}</td></tr>')
        gain = case.get("graph_speedup")
        gains.append(f'<tr><th scope="row">{escape(case["name"])}</th>'
                     f'<td>{_number((gain or {}).get("native_e2e_reduction_percent"))}</td>'
                     f'<td>{_number((gain or {}).get("simulation_e2e_reduction_percent"))}</td></tr>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CUDA Graph 源码建模修复与误差复核</title><style>
*{box-sizing:border-box}body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f3f5f8;color:#18263a}main{max-width:1380px;margin:auto;padding:28px 20px}h1{font-size:28px;line-height:1.35}h2{font-size:21px;margin-top:28px}p{max-width:1080px}.table{max-width:100%;overflow:auto;background:white;border:1px solid #d9e1eb;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}th,td{text-align:right;padding:9px;border-bottom:1px solid #d9e1eb}th:first-child{text-align:left}thead{background:#eaf0f8}tbody tr:nth-child(even){background:#f8fafc}small{display:block;color:#526176;font-size:12px}.unavailable{white-space:normal;text-align:left;min-width:220px}.status{padding:16px;background:#fff5de;border-left:4px solid #ab7600}a{color:#075da7;overflow-wrap:anywhere}.facts{padding-left:24px}.facts li{margin:8px 0}.table:focus-visible{outline:3px solid #3469b2}@media(max-width:600px){main{padding:20px 12px}h1{font-size:24px}body{font-size:15px}.status{padding:12px}}
</style><main><h1>CUDA Graph：源码建模修复与误差复核</h1>
<p class="status">Graph 设备调度成本问题尚未完整解决。这里验证已确认的源码建模修复；独立探针暴露了普通启动的批次供给行为及性能分析工具的扰动，因此没有把这些未合格数值写入预设。</p>
<p>仿真使用上一轮同一批真实 GGUF 预设、同一硬件、512 个输入 token / 128 个输出 token 负载和 Graph 开/关配置，由前端导入、校验、提交。native 使用上一轮已核实的同配置实测，并非本轮重新计时。旧报告保留，未用模型 native 总耗时拟合补偿项。</p>'''
    if timing:
        html += (f'<p>复用 native 测量时间：{escape(str(timing.get("started_local", "未记录")))} 至 '
                 f'{escape(str(timing.get("finished_local", "未记录")))}（'
                 f'{escape(str(timing.get("environment", {}).get("timezone", "时区未记录")))}）。'
                 '<a href="native_timing_manifest.json">原始实测清单与运行配置</a>。</p>')
    else:
        html += '<p>native 原始测量时间清单尚未提供；不推断测量时间。</p>'
    expected = inputs["expected_comparisons"]
    html += (f'<p class="status">输入与成本核对：<strong>原始 GGUF 预设 {inputs["original_gguf_presets"]}/{expected}</strong> · '
             f'<strong>明确 changes={{}} {inputs["explicit_empty_changes"]}/{expected}</strong> · '
             f'<strong>物理成本参与通过 {inputs["physical_participation_passed"]}/{expected}</strong>。'
             '核对实际前端提交的预设 ID、GGUF 文件来源与修改记录；物理参与通过表示成本确实参与，不能据此证明成本数值准确。</p>')
    input_rows = []
    for row in inputs["rows"]:
        input_rows.append('<tr><th scope="row">' + escape(row["name"]) + '</th><td>'
            + ("开启" if row["graph_mode"] == "on" else "关闭") + '</td><td>'
            + escape(str(row["submitted_preset_id"] or "未提供")) + '</td><td>'
            + escape(str(row["gguf_filename"] or "未提供")) + '</td><td>'
            + ("{}" if row["explicit_empty_changes"] else escape(json.dumps(row["gguf_preset_changes"], ensure_ascii=False)))
            + '</td><td>' + ("通过" if row["physical_participation_passed"] else "未通过 / 待结果核对") + '</td></tr>')
    html += ('<details><summary>逐组查看 GGUF 输入和物理成本核对</summary><div class="table" tabindex="0" role="region" '
             'aria-label="GGUF 输入和物理成本核对"><table><thead><tr><th>模型</th><th>Graph</th><th>提交预设 ID</th>'
             '<th>实际 GGUF 文件</th><th>changes</th><th>物理参与</th></tr></thead><tbody>'
             + ''.join(input_rows) + '</tbody></table></div></details>')
    html += '''<h2>本轮已修复的逻辑</h2><ul class="facts">
<li>依据固定 llama.cpp 源码保留 residual 与归一化的真实边界，去掉不存在的 ADD → RMSNorm 融合。</li>
<li>QKV 和 FFN 按 GGUF 的真实权重矩阵拆分调用；同格式 Q8 也保留 Q/K/V 各自边界，真实合并矩阵不被凭空拆分。</li>
<li>补上注意力输出整理（CONT）的实际读写；单 token 的设备内拷贝与多 token 的拷贝 kernel 分开表达。</li>
<li>启用最后一层输出行选择：Qwen2.5、Qwen3 和 Llama 在最终 FFN 前选择输出行，混合架构在最终归一化后选择。GET_ROWS 读取正确行偏移，索引使用独立缓冲区。</li>
</ul><p>计算与物理 DRAM 成本继续参与。以上改变来自结构和源码逻辑；结构归属检查通过，并不代表执行顺序、融合内部缓存或设备调度成本已完整解决。</p>
<h2>独立测量与来源归属</h2>'''
    links = []
    for name, label in (("README.md", "独立测量结论与限制"),
                        ("independent_dispatch_summary.json", "独立测量分类统计"),
                        ("queue_candidate_holdout.json", "早期队列候选探索（事后分析）"),
                        ("queue_model_probe/experiment_plan.json", "31 组队列实验的预先划分计划"),
                        ("queue_model_probe/queue_model_identification.json", "31 组队列模型识别与留出检验"),
                        ("queue_model_probe_quiet/quiet_comparison.json", "低 CPU 负载复测与两轮比较"),
                        ("queue_model_probe_quiet/queue_model_identification.json", "固定原模型的复测留出结果"),
                        ("source_dispatch_coverage.html", "源码调用归属检查报告"),
                        ("source_dispatch_coverage.json", "源码调用归属明细")):
        if (output / name).is_file():
            links.append(f'<a href="{name}">{label}</a>')
    html += '<p>' + (' · '.join(links) if links else '独立测量及来源归属附件尚未提供。') + '</p>'
    if independent:
        html += ('<p>独立探针未取得可部署的通用设备调度成本：普通提交仍受驱动供给批次影响；'
                 'CUPTI 观测改变了 Graph 执行；PDL 还涉及阶段重叠。测量保留为诊断数据，未写入预测预设。</p>')
    if queue_identification:
        html += ('<p>队列实验分为两阶段：早期候选是对已看过数据的事后探索；后续 31 组先声明训练与留出划分，'
                 '再用训练组识别模型、用留出组检验。后者仍标记为未取得预测资格：测量期间有并发仿真 CPU 负载，'
                 + ('随后已按固定原模型完成低 CPU 负载复测。' if quiet else '还需固定参数的低 CPU 负载复测。')
                 + '不能把这些实验称作已部署的通用 CUDA Graph 成本。</p>')
    if quiet:
        start = datetime.fromisoformat(quiet["started_beijing"]).strftime("%Y-%m-%d %H:%M:%S")
        end = datetime.fromisoformat(quiet["finished_beijing"]).strftime("%Y-%m-%d %H:%M:%S")
        cpu_samples = [row["cpu_percent"] for row in (quiet_preflight or {}).get("cpu_samples", [])]
        cpu_range = f"{min(cpu_samples):.1f}%–{max(cpu_samples):.1f}%" if cpu_samples else "未记录"
        gpu_fields = (quiet_preflight or {}).get("gpu_status_csv", "").split(",")
        gpu_utilization = gpu_fields[1].strip().replace(" ", "") if len(gpu_fields) > 1 else "未记录"
        holdouts = {row["case_id"]: row.get("low_cpu", {}).get("span_error_percent")
                    for row in quiet.get("holdout_comparison", [])}
        observer_changes = " / ".join(_number(row.get("difference_percent"), signed=True)
                                      for row in quiet.get("observer_controls", [])) or "未记录"
        fixed = quiet.get("parameter_refit") is False and quiet.get("fitted_model_equal") is True
        html += (f'<p><strong>低 CPU 负载复测：北京时间 {start} 至 {end}</strong>，'
                 f'{quiet["case_count"]} 个配置 × {quiet["repetitions"]} 次重复，'
                 + ('固定原模型，没有重新训练或调整参数。' if fixed else '参数一致性未确认，不能宣称固定模型复测。')
                 + f'CPU 预检 {cpu_range}，但 GPU 仍有 {escape(gpu_utilization)} 活动且归属未完全确认；'
                 '这不代表 GPU 空闲或完全无竞争。目录沿用本轮起始日期 10 月 8 日。</p>')
        html += (f'<p>固定模型的 255 节点留出误差为 {_number(holdouts.get("nodes_holdout_255"), signed=True)}，'
                 f'交替 / 打乱 body 留出误差为 {_number(holdouts.get("mixed_alternating_holdout"), signed=True)} / '
                 f'{_number(holdouts.get("mixed_shuffled_holdout"), signed=True)}；'
                 f'CPU 时间戳观测开关的 event 中位数差异仍为 {observer_changes}。'
                 '这些差异不能全部归因于观测工具，也不能据此证明观测无扰动。复测后仍未取得通用预测资格，'
                 '没有部署、修改预设或依据留出误差重新拟合。</p>')
    if coverage:
        stats = coverage.get("summary", {})
        html += (f'<p>来源归属范围：{stats.get("matched_models", 0)}/{stats.get("model_count", 0)} 个模型、'
                 f'{stats.get("matched_regions", 0)}/{stats.get("audited_regions", 0)} 个已审计区域匹配。'
                 '这是首层及尾层在 prefill/decode 下的静态归属核对，未执行 GPU；'
                 'Qwen3.8-27B 的混合架构归属适配仍未完成。完整范围以附件为准。</p>')
    html += '<h2>修复前后预测误差</h2>'
    if e2e_summary["matched_pairs"]:
        html += (f'<p class="status"><strong>E2E 平均绝对百分比误差：修复前 {_number(e2e_summary["previous_mape_percent"])}'
                 f' → 修复后 {_number(e2e_summary["current_mape_percent"])}。</strong>'
                 f'绝对误差改善 {e2e_summary["improved_pairs"]} 组，变差 {e2e_summary["worsened_pairs"]} 组，'
                 f'不变 {e2e_summary["unchanged_pairs"]} 组；纳入 {e2e_summary["matched_pairs"]}/{e2e_summary["expected_pairs"]} 组。</p>')
    else:
        html += '<p class="status">E2E 汇总暂无完整的旧新有效对照，平均误差不计算。</p>'
    html += ('<p>E2E 汇总按“模型 × Graph 开关”每组等权，先取每组 |仿真 / native 中位数 − 1| × 100%，再取算术平均。'
             '只纳入新旧均有效且误差为有限数的配对，缺项不补零；改善、变差与不变按未四舍五入的绝对误差判定。'
             '此平均值不按模型规模或耗时加权，也不代表所有模型都改善。</p>')
    html += f'<p>有效比较 {summary["completed_comparisons"]}/{summary["expected_comparisons"]}。单位：毫秒；有符号误差 =（仿真/native − 1）×100%。缺失或失败结果不由历史值补齐。</p>'
    html += '<p>负误差表示预测偏快，正误差表示预测偏慢。“绝对误差减少” = |原误差| − |现误差|，正值表示改善，负值表示变差；pp 表示百分点。缺少旧结果时显示“—”，不填入假定值。窄屏可横向滚动表格。</p>'
    html += '<div class="table" tabindex="0" role="region" aria-label="修复前后预测误差"><table><thead><tr><th>模型</th><th>Graph</th><th>指标</th><th>native</th><th>修复前预测</th><th>修复后预测</th><th>原误差</th><th>现误差</th><th>绝对误差减少</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    html += '<h2>Graph 总耗时降低比例</h2><p>降低比例 =（关闭耗时 − 开启耗时）/ 关闭耗时。仿真两组均核对通过后才计算；“—”表示配对尚不完整。</p><div class="table" tabindex="0" role="region" aria-label="Graph 总耗时降低比例"><table><thead><tr><th>模型</th><th>native</th><th>当前仿真</th></tr></thead><tbody>' + ''.join(gains) + '</tbody></table></div>'
    if http_review:
        html += ('<h2>运行监控发现与后续 HTTP 修复</h2>'
                 f'<p>本轮后台仿真 {http_review["completed_simulations"]} 组完成、{http_review["failed_simulations"]} 组失败。'
                 f'收尾检查发现，页面提交后关闭曾触发 {http_review["observed_polling_disconnects"]} 次轮询通信异常（端口 '
                 + ' / '.join(map(str, http_review["service_ports"]))
                 + '，ConnectionAbortedError / WinError 10053）。旧处理将向已断开连接写响应的异常误记为内部错误，'
                 '又尝试向同一连接发送 HTTP 500；这些通信异常没有使后台仿真失败。</p>'
                 '<p>修复仅处理响应头和响应体写出时的连接中断，关闭该 HTTP 连接；后台任务及仿真数值不变。'
                 '其他 I/O 错误和后端自身的连接异常仍正常报错。'
                 '<strong>此修复发生在 20 组复跑结束后，并通过针对性测试；没有在 HTTP 修复后再次重跑这 20 组。</strong></p>'
                 '<p><a href="http_observer_disconnect_review.json">通信异常与修复说明</a> · '
                 '<a href="../../tests/test_web_disconnected_response.py">断连回归测试</a> · '
                 '<a href="../../src/heterollm_sim/web.py">HTTP 响应处理代码</a></p>')
    if render_checks and render_checks.get("checks"):
        render_links = ['<a href="render_checks/result_page_checks.json">结果页检查记录</a>']
        for check in render_checks["checks"]:
            filename = Path(check["result_path"]).name.removesuffix("_result.json")
            for viewport, label in (("desktop", "桌面"), ("narrow", "窄屏")):
                relative = f"render_checks/{filename}_{viewport}.png"
                if (output / relative).is_file():
                    render_links.append(f'<a href="{escape(relative)}">Graph {escape(check["mode"])} · {label}截图</a>')
        html += ('<h2>代表性结果页检查</h2><p>对已完成结果，通过原服务的只读接口取得真实快照，并调用现有完成处理函数显示。'
                 '检查桌面和窄屏的指标、请求明细与页面异常；没有重提仿真任务。此项仅验证结果渲染，'
                 '不代表恢复后端到端轮询展示已验证。</p><p>' + ' · '.join(render_links) + '</p>')
    html += '<p><a href="comparison_summary.json">完整对照数据</a> · <a href="correction_inputs.json">输入与实测复用说明</a> · <a href="../gguf_preset_native_validation_2026-10-08/report.html">修复前报告</a></p></main></html>'
    (output / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.html").write_text(html, encoding="utf-8")
    return {"completed": summary["completed_comparisons"], "expected": summary["expected_comparisons"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "report"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    print(json.dumps((prepare if args.action == "prepare" else report)(args.output, args.baseline), ensure_ascii=False))


if __name__ == "__main__":
    main()
