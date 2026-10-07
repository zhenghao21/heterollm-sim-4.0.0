"""Render the saved frontend/native validation evidence as a standalone HTML file."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "docs/frontend_native_validation_2026-10-07"
HARDWARE_NAMES = {
    "local-native-rtx5080-9950x3d-gddr7-ddr5": "本机 RTX 5080 / DDR5",
    "nvidia-b200-1gpu-2hbf-2hbm": "B200 / 2 HBF / 2 HBM",
    "nvidia-b200-1gpu-3hbf-2hbm": "B200 / 3 HBF / 2 HBM",
}
STATUS_NAMES = {
    "completed": "仿真完成", "compared": "已完成配对比较",
    "capacity_rejected": "容量不足，已拒绝", "pending": "待验证",
    "running": "运行中", "queued": "排队中", "cancelled": "已取消",
    "program_error": "程序错误", "configuration_mismatch": "配对配置不一致",
    "input_contract_failed": "输入配置检查失败", "validated_not_run": "已校验，未运行",
    "validation_rejected": "校验拒绝", "validation_response_invalid": "校验响应异常",
    "validation_capture_incomplete": "校验记录缺少提交输入",
    "failed": "失败", "validation_failed": "结果检查失败",
}
METRICS = (
    ("ttft", "首 token 延迟 TTFT", "ms", 1e-6),
    ("tpot", "后续 token 平均延迟 TPOT", "ms", 1e-6),
    ("e2e", "总时延 E2E", "ms", 1e-6),
    ("decode_tokens_per_second", "解码吞吐量", "token/s", 1),
    ("output_tokens_per_engine_second", "全程输出吞吐量", "token/s", 1),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def text(value):
    return escape(str(value), quote=True)


def status_name(status):
    return STATUS_NAMES.get(status, str(status or "未记录"))


def number(value, scale=1, signed=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "—"
    value *= scale
    if value == 0:
        return "0"
    if abs(value) < 0.0001:
        return format(value, "+.4g" if signed else ".4g")
    return format(value, "+.4f" if signed else ".4f").rstrip("0").rstrip(".")


def metric_cell(value, *, scale=1, signed=False):
    raw = "" if value is None else f' data-value="{text(value)}" data-scale="{scale}"'
    return f"<td class=number{raw}>{number(value, scale, signed)}</td>"


def link(path: Path, directory: Path, label):
    if not path.is_file():
        return text(label)
    return f'<a href="{text(path.relative_to(directory).as_posix())}">{text(label)}</a>'


def localized_time(value):
    if not value:
        return "未记录"
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return moment.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S（北京时间）")
    except ValueError:
        return str(value)


def model_name(model_id):
    family, _, size = model_id.partition("-")
    names = {"llama3_1": "Llama 3.1", "llama3_2": "Llama 3.2", "llama3_3": "Llama 3.3",
             "qwen2_5": "Qwen2.5", "qwen3": "Qwen3"}
    return names.get(family, family) + " " + size.replace("_", ".").upper()


def verify_live_execution(evidence):
    counts = evidence.get("physical_execution") or {}
    live, total = counts.get("physical_live_batch_count"), counts.get("batch_count")
    if type(live) is not int or type(total) is not int or total <= 0 or live != total:
        raise ValueError("completed comparison lacks persistent physical execution for every cohort")
    return f"持久物理执行 {live} / {total} 批次"


def index_attempts(data):
    attempts = {}
    for row in data.get("attempts", []):
        identity = row.get("attempt_id")
        if not identity:
            raise ValueError("comparison.json lacks unique attempt_id; regenerate it with the analyzer")
        if identity in attempts:
            raise ValueError("comparison.json contains duplicate attempt_id: " + identity)
        attempts[identity] = row
    return attempts


def run_wall_seconds(attempt):
    if attempt.get("job_status") not in {"completed", "failed", "cancelled"}:
        return None
    try:
        start = datetime.fromisoformat(attempt["started_at"].replace("Z", "+00:00"))
        finish = datetime.fromisoformat(attempt["finished_at"].replace("Z", "+00:00"))
        if start.tzinfo is None or finish.tzinfo is None:
            return None
        elapsed = (finish - start).total_seconds()
        return elapsed if elapsed >= 0 else None
    except (KeyError, TypeError, AttributeError, ValueError):
        return None


def matrix_section(data, directory):
    rows = data.get("matrix", [])
    hardware = list(dict.fromkeys(row["hardware_id"] for row in rows))
    models = list(dict.fromkeys(row["model_id"] for row in rows))
    by_pair = {(row["hardware_id"], row["model_id"]): row for row in rows}
    if len(by_pair) != len(rows):
        raise ValueError("comparison.json contains duplicate matrix pairs")
    attempts = index_attempts(data)
    expected = data.get("expected_scope", {}).get("matrix_combinations", 63)
    counts = Counter(row.get("status", "pending") for row in rows)
    coverage = "、".join(f"{status_name(key)} {value}" for key, value in counts.items())
    heading = "矩阵记录已齐" if len(rows) == expected and not any(
        counts.get(key) for key in ("pending", "running", "queued", "validated_not_run")
    ) else "矩阵验证进行中"
    out = [f'<section><h2>前端仿真矩阵 · {heading}</h2><p>{text(coverage)}。目标 {expected} 个组合。</p>',
           '<p class=note>负载为单请求、512 输入 / 128 输出，llama 调度，batch / ubatch 512、context 640。'
           '“仿真完成”表示运行完成；容量拒绝、待验证和失败分别保留，不作为通过。每格下方的 native 状态独立于仿真状态。</p>'
           '<p class=note>本次仿真运行耗时 = finished_at − started_at，不含前端校验；包含并发竞争、暂停及内存压力影响，'
           '不能用作优化速度基准。未终态、未创建运行或时间缺失时显示不可用。</p>',
           '<div class=table-wrap><table class=matrix><thead><tr><th rowspan=2>模型预设</th>']
    out.extend(f'<th colspan=2>{text(HARDWARE_NAMES.get(hw, hw))}</th>' for hw in hardware)
    out.append('</tr><tr>')
    out.extend('<th>状态</th><th>本次仿真运行耗时(s)</th>' for hw in hardware)
    out.append('</tr></thead><tbody>')
    for model in models:
        out.append(f'<tr><th scope=row>{text(model_name(model))}</th>')
        for hw in hardware:
            row = by_pair.get((hw, model), {"status": "pending"})
            status = row.get("status", "pending")
            attempt = attempts.get(row.get("latest_attempt"), {})
            evidence = attempt.get("result_file") or attempt.get("validation_file") or attempt.get("submission_file")
            label = link(directory / evidence, directory, status_name(status)) if evidence else text(status_name(status))
            live_label = ('<small>' + text(verify_live_execution(attempt)) + '</small>') if status == "completed" else ""
            out.append(f'<td data-hardware="{text(hw)}" data-model="{text(model)}" data-status="{text(status)}">'
                       f'<span class="status {text(status)}">{label}</span>{live_label}<small>native 未测</small></td>')
            elapsed = run_wall_seconds(attempt)
            out.append(metric_cell(elapsed) if elapsed is not None else '<td class=note>不可用</td>')
        out.append('</tr>')
    out.append('</tbody></table></div><p class=note>下方两组 native 比较使用相同 GGUF 文件单独配对；'
               '目录中的默认模型预设没有逐一完成 native 实测。B200 / HBF 两套硬件没有对应实机数据。</p></section>')
    return "".join(out)


def native_statistics(native, metric):
    values = []
    for sample in native.get("samples", []):
        if sample.get("status") != "completed":
            continue
        key = "engine_" + metric + "_ns"
        value = sample.get(key)
        if metric == "decode_tokens_per_second":
            tpot = sample.get("engine_tpot_ns")
            value = 1e9 / tpot if isinstance(tpot, (int, float)) and tpot > 0 else None
        elif metric == "output_tokens_per_engine_second":
            e2e = sample.get("engine_e2e_ns")
            tokens = sample.get("visible_output_tokens")
            value = tokens * 1e9 / e2e if isinstance(e2e, (int, float)) and e2e > 0 and isinstance(tokens, int) else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(value)
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "sample_count": len(values)} if values else {}


def verify_metric(metric):
    sim = metric["simulation"]
    median = metric["native"]["median"]
    if not math.isclose(metric["signed_error"], sim - median, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("comparison.json signed error disagrees with its native and simulation values")
    if median and not math.isclose(metric["signed_error_percent"], (sim - median) / median * 100,
                                    rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("comparison.json percent error disagrees with its native and simulation values")


def native_section(data, directory):
    out = ['<section><h2>本机 native 配对比较</h2><p>统一使用引擎口径：从请求开始处理到首个 / 最后一个可见 token，'
           '不包含客户端 HTTP / SSE 传输时间。TPOT =（最后 token 时间 − 首 token 时间）/ 127。'
           '误差 = 仿真 − native 中位数；百分比误差以 native 中位数为分母，保留正负号，不设置精度合格阈值。</p>'
           '<p class=note>主比较要求模拟 GPU 的 graph_enabled 显式为 false，且 native 子进程记录 '
           'GGML_CUDA_DISABLE_GRAPHS=1；没有该环境证据的历史测量另列，不称所有设置已对齐。</p>']
    for pair in data.get("native_comparisons", []):
        case = pair["case_id"]
        native_path = directory / ("native_" + case + ".json")
        native = read_json(native_path) if native_path.is_file() else {}
        name = "Qwen3-0.6B F16" if case == "qwen3_0_6b_f16" else "27B 混合量化" if case == "qwen3_8_27b_mixed" else case
        filename = str(native.get("identity", {}).get("model_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
        out.append(f'<h3>{text(name)} <span class=status>{text(status_name(pair.get("status")))}</span></h3>')
        if pair.get("status") == "compared":
            out.append('<p class=note>' + text(verify_live_execution(pair)) + '</p>')
        if filename:
            out.append(f'<p class=note>相同文件配对：{text(filename)}。'
                       f'native：{len(native.get("warmups", []))} 次预热、{len(native.get("samples", []))} 次正式测量。'
                       '表中范围是这几次测量的最小值—最大值，不是置信区间。</p>')
        config = native.get("configuration", {})
        if config:
            out.append('<p class=note>配对设置：context ' + text(config.get("context")) + '；batch / ubatch '
                       + text(config.get("batch")) + ' / ' + text(config.get("ubatch")) + '；KV '
                       + text(config.get("cache_type_k")) + ' / ' + text(config.get("cache_type_v"))
                       + '；Flash Attention ' + text(config.get("flash_attn"))
                       + '；threads ' + text(config.get("threads")) + '。此设置与上方默认预设矩阵分别核验。</p>')
        errors = pair.get("validation_errors") or pair.get("missing_files") or []
        if errors:
            out.append('<p class=warning>尚不可完成同口径比较：' + text("；".join(errors)) + '</p>')
        out.append('<div class=table-wrap><table><thead><tr><th>指标</th><th>单位</th><th>仿真</th>'
                   '<th>native 中位数</th><th>native 最小值</th><th>native 最大值</th><th>有符号误差</th>'
                   '<th>有符号误差 %</th><th>实测次数</th></tr></thead><tbody>')
        for key, label, unit, scale in METRICS:
            metric = pair.get("metrics", {}).get(key) if pair.get("status") == "compared" else None
            if metric:
                verify_metric(metric)
            stats = metric["native"] if metric else native_statistics(native, key)
            out.append(f'<tr data-case="{text(case)}" data-metric="{key}"><th scope=row>{label}</th><td>{unit}</td>')
            out.append(metric_cell(metric.get("simulation") if metric else None, scale=scale))
            out.extend(metric_cell(stats.get(field), scale=scale) for field in ("median", "min", "max"))
            out.append(metric_cell(metric.get("signed_error") if metric else None, scale=scale, signed=True))
            out.append(metric_cell(metric.get("signed_error_percent") if metric else None, signed=True))
            out.append(metric_cell(stats.get("sample_count")))
            out.append('</tr>')
        out.append('</tbody></table></div><p class=note>' + link(native_path, directory, "native 原始记录") + ' · '
                   + link(directory / ("ui_pair_" + case + "_result.json"), directory, "仿真结果记录") + '</p>')
    out.append('</section>')
    return "".join(out)


def historical_native_graph_section(directory):
    paths = sorted(directory.glob("native_default_config_*.json"))
    if not paths:
        return ""
    out = ['<section><h2>历史默认环境测量 · CUDA Graph 状态未核实</h2>'
           '<p class=warning>以下保留最初测量事实。当时二进制支持 CUDA Graph，但没有记录禁用环境变量或实际 capture 状态；'
           '模拟配置关闭 Graph，因此这些差距不能作为严格相同设置下的主比较。不要从编译开关推断每个请求都使用了 Graph。</p>'
           '<div class=table-wrap><table><thead><tr><th>配对模型</th><th>指标</th><th>仿真</th>'
           '<th>原 native 中位数</th><th>原 native 范围</th><th>有符号差距（%）</th></tr></thead><tbody>']
    for path in paths:
        case = path.stem.removeprefix("native_default_config_")
        simulation_path = directory / ("ui_pair_" + case + "_result.json")
        if not simulation_path.is_file():
            continue
        job = read_json(simulation_path)
        requests = job.get("report", {}).get("requests", {})
        requests = list(requests.values()) if isinstance(requests, dict) else requests
        if job.get("status") != "completed" or len(requests) != 1:
            continue
        request = requests[0]
        native = read_json(path)
        for key, label, unit, scale in METRICS:
            stats = native_statistics(native, key)
            simulated = request.get("engine_" + key + "_ns")
            if key == "decode_tokens_per_second":
                tpot = request.get("engine_tpot_ns")
                simulated = 1e9 / tpot if isinstance(tpot, (float, int)) and tpot > 0 else None
            elif key == "output_tokens_per_engine_second":
                e2e = request.get("engine_e2e_ns")
                tokens = request.get("visible_output_tokens")
                simulated = tokens * 1e9 / e2e if isinstance(e2e, (float, int)) and e2e > 0 and isinstance(tokens, int) else None
            median = stats.get("median")
            relative = (simulated - median) / median * 100 if isinstance(simulated, (float, int)) and median else None
            out.append('<tr><th>' + link(path, directory, case) + '</th><td>' + text(label + '（' + unit + '）') + '</td>'
                       + metric_cell(simulated, scale=scale) + metric_cell(median, scale=scale)
                       + '<td>' + number(stats.get("min"), scale) + ' — ' + number(stats.get("max"), scale) + '</td>'
                       + metric_cell(relative, signed=True) + '</tr>')
    out.append('</tbody></table></div></section>')
    return "".join(out)


def recording_section(data):
    recordings = {row["prefix"]: row for row in data.get("attempts", []) + data.get("diagnostic_attempts", [])
                  if row.get("result_recording", {}).get("schema") == "frontend-job-compact/v1"}
    available = sum(row.get("raw_archive_available_locally") is True for row in recordings.values())
    return ('<section><h2>结果保存方式</h2><p>本次汇总有 ' + str(len(recordings))
            + ' 份终态记录带有精简标记，其中 ' + str(available)
            + ' 份原始归档当前在本机可见。完整结果先以 gzip 保存到仓库外，并全量回读核对；'
            '精简记录的 recording 字段列出归档路径与省略字段。</p>'
            '<p>精简仅省略 report.visualization 和 report.batch_history。请求时延、物理成本汇总、'
            '批次索引、运行时放置和指标口径全部保留。这是保存内容精简，不减少仿真计算，也不改变预测值。</p>'
            '<p class=note>报告层的批次历史和可视化重复序列化同一份 cost.metadata，造成文件体积膨胀。'
            '本轮保留服务端行为，仅调整验证记录的保存方式；未完成归档的文件不视为已精简保存。</p></section>')


def paired_cost_scope_section(directory):
    path = directory / "paired_cost_audit.json"
    if not path.is_file():
        return ""
    audit = read_json(path)
    out = ['<section><h2>配对成本覆盖范围</h2><p>以下限制与数值误差一起保留；运行完成不表示所有原生指令时序都已被复现。</p><ul>']
    if audit.get("native_latency_used_to_fit_parameters") is False:
        out.append('<li>没有使用本次 native 时延拟合参数。硬件容量及分析效率假设保持原声明，修正的是算子、调用、存储和任务归属。</li>')
    out.extend([
        '<li>GPU 原生 kernel 路径依据源码规则判断，尚无实际 dispatch 跟踪；寄存器占用及可达到的吞吐率仍是分析假设。</li>',
        '<li>两组配对声明运行时 sin/cos，已计算函数调用与 Q/K/位置读写；频率 powf、部分角度运算、YaRN 插值及编译后指令展开仍有未覆盖部分。</li>',
        '<li>主机输出与采样已参与成本；完成中断、部分堆排序/分支/缓存行为及关闭阶段的开销仍不完整。</li>',
        '<li>DRAM 物理成本替代对应端点占位成本，与原设备流包络按并发完成关系共同决定时间。设备流包络可能主导耗时；物理配置齐全本身不保证预测准确。</li>',
        '<li>16 / 2 短样本只验证执行合同；正式误差使用前端提交的 512 / 128 结果与 5 次 native 测量。</li>',
        '</ul>',
    ])
    if audit.get("native_placement_evidence", {}).get("diagnostics"):
        out.append('<p>同配置独立详细日志检查确认：0.6B 的 28 主层及输出层、27B 的 64 主层及输出层均分配到 CUDA0；'
                   '27B 的额外 MTP 层权重明确跳过。两组只见输入 token embedding 从 CUDA_Host 改为 CPU，'
                   '未见其他模型权重的兼容性 CPU 回退。正式测量的默认日志级别过滤了这些信息。</p>'
                   '<p class=note>详细日志是独立路径诊断，不是正式 5 次测量逐进程记录，也不追加到时延样本。'
                   '27B 路径诊断在第 47 个 decode 后主动结束；它只用于检查分配路径。</p>')
    out.append('<p class=note>源码版本：' + text(audit.get("native_source_revision", "未记录"))
               + ' · ' + link(path, directory, "完整成本审计记录") + '</p></section>')
    return "".join(out)


def optimization_matrix_section(directory):
    live_path = directory / "optimization_equivalence_live_physical.json"
    path = live_path if live_path.is_file() else directory / "optimization_matrix_equivalence.json"
    if not path.is_file():
        return ""
    data = read_json(path)
    rows = data.get("cases", [])
    completed = data.get("status") == "completed" and len(rows) == data.get("expected_cases")
    progress = ('已记录全部组合。' if completed else
                '已停止并保留为历史记录，尚未覆盖全部组合。' if data.get("status") == "stopped_superseded" else
                '仍在进行，尚未覆盖全部组合。')
    out = ['<section><h2>21 模型 × 3 硬件的性能优化等价诊断</h2><p>'
           + progress
           + '这组独立诊断采用 1 输入 / 2 输出 token，两侧使用相同的语义实现，'
           '只同时恢复旧 bank 复制、重复事务快照和模型属性读取路径，再与当前实现比较。'
           '它不替代上方正式 512 / 128 前端矩阵。</p>'
           '<div class=table-wrap><table><thead><tr><th>硬件</th><th>报告完全一致</th><th>双方一致容量拒绝</th>'
           '<th>差异 / 异常</th><th>尚未比较</th><th>比较数值字段</th><th>最大相对差（比例）</th></tr></thead><tbody>']
    if data.get("semantic_stage") == "persistent_live_physical":
        warning = ('<p class=warning>本次对照已使用跨批次持久物理状态，但执行进程在后续工作区容量、'
                   'HBF / 端点能耗修复之前加载代码。零差异只证明上述三项性能实现的切换；'
                   '不覆盖后续容量与能耗纠错，也不证明修复前成本完整。</p>')
        out[0] = out[0].replace('<div class=table-wrap>', warning + '<div class=table-wrap>', 1)
    elif data.get("semantic_stage") != "physical_io_fixed":
        warning = ('<p class=warning>这次诊断属于 direct / KV 物理读写修复前的历史阶段；'
                   '最终实现仍需重新执行完整等价诊断，不能把此处的零差异当作最新代码结论。</p>')
        out[0] = out[0].replace('<div class=table-wrap>', warning + '<div class=table-wrap>', 1)
    for hardware_id in data.get("hardware_templates", {}):
        cases = [row for row in rows if row.get("hardware_id") == hardware_id]
        comparable = [row for row in cases if "numeric_fields_compared" in row]
        count = Counter(row.get("status") for row in cases)
        bad = len(cases) - count["identical"] - count["consistent_capacity_rejection"]
        max_relative = max((row.get("max_relative_difference", 0) for row in comparable), default=None)
        if any(row.get("undefined_relative_difference_count", 0) for row in comparable):
            max_relative = None
        out.append('<tr><th>' + text(HARDWARE_NAMES.get(hardware_id, hardware_id)) + '</th>'
                   + metric_cell(count["identical"]) + metric_cell(count["consistent_capacity_rejection"])
                   + metric_cell(bad) + metric_cell(len(data.get("models", [])) - len(cases))
                   + metric_cell(sum(row.get("numeric_fields_compared", 0) for row in comparable))
                   + metric_cell(max_relative) + '</tr>')
    comparable = [row for row in rows if "numeric_fields_compared" in row]
    if comparable:
        counts = {(side, key): sum(row.get(side, {}).get("performance_counters", {}).get(key, 0) for row in comparable)
                  for side in ("before", "after") for key in ("snapshot_calls", "execution_graph_guard_calls")}
        out.append('</tbody></table></div><p class=note>已比较运行的事务快照调用总次数：'
                   + number(counts["before", "snapshot_calls"]) + ' → ' + number(counts["after", "snapshot_calls"])
                   + '；执行图完整性校验次数：' + number(counts["before", "execution_graph_guard_calls"])
                   + ' → ' + number(counts["after", "execution_graph_guard_calls"]) + '。这些计数确认性能路径实际发生切换。</p>')
    else:
        out.append('</tbody></table></div>')
    out.append('<p class=note>逐项比较完整返回报告的数值、字符串、布尔值及结构；数值相等的整数/浮点数视为等价。'
               '报告可视化本身有事件数量上限。容量拒绝没有预测结果，不计作零差异样本。'
               '三项优化在此同时切换，分别隔离的证据见前表；短负载结论不外推为任意长负载保证。'
               '进程运行耗时与诊断计数单独记录，不作为预测值或端到端加速证明。'
               + link(path, directory, '逐组合结果及精确替换定义') + '</p></section>')
    return "".join(out)


def evidence_section(directory):
    out = ['<section><h2>功能与语义修复</h2><ul>'
           '<li>修复前端编辑清除 Blackwell kernel 选择与显式 F32 隐藏状态配置的问题；硬件切换、撤销/重做和失败回滚同步恢复 kernel 选择。</li>'
           '<li>QKV 和输出投影使用显式注意力头维度；补入 Qwen2.5 的 Q/K/V 偏置、Qwen3 的 Q/K 归一化、注意力缩放及 MoE 专家加权合并。</li>'
           '<li>Embedding 按实际索引行读取；补齐 FP16 运算 / FP32 输出能力，混合量化权重按实际分片选择执行路径，不支持的 MMVF kernel 明确拒绝。</li>'
           '<li>分段投影保留共享输入的物理标识，为不同输出分片使用独立标识；MoE 加权任务继承既有 CPU/GPU 放置，避免意外跨设备执行。</li>'
           '<li>修复 direct 同一显存所有者丢失本地激活读写、KV 当前行与缓存身份的问题；跨批次使用同一物理状态，并由实际执行事件重新汇总成本。旧的每批次冷启动预估不能作为最终结果。</li>'
           '<li>物理端点继承当前场景的显式能耗系数；HBF 能耗按介质字节计费一次，避免漏计或在主机 / 内部传输重复计费。该修复也影响 DRAM 端点；旧运行总能耗须重新验证。</li>'
           '<li>容量不足显示专用错误码及所需分配量；补全主机编排参数，恢复硬件载入按钮状态。</li>'
           '</ul><p class=note>修正遗漏或错误语义会改变原预测值。这是修复前后差异，不是优化引入的近似误差，也不是相对 native 的预测误差。</p>']
    for filename, title, note in (
        ("attention_projection_geometry_numerical_change.json", "显式头维度修复的定量对照", "以下短负载只隔离该项几何修复，基线为修复前实现；最终矩阵仍使用 512 / 128。"),
        ("qwen25_bias_numerical_change.json", "Qwen2.5 偏置修复的定量对照", "以下为单层缩小回归场景，只隔离缺少三次偏置加法的影响；控制面与融合策略相同，不代表完整模型或 native 实测误差。"),
    ):
        geometry_path = directory / filename
        if not geometry_path.is_file():
            continue
        out.append('<h3>' + title + '</h3><p class=note>' + note + '</p>'
                   '<div class=table-wrap><table><thead><tr><th>场景</th><th>输入 / 输出</th><th>指标</th><th>修复前（ms）</th><th>修复后（ms）</th><th>变化（%）</th></tr></thead><tbody>')
        for case in read_json(geometry_path).get("cases", []):
            for key, label in (("engine_ttft_ns", "TTFT"), ("engine_tpot_ns", "TPOT"), ("engine_e2e_ns", "E2E")):
                metric = case.get("metrics", {}).get(key, {})
                out.append('<tr><th>' + text(case.get("case", "未记录")) + '</th><td>'
                           + text(str(case.get("prompt_tokens", "—")) + " / " + str(case.get("output_tokens", "—")))
                           + '</td><td>' + label + '</td>' + metric_cell(metric.get("before"), scale=1e-6)
                           + metric_cell(metric.get("after"), scale=1e-6)
                           + metric_cell(metric.get("relative_change_percent"), signed=True) + '</tr>')
        out.append('</tbody></table></div><p class=note>' + link(geometry_path, directory, "修复对照记录") + '</p>')
    undo_path = directory / "ui_hardware_undo_verification.json"
    if undo_path.is_file():
        out.append('<p class=note>' + link(undo_path, directory, "真实页面硬件切换、撤销与重做记录") + '</p>')
    energy_path = directory / "hbf_energy_correction.json"
    if energy_path.is_file():
        energy = read_json(energy_path)
        out.append('<h3>HBF 能耗成本修复</h3><p class=note>以下是同一实际物理小事务的成本纠错，不是性能优化近似误差。'
                   '原值为零时不计算百分比变化。</p><div class=table-wrap><table><thead><tr><th>内存 owner</th>'
                   '<th>介质字节</th><th>系数（pJ/byte）</th><th>修复前（pJ）</th><th>修复后（pJ）</th>'
                   '<th>差值（pJ）</th><th>完成时刻差（ns）</th></tr></thead><tbody>')
        for row in energy.get("owners", []):
            out.append('<tr><th>' + text(row.get("owner")) + '</th>'
                       + metric_cell(row.get("physical_bytes")) + metric_cell(row.get("coefficient_pj_per_byte"))
                       + metric_cell(row.get("energy_before_pj")) + metric_cell(row.get("energy_after_pj"))
                       + metric_cell(row.get("energy_delta_pj"), signed=True)
                       + metric_cell(row.get("completion_difference_ns"), signed=True) + '</tr>')
        out.append('</tbody></table></div><p class=note>' + link(energy_path, directory, "HBF 能耗修复记录")
                   + '。旧 HBF 仿真的能耗不能作为最终成本；该修复的事务对照未改变时序与字节数。</p>')
    out.append('<p class=note>默认预设矩阵验证功能和物理成本参与，不声明所有执行策略都与 native 一致。'
               'RoPE 三角函数计算方式、末层输出行选择等策略以各场景的显式配置为准；native 配对还要核对实际提交与准备场景的完整执行合同。</p>'
               '<p class=note>采用聚合缓存命中模型的默认路径，会按既有方向命中比例把未命中字节映射到物理子范围；'
               '总读写字节保持守恒，但具体未命中地址标记为 partial，不代表逐 cache line 的真实访问序列。</p></section>'
           '<section><h2>三项性能优化的等价性证据</h2><p>以下优化保持原有计算语义，分别量化验证数值差异。</p>'
           '<div class=table-wrap><table><thead><tr><th>变更</th><th>对照范围</th><th>数值字段数</th>'
           '<th>最大绝对差</th><th>最大相对差（比例）</th><th>检查结果</th></tr></thead><tbody>')
    for pattern, label in (("runtime_*.json", "复用已解析的模型执行视图"),
                           ("bank_snapshot_*.json", "直接复制 DRAM bank 状态"),
                           ("physical_transaction_optimization_equivalence.json", "消除重复的物理事务状态复制")):
        paths = sorted(directory.glob(pattern))
        if not paths:
            out.append(f'<tr><th>{label}</th><td colspan=5>证据未生成</td></tr>')
        for path in paths:
            evidence = read_json(path)
            cases = evidence.get("cases", [evidence])
            for index, case in enumerate(cases):
                absolute = case.get("max_absolute_error", case.get("max_prediction_absolute_error",
                                    case.get("max_absolute_prediction_error")))
                relative = case.get("max_relative_error", case.get("max_prediction_relative_error",
                                    case.get("max_relative_prediction_error")))
                differences = case.get("numeric_mismatches", case.get("numeric_changes", case.get("mismatches", [])))
                other = case.get("other_mismatches", case.get("timing_or_energy_changes", []))
                identical = absolute == 0 and relative == 0 and not differences and not other and case.get("all_fields_identical", True)
                verdict = "数值一致" if identical else "查看差异记录"
                scope = f'{case.get("prompt_tokens", evidence.get("prompt_tokens", "—"))} 输入 / {case.get("output_tokens", evidence.get("output_tokens", "—"))} 输出'
                if case.get("model"):
                    scope = str(case["model"]) + "；" + scope
                if "source_get_rows" in case:
                    scope += "；GET_ROWS 已绑定" if case["source_get_rows"] else "；GET_ROWS 未绑定"
                out.append('<tr><th>' + link(path, directory, label) + '</th><td>' + text(scope) + '</td>'
                           + metric_cell(case.get("numeric_fields_compared")) + metric_cell(absolute)
                           + metric_cell(relative) + '<td>' + verdict + '</td></tr>')
                before = case.get("before", {}).get("snapshot_calls")
                after = case.get("after", {}).get("snapshot_calls")
                if before is not None and after is not None:
                    out.append('<tr><td colspan=6 class=note>物理事务状态复制次数：'
                               + number(before) + ' → ' + number(after)
                               + '。保留外层完整回滚；这不是端到端加速倍数。</td></tr>')
            micro = evidence.get("microbenchmark")
            if micro:
                out.append('<tr><td colspan=6 class=note>状态复制微基准：' + number(micro.get("bank_count"))
                           + ' 个 bank、' + number(micro.get("samples")) + ' 组样本，中位耗时 '
                           + number(micro.get("median_seconds_per_snapshot", {}).get("before"), 1e3) + ' → '
                           + number(micro.get("median_seconds_per_snapshot", {}).get("after"), 1e3) + ' ms；加速 '
                           + number(micro.get("speedup")) + ' 倍。仅测状态复制，不代表端到端加速倍数。</td></tr>')
    out.append('</tbody></table></div><p class=note>零差异只覆盖记录中的测试模型与负载。'
               '全程耗时受同时运行的任务、首次编译与缓存影响，不据此推断端到端加速倍数。</p>')
    for path in sorted(directory.glob("embedding_audit_*.json")):
        evidence = read_json(path)
        count = sum(case.get("numeric_fields_compared", 0) for case in evidence.get("cases", []))
        out.append('<p class=note>历史中间记录：' + link(path, directory, "Embedding 审计字段对照")
                   + '，比较 ' + number(count) + ' 个数值字段。它发生在后续索引行读取语义修复之前，'
                   '不代表最终 Embedding 实现保持零差异，不计入上述三项性能优化。</p>')
    out.append('</section>')
    return "".join(out)


def test_section(directory):
    path = directory / "regression_results.json"
    if not path.is_file():
        return '<section><h2>代码回归测试</h2><p>尚无已保存的测试结果。</p></section>'
    results = read_json(path)
    out = ['<section><h2>代码回归测试</h2><p>测试结果与仿真矩阵进度分别记录；测试通过不表示全部组合已跑完，也不表示预测精度合格。</p>',
           '<div class=table-wrap><table><thead><tr><th>范围</th><th>命令</th><th>状态</th>'
           '<th>通过项数</th><th>耗时（秒）</th></tr></thead><tbody>']
    if results.get("current_code_validation_status") == "pending":
        out.insert(1, '<p class=warning>当前代码的最终完整回归尚待执行；下面是已完成的历史测试记录，不能代替最新代码验证。</p>')
    for run in results.get("runs", []):
        completed = run.get("status") == "completed"
        passed = completed and run.get("exit_code") == 0
        status = "通过" if passed else "失败" if completed else status_name(run.get("status"))
        out.append('<tr data-test-id="' + text(run["id"]) + '" data-status="' + text(run.get("status"))
                   + '"><th>' + text(run["scope"]) + '</th><td><code>' + text(run["command"])
                   + '</code></td><td>' + text(status) + '</td>'
                   + metric_cell(run.get("passed") if completed else None)
                   + metric_cell(run.get("duration_seconds") if completed else None) + '</tr>')
    out.append('</tbody></table></div>')
    for note in results.get("notes", []):
        out.append('<p class=note>' + text(note) + '</p>')
    out.append('<p class=note>' + link(path, directory, "测试结果记录") + '</p></section>')
    return "".join(out)


def result_consistency_section(directory):
    path = directory / "regression_results.json"
    if not path.is_file():
        return ""
    audit = read_json(path).get("result_consistency_audit")
    if not audit:
        return ""
    moe = audit["moe_cross_hardware"]
    nand = audit["nand_energy_distribution"]
    final = audit.get("final_235b_results")
    final_note = ""
    if final:
        e2e = final["cross_hardware_differences"]["engine_e2e_ns"]
        storage = final["cases"][0]["physical_traffic"]["storage_traffic"]
        rate = storage["owners"]["hbf0.memory"]["coefficient_pj_per_byte"]
        final_note = ('<p>最后两组 235B 均完成 128 个物理批次；只有 hbf0 发生 NAND 活动。'
                      '两组 NAND 介质流量均为 ' + text(f'{storage["physical_bytes"]:,}')
                      + ' bytes，按 ' + number(rate) + ' pJ/byte 计费 '
                      + text(f'{storage["media_energy_pj"]:,.0f}')
                      + ' pJ，资源分摊总和与介质账单一致。3HBF 减 2HBF E2E 差为 '
                      + text(format(e2e["signed_difference_3hbf_minus_2hbf"], ".10g"))
                      + ' ns，相对差为 ' + text(format(e2e["signed_relative_difference_to_2hbf"], ".10g"))
                      + '；其他时序末位差及各 owner 明细保存在审计记录，资源服务时间之和不能当成 E2E。</p>'
                      '<p class=note>' + link(directory / final["ui_screenshot"], directory,
                                            "235B / 3HBF 最终实际前端截图") + '</p>')
    return ('<section><h2>已完成结果的一致性检查</h2><p>此检查记录生成时的 '
            + str(audit["completed_matrix_rows_checked"])
            + ' 个完成组合均符合 512 输入 / 128 输出、llama 调度、1 请求完成及 128 个实际持久物理批次。'
            '405B 在两个 B200 配置中使用 hbf0、hbf1；3HBF 配置中的 hbf2 已存在但未使用，因此增加该组件不必改变结果。</p>'
            '<p>Qwen3-30B-A3B 的 3HBF 减 2HBF E2E 差为 '
            + text(format(moe["signed_difference_3hbf_minus_2hbf_ns"], ".10g")) + ' ns，除以 2HBF E2E 的相对差为 '
            + text(format(moe["signed_relative_difference_to_2hbf"], ".10g"))
            + '；两者物理字节及原总能耗相同。已检查的新版 NAND 结果中，逐资源分摊能耗之和减介质账单的最大绝对差为 '
            + text(format(abs(nand["signed_difference_resource_minus_media_pj"]), ".10g"))
            + ' pJ，发生于 2HBF / Qwen2.5-72B；有符号相对差为 '
            + text(format(nand["signed_relative_difference_to_media"], ".10g"))
            + '（以介质账单为分母）。这些是浮点累计末位的一致性检查，不是优化开关误差实验，'
            '不表示所有硬件间或所有字段严格相同。</p>' + final_note + '<p class=note>'
            + link(path, directory, "result_consistency_audit 原值、job ID 与比较口径") + '</p></section>')


def physical_capture_section(directory):
    path = directory / "physical_capture_equivalence.json"
    if not path.is_file():
        return '<section><h2>物理明细保留开关的数值对照</h2><p>证据尚未生成。</p></section>'
    evidence = read_json(path)
    rows = evidence.get("core_comparisons", [])
    out = ['<section><h2>物理明细保留开关的数值对照</h2>'
           '<p>正式汇总路径关闭逐 burst 明细保留，仍执行物理内核。以下将保留 / 不保留明细的实现直接比较；'
           '排除刻意省略的展开轨迹及实现标记，列出实际浮点差异，不以测试容差当作零差异。</p>'
           '<div class=table-wrap><table><thead><tr><th>硬件 / 内存</th><th>请求数</th>'
           '<th>时间变化字段数</th><th>最大时间差（ns）</th><th>最大相对时间差（%）</th>'
           '<th>最大状态差</th><th>字节 / 能耗 / 计数变化字段数</th><th>非数值不一致数</th></tr></thead><tbody>']
    maximum = None
    for row in rows:
        differences = row.get("differences", {})
        timing, state = differences.get("times", {}), differences.get("state", {})
        maximum = max(maximum or 0, timing.get("max_absolute_difference", 0))
        label = HARDWARE_NAMES.get(row.get("hardware_id"), row.get("hardware_id", "未记录"))
        label += ' / ' + str(row.get("component_id", "未记录")) + ' (' + str(row.get("kind", "未记录")) + ')'
        other = sum(len(value.get("non_numeric_mismatches", [])) for value in differences.values())
        out.append('<tr data-capture-hardware="' + text(row.get("hardware_id"))
                   + '" data-capture-component="' + text(row.get("component_id")) + '"><th>' + text(label) + '</th>'
                   + metric_cell(row.get("request_count")) + metric_cell(timing.get("changed_values"))
                   + metric_cell(timing.get("max_absolute_difference"))
                   + metric_cell(timing.get("max_relative_difference_percent"))
                   + metric_cell(state.get("max_absolute_difference")) + '<td>'
                   + ' / '.join(number(differences.get(key, {}).get("changed_values"))
                                for key in ("bytes", "energy_pj", "counters")) + '</td>'
                   + metric_cell(other) + '</tr>')
    out.append('</tbody></table></div><p class=note>本组最大时间绝对差：<strong>'
               + text(maximum if maximum is not None else "未记录") + ' ns</strong>。'
               '状态差来自 bank / channel 时间状态；字节、能耗及计数分别检查，不能从时间差推断。'
               '能耗沿用各预设的 pJ/byte，与物理字节相乘；并未独立校准 DRAM 能耗。</p>')
    whole = evidence.get("whole_run_comparison", {})
    if whole:
        difference = whole.get("differences", {})
        out.append('<h3>完整短流程对照</h3><p class=note>' + text(whole.get("scope", "未记录范围"))
                   + '</p><table><thead><tr><th>数值字段数</th><th>变化字段数</th><th>最大绝对差</th>'
                   '<th>最大相对差（%）</th><th>非数值不一致数</th></tr></thead><tbody><tr>'
                   + metric_cell(difference.get("numeric_values_compared"))
                   + metric_cell(difference.get("changed_values"))
                   + metric_cell(difference.get("max_absolute_difference"))
                   + metric_cell(difference.get("max_relative_difference_percent"))
                   + metric_cell(len(difference.get("non_numeric_mismatches", []))) + '</tr></tbody></table>')
    out.append('<p class=note>以上结论只覆盖所列组件、请求与短流程，不代表所有模型都零差异，也不代表相对 native 的预测误差。'
               + link(path, directory, "逐项数值对照记录") + '</p></section>')
    return "".join(out)


def physical_participation_section(data):
    attempts = index_attempts(data)
    completed = [attempts[row["latest_attempt"]] for row in data.get("matrix", [])
                 if row.get("status") == "completed" and row.get("latest_attempt") in attempts]
    out = ['<section><h2>实际物理成本参与</h2><p>只列最终矩阵中已完成的仿真；字节及能耗来自实际执行汇总。'
           '服务时间会重叠，不能相加视为总延迟。总能耗标明来源：修复后前端运行，或同次物理记录重新计费；'
           '后者保留原运行和原能耗，不代表重新运行。无完整依据的旧能耗仍标未复核。</p><p class=warning>'
           '读写切换次数、refresh 等待、turnaround 等待尚无完整独立统计；新报告使用 null / 可用性标记。'
           '历史结果中的默认 0 不可视为实测零值；空 organization_profiles 也不表示物理组织信息完整。'
           'turnaround 时序成本参与物理内核，但独立等待小计未计量；当前内核没有 refresh 参数，不能宣称已模拟刷新成本。'
           'NAND 的 read_operations、pages_touched、media_waves 等旧默认 0 同样不能证明未发生读取；'
           '新汇总保留实际 pages_read / pages_programmed，缺少的独立计数与历史逻辑读写方向标为未记录。</p>']
    if not completed:
        out.append('<p>尚无最终已完成矩阵结果可列示。</p></section>')
        return "".join(out)
    out.append('<details><summary>展开 ' + str(len(completed)) + ' 个已完成组合的物理汇总</summary>'
               '<div class=table-wrap><table><thead><tr><th>硬件 / 模型</th><th>持久物理批次</th>'
               '<th>DRAM 任务数</th><th>DRAM 读（GB）</th><th>DRAM 写（GB）</th>'
               '<th>NAND 读（GB）</th><th>NAND 写（GB）</th><th>总能耗（J）</th></tr></thead><tbody>')
    for row in completed:
        participation = row.get("participation", {})
        traffic = participation.get("physical_traffic", {})
        dram, nand = traffic.get("dram_traffic", {}), traffic.get("storage_traffic", {})
        label = HARDWARE_NAMES.get(row.get("hardware_id"), row.get("hardware_id", "未记录"))
        label += ' / ' + model_name(row.get("model_id", "未记录"))
        out.append('<tr data-physical-prefix="' + text(row["prefix"]) + '"><th>' + text(label) + '</th><td>'
                   + text(verify_live_execution(row)) + '</td>' + metric_cell(dram.get("task_count"))
                   + metric_cell(dram.get("physical_read_bytes"), scale=1e-9)
                   + metric_cell(dram.get("physical_write_bytes"), scale=1e-9)
                   + metric_cell(nand.get("physical_read_bytes"), scale=1e-9)
                   + metric_cell(nand.get("physical_write_bytes"), scale=1e-9)
                   + energy_cell(row) + '</tr>')
    out.append('</tbody></table></div></details><p class=note>GB = 10⁹ 字节；“—”表示没有该统计值，不按零处理。</p></section>')
    return "".join(out)


def energy_cell(row):
    if row.get("energy_accounting") == "physical_profile_energy_v2":
        return metric_cell(row.get("participation", {}).get("total_energy_pj"), scale=1e-12).replace(
            "</td>", '<br><small>修复后前端运行</small></td>')
    record = row.get("postprocessed_energy", {})
    if record.get("status") == "repriced" and record.get("job_id") == row.get("job_id"):
        return metric_cell(record.get("repriced_total_energy_pj"), scale=1e-12).replace(
            "</td>", '<br><small>物理记录重计</small></td>')
    return '<td class=note>未按修复后代码复核</td>'


def energy_repricing_section(data, directory):
    path = directory / "energy_repricing.json"
    if not path.is_file():
        return ""
    evidence = read_json(path)
    rows = [row for row in data.get("attempts", []) if row.get("postprocessed_energy")]
    validation = evidence.get("validation", {})
    out = ['<section><h2>同次物理执行记录重新计费</h2><p>只处理有完整归档的纯 DRAM 已完成记录。'
           '逐批次资源流量、原能耗及 job ID 与原报告核对一致；使用同次前端提交的显式内存 profile 系数。'
           '新总能耗 = 原总能耗 − 原物理 DRAM 能耗 + 各 owner 物理字节 × 对应系数。'
           '计算、缓存和链路的其余能耗保留原值；原结果未修改，也没有重新仿真。</p>'
           '<p class=note>归档没有逐 task 元数据，因此同时审查系数生成路径：普通 DRAM 任务来自组件 profile，'
           '历史端点路径缺少系数。提交或归档出现显式任务系数覆盖、未知 owner、字节不守恒、缺批次或 NAND 活动时拒绝此重计。'
           '这是仿真能耗的重新计费，不是硬件功耗实测。</p><p>已用 '
           + number(validation.get("v2_execution_count")) + ' 份修复后前端执行校验公式；总能耗还原最大绝对差 '
           + number(validation.get("maximum_total_reconstruction_difference_pj"))
           + ' pJ。一般浮点求和可能出现末位差。</p>']
    out.append('<details><summary>展开 ' + str(len(rows)) + ' 份物理记录重计</summary>'
               '<div class=table-wrap><table><thead><tr><th>运行 / job ID</th><th>原总能耗（J）</th>'
               '<th>重计总能耗（J）</th><th>变化（J）</th><th>各 owner 依据</th></tr></thead><tbody>')
    for attempt in rows:
        row = attempt["postprocessed_energy"]
        owner_text = '; '.join(str(owner["owner"]) + ': ' + str(owner["physical_bytes"])
                               + ' bytes × ' + str(owner["coefficient_pj_per_byte"]) + ' pJ/byte'
                               for owner in row["owners"])
        out.append('<tr data-repriced-prefix="' + text(attempt["prefix"]) + '"><th>'
                   + text(attempt["prefix"]) + '<br><small>' + text(row["job_id"]) + '</small></th>'
                   + metric_cell(row["original_total_energy_pj"], scale=1e-12)
                   + metric_cell(row["repriced_total_energy_pj"], scale=1e-12)
                   + metric_cell(row["delta_energy_pj"], scale=1e-12, signed=True)
                   + '<td>' + text(owner_text) + '</td></tr>')
    out.append('</tbody></table></div></details><p class=note>'
               + link(path, directory, "逐 owner 原值、系数、差额与归档核对依据") + '</p></section>')
    return "".join(out)


def workspace_allocator_section(directory):
    path = directory / "workspace_allocator_equivalence.json"
    if not path.is_file():
        return ""
    evidence = read_json(path)
    sequence = evidence.get("allocator_sequence", {})
    out = ['<section><h2>工作区分配排序优化</h2><p>仅优化 arena 候选筛选与排序；同一对照保留相同的工作区预留、'
           '物理地址和能耗修复。固定分配序列比较 ' + number(sequence.get("allocation_alias_rejection_events_compared"))
           + ' 个分配、别名及拒绝事件，地址最大差 ' + number(sequence.get("address_difference_bytes"))
           + ' bytes。</p><div class=table-wrap><table><thead><tr><th>模型</th><th>输入 / 输出</th>'
           '<th>数值字段</th><th>数值不一致</th><th>其他不一致</th><th>最大绝对差</th></tr></thead><tbody>']
    for row in evidence.get("models", []):
        workload = row.get("workload", {})
        out.append('<tr data-arena-model="' + text(row["model_id"]) + '"><th>' + text(model_name(row["model_id"]))
                   + '</th><td>' + text(str(workload.get("prompt_tokens")) + ' / ' + str(workload.get("output_tokens")))
                   + '</td>' + metric_cell(row.get("numeric_fields_compared"))
                   + metric_cell(row.get("numeric_mismatch_count")) + metric_cell(row.get("other_mismatch_count"))
                   + metric_cell(row.get("max_absolute_difference")) + '</tr>')
    out.append('</tbody></table></div><p class=note>导出场景缩小负载后继续保留既有 arena，避免空间被重复分配；'
               'ledger 显示及控制面指纹同步已修复。缓存键仅忽略 mapping_stale / mapping_stale_reason 两个 UI 标志。'
               '这些边界修复不改变本次从原始硬件及模型预设直接运行的矩阵放置、物理地址与预测；'
               '上述零差异仅指列出的排序实现对照，不扩大到其他成本纠错。'
               + link(path, directory, "分配序列与实际模型对照记录") + '</p></section>')
    return "".join(out)


def dram_batch_section(directory):
    path = directory / "dram_batch_equivalence.json"
    if not path.is_file():
        return ""
    evidence = read_json(path)
    cases = []
    for row in evidence.get("core_cases", []):
        label = HARDWARE_NAMES.get(row.get("hardware"), row.get("hardware", "未记录"))
        cases.append((label + ' / ' + str(row.get("component", "未记录")), row))
    for key, label in (("native_v_columns", "1024 次 V cache 列写请求"),
                       ("interleaved_multiowner_overlap_case", "多 owner 交错与重叠读写")):
        if key in evidence:
            cases.append((label, evidence[key]))
    for row in evidence.get("real_source_layers", []):
        cases.append((str(row.get("case")) + '：实际 attention 层，' + number(row.get("repetitions")) + ' 次', row))
    whole = evidence.get("real_source_whole_model")
    if whole:
        cases.append((str(whole.get("case")) + '：完整模型 4 / 3，' + number(whole.get("repetitions")) + ' 次', whole))
    out = ['<section><h2>逐请求 DRAM 批量执行的独立对照</h2>'
           '<p>此项只减少相同物理内核状态的重复装箱，保留每个请求的地址、字节数、提交次序及并发限制。'
           '对照双方均关闭明细保留，单独量化批量执行变更；与上一节的明细开关差异分开。</p>'
           '<p class=note>启用范围：连续至少 16 个同 owner、同方向的小请求，每请求少于 64 burst；'
           '写请求按地址升序且不重叠，每批最多 1024 个请求。详细轨迹路径保留逐条执行。</p>'
           '<div class=table-wrap><table><thead><tr><th>对照范围</th><th>数值字段数</th><th>变化字段数</th>'
           '<th>最大绝对差</th><th>最大相对差（%）</th><th>逐条耗时（秒）</th>'
           '<th>批量耗时（秒）</th><th>加速倍数</th></tr></thead><tbody>']
    for label, row in cases:
        difference = row.get("difference", {})
        out.append('<tr><th>' + text(label) + '</th>'
                   + metric_cell(difference.get("numeric_values_compared"))
                   + metric_cell(difference.get("changed_values"))
                   + metric_cell(difference.get("max_absolute_difference"))
                   + metric_cell(difference.get("max_relative_difference_percent"))
                   + metric_cell(row.get("serial_wall_seconds")) + metric_cell(row.get("batch_wall_seconds"))
                   + metric_cell(row.get("speedup")) + '</tr>')
        if difference.get("non_numeric_mismatches"):
            out.append('<tr><td colspan=8 class=warning>存在非数值差异，请查看原始记录。</td></tr>')
    out.append('</tbody></table></div><p class=note>核心微基准为单次记录；层级 / 完整模型耗时为重复测量的中位数，'
               '测量时其他 UI 仿真也在运行。完整模型短负载对照不等于正式 512 / 128，也不是 native 预测误差；'
               '零差异只对记录中的数值与最终物理状态成立，不泛化到所有模型。</p><p class=note>'
               + link(path, directory, "批量执行数值与耗时证据") + '</p>')
    representative_path = directory / "dram_batch_representative_equivalence.json"
    if representative_path.is_file():
        representative = read_json(representative_path)
        out.append('<h3>其他完整模型的非触发路径兼容性</h3><p>以下使用 B200 + 2HBF、1 输入 / 2 输出，'
                   '比较完整报告及物理内核、时间线、分配器和缓存的最终状态。'
                   '三个用例都没有触发 submit_many，因此只证明启用开关后的非触发路径兼容，'
                   '不算批量路径覆盖，也不把墙钟耗时波动列为加速收益。</p>'
                   '<div class=table-wrap><table><thead><tr><th>模型</th><th>数值字段</th><th>变化字段</th>'
                   '<th>最大绝对差</th><th>submit_many 次数</th><th>批量结果数</th></tr></thead><tbody>')
        for row in representative.get("cases", []):
            diff = row.get("difference", {})
            counts = row.get("active_path_verification", {}).get("counters", {})
            out.append('<tr data-batch-compatibility-model="' + text(row["model"]) + '"><th>'
                       + text(row["model"]) + '</th>' + metric_cell(diff.get("numeric_values_compared"))
                       + metric_cell(diff.get("changed_values")) + metric_cell(diff.get("max_absolute_difference"))
                       + metric_cell(counts.get("submit_many_calls"))
                       + metric_cell(counts.get("compiled_batch_results")) + '</tr>')
        out.append('</tbody></table></div><p class=note>'
                   + link(representative_path, directory, "三模型兼容性与触发计数") + '</p>')
    out.append('</section>')
    return "".join(out)


STYLE = """
:root{color-scheme:light;font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif;color:#172c3c;background:#edf2f5}
body{max-width:1180px;margin:32px auto;padding:0 20px}header,section{background:white;padding:24px 28px;margin:18px 0;border-radius:12px;border:1px solid #d6e0e7}
h1{font-size:28px;line-height:1.25;margin:8px 0 18px}h2{font-size:20px;margin-top:0}h3{font-size:17px;margin-top:28px}.eyebrow{color:#486c83;font-size:13px}.note{color:#536575;font-size:13px}.warning{color:#9a4911}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;border:1px solid #dfe6eb;text-align:left;vertical-align:top}thead th{background:#eff4f7}tbody th{font-weight:550}td.number{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.table-wrap{overflow:auto}.matrix{min-width:1000px}small{display:block;font-size:11px;color:#667985;margin-top:4px}.status{font-size:13px}.completed{color:#176847}.capacity_rejected{color:#976224}.running,.queued{color:#225ca0}.pending,.validated_not_run{color:#657483}.program_error,.failed,.input_contract_failed{color:#a32130}a{color:inherit;text-decoration:underline;text-underline-offset:3px}li{margin:6px 0}footer{color:#536575;font-size:12px;padding:12px 4px}
@media print{body{background:white;margin:0;padding:0;max-width:none}header,section{border:0;padding:10px 0;break-inside:avoid}.matrix{min-width:0}a{text-decoration:none}th,td{padding:6px}.table-wrap{overflow:visible}}
"""


def render(data, directory):
    stamp = localized_time(data.get("generated_at_utc"))
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>前端仿真与 native 实测对照</title><style>' + STYLE + '</style></head><body>'
            '<header><div class=eyebrow>HeteroLLM Simulator · 验证记录</div><h1>前端仿真与 native 实测对照</h1>'
            '<p>63 个默认预设组合验证前端执行流程；两组相同 GGUF 文件验证本机预测误差。'
            '执行成功、容量边界和预测误差分别列出。</p><p class=note>数据生成时间：' + text(stamp)
            + ' · ' + link(directory / 'comparison.json', directory, '汇总 JSON') + '</p></header>'
            + matrix_section(data, directory) + native_section(data, directory) + historical_native_graph_section(directory)
            + physical_participation_section(data)
            + result_consistency_section(directory)
            + energy_repricing_section(data, directory)
            + paired_cost_scope_section(directory) + evidence_section(directory) + physical_capture_section(directory)
            + dram_batch_section(directory)
            + workspace_allocator_section(directory)
            + optimization_matrix_section(directory)
            + test_section(directory) + recording_section(data)
            + '<footer>本文件只读取已保存证据，无外部依赖、不联网、不启动仿真。'
            '数据按生成时的 comparison.json 展示；重新生成后更新状态。</footer></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    directory = args.directory.resolve()
    data = read_json(directory / "comparison.json")
    output = directory / "report.html"
    output.write_text(render(data, directory), encoding="utf-8")
    print(output)
    print("matrix:", dict(Counter(row.get("status", "pending") for row in data.get("matrix", []))))
    print("native:", {row["case_id"]: row.get("status") for row in data.get("native_comparisons", [])})


if __name__ == "__main__":
    main()
