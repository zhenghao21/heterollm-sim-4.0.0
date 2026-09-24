#!/usr/bin/env python3
"""Render a read-only multiworkload summary snapshot with standard-library tools."""
import argparse
from collections import Counter, OrderedDict
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import quote


def text(value):
    return escape(str(value))


def number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "—"
    return format(value, ".9g")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_json(path):
    # The runner may be replacing/writing an artifact at this instant.
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(0.2)


def read_summary(path):
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ValueError("summary 必须包含 rows 列表")
    return data


def file_stamp(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino
    except FileNotFoundError:
        return None


def cell_snapshot(run_dir):
    """Read each declared file once; retry if the inventory or any file changes.

    Only manifest-declared cells are included, never diagnostic reruns or a
    mixture with the stalled summary. Missing cells remain unobserved, not zero.
    """
    manifest_path = run_dir / "manifest.json"
    for attempt in range(5):
        manifest_stamp = file_stamp(manifest_path)
        manifest = read_json(manifest_path)
        stems = list(dict.fromkeys(load[0] + "__" + case for load in manifest["workloads"]
                                   for group in manifest["groups"] for case in group["cases"]))
        if any(Path(stem).name != stem for stem in stems):
            raise ValueError("manifest 中的点名称必须是本目录文件名")
        paths = [run_dir / folder / (stem + ".json") for stem in stems for folder in ("cells", "configs")]
        before = {path: file_stamp(path) for path in paths}
        artifacts = {path: read_json(path) for path in paths if before[path] is not None}
        if (manifest_stamp == file_stamp(manifest_path) and
                all(before[path] == file_stamp(path) for path in paths)):
            return manifest, artifacts, datetime.now().astimezone().isoformat(timespec="seconds")
        if attempt < 4:
            time.sleep(0.2)
    raise ValueError("快照期间输入持续变化；未生成混合时间的报告，请稍后重试")


def summary_from_cells(run_dir):
    # Import only existing pure validators; do not run/build scenarios or write
    # bytecode into the frozen source tree.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from tools import multiworkload_architecture_matrix as matrix
        from tools import memory_tier_sweep as sweep
    finally:
        sys.dont_write_bytecode = previous
    manifest, artifacts, snapshot_at = cell_snapshot(run_dir)
    source_sha = matrix.q.stable_hash(manifest["source"])
    cancelled = set(manifest.get("cancelled_cells", []))
    rows = []
    for group in manifest["groups"]:
        if len(group["cases"]) != len(group["values"]):
            raise ValueError("manifest 的 cases / values 长度不匹配")
        for workload in manifest["workloads"]:
            load = dict(zip(("id", "prompt_tokens", "output_tokens", "batch"), workload))
            points, cells, configs = [], [], []
            cancelled_cases = []
            for case, value in zip(group["cases"], group["values"]):
                stem = load["id"] + "__" + case
                result_path, config_path = (Path(folder) / (stem + ".json") for folder in ("cells", "configs"))
                if stem in cancelled:
                    if run_dir / result_path in artifacts:
                        raise ValueError("不能用取消记录隐藏已落盘结果：" + stem)
                    cancelled_cases.append(case)
                    continue
                cell = artifacts.get(run_dir / result_path, {})
                config = artifacts.get(run_dir / config_path)
                status = cell.get("status", "PENDING")
                observed = status != "PENDING"
                obs = (cell.get("observation") or {}) if observed else {}
                summary = obs.get("summary") or {}
                raw_metrics = obs.get("metrics") or {}
                metrics = {key + "_ms": raw_metrics.get(key + "_ns") / 1e6
                           if finite(raw_metrics.get(key + "_ns")) else None
                           for key in ("ttft", "tpot", "e2e")} if status == "SIMULATED" else {}
                throughput = (summary.get("throughput") or {}).get("visible_output_tokens_per_s")
                failed = [check for check in cell.get("checks", []) if check.get("status") != "PASS"] if observed else []
                errors = []
                if status == "SIMULATED":
                    if config is None:
                        errors.append("无有效配置：不能参与输入可比性检查")
                    elif cell.get("config_sha256") != matrix.q.stable_hash(config):
                        errors.append("配置哈希与正式点不匹配")
                    if cell.get("source_stable") is not True or cell.get("source_sha256") != source_sha:
                        errors.append("源码身份与冻结 manifest 不匹配或未确认稳定")
                    if cell.get("id") != stem or cell.get("variant") != case or cell.get("workload") != load:
                        errors.append("点身份或负载与 manifest 不匹配")
                    if (not obs.get("model_sha256") or not obs.get("workload_sha256") or
                            not isinstance(obs.get("placement"), dict)):
                        errors.append("实际模型/负载身份或映射缺失")
                    if (failed or not all(finite(v) and v > 0 for v in metrics.values()) or
                            not finite(throughput) or throughput < 0):
                        errors.append("成功状态与检查或完整有限指标不一致")
                    if (summary.get("completed_requests") != load["batch"] or summary.get("rejected_requests") != 0 or
                            obs.get("expected_requests") != load["batch"] or
                            obs.get("expected_output_tokens") != load["batch"] * load["output_tokens"] or
                            obs.get("actual_output_tokens") != load["batch"] * load["output_tokens"]):
                        errors.append("成功状态与请求/输出完成量不一致")
                point = {
                    "case": case, "value": value, "status": status,
                    "config": config_path.as_posix() if config is not None else None,
                    "result": result_path.as_posix() if cell else None,
                    "metrics_ms": metrics,
                    "throughput_tokens_per_s": throughput if status == "SIMULATED" and finite(throughput) else None,
                    "resource_accounted_bytes": obs.get("reported_resource_bytes") if status == "SIMULATED" else None,
                    "busiest_resource": summary.get("bottleneck_resource") if status == "SIMULATED" else None,
                    "completed_requests": summary.get("completed_requests") if observed else None,
                    "rejected_requests": summary.get("rejected_requests") if observed else None,
                    "actual_output_tokens": obs.get("actual_output_tokens") if observed else None,
                    "expected_requests": load["batch"] if observed else None,
                    "expected_output_tokens": load["batch"] * load["output_tokens"] if observed else None,
                    "failed_checks": failed,
                    "error": cell.get("error") if observed else None,
                    "comparison_eligible": status == "SIMULATED" and not errors,
                    "comparison_errors": errors,
                }
                points.append(point)
                cells.append(cell)
                configs.append(config)
            valid = [i for i, p in enumerate(points) if p["comparison_eligible"]]
            compared = [cells[i] for i in valid]
            changed = [matrix.differences(configs[valid[0]], configs[i]) for i in valid[1:]]
            unexpected = sorted({p for change in changed for p in change if not matrix.allowed_change(p, group["kind"])})
            comparison_errors = []
            try:
                fixed_mapping = len(valid) >= 2 and matrix.actual_mapping_matches(compared, group["kind"])
            except (KeyError, TypeError, ValueError) as exc:
                fixed_mapping = False
                comparison_errors.append("实际映射检查缺失/无效：" + str(exc))
            feasible = len(valid) >= 2 and not unexpected and fixed_mapping and all(changed)
            comparable = feasible and len(valid) == len(points) and not cancelled_cases
            trend = sweep.monotonicity([
                {"value": points[i]["value"], "observation": cells[i]["observation"]} for i in valid
            ], parameter=group["kind"], direction=group["direction"]) if feasible and "direction" in group else None
            trend_ok = not trend or trend["status"] == "PASS"
            credibility = ("PARTIAL_USER_CANCELLED_UNVALIDATED" if cancelled_cases and
                           len(valid) == len(points) and trend_ok else
                           "MECHANISM_ONLY_UNVALIDATED" if comparable and trend_ok else
                           "PARTIAL_CAPACITY_LIMIT_UNVALIDATED" if feasible and trend_ok and
                           any(p["status"] == "BLOCKED" for p in points) and
                           all(p["comparison_eligible"] or p["status"] == "BLOCKED" for p in points) else
                           "PENDING" if any(p["status"] == "PENDING" for p in points) else "NOT_ACCEPTED")
            delta = {key: cells[-1]["observation"]["metrics"][key] / cells[0]["observation"]["metrics"][key] - 1
                     for key in ("ttft_ns", "tpot_ns", "e2e_ns")} if comparable else None
            rows.append({"question_id": group["id"], "question": group["question"], "kind": group["kind"],
                         "load": load, "points": points, "comparable": comparable,
                         "cancelled_cases": cancelled_cases,
                         "feasible_points_comparable": feasible, "fixed_mapping": fixed_mapping,
                         "compared_cases": [points[i]["case"] for i in valid], "comparison_errors": comparison_errors,
                         "unexpected_changes": unexpected, "actual_changed_paths": changed,
                         "delta_last_vs_first": delta, "trend_check": trend, "credibility": credibility})
    return {"schema": "multiworkload-matrix/report-v1", "validation_status": "UNVALIDATED",
            "data_source": "report_summary.json", "source_mode": "manifest/cells/configs",
            "snapshot_at": snapshot_at, "manifest_source_sha256": source_sha,
            "cancelled_cells": sorted(cancelled),
            "model": manifest.get("model", {}), "limits": manifest.get("limits", []), "rows": rows}


def link(run_dir, relative, label):
    if not relative:
        return text(label) + ("（无有效配置）" if label == "输入配置" else "（未提供）")
    candidate = (run_dir / str(relative)).resolve()
    try:
        local = candidate.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return text(label) + "（路径不在运行目录内）"
    if not candidate.is_file():
        return text(label) + ("（无有效配置）" if label == "输入配置" else "（文件不存在）")
    return '<a href="' + quote(local, safe="/") + '">' + text(label) + '</a>'


def conclusion(row):
    points = row.get("points", [])
    states = [p.get("status", "PENDING") for p in points]
    notes = []
    if row.get("cancelled_cases"):
        notes.append("用户取消并删除未完成点：" + ", ".join(row["cancelled_cases"]) +
                     "；不计成功/失败，不再排队。仅展示保留点，不能外推原完整扫描范围。")
    if "PENDING" in states or not points:
        notes.append("仍有待完成点；本行结论暂不完整。")
    if "HOST_TIMEOUT" in states:
        notes.append("HOST_TIMEOUT 仅为宿主执行限时，不代表模拟器件不可行或速度慢。")
    failures = sorted(set(states) & {"CHECK_FAILED", "FAILED", "INVALIDATED"})
    if failures:
        notes.append("存在 " + "/".join(failures) + " 点：不完整或无效，其性能和趋势均排除；完成量与失败检查见逐点说明。")
    if "BLOCKED" in states:
        capacity = any(p.get("status") == "BLOCKED" and
                       p.get("case") == "cim_scratch_4194304" for p in points)
        notes.append("P4 声明的 4 MiB scratch + 2 MiB array = 6 MiB，超过本配置 3 MiB 物理容量；该 BLOCKED 点无性能预测，具体拒绝原因见逐点说明。"
                     if capacity else "BLOCKED：本次配置构建或执行受阻，原因见逐点说明；不作为性能比较点。")
        notes.append("这不表示真实目标硬件在其他容量或调度策略下必然不可行。")
    comparable = row.get("comparable") or row.get("feasible_points_comparable")
    valid = [p for p in points if p.get("status") == "SIMULATED" and p.get("comparison_eligible", True)]
    keys = ("ttft_ms", "tpot_ms", "e2e_ms")
    vectors = [[p.get("metrics_ms", {}).get(k) for k in keys] +
               [p.get("throughput_tokens_per_s")] for p in valid]
    complete = vectors and all(finite(v)
                               for vec in vectors for v in vec)
    if comparable and len(valid) >= 2 and complete:
        scope = ("保留的可比子集" if row.get("cancelled_cases") else
                 "全部点" if row.get("comparable") else f"仅 {len(valid)}/{len(points)} 个成功且输入齐全的点")
        if all(vec == vectors[0] for vec in vectors[1:]):
            notes.append(scope + "的 TTFT / TPOT / E2E / 吞吐均相同：未观察到端到端差异；不能断言物理架构无差异。")
            if row.get("kind") in ("topology", "flash_topology"):
                notes.append("可比性成立不等于已验证共享争用代价；此处没有端到端争用效应的证据。")
            if row.get("kind") == "scratch":
                notes.append("仅限固定 tile、冷加载与单槽生命周期；平坦结果不能外推双缓冲或跨调用复用收益。")
        else:
            changes = []
            for index, label in enumerate(("TTFT", "TPOT", "E2E", "吞吐")):
                first, last = vectors[0][index], vectors[-1][index]
                changes.append(label + (f" {((last / first) - 1) * 100:+.3f}%" if first else "（基线为零，不算百分比）"))
            notes.append(scope + "末点相对首点（" + str(valid[-1].get("value")) + " 对 " +
                         str(valid[0].get("value")) + "）：" + "；".join(changes) + "。")
            notes.append("时延降低 / 吞吐增加表示该负载的模拟性能改善，不等于硬件实测结论。")
    elif len(valid) >= 2:
        notes.append("可比性未成立或指标缺失，仅展示数值，不归因于单一参数。")
    elif not notes:
        notes.append("有效完成点不足，暂不能比较。")
    trend = row.get("trend_check")
    if trend and comparable and len(valid) >= 2 and complete:
        if trend.get("status") == "PASS":
            notes.append("弱单调检查通过（允许平坦）；" +
                         ("已观察到参数变化引起的数值差异。" if trend.get("observed_sensitivity") else "未观察到参数敏感性。"))
        else:
            notes.append("弱单调检查：" + str(trend.get("status", "未知")) + "。")
        notes.append("只检查上述成功可比点的模型内趋势，不是硬件预测准确性验证。")
    return " ".join(notes)


def point_html(run_dir, point, load=None):
    status = point.get("status", "PENDING")
    metrics = point.get("metrics_ms") or {}
    enabled = status == "SIMULATED"
    values = [("TTFT", metrics.get("ttft_ms"), "ms"),
              ("TPOT", metrics.get("tpot_ms"), "ms"),
              ("E2E", metrics.get("e2e_ms"), "ms"),
              ("吞吐", point.get("throughput_tokens_per_s"), "token/s")]
    bits = ['<strong>' + text(point.get("value", "—")) + '</strong>',
            '<div class="case">' + text(point.get("case", "—")) + '</div>',
            '<span class="status">' + text(status) + '</span>', '<dl>']
    for label, value, unit in values:
        bits.append('<dt>' + label + '</dt><dd>' + number(value if enabled else None) + ' ' + unit + '</dd>')
    bits.append('</dl><div class="links">' + link(run_dir, point.get("config"), "输入配置") + ' · ' +
                link(run_dir, point.get("result"), "点结果") + '</div>')
    if enabled and point.get("resource_accounted_bytes") is not None:
        bits.append('<div class="muted">多资源累计 bytes：' + number(point["resource_accounted_bytes"]) + '</div>')
    if status != "PENDING":
        load = load or {}
        expected_requests = point.get("expected_requests", load.get("batch"))
        expected_output = point.get("expected_output_tokens")
        if expected_output is None and finite(load.get("batch")) and finite(load.get("output_tokens")):
            expected_output = load["batch"] * load["output_tokens"]
        bits.append('<div class="muted">请求：完成 ' + number(point.get("completed_requests")) +
                    ' / 期望 ' + number(expected_requests) + '；拒绝 ' + number(point.get("rejected_requests")) +
                    '<br>输出：实际 ' + number(point.get("actual_output_tokens")) + ' / 期望 ' + number(expected_output) + ' tokens</div>')
    if point.get("failed_checks"):
        bits.append('<div class="error">失败检查：' + text("；".join(
            str(check.get("check", "未知")) + "=" + str(check.get("status", "未知"))
            for check in point["failed_checks"])) + '</div><details><summary>检查详情</summary><pre>' +
            text(json.dumps(point["failed_checks"], ensure_ascii=False, indent=2)) + '</pre></details>')
    elif status == "CHECK_FAILED":
        bits.append('<div class="error">正式检查未通过；本汇总未提供失败项，请查看点结果。性能不展示。</div>')
    if point.get("comparison_errors"):
        bits.append('<div class="error">不参与可比性检查：' + text("；".join(point["comparison_errors"])) + '</div>')
    if point.get("error"):
        bits.append('<details class="error"><summary>错误原因</summary><pre>' + text(point["error"]) + '</pre></details>')
    return "".join(bits)


CSS = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#172538;font:14px/1.55 system-ui,'Microsoft YaHei',sans-serif}
main{max-width:1800px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 8px}h2{font-size:18px}a{color:#185ca8}header,.notes{background:white;border:1px solid #dbe3ed;border-radius:8px;padding:18px;margin-bottom:16px}
.notes li{margin:5px 0}.muted,.case{color:#58677b;font-size:12px}.case{overflow-wrap:anywhere;margin:4px 0}.group{background:white;border:1px solid #dbe3ed;border-radius:6px;margin:14px 0}.group>summary{cursor:pointer;padding:14px;font-weight:650;background:#edf3fa}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:1650px}th,td{padding:12px;border:1px solid #dbe3ed;text-align:left;vertical-align:top}th{background:#f2f6fb;font-size:13px}tbody tr:nth-child(even){background:#fafbfd}.question{min-width:140px}.load{min-width:150px}.point{min-width:230px}.conclusion{min-width:300px;max-width:400px}.credibility{min-width:240px}dl{display:grid;grid-template-columns:50px 1fr;gap:4px;margin:10px 0}dt,dd{margin:0}dd{white-space:nowrap;font-variant-numeric:tabular-nums}.status{font-size:11px;background:#eef1f5;border:1px solid #d7e0eb;border-radius:4px;padding:2px 5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;max-width:370px}button{padding:6px 12px;background:white;border:1px solid #b8c8da;border-radius:4px;cursor:pointer;margin-right:8px}.raw{overflow-wrap:anywhere;font-size:12px}
@media print{@page{size:A3 landscape;margin:8mm}body{background:white;font-size:9px}main{max-width:none;padding:0}button{display:none}.scroll{overflow:visible}table{min-width:0;table-layout:fixed}th,td{padding:4px;min-width:0!important;max-width:none!important}dd{white-space:normal}dl{display:block}dt{font-weight:bold}.case,.muted,.raw{font-size:8px}tr{break-inside:avoid}.group{break-inside:auto}thead{display:table-header-group}a{color:inherit;text-decoration:underline}}
"""


def render(run_dir, data):
    rows = data["rows"]
    groups = OrderedDict()
    unique = {}
    for row in rows:
        groups.setdefault(row.get("question_id", "未知"), []).append(row)
        for point in row.get("points", []):
            key = point.get("result") or (row.get("load", {}).get("id"), point.get("case"))
            unique[key] = point.get("status", "PENDING")
    counts = Counter(unique.values())
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    source = data.get("data_source", "summary.json")
    loads = {row.get("load", {}).get("id") for row in rows}
    title_state = "运行中快照" if counts.get("PENDING") else "结果快照"
    out = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多负载完整结果表</title><style>', CSS,
           '</style></head><body><main><header><h1>多负载架构与参数 · ' + title_state + '</h1><p>',
           f'{len(rows)} 行 · {len(groups)} 个问题 × {len(loads)} 种负载 · {len(unique)} 个去重点 · ',
           text(" / ".join(f"{k}: {v}" for k, v in sorted(counts.items()))),
           text(f"；用户取消并删除：{len(data.get('cancelled_cells', []))} 点（不计成功/失败）"),
           '</p><div class="muted">页面生成时间：', text(now), '；数据源：', link(run_dir, source, source),
           '；读取方式：', text(data.get("source_mode", "原 summary 模式")),
           '；数据快照时间：', text(data.get("snapshot_at", "原汇总未记录；不等于页面生成时间")),
           '；验证状态：', text(data.get("validation_status", "未知")),
           '。页面不会自动刷新。cells 模式只读取 manifest 声明的正式点，校验读取前后文件未变；不混入停更 summary 或额外诊断重跑。</div></header>',
           '<section class="notes"><h2>阅读口径与可信边界</h2><ul>',
           '<li>UNVALIDATED = 未做目标器件校准；绝对时间为分析仿真值，不是硬件实测或线上 p99。TTFT 为首 token 时延、TPOT 为后续每 token 时延、E2E 为端到端时延，均为请求 p50（ms）；吞吐为可见输出 token/s。</li>',
           '<li>输入/输出/请求数来自每行负载；各点按声明顺序排列，单位见问题标题。缺失或非 SIMULATED 指标显示 —，不是零；PENDING 不填观测值。缺少配置时标注无有效配置，不提供失效链接。</li>',
           '<li>公平性只限同一问题、同一负载行内按声明唯一变量比较：I1 是 KV 放置，I2/I3/I4 是互联，P1–P4 是参数扫描。不能跨 I1/P1/I4 或 HBM 参考 GPU 与 DRAM-SoC 用绝对时延排名；I3 不是 CIM 开/关实验，不能据此宣称 CIM 总体加速。</li>',
           '<li>P1/P2 使用 hbf_active_weights：MLP 权重在活动 HBF、KV 仍在 HBM0，1600 GB/s 乐观读接口不是 NAND 页模型，也不是 I1 的 KV 置 HBF 实验；I4 仅测试远端 Flash 权重路径，未测试 KV swap。</li>',
           '<li>负载是全部请求同时到达、无 MTP，不覆盖在线到达率。CIM 仅首层 MLP 冷加载，固定 8×256×256 tile 和单槽生命周期；模型是本地 GGUF 文本主干、SoC 是未校准代理参数。</li>',
           '<li>若共享与独立配置的结果全部相同，只能称未观察到端到端差异，不能断言物理架构无差异。时延/吞吐变化仅限当前负载及模型假设。</li>',
           '<li>旧 v3 的 P4 中 4 MiB scratch + 2 MiB array 超过 3 MiB 物理容量；新可执行扫描采用 256/512/1024 KiB，不把旧超容量点改称成功。以本表逐点配置为准。BLOCKED / CHECK_FAILED 不能外推为目标硬件必然不可行；HOST_TIMEOUT 仅是宿主限时。</li>',
           '<li>新矩阵采用 eager KV 准入：先预留该请求生成全过程的空间，容量不足的请求等待；不减少提交请求、context 或 output。排队计入 TTFT/E2E，同一行调度策略相同，不能与旧 lazy/no-preemption 结果混比。</li>',
           '<li>数据量是多资源累计 bytes（同一传输可能跨资源重复计入），不是显存峰值、模型大小或唯一数据量。</li>',
           '</ul><p>', link(run_dir, "manifest.json", "运行清单"), ' · ', link(run_dir, "model_config.json", "模型配置"),
           '</p><button onclick="document.querySelectorAll(\'.group\').forEach(x=>x.open=true)">展开全部</button>',
           '<button onclick="document.querySelectorAll(\'.group\').forEach(x=>x.open=false)">折叠全部</button>',
           '<button onclick="window.print()">打印</button></section>']
    if data.get("model"):
        model = data["model"]
        out.append('<p class="muted">模型身份：' + text(model.get("gguf_path", "未知")) +
                   '；GGUF 架构：' + text(model.get("architecture", "未知")) +
                   '；量化标签：' + text(model.get("quantization_label", "未知")) + '（以实际 tensor 类型为准）；完整限制见运行清单。</p>')
    for question_id, members in groups.items():
        width = max((len(r.get("points", [])) for r in members), default=0)
        title = members[0].get("question", question_id)
        unit = {"hbf_latency": "ns", "dram_latency": "ns", "hbf_outstanding": "请求数", "scratch": "bytes"}.get(members[0].get("kind"), "对照配置")
        out.append('<details class="group" open><summary>' + text(question_id) + ' · ' + text(title) + f'（{len(members)} 行；参数：' + text(unit) + '）</summary><div class="scroll"><table><thead><tr><th scope="col">问题</th><th scope="col">负载 / 输入文件</th>')
        out.extend('<th scope="col">点 ' + str(i + 1) + '</th>' for i in range(width))
        out.append('<th scope="col">定性结论</th><th scope="col">可比性 / 证据边界</th></tr></thead><tbody>')
        for row in members:
            load = row.get("load", {})
            out.append('<tr class="result-row"><td class="question">' + text(question_id) + '<br>' + text(row.get("question", "")) + '</td><td class="load"><strong>' + text(load.get("id", "—")) + '</strong><br>输入：' + number(load.get("prompt_tokens")) + ' tokens<br>输出：' + number(load.get("output_tokens")) + ' tokens<br>请求数：' + number(load.get("batch")) + '<p class="muted">各点输入配置链接见右侧；公共模型配置见页首。</p></td>')
            for point in row.get("points", []):
                out.append('<td class="point">' + point_html(run_dir, point, load) + '</td>')
            out.extend('<td>—</td>' for _ in range(width - len(row.get("points", []))))
            out.append('<td class="conclusion">' + text(conclusion(row)) + '</td><td class="credibility"><strong class="raw">' + text(row.get("credibility", "未知")) + '</strong><p>全点可比：' + ("是" if row.get("comparable") else "否") + '<br>成功且输入齐全的点可比：' + ("是" if row.get("feasible_points_comparable") else "否") + '<br>固定映射：' + ("是" if row.get("fixed_mapping") else "否") + '</p><p>未校准；检查通过不等于硬件预测可信，弱单调通过也不保证参数敏感性。</p>')
            audit = {k: row.get(k) for k in ("compared_cases", "comparison_errors", "unexpected_changes", "actual_changed_paths", "trend_check")}
            out.append('<details><summary>可比性与趋势原始检查</summary><pre>' + text(json.dumps(audit, ensure_ascii=False, indent=2)) + '</pre></details></td></tr>')
        out.append('</tbody></table></div></details>')
    out.append('</main><script>let printState=[];window.addEventListener("beforeprint",()=>{printState=[...document.querySelectorAll(".group")].map(x=>[x,x.open]);printState.forEach(([x])=>x.open=true)});window.addEventListener("afterprint",()=>printState.forEach(([x,open])=>x.open=open));</script></body></html>')
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="从 summary.json 生成同目录中文 results.html；不运行仿真、不修改输入。")
    parser.add_argument("run_dir", nargs="?", type=Path, help="运行目录（也可使用 --run-dir）")
    parser.add_argument("--run-dir", dest="run_dir_option", type=Path)
    parser.add_argument("--from-cells", action="store_true", help="只读正式 manifest/cells/configs，另存 report_summary.json；不运行仿真")
    args = parser.parse_args()
    run_dir = args.run_dir_option or args.run_dir
    if run_dir is None or (args.run_dir_option and args.run_dir):
        parser.error("请提供一个 run-dir")
    try:
        if args.from_cells:
            data = summary_from_cells(run_dir)
            (run_dir / "report_summary.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        else:
            data = read_summary(run_dir / "summary.json")
        output = run_dir / "results.html"
        output.write_text(render(run_dir, data), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, "无法生成结果表（原始 JSON 未修改）：" + str(exc) + "\n")
    print(str(output.resolve()))
    print(f"{len(data['rows'])} rows rendered")


if __name__ == "__main__":
    main()
