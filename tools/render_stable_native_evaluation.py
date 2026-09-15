"""Render a frozen stable-native evaluation, without importing prediction code.

Usage: python tools/render_stable_native_evaluation.py --sim-dir SIM_DIRECTORY
Optional: --native-report NATIVE_162_REPORT.json --output REPORT_DIRECTORY
Reads bounded JSON evidence only. Writes report.md, report.html and three SVGs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
from xml.sax.saxutils import escape as xe

GROUPS = ("qwen25", "qwen35", "qwen38", "smollm2", "tinyllama", "qwen38_gpu")
LABELS = {"qwen25": "Qwen2.5", "qwen35": "Qwen3.5", "qwen38": "Qwen3.8 27B · CPU", "smollm2": "SmolLM2", "tinyllama": "TinyLlama", "qwen38_gpu": "Qwen3.8 27B · GPU"}
PROMPTS, OUTPUTS, PARALLELS = (128, 512, 1536), (32, 128, 256), (1, 2, 4)
GRID = tuple(itertools.product(PROMPTS, OUTPUTS, PARALLELS))
METRICS = {"ttft": "engine_ttft_ms", "tpot": "engine_tpot_ms", "e2e": "engine_e2e_ms"}
NAMES = {"ttft": "TTFT 首字延迟", "tpot": "TPOT 每输出 token 耗时", "e2e": "E2E 请求总耗时"}
FIELDS = ("native_median_ms", "simulator_median_ms", "signed_error_ms", "absolute_error_ms", "signed_error_pct", "absolute_percentage_error_pct")
STATES = {"scored": "已评分", "excluded": "未选择", "failed": "预测失败", "pending": "待预测", "unscored": "未评分", "unfrozen": "未冻结", "missing": "缺少原格"}
SHORT = {"scored": "已评分", "excluded": "未选", "failed": "失败", "pending": "待算", "unscored": "未评", "unfrozen": "未冻", "missing": "缺格"}
BINS = ((5, "#d6eee4", "#164d3c", "0–5%"), (10, "#cce5f5", "#143e5a", "5–10%"), (25, "#fff0c4", "#664d12", "10–25%"), (50, "#ffd3a2", "#743b0d", "25–50%"), (100, "#f4aaa3", "#6e211b", "50–100%"), (math.inf, "#c3a1d5", "#3f2155", ">100%"))
ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 64 * 1024 * 1024


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_json(path):
    """Hash only the already-read, bounded JSON, never model/runtime references."""
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() != ".json" or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("仅接受不超过 64 MiB 的 JSON 证据文件: " + str(path))
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须为对象: " + str(path))
    return data, {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def resolve_ref(ref, bases):
    value = ref.get("path") if isinstance(ref, dict) else ref
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    choices = (candidate,) if candidate.is_absolute() else tuple(base / candidate for base in bases)
    return next((p.resolve() for p in choices if p.is_file()), None)


def keyed(rows, label):
    result = {}
    for row in rows or []:
        if not isinstance(row, dict):
            raise ValueError(label + " 含非对象条目")
        ident = row.get("cell_id", row.get("id"))
        if not isinstance(ident, str) or not ident or ident in result:
            raise ValueError(label + " 缺少或重复 cell_id: " + str(ident))
        result[ident] = row
    return result


def coordinate(row):
    ident = str(row.get("cell_id", row.get("id", "")))
    config = row.get("config") or (row.get("static_inputs") or {}).get("config") or {}
    group = row.get("model_key")
    if group not in GROUPS:
        group = next((g for g in sorted(GROUPS, key=len, reverse=True) if ident.startswith(g + "_")), None)
    p = row.get("prompt_tokens", config.get("expected_prompt_tokens", config.get("prompt_tokens")))
    o = row.get("output_tokens", config.get("output", config.get("output_tokens")))
    c = row.get("parallel", config.get("parallel"))
    match = re.search(r"(?:^|_)p(\d+)_o(\d+)_c(\d+)(?:_|$)", ident)
    if match and any(v is None for v in (p, o, c)):
        p, o, c = map(int, match.groups())
    if group not in GROUPS or any(type(v) is not int for v in (p, o, c)) or (p, o, c) not in GRID:
        raise ValueError("无法映射到六组 162 格: " + ident)
    return group, p, o, c


def reasons(row):
    result = []
    for key in ("selection_exclusion_reasons", "reasons", "metadata_errors", "actuals_errors"):
        if isinstance(row.get(key), list):
            result.extend(str(v) for v in row[key] if v)
    return list(dict.fromkeys(result))


def reason_text(reason):
    known = {"native_incomplete_or_failed": "原生执行未完成或失败", "evidence_or_freeze_not_verified": "原生证据或冻结未验证", "diagnostic_only": "仅供诊断", "prediction unavailable or incomplete": "预测尚不可用或不完整", "complete_verified_actuals_unavailable": "缺少已完成且验证过的原生测量", "gpu_planned_cell_missing": "GPU 原定场景缺失", "gpu_freeze_missing": "GPU 冻结记录缺失"}
    label = known.get(reason)
    if label is None and "not_strictly_below_5" in reason:
        label = "原生波动未严格小于 5%"
    if label is None and ("coverage" in reason or "sample" in reason):
        label = "覆盖或样本证据不足"
    return label + " [" + reason + "]" if label else reason


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    return {"median": statistics.median(values) if values else None, "p90": percentile(values, .9), "worst": max(values) if values else None, "min": min(values) if values else None}


def fmt(value, digits=2, signed=False):
    return "—" if value is None else format(value, ("+" if signed else "") + "." + str(digits) + "f")


def metric_record(raw):
    if not isinstance(raw, dict) or raw.get("status") != "scored":
        return None, str(raw.get("reason", "现有评分文件没有此指标")) if isinstance(raw, dict) else "现有评分文件没有此指标"
    if any(not number(raw.get(k)) for k in FIELDS):
        return None, "评分含缺失或非有限数值"
    if raw["native_median_ms"] <= 0 or raw["simulator_median_ms"] <= 0:
        return None, "评分的原生或模拟中位数不为正"
    delta = raw["simulator_median_ms"] - raw["native_median_ms"]
    check = {"signed_error_ms": delta, "absolute_error_ms": abs(delta), "signed_error_pct": 100 * delta / raw["native_median_ms"], "absolute_percentage_error_pct": 100 * abs(delta) / raw["native_median_ms"]}
    if any(not math.isclose(raw[k], v, rel_tol=1e-8, abs_tol=1e-8) for k, v in check.items()):
        return None, "现有评分字段互相矛盾；未重算或修正原记录"
    runs = raw.get("native_run_medians_ms")
    if not isinstance(runs, list) or len(runs) != 3 or any(not number(v) or v <= 0 for v in runs):
        return None, "评分缺少三次原生运行中位数"
    if not math.isclose(statistics.median(runs), raw["native_median_ms"], rel_tol=1e-8, abs_tol=1e-8):
        return None, "三次原生运行值与原生中位数不一致"
    result = {k: float(raw[k]) for k in FIELDS}
    result["native_run_medians_ms"] = list(runs)
    result["native_run_worst_abs_pct"] = max(abs(v - raw["native_median_ms"]) / raw["native_median_ms"] * 100 for v in runs)
    result["run_comparisons"] = [{"run": i + 1, "native_ms": v, "signed_ms": raw["simulator_median_ms"] - v,
        "signed_pct": 100 * (raw["simulator_median_ms"] - v) / v,
        "ape_pct": 100 * abs(raw["simulator_median_ms"] - v) / v} for i, v in enumerate(runs)]
    return result, None


def clock_exception_note(selection, native):
    """Describe explicitly marked replacement captures, never all 162 cells."""
    cell_ids, observed, lineages = set(), [], []
    present, approved = False, False
    for document in (selection, native):
        if not isinstance(document, dict):
            continue
        for key in ("clock_exception_lineage", "clock_supplement", "clock_supplement_lineage"):
            value = document.get(key)
            if value:
                present = True
                if isinstance(value, dict):
                    lineages.append(value)
                    approved = approved or bool(value.get("user_authorization"))
                    for item in value.get("cells", []):
                        if isinstance(item, dict):
                            if item.get("cell_id"):
                                cell_ids.add(str(item["cell_id"]))
                            observed.append(item)
        for key in ("selected_cells", "excluded_cells", "cells"):
            for row in document.get(key, []) or []:
                if not isinstance(row, dict):
                    continue
                source = row.get("source_per_cell") or {}
                exception = row.get("clock_exception") or {}
                marked = (source.get("source_id") == "clock_exception_supplement"
                          or source.get("replacement_reason") == "user_approved_clock_tolerance_exception"
                          or bool(exception))
                if marked:
                    present = True
                    approved = approved or exception.get("approved") is True
                    ident = row.get("cell_id", row.get("id"))
                    if ident:
                        cell_ids.add(str(ident))
                    observed.append(exception)
        for ident, source in (document.get("source_per_cell") or {}).items():
            if isinstance(source, dict) and (source.get("source_id") == "clock_exception_supplement"
                    or source.get("replacement_reason") == "user_approved_clock_tolerance_exception"):
                present = True
                cell_ids.add(str(ident))
    if not present:
        return None
    scope = str(len(cell_ids)) + " 格" if cell_ids else "标记场景"
    if cell_ids and all(ident.startswith("smollm2_") for ident in cell_ids):
        scope += " SmolLM2 场景"
    triples = set()
    for item in observed:
        target = item.get("target_mhz", item.get("expected_gpu_sm_clock_mhz"))
        tolerance = item.get("tolerance_mhz", item.get("exception_tolerance_mhz"))
        original = item.get("original_tolerance_mhz")
        if all(number(v) and v > 0 for v in (target, tolerance, original)):
            triples.add((target, tolerance, original))
    text = scope + ("为用户授权频率例外下的新采集。" if approved else "有频率例外补测来源记录。")
    if len(triples) == 1:
        target, tolerance, original = next(iter(triples))
        text += f"仅这些格采用 {target:g} ± {tolerance:g} MHz，原容差为 ±{original:g} MHz；其他格保留各自原冻结门槛。"
    else:
        text += "例外仅适用于这些标记格，具体容差见补测来源；其他格保留各自原冻结门槛。"
    text += "这不表示原 freeze 的硬件门禁通过；原 162 格分母与旧失败历史均保留。"
    if any(value.get("status") == "invalid" or value.get("verification_errors") for value in lineages):
        text += "例外来源仍有校验错误，不能据此认定补测有效。"
    return {"text": text, "cell_ids": sorted(cell_ids)}


def load_evaluation(sim_dir, errors_path=None, native_path=None, selection_path=None):
    sim_dir = Path(sim_dir).resolve(strict=True)
    freeze, freeze_ref = read_json(sim_dir / "freeze.json")
    if freeze.get("schema") != "stable-native-simulation-freeze/v1":
        raise ValueError("不支持的 freeze schema")
    warnings, sources = [], {"freeze": freeze_ref}
    if errors_path is None:
        candidates = [(int(m.group(1)), p) for p in sim_dir.glob("errors.*.json") if (m := re.fullmatch(r"errors\.(\d+)\.json", p.name))]
        errors_path = max(candidates, default=(0, None), key=lambda item: item[0])[1]
    errors = {}
    if errors_path is not None:
        errors, sources["errors"] = read_json(errors_path)
        if errors.get("schema") != "stable-native-simulation-errors/v1":
            raise ValueError("不支持的 errors schema")
        if errors.get("freeze_ref", {}).get("sha256") != freeze_ref["sha256"]:
            raise ValueError("errors 与当前 freeze 不一致；请选择匹配的评分快照")
    else:
        warnings.append("尚无 errors.NNNN.json：保留完整网格，所有指标显示未评分。")
    bases = (sim_dir, ROOT, Path(freeze.get("data_root") or ROOT))
    selection_path = selection_path or resolve_ref(freeze.get("selection_ref"), bases)
    if selection_path is None:
        raise ValueError("无法读取冻结选择；请用 --selection 指向原选择 JSON")
    selection, selection_ref = read_json(selection_path)
    if selection.get("schema") != "native-stable-dataset/v1":
        raise ValueError("选择文件必须为 native-stable-dataset/v1")
    if selection_ref["sha256"] != freeze.get("selection_sha256"):
        raise ValueError("选择文件与 freeze 中保存的 SHA 不一致")
    sources["selection"] = selection_ref
    native_path = native_path or resolve_ref(errors.get("native_report_ref"), bases)
    native = {}
    if native_path is not None and Path(native_path).resolve() != Path(selection_ref["path"]):
        native, sources["native_report"] = read_json(native_path)
        if native.get("schema") not in ("native-long-grid-162-screen/v1", "native-stable-dataset/v1"):
            raise ValueError("--native-report 应为完整 162 格报告或稳定选择 JSON")
        if native.get("selection_payload_sha256") and native["selection_payload_sha256"] != selection.get("payload_sha256"):
            raise ValueError("native report 绑定的稳定选择与当前选择不一致")
    for label, doc, key in (("选择", selection, "planned_cells"), ("冻结", freeze, "native_grid_denominator")):
        if doc.get(key, 162) != 162:
            raise ValueError(label + "的原格分母不是 162")
    selected, excluded = keyed(selection.get("selected_cells"), "选择"), keyed(selection.get("excluded_cells"), "排除")
    if set(selected) & set(excluded):
        raise ValueError("同一格同时出现在选择与排除清单")
    if selection.get("selected_count", len(selected)) != len(selected):
        raise ValueError("selected_count 与选择清单不一致")
    entries, scored = keyed(freeze.get("cells"), "冻结"), keyed(errors.get("cells"), "评分")
    if set(entries) != set(selected):
        warnings.append("冻结集合与原生稳定入选集合不完全一致；未冻结或缺失格单独列出，不缩小原始分母。")
    if scored and set(scored) != set(entries):
        warnings.append("评分快照未覆盖全部冻结格，或含额外格；仅呈现 162 格内、已冻结且入选的评分。")
    native_rows = keyed(native.get("cells", native.get("selected_cells", [])), "原生报告")
    by_coord = {}
    for mapping in (excluded, selected, native_rows):
        for ident, row in mapping.items():
            point = coordinate(row)
            previous = by_coord.get(point)
            if previous is not None and previous["cell_id"] != ident:
                raise ValueError("同一原始坐标有不同 cell_id: " + ident)
            by_coord[point] = {**(previous or {}), **row, "cell_id": ident}
    if len(by_coord) != 162:
        warnings.append(f"原生证据清单覆盖 {len(by_coord)}/162 格；其余原定坐标保留为灰色缺格。")
    coverage = native.get("coverage") or selection.get("coverage") or freeze.get("selector_coverage") or freeze.get("coverage") or []
    if isinstance(coverage, dict):
        coverage = list(coverage.values())
    coverage = {item.get("model_key"): item for item in coverage if isinstance(item, dict)}
    cells, unsupported = [], Counter()
    for group in GROUPS:
        for p, o, c in GRID:
            row = by_coord.get((group, p, o, c))
            ident = row["cell_id"] if row else f"{group}_p{p}_o{o}_c{c}__missing"
            cell = {"cell_id": ident, "group": group, "prompt": p, "output": o, "parallel": c, "selected": ident in selected, "frozen": ident in entries, "native": row, "prediction_status": "not_requested", "state": "excluded", "reasons": reasons(row or {}), "metrics": {}, "metric_reasons": {}}
            if row is None:
                cell.update(state="missing", reasons=["原生选择和报告均缺少此原定坐标"])
            elif ident not in selected:
                if not cell["reasons"]:
                    cell["reasons"] = ["未通过原生稳定性选择；来源未给出详细原因"]
            elif ident not in entries:
                cell.update(state="unfrozen", reasons=["原生稳定入选，但未纳入本次模拟冻结"])
            else:
                entry = entries[ident]
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", ident):
                    raise ValueError("不安全的预测 cell_id")
                pred_path = sim_dir / "predictions" / (ident + ".prediction.json")
                prediction, pred_ref = {}, None
                try:
                    if pred_path.is_file():
                        prediction, pred_ref = read_json(pred_path)
                    if prediction and (prediction.get("cell_id") != ident or prediction.get("freeze_ref", {}).get("sha256") != freeze_ref["sha256"]):
                        raise ValueError("预测身份与本次冻结不一致")
                    cell["prediction_status"] = prediction.get("status", "pending")
                    for item in prediction.get("unsupported_dimensions", []):
                        unsupported[(str(item.get("dimension", "unknown")), str(item.get("status", "conditional")), str(item.get("reason", "")))] += 1
                    if entry.get("preparation_error"):
                        cell.update(state="failed", reasons=[str(entry["preparation_error"])])
                    elif not prediction or prediction.get("status") == "pending":
                        cell.update(state="pending", reasons=["尚无完成的预测文件"])
                    elif prediction.get("status") != "predicted":
                        cell.update(state="failed", reasons=[str(prediction.get("reason") or "预测记录未成功完成")])
                    else:
                        score = scored.get(ident, {})
                        if score and (score.get("prediction_ref") or {}).get("sha256") != pred_ref["sha256"]:
                            cell.update(state="unscored", reasons=["评分对应另一份预测快照；请重新评分后呈现"])
                        else:
                            for alias, metric in METRICS.items():
                                record, why = metric_record(score.get("metrics", {}).get(metric, score.get("metrics", {}).get(alias)))
                                if record is None:
                                    cell["metric_reasons"][alias] = why
                                else:
                                    cell["metrics"][alias] = record
                            cell["state"] = "scored" if len(cell["metrics"]) == 3 else "unscored"
                            cell["reasons"] = [NAMES[m] + "：" + why for m, why in cell["metric_reasons"].items()]
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    cell.update(state="failed", prediction_status="invalid", reasons=["预测证据无法使用：" + str(exc)])
            cells.append(cell)
    stats = []
    for group in GROUPS:
        group_cells = [cell for cell in cells if cell["group"] == group]
        detailed = [cell["native"] for cell in group_cells if cell["native"] is not None and "evidence_verified" in cell["native"]]
        valid = coverage.get(group, {}).get("verified_cells")
        if type(valid) is not int or not 0 <= valid <= 27:
            valid = sum(r.get("evidence_verified") is True and not r.get("metadata_errors") and not r.get("actuals_errors") for r in detailed) if len(detailed) == 27 else None
        chosen = sum(cell["selected"] for cell in group_cells)
        declared = coverage.get(group, {}).get("selected_cells")
        if declared is not None and declared != chosen:
            warnings.append(f"{group} 的 coverage 入选数 {declared} 与选择清单 {chosen} 不一致；使用选择清单。")
        stats.append({"group": group, "native_valid": valid, "selected": chosen, "frozen": sum(cell["frozen"] for cell in group_cells), "predicted": sum(cell["prediction_status"] == "predicted" for cell in group_cells), "scored": sum(len(cell["metrics"]) == 3 for cell in group_cells), "failed": sum(cell["state"] == "failed" for cell in group_cells), "pending": sum(cell["state"] in ("pending", "unfrozen", "unscored") for cell in group_cells), "excluded": sum(not cell["selected"] for cell in group_cells)})
    attempts = native.get("failed_attempts", selection.get("failed_attempts", []))
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "sim_dir": str(sim_dir), "sources": sources, "score_created_utc": errors.get("created_utc"), "selection_created_utc": selection.get("created_utc"), "cells": cells, "groups": stats, "selected_count": len(selected), "warnings": list(dict.fromkeys(warnings)), "unsupported": unsupported, "failed_attempts": attempts if isinstance(attempts, list) else [], "hardware_scope_note": clock_exception_note(selection, native)}



def summaries(report, group=None):
    cells = [c for c in report["cells"] if group is None or c["group"] == group]
    result = []
    for metric in METRICS:
        valid = [(c, c["metrics"][metric]) for c in cells if metric in c["metrics"]]
        row = {"metric": metric, "count": len(valid), "selected": sum(c["selected"] for c in cells), "planned": len(cells)}
        for key in ("absolute_percentage_error_pct", "absolute_error_ms", "signed_error_pct", "signed_error_ms"):
            row[key] = distribution([v[key] for _, v in valid])
        row["worst_cell"] = max(valid, key=lambda pair: pair[1]["absolute_percentage_error_pct"])[0]["cell_id"] if valid else None
        result.append(row)
    return result


def heat_number(value, signed=False):
    if abs(value) < 1000:
        return fmt(value, 1, signed)
    return format(value, "+.1e" if signed else ".1e").replace("e+0", "e").replace("e+", "e")


def color(ape):
    for limit, bg, fg, _ in BINS:
        if ape <= limit:
            return bg, fg
    return BINS[-1][1:3]


def metric_reason(cell, metric):
    return cell["metric_reasons"].get(metric) or "；".join(reason_text(r) for r in cell["reasons"]) or "未评分"


def heatmap_svg(report, metric):
    title = NAMES[metric] + " · 162 格绝对百分比误差"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1320" viewBox="0 0 1280 1320" role="img" aria-labelledby="title-{metric} desc-{metric}">',
        f'<title id="title-{metric}">{xe(title)}</title>',
        f'<desc id="desc-{metric}">六组各27格，原135格加GPU27格。底色为APE，第二行为有符号百分比误差。灰格保留未选、失败或缺失原因。模拟确定性单次预测，原生三次正式测量用于初筛。</desc>',
        '<rect width="100%" height="100%" fill="#f5f7fb"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif">',
        f'<text x="36" y="44" font-size="25" font-weight="700" fill="#152a43">{xe(title)}</text>',
        '<text x="36" y="75" font-size="15" fill="#4d6075">按原生稳定性选择后的开发评估；不是盲测或泛化验收。分母始终为 162 格。</text>',
        '<text x="36" y="100" font-size="14" fill="#4d6075">色阶 = APE；Δ = (模拟 − 原生) / 原生。正值为高估，负值为低估。悬停查看数值与原因。</text>']
    for i, (_, bg, _, label) in enumerate(BINS):
        x = 36 + i * 152
        out += [f'<rect x="{x}" y="121" width="22" height="19" rx="3" fill="{bg}"/>', f'<text x="{x+31}" y="136" font-size="13" fill="#33465c">{label}</text>']
    out += ['<rect x="970" y="121" width="22" height="19" rx="3" fill="#e4e8ee"/>', '<text x="1000" y="136" font-size="13" fill="#33465c">灰：未选 / 失败 / 待算 / 未评</text>']
    lookup = {(c["group"], c["prompt"], c["output"], c["parallel"]): c for c in report["cells"]}
    stats = {s["group"]: s for s in report["groups"]}
    for i, group in enumerate(GROUPS):
        x, y = 26 + i % 2 * 622, 168 + i // 2 * 365
        stat = stats[group]
        count = sum(metric in lookup[(group, *point)]["metrics"] for point in GRID)
        out += [f'<rect x="{x}" y="{y}" width="606" height="345" rx="14" fill="white" stroke="#dfe5ec"/>',
            f'<text x="{x+19}" y="{y+29}" font-size="18" font-weight="700" fill="#152a43">{xe(LABELS[group])}</text>',
            f'<text x="{x+19}" y="{y+52}" font-size="12" fill="#53657a">有效原生 {stat["native_valid"] if stat["native_valid"] is not None else "未知"}/27 · 入选 {stat["selected"]}/27 · 本指标已评分 {count}/27</text>']
        gx, gy, cw, ch = x + 76, y + 110, 56, 64
        for oi, output in enumerate(OUTPUTS):
            out.append(f'<text x="{gx+(oi*3+1.5)*cw}" y="{y+78}" text-anchor="middle" font-size="13" fill="#33465c">输出 {output}</text>')
            for ci, parallel in enumerate(PARALLELS):
                out.append(f'<text x="{gx+(oi*3+ci+.5)*cw}" y="{y+100}" text-anchor="middle" font-size="11" fill="#53657a">并发 {parallel}</text>')
        for pi, prompt in enumerate(PROMPTS):
            out.append(f'<text x="{x+14}" y="{gy+pi*ch+29}" font-size="12" fill="#33465c">输入 {prompt}</text>')
            for oi, output in enumerate(OUTPUTS):
                for ci, parallel in enumerate(PARALLELS):
                    cell = lookup[(group, prompt, output, parallel)]
                    rec = cell["metrics"].get(metric)
                    left, top = gx + (oi*3+ci)*cw, gy + pi*ch
                    if rec:
                        bg, fg = color(rec["absolute_percentage_error_pct"])
                        a, b = heat_number(rec["absolute_percentage_error_pct"])+"%", "Δ "+heat_number(rec["signed_error_pct"], True)+"%"
                        detail = (f'{cell["cell_id"]} | 原生 {rec["native_median_ms"]:.6g} ms；模拟 {rec["simulator_median_ms"]:.6g} ms；APE {rec["absolute_percentage_error_pct"]:.6g}%；Δ {rec["signed_error_ms"]:+.6g} ms；'
                                  + '三次原生 ms: '+', '.join(fmt(v, 4) for v in rec["native_run_medians_ms"]))
                    else:
                        bg, fg = "#e4e8ee", "#566371"
                        state = cell["state"] if cell["state"] != "scored" else "unscored"
                        a, b = SHORT[state], "—"
                        detail = cell["cell_id"]+" | "+metric_reason(cell, metric)
                    out += [f'<g><title>{xe(detail)}</title><rect x="{left+1}" y="{top+1}" width="{cw-2}" height="{ch-2}" rx="5" fill="{bg}"/>',
                        f'<text x="{left+cw/2}" y="{top+26}" text-anchor="middle" font-size="12" font-weight="700" fill="{fg}">{xe(a)}</text>',
                        f'<text x="{left+cw/2}" y="{top+47}" text-anchor="middle" font-size="10" fill="{fg}">{xe(b)}</text></g>']
        footer = "零入选：不能评估本组误差。" if not stat["selected"] else "全部原定格保留；三次原生值及逐run误差见报告明细。"
        out.append(f'<text x="{x+19}" y="{y+328}" font-size="12" fill="#53657a">{footer}</text>')
    out.append(f'<text x="36" y="1290" font-size="12" fill="#53657a">评分快照：{xe(str(report["score_created_utc"] or "尚无"))} · 各格等权；灰格不补零。</text></g></svg>')
    return "\n".join(out)+"\n"


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def summary_table(report, group=None, markdown=False):
    headers = ["指标", "已评分 / 入选 / 原格", "APE 中位 / P90 / 最坏 (%)", "绝对误差 中位 / P90 / 最坏 (ms)", "Δ中位 (%)", "Δ中位 (ms)", "Δ范围 (%)"]
    rows = []
    for item in summaries(report, group):
        ape, absolute = item["absolute_percentage_error_pct"], item["absolute_error_ms"]
        pct, delta = item["signed_error_pct"], item["signed_error_ms"]
        rows.append([item["metric"].upper(), f'{item["count"]} / {item["selected"]} / {item["planned"]}', ' / '.join(fmt(ape[k]) for k in ("median", "p90", "worst")), ' / '.join(fmt(absolute[k]) for k in ("median", "p90", "worst")), fmt(pct["median"], signed=True), fmt(delta["median"], signed=True), fmt(pct["min"], signed=True)+" 至 "+fmt(pct["worst"], signed=True)])
    if markdown:
        return ['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |', *['| '+' | '.join(md(v) for v in row)+' |' for row in rows]]
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def method_text():
    return ["这是按原生稳定性选择后的开发评估，不是盲测，也不构成正式精度、跨进程重复性或泛化验收。选择只看原生 Engine 波动与有效覆盖，没有按模拟误差挑选格。",
        "完整分母固定为原始 135 格加 27B GPU 版 27 格，共六组 162 格。每组保留 3 种输入 × 3 种输出 × 3 种并发；零入选组不能评估误差。",
        "原生三次正式测量用于稳定性初筛；模拟为确定性单次预测。并发场景在每次原生运行内取请求中位数，再对三次run中位数取中位数；模拟侧取本次预测内请求中位数。",
        "稳定入选门槛为三项指标的 batch 与请求序位最大绝对偏差均严格小于 5%，且证据完整有效。下方另列的三次run中位数最大偏差只是辅助展示，不替代选择器全部稳定性条件。",
        "沿用 errors 的评分：有符号误差 Δ = 模拟 − 原生；APE = |Δ| / 原生 × 100%。逐run误差将同一个固定模拟值分别与原生三次值比较；R1/R2/R3 沿用评分文件保存的顺序。",
        "统计按格等权，只对有效评分求中位数、线性插值 P90 和最大值。灰格不补零，每项统计公开已评分、稳定入选和原格分母；仅成功样本的条件误差不能代表全部 162 格。",
        "本工具只读已有 JSON 与小型预测记录，核对记录身份及评分一致性；不读取模型、不运行预测、不修改原评分。机制限制取自预测记录，原生入选稳定不代表所有模拟机制已匹配。"]


def coverage_values(item):
    return [LABELS[item["group"]], 27, item["native_valid"] if item["native_valid"] is not None else "未知", item["selected"], item["frozen"], item["predicted"], item["scored"], item["excluded"], item["failed"], item["pending"]]


COVERAGE_HEADERS = ("组别", "原格", "有效原生", "稳定入选", "本次冻结", "预测成功", "三项已评分", "未选择", "预测失败", "待完成/未评")


def render_markdown(report):
    lines = ['# 原生稳定样本开发评估', '', f'生成：{report["created_utc"]}；评分：{report["score_created_utc"] or "尚无"}', '']
    lines += [text+'\n' for text in method_text()]
    if report.get('hardware_scope_note'):
        note = report['hardware_scope_note']
        lines += ['> **硬件口径：** '+md(note['text']), '']
        if note['cell_ids']:
            lines += ['涉及场景：'+', '.join('`'+md(ident)+'`' for ident in note['cell_ids']), '']
    lines += ['## 完整覆盖', '', '| '+' | '.join(COVERAGE_HEADERS)+' |', '| '+' | '.join('---' for _ in COVERAGE_HEADERS)+' |']
    for item in report["groups"]:
        lines.append('| '+' | '.join(md(v) for v in coverage_values(item))+' |')
    lines += ['', f'原格 162；入选 {report["selected_count"]}；未选择 {162-report["selected_count"]}；保留历史失败尝试 {len(report["failed_attempts"])} 条。有效原生数优先引用 coverage.verified_cells；未知值不补零。', '']
    if report["warnings"]:
        lines += ['## 数据状态', '', *['- '+md(w) for w in report["warnings"]], '']
    lines += ['## 全体已评分格的条件统计', '', *summary_table(report, markdown=True), '']
    for metric in METRICS:
        lines += [f'![{NAMES[metric]}完整162格热图](heatmap_{metric}.svg)', '']
    lines += ['## 六组误差', '']
    for item in report["groups"]:
        group = item["group"]
        lines += ['### '+LABELS[group], '']
        if item["selected"] == 0:
            lines += ['**零入选，不能评估本组误差。27 个原定格仍在热图及明细中。**', '']
        lines += [*summary_table(report, group, markdown=True), '']
        lines += [f'- {m["metric"].upper()} 最坏 APE 所在格：`{m["worst_cell"]}`' for m in summaries(report, group) if m["worst_cell"]]
        lines.append('')
    lines += ['## 全部 162 格与排除原因', '', '单格依次为 APE、Δ%、|Δ|ms；三次原生值和逐run误差见下一节。', '', '| 组别 / Cell ID | 输入 / 输出 / 并发 | 状态 | TTFT | TPOT | E2E | 原因 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for cell in report["cells"]:
        values = []
        for metric in METRICS:
            r = cell["metrics"].get(metric)
            values.append(fmt(r["absolute_percentage_error_pct"])+"%; Δ"+fmt(r["signed_error_pct"], signed=True)+"%; "+fmt(r["absolute_error_ms"])+"ms" if r else '—')
        why = '；'.join(reason_text(r) for r in cell["reasons"]) or '三项评分可用'
        fields = [LABELS[cell["group"]]+' / '+cell["cell_id"], f'{cell["prompt"]} / {cell["output"]} / {cell["parallel"]}', STATES[cell["state"]], *values, why]
        lines.append('| '+' | '.join(md(v) for v in fields)+' |')
    lines += ['', '## 固定模拟预测对三次原生运行', '', '顺序均为评分文件中的 R1 / R2 / R3。原生波动是三次run中位数相对其中位数的最大绝对偏差。', '', '| Cell ID | 指标 | 固定模拟 ms | 三次原生 ms | 三次 APE % | 三次 Δ% | 三次 Δms | 原生run波动 % |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for cell in report["cells"]:
        for metric, r in cell["metrics"].items():
            comps = r['run_comparisons']
            values = [cell['cell_id'], metric.upper(), fmt(r['simulator_median_ms']), ' / '.join(fmt(v) for v in r['native_run_medians_ms']), ' / '.join(fmt(v['ape_pct']) for v in comps), ' / '.join(fmt(v['signed_pct'], signed=True) for v in comps), ' / '.join(fmt(v['signed_ms'], signed=True) for v in comps), fmt(r['native_run_worst_abs_pct'])]
            lines.append('| '+' | '.join(md(v) for v in values)+' |')
    lines += ['', '## 机制覆盖与来源', '']
    for (dimension, status, reason), count in sorted(report['unsupported'].items()):
        lines.append(f'- `{md(dimension)}` / `{md(status)}`：{count} 格；{md(reason)}')
    if not report['unsupported']:
        lines.append('尚无机制覆盖说明；不据此推定全部机制已匹配。')
    lines += ['', '完整失败尝试历史保留在原生报告或选择文件的 failed_attempts 字段。', '']
    for name, ref in report['sources'].items():
        lines.append(f'- {name}：`{ref["path"]}`；JSON SHA256：`{ref["sha256"]}`')
    return '\n'.join(lines)+'\n'


def render_html(report, svgs):
    esc = html.escape
    out = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>162格原生稳定样本开发评估</title><style>
:root{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#152a43;background:#f3f6fa;font-size:15px}*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:32px}h1{font-size:32px;margin:0 0 12px}h2{margin:34px 0 16px;font-size:23px}p{line-height:1.75}header{background:#152a43;color:white;padding:30px;border-radius:18px}.badge{color:#bcdce7;font-size:13px;letter-spacing:.07em}.muted{color:#b5c8db}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:20px 0}.card,.panel{background:white;border:1px solid #dee5ed;border-radius:14px;padding:20px}.value{font-size:30px;font-weight:750}.label{color:#617387;font-size:13px;margin-top:7px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;padding:12px 10px;border-bottom:1px solid #e2e8ef;line-height:1.6}th{background:#edf2f8;white-space:nowrap}tbody tr:hover{background:#f8fafd}.zero{background:#edf0f4;color:#4c5c6e;padding:12px 16px;border-radius:9px}.alert{background:#fff1cd;padding:14px 18px;border-radius:10px;margin:12px 0}.heatmap{padding:0;overflow:auto;margin:16px 0}.heatmap svg{display:block;width:100%;min-width:980px;height:auto}details{background:white;border:1px solid #dfe5ec;border-radius:12px;padding:18px;margin:14px 0}summary{cursor:pointer;font-size:17px;font-weight:700}.details-body{margin-top:16px}.filters{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}.filters select,.filters input{font:inherit;padding:10px;border:1px solid #bac8d7;border-radius:8px;background:white}.cell-id{font-family:Consolas,monospace;font-size:11px;overflow-wrap:anywhere;max-width:225px;display:block}.metric{min-width:175px}.metric small{display:block;color:#63758a;font-size:11px}.metric details{margin:8px 0 0;padding:8px;border-radius:6px}.metric summary{font-size:11px}.runline{font-size:11px;border-top:1px solid #e1e7ef;margin-top:6px;padding-top:6px}.pill{background:#e4e8ee;color:#4d5e71;display:inline-block;border-radius:6px;padding:3px 7px;white-space:nowrap}.ok{background:#d6eee4;color:#164d3c}.reason{min-width:230px;max-width:400px;overflow-wrap:anywhere}.source{overflow-wrap:anywhere;font-family:Consolas,monospace;font-size:12px}.foot,.table-note{font-size:12px;color:#617387}a{color:#236b9c}li{line-height:1.7;margin:5px 0}@media(max-width:760px){main{padding:14px}h1{font-size:26px}header{padding:22px}.cards{grid-template-columns:repeat(2,1fr)}.panel{padding:14px}}@media print{main{max-width:none;padding:0}.filters{display:none}.heatmap svg{min-width:0}.panel,details{break-inside:avoid}header{background:white;color:#152a43;border:1px solid #ccd6e0}.muted,.badge{color:#526678}}
</style><main>''', '<header><div class="badge">DEVELOPMENT EVALUATION · 完整分母 162</div><h1>原生稳定样本开发评估</h1><p>原生三次正式测量用于稳定性初筛；模拟为确定性单次预测。<strong>这是按原生稳定性选择后的开发评估，不是盲测或泛化验收。</strong></p>', '<div class="muted">评分快照：'+esc(str(report['score_created_utc'] or '尚无'))+' · 生成：'+esc(report['created_utc'])+'</div></header><div class="cards">']
    predicted = sum(s['predicted'] for s in report['groups'])
    full = sum(s['scored'] for s in report['groups'])
    for value, label in (("162", "原始135格 + 新增GPU27格"), (f'{report["selected_count"]}/162', "原生稳定入选；不等于精度通过"), (f'{predicted}/{report["selected_count"]}', "已有成功预测 / 稳定入选"), (f'{full}/162', "三项完整评分 / 原始分母")):
        out.append('<div class="card"><div class="value">'+esc(value)+'</div><div class="label">'+esc(label)+'</div></div>')
    out.append('</div><h2>六组完整覆盖</h2><div class="panel scroll"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in COVERAGE_HEADERS)+'</tr></thead><tbody>')
    for item in report['groups']:
        out.append('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in coverage_values(item))+'</tr>')
    out.append('</tbody></table><p class="table-note">有效原生优先引用 coverage.verified_cells；未知值不补零。历史失败尝试保留 '+str(len(report['failed_attempts']))+' 条。零入选组没有可评估误差。</p></div>')
    if report.get('hardware_scope_note'):
        note = report['hardware_scope_note']
        out.append('<div class="panel"><strong>硬件口径</strong><p>'+esc(note['text'])+'</p>')
        if note['cell_ids']:
            out.append('<details><summary>查看频率例外场景</summary><ul>'+''.join('<li class="cell-id">'+esc(ident)+'</li>' for ident in note['cell_ids'])+'</ul></details>')
        out.append('</div>')
    for warning in report['warnings']:
        out.append('<div class="alert">'+esc(warning)+'</div>')
    out.append('<h2>全体已评分格的条件统计</h2><div class="panel">'+summary_table(report)+'<p class="table-note">按格等权；同时列出“已评分 / 入选 / 原格”。成功样本的条件误差不能代表全部162格。未选择及失败格完整保留。</p></div>')
    out.append('<h2>三项误差热图</h2><p>底色表示 APE，格内 Δ 为有符号百分比误差。灰色不代表 0%；悬停或查看完整明细了解原因。三张图使用同一色阶。</p>')
    for metric, svg in svgs.items():
        out.append('<div class="panel heatmap">'+svg+'</div><p class="foot"><a download href="heatmap_'+metric+'.svg">下载 '+metric.upper()+' SVG</a></p>')
    out.append('<h2>六组误差明细</h2>')
    for item in report['groups']:
        group = item['group']
        out.append('<details open><summary>'+esc(LABELS[group])+f' · 入选 {item["selected"]}/27</summary><div class="details-body">')
        if item['selected'] == 0:
            out.append('<p class="zero">零入选，不能评估本组误差。27个原定格完整保留。</p>')
        out.append(summary_table(report, group))
        for m in summaries(report, group):
            if m['worst_cell']:
                out.append('<p class="foot">'+esc(m['metric'].upper()+' 最坏APE所在格：'+m['worst_cell'])+'</p>')
        out.append('</div></details>')
    out.append('<h2>全部162格、三次原生值与逐run误差</h2><p>每个可评分指标都保留三次原生值，并可展开同一个固定模拟值对R1/R2/R3的误差。筛选只影响列表显示，不改变统计分母。</p><div class="filters"><label>组别 <select id="group"><option value="">全部六组</option>')
    out.extend('<option value="'+g+'">'+esc(LABELS[g])+'</option>' for g in GROUPS)
    out.append('</select></label><label>状态 <select id="state"><option value="">全部状态</option>')
    out.extend('<option value="'+state+'">'+label+'</option>' for state, label in STATES.items())
    out.append('</select></label><label>检索 <input id="search" placeholder="Cell ID 或排除原因"></label><span id="visible" aria-live="polite"></span></div><div class="panel scroll"><table id="cells"><thead><tr><th>组别 / Cell ID</th><th>输入 / 输出 / 并发</th><th>状态</th><th>TTFT</th><th>TPOT</th><th>E2E</th><th>原因</th></tr></thead><tbody>')
    for cell in report['cells']:
        out.append('<tr data-group="'+cell['group']+'" data-state="'+cell['state']+'"><td>'+esc(LABELS[cell['group']])+'<span class="cell-id">'+esc(cell['cell_id'])+'</span></td>')
        out.append(f'<td>{cell["prompt"]} / {cell["output"]} / {cell["parallel"]}</td><td><span class="pill'+(' ok' if cell['state']=='scored' else '')+'">'+STATES[cell['state']]+'</span></td>')
        for metric in METRICS:
            r = cell['metrics'].get(metric)
            if r:
                out.append('<td class="metric"><strong>'+fmt(r['absolute_percentage_error_pct'])+'% APE</strong><small>Δ '+fmt(r['signed_error_pct'], signed=True)+'% / '+fmt(r['signed_error_ms'], signed=True)+' ms</small><small>|Δ| '+fmt(r['absolute_error_ms'])+' ms</small><small>固定模拟 '+fmt(r['simulator_median_ms'])+' ms</small><small>原生R1/R2/R3：'+' / '.join(fmt(v) for v in r['native_run_medians_ms'])+' ms</small><small>run中位数最大偏差 '+fmt(r['native_run_worst_abs_pct'])+'%</small><details><summary>展开三次误差</summary>')
                for v in r['run_comparisons']:
                    out.append('<div class="runline">R'+str(v['run'])+' · 原生 '+fmt(v['native_ms'])+' ms<br>APE '+fmt(v['ape_pct'])+'%<br>Δ '+fmt(v['signed_pct'], signed=True)+'% / '+fmt(v['signed_ms'], signed=True)+' ms</div>')
                out.append('</details></td>')
            else:
                out.append('<td class="metric" title="'+esc(metric_reason(cell, metric), quote=True)+'">—</td>')
        why = '；'.join(reason_text(r) for r in cell['reasons']) or '三项评分可用'
        out.append('<td class="reason">'+esc(why)+'</td></tr>')
    out.append('</tbody></table></div><h2>评估口径与机制覆盖</h2><div class="panel"><ol>')
    out.extend('<li>'+esc(text)+'</li>' for text in method_text())
    out.append('</ol>')
    if report['unsupported']:
        out.append('<div class="scroll"><table><thead><tr><th>机制维度</th><th>状态</th><th>格数</th><th>原记录说明</th></tr></thead><tbody>')
        for (dimension, status, reason), count in sorted(report['unsupported'].items()):
            out.append('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in (dimension, status, count, reason))+'</tr>')
        out.append('</tbody></table></div>')
    else:
        out.append('<p>尚无机制覆盖说明；不据此推定全部机制已匹配。</p>')
    out.append('</div><h2>来源与可复查身份</h2><div class="panel"><p>完整历史失败尝试位于原生报告或选择文件的 failed_attempts 字段。本页不覆盖原始JSON。</p>')
    for name, ref in report['sources'].items():
        out.append('<p class="source"><strong>'+esc(name)+'</strong><br>'+esc(ref['path'])+'<br>JSON SHA256: '+esc(ref['sha256'])+'</p>')
    out.append('</div></main><script>const rows=[...document.querySelectorAll("#cells tbody tr")];function filter(){const g=document.getElementById("group").value,s=document.getElementById("state").value,q=document.getElementById("search").value.trim().toLowerCase();let n=0;for(const r of rows){const show=(!g||r.dataset.group===g)&&(!s||r.dataset.state===s)&&(!q||r.textContent.toLowerCase().includes(q));r.hidden=!show;if(show)n++;}document.getElementById("visible").textContent="当前显示 "+n+" / 162 格";}for(const id of ["group","state","search"])document.getElementById(id).addEventListener("input",filter);filter();</script></html>')
    return '\n'.join(out)+'\n'


def atomic_text(path, text):
    temp = path.with_name(path.name+'.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description='从已有评分生成完整162格中文报告；不会运行预测。')
    parser.add_argument('--sim-dir', type=Path, required=True, help='含freeze.json、errors.*.json、predictions/的评估目录')
    parser.add_argument('--errors', type=Path, help='指定评分快照；默认最大编号errors.NNNN.json')
    parser.add_argument('--native-report', type=Path, help='完整162格原生report.json；默认使用评分引用或冻结选择及coverage')
    parser.add_argument('--selection', type=Path, help='原冻结选择的显式位置；SHA须与freeze一致')
    parser.add_argument('--output', type=Path, help='报告输出目录；默认与--sim-dir相同')
    args = parser.parse_args(argv)
    try:
        report = load_evaluation(args.sim_dir, args.errors, args.native_report, args.selection)
        output = (args.output or args.sim_dir).resolve()
        svgs = {metric: heatmap_svg(report, metric) for metric in METRICS}
        documents = {'report.md': render_markdown(report), 'report.html': render_html(report, svgs), **{'heatmap_'+metric+'.svg': svg for metric, svg in svgs.items()}}
        output.mkdir(parents=True, exist_ok=True)
        for name, text in documents.items():
            atomic_text(output/name, text)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, '报告未生成：'+str(exc)+'\n')
    print(json.dumps({'output':str(output), 'files':list(documents), 'original_denominator':162, 'selected_cells':report['selected_count'], 'all_metrics_scored':sum(s['scored'] for s in report['groups']), 'evaluation_type':'development_post_selection', 'blind_evaluation':False}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
