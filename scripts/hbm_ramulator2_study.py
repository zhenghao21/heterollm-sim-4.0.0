"""Compare traceable Ramulator2 timing inputs inside this project's HBM core.

This does not execute Ramulator2 or claim vendor calibration. Geometry, clock,
directional budgets, controller and host streams stay fixed. A single authored
input can also change dependent RC/RTW values; those changes are reported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from calibrated_hbf_profiles import _hbm_physical_config
from hbm_sensitivity_study import workloads as sensitivity_workloads, quantile, save
from hbf_hbm_serving_study import POINT_RESULT_VERSION, run_case
from hbf_hbm_decode_study import decode_measurement, core_checks, EXECUTION_CONTRACT
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_contract import reconfigure_physical_config

GIB = 1024 ** 3
LARGE_MODELS = ("llama3_1-70b", "qwen3-235b-a22b")
SERVING_MODELS = ("small", "qwen2_5-0_5b") + LARGE_MODELS
DERIVED = {"activate_cycle_ns", "maximum_active_ns"}
INVARIANTS = (
    "generation", "capacity_bytes", "stacks", "dies_per_stack", "channels",
    "pseudo_channels_per_channel", "bank_groups_per_rank", "banks_per_group",
    "rows_per_bank", "row_bytes", "burst_bytes", "bank_geometry_scope",
    "stack_interface_mode", "command_clock_ns", "data_rate_mt_s",
    "interface_bandwidth_gb_s", "read_bandwidth_gb_s", "write_bandwidth_gb_s",
    "address_mapping", "interleave_bytes", "scheduler", "scheduler_window",
    "queue_scope", "max_outstanding_requests", "control_mode",
)


@lru_cache(maxsize=None)
def base_configuration(profile="reference"):
    return _hbm_physical_config(capacity_bytes=512 * GIB, read_gb_s=1600.,
                                write_gb_s=1600., parameter_profile=profile)


def timing_fields():
    return sorted(set(base_configuration().metadata["hbm_parameter_profile"]["parameters"])
        | set(base_configuration("ramulator2").metadata["hbm_parameter_profile"]["parameters"]))


def changed_fields(config, baseline=None):
    baseline = base_configuration() if baseline is None else baseline
    return [name for name in timing_fields() if getattr(config, name) != getattr(baseline, name)]


def variants():
    """One field authored at a time; derived RC is covered by RAS/RP cases."""
    return ["reference", "ramulator2"] + ["single_" + name for name in
        changed_fields(base_configuration("ramulator2")) if name not in DERIVED]


def timing_overrides(variant):
    if variant in ("reference", "ramulator2"):
        return {}
    field = variant.removeprefix("single_")
    if variant not in variants():
        raise ValueError("unknown timing comparison variant: " + variant)
    return {field: getattr(comparison_source(variant), field)}


def comparison_source(variant):
    upstream = base_configuration("ramulator2")
    if variant == "single_read_to_write_command_ns":
        reference = base_configuration()
        # Isolate the upstream turnaround formula while holding RL/WL at
        # their reference values. Copying the full pack's 17 CK would also
        # implicitly import the other pack's different pipeline assumptions.
        return reconfigure_physical_config(upstream, overrides={
            "read_latency_ns": reference.read_latency_ns,
            "write_latency_ns": reference.write_latency_ns})
    return upstream


@lru_cache(maxsize=None)
def configuration(variant):
    if variant in ("reference", "ramulator2"):
        return base_configuration(variant)
    upstream = comparison_source(variant).metadata
    fields = timing_overrides(variant)
    # Public timing override remains an authored input. Preserve the source of
    # the copied value separately instead of falsely relabeling the whole pack.
    return reconfigure_physical_config(base_configuration(), overrides={**fields,
        "metadata": {"ramulator2_comparison": {
            "base_profile": "reference", "copied_from_profile": "ramulator2",
            "comparison_mode": ("upstream_RTW_formula_with_reference_RL_WL" if variant == "single_read_to_write_command_ns"
                                else "copy_one_upstream_input"),
            "vendor_calibrated": False,
            "upstream_source": deepcopy(upstream["hbm_parameter_profile"].get("upstream_source", {})),
            "copied_parameter_sources": {field: deepcopy(upstream["parameter_provenance"][field])
                                         for field in fields}}}})


def address(config, bank=0, row=0, column=0):
    """Lane zero, the fixed bank-interleaved mapping used by this experiment."""
    return (((row * (config.row_bytes // config.burst_bytes) + column) * config.bank_count + bank)
            * config.lane_count * config.burst_bytes)


def workloads():
    config = base_configuration()
    result = sensitivity_workloads()
    def req(name, operation="read", *, bank=0, row=0, at=0.):
        return AccessRequest(name, operation, address(config, bank, row), 32, at)
    result.update({
        "cold_read": [req("cold-read")],
        "cold_write": [req("cold-write", "write")],
        "late_read_conflict": [req("warm"), req("late", at=100.), req("conflict", row=1, at=100.)],
        "write_row_conflicts": [req(str(i), "write", row=i % 2) for i in range(256)],
        "different_sid_reads": [req("warm-0"), req("warm-16", bank=16)] + [
            req(str(i), bank=16 * (i % 2), at=100.) for i in range(256)],
        "different_group_rw": [req("warm-0"), req("warm-4", bank=4)] + [
            req(str(i), "write" if i % 2 == 0 else "read", bank=4 * (i % 2), at=100.)
            for i in range(256)],
    })
    return result


def scalar_configuration(config):
    values = asdict(config) if not isinstance(config, dict) else config
    return {field: value for field, value in values.items() if field != "metadata"}


def executed_hbm_configuration(row):
    executed = row.get("core_execution_audit", {}).get("executed_physical_configurations", {})
    hbm = [item["physical_config"] for item in executed.values() if item["physical_config"].get("kind") == "HBM"]
    if len(hbm) != 1:
        raise ValueError("expected exactly one actually executed HBM configuration")
    return hbm[0]


def matches_current_inputs(row, *, serving_case=False):
    variant = row.get("variant")
    if variant not in variants() or row.get("point_result_version") != POINT_RESULT_VERSION:
        return False
    if row.get("timing_comparison") != variant_descriptor(variant):
        return False
    if serving_case:
        try:
            actual = scalar_configuration(executed_hbm_configuration(row))
        except (KeyError, ValueError):
            return False
        expected = scalar_configuration(configuration(variant))
        params = row.get("parameters", {})
        expected_profile = "ramulator2" if variant == "ramulator2" else "reference"
        return (actual == expected and params.get("hbm_parameter_profile") == expected_profile
            and (params.get("hbm_timing_overrides") or {}) == timing_overrides(variant)
            and params.get("hbm_capacity_bytes") == 512 * GIB)
    return row.get("physical_configuration") == scalar_configuration(configuration(variant))


def micro(variant, name, requests):
    detailed = len(requests) <= 4 and sum(request.byte_count for request in requests) <= 128
    core = DramCore(configuration(variant), capture_details=detailed)
    started = time.perf_counter()
    results = core.submit_many(requests)
    duration = max(row.completion_ns for row in results) - min(row.arrival_ns for row in results)
    latencies = [row.completion_ns - row.arrival_ns for row in results]
    names = ("activate_commands", "precharge_commands", "read_commands", "write_commands", "refresh_commands",
             "row_hits", "row_misses", "row_conflicts", "queue_wait_ns", "scheduler_reordered_requests")
    return {"variant": variant, "workload": name, "requests": len(requests),
        "point_result_version": POINT_RESULT_VERSION, "timing_comparison": variant_descriptor(variant),
        "physical_configuration": scalar_configuration(core.config),
        "logical_bytes": sum(row.logical_bytes for row in results), "duration_ns": duration,
        "effective_gb_s": sum(row.logical_bytes for row in results) / duration,
        "request_latency_ns": {"p50": quantile(latencies, .5), "p95": quantile(latencies, .95),
                               "p99": quantile(latencies, .99), "max": max(latencies)},
        "request_completion_ns": [row.completion_ns for row in results],
        "command_events": [{"request_id": row.request_id, "name": stage.name,
            "start_ns": stage.start_ns, "end_ns": stage.end_ns} for row in results for stage in row.stages
            if stage.name in {"ACTIVATE", "PRECHARGE", "READ_COMMAND", "WRITE_COMMAND", "BURST_TRANSFER"}] if detailed else [],
        "counters": {name: sum(row.counters.get(name, 0) for row in results) for name in names},
        "compiled_requests": sum(bool(row.counters.get("compiled_hbm_float64")) for row in results),
        "wall_seconds": time.perf_counter() - started}


def serving(variant, architecture, model, prompt, batch, output):
    measured = {}
    def observe(result):
        measured.update(decode_measurement(result.serving))
    row_id = f"{model}-{prompt}-{batch}-512-{variant}-{architecture}"
    started = time.perf_counter()
    row = run_case(architecture, row_id=row_id, model_id=model, prompt=prompt, batch=batch, output=output,
        hbm_capacity_bytes=512 * GIB, hbm_parameter_profile="ramulator2" if variant == "ramulator2" else "reference",
        hbm_timing_overrides=timing_overrides(variant) or None, result_observer=observe)
    if not row.get("executed_serving"):
        raise RuntimeError(f"Serving case {row_id} did not execute: {row.get('reason')}")
    row.update(point_result_version=POINT_RESULT_VERSION, memory_execution_contract=EXECUTION_CONTRACT,
        variant=variant, decode_measurement=measured, wall_seconds=time.perf_counter() - started,
        timing_comparison=variant_descriptor(variant))
    row["decode_core_checks"] = core_checks(row)
    if (row["completed_requests"] != batch or not all(row["decode_core_checks"].values())
            or not all(measured["checks"].values()) or not all(row["placement_verification"].values())
            or not matches_current_inputs(row, serving_case=True)):
        raise RuntimeError("Serving validation failed: " + row_id)
    return row


def serving_point(point):
    return serving(*point)


def planned_serving_points(selected=None, stage="all", models=None):
    selected = variants() if selected is None else selected
    stage = "all" if stage == "full-service" else stage
    cases = []
    # Named model points take longer; start them first so they do not become
    # a serial tail after all small-model work has completed.
    if stage in ("all", "large"):
        cases += [(variant, architecture, model, 128, 1, 4)
                  for model in LARGE_MODELS
                  for variant in ("reference", "ramulator2") if variant in selected
                  for architecture in ("A_hbm_only", "B_all_kv_hbf")]
    if stage in ("all", "named"):
        cases += [(variant, architecture, "qwen2_5-0_5b", 128, 1, 4)
                  for variant in ("reference", "ramulator2") if variant in selected
                  for architecture in ("A_hbm_only", "B_all_kv_hbf")]
    if stage in ("all", "serving"):
        cases += [(variant, architecture, "small", 1024, 4, 8)
                  for variant in selected for architecture in ("A_hbm_only", "B_all_kv_hbf")]
    return cases if models is None else [point for point in cases if point[2] in models]


def variant_descriptor(variant):
    config = configuration(variant)
    authored = list(timing_overrides(variant))
    changed = changed_fields(config)
    return {"name": variant, "authored_fields": authored, "changed_fields": changed,
        "dependent_changed_fields": [field for field in changed if field not in authored] if authored else [],
        "vendor_calibrated": False,
        "source": deepcopy(config.metadata.get("ramulator2_comparison", {}).get("copied_parameter_sources",
            config.metadata["hbm_parameter_profile"]["parameters"])),
        "fixed_configuration_matches_reference": all(getattr(config, field) == getattr(base_configuration(), field)
                                                      for field in INVARIANTS)}


def rounded_bound(config, field):
    """Earliest grid-aligned bound; contention can add further delay."""
    value = getattr(config, field)
    if isinstance(value, str):
        return {"value": value, "unit": "policy", "rule": "policy"}
    half = {"minimum_active_ns", "precharge_to_activate_ns", "read_to_precharge_ns", "write_to_precharge_ns"}
    quantum = config.command_clock_ns / (2 if field in half else 1)
    if field in {"read_latency_ns", "write_latency_ns"}:
        return {"value": value, "unit": "ns", "rule": "fixed_data_pipeline_latency"}
    return {"value": math.ceil(value / quantum) * quantum, "unit": "ns", "quantum_ns": quantum,
        "rule": "ceil_from_grid_aligned_anchor; command resource or anchor phase can add delay"}


def parameter_comparison():
    result = []
    for field in timing_fields():
        item = {"field": field, "differs": field in changed_fields(base_configuration("ramulator2"))}
        for name in ("reference", "ramulator2"):
            config = base_configuration(name)
            source = deepcopy(config.metadata["parameter_provenance"].get(field, {}))
            item[name] = {"source": source, "configured_value": getattr(config, field),
                          "rounded_bound": rounded_bound(config, field)}
        item["symbol"] = item["ramulator2"]["source"].get("symbol", field)
        low, high = sorted((getattr(base_configuration("fast"), field), getattr(base_configuration("slow"), field)))
        item["existing_sensitivity_range"] = [low, high]
        upstream_value = item["ramulator2"]["configured_value"]
        item["outside_existing_sensitivity_range"] = not isinstance(upstream_value, str) and not low <= upstream_value <= high
        item["covered_by_variants"] = [variant for variant in variants() if variant.startswith("single_")
                                       and field in changed_fields(configuration(variant))]
        result.append(item)
    return result


def aggregate(output_dir):
    micros = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output_dir / "micro").glob("*.json"))]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output_dir / "serving").glob("*.json"))]
    expected_micro = {(variant, workload) for variant in variants() for workload in workloads()}
    expected_serving = set(planned_serving_points())
    actual_serving = {(row["variant"], row["placement"], row["parameters"]["model_id"], row["parameters"]["prompt"],
        row["parameters"]["batch"], row["parameters"]["output"]) for row in rows}
    lookup = {(row["variant"], row["workload"]): row for row in micros}
    descriptions = {variant: variant_descriptor(variant) for variant in variants()}
    parameters = parameter_comparison()
    checks = {
        "all_planned_microbenchmarks_executed": expected_micro == set(lookup) and len(micros) == len(expected_micro),
        "all_planned_serving_cases_executed": expected_serving == actual_serving and len(rows) == len(expected_serving),
        "all_microbenchmarks_use_current_actual_inputs": bool(micros) and all(matches_current_inputs(row) for row in micros),
        "all_serving_cases_use_current_actual_inputs": bool(rows) and all(matches_current_inputs(row, serving_case=True) for row in rows),
        "all_different_timings_covered_individually_or_through_dependencies": all(
            not item["differs"] or item["covered_by_variants"] for item in parameters),
        "geometry_clock_caps_and_controller_fixed": all(item["fixed_configuration_matches_reference"] for item in descriptions.values()),
        "all_serving_cases_use_current_execution": bool(rows) and all(row.get("point_result_version") == POINT_RESULT_VERSION for row in rows),
        "all_serving_cases_have_real_prefill_decode_and_core_io": bool(rows) and all(
            all(row["decode_core_checks"].values()) and all(row["decode_measurement"]["checks"].values())
            and all(row["placement_verification"].values()) for row in rows),
    }
    micro_comparisons = []
    for (variant, workload), row in lookup.items():
        baseline = lookup.get(("reference", workload))
        if variant == "reference" or baseline is None:
            continue
        micro_comparisons.append({"variant": variant, "workload": workload,
            "bandwidth_change_percent": (row["effective_gb_s"] / baseline["effective_gb_s"] - 1) * 100,
            "p99_change_ns": row["request_latency_ns"]["p99"] - baseline["request_latency_ns"]["p99"],
            "max_completion_change_ns": max(abs(a-b) for a, b in zip(row["request_completion_ns"], baseline["request_completion_ns"])),
            "refresh_commands": row["counters"]["refresh_commands"]})
    serving_lookup = {(row["variant"], row["placement"], row["parameters"]["model_id"]): row for row in rows}
    serving_comparisons = []
    for variant in variants():
        for model in SERVING_MODELS:
            pair = [serving_lookup.get((variant, architecture, model)) for architecture in ("A_hbm_only", "B_all_kv_hbf")]
            if any(row is None for row in pair):
                continue
            a, b = pair
            entry = {"variant": variant, "model": model, "a_tpot_ns": a["tpot_ns"]["p50"],
                "b_tpot_ns": b["tpot_ns"]["p50"], "b_over_a": b["tpot_ns"]["p50"] / a["tpot_ns"]["p50"],
                "faster": "A" if a["tpot_ns"]["p50"] < b["tpot_ns"]["p50"] else "B"}
            for short, architecture in (("a", "A_hbm_only"), ("b", "B_all_kv_hbf")):
                measured = a if short == "a" else b
                entry[short + "_ttft_ns"] = measured["ttft_ns"]["p50"]
                entry[short + "_node_tps"] = measured["node_tps"]
                baseline = serving_lookup.get(("reference", architecture, model))
                entry[short + "_tpot_change_percent"] = ((entry[short + "_tpot_ns"] / baseline["tpot_ns"]["p50"] - 1) * 100
                    if baseline else None)
                for metric, baseline_value in (("ttft_ns", baseline["ttft_ns"]["p50"] if baseline else None),
                                               ("node_tps", baseline["node_tps"] if baseline else None)):
                    entry[short + "_" + metric + "_change_percent"] = (
                        (entry[short + "_" + metric] / baseline_value - 1) * 100 if baseline_value else None)
            serving_comparisons.append(entry)
    result = {"status": "passed" if all(checks.values()) else "incomplete", "checks": checks,
        "point_result_version": POINT_RESULT_VERSION, "memory_execution_contract": EXECUTION_CONTRACT,
        "vendor_calibrated": False, "executes_ramulator2": False,
        "scope": "Ramulator2 timing values transferred into this project's existing HBM command model; no vendor accuracy claim",
        "planned_micro_points": len(expected_micro), "planned_serving_points": len(expected_serving),
        "upstream_profile": deepcopy(base_configuration("ramulator2").metadata["hbm_parameter_profile"]),
        "parameters": parameters, "variants": descriptions,
        "execution_paths": {
            "micro_requests": sum(row["requests"] for row in micros),
            "micro_compiled_requests": sum(row["compiled_requests"] for row in micros),
            "serving_hbm_transactions": sum(owner["transactions"] for row in rows
                for owner in row["core_execution_audit"]["owners"] if owner["memory_kind"] == "HBM"),
            "serving_compiled_hbm_transactions": sum(owner["compiled_hbm_float64_transactions"] for row in rows
                for owner in row["core_execution_audit"]["owners"] if owner["memory_kind"] == "HBM"),
        },
        "configurations": {variant: asdict(configuration(variant)) for variant in variants()},
        "microbenchmarks": micros, "micro_comparisons": micro_comparisons,
        "serving_comparisons": serving_comparisons,
        "serving_cases": [{key: row[key] for key in ("id", "variant", "parameters", "placement", "tpot_ns",
            "ttft_ns", "node_tps", "makespan_ns", "model_structure",
            "decode_core_checks", "decode_measurement", "wall_seconds", "timing_comparison")} for row in rows]}
    save(output_dir / "results.json", result)
    write_report(output_dir / "report.md", result)
    return result


def write_report(path, result):
    upstream = result["upstream_profile"]["upstream_source"]
    execution = result["execution_paths"]
    lines = ["# 在本仿真器中采用 Ramulator2 来源时序参数的对照", "", f"执行版本：{POINT_RESULT_VERSION}；状态：{result['status']}。",
        f"计划/已执行：独立访存 {result['planned_micro_points']}/{len(result['microbenchmarks'])}；完整服务 {result['planned_serving_points']}/{len(result['serving_cases'])}。",
        f"固定来源：[Ramulator2 HBM3 源码]({upstream['url']})，提交 `{upstream['commit']}`。",
        "", "本实验把 Ramulator2 HBM3 的可映射时序输入放入本项目 HBM 命令模型；没有运行 Ramulator2 本体。",
        "两者的命令计时锚点和控制器、刷新调度实现不同，因此相同输入周期数不代表逐命令执行等价。",
        "两组都不是厂商标定。来源包含上游估算、上游采用的标准值和本模型补充规则，逐字段来源见 results.json。",
        "固定条件：32 Gb × 16 层、512 GiB、CK=0.625 ns、读写方向预算各 1600 GB/s、bank_interleaved、FCFS、device 队列深度 64。",
        "完整服务先运行 Prefill，再独立统计 Decode；A 为全 HBM，B 为 KV 使用 HBF。",
        f"实际执行路径：微基准 {execution['micro_compiled_requests']}/{execution['micro_requests']} 个请求使用编译数值核；服务 Core 审计中 {execution['serving_compiled_hbm_transactions']}/{execution['serving_hbm_transactions']} 个 HBM 事务使用该路径。小探针为保留命令事件使用详细参考核，其他不适用编译路径的事务也使用同一 JEDEC 命令模型。",
        "", "## 参数输入与执行边界", "",
        "源值/单位、配置值、时钟边界上的最早有效下限分开保存。下表配置和有效下限的时间单位为 ns，原始输入按源单位标注；有效值不是含资源冲突的最终命令间隔。",
        "tRP 等受命令相位影响，后续 ACT 还需对齐上升沿；tRAS/tRP 修改会联动 tRC，RL/WL 修改会联动 tRTW。另设 tRTW 公式对照：保持参考组 RL=20 ns、WL=5 ns，仅采用上游公式得 31 CK=19.375 ns；不把全组不同 RL/WL 下的 17 CK 直接移入参考组。",
        "", "| 参数 | 上游原始输入 | 参考配置 | Ramulator2 配置 | 参考有效下限 | 对照有效下限 | 单项覆盖 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for item in result["parameters"]:
        r, u = item["reference"], item["ramulator2"]
        covered = ", ".join(name.removeprefix("single_") for name in item["covered_by_variants"]) or "两组相同"
        source = u["source"]
        raw = (f"{source['input_value']} {source['input_unit']}" if "input_value" in source
               else source.get("input_unit", "本模型规则"))
        upstream_name = source.get("upstream_parameter")
        if source.get("input_unit") == "formula" and upstream_name in result["upstream_profile"]["upstream_raw_timings"]:
            raw = f"{result['upstream_profile']['upstream_raw_timings'][upstream_name]} nCK（公式）"
        if source.get("source_kind", "").startswith("local_"):
            raw += "（本模型补充）"
        lines.append(f"| {item['symbol']} ({item['field']}) | {raw} | {r['configured_value']} | {u['configured_value']} | {r['rounded_bound']['value']} | {u['rounded_bound']['value']} | {covered} |")
    outside = [item["symbol"] for item in result["parameters"] if item["outside_existing_sensitivity_range"]]
    lines += ["", f"上游组有 {len(outside)} 项配置值超出现有 fast/slow 两端：{', '.join(outside)}。原三档只是选定工程场景，不能当作厂商保证边界或所有公开模型的包络；本次保留原预设并增加独立对照组。"]
    lines += ["", "## 逐项影响", "",
        "每一行只人工替换所列输入，自动联动字段另列。微负载值选取该组相对参考带宽变化绝对值最大的已执行负载；这不是 LLM 加速比例。",
        "", "| 人工替换输入 | 替换值 ns | 自动联动 | 影响最大的微负载 | 带宽变化 | Small A TPOT变化 | Small B TPOT变化 |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |"]
    def percent(value):
        return "未执行" if value is None else f"{value:+.4f}%"
    for variant, description in result["variants"].items():
        if not variant.startswith("single_"):
            continue
        micros = [item for item in result["micro_comparisons"] if item["variant"] == variant]
        most = max(micros, key=lambda item: abs(item["bandwidth_change_percent"])) if micros else None
        service = next((item for item in result["serving_comparisons"] if item["variant"] == variant and item["model"] == "small"), {})
        label = variant.removeprefix('single_') + ("（公式对照）" if variant == "single_read_to_write_command_ns" else "")
        authored = next(iter(timing_overrides(variant).values()))
        workload_label = ("全部已执行负载均为零" if most and most["bandwidth_change_percent"] == 0
                          else most["workload"] if most else "未执行")
        lines.append(f"| {label} | {authored:g} | {', '.join(description['dependent_changed_fields']) or '无'} | {workload_label} | {percent(most['bandwidth_change_percent'] if most else None)} | {percent(service.get('a_tpot_change_percent'))} | {percent(service.get('b_tpot_change_percent'))} |")
    lines += ["", "不同参数的影响不可直接相加；全组替换可能改变最先阻塞的约束。零变化只表示这些固定条件下未改变观测结果。",
        "尤其是跨 SID 的列命令间隔可能被每 PC 的方向预算遮蔽：32 B / 6.25 GB/s = 5.12 ns，已大于两组 tCCDR；因此专门增加跨 SID 探针，并保留其零影响。",
        "冷读/冷写分别检验 RCD 与管线延迟；晚到读后换行检验 RTP；写行冲突检验 WR；异组读写检验 WTR_S。所有探针保持相同非时序配置、请求地址和到达时间。",
        "", "## 完整服务", "", "TPOT 越小越好。Small 的输入/批量/输出为 1024/4/8；Qwen2.5-0.5B、Llama3.1-70B 和 Qwen3-235B-A22B 均为 128/1/4。命名模型只做 reference 与 Ramulator2 全组对照。",
        "模型结构与工作量由已执行记录提取；命名模型按 BF16 计算权重存储量，Small 使用 INT8，未下载或计算真实权重数值。",
        "", "| 模型 | 层数 | 模型权重 GiB | 输入 token | 并发请求 | 每请求输出 token | 架构预设支持级别 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for model in SERVING_MODELS:
        row = next((item for item in result["serving_cases"] if item["parameters"]["model_id"] == model), None)
        if row:
            shape, params = row["model_structure"], row["parameters"]
            lines.append(f"| {model} | {shape['layers']} | {shape['weight_bytes']/GIB:.5f} | {params['prompt']} | {params['batch']} | {params['output']} | {shape.get('preset_support_level', 'synthetic')} |")
    lines += ["", "Llama3.1-70B 为稠密模型；Qwen3-235B-A22B 为 MoE，总参数约 235B、每 token 名义激活约 22B，每层选择 128 个专家中的 8 个。其专家路由是分析近似，不是运行真实权重得出的路由。架构预设的 exact 只表示结构字段覆盖，不能表示性能经过实机标定。", "",
        "", "| 模型 | 对照组 | A TPOT µs | B TPOT µs | B/A | 更快 | A变化 | B变化 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for item in result["serving_comparisons"]:
        lines.append(f"| {item['model']} | {item['variant']} | {item['a_tpot_ns']/1000:.5f} | {item['b_tpot_ns']/1000:.5f} | {item['b_over_a']:.5f} | {item['faster']} | {percent(item['a_tpot_change_percent'])} | {percent(item['b_tpot_change_percent'])} |")
    lines += ["", "### 命名模型的首 token 与完整请求吞吐", "",
        "TTFT 为从请求到达到首 token 的时间，越小越好。完整请求吞吐为输出 token 总数除以从请求到达到仿真结束的时间，包含初始化等待与 Prefill；它不是稳定 Decode 吞吐，不能用 1/TPOT 替代。",
        "", "| 模型 | 对照组 | A TTFT ms | B TTFT ms | A TTFT变化 | B TTFT变化 | A完整吞吐 token/s | B完整吞吐 token/s | A吞吐变化 | B吞吐变化 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in result["serving_comparisons"]:
        if item["model"] == "small" or item["variant"] not in ("reference", "ramulator2"):
            continue
        lines.append(f"| {item['model']} | {item['variant']} | {item['a_ttft_ns']/1e6:.5f} | {item['b_ttft_ns']/1e6:.5f} | {percent(item['a_ttft_ns_change_percent'])} | {percent(item['b_ttft_ns_change_percent'])} | {item['a_node_tps']:.5f} | {item['b_node_tps']:.5f} | {percent(item['a_node_tps_change_percent'])} | {percent(item['b_node_tps_change_percent'])} |")
    lines += ["", "## 全组替换的独立访存结果", "", "| 负载 | 带宽变化 | P99 延迟变化 ns | 最大请求完成时刻变化 ns |",
        "| --- | ---: | ---: | ---: |"]
    for item in result["micro_comparisons"]:
        if item["variant"] == "ramulator2":
            lines.append(f"| {item['workload']} | {percent(item['bandwidth_change_percent'])} | {item['p99_change_ns']:+.5f} | {item['max_completion_change_ns']:.5f} |")
    lines += ["", "## 解释边界与复现", "",
        "- 逐项组保留 reference 的其他规则，尤其 RL/WL 自动联动仍使用 reference 的 PHY 假设；它们不应被解释为完整上游参数组。",
        "- 上游未映射的参数及原因保存在 upstream_profile；本轮不新增 REFpb、RFM、低功耗、初始化或错误注入。",
        "- 70B/235B 扩展只测试短输入、单请求、4 个输出 token，因此只有 3 个 Decode 间隔；未覆盖长上下文、高并发、容量不足或长期稳态，更不能据此外推所有大模型和 HBF 容量收益。",
        "- 从 0.5B 换成 70B/235B 时，层数、隐藏维度、注意力结构和稠密/MoE 结构也随模型改变；本实验扩大模型覆盖范围，不是仅改变参数数量的因果实验。",
        "- 本次复现命令：`.venv/Scripts/python.exe scripts/hbm_ramulator2_study.py --stage all --workers 8`。大模型扩展命令：`.venv/Scripts/python.exe scripts/hbm_ramulator2_study.py --stage large --workers 8 --resume`。使用项目虚拟环境以启用数值执行核，并发默认值为 2。可用 micro/serving/named/large/full-service/report 分阶段执行；`--resume` 复用本次目录中已完成且版本、变体、实际输入一致的点。",
        "- 部分执行状态始终为 incomplete；只有全部预定独立访存与完整服务点完成并通过实际执行检查，才标记 passed。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/hbm_ramulator2_20261015")
    parser.add_argument("--stage", choices=("micro", "serving", "named", "large", "full-service", "report", "all"), default="all")
    parser.add_argument("--variants", default="")
    parser.add_argument("--models", default="", help="Comma-separated serving model IDs; does not change the complete study plan")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    selected = args.variants.split(",") if args.variants else variants()
    selected_models = args.models.split(",") if args.models else None
    selected_workloads = args.workloads.split(",") if args.workloads else list(workloads())
    if any(variant not in variants() for variant in selected):
        parser.error("unknown variant")
    if selected_models is not None and any(model not in SERVING_MODELS for model in selected_models):
        parser.error("unknown serving model")
    if any(name not in workloads() for name in selected_workloads):
        parser.error("unknown workload")
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("micro", "all"):
        requests = workloads()
        for variant in selected:
            for name in selected_workloads:
                path = args.output_dir / "micro" / f"{variant}-{name}.json"
                if args.resume and path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if saved.get("variant") == variant and saved.get("workload") == name and matches_current_inputs(saved):
                        continue
                row = micro(variant, name, requests[name])
                save(path, row)
                print(json.dumps({"micro": variant + "/" + name, "gb_s": row["effective_gb_s"], "wall_s": row["wall_seconds"]}), flush=True)
    if args.stage in ("serving", "named", "large", "full-service", "all"):
        points = planned_serving_points(selected, args.stage, selected_models)
        if args.resume:
            pending = []
            for point in points:
                variant, architecture, model, prompt, batch, _ = point
                path = args.output_dir / "serving" / f"{model}-{prompt}-{batch}-512-{variant}-{architecture}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    params = saved.get("parameters", {})
                    saved_point = (saved.get("variant"), saved.get("placement"), params.get("model_id"),
                                   params.get("prompt"), params.get("batch"), params.get("output"))
                    if saved_point == point and matches_current_inputs(saved, serving_case=True):
                        continue
                pending.append(point)
            points = pending
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(serving_point, point) for point in points]
            for future in as_completed(futures):
                row = future.result()
                save(args.output_dir / "serving" / (row["id"] + ".json"), row)
                print(json.dumps({"serving": row["id"], "tpot_ns": row["tpot_ns"]["p50"], "wall_s": row["wall_seconds"]}), flush=True)
    report = aggregate(args.output_dir)
    print(json.dumps({"status": report["status"], "micro_points": len(report["microbenchmarks"]),
                      "serving_points": len(report["serving_cases"])}), flush=True)


if __name__ == "__main__":
    main()
