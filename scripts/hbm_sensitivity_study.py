"""Execute HBM timing and controller sensitivity through the actual cores.

The parameter envelopes are engineering assumptions, not confidence intervals
or a calibration to a commercial HBM part. All comparisons preserve geometry,
physical clock, directional caps and host request streams.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from calibrated_hbf_profiles import _hbm_physical_config
from hbf_hbm_serving_study import POINT_RESULT_VERSION, run_case
from hbf_hbm_decode_study import decode_measurement, core_checks, EXECUTION_CONTRACT
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_contract import reconfigure_physical_config
from heterollm_sim.hbm_timing_profiles import timing_profile_catalog

GIB = 1024**3
MIB = 1024**2
VARIANTS = {
    "reference": {"parameter_profile": "reference"},
    "fast": {"parameter_profile": "fast"},
    "slow": {"parameter_profile": "slow"},
    "unconfigured": {"parameter_profile": "none"},
    "no_refresh": {"parameter_profile": "reference", "overrides": {
        "refresh_policy": "disabled", "refresh_interval_ns": 0., "refresh_cycle_ns": 0.}},
    "no_activation_gates": {"parameter_profile": "reference", "overrides": {
        "activate_same_group_ns": 0., "activate_different_group_ns": 0., "four_activate_window_ns": 0.}},
    "no_rw_recovery": {"parameter_profile": "reference", "overrides": {
        "read_to_precharge_ns": 0., "write_to_read_same_group_ns": 0.,
        "write_to_read_different_group_ns": 0., "read_to_write_command_ns": 0.}},
    "row_local": {"parameter_profile": "reference", "controller": {
        "address_mapping": "row_local", "interleave_bytes": None}},
    "row_hit_first": {"parameter_profile": "reference", "controller": {
        "scheduler": "row_hit_first", "scheduler_window": 16}},
    "read_write_batch": {"parameter_profile": "reference", "controller": {
        "scheduler": "read_write_batch", "scheduler_window": 16}},
    "device_depth8": {"parameter_profile": "reference", "controller": {"max_outstanding_requests": 8}},
    "device_depth128": {"parameter_profile": "reference", "controller": {"max_outstanding_requests": 128}},
    "channel_depth1": {"parameter_profile": "reference", "controller": {
        "queue_scope": "channel", "max_outstanding_requests": 1}},
    "channel_depth8": {"parameter_profile": "reference", "controller": {
        "queue_scope": "channel", "max_outstanding_requests": 8}},
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configuration(variant, capacity_gib=512):
    spec = VARIANTS[variant]
    config = _hbm_physical_config(capacity_bytes=capacity_gib * GIB,
        read_gb_s=1600., write_gb_s=1600., parameter_profile=spec["parameter_profile"])
    changes = {**spec.get("overrides", {}), **spec.get("controller", {})}
    return reconfigure_physical_config(config, overrides=changes) if changes else config


def workloads():
    """Identical host addresses in every mapping; seed and sizes are fixed."""
    rng = random.Random(238)
    random_addresses = [rng.randrange(0, 64 * MIB // 64) * 64 for _ in range(1024)]
    # One hot bank and adjacent banks under the default mapping. Every group
    # mixes a cold row with a revisited hot row so finite-window scheduling
    # has useful choices without future arrivals or dependent writes.
    row_span = 256 * 64 * 1024
    locality = [((i // 8) % 8) * 8192 + (row_span if i % 4 == 1 else 0)
                + (i % 4) * 64 for i in range(512)]
    return {
        "sequential_read": [AccessRequest("stream-r", "read", 0, 16 * MIB)],
        "sequential_write": [AccessRequest("stream-w", "write", 0, 16 * MIB)],
        "random_read": [AccessRequest(str(i), "read", address, 64) for i, address in enumerate(random_addresses)],
        "mixed_rw": [AccessRequest(str(i), "write" if i % 3 == 0 else "read", address, 64)
                     for i, address in enumerate(random_addresses)],
        "row_locality": [AccessRequest(str(i), "read", address, 64) for i, address in enumerate(locality)],
        "activation_pressure": [AccessRequest(str(i), "read", (i % 64)*256*32 + (i//64)*row_span, 32)
                                for i in range(256)],
        "mixed_same_pc": [AccessRequest(str(i), "read" if i % 2 == 0 else "write",
                            (i % 64)*256*32 + (i//64)*32*256*64, 32) for i in range(256)],
        "hot_row_conflicts": [AccessRequest(str(i), "read", (i % 2)*row_span, 32) for i in range(256)],
    }


def quantile(values, fraction):
    values = sorted(values)
    return values[min(len(values)-1, int((len(values)-1)*fraction))]


def micro(variant, name, requests):
    config = configuration(variant)
    core = DramCore(config, capture_details=False)
    started = time.perf_counter()
    results = core.submit_many(requests)
    duration = max(r.completion_ns for r in results) - min(r.arrival_ns for r in results)
    counters = {key: sum(r.counters.get(key, 0) for r in results) for key in (
        "activate_commands", "precharge_commands", "read_commands", "write_commands", "refresh_commands",
        "row_hits", "row_misses", "row_conflicts", "queue_wait_ns", "scheduler_reordered_requests", "scheduler_batches")}
    latencies = [r.completion_ns-r.arrival_ns for r in results]
    return {"variant": variant, "workload": name, "requests": len(requests),
        "logical_bytes": sum(r.logical_bytes for r in results), "duration_ns": duration,
        "effective_gb_s": sum(r.logical_bytes for r in results)/duration,
        "request_latency_ns": {"p50": quantile(latencies, .5), "p95": quantile(latencies, .95), "p99": quantile(latencies, .99)},
        "counters": counters, "wall_seconds": time.perf_counter()-started,
        "compiled_requests": sum(bool(r.counters.get("compiled_hbm_float64")) for r in results)}


def serving(variant, architecture, model, prompt, batch, output, capacity_gib=512):
    spec = VARIANTS[variant]
    if spec.get("overrides"):
        raise ValueError("timing ablations are isolated microbenchmarks, not primary serving results")
    measured = {}
    def observe(result):
        measured.update(decode_measurement(result.serving))
    row_id = f"{model}-{prompt}-{batch}-{capacity_gib}-{variant}-{architecture}"
    started = time.perf_counter()
    row = run_case(architecture, row_id=row_id, model_id=model, prompt=prompt, batch=batch, output=output,
        hbm_capacity_bytes=capacity_gib*GIB, hbm_parameter_profile=spec["parameter_profile"],
        hbm_controller=spec.get("controller"), result_observer=observe)
    if not row.get("executed_serving"):
        raise RuntimeError(f"Serving case {row_id} did not execute: {row.get('reason')}")
    row.update(point_result_version=POINT_RESULT_VERSION, memory_execution_contract=EXECUTION_CONTRACT,
        variant=variant, decode_measurement=measured, wall_seconds=time.perf_counter()-started)
    row["decode_core_checks"] = core_checks(row)
    if (row["completed_requests"] != batch or not all(row["decode_core_checks"].values())
            or not all(measured["checks"].values()) or not all(row["placement_verification"].values())):
        raise RuntimeError(f"Serving validation failed: {row_id}")
    return row


def serving_point(point):
    return serving(*point)


def aggregate(output_dir, named_model="qwen2_5-0_5b"):
    micros = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output_dir / "micro").glob("*.json"))]
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output_dir / "serving").glob("*.json"))]
    pairs = {}
    for row in rows:
        params = row["parameters"]
        key = (params["model_id"], params["prompt"], params["batch"], params["hbm_capacity_bytes"], row["variant"])
        pairs.setdefault(key, {})[row["placement"]] = row
    comparisons = []
    for key, pair in pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair["A_hbm_only"], pair["B_all_kv_hbf"]
        comparisons.append(dict(zip(("model", "prompt", "batch", "capacity_bytes", "variant"), key),
            a_tpot_ns=a["tpot_ns"]["p50"], b_tpot_ns=b["tpot_ns"]["p50"],
            b_over_a=b["tpot_ns"]["p50"] / a["tpot_ns"]["p50"],
            faster="A" if a["tpot_ns"]["p50"] < b["tpot_ns"]["p50"] else "B"))
    expected_micro = {(v, w) for v in VARIANTS for w in workloads()}
    expected_cases = {(v, "small", 1024, 4, 512*GIB, a)
        for v, spec in VARIANTS.items() if not spec.get("overrides")
        for a in ("A_hbm_only", "B_all_kv_hbf")}
    expected_cases.update((v, model, prompt, batch, capacity, a)
        for v in ("reference", "fast", "slow")
        for model, prompt, batch, capacity in ((named_model, 128, 1, 512*GIB), ("small", 1024, 4, 1024*GIB))
        for a in ("A_hbm_only", "B_all_kv_hbf"))
    actual_cases = {(r["variant"], r["parameters"]["model_id"], r["parameters"]["prompt"],
        r["parameters"]["batch"], r["parameters"]["hbm_capacity_bytes"], r["placement"]) for r in rows}
    micro_lookup = {(r["variant"], r["workload"]): r for r in micros}
    checks = {
        "all_planned_microbenchmarks_executed": expected_micro <= set(micro_lookup),
        "all_planned_serving_cases_executed": expected_cases <= actual_cases,
        "all_serving_cases_use_current_execution": bool(rows) and all(r.get("point_result_version") == POINT_RESULT_VERSION for r in rows),
        "all_serving_cases_have_real_prefill_decode_and_core_io": bool(rows) and all(
            all(r["decode_core_checks"].values()) and all(r["decode_measurement"]["checks"].values()) for r in rows),
        "all_comparisons_have_both_architectures": len(comparisons)*2 == len(rows),
    }
    def change(variant, workload):
        if (variant, workload) not in micro_lookup or ("reference", workload) not in micro_lookup:
            return None
        return (micro_lookup[variant, workload]["effective_gb_s"] /
                micro_lookup["reference", workload]["effective_gb_s"] - 1)*100
    findings = {"no_refresh_sequential_read_percent_vs_reference": change("no_refresh", "sequential_read"),
        "no_activation_gates_pressure_percent_vs_reference": change("no_activation_gates", "activation_pressure"),
        "no_rw_recovery_mixed_same_pc_percent_vs_reference": change("no_rw_recovery", "mixed_same_pc"),
        "row_hit_first_conflicts_percent_vs_reference": change("row_hit_first", "hot_row_conflicts"),
        "read_write_batch_mixed_same_pc_percent_vs_reference": change("read_write_batch", "mixed_same_pc")}
    configurations = {k: asdict(configuration(k)) for k in VARIANTS}
    result = {"point_result_version": POINT_RESULT_VERSION, "memory_execution_contract": EXECUTION_CONTRACT,
        "status": "passed" if all(checks.values()) else "incomplete", "checks": checks,
        "planned_micro_points": len(expected_micro), "planned_serving_points": len(expected_cases),
        "findings": findings,
        "vendor_calibrated": False, "scope": "engineering envelopes and controller sensitivity; no commercial SKU accuracy claim",
        "timing_profiles": timing_profile_catalog(), "configurations": configurations,
        "microbenchmarks": micros, "serving_comparisons": comparisons,
        "serving_cases": [{"id": r["id"], "variant": r["variant"], "parameters": r["parameters"],
            "placement": r["placement"], "tpot_ns": r["tpot_ns"], "decode_tps": r["decode_measurement"]["node_decode_tps"],
            "core_checks": r["decode_core_checks"], "wall_seconds": r["wall_seconds"]} for r in rows]}
    save(output_dir / "results.json", result)
    lines = ["# HBM 基本时序、参数范围与控制器敏感性", "", f"执行版本：{POINT_RESULT_VERSION}；范围完成状态：{result['status']}。",
        f"计划/已执行：微基准 {len(expected_micro)}/{len(micros)}，完整服务 {len(expected_cases)}/{len(rows)}。只有全部计划点执行并通过Core/Decode检查才标为passed。",
        "", "三档参数是公开资料不足时的工程假设范围，未标定厂商器件；不代表统计置信区间。",
        "主参考保持 HBM3E 32 Gb × 16H、512 GiB、0.625 ns、读写方向预算各1600 GB/s。所有微基准使用相同主机地址与到达时间。",
        "none 对照明确省略若干时序与刷新；它用于解释修正的影响，不作为真实硬件基准。",
        "", "## 已观测的影响", "",
        *[f"- {label}：相对参考组的有效带宽变化 {findings[key]:+.2f}%。" for key, label in (
            ("no_refresh_sequential_read_percent_vs_reference", "关闭刷新，顺序读"),
            ("no_activation_gates_pressure_percent_vs_reference", "关闭激活频率约束，密集开行"),
            ("no_rw_recovery_mixed_same_pc_percent_vs_reference", "关闭读写恢复门，同PC混合读写"),
            ("row_hit_first_conflicts_percent_vs_reference", "优先已开行，热点行冲突"),
            ("read_write_batch_mixed_same_pc_percent_vs_reference", "读写分组，同PC混合读写")) if findings[key] is not None],
        "这些探针用于分离机制影响，不能直接当作LLM加速百分比；完整服务结果单独列出。",
        "", "## 参数与来源", "",
        "以下为配置中的时间下限，单位ns；发令仍按CK/半CK边界对齐。nCK输入随实际时钟转换，不把周期数当作ns。完整输入单位、公式、依据、人工覆盖和来源均保存在results.json的configurations元数据中。",
        "", "| 参数 | 快档 | 参考档 | 慢档 |", "| --- | ---: | ---: | ---: |"]
    for symbol, field in (("tRCDRD", "activate_to_read_ns"), ("tRAS", "minimum_active_ns"),
            ("tRP", "precharge_to_activate_ns"), ("tRC", "activate_cycle_ns"),
            ("tWR", "write_to_precharge_ns"), ("tRTP", "read_to_precharge_ns"),
            ("tRRD_S", "activate_different_group_ns"), ("tRRD_L", "activate_same_group_ns"),
            ("tFAW", "four_activate_window_ns"), ("tWTR_S", "write_to_read_different_group_ns"),
            ("tWTR_L", "write_to_read_same_group_ns"), ("tRTW", "read_to_write_command_ns"),
            ("RL", "read_latency_ns"), ("WL", "write_latency_ns"),
            ("tRFCab", "refresh_cycle_ns"), ("tREFI", "refresh_interval_ns")):
        values = [configurations[name][field] for name in ("fast", "reference", "slow")]
        lines.append(f"| {symbol} | {values[0]:g} | {values[1]:g} | {values[2]:g} |")
    lines.extend(["",
        "JEDEC JESD238提供规则、单位和部分编码范围；主组织32Gb×16H的tRFC仍为TBD，多项AC最小值没有给出。参考档不是JEDEC保证值，快/慢档也不是厂商上下限。公开Micron、Samsung、SK hynix产品资料未提供与此组织对应的完整AC表，因此vendor_calibrated=false。",
        "时序预设为hbm3e_timing_fast、hbm3e_timing_reference、hbm3e_timing_slow；研究入口另外保持1600GB/s方向预算。默认控制器仍为bank_interleaved、fcfs、device队列深度64；候选策略不因局部探针加速就替换默认。",
        "", "## 完整服务比较", "", "先执行真实 Prefill，再从提交事件计算独立 Decode；A 为全 HBM，B 为 KV 使用 HBF。", "",
        "| 模型/输入/批量/容量 | 参数或策略 | A TPOT µs | B TPOT µs | B/A | 更快 |",
        "| --- | --- | ---: | ---: | ---: | --- |"])
    for r in comparisons:
        lines.append(f"| {r['model']}/{r['prompt']}/{r['batch']}/{r['capacity_bytes']//GIB} GiB | {r['variant']} | {r['a_tpot_ns']/1000:.4f} | {r['b_tpot_ns']/1000:.4f} | {r['b_over_a']:.4f} | {r['faster']} |")
    lines.extend(["", "## 独立访存", "", "延迟为各请求从到达到完成的时间，包含排队；GB/s 为全部逻辑字节除以整个完成窗口。", "",
        "| 负载 | 参数或策略 | GB/s | P99 ns | ACT | REF | 重排请求 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for r in micros:
        c = r["counters"]
        lines.append(f"| {r['workload']} | {r['variant']} | {r['effective_gb_s']:.3f} | {r['request_latency_ns']['p99']:.3f} | {c['activate_commands']} | {c['refresh_commands']} | {c['scheduler_reordered_requests']} |")
    lines.extend(["", "## 解释边界", "",
        "- 不同队列范围的预算不同：device 是全设备请求数；channel 是每物理通道请求数。512 GiB配置有128个物理通道，因此 channel_depth1 的总名额上限为128，须与 device_depth128 一起比较。跨通道请求同时占用所有涉及通道的名额。",
        "- 调度作用于已有的聚合请求和有限窗口，不是完整逐命令 FR-FCFS 控制器；不跨尚未生成的算子请求重排。",
        "- 快/慢档同时改变多个未知时序；单项消融用于分离刷新、激活和恢复约束的作用。",
        "- HBF仍为既有逻辑页服务模型；本次未增加REFpb、RFM、低功耗、初始化或错误注入。",
        "- 方案排序仅适用于实际列出的模型、上下文、批量和参数范围；不替代完整20组研究矩阵。", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/hbm_sensitivity_20261015")
    parser.add_argument("--stage", choices=("micro", "serving", "named", "capacity", "report", "all"), default="all")
    parser.add_argument("--variants", default="")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--named-model", default="qwen2_5-0_5b")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    selected = args.variants.split(",") if args.variants else list(VARIANTS)
    if any(v not in VARIANTS for v in selected):
        parser.error("unknown variant")
    if args.workloads and any(w not in workloads() for w in args.workloads.split(",")):
        parser.error("unknown workload")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be 1..4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("micro", "all"):
        for variant in selected:
            for name, requests in workloads().items():
                if args.workloads and name not in args.workloads.split(","):
                    continue
                row = micro(variant, name, requests)
                save(args.output_dir / "micro" / f"{variant}-{name}.json", row)
                print(json.dumps({"micro": variant+"/"+name, "gb_s": row["effective_gb_s"], "wall_s": row["wall_seconds"]}), flush=True)
    if args.stage in ("serving", "named", "capacity", "all"):
        cases = []
        if args.stage in ("serving", "all"):
            cases.extend((variant, "small", 1024, 4, 8, 512) for variant in selected if not VARIANTS[variant].get("overrides"))
        if args.stage in ("named", "all"):
            cases.extend((variant, args.named_model, 128, 1, 4, 512) for variant in ("reference", "fast", "slow") if variant in selected)
        if args.stage in ("capacity", "all"):
            cases.extend((variant, "small", 1024, 4, 8, 1024) for variant in ("reference", "fast", "slow") if variant in selected)
        points = [(variant, architecture, model, prompt, batch, output, capacity)
                  for variant, model, prompt, batch, output, capacity in cases
                  for architecture in ("A_hbm_only", "B_all_kv_hbf")]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(serving_point, point) for point in points]
            for future in as_completed(futures):
                row = future.result()
                save(args.output_dir / "serving" / (row["id"]+".json"), row)
                print(json.dumps({"serving": row["id"], "tpot_ns": row["tpot_ns"]["p50"], "wall_s": row["wall_seconds"]}), flush=True)
    report = aggregate(args.output_dir, args.named_model)
    print(json.dumps({"micro_points": len(report["microbenchmarks"]), "serving_points": len(report["serving_cases"])}), flush=True)


if __name__ == "__main__":
    main()
