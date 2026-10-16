"""Full small-Dense Decode sensitivity using the public hardware catalog.

The scripts select catalog profiles; no physical config is generated or patched
here. All three variants execute real prefill and the same Decode workload.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from hbf_hbm_decode_study import core_checks, decode_measurement
from hbf_hbm_experiments import group
from hbf_hbm_serving_study import run_case
from heterollm_sim.memory_types import DRAM_BANK_GROUP_TIMING_FIELDS

PROFILES = (
    "hbm3e_reference_flat",
    "hbm3e_reference",
    "hbm3e_reference_synthetic_timing",
)
ARCHITECTURES = ("A_hbm_only", "B_all_kv_hbf")
CONTEXTS = (32, 128, 512)


def run_point(point):
    profile, context, architecture, output = point
    measured = {}
    row = run_case(architecture, row_id=f"bank-group/{profile}/{context}/{architecture}",
        model_id="small", prompt=context, batch=1, output=output,
        hbm_timing_profile=profile, hbm_parameter_profile="none",
        result_observer=lambda result: measured.update(decode_measurement(result.serving)))
    if not row.get("executed_serving"):
        raise RuntimeError("Bank-group point was rejected: " + str(row))
    row["decode_measurement"] = measured
    row["decode_core_checks"] = core_checks(row)
    row["hardware_profile"] = profile
    configs = row["core_execution_audit"]["executed_physical_configurations"]
    row["executed_hbm_parameters"] = {
        owner: item["physical_config"] for owner, item in configs.items()
        if item["physical_config"]["kind"] == "HBM"
    }
    return row


def summarize(rows, *, contexts=CONTEXTS, output=8):
    lookup = {(r["hardware_profile"], r["prompt_tokens"], r["placement"]): r for r in rows}
    expected = {(p, c, a) for p in PROFILES for c in contexts for a in ARCHITECTURES}
    comparisons = []
    def non_geometry(row):
        return {owner: {key: value for key, value in config.items()
                        if key not in {"metadata", "bank_groups_per_rank", "banks_per_group"}}
                for owner, config in row["executed_hbm_parameters"].items()}
    for context in contexts:
        for architecture in ARCHITECTURES:
            if not all((p, context, architecture) in lookup for p in PROFILES):
                continue
            baseline, control, enabled = [lookup[p, context, architecture] for p in PROFILES]
            rate = lambda row: row["decode_measurement"]["node_decode_tps"]
            comparisons.append({
                "prompt_tokens": context, "placement": architecture,
                "baseline_decode_tps": rate(baseline), "geometry_control_decode_tps": rate(control),
                "bank_group_decode_tps": rate(enabled),
                "timing_over_control_tps": rate(enabled) / rate(control),
                "geometry_over_baseline_tps": rate(control) / rate(baseline),
                "same_inventory": baseline["fixed_inventory"] == control["fixed_inventory"] == enabled["fixed_inventory"],
                "same_non_geometry_parameters": non_geometry(baseline) == non_geometry(control),
                "row_ids": [r["id"] for r in (baseline, control, enabled)],
            })
    def hbm_records(row):
        return [o for o in row["core_execution_audit"]["owners"] if o["memory_kind"] == "HBM"]
    checks = {
        "all_requested_points_executed": set(lookup) == expected and len(rows) == len(expected),
        "actual_core_paths_verified": bool(rows) and all(all(r["decode_core_checks"].values()) for r in rows),
        "real_prefill_and_pure_decode": bool(rows) and all(all(r["decode_measurement"]["checks"].values()) for r in rows),
        "fixed_output_length": all(r["visible_output_tokens"] == output and r["completed_requests"] == 1 for r in rows),
        "same_inventory": bool(comparisons) and all(c["same_inventory"] for c in comparisons),
        "geometry_control_preserves_other_inputs": bool(comparisons) and all(c["same_non_geometry_parameters"] for c in comparisons),
        "all_hbm_transactions_use_jedec_control": bool(rows) and all(
            hbm_records(r) and all(o.get("jedec_control_transactions") == o["transactions"] > 0 for o in hbm_records(r))
            for r in rows),
        "group_timing_matches_executed_configuration": bool(rows) and all(
            o["bank_group_timing_transactions"] == (o["transactions"] if any(
                r["executed_hbm_parameters"][o["physical_owner"]][name] > 0
                for name in DRAM_BANK_GROUP_TIMING_FIELDS) else 0)
            for r in rows for o in hbm_records(r)),
        "actual_parameters_recorded": bool(rows) and all(r["executed_hbm_parameters"] and all(
            all(name in config for name in DRAM_BANK_GROUP_TIMING_FIELDS)
            for config in r["executed_hbm_parameters"].values()) for r in rows),
    }
    return group("20_bank_group_decode", "统一参数入口下的 Bank group Decode 对照", rows, checks,
        ["同一小Dense完整服务、真实prefill后Decode；平坦分组消融、HBM3E参考组织及合法合成时序对照"],
        ["小Dense为1层256 hidden/512 FFN的合成模型，不是8B/70B性能预测。",
         "标准CCD约束和显式合成组时序均来自统一目录，具体来源分开记录，未校准厂商器件。",
         "只改变分组和显式时序；总bank数量、容量、带宽、GPU、模型及工作量保持一致。",
         "启用组时序时使用精确逐burst路径，大模型的相同矩阵本轮未执行。",
         "参考组织按JEDEC可选密度/层数选择以覆盖用户容量；未核实为目标厂商器件，未知项见metadata。",
         "平坦分组是有意设置的合成消融，不是JEDEC器件组织；所有分支均执行已知时序硬下限。",
         "未知阵列门控的零值只表示未模拟，不表示JEDEC允许零延迟；实际字段及未知项随结果保存。"],
        comparisons=comparisons, contexts=list(contexts), output_tokens=output,
        hardware_profiles=list(PROFILES),
        parameter_entrypoint="physical_presets.resolve_physical_config -> physical_contract -> scenario -> DramCore/NandCore")


def run_bank_group_study(*, workers=2, contexts=CONTEXTS, output=8, point_dir=None):
    points = [(p, c, a, output) for c in contexts for p in PROFILES for a in ARCHITECTURES]
    rows = []
    if point_dir:
        point_dir = Path(point_dir)
        point_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(run_point, points):
            rows.append(row)
            if point_dir:
                (point_dir / (row["id"].replace("/", "__") + ".json")).write_text(
                    json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps({"completed": row["id"], "decode_tps": row["decode_measurement"]["node_decode_tps"],
                              "wall_seconds": row["wall_seconds"]}, ensure_ascii=False), flush=True)
    return summarize(rows, contexts=contexts, output=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--contexts", default="32,128,512")
    parser.add_argument("--output-tokens", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.output_tokens < 2:
        parser.error("workers must be 1..8 and output-tokens >=2")
    result = run_bank_group_study(workers=args.workers, contexts=tuple(map(int, args.contexts.split(","))),
        output=args.output_tokens, point_dir=args.output.parent / "bank_group_points")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "output": str(args.output)}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
