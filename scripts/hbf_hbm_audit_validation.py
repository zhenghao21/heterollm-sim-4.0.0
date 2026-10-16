"""Re-run bounded A/B, decode and capacity cases after the model audit fixes.

Uses the normal full-operator serving runner and its actual Core observer.
This is a synthetic regression matrix, not the complete named-model study.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from calibrated_hbf_profiles import _hbm_physical_config, _hbf_core
from hbf_hbm_serving_study import POINT_RESULT_VERSION, run_case
from hbf_hbm_decode_study import decode_measurement, core_checks, EXECUTION_CONTRACT
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for gib in (512, 1024):
        for architecture in ("A_hbm_only", "B_all_kv_hbf"):
            measured = {}
            def observe(result):
                measured.update(decode_measurement(result.serving))
            row = run_case(architecture, row_id=f"audit-{gib}GiB-{architecture}",
                model_id="small", prompt=128, batch=1, output=8,
                hbm_capacity_bytes=gib * 1024**3, result_observer=observe)
            row.update(point_result_version=POINT_RESULT_VERSION,
                       memory_execution_contract=EXECUTION_CONTRACT,
                       decode_measurement=measured)
            if not row.get("executed_serving"):
                save(output_dir / (row["id"] + ".json"), row)
                raise RuntimeError("Validation case did not execute: " + row["id"])
            row["decode_core_checks"] = core_checks(row)
            save(output_dir / (row["id"] + ".json"), row)
            rows.append(row)
            print(json.dumps({"completed": row["id"], "tpot_ns": row["tpot_ns"]["p50"],
                              "decode_tps": measured["node_decode_tps"]}), flush=True)

    configs, micros = {}, []
    for gib in (512, 1024):
        config = _hbm_physical_config(capacity_bytes=gib * 1024**3,
                                      read_gb_s=1600., write_gb_s=1600.)
        configs[str(gib)] = asdict(config)
        result = DramCore(config, capture_details=False).submit(
            AccessRequest("cold-read", "read", 0, 64 * 1024**2))
        micros.append({"medium": "HBM", "capacity_gib": gib,
                       "operation": "read", "bytes": result.logical_bytes,
                       "completion_ns": result.completion_ns,
                       "achieved_gb_s": result.actual_bandwidth_gb_s})
    for operation in ("read", "write"):
        core = _hbf_core(capacity_bytes=512 * 1024**3, read_gb_s=1600., write_gb_s=16.,
                         page_bytes=4096, read_latency_ns=4000., program_ns=75000., parallel_units=64,
                         host_granularity_bytes=64, host_channels=16, host_full_duplex=True,
                         read_cache_pages_per_bank=2)
        configs["hbf"] = asdict(core.config)
        result = core.submit(AccessRequest("cold-" + operation, operation, 0, 64 * 1024**2))
        micros.append({"medium": "HBF", "capacity_gib": 512, "operation": operation,
                       "bytes": result.logical_bytes, "completion_ns": result.completion_ns,
                       "achieved_gb_s": result.actual_bandwidth_gb_s,
                       "pages_read": result.pages_read, "pages_programmed": result.pages_programmed,
                       "host_transfer_bytes": result.host_transfer_bytes,
                       "read_cache_hits": result.counters.get("read_cache_hits", 0)})
    checks = {
        "four_serving_cases_completed": len(rows) == 4 and all(r["completed_requests"] == 1 for r in rows),
        "placements_verified": all(all(r["placement_verification"].values()) for r in rows),
        "actual_core_execution_verified": all(all(r["decode_core_checks"].values()) for r in rows),
        "pure_decode_windows_verified": all(all(r["decode_measurement"]["checks"].values()) for r in rows),
        "cache_observations_reach_public_report": all(
            r["resources"]["storage_traffic"].get("read_cache_hits") is not None
            and r["resources"]["storage_traffic"].get("read_cache_misses") is not None
            for r in rows if r["placement"] == "B_all_kv_hbf"),
        "same_hbm_clock_across_capacities": configs["512"]["command_clock_ns"] == configs["1024"]["command_clock_ns"] == 0.625,
        "capacity_changes_stack_count": (configs["512"]["stacks"], configs["1024"]["stacks"]) == (8, 16),
        "hbf_new_granularity": (configs["hbf"]["page_bytes"], configs["hbf"]["host_granularity_bytes"]) == (4096, 64),
        "hbf_duplex_cache_enabled": configs["hbf"]["host_full_duplex"] and configs["hbf"]["read_cache_pages_per_bank"] == 2,
    }
    report = {"point_result_version": POINT_RESULT_VERSION, "memory_execution_contract": EXECUTION_CONTRACT,
              "status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "scope": "4 synthetic full-operator serving cases, real decode windows, 512/1024 GiB inventories and 64 MiB cold-stream probes",
              "external_prediction_accuracy_validated": False, "full_research_matrix_rerun": False,
              "configurations": configs, "microbenchmarks": micros, "rows": rows}
    save(output_dir / "results.json", report)
    lines = ["# HBM / HBF 整改后验证", "", f"执行版本：{POINT_RESULT_VERSION}。检查：{report['status']}。", "",
             "以下为当前代码真实运行的仿真结果。服务使用小型合成模型、128输入token、8输出token、单请求；不能替代8B/70B长上下文完整研究。", "",
             "| HBM容量 | 方案 | TPOT（µs） | 独立Decode TPS |", "| --- | --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['parameters']['hbm_capacity_bytes'] // 1024**3} GiB | {row['placement']} | {row['tpot_ns']['p50']/1000:.6f} | {row['decode_measurement']['node_decode_tps']:.3f} |")
    lines.extend(["", "| 介质 | 容量 | 64 MiB冷顺序操作 | 有效GB/s |", "| --- | --- | --- | ---: |"])
    for row in micros:
        lines.append(f"| {row['medium']} | {row['capacity_gib']} GiB | {row['operation']} | {row['achieved_gb_s']:.6f} |")
    lines.extend(["", "HBM两种容量固定0.625 ns，8/16 stacks；方向预算均为1600 GB/s。HBF为4 KiB页、64 B主机计费、64内部单元、16主机通道、全双工、每bank两页读缓存。HBF接口1600/16 GB/s与介质理论65.536/3.495253 GB/s分别列示。", "",
                  "服务中的HBF采用逻辑页服务分析，未计主机重映射、顺序写和自动擦除；OCP完整页协议行为由独立回归验证。普通写完成不采用主机早确认。", "",
                  "原始逐点文件与results.json保留配置、实际Core调用和检查项。未重新运行全部20组研究或替换历史图表。", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks, "output": str(output_dir)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/hbm_hbf_remediation_20261015")
    raise SystemExit(run(parser.parse_args().output_dir))
