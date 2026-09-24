"""Checkpointed multi-load experiments using the ordinary simulator (no extrapolation)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import qwen38_memory_scenario as q
from tools import memory_tier_sweep as sweep
from heterollm_sim.config import scenario_from_dict

WORKLOADS = (
    ("L1_short_low", 128, 32, 1), ("L2_context_low", 2048, 128, 1),
    ("L3_context_batch", 2048, 128, 4), ("L4_long_context", 8192, 128, 4),
    ("L5_ultra_context", 32768, 128, 16), ("L6_long_decode", 2048, 512, 16),
    ("L7_prefill_heavy", 32768, 32, 4), ("L8_short_high_batch", 128, 128, 32),
)
GROUPS = (
    {"id": "I1", "question": "KV 位于 HBM / HBF 的影响（放置对照，非纯互联）", "kind": "placement",
     "cases": ["kv_hbm", "kv_hbf"], "values": ["HBM", "HBF"]},
    {"id": "I2", "question": "双 DRAM 独立 / 共享 PHY-NoC", "kind": "topology",
     "cases": ["dram_independent", "dram_shared"], "values": ["independent", "shared"]},
    {"id": "I3", "question": "启用首层 MLP 冷加载 CIM 后，独立 / 共享 PHY-NoC", "kind": "topology",
     "cases": ["cim_independent", "cim_shared"], "values": ["independent", "shared"]},
    {"id": "I4", "question": "远端 Flash 权重路径与 HBM 路径独立 / 共享", "kind": "flash_topology",
     "cases": ["flash_independent", "flash_shared"], "values": ["independent", "shared"]},
    {"id": "P1", "question": "高读带宽 HBF 的读延迟敏感性", "kind": "hbf_latency",
     "cases": [f"hbf_lat_{n}" for n in (1000, 4000, 10000, 20000)],
     "values": [1000, 4000, 10000, 20000], "unit": "ns", "direction": "nondecreasing"},
    {"id": "P2", "question": "高读带宽 HBF 的未完成请求数上限敏感性", "kind": "hbf_outstanding",
     "cases": ["hbf_lat_10000"] + [f"hbf_q_{n}" for n in (256, 1024, 4096, 16384)],
     "values": [32, 256, 1024, 4096, 16384], "unit": "requests", "direction": "nonincreasing"},
    {"id": "P3", "question": "dram0 读延迟敏感性（dram1 保持 100 ns）", "kind": "dram_latency",
     "cases": ["dram_lat_40", "dram_lat_60", "dram_independent", "dram_lat_200"],
     "values": [40, 60, 100, 200], "unit": "ns", "direction": "nondecreasing"},
    {"id": "P4", "question": "CIM 转换暂存区容量敏感性（固定 tile 和单槽生命周期）", "kind": "scratch",
     "cases": ["cim_scratch_262144", "cim_scratch_524288", "cim_independent"],
     "values": [262144, 524288, 1048576], "unit": "B", "direction": "nonincreasing"},
)
DEFAULT_OUTPUT = q.OUTPUT_ROOT / "multiworkload-matrix-v2"


def profile_update(scenario, component_id, **values):
    component = scenario.hardware.get_component(component_id)
    profiles = {kind: dict(items) for kind, items in scenario.component_profiles.items()}
    kind = scenario.component_profile_kind(component)
    profiles[kind][component.cost_profile_id] = replace(
        scenario.resolve_component_profile(component_id), **values)
    # ScenarioConfig validates aliases on construction: update both atomically.
    metadata = {**component.metadata, **{k: v for k, v in values.items()
        if component.normalized_kind == "hbf" and k in ("read_latency_ns", "max_outstanding_requests")}}
    media_rates = {field.replace("_gb_s", "_gbps"): value * 8 for field, value in values.items()
                   if component.normalized_kind == "hbf" and field in ("read_bandwidth_gb_s", "write_bandwidth_gb_s")}
    hardware = replace(scenario.hardware, components=tuple(replace(c, metadata=metadata, **media_rates)
        if c.component_id == component_id else c for c in scenario.hardware.components))
    return replace(scenario, component_profiles=profiles, hardware=hardware)


def shared_fabric(scenario, flash=False):
    """Change only physical sharing, never capacities, rates, profiles or mapping."""
    selected = {"gpu-hbf0", "gpu-hbm0", "gpu-hbm1", "gpu-hbm2"} if flash else {"vertical_dram0", "vertical_dram1", "soc_cim_noc"}
    owner = "gpu0.shared_memory_fabric" if flash else "stack0.shared_phy_noc"
    hardware = scenario.hardware
    owners = dict(hardware.metadata.get("physical_resource_owners", {}))
    for link in hardware.links:
        if link.link_id in selected:
            owners["link." + link.link_id] = owner
    links = tuple(replace(link, metadata={**link.metadata, "physical_resource_owner": owner})
                  if link.link_id in selected else link for link in hardware.links)
    return replace(scenario, hardware=replace(hardware, links=links,
        metadata={**hardware.metadata, "physical_resource_owners": owners, "phy_noc_mode": "shared"}))


def build_case(model, variant, workload):
    _, prompt, output, batch = workload
    if variant.startswith("kv_"):
        base_name, component = "hbf_active_kv", "hbf0"
    elif variant.startswith("flash_"):
        base_name, component = "hbf_media_remote_weights", "hbf0"
    elif variant.startswith("hbf_"):
        base_name, component = "hbf_active_weights", "hbf0"
    elif variant.startswith("cim_"):
        base_name, component = "dual_dram_cim_tiled", "dram0"
    else:
        base_name, component = "dual_dram_only", "dram0"
    base = q.build_scenario(model, base_name, prompt_tokens=prompt, output_tokens=output, batch=batch)
    # Reserve generation headroom before admission; excess requests queue, not
    # disappear. The same policy is used by every point, with no hardware change.
    base = replace(base, placement=replace(base.placement,
        kv_policy=replace(base.placement.kv_policy, allocation_policy="eager")))
    if variant.startswith("flash_"):
        base = replace(base, hardware=replace(base.hardware, metadata={**base.hardware.metadata,
            "physical_resource_owners": dict(base.hardware.metadata.get("physical_resource_owners", {}))}))
    if variant.startswith("hbf_"):
        # An explicitly optimistic active-memory interface, not a NAND page model.
        base = profile_update(base, "hbf0", bandwidth_gb_s=1600.0,
                              read_bandwidth_gb_s=1600.0, write_bandwidth_gb_s=4.0,
                              read_latency_ns=10000.0, max_outstanding_requests=32)
        base = replace(base, hardware=replace(base.hardware,
            components=tuple(replace(c, read_bandwidth_gbps=12800.0,
                ports=tuple(replace(p, bandwidth_gbps=12800.0) for p in c.ports),
                metadata={**c.metadata, "read_latency_ns": 10000.0}) if c.component_id == "hbf0"
                else replace(c, ports=tuple(replace(p, bandwidth_gbps=12800.0)
                    if p.port_id == "hbf" else p for p in c.ports)) if c.component_id == "gpu0"
                else c for c in base.hardware.components),
            links=tuple(replace(link, bandwidth_gbps=12800.0) if link.link_id == "gpu-hbf0"
                        else link for link in base.hardware.links)))
    scenario, _ = sweep.fixed_reference(base, component)
    if variant == "kv_hbm":
        options = deepcopy(scenario.placement.metadata["control_plane"]["policy"]["options"])
        options["kv_cache_target"] = "hbm0"
        if "kv_layer_targets" in options:
            options["kv_layer_targets"] = {layer: "hbm0" for layer in options["kv_layer_targets"]}
        scenario = q._constraints(scenario, options)
        base_name = "hbf_idle"  # HBF is deliberately idle, not missing coverage.
    if variant.endswith("_shared"):
        flash = variant.startswith("flash_")
        scenario = shared_fabric(scenario, flash=flash)
        if not flash:
            base_name += "_shared"
    if variant.startswith("hbf_lat_"):
        latency = int(variant.rsplit("_", 1)[1])
        scenario = profile_update(scenario, "hbf0", read_latency_ns=float(latency))
        scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
            replace(c, metadata={**c.metadata, "read_latency_ns": float(latency)})
            if c.component_id == "hbf0" else c for c in scenario.hardware.components)))
    if variant.startswith("hbf_q_"):
        count = int(variant.rsplit("_", 1)[1])
        scenario = profile_update(scenario, "hbf0", max_outstanding_requests=count)
        scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
            replace(c, metadata={**c.metadata, "max_outstanding_requests": count})
            if c.component_id == "hbf0" else c for c in scenario.hardware.components)))
    if variant.startswith("dram_lat_"):
        scenario = profile_update(scenario, "dram0", read_latency_ns=float(variant.rsplit("_", 1)[1]))
    if variant.startswith("cim_scratch_"):
        scenario = profile_update(scenario, "cim0", conversion_scratch_capacity_bytes=int(variant.rsplit("_", 1)[1]))
    return scenario, base_name


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [p for key in sorted(left.keys() | right.keys()) for p in
                (differences(left[key], right[key], path + "/" + str(key))
                 if key in left and key in right else [path + "/" + str(key)])]
    if isinstance(left, list) and isinstance(right, list):
        for key in ("component_id", "link_id"):
            if left and all(isinstance(v, dict) and key in v for v in left + right):
                a, b = ({v[key]: v for v in side} for side in (left, right))
                if len(a) == len(left) and len(b) == len(right):
                    return differences(a, b, path)
        if len(left) == len(right):
            return [p for i, (a, b) in enumerate(zip(left, right))
                    for p in differences(a, b, path + "/" + str(i))]
    return [] if left == right else [path]


def allowed_change(path, kind):
    if kind in ("topology", "flash_topology"):
        links = ("vertical_dram0", "vertical_dram1", "soc_cim_noc") if kind == "topology" else ("gpu-hbf0", "gpu-hbm0", "gpu-hbm1", "gpu-hbm2")
        return path in ["/hardware/links/" + link + "/metadata/physical_resource_owner" for link in links] or \
               path in ["/hardware/metadata/physical_resource_owners/link." + link for link in links] or \
               path == "/hardware/metadata/phy_noc_mode"
    if kind == "placement":
        return path.startswith("/placement/metadata/control_plane/policy/options/kv_layer_targets/") or \
               path == "/placement/metadata/control_plane/policy/options/kv_cache_target"
    field = {"hbf_latency": "read_latency_ns", "dram_latency": "read_latency_ns",
             "hbf_outstanding": "max_outstanding_requests", "scratch": "conversion_scratch_capacity_bytes"}[kind]
    # Exact owners/profile ids are checked, not arbitrary same-named fields elsewhere.
    profile = {"hbf_latency": "host_memory/example-hbf-memory", "hbf_outstanding": "host_memory/example-hbf-memory",
               "dram_latency": "host_memory/example-dram0", "scratch": "cim/legacy-cim"}[kind]
    return path == "/profiles/components/" + profile + "/" + field or \
           (kind.startswith("hbf_") and path == "/hardware/components/hbf0/metadata/" + field)


def actual_mapping_matches(cells, kind):
    observations = [c["observation"] for c in cells]
    if any(a[key] != observations[0][key] for a in observations[1:]
           for key in ("model_sha256", "workload_sha256")):
        return False
    mappings = [sweep.placement_mapping(o["placement"]) for o in observations]
    if kind == "placement":
        allowed = ("/tensor_to_component/kv_cache", "/memory_tiers/kv_cache_component")
        return all(p in allowed or p.startswith("/memory_tiers/kv_layer_components/")
                   for m in mappings[1:] for p in differences(mappings[0], m))
    return all(m == mappings[0] for m in mappings[1:])


def source_identity():
    return {"core": q.source_identity(), "matrix_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def run_cell(variant, workload, out):
    started = time.monotonic()
    cell_id = workload[0] + "__" + variant
    result = {"id": cell_id, "variant": variant, "workload": dict(zip(
        ("id", "prompt_tokens", "output_tokens", "batch"), workload)), "validation_status": "UNVALIDATED"}
    source = source_identity()
    try:
        model, _ = q.load_model()
        scenario, coverage_name = build_case(model, variant, workload)
        authored = q.scenario_payload(scenario)
        if q.scenario_payload(scenario_from_dict(authored)) != authored:
            raise ValueError("config_roundtrip_mismatch")
        compact = {k: v for k, v in authored.items() if k != "model"}
        compact["model_config_sha256"] = q.stable_hash(authored["model"])
        write_json(out / "configs" / (cell_id + ".json"), compact)
        result["config_sha256"] = q.stable_hash(compact)
        expected = sweep.placement_mapping(sweep.solve_placement(scenario))
        result["planned_mapping_sha256"] = q.stable_hash(expected)
        observation = q.execute_scenario(scenario)
        result["observation"] = observation
        checks = sweep.validate_observation(observation)
        coverage = sweep.scenario_coverage(coverage_name, observation)
        checks.append(sweep.coverage_check(coverage_name, coverage))
        sweep.require_fixed_mapping(expected, sweep.placement_mapping(observation["placement"]))
        checks.append(sweep.check("declared_workload", "PASS" if
            observation["expected_requests"] == workload[3] and observation["expected_output_tokens"] == workload[2] * workload[3]
            else "FAIL", result["workload"]))
        if variant == "kv_hbm":
            activity = sweep.component_activity(observation, "hbf0")
            checks.append(sweep.check("unused_hbf_has_no_free_service", "PASS" if activity and
                activity["bytes"] == activity["service_ns"] == 0 else "FAIL", activity))
        if variant.startswith("cim_"):
            profile = scenario.resolve_component_profile("cim0")
            audits = [b["audit"] for b in observation["cim_weight_conversion_batches"] if b["audit"].get("tile_count", 0)]
            safe = bool(audits) and all(a["array_peak_bytes"] <= profile.weight_capacity_bytes and
                a["scratch_peak_bytes"] <= profile.conversion_scratch_capacity_bytes and a["no_free_traffic"] and
                a["packed_read_bytes"] == a["packed_weight_payload_read_bytes"] + a["packed_weight_metadata_read_bytes"] and
                a["scratch_lifetime"] == "incoming_through_output_atomic_scratch_slot" for a in audits)
            checks.append(sweep.check("tiled_capacity_and_traffic_contract", "PASS" if safe else "FAIL",
                "Checks reported tile peaks and cold/single-slot contract, not independent silicon or overlap measurements."))
        result.update(status="SIMULATED" if all(c["status"] == "PASS" for c in checks) else "CHECK_FAILED",
                      observation=observation, checks=checks, coverage=coverage)
    except Exception as exc:
        result.update(status="BLOCKED" if isinstance(exc, q.UnsupportedScenario) else "FAILED",
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    result["source_sha256"] = q.stable_hash(source)
    result["source_stable"] = source == source_identity()
    if not result["source_stable"]:
        result["status"] = "INVALIDATED"
    write_json(out / "cells" / (cell_id + ".json"), result)
    print(cell_id, result["status"], result["elapsed_s"], flush=True)


def summarize(out, groups, workloads):
    rows = []
    for group in groups:
        for workload in workloads:
            cells, configs = [], []
            for case in group["cases"]:
                stem = workload[0] + "__" + case + ".json"
                path = out / "cells" / stem
                cells.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "PENDING"})
                path = out / "configs" / stem
                configs.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
            valid_indices = [i for i, c in enumerate(cells) if c["status"] == "SIMULATED"]
            valid_cells = [cells[i] for i in valid_indices]
            ready = len(valid_cells) == len(cells)
            changed = [differences(configs[0], c) if configs[0] is not None and c is not None else ["MISSING_CONFIG"] for c in configs[1:]]
            unexpected = sorted({p for change in changed for p in change if not allowed_change(p, group["kind"])})
            fixed_mapping = len(valid_cells) >= 2 and actual_mapping_matches(valid_cells, group["kind"])
            comparable = ready and not unexpected and fixed_mapping and all(changed)
            valid_comparable = len(valid_cells) >= 2 and not unexpected and fixed_mapping and all(changed)
            data = []
            for value, case, cell in zip(group["values"], group["cases"], cells):
                obs = cell.get("observation", {})
                summary = obs.get("summary", {})
                raw_metrics = obs.get("metrics", {})
                metrics_ms = {k.removesuffix("_ns") + "_ms": v / 1e6
                              for k, v in raw_metrics.items() if cell["status"] == "SIMULATED"
                              and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
                data.append({"case": case, "value": value, "status": cell["status"],
                    "config": "configs/" + workload[0] + "__" + case + ".json",
                    "result": "cells/" + workload[0] + "__" + case + ".json",
                    "metrics_ms": metrics_ms,
                    "throughput_tokens_per_s": summary.get("throughput", {}).get("visible_output_tokens_per_s") if cell["status"] == "SIMULATED" else None,
                    "busiest_resource": summary.get("bottleneck_resource"),
                    "resource_accounted_bytes": obs.get("reported_resource_bytes"),
                    "completed_requests": summary.get("completed_requests"), "actual_output_tokens": obs.get("actual_output_tokens"),
                    "error": cell.get("error")})
            trend = None
            if valid_comparable and "direction" in group:
                trend = sweep.monotonicity([{"value": value, "observation": cell["observation"]}
                    for value, cell in zip(group["values"], cells) if cell["status"] == "SIMULATED"],
                    parameter=group["kind"], direction=group["direction"])
            delta = {k: cells[-1]["observation"]["metrics"][k] / cells[0]["observation"]["metrics"][k] - 1
                     for k in ("ttft_ns", "tpot_ns", "e2e_ns")} if comparable else None
            rows.append({"question_id": group["id"], "question": group["question"], "kind": group["kind"],
                "load": dict(zip(("id", "prompt_tokens", "output_tokens", "batch"), workload)),
                "points": data, "comparable": comparable, "feasible_points_comparable": valid_comparable, "unexpected_changes": unexpected,
                "actual_changed_paths": changed, "fixed_mapping": fixed_mapping,
                "delta_last_vs_first": delta, "trend_check": trend,
                "credibility": "MECHANISM_ONLY_UNVALIDATED" if comparable and (not trend or trend["status"] == "PASS")
                    else "PARTIAL_CAPACITY_LIMIT_UNVALIDATED" if valid_comparable and all(c["status"] in ("SIMULATED", "BLOCKED") for c in cells)
                        and (not trend or trend["status"] == "PASS")
                    else "NOT_ACCEPTED" if all(c["status"] != "PENDING" for c in cells) else "PENDING"})
    write_json(out / "summary.json", {"schema": "multiworkload-matrix/v2", "validation_status": "UNVALIDATED", "rows": rows})
    lines = ["# 多负载架构与参数仿真结果", "", "所有绝对时间均为未做目标器件校准的分析仿真，不是硬件实测。",
             "同一行各点按表列顺序对应。TTFT/TPOT/E2E 为请求 p50；不是线上 p99。", "",
             "| 问题 | 负载 输入/输出/请求数 | 参数或对照顺序 | TTFT ms | TPOT ms | E2E ms | 状态/可信边界 |", "|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        load = row["load"]
        def numbers(key):
            return " / ".join(f'{p["metrics_ms"][key]:.3f}' if key in p["metrics_ms"] else p["status"] for p in row["points"])
        lines.append(f'| {row["question_id"]} {row["question"]} | {load["id"]}: {load["prompt_tokens"]}/{load["output_tokens"]}/{load["batch"]} | '
            + " / ".join(str(p["value"]) for p in row["points"]) + f' | {numbers("ttft_ms")} | {numbers("tpot_ms")} | {numbers("e2e_ms")} | {row["credibility"]} |')
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--loads", nargs="*", help="L1 etc.; defaults to all eight, never changes a load")
    parser.add_argument("--groups", nargs="*", help="I1 etc.; defaults to all eight")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--case-timeout-s", type=float, default=0,
                        help="Host wall-clock limit; 0 waits for complete execution")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(q.OUTPUT_ROOT.resolve()):
        parser.error("outputs must stay inside the memory-tier experiment directory")
    workloads = [w for w in WORKLOADS if not args.loads or w[0].split("_")[0] in args.loads]
    groups = [g for g in GROUPS if not args.groups or g["id"] in args.groups]
    if not workloads or not groups or args.workers < 1 or args.case_timeout_s < 0 or not math.isfinite(args.case_timeout_s):
        parser.error("invalid loads/groups/workers/timeout")
    if args.case:
        run_cell(args.case, workloads[0], out)
        return
    if args.report_only:
        summarize(out, groups, workloads)
        return
    manifest_path = out / "manifest.json"
    source = source_identity()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source"] != source:
            raise ValueError("source changed; use a new output directory, not stale checkpoints")
    else:
        model, identity = q.load_model()
        write_json(out / "model_config.json", q.scenario_payload(q.build_scenario(model))["model"])
        write_json(manifest_path, {"schema": "multiworkload-matrix/v2", "source": source, "model": identity,
            "workloads": workloads, "groups": groups, "native_execution": False,
            "host_execution": {"workers": args.workers, "case_timeout_s": args.case_timeout_s,
                               "timeout_semantics": "0 means no wall-clock cutoff"},
            "validation_status": "UNVALIDATED", "limits": q.LIMITATIONS + [
                "All requests arrive together; no online arrival-rate or p99 claim.",
                "P1/P2 use an optimistic 1600 GB/s active-memory interface, not the NAND cold-page model.",
                "CIM is cold, first-layer MLP only; fixed 8x256x256 tiles and conservative single-slot lifecycle.",
                "P4 uses feasible 256/512/1024 KiB scratch within fixed 3 MiB CIM; the old impossible 4 MiB point remains in v3, not relabeled.",
                "KV eager admission reserves full generation headroom; all submitted requests and queuing time remain included.",
                "HOST_TIMEOUT is a host execution limit, never simulated device infeasibility."]})
    variants = list(dict.fromkeys(case for group in groups for case in group["cases"]))
    jobs = [(case, workload) for workload in workloads for case in variants
            if not (out / "cells" / (workload[0] + "__" + case + ".json")).exists()]
    print(f"START {len(jobs)} new cases; workers={args.workers}; out={out}", flush=True)

    def launch(job):
        case, workload = job
        cell_id = workload[0] + "__" + case
        command = [sys.executable, str(Path(__file__).resolve()), "--case", case, "--loads",
                   workload[0].split("_")[0], "--output", str(out)]
        try:
            finished = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=args.case_timeout_s or None, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            path = out / "cells" / (cell_id + ".json")
            if not path.exists():
                write_json(path, {"id": cell_id, "status": "WORKER_FAILED", "error": finished.stderr[-8000:]})
        except subprocess.TimeoutExpired:
            write_json(out / "cells" / (cell_id + ".json"), {"id": cell_id, "status": "HOST_TIMEOUT",
                "error": f"Host wall-clock exceeded {args.case_timeout_s}s; no device feasibility conclusion."})
        result = json.loads((out / "cells" / (cell_id + ".json")).read_text(encoding="utf-8"))
        return cell_id, result["status"], result.get("elapsed_s")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            print(count, "/", len(jobs), *future.result(), flush=True)
            if count % 8 == 0:
                summarize(out, groups, workloads)
    rows = summarize(out, groups, workloads)
    print("DONE", len(rows), "question/load rows; comparable", sum(r["comparable"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
