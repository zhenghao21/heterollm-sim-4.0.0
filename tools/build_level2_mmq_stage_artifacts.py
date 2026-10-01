"""Build independent MMQ conversion/fixup evidence from existing NCU traces.

The main MMQ collector intentionally discards these kernels.  This utility
replays the raw CSVs and emits a separate stage ledger.  It never reads native
LLM timing.  Geometry or repeatability disagreements are retained as rejected
evidence; no stage is installed without the missing M sweep and holdout.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.config import scenario_from_dict  # noqa: E402
from heterollm_sim.cost_models import GPUProfile, HBMProfile, TensorKernelWorkload, estimate_gpu_tensor_kernel  # noqa: E402
from heterollm_sim.mmq_work import derive_mmq_work  # noqa: E402
from heterollm_sim.mmq_level2_surfaces import mmq_stage_dispatch_signature  # noqa: E402
from heterollm_sim.mmq_level2_surfaces import MMQStageSample, evaluate_mmq_stage_holdout  # noqa: E402

SOURCE = ROOT / "artifacts/development/level2_mmq_prefill_exact_20260930"
OUT = ROOT / "artifacts/development/level2_mmq_stage_20260930"
SCENARIO = ROOT / "artifacts/development/ui_native_matched_20260929/qwen25_p512_o128_c1__fixed_runtime.scenario.json"
RUNTIME_SHA256 = "8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"
REQUIRED_M = (64, 128, 256, 512, 1024)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_csv(path: Path) -> dict[str, dict[str, object]]:
    text = path.read_text(encoding="utf-8-sig")
    marker = '"ID"'
    if marker not in text:
        raise ValueError("NCU CSV has no metric header")
    rows = csv.DictReader(text[text.index(marker):].splitlines())
    groups: dict[str, dict[str, object]] = {}
    for row in rows:
        ident = str(row.get("ID", ""))
        if not ident:
            continue
        name = " ".join(str(row.get("Kernel Name", "")).split())
        group = groups.setdefault(ident, {
            "name": name,
            "block": tuple(int(x.strip()) for x in str(row["Block Size"]).strip("()").split(",")),
            "grid": tuple(int(x.strip()) for x in str(row["Grid Size"]).strip("()").split(",")),
            "metrics": {},
        })
        group["metrics"][str(row.get("Metric Name", ""))] = str(row.get("Metric Value", "")).replace(",", "")
    return groups


def stage_from_name(name: str) -> str | None:
    if "quantize_mmq_q8_1" in name:
        return "activation_repack"
    if "mul_mat_q_stream_k_fixup" in name:
        return "stream_k_fixup"
    return None


def analytical_ns(gpu, hbm, source, stage: str) -> float:
    if stage == "activation_repack":
        workload = TensorKernelWorkload(source.conversion_operations, source.conversion_read_bytes,
                                        source.conversion_write_bytes, streaming_fraction=1.0,
                                        name="mmq_activation_repack")
    else:
        if not source.fixup_launch or source.fixup_operations == 0:
            return 0.0
        workload = TensorKernelWorkload(source.fixup_operations, source.fixup_read_bytes,
                                        source.fixup_write_bytes, streaming_fraction=1.0,
                                        name="mmq_stream_k_fixup")
    return float(estimate_gpu_tensor_kernel(gpu, hbm, workload).service_ns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", default=None,
                        help="exact-Q8 grid(s) containing NCU CSV/JSON pairs")
    args = parser.parse_args()
    source_roots = [path.resolve() for path in (args.source or [SOURCE])]
    OUT.mkdir(parents=True, exist_ok=True)
    scenario = scenario_from_dict(json.loads(SCENARIO.read_text(encoding="utf-8")))
    gpu = scenario.resolve_component_profile("gpu0", GPUProfile)
    hbm = scenario.resolve_component_profile("hbm0", HBMProfile)
    raw_rows = []
    for source_root in source_roots:
        raw_rows.extend(json.loads((source_root / "measurements.json").read_text(encoding="utf-8")))
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for raw in raw_rows:
        fmt = str(raw["format"]).casefold()
        m, n, k = map(int, raw["shape"])
        source = derive_mmq_work(m=m, n=n, k=k, weight_format=fmt.upper(), sm_count=gpu.sm_count,
                                 shared_memory_per_block=101376, runtime_binary_sha256=RUNTIME_SHA256)
        source_root = next((root for root in source_roots
                            if (root / f"{raw['format']}_{m}_{n}_{k}.csv").exists()), None)
        if source_root is None:
            rejected.append({"format": fmt, "shape": [m, n, k], "reason": "raw_csv_missing"})
            continue
        csv_path = source_root / f"{raw['format']}_{m}_{n}_{k}.csv"
        groups = parse_csv(csv_path)
        by_stage: dict[str, list[tuple[str, dict[str, object]]]] = {}
        for ident, group in groups.items():
            stage = stage_from_name(str(group["name"]))
            if stage:
                by_stage.setdefault(stage, []).append((ident, group))
        for stage in ("activation_repack", "stream_k_fixup"):
            groups_for_stage = by_stage.get(stage, [])
            if stage == "stream_k_fixup" and not source.fixup_launch:
                if groups_for_stage:
                    rejected.append({"format": fmt, "shape": [m, n, k], "stage": stage,
                                     "reason": "fixup_observed_but_source_geometry_has_no_fixup"})
                continue
            if stage == "stream_k_fixup" and not groups_for_stage:
                rejected.append({"format": fmt, "shape": [m, n, k], "stage": stage,
                                 "reason": "expected_fixup_launch_missing_from_trace"})
                continue
            required_metrics = {"gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum"}
            if len(groups_for_stage) < 2 or any(not required_metrics <= set(group["metrics"]) for _, group in groups_for_stage):
                rejected.append({"format": fmt, "shape": [m, n, k], "stage": stage,
                                 "reason": "independent_stage_intervals_missing"})
                continue
            # Use every separately captured launch.  A large first-call interval
            # is preserved in CV and therefore fails closed rather than being
            # silently dropped.
            durations = tuple(float(group["metrics"]["gpu__time_duration.sum"]) for _, group in groups_for_stage)
            read_bytes = tuple(float(group["metrics"]["dram__bytes_read.sum"]) for _, group in groups_for_stage)
            write_bytes = tuple(float(group["metrics"]["dram__bytes_write.sum"]) for _, group in groups_for_stage)
            geometries = {(tuple(group["block"]), tuple(group["grid"])) for _, group in groups_for_stage}
            if len(geometries) != 1:
                rejected.append({"format": fmt, "shape": [m, n, k], "stage": stage,
                                 "reason": "stage_launch_geometry_changed"})
                continue
            cv = statistics.stdev(durations) / statistics.mean(durations) * 100.0
            group = groups_for_stage[-1][1]
            metric = group["metrics"]
            rows.append({
                "format": fmt, "phase": "prefill", "kernel_family": "mmq", "stage": stage,
                "shape": [m, n, k], "split": raw["split"],
                "device_ns": statistics.median(durations), "samples_ns": list(durations), "cv_pct": cv,
                "analytical_ns": analytical_ns(gpu, hbm, source, stage),
                "dispatch_signature": mmq_stage_dispatch_signature(stage, source),
                "runtime_binary_sha256": raw.get("runtime_binary_sha256", RUNTIME_SHA256),
                "geometry": {"block": list(group["block"]), "grid": list(group["grid"]),
                             "registers_per_thread": float(metric.get("launch__registers_per_thread", 0)),
                             "shared_memory_per_block": float(metric.get("launch__shared_mem_per_block_allocated", 0)),
                             "dynamic_shared_memory": float(metric.get("launch__shared_mem_per_block_dynamic", 0)),
                             "static_shared_memory": float(metric.get("launch__shared_mem_per_block_static", 0))},
                "csv_sha256": sha256(csv_path), "raw_sha256": sha256(source_root / f"{raw['format']}_{m}_{n}_{k}.json"),
                "measurement_boundary": "cuda_device_stage_kernel_interval",
                "target_llm_timing_used": False,
                "source_main_measurement": str((source_root / "measurements.json").relative_to(ROOT)).replace("\\", "/"),
            })
    protocol = {
        "schema": "heterollm.level2.mmq.stage.grid/v1", "hardware_id": "nvidia-rtx-5080",
        "runtime_binary_sha256": RUNTIME_SHA256, "source_measurement_dirs": [str(root.relative_to(ROOT)).replace("\\", "/") for root in source_roots],
        "scenario": str(SCENARIO.relative_to(ROOT)).replace("\\", "/"), "phase": "prefill",
        "stages": ["activation_repack", "stream_k_fixup"], "required_prefill_m": list(REQUIRED_M),
        "measured_prefill_m": sorted({int(row["shape"][0]) for row in raw_rows}), "target_llm_timing_used": False,
        "measurement_boundary": "cuda_device_stage_kernel_interval",
        "resource_audit": "block/grid/registers/shared captured per stage; no occupancy inference",
        "production_install_policy": "accepted only with complete M sweep and independent holdout",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (OUT / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    measured_m = sorted({int(row["shape"][0]) for row in raw_rows})
    complete_m_sweep = set(REQUIRED_M) <= set(measured_m)
    stage_evaluation = {}
    accepted_stages, accepted_formats = set(), set()
    for stage in ("activation_repack", "stream_k_fixup"):
        stage_evaluation[stage] = {}
        for fmt in sorted({str(row["format"]) for row in rows if row["stage"] == stage}):
            stage_evaluation[stage][fmt] = {}
            signatures = sorted({str(row["dispatch_signature"]) for row in rows
                                 if row["stage"] == stage and row["format"] == fmt})
            for signature in signatures:
                def convert(row):
                    shape = tuple(int(value) for value in row["shape"])
                    return MMQStageSample(stage, fmt, *shape, float(row["device_ns"]),
                                          float(row["analytical_ns"]),
                                          statistics.stdev(row["samples_ns"]), len(row["samples_ns"]),
                                          f"{row['source_main_measurement']}#csv_sha256={row['csv_sha256']}",
                                          signature, str(row["runtime_binary_sha256"]),
                                          resources=tuple(sorted((str(key), value) for key, value in
                                                                 (row.get("geometry") or {}).items())),
                                          split=str(row["split"]))
                train = [convert(row) for row in rows if row["stage"] == stage and row["format"] == fmt
                         and row["dispatch_signature"] == signature and row["split"] == "train"]
                holdout = [convert(row) for row in rows if row["stage"] == stage and row["format"] == fmt
                           and row["dispatch_signature"] == signature and row["split"] == "holdout"]
                result = ({"accepted": False, "reasons": ["train_or_holdout_missing"]} if not train or not holdout else
                          evaluate_mmq_stage_holdout(stage=stage, weight_format=fmt, training=train,
                                                      holdout=holdout, runtime_binary_sha256=RUNTIME_SHA256,
                                                      dispatch_signature=signature))
                result = {**result, "production_accepted": bool(result.get("accepted") and complete_m_sweep),
                          "production_rejection_reasons": ([] if complete_m_sweep else ["m_sweep_missing_64_128_256_1024"])}
                stage_evaluation[stage][fmt][signature] = result
                if result.get("accepted") and complete_m_sweep:
                    accepted_stages.add(stage); accepted_formats.add(fmt)
    report = {
        "schema": "heterollm.level2.mmq.stage.holdout/v1", "measurement_source": str((OUT / "measurements.json").relative_to(ROOT)).replace("\\", "/"),
        "runtime_binary_sha256": RUNTIME_SHA256, "accepted_stages": sorted(accepted_stages), "accepted_formats": sorted(accepted_formats),
        "required_prefill_m": list(REQUIRED_M), "measured_prefill_m": measured_m,
        "rows": rows, "rejected_rows": rejected, "stage_evaluation": stage_evaluation,
        "rejection_reasons": (["m_sweep_missing_64_128_256_1024"] if not set(REQUIRED_M) <= set(measured_m) else [])
                              + ([] if accepted_stages else ["no_independent_stage_holdout_surface_installed"]),
        "target_llm_timing_used": False,
    }
    (OUT / "holdout_evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sweep = OUT / "prefill_m_sweep"
    sweep.mkdir(exist_ok=True)
    # Freeze the smallest complete N/K grid for the outstanding M axis before
    # any later collection.  Existing M=512 evidence is not relabeled train
    # evidence for missing M values, and target LLM totals are never consumed.
    sweep_protocol = {
        **protocol, "schema": "heterollm.level2.mmq.prefill.m-sweep/v1",
        "harness": "python_exact_q8_mmq", "fixed_m": None,
        "m_axis": [64, 128, 256, 1024],
        "training_axes_n": [2048, 4096], "training_axes_k": [2048, 4096],
        "training_shapes_by_format": {
            fmt: [[m, n, k] for m in (64, 128, 256, 1024)
                  for n in (2048, 4096) for k in (2048, 4096)]
            for fmt in ("Q4_K", "Q6_K", "IQ4_XS")
        },
        "holdout_shapes_by_format": {
            fmt: [[m, 3072, 3072] for m in (64, 128, 256, 1024)]
            for fmt in ("Q4_K", "Q6_K", "IQ4_XS")
        },
        "warmup": 8, "repeats": 40, "seed": 20260930,
        "launch_skip": 9, "launch_count": 18, "per_case_timeout_s": 240,
        "correctness_atol": 0.05, "correctness_rtol": 0.03,
        "probe_sha256": sha256(ROOT / "tools/probe_synthetic_mmq.py"),
        "status": "not_collected", "installable": False,
        "reason": "independent_M_sweep_device_measurement_and_holdout_not_yet_available",
    }
    if not (sweep / "protocol.json").exists():
        (sweep / "protocol.json").write_text(json.dumps(sweep_protocol, indent=2) + "\n", encoding="utf-8")
    if not (sweep / "measurements.json").exists():
        (sweep / "measurements.json").write_text("[]\n", encoding="utf-8")
    if not (sweep / "holdout_evaluation.json").exists():
        (sweep / "holdout_evaluation.json").write_text(json.dumps({
        "schema": "heterollm.level2.mmq.prefill.m-sweep.holdout/v1",
        "runtime_binary_sha256": RUNTIME_SHA256, "accepted_formats": [],
        "measured_m": sorted({int(row["shape"][0]) for row in raw_rows}), "required_m": list(REQUIRED_M),
        "rows": [], "accepted": False,
        "reason": "independent_M_sweep_device_measurement_and_holdout_not_yet_available",
        "target_llm_timing_used": False,
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "rejected": len(rejected), "accepted_stages": sorted(accepted_stages)}, indent=2))


if __name__ == "__main__":
    main()
