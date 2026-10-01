"""Evaluate the independent prefill-MMQ holdout before surface installation."""
from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import GemmWorkload, GPUProfile, HBMProfile, estimate_gpu_gemm
from heterollm_sim.kernel_model import kernel_calibration_dispatch_signature
from heterollm_sim.mmq_work import derive_mmq_work

RUNTIME_SHA256 = "8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"
MEASUREMENTS = ROOT / "artifacts/development/level2_mmq_prefill_exact_20260930/measurements.json"
SCENARIO = ROOT / "artifacts/development/ui_native_matched_20260929/qwen25_p512_o128_c1__fixed_runtime.scenario.json"
OUTPUT = MEASUREMENTS.parent / "holdout_evaluation.json"


def source(row):
    m, n, k = map(int, row["shape"])
    return derive_mmq_work(m=m, n=n, k=k, weight_format=row["format"],
                           sm_count=84, shared_memory_per_block=101376,
                           runtime_binary_sha256=RUNTIME_SHA256)


def workload(row):
    work = source(row)
    bits = {"Q4_K": 4, "Q6_K": 6, "IQ4_XS": 4}[row["format"]]
    return GemmWorkload(
        m=work.m, k=work.k, n=work.n, activation_bits=32, weight_bits=bits,
        output_bits=32, accumulator_bits=32, packed_weight_formats=(row["format"].casefold(),),
        weight_storage_bytes=work.logical_weight_bytes,
        activation_storage_bytes=work.consumer_unique_bytes,
        output_storage_bytes=work.native_output_bytes, mmq_work=work,
        cache_protocol="cold_streaming", execution_phase="prefill",
        activation_dtype="fp32", layout="contiguous")


def analytical(gpu, hbm, row):
    profile = gpu.kernel_model
    if profile is not None:
        geometry = row.get("geometry") or {}
        block = geometry.get("block") or (32, 8, 1)
        registers = int(geometry.get("registers_per_thread") or 32)
        shared = int(geometry.get("shared_memory_per_block") or 0)
        warps = int(block[1])
        fmt = row["format"].casefold()
        profile = replace(
            profile,
            kernels=tuple(
                replace(kernel, registers_per_thread=registers,
                        shared_memory_per_cta=shared, warps_per_cta=warps)
                if kernel.phase == "prefill" and kernel.kernel_family == f"cuda_mmq_{fmt}"
                else kernel
                for kernel in profile.kernels
            ),
        )
        gpu = replace(gpu, kernel_model=profile)
    return float(estimate_gpu_gemm(gpu, hbm, workload(row)).metadata["prediction"]["analytical_ns"])


def bilinear_log(points, row):
    m, n, k = map(int, row["shape"])
    by_shape = {tuple(item["shape"]): item for item in points}
    ns = sorted({shape[1] for shape in by_shape})
    ks = sorted({shape[2] for shape in by_shape})
    n0, n1 = max(value for value in ns if value <= n), min(value for value in ns if value >= n)
    k0, k1 = max(value for value in ks if value <= k), min(value for value in ks if value >= k)
    def ratio(nn, kk):
        item = by_shape[(m, nn, kk)]
        return float(item["median_ns"]) / float(item["analytical_ns"])
    def frac(value, lo, hi):
        return 0.0 if lo == hi else math.log(value / lo) / math.log(hi / lo)
    fn, fk = frac(n, n0, n1), frac(k, k0, k1)
    return ((ratio(n0, k0) * (1-fn) + ratio(n1, k0) * fn) * (1-fk)
            + (ratio(n0, k1) * (1-fn) + ratio(n1, k1) * fn) * fk)


def main():
    scenario = scenario_from_dict(json.loads(SCENARIO.read_text(encoding="utf-8")))
    gpu = scenario.resolve_component_profile("gpu0", GPUProfile)
    hbm = scenario.resolve_component_profile("hbm0", HBMProfile)
    rows = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    train = [row for row in rows if row.get("split") == "train"]
    holdout = [row for row in rows if row.get("split") == "holdout"]
    training = []
    rejected_training = []
    for row in train:
        samples = tuple(float(value) for value in row["samples_ns"])
        cv = statistics.stdev(samples) / statistics.mean(samples) * 100 if len(samples) > 1 else 0.0
        if cv >= 10.0:
            rejected_training.append({"format": row["format"], "shape": row["shape"], "cv_pct": cv})
            continue
        item = dict(row)
        item["analytical_ns"] = analytical(gpu, hbm, row)
        item["dispatch_signature"] = kernel_calibration_dispatch_signature(workload(row))
        training.append(item)
    output = []
    for row in holdout:
        item = dict(row)
        item["analytical_ns"] = analytical(gpu, hbm, row)
        item["dispatch_signature"] = kernel_calibration_dispatch_signature(workload(row))
        points = [candidate for candidate in training
                  if candidate["format"] == row["format"]
                  and candidate["dispatch_signature"] == item["dispatch_signature"]]
        predicted = item["analytical_ns"] * bilinear_log(points, item)
        cv = statistics.stdev(row["samples_ns"]) / statistics.mean(row["samples_ns"]) * 100
        ape = abs(predicted - row["median_ns"]) / row["median_ns"] * 100
        item.update(predicted_ns=predicted, cv_pct=cv, ape_pct=ape,
                    eligible=bool(cv < 10.0 and ape < 10.0),
                    acceptance_gate="cv<10% and holdout_ape<10%",
                    transfer_to_llm_validated=False)
        output.append(item)
    result = {"schema": "heterollm.level2.mmq.prefill.holdout/v1",
              "measurement_source": str(MEASUREMENTS.relative_to(ROOT)).replace("\\", "/"),
              "runtime_binary_sha256": RUNTIME_SHA256,
              "surface_model": "analytical_baseline_times_log_bilinear_ratio",
              "training_rows": len(training), "rejected_training_rows": rejected_training,
              "rows": output,
              "accepted_formats": sorted({row["format"] for row in output if row["eligible"]}),
              "all_holdouts_pass": bool(output and all(row["eligible"] for row in output))}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
