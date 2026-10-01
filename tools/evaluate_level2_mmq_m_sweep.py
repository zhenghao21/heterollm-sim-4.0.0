"""Evaluate the independent prefill-MMQ M-axis holdouts.

This evaluator uses only the analytical MMQ work model and independent CUDA
device intervals.  It never reads native LLM timing.  A format is accepted
only when every declared M-specific 2x2 N/K training cell has a measured
3072x3072 holdout with CV<10% and APE<10% under the calibrated analytical
ratio surface.
"""
from __future__ import annotations

import json
import statistics
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import GemmWorkload, GPUProfile, HBMProfile
from heterollm_sim.kernel_model import (KernelCapability, KernelSample,
                                        performance_surface)
from heterollm_sim.kernel_model import kernel_calibration_dispatch_signature
from heterollm_sim.mmq_work import derive_mmq_work

SOURCE = ROOT / "artifacts/development/level2_mmq_stage_20260930/prefill_m_sweep"
OUT = SOURCE / "holdout_evaluation.json"
SCENARIO = ROOT / "artifacts/development/ui_native_matched_20260929/qwen25_p512_o128_c1__fixed_runtime.scenario.json"
RUNTIME_SHA = "8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"
TYPE_IDS = {"q4_k": 12, "q6_k": 14, "iq4_xs": 23}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _signature(row: dict, source) -> str:
    return f"mmq:type={TYPE_IDS[row['format'].casefold()]}:j={source.j}:fallback={int(int(row['shape'][1]) % 128 != 0)}:stream_k={int(source.fixup_launch)}"


def _workload(row: dict, source) -> GemmWorkload:
    m, n, k = map(int, row["shape"])
    bits = {"q4_k": 4, "q6_k": 6, "iq4_xs": 4}[row["format"].casefold()]
    return GemmWorkload(m=m, n=n, k=k, activation_bits=32, weight_bits=bits,
                        output_bits=32, accumulator_bits=32,
                        packed_weight_formats=(row["format"].casefold(),),
                        weight_storage_bytes=source.logical_weight_bytes,
                        activation_storage_bytes=source.consumer_unique_bytes,
                        output_storage_bytes=source.native_output_bytes,
                        mmq_work=source, cache_protocol="cold_streaming",
                        execution_phase="prefill", activation_dtype="fp32",
                        layout="contiguous")


def _cells(samples):
    points = {sample.shape for sample in samples}
    axes = [sorted({point[i] for point in points}) for i in range(3)]
    cells = []
    for m in axes[0]:
        for n0, n1 in zip(axes[1], axes[1][1:]):
            for k0, k1 in zip(axes[2], axes[2][1:]):
                low, high = (m, n0, k0), (m, n1, k1)
                if all(point in points for point in product((m,), (n0, n1), (k0, k1))):
                    cells.append((low, high))
    return tuple(cells)


def main() -> None:
    scenario = scenario_from_dict(_load(SCENARIO))
    gpu = scenario.resolve_component_profile("gpu0", GPUProfile)
    hbm = scenario.resolve_component_profile("hbm0", HBMProfile)
    rows = _load(SOURCE / "measurements.json")
    result = {"schema": "heterollm.level2.mmq.prefill.m-sweep.holdout/v2",
              "measurement_source": str((SOURCE / "measurements.json").relative_to(ROOT)).replace("\\", "/"),
              "runtime_binary_sha256": RUNTIME_SHA, "target_llm_timing_used": False,
              "gate": "complete M-specific 2x2 N/K cells + CV<10% + holdout APE<10%",
              "required_m": [64, 128, 256, 512, 1024], "formats": [],
              "accepted_formats": [], "production_qualified": False}
    for fmt in sorted({str(row["format"]).casefold() for row in rows}):
        fmt_rows = [row for row in rows if str(row["format"]).casefold() == fmt]
        training = [row for row in fmt_rows if row.get("split") == "train"]
        holdouts = [row for row in fmt_rows if row.get("split") == "holdout"]
        by_group = {}
        for row in training:
            source = derive_mmq_work(m=int(row["shape"][0]), n=int(row["shape"][1]), k=int(row["shape"][2]),
                                     weight_format=fmt.upper(), sm_count=gpu.sm_count,
                                     shared_memory_per_block=101376, runtime_binary_sha256=RUNTIME_SHA)
            workload = _workload(row, source)
            signature = _signature(row, source)
            baseline = float(__import__("tools.build_level2_surfaces", fromlist=["_analytical_ns"])._analytical_ns(
                gpu, hbm, {**row, "phase": "prefill", "kernel_family": "mmq"}, fmt))
            by_group.setdefault((int(row["shape"][0]), signature), []).append(
                KernelSample(*map(int, row["shape"]), float(row["median_ns"]), baseline,
                             float(statistics.stdev(row["samples_ns"])), len(row["samples_ns"]),
                             f"{row.get('raw_sha256','')}:independent_m_sweep", None, signature))
        format_rows = []
        accepted = True
        for m, signature in sorted(by_group):
            samples = tuple(by_group[(m, signature)])
            cells = _cells(samples)
            cap = KernelCapability("mmq_m_sweep", (fmt,), "fp32", "prefill", "tensor",
                                   "independent synthetic MMQ M sweep", output_bits=32,
                                   min_shape=(m, 1, 1), max_shape=(m, 1_000_000, 1_000_000),
                                   samples=samples, dispatch_signature=signature,
                                   calibration_cells=cells, calibration_source_bound=True)
            candidates = [row for row in holdouts if int(row["shape"][0]) == m]
            if not candidates:
                accepted = False
                format_rows.append({"m": m, "dispatch_signature": signature, "accepted": False,
                                    "reason": "m_holdout_missing"})
                continue
            for row in candidates:
                source = derive_mmq_work(m=int(row["shape"][0]), n=int(row["shape"][1]), k=int(row["shape"][2]),
                                         weight_format=fmt.upper(), sm_count=gpu.sm_count,
                                         shared_memory_per_block=101376, runtime_binary_sha256=RUNTIME_SHA)
                baseline = float(__import__("tools.build_level2_surfaces", fromlist=["_analytical_ns"])._analytical_ns(
                    gpu, hbm, {**row, "phase": "prefill", "kernel_family": "mmq"}, fmt))
                predicted, _, audit = performance_surface(cap, tuple(map(int, row["shape"])), baseline)
                cv = float(row.get("cv_pct", 100.0))
                ape = 100 * abs(predicted - float(row["median_ns"])) / float(row["median_ns"])
                ok = audit.get("model") == "calibrated_analytical" and cv < 10 and ape < 10
                accepted &= ok
                format_rows.append({"format": fmt, "shape": row["shape"], "m": m,
                                    "dispatch_signature": signature, "predicted_ns": predicted,
                                    "observed_ns": row["median_ns"], "cv_pct": cv,
                                    "ape_pct": ape, "reason": audit.get("reason"), "eligible": ok})
        result["formats"].append({"format": fmt, "rows": format_rows, "accepted": accepted,
                                  "training_rows": len(training), "holdout_rows": len(holdouts)})
        if accepted:
            result["accepted_formats"].append(fmt)
    result["production_qualified"] = bool(result["accepted_formats"])
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
