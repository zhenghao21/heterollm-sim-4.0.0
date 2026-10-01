"""Collect and audit independent Q5_K/Q8_0 Level-2 MMVQ surfaces.

The protocol is deliberately format-local: each format gets a complete M=1
N/K four-corner grid and one interior held-out point.  The exact-Q8 probe is
the only harness; no native LLM timing is read.  A failed correctness run
stops immediately and is left in the artifact directory for audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NCU = Path("C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.1.1/ncu.bat")
PROBE = ROOT / "tools/probe_synthetic_mmvq.py"
DLL_DIR = ROOT / "source/llama.cpp-native-thread-control/build-native-thread-control/bin"
OUT = ROOT / "artifacts/development/level2_format_holdouts_20260930"
RUNTIME_SHA256 = "8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"
FORMATS = ("Q5_K", "Q8_0")
TRAIN = ((1, 2048, 2048), (1, 2048, 4096),
         (1, 4096, 2048), (1, 4096, 4096))
HOLDOUT = ((1, 3072, 3072),)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: str) -> float:
    return float(str(value).replace(",", "").strip())


def _module(raw: dict) -> dict:
    modules = raw.get("loaded_modules_after", {})
    if isinstance(modules, dict):
        value = modules.get("ggml-cuda.dll")
        if value is None:
            value = next((v for k, v in modules.items()
                          if str(k).casefold() == "ggml-cuda.dll"), None)
    else:
        value = next((v for v in modules
                      if str(v.get("name", "")).casefold() == "ggml-cuda.dll"), None)
    if not isinstance(value, dict) or not value.get("sha256"):
        raise RuntimeError("ggml-cuda.dll identity missing")
    return value


def parse_case(raw_path: Path, csv_path: Path, fmt: str, shape: tuple[int, int, int], split: str) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("status") != "measured" or raw.get("modules_stable") is not True:
        raise RuntimeError(f"measurement/module gate failed: {raw_path}")
    if not raw.get("first_call_correctness", {}).get("passed"):
        raise RuntimeError(f"first-call correctness gate failed: {raw_path}")
    if not raw.get("final_correctness", {}).get("passed"):
        raise RuntimeError(f"final correctness gate failed: {raw_path}")
    module = _module(raw)
    if module["sha256"] != RUNTIME_SHA256:
        raise RuntimeError(f"runtime identity mismatch: {raw_path}")
    text = csv_path.read_text(encoding="utf-8-sig")
    if '"ID"' not in text:
        raise RuntimeError(f"Nsight CSV has no metric table: {csv_path}")
    events = csv.DictReader(text[text.index('"ID"'):].splitlines())
    groups: dict[str, dict[str, float | str]] = {}
    symbols: set[str] = set()
    for event in events:
        name = " ".join(str(event.get("Kernel Name", "")).split())
        if "mul_mat_vec_q<" not in name:
            continue
        ident = str(event.get("ID", ""))
        symbols.add(name)
        row = groups.setdefault(ident, {"symbol": name})
        metric = str(event.get("Metric Name", ""))
        if metric in {"gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum"}:
            row[metric] = _number(event["Metric Value"])
    required = ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum")
    formal = [row for row in groups.values() if all(key in row for key in required)]
    if len(symbols) != 1 or len(formal) < 2:
        raise RuntimeError(f"MMVQ launch identity/count gate failed: {csv_path}")
    formal = formal[-8:]
    times = [float(row[required[0]]) for row in formal]
    bandwidth = [(float(row[required[1]]) + float(row[required[2]])) / t
                 for row, t in zip(formal, times)]
    return {
        "format": fmt,
        "shape": list(shape),
        "split": split,
        "symbol": next(iter(symbols)),
        "resource_signature": next(iter(symbols)),
        "median_ns": statistics.median(times),
        "cv_pct": statistics.stdev(times) / statistics.mean(times) * 100.0,
        "samples_ns": times,
        "bandwidth_gb_s": statistics.median(bandwidth),
        "csv_sha256": sha256(csv_path),
        "raw_sha256": sha256(raw_path),
        "runtime_binary_sha256": module["sha256"],
        "measurement_boundary": "cuda_device_kernel_interval",
        "cache_control": "all",
        "clock_control": "none",
        "target_llm_timing_used": False,
        "correctness": raw["final_correctness"],
    }


def _dispatch_signature(fmt: str, shape: tuple[int, int, int]) -> str:
    # Reuse the production source-work and signature path.  Importing the
    # builder is data-only and does not mutate the checked-in surface.
    sys.path.insert(0, str(ROOT / "tools"))
    from build_level2_surfaces import _source_work  # type: ignore
    sys.path.insert(0, str(ROOT / "src"))
    from heterollm_sim.cost_models import GemmWorkload  # type: ignore
    from heterollm_sim.kernel_model import kernel_calibration_dispatch_signature  # type: ignore
    m, n, k = shape
    source = _source_work(m, n, k, fmt.casefold())
    return kernel_calibration_dispatch_signature(GemmWorkload(
        m=m, k=k, n=n, activation_bits=32, weight_bits=(5 if fmt == "Q5_K" else 8),
        output_bits=32, accumulator_bits=32, packed_weight_formats=(fmt.casefold(),),
        weight_storage_bytes=source.logical_weight_bytes,
        activation_storage_bytes=source.consumer_q8_1_unique_bytes,
        output_storage_bytes=source.output_f32_bytes, mmvq_work=source,
        cache_protocol="cold_streaming", execution_phase="decode",
        activation_dtype="fp32", layout="contiguous"))


def _analytical(row: dict) -> float:
    sys.path.insert(0, str(ROOT / "tools"))
    from build_level2_surfaces import _analytical_ns  # type: ignore
    from heterollm_sim.config import scenario_from_dict  # type: ignore
    from heterollm_sim.cost_models import GPUProfile, HBMProfile  # type: ignore
    scenario_path = ROOT / "artifacts/development/ui_native_matched_20260929/qwen25_p512_o128_c1__fixed_runtime.scenario.json"
    scenario = scenario_from_dict(json.loads(scenario_path.read_text(encoding="utf-8")))
    gpu = scenario.resolve_component_profile("gpu0", GPUProfile)
    hbm = scenario.resolve_component_profile("hbm0", HBMProfile)
    return _analytical_ns(gpu, hbm, row, str(row["format"]).casefold())


def _bilinear_log(training: list[dict], row: dict) -> float:
    points = {tuple(item["shape"]): item for item in training}
    n, k = row["shape"][1], row["shape"][2]
    def ratio(nn: int, kk: int) -> float:
        item = points[(1, nn, kk)]
        return float(item["median_ns"]) / float(item["analytical_ns"])
    fn = math.log(n / 2048) / math.log(4096 / 2048)
    fk = math.log(k / 2048) / math.log(4096 / 2048)
    return ((ratio(2048, 2048) * (1 - fn) + ratio(4096, 2048) * fn) * (1 - fk)
            + (ratio(2048, 4096) * (1 - fn) + ratio(4096, 4096) * fn) * fk)


def evaluate(rows: list[dict], protocol: dict) -> dict:
    output = []
    for fmt in FORMATS:
        fmt_rows = [dict(row) for row in rows if row["format"] == fmt]
        training = [row for row in fmt_rows if row["split"] == "train"]
        holdout = [row for row in fmt_rows if row["split"] == "holdout"]
        for row in fmt_rows:
            row["dispatch_signature"] = _dispatch_signature(fmt, tuple(row["shape"]))
            row["analytical_ns"] = _analytical(row)
        training = [row for row in training if row["cv_pct"] < 10.0]
        holdout_results = []
        for row in holdout:
            predicted = row["analytical_ns"] * _bilinear_log(training, row) if len(training) == 4 else None
            ape = None if predicted is None else abs(predicted - row["median_ns"]) / row["median_ns"] * 100.0
            row.update(predicted_ns=predicted, ape_pct=ape,
                       eligible=bool(predicted is not None and row["cv_pct"] < 10.0 and ape < 10.0))
            holdout_results.append(row)
        all_rows = training + holdout
        symbols = {row["resource_signature"] for row in all_rows}
        signatures = {row["dispatch_signature"] for row in all_rows}
        source_same = len(signatures) == 1
        resource_same = len(symbols) == 1
        complete = len([row for row in fmt_rows if row["split"] == "train"]) == 4
        accepted = bool(complete and len(training) == 4 and holdout_results and
                       all(item["eligible"] for item in holdout_results) and
                       source_same and resource_same and
                       all(row["runtime_binary_sha256"] == RUNTIME_SHA256 for row in all_rows))
        output.append({
            "format": fmt, "training_rows": len(training), "required_training_rows": 4,
            "holdout_rows": holdout_results, "complete_four_corner_grid": complete,
            "source_signature_same_domain": source_same,
            "resource_signature_same_domain": resource_same,
            "dispatch_signatures": sorted(signatures),
            "resource_signatures": sorted(symbols),
            "accepted": accepted,
            "decision": "install" if accepted else "reject_fail_closed",
        })
    result = {
        "schema": "heterollm.level2.format_holdouts/v1",
        "protocol": "protocol.json",
        "runtime_binary_sha256": RUNTIME_SHA256,
        "measurement_source": "independent_exact_q8_input_ncu_device_intervals",
        "native_llm_timing_used": False,
        "gate": "complete four corners + correctness + CV<10% + heldout APE<10% + source/resource same-domain",
        "formats": output,
        "production_acceptance": {item["format"]: item["accepted"] for item in output},
        "all_formats_accepted": bool(output and all(item["accepted"] for item in output)),
        "install_policy": "install only formats with accepted=true; otherwise fail-closed",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    protocol = {
        "schema": "level2-format-holdouts/v1",
        "purpose": "Independent Q5_K/Q8_0 exact-Q8 MMVQ device-kernel measurement; no native LLM timing calibration",
        "harness": "python_exact_q8", "probe": str(PROBE.relative_to(ROOT)).replace("\\", "/"),
        "probe_sha256": sha256(PROBE), "formats": list(FORMATS),
        "training_shapes_by_format": {fmt: [list(shape) for shape in TRAIN] for fmt in FORMATS},
        "holdout_shapes_by_format": {fmt: [list(shape) for shape in HOLDOUT] for fmt in FORMATS},
        "warmup": 8, "repeats": 8, "seed": 20260930, "launch_skip": 9, "launch_count": 8,
        "cache_control": "all", "clock_control": "none", "per_case_timeout_s": 180,
        "gate_cv_pct": 10, "gate_ape_pct": 10, "runtime_binary_sha256": RUNTIME_SHA256,
        "correctness_gate": "atol=0.001,rtol=0.001 packed-weight double-dot exact-Q8_1 input",
        "no_retry_on_failure": True, "install_policy": "fail_closed",
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep a separately signed protocol record for each format.  The combined
    # file is convenient for one invocation; these records make it impossible
    # to mistake one format's grid for the other's independent experiment.
    for fmt in FORMATS:
        per_format = dict(protocol)
        per_format["format"] = fmt
        per_format["training_shapes"] = [list(shape) for shape in TRAIN]
        per_format["holdout_shapes"] = [list(shape) for shape in HOLDOUT]
        per_format.pop("training_shapes_by_format", None)
        per_format.pop("holdout_shapes_by_format", None)
        (out / f"protocol_{fmt}.json").write_text(
            json.dumps(per_format, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if not args.audit_only:
        if not NCU.exists(): raise RuntimeError(f"Nsight Compute not found: {NCU}")
        env = dict(os.environ); env["PATH"] = str(DLL_DIR) + ";E:/cuda/bin;" + env.get("PATH", "")
        rows = []
        cases = [(split, fmt, shape) for split in ("train", "holdout")
                 for fmt in FORMATS for shape in (TRAIN if split == "train" else HOLDOUT)]
        for index, (split, fmt, shape) in enumerate(cases, 1):
            ident = f"{fmt}_{shape[0]}_{shape[1]}_{shape[2]}"
            raw, csv_path, log = out / f"{ident}.json", out / f"{ident}.csv", out / f"{ident}.log"
            if not raw.exists() or not csv_path.exists():
                cmd = [str(NCU), "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum",
                       "--cache-control", "all", "--clock-control", "none", "--launch-skip", "9",
                       "--launch-count", "8", "--csv", "--log-file", str(csv_path), sys.executable,
                       str(PROBE), "--quant", fmt, "--m", "1", "--n", str(shape[1]), "--k", str(shape[2]),
                       "--warmup", "8", "--repeats", "8", "--seed", "20260930", "--output", str(raw)]
                with log.open("w", encoding="utf-8") as handle:
                    subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=handle,
                                   timeout=180, check=True)
            rows.append(parse_case(raw, csv_path, fmt, shape, split))
            (out / "measurements.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[{index}/{len(cases)}] {ident}", flush=True)
    else:
        rows = json.loads((out / "measurements.json").read_text(encoding="utf-8"))
    report = evaluate(rows, protocol)
    (out / "holdout_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
