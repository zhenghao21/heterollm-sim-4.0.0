"""Collect the pre-registered Level-2 cold MMVQ shape grid.

The executable is the standalone synthetic GGML harness.  This collector
records CUDA kernel intervals from Nsight Compute; it never loads a GGUF or
reads an LLM latency result.  The grid is deliberately generic and is used to
cover interpolation cells around model-derived operator shapes, not to fit an
end-to-end answer.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NCU = Path("C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.1.1/ncu.bat")
EXE = ROOT / "artifacts/development/generic_gemm_microbench_v1/generic-gemm-microbench.exe"
DLL_DIR = ROOT / "source/llama.cpp-native-thread-control/build-native-thread-control/bin"
OUT = ROOT / "artifacts/development/level2_shape_grid_20260930"
PROTOCOL = OUT / "protocol.json"
MEASUREMENTS = OUT / "measurements.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_number(value: str) -> float:
    return float(value.replace(",", ""))


def _parse_ncu(csv_path: Path, raw_path: Path, fmt: str, shape: tuple[int, int, int], split: str) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("status") != "measured" or not raw.get("modules_stable"):
        raise RuntimeError(f"invalid measurement status: {raw_path}")
    if not raw.get("first_call_correctness", {}).get("passed") or not raw.get("final_correctness", {}).get("passed"):
        raise RuntimeError(f"numerical correctness gate failed: {raw_path}")
    text = csv_path.read_text(encoding="utf-8-sig")
    marker = '"ID"'
    if marker not in text:
        raise RuntimeError(f"Nsight CSV has no metric table: {csv_path}")
    events = csv.DictReader(text[text.index(marker):].splitlines())
    groups: dict[str, dict[str, float]] = {}
    symbols: set[str] = set()
    for event in events:
        if "mul_mat_vec_q<" not in event.get("Kernel Name", ""):
            continue
        symbols.add(event["Kernel Name"])
        groups.setdefault(event["ID"], {})[event["Metric Name"]] = _metric_number(event["Metric Value"])
    if len(symbols) != 1 or len(groups) < 3:
        raise RuntimeError(f"MMVQ launch identity/count gate failed: {csv_path}")
    ordered = list(groups.values())[-10:]
    required = ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum")
    if any(any(key not in row for key in required) for row in ordered):
        raise RuntimeError(f"required NCU metrics missing: {csv_path}")
    times = [row["gpu__time_duration.sum"] for row in ordered]
    bandwidth = [
        (row["dram__bytes_read.sum"] + row["dram__bytes_write.sum"]) / row["gpu__time_duration.sum"]
        for row in ordered
    ]
    modules = raw['loaded_modules_after']
    module = (modules.get('ggml-cuda.dll') if isinstance(modules, dict) else
              next((item for item in modules if item.get('name', '').lower() == 'ggml-cuda.dll'), None))
    if module is None:
        raise RuntimeError('CUDA module identity missing')
    return {
        "format": fmt,
        "shape": list(shape),
        "split": split,
        "symbol": next(iter(symbols)),
        "median_ns": statistics.median(times),
        "cv_pct": statistics.stdev(times) / statistics.mean(times) * 100.0,
        "samples_ns": times,
        "bandwidth_gb_s": statistics.median(bandwidth),
        "csv_sha256": sha256(csv_path),
        "raw_sha256": sha256(raw_path),
        "runtime_binary_sha256": module['sha256'],
        "measurement_boundary": "cuda_device_kernel_interval",
        "cache_control": "all",
        "clock_control": "none",
        "target_llm_timing_used": False,
    }


def main() -> None:
    global OUT, PROTOCOL, MEASUREMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve(); PROTOCOL = OUT / 'protocol.json'; MEASUREMENTS = OUT / 'measurements.json'
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    exact_q8 = protocol.get('harness') == 'python_exact_q8'
    if exact_q8:
        probe = ROOT / 'tools/probe_synthetic_mmvq.py'
        if sha256(probe) != protocol['probe_sha256']:
            raise RuntimeError('probe identity changed')
    elif sha256(EXE) != protocol["executable_sha256"]:
        raise RuntimeError("standalone microbenchmark identity changed")
    if not NCU.exists():
        raise RuntimeError(f"Nsight Compute not found: {NCU}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    env = dict(os.environ)
    env["PATH"] = str(DLL_DIR) + ";E:/cuda/bin;" + env.get("PATH", "")
    cases = []
    excluded = {
        (item["format"], tuple(item["shape"]))
        for item in protocol.get("excluded_cases", ())
    }
    for split in ("train", "holdout"):
        for fmt in protocol["formats"]:
            shape_map = protocol.get(
                "training_shapes_by_format" if split == "train" else "holdout_shapes_by_format"
            )
            shared_key = "training_shapes" if split == "train" else "holdout_shapes"
            source_shapes = (
                shape_map.get(fmt)
                if shape_map and fmt in shape_map
                else protocol[shared_key]
            )
            for raw_shape in source_shapes:
                shape = tuple(int(x) for x in raw_shape)
                if (fmt, shape) in excluded:
                    continue
                cases.append((split, fmt, shape))
    for index, (split, fmt, shape) in enumerate(cases, 1):
        ident = f"{fmt}_{shape[0]}_{shape[1]}_{shape[2]}"
        raw = OUT / f"{ident}.json"
        csv_path = OUT / f"{ident}.csv"
        log = OUT / f"{ident}.log"
        if not raw.exists() or not csv_path.exists():
            cmd = [
                str(NCU),
                "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum",
                "--cache-control", "all", "--clock-control", "none",
                "--launch-skip", str(protocol.get('launch_skip', 9)),
                "--launch-count", str(protocol.get('launch_count', 6)),
                "--csv", "--log-file", str(csv_path),
            ]
            if exact_q8:
                cmd += [sys.executable, str(probe), '--quant', fmt,
                        '--m', str(shape[0]), '--n', str(shape[1]), '--k', str(shape[2]),
                        '--warmup', str(protocol['warmup']), '--repeats', str(protocol['repeats']),
                        '--seed', str(protocol['seed']), '--output', str(raw)]
            else:
                cmd += [str(EXE), "--device", "cuda", "--quant", fmt,
                "--m", str(shape[0]), "--n", str(shape[1]), "--k", str(shape[2]),
                "--threads", "16", "--warmup", "3", "--repeats", "20",
                "--samples", "32", "--seed", "20260930", "--atol", ".05", "--rtol", ".03",
                "--run", "--output", str(raw),
                ]
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=handle,
                               timeout=int(protocol["per_case_timeout_s"]), check=True)
        row = _parse_ncu(csv_path, raw, fmt, shape, split)
        if row['runtime_binary_sha256'] != protocol.get('runtime_binary_sha256', row['runtime_binary_sha256']):
            raise RuntimeError('runtime identity changed')
        rows.append(row)
        MEASUREMENTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(cases)}] {ident} {row['median_ns']:.0f} ns cv={row['cv_pct']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
