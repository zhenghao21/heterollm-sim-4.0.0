"""Collect an independent Level-2 MMQ prefill shape grid.

The harness is the standalone generic GGML GEMM executable.  The collector
keeps only the CUDA ``mul_mat_q`` main-kernel interval; activation repacking
and the separate stream-K fixup remain distinct owners in the simulator.
No GGUF, native LLM timing, or target-model latency is read here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
OUT = ROOT / "artifacts/development/level2_mmq_prefill_20260930"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: str) -> float:
    return float(str(value).replace(",", "").strip())


def _tuple_geometry(value: str) -> tuple[int, int, int]:
    value = str(value).strip().strip("()")
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 3:
        raise ValueError(f"invalid CUDA geometry: {value!r}")
    return parts


def _kernel_rows(csv_path: Path) -> list[dict]:
    text = csv_path.read_text(encoding="utf-8-sig")
    marker = '"ID"'
    if marker not in text:
        raise RuntimeError(f"Nsight CSV has no metric header: {csv_path}")
    reader = csv.DictReader(io.StringIO(text[text.index(marker):]))
    groups: dict[str, dict[str, object]] = {}
    for event in reader:
        name = " ".join(str(event.get("Kernel Name", "")).split())
        if "mul_mat_q<" not in name or "stream_k_fixup" in name:
            continue
        ident = str(event.get("ID", ""))
        row = groups.setdefault(ident, {
            "kernel_name": name,
            "block": _tuple_geometry(event["Block Size"]),
            "grid": _tuple_geometry(event["Grid Size"]),
            "metrics": {},
        })
        metric = str(event.get("Metric Name", ""))
        if metric in {
            "gpu__time_duration.sum",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_allocated",
            "launch__shared_mem_per_block_dynamic",
            "launch__shared_mem_per_block_static",
        }:
            row["metrics"][metric] = _number(event["Metric Value"])
    rows = []
    for ident, row in sorted(groups.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        required = {"gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum"}
        if not required <= set(row["metrics"]):
            continue
        rows.append({"id": ident, **row})
    return rows


def _parse_case(raw_path: Path, csv_path: Path, fmt: str, shape: tuple[int, int, int], split: str) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("status") != "measured" or not raw.get("modules_stable"):
        raise RuntimeError(f"measurement status gate failed: {raw_path}")
    if not raw.get("first_call_correctness", {}).get("passed") or not raw.get("final_correctness", {}).get("passed"):
        raise RuntimeError(f"numerical correctness gate failed: {raw_path}")
    main = _kernel_rows(csv_path)
    if len(main) < 2:
        raise RuntimeError(f"MMQ main kernel launch count gate failed: {csv_path}")
    # NCU launch filtering can include an incomplete first row.  Keep the last
    # formal launches, which are the repeated fixed-buffer measurements.
    formal = main[-min(6, len(main)):]
    times = [float(row["metrics"]["gpu__time_duration.sum"]) for row in formal]
    bandwidth = [
        (row["metrics"]["dram__bytes_read.sum"] + row["metrics"]["dram__bytes_write.sum"]) / time
        for row, time in zip(formal, times)
    ]
    if len({row["kernel_name"] for row in formal}) != 1:
        raise RuntimeError(f"MMQ specialization changed across launches: {csv_path}")
    geometry = formal[-1]
    modules = raw.get("loaded_modules_after", ())
    if isinstance(modules, dict):
        module = modules.get("ggml-cuda.dll")
        if module is None:
            module = next((value for key, value in modules.items()
                           if str(key).casefold() == "ggml-cuda.dll"), None)
    else:
        module = next((item for item in modules
                       if isinstance(item, dict)
                       and str(item.get("name", "")).casefold() == "ggml-cuda.dll"), None)
    if module is None:
        raise RuntimeError(f"ggml-cuda.dll identity missing: {raw_path}")
    metrics = geometry["metrics"]
    return {
        "format": fmt,
        "shape": list(shape),
        "phase": "prefill",
        "kernel_family": "mmq",
        "split": split,
        "symbol": geometry["kernel_name"],
        "median_ns": statistics.median(times),
        "cv_pct": statistics.stdev(times) / statistics.mean(times) * 100.0,
        "samples_ns": times,
        "bandwidth_gb_s": statistics.median(bandwidth),
        "geometry": {
            "block": geometry["block"],
            "grid": geometry["grid"],
            "registers_per_thread": metrics.get("launch__registers_per_thread"),
            "shared_memory_per_block": metrics.get("launch__shared_mem_per_block_allocated"),
            "dynamic_shared_memory": metrics.get("launch__shared_mem_per_block_dynamic"),
            "static_shared_memory": metrics.get("launch__shared_mem_per_block_static"),
        },
        "csv_sha256": sha256(csv_path),
        "raw_sha256": sha256(raw_path),
        "runtime_binary_sha256": module["sha256"],
        "measurement_boundary": "cuda_device_main_mmq_kernel_interval",
        "separate_kernels": ["quantize_mmq_q8_1", "mul_mat_q", "mul_mat_q_stream_k_fixup"],
        "cache_control": "all",
        "clock_control": "none",
        "target_llm_timing_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    exact_q8 = protocol.get("harness") == "python_exact_q8_mmq"
    if exact_q8:
        probe = ROOT / "tools/probe_synthetic_mmq.py"
        if sha256(probe) != protocol["probe_sha256"]:
            raise RuntimeError("exact-Q8 probe identity changed")
    elif sha256(EXE) != protocol["executable_sha256"]:
        raise RuntimeError("generic microbenchmark identity changed")
    if not NCU.exists():
        raise RuntimeError(f"Nsight Compute not found: {NCU}")
    rows: list[dict] = []
    env = dict(os.environ)
    env["PATH"] = str(DLL_DIR) + ";E:/cuda/bin;" + env.get("PATH", "")
    cases = []
    excluded = {(item["format"], tuple(item["shape"]))
                for item in protocol.get("excluded_cases", ())}
    for split in ("train", "holdout"):
        by_format = protocol[
            "training_shapes_by_format" if split == "train" else "holdout_shapes_by_format"
        ]
        for fmt, shapes in by_format.items():
            for raw_shape in shapes:
                shape = tuple(int(value) for value in raw_shape)
                if (fmt, shape) not in excluded:
                    cases.append((split, fmt, shape))
    metrics = ",".join((
        "dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum",
        "launch__registers_per_thread", "launch__shared_mem_per_block_allocated",
        "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static",
    ))
    for index, (split, fmt, shape) in enumerate(cases, 1):
        ident = f"{fmt}_{shape[0]}_{shape[1]}_{shape[2]}"
        raw = out / f"{ident}.json"
        csv_path = out / f"{ident}.csv"
        log = out / f"{ident}.log"
        if not raw.exists() or not csv_path.exists():
            cmd = [
                str(NCU), "--metrics", metrics,
                "--cache-control", "all", "--clock-control", "none",
                "--launch-skip", str(protocol.get("launch_skip", 9)),
                "--launch-count", str(protocol.get("launch_count", 6)),
                "--csv", "--log-file", str(csv_path),
            ]
            if exact_q8:
                cmd += [sys.executable, str(probe), "--quant", fmt,
                        "--m", str(shape[0]), "--n", str(shape[1]), "--k", str(shape[2]),
                        "--warmup", str(protocol["warmup"]), "--repeats", str(protocol["repeats"]),
                        "--seed", str(protocol["seed"]),
                        "--atol", str(protocol.get("correctness_atol", .001)),
                        "--rtol", str(protocol.get("correctness_rtol", .001)),
                        "--output", str(raw)]
            else:
                cmd += [str(EXE), "--device", "cuda", "--quant", fmt,
                        "--m", str(shape[0]), "--n", str(shape[1]), "--k", str(shape[2]),
                        "--threads", "16", "--warmup", str(protocol["warmup"]),
                        "--repeats", str(protocol["repeats"]), "--samples", "32",
                        "--seed", str(protocol["seed"]), "--atol", ".05", "--rtol", ".03",
                        "--run", "--output", str(raw)]
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=handle,
                               timeout=int(protocol["per_case_timeout_s"]), check=True)
        row = _parse_case(raw, csv_path, fmt, shape, split)
        if row["runtime_binary_sha256"] != protocol["runtime_binary_sha256"]:
            raise RuntimeError("runtime identity changed")
        rows.append(row)
        (out / "measurements.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{index}/{len(cases)}] {ident} {row['median_ns']:.0f} ns cv={row['cv_pct']:.2f}% {row['symbol']}", flush=True)


if __name__ == "__main__":
    main()
