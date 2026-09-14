"""Collect kernel/API/memory evidence for the native llama.cpp parity run.

The profiler is intentionally separate from ``native_llama_compare.py``.  It
wraps the exact same llama-server command with Nsight Systems, performs one
uncached warmup and one formal completion, then exports machine-readable
summaries.  Nsight's aggregate numbers are evidence for calibration; they are
never treated as one operator's service time.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heterollm_sim.gguf_parity import read_gguf_metadata, build_model_from_gguf, compare_gguf_to_model, assert_gguf_parity
from native_llama_compare import probe_hardware, _trace_artifact_ref

DEFAULT_EXE = r"C:\Users\A\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.33.0\llama-server.exe"
DEFAULT_MODEL = r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\native_benchmark_20260912\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe"


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_health(base: str, proc: subprocess.Popen) -> None:
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"profiled llama-server exited with code {proc.returncode}")
        try:
            with urlopen(base + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("llama-server health check timed out")


def run_stats(nsys: Path, report: Path, report_name: str, output: Path) -> dict:
    """Export one Nsight Systems report as CSV and retain command evidence."""
    report = report.resolve()
    output = output.resolve()
    cmd = [str(nsys), "stats", "--report", report_name, "--format", "csv",
           "--output", str(output), "--force-overwrite", "true",
           "--force-export", "true", str(report)]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    # Nsight writes the CSV through --output.  Older builds may still emit it
    # on stdout, so retain that as a fallback for a portable artifact.
    generated = list(output.parent.glob(output.name + "_*.csv"))
    if generated and (not output.exists() or output.stat().st_size == 0):
        generated[0].replace(output)
    if not output.exists() or output.stat().st_size == 0:
        output.write_text(completed.stdout, encoding="utf-8")
    return {"command": cmd, "returncode": completed.returncode, "stderr": completed.stderr[-4000:], "csv": str(output)}


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    # Nsight may prepend a short comment/header block before the CSV header.
    start = next((i for i, line in enumerate(lines) if line.startswith('"') or "," in line), 0)
    try:
        return list(csv.DictReader(lines[start:]))
    except csv.Error:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.environ.get("LLAMA_SERVER_EXE", DEFAULT_EXE))
    ap.add_argument("--model", default=os.environ.get("LLAMA_GGUF", DEFAULT_MODEL))
    ap.add_argument("--nsys", default=os.environ.get("NSYS_EXE", DEFAULT_NSYS))
    ap.add_argument("--prompt", default="Explain why deterministic benchmarking matters.")
    ap.add_argument("--predict", type=int, default=8)
    ap.add_argument("--output-mode", choices=("natural", "fixed"), default="natural",
                    help="输出策略；fixed 发送 ignore_eos=True，确保 phase 微基准按请求长度完成")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ubatch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-predict", type=int, default=2)
    ap.add_argument("--capture-settle-s", type=float, default=0.5,
                    help="seconds to let Nsight start before the formal request")
    ap.add_argument("--disable-graphs", action="store_true",
                    help="record a legacy GGML_CUDA_DISABLE_GRAPHS hint; use a binary built with GGML_CUDA_GRAPHS=OFF for direct dispatch")
    ap.add_argument("--port", type=int, default=18482)
    ap.add_argument("--output", type=Path, default=Path("artifacts/native_profile.json"))
    args = ap.parse_args()
    exe, model, nsys = Path(args.exe), Path(args.model), Path(args.nsys)
    if not exe.exists() or not model.exists() or not nsys.exists():
        raise SystemExit("llama-server、GGUF 或 nsys 路径不存在")
    binary_paths = [exe.resolve(), *sorted(exe.resolve().parent.glob("*.dll"))]
    binary_artifacts = [_trace_artifact_ref(path) for path in binary_paths]
    gguf = read_gguf_metadata(model)
    parity = compare_gguf_to_model(gguf, build_model_from_gguf(gguf), context_length=args.ctx)
    assert_gguf_parity(parity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_base = args.output.with_suffix("").resolve()
    server_log = args.output.with_suffix(".llama.log")
    nsys_report = report_base.with_suffix(".nsys-rep")
    server_cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(args.port),
        "-c", str(args.ctx), "-ngl", str(args.gpu_layers), "-np", str(args.parallel),
        "-b", str(args.batch), "-ub", str(args.ubatch), "-t", str(args.threads), "-tb", str(args.threads),
        "-fa", "off", "--load-mode", "mmap", "-kvo", "--op-offload", "-sm", "layer", "-mg", "0",
        "-ctk", "f16", "-ctv", "f16", "-kvu", "-cb", "--perf", "--metrics", "--warmup", "--spec-type", "none"]
    session = f"llamasim{os.getpid()}"
    nsys_cmd = [str(nsys), "launch", f"--session-new={session}", "--trace=cuda,nvtx,wddm",
        "--cuda-graph-trace=node", "--cuda-memory-usage=true", "--wait=all", "--", *server_cmd]
    start_cmd = [str(nsys), "start", f"--session={session}",
        "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "--export=sqlite", "-o", str(report_base)]
    started = time.time()
    profile_env = os.environ.copy()
    if args.disable_graphs:
        profile_env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    with server_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(nsys_cmd, stdout=log, stderr=log, env=profile_env)
    base = f"http://127.0.0.1:{args.port}"
    warmup = formal = None
    try:
        wait_health(base, proc)
        warmup = post_json(base + "/completion", {"prompt": args.prompt, "n_predict": max(1, args.warmup_predict), "temperature": 0.0, "top_k": 1, "seed": args.seed, "cache_prompt": False, "stream": False, "ignore_eos": args.output_mode == "fixed"})
        capture_start = subprocess.run(start_cmd, capture_output=True, text=True, timeout=60)
        if capture_start.returncode:
            raise RuntimeError("Nsight collection start failed: " + capture_start.stderr)
        # ``nsys start`` acknowledges the session before all CUDA/NVTX
        # collectors are necessarily attached.  Without a short settle
        # interval the first part of prefill can be absent from the trace,
        # which makes an apparently complete kernel table an invalid phase
        # coverage claim.  This delay is outside the native timing payload;
        # it only protects the diagnostic capture boundary.
        if args.capture_settle_s > 0:
            time.sleep(args.capture_settle_s)
        formal = post_json(base + "/completion", {"prompt": args.prompt, "n_predict": args.predict, "temperature": 0.0, "top_k": 1, "seed": args.seed, "cache_prompt": False, "stream": False, "ignore_eos": args.output_mode == "fixed"})
        capture_stop = subprocess.run([str(nsys), "stop", f"--session={session}"], capture_output=True, text=True, timeout=60)
    finally:
        shutdown = subprocess.run([str(nsys), "shutdown", f"--session={session}", "--kill=true"], capture_output=True, text=True, timeout=30)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)
    # Nsight writes report files after the wrapped process exits.
    deadline = time.time() + 60
    while time.time() < deadline and not nsys_report.exists():
        time.sleep(0.5)
    stat_specs = {
        "kernel": "cuda_gpu_kern_sum",
        "api": "cuda_api_sum",
        "memcpy": "cuda_gpu_mem_time_sum",
    }
    stats = {}
    for key, name in stat_specs.items():
        csv_path = args.output.with_name(args.output.stem + f".{key}.csv")
        stats[key] = run_stats(nsys, nsys_report, name, csv_path)
        stats[key]["rows"] = csv_rows(csv_path)
    timings = (formal or {}).get("timings", {})
    binary_artifacts_after = [_trace_artifact_ref(path) for path in binary_paths]
    if binary_artifacts_after != binary_artifacts:
        raise RuntimeError("native binary or DLL changed during profiling")
    result = {
        "schema": "native-llama-nsys-profile/v1",
        "validity_status": "profiled_stage_evidence" if stats["kernel"]["rows"] else "kernel_evidence_missing",
        "command": nsys_cmd,
        "server_command": server_cmd,
        "native_binary_artifacts": binary_artifacts,
        "native_binary_identity_status": "captured_before_and_after",
        "environment": {"GGML_CUDA_DISABLE_GRAPHS": "1", "note": "runtime hint is ineffective when GGML_CUDA_GRAPHS was compiled in; use a GGML_CUDA_GRAPHS=OFF binary"} if args.disable_graphs else {},
        "capture_start_command": start_cmd,
        "capture_start": {"stdout": capture_start.stdout, "stderr": capture_start.stderr} if 'capture_start' in locals() else None,
        "capture_settle_s": args.capture_settle_s,
        "capture_stop": {"stdout": capture_stop.stdout, "stderr": capture_stop.stderr} if 'capture_stop' in locals() else None,
        "model": str(model.resolve()),
        "request": {"prompt": args.prompt, "requested_output_tokens": args.predict,
                    "output_mode": args.output_mode, "ignore_eos": args.output_mode == "fixed",
                    "warmup_output_tokens": args.warmup_predict},
        "output_policy": {"mode": args.output_mode, "ignore_eos": args.output_mode == "fixed"},
        "hardware": probe_hardware(),
        "gguf": parity,
        "warmup": {"timings": (warmup or {}).get("timings", {})},
        "formal": {"timings": timings, "prompt_n": timings.get("prompt_n"), "predicted_n": timings.get("predicted_n")},
        "report": {"nsys_rep": str(nsys_report), "server_log": str(server_log), "elapsed_wall_s": time.time() - started},
        "stats": stats,
        "calibration_scope": {
            "prompt_eval": "formal request only; collection starts after warmup; prompt/decode split still requires NVTX markers",
            "decode_eval": "formal request only; collection starts after warmup; llama.cpp aggregate prompt/eval timings remain stage boundaries",
            "launch_and_sync": "cuda_api_sum rows, never folded into GEMM demand",
            "host_output": "cuda_gpu_mem_time_sum plus llama.cpp timing; D2H attribution requires explicit memcpy rows",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
