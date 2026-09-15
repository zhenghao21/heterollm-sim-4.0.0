"""Collect kernel/API/memory evidence for the native llama.cpp parity run.

The profiler is intentionally separate from ``native_llama_compare.py``.  It
wraps the exact same llama-server command with Nsight Systems, performs one
uncached warmup and a barrier-synchronised formal request cohort, then exports
machine-readable summaries. Nsight's aggregate numbers are mechanism evidence;
they are never treated as one operator's service time or an uninstrumented
latency benchmark.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from heterollm_sim.gguf_parity import read_gguf_metadata, build_model_from_gguf, compare_gguf_to_model, assert_gguf_parity
try:
    from native_llama_compare import (
        probe_hardware,
        _trace_artifact_ref,
        post_parallel_json,
        post_parallel_stream_json,
        get_json,
        _aggregate_request_records,
        _native_request_record,
    )
except ModuleNotFoundError:  # import as ``tools.native_llama_profile`` in tests
    from tools.native_llama_compare import (
        probe_hardware,
        _trace_artifact_ref,
        post_parallel_json,
        post_parallel_stream_json,
        get_json,
        _aggregate_request_records,
        _native_request_record,
    )

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


def _row_name(row: dict[str, object]) -> str | None:
    """Return an operator/kernel name from any supported nsys CSV spelling."""
    for key in ("Name", "name", "Kernel Name", "kernel_name", "API Name", "api_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _formal_request_records(
    responses: list[dict], boundaries: list[dict[str, object]], requested_output: int,
) -> list[dict[str, object]]:
    """Normalize every formal response while retaining one record per request."""
    records: list[dict[str, object]] = []
    for index, response in enumerate(responses):
        boundary = boundaries[index] if index < len(boundaries) else {}
        record = _native_request_record(response, boundary, index, requested_output)
        # The profiling payload is the authoritative place for the slot and
        # request association. Different llama.cpp revisions use different
        # response keys, so preserve whichever one is present and fail closed
        # when none is available.
        slot = next((response.get(key) for key in ("slot_id", "slot", "slot_index")
                     if response.get(key) is not None), None)
        if slot is None:
            slot = next((boundary.get(key) for key in ("slot_id", "slot", "slot_index")
                         if boundary.get(key) is not None), None)
        if isinstance(slot, (int, str)) and not isinstance(slot, bool):
            record["slot_id"] = slot
            record["slot_mapping_status"] = "observed"
            record["slot_mapping_source"] = "response_or_boundary"
        else:
            record["slot_id"] = None
            record["slot_mapping_status"] = "unavailable"
            record["slot_mapping_source"] = None
        invocation_ids = response.get("invocation_ids", boundary.get("invocation_ids", []))
        record["invocation_ids"] = invocation_ids if isinstance(invocation_ids, list) else []
        record["invocation_mapping_status"] = "observed" if record["invocation_ids"] else "unavailable"
        record["invocation_mapping_source"] = "response_or_boundary" if record["invocation_ids"] else None
        record["response_keys"] = sorted(str(key) for key in response.keys())
        records.append(record)
    return records


def _profile_coverage(
    *, responses: list[dict], boundaries: list[dict[str, object]],
    requested_parallel: int, requested_output: int,
    formal_records: list[dict[str, object]] | None = None,
    stats: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Summarize evidence completeness; kernel rows alone are insufficient."""
    records = formal_records if formal_records is not None else _formal_request_records(
        responses, boundaries, requested_output
    )
    observed_requests = len(responses)
    # Counters and client arrivals do not establish semantic trace phases.
    # Report response completeness separately; raw trace attribution is needed
    # before any phase can claim complete operator coverage.
    request_boundary_observed = sum(item.get("status") == "measured" for item in boundaries)
    phase_coverage = {name: {"expected_requests": requested_parallel,
                             "observed_requests": 0, "status": "incomplete",
                             "reason": "raw semantic trace association unavailable in summary CSV"}
                      for name in ("prompt_eval", "decode", "engine_request_boundary")}
    stat_summary: dict[str, object] = {}
    total_rows = 0
    unknown_rows = 0
    mapping_failures = 0
    semantic_unverified = 0
    owner_unverified = 0
    for kind, payload in stats.items():
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        rows = rows if isinstance(rows, list) else []
        kind_unknown = 0
        kind_mapping_failures = 0
        kind_semantic_unverified = 0
        kind_owner_unverified = 0
        for row in rows:
            if not isinstance(row, dict):
                kind_unknown += 1
                continue
            name = _row_name(row)
            semantic_value = row.get("semantic_status", row.get("mapping_status"))
            semantic = str(semantic_value or "").lower()
            if name is None or name.lower() in {"unknown", "unresolved", ""}:
                kind_unknown += 1
            if semantic_value is None:
                kind_semantic_unverified += 1
            # Owner/stage mapping is required for kernel attribution. API and
            # memcpy tables are retained as timing diagnostics and do not need
            # an operator owner.
            if kind == "kernel":
                owner = row.get("owner", row.get("stage", row.get("operator")))
                if not isinstance(owner, str) or not owner.strip():
                    kind_owner_unverified += 1
            if semantic in {"unknown", "unmatched", "mapping_failure", "failed", "unresolved"}:
                kind_mapping_failures += 1
        total_rows += len(rows)
        unknown_rows += kind_unknown
        mapping_failures += kind_mapping_failures
        semantic_unverified += kind_semantic_unverified
        owner_unverified += kind_owner_unverified
        stat_summary[kind] = {
            "rows": len(rows),
            "unknown_rows": kind_unknown,
            "mapping_failure_rows": kind_mapping_failures,
            "semantic_unverified_rows": kind_semantic_unverified,
            "owner_unverified_rows": kind_owner_unverified,
            "known_row_ratio": ((len(rows) - kind_unknown) / len(rows) if rows else 0.0),
            "owner_mapping_ratio": ((len(rows) - kind_owner_unverified) / len(rows)
                                    if rows and kind == "kernel" else None),
        }
    slot_observed = sum(item.get("slot_mapping_status") == "observed" for item in records)
    invocation_observed = sum(item.get("invocation_mapping_status") == "observed" for item in records)
    request_coverage_complete = observed_requests == requested_parallel
    phase_coverage_complete = all(item["status"] == "complete" for item in phase_coverage.values())
    request_mapping_complete = (
        observed_requests > 0 and slot_observed == observed_requests
        and invocation_observed == observed_requests
    )
    row_coverage_complete = (
        total_rows > 0 and unknown_rows == 0 and mapping_failures == 0
        and owner_unverified == 0 and semantic_unverified == 0
    )
    return {
        "requested_parallel": requested_parallel,
        "formal_request_count": observed_requests,
        "request_count_status": "complete" if request_coverage_complete else "incomplete",
        "phase_coverage": phase_coverage,
        "client_boundary_observed_requests": request_boundary_observed,
        "slot_mapping": {
            "observed_requests": slot_observed,
            "expected_requests": observed_requests,
            "status": "complete" if observed_requests and slot_observed == observed_requests else "unavailable",
        },
        "invocation_mapping": {
            "observed_requests": invocation_observed,
            "expected_requests": observed_requests,
            "status": "complete" if observed_requests and invocation_observed == observed_requests else "unavailable",
        },
        "stats": stat_summary,
        "total_stat_rows": total_rows,
        "unknown_row_count": unknown_rows,
        "mapping_failure_count": mapping_failures,
        "semantic_unverified_row_count": semantic_unverified,
        "owner_unverified_row_count": owner_unverified,
        "request_mapping_status": "complete" if request_mapping_complete else "incomplete",
        "unknown_row_ratio": unknown_rows / total_rows if total_rows else 1.0,
        "mapping_failure_ratio": mapping_failures / total_rows if total_rows else 1.0,
        "evidence_status": "complete" if (
            request_coverage_complete and phase_coverage_complete
            and request_mapping_complete and row_coverage_complete
        ) else "incomplete",
    }


def _validity_status(
    *, coverage: dict[str, object], stats: dict[str, dict[str, object]],
    capture_mode: str,
) -> str:
    """Classify evidence using request and semantic coverage, never row count alone."""
    kernel_payload = stats.get("kernel", {})
    kernel_rows = kernel_payload.get("rows", []) if isinstance(kernel_payload, dict) else []
    if not isinstance(kernel_rows, list) or not kernel_rows:
        return "evidence_missing"
    if capture_mode == "benchmark":
        # This executable path is Nsight-instrumented; it cannot be used as
        # an uninstrumented latency benchmark even when all rows are present.
        return "benchmark_evidence_ineligible"
    return "diagnostic_evidence_complete" if coverage.get("evidence_status") == "complete" else "diagnostic_evidence_incomplete"


def _profiling_perturbation_declaration(capture_mode: str) -> dict[str, object]:
    return {
        "capture_mode": capture_mode,
        "instrumentation": ["nsys", "cuda", "nvtx", "wddm"],
        "expected_overhead": "nonzero_and_unmeasured" if capture_mode == "diagnostic" else "unknown",
        "benchmark_eligible": False,
        "statement": "Nsight/CUDA/NVTX collection is mechanism evidence with possible observer effect; it must not be used as an uninstrumented latency benchmark.",
    }


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
    ap.add_argument("--request-timing", choices=("stream", "none"), default="stream",
                    help="正式请求边界；stream 复用并发 barrier runner，none 仅收集非流式响应")
    ap.add_argument("--capture-mode", choices=("diagnostic", "benchmark"), default="diagnostic",
                    help="Nsight/CUDA 采集默认属于 diagnostic；benchmark 只作为不合资格声明")
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
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if args.predict < 1:
        raise SystemExit("--predict must be >= 1")
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
    formal_responses: list[dict] = []
    formal_boundaries: list[dict[str, object]] = []
    slots_before = slots_after = None
    capture_start = capture_stop = None
    try:
        wait_health(base, proc)
        try:
            slots_before = get_json(base + "/slots")
        except Exception:
            slots_before = None
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
        formal_payload = {"prompt": args.prompt, "n_predict": args.predict,
                          "temperature": 0.0, "top_k": 1, "seed": args.seed,
                          "cache_prompt": False,
                          "stream": args.request_timing == "stream",
                          "ignore_eos": args.output_mode == "fixed"}
        if args.request_timing == "stream":
            pairs = post_parallel_stream_json(base + "/completion", formal_payload, args.parallel)
            formal_responses = [pair[0] for pair in pairs]
            formal_boundaries = [pair[1] for pair in pairs]
        else:
            formal_responses = post_parallel_json(base + "/completion", formal_payload, args.parallel)
            formal_boundaries = [{"mode": "disabled", "status": "unavailable"}
                                 for _ in formal_responses]
        formal = formal_responses[0] if formal_responses else {}
        try:
            slots_after = get_json(base + "/slots")
        except Exception:
            slots_after = None
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
    formal_records = _formal_request_records(formal_responses, formal_boundaries, args.predict)
    formal_aggregate = _aggregate_request_records(
        formal_records,
        batch_client_wall_ms=(formal_boundaries[0].get("batch_client_wall_ms")
                              if formal_boundaries else None),
    )
    coverage = _profile_coverage(
        responses=formal_responses,
        boundaries=formal_boundaries,
        requested_parallel=args.parallel,
        requested_output=args.predict,
        formal_records=formal_records,
        stats=stats,
    )
    binary_artifacts_after = [_trace_artifact_ref(path) for path in binary_paths]
    if binary_artifacts_after != binary_artifacts:
        raise RuntimeError("native binary or DLL changed during profiling")
    result = {
        "schema": "native-llama-nsys-profile/v2",
        "capture_mode": args.capture_mode,
        "validity_status": _validity_status(coverage=coverage, stats=stats, capture_mode=args.capture_mode),
        "formal_runner": {
            "name": "post_parallel_stream_json" if args.request_timing == "stream" else "post_parallel_json",
            "barrier": True,
            "requested_parallel": args.parallel,
            "actual_requests": len(formal_responses),
        },
        "command": nsys_cmd,
        "server_command": server_cmd,
        "native_binary_artifacts": binary_artifacts,
        "native_binary_identity_status": "captured_before_and_after",
        "environment": {"GGML_CUDA_DISABLE_GRAPHS": "1", "note": "runtime hint is ineffective when GGML_CUDA_GRAPHS was compiled in; use a GGML_CUDA_GRAPHS=OFF binary"} if args.disable_graphs else {},
        "capture_start_command": start_cmd,
        "capture_start": {"stdout": capture_start.stdout, "stderr": capture_start.stderr} if capture_start is not None else None,
        "capture_settle_s": args.capture_settle_s,
        "capture_stop": {"stdout": capture_stop.stdout, "stderr": capture_stop.stderr} if capture_stop is not None else None,
        "model": str(model.resolve()),
        "request": {"prompt": args.prompt, "requested_output_tokens": args.predict,
                    "output_mode": args.output_mode, "ignore_eos": args.output_mode == "fixed",
                    "warmup_output_tokens": args.warmup_predict,
                    "requested_parallel": args.parallel,
                    "request_timing": args.request_timing},
        "output_policy": {"mode": args.output_mode, "ignore_eos": args.output_mode == "fixed"},
        "hardware": probe_hardware(),
        "gguf": parity,
        "warmup": {"timings": (warmup or {}).get("timings", {})},
        "slots_before": slots_before,
        "slots_after": slots_after,
        "formal": {"timings": timings, "prompt_n": timings.get("prompt_n"), "predicted_n": timings.get("predicted_n"),
                   "request_count": len(formal_responses),
                   "requests": formal_records,
                   "aggregate": formal_aggregate},
        "formal_request_count": len(formal_responses),
        "formal_requests": formal_records,
        "formal_aggregate": formal_aggregate,
        "coverage": coverage,
        "profiling_perturbation": _profiling_perturbation_declaration(args.capture_mode),
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
