"""Run one reproducible native llama.cpp measurement beside the simulator.

The script intentionally uses the same token counts, context, batch, ubatch,
threads, KV type, GPU layer count, and parallel slots for both sides.  It is a
small audit harness; it does not pretend that llama.cpp's kernel timing is a
calibration of the analytical cost model.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
from dataclasses import replace
import sys
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
import math
from collections.abc import Mapping
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.model_presets import materialize_model_payload
from heterollm_sim.config import model_from_dict
from heterollm_sim.ir import model_graph_execution_view
from heterollm_sim.reporting import run_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.llama_scenario import apply_llama_runtime_config
from heterollm_sim.gguf_parity import read_gguf_metadata, compare_gguf_to_model, assert_gguf_parity, build_model_from_gguf
from heterollm_sim.calibration import load_native_calibration, apply_native_calibration
from heterollm_sim.serde import stable_hash


DEFAULT_EXE = r"C:\Users\A\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.33.0\llama-server.exe"
DEFAULT_MODEL = r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0\artifacts\native_benchmark_20260912\models\qwen2.5-0.5b-instruct-q4_k_m.gguf"


def _native_measurements_digest(native: Mapping[str, object]) -> str:
    """Bind captured counters, token boundaries and extractor output together."""
    # ``evidence.native_measurements_sha256`` is stored inside the native
    # object itself, so hashing the object including ``evidence`` would be
    # self-referential and could never be reproduced during replay.  Hash the
    # measured payload only; the surrounding evidence manifest binds this
    # digest to the contract, binary and extractor identities separately.
    if not isinstance(native, Mapping):
        return stable_hash({})
    measured = dict(native)
    measured.pop("evidence", None)
    return stable_hash(measured)


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def post_stream_json(url: str, payload: dict) -> tuple[dict, dict[str, object]]:
    """Run a streaming completion and measure request lifecycle boundaries.

    ``prompt_ms`` in the locked llama.cpp build is the slot engine counter from
    ``t_start`` to ``t_prompt_last`` (the first sample/accept boundary).  It
    deliberately does not include admission, HTTP, or server-side scheduling.
    A streamed completion gives this harness an independent client request clock: the clock
    starts immediately before the POST and the first non-empty ``content``
    chunk is treated as the first visible token.  The final chunk's timing
    object is retained so existing token/perf parsing keeps working.

    The parser accepts both llama.cpp SSE (``data: {...}``) and a plain JSON
    response.  The latter is useful with older servers that silently ignore
    ``stream=true``; in that case only the client boundary is unavailable.
    The engine counter remains available from the final timing object when
    the locked server emits it.
    """
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    chunks: list[dict] = []
    content_parts: list[str] = []
    first_event_ms: float | None = None
    first_content_ms: float | None = None
    last_content_ms: float | None = None
    first_token_ms: float | None = None
    last_token_ms: float | None = None
    token_source: str | None = None
    token_chunk_times_ms: list[float] = []
    response_mode = "sse"
    with urlopen(req, timeout=180) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                encoded = line[5:].strip()
                if encoded == "[DONE]":
                    break
            else:
                # A non-SSE body is accepted for compatibility, but cannot
                # carry a first-token boundary once the whole body arrived.
                response_mode = "json"
                encoded = line
            try:
                item = json.loads(encoded)
            except json.JSONDecodeError:
                # Some proxies split an SSE JSON object over multiple lines;
                # llama.cpp itself emits one object per line.  Ignore only
                # non-data noise and keep the boundary unavailable if needed.
                continue
            if not isinstance(item, dict):
                continue
            chunks.append(item)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if first_event_ms is None:
                first_event_ms = elapsed_ms
            # Native /completion SSE exposes token IDs even for tokens whose
            # detokenized content is empty (special/EOS tokens).  Prefer that
            # unambiguous token boundary; a content-only older server gets an
            # explicitly labelled first-visible-content fallback.
            has_token_ids = bool(item.get("tokens"))
            if has_token_ids:
                token_chunk_times_ms.append(elapsed_ms)
                last_token_ms = elapsed_ms
                if first_token_ms is None:
                    first_token_ms = elapsed_ms
                    token_source = "stream_first_token_ids"
            piece = item.get("content")
            if piece is not None:
                piece_text = str(piece)
                content_parts.append(piece_text)
                if piece_text and first_content_ms is None:
                    first_content_ms = elapsed_ms
                if piece_text:
                    last_content_ms = elapsed_ms
            if response_mode == "json":
                break
    if not chunks:
        raise ValueError("llama.cpp streaming response contained no JSON chunks")
    # The final SSE chunk normally carries timings and token counters.  Merge
    # all chunks defensively so older builds that attach them earlier remain
    # usable.
    merged: dict[str, object] = {}
    for item in chunks:
        merged.update(item)
    if content_parts:
        merged["content"] = "".join(content_parts)
    if first_token_ms is None and first_content_ms is not None:
        first_token_ms = first_content_ms
        token_source = "stream_first_nonempty_content"
    ended_ms = (time.perf_counter() - started) * 1000.0
    # [DONE] is a transport/control event.  Client E2E ends at the last real
    # token (or the last non-empty content for older servers), while receipt
    # of the stream terminator is retained as separate protocol overhead.
    last_real_token_ms = last_token_ms if last_token_ms is not None else last_content_ms
    boundary = {
        "mode": response_mode,
        "request_start": "client_before_http_post",
        "first_event_ms": first_event_ms,
        "first_content_ms": first_content_ms,
        "request_to_first_token_ms": first_token_ms if response_mode == "sse" else None,
        "request_to_last_token_ms": last_real_token_ms if response_mode == "sse" else None,
        "request_to_end_ms": last_real_token_ms if response_mode == "sse" else None,
        "stream_end_ms": ended_ms,
        "stream_control_overhead_ms": (ended_ms - last_real_token_ms
                                         if response_mode == "sse" and last_real_token_ms is not None else None),
        "first_token_source": token_source if response_mode == "sse" else None,
        "status": "measured" if first_token_ms is not None and response_mode == "sse" else "unavailable",
        "token_chunk_times_ms": token_chunk_times_ms,
        "chunk_count": len(chunks),
    }
    return merged, boundary


def post_parallel_stream_json(url: str, payload: dict, parallel: int) -> list[tuple[dict, dict[str, object]]]:
    """Submit ``parallel`` independent streaming requests behind one barrier."""
    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    if parallel == 1:
        batch_start = time.perf_counter()
        response, boundary = post_stream_json(url, payload)
        batch_end = time.perf_counter()
        boundary.update({"batch_start_monotonic_s": batch_start, "batch_end_monotonic_s": batch_end,
                         "batch_client_wall_ms": (batch_end - batch_start) * 1000.0,
                         "batch_client_makespan_ms": boundary.get("request_to_end_ms")})
        return [(response, boundary)]
    barrier = threading.Barrier(parallel)
    batch_start = time.perf_counter()

    def _one(_: int) -> tuple[dict, dict[str, object]]:
        barrier.wait(timeout=30)
        response, boundary = post_stream_json(url, payload)
        boundary["batch_start_monotonic_s"] = batch_start
        return response, boundary

    with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="llama-request") as pool:
        results = list(pool.map(_one, range(parallel)))
    batch_end = time.perf_counter()
    batch_wall_ms = (batch_end - batch_start) * 1000.0
    for _response, boundary in results:
        boundary["batch_end_monotonic_s"] = batch_end
        boundary["batch_client_wall_ms"] = batch_wall_ms
        boundary["batch_client_makespan_ms"] = max(
            (float(item.get("request_to_end_ms")) for _r, item in results
             if item.get("request_to_end_ms") is not None), default=None
        )
    return results


def post_parallel_json(url: str, payload: dict, parallel: int) -> list[dict]:
    """Submit ``parallel`` non-streaming requests behind one start barrier.

    ``llama-server -np N`` only exercises concurrent slots when requests are
    actually in flight at the same time.  A plain ``pool.map`` can otherwise
    serialize short requests on a fast local server, so keep the same barrier
    semantics as :func:`post_parallel_stream_json` for legacy timing modes.
    """
    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    if parallel == 1:
        return [post_json(url, payload)]
    barrier = threading.Barrier(parallel)

    def _one(_: int) -> dict:
        barrier.wait(timeout=30)
        return post_json(url, payload)

    with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="llama-request") as pool:
        return list(pool.map(_one, range(parallel)))


def _percentile_summary(values: list[float]) -> dict[str, object]:
    """Return stable aggregate statistics for one per-request metric.

    The matrix evaluator consumes p50/p90 and worst-case values.  Keep the
    raw count and min/max as well so an empty or partially measured batch is
    explicit instead of being converted to zero.  ``statistics.quantiles``
    with the inclusive method gives deterministic interpolation for N>1 and
    naturally handles a singleton batch.
    """
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {
            "count": 0,
            "min_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "max_ms": None,
            "mean_ms": None,
        }
    ordered = sorted(finite)
    if len(ordered) == 1:
        p50 = p90 = ordered[0]
    else:
        quantiles = statistics.quantiles(ordered, n=100, method="inclusive")
        p50 = quantiles[49]
        p90 = quantiles[89]
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "p50_ms": p50,
        "p90_ms": p90,
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
    }


def _engine_boundary_timing(boundary: Mapping[str, object] | None,
                            output_tokens: int) -> dict[str, object]:
    """Read a verified engine clock without substituting stage counters.

    The final engine boundary is the last generated token, including sampling;
    a generic request_end after response enqueue is deliberately insufficient.
    """
    unavailable = {"engine_ttft_ms": None, "engine_tpot_ms": None,
                   "engine_e2e_ms": None, "engine_timing_status": "unavailable"}
    if not isinstance(boundary, Mapping) or boundary.get("status") != "measured":
        return unavailable
    values = [boundary.get(name) for name in ("request_begin_ns", "first_token_ns", "last_token_ns")]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        return unavailable
    begin, first, last = map(float, values)
    if not all(math.isfinite(value) for value in (begin, first, last)) or not begin <= first <= last:
        return unavailable
    token_times = boundary.get("token_times_ns")
    if (not isinstance(token_times, list) or len(token_times) != output_tokens
            or not token_times or token_times[0] != first or token_times[-1] != last
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(float(value)) for value in token_times)
            or any(float(a) > float(b) for a, b in zip(token_times, token_times[1:]))):
        return unavailable
    return {"engine_ttft_ms": (first - begin) / 1e6,
            "engine_tpot_ms": (last - first) / (output_tokens - 1) / 1e6 if output_tokens > 1 else None,
            "engine_e2e_ms": (last - begin) / 1e6,
            "engine_timing_status": "measured"}


def _engine_counter_timing(timings: Mapping[str, object], output_tokens: int) -> dict[str, object]:
    """Use llama.cpp ``server_slot_stats`` counters as a proven engine clock.

    In the locked semantic server, ``t_start`` is set when a slot starts
    execution; ``t_prompt_last`` is updated after the first sample/accept and
    ``t_gen_last`` after subsequent generation.  The JSON ``prompt_ms`` and
    ``predicted_ms`` values therefore map to engine TTFT and the remaining
    decode wall, respectively.  They are kept separate from client stream
    timestamps.
    """
    prompt_ms = timings.get("prompt_ms")
    eval_ms = timings.get("predicted_ms")
    if (not isinstance(prompt_ms, (int, float)) or isinstance(prompt_ms, bool)
            or not isinstance(eval_ms, (int, float)) or isinstance(eval_ms, bool)
            or not math.isfinite(float(prompt_ms)) or not math.isfinite(float(eval_ms))
            or float(prompt_ms) < 0 or float(eval_ms) < 0 or output_tokens < 0):
        return {"engine_ttft_ms": None, "engine_tpot_ms": None,
                "engine_e2e_ms": None, "engine_timing_status": "unavailable",
                "engine_timing_source": "llama.cpp.server_slot_stats.unavailable"}
    # n_gen_steps() is n_gen - 1: the first token is produced from prompt
    # logits, so predicted_ms is divided by output_tokens-1 for TPOT.
    return {"engine_ttft_ms": float(prompt_ms),
            "engine_tpot_ms": float(eval_ms) / (output_tokens - 1) if output_tokens > 1 else None,
            "engine_e2e_ms": float(prompt_ms) + float(eval_ms),
            "engine_timing_status": "counter_proven",
            "measurement_status": "complete",
            "measurement_kind": "verified_slot_counters",
            "semantic_validation_status": "verified",
            "timing_contract_id": "engine-boundary/v1",
            "engine_timing_source": "llama.cpp.server_slot_stats.t_start_prompt_last_gen_last"}


def _native_request_record(response: Mapping[str, object],
                           boundary: Mapping[str, object],
                           request_index: int, requested_output: int) -> dict[str, object]:
    """Normalize one native response without discarding its raw timing data."""
    timings = response.get("timings", {})
    if not isinstance(timings, Mapping):
        timings = {}
    prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
    eval_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
    prompt_tokens = int(timings.get("prompt_n", response.get("tokens_evaluated", 0)) or 0)
    output_tokens = int(timings.get("predicted_n", response.get("tokens_predicted", requested_output)) or 0)
    first = boundary.get("request_to_first_token_ms")
    end = boundary.get("request_to_end_ms")
    client_tpot = ((float(end) - float(first)) / (output_tokens - 1)
                   if output_tokens > 1 and first is not None and end is not None else None)
    eval_tpot = eval_ms / (output_tokens - 1) if output_tokens > 1 else None
    engine_boundary = response.get("engine_boundary", boundary.get("engine_boundary"))
    engine = _engine_boundary_timing(engine_boundary, output_tokens)
    if engine["engine_timing_status"] == "unavailable":
        engine = _engine_counter_timing(timings, output_tokens)
    return {
        "request_id": f"request-{request_index:04d}",
        "request_index": request_index,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_eval_ms": prompt_ms,
        "eval_ms": eval_ms,
        "tpot_ms": client_tpot,
        "client_tpot_ms": client_tpot,
        "eval_tpot_ms": eval_tpot,
        "engine_tpot_ms": engine["engine_tpot_ms"],
        "engine_ttft_ms": engine["engine_ttft_ms"],
        "engine_e2e_ms": engine["engine_e2e_ms"],
        "engine_timing_status": engine["engine_timing_status"],
        "engine_timing_source": engine.get("engine_timing_source", "explicit_engine_boundary"),
        "engine_boundary": dict(engine_boundary) if isinstance(engine_boundary, Mapping) else None,
        "total_ms": prompt_ms + eval_ms,
        "ttft_ms": float(first) if first is not None else None,
        "e2e_ms": float(end) if end is not None else None,
        "client_ttft_ms": float(first) if first is not None else None,
        "client_e2e_ms": float(end) if end is not None else None,
        "request_to_first_token_ms": float(first) if first is not None else None,
        "request_to_end_ms": float(end) if end is not None else None,
        "request_boundary": dict(boundary),
    }


def _aggregate_request_records(records: list[Mapping[str, object]],
                               *, batch_client_wall_ms: float | None = None) -> dict[str, object]:
    """Aggregate native or simulator request records using the same schema."""
    metric_names = (
        "prompt_eval_ms", "eval_ms", "tpot_ms", "client_tpot_ms", "engine_tpot_ms",
        "engine_ttft_ms", "engine_e2e_ms", "total_ms", "client_ttft_ms", "client_e2e_ms",
        "request_to_first_token_ms", "request_to_end_ms",
    )
    aggregate: dict[str, object] = {name: _percentile_summary(
        [float(item[name]) for item in records if item.get(name) is not None]
    ) for name in metric_names}
    end_values = [float(item["request_to_end_ms"]) for item in records
                  if item.get("request_to_end_ms") is not None]
    aggregate["request_count"] = len(records)
    aggregate["makespan_ms"] = max(end_values) if end_values else None
    aggregate["batch_client_wall_ms"] = batch_client_wall_ms
    # Friendly aliases match the simulator payload and matrix terminology;
    # the request_* names remain canonical for boundary provenance.
    aggregate["ttft_ms"] = aggregate["request_to_first_token_ms"]
    aggregate["e2e_ms"] = aggregate["request_to_end_ms"]
    # Explicit boundary aliases are the stable API.  Legacy aliases above are
    # retained so existing matrix readers continue to work.
    aggregate["ttft_engine_ms"] = aggregate["engine_ttft_ms"]
    aggregate["tpot_engine_ms"] = aggregate["engine_tpot_ms"]
    aggregate["e2e_engine_ms"] = aggregate["engine_e2e_ms"]
    aggregate["ttft_client_ms"] = aggregate["client_ttft_ms"]
    aggregate["tpot_client_ms"] = aggregate["client_tpot_ms"]
    aggregate["e2e_client_ms"] = aggregate["client_e2e_ms"]
    return aggregate


def _simulator_request_timing(simulation, metric: object) -> dict[str, object]:
    """Project realized tasks onto the engine and client timing contracts.

    Engine timing starts at the first physical prefill invocation after host
    preparation, ends TTFT at the first generated token, and ends E2E at the
    last generated token.  The request-done marker is excluded because it may
    include response queue or transport work.  Prefill batch ends are retained
    only as a diagnostic field.
    """
    request_id = str(getattr(metric, "request_id", ""))
    arrival = getattr(metric, "arrival_ns", None)
    # OnlineScenarioResult retains the serving timeline.  Its serving metrics
    # carry the post-admission engine start; the aggregate RequestMetrics
    # object does not, so recover it by request id and keep a labelled legacy
    # fallback for synthetic/old fixtures.
    serving_metrics = getattr(getattr(simulation, "serving", None), "request_metrics", {}) or {}
    serving_metric = serving_metrics.get(request_id) if isinstance(serving_metrics, Mapping) else None
    start = getattr(serving_metric, "start_ns", None)
    if start is None:
        start = getattr(metric, "start_ns", None)
    first = getattr(metric, "first_token_ns", None)
    finish = getattr(metric, "finish_ns", None)
    tokens = int(getattr(metric, "visible_output_tokens", 0) or 0)
    tpot_ns = getattr(metric, "tpot_ns", None)
    prefill_ends = [float(getattr(batch, "end_ns")) for batch in getattr(simulation.serving, "batches", ()) or ()
                    if str(getattr(batch, "kind", "")).lower() == "prefill"
                    and request_id in tuple(getattr(batch, "request_ids", ()) or ())
                    and getattr(batch, "end_ns", None) is not None]
    prefill_end = max(prefill_ends) if prefill_ends else None
    token_events = [event for event in getattr(simulation.serving, "events", ()) or ()
                    if str(getattr(event, "request_id", "")) == request_id
                    and str(getattr(event, "event_type", "")) == "tokens_committed"]
    last_engine_token = max((float(getattr(event, "timestamp_ns")) for event in token_events
                             if getattr(event, "timestamp_ns", None) is not None), default=None)
    if last_engine_token is None and first is not None and tpot_ns is not None and tokens > 1:
        last_engine_token = float(first) + float(tpot_ns) * (tokens - 1)
    if first is not None and start is not None:
        engine_ttft_ns = float(first) - float(start)
        engine_ttft_source = "simulator.first_engine_token-start"
    else:
        engine_ttft_ns = None
        engine_ttft_source = "unavailable"
    engine_e2e_ns = (last_engine_token - float(start)
                     if last_engine_token is not None and start is not None else None)
    client_ttft_ns = (float(first) - float(arrival)
                      if first is not None and arrival is not None else None)
    client_e2e_ns = (client_ttft_ns + float(tpot_ns) * (tokens - 1)
                     if client_ttft_ns is not None and tpot_ns is not None and tokens > 1 else
                     (float(finish) - float(arrival)
                      if finish is not None and arrival is not None else None))
    return {
        "engine_ttft_ms": engine_ttft_ns / 1e6 if engine_ttft_ns is not None else None,
        "engine_tpot_ms": float(tpot_ns) / 1e6 if tpot_ns is not None else None,
        "engine_e2e_ms": engine_e2e_ns / 1e6 if engine_e2e_ns is not None else None,
        "client_ttft_ms": client_ttft_ns / 1e6 if client_ttft_ns is not None else None,
        "client_tpot_ms": float(tpot_ns) / 1e6 if tpot_ns is not None else None,
        "client_e2e_ms": client_e2e_ns / 1e6 if client_e2e_ns is not None else None,
        "engine_ttft_source": engine_ttft_source,
        "engine_request_begin_ns": start,
        "engine_last_token_ns": last_engine_token,
        "prefill_end_ns": prefill_end,
    }

def get_text(url: str) -> str:
    with urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8", errors="replace")


def get_json(url: str):
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_perf_log(path: Path) -> dict[str, object]:
    """Extract llama.cpp's stage and graph reuse evidence from stderr."""
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    result: dict[str, object] = {"prompt": [], "decode": [], "total": [], "graphs_reused": []}
    patterns = {
        "prompt": r"prompt eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens",
        "decode": r"(?<!prompt )eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens",
        "total": r"total time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens",
        "graphs_reused": r"graphs reused\s*=\s*([0-9]+)",
    }
    for key, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            if key == "graphs_reused":
                result[key].append(int(match.group(1)))
            else:
                result[key].append({"ms": float(match.group(1)), "tokens": int(match.group(2))})
    return result


def metric_snapshot(text: str) -> dict[str, float]:
    """Parse llama.cpp Prometheus counters/gauges without a dependency."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" in line:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def wait_health(base: str, proc: subprocess.Popen) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            with urlopen(base + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("llama-server health check timed out")


def probe_hardware() -> dict[str, object]:
    """Capture a read-only host snapshot used by the native process.

    All commands are inventory queries.  A failed query stays explicit in the
    result so callers cannot accidentally treat an unknown device as the
    RTX 5080 reference profile.
    """
    gpu: dict[str, object] = {
        "name": "unknown", "uuid": None, "memory_mib": None,
        "memory_free_mib": None, "memory_used_mib": None, "driver": None,
        "compute_capability": None, "clocks": {}, "pcie": {},
    }
    gpu_query = (
        "name,uuid,memory.total,memory.free,memory.used,driver_version,"
        "compute_cap,clocks.current.graphics,clocks.current.memory,clocks.current.sm"
    )
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={gpu_query}",
             "--format=csv,noheader,nounits"], text=True, timeout=10,
        )
        rows = list(csv.reader(line for line in raw.splitlines() if line.strip()))
        if rows:
            values = [(item.strip() or None) for item in rows[0]]
            values += [None] * (10 - len(values))
            name, uuid, total, free, used, driver, capability, graphics, memory_clock, sm_clock = values[:10]

            def _int_or_none(value: object) -> int | None:
                if value is None or str(value).casefold() in {"n/a", "na", "unknown"}:
                    return None
                try:
                    return int(float(str(value)))
                except (TypeError, ValueError):
                    return None

            gpu.update({
                "name": name or "unknown", "uuid": uuid,
                "memory_mib": _int_or_none(total),
                "memory_free_mib": _int_or_none(free),
                "memory_used_mib": _int_or_none(used),
                "driver": driver, "compute_capability": capability,
                "clocks": {"graphics_mhz": _int_or_none(graphics),
                            "memory_mhz": _int_or_none(memory_clock),
                            "sm_mhz": _int_or_none(sm_clock)},
            })
    except (OSError, subprocess.SubprocessError, csv.Error):
        pass
    try:
        pcie_raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
             "--format=csv,noheader,nounits"], text=True, timeout=10,
        )
        rows = list(csv.reader(line for line in pcie_raw.splitlines() if line.strip()))
        if rows:
            values = [part.strip() for part in rows[0]]
            if len(values) >= 4:
                def _pcie_int(value: str) -> int | None:
                    try:
                        return int(float(value))
                    except (TypeError, ValueError):
                        return None
                gpu["pcie"] = {
                    "gen_current": _pcie_int(values[0]), "gen_max": _pcie_int(values[1]),
                    "width_current": _pcie_int(values[2]), "width_max": _pcie_int(values[3]),
                }
    except (OSError, subprocess.SubprocessError, csv.Error):
        pass

    cpu_name = "unknown"
    cpu_topology: dict[str, object] = {
        "physical_cores": None, "logical_processors": None,
        "max_clock_mhz": None, "current_clock_mhz": None,
    }
    host_memory: dict[str, object] = {}
    try:
        probe = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "$p=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed); $s=Get-CimInstance Win32_ComputerSystem; $m=@(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,ConfiguredClockSpeed); [pscustomobject]@{processors=$p; total_memory=[int64]$s.TotalPhysicalMemory; modules=$m} | ConvertTo-Json -Compress"],
            text=True, timeout=10,
        ).strip()
        data = json.loads(probe)
        processors = data.get("processors", [])
        if isinstance(processors, Mapping):
            processors = [processors]
        if processors:
            first = processors[0]
            cpu_name = str(first.get("Name") or cpu_name)
            cpu_topology = {
                "physical_cores": sum(int(item.get("NumberOfCores") or 0) for item in processors),
                "logical_processors": sum(int(item.get("NumberOfLogicalProcessors") or 0) for item in processors),
                "max_clock_mhz": int(first.get("MaxClockSpeed") or 0) or None,
                "current_clock_mhz": int(first.get("CurrentClockSpeed") or 0) or None,
            }
        modules = data.get("modules", [])
        if isinstance(modules, Mapping):
            modules = [modules]
        host_memory = {"total_bytes": int(data.get("total_memory", 0)), "modules": modules}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return {
        "cpu": cpu_name,
        "cpu_topology": cpu_topology,
        # Scalar aliases make the snapshot convenient for simple matrix
        # serializers while retaining the historical ``cpu`` string field.
        "cpu_cores": cpu_topology.get("physical_cores"),
        "cpu_threads": cpu_topology.get("logical_processors"),
        "gpu": gpu,
        "host_memory": host_memory,
    }


def _pcie_one_way_bandwidth(snapshot: Mapping[str, object], generation: int, width: int) -> float:
    """Resolve declared PCIe throughput without fitting it from timing error."""
    declared = snapshot.get("bandwidth_gbps_one_way", snapshot.get("bandwidth_gbps"))
    if declared is not None:
        try:
            value = float(declared)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    # PCIe effective GB/s per lane (8b/10b, 128b/130b and 242b/256b coding).
    per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.877, 7: 15.754}.get(generation)
    if per_lane is None:
        raise ValueError(f"unsupported PCIe generation in hardware snapshot: {generation}")
    # The simulator's ``bandwidth_gbps`` contract is decimal gigabits/s;
    # the table above is effective gigabytes/s per lane.
    return per_lane * width * 8.0


def _hardware_inputs(hardware_snapshot: Mapping[str, object] | None) -> dict[str, object]:
    """Normalize and validate the measured input accepted by the parity harness."""
    if hardware_snapshot is None:
        return {
            "cpu_name": "AMD Ryzen 9 9950X3D", "physical_cores": None,
            # Legacy calls have no live topology evidence; auto threads must
            # therefore be rejected instead of borrowing a fixed portrait.
            "logical_processors": None, "gpu_name": "NVIDIA GeForce RTX 5080",
            "gpu": {"name": "NVIDIA GeForce RTX 5080", "memory_mib": 16303,
                     "pcie": {"gen_current": 5, "gen_max": 5, "width_current": 8, "width_max": 16},
                     "bandwidth_gbps_one_way": 256.0},
            "host_memory_bytes": 134_939_398_144, "source": "legacy-reference-default",
        }
    if not isinstance(hardware_snapshot, Mapping):
        raise TypeError("hardware_snapshot must be a mapping returned by probe_hardware")
    cpu_raw = hardware_snapshot.get("cpu", hardware_snapshot.get("cpu_name"))
    cpu = dict(cpu_raw) if isinstance(cpu_raw, Mapping) else {"name": cpu_raw}
    topology = hardware_snapshot.get("cpu_topology")
    if isinstance(topology, Mapping):
        cpu.update({key: value for key, value in topology.items() if key not in cpu or not cpu[key]})
    cpu.setdefault("physical_cores", hardware_snapshot.get("cpu_cores"))
    cpu.setdefault("logical_processors", hardware_snapshot.get("cpu_threads"))
    cpu_name = str(cpu.get("name", cpu.get("model")) or "").strip()
    gpu_raw = hardware_snapshot.get("gpu")
    if not isinstance(gpu_raw, Mapping):
        raise ValueError("hardware_snapshot.gpu must be a mapping with name, VRAM, and PCIe data")
    gpu_name = str(gpu_raw.get("name", gpu_raw.get("model")) or "").strip()
    if "9950x3d" not in cpu_name.casefold().replace(" ", ""):
        raise ValueError(f"unsupported CPU in hardware snapshot: {cpu_name or '<unknown>'}")
    gpu_match = re.search(r"\brtx\s*5080\b", gpu_name, re.IGNORECASE)
    gpu_suffix = gpu_name[gpu_match.end():].strip(" -()[]") if gpu_match else ""
    if gpu_match is None or gpu_suffix:
        raise ValueError(f"unsupported GPU in hardware snapshot: {gpu_name or '<unknown>'}")
    pcie = gpu_raw.get("pcie", hardware_snapshot.get("pcie"))
    if not isinstance(pcie, Mapping):
        raise ValueError("hardware_snapshot.gpu.pcie is required for a measured scenario")
    try:
        memory_mib = int(gpu_raw.get("memory_mib", gpu_raw.get("memory_total_mib", gpu_raw.get("vram_mib"))))
        host_raw = hardware_snapshot.get("host_memory") or {}
        host_memory_bytes = int(host_raw.get("total_bytes", host_raw.get("total_memory_bytes")))
        generation = int(pcie.get("gen_current"))
        width = int(pcie.get("width_current"))
    except (TypeError, ValueError, AttributeError):
        raise ValueError("hardware snapshot must declare positive GPU VRAM, host memory, and PCIe values") from None
    if memory_mib <= 0 or host_memory_bytes <= 0 or generation <= 0 or width <= 0:
        raise ValueError("hardware snapshot must declare positive GPU VRAM, host memory, and PCIe values")
    return {
        "cpu_name": cpu_name,
        "physical_cores": int(cpu.get("physical_cores", cpu.get("core_count")) or 0) or None,
        "logical_processors": int(cpu.get("logical_processors", cpu.get("thread_count")) or 0) or None,
        "gpu_name": gpu_name, "gpu": gpu_raw, "host_memory_bytes": host_memory_bytes,
        "pcie": pcie, "pcie_bandwidth_gbps": _pcie_one_way_bandwidth({**gpu_raw, **pcie}, generation, width),
        "source": "measured-hardware-snapshot",
    }


def _hardware_fingerprint(snapshot: Mapping[str, object]) -> str:
    """Hash stable CPU/GPU identity, excluding volatile clocks/free memory.

    ``probe_hardware`` also records current CPU/GPU clocks and memory usage for
    diagnostics.  Those values can change between profile capture and replay,
    so they must never participate in the identity gate used for calibration.
    """
    gpu = snapshot.get("gpu", {}) if isinstance(snapshot, Mapping) else {}
    cpu = snapshot.get("cpu_topology", {}) if isinstance(snapshot, Mapping) else {}
    pcie = gpu.get("pcie", {}) if isinstance(gpu, Mapping) else {}
    def _field(mapping: Mapping[str, object], *names: str) -> object:
        for name in names:
            if name in mapping:
                return mapping[name]
        return None
    return stable_hash({
        "cpu_name": str(snapshot.get("cpu", "")).strip() if isinstance(snapshot, Mapping) else None,
        "gpu_name": gpu.get("name") if isinstance(gpu, Mapping) else None,
        "gpu_uuid": gpu.get("uuid") if isinstance(gpu, Mapping) else None,
        "compute_capability": gpu.get("compute_capability") if isinstance(gpu, Mapping) else None,
        "driver": gpu.get("driver") if isinstance(gpu, Mapping) else None,
        "cpu_topology": {
            "physical_cores": _field(cpu, "physical_cores", "core_count"),
            "logical_processors": _field(cpu, "logical_processors", "thread_count"),
            "max_clock_mhz": _field(cpu, "max_clock_mhz", "max_clock"),
        },
        # Current link values are sampled run state and can drop while the GPU
        # is idle.  Only negotiated maxima belong to the stable identity gate;
        # the complete current snapshot remains available for service-model
        # diagnostics and is recorded separately by the caller.
        "pcie": {k: pcie.get(k) for k in ("gen_max", "width_max")},
    })


def _trace_artifact_ref(path: Path) -> dict[str, object]:
    """Return a reproducible reference for an optional raw profiler artifact.

    The artifact is deliberately referenced by path and digest rather than
    embedded in the comparison payload.  This keeps large NSYS/CUPTI files
    out of JSON while allowing replay to fail closed when a file is missing or
    has changed.
    """
    resolved = Path(path).resolve()
    if not resolved.exists() or not resolved.is_file():
        return {"path": str(resolved), "status": "missing", "sha256": None, "bytes": None}
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(resolved), "status": "captured", "sha256": digest.hexdigest(), "bytes": size}


def _runtime_artifact_refs(exe: Path) -> list[dict[str, object]]:
    """Capture the executable and colocated runtime DLL identities.

    A profile calibrated against one ggml-cuda/llama runtime must not be
    silently reused after a rebuild.  The list is intentionally limited to
    files colocated with the selected executable so it remains reproducible
    and does not guess at unrelated system DLLs.
    """
    resolved = Path(exe).resolve()
    paths = [resolved, *sorted(resolved.parent.glob("*.dll"))]
    return [_trace_artifact_ref(path) for path in paths]


_EXTRACTOR_FUNCTIONS = (
    "post_stream_json", "post_parallel_stream_json", "_percentile_summary",
    "_engine_boundary_timing", "_engine_counter_timing", "_native_request_record",
    "_aggregate_request_records", "parse_perf_log",
    "metric_snapshot",
)


def native_extractor_identity() -> dict[str, object]:
    """Hash only native timing/parsing entry points, excluding simulator code."""
    trees: dict[str, str] = {}
    for name in _EXTRACTOR_FUNCTIONS:
        fn = globals().get(name)
        if fn is None:
            continue
        source = inspect.getsource(fn)
        trees[name] = ast.dump(ast.parse(source), include_attributes=False)
    basis = {"schema": "selected_function_ast_v1", "functions": trees}
    return {
        "path": str(Path(__file__).resolve()),
        "status": "captured",
        "sha256_basis": "selected_function_ast_v1",
        "functions": list(trees),
        "implementation_sha256": stable_hash(basis),
        # ``sha256`` is the semantic extractor identity.  The enclosing file
        # also contains simulator code, so its byte hash is provenance only
        # and must not invalidate simulator-only replay.
        "sha256": stable_hash(basis),
        "file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def tokenize_prompt(model: str | Path, prompt: str, *, tokenizer_exe: str | Path | None = None) -> dict[str, object]:
    """Tokenize before timing starts, using the locked llama.cpp tokenizer."""
    exe = Path(tokenizer_exe) if tokenizer_exe else Path(model).resolve().parent.parent / "source" / "llama-tokenize.exe"
    if not exe.exists():
        # The normal layout places llama-tokenize beside llama-server.
        exe = Path(__file__).resolve().parents[1] / "source" / "llama.cpp-semantic" / "build-semantic-direct" / "bin" / "llama-tokenize.exe"
    if not exe.exists():
        raise FileNotFoundError(f"llama-tokenize executable not found: {exe}")
    cp = subprocess.run([str(exe), "-m", str(model), "-p", prompt, "--ids", "--show-count"],
                        capture_output=True, text=True, timeout=180, check=False)
    text = (cp.stdout or "") + "\n" + (cp.stderr or "")
    match = re.search(r"Total number of tokens:\s*(\d+)", text)
    if cp.returncode != 0 or match is None:
        raise RuntimeError(f"tokenization failed: returncode={cp.returncode}; output={text[-500:]}")
    ids_match = re.search(r"(?m)^\s*\[[-\d, ]*\]\s*$", cp.stdout or "")
    ids = json.loads(ids_match.group(0)) if ids_match else None
    count = int(match.group(1))
    if ids is not None and len(ids) != count:
        raise RuntimeError(f"tokenizer count mismatch: ids={len(ids)} reported={count}")
    return {"count": count, "ids": ids, "executable": str(exe.resolve()), "stdout": cp.stdout}


def build_matching_scenario(prompt_tokens: int, output_tokens: int, *, ctx: int,
                            parallel: int, batch: int, ubatch: int, threads: int,
                            gpu_layers: int, seed: int = 42,
                            coherent_dma_mode: str = "pipelined",
                            model=None,
                            hardware_snapshot: Mapping[str, object] | None = None):
    """Build a parity scenario using measured physical host inputs.

    The optional snapshot keeps legacy callers working while allowing the
    matrix runner to bind VRAM, host memory, and negotiated PCIe facts.  Cost
    efficiencies stay analytical and are never inferred from timing errors.
    """
    measured = _hardware_inputs(hardware_snapshot)
    if threads == -1:
        threads = measured.get("logical_processors")
        if not isinstance(threads, int) or threads <= 0:
            raise ValueError("threads=-1 requires logical_processors in hardware_snapshot")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer or -1")
    base = build_reference_scenario()
    model = model or model_from_dict(materialize_model_payload("qwen2_5-0_5b"))
    view = model_graph_execution_view(model.graph, schema_version=model.schema_version)
    p = base.placement
    rank_mapping = tuple(
        replace(item, cim_component_id=None) for item in p.parallel.rank_mapping
    )
    parallel_spec = replace(
        p.parallel,
        rank_mapping=rank_mapping,
        layer_to_stage={item.layer_id: 0 for item in view.layer_instances},
    )
    # Keep the typed profile registry but collapse the illustrative 8-HBM/CIM
    # topology to one GPU memory device, matching the RTX 5080 experiment.
    gpu_raw = measured["gpu"]
    pcie = measured.get("pcie", gpu_raw.get("pcie", {}))
    pcie_generation = int(pcie.get("gen_current", 5))
    pcie_width = int(pcie.get("width_current", 8))
    pcie_bandwidth = float(measured.get("pcie_bandwidth_gbps", gpu_raw.get("bandwidth_gbps_one_way", 256.0)))
    vram_bytes = int(gpu_raw.get("memory_mib", 16303)) * 1024**2
    host_memory_bytes = int(measured["host_memory_bytes"])
    components = []
    for component in base.hardware.components:
        if component.component_id == "gpu0":
            ports = tuple(
                replace(port, bandwidth_gbps=7680.0) if port.port_id == "hbm0" else
                replace(port, lanes=pcie_width, bandwidth_gbps=pcie_bandwidth) if port.port_id == "pcie0" else port
                for port in component.ports
            )
            components.append(replace(component, ports=ports, capacity_bytes=50 * 1024**2, peak_ops_per_s=112_600_000_000_000.0, metadata={
                **component.metadata, "measured_gpu": measured["gpu_name"], "gpu_uuid": gpu_raw.get("uuid"),
                "driver": gpu_raw.get("driver"), "clocks": gpu_raw.get("clocks", {}),
                "pcie_generation_current": pcie_generation, "pcie_width_current": pcie_width,
            }))
        elif component.component_id == "hostmem0":
            components.append(replace(component, ports=tuple(
                replace(port, bandwidth_gbps=716.8) if port.port_id == "ddr0" else port
                for port in component.ports
            ), capacity_bytes=host_memory_bytes))
        elif component.component_id == "hbm0":
            components.append(replace(component, ports=tuple(replace(port, bandwidth_gbps=7680.0) for port in component.ports), capacity_bytes=vram_bytes))
        elif component.component_id == "cpu0":
            components.append(replace(component, ports=tuple(
                replace(port, lanes=pcie_width, bandwidth_gbps=pcie_bandwidth) if port.port_id == "pcie0" else
                replace(port, bandwidth_gbps=716.8) if port.port_id == "ddr0" else port
                for port in component.ports
            )))
    hardware = replace(
        base.hardware,
        name="RTX5080-local",
        components=tuple(components),
        links=tuple(
            replace(l, bandwidth_gbps=7680.0) if l.link_id == "gpu-hbm0" else
            replace(l, lanes=pcie_width, bandwidth_gbps=pcie_bandwidth, metadata={**l.metadata, "gen_current": pcie_generation, "width_current": pcie_width}) if l.link_id == "cpu-gpu-pcie" else
            replace(l, bandwidth_gbps=716.8) if l.link_id == "cpu-hostmem-ddr" else l
            for l in base.hardware.links if l.link_id in {"cpu-gpu-pcie", "cpu-hostmem-ddr", "gpu-hbm0"}
        ),
        metadata={**base.hardware.metadata, "measured_host": measured["cpu_name"], "measured_gpu": measured["gpu_name"], "gpu_vram_mib": vram_bytes // 1024**2, "gpu_uuid": gpu_raw.get("uuid"), "gpu_driver": gpu_raw.get("driver"), "gpu_clocks": gpu_raw.get("clocks", {}), "pcie_link_gen_current": pcie_generation, "pcie_link_width_current": pcie_width, "pcie_link_bandwidth_gbps_one_way": pcie_bandwidth, "host_memory_total_bytes": host_memory_bytes, "hardware_snapshot_source": measured["source"], "llama_cpp_gpu_layers": gpu_layers},
    )
    # Qwen3.8's hybrid CPU graph lowers an 8-token prefill into two physical
    # 4-token graph invocations even when llama-server is launched with
    # ``-ub 64``.  The CPU operator trace records M=4 for both prefill
    # invocations.  Bind the simulator's chunking to that measured graph
    # boundary so the stage/shape calibration and the native request have
    # identical invocation geometry.  Other models keep the CLI ubatch.
    prefill_chunk_tokens = ubatch
    model_name = str(getattr(model, "name", "")).casefold()
    # The GGUF importer exposes this family as ``GGUF-qwen35``; the 27B
    # artifact is the 65-block variant.  Keep the layer-count guard so a
    # smaller Qwen3.5 fixture does not inherit an unverified chunk size.
    # Only Qwen3.8 currently has a validated shape-specific profile for the
    # measured invocation geometry.  Qwen2.5/TinyLlama retain their proven
    # phase policy until each new input shape has a stable holdout sample.
    exact_shape_calibration = (
        "qwen3.8" in model_name or "qwen3_8" in model_name
        or ("gguf-qwen35" in model_name and int(getattr(model, "num_layers", 0)) >= 64)
    )
    # Qwen2.5 has an opt-in kernel-basis profile with reliable M-specific
    # buckets.  Keep phase fallback for shapes absent from that profile; this
    # does not alter Qwen3.5/Qwen3.8's existing policies.
    kernel_shape_if_available = (
        not exact_shape_calibration
        and str(getattr(model, "architecture", "")).casefold() == "qwen2"
    )
    # M=4 was observed only for the calibrated 8-token, single-request CPU
    # capture.  It is not a model-wide scheduler rule: applying it to a long
    # prompt creates dozens of artificial graph invocations and recharges
    # their fixed frontend costs.  Outside that exact evidence tuple, use
    # the runtime's declared ubatch limit.
    if (exact_shape_calibration and gpu_layers == 0 and prompt_tokens == 8
            and parallel == 1 and batch == 64 and ubatch == 64 and threads == 16):
        prefill_chunk_tokens = min(ubatch, 4)
    parity_requests = tuple(
        replace(base.workload.requests[0], request_id=f"request-{index:04d}",
                arrival_ns=0.0, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens)
        for index in range(parallel)
    )
    workload = replace(
        base.workload,
        name="native-llama-parity",
        random_seed=seed,
        requests=parity_requests,
        request_count=parallel,
        mtp=None,
        scheduler=replace(base.workload.scheduler, mode="continuous", max_num_seqs=parallel, max_num_batched_tokens=batch, max_num_ubatch_tokens=ubatch, prefill_chunk_tokens=prefill_chunk_tokens, preemption_enabled=False),
    )
    # llama.cpp keeps KV pages on the active execution memory.  With no
    # offloaded transformer layers (``-ngl 0``), that is host DRAM; when at
    # least one layer is on CUDA, K/V tensors are allocated in the GPU HBM
    # arena (``--kvo``).  The reference scenario defaults to hbm0, so leaving
    # it unchanged would charge CPU-only runs to an unavailable GPU cache and
    # hide the KV read/append traffic from the CPU resource path.
    kv_component = "hostmem0" if gpu_layers == 0 else "hbm0"
    # The reference scenario carries a generated control-plane decision for a
    # different model/topology.  Reusing it makes the strict fingerprint gate
    # reject freshly built parity scenarios after any planner change.  Keep
    # policy options but discard generated output; runtime lowering will build
    # a decision for the current model and hardware.
    placement_metadata = dict(p.metadata)
    # The Qwen3.8 CPU trace proves that the two physical FFN matrices are
    # submitted as sequential ``ffn_gate`` and ``ffn_up`` MUL_MAT calls.  Bind
    # that capability only for this measured model/runtime combination; other
    # models continue to use the conservative fused workload unless they
    # declare their own physical projection contract.
    if (
        gpu_layers == 0
        and (
            "qwen3.8" in model_name
            or "qwen3_8" in model_name
            or ("gguf-qwen35" in model_name and int(getattr(model, "num_layers", 0)) >= 64)
        )
    ):
        placement_metadata["llama_cpp_physical_projection_invocations"] = True
    control_plane = placement_metadata.get("control_plane")
    if isinstance(control_plane, Mapping):
        control_plane = dict(control_plane)
        control_plane["decision"] = {}
        control_plane["evidence"] = {}
        placement_metadata["control_plane"] = control_plane
    placement = replace(p, model_name=model.name, hardware_name=hardware.name,
                        parallel=parallel_spec,
                        metadata={**placement_metadata, "coherent_dma_mode": coherent_dma_mode,
                                  # Shape-specific CPU calibration is currently
                                  # validated for the 65-block Qwen3.8 trace.
                                  # Other model profiles retain their proven
                                  # phase aggregate until matching shape
                                  # evidence is collected.
                                  "native_calibration_shape_policy": (
                                      "exact" if exact_shape_calibration
                                      else "kernel_shape_if_available" if kernel_shape_if_available
                                      else "phase"
                                  ),
                                  "llama_cpp_kv_component": kv_component},
                        kv_policy=replace(p.kv_policy, cache_component=kv_component,
                                          dtype="fp16", offload_component=None, offload_ratio=0.0))
    profiles = {kind: dict(values) for kind, values in base.component_profiles.items()}
    gpu_profile = profiles["gpu"]["legacy-gpu"]
    tensor = gpu_profile.tensor_core
    # Dense BF16 peak = 112.6 TFLOP/s at the declared 84 SM / 2.617 GHz.
    cycles = 84 * 4 * 2.617e9 * (2 * tensor.mma_m * tensor.mma_n * tensor.mma_k) * 0.5 / 112.6e12
    profiles["gpu"]["legacy-gpu"] = replace(gpu_profile, name="RTX5080-analytical", tensor_core=replace(tensor, sm_count=84, frequency_ghz=2.617, cycles_per_mma=cycles))
    profiles["hbm"]["legacy-hbm"] = replace(profiles["hbm"]["legacy-hbm"], bandwidth_gb_s=960.0)
    profiles["host_memory"]["legacy-host-memory"] = replace(profiles["host_memory"]["legacy-host-memory"], name="DDR5-5600-dual-channel", bandwidth_gb_s=89.6)
    cpu_profile = profiles["cpu"]["legacy-cpu"]
    profiles["cpu"]["legacy-cpu"] = replace(cpu_profile, name="Ryzen9950X3D-analytical", pipeline=replace(cpu_profile.pipeline, core_count=threads, frequency_ghz=4.3))
    runtime_config = LlamaCppRuntimeConfig(
        threads=threads, threads_batch=threads, batch=batch, ubatch=ubatch,
        context=ctx, parallel=parallel, gpu_layers=gpu_layers,
        flash_attn=False, kv_type_k="f16", kv_type_v="f16",
        kv_unified=True, cont_batching=True, warmup=True, seed=seed,
        mmap=True, mlock=False, offload_kqv=True, op_offload=True,
        split_mode="layer", main_gpu=0,
    )
    placement_risks = []
    logical_processors = measured.get("logical_processors")
    if isinstance(logical_processors, int) and threads > logical_processors:
        placement_risks.append("requested_threads_exceed_logical_processors")
    physical_cores = measured.get("physical_cores")
    if isinstance(physical_cores, int) and threads > physical_cores:
        placement_risks.append("requested_threads_use_hyperthreads")
    if gpu_layers == 0:
        placement_risks.append("gpu_offload_disabled_cpu_only")
    elif gpu_layers > 0 and gpu_layers < model.num_layers:
        placement_risks.append("partial_gpu_layer_offload")
    placement = replace(placement, metadata={
        **placement.metadata, "hardware_snapshot_source": measured["source"],
        "placement_risks": tuple(placement_risks), "cpu_identity": measured["cpu_name"],
        "gpu_identity": measured["gpu_name"],
        "hardware_fingerprint": _hardware_fingerprint(hardware_snapshot) if hardware_snapshot is not None else None,
    })
    authored = replace(base, name="native-llama-parity-rtx5080", model=model, hardware=hardware, placement=placement, workload=workload, component_profiles=profiles, fusion_policy=replace(base.fusion_policy, flash_attention=False), llama_cpp_config=runtime_config, assumptions=base.assumptions + (f"llama.cpp ctx={ctx} batch={batch} ubatch={ubatch} threads={threads} parallel={parallel} gpu_layers={gpu_layers} ctk=ctv:f16",))
    lowered = apply_llama_runtime_config(authored, runtime_config, materialize_placement=True)
    # Keep the llama.cpp per-layer KV arena decision explicit.  ``-ngl`` uses
    # a tail placement: CPU-prefix layers persist K/V in host DRAM and the
    # CUDA suffix persists K/V in HBM.  The planner still has a deterministic
    # fallback for authored scenarios without this map.
    execution_view = model_graph_execution_view(
        lowered.model.graph,
        schema_version=lowered.model.schema_version,
    )
    components = {
        component.component_id: component
        for component in lowered.hardware.components
    }
    host_memory = next(
        (component.component_id for component in lowered.hardware.components
         if component.normalized_kind in {"host_memory", "dram", "ddr", "ddr_memory"}),
        None,
    )
    gpu_memory = next(
        (component.component_id for component in lowered.hardware.components
         if component.normalized_kind in {"hbm", "gpu_memory"}),
        None,
    )
    layer_map = {}
    for instance in execution_view.layer_instances:
        layer = instance.layer
        target = lowered.placement.op_to_component.get(
            layer.layer_id + ".attention",
            lowered.placement.op_to_component.get(
                layer.layer_id + ".linear_attention"
            ),
        )
        target_component = components.get(str(target)) if target else None
        if target_component is None:
            continue
        if target_component.normalized_kind in {"cpu", "host", "processor"}:
            owner = host_memory
        elif target_component.normalized_kind in {"gpu", "cuda", "accelerator"}:
            owner = gpu_memory
        else:
            owner = None
        if owner:
            layer_map[str(layer.layer_id)] = owner
    metadata = dict(lowered.placement.metadata)
    metadata["llama_cpp_kv_layer_components"] = layer_map
    lowered = replace(
        lowered,
        placement=replace(lowered.placement, metadata=metadata),
    )
    return lowered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.environ.get("LLAMA_SERVER_EXE", DEFAULT_EXE))
    ap.add_argument("--model", default=os.environ.get("LLAMA_GGUF", DEFAULT_MODEL))
    ap.add_argument("--prompt", default="Explain why deterministic benchmarking matters.")
    ap.add_argument("--predict", type=int, default=8)
    ap.add_argument("--output-mode", choices=("natural", "fixed"), default="natural",
                    help="输出策略；fixed 使用 ignore_eos 并按请求长度预先生成仿真预测，natural 保留提前 EOS")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--port", type=int, default=0, help="server port; 0 chooses a free port")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ubatch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--stop", action="append", default=[])
    ap.add_argument("--warmup-predict", type=int, default=2)
    ap.add_argument("--request-timing", choices=("stream", "prompt_eval", "none"), default="stream",
                    help="请求边界计时来源；stream 测量 request-to-first-token，prompt_eval 仅保留旧阶段诊断")
    ap.add_argument("--coherent-dma-mode", choices=("pipelined", "strict_serialized"), default="pipelined")
    ap.add_argument("--raw-trace-artifact", action="append", type=Path, default=[],
                    help="可选原始 NVTX/CUPTI/NSYS 文件；只记录路径与 SHA，不把大文件嵌入 payload")
    ap.add_argument("--extractor-output-artifact", action="append", type=Path, default=[],
                    help="可选 trace extractor 输出文件；只记录路径与 SHA")
    ap.add_argument("--extractor-script-artifact", action="append", type=Path, default=[],
                    help="可选 trace extractor 脚本或版本文件；只记录路径与 SHA")
    ap.add_argument("--output", type=Path, default=Path("artifacts/native_compare.json"))
    ap.add_argument("--calibration-profile", type=Path, default=None,
                    help="可选 native-calibration/v1；仅将已测 CUDA launch 系数下沉到 gpu.frontend")
    ap.add_argument("--apply-launch-calibration", action="store_true",
                        help="显式启用 launch 系数；默认只保存校准证据，避免 CUDA graph API 时间重复计入每个算子")
    ap.add_argument("--apply-stage-calibration", action="store_true",
                    help="显式启用已有且身份匹配的 operator stage/shape 校准；未知或缺少 shape 的阶段保持分析模型")
    ap.add_argument("--apply-memory-calibration", action="store_true",
                    help="显式按 stage/phase 实测 effective bandwidth 重写 CPU memory service；不修改 cache/transfer bytes")
    ap.add_argument("--apply-phase-boundary-calibration", action="store_true",
                    help="显式启用每个物理 prefill/decode invocation 一次的 launch+sync 边界证据")
    ap.add_argument("--apply-request-boundary-calibration", action="store_true",
                    help="显式启用同 prompt/output 场景下 request_begin/first_token/request_end marker 的一次性边界证据")
    args = ap.parse_args()
    if not Path(args.exe).exists() or not Path(args.model).exists():
        raise SystemExit("llama.cpp executable or GGUF model not found")
    gguf = read_gguf_metadata(args.model)
    gguf_model = build_model_from_gguf(gguf)
    parity = compare_gguf_to_model(gguf, gguf_model, context_length=args.ctx)
    assert_gguf_parity(parity)
    # The prediction shape is frozen before starting llama-server.  Native
    # timing and natural-EOS token counts must never be fed back into it.
    prompt_tokenization = tokenize_prompt(args.model, args.prompt, tokenizer_exe=Path(args.exe).with_name("llama-tokenize.exe"))
    prompt_tokens = int(prompt_tokenization["count"])
    predicted_output_tokens = int(args.predict)
    hardware = probe_hardware()
    sim_scenario = build_matching_scenario(prompt_tokens, predicted_output_tokens, ctx=args.ctx, parallel=args.parallel,
                                           batch=args.batch, ubatch=args.ubatch, threads=args.threads, gpu_layers=args.gpu_layers,
                                           seed=args.seed, coherent_dma_mode=args.coherent_dma_mode, model=gguf_model,
                                           hardware_snapshot=hardware)
    sim_scenario = replace(sim_scenario, placement=replace(
        sim_scenario.placement,
        metadata={**sim_scenario.placement.metadata, "prompt_fingerprint": stable_hash(args.prompt)},
    ))
    calibration = None
    if args.calibration_profile is not None:
        calibration = load_native_calibration(args.calibration_profile)
        sim_scenario = apply_native_calibration(
            sim_scenario, calibration,
            apply_launch=args.apply_launch_calibration,
            apply_stage=args.apply_stage_calibration,
            apply_memory=args.apply_memory_calibration,
            apply_phase_boundary=args.apply_phase_boundary_calibration,
            apply_request_boundary=args.apply_request_boundary_calibration,
        )
    prediction_sim = run_scenario(sim_scenario, retention_policy="aggregate")
    prediction_artifact = args.output.with_suffix(".prediction.json")
    prediction_artifact.parent.mkdir(parents=True, exist_ok=True)
    prediction_artifact.write_text(json.dumps({
        "schema": "native-simulator-prediction/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "saved_before_native_reveal", "output_mode": args.output_mode,
        "prompt_tokenization": prompt_tokenization, "requested_output_tokens": predicted_output_tokens,
        "model": str(Path(args.model).resolve()), "gguf_sha256": parity["gguf"]["sha256"],
        "hardware_fingerprint": _hardware_fingerprint(hardware), "configuration": {
            "ctx": args.ctx, "parallel": args.parallel, "batch": args.batch, "ubatch": args.ubatch,
            "threads": args.threads, "gpu_layers": args.gpu_layers, "seed": args.seed,
            "temperature": args.temperature, "top_k": args.top_k, "warmup_predict": args.warmup_predict,
            "request_timing": args.request_timing,
        },
        "calibration_profile": str(args.calibration_profile) if args.calibration_profile else None,
        "request_metrics": {rid: {"ttft_ns": getattr(m, "ttft_ns", None),
                                   "client_e2e_ns": getattr(m, "client_e2e_ns", None),
                                   "e2e_ns": getattr(m, "e2e_ns", None),
                                   "engine_timing": _simulator_request_timing(prediction_sim, m),
                                   "visible_output_tokens": getattr(m, "visible_output_tokens", None)}
                            for rid, m in prediction_sim.metrics.request_metrics.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_sha256 = hashlib.sha256(prediction_artifact.read_bytes()).hexdigest()
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    if args.port == 0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        sock.close()
    else:
        port = args.port
    cmd = [args.exe, "-m", args.model, "--host", "127.0.0.1", "--port", str(port), "-c", str(args.ctx), "-ngl", str(args.gpu_layers), "-np", str(args.parallel), "-b", str(args.batch), "-ub", str(args.ubatch), "-t", str(args.threads), "-tb", str(args.threads), "-fa", "off", "--load-mode", "mmap", "-kvo", "--op-offload", "-sm", "layer", "-mg", "0", "-ctk", "f16", "-ctv", "f16", "-kvu", "-cb", "--perf", "--metrics", "--warmup", "--spec-type", "none"]
    log_path = args.output.with_suffix(".llama.log")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    try:
        base = f"http://127.0.0.1:{port}"
        wait_health(base, proc)
        try:
            slots_before = get_json(base + "/slots")
        except Exception:
            slots_before = None
        prompt = args.prompt
        warmup_payload = {"prompt": prompt, "n_predict": max(1, args.warmup_predict), "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed, "cache_prompt": False, "stream": False}
        warmup_started = time.perf_counter()
        warmup = post_json(base + "/completion", warmup_payload)
        warmup_wall_ms = (time.perf_counter() - warmup_started) * 1000.0
        # The formal baseline must be captured after warmup.  Otherwise the
        # delta mixes warmup work into the measured request counters.
        metrics_before = get_text(base + "/metrics")
        try:
            slots_before = get_json(base + "/slots")
        except Exception:
            slots_before = None
        started = time.perf_counter()
        formal_payload = {"prompt": prompt, "n_predict": args.predict, "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed, "cache_prompt": False, "stream": args.request_timing == "stream", "ignore_eos": args.output_mode == "fixed"}
        if args.stop:
            formal_payload["stop"] = list(args.stop)
        request_boundary: dict[str, object] = {
            "mode": "disabled", "request_start": None,
            "first_event_ms": None, "first_content_ms": None,
            "request_to_first_token_ms": None, "request_to_end_ms": None,
            "first_token_source": None, "status": "unavailable", "chunk_count": 0,
        }
        native_requests: list[dict] = []
        request_boundaries: list[dict[str, object]] = []
        if args.request_timing == "stream":
            pairs = post_parallel_stream_json(base + "/completion", formal_payload, args.parallel)
            native_requests = [item[0] for item in pairs]
            request_boundaries = [item[1] for item in pairs]
            native = native_requests[0]
            request_boundary = request_boundaries[0]
        elif args.request_timing == "prompt_eval":
            native_requests = post_parallel_json(base + "/completion", formal_payload, args.parallel)
            native = native_requests[0]
            request_boundaries = [dict(request_boundary, mode="prompt_eval_legacy") for _ in native_requests]
            request_boundary["mode"] = "prompt_eval_legacy"
        else:
            native_requests = post_parallel_json(base + "/completion", formal_payload, args.parallel)
            native = native_requests[0]
            request_boundaries = [dict(request_boundary) for _ in native_requests]
        wall_ms = (time.perf_counter() - started) * 1000.0
        metrics_after = get_text(base + "/metrics")
        try:
            slots_after = get_json(base + "/slots")
        except Exception:
            slots_after = None
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()
    timings = native.get("timings", {})
    # Use tokenizer evidence and the pre-reveal prediction shape.  Missing
    # native counters are an evidence failure; never backfill them with the
    # requested output length.
    native_prompt_value = timings.get("prompt_n", native.get("tokens_evaluated"))
    native_output_value = timings.get("predicted_n", native.get("tokens_predicted"))
    if native_prompt_value is None or native_output_value is None:
        raise ValueError("native token counters are missing; token parity is unavailable")
    output_tokens = int(native_output_value)
    sim = prediction_sim
    req = next(iter(sim.metrics.request_metrics.values()))
    req_e2e_ns = getattr(req, "e2e_ns", None)
    if req_e2e_ns is None:
        req_e2e_ns = (
            getattr(req, "finish_ns", None) - getattr(req, "arrival_ns", 0.0)
            if getattr(req, "finish_ns", None) is not None else None
        )
    req_client_e2e_ns = getattr(req, "client_e2e_ns", None)
    if req_client_e2e_ns is None:
        req_client_e2e_ns = req_e2e_ns
    simulator_request_records: list[dict[str, object]] = []
    for request_index, request_id in enumerate(sorted(sim.metrics.request_metrics)):
        metric = sim.metrics.request_metrics[request_id]
        metric_e2e_ns = getattr(metric, "e2e_ns", None)
        if metric_e2e_ns is None and getattr(metric, "finish_ns", None) is not None:
            metric_e2e_ns = float(metric.finish_ns) - float(getattr(metric, "arrival_ns", 0.0))
        metric_client_e2e_ns = getattr(metric, "client_e2e_ns", None) or metric_e2e_ns
        timing = _simulator_request_timing(sim, metric)
        visible_tokens = int(getattr(metric, "visible_output_tokens", 0) or 0)
        client_ttft_ms = timing["client_ttft_ms"]
        client_tpot_ms = timing["client_tpot_ms"]
        simulator_request_records.append({
            "request_id": str(request_id),
            "request_index": request_index,
            "prompt_tokens": prompt_tokens,
            "output_tokens": visible_tokens,
            # Legacy aliases retain their historical client boundary.
            "ttft_ms": client_ttft_ms,
            "tpot_ms": client_tpot_ms,
            "client_tpot_ms": client_tpot_ms,
            "engine_tpot_ms": timing["engine_tpot_ms"],
            "engine_ttft_ms": timing["engine_ttft_ms"],
            "e2e_ms": timing["client_e2e_ms"],
            "client_ttft_ms": client_ttft_ms,
            "client_e2e_ms": timing["client_e2e_ms"],
            "engine_e2e_ms": timing["engine_e2e_ms"],
            "request_to_first_token_ms": client_ttft_ms,
            "request_to_end_ms": timing["client_e2e_ms"],
            "engine_request_to_end_ms": timing["engine_e2e_ms"],
            "engine_ttft_source": timing["engine_ttft_source"],
            "engine_request_begin_ns": timing["engine_request_begin_ns"],
            "prefill_end_ns": timing["prefill_end_ns"],
        })
    simulator_aggregate = {
        "ttft_ms": _percentile_summary([
            float(item["client_ttft_ms"]) for item in simulator_request_records
            if item.get("client_ttft_ms") is not None
        ]),
        "engine_ttft_ms": _percentile_summary([
            float(item["engine_ttft_ms"]) for item in simulator_request_records
            if item.get("engine_ttft_ms") is not None
        ]),
        "tpot_ms": _percentile_summary([
            float(item["client_tpot_ms"]) for item in simulator_request_records
            if item.get("client_tpot_ms") is not None
        ]),
        "engine_tpot_ms": _percentile_summary([
            float(item["engine_tpot_ms"]) for item in simulator_request_records
            if item.get("engine_tpot_ms") is not None
        ]),
        "e2e_ms": _percentile_summary([
            float(item["client_e2e_ms"]) for item in simulator_request_records
            if item.get("client_e2e_ms") is not None
        ]),
        "client_e2e_ms": _percentile_summary([
            float(item["client_e2e_ms"]) for item in simulator_request_records
            if item.get("client_e2e_ms") is not None
        ]),
        "engine_e2e_ms": _percentile_summary([
            float(item["engine_e2e_ms"]) for item in simulator_request_records
            if item.get("engine_e2e_ms") is not None
        ]),
        "client_ttft_ms": _percentile_summary([
            float(item["client_ttft_ms"]) for item in simulator_request_records
            if item.get("client_ttft_ms") is not None
        ]),
        "request_count": len(simulator_request_records),
        "makespan_ms": float(getattr(sim.serving, "makespan_ns", 0.0)) / 1e6,
        "client_makespan_ms": max((float(item["client_e2e_ms"]) for item in simulator_request_records
                                    if item.get("client_e2e_ms") is not None), default=None),
        "batch_client_wall_ms": None,
    }
    token_parity = {
        "native_prompt": int(native_prompt_value),
        "simulator_prompt": prompt_tokens,
        "native_output": output_tokens,
        "simulator_output": req.visible_output_tokens,
        "prompt_tokenizer": prompt_tokens,
        "ok": int(native_prompt_value) == prompt_tokens and output_tokens == req.visible_output_tokens,
    }
    # Preserve the complete native capture even when parity fails.  Eligibility
    # and scoring consume this flag later; raising here used to discard the
    # structured response and its failure context.
    token_parity_error = None if token_parity["ok"] else "native/simulator token parity gate failed: output token count differs"
    native_prompt_ms = float(timings.get("prompt_ms", 0.0)); native_eval_ms = float(timings.get("predicted_ms", 0.0))
    native_request_records = [
        _native_request_record(response, boundary, index, args.predict)
        for index, (response, boundary) in enumerate(zip(native_requests, request_boundaries))
    ]
    # Preserve both boundary families.  Engine counters are primary; stream
    # timestamps remain secondary client diagnostics.
    native_tpot_ms = native_request_records[0].get("client_tpot_ms") if native_request_records else None
    native_aggregate = _aggregate_request_records(
        native_request_records,
        batch_client_wall_ms=wall_ms,
    )
    runtime_config = LlamaCppRuntimeConfig(
        threads=args.threads, threads_batch=args.threads,
        batch=args.batch, ubatch=args.ubatch, context=args.ctx,
        parallel=args.parallel, gpu_layers=args.gpu_layers, flash_attn=False,
        kv_type_k="f16", kv_type_v="f16", kv_unified=True,
        cont_batching=True, warmup=True, seed=args.seed,
    )
    request_identity = {
        "model_path": str(Path(args.model).resolve()),
        "gguf_sha256": ((parity.get("gguf") or {}).get("sha256")
                         if isinstance(parity.get("gguf"), Mapping) else None),
        "runtime_fingerprint": runtime_config.fingerprint,
        "hardware_fingerprint": _hardware_fingerprint(hardware),
        "prompt_fingerprint": stable_hash(args.prompt),
        "configuration": {
            "ctx": args.ctx,
            "parallel": args.parallel,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "flash_attn": False,
            "seed": args.seed,
        },
    }
    for record in native_request_records:
        record["identity"] = dict(request_identity)
    for record in simulator_request_records:
        record["identity"] = dict(request_identity)
    before_metrics = metric_snapshot(metrics_before if 'metrics_before' in locals() else "")
    after_metrics = metric_snapshot(metrics_after if 'metrics_after' in locals() else "")
    metric_delta = {key: after_metrics[key] - before_metrics.get(key, 0.0) for key in after_metrics}
    boundary_measured = request_boundary.get("status") == "measured"
    native_request_ttft = request_boundary.get("request_to_first_token_ms") if boundary_measured else None
    native_request_e2e = request_boundary.get("request_to_end_ms") if boundary_measured else None
    native_total_ms = float(timings.get("predicted_ms", 0.0) + timings.get("prompt_ms", 0.0))
    native_engine_tpot_ms = native_eval_ms / (output_tokens - 1) if output_tokens > 1 else None
    native_payload = {"prompt_eval_ms": native_prompt_ms, "eval_ms": native_eval_ms,
                      # Engine-level metrics are the primary comparison
                      # contract; prompt/eval counters are llama.cpp's model
                      # execution boundary.  Client stream fields remain
                      # available for secondary service diagnostics.
                      "engine_ttft_ms": native_request_records[0].get("engine_ttft_ms") if native_request_records else None,
                      "engine_tpot_ms": native_request_records[0].get("engine_tpot_ms") if native_request_records else None,
                      "engine_e2e_ms": native_request_records[0].get("engine_e2e_ms") if native_request_records else None,
                      "engine_timing_status": native_request_records[0].get("engine_timing_status") if native_request_records else "unavailable",
                      "engine_timing_source": native_request_records[0].get("engine_timing_source") if native_request_records else None,
                      "tpot_ms": native_tpot_ms, "client_tpot_ms": native_tpot_ms,
                      "client_ttft_ms": native_request_ttft,
                      "client_e2e_ms": native_request_e2e,
                      "total_ms": native_total_ms,
                      "request_to_first_token_ms": native_request_ttft,
                      "request_to_end_ms": native_request_e2e,
                      "stop_type": native.get("stop_type"), "truncated": native.get("truncated"),
                      "actual_output_tokens": output_tokens,
                      "client_wall_ms": wall_ms, "request_boundary": request_boundary,
                      "timings": timings, "perf_log": parse_perf_log(log_path),
                      "metrics_delta": metric_delta, "metrics_before": before_metrics,
                      "metrics_after": after_metrics,
                      "slots_before": slots_before if 'slots_before' in locals() else None,
                      "slots_after": slots_after if 'slots_after' in locals() else None,
                      # Preserve every concurrent request.  ``native`` above
                      # remains request-0 for backwards compatibility with
                      # existing single-request consumers.
                      "requests": native_request_records,
                      "aggregate": native_aggregate,
                      "identity": request_identity}
    # Immutable evidence manifest used by simulator-only replay.  The llama
    # server log is retained byte-for-byte as the raw event stream; parsed
    # perf_log is the extractor output and is hashed independently.
    raw_trace_sha256 = hashlib.sha256(log_path.read_bytes()).hexdigest() if log_path.exists() else None
    extractor_bytes = json.dumps(native_payload["perf_log"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw_event_count = 0
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as raw_handle:
            raw_event_count = sum(1 for _ in raw_handle)
    timing_contract = {
        "id": "engine-stage+client-real-token/v3",
        "engine_ttft": "first_engine_token-engine_request_begin;counter=t_prompt_last-t_start",
        "engine_e2e": "last_engine_token-engine_request_begin;counter=t_gen_last-t_start",
        "engine_tpot": "(last_engine_token-first_engine_token)/(output_tokens-1);counter=(t_gen_last-t_prompt_last)/(output_tokens-1);null_if_output_tokens<=1",
        "ttft": "first_real_token-client_request_start",
        "e2e": "last_real_token-client_request_start;DONE_excluded",
        "tpot": "(last_real_token-first_real_token)/(output_tokens-1);null_if_output_tokens<=1",
        "unit": "ms", "clock": "time.perf_counter",
    }
    native_contract_inputs = {
        "command": cmd,
        "configuration": {
            "ctx": args.ctx, "parallel": args.parallel, "batch": args.batch, "ubatch": args.ubatch,
            "threads": args.threads, "threads_batch": args.threads, "gpu_layers": args.gpu_layers,
            "flash_attn": False, "mmap": True, "mlock": False, "offload_kqv": True, "op_offload": True,
            "split_mode": "layer", "main_gpu": 0, "cpu_range": None, "cpu_range_batch": None, "numa": None,
            "kv_type_k": "f16", "kv_type_v": "f16", "kv_unified": True, "continuous_batching": True,
            "coherent_dma_mode": args.coherent_dma_mode, "mtp": False, "temperature": args.temperature,
            "top_k": args.top_k, "seed": args.seed, "stop": list(args.stop),
            "warmup_predict": args.warmup_predict, "request_timing": args.request_timing,
        },
        "request": {"prompt": args.prompt, "prompt_fingerprint": stable_hash(args.prompt),
                     "requested_output_tokens": args.predict, "output_mode": args.output_mode,
                     "ignore_eos": args.output_mode == "fixed", "warmup_output_tokens": args.warmup_predict},
        "output_policy": {"mode": args.output_mode, "ignore_eos": args.output_mode == "fixed"},
        "token_counts": {"prompt": prompt_tokens, "output": output_tokens, "requested_output": args.predict},
        "runtime_config": runtime_config.to_dict(), "identity": request_identity,
        "hardware": {**hardware, "llama_cpp": "0.3.0-dev build1 commit 0f3a71b"},
    }
    native_payload["evidence"] = {
        "schema": "native-evidence/v1",
        "native_binary": _trace_artifact_ref(Path(args.exe)),
        "runtime_artifacts": _runtime_artifact_refs(Path(args.exe)),
        "extractor": native_extractor_identity(),
        "timing_contract": {**timing_contract, "sha256": hashlib.sha256(json.dumps(timing_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        "engine_timing": {
            "status": native_payload.get("engine_timing_status", "unavailable"),
            "source": native_payload.get("engine_timing_source"),
            "fields": ["engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms"],
            "boundary": "server_slot_stats.t_start→t_prompt_last→t_gen_last" if native_payload.get("engine_timing_status") == "counter_proven" else "explicit_engine_boundary",
        },
        # Bind the measured native object itself.  Configuration identity and
        # extractor output hashes alone do not protect aggregate engine
        # counters from post-capture edits.
        "native_measurements_sha256": _native_measurements_digest(native_payload),
        "native_contract_sha256": stable_hash(native_contract_inputs),
        "raw_trace_events": {
            "path": str(log_path.resolve()),
            "sha256": raw_trace_sha256,
            "format": "llama.cpp-server-log",
            "event_count": raw_event_count if log_path.exists() else 0,
            "status": "captured" if raw_trace_sha256 else "missing",
            "supplemental_artifacts": [_trace_artifact_ref(path) for path in args.raw_trace_artifact],
        },
        "token_timestamps": {
            "status": "captured",
            "scope": "per_request",
            "field": "native.requests[].request_boundary.token_chunk_times_ms",
        },
        "request_boundaries": {
            "status": "captured" if request_boundaries else "missing",
            "scope": "per_request",
            "field": "native.requests[].request_boundary",
        },
        "extractor_output": {
            "status": "captured" if native_payload.get("perf_log") else "missing",
            "field": "native.perf_log",
            "sha256": hashlib.sha256(extractor_bytes).hexdigest(),
        },
        "extractor_output_artifacts": [_trace_artifact_ref(path) for path in args.extractor_output_artifact],
        "extractor_script_artifacts": [_trace_artifact_ref(path) for path in args.extractor_script_artifact],
        "identity": {
            "status": "captured",
            "field": "identity",
            "configuration_fields": sorted(request_identity["configuration"]),
        },
    }
    result = {"schema": "native-simulator-comparison/v2",
              "prediction_artifact": str(prediction_artifact.resolve()),
              "prediction_sha256": prediction_sha256,
              "prediction_status": "saved_before_native_reveal",
              "native_reveal_timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "validity_status": "request_boundary_aligned" if boundary_measured else "timing_comparison_only",
              "validity_note": "Engine TTFT/TPOT/E2E are the primary acceptance metrics (prompt_eval/eval boundary); client stream TTFT/TPOT/E2E remain secondary diagnostics.",
              "command": cmd, "server": {"port": port, "pid": proc.pid}, "log": str(log_path), "model": str(Path(args.model).resolve()), "gguf": parity,
              "parity": {"geometry": parity, "ctx": {"native": args.ctx, "simulator": args.ctx, "ok": True}, "tokens": token_parity},
              "eligibility": {"status": "eligible" if token_parity["ok"] else "ineligible", "reasons": [] if token_parity["ok"] else [token_parity_error]},
              "configuration": {"ctx": args.ctx, "parallel": args.parallel, "batch": args.batch, "ubatch": args.ubatch, "threads": args.threads, "threads_batch": args.threads, "gpu_layers": args.gpu_layers, "flash_attn": False, "mmap": True, "mlock": False, "offload_kqv": True, "op_offload": True, "split_mode": "layer", "main_gpu": 0, "cpu_range": None, "cpu_range_batch": None, "numa": None, "kv_type_k": "f16", "kv_type_v": "f16", "kv_unified": True, "continuous_batching": True, "coherent_dma_mode": args.coherent_dma_mode, "mtp": False, "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed, "stop": list(args.stop), "warmup_predict": args.warmup_predict, "request_timing": args.request_timing},
              "token_counts": {"prompt": prompt_tokens, "output": output_tokens, "requested_output": args.predict}, "output_policy": {"mode": args.output_mode, "ignore_eos": args.output_mode == "fixed"}, "warmup": {"wall_ms": warmup_wall_ms, "timings": warmup.get("timings", {})},
              "native": native_payload, "evidence": native_payload["evidence"], "simulator": {"ttft_ms": simulator_request_records[0].get("client_ttft_ms") if simulator_request_records else None, "client_ttft_ms": simulator_request_records[0].get("client_ttft_ms") if simulator_request_records else None, "engine_ttft_ms": simulator_request_records[0].get("engine_ttft_ms") if simulator_request_records else None, "tpot_ms": simulator_request_records[0].get("client_tpot_ms") if simulator_request_records else None, "client_tpot_ms": simulator_request_records[0].get("client_tpot_ms") if simulator_request_records else None, "engine_tpot_ms": simulator_request_records[0].get("engine_tpot_ms") if simulator_request_records else None, "e2e_ms": simulator_request_records[0].get("client_e2e_ms") if simulator_request_records else None, "client_e2e_ms": simulator_request_records[0].get("client_e2e_ms") if simulator_request_records else None, "engine_e2e_ms": simulator_request_records[0].get("engine_e2e_ms") if simulator_request_records else None, "makespan_ms": float(getattr(sim.serving, "makespan_ns", 0.0)) / 1e6, "client_makespan_ms": simulator_aggregate.get("client_makespan_ms"), "requests": simulator_request_records, "aggregate": simulator_aggregate}, "hardware": {**hardware, "llama_cpp": "0.3.0-dev build1 commit 0f3a71b"},
              "identity": request_identity,
              "parallel_support": {"requested": args.parallel, "native_requests": len(native_request_records), "simulator_requests": len(simulator_request_records), "status": "modeled" if len(simulator_request_records) == args.parallel else "mismatch"}}
    # Engine is the primary acceptance boundary.  Client values are retained
    # as a secondary service diagnostic and never mixed into this headline.
    result["relative_error_pct"] = {
        "ttft_ms": (100.0 * (result["simulator"]["engine_ttft_ms"] - native_prompt_ms) / native_prompt_ms
                    if native_prompt_ms not in (None, 0) else None),
        "tpot_ms": (100.0 * (result["simulator"]["engine_tpot_ms"] - native_engine_tpot_ms) / native_engine_tpot_ms
                    if native_engine_tpot_ms not in (None, 0) and result["simulator"].get("engine_tpot_ms") is not None else None),
        "e2e_ms": (100.0 * (result["simulator"]["engine_e2e_ms"] - native_total_ms) / native_total_ms
                   if native_total_ms not in (None, 0) else None),
    }
    result["relative_error_client_pct"] = {
        "ttft_ms": (100.0 * (result["simulator"]["client_ttft_ms"] - native_request_ttft) / native_request_ttft
                    if native_request_ttft not in (None, 0) and result["simulator"].get("client_ttft_ms") is not None else None),
        "tpot_ms": (100.0 * (result["simulator"]["client_tpot_ms"] - native_tpot_ms) / native_tpot_ms
                    if native_tpot_ms not in (None, 0) and result["simulator"].get("client_tpot_ms") is not None else None),
        "e2e_ms": (100.0 * (result["simulator"]["client_e2e_ms"] - native_request_e2e) / native_request_e2e
                   if native_request_e2e not in (None, 0) and result["simulator"].get("client_e2e_ms") is not None else None),
    }
    # Aggregate p50 errors are the appropriate headline for a concurrent
    # batch; request-0 errors above remain for backwards compatibility.
    native_agg = native_aggregate
    sim_agg = simulator_aggregate
    def _p50_error(native_metric: Mapping[str, object], sim_metric: Mapping[str, object]) -> float | None:
        native_p50 = native_metric.get("p50_ms")
        sim_p50 = sim_metric.get("p50_ms")
        if native_p50 in (None, 0) or sim_p50 is None:
            return None
        return 100.0 * (float(sim_p50) - float(native_p50)) / float(native_p50)
    result["relative_error_aggregate_pct"] = {
        "ttft_ms": _p50_error(native_agg["engine_ttft_ms"], sim_agg["engine_ttft_ms"]),
        "tpot_ms": _p50_error(native_agg["engine_tpot_ms"], sim_agg["engine_tpot_ms"]),
        "e2e_ms": _p50_error(native_agg["engine_e2e_ms"], sim_agg["engine_e2e_ms"]),
        "makespan_ms": (100.0 * (float(sim_agg["makespan_ms"]) - float(native_agg["makespan_ms"])) / float(native_agg["makespan_ms"])
                        if native_agg.get("makespan_ms") not in (None, 0) and sim_agg.get("makespan_ms") is not None else None),
    }
    result["relative_error_client_aggregate_pct"] = {
        "ttft_ms": _p50_error(native_agg["client_ttft_ms"], sim_agg["client_ttft_ms"]),
        "tpot_ms": _p50_error(native_agg["client_tpot_ms"], sim_agg["tpot_ms"]),
        "e2e_ms": _p50_error(native_agg["client_e2e_ms"], sim_agg["client_e2e_ms"]),
    }
    # A compact top-level view makes matrix tooling independent of whether it
    # consumes native or simulator payloads, while retaining the detailed
    # arrays under each side for auditability.
    result["requests"] = {"native": native_request_records, "simulator": simulator_request_records}
    result["relative_error_diagnostic_pct"] = {
        "ttft_ms": (100.0 * (result["simulator"]["ttft_ms"] - native_prompt_ms) / native_prompt_ms if native_prompt_ms else None),
        "e2e_ms": (100.0 * (result["simulator"]["engine_e2e_ms"] - native_total_ms) / native_total_ms if native_total_ms else None),
    }
    result["timing_comparison"] = {
        "headline": {
            "boundary": "engine",
            "ttft_ms": {"native_field": "native.engine_ttft_ms", "simulator_field": "simulator.engine_ttft_ms", "status": "measured" if native_prompt_ms is not None else "unavailable"},
            "tpot_ms": {"native_field": "native.engine_tpot_ms", "simulator_field": "simulator.engine_tpot_ms", "status": "measured" if native_engine_tpot_ms is not None else "unavailable"},
            "e2e_ms": {"native_field": "native.engine_e2e_ms", "simulator_field": "simulator.engine_e2e_ms", "status": "measured" if native_total_ms is not None else "unavailable"},
        },
        "secondary_client": {
            "boundary": "client_stream",
            "ttft_ms": {"native_field": "native.client_ttft_ms", "simulator_field": "simulator.client_ttft_ms", "status": "measured" if native_request_ttft is not None else "unavailable"},
            "tpot_ms": {"native_field": "native.client_tpot_ms", "simulator_field": "simulator.client_tpot_ms", "status": "measured" if native_tpot_ms is not None else "unavailable"},
            "e2e_ms": {"native_field": "native.client_e2e_ms", "simulator_field": "simulator.client_e2e_ms", "status": "measured" if native_request_e2e is not None else "unavailable"},
        },
    }
    result["runtime_config"] = runtime_config.to_dict()
    # Preserve the authored input alongside native token counts.  A prompt
    # label such as "medium" is not enough to prove train/holdout parity.
    result["request"] = {
        "prompt": args.prompt,
        "requested_output_tokens": args.predict,
        "output_mode": args.output_mode,
        "ignore_eos": args.output_mode == "fixed",
        "warmup_output_tokens": args.warmup_predict,
        "prompt_fingerprint": stable_hash(args.prompt),
    }
    result["calibration"] = calibration.to_dict() if calibration is not None else None
    result["calibration_apply_stage"] = bool(args.apply_stage_calibration)
    result["calibration_apply_memory"] = bool(args.apply_memory_calibration)
    result["calibration_apply_phase_boundary"] = bool(args.apply_phase_boundary_calibration)
    result["calibration_apply_request_boundary"] = bool(args.apply_request_boundary_calibration)
    result["runtime_fingerprint"] = runtime_config.fingerprint
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())








