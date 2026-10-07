#!/usr/bin/env python3
"""Collect llama-server request timings for a fixed single-request workload."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit


PROMPT_SEED = (
    "This is a fixed local inference benchmark prompt. Continue the passage "
    "with a clear and concise explanation. The purpose is to measure model "
    "inference latency under a repeatable token workload. "
)


def _json_request(base_url: str, method: str, path: str, payload: Any = None,
                  timeout: float = 15.0) -> tuple[int, Any]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if not raw:
            return response.status, None
        try:
            return response.status, json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return response.status, raw.decode("utf-8", errors="replace")
    finally:
        connection.close()


def _stream_completion(base_url: str, payload: dict[str, Any], timeout: float
                       ) -> tuple[dict[str, Any], int, int]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "Cache-Control": "no-cache",
    }
    started_ns = time.perf_counter_ns()
    first_visible_ns: int | None = None
    last_visible_ns: int | None = None
    visible_events: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []
    response_status = 0
    try:
        connection.request("POST", "/completion", body=body, headers=headers)
        response = connection.getresponse()
        response_status = response.status
        if response.status != 200:
            error_body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError("completion HTTP {}: {}".format(response.status, error_body))
        while True:
            line = response.readline()
            if not line:
                break
            if line in (b"\n", b"\r\n"):
                if data_lines:
                    raw = "\n".join(data_lines)
                    data_lines.clear()
                    if raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError as error:
                        raise RuntimeError("invalid SSE JSON: {}".format(raw[:300])) from error
                    if isinstance(event, dict):
                        events.append(event)
                        content = event.get("content")
                        visible = event.get("stop") is not True and isinstance(content, str) and bool(content)
                        if visible:
                            observed_ns = time.perf_counter_ns()
                            if first_visible_ns is None:
                                first_visible_ns = observed_ns
                            last_visible_ns = observed_ns
                            visible_events.append({"content": content, "elapsed_ns": observed_ns - started_ns})
                elif not line:
                    break
                continue
            if line.startswith(b"data:"):
                data_lines.append(line[5:].decode("utf-8", errors="replace").lstrip())
        if data_lines:
            raw = "\n".join(data_lines)
            if raw and raw != "[DONE]":
                event = json.loads(raw)
                if isinstance(event, dict):
                    events.append(event)
                    content = event.get("content")
                    if event.get("stop") is not True and isinstance(content, str) and content:
                        observed_ns = time.perf_counter_ns()
                        if first_visible_ns is None:
                            first_visible_ns = observed_ns
                        last_visible_ns = observed_ns
                        visible_events.append({"content": content, "elapsed_ns": observed_ns - started_ns})
    finally:
        connection.close()
    if first_visible_ns is None or last_visible_ns is None:
        raise RuntimeError("completion stream ended without a visible token")
    final = next((event for event in reversed(events) if event.get("stop") is True), None)
    if final is None:
        raise RuntimeError("completion stream ended without a final stop event")
    return {
        "events": events,
        "final": final,
        "request_start_ns": started_ns,
        "first_visible_ns": first_visible_ns,
        "last_visible_ns": last_visible_ns,
        "visible_content_events": visible_events,
        "ttft_ns": first_visible_ns - started_ns,
        "tpot_ns": (last_visible_ns - first_visible_ns) / max(1, int(payload["n_predict"]) - 1),
        "e2e_ns": last_visible_ns - started_ns,
        "http_status": response_status,
    }, started_ns, time.perf_counter_ns()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path, help="llama-server executable")
    parser.add_argument("--model", required=True, type=Path, help="GGUF model file")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=640)
    parser.add_argument("--expected-effective-context", type=int,
                        help="require the server's observed context to match the paired simulation")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--flash-attn", choices=("off", "on"), required=True)
    parser.add_argument("--disable-cuda-graphs", action="store_true",
                        help="set GGML_CUDA_DISABLE_GRAPHS=1 only in the llama-server child environment")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("prompt_tokens", "output_tokens", "context", "batch", "ubatch", "threads", "parallel", "repetitions"):
        if getattr(args, name) <= 0:
            raise ValueError("--{} must be positive".format(name.replace("_", "-")))
    if args.warmup < 0:
        raise ValueError("--warmup must be zero or positive")
    if args.expected_effective_context is not None and args.expected_effective_context <= 0:
        raise ValueError("--expected-effective-context must be positive")
    if args.gpu_layers < -1:
        raise ValueError("--gpu-layers must be -1 or greater")
    if args.parallel != 1:
        raise ValueError("this collector currently supports --parallel 1 only")
    if args.prompt_tokens + args.output_tokens > args.context:
        raise ValueError("prompt plus output tokens exceed --context")
    if not args.server.is_file():
        raise FileNotFoundError("server executable not found: {}".format(args.server))
    if not args.model.is_file():
        raise FileNotFoundError("model file not found: {}".format(args.model))


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(process: subprocess.Popen[bytes], base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup with code {}".format(process.returncode))
        try:
            status, payload = _json_request(base_url, "GET", "/health", timeout=2.0)
            if status == 200 and isinstance(payload, dict) and payload.get("status") == "ok":
                return
            last_error = "health response was HTTP {}: {!r}".format(status, payload)
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError("llama-server health timeout: {}".format(last_error))


def _make_prompt_tokens(base_url: str, expected: int) -> list[int]:
    text = PROMPT_SEED * max(2, math.ceil(expected / 8))
    for _ in range(8):
        status, payload = _json_request(base_url, "POST", "/tokenize", {
            "content": text,
            "add_special": False,
            "parse_special": False,
            "with_pieces": False,
        })
        if status != 200 or not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
            raise RuntimeError("/tokenize failed: HTTP {} {!r}".format(status, payload))
        tokens = payload["tokens"]
        if len(tokens) >= expected:
            return [int(token) for token in tokens[:expected]]
        text += PROMPT_SEED * max(1, math.ceil((expected - len(tokens)) / 8))
    raise RuntimeError("could not construct prompt with exactly {} tokens".format(expected))


def _completion_payload(prompt: list[int], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": args.output_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": False,
        "return_progress": False,
        "timings_per_token": True,
        "return_tokens": True,
    }


def _event_tokens(events: list[dict[str, Any]]) -> list[int]:
    tokens: list[int] = []
    for event in events:
        value = event.get("tokens")
        if isinstance(value, list):
            tokens.extend(int(token) for token in value if isinstance(token, int) and not isinstance(token, bool))
    return tokens


def _server_timings(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        value = event.get("timings")
        if isinstance(value, dict):
            return value
    return None


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "interval_min": min(values),
        "interval_max": max(values),
    }


def _read_stderr(process: subprocess.Popen[bytes], lines: list[str]) -> None:
    assert process.stderr is not None
    for raw in iter(process.stderr.readline, b""):
        lines.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))


def _actual_context(props: Any, stderr_lines: list[str], requested: int) -> dict[str, Any]:
    default_settings = props.get("default_generation_settings", {}) if isinstance(props, dict) else {}
    props_context = default_settings.get("n_ctx") if isinstance(default_settings, dict) else None
    slot_context = None
    for line in stderr_lines:
        match = re.search(r"n_ctx_slot\s*=\s*(\d+)", line)
        if match:
            slot_context = int(match.group(1))
    return {
        "requested_context": requested,
        "props_n_ctx": props_context,
        "startup_n_ctx_slot": slot_context,
    }


def _require_effective_context(props: Any, stderr_lines: list[str], requested: int,
                               expected: int | None) -> None:
    if expected is None:
        return
    actual = _actual_context(props, stderr_lines, requested)
    observed = [actual[key] for key in ("props_n_ctx", "startup_n_ctx_slot")
                if actual[key] is not None]
    if not observed or any(type(value) is not int or value != expected for value in observed):
        raise RuntimeError("server effective context does not match paired simulation: "
                           "expected {}, observed {}".format(expected, actual))


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    args.server = args.server.resolve()
    args.model = args.model.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    port = _free_local_port()
    base_url = "http://127.0.0.1:{}".format(port)
    command = [
        str(args.server), "--model", str(args.model), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(args.context), "--batch-size", str(args.batch), "--ubatch-size", str(args.ubatch),
        "--threads", str(args.threads), "--threads-batch", str(args.threads), "--parallel", str(args.parallel),
        "--gpu-layers", str(args.gpu_layers), "--flash-attn", args.flash_attn,
        "--cache-type-k", "f16", "--cache-type-v", "f16", "--kv-unified", "--no-cache-prompt", "--fit", "off", "--perf",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    child_env = os.environ.copy()
    if args.disable_cuda_graphs:
        child_env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    # The locked source tests presence, not numeric truth. Record only this
    # relevant key, never the complete inherited environment or credentials.
    effective_env = {"GGML_CUDA_DISABLE_GRAPHS": child_env.get("GGML_CUDA_DISABLE_GRAPHS")}
    graph_configuration = {
        "effective_env": effective_env,
        "cuda_graphs_disabled": "GGML_CUDA_DISABLE_GRAPHS" in child_env,
        "cuda_graphs_disable_requested": args.disable_cuda_graphs,
    }
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=creationflags, env=child_env,
    )
    stderr_lines: list[str] = []
    stderr_thread = threading.Thread(target=_read_stderr, args=(process, stderr_lines), daemon=True)
    stderr_thread.start()
    samples: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    props: Any = None
    prompt_tokens: list[int] = []
    try:
        _wait_ready(process, base_url, args.startup_timeout)
        status, props = _json_request(base_url, "GET", "/props")
        if status != 200 or not isinstance(props, dict):
            raise RuntimeError("GET /props failed: HTTP {} {!r}".format(status, props))
        _require_effective_context(props, stderr_lines, args.context, args.expected_effective_context)
        prompt_tokens = _make_prompt_tokens(base_url, args.prompt_tokens)
        payload = _completion_payload(prompt_tokens, args)
        for index in range(args.warmup):
            result, _, _ = _stream_completion(base_url, payload, args.request_timeout)
            final = result["final"]
            timings = _server_timings(result["events"])
            warmups.append({
                "index": index + 1,
                "prompt_tokens_requested": args.prompt_tokens,
                "prompt_n": timings.get("prompt_n") if timings else None,
                "tokens_predicted": final.get("tokens_predicted"),
                "cache_n": timings.get("cache_n") if timings else None,
                "server_reported_cached_token_count": final.get("tokens_cached"),
            })
        for index in range(args.repetitions):
            result, _, _ = _stream_completion(base_url, payload, args.request_timeout)
            final = result["final"]
            timings = _server_timings(result["events"])
            prompt_n = timings.get("prompt_n") if timings else None
            prompt_ms = timings.get("prompt_ms") if timings else None
            predicted_ms = timings.get("predicted_ms") if timings else None
            predicted_n = timings.get("predicted_n") if timings else None
            predicted_per_token_ms = timings.get("predicted_per_token_ms") if timings else None
            output_count = final.get("tokens_predicted")
            if not isinstance(output_count, int):
                output_count = len(_event_tokens(result["events"]))
            sample = {
                "request_id": "request-0000",
                "repetition": index + 1,
                "prompt_tokens_requested": args.prompt_tokens,
                "prompt_n": prompt_n,
                "prompt_tokens_processed": final.get("tokens_evaluated"),
                "cache_n": timings.get("cache_n") if timings else None,
                "server_reported_cached_token_count": final.get("tokens_cached"),
                "visible_output_tokens": output_count,
                "engine_ttft_ns": float(prompt_ms) * 1_000_000 if isinstance(prompt_ms, (int, float)) else None,
                "engine_tpot_ns": float(predicted_per_token_ms) * 1_000_000 if isinstance(predicted_per_token_ms, (int, float)) else None,
                "engine_e2e_ns": (float(prompt_ms) + float(predicted_ms)) * 1_000_000 if isinstance(prompt_ms, (int, float)) and isinstance(predicted_ms, (int, float)) else None,
                "client_ttft_ns": result["ttft_ns"],
                "client_tpot_ns": result["tpot_ns"],
                "client_e2e_ns": result["e2e_ns"],
                "client_request_start_to_first_visible_token_ns": result["ttft_ns"],
                "client_first_visible_token_elapsed_ns": result["first_visible_ns"] - result["request_start_ns"],
                "client_last_visible_token_elapsed_ns": result["last_visible_ns"] - result["request_start_ns"],
                "client_visible_content_events": result["visible_content_events"],
                "server_timings": timings,
                "server_prompt_ms": prompt_ms,
                "server_prompt_n": prompt_n,
                "server_cache_n": timings.get("cache_n") if timings else None,
                "server_predicted_ms": predicted_ms,
                "server_predicted_n": predicted_n,
                "server_predicted_per_token_ms": predicted_per_token_ms,
                "stop_type": final.get("stop_type"),
                "stop": final.get("stop"),
            }
            errors = []
            if prompt_n != args.prompt_tokens:
                errors.append("server timings.prompt_n was {!r}, expected {}".format(prompt_n, args.prompt_tokens))
            if final.get("tokens_evaluated") != args.prompt_tokens:
                errors.append("server tokens_evaluated was {!r}, expected {}".format(final.get("tokens_evaluated"), args.prompt_tokens))
            cache_n = timings.get("cache_n") if timings else None
            if cache_n != 0:
                errors.append("server timings.cache_n was {!r}, expected 0 prompt tokens reused".format(cache_n))
            if output_count != args.output_tokens:
                errors.append("generated {} tokens, expected {}".format(output_count, args.output_tokens))
            if not isinstance(prompt_ms, (int, float)) or not isinstance(predicted_ms, (int, float)):
                errors.append("server timings did not include numeric prompt_ms and predicted_ms")
            if predicted_n != args.output_tokens:
                errors.append("server timings.predicted_n was {!r}, expected {}".format(predicted_n, args.output_tokens))
            if not isinstance(predicted_per_token_ms, (int, float)):
                errors.append("server timings did not include numeric predicted_per_token_ms")
            if errors:
                sample["status"] = "failed"
                sample["errors"] = errors
                samples.append(sample)
                raise RuntimeError("formal sample {} failed: {}".format(index + 1, "; ".join(errors)))
            sample["status"] = "completed"
            samples.append(sample)
        metric_summary = {
            group: {
                name: _summarize([float(sample["{}_{}_ns".format(group, name)]) for sample in samples])
                for name in ("ttft", "tpot", "e2e")
            }
            for group in ("engine", "client")
        }
        metric_summary["server_phases_ms"] = {
            name: _summarize([float(sample[name]) for sample in samples])
            for name in ("server_prompt_ms", "server_predicted_ms")
        }
        median_request = {
            "prompt_tokens": args.prompt_tokens,
            "visible_output_tokens": args.output_tokens,
            "ttft_ns": metric_summary["engine"]["ttft"]["median"],
            "tpot_ns": metric_summary["engine"]["tpot"]["median"],
            "e2e_ns": metric_summary["engine"]["e2e"]["median"],
        }
        median_client_request = {
            "prompt_tokens": args.prompt_tokens,
            "visible_output_tokens": args.output_tokens,
            "ttft_ns": metric_summary["client"]["ttft"]["median"],
            "tpot_ns": metric_summary["client"]["tpot"]["median"],
            "e2e_ns": metric_summary["client"]["e2e"]["median"],
        }
        return {
            "schema": "heterollm.native-benchmark/v1",
            "status": "completed",
            "measurement_boundary": {
                "engine_ttft": "server timings.prompt_ms from request timing start through first-token sampling",
                "engine_tpot": "server timings.predicted_per_token_ms = predicted_ms / (predicted_n - 1)",
                "engine_e2e": "server timings.prompt_ms + predicted_ms; excludes client HTTP/SSE transport",
                "client": "perf_counter_ns immediately before HTTP POST /completion to first/last non-empty content event",
                "stream_end": "stop=true final completion event is an empty notification and is excluded from visible-token timestamps",
            },
            "server_field_semantics": {
                "timings.prompt_n": "prompt tokens processed by the engine; checked against the requested prompt length",
                "timings.cache_n": "prompt tokens reused from prior context; checked as zero",
                "tokens_cached": "server completion field populated from slot.prompt.n_tokens; retained as reported cached-token count and not used as this-request cache hits",
            },
            "identity": {
                "model_path": str(args.model),
                "server_path": str(args.server),
                "server_build_info": props.get("build_info"),
            },
            "actual_server_context": _actual_context(props, stderr_lines, args.context),
            "configuration": {
                **graph_configuration,
                "model_path": str(args.model),
                "server_command": command,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
                "context": args.context,
                "expected_effective_context": args.expected_effective_context,
                "batch": args.batch,
                "ubatch": args.ubatch,
                "threads": args.threads,
                "threads_batch": args.threads,
                "gpu_layers_requested": args.gpu_layers,
                "flash_attn": args.flash_attn,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "parallel": args.parallel,
                "temperature": 0.0,
                "seed": 0,
                "ignore_eos": True,
                "cache_prompt": False,
                "speculative_decoding": False,
                "requested_repetitions": args.repetitions,
                "requested_warmups": args.warmup,
            },
            "native_server_props": props,
            "prompt": {"token_count": len(prompt_tokens), "token_ids": prompt_tokens},
            "warmups": warmups,
            "samples": samples,
            "summary": metric_summary,
            "requests": {"request-0000": median_request},
            "client_requests": {"request-0000": median_client_request},
            "repetitions": samples,
            "server_stderr": stderr_lines,
        }
    except Exception as error:
        failure = "{}: {}".format(type(error).__name__, error)
        return {
            "schema": "heterollm.native-benchmark/v1",
            "status": "failed",
            "error": failure,
            "identity": {"model_path": str(args.model), "server_path": str(args.server)},
            "actual_server_context": _actual_context(props, stderr_lines, args.context),
            "configuration": {"server_command": command, **graph_configuration},
            "native_server_props": props,
            "prompt": {"token_count": len(prompt_tokens), "token_ids": prompt_tokens},
            "warmups": warmups,
            "samples": samples,
            "server_stderr": stderr_lines,
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        stderr_thread.join(timeout=2)


def main() -> int:
    args = _parse_args()
    try:
        result = run(args)
    except Exception as error:
        result = {"schema": "heterollm.native-benchmark/v1", "status": "failed", "error": "{}: {}".format(type(error).__name__, error)}
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("{}: {}".format(result.get("status"), output_path))
    if result.get("status") != "completed":
        print(result.get("error", "benchmark failed"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
