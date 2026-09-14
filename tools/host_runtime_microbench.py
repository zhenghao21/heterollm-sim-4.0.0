"""Measure host/runtime overheads without running an LLM generation.

The benchmark deliberately covers only work outside the model execution graph:
the locked llama-tokenize executable, loopback SSE framing, and JSON
encode/decode.  Its output is evidence for the host-front-end cost model; it
must never be used to fit model or operator latency.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-tokenize.exe"
DEFAULT_MODEL = ROOT / "artifacts/native_benchmark_20260912/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


def _stats(values_ns: list[int]) -> dict[str, float | int | str]:
    if not values_ns:
        raise ValueError("at least one sample is required")
    values_ms = [v / 1_000_000 for v in values_ns]
    ordered = sorted(values_ms)
    p90 = ordered[min(len(ordered) - 1, max(0, int((len(ordered) * 0.90) - 1e-12)))]
    return {
        "unit": "ms",
        "samples": len(values_ms),
        "min": min(values_ms),
        "median": statistics.median(values_ms),
        "p90": p90,
        "max": max(values_ms),
        "mean": statistics.mean(values_ms),
        "stdev": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def benchmark_tokenizer(tokenizer: Path, model: Path, prompts: list[str], samples: int) -> dict[str, object]:
    if not tokenizer.exists():
        raise FileNotFoundError(f"tokenizer executable not found: {tokenizer}")
    if not model.exists():
        raise FileNotFoundError(f"model required by tokenizer not found: {model}")
    rows: list[dict[str, object]] = []
    for prompt in prompts:
        timings: list[int] = []
        token_counts: list[int] = []
        command = [str(tokenizer), "-m", str(model), "-p", prompt, "--ids", "--show-count"]
        for _ in range(samples):
            started = time.perf_counter_ns()
            cp = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
            elapsed = time.perf_counter_ns() - started
            if cp.returncode != 0:
                raise RuntimeError(f"llama-tokenize failed ({cp.returncode}): {(cp.stderr or cp.stdout)[-500:]}")
            text = (cp.stdout or "") + "\n" + (cp.stderr or "")
            marker = "Total number of tokens:"
            if marker not in text:
                raise RuntimeError("llama-tokenize output did not contain token count")
            count = int(text.split(marker, 1)[1].split()[0])
            timings.append(elapsed)
            token_counts.append(count)
        rows.append({
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "token_count": token_counts[0] if len(set(token_counts)) == 1 else token_counts,
            "command": command,
            "latency": _stats(timings),
            "per_token_ms_median": (statistics.median(timings) / token_counts[0] / 1_000_000),
        })
    return {"executable": str(tokenizer.resolve()), "sha256": _sha256(tokenizer), "model": str(model.resolve()), "samples_per_prompt": samples, "cases": rows}


class _SSEHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes
    count: int

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for i in range(self.count):
            body = json.dumps({"index": i, "content": "x", "token": i}, separators=(",", ":")).encode()
            self.wfile.write(b"data: " + body + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *_args: object) -> None:
        return


def benchmark_loopback_sse(samples: int, token_count: int) -> dict[str, object]:
    if token_count < 2:
        raise ValueError("token_count must be at least 2")
    class Handler(_SSEHandler):
        count = token_count
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    first_ns: list[int] = []
    last_ns: list[int] = []
    request_ns: list[int] = []
    try:
        url = f"http://127.0.0.1:{server.server_port}/completion"
        for _ in range(samples):
            started = time.perf_counter_ns()
            req = Request(url, data=b"{}", headers={"Content-Type": "application/json"})
            first = last = None
            with urlopen(req, timeout=30) as response:
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    payload = raw[6:].strip()
                    if payload == b"[DONE]":
                        continue
                    now = time.perf_counter_ns()
                    first = first or now
                    last = now
            ended = time.perf_counter_ns()
            if first is None or last is None:
                raise RuntimeError("loopback SSE returned no token events")
            first_ns.append(first - started)
            last_ns.append(last - started)
            request_ns.append(ended - started)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    return {"url": "http://127.0.0.1", "token_count": token_count, "samples": samples,
            "ttft": _stats(first_ns), "last_token_e2e": _stats(last_ns), "request_end_including_done": _stats(request_ns),
            "done_tail_ms_median": statistics.median([(e - l) / 1_000_000 for e, l in zip(request_ns, last_ns)])}


def benchmark_json(samples: int, items: int = 16) -> dict[str, object]:
    value = {"id": "microbench", "tokens": [{"id": i, "content": "x"} for i in range(items)], "timing": {"prompt_ms": 1.25, "predicted_ms": 3.5}}
    encoded: list[int] = []
    decoded: list[int] = []
    for _ in range(samples):
        t0 = time.perf_counter_ns(); blob = json.dumps(value, separators=(",", ":")).encode(); encoded.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns(); json.loads(blob); decoded.append(time.perf_counter_ns() - t0)
    return {"items": items, "payload_bytes": len(blob), "samples": samples, "encode": _stats(encoded), "decode": _stats(decoded)}


def run(*, tokenizer: Path, model: Path, samples: int) -> dict[str, object]:
    prompts = ["Hi.", "Explain how deterministic benchmarking affects reproducibility in language model inference.", "Explain how deterministic benchmarking affects reproducibility in language model inference. " * 8]
    return {
        "schema_version": "host-runtime-microbench/v1",
        "purpose": "development_evidence_only",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "measurement_scope": {
            "native_llm_inference_started": False,
            "target_llm_timing_read": False,
            "model_latency_fit": False,
        },
        "tokenizer": {
            **benchmark_tokenizer(tokenizer, model, prompts, samples),
            "interpretation": "CLI wall time includes process startup and model metadata load; it is not an in-process llama-server tokenization coefficient.",
        },
        "loopback_sse": benchmark_loopback_sse(samples, 8),
        "json": benchmark_json(samples),
        "prohibited_use": "Do not fit model/operator latency or scenario multipliers from these measurements.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--samples", type=int, default=9)
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/development/host_runtime_microbench_v1.json")
    args = ap.parse_args()
    if args.samples < 1:
        ap.error("--samples must be positive")
    result = run(tokenizer=args.tokenizer, model=args.model, samples=args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "samples": args.samples}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
