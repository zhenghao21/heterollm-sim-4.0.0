#!/usr/bin/env python3
"""Verify CUDA Graph events with the actual server request; never retain timings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import native_benchmark as benchmark


def diagnose(server: Path, case_path: Path, trace_path: Path) -> dict:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    config = case["configuration"]
    command = list(config["server_command"])
    command[0] = str(server.resolve())
    port = benchmark._free_local_port()
    command[command.index("--port") + 1] = str(port)
    base_url = f"http://127.0.0.1:{port}"
    child_env = os.environ.copy()
    for key in ("HETEROLLM_CUDA_GRAPH_DRY_RUN", "HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY"):
        child_env.pop(key, None)
    for key, value in config["effective_env"].items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value
    child_env["HETEROLLM_CUDA_GRAPH_TRACE"] = str(trace_path.resolve())
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    process = subprocess.Popen(command, env=child_env, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    reader = threading.Thread(target=benchmark._read_stderr, args=(process, logs), daemon=True)
    reader.start()
    try:
        benchmark._wait_ready(process, base_url, 180)
        status, props = benchmark._json_request(base_url, "GET", "/props")
        if status != 200:
            raise RuntimeError("server properties unavailable")
        benchmark._require_effective_context(props, logs, config["context"], config["expected_effective_context"])
        payload = benchmark._completion_payload(case["prompt"]["token_ids"],
            SimpleNamespace(output_tokens=config["output_tokens"]))
        completed = []
        for repetition in range(config["requested_warmups"] + 1):
            result, _, _ = benchmark._stream_completion(base_url, payload, 900)
            timing = benchmark._server_timings(result["events"])
            if not timing or timing.get("prompt_n") != config["prompt_tokens"] or timing.get("cache_n") != 0:
                raise RuntimeError("diagnostic prompt length or cache state differs from timing run")
            if result["final"].get("tokens_predicted") != config["output_tokens"]:
                raise RuntimeError("diagnostic output length differs from timing run")
            completed.append({"warmup": repetition < config["requested_warmups"],
                              "prompt_tokens": config["prompt_tokens"], "output_tokens": config["output_tokens"]})
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        reader.join(timeout=2)
    snapshots, events = 0, Counter()
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["kind"] == "snapshot":
                snapshots += 1
            elif record["kind"] == "event":
                events[record["event"]] += 1
    disabled = config["cuda_graphs_disabled"]
    if (disabled and events["replay"] != 0) or (not disabled and events["replay"] == 0):
        raise RuntimeError("observed CUDA Graph replay contradicts requested diagnostic mode")
    return {"status": "completed", "source_revision": "d3146f2b56c2db4711ac8391871c9e529d1946d7",
        "paired_timing_case": str(case_path.resolve()), "diagnostic_server": str(server.resolve()),
        "trace_path": str(trace_path.resolve()), "latency_values_retained": False,
        "cuda_graphs_mode_requested": config["cuda_graphs_mode_requested"],
        "ctx_checkpoints_requested": config.get("ctx_checkpoints_requested"),
        "effective_env": config["effective_env"], "server_command": command,
        "requests": completed, "backend_calls": snapshots, "actual_events": dict(events)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.server, args.case, args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "events": result["actual_events"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
