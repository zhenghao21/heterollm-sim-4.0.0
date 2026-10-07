#!/usr/bin/env python3
"""Inspect native placement with verbose logs; never supplies latency samples."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import native_benchmark as native


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "frontend_native_validation_2026-10-07"


def gpu_snapshot() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def diagnose(slug: str) -> dict:
    reference_path = REPORTS / f"native_{slug}.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    configuration = reference["configuration"]
    command = list(configuration["server_command"])
    port = native._free_local_port()
    command[command.index("--port") + 1] = str(port)
    command += ["--verbosity", "5"]
    base_url = f"http://127.0.0.1:{port}"
    child_env = os.environ.copy()
    child_env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    prefix = REPORTS / f"diagnostic_native_path_{slug}"
    result = {
        "schema": "heterollm.native-path-diagnostic/v1",
        "purpose": "Placement and operation-path audit only; all diagnostic timings are excluded from comparison.",
        "used_for_latency_comparison": False,
        "reference_measurement": reference_path.name,
        "command": command,
        "effective_env": {"GGML_CUDA_DISABLE_GRAPHS": "1"},
        "stdout_file": prefix.name + "_stdout.log",
        "stderr_file": prefix.name + "_stderr.log",
        "status": "started",
    }
    process = None
    with prefix.with_name(prefix.name + "_stdout.log").open("wb") as stdout, prefix.with_name(prefix.name + "_stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env=child_env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            result["pid"] = process.pid
            native._wait_ready(process, base_url, 180)
            status, props = native._json_request(base_url, "GET", "/props")
            if status != 200:
                raise RuntimeError(f"props HTTP {status}")
            native._require_effective_context(props, [], 768, 768)
            result["server_build_info"] = props.get("build_info")
            result["effective_context"] = native._actual_context(props, [], 768)
            result["gpu_after_loading"] = gpu_snapshot()
            prompt = reference["prompt"]["token_ids"]
            if len(prompt) != 512:
                raise ValueError("reference prompt must contain exactly 512 tokens")
            payload = native._completion_payload(prompt, SimpleNamespace(output_tokens=128))
            response, _, _ = native._stream_completion(base_url, payload, 900)
            timings = native._server_timings(response["events"])
            if timings is None:
                raise RuntimeError("missing server counters")
            final = response["final"]
            observed = {
                "prompt_n": timings.get("prompt_n"),
                "cache_n": timings.get("cache_n"),
                "tokens_evaluated": final.get("tokens_evaluated"),
                "tokens_predicted": final.get("tokens_predicted"),
            }
            result["request_contract"] = observed
            if observed != {"prompt_n": 512, "cache_n": 0, "tokens_evaluated": 512, "tokens_predicted": 128}:
                raise RuntimeError(f"invalid diagnostic request contract: {observed}")
            result["gpu_after_request_server_alive"] = gpu_snapshot()
            result["gpu_snapshot_scope"] = "Device-wide allocations at these instants, including other processes; not peak VRAM or per-process allocation. Model/KV/compute buffer sizes are in stderr."
            result["status"] = "completed"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    stdout_path = prefix.with_name(prefix.name + "_stdout.log")
    if stdout_path.stat().st_size == 0:
        stdout_path.unlink()
        result["stdout_file"] = None
        result["stdout_retention"] = "Empty output; no file retained."
    prefix.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    for slug in ("qwen3_0_6b_f16", "qwen3_8_27b_mixed"):
        result = diagnose(slug)
        print(f"{slug}: {result['status']}", flush=True)
        if result["status"] != "completed":
            raise SystemExit(result.get("error", "diagnostic failed"))
