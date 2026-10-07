"""Run graph-disabled native pairs while pausing only this repo's UI workers.

Default is a read-only inventory. --execute temporarily suspends the verified
workers on ports 8765-8785 and resumes every worker suspended here in a finally block. It does
not cancel jobs, terminate simulation servers, or change the parent env.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/frontend_native_validation_2026-10-07"
PYTHON = ROOT / ".venv/Scripts/python.exe"
CASES = ("qwen3_0_6b_f16", "qwen3_8_27b_mixed")


def ui_port(command):
    if len(command) != 7 or command[1:5] != ["-m", "heterollm_sim.cli", "ui", "--no-browser"] or command[5] != "--port":
        return None
    try:
        port = int(command[6])
    except ValueError:
        return None
    return port if 8765 <= port <= 8785 else None


def workers():
    selected = []
    for process in psutil.process_iter():
        try:
            command = process.cmdline()
            port = ui_port(command)
            if port is None or Path(process.cwd()).resolve() != ROOT:
                continue
            parent = process.parent()
            if parent is None:
                continue
            parent_command = parent.cmdline()
            # Windows venv launcher lives in this checkout; only its matching
            # interpreter child with this checkout's cwd may be suspended.
            if ui_port(parent_command) != port or Path(parent_command[0]).resolve() != PYTHON:
                continue
            selected.append((process, {"pid": process.pid, "parent_pid": parent.pid,
                "port": port, "create_time": process.create_time(), "cwd": process.cwd(),
                "executable": process.exe(), "parent_executable": parent_command[0],
                "status_before": process.status(), "suspended": False, "resumed": False}))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(selected, key=lambda item: item[1]["port"])


def gpu_state():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader"], text=True, capture_output=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--diagnostic-pid", type=int, action="append", default=[],
                        help="explicitly coordinated diagnostic worker PID, with matching repo-venv parent and repo cwd")
    args = parser.parse_args()
    targets = workers()
    for pid in args.diagnostic_pid:
        process = psutil.Process(pid)
        parent = process.parent()
        command = process.cmdline()
        parent_command = parent.cmdline() if parent else []
        if (Path(process.cwd()).resolve() != ROOT or len(command) < 2 or len(parent_command) < 2
                or Path(parent_command[0]).resolve() != PYTHON or command[1:] != parent_command[1:]):
            raise RuntimeError("Diagnostic PID does not match this repo's verified launcher chain")
        targets.append((process, {"pid": pid, "parent_pid": parent.pid, "port": None,
            "kind": "explicitly_coordinated_diagnostic", "command": command,
            "create_time": process.create_time(), "cwd": process.cwd(), "executable": process.exe(),
            "parent_executable": parent_command[0], "status_before": process.status(),
            "suspended": False, "resumed": False}))
    if not args.execute:
        print(json.dumps([row for _, row in targets], ensure_ascii=False, indent=2))
        return 0
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "controlled_cuda_graph_disabled_native_remeasurement",
        "scope": "Verified interpreter children of this repo's venv UI launchers on ports 8765-8785 and explicitly coordinated diagnostic PIDs; no jobs cancelled.",
        "workers": [row for _, row in targets], "runs": [], "status": "running"}
    path = REPORT / "native_environment_no_cuda_graphs.json"
    def save():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    suspended = []
    try:
        for process, row in targets:
            if row["status_before"] == psutil.STATUS_STOPPED:
                row["untouched_reason"] = "already_suspended_before_this_measurement"
                continue
            process.suspend()
            suspended.append((process, row))
            row["suspended"] = True
            row["status_while_suspended"] = process.status()
            save()
        manifest["cpu_before_load_percent"] = psutil.cpu_percent(interval=2.0)
        manifest["gpu_before"] = gpu_state()
        competing = [p.pid for p in psutil.process_iter(["name"])
            if str(p.info.get("name", "")).lower() in {"llama-server.exe", "llama-cli.exe"}]
        manifest["competing_inference_pids"] = competing
        if competing:
            raise RuntimeError("Another native inference process is active")
        if manifest["cpu_before_load_percent"] > 10:
            raise RuntimeError("CPU still busy after pausing simulation workers")
        save()
        for slug in CASES:
            historical = REPORT / ("native_default_config_" + slug + ".json")
            old = json.loads(historical.read_text(encoding="utf-8"))
            config = old["configuration"]
            command = [str(PYTHON), str(ROOT / "tools/native_benchmark.py"),
                "--server", old["identity"]["server_path"], "--model", config["model_path"],
                "--output", str(REPORT / ("native_" + slug + ".json")),
                "--prompt-tokens", "512", "--output-tokens", "128", "--context", "768",
                "--expected-effective-context", "768", "--batch", "512", "--ubatch", "512",
                "--threads", "16", "--parallel", "1", "--gpu-layers", "-1", "--flash-attn", "off",
                "--repetitions", "5", "--warmup", "2", "--disable-cuda-graphs"]
            started = time.perf_counter()
            result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True,
                timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            row = {"case_id": slug, "exit_code": result.returncode,
                "wall_seconds": time.perf_counter() - started, "command": command,
                "stdout": result.stdout, "stderr": result.stderr, "gpu_after": gpu_state()}
            manifest["runs"].append(row)
            save()
            print(slug, result.returncode, row["wall_seconds"], flush=True)
            if result.returncode:
                raise RuntimeError("Native benchmark failed: " + slug)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        for process, row in reversed(suspended):
            try:
                if process.create_time() != row["create_time"]:
                    raise RuntimeError("PID identity changed before resume")
                process.resume()
                row["resumed"] = True
                row["status_after_resume"] = process.status()
            except psutil.NoSuchProcess:
                row["exited_before_resume"] = True
            except BaseException as error:
                row["resume_error"] = type(error).__name__ + ": " + str(error)
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        failed_resumes = [row["pid"] for _, row in suspended if not row["resumed"] and not row.get("exited_before_resume")]
        if failed_resumes:
            raise RuntimeError("Workers require resume: " + str(failed_resumes))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
