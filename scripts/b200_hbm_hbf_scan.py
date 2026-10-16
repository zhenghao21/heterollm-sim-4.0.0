"""Full-length B200 HBM/HBF scan with independent, resumable point processes.

No token sampling or numerical extrapolation is used. A point is published only
after the requested single request has actually completed in the serving kernel.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

MODELS = (
    {"model_id": "qwen2_5-0_5b", "label": "Qwen2.5-0.5B", "parameters_b": .5},
    {"model_id": "qwen2_5-1_5b", "label": "Qwen2.5-1.5B", "parameters_b": 1.5},
    {"model_id": "qwen2_5-3b", "label": "Qwen2.5-3B", "parameters_b": 3.},
    {"model_id": "qwen2_5-7b", "label": "Qwen2.5-7B", "parameters_b": 7.},
)
PROMPTS = (64 * 1024, 256 * 1024, 512 * 1024)
OUTPUTS = (1024, 5 * 1024, 10 * 1024)
ARCHITECTURES = ("HBM", "HBF")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def scenario_gc_thresholds():
    """Reduce full-heap scans in these dedicated simulation processes.

    Page records are mostly acyclic and are reclaimed by reference counting.
    Cyclic collection still runs, at wider allocation intervals. Keep this
    process-level setting out of the backend API, which can have thread peers.
    """
    previous = gc.get_threshold()
    configured = tuple(max(current, minimum) for current, minimum in
                       zip(previous, (7000, 100, 100)))
    gc.set_threshold(*configured)
    try:
        yield configured
    finally:
        gc.set_threshold(*previous)


def json_default(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)
                if not field.name.startswith("_")}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__ + " is not JSON serializable")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=json_default), encoding="utf-8")
    # Windows readers and file scanners can briefly block replacement. Retry
    # publication without rerunning the simulation or exposing partial JSON.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def point_id(model_id, architecture, prompt_tokens, output_tokens):
    return f"{model_id}__{architecture}__P{prompt_tokens}__D{output_tokens}"


def all_points():
    # Start with the shortest input/output pair across model sizes, then grow
    # output length and context. Each point still executes its full request.
    return [{"point_id": point_id(model["model_id"], architecture, prompt, output),
             "model_id": model["model_id"], "architecture": architecture,
             "prompt_tokens": prompt, "output_tokens": output}
            for prompt in PROMPTS for output in OUTPUTS for model in MODELS
            for architecture in ARCHITECTURES]


def initialize(output_dir):
    from b200_hbm_hbf_scan_config import describe_scan_parameters
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("points", "progress", "logs", "configurations"):
        (output_dir / name).mkdir(exist_ok=True)
    path = output_dir / "manifest.json"
    manifest = {"schema": "b200-hbm-hbf-scan-v1", "models": MODELS,
        "prompts": PROMPTS, "outputs": OUTPUTS, "architectures": ARCHITECTURES,
        "points": all_points(), "point_count": 72,
        "parameters": describe_scan_parameters(),
        "execution": "full_length_physical_serving", "retention_policy": "aggregate",
        "ttft_definition": "request arrival to first visible output token",
        "tpot_definition": "(last visible token time - first visible token time) / (output_tokens - 1)"}
    # JSON normalization makes tuples/lists compare by their persisted meaning.
    manifest = json.loads(json.dumps(manifest, default=json_default))
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "created_at"} != manifest:
            raise ValueError("scan parameters differ from this output directory; choose a new directory")
        return saved
    manifest["created_at"] = utc_now()
    write_json(path, manifest)
    return manifest


def collect_result(result, point):
    from heterollm_sim.serving import RequestStatus
    from heterollm_sim.reporting import _sum_batch_storage_traffic, _sum_batch_dram_traffic
    requests = list(result.serving.request_metrics.values())
    if len(requests) != 1:
        raise ValueError("single-request baseline returned a different request count")
    request = requests[0]
    if request.status is not RequestStatus.FINISHED or request.visible_output_tokens != point["output_tokens"]:
        raise ValueError(f"request did not finish all requested tokens: {request}")
    if request.ttft_ns is None or (point["output_tokens"] > 1 and request.tpot_ns is None):
        raise ValueError("completed request has no TTFT or TPOT")
    placement = result.scenario.placement
    if not placement.op_to_component or set(placement.op_to_component.values()) != {"gpu0"}:
        raise ValueError("model operators were moved away from the fixed B200 compute endpoint")
    expected = "hbm0" if point["architecture"] == "HBM" else "hbf0"
    if placement.kv_policy.cache_component != expected:
        raise ValueError("KV cache is not on the selected memory tier")
    if point["architecture"] == "HBF" and any(
            component.normalized_kind == "hbm" for component in result.scenario.hardware.components):
        raise ValueError("all-HBF deployment unexpectedly contains HBM")
    resources = defaultdict(float)
    for batch in result.serving.batches:
        metadata = batch.cost.metadata
        if metadata.get("physical_execution_scope") != "persistent_live_kernel":
            raise ValueError("batch metrics were not produced by the live physical kernel")
        for resource, value in metadata.get("resource_busy_ns", {}).items():
            resources[resource] += value
    return {"status": "completed", "ttft_ns": request.ttft_ns,
        "tpot_ns": request.tpot_ns, "request_metrics": request,
        "batch_count": len(result.serving.batches),
        "physical_execution_scope": "persistent_live_kernel",
        "resource_busy_ns": dict(resources),
        "storage_traffic": _sum_batch_storage_traffic(result),
        "dram_traffic": _sum_batch_dram_traffic(result),
        "resolved_placement": {"op_to_component": placement.op_to_component,
            "tensor_to_component": placement.tensor_to_component,
            "active_memory": expected, "kv": placement.kv_policy.cache_component}}


def worker(output_dir, point):
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.execution_control import ExecutionControl, ExecutionCancelledError
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.memory_allocator import AllocationError

    started = time.monotonic()
    progress_path = output_dir / "progress" / (point["point_id"] + ".json")
    row = {**point, "started_at": utc_now(), "ttft_ns": None, "tpot_ns": None,
           "python_executable": sys.executable}
    last_write = 0.0
    last_stage = None

    def progress(event):
        nonlocal last_write, last_stage
        now = time.monotonic()
        if now - last_write < 5 and last_stage == event.stage and event.completed != event.total:
            return
        last_write, last_stage = now, event.stage
        write_json(progress_path, {**point, "status": "running", "pid": os.getpid(),
            "updated_at": utc_now(), "wall_seconds": now - started, "progress": event})
        print(json.dumps({"stage": event.stage, "completed": event.completed,
            "total": event.total, "wall_seconds": round(now - started, 2)}), flush=True)

    previous_path = output_dir / "points" / (point["point_id"] + ".json")
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("status") == "failed":
            print("retrying previous failure: " + str(previous.get("reason")), flush=True)
            previous_path.unlink()
    write_json(progress_path, {**point, "status": "starting", "pid": os.getpid(), "updated_at": utc_now()})
    try:
        scenario = build_scan_scenario(point["model_id"], point["architecture"],
                                       point["prompt_tokens"], point["output_tokens"])
        write_json(output_dir / "configurations" / (point["point_id"] + ".json"), scenario)
        row["model"] = {"name": scenario.model.name, "layers": scenario.model.num_layers,
            "weight_bytes": scenario.model.total_declared_weight_bytes,
            "metadata": scenario.model.metadata}
        with scenario_gc_thresholds() as thresholds:
            row["gc_thresholds"] = thresholds
            result = run_scenario(scenario, retention_policy="aggregate", control=ExecutionControl(
                progress_callback=progress,
                cancellation_callback=lambda: (output_dir / "stop.request").exists()))
        row.update(collect_result(result, point))
    except ExecutionCancelledError as error:
        row.update(status="interrupted", reason=f"{type(error).__name__}: {error}")
    except (AllocationError, MemoryError) as error:
        # A host Python MemoryError is an execution failure, not evidence that
        # the simulated accelerator lacks capacity.
        modeled_capacity = isinstance(error, AllocationError) or (
            isinstance(error, MemoryError) and str(error).startswith(("HBF ", "insufficient HBF")))
        row.update(status="capacity_rejected" if modeled_capacity else "failed",
                   reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    except Exception as error:
        row.update(status="failed", reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    row.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
    write_json(output_dir / "points" / (point["point_id"] + ".json"), row)
    write_json(progress_path, {**point, "status": row["status"], "updated_at": utc_now(),
        "wall_seconds": row["wall_seconds"], "reason": row.get("reason")})
    print(json.dumps({"point_id": point["point_id"], "status": row["status"],
                      "wall_seconds": row["wall_seconds"]}), flush=True)
    return 1 if row["status"] == "failed" else 0


def run_queue(output_dir, workers, retry_failed=False):
    from b200_hbm_hbf_scan_report import write_report, refresh_markdown
    manifest = initialize(output_dir)
    # Keep one OS file lock for the manager lifetime, preventing two managers
    # from submitting the same scan. No hashes or copied source trees are used.
    with (output_dir / "manager.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            if lock.read(1) == b"":
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending, finished = [], []
        for point in manifest["points"]:
            path = output_dir / "points" / (point["point_id"] + ".json")
            saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if saved is not None and saved["status"] in {"completed", "capacity_rejected"}:
                finished.append(saved)
            elif saved is not None and saved["status"] == "failed" and not retry_failed:
                raise ValueError("failed point needs correction and --retry-failed: " + point["point_id"])
            else:
                pending.append(point)
        write_report(output_dir)
        last_report_at = time.monotonic()
        running = {}
        stop_reason = None
        try:
            while pending or running:
                if (output_dir / "stop.request").exists():
                    stop_reason = "stop_requested"
                    break
                changed = False
                while pending and len(running) < workers:
                    # Keep one job per medium. This avoids two large HBF page
                    # maps occupying the host at once while retaining an A/B pair.
                    occupied = {item[1]["architecture"] for item in running.values()}
                    index = next((i for i, point in enumerate(pending)
                                  if point["architecture"] not in occupied), None)
                    if index is None:
                        break
                    point = pending.pop(index)
                    log = (output_dir / "logs" / (point["point_id"] + ".log")).open("a", encoding="utf-8")
                    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                        "--output-dir", str(output_dir), "--point-id", point["point_id"]]
                    flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                    running[point["point_id"]] = (process, point, log, utc_now())
                    changed = True
                    print("started " + point["point_id"] + " pid=" + str(process.pid), flush=True)
                for identifier, (process, point, log, start) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del running[identifier]
                    path = output_dir / "points" / (identifier + ".json")
                    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                        **point, "status": "failed", "ttft_ns": None, "tpot_ns": None,
                        "reason": f"worker exited {code} without a complete point result"}
                    if not path.exists():
                        write_json(path, saved)
                    finished.append(saved)
                    changed = True
                    if saved["status"] == "failed":
                        stop_reason = "point_failed"
                write_json(output_dir / "status.json", {"status": "needs_attention" if stop_reason else "running",
                    "manager_pid": os.getpid(), "updated_at": utc_now(), "total": len(manifest["points"]),
                    "completed": sum(row["status"] == "completed" for row in finished),
                    "capacity_rejected": sum(row["status"] == "capacity_rejected" for row in finished),
                    "failed": sum(row["status"] == "failed" for row in finished),
                    "pending": len(pending), "running": [{**point, "pid": process.pid, "started_at": start}
                        for process, point, _log, start in running.values()], "reason": stop_reason})
                if changed:
                    write_report(output_dir)
                    last_report_at = time.monotonic()
                elif time.monotonic() - last_report_at >= 60:
                    refresh_markdown(output_dir)
                    last_report_at = time.monotonic()
                if stop_reason:
                    break
                if running:
                    time.sleep(5)
        except BaseException as error:
            stop_reason = "manager_interrupted" if isinstance(error, KeyboardInterrupt) else "manager_failed"
            raise
        finally:
            for process, point, log, start in running.values():
                process.terminate()
                process.wait()
                log.close()
                write_json(output_dir / "progress" / (point["point_id"] + ".json"), {
                    **point, "status": "interrupted", "updated_at": utc_now(), "reason": stop_reason or "manager_interrupted"})
            queue_state = ("paused" if stop_reason == "stop_requested" else
                           "needs_attention" if stop_reason else "completed")
            write_json(output_dir / "status.json", {"status": queue_state,
                "manager_pid": os.getpid(), "updated_at": utc_now(), "total": len(manifest["points"]),
                "completed": sum(row["status"] == "completed" for row in finished),
                "capacity_rejected": sum(row["status"] == "capacity_rejected" for row in finished),
                "failed": sum(row["status"] == "failed" for row in finished),
                "pending": len(pending) + len(running), "running": [], "reason": stop_reason})
            write_report(output_dir)
    return 1 if stop_reason else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "report", "stop"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--point-id")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.command == "run":
        return run_queue(output_dir, args.workers, args.retry_failed)
    if args.command == "worker":
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        point = next(point for point in manifest["points"] if point["point_id"] == args.point_id)
        return worker(output_dir, point)
    if args.command == "report":
        from b200_hbm_hbf_scan_report import write_report
        write_report(output_dir)
    if args.command == "stop":
        (output_dir / "stop.request").write_text(utc_now(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
