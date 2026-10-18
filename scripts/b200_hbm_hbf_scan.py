"""B200 HBM/HBF scan with explicit fidelity and resumable point processes.

Fast mode executes Prefill and a real Decode prefix. Refused forecasts are
recorded and never silently start a detailed suffix in this scan.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from b200_hbm_hbf_scan_report import PHYSICAL_WEIGHT_SEMANTICS, validate_physical_weight_semantics

MODELS = (
    {"model_id": "qwen2_5-0_5b", "label": "Qwen2.5-0.5B", "parameters_b": .5},
    {"model_id": "qwen2_5-1_5b", "label": "Qwen2.5-1.5B", "parameters_b": 1.5},
    {"model_id": "qwen2_5-3b", "label": "Qwen2.5-3B", "parameters_b": 3.},
    {"model_id": "qwen2_5-7b", "label": "Qwen2.5-7B", "parameters_b": 7.},
)
PROMPTS = (64 * 1024, 256 * 1024, 512 * 1024)
OUTPUTS = (1024, 5 * 1024, 10 * 1024)
ARCHITECTURES = ("HBM", "HBF")
SUCCESS_STATUSES = {"completed", "forecast_completed"}
REUSABLE_STATUSES = SUCCESS_STATUSES | {"capacity_rejected", "forecast_rejected"}


class FastForecastRejected(RuntimeError):
    """The formal fast queue does not execute a rejected forecast's suffix."""


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
    # output length and context. Fast forecasts retain the full request budget.
    return [{"point_id": point_id(model["model_id"], architecture, prompt, output),
             "model_id": model["model_id"], "architecture": architecture,
             "prompt_tokens": prompt, "output_tokens": output}
            for prompt in PROMPTS for output in OUTPUTS for model in MODELS
            for architecture in ARCHITECTURES]


def initialize(output_dir, simulation_mode="fast"):
    from b200_hbm_hbf_scan_config import describe_scan_parameters
    from heterollm_sim.simulation_options import SimulationConfig
    simulation = SimulationConfig(mode=simulation_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("points", "progress", "logs", "configurations"):
        (output_dir / name).mkdir(exist_ok=True)
    path = output_dir / "manifest.json"
    manifest = {"schema": "b200-hbm-hbf-scan-v2", "models": MODELS,
        "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS,
        "prompts": PROMPTS, "outputs": OUTPUTS, "architectures": ARCHITECTURES,
        "points": all_points(), "point_count": 72,
        "parameters": describe_scan_parameters(),
        "simulation": simulation,
        "execution": "detailed_prefill_sampled_decode_forecast" if simulation_mode == "fast" else "full_length_physical_serving",
        "forecast_refusal_policy": "record_and_continue_without_detailed_suffix",
        "retention_policy": "aggregate",
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
    placement = result.scenario.placement
    if not placement.op_to_component or set(placement.op_to_component.values()) != {"gpu0"}:
        raise ValueError("model operators were moved away from the fixed B200 compute endpoint")
    expected = "hbm0" if point["architecture"] == "HBM" else "hbf0"
    if not placement.tensor_to_component or set(placement.tensor_to_component.values()) != {expected}:
        raise ValueError("model tensors are not all on the selected memory tier")
    if placement.kv_policy.cache_component != expected:
        raise ValueError("KV cache is not on the selected memory tier")
    if point["architecture"] == "HBF" and any(
            component.normalized_kind == "hbm" for component in result.scenario.hardware.components):
        raise ValueError("all-HBF deployment unexpectedly contains HBM")
    resolved = {"op_to_component": placement.op_to_component,
        "tensor_to_component": placement.tensor_to_component,
        "active_memory": expected, "kv": placement.kv_policy.cache_component}
    forecast = getattr(result, "forecast", None)
    if forecast is not None:
        approximation = deepcopy(result.approximation)
        decision = approximation.get("paired_budget_decision")
        if decision is not None and decision.get("accepted") is not True:
            raise ValueError("forecast has no accepted conditional timing budget")
        if (forecast.output_tokens != point["output_tokens"]
                or forecast.prompt_tokens != point["prompt_tokens"]
                or forecast.sampled_decode_steps + forecast.predicted_decode_steps != point["output_tokens"] - 1
                or approximation.get("effective_mode") != "fast"
                or not all(math.isfinite(value) and value > 0 for value in (forecast.ttft_ns, forecast.tpot_ns))):
            raise ValueError("forecast does not cover the original complete request")
        return {"status": "forecast_completed", "simulation_mode": "fast",
            "ttft_ns": forecast.ttft_ns, "tpot_ns": forecast.tpot_ns,
            "visible_output_tokens": forecast.output_tokens,
            "metric_fidelity": {"ttft": "detailed", "tpot": "estimated"},
            "physical_execution_scope": "prefix_only", "approximation": approximation,
            "resolved_placement": resolved}
    if hasattr(result, "approximation"):
        raise ValueError("formal fast scan unexpectedly returned detailed fallback")
    requests = list(result.serving.request_metrics.values())
    if len(requests) != 1:
        raise ValueError("single-request baseline returned a different request count")
    request = requests[0]
    if request.status is not RequestStatus.FINISHED or request.visible_output_tokens != point["output_tokens"]:
        raise ValueError(f"request did not finish all requested tokens: {request}")
    if request.ttft_ns is None or (point["output_tokens"] > 1 and request.tpot_ns is None):
        raise ValueError("completed request has no TTFT or TPOT")
    resources = defaultdict(float)
    for batch in result.serving.batches:
        metadata = batch.cost.metadata
        if metadata.get("physical_execution_scope") != "persistent_live_kernel":
            raise ValueError("batch metrics were not produced by the live physical kernel")
        for resource, value in metadata.get("resource_busy_ns", {}).items():
            resources[resource] += value
    return {"status": "completed", "ttft_ns": request.ttft_ns,
        "simulation_mode": "detailed", "metric_fidelity": {"ttft": "detailed", "tpot": "detailed"},
        "tpot_ns": request.tpot_ns, "request_metrics": request,
        "batch_count": len(result.serving.batches),
        "physical_execution_scope": "persistent_live_kernel",
        "resource_busy_ns": dict(resources),
        "storage_traffic": _sum_batch_storage_traffic(result),
        "dram_traffic": _sum_batch_dram_traffic(result),
        "resolved_placement": resolved}


def load_paired_reference(output_dir, point, scenario, manifest):
    """Reuse the matching completed HBM interval; never run a reference here."""
    from heterollm_sim.fast_simulation import PairedTimingReference
    validate_physical_weight_semantics(manifest, "paired-reference manifest")
    if point["architecture"] != "HBF" or scenario.simulation.mode != "fast":
        return None
    identifier = point_id(point["model_id"], "HBM", point["prompt_tokens"], point["output_tokens"])
    path = output_dir / "points" / (identifier + ".json")
    if not path.exists():
        return None
    source = json.loads(path.read_text(encoding="utf-8"))
    validate_physical_weight_semantics(source, "paired HBM reference")
    if source.get("status") != "forecast_completed":
        return None
    expected = {**point, "point_id": identifier, "architecture": "HBM"}
    if any(source.get(key) != expected[key] for key in (
            "point_id", "model_id", "architecture", "prompt_tokens", "output_tokens")):
        raise ValueError("paired HBM reference does not match the original model and request")
    approximation = source.get("approximation", {})
    parameters = manifest.get("parameters", {})
    if (source.get("simulation_mode") != "fast"
            or source.get("model", {}).get("name") != scenario.model.name
            or approximation.get("effective_mode") != "fast"
            or approximation.get("target_relative_error") != scenario.simulation.fast_decode.target_relative_error
            or source.get("metric_fidelity", {}).get("tpot") != "estimated"
            or source.get("resolved_placement", {}).get("active_memory") != "hbm0"
            or parameters.get("weights_dtype") != "bf16" or parameters.get("kv_dtype") != "bf16"):
        raise ValueError("paired HBM reference has incompatible fidelity, placement or precision")
    counts = [approximation.get("sampled_decode_steps"), approximation.get("predicted_decode_steps")]
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts)
            or sum(counts) != point["output_tokens"] - 1):
        raise ValueError("paired HBM reference does not cover the complete Decode budget")
    parts = approximation.get("error_budget_breakdown", {})
    combined = parts.get("combined", {})
    values = [source.get("tpot_ns"), parts.get("forecast_tail_ns"),
              combined.get("lower_ns"), combined.get("upper_ns")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("paired HBM reference has no finite positive conditional interval")
    tpot, tail_estimate, tail_lower, tail_upper = values
    estimate = tpot * (point["output_tokens"] - 1)
    prefix = estimate - tail_estimate
    if prefix < -max(1., estimate) * 1e-12 or not tail_lower <= tail_estimate <= tail_upper:
        raise ValueError("paired HBM reference has an inconsistent exact-prefix or tail interval")
    prefix = max(0., prefix)
    return PairedTimingReference(model_name=scenario.model.name,
        prompt_tokens=point["prompt_tokens"], output_tokens=point["output_tokens"],
        memory_kind="hbm", target_relative_error=scenario.simulation.fast_decode.target_relative_error,
        decode_estimate_ns=estimate, decode_lower_ns=prefix + tail_lower,
        decode_upper_ns=prefix + tail_upper, source_id=identifier,
        weight_dtype=parameters["weights_dtype"], kv_dtype=parameters["kv_dtype"])


def archive_rejected_point(output_dir, point, row):
    """Keep the refused attempt before the explicitly selected retry replaces it."""
    if row.get("status") != "forecast_rejected":
        raise ValueError("only a forecast-rejected point can be archived for this retry")
    directory = output_dir / "points_history"
    directory.mkdir(exist_ok=True)
    stamp = str(row.get("finished_at") or utc_now()).replace(":", "").replace("+", "_")
    destination = directory / (point["point_id"] + "__" + stamp + ".json")
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != row:
            raise ValueError("retry history already contains a different attempt")
    else:
        write_json(destination, row)
    (output_dir / "points" / (point["point_id"] + ".json")).unlink()
    return destination.relative_to(output_dir).as_posix()


def worker(output_dir, point):
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.execution_control import ExecutionControl, ExecutionCancelledError
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.memory_allocator import AllocationError
    from heterollm_sim.simulation_options import simulation_from_dict

    started = time.monotonic()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_physical_weight_semantics(manifest, "worker manifest")
    simulation = simulation_from_dict(manifest["simulation"])
    progress_path = output_dir / "progress" / (point["point_id"] + ".json")
    row = {**point, "started_at": utc_now(), "ttft_ns": None, "tpot_ns": None,
           "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS,
           "python_executable": sys.executable, "simulation_mode": simulation.mode,
           "simulation_config": simulation, "pid": os.getpid()}
    last_write = 0.0
    last_stage = None
    approximation = {}
    real_steps = {"prefill": 0, "decode": 0}

    def progress(event):
        nonlocal last_write, last_stage, approximation
        now = time.monotonic()
        if event.stage == "serving_cohorts" and event.completed:
            kind = event.metadata.get("cohort_kind")
            if kind in real_steps:
                real_steps[kind] += 1
        if "approximation" in event.metadata:
            approximation = deepcopy(event.metadata["approximation"])
        if (not event.stage.startswith("fast_") and now - last_write < 5
                and last_stage == event.stage and event.completed != event.total):
            return
        last_write, last_stage = now, event.stage
        write_json(progress_path, {**point, "status": "running", "pid": os.getpid(),
            "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS,
            "simulation_mode": simulation.mode, "approximation": approximation,
            "real_prefill_steps": real_steps["prefill"], "real_decode_steps": real_steps["decode"],
            "updated_at": utc_now(), "wall_seconds": now - started, "progress": event})
        print(json.dumps({"stage": event.stage, "completed": event.completed,
            "total": event.total, "wall_seconds": round(now - started, 2)}), flush=True)
        if simulation.mode == "fast" and event.stage == "fast_mode_fallback":
            raise FastForecastRejected(", ".join(event.metadata.get("fallback_reasons", ())))

    previous_path = output_dir / "points" / (point["point_id"] + ".json")
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("status") == "failed":
            print("retrying previous failure: " + str(previous.get("reason")), flush=True)
            previous_path.unlink()
    write_json(progress_path, {**point, "status": "starting", "pid": os.getpid(), "updated_at": utc_now(),
        "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS})
    try:
        scenario = build_scan_scenario(point["model_id"], point["architecture"],
                                       point["prompt_tokens"], point["output_tokens"])
        scenario = replace(scenario, simulation=simulation)
        paired_reference = load_paired_reference(output_dir, point, scenario, manifest)
        run_options = {"paired_reference": paired_reference} if paired_reference is not None else {}
        if paired_reference is not None:
            row["paired_reference"] = paired_reference
        write_json(output_dir / "configurations" / (point["point_id"] + ".json"), scenario)
        row["model"] = {"name": scenario.model.name, "layers": scenario.model.num_layers,
            "weight_bytes": scenario.model.total_declared_weight_bytes,
            "metadata": scenario.model.metadata}
        with scenario_gc_thresholds() as thresholds:
            row["gc_thresholds"] = thresholds
            result = run_scenario(scenario, retention_policy="aggregate", control=ExecutionControl(
                progress_callback=progress,
                cancellation_callback=lambda: (output_dir / "stop.request").exists()), **run_options)
        row.update(collect_result(result, point))
    except FastForecastRejected as error:
        row.update(status="forecast_rejected", reason=str(error), approximation=approximation,
            physical_execution_scope="prefix_only", metric_fidelity={"ttft": "unavailable", "tpot": "unavailable"})
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
    row.update(real_prefill_steps=real_steps["prefill"], real_decode_steps=real_steps["decode"])
    if approximation and "approximation" not in row:
        row["approximation"] = approximation
    write_json(output_dir / "points" / (point["point_id"] + ".json"), row)
    write_json(progress_path, {**point, "pid": os.getpid(), "simulation_mode": simulation.mode,
        "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS,
        "real_prefill_steps": real_steps["prefill"], "real_decode_steps": real_steps["decode"],
        "approximation": row.get("approximation", {}), "status": row["status"], "updated_at": utc_now(),
        "wall_seconds": row["wall_seconds"], "reason": row.get("reason")})
    print(json.dumps({"point_id": point["point_id"], "status": row["status"],
                      "wall_seconds": row["wall_seconds"]}), flush=True)
    return 1 if row["status"] in {"failed", "interrupted"} else 0


def process_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_bool, ctypes.c_uint]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def assert_no_live_workers(output_dir):
    # On Windows Popen may identify a venv launcher. Progress is written by
    # the actual interpreter, which can outlive an interrupted manager.
    for path in (output_dir / "progress").glob("*.json"):
        progress = json.loads(path.read_text(encoding="utf-8"))
        pid = progress.get("pid")
        if progress.get("status") in {"starting", "running"} and process_alive(pid):
            raise RuntimeError(f"scan worker {pid} is still alive; refusing duplicate queue: {path.name}")


def queue_counts(finished):
    return {"completed": sum(row["status"] in SUCCESS_STATUSES for row in finished),
        "forecast_completed": sum(row["status"] == "forecast_completed" for row in finished),
        "capacity_rejected": sum(row["status"] == "capacity_rejected" for row in finished),
        "forecast_rejected": sum(row["status"] == "forecast_rejected" for row in finished),
        "failed": sum(row["status"] == "failed" for row in finished)}


def retry_group_counts(finished, identifiers):
    rows = {row["point_id"]: row for row in finished if row.get("point_id") in identifiers}
    complete = len(rows) == len(identifiers)
    accepted = all(row.get("status") == "forecast_completed"
        and row.get("physical_weight_semantics") == PHYSICAL_WEIGHT_SEMANTICS
        and row.get("approximation", {}).get("paired_budget_decision", {}).get("accepted") is True
        for row in rows.values())
    return {"retry_priority_points": list(identifiers), "retry_group_total": len(identifiers),
        "retry_group_completed": len(rows), "retry_group_accepted": accepted if complete and identifiers else None}


def validate_retry_selection(output_dir, manifest, identifiers, simulation_mode):
    validate_physical_weight_semantics(manifest, "retry manifest")
    known = {point["point_id"]: point for point in manifest["points"]}
    if len(set(identifiers)) != len(identifiers) or any(identifier not in known for identifier in identifiers):
        raise ValueError("retry point IDs must be unique members of the same scan")
    if identifiers and simulation_mode != "fast":
        raise ValueError("priority revalidation requires fast mode")
    if identifiers and (output_dir / "stop.request").exists():
        raise ValueError("a stop marker still pauses this queue; do not clear a user pause implicitly")
    # Validate the entire explicit retry selection before changing any files.
    for identifier in identifiers:
        path = output_dir / "points" / (identifier + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            validate_physical_weight_semantics(saved, "retry point " + identifier)
            if saved.get("status") not in REUSABLE_STATUSES | {"interrupted"}:
                raise ValueError("retry selection contains an execution failure: " + identifier)
            if saved.get("status") == "capacity_rejected":
                raise ValueError("forecast-budget retry cannot retry a capacity rejection: " + identifier)
            if saved.get("status") in SUCCESS_STATUSES and saved.get("approximation", {}).get("paired_budget_decision", {}).get("accepted") is not True:
                raise ValueError("an already successful point must not be rerun: " + identifier)
        elif not any((output_dir / "points_history").glob(identifier + "__*.json")):
            raise ValueError("retry selection has no rejected attempt to preserve: " + identifier)


def run_queue(output_dir, workers=1, retry_failed=False, simulation_mode="fast", retry_point_ids=()):
    from b200_hbm_hbf_scan_report import write_report, refresh_markdown
    if workers not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    retry_point_ids = tuple(retry_point_ids)
    if retry_point_ids:
        path = output_dir / "manifest.json"
        if not path.exists():
            raise ValueError("priority revalidation requires the original scan manifest")
        validate_retry_selection(output_dir, json.loads(path.read_text(encoding="utf-8")), retry_point_ids, simulation_mode)
    manifest = initialize(output_dir, simulation_mode)
    validate_physical_weight_semantics(manifest, "queue manifest")
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
        assert_no_live_workers(output_dir)
        validate_retry_selection(output_dir, manifest, retry_point_ids, simulation_mode)
        # Validate the active set before archiving any explicitly selected
        # forecast refusal. A stale result must never cause a partial migration.
        saved_rows = {}
        for point in manifest["points"]:
            path = output_dir / "points" / (point["point_id"] + ".json")
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                validate_physical_weight_semantics(saved, "active point " + point["point_id"])
                saved_rows[point["point_id"]] = saved
        pending, finished = [], []
        for point in manifest["points"]:
            saved = saved_rows.get(point["point_id"])
            if point["point_id"] in retry_point_ids and saved is not None and saved["status"] == "forecast_rejected":
                archive_rejected_point(output_dir, point, saved)
                saved = None
            if saved is not None and saved["status"] in REUSABLE_STATUSES:
                finished.append(saved)
            elif saved is not None and saved["status"] == "failed" and not retry_failed:
                raise ValueError("failed point needs correction and --retry-failed: " + point["point_id"])
            else:
                pending.append(point)
        priority_order = {identifier: index for index, identifier in enumerate(retry_point_ids)}
        pending.sort(key=lambda point: priority_order.get(point["point_id"], len(priority_order)))
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
                retry_phase = bool(retry_point_ids) and retry_group_counts(finished, retry_point_ids)["retry_group_completed"] < len(retry_point_ids)
                while pending and len(running) < workers:
                    # Keep one job per medium. This avoids two large HBF page
                    # maps occupying the host at once while retaining an A/B pair.
                    occupied = {item[1]["architecture"] for item in running.values()}
                    index = next((i for i, point in enumerate(pending)
                                  if point["architecture"] not in occupied
                                  and (not retry_phase or point["point_id"] in retry_point_ids)), None)
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
                        "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS,
                        "reason": f"worker exited {code} without a complete point result"}
                    validate_physical_weight_semantics(saved, "finished point " + identifier)
                    if not path.exists():
                        write_json(path, saved)
                    finished.append(saved)
                    changed = True
                    if saved["status"] == "failed":
                        stop_reason = "point_failed"
                    elif saved["status"] == "interrupted":
                        pending.insert(0, point)
                        stop_reason = "worker_interrupted"
                retry_status = retry_group_counts(finished, retry_point_ids)
                if retry_status["retry_group_accepted"] is False and not stop_reason:
                    stop_reason = "retry_group_not_accepted"
                write_json(output_dir / "status.json", {"status": "needs_attention" if stop_reason else "running",
                    "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS,
                    "manager_pid": os.getpid(), "updated_at": utc_now(), "total": len(manifest["points"]),
                    "simulation_mode": simulation_mode, **queue_counts(finished),
                    **retry_status,
                    "pending": len(pending), "running": [{**point, "pid": process.pid, "started_at": start,
                        "simulation_mode": simulation_mode}
                        for process, point, _log, start in running.values()],
                    "reason": stop_reason or ("priority_revalidation" if retry_phase else None)})
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
            if (running or stop_reason == "retry_group_not_accepted") and not (output_dir / "stop.request").exists():
                (output_dir / "stop.request").write_text(stop_reason or "manager_interrupted", encoding="utf-8")
            deadline = time.monotonic() + 55
            while any(process.poll() is None for process, *_ in running.values()) and time.monotonic() < deadline:
                # Cooperative cancellation reaches the actual interpreter;
                # terminating only the Windows launcher can orphan a worker.
                time.sleep(.2)
            still_running = []
            for process, point, log, start in running.values():
                log.close()
                if process.poll() is None:
                    still_running.append({**point, "pid": process.pid, "started_at": start,
                                          "simulation_mode": simulation_mode})
                else:
                    path = output_dir / "points" / (point["point_id"] + ".json")
                    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
                    if saved and saved["status"] in REUSABLE_STATUSES | {"failed"}:
                        finished.append(saved)
                    else:
                        pending.append(point)
            queue_state = ("paused" if stop_reason == "stop_requested" else
                           "needs_attention" if stop_reason else "completed")
            if still_running:
                queue_state = "needs_attention"
                stop_reason = "workers_still_stopping"
            elif queue_state == "completed" and any(row["status"] not in SUCCESS_STATUSES for row in finished):
                queue_state = "completed_with_rejections"
            write_json(output_dir / "status.json", {"status": queue_state,
                "physical_weight_semantics": PHYSICAL_WEIGHT_SEMANTICS,
                "manager_pid": os.getpid(), "updated_at": utc_now(), "total": len(manifest["points"]),
                "simulation_mode": simulation_mode, **queue_counts(finished),
                **retry_group_counts(finished, retry_point_ids),
                "pending": len(pending), "running": still_running, "reason": stop_reason})
            write_report(output_dir)
    return 1 if stop_reason else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "report", "stop"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--simulation-mode", choices=("fast", "detailed"), default="fast")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-point-id", action="append", default=[],
                        help="Revalidate selected archived points or forecast refusals first, preserving prior attempts.")
    parser.add_argument("--point-id")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.command == "run":
        return run_queue(output_dir, args.workers, args.retry_failed, args.simulation_mode, args.retry_point_id)
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
