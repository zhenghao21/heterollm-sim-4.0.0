"""Run the bounded 9 + 6 + 15 native-backend comparison, never an old scan.

A candidate is one (DP, TP, PP, EP) layout plus a per-replica batch, executed
for one data-parallel replica through run_scenario in its own subprocess; the
cluster scales that replica's per-GPU throughput by the GPU count. Each
configuration selects, by per-GPU throughput, the best passing candidate among
those actually executed; it is not a global optimum. Both modes execute
through run_scenario. Explicit fast mode keeps complete native cohort samples
separate from a conditional forecast; it never runs a rejected forecast's
detailed suffix or an old detailed reference.

``--search`` places the candidates by the paper-style deployment search
(exploring_hbf_native_search); otherwise every case runs one fixed layout at
fixed batches, for control points.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import csv
import functools
import io
import json
import math
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
DEFAULT_DIRECTORY = ROOT / "artifacts/exploring_hbf_native_20261022"
# The paused v15 scan keeps its manager.lock/stop.request; no v16 process may
# open them.
RETIRED_DIRECTORIES = (ROOT / "artifacts/exploring_hbf_native_20261019",)
CONTRACT = "native_continuous_refill_layout_detailed_v16"
FAST_CONTRACT = "native_stationary_layout_fast_v16"
BACKEND = "native_serving_physical"
PHASES = ("dense_native", "dense_plus", "maverick_native")
TERMINAL = {"completed", "capacity_rejected", "slo_rejected", "all_candidates_rejected", "failed"}
CANDIDATE_TERMINAL = {"completed", "capacity_rejected", "slo_rejected", "forecast_rejected", "failed"}
SLO_NS = 100_000_000.
# Wall caps per concurrent worker: a candidate started while k workers run
# (itself included) gets k times them, as concurrent workers share the machine.
FAST_CANDIDATE_WALL_TARGET_SECONDS = 600.
FAST_CANDIDATE_WALL_BUDGET_SECONDS = 1800.
# One sampled Prefill pass per model, prompt and layout, outside every
# candidate cap; the detailed full pass has none.
PREFILL_WALL_BUDGET_SECONDS = 1800.
MAX_WORKERS = os.cpu_count() or 1
# Physical memory kept free beside the predicted peaks of the running and the
# next candidate; a candidate without a measured peak of its model is
# predicted at DEFAULT_CANDIDATE_PEAK_BYTES.
WORKER_MEMORY_RESERVE_BYTES = 8*1024**3
DEFAULT_CANDIDATE_PEAK_BYTES = 16*1024**3
STATUS_LABELS = {"pending": "等待", "running": "运行中", "completed": "完成",
    "completed_with_rejections": "结束（含拒绝）", "all_candidates_rejected": "候选均被拒绝",
    "capacity_rejected": "容量拒绝", "slo_rejected": "TPOT约束拒绝", "forecast_rejected": "预测预算拒绝", "paused": "暂停", "failed": "失败"}


class ManagerConflict(RuntimeError):
    """Another owner must be inspected; do not mutate its report or stop flag."""


from heterollm_sim.execution_control import NativeFastWallBudgetError
import exploring_hbf_native_search as search_plan

SEARCH_TOP_LAYOUTS = search_plan.TOP_LAYOUTS
SEARCH_LABELS = {"slo_accepted": "插值到100ms并通过验收", "acceptance_failed_fallback": "验收未过，退回已测通过点",
    "native_wall_limited": "受仿真开销限制，未到100ms", "native_capacity_limited": "受原生容量限制，未到100ms",
    "native_memory_limited": "受仿真内存限制，未到100ms",
    "points_exhausted": "批量点用尽，未到100ms", "best_measured_point": "已测最佳通过点"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value) if not field.name.startswith("_")}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__ + " is not JSON serializable")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        for attempt in range(10):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.05)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                default=json_default) + "\n")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def positive_batches(values):
    if not values or any(type(value) is not int or value < 1 for value in values):
        raise ValueError("batch sizes must be positive integers")
    return sorted(set(values))


def selected_cases(case_ids=None):
    """The case matrix, or the requested cases of it in matrix order."""
    cases = case_matrix()
    if case_ids is None:
        return cases
    unknown = sorted(set(case_ids) - {case["case_id"] for case in cases})
    if unknown or not case_ids:
        raise ValueError("unknown or empty case selection: " + ", ".join(unknown))
    return [case for case in cases if case["case_id"] in set(case_ids)]


def case_matrix():
    from exploring_hbf_paper_config import workload_catalog
    rows = []
    for phase, model, hardware in (
        ("dense_native", "llama3_405b", ("HBM4", "CONV", "HBF")),
        ("dense_plus", "llama3_405b", ("CONV+", "HBF+")),
        ("maverick_native", "llama4_maverick", ("HBM4", "CONV", "HBF", "CONV+", "HBF+")),
    ):
        for identity in hardware:
            for workload in workload_catalog():
                rows.append({"case_id": f"native__{model}__{identity.replace('+', '_plus')}__{workload.workload_id}__G8",
                    "phase": phase, "model": model, "hardware": identity, "workload": workload.workload_id,
                    "input_tokens": workload.input_tokens, "output_tokens": workload.output_tokens,
                    "gpu_count": 8, "weight_dtype": "bf16", "kv_dtype": "bf16"})
    return rows


LAYOUT_KEYS = ("dp", "tp", "pp", "ep")


def make_layout(dp, tp, pp, ep):
    """A data-parallel replica count and the replica's parallel degrees."""
    values = dict(zip(LAYOUT_KEYS, (dp, tp, pp, ep)))
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError("dp, tp, pp and ep must be positive integers")
    return values


CONTROL_LAYOUT = make_layout(1, 8, 1, 1)


def layout_label(layout):
    return "DP{dp}_TP{tp}_PP{pp}_EP{ep}".format(**layout)


def replica_gpus(layout):
    return layout["tp"] * layout["pp"] * layout["ep"]


def check_layout(case, layout):
    """The replicas must fill the case's GPUs exactly; DP replicas are independent copies."""
    if layout["dp"] * replica_gpus(layout) != case["gpu_count"]:
        raise ValueError(f"layout {layout_label(layout)} does not use exactly {case['gpu_count']} GPUs")


@functools.lru_cache(maxsize=None)
def _model_layers(model_id):
    from exploring_hbf_native_models import build_native_paper_model
    from heterollm_sim.ir import model_graph_execution_layers
    return model_graph_execution_layers(build_native_paper_model(model_id).graph)


def layout_rejection(case, layout, batch):
    """Why this candidate cannot be built (EP split of the batch or experts), else None."""
    from exploring_hbf_native_config import validate_native_layout
    try:
        check_layout(case, layout)
        validate_native_layout(_model_layers(case["model"]), batch, layout["ep"])
    except ValueError as error:
        return str(error)
    return None


def candidate_id(case, layout, batch):
    return case["case_id"] + f"__{layout_label(layout)}__B{batch}"


def check_grid(cases, layout, batches):
    """Every (case, batch) must be buildable at this layout, else nothing is started."""
    for case in cases:
        for batch in batches:
            reason = layout_rejection(case, layout, batch)
            if reason:
                raise ValueError(f"{candidate_id(case, layout, batch)} cannot be built: {reason}")


def params_layout(params):
    return make_layout(*(params[key] for key in LAYOUT_KEYS))


def params_candidate_id(params):
    return candidate_id(params, params_layout(params), params["batch_size"])


def runner_contract(mode):
    if mode not in {"detailed", "fast"}:
        raise ValueError("mode must be detailed or fast")
    return FAST_CONTRACT if mode == "fast" else CONTRACT


def parameters(case, layout, batch, mode="detailed"):
    check_layout(case, layout)
    return {**case, **layout, "batch_size": batch, "runner_contract": runner_contract(mode),
            **({"simulation_mode": "fast",
                "candidate_wall_budget_seconds_per_worker": FAST_CANDIDATE_WALL_BUDGET_SECONDS,
                "candidate_wall_target_seconds_per_worker": FAST_CANDIDATE_WALL_TARGET_SECONDS,
                "target_relative_error": .10} if mode == "fast" else {}),
            "warmup_decode_cohorts": case["output_tokens"] - 1,
            "measurement_decode_cohorts": 2 * (case["output_tokens"] - 1),
            "slo_tpot_ns": SLO_NS}


def pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Unknown/access denied is conservatively live.
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _try_lock(handle):
    """Lock an open file's first byte without waiting; OSError if another process holds it."""
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ManagerLock:
    def __init__(self, directory, mode="detailed"):
        self.path = Path(directory) / "manager.lock"
        self.handle = None
        self.contract = runner_contract(mode)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            _try_lock(handle)
        except OSError as error:
            handle.close()
            raise ManagerConflict("native comparison manager lock is live") from error
        self.handle = handle
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "started_at": utc_now(), "contract": self.contract}, handle)
        handle.flush()
        return self

    def __exit__(self, *unused):
        # OS locks are released after a crash. Never delete/recreate the lock
        # path: stale-file deletion can race and admit two new managers.
        handle = self.handle
        if handle is not None:
            _unlock(handle)
            handle.close()
            self.handle = None


class ProgressWriter:
    """Progress of one candidate subprocess, keyed by its candidate id."""

    def __init__(self, directory, identity):
        self.path = Path(directory) / "progress" / (identity + ".json")
        self.started = time.perf_counter()
        self.state = {"candidate_id": identity, "pid": os.getpid(), "state": "running", "stage": "starting"}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.candidate_started = self.started
        self.phase_started = self.started
        self.phase_seconds = {}
        self.queue = SimpleQueue()

    def _snapshot_locked(self):
        now = time.perf_counter()
        phases = dict(self.phase_seconds)
        phase = self.state.get("stage", "starting")
        phases[phase] = phases.get(phase, 0.) + now - self.phase_started
        usage = process_memory_bytes()
        return {**self.state, "updated_at": utc_now(), "working_set_bytes": usage[0] if usage else None,
            "wall_seconds": now - self.started,
            "candidate_wall_seconds": now - self.candidate_started,
            "wall_breakdown_seconds": phases,
            "wall_breakdown_scope": "exclusive wall intervals between native progress stage transitions; not simulated hardware latency"}

    def begin_candidate(self, params):
        with self.lock:
            self._drain_queue_locked()
            self.candidate_started = self.phase_started = time.perf_counter()
            self.phase_seconds = {}
            self.state = {"case_id": params["case_id"], "pid": os.getpid(), "state": "running",
                "stage": "starting", "batch_size": params["batch_size"],
                "layout": params_layout(params), "candidate_id": params_candidate_id(params),
                "candidate_started_at": utc_now(),
                "cumulative_progress": {"cohorts": 0, "prefill_cohorts": 0, "decode_cohorts": 0,
                    "prefill_compute_tokens": 0, "visible_output_tokens": 0,
                    "measurement_visible_output_tokens": 0,
                    "measurement_decode_cohorts": 0, "warmup_decode_cohorts": 0,
                    "target_decode_cohorts": params["warmup_decode_cohorts"] + params["measurement_decode_cohorts"]},
                "current_cohort": None}

    def snapshot(self):
        with self.lock:
            self._drain_queue_locked()
            return self._snapshot_locked()

    def publish(self):
        with self.lock:
            self._drain_queue_locked()
            write_json(self.path, self._snapshot_locked())

    def report(self, progress):
        with self.lock:
            self._drain_queue_locked()
            self._report_locked(progress)

    def _report_locked(self, progress):
        if self.state.get("stage") != progress.stage:
            emitted = progress.emitted_monotonic
            now = time.perf_counter() if emitted is None else emitted
            previous = self.state.get("stage", "starting")
            self.phase_seconds[previous] = self.phase_seconds.get(previous, 0.) + now - self.phase_started
            self.phase_started = now
        self.state.update(stage=progress.stage, completed=progress.completed, total=progress.total,
            simulated_time_ns=progress.simulated_time_ns, message=progress.message,
            last_progress_at=utc_now(), metadata=dict(progress.metadata))
        for key in ("cumulative_progress", "current_cohort", "physical_layer_progress"):
            if key in progress.metadata:
                value = progress.metadata[key]
                self.state[key] = dict(value) if isinstance(value, dict) else value
        if progress.stage == "cohort_tasks":
            current = dict(self.state.get("current_cohort") or {})
            current.update(completed_tasks=progress.completed, total_tasks=progress.total)
            self.state["current_cohort"] = current
        elif progress.stage == "cohort_preview_tasks":
            current = dict(self.state.get("current_cohort") or {})
            current.update(preview_completed_tasks=progress.completed, preview_total_tasks=progress.total)
            self.state["current_cohort"] = current

    def _drain_queue_locked(self):
        while True:
            try:
                progress = self.queue.get_nowait()
            except Empty:
                return
            self._report_locked(progress)

    def _heartbeat(self):
        next_publish = time.perf_counter() + 5.
        while not self.stop.wait(.1):
            with self.lock:
                self._drain_queue_locked()
            if time.perf_counter() >= next_publish:
                self.publish()
                next_publish = time.perf_counter() + 5.

    def __enter__(self):
        self.publish()
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()
        self.publish()


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summary_row(row):
    """The candidate row without its physical layer records and boundary ledgers.

    Managers, the search replay, reports and points read only this; the full
    row stays in candidates/.
    """
    summary = {key: value for key, value in row.items()
               if key not in {"physical_layer_samples", "physical_boundary_accounting"}}
    samples = row.get("physical_layer_samples")
    if samples is not None:
        summary["physical_layer_summary"] = {"completed_windows": len(samples),
            "executed_tasks": sum(sample.get("executed_task_count", 0) for sample in samples),
            "complete_cohorts": 0, "visible_output_tokens": 0}
    if isinstance(row.get("approximation"), dict):
        summary["approximation"] = {key: value for key, value in row["approximation"].items()
                                    if key not in {"physical_boundary_accounting", "physical_ingress_accounting"}}
    return summary


def save_candidate(directory, row):
    directory = Path(directory)
    write_json(directory / "candidates" / (row["candidate_id"] + ".json"), row)
    write_json(directory / "summaries" / (row["candidate_id"] + ".json"), summary_row(row))


def read_candidate(directory, identity):
    """A candidate's summary row, remade from its full row when that is newer."""
    directory = Path(directory)
    full, summary = directory / "candidates" / (identity + ".json"), directory / "summaries" / (identity + ".json")
    try:
        stale = full.stat().st_mtime_ns > summary.stat().st_mtime_ns
    except FileNotFoundError:
        stale = full.exists()
    if stale:
        row = summary_row(read_json(full))
        write_json(summary, row)
        return row
    return read_json(summary)


def replica_scaling(replica_tps, params):
    """Per-GPU and cluster throughput of one simulated replica.

    The simulated scenario is one DP replica of tp*pp*ep GPUs; DP replicas are
    independent copies with their own PD ingress, so the cluster is the
    per-GPU rate times every GPU of the case.
    """
    layout = params_layout(params)
    if layout["dp"] * replica_gpus(layout) != params["gpu_count"]:
        raise RuntimeError("candidate layout does not fill the case's GPUs")
    per_gpu = replica_tps / replica_gpus(layout)
    return {"replica_tokens_per_second": replica_tps, "tokens_per_second_per_gpu": per_gpu,
            "cluster_tokens_per_second": per_gpu * params["gpu_count"]}


def measure_result(result, params):
    """Validate the native (start,end] window against real token events."""
    serving = result.serving
    window = dict(serving.decode_measurement)
    start, end = window.get("start_ns"), window.get("end_ns")
    if not window.get("complete") or not all(_numeric(value) for value in (start, end)) or end <= start:
        raise RuntimeError("native Decode measurement did not complete a positive physical window")
    if window.get("measured_decode_cohorts") != params["measurement_decode_cohorts"]:
        raise RuntimeError("native Decode measurement cohort count differs from the authored window")
    if (not window.get("batch_sizes") or any(size != params["batch_size"] for size in window["batch_sizes"])):
        raise RuntimeError("native Decode measurement did not retain the authored full batch")
    ingress = serving.external_kv_ingress_metrics
    if (ingress.get("protocol") != "native_external_kv_v1" or ingress.get("imported_requests", 0) < 1
            or ingress.get("imported_logical_bytes", 0) <= 0):
        raise RuntimeError("native external KV ingress has no real imported request/bytes")
    source_requests = {item.request_id: item for item in result.scenario.workload.requests}
    import_starts = {event.request_id: float(event.timestamp_ns) for event in serving.events
                     if event.event_type == "external_kv_import_start"}
    token_times = {}
    visible = 0
    for event in serving.events:
        if event.event_type != "tokens_committed":
            continue
        count = event.details.get("visible_tokens", 0)
        if type(count) is not int or count < 0:
            raise RuntimeError("native visible-token event has invalid token count")
        token_times.setdefault(event.request_id, []).extend([float(event.timestamp_ns)] * count)
        if start < event.timestamp_ns <= end:
            visible += count
    samples, ttfts = {}, []
    for request_id, metric in serving.request_metrics.items():
        status = str(getattr(metric.status, "value", metric.status)).lower()
        if status in {"rejected", "failed"}:
            raise RuntimeError(f"native request {request_id} {status}: {metric.rejection_reason}")
        spec = source_requests[request_id]
        times = token_times.get(request_id, [])
        if (status == "finished" and not spec.metadata.get("warmup_seed", False)
                and len(times) == params["output_tokens"] and times
                and start < times[0] <= times[-1] <= end):
            samples[request_id] = (times[-1] - times[0]) / (len(times) - 1)
            if request_id not in import_starts:
                raise RuntimeError(f"native fresh request {request_id} has no paid KV import start")
            ttfts.append(times[0] - import_starts[request_id])
    if not samples or visible <= 0:
        raise RuntimeError("native measurement has no complete fresh full-output request or no real visible tokens")
    native_samples = window.get("complete_request_tpot_ns", {})
    if set(native_samples) != set(samples) or any(not math.isclose(value, native_samples[key], rel_tol=1e-12, abs_tol=1e-3)
                                                for key, value in samples.items()):
        raise RuntimeError("native complete-request TPOT statistics disagree with token events")
    if window.get("visible_output_tokens") != visible:
        raise RuntimeError("native measurement visible-token total disagrees with token events")
    duration = end - start
    mean_tpot = statistics.mean(samples.values())
    metrics = {**replica_scaling(visible * 1e9 / duration, params),
        "mean_complete_request_tpot_ns": mean_tpot, "max_complete_request_tpot_ns": max(samples.values()),
        "mean_complete_request_ingress_ttft_ns": statistics.mean(ttfts),
        "complete_fresh_requests": len(samples), "measurement_visible_output_tokens": visible,
        "measurement_duration_ns": duration, "mean_iteration_ns": window.get("mean_iteration_ns"),
        "max_iteration_ns": window.get("max_iteration_ns")}
    return metrics, {"window": window, "complete_request_tpot_ns": samples,
                     "slo_tpot_ns": SLO_NS, "slo_statistic": "unweighted_mean_complete_fresh_request_tpot",
                     "slo_met": mean_tpot <= SLO_NS,
                     "unfinished_requests_at_measurement_stop": sum(str(getattr(row.status, "value", row.status)).lower()
                        not in {"finished", "rejected", "failed"} for row in serving.request_metrics.values())}


def traffic_result(result):
    from heterollm_sim.reporting import _sum_batch_dram_traffic, _sum_batch_storage_traffic
    ingress = dict(result.serving.external_kv_ingress_metrics)
    ledger = {key: value for key, value in ingress.get("cost_ledger", {}).items() if key != "phases"}
    ingress["cost_ledger"] = ledger
    sram_ledgers = [batch.cost.metadata["sram_traffic"] for batch in result.serving.batches
                    if "sram_traffic" in batch.cost.metadata]
    has_sram = any(item.normalized_kind in {"sram", "shared_memory"} for item in result.scenario.hardware.components)
    fields = ("logical_read_bytes", "logical_write_bytes", "physical_read_bytes", "physical_write_bytes", "physical_bytes")
    sram = {key: (sum(row[key] for row in sram_ledgers)
                  if sram_ledgers and all(_numeric(row.get(key)) for row in sram_ledgers)
                  else None if has_sram else 0) for key in fields}
    dram, nand = _sum_batch_dram_traffic(result), _sum_batch_storage_traffic(result)
    combined = {}
    for kind, batch, pd in (("DRAM", dram, ledger.get("dram_traffic", {})),
                            ("NAND", nand, ledger.get("storage_traffic", {})),
                            ("SRAM", sram, {})):
        combined[kind] = {key: (batch.get(key, 0) + pd.get(key, 0)
            if _numeric(batch.get(key, 0)) and _numeric(pd.get(key, 0)) else None) for key in fields}
    return {"scope": "whole bounded native run including warmup and submitted final/cancel work; not measurement-window-only; phase time spans are not exclusive costs",
            "batch_and_kv_migration": {"DRAM": dram, "NAND": nand, "SRAM": sram},
            "external_kv_ingress": ingress, "combined_physical_traffic": combined,
            "accounting": "external ingress is added once to the separate native batch/KV-migration ledgers"}


CAPACITY_MARKERS = ("exceeds available capacity", "exceeds physical capacity", "exceeds effective capacity",
    "insufficient capacity", "超过有效容量", "超出可用容量", "内存容量不足")


def _capacity_diagnostic(row):
    text = (str(row.get("message_en", "")) + " " + str(row.get("message_zh", ""))).lower()
    if any(marker in text for marker in CAPACITY_MARKERS):
        return True
    # The control plane reports a tensor that the total capacity constraint
    # left without any component as an unplaced item at the capacity check.
    return (row.get("code") == "runtime_placement_unplaced" and row.get("stage") == "capacity_check"
            and "总容量约束" in str(row.get("message_zh", "")))


def capacity_error(error):
    from heterollm_sim.memory_allocator import AllocationError
    if isinstance(error, AllocationError):
        # The allocator also uses this exception for invalid aliases,
        # generations and overlapping addresses. Only allocation exhaustion
        # is a legitimate capacity rejection; correctness errors must stop.
        return str(error).startswith("unable to allocate ") and " bytes in capacity " in str(error)
    details = getattr(error, "details", {})
    diagnostics = details.get("diagnostics", []) if isinstance(details, dict) else []
    return bool(diagnostics) and all(_capacity_diagnostic(row) for row in diagnostics)


def native_fast_assumptions(scenario):
    from exploring_hbf_native_config import PHYSICAL_WINDOW_ASSUMPTION
    from heterollm_sim.native_stationary import SAME_CLASS_ADDRESS_ASSUMPTION
    scope = ("Native layer runs on independent copies of the first continuous-refill boundary are physical; "
             "the stationary step, PD-import extra, refill schedule, cohorts and visible Tokens are separate conditional predictions.")
    return tuple(dict.fromkeys((*[value for value in scenario.assumptions if value != PHYSICAL_WINDOW_ASSUMPTION],
        scope, SAME_CLASS_ADDRESS_ASSUMPTION, "Each point's independent error budget retains the 10% target; "
        "conditional budgets are not full-window detailed validation or silicon/paper accuracy guarantees.")))


def dp_scaled_bounds(bounds, params):
    """Replicate a single replica's cluster-rate interval over the DP copies."""
    bounds = dict(bounds)
    row = bounds.get("cluster_tokens_per_second")
    if row:
        bounds["cluster_tokens_per_second"] = {key: value * params["dp"]
            if key in {"estimate", "lower", "upper"} and _numeric(value) else value for key, value in row.items()}
    return bounds


def native_fast_result(result, params):
    """Keep the physical prefix and the unexecuted projection separate.

    The forecast describes one replica; row metrics and intervals are scaled to
    the cluster, the saved forecast metadata stays the single-replica value.
    """
    from heterollm_sim.reporting import FastScenarioResult
    if not isinstance(result, FastScenarioResult) or result.detailed is not None:
        raise RuntimeError("native fast candidate did not return a separate sampled/forecast result")
    forecast = result.native_window_forecast
    if result.sampled is not None:
        raise RuntimeError("native fast sampling cannot publish a partial cohort")
    details = {"approximation": dict(result.approximation),
        "physical_layer_samples": [sample.to_metadata() for sample in result.layer_window_samples],
        "physical_boundary_accounting": result.approximation.get("physical_boundary_accounting"),
        "traffic": None, "assumptions": list(native_fast_assumptions(result.scenario)),
        "replica_scaling": {"dp": params["dp"], "gpus_per_replica": replica_gpus(params_layout(params)),
            "scope": "native_window_forecast and approximation hold the single simulated replica; metrics and metric intervals below are scaled to the cluster"}}
    if forecast is None:
        rejected = {"status": "forecast_rejected", "metrics": None,
            "rejection_reason": ";".join(result.approximation.get("rejection_reasons", ()))
                or "native_window_projection_unavailable"}
        bounds = result.approximation.get("metric_bounds")
        if bounds:
            # A budget-rejected point still reports its conditional intervals,
            # marked as missing the target; they never enter selection.
            rejected["rejected_metric_bounds"] = {"target_relative_error": params["target_relative_error"],
                "target_met": False, "label": "未达到10%目标的条件区间，不是通过验收的结果",
                "metrics": dp_scaled_bounds(bounds, params)}
        return {**details, **rejected}
    bounds = forecast.metric_bounds.get("mean_complete_request_tpot_ns", {})
    upper = bounds.get("upper")
    if not _numeric(upper) or upper <= 0:
        raise RuntimeError("accepted native fast forecast has no positive TPOT upper bound")
    metrics = {**forecast.metrics, **replica_scaling(forecast.metrics["cluster_tokens_per_second"], params)}
    return {**details, "status": "completed" if upper <= SLO_NS else "slo_rejected",
        "metrics": metrics if upper <= SLO_NS else None,
        "native_window_forecast": forecast.to_metadata(),
        "forecast_tokens": forecast.predicted_visible_output_tokens,
        "sampled_tokens": forecast.sampled_visible_output_tokens,
        "diagnostics": {"observed_metrics": metrics, "metric_bounds": dp_scaled_bounds(forecast.metric_bounds, params),
            "slo_met": upper <= SLO_NS, "slo_tpot_ns": SLO_NS,
            "slo_statistic": "conditional_upper_bound_of_mean_complete_fresh_request_tpot"},
        **({"rejection_reason": "mean_complete_fresh_request_tpot_upper_bound_exceeds_100ms"}
            if upper > SLO_NS else {})}


def native_capacity_max_batch(row, params, scenario):
    """Largest batch the native KV holds, from the reservations it held when it refused one; None if unknown."""
    from exploring_hbf_native_config import native_capacity_batch
    held = row["approximation"].get("kv_requests_held_by_group")
    if not held:
        return None
    batch = native_capacity_batch(min(held), params["batch_size"],
        scenario.workload.metadata["external_kv_ingress"]["max_inflight_requests"],
        params["output_tokens"], params["ep"])
    return batch if batch < params["batch_size"] else None


def prefill_parameters(params):
    """The Prefill replica's identity: model, prompt and layout, not the memory or the batch."""
    fast = params.get("simulation_mode") == "fast"
    return {"model": params["model"], "workload": params["workload"], "input_tokens": params["input_tokens"],
            **{key: params[key] for key in ("tp", "pp", "ep")}, "prefill_hardware": "HBM4",
            "pass": "sampled" if fast else "full", "runner_contract": params["runner_contract"],
            **({"target_relative_error": params["target_relative_error"]} if fast else {})}


def prefill_id(expected):
    return (f"prefill__{expected['model']}__{expected['workload']}__TP{expected['tp']}PP{expected['pp']}"
            f"EP{expected['ep']}__{expected['pass']}")


@contextlib.contextmanager
def waiting_lock(path, control):
    """Hold ``path`` exclusively, waiting cancellably while another worker holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        while True:
            try:
                _try_lock(handle)
                break
            except OSError:
                control.raise_if_cancelled()
                time.sleep(1.)
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()


def prefill_pass(params, directory, *, builder=None, runner=None, wall_factor=1):
    """The candidate's native Prefill pass, run once per run directory.

    Every memory configuration and batch of one model, prompt and layout
    shares it: the Prefill replica is HBM4 with the Decode replica's layout.
    Fast candidates take the layer-sampled pass under the same independent
    budget and its own wall cap (per concurrent worker, like the candidate's,
    but outside it); detailed ones one native run of every layer, without a
    wall cap like the detailed run.
    """
    from heterollm_sim.execution_control import ExecutionControl
    from exploring_hbf_native_config import build_native_prefill_scenario
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.simulation_options import NativePrefillPassConfig, SimulationConfig
    directory = Path(directory)
    expected = prefill_parameters(params)
    identity = prefill_id(expected)
    path = directory / "prefill" / (identity + ".json")
    full = expected["pass"] == "full"
    cap = PREFILL_WALL_BUDGET_SECONDS * wall_factor
    started = time.perf_counter()
    control = ExecutionControl(stop_request_path=directory / "stop.request",
                               deadline_monotonic=None if full else started + cap)
    with waiting_lock(path.with_suffix(".lock"), control):
        saved = read_json(path)
        if saved is not None and saved.get("parameters") == expected and saved.get("backend") == BACKEND:
            return saved
        scenario = (builder or build_native_prefill_scenario)(params["model"], params["workload"],
            tp=params["tp"], pp=params["pp"], ep=params["ep"], full_pass=full)
        scenario = replace(scenario, simulation=SimulationConfig(mode="fast", native_prefill=NativePrefillPassConfig(
            full_pass=full, max_wall_seconds=None if full else cap,
            **({} if full else {"target_relative_error": expected["target_relative_error"]}))))
        result = (runner or run_scenario)(scenario, retention_policy="aggregate", control=control)
        approximation = result.approximation
        row = {"backend": BACKEND, "prefill_id": identity, "parameters": expected,
               "status": "accepted" if approximation["prefill_pass"]["accepted"] else "rejected",
               "prefill_pass": approximation["prefill_pass"],
               "rejection_reasons": list(approximation.get("rejection_reasons", ())),
               "physical_layer_samples": [sample.to_metadata() for sample in result.layer_window_samples],
               "assumptions": list(scenario.assumptions), "finished_at": utc_now(),
               "wall_seconds": time.perf_counter() - started}
        write_json(path, row)
        return row


def with_prefill_ttft(row, params, prefill):
    """Add the Prefill pass and TTFT = Prefill pass + ingress TTFT to a completed candidate.

    Fast candidates add the two intervals; the sum keeps both parts' relative
    budget. Detailed candidates add the observed full pass to the measured
    ingress TTFT. A rejected pass leaves both values unknown.
    """
    from heterollm_sim.native_stationary import _interval, _within
    pass_ = prefill["prefill_pass"]
    metrics = row["metrics"]
    row["prefill_pass"] = {"prefill_id": prefill["prefill_id"], "status": prefill["status"],
        "metric_bounds": pass_["metric_bounds"], "rejection_reasons": prefill["rejection_reasons"],
        "wall_seconds": prefill["wall_seconds"]}
    metrics.update(prefill_pass_ns=None, mean_complete_request_ttft_ns=None)
    if prefill["status"] != "accepted":
        return
    if params.get("simulation_mode") == "fast":
        bounds = row["diagnostics"]["metric_bounds"]
        ingress, prefill_bounds = bounds["mean_complete_request_ingress_ttft_ns"], pass_["metric_bounds"]["prefill_pass_ns"]
        total = _interval(prefill_bounds["lower"] + ingress["lower"], prefill_bounds["upper"] + ingress["upper"])
        bounds.update(prefill_pass_ns=prefill_bounds, mean_complete_request_ttft_ns=total)
        target = params["target_relative_error"]
        if _within(total, target / (2 + target)):
            metrics.update(prefill_pass_ns=prefill_bounds["estimate"], mean_complete_request_ttft_ns=total["estimate"])
    else:
        value = pass_["error_budget"]["observed_pass_ns"]
        metrics.update(prefill_pass_ns=value,
                       mean_complete_request_ttft_ns=value + metrics["mean_complete_request_ingress_ttft_ns"])


def execute_candidate(params, directory, progress, *, builder=None, runner=None, prefill=None, wall_factor=1):
    """Run one candidate; a fast one within its per-worker caps times ``wall_factor``."""
    from heterollm_sim.execution_control import ExecutionControl, ExecutionCancelledError
    from exploring_hbf_native_config import build_native_comparison_scenario
    from heterollm_sim.reporting import run_scenario
    directory = Path(directory)
    started = time.perf_counter()
    fast = params.get("simulation_mode") == "fast"
    budget = params["candidate_wall_budget_seconds_per_worker"] * wall_factor if fast else None
    row = {"runner_contract": params["runner_contract"], "backend": BACKEND, "parameters": params,
           "candidate_id": params_candidate_id(params), "status": "running",
           "metrics": None, "started_at": utc_now(), "forecast_tokens": 0, "wall_factor": wall_factor,
           **({"candidate_wall_budget_seconds": budget} if fast else {}),
           "fidelity": ("native_stationary_layer_samples_and_separate_conditional_forecast"
                if fast else "complete_bounded_native_physical_window_no_forecast_no_old_reference")}
    try:
        native_progress = type(progress) is ProgressWriter and type(progress.queue) is SimpleQueue
        control = ExecutionControl(
            progress_queue=progress.queue if native_progress else None,
            progress_callback=None if native_progress else progress.report,
            stop_request_path=directory / "stop.request",
            deadline_monotonic=started + budget if fast else None)
        control.raise_if_cancelled()
        scenario = (builder or build_native_comparison_scenario)(params["model"], params["hardware"], params["workload"],
            batch_size=params["batch_size"], tp=params["tp"], pp=params["pp"], ep=params["ep"])
        if fast:
            from heterollm_sim.simulation_options import SimulationConfig, NativeWindowFastConfig
            remaining = budget - (time.perf_counter() - started)
            if remaining <= 0:
                raise NativeFastWallBudgetError("native candidate construction exhausted the wall budget")
            scenario = replace(scenario, simulation=SimulationConfig(mode="fast",
                native_window=NativeWindowFastConfig(max_wall_seconds=remaining)),
                assumptions=native_fast_assumptions(scenario))
        write_json(directory / "configurations" / (row["candidate_id"] + ".json"), scenario)
        control.raise_if_cancelled()
        options = {"retention_policy": "aggregate", "control": control}
        result = (runner or run_scenario)(scenario, **options)
        try:
            control.raise_if_cancelled()
        except NativeFastWallBudgetError:
            # The completed physical layer records belong to this attempt
            # even if its final result arrives after the outer deadline.
            if fast:
                row.update(native_fast_result(result, params))
            raise
        if fast:
            row.update(native_fast_result(result, params))
            if any(marker in row.get("rejection_reason", "") for marker in search_plan.CAPACITY_REJECTIONS):
                row["native_capacity_max_batch"] = native_capacity_max_batch(row, params, scenario)
            if time.perf_counter() - started >= budget:
                row.update(status="forecast_rejected", metrics=None,
                    rejection_reason="native_total_candidate_wall_budget_exhausted")
        else:
            metrics, diagnostics = measure_result(result, params)
            row.update(diagnostics={**diagnostics, "observed_metrics": metrics}, traffic=traffic_result(result),
                       assumptions=list(result.scenario.assumptions))
            if diagnostics["slo_met"]:
                row.update(status="completed", metrics=metrics)
            else:
                row.update(status="slo_rejected", rejection_reason="mean_complete_fresh_request_tpot_exceeds_100ms")
        if row["status"] == "completed":
            prefill_started = time.perf_counter()
            with_prefill_ttft(row, params, (prefill or functools.partial(prefill_pass, wall_factor=wall_factor))(
                params, directory))
            # The search projects a point's Decode wall without the shared Prefill pass.
            row["prefill_wall_seconds"] = time.perf_counter() - prefill_started
    except NativeFastWallBudgetError as error:
        row.update(status="forecast_rejected", metrics=None,
            rejection_reason="native_total_candidate_wall_budget_exhausted", reason=str(error))
    except ExecutionCancelledError as error:
        row.update(status="paused", reason=str(error))
    except Exception as error:
        row.update(status="capacity_rejected" if capacity_error(error) else "failed",
                   error={"type": type(error).__name__, "message": str(error),
                          "details": getattr(error, "details", None), "traceback": traceback.format_exc()})
    row.update(finished_at=utc_now(), wall_seconds=time.perf_counter() - started)
    if hasattr(progress, "snapshot"):
        observation = progress.snapshot()
        row["progress"] = {key: observation.get(key) for key in (
            "cumulative_progress", "current_cohort", "wall_breakdown_seconds", "wall_breakdown_scope")}
    return row


def select_case(case, planned, rows, mode="detailed"):
    """``planned`` lists the candidate ids this case was asked to evaluate."""
    failed = next((row for row in rows if row["status"] == "failed"), None)
    accepted = [row for row in rows if row["status"] == "completed"]
    complete = len(rows) == len(planned) and all(row["status"] in CANDIDATE_TERMINAL for row in rows)
    state = ("failed" if failed else "paused" if any(row["status"] == "paused" for row in rows)
             else "completed" if complete and accepted else "all_candidates_rejected" if complete else "running")
    selected = (max(accepted, key=lambda row: row["metrics"]["tokens_per_second_per_gpu"])
                if state == "completed" else None)
    return {"case": case, "case_id": case["case_id"], "runner_contract": runner_contract(mode), "backend": BACKEND,
        "status": state, "planned_candidate_ids": list(planned), "candidate_ids": [row["candidate_id"] for row in rows],
        "candidate_statuses": {row["candidate_id"]: row["status"] for row in rows},
        "selected": selected, "error": failed.get("error") if failed else None, "updated_at": utc_now(),
        "selection_scope": ("highest conditionally accepted projected per-GPU throughput among the sampled (DP, TP, PP, EP) layout and per-replica batch candidates; not a global optimum or maximum feasible batch"
            if mode == "fast" else "highest measured passing per-GPU throughput among the explicitly executed (DP, TP, PP, EP) layout and per-replica batch candidates; not a global optimum or maximum feasible batch")}


def worker_candidate(case, layout, batch, directory, mode="detailed", wall_factor=1):
    """Run or reuse exactly one candidate in this process; the manager owns the case point."""
    directory = Path(directory)
    params = parameters(case, layout, batch, mode)
    identity = params_candidate_id(params)
    with ProgressWriter(directory, identity) as progress:
        row = read_candidate(directory, identity)
        if row is not None:
            if row.get("parameters") != params or row.get("backend") != BACKEND:
                raise RuntimeError("saved candidate parameters/backend differ; historical results cannot be relabeled")
            if row["status"] not in CANDIDATE_TERMINAL:
                row = None
        if row is None:
            progress.begin_candidate(params)
            progress.publish()
            row = execute_candidate(params, directory, progress, wall_factor=wall_factor)
            usage = process_memory_bytes()
            # The search predicts later candidates' memory from this peak.
            row["peak_memory_bytes"] = usage[1] if usage else None
            save_candidate(directory, row)
        print(json.dumps({"candidate_id": identity, "status": row["status"],
                          "wall_seconds": row.get("wall_seconds")}), flush=True)
        with progress.lock:
            progress.state.update(state=row["status"])
    return 1 if row["status"] == "failed" else 0


def refresh(directory, cases, planned, running=(), *, state="running", phase=None, error=None, mode="detailed",
            search=False):
    """``planned`` maps each case id to the (layout, per-replica batch) candidates it is asked to evaluate."""
    directory = Path(directory)
    points, candidates = [], []
    active_ids = {row["case_id"] for row in running}
    for case in cases:
        grid = planned[case["case_id"]]
        planned_ids = [candidate_id(case, layout, batch) for layout, batch in grid]
        point = read_json(directory / "points" / (case["case_id"] + ".json"))
        if point and (point.get("case") != case or point.get("backend") != BACKEND
                      or point.get("runner_contract") != runner_contract(mode)):
            point = {"case": case, "case_id": case["case_id"], "status": "failed", "selected": None,
                     "error": {"message": "foreign or mismatched historical result is not native evidence"}}
        elif point and not set(planned_ids).issubset(point.get("planned_candidate_ids", [])):
            point = {**point, "status": "running" if case["case_id"] in active_ids else "pending", "selected": None,
                     "previous_candidate_ids": point.get("planned_candidate_ids"), "planned_candidate_ids": planned_ids}
        points.append(point or {"case": case, "case_id": case["case_id"],
            "status": "running" if case["case_id"] in active_ids else "pending", "selected": None})
        for layout, batch in grid:
            expected = parameters(case, layout, batch, mode)
            row = read_candidate(directory, candidate_id(case, layout, batch))
            if row is not None:
                if row.get("parameters") != expected or row.get("backend") != BACKEND:
                    row = {"parameters": expected, "candidate_id": candidate_id(case, layout, batch),
                           "status": "failed", "metrics": None, "error": {
                               "message": "foreign or mismatched candidate excluded from native metrics"}}
                candidates.append(row)
    counts = Counter(point["status"] for point in points)
    layouts = list({layout_label(layout): layout for grid in planned.values() for layout, _batch in grid}.values())
    batches = sorted({batch for grid in planned.values() for _layout, batch in grid})
    status = {"state": state, "updated_at": utc_now(), "manager_pid": os.getpid(), "current_phase": phase,
        "total_cases": len(cases), "total_candidates": sum(len(grid) for grid in planned.values()),
        "counts": dict(counts), "candidate_counts": dict(Counter(row["status"] for row in candidates)),
        "running": list(running), "layouts": layouts, "batch_sizes": batches, "backend": BACKEND,
        "runner_contract": runner_contract(mode), "simulation_mode": mode, "deployment_search": search,
        "stop_requested": (directory / "stop.request").exists(), "error": error}
    write_json(directory / "status.json", status)
    write_json(directory / "results.json", {"status": status, "cases": points, "candidates": candidates})
    output = io.StringIO(newline="")
    interval_names = ("cluster_tokens_per_second", "mean_complete_request_tpot_ns")
    writer = csv.DictWriter(output, fieldnames=["case_id", "model", "hardware", "workload", *LAYOUT_KEYS, "batch_size", "status",
        "tokens_per_second_per_gpu", "cluster_tokens_per_second", "mean_complete_request_tpot_ns",
        "mean_complete_request_ttft_ns", "prefill_pass_ns", "wall_seconds", "reason",
        *(f"unaccepted_{name}_{side}" for name in interval_names for side in ("lower", "upper")),
        "unaccepted_max_relative_error"])
    writer.writeheader()
    for row in candidates:
        p, metrics = row["parameters"], row.get("metrics") or {}
        unaccepted = (row.get("rejected_metric_bounds") or {}).get("metrics") or {}
        writer.writerow({"case_id": p["case_id"], **{key: p[key] for key in ("model", "hardware", "workload", *LAYOUT_KEYS, "batch_size")},
            "status": row["status"], **{key: metrics.get(key) for key in ("tokens_per_second_per_gpu", "cluster_tokens_per_second",
                "mean_complete_request_tpot_ns", "mean_complete_request_ttft_ns", "prefill_pass_ns")},
            "wall_seconds": row.get("wall_seconds"), "reason": row.get("rejection_reason") or (row.get("error") or {}).get("message", ""),
            **{f"unaccepted_{name}_{side}": (unaccepted.get(name) or {}).get(side)
               for name in interval_names for side in ("lower", "upper")},
            "unaccepted_max_relative_error": max((value["relative_error_budget"] for value in unaccepted.values()
                                                  if _numeric(value.get("relative_error_budget"))), default=None)})
    atomic_text(directory / "summary.csv", output.getvalue())
    gpus = "/".join(str(count) for count in sorted({case["gpu_count"] for case in cases}))
    lines = ["# 原生后端论文对照实验", "", f"更新：{status['updated_at']}；状态：{STATUS_LABELS.get(state,state)}。",
        f"配置 {len(cases)} 组，成功 {counts.get('completed',0)}，全候选拒绝 {counts.get('all_candidates_rejected',0)}，失败 {counts.get('failed',0)}。", "",
        (f"每配置固定 {gpus} GPU；部署按论文式搜索：旁路模型筛选布局并给出批量区间，原生在旁路排名前{SEARCH_TOP_LAYOUTS}且原生可组合的布局上各测2–3个批量点，"
            "把TPOT上界线性插值到100ms，再在最佳布局的插值批量上做一次原生验收；最终数字只取原生候选，旁路数值只用于放置批量点。"
            "够不到100ms的布局停在已测最大点并标注受限，不外推；验收未通过时退回已测通过点。"
            if search else
            f"每配置固定 {gpus} GPU；有限候选布局 {', '.join(layout_label(layout) for layout in layouts)}，每副本批量 {', '.join(map(str,batches))}。")
        + f"DP 副本是互相独立的拷贝，只仿真一个副本，集群吞吐为单GPU吞吐乘GPU总数。每配置选择通过 100ms 请求均值TPOT约束的最高{'预算接受的预测' if mode == 'fast' else '实测'}单GPU吞吐，不表示全局最优部署或最大批量。",
        ("完整原生批次样本与预测窗口分开保存；10%为明示条件模型预算，不是未执行全窗口的实测保证。预测尾段没有物理轨迹，预算拒绝时指标为空，不回退详细后缀。"
            if mode == "fast" else "完整原生预热与有界测量窗口，无预测尾段；零预测Token不等于与论文或实物误差为零。历史详细参考和旧扫描均未重跑。"),
        ("预测吞吐使用相同测量窗口及可见Token定义，TPOT使用窗口内完整新请求的首末Token预测区间；预测完成不计为物理完成。"
            if mode == "fast" else "吞吐按原生窗口 (开始,结束] 的真实可见Token计算；TPOT仅使用首末Token都在窗口内的完整新请求。未在窗口结束的请求不计为完成。"),
        ("TTFT = Prefill通路 + 入口TTFT。Prefill副本为HBM4、与Decode副本相同的TP/PP/EP布局，请求到达即单独一次前向计算前P-1个提示Token（不排队、不分块），"
            "到最后一层完成为止（此时开始导出KV）；入口TTFT从该请求实付KV导入开始到首个Token，Decode侧计算最后一个提示Token并给出首Token。"
            + ("Prefill通路为分层采样区间，与入口TTFT区间相加，仍满足同一独立预算。" if mode == "fast"
               else "Prefill通路为一次覆盖全部层的原生运行。")),
        ("DRAM/NAND/SRAM与PD真实账本只覆盖采样前缀；全窗口物理流量未知。预测尾段不生成虚构的流量、利用率或物理状态。"
            if mode == "fast" else "DRAM/NAND/SRAM与PD流量为整个有界运行的真实账本，包含预热和已提交的收尾费用，不是仅测量窗口流量或独占关键路径分解。"), "",
        "累计Token、批次及阶段壁钟见[进度分析](progress_analysis.md)；真实慢路径调查见[原生后端对照说明](../../docs/exploring_hbf_native_comparison.md)。局部事务提速不能代替完整候选耗时验收。", "",
        "| 模型 | 存储 | 输入/输出Token | 配置状态 | 选中布局 | 选中批量/副本 | tokens/s/GPU | 集群tokens/s | 请求均值TPOT(ms) | 请求均值TTFT(ms，含Prefill) |"
        + (" 搜索结果 |" if search else ""),
        "|---|---|---|---|---|---:|---:|---:|---:|---:|" + ("---|" if search else "")]
    for point in points:
        case, selected = point["case"], point.get("selected")
        m = selected["metrics"] if selected else {}
        number = lambda value: f"{value:.3f}" if _numeric(value) else "—"
        model = "Llama 3.1 405B" if case["model"] == "llama3_405b" else "Llama 4 Maverick"
        lines.append(f"| {model} | {case['hardware']} | {case['input_tokens']}/{case['output_tokens']} | {STATUS_LABELS.get(point['status'],point['status'])} | "
            f"{layout_label(params_layout(selected['parameters'])) if selected else '—'} | "
            f"{selected['parameters']['batch_size'] if selected else '—'} | {number(m.get('tokens_per_second_per_gpu'))} | "
            f"{number(m.get('cluster_tokens_per_second'))} | {number(m.get('mean_complete_request_tpot_ns', 0)/1e6) if m else '—'} | "
            f"{number(m['mean_complete_request_ttft_ns']/1e6) if _numeric(m.get('mean_complete_request_ttft_ns')) else '—'} |"
            + (f" {SEARCH_LABELS.get((point.get('search') or {}).get('result'), '—')} |" if search else ""))
    unaccepted = [row for row in candidates if row.get("rejected_metric_bounds")]
    if unaccepted:
        def span(bounds, scale=1.):
            return f"{bounds['lower']/scale:.3f}～{bounds['upper']/scale:.3f}" if bounds else "—"
        lines += ["", "## 预算拒绝候选的条件区间（未达到10%目标）", "",
            "以下区间来自同一条件预算模型，但误差预算超出10%目标对应的门槛：不是通过验收的结果，不参与选优，也不代表实测值。", "",
            "| 模型 | 存储 | 输入/输出Token | 布局 | 批量/副本 | 集群tokens/s区间 | 请求均值TPOT(ms)区间 | 入口TTFT(ms)区间（不含Prefill） | 最大相对误差 |",
            "|---|---|---|---|---:|---|---|---|---:|"]
        for row in unaccepted:
            p, bounds = row["parameters"], row["rejected_metric_bounds"]["metrics"]
            errors = [value["relative_error_budget"] for value in bounds.values() if _numeric(value.get("relative_error_budget"))]
            error = f"{100*max(errors):.2f}%（未达标）" if errors else "—"
            lines.append(f"| {'Llama 3.1 405B' if p['model'] == 'llama3_405b' else 'Llama 4 Maverick'} | {p['hardware']} | "
                f"{p['input_tokens']}/{p['output_tokens']} | {layout_label(params_layout(p))} | {p['batch_size']} | {span(bounds.get('cluster_tokens_per_second'))} | "
                f"{span(bounds.get('mean_complete_request_tpot_ns'), 1e6)} | {span(bounds.get('mean_complete_request_ingress_ttft_ns'), 1e6)} | "
                f"{error} |")
    if running:
        lines += ["", "正在执行："]
        for job in running:
            p = read_json(directory / "progress" / (job["candidate_id"] + ".json"), {})
            current = p.get("current_cohort") or {}
            cumulative = p.get("cumulative_progress") or {}
            lines.append(f"- {job['candidate_id']}：真实PID {p.get('pid', '等待启动')}，批量 {p.get('batch_size', '—')}，"
                f"阶段 {p.get('message') or p.get('stage', '—')}，候选壁钟 {p.get('candidate_wall_seconds', p.get('wall_seconds', 0)):.1f}秒。")
            lines.append(f"  已完成批次 {cumulative.get('cohorts', '未知')}，实际Prefill Token {cumulative.get('prefill_compute_tokens', '未知')}，"
                f"实际可见输出Token {cumulative.get('visible_output_tokens', '未知')}；Decode批次 "
                f"{cumulative.get('decode_cohorts', '未知')}/{cumulative.get('target_decode_cohorts', '未知')}，"
                f"其中预热 {cumulative.get('warmup_decode_cohorts', '未知')}、测量 {cumulative.get('measurement_decode_cohorts', '未知')}。")
            lines.append(f"  当前批次 {current.get('index', '未知')}，拓扑任务 "
                f"{current.get('completed_tasks', p.get('completed', '未知'))}/{current.get('total_tasks', p.get('total', '未知'))}；"
                "任务比例不代表候选完成比例。")
            phases = p.get("wall_breakdown_seconds") or {}
            if phases:
                from exploring_hbf_native_status import STAGE_LABELS
                lines.append("  原生阶段累计壁钟：" + "，".join(
                    f"{STAGE_LABELS.get(key, '原生内部阶段')} {value:.2f}秒" for key, value in phases.items()) + "。")
    if error:
        lines += ["", "失败原因：" + str(error)]
    atomic_text(directory / "report.md", "\n".join(lines) + "\n")
    return status


def _reusable_point(case, planned, directory, mode="detailed"):
    point = read_json(Path(directory) / "points" / (case["case_id"] + ".json"))
    if not point:
        return None
    if point.get("case") != case or point.get("backend") != BACKEND or point.get("runner_contract") != runner_contract(mode):
        raise RuntimeError("saved case is not the same native comparison; refusing historical reuse")
    if point["status"] in TERMINAL and set(planned).issubset(point.get("planned_candidate_ids", [])):
        return point
    return None


def write_case_point(directory, case, layout, batches, mode="detailed"):
    """Select the case's point from whatever planned candidate rows are saved."""
    directory = Path(directory)
    planned = [candidate_id(case, layout, batch) for batch in batches]
    saved = (read_candidate(directory, identity) for identity in planned)
    point = select_case(case, planned, [row for row in saved if row is not None], mode)
    write_json(directory / "points" / (case["case_id"] + ".json"), point)
    return point


def _memory_status():
    """(total, available) physical memory in bytes."""
    if os.name != "nt":
        return (os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"),
                os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))
    import ctypes
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *((name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical",
                        "total_page_file", "available_page_file", "total_virtual", "available_virtual",
                        "available_extended_virtual"))]
    status = Status(length=ctypes.sizeof(Status))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.total_physical, status.available_physical


def available_memory_bytes():
    return _memory_status()[1]


def total_memory_bytes():
    return _memory_status()[0]


def process_memory_bytes():
    """(working set, peak working set) of this process in bytes, None if unreadable.

    A worker reports its own: on Windows the manager's child is the venv
    launcher, not the interpreter that runs the candidate.
    """
    if os.name != "nt":
        try:
            fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
                          if ":" in line)
            return int(fields["VmRSS"].split()[0]) * 1024, int(fields["VmHWM"].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                    *((name, ctypes.c_size_t) for name in ("peak_working_set", "working_set",
                        "peak_paged_pool", "paged_pool", "peak_nonpaged_pool", "nonpaged_pool",
                        "pagefile", "peak_pagefile"))]
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    counters = Counters(cb=ctypes.sizeof(Counters))
    if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return counters.working_set, counters.peak_working_set


def known_peaks(directory):
    """(parameters, peak memory) of every saved candidate that measured its peak."""
    rows = (read_json(path) for path in sorted((Path(directory) / "summaries").glob("*.json")))
    return [(row["parameters"], row["peak_memory_bytes"]) for row in rows
            if row and _numeric(row.get("peak_memory_bytes"))]


def predicted_peak_bytes(known, case, layout, batch):
    """A candidate's peak memory from measured peaks, linear in the batch.

    Same model, hardware, workload and layout first, then model and
    workload, then model; DEFAULT_CANDIDATE_PEAK_BYTES without any.
    """
    params = {**case, **layout}
    for keys in (("model", "hardware", "workload", *LAYOUT_KEYS), ("model", "workload"), ("model",)):
        points = [(row["batch_size"], peak) for row, peak in known if all(row[key] == params[key] for key in keys)]
        if not points:
            continue
        if len({b for b, _ in points}) == 1:
            return max(peak for _b, peak in points) * max(1., batch / points[0][0])
        intercept, slope = search_plan.fit_line(points)
        floor = max((peak for b, peak in points if b <= batch), default=0.)
        return max(intercept + max(0., slope) * batch, floor)
    return DEFAULT_CANDIDATE_PEAK_BYTES


def worker_ceiling(workers):
    """Most concurrent workers: an integer in [1, MAX_WORKERS], or "auto" for MAX_WORKERS."""
    if workers == "auto":
        return MAX_WORKERS
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer in [1, {MAX_WORKERS}] or auto")
    return workers


def _working_set(directory, identity):
    """A running worker's last reported working set in bytes, 0 before its first report."""
    progress = read_json(Path(directory) / "progress" / (identity + ".json")) or {}
    usage = progress.get("working_set_bytes") if progress.get("candidate_id") == identity else None
    return usage if _numeric(usage) else 0


def admit(queue, active, known, ceiling, directory):
    """The head of ``queue`` to start now, with predicted peaks, and the wall factor they share.

    A candidate started while k workers run (itself included) gets k times the
    per-worker wall caps, so concurrency never rises above the smallest k of a
    running candidate. Candidates start in queue order while their predicted
    peaks fit in the free memory left by the reserve and by the running
    candidates' predicted remaining growth; with nothing running one always
    starts.
    """
    limit = min([ceiling, *(job["factor"] for job in active.values())])
    if len(active) >= limit:
        return [], len(active)
    free = available_memory_bytes() - WORKER_MEMORY_RESERVE_BYTES
    for identity, job in active.items():
        free -= max(0, job["peak"] - _working_set(directory, identity))
    chosen = []
    for item in queue:
        if len(active) + len(chosen) >= limit:
            break
        peak = predicted_peak_bytes(known, *item)
        if (active or chosen) and peak > free:
            break
        chosen.append((item, peak))
        free -= peak
    return chosen, len(active) + len(chosen)


def _start(queue, active, known, ceiling, directory, mode, spawn):
    """Start the admitted head of ``queue`` of (case, layout, batch) items."""
    chosen, factor = admit(queue, active, known, ceiling, directory)
    for item, peak in chosen:
        queue.remove(item)
        case, layout, batch = item
        process = (spawn(case, layout, batch, directory) if spawn
                   else _spawn(case, layout, batch, directory, mode, factor))
        active[candidate_id(case, layout, batch)] = {"process": process, "case": case, "layout": layout,
                                                     "batch": batch, "factor": factor, "peak": peak}


def checked_directory(directory):
    """Refuse the paused v15 scan: opening its manager.lock would truncate it."""
    directory = Path(directory)
    if any(directory.resolve() == retired.resolve() for retired in RETIRED_DIRECTORIES):
        raise ValueError("that directory holds the paused v15 scan; choose a new results directory")
    return directory


def candidate_command(case, layout, batch, directory, mode="detailed", wall_factor=1):
    return [sys.executable, str(Path(__file__).resolve()), "--worker-case-id", case["case_id"],
            "--layout", *(str(layout[key]) for key in LAYOUT_KEYS), "--batch-size", str(batch),
            "--mode", mode, "--directory", str(directory), "--wall-factor", str(wall_factor)]


def _spawn(case, layout, batch, directory, mode="detailed", wall_factor=1):
    """One fresh subprocess per candidate: a candidate's heap never fragments the next one's."""
    directory = checked_directory(directory)
    log = directory / "logs" / (params_candidate_id(parameters(case, layout, batch, mode)) + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent workers share the cores: each gets its slice of HBM channel threads.
    env = {**os.environ, "HETEROLLM_HBM_CHANNEL_THREADS": str(max(1, MAX_WORKERS//wall_factor))}
    with log.open("a", encoding="utf-8") as stdout, log.with_suffix(".err.log").open("a", encoding="utf-8") as stderr:
        return subprocess.Popen(candidate_command(case, layout, batch, directory, mode, wall_factor), cwd=ROOT,
            stdout=stdout, stderr=stderr, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def _settle_candidate(directory, case, layout, batch, code, mode):
    """The saved row of an exited worker; a worker that left none has failed."""
    params = parameters(case, layout, batch, mode)
    identity = params_candidate_id(params)
    row = read_candidate(directory, identity)
    if row is not None and row.get("parameters") != params:
        raise RuntimeError("saved candidate parameters differ; historical results cannot be relabeled")
    if row is None or row.get("status") not in CANDIDATE_TERMINAL | {"paused"}:
        row = {"runner_contract": params["runner_contract"], "backend": BACKEND, "parameters": params,
               "candidate_id": identity, "status": "failed", "metrics": None, "forecast_tokens": 0,
               "error": {"type": "WorkerExit", "message": f"worker exited {code} without a terminal candidate result"}}
        save_candidate(directory, row)
    return row


def _saved_terminal(directory, case, layout, batch, mode):
    row = read_candidate(directory, candidate_id(case, layout, batch))
    return (row is not None and row.get("parameters") == parameters(case, layout, batch, mode)
            and row.get("backend") == BACKEND and row.get("status") in CANDIDATE_TERMINAL)


def _jobs(active):
    return [{"case_id": job["case"]["case_id"], "candidate_id": key, "launcher_pid": job["process"].pid,
             "wall_factor": job["factor"], "predicted_peak_bytes": job["peak"]} for key, job in active.items()]


def _refuse_live_workers(directory):
    for path in (directory / "progress").glob("*.json") if (directory / "progress").exists() else ():
        p = read_json(path, {})
        if p.get("state") == "running" and pid_alive(p.get("pid")):
            raise ManagerConflict("a native worker is still alive; refusing duplicate manager or automatic worker restart")


def run_manager(directory=DEFAULT_DIRECTORY, *, layout=CONTROL_LAYOUT, batches=(1, 8), workers="auto",
                spawn=None, poll_seconds=1., mode="detailed", case_ids=None):
    """Every case runs its candidates at one fixed layout; each candidate is its own subprocess."""
    directory, batches = checked_directory(directory), positive_batches(list(batches))
    ceiling = worker_ceiling(workers)
    layout = make_layout(*(layout[key] for key in LAYOUT_KEYS))
    cases = selected_cases(case_ids)
    check_grid(cases, layout, batches)
    grid = [(layout, batch) for batch in batches]
    planned = {case["case_id"]: grid for case in cases}
    directory.mkdir(parents=True, exist_ok=True)
    with ManagerLock(directory, mode):
        _refuse_live_workers(directory)
        previous = read_json(directory / "run_manifest.json")
        if previous and (previous.get("runner_contract") != runner_contract(mode) or previous.get("cases") != cases
                         or previous.get("layout") != layout
                         or not set(previous.get("batch_sizes", [])).issubset(batches)):
            raise RuntimeError("existing native manifest differs or requested batch set shrinks; refusing silent parameter changes")
        write_json(directory / "run_manifest.json", {"runner_contract": runner_contract(mode), "backend": BACKEND,
            "cases": cases, "layout": layout, "batch_sizes": batches, "max_parallel_configurations": workers,
            "worker_memory_reserve_bytes": WORKER_MEMORY_RESERVE_BYTES,
            "phases": list(PHASES), "simulation_mode": mode,
            "selection_scope": "finite actually executed (layout, batch) candidates by per-GPU throughput, not optimum or maximum capacity"})
        active, last_report = {}, 0.
        error = None
        # One queue in phase order: a free worker never waits at a phase end.
        queue = []
        for case in cases:
            previous = _reusable_point(case, [candidate_id(case, layout, batch) for batch in batches], directory, mode)
            if previous and previous["status"] == "failed":
                error = previous.get("error") or "existing failed case requires diagnosis"
                break
            if previous is None:
                pending = [batch for batch in batches if not _saved_terminal(directory, case, layout, batch, mode)]
                queue.extend((case, layout, batch) for batch in pending)
                if not pending:
                    write_case_point(directory, case, layout, batches, mode)
        known = known_peaks(directory)

        while queue or active:
            stop = (directory / "stop.request").exists()
            if queue and not stop and not error:
                _start(queue, active, known, ceiling, directory, mode, spawn)
            changed = False
            for identity, job in list(active.items()):
                code = job["process"].poll()
                if code is None:
                    continue
                del active[identity]
                changed = True
                case, batch = job["case"], job["batch"]
                row = _settle_candidate(directory, case, layout, batch, code, mode)
                if _numeric(row.get("peak_memory_bytes")):
                    known.append((row["parameters"], row["peak_memory_bytes"]))
                write_case_point(directory, case, layout, batches, mode)
                if row["status"] == "failed" or code not in (0,):
                    error = row.get("error") or f"worker exit {code}"
                    if not (directory / "stop.request").exists():
                        write_json(directory / "stop.request", {"reason": "native_case_failed", "candidate_id": identity, "error": error})
            live = {job["case"]["case_id"] for job in active.values()}
            phase = next((case["phase"] for case in cases if case["case_id"] in live
                          or any(item[0] is case for item in queue)), None)
            jobs = _jobs(active)
            if changed or time.monotonic() - last_report >= 60 or not active:
                refresh(directory, cases, planned, jobs, phase=phase, error=error, mode=mode)
                last_report = time.monotonic()
            if not active and ((directory / "stop.request").exists() or error):
                state = "failed" if error else "paused"
                refresh(directory, cases, planned, state=state, phase=phase, error=error, mode=mode)
                return 1 if error else 0
            if active:
                time.sleep(poll_seconds)
        points = [read_json(directory / "points" / (case["case_id"] + ".json"), {}) for case in cases]
        state = "completed_with_rejections" if any(point.get("status") == "all_candidates_rejected"
                                                  for point in points) else "completed"
        refresh(directory, cases, planned, state=state, mode=mode)
        return 0


def bypass_screen(case):
    """Every legal bypass layout of the case, ranked, with its batch bracket (screening only)."""
    from exploring_hbf_paper_config import hardware_by_id, workload_catalog
    from exploring_hbf_search import search_candidates
    from heterollm_sim.paper_models import get_paper_model
    workload = next(row for row in workload_catalog() if row.workload_id == case["workload"])
    return search_candidates(get_paper_model(case["model"]), hardware_by_id(case["hardware"]), case["gpu_count"],
                             workload, slo_seconds=SLO_NS / 1e9)


def case_screen(directory, case, screen=None):
    """The case's bypass screen and the layouts it hands to native, computed once per results directory."""
    path = Path(directory) / "screen" / (case["case_id"] + ".json")
    saved = read_json(path)
    if saved is not None and saved.get("case") == case and saved.get("deployment_search") == search_plan.SETTINGS:
        return saved
    rows = (screen or bypass_screen)(case)
    chosen, skipped = search_plan.screen_layouts(
        rows, lambda layout, quantum: layout_rejection(case, make_layout(*(layout[key] for key in LAYOUT_KEYS)), quantum))
    value = {"case": case, "deployment_search": search_plan.SETTINGS, "layouts": chosen, "skipped_layouts": skipped,
             "bypass_rows": rows, "scope": "bypass screening only ranks layouts and places native batch points; it reports no number"}
    write_json(path, value)
    return value


def _search_lookup(directory, case, layout):
    def lookup(batch):
        row = read_candidate(directory, candidate_id(case, layout, batch))
        if row is None:
            return None
        if row.get("parameters") != parameters(case, layout, batch, "fast") or row.get("backend") != BACKEND:
            raise RuntimeError("saved candidate parameters/backend differ; historical results cannot be relabeled")
        return row if row.get("status") in CANDIDATE_TERMINAL else None
    return lookup


def search_case(directory, case, screen):
    """Replay the case's search from its saved rows; write its point. Returns (plan, planned candidates, point)."""
    directory = Path(directory)
    entries = screen["layouts"]
    plan = search_plan.plan_case(entries, [_search_lookup(directory, case, entry["layout"]) for entry in entries],
        SLO_NS, FAST_CANDIDATE_WALL_BUDGET_SECONDS, [replica_gpus(entry["layout"]) for entry in entries],
        memory_budget_bytes=total_memory_bytes() - WORKER_MEMORY_RESERVE_BYTES)
    planned = []
    for layout_plan in plan["layouts"]:
        planned += [(layout_plan["layout"], run["batch"]) for run in layout_plan["runs"]]
        if "next" in layout_plan:
            planned.append((layout_plan["layout"], layout_plan["next"]))
    acceptance = plan["acceptance"]
    if acceptance and acceptance["kind"] == "interpolated" and (acceptance["layout"], acceptance["batch"]) not in planned:
        planned.append((acceptance["layout"], acceptance["batch"]))
    rows = [row for row in (read_candidate(directory, candidate_id(case, layout, batch))
                            for layout, batch in planned) if row is not None]
    point = select_case(case, [candidate_id(case, layout, batch) for layout, batch in planned], rows, "fast")
    selected = point["selected"]
    point["search"] = {"result": (search_plan.search_result(plan, selected["parameters"]["batch_size"],
                                                             params_layout(selected["parameters"])) if selected else None),
        "layouts": [{key: layout_plan.get(key) for key in ("layout", "quantum", "bypass", "runs", "points", "outcome", "next")}
                    for layout_plan in plan["layouts"]],
        "acceptance": acceptance, "skipped_layouts": screen["skipped_layouts"], "settings": search_plan.SETTINGS}
    write_json(directory / "points" / (case["case_id"] + ".json"), point)
    return plan, planned, point


def run_search_manager(directory=DEFAULT_DIRECTORY, *, workers="auto", spawn=None, poll_seconds=1., screen=None,
                       case_ids=None):
    """Paper-style deployment search over every case; each native point is its own fast subprocess."""
    mode = "fast"
    directory = checked_directory(directory)
    ceiling = worker_ceiling(workers)
    cases = selected_cases(case_ids)
    directory.mkdir(parents=True, exist_ok=True)
    with ManagerLock(directory, mode):
        _refuse_live_workers(directory)
        manifest = {"runner_contract": runner_contract(mode), "backend": BACKEND, "cases": cases,
            "deployment_search": search_plan.SETTINGS, "slo_tpot_ns": SLO_NS, "max_parallel_configurations": workers,
            "worker_memory_reserve_bytes": WORKER_MEMORY_RESERVE_BYTES,
            "phases": list(PHASES), "simulation_mode": mode,
            "selection_scope": "highest conditionally accepted per-GPU throughput among the native points the deployment search ran; not a global optimum"}
        previous = read_json(directory / "run_manifest.json")
        if previous and any(previous.get(key) != manifest[key] for key in ("runner_contract", "cases", "deployment_search")):
            raise RuntimeError("existing native manifest differs; refusing silent parameter changes")
        write_json(directory / "run_manifest.json", manifest)
        screens = {case["case_id"]: case_screen(directory, case, screen) for case in cases}

        def replay():
            return {case["case_id"]: search_case(directory, case, screens[case["case_id"]]) for case in cases}

        plans, active, error, last_report, known = replay(), {}, None, 0., known_peaks(directory)
        failed = next((point for _plan, _planned, point in plans.values() if point["status"] == "failed"), None)
        if failed:
            error = failed.get("error") or "existing failed case requires diagnosis"
        while True:
            stop = (directory / "stop.request").exists()
            queue = [(case, layout, batch) for case in cases for layout, batch in plans[case["case_id"]][0]["next"]
                     if candidate_id(case, layout, batch) not in active]
            if queue and not stop and not error:
                _start(queue, active, known, ceiling, directory, mode, spawn)
            changed = False
            for identity, job in list(active.items()):
                code = job["process"].poll()
                if code is None:
                    continue
                del active[identity]
                changed = True
                row = _settle_candidate(directory, job["case"], job["layout"], job["batch"], code, mode)
                if _numeric(row.get("peak_memory_bytes")):
                    known.append((row["parameters"], row["peak_memory_bytes"]))
                if row["status"] == "failed" or code not in (0,):
                    error = row.get("error") or f"worker exit {code}"
                    if not (directory / "stop.request").exists():
                        write_json(directory / "stop.request", {"reason": "native_case_failed", "candidate_id": identity, "error": error})
            if changed:
                plans = replay()
            planned = {key: value[1] for key, value in plans.items()}
            live = {job["case"]["case_id"] for job in active.values()}
            phase = next((case["phase"] for case in cases if case["case_id"] in live
                          or plans[case["case_id"]][0]["next"]), None)
            jobs = _jobs(active)
            if changed or time.monotonic() - last_report >= 60 or not active:
                refresh(directory, cases, planned, jobs, phase=phase, error=error, mode=mode, search=True)
                last_report = time.monotonic()
            if not active and ((directory / "stop.request").exists() or error):
                refresh(directory, cases, planned, state="failed" if error else "paused", phase=phase, error=error,
                        mode=mode, search=True)
                return 1 if error else 0
            if not active and not any(plan["next"] for plan, _planned, _point in plans.values()):
                break
            if active:
                time.sleep(poll_seconds)
        points = [read_json(directory / "points" / (case["case_id"] + ".json"), {}) for case in cases]
        state = "completed_with_rejections" if any(point.get("status") == "all_candidates_rejected"
                                                  for point in points) else "completed"
        refresh(directory, cases, planned, state=state, mode=mode, search=True)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY,
        help="results directory; the paused v15 scan directory is refused")
    parser.add_argument("--layout", nargs=4, type=int, default=[CONTROL_LAYOUT[key] for key in LAYOUT_KEYS],
        metavar=("DP", "TP", "PP", "EP"), help="data-parallel replicas and one replica's TP/PP/EP degrees")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8], help="per-replica batch candidates")
    parser.add_argument("--workers", default="auto",
        help=f"并行候选数上限，1～{MAX_WORKERS}，或auto（默认，上限为核数）；按预测峰值内存逐个增开，"
             f"保留{WORKER_MEMORY_RESERVE_BYTES//1024**3}GiB余量；k个并行时候选壁钟上限放宽为k倍")
    parser.add_argument("--mode", choices=("detailed", "fast"), default="detailed",
        help="fast候选总墙钟上限1800秒，优化目标600秒；上限覆盖配置、初始化、物理样本和预算")
    parser.add_argument("--search", action="store_true",
        help="论文式部署搜索（总是fast）：旁路筛选、原生2–3个批量点、插值到100ms、一次原生验收；忽略--layout/--batch-sizes")
    parser.add_argument("--cases", nargs="+", metavar="CASE_ID",
        help="只运行这些配置（按矩阵顺序）；所选集合写入运行清单，同一目录之后不能改变")
    parser.add_argument("--worker-case-id", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--wall-factor", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(argv)
    if args.workers != "auto":
        try:
            args.workers = int(args.workers)
            worker_ceiling(args.workers)
        except ValueError:
            parser.error(f"--workers must be an integer in [1, {MAX_WORKERS}] or auto")
    if args.search:
        args.mode = "fast"
    try:
        directory = checked_directory(args.directory)
        layout = make_layout(*args.layout)
        batches = positive_batches(args.batch_sizes)
        cases = selected_cases(args.cases)
        if not args.worker_case_id and not args.search:
            check_grid(cases, layout, batches)
    except ValueError as error:
        parser.error(str(error))
    if args.worker_case_id:
        case = next((row for row in case_matrix() if row["case_id"] == args.worker_case_id), None)
        if case is None or args.batch_size is None:
            parser.error("a worker needs a known native case and --batch-size")
        try:
            return worker_candidate(case, layout, args.batch_size, directory, args.mode, args.wall_factor)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return 1
    if args.background:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "manager.log").open("a", encoding="utf-8") as output, (directory / "manager.err.log").open("a", encoding="utf-8") as errors:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--workers", str(args.workers),
                "--mode", args.mode, "--directory", str(directory), "--layout", *map(str, args.layout),
                "--batch-sizes", *map(str, batches), *(["--search"] if args.search else []),
                *(["--cases", *args.cases] if args.cases else [])], cwd=ROOT,
                stdout=output, stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt")
        print(json.dumps({"launcher_pid": process.pid, "directory": str(directory), "note": "actual manager/worker PIDs are in status.json/progress"}))
        return 0
    try:
        if args.search:
            return run_search_manager(directory, workers=args.workers, case_ids=args.cases)
        return run_manager(directory, layout=layout, batches=batches, workers=args.workers, mode=args.mode,
                           case_ids=args.cases)
    except ManagerConflict as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(failure["traceback"], file=sys.stderr, flush=True)
        if not (directory / "stop.request").exists():
            write_json(directory / "stop.request", {"reason": "native_manager_failed", "error": failure})
        refresh(directory, cases, {case["case_id"]: [] if args.search else [(layout, batch) for batch in batches]
                                   for case in cases}, state="failed", error=failure, mode=args.mode, search=args.search)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
