"""Run the bounded 9 + 6 + 15 native-backend comparison, never an old scan.

Each configuration validates a finite set of real batch candidates on fixed
TP8/PP1/EP1. The selected passing candidate is not a global optimum. Both
modes execute through run_scenario. Explicit fast mode keeps complete native
cohort samples separate from a conditional forecast; it never runs a rejected
forecast's detailed suffix or an old detailed reference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import csv
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
DIRECTORY = ROOT / "artifacts/exploring_hbf_native_20261019"
CONTRACT = "native_fixed_tp8_finite_batch_controls_v1"
FAST_CONTRACT = "native_sampled_tp8_finite_batch_controls_v2"
FAST_LAYER_CONTRACT = "native_layer_sampled_tp8_finite_batch_controls_v3"
BACKEND = "native_serving_physical"
PHASES = ("dense_native", "dense_plus", "maverick_native")
TERMINAL = {"completed", "capacity_rejected", "slo_rejected", "all_candidates_rejected", "failed"}
CANDIDATE_TERMINAL = {"completed", "capacity_rejected", "slo_rejected", "forecast_rejected", "failed"}
SLO_NS = 100_000_000.
FAST_CANDIDATE_WALL_TARGET_SECONDS = 600.
FAST_CANDIDATE_WALL_BUDGET_SECONDS = 1800.
STATUS_LABELS = {"pending": "等待", "running": "运行中", "completed": "完成",
    "completed_with_rejections": "结束（含拒绝）", "all_candidates_rejected": "候选均被拒绝",
    "capacity_rejected": "容量拒绝", "slo_rejected": "TPOT约束拒绝", "forecast_rejected": "预测预算拒绝", "paused": "暂停", "failed": "失败"}


class ManagerConflict(RuntimeError):
    """Another owner must be inspected; do not mutate its report or stop flag."""


from heterollm_sim.execution_control import NativeFastWallBudgetError


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
                    "gpu_count": 8, "tp": 8, "pp": 1, "ep": 1, "weight_dtype": "bf16", "kv_dtype": "bf16"})
    return rows


def candidate_id(case, batch):
    return case["case_id"] + f"__B{batch}"


def runner_contract(mode, sampling_unit="cohorts"):
    if mode not in {"detailed", "fast"}:
        raise ValueError("mode must be detailed or fast")
    if sampling_unit not in {"cohorts", "layer_windows"}:
        raise ValueError("unknown native sampling unit")
    return (FAST_LAYER_CONTRACT if sampling_unit == "layer_windows" else FAST_CONTRACT) if mode == "fast" else CONTRACT


def parameters(case, batch, mode="detailed", sampling_unit="cohorts"):
    return {**case, "batch_size": batch, "runner_contract": runner_contract(mode, sampling_unit),
            **({"simulation_mode": "fast", "candidate_wall_budget_seconds": FAST_CANDIDATE_WALL_BUDGET_SECONDS,
                "candidate_wall_target_seconds": FAST_CANDIDATE_WALL_TARGET_SECONDS,
                "target_relative_error": .10, "sampling_unit": sampling_unit} if mode == "fast" else {}),
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


class ManagerLock:
    def __init__(self, directory, mode="detailed", sampling_unit="cohorts"):
        self.path = Path(directory) / "manager.lock"
        self.handle = None
        self.contract = runner_contract(mode, sampling_unit)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            self.handle = None


class ProgressWriter:
    def __init__(self, directory, case_id):
        self.path = Path(directory) / "progress" / (case_id + ".json")
        self.started = time.perf_counter()
        self.state = {"case_id": case_id, "pid": os.getpid(), "state": "running", "stage": "starting"}
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
        return {**self.state, "updated_at": utc_now(),
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
                "sampling_unit": params.get("sampling_unit"),
                "candidate_id": candidate_id(params, params["batch_size"]),
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
            if _numeric(metric.ttft_ns):
                ttfts.append(metric.ttft_ns)
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
    cluster_tps = visible * 1e9 / duration
    metrics = {"cluster_tokens_per_second": cluster_tps,
        "tokens_per_second_per_gpu": cluster_tps / params["gpu_count"],
        "mean_complete_request_tpot_ns": mean_tpot, "max_complete_request_tpot_ns": max(samples.values()),
        "mean_complete_request_ingress_ttft_ns": statistics.mean(ttfts) if ttfts else None,
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


def capacity_error(error):
    from heterollm_sim.memory_allocator import AllocationError
    if isinstance(error, AllocationError):
        # The allocator also uses this exception for invalid aliases,
        # generations and overlapping addresses. Only allocation exhaustion
        # is a legitimate capacity rejection; correctness errors must stop.
        return str(error).startswith("unable to allocate ") and " bytes in capacity " in str(error)
    details = getattr(error, "details", {})
    diagnostics = details.get("diagnostics", []) if isinstance(details, dict) else []
    if not diagnostics:
        return False
    markers = ("exceeds available capacity", "exceeds physical capacity", "exceeds effective capacity",
               "insufficient capacity", "超过有效容量", "超出可用容量", "内存容量不足")
    return all(any(marker in (str(row.get("message_en", "")) + " " + str(row.get("message_zh", ""))).lower()
                   for marker in markers) for row in diagnostics)


def matched_fast_reference(params, directory):
    """Use only the same authored HBM4 fast candidate, never an old reference."""
    if params["hardware"] == "HBM4":
        return None
    reference_case = next((case for case in case_matrix()
        if case["hardware"] == "HBM4" and all(case[key] == params[key]
            for key in ("model", "workload", "input_tokens", "output_tokens", "tp", "pp", "ep"))), None)
    if reference_case is None:
        return None
    row = read_json(Path(directory) / "candidates" /
        (candidate_id(reference_case, params["batch_size"]) + ".json"), {})
    unit = params.get("sampling_unit", "cohorts")
    if (row.get("parameters") != parameters(reference_case, params["batch_size"], "fast", unit)
            or row.get("runner_contract") != runner_contract("fast", unit) or row.get("backend") != BACKEND
            or row.get("status") != "completed"):
        return None
    return row.get("approximation", {}).get("error_budget", {}).get("paired_reference")


def native_fast_assumptions(scenario, sampling_unit):
    from exploring_hbf_native_config import PHYSICAL_WINDOW_ASSUMPTION
    scope = ("Independent complete native layer windows and boundary samples are physical; "
             "omitted layers, cohorts and visible Tokens are separate conditional predictions."
             if sampling_unit == "layer_windows" else
             "Sampled complete native cohorts are physical; the remaining service window is a separate conditional prediction.")
    return tuple(dict.fromkeys((*[value for value in scenario.assumptions if value != PHYSICAL_WINDOW_ASSUMPTION],
        scope, "Single-point and fixed-direction HBM4 ratio error budgets retain the 10% target; "
        "conditional budgets are not full-window detailed validation or silicon/paper accuracy guarantees.")))


def native_fast_result(result, params):
    """Keep the physical prefix and the unexecuted projection separate."""
    from heterollm_sim.reporting import FastScenarioResult
    if not isinstance(result, FastScenarioResult) or result.detailed is not None:
        raise RuntimeError("native fast candidate did not return a separate sampled/forecast result")
    forecast = result.native_window_forecast
    unit = params.get("sampling_unit", "cohorts")
    if (forecast is not None and forecast.sampling_unit != unit
            or unit == "layer_windows" and result.sampled is not None):
        raise RuntimeError("native fast sampling contract does not match its physical result scope")
    details = {"approximation": dict(result.approximation),
        "physical_prefix_traffic": traffic_result(result.sampled) if result.sampled is not None else None,
        "physical_layer_samples": [sample.to_metadata() for sample in result.layer_window_samples],
        "physical_boundary_accounting": result.approximation.get("physical_boundary_accounting"),
        "traffic": None, "assumptions": list(native_fast_assumptions(result.scenario, unit))}
    if forecast is None:
        return {**details, "status": "forecast_rejected", "metrics": None,
            "rejection_reason": ";".join(result.approximation.get("rejection_reasons", ()))
                or "native_window_projection_unavailable"}
    bounds = forecast.metric_bounds.get("mean_complete_request_tpot_ns", {})
    upper = bounds.get("upper")
    if not _numeric(upper) or upper <= 0:
        raise RuntimeError("accepted native fast forecast has no positive TPOT upper bound")
    metrics = dict(forecast.metrics)
    return {**details, "status": "completed" if upper <= SLO_NS else "slo_rejected",
        "metrics": metrics if upper <= SLO_NS else None,
        "native_window_forecast": forecast.to_metadata(),
        "forecast_tokens": forecast.predicted_visible_output_tokens,
        "sampled_tokens": forecast.sampled_visible_output_tokens,
        "diagnostics": {"observed_metrics": metrics, "metric_bounds": dict(forecast.metric_bounds),
            "slo_met": upper <= SLO_NS, "slo_tpot_ns": SLO_NS,
            "slo_statistic": "conditional_upper_bound_of_mean_complete_fresh_request_tpot"},
        **({"rejection_reason": "mean_complete_fresh_request_tpot_upper_bound_exceeds_100ms"}
            if upper > SLO_NS else {})}


def execute_candidate(params, directory, progress, *, builder=None, runner=None):
    from heterollm_sim.execution_control import ExecutionControl, ExecutionCancelledError
    from exploring_hbf_native_config import build_native_comparison_scenario
    from heterollm_sim.reporting import run_scenario
    directory = Path(directory)
    started = time.perf_counter()
    fast = params.get("simulation_mode") == "fast"
    row = {"runner_contract": params["runner_contract"], "backend": BACKEND, "parameters": params,
           "candidate_id": candidate_id(params, params["batch_size"]), "status": "running",
           "metrics": None, "started_at": utc_now(), "forecast_tokens": 0,
           "fidelity": ("native_boundary_conditioned_layer_windows_and_separate_conditional_forecast"
                if params.get("sampling_unit") == "layer_windows" else
                "native_physical_cohort_samples_and_separate_conditional_window_forecast"
                if fast else "complete_bounded_native_physical_window_no_forecast_no_old_reference")}
    try:
        native_progress = type(progress) is ProgressWriter and type(progress.queue) is SimpleQueue
        control = ExecutionControl(
            progress_queue=progress.queue if native_progress else None,
            progress_callback=None if native_progress else progress.report,
            stop_request_path=directory / "stop.request",
            deadline_monotonic=started + params["candidate_wall_budget_seconds"] if fast else None)
        control.raise_if_cancelled()
        scenario = (builder or build_native_comparison_scenario)(params["model"], params["hardware"], params["workload"],
            batch_size=params["batch_size"], tp=8, pp=1, ep=1)
        if fast:
            from heterollm_sim.simulation_options import SimulationConfig, NativeWindowFastConfig
            remaining = params["candidate_wall_budget_seconds"] - (time.perf_counter() - started)
            if remaining <= 0:
                raise NativeFastWallBudgetError("native candidate construction exhausted the wall budget")
            scenario = replace(scenario, simulation=SimulationConfig(mode="fast",
                native_window=NativeWindowFastConfig(max_wall_seconds=remaining,
                    sampling_unit=params.get("sampling_unit", "cohorts"),
                    layer_window_size=params.get("layer_window_size", 1),
                    max_layer_window_size=params.get("max_layer_window_size"))),
                assumptions=native_fast_assumptions(scenario, params.get("sampling_unit", "cohorts")))
        write_json(directory / "configurations" / (row["candidate_id"] + ".json"), scenario)
        control.raise_if_cancelled()
        options = {"retention_policy": "aggregate", "control": control}
        if fast:
            reference = matched_fast_reference(params, directory)
            if reference is not None:
                options["paired_reference"] = reference
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
            if time.perf_counter() - started >= params["candidate_wall_budget_seconds"]:
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


def select_case(case, batches, rows, mode="detailed", sampling_unit="cohorts"):
    failed = next((row for row in rows if row["status"] == "failed"), None)
    accepted = [row for row in rows if row["status"] == "completed"]
    complete = len(rows) == len(batches) and all(row["status"] in CANDIDATE_TERMINAL for row in rows)
    state = ("failed" if failed else "paused" if any(row["status"] == "paused" for row in rows)
             else "completed" if complete and accepted else "all_candidates_rejected" if complete else "running")
    selected = max(accepted, key=lambda row: row["metrics"]["cluster_tokens_per_second"]) if state == "completed" else None
    return {"case": case, "case_id": case["case_id"], "runner_contract": runner_contract(mode, sampling_unit), "backend": BACKEND,
        "status": state, "batch_sizes": batches, "candidate_ids": [row["candidate_id"] for row in rows],
        "candidate_statuses": {row["parameters"]["batch_size"]: row["status"] for row in rows},
        "selected": selected, "error": failed.get("error") if failed else None, "updated_at": utc_now(),
        "selection_scope": ("highest conditionally accepted projected throughput among sampled fixed-TP8 batch candidates; not a global optimum or maximum feasible batch"
            if mode == "fast" else "highest measured passing throughput among the explicitly executed fixed-TP8 batch candidates; not a global optimum or maximum feasible batch")}


def worker_case(case, batches, directory, mode="detailed", sampling_unit="cohorts"):
    directory = Path(directory)
    rows = []
    with ProgressWriter(directory, case["case_id"]) as progress:
        for batch in batches:
            params = parameters(case, batch, mode, sampling_unit)
            path = directory / "candidates" / (candidate_id(case, batch) + ".json")
            row = read_json(path)
            if row is not None:
                if row.get("parameters") != params or row.get("backend") != BACKEND:
                    raise RuntimeError("saved candidate parameters/backend differ; historical results cannot be relabeled")
                if row["status"] not in CANDIDATE_TERMINAL:
                    row = None
            if row is None:
                progress.begin_candidate(params)
                progress.publish()
                row = execute_candidate(params, directory, progress)
                write_json(path, row)
            rows.append(row)
            point = select_case(case, batches, rows, mode, sampling_unit)
            write_json(directory / "points" / (case["case_id"] + ".json"), point)
            print(json.dumps({"case_id": case["case_id"], "batch_size": batch,
                              "status": row["status"], "wall_seconds": row.get("wall_seconds")}), flush=True)
            if row["status"] in {"failed", "paused"}:
                break
        with progress.lock:
            progress.state.update(state=point["status"])
    return 1 if point["status"] == "failed" else 0


def refresh(directory, cases, batches, running=(), *, state="running", phase=None, error=None, mode="detailed", sampling_unit="cohorts"):
    directory = Path(directory)
    points, candidates = [], []
    active_ids = {row["case_id"] for row in running}
    for case in cases:
        point = read_json(directory / "points" / (case["case_id"] + ".json"))
        if point and (point.get("case") != case or point.get("backend") != BACKEND
                      or point.get("runner_contract") != runner_contract(mode, sampling_unit)):
            point = {"case": case, "case_id": case["case_id"], "status": "failed", "selected": None,
                     "error": {"message": "foreign or mismatched historical result is not native evidence"}}
        elif point and not set(batches).issubset(point.get("batch_sizes", [])):
            point = {**point, "status": "running" if case["case_id"] in active_ids else "pending", "selected": None,
                     "previous_batch_sizes": point.get("batch_sizes"), "batch_sizes": batches}
        points.append(point or {"case": case, "case_id": case["case_id"],
            "status": "running" if case["case_id"] in active_ids else "pending", "selected": None})
        for batch in batches:
            row = read_json(directory / "candidates" / (candidate_id(case, batch) + ".json"))
            if row is not None:
                if row.get("parameters") != parameters(case, batch, mode, sampling_unit) or row.get("backend") != BACKEND:
                    row = {"parameters": parameters(case, batch, mode, sampling_unit), "candidate_id": candidate_id(case, batch),
                           "status": "failed", "metrics": None, "error": {
                               "message": "foreign or mismatched candidate excluded from native metrics"}}
                candidates.append(row)
    counts = Counter(point["status"] for point in points)
    status = {"state": state, "updated_at": utc_now(), "manager_pid": os.getpid(), "current_phase": phase,
        "total_cases": len(cases), "total_candidates": len(cases) * len(batches),
        "counts": dict(counts), "candidate_counts": dict(Counter(row["status"] for row in candidates)),
        "running": list(running), "batch_sizes": batches, "backend": BACKEND, "runner_contract": runner_contract(mode, sampling_unit),
        "simulation_mode": mode, "sampling_unit": sampling_unit if mode == "fast" else None,
        "stop_requested": (directory / "stop.request").exists(), "error": error}
    write_json(directory / "status.json", status)
    write_json(directory / "results.json", {"status": status, "cases": points, "candidates": candidates})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["case_id", "model", "hardware", "workload", "batch_size", "status",
        "tokens_per_second_per_gpu", "cluster_tokens_per_second", "mean_complete_request_tpot_ns", "wall_seconds", "reason"])
    writer.writeheader()
    for row in candidates:
        p, metrics = row["parameters"], row.get("metrics") or {}
        writer.writerow({"case_id": p["case_id"], **{key: p[key] for key in ("model", "hardware", "workload", "batch_size")},
            "status": row["status"], **{key: metrics.get(key) for key in ("tokens_per_second_per_gpu", "cluster_tokens_per_second", "mean_complete_request_tpot_ns")},
            "wall_seconds": row.get("wall_seconds"), "reason": row.get("rejection_reason") or (row.get("error") or {}).get("message", "")})
    atomic_text(directory / "summary.csv", output.getvalue())
    lines = ["# 原生后端论文对照实验", "", f"更新：{status['updated_at']}；状态：{STATUS_LABELS.get(state,state)}。",
        f"配置 {len(cases)} 组，成功 {counts.get('completed',0)}，全候选拒绝 {counts.get('all_candidates_rejected',0)}，失败 {counts.get('failed',0)}。", "",
        f"固定 8 GPU、TP8/PP1/EP1；有限候选批量 {', '.join(map(str,batches))}。每配置选择通过 100ms 请求均值TPOT约束的最高{'预算接受的预测' if mode == 'fast' else '实测'}吞吐，不表示最优部署或最大批量。",
        ("完整原生批次样本与预测窗口分开保存；10%为明示条件模型预算，不是未执行全窗口的实测保证。预测尾段没有物理轨迹，预算拒绝时指标为空，不回退详细后缀。"
            if mode == "fast" else "完整原生预热与有界测量窗口，无预测尾段；零预测Token不等于与论文或实物误差为零。历史详细参考和旧扫描均未重跑。"),
        ("预测吞吐使用相同测量窗口及可见Token定义，TPOT使用窗口内完整新请求的首末Token预测区间；预测完成不计为物理完成。"
            if mode == "fast" else "吞吐按原生窗口 (开始,结束] 的真实可见Token计算；TPOT仅使用首末Token都在窗口内的完整新请求。未在窗口结束的请求不计为完成。"),
        "入口TTFT从请求声明的到达时刻计算；本轮四波次请求均从0时刻排队，包含预热期间的排队等待，不是从PD实际开始计时，也不含外部Prefill计算。",
        ("DRAM/NAND/SRAM与PD真实账本只覆盖采样前缀；全窗口物理流量未知。预测尾段不生成虚构的流量、利用率或物理状态。"
            if mode == "fast" else "DRAM/NAND/SRAM与PD流量为整个有界运行的真实账本，包含预热和已提交的收尾费用，不是仅测量窗口流量或独占关键路径分解。"), "",
        "累计Token、批次及阶段壁钟见[进度分析](progress_analysis.md)；真实慢路径调查见[原生后端对照说明](../../docs/exploring_hbf_native_comparison.md)。局部事务提速不能代替完整候选耗时验收。", "",
        "| 模型 | 存储 | 输入/输出Token | 配置状态 | 选中批量 | tokens/s/GPU | 集群tokens/s | 请求均值TPOT(ms) |", "|---|---|---|---|---:|---:|---:|---:|"]
    for point in points:
        case, selected = point["case"], point.get("selected")
        m = selected["metrics"] if selected else {}
        number = lambda value: f"{value:.3f}" if _numeric(value) else "—"
        model = "Llama 3.1 405B" if case["model"] == "llama3_405b" else "Llama 4 Maverick"
        lines.append(f"| {model} | {case['hardware']} | {case['input_tokens']}/{case['output_tokens']} | {STATUS_LABELS.get(point['status'],point['status'])} | "
            f"{selected['parameters']['batch_size'] if selected else '—'} | {number(m.get('tokens_per_second_per_gpu'))} | "
            f"{number(m.get('cluster_tokens_per_second'))} | {number(m.get('mean_complete_request_tpot_ns', 0)/1e6) if m else '—'} |")
    if running:
        lines += ["", "正在执行："]
        for job in running:
            p = read_json(directory / "progress" / (job["case_id"] + ".json"), {})
            current = p.get("current_cohort") or {}
            cumulative = p.get("cumulative_progress") or {}
            lines.append(f"- {job['case_id']}：真实PID {p.get('pid', '等待启动')}，批量 {p.get('batch_size', '—')}，"
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


def _reusable_point(case, batches, directory, mode="detailed", sampling_unit="cohorts"):
    point = read_json(Path(directory) / "points" / (case["case_id"] + ".json"))
    if not point:
        return None
    if point.get("case") != case or point.get("backend") != BACKEND or point.get("runner_contract") != runner_contract(mode, sampling_unit):
        raise RuntimeError("saved case is not the same native comparison; refusing historical reuse")
    if point["status"] in TERMINAL and set(batches).issubset(point.get("batch_sizes", [])):
        return point
    return None


def _spawn(case, batches, directory, mode="detailed", sampling_unit="cohorts"):
    if Path(directory).resolve() != DIRECTORY.resolve():
        raise ValueError("production native workers may only write the dedicated native results directory")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-case-id", case["case_id"],
               "--mode", mode, "--sampling-unit", sampling_unit, "--batch-sizes", *map(str, batches)]
    log = Path(directory) / "logs" / (case["case_id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stdout, log.with_suffix(".err.log").open("a", encoding="utf-8") as stderr:
        return subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def run_manager(directory=DIRECTORY, *, batches=(1, 8), workers=2, spawn=None, poll_seconds=1., mode="detailed", sampling_unit="cohorts"):
    directory, batches = Path(directory), positive_batches(list(batches))
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError("at most two independent configurations may run")
    cases = case_matrix()
    directory.mkdir(parents=True, exist_ok=True)
    with ManagerLock(directory, mode, sampling_unit):
        for path in (directory / "progress").glob("*.json") if (directory / "progress").exists() else ():
            p = read_json(path, {})
            if p.get("state") == "running" and pid_alive(p.get("pid")):
                raise ManagerConflict("a native worker is still alive; refusing duplicate manager or automatic worker restart")
        previous = read_json(directory / "run_manifest.json")
        if previous and (previous.get("runner_contract") != runner_contract(mode, sampling_unit) or previous.get("cases") != cases
                         or not set(previous.get("batch_sizes", [])).issubset(batches)):
            raise RuntimeError("existing native manifest differs or requested batch set shrinks; refusing silent parameter changes")
        write_json(directory / "run_manifest.json", {"runner_contract": runner_contract(mode, sampling_unit), "backend": BACKEND,
            "cases": cases, "batch_sizes": batches, "max_parallel_configurations": workers,
            "phases": list(PHASES), "simulation_mode": mode, "sampling_unit": sampling_unit,
            "fixed_deployment": "TP8/PP1/EP1",
            "selection_scope": "finite actually executed batches, not optimum or maximum capacity"})
        active, last_report = {}, 0.
        error = None
        for phase in PHASES:
            queue = []
            for case in (row for row in cases if row["phase"] == phase):
                previous = _reusable_point(case, batches, directory, mode, sampling_unit)
                if previous and previous["status"] == "failed":
                    error = previous.get("error") or "existing failed case requires diagnosis"
                    break
                if previous is None:
                    queue.append(case)
            if error:
                break
            while queue or active:
                stop = (directory / "stop.request").exists()
                while queue and len(active) < workers and not stop and not error:
                    case = queue.pop(0)
                    process = spawn(case, batches, directory) if spawn else _spawn(case, batches, directory, mode, sampling_unit)
                    active[case["case_id"]] = (process, case)
                changed = False
                for identity, (process, case) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    del active[identity]
                    changed = True
                    point = read_json(directory / "points" / (identity + ".json"))
                    if not point or point.get("status") not in TERMINAL | {"paused"}:
                        point = {"case": case, "case_id": identity, "status": "failed", "runner_contract": runner_contract(mode, sampling_unit),
                                 "backend": BACKEND, "selected": None, "batch_sizes": batches,
                                 "error": {"type": "WorkerExit", "message": f"worker exited {code} without a terminal case result"}}
                        write_json(directory / "points" / (identity + ".json"), point)
                    if point["status"] == "failed" or code not in (0,):
                        error = point.get("error") or f"worker exit {code}"
                        if not (directory / "stop.request").exists():
                            write_json(directory / "stop.request", {"reason": "native_case_failed", "case_id": identity, "error": error})
                jobs = [{"case_id": key, "launcher_pid": process.pid} for key, (process, _case) in active.items()]
                if changed or time.monotonic() - last_report >= 60 or not active:
                    refresh(directory, cases, batches, jobs, phase=phase, error=error, mode=mode, sampling_unit=sampling_unit)
                    last_report = time.monotonic()
                if not active and ((directory / "stop.request").exists() or error):
                    state = "failed" if error else "paused"
                    refresh(directory, cases, batches, state=state, phase=phase, error=error, mode=mode, sampling_unit=sampling_unit)
                    return 1 if error else 0
                if active:
                    time.sleep(poll_seconds)
        if error:
            refresh(directory, cases, batches, state="failed", error=error, mode=mode, sampling_unit=sampling_unit)
            return 1
        points = [read_json(directory / "points" / (case["case_id"] + ".json"), {}) for case in cases]
        state = "completed_with_rejections" if any(point.get("status") == "all_candidates_rejected"
                                                  for point in points) else "completed"
        refresh(directory, cases, batches, state=state, mode=mode, sampling_unit=sampling_unit)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mode", choices=("detailed", "fast"), default="detailed",
        help="fast候选总墙钟上限1800秒，优化目标600秒；上限覆盖配置、初始化、物理样本和预算")
    parser.add_argument("--sampling-unit", choices=("cohorts", "layer_windows"), default="cohorts")
    parser.add_argument("--worker-case-id", help=argparse.SUPPRESS)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "fast" and args.sampling_unit == "layer_windows":
        parser.error("layer windows require fast mode")
    batches = positive_batches(args.batch_sizes)
    if args.worker_case_id:
        case = next((row for row in case_matrix() if row["case_id"] == args.worker_case_id), None)
        if case is None:
            parser.error("unknown native case")
        try:
            return worker_case(case, batches, DIRECTORY, args.mode, args.sampling_unit)
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            write_json(DIRECTORY / "points" / (case["case_id"] + ".json"), {
                "case": case, "case_id": case["case_id"], "status": "failed", "batch_sizes": batches,
                "runner_contract": runner_contract(args.mode, args.sampling_unit), "backend": BACKEND, "selected": None, "error": failure})
            print(failure["traceback"], file=sys.stderr, flush=True)
            return 1
    if args.background:
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        with (DIRECTORY / "manager.log").open("a", encoding="utf-8") as output, (DIRECTORY / "manager.err.log").open("a", encoding="utf-8") as errors:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--workers", str(args.workers),
                "--mode", args.mode, "--sampling-unit", args.sampling_unit,
                "--batch-sizes", *map(str, batches)], cwd=ROOT, stdout=output, stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt")
        print(json.dumps({"launcher_pid": process.pid, "directory": str(DIRECTORY), "note": "actual manager/worker PIDs are in status.json/progress"}))
        return 0
    try:
        return run_manager(DIRECTORY, batches=batches, workers=args.workers, mode=args.mode, sampling_unit=args.sampling_unit)
    except ManagerConflict as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(failure["traceback"], file=sys.stderr, flush=True)
        if not (DIRECTORY / "stop.request").exists():
            write_json(DIRECTORY / "stop.request", {"reason": "native_manager_failed", "error": failure})
        refresh(DIRECTORY, case_matrix(), batches, state="failed", error=failure, mode=args.mode, sampling_unit=args.sampling_unit)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
