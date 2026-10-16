"""Run selected serving points with observational, per-call HBM diagnostics.

No numeric inputs, returned objects, exceptions, or execution order are changed.
The heartbeat observes long calls from a separate thread; no burst logging occurs.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import hbm_ramulator2_study as study


class ExecutionMonitor:
    def __init__(self, directory, interval=10.):
        self.directory = Path(directory)
        self.interval = interval
        self.started = time.perf_counter()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.local = threading.local()
        self.metrics = {}
        self.active = {}
        self.last_requests = deque(maxlen=20)
        self.slowest_requests = []
        self.physical_bytes = {"read": 0, "write": 0}
        self.physical_bytes_by_phase = {}
        self.patches = []
        self.status = "running"
        self.errors = []
        self.request_sequence = 0

    def _request_description(self, state, request):
        frame = sys._getframe(2)
        stack = []
        for _ in range(40):
            if frame is None:
                break
            stack.append(frame.f_code.co_name)
            frame = frame.f_back
        config = state.core.config
        first = request.address // config.burst_bytes
        stop = (request.address + request.byte_count - 1) // config.burst_bytes + 1
        return {"request_id": request.request_id, "operation": request.operation.value,
            "logical_bytes": request.byte_count, "physical_bytes": (stop-first)*config.burst_bytes,
            "arrival_ns": request.arrival_ns, "address": request.address,
            "core_object_id": id(state.core), "capture_details": state.core.capture_details,
            "phase": getattr(self.local, "phase", "other_or_live"),
            "call_stack_functions": stack,
            "scope": "all numeric invocations, including previews and initialization; not only committed serving"}

    def patch(self, owner, attribute, category, *, numeric=False, phase=None):
        original = getattr(owner, attribute)
        @wraps(original)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            previous_phase = getattr(self.local, "phase", "other_or_live")
            if phase is not None:
                self.local.phase = phase
            stack = getattr(self.local, "stack", None)
            if stack is None:
                self.local.stack = stack = []
            frame = {"nested_seconds": 0.}
            stack.append(frame)
            request_info = self._request_description(args[0], args[1]) if numeric else None
            with self.lock:
                self.request_sequence += bool(numeric)
                sequence = self.request_sequence
                if numeric:
                    request_info.update(sequence=sequence, wall_start_seconds=started-self.started)
                    self.active[sequence] = request_info
                metric = self.metrics.setdefault(category, {"calls": 0, "completed_calls": 0,
                    "exceptions": 0, "inclusive_wall_seconds": 0., "exclusive_wall_seconds": 0.})
                metric["calls"] += 1
            result = None
            exception_name = None
            try:
                result = original(*args, **kwargs)
                return result
            except BaseException as error:
                exception_name = type(error).__name__
                raise
            finally:
                ended = time.perf_counter()
                duration = ended-started
                stack.pop()
                self.local.phase = previous_phase
                if stack:
                    stack[-1]["nested_seconds"] += duration
                with self.lock:
                    metric["completed_calls"] += 1
                    metric["exceptions"] += exception_name is not None
                    metric["inclusive_wall_seconds"] += duration
                    metric["exclusive_wall_seconds"] += duration-frame["nested_seconds"]
                    if numeric:
                        self.active.pop(sequence, None)
                        request_info.update(wall_end_seconds=ended-self.started, wall_seconds=duration,
                            exception=exception_name, declined=result is None,
                            completion_ns=getattr(result, "completion_ns", None))
                        if result is not None:
                            operation = request_info["operation"]
                            moved = int(result.counters.get("physical_"+operation+"_bytes", 0))
                            self.physical_bytes[operation] += moved
                            phase_bytes = self.physical_bytes_by_phase.setdefault(request_info["phase"], {"read": 0, "write": 0})
                            phase_bytes[operation] += moved
                        self.last_requests.append(request_info)
                        self.slowest_requests.append(request_info)
                        self.slowest_requests.sort(key=lambda item: item["wall_seconds"], reverse=True)
                        del self.slowest_requests[20:]
                        self._append_request(request_info)
        setattr(owner, attribute, measured)
        self.patches.append((owner, attribute, original))

    def _append_request(self, record):
        try:
            self.request_log.write(json.dumps(record, ensure_ascii=False)+"\n")
        except Exception as error:
            self.errors.append(repr(error))

    def write_progress(self):
        try:
            with self.lock:
                elapsed = time.perf_counter()-self.started
                payload = {"status": self.status, "pid": os.getpid(), "elapsed_wall_seconds": elapsed,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "timing_semantics": "Inclusive call categories can nest; exclusive seconds subtract nested observed calls. Entry bookkeeping adds small overhead; final log writes and unwrapped outer paths are not included. Category sums are not whole-run wall time. In-flight calls have not entered completed totals.",
                    "byte_semantics": "Successful numeric invocations across all preview/initialization/live runtimes; repeated speculative work is counted. These are not committed serving bytes.",
                    "metrics": deepcopy(self.metrics), "physical_bytes": dict(self.physical_bytes),
                    "physical_bytes_by_phase": deepcopy(self.physical_bytes_by_phase),
                    "current_requests": [dict(item, elapsed_wall_seconds=elapsed-item["wall_start_seconds"])
                                         for item in self.active.values()],
                    "last_requests": list(self.last_requests), "slowest_requests": list(self.slowest_requests),
                    "monitor_errors": list(self.errors)}
                self.request_log.flush()
            temporary = self.directory / "progress.json.tmp"
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.directory / "progress.json")
        except Exception as error:
            self.errors.append(repr(error))

    def _heartbeat(self):
        while not self.stop_event.wait(self.interval):
            self.write_progress()

    def __enter__(self):
        from heterollm_sim import _compiled_hbm, data_motion, runtime, planner
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request_log = (self.directory / "numeric_requests.jsonl").open("w", encoding="utf-8")
        self.patch(_compiled_hbm, "_pack_hbm_numeric", "packing")
        self.patch(_compiled_hbm._HBMNumericState, "execute", "numeric_execute", numeric=True)
        self.patch(_compiled_hbm._HBMNumericState, "commit_controller", "commit_controller")
        self.patch(_compiled_hbm._HBMNumericState, "commit_timeline", "commit_timeline")
        self.patch(data_motion.PhysicalRuntimeContext, "snapshot", "physical_runtime_snapshot")
        self.patch(data_motion.PhysicalRuntimeContext, "restore", "physical_runtime_restore")
        self.patch(data_motion, "_copy_hbm_control_state", "hbm_state_copy")
        self.patch(runtime, "deepcopy", "control_runtime_deepcopy")
        self.patch(planner, "execute_cost_schedule", "isolated_cohort_schedule", phase="isolated_cohort")
        self.write_progress()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.stop_event.set()
        self.thread.join()
        self.status = "failed" if exception_type else "completed"
        if exception_type:
            self.errors.append(repr(exception))
        for owner, attribute, original in reversed(self.patches):
            setattr(owner, attribute, original)
        self.write_progress()
        self.request_log.close()


def monitored_point(point, output_dir, interval=10.):
    variant, architecture, model, prompt, batch, _ = point
    point_id = f"{model}-{prompt}-{batch}-512-{variant}-{architecture}"
    with ExecutionMonitor(Path(output_dir)/"progress"/point_id, interval):
        row = study.serving(*point)
    study.save(Path(output_dir)/"serving"/(row["id"]+".json"), row)
    return {"id": row["id"], "tpot_ns": row["tpot_ns"]["p50"], "wall_seconds": row["wall_seconds"]}


def self_check(output_dir):
    point = ("reference", "A_hbm_only", "small", 128, 1, 4)
    baseline = study.serving(*point)
    with ExecutionMonitor(Path(output_dir)/"self_check_progress", .1):
        observed = study.serving(*point)
    baseline_wall, observed_wall = baseline.pop("wall_seconds"), observed.pop("wall_seconds")
    if baseline != observed:
        study.save(Path(output_dir)/"self_check_mismatch.json", {"baseline": baseline, "monitored": observed})
        raise AssertionError("Monitoring changed a simulation result")
    result = {"passed": True, "point": point, "excluded_fields": ["wall_seconds"],
              "baseline_wall_seconds": baseline_wall, "monitored_wall_seconds": observed_wall}
    study.save(Path(output_dir)/"self_check.json", result)
    return result


def source_description(note):
    files = ["src/heterollm_sim/_hbm_numeric.py", "src/heterollm_sim/_compiled_hbm.py",
             "src/heterollm_sim/data_motion.py", "scripts/hbm_ramulator2_study.py", "scripts/run_hbm_monitored.py"]
    return {"started_at": datetime.now(timezone.utc).isoformat(), "description": note,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_files": {name: {"modified_at": datetime.fromtimestamp((ROOT/name).stat().st_mtime, timezone.utc).isoformat(),
                                     "size_bytes": (ROOT/name).stat().st_size} for name in files},
        "scope": "Version notes for this run; source may include uncommitted changes. No content digest or frozen source copy."}


def require_new_round(directory):
    directory = Path(directory)
    markers = ("plan.json", "source_version.json", "progress", "completion.json", "interruption.json")
    if any((directory/name).exists() for name in markers) or any((directory/"serving").glob("*.json")):
        raise ValueError("Use a new round directory; existing run plans, progress and results must not be overwritten")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="llama3_1-70b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--interval", type=float, default=10.)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--source-note", default="Validated exact HBM performance optimizations; observational per-call timing.")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(args.output_dir)), flush=True)
        return
    models = args.models.split(",")
    if any(model not in study.LARGE_MODELS for model in models):
        parser.error("unknown large model")
    if not 1 <= args.workers <= 8 or args.interval <= 0:
        parser.error("workers must be 1..8 and interval must be positive")
    points = study.planned_serving_points(["reference", "ramulator2"], "large", models)
    try:
        require_new_round(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    study.save(args.output_dir/"source_version.json", source_description(args.source_note))
    study.save(args.output_dir/"plan.json", {"points": points, "workers": args.workers, "heartbeat_seconds": args.interval})
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(monitored_point, point, args.output_dir, args.interval) for point in points]
        for future in as_completed(futures):
            completed.append(future.result())
            study.save(args.output_dir/"completion.json", {"status": "complete" if len(completed)==len(points) else "running",
                       "completed": completed, "planned_points": len(points)})
            print(json.dumps(completed[-1]), flush=True)


if __name__ == "__main__":
    main()
