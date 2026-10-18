"""Validate opt-in fast forecasts against complete same-budget detailed requests.

This diagnostic does not start or resume the formal scan.  The suite starts
separate simulation processes in a fixed serial order and never substitutes a
short-output request for the detailed reference.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
CASE_FIELDS = ("architecture", "model_id", "prompt_tokens", "output_tokens")


class FastForecastUnavailable(RuntimeError):
    """The diagnostic required a forecast, so do not rerun a full reference."""


def bound_runtime_command(argv):
    """Honor an explicit per-experiment runtime binding before any imports.

    A long validation may span source edits.  Its already-running manager
    continues launching this entry point, so the experiment directory binds
    later workers to the same retained runtime as its first workers.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--result", type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.result is None:
        return None
    binding_path = args.result.resolve().parent / "runtime_binding.json"
    if not binding_path.exists():
        return None
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("schema") != "fast-validation-runtime/v1":
        raise ValueError("unsupported validation runtime binding")
    runtime_root = Path(binding["runtime_root"]).resolve(strict=True)
    script = runtime_root / "scripts" / "benchmark_fast_mode.py"
    if script.resolve() == Path(__file__).resolve():
        return None
    if not script.is_file() or not (runtime_root / "src" / "heterollm_sim" / "__init__.py").is_file():
        raise ValueError("retained validation runtime is incomplete")
    return [sys.executable, "-u", str(script), *argv]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                             encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.1)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_request_endpoint(result) -> dict:
    """Read the same public latency boundaries without constructing a report.

    Ordinary scan requests use arrival-scoped request metrics. Engine-scoped
    requests additionally need the last visible commit, exactly as the public
    report does; no batch metadata or resource summaries are visited.
    """
    approximation = dict(getattr(result, "approximation", {"effective_mode": "detailed"}))
    forecast = getattr(result, "forecast", None)
    if forecast is not None:
        request = {"request_id": forecast.request_id, "status": "forecast_completed",
            "arrival_ns": forecast.arrival_ns, "first_token_ns": forecast.first_token_ns,
            "finish_ns": forecast.finish_ns, "ttft_ns": forecast.ttft_ns,
            "tpot_ns": forecast.tpot_ns, "e2e_ns": forecast.finish_ns - forecast.arrival_ns,
            "visible_output_tokens": forecast.output_tokens, "latency_boundary": "arrival"}
        semantics = {"precision": "detailed_prefix_with_forecast_suffix",
            "ttft": "detailed", "tpot": "estimated_mean",
            "transactions": "only_sampled_prefix_executed"}
        fidelity = "approximate"
    else:
        detailed = getattr(result, "detailed", None) or result
        serving = getattr(detailed, "serving", None)
        if serving is None or len(serving.request_metrics) != 1:
            raise ValueError("diagnostic requires exactly one online request")
        metric = next(iter(serving.request_metrics.values()))
        status = getattr(metric.status, "value", metric.status)
        engine_start = metric.engine_start_ns
        boundary = "engine" if status == "finished" and engine_start is not None else "arrival"
        ttft, tpot = metric.ttft_ns, metric.tpot_ns
        end = metric.finish_ns
        e2e = end - metric.arrival_ns if end is not None else None
        if boundary == "engine":
            last_token = next((event.timestamp_ns for event in reversed(serving.events)
                if event.event_type == "tokens_committed"
                and event.request_id == metric.request_id
                and max(0, int(event.details.get("visible_tokens", 0))) > 0), None)
            ttft = metric.first_token_ns - engine_start if metric.first_token_ns is not None else None
            tpot = ((last_token - metric.first_token_ns) / (metric.visible_output_tokens - 1)
                if last_token is not None and metric.first_token_ns is not None
                and metric.visible_output_tokens > 1 else None)
            e2e = last_token - engine_start if last_token is not None else None
        request = {"request_id": metric.request_id, "status": status,
            "arrival_ns": metric.arrival_ns, "first_token_ns": metric.first_token_ns,
            "finish_ns": end, "ttft_ns": ttft, "tpot_ns": tpot, "e2e_ns": e2e,
            "visible_output_tokens": metric.visible_output_tokens, "latency_boundary": boundary}
        semantics = {"precision": "detailed", "latency": {"primary_boundary": boundary},
            "transactions": "complete_detailed_execution"}
        fidelity = "aggregate"
    return {"request": request, "approximation": approximation,
        "report": {"requests": {request["request_id"]: request},
            "simulation_mode": "fast" if hasattr(result, "approximation") else "detailed",
            "result_semantics": semantics, "trace_fidelity": fidelity,
            "projection_scope": "request_endpoint_only; resource_and_batch_reports_not_constructed"}}


class BenchmarkProgress:
    """Bounded wall-clock observations at existing public cohort boundaries."""

    def __init__(self, args, *, clock=time.perf_counter):
        self.args, self.clock = args, clock
        self.process_start = self.start = clock()
        self.last_publish = -math.inf
        self.last_cohort_wall = self.serving_start_wall = self.last_prefill = None
        self.prefill_steps = self.decode_steps = 0
        self.phase = "setup"
        self.stage = "starting"
        self.status = "starting"
        self.fallback_reasons = []
        self.approximation = {}
        self.calibration_attempts = []
        self.recent_decode = deque(maxlen=16)
        self.windows = deque(maxlen=16)
        self.window_count = 0
        self.phase_wall = {"prefill": 0.0, "decode": 0.0}
        self.next_stop_check = -math.inf
        self.stop_requested = False

    def begin_simulation(self):
        self.start = self.clock()
        self.last_publish = -math.inf
        self.status = "running"
        self.publish(force=True)

    def observe(self, event):
        now = self.clock()
        wall = now - self.start
        self.stage = event.stage
        force = False
        # Capture diagnostics before callbacks can reject a forecast. The
        # simulator keeps extending these objects as it samples, so retain
        # independent values for both progress and failed-run results.
        if "approximation" in event.metadata:
            self.approximation = deepcopy(event.metadata["approximation"])
        if event.stage == "fast_calibration" and "calibration_attempt" in event.metadata:
            self.calibration_attempts.append(deepcopy(event.metadata["calibration_attempt"]))
        if event.stage == "serving_cohorts":
            if not event.completed:
                self.serving_start_wall = wall
                self.last_cohort_wall = wall
            else:
                kind = event.metadata.get("cohort_kind")
                if kind in self.phase_wall:
                    force = kind != self.phase
                    self.phase = kind
                    duration = (wall - self.last_cohort_wall
                                if self.last_cohort_wall is not None else None)
                    if duration is not None:
                        self.phase_wall[kind] += duration
                    if kind == "prefill":
                        self.prefill_steps += 1
                        self.last_prefill = wall
                    else:
                        self.decode_steps += 1
                        if duration is not None:
                            self.recent_decode.append(duration)
                            if self.decode_steps % 16 == 0:
                                self.window_count += 1
                                self.windows.append({"first_decode_step": self.decode_steps - 15,
                                    "last_decode_step": self.decode_steps,
                                    "mean_step_wall_seconds": math.fsum(self.recent_decode) / 16,
                                    "end_wall_seconds": wall})
                    self.last_cohort_wall = wall
        if event.stage == "fast_mode_fallback":
            self.fallback_reasons = list(event.metadata.get("fallback_reasons", ()))
            force = True
        if event.stage.startswith("fast_") or event.stage == "serving_complete":
            force = True
        self.publish(force=force, now=now)
        if event.stage == "fast_mode_fallback" and getattr(self.args, "require_forecast", False):
            # simulate_fast emits this event before runtime.run().  A candidate
            # diagnostic already has the complete detailed reference; reject
            # the candidate rather than accidentally repeat hours of Decode.
            raise FastForecastUnavailable(
                "fast forecast rejected: " + ", ".join(self.fallback_reasons))

    def snapshot(self, *, now=None):
        now = self.clock() if now is None else now
        return {"model_id": self.args.model, "architecture": self.args.architecture,
            "requested_mode": self.args.mode, "prompt_tokens": self.args.prompt,
            "output_tokens": self.args.output, "full_output_tokens": self.args.output,
            "total_decode_steps": self.args.output - 1, "process_id": os.getpid(),
            "status": self.status, "phase": self.phase, "stage": self.stage,
            "prefill_steps": self.prefill_steps, "decode_steps": self.decode_steps,
            "wall_seconds": now - self.start, "process_wall_seconds": now - self.process_start,
            "wall_scope": "simulation_excludes_imports_and_scenario_build",
            "recent_decode_steps": len(self.recent_decode),
            "recent_16_decode_step_mean_seconds": (
                math.fsum(self.recent_decode) / len(self.recent_decode) if self.recent_decode else None),
            "windows": list(self.windows), "window_count": self.window_count,
            "windows_retained": "last_16_nonoverlapping_16_step_wall_windows",
            "fallback_reasons": self.fallback_reasons,
            "approximation": deepcopy(self.approximation),
            "calibration_attempts": deepcopy(self.calibration_attempts),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "limited fast-mode diagnostics; not a formal scan"}

    def publish(self, *, force=False, now=None, extra=None):
        now = self.clock() if now is None else now
        if not force and now - self.last_publish < 10:
            return
        path = getattr(self.args, "progress_file", None)
        if path is not None:
            write_json(path, {**self.snapshot(now=now), **(extra or {})})
        self.last_publish = now

    def finish(self, status, **extra):
        self.status, self.phase, self.stage = status, status, status
        self.publish(force=True, extra=extra)

    def cancellation_requested(self):
        # A filesystem probe at every physical task would distort the timing
        # being measured. Cancellation remains cooperative at kernel checks;
        # only the optional external marker is polled, at most twice a second.
        if self.stop_requested:
            return True
        now = self.clock()
        if now >= self.next_stop_check:
            self.next_stop_check = now + .5
            self.stop_requested = self.args.stop_file.exists()
        return self.stop_requested

    def timing_summary(self, wall, extraction_wall=0.0, result=None):
        manifest = getattr(result, "manifest", None)
        metadata = getattr(manifest, "metadata", {})
        return {"measurement": "existing_public_progress_boundaries; no_profiler_or_kernel_hooks",
            "setup_to_serving_start_seconds": self.serving_start_wall,
            "prefill_cohorts_wall_seconds": self.phase_wall["prefill"],
            "decode_cohorts_wall_seconds": self.phase_wall["decode"],
            "after_last_cohort_seconds": (wall - self.last_cohort_wall
                if self.last_cohort_wall is not None else None),
            "endpoint_extraction_seconds": extraction_wall,
            "prefill_steps": self.prefill_steps, "decode_steps": self.decode_steps,
            "control_plane": metadata.get("control_plane", {}),
            "wall_to_first_token_seconds": self.last_prefill,
            "decode_window_scope": "cohort completion intervals including public callback overhead"}


def positive(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def relative_error(reference, estimate):
    if not positive(reference) or not positive(estimate):
        return None
    return abs(estimate - reference) / reference


def compare_case(detailed: dict, fast: dict, target: float = .10) -> dict:
    """Use endpoint metrics only; a forecast is never transaction equivalence."""
    reasons = []
    case = {key: detailed.get(key) for key in CASE_FIELDS}
    if any(detailed.get(key) != fast.get(key) for key in CASE_FIELDS):
        reasons.append("different_full_request_specification")
    if detailed.get("requested_mode") != "detailed" or fast.get("requested_mode") != "fast":
        reasons.append("incorrect_reference_or_candidate_mode")
    if detailed.get("latency_boundary") != fast.get("latency_boundary"):
        reasons.append("different_public_latency_boundary")
    output = detailed.get("output_tokens")
    if (detailed.get("request_status") != "finished"
            or fast.get("request_status") not in {"finished", "forecast_completed"}
            or detailed.get("visible_output_tokens") != output
            or fast.get("visible_output_tokens") != output):
        reasons.append("incomplete_request_or_wrong_output_budget")
    errors = {name: relative_error(detailed.get(name), fast.get(name))
              for name in ("ttft_ns", "tpot_ns")}
    if any(value is None for value in errors.values()):
        reasons.append("invalid_latency_metric")
    matched_timing = (not detailed.get("instrumented", True)
                      and not fast.get("instrumented", True)
                      and detailed.get("affinity") == fast.get("affinity")
                      and detailed.get("gc_thresholds") == fast.get("gc_thresholds"))
    if not matched_timing:
        reasons.append("unmatched_or_instrumented_timing_environment")
    valid = not reasons
    within = valid and all(value <= target for value in errors.values())
    approximation = fast.get("approximation", {})
    def speedup(name):
        left, right = detailed.get(name), fast.get(name)
        return left / right if valid and positive(left) and positive(right) else None
    return {
        **case, "target_relative_error": target,
        "valid_comparison": valid, "within_target": within,
        "relative_errors": errors, "invalid_reasons": reasons,
        "latency_ns": {name: {"detailed": detailed.get(name), "fast": fast.get(name)}
                       for name in ("ttft_ns", "tpot_ns")},
        "wall_seconds": {"detailed": detailed.get("wall_seconds"),
                         "fast": fast.get("wall_seconds")},
        "speedup": {"total_wall": speedup("wall_seconds"),
                    "decode_wall": speedup("decode_wall_seconds")},
        "effective_mode": approximation.get("effective_mode"),
        "sampled_decode_steps": approximation.get("sampled_decode_steps"),
        "predicted_decode_steps": approximation.get("predicted_decode_steps"),
        "fallback_reasons": approximation.get("fallback_reasons", []),
        "prefix_validation_relative_error": approximation.get("prefix_validation_relative_error"),
        "model_relative_half_width": approximation.get("model_relative_half_width"),
        "full_request_validation_status": (
            "within_target_for_this_request" if within else "failed_or_invalid_for_this_request"),
        "transaction_equivalence_claimed": False,
    }


def compare_pair(rows: list[dict], target: float = .10) -> dict:
    indexed = {(row["architecture"], row["requested_mode"]): row for row in rows}
    required = {(arch, mode) for arch in ("HBM", "HBF") for mode in ("detailed", "fast")}
    if len(rows) != 4 or set(indexed) != required:
        raise ValueError("comparison requires exactly one HBM/HBF result in each mode")
    cases = {arch: compare_case(indexed[arch, "detailed"], indexed[arch, "fast"], target)
             for arch in ("HBM", "HBF")}
    matching = all(indexed["HBM", "detailed"].get(key) == indexed["HBF", "detailed"].get(key)
                   for key in CASE_FIELDS if key != "architecture")
    ratios = {}
    for metric in ("ttft_ns", "tpot_ns"):
        values = {}
        for mode in ("detailed", "fast"):
            hbm, hbf = indexed["HBM", mode].get(metric), indexed["HBF", mode].get(metric)
            values[mode] = hbf / hbm if positive(hbm) and positive(hbf) else None
        error = relative_error(values["detailed"], values["fast"])
        valid = matching and all(case["valid_comparison"] for case in cases.values())
        ratios[metric] = {"hbf_over_hbm": values, "relative_error": error,
                          "within_target": valid and error is not None and error <= target}
    return {"purpose": "fast mode diagnostics; not a formal scan",
            "target_relative_error": target, "cases": cases,
            "ratio_comparison": ratios,
            "all_within_target": (matching and all(case["within_target"] for case in cases.values())
                                  and all(row["within_target"] for row in ratios.values())),
            "accuracy_scope": "these complete requests only; no global error guarantee",
            "speedup_scope": "separate serial processes; same affinity and full output budget"}


def set_affinity(cpu: int | None):
    if cpu is None:
        return None
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << cpu):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(0, {cpu})
    return {"logical_cpu": cpu}


def benchmark_case(args) -> dict:
    progress_state = BenchmarkProgress(args)
    progress_state.publish(force=True)
    try:
        return _benchmark_case(args, progress_state)
    except BaseException as error:
        status = ("cancelled" if isinstance(error, KeyboardInterrupt)
            or type(error).__name__ == "ExecutionCancelledError" else "failed")
        failure = {"error_type": type(error).__name__, "reason": str(error)}
        wall = progress_state.clock() - progress_state.start
        failure["timing_breakdown"] = progress_state.timing_summary(wall)
        failure["request_status"] = status
        progress_state.finish(status, **failure)
        if getattr(args, "result", None) is not None:
            write_json(args.result, {**progress_state.snapshot(), **failure,
                "finished_at": datetime.now(timezone.utc).isoformat()})
        raise


def _benchmark_case(args, progress_state) -> dict:
    # Import/preload before timing, identically in both processes.  This does
    # not hide a JIT compile if the actual scenario introduces a new signature.
    import benchmark_scan_runtime  # noqa: F401
    from dataclasses import asdict, replace
    from b200_hbm_hbf_scan import scenario_gc_thresholds
    from b200_hbm_hbf_scan_config import build_scan_scenario
    from heterollm_sim.execution_control import ExecutionControl
    from heterollm_sim.reporting import run_scenario

    affinity = set_affinity(args.cpu)
    scenario = build_scan_scenario(args.model, args.architecture, args.prompt, args.output)
    scenario = replace(scenario, simulation=simulation_options(args))
    progress_state.begin_simulation()
    start = progress_state.start
    cpu_start = time.process_time()
    def progress(event):
        progress_state.observe(event)
        if event.stage.startswith("fast_"):
            print(json.dumps({"architecture": args.architecture, "mode": args.mode,
                "stage": event.stage, "completed": event.completed,
                "metadata": dict(event.metadata), "wall_seconds": time.perf_counter() - start},
                ensure_ascii=False), flush=True)
        if event.stage == "serving_cohorts" and event.completed:
            kind = event.metadata.get("cohort_kind")
            if kind != "decode" or event.completed % 16 == 0:
                print(json.dumps({"architecture": args.architecture, "mode": args.mode,
                    "stage": event.stage, "completed": event.completed, "kind": kind,
                    "wall_seconds": time.perf_counter() - start}), flush=True)
    with scenario_gc_thresholds() as thresholds:
        result = run_scenario(scenario, retention_policy="aggregate",
            control=ExecutionControl(progress_callback=progress,
                cancellation_callback=progress_state.cancellation_requested
                    if getattr(args, "stop_file", None) is not None else None))
    wall = time.perf_counter() - start
    cpu_seconds = time.process_time() - cpu_start
    report_start = time.perf_counter()
    endpoint = extract_request_endpoint(result)
    extraction_wall = time.perf_counter() - report_start
    request = endpoint["request"]
    last_prefill = progress_state.last_prefill
    row = {"architecture": args.architecture, "model_id": args.model,
        "prompt_tokens": args.prompt, "output_tokens": args.output,
        "requested_mode": args.mode, "simulation_config": asdict(scenario.simulation),
        "request_id": request["request_id"], "request_status": request["status"],
        "visible_output_tokens": request["visible_output_tokens"],
        "ttft_ns": request["ttft_ns"], "tpot_ns": request["tpot_ns"],
        "latency_boundary": request["latency_boundary"],
        "e2e_ns": request["e2e_ns"], "wall_seconds": wall, "cpu_seconds": cpu_seconds,
        "decode_wall_seconds": wall - last_prefill if last_prefill is not None else None,
        "wall_to_first_token_seconds": last_prefill,
        "decode_wall_scope": "after final Prefill cohort through run_scenario return; includes reduction/forecast",
        "report_wall_seconds": extraction_wall,
        "report_scope": "request_endpoint_only; full_report_not_constructed",
        "affinity": affinity, "gc_thresholds": list(thresholds),
        "instrumented": False, "process_id": os.getpid(),
        "approximation": endpoint["approximation"],
        "calibration_attempts": deepcopy(progress_state.calibration_attempts),
        # Endpoint validation does not need hundreds of repeated batch task
        # payloads. Retain the public request endpoint, not a second
        # detailed transaction archive disguised as a benchmark result.
        "report": endpoint["report"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "fast mode complete request diagnostics; not a formal scan"}
    if getattr(args, "report_timing", False):
        row["timing_breakdown"] = progress_state.timing_summary(wall, extraction_wall, result)
        print(json.dumps({"timing_breakdown": row["timing_breakdown"]}, ensure_ascii=False), flush=True)
    write_json(args.result, row)
    progress_state.approximation = deepcopy(row["approximation"])
    progress_state.finish("completed", result_file=str(args.result),
        effective_mode=row["approximation"].get("effective_mode"),
        predicted_decode_steps=row["approximation"].get("predicted_decode_steps", 0),
        request_status=row["request_status"], visible_output_tokens=row["visible_output_tokens"],
        ttft_ns=row["ttft_ns"], tpot_ns=row["tpot_ns"])
    summary = {key: row[key] for key in ("architecture", "requested_mode", "ttft_ns",
        "tpot_ns", "wall_seconds")}
    summary["approximation"] = {key: row["approximation"].get(key) for key in (
        "effective_mode", "sampled_decode_steps", "predicted_decode_steps", "fallback_reasons")}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return row


def simulation_options(args):
    from heterollm_sim.simulation_options import FastDecodeConfig, SimulationConfig
    options = {"target_relative_error": args.target}
    if getattr(args, "sampling_strategy", None) is not None:
        options["sampling_strategy"] = args.sampling_strategy
    if getattr(args, "max_sample_tokens", None) is not None:
        options["max_sample_tokens"] = args.max_sample_tokens
    return SimulationConfig(mode=args.mode, fast_decode=FastDecodeConfig(**options))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2_5-0_5b")
    parser.add_argument("--prompt", type=int, default=2048)
    parser.add_argument("--output", type=int, default=257)
    parser.add_argument("--cpu", type=int,
                        help="Optional logical CPU affinity; default keeps native channel threads parallel")
    parser.add_argument("--target", type=float, default=.10)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts" / "fast_mode_20261017")
    parser.add_argument("--mode", choices=("detailed", "fast"))
    parser.add_argument("--architecture", choices=("HBM", "HBF"))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--progress-file", type=Path,
                        help="Atomically publish bounded progress at most every 10 seconds")
    parser.add_argument("--stop-file", type=Path,
                        help="Cooperatively cancel when this path exists; never remove it")
    parser.add_argument("--report-timing", action="store_true",
                        help="Emit timing from existing public cohort boundaries and manifest metadata")
    parser.add_argument("--sampling-strategy", choices=("adaptive", "fixed"),
                        help="Explicit fast-mode sampling strategy; the complete request budget is unchanged")
    parser.add_argument("--max-sample-tokens", type=int,
                        help="Explicit real Decode sampling cap; default remains 128")
    parser.add_argument("--require-forecast", action="store_true",
                        help="Reject a fast candidate before full detailed fallback; use an existing reference")
    parser.add_argument("--compare", nargs=4, type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.target <= .10 or not math.isfinite(args.target):
        parser.error("--target must be in (0, .10]")
    if args.prompt <= 0 or args.output <= 1 or (args.cpu is not None and args.cpu < 0):
        parser.error("positive prompt, output > 1 and nonnegative CPU are required")
    if args.max_sample_tokens is not None and args.max_sample_tokens <= 0:
        parser.error("--max-sample-tokens must be positive")
    if args.compare:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.compare]
    elif args.mode:
        if not args.architecture or not args.result:
            parser.error("a single run requires --mode, --architecture and --result")
        if args.require_forecast and args.mode != "fast":
            parser.error("--require-forecast requires --mode fast")
        benchmark_case(args)
        return 0
    else:
        rows = []
        for architecture in ("HBM", "HBF"):
            for mode in ("detailed", "fast"):
                path = args.directory / f"{args.model}_{architecture}_P{args.prompt}_D{args.output}_{mode}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--model", args.model,
                    "--prompt", str(args.prompt), "--output", str(args.output),
                    "--mode", mode, "--architecture", architecture, "--result", str(path),
                    "--target", str(args.target)]
                if args.cpu is not None:
                    command.extend(("--cpu", str(args.cpu)))
                if args.progress_file is not None:
                    command.extend(("--progress-file", str(args.progress_file)))
                if args.stop_file is not None:
                    command.extend(("--stop-file", str(args.stop_file)))
                if args.report_timing:
                    command.append("--report-timing")
                if args.sampling_strategy is not None:
                    command.extend(("--sampling-strategy", args.sampling_strategy))
                if args.max_sample_tokens is not None:
                    command.extend(("--max-sample-tokens", str(args.max_sample_tokens)))
                if args.require_forecast and mode == "fast":
                    command.append("--require-forecast")
                subprocess.run(command, cwd=ROOT, check=True)
                rows.append(json.loads(path.read_text(encoding="utf-8")))
                write_json(args.directory / "comparison_partial.json", {
                    "completed_runs": len(rows), "total_runs": 4,
                    "runs": [{key: row[key] for key in (*CASE_FIELDS, "requested_mode", "wall_seconds", "approximation")}
                             for row in rows]})
    comparison = compare_pair(rows, args.target)
    path = args.result or args.directory / f"{args.model}_P{args.prompt}_D{args.output}_comparison.json"
    write_json(path, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return 0 if comparison["all_within_target"] else 1


if __name__ == "__main__":
    retained_command = bound_runtime_command(sys.argv[1:])
    if retained_command is not None:
        print(json.dumps({"validation_runtime_script": retained_command[2]}, ensure_ascii=False), flush=True)
        raise SystemExit(subprocess.call(retained_command, cwd=ROOT))
    raise SystemExit(main())
