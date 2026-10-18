"""Measure mutually exclusive Decode wall intervals without changing simulation.

This is a diagnostic run, not a throughput benchmark.  The ordinary benchmark
result is written separately so its complete simulation fields can be compared.
Only public Python call boundaries are wrapped; native internals stay untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import wraps
import importlib
import inspect
import json
from pathlib import Path
import sys
import threading
import time


class WallCollector:
    """A main-thread nested interval stack, using integer nanoseconds."""

    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.thread_id = threading.get_ident()
        self.enabled = False
        self.stack = []
        self.stats = {}
        self.cohorts = []
        self.boundaries = []
        self._patches = []
        self._wrappers = {}
        self.interval_start_ns = None
        self.observations = {}

    def wrap(self, function, label, category):
        key = (id(function), label, category)
        if key in self._wrappers:
            return self._wrappers[key]

        @wraps(function)
        def measured(*args, **kwargs):
            if not self.enabled or threading.get_ident() != self.thread_id:
                return function(*args, **kwargs)
            started = self.clock()
            effective_category = ("preview_schedule" if self.stack and self.stack[-1][1] == "preview_schedule"
                                  else category)
            frame = [0, effective_category]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = self.clock() - started
                self.stack.pop()
                if self.stack:
                    self.stack[-1][0] += elapsed
                exclusive = elapsed - frame[0]
                if exclusive < 0:
                    raise RuntimeError("nested timer produced a negative exclusive interval")
                row = self.stats.setdefault((label, effective_category), [effective_category, 0, 0, 0])
                row[1] += 1
                row[2] += elapsed
                row[3] += exclusive

        self._wrappers[key] = measured
        return measured

    def patch(self, owner, attribute, category, *, label=None):
        label = label or (getattr(owner, "__name__", type(owner).__name__) + "." + attribute)
        original = getattr(owner, attribute, None)
        descriptor = inspect.getattr_static(owner, attribute, None)
        boundary = {"label": label, "category": category,
                    "hooked": False, "mechanism": "public Python call boundary"}
        self.boundaries.append(boundary)
        if original is None or not callable(original):
            boundary["reason"] = "boundary absent in selected implementation"
            return
        if isinstance(descriptor, (staticmethod, classmethod)):
            replacement = type(descriptor)(self.wrap(descriptor.__func__, label, category))
        else:
            replacement = self.wrap(original, label, category)
        try:
            setattr(owner, attribute, replacement)
        except (AttributeError, TypeError) as error:
            boundary["reason"] = type(error).__name__ + ": " + str(error)
            return
        self._patches.append((owner, attribute, descriptor))
        boundary["hooked"] = True

    def restore(self):
        for owner, attribute, original in reversed(self._patches):
            setattr(owner, attribute, original)
        self._patches.clear()

    def complete_cohort(self, now_ns, completed, kind, simulated_time_ns):
        if self.stack:
            raise RuntimeError("cohort completed inside an instrumented call")
        if self.interval_start_ns is not None and kind == "decode":
            wall_ns = now_ns - self.interval_start_ns
            category_ns = defaultdict(int)
            functions = []
            for (label, _category), (category, calls, inclusive, exclusive) in self.stats.items():
                category_ns[category] += exclusive
                functions.append({"function": label, "category": category, "calls": calls,
                                  "inclusive_ns": inclusive, "exclusive_ns": exclusive})
            measured_ns = sum(category_ns.values())
            if measured_ns > wall_ns:
                raise RuntimeError("instrumented intervals exceed complete cohort wall time")
            category_ns["unattributed"] = wall_ns - measured_ns
            categories = [{"category": category, "exclusive_ns": value,
                           "percent": 100.0 * value / wall_ns if wall_ns else 0.0}
                          for category, value in sorted(category_ns.items(), key=lambda item: -item[1])]
            self.cohorts.append({"completed": completed, "decode_index": len(self.cohorts) + 1,
                                 "window": "first" if not self.cohorts else "steady",
                                 "wall_ns": wall_ns, "simulated_time_ns": simulated_time_ns,
                                 "classified_ns": measured_ns,
                                 "unattributed_ns": wall_ns - measured_ns,
                                 "closure_error_ns": wall_ns - sum(row["exclusive_ns"] for row in categories),
                                 "observations": self.observations,
                                 "categories": categories,
                                 "functions": sorted(functions, key=lambda row: -row["exclusive_ns"])})
        self.stats.clear()
        self.observations = {}
        self.interval_start_ns = now_ns

    def result(self):
        windows = {}
        all_calls = defaultdict(int)
        for window in ("first", "steady", "all"):
            cohorts = [row for row in self.cohorts if window == "all" or row["window"] == window]
            wall_ns = sum(row["wall_ns"] for row in cohorts)
            categories = defaultdict(int)
            functions = {}
            for cohort in cohorts:
                for row in cohort["categories"]:
                    categories[row["category"]] += row["exclusive_ns"]
                for row in cohort["functions"]:
                    target = functions.setdefault((row["function"], row["category"]), {**row, "calls": 0,
                        "inclusive_ns": 0, "exclusive_ns": 0})
                    for key in ("calls", "inclusive_ns", "exclusive_ns"):
                        target[key] += row[key]
            windows[window] = {"steps": len(cohorts), "wall_ns": wall_ns,
                "seconds_per_step": wall_ns / 1e9 / len(cohorts) if cohorts else None,
                "closure_error_ns": wall_ns - sum(categories.values()),
                "categories": [{"category": category, "exclusive_ns": value,
                    "percent": 100.0 * value / wall_ns if wall_ns else 0.0}
                    for category, value in sorted(categories.items(), key=lambda item: -item[1])],
                "functions": sorted(functions.values(), key=lambda row: -row["exclusive_ns"])}
        for row in windows["all"]["functions"]:
            all_calls[row["function"]] += row["calls"]
        return {"profile_version": 2,
                "scope": "Decode cohort wall intervals; main thread exclusive nested timers",
                "timing_valid_for_speedup": False,
                "units": "integer wall nanoseconds",
                "coverage_notes": [
                    "Categories sum exactly to cohort wall time, including explicit unattributed residual.",
                    "Zero closure error proves interval accounting only; call counts disclose hook coverage.",
                    "Unattributed includes scheduler gaps and the preceding cohort report/printing overhead.",
                    "Native Cython internals are charged to their nearest instrumented public boundary.",
                    "HBM numeric wall includes channel launch and waiting; parallel worker times are never added.",
                    "HOST/DRAM single and batch numeric work is included in its adapter boundary; Numba globals are untouched.",
                    "NAND bulk numeric includes its authored per-page fabric link reservations.",
                    "NAND scalar Python execution is included in nand_adapter, with state publication split out.",
                    "First Decode includes graph capture and binding warmup; steady means all subsequent Decode steps.",
                    "Timing hooks add overhead; compare a separate uninstrumented run before interpreting percentages.",
                    "Hook presence does not prove a call path used it; each boundary reports observed Decode calls.",
                ], "windows": windows, "cohorts": self.cohorts,
                "boundaries": [{**row, "observed_decode_calls": all_calls[row["label"]]}
                               for row in self.boundaries]}


def install_boundaries(collector):
    modules = {name: importlib.import_module("heterollm_sim." + name) for name in (
        "serving", "planner", "scalable_serving", "data_motion", "event_kernel",
        "_compiled_hbm", "_hbm_numeric", "_compiled_dram", "_compiled_nand",
        "_nand_execution", "_io_execution", "_token_binding", "_physical_binding",
        "hbf_staging", "hbf_staging_runtime", "live_execution_metrics", "nand_core")}

    def module(name, attribute, category, *, label=None):
        collector.patch(modules[name], attribute, category, label=label)

    def methods(name, class_name, attributes, category):
        owner = getattr(modules[name], class_name)
        for attribute in attributes:
            collector.patch(owner, attribute, category, label=name + "." + class_name + "." + attribute)

    module("serving", "_lower_cost", "planner_lowering_binding")
    for attribute in ("_estimate_serving_cohort_cost", "_lower_serving_cohort"):
        module("planner", attribute, "planner_lowering_binding")
    for attribute in ("_task_segment_clone_metadata", "_task_segment_clone_metadata_with_plan"):
        module("planner", attribute, "planner_metadata_copy")
    methods("_physical_binding", "StaticPhysicalBinding", ("bind",), "planner_metadata_copy")
    methods("planner", "_TaskSegmentTemplate", ("replay",), "planner_lowering_binding")
    module("planner", "execute_cost_schedule", "preview_schedule", label="scalable_serving.execute_cost_schedule")
    # Patch the real module too, without wrapping the planner alias twice.
    module("scalable_serving", "execute_cost_schedule", "preview_schedule", label="scalable_serving.execute_cost_schedule")
    for attribute in ("_physical_task_metadata", "_compact_execution_stages"):
        module("planner", attribute, "planner_stage_serialization")
    methods("serving", "_OnlineRuntime", ("_execute_cohort",), "serving_prepare_commit")
    methods("serving", "_OnlineRuntime", ("_schedule_execution_stages", "_replay_authored_token_program"), "live_dispatch")
    methods("serving", "_ExecutionStageMetadataCache", ("resolve", "authored_token_program", "_final_layout_key"), "live_stage_binding")
    methods("_token_binding", "AuthoredTokenProgram", ("bind",), "live_token_binding")
    methods("event_kernel", "UnifiedEventKernel", ("_drain_prevalidated_compiled",), "live_dispatch")
    methods("event_kernel", "UnifiedEventKernel", ("_resolve_task_execution",), "physical_dispatch")
    methods("event_kernel", "UnifiedEventKernel", ("_account_task_execution",), "physical_event_accounting")
    methods("data_motion", "PhysicalRuntimeContext", ("snapshot", "commit_transaction", "restore"), "physical_transaction_snapshot")
    for attribute in ("register_physical_allocations", "prepare_physical_task", "lower_physical_allocations"):
        module("data_motion", attribute, "physical_validation_lowering")
    for attribute in ("resolve_physical_task", "_execute_physical_task", "execute_prepared_physical_task"):
        module("data_motion", attribute, "physical_execution_accounting")
    methods("hbf_staging", "HbfStagingController", ("read", "write", "flush", "release"), "hbf_staging")
    methods("hbf_staging", "HbfStagingController", ("_read_full_pages", "_write_full_pages"), "hbf_page_mapping_and_state")
    methods("hbf_staging", "HbfStagingController", ("_transfer",), "hbf_link_accounting")
    methods("_io_execution", "_PhysicalIO", ("transfer", "_route"), "hbf_link_accounting")
    methods("_io_execution", "_PhysicalIO", ("_memory", "_flash_pages"), "hbf_io_adapter")
    for attribute in ("snapshot", "commit", "restore"):
        module("hbf_staging_runtime", attribute, "physical_transaction_snapshot")
    for attribute in ("execute_compiled", "execute_compiled_many"):
        module("_compiled_dram", attribute, "host_dram_adapter_and_numeric", label="_compiled_dram." + attribute)
    # Cython imported this function directly, so patch its stored alias as well.
    module("_io_execution", "execute_compiled", "host_dram_adapter_and_numeric", label="_compiled_dram.execute_compiled")
    for attribute in ("execute_hbm_numeric", "execute_hbm_numeric_many", "_pack_hbm_numeric"):
        module("_compiled_hbm", attribute, "hbm_adapter")
    methods("_compiled_hbm", "_HBMNumericState", ("execute", "refresh_resources", "commit_timeline", "commit_controller", "checkpoint"), "hbm_adapter")
    # Installed before any controller captures its run_numeric instance field.
    module("_hbm_numeric", "run_hbm_numeric", "hbm_numeric_and_thread_wait")
    module("_nand_execution", "execute_pages", "nand_adapter")
    methods("nand_core", "NandCore", ("submit", "submit_read_pages", "submit_write_pages", "_submit_ocp_single_page_summary"), "nand_adapter")
    methods("nand_core", "NandCore", ("_nand_publish_numeric",), "nand_state_publication")
    module("_compiled_nand", "_run", "nand_numeric")
    methods("live_execution_metrics", "LiveCohortMetrics", ("observe", "summary", "metadata"), "live_metrics_summary")
    for attribute in ("_resource_busy_by_direction", "_summarize_dram_task_traffic", "_summarize_nand_task_traffic", "_summarize_kv_task_traffic"):
        module("planner", attribute, "metrics_summary")

    runtime = modules["serving"]._OnlineRuntime
    original_execute = runtime._execute
    measured_execute = collector.wrap(original_execute, "serving._OnlineRuntime._execute", "serving_rollback_and_control")

    @wraps(original_execute)
    def execute(self, cohort):
        collector.enabled = str(cohort.kind) == "decode"
        result = measured_execute(self, cohort)
        if collector.enabled:
            stats = getattr(self.lowerer, "template_cache_stats", None)
            if stats is not None:
                collector.observations["provider_template_cache"] = dict(stats)
        return result

    collector._patches.append((runtime, "_execute", original_execute))
    runtime._execute = execute
    collector.boundaries.append({"label": "serving._OnlineRuntime._execute",
                                "category": "serving_rollback_and_control", "hooked": True,
                                "mechanism": "activates diagnostics only for Decode"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("HBM", "HBF"), required=True)
    parser.add_argument("--prompt", type=int, default=32)
    parser.add_argument("--output", type=int, default=9)
    parser.add_argument("--model", default="qwen2_5-0_5b")
    parser.add_argument("--result", type=Path, required=True, help="Ordinary complete benchmark result")
    parser.add_argument("--profile-result", type=Path, required=True, help="Mutually exclusive wall breakdown")
    parser.add_argument("--implementation-root", type=Path)
    args = parser.parse_args()
    import benchmark_scan_runtime as benchmark

    point = {"architecture": args.architecture, "model_id": args.model,
             "prompt_tokens": args.prompt, "output_tokens": args.output}
    scenario = benchmark.build_scan_scenario(args.model, args.architecture, args.prompt, args.output)
    collector = WallCollector()
    install_boundaries(collector)
    started_ns = time.perf_counter_ns()
    started = started_ns / 1e9
    cpu_started = time.process_time()
    previous_ns, previous_cpu = started_ns, cpu_started
    cohort_timings = []

    def report(event):
        nonlocal previous_ns, previous_cpu
        if event.stage not in {"control_plane", "serving_cohorts"}:
            return
        now_ns, cpu_now = time.perf_counter_ns(), time.process_time()
        kind = event.metadata.get("cohort_kind")
        if event.stage == "serving_cohorts":
            collector.complete_cohort(now_ns, event.completed, kind, event.simulated_time_ns)
            if event.completed:
                cohort_timings.append({"completed": event.completed, "kind": kind,
                    "wall_seconds": (now_ns - previous_ns) / 1e9,
                    "elapsed_wall_seconds": (now_ns - started_ns) / 1e9,
                    "cpu_seconds": cpu_now - previous_cpu, "profiled": True,
                    "simulated_time_ns": event.simulated_time_ns})
            previous_ns, previous_cpu = now_ns, cpu_now
        print(json.dumps({"stage": event.stage, "completed": event.completed,
                          "kind": kind, "wall_seconds": now_ns / 1e9 - started}), flush=True)

    try:
        with benchmark.scenario_gc_thresholds() as thresholds:
            result = benchmark.run_scenario(scenario, retention_policy="aggregate",
                control=benchmark.ExecutionControl(progress_callback=report))
        elapsed = (time.perf_counter_ns() - started_ns) / 1e9
        cpu_seconds = time.process_time() - cpu_started
    finally:
        collector.restore()
    row = {**point, **benchmark.collect_result(result, point), "wall_seconds": elapsed,
        "cpu_seconds": cpu_seconds, "finished_at": benchmark.utc_now(),
        "python_executable": sys.executable, "implementation_root": str(benchmark.IMPLEMENTATION_ROOT),
        "preloaded_modules": benchmark.PRELOADED_MODULES, "transaction_capture": None,
        "source_paths": {}, "gc_thresholds": thresholds, "cohort_timings": cohort_timings,
        "decode_unprofiled_steps": 0, "decode_unprofiled_wall_seconds": 0,
        "profiled": True, "purpose": "Decode wall decomposition; not a formal scan point"}
    row["benchmark_timing"] = benchmark.timing_summary(row)
    profile = collector.result()
    profile["point"] = point
    for destination, content in ((args.result, row), (args.profile_result, profile)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        benchmark.write_json(destination, content)
    print(json.dumps({"profile_result": str(args.profile_result),
                      "steady_seconds_per_step": profile["windows"]["steady"]["seconds_per_step"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
