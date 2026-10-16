"""Observe actual memory-core submissions in the persistent serving kernel.

Usage (inside one worker process, with no concurrent simulation threads)::

    with CoreExecutionAudit() as audit:
        result = run_scenario(scenario)
    evidence = audit.summary()

The research ``serving_row`` collector already installs this observer and adds
``core_execution_audit`` to its row; do not wrap ``serving_row`` a second time.

This observer does not change timing or choose execution modes. Planner previews,
isolated replay kernels, and the bootstrap before online serving are excluded by
runtime identity, even if they use exactly the same physical owner name. Both
submit and submit_many are instrumented: a compiled batch need not call submit.
Statistics count completed Core submissions, not a reconstructed traffic model.
If a live transaction is rolled back, the summary explicitly becomes unsuitable
for committed-traffic totals; successful A/B runs normally have no such restore.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_service import DRAM_CONTROL_COUNT_FIELDS, DRAM_CONTROL_TIME_FIELDS
from heterollm_sim.nand_core import NandCore
from heterollm_sim.serving import _OnlineRuntime


class CoreExecutionAudit:
    """Process-local, bounded-memory live-Core observer; use once per run.

    Summaries are grouped by owner and by executed cohort, with absolute first
    submission/last completion times. A cohort's ``phase`` is its actual serving
    phase (mixed cohorts remain mixed); no time-based attribution is guessed.
    Optional call records are capped, and truncation is explicit. Neither a
    physical configuration nor host-DRAM traffic is evidence for HBM execution.
    """

    _active = False

    def __init__(self, *, max_call_records: int = 0):
        if isinstance(max_call_records, bool) or not isinstance(max_call_records, int) or max_call_records < 0:
            raise ValueError("max_call_records must be a non-negative integer")
        self.max_call_records = max_call_records
        self._contexts = {}  # Strong references prevent object-ID reuse.
        self._cores = {}
        self._rows = {}
        self._cohorts = {}
        self._current_cohort = None
        self._inside_many = set()
        self._call_records = []
        self._call_records_dropped = 0
        self._restores = 0
        self._failures = 0
        self._run_count = 0
        self._run_finished = 0
        self._stack = None

    def _register_runtime(self, runtime):
        self._contexts[id(runtime)] = runtime
        for owner, active in runtime.runtimes.items():
            self._cores[id(active.core)] = (active.core, str(owner))

    @staticmethod
    def _empty_row(core, owner):
        return {"physical_owner": owner, "core": type(core).__name__,
                "memory_kind": core.config.kind.value, "entrypoint_calls": {},
                "transactions": 0, "logical_read_bytes": 0, "logical_write_bytes": 0,
                "physical_read_bytes": 0, "physical_write_bytes": 0,
                "transfer_bytes": 0, "burst_count": 0, "pages_read": 0,
                "pages_programmed": 0, "erase_operations": 0,
                "read_cache_hits": 0, "read_cache_misses": 0,
                "bank_group_timing_transactions": 0,
                "bank_group_exact_fallback_transactions": 0,
                "jedec_control_transactions": 0,
                "compiled_hbm_float64_transactions": 0,
                **{key: 0 for key in DRAM_CONTROL_COUNT_FIELDS + DRAM_CONTROL_TIME_FIELDS},
                "control_metric_recorded_transactions": {
                    key: 0 for key in DRAM_CONTROL_COUNT_FIELDS + DRAM_CONTROL_TIME_FIELDS},
                "first_arrival_ns": None, "last_completion_ns": None}

    def _record(self, core, entrypoint, requests, results):
        owner = self._cores[id(core)][1]
        operation_counts = Counter(request.operation.value for request in requests)
        sample = self._empty_row(core, owner)
        sample["entrypoint_calls"] = {entrypoint: 1}
        sample["transactions"] = len(results)
        sample["first_arrival_ns"] = min((r.arrival_ns for r in requests), default=None)
        sample["last_completion_ns"] = max((r.completion_ns for r in results), default=None)
        for request, result in zip(requests, results):
            sample["jedec_control_transactions"] += int(result.counters.get("control_mode") == "jedec_hbm")
            sample["compiled_hbm_float64_transactions"] += int(bool(result.counters.get("compiled_hbm_float64")))
            sample["bank_group_timing_transactions"] += int(bool(result.counters.get("bank_group_timing_enabled")))
            sample["bank_group_exact_fallback_transactions"] += int(bool(result.counters.get("bank_group_timing_fallback")))
            for key in DRAM_CONTROL_COUNT_FIELDS + DRAM_CONTROL_TIME_FIELDS:
                value = result.counters.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    sample[key] += value
                    sample["control_metric_recorded_transactions"][key] += 1
            if request.operation.value in {"read", "write"}:
                sample["logical_" + request.operation.value + "_bytes"] += result.logical_bytes
            sample["transfer_bytes"] += result.transfer_bytes
            for key in ("physical_read_bytes", "physical_write_bytes", "burst_count",
                        "pages_read", "pages_programmed", "erase_operations",
                        "read_cache_hits", "read_cache_misses"):
                sample[key] += int(result.counters.get(key, 0))
        key = (owner, type(core).__name__)
        self._merge(self._rows.setdefault(key, self._empty_row(core, owner)), sample)
        if self._current_cohort is not None:
            cohort = self._cohorts[self._current_cohort]
            self._merge(cohort["owners"].setdefault(key, self._empty_row(core, owner)), sample)
        if self.max_call_records:
            if len(self._call_records) < self.max_call_records:
                self._call_records.append({**sample, "cohort_id": self._current_cohort,
                                           "operation_counts": dict(operation_counts)})
            else:
                self._call_records_dropped += 1

    @staticmethod
    def _merge(row, sample):
        for key, value in sample.items():
            if key in {"entrypoint_calls", "control_metric_recorded_transactions"}:
                for entrypoint, count in value.items():
                    row[key][entrypoint] = row[key].get(entrypoint, 0) + count
            elif key == "first_arrival_ns":
                if value is not None:
                    row[key] = value if row[key] is None else min(row[key], value)
            elif key == "last_completion_ns":
                if value is not None:
                    row[key] = value if row[key] is None else max(row[key], value)
            elif isinstance(value, (float, int)):
                row[key] += value

    def __enter__(self):
        if CoreExecutionAudit._active or self._stack is not None:
            raise RuntimeError("CoreExecutionAudit must not be nested or used concurrently")
        CoreExecutionAudit._active = True
        stack = self._stack = ExitStack()
        audit = self
        original_run = _OnlineRuntime.run
        original_runtime = PhysicalRuntimeContext.runtime
        original_restore = PhysicalRuntimeContext.restore
        original_cohort = _OnlineRuntime._execute_cohort

        @wraps(original_run)
        def run(online, *args, **kwargs):
            audit._register_runtime(online._execution_resource_kernel.physical_runtime)
            audit._run_count += 1
            result = original_run(online, *args, **kwargs)
            audit._run_finished += 1
            return result

        @wraps(original_runtime)
        def runtime(context, *args, **kwargs):
            active = original_runtime(context, *args, **kwargs)
            if id(context) in audit._contexts:
                owner = args[1] if len(args) > 1 else kwargs["owner"]
                audit._cores[id(active.core)] = (active.core, str(owner))
            return active

        @wraps(original_restore)
        def restore(context, *args, **kwargs):
            result = original_restore(context, *args, **kwargs)
            if id(context) in audit._contexts:
                audit._restores += 1
                audit._register_runtime(context)
            return result

        @wraps(original_cohort)
        def cohort(online, item, *args, **kwargs):
            previous = audit._current_cohort
            key = str(item.cohort_id)
            phases = sorted(item.phase_token_counts)
            audit._cohorts[key] = {"cohort_id": key, "phase": phases[0] if len(phases) == 1 else "mixed",
                                   "phase_token_counts": dict(item.phase_token_counts),
                                   "request_ids": list(item.request_ids), "owners": {}}
            audit._current_cohort = key
            try:
                return original_cohort(online, item, *args, **kwargs)
            finally:
                audit._current_cohort = previous

        def wrap_submit(original, entrypoint):
            @wraps(original)
            def submit(core, request, *args, **kwargs):
                tracked = id(core) in audit._cores and id(core) not in audit._inside_many
                if not tracked:
                    return original(core, request, *args, **kwargs)
                requests = tuple(request) if entrypoint == "submit_many" else (request,)
                if entrypoint == "submit_many":
                    audit._inside_many.add(id(core))
                try:
                    result = original(core, requests if entrypoint == "submit_many" else request, *args, **kwargs)
                except Exception:
                    audit._failures += 1
                    raise
                finally:
                    if entrypoint == "submit_many":
                        audit._inside_many.remove(id(core))
                results = result if entrypoint == "submit_many" else (result,)
                audit._record(core, entrypoint, requests, results)
                return result
            return submit

        stack.enter_context(patch.object(_OnlineRuntime, "run", run))
        stack.enter_context(patch.object(_OnlineRuntime, "_execute_cohort", cohort))
        stack.enter_context(patch.object(PhysicalRuntimeContext, "runtime", runtime))
        stack.enter_context(patch.object(PhysicalRuntimeContext, "restore", restore))
        for core in (DramCore, NandCore):
            for entrypoint in ("submit", "submit_many"):
                if hasattr(core, entrypoint):
                    stack.enter_context(patch.object(core, entrypoint, wrap_submit(getattr(core, entrypoint), entrypoint)))
        return self

    def __exit__(self, *exc):
        try:
            return self._stack.__exit__(*exc)
        finally:
            CoreExecutionAudit._active = False

    def summary(self):
        cohorts = [{**row, "owners": list(row["owners"].values())} for row in self._cohorts.values()]
        configurations = {}
        for core, owner in self._cores.values():
            if (owner, type(core).__name__) in self._rows:
                configurations[owner] = {"core": type(core).__name__, "physical_config": asdict(core.config)}
        result = deepcopy({
            "schema_version": "heterollm.live-core-audit/v1",
            "scope": "successful Core submissions on the persistent online-serving kernel; excludes planner previews, isolated replay and pre-serving initialization",
            "method": "instrumented DramCore/NandCore submit and submit_many, selected by live PhysicalRuntimeContext identity",
            "run_count": self._run_count, "finished_run_count": self._run_finished,
            "live_runtime_restores": self._restores, "failed_entrypoint_calls": self._failures,
            "valid_for_committed_byte_totals": self._run_count == self._run_finished > 0 and not self._restores and not self._failures,
            "owners": list(self._rows.values()), "cohorts": cohorts,
            "executed_physical_configurations": configurations,
            "call_records": self._call_records, "call_records_dropped": self._call_records_dropped,
            "call_records_enabled": bool(self.max_call_records),
            "limitations": ["Cohort phase attribution follows actual execution; mixed cohorts are not split artificially.",
                            "Physical bytes are Core result counters; NAND internal/host transfer bytes are separate from media read/program bytes.",
                            "Instrumenting these process-global entry points requires one simulation thread per process."]})
        rows = result["owners"] + result["call_records"] + [
            row for cohort in result["cohorts"] for row in cohort["owners"]]
        for row in rows:
            for key, count in row["control_metric_recorded_transactions"].items():
                if count != row["transactions"]:
                    row[key] = None
        return result
