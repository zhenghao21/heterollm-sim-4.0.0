"""Bounded host profiling only: never a formal matrix result or extrapolation."""
from __future__ import annotations
import argparse
import cProfile
import ctypes
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.execution_control import ExecutionControl, ExecutionCancelledError
from heterollm_sim.planner import TopologyAwareBatchCostProvider
from heterollm_sim.reporting import run_scenario


def memory_mib():
    if sys.platform != "win32":
        return {}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    get_memory = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return {key: round(getattr(counters, key) / 2**20, 2) for key in
            ("WorkingSetSize", "PeakWorkingSetSize", "PagefileUsage")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "artifacts/memory_tier_scenarios_20260922/multiworkload-matrix-v3/configs/L8_short_high_batch__dram_independent.json")
    parser.add_argument("--decode-batches", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=120, help="Cooperative limit; one lowering may overrun it")
    args = parser.parse_args()
    if not 1 <= args.decode_batches <= 5 or not 0 < args.seconds <= 300:
        parser.error("profiling requires 1..5 decode batches and 0 < seconds <= 300")
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    model_hash = payload.pop("model_config_sha256", None)
    if "model" not in payload:
        from tools.qwen38_memory_scenario import load_model
        from heterollm_sim.serde import to_primitive, stable_hash
        model, _ = load_model()
        payload["model"] = to_primitive(model)
        if model_hash and stable_hash(model) != model_hash:
            raise ValueError("Original config model hash mismatch")
    scenario = scenario_from_dict(payload)
    baseline = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src/heterollm_sim").rglob("*.py")}
    started = time.perf_counter()
    decoded = 0
    def cancelled():
        return decoded >= args.decode_batches or time.perf_counter() - started >= args.seconds
    def progress(event):
        nonlocal decoded
        if event.stage == "serving_cohorts":
            if event.metadata.get("cohort_kind") == "decode":
                decoded += 1
            print(json.dumps({"progress": event.completed, "metadata": dict(event.metadata),
                  "host_s": round(time.perf_counter()-started, 3), "memory_mib": memory_mib()}), flush=True)
    control = ExecutionControl(progress_callback=progress, cancellation_callback=cancelled)
    class ObservedProvider(TopologyAwareBatchCostProvider):
        def estimate(self, scenario, cohort):
            before = time.perf_counter()
            result = super().estimate(scenario, cohort)
            print(json.dumps({"estimate_kind": cohort.kind, "items": len(cohort.items),
                "contexts": sorted({i.context_tokens for i in cohort.items}),
                "host_s": round(time.perf_counter()-before, 3), "memory_mib": memory_mib(),
                "caches": {name: getattr(self, name) for name in ("template_cache_stats", "leaf_cache_stats", "metadata_cache_stats", "compiled_graph_cache_stats")}}), flush=True)
            return result
    profile = cProfile.Profile()
    print(json.dumps({"PROFILING_ONLY": True, "config": str(args.config), "workload": {"name": scenario.workload.name, "requests": len(scenario.workload.requests), "scheduler": payload["workload"]["scheduler"]}, "memory_mib": memory_mib()}), flush=True)
    profile.enable()
    try:
        provider = ObservedProvider(scenario, execution_control=control)
        run_scenario(scenario, control=control, batch_lowerer=provider)
    except ExecutionCancelledError:
        print("PROFILING_ONLY: intentional bounded stop", flush=True)
    finally:
        profile.disable()
        print(json.dumps({"host_s": time.perf_counter()-started, "decode_batches_completed": decoded,
              "memory_mib": memory_mib(), "source_changed_during_profile": [str(p.relative_to(ROOT)) for p,h in baseline.items() if hashlib.sha256(p.read_bytes()).hexdigest()!=h]}))
        pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(30)
        pstats.Stats(profile).strip_dirs().sort_stats("tottime").print_stats(20)

if __name__ == "__main__":
    main()
