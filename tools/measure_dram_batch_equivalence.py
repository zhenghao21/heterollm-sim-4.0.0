"""Measure ordered request batching against the unchanged serial executor."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from statistics import median
from unittest.mock import patch

from heterollm_sim import data_motion, planner
from heterollm_sim.architecture_presets import materialize_architecture_payload
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.dram_core import DramCore
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.memory_types import AccessRequest, DramConfig, Operation, parse_physical_memory_config
from measure_physical_capture_equivalence import HARDWARE, Difference, requests, state


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/frontend_native_validation_2026-10-07"


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if key != "compiled_batch_float64"}
    if isinstance(value, (list, tuple)):
        return tuple(clean(item) for item in value)
    return value


def request_comparison(config, sequence):
    # Compilation is excluded from the repeated-execution speed comparison.
    DramCore(config, capture_details=False).submit_many(sequence[:2])
    cores = [DramCore(config, capture_details=False) for _ in range(2)]
    start = time.perf_counter()
    expected = tuple(cores[0].submit(item) for item in sequence)
    serial_seconds = time.perf_counter() - start
    start = time.perf_counter()
    actual = cores[1].submit_many(sequence)
    batch_seconds = time.perf_counter() - start
    diff = Difference()
    diff.compare([clean(asdict(row)) for row in expected], [clean(asdict(row)) for row in actual], "requests")
    diff.compare(state(cores[0]), state(cores[1]), "final_state")
    return {"requests": len(sequence), "serial_wall_seconds": serial_seconds,
            "batch_wall_seconds": batch_seconds, "speedup": serial_seconds / batch_seconds,
            "difference": diff.result()}


def physical_task_comparison(config):
    rows = []
    for owner, operation, count in (("a", "write", 1024), ("b", "read", 16),
                                  ("a", "read", 1024), ("a", "write", 8)):
        rows += [{"physical_owner": owner, "operation": operation, "address": 1536 * i + 1026,
                  "byte_count": 2} for i in range(count)]
    rows += [{"physical_owner": "a", "operation": op, "address": address, "byte_count": size}
             for op, address, size in (("write", 1026, 4), ("write", 1028, 4), ("read", 1026, 6))]
    task = TaskSpec(task_id="batch-check", request_id="cohort-check", name="batch-check", category=TaskCategory.MEMORY,
        metadata={"physical_memory_config": asdict(config),
            "physical_memory_configs": {key: asdict(config) for key in ("a", "b")},
            "physical_energy_pj_per_byte_by_owner": {"a": 4.0, "b": 7.0}, "memory_accesses": rows})
    results, clocks, states = [], [], []
    for enabled in (False, True):
        runtime = data_motion.PhysicalRuntimeContext(capture_details=False)
        start = time.perf_counter()
        with patch.object(data_motion, "_USE_COMPILED_DRAM_BATCH", enabled):
            result = data_motion.resolve_physical_task(task, runtime, 17.0)
        clocks.append(time.perf_counter() - start)
        results.append(clean(asdict(result)))
        states.append({key: state(value.core) for key, value in runtime.runtimes.items()})
    diff = Difference()
    diff.compare(results[0], results[1], "resolved_task")
    diff.compare(states[0], states[1], "final_state")
    return {"access_count": len(rows), "serial_wall_seconds": clocks[0], "batch_wall_seconds": clocks[1],
            "speedup": clocks[0] / clocks[1], "difference": diff.result()}


def source_layer_comparison(slug):
    scenario = scenario_from_dict(json.loads((REPORT / ("scenario_" + slug + "_512_128.json")).read_text(encoding="utf-8")))
    with planner._compilation_scope(scenario):
        builder = planner._TaskBuilder(scenario.workload.requests[0])
        layer = next(row for row in planner._execution_layers(scenario) if row.sequence_mixer == "full_attention")
        planner._compile_parallel_layer_body(builder, scenario, planner._parallel_plan(scenario),
            planner._topology_router(scenario), layer, token_batch=1, context_tokens=513,
            kv_read_tokens=512, kv_append_tokens=1, kv_materialized_tokens=1,
            linear_state_runtime=None, phase="decode", dependencies=())
    tasks = planner._promote_physical_allocation_extents(builder.tasks)
    values, clocks = [], {False: [], True: []}
    diff = Difference()
    for repetition in range(5):
        pair = []
        for enabled in (False, True):
            kernel = UnifiedEventKernel.from_closed_graph(tasks,
                resource_capacities=planner._scenario_resource_capacities(scenario),
                resource_owners=planner._scenario_resource_owners(scenario), capture_physical_details=False)
            metrics = LiveCohortMetrics(resource_owners=kernel.resource_owners)
            start = time.perf_counter()
            with patch.object(data_motion, "_USE_COMPILED_DRAM_BATCH", enabled):
                while kernel.has_active_tasks:
                    metrics.observe(kernel.step())
            clocks[enabled].append(time.perf_counter() - start)
            pair.append({"metrics": clean(metrics.metadata()), "duration_ns": kernel.makespan_ns,
                "energy_pj": metrics.summary().energy_pj,
                "final_state": {key: state(value.core) for key, value in kernel.physical_runtime.runtimes.items()
                                if isinstance(value.core, DramCore)}})
        diff.compare(pair[0], pair[1], "source_layer_run_" + str(repetition))
    return {"case": slug, "scope": "Actual full-attention layer, decode at logical context 513 / padded read 768.",
            "repetitions": 5, "serial_wall_seconds": median(clocks[False]), "batch_wall_seconds": median(clocks[True]),
            "serial_wall_samples": clocks[False], "batch_wall_samples": clocks[True],
            "speedup": median(clocks[False]) / median(clocks[True]), "difference": diff.result()}


def source_whole_comparison(slug="qwen3_0_6b_f16", repetitions=3):
    """Run the complete real model; shorten only the request to 4/3 tokens."""
    from heterollm_sim import reporting
    original = reporting.bootstrap_control_plane
    scenario = scenario_from_dict(json.loads((REPORT / ("scenario_" + slug + "_512_128.json")).read_text(encoding="utf-8")))
    request = replace(scenario.workload.requests[0], prompt_tokens=4, output_tokens=3)
    scenario = replace(scenario, workload=replace(scenario.workload,
        requests=(request,), prompt_tokens=4, output_tokens=3))
    clocks = {False: [], True: []}
    diff = Difference()
    for repetition in range(repetitions):
        pair = {}
        # Alternate order to avoid attributing a consistent first-run cost to
        # one implementation. Both paths keep the same physical parameters.
        for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
            bootstraps = []
            def boot(scenario, **kwargs):
                result = original(scenario, **{**kwargs, "capture_physical_details": False})
                bootstraps.append(result)
                return result
            start = time.perf_counter()
            with patch.object(data_motion, "_USE_COMPILED_DRAM_BATCH", enabled), patch.object(reporting, "bootstrap_control_plane", boot):
                result = reporting.run_scenario(scenario, retention_policy="aggregate")
            clocks[enabled].append(time.perf_counter() - start)
            kernel = bootstraps[0].kernel
            assert not kernel.physical_runtime.capture_details
            assert all(not item.core.capture_details for item in kernel.physical_runtime.runtimes.values())
            pair[enabled] = {
                "batches": [{"duration_ns": batch.cost.duration_ns, "energy_pj": batch.cost.energy_pj,
                    "metrics": {key: clean(batch.cost.metadata[key]) for key in (
                        "resource_accounted_bytes", "resource_busy_ns", "category_time_ns",
                        "critical_path_category_ns", "dram_traffic", "storage_traffic")}}
                    for batch in result.serving.batches],
                "requests": [asdict(row) for row in result.serving.request_metrics.values()],
                "physical_final_state": {key: state(runtime.core) for key, runtime in kernel.physical_runtime.runtimes.items()
                    if isinstance(runtime.core, DramCore)}}
            print(slug, "whole", repetition, "batch" if enabled else "serial", clocks[enabled][-1], flush=True)
        diff.compare(pair[False], pair[True], "source_whole_run_" + str(repetition))
    return {"case": slug, "scope": "Complete actual GGUF model, input 4/output 3; original 768-token capacity and all source contracts, host output, live physical state preserved. This is an implementation comparison, not a 512/128 native error estimate.",
            "repetitions": repetitions, "serial_wall_seconds": median(clocks[False]), "batch_wall_seconds": median(clocks[True]),
            "serial_wall_samples": clocks[False], "batch_wall_samples": clocks[True],
            "speedup": median(clocks[False]) / median(clocks[True]), "difference": diff.result()}


def main():
    rows, gddr = [], None
    for hardware in HARDWARE:
        for component in materialize_architecture_payload(hardware)["components"]:
            raw = component.get("metadata", {}).get("physical_memory_config")
            if not raw:
                continue
            config = parse_physical_memory_config(raw)
            if not isinstance(config, DramConfig):
                continue
            rows.append({"hardware": hardware, "component": component["component_id"],
                         **request_comparison(config, requests(config))})
            if config.kind.value == "GDDR":
                gddr = config
    columns = tuple(AccessRequest("column-" + str(i), Operation.WRITE, i * 1536 + 1026, 2, 1e6) for i in range(1024))
    result = {"scope": "False-capture serial submit versus ordered submit_many; no changed physical parameters or merged accesses.",
              "reference": "Current production scalar/compiled per-request executor; unlike True/False capture comparison, this isolates the new batching change.",
              "production_gate": "At least 16 consecutive same-owner same-direction requests, each fewer than 64 bursts; writes must have ascending nonoverlapping logical byte ranges; at most 1024 requests per batch. Detailed capture retains the original scalar path.",
              "timing_context": "Diagnostic wall times may be affected by concurrent UI simulations. Repeated medians show benefit in this environment, not an isolated machine throughput guarantee.",
              "enabled_by_default": data_motion._USE_COMPILED_DRAM_BATCH,
              "interpretation": "Differences are measured exactly for recorded cases, not asserted from a tolerance check. This is implementation equivalence, not improved native predictive accuracy.",
              "core_cases": rows, "native_v_columns": request_comparison(gddr, columns),
              "interleaved_multiowner_overlap_case": physical_task_comparison(gddr),
              "real_source_layers": [source_layer_comparison(slug) for slug in ("qwen3_0_6b_f16", "qwen3_8_27b_mixed")]}
    path = REPORT / "dram_batch_equivalence.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["real_source_whole_model"] = source_whole_comparison()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
