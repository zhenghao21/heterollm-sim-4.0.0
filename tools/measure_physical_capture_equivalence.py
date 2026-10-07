"""Quantify trace-retention arithmetic differences; never fit native timings."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from unittest.mock import patch

from heterollm_sim.architecture_presets import materialize_architecture_payload
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest, DramConfig, Operation, parse_physical_memory_config


HARDWARE = ("local-native-rtx5080-9950x3d-gddr7-ddr5", "nvidia-b200-1gpu-2hbf-2hbm",
            "nvidia-b200-1gpu-3hbf-2hbm")


class Difference:
    def __init__(self):
        self.count = 0
        self.changed = 0
        self.max_absolute = 0.0
        self.max_relative = 0.0
        self.max_path = None
        self.non_numeric_mismatches = []

    def compare(self, expected, actual, path=""):
        if isinstance(expected, dict) and isinstance(actual, dict):
            if expected.keys() != actual.keys():
                self.non_numeric_mismatches.append(path + ":keys")
            for key in expected.keys() & actual.keys():
                self.compare(expected[key], actual[key], path + "/" + str(key))
        elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
            if len(expected) != len(actual):
                self.non_numeric_mismatches.append(path + ":length")
            for i, (left, right) in enumerate(zip(expected, actual)):
                self.compare(left, right, path + "/" + str(i))
        elif (isinstance(expected, (int, float)) and not isinstance(expected, bool)
              and isinstance(actual, (int, float)) and not isinstance(actual, bool)):
            self.count += 1
            error = abs(actual - expected)
            self.changed += error != 0
            if error > self.max_absolute:
                self.max_absolute, self.max_path = error, path
            if expected != 0:
                self.max_relative = max(self.max_relative, error / abs(expected))
            elif error:
                self.non_numeric_mismatches.append(path + ":nonzero_against_zero")
        elif expected != actual:
            self.non_numeric_mismatches.append(path)

    def result(self):
        return {"numeric_values_compared": self.count, "changed_values": self.changed,
                "max_absolute_difference": self.max_absolute,
                "max_relative_difference": self.max_relative,
                "max_relative_difference_percent": 100 * self.max_relative,
                "max_absolute_path": self.max_path,
                "non_numeric_mismatches": self.non_numeric_mismatches}


def requests(config):
    burst = config.burst_bytes
    row_stride = config.lane_count * config.ranks_per_channel * config.bank_groups_per_rank * config.banks_per_group * config.row_bytes
    bulk = 9217 * burst + 7
    items = [("cold", Operation.READ, 0, burst, 0),
             ("row-hit", Operation.READ, 0, burst, 0),
             ("row-conflict", Operation.READ, row_stride, burst, 0),
             ("write-turnaround", Operation.WRITE, row_stride, burst, 0),
             ("bulk-read", Operation.READ, 17, bulk, 0),
             ("bulk-write", Operation.WRITE, 17, bulk, 0),
             ("bulk-read-again", Operation.READ, 17, bulk, 0)]
    # Discrete element/column accesses use the core's supported explicit
    # request sequence; no synthetic aggregate-stride approximation is used.
    items += [("strided-" + str(i), Operation.WRITE if i % 2 else Operation.READ,
               2 * row_stride + i * 1536 + 2, 2, 0) for i in range(32)]
    items += [("large-clock", Operation.READ, 0, 9 * burst, 1e12),
              ("large-clock-write", Operation.WRITE, 0, 9 * burst, 1e12)]
    return tuple(AccessRequest(*item) for item in items)


def state(core):
    return {"banks": {key: asdict(value) for key, value in core._banks.items()},
            "ready_ns": dict(core.timeline.ready_ns), "lane_available": dict(core.timeline.lane_available),
            "busy_ns": dict(core.timeline.busy_ns), "last_intervals": dict(core.timeline.last_intervals),
            "bytes_moved": dict(core.timeline.bytes_moved), "directions": dict(core.timeline.directions),
            "touched": sorted(core.timeline._touched), "inflight": list(core._inflight),
            "acceptance_ns": core._acceptance_ns}


def result_fields(value, energy_rate):
    counters = value.counters
    return {"times": {"arrival_ns": value.arrival_ns, "completion_ns": value.completion_ns,
                      "latency_ns": value.latency_ns, "queue_wait_ns": value.queue_wait_ns},
            "bytes": {"logical_bytes": value.logical_bytes, "transfer_bytes": value.transfer_bytes,
                      "physical_read_bytes": value.physical_read_bytes, "physical_write_bytes": value.physical_write_bytes},
            "energy_pj": value.transfer_bytes * energy_rate,
            "counters": {key: counters[key] for key in ("row_hits", "row_misses", "row_conflicts", "burst_count")}}


def core_comparisons():
    rows = []
    for hardware in HARDWARE:
        for component in materialize_architecture_payload(hardware)["components"]:
            raw = component.get("metadata", {}).get("physical_memory_config")
            if raw is None:
                continue
            config = parse_physical_memory_config(raw)
            if not isinstance(config, DramConfig):
                continue
            energy_rate = component["metadata"]["cost_profile_template"]["energy_pj_per_byte"]
            streams = requests(config)
            cores = [DramCore(config, capture_details=value) for value in (True, False)]
            diffs = {key: Difference() for key in ("times", "bytes", "energy_pj", "counters", "state")}
            timings = [0.0, 0.0]
            accelerated = 0
            for request in streams:
                values = []
                for index, core in enumerate(cores):
                    start = time.perf_counter()
                    values.append(core.submit(request))
                    timings[index] += time.perf_counter() - start
                expected, actual = [result_fields(value, energy_rate) for value in values]
                for key in expected:
                    diffs[key].compare(expected[key], actual[key], request.request_id)
                diffs["state"].compare(state(cores[0]), state(cores[1]), request.request_id)
                accelerated += bool(values[1].counters.get("compiled_float64") or values[1].counters.get("accelerated_row_hit_bursts"))
            rows.append({"hardware_id": hardware, "component_id": component["component_id"],
                         "kind": config.kind.value, "request_count": len(streams),
                         "compiled_or_accelerated_requests": accelerated,
                         "capture_true_wall_seconds": timings[0], "capture_false_wall_seconds": timings[1],
                         "energy_rate_pj_per_byte": energy_rate,
                         "differences": {key: diff.result() for key, diff in diffs.items()}})
            print(hardware, component["component_id"], "complete", flush=True)
    return rows


def tiny_scenario():
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    from heterollm_sim.llama_scenario import prepare_llama_scenario
    from heterollm_sim.reference import build_llama_default_scenario
    from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
    base = build_llama_default_scenario()
    graph = build_model_graph_from_layer_specs("capture-equivalence-tiny",
        (LayerSpec(layer_id="layer0", kind="dense", hidden_size=64, intermediate_size=128,
                   attention_heads=4, kv_heads=2, dtype="fp16", weight_bytes=73728),),
        architecture="llama", vocabulary_size=32, max_sequence_length=4096,
        embedding_weight_bytes=4096, tie_word_embeddings=True)
    request = replace(base.workload.requests[0], prompt_tokens=4, output_tokens=3)
    return prepare_llama_scenario(replace(base, model=ModelSpec(name="capture-equivalence-tiny", graph=graph),
        placement=replace(base.placement, model_name="capture-equivalence-tiny",
                          parallel=replace(base.placement.parallel, layer_to_stage={})),
        workload=replace(base.workload, requests=(request,), prompt_tokens=4, output_tokens=3),
        llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=-1)))


def whole_run_comparison():
    from heterollm_sim import reporting
    original = reporting.bootstrap_control_plane
    runs, flags = [], []
    for capture in (True, False):
        bootstraps = []
        def boot(scenario, **kwargs):
            result = original(scenario, **{**kwargs, "capture_physical_details": capture})
            bootstraps.append(result)
            return result
        start = time.perf_counter()
        with patch.object(reporting, "bootstrap_control_plane", boot):
            result = reporting.run_scenario(tiny_scenario(), retention_policy="aggregate")
        kernel = bootstraps[0].kernel
        flags.append({"requested": capture, "context": kernel.physical_runtime.capture_details,
                      "cores": {key: runtime.core.capture_details for key, runtime in kernel.physical_runtime.runtimes.items()}})
        assert flags[-1]["context"] == capture and all(value == capture for value in flags[-1]["cores"].values())
        batches = [{"duration_ns": batch.cost.duration_ns, "energy_pj": batch.cost.energy_pj,
                    "metrics": {key: batch.cost.metadata[key] for key in (
                        "resource_accounted_bytes", "resource_busy_ns", "category_time_ns",
                        "critical_path_category_ns", "dram_traffic", "storage_traffic")}}
                   for batch in result.serving.batches]
        runs.append({"wall_seconds": time.perf_counter() - start,
                     "batch_count": len(batches), "batches": batches,
                     "requests": [asdict(row) for row in result.serving.request_metrics.values()],
                     "physical_final_state": {key: state(runtime.core) for key, runtime in kernel.physical_runtime.runtimes.items()
                                              if isinstance(runtime.core, DramCore)}})
    diff = Difference()
    diff.compare({key: value for key, value in runs[0].items() if key != "wall_seconds"},
                 {key: value for key, value in runs[1].items() if key != "wall_seconds"})
    return {"scope": "Synthetic 1-layer llama, hidden 64, FFN 128, vocabulary 32; input 4/output 3; complete bootstrap and serving.",
            "capture_flags": flags, "capture_true_wall_seconds": runs[0]["wall_seconds"],
            "capture_false_wall_seconds": runs[1]["wall_seconds"], "differences": diff.result()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/frontend_native_validation_2026-10-07/physical_capture_equivalence.json"))
    args = parser.parse_args()
    data = {"scope": "Retention True per-burst versus False production exact acceleration; all DRAM components of the three formal architecture presets.",
            "excluded": ["expanded traces/mappings, deliberately absent for False", "acceleration implementation markers"],
            "interpretation": "Exact numeric differences are measured, not rounded to pytest tolerance. This measures implementation equivalence, not native predictive accuracy.",
            "energy_basis": "Production declared physical bytes times each preset's unchanged pJ/byte; DRAM core itself has no independent energy calibration.",
            "core_comparisons": core_comparisons()}
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    data["whole_run_comparison"] = whole_run_comparison()
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
