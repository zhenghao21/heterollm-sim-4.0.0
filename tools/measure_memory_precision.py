"""Measure floating-point drift between exact memory fast paths and references.

This is a focused software equivalence audit, not a hardware calibration.
It executes the same request sequences through the current path and a reference
path that forces the detailed burst/page transition loop, then records measured
completion, resource-calendar, busy-time, and core-state deltas.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

from heterollm_sim.architecture_presets import architecture_preset_detail, list_architecture_presets
from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import (
    AccessRequest, DramConfig, MemoryKind, NandConfig, Operation,
    parse_physical_memory_config,
)
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.nand_core import NandCore


OUTPUT = Path("docs/memory_precision_2026-10-07.json")
ACCEL_DIAGNOSTICS = {"accelerated_row_hit_bursts", "accelerated_pages"}


class BurstReference(DramCore):
    def _can_accelerate(self):
        return False


class PageReference(NandCore):
    def _execute_accepted(self, request):
        return self._execute_detailed_accepted(request)


def generic_dram(**changes):
    config = DramConfig(
        kind=MemoryKind.GDDR, data_lanes=2, bank_groups_per_rank=2,
        banks_per_group=2, rows_per_bank=1024, row_bytes=8192,
        interface_bandwidth_gb_s=80.0, max_expanded_segments=8192,
    )
    return replace(config, capacity_bytes=None, **changes)


def nand_config(**changes):
    config = NandConfig(
        kind="HBF", channels=4, planes_per_lun=2, blocks_per_plane=1024,
        page_bytes=64, host_granularity_bytes=64, pages_per_block=8,
        host_bandwidth_gb_s=8, internal_bandwidth_gb_s=16,
        page_read_ns=50, page_program_ns=400, block_erase_ns=70, front_ns=7,
    )
    return replace(config, capacity_bytes=None, **changes)


def normalized(value):
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    if isinstance(value, set):
        return sorted(normalized(v) for v in value)
    if isinstance(value, float):
        return value
    return value


def metric_diff(actual, reference):
    """Return measured maxima; relative denominator is abs(reference).

    When reference is zero, equal zeros produce relative error 0 and nonzero
    actual values are counted separately rather than assigned an infinite or
    misleading percentage.
    """
    actual, reference = normalized(actual), normalized(reference)
    if isinstance(actual, dict) and isinstance(reference, dict):
        paths = sorted(set(actual) | set(reference))
        values = []
        for key in paths:
            if key not in actual or key not in reference:
                values.append({"path": str(key), "key_mismatch": True})
            else:
                child = metric_diff(actual[key], reference[key])
                for row in child:
                    row["path"] = f"{key}.{row.get('path', '')}".rstrip(".")
                values.extend(child)
        return values
    if isinstance(actual, list) and isinstance(reference, list):
        values = []
        if len(actual) != len(reference):
            values.append({"path": "length", "key_mismatch": True,
                           "actual": len(actual), "reference": len(reference)})
        for i, (a, r) in enumerate(zip(actual, reference)):
            for row in metric_diff(a, r):
                row["path"] = f"[{i}].{row.get('path', '')}".rstrip(".")
                values.append(row)
        return values
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(reference, (int, float)) and not isinstance(reference, bool):
        absolute = abs(float(actual) - float(reference))
        ref_abs = abs(float(reference))
        relative = absolute / ref_abs if ref_abs else (0.0 if absolute == 0 else None)
        return [{"path": "", "actual": actual, "reference": reference,
                 "absolute_error": absolute, "relative_error": relative,
                 "zero_reference_nonzero_actual": bool(ref_abs == 0 and absolute != 0)}]
    if actual == reference:
        return []
    return [{"path": "", "key_mismatch": True, "actual": actual, "reference": reference}]


def summarize_diffs(diffs):
    numeric = [d for d in diffs if "absolute_error" in d]
    return {
        "max_absolute_error": max((d["absolute_error"] for d in numeric), default=0.0),
        "max_relative_error": max((d["relative_error"] for d in numeric if d["relative_error"] is not None), default=0.0),
        "zero_reference_nonzero_actual_count": sum(d.get("zero_reference_nonzero_actual", False) for d in numeric),
        "key_mismatch_count": sum(d.get("key_mismatch", False) for d in diffs),
        "max_absolute_location": max(numeric, key=lambda d: d["absolute_error"])["path"] if numeric else None,
        "max_relative_location": max((d for d in numeric if d["relative_error"] is not None), key=lambda d: d["relative_error"])["path"] if any(d["relative_error"] is not None for d in numeric) else None,
    }


def discrete_mismatches(actual, reference, path=""):
    """Find any mismatched non-float state, including integer counters."""
    mismatches = []
    if isinstance(actual, dict) and isinstance(reference, dict):
        if set(actual) != set(reference):
            mismatches.append({"path": path, "kind": "keys", "actual": sorted(actual), "reference": sorted(reference)})
        for key in sorted(set(actual) & set(reference), key=str):
            mismatches.extend(discrete_mismatches(actual[key], reference[key], f"{path}.{key}".strip(".")))
        return mismatches
    if isinstance(actual, (list, tuple)) and isinstance(reference, (list, tuple)):
        if len(actual) != len(reference):
            mismatches.append({"path": path, "kind": "length", "actual": len(actual), "reference": len(reference)})
        for index, (a, r) in enumerate(zip(actual, reference)):
            mismatches.extend(discrete_mismatches(a, r, f"{path}[{index}]"))
        return mismatches
    if isinstance(actual, float) or isinstance(reference, float):
        return mismatches
    if actual != reference:
        mismatches.append({"path": path, "kind": "value", "actual": actual, "reference": reference})
    return mismatches


def timeline_state(core):
    t = core.timeline
    return {
        "ready_ns": t.ready_ns,
        "lane_available": t.lane_available,
        "busy_ns": t.busy_ns,
        "last_intervals": t.last_intervals,
        "bytes_moved": t.bytes_moved,
        "directions": t.directions,
        "touched": t._touched,
    }


def core_state(core):
    if isinstance(core, DramCore):
        return {"banks": {key: {"open_row": val.open_row, "ready_ns": val.ready_ns}
                           for key, val in core._banks.items()},
                "inflight": core._inflight, "acceptance_ns": core._acceptance_ns}
    return {"array_ready": core._array_ready, "buffer_ready": core._buffer_ready,
            "inflight": core._inflight, "acceptance_ns": core._acceptance_ns}


def integer_metrics(result):
    counters = {k: v for k, v in result.counters.items()
                if k not in ACCEL_DIAGNOSTICS and isinstance(v, int) and not isinstance(v, bool)}
    return {"logical_bytes": result.logical_bytes, "transfer_bytes": result.transfer_bytes,
            "integer_counters": counters}


def compare_case(name, memory_type, config, requests, *, seed=None):
    if memory_type == "dram":
        actual = DramCore(config, capture_details=False)
        reference = BurstReference(config, capture_details=False)
    else:
        actual = NandCore(config, capture_details=False)
        reference = PageReference(config, capture_details=False)
    serial_numeric = None
    if memory_type == "dram":
        serial_numeric = DramCore(config, capture_details=False)
    if seed:
        for core in (actual, reference):
            seed(core)
        if serial_numeric is not None:
            seed(serial_numeric)
    rows = []
    all_integer_mismatches = []
    serial_integer_mismatches = []
    max_completion = {"absolute_error": 0.0, "relative_error": 0.0, "zero_reference_nonzero_actual": False}
    max_duration = {"absolute_error": 0.0, "relative_error": 0.0, "zero_reference_nonzero_actual": False}
    for request in requests:
        a, e = actual.submit(request), reference.submit(request)
        s = None
        if serial_numeric is not None:
            # The compiled wrapper still owns input packing/state commit, while
            # the kernel visits every burst and disables row-cycle folding.
            # The Python reference below independently checks this numeric loop.
            from heterollm_sim import _dram_numeric
            from precision_dram_reference import run_numeric_serial
            original_numeric = _dram_numeric.run_numeric
            try:
                _dram_numeric.run_numeric = run_numeric_serial
                s = serial_numeric.submit(request)
            finally:
                _dram_numeric.run_numeric = original_numeric
            serial_mismatch = [x for x in metric_diff(integer_metrics(s), integer_metrics(e))
                               if x.get("key_mismatch") or x.get("absolute_error", 0) != 0]
            serial_integer_mismatches.extend({"request_id": request.request_id, **x} for x in serial_mismatch)
        diff = metric_diff(a.completion_ns, e.completion_ns)[0]
        if diff["absolute_error"] > max_completion["absolute_error"]:
            max_completion = diff
        actual_duration = a.completion_ns - request.arrival_ns
        reference_duration = e.completion_ns - request.arrival_ns
        duration_diff = metric_diff(actual_duration, reference_duration)[0]
        if duration_diff["absolute_error"] > max_duration["absolute_error"]:
            max_duration = duration_diff
        am, em = integer_metrics(a), integer_metrics(e)
        im = metric_diff(am, em)
        mismatches = [x for x in im if x.get("key_mismatch") or x.get("absolute_error", 0) != 0]
        all_integer_mismatches.extend({"request_id": request.request_id, **x} for x in mismatches)
        rows.append({"request_id": request.request_id, "operation": request.operation.value,
                     "address": request.address, "byte_count": request.byte_count,
                     "arrival_ns": request.arrival_ns,
                     "actual_completion_ns": a.completion_ns,
                     "reference_completion_ns": e.completion_ns,
                     "actual_duration_ns": actual_duration,
                     "reference_duration_ns": reference_duration,
                     "completion_timestamp_absolute_error_ns": diff["absolute_error"],
                     "completion_timestamp_relative_error": diff["relative_error"],
                     "duration_absolute_error_ns": duration_diff["absolute_error"],
                     "duration_relative_error": duration_diff["relative_error"],
                     "integer_counters_match": not mismatches,
                     "integer_counter_mismatches": mismatches,
                     "actual_acceleration_counters": {k: v for k, v in a.counters.items() if k in ACCEL_DIAGNOSTICS},
                     "serial_numeric_completion_ns": s.completion_ns if s is not None else None,
                     "serial_numeric_vs_python_absolute_error_ns": abs(s.completion_ns - e.completion_ns) if s is not None else None,
                     "serial_numeric_vs_python_timestamp_relative_error": metric_diff(s.completion_ns, e.completion_ns)[0]["relative_error"] if s is not None else None,
                     "serial_numeric_vs_python_duration_absolute_error_ns": abs((s.completion_ns - request.arrival_ns) - (e.completion_ns - request.arrival_ns)) if s is not None else None,
                     "serial_numeric_vs_python_duration_relative_error": metric_diff(s.completion_ns - request.arrival_ns, e.completion_ns - request.arrival_ns)[0]["relative_error"] if s is not None else None})
    timeline_diffs = metric_diff(timeline_state(actual), timeline_state(reference))
    state_diffs = metric_diff(core_state(actual), core_state(reference))
    all_diffs = timeline_diffs + state_diffs
    discrete_state_mismatches = discrete_mismatches(timeline_state(actual), timeline_state(reference), "timeline")
    discrete_state_mismatches += discrete_mismatches(core_state(actual), core_state(reference), "core")
    serial_timeline_error = serial_state_error = None
    serial_discrete_mismatches = []
    if serial_numeric is not None:
        serial_timeline_error = summarize_diffs(metric_diff(timeline_state(serial_numeric), timeline_state(reference)))
        serial_state_error = summarize_diffs(metric_diff(core_state(serial_numeric), core_state(reference)))
        serial_discrete_mismatches = discrete_mismatches(timeline_state(serial_numeric), timeline_state(reference), "timeline")
        serial_discrete_mismatches += discrete_mismatches(core_state(serial_numeric), core_state(reference), "core")
    result = {
        "case": name, "memory_type": memory_type, "config": asdict(config),
        "request_count": len(requests), "requests": rows,
        "max_completion_error": max_completion,
        "max_duration_error": max_duration,
        "timeline_error": summarize_diffs(timeline_diffs),
        "core_state_error": summarize_diffs(state_diffs),
        "integer_counter_mismatch_count": len(all_integer_mismatches),
        "integer_counter_mismatches": all_integer_mismatches,
        "integer_counters_pass": not all_integer_mismatches,
        "discrete_timeline_or_state_mismatch_count": len(discrete_state_mismatches),
        "discrete_timeline_or_state_mismatches": discrete_state_mismatches,
        "discrete_state_pass": not discrete_state_mismatches,
        "serial_numeric_vs_python_max_completion_absolute_error_ns": max((row["serial_numeric_vs_python_absolute_error_ns"] or 0.0 for row in rows), default=0.0),
        "serial_numeric_vs_python_max_duration_absolute_error_ns": max((row["serial_numeric_vs_python_duration_absolute_error_ns"] or 0.0 for row in rows), default=0.0),
        "serial_numeric_vs_python_integer_counter_mismatch_count": len(serial_integer_mismatches),
        "serial_numeric_vs_python_integer_counter_mismatches": serial_integer_mismatches,
        "serial_numeric_vs_python_timeline_error": serial_timeline_error,
        "serial_numeric_vs_python_core_state_error": serial_state_error,
        "serial_numeric_vs_python_discrete_state_mismatches": serial_discrete_mismatches,
        "serial_numeric_vs_python_integer_counters_pass": not serial_integer_mismatches and not serial_discrete_mismatches,
        "all_numeric_state_error": summarize_diffs(all_diffs),
    }
    return result


def preset_inventory():
    inventory = []
    for preset in list_architecture_presets():
        detail = architecture_preset_detail(preset["id"])
        components = []
        for component in detail["hardware"]["components"]:
            raw = component.get("metadata", {}).get("physical_memory_config")
            parsed = parse_physical_memory_config(raw) if raw else None
            components.append({"component_id": component["component_id"],
                               "kind": component.get("kind"),
                               "component_preset_id": component.get("metadata", {}).get("component_preset_id"),
                               "physical_memory_config_present": raw is not None,
                               "parsed_memory_kind": (parsed.kind.value if isinstance(parsed, DramConfig) else parsed.kind) if parsed else None,
                               "generation": parsed.generation if isinstance(parsed, DramConfig) else None})
        inventory.append({"hardware_preset_id": preset["id"], "components": components,
                          "physical_memory_config_components": [x["component_id"] for x in components if x["physical_memory_config_present"]],
                          "status": "available" if any(x["physical_memory_config_present"] for x in components) else "no physical_memory_config in preset"})
    return inventory


def run():
    inventory = preset_inventory()
    native_gddr_raw = get_component_preset("gddr7-16gb-30_0-256bit").component.metadata["physical_memory_config"]
    native_gddr = parse_physical_memory_config(native_gddr_raw)
    # DDR5 appears in the native architecture as a DDR5-5600 host-memory
    # component preset, but that component does not publish physical_memory_config.
    # Therefore this constructed DDR5 config is clearly labeled as derived, not
    # mistaken for an actual declared physical_memory_config.
    ddr5_derived = DramConfig(
        kind=MemoryKind.DDR, channels=1, data_lanes=2, data_width_bits=64,
        data_rate_mt_s=5600, interface_bandwidth_gb_s=44.8,
        ranks_per_channel=1, bank_groups_per_rank=8, banks_per_group=4,
        rows_per_bank=65536, row_bytes=1024, burst_bytes=64,
        open_ns=14, close_ns=14, read_latency_ns=60, write_latency_ns=60,
        burst_interval_ns=64 / 44.8,
        metadata={"audit_basis": "derived from native Samsung DDR5-5600 module throughput; geometry/timing are representative, not declared physical_memory_config"},
    )
    dram_configs = [("native_gddr7_physical_memory_config", native_gddr),
                    ("native_ddr5_5600_derived_representative", ddr5_derived),
                    ("heterogeneous_shared_command", generic_dram(data_lanes=3, banks_per_group=1, row_bytes=512, interleave_bytes=128,
                                                                  metadata={"shared_command_resource": "dram:command:shared"})),
                    ("nested_resource_alias_fallback", generic_dram(metadata={"data_resource_prefix": "dram:bank:0:0:0:0:0"}))]
    dram_results = []
    for label, config in dram_configs:
        burst = config.burst_bytes
        address = 193 if label not in {"native_gddr7_physical_memory_config", "native_ddr5_5600_derived_representative"} else 64
        size = 10000 * burst + 7
        requests = [
            AccessRequest(f"{label}-warm", Operation.READ, address, burst, 0.0),
            AccessRequest(f"{label}-size-512-bursts", Operation.READ, address, 512 * burst, 0.0),
            AccessRequest(f"{label}-size-8193-bursts", Operation.READ, address, 8193 * burst, 0.0),
            AccessRequest(f"{label}-aligned-read", Operation.READ, address, size, 0.0),
            AccessRequest(f"{label}-mixed-write", Operation.WRITE, address, size, 0.0),
            AccessRequest(f"{label}-read-followup", Operation.READ, address, size, 0.0),
        ]
        if config.effective_capacity_bytes > size:
            high_address = config.effective_capacity_bytes - size
            requests.extend([
                AccessRequest(f"{label}-high-address-long-clock", Operation.READ, high_address, size, 1e12),
                AccessRequest(f"{label}-high-address-write-followup", Operation.WRITE, high_address, 512 * burst, 1e12),
                AccessRequest(f"{label}-very-long-clock-1e15", Operation.READ, high_address, 8193 * burst, 1e15),
            ])
        else:
            requests.append(AccessRequest(f"{label}-long-clock", Operation.READ, address + size - burst, burst, 1e12))
        dram_results.append(compare_case(label, "dram", config, requests))
    # Explicit 100k repeated writes reproduces the long-serial-add roundoff case.
    roundoff_cfg = nand_config(page_bytes=4096, host_granularity_bytes=4096,
                               pages_per_block=256, blocks_per_plane=512, channels=4,
                               host_bandwidth_gb_s=3.7, internal_bandwidth_gb_s=11.3,
                               page_read_ns=50000.2, page_program_ns=200000.1, front_ns=7.1)
    nand_results = []
    nand_results.append(compare_case(
        "nand_aligned_read_write_and_partial_rmw", "nand", nand_config(), [
            AccessRequest("aligned-read", Operation.READ, 0, 5000 * 64, 0.0),
            AccessRequest("aligned-write", Operation.WRITE, 0, 5000 * 64, 0.0),
            AccessRequest("partial-rmw-write", Operation.WRITE, 17, 5000 * 64 - 19, 0.0),
            AccessRequest("partial-rmw-read-followup", Operation.READ, 17, 5000 * 64 - 19, 10_000_000.0),
        ]))
    nand_results.append(compare_case(
        "nand_shared_host_alias_and_followup", "nand",
        nand_config(metadata={"host_resource_id": "shared:host", "channel_resource_prefix": "nand:array"}), [
            AccessRequest("alias-partial-read", Operation.READ, 17, 5000 * 64 - 19, 0.0),
            AccessRequest("alias-mixed-write", Operation.WRITE, 17, 5000 * 64 - 19, 0.0),
        ]))
    nand_results.append(compare_case(
        "nand_100000_page_write_roundoff", "nand", roundoff_cfg,
        [AccessRequest("100000-page-write", Operation.WRITE, 0, 100000 * 4096, 0.0),
         AccessRequest("100000-page-read-followup", Operation.READ, 0, 100000 * 4096, 1e12)]))
    return {
        "title": "Measured exact memory acceleration precision audit",
        "date": "2026-10-07",
        "purpose": "Quantify current fast-path floating-point drift against burst/page reference loops.",
        "scope_limit": "These are software equivalence measurements for listed representative configurations, not guarantees for all configurations and not calibration against physical hardware.",
        "method": {
            "dram_reference": "DramCore subclass with _can_accelerate() == False, using burst execution.",
        "nand_reference": "NandCore subclass overriding _execute_accepted() to call _execute_detailed_accepted(), using page execution.",
        "dram_numeric_reference": "For DRAM cases, the compiled wrapper is also run with tools/precision_dram_reference.py run_numeric_serial, which visits each burst without row-cycle folding; this is compared against the original Python per-burst reference.",
            "comparison": "Measured per-request completion_ns and final resource calendars/busy/state; integer byte and operation counters are compared exactly and any mismatch fails the script.",
            "relative_error_denominator": "abs(reference value); if reference is zero and actual is zero relative error is 0; if reference is zero and actual is nonzero, the value is counted separately and relative_error is null.",
            "duration_relative_error_denominator": "For each request, abs(reference_completion_ns - request.arrival_ns); completion timestamp relative error separately uses abs(reference_completion_ns).",
            "floating_point_error_policy": "No tolerance is used to hide the measurement. Absolute and relative deltas are saved as observed IEEE-754 results.",
        },
        "architecture_preset_physical_memory_inventory": inventory,
        "physical_config_observation": {
            "native_rtx_preset": "Declares physical_memory_config for GDDR7 only. Its hostmem0 is the Samsung DDR5-5600 component preset but has no physical_memory_config field.",
            "native_ddr5_audit": "Included as a separately labeled derived representative configuration based on the declared DDR5-5600 throughput; its geometry/timings are not claimed as native physical config.",
            "b200_presets": "Both B200 architecture presets have no physical_memory_config on their HBF/HBM/other components, so no physical acceleration error is claimed for those presets.",
        },
        "dram_cases": dram_results,
        "nand_cases": nand_results,
        "integer_counters_pass": all(row["integer_counters_pass"] for row in dram_results + nand_results)
                                  and all(row["discrete_state_pass"] for row in dram_results + nand_results)
                                  and all(row["serial_numeric_vs_python_integer_counters_pass"] for row in dram_results),
    }


def main():
    started = time.monotonic()
    report = run()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "elapsed_seconds": report["elapsed_seconds"],
                      "integer_counters_pass": report["integer_counters_pass"],
                      "dram_cases": len(report["dram_cases"]), "nand_cases": len(report["nand_cases"]),
                      "nand_100000_page_completion_error": report["nand_cases"][-1]["max_completion_error"]}, ensure_ascii=False), flush=True)
    if not report["integer_counters_pass"]:
        raise SystemExit("integer counter mismatch; see report")


if __name__ == "__main__":
    main()
