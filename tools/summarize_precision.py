"""Recheck saved complete runs and quantify acceleration drift."""
from collections import Counter
import json
from pathlib import Path

from compare_precision_matrix import numeric_differences
from validate_preset_matrix import report_checks


ROOT = Path(__file__).resolve().parents[1]


def extrema(differences):
    rows = list(differences.items())
    if not rows:
        return {"fields": 0, "maximum_absolute_difference": 0.0,
                "maximum_relative_difference": 0.0, "zero_reference_nonzero_count": 0}
    absolute = max(rows, key=lambda item: item[1]["absolute_difference"])
    relative = [(key, row) for key, row in rows if row["relative_difference"] is not None]
    largest_relative = max(relative, key=lambda item: item[1]["relative_difference"]) if relative else None
    return {"fields": len(rows), "maximum_absolute_difference": absolute[1]["absolute_difference"],
            "maximum_absolute_field": absolute[0],
            "maximum_relative_difference": largest_relative[1]["relative_difference"] if largest_relative else None,
            "maximum_relative_field": largest_relative[0] if largest_relative else None,
            "zero_reference_nonzero_count": sum(row["relative_difference"] is None for _, row in rows)}


def main():
    directory = ROOT / "docs"
    matrix = json.loads((directory / "exact_preset_matrix_results_2026-10-07.json").read_text(encoding="utf-8"))
    if matrix["status"] != "complete" or matrix["total_combinations"] != 63:
        raise ValueError("The full preset matrix has not completed")
    native = "local-native-rtx5080-9950x3d-gddr7-ddr5"
    expected = {row["model"] for row in matrix["results"]
                if row["hardware"] == native and row["status"] == "completed"}
    baseline = json.loads((directory / "exact_matrix_native_2026-10-07.json").read_text(encoding="utf-8"))
    baseline_rows = {row["model"]: row for row in baseline["results"]}
    combined, prediction_differences, service_differences, throughput_differences = [], {}, {}, {}
    seen = set()
    numeric_fields_compared = nonzero_numeric_fields = 0
    for number in (1, 2, 3, 4):
        path = directory / f"precision_native_group{number}_2026-10-07.json"
        group = json.loads(path.read_text(encoding="utf-8"))
        if group["hardware"] != native or group["reference_mode"] != "serial_numeric":
            raise ValueError(f"Wrong precision reference: {path.name}")
        for row in group["results"]:
            model = row["model"]
            if model in seen or model not in expected or row["status"] != "completed":
                raise ValueError(f"Incomplete, duplicate or failed precision case: {model}")
            seen.add(model)
            if row["input"] != baseline_rows[model]["input"]:
                raise ValueError(f"Workload changed: {model}")
            for report in (row["accelerated"], row["reference"]):
                failures = report_checks({"report": report, "error": None})
                if failures:
                    raise ValueError(f"Invalid precision report: {model}: {failures}")
            counts = row["reference_call_counts"]
            if (counts.get("dram_reference_method") != "serial_numeric_full_burst"
                    or counts.get("dram_reference_full_burst_traversal") is not True
                    or counts["dram_skipped_row_hit_bursts"] != 0
                    or counts["dram_python_accelerated_calls"] != 0
                    or counts["dram_compiled_results"] <= 0
                    or counts["dram_total_bursts"] <= 0
                    or counts["nand_accelerated_pages"] != 0):
                raise ValueError(f"Reference skipped state transitions: {model}")
            if any(delta.get("absolute_difference", delta.get("max_absolute_difference", 0)) != 0
                   for delta in row["baseline_vs_accelerated_probe_numeric_differences"].values()):
                raise ValueError(f"Fresh accelerated run differs from API baseline: {model}")
            differences = numeric_differences(row["accelerated"], row["reference"])
            numeric_fields_compared += len(differences)
            nonzero_fields = sum(delta.get("absolute_difference", delta.get("max_absolute_difference", 0)) != 0
                                 for delta in differences.values())
            nonzero_numeric_fields += nonzero_fields
            integer_mismatches = [key for key, delta in differences.items()
                                  if isinstance(delta.get("accelerated"), int)
                                  and isinstance(delta.get("reference"), int)
                                  and delta["accelerated"] != delta["reference"]]
            discrete_fields = {"task_count", "batch_count", "completed_requests", "rejected_requests",
                               "visible_output_tokens", "proposed_tokens", "accepted_tokens", "rejected_tokens",
                               "preemptions", "swaps", "recomputes", "row_hits", "row_misses", "row_conflicts",
                               "burst_count", "page_count", "pages_read", "pages_programmed", "erase_operations"}
            for key, delta in differences.items():
                leaf = key.rsplit(".", 1)[-1]
                byte_counter = leaf.endswith("_bytes") and key.startswith(("summary.dram_traffic.", "summary.storage_traffic."))
                if leaf in discrete_fields or byte_counter or key in {"summary.resource_accounted_bytes", "summary.mtp.declared_weight_bytes"}:
                    if (type(delta.get("accelerated")) is not int or type(delta.get("reference")) is not int
                            or delta["accelerated"] != delta["reference"]):
                        integer_mismatches.append(key)
            if integer_mismatches:
                raise ValueError(f"Integer counters changed: {model}: {integer_mismatches}")
            # Recompute even reports produced by an older checker, from raw values.
            row["baseline_vs_reference_numeric_differences"] = differences
            row["relative_error_denominator"] = "abs(reference); zero reference with nonzero actual is undefined"
            selected = {key: delta for key, delta in differences.items()
                        if key.startswith(("summary.makespan_ns", "summary.ttft_ns", "summary.tpot_ns"))}
            service = {key: delta for key, delta in differences.items()
                       if key in {"summary.dram_traffic.service_ns", "summary.storage_traffic.service_ns"}}
            throughput = {key: delta for key, delta in differences.items()
                          if key.startswith("summary.throughput.")}
            for target, source in ((prediction_differences, selected),
                                   (service_differences, service),
                                   (throughput_differences, throughput)):
                target.update({f"{model}:{key}": value for key, value in source.items()})
            combined.append({"hardware": native, "model": model, "status": "completed",
                             "numeric_fields_compared": len(differences), "nonzero_numeric_fields": nonzero_fields,
                             "integer_counter_mismatches": 0, "prediction_time_ns": extrema(selected),
                             "memory_service_ns": extrema(service), "throughput": extrema(throughput),
                             "accelerated_call_counts": row["accelerated_call_counts"],
                             "reference_call_counts": counts, "detail_file": path.name})
        group["relative_error_denominator"] = "abs(reference); all saved reports rechecked using raw values"
        path.write_text(json.dumps(group, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if seen != expected or len(seen) != 11:
        raise ValueError(f"Incomplete precision coverage: {sorted(expected - seen)}")
    memory_path = directory / "memory_precision_2026-10-07.json"
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    if not memory["integer_counters_pass"]:
        raise ValueError("Memory differential checks found integer/state mismatches")
    memory_cases = memory["dram_cases"] + memory["nand_cases"]
    memory_summary = {
        "record": memory_path.name, "dram_cases": len(memory["dram_cases"]),
        "nand_cases": len(memory["nand_cases"]), "integer_counter_mismatches": 0,
        "maximum_request_duration_error_ns": max(row["max_duration_error"]["absolute_error"] for row in memory_cases),
        "maximum_completion_timestamp_error_ns": max(row["max_completion_error"]["absolute_error"] for row in memory_cases),
        "maximum_request_duration_relative_error": max(row["max_duration_error"]["relative_error"] for row in memory_cases
                                                     if row["max_duration_error"]["relative_error"] is not None),
        "maximum_calendar_or_state_error_ns": max(row["all_numeric_state_error"]["max_absolute_error"] for row in memory_cases),
        "serial_numeric_vs_original_python_max_duration_error_ns": max(row["serial_numeric_vs_python_max_duration_absolute_error_ns"]
                                                                        for row in memory["dram_cases"]),
        "cases": [{"case": row["case"], "duration_error": row["max_duration_error"],
                   "calendar_error": row["timeline_error"], "state_error": row["core_state_error"]}
                  for row in memory_cases]}
    b200 = [row for row in matrix["results"] if row["hardware"] != native]
    if len(b200) != 42 or any(row["status"] != "completed"
                            or row["dram_traffic"].get("task_count", 0) != 0
                            or row["storage_traffic"].get("task_count", 0) != 0 for row in b200):
        raise ValueError("Unexpected physical memory execution in a B200 preset")
    result = {
        "date": "2026-10-07", "status": "complete", "full_matrix": {
            "record": "exact_preset_matrix_results_2026-10-07.json", "combinations": 63,
            "totals": dict(Counter(row["status"] for row in matrix["results"]))},
        "full_workload_differential_models": len(combined), "hardware": native,
        "method": "Same default 512/128 workload; every DRAM burst traversed in float64 with row folding disabled; NAND reference uses the original page loop. The numeric single-burst kernel is separately checked against the original Python burst loop.",
        "relative_error_denominator": "abs(reference prediction); not an absolute timestamp for request-duration comparisons",
        "prediction_time_ns": extrema(prediction_differences),
        "memory_service_ns": extrema(service_differences), "throughput": extrema(throughput_differences),
        "integer_counter_mismatches": 0,
        "numeric_fields_compared": numeric_fields_compared, "nonzero_numeric_fields": nonzero_numeric_fields,
        "b200_scope": {"combinations": 42, "physical_memory_acceleration_executed": False,
                       "reason": "These presets have no physical_memory_config and their full reports contain zero physical DRAM/NAND tasks; they use existing analytical memory profiles."},
        "memory_differential": memory_summary, "results": combined,
        "scope_limit": "Measured optimization drift for these runs and listed memory fixtures, not a universal numerical error bound and not validation against real hardware."}
    output = directory / "precision_results_2026-10-07.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "models": len(combined),
                      "prediction_time_ns": result["prediction_time_ns"],
                      "memory_service_ns": result["memory_service_ns"],
                      "throughput": result["throughput"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
