"""Validate and summarize the saved full frontend preset matrix."""
from collections import Counter
import json
from pathlib import Path

from heterollm_sim.architecture_presets import list_architecture_presets
from heterollm_sim.model_presets import list_model_presets

from validate_preset_matrix import LLAMA_DEFAULTS, report_checks


def main():
    directory = Path(__file__).resolve().parents[1] / "docs"
    paths = [directory / name for name in (
        "exact_matrix_native_2026-10-07.json", "exact_matrix_b200_2hbf_2026-10-07.json",
        "exact_matrix_b200_3hbf_2026-10-07.json")]
    models = {row["id"] for row in list_model_presets()}
    hardware = {row.get("preset_id", row.get("id")) for row in list_architecture_presets()}
    all_rows, groups = [], []
    for path in paths:
        group = json.loads(path.read_text(encoding="utf-8"))
        rows = group["results"]
        if {row["model"] for row in rows} != models or len(rows) != len(models):
            raise ValueError(f"Incomplete or duplicate model coverage: {path.name}")
        for row in rows:
            if row["hardware"] != group["hardware"]:
                raise ValueError(f"Hardware mismatch: {row['model']}")
            if row["input"] != {"requests": 1, "prompt_tokens": 512, "output_tokens": 128,
                                "llama_cpp": LLAMA_DEFAULTS, "retention_policy": "aggregate"}:
                raise ValueError(f"Default workload changed: {row['model']}")
            if row.get("report_available"):
                if row["status"] not in {"completed", "validation_failed"}:
                    raise ValueError(f"Non-completed job with a report: {row['model']}: {row['status']}")
                if row.get("api_job_status", "completed") != "completed":
                    raise ValueError(f"API job did not complete: {row['model']}")
                failures = report_checks({"error": row.get("error"),
                                          "report": {"summary": row["summary"], "requests": row["requests"]}})
                if failures:
                    raise ValueError(f"Report validation failed: {row['model']}: {failures}")
                row.update(status="completed", api_job_status="completed", validation_failures=[])
            elif row["status"] != "capacity_rejected" or "llama.cpp shared device-memory capacity exhausted" not in json.dumps(row.get("error")):
                raise ValueError(f"Unexpected run failure: {row['model']}: {row.get('error')}")
        totals = dict(Counter(row["status"] for row in rows))
        groups.append({"hardware": group["hardware"], "totals": totals, "detail_file": path.name})
        group["totals"] = totals
        group["validation"] = {"completed_report_checks_passed": totals.get("completed", 0),
                               "note": "An initial checker read absent summary.request_count; all saved reports were rechecked against the actual request rows and rejected_requests after correcting it."}
        path.write_text(json.dumps(group, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for row in rows:
            summary = row.get("summary") or {}
            requests = row.get("requests") or {}
            all_rows.append({
                "hardware": row["hardware"], "model": row["model"], "status": row["status"],
                "job_id": row.get("job_id"), "elapsed_s": row["elapsed_s"], "error": row.get("error"),
                "report_available": row.get("report_available", False),
                "batch_count": summary.get("batch_count"), "completed_requests": summary.get("completed_requests"),
                "visible_output_tokens": sum(r["visible_output_tokens"] for r in requests.values()) if requests else None,
                "task_count": summary.get("task_count"), "makespan_ns": summary.get("makespan_ns"),
                "ttft_ns": summary.get("ttft_ns"), "tpot_ns": summary.get("tpot_ns"),
                "throughput": summary.get("throughput"),
                "dram_traffic": {k: v for k, v in summary.get("dram_traffic", {}).items()
                                 if k not in {"resource_totals", "organization_profiles"}},
                "storage_traffic": {k: v for k, v in summary.get("storage_traffic", {}).items()
                                    if k not in {"resource_totals", "organization_profiles"}},
            })
    if {group["hardware"] for group in groups} != hardware:
        raise ValueError("Incomplete hardware coverage")
    totals = dict(Counter(row["status"] for row in all_rows))
    result = {"audit_date": "2026-10-07", "status": "complete", "implementation": "exact_state_transition_acceleration",
              "model_count": len(models), "hardware_count": len(hardware), "total_combinations": len(all_rows),
              "workload": {"requests": 1, "prompt_tokens": 512, "output_tokens": 128,
                           "llama_cpp": LLAMA_DEFAULTS, "retention_policy": "aggregate"},
              "totals": totals, "hardware_totals": groups,
              "notes": ["Every combination was submitted to the same /api/run-jobs interface used by the frontend.",
                        "Completed reports were checked for one finished request with 128 visible output tokens, 128 cohorts, positive finite timing, and traffic counter consistency.",
                        "Capacity rejection means the unchanged model exceeds the shared device-memory capacity with gpu_layers=-1. No workload reduction or capacity override was used.",
                        "elapsed_s includes host execution and polling during three parallel hardware groups; it is not a serial performance benchmark or predicted inference duration.",
                        "These are execution and internal-consistency checks, not real-hardware calibration. Existing analytical preset assumptions remain in effect."],
              "results": all_rows}
    output = directory / "exact_preset_matrix_results_2026-10-07.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    historical_path = directory / "preset_matrix_results_2026-10-07.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical["notes"][-1] = "The original matrix above used historical approximate bulk paths. The current implementation was rerun over all 63 combinations; see exact_preset_matrix_results_2026-10-07.json."
    historical["exact_acceleration_validation"]["full_matrix_validation"] = {
        "status": "complete", "total_combinations": len(all_rows), "totals": totals, "record": output.name}
    historical_path.write_text(json.dumps(historical, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "totals": totals, "hardware_totals": groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
