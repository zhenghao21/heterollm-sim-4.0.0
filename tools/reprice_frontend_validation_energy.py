"""Reprice saved pure-DRAM execution records; never execute or alter a job."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RATE_FIELDS = {"physical_energy_pj_per_byte", "physical_energy_pj_per_byte_by_owner"}
DRAM_KINDS = {"ddr", "lpddr", "hbm", "gddr", "host_memory", "dram"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def explicit_rate_paths(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = path + "/" + str(key)
            if key in RATE_FIELDS:
                yield child
            yield from explicit_rate_paths(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from explicit_rate_paths(item, path + "/" + str(index))


def audit_generation_paths(root=ROOT):
    """Check the scope of the manually reviewed coefficient producer paths.

    The saved aggregate archives do not retain individual task metadata. The
    independent source audit established that ordinary generated DRAM tasks
    use component profile rates; historical endpoint lowering omitted a rate.
    A new use of either override field requires a new review, not automatic
    extension of this restricted accounting operation.
    """
    allowed = {
        "planner.py": {"_attach_direct_backing_physical_task", "_attach_physical_profile_energy",
                       "_attach_gddr_physical_task"},
        "data_motion.py": {"resolve_physical_task"},
    }
    occurrences = []
    for path in sorted((root / "src/heterollm_sim").rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        if not any(field in source for field in RATE_FIELDS):
            continue
        tree = ast.parse(source)

        def visit(node, function=None):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = node.name
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in RATE_FIELDS):
                require(path.parent == root / "src/heterollm_sim" and function in allowed.get(path.name, set()),
                        f"unreviewed physical energy field use: {path.name}:{node.lineno} {function}")
                occurrences.append({"file": str(path.relative_to(root)).replace("\\", "/"),
                                    "function": function, "line": node.lineno, "field": node.value})
            for child in ast.iter_child_nodes(node):
                visit(child, function)

        visit(tree)
    require({row["file"].split("/")[-1] for row in occurrences} == set(allowed),
            "reviewed physical energy generation paths not found")
    return {
        "status": "reviewed_current_producers_with_restricted_field_scope",
        "task_event_metadata_retained": False,
        "basis": [
            "planner direct DRAM, GPU backing and CPU backing coefficient producers read the resolved component profile",
            "already-physical endpoint lowering historically omitted a coefficient; the new bridge fills missing rates and preserves explicit overrides",
            "data_motion resolves physical bytes and consumes those coefficients; DRAM resource data bytes equal physical media bytes",
            "submitted metadata and saved archive must contain no explicit task coefficient overrides",
            "this audit is restricted to the saved generated preset/native-pair paths, not arbitrary TaskSpec programs",
        ],
        "field_occurrences": occurrences,
    }


def physical_resources(ledger):
    require(isinstance(ledger, dict), "missing DRAM ledger")
    rows = ledger.get("resource_totals")
    require(isinstance(rows, dict) and rows, "missing per-resource physical ledger")
    for name, row in rows.items():
        require(isinstance(row, dict) and isinstance(row.get("owner"), str), f"missing owner for {name}")
        require(type(row.get("bytes_moved")) is int and row["bytes_moved"] >= 0,
                f"invalid physical bytes for {name}")
        require(finite_nonnegative(row.get("energy_pj")), f"invalid physical energy for {name}")
    require(sum(row["bytes_moved"] for row in rows.values()) == ledger.get("physical_bytes"),
            "DRAM resource bytes do not equal physical media bytes")
    require(ledger.get("physical_read_bytes", -1) + ledger.get("physical_write_bytes", -1)
            == ledger.get("physical_bytes"), "DRAM direction bytes do not equal physical media bytes")
    return rows


def require_unused_nand(ledger):
    require(isinstance(ledger, dict) and ledger.get("task_count") == 0
            and not ledger.get("resource_totals")
            and all(ledger.get(key, 0) == 0 for key in
                    ("logical_bytes", "physical_bytes", "physical_read_bytes", "physical_write_bytes", "energy_pj")),
            "NAND repricing is outside this tool's scope")


def reprice_record(scenario, job, raw_job, *, prefix, input_accounting):
    require(job.get("status") == raw_job.get("status") == "completed", "job is not completed")
    require(job.get("job_id") and job["job_id"] == raw_job.get("job_id"), "raw job identity differs")
    require(not next(explicit_rate_paths(scenario), None), "submitted explicit task energy coefficient override")
    require(not next(explicit_rate_paths(raw_job), None), "archive has task energy overrides requiring individual review")
    report, raw_report = job["report"], raw_job["report"]
    summary = report["summary"]
    require(summary == raw_report.get("summary"), "raw summary differs from saved result")
    require(report.get("requests") == raw_report.get("requests"), "raw requests differ from saved result")
    require_unused_nand(summary.get("storage_traffic"))
    batches = raw_report.get("batch_history")
    require(isinstance(batches, list) and len(batches) == summary.get("batch_count")
            == summary.get("physical_live_batch_count") and len(batches) > 0,
            "complete persistent-live batch archive is required")
    profiles = {}
    hardware = scenario.get("hardware_input", {}).get("hardware", {})
    require(hardware, "explicit authoritative hardware_input is required")
    for component in hardware.get("components", []):
        binding = component.get("execution_profile", {})
        profile = binding.get("parameters", {})
        if component.get("kind") not in DRAM_KINDS or "energy_pj_per_byte" not in profile:
            continue
        owner = profile.get("resource_id")
        require(isinstance(owner, str) and owner and owner not in profiles, "ambiguous physical owner profile")
        rate = profile["energy_pj_per_byte"]
        require(finite_nonnegative(rate), "invalid authored profile coefficient")
        profiles[owner] = {"component_id": component["component_id"], "profile_id": binding.get("profile_id"),
                           "profile_kind": binding.get("profile_kind"), "coefficient_pj_per_byte": rate}
    resources = physical_resources(summary.get("dram_traffic"))
    merged = {}
    batch_energies = []
    for batch in batches:
        cost = batch["cost"]
        metadata = cost["metadata"]
        require(metadata.get("physical_execution_scope") == "persistent_live_kernel", "batch is not live physical execution")
        require_unused_nand(metadata.get("storage_traffic"))
        require(finite_nonnegative(cost.get("energy_pj")), "invalid batch energy")
        batch_energies.append(cost["energy_pj"])
        for name, row in physical_resources(metadata.get("dram_traffic")).items():
            target = merged.setdefault(name, {"owner": row["owner"], "bytes_moved": 0, "energy_pj": 0.0})
            require(target["owner"] == row["owner"], "resource owner changed between batches")
            for key in ("bytes_moved", "energy_pj"):
                target[key] += row[key]
    # reporting uses built-in sum (compensated on the bundled Python), not a
    # manual += loop; preserve its summation semantics for this equality check.
    require(sum(batch_energies) == summary.get("total_energy_pj"), "batch energy does not reproduce saved total")
    require(set(merged) == set(resources), "batch physical resource coverage differs")
    for name, row in resources.items():
        require(all(merged[name][key] == row[key] for key in ("owner", "bytes_moved", "energy_pj")),
                "batch physical resource sums differ from saved total")
    owners = {}
    for row in resources.values():
        owner = row["owner"]
        require(owner in profiles, f"unknown physical profile owner {owner}")
        target = owners.setdefault(owner, {"owner": owner, **profiles[owner], "physical_bytes": 0,
                                          "original_physical_energy_pj": 0.0})
        target["physical_bytes"] += row["bytes_moved"]
        target["original_physical_energy_pj"] += row["energy_pj"]
    for row in owners.values():
        row["repriced_physical_energy_pj"] = row["physical_bytes"] * row["coefficient_pj_per_byte"]
        row["delta_energy_pj"] = row["repriced_physical_energy_pj"] - row["original_physical_energy_pj"]
    original_physical = math.fsum(row["original_physical_energy_pj"] for row in owners.values())
    repriced_physical = math.fsum(row["repriced_physical_energy_pj"] for row in owners.values())
    original_total = summary["total_energy_pj"]
    require(finite_nonnegative(original_total) and original_total >= original_physical,
            "physical energy exceeds original total")
    repriced_total = math.fsum((original_total, -original_physical, repriced_physical))
    validation = input_accounting == "physical_profile_energy_v2"
    if validation:
        require(all(row["delta_energy_pj"] == 0 for row in owners.values()),
                "new v2 execution does not match profile repricing per owner")
        require(abs(repriced_total - original_total) <= math.ulp(original_total),
                "new v2 total differs beyond one final floating-point rounding unit")
    return {
        "prefix": prefix, "job_id": job["job_id"],
        "status": "validated_against_v2_execution" if validation else "repriced",
        "method": "same_execution_physical_record_repricing", "label_zh": "同次物理执行记录重新计费",
        "input_energy_accounting": input_accounting, "owners": list(owners.values()),
        "batch_count_verified": len(batches), "original_total_energy_pj": original_total,
        "original_physical_energy_pj": original_physical, "repriced_physical_energy_pj": repriced_physical,
        "unchanged_nonphysical_energy_pj": original_total - original_physical,
        "repriced_total_energy_pj": repriced_total, "delta_energy_pj": repriced_total - original_total,
        "new_simulation_executed": False,
        "v2_reconstruction_absolute_difference_pj": abs(repriced_total - original_total) if validation else None,
    }


def build(directory):
    audit = audit_generation_paths()
    comparison = read_json(directory / "comparison.json")
    entries = []
    for attempt in comparison.get("attempts", []):
        if attempt.get("status") != "completed" or attempt.get("record_kind") != "run_submission":
            continue
        prefix = attempt["prefix"]
        try:
            scenario = read_json(directory / attempt["submission_file"])["scenario"]
            job = read_json(directory / attempt["result_file"])
            recording = job.get("recording", {})
            require(recording.get("raw_archive_verified") == "full_json_round_trip_equal_before_compaction",
                    "verified original archive is required")
            archive = Path(recording["raw_archive_path"])
            require(archive.is_file(), "original archive is not available")
            with gzip.open(archive, "rt", encoding="utf-8") as stream:
                raw_job = json.load(stream)
            row = reprice_record(scenario, job, raw_job, prefix=prefix,
                                 input_accounting=attempt.get("energy_accounting", "not_revalidated"))
            row.update(submission_file=attempt["submission_file"], result_file=attempt["result_file"],
                       raw_archive_path=str(archive))
        except (ValueError, KeyError, OSError, TypeError) as error:
            row = {"prefix": prefix, "job_id": attempt.get("job_id"), "status": "not_repriced", "reason": str(error)}
        entries.append(row)
    verified = [row for row in entries if row["status"] == "validated_against_v2_execution"]
    return {
        "schema": "frontend-physical-energy-repricing/v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "saved_completed_pure_dram_generated_scenarios_only", "source_generation_audit": audit,
        "status_counts": dict(Counter(row["status"] for row in entries)), "entries": entries,
        "validation": {"v2_execution_count": len(verified),
            "maximum_total_reconstruction_difference_pj": max(
                (row["v2_reconstruction_absolute_difference_pj"] for row in verified), default=None)},
        "limitations": ["No job or original result was changed or rerun.",
            "Uses the same execution's physical byte records and submitted profile coefficients, not measured hardware power.",
            "Aggregate archives omit task metadata; restricted source producer audit and absence of submitted overrides are required.",
            "NAND and unknown/ambiguous coefficients are refused. This is not an execution replay.",
            "Floating-point total summation may differ in the last representable digit."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "docs/frontend_native_validation_2026-10-07")
    args = parser.parse_args()
    result = build(args.directory)
    target = args.directory / "energy_repricing.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status_counts": result["status_counts"], "validation": result["validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
