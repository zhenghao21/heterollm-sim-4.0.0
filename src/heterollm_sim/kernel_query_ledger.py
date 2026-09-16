"""Compact simulated physical-GEMM geometry; never native dispatch evidence."""
from __future__ import annotations
from collections import Counter
import json
from collections.abc import Mapping


def summarize_kernel_queries(tasks):
    signatures, reasons, seen = Counter(), Counter(), {}
    total = 0
    for task in tasks:
        meta = task.metadata
        if meta.get("phase") != "gpu_gemm":
            continue
        identity = getattr(task, "task_id", None)
        if not isinstance(identity, str) or not identity:
            raise ValueError("kernel query ledger requires task identity")
        geometry = meta.get("kernel_query_geometry")
        fingerprint = json.dumps({"geometry": geometry, "target": meta.get("target_component"),
            "consumer": meta.get("kernel_main_consumer_storage_bytes")}, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if identity in seen:
            if seen[identity] != fingerprint:
                raise ValueError("conflicting geometry for the same physical task identity")
            continue
        seen[identity] = fingerprint
        total += 1
        if not isinstance(geometry, Mapping):
            reasons["missing_typed_geometry"] += 1
            continue
        if any(type(geometry.get(key)) is not int or geometry[key] <= 0 for key in ("m", "n", "k_logical")):
            reasons["invalid_typed_shape"] += 1
            continue
        formats = geometry.get("weight_formats")
        if (not isinstance(formats, (tuple, list)) or not formats
                or any(not isinstance(value, str) or not value for value in formats)
                or not isinstance(meta.get("target_component"), str) or not meta["target_component"]
                or any(type(geometry.get(key)) is not int or geometry[key] <= 0 for key in
                    ("activation_storage_bytes", "output_storage_bytes", "accumulator_bits"))
                or type(meta.get("kernel_main_consumer_storage_bytes")) is not int
                or meta["kernel_main_consumer_storage_bytes"] <= 0
                or type(geometry.get("model_weight_read")) is not bool or type(geometry.get("rhs_is_activation")) is not bool):
            reasons["incomplete_format_device_or_storage"] += 1
            continue
        # Select only explicit typed facts. No model, prompt or time fields.
        row = {key: geometry.get(key) for key in (
            "m", "n", "k_logical", "weight_formats", "activation_storage_bytes",
            "output_storage_bytes", "accumulator_bits", "model_weight_read", "rhs_is_activation")}
        row["logical_input_storage_bytes"] = row.pop("activation_storage_bytes")
        row["main_consumer_storage_bytes"] = meta["kernel_main_consumer_storage_bytes"]
        row["target_component"] = meta.get("target_component")
        audit = meta.get("mmq_source_work", {})
        status = audit.get("status") if isinstance(audit, Mapping) else None
        row["source_path_status"] = status or "unqualified"
        row["predicted_family"] = "MMQ" if status == "applied" else "MMVQ" if status == "mmvq_precedes_mmq" else "unknown"
        row["k_executed"] = audit.get("k_execution") if status == "applied" else None
        row["cache_state"] = "unknown"
        row["native_dispatch_proven"] = False
        row["layout_proven"] = False
        row["calibration_eligible"] = False
        signatures[json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)] += 1
    return {"schema": "heterollm.simulated-kernel-query-ledger/v1",
        "count_scope": "deduplicated simulator gpu_gemm tasks; main only, no inferred native kernel counts",
        "gpu_gemm_tasks": total, "represented_tasks": sum(signatures.values()),
        "unrepresented_tasks": sum(reasons.values()), "missing_reason_counts": dict(sorted(reasons.items())),
        "signatures": [{"key": json.loads(key), "task_count": count} for key, count in sorted(signatures.items())],
        "complete_geometry": sum(signatures.values()) == total,
        "native_timing_used": False, "calibration_eligible": False,
        "limitations": ["Cache/stride/actual dispatch unproven; shape matches alone do not qualify a profile.",
                        "MMVQ executed K unknown; no padding inferred from conversion storage."]}
