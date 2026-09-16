"""Compact simulated physical-GEMM geometry; never native dispatch evidence."""
from __future__ import annotations
from collections import Counter
import json
from collections.abc import Mapping
from .mmvq_work import MMVQSourceContract, derive_mmvq_work, UnsupportedMMVQ


def _same_source_value(actual, expected):
    # Saved JSON arrays and in-memory tuples represent the same sequence, but
    # integer/bool/float fields must never be interchangeable.
    if isinstance(expected, Mapping):
        return (isinstance(actual, Mapping) and set(actual) == set(expected)
                and all(_same_source_value(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, (tuple, list)):
        return (type(actual) in (tuple, list) and len(actual) == len(expected)
                and all(_same_source_value(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def _source_mmvq_executed_k(meta, geometry):
    """Re-derive declared logical-K work without upgrading native/layout proof."""
    audit = meta.get("mmvq_source_work")
    if not isinstance(audit, Mapping):
        return None, "missing_mmvq_source_work"
    if (audit.get("status") != "source_geometry_unpriced"
            or audit.get("stage") != "matrix"
            or geometry["model_weight_read"] is not True
            or geometry["rhs_is_activation"] is not False):
        return None, "unqualified_mmvq_source_work"
    formats = geometry["weight_formats"]
    if len(formats) != 1:
        return None, "mixed_weight_formats"
    try:
        contract = MMVQSourceContract(
            1200, 1200, 32, audit.get("source_hashes", {}),
            audit.get("runtime_binary_sha256"), True, False, True,
        )
        work = derive_mmvq_work(
            m=geometry["m"], k=geometry["k_logical"], n=geometry["n"],
            weight_format=formats[0].upper(), contract=contract, allow_k_formats=True,
        )
        expected = work.to_metadata()
        actual = {key: audit.get(key) for key in expected}
        # JSON normalizes tuples produced by the planner and lists after save/load.
        if not _same_source_value(actual, expected):
            return None, "incomplete_or_noncanonical_mmvq_source_work"
        if (meta["kernel_main_consumer_storage_bytes"] != work.consumer_q8_1_unique_bytes
                or geometry["activation_storage_bytes"] != work.logical_input_f32_bytes
                or geometry["output_storage_bytes"] != work.output_f32_bytes
                or geometry["accumulator_bits"] != 32):
            return None, "mmvq_source_storage_mismatch"
        if audit.get("execution_component") != meta["target_component"]:
            return None, "mmvq_source_execution_component_mismatch"
        # These optional assertions must not contradict the narrow contract.
        # Absence leaves native/layout proof false; it never establishes layout.
        conditions = {"has_ids": False, "has_fusion": False,
            "channels": 1, "channel": 1, "nchannels": 1, "channel_count": 1,
            "samples": 1, "sample": 1, "nsamples": 1, "sample_count": 1,
            "ordinary_contiguous_2d": True, "force_cublas": False,
            "mmvq_dispatch_enabled": True, "layout": "ordinary_contiguous_2d"}
        if any(key in audit and not _same_source_value(audit[key], value)
                for key, value in conditions.items()):
            return None, "contradictory_mmvq_source_layout_or_dispatch"
    except (KeyError, TypeError, ValueError, UnsupportedMMVQ):
        return None, "unsupported_mmvq_source_contract"
    return work.k, "source_derived_mmvq_logical_K"



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
            "consumer": meta.get("kernel_main_consumer_storage_bytes"),
            "mmq_source_work": meta.get("mmq_source_work"), "mmvq_source_work": meta.get("mmvq_source_work")}, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
        row["k_execution_source"] = "existing_mmq_source_execution_extent" if status == "applied" else "unknown"
        row["k_execution_native_proven"] = False
        if status == "mmvq_precedes_mmq":
            row["k_executed"], basis = _source_mmvq_executed_k(meta, geometry)
            row["mmvq_k_execution_reason"] = basis
            if row["k_executed"] is not None:
                row["k_execution_source"] = basis
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
                        "MMVQ executed K is source-derived only with complete canonical work; otherwise unknown. No padding inferred from conversion storage."]}
