"""Audit immutable native payloads eligible for L1 Engine replay.

This scanner never launches llama.cpp and never uses native latency to fit a
simulator parameter.  It classifies payloads so partial/legacy records cannot
silently enter the Engine acceptance denominator.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_simulator_from_native import _validate_native_evidence  # noqa: E402


ALIASES = {
    "qwen25": ("qwen25", "qwen2.5"),
    "qwen35": ("qwen35", "qwen3.5"),
    "qwen38": ("qwen38", "qwen3.8"),
    "tinyllama": ("tinyllama",),
    "smollm2": ("smollm2", "smollm-2"),
}


def _family(model: object) -> str:
    text = str(model or "").lower()
    return next((key for key, values in ALIASES.items()
                 if any(value in text for value in values)), "unknown")


def _is_payload(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return (isinstance(value.get("native"), dict)
            and bool(value.get("model"))
            and isinstance(value.get("request"), dict)
            and isinstance(value.get("identity"), dict))


def scan(source_dirs: list[Path]) -> dict:
    rows = []
    skipped_non_payload = 0
    skipped_unknown_model = 0
    for directory in source_dirs:
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".prediction.json"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not _is_payload(payload):
                skipped_non_payload += 1
                continue
            family = _family(payload.get("model"))
            if family == "unknown":
                skipped_unknown_model += 1
                continue
            native = payload["native"]
            evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
            engine_timing = evidence.get("engine_timing") if isinstance(evidence.get("engine_timing"), dict) else {}
            fields = {name: native.get(name) for name in
                      ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")
                      if native.get(name) is not None}
            if not fields:
                continue
            errors = []
            if len(fields) == 3:
                errors, _ = _validate_native_evidence(payload, source_path=path)
            contract_id = (evidence.get("timing_contract") or {}).get("id")
            if (len(fields) == 3 and engine_timing.get("status") in ("counter_proven", "marker_proven")
                    and not errors):
                status = "replayable_engine_v3" if contract_id == "engine-stage+client-real-token/v3" else "full_engine_contract_migration_required"
            elif len(fields) == 3:
                status = "full_engine_evidence_invalid_or_stale"
            elif "engine_tpot_ms" in fields:
                status = "partial_engine_tpot_only"
            else:
                status = "stage_or_client_only"
            config = payload.get("configuration") if isinstance(payload.get("configuration"), dict) else {}
            counts = payload.get("token_counts") if isinstance(payload.get("token_counts"), dict) else {}
            identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
            rows.append({
                "source": str(path.resolve().relative_to(ROOT)).replace("\\", "/"),
                "family": family,
                "model": Path(str(payload.get("model"))).name,
                "prompt_tokens": counts.get("prompt"),
                "output_tokens": counts.get("output"),
                "parallel": config.get("parallel"),
                "gpu_layers": config.get("gpu_layers"),
                "engine_fields": sorted(fields),
                "engine_timing_status": engine_timing.get("status"),
                "timing_contract_id": contract_id,
                "validity_status": payload.get("validity_status"),
                "identity_complete": all(identity.get(key) not in (None, "", {}, []) for key in
                                          ("model_path", "gguf_sha256", "runtime_fingerprint",
                                           "hardware_fingerprint", "prompt_fingerprint", "configuration")),
                "status": status,
                "validation_errors": errors[:8],
            })
    aggregate = {}
    for family in sorted({row["family"] for row in rows}):
        group = [row for row in rows if row["family"] == family]
        aggregate[family] = {
            "payload_count": len(group),
            "replayable_engine_v3": sum(row["status"] == "replayable_engine_v3" for row in group),
            "full_engine_contract_migration_required": sum(row["status"] == "full_engine_contract_migration_required" for row in group),
            "partial_engine_tpot_only": sum(row["status"] == "partial_engine_tpot_only" for row in group),
            "full_engine_evidence_invalid_or_stale": sum(row["status"] == "full_engine_evidence_invalid_or_stale" for row in group),
            "gpu_layers": sorted({row["gpu_layers"] for row in group}),
        }
    return {
        "schema": "engine-holdout-candidate-audit/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Identify immutable native payloads eligible for L1 Engine simulator-only replay before E-01.",
        "policy": {
            "native_execution_count": 0,
            "l1_requires": ["all three explicit engine fields", "counter_proven or marker_proven evidence",
                            "current v3 timing contract", "identity and token boundary evidence"],
            "partial_payload_use": "diagnostic_only",
        },
        "payload_scan_count": len(rows),
        "identity_reuse_count": sum(row["identity_complete"] for row in rows),
        "aggregate": aggregate,
        "skipped_non_payload_count": skipped_non_payload,
        "skipped_unknown_model_count": skipped_unknown_model,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = scan(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(result["rows"]),
                      "aggregate": result["aggregate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
