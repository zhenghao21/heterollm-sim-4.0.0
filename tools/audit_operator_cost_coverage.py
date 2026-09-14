"""Audit joint operator-cost evidence coverage without fitting target latency.

The report is intentionally descriptive: it only counts explicit semantic
events in existing traces and never changes simulator parameters.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _phase(shape: str | None) -> str:
    # llama.cpp semantic shapes are N x M x ...; M=1 is decode, M>1 prefill.
    try:
        m = int(str(shape or "").split("x")[1])
    except (IndexError, TypeError, ValueError):
        return "unknown"
    return "decode" if m == 1 else "prefill" if m > 1 else "unknown"


def _kernel_family(name: str | None) -> str:
    text = str(name or "").lower()
    if "mul_mat_vec_q" in text:
        return "gpu_mmq_vec"
    if "mul_mat_q" in text:
        return "gpu_mmq"
    if "mul_mat_vec" in text:
        return "gpu_mmvq_or_vec"
    if "quantize_mmq" in text:
        return "mmq_input_quantize"
    if "quantize_" in text:
        return "input_quantize"
    if "rms_norm" in text or "norm" in text:
        return "normalization"
    return str(name or "unknown").split("<", 1)[0].strip() or "unknown"


def _layout(name: str | None) -> str:
    text = str(name or "")
    match = re.search(r"mmq_q8_1_([a-z0-9_]+)", text, re.I)
    if match:
        return match.group(1)
    if "mul_mat_vec_q" in text:
        return "vector_q"
    if "mul_mat_q" in text:
        return "matrix_q"
    return "unspecified"


def _stage(event: dict) -> str:
    text = str(event.get("semantic_operator") or event.get("semantic_operator_id") or "").lower()
    if any(x in text for x in ("qcur", "kcur", "vcur", "qkv")):
        return "attention_qkv"
    if any(x in text for x in ("ffn", "glu", "gate", "up", "down")):
        return "ffn"
    if any(x in text for x in ("cache", "set_rows", "kqv")):
        return "kv"
    if any(x in text for x in ("result_output", "get_rows", "lm_head")):
        return "lm_head"
    if any(x in text for x in ("ssm", "gated_delta", "conv_", "beta", "alpha")):
        return "linear_attention_aux"
    return str(event.get("stage") or "unknown")


def _events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for event in payload.get("events", []):
        if event.get("kind") != "kernel" or event.get("semantic_status") != "matched":
            continue
        shape = event.get("semantic_shape")
        kernel = event.get("kernel_name") or event.get("short_name")
        rows.append({
            "stage": _stage(event),
            "phase": _phase(shape),
            "shape": str(shape or "unknown"),
            "dtype": str(event.get("semantic_type") or "unknown"),
            "layout": _layout(kernel),
            "kernel_family": _kernel_family(kernel),
            "operator_id": str(event.get("semantic_operator_id") or event.get("semantic_operator") or "unknown").split("|", 1)[0],
            "duration_ns": float(event.get("duration_ns") or 0.0),
        })
    return rows


def build(pairs: list[tuple[Path, Path]]) -> dict:
    pair_results = []
    union = Counter(); intersections = Counter(); stage_totals = defaultdict(float)
    for train, holdout in pairs:
        tr, ho = _events(train), _events(holdout)
        tr_keys = Counter(tuple(r[k] for k in ("stage", "phase", "shape", "dtype", "layout", "kernel_family")) for r in tr)
        ho_keys = Counter(tuple(r[k] for k in ("stage", "phase", "shape", "dtype", "layout", "kernel_family")) for r in ho)
        inter = sorted(set(tr_keys) & set(ho_keys))
        missing_holdout = sorted(set(ho_keys) - set(tr_keys))
        for r in tr:
            union[tuple(r[k] for k in ("stage", "phase", "shape", "dtype", "layout", "kernel_family"))] += 1
            stage_totals[r["stage"]] += r["duration_ns"]
        for key in inter: intersections[key] += 1
        pair_results.append({
            "train": str(train), "holdout": str(holdout),
            "train_events": len(tr), "holdout_events": len(ho),
            "train_joint_keys": len(tr_keys), "holdout_joint_keys": len(ho_keys),
            "joint_intersection_keys": len(inter),
            "holdout_keys_missing_from_train": len(missing_holdout),
            "missing_examples": [list(k) for k in missing_holdout[:12]],
            "train_kernel_families": dict(Counter(r["kernel_family"] for r in tr)),
            "holdout_kernel_families": dict(Counter(r["kernel_family"] for r in ho)),
        })
    return {
        "schema": "operator-cost-coverage-audit/v1",
        "purpose": "descriptive evidence only; no target-LLM latency fitting",
        "pairs": pair_results,
        "aggregate": {
            "pair_count": len(pairs),
            "union_joint_keys": len(union),
            "keys_seen_in_train_and_holdout_pair": len(intersections),
            "stage_train_kernel_time_ns": dict(stage_totals),
            "joint_key_fields": ["stage", "phase", "shape", "dtype", "layout", "kernel_family"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", nargs=2, metavar=("TRAIN", "HOLDOUT"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = build([(Path(a), Path(b)) for a, b in args.pair])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "pairs": len(args.pair)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
