"""Calibrate explicit NVTX semantic operator rates against a holdout trace."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STAGES = ("attention_qkv", "ffn", "kv", "lm_head", "attention_output", "normalization", "quantize", "linear_attention_aux")
PHASES = ("prefill", "decode")


def _operator_id(label: object) -> str | None:
    """Extract the stable operator id prefix from an NVTX label.

    The semantic extractor emits labels in the form ``operator:<op>:<name>|``
    (and some older traces use ``[<name>]|``).  Wall-time calibration must use
    the same id as the event path; falling back to the complete label would
    create shape-specific buckets and, before this helper existed, raised a
    ``NameError`` for traces containing NVTX operator scopes.
    """
    text = str(label or "").strip()
    if not text:
        return None
    if text.startswith("operator:"):
        text = text[len("operator:"):]
        # Keep the operation and tensor name but discard shape/dtype fields.
        return text.split("|", 1)[0].strip() or None
    if text.startswith("[") and "]" in text:
        text = text[1:text.find("]")]
    return text.split("|", 1)[0].strip() or None
_RULES = (
    # Names are the semantic labels emitted by llama.cpp's graph.  Include
    # both projection kernels (qcur/ffn_inp) and the attention sub-operators
    # (kq/kqv); otherwise most matched MMQ work falls into ``unknown`` even
    # though an explicit NVTX operator range proves its ownership.
    ("attention_qkv", re.compile(r"(?:^|[^a-z0-9])(?:qcur|kcur|vcur|qkv[^a-z0-9]*out|rope)(?:$|[^a-z0-9])", re.I)),
    ("ffn", re.compile(r"(?:^|[^a-z0-9])(?:ffn[^a-z0-9]*(?:inp|gate|up|out)|glu)(?:$|[^a-z0-9])", re.I)),
    ("attention_output", re.compile(r"(?:^|[^a-z0-9])(?:kqv|kqv_out|kqv_wo|attn[^a-z0-9]*(?:out|wo|residual|pregate)|gate[_-]?reshaped|gate[_-]?sigmoid|soft[^a-z0-9]*max)(?:$|[^a-z0-9])", re.I)),
    ("kv", re.compile(r"(?:^|[^a-z0-9])(?:cache[^a-z0-9].*|set[^a-z0-9]*rows|kq(?:[^a-z0-9].*)?)(?:$|[^a-z0-9])", re.I)),
    ("lm_head", re.compile(r"(?:^|[^a-z0-9])(?:result[^a-z0-9]*output|get[^a-z0-9]*rows)(?:$|[^a-z0-9])", re.I)),
    ("normalization", re.compile(r"(?:^|[^a-z0-9])(?:rms[_-]?norm|norm)(?:$|[^a-z0-9])", re.I)),
    ("quantize", re.compile(r"(?:^|[^a-z0-9])(?:quantize|dequantize)(?:$|[^a-z0-9])", re.I)),
    # Qwen3.5 gated-delta-net/SSM operators are explicit owners but are not
    # interchangeable with the QKV, FFN, or full-attention output stages.
    ("linear_attention_aux", re.compile(
        r"(?:^|[^a-z0-9])(?:ssm[_-]?conv|gated[_-]?delta[_-]?net|"
        r"q[_-]?conv[_-]?predelta|k[_-]?conv[_-]?predelta|conv[_-]?input|"
        r"conv[_-]?state[_-]?update|cache_[rs]_l\d+|a[_-]?softplus|"
        r"beta[_-]?sigmoid|l[_-]?out)(?:$|[^a-z0-9])", re.I)),
)


def classify_operator(name: object) -> str | None:
    """Return a stage only for an explicit, covered operator label."""
    text = str(name or "").strip()
    for stage, pattern in _RULES:
        if pattern.search(text):
            return stage
    return None


def _events(payload: dict) -> list[dict]:
    """Return CUDA kernels eligible for operator-rate calibration.

    Memcpy rows may be nested in an operator NVTX range, but their duration is
    transfer work and must not be folded into a compute operator's rate.
    ``_measure`` reports those rows separately for coverage auditing.
    """
    return [event for event in payload.get("events", [])
            if event.get("kind") in {"kernel", "cpu_operator"}]


def _memcpy_events(payload: dict) -> list[dict]:
    return [event for event in payload.get("events", []) if event.get("kind") == "memcpy"]


def _event_operator_id(event: dict) -> str | None:
    """Prefer the extractor's stable id; never bucket on shape-bearing labels."""
    value = event.get("semantic_operator_id")
    if value is None:
        value = event.get("operator_id")
    if value is None:
        value = event.get("semantic_operator")
    text = str(value or "").strip()
    return text.split("|", 1)[0].strip() or None


def _trace_identity(payload: dict) -> dict:
    """Extract comparable model/runtime/hardware identity when a trace carries it."""
    source = payload.get("trace_identity")
    if not isinstance(source, dict):
        source = payload.get("metadata")
    if not isinstance(source, dict):
        source = payload
    keys = (
        "model_sha256", "model_key", "model_path", "model_bytes", "architecture",
        "runtime_fingerprint", "runtime_id", "hardware_fingerprint", "gpu_name",
        "backend_build_fingerprint", "backend_commit", "gpu_layers", "context_length",
        "batch_size", "ubatch_size", "threads", "flash_attention",
    )
    return {key: source[key] for key in keys if key in source and source[key] is not None}


def _phase_name(value: object) -> str | None:
    """Normalize the two runtime phases without guessing other phase names."""
    text = str(value or "").strip().lower()
    if text.startswith("phase:"):
        text = text[6:].strip()
    if text == "prefill" or text.startswith("prefill/") or text.startswith("prefill_"):
        return "prefill"
    if text == "decode" or text.startswith("decode/") or text.startswith("decode_") or re.fullmatch(r"decode\d+", text):
        return "decode"
    return None


def _phase_scopes(payload: dict) -> list[dict]:
    scopes = []
    for scope in payload.get("nvtx_events", []):
        # Lifecycle markers such as ``prefill_begin`` are not phase ranges.
        # Only an explicit phase field or a ``phase:...`` label can establish
        # containment for operator timing evidence.
        explicit = scope.get("phase_name") or scope.get("phase")
        label = str(scope.get("label") or "").strip()
        if explicit is None and not re.match(r"^phase\s*[:=/]", label, re.IGNORECASE):
            continue
        phase = _phase_name(explicit or label)
        if phase is not None:
            scopes.append({**scope, "phase": phase})
    return scopes


def _event_phase(event: dict, scopes: list[dict]) -> str | None:
    for key in ("phase", "phase_name", "runtime_phase"):
        phase = _phase_name(event.get(key))
        if phase is not None:
            return phase
    nested = event.get("nvtx")
    if isinstance(nested, dict):
        phase = _phase_name(nested.get("phase") or nested.get("phase_name"))
        if phase is not None:
            return phase
    # Extracted traces keep phase ranges in nvtx_events.  Use runtime times
    # where available; device activity timestamps alone are not phase evidence.
    start = event.get("runtime_start_ns")
    end = event.get("runtime_end_ns")
    if start is None or end is None:
        return None
    tid = event.get("runtime_global_tid")
    matches = [
        scope for scope in scopes
        if (scope.get("global_tid") is None or tid is None or scope.get("global_tid") == tid)
        and scope.get("start_ns", 0) <= start and end <= scope.get("end_ns", 0)
    ]
    if len(matches) == 1:
        return matches[0]["phase"]
    return None


def _shape_key(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    text = str(value).strip()
    return text or None


def _label_field(label: object, field: str) -> str | None:
    match = re.search(r"(?:^|[|;])" + re.escape(field) + r"=([^|;]+)", str(label or ""), re.IGNORECASE)
    return match.group(1).strip() if match else None


def _event_shape(event: dict) -> str | None:
    for key in ("token_shape", "token_shape_key", "semantic_shape", "shape", "shape_key"):
        value = _shape_key(event.get(key))
        if value is not None:
            return value
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        for key in ("token_shape", "token_shape_key", "shape", "shape_key"):
            value = _shape_key(metadata.get(key))
            if value is not None:
                return value
    return None


def _scope_shape(scope: dict) -> str | None:
    value = scope.get("semantic_shape") or scope.get("shape")
    if value is None:
        value = _label_field(scope.get("label"), "shape")
    return _shape_key(value)


def _wall_measure(payload: dict) -> dict:
    """Measure one wall-time occurrence per explicit operator scope.

    llama.cpp emits a parent operator NVTX range plus nested ranges with the
    same id.  Sort longest-first and discard only scopes contained by an
    already-selected same-id scope; sequential invocations remain distinct.
    """
    phase_scopes = _phase_scopes(payload)
    candidates = []
    for scope in payload.get("nvtx_events", []):
        if not scope.get("operator_name"):
            continue
        operator_id = scope.get("operator_id") or _operator_id(scope.get("label"))
        if not operator_id:
            continue
        stage = classify_operator(operator_id)
        if stage is None:
            continue
        begin, end = scope.get("start_ns"), scope.get("end_ns")
        if begin is None or end is None or end < begin:
            continue
        phase = _phase_name(scope.get("phase_name") or scope.get("phase"))
        if phase is None:
            matches = [item for item in phase_scopes
                       if (item.get("global_tid") is None or scope.get("global_tid") is None
                           or item.get("global_tid") == scope.get("global_tid"))
                       and item.get("start_ns", 0) <= begin and end <= item.get("end_ns", 0)]
            if len(matches) == 1:
                phase = matches[0]["phase"]
        candidates.append({
            "stage": stage, "operator_id": str(operator_id), "phase": phase,
            "shape": _scope_shape(scope), "global_tid": scope.get("global_tid"),
            "start_ns": begin, "end_ns": end, "duration_ns": end - begin,
        })
    # CPU operator traces already contain one wall interval per graph node;
    # use those intervals directly when no NVTX operator scopes are present.
    if not candidates:
        for event in _events(payload):
            if event.get("kind") != "cpu_operator":
                continue
            operator_id = event.get("semantic_operator_id") or event.get("operator_id")
            stage = classify_operator(operator_id or event.get("semantic_operator"))
            begin, end = event.get("start_ns"), event.get("end_ns")
            if not operator_id or stage is None or begin is None or end is None or end < begin:
                continue
            candidates.append({
                "stage": stage, "operator_id": str(operator_id),
                "phase": _phase_name(event.get("phase")),
                "shape": _shape_key(event.get("token_shape") or event.get("shape")),
                "global_tid": None, "start_ns": begin, "end_ns": end,
                "duration_ns": end - begin,
            })
    selected = []
    for candidate in sorted(candidates, key=lambda item: (item["operator_id"], str(item.get("global_tid")), item["start_ns"], -item["duration_ns"])):
        nested = any(
            prior["operator_id"] == candidate["operator_id"]
            and prior.get("global_tid") == candidate.get("global_tid")
            and prior["start_ns"] <= candidate["start_ns"]
            and candidate["end_ns"] <= prior["end_ns"]
            for prior in selected
        )
        if not nested:
            selected.append(candidate)
    totals = {stage: 0.0 for stage in STAGES}
    counts = {stage: 0 for stage in STAGES}
    grouped: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for item in selected:
        stage = item["stage"]
        totals[stage] += item["duration_ns"]
        counts[stage] += 1
        if item["phase"] is not None:
            shape = item["shape"] or "unknown"
            bucket = grouped.setdefault(stage, {}).setdefault(item["phase"], {}).setdefault(shape, {"total_ns": 0.0, "count": 0})
            bucket["total_ns"] += item["duration_ns"]
            bucket["count"] += 1
    return {"totals": totals, "counts": counts, "groups": grouped, "selected_count": len(selected)}


def _wall_stats(train_total: float, train_n: int, hold_total: float, hold_n: int) -> dict:
    result = _stats(train_total, train_n, hold_total, hold_n)
    return {
        "wall_train_total_ns": result["train_total_ns"],
        "wall_train_instances": result["train_instances"],
        "wall_ns_per_instance": result["train_ns_per_instance"],
        "wall_holdout_total_ns": result["holdout_total_ns"],
        "wall_holdout_instances": result["holdout_instances"],
        "wall_holdout_ns_per_instance": result["holdout_ns_per_instance"],
        "wall_holdout_predicted_ns": result["holdout_predicted_ns"],
        "wall_holdout_relative_error_pct": result["holdout_relative_error_pct"],
        "wall_status": result["status"],
    }


def _stats(train_total: float, train_n: int, hold_total: float, hold_n: int, *, blocked: bool = False) -> dict:
    rate = train_total / train_n if train_n else None
    hold_rate = hold_total / hold_n if hold_n else None
    predicted = rate * hold_n if rate is not None else None
    error = (predicted - hold_total) / hold_total * 100.0 if predicted is not None and hold_total else None
    return {
        "status": "calibrated" if train_n and hold_n and not blocked else "blocked_no_semantic_evidence",
        "train_total_ns": train_total,
        "train_instances": train_n,
        "train_ns_per_instance": rate,
        "holdout_total_ns": hold_total,
        "holdout_instances": hold_n,
        "holdout_ns_per_instance": hold_rate,
        "holdout_predicted_ns": predicted,
        "holdout_relative_error_pct": error,
    }


def _byte_stats(train_bytes: int, train_ns: float, hold_bytes: int, hold_ns: float) -> dict:
    """Report physical byte evidence and effective aggregate bandwidth."""
    train_bw = (float(train_bytes) / (train_ns * 1e-9) / 1e9) if train_ns > 0 else None
    hold_bw = (float(hold_bytes) / (hold_ns * 1e-9) / 1e9) if hold_ns > 0 else None
    return {
        "train_total_bytes": int(train_bytes),
        "holdout_total_bytes": int(hold_bytes),
        "train_effective_bandwidth_gbps": train_bw,
        "holdout_effective_bandwidth_gbps": hold_bw,
    }


def _measure(payload: dict) -> tuple[dict, dict]:
    totals = {stage: 0.0 for stage in STAGES}
    counts = {stage: 0 for stage in STAGES}
    byte_totals = {stage: 0 for stage in STAGES}
    grouped: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    operator_groups: dict[str, dict[str, dict[str, float | int]]] = {stage: {} for stage in STAGES}
    unknown_time = unknown_count = 0
    excluded_time = excluded_count = 0
    missing_phase_count = missing_shape_count = 0
    scopes = _phase_scopes(payload)
    memcpy_time = sum(float(event.get("duration_ns") or 0.0) for event in _memcpy_events(payload))
    memcpy_count = len(_memcpy_events(payload))
    for event in _events(payload):
        duration = float(event.get("duration_ns") or 0.0)
        if event.get("semantic_status") != "matched" or event.get("graph_node_id") is not None:
            unknown_time += duration
            unknown_count += 1
            continue
        stage = classify_operator(_event_operator_id(event) or event.get("semantic_operator"))
        if stage is None:
            # An explicit semantic owner outside the four calibrated stages
            # (quantize, norm, rope, attention output, etc.) is valid evidence
            # but intentionally excluded from target-stage rates.  Only an
            # unmatched/graph event is unknown and blocks coverage.
            excluded_time += duration
            excluded_count += 1
            continue
        totals[stage] += duration
        counts[stage] += 1
        byte_totals[stage] += int(event.get("total_bytes") or 0)
        operator_id = _event_operator_id(event) or "unknown"
        op_bucket = operator_groups[stage].setdefault(operator_id, {"total_ns": 0.0, "count": 0})
        op_bucket["total_ns"] += duration
        op_bucket["count"] += 1
        phase = _event_phase(event, scopes)
        shape = _event_shape(event)
        if phase is None:
            missing_phase_count += 1
        if shape is None:
            missing_shape_count += 1
        if phase is not None:
            shape = shape or "unknown"
            bucket = grouped.setdefault(stage, {}).setdefault(phase, {}).setdefault(
                shape, {"total_ns": 0.0, "count": 0, "bytes": 0}
            )
            bucket["total_ns"] += duration
            bucket["count"] += 1
            bucket["bytes"] += int(event.get("total_bytes") or 0)
    return {"totals": totals, "counts": counts, "bytes": byte_totals, "groups": grouped, "operator_groups": operator_groups}, {
        "kernel_count": len(_events(payload)),
        "count": unknown_count,
        "time_ns": unknown_time,
        "excluded_count": excluded_count,
        "excluded_time_ns": excluded_time,
        "memcpy_count": memcpy_count,
        "memcpy_time_ns": memcpy_time,
        "missing_phase_count": missing_phase_count,
        "missing_shape_count": missing_shape_count,
    }


def build(train_path: Path, holdout_path: Path, *, schema: str = "native-semantic-calibration/v1") -> dict:
    train = json.loads(train_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    if train.get("schema") not in (None, "native-nsys-trace/v1"):
        raise ValueError(f"unsupported train trace schema: {train.get('schema')}")
    if holdout.get("schema") not in (None, "native-nsys-trace/v1"):
        raise ValueError(f"unsupported holdout trace schema: {holdout.get('schema')}")
    train_m, train_unknown = _measure(train)
    hold_m, holdout_unknown = _measure(holdout)
    train_wall, holdout_wall = _wall_measure(train), _wall_measure(holdout)
    train_identity = _trace_identity(train)
    holdout_identity = _trace_identity(holdout)
    identity_fields = sorted(set(train_identity) | set(holdout_identity))
    identity_mismatch = [field for field in identity_fields if train_identity.get(field) != holdout_identity.get(field)]
    stages = {}
    for stage in STAGES:
        train_total, train_n = train_m["totals"][stage], train_m["counts"][stage]
        hold_total, hold_n = hold_m["totals"][stage], hold_m["counts"][stage]
        stage_result = _stats(train_total, train_n, hold_total, hold_n)
        stage_result.update(_byte_stats(
            train_m["bytes"][stage], train_total,
            hold_m["bytes"][stage], hold_total,
        ))
        stage_result.update(_wall_stats(
            train_wall["totals"][stage], train_wall["counts"][stage],
            holdout_wall["totals"][stage], holdout_wall["counts"][stage],
        ))
        # Keep stable invocation-level evidence alongside phase/shape buckets.
        # This is diagnostic data; planner calibration still requires an exact
        # stage/phase/shape match and never consumes aggregate rates implicitly.
        operator_results = {}
        train_ops = train_m["operator_groups"].get(stage, {})
        hold_ops = hold_m["operator_groups"].get(stage, {})
        for operator_id in sorted(set(train_ops) | set(hold_ops)):
            train_bucket, hold_bucket = train_ops.get(operator_id, {}), hold_ops.get(operator_id, {})
            operator_results[operator_id] = _stats(
                float(train_bucket.get("total_ns", 0.0)), int(train_bucket.get("count", 0)),
                float(hold_bucket.get("total_ns", 0.0)), int(hold_bucket.get("count", 0)),
                blocked=operator_id == "unknown",
            )
        stage_result["operators"] = operator_results
        phases = {}
        for phase in PHASES:
            shape_results = {}
            train_phase = train_m["groups"].get(stage, {}).get(phase, {})
            hold_phase = hold_m["groups"].get(stage, {}).get(phase, {})
            for shape in sorted(set(train_phase) | set(hold_phase)):
                train_bucket, hold_bucket = train_phase.get(shape, {}), hold_phase.get(shape, {})
                shape_results[shape] = _stats(
                    float(train_bucket.get("total_ns", 0.0)), int(train_bucket.get("count", 0)),
                    float(hold_bucket.get("total_ns", 0.0)), int(hold_bucket.get("count", 0)),
                    blocked=shape == "unknown",
                )
                shape_results[shape].update(_byte_stats(
                    int(train_bucket.get("bytes", 0)), float(train_bucket.get("total_ns", 0.0)),
                    int(hold_bucket.get("bytes", 0)), float(hold_bucket.get("total_ns", 0.0)),
                ))
            phase_train_total = sum(item["total_ns"] for item in train_phase.values())
            phase_train_n = sum(item["count"] for item in train_phase.values())
            phase_hold_total = sum(item["total_ns"] for item in hold_phase.values())
            phase_hold_n = sum(item["count"] for item in hold_phase.values())
            phase_result = _stats(
                phase_train_total, phase_train_n, phase_hold_total, phase_hold_n,
                blocked=("unknown" in shape_results),
            )
            phase_result.update(_byte_stats(
                sum(int(item.get("bytes", 0)) for item in train_phase.values()), phase_train_total,
                sum(int(item.get("bytes", 0)) for item in hold_phase.values()), phase_hold_total,
            ))
            train_wall_phase = train_wall["groups"].get(stage, {}).get(phase, {})
            hold_wall_phase = holdout_wall["groups"].get(stage, {}).get(phase, {})
            phase_result.update(_wall_stats(
                sum(item["total_ns"] for item in train_wall_phase.values()),
                sum(item["count"] for item in train_wall_phase.values()),
                sum(item["total_ns"] for item in hold_wall_phase.values()),
                sum(item["count"] for item in hold_wall_phase.values()),
            ))
            phase_result["token_shapes"] = shape_results
            phase_result["wall_token_shapes"] = {}
            for shape in sorted(set(train_wall_phase) | set(hold_wall_phase)):
                train_bucket, hold_bucket = train_wall_phase.get(shape, {}), hold_wall_phase.get(shape, {})
                phase_result["wall_token_shapes"][shape] = _wall_stats(
                    float(train_bucket.get("total_ns", 0.0)), int(train_bucket.get("count", 0)),
                    float(hold_bucket.get("total_ns", 0.0)), int(hold_bucket.get("count", 0)),
                )
            phases[phase] = phase_result
        stage_result["phases"] = phases
        stages[stage] = stage_result
    return {
        "schema": schema,
        "calibration_basis": "operator_wall",
        "train_trace": str(train_path.resolve()), "holdout_trace": str(holdout_path.resolve()),
        "train_identity": train_identity,
        "holdout_identity": holdout_identity,
        "stages": stages,
        "coverage": {
            "train_unknown_or_uncovered_count": train_unknown["count"],
            "train_unknown_or_uncovered_time_ns": train_unknown["time_ns"],
            "train_explicit_excluded_count": train_unknown["excluded_count"],
            "train_explicit_excluded_time_ns": train_unknown["excluded_time_ns"],
            "train_kernel_count": train_unknown["kernel_count"],
            "train_memcpy_excluded_count": train_unknown["memcpy_count"],
            "train_memcpy_excluded_time_ns": train_unknown["memcpy_time_ns"],
            "holdout_unknown_or_uncovered_count": holdout_unknown["count"],
            "holdout_unknown_or_uncovered_time_ns": holdout_unknown["time_ns"],
            "holdout_explicit_excluded_count": holdout_unknown["excluded_count"],
            "holdout_explicit_excluded_time_ns": holdout_unknown["excluded_time_ns"],
            "holdout_kernel_count": holdout_unknown["kernel_count"],
            "holdout_memcpy_excluded_count": holdout_unknown["memcpy_count"],
            "holdout_memcpy_excluded_time_ns": holdout_unknown["memcpy_time_ns"],
            "train_missing_phase_count": train_unknown["missing_phase_count"],
            "holdout_missing_phase_count": holdout_unknown["missing_phase_count"],
            "train_missing_token_shape_count": train_unknown["missing_shape_count"],
            "holdout_missing_token_shape_count": holdout_unknown["missing_shape_count"],
            "identity_match": not identity_mismatch,
            "identity_mismatch_fields": identity_mismatch,
            "status": "blocked" if (
                not train_unknown["kernel_count"] or not holdout_unknown["kernel_count"]
                or
                train_unknown["count"] or holdout_unknown["count"]
                or train_unknown["missing_phase_count"] or holdout_unknown["missing_phase_count"]
                or train_unknown["missing_shape_count"] or holdout_unknown["missing_shape_count"]
                or identity_mismatch
            ) else "covered",
            "policy": "Use semantic_status=matched explicit NVTX labels only; overlapping time is never inferred or redistributed.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("train", type=Path); ap.add_argument("holdout", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--schema", choices=("native-semantic-calibration/v1", "native-semantic-calibration/v2"), default="native-semantic-calibration/v1")
    args = ap.parse_args(); result = build(args.train, args.holdout, schema=args.schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
