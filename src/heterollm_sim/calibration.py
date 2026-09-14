"""Evidence-backed, stage-scoped calibration for native llama.cpp runs."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .config import ScenarioConfig
from .contracts import ResourceDemand
from .cost_models import CostPhase


@dataclass(frozen=True)
class NativeCalibrationProfile:
    """Measured coefficients; each coefficient has an explicit stage owner."""
    prompt_ns_per_token: float | None = None
    decode_ns_per_token: float | None = None
    launch_ns_per_call: float | None = None
    prefill_launch_ns_per_call: float | None = None
    decode_launch_ns_per_call: float | None = None
    synchronize_ns_per_call: float | None = None
    # One-time residual observed on the first decode invocation for a
    # small-batch CUDA graph.  This is kept separate from the recurring
    # launch/sync boundary: the semantic traces show a large first-kq gap,
    # while later decode invocations do not.  It is never applied without an
    # explicit first-invocation signal from the planner.
    decode_first_invocation_extra_ns: float | None = None
    decode_first_invocation_policy: str | None = None
    decode_first_invocation_evidence: Mapping[str, Any] | None = None
    # Optional aggregate launch+sync wall time measured for one explicit
    # llama.cpp phase invocation.  This is deliberately a mapping rather than
    # a global scalar: the planner emits one boundary task per physical
    # prefill/decode invocation and never distributes this value over
    # individual operators.
    phase_boundary_ns_per_invocation: Mapping[str, float] | None = None
    phase_boundary_policy: str | None = None
    # Request-lifecycle markers are additive costs at an explicitly named
    # boundary (request_begin, first_token, or request_end).  They are kept
    # separate from phase launch/sync evidence because a marker interval may
    # include host request handling that is not represented by a CUDA API
    # phase.  Applying them requires a paired marker profile and an exact
    # request shape; there is intentionally no shape-less fallback.
    request_marker_ns: Mapping[str, float] | None = None
    request_marker_policy: str | None = None
    request_marker_evidence: Mapping[str, Any] | None = None
    request_shape: Mapping[str, Any] | None = None
    d2h_ns_per_byte: float | None = None
    source: str = ""
    source_sha256: str | None = None
    kernel_stage_mapping: Mapping[str, Any] | None = None
    model_sha256: str | None = None
    hardware_fingerprint: str | None = None
    runtime_fingerprint: str | None = None
    backend_version: str | None = None
    # Semantic calibration artifacts may be explicitly blocked by incomplete
    # NVTX coverage or mismatched train/holdout identities.  Keep this gate
    # with the profile so callers cannot accidentally apply unsafe stages.
    coverage_status: str | None = None
    identity_mismatch: bool = False
    calibration_basis: str | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_ns_per_token", "decode_ns_per_token", "launch_ns_per_call", "prefill_launch_ns_per_call", "decode_launch_ns_per_call", "synchronize_ns_per_call", "decode_first_invocation_extra_ns", "d2h_ns_per_byte"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"calibration {name} must be a finite non-negative number or None")
        if self.phase_boundary_ns_per_invocation is not None:
            if not isinstance(self.phase_boundary_ns_per_invocation, Mapping):
                raise ValueError("calibration phase_boundary_ns_per_invocation must be a mapping or None")
            for phase, value in self.phase_boundary_ns_per_invocation.items():
                if str(phase) not in {"prefill", "decode"} or _finite_non_negative(value) is None:
                    raise ValueError("calibration phase boundary values must be finite and phase-scoped")
        if self.decode_first_invocation_policy is not None and str(
            self.decode_first_invocation_policy
        ) != "first_decode_only":
            raise ValueError(
                "decode first invocation policy must be first_decode_only or None"
            )
        if self.decode_first_invocation_evidence is not None and not isinstance(
            self.decode_first_invocation_evidence, Mapping
        ):
            raise ValueError("decode first invocation evidence must be a mapping or None")
        if self.request_marker_ns is not None:
            if not isinstance(self.request_marker_ns, Mapping):
                raise ValueError("calibration request_marker_ns must be a mapping or None")
            for marker, value in self.request_marker_ns.items():
                if str(marker) not in {"request_begin", "first_token", "request_end"}:
                    raise ValueError("calibration request marker names are unsupported")
                if _finite_non_negative(value) is None:
                    raise ValueError("calibration request marker values must be finite and non-negative")
        if self.request_marker_evidence is not None and not isinstance(self.request_marker_evidence, Mapping):
            raise ValueError("calibration request_marker_evidence must be a mapping or None")
        if self.request_shape is not None and not isinstance(self.request_shape, Mapping):
            raise ValueError("calibration request_shape must be a mapping or None")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "native-calibration/v1", **self.__dict__}


def load_native_calibration(path: str | Path) -> NativeCalibrationProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("schema") in {"native-calibration/v1", "native-semantic-calibration/v1", "native-semantic-calibration/v2"}:
        # Semantic holdout artifacts store stage coefficients under ``stages``
        # and deliberately omit global timing fields.  Keep that evidence
        # usable while preserving the strict, stage-scoped application path.
        if data.get("schema") in {"native-semantic-calibration/v1", "native-semantic-calibration/v2"}:
            coverage = data.get("coverage") if isinstance(data.get("coverage"), Mapping) else {}
            train_identity = data.get("train_identity") if isinstance(data.get("train_identity"), Mapping) else {}
            hold_identity = data.get("holdout_identity") if isinstance(data.get("holdout_identity"), Mapping) else {}
            identity_mismatch = bool(coverage.get("identity_mismatch_fields"))
            if train_identity or hold_identity:
                identity_mismatch = identity_mismatch or train_identity != hold_identity
            return NativeCalibrationProfile(
                launch_ns_per_call=_finite_non_negative(data.get("launch_ns_per_call")),
                prefill_launch_ns_per_call=_finite_non_negative(data.get("prefill_launch_ns_per_call")),
                decode_launch_ns_per_call=_finite_non_negative(data.get("decode_launch_ns_per_call")),
                synchronize_ns_per_call=_finite_non_negative(data.get("synchronize_ns_per_call")),
                phase_boundary_ns_per_invocation=data.get("phase_boundary_ns_per_invocation"),
                phase_boundary_policy=data.get("phase_boundary_policy"),
                request_marker_ns=data.get("request_marker_ns"),
                request_marker_policy=data.get("request_marker_policy"),
                request_marker_evidence=data.get("request_marker_evidence"),
                request_shape=data.get("request_shape"),
                prompt_ns_per_token=_finite_non_negative(data.get("prompt_ns_per_token")),
                decode_ns_per_token=_finite_non_negative(data.get("decode_ns_per_token")),
                source=str(Path(path).resolve()),
                kernel_stage_mapping=data.get("stages") or {},
                model_sha256=train_identity.get("model_sha256"),
                hardware_fingerprint=train_identity.get("hardware_fingerprint"),
                runtime_fingerprint=train_identity.get("runtime_fingerprint"),
                coverage_status=coverage.get("status"),
                identity_mismatch=identity_mismatch,
                calibration_basis=data.get("calibration_basis"),
            )
        data = dict(data)
        data.pop("schema", None)
        values = {k: data[k] for k in NativeCalibrationProfile.__dataclass_fields__ if k in data}
        values.setdefault("source", "")
        return NativeCalibrationProfile(**values)
    raise ValueError("unsupported calibration profile schema")


def profile_from_mapping(value: Mapping[str, Any] | NativeCalibrationProfile | None) -> NativeCalibrationProfile | None:
    """Coerce placement metadata into a validated profile, fail-closed."""
    if isinstance(value, NativeCalibrationProfile):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        values = {k: value[k] for k in NativeCalibrationProfile.__dataclass_fields__ if k in value}
        if "kernel_stage_mapping" not in values and isinstance(value.get("stages"), Mapping):
            values["kernel_stage_mapping"] = value["stages"]
        return NativeCalibrationProfile(**values)
    except (TypeError, ValueError):
        return None


def _finite_non_negative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _model_gguf_sha256(model: object) -> str | None:
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("gguf_sha256")
    nested = metadata.get("metadata")
    if value is None and isinstance(nested, Mapping):
        value = nested.get("gguf_sha256")
    return str(value) if value is not None else None


def _profile_identity_matches(
    profile: NativeCalibrationProfile,
    *,
    model_sha256: str | None = None,
    hardware_fingerprint: str | None = None,
    runtime_fingerprint: str | None = None,
) -> bool:
    """Fail closed when a profile names an identity we cannot prove."""
    for expected, actual in (
        (profile.model_sha256, model_sha256),
        (profile.hardware_fingerprint, hardware_fingerprint),
        (profile.runtime_fingerprint, runtime_fingerprint),
    ):
        if expected is not None and (actual is None or str(expected) != str(actual)):
            return False
    return True


def stage_calibration_ns_per_instance(
    profile: NativeCalibrationProfile,
    *,
    stage: str,
    phase: str,
    token_shape: str | None = None,
) -> float | None:
    """Return an evidence-backed coefficient for one exact invocation.

    No interpolation or fallback across stages is performed.  A token-shape
    coefficient is preferred when supplied; otherwise the phase aggregate is
    accepted only when it is explicitly marked calibrated.
    """
    resolution = resolve_stage_calibration(
        profile, stage=stage, phase=phase, token_shape=token_shape,
        allow_interpolation=False,
    )
    return (float(resolution["coefficient"])
            if resolution.get("coefficient") is not None else None)


def _shape_dimensions(value: object) -> tuple[float, ...] | None:
    """Parse the canonical numeric dimensions used by trace shape buckets.

    Shapes in old profiles are strings such as ``896x9x1x1`` while a few
    semantic traces use labels (``b1_t8``).  The latter are intentionally not
    interpolated because their axes do not have a stable contract.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    if "x" not in text:
        return None
    parts = text.split("x")
    if not parts or any(not p or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", p) for p in parts):
        return None
    try:
        dims = tuple(float(p) for p in parts)
    except ValueError:
        return None
    return dims if all(math.isfinite(v) and v > 0 for v in dims) else None


def _shape_bucket_mapping(phase_entry: Mapping[str, Any], wall_basis: bool) -> Mapping[str, Any]:
    buckets = phase_entry.get("wall_token_shapes" if wall_basis else "token_shapes")
    return buckets if isinstance(buckets, Mapping) else {}


EXACT_OPERATOR_KEY_FIELDS = (
    "stage", "phase", "shape", "dtype", "layout", "kernel_family",
)


def canonical_exact_operator_key(
    *,
    stage: object,
    phase: object,
    token_shape: object,
    dtype: object,
    layout: object,
    kernel_family: object,
) -> str | None:
    """Build the strict six-dimensional operator evidence key.

    Every dimension is required.  Values are trimmed and compared as strings;
    no dimension is inferred, wildcarded, or reduced to an owner/stage average.
    ``None`` is returned for malformed keys so callers can fail closed before
    touching a calibration profile.
    """
    values = (stage, phase, token_shape, dtype, layout, kernel_family)
    if any(value is None or not str(value).strip() for value in values):
        return None
    return "|".join(str(value).strip() for value in values)


def resolve_stage_calibration(
    profile: NativeCalibrationProfile,
    *,
    stage: str,
    phase: str,
    token_shape: str | None = None,
    allow_interpolation: bool = False,
) -> dict[str, Any]:
    """Resolve one stage coefficient and return auditable shape provenance.

    Interpolation is opt-in and only occurs along the canonical token-row axis when
    the requested point lies between two calibrated buckets with identical
    remaining dimensions.  Requests outside the measured interval are marked
    ``extrapolation`` and return no coefficient.  A missing/unsupported shape
    returns an explicit ``analytical_fallback`` mode so callers cannot mistake
    a phase aggregate for a shape hit.
    """
    result: dict[str, Any] = {
        "mode": "unavailable", "requested_shape": token_shape,
        "source_shapes": [], "coefficient": None, "uncertainty": None,
        "extrapolation": False,
    }
    mapping = profile.kernel_stage_mapping
    if profile.coverage_status is not None and profile.coverage_status != "covered":
        result["mode"] = "blocked_coverage"
        return result
    if profile.identity_mismatch:
        result["mode"] = "blocked_identity"
        return result
    wall_basis = profile.calibration_basis == "operator_wall"
    total_key = "wall_ns_per_instance" if wall_basis else "train_ns_per_instance"
    instances_key = "wall_train_instances" if wall_basis else "train_instances"
    holdout_instances_key = "wall_holdout_instances" if wall_basis else "holdout_instances"
    error_key = "wall_holdout_relative_error_pct" if wall_basis else "holdout_relative_error_pct"
    if not isinstance(mapping, Mapping):
        result["mode"] = "analytical_fallback"
        return result
    entry = mapping.get(str(stage))
    if not isinstance(entry, Mapping) or str(entry.get("status", "")) != "calibrated":
        result["mode"] = "analytical_fallback"
        return result
    phase_entry = (entry.get("phases") or {}).get(str(phase))
    if isinstance(phase_entry, Mapping) and str(phase_entry.get("status", "")) == "calibrated":
        if token_shape is not None:
            shape_buckets = _shape_bucket_mapping(phase_entry, wall_basis)
            shape_entry = shape_buckets.get(str(token_shape))
            shape_status_key = "wall_status" if wall_basis else "status"
            if (isinstance(shape_entry, Mapping)
                    and str(shape_entry.get(shape_status_key, "")) == "calibrated"):
                value = _finite_non_negative(shape_entry.get(total_key))
                if value is not None and _finite_non_negative(shape_entry.get(instances_key)):
                    result.update(mode="exact", source_shapes=[str(token_shape)], coefficient=value,
                                  uncertainty={"endpoint_holdout_relative_error_pct": shape_entry.get(error_key),
                                               "validated_error_bound_pct": None})
                    return result
            # A shape miss may be interpolated only under an explicit policy.
            if allow_interpolation:
                target = _shape_dimensions(str(token_shape))
                candidates: list[tuple[str, tuple[float, ...], float]] = []
                for key, candidate in shape_buckets.items():
                    if not isinstance(candidate, Mapping) or str(candidate.get(shape_status_key, "")) != "calibrated":
                        continue
                    dims = _shape_dimensions(str(key))
                    value = _finite_non_negative(candidate.get(total_key))
                    instances = _finite_non_negative(candidate.get(instances_key))
                    holdout_instances = _finite_non_negative(candidate.get(holdout_instances_key))
                    if (dims is None or value is None or not instances or not holdout_instances):
                        continue
                    candidates.append((str(key), dims, value))
                if target is not None and len(target) == 4 and target[2:] == (1.0, 1.0) and candidates:
                    # GEMM evidence is N x token_rows x 1 x 1.  Never interpolate
                    # feature widths, heads, KV length or an unnamed tensor axis.
                    for axis in (1,):
                        compatible = [item for item in candidates
                                      if len(item[1]) == len(target)
                                      and all(i == axis or item[1][i] == target[i]
                                              for i in range(len(target)))]
                        compatible.sort(key=lambda item: item[1][axis])
                        lower = [item for item in compatible if item[1][axis] < target[axis]]
                        upper = [item for item in compatible if item[1][axis] > target[axis]]
                        if not lower or not upper:
                            continue
                        lo, hi = lower[-1], upper[0]
                        span = hi[1][axis] - lo[1][axis]
                        if span <= 0:
                            continue
                        weight = (target[axis] - lo[1][axis]) / span
                        value = lo[2] + weight * (hi[2] - lo[2])
                        # Endpoint holdout error and bracket width are useful
                        # diagnostics, but neither proves an interpolation error
                        # bound.  Keep that bound explicitly unknown.
                        errors = []
                        for key in (lo[0], hi[0]):
                            raw = shape_buckets[key].get(error_key)
                            if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                                errors.append(abs(float(raw)))
                        uncertainty = {
                            "endpoint_holdout_max_abs_error_pct": max(errors) if errors else None,
                            "bracket_span_rows": span,
                            "interior_validation_status": "unvalidated",
                            "validated_error_bound_pct": None,
                        }
                        result.update(mode="interpolated", source_shapes=[lo[0], hi[0]],
                                      coefficient=value, uncertainty=uncertainty,
                                      interpolation_axis=axis, interpolation_weight=weight)
                        return result
            # Explicit shape metadata without an exact evidence row is not
            # allowed to fall back to a phase average, even when the profile
            # only contains a phase aggregate.
            result["mode"] = "extrapolation" if token_shape is not None else "analytical_fallback"
            result["extrapolation"] = result["mode"] == "extrapolation"
            return result
        value = _finite_non_negative(phase_entry.get(total_key))
        if value is not None and _finite_non_negative(phase_entry.get(instances_key)):
            result.update(mode="phase", coefficient=value)
            return result
    if phase_entry is None and token_shape is None:
        value = _finite_non_negative(entry.get(total_key))
        if value is not None and _finite_non_negative(entry.get(instances_key)):
            result.update(mode="stage", coefficient=value)
            return result
    result["mode"] = "analytical_fallback"
    return result


def resolve_exact_operator_calibration(
    profile: NativeCalibrationProfile,
    *,
    stage: str,
    phase: str,
    token_shape: str | None,
    dtype: str | None,
    layout: str | None,
    kernel_family: str | None,
) -> dict[str, Any]:
    """Resolve a six-dimensional operator key without cross-shape fallback.

    Exact operator evidence is optional and lives under
    ``kernel_stage_mapping.exact_operator_keys``.  Every dimension is part of
    the key; a missing dimension or key returns ``analytical_fallback`` and
    never consults a stage or phase aggregate.  This keeps sparse microbench
    evidence from becoming an accidental global latency multiplier.
    """
    result: dict[str, Any] = {
        "mode": "analytical_fallback",
        "requested_key": None,
        "source_shapes": [],
        "coefficient": None,
        "uncertainty": None,
        "reason": None,
    }
    mapping = profile.kernel_stage_mapping
    if profile.coverage_status is not None and profile.coverage_status != "covered":
        result["mode"] = "blocked_coverage"
        result["reason"] = "profile_coverage_not_covered"
        return result
    if profile.identity_mismatch:
        result["mode"] = "blocked_identity"
        result["reason"] = "profile_identity_mismatch"
        return result
    if not isinstance(mapping, Mapping):
        result["reason"] = "missing_exact_operator_mapping"
        return result
    key = canonical_exact_operator_key(
        stage=stage, phase=phase, token_shape=token_shape, dtype=dtype,
        layout=layout, kernel_family=kernel_family,
    )
    if key is None:
        result["reason"] = "missing_operator_key_dimension"
        return result
    result["requested_key"] = key
    entries = mapping.get("exact_operator_keys")
    if not isinstance(entries, Mapping):
        entries = mapping.get("_exact_operator_keys")
    if not isinstance(entries, Mapping):
        result["reason"] = "missing_exact_operator_mapping"
        return result
    entry = entries.get(key)
    if not isinstance(entry, Mapping) or str(entry.get("status", "")) != "calibrated":
        result["reason"] = "exact_operator_key_uncovered"
        return result
    wall_basis = profile.calibration_basis == "operator_wall"
    value = _finite_non_negative(entry.get("wall_ns_per_instance" if wall_basis else "train_ns_per_instance"))
    instances = _finite_non_negative(entry.get("wall_train_instances" if wall_basis else "train_instances"))
    if value is None or instances is None or instances <= 0:
        result["reason"] = "exact_operator_key_invalid_rate"
        return result
    result.update({
        "mode": "exact",
        "coefficient": value,
        "source_shapes": [str(token_shape)],
        "uncertainty": {
            "endpoint_holdout_relative_error_pct": entry.get(
                "wall_holdout_relative_error_pct" if wall_basis else "holdout_relative_error_pct"
            ),
            "validated_error_bound_pct": None,
        },
    })
    return result


def audit_exact_operator_coverage(
    profile: NativeCalibrationProfile,
    records: Any,
) -> dict[str, Any]:
    """Count exact-key hits and fail-closed fallbacks for trace records.

    ``records`` is an iterable of normalized mappings.  The six required
    fields may use trace aliases (``semantic_shape``, ``semantic_type``,
    ``semantic_layout``); missing aliases are intentionally counted as
    malformed/fallback rather than guessed from a model or kernel name.
    """
    total = exact_hits = fallback = malformed = 0
    reasons: dict[str, int] = {}
    for record in records or ():
        if not isinstance(record, Mapping):
            continue
        total += 1
        key = canonical_exact_operator_key(
            stage=record.get("stage"), phase=record.get("phase") or record.get("execution_phase"),
            token_shape=record.get("shape") or record.get("token_shape") or record.get("semantic_shape"),
            dtype=record.get("dtype") or record.get("semantic_type"),
            layout=record.get("layout") or record.get("semantic_layout"),
            kernel_family=record.get("kernel_family"),
        )
        if key is None:
            malformed += 1
            reasons["missing_operator_key_dimension"] = reasons.get("missing_operator_key_dimension", 0) + 1
            continue
        resolution = resolve_exact_operator_calibration(
            profile,
            stage=record.get("stage"),
            phase=record.get("phase") or record.get("execution_phase"),
            token_shape=record.get("shape") or record.get("token_shape") or record.get("semantic_shape"),
            dtype=record.get("dtype") or record.get("semantic_type"),
            layout=record.get("layout") or record.get("semantic_layout"),
            kernel_family=record.get("kernel_family"),
        )
        if resolution.get("mode") == "exact":
            exact_hits += 1
        else:
            fallback += 1
            reason = str(resolution.get("reason") or resolution.get("mode") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "schema": "exact-operator-coverage/v1",
        "key_fields": list(EXACT_OPERATOR_KEY_FIELDS),
        "records": total,
        "exact_hits": exact_hits,
        "fallback_records": fallback + malformed,
        "malformed_records": malformed,
        "exact_hit_rate_pct": (100.0 * exact_hits / total) if total else 0.0,
        "fallback_rate_pct": (100.0 * (fallback + malformed) / total) if total else 0.0,
        "fallback_reasons": reasons,
    }


def stage_memory_bandwidth_gbps(profile: NativeCalibrationProfile, *, stage: str,
                                phase: str, token_shape: str | None = None) -> float | None:
    """Return measured physical-byte bandwidth for one exact stage/phase."""
    mapping = profile.kernel_stage_mapping
    if profile.coverage_status is not None and profile.coverage_status != "covered":
        return None
    if profile.identity_mismatch or not isinstance(mapping, Mapping):
        return None
    entry = mapping.get(str(stage))
    phase_entry = entry.get("phases", {}).get(str(phase)) if isinstance(entry, Mapping) else None
    if not isinstance(phase_entry, Mapping) or phase_entry.get("status") != "calibrated":
        return None
    bucket = (phase_entry.get("token_shapes", {}).get(str(token_shape))
              if token_shape is not None and isinstance(phase_entry.get("token_shapes"), Mapping) else None)
    if token_shape is not None and (not isinstance(bucket, Mapping) or bucket.get("status") != "calibrated"):
        return None
    if not isinstance(bucket, Mapping):
        bucket = phase_entry
    value = _finite_non_negative(bucket.get("train_effective_bandwidth_gbps"))
    if value is None and token_shape is None and isinstance(entry, Mapping):
        value = _finite_non_negative(entry.get("train_effective_bandwidth_gbps"))
    return value if value and value > 0 else None


def memory_calibration_ns(phase: CostPhase, metadata: Mapping[str, Any],
                          profile: NativeCalibrationProfile) -> float | None:
    """Compute CPU memory service time from planner bytes and measured GB/s."""
    stage, execution_phase, token_shape = _invocation_fields(metadata)
    if not stage or not execution_phase:
        return None
    bandwidth = stage_memory_bandwidth_gbps(profile, stage=stage, phase=execution_phase,
                                            token_shape=token_shape)
    memory = next((d for d in phase.demands if ".memory" in d.resource_id.lower()), None)
    if bandwidth is None or memory is None or memory.bytes_moved <= 0:
        return None
    return float(memory.bytes_moved) / bandwidth


def _invocation_fields(metadata: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    stage = metadata.get("calibration_stage") or metadata.get("coverage_component")
    # ``stage`` in planner metadata is commonly an integer PP index; never
    # reinterpret that number as a semantic calibration stage.
    if stage is None and isinstance(metadata.get("stage"), str):
        stage = metadata.get("stage")
    phase = metadata.get("execution_phase") or metadata.get("phase")
    if isinstance(phase, str):
        lowered_phase = phase.lower()
        if "prefill" in lowered_phase:
            phase = "prefill"
        elif "decode" in lowered_phase:
            phase = "decode"
    shape = metadata.get("token_shape") or metadata.get("shape")
    projection = str(metadata.get("projection_id", "")).lower()
    event_kind = str(metadata.get("event_kind", "")).lower()
    linear_op = str(metadata.get("linear_op", "")).lower()
    if linear_op in {
        "local_conv", "scan_recurrent_update", "gate_norm_reduce",
        "gate_norm_apply", "residual",
    }:
        # These are the explicit Qwen3.5 gated-delta-net/SSM primitives.  They
        # have no weight projection id, so retain a dedicated stage instead of
        # forcing them into QKV/FFN or dropping them as unknown.
        stage = "linear_attention_aux"
    if event_kind in {"activation_quantization", "dequantize", "quantize"}:
        stage = "quantize"
    elif (event_kind in {"input_norm_reduce", "input_norm_apply",
                             "attention_q_norm_reduce", "attention_q_norm_apply",
                             "attention_k_norm_reduce", "attention_k_norm_apply",
                             "post_attention_norm_reduce", "post_attention_norm_apply",
                             "fused_residual_norm", "final_norm_reduce", "final_norm_apply"}
          or "norm" in event_kind):
        stage = "normalization"
    elif event_kind in {"fused_attention", "softmax_reduce", "softmax_normalize",
                            "attention_gate", "attention_residual", "attention_projection_join"}:
        stage = "attention_output"
    if stage is None and event_kind == "lm_head_projection":
        stage = "lm_head"
        # lm_head is a single named projection in the graph rather than a
        # layer projection id; use a synthetic explicit id for the strict
        # calibration gate below.
        if not projection:
            projection = "lm_head"
    # Planner coverage labels describe the layer family; semantic profiles
    # describe the measured operator family.  Bridge only unambiguous ids.
    if stage in {"full_attention", "attention", "attention_projection"}:
        # The semantic profile has QKV evidence only.  Do not silently reuse
        # it for attention.output or QK/PV kernels, which have different
        # shapes and launch paths.
        stage = "attention_qkv" if projection == "attention.qkv" else None
    elif stage == "linear_attention":
        stage = "attention_qkv" if projection == "linear_attention.qkv" else None
    elif stage in {"feed_forward", "ffn_projection", "ffn", "dense_ffn", "routed_expert", "shared_expert"}:
        stage = "ffn" if projection in {"mlp.gate", "mlp.up", "mlp.up_gate", "mlp.down"} else None
    elif stage in {"output", "output_projection", "lm-head", "lm_head"}:
        stage = "lm_head" if projection == "lm_head" else None
    elif stage in {"attention_output", "normalization", "quantize"}:
        stage = str(stage)
    return (
        str(stage) if stage is not None else None,
        str(phase) if phase is not None else None,
        str(shape) if shape is not None else None,
    )


def calibrate_cost_phase(
    phase: CostPhase,
    metadata: Mapping[str, Any],
    profile: NativeCalibrationProfile,
    *,
    model_sha256: str | None = None,
    hardware_fingerprint: str | None = None,
    runtime_fingerprint: str | None = None,
    apply_memory: bool = False,
) -> CostPhase:
    """Apply one exact stage coefficient to one planner phase.

    The caller must provide a projection invocation id.  Unknown stages,
    missing shape evidence, identity mismatches, and blocked coverage return
    the original phase unchanged.
    """
    if not isinstance(phase, CostPhase) or not isinstance(metadata, Mapping):
        return phase
    if not _profile_identity_matches(
        profile,
        model_sha256=model_sha256,
        hardware_fingerprint=hardware_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
    ):
        return phase
    stage, execution_phase, token_shape = _invocation_fields(metadata)
    if not stage or not execution_phase or stage == "unknown":
        return phase
    # A stage total is only safe to charge to a named projection invocation.
    if (not metadata.get("projection_id")
            and str(metadata.get("event_kind", "")).lower() != "lm_head_projection"
            and stage not in {"attention_output", "normalization", "quantize", "linear_attention_aux"}):
        return phase
    # A kernel-basis profile may carry stable invocation-shape buckets while
    # its wall-time buckets include host scheduling and stream synchronization.
    # Opt in explicitly at the scenario boundary.  A missing shape (or any
    # non-kernel profile) falls back to the existing phase coefficient, so an
    # unverified shape can never disable the proven phase path or invent a
    # coefficient.
    shape_policy = str(metadata.get("native_calibration_shape_policy", ""))
    # An exact operator-key policy is stricter than stage/shape calibration:
    # all semantic dimensions must be present and matched.  It deliberately
    # bypasses the phase aggregate and interpolation paths below.
    explicit_exact_policy = str(
        metadata.get("native_calibration_operator_key_policy", "")
    ).strip().lower() in {"exact", "exact_key", "exact_operator_key"}
    # A profile carrying an exact-key table opts into the strict gate even if
    # an older caller omitted the metadata switch.  This prevents accidental
    # fallback to a stage/phase aggregate when sparse exact evidence exists.
    profile_mapping = profile.kernel_stage_mapping
    profile_has_exact_table = (
        isinstance(profile_mapping, Mapping)
        and isinstance(profile_mapping.get("exact_operator_keys") or profile_mapping.get("_exact_operator_keys"), Mapping)
    )
    exact_key_policy = explicit_exact_policy or profile_has_exact_table
    if exact_key_policy:
        resolution = resolve_exact_operator_calibration(
            profile,
            stage=stage,
            phase=execution_phase,
            token_shape=token_shape,
            dtype=(metadata.get("semantic_dtype") or metadata.get("semantic_type")
                   or metadata.get("dtype") or metadata.get("calibration_dtype")),
            layout=metadata.get("semantic_layout") or metadata.get("layout") or metadata.get("calibration_layout"),
            kernel_family=metadata.get("kernel_family") or metadata.get("calibration_kernel_family"),
        )
        coefficient = (float(resolution["coefficient"])
                       if resolution.get("coefficient") is not None else None)
        memory_ns = memory_calibration_ns(phase, metadata, profile) if apply_memory else None
        if coefficient is None and memory_ns is None:
            phase_metadata = dict(phase.metadata)
            phase_metadata["native_calibration_operator_key_resolution"] = resolution
            return CostPhase(name=phase.name, category=phase.category,
                             demands=phase.demands, metadata=phase_metadata)
    else:
        resolution = None
        coefficient = None
    kernel_shape_mode = shape_policy == "kernel_shape_if_available"
    interpolation_requested = shape_policy in {
        "interpolate_within_evidence",
        "kernel_shape_interpolate_within_evidence",
    }
    if not exact_key_policy and kernel_shape_mode and profile.calibration_basis == "kernel":
        resolution = resolve_stage_calibration(
            profile, stage=stage, phase=execution_phase, token_shape=token_shape,
            allow_interpolation=interpolation_requested,
        )
        coefficient = (float(resolution["coefficient"])
                       if resolution.get("coefficient") is not None else None)
        # Preserve the historical explicit kernel-shape policy: a missing
        # bucket may use the phase aggregate, but only as a labelled
        # analytical fallback (never as an unreported exact hit).
        if coefficient is None and not interpolation_requested:
            fallback = resolve_stage_calibration(
                profile, stage=stage, phase=execution_phase, token_shape=None,
            )
            coefficient = (float(fallback["coefficient"])
                           if fallback.get("coefficient") is not None else None)
            if resolution.get("mode") == "extrapolation" and fallback.get("coefficient") is not None:
                resolution["mode"] = "analytical_fallback"
                resolution["fallback_mode"] = fallback.get("mode")
    elif not exact_key_policy:
        resolution = resolve_stage_calibration(
            profile, stage=stage, phase=execution_phase,
            token_shape=None if kernel_shape_mode else token_shape,
            allow_interpolation=interpolation_requested,
        )
        coefficient = (float(resolution["coefficient"])
                       if resolution.get("coefficient") is not None else None)
    memory_ns = memory_calibration_ns(phase, metadata, profile) if apply_memory else None
    if coefficient is None and memory_ns is None:
        # Keep provenance even when calibration is blocked or out of range;
        # the analytical phase remains intact, but callers can distinguish a
        # deliberate fallback from a shape hit.
        if token_shape is not None and interpolation_requested:
            phase_metadata = dict(phase.metadata)
            phase_metadata["native_calibration_shape_resolution"] = resolution
            return CostPhase(name=phase.name, category=phase.category,
                             demands=phase.demands, metadata=phase_metadata)
        return phase
    instances = _finite_non_negative(metadata.get("calibration_instances", 1))
    if instances is None or instances <= 0:
        return phase
    target_ns = coefficient * instances if coefficient is not None else None
    phase_launch = (profile.prefill_launch_ns_per_call if execution_phase == "prefill"
                    else profile.decode_launch_ns_per_call if execution_phase == "decode"
                    else None)
    # Change only the explicit compute demand; memory and transfer demands
    # retain their analytical service and are never used to distribute timing.
    compute_indices = [
        i for i, demand in enumerate(phase.demands)
        if any(marker in demand.resource_id.lower() for marker in ("tensor", "scalar", "compute", "sfu"))
    ]
    if not compute_indices and memory_ns is None:
        return phase
    demands = list(phase.demands)
    if phase_launch is not None:
        for i, demand in enumerate(demands):
            if "frontend" in demand.resource_id.lower() or "launch" in demand.resource_id.lower():
                demands[i] = ResourceDemand(
                    resource_id=demand.resource_id,
                    service_ns=phase_launch * instances,
                    bytes_moved=demand.bytes_moved,
                    energy_pj=demand.energy_pj,
                    work_units=demand.work_units,
                )
                break
    if coefficient is not None and compute_indices:
        index = max(compute_indices, key=lambda i: phase.demands[i].service_ns)
        demands[index] = ResourceDemand(
            resource_id=demands[index].resource_id,
            service_ns=target_ns,
            bytes_moved=demands[index].bytes_moved,
            energy_pj=demands[index].energy_pj,
            work_units=demands[index].work_units,
        )
    if memory_ns is not None:
        for i, demand in enumerate(demands):
            if ".memory" in demand.resource_id.lower():
                demands[i] = ResourceDemand(
                    resource_id=demand.resource_id, service_ns=memory_ns,
                    bytes_moved=demand.bytes_moved, energy_pj=demand.energy_pj,
                    work_units=demand.work_units,
                )
                break
    phase_metadata = dict(phase.metadata)
    phase_metadata.update({
        "native_calibration_applied": True,
        "native_calibration_stage": stage,
        "native_calibration_phase": execution_phase,
        "native_calibration_token_shape": token_shape,
        "native_calibration_ns_per_instance": coefficient,
        "native_calibration_instances": instances,
        "native_calibration_shape_resolution": resolution,
        **({"native_calibration_operator_key_resolution": resolution}
           if exact_key_policy else {}),
        **({"native_launch_calibration_applied": True,
            "native_launch_calibration_phase": execution_phase,
            "native_launch_ns_per_call": phase_launch}
           if phase_launch is not None else {}),
        **({"native_memory_calibration_applied": True,
            "native_memory_calibration_service_ns": memory_ns} if memory_ns is not None else {}),
    })
    return CostPhase(name=phase.name, category=phase.category, demands=tuple(demands), metadata=phase_metadata)


def synchronize_calibration_ns(
    metadata: Mapping[str, Any], profile: NativeCalibrationProfile
) -> float | None:
    """Return sync cost only for an explicit synchronization boundary count."""
    if str(metadata.get("event_kind", "")).lower() not in {"sync", "synchronize", "synchronization"}:
        return None
    count = _finite_non_negative(metadata.get("sync_boundary_count"))
    per_call = _finite_non_negative(profile.synchronize_ns_per_call)
    if count is None or per_call is None or count <= 0:
        return None
    return per_call * count


def phase_boundary_calibration_ns(
    profile: NativeCalibrationProfile | Mapping[str, Any] | None,
    execution_phase: object,
    *,
    model_sha256: str | None = None,
    hardware_fingerprint: str | None = None,
    runtime_fingerprint: str | None = None,
) -> float | None:
    """Return one measured launch+sync boundary cost per physical invocation.

    Evidence must explicitly declare ``one_task_per_phase_invocation``.  A
    phase aggregate without that policy is intentionally blocked: API calls
    are not one-to-one with planner operators and spreading them would count
    launch/synchronization more than once.
    """
    if not isinstance(profile, NativeCalibrationProfile):
        profile = profile_from_mapping(profile)
    if profile is None or profile.phase_boundary_policy != "one_task_per_phase_invocation":
        return None
    if profile.coverage_status is not None and profile.coverage_status != "covered":
        return None
    if profile.identity_mismatch or not _profile_identity_matches(
        profile,
        model_sha256=model_sha256,
        hardware_fingerprint=hardware_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
    ):
        return None
    phase = str(execution_phase or "").strip().lower()
    value = (profile.phase_boundary_ns_per_invocation or {}).get(phase)
    return _finite_non_negative(value)


def decode_first_invocation_extra_ns(
    profile: NativeCalibrationProfile | Mapping[str, Any] | None,
    *,
    first_invocation: bool,
    model_sha256: str | None = None,
    hardware_fingerprint: str | None = None,
    runtime_fingerprint: str | None = None,
) -> float | None:
    """Return an evidence-backed one-time CUDA decode startup residual.

    The value is intentionally opt-in and only applies to the first decode
    invocation.  It is distinct from the recurring phase boundary and from
    per-kernel launch latency, so later tokens cannot be charged repeatedly.
    """
    if not first_invocation:
        return None
    if not isinstance(profile, NativeCalibrationProfile):
        profile = profile_from_mapping(profile)
    if (
        profile is None
        or profile.decode_first_invocation_extra_ns is None
        or profile.decode_first_invocation_policy != "first_decode_only"
    ):
        return None
    if profile.coverage_status is not None and profile.coverage_status != "covered":
        return None
    if profile.identity_mismatch or not _profile_identity_matches(
        profile,
        model_sha256=model_sha256,
        hardware_fingerprint=hardware_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
    ):
        return None
    return _finite_non_negative(profile.decode_first_invocation_extra_ns)


_REQUEST_MARKER_ALIASES = {
    "request_begin": "request_begin",
    "request_arrival": "request_begin",
    "first_token": "first_token",
    "request_end": "request_end",
    "request_done": "request_end",
}


def request_marker_calibration_ns(
    profile: NativeCalibrationProfile | Mapping[str, Any] | None,
    marker: object,
    *,
    model_sha256: str | None = None,
    hardware_fingerprint: str | None = None,
    runtime_fingerprint: str | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    prompt_fingerprint: str | None = None,
) -> float | None:
    """Return one exact request-marker additive cost, or ``None``.

    A marker profile is deliberately stricter than phase calibration.  It
    must declare ``additive_once_per_request``, carry a calibrated evidence
    row for this marker, and bind both prompt/output token counts.  This
    prevents a request boundary measured for one prompt from being reused on
    another shape or from silently applying a total request wall time twice.
    """
    if not isinstance(profile, NativeCalibrationProfile):
        profile = profile_from_mapping(profile)
    if profile is None or profile.request_marker_policy != "additive_once_per_request":
        return None
    if profile.coverage_status is not None and profile.coverage_status != "covered":
        return None
    if profile.identity_mismatch or not _profile_identity_matches(
        profile,
        model_sha256=model_sha256,
        hardware_fingerprint=hardware_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
    ):
        return None
    canonical = _REQUEST_MARKER_ALIASES.get(str(marker or "").strip().lower())
    if canonical is None or not isinstance(profile.request_marker_ns, Mapping):
        return None
    evidence = profile.request_marker_evidence
    evidence_row = evidence.get(canonical) if isinstance(evidence, Mapping) else None
    if not isinstance(evidence_row, Mapping) or str(evidence_row.get("status", "")) != "calibrated":
        return None
    shape = profile.request_shape
    # Marker evidence without an exact shape is not safe to apply.  Require
    # both dimensions even when the caller only wants the first-token marker.
    if not isinstance(shape, Mapping) or "prompt_tokens" not in shape or "output_tokens" not in shape:
        return None
    try:
        expected_prompt = int(shape["prompt_tokens"])
        expected_output = int(shape["output_tokens"])
    except (TypeError, ValueError, OverflowError):
        return None
    if prompt_tokens is None or output_tokens is None:
        return None
    if int(prompt_tokens) != expected_prompt or int(output_tokens) != expected_output:
        return None
    expected_fingerprint = shape.get("prompt_fingerprint")
    if expected_fingerprint is not None and (
        prompt_fingerprint is None or str(expected_fingerprint) != str(prompt_fingerprint)
    ):
        return None
    return _finite_non_negative(profile.request_marker_ns.get(canonical))


def launch_calibration_ns(
    profile: NativeCalibrationProfile | Mapping[str, Any] | None,
    execution_phase: object,
) -> float | None:
    """Return an explicitly phase-scoped CUDA launch rate.

    A launch coefficient is only safe when its prefill/decode scope is
    declared.  The legacy ``launch_ns_per_call`` field is intentionally not
    used here: applying that global value to every lowered primitive can
    charge one native graph launch dozens of times.  Callers that have a
    measured phase-specific rate can bind it to the corresponding
    ``kernel_launch`` phase while leaving the other phase analytical.
    """
    if not isinstance(profile, NativeCalibrationProfile):
        profile = profile_from_mapping(profile)
    if profile is None:
        return None
    phase = str(execution_phase or "").strip().lower()
    if phase == "prefill":
        return _finite_non_negative(profile.prefill_launch_ns_per_call)
    if phase == "decode":
        return _finite_non_negative(profile.decode_launch_ns_per_call)
    return None


def apply_native_calibration(
    scenario: ScenarioConfig,
    profile: NativeCalibrationProfile,
    *,
    apply_launch: bool = False,
    apply_stage: bool = False,
    apply_memory: bool = False,
    apply_phase_boundary: bool = False,
    apply_request_boundary: bool = False,
) -> ScenarioConfig:
    """Apply only launch evidence to GPU frontend demand.

    Prompt/decode coefficients remain metadata until the matching task-level
    markers are available.  This fail-closed behavior prevents a stage total
    from being silently charged to every GEMM.
    """
    if not isinstance(profile, NativeCalibrationProfile):
        raise TypeError("profile must be NativeCalibrationProfile")
    # Launch timing is hardware/backend-specific too.  Refuse to mutate the
    # GPU profile when a supplied identity disagrees with this scenario; keep
    # the evidence in metadata for auditability.
    identity_ok = _profile_identity_matches(
        profile,
        model_sha256=_model_gguf_sha256(scenario.model),
        hardware_fingerprint=scenario.placement.metadata.get("hardware_fingerprint"),
        runtime_fingerprint=(scenario.llama_cpp_config.fingerprint
                             if scenario.llama_cpp_config is not None else None),
    )
    # Stage, memory, phase-boundary and request-boundary calibration are
    # independent of the optional CUDA launch override.  The previous
    # implementation returned here whenever ``apply_launch`` was false,
    # silently discarding an explicit ``apply_stage=True`` request.  Keep the
    # launch mutation gated, but always retain the other calibration gates so
    # planner-level stage costs can be applied.
    if not identity_ok:
        metadata = dict(scenario.placement.metadata)
        metadata["native_calibration"] = profile.to_dict()
        metadata["native_calibration_apply_stage"] = bool(apply_stage)
        metadata["native_calibration_apply_memory"] = bool(apply_memory)
        metadata["native_calibration_apply_phase_boundary"] = bool(apply_phase_boundary)
        metadata["native_calibration_apply_request_boundary"] = bool(apply_request_boundary)
        metadata["native_calibration_apply_launch"] = False
        return replace(scenario, placement=replace(scenario.placement, metadata=metadata))
    profiles = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    gpu_registry = profiles.get("gpu", {})
    gpu_components = [c for c in scenario.hardware.components if c.normalized_kind == "gpu"]
    target_gpu_id = None
    if gpu_components:
        main_gpu = getattr(scenario.llama_cpp_config, "main_gpu", 0) if scenario.llama_cpp_config else 0
        target_gpu_id = next((c.component_id for c in gpu_components if c.component_id == f"gpu{main_gpu}"), gpu_components[0].component_id)
    changed = False
    for key, gpu in tuple(gpu_registry.items()):
        bound = next((c for c in gpu_components if c.cost_profile_id == key), None)
        if (apply_launch and profile.launch_ns_per_call is not None
                and profile.launch_ns_per_call >= 0 and bound is not None
                and bound.component_id == target_gpu_id):
            gpu_registry[key] = replace(gpu, kernel_launch_ns=float(profile.launch_ns_per_call))
            changed = True
    if not changed:
        metadata = dict(scenario.placement.metadata)
        metadata["native_calibration"] = profile.to_dict()
        metadata["native_calibration_apply_stage"] = bool(apply_stage)
        metadata["native_calibration_apply_memory"] = bool(apply_memory)
        metadata["native_calibration_apply_phase_boundary"] = bool(apply_phase_boundary)
        metadata["native_calibration_apply_request_boundary"] = bool(apply_request_boundary)
        metadata["native_calibration_apply_launch"] = False
        return replace(scenario, placement=replace(scenario.placement, metadata=metadata))
    metadata = dict(scenario.placement.metadata)
    metadata["native_calibration"] = profile.to_dict()
    metadata["native_calibration_apply_stage"] = bool(apply_stage)
    metadata["native_calibration_apply_memory"] = bool(apply_memory)
    metadata["native_calibration_apply_phase_boundary"] = bool(apply_phase_boundary)
    metadata["native_calibration_apply_request_boundary"] = bool(apply_request_boundary)
    assumptions = tuple(dict.fromkeys((*scenario.assumptions,
        "native calibration: stage/memory/phase gates retained; CUDA launch coefficient applied only when explicitly enabled")))
    return replace(scenario, component_profiles=profiles, placement=replace(scenario.placement, metadata=metadata), assumptions=assumptions)


__all__ = [
    "NativeCalibrationProfile", "load_native_calibration", "apply_native_calibration",
    "profile_from_mapping",
    "EXACT_OPERATOR_KEY_FIELDS", "canonical_exact_operator_key", "audit_exact_operator_coverage",
    "stage_calibration_ns_per_instance", "resolve_stage_calibration", "resolve_exact_operator_calibration", "calibrate_cost_phase", "synchronize_calibration_ns", "phase_boundary_calibration_ns", "decode_first_invocation_extra_ns", "request_marker_calibration_ns",
    "launch_calibration_ns",
]
