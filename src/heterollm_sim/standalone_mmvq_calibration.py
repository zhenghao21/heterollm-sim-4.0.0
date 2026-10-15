"""Strict diagnostic import for native standalone MMVQ measurements.

The full LLM trace is useful for ownership and error ranking, but it cannot be
used as a per-kernel rate.  This module accepts only an isolated native node
measurement from the same binary, model, source contract, and GPU, and annotates
its eligibility for diagnostic review.  It never installs a measured rate or
correction into a prediction profile; no interpolation or whole-model latency
fitting is performed.

Rows for the ``M=512`` control group are retained in the receipt, but are not
installed into the MMVQ surface.  That source path is MMQ on the locked
llama.cpp dispatch contract and therefore needs its own MMQ surface owner.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

STANDALONE_MMVQ_SCHEMA = "heterollm.native-mmvq-standalone/v1"
CUDA_EVENT_BOUNDARY = "cuda_event_isolated_native_node"
# The backend CUDA stream is private, so a public CUDA event can delimit the
# one-op graph while CUPTI supplies the isolated main-kernel interval inside
# that range.  This is accepted only for a graph containing one main node.
CUPTI_NODE_BOUNDARY = "cuda_event_operator_envelope+cupti_isolated_native_node"
_ACCEPTED_TIMING_BOUNDARIES = frozenset({CUDA_EVENT_BOUNDARY, CUPTI_NODE_BOUNDARY})
_SHA256_LENGTH = 64
_FORMATS = frozenset({"q5_0", "q8_0"})
_KERNEL_FAMILIES = frozenset({"mmvq", "mmq"})
_CACHE_PROTOCOLS = frozenset({
    "cold_streaming", "warm_steady_state", "warm_reuse",
    "warm_reuse_with_output_readback",
    "explicit_l2_eviction_attempt",
})


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _shape(value: Any) -> tuple[int, int, int]:
    if isinstance(value, Mapping):
        value = [value.get(axis) for axis in ("m", "n", "k")]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("shape must be an M,N,K triple")
    return tuple(_positive_int(item, "shape coordinate") for item in value)


def _geometry(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-dimensional launch geometry")
    return tuple(_positive_int(item, name) for item in value)


def _durations(value: Any, name: str, *, minimum: int = 2) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise ValueError(f"{name} requires at least {minimum} repeated CUDA-event samples")
    return tuple(_positive_float(item, name) for item in value)


def _median_and_stddev(values: Sequence[float]) -> tuple[float, float]:
    median = float(statistics.median(values))
    # A single timing value is never accepted by this module, but keeping the
    # fallback makes the helper safe when used by diagnostic callers.
    stddev = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return median, stddev


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True)
class StandaloneMMVQSample:
    """One isolated native node with exact source and launch identity."""

    case_id: str
    stage: str
    phase: str
    kernel_family: str
    weight_format: str
    shape: tuple[int, int, int]
    # Optional because the raw standalone harness deliberately does not bind
    # an analytical prediction.  A missing baseline keeps the row diagnostic
    # and can never make it eligible for a prediction surface.
    analytical_ns: float | None
    formal_device_ns: tuple[float, ...]
    holdout_device_ns: tuple[float, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dispatch_signature: str
    timing_boundary: str
    cache_protocol: str
    fusion_mode: str
    activation_dtype: str
    numerical_validation_passed: bool
    numerical_validation_count: int
    source_geometry_valid: bool = True
    source_geometry_reason: str | None = None

    @property
    def m(self) -> int:
        return self.shape[0]

    @property
    def n(self) -> int:
        return self.shape[1]

    @property
    def k(self) -> int:
        return self.shape[2]

    @property
    def formal_median_ns(self) -> float:
        return _median_and_stddev(self.formal_device_ns)[0]

    @property
    def formal_stddev_ns(self) -> float:
        return _median_and_stddev(self.formal_device_ns)[1]

    @property
    def formal_cv(self) -> float:
        return self.formal_stddev_ns / self.formal_median_ns

    @property
    def holdout_median_ns(self) -> float | None:
        return (_median_and_stddev(self.holdout_device_ns)[0]
                if self.holdout_device_ns else None)

    @property
    def holdout_relative_error(self) -> float | None:
        median = self.holdout_median_ns
        return (abs(self.formal_median_ns - median) / median
                if median is not None else None)

    @property
    def eligibility_reasons(self) -> tuple[str, ...]:
        """Return explicit rejection reasons instead of hiding bad rows."""
        reasons: list[str] = []
        if self.stage != "main":
            reasons.append("stage_is_not_main")
        if self.phase != "decode":
            reasons.append("phase_is_not_decode")
        if self.kernel_family != "mmvq":
            reasons.append("kernel_family_is_not_mmvq")
        if self.m != 1:
            reasons.append("non_exact_m1_mmvq_shape")
        if self.analytical_ns is None:
            reasons.append("analytical_baseline_unbound")
        if self.m > 8:
            reasons.append("m512_or_large_batch_is_mmq_control")
        if self.timing_boundary not in _ACCEPTED_TIMING_BOUNDARIES:
            reasons.append("timing_boundary_not_isolated_cuda_event")
        if self.cache_protocol != "cold_streaming":
            reasons.append("warm_cache_not_valid_for_cold_surface")
        if self.fusion_mode != "none":
            reasons.append("fused_node_not_a_standalone_mmvq_sample")
        if not self.source_geometry_valid:
            reasons.append(self.source_geometry_reason or "source_geometry_binding_failed")
        if not self.numerical_validation_passed:
            reasons.append("numerical_validation_failed")
        if not self.holdout_device_ns:
            reasons.append("independent_holdout_missing")
        if self.formal_cv >= 0.10:
            reasons.append("formal_repeatability_cv_ge_10_percent")
        if self.holdout_relative_error is not None and self.holdout_relative_error >= 0.10:
            reasons.append("holdout_error_ge_10_percent")
        return tuple(reasons)

    @property
    def mmvq_surface_eligible(self) -> bool:
        """Only exact M=1 decode MMVQ rows can become a production sample."""
        return not self.eligibility_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "phase": self.phase,
            "kernel_family": self.kernel_family,
            "weight_format": self.weight_format,
            "shape": list(self.shape),
            "analytical_ns": self.analytical_ns,
            "formal_device_ns": list(self.formal_device_ns),
            "holdout_device_ns": list(self.holdout_device_ns),
            "grid": list(self.grid),
            "block": list(self.block),
            "dispatch_signature": self.dispatch_signature,
            "timing_boundary": self.timing_boundary,
            "cache_protocol": self.cache_protocol,
            "fusion_mode": self.fusion_mode,
            "activation_dtype": self.activation_dtype,
            "source_geometry_valid": self.source_geometry_valid,
            "source_geometry_reason": self.source_geometry_reason,
            "numerical_validation": {
                "passed": self.numerical_validation_passed,
                "count": self.numerical_validation_count,
            },
            "mmvq_surface_eligible": self.mmvq_surface_eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
            "formal_median_ns": self.formal_median_ns,
            "formal_stddev_ns": self.formal_stddev_ns,
            "formal_cv": self.formal_cv,
            "holdout_relative_error": self.holdout_relative_error,
        }


def _parse_sample(raw: Any, index: int, top_cache_protocol: str) -> StandaloneMMVQSample:
    raw = _mapping(raw, f"rows[{index}]")
    case_id = raw.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"rows[{index}].case_id is required")
    stage = raw.get("stage")
    if stage not in {"main", "conversion", "fixup", "control"}:
        raise ValueError(f"rows[{index}].stage is unsupported")
    phase = raw.get("phase")
    if phase not in {"prefill", "decode", "control"}:
        raise ValueError(f"rows[{index}].phase is unsupported")
    kernel_family = str(raw.get("kernel_family", "")).casefold()
    if kernel_family not in _KERNEL_FAMILIES:
        raise ValueError(f"rows[{index}].kernel_family is unsupported")
    weight_format = str(raw.get("weight_format", raw.get("quant", ""))).casefold()
    if weight_format not in _FORMATS:
        raise ValueError(f"rows[{index}].weight_format must be q5_0 or q8_0")
    shape = _shape(raw.get("shape"))
    analytical_raw = raw.get("analytical_ns")
    analytical_ns = (_positive_float(analytical_raw, f"rows[{index}].analytical_ns")
                     if analytical_raw is not None else None)
    formal = _durations(raw.get("formal_device_ns"), f"rows[{index}].formal_device_ns", minimum=3)
    holdout_raw = raw.get("holdout_device_ns", ())
    holdout = (_durations(holdout_raw, f"rows[{index}].holdout_device_ns", minimum=2)
               if holdout_raw else ())
    grid = _geometry(raw.get("grid"), f"rows[{index}].grid")
    block = _geometry(raw.get("block"), f"rows[{index}].block")
    dispatch_signature = raw.get("dispatch_signature")
    if not isinstance(dispatch_signature, str) or not dispatch_signature.strip():
        raise ValueError(f"rows[{index}].dispatch_signature is required")
    timing_boundary = raw.get("timing_boundary")
    if timing_boundary not in _ACCEPTED_TIMING_BOUNDARIES:
        raise ValueError(f"rows[{index}] is not a CUDA-event/CUPTI isolated node measurement")
    cache_protocol = raw.get("cache_protocol", top_cache_protocol)
    if cache_protocol not in _CACHE_PROTOCOLS or cache_protocol != top_cache_protocol:
        raise ValueError(f"rows[{index}].cache_protocol differs from receipt")
    fusion_mode = raw.get("fusion_mode", "none")
    if fusion_mode not in {"none", "native_fused"}:
        raise ValueError(f"rows[{index}].fusion_mode is unsupported")
    activation_dtype = raw.get("activation_dtype", "fp32")
    if activation_dtype not in {"fp16", "fp32"}:
        raise ValueError(f"rows[{index}].activation_dtype is unsupported")
    numeric = _mapping(raw.get("numerical_validation"), f"rows[{index}].numerical_validation")
    passed = numeric.get("passed")
    if type(passed) is not bool:
        raise ValueError(f"rows[{index}].numerical_validation.passed must be boolean")
    count = _positive_int(numeric.get("count"), f"rows[{index}].numerical_validation.count")
    return StandaloneMMVQSample(
        case_id=case_id.strip(), stage=stage, phase=phase,
        kernel_family=kernel_family, weight_format=weight_format,
        shape=shape, analytical_ns=analytical_ns, formal_device_ns=formal,
        holdout_device_ns=holdout, grid=grid, block=block,
        dispatch_signature=dispatch_signature.strip(), timing_boundary=timing_boundary,
        cache_protocol=cache_protocol, fusion_mode=fusion_mode,
        activation_dtype=activation_dtype,
        numerical_validation_passed=passed, numerical_validation_count=count,
    )


def _source_geometry_check(
    sample: StandaloneMMVQSample,
    *,
    runtime_binary_sha256: str,
    source_hashes: Mapping[str, str],
) -> tuple[bool, str | None]:
    """Cross-check a native row against the locked MMVQ source dispatcher."""
    # M=512 is intentionally an MMQ control.  It must remain in the receipt,
    # but it is not an MMVQ source geometry candidate.
    if sample.kernel_family != "mmvq" or sample.m != 1:
        return True, None
    from .mmvq_work import MMVQSourceContract, SOURCE_SHA256, UnsupportedMMVQ, derive_mmvq_work
    if any(source_hashes.get(path) != digest for path, digest in SOURCE_SHA256.items()):
        return False, "source_hashes_do_not_match_locked_mmvq_contract"
    try:
        source = derive_mmvq_work(
            m=sample.m, n=sample.n, k=sample.k,
            weight_format=sample.weight_format.upper(),
            contract=MMVQSourceContract(
                1200, 1200, 32, SOURCE_SHA256, runtime_binary_sha256,
                True, False, True,
            ),
            allow_k_formats=False,
            allow_iq4_xs=False,
        )
    except UnsupportedMMVQ:
        return False, "shape_or_format_outside_locked_mmvq_dispatch"
    type_id = {"q5_0": 6, "q8_0": 8}[sample.weight_format]
    expected_signature = (
        f"mmvq:type={type_id}:m={source.m}:fusion=0:small_k={int(source.small_k)}:"
        f"halve_iters=0:warps={source.warps_per_cta}:rows={source.rows_per_cta}"
    )
    if source.grid != sample.grid:
        return False, "grid_differs_from_locked_mmvq_dispatch"
    if source.block != sample.block:
        return False, "block_differs_from_locked_mmvq_dispatch"
    if sample.dispatch_signature != expected_signature:
        return False, "dispatch_signature_differs_from_locked_mmvq_dispatch"
    return True, None


@dataclass(frozen=True)
class StandaloneMMVQReceipt:
    hardware_id: str
    runtime_binary_sha256: str
    gguf_sha256: str
    runtime_components: tuple[tuple[str, str], ...]
    source_hashes: Mapping[str, str]
    cache_protocol: str
    samples: tuple[StandaloneMMVQSample, ...]
    qualification: Mapping[str, Any]
    evidence_path: str
    schema: str = STANDALONE_MMVQ_SCHEMA

    @property
    def qualified_samples(self) -> tuple[StandaloneMMVQSample, ...]:
        if not self.environment_qualified:
            return ()
        return tuple(sample for sample in self.samples if sample.mmvq_surface_eligible)

    @property
    def environment_qualified(self) -> bool:
        # A non-empty active process list or a profiler warning invalidates a
        # timing surface.  Preserve the diagnostic row instead of silently
        # dropping it from the receipt.
        return (self.qualification.get("exclusive_gpu") is True
                and self.qualification.get("active_compute_processes") == 0
                and self.qualification.get("diagnostics_clean") is True)

    @property
    def environment_rejection_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.qualification.get("exclusive_gpu") is not True:
            reasons.append("exclusive_gpu_not_proven")
        if self.qualification.get("active_compute_processes") != 0:
            reasons.append("active_compute_processes_present")
        if self.qualification.get("unknown_graphics_contexts", 0) != 0:
            reasons.append("unknown_graphics_contexts_present")
        if self.qualification.get("diagnostics_clean") is not True:
            reasons.append("diagnostics_not_clean")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hardware_id": self.hardware_id,
            "runtime_binary_sha256": self.runtime_binary_sha256,
            "gguf_sha256": self.gguf_sha256,
            "runtime_components": [
                {"path": path, "sha256": digest}
                for path, digest in self.runtime_components
            ],
            "source_hashes": dict(self.source_hashes),
            "cache_protocol": self.cache_protocol,
            "qualification": _plain_json(self.qualification),
            "environment_qualified": self.environment_qualified,
            "environment_rejection_reasons": list(self.environment_rejection_reasons),
            "samples": [sample.to_dict() for sample in self.samples],
            "qualified_sample_count": len(self.qualified_samples),
            "evidence_path": self.evidence_path,
        }


def load_standalone_mmvq_receipt(path: str | Path, *, expected_sha256: str | None = None) -> StandaloneMMVQReceipt:
    """Load a receipt without accepting unqualified rows as calibration."""
    path = Path(path).resolve()
    raw_bytes = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
        raise ValueError("standalone MMVQ receipt SHA-256 mismatch")
    raw = _mapping(json.loads(raw_bytes.decode("utf-8")), "receipt")
    if raw.get("schema") != STANDALONE_MMVQ_SCHEMA or raw.get("status") != "measured":
        raise ValueError("unsupported or unfinished standalone MMVQ receipt")
    hardware_id = raw.get("hardware_id")
    if not isinstance(hardware_id, str) or not hardware_id.strip():
        raise ValueError("hardware_id is required")
    runtime_sha = _sha(raw.get("runtime_binary_sha256"), "runtime_binary_sha256")
    gguf_sha = _sha(raw.get("gguf_sha256"), "gguf_sha256")
    components_raw = raw.get("runtime_components")
    if not isinstance(components_raw, (list, tuple)) or not components_raw:
        raise ValueError("runtime_components are required for DLL identity binding")
    components: list[tuple[str, str]] = []
    for index, component in enumerate(components_raw):
        component = _mapping(component, f"runtime_components[{index}]")
        component_path = component.get("path")
        if not isinstance(component_path, str) or not component_path.strip():
            raise ValueError("runtime component path is required")
        components.append((component_path.strip(), _sha(component.get("sha256"), "runtime component SHA")))
    if not any(Path(component_path).name.casefold() == "ggml-cuda.dll"
               for component_path, _ in components):
        raise ValueError("runtime_components must bind ggml-cuda.dll")
    source_raw = _mapping(raw.get("source_hashes"), "source_hashes")
    if not source_raw:
        raise ValueError("source_hashes are required")
    source_hashes = {str(key): _sha(value, "source hash") for key, value in source_raw.items()}
    cache_protocol = raw.get("cache_protocol")
    if cache_protocol not in _CACHE_PROTOCOLS:
        raise ValueError("unsupported standalone MMVQ cache protocol")
    qualification = _mapping(raw.get("qualification"), "qualification")
    # The runner emits the aggregate measurement list as ``rows``; accept
    # ``samples`` as the normalized interchange spelling used by hand-built
    # receipts.  Both remain diagnostic until a separately reviewed source
    # fix exists.
    samples_raw = raw.get("samples", raw.get("rows"))
    if not isinstance(samples_raw, (list, tuple)) or not samples_raw:
        raise ValueError("standalone MMVQ receipt requires rows")
    annotated_samples = []
    for index, row in enumerate(samples_raw):
        sample = _parse_sample(row, index, cache_protocol)
        geometry_ok, geometry_reason = _source_geometry_check(
            sample, runtime_binary_sha256=runtime_sha, source_hashes=source_hashes
        )
        annotated_samples.append(replace(
            sample, source_geometry_valid=geometry_ok,
            source_geometry_reason=geometry_reason,
        ))
    samples = tuple(annotated_samples)
    # The runner uses one case id for the M=1 MMVQ row and its M=512 MMQ
    # control.  Distinguish those diagnostic rows by family and exact shape.
    sample_keys = {(sample.case_id, sample.kernel_family, sample.shape)
                   for sample in samples}
    if len(sample_keys) != len(samples):
        raise ValueError("standalone MMVQ case/family/shape values must be unique")
    return StandaloneMMVQReceipt(
        hardware_id=hardware_id.strip(), runtime_binary_sha256=runtime_sha,
        gguf_sha256=gguf_sha, runtime_components=tuple(components),
        source_hashes=source_hashes, cache_protocol=cache_protocol,
        samples=samples, qualification=qualification,
        evidence_path=str(path),
    )


__all__ = [
    "CUDA_EVENT_BOUNDARY", "CUPTI_NODE_BOUNDARY", "STANDALONE_MMVQ_SCHEMA",
    "StandaloneMMVQReceipt", "StandaloneMMVQSample",
    "load_standalone_mmvq_receipt",
]
