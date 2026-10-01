"""Independent Level-2 contracts for the non-main MMQ kernels.

The checked-in main MMQ surface deliberately measures only ``mul_mat_q``.
This module keeps activation repacking and stream-K fixup as separate owners.
It is intentionally fail-closed: a source/runtime binding without repeated
device intervals is useful provenance, but it is never a timing correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import re
from typing import Mapping, Sequence


MMQ_STAGE_SCHEMA = "heterollm.level2.mmq.stage/v1"
MMQ_STAGE_NAMES = frozenset({"activation_repack", "stream_k_fixup"})
MMQ_REQUIRED_PREFILL_M = (64, 128, 256, 512, 1024)
MMQ_HOLDOUT_APE_LIMIT = 0.10
_TYPE_IDS = {"q4_k": 12, "q5_k": 13, "q6_k": 14, "iq4_xs": 23,
             "iq3_s": 21, "q4_0": 2, "q5_0": 6, "q8_0": 8}


def _shape(value: Sequence[int], name: str = "shape") -> tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must be an M,N,K triple")
    result = tuple(value)
    if any(type(item) is not int or item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def mmq_stage_dispatch_signature(stage: str, source) -> str:
    """Return the source specialization key for a non-main MMQ phase."""
    if stage not in MMQ_STAGE_NAMES:
        raise ValueError("unsupported MMQ stage")
    fmt = str(source.weight_format).casefold()
    type_id = _TYPE_IDS.get(fmt)
    if type_id is None:
        raise ValueError(f"unsupported MMQ format {fmt}")
    if stage == "activation_repack":
        return ("mmq_repack:type={}:layout={}:channels=1:ids=0:scatter=0".format(
            type_id, source.conversion_layout.casefold()))
    return "mmq_fixup:type={}:j={}:fallback={}:stream_k={}".format(
        type_id, source.j, int(source.n % 128 != 0), int(source.fixup_launch))


def mmq_stage_source_binding(stage: str, source, *, source_hashes: Mapping[str, str] | None = None) -> dict[str, object]:
    """Describe identity checks shared by planner metadata and surface import."""
    if stage not in MMQ_STAGE_NAMES:
        raise ValueError("unsupported MMQ stage")
    runtime = getattr(source, "runtime_binary_sha256", "")
    binding = {
        "schema": "heterollm.mmq-stage-source-binding/v1",
        "stage": stage,
        "weight_format": str(source.weight_format).casefold(),
        "runtime_binary_sha256": runtime,
        "dispatch_signature": mmq_stage_dispatch_signature(stage, source),
        "source_hashes": dict(source_hashes or {}),
        "source_geometry_bound": bool(runtime),
        "timing_surface_bound": False,
        "fail_closed_reason": "no_independent_stage_holdout_surface",
    }
    return binding


@dataclass(frozen=True)
class MMQStageSample:
    stage: str
    weight_format: str
    m: int
    n: int
    k: int
    device_ns: float
    analytical_ns: float
    stddev_ns: float
    sample_count: int
    evidence: str
    dispatch_signature: str
    runtime_binary_sha256: str
    resources: tuple[tuple[str, object], ...] = ()
    split: str = "train"

    def __post_init__(self):
        if self.stage not in MMQ_STAGE_NAMES:
            raise ValueError("unsupported MMQ stage sample")
        if str(self.weight_format) != self.weight_format.casefold() or not self.weight_format:
            raise ValueError("weight_format must be canonical lower-case")
        _shape((self.m, self.n, self.k))
        for key in ("device_ns", "analytical_ns"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.stddev_ns < 0 or not math.isfinite(self.stddev_ns):
            raise ValueError("stddev_ns must be finite and nonnegative")
        if type(self.sample_count) is not int or self.sample_count < 2:
            raise ValueError("independent stage evidence needs at least two samples")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("stage sample evidence is required")
        if not isinstance(self.dispatch_signature, str) or not self.dispatch_signature:
            raise ValueError("stage dispatch signature is required")
        _sha(self.runtime_binary_sha256, "runtime_binary_sha256")
        if self.split not in {"train", "holdout"}:
            raise ValueError("stage sample split must be train or holdout")
        if not isinstance(self.resources, tuple):
            raise ValueError("resources must be immutable")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.m, self.n, self.k


@dataclass(frozen=True)
class MMQStageSurface:
    stage: str
    weight_format: str
    runtime_binary_sha256: str
    dispatch_signature: str
    samples: tuple[MMQStageSample, ...] = ()
    calibration_cells: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = ()
    validation_relative_error: float | None = None
    accepted: bool = False
    rejection_reason: str = "no_independent_stage_holdout_surface"

    def __post_init__(self):
        if self.stage not in MMQ_STAGE_NAMES:
            raise ValueError("unsupported MMQ stage")
        if self.weight_format != self.weight_format.casefold() or not self.weight_format:
            raise ValueError("weight_format must be canonical lower-case")
        _sha(self.runtime_binary_sha256, "runtime_binary_sha256")
        if not self.dispatch_signature:
            raise ValueError("dispatch_signature is required")
        if any(sample.stage != self.stage or sample.weight_format != self.weight_format
               or sample.runtime_binary_sha256 != self.runtime_binary_sha256
               or sample.dispatch_signature != self.dispatch_signature for sample in self.samples):
            raise ValueError("stage samples do not share identity")
        if self.accepted and (not self.samples or self.validation_relative_error is None):
            raise ValueError("accepted stage surface requires samples and holdout error")
        if self.validation_relative_error is not None and self.validation_relative_error < 0:
            raise ValueError("validation_relative_error must be nonnegative")

    def predict(self, shape: Sequence[int], *, runtime_binary_sha256: str,
                dispatch_signature: str, analytical_ns: float | None = None) -> dict[str, object]:
        shape = _shape(shape)
        base = {"model": "analytical", "accepted": False, "extrapolated": False,
                "distance_to_calibration_domain": None, "uncertainty_kind": "unvalidated"}
        if runtime_binary_sha256 != self.runtime_binary_sha256:
            return {**base, "reason": "stage_runtime_binding_mismatch"}
        if dispatch_signature != self.dispatch_signature:
            return {**base, "reason": "stage_dispatch_signature_mismatch"}
        if not self.accepted:
            return {**base, "reason": self.rejection_reason}
        rows = {sample.shape: sample for sample in self.samples}
        if analytical_ns is None:
            if shape not in rows:
                return {**base, "reason": "stage_analytical_cost_missing"}
            analytical_ns = rows[shape].analytical_ns
        from .kernel_model import performance_surface
        predicted, _, prediction = performance_surface(
            _capability(self.samples, self.calibration_cells, self.validation_relative_error),
            shape, analytical_ns)
        return {**prediction, "accepted": prediction["model"] == "calibrated_analytical",
                "prediction_ns": predicted}


def stage_complete_cells(samples: Sequence[MMQStageSample]):
    """Adjacent, fully measured N/K cells for each calibrated M.

    MMQ source templates change at M boundaries (J/CTA/stream-K).  A cell is
    therefore bounded at a fixed M; interpolation is never allowed to cross
    one of those specializations.
    """
    points = {sample.shape for sample in samples}
    cells = []
    for m in sorted({point[0] for point in points}):
        n_axis = sorted({point[1] for point in points if point[0] == m})
        k_axis = sorted({point[2] for point in points if point[0] == m})
        n_spans = list(zip(n_axis, n_axis[1:])) or [(n_axis[0], n_axis[0])] if n_axis else []
        k_spans = list(zip(k_axis, k_axis[1:])) or [(k_axis[0], k_axis[0])] if k_axis else []
        for (n_low, n_high), (k_low, k_high) in product(n_spans, k_spans):
            corners = ((m, n, k) for n, k in product((n_low, n_high), (k_low, k_high)))
            if all(point in points for point in corners):
                cells.append(((m, n_low, k_low), (m, n_high, k_high)))
    return tuple(cells)


def _capability(samples, cells, validation_relative_error=None):
    """Reuse the existing Level-2 ratio interpolator for stage-only kernels."""
    from .kernel_model import KernelCapability, KernelSample
    sample = samples[0]
    return KernelCapability(
        kernel_family=sample.stage, weight_formats=(sample.weight_format,),
        activation_dtype="fp32", phase="prefill", compute_primitive="simt",
        evidence="independent_MMQ_stage_device_interval",
        output_bits=32, min_shape=tuple(min(row.shape[i] for row in samples) for i in range(3)),
        max_shape=tuple(max(row.shape[i] for row in samples) for i in range(3)),
        samples=tuple(KernelSample(*row.shape, row.device_ns, row.analytical_ns, row.stddev_ns,
                                  row.sample_count, row.evidence, None, row.dispatch_signature)
                      for row in samples),
        dispatch_signature=sample.dispatch_signature, calibration_cells=cells,
        validation_relative_error=validation_relative_error,
    )


def evaluate_mmq_stage_holdout(*, stage: str, weight_format: str,
                               training: Sequence[MMQStageSample],
                               holdout: Sequence[MMQStageSample],
                               runtime_binary_sha256: str,
                               dispatch_signature: str,
                               ape_limit: float = MMQ_HOLDOUT_APE_LIMIT) -> dict[str, object]:
    """Evaluate a complete joint N/K cell and return an install decision."""
    _sha(runtime_binary_sha256, "runtime_binary_sha256")
    if stage not in MMQ_STAGE_NAMES or not isinstance(training, (tuple, list)):
        raise ValueError("invalid stage holdout input")
    # A noisy training cell is evidence of a specialization that cannot be
    # installed, but must not poison an otherwise valid narrow surface.  Keep
    # it in the artifact ledger and remove it from the accepted interpolation
    # grid.  Holdout repeatability failures remain a hard rejection.
    rejected_training = []
    clean_training = []
    for sample in training:
        cv = sample.stddev_ns / sample.device_ns
        if cv >= 0.10:
            rejected_training.append(sample.shape)
        else:
            clean_training.append(sample)
    rows = [*clean_training, *holdout]
    reasons: list[str] = []
    for sample in rows:
        if (sample.stage != stage or sample.weight_format != weight_format.casefold()
                or sample.runtime_binary_sha256 != runtime_binary_sha256
                or sample.dispatch_signature != dispatch_signature):
            reasons.append("identity_mismatch")
        cv = sample.stddev_ns / sample.device_ns
        if sample.split == "holdout" and cv >= 0.10:
            reasons.append("repeatability_cv_ge_10_percent")
    resource_keys = {"block", "registers_per_thread", "shared_memory_per_block",
                     "dynamic_shared_memory", "static_shared_memory"}
    def freeze(value):
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((str(key), freeze(item)) for key, item in value.items()))
        return value
    resource_bindings = {
        tuple((key, freeze(value)) for key, value in sample.resources if key in resource_keys)
        for sample in rows if sample.resources
    }
    if len(resource_bindings) > 1:
        reasons.append("stage_resource_binding_changed_within_specialization")
    errors = []
    domain_rejections = []
    from .kernel_model import performance_surface
    cells = stage_complete_cells(clean_training) if clean_training else ()
    capability = _capability(clean_training, cells) if clean_training else None
    for sample in holdout:
        if sample.m not in {row.m for row in clean_training}:
            domain_rejections.append(sample.m)
            continue
        if capability is None or not cells:
            domain_rejections.append(sample.m)
            continue
        prediction, _, audit = performance_surface(capability, sample.shape, sample.analytical_ns)
        if audit["model"] != "calibrated_analytical":
            domain_rejections.append(sample.m)
            continue
        errors.append(abs(prediction - sample.device_ns) / sample.device_ns)
    max_ape = max(errors, default=None)
    accepted = bool(holdout) and not reasons and max_ape is not None and max_ape < ape_limit
    if not accepted and not reasons:
        reasons.append("holdout_error_gate_failed")
    return {"schema": "heterollm.level2.mmq.stage.holdout/v1", "stage": stage,
            "weight_format": weight_format.casefold(), "runtime_binary_sha256": runtime_binary_sha256,
            "dispatch_signature": dispatch_signature, "holdout_count": len(holdout),
            "max_ape": max_ape, "ape_limit": ape_limit, "accepted": accepted,
            "reasons": tuple(dict.fromkeys(reasons)),
            "domain_rejected_m": tuple(sorted(set(domain_rejections))),
            "accepted_m": tuple(sorted({sample.m for sample in holdout} - set(domain_rejections))),
            "rejected_training_count": len(rejected_training),
            "rejected_training_shapes": tuple(rejected_training),
            "required_prefill_m": MMQ_REQUIRED_PREFILL_M,
            "measured_prefill_m": tuple(sorted({row.m for row in rows}))}


def uncalibrated_mmq_stage_surface(stage: str, weight_format: str,
                                   *, runtime_binary_sha256: str,
                                   dispatch_signature: str,
                                   reason: str = "no_independent_stage_holdout_surface") -> MMQStageSurface:
    """Construct the explicit production fallback for an unmeasured stage."""
    return MMQStageSurface(stage=stage, weight_format=weight_format.casefold(),
                           runtime_binary_sha256=_sha(runtime_binary_sha256, "runtime_binary_sha256"),
                           dispatch_signature=dispatch_signature, rejection_reason=reason)


__all__ = [
    "MMQ_HOLDOUT_APE_LIMIT", "MMQ_REQUIRED_PREFILL_M", "MMQ_STAGE_NAMES",
    "MMQ_STAGE_SCHEMA", "MMQStageSample", "MMQStageSurface",
    "evaluate_mmq_stage_holdout", "mmq_stage_dispatch_signature",
    "mmq_stage_source_binding", "stage_complete_cells", "uncalibrated_mmq_stage_surface",
]
