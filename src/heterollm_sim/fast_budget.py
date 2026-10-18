"""Directional conditional timing bounds for independently sampled stores.

These intervals encode the fast mode's declared model assumptions.  Passing
them does not establish error against an unexecuted detailed suffix.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math

from .precision import canonical_dtype


def _target(value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= .10):
        raise ValueError("paired timing target must be finite and in (0, 0.10]")


def timing_interval(estimate: float, lower: float, upper: float) -> dict:
    """Return errors relative to the possible true timing, not the estimate."""
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
            for value in (estimate, lower, upper))
            or not lower <= estimate <= upper):
        raise ValueError("timing interval requires positive finite lower <= estimate <= upper")
    over, under = estimate / lower - 1., 1. - estimate / upper
    if not all(math.isfinite(value) for value in (over, under)):
        raise ValueError("timing interval relative error must be finite")
    return {"estimate_ns": estimate, "lower_ns": lower, "upper_ns": upper,
            "overestimate_relative_error": max(0., over),
            "underestimate_relative_error": max(0., under),
            "relative_error_budget": max(0., over, under)}


def _within(interval: dict, target: float) -> bool:
    # Exact products of the authored floating values distinguish the target
    # from its next representable overrun. Rounded (1+target)*lower can
    # otherwise swallow an above-target ULP. This adds no error margin.
    estimate, lower, upper = (Fraction(interval[name])
        for name in ("estimate_ns", "lower_ns", "upper_ns"))
    threshold = Fraction(target)
    return estimate <= (1 + threshold) * lower and estimate >= (1 - threshold) * upper


def minimax_decode_estimate(lower: float, upper: float, *,
                            candidate_memory_kind: str | None,
                            reference: PairedTimingReference | None = None) -> float:
    """Choose a point inside fixed bounds; never narrow or regenerate them.

    Relative errors use possible true timings as denominators. With no
    reference their minimax point is the harmonic midpoint. For an HBF
    numerator the paired errors have the same form at enlarged effective
    endpoints. For an HBM denominator, both increasing/decreasing envelopes
    contain a reciprocal term. Their minimum is at one of four analytic
    intersections (or a constrained endpoint), not a relaxed error target.
    """
    timing_interval(lower, lower, upper)
    def harmonic(lo, hi):
        # Algebraically 2*lo*hi/(lo+hi), without either large product or
        # sum. Adding the correction also preserves exact simple boundaries.
        return lo + ((hi - lo) / hi) * (lo / (1. + lo / hi))
    if reference is None:
        return harmonic(lower, upper)
    if (not isinstance(reference, PairedTimingReference)
            or candidate_memory_kind not in {"hbm", "hbf"}
            or candidate_memory_kind == reference.memory_kind):
        raise ValueError("minimax reference must describe the other HBM/HBF store")
    effective_lower = lower * (reference.decode_estimate_ns / reference.decode_upper_ns)
    effective_upper = upper * (reference.decode_estimate_ns / reference.decode_lower_ns)
    if not all(math.isfinite(value) and value > 0
               for value in (effective_lower, effective_upper)):
        raise ValueError("minimax effective endpoints must be positive and finite")
    if candidate_memory_kind == "hbf":
        return min(upper, max(lower, harmonic(effective_lower, effective_upper)))
    # max(single over, paired under) increases with E, while
    # max(single under, paired over) decreases. Enumerate their crossings.
    candidates = (lower, upper, harmonic(lower, upper),
        math.sqrt(lower) * math.sqrt(effective_upper),
        math.sqrt(effective_lower) * math.sqrt(upper),
        effective_lower + (effective_upper - effective_lower) / 2.)
    def score(estimate):
        return max(estimate / lower - 1., 1. - estimate / upper,
                   effective_upper / estimate - 1., 1. - effective_lower / estimate)
    return min((min(upper, max(lower, value)) for value in candidates), key=score)


@dataclass(frozen=True)
class PairedTimingReference:
    """One matched request's whole-Decode estimate and conditional bounds."""

    model_name: str
    prompt_tokens: int
    output_tokens: int
    memory_kind: str
    target_relative_error: float
    decode_estimate_ns: float
    decode_lower_ns: float
    decode_upper_ns: float
    source_id: str
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("paired reference requires a nonempty model_name")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("paired reference requires a nonempty source_id")
        for name, minimum in (("prompt_tokens", 1), ("output_tokens", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"paired reference {name} must be an integer >= {minimum}")
        if not isinstance(self.memory_kind, str) or self.memory_kind not in {"hbm", "hbf"}:
            raise ValueError("paired reference memory_kind must be hbm or hbf")
        if any(not isinstance(dtype, str) or canonical_dtype(dtype) != "bf16"
               for dtype in (self.weight_dtype, self.kv_dtype)):
            raise ValueError("paired reference requires BF16 weights and KV")
        _target(self.target_relative_error)
        interval = timing_interval(self.decode_estimate_ns, self.decode_lower_ns, self.decode_upper_ns)
        if not _within(interval, self.target_relative_error):
            raise ValueError("paired reference itself exceeds the single-point error target")


def validate_paired_reference(reference: PairedTimingReference, *, model_name: str,
                              prompt_tokens: int, output_tokens: int,
                              memory_kind: str, target_relative_error: float) -> None:
    """Reject mismatches explicitly; an invalid reference is never ignored."""
    if not isinstance(reference, PairedTimingReference):
        raise ValueError("paired_reference must be a PairedTimingReference")
    _target(target_relative_error)
    for name, value in (("model_name", model_name), ("prompt_tokens", prompt_tokens),
                        ("output_tokens", output_tokens),
                        ("target_relative_error", target_relative_error)):
        if getattr(reference, name) != value:
            raise ValueError(f"paired reference {name} does not match this request")
    if not isinstance(memory_kind, str) or memory_kind not in {"hbm", "hbf"} or memory_kind == reference.memory_kind:
        raise ValueError("paired reference must describe the other HBM/HBF storage kind")


def paired_timing_decision(*, decode_estimate_ns: float,
                           decode_lower_ns: float | None, decode_upper_ns: float | None,
                           target_relative_error: float,
                           candidate_memory_kind: str | None,
                           reference: PairedTimingReference | None = None) -> dict:
    """Require both single-point and exact directional ratio interval bounds.

    Without the other store, retain the symmetric target/(2+target)
    sufficient condition.  A matched reference instead uses its actual
    asymmetric interval; no global error target is enlarged.
    """
    _target(target_relative_error)
    if reference is not None:
        if not isinstance(reference, PairedTimingReference):
            raise ValueError("paired_reference must be a PairedTimingReference")
        if reference.target_relative_error != target_relative_error:
            raise ValueError("paired reference target_relative_error does not match this request")
        if (not isinstance(candidate_memory_kind, str) or candidate_memory_kind not in {"hbm", "hbf"}
                or candidate_memory_kind == reference.memory_kind):
            raise ValueError("paired reference must describe the other HBM/HBF storage kind")
    result = {
        "schema": "directional_decode_interval_pair/v1",
        "semantics": "conditional_model_bounds_not_measured_error",
        "strategy": "matched_directional_pair" if reference else "independent_symmetric_sufficient_bound",
        "target_relative_error": target_relative_error,
        "independent_symmetric_target_relative_error": target_relative_error / (2. + target_relative_error),
        "reference_source_id": reference.source_id if reference else None,
        "candidate_memory_kind": candidate_memory_kind,
        "reference_memory_kind": reference.memory_kind if reference else None,
        "reference": asdict(reference) if reference else None,
        "candidate_decode_interval": None, "reference_decode_interval": None,
        "candidate_relative_error_budget": None, "reference_relative_error_budget": None,
        "pair_ratio_direction": "hbf/hbm" if reference else None,
        "pair_ratio_interval": None, "pair_relative_error_budget": None,
        "candidate_within_target": False, "reference_within_target": None,
        "pair_within_target": None, "accepted": False, "reasons": [],
    }
    if reference:
        ref_interval = timing_interval(reference.decode_estimate_ns,
            reference.decode_lower_ns, reference.decode_upper_ns)
        result.update(reference_decode_interval=ref_interval,
                      reference_relative_error_budget=ref_interval["relative_error_budget"],
                      reference_within_target=_within(ref_interval, target_relative_error))
    try:
        candidate = timing_interval(decode_estimate_ns, decode_lower_ns, decode_upper_ns)
    except ValueError:
        result["reasons"] = ["candidate_conditional_decode_interval_unavailable"]
        return result
    result.update(candidate_decode_interval=candidate,
                  candidate_relative_error_budget=candidate["relative_error_budget"],
                  candidate_within_target=_within(candidate, target_relative_error))
    if not result["candidate_within_target"]:
        result["reasons"].append("candidate_single_point_conditional_error_exceeds_target")
    if reference is None:
        independent = _within(candidate, result["independent_symmetric_target_relative_error"])
        if not independent:
            result["reasons"].append("independent_symmetric_ratio_sufficient_bound_exceeded")
        result["accepted"] = result["candidate_within_target"] and independent
        return result
    # The requested comparison is HBF/HBM even when HBM is the candidate.
    # Reciprocal relative error is asymmetric, so accepting HBM/HBF would
    # not by itself establish the target on the requested HBF/HBM ratio.
    numerator, denominator = ((candidate, ref_interval) if candidate_memory_kind == "hbf"
                              else (ref_interval, candidate))
    ratio = timing_interval(numerator["estimate_ns"] / denominator["estimate_ns"],
        numerator["lower_ns"] / denominator["upper_ns"],
        numerator["upper_ns"] / denominator["lower_ns"])
    # The original timing products avoid both ratio-division roundoff and
    # swallowed nextafter overruns. Fractions are used only at this tiny
    # acceptance gate, never on the physical simulation path.
    ne, nl, nu = (Fraction(numerator[name])
        for name in ("estimate_ns", "lower_ns", "upper_ns"))
    de, dl, du = (Fraction(denominator[name])
        for name in ("estimate_ns", "lower_ns", "upper_ns"))
    threshold = Fraction(target_relative_error)
    pair_within = (
        ne * du <= (1 + threshold) * nl * de
        and ne * dl >= (1 - threshold) * nu * de)
    result.update(pair_ratio_interval={key.removesuffix("_ns"): value for key, value in ratio.items()},
                  pair_relative_error_budget=ratio["relative_error_budget"],
                  pair_within_target=pair_within)
    if not result["pair_within_target"]:
        result["reasons"].append("directional_pair_conditional_error_exceeds_target")
    result["accepted"] = bool(result["candidate_within_target"]
        and result["reference_within_target"] and result["pair_within_target"])
    return result


__all__ = ["PairedTimingReference", "minimax_decode_estimate", "paired_timing_decision",
           "timing_interval", "validate_paired_reference"]
