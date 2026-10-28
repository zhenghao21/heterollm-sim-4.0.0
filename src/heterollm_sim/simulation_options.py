"""Explicit fidelity settings for detailed and sampled Decode execution."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Mapping


def _integer(value: Any, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))


def _bounded_fraction(value: Any, name: str, maximum: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 < value <= maximum or not math.isfinite(value)):
        raise ValueError("{} must be finite and in (0, {}]".format(name, maximum))


@dataclass(frozen=True)
class FastDecodeConfig:
    """Sampling budgets and acceptance targets, never an accuracy guarantee.

    Fixed sampling uses the three configured window lengths directly.
    Adaptive sampling caps warmup at four and starts from smaller windows,
    rounding training halves and holdout to complete HBF flush periods.
    Both strategies obey max_sample_tokens and the same safety guards.
    """

    warmup_tokens: int = 8
    calibration_tokens: int = 32
    validation_tokens: int = 16
    max_sample_tokens: int = 128
    target_relative_error: float = .10
    max_context_growth: float = .20
    segment_tokens: int = 256
    sampling_strategy: str = "adaptive"

    def __post_init__(self) -> None:
        if not isinstance(self.sampling_strategy, str) or self.sampling_strategy not in {"adaptive", "fixed"}:
            raise ValueError("simulation.fast_decode.sampling_strategy must be adaptive or fixed")
        _integer(self.warmup_tokens, "simulation.fast_decode.warmup_tokens", 0)
        _integer(self.calibration_tokens, "simulation.fast_decode.calibration_tokens", 2)
        _integer(self.validation_tokens, "simulation.fast_decode.validation_tokens", 2)
        _integer(self.max_sample_tokens, "simulation.fast_decode.max_sample_tokens", 1)
        _integer(self.segment_tokens, "simulation.fast_decode.segment_tokens", 1)
        minimum = (self.warmup_tokens + self.calibration_tokens + self.validation_tokens
                   if self.sampling_strategy == "fixed" else min(self.warmup_tokens, 4) + 4)
        if self.max_sample_tokens < minimum:
            raise ValueError("simulation.fast_decode.max_sample_tokens must cover warmup, calibration and validation tokens")
        _bounded_fraction(self.target_relative_error, "simulation.fast_decode.target_relative_error", .10)
        _bounded_fraction(self.max_context_growth, "simulation.fast_decode.max_context_growth", .20)


@dataclass(frozen=True)
class NativeWindowFastConfig:
    """Native stationary samples with an independently bounded window forecast.

    The first continuous-refill cohort boundary is sampled on independent
    copies: at least ``calibration_layers`` from the stack's start with the
    step prefix and ``holdout_layers`` from its end with the step suffix,
    extended until both runs step every layer class. Samples
    never claim a completed cohort or a visible output Token.

    ``tp_rank_symmetry`` selects complete execution (off), exact physical
    transition reuse outside PD windows (quiet), or an explicit experimental
    controller-state symmetry approximation inside PD windows too (all).
    Unsupported graphs/geometries and unequal dispatch arrivals run in full.

    ``pd_concurrency_envelope`` uses the approved monotone envelope between
    two measured PD concurrency endpoints (1 and k_max) by default. Set it
    to False to measure every required concurrency count for comparison.
    """

    max_wall_seconds: float = 3000.
    target_relative_error: float = .10
    # PP stage/transfer windows use the same declared conditional target as
    # ordinary fast Decode. Sampling quality, rather than a looser target,
    # determines whether a PP forecast is admitted.
    pipeline_target_relative_error: float = .10
    calibration_layers: int = 4
    holdout_layers: int = 2
    tp_rank_symmetry: str = "quiet"
    pd_concurrency_envelope: bool = True
    # Two samples only provide a very wide max-deviation interval. Four
    # cohorts keep the PP forecast conditional while making the interval
    # useful for layout screening; callers may lower this explicitly when
    # they accept the corresponding rejection rate.
    pipeline_sample_decode_cohorts: int = 4
    # Bound the request fan-out of a PP layer sample.  The forecast is still
    # conditioned on the full batch; only the native representative graph is
    # capped so a B=512 sample does not clone and schedule 512 request lanes.
    pipeline_sample_requests: int = 32
    # Cheap stability probe before the remaining stage samples are executed.
    # A probe that already exceeds the unchanged 10% budget stops early.
    pipeline_probe_decode_cohorts: int = 2
    pipeline_stage_sampling: bool = True

    def __post_init__(self) -> None:
        if (isinstance(self.max_wall_seconds, bool)
                or not isinstance(self.max_wall_seconds, (int, float))
                or not math.isfinite(self.max_wall_seconds)
                or not self.max_wall_seconds > 0):
            raise ValueError("simulation.native_window.max_wall_seconds must be finite and positive")
        _bounded_fraction(self.target_relative_error, "simulation.native_window.target_relative_error", .10)
        _bounded_fraction(self.pipeline_target_relative_error,
                          "simulation.native_window.pipeline_target_relative_error", .10)
        _integer(self.calibration_layers, "simulation.native_window.calibration_layers", 4)
        _integer(self.holdout_layers, "simulation.native_window.holdout_layers", 2)
        if type(self.tp_rank_symmetry) is not str or self.tp_rank_symmetry not in ("off", "quiet", "all"):
            raise ValueError("simulation.native_window.tp_rank_symmetry must be off, quiet or all")
        if type(self.pd_concurrency_envelope) is not bool:
            raise ValueError("simulation.native_window.pd_concurrency_envelope must be a boolean")
        _integer(self.pipeline_sample_decode_cohorts, "simulation.native_window.pipeline_sample_decode_cohorts", 2)
        _integer(self.pipeline_sample_requests, "simulation.native_window.pipeline_sample_requests", 1)
        _integer(self.pipeline_probe_decode_cohorts, "simulation.native_window.pipeline_probe_decode_cohorts", 1)
        if self.pipeline_probe_decode_cohorts > self.pipeline_sample_decode_cohorts:
            raise ValueError("simulation.native_window.pipeline_probe_decode_cohorts must not exceed pipeline_sample_decode_cohorts")
        if type(self.pipeline_stage_sampling) is not bool:
            raise ValueError("simulation.native_window.pipeline_stage_sampling must be a boolean")


@dataclass(frozen=True)
class NativePrefillPassConfig:
    """One request's Prefill forward pass, natively from its cohort boundary.

    ``full_pass`` runs the prefix and every layer in one native window;
    otherwise at least ``calibration_layers`` from the stack's start (with
    the prefix) and ``holdout_layers`` from its end, extended until both runs
    step every layer class, are run on independent copies of the boundary
    and the pass is composed within ``target_relative_error``. Only
    ``chunked_sampling`` explicitly enables the experimental per-chunk layer
    forecast, including its declared future-boundary-state approximation. Only
    a full pass may run without a wall limit (``max_wall_seconds`` None),
    like a detailed run. Two layers each way keep a 100K-token 405B pass
    (about 230 s of wall per layer) inside the default 1800 s.
    """

    max_wall_seconds: float | None = 1800.
    target_relative_error: float = .10
    calibration_layers: int = 2
    holdout_layers: int = 2
    full_pass: bool = False
    chunked_sampling: bool = False
    operator_reuse_sampling: bool = False
    chunk_boundary_probe: bool = False
    chunk_interval_model: str = "per_access"
    numeric_transition_cache: bool = False
    numeric_memo_max_bytes: int = 2 * 1024**3
    tp_rank_symmetry: str = "off"

    def __post_init__(self) -> None:
        if self.chunk_interval_model not in {"per_access", "history"}:
            raise ValueError("simulation.native_prefill.chunk_interval_model must be per_access or history")
        if self.chunk_interval_model == "history" and not self.chunk_boundary_probe:
            raise ValueError("simulation.native_prefill.history requires chunk_boundary_probe")
        if type(self.numeric_transition_cache) is not bool:
            raise ValueError("simulation.native_prefill.numeric_transition_cache must be a boolean")
        if (type(self.numeric_memo_max_bytes) is not int
                or not 0 < self.numeric_memo_max_bytes <= 2 * 1024**3):
            raise ValueError("simulation.native_prefill.numeric_memo_max_bytes must be positive and at most 2 GiB")
        if self.tp_rank_symmetry not in {"off", "quiet"}:
            raise ValueError("simulation.native_prefill.tp_rank_symmetry must be off or quiet")
        if type(self.chunk_boundary_probe) is not bool:
            raise ValueError("simulation.native_prefill.chunk_boundary_probe must be a boolean")
        if type(self.chunked_sampling) is not bool:
            raise ValueError("simulation.native_prefill.chunked_sampling must be a boolean")
        if type(self.operator_reuse_sampling) is not bool:
            raise ValueError("simulation.native_prefill.operator_reuse_sampling must be a boolean")
        if self.operator_reuse_sampling and not self.chunked_sampling:
            raise ValueError("simulation.native_prefill.operator_reuse_sampling requires chunked_sampling")
        if self.full_pass and self.chunked_sampling:
            raise ValueError("simulation.native_prefill.chunked_sampling cannot combine with full_pass")
        if type(self.full_pass) is not bool:
            raise ValueError("simulation.native_prefill.full_pass must be a boolean")
        if self.max_wall_seconds is None:
            if not self.full_pass:
                raise ValueError("simulation.native_prefill.max_wall_seconds may be null only for a full pass")
        elif (isinstance(self.max_wall_seconds, bool)
                or not isinstance(self.max_wall_seconds, (int, float))
                or not math.isfinite(self.max_wall_seconds)
                or not self.max_wall_seconds > 0):
            raise ValueError("simulation.native_prefill.max_wall_seconds must be null or finite and positive")
        _bounded_fraction(self.target_relative_error, "simulation.native_prefill.target_relative_error", .10)
        _integer(self.calibration_layers, "simulation.native_prefill.calibration_layers", 2)
        _integer(self.holdout_layers, "simulation.native_prefill.holdout_layers", 2)


@dataclass(frozen=True)
class SimulationConfig:
    """Select the execution fidelity independently of result retention."""

    mode: str = "detailed"
    fast_decode: FastDecodeConfig = field(default_factory=FastDecodeConfig)
    native_window: NativeWindowFastConfig | None = None
    native_prefill: NativePrefillPassConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"detailed", "fast"}:
            raise ValueError("simulation.mode must be detailed or fast")
        if not isinstance(self.fast_decode, FastDecodeConfig):
            raise ValueError("simulation.fast_decode must be a FastDecodeConfig")
        if self.native_window is not None:
            if not isinstance(self.native_window, NativeWindowFastConfig):
                raise ValueError("simulation.native_window must be a NativeWindowFastConfig")
            if self.mode != "fast":
                raise ValueError("simulation.native_window requires fast mode")
        if self.native_prefill is not None:
            if not isinstance(self.native_prefill, NativePrefillPassConfig):
                raise ValueError("simulation.native_prefill must be a NativePrefillPassConfig")
            if self.mode != "fast" or self.native_window is not None:
                raise ValueError("simulation.native_prefill requires fast mode without native_window")


def simulation_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    """Parse the strict authoring schema without dropping misspelled options."""

    if not isinstance(data, Mapping):
        raise ValueError("simulation must be an object")
    unknown = set(data) - {"mode", "fast_decode", "native_window", "native_prefill"}
    if unknown:
        raise ValueError("unknown simulation fields: {}".format(", ".join(sorted(map(str, unknown)))))
    raw_fast = data.get("fast_decode", {})
    if not isinstance(raw_fast, Mapping):
        raise ValueError("simulation.fast_decode must be an object")
    allowed = {item.name for item in fields(FastDecodeConfig)}
    unknown_fast = set(raw_fast) - allowed
    if unknown_fast:
        raise ValueError("unknown simulation.fast_decode fields: {}".format(", ".join(sorted(map(str, unknown_fast)))))
    native_window = data.get("native_window")
    if native_window is not None:
        if not isinstance(native_window, Mapping):
            raise ValueError("simulation.native_window must be an object or null")
        allowed_native = {item.name for item in fields(NativeWindowFastConfig)}
        unknown_native = set(native_window) - allowed_native
        if unknown_native:
            raise ValueError("unknown simulation.native_window fields: {}".format(", ".join(sorted(map(str, unknown_native)))))
        native_window = NativeWindowFastConfig(**native_window)
    native_prefill = data.get("native_prefill")
    if native_prefill is not None:
        if not isinstance(native_prefill, Mapping):
            raise ValueError("simulation.native_prefill must be an object or null")
        unknown_prefill = set(native_prefill) - {item.name for item in fields(NativePrefillPassConfig)}
        if unknown_prefill:
            raise ValueError("unknown simulation.native_prefill fields: {}".format(", ".join(sorted(map(str, unknown_prefill)))))
        native_prefill = NativePrefillPassConfig(**native_prefill)
    return SimulationConfig(mode=data.get("mode", "detailed"), fast_decode=FastDecodeConfig(**raw_fast),
                            native_window=native_window, native_prefill=native_prefill)


__all__ = ["FastDecodeConfig", "NativePrefillPassConfig", "NativeWindowFastConfig", "SimulationConfig",
           "simulation_from_dict"]
