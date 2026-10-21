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
    """Native physical samples with an independently bounded window forecast.

    ``cohorts`` executes complete native batches. ``layer_windows`` executes
    explicitly bounded layer groups and never claims a completed cohort or
    visible output Token merely because those layer tasks finished.
    """

    sampling_unit: str = "cohorts"
    warmup_cohorts: int = 1
    calibration_cohorts: int = 2
    validation_cohorts: int = 2
    max_sample_cohorts: int = 8
    max_wall_seconds: float = 1800.
    target_relative_error: float = .10
    layer_window_size: int = 1
    # None keeps a fixed physical window. An explicit maximum permits
    # bounded doubling only after an eligible conditional-budget refusal.
    max_layer_window_size: int | None = None
    calibration_layer_windows: int = 2
    validation_layer_windows: int = 2
    # The full-batch cap spans all native layer classes. Calibration and
    # validation counts apply independently to each structural class.
    max_sample_layer_windows: int = 16
    # A multislot seed-shape anchor has its own disjoint calibration and
    # holdout set. It does not borrow the full-batch sample cap or Tokens.
    max_seed_anchor_layer_windows: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.sampling_unit, str) or self.sampling_unit not in {"cohorts", "layer_windows"}:
            raise ValueError("simulation.native_window.sampling_unit must be cohorts or layer_windows")
        for name, minimum in (("warmup_cohorts", 0), ("calibration_cohorts", 2),
                              ("validation_cohorts", 2), ("max_sample_cohorts", 4)):
            _integer(getattr(self, name), "simulation.native_window." + name, minimum)
        if self.max_sample_cohorts < self.sample_cohorts:
            raise ValueError("simulation.native_window.max_sample_cohorts must cover all sample windows")
        if (isinstance(self.max_wall_seconds, bool)
                or not isinstance(self.max_wall_seconds, (int, float))
                or not math.isfinite(self.max_wall_seconds)
                or not 0 < self.max_wall_seconds <= 1800.):
            raise ValueError("simulation.native_window.max_wall_seconds must be finite and in (0, 1800]")
        _bounded_fraction(self.target_relative_error, "simulation.native_window.target_relative_error", .10)
        for name, minimum in (("layer_window_size", 1), ("calibration_layer_windows", 2),
                              ("validation_layer_windows", 2), ("max_sample_layer_windows", 4),
                              ("max_seed_anchor_layer_windows", 4)):
            _integer(getattr(self, name), "simulation.native_window." + name, minimum)
        if self.max_layer_window_size is not None:
            _integer(self.max_layer_window_size, "simulation.native_window.max_layer_window_size", 1)
            if self.max_layer_window_size < self.layer_window_size:
                raise ValueError("simulation.native_window.max_layer_window_size must cover layer_window_size")
        if self.max_sample_layer_windows < self.sample_layer_windows:
            raise ValueError("simulation.native_window.max_sample_layer_windows must cover calibration and disjoint validation windows")
        if self.max_seed_anchor_layer_windows < self.sample_layer_windows:
            raise ValueError("simulation.native_window.max_seed_anchor_layer_windows must cover calibration and disjoint validation windows")

    @property
    def sample_cohorts(self) -> int:
        return self.warmup_cohorts + self.calibration_cohorts + self.validation_cohorts

    @property
    def sample_layer_windows(self) -> int:
        return self.calibration_layer_windows + self.validation_layer_windows


@dataclass(frozen=True)
class SimulationConfig:
    """Select the execution fidelity independently of result retention."""

    mode: str = "detailed"
    fast_decode: FastDecodeConfig = field(default_factory=FastDecodeConfig)
    native_window: NativeWindowFastConfig | None = None

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


def simulation_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    """Parse the strict authoring schema without dropping misspelled options."""

    if not isinstance(data, Mapping):
        raise ValueError("simulation must be an object")
    unknown = set(data) - {"mode", "fast_decode", "native_window"}
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
    return SimulationConfig(mode=data.get("mode", "detailed"), fast_decode=FastDecodeConfig(**raw_fast),
                            native_window=native_window)


__all__ = ["FastDecodeConfig", "NativeWindowFastConfig", "SimulationConfig", "simulation_from_dict"]
