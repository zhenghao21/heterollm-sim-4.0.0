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
class SimulationConfig:
    """Select the execution fidelity independently of result retention."""

    mode: str = "detailed"
    fast_decode: FastDecodeConfig = field(default_factory=FastDecodeConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"detailed", "fast"}:
            raise ValueError("simulation.mode must be detailed or fast")
        if not isinstance(self.fast_decode, FastDecodeConfig):
            raise ValueError("simulation.fast_decode must be a FastDecodeConfig")


def simulation_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    """Parse the strict authoring schema without dropping misspelled options."""

    if not isinstance(data, Mapping):
        raise ValueError("simulation must be an object")
    unknown = set(data) - {"mode", "fast_decode"}
    if unknown:
        raise ValueError("unknown simulation fields: {}".format(", ".join(sorted(map(str, unknown)))))
    raw_fast = data.get("fast_decode", {})
    if not isinstance(raw_fast, Mapping):
        raise ValueError("simulation.fast_decode must be an object")
    allowed = {item.name for item in fields(FastDecodeConfig)}
    unknown_fast = set(raw_fast) - allowed
    if unknown_fast:
        raise ValueError("unknown simulation.fast_decode fields: {}".format(", ".join(sorted(map(str, unknown_fast)))))
    return SimulationConfig(mode=data.get("mode", "detailed"), fast_decode=FastDecodeConfig(**raw_fast))


__all__ = ["FastDecodeConfig", "SimulationConfig", "simulation_from_dict"]
