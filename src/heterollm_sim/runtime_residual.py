"""Independent CUDA runtime residual calibration.

The kernel model accounts for device work.  This module accounts only for the
runtime submission envelope: one ordinary launch per kernel, or one graph
replay per captured graph.  It deliberately has no model/LLM timing input and
refuses records that cannot separate the two paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping


def _finite(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0
            or (not zero and value == 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


@dataclass(frozen=True)
class RuntimeResidualCalibration:
    """A source/runtime-bound residual with explicit ordinary/graph paths.

    ``ordinary_launch_ns`` is charged once per kernel node.  ``graph_replay_ns``
    is charged once per replay, independent of the number of nodes captured in
    that replay.  The latter is valid only when ``graph_enabled`` is bound by
    the runtime adapter; this record does not infer capture from a flag.
    """

    hardware_id: str
    runtime_id: str
    architecture: str
    ordinary_launch_ns: float
    graph_replay_ns: float
    evidence: str
    source_kind: str = "independent_synthetic_runtime_microbenchmark"
    protocol: str = "heterollm.runtime-residual/v1"
    validation_relative_error: float | None = None
    qualified: bool = False

    def __post_init__(self):
        for key in ("hardware_id", "runtime_id", "architecture", "evidence", "source_kind", "protocol"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be nonempty text")
        _finite(self.ordinary_launch_ns, "ordinary_launch_ns", zero=True)
        _finite(self.graph_replay_ns, "graph_replay_ns", zero=True)
        if self.validation_relative_error is not None:
            _finite(self.validation_relative_error, "validation_relative_error", zero=True)
        if self.source_kind != "independent_synthetic_runtime_microbenchmark":
            raise ValueError("runtime residual calibration must be independent synthetic runtime evidence")
        if self.protocol != "heterollm.runtime-residual/v1":
            raise ValueError("unsupported runtime residual protocol")
        if type(self.qualified) is not bool:
            raise ValueError("qualified must be boolean")

    def cost(self, kernel_count: int, *, captured: bool, replay_count: int = 1) -> dict:
        if type(kernel_count) is not int or kernel_count < 1:
            raise ValueError("kernel_count must be a positive integer")
        if type(replay_count) is not int or replay_count < 1:
            raise ValueError("replay_count must be a positive integer")
        if type(captured) is not bool:
            raise ValueError("captured must be boolean")
        if captured:
            service = replay_count * self.graph_replay_ns
            submissions = replay_count
            mode = "cuda_graph_replay"
        else:
            service = kernel_count * self.ordinary_launch_ns
            submissions = kernel_count
            mode = "ordinary_launch"
        return {
            "mode": mode,
            "kernel_count": kernel_count,
            "submission_count": submissions,
            "service_ns": service,
            "ordinary_launch_ns": self.ordinary_launch_ns,
            "graph_replay_ns": self.graph_replay_ns,
            "qualified": self.qualified,
            "validation_relative_error": self.validation_relative_error,
            "evidence": self.evidence,
            "source_kind": self.source_kind,
        }


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate runtime residual field: " + key)
        result[key] = value
    return result


def load_runtime_residual_calibration(path: str | Path, *, expected_sha256: str | None = None,
                                      hardware_id: str | None = None,
                                      runtime_id: str | None = None,
                                      architecture: str | None = None) -> RuntimeResidualCalibration:
    """Load repeated direct-vs-graph microbenchmark measurements.

    The input is intentionally raw and small.  Each sample must contain two
    or more host submission durations for both paths.  No native LLM field is
    accepted; malformed/under-specified data fails closed.
    """
    raw_bytes = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
        raise ValueError("runtime residual evidence SHA-256 mismatch")
    data = json.loads(raw_bytes.decode("utf-8-sig"), object_pairs_hook=_unique_pairs)
    required = {
        "schema", "source_kind", "target_llm_latency_used", "measurement_boundary",
        "hardware_id", "runtime_id", "architecture", "samples",
    }
    if not isinstance(data, Mapping) or set(data) - required - {"qualified", "validation_relative_error", "device", "cc", "driver_version", "runtime_version"}:
        raise ValueError("invalid runtime residual schema")
    if set(data) < required or data["schema"] != "heterollm.runtime-residual/v1":
        raise ValueError("runtime residual schema mismatch")
    if data["source_kind"] != "independent_synthetic_runtime_microbenchmark":
        raise ValueError("runtime residual evidence is not independent")
    if data["target_llm_latency_used"] is not False:
        raise ValueError("target LLM latency cannot calibrate runtime residual")
    if data["measurement_boundary"] != "host_submission_cuda_event_pair":
        raise ValueError("runtime residual measurement boundary is not paired submission/device timing")
    for key, expected in (("hardware_id", hardware_id), ("runtime_id", runtime_id), ("architecture", architecture)):
        if expected is not None and data[key] != expected:
            raise ValueError(f"runtime residual identity mismatch: {key}")
    samples = data["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("runtime residual samples must be a nonempty array")
    ordinary = []
    graph = []
    for row in samples:
        if not isinstance(row, Mapping) or set(row) - {"ordinary_durations_ns", "graph_durations_ns", "kernel_count", "replay_count", "split"}:
            raise ValueError("invalid runtime residual sample")
        count = row.get("kernel_count")
        replay_count = row.get("replay_count", 1)
        if type(count) is not int or count < 1 or type(replay_count) is not int or replay_count < 1:
            raise ValueError("runtime residual sample counts must be positive integers")
        for key, bucket in (("ordinary_durations_ns", ordinary), ("graph_durations_ns", graph)):
            values = row.get(key)
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"{key} requires repeated measurements")
            for value in values:
                _finite(value, key)
            median = statistics.median(values)
            bucket.append(median / (count if key == "ordinary_durations_ns" else replay_count))
    # The direct path contains one fixed host enqueue component plus N_kernel
    # submissions, so compare per-kernel medians rather than treating varying
    # kernel-count cases as repeated measurements of one scalar.  This keeps
    # the residual independent from any LLM request total.
    if not ordinary or not graph:
        raise ValueError("runtime residual samples are empty")
    return RuntimeResidualCalibration(
        str(data["hardware_id"]), str(data["runtime_id"]), str(data["architecture"]),
        statistics.median(ordinary), statistics.median(graph),
        "sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        validation_relative_error=data.get("validation_relative_error"),
        # Qualification is derived by the reducer from the predeclared
        # holdout/CV gates; never trust a caller-provided flag in raw evidence.
        qualified=False,
    )


__all__ = ["RuntimeResidualCalibration", "load_runtime_residual_calibration"]
