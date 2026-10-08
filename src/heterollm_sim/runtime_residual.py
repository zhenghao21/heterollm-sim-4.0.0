"""Independent CUDA Graph host lifecycle costs with held-out qualification.

CUDA event intervals are diagnostic data only: host API timing cannot replace
GPU kernel dispatch, computation, or physical memory service. Neither fitting
nor qualification accepts model latency data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

SCHEMA = "heterollm.cuda-graph-runtime/v2"
PHASES = ("ordinary_submit", "capture", "instantiate", "update",
          "first_launch_submit", "replay_submit", "destroy_exec", "destroy_graph")
DEVICE_PHASES = ("ordinary_device", "first_launch_device", "replay_device")
OPTIONAL_PHASES = ("update_failure",)
DEFAULT_MAX_RELATIVE_ERROR = 0.20
DEFAULT_ROBUST_NOISE_MULTIPLIER = 3.0
# Fixed before collecting measurements. Noise cannot enlarge the tolerance
# without bound: a noisy phase must not certify its own inaccurate prediction.
DEFAULT_ABSOLUTE_NOISE_CAP_NS = 1000.0
IDENTITY_FIELDS = ("hardware_id", "runtime_id", "architecture", "device", "cc",
                   "driver_version", "runtime_version", "cpu_id", "os_id")


def _number(value, label, *, allow_zero=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"{label} must be a finite {'nonnegative' if allow_zero else 'positive'} number")
    return float(value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate calibration field: {key}")
        result[key] = value
    return result


def _interpolate(points, n):
    for (n0, y0), (n1, y1) in zip(points, points[1:]):
        if n0 <= n <= n1:
            return y0 + (n - n0) / (n1 - n0) * (y1 - y0)
    raise ValueError(f"node_count {n} outside measured range [{points[0][0]}, {points[-1][0]}]")


def _policy(max_error=None):
    gate = DEFAULT_MAX_RELATIVE_ERROR if max_error is None else _number(max_error, "max_validation_relative_error")
    if gate > DEFAULT_MAX_RELATIVE_ERROR:
        raise ValueError("validation gate may only tighten the fixed 20% relative error limit")
    return {"max_relative_error": gate,
            "robust_noise_multiplier": DEFAULT_ROBUST_NOISE_MULTIPLIER,
            "absolute_noise_cap_ns": DEFAULT_ABSOLUTE_NOISE_CAP_NS}


def _score(samples, holdouts, policy):
    """Recompute every claim from training points and held-out repetitions."""
    cases, relative, absolute = [], dict.fromkeys(PHASES, 0.0), dict.fromkeys(PHASES, 0.0)
    qualified = {topology: dict.fromkeys(PHASES, True) for topology in samples}
    seen = set()
    for case in holdouts:
        topology, phase, n = case["topology"], case["phase"], case["node_count"]
        if topology not in samples or phase not in PHASES or type(n) is not int or n < 1:
            raise ValueError("invalid held-out topology, phase or node count")
        key = topology, phase, n
        if key in seen:
            raise ValueError("duplicate held-out topology/phase/size")
        seen.add(key)
        points = samples[topology][phase]
        if n in {point[0] for point in points}:
            raise ValueError("held-out size overlaps training sizes")
        repetitions = case["repeated_ns"]
        if not isinstance(repetitions, (list, tuple)) or len(repetitions) < 3:
            raise ValueError("held-out phase requires at least three repetitions")
        values = [_number(v, "held-out repetition", allow_zero=False) for v in repetitions]
        measured = statistics.median(values)
        predicted = _interpolate(points, n)
        mad = statistics.median(abs(v - measured) for v in values)
        noise = policy["robust_noise_multiplier"] * 1.4826 * mad
        noise_cap = policy["absolute_noise_cap_ns"]
        tolerance = max(policy["max_relative_error"] * measured, min(noise, noise_cap))
        absolute_error = abs(predicted - measured)
        relative_error = absolute_error / measured
        # A high-spread measurement is inconclusive, even if its median happens
        # to match. Do not advertise a 100%-noise phase as an accurate fit.
        stable = 1.4826 * mad <= max(policy["max_relative_error"] * measured, noise_cap)
        accepted = absolute_error <= tolerance and stable
        qualified[topology][phase] &= accepted
        relative[phase] = max(relative[phase], relative_error)
        absolute[phase] = max(absolute[phase], absolute_error)
        cases.append({"topology": topology, "phase": phase, "node_count": n,
                      "repeated_ns": values, "predicted_ns": predicted, "measured_ns": measured,
                      "absolute_error_ns": absolute_error, "relative_error": relative_error,
                      "repeat_mad_ns": mad, "absolute_noise_tolerance_ns": noise,
                      "effective_tolerance_ns": tolerance, "measurement_stable": stable,
                      "qualified": accepted})
    for topology in samples:
        phase_sizes = [{n for t, p, n in seen if t == topology and p == phase} for phase in PHASES]
        if not phase_sizes[0] or any(sizes != phase_sizes[0] for sizes in phase_sizes):
            raise ValueError("each topology requires held-out coverage of every phase at identical sizes")
    intervals = {}
    for topology, phases in samples.items():
        intervals[topology] = {}
        for phase, points in phases.items():
            intervals[topology][phase] = []
            for (low, _), (high, _) in zip(points, points[1:]):
                validations = [case for case in cases if case["topology"] == topology and case["phase"] == phase and low < case["node_count"] < high]
                intervals[topology][phase].append({"low": low, "high": high,
                    "holdout_sizes": [case["node_count"] for case in validations],
                    "qualified": bool(validations) and all(case["qualified"] for case in validations)})
            qualified[topology][phase] &= all(interval["qualified"] for interval in intervals[topology][phase])
    return {"validation_relative_error": max(relative.values()),
            "validation_relative_error_by_phase": relative,
            "validation_absolute_error_ns_by_phase": absolute,
            "validation_cases": tuple(cases), "qualification_by_topology_phase": qualified,
            "qualification_by_interval": intervals,
            "qualified": all(all(phases.values()) for phases in qualified.values())}


@dataclass(frozen=True)
class RuntimeResidualCalibration:
    hardware_id: str
    runtime_id: str
    architecture: str
    evidence: str
    topology_phase_samples: Mapping[str, Mapping[str, tuple[tuple[int, float], ...]]]
    topology_ranges: Mapping[str, tuple[int, int]]
    device: str = "unknown"
    cc: str = "unknown"
    driver_version: str = "unknown"
    runtime_version: str = "unknown"
    cpu_id: str = "unknown"
    os_id: str = "unknown"
    validation_relative_error: float | None = None
    validation_relative_error_by_phase: Mapping[str, float] | None = None
    validation_absolute_error_ns_by_phase: Mapping[str, float] | None = None
    validation_cases: tuple[Mapping[str, object], ...] = ()
    qualification_policy: Mapping[str, float] | None = None
    qualification_by_topology_phase: Mapping[str, Mapping[str, bool]] | None = None
    qualification_by_interval: Mapping[str, Mapping[str, tuple[Mapping[str, object], ...]]] | None = None
    training_cases: tuple[Mapping[str, object], ...] = ()
    qualified: bool = False
    source_kind: str = "independent_synthetic_runtime_microbenchmark"
    protocol: str = SCHEMA

    def __post_init__(self):
        for name in (*IDENTITY_FIELDS, "evidence"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty text")
        if self.protocol != SCHEMA or self.source_kind != "independent_synthetic_runtime_microbenchmark":
            raise ValueError("unsupported or non-independent runtime calibration")
        if type(self.qualified) is not bool:
            raise ValueError("qualified must be boolean")

    def costs(self, topology: str, node_count: int, *, phases=None) -> dict[str, float]:
        if topology not in self.topology_phase_samples:
            raise ValueError(f"unmeasured CUDA graph topology: {topology}")
        if type(node_count) is not int or node_count < 1:
            raise ValueError("node_count must be a positive integer")
        requested = tuple(PHASES if phases is None else phases)
        if not requested or any(p not in PHASES for p in requested):
            raise ValueError("requested CUDA lifecycle phase has no independent measured cost")
        # Qualify only phases that actually occur. A failed cold instantiate
        # fit cannot silently become zero, nor block an unrelated warm replay.
        low, high = self.topology_ranges[topology]
        if not low <= node_count <= high:
            raise ValueError(f"node_count {node_count} outside measured range [{low}, {high}]")
        phase_intervals = (self.qualification_by_interval or {}).get(topology, {})
        failed = [p for p in requested if not any(interval["low"] <= node_count <= interval["high"] and interval["qualified"]
                  for interval in phase_intervals.get(p, ()))]
        if failed:
            raise ValueError("CUDA Graph runtime calibration is not qualified for " + topology + ": " + ", ".join(failed))
        return {p: _interpolate(self.topology_phase_samples[topology][p], node_count) for p in requested}

    def cost(self, kernel_count: int, *, captured: bool, replay_count: int = 1, topology: str | None = None) -> dict:
        if topology is None:
            if len(self.topology_phase_samples) != 1:
                raise ValueError("topology is required for multi-topology CUDA Graph calibration")
            topology = next(iter(self.topology_phase_samples))
        if type(captured) is not bool or type(replay_count) is not int or replay_count < 1:
            raise ValueError("invalid captured/replay_count")
        phase = "replay_submit" if captured else "ordinary_submit"
        costs = self.costs(topology, kernel_count, phases=(phase,))
        return {"mode": "cuda_graph_replay" if captured else "ordinary_launch", "kernel_count": kernel_count,
                "submission_count": replay_count if captured else kernel_count,
                "service_ns": costs[phase] * replay_count, "phase_costs_ns": costs,
                "qualified": True, "validation_relative_error": self.validation_relative_error,
                "evidence": self.evidence, "source_kind": self.source_kind, "topology": topology}

    def measured_diagnostic(self, topology: str, node_count: int, *, phases=None) -> dict:
        """Exact measured sizes only, explicitly outside predictive qualification.

        Intended for sensitivity/error reports. The returned object is not a
        production cost map; callers must retain its unqualified/uncertainty
        labels. No holdout is converted into a fitted training point.
        """
        if type(node_count) is not int or node_count < 1:
            raise ValueError("node_count must be a positive integer")
        requested = tuple(PHASES if phases is None else phases)
        if not requested or any(phase not in PHASES for phase in requested):
            raise ValueError("unknown diagnostic lifecycle phase")
        records = {}
        for phase in requested:
            train = [case for case in self.training_cases if case["topology"] == topology and case["node_count"] == node_count and case["phase"] == phase]
            holdout = [case for case in self.validation_cases if case["topology"] == topology and case["node_count"] == node_count and case["phase"] == phase]
            matches = train or holdout
            if len(matches) != 1:
                raise ValueError("diagnostic requires an exact independently measured topology/size/phase")
            values = matches[0]["repeated_ns"]
            median = statistics.median(values)
            records[phase] = {"median_ns": median,
                "repeat_mad_ns": statistics.median(abs(v-median) for v in values),
                "minimum_ns": min(values), "maximum_ns": max(values),
                "repeat_count": len(values), "source_split": "train" if train else "holdout"}
        return {"mode": "independent_measurement_diagnostic", "prediction_qualified": False,
                "topology": topology, "node_count": node_count, "evidence": self.evidence,
                "phase_measurements": records,
                "warning": "Exact observed measurements for diagnostic sensitivity only; not a qualified predictive calibration."}


def _identity(data, expected):
    for key in IDENTITY_FIELDS:
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} must be nonempty text")
        if expected.get(key) is not None and data[key] != expected[key]:
            raise ValueError(f"runtime calibration identity mismatch: {key}")


def load_runtime_residual_calibration(path: str | Path, *, max_validation_relative_error=None, **identity) -> RuntimeResidualCalibration:
    if set(identity) - set(IDENTITY_FIELDS):
        raise TypeError("unknown runtime calibration identity argument")
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs)
    required = {"schema", "source_kind", "target_llm_latency_used", "measurement_boundary", "samples", *IDENTITY_FIELDS}
    if not isinstance(data, Mapping) or set(data) != required or data["schema"] != SCHEMA:
        raise ValueError("invalid CUDA Graph runtime calibration schema")
    if data["source_kind"] != "independent_synthetic_runtime_microbenchmark" or data["target_llm_latency_used"] is not False:
        raise ValueError("only independent non-LLM calibration data is accepted")
    if data["measurement_boundary"] != "host_wall_and_cuda_event_separate":
        raise ValueError("host and device timing boundaries must be recorded separately")
    _identity(data, identity)
    rows = data["samples"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("samples must be a nonempty array")
    training, training_cases, holdouts, seen = {}, [], [], set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"topology", "node_count", "split", "timings_ns"}:
            raise ValueError("invalid microbenchmark sample")
        topology, n, split, timings = (row[k] for k in ("topology", "node_count", "split", "timings_ns"))
        if not isinstance(topology, str) or not topology or type(n) is not int or n < 1 or split not in ("train", "holdout"):
            raise ValueError("invalid sample topology, size or split")
        if (topology, n) in seen:
            raise ValueError("duplicate graph size or train/holdout overlap")
        seen.add((topology, n))
        if (not isinstance(timings, Mapping) or not set(PHASES + DEVICE_PHASES).issubset(timings)
                or set(timings) - set(PHASES + DEVICE_PHASES + OPTIONAL_PHASES)):
            raise ValueError("invalid timing fields")
        for phase, values in timings.items():
            if not isinstance(values, list) or len(values) < 3:
                raise ValueError(f"{phase} requires at least three repeated timings")
            values = [_number(v, phase, allow_zero=False) for v in values]
            if phase in DEVICE_PHASES + OPTIONAL_PHASES:
                continue
            if split == "train":
                training.setdefault(topology, {}).setdefault(phase, []).append((n, statistics.median(values)))
                training_cases.append({"topology": topology, "phase": phase, "node_count": n, "repeated_ns": values})
            else:
                holdouts.append({"topology": topology, "phase": phase, "node_count": n, "repeated_ns": values})
    ranges = {}
    for topology, phases in training.items():
        for phase, points in phases.items():
            if len(points) < 2:
                raise ValueError("every topology requires at least two training sizes")
            phases[phase] = tuple(sorted(points))
        ranges[topology] = (phases[PHASES[0]][0][0], phases[PHASES[0]][-1][0])
    for case in holdouts:
        if case["topology"] not in ranges or not ranges[case["topology"]][0] < case["node_count"] < ranges[case["topology"]][1]:
            raise ValueError("held-out sizes must be inside the measured training range")
    policy = _policy(max_validation_relative_error)
    scores = _score(training, holdouts, policy)
    return RuntimeResidualCalibration(**{key: data[key] for key in IDENTITY_FIELDS}, evidence=str(Path(path).resolve()),
                                      topology_phase_samples=training, topology_ranges=ranges,
                                      qualification_policy=policy, training_cases=tuple(training_cases), **scores)


def runtime_calibration_to_dict(calibration: RuntimeResidualCalibration) -> dict:
    if not isinstance(calibration, RuntimeResidualCalibration):
        raise TypeError("calibration must be RuntimeResidualCalibration")
    # json normalization exactly matches serde.to_primitive's dataclass shape.
    return json.loads(json.dumps(asdict(calibration)))


def _consistent(claim, derived, label):
    if isinstance(derived, Mapping):
        if not isinstance(claim, Mapping) or set(claim) != set(derived):
            raise ValueError(f"serialized {label} is inconsistent")
        for key, value in derived.items():
            _consistent(claim[key], value, label + "." + str(key))
    elif isinstance(derived, (tuple, list)):
        if not isinstance(claim, (tuple, list)) or len(claim) != len(derived):
            raise ValueError(f"serialized {label} is inconsistent")
        for left, right in zip(claim, derived):
            _consistent(left, right, label)
    elif isinstance(derived, float):
        _number(claim, label)
        if not math.isclose(claim, derived, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"serialized {label} does not match recomputed measurement")
    elif type(claim) is not type(derived) or claim != derived:
        raise ValueError(f"serialized {label} does not match recomputed measurement")


def runtime_calibration_from_dict(data: Mapping, *, max_validation_relative_error=None, **identity) -> RuntimeResidualCalibration:
    if set(identity) - set(IDENTITY_FIELDS):
        raise TypeError("unknown runtime calibration identity argument")
    required = set(RuntimeResidualCalibration.__dataclass_fields__)
    if not isinstance(data, Mapping) or set(data) != required or data["protocol"] != SCHEMA:
        raise ValueError("invalid serialized runtime calibration")
    _identity(data, identity)
    samples, ranges = data["topology_phase_samples"], data["topology_ranges"]
    if not isinstance(samples, Mapping) or not samples or not isinstance(ranges, Mapping) or set(samples) != set(ranges):
        raise ValueError("invalid topology calibration map")
    normalized, normalized_ranges = {}, {}
    for topology, phases in samples.items():
        if not isinstance(topology, str) or not topology or not isinstance(phases, Mapping) or set(phases) != set(PHASES):
            raise ValueError("invalid topology phase samples")
        normalized[topology], reference_sizes = {}, None
        for phase, points in phases.items():
            if not isinstance(points, (list, tuple)) or len(points) < 2:
                raise ValueError("each phase requires at least two training sizes")
            parsed = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2 or type(point[0]) is not int or point[0] < 1:
                    raise ValueError("invalid training phase point")
                parsed.append((point[0], _number(point[1], phase, allow_zero=False)))
            sizes = [point[0] for point in parsed]
            if sizes != sorted(set(sizes)) or (reference_sizes is not None and sizes != reference_sizes):
                raise ValueError("training sizes must be strictly ordered and identical across phases")
            reference_sizes = sizes
            normalized[topology][phase] = tuple(parsed)
        bounds = ranges[topology]
        if not isinstance(bounds, (list, tuple)) or list(bounds) != [reference_sizes[0], reference_sizes[-1]] or any(type(n) is not int for n in bounds):
            raise ValueError("topology range does not match training samples")
        normalized_ranges[topology] = tuple(bounds)
    policy = data["qualification_policy"]
    if not isinstance(policy, Mapping) or set(policy) != set(_policy()):
        raise ValueError("invalid serialized qualification policy")
    _policy(policy["max_relative_error"])
    # A profile may tighten the relative gate, but may not enlarge the fixed
    # noise rule to turn a failed measurement into a successful calibration.
    if policy["robust_noise_multiplier"] != DEFAULT_ROBUST_NOISE_MULTIPLIER or policy["absolute_noise_cap_ns"] != DEFAULT_ABSOLUTE_NOISE_CAP_NS:
        raise ValueError("unsupported qualification noise policy")
    training_cases = data["training_cases"]
    if not isinstance(training_cases, (list, tuple)):
        raise ValueError("training repetitions must be an array")
    seen = set()
    for case in training_cases:
        if not isinstance(case, Mapping) or set(case) != {"topology", "phase", "node_count", "repeated_ns"}:
            raise ValueError("invalid training repetition case")
        key = case["topology"], case["phase"], case["node_count"]
        if key in seen or key[0] not in normalized or key[1] not in PHASES or type(key[2]) is not int:
            raise ValueError("invalid training repetition identity")
        seen.add(key)
        repetitions = case["repeated_ns"]
        if not isinstance(repetitions, (list, tuple)) or len(repetitions) < 3:
            raise ValueError("training phase requires at least three repetitions")
        values = [_number(value, "training repetition", allow_zero=False) for value in repetitions]
        expected = dict(normalized[key[0]][key[1]]).get(key[2])
        if expected is None or not math.isclose(expected, statistics.median(values), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("training medians do not match repeated measurements")
    expected_keys = {(topology, phase, n) for topology, phases in normalized.items() for phase, points in phases.items() for n, _ in points}
    if seen != expected_keys:
        raise ValueError("training repetition coverage is incomplete")
    cases = data["validation_cases"]
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("serialized validation cases must be nonempty")
    essentials = []
    for case in cases:
        if not isinstance(case, Mapping) or not {"topology", "phase", "node_count", "repeated_ns"}.issubset(case):
            raise ValueError("invalid serialized held-out case")
        essentials.append({k: case[k] for k in ("topology", "phase", "node_count", "repeated_ns")})
    scores = _score(normalized, essentials, policy)
    for key, value in scores.items():
        _consistent(data[key], value, key)
    effective_policy = _policy(max_validation_relative_error) if max_validation_relative_error is not None else dict(policy)
    if effective_policy != policy:
        scores = _score(normalized, essentials, effective_policy)
    return RuntimeResidualCalibration(**{key: data[key] for key in IDENTITY_FIELDS}, evidence=data["evidence"],
                                      source_kind=data["source_kind"], protocol=data["protocol"],
                                      topology_phase_samples=normalized, topology_ranges=normalized_ranges,
                                      qualification_policy=effective_policy, training_cases=tuple(dict(case) for case in training_cases), **scores)


__all__ = ["RuntimeResidualCalibration", "load_runtime_residual_calibration", "runtime_calibration_to_dict", "runtime_calibration_from_dict", "SCHEMA", "PHASES", "DEVICE_PHASES"]


STRUCTURE_SCHEMA = "heterollm.cuda-graph-structure-measurements/v1"


def _measurement_summary(values):
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise ValueError("independent phase measurement needs at least three repetitions")
    repetitions = [_number(value, "independent timing", allow_zero=False) for value in values]
    median = statistics.median(repetitions)
    return {"median_ns": median, "repeat_mad_ns": statistics.median(abs(value-median) for value in repetitions),
            "minimum_ns": min(repetitions), "maximum_ns": max(repetitions), "repeat_count": len(repetitions)}


@dataclass(frozen=True)
class RuntimeStructureMeasurements:
    """Exact structural measurements for an explicitly enabled experiment.

    No interpolation, extrapolation, or qualification is implied. The strict
    costs() entry point always rejects this type; experimental_costs() is an
    explicit opt-in and its caller must retain prediction_qualified=False.
    """
    hardware_id: str
    runtime_id: str
    architecture: str
    evidence: str
    samples: tuple[Mapping[str, object], ...]
    device: str
    cc: str
    driver_version: str
    runtime_version: str
    cpu_id: str
    os_id: str
    limitations: tuple[str, ...]
    update_pairs: tuple[Mapping[str, object], ...] = ()
    protocol: str = STRUCTURE_SCHEMA
    source_kind: str = "independent_synthetic_runtime_microbenchmark"
    qualified: bool = False
    validation_relative_error: float | None = None

    def __post_init__(self):
        _identity(self.__dict__, {})
        if self.protocol != STRUCTURE_SCHEMA or self.source_kind != "independent_synthetic_runtime_microbenchmark":
            raise ValueError("unsupported independent structural measurement source")
        if self.qualified is not False or self.validation_relative_error is not None:
            raise ValueError("experimental structure measurements cannot claim predictive qualification")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("structural measurement evidence is required")
        if not isinstance(self.limitations, tuple) or not self.limitations or any(not isinstance(item, str) or not item for item in self.limitations):
            raise ValueError("experimental structural limitations must be retained")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("structural measurement samples are required")
        seen = set()
        for sample in self.samples:
            if not isinstance(sample, Mapping) or set(sample) != {"topology", "node_count", "source_structures", "timings_ns", "phase_measurements"}:
                raise ValueError("invalid structural measurement sample")
            key = sample["topology"], sample["node_count"]
            if not isinstance(key[0], str) or not key[0] or type(key[1]) is not int or key[1] < 1 or key in seen:
                raise ValueError("invalid or duplicate structural measurement identity")
            seen.add(key)
            if not isinstance(sample["source_structures"], (list, tuple)) or not sample["source_structures"]:
                raise ValueError("structural compiler provenance is required")
            timings = sample["timings_ns"]
            if (not isinstance(timings, Mapping) or not set(PHASES+DEVICE_PHASES).issubset(timings)
                    or set(timings) - set(PHASES+DEVICE_PHASES+OPTIONAL_PHASES)):
                raise ValueError("structural measurements must cover every lifecycle phase")
            _consistent(sample["phase_measurements"], {phase: _measurement_summary(values) for phase, values in timings.items()}, "structural phase measurements")
        pair_keys = set()
        for pair in self.update_pairs:
            if not isinstance(pair, Mapping) or set(pair) != {"old", "new", "phase", "repeated_ns", "measurement"} or pair["phase"] != "update_failure":
                raise ValueError("invalid independent update structure pair")
            keys = []
            for role in ("old", "new"):
                descriptor = pair[role]
                if not isinstance(descriptor, Mapping) or set(descriptor) != {"topology", "node_count"}:
                    raise ValueError("invalid independent update structure descriptor")
                self._sample(descriptor["topology"], descriptor["node_count"])
                keys.append((descriptor["topology"], descriptor["node_count"]))
            key = tuple(keys)
            if key in pair_keys:
                raise ValueError("duplicate independent update structure pair")
            pair_keys.add(key)
            _consistent(pair["measurement"], _measurement_summary(pair["repeated_ns"]), "update pair measurement")

    def _sample(self, topology, node_count):
        if type(node_count) is not int or node_count < 1:
            raise ValueError("node_count must be a positive integer")
        matches = [sample for sample in self.samples if sample["topology"] == topology and sample["node_count"] == node_count]
        if len(matches) != 1:
            raise ValueError("experimental runtime pricing requires this exact independently measured typed topology and node count")
        return matches[0]

    def costs(self, topology, node_count, *, phases=None):
        raise ValueError("structural measurements are not qualified; explicit experimental_exact_structure mode is required")

    def experimental_costs(self, topology, node_count, *, phases=None):
        requested = tuple(PHASES if phases is None else phases)
        if not requested or any(phase not in PHASES + OPTIONAL_PHASES for phase in requested):
            raise ValueError("CUDA lifecycle phase has no independent experimental measurement")
        sample = self._sample(topology, node_count)
        if any(phase not in sample["timings_ns"] for phase in requested):
            raise ValueError("CUDA lifecycle phase has no independent experimental measurement")
        return {phase: _measurement_summary(sample["timings_ns"][phase])["median_ns"] for phase in requested}

    def uncertainty(self, topology, node_count, *, phases=None):
        requested = tuple(PHASES if phases is None else phases)
        if not requested or any(phase not in PHASES + OPTIONAL_PHASES for phase in requested):
            raise ValueError("CUDA lifecycle phase has no independent experimental measurement")
        sample = self._sample(topology, node_count)
        if any(phase not in sample["timings_ns"] for phase in requested):
            raise ValueError("CUDA lifecycle phase has no independent experimental measurement")
        return {phase: _measurement_summary(sample["timings_ns"][phase]) for phase in requested}

    def update_pair_measurement(self, old, new):
        matches = [pair for pair in self.update_pairs if dict(pair["old"]) == dict(old) and dict(pair["new"]) == dict(new)]
        if len(matches) != 1:
            raise ValueError("failed-update cost requires this exact independently measured old/new structure pair")
        return _measurement_summary(matches[0]["repeated_ns"])


def load_runtime_structure_measurements(path, **identity):
    if set(identity) - set(IDENTITY_FIELDS):
        raise TypeError("unknown structural measurement identity argument")
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs)
    if (not isinstance(data, Mapping) or data.get("schema") != STRUCTURE_SCHEMA
            or data.get("target_llm_latency_used") is not False or data.get("prediction_qualified") is not False
            or data.get("cost_scope") != "host_api_lifecycle_only"
            or data.get("measurement_boundary") != "host_wall_and_cuda_event_separate"):
        raise ValueError("invalid independent experimental structural measurement artifact")
    _identity(data, identity)
    return RuntimeStructureMeasurements(**{key: data[key] for key in IDENTITY_FIELDS},
        evidence=str(Path(path).resolve()), samples=tuple(data["samples"]), limitations=tuple(data["limitations"]),
        update_pairs=tuple(data.get("update_pairs", ())),
        source_kind=data["source_kind"])


def runtime_structure_measurements_from_dict(data, **identity):
    if set(identity) - set(IDENTITY_FIELDS):
        raise TypeError("unknown structural measurement identity argument")
    if not isinstance(data, Mapping) or set(data) != set(RuntimeStructureMeasurements.__dataclass_fields__):
        raise ValueError("invalid serialized experimental structure measurements")
    _identity(data, identity)
    values = dict(data)
    for key in ("samples", "limitations", "update_pairs"):
        if not isinstance(values[key], (list, tuple)):
            raise ValueError(f"{key} must be an array")
        values[key] = tuple(values[key])
    return RuntimeStructureMeasurements(**values)


__all__ += ["RuntimeStructureMeasurements", "load_runtime_structure_measurements", "runtime_structure_measurements_from_dict", "STRUCTURE_SCHEMA"]
