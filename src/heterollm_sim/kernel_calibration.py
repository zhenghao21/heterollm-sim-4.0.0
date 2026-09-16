"""Strict independent synthetic-kernel evidence; no LLM latency fitting.

This opt-in module does not change planner defaults.  A device-kernel elapsed
measurement is a conditional duration floor, not pure compute throughput, HBM
service, a graph wall time, or an accuracy guarantee for an LLM workload.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import ResourceDemand, TaskCategory
from .cost_models import CostPhase

PROFILE_SCHEMA = "heterollm.synthetic-kernel-profile/v2"
OWNER_SCHEMA = "heterollm.kernel-single-stream-owner/v1"
DEVICE_BOUNDARY = "cuda_device_kernel_interval"
SOURCE_KIND = "independent_synthetic_operator"
ROLES = frozenset({"conversion", "main", "fixup"})
KEY_FIELDS = frozenset({
    "op", "role", "m", "n", "k_logical", "k_executed",
    "activation_dtype", "weight_format", "output_dtype", "accumulator_dtype",
    "layout", "strides", "kernel_family", "kernel_variant", "dispatch_signature",
    "cache_protocol", "effective_hardware_sha256", "runtime_sha256", "launch_geometry",
})
_REQUIRED_EVIDENCE_KINDS = frozenset({
    "protocol", "probe_source", "extractor", "raw_events", "numerical_validation",
    "quality_report", "hardware_runtime_observation", "build_manifest",
    "effective_hardware_profile", "runtime_binary", "native_source", "measurement_bundle",
    "raw_application", "raw_sqlite", "tool_log", "telemetry", "clock_receipt",
    "process_spec", "process_start", "collection_freeze",
})
_GATE_KINDS = {"numerical": "numerical_validation", "stability": "quality_report",
               "profiling": "quality_report"}
_HASH = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return value


def _fields(value: Any, names: set[str] | frozenset[str], name: str) -> Mapping[str, Any]:
    result = _mapping(value, name)
    if set(result) != names:
        raise ValueError(f"{name} has missing or unsupported fields")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be finite and positive") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _json(data: bytes) -> Mapping[str, Any]:
    return _mapping(json.loads(data.decode("utf-8-sig"), object_pairs_hook=_json_object,
                               parse_constant=_invalid_constant), "JSON document")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def canonical_kernel_key(query: Mapping[str, Any]) -> str:
    """Return an exact joint key; no output-shape shortcut or interpolation."""
    query = _fields(query, KEY_FIELDS, "kernel query")
    _text(query["role"], "role")
    if query["op"] != "ggml_mul_mat" or query["role"] not in ROLES:
        raise ValueError("unsupported physical operation or kernel role")
    for name in ("m", "n", "k_logical", "k_executed"):
        _positive_int(query[name], name)
    if query["k_executed"] < query["k_logical"]:
        raise ValueError("executed K cannot be smaller than logical K")
    for name in ("activation_dtype", "weight_format", "output_dtype", "accumulator_dtype",
                 "layout", "kernel_family", "kernel_variant", "dispatch_signature", "cache_protocol"):
        _text(query[name], name)
    for name in ("effective_hardware_sha256", "runtime_sha256"):
        _sha(query[name], name)
    strides = _fields(query["strides"], {"activation_bytes", "weight_bytes", "output_bytes"}, "strides")
    for name, values in strides.items():
        if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 4:
            raise ValueError(f"{name} must contain two to four byte strides")
        for stride in values:
            _positive_int(stride, name)
    geometry = _fields(query["launch_geometry"], {"grid", "block", "static_shared_bytes", "dynamic_shared_bytes"}, "launch geometry")
    for field in ("grid", "block"):
        if not isinstance(geometry[field], (tuple, list)) or len(geometry[field]) != 3:
            raise ValueError("launch geometry must have three axes")
        for value in geometry[field]:
            _positive_int(value, field)
    for field in ("static_shared_bytes", "dynamic_shared_bytes"):
        if type(geometry[field]) is not int or geometry[field] < 0:
            raise ValueError("shared memory must be a nonnegative byte count")
    return json.dumps(_plain(query), sort_keys=True, separators=(",", ":"), allow_nan=False)


def runtime_identity_sha256(runtime_facts: Mapping[str, Any]) -> str:
    """Hash the captured driver/DLL/source/environment facts, excluding the hash itself."""
    facts = _fields(runtime_facts, {"observed", "driver_version", "dll_sha256", "source_sha256",
                                   "kernel_environment"}, "runtime identity facts")
    payload = json.dumps(_plain(facts), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KernelEvidenceFile:
    evidence_id: str
    kind: str
    path: Path
    sha256: str
    size_bytes: int
    document: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class KernelEntry:
    key: str
    status: str
    device_ns: float | None
    sample_count: int
    rejection_reason: str | None
    evidence_ids: tuple[str, ...]
    gates: Mapping[str, Any]


@dataclass(frozen=True)
class KernelCalibrationProfile:
    profile_id: str
    sha256: str
    hardware: Mapping[str, Any]
    runtime: Mapping[str, Any]
    cache: Mapping[str, Any]
    entries: Mapping[str, KernelEntry]
    evidence: Mapping[str, KernelEvidenceFile]
    # Only the loader sets this after checking all referenced bytes.  This is
    # an evidence-integrity marker, never proof that the measurements are true.
    evidence_files_verified: bool = False
    semantic_evidence_verified: bool = False


def _verified_bytes(path: Path, expected_sha: str, expected_size: int | None,
                    *, keep: bool, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    _sha(expected_sha, "file SHA")
    if keep and path.stat().st_size > max_bytes:
        raise ValueError("profile or execution-contract document is too large")
    digest = hashlib.sha256()
    parts: list[bytes] = []
    count = 0
    before = path.stat()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            count += len(block)
            if keep:
                parts.append(block)
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after) or count != before.st_size:
        raise ValueError("evidence file changed while being read")
    if expected_size is not None and count != expected_size:
        raise ValueError("evidence file size mismatch")
    if digest.hexdigest() != expected_sha:
        raise ValueError("evidence content SHA mismatch")
    return b"".join(parts)


def _identity_facts(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    hardware = _fields(data["hardware"], {"observed", "gpu_uuid", "compute_capability", "sm_count", "l2_bytes",
                                                  "effective_profile_sha256"}, "hardware")
    if hardware["observed"] is not True:
        raise ValueError("hardware identity must be actually observed")
    _text(hardware["gpu_uuid"], "gpu_uuid")
    for name in ("compute_capability", "sm_count", "l2_bytes"):
        _positive_int(hardware[name], name)
    _sha(hardware["effective_profile_sha256"], "effective hardware profile SHA")
    runtime = _fields(data["runtime"], {"observed", "profile_sha256", "driver_version", "dll_sha256",
                                                "source_sha256", "kernel_environment"}, "runtime")
    if runtime["observed"] is not True:
        raise ValueError("runtime identity must be actually observed")
    _sha(runtime["profile_sha256"], "runtime profile SHA")
    _text(runtime["driver_version"], "driver_version")
    for name in ("dll_sha256", "source_sha256"):
        for label, value in _mapping(runtime[name], name).items():
            _text(label, name)
            _sha(value, name)
    for label, value in _mapping(runtime["kernel_environment"], "kernel_environment").items():
        _text(label, "environment name")
        if value is not None and not isinstance(value, str):
            raise ValueError("captured kernel environment must contain strings or explicit absence")
    captured_runtime_hash = runtime_identity_sha256({name: value for name, value in runtime.items()
                                                     if name != "profile_sha256"})
    if runtime["profile_sha256"] != captured_runtime_hash:
        raise ValueError("runtime profile SHA does not bind captured DLL/source/environment facts")
    cache = _fields(data["cache"], {"observed", "protocol", "l2_bytes", "sweep_bytes"}, "cache")
    if cache["observed"] is not True or cache["l2_bytes"] != hardware["l2_bytes"]:
        raise ValueError("cache facts must match the observed hardware")
    _positive_int(cache["l2_bytes"], "cache L2 bytes")
    if cache["protocol"] == "cold_sweep_ge_4_l2":
        if _positive_int(cache["sweep_bytes"], "cache sweep bytes") < 4 * cache["l2_bytes"]:
            raise ValueError("cold sweep must cover at least four times L2")
    elif cache["protocol"] == "hot_same_allocation_repeat":
        if type(cache["sweep_bytes"]) is not int or cache["sweep_bytes"] != 0:
            raise ValueError("hot-repeat evidence cannot include an eviction sweep")
    else:
        raise ValueError("unsupported cache protocol")
    return hardware, runtime, cache



def _digest_document(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _equal(actual: Any, expected: Any, label: str) -> None:
    if _plain(actual) != _plain(expected):
        raise ValueError(label + " does not match independently derived evidence")


def _close(actual: Any, expected: float, label: str) -> None:
    number = _positive_number(actual, label)
    if not math.isclose(number, expected, rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError(label + " does not match independently derived evidence")


def _distribution(values: list[float]) -> dict[str, float | int]:
    values = sorted(_positive_number(value, "measured sample") for value in values)
    if not values:
        raise ValueError("measured samples missing")
    def quantile(q: float) -> float:
        i = (len(values) - 1) * q
        lo = int(i)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (i - lo)
    low, high = quantile(.1), quantile(.9)
    return {"count": len(values), "minimum_ns": values[0], "maximum_ns": values[-1],
            "median_ns": statistics.median(values), "p10_ns": low, "p90_ns": high,
            "p90_div_p10": high / low}


def _check_distribution(declared: Any, values: list[float], label: str) -> None:
    expected = _distribution(values)
    declared = _mapping(declared, label)
    for key, value in expected.items():
        if key == "count":
            _equal(declared.get(key), value, label + ":count")
        else:
            _close(declared.get(key), float(value), label + ":" + key)


_EFFECTIVE_SCHEMA = "heterollm.kernel-effective-hardware/v1"
_FACT_FIELDS = {"gpu_uuid", "compute_capability", "sm_count", "l2_bytes"}
_CONFIG_FIELDS = {"compute", "memory", "cache", "scheduling", "placement"}
_CONTEXT_SEAL = object()


def effective_hardware_sha256(document: Mapping[str, Any]) -> str:
    """Hash complete actual effective configuration, including HBM/HBF sections."""
    document = _fields(document, {"schema", "observed_device", "simulator_configuration"}, "effective hardware")
    if document["schema"] != _EFFECTIVE_SCHEMA:
        raise ValueError("unsupported effective hardware schema")
    facts = _fields(document["observed_device"], _FACT_FIELDS, "observed device")
    _text(facts["gpu_uuid"], "GPU UUID")
    for name in _FACT_FIELDS - {"gpu_uuid"}:
        _positive_int(facts[name], name)
    configuration = _fields(document["simulator_configuration"], _CONFIG_FIELDS, "complete effective configuration")
    for section in configuration.values():
        _mapping(section, "effective hardware section")
    return _digest_document(document)


@dataclass(frozen=True)
class KernelExecutionContext:
    """Factory-built actual planner/runtime handoff; never inferred from an owner file."""
    hardware_sha256: str
    hardware: Mapping[str, Any]
    runtime_sha256: str
    device_id: str
    stream_id: str
    resources: Mapping[str, Any]
    invocation_keys: Mapping[str, str]
    phase_signatures: Mapping[str, str]
    _seal: Any = None


def _phase_signature(phase: CostPhase) -> str:
    return _digest_document({"name": phase.name, "category": phase.category.value,
        "demands": [{"resource_id": d.resource_id, "service_ns": d.service_ns,
                     "bytes_moved": d.bytes_moved, "energy_pj": d.energy_pj,
                     "work_units": d.work_units} for d in phase.demands],
        "metadata": _plain(phase.metadata)})


def create_kernel_execution_context(*, effective_hardware: Mapping[str, Any],
        runtime_facts: Mapping[str, Any], device_id: str, stream_id: str,
        stream_count: int, concurrent_kernels: bool, resource_directory: Mapping[str, Any],
        invocations: Mapping[str, tuple[CostPhase, Mapping[str, Any]]]) -> KernelExecutionContext:
    """Explicit actual resource catalog+invocation binding; absent context means fallback.

    The planner must supply its complete effective configuration. This factory
    cannot inspect caller state omitted from that configuration. No automatic
    planner integration is enabled by this module.
    """
    hardware_sha = effective_hardware_sha256(effective_hardware)
    runtime_sha = runtime_identity_sha256(runtime_facts)
    if runtime_facts.get("observed") is not True:
        raise ValueError("observed runtime context required")
    if type(stream_count) is not int or stream_count != 1 or concurrent_kernels is not False:
        raise ValueError("single-stream runtime context required")
    if not re.fullmatch(r"gpu[0-9]+", _text(device_id, "device id")):
        raise ValueError("GPU context device required")
    if not re.fullmatch(r"[A-Za-z0-9_]+", _text(stream_id, "stream id")):
        raise ValueError("canonical stream id required")
    catalog = _mapping(resource_directory, "actual resource directory")
    frozen_catalog = {}
    for resource_id, raw in catalog.items():
        resource = _fields(raw, {"device_id", "device_kind", "kind", "physical_owner"}, "resource")
        _text(resource_id, "resource id")
        for value in resource.values():
            _text(value, "resource fact")
        if resource["physical_owner"] not in catalog:
            raise ValueError("resource physical owner absent from directory")
        frozen_catalog[resource_id] = dict(resource)
    envelope = device_id + ".kernel_stream." + stream_id
    expected = {"device_id": device_id, "device_kind": "gpu", "kind": "kernel_envelope", "physical_owner": envelope}
    if frozen_catalog.get(envelope) != expected:
        raise ValueError("dedicated GPU kernel envelope absent from actual directory")
    keys, signatures = {}, {}
    for invocation_id, value in _mapping(invocations, "actual invocations").items():
        _text(invocation_id, "invocation id")
        if not isinstance(value, (tuple, list)) or len(value) != 2 or not isinstance(value[0], CostPhase):
            raise ValueError("actual CostPhase and physical key required")
        phase, query = value
        key = canonical_kernel_key(query)
        if query["effective_hardware_sha256"] != hardware_sha or query["runtime_sha256"] != runtime_sha:
            raise ValueError("invocation key differs from actual context identities")
        if phase.category != TaskCategory.COMPUTE:
            raise ValueError("GPU compute invocation required")
        owners = []
        for demand in phase.demands:
            resource = frozen_catalog.get(demand.resource_id)
            if (resource is None or resource["device_id"] != device_id or resource["device_kind"] != "gpu"
                    or resource["kind"] not in {"tensor", "scalar", "sfu", "memory", "cache"}
                    or any(v in demand.resource_id.casefold() for v in ("frontend", "launch", "cpu", "host"))):
                raise ValueError("actual phase resources are not device kernel resources")
            owners.append(resource["physical_owner"])
        if len(owners) != len(set(owners)) or envelope in owners:
            raise ValueError("duplicate physical owner in actual invocation")
        keys[invocation_id], signatures[invocation_id] = key, _phase_signature(phase)
    return KernelExecutionContext(hardware_sha, _freeze(effective_hardware), runtime_sha,
        device_id, stream_id, _freeze(frozen_catalog), _freeze(keys), _freeze(signatures), _CONTEXT_SEAL)


_BUNDLE_SCHEMA = "heterollm.kernel-measurement-bundle/v1"
_EXPECTED_CALLS = [("first_call", 0)] + [("warmup", i) for i in range(5)] + [("formal", i) for i in range(30)]
_PROCESS_MASK = 0xFFFFFFFFFF000000
_ROLES = {"quantize_q8_1": ("conversion", "cuda_mmvq"), "mul_mat_vec_q": ("main", "cuda_mmvq"),
          "quantize_mmq_q8_1": ("conversion", "cuda_mmq"), "mul_mat_q": ("main", "cuda_mmq"),
          "mul_mat_q_stream_k_fixup": ("fixup", "cuda_mmq")}


def _document(evidence: Mapping[str, KernelEvidenceFile], evidence_id: str, kind: str) -> Mapping[str, Any]:
    ref = evidence.get(evidence_id)
    if ref is None or ref.kind != kind or ref.document is None:
        raise ValueError("missing semantic " + kind + " evidence")
    return ref.document


def _reference_matches(reference: Any, evidence: KernelEvidenceFile) -> None:
    reference = _fields(reference, {"path", "sha256", "bytes"}, "source reference")
    if (Path(reference["path"]).resolve() != evidence.path or reference["sha256"] != evidence.sha256
            or reference["bytes"] != evidence.size_bytes):
        raise ValueError("report references a different raw artifact")


def _all_numeric_rows(app: Mapping[str, Any], config: Mapping[str, Any]) -> list[float]:
    """Recompute raw numeric gates. Frozen reference arithmetic remains an assumption."""
    if (app.get("schema") != "single-operator-surface-probe/v2"
            or app.get("timing_contract", {}).get("id") != "actual-backend-single-graph-envelope/v2"
            or app.get("status") != "measured" or app.get("control_mode") is not False):
        raise ValueError("successful full-call event raw required")
    for raw_name, config_name in (("M", "M"), ("N", "N"), ("K", "K"), ("weight_format", "quant")):
        _equal(app.get(raw_name), config[config_name], "raw shape/quant")
    for name, value in {"input_dtype": "F32", "output_dtype": "F32", "layout": "ordinary_contiguous_2d",
        "device": "cuda", "cuda_index": 0, "threads": 1, "graph_computations_per_batch": 1,
        "graph_compute_calls": 36, "warmup_requested": 5, "formal_repeats_requested": 30,
        "nvtx_enabled": True, "cache_policy": "untimed_read_write_sweep_at_least_4x_device_L2"}.items():
        _equal(app.get(name), value, "raw " + name)
    _equal(app.get("expected_source_path"), config["expected_source_path"], "raw source path")
    if app.get("environment", {}).get("GGML_CUDA_DISABLE_GRAPHS") != "1":
        raise ValueError("CUDA graph state differs")
    l2 = _positive_int(app.get("gpu_l2_bytes"), "raw L2")
    if _positive_int(app.get("cache_eviction_bytes"), "raw sweep") < max(128 << 20, 4 * l2):
        raise ValueError("cold cache evidence insufficient")
    modules = app.get("loaded_modules_before")
    if not isinstance(modules, (list, tuple)) or not modules or modules != app.get("loaded_modules_after"):
        raise ValueError("loaded native modules differ or absent")
    for module in modules:
        _sha(module.get("sha256"), "loaded module SHA")
        _positive_int(module.get("bytes"), "loaded module bytes")
        _text(module.get("path"), "loaded module path")
    cc = app.get("correctness_contract", {})
    for name, value in {"absolute_tolerance": .05, "relative_tolerance": .03,
                       "path_absolute_tolerance": .0001, "path_relative_tolerance": .00001,
                       "reference_mode": "dual_math_and_source_path"}.items():
        _equal(cc.get(name), value, "frozen numeric tolerance")
    count = min(4096, config["M"] * config["N"])
    def numeric(document: Any) -> None:
        document = _mapping(document, "raw numeric record")
        samples = document.get("samples")
        if (document.get("finite_all_outputs") is not True or document.get("passed") is not True
                or document.get("sample_count") != count or not isinstance(samples, (tuple, list)) or len(samples) != count):
            raise ValueError("raw numeric output/count rejected")
        math_passes, errors = [], []
        for i, row in enumerate(samples):
            for name in ("actual", "reference", "math_reference", "path_reference", "math_absolute_error", "path_absolute_error"):
                v = row.get(name)
                if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                    raise ValueError("raw numeric nonfinite sample")
            flat = i * (config["M"] * config["N"] - 1) // max(count - 1, 1)
            if row.get("m_index") != flat // config["N"] or row.get("n_index") != flat % config["N"]:
                raise ValueError("raw numerical sample positions differ")
            md, pd = abs(row["actual"] - row["reference"]), abs(row["actual"] - row["path_reference"])
            mp, pp = md <= .05 + .03 * abs(row["reference"]), pd <= .0001 + .00001 * abs(row["path_reference"])
            if (row["reference"] != row["math_reference"] or not math.isclose(md, row["math_absolute_error"], abs_tol=1e-12)
                    or not math.isclose(pd, row["path_absolute_error"], abs_tol=1e-12)
                    or row.get("math_pass") is not mp or row.get("path_pass") is not pp
                    or row.get("pass") is not pp or not pp):
                raise ValueError("raw numeric rederived gate failed")
            math_passes.append(mp); errors.append(pd)
        if document.get("math_passed") is not all(math_passes):
            raise ValueError("mathematical diagnostics not retained")
        if not math.isclose(document.get("path_max_absolute_error", -1), max(errors), abs_tol=1e-12):
            raise ValueError("raw numeric max not reproducible")
    numeric(app.get("first_call_correctness")); numeric(app.get("final_correctness"))
    runs = app.get("runs")
    if not isinstance(runs, (tuple, list)) or [(r.get("phase"), r.get("index")) for r in runs] != _EXPECTED_CALLS:
        raise ValueError("all first/warmup/formal calls required")
    frequency = _positive_int(app.get("qpc_frequency"), "QPC frequency")
    previous_end, formal = 0, []
    for run in runs:
        keys = ["qpc_pre_sync_start", "qpc_pre_sync_end", "qpc_evict_start", "qpc_evict_submit_end", "qpc_evict_end",
                "qpc_nvtx_push_start", "qpc_nvtx_push_end", "qpc_start", "qpc_record_begin_start", "qpc_record_begin_end",
                "qpc_submit_start", "qpc_submit_end", "qpc_record_end_start", "qpc_record_end_end", "qpc_wait_start",
                "qpc_wait_end", "qpc_end", "qpc_nvtx_pop_start", "qpc_nvtx_pop_end", "qpc_validation_start", "qpc_validation_end"]
        ticks = [_positive_int(run.get(k), k) for k in keys]
        if ticks != sorted(ticks) or ticks[0] < previous_end:
            raise ValueError("absolute QPC sequence rejected")
        previous_end = ticks[-1]
        if run.get("graph_computations") != 1 or any(run.get(k) != 0 for k in (
                "ggml_status", "cuda_submit_status", "cuda_wait_status", "eviction_status",
                "cuda_begin_record_status", "cuda_end_record_status", "cuda_query_after_wait", "cuda_elapsed_status")):
            raise ValueError("raw graph/kernel status failed")
        if run.get("cuda_query_before_wait") not in (0, 600):
            raise ValueError("CUDA event query failed")
        _positive_number(run.get("event_envelope_ms"), "event envelope")
        wall = (run["qpc_end"] - run["qpc_start"]) * 1e9 / frequency
        _close(run.get("host_wall_ns"), wall, "QPC-derived wall")
        _close(run.get("host_per_graph_ns"), wall, "single graph wall")
        expected_label = (f"operator_surface/v1|phase={run['phase']}|index={run['index']}|op=MUL_MAT|M={config['M']}"
            f"|N={config['N']}|K={config['K']}|quant={config['quant']}|input=F32|output=F32|layout=contiguous2d"
            f"|expected_path={config['expected_source_path']}")
        _equal(run.get("nvtx_label"), expected_label, "actual semantic marker")
        numeric(run.get("correctness"))
        if run["phase"] == "formal":
            formal.append(wall)
    return formal


def _trace_tables(sqlite_ref: KernelEvidenceFile, raw: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Independently read original SQLite; mapped/quality pass declarations are not trusted."""
    _reference_matches(raw.get("source_sqlite"), sqlite_ref)
    if raw.get("schema") != "operator-matrix-raw-nsys/v1" or raw.get("time_unit") != "ns":
        raise ValueError("original Nsight raw schema required")
    _verified_bytes(sqlite_ref.path, sqlite_ref.sha256, sqlite_ref.size_bytes, keep=False)
    tables = {}
    with closing(sqlite3.connect(sqlite_ref.path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds", "TARGET_INFO_GPU"}
        if not required <= names:
            raise ValueError("original SQLite required tables missing")
        for name in names:
            if name.startswith(("CUPTI_ACTIVITY_KIND_", "NVTX", "DIAGNOSTIC", "PROFILER_OVERHEAD", "TARGET_INFO_", "ENUM_DIAGNOSTIC")) or name in {"StringIds", "ENUM_NSYS_EVENT_CLASS"}:
                if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                    raise ValueError("unsupported SQLite table identifier")
                rows = [dict(r) for r in connection.execute('SELECT rowid AS evidence_rowid,* FROM "' + name + '"')]
                for row in rows:
                    for key, value in row.items():
                        if isinstance(value, bytes):
                            row[key] = {"binary_hex": value.hex()}
                tables[name] = rows
    strings = {r["id"]: r["value"] for r in tables["StringIds"]}
    for row in tables["NVTX_EVENTS"]:
        row["resolved_text"] = strings.get(row.get("textId"), row.get("text"))
    for row in tables["CUPTI_ACTIVITY_KIND_RUNTIME"]:
        row["name_text"] = strings.get(row["nameId"])
    for row in tables["CUPTI_ACTIVITY_KIND_KERNEL"]:
        row["demangled_name_text"] = strings.get(row["demangledName"], "")
        row["short_name_text"] = strings.get(row["shortName"], "")
    _equal(raw.get("tables"), tables, "retained raw events versus original SQLite")
    _verified_bytes(sqlite_ref.path, sqlite_ref.sha256, sqlite_ref.size_bytes, keep=False)
    return tables


def _observed_device(tables: Mapping[str, Any], device_id: int) -> dict[str, Any]:
    rows = [r for r in tables["TARGET_INFO_GPU"] if r.get("id") == device_id]
    if len(rows) != 1:
        raise ValueError("unique actual GPU observation required")
    row = rows[0]
    uuid = _text(row.get("uuid"), "observed GPU UUID")
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    major, minor = _positive_int(row.get("computeMajor"), "compute major"), row.get("computeMinor")
    if type(minor) is not int or minor < 0:
        raise ValueError("compute minor invalid")
    return {"gpu_uuid": uuid, "sm_count": _positive_int(row.get("smCount"), "observed SM count"),
            "l2_bytes": _positive_int(row.get("l2CacheSize"), "observed L2"), "compute_capability": 100 * major + minor}


def _correlate_kernel_samples(tables: Mapping[str, Any], app: Mapping[str, Any], config: Mapping[str, Any],
                              key: Mapping[str, Any]) -> tuple[list[float], list[float], dict[str, Any]]:
    markers, apis, kernels = (tables[n] for n in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL"))
    wanted = {r["nvtx_label"]: (r["phase"], r["index"]) for r in app["runs"]}
    selected = [m for m in markers if m.get("resolved_text") in wanted]
    if len(selected) != 36 or len({m["resolved_text"] for m in selected}) != 36:
        raise ValueError("36 unique actual NVTX calls required")
    formal, totals, used, ownership, call_order = [], [], set(), set(), []
    for marker in sorted(selected, key=lambda m: m["start"]):
        begin, end, tid = marker["start"], marker["end"], marker["globalTid"]
        if type(begin) is not int or type(end) is not int or begin >= end:
            raise ValueError("invalid actual NVTX time")
        owned = [a for a in apis if a["globalTid"] == tid and begin <= a["start"] and a["end"] <= end]
        if any(a.get("returnValue") != 0 and not (re.fullmatch(r"cudaEventQuery(?:_v[0-9]+)?", str(a.get("name_text", ""))) is not None and a.get("returnValue") == 600) for a in owned):
            raise ValueError("CUDA runtime API failed")
        launches = [a for a in owned if "LaunchKernel" in str(a.get("name_text", ""))]
        mapped = []
        for launch in launches:
            matches = [k for k in kernels if k["globalPid"] == (tid & _PROCESS_MASK) and k["correlationId"] == launch["correlationId"]]
            if len(matches) != 1:
                raise ValueError("unique launch/kernel correlation required")
            kernel = matches[0]
            if kernel["evidence_rowid"] in used:
                raise ValueError("kernel belongs to multiple calls")
            used.add(kernel["evidence_rowid"])
            if not begin <= kernel["start"] < kernel["end"] <= end:
                raise ValueError("kernel outside full-call marker")
            ownership.add((kernel["deviceId"], kernel["streamId"], kernel["globalPid"]))
            mapped.append(kernel)
        target_devices = {k["deviceId"] for k in mapped}
        owned_rows = {k["evidence_rowid"] for k in mapped}
        for other in kernels:
            if (other["deviceId"] in target_devices and other["evidence_rowid"] not in owned_rows
                    and other["start"] < end and other["end"] > begin):
                raise ValueError("extra overlapping same-device GPU kernel activity")
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            for other in tables.get(table, []):
                if other.get("deviceId") in target_devices and other["start"] < end and other["end"] > begin:
                    raise ValueError("extra overlapping same-device GPU memory activity")
        mapped.sort(key=lambda k: (k["start"], k["evidence_rowid"]))
        names = [k["short_name_text"] for k in mapped]
        if config["expected_source_path"] == "MMVQ_Q8_1_HALF":
            if names != ["quantize_q8_1", "mul_mat_vec_q"]:
                raise ValueError("actual MMVQ chain differs")
        elif config["expected_source_path"] == "MMQ_Q8_1_D4_F32":
            if names not in (["quantize_mmq_q8_1", "mul_mat_q"], ["quantize_mmq_q8_1", "mul_mat_q", "mul_mat_q_stream_k_fixup"]):
                raise ValueError("actual MMQ chain differs")
            if not re.search(r"\(mmq_q8_1_ds_layout\)0\s*,", mapped[0]["demangled_name_text"]):
                raise ValueError("actual MMQ D4 layout not confirmed")
        else:
            raise ValueError("unsupported source path")
        if any(first["end"] > second["start"] for first, second in zip(mapped, mapped[1:])):
            raise ValueError("overlapping kernels contradict single stream")
        owned_ids = {(tid & _PROCESS_MASK, a["correlationId"]) for a in owned}
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if any((event.get("globalPid"), event.get("correlationId")) in owned_ids for event in tables.get(table, [])):
                raise ValueError("unexpected GPU memory activity")
        requested = []
        for kernel in mapped:
            role, family = _ROLES.get(kernel["short_name_text"], (None, None))
            if role in {"main", "fixup"} and not re.search(r"\(ggml_type\)" + str({"Q5_0": 6, "Q8_0": 8}[config["quant"]]) + r"\s*,", kernel["demangled_name_text"]):
                raise ValueError("actual kernel quantization differs")
            geometry = {"grid": [kernel[n] for n in ("gridX", "gridY", "gridZ")],
                        "block": [kernel[n] for n in ("blockX", "blockY", "blockZ")],
                        "static_shared_bytes": kernel["staticSharedMemory"], "dynamic_shared_bytes": kernel["dynamicSharedMemory"]}
            if role == key["role"]:
                if family != key["kernel_family"] or kernel["demangled_name_text"] != key["kernel_variant"]:
                    raise ValueError("kernel family/variant key not observed")
                _equal(geometry, key["launch_geometry"], "actual launch geometry")
                requested.append(kernel["end"] - kernel["start"])
        if len(requested) != 1:
            raise ValueError("requested role not uniquely observed in every call")
        phase, index = wanted[marker["resolved_text"]]
        call_order.append((phase, index))
        if phase == "formal":
            formal += requested
            totals.append(sum(k["end"] - k["start"] for k in mapped))
    if call_order != _EXPECTED_CALLS or len(ownership) != 1:
        raise ValueError("actual call order/device/stream mismatch")
    severities = {}
    for row in tables.get("ENUM_DIAGNOSTIC_SEVERITY_LEVEL", []):
        identity = row.get("id")
        label = row.get("label", row.get("name"))
        if identity in severities or type(identity) is not int or not isinstance(label, str):
            raise ValueError("invalid or ambiguous profiler diagnostic severity enum")
        severities[identity] = label
    for row in tables.get("DIAGNOSTIC_EVENT", []):
        # Resolve against this trace's original enum. Preserve all raw messages;
        # only exactly-known informational levels are nonblocking.
        severity = severities.get(row.get("severity"))
        if severity not in {"Info", "Verbose"}:
            raise ValueError("profiler diagnostic warning/error/unknown severity rejected")
    device_id, stream_id, _process_id = next(iter(ownership))
    return formal, totals, {"device_id": device_id, "stream_id": stream_id, "hardware": _observed_device(tables, device_id)}



def _evidence_json_limit(kind: str) -> int:
    # 38*4096 numerical rows with long F32/F64 strings fit <128MiB.
    # Keep profile/owner/config limits tight; aggregate retained JSON is bounded too.
    return {"raw_application": 128 << 20, "raw_events": 128 << 20,
            "quality_report": 64 << 20, "telemetry": 64 << 20}.get(kind, 16 << 20)


def _validate_pair_independence(pairs: Any, evidence: Mapping[str, KernelEvidenceFile]) -> None:
    seen_paths, sqlite_hashes, identities = set(), set(), set()
    for pair in pairs:
        for field in ("raw_events_id", "raw_sqlite_id", "profile_app_id", "direct_app_id",
                      "profile_observation_id", "direct_observation_id", "profile_spec_id", "direct_spec_id",
                      "profile_start_id", "direct_start_id", "profile_telemetry_id", "direct_telemetry_id"):
            item = evidence.get(pair.get(field))
            if item is None or item.path in seen_paths:
                raise ValueError("process pair reuses or lacks an original evidence artifact")
            seen_paths.add(item.path)
            if field == "raw_sqlite_id":
                if item.sha256 in sqlite_hashes:
                    raise ValueError("duplicate original SQLite sample source")
                sqlite_hashes.add(item.sha256)
        for mode in ("profile", "direct"):
            receipt = _document(evidence, pair[mode + "_observation_id"], "hardware_runtime_observation")
            identity = (receipt.get("process_pid"), receipt.get("utc_started"), receipt.get("qpc_launch_start"))
            if (type(identity[0]) is not int or identity[0] <= 0 or not isinstance(identity[1], str)
                    or not identity[1] or type(identity[2]) is not int or identity[2] <= 0 or identity in identities):
                raise ValueError("independent native process identity required")
            identities.add(identity)


def _validate_collection_freeze(freeze: Mapping[str, Any], protocol_ref: KernelEvidenceFile) -> None:
    _reference_matches(freeze.get("protocol_ref"), protocol_ref)
    references = []
    for group in ("files", "probe_files", "critical_tool_files"):
        items = freeze.get(group)
        if not isinstance(items, (tuple, list)) or not items:
            raise ValueError("frozen source/tool identity set missing")
        references.extend(items)
    for item in references:
        item = _fields(item, {"path", "bytes", "sha256"}, "frozen file")
        _verified_bytes(Path(item["path"]).resolve(), item["sha256"], item["bytes"], keep=False)
    if not any(Path(r["path"]).resolve() == protocol_ref.path and r["sha256"] == protocol_ref.sha256 for r in references):
        raise ValueError("collection protocol absent from frozen closure")


def _validate_clock_receipt(clock: Mapping[str, Any], uuid: str, evidence: Mapping[str, KernelEvidenceFile]) -> None:
    for field, value in {"schema": "operator-clock-control-receipt/v1", "gpu_uuid": uuid,
        "target_sm_clock_mhz": 2400, "sm_clock_tolerance_mhz": 30, "requested_lock_min_mhz": 2400,
        "requested_lock_max_mhz": 2400, "lock_command_returncode": 0, "restore_on_exit_planned": True}.items():
        _equal(clock.get(field), value, "clock control receipt")
    command = clock.get("command")
    if not isinstance(command, (tuple, list)) or "-lgc" not in command or "2400,2400" not in command:
        raise ValueError("clock lock command not bound")
    _text(clock.get("created_utc"), "clock session start")
    for name in ("stdout_ref", "stderr_ref"):
        reference = clock.get(name)
        matches = [r for r in evidence.values() if r.kind == "tool_log" and isinstance(reference, Mapping)
                   and r.path == Path(reference.get("path", "")).resolve()]
        if len(matches) != 1:
            raise ValueError("immutable clock command log missing")
        _reference_matches(reference, matches[0])


def _validate_process_binding(receipt: Mapping[str, Any], start: Mapping[str, Any], spec: Mapping[str, Any],
        spec_ref: KernelEvidenceFile, freeze_ref: KernelEvidenceFile, clock_ref: KernelEvidenceFile) -> None:
    _reference_matches(receipt.get("spec_ref"), spec_ref)
    _reference_matches(start.get("spec"), spec_ref)
    if receipt.get("external_approved_sha256") != freeze_ref.sha256:
        raise ValueError("process not bound to externally approved freeze")
    for boundary in (start.get("freeze_before"), receipt.get("freeze_after")):
        boundary = _mapping(boundary, "process freeze boundary")
        if boundary.get("passed") is not True or boundary.get("external_approved_sha256") != freeze_ref.sha256 or boundary.get("required_sets_complete") is not True:
            raise ValueError("process freeze validation failed")
        _reference_matches(boundary.get("freeze_ref"), freeze_ref)
    if Path(spec.get("freeze", "")).resolve() != freeze_ref.path:
        raise ValueError("process spec references different freeze")
    for document in (receipt, spec):
        _reference_matches(document.get("clock_control_binding", {}).get("receipt_ref"), clock_ref)
    if receipt.get("child_process_exited") is not True or receipt.get("telemetry_errors") not in ([], ()):
        raise ValueError("native process completion/telemetry failed")
    if receipt.get("qpc_process_complete", 0) <= receipt.get("qpc_launch_start", 0):
        raise ValueError("native process lifetime missing")


def _clock_readback_gate(app: Mapping[str, Any], samples: Any) -> dict[str, Any]:
    if not isinstance(samples, (tuple, list)) or not samples:
        raise ValueError("original telemetry samples missing")
    formal = [r for r in app["runs"] if r["phase"] == "formal"]
    frequency = _positive_int(app["qpc_frequency"], "clock QPC frequency")
    ordered = sorted((s for s in samples if type(s.get("qpc_ticks")) is int), key=lambda r: r["qpc_ticks"])
    if len({s["qpc_ticks"] for s in ordered}) != len(ordered):
        raise ValueError("duplicate telemetry sampling timestamps")
    used = set()
    for row in formal:
        begin, end = row["qpc_start"], row["qpc_end"]
        before = [i for i, r in enumerate(ordered) if r["qpc_ticks"] <= begin]
        after = [i for i, r in enumerate(ordered) if r["qpc_ticks"] >= end]
        if not before or not after:
            raise ValueError("formal interval not bracketed by measured clock")
        lo, hi = before[-1], after[0]
        if begin - ordered[lo]["qpc_ticks"] > frequency * .025 or ordered[hi]["qpc_ticks"] - end > frequency * .025:
            raise ValueError("formal clock bracket wider than frozen25ms")
        used.update(range(lo, hi + 1))
    values = []
    for index in used:
        reading = ordered[index].get("sm_mhz", {})
        if reading.get("status") != 0 or type(reading.get("value")) is not int or abs(reading["value"] - 2400) > 30:
            raise ValueError("measured SM clock outside frozen2400+/-30 domain")
        values.append(reading["value"])
    if len(formal) != 30 or not values:
        raise ValueError("complete formal clock sample coverage required")
    return {"target_sm_clock_mhz": 2400, "tolerance_mhz": 30, "formal_intervals": 30,
            "bracketed_intervals": 30, "sm_clock_min_mhz": min(values), "sm_clock_max_mhz": max(values)}

def _validate_measurement_entry(item: Mapping[str, Any], evidence: Mapping[str, KernelEvidenceFile],
                                hardware: Mapping[str, Any], runtime: Mapping[str, Any], cache: Mapping[str, Any]) -> None:
    bundle_ids = [value for value in item["evidence_ids"] if evidence[value].kind == "measurement_bundle"]
    if len(bundle_ids) != 1:
        raise ValueError("accepted entry requires one independently verifiable measurement bundle")
    bundle = _fields(_document(evidence, bundle_ids[0], "measurement_bundle"), {
        "schema", "config", "collection_protocol_id", "effective_hardware_id", "pairs",
        "numeric_report_id", "quality_report_id", "dispatch_source_id", "collection_freeze_id",
        "approved_freeze_sha256", "clock_receipt_id"}, "measurement bundle")
    if bundle["schema"] != _BUNDLE_SCHEMA:
        raise ValueError("unsupported measurement bundle")
    config = _fields(bundle["config"], {"id", "group", "quant", "M", "N", "K", "expected_source_path"}, "collection config")
    if config["group"] != "training":
        raise ValueError("validation/aligned holdouts cannot become calibration entries")
    protocol = _document(evidence, bundle["collection_protocol_id"], "protocol")
    if protocol.get("schema") != "operator-matrix-collection-protocol/v2" or config not in protocol.get("configs", []):
        raise ValueError("config absent from frozen collection protocol")
    policy = protocol.get("quality_policy", {})
    for field, value in {"process_pairs": 3, "formal_calls_per_process": 30,
        "kernel_formal_p90_div_p10_max": 1.5, "profile_process_median_max_relative_deviation": .05,
        "profile_direct_host_median_max_relative_difference": .2, "direct_host_formal_p90_div_p10_max": 1.5, "target_sm_clock_mhz": 2400, "sm_clock_tolerance_mhz": 30,
        "maximum_formal_clock_bracket_gap_ms": 25, "every_formal_interval_bracketed_required": True}.items():
        _equal(policy.get(field), value, "frozen collection quality policy")
    effective = _document(evidence, bundle["effective_hardware_id"], "effective_hardware_profile")
    digest = effective_hardware_sha256(effective)
    if digest != hardware["effective_profile_sha256"]:
        raise ValueError("effective hardware canonical hash differs")
    _equal(effective["observed_device"], {name: hardware[name] for name in _FACT_FIELDS}, "effective hardware observation")
    key = item["key"]
    _equal([key[n] for n in ("m", "n", "k_logical")], [config[n] for n in ("M", "N", "K")], "physical shape")
    if (key["weight_format"] != config["quant"] or key["activation_dtype"] != "F32" or key["output_dtype"] != "F32"
            or key["accumulator_dtype"] != "F32" or key["layout"] != "contiguous_2d" or config["K"] % 32):
        raise ValueError("unsupported physical tensor semantics")
    row_bytes = config["K"] // 32 * {"Q5_0": 22, "Q8_0": 34}[config["quant"]]
    _equal(key["strides"], {"activation_bytes": [4, 4 * config["K"]],
        "weight_bytes": [{"Q5_0": 22, "Q8_0": 34}[config["quant"]], row_bytes],
        "output_bytes": [4, 4 * config["N"]]}, "actual contiguous stride semantics")
    source = evidence.get(bundle["dispatch_source_id"])
    if source is None or source.kind != "native_source" or source.sha256 not in runtime["source_sha256"].values():
        raise ValueError("locked dispatch source missing")
    # Traces expose no arguments. Support only MMVQ main logical K in v2;
    # conversion/MMQ/fixup require an independent executed-work contract.
    if key["role"] != "main" or config["expected_source_path"] != "MMVQ_Q8_1_HALF":
        raise ValueError("physical executed K unproven for this role/path; retain rejected")
    if source.path.name != "mmvq.cu":
        raise ValueError("MMVQ logical K rule requires locked mmvq.cu")
    source_text = _verified_bytes(source.path, source.sha256, source.size_bytes, keep=True).decode("utf-8-sig")
    if (not re.search(r"const\s+int\s+blocks_per_row_x\s*=\s*ncols_x\s*/\s*qk\s*;", source_text)
            or not re.search(r"for\s*\(int kbx\s*=.*?;\s*kbx < blocks_per_row_x;\s*kbx \+= blocks_per_iter\)", source_text)
            or "mul_mat_vec_q" not in source_text):
        raise ValueError("locked MMVQ reduction source anchors missing")
    if key["k_executed"] != key["k_logical"] or key["dispatch_signature"] != "mmvq-logical-K/v1:" + source.sha256:
        raise ValueError("executed K not bound to locked source rule")
    pairs = bundle["pairs"]
    if not isinstance(pairs, (list, tuple)) or len(pairs) != 3 or {p.get("pair") for p in pairs} != {0, 1, 2}:
        raise ValueError("three independent process pairs required")
    numerical = _document(evidence, bundle["numeric_report_id"], "numerical_validation")
    quality = _document(evidence, bundle["quality_report_id"], "quality_report")
    if numerical.get("schema") != "heterollm.kernel-numeric-recheck/v1":
        raise ValueError("numeric report schema missing")
    if item["gates"]["numerical"]["evidence_id"] != bundle["numeric_report_id"] or any(
            item["gates"][gate]["evidence_id"] != bundle["quality_report_id"] for gate in ("stability", "profiling")):
        raise ValueError("profile gates cite different measurement reports")
    _equal(quality.get("config"), config, "quality config")
    if quality.get("measurement_cost_eligible") is not True or _plain(quality.get("issues")) != []:
        raise ValueError("source quality report rejects measurements")
    _equal(quality.get("fixed_policy"), policy, "quality frozen policy")
    reported_pairs = quality.get("pairs", [])
    if len(reported_pairs) != 3 or {p.get("pair") for p in reported_pairs} != {0, 1, 2}:
        raise ValueError("quality report requires three unique pair records")
    reports = {p["pair"]: p for p in reported_pairs}
    _validate_pair_independence(pairs, evidence)
    freeze_ref = evidence.get(bundle["collection_freeze_id"])
    if freeze_ref is None or freeze_ref.kind != "collection_freeze" or freeze_ref.sha256 != _sha(bundle["approved_freeze_sha256"], "external approved freeze"):
        raise ValueError("external approved collection freeze not bound")
    freeze = _document(evidence, bundle["collection_freeze_id"], "collection_freeze")
    _validate_collection_freeze(freeze, evidence[bundle["collection_protocol_id"]])
    clock_ref = evidence.get(bundle["clock_receipt_id"])
    if clock_ref is None or clock_ref.kind != "clock_receipt":
        raise ValueError("clock receipt absent")
    clock = _document(evidence, bundle["clock_receipt_id"], "clock_receipt")
    _validate_clock_receipt(clock, hardware["gpu_uuid"], evidence)
    numeric_refs, all_samples, process_medians, signatures, streams = [], [], [], [], set()

    for pair in pairs:
        pair = _fields(pair, {"pair", "raw_events_id", "raw_sqlite_id", "profile_app_id", "direct_app_id",
            "profile_observation_id", "direct_observation_id", "profile_stdout_id", "profile_stderr_id",
            "export_stdout_id", "export_stderr_id", "profile_telemetry_id", "direct_telemetry_id",
            "profile_spec_id", "direct_spec_id", "profile_start_id", "direct_start_id"}, "measurement pair")
        profile = _document(evidence, pair["profile_app_id"], "raw_application")
        direct = _document(evidence, pair["direct_app_id"], "raw_application")
        ph, dh = _all_numeric_rows(profile, config), _all_numeric_rows(direct, config)
        clock_gates = {}
        for mode, application in (("profile", profile), ("direct", direct)):
            telemetry = _document(evidence, pair[mode + "_telemetry_id"], "telemetry")
            clock_gates[mode] = _clock_readback_gate(application, telemetry.get("samples", []))
        numeric_refs += [evidence[pair["profile_app_id"]].sha256, evidence[pair["direct_app_id"]].sha256]
        signature = lambda app: {"quantization": app.get("quantization"), "environment": app.get("environment"),
                                 "modules": app.get("loaded_modules_before")}
        _equal(signature(profile), signature(direct), "paired input/runtime bytes")
        signatures.append(signature(profile))
        module_hashes = {Path(m["path"]).name.casefold(): m["sha256"] for m in profile["loaded_modules_before"]}
        if any(module_hashes.get(Path(name).name.casefold()) != sha for name, sha in runtime["dll_sha256"].items()):
            raise ValueError("loaded native DLL differs from runtime profile")
        for name, value in runtime["kernel_environment"].items():
            _equal(profile["environment"].get(name), value, "observed kernel environment")
        raw = _document(evidence, pair["raw_events_id"], "raw_events")
        sqlite_ref = evidence.get(pair["raw_sqlite_id"])
        if sqlite_ref is None or sqlite_ref.kind != "raw_sqlite":
            raise ValueError("original SQLite evidence required")
        tables = _trace_tables(sqlite_ref, raw)
        samples, totals, observed = _correlate_kernel_samples(tables, profile, config, key)
        streams.add((observed["device_id"], observed["stream_id"]))
        _equal(observed["hardware"], effective["observed_device"], "SQLite hardware versus effective profile")
        _equal(profile.get("gpu_l2_bytes"), observed["hardware"]["l2_bytes"], "CUDA versus SQLite L2")
        _equal(cache["l2_bytes"], observed["hardware"]["l2_bytes"], "cache versus SQLite L2")
        _equal(cache["sweep_bytes"], profile.get("cache_eviction_bytes"), "actual cache sweep")
        if 100 * profile.get("compute_capability_major", -1) + profile.get("compute_capability_minor", -1) != observed["hardware"]["compute_capability"]:
            raise ValueError("CUDA versus SQLite compute capability")
        for kind in ("profile", "direct"):
            receipt = _document(evidence, pair[kind + "_observation_id"], "hardware_runtime_observation")
            if receipt.get("status") != "completed" or receipt.get("returncode") != 0:
                raise ValueError("native process receipt failed")
            identity = receipt.get("gpu_identity", {})
            if identity.get("uuid") != observed["hardware"]["gpu_uuid"] or identity.get("driver_version") != runtime["driver_version"]:
                raise ValueError("runtime receipt hardware/driver identity differs")
            app_ref = evidence[pair[kind + "_app_id"]]
            spec_ref = evidence[pair[kind + "_spec_id"]]
            start = _document(evidence, pair[kind + "_start_id"], "process_start")
            spec = _document(evidence, pair[kind + "_spec_id"], "process_spec")
            _validate_process_binding(receipt, start, spec, spec_ref, freeze_ref, clock_ref)
            telemetry_ref = evidence[pair[kind + "_telemetry_id"]]
            if not any(r.get("sha256") == telemetry_ref.sha256 and Path(r.get("path", "")).resolve() == telemetry_ref.path for r in receipt.get("artifacts", [])):
                raise ValueError("receipt does not bind original telemetry")
            if not any(r.get("sha256") == app_ref.sha256 and Path(r.get("path", "")).resolve() == app_ref.path
                       for r in receipt.get("artifacts", [])):
                raise ValueError("process receipt does not bind application raw")
        for log_name in ("profile_stdout_id", "profile_stderr_id", "export_stdout_id", "export_stderr_id"):
            log = evidence[pair[log_name]]
            if log.kind != "tool_log":
                raise ValueError("profiler/export diagnostics absent")
            text = _verified_bytes(log.path, log.sha256, log.size_bytes, keep=True).decode("utf-8", errors="replace")
            if re.search(r"\b(warning|error|fatal|unsupported|incompatible)\b", text, re.I):
                raise ValueError("profiler/export reported diagnostic")
        if (_distribution(samples)["p90_div_p10"] > 1.5 or _distribution(totals)["p90_div_p10"] > 1.5
                or _distribution(dh)["p90_div_p10"] > 1.5):
            raise ValueError("independently recomputed stability gate failed")
        ratio = abs(statistics.median(ph) - statistics.median(dh)) / statistics.median(dh)
        if ratio > .2:
            raise ValueError("independently recomputed profiling gate failed")
        report = reports.get(pair["pair"])
        if report is None or any(report.get(k) is not True for k in ("numerics_all_rows", "trace_chain_complete", "source_path_matches_observed_family", "trace_warning_free")):
            raise ValueError("quality pair rejects source evidence")
        if report.get("clock_domain_validated") is not True:
            raise ValueError("reported clock domain failed")
        for mode in ("profile", "direct"):
            gate = report.get("clock_domain_gates", {}).get(mode, {})
            if gate.get("passed") is not True or gate.get("issues") not in ([], ()):
                raise ValueError("reported clock gate failed")
            for field in ("target_sm_clock_mhz", "tolerance_mhz", "formal_intervals", "bracketed_intervals", "sm_clock_min_mhz", "sm_clock_max_mhz"):
                _equal(gate.get(field), clock_gates[mode][field], "recomputed clock report")
        _check_distribution(report.get("profile_kernel"), totals, "reported profile kernel union")
        _check_distribution(report.get("profile_host"), ph, "reported profile wall")
        _check_distribution(report.get("direct_host"), dh, "reported direct wall")
        for kind in ("profile", "direct"):
            _reference_matches(report.get(kind + "_raw", {}).get("source"), evidence[pair[kind + "_app_id"]])
        all_samples += samples
        process_medians.append(statistics.median(samples))
    if len(streams) != 1 or any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("actual stream/input/runtime differs across pairs")
    center = statistics.median(process_medians)
    if max(abs(v - center) / center for v in process_medians) > .05:
        raise ValueError("across-process kernel stability gate failed")
    _equal(numerical.get("raw_application_sha256"), numeric_refs, "numeric report source list")
    if numerical.get("all_calls") != 216 or numerical.get("failed_rows") != 0:
        raise ValueError("numeric report sample denominator or failures differs")
    if item["sample_count"] != len(all_samples):
        raise ValueError("entry sample count differs from original kernel intervals")
    _close(item["device_ns"], statistics.median(all_samples), "entry median device ns")


def load_kernel_calibration(path: str | Path, *, expected_sha256: str) -> KernelCalibrationProfile:
    """Load one frozen profile and verify all referenced evidence bytes.

    Accepted rows require independent SQLite correlation, all raw numerical
    gates and paired quality recomputation. Missing semantic evidence rejects.
    """
    path = Path(path).resolve()
    data = _fields(_json(_verified_bytes(path, expected_sha256, None, keep=True)), {
        "schema", "profile_id", "source_kind", "target_llm_latency_used", "measurement_boundary",
        "hardware", "runtime", "cache", "evidence_files", "entries",
    }, "profile")
    if data["schema"] != PROFILE_SCHEMA or data["source_kind"] != SOURCE_KIND:
        raise ValueError("unsupported profile schema or non-independent source kind")
    if data["target_llm_latency_used"] is not False:
        raise ValueError("target LLM latency use must be explicitly false")
    if data["measurement_boundary"] != DEVICE_BOUNDARY:
        raise ValueError("only individual CUDA device-kernel intervals are accepted")
    profile_id = _text(data["profile_id"], "profile_id")
    hardware, runtime, cache = _identity_facts(data)
    refs = data["evidence_files"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("evidence_files must not be empty")
    evidence: dict[str, KernelEvidenceFile] = {}
    retained_bytes = 0
    seen_paths: set[Path] = set()
    for reference in refs:
        reference = _fields(reference, {"id", "kind", "path", "sha256", "bytes"}, "evidence reference")
        evidence_id = _text(reference["id"], "evidence id")
        if evidence_id in evidence:
            raise ValueError("duplicate evidence id")
        kind = _text(reference["kind"], "evidence kind")
        if kind not in _REQUIRED_EVIDENCE_KINDS | {"execution_contract"}:
            raise ValueError("unsupported evidence kind")
        ref_path = Path(_text(reference["path"], "evidence path"))
        ref_path = (path.parent / ref_path).resolve() if not ref_path.is_absolute() else ref_path.resolve()
        size = reference["bytes"]
        if type(size) is not int or size < 0 or (kind != "tool_log" and size == 0):
            raise ValueError("evidence bytes must be a positive integer (logs may be empty)")
        sha = _sha(reference["sha256"], "evidence SHA")
        json_kind = kind in {"execution_contract", "protocol", "measurement_bundle", "raw_events", "raw_application",
            "numerical_validation", "quality_report", "hardware_runtime_observation", "effective_hardware_profile",
            "telemetry", "clock_receipt", "process_spec", "process_start", "collection_freeze"}
        if ref_path in seen_paths:
            raise ValueError("duplicate evidence path must reuse its existing id")
        seen_paths.add(ref_path)
        if json_kind:
            retained_bytes += size
            if retained_bytes > 1024 * 1024 * 1024:
                raise ValueError("semantic JSON aggregate byte budget exceeded")
        payload = _verified_bytes(ref_path, sha, size, keep=json_kind, max_bytes=_evidence_json_limit(kind))
        evidence[evidence_id] = KernelEvidenceFile(
            evidence_id, kind, ref_path, sha, size,
            _freeze(_json(payload)) if json_kind else None,
        )
    if not _REQUIRED_EVIDENCE_KINDS <= {item.kind for item in evidence.values()}:
        raise ValueError("required evidence kinds are missing")
    evidence_by_kind = {kind: {item.sha256 for item in evidence.values() if item.kind == kind}
                        for kind in _REQUIRED_EVIDENCE_KINDS}
    effective_hashes = {effective_hardware_sha256(item.document) for item in evidence.values()
                        if item.kind == "effective_hardware_profile" and item.document is not None}
    if hardware["effective_profile_sha256"] not in effective_hashes:
        raise ValueError("effective hardware profile SHA lacks matching file evidence")
    for field, kind in (("dll_sha256", "runtime_binary"), ("source_sha256", "native_source")):
        if not set(runtime[field].values()) <= evidence_by_kind[kind]:
            raise ValueError("runtime identity has unverified DLL or source hashes")
    raw_entries = data["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("entries must not be empty")
    entries: dict[str, KernelEntry] = {}
    for item in raw_entries:
        item = _fields(item, {"key", "status", "device_ns", "sample_count", "rejection_reason", "evidence_ids", "gates"}, "entry")
        key = canonical_kernel_key(item["key"])
        if key in entries:
            raise ValueError("duplicate exact kernel key")
        key_data = item["key"]
        if (key_data["effective_hardware_sha256"] != hardware["effective_profile_sha256"]
                or key_data["runtime_sha256"] != runtime["profile_sha256"]
                or key_data["cache_protocol"] != cache["protocol"]):
            raise ValueError("entry hardware/runtime/cache facts do not match its profile")
        status = _text(item["status"], "entry status")
        if status not in {"accepted", "rejected"}:
            raise ValueError("entry must be accepted or retained as rejected")
        reason = item["rejection_reason"]
        if status == "accepted" and reason is not None:
            raise ValueError("accepted entry cannot carry a rejection reason")
        if status == "rejected":
            _text(reason, "rejection_reason")
        duration = None if item["device_ns"] is None else _positive_number(item["device_ns"], "device_ns")
        count = item["sample_count"]
        if type(count) is not int or count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if status == "accepted" and (duration is None or count < 2):
            raise ValueError("accepted entries require a duration and repeated measurements")
        ids = item["evidence_ids"]
        if (not isinstance(ids, list) or not ids or any(not isinstance(value, str) for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("entry evidence ids must be non-empty and unique")
        if any(not isinstance(value, str) or value not in evidence for value in ids):
            raise ValueError("entry references unknown evidence")
        if not any(evidence[value].kind == "raw_events" for value in ids):
            raise ValueError("entry must reference raw events")
        gates = _fields(item["gates"], set(_GATE_KINDS), "entry gates")
        for gate_name, expected_kind in _GATE_KINDS.items():
            gate = _fields(gates[gate_name], {"passed", "evidence_id"}, gate_name)
            if type(gate["passed"]) is not bool:
                raise ValueError("gate decisions must be explicit booleans")
            if status == "accepted" and gate["passed"] is not True:
                raise ValueError("accepted entry has a failed quality gate")
            gate_id = gate["evidence_id"]
            if not isinstance(gate_id, str) or gate_id not in ids or evidence[gate_id].kind != expected_kind:
                raise ValueError("quality gate lacks correctly typed evidence")
        if status == "accepted":
            try:
                _validate_measurement_entry(item, evidence, hardware, runtime, cache)
            except (KeyError, TypeError, AttributeError, sqlite3.Error) as error:
                raise ValueError("incomplete semantic measurement evidence: " + str(error)) from error
        entries[key] = KernelEntry(key, status, duration, count, reason, tuple(ids), _freeze(gates))
    return KernelCalibrationProfile(profile_id, expected_sha256, _freeze(hardware), _freeze(runtime),
                                    _freeze(cache), MappingProxyType(entries), MappingProxyType(evidence), True, True)


def resolve_kernel_calibration(profile: KernelCalibrationProfile | None,
                               query: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve exact joint coverage, retaining rejected entries and reasons."""
    result: dict[str, Any] = {"mode": "analytical_fallback", "reason": None, "device_ns": None,
                              "requested_key": None, "profile_sha256": None, "evidence_ids": []}
    if profile is None:
        result["reason"] = "profile_disabled"
        return result
    if not isinstance(profile, KernelCalibrationProfile) or profile.evidence_files_verified is not True or profile.semantic_evidence_verified is not True:
        result["reason"] = "profile_evidence_not_verified"
        return result
    result["profile_sha256"] = profile.sha256
    try:
        key = canonical_kernel_key(query)
    except (TypeError, ValueError) as error:
        result.update(reason="invalid_kernel_query", detail=str(error))
        return result
    result["requested_key"] = key
    for field, expected, reason in (
        ("effective_hardware_sha256", profile.hardware["effective_profile_sha256"], "effective_hardware_mismatch"),
        ("runtime_sha256", profile.runtime["profile_sha256"], "runtime_mismatch"),
        ("cache_protocol", profile.cache["protocol"], "cache_protocol_mismatch"),
    ):
        if query[field] != expected:
            result["reason"] = reason
            return result
    entry = profile.entries.get(key)
    if entry is None:
        result["reason"] = "unseen_joint_kernel_shape"
        return result
    result["evidence_ids"] = list(entry.evidence_ids)
    if entry.status != "accepted":
        result.update(reason="entry_rejected", detail=entry.rejection_reason)
        return result
    result.update(mode="exact", reason=None, device_ns=entry.device_ns, sample_count=entry.sample_count)
    return result


def _native_conflict(metadata: Mapping[str, Any]) -> bool:
    for name, value in metadata.items():
        if (str(name).startswith("native_") and str(name).endswith("calibration_applied")
                and value is not False and value is not None):
            return True
        if isinstance(value, Mapping) and _native_conflict(value):
            return True
    return False


def _owner(profile: KernelCalibrationProfile, evidence_id: str | None,
           query: Mapping[str, Any], phase: CostPhase) -> tuple[Mapping[str, Any] | None, str | None]:
    evidence = profile.evidence.get(evidence_id) if isinstance(evidence_id, str) else None
    if evidence is None or evidence.kind != "execution_contract" or evidence.document is None:
        return None, "verified_execution_contract_required"
    try:
        owner = _fields(evidence.document, {"schema", "verified", "device_kind", "device_id", "stream_id", "owner_resource_id",
                                           "stream_count", "concurrent_kernels", "roles", "effective_hardware_sha256",
                                           "runtime_sha256", "cache_protocol"}, "execution contract")
        if (owner["schema"] != OWNER_SCHEMA or owner["verified"] is not True
                or type(owner["stream_count"]) is not int or owner["stream_count"] != 1
                or owner["concurrent_kernels"] is not False):
            return None, "verified_single_stream_required"
        if owner["device_kind"] != "gpu":
            return None, "gpu_execution_owner_required"
        for name in ("device_id", "stream_id", "owner_resource_id"):
            _text(owner[name], name)
        if owner["owner_resource_id"] != owner["device_id"] + ".kernel_stream." + owner["stream_id"]:
            return None, "unowned_envelope_resource"
        roles = owner["roles"]
        if not isinstance(roles, (tuple, list)) or not roles or any(role not in ROLES for role in roles):
            return None, "invalid_execution_roles"
        if len(set(roles)) != len(roles) or query["role"] not in roles:
            return None, "execution_role_mismatch"
        for field in ("effective_hardware_sha256", "runtime_sha256", "cache_protocol"):
            if owner[field] != query[field]:
                return None, "execution_identity_mismatch"
        if (phase.metadata.get("target_component") != owner["device_id"]
                or phase.metadata.get("kernel_stream_id") != owner["stream_id"]
                or phase.metadata.get("kernel_role") != query["role"]):
            return None, "phase_device_role_or_stream_mismatch"
        if phase.metadata.get("kernel_measurement_boundary") != DEVICE_BOUNDARY:
            return None, "device_kernel_phase_required"
        return owner, None
    except (TypeError, ValueError):
        return None, "invalid_execution_contract"


def apply_kernel_calibration(phase: CostPhase, profile: KernelCalibrationProfile | None,
                             query: Mapping[str, Any], *, execution_contract_id: str | None = None,
                              execution_context: KernelExecutionContext | None = None, invocation_id: str | None = None) -> CostPhase:
    """Append one conditional device envelope in the SAME concurrent phase.

    Existing analytical demands, bytes, energy and launch phases are preserved.
    The returned duration is max(analytical, measured), never their sum.  Only a
    hash-verified, explicitly single-stream owner may introduce the envelope.
    """
    if profile is None:
        return phase
    resolution = resolve_kernel_calibration(profile, query)

    def unchanged(reason: str) -> CostPhase:
        return replace(phase, metadata={**phase.metadata, "kernel_calibration_resolution": {
            **resolution, "mode": "analytical_fallback", "reason": reason}})

    if resolution["mode"] != "exact":
        return unchanged(str(resolution["reason"]))
    if phase.metadata.get("kernel_calibration_applied") is not None:
        return unchanged("kernel_envelope_already_applied")
    if _native_conflict(phase.metadata):
        return unchanged("native_calibration_ownership_conflict")
    if phase.category != TaskCategory.COMPUTE:
        return unchanged("device_compute_phase_required")
    if (phase.name == "kernel_launch" or any("frontend" in demand.resource_id.casefold()
            or "launch" in demand.resource_id.casefold() for demand in phase.demands)):
        return unchanged("launch_phase_cannot_receive_device_envelope")
    assert profile is not None
    owner, reason = _owner(profile, execution_contract_id, query, phase)
    if owner is None:
        return unchanged(str(reason))
    resource_id = str(owner["owner_resource_id"])
    if any(demand.resource_id == resource_id for demand in phase.demands):
        return unchanged("kernel_envelope_owner_already_demanded")
    context = execution_context
    if not isinstance(context, KernelExecutionContext) or context._seal is not _CONTEXT_SEAL:
        return unchanged("actual_planner_runtime_context_required")
    if (context.hardware_sha256 != query["effective_hardware_sha256"]
            or context.runtime_sha256 != query["runtime_sha256"]):
        return unchanged("actual_context_identity_mismatch")
    if (context.device_id != owner["device_id"] or context.stream_id != owner["stream_id"]
            or resource_id != context.device_id + ".kernel_stream." + context.stream_id):
        return unchanged("actual_context_owner_mismatch")
    if (invocation_id not in context.invocation_keys or context.invocation_keys[invocation_id] != canonical_kernel_key(query)
            or context.phase_signatures[invocation_id] != _phase_signature(phase)):
        return unchanged("actual_invocation_phase_or_key_mismatch")
    duration = float(resolution["device_ns"])
    evidence = profile.evidence[str(execution_contract_id)]
    metadata = {**phase.metadata, "kernel_calibration_applied": True, "kernel_calibration_resolution": resolution,
                "kernel_calibration": {
                    "source_kind": SOURCE_KIND, "measurement_boundary": DEVICE_BOUNDARY,
                    "prediction_source": "microbench_exact_device_floor_plus_analysis",
                    "profile_id": profile.profile_id, "profile_sha256": profile.sha256,
                    "envelope_resource_id": resource_id, "execution_contract_id": execution_contract_id,
                    "execution_contract_sha256": evidence.sha256, "analytical_service_ns": phase.service_ns,
                    "measured_device_ns": duration, "combined_service_ns": max(phase.service_ns, duration),
                    "analytical_exceeds_measured": phase.service_ns > duration,
                    "accounting": "same_phase_concurrent_max_not_additive",
                    "conditional": True, "exact_replacement": False, "validated_llm_scope": False,
                    "resource_occupancy_calibrated": False, "hardware_transfer_validated": False,
                    "cache_transfer_validated": False,
                }}
    return replace(phase, demands=phase.demands + (ResourceDemand(resource_id, duration),), metadata=metadata)
