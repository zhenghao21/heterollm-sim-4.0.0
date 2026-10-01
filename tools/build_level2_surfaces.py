"""Build the checked-in Level-2 MMVQ/MMQ correction surfaces.

This tool only converts independent synthetic cold-kernel measurements into
``KernelSample`` literals.  It never reads native LLM TTFT/TPOT/E2E timings and
it deliberately installs only domains accepted by the holdout audit:

* N interpolation at M=1/2/4, K=4096;
* K interpolation at M=1, N=4096;
* K interpolation at M=2/4, N=4096 for Q6_K and IQ4_XS.
* format-local M=1 N/K four-corner cells for Q5_K/Q8_0 after their own
  independent holdout report accepts them.

The Q4_K M=2/4 K-axis anomaly is intentionally omitted.  A generated module
is committed as source data so importing the simulator does not touch the
measurement artifacts or the native DLL.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.config import scenario_from_dict  # noqa: E402
from heterollm_sim.cost_models import (  # noqa: E402
    GemmWorkload,
    GPUProfile,
    HBMProfile,
    estimate_gpu_gemm,
)
from heterollm_sim.kernel_model import (  # noqa: E402
    kernel_calibration_dispatch_signature,
)
from heterollm_sim.mmq_work import derive_mmq_work  # noqa: E402
from heterollm_sim.mmvq_work import (  # noqa: E402
    MMVQSourceContract,
    SOURCE_SHA256,
    derive_mmvq_work,
)

RUNTIME_SHA256 = (
    "8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297"
)
SCENARIO = (
    ROOT
    / "artifacts/development/ui_native_matched_20260929/"
    "qwen25_p512_o128_c1__fixed_runtime.scenario.json"
)
MEASUREMENT_SOURCES = (
    # Exact-Q8_1-input grid.  This is a generic operator grid (not a target
    # LLM timing fit) and contains the Q5_K/Q5_0 shapes absent from the first
    # cold-surface matrix.
    ROOT / "artifacts/development/level2_exact_q8_grid_20260930/measurements.json",
    # Broader M=1 shape grid for the native-matched workload families.
    ROOT / "artifacts/development/level2_shape_grid_20260930/measurements.json",
    # Latest M=1 N-axis capture.
    ROOT / "artifacts/development/cold_surface_20260930/measurements.json",
    # Complete M=2/4 N-axis grid used by the acceptance record.
    ROOT / "artifacts/development/cold_surface_m24_20260929/measurements.json",
    # M=1 K-axis grid.
    ROOT / "artifacts/development/cold_surface_k_20260929/measurements.json",
    # M=2/4 K-axis grid; Q4_K is rejected below because its holdout failed.
    ROOT / "artifacts/development/cold_surface_m24_k_20260929/measurements.json",
    # Independent M=512 prefill MMQ main-kernel grid.  This source is kept
    # separate from the decode MMVQ grid and is installed only after its own
    # holdout gate passes.
    ROOT / "artifacts/development/level2_mmq_prefill_exact_20260930/measurements.json",
    # Independent prefill M-axis joint grid.  It is installed only after the
    # dedicated M-specific holdout evaluator accepts each format.
    ROOT / "artifacts/development/level2_mmq_stage_20260930/prefill_m_sweep/measurements.json",
    # Format-local Q5_K/Q8_0 grid.  Its production eligibility is read from
    # the adjacent holdout report; missing or malformed reports fail closed.
    ROOT / "artifacts/development/level2_format_holdouts_20260930/measurements.json",
)
OUTPUT = ROOT / "src/heterollm_sim/kernel_level2_surfaces.py"
MANIFEST = ROOT / "artifacts/development/kernel_level2_surface_manifest.json"
INSTALL_FORMATS = frozenset({"q4_k", "q6_k", "iq4_xs"})
MMQ_SOURCE_DIR = "level2_mmq_prefill_exact_20260930"
MMQ_STAGE_SOURCE_DIR = "level2_mmq_stage_20260930"
FORMAT_HOLDOUT_SOURCE_DIR = "level2_format_holdouts_20260930"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mmq_accepted_formats() -> frozenset[str]:
    path = ROOT / "artifacts/development/level2_mmq_prefill_exact_20260930/holdout_evaluation.json"
    if not path.exists():
        return frozenset()
    data = _load(path)
    return frozenset(str(value).casefold() for value in data.get("accepted_formats", ()))


def _mmq_m_sweep_accepted_formats() -> frozenset[str]:
    path = ROOT / "artifacts/development/level2_mmq_stage_20260930/prefill_m_sweep/holdout_evaluation.json"
    if not path.exists():
        return frozenset()
    data = _load(path)
    return frozenset(str(value).casefold() for value in data.get("accepted_formats", ())
                     if data.get("production_qualified") is True)


def _format_holdout_accepted() -> frozenset[str]:
    path = ROOT / f"artifacts/development/{FORMAT_HOLDOUT_SOURCE_DIR}/holdout_evaluation.json"
    if not path.exists():
        return frozenset()
    data = _load(path)
    return frozenset(str(fmt).casefold() for fmt, accepted
                     in data.get("production_acceptance", {}).items() if accepted is True)


def _mmq_stage_manifest() -> dict[str, object]:
    """Copy the stage gate into the generated manifest without installing it."""
    root = ROOT / f"artifacts/development/{MMQ_STAGE_SOURCE_DIR}"
    protocol_path = root / "protocol.json"
    report_path = root / "holdout_evaluation.json"
    protocol = _load(protocol_path) if protocol_path.exists() else {}
    report = _load(report_path) if report_path.exists() else {}
    return {
        "measurement_source": str((root / "measurements.json").relative_to(ROOT)).replace("\\", "/"),
        "holdout_evaluation": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "required_prefill_m": report.get("required_prefill_m", protocol.get("required_prefill_m", [64, 128, 256, 512, 1024])),
        "measured_prefill_m": report.get("measured_prefill_m", protocol.get("measured_prefill_m", [])),
        "accepted_stages": report.get("accepted_stages", []),
        "accepted_formats": report.get("accepted_formats", []),
        "production_policy": "install only exact source/runtime/dispatch stage surfaces whose independent holdout gate passes; otherwise fail-closed",
    }


def _eligible(row: dict, source_path: Path) -> bool:
    if row.get("split") != "train":
        return False
    fmt = str(row["format"]).casefold()
    m, n, k = map(int, row["shape"])
    name = source_path.parent.name
    if name == MMQ_SOURCE_DIR:
        return row.get("phase") == "prefill" and row.get("kernel_family") == "mmq" and m == 512
    if name == "prefill_m_sweep":
        return row.get("phase") == "prefill" and row.get("kernel_family") == "mmq"
    if name == "level2_shape_grid_20260930":
        return m == 1 and fmt not in {"q5_k", "q8_0"}
    if name == "level2_exact_q8_grid_20260930":
        # Q5_K/Q8_0 now have a dedicated format-local protocol below; do not
        # mix their older evidence points into that complete four-corner cell.
        return m == 1 and fmt not in {"q5_k", "q8_0"}
    if name == FORMAT_HOLDOUT_SOURCE_DIR:
        return m == 1 and fmt in _format_holdout_accepted()
    if name == "cold_surface_20260930":
        # The broader 2026-09-30 M=1 grid below supersedes this earlier
        # partial grid.  Mixing its extra axis points would create artificial
        # interpolation holes in otherwise complete new cells.
        return False
    if name == "cold_surface_m24_20260929":
        return m in (2, 4)
    if name == "cold_surface_k_20260929":
        return False
    if name == "cold_surface_m24_k_20260929":
        return m in (2, 4) and n == 4096 and fmt != "q4_k"
    return False


def _source_work(m: int, n: int, k: int, fmt: str):
    contract = MMVQSourceContract(
        1200,
        1200,
        32,
        SOURCE_SHA256,
        RUNTIME_SHA256,
        True,
        False,
        True,
    )
    return derive_mmvq_work(
        m=m,
        k=k,
        n=n,
        weight_format=fmt.upper(),
        contract=contract,
        allow_k_formats=True,
        allow_iq4_xs=True,
    )


def _mmq_source_work(m: int, n: int, k: int, fmt: str):
    return derive_mmq_work(
        m=m, n=n, k=k, weight_format=fmt.upper(), sm_count=84,
        shared_memory_per_block=101376, runtime_binary_sha256=RUNTIME_SHA256,
    )


def _analytical_ns(gpu, hbm, row: dict, fmt: str) -> float:
    m, n, k = map(int, row["shape"])
    family = row.get("kernel_family", "mmvq")
    phase = row.get("phase", "decode")
    source = _mmq_source_work(m, n, k, fmt) if family == "mmq" else _source_work(m, n, k, fmt)
    bits = {"q4_k": 4, "q5_k": 5, "q5_0": 5, "q6_k": 6,
            "iq4_xs": 4, "q8_0": 8}[fmt]
    workload = GemmWorkload(
        m=m,
        k=k,
        n=n,
        activation_bits=32,
        weight_bits=bits,
        output_bits=32,
        accumulator_bits=32,
        packed_weight_formats=(fmt,),
        weight_storage_bytes=source.logical_weight_bytes,
        activation_storage_bytes=(source.consumer_unique_bytes if family == "mmq"
                                   else source.consumer_q8_1_unique_bytes),
        output_storage_bytes=(source.native_output_bytes if family == "mmq"
                              else source.output_f32_bytes),
        mmq_work=source if family == "mmq" else None,
        mmvq_work=source if family != "mmq" else None,
        cache_protocol="cold_streaming",
        execution_phase=phase,
        activation_dtype="fp32",
        layout="contiguous",
    )
    gpu_for_estimate = gpu
    if family == "mmq":
        geometry = row.get("geometry") or {}
        block = geometry.get("block") or (32, 8, 1)
        registers = int(geometry.get("registers_per_thread") or 32)
        shared = int(geometry.get("shared_memory_per_block") or 0)
        warps = int(block[1])
        profile = gpu.kernel_model
        if profile is not None:
            profile = replace(
                profile,
                kernels=tuple(
                    replace(kernel, registers_per_thread=registers,
                            shared_memory_per_cta=shared, warps_per_cta=warps)
                    if kernel.phase == phase and kernel.kernel_family == f"cuda_mmq_{fmt}"
                    else kernel
                    for kernel in profile.kernels
                ),
            )
            gpu_for_estimate = replace(gpu, kernel_model=profile)
    estimate = estimate_gpu_gemm(gpu_for_estimate, hbm, workload)
    # The surface corrects the device kernel phase, not its separate launch.
    return float(estimate.metadata["prediction"]["analytical_ns"])


def _build_rows() -> tuple[list[dict], dict]:
    scenario = scenario_from_dict(_load(SCENARIO))
    gpu = scenario.resolve_component_profile("gpu0", GPUProfile)
    hbm = scenario.resolve_component_profile("hbm0", HBMProfile)
    if gpu.kernel_model is None:
        raise RuntimeError("native-matched scenario has no analytical kernel profile")

    # First source wins at shared grid corners.  This gives the N-axis capture
    # precedence at K=4096 while retaining the independent K-axis endpoints.
    unique: dict[tuple[str, str, str, int, int, int], dict] = {}
    source_refs: dict[str, dict] = {}
    rejected_training_rows = []
    for path in MEASUREMENT_SOURCES:
        digest = _sha256(path)
        source_refs[str(path.relative_to(ROOT)).replace("\\", "/")] = {
            "sha256": digest,
            "rows": 0,
        }
        for row in _load(path):
            if not _eligible(row, path):
                continue
            fmt = str(row["format"]).casefold()
            family = str(row.get("kernel_family", "mmvq")).casefold()
            phase = str(row.get("phase", "decode")).casefold()
            allowed = (INSTALL_FORMATS | _format_holdout_accepted()
                       if family == "mmvq" else
                       (_mmq_accepted_formats() | _mmq_m_sweep_accepted_formats()))
            if fmt not in allowed or family not in {"mmvq", "mmq"}:
                continue
            samples_ns = tuple(float(x) for x in row.get("samples_ns", ()))
            if len(samples_ns) >= 2:
                cv_pct = statistics.stdev(samples_ns) / statistics.mean(samples_ns) * 100.0
                if cv_pct >= 10.0:
                    rejected_training_rows.append({
                        "format": row["format"], "shape": row["shape"],
                        "cv_pct": cv_pct,
                        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "reason": "training_repeatability_cv_not_below_10_percent",
                    })
                    continue
            m, n, k = map(int, row["shape"])
            key = (family, phase, fmt, m, n, k)
            if key in unique:
                continue
            source_refs[str(path.relative_to(ROOT)).replace("\\", "/")]["rows"] += 1
            unique[key] = {
                "format": fmt,
                "phase": phase,
                "kernel_family": family,
                "shape": (m, n, k),
                "median_ns": float(row["median_ns"]),
                "samples_ns": tuple(float(x) for x in row["samples_ns"]),
                "bandwidth_gb_s": float(row["bandwidth_gb_s"]),
                "csv_sha256": str(row.get("csv_sha256", row.get("raw_sha256", ""))),
                "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                "geometry": row.get("geometry"),
            }

    output = []
    for key in sorted(unique):
        row = unique[key]
        fmt = row["format"]
        family = row["kernel_family"]
        source = (_mmq_source_work(*row["shape"], fmt)
                  if family == "mmq" else _source_work(*row["shape"], fmt))
        baseline = _analytical_ns(gpu, hbm, row, fmt)
        samples = row["samples_ns"]
        stddev = statistics.stdev(samples) if len(samples) >= 2 else 0.0
        signature_workload = GemmWorkload(
            m=row["shape"][0],
            k=row["shape"][2],
            n=row["shape"][1],
            activation_bits=32,
            weight_bits={"q4_k": 4, "q5_k": 5, "q5_0": 5, "q6_k": 6,
                         "iq4_xs": 4, "q8_0": 8}[fmt],
            output_bits=32,
            accumulator_bits=32,
            packed_weight_formats=(fmt,),
            weight_storage_bytes=source.logical_weight_bytes,
            activation_storage_bytes=(source.consumer_unique_bytes if family == "mmq"
                                       else source.consumer_q8_1_unique_bytes),
            output_storage_bytes=(source.native_output_bytes if family == "mmq"
                                  else source.output_f32_bytes),
            mmq_work=source if family == "mmq" else None,
            mmvq_work=source if family != "mmq" else None,
            cache_protocol="cold_streaming",
            execution_phase=row["phase"],
            activation_dtype="fp32",
        )
        signature = kernel_calibration_dispatch_signature(signature_workload)
        output.append(
            {
                "format": fmt,
                "phase": row["phase"],
                "kernel_family": family,
                "shape": row["shape"],
                "device_ns": row["median_ns"],
                "analytical_ns": baseline,
                "stddev_ns": stddev,
                "sample_count": len(samples),
                "evidence": (
                    f"{row['source']}#csv_sha256={row['csv_sha256']};"
                    f"runtime_sha256={RUNTIME_SHA256};cache_control=all"
                ),
                "achieved_bandwidth_gb_s": row["bandwidth_gb_s"],
                "dispatch_signature": signature,
                "geometry": row.get("geometry"),
            }
        )
    return output, {
        "scenario": str(SCENARIO.relative_to(ROOT)).replace("\\", "/"),
        "scenario_sha256": _sha256(SCENARIO),
        "runtime_binary_sha256": RUNTIME_SHA256,
        "measurement_sources": source_refs,
        "surface_domains": [
            "N interpolation at M=1/2/4,K=4096 for Q4_K/Q6_K/IQ4_XS",
            "K interpolation at M=1,N=4096 for Q4_K/Q6_K/IQ4_XS",
            "K interpolation at M=2/4,N=4096 for Q6_K/IQ4_XS",
            "prefill MMQ main-kernel M=512 N/K joint grid for Q4_K/Q6_K/IQ4_XS",
            "prefill MMQ M=64/128/256/1024 N/K joint cells with independent 3072 holdouts",
        ],
        "excluded_domain": "Q4_K K interpolation at M=2/4,N=4096",
        "not_installed_formats": sorted({"q5_k", "q5_0", "q8_0"} - _format_holdout_accepted()),
        "format_holdout_evaluation": f"artifacts/development/{FORMAT_HOLDOUT_SOURCE_DIR}/holdout_evaluation.json",
        "holdout_evaluation": "artifacts/development/level2_shape_grid_20260930/holdout_evaluation.json",
        "mmq_holdout_evaluation": "artifacts/development/level2_mmq_prefill_exact_20260930/holdout_evaluation.json",
        "mmq_m_sweep_holdout_evaluation": "artifacts/development/level2_mmq_stage_20260930/prefill_m_sweep/holdout_evaluation.json",
        "mmq_stage_surface_manifest": _mmq_stage_manifest(),
        "sample_count": len(output),
        "rejected_training_rows": rejected_training_rows,
    }


def _validated_errors() -> dict[str, float]:
    path = ROOT / "artifacts/development/level2_shape_grid_20260930/holdout_evaluation.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for row in rows:
        if row.get("eligible"):
            fmt = str(row["format"]).casefold()
            result[fmt] = max(result.get(fmt, 0.0), float(row["ape_pct"]) / 100.0)
    format_report = ROOT / f"artifacts/development/{FORMAT_HOLDOUT_SOURCE_DIR}/holdout_evaluation.json"
    if format_report.exists():
        data = _load(format_report)
        for item in data.get("formats", ()):
            fmt = str(item.get("format", "")).casefold()
            for row in item.get("holdout_rows", ()):
                if row.get("eligible"):
                    result[fmt] = max(result.get(fmt, 0.0), float(row["ape_pct"]) / 100.0)
    return result


def _validated_errors_by_key() -> dict[str, float]:
    result = {f"decode:mmvq:{fmt}": value for fmt, value in _validated_errors().items()}
    path = ROOT / "artifacts/development/level2_mmq_prefill_exact_20260930/holdout_evaluation.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows", ())
        for row in rows:
            if row.get("eligible"):
                fmt = str(row["format"]).casefold()
                key = f"prefill:mmq:{fmt}"
                result[key] = max(result.get(key, 0.0), float(row["ape_pct"]) / 100.0)
    sweep = ROOT / "artifacts/development/level2_mmq_stage_20260930/prefill_m_sweep/holdout_evaluation.json"
    if sweep.exists():
        data = _load(sweep)
        for group in data.get("formats", ()):
            if not group.get("accepted"):
                continue
            fmt = str(group.get("format", "")).casefold()
            errors = [float(row["ape_pct"]) / 100.0 for row in group.get("rows", ())
                      if row.get("eligible") and row.get("ape_pct") is not None]
            if errors:
                result[f"prefill:mmq:{fmt}"] = max(result.get(f"prefill:mmq:{fmt}", 0.0), max(errors))
    return result


def _resource_by_key(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        geometry = row.get("geometry")
        if not geometry or row.get("kernel_family") != "mmq":
            continue
        key = f"{row.get('phase', 'decode')}:{row.get('kernel_family', 'mmvq')}:{row['format']}:{row['dispatch_signature']}"
        item = {
            "registers_per_thread": geometry.get("registers_per_thread"),
            "shared_memory_per_cta": geometry.get("shared_memory_per_block"),
            "warps_per_cta": geometry.get("block", [32, 0, 1])[1],
            "block": geometry.get("block"),
        }
        if key in result and result[key] != item:
            raise RuntimeError(f"MMQ resource geometry changed inside specialization {key}")
        result[key] = item
    return result


def _module(rows: list[dict], validation_errors: dict[str, float],
            validation_by_key: dict[str, float], resources: dict[str, dict]) -> str:
    lines = [
        '"""Generated Level-2 calibrated analytical MMVQ/MMQ surfaces.',
        "",
        "Source data is independent synthetic cold-kernel evidence.  This module",
        "does not contain native LLM timing and is intentionally narrow.",
        '"""',
        "from __future__ import annotations",
        "",
        "from .kernel_model import KernelSample",
        "",
        "LEVEL2_SCHEMA = 'kernel-level2-calibrated-analytical/v1'",
        "LEVEL2_HARDWARE_ID = 'nvidia-rtx-5080'",
        f"LEVEL2_RUNTIME_SHA256 = '{RUNTIME_SHA256}'",
        "LEVEL2_CACHE_PROTOCOL = 'cold_streaming'",
        "LEVEL2_PROVENANCE = 'independent_synthetic_cold_shape_holdouts_v1'",
        f"LEVEL2_VALIDATION_ERROR_BY_FORMAT = {validation_errors!r}",
        f"LEVEL2_VALIDATION_ERROR_BY_KEY = {validation_by_key!r}",
        f"LEVEL2_RESOURCE_BY_KEY = {resources!r}",
        "",
        "_ROWS = (",
    ]
    for row in rows:
        lines.append(f"    {row!r},")
    lines.extend(
        [
            ")",
            "",
            "",
            "def level2_samples(weight_format, *, output_bits=32, activation_dtype='fp32', phase='decode', kernel_family='mmvq'):",
            "    \"\"\"Return immutable samples for one source-bound kernel dispatch.\"\"\"",
            "    fmt = str(weight_format).casefold()",
            "    return tuple(",
            "        KernelSample(",
            "            row['shape'][0], row['shape'][1], row['shape'][2],",
            "            row['device_ns'], row['analytical_ns'], row['stddev_ns'],",
            "            row['sample_count'], row['evidence'],",
            "            row['achieved_bandwidth_gb_s'], row['dispatch_signature'],",
            "        )",
            "        for row in _ROWS",
            # The measured CUDA main kernel consumes the source-produced Q8_1
            # vector.  It is therefore shared by the F32-hidden and the
            # fp16-declared logical paths; the conversion stage remains a
            # separate planner task and is not included in this wall.
            "        if row['format'] == fmt and row.get('phase', 'decode') == phase and row.get('kernel_family', 'mmvq') == kernel_family and output_bits == 32 and activation_dtype in {'fp16', 'fp32'}",
            "    )",
            "",
            "",
            "def level2_resource(weight_format, dispatch_signature, *, phase='decode', kernel_family='mmvq'):",
            "    key = f'{phase}:{kernel_family}:{str(weight_format).casefold()}:{dispatch_signature}'",
            "    return LEVEL2_RESOURCE_BY_KEY.get(key)",
            "",
            "",
            "__all__ = [",
            "    'LEVEL2_SCHEMA', 'LEVEL2_HARDWARE_ID', 'LEVEL2_RUNTIME_SHA256',",
            "    'LEVEL2_CACHE_PROTOCOL', 'LEVEL2_PROVENANCE', 'level2_samples',",
            "    'LEVEL2_VALIDATION_ERROR_BY_FORMAT', 'LEVEL2_VALIDATION_ERROR_BY_KEY',",
            "    'LEVEL2_RESOURCE_BY_KEY', 'level2_resource',",
            "]",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows, manifest = _build_rows()
    validation_errors = _validated_errors()
    validation_by_key = _validated_errors_by_key()
    resources = _resource_by_key(rows)
    manifest["validation_error_by_format"] = validation_errors
    manifest["validation_error_by_key"] = validation_by_key
    manifest["resource_by_key"] = resources
    OUTPUT.write_text(_module(rows, validation_errors, validation_by_key, resources), encoding="utf-8", newline="\n")
    manifest["output"] = str(OUTPUT.relative_to(ROOT)).replace("\\", "/")
    manifest["output_sha256"] = _sha256(OUTPUT)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
