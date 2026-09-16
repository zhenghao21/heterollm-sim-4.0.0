"""Host-only static math/protocol/source checks for R19 graph-gap preparation/integration.
This script does not compile, load a native library, enumerate CUDA, or run a benchmark.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

PACKAGE = Path(__file__).resolve().parent


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> bytes:
    return struct.pack("<f", value)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()

    protocol_path = PACKAGE / "protocol.json"
    helper_path = PACKAGE / "graph_gap_probe.cpp"
    main_path = PACKAGE / "graph_gap_main.cpp"
    provenance_path = PACKAGE / "source_provenance.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    helper = helper_path.read_text(encoding="utf-8-sig")
    entry = main_path.read_text(encoding="utf-8-sig")

    assert protocol["status"] == "source_prepared_not_compiled_not_measured"
    configs = protocol["configs"]
    assert len(configs) == 6
    assert {(row["elements"], row["nodes"]) for row in configs} == {
        (elements, nodes) for elements in (1024, 262144) for nodes in (1, 8, 32)
    }
    execution = protocol["execution"]
    assert (execution["first"], execution["warmup"], execution["formal"], execution["calls_per_process"]) == (1, 5, 30, 36)
    assert execution["pair_repetitions"] == 3
    assert execution["maximum_preregistered_arm_config_pairs"] == 36
    pilot = execution["pilot"]
    assert pilot == {
        "id": "r19_graph_gap_pilot_e262144_g8",
        "config": "scale_f32_e262144_g8",
        "arms": ["control", "buffered"],
        "pairs_per_arm": 3,
        "modes": ["direct", "profile"],
        "native_processes": 12,
        "expected_exports": 6,
        "sequence": "Three sequential pairs; for each arm/pair run direct and profile as independently retained executions. No other configuration is launchable by invoke.ps1 in this revision.",
    }
    assert protocol["runtime"]["environment"]["GGML_CUDA_DISABLE_GRAPHS"] is None
    assert protocol["quality"]["direct_profile_relative_median_wall_limit"] == 0.20
    assert protocol["quality"]["formal_p90_p10_max"] == 1.5
    assert protocol["quality"]["three_process_median_relative_spread_max"] == 0.05
    assert "--cuda-graph-trace=node" in protocol["profiler"]["profile_options"]
    assert "global LLM" in protocol["controlled_invariants"]["analysis"]
    assert protocol["arms"]["buffered"]["numeric_coverage"].startswith("all stages in exactly three")
    assert provenance["state"] == "r20_copy_preparation_only_not_compiled_or_frozen"
    assert provenance["upstream_r19_provenance"]["sha256"]
    assert provenance["arm_contract"] == ["control", "buffered"]
    copied = provenance["copied_sources"]
    assert len(copied) == 1 and copied[0]["from"]["sha256"] == copied[0]["to"]["sha256"]
    assert Path(copied[0]["to"]["path"]).resolve() == PACKAGE / "frozen_module_guard.h"

    # Deterministic generator visits all possible numerators; SCALE-by-half is exact through stage 32.
    assert math.gcd(73, 255) == 1
    assert {((index * 73 + 19) % 255) - 127 for index in range(255)} == set(range(-127, 128))
    checked = 0
    for numerator in range(-127, 128):
        iterative = f32(numerator / 256)
        for stage in range(1, 33):
            iterative = f32(iterative * f32(0.5))
            closed = f32(float(Fraction(numerator, 1 << (8 + stage))))
            assert math.isfinite(iterative) and bits(iterative) == bits(closed)
            if numerator:
                assert abs(closed) >= 2 ** -40 and abs(closed) >= 2 ** -126
            checked += 1

    # Helper guards: focused execution unit; exactly one source-level final sync; no per-phase D2H/write.
    assert len(helper.encode("utf-8")) < 24000
    assert helper.count("ggml_backend_synchronize(harness.backend)") == 1
    assert "ArmResult run_control_arm" in helper and "ArmResult run_buffered_arm" in helper
    assert "std::array<CallSample, kTotalCalls> calls{};" in helper
    assert "std::array<ValidationBlock, 3> blocks{};" in helper
    assert "buffered_error_flush" in helper
    assert "blocks[0] = check_all_stages(harness, \"first\", 0);" in helper
    assert "blocks[1] = check_all_stages(harness, \"post_warmup\", kFirstCalls + kWarmupCalls - 1);" in helper
    assert "blocks[2] = check_all_stages(harness, \"post_formal\", kTotalCalls - 1);" in helper
    collect = helper.split("static void collect_phase", 1)[1].split("template <class Emit, class Flush>\nArmResult run_buffered_arm", 1)[0]
    assert "timed_call" in collect
    assert "ggml_backend_tensor_get" not in collect and "emit(" not in collect
    buffered = helper.split("ArmResult run_buffered_arm", 1)[1]
    assert buffered.index("blocks[2]") < buffered.index("emit_arm_metadata")
    control = helper.split("ArmResult run_control_arm", 1)[1].split("static void collect_phase", 1)[0]
    assert "per_call_validate_and_write" in control
    assert "if (ordinal == 0 || ordinal == kTotalCalls - 1)" in control
    assert "graph_submit/%s/%s/%d" in helper

    # Entrypoint guards: no implicit run, raw output is exclusive, and full raw hash is emitted only in a sidecar after close.
    assert "--identity-check" in entry and "--run" in entry
    assert "select exactly one of --identity-check or --run" in entry
    assert "PinnedFrozenModules<NativeModuleApi>" in entry
    assert "raw->close();" in entry and "receipt_json" in entry and "raw_sha256" in entry
    assert "graph_gap_probe::run_control_arm" in entry and "graph_gap_probe::run_buffered_arm" in entry
    assert r'observed_device_kernel_count\":null' in entry
    assert r'calibration_ready\":false' in entry
    assert "ggml_backend_cuda_init(0)" in entry
    assert r'source_runtime_equivalence_proven\":false' in entry

    python_sources = {}
    for path in PACKAGE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"))
        python_sources[path.name] = source_sha256(path)
    ps1 = (PACKAGE / "invoke.ps1").read_text(encoding="utf-8-sig")
    assert "[switch]$IdentityCheck" in ps1 and "[switch]$Run" in ps1
    assert "Run is outside the reviewed R19 pilot scope" in ps1
    assert "compiled_host_tested_not_gpu_executed" in ps1
    assert "profile_options" in ps1 and "[switch]$Run" in ps1

    result = {
        "schema": "graph-gap-preparation-check/v2",
        "pass": True,
        "gpu_access": False,
        "native_libraries_loaded": False,
        "numerator_stage_checks": checked,
        "protocol_sha256": source_sha256(protocol_path),
        "helper_cpp_sha256": source_sha256(helper_path),
        "entry_cpp_sha256": source_sha256(main_path),
        "provenance_sha256": source_sha256(provenance_path),
        "python_sources_sha256": python_sources,
        "checks": [
            "six fixed configurations",
            "one preregistered E262144/G8 pilot with two arms, three pairs, direct/profile",
            "exact F32 closed-form SCALE reference",
            "native CUDA Graph default retained",
            "focused A/B helper and delayed buffered output structure",
            "review-gated standalone entrypoint and sidecar raw hash contract",
            "identity-only invocation path and pilot-only launcher scope",
            "Python AST",
        ],
        "not_checked": [
            "C++ compilation, dynamic linking, or DLL identity execution",
            "GPU numerical results, timing, cache state, kernel count, graph replay, or overlap",
            "hardware telemetry, idle-window conditions, profiler export parsing",
            "any LLM-time fitting, cost coefficient, calibration, or inference",
        ],
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()