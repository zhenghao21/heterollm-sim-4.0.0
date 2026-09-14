"""Create and verify the single v4 freeze manifest used by the next run."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_llama_compare import _hardware_fingerprint, probe_hardware
from generalization_acceptance_matrix import BINARY, MODELS, PROFILE, PROMPTS, OUTPUTS, PARALLEL, ROOT

ENGINE_TIMING_CONTRACT = {
    "id": "engine-stage+client-real-token/v3",
    "engine_ttft": "first_engine_token-engine_request_begin;counter=t_prompt_last-t_start",
    "engine_e2e": "last_engine_token-engine_request_begin;counter=t_gen_last-t_start",
    "engine_tpot": "(last_engine_token-first_engine_token)/(output_tokens-1);counter=(t_gen_last-t_prompt_last)/(output_tokens-1);null_if_output_tokens<=1",
    "ttft": "first_real_token-request_start;client_diagnostic",
    "e2e": "last_real_token-request_start;DONE_excluded;client_diagnostic",
    "tpot": "(last_real_token-first_real_token)/(output_tokens-1);client_diagnostic",
    "unit": "milliseconds",
    "clock": "monotonic_ns",
}

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

ENGINE_TIMING_CONTRACT_SHA256 = canonical_hash(ENGINE_TIMING_CONTRACT)

MANIFEST = ROOT / "artifacts/multimodel_next/blind_generalization_freeze_v4.json"
QWEN38 = (("short", "short", 1), ("short", "medium", 4), ("short", "long", 2),
          ("medium", "short", 4), ("medium", "medium", 2), ("medium", "long", 1),
          ("long", "short", 2), ("long", "medium", 1), ("long", "long", 4))
SOURCE_FILES = (
    Path(__file__).resolve(), ROOT / "tools/native_llama_compare.py",
    ROOT / "tools/generalization_acceptance_matrix.py", ROOT / "tools/merge_generalization_acceptance.py",
    ROOT / "tools/evaluation_contract.py", ROOT / "tools/native_error_matrix.py",
    ROOT / "tools/plot_error_heatmaps.py", ROOT / "tools/extract_nsys_trace.py",
    ROOT / "tools/extract_cuda_api_phase.py", ROOT / "src/heterollm_sim/planner.py",
    ROOT / "src/heterollm_sim/calibration.py", ROOT / "src/heterollm_sim/gguf_parity.py",
    ROOT / "tools/unified_evidence_manifest.py", ROOT / "tools/replay_simulator_from_native.py",
    ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-server.exe",
    ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-tokenize.exe",
    MODELS["qwen25"], MODELS["qwen35"], MODELS["qwen38"], MODELS["tinyllama"], MODELS["smollm2"],
    PROFILE["qwen25"][0], PROFILE["qwen35"][0], PROFILE["qwen38"][0], PROFILE["tinyllama"][0],
)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def create():
    missing = [str(p) for p in SOURCE_FILES if not p.exists()]
    if missing: raise SystemExit("missing freeze input: " + ", ".join(missing))
    hw = probe_hardware()
    scenarios = {m: [{"prompt_band": p, "output_band": o, "parallel": n} for p in PROMPTS for o in OUTPUTS for n in PARALLEL] for m in MODELS}
    scenarios["qwen38"] = [{"prompt_band": p, "output_band": o, "parallel": n} for p, o, n in QWEN38]
    manifest = {
        "schema": "blind-generalization-freeze/v4", "freeze_id": "v4-mechanism-validation",
        "status": "frozen_not_run", "purpose": "mechanism_validation", "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Freeze before execution; prediction is saved before native reveal; no result-driven tuning in this version.",
        "matrix": {"models": list(MODELS), "prompt_bands": list(PROMPTS), "output_bands": list(OUTPUTS), "parallel_values": list(PARALLEL), "repeats": 3,
                   "scenario_count_by_model": {m: len(v) for m, v in scenarios.items()},
                   "execution_count": sum(len(v) for v in scenarios.values()) * 3,
                   "qwen38_selected_scenarios": scenarios["qwen38"], "qwen38_unselected_scenario_count": 18,
                   "ctx": 2048, "batch": 64, "ubatch": 64, "threads": 16, "gpu_layers": 0, "request_timing": "stream",
                   "output_protocol": "fixed_ignore_eos", "output_tokens": {"short": 8, "medium": 32, "long": 128}},
        "scenarios": scenarios,
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in SOURCE_FILES},
        "model_sha256": {m: digest(path) for m, path in MODELS.items()},
        "profile_sha256": {m: digest(PROFILE[m][0]) for m in PROFILE if PROFILE[m][0] is not None},
        "binary_sha256": digest(BINARY), "tokenizer_sha256": digest(ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-tokenize.exe"),
        "hardware_snapshot": hw, "hardware_fingerprint": _hardware_fingerprint(hw),
        "measurement_contract": {"ttft": "first real token - request start", "e2e": "last real token - request start; DONE excluded",
                                 "tpot": "(last real token-first real token)/(N_out-1), N_out>1",
                                 "engine_timing_contract": ENGINE_TIMING_CONTRACT,
                                 "engine_timing_contract_sha256": ENGINE_TIMING_CONTRACT_SHA256,
                                 "engine_primary_metrics": ["engine_ttft", "engine_tpot", "engine_e2e"],
                                 "batch": "per-request timestamps plus common batch start/end/makespan", "eos": "fixed ignore_eos; natural EOS is separate exploratory data"},
        "required_identity": ["source_sha256", "model_sha256", "profile_sha256", "binary_sha256", "tokenizer_sha256", "hardware_fingerprint"],
        "required_source_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in SOURCE_FILES],
        "required_contract": ["engine_timing_contract", "engine_timing_contract_sha256"],
        "acceptance_status": "not_final_independent_blind_set", "limitations": ["Qwen3.8 has 9 selected triples; 18 remain unvalidated", "cross-hardware is unverified"],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(MANIFEST), "status": manifest["status"], "executions": manifest["matrix"]["execution_count"]}, ensure_ascii=False))

def verify():
    if not MANIFEST.exists(): raise SystemExit("manifest missing")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if m.get("schema") != "blind-generalization-freeze/v4" or m.get("status") != "frozen_not_run": raise SystemExit("invalid freeze status/schema")
    required = set(m.get("required_contract") or ())
    if not {"engine_timing_contract", "engine_timing_contract_sha256"}.issubset(required):
        raise SystemExit("engine timing contract is not required by freeze")
    measurement = m.get("measurement_contract") or {}
    contract = measurement.get("engine_timing_contract")
    if contract != ENGINE_TIMING_CONTRACT:
        raise SystemExit("engine timing contract mismatch")
    if measurement.get("engine_timing_contract_sha256") != ENGINE_TIMING_CONTRACT_SHA256:
        raise SystemExit("engine timing contract sha256 mismatch")
    if measurement.get("engine_primary_metrics") != ["engine_ttft", "engine_tpot", "engine_e2e"]:
        raise SystemExit("engine primary metric list mismatch")
    if m.get("policy") != "Freeze before execution; prediction is saved before native reveal; no result-driven tuning in this version.":
        raise SystemExit("prediction-before-native policy mismatch")
    if not isinstance(m.get("required_identity"), list) or any(not item for item in m["required_identity"]):
        raise SystemExit("required identity list missing")
    expected_sources = {str(path.relative_to(ROOT)).replace("\\", "/") for path in SOURCE_FILES}
    declared_sources = set(m.get("source_sha256") or {})
    if declared_sources != expected_sources:
        raise SystemExit("freeze source coverage mismatch")
    if set(m.get("required_source_files") or ()) != expected_sources:
        raise SystemExit("freeze required source file list mismatch")
    for rel, want in (m.get("source_sha256") or {}).items():
        p = ROOT / rel
        if not p.exists() or len(str(want)) != 64 or digest(p).lower() != str(want).lower(): raise SystemExit(f"source hash mismatch: {rel}")
    for section in ("model_sha256", "profile_sha256"):
        for name, want in (m.get(section) or {}).items():
            p = MODELS.get(name) if section == "model_sha256" else PROFILE.get(name, (None,))[0]
            if p is None or not p.exists() or len(str(want)) != 64 or digest(p).lower() != str(want).lower(): raise SystemExit(f"{section} mismatch: {name}")
    if len(str(m.get("binary_sha256"))) != 64 or digest(BINARY).lower() != str(m.get("binary_sha256")).lower(): raise SystemExit("binary hash mismatch")
    tok = ROOT / "source/llama.cpp-semantic/build-semantic-direct/bin/llama-tokenize.exe"
    if len(str(m.get("tokenizer_sha256"))) != 64 or digest(tok).lower() != str(m.get("tokenizer_sha256")).lower(): raise SystemExit("tokenizer hash mismatch")
    if not m.get("hardware_snapshot") or not m.get("hardware_fingerprint"): raise SystemExit("hardware identity missing")
    actual_hw = probe_hardware()
    if _hardware_fingerprint(actual_hw) != m.get("hardware_fingerprint"):
        raise SystemExit("hardware fingerprint mismatch")
    print(json.dumps({"manifest": str(MANIFEST), "verified": True, "status": m["status"]}, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--create", action="store_true"); ap.add_argument("--verify", action="store_true"); args = ap.parse_args()
    if args.create == args.verify: ap.error("choose exactly one of --create or --verify")
    (create if args.create else verify)()
