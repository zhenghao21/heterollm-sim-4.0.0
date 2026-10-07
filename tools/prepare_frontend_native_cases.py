"""Prepare current physical-hardware scenarios; do not run any benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from heterollm_sim.config import HostOutputContract, SamplingPolicy, scenario_from_dict
from heterollm_sim.gguf_parity import build_model_from_gguf, read_gguf_metadata
from heterollm_sim.llama_scenario import prepare_llama_scenario
from heterollm_sim.llama_tensor_storage import (
    apply_llama_tensor_storage_contract,
    derive_llama_tensor_storage_contract,
    qualify_llama_tensor_storage_contract,
)
from heterollm_sim.physical_contract import require_physical_memory_config
from heterollm_sim.serde import to_primitive
from heterollm_sim.web import scenario_to_payload
from validate_preset_matrix import scenario_for


ROOT = Path(__file__).resolve().parents[1]
HARDWARE = "local-native-rtx5080-9950x3d-gddr7-ddr5"


def apply_native_output_contract(scenario, native_path: Path):
    """Bind observed native settings, without copying any measured latency."""
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if native.get("status") != "completed":
        raise ValueError("native output contract requires a completed native settings record")
    configuration = native["configuration"]
    settings = native["native_server_props"]["default_generation_settings"]["params"]
    if configuration.get("temperature") != 0 or settings.get("backend_sampling") is not False:
        raise ValueError("only recorded temperature-zero CPU sampling is qualified")
    if settings.get("samplers") != ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]:
        raise ValueError("native sampler ordering differs from the qualified chain")
    neutral = {"repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
               "dry_multiplier": 0.0, "top_n_sigma": -1.0, "typical_p": 1.0,
               "xtc_probability": 0.0, "dynatemp_range": 0.0, "mirostat": 0}
    if any(settings.get(key) != value for key, value in neutral.items()):
        raise ValueError("native sampler contains an unqualified active extra stage")
    contract = HostOutputContract("hostmem0", scenario.model.vocabulary_size, "fp32", 4)
    sampling = SamplingPolicy("greedy", temperature=0.0, implementation="llama_cpp_cpu_chain",
        top_k=settings["top_k"], top_p=settings["top_p"], min_p=settings["min_p"], min_keep=settings["min_keep"])
    return replace(scenario, host_output_contract=contract, sampling_policy=sampling,
        workload=replace(scenario.workload, metadata={**scenario.workload.metadata,
            "native_output_contract_evidence": {
                "native_settings_record": native_path.name,
                "logits_source": "llama-context.cpp: n_outputs*n_vocab*sizeof(float) asynchronous host copy",
                "sampling_source": "common/sampling.cpp: ordered chain remains active at temperature zero",
                "measured_latency_used": False,
                "remaining_partial_costs": ["completion interrupt latency", "sampler heap repair/sort/branch/cache/compiler behavior",
                    "disabled sampler and EOS logit-bias overhead", "GET_ROWS F16 conversion compute"],
            }}))


def prepare(output: Path, source_root: Path, server: Path, models: list[tuple[str, Path]], *, native_output_contract=False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    contract = derive_llama_tensor_storage_contract(source_root)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if not server.is_file():
        raise FileNotFoundError(server)
    result = {
        "status": "prepared_not_measured",
        "hardware_preset": HARDWARE,
        "source_commit": source_commit,
        "source_contract": contract,
        "native_samples_reused": False,
        "latency_calibration_applied": False,
        "cases": [],
        "methodology": {
            "submission": "Each scenario must be imported and submitted through the frontend.",
            "workload": "Single request; 512 input tokens and 128 output tokens; no prompt reuse.",
            "context": "Request and require effective context 768 on native; simulate context 768.",
            "execution": "Run native only while frontend simulation jobs are idle.",
            "native_sampling": "Two warmups and five formal repetitions per model.",
            "comparison": "Compare server engine timing medians; preserve client timing separately.",
            "identity": "Use the existing GGUF full-identity import contract, not a model-name substitution.",
            "limitation": "Physical preset parameters include analytical assumptions; this is an accuracy measurement, not an accuracy guarantee.",
        },
    }
    for slug, model_path in models:
        gguf = read_gguf_metadata(model_path)
        model = build_model_from_gguf(gguf)
        payload = scenario_for(HARDWARE, "qwen3-0_6b")
        payload["name"] = f"frontend-native/{slug}/512-128"
        payload["model"] = to_primitive(model)
        payload["placement"]["model_name"] = model.name
        payload["placement"]["parallel"]["rank_mapping"] = [{
            "rank": 0, "component_id": "gpu0", "tp_rank": 0, "pp_rank": 0,
            "ep_rank": 0, "memory_component_id": "gddr0", "cim_component_id": None,
        }]
        payload["placement"]["kv_policy"].update(
            cache_component="gddr0", offload_component=None, offload_ratio=0.0,
            dtype="fp16", layout_mode="legacy_single", kv_unified=True,
        )
        payload["profiles"]["llama_cpp"].update(
            context=768, device_memory_tiering=False, threads=16, threads_batch=16,
            flash_attn=False, kv_type_k="f16", kv_type_v="f16", kv_unified=True,
            cont_batching=True, warmup=True, seed=0, mmap=True, mlock=False,
            op_offload=True, split_mode="layer", main_gpu=0,
        )
        parity = {
            "native_model_path": str(model_path.resolve()),
            "native_model_file": model_path.name,
            "native_runtime_context": {"requested": 768, "effective_required": 768},
        }
        payload["workload"]["metadata"].update(parity)
        if gguf.architecture not in {"qwen3", "qwen35"}:
            raise ValueError("native CUDA RoPE contract requires a qualified Qwen3/Qwen3.5 graph")
        payload["workload"]["metadata"]["native_rope_source_contract"] = {
            "schema": "llama.cpp.cuda-rope/v1", "strategy": "runtime_sin_cos",
            "position_components": 1 if gguf.architecture == "qwen3" else 4,
            "source_revision": source_commit, "timing_completeness": "partial",
        }
        payload["workload"]["requests"][0]["metadata"].update(parity)
        scenario = scenario_from_dict(payload)
        scenario = apply_llama_tensor_storage_contract(scenario, contract, f32_hidden_storage=True)
        native_output = output / f"native_{slug}.json"
        if native_output_contract:
            scenario = apply_native_output_contract(scenario, native_output)
        qualified = qualify_llama_tensor_storage_contract(scenario)
        if not qualified["qualified"]:
            raise ValueError(f"GGUF source contract failed for {slug}: {qualified}")
        scenario = prepare_llama_scenario(scenario)
        # Keep the exported authoring contract consistent before any UI import.
        prepared = scenario_to_payload(scenario)
        roundtrip = scenario_from_dict(prepared)
        if roundtrip.placement.kv_policy.dtype not in {"f16", "fp16"}:
            raise ValueError("prepared KV dtype differs from native f16")
        memories = {}
        for component in roundtrip.hardware.components:
            if component.component_id in {"hostmem0", "gddr0"}:
                physical = require_physical_memory_config(component)
                memories[component.component_id] = {
                    "kind": physical.kind, "capacity_bytes": component.capacity_bytes,
                }
        scenario_path = output / f"scenario_{slug}_512_128.json"
        scenario_path.write_text(json.dumps(prepared, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        command = [
            str(Path(sys.executable).resolve()), str(ROOT / "tools/native_benchmark.py"),
            "--server", str(server.resolve()), "--model", str(model_path.resolve()),
            "--output", str(native_output.resolve()), "--prompt-tokens", "512",
            "--output-tokens", "128", "--context", "768", "--expected-effective-context", "768",
            "--batch", "512", "--ubatch", "512", "--threads", "16",
            "--gpu-layers", "-1", "--parallel", "1", "--warmup", "2", "--repetitions", "5",
            "--flash-attn", "off",
        ]
        result["cases"].append({
            "case_id": slug,
            "scenario_file": scenario_path.name,
            "scenario_path": str(scenario_path.resolve()),
            "native_output_path": str(native_output.resolve()),
            "native_command": command,
            "source_qualification": qualified,
            "physical_memories": memories,
            "model": {
                "path": str(model_path.resolve()), "sha256": gguf.sha256,
                "architecture": gguf.architecture, "tensor_count": gguf.tensor_count,
                "tensor_types": dict(Counter(t.type_name for t in gguf.tensor_directory)),
                "main_layer_count": gguf.n_layer, "excluded_nextn_layers": gguf.n_layer_nextn,
                "hidden_size": gguf.n_embd, "head_count": gguf.n_head,
                "kv_head_count": gguf.n_head_kv, "vocabulary_size": gguf.vocab_size,
            },
            "scenario_configuration": {
                "llama_cpp": to_primitive(roundtrip.llama_cpp_config),
                "scheduler": to_primitive(roundtrip.workload.scheduler),
                "kv_policy": to_primitive(roundtrip.placement.kv_policy),
                "rank_mapping": to_primitive(roundtrip.placement.parallel.rank_mapping),
                "f32_hidden_storage": roundtrip.workload.metadata.get("llama_cpp_f32_hidden_storage"),
                "host_output": to_primitive(roundtrip.host_output_contract),
                "sampling": to_primitive(roundtrip.sampling_policy),
            },
        })
        print(f"prepared {slug}: {scenario_path}", flush=True)
    (output / "preparation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/frontend_native_validation_2026-10-07")
    parser.add_argument("--source-root", type=Path, default=Path("F:/codex_project/_runtime_sources/llama.cpp"))
    parser.add_argument("--server", type=Path)
    parser.add_argument("--native-output-contract", action="store_true",
                        help="After native measurement, bind recorded CPU sampler settings and F32 host logits; no latency fitting")
    parser.add_argument("--small-model", type=Path, default=ROOT.parent / "models/Qwen3-0.6B-f16.gguf")
    parser.add_argument("--large-model", type=Path, default=Path("C:/Users/A/.lmstudio/models/canhdu/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS-GGUF/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf"))
    args = parser.parse_args()
    server = args.server or args.source_root / "build-native-5080-sm120/bin/llama-server.exe"
    prepare(args.output, args.source_root, server, [
        ("qwen3_0_6b_f16", args.small_model), ("qwen3_8_27b_mixed", args.large_model),
    ], native_output_contract=args.native_output_contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
