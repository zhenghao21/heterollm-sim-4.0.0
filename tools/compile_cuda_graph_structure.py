"""Generate source-assisted CUDA structure inputs without target-model timings.

This invokes the diagnostic native runtime and needs its GPU and model weights.
Dry compilation and capture-only compilation are separate child processes;
neither live event records nor a previous model's structure is an input.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cuda_graph_contract import build_cuda_graph_contract
from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION
from heterollm_sim.cuda_graph_serving import _load_capture_topologies
from heterollm_sim.cuda_graph_structure import load_cuda_graph_structure
from heterollm_sim.gguf_parity import build_model_from_gguf, read_gguf_metadata_only
from heterollm_sim.serde import to_primitive


MODE_VARIABLES = ("HETEROLLM_CUDA_GRAPH_TRACE", "HETEROLLM_CUDA_GRAPH_DRY_RUN",
    "HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY", "HETEROLLM_CUDA_GRAPH_LABEL",
    "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_GRAPH_OPT", "LLAMA_GRAPH_REUSE_DISABLE")


def validate_inputs(args, payload, gguf):
    for key in ("prompt_tokens", "output_tokens", "context", "repetitions"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if args.warmups < 0 or args.prompt_tokens > 512 or args.prompt_tokens + args.output_tokens > args.context:
        raise ValueError("probe requires prompt <= 512, prompt+output <= context and nonnegative warmups")
    if type(args.cuda_device) is not int or args.cuda_device < 0 or args.timeout <= 0:
        raise ValueError("cuda_device must be nonnegative and timeout must be positive")
    if args.context % 256:
        raise ValueError("context must be a multiple of 256 to match native effective context")
    if gguf.architecture not in {"qwen3", "qwen35"}:
        raise ValueError("source-assisted scenario coverage currently requires qwen3 or qwen35")
    # Do not guess tokenizer defaults: the harness uses the resolved BOS/EOS.
    startup_tokens = []
    for key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
        value = gguf.metadata.get(key)
        if type(value) is not int or gguf.vocab_size is None or not 0 <= value < gguf.vocab_size:
            raise ValueError("probe preparation requires explicit valid GGUF BOS and EOS IDs")
        startup_tokens.append(value)
    scenario = scenario_from_dict(payload)
    config = scenario.llama_cpp_config
    expected = {"policy": "llama_cpp", "context": args.context, "batch": 512, "ubatch": 512, "parallel": 1,
        "threads": 16, "threads_batch": 16, "gpu_layers": -1, "flash_attn": False,
        "kv_type_k": "f16", "kv_type_v": "f16", "kv_unified": True,
        "offload_kqv": True, "op_offload": True, "device_memory_tiering": False,
        "split_mode": "layer", "main_gpu": 0, "warmup": True, "tensor_split": None,
        "device": None, "mmap": True, "mlock": False, "numa": None,
        "cpu_range": None, "cpu_range_batch": None}
    if config is None or any(getattr(config, key) != value for key, value in expected.items()):
        raise ValueError("scenario runtime differs from fixed probe configuration")
    if scenario.workload.mtp is not None or scenario.workload.scheduler.max_num_seqs != 1:
        raise ValueError("probe supports single-sequence non-MTP scenarios only")
    model_path = scenario.workload.metadata.get("native_model_path")
    if not model_path or Path(model_path).resolve() != args.model.resolve():
        raise ValueError("scenario GGUF path differs from requested model")
    authored, imported = to_primitive(scenario.model), to_primitive(build_model_from_gguf(gguf))
    for model in (authored, imported):
        # Metadata-only import intentionally does not reread/hash weight data.
        # Compare every executable node, tensor geometry and quantization type.
        for attributes in (model["metadata"], model["graph"]["attributes"]):
            attributes.get("metadata", {}).pop("gguf_sha256", None)
            attributes.pop("ui", None)
            attributes.pop("artifact_id", None)
    if authored != imported:
        raise ValueError("scenario model graph/geometry/weight formats differ from GGUF; reimport the selected model")
    if not scenario.workload.requests or any(
            request.prompt_tokens != args.prompt_tokens or request.output_tokens != args.output_tokens
            for request in scenario.workload.requests):
        raise ValueError("base scenario requests differ from explicit probe workload")
    return startup_tokens


def request_plan(prompt, output, warmups, repeats, startup_count):
    requests, prefixes, labels = [], {}, []
    def add(name, prefix, prompt_count, output_count):
        request_id = f"{len(requests):02d}_{name}"
        requests.append({"request_id": request_id, "arrival_ns": 0.0,
                         "prompt_tokens": prompt_count, "output_tokens": output_count})
        prefixes[request_id] = prefix
        if prefix in {"model_warmup", "model_seq_rm_probe"}:
            labels.append(prefix)
        else:
            labels.append(prefix + ":prefill")
            labels.extend(prefix + f":decode:{index}" for index in range(1, output_count))
        return request_id
    add("model_warmup", "model_warmup", startup_count, 1)
    add("model_seq_rm_probe", "model_seq_rm_probe", 2, 1)
    for index in range(warmups):
        add(f"warmup{index}", f"warmup:{index}", prompt, output)
    for index in range(repeats):
        comparison = add(f"measured{index}", f"measured:{index}", prompt, output)
    return requests, prefixes, labels, comparison


def build_fragment(payload, requests, prefixes, comparison, paths):
    program = {"schema": "heterollm.cuda-graph-source-program/v1", "graph_enabled": True,
        "dry_program_path": str(paths["dry"]), "capture_program_path": str(paths["capture"]),
        "request_prefixes": prefixes,
        "request_stages": {key: prefix if prefix in {"model_warmup", "model_seq_rm_probe"} else "request"
                           for key, prefix in prefixes.items()}}
    compiled = deepcopy(payload)
    workload = compiled["workload"]
    template = workload["requests"][0]
    workload["requests"] = [{**deepcopy(template), **request} for request in requests]
    workload.update(request_count=len(requests), prompt_tokens=0, output_tokens=0)
    workload["scheduler"]["max_num_seqs"] = 1
    workload["metadata"].update(cuda_graph_structural_program=program,
        cuda_graph_comparison_request_id=comparison, native_ctx_checkpoints=0)
    program["contract"] = build_cuda_graph_contract(scenario_from_dict(compiled))
    return {"schema": "heterollm.cuda-graph-preparation/v1", "requests": requests,
            "comparison_request_id": comparison, "program": program}


def prepend_contract(path, contract):
    # Stream large topology files; do not hold a second copy in memory.
    temporary = path.with_suffix(path.suffix + ".with-contract")
    with temporary.open("wb") as target, path.open("rb") as source:
        target.write((json.dumps({"kind": "source_contract", "contract": contract},
                                 separators=(",", ":")) + "\n").encode("utf-8"))
        while block := source.read(1024 * 1024):
            target.write(block)
    temporary.replace(path)


def probe_environment(mode, trace):
    env = os.environ.copy()
    for key in MODE_VARIABLES:
        env.pop(key, None)
    for key in ("GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_CUBLAS_COMPUTE_TYPE",
                "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "GGML_OP_OFFLOAD_MIN_BATCH"):
        if key in env:
            raise ValueError(f"unsupported inherited dispatch override: {key}")
    env["HETEROLLM_CUDA_GRAPH_TRACE"] = str(trace.resolve())
    env["HETEROLLM_CUDA_GRAPH_DRY_RUN" if mode == "dry" else "HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY"] = "1"
    return env


def require_local_device(args, payload):
    """The bundled probe/cost setup is qualified only for this local platform."""
    scenario = scenario_from_dict(payload)
    gpus = [component for component in scenario.hardware.components if component.normalized_kind == "gpu"]
    if len(gpus) != 1:
        raise ValueError("structural compiler currently supports exactly one scenario GPU")
    profile = scenario.component_profiles["gpu"][gpus[0].cost_profile_id].kernel_model
    if (profile is None or profile.hardware_id != "nvidia-rtx-5080"
            or profile.architecture != "sm120-source-rule-analytical"):
        raise ValueError("bundled structural compiler requires the RTX 5080 sm120 scenario profile")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = [[field.strip() for field in row] for row in csv.reader(raw.splitlines())]
    if len(rows) != 1:
        raise ValueError("structural compiler currently requires exactly one physical GPU to bind its CUDA ordinal")
    selected = [row for row in rows if len(row) == 4 and row[0] == str(args.cuda_device)]
    if len(selected) != 1 or selected[0][1:] != ["NVIDIA GeForce RTX 5080", "12.0", "617.14"]:
        raise ValueError("selected GPU/driver differs from supported RTX 5080, compute 12.0, driver 617.14")
    return dict(zip(("index", "name", "compute_capability", "driver"), selected[0]))


def compile_structure(args):
    for path in (args.model, args.probe, args.scenario):
        if not path.is_file():
            raise FileNotFoundError(path)
    revision = subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("source checkout revision differs from the supported structural compiler")
    payload = json.loads(args.scenario.read_text(encoding="utf-8"))
    gguf = read_gguf_metadata_only(args.model)
    startup = validate_inputs(args, payload, gguf)
    device = require_local_device(args, payload)
    requests, prefixes, labels, comparison = request_plan(
        args.prompt_tokens, args.output_tokens, args.warmups, args.repetitions, len(startup))
    output = args.output_dir.resolve()
    paths = {kind: output / f"{kind}.jsonl" for kind in ("dry", "capture")}
    artifact = output / "program_fragment.json"
    for path in (*paths.values(), artifact, output / "dry.log", output / "capture.log"):
        if path.exists():
            raise FileExistsError(f"use a new output directory; refusing to replace {path}")
    envs = {mode: probe_environment(mode, path) for mode, path in paths.items()}
    for env in envs.values():
        env.pop("GGML_CUDA_DEVICES", None)
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    fragment = build_fragment(payload, requests, prefixes, comparison, paths)
    output.mkdir(parents=True, exist_ok=True)
    command = [str(args.probe.resolve()), str(args.model.resolve()), str(args.prompt_tokens),
               str(args.output_tokens), str(args.context), str(args.warmups), str(args.repetitions)]
    for mode, trace in paths.items():
        with (output / f"{mode}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=envs[mode], stdout=subprocess.PIPE, stderr=log,
                text=True, timeout=args.timeout, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
        expected = {"status": "complete", "prompt": args.prompt_tokens, "output": args.output_tokens,
                    "warmups": args.warmups, "repeats": args.repetitions,
                    "uses_sampled_tokens": False, "latency_measurements": False}
        if not records or any(records[-1].get(key) != value for key, value in expected.items()):
            raise ValueError(f"{mode} probe did not confirm the requested no-timing workload")
    dry = load_cuda_graph_structure(paths["dry"])
    capture = _load_capture_topologies(paths["capture"])
    if [call.label for call in dry.calls] != labels or set(capture) != set(labels):
        raise ValueError("probe outputs do not cover exactly the startup/warmup/measured invocation plan")
    if any(call.device != 0 for call in dry.calls) or len({call.context_id for call in dry.calls}) != 1:
        raise ValueError("probe emitted unsupported multi-device/backend structure")
    for path in paths.values():
        prepend_contract(path, fragment["program"]["contract"])
    fragment["producer"] = {"source_revision": revision, "probe": str(args.probe.resolve()),
            "model": str(args.model.resolve()), "base_scenario": str(args.scenario.resolve()),
            "command": command, "ctx_checkpoints": 0, "startup_token_ids": startup,
            "cuda_visible_devices": str(args.cuda_device),
            "device": device,
            "target_llm_latency_used": False, "live_trace_used": False,
            "requires_native_runtime_and_gpu": True, "structural_calls": len(dry.calls),
            "scope": "source_assisted_dry_and_capture_only_not_cpu_only_prediction"}
    # The fragment is the completion artifact; failed probes never publish it.
    artifact.write_text(json.dumps(fragment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return fragment, artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "probe", "scenario", "source-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prompt-tokens", "output-tokens", "context", "warmups", "repetitions"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--cuda-device", type=int, default=0,
                        help="one physical CUDA device exposed to each probe child")
    args = parser.parse_args()
    _, artifact = compile_structure(args)
    print(artifact)


if __name__ == "__main__":
    main()
