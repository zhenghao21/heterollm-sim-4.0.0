"""Freeze diagnostic-only grid predictions without executing native or using answers.

Inputs: native-repeatability protocol, sibling prompts.json, and a frozen hardware
snapshot (protocol.hardware_snapshot_ref, or sibling hardware.json). Paths to data
are resolved against --data-root; Python imports always come from this source copy.
Outputs are exclusive-create prediction files and manifest.json. No calibration,
native measurements, profiler files, live hardware probes or native subprocesses
are consumed. Formal eligibility is deliberately false for every cell.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from heterollm_sim import reporting
from heterollm_sim.gguf_parity import read_gguf_metadata, build_model_from_gguf
from heterollm_sim.serde import stable_hash
from tools.native_llama_compare import build_matching_scenario, _simulator_request_timing

METRICS = ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")
RUNTIME = "source/llama.cpp-native-thread-control/build-native-thread-control/bin/llama-server.exe"
LIMITATIONS = [
    "CPU affinity mask, strict worker binding, polling and OpenMP controls have no typed simulator model.",
    "Native GPU clock is fixed at 2400 MHz; the builder still uses its 2617 MHz analytical input. No cost correction is applied.",
    "Native unified KV total context is 2048*C; simulator slot context is 2048. Physical shared-pool equivalence is unproven.",
    "The new thread-control binary and op-offload source identity cannot inherit the old semantic binary dispatch contract.",
    "Hybrid/recurrent ubatch behavior is unproven. These are diagnostic predictions, not accuracy acceptance evidence.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def file_ref(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "size_bytes": path.stat().st_size}


def read_document(path, expected_sha=None):
    ref = file_ref(path)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("input changed during read: " + str(path))
    if expected_sha and ref["sha256"] != expected_sha:
        raise ValueError("input SHA256 mismatch: " + str(path))
    document = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("JSON object required: " + str(path))
    return document, ref


def write_new(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    payload["content_sha256"] = stable_hash(payload)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return file_ref(path)


def resolve_data(path, data_root):
    path = Path(path)
    resolved = (path if path.is_absolute() else data_root / path).resolve(strict=True)
    if not resolved.is_relative_to(data_root):
        raise ValueError("input outside explicit data root: " + str(resolved))
    return resolved


def referenced_document(value, default, data_root):
    expected = value.get("sha256") if isinstance(value, dict) else None
    path = value.get("path") if isinstance(value, dict) else value
    path = Path(default) if path is None else resolve_data(path, data_root)
    return read_document(path, expected)


def expand_cells(protocol):
    """Expand declarative jobs/conditions only; process blocks repeat one prediction."""
    if protocol.get("schema") != "native-repeatability-protocol/v1":
        raise ValueError("native-repeatability-protocol/v1 required")
    jobs = protocol.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("nonempty protocol jobs required")
    cells, seen = [], set()
    for index, job in enumerate(jobs):
        base = {**protocol.get("defaults", {}), **job}
        job_id = str(job.get("id", f"job{index:03d}"))
        conditions = job.get("conditions", protocol.get("conditions", [{"id": "baseline"}]))
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("nonempty conditions required")
        for condition in conditions:
            cell = {**base, **condition}
            cell["job_id"] = job_id
            cell["condition_id"] = str(condition.get("id", "baseline"))
            cell["cell_id"] = job_id + "__" + cell["condition_id"]
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", cell["cell_id"]) or cell["cell_id"] in seen:
                raise ValueError("unsafe or duplicate cell_id: " + cell["cell_id"])
            seen.add(cell["cell_id"])
            cell["environment"] = {**protocol.get("runtime_environment", {}), **protocol.get("environment", {}),
                **protocol.get("defaults", {}).get("environment", {}),
                **job.get("environment", {}), **condition.get("environment", {})}
            cells.append(cell)
    return cells


def source_identity():
    paths = sorted(set((ROOT / "src").rglob("*.py")) | set((ROOT / "tools").glob("*.py")))
    refs = [file_ref(path) for path in paths]
    return {"execution_source_root": str(ROOT), "sha256": stable_hash(refs), "files": refs}


def frozen_hardware(protocol, campaign, data_root):
    inline = protocol.get("hardware_snapshot", protocol.get("hardware"))
    if isinstance(inline, dict) and "gpu" in inline:
        return inline, {"source": "protocol_inline_frozen_probe", "sha256": stable_hash(inline)}
    value = next((protocol[k] for k in ("hardware_snapshot_ref", "hardware_ref", "hardware_probe_ref", "hardware_snapshot_path") if k in protocol), None)
    document, ref = referenced_document(value, campaign / "hardware.json", data_root)
    hardware = document.get("hardware", document.get("hardware_snapshot", document))
    if not isinstance(hardware, dict) or "gpu" not in hardware:
        raise ValueError("frozen actual hardware probe must include gpu")
    return hardware, ref


def prompt_for(cell, prompt_models, model_path, data_root):
    p = cell.get("expected_prompt_tokens", cell.get("prompt_tokens"))
    if p is None and isinstance(cell.get("prompt_token_ids"), list):
        p = len(cell["prompt_token_ids"])
    if isinstance(p, bool) or not isinstance(p, int) or p <= 0:
        raise ValueError("explicit positive expected_prompt_tokens required")
    matching = [item for item in prompt_models
                if (Path(item["model"]) if Path(item["model"]).is_absolute()
                    else data_root / item["model"]).resolve() == model_path]
    if len(matching) != 1:
        raise ValueError("exactly one prompts.json model entry required")
    entry = matching[0]
    prompt = entry.get("prompts", {}).get(str(p))
    if not isinstance(prompt, dict):
        raise ValueError("prompt length absent from prompts.json")
    ids = prompt.get("prompt_token_ids", prompt.get("ids"))
    if not isinstance(ids, list) or len(ids) != p or any(type(v) is not int or v < 0 for v in ids):
        raise ValueError("prompt token IDs/count invalid")
    if cell.get("prompt_token_ids") is not None and cell["prompt_token_ids"] != ids:
        raise ValueError("protocol prompt IDs differ from frozen prompts.json")
    ids_hash = stable_hash(ids)
    if prompt.get("ids_sha256") is not None and prompt["ids_sha256"] != ids_hash:
        raise ValueError("prompt token IDs SHA256 mismatch")
    return p, {"model_id": entry.get("model_id"), "model_sha256": entry.get("model_sha256"),
               "prompt_tokens": p, "prompt_token_ids_sha256": ids_hash}


def configuration(cell):
    config = {"ctx": 2048, "parallel": cell.get("parallel", 1), "batch": 64,
              "ubatch": 64, "threads": 16, "gpu_layers": cell.get("gpu_layers", -1),
              "seed": 42, "coherent_dma_mode": "pipelined", "op_offload": True}
    for key in ("parallel",):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("batch", "ubatch", "threads", "seed"):
        if key in cell and cell[key] != config[key]:
            raise ValueError("protocol differs from fixed diagnostic configuration: " + key)
    if type(config["gpu_layers"]) is not int:
        raise ValueError("gpu_layers must be an integer")
    if cell.get("kv_unified_per_slot", 2048) != 2048:
        raise ValueError("kv_unified_per_slot must be 2048")
    output = cell.get("output", cell.get("output_tokens"))
    if type(output) is not int or output < 1:
        raise ValueError("positive output token count required")
    return config, output


def aggregate(requests, planned):
    result = {}
    for key in METRICS:
        values = [r[key] for r in requests if r.get(key) is not None]
        result[key] = {"planned_requests": planned, "observed_requests": len(values),
            "missing_requests": max(0, planned - len(values)),
            "mean_ms": statistics.mean(values) if len(values) == planned else None,
            "median_ms": statistics.median(values) if len(values) == planned else None,
            "min_ms": min(values) if len(values) == planned else None,
            "max_ms": max(values) if len(values) == planned else None}
    return result


def predict_cell(cell, *, prompt_models, data_root, hardware, runtime, model_cache):
    """The only simulation entry; accepts static inputs and never native answers."""
    config, output = configuration(cell)
    model_path = resolve_data(cell["model"], data_root)
    p, prompt_identity = prompt_for(cell, prompt_models, model_path, data_root)
    if p + output > 2048:
        raise ValueError("prompt + output exceeds per-slot context")
    cache_key = str(model_path)
    if cache_key not in model_cache:
        try:
            gguf = read_gguf_metadata(model_path)
            model_cache[cache_key] = (gguf, build_model_from_gguf(gguf))
        except Exception as exc:
            model_cache[cache_key] = exc
    cached = model_cache[cache_key]
    if isinstance(cached, Exception):
        raise ValueError("GGUF model initialization failed: " + str(cached)) from cached
    gguf, model = cached
    if prompt_identity["model_sha256"] and prompt_identity["model_sha256"] != gguf.sha256:
        raise ValueError("GGUF differs from frozen prompt model SHA256")
    environment = cell["environment"]
    if not isinstance(environment, dict) or any(v is not None and not isinstance(v, str) for v in environment.values()):
        raise ValueError("runtime environment must explicitly map names to strings or null (unset)")
    scenario = build_matching_scenario(p, output, model=model, hardware_snapshot=hardware,
        runtime_binary=runtime, runtime_environment=dict(environment), **config)
    metadata = dict(scenario.workload.metadata)
    metadata["serving_runtime"] = {**metadata.get("serving_runtime", {}), "kv_slot_context_tokens": 2048}
    scenario = replace(scenario, workload=replace(scenario.workload, metadata=metadata))
    result = reporting.run_scenario(scenario, retention_policy="aggregate")
    requests = []
    for rank, (rid, metric) in enumerate(sorted(result.metrics.request_metrics.items())):
        timing = _simulator_request_timing(result, metric)
        record = {"request_id": str(rid), "request_index": rank,
                  "prompt_tokens": p, "requested_output_tokens": output,
                  "visible_output_tokens": getattr(metric, "visible_output_tokens", None)}
        for key in METRICS:
            value = timing.get(key)
            record[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None
        record["engine_start_source"] = timing.get("engine_start_source")
        record["engine_last_token_source"] = timing.get("engine_last_token_source")
        requests.append(record)
    complete = (len(requests) == config["parallel"]
                and all(r["visible_output_tokens"] == output for r in requests)
                and all(r[k] is not None for r in requests for k in METRICS))
    identity = {"model": {"path": str(model_path), "sha256": gguf.sha256}, "prompt": prompt_identity,
        "configuration": config, "requested_output_tokens": output,
        "native_unified_total_context": 2048 * config["parallel"], "simulator_slot_context": 2048,
        "runtime_environment": dict(environment),
        "unmodeled_native_controls": {k: cell.get(k) for k in ("worker_cpu_mask", "poll", "priority", "threads_batch")}}
    return {"status": "diagnostic_prediction" if complete else "incomplete_prediction",
        "reason": None if complete else "missing request, wrong output count, or missing engine boundary timing",
        "input_identity": identity, "input_sha256": stable_hash(identity),
        "requests": requests, "aggregate": aggregate(requests, config["parallel"])}


def build_predictions(protocol_path, output, *, data_root=None):
    protocol_path, output = Path(protocol_path).resolve(strict=True), Path(output).resolve()
    protocol, protocol_ref = read_document(protocol_path)
    data_root = Path(data_root or protocol.get("data_root") or ROOT).resolve(strict=True)
    cells = expand_cells(protocol)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite or mix predictions: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    prompts_value = protocol.get("prompts_ref", protocol.get("prompt_manifest_ref", protocol.get("prompts_path")))
    prompts, prompts_ref = referenced_document(prompts_value, protocol_path.parent / "prompts.json", data_root)
    hardware, hardware_ref = frozen_hardware(protocol, protocol_path.parent, data_root)
    runtime = resolve_data(protocol.get("exe", RUNTIME), data_root)
    expected_runtime = (data_root / RUNTIME).resolve()
    if runtime != expected_runtime:
        raise ValueError("the actual new thread-control runtime is required")
    runtime_refs = [file_ref(runtime), *[file_ref(p) for p in sorted(runtime.parent.glob("*.dll"))]]
    new_source = data_root / "source/llama.cpp-native-thread-control"
    source_binding_refs = [file_ref(new_source / name) for name in
        ("evidence/build_receipt.json", "evidence/source_manifest.json") if (new_source / name).is_file()]
    source = source_identity()
    common = {"protocol_ref": protocol_ref, "prompts_ref": prompts_ref, "hardware_ref": hardware_ref,
        "hardware_sha256": stable_hash(hardware), "source_sha256": source["sha256"],
        "runtime_identity": {"server": str(runtime), "sha256": stable_hash(runtime_refs), "files": runtime_refs,
            "new_source_binding_refs": source_binding_refs},
        "execution_source_root": str(ROOT), "data_root": str(data_root)}
    cache, entries = {}, []
    for cell in cells:
        created = now()
        try:
            prediction = predict_cell(cell, prompt_models=prompts["models"], data_root=data_root,
                hardware=hardware, runtime=runtime, model_cache=cache)
        except Exception as exc:
            planned = cell.get("parallel", 1)
            prediction = {"status": "failed", "reason": type(exc).__name__ + ": " + str(exc),
                "input_identity": {"cell_id": cell["cell_id"], "configuration": {k: cell.get(k) for k in
                    ("model", "expected_prompt_tokens", "output", "parallel", "gpu_layers")}},
                "requests": [{"request_index": i, **dict.fromkeys(METRICS), "reason": str(exc)}
                             for i in range(planned if type(planned) is int and planned > 0 else 1)],
                "aggregate": aggregate([], planned if type(planned) is int and planned > 0 else 1)}
            prediction["input_sha256"] = stable_hash(prediction["input_identity"])
        document = {"schema": "native-grid-prediction/v1", "created_utc": created,
            "finished_utc": now(), "cell_id": cell["cell_id"], "job_id": cell["job_id"],
            "condition_id": cell["condition_id"], "formal_prediction_eligible": False,
            "formal_prediction_ineligibility_reasons": list(LIMITATIONS), "calibration_applied": False,
            "native_answers_used": False, "input_provenance": common, **prediction}
        ref = write_new(output / (cell["cell_id"] + ".prediction.json"), document)
        entries.append({"cell_id": cell["cell_id"], "job_id": cell["job_id"], "condition_id": cell["condition_id"],
                        "status": prediction["status"], "reason": prediction["reason"], "prediction_ref": ref})
        print(f"[{len(entries)}/{len(cells)}] {cell['cell_id']}: {prediction['status']}", flush=True)
    manifest = {"schema": "native-grid-prediction-manifest/v1", "created_utc": now(),
        "planned_cells": len(cells), "prediction_files": len(entries),
        "successful_cells": sum(e["status"] == "diagnostic_prediction" for e in entries),
        "failed_or_incomplete_cells": sum(e["status"] != "diagnostic_prediction" for e in entries),
        "formal_prediction_eligible": False, "formal_prediction_acceptance": False,
        "formal_prediction_ineligibility_reasons": list(LIMITATIONS),
        "static_source": source, "input_provenance": common, "cells": entries,
        "prediction_refs": [e["prediction_ref"] for e in entries]}
    write_new(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    build_predictions(args.protocol, args.output, data_root=args.data_root)


if __name__ == "__main__":
    main()
