"""The automatic producer accepts fresh GGUF inputs, never timing records."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cuda_graph_contract import require_cuda_graph_source_contract
from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    import compile_cuda_graph_structure as compiler
    import prepare_cuda_graph_cases as preparation
finally:
    sys.path.remove(str(TOOLS))

REQUIRE_LOCAL_DEVICE = compiler.require_local_device


def setup_case(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / "docs/cuda_graph_validation_2026-10-08/scenario_qwen3_0_6b_f16_512_128.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    model, probe = tmp_path / "new-qwen.gguf", tmp_path / "probe.exe"
    model.write_bytes(b"test metadata supplied separately")
    probe.write_bytes(b"mock subprocess only")
    payload["workload"]["metadata"]["native_model_path"] = str(model)
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")
    args = SimpleNamespace(model=model, probe=probe, scenario=scenario_path,
        source_root=tmp_path, output_dir=tmp_path / "output", prompt_tokens=512,
        output_tokens=128, context=768, warmups=2, repetitions=1, timeout=90, cuda_device=0)
    source_model = deepcopy(scenario_from_dict(payload).model)
    monkeypatch.setattr(compiler, "build_model_from_gguf", lambda _: source_model)
    monkeypatch.setattr(compiler, "require_local_device", lambda *a: {
        "index": "0", "name": "NVIDIA GeForce RTX 5080", "compute_capability": "12.0", "driver": "617.14"})
    gguf = SimpleNamespace(architecture="qwen3", vocab_size=source_model.vocabulary_size,
        metadata={"tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2})
    return args, payload, gguf


def fake_probe(args, records, *, corrupt=False):
    def run(command, **kwargs):
        records.append((command, kwargs["env"]))
        env = kwargs["env"]
        dry = "HETEROLLM_CUDA_GRAPH_DRY_RUN" in env
        _, _, labels, _ = compiler.request_plan(512, 128, 2, 1, 2)
        rows = [{"kind": "node_property", "id": 0, "bytes": "aabb"}]
        for index, label in enumerate(labels[:-1] if corrupt else labels):
            rows.append({"kind": "snapshot", "source_revision": SOURCE_REVISION, "call_id": index,
                "label": label, "device": 0, "context_id": "ctx", "graph_key": "key",
                "graph_uid": 1, "n_nodes": 1, "compatible": True, "dry_run": dry,
                "capture_only": not dry, "node_property_refs": [0]})
            if not dry:
                rows.append({"kind": "cuda_structure", "source_revision": SOURCE_REVISION,
                    "capture_only": True, "call_id": index, "nodes": [{"id": "a", "type": 0,
                    "function": "fn", "grid": [1, 1, 1], "block": [32, 1, 1], "shared_bytes": 0}],
                    "edges": [], "edge_data": []})
        Path(env["HETEROLLM_CUDA_GRAPH_TRACE"]).write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return SimpleNamespace(stdout=json.dumps({"status": "complete", "prompt": 512,
            "output": 128, "warmups": 2, "repeats": 1, "uses_sampled_tokens": False,
            "latency_measurements": False}))
    return run


@pytest.mark.parametrize("architecture", ["qwen2", "qwen3", "qwen35", "llama"])
def test_fresh_model_generates_only_dry_capture_and_bound_fragment(tmp_path, monkeypatch, architecture):
    args, payload, gguf = setup_case(tmp_path, monkeypatch)
    gguf.architecture = architecture
    records = []
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda *a, **k: SOURCE_REVISION)
    monkeypatch.setattr(compiler, "read_gguf_metadata_only", lambda _: gguf)
    monkeypatch.setattr(compiler.subprocess, "run", fake_probe(args, records))
    for key in compiler.MODE_VARIABLES:
        monkeypatch.setenv(key, "stale inherited value")
    fragment, artifact = compiler.compile_structure(args)
    assert artifact.is_file()
    assert len(records) == 2
    assert all(record[0][1] == str(args.model) for record in records)
    assert records[0][1].get("HETEROLLM_CUDA_GRAPH_DRY_RUN") == "1"
    assert "HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY" not in records[0][1]
    assert records[1][1].get("HETEROLLM_CUDA_GRAPH_CAPTURE_ONLY") == "1"
    assert "HETEROLLM_CUDA_GRAPH_DRY_RUN" not in records[1][1]
    assert all("GGML_CUDA_GRAPH_OPT" not in env and "GGML_CUDA_DISABLE_GRAPHS" not in env
               for _, env in records)
    assert fragment["producer"]["structural_calls"] == 386
    assert fragment["producer"]["live_trace_used"] is False
    assert fragment["program"]["request_stages"]["01_model_seq_rm_probe"] == "model_seq_rm_probe"
    for mode in ("dry", "capture"):
        require_cuda_graph_source_contract(args.output_dir / f"{mode}.jsonl", fragment["program"]["contract"])
    with pytest.raises(FileExistsError):
        compiler.compile_structure(args)


@pytest.mark.parametrize("mutation", ["architecture", "startup", "context", "model", "workload", "source", "model_graph", "weight_format", "weight_shape", "weight_offset", "policy", "experts", "derived"])
def test_invalid_inputs_fail_before_gpu_process(tmp_path, monkeypatch, mutation):
    args, payload, gguf = setup_case(tmp_path, monkeypatch)
    revision = SOURCE_REVISION
    if mutation == "architecture": gguf.architecture = "uncovered"
    elif mutation == "startup": gguf.metadata.pop("tokenizer.ggml.eos_token_id")
    elif mutation == "context": args.context = 640
    elif mutation == "model": payload["workload"]["metadata"]["native_model_path"] = "different.gguf"
    elif mutation == "workload": args.output_tokens = 64
    elif mutation == "model_graph": payload["model"]["graph"]["attributes"]["metadata"]["gguf_declared_block_count"] += 1
    elif mutation == "weight_format": payload["model"]["metadata"]["metadata"]["gguf_file_quantization"] = "Q8_0"
    elif mutation == "weight_shape": payload["model"]["graph"]["attributes"]["metadata"]["gguf_embedding_binding"]["shape"][0] += 1
    elif mutation == "weight_offset": payload["model"]["graph"]["attributes"]["metadata"]["gguf_embedding_binding"]["offset"] += 32
    elif mutation == "experts":
        gguf.architecture = "llama"
        gguf.metadata["llama.expert_count"] = 8
    elif mutation == "derived": payload["model"]["metadata"]["gguf_preset_changes"] = {"layer_count": {"before": 28, "after": 29}}
    elif mutation == "policy": payload["profiles"]["llama_cpp"]["policy"] = "generic"
    else: revision = "other-source"
    args.scenario.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda *a, **k: revision)
    monkeypatch.setattr(compiler, "read_gguf_metadata_only", lambda _: gguf)
    monkeypatch.setattr(compiler.subprocess, "run", lambda *a, **k: pytest.fail("must not start GPU process"))
    with pytest.raises(ValueError): compiler.compile_structure(args)


def test_catalog_labels_and_tensor_order_do_not_change_compilation(tmp_path, monkeypatch):
    args, payload, gguf = setup_case(tmp_path, monkeypatch)
    payload["model"]["name"] = "Qwen catalog display name"
    payload["model"]["metadata"].update(model_preset_id="catalog-id", gguf_preset_origin={"url": "source"},
                                          gguf_preset_changes={})
    payload["model"]["graph"]["tensors"].reverse()
    assert compiler.validate_inputs(args, payload, gguf) == [1, 2]


@pytest.mark.parametrize("field", ["size", "data_start", "directory_end", "tensor_count"])
def test_source_geometry_is_checked_before_source_labels_are_removed(tmp_path, monkeypatch, field):
    args, payload, gguf = setup_case(tmp_path, monkeypatch)
    source = dict(path="remote.gguf", size=100, data_start=32, directory_end=30, tensor_count=1)
    gguf.sources = ({**source, "path": "local.gguf"},)
    payload["model"]["metadata"]["metadata"]["gguf_sources"] = [source]
    assert compiler.validate_inputs(args, payload, gguf) == [1, 2]
    source[field] += 1
    with pytest.raises(ValueError, match="source file geometry"):
        compiler.validate_inputs(args, payload, gguf)


@pytest.mark.parametrize("mutation", [None, "identity", "tensor", "metadata", "split"])
def test_preset_full_directory_identity_and_metadata_are_checked(monkeypatch, mutation):
    from heterollm_sim.gguf_model_catalog import GGUFModelCatalog, inventory_to_gguf
    record = GGUFModelCatalog()._records()["qwen3-0_6b"]
    gguf = inventory_to_gguf(record["inventory"])
    gguf = replace(gguf, sources=({"path": "actual.gguf", "data_start": 0},))
    payload = {"model": {"metadata": {"model_preset_id": record["id"]}}}
    monkeypatch.setattr(GGUFModelCatalog, "_records", lambda self: {record["id"]: record})
    if mutation == "identity": gguf = replace(gguf, sha256="different")
    elif mutation == "tensor":
        # Even a tensor unused by a particular invocation must still match.
        last = replace(gguf.tensor_directory[-1], offset=gguf.tensor_directory[-1].offset + 32)
        gguf = replace(gguf, tensor_directory=(*gguf.tensor_directory[:-1], last))
    elif mutation == "metadata": gguf = replace(gguf, metadata={**gguf.metadata, "general.name": "different"})
    elif mutation == "split": gguf = replace(gguf, sources=(*gguf.sources, {"path": "second.gguf"}))
    if mutation is None:
        compiler.validate_preset_inventory(payload, gguf)
    else:
        with pytest.raises(ValueError): compiler.validate_preset_inventory(payload, gguf)


def test_attach_case_rejects_path_components(tmp_path):
    with pytest.raises(ValueError, match="case ID"):
        preparation.attach_graph_experiment(tmp_path, "../different", Path("unused"), Path("unused"))


def test_probe_failure_does_not_publish_fragment(tmp_path, monkeypatch):
    args, _, gguf = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda *a, **k: SOURCE_REVISION)
    monkeypatch.setattr(compiler, "read_gguf_metadata_only", lambda _: gguf)
    def fail(*a, **k): raise subprocess.CalledProcessError(2, a[0])
    monkeypatch.setattr(compiler.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError): compiler.compile_structure(args)
    assert not (args.output_dir / "program_fragment.json").exists()


def test_incomplete_structure_does_not_publish_fragment(tmp_path, monkeypatch):
    args, _, gguf = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda *a, **k: SOURCE_REVISION)
    monkeypatch.setattr(compiler, "read_gguf_metadata_only", lambda _: gguf)
    monkeypatch.setattr(compiler.subprocess, "run", fake_probe(args, [], corrupt=True))
    with pytest.raises(ValueError, match="exactly"): compiler.compile_structure(args)
    assert not (args.output_dir / "program_fragment.json").exists()


def test_configured_output_contract_does_not_require_native_results(tmp_path, monkeypatch):
    _, payload, _ = setup_case(tmp_path, monkeypatch)
    scenario = preparation.apply_configured_output_contract(scenario_from_dict(payload), SOURCE_REVISION)
    assert scenario.host_output_contract.vocabulary_size == scenario.model.vocabulary_size
    assert scenario.sampling_policy.top_k == 40
    assert scenario.sampling_policy.temperature == 0
    assert scenario.workload.metadata["native_output_contract_evidence"]["native_settings_record"] is None
    with pytest.raises(ValueError, match="pinned"):
        preparation.apply_configured_output_contract(scenario, "unsupported")


def test_new_fragment_attaches_and_survives_frontend_normalization(tmp_path, monkeypatch):
    from heterollm_sim.cuda_graph_contract import validate_cuda_graph_scenario_contract
    from heterollm_sim.llama_scenario import prepare_llama_scenario
    from heterollm_sim.web import scenario_to_payload

    args, payload, gguf = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda *a, **k: SOURCE_REVISION)
    monkeypatch.setattr(compiler, "read_gguf_metadata_only", lambda _: gguf)
    monkeypatch.setattr(compiler.subprocess, "run", fake_probe(args, []))
    _, fragment_path = compiler.compile_structure(args)
    (tmp_path / "scenario_fresh_512_128.json").write_text(json.dumps(payload), encoding="utf-8")
    costs = Path(__file__).parents[1] / "docs/cuda_graph_validation_2026-10-08/runtime_typed_chain_measurements.json"
    preparation.attach_graph_experiment(tmp_path, "fresh", fragment_path, costs)
    for mode in ("off", "on"):
        prepared = json.loads((tmp_path / f"scenario_fresh_graph_{mode}.json").read_text(encoding="utf-8"))
        scenario = prepare_llama_scenario(scenario_from_dict(prepared))
        validate_cuda_graph_scenario_contract(scenario)
        normalized = scenario_to_payload(scenario)
        scenario = prepare_llama_scenario(scenario_from_dict(normalized))
        validate_cuda_graph_scenario_contract(scenario)


@pytest.mark.parametrize("row,accepted", [
    ("0, NVIDIA GeForce RTX 5080, 12.0, 617.14", True),
    ("0, NVIDIA A100-SXM4-80GB, 8.0, 617.14", False),
    ("0, NVIDIA GeForce RTX 5080, 12.0, 999.99", False),
])
def test_compiler_binds_actual_gpu_and_driver(tmp_path, monkeypatch, row, accepted):
    args, payload, _ = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda *a, **k: row)
    if accepted:
        assert REQUIRE_LOCAL_DEVICE(args, payload)["driver"] == "617.14"
    else:
        with pytest.raises(ValueError, match="GPU/driver"):
            REQUIRE_LOCAL_DEVICE(args, payload)


def test_compiler_rejects_ambiguous_multi_gpu_ordinals(tmp_path, monkeypatch):
    args, payload, _ = setup_case(tmp_path, monkeypatch)
    row = "0, NVIDIA GeForce RTX 5080, 12.0, 617.14\n1, NVIDIA GeForce RTX 5080, 12.0, 617.14"
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda *a, **k: row)
    with pytest.raises(ValueError, match="exactly one physical GPU"):
        REQUIRE_LOCAL_DEVICE(args, payload)
