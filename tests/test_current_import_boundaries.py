"""Model imports and llama.cpp adapters share current nested GGUF evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.config import model_from_dict
from heterollm_sim.gguf_model_catalog import GGUFModelCatalog
from heterollm_sim.llama_scenario import (
    llama_cpp_gpu_layer_mapping, llama_final_norm_static_binding,
)
from heterollm_sim.llama_tensor_storage import _qualification
from heterollm_sim.model_artifacts import load_model_artifact, save_model_artifact
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


def _scenario(model):
    base = build_reference_scenario()
    return replace(base, model=model, workload=replace(base.workload, mtp=None),
                   placement=replace(base.placement, model_name=model.name))


@pytest.fixture
def gguf_model(tmp_path):
    return model_from_dict(GGUFModelCatalog(tmp_path).detail("qwen3-0_6b")["model"])


def _with_evidence(model, evidence, *, flat=False):
    attributes = deepcopy(model.graph.attributes)
    attributes["metadata"] = {} if flat else evidence
    return replace(model, graph=replace(model.graph, attributes=attributes),
                   metadata=evidence if flat else attributes)


def test_saved_derived_gguf_uses_nested_evidence_and_qualifies_after_artifact_roundtrip(tmp_path):
    catalog = GGUFModelCatalog(tmp_path)
    detail = catalog.derive("qwen3-0_6b", "current-derived", {"layer_count": 1}, save=True)
    restored = GGUFModelCatalog(tmp_path).detail(detail["preset"]["id"])
    model = model_from_dict(restored["model"])
    evidence = model.metadata["metadata"]
    assert evidence["gguf_derived_inventory"] is True
    assert evidence["gguf_preset_origin"] == detail["preset"]["gguf_source"]
    assert evidence["gguf_preset_changes"]["layer_count"]["after"] == 1
    assert not {"gguf_derived_inventory", "gguf_preset_origin", "gguf_preset_changes"}.intersection(model.metadata)
    saved = save_model_artifact(model, artifact_dir=tmp_path, artifact_id="current-derived")
    artifact = load_model_artifact(saved["artifact_id"], artifact_dir=tmp_path)
    loaded = model_from_dict(artifact["model"])
    audit = _qualification(_scenario(loaded), loaded.metadata["llama_cpp_tensor_storage_contract"])
    assert audit["qualified"], audit["reasons"]
    assert audit["embedding_binding"]["shape"] == [1024, 151936]


def test_final_norm_consumes_current_gguf_tensor_and_epsilon(gguf_model):
    evidence = gguf_model.metadata["metadata"]
    result = llama_final_norm_static_binding(_scenario(gguf_model), LlamaCppRuntimeConfig())
    assert result["status"] == "source_conditional"
    assert result["weight"]["name"] == "output_norm.weight"
    assert result["weight"]["n_bytes"] == evidence["gguf_output_norm_binding"]["n_bytes"]
    assert result["epsilon"] == evidence["gguf_norm_epsilon"]


def test_final_norm_does_not_read_flat_gguf_evidence(gguf_model):
    model = _with_evidence(gguf_model, deepcopy(gguf_model.metadata["metadata"]), flat=True)
    result = llama_final_norm_static_binding(_scenario(model), LlamaCppRuntimeConfig())
    assert result["status"] == "weight_binding_missing"


@pytest.mark.parametrize("flat", [False, True])
def test_final_norm_does_not_use_retired_binding_fields(gguf_model, flat):
    original = gguf_model.metadata["metadata"]
    evidence = {"final_norm_weight_binding": original["gguf_output_norm_binding"],
                "final_norm_epsilon": 0.125}
    model = _with_evidence(gguf_model, evidence, flat=flat)
    result = llama_final_norm_static_binding(_scenario(model), LlamaCppRuntimeConfig())
    assert result["status"] == "weight_binding_missing"


def test_final_norm_does_not_fill_missing_epsilon_from_retired_field(gguf_model):
    evidence = deepcopy(gguf_model.metadata["metadata"])
    evidence.pop("gguf_norm_epsilon")
    evidence["final_norm_epsilon"] = 0.125
    result = llama_final_norm_static_binding(
        _scenario(_with_evidence(gguf_model, evidence)), LlamaCppRuntimeConfig())
    assert result["status"] == "source_conditional"
    assert result["epsilon"] is None


@pytest.mark.parametrize("flat", [False, True])
def test_native_mtp_loading_mapping_requires_current_nested_evidence(gguf_model, flat):
    layers = gguf_model.num_layers
    evidence = {"gguf_declared_block_count": layers + 2,
                "gguf_imported_executable_layers": layers, "gguf_mtp_layer_count": 2}
    scenario = _scenario(_with_evidence(gguf_model, evidence, flat=flat))
    config = LlamaCppRuntimeConfig(gpu_layers=layers + 3)
    if flat:
        with pytest.raises(ValueError, match="without exact GGUF"):
            llama_cpp_gpu_layer_mapping(scenario, config)
    else:
        result = llama_cpp_gpu_layer_mapping(scenario, config)
        assert result["simulator_gpu_layers"] == layers + 1
        assert result["excluded_mtp_loading_units"] == 2
        assert result["mapping_applied"] is True
