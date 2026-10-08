"""The bundled mixed-quantization preset keeps the native comparison structure."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from heterollm_sim.config import model_from_dict
from heterollm_sim.ir import model_graph_execution_view
from heterollm_sim.model_catalog import ModelCatalog
from heterollm_sim.model_presets import (
    UnsupportedPresetError,
    get_model_preset,
    materialize_model_payload,
    materialize_preset_definition,
)
from heterollm_sim.planner import _f32_hidden_storage_enabled
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


PRESET_ID = "qwen3_8-27b-iq3-s-iq4-xs"
ROOT = Path(__file__).resolve().parents[1]


def test_gguf_preset_is_searchable_and_materializes_without_local_weights(tmp_path):
    catalog = ModelCatalog(cache_dir=tmp_path)
    page = catalog.page(query="Qwen3.8-27B", architecture="qwen35", model_kind="hybrid")
    assert page["total"] == 1
    assert page["items"][0]["id"] == PRESET_ID
    detail = catalog.detail(PRESET_ID)
    metadata = detail["preset"]
    assert metadata["generation_allowed"] is True
    assert metadata["layer_count"] == 64
    assert metadata["provenance_status"] == "gguf_tensor_metadata"
    assert metadata["source_sha"] is None
    assert metadata["config_hash"] is None
    assert metadata["architecture_evidence"]["config_url"] == ""
    model = model_from_dict(detail["model"])
    view = model_graph_execution_view(model.graph, schema_version=model.schema_version)
    assert model.name == "Qwen3.8-27B"
    assert len(view.layer_instances) == 64
    assert model.graph.attributes["metadata"]["gguf_physical_weight_bytes"] == 14854119424


def test_gguf_preset_preserves_comparison_graph_and_independent_materializations():
    original = json.loads((ROOT / "docs/cuda_graph_validation_2026-10-08" /
        "scenario_qwen3_8_27b_mixed_graph_off.json").read_text(encoding="utf-8"))["model"]
    payload = materialize_model_payload(PRESET_ID)
    for field in ("operators", "tensors"):
        assert payload["graph"][field] == original["graph"][field]
    before, after = (model_from_dict(item) for item in (original, payload))
    old_view, new_view = (model_graph_execution_view(item.graph, schema_version=item.schema_version)
                          for item in (before, after))
    assert old_view.layer_instances == new_view.layer_instances
    assert old_view.embedding_weight_bytes == new_view.embedding_weight_bytes
    assert old_view.output_weight_bytes == new_view.output_weight_bytes
    payload["graph"]["operators"][0]["parameters"]["max_sequence_length"] = 1
    assert materialize_model_payload(PRESET_ID)["graph"]["operators"] == original["graph"]["operators"]


@pytest.mark.parametrize("changes", [
    {"model_payload_resource": "../../pyproject.toml"},
    {"preset_id": "untrusted-import"},
])
def test_imported_definitions_cannot_select_bundled_resource(changes):
    definition = replace(get_model_preset(PRESET_ID), **changes)
    with pytest.raises(UnsupportedPresetError, match="只有内置模型预设"):
        materialize_preset_definition(definition)


def test_model_hidden_storage_declaration_is_runtime_scoped_and_workload_overridable():
    base = build_llama_default_scenario()
    model = model_from_dict(materialize_model_payload(PRESET_ID))
    workload = replace(base.workload, metadata={})
    native = replace(base, model=model, workload=workload,
                     llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=-1))
    assert _f32_hidden_storage_enabled(native) is True
    assert _f32_hidden_storage_enabled(replace(native, llama_cpp_config=None)) is False
    assert _f32_hidden_storage_enabled(replace(native, model=base.model)) is False
    disabled = replace(workload, metadata={"llama_cpp_f32_hidden_storage": False})
    assert _f32_hidden_storage_enabled(replace(native, workload=disabled)) is False
    for value in ("true", 1, None):
        invalid = replace(workload, metadata={"llama_cpp_f32_hidden_storage": value})
        with pytest.raises(ValueError, match="explicit boolean"):
            _f32_hidden_storage_enabled(replace(native, workload=invalid))
    invalid_model = replace(model, metadata={"llama_cpp_f32_hidden_storage": "true"})
    with pytest.raises(ValueError, match="explicit boolean"):
        _f32_hidden_storage_enabled(replace(native, model=invalid_model))
