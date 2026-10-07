import pytest

from heterollm_sim.model_presets import (
    get_model_preset,
    materialize_model_payload,
    materialize_preset_definition,
    list_model_presets,
    UnsupportedPresetError,
)
from heterollm_sim.model_catalog import definition_from_huggingface, _import_preset_id
from heterollm_sim.config import model_from_dict


def test_bundled_catalog_contains_only_qwen_and_llama_families():
    presets = list_model_presets()
    assert len(presets) == 21
    assert {item["family"] for item in presets} == {
        "Qwen2.5",
        "Qwen3",
        "Llama3.1",
        "Llama3.2",
        "Llama3.3",
    }
    assert all(item["source_repo"].startswith(("Qwen/", "meta-llama/")) for item in presets)


def test_every_bundled_preset_materializes_to_an_executable_graph():
    for item in list_model_presets():
        payload = materialize_model_payload(item["id"])
        graph = payload["graph"]
        assert graph["executable"] is True
        layer_group = next(
            operator for operator in graph["operators"]
            if operator["op_kind"] == "layer_group"
        )
        assert layer_group["parameters"]["hidden_size"] > 0


def test_qwen3_presets_keep_official_independent_head_dim():
    for preset_id in (
        "qwen3-0_6b",
        "qwen3-4b",
        "qwen3-32b",
        "qwen3-30b-a3b",
        "qwen3-235b-a22b",
    ):
        definition = get_model_preset(preset_id)
        assert definition.patterns[0].attention_head_dim == 128
        payload = materialize_model_payload(preset_id)
        mixer = next(
            operator
            for operator in payload["graph"]["operators"]
            if operator["op_kind"] in {"attention", "linear_attention"}
        )
        assert mixer["parameters"]["attention_head_dim"] == 128


def test_bundled_presets_follow_official_tied_output_head_configs():
    # These flags are copied from the corresponding official config.json
    # snapshots.  Keep the independent entries explicit too, so a default
    # change cannot silently alter their output-head storage contract.
    expected = {
        "qwen2_5-0_5b": True,
        "qwen2_5-1_5b": True,
        "qwen2_5-3b": True,
        "qwen2_5-7b": False,
        "qwen2_5-14b": False,
        "qwen2_5-32b": False,
        "qwen2_5-72b": False,
        "qwen3-0_6b": True,
        "qwen3-1_7b": True,
        "qwen3-4b": True,
        "qwen3-8b": False,
        "qwen3-14b": False,
        "qwen3-32b": False,
        "qwen3-30b-a3b": False,
        "qwen3-235b-a22b": False,
        "llama3_2-1b": True,
        "llama3_2-3b": True,
        "llama3_1-8b": False,
        "llama3_1-70b": False,
        "llama3_1-405b": False,
        "llama3_3-70b": False,
    }
    for preset_id, tied in expected.items():
        definition = get_model_preset(preset_id)
        assert definition.tie_word_embeddings is tied
        model = model_from_dict(materialize_model_payload(preset_id))
        if tied:
            assert model.output_weight_bytes == 0
            lm_head = next(
                tensor
                for tensor in model.graph.tensors
                if tensor.tensor_id == "lm_head_weights"
            )
            assert lm_head.attributes["storage_id"] == "embedding_weights"
        else:
            assert model.output_weight_bytes == (
                definition.vocabulary_size
                * definition.patterns[0].hidden_size
                * 2
            )


def test_catalog_import_accepts_explicitly_disabled_sliding_window():
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
        "max_position_embeddings": 131072,
        "sliding_window": 4096,
        "use_sliding_window": False,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
    }
    definition = definition_from_huggingface(
        "Qwen/Qwen2.5-7B",
        "main",
        "a" * 40,
        "b" * 64,
        {"id": "Qwen/Qwen2.5-7B", "gated": False, "cardData": {"license": "apache-2.0"}},
        config,
    )
    assert definition.support_level == "exact"
    assert definition.patterns[0].attention_head_dim == 128
    assert definition.tie_word_embeddings is False


def test_import_with_missing_dtype_or_tied_embedding_contract_is_metadata_only():
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 1024,
    }
    definition = definition_from_huggingface(
        "Example/model",
        "main",
        "a" * 40,
        "b" * 64,
        {"id": "Example/model", "gated": False},
        config,
    )

    assert definition.support_level == "out_of_domain"
    assert definition.coverage == "metadata_only"
    assert any("torch_dtype/dtype is absent or auto" in item for item in definition.limitations)
    assert any("tie_word_embeddings must be a boolean" in item for item in definition.limitations)
    with pytest.raises(UnsupportedPresetError):
        materialize_preset_definition(definition)


def test_import_identity_includes_revision_or_commit():
    first = _import_preset_id("Example/model.a", resolved_sha="a" * 40)
    second = _import_preset_id("Example/model-a", resolved_sha="b" * 40)
    assert first != second
