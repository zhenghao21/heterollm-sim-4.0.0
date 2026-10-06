from heterollm_sim.model_presets import (
    get_model_preset,
    materialize_model_payload,
    list_model_presets,
)
from heterollm_sim.model_catalog import definition_from_huggingface, _import_preset_id


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


def test_import_identity_includes_revision_or_commit():
    first = _import_preset_id("Example/model.a", resolved_sha="a" * 40)
    second = _import_preset_id("Example/model-a", resolved_sha="b" * 40)
    assert first != second
