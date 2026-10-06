from heterollm_sim.model_presets import (
    materialize_model_payload,
    list_model_presets,
)


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
