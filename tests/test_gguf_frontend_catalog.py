import http.client
import json
import threading
from copy import deepcopy
from dataclasses import replace

import pytest

from heterollm_sim.config import model_from_dict
from heterollm_sim.gguf_model_catalog import GGUFModelCatalog, inventory_to_gguf
from heterollm_sim.gguf_parity import build_model_from_gguf
from heterollm_sim.ir import model_graph_execution_view
from heterollm_sim.llama_tensor_storage import qualify_llama_tensor_storage_contract, SOURCE_KEY
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.serde import to_primitive
from heterollm_sim.web import build_server


def test_frontend_preserves_all_entries_and_gates_unadapted_architectures(tmp_path):
    catalog = GGUFModelCatalog(tmp_path)
    items = catalog.list_metadata()
    assert len(items) == 22
    assert sum(row["source_status"] == "gguf_ready" for row in items) == 20
    assert {row["id"] for row in items if row["source_status"] == "gguf_unsupported"} == {
        "qwen3-30b-a3b", "qwen3-235b-a22b",
    }
    for row in items:
        detail = catalog.detail(row["id"])
        if row["source_status"] == "gguf_unsupported":
            assert row["generation_allowed"] is False
            assert detail["model"] is None
            with pytest.raises(ValueError, match="架构待适配"):
                catalog.configuration(row["id"])
            with pytest.raises(ValueError, match="架构待适配"):
                catalog.derive(row["id"], "cannot enable by edit", {"layer_count": 2}, save=True)
        else:
            assert row["generation_allowed"] is True


def test_all_ready_presets_share_direct_gguf_graph_and_runtime_contract(tmp_path):
    catalog = GGUFModelCatalog(tmp_path)
    for preset_id, record in catalog._records().items():
        if not catalog.detail(preset_id)["preset"]["generation_allowed"]:
            with pytest.raises(ValueError, match="unsupported GGUF architecture"):
                build_model_from_gguf(inventory_to_gguf(record["inventory"]))
            continue
        direct = to_primitive(build_model_from_gguf(inventory_to_gguf(record["inventory"])))
        loaded = catalog.detail(preset_id)["model"]
        assert loaded["graph"] == direct["graph"]
        assert loaded["metadata"]["llama_cpp_f32_hidden_storage"] is True
        for key, value in direct["metadata"].items():
            assert loaded["metadata"][key] == value


def test_missing_source_keeps_entry_visible_but_cannot_materialize(tmp_path, monkeypatch):
    catalog = GGUFModelCatalog(tmp_path)
    records = catalog._records()
    del records["qwen2_5-0_5b"]
    monkeypatch.setattr(catalog, "_records", lambda: records)
    detail = catalog.detail("qwen2_5-0_5b")
    assert detail["preset"]["source_status"] == "pending_gguf"
    assert detail["preset"]["generation_allowed"] is False
    assert detail["model"] is None
    with pytest.raises(ValueError, match="先绑定 GGUF"):
        catalog.configuration("qwen2_5-0_5b")
    with pytest.raises(ValueError, match="已绑定 GGUF"):
        catalog.derive("qwen2_5-0_5b", "cannot use config estimates", {}, save=True)


def test_remote_origins_include_complete_file_identity_revision_and_variant(tmp_path):
    catalog = GGUFModelCatalog(tmp_path)
    remote = [record for record in catalog._records().values() if record["origin"].get("repo")]
    assert len(remote) == 17
    for record in remote:
        origin, raw = record["origin"], record["inventory"]
        assert len(origin["revision"]) == 40
        assert origin["variant"]
        assert len(origin["files"]) == len(raw["sources"])
        assert {row["filename"] for row in origin["files"]} == {row["rfilename"] for row in raw["sources"]}
        assert all(row["size_bytes"] > 0 and len(row["sha256"]) == 64 for row in origin["files"])
        if "Instruct" in origin["variant"]:
            assert "Instruct" in record["name"]
        if len(origin["files"]) > 1:
            assert not raw["sha256"]  # do not invent a monolithic file identity
    llama = catalog.configuration("llama3_1-405b")
    assert llama["parameters"]["kv_heads"] == 16
    assert llama["origin"]["variant"] == "Instruct (16 KV heads)"


def test_missing_split_source_is_disabled_in_list_detail_and_editor(tmp_path, monkeypatch):
    catalog = GGUFModelCatalog(tmp_path)
    records = deepcopy(catalog._records())
    raw = records["llama3_1-405b"]["inventory"]
    raw["sources"] = raw["sources"][:-1]
    monkeypatch.setattr(catalog, "_records", lambda: records)
    item = next(row for row in catalog.list_metadata() if row["id"] == "llama3_1-405b")
    assert item["generation_allowed"] is False
    assert item["source_status"] == "gguf_incomplete"
    assert catalog.detail(item["id"])["model"] is None
    with pytest.raises(ValueError, match="split inventory"):
        catalog.configuration(item["id"])
    with pytest.raises(ValueError, match="split inventory"):
        catalog.derive(item["id"], "invalid source", {}, save=True)


def test_split_import_and_derivative_preserve_source_identity_and_persistence(tmp_path, monkeypatch):
    import heterollm_sim.gguf_model_catalog as module
    catalog = GGUFModelCatalog(tmp_path)
    gguf = inventory_to_gguf(catalog._records()["llama3_1-405b"]["inventory"])
    monkeypatch.setattr(module, "read_gguf_metadata", lambda _: gguf)
    imported = catalog.import_gguf("405b-00001-of-00006.gguf", name="local split source")
    derived = catalog.derive(imported["preset"]["id"], "split derivative", {"layer_count": 2}, save=True)
    assert len(derived["preset"]["gguf_source"]["files"]) == 6
    for detail in (imported, derived):
        base = build_llama_default_scenario()
        model = model_from_dict(detail["model"])
        scenario = replace(base, model=model,
            workload=replace(base.workload, metadata={SOURCE_KEY: model.metadata[SOURCE_KEY]}))
        audit = qualify_llama_tensor_storage_contract(scenario)
        assert audit["qualified"], audit
    assert GGUFModelCatalog(tmp_path).detail(derived["preset"]["id"]) == derived
    inventory = catalog._records()[derived["preset"]["id"]]["inventory"]
    assert not inventory["sources"]
    assert not any(key.startswith("split.") for key in inventory["metadata"])


@pytest.mark.parametrize("preset_id", [
    "qwen3-0_6b", "qwen3-4b", "qwen3_8-27b-iq3-s-iq4-xs",
    "qwen2_5-0_5b", "llama3_2-1b", "llama3_1-405b",
])
def test_derived_dimensions_rebuild_physical_weights_and_do_not_mutate_original(tmp_path, preset_id):
    catalog = GGUFModelCatalog(tmp_path)
    original = catalog.detail(preset_id)
    changed = catalog.derive(preset_id, "custom dimensions", {
        "layer_count": 4, "hidden_size": 2048, "intermediate_size": 4096,
        "attention_heads": 16, "kv_heads": 4, "vocabulary_size": 64000,
        "max_sequence_length": 8192, "weight_quantization": "Q4_K",
        **({"head_dim": 128} if preset_id.startswith("llama") else {}),
    }, save=True)
    model = model_from_dict(changed["model"])
    view = model_graph_execution_view(model.graph, schema_version=model.schema_version)
    assert len(view.layer_instances) == 4
    for item in view.layer_instances:
        assert item.layer.hidden_size == 2048
        assert item.layer.intermediate_size == 4096
    assert model.vocabulary_size == 64000
    assert model.max_sequence_length == 8192
    assert model.embedding_weight_bytes == 2048 * 64000 // 256 * 144
    assert changed["preset"]["source_status"] == "gguf_derived"
    assert not model.graph.attributes["metadata"]["gguf_sha256"]
    assert model.metadata["gguf_preset_origin"] == original["preset"]["gguf_source"]
    assert catalog.detail(preset_id) == original
    assert GGUFModelCatalog(tmp_path).detail(changed["preset"]["id"]) == changed
    if preset_id.startswith("llama"):
        inventory = catalog._records()[changed["preset"]["id"]]["inventory"]
        rope = next(tensor for tensor in inventory["tensor_directory"] if tensor["name"] == "rope_freqs.weight")
        assert tuple(rope["shape"]) == (64,)
        assert rope["n_bytes"] == 256
        assert inventory["metadata"]["llama.rope.dimension_count"] == 128


@pytest.mark.parametrize("parameters", [
    {"hidden_size": 1025, "weight_quantization": "Q4_K"},
    {"attention_heads": 7, "kv_heads": 4}, {"layer_count": 0},
    {"layer_count": True}, {"head_dim": "128"}, {"unknown": 1},
])
def test_invalid_edit_does_not_create_preset(tmp_path, parameters):
    catalog = GGUFModelCatalog(tmp_path)
    with pytest.raises(ValueError):
        catalog.derive("qwen3-0_6b", "invalid", parameters, save=True)
    assert not list(tmp_path.glob("*.gguf-preset.json"))


def test_preview_is_not_persisted_and_derived_get_rows_still_qualifies(tmp_path):
    catalog = GGUFModelCatalog(tmp_path)
    detail = catalog.derive("qwen3-0_6b", "preview", {"layer_count": 2})
    assert not list(tmp_path.glob("*.gguf-preset.json"))
    base = build_llama_default_scenario()
    contract = detail["model"]["metadata"][SOURCE_KEY]
    scenario = replace(base, model=model_from_dict(detail["model"]),
        workload=replace(base.workload, metadata={SOURCE_KEY: contract}))
    assert qualify_llama_tensor_storage_contract(scenario)["qualified"] is True


def test_http_editor_save_restart_and_unsupported_gate(tmp_path):
    server = build_server("127.0.0.1", 0, catalog_cache_dir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(method, path, payload=None):
        conn = http.client.HTTPConnection(*server.server_address, timeout=15)
        try:
            conn.request(method, path, body=json.dumps(payload) if payload is not None else None,
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()
    try:
        status, pending = request("GET", "/api/model-presets/qwen3-30b-a3b")
        assert status == 200 and pending["model"] is None
        assert request("GET", "/api/model-presets/qwen3-30b-a3b/configuration")[0] == 422
        assert request("POST", "/api/model-files/import", {"preset_id": "qwen3-30b-a3b"})[0] == 422
        status, config = request("GET", "/api/model-presets/qwen3-0_6b/configuration")
        assert status == 200
        payload = {"source_preset_id": config["preset_id"], "name": "HTTP custom", "parameters": {"layer_count": 2}}
        assert request("POST", "/api/model-presets/preview", payload)[0] == 200
        assert not list(tmp_path.glob("*.gguf-preset.json"))
        status, saved = request("POST", "/api/model-presets", payload)
        assert status == 201
        assert request("GET", "/api/model-presets/" + saved["preset"]["id"])[1] == saved
        assert GGUFModelCatalog(tmp_path).detail(saved["preset"]["id"]) == saved
        assert request("GET", "/api/model-presets/qwen3-0_6b/configuration")[1]["parameters"]["layer_count"] == 28
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_binding_rejects_same_layer_count_with_wrong_hidden_width(tmp_path, monkeypatch):
    import heterollm_sim.gguf_model_catalog as module
    catalog = GGUFModelCatalog(tmp_path)
    records = catalog._records()
    # Both files have 28 layers and the same vocabulary; geometry must still match.
    other = inventory_to_gguf(records["qwen3-1_7b"]["inventory"])
    records.pop("qwen3-0_6b")
    monkeypatch.setattr(catalog, "_records", lambda: records)
    monkeypatch.setattr(module, "read_gguf_metadata", lambda _: other)
    with pytest.raises(ValueError, match="hidden_size"):
        catalog.import_gguf("source.gguf", preset_id="qwen3-0_6b")
    assert not list(tmp_path.glob("*.gguf-preset.json"))


def test_import_and_save_failure_do_not_replace_source_or_publish_partial_files(tmp_path, monkeypatch):
    import heterollm_sim.gguf_model_catalog as module
    catalog = GGUFModelCatalog(tmp_path)
    original = catalog.detail("qwen3-0_6b")
    source = inventory_to_gguf(catalog._records()["qwen3-0_6b"]["inventory"])
    monkeypatch.setattr(module, "read_gguf_metadata", lambda _: source)
    imported = catalog.import_gguf("local-f16.gguf", name="Imported F16")
    assert GGUFModelCatalog(tmp_path).detail(imported["preset"]["id"]) == imported
    assert imported["model"]["graph"] == original["model"]["graph"]
    before_files = set(tmp_path.iterdir())
    def fail_link(*args):
        raise OSError("disk publication failed")
    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="disk publication"):
        catalog.derive("qwen3-0_6b", "failed save", {"layer_count": 4}, save=True)
    assert set(tmp_path.iterdir()) == before_files
    assert catalog.detail("qwen3-0_6b") == original
