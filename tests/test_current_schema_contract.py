"""Current configuration surfaces do not reinterpret retired inputs."""

from dataclasses import replace
import hashlib
import json

import pytest

from heterollm_sim import schema_v4
from heterollm_sim.architecture_presets import materialize_architecture_payload
from heterollm_sim.component_presets import (
    ComponentPresetCatalog, ComponentPresetMutationError,
    get_component_preset, materialize_component_payload,
)
from heterollm_sim.config import ScenarioConfig, hardware_from_dict, scenario_from_dict
from heterollm_sim.data_motion import resolve_service
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import make_hbm_config
from heterollm_sim.model_catalog import CatalogError, ModelCatalog, _import_preset_id
from heterollm_sim.llama_tensor_storage import _qualification
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.topology import _component_gddr_generation, validate_topology
from heterollm_sim.web import hardware_input_payload, scenario_to_payload


def test_v3_input_and_importer_interface_are_removed():
    with pytest.raises(ValueError, match="must be exactly 4.0.0"):
        scenario_from_dict({"schema_version": "3.0.0"})
    assert not hasattr(schema_v4, "_migrate_v3_scenario_payload")
    assert not hasattr(schema_v4, "main")


def test_reference_roundtrip_uses_component_bindings_only():
    original = build_reference_scenario()
    restored = scenario_from_dict(scenario_to_payload(original))
    for component in original.hardware.components:
        if original.component_profile_kind(component) is not None:
            assert restored.resolve_component_profile(component.component_id) == original.resolve_component_profile(component)
    for kind in ("gpu", "hbm", "gddr", "cpu", "host_memory", "cim"):
        assert not hasattr(ScenarioConfig, kind + "_profile")


def test_native_snapshot_marker_does_not_replace_authored_hardware():
    original = build_reference_scenario()
    original = replace(original, hardware=replace(original.hardware, metadata={
        **original.hardware.metadata,
        "native_hardware_preset_id": "local-rtx5080-9950x3d",
    }))
    restored = scenario_from_dict(scenario_to_payload(original))
    assert restored.hardware.components == original.hardware.components


def test_memory_service_link_requires_explicit_service_reference():
    scenario = build_reference_scenario()
    link = next(item for item in scenario.hardware.links if item.link_id == "gpu-hbm0")
    old_link = replace(link, metadata={"bandwidth_source": "memory_component"})
    with pytest.raises(ValueError, match="service_ref is required"):
        replace(scenario, hardware=replace(scenario.hardware, links=tuple(
            old_link if item.link_id == link.link_id else item
            for item in scenario.hardware.links
        )))
    current_link = replace(old_link, bandwidth_gbps=0., latency_ns=0.,
        metadata={"bandwidth_source": "memory_component", "service_ref": "hbm0.access"})
    resolved = replace(scenario, hardware=replace(scenario.hardware, links=tuple(
        current_link if item.link_id == link.link_id else item
        for item in scenario.hardware.links
    )))
    assert next(item for item in resolved.hardware.links if item.link_id == link.link_id).metadata["service_ref"] == "hbm0.access"


def test_current_native_architecture_declares_memory_service_references():
    hardware = materialize_architecture_payload("local-native-rtx5080-9950x3d-gddr7-ddr5")
    restored = hardware_from_dict(hardware)
    assert restored.component_map()["gddr0"].metadata["physical_memory_config"]["generation"] == "GDDR7"
    links = [item for item in hardware["links"]
             if item["metadata"].get("bandwidth_source") == "memory_component"]
    assert len(links) == 2
    assert {item["metadata"]["service_ref"] for item in links} == {
        "hostmem0.access", "gddr0.access",
    }
    with pytest.raises(KeyError):
        get_component_preset("ocp-hbf-2026-512gb")


def test_service_uses_explicit_profile_and_current_parameter_names():
    physical = make_hbm_config(capacity_bytes=4096)
    memory = ComponentSpec("memory", "hbm", capacity_bytes=physical.capacity_bytes,
        metadata={"physical_memory_config": physical,
                  "profile": {"bandwidth_gb_s": 1.},
                  "memory_profile": {"bandwidth_gb_s": 2.},
                  "memory_service_model": "overlapped",
                  "transfer_granularity_bytes": 4096,
                  "queue_depth": 2})
    service = resolve_service(memory)
    assert service.read_bandwidth_gb_s > 2.
    assert (service.service_model, service.transaction_granularity, service.queue_depth) == (
        "analytical", 256, 32,
    )
    current = resolve_service(memory, {"bandwidth_gb_s": 1., "transaction_bytes": 128,
                                      "max_outstanding_requests": 3, "service_model": "serialized"})
    assert (current.read_bandwidth_gb_s, current.transaction_granularity,
            current.queue_depth, current.service_model) == (1., 128, 3, "serialized")
    with pytest.raises(ValueError, match="transaction_bytes"):
        resolve_service(memory, {"transaction_bytes": 0})


def test_component_mutation_roundtrip_uses_id_and_component(tmp_path):
    catalog = ComponentPresetCatalog(tmp_path)
    component = materialize_component_payload("nvidia-rtx-5080")
    created = catalog.create({"id": "current-test-gpu", "component": component})
    restored = ComponentPresetCatalog(tmp_path).detail("current-test-gpu")
    assert restored["component"] == created["component"]
    updated = catalog.update("current-test-gpu", {"name": "Updated GPU"})
    assert updated["preset"]["name"] == "Updated GPU"


@pytest.mark.parametrize("old_field", ["preset_id", "component_spec", "preset"])
def test_component_mutation_rejects_retired_shapes(tmp_path, old_field):
    current = {"id": "current-test-gpu", "component": materialize_component_payload("nvidia-rtx-5080")}
    if old_field == "preset":
        payload = {"preset": current}
    else:
        payload = dict(current)
        canonical = "id" if old_field == "preset_id" else "component"
        payload[old_field] = payload.pop(canonical)
    with pytest.raises(ComponentPresetMutationError, match="未知字段"):
        ComponentPresetCatalog(tmp_path).create(payload)


@pytest.mark.parametrize("field", ["capacity_bytes", "read_bandwidth_gbps", "write_bandwidth_gbps", "bandwidth_gbps"])
def test_component_mutation_rejects_gpu_memory_fields_instead_of_erasing_them(tmp_path, field):
    component = materialize_component_payload("nvidia-rtx-5080")
    component[field] = 1
    with pytest.raises(ComponentPresetMutationError, match=field + " 必须为 0"):
        ComponentPresetCatalog(tmp_path).create({"id": "current-test-gpu", "component": component})


def test_imported_model_identity_requires_a_resolved_commit(tmp_path):
    catalog = ModelCatalog(tmp_path)
    commit = "a" * 40
    other_commit = "b" * 40
    assert catalog._record_path("example/model", resolved_sha=commit) != catalog._record_path("example/model", resolved_sha=other_commit)
    assert _import_preset_id("example/model", resolved_sha=commit) != _import_preset_id("example/model", resolved_sha=other_commit)
    with pytest.raises(TypeError):
        catalog._record_path("example/model")
    with pytest.raises(CatalogError, match="resolved repository commit"):
        _import_preset_id("example/model", resolved_sha="")


class _CurrentRepositoryClient:
    def repository_config(self, repo_id, revision):
        commit = "a" * 40
        return ({"id": repo_id, "sha": commit}, {
            "model_type": "llama", "num_hidden_layers": 1, "hidden_size": 64,
            "intermediate_size": 128, "num_attention_heads": 2,
            "vocab_size": 128, "max_position_embeddings": 2048,
            "torch_dtype": "float16", "tie_word_embeddings": False,
        }, commit, "current-test-config")


@pytest.mark.parametrize("retired_shape", ["repo_only_path", "old_identity", "no_commit", "old_version"])
def test_model_catalog_reload_rejects_retired_cache_records(tmp_path, retired_shape):
    catalog = ModelCatalog(tmp_path, hf_client=_CurrentRepositoryClient())
    repo_id = "example/current-model"
    current_id = catalog.import_repository(repo_id)["preset"]["id"]
    current_path = catalog._record_path(repo_id, resolved_sha="a" * 40)
    assert ModelCatalog(tmp_path).detail(current_id)["preset"]["id"] == current_id
    if retired_shape == "repo_only_path":
        current_path.rename(tmp_path / (hashlib.sha256(repo_id.encode()).hexdigest()[:32] + ".json"))
    else:
        record = json.loads(current_path.read_text(encoding="utf-8"))
        if retired_shape == "old_identity":
            record["definition"]["preset_id"] = "hf-example-current-model"
        elif retired_shape == "no_commit":
            record.pop("resolved_sha")
        else:
            record["cache_version"] = 0
        current_path.write_text(json.dumps(record), encoding="utf-8")
    reloaded = ModelCatalog(tmp_path)
    assert all(item["source"] != "huggingface_import" for item in reloaded.list_metadata())


def test_model_catalog_reload_preserves_current_repository_deduplication(tmp_path):
    catalog = ModelCatalog(tmp_path, hf_client=_CurrentRepositoryClient())
    original_id = catalog.import_repository("example/current-model")["preset"]["id"]
    duplicate = catalog.import_repository("example/current-copy")
    assert duplicate["provenance"]["deduplicated_as"] == original_id
    assert "example/current-copy" in ModelCatalog(tmp_path).detail(original_id)["preset"]["variants"]


@pytest.mark.parametrize("retired_field", ["schema_version", "parameters", "execution_profile", "cost_profile_id"])
def test_hardware_input_rejects_retired_envelopes_and_duplicate_profile_sources(retired_field):
    scenario = build_reference_scenario()
    hardware_input = hardware_input_payload(scenario)
    assert hardware_input["schema_version"] == "4.0.0"
    if retired_field == "schema_version":
        hardware_input["schema_version"] = "4.0"
    elif retired_field == "parameters":
        hardware_input["hardware"]["parameters"]["components"] = {}
    else:
        component = next(item for item in hardware_input["hardware"]["components"]
                         if "execution_profile" in item)
        if retired_field == "execution_profile":
            component["execution_profile"]["profile"] = {}
        else:
            component["cost_profile_id"] = "old-profile"
    with pytest.raises(ValueError, match="schema_version|unknown fields|cost_profile_id"):
        scenario_from_dict({**scenario_to_payload(scenario), "hardware_input": hardware_input})


def test_model_layer_group_requires_layer_ids_in_parameters():
    scenario = build_reference_scenario()
    group = next(item for item in scenario.model.graph.operators if item.op_kind == "layer_group")
    parameters = dict(group.parameters)
    parameters.pop("layer_ids")
    stale_group = replace(group, parameters=parameters)
    graph = replace(scenario.model.graph, operators=tuple(
        stale_group if item.operator_id == group.operator_id else item
        for item in scenario.model.graph.operators))
    with pytest.raises(ValueError, match="layer_ids"):
        replace(scenario.model, graph=graph)


@pytest.mark.parametrize("component_kind,error", [("switch", "does not use execution_profile"),
                                                  ("hbf", "kind does not match")])
def test_hardware_input_does_not_discard_irrelevant_execution_profiles(component_kind, error):
    scenario = build_reference_scenario()
    hardware_input = hardware_input_payload(scenario)
    component = next(item for item in hardware_input["hardware"]["components"] if item["kind"] == "gpu")
    component["kind"] = component_kind
    component["metadata"]["access_mode"] = "memory"
    hardware_input["hardware"]["parameters"]["runtime"]["gpu_controllers"] = {}
    with pytest.raises(ValueError, match=error):
        scenario_from_dict({**scenario_to_payload(scenario), "hardware_input": hardware_input})


@pytest.mark.parametrize("nested", [False, True])
def test_gguf_audit_bindings_require_current_nested_metadata(nested):
    scenario = build_reference_scenario()
    evidence = {"gguf_sha256": "a" * 64,
                "gguf_embedding_binding": {"shape": [1, 1], "n_bytes": 1, "type": "F32"}}
    attributes = {**scenario.model.graph.attributes}
    if nested:
        attributes["metadata"] = evidence
        model = replace(scenario.model, graph=replace(scenario.model.graph, attributes=attributes), metadata=attributes)
    else:
        model = replace(scenario.model, metadata=evidence)
    audit = _qualification(replace(scenario, model=model), {})
    assert ("typed_gguf_identity_required" in audit["reasons"]) is not nested
    assert ("gguf_embedding_binding_missing" in audit["reasons"]) is not nested


@pytest.mark.parametrize("preset_id,generation", [
    ("gddr6-16gb-20_0-256bit", "GDDR6"),
    ("gddr6x-16gb-21_0-256bit", "GDDR6X"),
    ("gddr7-16gb-30_0-256bit", "GDDR7"),
])
def test_gddr_presets_roundtrip_with_one_physical_generation(preset_id, generation):
    component = materialize_component_payload(preset_id)
    assert "generation" not in component
    assert "generation" not in component["metadata"]
    assert component["metadata"]["physical_memory_config"]["generation"] == generation
    hardware = hardware_from_dict({"name": "gddr", "components": [component]})
    assert _component_gddr_generation(hardware.components[0]) == generation


def test_gddr_metadata_cannot_supply_or_override_the_physical_generation():
    component = get_component_preset("gddr7-16gb-30_0-256bit").component
    stale_metadata = {**component.metadata, "generation": "GDDR6",
                      "memory_generation": "GDDR6X", "technology": {"generation": "GDDR6"}}
    current = replace(component, metadata=stale_metadata)
    assert _component_gddr_generation(current) == "GDDR7"
    physical = dict(component.metadata["physical_memory_config"])
    physical.pop("generation")
    with pytest.raises(ValueError, match="physical_memory_config.generation"):
        replace(component, metadata={**stale_metadata, "physical_memory_config": physical})
    with pytest.raises(TypeError, match="generation"):
        ComponentSpec("memory", "gddr", generation="GDDR7", metadata=component.metadata)


@pytest.mark.parametrize("kind", ["GDDR6", "GDDR6X", "GDDR7"])
def test_gddr_generation_is_not_a_component_kind(kind):
    component = get_component_preset("gddr7-16gb-30_0-256bit").component
    with pytest.raises(ValueError, match="GDDR component kind must be gddr"):
        replace(component, kind=kind)


@pytest.mark.parametrize("field", ["supported_gddr_generations", "memory_generation",
                                   "supported_generations", "gddr_generations", "memory_generations"])
def test_gddr_controller_capability_requires_the_current_metadata_field(field):
    hardware = hardware_from_dict(materialize_architecture_payload("local-native-rtx5080-9950x3d-gddr7-ddr5"))
    gpu = hardware.component_map()["gpu0"]
    gpu = replace(gpu, metadata={field: ["GDDR7"]}, ports=tuple(
        replace(port, version="1.0", metadata={}) if port.protocol.upper() == "GDDR" else port
        for port in gpu.ports))
    hardware = replace(hardware, components=tuple(
        gpu if component.component_id == gpu.component_id else component
        for component in hardware.components))
    codes = validate_topology(hardware).error_codes
    assert ("gddr_controller_capability_missing" in codes) is (field != "supported_gddr_generations")


def test_obsolete_standalone_mmvq_installer_is_removed():
    from heterollm_sim import standalone_mmvq_calibration

    assert not hasattr(standalone_mmvq_calibration, "install_standalone_mmvq_surfaces")
