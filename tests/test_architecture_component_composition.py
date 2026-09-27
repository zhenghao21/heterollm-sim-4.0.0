from heterollm_sim.architecture_presets import materialize_architecture_payload
from heterollm_sim.component_presets import ComponentPresetCatalog
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.topology import validate_topology


def test_curated_b200_architecture_references_curated_component_presets():
    payload = materialize_architecture_payload("nvidia-b200-1gpu-2hbf-2hbm")
    components = {item["component_id"]: item for item in payload["components"]}
    assert components["gpu0"]["metadata"]["component_preset_id"] == "nvidia-b200-sxm-gpu"
    assert components["hbf0"]["metadata"]["component_preset_id"] == "sk-hynix-hbf-512gb"
    assert components["hbf1"]["metadata"]["component_preset_id"] == "sk-hynix-hbf-512gb"
    assert components["hbm0"]["metadata"]["component_preset_id"] == "samsung-hbm3e-36gb-9_2"
    assert components["hbm1"]["metadata"]["component_preset_id"] == "samsung-hbm3e-36gb-9_2"


def test_hbf_write_capability_is_explicitly_unknown_and_not_free():
    payload = materialize_architecture_payload("nvidia-b200-1gpu-2hbf-2hbm")
    for component in payload["components"]:
        if component["kind"] != "hbf":
            continue
        assert component["write_bandwidth_gbps"] == 0.0
        assert component["metadata"]["write_capability_status"] == "unknown"
        assert component["metadata"]["capability_status"]["write_bandwidth_gbps"] == "not_published"


def test_local_component_override_is_applied_when_architecture_is_materialized(tmp_path):
    catalog = ComponentPresetCatalog(tmp_path)
    preset = catalog.detail("samsung-hbm3e-36gb-9_2")
    preset["component"]["capacity_bytes"] = 48_000_000_000
    preset["component"]["read_bandwidth_gbps"] = 8_000.0
    preset["component"]["write_bandwidth_gbps"] = 7_200.0
    catalog.update("samsung-hbm3e-36gb-9_2", preset)
    payload = materialize_architecture_payload("nvidia-b200-1gpu-2hbf-2hbm", component_catalog=catalog)
    nodes = {node["component_id"]: node for node in payload["components"]}
    report = validate_topology(hardware_from_dict(payload))
    assert report.is_valid, report.format_en()
    assert nodes["hbm0"]["capacity_bytes"] == 48_000_000_000
    assert nodes["hbm1"]["write_bandwidth_gbps"] == 7_200.0
    assert nodes["hbm0"]["metadata"]["cost_profile_template"]["read_bandwidth_gb_s"] == 1_000.0
    # Explicit local edits do not mutate the offline bundled catalog.
    bundled = materialize_architecture_payload("nvidia-b200-1gpu-2hbf-2hbm")
    assert next(node for node in bundled["components"] if node["component_id"] == "hbm0")["capacity_bytes"] == 36_000_000_000
