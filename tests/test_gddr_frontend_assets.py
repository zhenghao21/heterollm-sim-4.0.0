from pathlib import Path

from heterollm_sim.component_presets import get_component_preset, list_component_presets
from heterollm_sim.protocol_presets import list_protocol_presets
from heterollm_sim.architecture_presets import get_architecture_preset


def test_gddr_component_presets_are_first_class_and_keep_generation():
    ids = {item["id"] for item in list_component_presets()}
    expected = {
        "gddr6-16gb-20_0-256bit",
        "gddr6x-16gb-21_0-256bit",
        "gddr7-16gb-30_0-256bit",
    }
    assert expected <= ids
    for preset_id in expected:
        component = get_component_preset(preset_id).component
        assert component.kind == "gddr"
        config = component.metadata["physical_memory_config"]
        assert config["kind"] == "GDDR"
        assert config["generation"] in {"GDDR6", "GDDR6X", "GDDR7"}
        assert component.metadata["cost_profile_key"] == "gddr"


def test_gddr_protocol_presets_use_family_plus_generation():
    presets = [item for item in list_protocol_presets() if item["protocol"] == "GDDR"]
    assert {item["version"] for item in presets} == {"GDDR6", "GDDR6X", "GDDR7"}
    assert {item["bandwidth"]["effective_one_way_gbps"] for item in presets} == {5120.0, 5376.0, 7680.0}


def test_webui_exposes_gddr_palette_protocol_and_validation():
    root = Path(__file__).parents[1] / "src" / "heterollm_sim" / "webui"
    index = (root / "index.html").read_text(encoding="utf-8")
    app = (root / "app.js").read_text(encoding="utf-8")
    assert 'data-add-kind="gddr"' in index
    assert '<option value="GDDR">GDDR</option>' in index
    assert '"GDDR6", "GDDR6X", "GDDR7"' in app
    assert 'physical_memory_config.kind 必须是 DDR、LPDDR、HBM、GDDR、SSD 或 HBF。' in app


def test_native_rtx_architecture_uses_formal_gddr_endpoint_and_link():
    preset = get_architecture_preset("local-native-rtx5080-9950x3d-gddr7-ddr5")
    memory = next(item for item in preset.hardware.components if item.component_id == "gddr0")
    link = next(item for item in preset.hardware.links if item.link_id == "gpu-gddr7")
    assert memory.kind == "gddr"
    assert memory.metadata["memory_type"] == "GDDR7"
    assert link.protocol == "GDDR"
    assert link.version == "GDDR7"
    assert link.target_component == "gddr0"
