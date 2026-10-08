from dataclasses import replace

import pytest

from heterollm_sim import architecture_presets
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v4 import runtime_profile_from_dict
from heterollm_sim.serde import to_primitive


@pytest.mark.parametrize(
    "preset_id", sorted(architecture_presets.PUBLIC_ARCHITECTURE_PRESET_IDS)
)
def test_loadable_architecture_supplies_existing_profiles_for_its_actual_gpus(preset_id):
    detail = architecture_presets.architecture_preset_detail(preset_id)
    gpu_ids = {
        component["component_id"]
        for component in detail["hardware"]["components"]
        if component["kind"] == "gpu"
    }
    assert detail["preset"]["loadable"]
    assert gpu_ids
    runtime = detail["profiles"]["runtime"]
    assert set(runtime["gpu_controllers"]) == gpu_ids
    assert to_primitive(runtime_profile_from_dict(runtime)) == runtime

    reference = build_reference_scenario()
    expected = to_primitive(reference.runtime_profile)
    gpu_template = expected["gpu_controllers"]["gpu0"]
    expected["gpu_controllers"] = {gpu_id: gpu_template for gpu_id in gpu_ids}
    assert runtime == expected
    assert detail["profiles"]["fusion"] == to_primitive(reference.fusion_policy)
    basis = detail["profiles_parameter_basis"]
    assert basis["kind"] == "existing_analytical_template"
    assert basis["is_native_measurement"] is False
    assert basis["is_manufacturer_specification"] is False


def test_gpu_free_architecture_does_not_invent_a_gpu_controller(monkeypatch):
    preset_id = "local-native-rtx5080-9950x3d-gddr7-ddr5"
    original = architecture_presets.get_architecture_preset(preset_id)
    gpu_free = replace(original, hardware=replace(
        original.hardware,
        components=tuple(component for component in original.hardware.components if component.kind != "gpu"),
        links=(),
    ))
    monkeypatch.setattr(architecture_presets, "get_architecture_preset", lambda _: gpu_free)

    detail = architecture_presets.architecture_preset_detail(preset_id)

    assert "runtime" not in detail["profiles"]
    assert "fusion" in detail["profiles"]


def test_unloadable_architecture_does_not_supply_runtime_profiles(monkeypatch):
    preset_id = "local-native-rtx5080-9950x3d-gddr7-ddr5"
    original = architecture_presets.get_architecture_preset(preset_id)
    monkeypatch.setattr(
        architecture_presets, "get_architecture_preset",
        lambda _: replace(original, loadable=False),
    )

    detail = architecture_presets.architecture_preset_detail(preset_id)

    assert detail["hardware"] is None
    assert detail["profiles"] == {}
    assert detail["profiles_parameter_basis"] == {}
