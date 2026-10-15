"""Catalog and bound-service payloads expose one current field per value."""
from dataclasses import replace

from heterollm_sim import architecture_presets
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.data_motion import resolve_service
from heterollm_sim.web import scenario_to_payload
from test_hbm_config_contract import scenario_with_asymmetric_hbm


def test_effective_service_rates_survive_current_scenario_round_trip():
    original = scenario_with_asymmetric_hbm(profile_read=96., profile_write=32.)
    restored = scenario_from_dict(scenario_to_payload(original))
    for scenario in (original, restored):
        memory = scenario.hardware.get_component("hbm0")
        service = memory.metadata["memory_service"]
        assert (service["bandwidth_gb_s"], service["read_bandwidth_gb_s"],
                service["write_bandwidth_gb_s"]) == (128., 96., 32.)
        assert (service["physical_bandwidth_gb_s"], service["physical_read_bandwidth_gb_s"],
                service["physical_write_bandwidth_gb_s"]) == (512., 128., 64.)
        assert not any(key.startswith("effective_") for key in service)
        resolved = resolve_service(memory)
        assert (resolved.read_bandwidth_gb_s, resolved.write_bandwidth_gb_s) == (96., 32.)


def test_architecture_catalog_detail_has_one_hardware_and_support_contract():
    page = architecture_presets.architecture_preset_page()
    for row in page["items"]:
        detail = architecture_presets.architecture_preset_detail(row["id"])
        assert detail["preset"] == row
        assert "evidence_level" not in row
        assert row["support_level"] in page["filters"]["support_level"]
        assert row["topology_evidence"]
        assert not {"components", "links", "groups", "topology_view"}.intersection(detail)
        hardware = detail["hardware"]
        assert hardware["components"] and hardware["links"]
        view = hardware["metadata"]["topology_view"]
        assert view["groups"] and view["layout"]["positions"]


def test_nonloadable_architecture_keeps_its_display_metadata(monkeypatch):
    preset_id = next(iter(architecture_presets.PUBLIC_ARCHITECTURE_PRESET_IDS))
    current = architecture_presets.get_architecture_preset(preset_id)
    monkeypatch.setitem(architecture_presets._BY_ID, preset_id,
                        replace(current, loadable=False))
    detail = architecture_presets.architecture_preset_detail(preset_id)
    assert detail["hardware"] is None
    assert detail["profiles"] == {}
    assert detail["preset"]["loadable"] is False
    assert detail["preset"]["support_level"] == current.support_level
    assert detail["preset"]["sources"] and detail["preset"]["topology_evidence"]
    assert not {"components", "links", "groups", "topology_view"}.intersection(detail)
