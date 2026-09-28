from dataclasses import replace
import pytest
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.web import scenario_to_payload


def test_memory_service_resolves_effective_and_physical_caps():
    scenario = build_reference_scenario()
    service = scenario.hardware.get_component("hbm0").metadata["memory_service"]
    assert service["service_id"] == "hbm0.access"
    assert service["bandwidth_gb_s"] <= service["physical_bandwidth_gb_s"]
    assert service["read_latency_ns"] == 0


def test_explicit_service_ref_rejects_duplicate_link_latency():
    scenario = build_reference_scenario()
    link = next(item for item in scenario.hardware.links if item.link_id == "gpu-hbm0")
    bad = replace(link, metadata={**link.metadata, "service_ref": "hbm0.access"}, latency_ns=1.0)
    hardware = replace(scenario.hardware, links=tuple(bad if item is link else item for item in scenario.hardware.links))
    with pytest.raises(ValueError, match=r"hardware.links\[gpu-hbm0\].latency_ns"):
        replace(scenario, hardware=hardware)


def test_service_ref_survives_payload_roundtrip():
    scenario = build_reference_scenario()
    link = next(item for item in scenario.hardware.links if item.link_id == "gpu-hbm0")
    hardware = replace(scenario.hardware, links=tuple(replace(item, bandwidth_gbps=4096.0, latency_ns=0.0, metadata={**item.metadata, "service_ref": "hbm0.access"}) if item is link else item for item in scenario.hardware.links))
    parsed = scenario_from_dict(scenario_to_payload(replace(scenario, hardware=hardware)))
    assert parsed.hardware.get_component("hbm0").metadata["memory_service"]["service_id"] == "hbm0.access"



def test_replacing_profile_rebuilds_service_without_stale_physical_defaults():
    scenario = build_reference_scenario()
    registries = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    registries["hbm"]["legacy-hbm"] = replace(
        registries["hbm"]["legacy-hbm"], read_latency_ns=123, resource_id="new.hbm.owner"
    )
    changed = replace(scenario, component_profiles=registries)
    memory = changed.hardware.get_component("hbm0")
    assert changed.resolve_component_profile(memory).read_latency_ns == 123
    assert memory.metadata["memory_service"]["read_latency_ns"] == 123
    assert memory.metadata["memory_service"]["physical_owner"] == "new.hbm.owner"
    assert "read_latency_ns" not in memory.metadata


def test_import_conflicting_physical_latency_reports_both_paths():
    payload = scenario_to_payload(build_reference_scenario())
    payload.pop("hardware_input")
    memory = next(item for item in payload["hardware"]["components"] if item["component_id"] == "hbm0")
    memory["metadata"]["read_latency_ns"] = 123
    with pytest.raises(ValueError, match=r"hardware.components\[hbm0\].metadata.read_latency_ns=123 conflicts with profiles.components.hbm.legacy-hbm.read_latency_ns=0"):
        scenario_from_dict(payload)
