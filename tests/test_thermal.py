from dataclasses import replace

import pytest

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.thermal import ThermalOperatingPoint, apply_thermal_operating_point


@pytest.mark.parametrize("measured", (None, 100.0))
def test_thermal_derating_reaches_shared_profile_and_route_bandwidth(measured):
    baseline = build_reference_scenario()
    memory = baseline.hardware.get_component("hbm0")
    memory = replace(memory, metadata={**memory.metadata, "thermal_domain_id": "package"})
    link = next(link for link in baseline.hardware.links if link.link_id == "gpu-hbm0")
    link = replace(link, metadata={**link.metadata, "thermal_domain_id": "package"})
    profiles = {kind: dict(registry) for kind, registry in baseline.component_profiles.items()}
    profiles["hbm"][memory.cost_profile_id] = replace(
        profiles["hbm"][memory.cost_profile_id], measured_effective_bandwidth_gb_s=measured,
    )
    baseline = replace(baseline, component_profiles=profiles, hardware=replace(
        baseline.hardware,
        components=tuple(memory if c.component_id == memory.component_id else c for c in baseline.hardware.components),
        links=tuple(link if candidate.link_id == link.link_id else candidate for candidate in baseline.hardware.links),
    ))
    point = ThermalOperatingPoint(
        "package", enabled=True, memory_bandwidth_scale=0.5,
        link_bandwidth_scale=0.5, evidence="Analytical sensitivity test",
    )
    derated = apply_thermal_operating_point(baseline, point)
    original_profile = baseline.resolve_component_profile("hbm0")
    profile = derated.resolve_component_profile("hbm0")

    assert derated.hardware.get_component("hbm0").bandwidth_gbps == memory.bandwidth_gbps * 0.5
    assert profile.effective_bandwidth_gb_s == pytest.approx(original_profile.effective_bandwidth_gb_s * 0.5)
    assert TopologyRouter(derated.hardware).route("gpu0", "hbm0", 1024)[0].bandwidth_gbps == pytest.approx(
        TopologyRouter(baseline.hardware).route("gpu0", "hbm0", 1024)[0].bandwidth_gbps * 0.5
    )
    assert derated.hardware.get_component("hbm1") == baseline.hardware.get_component("hbm1")
    assert derated.resolve_component_profile("hbm1") == baseline.resolve_component_profile("hbm1")
    assert baseline.hardware.get_component("hbm0") == memory
