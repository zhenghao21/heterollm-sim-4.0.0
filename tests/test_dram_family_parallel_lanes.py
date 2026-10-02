from dataclasses import replace

import pytest

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cost_models import HBMProfile, HostMemoryProfile
from heterollm_sim.data_motion import endpoint_service, memory_service
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.web import scenario_to_payload


@pytest.mark.parametrize("profile_type", [HBMProfile, HostMemoryProfile])
def test_dram_family_parallel_lanes_default_preserves_service(profile_type):
    profile = profile_type(bandwidth_gb_s=100.0, read_latency_ns=80.0)
    expected = memory_service(
        read_bytes=8192,
        write_bytes=0,
        bandwidth_gb_s=profile.effective_bandwidth_gb_s,
        read_latency_ns=profile.read_latency_ns,
        write_latency_ns=profile.write_latency_ns,
        transaction_bytes=profile.transaction_bytes,
        max_outstanding_requests=profile.max_outstanding_requests,
    )
    actual = profile.memory_service(8192)
    assert actual == expected
    assert actual["parallel_lanes"] == 1
    assert actual["request_window"] == profile.max_outstanding_requests


@pytest.mark.parametrize("service_model", ["analytical", "serialized", "overlapped"])
def test_dram_family_parallel_lanes_expands_latency_window(service_model):
    base = HBMProfile(
        bandwidth_gb_s=100.0,
        service_model=service_model,
        read_latency_ns=100.0,
        transaction_bytes=256,
        max_outstanding_requests=4,
    )
    wide = replace(base, parallel_lanes=2)
    base_service = base.memory_service(256 * 8)
    wide_service = wide.memory_service(256 * 8)
    assert wide_service["physical_bytes"] == base_service["physical_bytes"]
    assert wide_service["bandwidth_service_ns"] == base_service["bandwidth_service_ns"]
    assert wide_service["request_window"] == 8
    assert wide_service["effective_outstanding"] == 8
    assert wide_service["latency_service_ns"] < base_service["latency_service_ns"]


def test_dram_family_parallel_lanes_reaches_endpoint_and_config_roundtrip():
    scenario = build_reference_scenario()
    profiles = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    profiles["hbm"]["legacy-hbm"] = replace(
        profiles["hbm"]["legacy-hbm"],
        parallel_lanes=4,
        read_latency_ns=100.0,
    )
    changed = replace(scenario, component_profiles=profiles)
    component = changed.hardware.get_component("hbm0")
    endpoint = endpoint_service(component, 256 * 64, read=True, name="test")
    assert endpoint.metadata["parallel_lanes"] == 4
    assert endpoint.metadata["request_window"] == 32 * 4
    assert endpoint.metadata["effective_outstanding"] == 64
    assert endpoint.metadata["transferred_bytes"] == 256 * 64

    parsed = scenario_from_dict(scenario_to_payload(changed))
    assert parsed.hbm_profile.parallel_lanes == 4
    assert parsed.hardware.get_component("hbm0").metadata["memory_service"]["parallel_lanes"] == 4


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "2"])
def test_dram_family_parallel_lanes_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        HBMProfile(parallel_lanes=bad)
    with pytest.raises(ValueError):
        HostMemoryProfile(parallel_lanes=bad)
