"""Directional payload timing remains shape-scaled and cache-resolved."""

import json
from dataclasses import asdict, fields, replace

import pytest

from heterollm_sim.cost_models import HBMProfile, HostMemoryProfile, MemoryWorkload
from tests.test_memory_profiles import CASES, demands
from tests.test_cost_models import cpu_profile, gpu_profile


@pytest.mark.parametrize("profile_type", (HostMemoryProfile, HBMProfile))
def test_defaults_and_equal_directions_preserve_exact_scalar_cost(profile_type):
    profile = profile_type(bandwidth_gb_s=13.0, efficiency=0.75)
    assert profile.effective_read_bandwidth_gb_s == profile.effective_write_bandwidth_gb_s == 9.75
    for reads, writes in ((0, 0), (1, 0), (0, 1), (257, 259), (12345, 6789)):
        for override in (None, 0.73, 9.75):
            expected = (reads + writes) / (9.75 if override is None else override)
            assert profile.memory_service(reads, writes, bandwidth_gb_s=override)["service_ns"] == expected
    symmetric = replace(profile, read_bandwidth_gb_s=13, write_bandwidth_gb_s=13)
    assert symmetric.memory_service(12345, 6789)["service_ns"] == profile.memory_service(12345, 6789)["service_ns"]


@pytest.mark.parametrize("profile_type", (HostMemoryProfile, HBMProfile))
def test_directional_rates_and_shape_override(profile_type):
    profile = profile_type(bandwidth_gb_s=100, efficiency=0.5,
                           read_bandwidth_gb_s=200, write_bandwidth_gb_s=10)
    assert profile.effective_bandwidth_gb_s == 50
    assert profile.effective_read_bandwidth_gb_s == 100
    assert profile.effective_write_bandwidth_gb_s == 5
    assert profile.memory_service(1000)["service_ns"] == 10
    assert profile.memory_service(0, 1000)["service_ns"] == 200
    mixed = profile.memory_service(1000, 1000, bandwidth_gb_s=25)
    assert mixed["read_bandwidth_gb_s"] == 50
    assert mixed["write_bandwidth_gb_s"] == 2.5
    assert mixed["bandwidth_gb_s"] == 25  # Reference, not an averaged direction.
    assert mixed["service_ns"] == 420
    fallback = replace(profile, read_bandwidth_gb_s=None)
    assert fallback.memory_service(1000, bandwidth_gb_s=25)["service_ns"] == 40


@pytest.mark.parametrize("profile_type", (HostMemoryProfile, HBMProfile))
def test_directional_latency_and_zero_payload(profile_type):
    profile = profile_type(bandwidth_gb_s=100, read_bandwidth_gb_s=100,
                           write_bandwidth_gb_s=1, read_latency_ns=100,
                           write_latency_ns=300, max_outstanding_requests=1)
    assert profile.memory_service(1, 1)["service_ns"] == 400
    assert profile.memory_service(1)["service_ns"] == 100
    assert profile.memory_service(0, 1000)["service_ns"] == 1200
    zero = profile.memory_service(0, 0)
    for key in ("service_ns", "bandwidth_service_ns", "latency_service_ns",
                "read_bandwidth_service_ns", "write_bandwidth_service_ns"):
        assert zero[key] == 0


@pytest.mark.parametrize("profile_type", (HostMemoryProfile, HBMProfile))
@pytest.mark.parametrize("field_name", ("read_bandwidth_gb_s", "write_bandwidth_gb_s"))
@pytest.mark.parametrize("invalid", (0, -1, float("inf"), float("-inf"), float("nan"), True, False))
def test_explicit_directional_rate_validation(profile_type, field_name, invalid):
    with pytest.raises(ValueError, match=field_name):
        profile_type(bandwidth_gb_s=10, **{field_name: invalid})


def test_dataclass_roundtrip_and_host_to_hbm_preserve_directions():
    host = HostMemoryProfile(bandwidth_gb_s=13, efficiency=0.75,
                             read_bandwidth_gb_s=21, write_bandwidth_gb_s=3)
    assert HostMemoryProfile(**json.loads(json.dumps(asdict(host)))) == host
    hbm = HBMProfile(**{f.name: getattr(host, f.name) for f in fields(HBMProfile)})
    assert hbm.memory_service(123, 456) == host.memory_service(123, 456)
    assert HBMProfile(**json.loads(json.dumps(asdict(hbm)))) == hbm


@pytest.mark.parametrize("device_factory,profile_type,estimator,workload", CASES,
                         ids=[case[2].__name__ for case in CASES])
def test_all_gpu_and_cpu_paths_apply_directional_rates(
    device_factory, profile_type, estimator, workload,
):
    device = device_factory()
    profile = profile_type(bandwidth_gb_s=12, efficiency=0.75, energy_pj_per_byte=2)
    old = estimator(device, profile, workload)
    configured = replace(profile, read_bandwidth_gb_s=24, write_bandwidth_gb_s=0.125)
    new = estimator(device, configured, workload)
    report = new.metadata["backing_memory_service"]
    shape = report["bandwidth_gb_s"] / profile.effective_bandwidth_gb_s
    assert report["read_bandwidth_gb_s"] == pytest.approx(18 * shape)
    assert report["write_bandwidth_gb_s"] == pytest.approx(0.09375 * shape)
    expected = report["physical_read_bytes"] / (18 * shape) + report["physical_write_bytes"] / (0.09375 * shape)
    assert report["service_ns"] == pytest.approx(expected)
    demand = next(d for d in demands(new) if d.resource_id == configured.resource_id)
    assert demand.service_ns == pytest.approx(expected)
    assert new.bytes_moved == old.bytes_moved
    assert new.energy_pj == old.energy_pj
    assert new.metadata["cache"]["physical_bytes"] == old.metadata["cache"]["physical_bytes"]


@pytest.mark.parametrize("factory,profile_type", ((cpu_profile, HostMemoryProfile), (gpu_profile, HBMProfile)))
def test_pure_read_generic_path_is_not_limited_by_slow_writes(factory, profile_type):
    device = factory()
    memory = profile_type(bandwidth_gb_s=10)
    workload = MemoryWorkload(read_bytes=32768, write_bytes=0)
    old = device.estimate_memory(memory, workload)
    new = device.estimate_memory(replace(memory, write_bandwidth_gb_s=0.0001), workload)
    assert demands(new) == demands(old)
    assert new.service_ns == old.service_ns
