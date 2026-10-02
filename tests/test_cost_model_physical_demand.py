from dataclasses import replace

import pytest

from heterollm_sim.cost_models import (
    GemmWorkload,
    HBMProfile,
    HostMemoryProfile,
    MemoryWorkload,
    estimate_cpu_gemm,
    estimate_cpu_memory,
    estimate_gpu_gemm,
    estimate_gpu_memory,
)
from tests.test_cost_models import cpu_profile, gpu_profile


def _resource_demand(estimate, resource_id):
    return next(
        demand
        for phase in estimate.phases
        for demand in phase.demands
        if demand.resource_id == resource_id
    )


@pytest.mark.parametrize(
    "profile_type,device_factory,estimator",
    [
        (HBMProfile, gpu_profile, estimate_gpu_memory),
        (HostMemoryProfile, cpu_profile, estimate_cpu_memory),
    ],
)
@pytest.mark.parametrize("service_model", ["serialized", "overlapped"])
def test_memory_cost_demand_matches_rounded_backing_service(
    profile_type, device_factory, estimator, service_model
):
    profile = profile_type(
        bandwidth_gb_s=100.0,
        service_model=service_model,
        transaction_bytes=256,
        max_outstanding_requests=1,
        energy_pj_per_byte=2.0,
    )
    estimate = estimator(device_factory(), profile, MemoryWorkload(read_bytes=1, write_bytes=1))
    service = estimate.metadata["backing_memory_service"]
    demand = _resource_demand(estimate, profile.resource_id)
    assert service["physical_bytes"] == 512
    assert estimate.metadata["cache"]["physical_bytes"] == 512
    assert demand.bytes_moved == 512
    assert demand.energy_pj == 1024.0


@pytest.mark.parametrize(
    "profile_type,device_factory,estimator",
    [
        (HBMProfile, gpu_profile, estimate_gpu_gemm),
        (HostMemoryProfile, cpu_profile, estimate_cpu_gemm),
    ],
)
@pytest.mark.parametrize("service_model", ["serialized", "overlapped"])
def test_gemm_cost_demand_matches_rounded_backing_service(
    profile_type, device_factory, estimator, service_model
):
    profile = profile_type(
        bandwidth_gb_s=100.0,
        service_model=service_model,
        transaction_bytes=256,
        max_outstanding_requests=1,
        energy_pj_per_byte=2.0,
    )
    workload = GemmWorkload(m=1, n=1, k=1, activation_bits=16, weight_bits=16, output_bits=32)
    estimate = estimator(device_factory(), profile, workload)
    service = estimate.metadata["backing_memory_service"]
    demand = _resource_demand(estimate, profile.resource_id)
    assert service["physical_bytes"] == estimate.metadata["cache"]["physical_bytes"]
    assert demand.bytes_moved == service["physical_bytes"]
    assert demand.energy_pj == service["physical_bytes"] * 2.0


def test_analytical_memory_demand_retains_payload_scope():
    profile = HBMProfile(bandwidth_gb_s=100.0, energy_pj_per_byte=2.0)
    estimate = estimate_gpu_memory(gpu_profile(), profile, MemoryWorkload(read_bytes=1, write_bytes=1))
    demand = _resource_demand(estimate, profile.resource_id)
    assert demand.bytes_moved == 2
    assert demand.energy_pj == 4.0
