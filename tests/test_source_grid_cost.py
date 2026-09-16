from dataclasses import replace
import pytest
from heterollm_sim.cost_models import TensorKernelWorkload, HBMProfile, estimate_gpu_tensor_kernel
from heterollm_sim.reference import build_reference_scenario


def base():
    gpu=build_reference_scenario().component_profiles["gpu"]["legacy-gpu"]
    gpu=replace(gpu,tensor_core=replace(gpu.tensor_core,sm_count=84))
    return gpu,HBMProfile(bandwidth_gb_s=960.)

def test_cta_bound_changes_compute_only_and_preserves_io_energy_launch():
    gpu,memory=base();load=TensorKernelWorkload(1000000,4096,1024,streaming_fraction=1.)
    old=estimate_gpu_tensor_kernel(gpu,memory,load)
    new=estimate_gpu_tensor_kernel(gpu,memory,replace(load,source_grid_ctas=4))
    before={d.resource_id:d for p in old.phases for d in p.demands}
    after={d.resource_id:d for p in new.phases for d in p.demands}
    assert after[gpu.scalar_resource_id].service_ns==pytest.approx(before[gpu.scalar_resource_id].service_ns*(21*gpu.occupancy*gpu.attainable_efficiency))
    for name,demand in before.items():
        if name!=gpu.scalar_resource_id:assert after[name]==demand
    assert old.bytes_moved==new.bytes_moved and old.energy_pj==new.energy_pj
    assert new.metadata["source_grid_parallelism"]["memory_bandwidth_fraction"]==1.

def test_saturated_grid_does_not_invent_extra_wave_penalty():
    gpu,memory=base();load=TensorKernelWorkload(1000000,4096,1024)
    old=estimate_gpu_tensor_kernel(gpu,memory,load)
    new=estimate_gpu_tensor_kernel(gpu,memory,replace(load,source_grid_ctas=128))
    assert old.service_ns==new.service_ns
    assert tuple(p.demands for p in old.phases)==tuple(p.demands for p in new.phases)

@pytest.mark.parametrize("value",[0,-1,True,1.5])
def test_invalid_grid_is_rejected(value):
    with pytest.raises(ValueError):TensorKernelWorkload(1,4,4,source_grid_ctas=value)


def test_preexisting_low_efficiency_is_not_penalized_a_second_time():
    gpu,memory=base();gpu=replace(gpu,attainable_efficiency=.01)
    load=TensorKernelWorkload(1000000,4096,1024)
    a=estimate_gpu_tensor_kernel(gpu,memory,load)
    b=estimate_gpu_tensor_kernel(gpu,memory,replace(load,source_grid_ctas=4))
    assert a.service_ns==b.service_ns
    assert b.metadata["source_grid_parallelism"]["scalar_throughput_fraction"]==1.
