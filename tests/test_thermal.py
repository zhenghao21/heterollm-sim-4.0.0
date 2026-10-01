from dataclasses import replace

import pytest

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.thermal import ThermalOperatingPoint, apply_thermal_operating_point


@pytest.mark.parametrize("measured", (None, 100.0))
def test_thermal_derating_reaches_shared_profile_and_route_bandwidth(measured):
    baseline = build_reference_scenario()
    # This test asserts off-domain independence. The reference HBM banks now
    # deliberately share an aggregate controller, so declare independent bank
    # ceilings rather than assuming an aggregate member cannot affect peers.
    independent_profiles = {kind:dict(registry) for kind,registry in baseline.component_profiles.items()}
    independent_profiles["hbm"] = {key:replace(value,bandwidth_gb_s=512) for key,value in independent_profiles["hbm"].items()}
    baseline = replace(baseline, component_profiles=independent_profiles, hardware=replace(baseline.hardware, components=tuple(
        replace(c, metadata={**c.metadata, 'memory_bandwidth_scope':'per_component',
                             'memory_aggregate_owner':''})
        if c.normalized_kind in ('hbm','hbm_stack') else c
        for c in baseline.hardware.components)))
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
    memory = baseline.hardware.get_component("hbm0")
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

def test_thermal_gpu_change_invalidates_measured_wall_not_original_profile():
    from heterollm_sim.thermal import _derate_profile
    from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile, KernelSample
    from heterollm_sim.cost_models import GemmWorkload, HBMProfile, estimate_gpu_gemm
    gpu = next(iter(build_reference_scenario().component_profiles['gpu'].values()))
    descriptor = KernelCapability('synthetic',('fp16',),'fp16','decode','simt','test only',
        samples=(KernelSample(1,1024,1024,100,200,1,20,'test'),),
        surface_model='measured_surrogate')
    gpu = replace(gpu,kernel_model=KernelModelProfile('gpu','runtime','arch',(descriptor,)))
    workload = GemmWorkload(1,1024,1024,activation_bits=16,weight_bits=16,execution_phase='decode')
    before = estimate_gpu_gemm(gpu,HBMProfile(1000),workload)
    assert before.metadata['prediction']['model'] == 'measured_surrogate'
    for changes in ({'frequency_scale':.5}, {'memory_bandwidth_scale':.5}, {'latency_scale':2}):
        point = ThermalOperatingPoint('gpu',enabled=True,evidence='sensitivity',**changes)
        changed = _derate_profile(gpu,point)
        assert not changed.kernel_model.kernels[0].samples
        assert 'invalidated' in changed.kernel_model.kernels[0].evidence
        assert estimate_gpu_gemm(changed,HBMProfile(1000),workload).metadata['prediction']['model']=='analytical'
    assert gpu.kernel_model.kernels[0].samples
    unchanged = _derate_profile(gpu,ThermalOperatingPoint('gpu',enabled=True,evidence='identity'))
    assert unchanged.kernel_model == gpu.kernel_model



def test_thermal_aggregate_bank_rebinds_shared_physical_ceiling():
    baseline = build_reference_scenario()
    bank = baseline.hardware.get_component('hbm0')
    baseline = replace(baseline, hardware=replace(baseline.hardware, components=tuple(
        replace(c,metadata={**c.metadata,'thermal_domain_id':'one-bank'})
        if c.component_id == 'hbm0' else c for c in baseline.hardware.components)))
    original = baseline.hardware.get_component('hbm1')
    changed = apply_thermal_operating_point(baseline,ThermalOperatingPoint(
        'one-bank',enabled=True,memory_bandwidth_scale=.5,evidence='sensitivity'))
    peer = changed.hardware.get_component('hbm1')
    assert peer.bandwidth_gbps == original.bandwidth_gbps
    assert peer.metadata['memory_service']['physical_bandwidth_gb_s'] == pytest.approx(
        original.metadata['memory_service']['physical_bandwidth_gb_s'] - bank.bandwidth_gbps / 16)
    assert baseline.hardware.get_component('hbm1') == original

@pytest.mark.parametrize('domain_kind', ['memory','link'])
def test_memory_or_link_only_thermal_change_invalidates_gpu_surfaces(domain_kind):
    from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile, KernelSample
    from heterollm_sim.cost_models import GemmWorkload, estimate_gpu_gemm
    baseline=build_reference_scenario()
    profiles={kind:dict(registry) for kind,registry in baseline.component_profiles.items()}
    sample=KernelSample(1,1024,1024,100,200,1,20,'synthetic')
    descriptor=KernelCapability('test',('fp16',),'fp16','decode','simt','test',samples=(sample,))
    profiles['gpu']={key:replace(gpu,kernel_model=KernelModelProfile('gpu','runtime','arch',(descriptor,)))
                     for key,gpu in profiles['gpu'].items()}
    hardware=baseline.hardware
    if domain_kind=='memory':
        hardware=replace(hardware,components=tuple(replace(c,metadata={**c.metadata,'thermal_domain_id':'external'})
            if c.component_id=='hbm0' else c for c in hardware.components))
    else:
        hardware=replace(hardware,links=tuple(replace(link,metadata={**link.metadata,'thermal_domain_id':'external'})
            if link.link_id=='gpu-hbm0' else link for link in hardware.links))
    baseline=replace(baseline,hardware=hardware,component_profiles=profiles)
    unchanged=apply_thermal_operating_point(baseline,ThermalOperatingPoint('external',enabled=True,evidence='identity'))
    assert unchanged.resolve_component_profile('gpu0').kernel_model.kernels[0].samples
    changed=apply_thermal_operating_point(baseline,ThermalOperatingPoint('external',enabled=True,
        memory_bandwidth_scale=.5 if domain_kind=='memory' else 1,
        link_bandwidth_scale=.5 if domain_kind=='link' else 1,evidence='test'))
    gpu=changed.resolve_component_profile('gpu0')
    assert gpu.tensor_core.frequency_ghz==baseline.resolve_component_profile('gpu0').tensor_core.frequency_ghz
    assert baseline.resolve_component_profile('gpu0').kernel_model.kernels[0].samples
    assert not gpu.kernel_model.kernels[0].samples
    assert changed.hardware.metadata['kernel_calibration_invalidation']['scope']=='all_gpu_profiles'
    cost=estimate_gpu_gemm(gpu,changed.resolve_component_profile('hbm0'),
        GemmWorkload(1,1024,1024,activation_bits=16,weight_bits=16,execution_phase='decode'))
    assert cost.metadata['prediction']['model']=='analytical'
