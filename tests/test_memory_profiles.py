"""Opt-in analytical backing-memory timing; defaults preserve legacy costs."""

from dataclasses import replace

import pytest

from heterollm_sim.contracts import EvidenceStatus
from heterollm_sim.communication import declared_resource_owners
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.planner import (
    _compilation_scope,
    _rank_memory_resource,
    _parallel_plan,
    _route_resident_memory_demands,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.cost_models import (
    CacheHierarchyProfile,
    ElementwiseWorkload,
    FusedAttentionWorkload,
    GemmWorkload,
    HBMProfile,
    HostMemoryProfile,
    MemoryWorkload,
    ReductionWorkload,
    TensorKernelWorkload,
    estimate_cpu_elementwise,
    estimate_cpu_gemm,
    estimate_cpu_memory,
    estimate_cpu_reduction,
    estimate_gpu_elementwise,
    estimate_gpu_fused_attention,
    estimate_gpu_gemm,
    estimate_gpu_memory,
    estimate_gpu_reduction,
    estimate_gpu_tensor_kernel,
)
from tests.test_cost_models import cache_hierarchy, cpu_profile, gpu_profile


PROFILES = (HBMProfile, HostMemoryProfile)


@pytest.mark.parametrize("profile_type", PROFILES)
def test_defaults_preserve_exact_payload_bandwidth_cost(profile_type):
    profile = profile_type(bandwidth_gb_s=13.0, efficiency=0.75)
    assert (profile.read_latency_ns, profile.write_latency_ns) == (0.0, 0.0)
    assert (profile.transaction_bytes, profile.max_outstanding_requests) == (256, 32)
    for reads, writes in ((0, 0), (1, 0), (0, 1), (257, 259), (12345, 6789)):
        expected = (reads + writes) / (13.0 * 0.75)
        for candidate in (profile, replace(profile, transaction_bytes=1,
                                           max_outstanding_requests=1)):
            service = candidate.memory_service(reads, writes)
            assert service["service_ns"] == expected
            assert service["bandwidth_service_ns"] == expected
            assert service["latency_service_ns"] == 0.0
            assert service["logical_bytes"] == service["physical_bytes"] == reads + writes
            assert not service["latency_model_enabled"]


@pytest.mark.parametrize("profile_type", PROFILES)
def test_zero_traffic_has_no_latency_transactions_or_service(profile_type):
    service = profile_type(
        bandwidth_gb_s=1.0, read_latency_ns=1e9, write_latency_ns=2e9,
    ).memory_service(0, 0)
    for key in ("service_ns", "bandwidth_service_ns", "latency_service_ns",
                "single_request_latency_ns", "request_slot_time_ns",
                "concurrency_service_ns", "read_transactions", "write_transactions",
                "transaction_count", "effective_outstanding", "physical_bytes"):
        assert service[key] == 0
    assert service["service_source"] == "zero_traffic"
    assert not service["latency_bound"]


@pytest.mark.parametrize("profile_type", PROFILES)
def test_latency_is_per_transaction_not_per_byte(profile_type):
    profile = profile_type(bandwidth_gb_s=1e9, read_latency_ns=100.0,
                           write_latency_ns=1e9, max_outstanding_requests=1)
    for payload in (1, 128, 256):
        service = profile.memory_service(payload)
        assert service["read_transactions"] == 1
        assert service["write_transactions"] == 0
        assert service["service_ns"] == 100.0
        assert service["physical_bytes"] == payload
    assert profile.memory_service(257)["service_ns"] == 200.0
    concurrent = replace(profile, max_outstanding_requests=32).memory_service(257)
    assert concurrent["service_ns"] == 100.0
    assert concurrent["effective_outstanding"] == 2
    assert concurrent["physical_bytes"] == 257  # Do not pad to 512 bytes.


@pytest.mark.parametrize("profile_type", PROFILES)
def test_latency_and_concurrency_controlled_cases(profile_type):
    profile = profile_type(bandwidth_gb_s=1e9, read_latency_ns=100.0)
    assert profile.memory_service(64 * 256)["service_ns"] == 200.0
    assert replace(profile, read_latency_ns=200.0).memory_service(64 * 256)["service_ns"] == 400.0
    assert replace(profile, max_outstanding_requests=16).memory_service(64 * 256)["service_ns"] == 400.0
    assert replace(profile, max_outstanding_requests=64).memory_service(64 * 256)["service_ns"] == 100.0
    service = profile.memory_service(33 * 256)
    # Aggregate slot-time envelope, NOT an integer-wave or cycle-accurate schedule.
    assert service["service_ns"] == 103.125
    assert service["service_source"] == "latency_concurrency"
    assert service["evidence"] == EvidenceStatus.ANALYTICAL.value
    assert service["cycle_accurate"] is False
    assert "request_dependencies_and_achieved_mlp" in service["unmodeled_terms"]


@pytest.mark.parametrize("profile_type", PROFILES)
def test_reads_and_writes_share_one_outstanding_window(profile_type):
    profile = profile_type(bandwidth_gb_s=1e9, read_latency_ns=100.0,
                           write_latency_ns=300.0, max_outstanding_requests=16)
    service = profile.memory_service(64 * 256, 32 * 256)
    assert (service["read_transactions"], service["write_transactions"]) == (64, 32)
    assert service["transaction_count"] == 96
    assert service["effective_outstanding"] == 16
    assert service["request_slot_time_ns"] == 16000.0
    assert service["service_ns"] == 1000.0  # Not max(read bound, write bound).
    assert profile.memory_service(0, 1)["service_ns"] == 300.0
    assert profile.memory_service(1, 0)["service_ns"] == 100.0


@pytest.mark.parametrize("profile_type", PROFILES)
def test_bandwidth_bound_and_effective_override_are_not_double_discounted(profile_type):
    profile = profile_type(bandwidth_gb_s=16.0, efficiency=0.5,
                           read_latency_ns=10.0, write_latency_ns=20.0)
    service = profile.memory_service(8192, 8192)
    assert service["service_ns"] == 2048.0
    assert service["service_source"] == "bandwidth"
    assert not service["latency_bound"]
    assert replace(profile, bandwidth_gb_s=32.0).memory_service(8192, 8192)["service_ns"] == 1024.0
    assert replace(profile, max_outstanding_requests=64).memory_service(8192, 8192)["service_ns"] == 2048.0
    override = profile.memory_service(8192, 8192, bandwidth_gb_s=4.0)
    assert override["bandwidth_gb_s"] == 4.0
    assert override["service_ns"] == 4096.0
    with pytest.raises(ValueError, match="bandwidth_gb_s"):
        profile.memory_service(1, bandwidth_gb_s=0.0)


@pytest.mark.parametrize("profile_type", PROFILES)
def test_timing_fields_and_byte_inputs_are_validated(profile_type):
    for field in ("read_latency_ns", "write_latency_ns"):
        for invalid in (-1, float("inf"), float("nan"), True):
            with pytest.raises(ValueError, match=field):
                profile_type(bandwidth_gb_s=1, **{field: invalid})
    for field in ("transaction_bytes", "max_outstanding_requests"):
        for invalid in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match=field):
                profile_type(bandwidth_gb_s=1, **{field: invalid})
    profile = profile_type(bandwidth_gb_s=1)
    for invalid in (-1, 0.5, True):
        with pytest.raises(ValueError, match="read_bytes"):
            profile.memory_service(invalid)
        with pytest.raises(ValueError, match="write_bytes"):
            profile.memory_service(0, invalid)


CASES = (
    (gpu_profile, HBMProfile, estimate_gpu_gemm, GemmWorkload(m=16, n=16, k=16)),
    (gpu_profile, HBMProfile, estimate_gpu_fused_attention,
     FusedAttentionWorkload(batch_tokens=2, context_tokens=32, hidden_size=64)),
    (gpu_profile, HBMProfile, estimate_gpu_tensor_kernel,
     TensorKernelWorkload(operations=32, read_bytes=513, write_bytes=257, streaming_fraction=1.0)),
    (gpu_profile, HBMProfile, estimate_gpu_elementwise, ElementwiseWorkload(elements=256)),
    (gpu_profile, HBMProfile, estimate_gpu_reduction,
     ReductionWorkload(input_elements=256, output_elements=8)),
    (gpu_profile, HBMProfile, estimate_gpu_memory, MemoryWorkload(read_bytes=513, write_bytes=257)),
    (cpu_profile, HostMemoryProfile, estimate_cpu_gemm, GemmWorkload(m=16, n=16, k=16)),
    (cpu_profile, HostMemoryProfile, estimate_cpu_elementwise, ElementwiseWorkload(elements=256)),
    (cpu_profile, HostMemoryProfile, estimate_cpu_reduction,
     ReductionWorkload(input_elements=256, output_elements=8)),
    (cpu_profile, HostMemoryProfile, estimate_cpu_memory, MemoryWorkload(read_bytes=513, write_bytes=257)),
)


def demands(estimate):
    return tuple(demand for phase in estimate.phases for demand in phase.demands)


@pytest.mark.parametrize("device_factory,profile_type,estimator,workload", CASES,
                         ids=[case[2].__name__ for case in CASES])
def test_all_paths_use_resolved_backing_bytes_and_preserve_default_costs(
    device_factory, profile_type, estimator, workload,
):
    device = device_factory()
    profile = profile_type(bandwidth_gb_s=12.0, efficiency=0.75, energy_pj_per_byte=2.0)
    baseline = estimator(device, profile, workload)
    # Transaction geometry alone must not alter any legacy numerical demand.
    explicit_zero = estimator(device, replace(profile, transaction_bytes=7,
                                              max_outstanding_requests=1), workload)
    assert explicit_zero.service_ns == baseline.service_ns
    assert explicit_zero.energy_pj == baseline.energy_pj
    assert explicit_zero.utilization == baseline.utilization
    assert explicit_zero.bytes_moved == baseline.bytes_moved
    assert demands(explicit_zero) == demands(baseline)
    configured = replace(profile, read_latency_ns=1e5, write_latency_ns=2e5)
    estimate = estimator(device, configured, workload)
    cache = estimate.metadata["cache"]
    service = estimate.metadata["backing_memory_service"]
    physical_bytes = cache["backing_read_bytes"] + cache["backing_write_bytes"]
    assert cache["physical_bytes"] == service["physical_bytes"] == physical_bytes
    assert service["read_transactions"] == (cache["backing_read_bytes"] + 255) // 256
    assert service["write_transactions"] == (cache["backing_write_bytes"] + 255) // 256
    memory = next(row for row in demands(estimate) if row.resource_id == profile.resource_id)
    old_memory = next(row for row in demands(baseline) if row.resource_id == profile.resource_id)
    assert memory.bytes_moved == old_memory.bytes_moved == physical_bytes
    assert memory.energy_pj == old_memory.energy_pj == physical_bytes * 2.0
    assert old_memory.service_ns == service["bandwidth_service_ns"]
    assert memory.service_ns == service["service_ns"] > old_memory.service_ns
    assert estimate.service_ns > baseline.service_ns
    assert tuple(row for row in demands(estimate) if row.resource_id != profile.resource_id) == tuple(
        row for row in demands(baseline) if row.resource_id != profile.resource_id
    )


@pytest.mark.parametrize("device_factory,profile_type,estimator", (
    (gpu_profile, HBMProfile, estimate_gpu_memory),
    (cpu_profile, HostMemoryProfile, estimate_cpu_memory),
))
def test_cache_filtering_happens_before_transactions_and_no_payload_padding(
    device_factory, profile_type, estimator,
):
    first = cache_hierarchy(capacity_bytes=4096).levels[0]
    hierarchy = CacheHierarchyProfile(levels=(
        first, replace(first, name="l2", resource_id="device.l2", capacity_bytes=8192),
    ))
    device = device_factory(cache=hierarchy)
    memory = profile_type(bandwidth_gb_s=1e9, read_latency_ns=100.0,
                          write_latency_ns=200.0, energy_pj_per_byte=2.0)
    workload = MemoryWorkload(read_bytes=1025, write_bytes=259,
                              reuse_factor=4.0, streaming_fraction=0.0)
    estimate = estimator(device, memory, workload)
    cache = estimate.metadata["cache"]
    assert cache["logical_bytes"] == 1284
    assert cache["levels"][0]["read_miss_bytes"] == 256
    assert cache["levels"][1]["read_miss_bytes"] == 64
    assert cache["physical_read_bytes"] == 64
    assert cache["physical_write_bytes"] == 259  # Closed interval still flushes all writes.
    assert cache["physical_bytes"] == 323
    service = estimate.metadata["backing_memory_service"]
    assert (service["read_transactions"], service["write_transactions"]) == (1, 2)
    assert service["effective_outstanding"] == 3
    assert service["service_ns"] == 200.0
    backing = next(row for row in demands(estimate) if row.resource_id == memory.resource_id)
    assert backing.bytes_moved == 323
    assert backing.energy_pj == 646.0


@pytest.mark.parametrize("device_factory,profile_type,estimator", (
    (gpu_profile, HBMProfile, estimate_gpu_memory),
    (cpu_profile, HostMemoryProfile, estimate_cpu_memory),
))
def test_all_cache_hits_have_zero_backing_cost_even_with_high_latency(
    device_factory, profile_type, estimator,
):
    # The aggregate byte hit curve rounds this tiny fully reused payload to a hit.
    device = device_factory(cache=cache_hierarchy(capacity_bytes=4096))
    memory = profile_type(bandwidth_gb_s=100.0, energy_pj_per_byte=2.0)
    workload = MemoryWorkload(read_bytes=1, reuse_factor=4.0, streaming_fraction=0.0)
    baseline = estimator(device, memory, workload)
    estimate = estimator(device, replace(memory, read_latency_ns=1e9), workload)
    cache = estimate.metadata["cache"]
    assert cache["logical_bytes"] == 1
    assert cache["physical_bytes"] == 0
    assert estimate.metadata["backing_memory_service"]["service_source"] == "zero_traffic"
    backing = next(row for row in demands(estimate) if row.resource_id == memory.resource_id)
    assert (backing.service_ns, backing.bytes_moved, backing.energy_pj) == (0.0, 0, 0.0)
    assert demands(estimate) == demands(baseline)
    assert estimate.service_ns == baseline.service_ns


def test_gemm_shape_bandwidth_override_preserves_legacy_value():
    gpu = gpu_profile()
    gpu = replace(gpu, tensor_core=replace(gpu.tensor_core, sm_count=8))
    hbm = HBMProfile(bandwidth_gb_s=800.0, efficiency=0.5)
    workload = GemmWorkload(m=16, n=16, k=16)
    estimate = estimate_gpu_gemm(gpu, hbm, workload)
    bandwidth = estimate.metadata["hbm_bandwidth"]["shape_effective_hbm_bandwidth_gb_s"]
    service = estimate.metadata["backing_memory_service"]
    assert bandwidth < hbm.effective_bandwidth_gb_s
    assert service["bandwidth_gb_s"] == bandwidth
    assert service["service_ns"] == workload.minimum_io_bytes / bandwidth


def test_compute_only_and_launch_only_do_not_invent_memory_service():
    gpu = gpu_profile()
    hbm = HBMProfile(bandwidth_gb_s=1.0, read_latency_ns=1e9, write_latency_ns=1e9)
    for workload in (TensorKernelWorkload(operations=64, read_bytes=0, write_bytes=0),
                     TensorKernelWorkload(operations=0, read_bytes=0, write_bytes=0, launch_only=True)):
        baseline = estimate_gpu_tensor_kernel(gpu, replace(hbm, read_latency_ns=0.0,
                                                          write_latency_ns=0.0), workload)
        estimate = estimate_gpu_tensor_kernel(gpu, hbm, workload)
        assert demands(estimate) == demands(baseline)
        assert all(row.resource_id != hbm.resource_id for row in demands(estimate))
        assert estimate.service_ns == baseline.service_ns


def test_resident_router_uses_phase_cache_direction_bytes():
    scenario = build_reference_scenario()
    scenario = replace(scenario, hardware=replace(
        scenario.hardware,
        components=tuple(
            replace(component, metadata={**component.metadata,
                                         "resident_access_path": "topology"})
            if component.component_id == "hbm0" else component
            for component in scenario.hardware.components
        ),
    ))
    from heterollm_sim.planner import compile_scenario
    schedule = compile_scenario(scenario)
    routed = next(task for task in schedule.tasks
                   if task.metadata.get("resident_memory_paths"))
    paths = routed.metadata["resident_memory_paths"]
    cache = routed.metadata["phase_metadata"]["cache"]
    assert paths[0]["read_bytes"] == cache["physical_read_bytes"]
    assert paths[0]["write_bytes"] == cache["physical_write_bytes"]
    assert paths[0]["direction_evidence"] == "resolved_backing_traffic"


def test_resident_router_coalesces_hops_with_one_shared_phy_owner():
    scenario = build_reference_scenario()
    link = next(row for row in scenario.hardware.links if row.link_id == "gpu-hbm0")
    port = scenario.hardware.component_map()["hbm0"].ports[0]
    bridge = ComponentSpec("bridge", "memory", ports=(
        replace(port, port_id="in"), replace(port, port_id="out"),
    ))
    links = tuple(row for row in scenario.hardware.links if row.link_id != link.link_id) + (
        replace(link, link_id="memory-bridge", source_component="hbm0", source_port="host",
                target_component="bridge", target_port="in", latency_ns=0.0,
                metadata={"shared_bidirectional": True}),
        replace(link, link_id="bridge-gpu", source_component="bridge", source_port="out",
                target_component="gpu0", target_port="hbm0", latency_ns=0.0,
                metadata={"shared_bidirectional": True}),
    )
    components = tuple(
        replace(component, metadata={**component.metadata, "resident_access_path": "topology"})
        if component.component_id == "hbm0" else component
        for component in scenario.hardware.components
    ) + (bridge,)
    scenario = replace(scenario, hardware=replace(
        scenario.hardware, links=links, components=components,
        metadata={**scenario.hardware.metadata, "physical_resource_owners": {
            "link.memory-bridge": "shared.phy", "link.bridge-gpu": "shared.phy",
        }},
    ))
    with _compilation_scope(scenario):
        resource = _rank_memory_resource(scenario, _parallel_plan(scenario).ranks[0])
        demands, metadata = _route_resident_memory_demands(
            (ResourceDemand(resource, 1.0, bytes_moved=256),),
            {"cache": {"physical_read_bytes": 256, "physical_write_bytes": 0}},
        )
    owners = declared_resource_owners(scenario.hardware)
    task = TaskSpec("t", "r", "resident", TaskCategory.MEMORY,
                    demands=demands, metadata=metadata)
    kernel = UnifiedEventKernel.from_closed_graph((task,), resource_owners=owners)
    kernel.step()
    phy_demands = [row for row in demands if owners.get(row.resource_id, row.resource_id) == "shared.phy"]
    assert len(phy_demands) == 1
    assert phy_demands[0].bytes_moved == 512  # Two distinct traversals of one physical PHY.
    assert phy_demands[0].service_ns == 2 * 256 * 8 / link.bandwidth_gbps


def test_resident_router_keeps_compute_identity_for_shared_rank_memory():
    scenario = build_reference_scenario()
    components = scenario.hardware.component_map()
    components["hbm0"] = replace(
        components["hbm0"], metadata={"resident_access_path": "topology"},
        ports=tuple(replace(port, max_links=2) for port in components["hbm0"].ports),
    )
    components["gpu1"] = replace(components["gpu0"], component_id="gpu1")
    link = next(row for row in scenario.hardware.links if row.link_id == "gpu-hbm0")
    first = scenario.placement.parallel.rank_mapping[0]
    scenario = replace(
        scenario,
        hardware=replace(scenario.hardware, components=tuple(components.values()),
                         links=scenario.hardware.links + (
                             replace(link, link_id="gpu1-hbm0", source_component="gpu1"),)),
        placement=replace(scenario.placement, parallel=replace(
            scenario.placement.parallel, tp_degree=2, rank_mapping=(
                first, replace(first, rank=1, component_id="gpu1", tp_rank=1),
            ),
        )),
    )
    with _compilation_scope(scenario):
        ranks = _parallel_plan(scenario).ranks
        assert _rank_memory_resource(scenario, ranks[0]) == _rank_memory_resource(scenario, ranks[1])
        for rank in ranks:
            _, metadata = _route_resident_memory_demands(
                (ResourceDemand(_rank_memory_resource(scenario, rank), 1.0, bytes_moved=256),),
                {"rank": rank.rank, "target_component": rank.component_id,
                 "cache": {"physical_read_bytes": 256, "physical_write_bytes": 0}},
            )
            assert metadata["resident_memory_paths"][0]["compute_component"] == rank.component_id


# CIM arithmetic tests: independent of resident-route regression ownership.
def _fp16_cim_profile(**overrides):
    from heterollm_sim.cost_models import DigitalSramCimProfile
    values = dict(arithmetic_mode="fp16_fp32_analytical", float_cycles_per_eval=7.0,
                  float_accumulator_outputs_per_cycle=2.0,
                  float_contract_basis="synthetic FP16 tile / FP32 reduction hardware assumption",
                  array_count=1, p_m=1, p_k=4, p_n=4, frequency_ghz=1.0)
    values.update(overrides)
    return DigitalSramCimProfile(**values)


def test_cim_fp16_uses_float_tile_cost_not_integer_bit_slices():
    from heterollm_sim.cost_models import estimate_cim_gemm
    workload = GemmWorkload(m=2, k=8, n=4, activation_bits=16, weight_bits=16,
                            accumulator_bits=32, cim_arithmetic="fp16")
    estimate = estimate_cim_gemm(_fp16_cim_profile(), workload)
    assert estimate.metadata["array_cycles"] == 4 * 7.0
    assert estimate.metadata["bit_slice_count"] == 1
    assert estimate.phase("cim_accumulate_reduce").metadata["accumulator_service_ns"] == 4.0
    assert estimate.metadata["resident_weight_bytes"] == 64
    assert estimate.metadata["transfer_weight_bytes"] == 64
    audit = estimate.metadata["arithmetic_contract"]
    assert audit["accumulation_model"] == "fp32_rounded_partial_sums"
    assert audit["evidence"] == "analytical"
    assert not audit["numerical_equivalence_verified"] and not audit["calibrated"]
    assert not audit["bit_slice_model_applied"]
    assert all(phase.metadata["arithmetic_contract"] == audit for phase in estimate.phases)
    changed = estimate_cim_gemm(replace(_fp16_cim_profile(), input_parallel_bits=16,
                                       weight_parallel_bits=16, cycles_per_eval=99,
                                       accumulator_outputs_per_cycle=999), workload)
    assert changed.service_ns == estimate.service_ns


def test_cim_float_mode_requires_explicit_complete_hardware_assumptions():
    from heterollm_sim.cost_models import DigitalSramCimProfile, estimate_cim_gemm
    for overrides in ({"float_cycles_per_eval": None}, {"float_cycles_per_eval": 0},
                      {"float_accumulator_outputs_per_cycle": None},
                      {"float_contract_basis": ""}, {"accumulator_bits": 48}):
        with pytest.raises(ValueError):
            _fp16_cim_profile(**overrides)
    workload = GemmWorkload(m=1, k=8, n=4, activation_bits=16, weight_bits=16,
                            accumulator_bits=32, cim_arithmetic="fp16")
    with pytest.raises(ValueError, match="arithmetic contract"):
        estimate_cim_gemm(DigitalSramCimProfile(accumulator_bits=64), workload)
    with pytest.raises(ValueError, match="arithmetic contract"):
        estimate_cim_gemm(_fp16_cim_profile(), replace(workload, cim_arithmetic="integer"))
    with pytest.raises(ValueError, match="smaller than required"):
        estimate_cim_gemm(DigitalSramCimProfile(), replace(workload, cim_arithmetic="integer"))


@pytest.mark.parametrize("fmt", ["iq3_s", "iq4_xs", "q4_0", "q8_0"])
@pytest.mark.parametrize("resident", [False, True])
def test_cim_rejects_packed_weights_even_if_resident(fmt, resident):
    from heterollm_sim.cost_models import DigitalSramCimProfile, estimate_cim_gemm
    workload = GemmWorkload(m=1, k=8, n=4, packed_weight_formats=(fmt,))
    with pytest.raises(ValueError, match="packed weights"):
        estimate_cim_gemm(DigitalSramCimProfile(), workload, weights_resident=resident)
    with pytest.raises(ValueError, match="packed weights"):
        estimate_cim_gemm(_fp16_cim_profile(), replace(workload, activation_bits=16,
                          weight_bits=16, cim_arithmetic="fp16"), weights_resident=resident)


def test_cim_fp16_dense_storage_is_verified_not_invented():
    from heterollm_sim.cost_models import estimate_cim_gemm
    workload = GemmWorkload(m=1, k=8, n=4, activation_bits=16, weight_bits=16,
                            packed_weight_formats=("F16",), cim_arithmetic="fp16")
    assert estimate_cim_gemm(_fp16_cim_profile(), workload).metadata["transfer_weight_bytes"] == 64
    for overrides in ({"weight_storage_bytes": 16}, {"activation_storage_bytes": 32},
                      {"output_storage_bytes": 4}):
        with pytest.raises(ValueError, match="materialized dense FP16"):
            estimate_cim_gemm(_fp16_cim_profile(), replace(workload, **overrides))
    with pytest.raises(ValueError, match="FP16 operands"):
        estimate_cim_gemm(_fp16_cim_profile(), replace(workload, activation_bits=32))
    with pytest.raises(ValueError, match="capacity"):
        estimate_cim_gemm(_fp16_cim_profile(weight_capacity_bytes=32), workload)
