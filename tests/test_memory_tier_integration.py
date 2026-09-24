"""Regression checks for physical memory ownership and routed kernel accesses."""
from dataclasses import replace

import pytest

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.cost_models import HostMemoryProfile
from heterollm_sim.engine import simulate_schedule
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.planner import compile_scenario, _gpu_profiles
from heterollm_sim.reference import build_reference_scenario


def test_explicit_dram_profile_is_not_replaced_with_nearby_hbm():
    base = build_reference_scenario()
    memory = HostMemoryProfile(12.0, read_latency_ns=500.0, write_latency_ns=700.0,
                               transaction_bytes=4096, max_outstanding_requests=2,
                               resource_id="dram0.memory")
    hardware = replace(base.hardware, components=base.hardware.components + (
        ComponentSpec("dram0", "dram", capacity_bytes=1 << 30, cost_profile_id="dram-test"),
    ))
    profiles = {**base.component_profiles, "host_memory": {
        **base.component_profiles["host_memory"], "dram-test": memory,
    }}
    scenario = replace(base, hardware=hardware, component_profiles=profiles)
    _, resolved = _gpu_profiles(scenario, "gpu0", "dram0")
    assert resolved.bandwidth_gb_s == 12.0
    assert resolved.read_latency_ns == 500.0
    assert resolved.write_latency_ns == 700.0
    assert resolved.transaction_bytes == 4096
    assert resolved.memory_service(8192) == memory.memory_service(8192)


def _routed_reference(latency_ns):
    base = build_reference_scenario()
    components = tuple(replace(c, metadata={**c.metadata, "resident_access_path": "topology"})
                       if c.component_id == "hbm0" else c for c in base.hardware.components)
    links = tuple(replace(l, latency_ns=latency_ns) if l.link_id == "gpu-hbm0" else l
                  for l in base.hardware.links)
    return replace(base, hardware=replace(base.hardware, components=components, links=links))


def test_local_kernel_traffic_reserves_its_physical_link_and_latency_matters():
    fast = compile_scenario(_routed_reference(0.0))
    slow = compile_scenario(_routed_reference(1e6))
    routed = [task for task in fast.tasks if task.metadata.get("resident_memory_paths")]
    assert routed
    for task in routed:
        for access in task.metadata["resident_memory_paths"]:
            assert access["read_bytes"] + access["write_bytes"] == access["logical_bytes"]
            for hop in access["hops"]:
                assert any(d.resource_id == hop["resource_id"] and d.bytes_moved >= hop["bytes"]
                           for d in task.demands)
    assert simulate_schedule(slow).makespan_ns > simulate_schedule(fast).makespan_ns
    assert not any(t.metadata.get("resident_memory_paths")
                   for t in compile_scenario(build_reference_scenario()).tasks)


def test_unused_memory_path_does_not_add_any_kernel_work():
    base = build_reference_scenario()
    unused = replace(base, hardware=replace(base.hardware, components=tuple(
        replace(c, metadata={"resident_access_path": "topology"}) if c.component_id == "hbm7" else c
        for c in base.hardware.components)))
    original, changed = compile_scenario(base), compile_scenario(unused)
    assert original.tasks == changed.tasks


def test_flash_endpoint_retains_page_rounding_under_overlapped_service():
    flash = ComponentSpec("flash", "hbf", capacity_bytes=1 << 30,
        read_bandwidth_gbps=800.0, write_bandwidth_gbps=400.0,
        metadata={"read_latency_ns": 1000.0, "write_latency_ns": 2000.0,
                  "transfer_granularity_bytes": 4096, "max_outstanding_requests": 2})
    serial = TopologyRouter._endpoint_phase(flash, 4097, read=True, name="read")
    pipelined = TopologyRouter._endpoint_phase(replace(flash, metadata={
        **flash.metadata, "memory_service_model": "overlapped"}), 4097, read=True, name="read")
    assert serial.metadata["transferred_bytes"] == pipelined.metadata["transferred_bytes"] == 8192
    assert pipelined.demands[0].service_ns == 1000.0
    assert serial.demands[0].service_ns == 1000.0 + 81.92
    assert TopologyRouter._endpoint_phase(flash, 0, read=True, name="read").demands[0].service_ns == 0
    with pytest.raises(ValueError, match="memory_service_model"):
        TopologyRouter._endpoint_phase(replace(flash, metadata={"memory_service_model": "fake"}),
                                      10, read=True, name="read")


def test_shared_memory_routes_are_bound_to_the_consuming_rank():
    from heterollm_sim.ir import RankMappingSpec, ParallelSpec, PortSpec, LinkSpec
    from heterollm_sim.contracts import ResourceDemand
    from heterollm_sim.planner import _compilation_scope, _rank_memory_resource, _parallel_plan, _route_resident_memory_demands
    base = _routed_reference(0)
    gpu = base.hardware.get_component("gpu0")
    memory = base.hardware.get_component("hbm0")
    hardware = replace(base.hardware, components=tuple(
        replace(c, ports=c.ports + (PortSpec("gpu1", "HBM", "device", bandwidth_gbps=100),))
        if c.component_id == "hbm0" else c for c in base.hardware.components
    ) + (replace(gpu, component_id="gpu1"),), links=base.hardware.links + (
        LinkSpec("gpu1-hbm0", "gpu1", "hbm0", "hbm0", "gpu1", "HBM", bandwidth_gbps=100),))
    ranks = tuple(RankMappingSpec(i, "gpu" + str(i), i, 0, 0, memory_component_id="hbm0") for i in range(2))
    scenario = replace(base, hardware=hardware, placement=replace(base.placement,
        parallel=ParallelSpec(tp_degree=2, rank_mapping=ranks)))
    with _compilation_scope(scenario):
        plan = _parallel_plan(scenario)
        for rank in plan.ranks:
            demands, meta = _route_resident_memory_demands(
                (ResourceDemand(_rank_memory_resource(scenario, rank), 10, bytes_moved=100),),
                {"rank": rank.rank})
            assert meta["resident_memory_paths"][0]["compute_component"] == rank.component_id
            expected = "gpu-hbm0" if rank.rank == 0 else "gpu1-hbm0"
            assert meta["resident_memory_paths"][0]["hops"][0]["link_id"] == expected


def test_cim_planner_preserves_explicit_fp16_arithmetic_and_audit():
    from heterollm_sim.cost_models import GemmWorkload
    from heterollm_sim.planner import _TaskBuilder, _add_rank_gemm, _parallel_plan, _compilation_scope
    from tests.test_memory_profiles import _fp16_cim_profile
    base = build_reference_scenario()
    profiles = {**base.component_profiles, "cim": {**base.component_profiles["cim"],
                "legacy-cim": _fp16_cim_profile()}}
    scenario = replace(base, component_profiles=profiles)
    workload = GemmWorkload(m=1, k=8, n=4, activation_bits=16, weight_bits=16,
                            accumulator_bits=32, cim_arithmetic="fp16")
    with _compilation_scope(scenario):
        plan = _parallel_plan(scenario)
        builder = _TaskBuilder(scenario.workload.requests[0])
        _add_rank_gemm(builder, scenario, TopologyRouter(scenario.hardware), plan, plan.ranks[0],
                       workload, "cim0", "cim.contract.probe", ())
    cim_tasks = [task for task in builder.tasks
                 if task.metadata.get("phase_metadata", {}).get("arithmetic_contract")]
    assert cim_tasks
    for task in cim_tasks:
        audit = task.metadata["phase_metadata"]["arithmetic_contract"]
        assert audit["operand_arithmetic"] == "fp16"
        assert not audit["numerical_equivalence_verified"]
