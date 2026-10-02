"""Explicit GPU buffer exposure must use its physical memory without staging."""
from dataclasses import replace

from heterollm_sim import planner
from heterollm_sim.contracts import ResourceDemand, TaskCategory
from heterollm_sim.cost_models import CostPhase, GemmWorkload, HostMemoryProfile
from heterollm_sim.reference import build_reference_scenario


def _case(bandwidth=10.0, latency=100.0, direct=True):
    base = build_reference_scenario()
    memory = replace(base.hardware.get_component("hbm1"), kind="hbf", cost_profile_id="direct-hbf",
        bandwidth_gbps=bandwidth * 8, read_bandwidth_gbps=bandwidth * 8,
        write_bandwidth_gbps=bandwidth * 8,
        metadata={"access_mode": "memory", "write_buffer_bytes": 0, "writable": True, "memory_service_owner": "hbf.memory", "transfer_granularity_bytes": 256, "max_outstanding_requests": 32, "read_latency_ns": latency, "write_latency_ns": latency})
    profile = HostMemoryProfile(bandwidth, read_latency_ns=latency, write_latency_ns=latency, transaction_bytes=256, max_outstanding_requests=32,
                                resource_id="hbf.memory")
    return replace(base,
        hardware=replace(base.hardware, components=tuple(
            memory if c.component_id == memory.component_id else c for c in base.hardware.components)),
        component_profiles={**base.component_profiles, "host_memory": {
            **base.component_profiles["host_memory"], "direct-hbf": profile}},
        placement=replace(base.placement, tensor_to_component={"model_weights": "hbm1"},
            metadata={"llama_backend_memory": {
                "hbm1": {"device_id": "gpu0", "access": "direct"}}} if direct else {}))


def _nand_case():
    case = _case()
    component = case.hardware.get_component("hbm1")
    nand_media = {
        "version": "nand_media_v1",
        "host_transaction_bytes": 64,
        "host_max_request_bytes": 4096,
        "media_page_bytes": 8192,
        "command_queue_depth": 4,
        "media_parallelism": 1,
        "page_read_latency_ns": 1000.0,
        "page_program_latency_ns": 2000.0,
        "physical_planes": 1,
        "physical_dies": 2,
        "physical_channels": 1,
        "access_pattern": "contiguous_page_aligned",
    }
    component = replace(component, metadata={**component.metadata, "nand_media": nand_media})
    return replace(case, hardware=replace(case.hardware, components=tuple(
        component if item.component_id == "hbm1" else item
        for item in case.hardware.components
    )))


def _gemm_tasks(case, request_id="r"):
    workload = GemmWorkload(m=1, k=1024, n=1024, activation_bits=16, weight_bits=16,
                            accumulator_bits=32)
    with planner._compilation_scope(case):
        plan = planner._parallel_plan(case)
        builder = planner._TaskBuilder(replace(case.workload.requests[0], request_id=request_id))
        planner._add_rank_gemm(builder, case, planner._topology_router(case), plan, plan.ranks[0],
            workload, "gpu0", "probe", ())
    return builder.tasks, workload


def test_weight_direct_read_moves_only_weight_service_and_never_copies():
    tasks, workload = _gemm_tasks(_case())
    matrix = next(t for t in tasks if t.metadata.get("phase") == "gpu_gemm")
    audit = matrix.metadata["phase_metadata"]["direct_memory_access"]
    assert audit["read_bytes"] == workload.weight_bytes
    assert audit["local_read_bytes"] == workload.activation_bytes
    assert audit["local_write_bytes"] == workload.output_bytes
    direct = next(d for d in matrix.demands if d.resource_id == audit["resource_id"])
    assert direct.bytes_moved == workload.weight_bytes
    assert not any(t.metadata.get("access_kind") == "COPY" for t in tasks)
    assert not any(t.metadata.get("event_kind") == "model_weight_read" for t in tasks)


def test_bandwidth_latency_and_shared_physical_owner_control_weight_service():
    fast, _ = _gemm_tasks(_case(100., 1.))
    slow, _ = _gemm_tasks(_case(1., 1000.), "other-request")
    def access(tasks):
        return next(t for t in tasks if t.metadata.get("phase") == "gpu_gemm")
    fast_task, slow_task = access(fast), access(slow)
    owner = fast_task.metadata["phase_metadata"]["direct_memory_access"]["resource_id"]
    assert slow_task.metadata["phase_metadata"]["direct_memory_access"]["resource_id"] == owner
    assert next(d.service_ns for d in slow_task.demands if d.resource_id == owner) > next(
        d.service_ns for d in fast_task.demands if d.resource_id == owner)


def test_direct_exposure_is_required_and_hbm_baseline_stays_local():
    remote, _ = _gemm_tasks(_case(direct=False))
    assert any(t.metadata.get("event_kind") == "model_weight_read" for t in remote)
    base = build_reference_scenario()
    base = replace(base, placement=replace(base.placement, tensor_to_component={"model_weights": "hbm0"}))
    normal, _ = _gemm_tasks(base)
    exposed = replace(base, placement=replace(base.placement, metadata={"llama_backend_memory": {
        "hbm0": {"device_id": "gpu0", "access": "direct"}}}))
    direct, _ = _gemm_tasks(exposed)
    assert [(t.name, t.demands) for t in normal] == [(t.name, t.demands) for t in direct]


def test_direct_kv_read_has_hbf_service_without_hbm_scratch():
    case = _case()
    with planner._compilation_scope(case):
        plan = planner._parallel_plan(case)
        builder = planner._TaskBuilder(case.workload.requests[0])
        task_id = planner._add_kv_access(
            builder, case, planner._topology_router(case), plan, plan.ranks[0],
            "hbm1", "gpu0", 4096, (), name="kv", metadata={"memory_direction": "read"})
    task = builder.tasks[-1]
    assert task_id == task.task_id
    assert any(d.resource_id == "hbf.memory" and d.bytes_moved == 4096 for d in task.demands)
    assert not any(d.resource_id == "gpu0.hbm" and d.bytes_moved == 4096 for d in task.demands)


def test_direct_nand_read_uses_page_geometry_and_media_latency():
    tasks, workload = _gemm_tasks(_nand_case())
    matrix = next(t for t in tasks if t.metadata.get("phase") == "gpu_gemm")
    audit = matrix.metadata["phase_metadata"]["direct_memory_access"]
    bill = audit["memory_service"]
    assert bill["operation"] == "read"
    assert bill["media_page_bytes"] == 8192
    assert bill["pages_touched"] == workload.weight_bytes // 8192
    assert bill["page_read_latency_ns"] == 1000.0
    assert bill["media_waves"] == workload.weight_bytes // 8192
    direct = next(d for d in matrix.demands if d.resource_id == audit["resource_id"])
    assert direct.service_ns == bill["service_ns"]
    assert direct.bytes_moved == bill["physical_bytes"]
    assert direct.service_ns > 600000.0


def test_direct_nand_program_uses_media_rmw_and_physical_bytes():
    case = _nand_case()
    with planner._compilation_scope(case):
        plan = planner._parallel_plan(case)
        rank = plan.ranks[0]
        _, local = planner._gpu_profiles(case, rank.component_id, rank.memory_component_id)
        phase = CostPhase(
            "direct_write",
            TaskCategory.MEMORY,
            (ResourceDemand(local.resource_id, 1.0, bytes_moved=4096),),
            metadata={"cache": {"physical_read_bytes": 0, "physical_write_bytes": 4096}},
        )
        updated = planner._direct_memory_phase(
            case, rank, phase, "hbm1", read_bytes=0, write_bytes=4096
        )
    bill = updated.metadata["direct_memory_access"]["memory_service"]
    assert bill["operation"] == "program"
    assert bill["rmw_read_operations"] == 1
    assert bill["physical_write_bytes"] == 8192
    assert updated.metadata["direct_memory_access"]["physical_bytes"] == bill["physical_bytes"]
