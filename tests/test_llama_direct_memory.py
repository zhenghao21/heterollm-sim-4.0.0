"""Explicit GPU buffer exposure must use its physical memory without staging."""
from dataclasses import replace

from heterollm_sim import planner
from heterollm_sim.cost_models import GemmWorkload, HostMemoryProfile
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
