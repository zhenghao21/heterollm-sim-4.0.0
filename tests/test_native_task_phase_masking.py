"""Only a native independent service can hide an entire REF-on HBM branch."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from heterollm_sim.contracts import (_PreparedExecutionStage, _PreparedExecutionTask,
    ResourceDemand, TaskSpec)
from heterollm_sim.cost_models import (CacheHierarchyProfile, CacheLevelProfile,
    GPUProfile, TensorCoreProfile)
from heterollm_sim.native_window_producer import summarize_native_stages
from heterollm_sim.native_layer_sampling import _independent_kernel
from test_native_refresh_source import source


def _task(config, identity="body", *, dependencies=(), service=20000., stream=False):
    md = {"rank": 0, "physical_owner": "hbm", "physical_memory_config": config,
        "memory_accesses": ({"physical_owner": "hbm", "resource_id": "hbm",
            "operation": "read", "address": 0, "byte_count": 32768},)}
    resource = "gpu.stream" if stream else "gpu.tensor"
    if stream:
        md["phase_metadata"] = {"kernel_model": {"physical_memory_envelope": {
            "resource_suffix": ".stream", "dependency_ns": 0.}}}
    return _PreparedExecutionTask(identity, tuple(dependencies), ("r",),
        (ResourceDemand(resource, service, work_units=1.),), metadata=md)


def _summary(data, tasks, *, compute_resource="gpu.tensor"):
    runtime, _config, physical, _core = data
    stage = _PreparedExecutionStage("body", 0, (), ("r",), "gpu", 0., tuple(tasks))
    gpu = GPUProfile(TensorCoreProfile(1, 1, 1., resource_id=compute_resource),
        CacheHierarchyProfile((CacheLevelProfile("L2", 128, 32, 1., 100.),)),
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1.,
        reduction_ops_per_cycle_per_sm=1., special_function_units_per_sm=1,
        special_function_ops_per_cycle=1.)
    return summarize_native_stages((stage,), NS(component_profiles={"gpu": {"gpu": gpu}}), physical,
        allow_bounded_clock=True, native_runtime=runtime)


def _events(data, tasks):
    runtime, _config, physical, _core = data
    kernel, _executor = _independent_kernel(runtime._execution_resource_kernel)
    origin = max(runtime.now, kernel.makespan_ns,
        max(physical.timeline.ready_ns.values(), default=0.))
    kernel.submit(tuple(TaskSpec(task.task_id, "r", task.task_id, task.category,
        task.dependencies, task.demands, earliest_start_ns=origin,
        metadata=task.metadata) for task in tasks))
    events = []
    while (event := kernel.step()) is not None:
        events.append(event)
    return origin, events


@pytest.mark.parametrize("arrival", (0., 3899.5, 3900., 1e8+.17))
def test_real_native_HBM_is_covered_without_advancing_source_or_removing_REF(arrival):
    data = source(arrival=arrival)
    task = _task(data[1])
    packed = data[3]._hbm_numeric_state
    before = tuple(array.copy() for array in packed.arrays)
    summary = _summary(data, (task,))
    proof, = summary["atomic_task_phase_mask_proofs"]
    assert proof["qualified"], proof
    assert proof["complete_HBM_service_upper_ns"] < proof["nonphysical_service_lower_ns"]
    assert proof["descriptor_bounds"][0]["residual_and_feedback_REF_upper_ns"] > 0.
    assert proof["unpaid_source_carry_upper_ns"] == 0.
    assert summary["ordered_resource_phase_per_cohort_ns"] == 0.
    assert summary["row_phase_per_cohort_ns"] > 0.  # Original physical cost is retained.
    assert summary["hbm_clock_domain_limit_ns"] <= proof["hbm_clock_domain_limit_ns"]
    assert data[3]._hbm_numeric_state is packed
    assert all(np.array_equal(old, new) for old, new in zip(before, packed.arrays))
    origin, events = _events(data, (task,))
    assert len(events) == 1
    event = events[0]
    assert event.end_ns == origin+task.demands[0].service_ns
    bill = event.task.metadata["physical_execution_by_owner"]["hbm"]
    assert bill["completion_ns"]-bill["arrival_ns"] <= proof["complete_HBM_service_upper_ns"]
    assert bill["physical_read_bytes"] > 0


def test_same_endpoint_does_not_hide_an_unordered_HBM_branch():
    data = source(arrival=3899.5)
    parent = _task(data[1], "a")
    branch = _task(data[1], "b", service=0.)
    branch = replace(branch, demands=(ResourceDemand("gpu.branch", 0.),))
    summary = _summary(data, (parent, branch))
    proof = next(row for row in summary["atomic_task_phase_mask_proofs"] if row["task_id"] == "a")
    assert not proof["qualified"]
    assert proof["reason"] == "native_HBM_owner_has_unordered_Task_branch"
    assert summary["ordered_resource_phase_per_cohort_ns"] > 0.
    origin, events = _events(data, (parent, branch))
    actual = {event.task.task_id: event for event in events}
    assert actual["a"].end_ns == origin+20000.
    assert actual["b"].end_ns < actual["a"].end_ns


def test_ordered_successor_keeps_its_own_controller_phase():
    data = source()
    parent = _task(data[1], "a")
    child = _task(data[1], "b", dependencies=("a",), service=0.)
    summary = _summary(data, (parent, child))
    proof = {row["task_id"]: row for row in summary["atomic_task_phase_mask_proofs"]}
    assert proof["a"]["qualified"]
    assert not proof["b"]["qualified"]
    assert summary["ordered_resource_phase_per_cohort_ns"] > 0.
    assert proof["a"]["later_controller_state_guards_retained"]
    _origin, events = _events(data, (parent, child))
    assert events[1].start_ns >= events[0].end_ns


@pytest.mark.parametrize("alias", (False, True))
@pytest.mark.parametrize("ordered", (False, True))
def test_ordinary_direct_HBM_port_writer_is_not_hidden_by_descriptor_inventory(alias, ordered):
    data = source()
    resource = "gpu.direct_port" if alias else "hbm:dram:data:0"
    if alias:
        data[2].timeline.resource_aliases[resource] = "hbm:dram:data:0"
    writer = _PreparedExecutionTask("aaa", (), ("r",),
        (ResourceDemand(resource, 25000., work_units=1.),), metadata={"rank": 0})
    masked = _task(data[1], "mask", dependencies=("aaa",) if ordered else ())
    summary = _summary(data, (writer, masked))
    proof, = summary["atomic_task_phase_mask_proofs"]
    assert not proof["qualified"], proof
    assert proof["reason"] == "native_HBM_owner_has_unmodelled_direct_port_writer"
    assert summary["ordered_resource_phase_per_cohort_ns"] > 0.
    origin, events = _events(data, (writer, masked))
    actual = {event.task.task_id: event for event in events}
    assert actual["aaa"].end_ns == origin+25000.
    if not ordered and not alias:
        assert actual["mask"].end_ns > origin+masked.demands[0].service_ns


@pytest.mark.parametrize("identity", ("resource_id", "physical_owner", "staging"))
@pytest.mark.parametrize("alias", (False, True))
def test_native_placeholder_or_alias_cannot_be_used_as_compute_lower(identity, alias):
    data = source()
    replaced = "hbm.preview" if identity == "resource_id" else "hbm"
    demand_resource = "gpu.tensor_alias" if alias else replaced
    if alias:
        data[2].timeline.resource_aliases[demand_resource] = replaced
    task = _task(data[1])
    md = {**task.metadata, "memory_accesses": ({
        **task.metadata["memory_accesses"][0], "resource_id": "hbm.preview"},)}
    if identity == "staging":
        replaced = "link.replaced"
        demand_resource = "gpu.tensor_alias" if alias else replaced
        if alias:
            data[2].timeline.resource_aliases[demand_resource] = replaced
        md["hbf_staging_replaced_link_ids"] = (replaced,)
    task = replace(task, demands=(ResourceDemand(demand_resource, 20000., work_units=1.),),
        metadata=md)
    summary = _summary(data, (task,), compute_resource=demand_resource)
    proof, = summary["atomic_task_phase_mask_proofs"]
    assert not proof["qualified"], proof
    assert proof["reason"] == "native_independent_compute_or_link_lower_unavailable"
    assert summary["ordered_resource_phase_per_cohort_ns"] > 0.
    origin, events = _events(data, (task,))
    assert events[0].end_ns < origin+20000.
    assert demand_resource not in events[0].task.metadata["physical_nonmemory_demand_ids"]


def test_unknown_positive_work_units_are_not_native_compute_provenance():
    data = source()
    task = replace(_task(data[1]), demands=(ResourceDemand("unknown.work", 20000., work_units=1.),))
    proof, = _summary(data, (task,))["atomic_task_phase_mask_proofs"]
    assert not proof["qualified"]
    assert proof["reason"] == "native_independent_compute_or_link_lower_unavailable"


@pytest.mark.parametrize("alias", ("none", "incoming", "outgoing"))
def test_NAND_host_link_is_a_replaced_physical_demand_not_a_compute_lower(alias):
    from heterollm_sim.memory_types import NandConfig

    data = source()
    config = NandConfig(kind="HBF", metadata={"host_resource_id": "link.nand_host"})
    data[2].runtime(config, "nand")
    task = _task(data[1])
    resource = "link.nand_host" if alias == "none" else "gpu.tensor"
    if alias == "incoming":
        data[2].timeline.resource_aliases[resource] = "link.nand_host"
    elif alias == "outgoing":
        data[2].timeline.resource_aliases["link.nand_host"] = resource
    task = replace(task, demands=(ResourceDemand(resource, 1e9, work_units=1.),), metadata={
        **task.metadata, "physical_memory_configs": {"hbm": data[1], "nand": config},
        "memory_accesses": (*task.metadata["memory_accesses"], {
            "physical_owner": "nand", "resource_id": "nand", "operation": "read",
            "address": 0, "byte_count": 4096})})
    summary = _summary(data, (task,), compute_resource=resource)
    proof, = summary["atomic_task_phase_mask_proofs"]
    assert not proof["qualified"]
    assert proof["reason"] == ("native_independent_lower_does_not_cover_complete_HBM_upper"
        if alias == "none" else "native_independent_compute_or_link_lower_unavailable")
    assert summary["ordered_resource_phase_per_cohort_ns"] > 0.
    origin, events = _events(data, (task,))
    assert events[0].end_ns < origin+1e9
    assert resource not in events[0].task.metadata["physical_nonmemory_demand_ids"]


def _with_native_NAND_payload(data, *, address=0, count=65536, **config_values):
    from heterollm_sim.memory_types import NandConfig

    config = NandConfig(kind="HBF", page_bytes=4096, channels=2, host_channels=2,
        read_cache_pages_per_bank=16, **config_values)
    core = data[2].runtime(config, "nand").core
    task = _task(data[1])
    task = replace(task, demands=(ResourceDemand("nand.preview", 1e9, work_units=1.),),
        metadata={**task.metadata,
            "physical_memory_configs": {"hbm": data[1], "nand": config},
            "memory_accesses": (*task.metadata["memory_accesses"], {
                "physical_owner": "nand", "resource_id": "nand.preview", "operation": "read",
                "address": address, "byte_count": count})})
    return task, core


@pytest.mark.parametrize("warm", (False, True))
@pytest.mark.parametrize("address,count", ((0, 65536), (23, 5121)))
def test_actual_NAND_payload_covers_HBM_including_cache_hit_and_partial_pages(warm, address, count):
    from heterollm_sim.memory_types import AccessRequest

    data = source()
    task, nand = _with_native_NAND_payload(data, address=address, count=count)
    if warm:
        nand.submit(AccessRequest("warm", "read", address, count))
    summary = _summary(data, (task,))
    proof, = summary["atomic_task_phase_mask_proofs"]
    assert proof["qualified"], proof
    assert proof["nonphysical_service_lower_ns"] == 0.
    payload, = proof["native_NAND_payload_lowers"]
    assert payload["mandatory_payload_lower_ns"] < count/nand.config.host_bandwidth_for("read")
    assert payload["floating_rounding_reserve_ns"] > 0.
    assert proof["independent_service_lower_ns"] >= proof["complete_HBM_service_upper_ns"]
    origin, events = _events(data, (task,))
    event, = events
    assert event.end_ns >= origin+proof["independent_service_lower_ns"]
    bill = event.task.metadata["physical_execution_by_owner"]
    assert bill["nand"]["host_transfer_bytes"] >= count
    assert bill["hbm"]["physical_read_bytes"] > 0
    assert "nand.preview" not in event.task.metadata["physical_nonmemory_demand_ids"]
    if warm:
        assert bill["nand"]["read_cache_hits"] > 0


@pytest.mark.parametrize("kind", ("shared", "alias", "lanes", "buffer", "staging"))
def test_NAND_payload_does_not_mask_unknown_or_shared_physical_paths(kind):
    data = source()
    values = {"metadata": {"host_resource_id": "hbm:dram:data:0"}} if kind == "shared" else {}
    if kind == "buffer":
        values["write_buffer_bytes"] = 8192
    task, _nand = _with_native_NAND_payload(data, **values)
    if kind == "alias":
        data[2].timeline.resource_aliases["nand:nand:host:0"] = "foreign.host"
    elif kind == "lanes":
        data[0]._execution_resource_kernel.ensure_resource_capacities({"nand:nand:host:0": 2})
    elif kind == "staging":
        task = replace(task, metadata={**task.metadata, "hbf_host_staging_by_owner": {"nand": {"active": True}}})
    proof, = _summary(data, (task,))["atomic_task_phase_mask_proofs"]
    assert not proof["qualified"], proof


@pytest.mark.parametrize("kind", ("short", "stream", "queue", "stateful", "alias", "external", "clock"))
def test_unknown_or_uncovered_native_branch_keeps_all_phase(kind):
    data = source()
    runtime, config, physical, _core = data
    task = _task(config, service=1. if kind == "short" else 20000., stream=kind == "stream")
    if kind == "queue":
        runtime._execution_resource_kernel.resource_available["gpu.tensor"] = 1e6
    elif kind == "stateful":
        task = replace(task, metadata={**task.metadata, "stateful_l2": {"enabled": True}})
    elif kind == "alias":
        physical.timeline.resource_aliases["gpu.tensor"] = "hbm:dram:data:0"
    elif kind == "external":
        runtime._native_window_owns_execution_kernel = False
    elif kind == "clock":
        runtime.now = 1e15
    summary = _summary(data, (task,))
    proof, = summary["atomic_task_phase_mask_proofs"]
    assert not proof["qualified"], proof
    assert summary["ordered_resource_phase_per_cohort_ns"] > 0.

