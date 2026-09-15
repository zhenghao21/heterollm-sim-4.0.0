"""Mechanism checks for explicit sharing, finite transfers and lifecycle."""

from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from heterollm_sim.communication import (
    TransferPhase, TopologyRouter, declared_resource_owners, plan_transfer_pipeline,
)
from heterollm_sim.contracts import (
    ResourceDemand, RunManifest, TaskCategory, TaskSpec, TraceMarker,
)
from heterollm_sim.engine import ScheduleIR, simulate_schedule
from heterollm_sim.event_kernel import CompiledGraphLayout, UnifiedEventKernel
from heterollm_sim.invocation import InvocationLifecycle, InvocationOverheads
from heterollm_sim.ir import HardwareSpec, ComponentSpec, LinkSpec, PortSpec
from heterollm_sim.metrics import summarize_metrics
from heterollm_sim.planner import RequestTaskChunk
from heterollm_sim.streaming_des import execute_incremental_schedule


def task(name, *demands, dependencies=(), category=TaskCategory.COMPUTE, marker=None):
    return TaskSpec(name, "req", name, category, dependencies=dependencies,
                    demands=tuple(demands), marker=marker)


def drain(kernel):
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
    return {event.task.task_id: event for event in events}


def manifest():
    return RunManifest(schema_version="4.0.0", run_id="owners", random_seed=1,
                       simulator_version="test", model_name="model",
                       hardware_name="hardware", workload_name="workload")


OWNERS = {"hbm.compute": "hbm.physical", "component.vram.write": "hbm.physical",
          "component.vram.read": "hbm.physical"}


def test_compute_and_copy_share_only_the_declared_memory_owner():
    tasks = (
        task("0.compute", ResourceDemand("tensor", 30), ResourceDemand("hbm.compute", 10)),
        task("1.copy", ResourceDemand("dma", 15), ResourceDemand("component.vram.write", 8)),
        task("2.scalar", ResourceDemand("scalar", 7)),
    )
    kernel = UnifiedEventKernel.from_closed_graph(tasks, resource_owners=OWNERS)
    events = drain(kernel)
    assert events["0.compute"].start_ns == 0
    assert events["1.copy"].start_ns == 10
    assert events["1.copy"].end_ns == 25 < events["0.compute"].end_ns
    assert events["2.scalar"].start_ns == 0
    assert events["1.copy"].resource_predecessors["component.vram.write"]["resource_id"] == "hbm.compute"
    assert kernel.metrics["physical_owner_service_ns"]["hbm.physical"] == 18
    assert kernel.resource_busy_ns["hbm.compute"] == 10
    assert kernel.resource_busy_ns["component.vram.write"] == 8


def test_full_duplex_transfer_is_independent_only_if_owners_are_distinct():
    tasks = (task("read", ResourceDemand("component.vram.read", 10)),
             task("write", ResourceDemand("component.vram.write", 12)))
    shared = UnifiedEventKernel.from_closed_graph(tasks, resource_owners=OWNERS)
    drain(shared)
    separate = UnifiedEventKernel.from_closed_graph(tasks)
    drain(separate)
    assert shared.makespan_ns == 22
    assert separate.makespan_ns == 12


def test_shared_owner_keeps_independent_dma_lanes_and_compiled_parity():
    tasks = tuple(task(str(index), ResourceDemand("copy." + str(index), 10)) for index in range(3))
    owners = {"copy." + str(index): "dma" for index in range(3)}
    plain = UnifiedEventKernel(resource_capacities={"dma": 2}, resource_owners=owners)
    plain.add_tasks(tasks)
    expected = drain(plain)
    compiled = UnifiedEventKernel(resource_capacities={"dma": 2}, resource_owners=owners)
    actual = {event.task.task_id: event for event in compiled._drain_prevalidated_compiled(
        tasks, CompiledGraphLayout.compile(tasks))}
    assert [(e.start_ns, e.end_ns) for e in actual.values()] == [(e.start_ns, e.end_ns) for e in expected.values()]
    assert [e.start_ns for e in actual.values()] == [0, 0, 10]
    assert actual["2"].resource_lanes["copy.2"] == 0
    assert compiled.metrics == plain.metrics


def test_conflicting_owner_aliases_and_duplicate_demand_fail_before_submission():
    with pytest.raises(ValueError, match="alias chains"):
        UnifiedEventKernel(resource_owners={"a": "b", "b": "c"})
    with pytest.raises(ValueError, match="conflicting capacity"):
        UnifiedEventKernel(resource_owners={"a": "memory", "b": "memory"}, resource_capacities={"a": 1, "b": 2})
    kernel = UnifiedEventKernel(resource_owners={"a": "memory", "b": "memory"})
    before = kernel.metrics
    with pytest.raises(ValueError, match="coalesce"):
        kernel.submit((task("double", ResourceDemand("a", 1), ResourceDemand("b", 2)),))
    assert kernel.active_task_count == 0
    assert kernel.metrics == before


def test_owner_mapping_preserves_exact_and_streaming_critical_path():
    tasks = (
        task("arrival", marker=TraceMarker.REQUEST_ARRIVAL),
        task("a", ResourceDemand("hbm.compute", 10), ResourceDemand("tensor", 30), dependencies=("arrival",)),
        task("b", ResourceDemand("component.vram.write", 25), dependencies=("arrival",), category=TaskCategory.COMMUNICATION),
        task("done", dependencies=("b",), marker=TraceMarker.REQUEST_DONE),
    )
    schedule = ScheduleIR(manifest(), tasks, resource_owners=OWNERS)
    exact = simulate_schedule(schedule)
    fake_schedule = SimpleNamespace(
        scenario=SimpleNamespace(workload=SimpleNamespace(requests=())),
        requests=(SimpleNamespace(request_id="req", arrival_ns=0),),
        manifest=manifest(), resource_capacities={}, resource_owners=OWNERS,
    )
    # The numerical test targets shared-owner causal accounting, independent
    # of the language-model planner and its expensive default presets.
    with patch("heterollm_sim.streaming_des._compilation_scope") as scope, \
         patch("heterollm_sim.streaming_des._request_task_chunks", return_value=iter((RequestTaskChunk(tasks, "done", True),))):
        scope.return_value.__enter__.return_value = None
        streamed = execute_incremental_schedule(fake_schedule)
    assert exact.makespan_ns == streamed.trace.makespan_ns == 35
    assert summarize_metrics(exact).critical_path_category_ns == streamed.metrics.critical_path_category_ns


def phase(name, resource, duration, byte_count=0):
    return TransferPhase(name, (ResourceDemand(resource, duration, bytes_moved=byte_count),), {})


def pipeline(sizes=(4, 4, 4, 4), *, capacity=8, credits=4, consume_ns=7):
    return plan_transfer_pipeline(
        tuple((phase("produce", "link", 2, size),) for size in sizes),
        chunk_byte_counts=sizes, buffer_capacity_bytes=capacity,
        max_inflight_chunks=credits,
        consumers=tuple(phase("consume", "tensor", consume_ns, size) for size in sizes),
    )


def test_finite_buffer_backpressure_and_chunk_readiness():
    plan = pipeline()
    kernel = UnifiedEventKernel.from_closed_graph(plan.tasks)
    events = drain(kernel)
    assert events["transfer_pipeline.chunk0002.stage00"].start_ns >= events[plan.consumer_done_task_ids[0]].end_ns
    boundaries = []
    for index, size in enumerate(plan.chunk_byte_counts):
        producer = events["transfer_pipeline.chunk{:04d}.stage00".format(index)]
        consumer = events[plan.consumer_done_task_ids[index]]
        ready = events[plan.chunk_ready_task_ids[index]]
        assert consumer.start_ns >= ready.end_ns
        boundaries.extend(((producer.start_ns, size), (consumer.end_ns, -size)))
    reserved = 0
    for _, delta in sorted(boundaries, key=lambda pair: (pair[0], pair[1])):
        reserved += delta
        assert 0 <= reserved <= plan.buffer_capacity_bytes
    assert reserved == 0
    assert plan.ideal_lower_bound_ns <= kernel.makespan_ns <= plan.serialized_upper_bound_ns


def test_one_credit_is_serial_and_more_buffer_can_overlap():
    serial = pipeline(capacity=4, credits=1)
    overlap = pipeline(capacity=16, credits=4)
    ks = UnifiedEventKernel.from_closed_graph(serial.tasks)
    ko = UnifiedEventKernel.from_closed_graph(overlap.tasks)
    drain(ks)
    drain(ko)
    assert ks.makespan_ns == serial.serialized_upper_bound_ns == 36
    assert ko.makespan_ns == 30
    assert ko.makespan_ns < ks.makespan_ns


def test_byte_capacity_and_credit_limit_are_independent_constraints():
    by_bytes = pipeline(sizes=(5, 2, 4, 3), capacity=7, credits=4)
    by_credit = pipeline(sizes=(5, 2, 4, 3), capacity=100, credits=1)
    first = next(t for t in by_bytes.tasks if t.task_id.endswith("chunk0002.stage00"))
    assert first.metadata["credit_dependency_ids"] == (by_bytes.consumer_done_task_ids[0],)
    events = drain(UnifiedEventKernel.from_closed_graph(by_credit.tasks))
    for index in range(1, 4):
        assert events["transfer_pipeline.chunk{:04d}.stage00".format(index)].start_ns >= events[by_credit.consumer_done_task_ids[index - 1]].end_ns
    with pytest.raises(ValueError, match="chunk exceeds"):
        pipeline(sizes=(8,), capacity=7)


def test_pipeline_reuses_transaction_costs_without_dividing_startup_latency():
    port = PortSpec("p", "pcie", "endpoint", bandwidth_gbps=8)
    source = ComponentSpec("src", "gpu", ports=(port,), read_bandwidth_gbps=0)
    target = ComponentSpec("dst", "gpu", ports=(port,), write_bandwidth_gbps=0)
    hardware = HardwareSpec("route", (source, target), (LinkSpec("link", "src", "p", "dst", "p", "pcie", bandwidth_gbps=8, latency_ns=3),))
    router = TopologyRouter(hardware)
    plan = router.transfer_pipeline("src", "dst", 10, chunk_size_bytes=4,
                                    buffer_capacity_bytes=8, max_inflight_chunks=2)
    assert plan.chunk_byte_counts == (4, 4, 2)
    transfer_tasks = [task for task in plan.tasks if not task.metadata["is_chunk_consumer"]]
    assert [task.demands[0].service_ns for task in transfer_tasks] == [7, 7, 5]
    assert sum(task.demands[0].bytes_moved for task in transfer_tasks) == 10


def test_hardware_ownership_requires_explicit_declaration():
    memory = ComponentSpec("mem", "hbm", metadata={"memory_service_owner": "controller"})
    hardware = HardwareSpec("h", (memory,), (), metadata={"physical_resource_owners": {"hbm.compute": "controller"}})
    assert declared_resource_owners(hardware) == {
        "hbm.compute": "controller", "component.mem.read": "controller", "component.mem.write": "controller",
    }
    assert "dma" not in declared_resource_owners(hardware)
    assert declared_resource_owners(replace(hardware, metadata={}, components=(replace(memory, metadata={}),))) == {}


def overheads():
    return InvocationOverheads(5, 9, 3, 2, 1, source_refs=("independent-microbenchmark-fixture",))


def body(invocation):
    return (task(invocation + ".compute", ResourceDemand("tensor", 10)),)


def test_invocation_first_call_capture_replay_and_update_are_separate_events():
    lifecycle = InvocationLifecycle("binary+runtime+device-fixture", overheads())
    first = lifecycle.lower(body("a"), invocation_id="a", context_id="context", graph_key="decode", graph_signature="shape-a", timing_role="warmup")
    second = lifecycle.lower(body("b"), invocation_id="b", context_id="context", graph_key="decode", graph_signature="shape-a")
    third = lifecycle.lower(body("c"), invocation_id="c", context_id="context", graph_key="decode", graph_signature="shape-b", allow_graph_update=True)
    assert [p.transition for p in (first, second, third)] == ["capture", "replay", "update"]
    assert [p.initialization_applied for p in (first, second, third)] == [True, False, False]
    assert sum(first.overhead_ns.values()) == 17
    assert sum(second.overhead_ns.values()) == 3
    assert sum(third.overhead_ns.values()) == 6
    events = drain(UnifiedEventKernel.from_closed_graph(first.tasks + second.tasks + third.tasks))
    assert events[first.completion_task_id].end_ns == 27
    assert events[second.completion_task_id].end_ns == 40
    assert events[third.completion_task_id].end_ns == 56
    assert all(task.metadata["timing_role"] == "warmup" for task in first.tasks)
    assert all(task.metadata["validation_status"] == "mechanism_only" for task in first.tasks)


def test_invocation_new_context_recapture_and_invalid_calls_do_not_consume_state():
    lifecycle = InvocationLifecycle("runtime", overheads())
    first = lifecycle.lower(body("a"), invocation_id="a", context_id="context", graph_key="decode", graph_signature="shape-a")
    with pytest.raises(ValueError, match="required together"):
        lifecycle.lower(body("b"), invocation_id="b", context_id="context", graph_key="decode")
    changed = lifecycle.lower(body("b"), invocation_id="b", context_id="context", graph_key="decode", graph_signature="shape-b")
    other = lifecycle.lower(body("c"), invocation_id="c", context_id="new-context", graph_key="decode", graph_signature="shape-a")
    assert changed.transition == "recapture"
    assert other.initialization_applied
    with pytest.raises(ValueError, match="already lowered"):
        lifecycle.lower(body("a"), invocation_id="a", context_id="context")
    with pytest.raises(ValueError, match="source_refs"):
        InvocationOverheads(initialization_ns=1)


def test_cost_and_dynamic_runtime_honor_shared_owners():
    from heterollm_sim.scalable_serving import execute_cost_schedule
    from heterollm_sim.runtime import ControlPlaneRuntime
    tasks = (task("a", ResourceDemand("hbm.compute", 10)),
             task("b", ResourceDemand("component.vram.write", 12)))
    schedule = ScheduleIR(manifest(), tasks, resource_owners=OWNERS)
    assert execute_cost_schedule(schedule).makespan_ns == 22
    result = ControlPlaneRuntime({"resource_owners": OWNERS}).run(tasks, {})
    assert result.makespan_ns == 22
    assert result.kernel_metrics["physical_owner_service_ns"]["hbm.physical"] == 22


def test_runtime_adapter_preserves_owner_mapping():
    from heterollm_sim.runtime_adapters import LlamaCppAdapter
    plan = LlamaCppAdapter().lower(())
    assert plan.to_schedule(manifest(), resource_owners=OWNERS).resource_owners == OWNERS


def test_invocation_overheads_do_not_run_before_declared_request_readiness():
    lifecycle = InvocationLifecycle("runtime", overheads())
    inputs = tuple(replace(t, earliest_start_ns=100) for t in body("a"))
    plan = lifecycle.lower(inputs, invocation_id="a", context_id="context")
    events = drain(UnifiedEventKernel.from_closed_graph(plan.tasks))
    assert min(event.start_ns for event in events.values()) == 100
    assert events[plan.completion_task_id].end_ns == 118


def test_coherent_fold_does_not_drop_same_physical_owner_service():
    port = PortSpec("p", "pcie", "endpoint", bandwidth_gbps=8)
    source = ComponentSpec("src", "host_memory", ports=(port,), read_bandwidth_gbps=8,
                           metadata={"memory_service_owner": "memory-controller"})
    target = ComponentSpec("dst", "hbm", ports=(port,), write_bandwidth_gbps=8,
                           metadata={"memory_service_owner": "memory-controller"})
    link = LinkSpec("link", "src", "p", "dst", "p", "pcie", bandwidth_gbps=8,
                    metadata={"transfer_execution": "coherent_dma"})
    router = TopologyRouter(HardwareSpec("same-owner", (source, target), (link,)))
    phases = router.transfer_phases("src", "dst", 8)
    assert len(phases) == 3
    assert all(p.metadata.get("transfer_execution") != "coherent_dma" for p in phases)
