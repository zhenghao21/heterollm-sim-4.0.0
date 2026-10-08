"""Source lifecycle transitions and task binding, independent of LLM latency."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cuda_graph_lifecycle import (
    CudaGraphInvocation, CudaGraphRuntime, CudaGraphState,
    bind_cuda_graph_tasks,
)


def invocation(**changes):
    return replace(CudaGraphInvocation(
        context_id="cuda:0", graph_key="tensor-object:17", graph_uid=0,
        node_properties=(("tensor:17", "output:buffer:1", (1, 256), (4, 4)),),
        compatible=True, compatibility_reason="no_synchronizing_dispatch"), **changes)


def execute(runtime, call=None, now=0):
    transition = runtime.prepare(call or invocation(), host_time_us=now)
    runtime.commit(transition)
    return transition


def captured_runtime():
    runtime = CudaGraphRuntime()
    execute(runtime)
    return runtime, execute(runtime)


def task(name, phase, dependencies=(), *, target="gpu0", service=10.0, byte_count=0):
    return TaskSpec(name, "request", name, TaskCategory.COMPUTE,
        dependencies=tuple(dependencies),
        demands=(ResourceDemand("gpu0.frontend" if phase == "kernel_launch" else "gpu0.memory",
                                service, bytes_moved=byte_count),),
        metadata={"phase": phase, "target_component": target})


def test_first_stable_call_captures_and_updates_fresh_executable():
    runtime = CudaGraphRuntime()
    direct = execute(runtime)
    assert not direct.use_graph
    assert direct.events == ("ordinary_submit",)
    capture = execute(runtime)
    assert capture.events == ("capture", "instantiate", "update", "first_launch_submit")
    replay = execute(runtime)
    assert replay.events == ("replay_submit",)
    assert replay.executable_id == capture.executable_id
    assert replay.replay_id != capture.replay_id
    assert capture.metadata()["body_executions"] == 1
    assert capture.metadata()["capture_executes_body"] is False


def test_changed_buffer_pointer_resets_warmup_before_recapture():
    runtime, capture = captured_runtime()
    changed = invocation(node_properties=(("tensor:17", "output:buffer:2", (1, 256), (4, 4)),))
    direct = execute(runtime, changed)
    assert direct.reason == "properties_changed_reset_warmup"
    assert not direct.use_graph
    with pytest.raises(ValueError, match="update compatibility"):
        runtime.prepare(changed, host_time_us=0)
    recapture = execute(runtime, replace(changed, update_result="success"))
    assert recapture.events == ("destroy_graph", "capture", "update", "replay_submit")
    assert recapture.executable_id == capture.executable_id


def test_update_constraints_failure_explicitly_reinstantiates():
    runtime, old = captured_runtime()
    changed = invocation(node_properties=(("new-op",), ("new-op-2",)),
                         update_result="constraints_failure")
    execute(runtime, changed)
    result = execute(runtime, changed)
    assert result.events == ("destroy_graph", "capture", "update_failure", "destroy_exec",
                             "instantiate", "first_launch_submit")
    assert result.executable_id != old.executable_id


def test_unknown_update_can_only_produce_an_explicit_unpriced_diagnostic():
    runtime, _ = captured_runtime()
    changed = invocation(node_properties=(("changed",),))
    execute(runtime, changed)
    pending = runtime.prepare(changed, host_time_us=0, allow_unknown_update=True)
    assert pending.use_graph
    assert pending.events[-1] == "launch_submit_unresolved"
    assert pending.update_compatibility_pending
    assert pending.metadata()["pricing_ready"] is False
    assert pending.metadata()["executable_id"] is None
    runtime.commit(pending)
    later = execute(runtime, changed)
    assert later.metadata()["pricing_ready"] is False


def test_same_nonzero_uid_uses_source_shortcut_but_enforces_node_count():
    runtime = CudaGraphRuntime()
    execute(runtime, invocation(graph_uid=19))
    # The native shortcut intentionally skips property equality when the
    # runtime-provided nonzero UID proves this is the same GGML graph.
    result = execute(runtime, invocation(graph_uid=19, node_properties=(("different",),)))
    assert result.use_graph
    with pytest.raises(ValueError, match="node count"):
        runtime.prepare(invocation(graph_uid=19, node_properties=((1,), (2,))), host_time_us=0)


def test_incompatible_invocation_does_not_fake_property_update():
    runtime, capture = captured_runtime()
    incompatible = invocation(node_properties=(("unsupported",),), compatible=False,
                              compatibility_reason="mul_mat_id_needs_sync")
    result = execute(runtime, incompatible)
    assert result.reason == "incompatible:mul_mat_id_needs_sync"
    assert not result.use_graph
    result = execute(runtime)
    assert result.events == ("replay_submit",)
    assert result.executable_id == capture.executable_id


def test_keys_and_device_contexts_have_independent_warmup():
    runtime, _ = captured_runtime()
    assert not execute(runtime, invocation(graph_key="tensor-object:18")).use_graph
    assert not execute(runtime, invocation(context_id="cuda:1")).use_graph
    assert execute(runtime).events == ("replay_submit",)


def test_cache_eviction_exact_boundary_restarts_warmup_with_new_executable_identity():
    runtime, old = captured_runtime()
    # Sweep at 5s leaves the original graph, because its age is <10s.
    other = invocation(graph_key="other")
    at_five = execute(runtime, other, now=5_000_000)
    assert at_five.evicted == ()
    at_ten = execute(runtime, now=10_000_000)
    assert len(at_ten.evicted) == 1
    assert not at_ten.use_graph
    new = execute(runtime, now=10_000_001)
    assert new.events[-1] == "first_launch_submit"
    assert old.executable_id != new.executable_id


def test_prepare_does_not_advance_state_and_stale_or_foreign_commits_fail():
    runtime = CudaGraphRuntime()
    first = runtime.prepare(invocation(), host_time_us=0)
    assert runtime.state.revision == 0
    foreign = CudaGraphRuntime()
    with pytest.raises(ValueError, match="stale"):
        foreign.commit(first)
    runtime.commit(first)
    with pytest.raises(ValueError, match="stale"):
        runtime.commit(first)


def test_unknown_initialization_and_failed_execution_do_not_claim_cold_state():
    for runtime in (CudaGraphRuntime(CudaGraphState.unknown()), CudaGraphRuntime()):
        runtime.failed()
        with pytest.raises(ValueError, match="initialization is unknown"):
            runtime.prepare(invocation(), host_time_us=0)


def test_binding_waits_for_all_inputs_and_preserves_compute_dram_demands():
    _, transition = captured_runtime()
    tasks = (
        task("input1", "dma"), task("input2", "dma"),
        task("launch1", "kernel_launch", ("input1",)),
        task("body1", "memory", ("launch1",), service=800, byte_count=2048),
        task("launch2", "kernel_launch", ("body1", "input2")),
        task("body2", "compute", ("launch2",), service=600),
    )
    result = bind_cuda_graph_tasks(tasks, transition,
        member_task_ids=("launch1", "body1", "launch2", "body2"))
    assert result[2].dependencies == ("input1", "input2")
    assert tuple(t.demands for t in result) == tuple(t.demands for t in tasks)
    assert result[3].dependencies == tasks[3].dependencies
    assert result[2].metadata["cuda_graph_lifecycle_events"] == transition.events
    assert "cuda_graph_lifecycle_events" not in result[4].metadata
    assert {t.metadata["cuda_graph_id"] for t in result[2:]} == {transition.replay_id}


def test_binding_rejects_host_roundtrip_inside_a_claimed_device_split():
    _, transition = captured_runtime()
    tasks = (task("launch1", "kernel_launch"),
             task("body1", "compute", ("launch1",)),
             task("host", "cpu", ("body1",), target="cpu0"),
             task("launch2", "kernel_launch", ("host",)),
             task("body2", "compute", ("launch2",)))
    with pytest.raises(ValueError, match="host dependency split"):
        bind_cuda_graph_tasks(tasks, transition,
            member_task_ids=("launch1", "body1", "launch2", "body2"))


def test_planner_rejects_graph_enabled_without_lifecycle_binding(monkeypatch):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_component", lambda *args: SimpleNamespace(kind="gpu"))
    monkeypatch.setattr(planner, "_gpu_profiles", lambda *args:
        (SimpleNamespace(kernel_model=SimpleNamespace(graph_enabled=True)), None))
    context_token = planner._COMPILATION_CONTEXT.set(SimpleNamespace(scenario=object()))
    try:
        with pytest.raises(ValueError, match="ordinary-launch fallback is disabled"):
            planner._apply_planned_graph_launches((task("launch", "kernel_launch"),))
    finally:
        planner._COMPILATION_CONTEXT.reset(context_token)


def test_planner_graph_off_preserves_existing_task_costs(monkeypatch):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_component", lambda *args: SimpleNamespace(kind="gpu"))
    monkeypatch.setattr(planner, "_gpu_profiles", lambda *args:
        (SimpleNamespace(kernel_model=SimpleNamespace(graph_enabled=False)), None))
    context_token = planner._COMPILATION_CONTEXT.set(SimpleNamespace(scenario=object()))
    tasks = (task("launch", "kernel_launch"),)
    try:
        assert planner._apply_planned_graph_launches(tasks) == tasks
    finally:
        planner._COMPILATION_CONTEXT.reset(context_token)


def test_planner_graph_off_independent_costs_require_a_binding(monkeypatch):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_component", lambda *args: SimpleNamespace(kind="gpu"))
    monkeypatch.setattr(planner, "_gpu_profiles", lambda *args:
        (SimpleNamespace(kernel_model=SimpleNamespace(graph_enabled=False,
            runtime_calibration=object())), None))
    context_token = planner._COMPILATION_CONTEXT.set(SimpleNamespace(scenario=object()))
    try:
        with pytest.raises(ValueError, match="ordinary-launch fallback is disabled"):
            planner._apply_planned_graph_launches((task("launch", "kernel_launch"),))
    finally:
        planner._COMPILATION_CONTEXT.reset(context_token)


def test_planner_non_cuda_launch_never_resolves_gpu_profile(monkeypatch):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_component", lambda *args: SimpleNamespace(kind="cpu"))
    def unexpected(*args):
        raise AssertionError("CPU/CIM launch must not resolve CUDA profile")
    monkeypatch.setattr(planner, "_gpu_profiles", unexpected)
    context_token = planner._COMPILATION_CONTEXT.set(SimpleNamespace(scenario=object()))
    tasks = (task("launch", "kernel_launch", target="cpu0"),)
    try:
        assert planner._apply_planned_graph_launches(tasks) == tasks
    finally:
        planner._COMPILATION_CONTEXT.reset(context_token)


def test_planner_keeps_inserted_host_cost_tasks_and_direct_warmup_binding(monkeypatch):
    from heterollm_sim import planner
    monkeypatch.setattr(planner, "_component", lambda *args: SimpleNamespace(kind="gpu"))
    runtime = CudaGraphRuntime()
    transition = execute(runtime)
    tasks = bind_cuda_graph_tasks((task("launch", "kernel_launch"),
                                   task("body", "memory", ("launch",), byte_count=100)),
                                  transition, member_task_ids=("launch", "body"))
    inserted = task("launch.host", "cuda_runtime_ordinary_submit", target="cpu0")
    calls = []
    monkeypatch.setattr(planner, "_gpu_profiles", lambda *args:
        (SimpleNamespace(kernel_model=SimpleNamespace(graph_enabled=True)), None))

    def price(members, profile):
        calls.append(members)
        return (inserted, *members)

    monkeypatch.setattr(planner, "apply_captured_graph_launch", price)
    context_token = planner._COMPILATION_CONTEXT.set(SimpleNamespace(scenario=object()))
    try:
        result = planner._apply_planned_graph_launches(tasks)
        assert tuple(t.task_id for t in result) == ("launch.host", "launch", "body")
        assert calls == [tasks]
        assert result[-1].demands == tasks[-1].demands
    finally:
        planner._COMPILATION_CONTEXT.reset(context_token)
