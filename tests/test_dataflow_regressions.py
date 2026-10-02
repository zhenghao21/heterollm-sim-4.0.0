from heterollm_sim.contracts import RunManifest, TaskCategory, TaskSpec, ResourceDemand, TraceMarker
from heterollm_sim.engine import ScheduleIR, simulate_schedule
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.metrics import summarize_metrics
from heterollm_sim.runtime_adapters import LlamaCppAdapter
from heterollm_sim.scalable_serving import execute_cost_schedule
from heterollm_sim.streaming_des import execute_incremental_schedule
from heterollm_sim.contracts import RetentionPolicy
from types import SimpleNamespace
import pytest


def _manifest():
    return RunManifest(schema_version="4.0.0", run_id="regression", random_seed=1,
                       simulator_version="test", model_name="m", hardware_name="h",
                       workload_name="w")


def test_critical_path_keeps_dependency_after_resource_queue_wait():
    def t(task_id, category, resource, service, dependencies=(), marker=None):
        return TaskSpec(task_id=task_id, request_id="r", name=task_id,
                        category=category, dependencies=dependencies,
                        demands=(ResourceDemand(resource, service),), marker=marker)
    delayed = t("c", TaskCategory.COMPUTE, "gpu", 1, ("a",))
    delayed = TaskSpec(**{**delayed.__dict__, "earliest_start_ns": 100})
    trace = simulate_schedule(ScheduleIR(_manifest(), (
        t("a", TaskCategory.COMMUNICATION, "link", 5), delayed
    )))
    critical = summarize_metrics(trace).critical_path_category_ns
    assert critical[TaskCategory.COMMUNICATION] == 5
    assert critical[TaskCategory.COMPUTE] == 1


def test_throughput_excludes_idle_arrival_prefix():
    arrival = TaskSpec("arrival", "r", "arrival", TaskCategory.POLICY,
                       earliest_start_ns=100, marker=TraceMarker.REQUEST_ARRIVAL)
    done = TaskSpec("done", "r", "done", TaskCategory.OUTPUT,
                    dependencies=("arrival",), demands=(ResourceDemand("gpu", 10),),
                    marker=TraceMarker.REQUEST_DONE)
    metrics = summarize_metrics(simulate_schedule(ScheduleIR(_manifest(), (arrival, done))))
    assert metrics.throughput["requests_per_s"] == 100_000_000.0


def test_llama_cpp_rejects_unsupported_negative_gpu_layer_count():
    with pytest.raises(ValueError):
        LlamaCppAdapter().lower((), batch_size=64, ubatch_size=64,
                                parallel=1, gpu_layers=-2)


def test_multilane_exact_streaming_and_cost_agree(monkeypatch):
    tasks = (
        TaskSpec("a", "r", "a", TaskCategory.COMMUNICATION,
                 demands=(ResourceDemand("dma", 10),)),
        TaskSpec("b", "r", "b", TaskCategory.COMMUNICATION,
                 demands=(ResourceDemand("dma", 10),)),
        TaskSpec("c", "r", "c", TaskCategory.COMPUTE,
                 dependencies=("a", "b"), demands=(ResourceDemand("cpu", 1),)),
    )
    schedule = ScheduleIR(_manifest(), tasks, resource_capacities={"dma": 2})
    exact = simulate_schedule(schedule)
    cost = execute_cost_schedule(schedule)
    assert exact.makespan_ns == cost.makespan_ns == 11
    assert summarize_metrics(exact).resource_utilization["dma"] <= 1.0
    stream_schedule = SimpleNamespace(
        scenario=object(), manifest=_manifest(), resource_capacities={"dma": 2}
    )
    request = SimpleNamespace(request_id="r")
    chunk = SimpleNamespace(tasks=tasks, terminal_task_id="c", final=True)
    monkeypatch.setattr("heterollm_sim.streaming_des._request_iterator", lambda _s: iter((request,)))
    monkeypatch.setattr("heterollm_sim.streaming_des._request_task_chunks", lambda _s, _r: iter((chunk,)))
    monkeypatch.setattr("heterollm_sim.streaming_des._compilation_scope", lambda _s: __import__("contextlib").nullcontext())
    streaming = execute_incremental_schedule(stream_schedule, retention_policy=RetentionPolicy.EXACT)
    assert streaming.trace.makespan_ns == exact.makespan_ns
    assert streaming.metrics.resource_utilization["dma"] <= 1.0
    assert streaming.metrics.critical_path_category_ns == summarize_metrics(exact).critical_path_category_ns


@pytest.mark.parametrize("retention", list(RetentionPolicy))
@pytest.mark.parametrize("declared", ["read", "memory"])
@pytest.mark.parametrize("used", ["write", "memory"])
def test_shared_capacity_matches_execution_and_streaming_reports(
    monkeypatch, retention, declared, used
):
    owners = {"read": "memory", "write": "memory"}
    tasks = tuple(
        TaskSpec(str(i), "r", str(i), TaskCategory.COMMUNICATION,
                 demands=(ResourceDemand(used, 10),))
        for i in range(2)
    ) + (TaskSpec("done", "r", "done", TaskCategory.POLICY,
                  dependencies=("0", "1")),)
    capacities = {declared: 2}
    schedule = ScheduleIR(_manifest(), tasks, resource_capacities=capacities,
                          resource_owners=owners)
    exact = simulate_schedule(schedule)
    cost = execute_cost_schedule(schedule)
    stream_schedule = SimpleNamespace(scenario=object(), manifest=_manifest(),
                                      resource_capacities=capacities, resource_owners=owners)
    request = SimpleNamespace(request_id="r")
    chunk = SimpleNamespace(tasks=tasks, terminal_task_id="done", final=True)
    monkeypatch.setattr("heterollm_sim.streaming_des._request_iterator", lambda _s: iter((request,)))
    monkeypatch.setattr("heterollm_sim.streaming_des._request_task_chunks", lambda _s, _r: iter((chunk,)))
    monkeypatch.setattr("heterollm_sim.streaming_des._compilation_scope", lambda _s: __import__("contextlib").nullcontext())
    streaming = execute_incremental_schedule(stream_schedule, retention_policy=retention)
    assert exact.makespan_ns == cost.makespan_ns == streaming.trace.makespan_ns == 10
    assert exact.resource_capacities[used] == streaming.trace.resource_capacities[used] == 2
    assert summarize_metrics(exact).resource_utilization[used] == 1.0
    assert streaming.metrics.resource_utilization[used] == 1.0
    assert streaming.runtime_kernel_metrics["resource_capacities"][used] == 2
    assert capacities == {declared: 2}


def test_streaming_reports_inherited_live_kernel_capacity(monkeypatch):
    owners = {"read": "memory", "write": "memory"}
    tasks = (
        TaskSpec("0", "r", "0", TaskCategory.COMMUNICATION,
                 demands=(ResourceDemand("write", 10),)),
        TaskSpec("1", "r", "1", TaskCategory.COMMUNICATION,
                 demands=(ResourceDemand("write", 10),)),
        TaskSpec("done", "r", "done", TaskCategory.POLICY,
                 dependencies=("0", "1")),
    )
    kernel = UnifiedEventKernel(
        resource_capacities={"memory": 2}, resource_owners=owners
    )
    schedule = SimpleNamespace(
        scenario=object(), manifest=_manifest(), resource_capacities={},
        resource_owners=owners,
    )
    request = SimpleNamespace(request_id="r")
    chunk = SimpleNamespace(tasks=tasks, terminal_task_id="done", final=True)
    monkeypatch.setattr("heterollm_sim.streaming_des._request_iterator", lambda _s: iter((request,)))
    monkeypatch.setattr("heterollm_sim.streaming_des._request_task_chunks", lambda _s, _r: iter((chunk,)))
    monkeypatch.setattr("heterollm_sim.streaming_des._compilation_scope", lambda _s: __import__("contextlib").nullcontext())
    result = execute_incremental_schedule(schedule, execution_kernel=kernel)
    assert result.trace.makespan_ns == 10
    assert result.trace.resource_capacities["write"] == 2
    assert result.metrics.resource_utilization["write"] == 1.0
