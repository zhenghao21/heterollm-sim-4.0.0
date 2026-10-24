"""Real layer subgraphs remain isolated native physical work, not Tokens."""
from dataclasses import replace
import sys
from pathlib import Path

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim import native_layer_sampling as layer
from test_token_owned_history import bits


def task(identity, dependencies=(), *, layer_id=None):
    return TaskSpec(identity, "r0", identity, TaskCategory.COMPUTE,
        tuple(dependencies), (ResourceDemand("gpu", 2.),),
        metadata={} if layer_id is None else {"layer_id": layer_id})


def rows(count):
    return [{"layer_id": f"L{index}", "layer_index": index,
             "class_key": "same", "signature": {"width": 4}}
            for index in range(count)]


def test_graph_proof_requires_all_rank_terminals_before_next_layer():
    graph = (task("prefix"), task("a", ("prefix",), layer_id="L0"),
             task("b", ("prefix",), layer_id="L0"),
             task("next", ("a",), layer_id="L1"), task("suffix", ("next",)))
    proof, _ = layer._graph_facts(graph, rows(2))
    assert not proof["qualified"] and not proof["serial_layer_boundaries_proven"]
    joined = (*graph[:3], replace(graph[3], dependencies=("a", "b")), graph[4])
    proof, boundary = layer._graph_facts(joined, rows(2))
    assert proof["qualified"] and proof["nonserial_non_layer_groups"] == []
    assert boundary == {"before_first_layer": ["prefix"], "after_last_layer": ["suffix"]}


def test_nonlayer_work_that_overlaps_a_layer_is_explicitly_identified():
    graph = (task("prefix"), task("independent"),
             task("a", ("prefix",), layer_id="L0"),
             task("next", ("a",), layer_id="L1"), task("suffix", ("next",)))
    proof, boundary = layer._graph_facts(graph, rows(2))
    assert proof["nonserial_non_layer_groups"] == ["independent"]
    assert boundary["independent"] == ["independent"]


def test_isolated_window_runs_same_tasks_as_original_native_step_engine(monkeypatch):
    source = UnifiedEventKernel()
    graph = (task("omitted"), task("a", ("omitted",), layer_id="L0"),
             task("b", ("a",), layer_id="L0"))
    observed = []
    original = layer.LiveCohortMetrics.observe
    def observe(self, event):
        observed.append((event.task.task_id, event.start_ns.hex(), event.end_ns.hex(),
                         bits(event.demands), bits(event.resource_predecessors)))
        original(self, event)
    monkeypatch.setattr(layer.LiveCohortMetrics, "observe", observe)
    result = layer._execute_window(source, graph, ("a", "b"), origin_ns=10.,
                                   control=ExecutionControl(), sample_id="proof")
    reference, _executor = layer._independent_kernel(source)
    reference.submit(tuple(replace(row, dependencies=tuple(key for key in row.dependencies if key != "omitted"),
                                   earliest_start_ns=10.) for row in graph[1:]))
    expected = []
    while (event := reference.step()) is not None:
        expected.append((event.task.task_id, event.start_ns.hex(), event.end_ns.hex(),
                         bits(event.demands), bits(event.resource_predecessors)))
    assert observed == expected
    assert source.completed_count == 0 and source.makespan_ns == 0.
    assert result["input_boundary"]["omitted_dependency_task_ids"] == ["omitted"]
    assert result["duration_ns"] == result["end_ns"]-result["start_ns"] == 4.
    assert result["visible_output_tokens"] == 0 and not result["full_cohort_committed"]


@pytest.fixture(params=("HBM4", "HBF"))
def native_prepared(request, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from exploring_hbf_native_config import build_native_comparison_scenario
    from exploring_hbf_native_pilot import build_pilot_scenario
    from heterollm_sim.serving import _OnlineRuntime
    from heterollm_sim.reporting import run_scenario

    class Captured(BaseException):
        pass
    found = {}
    def capture(self, active, cohort, cost, stages, ready):
        assert active is self
        found.update(runtime=self, cohort=cohort, cost=cost, stages=stages, ready=ready)
        raise Captured()
    monkeypatch.setattr(_OnlineRuntime, "_native_layer_sampler", capture, raising=False)
    scenario = build_native_comparison_scenario("llama3_405b", request.param, "SHORT",
        tp=2, batch_size=1, model_override=build_pilot_scenario("hbm_tp1").model,
        prompt_override=8, output_override=4, warmup_override=0, measurement_override=2)
    with pytest.raises(Captured):
        run_scenario(scenario, retention_policy="aggregate")
    return found


@pytest.mark.parametrize("width", (1, 2))
def test_real_media_and_collective_events_match_original_native_steps(native_prepared, monkeypatch, width):
    runtime = native_prepared["runtime"]
    source = runtime._execution_resource_kernel
    origin = max(runtime.now, native_prepared["ready"], source.makespan_ns)
    specs, _ = runtime._stage_task_specs(native_prepared["stages"],
        {stage.stage_id: origin for stage in native_prepared["stages"]}, "comparison")
    layers = ("dense0",) if width == 1 else ("dense0", "dense1")
    wanted = frozenset(row.task_id for row in specs if row.metadata.get("layer_id") in layers)
    assert {row.metadata["layer_id"] for row in specs if row.task_id in wanted} == set(layers)
    seen = []
    original = layer.LiveCohortMetrics.observe
    def observe(self, event):
        seen.append(bits(event))
        original(self, event)
    monkeypatch.setattr(layer.LiveCohortMetrics, "observe", observe)
    record = layer._execute_window(source, specs, wanted, origin_ns=origin,
        control=ExecutionControl(), sample_id="real-physical-comparison")
    selected = tuple(replace(row, dependencies=tuple(key for key in row.dependencies if key in wanted),
                             earliest_start_ns=max(row.earliest_start_ns, origin))
                     for row in specs if row.task_id in wanted)
    reference, _executor = layer._independent_kernel(source)
    reference.submit(selected)
    reference._index_physical_allocation_uses(selected)
    expected = []
    while (event := reference.step()) is not None:
        expected.append(bits(event))
    assert seen == expected
    assert record["physical_task_count"] > 0
    assert any(row.metadata.get("event_kind") == "collective_transfer" for row in selected)
