"""Real layer subgraphs remain isolated native physical work, not Tokens."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import sys
from pathlib import Path

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim import native_layer_sampling as layer
from heterollm_sim.native_fast_window import NativeLayerWindowSample
from heterollm_sim.simulation_options import NativeWindowFastConfig
from test_token_owned_history import bits


def task(identity, dependencies=(), *, layer_id=None):
    return TaskSpec(identity, "r0", identity, TaskCategory.COMPUTE,
        tuple(dependencies), (ResourceDemand("gpu", 2.),),
        metadata={} if layer_id is None else {"layer_id": layer_id})


def rows(count):
    return [{"layer_id": f"L{index}", "layer_index": index,
             "class_key": "same", "signature": {"width": 4}}
            for index in range(count)]


def test_window_selection_uses_disjoint_real_validation_layers():
    selected = layer._select_windows(rows(12), NativeWindowFastConfig(sampling_unit="layer_windows"))
    assert [row["role"] for row in selected] == ["calibration"]*2+["validation"]*2
    identities = [item["layer_id"] for row in selected for item in row["layers"]]
    assert len(identities) == len(set(identities))
    assert identities[0] == "L0" and identities[-1] == "L11"


@pytest.mark.parametrize("size", (2, 4))
def test_contiguous_windows_preserve_disjoint_holdout_and_never_cross_classes(size):
    inventory = rows(32)
    for row in inventory[16:]:
        row["class_key"] = "different_native_geometry"
    selected = layer._select_windows(inventory, NativeWindowFastConfig(
        sampling_unit="layer_windows", layer_window_size=size))
    identities = []
    for window in selected:
        indices = [row["layer_index"] for row in window["layers"]]
        assert indices == list(range(indices[0], indices[0]+size))
        assert len({row["class_key"] for row in window["layers"]}) == 1
        identities.extend(row["layer_id"] for row in window["layers"])
    assert len(selected) == 8
    assert len(identities) == len(set(identities)) == 8*size


def test_small_classes_are_exhaustive_without_fabricated_validation():
    selected = layer._select_windows(rows(2), NativeWindowFastConfig(sampling_unit="layer_windows"))
    assert [row["role"] for row in selected] == ["exhaustive", "exhaustive"]
    assert [row["layers"][0]["layer_id"] for row in selected] == ["L0", "L1"]


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


def _sample(prepared, control=None):
    return layer.sample_native_layer_windows(prepared["runtime"], prepared["cohort"],
        prepared["cost"], prepared["stages"], prepared["ready"],
        NativeWindowFastConfig(sampling_unit="layer_windows"), control or ExecutionControl())


def test_real_native_layers_include_collectives_and_leave_source_unchanged(native_prepared):
    runtime = native_prepared["runtime"]
    source = runtime._execution_resource_kernel
    def logical_state():
        saved = layer._independent_kernel(source)[0].physical_runtime.snapshot()
        return bits({"timeline": saved["timeline"], "allocator_state": saved["allocator_state"],
            "committed_owners": saved["committed_owners"], "hbf_staging": saved["hbf_staging"],
            "core_state": {owner: {"clock_ns": item["clock_ns"], "core": item["core"]}
                           for owner, item in saved["runtime_state"].items()}})
    before = logical_state()
    source_times = (runtime.now, source.makespan_ns, runtime._completed_decode_steps, len(runtime.events))
    result = _sample(native_prepared)
    assert source_times == (runtime.now, source.makespan_ns, runtime._completed_decode_steps, len(runtime.events))
    assert before == logical_state()
    assert not runtime.batches
    assert len(result["records"]) == 2
    fields = tuple(NativeLayerWindowSample.__dataclass_fields__)
    typed = [NativeLayerWindowSample(**{key: row[key] for key in fields}) for row in result["records"]]
    assert all(row.rank_ids == (0, 1) for row in typed)
    assert all(row.executed_task_count == 99 for row in typed)
    assert all(row.role == "exhaustive" for row in typed)
    assert {row["position"] for row in result["non_layer_samples"]} == {"before_first_layer", "after_last_layer"}
    assert result["non_layer_coverage_complete"]
    assert result["sampled_task_count"] == result["original_task_count"] == 257
    assert result["layer_composition"]["serial_layer_boundaries_proven"]
    assert result["layer_composition"]["boundary_guard_ns"] == 0
    assert result["full_cohorts_committed"] == result["visible_output_tokens"] == 0


def test_cancellation_preserves_only_completed_physical_windows(native_prepared, monkeypatch):
    native_prepared["runtime"]._capture_native_layer_phase_replays = True
    original = layer._execute_window
    calls = []
    class Deadline(Exception):
        pass
    def execute(*args, **kwargs):
        calls.append(kwargs["sample_id"])
        if len(calls) == 2:
            raise Deadline()
        return original(*args, **kwargs)
    monkeypatch.setattr(layer, "_execute_window", execute)
    with pytest.raises(Deadline):
        _sample(native_prepared)
    preserved = native_prepared["runtime"]._native_layer_sample_records
    assert len(preserved) == 1 and preserved[0]["executed_task_count"] == 99
    assert preserved[0]["sample_state"] == "independent_native_kernel_copy_not_full_cohort_prefix"
    assert set(native_prepared["runtime"]._native_layer_phase_replays) == {preserved[0]["sample_id"]}


def test_phase_replay_capture_is_explicit_private_and_complete(native_prepared):
    runtime = native_prepared["runtime"]
    runtime._capture_native_layer_phase_replays = True
    result = _sample(native_prepared)
    captured = runtime._native_layer_phase_replays
    assert set(captured) == {row["sample_id"] for row in result["records"]}
    for row in result["records"]:
        replay = captured[row["sample_id"]]
        assert replay["source_kernel"] is runtime._execution_resource_kernel
        assert replay["complete_native_window"] and not replay["source_kernel_modified"]
        assert len(replay["resolved_events"]) == len(replay["selected_specs"]) == row["task_count"]
        assert {event.task.task_id for event in replay["resolved_events"]} == {
            spec.task_id for spec in replay["selected_specs"]}
        assert "resolved_events" not in row and "selected_specs" not in row
        physical = [event for event in replay["resolved_events"]
                    if event.task.metadata.get("physical_execution_by_owner")]
        assert physical
        assert all(type(access["address"]) is int for event in physical
                   for access in event.task.metadata["memory_accesses"])


def test_phase_replay_capture_is_disabled_by_default(native_prepared):
    _sample(native_prepared)
    assert not hasattr(native_prepared["runtime"], "_native_layer_phase_replays")


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
