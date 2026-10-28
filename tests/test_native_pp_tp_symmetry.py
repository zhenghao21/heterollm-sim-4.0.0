"""Pipeline TP siblings reuse only equal physical transitions within a stage."""
from dataclasses import replace

import pytest

from heterollm_sim import native_layer_sampling as sampling
from heterollm_sim.execution_control import ExecutionControl
from test_native_ep_tp_symmetry import semantic, window
from test_native_pp_schedule import pp_scenario
from test_native_tp_symmetry import exact
from test_token_owned_history import compare_physical_state


@pytest.fixture(params=(1, 2))
def pp_mirror_prepared(request, monkeypatch):
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.serving import _OnlineRuntime

    class Captured(BaseException):
        pass

    found = {}

    def capture(self, active, cohort, cost, stages, ready):
        assert active is self
        found.update(runtime=self, stages=stages, ready=ready)
        raise Captured()

    monkeypatch.setattr(_OnlineRuntime, "_native_layer_sampler", capture, raising=False)
    scenario = pp_scenario(request.param, tp=2, pp=2, batch=4)
    with pytest.raises(Captured):
        run_scenario(scenario, retention_policy="aggregate")
    runtime = found["runtime"]
    source = runtime._execution_resource_kernel
    origin = max(runtime.now, found["ready"], source.makespan_ns)
    specs, _ = runtime._stage_task_specs(found["stages"],
        {stage.stage_id: origin for stage in found["stages"]}, "pp-exact")
    # Layer work only, as in the sampler's windows without the cohort
    # prefix: an unranked physical prefix leaves rank states unequal.
    wanted = frozenset(task.task_id for task in specs if task.metadata.get("layer_id") is not None)
    selected = tuple(replace(task,
        dependencies=tuple(parent for parent in task.dependencies if parent in wanted),
        earliest_start_ns=max(float(task.earliest_start_ns), origin))
        for task in specs if task.task_id in wanted)
    return {**found, "source": source, "origin": origin, "specs": specs,
            "wanted": wanted, "selected": selected, "scenario": runtime.plan.scenario}


def test_pp_tp_reuses_each_stage_and_preserves_every_event_and_state(pp_mirror_prepared, monkeypatch):
    prepared = pp_mirror_prepared
    expected, reference, events = window(prepared, monkeypatch, "off")
    actual, candidate, seen = window(prepared, monkeypatch, "quiet")
    assert actual["tp_symmetry"]["candidate_tasks"] > 0
    # Unequal frontend/PD arrivals are an exact-proof rejection.  The native
    # path must preserve the full result and may therefore legitimately fall
    # back to zero mirrored tasks for this fixture.
    assert 0 <= actual["tp_symmetry"]["mirrored_tasks"] <= actual["tp_symmetry"]["candidate_tasks"]
    assert actual["tp_symmetry"]["mode"] == "exact"
    assert tuple(map(exact, seen)) == tuple(map(exact, events))
    assert semantic(actual) == semantic(expected)
    coordinates = {row.rank: row for row in prepared["scenario"].placement.parallel.rank_mapping}
    tasks = {task.task_id: task for task in prepared["selected"]}
    mirror = candidate._native_tp_mirror
    paired_stages = set()
    for follower, (leader, _rename, _source_owners, _owners) in mirror.pairs.items():
        left = coordinates[tasks[leader].metadata["rank"]]
        right = coordinates[tasks[follower].metadata["rank"]]
        assert left.tp_rank == 0 and right.tp_rank != 0
        assert (left.pp_rank, left.ep_rank) == (right.pp_rank, right.ep_rank)
        paired_stages.add(left.pp_rank)
    assert paired_stages == {0, 1}
    mirror.materialize(candidate)
    compare_physical_state(candidate, reference)


def test_asymmetric_pp_follower_keeps_only_its_stage_on_native_path(pp_mirror_prepared):
    from heterollm_sim.data_motion import resolve_service

    prepared = pp_mirror_prepared
    source, _ = sampling._independent_kernel(prepared["source"])
    scenario = prepared["scenario"]
    follower = next(row for row in scenario.placement.parallel.rank_mapping
                    if row.pp_rank == 1 and row.tp_rank == 1)
    memory = next(component for component in scenario.hardware.components
                  if component.component_id == follower.memory_component_id)
    source.physical_runtime.runtimes[resolve_service(memory).physical_owner].clock_ns += .125
    mirror = sampling._TPRankMirror(source, prepared["selected"], scenario)
    tasks = {task.task_id: task for task in prepared["selected"]}
    assert mirror.pairs
    assert all(tasks[identity].metadata["rank"] != follower.rank for identity in mirror.pairs)
    assert {tasks[identity].metadata["rank"] for identity in mirror.pairs} == {1}


def test_pp_does_not_enable_the_separate_approximate_pd_reuse(pp_mirror_prepared):
    prepared = pp_mirror_prepared
    mirror = sampling._TPRankMirror(prepared["source"], prepared["selected"],
        prepared["scenario"], approximate_imports=True)
    assert not mirror.pairs
