"""TP reuse stays within one EP group and preserves native graph guards.

Positive coverage requires actual reuse,
not merely that an unsupported graph silently falls back.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim import native_layer_sampling as sampling
from heterollm_sim.execution_control import ExecutionControl
from test_native_tp_symmetry import exact
from test_token_owned_history import compare_physical_state
import native_toy


@pytest.fixture
def ep_mirror_prepared(monkeypatch):
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
    scenario = native_toy.moe_toy_scenario(4, batch_size=4, output=6,
        warmup=2, measurement=4, ep=2)
    with pytest.raises(Captured):
        run_scenario(scenario, retention_policy="aggregate")
    runtime = found["runtime"]
    source = runtime._execution_resource_kernel
    origin = max(runtime.now, found["ready"], source.makespan_ns)
    specs, _ = runtime._stage_task_specs(found["stages"],
        {stage.stage_id: origin for stage in found["stages"]}, "ep-exact")
    layers = {"layer-000", "layer-001"}
    wanted = frozenset(task.task_id for task in specs if task.metadata.get("layer_id") in layers)
    selected = tuple(replace(task,
        dependencies=tuple(parent for parent in task.dependencies if parent in wanted),
        earliest_start_ns=max(float(task.earliest_start_ns), origin))
        for task in specs if task.task_id in wanted)
    assert selected and {task.metadata["layer_id"] for task in selected} == layers
    return {**found, "source": source, "origin": origin, "specs": specs,
            "wanted": wanted, "selected": selected, "scenario": runtime.plan.scenario}


def window(prepared, monkeypatch, mode, *, with_import=False):
    seen, copies = [], []
    original_observe = sampling.LiveCohortMetrics.observe
    original_copy = sampling._independent_kernel

    def observe(self, event):
        seen.append(event)
        original_observe(self, event)

    def copy(*args, **kwargs):
        result = original_copy(*args, **kwargs)
        copies.append(result[0])
        return result

    with monkeypatch.context() as patch:
        patch.setattr(sampling.LiveCohortMetrics, "observe", observe)
        patch.setattr(sampling, "_independent_kernel", copy)
        result = sampling._execute_window(prepared["source"], prepared["specs"],
            prepared["wanted"], origin_ns=prepared["origin"], control=ExecutionControl(),
            sample_id="ep-exact", tp_symmetry=(prepared["scenario"], mode),
            concurrent_imports=(prepared["runtime"]._external_executor, []) if with_import else None)
    assert len(copies) == 1
    return result, copies[0], tuple(seen)


def semantic(row):
    return exact({key: value for key, value in row.items()
                  if key not in ("wall_seconds", "wall_breakdown_seconds", "tp_symmetry")})


@pytest.mark.parametrize("mode,with_import", (("quiet", False), ("all", False), ("all", True)))
def test_ep_groups_really_reuse_native_moe_and_keep_events_state(
        ep_mirror_prepared, monkeypatch, mode, with_import):
    prepared = ep_mirror_prepared
    expected, reference, events = window(prepared, monkeypatch, "off", with_import=with_import)
    actual, candidate, seen = window(prepared, monkeypatch, mode, with_import=with_import)
    assert actual["tp_symmetry"]["candidate_tasks"] > 0
    assert actual["tp_symmetry"]["mirrored_tasks"] > 0
    assert actual["tp_symmetry"]["mode"] == ("approximate_PD" if with_import else "exact")
    assert tuple(map(exact, seen)) == tuple(map(exact, events))
    assert semantic(actual) == semantic(expected)
    assert any(event.task.metadata.get("collective_kind") == "all_to_all"
               and event.task.metadata.get("collective_name", "").endswith(".expert_dispatch")
               for event in seen)
    assert any(event.task.metadata.get("collective_kind") == "all_to_all"
               and event.task.metadata.get("collective_name", "").endswith(".expert_combine")
               for event in seen)
    parallel = prepared["scenario"].placement.parallel
    coordinates = {row.rank: row for row in parallel.rank_mapping}
    tasks = {task.task_id: task for task in prepared["selected"]}
    mirror = candidate._native_tp_mirror
    for follower, (leader, _rename, _source_owners, _owners) in mirror.pairs.items():
        left = coordinates[tasks[leader].metadata["rank"]]
        right = coordinates[tasks[follower].metadata["rank"]]
        assert left.tp_rank == 0 and right.tp_rank != 0
        assert (left.ep_rank, left.pp_rank) == (right.ep_rank, right.pp_rank)
    # Lazy copies are part of the implementation; compare resumed native
    # controller/allocator state only after the existing public materializer.
    mirror.materialize(candidate)
    if not with_import:
        compare_physical_state(candidate, reference)
    else:
        # The existing PD approximation may publish first-touch keys in the
        # renamed representative's order and leaves its per-request touched
        # recorder at the last real writer. This inert live-import scope proves
        # equal controller/calendar values and events, not a resumable PD state.
        for field in ("metrics", "resource_available", "resource_last_interval",
                      "_resource_lane_available", "_resource_lane_last_interval",
                      "completed_end_ns", "completion_leases"):
            assert exact(getattr(candidate, field)) == exact(getattr(reference, field)), field
        left, right = (kernel.physical_runtime.snapshot() for kernel in (candidate, reference))
        for field in ("timeline", "allocator_state", "committed_owners", "hbf_staging"):
            values = [snapshot[field] for snapshot in (left, right)]
            if field == "timeline":
                values = [{key: value for key, value in rows.items() if key != "_touched"} for rows in values]
            assert exact(values[0]) == exact(values[1]), field
        for owner, actual in left["runtime_state"].items():
            expected = right["runtime_state"][owner]
            assert exact((actual["clock_ns"], actual["core"])) == exact((expected["clock_ns"], expected["core"]))


def test_asymmetric_initial_ep_group_keeps_its_original_work(ep_mirror_prepared):
    from heterollm_sim.data_motion import resolve_service
    prepared = ep_mirror_prepared
    source, _ = sampling._independent_kernel(prepared["source"])
    scenario = prepared["scenario"]
    follower = next(row for row in scenario.placement.parallel.rank_mapping
                    if row.ep_rank == 0 and row.tp_rank == 1)
    memory = next(component for component in scenario.hardware.components
                  if component.component_id == follower.memory_component_id)
    source.physical_runtime.runtimes[resolve_service(memory).physical_owner].clock_ns += .125
    mirror = sampling._TPRankMirror(source, prepared["selected"], scenario, allow_ep_groups=True)
    tasks = {task.task_id: task for task in prepared["selected"]}
    assert all(tasks[identity].metadata["rank"] != follower.rank for identity in mirror.pairs)


@pytest.mark.parametrize("approximate_imports", (False, True))
def test_ep_groups_require_independent_rank_packages(ep_mirror_prepared, approximate_imports):
    prepared = ep_mirror_prepared
    scenario = prepared["scenario"]
    components = {row.component_id: row for row in scenario.hardware.components}
    mappings = scenario.placement.parallel.rank_mapping
    first, second = (components[mappings[index].component_id].package_id for index in (0, 1))
    changed = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        replace(component, package_id=first) if component.package_id == second else component
        for component in scenario.hardware.components)))
    mirror = sampling._TPRankMirror(prepared["source"], prepared["selected"], changed,
                                    allow_ep_groups=True, approximate_imports=approximate_imports)
    assert not mirror.pairs


@pytest.mark.parametrize("approximate_imports", (False, True))
def test_ep_groups_reject_missing_tp_coordinate(ep_mirror_prepared, approximate_imports):
    prepared = ep_mirror_prepared
    scenario = prepared["scenario"]
    parallel = scenario.placement.parallel
    rows = list(parallel.rank_mapping)
    rows[1] = replace(rows[1], tp_rank=rows[0].tp_rank)
    # Deliberately malformed raw input exercises the mirror's own guard,
    # before ParallelSpec validation could reject the duplicated coordinate.
    raw = SimpleNamespace(hardware=scenario.hardware,
        placement=SimpleNamespace(parallel=SimpleNamespace(tp_degree=parallel.tp_degree,
            pp_degree=parallel.pp_degree, ep_degree=parallel.ep_degree, rank_mapping=tuple(rows))))
    mirror = sampling._TPRankMirror(prepared["source"], prepared["selected"], raw,
                                    allow_ep_groups=True, approximate_imports=approximate_imports)
    assert not mirror.pairs


def test_quiet_keeps_pd_windows_on_the_full_native_path(ep_mirror_prepared, monkeypatch):
    prepared = ep_mirror_prepared
    expected, _reference, events = window(prepared, monkeypatch, "off", with_import=True)
    actual, _candidate, seen = window(prepared, monkeypatch, "quiet", with_import=True)
    assert actual["tp_symmetry"] is None
    assert tuple(map(exact, seen)) == tuple(map(exact, events))
    assert semantic(actual) == semantic(expected)


def test_approximate_ep_pairing_still_requires_the_original_dependency_edges(ep_mirror_prepared):
    prepared = ep_mirror_prepared
    mirror = sampling._TPRankMirror(prepared["source"], prepared["selected"],
        prepared["scenario"], approximate_imports=True, allow_ep_groups=True)
    target = next(task for task in prepared["selected"] if task.task_id in mirror.pairs and task.dependencies)
    tasks = tuple(replace(task, dependencies=()) if task is target else task for task in prepared["selected"])
    changed = sampling._TPRankMirror(prepared["source"], tasks,
        prepared["scenario"], approximate_imports=True, allow_ep_groups=True)
    assert target.task_id not in changed.pairs


def test_approximate_ep_pairing_runs_unequal_dispatch_arrivals_in_full(ep_mirror_prepared):
    prepared = ep_mirror_prepared
    kernel, _ = sampling._independent_kernel(prepared["source"])
    mirror = sampling._TPRankMirror(kernel, prepared["selected"],
        prepared["scenario"], approximate_imports=True, allow_ep_groups=True)
    identity = next(iter(mirror.pairs))
    task = next(task for task in prepared["selected"] if task.task_id == identity)
    leader = mirror.pairs[identity][0]
    mirror.records[leader] = {"start": prepared["origin"] + 1.}
    seen = []
    def full(kernel, original, start, demands):
        seen.append((original, start, demands))
        return original, demands, False, False, None
    result = mirror.resolve(kernel, task, prepared["origin"], task.demands, full)
    assert seen == [(task, prepared["origin"], task.demands)]
    assert result[0] is task and mirror.stats["fallback_tasks"] == 1
    assert mirror.stats["mirrored_tasks"] == 0


def test_pd_phase_invalidates_before_external_write_and_frontier_publication(ep_mirror_prepared, monkeypatch):
    from heterollm_sim import _resource_ledger
    prepared = ep_mirror_prepared
    kernel, _ = sampling._independent_kernel(prepared["source"])
    mirror = sampling._TPRankMirror(kernel, prepared["selected"],
        prepared["scenario"], approximate_imports=True, allow_ep_groups=True)
    assert mirror.pairs
    kernel._native_tp_mirror = mirror
    calls = []
    invalidate = mirror.invalidate
    def invalidate_and_record(kernel):
        calls.append("invalidate")
        invalidate(kernel)
    monkeypatch.setattr(mirror, "invalidate", invalidate_and_record)
    state = {"request_id": "external", "done": False, "destinations": (),
        "completion_ns": prepared["origin"], "final": False, "phase_token": object()}
    def advance(state, token):
        assert token is state["phase_token"] and not mirror.tokens
        calls.append("external-phase")
        state.update(final=True, last_write=(prepared["origin"], 1))
    executor = SimpleNamespace(_async_plan_states={"external": state}, _advance_external_phase=advance)
    for name in ("publish_timeline_frontiers", "publish_owner_frontiers"):
        original = getattr(_resource_ledger, name)
        def publish(*args, _name=name, _original=original):
            calls.append(_name)
            return _original(*args)
        monkeypatch.setattr(_resource_ledger, name, publish)
    mirror.tokens["stale"] = 1
    imports = sampling._ConcurrentImports(kernel, executor, ["external"])
    imports.advance_until(prepared["origin"])
    assert calls == ["invalidate", "external-phase", "invalidate",
                     "publish_timeline_frontiers", "publish_owner_frontiers"]
    assert mirror.pairs and not mirror.tokens
    assert imports.submitted == 1 and imports.writes == [(prepared["origin"], 1, "external")]
