"""Deferred resource paths preserve causal ties and complete accounting."""
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import KernelEvent
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.planner import _accumulate_physical_resource_totals, _resource_busy_by_direction


def event(task_id, category, start, end, *, dependencies=(), predecessor=None,
          resource="wire", lane=0, release=None):
    demand = ResourceDemand(resource, end - start, bytes_moved=3)
    metadata = {}
    if release is not None:
        metadata = {"physical_demands_resource_ids": (resource,),
                    "physical_resource_lane_completions": ((resource, lane, release),)}
    task = TaskSpec(task_id, "request", task_id, category, dependencies=dependencies,
                    demands=(demand,), metadata=metadata)
    return KernelEvent(task, start, end, start, start, (demand,),
        {resource: predecessor} if predecessor is not None else {}, {resource: lane})


def collect(events):
    live = LiveCohortMetrics()
    for item in events:
        live.observe(item)
    return live.summary()


@pytest.mark.parametrize("dependency,release_task,winning_category", [
    ("z-dependency", "a-resource", TaskCategory.MEMORY),
    ("a-dependency", "z-resource", TaskCategory.COMPUTE),
])
def test_resource_equal_distance_keeps_full_lexicographic_path_rule(
        dependency, release_task, winning_category):
    items = (
        event(dependency, TaskCategory.COMPUTE, 0., 5., resource="other"),
        event(release_task, TaskCategory.MEMORY, 0., 5., lane=1, release=5.),
        event("consumer", TaskCategory.OUTPUT, 5., 7., dependencies=(dependency,), lane=1,
              predecessor={"task_id": release_task, "resource_id": "wire", "end_ns": 5., "lane": 1}),
    )
    result = collect(items)
    assert result.critical_path_category_ns == {TaskCategory.OUTPUT: 2., winning_category: 5.}
    assert result.bytes_moved == 9
    assert result.resource_busy_ns == {"other": 5., "wire": 7.}


def test_external_cohort_predecessor_does_not_reuse_old_local_lane_path():
    result = collect((
        event("local-old", TaskCategory.MEMORY, 0., 4.),
        event("consumer", TaskCategory.COMPUTE, 8., 10., predecessor={
            "task_id": "external-cohort-task", "resource_id": "wire", "end_ns": 8.}),
    ))
    assert result.critical_path_category_ns == {TaskCategory.COMPUTE: 2.}
    assert result.makespan_ns == 10.


def test_internal_late_release_and_zero_paths_do_not_add_false_waits():
    result = collect((
        event("empty", TaskCategory.MEMORY, 0., 0., release=0.),
        event("memory", TaskCategory.MEMORY, 0., 20., predecessor={
            "task_id": "empty", "resource_id": "wire", "end_ns": 0.}, release=20.),
        event("overlap", TaskCategory.COMPUTE, 0., 30., predecessor={
            "task_id": "memory", "resource_id": "wire", "end_ns": 20.}),
    ))
    assert result.critical_path_category_ns == {TaskCategory.COMPUTE: 30.}


def test_physical_resource_totals_preserve_order_duplicates_and_owner_conflicts():
    task = SimpleNamespace(demands=(ResourceDemand("wire", 0.), ResourceDemand("other", 0.),
                                   ResourceDemand("wire", 0.)))
    totals = {}
    for value, owner in ((1e16, "first"), (1., "second"), (-1e16, "second")):
        _accumulate_physical_resource_totals(totals, task,
            {"resource_busy_ns": {"wire": value, "absent": 900.},
             "resource_bytes": {"wire": 7}, "resource_owners": {"wire": owner}}, {})
    assert totals == {"wire": {"owner": "wire", "bytes_moved": 42, "service_ns": 0.}}
    _accumulate_physical_resource_totals({}, task,
        {"resource_busy_ns": {"absent": 1.}, "resource_bytes": None}, {})
    with pytest.raises(ValueError, match="resource_bytes"):
        _accumulate_physical_resource_totals({}, task,
            {"resource_busy_ns": {"wire": 1.}, "resource_bytes": None}, {})


def test_direction_reducer_preserves_overrides_write_shares_and_float_order():
    def task(direction, overrides, modeled_write, demands):
        return SimpleNamespace(name="ordinary", metadata={"memory_direction": direction,
            "resource_directions": overrides, "modeled_memory_write_bytes": modeled_write},
            demands=tuple(ResourceDemand(resource, duration, bytes_moved=moved)
                          for resource, duration, moved in demands))
    tasks = (
        task("read", {"explicit": "WrItE", "invalid": 7}, 3,
             (("shared", 7., 4), ("explicit", 7., 4), ("invalid", 8., 4))),
        task("write", None, 0, (("ordinary", 2., 0), ("zero", 0., 9))),
        *(task("read", {}, 0, (("rounding", duration, 0),)) for duration in (1e16, 1., 1.)),
    )
    assert _resource_busy_by_direction(tasks) == {
        "read": {"invalid": 2., "rounding": 1e16, "shared": 1.75},
        "write": {"explicit": 7., "invalid": 6., "ordinary": 2., "shared": 5.25},
    }
