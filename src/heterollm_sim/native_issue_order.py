"""Prove a state-independent issue order on every native arbitration point.

One-sided layer composition needs monotonicity: no admissible initial
memory state may finish a layer faster than the best-case reference D0
(every access a row hit, no REF, no turnaround, every NAND cache hit). The
native kernel starts a ready Task by its static key (effective ready,
dependency ready, earliest start, phase, sequence, Task id) inside its
resource group. If every pair of DAG-unordered Tasks sharing a calendar or a
physical owner keeps one order in every state, the kernel is a fixed-order
max-plus system in its service times, and an extra row/REF/cache delay can
only postpone completions. Physical owners must also not reorder: HBM needs
the ``fcfs`` scheduler; NAND/SRAM cores expose no reordering scheduler.

Only three pair shapes are accepted; any other unordered pair rejects the
proof so the caller keeps the symmetric composition:

* ``same_release``: equal dependency sets, same stage and resource group.
  Both become ready at one instant in every state; the static tie-break is
  state-independent.
* ``serialized_release``: each Task depends on exactly one parent, and the
  two parents form a ``same_release`` pair of fixed-service non-physical
  Tasks on single-lane resources. The later parent starts after the earlier
  one has released all of its lanes, so its positive service makes the two
  dependency-ready times strictly ordered in every state.
* ``dominated_release``: one Task has an ancestor in the same stage whose
  dependency set contains the other Task's whole dependency set and whose
  service is positive. That ancestor cannot finish before the other Task is
  ready, so the dependency-ready times are strictly ordered in every state
  (closed ring AllReduce: a write waits for its neighbour's read and link).
"""
from __future__ import annotations

from collections import defaultdict

_SCOPE_LIMIT = 4096


def _positive_service(task):
    md = task.metadata
    return (any(demand.service_ns > 0. for demand in task.demands)
            or any(type(access.get("byte_count")) is int and access["byte_count"] > 0
                   for access in md.get("memory_accesses", ())))


def native_issue_order_proof(stages, rows, physical_runtime, native_runtime):
    """Return the issue-order proof for ``rows`` = (stage id, Task, resources)."""
    from .memory_types import DramConfig

    unavailable = {"qualified": False, "task_count": len(rows), "physical_state_advanced": False}
    if native_runtime is None or physical_runtime is None:
        return {**unavailable, "reason": "native_kernel_or_physical_runtime_unavailable"}
    if len(rows) > _SCOPE_LIMIT:
        return {**unavailable, "reason": "native_issue_order_exceeds_bounded_layer_scope"}
    kernel = native_runtime._execution_resource_kernel
    owner_of, capacities = kernel._physical_owner, kernel.resource_capacities
    for owner, active in physical_runtime.runtimes.items():
        config = active.core.config
        if isinstance(config, DramConfig) and config.scheduler != "fcfs" and any(
                f"@physical-owner:{owner}" in resources for _stage, _task, resources in rows):
            return {**unavailable, "reason": "native_physical_owner_may_reorder:"+owner}

    # Prepared roots inherit every terminal Task of their dependency stages.
    index = {(stage, task.task_id): position for position, (stage, task, _r) in enumerate(rows)}
    stage_of = {stage.stage_id: stage for stage in stages}
    terminals = defaultdict(list)
    referenced = defaultdict(set)
    for stage, task, _resources in rows:
        referenced[stage].update(task.dependencies)
    for position, (stage, task, _resources) in enumerate(rows):
        if task.task_id not in referenced[stage]:
            terminals[stage].append(position)
    parents, releases = [], []
    for stage, task, _resources in rows:
        if task.dependencies:
            known = [index[(stage, parent)] for parent in task.dependencies if (stage, parent) in index]
            external = {("task", stage, parent) for parent in task.dependencies if (stage, parent) not in index}
        else:
            parent_stages = stage_of[stage].dependencies if stage in stage_of else ()
            known = [position for parent in parent_stages for position in terminals.get(parent, ())]
            external = {("stage", parent) for parent in parent_stages if parent not in terminals}
        parents.append(tuple(dict.fromkeys(known)))
        releases.append(frozenset(known) | frozenset(external))
    children = [[] for _ in rows]
    indegree = [len(row) for row in parents]
    for child, row in enumerate(parents):
        for parent in row:
            children[parent].append(child)
    ancestors = [0]*len(rows)
    ready = [position for position, count in enumerate(indegree) if not count]
    visited = 0
    while ready:
        position = ready.pop()
        visited += 1
        for child in children[position]:
            ancestors[child] |= ancestors[position] | (1 << position)
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
    if visited != len(rows):
        return {**unavailable, "reason": "native_issue_order_graph_cyclic"}

    tasks = [task for _stage, task, _resources in rows]
    stage_ids = [stage for stage, _task, _resources in rows]
    groups = [tuple(sorted({demand.resource_id for demand in task.demands})) for task in tasks]
    points = defaultdict(list)
    for position, (_stage, task, resources) in enumerate(rows):
        named = {owner_of(demand.resource_id) for demand in task.demands}
        named.update(resource for resource in resources if resource.startswith("@physical-"))
        for resource in named:
            points[resource].append(position)

    def same_release(a, b):
        return stage_ids[a] == stage_ids[b] and groups[a] == groups[b] and releases[a] == releases[b]

    def serialized(a, b):
        if len(parents[a]) != 1 or len(parents[b]) != 1 or releases[a] != frozenset(parents[a]) \
                or releases[b] != frozenset(parents[b]):
            return False
        x, y = parents[a][0], parents[b][0]
        return (x != y and same_release(x, y)
                and not tasks[x].metadata.get("memory_accesses") and not tasks[y].metadata.get("memory_accesses")
                and all(capacities.get(resource, 1) == 1 for resource in groups[x])
                and _positive_service(tasks[x]) and _positive_service(tasks[y]))

    dominators = {}

    def dominator_mask(a):
        key = (stage_ids[a], releases[a])
        if key not in dominators:
            dominators[key] = sum(1 << position for position in range(len(rows))
                if stage_ids[position] == stage_ids[a] and releases[a] <= releases[position]
                and _positive_service(tasks[position]))
        return dominators[key]

    counts, unproven, seen = defaultdict(int), [], set()
    for resource, members in points.items():
        for i, a in enumerate(members):
            for b in members[i+1:]:
                if (ancestors[b] >> a) & 1 or (ancestors[a] >> b) & 1:
                    continue
                pair = (min(a, b), max(a, b))
                if pair in seen:
                    continue
                seen.add(pair)
                if stage_ids[a] != stage_ids[b] or groups[a] != groups[b]:
                    rule = None
                elif releases[a] == releases[b]:
                    rule = "same_release"
                elif serialized(a, b):
                    rule = "serialized_release"
                elif ancestors[b] & dominator_mask(a) or ancestors[a] & dominator_mask(b):
                    rule = "dominated_release"
                else:
                    rule = None
                if rule is None:
                    unproven.append({"resource": resource, "tasks": [tasks[a].task_id, tasks[b].task_id]})
                else:
                    counts[rule] += 1
    proof = {"qualified": not unproven, "task_count": len(rows), "physical_state_advanced": False,
        "arbitration_points": len(points), "unordered_pairs_by_rule": dict(sorted(counts.items())),
        "unproven_pair_count": len(unproven), "unproven_pairs": unproven[:8],
        "basis": "native_static_ready_key_FCFS_per_resource_group; every_DAG_unordered_pair_on_a_kernel_calendar_or_physical_owner_has_a_state_independent_order; HBM_fcfs; one_sided_guards_require_this_monotonicity"}
    if unproven:
        proof["reason"] = "native_issue_order_may_depend_on_initial_state"
    return proof


__all__ = ["native_issue_order_proof"]
