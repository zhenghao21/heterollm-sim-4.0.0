"""Conditional max-plus composition of sampled Prefill chunk task graphs.

This module never owns or advances a serving runtime.  A chunk's ``nodes``
retain its host/controller preparation DAG and replace serial native layer
work with sampled service intervals.  ``duration_ns`` is service, excluding
predecessor waits; subtracting a whole entry span is not an admissible way
to remove its prefix.  ``ready_ns`` contains absolute external constraints.

The same stage in consecutive chunks retains the native cross-chunk stage
barrier.  ``resource_ids`` declare *serial term* resources in the supplied
node order, not every resource touched briefly inside a collapsed layer.
Additional native device fences can be supplied as ``extra_edges``.

The interval propagation is monotone for that fixed graph/order.  Whether
the sampled intervals cover future chunk controller/cache/address states
is a separate, explicit modeling condition, not a physical error proof.
"""
from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
import math


FIXED_ORDER_ASSUMPTION = (
    "The declared chunk/task dependencies and serial resource order remain fixed across "
    "the composed interval; collapsed terms preserve the preparation/device boundaries "
    "needed for cross-chunk overlap.")
CHUNK_BOUNDARY_STATE_ASSUMPTION = (
    "Each chunk's sampled service intervals represent its declared future boundary "
    "controller, cache and allocation state; omitted chunks are not executed, and no "
    "complete native runtime state is produced by this composition.")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite non-negative number")
    return value


def _bounds(row, name):
    lower = _number(row["lower"], name + ".lower")
    upper = _number(row["upper"], name + ".upper")
    if lower > upper:
        raise ValueError(name + " has reversed bounds")
    return lower, upper


def _interval(lower, upper):
    total = lower + upper
    return {"lower": lower, "upper": upper,
            "estimate": 0.0 if not total else 2.0 * lower * upper / total,
            "relative_error_budget": 0.0 if not total else (upper - lower) / total}


def _identity(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(name + " must be a non-empty string")
    return value


def compose_prefill_chunks(chunks, *, origin_ns=0.0, target_relative_error=0.10,
                           assumptions=(), extra_edges=(), interval_guard_ns=0.0):
    """Compose pure chunk descriptions, ending at the final chunk's last layer.

    Each chunk is ``{chunk_id, nodes, last_layer_node_id}``.  A node has
    ``node_id``, ``stage_id``, ``duration_ns: {lower, upper}``, optional local
    ``dependencies``, optional serial ``resource_ids`` and optional absolute
    ``ready_ns: {lower, upper}``.  Nodes list a fixed resource order, but need
    not otherwise be topologically sorted.  ``extra_edges`` is a sequence
    of ``((from_chunk_id, from_node_id), (to_chunk_id, to_node_id))`` pairs.

    ``interval_guard_ns`` widens both ends for an observed PP stage-overlap
    envelope when concurrent stage service was collapsed into serial terms.
    In keeping with the existing Prefill budget, the independent interval
    limit is target/(2+target).  Passing it is conditional model acceptance;
    it is not full-prompt validation or a hardware accuracy guarantee.
    """
    origin = _number(origin_ns, "origin_ns")
    interval_guard_ns = _number(interval_guard_ns, "interval_guard_ns")
    target = _number(target_relative_error, "target_relative_error")
    if target <= 0 or target >= 1:
        raise ValueError("target_relative_error must be between zero and one")
    chunks = tuple(chunks)
    if not chunks:
        raise ValueError("Prefill composition requires at least one chunk")
    nodes, dependencies, by_chunk, chunk_stages = {}, {}, {}, {}
    last_resource_user = {}
    terminals = []
    for chunk in chunks:
        chunk_id = _identity(chunk["chunk_id"], "chunk_id")
        if chunk_id in by_chunk:
            raise ValueError("duplicate Prefill chunk_id: " + chunk_id)
        local, stages = {}, defaultdict(list)
        for node in chunk["nodes"]:
            node_id = _identity(node["node_id"], "node_id")
            stage_id = _identity(node["stage_id"], "stage_id")
            if node_id in local:
                raise ValueError("duplicate Prefill node_id: " + node_id)
            key = (chunk_id, node_id)
            duration = _bounds(node["duration_ns"], "duration_ns")
            ready = _bounds(node.get("ready_ns", {"lower": origin, "upper": origin}), "ready_ns")
            local[node_id] = key
            nodes[key] = {"duration": duration, "ready": ready, "stage_id": stage_id}
            dependencies[key] = {(chunk_id, parent) for parent in node.get("dependencies", ())}
            stages[stage_id].append(key)
            for resource in dict.fromkeys(node.get("resource_ids", ())):
                _identity(resource, "resource_id")
                previous = last_resource_user.get(resource)
                if previous is not None:
                    dependencies[key].add(previous)
                last_resource_user[resource] = key
        if not local:
            raise ValueError("Prefill chunks must contain task/term nodes")
        terminal_id = chunk["last_layer_node_id"]
        if terminal_id not in local:
            raise ValueError("last_layer_node_id is absent from its chunk")
        terminals.append(local[terminal_id])
        by_chunk[chunk_id], chunk_stages[chunk_id] = local, stages

    for key, parents in dependencies.items():
        if any(parent not in nodes for parent in parents):
            raise ValueError("Prefill node has an unknown local dependency: " + repr(key))

    # The runtime's request-stage frontier is retained until the same stage
    # next appears, including when an intervening chunk omits that stage.
    previous_stage = {}
    for chunk in chunks:
        stages = chunk_stages[chunk["chunk_id"]]
        for stage_id, stage_nodes in stages.items():
            members = set(stage_nodes)
            roots = [key for key in stage_nodes if not dependencies[key] & members]
            used = {parent for key in stage_nodes for parent in dependencies[key] if parent in members}
            ends = members - used
            for root in roots:
                dependencies[root].update(previous_stage.get(stage_id, ()))
            previous_stage[stage_id] = ends

    for parent, child in extra_edges:
        parent, child = tuple(parent), tuple(child)
        if parent not in nodes or child not in nodes:
            raise ValueError("extra Prefill edge has an unknown node")
        dependencies[child].add(parent)

    followers, remaining = defaultdict(list), {}
    for key, parents in dependencies.items():
        remaining[key] = len(parents)
        for parent in parents:
            followers[parent].append(key)
    ready_queue = deque(key for key in nodes if not remaining[key])
    intervals = {}
    while ready_queue:
        key = ready_queue.popleft()
        row, parents = nodes[key], dependencies[key]
        start_lo = max(origin, row["ready"][0], *(intervals[parent][1][0] for parent in parents))
        start_hi = max(origin, row["ready"][1], *(intervals[parent][1][1] for parent in parents))
        end = (start_lo + row["duration"][0], start_hi + row["duration"][1])
        if not all(math.isfinite(value) for value in end):
            raise ValueError("Prefill composition overflowed its time bounds")
        intervals[key] = ((start_lo, start_hi), end)
        for child in followers[key]:
            remaining[child] -= 1
            if not remaining[child]:
                ready_queue.append(child)
    if len(intervals) != len(nodes):
        raise ValueError("Prefill task/stage/resource constraints contain a cycle")

    finish = intervals[terminals[-1]][1]
    metric = _interval(max(0.0, finish[0] - origin - interval_guard_ns),
                       finish[1] - origin + interval_guard_ns)
    effective_target = target / (2 + target)
    estimate, lower, upper = (Fraction(metric[name]) for name in ("estimate", "lower", "upper"))
    valid_lower = lower > 0
    within = (estimate <= (1 + Fraction(effective_target)) * lower
              and estimate >= (1 - Fraction(effective_target)) * upper)
    reasons = ([] if valid_lower and within else ["native_prefill_independent_budget_exceeded"
               if valid_lower else "native_prefill_pass_lower_bound_not_positive"])
    all_assumptions = list(dict.fromkeys((FIXED_ORDER_ASSUMPTION, CHUNK_BOUNDARY_STATE_ASSUMPTION,
                                         *assumptions)))
    return {"accepted": not reasons, "reasons": reasons,
        "metrics": {"prefill_pass_ns": metric["estimate"]} if not reasons else {},
        "metric_bounds": {"prefill_pass_ns": metric},
        "chunk_last_layer_bounds": {chunk["chunk_id"]: _interval(*intervals[terminal][1])
                                    for chunk, terminal in zip(chunks, terminals)},
        "node_bounds": {chunk_id: {node_id: {"start_ns": _interval(*intervals[key][0]),
                                             "end_ns": _interval(*intervals[key][1])}
                                  for node_id, key in local.items()}
                        for chunk_id, local in by_chunk.items()},
        "error_budget": {"basis": "fixed_order_chunk_task_DAG_max_plus_intervals",
            "target_relative_error": target, "effective_independent_target": effective_target,
            "conditional_only": True, "validation_status": "not_full_prompt_validation",
            "origin_ns": origin, "terminal": terminals[-1], "chunk_count": len(chunks),
            "node_count": len(nodes), "edge_count": sum(map(len, dependencies.values())),
            "assumptions": all_assumptions,
            "interval_guard_ns": interval_guard_ns,
            "physical_runtime_state_advanced": False, "physical_runtime_state_resumable": False}}


__all__ = ["compose_prefill_chunks", "FIXED_ORDER_ASSUMPTION", "CHUNK_BOUNDARY_STATE_ASSUMPTION"]
