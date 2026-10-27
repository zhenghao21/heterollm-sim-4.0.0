"""Small independent schedule examples for the pure chunk interval algebra."""
from itertools import product

import pytest

from heterollm_sim.native_prefill_chunk_composition import compose_prefill_chunks


def node(identity, stage, duration, *parents, resources=(), ready=(0, 0)):
    lower, upper = duration if isinstance(duration, tuple) else (duration, duration)
    return {"node_id": identity, "stage_id": stage,
            "duration_ns": {"lower": lower, "upper": upper},
            "dependencies": parents, "resource_ids": resources,
            "ready_ns": {"lower": ready[0], "upper": ready[1]}}


def chunk(identity, nodes, last="last"):
    return {"chunk_id": identity, "nodes": nodes, "last_layer_node_id": last}


def test_three_chunk_host_preparation_overlaps_prior_device_work_and_excludes_head():
    chunks = [chunk(str(i), [node("prepare", "host", 4),
        node("first", "gpu", 6, "prepare"), node("last", "gpu", 4, "first")]) for i in range(3)]
    chunks[-1]["nodes"].append(node("head", "gpu", 100, "last"))
    result = compose_prefill_chunks(chunks)
    assert result["accepted"]
    assert result["metrics"]["prefill_pass_ns"] == 34  # 4 + 3*10, not 3*(4+10).
    assert result["node_bounds"]["1"]["prepare"]["start_ns"]["estimate"] == 4
    assert result["node_bounds"]["1"]["first"]["start_ns"]["estimate"] == 14
    assert result["node_bounds"]["2"]["head"]["end_ns"]["estimate"] == 134
    assert result["error_budget"]["physical_runtime_state_advanced"] is False


def test_every_duration_endpoint_is_inside_max_plus_interval():
    variable = [(3, 4), (9, 10), (2, 3), (8, 9)]

    def make(values):
        return [chunk("a", [node("prepare", "host", values[0]),
                            node("last", "gpu", values[1], "prepare")]),
                chunk("b", [node("prepare", "host", values[2]),
                            node("last", "gpu", values[3], "prepare")])]

    predicted = compose_prefill_chunks(make(variable))
    endpoints = [compose_prefill_chunks(make(values))["metric_bounds"]["prefill_pass_ns"]["estimate"]
                 for values in product(*variable)]
    bounds = predicted["metric_bounds"]["prefill_pass_ns"]
    assert bounds["lower"] == min(endpoints) == 20
    assert bounds["upper"] == max(endpoints) == 23
    assert not predicted["accepted"]  # Same independent 10%/2.1 budget as native Prefill.
    assert predicted["reasons"] == ["native_prefill_independent_budget_exceeded"]


def test_parallel_stage_roots_wait_for_every_prior_stage_terminal():
    a = chunk("a", [node("left", "preparation", 4), node("right", "preparation", 7),
                    node("last", "gpu", 2, "left", "right")])
    b = chunk("b", [node("prepare", "preparation", 1),
                    node("last", "gpu", 2, "prepare")])
    result = compose_prefill_chunks([a, b])
    assert result["node_bounds"]["b"]["prepare"]["start_ns"]["estimate"] == 7
    assert result["metrics"]["prefill_pass_ns"] == 11


def test_explicit_resource_order_and_cross_chunk_fence_are_respected():
    a = chunk("a", [node("prep", "host-a", 2, resources=("cpu",)),
                    node("last", "gpu-a", 10, "prep")])
    b = chunk("b", [node("prep", "host-b", 3, resources=("cpu",)),
                    node("last", "gpu-b", 10, "prep")])
    overlapped = compose_prefill_chunks([a, b])
    fenced = compose_prefill_chunks([a, b], extra_edges=[(("a", "last"), ("b", "prep"))])
    assert overlapped["metrics"]["prefill_pass_ns"] == 15
    assert fenced["metrics"]["prefill_pass_ns"] == 25


def test_stage_frontier_survives_a_chunk_without_that_stage_and_absolute_origin():
    a = chunk("a", [node("last", "gpu", 10, ready=(100, 100))])
    b = chunk("b", [node("last", "other", 1, ready=(100, 100))])
    c = chunk("c", [node("last", "gpu", 2, ready=(100, 100))])
    result = compose_prefill_chunks([a, b, c], origin_ns=100)
    assert result["metrics"]["prefill_pass_ns"] == 12
    assert result["chunk_last_layer_bounds"]["c"]["estimate"] == 112


def test_fixed_resource_order_cannot_silently_override_a_conflicting_dependency():
    chunks = [chunk("a", [node("last", "gpu", 1, "before", resources=("lane",)),
                           node("before", "host", 1, resources=("lane",))])]
    with pytest.raises(ValueError, match="cycle"):
        compose_prefill_chunks(chunks)


@pytest.mark.parametrize("changes", [
    {"duration_ns": {"lower": -1, "upper": 1}},
    {"duration_ns": {"lower": 2, "upper": 1}},
    {"ready_ns": {"lower": 0, "upper": float("nan")}},
    {"dependencies": ["missing"]},
])
def test_invalid_service_or_graph_input_is_rejected(changes):
    entry = {**node("last", "gpu", 1), **changes}
    with pytest.raises(ValueError):
        compose_prefill_chunks([chunk("a", [entry])])
