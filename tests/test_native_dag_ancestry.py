"""Bounded native reachability keeps every authored edge, without DFS loss."""
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.contracts import _PreparedExecutionStage
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.native_window_producer import (_native_dag_ancestry,
    _native_dag_precedes, _busy_segment_phase_bound, summarize_native_stages)
from test_hbm_compiled_control import configuration
from test_native_window_producer import task


def test_original_diamond_and_unrelated_branch_are_exact_even_in_reverse_enumeration():
    tasks = (task("a"), task("b", dependencies=("a",)), task("c", dependencies=("a",)),
             task("d", dependencies=("b", "c")), task("unrelated"))
    reach, proof = _native_dag_ancestry(tuple(reversed(tasks)))
    assert proof["qualified"] and proof["known_edge_count"] == 4
    assert all(_native_dag_precedes(reach, identity, "d") for identity in ("a", "b", "c", "d"))
    assert not _native_dag_precedes(reach, "b", "c")
    assert not _native_dag_precedes(reach, "unrelated", "d")
    assert not _native_dag_precedes(reach, "d", "a")


def test_missing_parents_cannot_supply_an_unknown_path_or_join_unrelated_branches():
    reach, proof = _native_dag_ancestry((task("a", dependencies=("unknown",)),
        task("b", dependencies=("unknown",)), task("c", dependencies=("a",))))
    assert proof["missing_parent_count"] == 1
    assert _native_dag_precedes(reach, "a", "c")
    assert not _native_dag_precedes(reach, "unknown", "c")
    assert not _native_dag_precedes(reach, "a", "b")


@pytest.mark.parametrize("tasks,reason", (
    ((task("a"), task("a")), "duplicate"),
    ((task("a", dependencies=("b",)), task("b", dependencies=("a",))), "cycle"),
    ((task("a", dependencies=("a",)),), "cycle"),
    ((NS(task_id="a", dependencies=()),), "unknown"),
))
def test_unknown_or_cyclic_graph_cannot_supply_an_exact_reachability_proof(tasks, reason):
    reach, proof = _native_dag_ancestry(tasks)
    assert reach is None and not proof["qualified"] and reason in proof["reason"]


def test_layer_limit_supports_a_full_4096_node_path_and_declines_a_larger_graph():
    tasks = tuple(task(str(index), dependencies=() if not index else (str(index-1),))
                  for index in range(4096))
    reach, proof = _native_dag_ancestry(tasks)
    assert proof["qualified"] and _native_dag_precedes(reach, "0", "4095")
    assert not _native_dag_precedes(reach, "4095", "0")
    assert _native_dag_ancestry((*tasks, task("extra")))[0] is None


def test_deep_original_dependency_path_no_longer_exhausts_resource_order_search():
    config = configuration(refresh_policy="all_bank", refresh_interval_ns=3900., refresh_cycle_ns=40.)
    metadata = {"physical_owner": "memory", "physical_memory_config": config,
        "memory_accesses": ({"operation": "read", "byte_count": 32,
            "physical_owner": "memory", "resource_id": "memory"},)}
    tasks = [task("0", 0, resources=("gpu0",), metadata=metadata)]
    tasks.extend(task(str(index), 0, (str(index-1),)) for index in range(1, 90))
    tasks.append(task("90", 0, ("89",), ("gpu0",), metadata))
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., tuple(tasks))
    summary = summarize_native_stages((stage,), NS(), NS(timeline=ResourceTimeline(), runtimes={}, allocators={}))
    assert summary["native_DAG_ancestry_proof"]["qualified"]
    assert summary["native_resource_order_proof"][0]["resource_orders_proven"]
    assert summary["ordering_reachability_nodes_examined"] == 0


def test_many_closed_segments_keep_all_original_same_owner_isolation_proofs():
    rows, bills = [], {}
    for rank in range(8):
        previous = None
        for index in range(45):
            identity = f"rank{rank}:{index}"
            item = task(identity, rank, () if previous is None else (previous,))
            owner = f"memory{rank}"
            rows.append((item, 100., {"@physical-owner:"+owner}))
            bills[identity] = {"refresh_ns": 100., "terms": {owner: [(3900., 100., 10.)]},
                "qualified": True, "other_physical": False, "nonphysical_upper_ns": 0.}
            barrier = identity+":native_NAND_wait"
            rows.append((task(barrier, rank, (identity,)), 0., {f"gpu{rank}"}))
            bills[barrier] = {"refresh_ns": 0., "terms": {}, "qualified": False,
                "other_physical": True, "nonphysical_upper_ns": 2000.}
            previous = barrier
    _bound, proof = _busy_segment_phase_bound(rows, bills,
        owner_ranks={f"memory{rank}": {rank} for rank in range(8)})
    segments = proof["segments"]
    assert len(segments) == 8*45 and all(row["qualified"] for row in segments)
    assert all(row["task_count"] == 1 for row in segments)  # No wait was contracted away.


def test_cyclic_graph_refuses_summary_and_busy_segment_proof():
    tasks = (task("a", 0, ("b",)), task("b", 0, ("a",)))
    stage = _PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., tasks)
    summary = summarize_native_stages((stage,), NS())
    assert "native_DAG_contains_a_cycle" in summary["unsupported"]
    assert not summary["native_resource_order_proof"][0]["closed_hbm_busy_segments"]["qualified"]


def test_independent_root_cannot_share_the_first_Tasks_unbounded_external_arrival_wait():
    rows = [(task("external", 1), 0., {"gpu1"}),
        (task("late", 0, ("external",)), 100., {"@physical-owner:memory"}),
        (task("early", 0), 100., {"@physical-owner:memory"})]
    bills = {"external": {"refresh_ns": 0., "terms": {}, "qualified": False,
        "other_physical": False, "nonphysical_upper_ns": 1e9}}
    for identity in ("late", "early"):
        bills[identity] = {"refresh_ns": 100., "terms": {"memory": [(3900., 100., 10.)]},
            "qualified": True, "other_physical": False, "nonphysical_upper_ns": 0.}
    bound, proof = _busy_segment_phase_bound(rows, bills, owner_ranks={"memory": {0}})
    assert [row["task_count"] for row in proof["segments"]] == [1, 1]
    assert all(not row["qualified"] for row in proof["segments"])
    assert bound == 200.  # Both real arrivals retain their own residual REF.
