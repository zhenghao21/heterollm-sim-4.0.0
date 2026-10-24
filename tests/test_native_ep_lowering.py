"""Attention data parallelism: per-EP-group lowering, MoE coupling, ep=1 identity.

The pinned ep=1 digests cover task count, names, categories, dependency order,
demands and canonical metadata; they were measured on the pre-change tree and
on the single-group lowering in two separate processes, which agree exactly.
"""
from dataclasses import fields, is_dataclass, replace
import hashlib
from typing import Mapping

import pytest

import native_toy
from heterollm_sim import planner
from heterollm_sim.communication import TopologyRouter, plan_collective
from heterollm_sim.final_layer_output_selection import FinalLayerOutputSelection
from heterollm_sim.ir import MTPPolicy
from heterollm_sim.native_layer_sampling import _graph_facts, layer_inventory
from heterollm_sim.planner import (
    compile_serving_cohort_schedule,
    native_kv_ingress_inventory,
)
from heterollm_sim.serving import BatchCohort, BatchItem


def _canonical(value):
    """Hash-seed independent form: sorted mappings/sets, dataclass fields."""

    if isinstance(value, Mapping):
        return tuple(sorted((_canonical(key), _canonical(item))
                            for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_canonical(item) for item in value))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__,
                tuple((field.name, _canonical(getattr(value, field.name)))
                      for field in fields(value)))
    return repr(value)


def _structure_digest(tasks):
    """Compact digest of everything that shapes the simulation graph."""

    index = {task.task_id: position for position, task in enumerate(tasks)}
    rows = [
        (
            task.name,
            str(task.category),
            tuple(index.get(dependency, dependency) for dependency in task.dependencies),
            tuple(
                (demand.resource_id, demand.service_ns, demand.work_units, demand.bytes_moved)
                for demand in task.demands
            ),
            _canonical(task.metadata),
        )
        for task in tasks
    ]
    return len(rows), hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def _item(request_id, phase, tokens, context, *, owner):
    """One batch item; group 0 is the default so a single-group batch matches."""

    return BatchItem(request_id, phase, tokens, context, main_tokens=tokens,
                     draft_tokens=0, verifier_tokens=0, committed_tokens=tokens,
                     logit_tokens=1, **({"ep_group": owner} if owner else {}))


def _cohort(scenario, kind, batch, *, ep_degree=1):
    ids = [request.request_id for request in scenario.workload.requests][:batch]
    if kind == "decode":
        return BatchCohort("ep-decode", "decode", 0.0, tuple(
            _item(ids[i], "decode", 1, 5 + 3 * i, owner=i % ep_degree)
            for i in range(batch)
        ))
    return BatchCohort("ep-mixed", "mixed", 0.0, tuple(
        [_item(ids[0], "prefill", 3, 0, owner=0 % ep_degree)]
        + [_item(ids[i], "decode", 1, 6 + i, owner=i % ep_degree)
           for i in range(1, batch)]
    ))


def _maverick(*, tp=2, ep=1, batch_size=4):
    return native_toy.maverick_pattern_scenario(batch_size=batch_size, tp=tp, ep=ep)


def _tasks(scenario, kind, batch, *, ep_degree=1):
    return compile_serving_cohort_schedule(
        scenario, _cohort(scenario, kind, batch, ep_degree=ep_degree)
    ).tasks


def test_ep1_toy_lowering_is_byte_identical_to_the_single_group_graph():
    build = lambda: native_toy.toy_scenario(layer_count=2, batch_size=4)
    assert _structure_digest(_tasks(build(), "decode", 4)) == (241, "cb7f83abda8671fc")
    assert _structure_digest(_tasks(build(), "mixed", 4)) == (453, "1b226b301f496788")


def test_ep1_maverick_lowering_is_byte_identical_to_the_single_group_graph():
    build = lambda: _maverick()
    assert _structure_digest(_tasks(build(), "decode", 4)) == (1159, "0608110c342865a8")
    assert _structure_digest(_tasks(build(), "mixed", 4)) == (2225, "04cf4f137dde6f40")


def test_ep1_lowering_has_no_attention_data_parallel_scaffolding():
    for scenario in (native_toy.toy_scenario(layer_count=2, batch_size=4), _maverick()):
        for kind in ("decode", "mixed"):
            tasks = _tasks(scenario, kind, 4)
            names = [task.name for task in tasks]
            assert not [name for name in names if ".ep" in name]
            assert not [task for task in tasks if task.metadata.get("event_kind") == "layer_barrier"]
            assert not [task for task in tasks if task.metadata.get("collective_pairwise")]
            assert not [task for task in tasks
                        if {key for key in ("ep_group_rows", "ep_groups", "ep_partitions")
                            if key in task.metadata}]
            # Every embedding join is invocation-level (no group suffix) and no
            # cross-group layer barrier exists.
            assert [name for name in names if name.endswith(".embedding.complete")]
            assert not [name for name in names if name.endswith(".layer_complete")]


def test_ep2_partitions_split_the_invocation_by_owner():
    scenario = _maverick(ep=2, batch_size=4)
    plan = planner._parallel_plan(scenario)
    assert (plan.tp_degree, plan.pp_degree, plan.ep_degree) == (2, 1, 2)
    assert [rank.rank for rank in plan.tp_group(0, 0)] == [0, 1]
    assert [rank.rank for rank in plan.tp_group(0, 1)] == [2, 3]
    group, = planner._serving_invocation_groups(
        scenario, _cohort(scenario, "decode", 4, ep_degree=2)
    )
    partitions = group.ep_partitions(plan.ep_degree)
    assert [partition.ep_rank for partition in partitions] == [0, 1]
    assert [partition.token_batch for partition in partitions] == [2, 2]
    assert [partition.request_ids for partition in partitions] == [
        tuple(lane.request_id for lane in partition.lanes) for partition in partitions
    ]
    assert {lane.ep_group for partition in partitions for lane in partition.lanes} == {0, 1}
    # The parent keeps the union of lanes and its own identity.
    assert group.token_batch == 4
    assert sum(len(partition.lanes) for partition in partitions) == len(group.lanes)


def test_ep_group_names_are_suffixed_only_when_the_plan_has_several_groups():
    for ep in (1, 2):
        names = [task.name for task in _tasks(_maverick(ep=ep), "decode", 4, ep_degree=ep)]
        assert any(".ep000." in name for name in names) is (ep > 1)
        assert any(".ep001." in name for name in names) is (ep > 1)
        # The cross-group expert exchange and the layer barrier stay shared.
        assert not [name for name in names if ".ep" in name and ".expert_dispatch" in name]
        assert not [name for name in names if ".ep" in name and ".expert_combine" in name]
        assert not [name for name in names if ".ep" in name and name.endswith(".layer_complete")]


def test_ep2_kv_buffers_live_only_on_the_owner_groups_ranks():
    scenario = _maverick(ep=2, batch_size=4)
    plan = planner._parallel_plan(scenario)
    owners = {}
    for task in _tasks(scenario, "decode", 4, ep_degree=2):
        for row in task.metadata.get("physical_allocations", ()):
            buffer_id = str(row["buffer_id"])
            if buffer_id.startswith("kv:"):
                owner = row.get("physical_owner") or task.metadata["physical_owner"]
                owners.setdefault(buffer_id.split(":")[1], set()).add(owner)
    ids = [request.request_id for request in scenario.workload.requests][:4]

    def group_storage(ep_rank):
        # ``physical_owner`` names the KV pool on each rank's memory component.
        return {
            "{}.memory".format(planner._kv_components(scenario, rank)[0])
            for rank in plan.tp_group(0, ep_rank)
        }

    assert set(owners) == set(ids)
    for index, request_id in enumerate(ids):
        owner = index % plan.ep_degree
        # KV lives on the owner group's ranks and nowhere else.
        assert owners[request_id] == group_storage(owner)
        others = set().union(*(
            group_storage(ep_rank)
            for ep_rank in range(plan.ep_degree) if ep_rank != owner
        ))
        assert not owners[request_id] & others


def _pair_transfer_bytes(tasks, marker):
    """Transfer bytes of one pairwise all-to-all round, one entry per peer."""

    rows = {}
    for task in tasks:
        name = task.name
        if marker in name and ".round" in name and name.endswith(".link01"):
            head = name.split(marker + ".", 1)[0]
            rows.setdefault(head, []).append(sum(
                demand.bytes_moved for demand in task.demands
                if str(demand.resource_id).startswith("link.")
            ))
    return rows


def test_ep2_dispatch_and_combine_bytes_follow_the_paper_formula():
    scenario = _maverick(ep=2, batch_size=4)
    plan = planner._parallel_plan(scenario)
    tasks = _tasks(scenario, "decode", 4, ep_degree=2)
    layer = next(current for current in planner._execution_layers(scenario) if current.is_moe)
    rows_by_ep = (2, 2)
    send = tuple(
        planner._activation_bytes(
            layer, rows * layer.experts_per_token * layer.hidden_size, scenario=scenario
        )
        for rows in rows_by_ep
    )
    # The paper's per-peer bytes: ceil(local_routed_bytes / ep).
    chunk = -(-send[0] // plan.ep_degree)
    for marker in ("expert_dispatch", "expert_combine"):
        transfers = _pair_transfer_bytes(tasks, marker)
        assert transfers
        assert {value for values in transfers.values() for value in values} == {chunk}
    pairwise = [task.metadata for task in tasks
                if task.metadata.get("collective_pairwise") is True]
    assert pairwise
    assert {row["collective_pair_bytes_total"] for row in pairwise} == {
        sum(-(-value // plan.ep_degree) for value in send)
    }
    assert {row["ep_group_rows"] for row in pairwise} == {rows_by_ep}


def test_ep2_empty_group_still_joins_the_expert_exchange():
    scenario = _maverick(ep=2, batch_size=4)
    cohort = BatchCohort("ep-decode", "decode", 0.0, tuple(
        BatchItem(request.request_id, "decode", 1, 5 + 3 * index, main_tokens=1,
                  draft_tokens=0, verifier_tokens=0, committed_tokens=1, logit_tokens=1,
                  ep_group=0)
        for index, request in enumerate(scenario.workload.requests[:4])
    ))
    tasks = compile_serving_cohort_schedule(scenario, cohort).tasks
    names = [task.name for task in tasks]
    # The idle group lowers no dense path but still receives and computes.
    assert not [name for name in names if ".ep001." in name]
    assert [name for name in names if ".expert_dispatch." in name and name.endswith(".link01")]
    assert [name for name in names if ".expert_combine." in name and name.endswith(".link01")]
    pairwise = [task.metadata for task in tasks if task.metadata.get("collective_pairwise") is True]
    assert {row["ep_group_rows"] for row in pairwise} == {(4, 0)}
    layer = next(current for current in planner._execution_layers(scenario) if current.is_moe)
    send = planner._activation_bytes(
        layer, 4 * layer.experts_per_token * layer.hidden_size, scenario=scenario
    )
    assert {row["collective_pair_bytes_total"] for row in pairwise} == {-(-send // 2)}
    facts, _boundary = _graph_facts(tasks, layer_inventory(scenario, tasks))
    assert facts["qualified"] is True and facts["failures"] == []
    assert all(len(terminals) == 1 for terminals in facts["terminals"].values())


def test_ep2_has_exactly_one_cross_group_barrier_per_layer():
    scenario = _maverick(ep=2, batch_size=4)
    tasks = _tasks(scenario, "decode", 4, ep_degree=2)
    layers = [layer.layer_id for layer in planner._execution_layers(scenario)]
    barriers = [task for task in tasks if task.metadata.get("event_kind") == "layer_barrier"]
    assert [barrier.metadata["layer_id"] for barrier in barriers] == layers
    assert [barrier.name for barrier in barriers] == [
        "ep-decode.decode.{}.layer_complete".format(layer) for layer in layers
    ]
    facts, boundary = _graph_facts(tasks, layer_inventory(scenario, tasks))
    assert facts["qualified"] is True and facts["failures"] == []
    assert facts["nonserial_non_layer_groups"] == []
    assert facts["layer_order"] == layers
    assert all(len(terminals) == 1 for terminals in facts["terminals"].values())
    assert set(boundary) <= {"before_first_layer", "after_last_layer"}
    # The final-norm/lm_head suffix starts only after the last layer barrier.
    index = {task.task_id: task for task in tasks}
    pending = [task.task_id for task in tasks if ".lm_head_all_gather" in task.name]
    assert pending
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(index[current].dependencies)
    assert barriers[-1].task_id in seen


def test_ep2_ingress_inventory_matches_the_full_lowering_on_owner_ranks_only():
    scenario = _maverick(ep=2, batch_size=4)
    plan = planner._parallel_plan(scenario)
    for owner in range(plan.ep_degree):
        request = scenario.workload.requests[owner]
        history = request.prompt_tokens - 1
        item = BatchItem(request.request_id, "prefill", 1, history, kv_append_tokens=1,
                         kv_materialized_tokens=1, main_tokens=0, draft_tokens=0,
                         verifier_tokens=1, committed_tokens=0, logit_tokens=1,
                         completion_cursor=request.prompt_tokens, ep_group=owner)
        schedule = compile_serving_cohort_schedule(
            scenario, BatchCohort("inventory", "prefill", 0.0, (item,))
        )
        expected, contracts = {}, []
        for task in schedule.tasks:
            for row in task.metadata.get("physical_allocations", ()):
                if str(row["buffer_id"]).startswith("kv:" + request.request_id + ":"):
                    key = (row.get("physical_owner") or task.metadata["physical_owner"],
                           row["buffer_id"], row.get("generation", 0))
                    expected[key] = {**row, "physical_owner": key[0]}
            contract = task.metadata.get("generic_kv_storage")
            if contract is not None and contract not in contracts:
                contracts.append(contract)
        inventory = native_kv_ingress_inventory(
            scenario, request.request_id, history, ep_group=owner
        )
        actual = {(row["physical_owner"], row["buffer_id"], row.get("generation", 0)): row
                  for row in inventory["physical_allocations"]}
        assert actual == expected
        assert len(inventory["contracts"]) == len(contracts)
        assert all(contract in contracts for contract in inventory["contracts"])
        assert {key[0] for key in actual} == {
            "{}.memory".format(planner._kv_components(scenario, rank)[0])
            for rank in plan.tp_group(0, owner)
        }


def test_ep2_ingress_inventory_validates_the_owner_group():
    scenario = _maverick(ep=2, batch_size=4)
    request = scenario.workload.requests[0]
    with pytest.raises(ValueError, match="outside 0..1"):
        native_kv_ingress_inventory(
            scenario, request.request_id, request.prompt_tokens - 1, ep_group=2
        )
    with pytest.raises(ValueError, match="must be an integer"):
        native_kv_ingress_inventory(
            scenario, request.request_id, request.prompt_tokens - 1, ep_group=0.0
        )


def test_ep2_cohort_cache_key_separates_request_ownership():
    scenario = _maverick(ep=2, batch_size=2)
    ids = [request.request_id for request in scenario.workload.requests][:2]

    def key(owners):
        cohort = BatchCohort("ep-decode", "decode", 0.0, tuple(
            BatchItem(ids[index], "decode", 1, 5 + index, main_tokens=1, draft_tokens=0,
                      verifier_tokens=0, committed_tokens=1, logit_tokens=1,
                      ep_group=owners[index])
            for index in range(2)
        ))
        return planner._serving_cohort_cache_key(cohort)

    assert key((0, 1)) != key((1, 0))
    assert key((0, 1)) == key((0, 1))


def test_offline_request_partitions_put_every_row_on_the_owner_group():
    scenario = _maverick(ep=2, batch_size=2)
    plan = planner._parallel_plan(scenario)
    for index, request in enumerate(scenario.workload.requests[:2]):
        partitions = planner._offline_request_partitions(
            scenario, plan, request,
            token_batch=1, context_tokens=7, kv_read_tokens=6, kv_append_tokens=1,
            kv_materialized_tokens=1, logit_row_indices=(0,), committed_logit_rows=1,
        )
        owner = index % plan.ep_degree
        assert [partition.ep_rank for partition in partitions] == [0, 1]
        assert partitions[owner].token_batch == 1
        assert partitions[owner].logit_row_indices == (0,)
        assert [partition.token_batch for partition in partitions
                if partition.ep_rank != owner] == [0]


def test_ep2_rejects_features_that_only_exist_for_one_rank():
    scenario = _maverick(ep=2, batch_size=4)
    plan = planner._parallel_plan(scenario)
    router = planner._topology_router(scenario)
    builder = planner._TaskBuilder(scenario.workload.requests[0])
    selection = FinalLayerOutputSelection("llama", 2, (0,))
    with pytest.raises(ValueError, match="final output selection"):
        planner._compile_parallel_iteration(
            builder, scenario, plan, router,
            partitions=tuple(planner._EpPartition(
                index, 1, 5 + index, 4 + index, 1, 1) for index in range(plan.ep_degree)),
            output_selection=selection,
            phase="rejected",
            dependencies=(),
        )
    group, = planner._serving_invocation_groups(
        scenario, _cohort(scenario, "decode", 2, ep_degree=2))
    assert planner._serving_invocation_segment_binding(
        builder, scenario, plan, group, ()) is None
    assert builder.tasks == []


def test_ep2_validation_rejects_mtp_but_ep1_keeps_it():
    for ep, rejected in ((1, False), (2, True)):
        scenario = _maverick(ep=ep, batch_size=4)
        scenario = replace(scenario, workload=replace(scenario.workload, mtp=MTPPolicy()))
        errors = planner._validate_scenario_uncached(scenario).errors_en
        assert any("MTP lowering is single-rank" in error for error in errors) is rejected


def test_plan_collective_pair_bytes_drive_each_transfer():
    scenario = _maverick()
    router = TopologyRouter(scenario.hardware)
    plan = plan_collective(
        router, "all_to_all", ("gpu0", "gpu1"), 256,
        pair_bytes=((0, 128), (64, 0)),
    )
    assert plan.pair_bytes == ((0, 128), (64, 0))
    assert [(row.source_component, row.target_component, row.byte_count)
            for round_ in plan.rounds for row in round_.transfers] == [
        ("gpu0", "gpu1", 128), ("gpu1", "gpu0", 64)]
    assert plan_collective(
        router, "all_to_all", ("gpu0", "gpu1"), 0, pair_bytes=((0, 0), (0, 0))
    ).rounds == ()
    scalar = plan_collective(router, "all_to_all", ("gpu0", "gpu1"), 256)
    assert scalar.pair_bytes is None and scalar.rounds


def test_plan_collective_rejects_pair_bytes_outside_all_to_all():
    scenario = _maverick()
    router = TopologyRouter(scenario.hardware)
    with pytest.raises(ValueError, match="only valid for all_to_all"):
        plan_collective(router, "all_reduce", ("gpu0", "gpu1"), 16,
                        pair_bytes=((0, 8), (8, 0)))
    with pytest.raises(ValueError, match="matrix"):
        plan_collective(router, "all_to_all", ("gpu0", "gpu1"), 16,
                        pair_bytes=((0, 8),))


def test_ep2_collective_cache_separates_pair_matrices():
    scenario = _maverick(ep=2, batch_size=4)
    participants = ("gpu0", "gpu2")
    with planner._compilation_scope(scenario) as context:
        router = context.router()

        def collective(byte_count):
            return planner._collective_plan(
                scenario, router, "all_to_all", participants, 0,
                algorithm="ring", routing_policy="lowest_latency",
                pair_bytes=((0, byte_count), (byte_count, 0)),
            )

        assert planner._active_compilation_context(scenario) is context
        first = collective(10)
        second = collective(20)
        repeat = collective(10)
    assert [row.byte_count for round_ in first.rounds for row in round_.transfers] == [10, 10]
    assert [row.byte_count for round_ in second.rounds for row in round_.transfers] == [20, 20]
    assert repeat is first
