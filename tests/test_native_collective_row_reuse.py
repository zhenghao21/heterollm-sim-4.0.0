"""Row reuse needs the real allocator replay and one row per physical bank."""
import copy
from dataclasses import replace

import pytest

from heterollm_sim.contracts import (_PreparedExecutionStage, _PreparedExecutionTask,
    ResourceDemand, TaskCategory, TaskSpec)
from heterollm_sim.native_collective_row_reuse import (_reopen_terms,
    native_allocation_scope)
from heterollm_sim.native_layer_sampling import _independent_kernel
from test_native_collective_refresh_bound import bound
from test_native_refresh_source import source

GENERATION = 7


def _io(identity, deps, owner, config, direction, buffer_id, common, size=4096):
    metadata = {**common, "memory_direction": direction, "physical_owner": owner,
        "physical_memory_config": config,
        "memory_accesses": ({"physical_owner": owner, "resource_id": owner,
            "operation": direction, "byte_count": size, "buffer_id": buffer_id,
            "offset_bytes": 0, "address_source": "stable_buffer_tensor_offset",
            "allocation_generation": GENERATION, "allocation_size_bytes": size},),
        "physical_allocations": ({"buffer_id": buffer_id, "size_bytes": size,
            "generation": GENERATION, "physical_owner": owner, "inferred": False},)}
    return _PreparedExecutionTask(identity, deps, ("cohort-r",), (),
        category=TaskCategory.MEMORY, metadata=metadata)


def buffer_ring(config, owners, *, name="allreduce", dependencies=(), link_ns=504.551111, sizes=None):
    tasks, previous = [], tuple(dependencies)
    for number in range(2*(len(owners)-1)):
        writes = []
        for index, owner in enumerate(owners):
            target = owners[(index+1) % len(owners)]
            prefix = f"{name}.round{number:02d}.tx{index:02d}"
            common = {"collective_name": name, "collective_kind": "all_reduce",
                "event_kind": "collective_transfer", "algorithm": "ring", "layer_id": "layer-000"}
            size = 4096 if sizes is None else sizes[index]
            read = _io(prefix+"."+owner+".read", previous, owner, config, "read",
                       "transfer:"+prefix, common, size)
            link = _PreparedExecutionTask(prefix+".link", (read.task_id,), ("cohort-r",),
                (ResourceDemand("link."+owner+"."+target, link_ns),),
                category=TaskCategory.COMMUNICATION,
                metadata={"event_kind": "collective_transfer", "layer_id": "layer-000"})
            write = _io(prefix+"."+target+".write", (link.task_id,), target, config, "write",
                        "transfer:"+prefix, common, size)
            tasks.extend((read, link, write))
            writes.append(write.task_id)
        barrier = _PreparedExecutionTask(f"{name}.round{number:02d}.barrier",
            tuple(writes), ("cohort-r",), (),
            metadata={"event_kind": "collective_barrier", "layer_id": "layer-000"})
        tasks.append(barrier)
        previous = (barrier.task_id,)
    return tuple(tasks)


def fixture(ranks=2, extra=(), link_ns=504.551111, sizes=None):
    data = source(arrival=3900.)
    runtime, config, physical, _core = data
    owners = ("hbm",)+tuple("hbm"+str(index) for index in range(1, ranks))
    for owner in owners[1:]:
        physical.runtime(config, owner)
    tasks = tuple(extra(config, owners)) if extra else ()
    tasks += buffer_ring(config, owners, dependencies=tuple(task.task_id for task in tasks),
                         link_ns=link_ns, sizes=sizes)
    return data, {owner: {index} for index, owner in enumerate(owners)}, tasks


def scope_for(data, tasks):
    stage = _PreparedExecutionStage("stage", 0, (), ("cohort-r",), "gpu", 0., tasks)
    composition = {"qualified": True, "nonserial_non_layer_groups": []}
    return native_allocation_scope((stage,), composition, {}, {},
                                   data[0]._execution_resource_kernel)


def reuse(proof):
    collective, = proof["collectives"]
    return collective["closed_collective_row_reuse"]


def observed_bases(data, tasks):
    """Execute on an independent native kernel copy and record real bases."""
    kernel, _executor = _independent_kernel(data[0]._execution_resource_kernel)
    origin = max(kernel.makespan_ns, max(data[2].timeline.ready_ns.values(), default=0.))
    specs = tuple(TaskSpec(task.task_id, "cohort-r", task.task_id, task.category,
        task.dependencies, task.demands, earliest_start_ns=origin, metadata=task.metadata)
        for task in tasks)
    kernel.submit(specs)
    kernel._index_physical_allocation_uses(specs)
    bases = {}
    while (event := kernel.step()) is not None:
        for access in event.task.metadata.get("memory_accesses", ()):
            allocation = kernel.physical_runtime.allocators[access["physical_owner"]].get_allocation(
                access["buffer_id"], GENERATION)
            bases[event.task.task_id] = (access["physical_owner"], event.task.metadata["memory_direction"],
                                         allocation.base_address)
    return bases


@pytest.mark.parametrize("ranks", (2, 8))
def test_replayed_first_fit_bases_contain_native_execution_and_reduce_rows(ranks):
    data, owners, tasks = fixture(ranks)
    plain, plain_proof, _ = bound(data, owners, tasks)
    value, proof, _ = bound(data, owners, tasks, allocation_scope=scope_for(data, tasks))
    row = reuse(proof)
    assert row["qualified"] and row["applied"], row
    assert value < plain
    assert proof["collectives"][0]["row_and_cache_phase_upper_ns"] < row["original_row_phase_upper_ns"]
    assert reuse(plain_proof)["applied"] is False
    for owner, facts in row["owners"].items():
        assert facts["read_base"] == 0 and facts["write_bases"] == [0, 4096]
        assert facts["single_row_per_physical_bank"]
        # Round 0 keeps the complete guard; later IOs only pay reopen terms.
        assert facts["reduced_row_guard_ns"]["read"] < 270.24
    actual = observed_bases(data, tasks)
    for owner, direction, base in actual.values():
        facts = row["owners"][owner]
        assert base in ({facts["read_base"]} if direction == "read" else set(facts["write_bases"]))
    assert not proof["physical_state_advanced"]


def test_second_write_range_on_a_conflicting_row_is_rejected():
    data, owners, tasks = fixture()
    # Leave only one 4 KiB hole below the next row period on every owner;
    # the second write candidate then lands on the same banks, next row.
    for owner in owners:
        allocator = data[2].allocators[owner]
        period = data[1].lane_count*data[1].bank_count*data[1].row_bytes
        allocator.allocate("source-filler", period-4096, 0, address=4096)
    value, proof, _ = bound(data, owners, tasks, allocation_scope=scope_for(data, tasks))
    row = reuse(proof)
    assert row["qualified"] is False
    assert row["reason"] == "collective_row_reuse_bank_row_conflict"
    assert proof["collectives"][0]["row_and_cache_phase_upper_ns"] == bound(data, owners, tasks)[1][
        "collectives"][0]["row_and_cache_phase_upper_ns"]


def test_unordered_live_lifetime_at_an_allocation_is_rejected():
    def unordered(config, owners):
        common = {"layer_id": "layer-000", "event_kind": "gpu_memory"}
        return (_io("left", (), owners[0], config, "write", "left-buffer", common),
                _io("right", (), owners[0], config, "write", "right-buffer", common))
    data, owners, tasks = fixture(extra=unordered)
    _value, proof, _ = bound(data, owners, tasks, allocation_scope=scope_for(data, tasks))
    assert reuse(proof)["reason"].startswith("collective_row_reuse_unordered_buffer_lifetime")


@pytest.mark.parametrize("mutation, reason", (
    (lambda md: {**md, "layer_id": "layer-001"}, "buffer_users_cross_layers_or_unseparated_non_layer"),
    (lambda md: {**md, "physical_allocations": ({**md["physical_allocations"][0], "generation": 0},),
        "memory_accesses": ({**md["memory_accesses"][0], "allocation_generation": 0},)},
        "persistent_aliased_explicit_or_resident_buffer"),
    (lambda md: {**md, "physical_allocations": ({**md["physical_allocations"][0],
        "allocation_arena": "resident"},)}, "persistent_aliased_explicit_or_resident_buffer"),
))
def test_cohort_scope_rejects_surviving_or_shared_buffers(mutation, reason):
    data, owners, tasks = fixture()
    # The round-0 write shares its buffer identity with that round's read on
    # another owner only; mutate one owner-local user of hbm's read buffer.
    target = next(task for task in tasks if task.task_id.endswith("tx00.hbm.read"))
    duplicate = replace(target, task_id="duplicate-user", dependencies=(tasks[-1].task_id,),
                        metadata=mutation(target.metadata))
    scope = scope_for(data, tasks+(duplicate,))
    assert scope["owners"]["hbm"]["qualified"] is False
    assert reason in scope["owners"]["hbm"]["reasons"]


def test_preexisting_source_buffer_rejects_owner_scope():
    data, owners, tasks = fixture()
    data[2].allocators["hbm"].allocate("transfer:allreduce.round00.tx00", 4096, GENERATION)
    scope = scope_for(data, tasks)
    assert "cohort_buffer_already_registered_in_source" in scope["owners"]["hbm"]["reasons"]
    _value, proof, _ = bound(data, owners, tasks, allocation_scope=scope)
    assert reuse(proof)["reason"].startswith("collective_row_reuse_owner_lifetime_not_closed")


def test_allocation_only_owner_user_inside_ring_rejects_the_closed_collective():
    data, owners, tasks = fixture()
    data[2].runtime(data[1], "hbm-side")
    common = {"layer_id": "layer-000", "event_kind": "gpu_memory"}
    side = _io("side", (), "hbm-side", data[1], "write", "side-buffer", common)
    # Its only access is elsewhere; the extent it declares on a ring owner
    # still changes that owner's free list during the collective.
    side = replace(side, metadata={**side.metadata, "physical_allocations": (
        *side.metadata["physical_allocations"], {"buffer_id": "side-extent", "size_bytes": 4096,
        "generation": GENERATION, "physical_owner": "hbm", "inferred": False})})
    _value, proof, _ = bound(data, owners, tasks+(side,), allocation_scope=scope_for(data, tasks+(side,)))
    assert proof["qualified"] is False
    assert proof["rejections"][0]["reason"].startswith("common_REF_original_subgraph_not_closed")


def test_reopen_requires_REF_to_expire_every_command_history():
    config = source()[1]
    terms = _reopen_terms(config)
    assert terms["refresh_cycle_ns"] > terms["REF_expired_history_upper_ns"]
    # A validated HBM4 configuration cannot express this; bypass only here.
    short = copy.copy(config)
    object.__setattr__(short, "refresh_cycle_ns", terms["REF_expired_history_upper_ns"])
    with pytest.raises(ValueError, match="REF_does_not_expire_history"):
        _reopen_terms(short)


@pytest.mark.parametrize("operation", ("read", "write"))
@pytest.mark.parametrize("size", (32, 4096, 32768, 65536, 98304))
@pytest.mark.parametrize("offset", (0, 32, 4064))
def test_conflict_free_repeat_bounds_cover_native_closed_hit_and_REF_states(operation, size, offset):
    from exploring_hbf_paper_config import build_paper_hbm_config
    from heterollm_sim.dram_core import DramCore
    from heterollm_sim.memory_types import AccessRequest
    from heterollm_sim.native_collective_row_reuse import _range_geometry, _repeat_io_bounds
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper

    config = build_paper_hbm_config("HBM4")
    bounds = _repeat_io_bounds(config, _reopen_terms(config), size, operation,
                               [_range_geometry(config, offset, size)])
    blocked = native_hbm_refresh_service_upper(config, size, operation, independent_channels=True,
        native_refresh_reachable=True)["refresh_epoch_upper_ns"]
    interval = config.refresh_interval_ns
    core = DramCore(config, capture_details=False)

    def submit(name, arrival):
        return core.submit(AccessRequest(name, operation, offset, size, arrival_ns=arrival))

    closed = submit("closed", 4*interval+700.)
    hit = submit("hit", closed.completion_ns+50.)
    # A whole REF epoch passes: every bank is closed again by native REF.
    after_ref = submit("after-REF", 6*interval+700.)
    # REF due inside the IO: one blackout is charged separately.
    straddle = submit("straddle", 8*interval-5.)
    for result in (closed, hit, after_ref):
        assert result.latency_ns <= bounds["active_upper_ns"]
    assert straddle.latency_ns <= bounds["active_upper_ns"]+blocked
    assert closed.latency_ns-hit.latency_ns <= bounds["row_guard_ns"]
    assert after_ref.latency_ns-hit.latency_ns <= bounds["row_guard_ns"]


def test_cohort_single_row_entry_requires_one_row_period_and_clean_source():
    from heterollm_sim.native_collective_row_reuse import _cohort_single_row_entry

    data = source()
    config, physical = data[1], data[2]
    period = config.lane_count*config.bank_count*config.row_bytes
    facts = {"source_highest_end_byte": 0, "max_group_cumulative_extent_bytes": 8192}
    io = [{"candidates": [(0, 4096)]}]
    # The fixture's real 32 KiB source read left banks open on row 0.
    assert _cohort_single_row_entry("hbm", config, facts, physical, io) == (0, None)
    assert _cohort_single_row_entry("hbm", config, {**facts,
        "max_group_cumulative_extent_bytes": period+1}, physical, io)[1] == \
        "cumulative_extent_bound_exceeds_one_row_period"
    from heterollm_sim.memory_types import AccessRequest
    data[3].submit(AccessRequest("other-row", "read", period, 32, arrival_ns=1e6))
    assert _cohort_single_row_entry("hbm", config, facts, physical, io)[1] == "source_bank_open_on_other_row"


def _layer_reuse(data, tasks):
    from heterollm_sim.native_collective_row_reuse import native_layer_row_reuse
    from heterollm_sim.native_window_producer import _native_dag_ancestry
    ancestry, _ = _native_dag_ancestry(tasks)
    owners = {access["physical_owner"] for task in tasks for access in task.metadata.get("memory_accesses", ())}
    return native_layer_row_reuse(tasks, exact_ancestry=ancestry, physical_runtime=data[2],
        allocation_scope=scope_for(data, tasks), configs={owner: data[1] for owner in owners},
        source_qualified={owner: True for owner in owners})


def test_layer_row_reuse_covers_every_IO_only_when_no_image_can_hold_another_row():
    data, _owners, tasks = fixture(ranks=2)
    clean = _layer_reuse(data, tasks)
    assert not clean["rejections"]
    assert all(row["cohort_entry_row"] == 0 and row["first_touch_io_count"] == 0
               for row in clean["owners"].values())
    io_count = sum(len(task.metadata.get("memory_accesses", ())) for task in tasks)
    assert len(clean["io_guards"]) == len(clean["io_active_uppers"]) == io_count
    # A source bank open on another row: the first touches keep their guard.
    from heterollm_sim.memory_types import AccessRequest
    period = data[1].lane_count*data[1].bank_count*data[1].row_bytes
    data[3].submit(AccessRequest("other-row", "read", period, 4096, arrival_ns=1e6))
    dirty = _layer_reuse(data, tasks)
    hbm = dirty["owners"]["hbm"]
    assert hbm["cohort_entry_row"] is None and hbm["cohort_entry_row_reason"] == "source_bank_open_on_other_row"
    assert 0 < hbm["first_touch_io_count"] < hbm["io_count"]
    first_round = [task.task_id for task in tasks if ".round00." in task.task_id
                   and task.metadata.get("memory_accesses") and task.metadata["physical_owner"] == "hbm"]
    assert any((task_id, 0) not in dirty["io_guards"] for task_id in first_round)


def _timed_bases(data, tasks, origin):
    kernel, _executor = _independent_kernel(data[0]._execution_resource_kernel)
    specs = tuple(TaskSpec(task.task_id, "cohort-r", task.task_id, task.category,
        task.dependencies, task.demands, earliest_start_ns=origin, metadata=task.metadata)
        for task in tasks)
    kernel.submit(specs)
    kernel._index_physical_allocation_uses(specs)
    rows = {}
    while (event := kernel.step()) is not None:
        for access in event.task.metadata.get("memory_accesses", ()):
            allocation = kernel.physical_runtime.allocators[access["physical_owner"]].get_allocation(
                access["buffer_id"], GENERATION)
            rows[event.task.task_id] = (event.start_ns, event.end_ns, allocation.base_address)
    return rows


@pytest.mark.parametrize("shift", (0., 150., 600., 1300., 2100., 3000., 3700.))
def test_native_writes_reuse_the_read_extent_outside_REF_windows(shift):
    from heterollm_sim.native_collective_row_reuse import _range_geometry, _repeat_io_bounds
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper

    data, owners, tasks = fixture(ranks=8)
    config = data[1]
    blocked = native_hbm_refresh_service_upper(config, 4096, "read", independent_channels=True,
        native_refresh_reachable=True)["refresh_epoch_upper_ns"]
    read_upper = _repeat_io_bounds(config, _reopen_terms(config), 4096, "read",
                                   [_range_geometry(config, 0, 4096)])["active_upper_ns"]
    interval = config.refresh_interval_ns
    origin = 5*interval+shift
    rows = _timed_bases(data, tasks, origin)
    rounds = sorted({task.task_id.split(".round")[1][:2] for task in tasks if ".round" in task.task_id})
    second_writes = 0
    for number in rounds:
        reads = [value for key, value in rows.items() if f".round{number}." in key and key.endswith(".read")]
        writes = [value for key, value in rows.items() if f".round{number}." in key and key.endswith(".write")]
        start = min(value[0] for value in reads)
        # Every read lands on B1; a second write range needs a REF epoch
        # inside this round's read window.
        assert {value[2] for value in reads} == {0}
        if any(value[2] != 0 for value in writes):
            second_writes += 1
            epoch = (start-blocked)//interval*interval
            assert any(start-blocked <= epoch+k*interval <= start+read_upper for k in range(3))
    assert second_writes <= len(rounds)


def test_epoch_reopen_bound_requires_its_round_lemmas_and_never_exceeds_the_round_bound():
    from heterollm_sim.native_collective_refresh_bound import _epoch_reopen_bound

    config = source()[1]
    owners = {"hbm": {"hit_row_guard_ns": {"read": 5., "write": 6.},
                      "single_reopen_ns": {"read": 50., "write": 30.}}}
    good = _epoch_reopen_bound(owners, {"hbm": config}, 170., 14, 0., epochs=4, blocked=598.8,
                               read_active_upper=112., link_lower=504.)
    assert good["qualified"] and good["row_phase_upper_ns"] >= 170.+13*11.+12*80.
    assert _epoch_reopen_bound(owners, {"hbm": config}, 170., 14, 0., epochs=4, blocked=598.8,
        read_active_upper=600., link_lower=504.)["reason"] == "read_may_outlive_neighbour_link_outside_REF"
    assert _epoch_reopen_bound(owners, {"hbm": config}, 170., 14, 0., epochs=4,
        blocked=config.refresh_cycle_ns+60., read_active_upper=50., link_lower=50.)["reason"] == \
        "REF_blackout_may_admit_two_round_starts"
    data, ranks, tasks = fixture(ranks=8)
    _value, proof, _ = bound(data, ranks, tasks, allocation_scope=scope_for(data, tasks))
    reuse = proof["collectives"][0]["closed_collective_row_reuse"]
    epoch = reuse["REF_epoch_reopen_bound"]
    assert epoch["qualified"], epoch
    assert reuse["row_phase_upper_ns"] <= epoch["row_phase_upper_ns"]
    assert reuse["row_phase_upper_ns"] <= reuse["original_row_phase_upper_ns"]


def test_short_links_really_move_writes_to_the_second_extent_and_reject_the_epoch_bound():
    # One owner's read is far slower than its upstream neighbour's.
    data, ranks, tasks = fixture(ranks=8, link_ns=1., sizes=(65536,)+(4096,)*7)
    rows = _timed_bases(data, tasks, 5*data[1].refresh_interval_ns+300.)
    # Without a whole link between them, a write can register while its
    # owner's read is still live: the allocator then really returns B2.
    assert any(key.endswith(".write") and value[2] != 0 for key, value in rows.items())
    _value, proof, _ = bound(data, ranks, tasks, allocation_scope=scope_for(data, tasks))
    epoch = proof["collectives"][0]["closed_collective_row_reuse"]["REF_epoch_reopen_bound"]
    # Both round lemmas fail for a 1 ns link; either one must reject.
    assert epoch["qualified"] is False
    assert epoch["reason"] in {"read_may_outlive_neighbour_link_outside_REF",
                               "REF_blackout_may_admit_two_round_starts"}


def _epoch_inputs(link=504.551, history_free_write=23.04):
    from types import SimpleNamespace
    owners = {"hbm": {"REF_expired_history_upper_ns": 48.64,
        "hit_active_upper_ns": {"read": 87.04, "write_history_free": history_free_write}}}
    waves = [{"read_wave_upper_ns": 112.56, "link_wave_upper_ns": link, "write_wave_upper_ns": 97.20,
              "complete_serial_wave_upper_ns": 112.56+link+97.20} for _ in range(14)]
    return owners, {"hbm": SimpleNamespace(refresh_cycle_ns=530.)}, waves


@pytest.mark.parametrize("link,epochs,expected", ((504.551, 4, 3), (520., 4, 4), (504.551, 2, 2)))
def test_refined_epoch_count_is_the_largest_consistent_fixed_point_and_never_exceeds_the_old(link, epochs, expected):
    from heterollm_sim.native_collective_refresh_bound import _refined_epoch_count
    owners, configs, waves = _epoch_inputs(link)
    refined = _refined_epoch_count(owners, configs, waves, epochs, interval=3900., blocked=598.8,
                                   read_active_upper=112.56, link_lower=link)
    assert refined["qualified"] and refined["same_round_write_history_free"]
    assert refined["REF_epoch_count_upper"] == expected <= epochs


def test_refined_epoch_count_falls_back_without_its_lemmas_or_hit_bounds():
    import math
    from heterollm_sim.native_collective_refresh_bound import _refined_epoch_count
    owners, configs, waves = _epoch_inputs()
    kwargs = {"interval": 3900., "blocked": 598.8, "read_active_upper": 112.56}
    assert _refined_epoch_count(owners, configs, waves, 4, **kwargs, link_lower=60.) == {
        "qualified": False, "REF_epoch_count_upper": 4, "reason": "epoch_round_lemmas_unproven"}
    missing = {"hbm": {key: value for key, value in owners["hbm"].items() if key != "hit_active_upper_ns"}}
    assert _refined_epoch_count(missing, configs, waves, 4, **kwargs, link_lower=504.551)[
        "reason"] == "hit_active_upper_unavailable"
    # A link shorter than read+history keeps the write's command history.
    short = _refined_epoch_count(owners, configs, waves, 4, **kwargs, link_lower=150.)
    assert short["same_round_write_history_free"] is False
    assert short["hit_write_active_upper_ns"] == math.nextafter(23.04+48.64, math.inf)


def test_real_ring_uses_the_refined_epoch_count_for_shared_REF():
    data, ranks, tasks = fixture(ranks=8)
    _value, proof, _ = bound(data, ranks, tasks, allocation_scope=scope_for(data, tasks))
    group = proof["collectives"][0]
    refinement = group["closed_collective_row_reuse"]["REF_epoch_count_refinement"]
    assert refinement["qualified"], refinement
    assert refinement["REF_epoch_count_upper"] <= refinement["previous_epoch_count_upper"]
    assert group["common_REF_epoch_count_upper"] == refinement["REF_epoch_count_upper"]
    short_data, short_ranks, short_tasks = fixture(ranks=8, link_ns=1., sizes=(65536,)+(4096,)*7)
    _value, short_proof, _ = bound(short_data, short_ranks, short_tasks,
                                   allocation_scope=scope_for(short_data, short_tasks))
    short_reuse = short_proof["collectives"][0]["closed_collective_row_reuse"]
    assert short_reuse.get("REF_epoch_count_refinement", {}).get("qualified") is not True
