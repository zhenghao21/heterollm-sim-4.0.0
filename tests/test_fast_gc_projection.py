"""A GC forecast may use only a proved empty-victim reservoir and real IO costs."""
from copy import deepcopy
from dataclasses import replace
from itertools import product

import pytest

from heterollm_sim.fast_gc import (
    _calendar_carry_in, _host_page_cost, gc_service_bounds, project_zero_copy_gc,
    qualify_immutable_reads,
)
from heterollm_sim.hbf_staging import HbfStagingController, HbfStagingIO, PAGE
from heterollm_sim.memory_types import AccessRequest, Operation
from heterollm_sim.physical_presets import resolve_physical_config


def config():
    return resolve_physical_config("hbf_ocp_0_7", overrides={
        "channels": 1, "host_channels": 1, "planes_per_lun": 2,
        "blocks_per_plane": 4, "pages_per_block": 4,
        "capacity_bytes": 8 * 4 * PAGE})


def callbacks(programs=None):
    identity = lambda address, size, timestamp: timestamp
    def write(address, size, timestamp):
        if programs is not None:
            programs.append(address)
        return timestamp
    return HbfStagingIO(identity, identity, identity, write,
                        lambda direction, size, timestamp: timestamp)


def exhausted(*, pages=28, preload=False):
    controller = HbfStagingController(config(), 64 * PAGE)
    if preload:
        controller.preload("weights", 0, 1)
    controller.write("old", 1, 0, pages * PAGE, 0., callbacks())
    controller.release("old", 1)
    return controller


def services():
    return {"carry_in_serial_guard_ns": 7., "per_unit_costs": {
        "gc_copy_page_serial_upper_ns": 100.,
        "ordinary_program_page_serial_upper_ns": 10.,
        "ordinary_read_page_serial_upper_ns": 6.,
        "read_cache_initial_state_serial_upper_ns": 5.}}


def project(controller, pages, *, headroom=0, **updates):
    arguments = dict(program_pages=pages, free_program_pages=headroom,
        geometry_qualified=True, temporary_read_pages_per_decode=3,
        remaining_decode_steps=5, service_bounds=services())
    arguments.update(updates)
    return project_zero_copy_gc(controller, **arguments)


@pytest.mark.parametrize("pages,cycles", [(0, 0), (1, 1), (4, 1), (5, 2), (28, 7)])
def test_exact_empty_reservoir_has_serial_erase_and_explicit_proven_zero_terms(pages, cycles):
    controller = exhausted()
    before = controller.snapshot()
    result = project(controller, pages)
    assert result["qualified"] and result["zero_copy_proven"]
    assert result["empty_victim_reservoir"] == 7
    assert result["cycles_upper"] == cycles
    assert result["future_gc_erase_serial_ns"] == cycles * config().block_erase_ns
    assert result["future_gc_target_page_zero_erases_upper"] == cycles
    assert result["copied_pages_upper"] == 0
    for field in ("gc_copy_nand_read_serial_upper_ns", "gc_copy_nand_write_serial_upper_ns",
                  "gc_copy_host_read_serial_upper_ns", "gc_copy_host_write_serial_upper_ns",
                  "gc_copy_link_from_hbf_serial_upper_ns", "gc_copy_link_to_hbf_serial_upper_ns"):
        assert result[field] == 0.
    assert result["future_write_placement_serial_guard_ns"] == pages * 10.
    assert result["future_read_placement_serial_guard_ns"] == 3 * 5 * 6. + 5.
    assert not result["physical_state_advanced"] and not result["physical_io_executed"]
    assert controller.snapshot() == before


def test_one_fewer_empty_victim_never_uses_unproved_live_copy_estimate():
    result = project(exhausted(), 29)
    assert not result["qualified"] and not result["zero_copy_proven"]
    assert result["cycles_upper"] == 8 and result["empty_victim_reservoir"] == 7
    assert result["reason"] == "live_gc_copy_remap_bound_unavailable"
    assert result["copied_pages_upper"] is None
    assert not result["diagnostic_live_copy_bound_used_for_acceptance"]


def test_empty_active_block_is_not_a_future_empty_victim_reservoir():
    controller = exhausted(pages=25)
    assert len(controller._active) == 1
    assert project(controller, 27, headroom=3)["qualified"]
    rejected = project(controller, 28, headroom=3)
    assert rejected["empty_victim_reservoir"] == 6
    assert rejected["cycles_upper"] == 7 and not rejected["qualified"]


def test_preloaded_weights_and_spare_are_not_counted_as_empty_victims():
    controller = exhausted(pages=24, preload=True)
    result = project(controller, 24)
    assert result["qualified"] and result["empty_victim_reservoir"] == 6
    assert result["reservoir_excludes_immutable_preloads"]
    assert result["reservoir_excludes_spare"]


def test_forecasted_zero_copies_and_cycle_bound_cover_real_scalar_allocator():
    controller = exhausted()
    result = project(controller, 28)
    programs = []
    before_gc = controller.stats.get("gc_cycles", 0)
    controller.write("tail", 2, 0, 28 * PAGE, 0., callbacks(programs))
    assert controller.stats.get("gc_copied_pages", 0) == 0
    assert controller.stats["gc_cycles"] - before_gc == result["cycles_upper"]
    from heterollm_sim.memory_mapping import map_nand_address
    assert sum(map_nand_address(config(), address).page == 0 for address in programs) == 7


@pytest.mark.parametrize("updates,reason", [
    ({"geometry_qualified": False}, "gc_fixed_write_geometry_unqualified"),
    ({"temporary_read_pages_per_decode": None}, "gc_temporary_read_budget_unqualified"),
    ({"remaining_decode_steps": None}, "gc_temporary_read_budget_unqualified"),
    ({"service_bounds": None}, "gc_complete_service_bounds_unavailable"),
    ({"free_program_pages": 1}, "gc_program_headroom_does_not_match_live_controller"),
])
def test_unavailable_proof_is_a_rejection_not_a_zero_cost(updates, reason):
    result = project(exhausted(), 4, **updates)
    assert not result["qualified"] and result["reason"] == reason


def test_custom_allocator_hook_is_not_certified():
    controller = exhausted()
    controller._collect = lambda *args: pytest.fail("projection executed physical collection")
    result = project(controller, 4)
    assert result["reason"] == "gc_custom_allocator_or_controller_unqualified"


def real_physical(*, capture=False):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    from heterollm_sim.hbf_staging_runtime import configure
    from test_hbf_staging_runtime import setup
    configured, binding = setup(capacity=16 * PAGE)
    physical = PhysicalRuntimeContext(capture_details=capture)
    state = configure(physical, "flash", configured, binding)
    return physical, state["controller"], binding


@pytest.mark.parametrize("capture", [False, True])
def test_complete_real_page_chain_fits_services_without_mutating_physical_runtime(capture):
    from heterollm_sim._io_execution import _PhysicalIO
    physical, controller, binding = real_physical(capture=capture)
    state = physical.hbf_staging["flash"]
    io = _PhysicalIO(physical, "flash", state, binding, [])
    original = deepcopy(controller.snapshot())
    result = gc_service_bounds(physical, controller, binding, 0.)
    assert controller.snapshot() == original
    costs = result["per_unit_costs"]
    now = io.write_flash(controller._address(0, 0), PAGE, 0.)
    # Ordinary program bound excludes separately budgeted page-zero erase.
    assert now <= costs["nand_program_page_serial_upper_ns"] + controller.config.block_erase_ns
    now = io.read_flash(controller._address(0, 0), PAGE, now)
    start = now
    now = io.transfer("from_hbf", PAGE, now)
    now = io.write_host(controller.host_capacity_bytes - PAGE, PAGE, now)
    now = io.read_host(controller.host_capacity_bytes - PAGE, PAGE, now)
    now = io.transfer("to_hbf", PAGE, now)
    now = io.write_flash(controller._address(1, 0), PAGE, now)
    assert now - start <= (costs["gc_copy_page_serial_upper_ns"]
        - costs["nand_read_page_serial_upper_ns"] + controller.config.block_erase_ns)
    before = deepcopy(physical.timeline.lane_available)
    flags = {owner: (active.core.__dict__.get("_nand_numeric_enabled"),
                    active.core.__dict__.get("_nand_state_exposed"))
             for owner, active in physical.runtimes.items()}
    delayed = gc_service_bounds(physical, controller, binding, 123.)
    assert delayed["carry_in_serial_guard_ns"] >= now - 123.
    assert physical.timeline.lane_available == before
    assert flags == {owner: (active.core.__dict__.get("_nand_numeric_enabled"),
                            active.core.__dict__.get("_nand_state_exposed"))
                     for owner, active in physical.runtimes.items()}


def test_authoritative_numeric_cache_clocks_are_read_without_publishing():
    physical, controller, binding = real_physical()
    core = physical.runtimes["flash"].core
    addresses = [controller._address(0, page) for page in (0, 1)]
    assert core.submit_read_pages(addresses, 1_000_000.) is not None
    packed = core.__dict__["_nand_numeric_state"]
    assert packed is not None
    before = tuple(array.copy() for array in packed.arrays)
    enabled = core.__dict__["_nand_numeric_enabled"]
    exposed = core.__dict__["_nand_state_exposed"]
    timeline_before = deepcopy(physical.timeline.__dict__)
    cost = gc_service_bounds(physical, controller, binding, 1_000_000.)
    assert cost["carry_in_serial_guard_ns"] > 0
    assert core.__dict__["_nand_numeric_state"] is packed
    assert core.__dict__["_nand_numeric_enabled"] is enabled
    assert core.__dict__["_nand_state_exposed"] is exposed
    assert physical.timeline.__dict__ == timeline_before
    import numpy as np
    for actual, expected in zip(packed.arrays, before, strict=True):
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("capture", [False, True])
def test_real_empty_gc_projection_covers_later_physical_target_erases(capture):
    from heterollm_sim._io_execution import _PhysicalIO
    physical, controller, binding = real_physical(capture=capture)
    io = _PhysicalIO(physical, "flash", physical.hbf_staging["flash"], binding, [])
    pages = (controller._block_count - 1) * controller.config.pages_per_block
    controller.write("released", 1, 0, pages * PAGE, 0., io)
    controller.release("released", 1)
    before_state = controller.snapshot()
    core = physical.runtimes["flash"].core
    native = core.__dict__.get("_nand_numeric_state")
    arrays = None if native is None else tuple(array.copy() for array in native.arrays)
    services = gc_service_bounds(physical, controller, binding, 0.)
    projected = project_zero_copy_gc(controller, program_pages=pages, free_program_pages=0,
        geometry_qualified=True, temporary_read_pages_per_decode=0,
        remaining_decode_steps=1, service_bounds=services)
    assert projected["qualified"] and controller.snapshot() == before_state
    assert core.__dict__.get("_nand_numeric_state") is native
    if native is not None:
        import numpy as np
        assert all(np.array_equal(actual, expected) for actual, expected in zip(native.arrays, arrays, strict=True))
    start = services["carry_in_serial_guard_ns"]
    end = controller.write("future", 2, 0, pages * PAGE, start, io)
    assert controller.stats.get("gc_copied_pages", 0) == 0
    assert controller.stats["gc_cycles"] <= projected["cycles_upper"]
    assert end - start <= (projected["future_write_placement_serial_guard_ns"]
        + projected["future_gc_erase_serial_ns"])


def test_absolute_cutoff_and_prior_slot_lifetimes_are_not_mixed_with_runtime_origin():
    physical, controller, binding = real_physical()
    origin = 10**12
    controller._scratch_ready = origin + 20.
    controller._slot_ready[0] = origin + 70.
    controller._block_ready[0] = origin + 50.
    physical.timeline.lane_available["pending"] = [origin + 3., origin + 100.]
    assert _calendar_carry_in(physical, controller, origin + 5.) == 95.


def test_resident_reads_must_resolve_to_the_actual_immutable_preload_and_extent():
    physical, controller, _ = real_physical()
    allocator = physical.allocators["flash"]
    allocator.allocate("embedding", 2 * PAGE, allocation_arena="resident")
    allocator.allocate("head", PAGE, alias_of="embedding", alias_offset_bytes=PAGE)
    allocator.allocate("other-resident-state", PAGE, allocation_arena="resident")
    controller.preload("embedding", 0, 2 * PAGE)
    before, allocations = controller.snapshot(), dict(allocator._allocations)
    accepted = qualify_immutable_reads(physical, controller, "flash", (("head", 0, 0, PAGE),))
    assert accepted["qualified"] and accepted["canonical_read_range_count"] == 1
    assert not qualify_immutable_reads(physical, controller, "flash",
        (("other-resident-state", 0, 0, PAGE),))["qualified"]
    assert not qualify_immutable_reads(physical, controller, "flash",
        (("head", 0, 0, PAGE + 1),))["qualified"]
    assert controller.snapshot() == before and allocator._allocations == allocations


def test_real_link_alias_combines_both_demands_and_the_serial_cost_still_covers_it():
    from heterollm_sim._io_execution import _PhysicalIO
    physical, controller, binding = real_physical()
    binding = {**binding, "routes": {**binding["routes"], "to_hbf": tuple(
        {**hop, "queue_depth": 1} for hop in binding["routes"]["to_hbf"])}}
    physical.timeline.resource_aliases = {"to_hbf.inflight": "to_hbf"}
    io = _PhysicalIO(physical, "flash", physical.hbf_staging["flash"], binding, [])
    costs = gc_service_bounds(physical, controller, binding, 0.)["per_unit_costs"]
    actual = io.transfer("to_hbf", PAGE, 0.)
    assert actual <= costs["link_to_hbf_page_serial_upper_ns"]


@pytest.mark.parametrize("change", ["missing_route", "unknown_host", "nonfinite_clock", "custom_cache"])
def test_real_unknown_contracts_cannot_receive_zero_service(change):
    physical, controller, binding = real_physical()
    if change == "missing_route":
        binding = {**binding, "routes": {**binding["routes"], "to_hbf": ()}}
    elif change == "unknown_host":
        binding = {**binding, "host_owner": "unknown-host"}
    elif change == "nonfinite_clock":
        physical.timeline.ready_ns["test"] = float("nan")
    else:
        physical.runtimes["flash"].core._invalidate_cache = lambda *args, **kwargs: None
    with pytest.raises((ValueError, TypeError)):
        gc_service_bounds(physical, controller, binding, 0.)


@pytest.mark.parametrize("model", ["qwen2_5-0_5b", "qwen2_5-1_5b", "qwen2_5-3b", "qwen2_5-7b"])
def test_formal_read_geometry_and_routes_are_ir_only_and_context_invariant(model):
    from test_fast_erase_bound import formal_blueprint_runtime
    from heterollm_sim.fast_state import _fixed_decode_write_geometry
    runner, rows = formal_blueprint_runtime(model)
    result = _fixed_decode_write_geometry(runner, rows, 75775)
    assert result["qualified"] and result["temporary_read_geometry_qualified"], result
    assert result["temporary_read_page_upper_by_owner"]["hbf0.memory"] > 0
    bindings = result["gc_staging_bindings_by_owner"]["hbf0.memory"]
    assert all(binding["routes"][route] for binding in bindings
               for route in ("to_hbf", "from_hbf", "write_hbf", "read_hbf"))
    assert not hasattr(runner, "_execution_resource_kernel")


def test_unproved_read_geometry_rejects_only_gc_domain_and_keeps_old_write_certificate(monkeypatch):
    from heterollm_sim import planner
    from heterollm_sim.fast_state import _fixed_decode_write_geometry
    from test_fast_erase_bound import formal_blueprint_runtime
    runner, rows = formal_blueprint_runtime()
    original = planner._lower_serving_cohort
    def changed_read(scenario, cohort):
        result = original(scenario, cohort)
        if cohort.items[0].context_tokens <= 70000:
            return result
        tasks, changed = [], False
        for task in result.schedule.tasks:
            accesses = []
            for access in task.metadata.get("memory_accesses", ()):
                if (not changed and access.get("operation") == "read"
                        and access.get("physical_owner") == "hbf0.memory"
                        and access.get("allocation_generation")):
                    access = {**access, "byte_count": access["byte_count"] + 1}
                    changed = True
                accesses.append(access)
            tasks.append(replace(task, metadata={**task.metadata, "memory_accesses": tuple(accesses)})
                         if accesses else task)
        assert changed
        return replace(result, schedule=replace(result.schedule, tasks=tuple(tasks)))
    monkeypatch.setattr(planner, "_lower_serving_cohort", changed_read)
    result = _fixed_decode_write_geometry(runner, rows, 75775)
    assert result["qualified"]
    assert not result["temporary_read_geometry_qualified"]
    assert result["temporary_read_geometry_unavailable_reason"] == "non_kv_read_or_alias_geometry_changes_with_context"


def lru_misses(trace, cache, capacity):
    cache = list(cache)
    misses = 0
    for key in trace:
        if key in cache:
            cache.remove(key)
        else:
            misses += 1
            if len(cache) == capacity:
                cache.pop(0)
        cache.append(key)
    return misses


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_lru_temporary_insertion_bound_covers_eviction_cascades(capacity):
    # One insertion can cause k misses in later persistent reads, not just
    # one. Exhaust short traces from arbitrary initial persistent contents.
    initial = tuple(range(capacity))
    alphabet = tuple(range(capacity + 1))
    for trace in product(alphabet, repeat=5):
        baseline = lru_misses(trace, initial, capacity)
        for index in range(6):
            modified = trace[:index] + ("temporary",) + trace[index:]
            assert lru_misses(modified, initial, capacity) - baseline <= capacity + 1
